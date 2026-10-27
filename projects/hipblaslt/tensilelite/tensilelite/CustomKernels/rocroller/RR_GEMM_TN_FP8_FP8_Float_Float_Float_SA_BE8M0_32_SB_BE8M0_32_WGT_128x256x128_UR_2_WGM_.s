// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 13471) 
s_mul_hi_i32 s1, s16, s10 // (op 13610) 
v_lshrrev_b32 v0, 7, v1 // (op 13532) 
v_and_b32 v3, 15, v1 // (op 13532) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 13532) 


v_and_b32 v0, 48, v1 // (op 13532) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 13532) 


v_lshrrev_b32 v0, 6, v1 // (op 13589) 
v_and_b32 v4, 1, v0 // (op 13589) 

v_and_b32 v0, 15, v1 // (op 13589) 
v_lshl_add_u32 v5, v4, 7, v0 // (op 13589) 


v_and_b32 v0, 48, v1 // (op 13589) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 13589) 


v_lshrrev_b32 v0, 7, v1 // (op 13656) 
v_and_b32 v5, 15, v1 // (op 13656) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 13656) 


v_and_b32 v0, 48, v1 // (op 13656) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 13656) 


v_lshrrev_b32 v0, 6, v1 // (op 13700) 
v_and_b32 v6, 1, v0 // (op 13700) 

v_and_b32 v0, 15, v1 // (op 13700) 
v_lshl_add_u32 v7, v6, 7, v0 // (op 13700) 


v_and_b32 v0, 48, v1 // (op 13700) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 13700) 


v_lshrrev_b32 v0, 7, v1 // (op 13471) 
s_add_i32 s18, s0, s16 // (op 13471) 

v_and_b32 v7, 63, v1 // (op 13471) 
s_ashr_i32 s0, s18, 31 // (op 13471) 
v_lshrrev_b32 v8, 4, v7 // (op 13471) 
s_cmp_eq_i32 s10, 0 // (op 13471) 
s_mov_b32 s19, scc // (op 13471) 
v_lshlrev_b32 v9, 2, v8 // (op 13471) 

s_cmp_eq_i32 s19, 0 // (op 13471) 
s_cselect_b32 s49, 0, -1 // (op 13471) 

v_lshl_add_u32 v8, v0, 6, v9 // (op 13471) 


s_and_b32 s19, s13, 31 // (op 13471) 
v_lshrrev_b32 v0, 2, v7 // (op 13471) 

v_and_b32 v7, 3, v0 // (op 13471) 

v_lshrrev_b32 v0, 7, v1 // (op 13610) 
v_and_b32 v9, 63, v1 // (op 13610) 
s_lshl_b32 s50, 1, s19 // (op 13471) 
v_and_b32 v10, 3, v1 // (op 13471) 
s_add_i32 s51, s50, s49 // (op 13471) 


s_and_b32 s49, s0, s51 // (op 13471) 


s_add_i32 s0, s1, s16 // (op 13610) 

s_ashr_i32 s1, s0, 31 // (op 13610) 
v_lshl_add_u32 v11, v7, 2, v10 // (op 13471) 


s_add_i32 s50, s18, s49 // (op 13471) 


v_lshrrev_b32 v7, 6, v1 // (op 13471) 
s_ashr_i32 s18, s50, s19 // (op 13471) 


v_and_b32 v10, 1, v7 // (op 13471) 

s_lshl_b32 s19, s13, 24 // (op 13471) 
v_lshl_add_u32 v7, v10, 7, v11 // (op 13471) 


s_ashr_i32 s49, s19, 31 // (op 13471) 

s_xor_b32 s19, s18, s49 // (op 13471) 

s_sub_i32 s18, s19, s49 // (op 13471) 


s_mul_hi_i32 s19, s18, s14 // (op 13471) 
s_add_i32 s49, s19, s18 // (op 13471) 

s_ashr_i32 s19, s49, 31 // (op 13471) 
s_cmp_eq_i32 s14, 0 // (op 13471) 
s_mov_b32 s50, scc // (op 13471) 
s_cmp_eq_i32 s50, 0 // (op 13471) 
s_cselect_b32 s51, 0, -1 // (op 13471) 

s_and_b32 s50, s15, 31 // (op 13471) 
v_lshrrev_b32 v10, 4, v9 // (op 13610) 
s_cmp_eq_i32 s10, 0 // (op 13610) 
s_mov_b32 s52, scc // (op 13610) 
v_lshlrev_b32 v11, 2, v10 // (op 13610) 

s_cmp_eq_i32 s52, 0 // (op 13610) 
s_cselect_b32 s53, 0, -1 // (op 13610) 

s_lshl_b32 s52, 1, s50 // (op 13471) 
v_lshl_add_u32 v10, v0, 6, v11 // (op 13610) 


v_lshrrev_b32 v0, 2, v9 // (op 13610) 

s_add_i32 s54, s52, s51 // (op 13471) 


s_and_b32 s51, s19, s54 // (op 13471) 


s_add_i32 s19, s49, s51 // (op 13471) 


s_ashr_i32 s49, s19, s50 // (op 13471) 


s_lshl_b32 s19, s15, 24 // (op 13471) 
s_ashr_i32 s50, s19, 31 // (op 13471) 

s_xor_b32 s19, s49, s50 // (op 13471) 

s_sub_i32 s49, s19, s50 // (op 13471) 


s_mul_i32 s19, s18, s11 // (op 13471) 
s_sub_u32 s50, s16, s19 // (op 13471) 

s_mul_i32 s19, s49, s11 // (op 13471) 
s_add_u32 s19, s19, s50 // (op 13471) 

s_mul_hi_i32 s50, s16, s9 // (op 13471) 
s_add_i32 s51, s50, s16 // (op 13471) 

s_ashr_i32 s50, s51, 31 // (op 13471) 
s_cmp_eq_i32 s9, 0 // (op 13471) 
s_mov_b32 s52, scc // (op 13471) 
s_cmp_eq_i32 s52, 0 // (op 13471) 
s_cselect_b32 s54, 0, -1 // (op 13471) 

s_and_b32 s52, s13, 31 // (op 13610) 
v_and_b32 v9, 3, v0 // (op 13610) 

s_lshl_b32 s55, 1, s52 // (op 13610) 
v_and_b32 v0, 3, v1 // (op 13610) 
s_add_i32 s56, s55, s53 // (op 13610) 


s_and_b32 s53, s1, s56 // (op 13610) 


v_lshl_add_u32 v11, v9, 2, v0 // (op 13610) 


s_add_i32 s1, s0, s53 // (op 13610) 


v_lshrrev_b32 v0, 6, v1 // (op 13610) 
s_ashr_i32 s0, s1, s52 // (op 13610) 


v_and_b32 v9, 1, v0 // (op 13610) 

s_lshl_b32 s1, s13, 24 // (op 13610) 
v_lshl_add_u32 v0, v9, 7, v11 // (op 13610) 


s_ashr_i32 s52, s1, 31 // (op 13610) 

s_xor_b32 s1, s0, s52 // (op 13610) 

s_sub_i32 s0, s1, s52 // (op 13610) 


s_mul_hi_i32 s1, s0, s14 // (op 13610) 
s_add_i32 s52, s1, s0 // (op 13610) 

s_ashr_i32 s1, s52, 31 // (op 13610) 
s_cmp_eq_i32 s14, 0 // (op 13610) 
s_mov_b32 s53, scc // (op 13610) 
s_cmp_eq_i32 s53, 0 // (op 13610) 
s_cselect_b32 s55, 0, -1 // (op 13610) 

s_and_b32 s53, s15, 31 // (op 13610) 
s_lshl_b32 s56, 1, s53 // (op 13610) 
s_add_i32 s57, s56, s55 // (op 13610) 


s_and_b32 s55, s1, s57 // (op 13610) 


s_add_i32 s1, s52, s55 // (op 13610) 


s_ashr_i32 s52, s1, s53 // (op 13610) 


s_lshl_b32 s1, s15, 24 // (op 13610) 
s_ashr_i32 s53, s1, 31 // (op 13610) 

s_xor_b32 s1, s52, s53 // (op 13610) 

s_sub_i32 s52, s1, s53 // (op 13610) 


s_mul_i32 s1, s0, s11 // (op 13610) 
s_sub_u32 s53, s16, s1 // (op 13610) 

s_mul_i32 s1, s52, s11 // (op 13610) 
s_add_u32 s1, s1, s53 // (op 13610) 

s_mul_hi_i32 s53, s16, s9 // (op 13610) 
s_add_i32 s55, s53, s16 // (op 13610) 

s_ashr_i32 s53, s55, 31 // (op 13610) 
s_cmp_eq_i32 s9, 0 // (op 13610) 
s_mov_b32 s56, scc // (op 13610) 
s_cmp_eq_i32 s56, 0 // (op 13610) 
s_cselect_b32 s57, 0, -1 // (op 13610) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 13471) 
s_lshl_b32 s58, 1, s56 // (op 13471) 
s_add_i32 s59, s58, s54 // (op 13471) 


s_and_b32 s54, s50, s59 // (op 13471) 


s_add_i32 s50, s51, s54 // (op 13471) 


s_ashr_i32 s51, s50, s56 // (op 13471) 


s_lshl_b32 s50, s45, 24 // (op 13471) 
s_ashr_i32 s54, s50, 31 // (op 13471) 

s_xor_b32 s50, s51, s54 // (op 13471) 

s_sub_i32 s51, s50, s54 // (op 13471) 


s_and_b32 s50, s45, 31 // (op 13610) 
s_lshl_b32 s54, 1, s50 // (op 13610) 
s_add_i32 s56, s54, s57 // (op 13610) 


s_and_b32 s54, s53, s56 // (op 13610) 


s_add_i32 s53, s55, s54 // (op 13610) 


s_ashr_i32 s54, s53, s50 // (op 13610) 


s_lshl_b32 s50, s45, 24 // (op 13610) 
s_ashr_i32 s53, s50, 31 // (op 13610) 

s_xor_b32 s50, s54, s53 // (op 13610) 

s_sub_i32 s54, s50, s53 // (op 13610) 


s_add_u32 s56, s6, 256 // (op 13471) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 13471) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 13471) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 13471) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 13471) 

s_mul_i32 s50, s11, s56 // (op 13471) 
s_mul_i32 s53, s8, s50 // (op 13471) 

s_mul_i32 s50, s51, s53 // (op 13471) 

s_sub_u32 s53, s16, s50 // (op 13471) 

s_mov_b32 s50, s53 // (op 13471) 
s_mul_hi_i32 s55, s50, s12 // (op 13471) 
s_add_i32 s58, s55, s50 // (op 13471) 


s_ashr_i32 s50, s58, 31 // (op 13471) 
s_cmp_eq_i32 s12, 0 // (op 13471) 
s_mov_b32 s55, scc // (op 13471) 
s_cmp_eq_i32 s55, 0 // (op 13471) 
s_cselect_b32 s59, 0, -1 // (op 13471) 

s_and_b32 s55, s44, 31 // (op 13471) 
s_lshl_b32 s60, 1, s55 // (op 13471) 
s_add_i32 s61, s60, s59 // (op 13471) 


s_and_b32 s59, s50, s61 // (op 13471) 


s_add_i32 s50, s58, s59 // (op 13471) 


s_ashr_i32 s58, s50, s55 // (op 13471) 


s_lshl_b32 s50, s44, 24 // (op 13471) 
s_ashr_i32 s55, s50, 31 // (op 13471) 

s_xor_b32 s50, s58, s55 // (op 13471) 

s_sub_i32 s58, s50, s55 // (op 13471) 


s_mul_i32 s50, s46, s11 // (op 13471) 
s_add_u32 s60, s20, 128 // (op 13471) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 13471) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 13471) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 13471) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 7 // (op 13471) 

s_sub_i32 s55, s60, s50 // (op 13471) 

s_mul_i32 s59, s58, s55 // (op 13471) 

s_sub_u32 s55, s53, s59 // (op 13471) 


s_add_u32 s53, s55, s50 // (op 13471) 


s_cmp_eq_i32 s51, 0 // (op 13471) 
s_mov_b32 s50, scc // (op 13471) 

s_cmp_eq_i32 s50, 0 // (op 13471) 
s_cselect_b32 s51, s53, s19 // (op 13471) 


s_mul_i32 s19, s49, s56 // (op 13471) 


v_lshl_add_u32 v9, s51, 7, v8 // (op 13471) 


s_sub_i32 s49, s18, s19 // (op 13471) 


s_cmp_eq_i32 s50, 0 // (op 13471) 
s_cselect_b32 s51, s58, s49 // (op 13471) 



v_lshl_add_u32 v8, s51, 8, v7 // (op 13471) 


v_mul_lo_u32 v7, v8, s2 // (op 13471) 

v_add_lshl_u32 v8, v9, v7, 2 // (op 13471) 


s_add_u32 s18, s6, 256 // (op 13610) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 13610) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 13610) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 13610) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 8 // (op 13610) 

s_mul_i32 s53, s11, s18 // (op 13610) 
s_mul_i32 s49, s8, s53 // (op 13610) 

s_mul_i32 s53, s54, s49 // (op 13610) 

s_sub_u32 s49, s16, s53 // (op 13610) 

s_mov_b32 s53, s49 // (op 13610) 
s_mul_hi_i32 s50, s53, s12 // (op 13610) 
s_add_i32 s51, s50, s53 // (op 13610) 


s_ashr_i32 s50, s51, 31 // (op 13610) 
s_cmp_eq_i32 s12, 0 // (op 13610) 
s_mov_b32 s53, scc // (op 13610) 
s_cmp_eq_i32 s53, 0 // (op 13610) 
s_cselect_b32 s55, 0, -1 // (op 13610) 

s_and_b32 s53, s44, 31 // (op 13610) 
s_lshl_b32 s56, 1, s53 // (op 13610) 
s_add_i32 s57, s56, s55 // (op 13610) 


s_and_b32 s55, s50, s57 // (op 13610) 


s_add_i32 s50, s51, s55 // (op 13610) 


s_ashr_i32 s51, s50, s53 // (op 13610) 


s_lshl_b32 s50, s44, 24 // (op 13610) 
s_ashr_i32 s53, s50, 31 // (op 13610) 

s_xor_b32 s50, s51, s53 // (op 13610) 

s_sub_i32 s51, s50, s53 // (op 13610) 


s_mul_i32 s50, s46, s11 // (op 13610) 
s_add_u32 s56, s20, 128 // (op 13610) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 13610) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 13610) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 13610) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 13610) 

s_sub_i32 s53, s56, s50 // (op 13610) 

s_mul_i32 s55, s51, s53 // (op 13610) 

s_sub_u32 s53, s49, s55 // (op 13610) 


s_add_u32 s49, s53, s50 // (op 13610) 


s_cmp_eq_i32 s54, 0 // (op 13610) 
s_mov_b32 s50, scc // (op 13610) 

s_cmp_eq_i32 s50, 0 // (op 13610) 
s_cselect_b32 s53, s49, s1 // (op 13610) 


s_mul_i32 s1, s52, s18 // (op 13610) 


v_lshl_add_u32 v7, s53, 7, v10 // (op 13610) 


s_sub_i32 s49, s0, s1 // (op 13610) 


s_cmp_eq_i32 s50, 0 // (op 13610) 
s_cselect_b32 s0, s51, s49 // (op 13610) 



v_lshl_add_u32 v9, s0, 8, v0 // (op 13610) 


v_mul_lo_u32 v0, v9, s4 // (op 13610) 

v_add_lshl_u32 v9, v7, v0, 2 // (op 13610) 


s_mul_i32 s1, s2, s7 // (op 13473) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 13473) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 13473) least significant
s_mul_hi_u32 s19, s2, s6 // (op 13473) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13473) most significant: sum
s_add_u32 s1, s1, s19 // (op 13473) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13473) 

s_mul_i32 s1, s4, s7 // (op 13612) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 13612) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 13612) least significant
s_mul_hi_u32 s50, s4, s6 // (op 13612) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 13612) most significant: sum
s_add_u32 s1, s1, s50 // (op 13612) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 2 // (op 13612) 

s_mov_b32 s52, s22 // (op 13473) reg expression
s_mov_b32 s53, s23 // (op 13473) reg expression
s_mov_b32 s54, s18 // (op 13473) reg expression
s_mov_b32 s55, 131072 // (op 13473) call()

s_mov_b32 s56, s24 // (op 13612) reg expression

s_mov_b32 s57, s25 // (op 13612) reg expression
s_mov_b32 s58, s50 // (op 13612) reg expression
s_mov_b32 s59, 131072 // (op 13612) call()


v_mov_b32 v0, s47 // (op 2) Move value
v_mov_b32 v7, s48 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
v_accvgpr_write a4, 0.00000 // (op 277) call()
v_accvgpr_write a5, 0.00000 // (op 277) call()
v_accvgpr_write a6, 0.00000 // (op 277) call()
v_accvgpr_write a7, 0.00000 // (op 277) call()
v_accvgpr_write a8, 0.00000 // (op 413) call()
v_accvgpr_write a9, 0.00000 // (op 413) call()
v_accvgpr_write a10, 0.00000 // (op 413) call()
v_accvgpr_write a11, 0.00000 // (op 413) call()
v_accvgpr_write a12, 0.00000 // (op 549) call()
v_accvgpr_write a13, 0.00000 // (op 549) call()
v_accvgpr_write a14, 0.00000 // (op 549) call()
v_accvgpr_write a15, 0.00000 // (op 549) call()
v_accvgpr_write a16, 0.00000 // (op 685) call()
v_accvgpr_write a17, 0.00000 // (op 685) call()
v_accvgpr_write a18, 0.00000 // (op 685) call()
v_accvgpr_write a19, 0.00000 // (op 685) call()
v_accvgpr_write a20, 0.00000 // (op 821) call()
v_accvgpr_write a21, 0.00000 // (op 821) call()
v_accvgpr_write a22, 0.00000 // (op 821) call()
v_accvgpr_write a23, 0.00000 // (op 821) call()
v_accvgpr_write a24, 0.00000 // (op 957) call()
v_accvgpr_write a25, 0.00000 // (op 957) call()
v_accvgpr_write a26, 0.00000 // (op 957) call()
v_accvgpr_write a27, 0.00000 // (op 957) call()
v_accvgpr_write a28, 0.00000 // (op 1093) call()
v_accvgpr_write a29, 0.00000 // (op 1093) call()
v_accvgpr_write a30, 0.00000 // (op 1093) call()
v_accvgpr_write a31, 0.00000 // (op 1093) call()
v_accvgpr_write a32, 0.00000 // (op 1405) call()
v_accvgpr_write a33, 0.00000 // (op 1405) call()
v_accvgpr_write a34, 0.00000 // (op 1405) call()
v_accvgpr_write a35, 0.00000 // (op 1405) call()
v_accvgpr_write a36, 0.00000 // (op 1926) call()
v_accvgpr_write a37, 0.00000 // (op 1926) call()
v_accvgpr_write a38, 0.00000 // (op 1926) call()
v_accvgpr_write a39, 0.00000 // (op 1926) call()
v_accvgpr_write a40, 0.00000 // (op 1927) call()
v_accvgpr_write a41, 0.00000 // (op 1927) call()
v_accvgpr_write a42, 0.00000 // (op 1927) call()
v_accvgpr_write a43, 0.00000 // (op 1927) call()
v_accvgpr_write a44, 0.00000 // (op 1928) call()
v_accvgpr_write a45, 0.00000 // (op 1928) call()
v_accvgpr_write a46, 0.00000 // (op 1928) call()
v_accvgpr_write a47, 0.00000 // (op 1928) call()
v_accvgpr_write a48, 0.00000 // (op 1929) call()
v_accvgpr_write a49, 0.00000 // (op 1929) call()
v_accvgpr_write a50, 0.00000 // (op 1929) call()
v_accvgpr_write a51, 0.00000 // (op 1929) call()
v_accvgpr_write a52, 0.00000 // (op 1930) call()
v_accvgpr_write a53, 0.00000 // (op 1930) call()
v_accvgpr_write a54, 0.00000 // (op 1930) call()
v_accvgpr_write a55, 0.00000 // (op 1930) call()
v_accvgpr_write a56, 0.00000 // (op 1931) call()
v_accvgpr_write a57, 0.00000 // (op 1931) call()
v_accvgpr_write a58, 0.00000 // (op 1931) call()
v_accvgpr_write a59, 0.00000 // (op 1931) call()
v_accvgpr_write a60, 0.00000 // (op 1932) call()
v_accvgpr_write a61, 0.00000 // (op 1932) call()
v_accvgpr_write a62, 0.00000 // (op 1932) call()
v_accvgpr_write a63, 0.00000 // (op 1932) call()
v_accvgpr_write a64, 0.00000 // (op 2623) call()
v_accvgpr_write a65, 0.00000 // (op 2623) call()
v_accvgpr_write a66, 0.00000 // (op 2623) call()
v_accvgpr_write a67, 0.00000 // (op 2623) call()
v_accvgpr_write a68, 0.00000 // (op 3144) call()
v_accvgpr_write a69, 0.00000 // (op 3144) call()
v_accvgpr_write a70, 0.00000 // (op 3144) call()
v_accvgpr_write a71, 0.00000 // (op 3144) call()
v_accvgpr_write a72, 0.00000 // (op 3145) call()
v_accvgpr_write a73, 0.00000 // (op 3145) call()
v_accvgpr_write a74, 0.00000 // (op 3145) call()
v_accvgpr_write a75, 0.00000 // (op 3145) call()
v_accvgpr_write a76, 0.00000 // (op 3146) call()
v_accvgpr_write a77, 0.00000 // (op 3146) call()
v_accvgpr_write a78, 0.00000 // (op 3146) call()
v_accvgpr_write a79, 0.00000 // (op 3146) call()
v_accvgpr_write a80, 0.00000 // (op 3147) call()
v_accvgpr_write a81, 0.00000 // (op 3147) call()
v_accvgpr_write a82, 0.00000 // (op 3147) call()
v_accvgpr_write a83, 0.00000 // (op 3147) call()
v_accvgpr_write a84, 0.00000 // (op 3148) call()
v_accvgpr_write a85, 0.00000 // (op 3148) call()
v_accvgpr_write a86, 0.00000 // (op 3148) call()
v_accvgpr_write a87, 0.00000 // (op 3148) call()
v_accvgpr_write a88, 0.00000 // (op 3149) call()
v_accvgpr_write a89, 0.00000 // (op 3149) call()
v_accvgpr_write a90, 0.00000 // (op 3149) call()
v_accvgpr_write a91, 0.00000 // (op 3149) call()
v_accvgpr_write a92, 0.00000 // (op 3150) call()
v_accvgpr_write a93, 0.00000 // (op 3150) call()
v_accvgpr_write a94, 0.00000 // (op 3150) call()
v_accvgpr_write a95, 0.00000 // (op 3150) call()
v_accvgpr_write a96, 0.00000 // (op 3841) call()
v_accvgpr_write a97, 0.00000 // (op 3841) call()
v_accvgpr_write a98, 0.00000 // (op 3841) call()
v_accvgpr_write a99, 0.00000 // (op 3841) call()
v_accvgpr_write a100, 0.00000 // (op 4362) call()
v_accvgpr_write a101, 0.00000 // (op 4362) call()
v_accvgpr_write a102, 0.00000 // (op 4362) call()
v_accvgpr_write a103, 0.00000 // (op 4362) call()
v_accvgpr_write a104, 0.00000 // (op 4363) call()
v_accvgpr_write a105, 0.00000 // (op 4363) call()
v_accvgpr_write a106, 0.00000 // (op 4363) call()
v_accvgpr_write a107, 0.00000 // (op 4363) call()
v_accvgpr_write a108, 0.00000 // (op 4364) call()
v_accvgpr_write a109, 0.00000 // (op 4364) call()
v_accvgpr_write a110, 0.00000 // (op 4364) call()
v_accvgpr_write a111, 0.00000 // (op 4364) call()
v_accvgpr_write a112, 0.00000 // (op 4365) call()
v_accvgpr_write a113, 0.00000 // (op 4365) call()
v_accvgpr_write a114, 0.00000 // (op 4365) call()
v_accvgpr_write a115, 0.00000 // (op 4365) call()
v_accvgpr_write a116, 0.00000 // (op 4366) call()
v_accvgpr_write a117, 0.00000 // (op 4366) call()
v_accvgpr_write a118, 0.00000 // (op 4366) call()
v_accvgpr_write a119, 0.00000 // (op 4366) call()
v_accvgpr_write a120, 0.00000 // (op 4367) call()
v_accvgpr_write a121, 0.00000 // (op 4367) call()
v_accvgpr_write a122, 0.00000 // (op 4367) call()
v_accvgpr_write a123, 0.00000 // (op 4367) call()
v_accvgpr_write a124, 0.00000 // (op 4368) call()
v_accvgpr_write a125, 0.00000 // (op 4368) call()
v_accvgpr_write a126, 0.00000 // (op 4368) call()
v_accvgpr_write a127, 0.00000 // (op 4368) call()
s_mul_hi_i32 s0, s16, s10 // (op 13361) 
v_lshrrev_b32 v10, 3, v1 // (op 13361) 
s_add_i32 s1, s0, s16 // (op 13361) 

v_and_b32 v11, -8, v1 // (op 13361) 
s_ashr_i32 s0, s1, 31 // (op 13361) 
v_sub_u32 v12, v1, v11 // (op 13361) 

s_cmp_eq_i32 s10, 0 // (op 13361) 
s_mov_b32 s18, scc // (op 13361) 
v_lshlrev_b32 v11, 4, v12 // (op 13361) 

s_cmp_eq_i32 s18, 0 // (op 13361) 
s_cselect_b32 s19, 0, -1 // (op 13361) 

v_mov_b32 v13, 0 // (op 13361) convert
v_mov_b32 v12, v11 // (op 13361) convert

s_and_b32 s18, s13, 31 // (op 13361) 
s_lshl_b32 s22, 1, s18 // (op 13361) 
s_add_i32 s23, s22, s19 // (op 13361) 


s_and_b32 s19, s0, s23 // (op 13361) 


s_add_i32 s0, s1, s19 // (op 13361) 


s_ashr_i32 s1, s0, s18 // (op 13361) 


s_lshl_b32 s0, s13, 24 // (op 13361) 
s_ashr_i32 s18, s0, 31 // (op 13361) 

s_xor_b32 s0, s1, s18 // (op 13361) 

s_sub_i32 s1, s0, s18 // (op 13361) 


s_mul_hi_i32 s0, s1, s14 // (op 13361) 
s_add_i32 s18, s0, s1 // (op 13361) 

s_ashr_i32 s0, s18, 31 // (op 13361) 
s_cmp_eq_i32 s14, 0 // (op 13361) 
s_mov_b32 s19, scc // (op 13361) 
s_cmp_eq_i32 s19, 0 // (op 13361) 
s_cselect_b32 s22, 0, -1 // (op 13361) 

s_and_b32 s19, s15, 31 // (op 13361) 
s_lshl_b32 s23, 1, s19 // (op 13361) 
s_add_i32 s24, s23, s22 // (op 13361) 


s_and_b32 s25, s0, s24 // (op 13361) 


s_add_i32 s0, s18, s25 // (op 13361) 


s_ashr_i32 s18, s0, s19 // (op 13361) 


s_lshl_b32 s0, s15, 24 // (op 13361) 
s_ashr_i32 s19, s0, 31 // (op 13361) 

s_xor_b32 s0, s18, s19 // (op 13361) 

s_sub_i32 s18, s0, s19 // (op 13361) 


s_mul_i32 s0, s1, s11 // (op 13361) 

s_sub_u32 s1, s16, s0 // (op 13361) 

s_mul_i32 s0, s18, s11 // (op 13361) 
s_add_u32 s0, s0, s1 // (op 13361) 


s_mul_hi_i32 s1, s16, s9 // (op 13361) 
s_add_i32 s18, s1, s16 // (op 13361) 

s_ashr_i32 s1, s18, 31 // (op 13361) 
s_cmp_eq_i32 s9, 0 // (op 13361) 
s_mov_b32 s19, scc // (op 13361) 
s_cmp_eq_i32 s19, 0 // (op 13361) 
s_cselect_b32 s22, 0, -1 // (op 13361) 

s_and_b32 s19, s45, 31 // (op 13361) 
s_lshl_b32 s23, 1, s19 // (op 13361) 
s_add_i32 s24, s23, s22 // (op 13361) 


s_and_b32 s25, s1, s24 // (op 13361) 


s_add_i32 s1, s18, s25 // (op 13361) 


s_ashr_i32 s18, s1, s19 // (op 13361) 


s_lshl_b32 s1, s45, 24 // (op 13361) 
s_ashr_i32 s19, s1, 31 // (op 13361) 

s_xor_b32 s1, s18, s19 // (op 13361) 

s_sub_i32 s18, s1, s19 // (op 13361) 


s_add_u32 s22, s6, 256 // (op 13361) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13361) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13361) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13361) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13361) 

s_mul_i32 s1, s11, s22 // (op 13361) 

s_mul_i32 s19, s8, s1 // (op 13361) 

s_mul_i32 s1, s18, s19 // (op 13361) 

s_sub_u32 s19, s16, s1 // (op 13361) 

s_mov_b32 s1, s19 // (op 13361) 
s_mul_hi_i32 s22, s1, s12 // (op 13361) 
s_add_i32 s23, s22, s1 // (op 13361) 


s_ashr_i32 s1, s23, 31 // (op 13361) 
s_cmp_eq_i32 s12, 0 // (op 13361) 
s_mov_b32 s22, scc // (op 13361) 
s_cmp_eq_i32 s22, 0 // (op 13361) 
s_cselect_b32 s24, 0, -1 // (op 13361) 

s_and_b32 s22, s44, 31 // (op 13361) 
s_lshl_b32 s25, 1, s22 // (op 13361) 
s_add_i32 s47, s25, s24 // (op 13361) 


s_and_b32 s24, s1, s47 // (op 13361) 


s_add_i32 s1, s23, s24 // (op 13361) 


s_ashr_i32 s23, s1, s22 // (op 13361) 


s_lshl_b32 s1, s44, 24 // (op 13361) 
s_ashr_i32 s22, s1, 31 // (op 13361) 

s_xor_b32 s1, s23, s22 // (op 13361) 

s_sub_i32 s23, s1, s22 // (op 13361) 


s_mul_i32 s1, s46, s11 // (op 13361) 
s_add_u32 s24, s20, 128 // (op 13361) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13361) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 13361) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 13361) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 13361) 

s_sub_i32 s22, s24, s1 // (op 13361) 

s_mul_i32 s24, s23, s22 // (op 13361) 


s_sub_u32 s25, s19, s24 // (op 13361) 


s_add_u32 s19, s25, s1 // (op 13361) 


s_cmp_eq_i32 s18, 0 // (op 13361) 
s_mov_b32 s1, scc // (op 13361) 

s_cmp_eq_i32 s1, 0 // (op 13361) 
s_cselect_b32 s18, s19, s0 // (op 13361) 



v_lshl_add_u32 v11, s18, 7, v10 // (op 13361) 


v_mul_lo_u32 v15, v11, s27 // (op 13361) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 13361) most significant: high of low * low
v_mul_lo_u32 v14, v11, s26 // (op 13361) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 13361) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 13361) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 13361) most significant half



s_mul_i32 s1, s26, s21 // (op 13363) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 13363) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 13363) least significant
s_mul_hi_u32 s19, s26, s20 // (op 13363) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13363) most significant: sum
s_add_u32 s1, s1, s19 // (op 13363) most significant: sum


s_mov_b32 s48, s28 // (op 13363) reg expression
s_mov_b32 s49, s29 // (op 13363) reg expression
s_mov_b32 s50, s0 // (op 13363) reg expression
s_mov_b32 s51, 131072 // (op 13363) call()

s_mul_hi_i32 s47, s16, s10 // (op 13380) 
v_lshrrev_b32 v10, 3, v1 // (op 13380) 
s_add_i32 s0, s47, s16 // (op 13380) 

v_and_b32 v11, -8, v1 // (op 13380) 
s_ashr_i32 s1, s0, 31 // (op 13380) 
v_sub_u32 v12, v1, v11 // (op 13380) 

s_cmp_eq_i32 s10, 0 // (op 13380) 
s_mov_b32 s47, scc // (op 13380) 
v_lshlrev_b32 v11, 4, v12 // (op 13380) 

s_cmp_eq_i32 s47, 0 // (op 13380) 
s_cselect_b32 s18, 0, -1 // (op 13380) 

v_mov_b32 v13, 0 // (op 13380) convert
v_mov_b32 v12, v11 // (op 13380) convert

s_and_b32 s19, s13, 31 // (op 13380) 
s_lshl_b32 s47, 1, s19 // (op 13380) 
s_add_i32 s22, s47, s18 // (op 13380) 


s_and_b32 s18, s1, s22 // (op 13380) 


s_add_i32 s1, s0, s18 // (op 13380) 


s_ashr_i32 s0, s1, s19 // (op 13380) 


s_lshl_b32 s1, s13, 24 // (op 13380) 
s_ashr_i32 s47, s1, 31 // (op 13380) 

s_xor_b32 s1, s0, s47 // (op 13380) 

s_sub_i32 s0, s1, s47 // (op 13380) 


s_mul_hi_i32 s1, s0, s14 // (op 13380) 
s_add_i32 s47, s1, s0 // (op 13380) 

s_ashr_i32 s1, s47, 31 // (op 13380) 
s_cmp_eq_i32 s14, 0 // (op 13380) 
s_mov_b32 s18, scc // (op 13380) 
s_cmp_eq_i32 s18, 0 // (op 13380) 
s_cselect_b32 s19, 0, -1 // (op 13380) 

s_and_b32 s18, s15, 31 // (op 13380) 
s_lshl_b32 s22, 1, s18 // (op 13380) 
s_add_i32 s23, s22, s19 // (op 13380) 


s_and_b32 s19, s1, s23 // (op 13380) 


s_add_i32 s1, s47, s19 // (op 13380) 


s_ashr_i32 s19, s1, s18 // (op 13380) 


s_lshl_b32 s1, s15, 24 // (op 13380) 
s_ashr_i32 s18, s1, 31 // (op 13380) 

s_xor_b32 s1, s19, s18 // (op 13380) 

s_sub_i32 s19, s1, s18 // (op 13380) 


s_mul_i32 s1, s0, s11 // (op 13380) 

s_sub_u32 s0, s16, s1 // (op 13380) 

s_mul_i32 s1, s19, s11 // (op 13380) 
s_add_u32 s1, s1, s0 // (op 13380) 


s_mul_hi_i32 s0, s16, s9 // (op 13380) 
s_add_i32 s47, s0, s16 // (op 13380) 

s_ashr_i32 s0, s47, 31 // (op 13380) 
s_cmp_eq_i32 s9, 0 // (op 13380) 
s_mov_b32 s18, scc // (op 13380) 
s_cmp_eq_i32 s18, 0 // (op 13380) 
s_cselect_b32 s19, 0, -1 // (op 13380) 

s_and_b32 s18, s45, 31 // (op 13380) 
s_lshl_b32 s22, 1, s18 // (op 13380) 
s_add_i32 s23, s22, s19 // (op 13380) 


s_and_b32 s19, s0, s23 // (op 13380) 


s_add_i32 s0, s47, s19 // (op 13380) 


s_ashr_i32 s19, s0, s18 // (op 13380) 


s_lshl_b32 s0, s45, 24 // (op 13380) 
s_ashr_i32 s18, s0, 31 // (op 13380) 

s_xor_b32 s0, s19, s18 // (op 13380) 

s_sub_i32 s19, s0, s18 // (op 13380) 


s_add_u32 s22, s6, 256 // (op 13380) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13380) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13380) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13380) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13380) 

s_mul_i32 s0, s11, s22 // (op 13380) 

s_mul_i32 s18, s8, s0 // (op 13380) 

s_mul_i32 s0, s19, s18 // (op 13380) 

s_sub_u32 s18, s16, s0 // (op 13380) 

s_mov_b32 s0, s18 // (op 13380) 
s_mul_hi_i32 s47, s0, s12 // (op 13380) 
s_add_i32 s22, s47, s0 // (op 13380) 


s_ashr_i32 s0, s22, 31 // (op 13380) 
s_cmp_eq_i32 s12, 0 // (op 13380) 
s_mov_b32 s47, scc // (op 13380) 
s_cmp_eq_i32 s47, 0 // (op 13380) 
s_cselect_b32 s23, 0, -1 // (op 13380) 

s_and_b32 s47, s44, 31 // (op 13380) 
s_lshl_b32 s24, 1, s47 // (op 13380) 
s_add_i32 s25, s24, s23 // (op 13380) 


s_and_b32 s23, s0, s25 // (op 13380) 


s_add_i32 s0, s22, s23 // (op 13380) 


s_ashr_i32 s22, s0, s47 // (op 13380) 


s_lshl_b32 s0, s44, 24 // (op 13380) 
s_ashr_i32 s47, s0, 31 // (op 13380) 

s_xor_b32 s0, s22, s47 // (op 13380) 

s_sub_i32 s22, s0, s47 // (op 13380) 


s_mul_i32 s0, s46, s11 // (op 13380) 
s_add_u32 s24, s20, 128 // (op 13380) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13380) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 13380) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 13380) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 7 // (op 13380) 

s_sub_i32 s23, s24, s0 // (op 13380) 

s_mul_i32 s47, s22, s23 // (op 13380) 


s_sub_u32 s22, s18, s47 // (op 13380) 


s_add_u32 s18, s22, s0 // (op 13380) 


s_cmp_eq_i32 s19, 0 // (op 13380) 
s_mov_b32 s0, scc // (op 13380) 

s_cmp_eq_i32 s0, 0 // (op 13380) 
s_cselect_b32 s19, s18, s1 // (op 13380) 



v_lshl_add_u32 v11, s19, 7, v10 // (op 13380) 


v_mul_lo_u32 v17, v11, s27 // (op 13380) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 13380) most significant: high of low * low
v_mul_lo_u32 v16, v11, s26 // (op 13380) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 13380) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 13380) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 13380) most significant half



s_mul_i32 s1, s26, s21 // (op 13382) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 13382) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 13382) least significant
s_mul_hi_u32 s18, s26, s20 // (op 13382) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13382) most significant: sum
s_add_u32 s1, s1, s18 // (op 13382) most significant: sum


s_mov_b32 s48, s28 // (op 13382) reg expression
s_mov_b32 s49, s29 // (op 13382) reg expression
s_mov_b32 s50, s0 // (op 13382) reg expression
s_mov_b32 s51, 131072 // (op 13382) call()

s_mul_hi_i32 s47, s16, s10 // (op 13425) 
v_lshrrev_b32 v10, 3, v1 // (op 13425) 
s_add_i32 s0, s47, s16 // (op 13425) 

v_and_b32 v11, -8, v1 // (op 13425) 
s_ashr_i32 s1, s0, 31 // (op 13425) 
v_sub_u32 v12, v1, v11 // (op 13425) 

s_cmp_eq_i32 s10, 0 // (op 13425) 
s_mov_b32 s47, scc // (op 13425) 
v_lshlrev_b32 v11, 4, v12 // (op 13425) 

s_cmp_eq_i32 s47, 0 // (op 13425) 
s_cselect_b32 s18, 0, -1 // (op 13425) 

v_mov_b32 v13, 0 // (op 13425) convert
v_mov_b32 v12, v11 // (op 13425) convert

s_and_b32 s19, s13, 31 // (op 13425) 
s_lshl_b32 s47, 1, s19 // (op 13425) 
s_add_i32 s22, s47, s18 // (op 13425) 


s_and_b32 s18, s1, s22 // (op 13425) 


s_add_i32 s1, s0, s18 // (op 13425) 


s_ashr_i32 s0, s1, s19 // (op 13425) 


s_lshl_b32 s1, s13, 24 // (op 13425) 
s_ashr_i32 s47, s1, 31 // (op 13425) 

s_xor_b32 s1, s0, s47 // (op 13425) 

s_sub_i32 s0, s1, s47 // (op 13425) 


s_mul_hi_i32 s1, s0, s14 // (op 13425) 
s_add_i32 s47, s1, s0 // (op 13425) 

s_ashr_i32 s1, s47, 31 // (op 13425) 
s_cmp_eq_i32 s14, 0 // (op 13425) 
s_mov_b32 s18, scc // (op 13425) 
s_cmp_eq_i32 s18, 0 // (op 13425) 
s_cselect_b32 s19, 0, -1 // (op 13425) 

s_and_b32 s18, s15, 31 // (op 13425) 
s_lshl_b32 s22, 1, s18 // (op 13425) 
s_add_i32 s23, s22, s19 // (op 13425) 


s_and_b32 s19, s1, s23 // (op 13425) 


s_add_i32 s1, s47, s19 // (op 13425) 


s_ashr_i32 s19, s1, s18 // (op 13425) 


s_lshl_b32 s1, s15, 24 // (op 13425) 
s_ashr_i32 s18, s1, 31 // (op 13425) 

s_xor_b32 s1, s19, s18 // (op 13425) 

s_sub_i32 s19, s1, s18 // (op 13425) 


s_add_u32 s22, s6, 256 // (op 13425) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13425) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13425) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13425) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13425) 

s_mul_i32 s1, s19, s22 // (op 13425) 

s_sub_i32 s47, s0, s1 // (op 13425) 


s_mul_hi_i32 s0, s16, s9 // (op 13425) 
s_add_i32 s1, s0, s16 // (op 13425) 

s_ashr_i32 s0, s1, 31 // (op 13425) 
s_cmp_eq_i32 s9, 0 // (op 13425) 
s_mov_b32 s18, scc // (op 13425) 
s_cmp_eq_i32 s18, 0 // (op 13425) 
s_cselect_b32 s19, 0, -1 // (op 13425) 

s_and_b32 s18, s45, 31 // (op 13425) 
s_lshl_b32 s24, 1, s18 // (op 13425) 
s_add_i32 s25, s24, s19 // (op 13425) 


s_and_b32 s19, s0, s25 // (op 13425) 


s_add_i32 s0, s1, s19 // (op 13425) 


s_ashr_i32 s1, s0, s18 // (op 13425) 


s_lshl_b32 s0, s45, 24 // (op 13425) 
s_ashr_i32 s18, s0, 31 // (op 13425) 

s_xor_b32 s0, s1, s18 // (op 13425) 

s_sub_i32 s1, s0, s18 // (op 13425) 


s_mul_i32 s0, s11, s22 // (op 13425) 

s_mul_i32 s18, s8, s0 // (op 13425) 

s_mul_i32 s0, s1, s18 // (op 13425) 

s_sub_u32 s18, s16, s0 // (op 13425) 

s_mul_hi_i32 s0, s18, s12 // (op 13425) 
s_add_i32 s19, s0, s18 // (op 13425) 


s_ashr_i32 s0, s19, 31 // (op 13425) 
s_cmp_eq_i32 s12, 0 // (op 13425) 
s_mov_b32 s18, scc // (op 13425) 
s_cmp_eq_i32 s18, 0 // (op 13425) 
s_cselect_b32 s22, 0, -1 // (op 13425) 

s_and_b32 s18, s44, 31 // (op 13425) 
s_lshl_b32 s23, 1, s18 // (op 13425) 
s_add_i32 s24, s23, s22 // (op 13425) 


s_and_b32 s25, s0, s24 // (op 13425) 


s_add_i32 s0, s19, s25 // (op 13425) 


s_ashr_i32 s19, s0, s18 // (op 13425) 


s_lshl_b32 s0, s44, 24 // (op 13425) 
s_ashr_i32 s18, s0, 31 // (op 13425) 

s_xor_b32 s0, s19, s18 // (op 13425) 

s_sub_i32 s19, s0, s18 // (op 13425) 


s_cmp_eq_i32 s1, 0 // (op 13425) 
s_mov_b32 s0, scc // (op 13425) 

s_cmp_eq_i32 s0, 0 // (op 13425) 
s_cselect_b32 s1, s19, s47 // (op 13425) 



v_lshl_add_u32 v11, s1, 8, v10 // (op 13425) 


v_mul_lo_u32 v19, v11, s31 // (op 13425) most significant: low of low * high
v_mul_hi_u32 v10, v11, s30 // (op 13425) most significant: high of low * low
v_mul_lo_u32 v18, v11, s30 // (op 13425) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 13425) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v18 // (op 13425) least significant half
v_addc_co_u32 v11, s[0:1], v13, v19, s[0:1] // (op 13425) most significant half



s_mul_i32 s1, s30, s7 // (op 13427) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 13427) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 13427) least significant
s_mul_hi_u32 s18, s30, s6 // (op 13427) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13427) most significant: sum
s_add_u32 s1, s1, s18 // (op 13427) most significant: sum


s_mov_b32 s60, s32 // (op 13427) reg expression
s_mov_b32 s61, s33 // (op 13427) reg expression
s_mov_b32 s62, s0 // (op 13427) reg expression
s_mov_b32 s63, 131072 // (op 13427) call()

s_mul_hi_i32 s47, s16, s10 // (op 13444) 
v_lshrrev_b32 v12, 3, v1 // (op 13444) 
s_add_i32 s0, s47, s16 // (op 13444) 

v_and_b32 v13, -8, v1 // (op 13444) 
s_ashr_i32 s1, s0, 31 // (op 13444) 
v_sub_u32 v18, v1, v13 // (op 13444) 

s_cmp_eq_i32 s10, 0 // (op 13444) 
s_mov_b32 s47, scc // (op 13444) 
v_lshlrev_b32 v13, 4, v18 // (op 13444) 

s_cmp_eq_i32 s47, 0 // (op 13444) 
s_cselect_b32 s18, 0, -1 // (op 13444) 

v_mov_b32 v19, 0 // (op 13444) convert
v_mov_b32 v18, v13 // (op 13444) convert

s_and_b32 s19, s13, 31 // (op 13444) 
s_lshl_b32 s47, 1, s19 // (op 13444) 
s_add_i32 s22, s47, s18 // (op 13444) 


s_and_b32 s18, s1, s22 // (op 13444) 


s_add_i32 s1, s0, s18 // (op 13444) 


s_ashr_i32 s0, s1, s19 // (op 13444) 


s_lshl_b32 s1, s13, 24 // (op 13444) 
s_ashr_i32 s47, s1, 31 // (op 13444) 

s_xor_b32 s1, s0, s47 // (op 13444) 

s_sub_i32 s0, s1, s47 // (op 13444) 


s_mul_hi_i32 s1, s0, s14 // (op 13444) 
s_add_i32 s47, s1, s0 // (op 13444) 

s_ashr_i32 s1, s47, 31 // (op 13444) 
s_cmp_eq_i32 s14, 0 // (op 13444) 
s_mov_b32 s18, scc // (op 13444) 
s_cmp_eq_i32 s18, 0 // (op 13444) 
s_cselect_b32 s19, 0, -1 // (op 13444) 

s_and_b32 s18, s15, 31 // (op 13444) 
s_lshl_b32 s22, 1, s18 // (op 13444) 
s_add_i32 s23, s22, s19 // (op 13444) 


s_and_b32 s19, s1, s23 // (op 13444) 


s_add_i32 s1, s47, s19 // (op 13444) 


s_ashr_i32 s19, s1, s18 // (op 13444) 


s_lshl_b32 s1, s15, 24 // (op 13444) 
s_ashr_i32 s18, s1, 31 // (op 13444) 

s_xor_b32 s1, s19, s18 // (op 13444) 

s_sub_i32 s19, s1, s18 // (op 13444) 


s_add_u32 s22, s6, 256 // (op 13444) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13444) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13444) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13444) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13444) 

s_mul_i32 s1, s19, s22 // (op 13444) 

s_sub_i32 s47, s0, s1 // (op 13444) 


s_mul_hi_i32 s0, s16, s9 // (op 13444) 
s_add_i32 s1, s0, s16 // (op 13444) 

s_ashr_i32 s0, s1, 31 // (op 13444) 
s_cmp_eq_i32 s9, 0 // (op 13444) 
s_mov_b32 s18, scc // (op 13444) 
s_cmp_eq_i32 s18, 0 // (op 13444) 
s_cselect_b32 s19, 0, -1 // (op 13444) 

s_and_b32 s18, s45, 31 // (op 13444) 
s_lshl_b32 s24, 1, s18 // (op 13444) 
s_add_i32 s25, s24, s19 // (op 13444) 


s_and_b32 s19, s0, s25 // (op 13444) 


s_add_i32 s0, s1, s19 // (op 13444) 


s_ashr_i32 s1, s0, s18 // (op 13444) 


s_lshl_b32 s0, s45, 24 // (op 13444) 
s_ashr_i32 s18, s0, 31 // (op 13444) 

s_xor_b32 s0, s1, s18 // (op 13444) 

s_sub_i32 s1, s0, s18 // (op 13444) 


s_mul_i32 s0, s11, s22 // (op 13444) 

s_mul_i32 s18, s8, s0 // (op 13444) 

s_mul_i32 s0, s1, s18 // (op 13444) 

s_sub_u32 s18, s16, s0 // (op 13444) 

s_mul_hi_i32 s0, s18, s12 // (op 13444) 
s_add_i32 s19, s0, s18 // (op 13444) 


s_ashr_i32 s0, s19, 31 // (op 13444) 
s_cmp_eq_i32 s12, 0 // (op 13444) 
s_mov_b32 s18, scc // (op 13444) 
s_cmp_eq_i32 s18, 0 // (op 13444) 
s_cselect_b32 s22, 0, -1 // (op 13444) 

s_and_b32 s18, s44, 31 // (op 13444) 
s_lshl_b32 s23, 1, s18 // (op 13444) 
s_add_i32 s24, s23, s22 // (op 13444) 


s_and_b32 s25, s0, s24 // (op 13444) 


s_add_i32 s0, s19, s25 // (op 13444) 


s_ashr_i32 s19, s0, s18 // (op 13444) 


s_lshl_b32 s0, s44, 24 // (op 13444) 
s_ashr_i32 s18, s0, 31 // (op 13444) 

s_xor_b32 s0, s19, s18 // (op 13444) 

s_sub_i32 s19, s0, s18 // (op 13444) 


s_cmp_eq_i32 s1, 0 // (op 13444) 
s_mov_b32 s0, scc // (op 13444) 

s_cmp_eq_i32 s0, 0 // (op 13444) 
s_cselect_b32 s1, s19, s47 // (op 13444) 



v_lshl_add_u32 v13, s1, 8, v12 // (op 13444) 


v_mul_lo_u32 v21, v13, s31 // (op 13444) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 13444) most significant: high of low * low
v_mul_lo_u32 v20, v13, s30 // (op 13444) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 13444) most significant: sum


v_add_co_u32 v12, s[0:1], v18, v20 // (op 13444) least significant half
v_addc_co_u32 v13, s[0:1], v19, v21, s[0:1] // (op 13444) most significant half



s_mul_i32 s1, s30, s7 // (op 13446) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 13446) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 13446) least significant
s_mul_hi_u32 s18, s30, s6 // (op 13446) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13446) most significant: sum
s_add_u32 s1, s1, s18 // (op 13446) most significant: sum


s_mov_b32 s60, s32 // (op 13446) reg expression
s_mov_b32 s61, s33 // (op 13446) reg expression
s_mov_b32 s62, s0 // (op 13446) reg expression
s_mov_b32 s63, 131072 // (op 13446) call()

s_mul_hi_i32 s47, s16, s10 // (op 13722) 
v_and_b32 v18, 63, v1 // (op 13722) 
s_add_i32 s0, s47, s16 // (op 13722) 

v_lshrrev_b32 v19, 4, v18 // (op 13722) 
s_ashr_i32 s1, s0, 31 // (op 13722) 
v_and_b32 v20, 3, v19 // (op 13722) 

s_cmp_eq_i32 s10, 0 // (op 13722) 
s_mov_b32 s47, scc // (op 13722) 
v_and_b32 v19, 15, v1 // (op 13722) 
s_cmp_eq_i32 s47, 0 // (op 13722) 
s_cselect_b32 s18, 0, -1 // (op 13722) 

v_lshl_add_u32 v21, v20, 4, v19 // (op 13722) 


s_and_b32 s19, s13, 31 // (op 13722) 
v_lshrrev_b32 v19, 7, v1 // (op 13722) 
s_lshl_b32 s47, 1, s19 // (op 13722) 
v_lshl_add_u32 v20, v19, 6, v21 // (op 13722) 


s_add_i32 s22, s47, s18 // (op 13722) 


v_lshrrev_b32 v19, 6, v18 // (op 13722) 

s_and_b32 s18, s1, s22 // (op 13722) 


v_lshlrev_b32 v18, 2, v19 // (op 13722) 

s_add_i32 s1, s0, s18 // (op 13722) 


s_ashr_i32 s0, s1, s19 // (op 13722) 


s_lshl_b32 s1, s13, 24 // (op 13722) 
s_ashr_i32 s47, s1, 31 // (op 13722) 

s_xor_b32 s1, s0, s47 // (op 13722) 

s_sub_i32 s0, s1, s47 // (op 13722) 


s_mul_hi_i32 s1, s0, s14 // (op 13722) 
s_add_i32 s47, s1, s0 // (op 13722) 

s_ashr_i32 s1, s47, 31 // (op 13722) 
s_cmp_eq_i32 s14, 0 // (op 13722) 
s_mov_b32 s18, scc // (op 13722) 
s_cmp_eq_i32 s18, 0 // (op 13722) 
s_cselect_b32 s19, 0, -1 // (op 13722) 

s_and_b32 s18, s15, 31 // (op 13722) 
s_lshl_b32 s22, 1, s18 // (op 13722) 
s_add_i32 s23, s22, s19 // (op 13722) 


s_and_b32 s19, s1, s23 // (op 13722) 


s_add_i32 s1, s47, s19 // (op 13722) 


s_ashr_i32 s19, s1, s18 // (op 13722) 


s_lshl_b32 s1, s15, 24 // (op 13722) 
s_ashr_i32 s18, s1, 31 // (op 13722) 

s_xor_b32 s1, s19, s18 // (op 13722) 

s_sub_i32 s19, s1, s18 // (op 13722) 


s_mul_i32 s1, s0, s11 // (op 13722) 

s_sub_u32 s0, s16, s1 // (op 13722) 

s_mul_i32 s1, s19, s11 // (op 13722) 
s_add_u32 s1, s1, s0 // (op 13722) 


s_mul_hi_i32 s0, s16, s9 // (op 13722) 
s_add_i32 s47, s0, s16 // (op 13722) 

s_ashr_i32 s0, s47, 31 // (op 13722) 
s_cmp_eq_i32 s9, 0 // (op 13722) 
s_mov_b32 s18, scc // (op 13722) 
s_cmp_eq_i32 s18, 0 // (op 13722) 
s_cselect_b32 s19, 0, -1 // (op 13722) 

s_and_b32 s18, s45, 31 // (op 13722) 
s_lshl_b32 s22, 1, s18 // (op 13722) 
s_add_i32 s23, s22, s19 // (op 13722) 


s_and_b32 s19, s0, s23 // (op 13722) 


s_add_i32 s0, s47, s19 // (op 13722) 


s_ashr_i32 s19, s0, s18 // (op 13722) 


s_lshl_b32 s0, s45, 24 // (op 13722) 
s_ashr_i32 s18, s0, 31 // (op 13722) 

s_xor_b32 s0, s19, s18 // (op 13722) 

s_sub_i32 s19, s0, s18 // (op 13722) 


s_add_u32 s22, s6, 256 // (op 13722) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13722) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13722) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13722) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13722) 

s_mul_i32 s0, s11, s22 // (op 13722) 

s_mul_i32 s18, s8, s0 // (op 13722) 

s_mul_i32 s0, s19, s18 // (op 13722) 

s_sub_u32 s18, s16, s0 // (op 13722) 

s_mov_b32 s0, s18 // (op 13722) 
s_mul_hi_i32 s47, s0, s12 // (op 13722) 
s_add_i32 s22, s47, s0 // (op 13722) 


s_ashr_i32 s0, s22, 31 // (op 13722) 
s_cmp_eq_i32 s12, 0 // (op 13722) 
s_mov_b32 s47, scc // (op 13722) 
s_cmp_eq_i32 s47, 0 // (op 13722) 
s_cselect_b32 s23, 0, -1 // (op 13722) 

s_and_b32 s47, s44, 31 // (op 13722) 
s_lshl_b32 s24, 1, s47 // (op 13722) 
s_add_i32 s25, s24, s23 // (op 13722) 


s_and_b32 s23, s0, s25 // (op 13722) 


s_add_i32 s0, s22, s23 // (op 13722) 


s_ashr_i32 s22, s0, s47 // (op 13722) 


s_lshl_b32 s0, s44, 24 // (op 13722) 
s_ashr_i32 s47, s0, 31 // (op 13722) 

s_xor_b32 s0, s22, s47 // (op 13722) 

s_sub_i32 s22, s0, s47 // (op 13722) 


s_mul_i32 s0, s46, s11 // (op 13722) 
s_add_u32 s24, s20, 128 // (op 13722) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13722) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 13722) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 13722) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 13722) 

s_sub_i32 s23, s24, s0 // (op 13722) 

s_mul_i32 s47, s22, s23 // (op 13722) 


s_sub_u32 s22, s18, s47 // (op 13722) 


s_add_u32 s18, s22, s0 // (op 13722) 


s_cmp_eq_i32 s19, 0 // (op 13722) 
s_mov_b32 s0, scc // (op 13722) 

s_cmp_eq_i32 s0, 0 // (op 13722) 
s_cselect_b32 s19, s18, s1 // (op 13722) 



v_lshl_add_u32 v19, s19, 7, v20 // (op 13722) 


v_mul_lo_u32 v20, v19, s34 // (op 13722) 
v_add_u32 v20, v20, v18 // (op 13722) 


s_mul_i32 s1, s34, s21 // (op 13724) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 13724) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 13724) least significant
s_mul_hi_u32 s18, s34, s20 // (op 13724) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13724) most significant: sum
s_add_u32 s1, s1, s18 // (op 13724) most significant: sum


s_mov_b32 s64, s36 // (op 13724) reg expression
s_mov_b32 s65, s37 // (op 13724) reg expression
s_mov_b32 s66, s0 // (op 13724) reg expression
s_mov_b32 s67, 131072 // (op 13724) call()

s_mul_hi_i32 s47, s16, s10 // (op 13743) 
v_and_b32 v18, 63, v1 // (op 13743) 
s_add_i32 s0, s47, s16 // (op 13743) 

v_lshrrev_b32 v19, 4, v18 // (op 13743) 
s_ashr_i32 s1, s0, 31 // (op 13743) 
v_and_b32 v21, 3, v19 // (op 13743) 

s_cmp_eq_i32 s10, 0 // (op 13743) 
s_mov_b32 s47, scc // (op 13743) 
v_and_b32 v19, 15, v1 // (op 13743) 
s_cmp_eq_i32 s47, 0 // (op 13743) 
s_cselect_b32 s18, 0, -1 // (op 13743) 

v_lshl_add_u32 v22, v21, 4, v19 // (op 13743) 


s_and_b32 s19, s13, 31 // (op 13743) 
v_lshrrev_b32 v19, 7, v1 // (op 13743) 
s_lshl_b32 s47, 1, s19 // (op 13743) 
v_lshl_add_u32 v21, v19, 6, v22 // (op 13743) 


s_add_i32 s22, s47, s18 // (op 13743) 


v_lshrrev_b32 v19, 6, v18 // (op 13743) 

s_and_b32 s18, s1, s22 // (op 13743) 


v_lshlrev_b32 v18, 2, v19 // (op 13743) 

s_add_i32 s1, s0, s18 // (op 13743) 


s_ashr_i32 s0, s1, s19 // (op 13743) 


s_lshl_b32 s1, s13, 24 // (op 13743) 
s_ashr_i32 s47, s1, 31 // (op 13743) 

s_xor_b32 s1, s0, s47 // (op 13743) 

s_sub_i32 s0, s1, s47 // (op 13743) 


s_mul_hi_i32 s1, s0, s14 // (op 13743) 
s_add_i32 s47, s1, s0 // (op 13743) 

s_ashr_i32 s1, s47, 31 // (op 13743) 
s_cmp_eq_i32 s14, 0 // (op 13743) 
s_mov_b32 s18, scc // (op 13743) 
s_cmp_eq_i32 s18, 0 // (op 13743) 
s_cselect_b32 s19, 0, -1 // (op 13743) 

s_and_b32 s18, s15, 31 // (op 13743) 
s_lshl_b32 s22, 1, s18 // (op 13743) 
s_add_i32 s23, s22, s19 // (op 13743) 


s_and_b32 s19, s1, s23 // (op 13743) 


s_add_i32 s1, s47, s19 // (op 13743) 


s_ashr_i32 s19, s1, s18 // (op 13743) 


s_lshl_b32 s1, s15, 24 // (op 13743) 
s_ashr_i32 s18, s1, 31 // (op 13743) 

s_xor_b32 s1, s19, s18 // (op 13743) 

s_sub_i32 s19, s1, s18 // (op 13743) 


s_mul_i32 s1, s0, s11 // (op 13743) 

s_sub_u32 s0, s16, s1 // (op 13743) 

s_mul_i32 s1, s19, s11 // (op 13743) 
s_add_u32 s1, s1, s0 // (op 13743) 


s_mul_hi_i32 s0, s16, s9 // (op 13743) 
s_add_i32 s47, s0, s16 // (op 13743) 

s_ashr_i32 s0, s47, 31 // (op 13743) 
s_cmp_eq_i32 s9, 0 // (op 13743) 
s_mov_b32 s18, scc // (op 13743) 
s_cmp_eq_i32 s18, 0 // (op 13743) 
s_cselect_b32 s19, 0, -1 // (op 13743) 

s_and_b32 s18, s45, 31 // (op 13743) 
s_lshl_b32 s22, 1, s18 // (op 13743) 
s_add_i32 s23, s22, s19 // (op 13743) 


s_and_b32 s19, s0, s23 // (op 13743) 


s_add_i32 s0, s47, s19 // (op 13743) 


s_ashr_i32 s19, s0, s18 // (op 13743) 


s_lshl_b32 s0, s45, 24 // (op 13743) 
s_ashr_i32 s18, s0, 31 // (op 13743) 

s_xor_b32 s0, s19, s18 // (op 13743) 

s_sub_i32 s19, s0, s18 // (op 13743) 


s_add_u32 s22, s6, 256 // (op 13743) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13743) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13743) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13743) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13743) 

s_mul_i32 s0, s11, s22 // (op 13743) 

s_mul_i32 s18, s8, s0 // (op 13743) 

s_mul_i32 s0, s19, s18 // (op 13743) 

s_sub_u32 s18, s16, s0 // (op 13743) 

s_mov_b32 s0, s18 // (op 13743) 
s_mul_hi_i32 s47, s0, s12 // (op 13743) 
s_add_i32 s22, s47, s0 // (op 13743) 


s_ashr_i32 s0, s22, 31 // (op 13743) 
s_cmp_eq_i32 s12, 0 // (op 13743) 
s_mov_b32 s47, scc // (op 13743) 
s_cmp_eq_i32 s47, 0 // (op 13743) 
s_cselect_b32 s23, 0, -1 // (op 13743) 

s_and_b32 s47, s44, 31 // (op 13743) 
s_lshl_b32 s24, 1, s47 // (op 13743) 
s_add_i32 s25, s24, s23 // (op 13743) 


s_and_b32 s23, s0, s25 // (op 13743) 


s_add_i32 s0, s22, s23 // (op 13743) 


s_ashr_i32 s22, s0, s47 // (op 13743) 


s_lshl_b32 s0, s44, 24 // (op 13743) 
s_ashr_i32 s47, s0, 31 // (op 13743) 

s_xor_b32 s0, s22, s47 // (op 13743) 

s_sub_i32 s22, s0, s47 // (op 13743) 


s_mul_i32 s0, s46, s11 // (op 13743) 
s_add_u32 s24, s20, 128 // (op 13743) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13743) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 13743) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 13743) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 7 // (op 13743) 

s_sub_i32 s23, s24, s0 // (op 13743) 

s_mul_i32 s47, s22, s23 // (op 13743) 


s_sub_u32 s22, s18, s47 // (op 13743) 


s_add_u32 s18, s22, s0 // (op 13743) 


s_cmp_eq_i32 s19, 0 // (op 13743) 
s_mov_b32 s0, scc // (op 13743) 

s_cmp_eq_i32 s0, 0 // (op 13743) 
s_cselect_b32 s19, s18, s1 // (op 13743) 



v_lshl_add_u32 v19, s19, 7, v21 // (op 13743) 


v_mul_lo_u32 v21, v19, s34 // (op 13743) 
v_add_u32 v21, v21, v18 // (op 13743) 


s_mul_i32 s1, s34, s21 // (op 13745) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 13745) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 13745) least significant
s_mul_hi_u32 s18, s34, s20 // (op 13745) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13745) most significant: sum
s_add_u32 s1, s1, s18 // (op 13745) most significant: sum


s_mov_b32 s64, s36 // (op 13745) reg expression
s_mov_b32 s65, s37 // (op 13745) reg expression
s_mov_b32 s66, s0 // (op 13745) reg expression
s_mov_b32 s67, 131072 // (op 13745) call()

s_mul_hi_i32 s47, s16, s10 // (op 13774) 
v_and_b32 v18, 63, v1 // (op 13774) 
s_add_i32 s0, s47, s16 // (op 13774) 

v_lshrrev_b32 v19, 4, v18 // (op 13774) 
s_ashr_i32 s1, s0, 31 // (op 13774) 
v_and_b32 v22, 3, v19 // (op 13774) 

s_cmp_eq_i32 s10, 0 // (op 13774) 
s_mov_b32 s47, scc // (op 13774) 
v_and_b32 v19, 15, v1 // (op 13774) 
s_cmp_eq_i32 s47, 0 // (op 13774) 
s_cselect_b32 s18, 0, -1 // (op 13774) 

v_lshl_add_u32 v23, v22, 4, v19 // (op 13774) 


s_and_b32 s19, s13, 31 // (op 13774) 
v_lshrrev_b32 v19, 6, v1 // (op 13774) 
s_lshl_b32 s47, 1, s19 // (op 13774) 
v_and_b32 v22, 1, v19 // (op 13774) 

s_add_i32 s22, s47, s18 // (op 13774) 


v_lshl_add_u32 v19, v22, 7, v23 // (op 13774) 


s_and_b32 s18, s1, s22 // (op 13774) 


v_lshrrev_b32 v22, 6, v18 // (op 13774) 

s_add_i32 s1, s0, s18 // (op 13774) 


s_ashr_i32 s0, s1, s19 // (op 13774) 


s_lshl_b32 s1, s13, 24 // (op 13774) 
s_ashr_i32 s47, s1, 31 // (op 13774) 

s_xor_b32 s1, s0, s47 // (op 13774) 

s_sub_i32 s0, s1, s47 // (op 13774) 


s_mul_hi_i32 s1, s0, s14 // (op 13774) 
s_add_i32 s47, s1, s0 // (op 13774) 

s_ashr_i32 s1, s47, 31 // (op 13774) 
s_cmp_eq_i32 s14, 0 // (op 13774) 
s_mov_b32 s18, scc // (op 13774) 
s_cmp_eq_i32 s18, 0 // (op 13774) 
s_cselect_b32 s19, 0, -1 // (op 13774) 

s_and_b32 s18, s15, 31 // (op 13774) 
s_lshl_b32 s22, 1, s18 // (op 13774) 
s_add_i32 s23, s22, s19 // (op 13774) 


s_and_b32 s19, s1, s23 // (op 13774) 


s_add_i32 s1, s47, s19 // (op 13774) 


s_ashr_i32 s19, s1, s18 // (op 13774) 


s_lshl_b32 s1, s15, 24 // (op 13774) 
s_ashr_i32 s18, s1, 31 // (op 13774) 

s_xor_b32 s1, s19, s18 // (op 13774) 

s_sub_i32 s19, s1, s18 // (op 13774) 


s_add_u32 s22, s6, 256 // (op 13774) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13774) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13774) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13774) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13774) 

s_mul_i32 s1, s19, s22 // (op 13774) 

s_sub_i32 s47, s0, s1 // (op 13774) 


s_mul_hi_i32 s0, s16, s9 // (op 13774) 
s_add_i32 s1, s0, s16 // (op 13774) 

s_ashr_i32 s0, s1, 31 // (op 13774) 
s_cmp_eq_i32 s9, 0 // (op 13774) 
s_mov_b32 s18, scc // (op 13774) 
s_cmp_eq_i32 s18, 0 // (op 13774) 
s_cselect_b32 s19, 0, -1 // (op 13774) 

s_and_b32 s18, s45, 31 // (op 13774) 
s_lshl_b32 s24, 1, s18 // (op 13774) 
s_add_i32 s25, s24, s19 // (op 13774) 


s_and_b32 s19, s0, s25 // (op 13774) 


s_add_i32 s0, s1, s19 // (op 13774) 


s_ashr_i32 s1, s0, s18 // (op 13774) 


s_lshl_b32 s0, s45, 24 // (op 13774) 
s_ashr_i32 s18, s0, 31 // (op 13774) 

s_xor_b32 s0, s1, s18 // (op 13774) 

s_sub_i32 s1, s0, s18 // (op 13774) 


s_mul_i32 s0, s11, s22 // (op 13774) 

s_mul_i32 s18, s8, s0 // (op 13774) 

s_mul_i32 s0, s1, s18 // (op 13774) 

s_sub_u32 s18, s16, s0 // (op 13774) 

s_mul_hi_i32 s0, s18, s12 // (op 13774) 
s_add_i32 s19, s0, s18 // (op 13774) 


s_ashr_i32 s0, s19, 31 // (op 13774) 
s_cmp_eq_i32 s12, 0 // (op 13774) 
s_mov_b32 s18, scc // (op 13774) 
s_cmp_eq_i32 s18, 0 // (op 13774) 
s_cselect_b32 s22, 0, -1 // (op 13774) 

s_and_b32 s18, s44, 31 // (op 13774) 
s_lshl_b32 s23, 1, s18 // (op 13774) 
s_add_i32 s24, s23, s22 // (op 13774) 


s_and_b32 s25, s0, s24 // (op 13774) 


s_add_i32 s0, s19, s25 // (op 13774) 


s_ashr_i32 s19, s0, s18 // (op 13774) 


s_lshl_b32 s0, s44, 24 // (op 13774) 
s_ashr_i32 s18, s0, 31 // (op 13774) 

s_xor_b32 s0, s19, s18 // (op 13774) 

s_sub_i32 s19, s0, s18 // (op 13774) 


s_cmp_eq_i32 s1, 0 // (op 13774) 
s_mov_b32 s0, scc // (op 13774) 

s_cmp_eq_i32 s0, 0 // (op 13774) 
s_cselect_b32 s1, s19, s47 // (op 13774) 



v_lshl_add_u32 v18, s1, 8, v19 // (op 13774) 


v_mul_lo_u32 v19, v18, s38 // (op 13774) 

v_lshl_add_u32 v18, v22, 2, v19 // (op 13774) 


s_mul_i32 s1, s38, s7 // (op 13776) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 13776) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 13776) least significant
s_mul_hi_u32 s18, s38, s6 // (op 13776) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13776) most significant: sum
s_add_u32 s1, s1, s18 // (op 13776) most significant: sum


s_mov_b32 s68, s40 // (op 13776) reg expression
s_mov_b32 s69, s41 // (op 13776) reg expression
s_mov_b32 s70, s0 // (op 13776) reg expression
s_mov_b32 s71, 131072 // (op 13776) call()

s_mul_hi_i32 s47, s16, s10 // (op 13795) 
v_and_b32 v19, 63, v1 // (op 13795) 
s_add_i32 s0, s47, s16 // (op 13795) 

v_lshrrev_b32 v22, 4, v19 // (op 13795) 
s_ashr_i32 s1, s0, 31 // (op 13795) 
v_and_b32 v23, 3, v22 // (op 13795) 

s_cmp_eq_i32 s10, 0 // (op 13795) 
s_mov_b32 s47, scc // (op 13795) 
v_and_b32 v22, 15, v1 // (op 13795) 
s_cmp_eq_i32 s47, 0 // (op 13795) 
s_cselect_b32 s18, 0, -1 // (op 13795) 

v_lshl_add_u32 v24, v23, 4, v22 // (op 13795) 


s_and_b32 s19, s13, 31 // (op 13795) 
v_lshrrev_b32 v22, 6, v1 // (op 13795) 
s_lshl_b32 s47, 1, s19 // (op 13795) 
v_and_b32 v23, 1, v22 // (op 13795) 

s_add_i32 s22, s47, s18 // (op 13795) 


v_lshl_add_u32 v22, v23, 7, v24 // (op 13795) 


s_and_b32 s18, s1, s22 // (op 13795) 


v_lshrrev_b32 v23, 6, v19 // (op 13795) 

s_add_i32 s1, s0, s18 // (op 13795) 


s_ashr_i32 s0, s1, s19 // (op 13795) 


s_lshl_b32 s1, s13, 24 // (op 13795) 
s_ashr_i32 s47, s1, 31 // (op 13795) 

s_xor_b32 s1, s0, s47 // (op 13795) 

s_sub_i32 s0, s1, s47 // (op 13795) 


s_mul_hi_i32 s1, s0, s14 // (op 13795) 
s_add_i32 s47, s1, s0 // (op 13795) 

s_ashr_i32 s1, s47, 31 // (op 13795) 
s_cmp_eq_i32 s14, 0 // (op 13795) 
s_mov_b32 s18, scc // (op 13795) 
s_cmp_eq_i32 s18, 0 // (op 13795) 
s_cselect_b32 s19, 0, -1 // (op 13795) 

s_and_b32 s18, s15, 31 // (op 13795) 
s_lshl_b32 s22, 1, s18 // (op 13795) 
s_add_i32 s23, s22, s19 // (op 13795) 


s_and_b32 s19, s1, s23 // (op 13795) 


s_add_i32 s1, s47, s19 // (op 13795) 


s_ashr_i32 s19, s1, s18 // (op 13795) 


s_lshl_b32 s1, s15, 24 // (op 13795) 
s_ashr_i32 s18, s1, 31 // (op 13795) 

s_xor_b32 s1, s19, s18 // (op 13795) 

s_sub_i32 s19, s1, s18 // (op 13795) 


s_add_u32 s22, s6, 256 // (op 13795) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13795) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13795) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13795) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13795) 

s_mul_i32 s1, s19, s22 // (op 13795) 

s_sub_i32 s47, s0, s1 // (op 13795) 


s_mul_hi_i32 s0, s16, s9 // (op 13795) 
s_add_i32 s1, s0, s16 // (op 13795) 

s_ashr_i32 s0, s1, 31 // (op 13795) 
s_cmp_eq_i32 s9, 0 // (op 13795) 
s_mov_b32 s18, scc // (op 13795) 
s_cmp_eq_i32 s18, 0 // (op 13795) 
s_cselect_b32 s19, 0, -1 // (op 13795) 

s_and_b32 s18, s45, 31 // (op 13795) 
s_lshl_b32 s24, 1, s18 // (op 13795) 
s_add_i32 s25, s24, s19 // (op 13795) 


s_and_b32 s19, s0, s25 // (op 13795) 


s_add_i32 s0, s1, s19 // (op 13795) 


s_ashr_i32 s1, s0, s18 // (op 13795) 


s_lshl_b32 s0, s45, 24 // (op 13795) 
s_ashr_i32 s18, s0, 31 // (op 13795) 

s_xor_b32 s0, s1, s18 // (op 13795) 

s_sub_i32 s1, s0, s18 // (op 13795) 


s_mul_i32 s0, s11, s22 // (op 13795) 

s_mul_i32 s18, s8, s0 // (op 13795) 

s_mul_i32 s0, s1, s18 // (op 13795) 

s_sub_u32 s18, s16, s0 // (op 13795) 

s_mul_hi_i32 s0, s18, s12 // (op 13795) 
s_add_i32 s19, s0, s18 // (op 13795) 


s_ashr_i32 s0, s19, 31 // (op 13795) 
s_cmp_eq_i32 s12, 0 // (op 13795) 
s_mov_b32 s18, scc // (op 13795) 
s_cmp_eq_i32 s18, 0 // (op 13795) 
s_cselect_b32 s22, 0, -1 // (op 13795) 

s_and_b32 s18, s44, 31 // (op 13795) 
s_lshl_b32 s23, 1, s18 // (op 13795) 
s_add_i32 s24, s23, s22 // (op 13795) 


s_and_b32 s25, s0, s24 // (op 13795) 


s_add_i32 s0, s19, s25 // (op 13795) 


s_ashr_i32 s19, s0, s18 // (op 13795) 


s_lshl_b32 s0, s44, 24 // (op 13795) 
s_ashr_i32 s18, s0, 31 // (op 13795) 

s_xor_b32 s0, s19, s18 // (op 13795) 

s_sub_i32 s19, s0, s18 // (op 13795) 


s_cmp_eq_i32 s1, 0 // (op 13795) 
s_mov_b32 s0, scc // (op 13795) 

s_cmp_eq_i32 s0, 0 // (op 13795) 
s_cselect_b32 s1, s19, s47 // (op 13795) 



v_lshl_add_u32 v19, s1, 8, v22 // (op 13795) 


v_mul_lo_u32 v22, v19, s38 // (op 13795) 

v_lshl_add_u32 v19, v23, 2, v22 // (op 13795) 


s_mul_i32 s1, s38, s7 // (op 13797) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 13797) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 13797) least significant
s_mul_hi_u32 s18, s38, s6 // (op 13797) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13797) most significant: sum
s_add_u32 s1, s1, s18 // (op 13797) most significant: sum


s_mov_b32 s68, s40 // (op 13797) reg expression
s_mov_b32 s69, s41 // (op 13797) reg expression
s_mov_b32 s70, s0 // (op 13797) reg expression
s_mov_b32 s71, 131072 // (op 13797) call()

v_and_b32 v22, -8, v1 // (op 13572) 
v_sub_u32 v23, v1, v22 // (op 13572) 

v_lshlrev_b32 v22, 4, v23 // (op 13572) 

v_lshrrev_b32 v23, 3, v1 // (op 13572) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 13572) 


s_nop 0
v_readfirstlane_b32 s47, v24 // (op 13572) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 5 // (op 13279) 
v_mov_b64 v[22:23], s[0:1] // (op 13279) convert


v_mov_b64 v[24:25], v[10:11] // (op 13279) 
s_mov_b32 m0, s47 // (op 13279) reg expression
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 13279) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 13279) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 13279) most significant half

s_add_u32 m0, m0, 4096 // (op 13279) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 13279) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 13279) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 13279) most significant half

s_add_u32 m0, m0, 4096 // (op 13279) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 13279) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 13279) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 13279) most significant half

s_add_u32 m0, m0, 4096 // (op 13279) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 13279) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 13279) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 13279) most significant half

s_add_u32 m0, m0, 4096 // (op 13279) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 13279) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 13279) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 13279) most significant half

s_add_u32 m0, m0, 4096 // (op 13279) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 13279) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 13279) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 13279) most significant half

s_add_u32 m0, m0, 4096 // (op 13279) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 13279) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 13279) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 13279) most significant half

s_add_u32 m0, m0, 4096 // (op 13279) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 13279) Load value direct to lds



v_and_b32 v22, -8, v1 // (op 13515) 
v_sub_u32 v23, v1, v22 // (op 13515) 

v_lshlrev_b32 v22, 4, v23 // (op 13515) 

v_lshrrev_b32 v23, 3, v1 // (op 13515) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 13515) 


s_nop 0
v_readfirstlane_b32 s47, v24 // (op 13515) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 13284) 
v_mov_b64 v[22:23], s[0:1] // (op 13284) convert


v_mov_b64 v[24:25], v[14:15] // (op 13284) 
s_add_u32 s0, 32768, s47 // (op 13284) 
s_mov_b32 m0, s0 // (op 13284) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 13284) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 13284) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 13284) most significant half

s_add_u32 m0, m0, 4096 // (op 13284) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 13284) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 13284) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 13284) most significant half

s_add_u32 m0, m0, 4096 // (op 13284) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 13284) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 13284) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 13284) most significant half

s_add_u32 m0, m0, 4096 // (op 13284) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 13284) Load value direct to lds






v_mov_b32 v22, v18 // (op 74) reg expression

buffer_load_dwordx2 v[24:25], v22, s[68:71], 0 offen // (op 74) Load value



s_lshl_b32 s47, s38, 6 // (op 722) 
v_add_u32 v22, v18, s47 // (op 722) 


buffer_load_dwordx2 v[26:27], v22, s[68:71], 0 offen // (op 722) Load value




v_mov_b32 v18, v20 // (op 70) reg expression

buffer_load_dwordx2 v[22:23], v18, s[64:67], 0 offen // (op 70) Load value


v_and_b32 v18, -8, v1 // (op 13683) 
v_sub_u32 v20, v1, v18 // (op 13683) 

v_lshlrev_b32 v18, 4, v20 // (op 13683) 

v_lshrrev_b32 v20, 3, v1 // (op 13683) 
v_lshl_add_u32 v28, v20, 7, v18 // (op 13683) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 13683) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 5 // (op 13289) 
v_mov_b64 v[28:29], s[0:1] // (op 13289) convert


v_mov_b64 v[30:31], v[10:11] // (op 13289) 
s_add_u32 s0, 49152, s47 // (op 13289) 
s_mov_b32 m0, s0 // (op 13289) reg expression
s_mov_b32 s1, 128 // (op 13289) call()
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 13289) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 13289) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 13289) most significant half

s_add_u32 m0, m0, 4096 // (op 13289) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 13289) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 13289) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 13289) most significant half

s_add_u32 m0, m0, 4096 // (op 13289) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 13289) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 13289) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 13289) most significant half

s_add_u32 m0, m0, 4096 // (op 13289) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 13289) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 13289) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 13289) most significant half

s_add_u32 m0, m0, 4096 // (op 13289) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 13289) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 13289) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 13289) most significant half

s_add_u32 m0, m0, 4096 // (op 13289) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 13289) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 13289) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 13289) most significant half

s_add_u32 m0, m0, 4096 // (op 13289) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 13289) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 13289) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 13289) most significant half

s_add_u32 m0, m0, 4096 // (op 13289) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 13289) Load value direct to lds






v_and_b32 v18, -8, v1 // (op 13639) 
v_sub_u32 v20, v1, v18 // (op 13639) 

v_lshlrev_b32 v18, 4, v20 // (op 13639) 

v_lshrrev_b32 v20, 3, v1 // (op 13639) 
v_lshl_add_u32 v10, v20, 7, v18 // (op 13639) 


s_nop 0
v_readfirstlane_b32 s47, v10 // (op 13639) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 13291) 
v_mov_b64 v[10:11], s[0:1] // (op 13291) convert


v_mov_b64 v[28:29], v[14:15] // (op 13291) 
s_add_u32 s0, 81920, s47 // (op 13291) 
s_mov_b32 m0, s0 // (op 13291) reg expression
s_mov_b32 s1, 128 // (op 13291) call()
buffer_load_dwordx4 v28, s[48:51], s1 offen offset: 0 lds // (op 13291) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v10 // (op 13291) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 13291) most significant half

s_add_u32 m0, m0, 4096 // (op 13291) 
buffer_load_dwordx4 v28, s[48:51], s1 offen offset: 0 lds // (op 13291) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v10 // (op 13291) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 13291) most significant half

s_add_u32 m0, m0, 4096 // (op 13291) 
buffer_load_dwordx4 v28, s[48:51], s1 offen offset: 0 lds // (op 13291) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v10 // (op 13291) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 13291) most significant half

s_add_u32 m0, m0, 4096 // (op 13291) 
buffer_load_dwordx4 v28, s[48:51], s1 offen offset: 0 lds // (op 13291) Load value direct to lds






s_waitcnt vmcnt(19)
s_barrier  // (op 14068) 



ds_read_b128 v[28:31], v4 // (op 12867) Load local data 
ds_read_b128 v[32:35], v4 offset:64 // (op 12867) Load local data 



ds_read_b128 v[36:39], v4 offset:2048 // (op 12872) Load local data 
ds_read_b128 v[40:43], v4 offset:2112 // (op 12872) Load local data 



ds_read_b128 v[44:47], v4 offset:4096 // (op 12877) Load local data 
ds_read_b128 v[48:51], v4 offset:4160 // (op 12877) Load local data 



ds_read_b128 v[52:55], v4 offset:6144 // (op 12882) Load local data 
ds_read_b128 v[56:59], v4 offset:6208 // (op 12882) Load local data 



ds_read_b128 v[60:63], v4 offset:8192 // (op 12887) Load local data 
ds_read_b128 v[64:67], v4 offset:8256 // (op 12887) Load local data 



ds_read_b128 v[68:71], v4 offset:10240 // (op 12892) Load local data 
ds_read_b128 v[72:75], v4 offset:10304 // (op 12892) Load local data 



ds_read_b128 v[76:79], v4 offset:12288 // (op 12897) Load local data 
ds_read_b128 v[80:83], v4 offset:12352 // (op 12897) Load local data 



ds_read_b128 v[84:87], v4 offset:14336 // (op 12902) Load local data 
ds_read_b128 v[88:91], v4 offset:14400 // (op 12902) Load local data 
s_waitcnt vmcnt(15)
s_barrier  // (op 14051) 



ds_read_b128 v[92:95], v3 offset:32768 // (op 12847) Load local data 
ds_read_b128 v[96:99], v3 offset:32832 // (op 12847) Load local data 



ds_read_b128 v[100:103], v3 offset:34816 // (op 12852) Load local data 
ds_read_b128 v[104:107], v3 offset:34880 // (op 12852) Load local data 



ds_read_b128 v[108:111], v3 offset:36864 // (op 12857) Load local data 
ds_read_b128 v[112:115], v3 offset:36928 // (op 12857) Load local data 



ds_read_b128 v[116:119], v3 offset:38912 // (op 12862) Load local data 
ds_read_b128 v[120:123], v3 offset:38976 // (op 12862) Load local data 
v_and_b32 v18, -8, v1 // (op 13487) 
v_sub_u32 v20, v1, v18 // (op 13487) 

v_lshlrev_b32 v18, 4, v20 // (op 13487) 

v_lshrrev_b32 v20, 3, v1 // (op 13487) 
v_lshl_add_u32 v10, v20, 7, v18 // (op 13487) 


s_nop 0
v_readfirstlane_b32 s47, v10 // (op 13487) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 13544) 
v_sub_u32 v20, v1, v18 // (op 13544) 

v_lshlrev_b32 v18, 4, v20 // (op 13544) 

v_lshrrev_b32 v20, 3, v1 // (op 13544) 
v_lshl_add_u32 v10, v20, 7, v18 // (op 13544) 


s_nop 0
v_readfirstlane_b32 s0, v10 // (op 13544) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 13624) 
v_sub_u32 v20, v1, v18 // (op 13624) 

v_lshlrev_b32 v18, 4, v20 // (op 13624) 

v_lshrrev_b32 v20, 3, v1 // (op 13624) 
v_lshl_add_u32 v10, v20, 7, v18 // (op 13624) 


s_nop 0
v_readfirstlane_b32 s1, v10 // (op 13624) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 13668) 
v_sub_u32 v20, v1, v18 // (op 13668) 

v_lshlrev_b32 v18, 4, v20 // (op 13668) 

v_lshrrev_b32 v20, 3, v1 // (op 13668) 
v_lshl_add_u32 v10, v20, 7, v18 // (op 13668) 


s_nop 0
v_readfirstlane_b32 s18, v10 // (op 13668) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s19, 0 // (op 2814) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 2813) 
s_mov_b32 s24, s22 // (op 2813) convert

s_and_b32 s25, s24, 4294967294 // (op 2813) 

s_cmp_lt_u32 s19, s25 // (op 2813) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopBottom_KLoop_2813 // (op 2813) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopBottom_KLoop_2813 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopTop_KLoop_2813:
s_add_u32 s19, s19, 2 // (op 2815) 
s_waitcnt vmcnt(14)
v_mov_b32 v10, v24 // (op 13184) reg expression
v_mov_b32 v11, v25 // (op 13184) reg expression
s_waitcnt lgkmcnt(8)
s_barrier  // (op 14071) 
s_waitcnt vmcnt(13)
v_mov_b32 v14, v26 // (op 13197) reg expression

v_mov_b32 v15, v27 // (op 13197) reg expression
s_lshl_b64 s[22:23], s[30:31], 5 // (op 13252) 
v_mov_b64 v[124:125], s[22:23] // (op 13252) convert


v_mov_b64 v[126:127], v[12:13] // (op 13252) 
v_mov_b64 v[128:129], v[126:127] // (op 13252) 
s_mov_b32 m0, s0 // (op 13252) reg expression
s_mov_b32 s22, 256 // (op 13252) call()
buffer_load_dwordx4 v128, s[60:63], s22 offen offset: 0 lds // (op 13252) Load value direct to lds
v_add_co_u32 v128, s[24:25], v128, v124 // (op 13252) least significant half
v_addc_co_u32 v129, s[24:25], 0, v129, s[24:25] // (op 13252) most significant half

s_add_u32 m0, m0, 4096 // (op 13252) 
buffer_load_dwordx4 v128, s[60:63], s22 offen offset: 0 lds // (op 13252) Load value direct to lds
v_add_co_u32 v128, s[24:25], v128, v124 // (op 13252) least significant half
v_addc_co_u32 v129, s[24:25], 0, v129, s[24:25] // (op 13252) most significant half

s_add_u32 m0, m0, 4096 // (op 13252) 
buffer_load_dwordx4 v128, s[60:63], s22 offen offset: 0 lds // (op 13252) Load value direct to lds
v_add_co_u32 v128, s[24:25], v128, v124 // (op 13252) least significant half
v_addc_co_u32 v129, s[24:25], 0, v129, s[24:25] // (op 13252) most significant half

s_add_u32 m0, m0, 4096 // (op 13252) 
s_waitcnt vmcnt(15)
v_mov_b32 v130, v22 // (op 13210) reg expression
v_mov_b32 v131, v23 // (op 13210) reg expression
buffer_load_dwordx4 v128, s[60:63], s22 offen offset: 0 lds // (op 13252) Load value direct to lds
v_add_co_u32 v128, s[24:25], v128, v124 // (op 13252) least significant half
v_addc_co_u32 v129, s[24:25], 0, v129, s[24:25] // (op 13252) most significant half

s_add_u32 m0, m0, 4096 // (op 13252) 
buffer_load_dwordx4 v128, s[60:63], s22 offen offset: 0 lds // (op 13252) Load value direct to lds
v_add_co_u32 v128, s[24:25], v128, v124 // (op 13252) least significant half
v_addc_co_u32 v129, s[24:25], 0, v129, s[24:25] // (op 13252) most significant half

s_add_u32 m0, m0, 4096 // (op 13252) 
buffer_load_dwordx4 v128, s[60:63], s22 offen offset: 0 lds // (op 13252) Load value direct to lds
v_add_co_u32 v128, s[24:25], v128, v124 // (op 13252) least significant half
v_addc_co_u32 v129, s[24:25], 0, v129, s[24:25] // (op 13252) most significant half

s_add_u32 m0, m0, 4096 // (op 13252) 
v_bfe_u32 v132, v130, 0, 8 // (op 12144) 
v_bfe_u32 v133, v130, 8, 8 // (op 12144) 
v_bfe_u32 v134, v130, 16, 8 // (op 12144) 
v_bfe_u32 v135, v130, 24, 8 // (op 12144) 
v_permlane16_swap_b32 v132, v133 // (op 12144) 
s_nop 0
v_permlane16_swap_b32 v134, v135 // (op 12144) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v132, v134 // (op 12144) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v133, v135 // (op 12144) 
buffer_load_dwordx4 v128, s[60:63], s22 offen offset: 0 lds // (op 13252) Load value direct to lds
v_add_co_u32 v128, s[24:25], v128, v124 // (op 13252) least significant half
v_addc_co_u32 v129, s[24:25], 0, v129, s[24:25] // (op 13252) most significant half

s_add_u32 m0, m0, 4096 // (op 13252) 
buffer_load_dwordx4 v128, s[60:63], s22 offen offset: 0 lds // (op 13252) Load value direct to lds



v_bfe_u32 v136, v10, 0, 8 // (op 12239) 
v_bfe_u32 v137, v10, 8, 8 // (op 12239) 
v_bfe_u32 v138, v10, 16, 8 // (op 12239) 
v_bfe_u32 v139, v10, 24, 8 // (op 12239) 
v_permlane16_swap_b32 v136, v137 // (op 12239) 
s_nop 0
v_permlane16_swap_b32 v138, v139 // (op 12239) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v136, v138 // (op 12239) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v137, v139 // (op 12239) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 14054) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[92:99], v[28:35], a[0:3], v132, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 76) 
s_lshl_b64 s[22:23], s[26:27], 5 // (op 13247) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[92:99], v[36:43], a[4:7], v132, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 311) 
v_mov_b64 v[124:125], s[22:23] // (op 13247) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:99], v[44:51], a[8:11], v132, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 447) 
v_mov_b64 v[128:129], v[16:17] // (op 13247) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:99], v[52:59], a[12:15], v132, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 583) 
v_mov_b64 v[140:141], v[128:129] // (op 13247) 
s_add_u32 s22, 32768, s47 // (op 13247) 
s_mov_b32 m0, s22 // (op 13247) reg expression
s_mov_b32 s23, 256 // (op 13247) call()
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 13247) Load value direct to lds
v_add_co_u32 v140, s[24:25], v140, v124 // (op 13247) least significant half
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 13247) most significant half

s_add_u32 m0, m0, 4096 // (op 13247) 
v_bfe_u32 v142, v14, 0, 8 // (op 12271) 
v_bfe_u32 v143, v14, 8, 8 // (op 12271) 
v_bfe_u32 v144, v14, 16, 8 // (op 12271) 
v_bfe_u32 v145, v14, 24, 8 // (op 12271) 
v_permlane16_swap_b32 v142, v143 // (op 12271) 
s_nop 0
v_permlane16_swap_b32 v144, v145 // (op 12271) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v142, v144 // (op 12271) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v143, v145 // (op 12271) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 13247) Load value direct to lds
v_add_co_u32 v140, s[24:25], v140, v124 // (op 13247) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:99], v[60:67], a[16:19], v132, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 719) 
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 13247) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:99], v[68:75], a[20:23], v132, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 855) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:99], v[76:83], a[24:27], v132, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 991) 
s_add_u32 m0, m0, 4096 // (op 13247) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:99], v[84:91], a[28:31], v132, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1127) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 13247) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[100:107], v[28:35], a[32:35], v133, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1895) 
v_add_co_u32 v140, s[24:25], v140, v124 // (op 13247) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[100:107], v[36:43], a[36:39], v133, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1857) 
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 13247) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[100:107], v[44:51], a[40:43], v133, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1819) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[100:107], v[52:59], a[44:47], v133, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1781) 
s_add_u32 m0, m0, 4096 // (op 13247) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[100:107], v[60:67], a[48:51], v133, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1743) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[100:107], v[68:75], a[52:55], v133, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1705) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 13247) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[100:107], v[76:83], a[56:59], v133, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1667) 
v_mov_b32 v18, v19 // (op 13192) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[100:107], v[84:91], a[60:63], v133, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1629) 
v_add_u32 v20, 8, v18 // (op 13192) 

s_waitcnt vmcnt(16)
s_barrier  // (op 14091) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[108:115], v[28:35], a[64:67], v134, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3113) 
ds_read_b128 v[146:149], v6 offset:49152 // (op 212) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[108:115], v[36:43], a[68:71], v134, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3075) 
ds_read_b128 v[150:153], v6 offset:49216 // (op 212) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:115], v[44:51], a[72:75], v134, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3037) 
ds_read_b128 v[154:157], v6 offset:51200 // (op 320) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:115], v[52:59], a[76:79], v134, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2999) 
ds_read_b128 v[158:161], v6 offset:51264 // (op 320) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:115], v[60:67], a[80:83], v134, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2961) 
ds_read_b128 v[162:165], v6 offset:53248 // (op 456) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[108:115], v[68:75], a[84:87], v134, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2923) 
ds_read_b128 v[166:169], v6 offset:53312 // (op 456) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[108:115], v[76:83], a[88:91], v134, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2885) 
ds_read_b128 v[170:173], v6 offset:55296 // (op 592) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[108:115], v[84:91], a[92:95], v134, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2847) 
ds_read_b128 v[174:177], v6 offset:55360 // (op 592) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[116:123], v[28:35], a[96:99], v135, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4331) 
ds_read_b128 v[178:181], v6 offset:57344 // (op 728) Load local data 
ds_read_b128 v[182:185], v6 offset:57408 // (op 728) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[116:123], v[36:43], a[100:103], v135, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4293) 
ds_read_b128 v[186:189], v6 offset:59392 // (op 864) Load local data 
ds_read_b128 v[190:193], v6 offset:59456 // (op 864) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[116:123], v[44:51], a[104:107], v135, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4255) 
buffer_load_dwordx2 v[24:25], v20, s[68:71], 0 offen // (op 13192) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[116:123], v[52:59], a[108:111], v135, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4217) 
s_lshl_b32 s22, s38, 6 // (op 13205) 

s_add_u32 s22, s22, 8 // (op 13205) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:123], v[60:67], a[112:115], v135, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4179) 
v_add_u32 v20, v18, s22 // (op 13205) 


ds_read_b128 v[194:197], v6 offset:61440 // (op 1000) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[116:123], v[68:75], a[116:119], v135, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4141) 
ds_read_b128 v[198:201], v6 offset:61504 // (op 1000) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[116:123], v[76:83], a[120:123], v135, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4103) 
buffer_load_dwordx2 v[26:27], v20, s[68:71], 0 offen // (op 13205) Load value




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[116:123], v[84:91], a[124:127], v135, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4065) 
v_add_u32 v19, 8, v19 // (op 13785) 


v_mov_b32 v18, v21 // (op 13218) 
v_add_u32 v20, 8, v18 // (op 13218) 

ds_read_b128 v[132:135], v6 offset:63488 // (op 1136) Load local data 
buffer_load_dwordx2 v[22:23], v20, s[64:67], 0 offen // (op 13218) Load value


v_add_u32 v21, 8, v21 // (op 13733) 
ds_read_b128 v[136:139], v6 offset:63552 // (op 1136) Load local data 
s_waitcnt vmcnt(15)
s_barrier  // (op 14084) 



v_add_u32 v18, 81920, v5 // (op 210) 
ds_read_b128 v[92:95], v18 // (op 210) Load local data 

v_add_u32 v18, 81984, v5 // (op 210) 
ds_read_b128 v[96:99], v18 // (op 210) Load local data 




v_add_u32 v18, 83968, v5 // (op 1637) 
ds_read_b128 v[100:103], v18 // (op 1637) Load local data 

v_add_u32 v18, 84032, v5 // (op 1637) 
ds_read_b128 v[104:107], v18 // (op 1637) Load local data 




v_add_u32 v18, 86016, v5 // (op 2855) 
ds_read_b128 v[108:111], v18 // (op 2855) Load local data 

v_add_u32 v18, 86080, v5 // (op 2855) 
ds_read_b128 v[112:115], v18 // (op 2855) Load local data 




v_add_u32 v18, 88064, v5 // (op 4073) 
ds_read_b128 v[202:205], v18 // (op 4073) Load local data 

v_add_u32 v18, 88128, v5 // (op 4073) 
ds_read_b128 v[206:209], v18 // (op 4073) Load local data 

v_bfe_u32 v140, v131, 0, 8 // (op 12159) 
v_bfe_u32 v141, v131, 8, 8 // (op 12159) 
v_bfe_u32 v142, v131, 16, 8 // (op 12159) 
v_bfe_u32 v143, v131, 24, 8 // (op 12159) 
v_permlane16_swap_b32 v140, v141 // (op 12159) 
s_nop 0
v_permlane16_swap_b32 v142, v143 // (op 12159) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v140, v142 // (op 12159) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v141, v143 // (op 12159) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 14095) 


s_lshl_b64 s[22:23], s[30:31], 5 // (op 13274) 
v_mov_b64 v[124:125], s[22:23] // (op 13274) convert


v_mov_b64 v[130:131], v[126:127] // (op 13274) 
s_add_u32 s22, 49152, s18 // (op 13274) 
s_mov_b32 m0, s22 // (op 13274) reg expression
s_mov_b32 s23, 384 // (op 13274) call()
v_bfe_u32 v210, v11, 0, 8 // (op 12254) 
v_bfe_u32 v211, v11, 8, 8 // (op 12254) 
v_bfe_u32 v212, v11, 16, 8 // (op 12254) 
v_bfe_u32 v213, v11, 24, 8 // (op 12254) 
v_permlane16_swap_b32 v210, v211 // (op 12254) 
s_nop 0
v_permlane16_swap_b32 v212, v213 // (op 12254) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v210, v212 // (op 12254) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v211, v213 // (op 12254) 
buffer_load_dwordx4 v130, s[60:63], s23 offen offset: 0 lds // (op 13274) Load value direct to lds

v_add_co_u32 v130, s[24:25], v130, v124 // (op 13274) least significant half
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[92:99], v[146:153], a[0:3], v140, v210 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 217) 
v_addc_co_u32 v131, s[24:25], 0, v131, s[24:25] // (op 13274) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[92:99], v[154:161], a[4:7], v140, v211 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 312) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:99], v[162:169], a[8:11], v140, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 448) 
s_add_u32 m0, m0, 4096 // (op 13274) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:99], v[170:177], a[12:15], v140, v213 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 584) 
buffer_load_dwordx4 v130, s[60:63], s23 offen offset: 0 lds // (op 13274) Load value direct to lds
v_add_co_u32 v130, s[24:25], v130, v124 // (op 13274) least significant half
v_addc_co_u32 v131, s[24:25], 0, v131, s[24:25] // (op 13274) most significant half

s_add_u32 m0, m0, 4096 // (op 13274) 
v_bfe_u32 v214, v15, 0, 8 // (op 12286) 
v_bfe_u32 v215, v15, 8, 8 // (op 12286) 
v_bfe_u32 v216, v15, 16, 8 // (op 12286) 
v_bfe_u32 v217, v15, 24, 8 // (op 12286) 
v_permlane16_swap_b32 v214, v215 // (op 12286) 
s_nop 0
v_permlane16_swap_b32 v216, v217 // (op 12286) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v214, v216 // (op 12286) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v215, v217 // (op 12286) 
buffer_load_dwordx4 v130, s[60:63], s23 offen offset: 0 lds // (op 13274) Load value direct to lds

v_add_co_u32 v130, s[24:25], v130, v124 // (op 13274) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:99], v[178:185], a[16:19], v140, v214 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 720) 
v_addc_co_u32 v131, s[24:25], 0, v131, s[24:25] // (op 13274) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:99], v[186:193], a[20:23], v140, v215 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 856) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:99], v[194:201], a[24:27], v140, v216 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 992) 
s_add_u32 m0, m0, 4096 // (op 13274) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:99], v[132:139], a[28:31], v140, v217 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1128) 
buffer_load_dwordx4 v130, s[60:63], s23 offen offset: 0 lds // (op 13274) Load value direct to lds
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[100:107], v[146:153], a[32:35], v141, v210 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1896) 
v_add_co_u32 v130, s[24:25], v130, v124 // (op 13274) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[100:107], v[154:161], a[36:39], v141, v211 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1858) 
v_addc_co_u32 v131, s[24:25], 0, v131, s[24:25] // (op 13274) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[100:107], v[162:169], a[40:43], v141, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1820) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[100:107], v[170:177], a[44:47], v141, v213 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1782) 
s_add_u32 m0, m0, 4096 // (op 13274) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[100:107], v[178:185], a[48:51], v141, v214 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1744) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[100:107], v[186:193], a[52:55], v141, v215 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1706) 
buffer_load_dwordx4 v130, s[60:63], s23 offen offset: 0 lds // (op 13274) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[100:107], v[194:201], a[56:59], v141, v216 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1668) 
v_add_co_u32 v130, s[24:25], v130, v124 // (op 13274) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[100:107], v[132:139], a[60:63], v141, v217 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1630) 
v_addc_co_u32 v131, s[24:25], 0, v131, s[24:25] // (op 13274) most significant half

s_add_u32 m0, m0, 4096 // (op 13274) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[108:115], v[146:153], a[64:67], v142, v210 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3114) 
buffer_load_dwordx4 v130, s[60:63], s23 offen offset: 0 lds // (op 13274) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[108:115], v[154:161], a[68:71], v142, v211 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3076) 
v_add_co_u32 v130, s[24:25], v130, v124 // (op 13274) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:115], v[162:169], a[72:75], v142, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3038) 
v_addc_co_u32 v131, s[24:25], 0, v131, s[24:25] // (op 13274) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:115], v[170:177], a[76:79], v142, v213 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3000) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:115], v[178:185], a[80:83], v142, v214 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2962) 
s_add_u32 m0, m0, 4096 // (op 13274) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[108:115], v[186:193], a[84:87], v142, v215 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2924) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[108:115], v[194:201], a[88:91], v142, v216 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2886) 
s_waitcnt vmcnt(13)
s_barrier  // (op 14077) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[108:115], v[132:139], a[92:95], v142, v217 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2848) 
ds_read_b128 v[28:31], v4 // (op 65) Load local data 
ds_read_b128 v[32:35], v4 offset:64 // (op 65) Load local data 



ds_read_b128 v[36:39], v4 offset:2048 // (op 308) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[202:209], v[146:153], a[96:99], v143, v210 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4332) 
ds_read_b128 v[40:43], v4 offset:2112 // (op 308) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[202:209], v[154:161], a[100:103], v143, v211 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4294) 
ds_read_b128 v[44:47], v4 offset:4096 // (op 444) Load local data 

ds_read_b128 v[48:51], v4 offset:4160 // (op 444) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[202:209], v[162:169], a[104:107], v143, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4256) 
ds_read_b128 v[52:55], v4 offset:6144 // (op 580) Load local data 

ds_read_b128 v[56:59], v4 offset:6208 // (op 580) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[202:209], v[170:177], a[108:111], v143, v213 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4218) 
ds_read_b128 v[60:63], v4 offset:8192 // (op 716) Load local data 


ds_read_b128 v[64:67], v4 offset:8256 // (op 716) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[202:209], v[178:185], a[112:115], v143, v214 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4180) 
ds_read_b128 v[68:71], v4 offset:10240 // (op 852) Load local data 

ds_read_b128 v[72:75], v4 offset:10304 // (op 852) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[202:209], v[186:193], a[116:119], v143, v215 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4142) 
buffer_load_dwordx4 v130, s[60:63], s23 offen offset: 0 lds // (op 13274) Load value direct to lds

v_add_co_u32 v130, s[24:25], v130, v124 // (op 13274) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[202:209], v[194:201], a[120:123], v143, v216 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4104) 
v_addc_co_u32 v131, s[24:25], 0, v131, s[24:25] // (op 13274) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[202:209], v[132:139], a[124:127], v143, v217 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4066) 
s_add_u32 m0, m0, 4096 // (op 13274) 




ds_read_b128 v[76:79], v4 offset:12288 // (op 988) Load local data 
ds_read_b128 v[80:83], v4 offset:12352 // (op 988) Load local data 
buffer_load_dwordx4 v130, s[60:63], s23 offen offset: 0 lds // (op 13274) Load value direct to lds








ds_read_b128 v[84:87], v4 offset:14336 // (op 1124) Load local data 
s_barrier  // (op 14088) 
v_add_co_u32 v12, vcc, 256, v12 // (op 13436) least significant half

v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 13436) most significant half
s_lshl_b64 s[22:23], s[26:27], 5 // (op 13269) 
v_mov_b64 v[10:11], s[22:23] // (op 13269) convert


v_mov_b64 v[14:15], v[128:129] // (op 13269) 
s_add_u32 s22, 81920, s1 // (op 13269) 
s_mov_b32 m0, s22 // (op 13269) reg expression
s_mov_b32 s23, 384 // (op 13269) call()
buffer_load_dwordx4 v14, s[48:51], s23 offen offset: 0 lds // (op 13269) Load value direct to lds
v_add_co_u32 v14, s[24:25], v14, v10 // (op 13269) least significant half
v_addc_co_u32 v15, s[24:25], 0, v15, s[24:25] // (op 13269) most significant half

s_add_u32 m0, m0, 4096 // (op 13269) 
ds_read_b128 v[88:91], v4 offset:14400 // (op 1124) Load local data 
buffer_load_dwordx4 v14, s[48:51], s23 offen offset: 0 lds // (op 13269) Load value direct to lds
v_add_co_u32 v14, s[24:25], v14, v10 // (op 13269) least significant half
v_addc_co_u32 v15, s[24:25], 0, v15, s[24:25] // (op 13269) most significant half

s_add_u32 m0, m0, 4096 // (op 13269) 
s_waitcnt vmcnt(13)
s_barrier  // (op 14060) 
buffer_load_dwordx4 v14, s[48:51], s23 offen offset: 0 lds // (op 13269) Load value direct to lds



ds_read_b128 v[92:95], v3 offset:32768 // (op 60) Load local data 
ds_read_b128 v[96:99], v3 offset:32832 // (op 60) Load local data 
v_add_co_u32 v14, s[24:25], v14, v10 // (op 13269) least significant half



ds_read_b128 v[100:103], v3 offset:34816 // (op 1625) Load local data 
v_addc_co_u32 v15, s[24:25], 0, v15, s[24:25] // (op 13269) most significant half

s_add_u32 m0, m0, 4096 // (op 13269) 
buffer_load_dwordx4 v14, s[48:51], s23 offen offset: 0 lds // (op 13269) Load value direct to lds




ds_read_b128 v[104:107], v3 offset:34880 // (op 1625) Load local data 




ds_read_b128 v[108:111], v3 offset:36864 // (op 2843) Load local data 
v_add_co_u32 v16, vcc, 256, v16 // (op 13372) least significant half
ds_read_b128 v[112:115], v3 offset:36928 // (op 2843) Load local data 
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 13372) most significant half



ds_read_b128 v[116:119], v3 offset:38912 // (op 4061) Load local data 
ds_read_b128 v[120:123], v3 offset:38976 // (op 4061) Load local data 
s_ashr_i64 s[22:23], s[42:43], 7 // (op 2813) 
s_mov_b32 s24, s22 // (op 2813) convert

s_and_b32 s25, s24, 4294967294 // (op 2813) 

s_cmp_lt_u32 s19, s25 // (op 2813) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopTop_KLoop_2813 // (op 2813) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopTop_KLoop_2813 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopBottom_KLoop_2813:















s_barrier  // (op 14137) 
s_mul_hi_i32 s47, s16, s10 // (op 13304) 
v_lshrrev_b32 v10, 3, v1 // (op 13304) 
s_add_i32 s0, s47, s16 // (op 13304) 

v_and_b32 v11, -8, v1 // (op 13304) 
s_ashr_i32 s1, s0, 31 // (op 13304) 
v_sub_u32 v12, v1, v11 // (op 13304) 

s_cmp_eq_i32 s10, 0 // (op 13304) 
s_mov_b32 s47, scc // (op 13304) 
v_lshlrev_b32 v11, 4, v12 // (op 13304) 

s_cmp_eq_i32 s47, 0 // (op 13304) 
s_cselect_b32 s18, 0, -1 // (op 13304) 

s_and_b32 s19, s13, 31 // (op 13304) 
s_lshl_b32 s47, 1, s19 // (op 13304) 
s_add_i32 s22, s47, s18 // (op 13304) 


s_and_b32 s18, s1, s22 // (op 13304) 


s_add_i32 s1, s0, s18 // (op 13304) 


s_ashr_i32 s0, s1, s19 // (op 13304) 


s_lshl_b32 s1, s13, 24 // (op 13304) 
s_ashr_i32 s47, s1, 31 // (op 13304) 

s_xor_b32 s1, s0, s47 // (op 13304) 

s_sub_i32 s0, s1, s47 // (op 13304) 


s_mul_hi_i32 s1, s0, s14 // (op 13304) 
s_add_i32 s47, s1, s0 // (op 13304) 

s_ashr_i32 s1, s47, 31 // (op 13304) 
s_cmp_eq_i32 s14, 0 // (op 13304) 
s_mov_b32 s18, scc // (op 13304) 
s_cmp_eq_i32 s18, 0 // (op 13304) 
s_cselect_b32 s19, 0, -1 // (op 13304) 

s_and_b32 s18, s15, 31 // (op 13304) 
s_lshl_b32 s22, 1, s18 // (op 13304) 
s_add_i32 s23, s22, s19 // (op 13304) 


s_and_b32 s19, s1, s23 // (op 13304) 


s_add_i32 s1, s47, s19 // (op 13304) 


s_ashr_i32 s19, s1, s18 // (op 13304) 


s_lshl_b32 s1, s15, 24 // (op 13304) 
s_ashr_i32 s18, s1, 31 // (op 13304) 

s_xor_b32 s1, s19, s18 // (op 13304) 

s_sub_i32 s19, s1, s18 // (op 13304) 


s_mul_i32 s1, s0, s11 // (op 13304) 

s_sub_u32 s0, s16, s1 // (op 13304) 

s_mul_i32 s1, s19, s11 // (op 13304) 
s_add_u32 s1, s1, s0 // (op 13304) 


s_mul_hi_i32 s0, s16, s9 // (op 13304) 
s_add_i32 s47, s0, s16 // (op 13304) 

s_ashr_i32 s0, s47, 31 // (op 13304) 
s_cmp_eq_i32 s9, 0 // (op 13304) 
s_mov_b32 s18, scc // (op 13304) 
s_cmp_eq_i32 s18, 0 // (op 13304) 
s_cselect_b32 s19, 0, -1 // (op 13304) 

s_and_b32 s18, s45, 31 // (op 13304) 
s_lshl_b32 s22, 1, s18 // (op 13304) 
s_add_i32 s23, s22, s19 // (op 13304) 


s_and_b32 s19, s0, s23 // (op 13304) 


s_add_i32 s0, s47, s19 // (op 13304) 


s_ashr_i32 s19, s0, s18 // (op 13304) 


s_lshl_b32 s0, s45, 24 // (op 13304) 
s_ashr_i32 s18, s0, 31 // (op 13304) 

s_xor_b32 s0, s19, s18 // (op 13304) 

s_sub_i32 s19, s0, s18 // (op 13304) 


s_add_u32 s22, s6, 256 // (op 13304) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13304) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13304) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13304) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13304) 

s_mul_i32 s0, s11, s22 // (op 13304) 

s_mul_i32 s18, s8, s0 // (op 13304) 

s_mul_i32 s0, s19, s18 // (op 13304) 

s_sub_u32 s18, s16, s0 // (op 13304) 

s_mov_b32 s0, s18 // (op 13304) 
s_mul_hi_i32 s47, s0, s12 // (op 13304) 
s_add_i32 s22, s47, s0 // (op 13304) 


s_ashr_i32 s0, s22, 31 // (op 13304) 
s_cmp_eq_i32 s12, 0 // (op 13304) 
s_mov_b32 s47, scc // (op 13304) 
s_cmp_eq_i32 s47, 0 // (op 13304) 
s_cselect_b32 s23, 0, -1 // (op 13304) 

s_and_b32 s47, s44, 31 // (op 13304) 
s_lshl_b32 s24, 1, s47 // (op 13304) 
s_add_i32 s25, s24, s23 // (op 13304) 


s_and_b32 s23, s0, s25 // (op 13304) 


s_add_i32 s0, s22, s23 // (op 13304) 


s_ashr_i32 s22, s0, s47 // (op 13304) 


s_lshl_b32 s0, s44, 24 // (op 13304) 
s_ashr_i32 s47, s0, 31 // (op 13304) 

s_xor_b32 s0, s22, s47 // (op 13304) 

s_sub_i32 s22, s0, s47 // (op 13304) 


s_mul_i32 s0, s46, s11 // (op 13304) 
s_add_u32 s24, s20, 128 // (op 13304) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13304) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 13304) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 13304) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 13304) 

s_sub_i32 s23, s24, s0 // (op 13304) 

s_mul_i32 s47, s22, s23 // (op 13304) 


s_sub_u32 s22, s18, s47 // (op 13304) 


s_add_u32 s18, s22, s0 // (op 13304) 


s_cmp_eq_i32 s19, 0 // (op 13304) 
s_mov_b32 s0, scc // (op 13304) 

s_cmp_eq_i32 s0, 0 // (op 13304) 
s_cselect_b32 s19, s18, s1 // (op 13304) 



v_lshl_add_u32 v12, s19, 7, v10 // (op 13304) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 13304) 
s_lshr_b32 s47, s0, 1 // (op 13304) 

v_lshl_add_u32 v10, s47, 8, v11 // (op 13304) 


v_mov_b32 v21, 0 // (op 13304) convert
v_mov_b32 v20, v10 // (op 13304) convert

v_mul_lo_u32 v11, v12, s27 // (op 13304) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 13304) most significant: high of low * low
v_mul_lo_u32 v10, v12, s26 // (op 13304) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 13304) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v20 // (op 13304) least significant half
v_addc_co_u32 v11, s[0:1], v11, v21, s[0:1] // (op 13304) most significant half



s_mul_i32 s1, s26, s21 // (op 13306) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 13306) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 13306) least significant
s_mul_hi_u32 s18, s26, s20 // (op 13306) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13306) most significant: sum
s_add_u32 s1, s1, s18 // (op 13306) most significant: sum


s_mov_b32 s48, s28 // (op 13306) reg expression
s_mov_b32 s49, s29 // (op 13306) reg expression
s_mov_b32 s50, s0 // (op 13306) reg expression
s_mov_b32 s51, 131072 // (op 13306) call()

s_mul_hi_i32 s47, s16, s10 // (op 13402) 
v_lshrrev_b32 v12, 3, v1 // (op 13402) 
s_add_i32 s0, s47, s16 // (op 13402) 

v_and_b32 v13, -8, v1 // (op 13402) 
s_ashr_i32 s1, s0, 31 // (op 13402) 
v_sub_u32 v14, v1, v13 // (op 13402) 

s_cmp_eq_i32 s10, 0 // (op 13402) 
s_mov_b32 s47, scc // (op 13402) 
v_lshlrev_b32 v13, 4, v14 // (op 13402) 

s_cmp_eq_i32 s47, 0 // (op 13402) 
s_cselect_b32 s18, 0, -1 // (op 13402) 

s_and_b32 s19, s13, 31 // (op 13402) 
s_lshl_b32 s47, 1, s19 // (op 13402) 
s_add_i32 s22, s47, s18 // (op 13402) 


s_and_b32 s18, s1, s22 // (op 13402) 


s_add_i32 s1, s0, s18 // (op 13402) 


s_ashr_i32 s0, s1, s19 // (op 13402) 


s_lshl_b32 s1, s13, 24 // (op 13402) 
s_ashr_i32 s47, s1, 31 // (op 13402) 

s_xor_b32 s1, s0, s47 // (op 13402) 

s_sub_i32 s0, s1, s47 // (op 13402) 


s_mul_hi_i32 s1, s0, s14 // (op 13402) 
s_add_i32 s47, s1, s0 // (op 13402) 

s_ashr_i32 s1, s47, 31 // (op 13402) 
s_cmp_eq_i32 s14, 0 // (op 13402) 
s_mov_b32 s18, scc // (op 13402) 
s_cmp_eq_i32 s18, 0 // (op 13402) 
s_cselect_b32 s19, 0, -1 // (op 13402) 

s_and_b32 s18, s15, 31 // (op 13402) 
s_lshl_b32 s22, 1, s18 // (op 13402) 
s_add_i32 s23, s22, s19 // (op 13402) 


s_and_b32 s19, s1, s23 // (op 13402) 


s_add_i32 s1, s47, s19 // (op 13402) 


s_ashr_i32 s19, s1, s18 // (op 13402) 


s_lshl_b32 s1, s15, 24 // (op 13402) 
s_ashr_i32 s18, s1, 31 // (op 13402) 

s_xor_b32 s1, s19, s18 // (op 13402) 

s_sub_i32 s19, s1, s18 // (op 13402) 


s_add_u32 s22, s6, 256 // (op 13402) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13402) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13402) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13402) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13402) 

s_mul_i32 s1, s19, s22 // (op 13402) 

s_sub_i32 s47, s0, s1 // (op 13402) 


s_mul_hi_i32 s0, s16, s9 // (op 13402) 
s_add_i32 s1, s0, s16 // (op 13402) 

s_ashr_i32 s0, s1, 31 // (op 13402) 
s_cmp_eq_i32 s9, 0 // (op 13402) 
s_mov_b32 s18, scc // (op 13402) 
s_cmp_eq_i32 s18, 0 // (op 13402) 
s_cselect_b32 s19, 0, -1 // (op 13402) 

s_and_b32 s18, s45, 31 // (op 13402) 
s_lshl_b32 s24, 1, s18 // (op 13402) 
s_add_i32 s25, s24, s19 // (op 13402) 


s_and_b32 s19, s0, s25 // (op 13402) 


s_add_i32 s0, s1, s19 // (op 13402) 


s_ashr_i32 s1, s0, s18 // (op 13402) 


s_lshl_b32 s0, s45, 24 // (op 13402) 
s_ashr_i32 s18, s0, 31 // (op 13402) 

s_xor_b32 s0, s1, s18 // (op 13402) 

s_sub_i32 s1, s0, s18 // (op 13402) 


s_mul_i32 s0, s11, s22 // (op 13402) 

s_mul_i32 s18, s8, s0 // (op 13402) 

s_mul_i32 s0, s1, s18 // (op 13402) 

s_sub_u32 s18, s16, s0 // (op 13402) 

s_mul_hi_i32 s0, s18, s12 // (op 13402) 
s_add_i32 s19, s0, s18 // (op 13402) 


s_ashr_i32 s0, s19, 31 // (op 13402) 
s_cmp_eq_i32 s12, 0 // (op 13402) 
s_mov_b32 s18, scc // (op 13402) 
s_cmp_eq_i32 s18, 0 // (op 13402) 
s_cselect_b32 s22, 0, -1 // (op 13402) 

s_and_b32 s18, s44, 31 // (op 13402) 
s_lshl_b32 s23, 1, s18 // (op 13402) 
s_add_i32 s24, s23, s22 // (op 13402) 


s_and_b32 s25, s0, s24 // (op 13402) 


s_add_i32 s0, s19, s25 // (op 13402) 


s_ashr_i32 s19, s0, s18 // (op 13402) 


s_lshl_b32 s0, s44, 24 // (op 13402) 
s_ashr_i32 s18, s0, 31 // (op 13402) 

s_xor_b32 s0, s19, s18 // (op 13402) 

s_sub_i32 s19, s0, s18 // (op 13402) 


s_cmp_eq_i32 s1, 0 // (op 13402) 
s_mov_b32 s0, scc // (op 13402) 

s_cmp_eq_i32 s0, 0 // (op 13402) 
s_cselect_b32 s1, s19, s47 // (op 13402) 



v_lshl_add_u32 v14, s1, 8, v12 // (op 13402) 


v_mul_lo_u32 v21, v14, s31 // (op 13402) most significant: low of low * high
v_mul_hi_u32 v12, v14, s30 // (op 13402) most significant: high of low * low
v_mul_lo_u32 v20, v14, s30 // (op 13402) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 13402) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 13402) 
s_lshr_b32 s47, s0, 1 // (op 13402) 

v_lshl_add_u32 v12, s47, 8, v13 // (op 13402) 


v_mov_b32 v19, 0 // (op 13402) convert
v_mov_b32 v18, v12 // (op 13402) convert

v_add_co_u32 v12, s[0:1], v18, v20 // (op 13402) least significant half
v_addc_co_u32 v13, s[0:1], v19, v21, s[0:1] // (op 13402) most significant half



s_mul_i32 s1, s30, s7 // (op 13404) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 13404) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 13404) least significant
s_mul_hi_u32 s18, s30, s6 // (op 13404) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13404) most significant: sum
s_add_u32 s1, s1, s18 // (op 13404) most significant: sum


s_mov_b32 s60, s32 // (op 13404) reg expression
s_mov_b32 s61, s33 // (op 13404) reg expression
s_mov_b32 s62, s0 // (op 13404) reg expression
s_mov_b32 s63, 131072 // (op 13404) call()

v_and_b32 v14, -8, v1 // (op 13500) 
v_sub_u32 v15, v1, v14 // (op 13500) 

v_lshlrev_b32 v14, 4, v15 // (op 13500) 

v_lshrrev_b32 v15, 3, v1 // (op 13500) 
v_lshl_add_u32 v16, v15, 7, v14 // (op 13500) 


s_nop 0
v_readfirstlane_b32 s47, v16 // (op 13500) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 13557) 
v_sub_u32 v15, v1, v14 // (op 13557) 

v_lshlrev_b32 v14, 4, v15 // (op 13557) 

v_lshrrev_b32 v15, 3, v1 // (op 13557) 
v_lshl_add_u32 v16, v15, 7, v14 // (op 13557) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 13557) Wait state hazard: VALU Write Readlane Hazard

s_mul_hi_i32 s1, s16, s10 // (op 13822) 
v_and_b32 v14, 63, v1 // (op 13822) 
s_add_i32 s18, s1, s16 // (op 13822) 

v_lshrrev_b32 v15, 4, v14 // (op 13822) 
s_ashr_i32 s1, s18, 31 // (op 13822) 
v_and_b32 v16, 3, v15 // (op 13822) 

s_cmp_eq_i32 s10, 0 // (op 13822) 
s_mov_b32 s19, scc // (op 13822) 
v_and_b32 v15, 15, v1 // (op 13822) 
s_cmp_eq_i32 s19, 0 // (op 13822) 
s_cselect_b32 s22, 0, -1 // (op 13822) 

v_lshl_add_u32 v17, v16, 4, v15 // (op 13822) 


s_and_b32 s19, s13, 31 // (op 13822) 
v_lshrrev_b32 v15, 7, v1 // (op 13822) 
s_lshl_b32 s23, 1, s19 // (op 13822) 
v_lshl_add_u32 v16, v15, 6, v17 // (op 13822) 


s_add_i32 s24, s23, s22 // (op 13822) 


v_lshrrev_b32 v15, 6, v14 // (op 13822) 

s_and_b32 s25, s1, s24 // (op 13822) 


v_lshlrev_b32 v14, 2, v15 // (op 13822) 

s_add_i32 s1, s18, s25 // (op 13822) 


s_ashr_i32 s18, s1, s19 // (op 13822) 


s_lshl_b32 s1, s13, 24 // (op 13822) 
s_ashr_i32 s19, s1, 31 // (op 13822) 

s_xor_b32 s1, s18, s19 // (op 13822) 

s_sub_i32 s18, s1, s19 // (op 13822) 


s_mul_hi_i32 s1, s18, s14 // (op 13822) 
s_add_i32 s19, s1, s18 // (op 13822) 

s_ashr_i32 s1, s19, 31 // (op 13822) 
s_cmp_eq_i32 s14, 0 // (op 13822) 
s_mov_b32 s22, scc // (op 13822) 
s_cmp_eq_i32 s22, 0 // (op 13822) 
s_cselect_b32 s23, 0, -1 // (op 13822) 

s_and_b32 s22, s15, 31 // (op 13822) 
s_lshl_b32 s24, 1, s22 // (op 13822) 
s_add_i32 s25, s24, s23 // (op 13822) 


s_and_b32 s23, s1, s25 // (op 13822) 


s_add_i32 s1, s19, s23 // (op 13822) 


s_ashr_i32 s19, s1, s22 // (op 13822) 


s_lshl_b32 s1, s15, 24 // (op 13822) 
s_ashr_i32 s22, s1, 31 // (op 13822) 

s_xor_b32 s1, s19, s22 // (op 13822) 

s_sub_i32 s19, s1, s22 // (op 13822) 


s_mul_i32 s1, s18, s11 // (op 13822) 

s_sub_u32 s18, s16, s1 // (op 13822) 

s_mul_i32 s1, s19, s11 // (op 13822) 
s_add_u32 s1, s1, s18 // (op 13822) 


s_mul_hi_i32 s18, s16, s9 // (op 13822) 
s_add_i32 s19, s18, s16 // (op 13822) 

s_ashr_i32 s18, s19, 31 // (op 13822) 
s_cmp_eq_i32 s9, 0 // (op 13822) 
s_mov_b32 s22, scc // (op 13822) 
s_cmp_eq_i32 s22, 0 // (op 13822) 
s_cselect_b32 s23, 0, -1 // (op 13822) 

s_and_b32 s22, s45, 31 // (op 13822) 
s_lshl_b32 s24, 1, s22 // (op 13822) 
s_add_i32 s25, s24, s23 // (op 13822) 


s_and_b32 s23, s18, s25 // (op 13822) 


s_add_i32 s18, s19, s23 // (op 13822) 


s_ashr_i32 s19, s18, s22 // (op 13822) 


s_lshl_b32 s18, s45, 24 // (op 13822) 
s_ashr_i32 s22, s18, 31 // (op 13822) 

s_xor_b32 s18, s19, s22 // (op 13822) 

s_sub_i32 s19, s18, s22 // (op 13822) 


s_add_u32 s22, s6, 256 // (op 13822) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13822) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13822) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13822) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13822) 

s_mul_i32 s18, s11, s22 // (op 13822) 

s_mul_i32 s22, s8, s18 // (op 13822) 

s_mul_i32 s18, s19, s22 // (op 13822) 

s_sub_u32 s22, s16, s18 // (op 13822) 

s_mov_b32 s18, s22 // (op 13822) 
s_mul_hi_i32 s23, s18, s12 // (op 13822) 
s_add_i32 s24, s23, s18 // (op 13822) 


s_ashr_i32 s18, s24, 31 // (op 13822) 
s_cmp_eq_i32 s12, 0 // (op 13822) 
s_mov_b32 s23, scc // (op 13822) 
s_cmp_eq_i32 s23, 0 // (op 13822) 
s_cselect_b32 s25, 0, -1 // (op 13822) 

s_and_b32 s23, s44, 31 // (op 13822) 
s_lshl_b32 s64, 1, s23 // (op 13822) 
s_add_i32 s65, s64, s25 // (op 13822) 


s_and_b32 s25, s18, s65 // (op 13822) 


s_add_i32 s18, s24, s25 // (op 13822) 


s_ashr_i32 s24, s18, s23 // (op 13822) 


s_lshl_b32 s18, s44, 24 // (op 13822) 
s_ashr_i32 s23, s18, 31 // (op 13822) 

s_xor_b32 s18, s24, s23 // (op 13822) 

s_sub_i32 s24, s18, s23 // (op 13822) 


s_mul_i32 s18, s46, s11 // (op 13822) 
s_add_u32 s64, s20, 128 // (op 13822) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 13822) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 13822) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 13822) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 7 // (op 13822) 

s_sub_i32 s23, s64, s18 // (op 13822) 

s_mul_i32 s25, s24, s23 // (op 13822) 


s_sub_u32 s23, s22, s25 // (op 13822) 


s_add_u32 s22, s23, s18 // (op 13822) 


s_cmp_eq_i32 s19, 0 // (op 13822) 
s_mov_b32 s18, scc // (op 13822) 

s_cmp_eq_i32 s18, 0 // (op 13822) 
s_cselect_b32 s19, s22, s1 // (op 13822) 



v_lshl_add_u32 v15, s19, 7, v16 // (op 13822) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 13822) 
s_lshr_b32 s1, s18, 1 // (op 13822) 

v_lshl_add_u32 v16, s1, 3, v14 // (op 13822) 


v_mul_lo_u32 v14, v15, s34 // (op 13822) 
v_add_u32 v14, v14, v16 // (op 13822) 


s_mul_i32 s19, s34, s21 // (op 13824) most significant: low of low * high
s_mul_i32 s1, s35, s20 // (op 13824) most significant: low of high * low
s_mul_i32 s18, s34, s20 // (op 13824) least significant
s_mul_hi_u32 s22, s34, s20 // (op 13824) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 13824) most significant: sum
s_add_u32 s19, s19, s22 // (op 13824) most significant: sum


s_mov_b32 s64, s36 // (op 13824) reg expression
s_mov_b32 s65, s37 // (op 13824) reg expression
s_mov_b32 s66, s18 // (op 13824) reg expression
s_mov_b32 s67, 131072 // (op 13824) call()

s_mul_hi_i32 s1, s16, s10 // (op 13849) 
v_and_b32 v15, 63, v1 // (op 13849) 
s_add_i32 s18, s1, s16 // (op 13849) 

v_lshrrev_b32 v16, 4, v15 // (op 13849) 
s_ashr_i32 s1, s18, 31 // (op 13849) 
v_and_b32 v17, 3, v16 // (op 13849) 

s_cmp_eq_i32 s10, 0 // (op 13849) 
s_mov_b32 s19, scc // (op 13849) 
v_and_b32 v16, 15, v1 // (op 13849) 
s_cmp_eq_i32 s19, 0 // (op 13849) 
s_cselect_b32 s22, 0, -1 // (op 13849) 

v_lshl_add_u32 v18, v17, 4, v16 // (op 13849) 


s_and_b32 s19, s13, 31 // (op 13849) 
v_lshrrev_b32 v16, 6, v1 // (op 13849) 
s_lshl_b32 s23, 1, s19 // (op 13849) 
v_and_b32 v17, 1, v16 // (op 13849) 

s_add_i32 s24, s23, s22 // (op 13849) 


v_lshl_add_u32 v16, v17, 7, v18 // (op 13849) 


s_and_b32 s25, s1, s24 // (op 13849) 


v_lshrrev_b32 v17, 6, v15 // (op 13849) 

s_add_i32 s1, s18, s25 // (op 13849) 


v_lshlrev_b32 v15, 2, v17 // (op 13849) 

s_ashr_i32 s18, s1, s19 // (op 13849) 


s_lshl_b32 s1, s13, 24 // (op 13849) 
s_ashr_i32 s19, s1, 31 // (op 13849) 

s_xor_b32 s1, s18, s19 // (op 13849) 

s_sub_i32 s18, s1, s19 // (op 13849) 


s_mul_hi_i32 s1, s18, s14 // (op 13849) 
s_add_i32 s19, s1, s18 // (op 13849) 

s_ashr_i32 s1, s19, 31 // (op 13849) 
s_cmp_eq_i32 s14, 0 // (op 13849) 
s_mov_b32 s22, scc // (op 13849) 
s_cmp_eq_i32 s22, 0 // (op 13849) 
s_cselect_b32 s23, 0, -1 // (op 13849) 

s_and_b32 s22, s15, 31 // (op 13849) 
s_lshl_b32 s24, 1, s22 // (op 13849) 
s_add_i32 s25, s24, s23 // (op 13849) 


s_and_b32 s23, s1, s25 // (op 13849) 


s_add_i32 s1, s19, s23 // (op 13849) 


s_ashr_i32 s19, s1, s22 // (op 13849) 


s_lshl_b32 s1, s15, 24 // (op 13849) 
s_ashr_i32 s22, s1, 31 // (op 13849) 

s_xor_b32 s1, s19, s22 // (op 13849) 

s_sub_i32 s19, s1, s22 // (op 13849) 


s_add_u32 s22, s6, 256 // (op 13849) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13849) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13849) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13849) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13849) 

s_mul_i32 s1, s19, s22 // (op 13849) 

s_sub_i32 s19, s18, s1 // (op 13849) 


s_mul_hi_i32 s1, s16, s9 // (op 13849) 
s_add_i32 s18, s1, s16 // (op 13849) 

s_ashr_i32 s1, s18, 31 // (op 13849) 
s_cmp_eq_i32 s9, 0 // (op 13849) 
s_mov_b32 s24, scc // (op 13849) 
s_cmp_eq_i32 s24, 0 // (op 13849) 
s_cselect_b32 s25, 0, -1 // (op 13849) 

s_and_b32 s24, s45, 31 // (op 13849) 
s_lshl_b32 s68, 1, s24 // (op 13849) 
s_add_i32 s69, s68, s25 // (op 13849) 


s_and_b32 s25, s1, s69 // (op 13849) 


s_add_i32 s1, s18, s25 // (op 13849) 


s_ashr_i32 s18, s1, s24 // (op 13849) 


s_lshl_b32 s1, s45, 24 // (op 13849) 
s_ashr_i32 s24, s1, 31 // (op 13849) 

s_xor_b32 s1, s18, s24 // (op 13849) 

s_sub_i32 s18, s1, s24 // (op 13849) 


s_mul_i32 s1, s11, s22 // (op 13849) 

s_mul_i32 s22, s8, s1 // (op 13849) 

s_mul_i32 s1, s18, s22 // (op 13849) 

s_sub_u32 s22, s16, s1 // (op 13849) 

s_mul_hi_i32 s1, s22, s12 // (op 13849) 
s_add_i32 s23, s1, s22 // (op 13849) 


s_ashr_i32 s1, s23, 31 // (op 13849) 
s_cmp_eq_i32 s12, 0 // (op 13849) 
s_mov_b32 s22, scc // (op 13849) 
s_cmp_eq_i32 s22, 0 // (op 13849) 
s_cselect_b32 s24, 0, -1 // (op 13849) 

s_and_b32 s22, s44, 31 // (op 13849) 
s_lshl_b32 s25, 1, s22 // (op 13849) 
s_add_i32 s68, s25, s24 // (op 13849) 


s_and_b32 s24, s1, s68 // (op 13849) 


s_add_i32 s1, s23, s24 // (op 13849) 


s_ashr_i32 s23, s1, s22 // (op 13849) 


s_lshl_b32 s1, s44, 24 // (op 13849) 
s_ashr_i32 s22, s1, 31 // (op 13849) 

s_xor_b32 s1, s23, s22 // (op 13849) 

s_sub_i32 s23, s1, s22 // (op 13849) 


s_cmp_eq_i32 s18, 0 // (op 13849) 
s_mov_b32 s1, scc // (op 13849) 

s_cmp_eq_i32 s1, 0 // (op 13849) 
s_cselect_b32 s18, s23, s19 // (op 13849) 



v_lshl_add_u32 v17, s18, 8, v16 // (op 13849) 


v_mul_lo_u32 v16, v17, s38 // (op 13849) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 13849) 
s_lshr_b32 s1, s18, 1 // (op 13849) 

v_lshl_add_u32 v17, s1, 3, v15 // (op 13849) 


v_add_u32 v15, v17, v16 // (op 13849) 


s_mul_i32 s19, s38, s7 // (op 13851) most significant: low of low * high
s_mul_i32 s1, s39, s6 // (op 13851) most significant: low of high * low
s_mul_i32 s18, s38, s6 // (op 13851) least significant
s_mul_hi_u32 s22, s38, s6 // (op 13851) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 13851) most significant: sum
s_add_u32 s19, s19, s22 // (op 13851) most significant: sum


s_mov_b32 s68, s40 // (op 13851) reg expression
s_mov_b32 s69, s41 // (op 13851) reg expression
s_mov_b32 s70, s18 // (op 13851) reg expression
s_mov_b32 s71, 131072 // (op 13851) call()

s_ashr_i64 s[18:19], s[42:43], 7 // (op 4010) 
s_and_b32 s1, s18, 4294967294 // (op 4010) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 4009) 
s_mov_b32 s22, s18 // (op 4009) convert

s_cmp_lt_u32 s1, s22 // (op 4009) 












s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_4009 // (op 4009) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_4009 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopTop_KLoopTail_4009:
s_barrier  // (op 14057) 
s_add_u32 s1, s1, 1 // (op 4011) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 13257) 
v_mov_b64 v[16:17], s[18:19] // (op 13257) convert


v_mov_b64 v[18:19], v[10:11] // (op 13257) 
v_mov_b64 v[20:21], v[18:19] // (op 13257) 
s_add_u32 s18, 32768, s47 // (op 13257) 
s_mov_b32 m0, s18 // (op 13257) reg expression
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 13257) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 13257) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 13257) most significant half

s_add_u32 m0, m0, 4096 // (op 13257) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 13257) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 13257) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 13257) most significant half

s_add_u32 m0, m0, 4096 // (op 13257) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 13257) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 13257) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 13257) most significant half

s_add_u32 m0, m0, 4096 // (op 13257) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 13257) Load value direct to lds




v_add_co_u32 v10, vcc, 128, v10 // (op 13294) least significant half
v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 13294) most significant half
s_barrier  // (op 14074) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 13263) 
v_mov_b64 v[16:17], s[18:19] // (op 13263) convert


v_mov_b64 v[18:19], v[12:13] // (op 13263) 
v_mov_b64 v[20:21], v[18:19] // (op 13263) 
s_mov_b32 m0, s0 // (op 13263) reg expression
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 13263) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 13263) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 13263) most significant half

s_add_u32 m0, m0, 4096 // (op 13263) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 13263) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 13263) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 13263) most significant half

s_add_u32 m0, m0, 4096 // (op 13263) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 13263) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 13263) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 13263) most significant half

s_add_u32 m0, m0, 4096 // (op 13263) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 13263) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 13263) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 13263) most significant half

s_add_u32 m0, m0, 4096 // (op 13263) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 13263) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 13263) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 13263) most significant half

s_add_u32 m0, m0, 4096 // (op 13263) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 13263) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 13263) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 13263) most significant half

s_add_u32 m0, m0, 4096 // (op 13263) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 13263) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 13263) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 13263) most significant half

s_add_u32 m0, m0, 4096 // (op 13263) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 13263) Load value direct to lds



v_add_co_u32 v12, vcc, 128, v12 // (op 13394) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 13394) most significant half
s_waitcnt vmcnt(8)
s_barrier  // (op 14064) 



ds_read_b128 v[16:19], v3 offset:32768 // (op 160) Load local data 
ds_read_b128 v[20:23], v3 offset:32832 // (op 160) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 14080) 



ds_read_b128 v[24:27], v4 // (op 162) Load local data 
ds_read_b128 v[28:31], v4 offset:64 // (op 162) Load local data 


v_mov_b32 v32, v14 // (op 164) 
v_mov_b32 v33, v32 // (op 164) reg expression

buffer_load_dwordx2 v[34:35], v33, s[64:67], 0 offen // (op 164) Load value


v_add_u32 v14, 4, v14 // (op 13812) 


v_mov_b32 v32, v15 // (op 166) 
v_mov_b32 v33, v32 // (op 166) reg expression

buffer_load_dwordx2 v[36:37], v33, s[68:71], 0 offen // (op 166) Load value

s_waitcnt vmcnt(1)
v_bfe_u32 v38, v34, 0, 8 // (op 12367) 



v_bfe_u32 v39, v34, 8, 8 // (op 12367) 
v_bfe_u32 v40, v34, 16, 8 // (op 12367) 
v_bfe_u32 v41, v34, 24, 8 // (op 12367) 
v_permlane16_swap_b32 v38, v39 // (op 12367) 
ds_read_b128 v[42:45], v4 offset:2048 // (op 291) Load local data 
ds_read_b128 v[46:49], v4 offset:2112 // (op 291) Load local data 
v_permlane16_swap_b32 v40, v41 // (op 12367) 



s_nop 1
v_permlane32_swap_b32 v38, v40 // (op 12367) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v39, v41 // (op 12367) 
ds_read_b128 v[50:53], v4 offset:4096 // (op 427) Load local data 

ds_read_b128 v[54:57], v4 offset:4160 // (op 427) Load local data 



ds_read_b128 v[58:61], v4 offset:6144 // (op 563) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v66, v36, 0, 8 // (op 12413) 
v_bfe_u32 v67, v36, 8, 8 // (op 12413) 
v_bfe_u32 v68, v36, 16, 8 // (op 12413) 
v_bfe_u32 v69, v36, 24, 8 // (op 12413) 
v_permlane16_swap_b32 v66, v67 // (op 12413) 
s_nop 0
v_permlane16_swap_b32 v68, v69 // (op 12413) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v66, v68 // (op 12413) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v67, v69 // (op 12413) 
ds_read_b128 v[62:65], v4 offset:6208 // (op 563) Load local data 




s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:23], v[24:31], a[0:3], v38, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 167) 
ds_read_b128 v[70:73], v4 offset:8192 // (op 699) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[16:23], v[42:49], a[4:7], v38, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 294) 
ds_read_b128 v[74:77], v4 offset:8256 // (op 699) Load local data 


s_lshl_b32 s18, s38, 6 // (op 704) 
v_add_u32 v33, v32, s18 // (op 704) 


buffer_load_dwordx2 v[34:35], v33, s[68:71], 0 offen // (op 704) Load value

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[16:23], v[50:57], a[8:11], v38, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 430) 




v_add_u32 v15, 4, v15 // (op 13839) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[16:23], v[58:65], a[12:15], v38, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 566) 
ds_read_b128 v[78:81], v4 offset:10240 // (op 835) Load local data 
ds_read_b128 v[82:85], v4 offset:10304 // (op 835) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v86, v34, 0, 8 // (op 12427) 



v_bfe_u32 v87, v34, 8, 8 // (op 12427) 
v_bfe_u32 v88, v34, 16, 8 // (op 12427) 
v_bfe_u32 v89, v34, 24, 8 // (op 12427) 
v_permlane16_swap_b32 v86, v87 // (op 12427) 
ds_read_b128 v[90:93], v4 offset:12288 // (op 971) Load local data 
v_permlane16_swap_b32 v88, v89 // (op 12427) 
s_nop 1
v_permlane32_swap_b32 v86, v88 // (op 12427) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v87, v89 // (op 12427) 
ds_read_b128 v[94:97], v4 offset:12352 // (op 971) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[16:23], v[70:77], a[16:19], v38, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 702) 
ds_read_b128 v[98:101], v4 offset:14336 // (op 1107) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[16:23], v[78:85], a[20:23], v38, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 838) 
ds_read_b128 v[102:105], v4 offset:14400 // (op 1107) Load local data 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[16:23], v[90:97], a[24:27], v38, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 974) 
ds_read_b128 v[106:109], v3 offset:34816 // (op 1417) Load local data 
ds_read_b128 v[110:113], v3 offset:34880 // (op 1417) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[16:23], v[98:105], a[28:31], v38, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1110) 




ds_read_b128 v[16:19], v3 offset:36864 // (op 2635) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[106:113], v[24:31], a[32:35], v39, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1421) 
ds_read_b128 v[20:23], v3 offset:36928 // (op 2635) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[106:113], v[50:57], a[40:43], v39, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1475) 
ds_read_b128 v[114:117], v3 offset:38912 // (op 3853) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[106:113], v[58:65], a[44:47], v39, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1502) 
ds_read_b128 v[118:121], v3 offset:38976 // (op 3853) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[106:113], v[42:49], a[36:39], v39, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1448) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[106:113], v[70:77], a[48:51], v39, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1529) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[106:113], v[90:97], a[56:59], v39, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1583) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[106:113], v[98:105], a[60:63], v39, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1610) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[106:113], v[78:85], a[52:55], v39, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1556) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[16:23], v[24:31], a[64:67], v40, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2639) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[16:23], v[50:57], a[72:75], v40, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2693) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[16:23], v[58:65], a[76:79], v40, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2720) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[16:23], v[42:49], a[68:71], v40, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2666) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[16:23], v[70:77], a[80:83], v40, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2747) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[16:23], v[90:97], a[88:91], v40, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2801) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[16:23], v[98:105], a[92:95], v40, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2828) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[16:23], v[78:85], a[84:87], v40, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2774) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[114:121], v[24:31], a[96:99], v41, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3857) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[114:121], v[50:57], a[104:107], v41, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3911) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[114:121], v[58:65], a[108:111], v41, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3938) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[114:121], v[42:49], a[100:103], v41, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3884) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[114:121], v[70:77], a[112:115], v41, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3965) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[114:121], v[90:97], a[120:123], v41, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4019) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[114:121], v[98:105], a[124:127], v41, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 4046) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[114:121], v[78:85], a[116:119], v41, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3992) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 4009) 
s_mov_b32 s22, s18 // (op 4009) convert

s_cmp_lt_u32 s1, s22 // (op 4009) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopTop_KLoopTail_4009 // (op 4009) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopTop_KLoopTail_4009 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_4009:




















s_barrier  // (op 14157) 

















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ConditionalFalse_0_beta_114
v_accvgpr_read v10, a0 // (op 42) 
v_accvgpr_read v11, a1 // (op 42) 
v_accvgpr_read v12, a2 // (op 42) 
v_accvgpr_read v13, a3 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 
v_mul_f32 v16, v7, v12 // (op 42) 
v_mul_f32 v17, v7, v13 // (op 42) 



v_mov_b32 v3, v9 // (op 48) reg expression

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 48) Store value


v_accvgpr_read v10, a4 // (op 5646) 
v_accvgpr_read v11, a5 // (op 5646) 
v_accvgpr_read v12, a6 // (op 5646) 
v_accvgpr_read v13, a7 // (op 5646) 
v_mul_f32 v14, v7, v10 // (op 5646) 
v_mul_f32 v15, v7, v11 // (op 5646) 
v_mul_f32 v16, v7, v12 // (op 5646) 
v_mul_f32 v17, v7, v13 // (op 5646) 



s_lshl_b32 s0, s4, 6 // (op 5648) 
v_add_u32 v3, v9, s0 // (op 5648) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5648) Store value


v_accvgpr_read v10, a8 // (op 5658) 
v_accvgpr_read v11, a9 // (op 5658) 
v_accvgpr_read v12, a10 // (op 5658) 
v_accvgpr_read v13, a11 // (op 5658) 
v_mul_f32 v14, v7, v10 // (op 5658) 
v_mul_f32 v15, v7, v11 // (op 5658) 
v_mul_f32 v16, v7, v12 // (op 5658) 
v_mul_f32 v17, v7, v13 // (op 5658) 



s_lshl_b32 s0, s4, 7 // (op 5660) 
v_add_u32 v3, v9, s0 // (op 5660) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5660) Store value


v_accvgpr_read v10, a12 // (op 5670) 
v_accvgpr_read v11, a13 // (op 5670) 
v_accvgpr_read v12, a14 // (op 5670) 
v_accvgpr_read v13, a15 // (op 5670) 
v_mul_f32 v14, v7, v10 // (op 5670) 
v_mul_f32 v15, v7, v11 // (op 5670) 
v_mul_f32 v16, v7, v12 // (op 5670) 
v_mul_f32 v17, v7, v13 // (op 5670) 



s_lshl_b32 s0, s4, 6 // (op 5672) 
s_mul_i32 s1, 3, s0 // (op 5672) 

v_add_u32 v3, v9, s1 // (op 5672) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5672) Store value


v_accvgpr_read v10, a16 // (op 5682) 
v_accvgpr_read v11, a17 // (op 5682) 
v_accvgpr_read v12, a18 // (op 5682) 
v_accvgpr_read v13, a19 // (op 5682) 
v_mul_f32 v14, v7, v10 // (op 5682) 
v_mul_f32 v15, v7, v11 // (op 5682) 
v_mul_f32 v16, v7, v12 // (op 5682) 
v_mul_f32 v17, v7, v13 // (op 5682) 



s_lshl_b32 s0, s4, 8 // (op 5684) 
v_add_u32 v3, v9, s0 // (op 5684) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5684) Store value


v_accvgpr_read v10, a20 // (op 5694) 
v_accvgpr_read v11, a21 // (op 5694) 
v_accvgpr_read v12, a22 // (op 5694) 
v_accvgpr_read v13, a23 // (op 5694) 
v_mul_f32 v14, v7, v10 // (op 5694) 
v_mul_f32 v15, v7, v11 // (op 5694) 
v_mul_f32 v16, v7, v12 // (op 5694) 
v_mul_f32 v17, v7, v13 // (op 5694) 



s_lshl_b32 s0, s4, 6 // (op 5696) 
s_mul_i32 s1, 5, s0 // (op 5696) 

v_add_u32 v3, v9, s1 // (op 5696) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5696) Store value


v_accvgpr_read v10, a24 // (op 5706) 
v_accvgpr_read v11, a25 // (op 5706) 
v_accvgpr_read v12, a26 // (op 5706) 
v_accvgpr_read v13, a27 // (op 5706) 
v_mul_f32 v14, v7, v10 // (op 5706) 
v_mul_f32 v15, v7, v11 // (op 5706) 
v_mul_f32 v16, v7, v12 // (op 5706) 
v_mul_f32 v17, v7, v13 // (op 5706) 



s_lshl_b32 s0, s4, 6 // (op 5708) 
s_mul_i32 s1, 6, s0 // (op 5708) 

v_add_u32 v3, v9, s1 // (op 5708) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5708) Store value


v_accvgpr_read v10, a28 // (op 5718) 
v_accvgpr_read v11, a29 // (op 5718) 
v_accvgpr_read v12, a30 // (op 5718) 
v_accvgpr_read v13, a31 // (op 5718) 
v_mul_f32 v14, v7, v10 // (op 5718) 
v_mul_f32 v15, v7, v11 // (op 5718) 
v_mul_f32 v16, v7, v12 // (op 5718) 
v_mul_f32 v17, v7, v13 // (op 5718) 



s_lshl_b32 s0, s4, 6 // (op 5720) 
s_mul_i32 s1, 7, s0 // (op 5720) 

v_add_u32 v3, v9, s1 // (op 5720) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5720) Store value


v_accvgpr_read v10, a32 // (op 5772) 
v_accvgpr_read v11, a33 // (op 5772) 
v_accvgpr_read v12, a34 // (op 5772) 
v_accvgpr_read v13, a35 // (op 5772) 
v_mul_f32 v14, v7, v10 // (op 5772) 
v_mul_f32 v15, v7, v11 // (op 5772) 
v_mul_f32 v16, v7, v12 // (op 5772) 
v_mul_f32 v17, v7, v13 // (op 5772) 



v_add_u32 v3, 64, v9 // (op 5775) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5775) Store value


v_accvgpr_read v10, a36 // (op 5793) 
v_accvgpr_read v11, a37 // (op 5793) 
v_accvgpr_read v12, a38 // (op 5793) 
v_accvgpr_read v13, a39 // (op 5793) 
v_mul_f32 v14, v7, v10 // (op 5793) 
v_mul_f32 v15, v7, v11 // (op 5793) 
v_mul_f32 v16, v7, v12 // (op 5793) 
v_mul_f32 v17, v7, v13 // (op 5793) 



s_lshl_b32 s0, s4, 6 // (op 5777) 
s_add_u32 s0, s0, 64 // (op 5777) 
v_add_u32 v3, v9, s0 // (op 5777) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5777) Store value


v_accvgpr_read v10, a40 // (op 5798) 
v_accvgpr_read v11, a41 // (op 5798) 
v_accvgpr_read v12, a42 // (op 5798) 
v_accvgpr_read v13, a43 // (op 5798) 
v_mul_f32 v14, v7, v10 // (op 5798) 
v_mul_f32 v15, v7, v11 // (op 5798) 
v_mul_f32 v16, v7, v12 // (op 5798) 
v_mul_f32 v17, v7, v13 // (op 5798) 



s_lshl_b32 s0, s4, 7 // (op 5779) 
s_add_u32 s0, s0, 64 // (op 5779) 
v_add_u32 v3, v9, s0 // (op 5779) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5779) Store value


v_accvgpr_read v10, a44 // (op 5803) 
v_accvgpr_read v11, a45 // (op 5803) 
v_accvgpr_read v12, a46 // (op 5803) 
v_accvgpr_read v13, a47 // (op 5803) 
v_mul_f32 v14, v7, v10 // (op 5803) 
v_mul_f32 v15, v7, v11 // (op 5803) 
v_mul_f32 v16, v7, v12 // (op 5803) 
v_mul_f32 v17, v7, v13 // (op 5803) 



s_lshl_b32 s0, s4, 6 // (op 5781) 
s_mul_i32 s1, 3, s0 // (op 5781) 
s_add_u32 s1, s1, 64 // (op 5781) 

v_add_u32 v3, v9, s1 // (op 5781) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5781) Store value


v_accvgpr_read v10, a48 // (op 5808) 
v_accvgpr_read v11, a49 // (op 5808) 
v_accvgpr_read v12, a50 // (op 5808) 
v_accvgpr_read v13, a51 // (op 5808) 
v_mul_f32 v14, v7, v10 // (op 5808) 
v_mul_f32 v15, v7, v11 // (op 5808) 
v_mul_f32 v16, v7, v12 // (op 5808) 
v_mul_f32 v17, v7, v13 // (op 5808) 



s_lshl_b32 s0, s4, 8 // (op 5783) 
s_add_u32 s0, s0, 64 // (op 5783) 
v_add_u32 v3, v9, s0 // (op 5783) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5783) Store value


v_accvgpr_read v10, a52 // (op 5813) 
v_accvgpr_read v11, a53 // (op 5813) 
v_accvgpr_read v12, a54 // (op 5813) 
v_accvgpr_read v13, a55 // (op 5813) 
v_mul_f32 v14, v7, v10 // (op 5813) 
v_mul_f32 v15, v7, v11 // (op 5813) 
v_mul_f32 v16, v7, v12 // (op 5813) 
v_mul_f32 v17, v7, v13 // (op 5813) 



s_lshl_b32 s0, s4, 6 // (op 5785) 
s_mul_i32 s1, 5, s0 // (op 5785) 
s_add_u32 s1, s1, 64 // (op 5785) 

v_add_u32 v3, v9, s1 // (op 5785) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5785) Store value


v_accvgpr_read v10, a56 // (op 5818) 
v_accvgpr_read v11, a57 // (op 5818) 
v_accvgpr_read v12, a58 // (op 5818) 
v_accvgpr_read v13, a59 // (op 5818) 
v_mul_f32 v14, v7, v10 // (op 5818) 
v_mul_f32 v15, v7, v11 // (op 5818) 
v_mul_f32 v16, v7, v12 // (op 5818) 
v_mul_f32 v17, v7, v13 // (op 5818) 



s_lshl_b32 s0, s4, 6 // (op 5787) 
s_mul_i32 s1, 6, s0 // (op 5787) 
s_add_u32 s1, s1, 64 // (op 5787) 

v_add_u32 v3, v9, s1 // (op 5787) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5787) Store value


v_accvgpr_read v10, a60 // (op 5823) 
v_accvgpr_read v11, a61 // (op 5823) 
v_accvgpr_read v12, a62 // (op 5823) 
v_accvgpr_read v13, a63 // (op 5823) 
v_mul_f32 v14, v7, v10 // (op 5823) 
v_mul_f32 v15, v7, v11 // (op 5823) 
v_mul_f32 v16, v7, v12 // (op 5823) 
v_mul_f32 v17, v7, v13 // (op 5823) 



s_lshl_b32 s0, s4, 6 // (op 5789) 
s_mul_i32 s1, 7, s0 // (op 5789) 
s_add_u32 s1, s1, 64 // (op 5789) 

v_add_u32 v3, v9, s1 // (op 5789) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5789) Store value


v_accvgpr_read v10, a64 // (op 5904) 
v_accvgpr_read v11, a65 // (op 5904) 
v_accvgpr_read v12, a66 // (op 5904) 
v_accvgpr_read v13, a67 // (op 5904) 
v_mul_f32 v14, v7, v10 // (op 5904) 
v_mul_f32 v15, v7, v11 // (op 5904) 
v_mul_f32 v16, v7, v12 // (op 5904) 
v_mul_f32 v17, v7, v13 // (op 5904) 



v_add_u32 v3, 128, v9 // (op 5907) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5907) Store value


v_accvgpr_read v10, a68 // (op 5925) 
v_accvgpr_read v11, a69 // (op 5925) 
v_accvgpr_read v12, a70 // (op 5925) 
v_accvgpr_read v13, a71 // (op 5925) 
v_mul_f32 v14, v7, v10 // (op 5925) 
v_mul_f32 v15, v7, v11 // (op 5925) 
v_mul_f32 v16, v7, v12 // (op 5925) 
v_mul_f32 v17, v7, v13 // (op 5925) 



s_lshl_b32 s0, s4, 6 // (op 5909) 
s_add_u32 s0, s0, 128 // (op 5909) 
v_add_u32 v3, v9, s0 // (op 5909) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5909) Store value


v_accvgpr_read v10, a72 // (op 5930) 
v_accvgpr_read v11, a73 // (op 5930) 
v_accvgpr_read v12, a74 // (op 5930) 
v_accvgpr_read v13, a75 // (op 5930) 
v_mul_f32 v14, v7, v10 // (op 5930) 
v_mul_f32 v15, v7, v11 // (op 5930) 
v_mul_f32 v16, v7, v12 // (op 5930) 
v_mul_f32 v17, v7, v13 // (op 5930) 



s_lshl_b32 s0, s4, 7 // (op 5911) 
s_add_u32 s0, s0, 128 // (op 5911) 
v_add_u32 v3, v9, s0 // (op 5911) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5911) Store value


v_accvgpr_read v10, a76 // (op 5935) 
v_accvgpr_read v11, a77 // (op 5935) 
v_accvgpr_read v12, a78 // (op 5935) 
v_accvgpr_read v13, a79 // (op 5935) 
v_mul_f32 v14, v7, v10 // (op 5935) 
v_mul_f32 v15, v7, v11 // (op 5935) 
v_mul_f32 v16, v7, v12 // (op 5935) 
v_mul_f32 v17, v7, v13 // (op 5935) 



s_lshl_b32 s0, s4, 6 // (op 5913) 
s_mul_i32 s1, 3, s0 // (op 5913) 
s_add_u32 s1, s1, 128 // (op 5913) 

v_add_u32 v3, v9, s1 // (op 5913) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5913) Store value


v_accvgpr_read v10, a80 // (op 5940) 
v_accvgpr_read v11, a81 // (op 5940) 
v_accvgpr_read v12, a82 // (op 5940) 
v_accvgpr_read v13, a83 // (op 5940) 
v_mul_f32 v14, v7, v10 // (op 5940) 
v_mul_f32 v15, v7, v11 // (op 5940) 
v_mul_f32 v16, v7, v12 // (op 5940) 
v_mul_f32 v17, v7, v13 // (op 5940) 



s_lshl_b32 s0, s4, 8 // (op 5915) 
s_add_u32 s0, s0, 128 // (op 5915) 
v_add_u32 v3, v9, s0 // (op 5915) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5915) Store value


v_accvgpr_read v10, a84 // (op 5945) 
v_accvgpr_read v11, a85 // (op 5945) 
v_accvgpr_read v12, a86 // (op 5945) 
v_accvgpr_read v13, a87 // (op 5945) 
v_mul_f32 v14, v7, v10 // (op 5945) 
v_mul_f32 v15, v7, v11 // (op 5945) 
v_mul_f32 v16, v7, v12 // (op 5945) 
v_mul_f32 v17, v7, v13 // (op 5945) 



s_lshl_b32 s0, s4, 6 // (op 5917) 
s_mul_i32 s1, 5, s0 // (op 5917) 
s_add_u32 s1, s1, 128 // (op 5917) 

v_add_u32 v3, v9, s1 // (op 5917) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5917) Store value


v_accvgpr_read v10, a88 // (op 5950) 
v_accvgpr_read v11, a89 // (op 5950) 
v_accvgpr_read v12, a90 // (op 5950) 
v_accvgpr_read v13, a91 // (op 5950) 
v_mul_f32 v14, v7, v10 // (op 5950) 
v_mul_f32 v15, v7, v11 // (op 5950) 
v_mul_f32 v16, v7, v12 // (op 5950) 
v_mul_f32 v17, v7, v13 // (op 5950) 



s_lshl_b32 s0, s4, 6 // (op 5919) 
s_mul_i32 s1, 6, s0 // (op 5919) 
s_add_u32 s1, s1, 128 // (op 5919) 

v_add_u32 v3, v9, s1 // (op 5919) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5919) Store value


v_accvgpr_read v10, a92 // (op 5955) 
v_accvgpr_read v11, a93 // (op 5955) 
v_accvgpr_read v12, a94 // (op 5955) 
v_accvgpr_read v13, a95 // (op 5955) 
v_mul_f32 v14, v7, v10 // (op 5955) 
v_mul_f32 v15, v7, v11 // (op 5955) 
v_mul_f32 v16, v7, v12 // (op 5955) 
v_mul_f32 v17, v7, v13 // (op 5955) 



s_lshl_b32 s0, s4, 6 // (op 5921) 
s_mul_i32 s1, 7, s0 // (op 5921) 
s_add_u32 s1, s1, 128 // (op 5921) 

v_add_u32 v3, v9, s1 // (op 5921) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5921) Store value


v_accvgpr_read v10, a96 // (op 6036) 
v_accvgpr_read v11, a97 // (op 6036) 
v_accvgpr_read v12, a98 // (op 6036) 
v_accvgpr_read v13, a99 // (op 6036) 
v_mul_f32 v14, v7, v10 // (op 6036) 
v_mul_f32 v15, v7, v11 // (op 6036) 
v_mul_f32 v16, v7, v12 // (op 6036) 
v_mul_f32 v17, v7, v13 // (op 6036) 



v_add_u32 v3, 192, v9 // (op 6039) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6039) Store value


v_accvgpr_read v10, a100 // (op 6057) 
v_accvgpr_read v11, a101 // (op 6057) 
v_accvgpr_read v12, a102 // (op 6057) 
v_accvgpr_read v13, a103 // (op 6057) 
v_mul_f32 v14, v7, v10 // (op 6057) 
v_mul_f32 v15, v7, v11 // (op 6057) 
v_mul_f32 v16, v7, v12 // (op 6057) 
v_mul_f32 v17, v7, v13 // (op 6057) 



s_lshl_b32 s0, s4, 6 // (op 6041) 
s_add_u32 s0, s0, 192 // (op 6041) 
v_add_u32 v3, v9, s0 // (op 6041) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6041) Store value


v_accvgpr_read v10, a104 // (op 6062) 
v_accvgpr_read v11, a105 // (op 6062) 
v_accvgpr_read v12, a106 // (op 6062) 
v_accvgpr_read v13, a107 // (op 6062) 
v_mul_f32 v14, v7, v10 // (op 6062) 
v_mul_f32 v15, v7, v11 // (op 6062) 
v_mul_f32 v16, v7, v12 // (op 6062) 
v_mul_f32 v17, v7, v13 // (op 6062) 



s_lshl_b32 s0, s4, 7 // (op 6043) 
s_add_u32 s0, s0, 192 // (op 6043) 
v_add_u32 v3, v9, s0 // (op 6043) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6043) Store value


v_accvgpr_read v10, a108 // (op 6067) 
v_accvgpr_read v11, a109 // (op 6067) 
v_accvgpr_read v12, a110 // (op 6067) 
v_accvgpr_read v13, a111 // (op 6067) 
v_mul_f32 v14, v7, v10 // (op 6067) 
v_mul_f32 v15, v7, v11 // (op 6067) 
v_mul_f32 v16, v7, v12 // (op 6067) 
v_mul_f32 v17, v7, v13 // (op 6067) 



s_lshl_b32 s0, s4, 6 // (op 6045) 
s_mul_i32 s1, 3, s0 // (op 6045) 
s_add_u32 s1, s1, 192 // (op 6045) 

v_add_u32 v3, v9, s1 // (op 6045) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6045) Store value


v_accvgpr_read v10, a112 // (op 6072) 
v_accvgpr_read v11, a113 // (op 6072) 
v_accvgpr_read v12, a114 // (op 6072) 
v_accvgpr_read v13, a115 // (op 6072) 
v_mul_f32 v14, v7, v10 // (op 6072) 
v_mul_f32 v15, v7, v11 // (op 6072) 
v_mul_f32 v16, v7, v12 // (op 6072) 
v_mul_f32 v17, v7, v13 // (op 6072) 



s_lshl_b32 s0, s4, 8 // (op 6047) 
s_add_u32 s0, s0, 192 // (op 6047) 
v_add_u32 v3, v9, s0 // (op 6047) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6047) Store value


v_accvgpr_read v10, a116 // (op 6077) 
v_accvgpr_read v11, a117 // (op 6077) 
v_accvgpr_read v12, a118 // (op 6077) 
v_accvgpr_read v13, a119 // (op 6077) 
v_mul_f32 v14, v7, v10 // (op 6077) 
v_mul_f32 v15, v7, v11 // (op 6077) 
v_mul_f32 v16, v7, v12 // (op 6077) 
v_mul_f32 v17, v7, v13 // (op 6077) 



s_lshl_b32 s0, s4, 6 // (op 6049) 
s_mul_i32 s1, 5, s0 // (op 6049) 
s_add_u32 s1, s1, 192 // (op 6049) 

v_add_u32 v3, v9, s1 // (op 6049) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6049) Store value


v_accvgpr_read v10, a120 // (op 6082) 
v_accvgpr_read v11, a121 // (op 6082) 
v_accvgpr_read v12, a122 // (op 6082) 
v_accvgpr_read v13, a123 // (op 6082) 
v_mul_f32 v14, v7, v10 // (op 6082) 
v_mul_f32 v15, v7, v11 // (op 6082) 
v_mul_f32 v16, v7, v12 // (op 6082) 
v_mul_f32 v17, v7, v13 // (op 6082) 



s_lshl_b32 s0, s4, 6 // (op 6051) 
s_mul_i32 s1, 6, s0 // (op 6051) 
s_add_u32 s1, s1, 192 // (op 6051) 

v_add_u32 v3, v9, s1 // (op 6051) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6051) Store value


v_accvgpr_read v10, a124 // (op 6087) 
v_accvgpr_read v11, a125 // (op 6087) 
v_accvgpr_read v12, a126 // (op 6087) 
v_accvgpr_read v13, a127 // (op 6087) 
v_mul_f32 v14, v7, v10 // (op 6087) 
v_mul_f32 v15, v7, v11 // (op 6087) 
v_mul_f32 v16, v7, v12 // (op 6087) 
v_mul_f32 v17, v7, v13 // (op 6087) 



s_lshl_b32 s0, s4, 6 // (op 6053) 
s_mul_i32 s1, 7, s0 // (op 6053) 
s_add_u32 s1, s1, 192 // (op 6053) 

v_add_u32 v3, v9, s1 // (op 6053) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6053) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ConditionalBottom_0_beta_114
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ConditionalFalse_0_beta_114:


v_add_u32 v3, 192, v8 // (op 6763) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6763) Load value

v_accvgpr_read v14, a96 // (op 6743) 
v_accvgpr_read v15, a97 // (op 6743) 
v_accvgpr_read v16, a98 // (op 6743) 
v_accvgpr_read v17, a99 // (op 6743) 
v_mul_f32 v18, v7, v14 // (op 6743) 
v_mul_f32 v19, v7, v15 // (op 6743) 
v_mul_f32 v20, v7, v16 // (op 6743) 
v_mul_f32 v21, v7, v17 // (op 6743) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6762) 
v_mul_f32 v15, v0, v11 // (op 6762) 
v_mul_f32 v16, v0, v12 // (op 6762) 
v_mul_f32 v17, v0, v13 // (op 6762) 

v_add_f32 v10, v14, v18 // (op 6744) 
v_add_f32 v11, v15, v19 // (op 6744) 
v_add_f32 v12, v16, v20 // (op 6744) 
v_add_f32 v13, v17, v21 // (op 6744) 




v_add_u32 v3, 192, v9 // (op 6746) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6746) Store value




s_lshl_b32 s0, s2, 6 // (op 6767) 
s_add_u32 s0, s0, 192 // (op 6767) 
v_add_u32 v3, v8, s0 // (op 6767) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6767) Load value

v_accvgpr_read v14, a100 // (op 6793) 
v_accvgpr_read v15, a101 // (op 6793) 
v_accvgpr_read v16, a102 // (op 6793) 
v_accvgpr_read v17, a103 // (op 6793) 
v_mul_f32 v18, v7, v14 // (op 6793) 
v_mul_f32 v19, v7, v15 // (op 6793) 
v_mul_f32 v20, v7, v16 // (op 6793) 
v_mul_f32 v21, v7, v17 // (op 6793) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6765) 
v_mul_f32 v15, v0, v11 // (op 6765) 
v_mul_f32 v16, v0, v12 // (op 6765) 
v_mul_f32 v17, v0, v13 // (op 6765) 

v_add_f32 v10, v14, v18 // (op 6766) 
v_add_f32 v11, v15, v19 // (op 6766) 
v_add_f32 v12, v16, v20 // (op 6766) 
v_add_f32 v13, v17, v21 // (op 6766) 




s_lshl_b32 s0, s4, 6 // (op 6748) 
s_add_u32 s0, s0, 192 // (op 6748) 
v_add_u32 v3, v9, s0 // (op 6748) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6748) Store value




s_lshl_b32 s0, s2, 7 // (op 6771) 
s_add_u32 s0, s0, 192 // (op 6771) 
v_add_u32 v3, v8, s0 // (op 6771) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6771) Load value

v_accvgpr_read v14, a104 // (op 6795) 
v_accvgpr_read v15, a105 // (op 6795) 
v_accvgpr_read v16, a106 // (op 6795) 
v_accvgpr_read v17, a107 // (op 6795) 
v_mul_f32 v18, v7, v14 // (op 6795) 
v_mul_f32 v19, v7, v15 // (op 6795) 
v_mul_f32 v20, v7, v16 // (op 6795) 
v_mul_f32 v21, v7, v17 // (op 6795) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6769) 
v_mul_f32 v15, v0, v11 // (op 6769) 
v_mul_f32 v16, v0, v12 // (op 6769) 
v_mul_f32 v17, v0, v13 // (op 6769) 

v_add_f32 v10, v14, v18 // (op 6770) 
v_add_f32 v11, v15, v19 // (op 6770) 
v_add_f32 v12, v16, v20 // (op 6770) 
v_add_f32 v13, v17, v21 // (op 6770) 




s_lshl_b32 s0, s4, 7 // (op 6750) 
s_add_u32 s0, s0, 192 // (op 6750) 
v_add_u32 v3, v9, s0 // (op 6750) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6750) Store value




s_lshl_b32 s0, s2, 6 // (op 6775) 
s_mul_i32 s1, 3, s0 // (op 6775) 
s_add_u32 s1, s1, 192 // (op 6775) 

v_add_u32 v3, v8, s1 // (op 6775) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6775) Load value

v_accvgpr_read v14, a108 // (op 6797) 
v_accvgpr_read v15, a109 // (op 6797) 
v_accvgpr_read v16, a110 // (op 6797) 
v_accvgpr_read v17, a111 // (op 6797) 
v_mul_f32 v18, v7, v14 // (op 6797) 
v_mul_f32 v19, v7, v15 // (op 6797) 
v_mul_f32 v20, v7, v16 // (op 6797) 
v_mul_f32 v21, v7, v17 // (op 6797) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6773) 
v_mul_f32 v15, v0, v11 // (op 6773) 
v_mul_f32 v16, v0, v12 // (op 6773) 
v_mul_f32 v17, v0, v13 // (op 6773) 

v_add_f32 v10, v14, v18 // (op 6774) 
v_add_f32 v11, v15, v19 // (op 6774) 
v_add_f32 v12, v16, v20 // (op 6774) 
v_add_f32 v13, v17, v21 // (op 6774) 




s_lshl_b32 s0, s4, 6 // (op 6752) 
s_mul_i32 s1, 3, s0 // (op 6752) 
s_add_u32 s1, s1, 192 // (op 6752) 

v_add_u32 v3, v9, s1 // (op 6752) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6752) Store value




s_lshl_b32 s0, s2, 8 // (op 6779) 
s_add_u32 s0, s0, 192 // (op 6779) 
v_add_u32 v3, v8, s0 // (op 6779) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6779) Load value

v_accvgpr_read v14, a112 // (op 6799) 
v_accvgpr_read v15, a113 // (op 6799) 
v_accvgpr_read v16, a114 // (op 6799) 
v_accvgpr_read v17, a115 // (op 6799) 
v_mul_f32 v18, v7, v14 // (op 6799) 
v_mul_f32 v19, v7, v15 // (op 6799) 
v_mul_f32 v20, v7, v16 // (op 6799) 
v_mul_f32 v21, v7, v17 // (op 6799) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6777) 
v_mul_f32 v15, v0, v11 // (op 6777) 
v_mul_f32 v16, v0, v12 // (op 6777) 
v_mul_f32 v17, v0, v13 // (op 6777) 

v_add_f32 v10, v14, v18 // (op 6778) 
v_add_f32 v11, v15, v19 // (op 6778) 
v_add_f32 v12, v16, v20 // (op 6778) 
v_add_f32 v13, v17, v21 // (op 6778) 




s_lshl_b32 s0, s4, 8 // (op 6754) 
s_add_u32 s0, s0, 192 // (op 6754) 
v_add_u32 v3, v9, s0 // (op 6754) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6754) Store value




s_lshl_b32 s0, s2, 6 // (op 6783) 
s_mul_i32 s1, 5, s0 // (op 6783) 
s_add_u32 s1, s1, 192 // (op 6783) 

v_add_u32 v3, v8, s1 // (op 6783) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6783) Load value

v_accvgpr_read v14, a116 // (op 6801) 
v_accvgpr_read v15, a117 // (op 6801) 
v_accvgpr_read v16, a118 // (op 6801) 
v_accvgpr_read v17, a119 // (op 6801) 
v_mul_f32 v18, v7, v14 // (op 6801) 
v_mul_f32 v19, v7, v15 // (op 6801) 
v_mul_f32 v20, v7, v16 // (op 6801) 
v_mul_f32 v21, v7, v17 // (op 6801) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6781) 
v_mul_f32 v15, v0, v11 // (op 6781) 
v_mul_f32 v16, v0, v12 // (op 6781) 
v_mul_f32 v17, v0, v13 // (op 6781) 

v_add_f32 v10, v14, v18 // (op 6782) 
v_add_f32 v11, v15, v19 // (op 6782) 
v_add_f32 v12, v16, v20 // (op 6782) 
v_add_f32 v13, v17, v21 // (op 6782) 




s_lshl_b32 s0, s4, 6 // (op 6756) 
s_mul_i32 s1, 5, s0 // (op 6756) 
s_add_u32 s1, s1, 192 // (op 6756) 

v_add_u32 v3, v9, s1 // (op 6756) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6756) Store value




s_lshl_b32 s0, s2, 6 // (op 6787) 
s_mul_i32 s1, 6, s0 // (op 6787) 
s_add_u32 s1, s1, 192 // (op 6787) 

v_add_u32 v3, v8, s1 // (op 6787) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6787) Load value

v_accvgpr_read v14, a120 // (op 6803) 
v_accvgpr_read v15, a121 // (op 6803) 
v_accvgpr_read v16, a122 // (op 6803) 
v_accvgpr_read v17, a123 // (op 6803) 
v_mul_f32 v18, v7, v14 // (op 6803) 
v_mul_f32 v19, v7, v15 // (op 6803) 
v_mul_f32 v20, v7, v16 // (op 6803) 
v_mul_f32 v21, v7, v17 // (op 6803) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6785) 
v_mul_f32 v15, v0, v11 // (op 6785) 
v_mul_f32 v16, v0, v12 // (op 6785) 
v_mul_f32 v17, v0, v13 // (op 6785) 

v_add_f32 v10, v14, v18 // (op 6786) 
v_add_f32 v11, v15, v19 // (op 6786) 
v_add_f32 v12, v16, v20 // (op 6786) 
v_add_f32 v13, v17, v21 // (op 6786) 




s_lshl_b32 s0, s4, 6 // (op 6758) 
s_mul_i32 s1, 6, s0 // (op 6758) 
s_add_u32 s1, s1, 192 // (op 6758) 

v_add_u32 v3, v9, s1 // (op 6758) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6758) Store value




s_lshl_b32 s0, s2, 6 // (op 6791) 
s_mul_i32 s1, 7, s0 // (op 6791) 
s_add_u32 s1, s1, 192 // (op 6791) 

v_add_u32 v3, v8, s1 // (op 6791) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6791) Load value

v_accvgpr_read v14, a124 // (op 6805) 
v_accvgpr_read v15, a125 // (op 6805) 
v_accvgpr_read v16, a126 // (op 6805) 
v_accvgpr_read v17, a127 // (op 6805) 
v_mul_f32 v18, v7, v14 // (op 6805) 
v_mul_f32 v19, v7, v15 // (op 6805) 
v_mul_f32 v20, v7, v16 // (op 6805) 
v_mul_f32 v21, v7, v17 // (op 6805) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6789) 
v_mul_f32 v15, v0, v11 // (op 6789) 
v_mul_f32 v16, v0, v12 // (op 6789) 
v_mul_f32 v17, v0, v13 // (op 6789) 

v_add_f32 v10, v14, v18 // (op 6790) 
v_add_f32 v11, v15, v19 // (op 6790) 
v_add_f32 v12, v16, v20 // (op 6790) 
v_add_f32 v13, v17, v21 // (op 6790) 




s_lshl_b32 s0, s4, 6 // (op 6760) 
s_mul_i32 s1, 7, s0 // (op 6760) 
s_add_u32 s1, s1, 192 // (op 6760) 

v_add_u32 v3, v9, s1 // (op 6760) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6760) Store value




v_add_u32 v3, 128, v8 // (op 6608) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6608) Load value

v_accvgpr_read v14, a64 // (op 6588) 
v_accvgpr_read v15, a65 // (op 6588) 
v_accvgpr_read v16, a66 // (op 6588) 
v_accvgpr_read v17, a67 // (op 6588) 
v_mul_f32 v18, v7, v14 // (op 6588) 
v_mul_f32 v19, v7, v15 // (op 6588) 
v_mul_f32 v20, v7, v16 // (op 6588) 
v_mul_f32 v21, v7, v17 // (op 6588) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6607) 
v_mul_f32 v15, v0, v11 // (op 6607) 
v_mul_f32 v16, v0, v12 // (op 6607) 
v_mul_f32 v17, v0, v13 // (op 6607) 

v_add_f32 v10, v14, v18 // (op 6589) 
v_add_f32 v11, v15, v19 // (op 6589) 
v_add_f32 v12, v16, v20 // (op 6589) 
v_add_f32 v13, v17, v21 // (op 6589) 




v_add_u32 v3, 128, v9 // (op 6591) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6591) Store value




s_lshl_b32 s0, s2, 6 // (op 6612) 
s_add_u32 s0, s0, 128 // (op 6612) 
v_add_u32 v3, v8, s0 // (op 6612) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6612) Load value

v_accvgpr_read v14, a68 // (op 6638) 
v_accvgpr_read v15, a69 // (op 6638) 
v_accvgpr_read v16, a70 // (op 6638) 
v_accvgpr_read v17, a71 // (op 6638) 
v_mul_f32 v18, v7, v14 // (op 6638) 
v_mul_f32 v19, v7, v15 // (op 6638) 
v_mul_f32 v20, v7, v16 // (op 6638) 
v_mul_f32 v21, v7, v17 // (op 6638) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6610) 
v_mul_f32 v15, v0, v11 // (op 6610) 
v_mul_f32 v16, v0, v12 // (op 6610) 
v_mul_f32 v17, v0, v13 // (op 6610) 

v_add_f32 v10, v14, v18 // (op 6611) 
v_add_f32 v11, v15, v19 // (op 6611) 
v_add_f32 v12, v16, v20 // (op 6611) 
v_add_f32 v13, v17, v21 // (op 6611) 




s_lshl_b32 s0, s4, 6 // (op 6593) 
s_add_u32 s0, s0, 128 // (op 6593) 
v_add_u32 v3, v9, s0 // (op 6593) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6593) Store value




s_lshl_b32 s0, s2, 7 // (op 6616) 
s_add_u32 s0, s0, 128 // (op 6616) 
v_add_u32 v3, v8, s0 // (op 6616) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6616) Load value

v_accvgpr_read v14, a72 // (op 6640) 
v_accvgpr_read v15, a73 // (op 6640) 
v_accvgpr_read v16, a74 // (op 6640) 
v_accvgpr_read v17, a75 // (op 6640) 
v_mul_f32 v18, v7, v14 // (op 6640) 
v_mul_f32 v19, v7, v15 // (op 6640) 
v_mul_f32 v20, v7, v16 // (op 6640) 
v_mul_f32 v21, v7, v17 // (op 6640) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6614) 
v_mul_f32 v15, v0, v11 // (op 6614) 
v_mul_f32 v16, v0, v12 // (op 6614) 
v_mul_f32 v17, v0, v13 // (op 6614) 

v_add_f32 v10, v14, v18 // (op 6615) 
v_add_f32 v11, v15, v19 // (op 6615) 
v_add_f32 v12, v16, v20 // (op 6615) 
v_add_f32 v13, v17, v21 // (op 6615) 




s_lshl_b32 s0, s4, 7 // (op 6595) 
s_add_u32 s0, s0, 128 // (op 6595) 
v_add_u32 v3, v9, s0 // (op 6595) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6595) Store value




s_lshl_b32 s0, s2, 6 // (op 6620) 
s_mul_i32 s1, 3, s0 // (op 6620) 
s_add_u32 s1, s1, 128 // (op 6620) 

v_add_u32 v3, v8, s1 // (op 6620) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6620) Load value

v_accvgpr_read v14, a76 // (op 6642) 
v_accvgpr_read v15, a77 // (op 6642) 
v_accvgpr_read v16, a78 // (op 6642) 
v_accvgpr_read v17, a79 // (op 6642) 
v_mul_f32 v18, v7, v14 // (op 6642) 
v_mul_f32 v19, v7, v15 // (op 6642) 
v_mul_f32 v20, v7, v16 // (op 6642) 
v_mul_f32 v21, v7, v17 // (op 6642) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6618) 
v_mul_f32 v15, v0, v11 // (op 6618) 
v_mul_f32 v16, v0, v12 // (op 6618) 
v_mul_f32 v17, v0, v13 // (op 6618) 

v_add_f32 v10, v14, v18 // (op 6619) 
v_add_f32 v11, v15, v19 // (op 6619) 
v_add_f32 v12, v16, v20 // (op 6619) 
v_add_f32 v13, v17, v21 // (op 6619) 




s_lshl_b32 s0, s4, 6 // (op 6597) 
s_mul_i32 s1, 3, s0 // (op 6597) 
s_add_u32 s1, s1, 128 // (op 6597) 

v_add_u32 v3, v9, s1 // (op 6597) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6597) Store value




s_lshl_b32 s0, s2, 8 // (op 6624) 
s_add_u32 s0, s0, 128 // (op 6624) 
v_add_u32 v3, v8, s0 // (op 6624) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6624) Load value

v_accvgpr_read v14, a80 // (op 6644) 
v_accvgpr_read v15, a81 // (op 6644) 
v_accvgpr_read v16, a82 // (op 6644) 
v_accvgpr_read v17, a83 // (op 6644) 
v_mul_f32 v18, v7, v14 // (op 6644) 
v_mul_f32 v19, v7, v15 // (op 6644) 
v_mul_f32 v20, v7, v16 // (op 6644) 
v_mul_f32 v21, v7, v17 // (op 6644) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6622) 
v_mul_f32 v15, v0, v11 // (op 6622) 
v_mul_f32 v16, v0, v12 // (op 6622) 
v_mul_f32 v17, v0, v13 // (op 6622) 

v_add_f32 v10, v14, v18 // (op 6623) 
v_add_f32 v11, v15, v19 // (op 6623) 
v_add_f32 v12, v16, v20 // (op 6623) 
v_add_f32 v13, v17, v21 // (op 6623) 




s_lshl_b32 s0, s4, 8 // (op 6599) 
s_add_u32 s0, s0, 128 // (op 6599) 
v_add_u32 v3, v9, s0 // (op 6599) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6599) Store value




s_lshl_b32 s0, s2, 6 // (op 6628) 
s_mul_i32 s1, 5, s0 // (op 6628) 
s_add_u32 s1, s1, 128 // (op 6628) 

v_add_u32 v3, v8, s1 // (op 6628) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6628) Load value

v_accvgpr_read v14, a84 // (op 6646) 
v_accvgpr_read v15, a85 // (op 6646) 
v_accvgpr_read v16, a86 // (op 6646) 
v_accvgpr_read v17, a87 // (op 6646) 
v_mul_f32 v18, v7, v14 // (op 6646) 
v_mul_f32 v19, v7, v15 // (op 6646) 
v_mul_f32 v20, v7, v16 // (op 6646) 
v_mul_f32 v21, v7, v17 // (op 6646) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6626) 
v_mul_f32 v15, v0, v11 // (op 6626) 
v_mul_f32 v16, v0, v12 // (op 6626) 
v_mul_f32 v17, v0, v13 // (op 6626) 

v_add_f32 v10, v14, v18 // (op 6627) 
v_add_f32 v11, v15, v19 // (op 6627) 
v_add_f32 v12, v16, v20 // (op 6627) 
v_add_f32 v13, v17, v21 // (op 6627) 




s_lshl_b32 s0, s4, 6 // (op 6601) 
s_mul_i32 s1, 5, s0 // (op 6601) 
s_add_u32 s1, s1, 128 // (op 6601) 

v_add_u32 v3, v9, s1 // (op 6601) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6601) Store value




s_lshl_b32 s0, s2, 6 // (op 6632) 
s_mul_i32 s1, 6, s0 // (op 6632) 
s_add_u32 s1, s1, 128 // (op 6632) 

v_add_u32 v3, v8, s1 // (op 6632) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6632) Load value

v_accvgpr_read v14, a88 // (op 6648) 
v_accvgpr_read v15, a89 // (op 6648) 
v_accvgpr_read v16, a90 // (op 6648) 
v_accvgpr_read v17, a91 // (op 6648) 
v_mul_f32 v18, v7, v14 // (op 6648) 
v_mul_f32 v19, v7, v15 // (op 6648) 
v_mul_f32 v20, v7, v16 // (op 6648) 
v_mul_f32 v21, v7, v17 // (op 6648) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6630) 
v_mul_f32 v15, v0, v11 // (op 6630) 
v_mul_f32 v16, v0, v12 // (op 6630) 
v_mul_f32 v17, v0, v13 // (op 6630) 

v_add_f32 v10, v14, v18 // (op 6631) 
v_add_f32 v11, v15, v19 // (op 6631) 
v_add_f32 v12, v16, v20 // (op 6631) 
v_add_f32 v13, v17, v21 // (op 6631) 




s_lshl_b32 s0, s4, 6 // (op 6603) 
s_mul_i32 s1, 6, s0 // (op 6603) 
s_add_u32 s1, s1, 128 // (op 6603) 

v_add_u32 v3, v9, s1 // (op 6603) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6603) Store value




s_lshl_b32 s0, s2, 6 // (op 6636) 
s_mul_i32 s1, 7, s0 // (op 6636) 
s_add_u32 s1, s1, 128 // (op 6636) 

v_add_u32 v3, v8, s1 // (op 6636) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6636) Load value

v_accvgpr_read v14, a92 // (op 6650) 
v_accvgpr_read v15, a93 // (op 6650) 
v_accvgpr_read v16, a94 // (op 6650) 
v_accvgpr_read v17, a95 // (op 6650) 
v_mul_f32 v18, v7, v14 // (op 6650) 
v_mul_f32 v19, v7, v15 // (op 6650) 
v_mul_f32 v20, v7, v16 // (op 6650) 
v_mul_f32 v21, v7, v17 // (op 6650) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6634) 
v_mul_f32 v15, v0, v11 // (op 6634) 
v_mul_f32 v16, v0, v12 // (op 6634) 
v_mul_f32 v17, v0, v13 // (op 6634) 

v_add_f32 v10, v14, v18 // (op 6635) 
v_add_f32 v11, v15, v19 // (op 6635) 
v_add_f32 v12, v16, v20 // (op 6635) 
v_add_f32 v13, v17, v21 // (op 6635) 




s_lshl_b32 s0, s4, 6 // (op 6605) 
s_mul_i32 s1, 7, s0 // (op 6605) 
s_add_u32 s1, s1, 128 // (op 6605) 

v_add_u32 v3, v9, s1 // (op 6605) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6605) Store value




v_add_u32 v3, 64, v8 // (op 6453) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6453) Load value

v_accvgpr_read v14, a32 // (op 6433) 
v_accvgpr_read v15, a33 // (op 6433) 
v_accvgpr_read v16, a34 // (op 6433) 
v_accvgpr_read v17, a35 // (op 6433) 
v_mul_f32 v18, v7, v14 // (op 6433) 
v_mul_f32 v19, v7, v15 // (op 6433) 
v_mul_f32 v20, v7, v16 // (op 6433) 
v_mul_f32 v21, v7, v17 // (op 6433) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6452) 
v_mul_f32 v15, v0, v11 // (op 6452) 
v_mul_f32 v16, v0, v12 // (op 6452) 
v_mul_f32 v17, v0, v13 // (op 6452) 

v_add_f32 v10, v14, v18 // (op 6434) 
v_add_f32 v11, v15, v19 // (op 6434) 
v_add_f32 v12, v16, v20 // (op 6434) 
v_add_f32 v13, v17, v21 // (op 6434) 




v_add_u32 v3, 64, v9 // (op 6436) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6436) Store value




s_lshl_b32 s0, s2, 6 // (op 6457) 
s_add_u32 s0, s0, 64 // (op 6457) 
v_add_u32 v3, v8, s0 // (op 6457) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6457) Load value

v_accvgpr_read v14, a36 // (op 6483) 
v_accvgpr_read v15, a37 // (op 6483) 
v_accvgpr_read v16, a38 // (op 6483) 
v_accvgpr_read v17, a39 // (op 6483) 
v_mul_f32 v18, v7, v14 // (op 6483) 
v_mul_f32 v19, v7, v15 // (op 6483) 
v_mul_f32 v20, v7, v16 // (op 6483) 
v_mul_f32 v21, v7, v17 // (op 6483) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6455) 
v_mul_f32 v15, v0, v11 // (op 6455) 
v_mul_f32 v16, v0, v12 // (op 6455) 
v_mul_f32 v17, v0, v13 // (op 6455) 

v_add_f32 v10, v14, v18 // (op 6456) 
v_add_f32 v11, v15, v19 // (op 6456) 
v_add_f32 v12, v16, v20 // (op 6456) 
v_add_f32 v13, v17, v21 // (op 6456) 




s_lshl_b32 s0, s4, 6 // (op 6438) 
s_add_u32 s0, s0, 64 // (op 6438) 
v_add_u32 v3, v9, s0 // (op 6438) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6438) Store value




s_lshl_b32 s0, s2, 7 // (op 6461) 
s_add_u32 s0, s0, 64 // (op 6461) 
v_add_u32 v3, v8, s0 // (op 6461) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6461) Load value

v_accvgpr_read v14, a40 // (op 6485) 
v_accvgpr_read v15, a41 // (op 6485) 
v_accvgpr_read v16, a42 // (op 6485) 
v_accvgpr_read v17, a43 // (op 6485) 
v_mul_f32 v18, v7, v14 // (op 6485) 
v_mul_f32 v19, v7, v15 // (op 6485) 
v_mul_f32 v20, v7, v16 // (op 6485) 
v_mul_f32 v21, v7, v17 // (op 6485) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6459) 
v_mul_f32 v15, v0, v11 // (op 6459) 
v_mul_f32 v16, v0, v12 // (op 6459) 
v_mul_f32 v17, v0, v13 // (op 6459) 

v_add_f32 v10, v14, v18 // (op 6460) 
v_add_f32 v11, v15, v19 // (op 6460) 
v_add_f32 v12, v16, v20 // (op 6460) 
v_add_f32 v13, v17, v21 // (op 6460) 




s_lshl_b32 s0, s4, 7 // (op 6440) 
s_add_u32 s0, s0, 64 // (op 6440) 
v_add_u32 v3, v9, s0 // (op 6440) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6440) Store value




s_lshl_b32 s0, s2, 6 // (op 6465) 
s_mul_i32 s1, 3, s0 // (op 6465) 
s_add_u32 s1, s1, 64 // (op 6465) 

v_add_u32 v3, v8, s1 // (op 6465) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6465) Load value

v_accvgpr_read v14, a44 // (op 6487) 
v_accvgpr_read v15, a45 // (op 6487) 
v_accvgpr_read v16, a46 // (op 6487) 
v_accvgpr_read v17, a47 // (op 6487) 
v_mul_f32 v18, v7, v14 // (op 6487) 
v_mul_f32 v19, v7, v15 // (op 6487) 
v_mul_f32 v20, v7, v16 // (op 6487) 
v_mul_f32 v21, v7, v17 // (op 6487) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6463) 
v_mul_f32 v15, v0, v11 // (op 6463) 
v_mul_f32 v16, v0, v12 // (op 6463) 
v_mul_f32 v17, v0, v13 // (op 6463) 

v_add_f32 v10, v14, v18 // (op 6464) 
v_add_f32 v11, v15, v19 // (op 6464) 
v_add_f32 v12, v16, v20 // (op 6464) 
v_add_f32 v13, v17, v21 // (op 6464) 




s_lshl_b32 s0, s4, 6 // (op 6442) 
s_mul_i32 s1, 3, s0 // (op 6442) 
s_add_u32 s1, s1, 64 // (op 6442) 

v_add_u32 v3, v9, s1 // (op 6442) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6442) Store value




s_lshl_b32 s0, s2, 8 // (op 6469) 
s_add_u32 s0, s0, 64 // (op 6469) 
v_add_u32 v3, v8, s0 // (op 6469) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6469) Load value

v_accvgpr_read v14, a48 // (op 6489) 
v_accvgpr_read v15, a49 // (op 6489) 
v_accvgpr_read v16, a50 // (op 6489) 
v_accvgpr_read v17, a51 // (op 6489) 
v_mul_f32 v18, v7, v14 // (op 6489) 
v_mul_f32 v19, v7, v15 // (op 6489) 
v_mul_f32 v20, v7, v16 // (op 6489) 
v_mul_f32 v21, v7, v17 // (op 6489) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6467) 
v_mul_f32 v15, v0, v11 // (op 6467) 
v_mul_f32 v16, v0, v12 // (op 6467) 
v_mul_f32 v17, v0, v13 // (op 6467) 

v_add_f32 v10, v14, v18 // (op 6468) 
v_add_f32 v11, v15, v19 // (op 6468) 
v_add_f32 v12, v16, v20 // (op 6468) 
v_add_f32 v13, v17, v21 // (op 6468) 




s_lshl_b32 s0, s4, 8 // (op 6444) 
s_add_u32 s0, s0, 64 // (op 6444) 
v_add_u32 v3, v9, s0 // (op 6444) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6444) Store value




s_lshl_b32 s0, s2, 6 // (op 6473) 
s_mul_i32 s1, 5, s0 // (op 6473) 
s_add_u32 s1, s1, 64 // (op 6473) 

v_add_u32 v3, v8, s1 // (op 6473) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6473) Load value

v_accvgpr_read v14, a52 // (op 6491) 
v_accvgpr_read v15, a53 // (op 6491) 
v_accvgpr_read v16, a54 // (op 6491) 
v_accvgpr_read v17, a55 // (op 6491) 
v_mul_f32 v18, v7, v14 // (op 6491) 
v_mul_f32 v19, v7, v15 // (op 6491) 
v_mul_f32 v20, v7, v16 // (op 6491) 
v_mul_f32 v21, v7, v17 // (op 6491) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6471) 
v_mul_f32 v15, v0, v11 // (op 6471) 
v_mul_f32 v16, v0, v12 // (op 6471) 
v_mul_f32 v17, v0, v13 // (op 6471) 

v_add_f32 v10, v14, v18 // (op 6472) 
v_add_f32 v11, v15, v19 // (op 6472) 
v_add_f32 v12, v16, v20 // (op 6472) 
v_add_f32 v13, v17, v21 // (op 6472) 




s_lshl_b32 s0, s4, 6 // (op 6446) 
s_mul_i32 s1, 5, s0 // (op 6446) 
s_add_u32 s1, s1, 64 // (op 6446) 

v_add_u32 v3, v9, s1 // (op 6446) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6446) Store value




s_lshl_b32 s0, s2, 6 // (op 6477) 
s_mul_i32 s1, 6, s0 // (op 6477) 
s_add_u32 s1, s1, 64 // (op 6477) 

v_add_u32 v3, v8, s1 // (op 6477) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6477) Load value

v_accvgpr_read v14, a56 // (op 6493) 
v_accvgpr_read v15, a57 // (op 6493) 
v_accvgpr_read v16, a58 // (op 6493) 
v_accvgpr_read v17, a59 // (op 6493) 
v_mul_f32 v18, v7, v14 // (op 6493) 
v_mul_f32 v19, v7, v15 // (op 6493) 
v_mul_f32 v20, v7, v16 // (op 6493) 
v_mul_f32 v21, v7, v17 // (op 6493) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6475) 
v_mul_f32 v15, v0, v11 // (op 6475) 
v_mul_f32 v16, v0, v12 // (op 6475) 
v_mul_f32 v17, v0, v13 // (op 6475) 

v_add_f32 v10, v14, v18 // (op 6476) 
v_add_f32 v11, v15, v19 // (op 6476) 
v_add_f32 v12, v16, v20 // (op 6476) 
v_add_f32 v13, v17, v21 // (op 6476) 




s_lshl_b32 s0, s4, 6 // (op 6448) 
s_mul_i32 s1, 6, s0 // (op 6448) 
s_add_u32 s1, s1, 64 // (op 6448) 

v_add_u32 v3, v9, s1 // (op 6448) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6448) Store value




s_lshl_b32 s0, s2, 6 // (op 6481) 
s_mul_i32 s1, 7, s0 // (op 6481) 
s_add_u32 s1, s1, 64 // (op 6481) 

v_add_u32 v3, v8, s1 // (op 6481) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6481) Load value

v_accvgpr_read v14, a60 // (op 6495) 
v_accvgpr_read v15, a61 // (op 6495) 
v_accvgpr_read v16, a62 // (op 6495) 
v_accvgpr_read v17, a63 // (op 6495) 
v_mul_f32 v18, v7, v14 // (op 6495) 
v_mul_f32 v19, v7, v15 // (op 6495) 
v_mul_f32 v20, v7, v16 // (op 6495) 
v_mul_f32 v21, v7, v17 // (op 6495) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6479) 
v_mul_f32 v15, v0, v11 // (op 6479) 
v_mul_f32 v16, v0, v12 // (op 6479) 
v_mul_f32 v17, v0, v13 // (op 6479) 

v_add_f32 v10, v14, v18 // (op 6480) 
v_add_f32 v11, v15, v19 // (op 6480) 
v_add_f32 v12, v16, v20 // (op 6480) 
v_add_f32 v13, v17, v21 // (op 6480) 




s_lshl_b32 s0, s4, 6 // (op 6450) 
s_mul_i32 s1, 7, s0 // (op 6450) 
s_add_u32 s1, s1, 64 // (op 6450) 

v_add_u32 v3, v9, s1 // (op 6450) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6450) Store value




v_mov_b32 v3, v8 // (op 126) reg expression

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 126) Load value

v_accvgpr_read v14, a0 // (op 123) 
v_accvgpr_read v15, a1 // (op 123) 
v_accvgpr_read v16, a2 // (op 123) 
v_accvgpr_read v17, a3 // (op 123) 
v_mul_f32 v18, v7, v14 // (op 123) 
v_mul_f32 v19, v7, v15 // (op 123) 
v_mul_f32 v20, v7, v16 // (op 123) 
v_mul_f32 v21, v7, v17 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 127) 
v_mul_f32 v15, v0, v11 // (op 127) 
v_mul_f32 v16, v0, v12 // (op 127) 
v_mul_f32 v17, v0, v13 // (op 127) 

v_add_f32 v10, v14, v18 // (op 124) 
v_add_f32 v11, v15, v19 // (op 124) 
v_add_f32 v12, v16, v20 // (op 124) 
v_add_f32 v13, v17, v21 // (op 124) 




v_mov_b32 v3, v9 // (op 125) reg expression

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 125) Store value




s_lshl_b32 s0, s2, 6 // (op 6271) 
v_add_u32 v3, v8, s0 // (op 6271) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6271) Load value

v_accvgpr_read v14, a4 // (op 6268) 
v_accvgpr_read v15, a5 // (op 6268) 
v_accvgpr_read v16, a6 // (op 6268) 
v_accvgpr_read v17, a7 // (op 6268) 
v_mul_f32 v18, v7, v14 // (op 6268) 
v_mul_f32 v19, v7, v15 // (op 6268) 
v_mul_f32 v20, v7, v16 // (op 6268) 
v_mul_f32 v21, v7, v17 // (op 6268) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6272) 
v_mul_f32 v15, v0, v11 // (op 6272) 
v_mul_f32 v16, v0, v12 // (op 6272) 
v_mul_f32 v17, v0, v13 // (op 6272) 

v_add_f32 v10, v14, v18 // (op 6269) 
v_add_f32 v11, v15, v19 // (op 6269) 
v_add_f32 v12, v16, v20 // (op 6269) 
v_add_f32 v13, v17, v21 // (op 6269) 




s_lshl_b32 s0, s4, 6 // (op 6270) 
v_add_u32 v3, v9, s0 // (op 6270) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6270) Store value




s_lshl_b32 s0, s2, 7 // (op 6283) 
v_add_u32 v3, v8, s0 // (op 6283) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6283) Load value

v_accvgpr_read v14, a8 // (op 6280) 
v_accvgpr_read v15, a9 // (op 6280) 
v_accvgpr_read v16, a10 // (op 6280) 
v_accvgpr_read v17, a11 // (op 6280) 
v_mul_f32 v18, v7, v14 // (op 6280) 
v_mul_f32 v19, v7, v15 // (op 6280) 
v_mul_f32 v20, v7, v16 // (op 6280) 
v_mul_f32 v21, v7, v17 // (op 6280) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6284) 
v_mul_f32 v15, v0, v11 // (op 6284) 
v_mul_f32 v16, v0, v12 // (op 6284) 
v_mul_f32 v17, v0, v13 // (op 6284) 

v_add_f32 v10, v14, v18 // (op 6281) 
v_add_f32 v11, v15, v19 // (op 6281) 
v_add_f32 v12, v16, v20 // (op 6281) 
v_add_f32 v13, v17, v21 // (op 6281) 




s_lshl_b32 s0, s4, 7 // (op 6282) 
v_add_u32 v3, v9, s0 // (op 6282) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6282) Store value




s_lshl_b32 s0, s2, 6 // (op 6295) 
s_mul_i32 s1, 3, s0 // (op 6295) 

v_add_u32 v3, v8, s1 // (op 6295) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6295) Load value

v_accvgpr_read v14, a12 // (op 6292) 
v_accvgpr_read v15, a13 // (op 6292) 
v_accvgpr_read v16, a14 // (op 6292) 
v_accvgpr_read v17, a15 // (op 6292) 
v_mul_f32 v18, v7, v14 // (op 6292) 
v_mul_f32 v19, v7, v15 // (op 6292) 
v_mul_f32 v20, v7, v16 // (op 6292) 
v_mul_f32 v21, v7, v17 // (op 6292) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6296) 
v_mul_f32 v15, v0, v11 // (op 6296) 
v_mul_f32 v16, v0, v12 // (op 6296) 
v_mul_f32 v17, v0, v13 // (op 6296) 

v_add_f32 v10, v14, v18 // (op 6293) 
v_add_f32 v11, v15, v19 // (op 6293) 
v_add_f32 v12, v16, v20 // (op 6293) 
v_add_f32 v13, v17, v21 // (op 6293) 




s_lshl_b32 s0, s4, 6 // (op 6294) 
s_mul_i32 s1, 3, s0 // (op 6294) 

v_add_u32 v3, v9, s1 // (op 6294) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6294) Store value




s_lshl_b32 s0, s2, 8 // (op 6307) 
v_add_u32 v3, v8, s0 // (op 6307) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6307) Load value

v_accvgpr_read v14, a16 // (op 6304) 
v_accvgpr_read v15, a17 // (op 6304) 
v_accvgpr_read v16, a18 // (op 6304) 
v_accvgpr_read v17, a19 // (op 6304) 
v_mul_f32 v18, v7, v14 // (op 6304) 
v_mul_f32 v19, v7, v15 // (op 6304) 
v_mul_f32 v20, v7, v16 // (op 6304) 
v_mul_f32 v21, v7, v17 // (op 6304) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6308) 
v_mul_f32 v15, v0, v11 // (op 6308) 
v_mul_f32 v16, v0, v12 // (op 6308) 
v_mul_f32 v17, v0, v13 // (op 6308) 

v_add_f32 v10, v14, v18 // (op 6305) 
v_add_f32 v11, v15, v19 // (op 6305) 
v_add_f32 v12, v16, v20 // (op 6305) 
v_add_f32 v13, v17, v21 // (op 6305) 




s_lshl_b32 s0, s4, 8 // (op 6306) 
v_add_u32 v3, v9, s0 // (op 6306) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6306) Store value




s_lshl_b32 s0, s2, 6 // (op 6319) 
s_mul_i32 s1, 5, s0 // (op 6319) 

v_add_u32 v3, v8, s1 // (op 6319) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6319) Load value

v_accvgpr_read v14, a20 // (op 6316) 
v_accvgpr_read v15, a21 // (op 6316) 
v_accvgpr_read v16, a22 // (op 6316) 
v_accvgpr_read v17, a23 // (op 6316) 
v_mul_f32 v18, v7, v14 // (op 6316) 
v_mul_f32 v19, v7, v15 // (op 6316) 
v_mul_f32 v20, v7, v16 // (op 6316) 
v_mul_f32 v21, v7, v17 // (op 6316) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6320) 
v_mul_f32 v15, v0, v11 // (op 6320) 
v_mul_f32 v16, v0, v12 // (op 6320) 
v_mul_f32 v17, v0, v13 // (op 6320) 

v_add_f32 v10, v14, v18 // (op 6317) 
v_add_f32 v11, v15, v19 // (op 6317) 
v_add_f32 v12, v16, v20 // (op 6317) 
v_add_f32 v13, v17, v21 // (op 6317) 




s_lshl_b32 s0, s4, 6 // (op 6318) 
s_mul_i32 s1, 5, s0 // (op 6318) 

v_add_u32 v3, v9, s1 // (op 6318) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6318) Store value




s_lshl_b32 s0, s2, 6 // (op 6331) 
s_mul_i32 s1, 6, s0 // (op 6331) 

v_add_u32 v3, v8, s1 // (op 6331) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6331) Load value

v_accvgpr_read v14, a24 // (op 6328) 
v_accvgpr_read v15, a25 // (op 6328) 
v_accvgpr_read v16, a26 // (op 6328) 
v_accvgpr_read v17, a27 // (op 6328) 
v_mul_f32 v18, v7, v14 // (op 6328) 
v_mul_f32 v19, v7, v15 // (op 6328) 
v_mul_f32 v20, v7, v16 // (op 6328) 
v_mul_f32 v21, v7, v17 // (op 6328) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6332) 
v_mul_f32 v15, v0, v11 // (op 6332) 
v_mul_f32 v16, v0, v12 // (op 6332) 
v_mul_f32 v17, v0, v13 // (op 6332) 

v_add_f32 v10, v14, v18 // (op 6329) 
v_add_f32 v11, v15, v19 // (op 6329) 
v_add_f32 v12, v16, v20 // (op 6329) 
v_add_f32 v13, v17, v21 // (op 6329) 




s_lshl_b32 s0, s4, 6 // (op 6330) 
s_mul_i32 s1, 6, s0 // (op 6330) 

v_add_u32 v3, v9, s1 // (op 6330) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6330) Store value




s_lshl_b32 s0, s2, 6 // (op 6343) 
s_mul_i32 s1, 7, s0 // (op 6343) 

v_add_u32 v3, v8, s1 // (op 6343) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6343) Load value

v_accvgpr_read v14, a28 // (op 6340) 
v_accvgpr_read v15, a29 // (op 6340) 
v_accvgpr_read v16, a30 // (op 6340) 
v_accvgpr_read v17, a31 // (op 6340) 
v_mul_f32 v18, v7, v14 // (op 6340) 
v_mul_f32 v19, v7, v15 // (op 6340) 
v_mul_f32 v20, v7, v16 // (op 6340) 
v_mul_f32 v21, v7, v17 // (op 6340) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6344) 
v_mul_f32 v15, v0, v11 // (op 6344) 
v_mul_f32 v16, v0, v12 // (op 6344) 
v_mul_f32 v17, v0, v13 // (op 6344) 

v_add_f32 v10, v14, v18 // (op 6341) 
v_add_f32 v11, v15, v19 // (op 6341) 
v_add_f32 v12, v16, v20 // (op 6341) 
v_add_f32 v13, v17, v21 // (op 6341) 




s_lshl_b32 s0, s4, 6 // (op 6342) 
s_mul_i32 s1, 7, s0 // (op 6342) 

v_add_u32 v3, v9, s1 // (op 6342) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6342) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ConditionalBottom_0_beta_114:








































s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 348
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 98304 // lds bytes
  .amdhsa_accum_offset 220
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
  - .name: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 98304
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 218
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

