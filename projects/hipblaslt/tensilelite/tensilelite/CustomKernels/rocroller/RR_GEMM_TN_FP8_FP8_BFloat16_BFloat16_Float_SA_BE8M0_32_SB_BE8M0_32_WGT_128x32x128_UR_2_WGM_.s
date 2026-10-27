// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 2085) 
s_mul_hi_i32 s1, s16, s10 // (op 2272) 
v_lshrrev_b32 v0, 6, v1 // (op 2198) 
v_and_b32 v3, 1, v0 // (op 2198) 

v_and_b32 v0, 15, v1 // (op 2198) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 2198) 


v_and_b32 v0, 48, v1 // (op 2198) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 2198) 


v_lshrrev_b32 v0, 6, v1 // (op 2357) 
v_and_b32 v4, 1, v0 // (op 2357) 

v_and_b32 v0, 15, v1 // (op 2357) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 2357) 


v_and_b32 v0, 48, v1 // (op 2357) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 2357) 


v_lshrrev_b32 v0, 7, v1 // (op 2085) 
s_add_i32 s18, s0, s16 // (op 2085) 

v_and_b32 v5, 63, v1 // (op 2085) 
s_ashr_i32 s0, s18, 31 // (op 2085) 
v_lshrrev_b32 v6, 4, v5 // (op 2085) 
s_cmp_eq_i32 s10, 0 // (op 2085) 
s_mov_b32 s19, scc // (op 2085) 
v_lshlrev_b32 v7, 2, v6 // (op 2085) 

s_cmp_eq_i32 s19, 0 // (op 2085) 
s_cselect_b32 s49, 0, -1 // (op 2085) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 2085) 


s_and_b32 s19, s13, 31 // (op 2085) 
v_lshrrev_b32 v0, 2, v5 // (op 2085) 

v_and_b32 v5, 3, v0 // (op 2085) 

v_lshrrev_b32 v0, 7, v1 // (op 2146) 
v_and_b32 v7, 15, v1 // (op 2146) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 2146) 


v_and_b32 v0, 48, v1 // (op 2146) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 2146) 


v_lshrrev_b32 v0, 7, v1 // (op 2256) 
v_and_b32 v8, 15, v1 // (op 2256) 
v_lshl_add_u32 v9, v0, 4, v8 // (op 2256) 


v_and_b32 v0, 63, v1 // (op 2256) 
v_lshrrev_b32 v8, 4, v0 // (op 2256) 

v_lshl_add_u32 v0, v9, 2, v8 // (op 2256) 


v_lshrrev_b32 v8, 7, v1 // (op 2272) 
v_and_b32 v9, 63, v1 // (op 2272) 
v_lshrrev_b32 v10, 7, v1 // (op 2318) 
v_and_b32 v11, 15, v1 // (op 2318) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 2318) 


v_and_b32 v10, 48, v1 // (op 2318) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 2318) 


v_lshrrev_b32 v10, 7, v1 // (op 2386) 
v_and_b32 v12, 15, v1 // (op 2386) 
v_lshl_add_u32 v13, v10, 4, v12 // (op 2386) 


v_and_b32 v10, 63, v1 // (op 2386) 
v_lshrrev_b32 v12, 4, v10 // (op 2386) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 2386) 


s_lshl_b32 s50, 1, s19 // (op 2085) 
v_and_b32 v12, 3, v1 // (op 2085) 
s_add_i32 s51, s50, s49 // (op 2085) 


s_and_b32 s49, s0, s51 // (op 2085) 


s_add_i32 s0, s1, s16 // (op 2272) 

s_ashr_i32 s1, s0, 31 // (op 2272) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 2085) 


s_add_i32 s50, s18, s49 // (op 2085) 


v_lshrrev_b32 v5, 6, v1 // (op 2085) 
s_ashr_i32 s18, s50, s19 // (op 2085) 


v_and_b32 v12, 1, v5 // (op 2085) 

s_lshl_b32 s19, s13, 24 // (op 2085) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 2085) 


s_ashr_i32 s49, s19, 31 // (op 2085) 

s_xor_b32 s19, s18, s49 // (op 2085) 

s_sub_i32 s18, s19, s49 // (op 2085) 


s_mul_hi_i32 s19, s18, s14 // (op 2085) 
s_add_i32 s49, s19, s18 // (op 2085) 

s_ashr_i32 s19, s49, 31 // (op 2085) 
s_cmp_eq_i32 s14, 0 // (op 2085) 
s_mov_b32 s50, scc // (op 2085) 
s_cmp_eq_i32 s50, 0 // (op 2085) 
s_cselect_b32 s51, 0, -1 // (op 2085) 

s_and_b32 s50, s15, 31 // (op 2085) 
v_lshrrev_b32 v12, 4, v9 // (op 2272) 
s_cmp_eq_i32 s10, 0 // (op 2272) 
s_mov_b32 s52, scc // (op 2272) 
v_lshlrev_b32 v13, 2, v12 // (op 2272) 

s_cmp_eq_i32 s52, 0 // (op 2272) 
s_cselect_b32 s53, 0, -1 // (op 2272) 

s_lshl_b32 s52, 1, s50 // (op 2085) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 2272) 


v_lshrrev_b32 v8, 2, v9 // (op 2272) 

s_add_i32 s54, s52, s51 // (op 2085) 


s_and_b32 s51, s19, s54 // (op 2085) 


s_add_i32 s19, s49, s51 // (op 2085) 


s_ashr_i32 s49, s19, s50 // (op 2085) 


s_lshl_b32 s19, s15, 24 // (op 2085) 
s_ashr_i32 s50, s19, 31 // (op 2085) 

s_xor_b32 s19, s49, s50 // (op 2085) 

s_sub_i32 s49, s19, s50 // (op 2085) 


s_mul_i32 s19, s18, s11 // (op 2085) 
s_sub_u32 s50, s16, s19 // (op 2085) 

s_mul_i32 s19, s49, s11 // (op 2085) 
s_add_u32 s19, s19, s50 // (op 2085) 

s_mul_hi_i32 s50, s16, s9 // (op 2085) 
s_add_i32 s51, s50, s16 // (op 2085) 

s_ashr_i32 s50, s51, 31 // (op 2085) 
s_cmp_eq_i32 s9, 0 // (op 2085) 
s_mov_b32 s52, scc // (op 2085) 
s_cmp_eq_i32 s52, 0 // (op 2085) 
s_cselect_b32 s54, 0, -1 // (op 2085) 

s_and_b32 s52, s13, 31 // (op 2272) 
v_and_b32 v9, 3, v8 // (op 2272) 

s_lshl_b32 s55, 1, s52 // (op 2272) 
v_and_b32 v8, 3, v1 // (op 2272) 
s_add_i32 s56, s55, s53 // (op 2272) 


s_and_b32 s53, s1, s56 // (op 2272) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 2272) 


s_add_i32 s1, s0, s53 // (op 2272) 


v_lshrrev_b32 v8, 6, v1 // (op 2272) 
s_ashr_i32 s0, s1, s52 // (op 2272) 


v_and_b32 v9, 1, v8 // (op 2272) 

s_lshl_b32 s1, s13, 24 // (op 2272) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 2272) 


s_ashr_i32 s52, s1, 31 // (op 2272) 

s_xor_b32 s1, s0, s52 // (op 2272) 

s_sub_i32 s0, s1, s52 // (op 2272) 


s_mul_hi_i32 s1, s0, s14 // (op 2272) 
s_add_i32 s52, s1, s0 // (op 2272) 

s_ashr_i32 s1, s52, 31 // (op 2272) 
s_cmp_eq_i32 s14, 0 // (op 2272) 
s_mov_b32 s53, scc // (op 2272) 
s_cmp_eq_i32 s53, 0 // (op 2272) 
s_cselect_b32 s55, 0, -1 // (op 2272) 

s_and_b32 s53, s15, 31 // (op 2272) 
s_lshl_b32 s56, 1, s53 // (op 2272) 
s_add_i32 s57, s56, s55 // (op 2272) 


s_and_b32 s55, s1, s57 // (op 2272) 


s_add_i32 s1, s52, s55 // (op 2272) 


s_ashr_i32 s52, s1, s53 // (op 2272) 


s_lshl_b32 s1, s15, 24 // (op 2272) 
s_ashr_i32 s53, s1, 31 // (op 2272) 

s_xor_b32 s1, s52, s53 // (op 2272) 

s_sub_i32 s52, s1, s53 // (op 2272) 


s_mul_i32 s1, s0, s11 // (op 2272) 
s_sub_u32 s53, s16, s1 // (op 2272) 

s_mul_i32 s1, s52, s11 // (op 2272) 
s_add_u32 s1, s1, s53 // (op 2272) 

s_mul_hi_i32 s53, s16, s9 // (op 2272) 
s_add_i32 s55, s53, s16 // (op 2272) 

s_ashr_i32 s53, s55, 31 // (op 2272) 
s_cmp_eq_i32 s9, 0 // (op 2272) 
s_mov_b32 s56, scc // (op 2272) 
s_cmp_eq_i32 s56, 0 // (op 2272) 
s_cselect_b32 s57, 0, -1 // (op 2272) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 2085) 
s_lshl_b32 s58, 1, s56 // (op 2085) 
s_add_i32 s59, s58, s54 // (op 2085) 


s_and_b32 s54, s50, s59 // (op 2085) 


s_add_i32 s50, s51, s54 // (op 2085) 


s_ashr_i32 s51, s50, s56 // (op 2085) 


s_lshl_b32 s50, s45, 24 // (op 2085) 
s_ashr_i32 s54, s50, 31 // (op 2085) 

s_xor_b32 s50, s51, s54 // (op 2085) 

s_sub_i32 s51, s50, s54 // (op 2085) 


s_and_b32 s50, s45, 31 // (op 2272) 
s_lshl_b32 s54, 1, s50 // (op 2272) 
s_add_i32 s56, s54, s57 // (op 2272) 


s_and_b32 s54, s53, s56 // (op 2272) 


s_add_i32 s53, s55, s54 // (op 2272) 


s_ashr_i32 s54, s53, s50 // (op 2272) 


s_lshl_b32 s50, s45, 24 // (op 2272) 
s_ashr_i32 s53, s50, 31 // (op 2272) 

s_xor_b32 s50, s54, s53 // (op 2272) 

s_sub_i32 s54, s50, s53 // (op 2272) 


s_add_u32 s56, s4, 32 // (op 2085) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 2085) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2085) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2085) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 2085) 

s_mul_i32 s50, s11, s56 // (op 2085) 
s_mul_i32 s53, s8, s50 // (op 2085) 

s_mul_i32 s50, s51, s53 // (op 2085) 

s_sub_u32 s53, s16, s50 // (op 2085) 

s_mov_b32 s50, s53 // (op 2085) 
s_mul_hi_i32 s55, s50, s12 // (op 2085) 
s_add_i32 s58, s55, s50 // (op 2085) 


s_ashr_i32 s50, s58, 31 // (op 2085) 
s_cmp_eq_i32 s12, 0 // (op 2085) 
s_mov_b32 s55, scc // (op 2085) 
s_cmp_eq_i32 s55, 0 // (op 2085) 
s_cselect_b32 s59, 0, -1 // (op 2085) 

s_and_b32 s55, s44, 31 // (op 2085) 
s_lshl_b32 s60, 1, s55 // (op 2085) 
s_add_i32 s61, s60, s59 // (op 2085) 


s_and_b32 s59, s50, s61 // (op 2085) 


s_add_i32 s50, s58, s59 // (op 2085) 


s_ashr_i32 s58, s50, s55 // (op 2085) 


s_lshl_b32 s50, s44, 24 // (op 2085) 
s_ashr_i32 s55, s50, 31 // (op 2085) 

s_xor_b32 s50, s58, s55 // (op 2085) 

s_sub_i32 s58, s50, s55 // (op 2085) 


s_mul_i32 s50, s46, s11 // (op 2085) 
s_add_u32 s60, s6, 128 // (op 2085) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 2085) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 2085) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 2085) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 7 // (op 2085) 

s_sub_i32 s55, s60, s50 // (op 2085) 

s_mul_i32 s59, s58, s55 // (op 2085) 

s_sub_u32 s55, s53, s59 // (op 2085) 


s_add_u32 s53, s55, s50 // (op 2085) 


s_cmp_eq_i32 s51, 0 // (op 2085) 
s_mov_b32 s50, scc // (op 2085) 

s_cmp_eq_i32 s50, 0 // (op 2085) 
s_cselect_b32 s51, s53, s19 // (op 2085) 


s_mul_i32 s19, s49, s56 // (op 2085) 


v_lshl_add_u32 v9, s51, 7, v6 // (op 2085) 


s_sub_i32 s49, s18, s19 // (op 2085) 


s_cmp_eq_i32 s50, 0 // (op 2085) 
s_cselect_b32 s51, s58, s49 // (op 2085) 



v_lshl_add_u32 v6, s51, 5, v5 // (op 2085) 


v_mul_lo_u32 v5, v6, s2 // (op 2085) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 2085) 


s_add_u32 s18, s4, 32 // (op 2272) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 2272) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 2272) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 2272) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 2272) 

s_mul_i32 s53, s11, s18 // (op 2272) 
s_mul_i32 s49, s8, s53 // (op 2272) 

s_mul_i32 s53, s54, s49 // (op 2272) 

s_sub_u32 s49, s16, s53 // (op 2272) 

s_mov_b32 s53, s49 // (op 2272) 
s_mul_hi_i32 s50, s53, s12 // (op 2272) 
s_add_i32 s51, s50, s53 // (op 2272) 


s_ashr_i32 s50, s51, 31 // (op 2272) 
s_cmp_eq_i32 s12, 0 // (op 2272) 
s_mov_b32 s53, scc // (op 2272) 
s_cmp_eq_i32 s53, 0 // (op 2272) 
s_cselect_b32 s55, 0, -1 // (op 2272) 

s_and_b32 s53, s44, 31 // (op 2272) 
s_lshl_b32 s56, 1, s53 // (op 2272) 
s_add_i32 s57, s56, s55 // (op 2272) 


s_and_b32 s55, s50, s57 // (op 2272) 


s_add_i32 s50, s51, s55 // (op 2272) 


s_ashr_i32 s51, s50, s53 // (op 2272) 


s_lshl_b32 s50, s44, 24 // (op 2272) 
s_ashr_i32 s53, s50, 31 // (op 2272) 

s_xor_b32 s50, s51, s53 // (op 2272) 

s_sub_i32 s51, s50, s53 // (op 2272) 


s_mul_i32 s50, s46, s11 // (op 2272) 
s_add_u32 s56, s6, 128 // (op 2272) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 2272) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2272) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2272) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 2272) 

s_sub_i32 s53, s56, s50 // (op 2272) 

s_mul_i32 s55, s51, s53 // (op 2272) 

s_sub_u32 s53, s49, s55 // (op 2272) 


s_add_u32 s49, s53, s50 // (op 2272) 


s_cmp_eq_i32 s54, 0 // (op 2272) 
s_mov_b32 s50, scc // (op 2272) 

s_cmp_eq_i32 s50, 0 // (op 2272) 
s_cselect_b32 s53, s49, s1 // (op 2272) 


s_mul_i32 s1, s52, s18 // (op 2272) 


v_lshl_add_u32 v5, s53, 7, v12 // (op 2272) 


s_sub_i32 s49, s0, s1 // (op 2272) 


s_cmp_eq_i32 s50, 0 // (op 2272) 
s_cselect_b32 s0, s51, s49 // (op 2272) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 2272) 


v_mul_lo_u32 v8, v9, s20 // (op 2272) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 2272) 


s_mul_i32 s1, s2, s5 // (op 2087) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 2087) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 2087) least significant
s_mul_hi_u32 s19, s2, s4 // (op 2087) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2087) most significant: sum
s_add_u32 s1, s1, s19 // (op 2087) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2087) 

s_mul_i32 s1, s20, s5 // (op 2274) most significant: low of low * high
s_mul_i32 s49, s21, s4 // (op 2274) most significant: low of high * low
s_mul_i32 s0, s20, s4 // (op 2274) least significant
s_mul_hi_u32 s50, s20, s4 // (op 2274) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 2274) most significant: sum
s_add_u32 s1, s1, s50 // (op 2274) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 2274) 

s_mov_b32 s52, s22 // (op 2087) reg expression
s_mov_b32 s53, s23 // (op 2087) reg expression
s_mov_b32 s54, s18 // (op 2087) reg expression
s_mov_b32 s55, 131072 // (op 2087) call()

s_mov_b32 s56, s24 // (op 2274) reg expression

s_mov_b32 s57, s25 // (op 2274) reg expression
s_mov_b32 s58, s50 // (op 2274) reg expression
s_mov_b32 s59, 131072 // (op 2274) call()


v_mov_b32 v5, s47 // (op 2) Move value
v_mov_b32 v8, s48 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 313) call()
v_accvgpr_write a5, 0.00000 // (op 313) call()
v_accvgpr_write a6, 0.00000 // (op 313) call()
v_accvgpr_write a7, 0.00000 // (op 313) call()
v_accvgpr_write a8, 0.00000 // (op 475) call()
v_accvgpr_write a9, 0.00000 // (op 475) call()
v_accvgpr_write a10, 0.00000 // (op 475) call()
v_accvgpr_write a11, 0.00000 // (op 475) call()
v_accvgpr_write a12, 0.00000 // (op 637) call()
v_accvgpr_write a13, 0.00000 // (op 637) call()
v_accvgpr_write a14, 0.00000 // (op 637) call()
v_accvgpr_write a15, 0.00000 // (op 637) call()
s_mul_hi_i32 s0, s16, s10 // (op 1852) 
v_lshrrev_b32 v12, 3, v1 // (op 1852) 
s_add_i32 s1, s0, s16 // (op 1852) 

v_and_b32 v13, -8, v1 // (op 1852) 
s_ashr_i32 s0, s1, 31 // (op 1852) 
v_sub_u32 v14, v1, v13 // (op 1852) 

s_cmp_eq_i32 s10, 0 // (op 1852) 
s_mov_b32 s18, scc // (op 1852) 
v_lshlrev_b32 v13, 4, v14 // (op 1852) 

s_cmp_eq_i32 s18, 0 // (op 1852) 
s_cselect_b32 s19, 0, -1 // (op 1852) 

v_mov_b32 v15, 0 // (op 1852) convert
v_mov_b32 v14, v13 // (op 1852) convert

s_and_b32 s18, s13, 31 // (op 1852) 
s_lshl_b32 s22, 1, s18 // (op 1852) 
s_add_i32 s23, s22, s19 // (op 1852) 


s_and_b32 s19, s0, s23 // (op 1852) 


s_add_i32 s0, s1, s19 // (op 1852) 


s_ashr_i32 s1, s0, s18 // (op 1852) 


s_lshl_b32 s0, s13, 24 // (op 1852) 
s_ashr_i32 s18, s0, 31 // (op 1852) 

s_xor_b32 s0, s1, s18 // (op 1852) 

s_sub_i32 s1, s0, s18 // (op 1852) 


s_mul_hi_i32 s0, s1, s14 // (op 1852) 
s_add_i32 s18, s0, s1 // (op 1852) 

s_ashr_i32 s0, s18, 31 // (op 1852) 
s_cmp_eq_i32 s14, 0 // (op 1852) 
s_mov_b32 s19, scc // (op 1852) 
s_cmp_eq_i32 s19, 0 // (op 1852) 
s_cselect_b32 s22, 0, -1 // (op 1852) 

s_and_b32 s19, s15, 31 // (op 1852) 
s_lshl_b32 s23, 1, s19 // (op 1852) 
s_add_i32 s24, s23, s22 // (op 1852) 


s_and_b32 s25, s0, s24 // (op 1852) 


s_add_i32 s0, s18, s25 // (op 1852) 


s_ashr_i32 s18, s0, s19 // (op 1852) 


s_lshl_b32 s0, s15, 24 // (op 1852) 
s_ashr_i32 s19, s0, 31 // (op 1852) 

s_xor_b32 s0, s18, s19 // (op 1852) 

s_sub_i32 s18, s0, s19 // (op 1852) 


s_mul_i32 s0, s1, s11 // (op 1852) 

s_sub_u32 s1, s16, s0 // (op 1852) 

s_mul_i32 s0, s18, s11 // (op 1852) 
s_add_u32 s0, s0, s1 // (op 1852) 


s_mul_hi_i32 s1, s16, s9 // (op 1852) 
s_add_i32 s18, s1, s16 // (op 1852) 

s_ashr_i32 s1, s18, 31 // (op 1852) 
s_cmp_eq_i32 s9, 0 // (op 1852) 
s_mov_b32 s19, scc // (op 1852) 
s_cmp_eq_i32 s19, 0 // (op 1852) 
s_cselect_b32 s22, 0, -1 // (op 1852) 

s_and_b32 s19, s45, 31 // (op 1852) 
s_lshl_b32 s23, 1, s19 // (op 1852) 
s_add_i32 s24, s23, s22 // (op 1852) 


s_and_b32 s25, s1, s24 // (op 1852) 


s_add_i32 s1, s18, s25 // (op 1852) 


s_ashr_i32 s18, s1, s19 // (op 1852) 


s_lshl_b32 s1, s45, 24 // (op 1852) 
s_ashr_i32 s19, s1, 31 // (op 1852) 

s_xor_b32 s1, s18, s19 // (op 1852) 

s_sub_i32 s18, s1, s19 // (op 1852) 


s_add_u32 s22, s4, 32 // (op 1852) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1852) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1852) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1852) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1852) 

s_mul_i32 s1, s11, s22 // (op 1852) 

s_mul_i32 s19, s8, s1 // (op 1852) 

s_mul_i32 s1, s18, s19 // (op 1852) 

s_sub_u32 s19, s16, s1 // (op 1852) 

s_mov_b32 s1, s19 // (op 1852) 
s_mul_hi_i32 s22, s1, s12 // (op 1852) 
s_add_i32 s23, s22, s1 // (op 1852) 


s_ashr_i32 s1, s23, 31 // (op 1852) 
s_cmp_eq_i32 s12, 0 // (op 1852) 
s_mov_b32 s22, scc // (op 1852) 
s_cmp_eq_i32 s22, 0 // (op 1852) 
s_cselect_b32 s24, 0, -1 // (op 1852) 

s_and_b32 s22, s44, 31 // (op 1852) 
s_lshl_b32 s25, 1, s22 // (op 1852) 
s_add_i32 s47, s25, s24 // (op 1852) 


s_and_b32 s24, s1, s47 // (op 1852) 


s_add_i32 s1, s23, s24 // (op 1852) 


s_ashr_i32 s23, s1, s22 // (op 1852) 


s_lshl_b32 s1, s44, 24 // (op 1852) 
s_ashr_i32 s22, s1, 31 // (op 1852) 

s_xor_b32 s1, s23, s22 // (op 1852) 

s_sub_i32 s23, s1, s22 // (op 1852) 


s_mul_i32 s1, s46, s11 // (op 1852) 
s_add_u32 s24, s6, 128 // (op 1852) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1852) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1852) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1852) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 1852) 

s_sub_i32 s22, s24, s1 // (op 1852) 

s_mul_i32 s24, s23, s22 // (op 1852) 


s_sub_u32 s25, s19, s24 // (op 1852) 


s_add_u32 s19, s25, s1 // (op 1852) 


s_cmp_eq_i32 s18, 0 // (op 1852) 
s_mov_b32 s1, scc // (op 1852) 

s_cmp_eq_i32 s1, 0 // (op 1852) 
s_cselect_b32 s18, s19, s0 // (op 1852) 



v_lshl_add_u32 v13, s18, 7, v12 // (op 1852) 


v_mul_lo_u32 v17, v13, s27 // (op 1852) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 1852) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 1852) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1852) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1852) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1852) most significant half



s_mul_i32 s1, s26, s7 // (op 1854) most significant: low of low * high
s_mul_i32 s18, s27, s6 // (op 1854) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 1854) least significant
s_mul_hi_u32 s19, s26, s6 // (op 1854) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1854) most significant: sum
s_add_u32 s1, s1, s19 // (op 1854) most significant: sum


s_mov_b32 s48, s28 // (op 1854) reg expression
s_mov_b32 s49, s29 // (op 1854) reg expression
s_mov_b32 s50, s0 // (op 1854) reg expression
s_mov_b32 s51, 131072 // (op 1854) call()

s_mul_hi_i32 s47, s16, s10 // (op 1871) 
v_lshrrev_b32 v12, 3, v1 // (op 1871) 
s_add_i32 s0, s47, s16 // (op 1871) 

v_and_b32 v13, -8, v1 // (op 1871) 
s_ashr_i32 s1, s0, 31 // (op 1871) 
v_sub_u32 v14, v1, v13 // (op 1871) 

s_cmp_eq_i32 s10, 0 // (op 1871) 
s_mov_b32 s47, scc // (op 1871) 
v_lshlrev_b32 v13, 4, v14 // (op 1871) 

s_cmp_eq_i32 s47, 0 // (op 1871) 
s_cselect_b32 s18, 0, -1 // (op 1871) 

v_mov_b32 v15, 0 // (op 1871) convert
v_mov_b32 v14, v13 // (op 1871) convert

s_and_b32 s19, s13, 31 // (op 1871) 
s_lshl_b32 s47, 1, s19 // (op 1871) 
s_add_i32 s22, s47, s18 // (op 1871) 


s_and_b32 s18, s1, s22 // (op 1871) 


s_add_i32 s1, s0, s18 // (op 1871) 


s_ashr_i32 s0, s1, s19 // (op 1871) 


s_lshl_b32 s1, s13, 24 // (op 1871) 
s_ashr_i32 s47, s1, 31 // (op 1871) 

s_xor_b32 s1, s0, s47 // (op 1871) 

s_sub_i32 s0, s1, s47 // (op 1871) 


s_mul_hi_i32 s1, s0, s14 // (op 1871) 
s_add_i32 s47, s1, s0 // (op 1871) 

s_ashr_i32 s1, s47, 31 // (op 1871) 
s_cmp_eq_i32 s14, 0 // (op 1871) 
s_mov_b32 s18, scc // (op 1871) 
s_cmp_eq_i32 s18, 0 // (op 1871) 
s_cselect_b32 s19, 0, -1 // (op 1871) 

s_and_b32 s18, s15, 31 // (op 1871) 
s_lshl_b32 s22, 1, s18 // (op 1871) 
s_add_i32 s23, s22, s19 // (op 1871) 


s_and_b32 s19, s1, s23 // (op 1871) 


s_add_i32 s1, s47, s19 // (op 1871) 


s_ashr_i32 s19, s1, s18 // (op 1871) 


s_lshl_b32 s1, s15, 24 // (op 1871) 
s_ashr_i32 s18, s1, 31 // (op 1871) 

s_xor_b32 s1, s19, s18 // (op 1871) 

s_sub_i32 s19, s1, s18 // (op 1871) 


s_mul_i32 s1, s0, s11 // (op 1871) 

s_sub_u32 s0, s16, s1 // (op 1871) 

s_mul_i32 s1, s19, s11 // (op 1871) 
s_add_u32 s1, s1, s0 // (op 1871) 


s_mul_hi_i32 s0, s16, s9 // (op 1871) 
s_add_i32 s47, s0, s16 // (op 1871) 

s_ashr_i32 s0, s47, 31 // (op 1871) 
s_cmp_eq_i32 s9, 0 // (op 1871) 
s_mov_b32 s18, scc // (op 1871) 
s_cmp_eq_i32 s18, 0 // (op 1871) 
s_cselect_b32 s19, 0, -1 // (op 1871) 

s_and_b32 s18, s45, 31 // (op 1871) 
s_lshl_b32 s22, 1, s18 // (op 1871) 
s_add_i32 s23, s22, s19 // (op 1871) 


s_and_b32 s19, s0, s23 // (op 1871) 


s_add_i32 s0, s47, s19 // (op 1871) 


s_ashr_i32 s19, s0, s18 // (op 1871) 


s_lshl_b32 s0, s45, 24 // (op 1871) 
s_ashr_i32 s18, s0, 31 // (op 1871) 

s_xor_b32 s0, s19, s18 // (op 1871) 

s_sub_i32 s19, s0, s18 // (op 1871) 


s_add_u32 s22, s4, 32 // (op 1871) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1871) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1871) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1871) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1871) 

s_mul_i32 s0, s11, s22 // (op 1871) 

s_mul_i32 s18, s8, s0 // (op 1871) 

s_mul_i32 s0, s19, s18 // (op 1871) 

s_sub_u32 s18, s16, s0 // (op 1871) 

s_mov_b32 s0, s18 // (op 1871) 
s_mul_hi_i32 s47, s0, s12 // (op 1871) 
s_add_i32 s22, s47, s0 // (op 1871) 


s_ashr_i32 s0, s22, 31 // (op 1871) 
s_cmp_eq_i32 s12, 0 // (op 1871) 
s_mov_b32 s47, scc // (op 1871) 
s_cmp_eq_i32 s47, 0 // (op 1871) 
s_cselect_b32 s23, 0, -1 // (op 1871) 

s_and_b32 s47, s44, 31 // (op 1871) 
s_lshl_b32 s24, 1, s47 // (op 1871) 
s_add_i32 s25, s24, s23 // (op 1871) 


s_and_b32 s23, s0, s25 // (op 1871) 


s_add_i32 s0, s22, s23 // (op 1871) 


s_ashr_i32 s22, s0, s47 // (op 1871) 


s_lshl_b32 s0, s44, 24 // (op 1871) 
s_ashr_i32 s47, s0, 31 // (op 1871) 

s_xor_b32 s0, s22, s47 // (op 1871) 

s_sub_i32 s22, s0, s47 // (op 1871) 


s_mul_i32 s0, s46, s11 // (op 1871) 
s_add_u32 s24, s6, 128 // (op 1871) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1871) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 1871) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 1871) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 7 // (op 1871) 

s_sub_i32 s23, s24, s0 // (op 1871) 

s_mul_i32 s47, s22, s23 // (op 1871) 


s_sub_u32 s22, s18, s47 // (op 1871) 


s_add_u32 s18, s22, s0 // (op 1871) 


s_cmp_eq_i32 s19, 0 // (op 1871) 
s_mov_b32 s0, scc // (op 1871) 

s_cmp_eq_i32 s0, 0 // (op 1871) 
s_cselect_b32 s19, s18, s1 // (op 1871) 



v_lshl_add_u32 v13, s19, 7, v12 // (op 1871) 


v_mul_lo_u32 v19, v13, s27 // (op 1871) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 1871) most significant: high of low * low
v_mul_lo_u32 v18, v13, s26 // (op 1871) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 1871) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 1871) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 1871) most significant half



s_mul_i32 s1, s26, s7 // (op 1873) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 1873) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 1873) least significant
s_mul_hi_u32 s18, s26, s6 // (op 1873) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1873) most significant: sum
s_add_u32 s1, s1, s18 // (op 1873) most significant: sum


s_mov_b32 s48, s28 // (op 1873) reg expression
s_mov_b32 s49, s29 // (op 1873) reg expression
s_mov_b32 s50, s0 // (op 1873) reg expression
s_mov_b32 s51, 131072 // (op 1873) call()

s_mul_hi_i32 s47, s16, s10 // (op 1916) 
v_lshrrev_b32 v12, 3, v1 // (op 1916) 
s_add_i32 s0, s47, s16 // (op 1916) 

v_and_b32 v13, -8, v1 // (op 1916) 
s_ashr_i32 s1, s0, 31 // (op 1916) 
v_sub_u32 v14, v1, v13 // (op 1916) 

s_cmp_eq_i32 s10, 0 // (op 1916) 
s_mov_b32 s47, scc // (op 1916) 
v_lshlrev_b32 v13, 4, v14 // (op 1916) 

s_cmp_eq_i32 s47, 0 // (op 1916) 
s_cselect_b32 s18, 0, -1 // (op 1916) 

v_mov_b32 v15, 0 // (op 1916) convert
v_mov_b32 v14, v13 // (op 1916) convert

s_and_b32 s19, s13, 31 // (op 1916) 
s_lshl_b32 s47, 1, s19 // (op 1916) 
s_add_i32 s22, s47, s18 // (op 1916) 


s_and_b32 s18, s1, s22 // (op 1916) 


s_add_i32 s1, s0, s18 // (op 1916) 


s_ashr_i32 s0, s1, s19 // (op 1916) 


s_lshl_b32 s1, s13, 24 // (op 1916) 
s_ashr_i32 s47, s1, 31 // (op 1916) 

s_xor_b32 s1, s0, s47 // (op 1916) 

s_sub_i32 s0, s1, s47 // (op 1916) 


s_mul_hi_i32 s1, s0, s14 // (op 1916) 
s_add_i32 s47, s1, s0 // (op 1916) 

s_ashr_i32 s1, s47, 31 // (op 1916) 
s_cmp_eq_i32 s14, 0 // (op 1916) 
s_mov_b32 s18, scc // (op 1916) 
s_cmp_eq_i32 s18, 0 // (op 1916) 
s_cselect_b32 s19, 0, -1 // (op 1916) 

s_and_b32 s18, s15, 31 // (op 1916) 
s_lshl_b32 s22, 1, s18 // (op 1916) 
s_add_i32 s23, s22, s19 // (op 1916) 


s_and_b32 s19, s1, s23 // (op 1916) 


s_add_i32 s1, s47, s19 // (op 1916) 


s_ashr_i32 s19, s1, s18 // (op 1916) 


s_lshl_b32 s1, s15, 24 // (op 1916) 
s_ashr_i32 s18, s1, 31 // (op 1916) 

s_xor_b32 s1, s19, s18 // (op 1916) 

s_sub_i32 s19, s1, s18 // (op 1916) 


s_add_u32 s22, s4, 32 // (op 1916) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1916) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1916) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1916) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1916) 

s_mul_i32 s1, s19, s22 // (op 1916) 

s_sub_i32 s47, s0, s1 // (op 1916) 


s_mul_hi_i32 s0, s16, s9 // (op 1916) 
s_add_i32 s1, s0, s16 // (op 1916) 

s_ashr_i32 s0, s1, 31 // (op 1916) 
s_cmp_eq_i32 s9, 0 // (op 1916) 
s_mov_b32 s18, scc // (op 1916) 
s_cmp_eq_i32 s18, 0 // (op 1916) 
s_cselect_b32 s19, 0, -1 // (op 1916) 

s_and_b32 s18, s45, 31 // (op 1916) 
s_lshl_b32 s24, 1, s18 // (op 1916) 
s_add_i32 s25, s24, s19 // (op 1916) 


s_and_b32 s19, s0, s25 // (op 1916) 


s_add_i32 s0, s1, s19 // (op 1916) 


s_ashr_i32 s1, s0, s18 // (op 1916) 


s_lshl_b32 s0, s45, 24 // (op 1916) 
s_ashr_i32 s18, s0, 31 // (op 1916) 

s_xor_b32 s0, s1, s18 // (op 1916) 

s_sub_i32 s1, s0, s18 // (op 1916) 


s_mul_i32 s0, s11, s22 // (op 1916) 

s_mul_i32 s18, s8, s0 // (op 1916) 

s_mul_i32 s0, s1, s18 // (op 1916) 

s_sub_u32 s18, s16, s0 // (op 1916) 

s_mul_hi_i32 s0, s18, s12 // (op 1916) 
s_add_i32 s19, s0, s18 // (op 1916) 


s_ashr_i32 s0, s19, 31 // (op 1916) 
s_cmp_eq_i32 s12, 0 // (op 1916) 
s_mov_b32 s18, scc // (op 1916) 
s_cmp_eq_i32 s18, 0 // (op 1916) 
s_cselect_b32 s22, 0, -1 // (op 1916) 

s_and_b32 s18, s44, 31 // (op 1916) 
s_lshl_b32 s23, 1, s18 // (op 1916) 
s_add_i32 s24, s23, s22 // (op 1916) 


s_and_b32 s25, s0, s24 // (op 1916) 


s_add_i32 s0, s19, s25 // (op 1916) 


s_ashr_i32 s19, s0, s18 // (op 1916) 


s_lshl_b32 s0, s44, 24 // (op 1916) 
s_ashr_i32 s18, s0, 31 // (op 1916) 

s_xor_b32 s0, s19, s18 // (op 1916) 

s_sub_i32 s19, s0, s18 // (op 1916) 


s_cmp_eq_i32 s1, 0 // (op 1916) 
s_mov_b32 s0, scc // (op 1916) 

s_cmp_eq_i32 s0, 0 // (op 1916) 
s_cselect_b32 s1, s19, s47 // (op 1916) 



v_lshl_add_u32 v13, s1, 5, v12 // (op 1916) 


v_mul_lo_u32 v21, v13, s31 // (op 1916) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 1916) most significant: high of low * low
v_mul_lo_u32 v20, v13, s30 // (op 1916) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 1916) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 1916) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 1916) most significant half



s_mul_i32 s1, s30, s5 // (op 1918) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 1918) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 1918) least significant
s_mul_hi_u32 s18, s30, s4 // (op 1918) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1918) most significant: sum
s_add_u32 s1, s1, s18 // (op 1918) most significant: sum


s_mov_b32 s60, s32 // (op 1918) reg expression
s_mov_b32 s61, s33 // (op 1918) reg expression
s_mov_b32 s62, s0 // (op 1918) reg expression
s_mov_b32 s63, 131072 // (op 1918) call()

s_mul_hi_i32 s47, s16, s10 // (op 1935) 
v_lshrrev_b32 v14, 3, v1 // (op 1935) 
s_add_i32 s0, s47, s16 // (op 1935) 

v_and_b32 v15, -8, v1 // (op 1935) 
s_ashr_i32 s1, s0, 31 // (op 1935) 
v_sub_u32 v20, v1, v15 // (op 1935) 

s_cmp_eq_i32 s10, 0 // (op 1935) 
s_mov_b32 s47, scc // (op 1935) 
v_lshlrev_b32 v15, 4, v20 // (op 1935) 

s_cmp_eq_i32 s47, 0 // (op 1935) 
s_cselect_b32 s18, 0, -1 // (op 1935) 

v_mov_b32 v21, 0 // (op 1935) convert
v_mov_b32 v20, v15 // (op 1935) convert

s_and_b32 s19, s13, 31 // (op 1935) 
s_lshl_b32 s47, 1, s19 // (op 1935) 
s_add_i32 s22, s47, s18 // (op 1935) 


s_and_b32 s18, s1, s22 // (op 1935) 


s_add_i32 s1, s0, s18 // (op 1935) 


s_ashr_i32 s0, s1, s19 // (op 1935) 


s_lshl_b32 s1, s13, 24 // (op 1935) 
s_ashr_i32 s47, s1, 31 // (op 1935) 

s_xor_b32 s1, s0, s47 // (op 1935) 

s_sub_i32 s0, s1, s47 // (op 1935) 


s_mul_hi_i32 s1, s0, s14 // (op 1935) 
s_add_i32 s47, s1, s0 // (op 1935) 

s_ashr_i32 s1, s47, 31 // (op 1935) 
s_cmp_eq_i32 s14, 0 // (op 1935) 
s_mov_b32 s18, scc // (op 1935) 
s_cmp_eq_i32 s18, 0 // (op 1935) 
s_cselect_b32 s19, 0, -1 // (op 1935) 

s_and_b32 s18, s15, 31 // (op 1935) 
s_lshl_b32 s22, 1, s18 // (op 1935) 
s_add_i32 s23, s22, s19 // (op 1935) 


s_and_b32 s19, s1, s23 // (op 1935) 


s_add_i32 s1, s47, s19 // (op 1935) 


s_ashr_i32 s19, s1, s18 // (op 1935) 


s_lshl_b32 s1, s15, 24 // (op 1935) 
s_ashr_i32 s18, s1, 31 // (op 1935) 

s_xor_b32 s1, s19, s18 // (op 1935) 

s_sub_i32 s19, s1, s18 // (op 1935) 


s_add_u32 s22, s4, 32 // (op 1935) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1935) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1935) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1935) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1935) 

s_mul_i32 s1, s19, s22 // (op 1935) 

s_sub_i32 s47, s0, s1 // (op 1935) 


s_mul_hi_i32 s0, s16, s9 // (op 1935) 
s_add_i32 s1, s0, s16 // (op 1935) 

s_ashr_i32 s0, s1, 31 // (op 1935) 
s_cmp_eq_i32 s9, 0 // (op 1935) 
s_mov_b32 s18, scc // (op 1935) 
s_cmp_eq_i32 s18, 0 // (op 1935) 
s_cselect_b32 s19, 0, -1 // (op 1935) 

s_and_b32 s18, s45, 31 // (op 1935) 
s_lshl_b32 s24, 1, s18 // (op 1935) 
s_add_i32 s25, s24, s19 // (op 1935) 


s_and_b32 s19, s0, s25 // (op 1935) 


s_add_i32 s0, s1, s19 // (op 1935) 


s_ashr_i32 s1, s0, s18 // (op 1935) 


s_lshl_b32 s0, s45, 24 // (op 1935) 
s_ashr_i32 s18, s0, 31 // (op 1935) 

s_xor_b32 s0, s1, s18 // (op 1935) 

s_sub_i32 s1, s0, s18 // (op 1935) 


s_mul_i32 s0, s11, s22 // (op 1935) 

s_mul_i32 s18, s8, s0 // (op 1935) 

s_mul_i32 s0, s1, s18 // (op 1935) 

s_sub_u32 s18, s16, s0 // (op 1935) 

s_mul_hi_i32 s0, s18, s12 // (op 1935) 
s_add_i32 s19, s0, s18 // (op 1935) 


s_ashr_i32 s0, s19, 31 // (op 1935) 
s_cmp_eq_i32 s12, 0 // (op 1935) 
s_mov_b32 s18, scc // (op 1935) 
s_cmp_eq_i32 s18, 0 // (op 1935) 
s_cselect_b32 s22, 0, -1 // (op 1935) 

s_and_b32 s18, s44, 31 // (op 1935) 
s_lshl_b32 s23, 1, s18 // (op 1935) 
s_add_i32 s24, s23, s22 // (op 1935) 


s_and_b32 s25, s0, s24 // (op 1935) 


s_add_i32 s0, s19, s25 // (op 1935) 


s_ashr_i32 s19, s0, s18 // (op 1935) 


s_lshl_b32 s0, s44, 24 // (op 1935) 
s_ashr_i32 s18, s0, 31 // (op 1935) 

s_xor_b32 s0, s19, s18 // (op 1935) 

s_sub_i32 s19, s0, s18 // (op 1935) 


s_cmp_eq_i32 s1, 0 // (op 1935) 
s_mov_b32 s0, scc // (op 1935) 

s_cmp_eq_i32 s0, 0 // (op 1935) 
s_cselect_b32 s1, s19, s47 // (op 1935) 



v_lshl_add_u32 v15, s1, 5, v14 // (op 1935) 


v_mul_lo_u32 v23, v15, s31 // (op 1935) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 1935) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 1935) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 1935) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 1935) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 1935) most significant half



s_mul_i32 s1, s30, s5 // (op 1937) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 1937) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 1937) least significant
s_mul_hi_u32 s18, s30, s4 // (op 1937) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1937) most significant: sum
s_add_u32 s1, s1, s18 // (op 1937) most significant: sum


s_mov_b32 s60, s32 // (op 1937) reg expression
s_mov_b32 s61, s33 // (op 1937) reg expression
s_mov_b32 s62, s0 // (op 1937) reg expression
s_mov_b32 s63, 131072 // (op 1937) call()

s_mul_hi_i32 s47, s16, s10 // (op 1980) 
v_lshrrev_b32 v20, 2, v1 // (op 1980) 
s_add_i32 s0, s47, s16 // (op 1980) 

v_lshlrev_b32 v21, 1, v20 // (op 1980) 

s_ashr_i32 s1, s0, 31 // (op 1980) 
v_and_b32 v20, -4, v1 // (op 1980) 
s_cmp_eq_i32 s10, 0 // (op 1980) 
s_mov_b32 s47, scc // (op 1980) 
v_sub_u32 v22, v1, v20 // (op 1980) 

s_cmp_eq_i32 s47, 0 // (op 1980) 
s_cselect_b32 s18, 0, -1 // (op 1980) 

s_and_b32 s19, s13, 31 // (op 1980) 
s_lshl_b32 s47, 1, s19 // (op 1980) 
s_add_i32 s22, s47, s18 // (op 1980) 


s_and_b32 s18, s1, s22 // (op 1980) 


s_add_i32 s1, s0, s18 // (op 1980) 


s_ashr_i32 s0, s1, s19 // (op 1980) 


s_lshl_b32 s1, s13, 24 // (op 1980) 
s_ashr_i32 s47, s1, 31 // (op 1980) 

s_xor_b32 s1, s0, s47 // (op 1980) 

s_sub_i32 s0, s1, s47 // (op 1980) 


s_mul_hi_i32 s1, s0, s14 // (op 1980) 
s_add_i32 s47, s1, s0 // (op 1980) 

s_ashr_i32 s1, s47, 31 // (op 1980) 
s_cmp_eq_i32 s14, 0 // (op 1980) 
s_mov_b32 s18, scc // (op 1980) 
s_cmp_eq_i32 s18, 0 // (op 1980) 
s_cselect_b32 s19, 0, -1 // (op 1980) 

s_and_b32 s18, s15, 31 // (op 1980) 
s_lshl_b32 s22, 1, s18 // (op 1980) 
s_add_i32 s23, s22, s19 // (op 1980) 


s_and_b32 s19, s1, s23 // (op 1980) 


s_add_i32 s1, s47, s19 // (op 1980) 


s_ashr_i32 s19, s1, s18 // (op 1980) 


s_lshl_b32 s1, s15, 24 // (op 1980) 
s_ashr_i32 s18, s1, 31 // (op 1980) 

s_xor_b32 s1, s19, s18 // (op 1980) 

s_sub_i32 s19, s1, s18 // (op 1980) 


s_mul_i32 s1, s0, s11 // (op 1980) 

s_sub_u32 s0, s16, s1 // (op 1980) 

s_mul_i32 s1, s19, s11 // (op 1980) 
s_add_u32 s1, s1, s0 // (op 1980) 


s_mul_hi_i32 s0, s16, s9 // (op 1980) 
s_add_i32 s47, s0, s16 // (op 1980) 

s_ashr_i32 s0, s47, 31 // (op 1980) 
s_cmp_eq_i32 s9, 0 // (op 1980) 
s_mov_b32 s18, scc // (op 1980) 
s_cmp_eq_i32 s18, 0 // (op 1980) 
s_cselect_b32 s19, 0, -1 // (op 1980) 

s_and_b32 s18, s45, 31 // (op 1980) 
s_lshl_b32 s22, 1, s18 // (op 1980) 
s_add_i32 s23, s22, s19 // (op 1980) 


s_and_b32 s19, s0, s23 // (op 1980) 


s_add_i32 s0, s47, s19 // (op 1980) 


s_ashr_i32 s19, s0, s18 // (op 1980) 


s_lshl_b32 s0, s45, 24 // (op 1980) 
s_ashr_i32 s18, s0, 31 // (op 1980) 

s_xor_b32 s0, s19, s18 // (op 1980) 

s_sub_i32 s19, s0, s18 // (op 1980) 


s_add_u32 s22, s4, 32 // (op 1980) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1980) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1980) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1980) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1980) 

s_mul_i32 s0, s11, s22 // (op 1980) 

s_mul_i32 s18, s8, s0 // (op 1980) 

s_mul_i32 s0, s19, s18 // (op 1980) 

s_sub_u32 s18, s16, s0 // (op 1980) 

s_mov_b32 s0, s18 // (op 1980) 
s_mul_hi_i32 s47, s0, s12 // (op 1980) 
s_add_i32 s22, s47, s0 // (op 1980) 


s_ashr_i32 s0, s22, 31 // (op 1980) 
s_cmp_eq_i32 s12, 0 // (op 1980) 
s_mov_b32 s47, scc // (op 1980) 
s_cmp_eq_i32 s47, 0 // (op 1980) 
s_cselect_b32 s23, 0, -1 // (op 1980) 

s_and_b32 s47, s44, 31 // (op 1980) 
s_lshl_b32 s24, 1, s47 // (op 1980) 
s_add_i32 s25, s24, s23 // (op 1980) 


s_and_b32 s23, s0, s25 // (op 1980) 


s_add_i32 s0, s22, s23 // (op 1980) 


s_ashr_i32 s22, s0, s47 // (op 1980) 


s_lshl_b32 s0, s44, 24 // (op 1980) 
s_ashr_i32 s47, s0, 31 // (op 1980) 

s_xor_b32 s0, s22, s47 // (op 1980) 

s_sub_i32 s22, s0, s47 // (op 1980) 


s_mul_i32 s0, s46, s11 // (op 1980) 
s_add_u32 s24, s6, 128 // (op 1980) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1980) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 1980) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 1980) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 1980) 

s_sub_i32 s23, s24, s0 // (op 1980) 

s_mul_i32 s47, s22, s23 // (op 1980) 


s_sub_u32 s22, s18, s47 // (op 1980) 


s_add_u32 s18, s22, s0 // (op 1980) 


s_cmp_eq_i32 s19, 0 // (op 1980) 
s_mov_b32 s0, scc // (op 1980) 

s_cmp_eq_i32 s0, 0 // (op 1980) 
s_cselect_b32 s19, s18, s1 // (op 1980) 



v_lshl_add_u32 v20, s19, 7, v21 // (op 1980) 


v_mul_lo_u32 v21, v20, s34 // (op 1980) 
v_add_u32 v21, v21, v22 // (op 1980) 


s_mul_i32 s1, s34, s7 // (op 1982) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1982) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1982) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1982) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1982) most significant: sum
s_add_u32 s1, s1, s18 // (op 1982) most significant: sum


s_mov_b32 s64, s36 // (op 1982) reg expression
s_mov_b32 s65, s37 // (op 1982) reg expression
s_mov_b32 s66, s0 // (op 1982) reg expression
s_mov_b32 s67, 131072 // (op 1982) call()

s_mul_hi_i32 s47, s16, s10 // (op 1999) 
v_lshrrev_b32 v20, 2, v1 // (op 1999) 
s_add_i32 s0, s47, s16 // (op 1999) 

v_lshlrev_b32 v22, 1, v20 // (op 1999) 

s_ashr_i32 s1, s0, 31 // (op 1999) 
v_and_b32 v20, -4, v1 // (op 1999) 
s_cmp_eq_i32 s10, 0 // (op 1999) 
s_mov_b32 s47, scc // (op 1999) 
v_sub_u32 v23, v1, v20 // (op 1999) 

s_cmp_eq_i32 s47, 0 // (op 1999) 
s_cselect_b32 s18, 0, -1 // (op 1999) 

s_and_b32 s19, s13, 31 // (op 1999) 
s_lshl_b32 s47, 1, s19 // (op 1999) 
s_add_i32 s22, s47, s18 // (op 1999) 


s_and_b32 s18, s1, s22 // (op 1999) 


s_add_i32 s1, s0, s18 // (op 1999) 


s_ashr_i32 s0, s1, s19 // (op 1999) 


s_lshl_b32 s1, s13, 24 // (op 1999) 
s_ashr_i32 s47, s1, 31 // (op 1999) 

s_xor_b32 s1, s0, s47 // (op 1999) 

s_sub_i32 s0, s1, s47 // (op 1999) 


s_mul_hi_i32 s1, s0, s14 // (op 1999) 
s_add_i32 s47, s1, s0 // (op 1999) 

s_ashr_i32 s1, s47, 31 // (op 1999) 
s_cmp_eq_i32 s14, 0 // (op 1999) 
s_mov_b32 s18, scc // (op 1999) 
s_cmp_eq_i32 s18, 0 // (op 1999) 
s_cselect_b32 s19, 0, -1 // (op 1999) 

s_and_b32 s18, s15, 31 // (op 1999) 
s_lshl_b32 s22, 1, s18 // (op 1999) 
s_add_i32 s23, s22, s19 // (op 1999) 


s_and_b32 s19, s1, s23 // (op 1999) 


s_add_i32 s1, s47, s19 // (op 1999) 


s_ashr_i32 s19, s1, s18 // (op 1999) 


s_lshl_b32 s1, s15, 24 // (op 1999) 
s_ashr_i32 s18, s1, 31 // (op 1999) 

s_xor_b32 s1, s19, s18 // (op 1999) 

s_sub_i32 s19, s1, s18 // (op 1999) 


s_mul_i32 s1, s0, s11 // (op 1999) 

s_sub_u32 s0, s16, s1 // (op 1999) 

s_mul_i32 s1, s19, s11 // (op 1999) 
s_add_u32 s1, s1, s0 // (op 1999) 


s_mul_hi_i32 s0, s16, s9 // (op 1999) 
s_add_i32 s47, s0, s16 // (op 1999) 

s_ashr_i32 s0, s47, 31 // (op 1999) 
s_cmp_eq_i32 s9, 0 // (op 1999) 
s_mov_b32 s18, scc // (op 1999) 
s_cmp_eq_i32 s18, 0 // (op 1999) 
s_cselect_b32 s19, 0, -1 // (op 1999) 

s_and_b32 s18, s45, 31 // (op 1999) 
s_lshl_b32 s22, 1, s18 // (op 1999) 
s_add_i32 s23, s22, s19 // (op 1999) 


s_and_b32 s19, s0, s23 // (op 1999) 


s_add_i32 s0, s47, s19 // (op 1999) 


s_ashr_i32 s19, s0, s18 // (op 1999) 


s_lshl_b32 s0, s45, 24 // (op 1999) 
s_ashr_i32 s18, s0, 31 // (op 1999) 

s_xor_b32 s0, s19, s18 // (op 1999) 

s_sub_i32 s19, s0, s18 // (op 1999) 


s_add_u32 s22, s4, 32 // (op 1999) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1999) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1999) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1999) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1999) 

s_mul_i32 s0, s11, s22 // (op 1999) 

s_mul_i32 s18, s8, s0 // (op 1999) 

s_mul_i32 s0, s19, s18 // (op 1999) 

s_sub_u32 s18, s16, s0 // (op 1999) 

s_mov_b32 s0, s18 // (op 1999) 
s_mul_hi_i32 s47, s0, s12 // (op 1999) 
s_add_i32 s22, s47, s0 // (op 1999) 


s_ashr_i32 s0, s22, 31 // (op 1999) 
s_cmp_eq_i32 s12, 0 // (op 1999) 
s_mov_b32 s47, scc // (op 1999) 
s_cmp_eq_i32 s47, 0 // (op 1999) 
s_cselect_b32 s23, 0, -1 // (op 1999) 

s_and_b32 s47, s44, 31 // (op 1999) 
s_lshl_b32 s24, 1, s47 // (op 1999) 
s_add_i32 s25, s24, s23 // (op 1999) 


s_and_b32 s23, s0, s25 // (op 1999) 


s_add_i32 s0, s22, s23 // (op 1999) 


s_ashr_i32 s22, s0, s47 // (op 1999) 


s_lshl_b32 s0, s44, 24 // (op 1999) 
s_ashr_i32 s47, s0, 31 // (op 1999) 

s_xor_b32 s0, s22, s47 // (op 1999) 

s_sub_i32 s22, s0, s47 // (op 1999) 


s_mul_i32 s0, s46, s11 // (op 1999) 
s_add_u32 s24, s6, 128 // (op 1999) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1999) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 1999) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 1999) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 7 // (op 1999) 

s_sub_i32 s23, s24, s0 // (op 1999) 

s_mul_i32 s47, s22, s23 // (op 1999) 


s_sub_u32 s22, s18, s47 // (op 1999) 


s_add_u32 s18, s22, s0 // (op 1999) 


s_cmp_eq_i32 s19, 0 // (op 1999) 
s_mov_b32 s0, scc // (op 1999) 

s_cmp_eq_i32 s0, 0 // (op 1999) 
s_cselect_b32 s19, s18, s1 // (op 1999) 



v_lshl_add_u32 v20, s19, 7, v22 // (op 1999) 


v_mul_lo_u32 v22, v20, s34 // (op 1999) 
v_add_u32 v22, v22, v23 // (op 1999) 


s_mul_i32 s1, s34, s7 // (op 2001) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 2001) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 2001) least significant
s_mul_hi_u32 s18, s34, s6 // (op 2001) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2001) most significant: sum
s_add_u32 s1, s1, s18 // (op 2001) most significant: sum


s_mov_b32 s64, s36 // (op 2001) reg expression
s_mov_b32 s65, s37 // (op 2001) reg expression
s_mov_b32 s66, s0 // (op 2001) reg expression
s_mov_b32 s67, 131072 // (op 2001) call()

s_mul_hi_i32 s47, s16, s10 // (op 2044) 
v_lshrrev_b32 v20, 6, v1 // (op 2044) 
s_add_i32 s0, s47, s16 // (op 2044) 

v_and_b32 v23, 1, v20 // (op 2044) 

s_ashr_i32 s1, s0, 31 // (op 2044) 
v_and_b32 v20, 15, v1 // (op 2044) 
s_cmp_eq_i32 s10, 0 // (op 2044) 
s_mov_b32 s47, scc // (op 2044) 
v_lshl_add_u32 v24, v23, 4, v20 // (op 2044) 


s_cmp_eq_i32 s47, 0 // (op 2044) 
s_cselect_b32 s18, 0, -1 // (op 2044) 

v_and_b32 v20, 63, v1 // (op 2044) 
s_and_b32 s19, s13, 31 // (op 2044) 
v_lshrrev_b32 v23, 4, v20 // (op 2044) 

s_lshl_b32 s47, 1, s19 // (op 2044) 
s_add_i32 s22, s47, s18 // (op 2044) 


s_and_b32 s18, s1, s22 // (op 2044) 


s_add_i32 s1, s0, s18 // (op 2044) 


s_ashr_i32 s0, s1, s19 // (op 2044) 


s_lshl_b32 s1, s13, 24 // (op 2044) 
s_ashr_i32 s47, s1, 31 // (op 2044) 

s_xor_b32 s1, s0, s47 // (op 2044) 

s_sub_i32 s0, s1, s47 // (op 2044) 


s_mul_hi_i32 s1, s0, s14 // (op 2044) 
s_add_i32 s47, s1, s0 // (op 2044) 

s_ashr_i32 s1, s47, 31 // (op 2044) 
s_cmp_eq_i32 s14, 0 // (op 2044) 
s_mov_b32 s18, scc // (op 2044) 
s_cmp_eq_i32 s18, 0 // (op 2044) 
s_cselect_b32 s19, 0, -1 // (op 2044) 

s_and_b32 s18, s15, 31 // (op 2044) 
s_lshl_b32 s22, 1, s18 // (op 2044) 
s_add_i32 s23, s22, s19 // (op 2044) 


s_and_b32 s19, s1, s23 // (op 2044) 


s_add_i32 s1, s47, s19 // (op 2044) 


s_ashr_i32 s19, s1, s18 // (op 2044) 


s_lshl_b32 s1, s15, 24 // (op 2044) 
s_ashr_i32 s18, s1, 31 // (op 2044) 

s_xor_b32 s1, s19, s18 // (op 2044) 

s_sub_i32 s19, s1, s18 // (op 2044) 


s_add_u32 s22, s4, 32 // (op 2044) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 2044) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2044) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2044) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 2044) 

s_mul_i32 s1, s19, s22 // (op 2044) 

s_sub_i32 s47, s0, s1 // (op 2044) 


s_mul_hi_i32 s0, s16, s9 // (op 2044) 
s_add_i32 s1, s0, s16 // (op 2044) 

s_ashr_i32 s0, s1, 31 // (op 2044) 
s_cmp_eq_i32 s9, 0 // (op 2044) 
s_mov_b32 s18, scc // (op 2044) 
s_cmp_eq_i32 s18, 0 // (op 2044) 
s_cselect_b32 s19, 0, -1 // (op 2044) 

s_and_b32 s18, s45, 31 // (op 2044) 
s_lshl_b32 s24, 1, s18 // (op 2044) 
s_add_i32 s25, s24, s19 // (op 2044) 


s_and_b32 s19, s0, s25 // (op 2044) 


s_add_i32 s0, s1, s19 // (op 2044) 


s_ashr_i32 s1, s0, s18 // (op 2044) 


s_lshl_b32 s0, s45, 24 // (op 2044) 
s_ashr_i32 s18, s0, 31 // (op 2044) 

s_xor_b32 s0, s1, s18 // (op 2044) 

s_sub_i32 s1, s0, s18 // (op 2044) 


s_mul_i32 s0, s11, s22 // (op 2044) 

s_mul_i32 s18, s8, s0 // (op 2044) 

s_mul_i32 s0, s1, s18 // (op 2044) 

s_sub_u32 s18, s16, s0 // (op 2044) 

s_mul_hi_i32 s0, s18, s12 // (op 2044) 
s_add_i32 s19, s0, s18 // (op 2044) 


s_ashr_i32 s0, s19, 31 // (op 2044) 
s_cmp_eq_i32 s12, 0 // (op 2044) 
s_mov_b32 s18, scc // (op 2044) 
s_cmp_eq_i32 s18, 0 // (op 2044) 
s_cselect_b32 s22, 0, -1 // (op 2044) 

s_and_b32 s18, s44, 31 // (op 2044) 
s_lshl_b32 s23, 1, s18 // (op 2044) 
s_add_i32 s24, s23, s22 // (op 2044) 


s_and_b32 s25, s0, s24 // (op 2044) 


s_add_i32 s0, s19, s25 // (op 2044) 


s_ashr_i32 s19, s0, s18 // (op 2044) 


s_lshl_b32 s0, s44, 24 // (op 2044) 
s_ashr_i32 s18, s0, 31 // (op 2044) 

s_xor_b32 s0, s19, s18 // (op 2044) 

s_sub_i32 s19, s0, s18 // (op 2044) 


s_cmp_eq_i32 s1, 0 // (op 2044) 
s_mov_b32 s0, scc // (op 2044) 

s_cmp_eq_i32 s0, 0 // (op 2044) 
s_cselect_b32 s1, s19, s47 // (op 2044) 



v_lshl_add_u32 v20, s1, 5, v24 // (op 2044) 


v_mul_lo_u32 v24, v20, s38 // (op 2044) 

v_add_u32 v20, v23, v24 // (op 2044) 


s_mul_i32 s1, s38, s5 // (op 2046) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 2046) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 2046) least significant
s_mul_hi_u32 s18, s38, s4 // (op 2046) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2046) most significant: sum
s_add_u32 s1, s1, s18 // (op 2046) most significant: sum


s_mov_b32 s68, s40 // (op 2046) reg expression
s_mov_b32 s69, s41 // (op 2046) reg expression
s_mov_b32 s70, s0 // (op 2046) reg expression
s_mov_b32 s71, 131072 // (op 2046) call()

s_mul_hi_i32 s47, s16, s10 // (op 2063) 
v_lshrrev_b32 v23, 6, v1 // (op 2063) 
s_add_i32 s0, s47, s16 // (op 2063) 

v_and_b32 v24, 1, v23 // (op 2063) 

s_ashr_i32 s1, s0, 31 // (op 2063) 
v_and_b32 v23, 15, v1 // (op 2063) 
s_cmp_eq_i32 s10, 0 // (op 2063) 
s_mov_b32 s47, scc // (op 2063) 
v_lshl_add_u32 v25, v24, 4, v23 // (op 2063) 


s_cmp_eq_i32 s47, 0 // (op 2063) 
s_cselect_b32 s18, 0, -1 // (op 2063) 

v_and_b32 v23, 63, v1 // (op 2063) 
s_and_b32 s19, s13, 31 // (op 2063) 
v_lshrrev_b32 v24, 4, v23 // (op 2063) 

s_lshl_b32 s47, 1, s19 // (op 2063) 
s_add_i32 s22, s47, s18 // (op 2063) 


s_and_b32 s18, s1, s22 // (op 2063) 


s_add_i32 s1, s0, s18 // (op 2063) 


s_ashr_i32 s0, s1, s19 // (op 2063) 


s_lshl_b32 s1, s13, 24 // (op 2063) 
s_ashr_i32 s47, s1, 31 // (op 2063) 

s_xor_b32 s1, s0, s47 // (op 2063) 

s_sub_i32 s0, s1, s47 // (op 2063) 


s_mul_hi_i32 s1, s0, s14 // (op 2063) 
s_add_i32 s47, s1, s0 // (op 2063) 

s_ashr_i32 s1, s47, 31 // (op 2063) 
s_cmp_eq_i32 s14, 0 // (op 2063) 
s_mov_b32 s18, scc // (op 2063) 
s_cmp_eq_i32 s18, 0 // (op 2063) 
s_cselect_b32 s19, 0, -1 // (op 2063) 

s_and_b32 s18, s15, 31 // (op 2063) 
s_lshl_b32 s22, 1, s18 // (op 2063) 
s_add_i32 s23, s22, s19 // (op 2063) 


s_and_b32 s19, s1, s23 // (op 2063) 


s_add_i32 s1, s47, s19 // (op 2063) 


s_ashr_i32 s19, s1, s18 // (op 2063) 


s_lshl_b32 s1, s15, 24 // (op 2063) 
s_ashr_i32 s18, s1, 31 // (op 2063) 

s_xor_b32 s1, s19, s18 // (op 2063) 

s_sub_i32 s19, s1, s18 // (op 2063) 


s_add_u32 s22, s4, 32 // (op 2063) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 2063) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2063) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2063) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 2063) 

s_mul_i32 s1, s19, s22 // (op 2063) 

s_sub_i32 s47, s0, s1 // (op 2063) 


s_mul_hi_i32 s0, s16, s9 // (op 2063) 
s_add_i32 s1, s0, s16 // (op 2063) 

s_ashr_i32 s0, s1, 31 // (op 2063) 
s_cmp_eq_i32 s9, 0 // (op 2063) 
s_mov_b32 s18, scc // (op 2063) 
s_cmp_eq_i32 s18, 0 // (op 2063) 
s_cselect_b32 s19, 0, -1 // (op 2063) 

s_and_b32 s18, s45, 31 // (op 2063) 
s_lshl_b32 s24, 1, s18 // (op 2063) 
s_add_i32 s25, s24, s19 // (op 2063) 


s_and_b32 s19, s0, s25 // (op 2063) 


s_add_i32 s0, s1, s19 // (op 2063) 


s_ashr_i32 s1, s0, s18 // (op 2063) 


s_lshl_b32 s0, s45, 24 // (op 2063) 
s_ashr_i32 s18, s0, 31 // (op 2063) 

s_xor_b32 s0, s1, s18 // (op 2063) 

s_sub_i32 s1, s0, s18 // (op 2063) 


s_mul_i32 s0, s11, s22 // (op 2063) 

s_mul_i32 s18, s8, s0 // (op 2063) 

s_mul_i32 s0, s1, s18 // (op 2063) 

s_sub_u32 s18, s16, s0 // (op 2063) 

s_mul_hi_i32 s0, s18, s12 // (op 2063) 
s_add_i32 s19, s0, s18 // (op 2063) 


s_ashr_i32 s0, s19, 31 // (op 2063) 
s_cmp_eq_i32 s12, 0 // (op 2063) 
s_mov_b32 s18, scc // (op 2063) 
s_cmp_eq_i32 s18, 0 // (op 2063) 
s_cselect_b32 s22, 0, -1 // (op 2063) 

s_and_b32 s18, s44, 31 // (op 2063) 
s_lshl_b32 s23, 1, s18 // (op 2063) 
s_add_i32 s24, s23, s22 // (op 2063) 


s_and_b32 s25, s0, s24 // (op 2063) 


s_add_i32 s0, s19, s25 // (op 2063) 


s_ashr_i32 s19, s0, s18 // (op 2063) 


s_lshl_b32 s0, s44, 24 // (op 2063) 
s_ashr_i32 s18, s0, 31 // (op 2063) 

s_xor_b32 s0, s19, s18 // (op 2063) 

s_sub_i32 s19, s0, s18 // (op 2063) 


s_cmp_eq_i32 s1, 0 // (op 2063) 
s_mov_b32 s0, scc // (op 2063) 

s_cmp_eq_i32 s0, 0 // (op 2063) 
s_cselect_b32 s1, s19, s47 // (op 2063) 



v_lshl_add_u32 v23, s1, 5, v25 // (op 2063) 


v_mul_lo_u32 v25, v23, s38 // (op 2063) 

v_add_u32 v23, v24, v25 // (op 2063) 


s_mul_i32 s1, s38, s5 // (op 2065) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 2065) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 2065) least significant
s_mul_hi_u32 s18, s38, s4 // (op 2065) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2065) most significant: sum
s_add_u32 s1, s1, s18 // (op 2065) most significant: sum


s_mov_b32 s68, s40 // (op 2065) reg expression
s_mov_b32 s69, s41 // (op 2065) reg expression
s_mov_b32 s70, s0 // (op 2065) reg expression
s_mov_b32 s71, 131072 // (op 2065) call()




buffer_load_ubyte v24, v20, s[68:71], 0 offen offset: 0 // (op 1616) Load value
v_and_b32 v25, -8, v1 // (op 2186) 
v_sub_u32 v26, v1, v25 // (op 2186) 

v_lshlrev_b32 v25, 4, v26 // (op 2186) 

v_lshrrev_b32 v26, 3, v1 // (op 2186) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 2186) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 2186) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s47 // (op 1798) reg expression
buffer_load_dwordx4 v12, s[60:63], 0 offen offset: 0 lds // (op 1798) Load value direct to lds

v_and_b32 v25, -8, v1 // (op 2129) 
v_sub_u32 v26, v1, v25 // (op 2129) 

v_lshlrev_b32 v25, 4, v26 // (op 2129) 

v_lshrrev_b32 v26, 3, v1 // (op 2129) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 2129) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 2129) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 1803) 
v_mov_b64 v[26:27], s[0:1] // (op 1803) convert


v_mov_b64 v[28:29], v[16:17] // (op 1803) 
s_add_u32 s0, 4096, s47 // (op 1803) 
s_mov_b32 m0, s0 // (op 1803) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 1803) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 1803) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1803) most significant half

s_add_u32 m0, m0, 4096 // (op 1803) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 1803) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 1803) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1803) most significant half

s_add_u32 m0, m0, 4096 // (op 1803) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 1803) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 1803) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1803) most significant half

s_add_u32 m0, m0, 4096 // (op 1803) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 1803) Load value direct to lds





v_mov_b64 v[26:27], s[34:35] // (op 1626) convert

v_mov_b32 v25, v21 // (op 1626) 
buffer_load_ubyte v28, v25, s[64:67], 0 offen offset: 0 // (op 1626) Load value
v_add_u32 v25, v25, v26 // (op 1626) 
buffer_load_ubyte v29, v25, s[64:67], 0 offen offset: 0 // (op 1626) Load value





buffer_load_ubyte v25, v20, s[68:71], 0 offen offset:4 // (op 1630) Load value

v_and_b32 v20, -8, v1 // (op 2345) 
v_sub_u32 v26, v1, v20 // (op 2345) 

v_lshlrev_b32 v20, 4, v26 // (op 2345) 

v_lshrrev_b32 v26, 3, v1 // (op 2345) 
v_lshl_add_u32 v27, v26, 7, v20 // (op 2345) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 2345) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s0, 20480, s47 // (op 1808) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s0 // (op 1808) reg expression
s_mov_b32 s1, 128 // (op 1808) call()
buffer_load_dwordx4 v12, s[60:63], s1 offen offset: 0 lds // (op 1808) Load value direct to lds




v_and_b32 v20, -8, v1 // (op 2301) 
v_sub_u32 v12, v1, v20 // (op 2301) 

v_lshlrev_b32 v13, 4, v12 // (op 2301) 

v_lshrrev_b32 v12, 3, v1 // (op 2301) 
v_lshl_add_u32 v20, v12, 7, v13 // (op 2301) 


s_nop 0
v_readfirstlane_b32 s47, v20 // (op 2301) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 1810) 
v_mov_b64 v[12:13], s[0:1] // (op 1810) convert


v_mov_b64 v[26:27], v[16:17] // (op 1810) 
s_add_u32 s0, 24576, s47 // (op 1810) 
s_mov_b32 m0, s0 // (op 1810) reg expression
s_mov_b32 s1, 128 // (op 1810) call()
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 1810) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 1810) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 1810) most significant half

s_add_u32 m0, m0, 4096 // (op 1810) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 1810) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 1810) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 1810) most significant half

s_add_u32 m0, m0, 4096 // (op 1810) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 1810) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 1810) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 1810) most significant half

s_add_u32 m0, m0, 4096 // (op 1810) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 1810) Load value direct to lds







v_mov_b64 v[12:13], s[34:35] // (op 1640) convert

v_mov_b32 v20, v21 // (op 1640) 
buffer_load_ubyte v16, v20, s[64:67], 0 offen offset:4 // (op 1640) Load value
v_add_u32 v20, v20, v12 // (op 1640) 
buffer_load_ubyte v17, v20, s[64:67], 0 offen offset:4 // (op 1640) Load value



v_and_b32 v12, -4, v1 // (op 2239) 
v_sub_u32 v13, v1, v12 // (op 2239) 

v_lshrrev_b32 v12, 2, v1 // (op 2239) 
v_lshl_add_u32 v20, v12, 3, v13 // (op 2239) 





s_waitcnt vmcnt(9)
ds_write_b8 v20, v28 offset:40960 // (op 1644) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v20, v29 offset:40964 // (op 1644) Store local data 

s_barrier  // (op 2449) 



ds_read_b128 v[30:33], v3 // (op 1677) Load local data 
ds_read_b128 v[34:37], v3 offset:64 // (op 1677) Load local data 
s_barrier  // (op 2433) 



ds_read_b128 v[38:41], v7 offset:4096 // (op 1658) Load local data 
ds_read_b128 v[42:45], v7 offset:4160 // (op 1658) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2465) 



ds_read_u8 v12, v0 offset:40960 // (op 1681) Load local data 



ds_read_b128 v[46:49], v7 offset:8192 // (op 1663) Load local data 
ds_read_b128 v[50:53], v7 offset:8256 // (op 1663) Load local data 



ds_read_u8 v13, v0 offset:41088 // (op 1686) Load local data 



ds_read_b128 v[54:57], v7 offset:12288 // (op 1668) Load local data 
ds_read_b128 v[58:61], v7 offset:12352 // (op 1668) Load local data 



ds_read_u8 v20, v0 offset:41216 // (op 1691) Load local data 



ds_read_b128 v[62:65], v7 offset:16384 // (op 1673) Load local data 
ds_read_b128 v[66:69], v7 offset:16448 // (op 1673) Load local data 



ds_read_u8 v21, v0 offset:41344 // (op 1696) Load local data 
v_and_b32 v26, -8, v1 // (op 2101) 
v_sub_u32 v27, v1, v26 // (op 2101) 

v_lshlrev_b32 v26, 4, v27 // (op 2101) 

v_lshrrev_b32 v27, 3, v1 // (op 2101) 
v_lshl_add_u32 v70, v27, 7, v26 // (op 2101) 


s_nop 0
v_readfirstlane_b32 s47, v70 // (op 2101) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -8, v1 // (op 2158) 
v_sub_u32 v27, v1, v26 // (op 2158) 

v_lshlrev_b32 v26, 4, v27 // (op 2158) 

v_lshrrev_b32 v27, 3, v1 // (op 2158) 
v_lshl_add_u32 v70, v27, 7, v26 // (op 2158) 


s_nop 0
v_readfirstlane_b32 s0, v70 // (op 2158) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 2210) 
v_sub_u32 v27, v1, v26 // (op 2210) 

v_lshrrev_b32 v26, 2, v1 // (op 2210) 
v_lshl_add_u32 v70, v26, 3, v27 // (op 2210) 


v_and_b32 v26, -8, v1 // (op 2286) 
v_sub_u32 v27, v1, v26 // (op 2286) 

v_lshlrev_b32 v26, 4, v27 // (op 2286) 

v_lshrrev_b32 v27, 3, v1 // (op 2286) 
v_lshl_add_u32 v71, v27, 7, v26 // (op 2286) 


s_nop 0
v_readfirstlane_b32 s1, v71 // (op 2286) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -8, v1 // (op 2330) 
v_sub_u32 v27, v1, v26 // (op 2330) 

v_lshlrev_b32 v26, 4, v27 // (op 2330) 

v_lshrrev_b32 v27, 3, v1 // (op 2330) 
v_lshl_add_u32 v71, v27, 7, v26 // (op 2330) 


s_nop 0
v_readfirstlane_b32 s18, v71 // (op 2330) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 2369) 
v_sub_u32 v27, v1, v26 // (op 2369) 

v_lshrrev_b32 v26, 2, v1 // (op 2369) 
v_lshl_add_u32 v71, v26, 3, v27 // (op 2369) 


s_mov_b32 s19, 0 // (op 639) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 638) 
s_mov_b32 s24, s22 // (op 638) convert

s_and_b32 s25, s24, 4294967294 // (op 638) 

s_cmp_lt_u32 s19, s25 // (op 638) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoop_638 // (op 638) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoop_638 if false)
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoop_638:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v12, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 84) 
s_add_u32 s19, s19, 2 // (op 640) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[30:37], a[4:7], v13, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 350) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:61], v[30:37], a[8:11], v20, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 512) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:69], v[30:37], a[12:15], v21, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 674) 


v_mov_b32 v26, v23 // (op 82) 

buffer_load_ubyte v24, v26, s[68:71], 0 offen offset:8 // (op 82) Load value
s_barrier  // (op 2452) 


v_mov_b64 v[72:73], v[14:15] // (op 1770) 

s_mov_b32 m0, s0 // (op 1770) reg expression
s_mov_b32 s22, 256 // (op 1770) call()
buffer_load_dwordx4 v72, s[60:63], s22 offen offset: 0 lds // (op 1770) Load value direct to lds

s_barrier  // (op 2436) 

s_lshl_b64 s[22:23], s[26:27], 5 // (op 1765) 
v_mov_b64 v[74:75], s[22:23] // (op 1765) convert


v_mov_b64 v[76:77], v[18:19] // (op 1765) 
v_mov_b64 v[78:79], v[76:77] // (op 1765) 
s_add_u32 s22, 4096, s47 // (op 1765) 
s_mov_b32 m0, s22 // (op 1765) reg expression
s_mov_b32 s23, 256 // (op 1765) call()
buffer_load_dwordx4 v78, s[48:51], s23 offen offset: 0 lds // (op 1765) Load value direct to lds
v_add_co_u32 v78, s[24:25], v78, v74 // (op 1765) least significant half
v_addc_co_u32 v79, s[24:25], 0, v79, s[24:25] // (op 1765) most significant half

s_add_u32 m0, m0, 4096 // (op 1765) 
buffer_load_dwordx4 v78, s[48:51], s23 offen offset: 0 lds // (op 1765) Load value direct to lds
v_add_co_u32 v78, s[24:25], v78, v74 // (op 1765) least significant half
v_addc_co_u32 v79, s[24:25], 0, v79, s[24:25] // (op 1765) most significant half

s_add_u32 m0, m0, 4096 // (op 1765) 
buffer_load_dwordx4 v78, s[48:51], s23 offen offset: 0 lds // (op 1765) Load value direct to lds
v_add_co_u32 v78, s[24:25], v78, v74 // (op 1765) least significant half
v_addc_co_u32 v79, s[24:25], 0, v79, s[24:25] // (op 1765) most significant half

s_add_u32 m0, m0, 4096 // (op 1765) 
buffer_load_dwordx4 v78, s[48:51], s23 offen offset: 0 lds // (op 1765) Load value direct to lds





v_mov_b64 v[74:75], s[34:35] // (op 17) convert

v_mov_b32 v27, v22 // (op 17) 
v_mov_b32 v78, v27 // (op 17) 
buffer_load_ubyte v28, v78, s[64:67], 0 offen offset:8 // (op 17) Load value
v_add_u32 v78, v78, v74 // (op 17) 
buffer_load_ubyte v29, v78, s[64:67], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 2497) 



s_waitcnt vmcnt(9)
ds_write_b8 v71, v16 offset:41472 // (op 240) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v71, v17 offset:41476 // (op 240) Store local data 
s_barrier  // (op 2488) 



ds_read_b128 v[30:33], v4 offset:20480 // (op 233) Load local data 
ds_read_b128 v[34:37], v4 offset:20544 // (op 233) Load local data 
s_barrier  // (op 2482) 



ds_read_b128 v[38:41], v11 offset:24576 // (op 231) Load local data 
ds_read_b128 v[42:45], v11 offset:24640 // (op 231) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2494) 



ds_read_u8 v12, v10 offset:41472 // (op 235) Load local data 



ds_read_b128 v[46:49], v11 offset:28672 // (op 357) Load local data 
ds_read_b128 v[50:53], v11 offset:28736 // (op 357) Load local data 



ds_read_u8 v13, v10 offset:41600 // (op 361) Load local data 



ds_read_b128 v[54:57], v11 offset:32768 // (op 519) Load local data 
ds_read_b128 v[58:61], v11 offset:32832 // (op 519) Load local data 



ds_read_u8 v20, v10 offset:41728 // (op 523) Load local data 



ds_read_b128 v[62:65], v11 offset:36864 // (op 681) Load local data 
ds_read_b128 v[66:69], v11 offset:36928 // (op 681) Load local data 



ds_read_u8 v21, v10 offset:41856 // (op 685) Load local data 
s_barrier  // (op 2560) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v12, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 238) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[30:37], a[4:7], v13, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 351) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:61], v[30:37], a[8:11], v20, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 513) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:69], v[30:37], a[12:15], v21, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 675) 



buffer_load_ubyte v25, v26, s[68:71], 0 offen offset:12 // (op 237) Load value

s_barrier  // (op 2491) 
v_add_u32 v23, 8, v23 // (op 2055) 


s_add_u32 s22, 20480, s18 // (op 1793) 

s_mov_b32 m0, s22 // (op 1793) reg expression
s_mov_b32 s23, 384 // (op 1793) call()
buffer_load_dwordx4 v72, s[60:63], s23 offen offset: 0 lds // (op 1793) Load value direct to lds



s_barrier  // (op 2485) 
v_add_co_u32 v14, vcc, 256, v14 // (op 1927) least significant half

v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1927) most significant half
s_lshl_b64 s[22:23], s[26:27], 5 // (op 1788) 
v_mov_b64 v[72:73], s[22:23] // (op 1788) convert


v_mov_b64 v[74:75], v[76:77] // (op 1788) 
s_add_u32 s22, 24576, s1 // (op 1788) 
s_mov_b32 m0, s22 // (op 1788) reg expression
s_mov_b32 s23, 384 // (op 1788) call()
buffer_load_dwordx4 v74, s[48:51], s23 offen offset: 0 lds // (op 1788) Load value direct to lds
v_add_co_u32 v74, s[24:25], v74, v72 // (op 1788) least significant half
v_addc_co_u32 v75, s[24:25], 0, v75, s[24:25] // (op 1788) most significant half

s_add_u32 m0, m0, 4096 // (op 1788) 
buffer_load_dwordx4 v74, s[48:51], s23 offen offset: 0 lds // (op 1788) Load value direct to lds
v_add_co_u32 v74, s[24:25], v74, v72 // (op 1788) least significant half
v_addc_co_u32 v75, s[24:25], 0, v75, s[24:25] // (op 1788) most significant half

s_add_u32 m0, m0, 4096 // (op 1788) 
buffer_load_dwordx4 v74, s[48:51], s23 offen offset: 0 lds // (op 1788) Load value direct to lds
v_add_co_u32 v74, s[24:25], v74, v72 // (op 1788) least significant half
v_addc_co_u32 v75, s[24:25], 0, v75, s[24:25] // (op 1788) most significant half

s_add_u32 m0, m0, 4096 // (op 1788) 
buffer_load_dwordx4 v74, s[48:51], s23 offen offset: 0 lds // (op 1788) Load value direct to lds






v_add_co_u32 v18, vcc, 256, v18 // (op 1863) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1863) most significant half
v_mov_b64 v[72:73], s[34:35] // (op 242) convert

v_mov_b32 v26, v27 // (op 242) 
buffer_load_ubyte v16, v26, s[64:67], 0 offen offset:12 // (op 242) Load value
v_add_u32 v26, v26, v72 // (op 242) 
buffer_load_ubyte v17, v26, s[64:67], 0 offen offset:12 // (op 242) Load value



v_add_u32 v22, 8, v22 // (op 1991) 
s_barrier  // (op 2468) 



s_waitcnt vmcnt(9)
ds_write_b8 v70, v28 offset:40960 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v70, v29 offset:40964 // (op 21) Store local data 
s_barrier  // (op 2458) 



ds_read_b128 v[30:33], v3 // (op 72) Load local data 
ds_read_b128 v[34:37], v3 offset:64 // (op 72) Load local data 
s_barrier  // (op 2442) 



ds_read_b128 v[38:41], v7 offset:4096 // (op 67) Load local data 
ds_read_b128 v[42:45], v7 offset:4160 // (op 67) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2475) 



ds_read_u8 v12, v0 offset:40960 // (op 77) Load local data 



ds_read_b128 v[46:49], v7 offset:8192 // (op 344) Load local data 
ds_read_b128 v[50:53], v7 offset:8256 // (op 344) Load local data 



ds_read_u8 v13, v0 offset:41088 // (op 348) Load local data 



ds_read_b128 v[54:57], v7 offset:12288 // (op 506) Load local data 
ds_read_b128 v[58:61], v7 offset:12352 // (op 506) Load local data 



ds_read_u8 v20, v0 offset:41216 // (op 510) Load local data 



ds_read_b128 v[62:65], v7 offset:16384 // (op 668) Load local data 
ds_read_b128 v[66:69], v7 offset:16448 // (op 668) Load local data 



ds_read_u8 v21, v0 offset:41344 // (op 672) Load local data 
s_ashr_i64 s[22:23], s[42:43], 7 // (op 638) 
s_mov_b32 s24, s22 // (op 638) convert

s_and_b32 s25, s24, 4294967294 // (op 638) 

s_cmp_lt_u32 s19, s25 // (op 638) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoop_638 // (op 638) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoop_638 if true)
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoop_638:













s_barrier  // (op 2544) 
s_mul_hi_i32 s47, s16, s10 // (op 1823) 
v_lshrrev_b32 v14, 3, v1 // (op 1823) 
s_add_i32 s0, s47, s16 // (op 1823) 

v_and_b32 v15, -8, v1 // (op 1823) 
s_ashr_i32 s1, s0, 31 // (op 1823) 
v_sub_u32 v18, v1, v15 // (op 1823) 

s_cmp_eq_i32 s10, 0 // (op 1823) 
s_mov_b32 s47, scc // (op 1823) 
v_lshlrev_b32 v15, 4, v18 // (op 1823) 

s_cmp_eq_i32 s47, 0 // (op 1823) 
s_cselect_b32 s18, 0, -1 // (op 1823) 

s_and_b32 s19, s13, 31 // (op 1823) 
s_lshl_b32 s47, 1, s19 // (op 1823) 
s_add_i32 s22, s47, s18 // (op 1823) 


s_and_b32 s18, s1, s22 // (op 1823) 


s_add_i32 s1, s0, s18 // (op 1823) 


s_ashr_i32 s0, s1, s19 // (op 1823) 


s_lshl_b32 s1, s13, 24 // (op 1823) 
s_ashr_i32 s47, s1, 31 // (op 1823) 

s_xor_b32 s1, s0, s47 // (op 1823) 

s_sub_i32 s0, s1, s47 // (op 1823) 


s_mul_hi_i32 s1, s0, s14 // (op 1823) 
s_add_i32 s47, s1, s0 // (op 1823) 

s_ashr_i32 s1, s47, 31 // (op 1823) 
s_cmp_eq_i32 s14, 0 // (op 1823) 
s_mov_b32 s18, scc // (op 1823) 
s_cmp_eq_i32 s18, 0 // (op 1823) 
s_cselect_b32 s19, 0, -1 // (op 1823) 

s_and_b32 s18, s15, 31 // (op 1823) 
s_lshl_b32 s22, 1, s18 // (op 1823) 
s_add_i32 s23, s22, s19 // (op 1823) 


s_and_b32 s19, s1, s23 // (op 1823) 


s_add_i32 s1, s47, s19 // (op 1823) 


s_ashr_i32 s19, s1, s18 // (op 1823) 


s_lshl_b32 s1, s15, 24 // (op 1823) 
s_ashr_i32 s18, s1, 31 // (op 1823) 

s_xor_b32 s1, s19, s18 // (op 1823) 

s_sub_i32 s19, s1, s18 // (op 1823) 


s_mul_i32 s1, s0, s11 // (op 1823) 

s_sub_u32 s0, s16, s1 // (op 1823) 

s_mul_i32 s1, s19, s11 // (op 1823) 
s_add_u32 s1, s1, s0 // (op 1823) 


s_mul_hi_i32 s0, s16, s9 // (op 1823) 
s_add_i32 s47, s0, s16 // (op 1823) 

s_ashr_i32 s0, s47, 31 // (op 1823) 
s_cmp_eq_i32 s9, 0 // (op 1823) 
s_mov_b32 s18, scc // (op 1823) 
s_cmp_eq_i32 s18, 0 // (op 1823) 
s_cselect_b32 s19, 0, -1 // (op 1823) 

s_and_b32 s18, s45, 31 // (op 1823) 
s_lshl_b32 s22, 1, s18 // (op 1823) 
s_add_i32 s23, s22, s19 // (op 1823) 


s_and_b32 s19, s0, s23 // (op 1823) 


s_add_i32 s0, s47, s19 // (op 1823) 


s_ashr_i32 s19, s0, s18 // (op 1823) 


s_lshl_b32 s0, s45, 24 // (op 1823) 
s_ashr_i32 s18, s0, 31 // (op 1823) 

s_xor_b32 s0, s19, s18 // (op 1823) 

s_sub_i32 s19, s0, s18 // (op 1823) 


s_add_u32 s22, s4, 32 // (op 1823) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1823) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1823) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1823) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1823) 

s_mul_i32 s0, s11, s22 // (op 1823) 

s_mul_i32 s18, s8, s0 // (op 1823) 

s_mul_i32 s0, s19, s18 // (op 1823) 

s_sub_u32 s18, s16, s0 // (op 1823) 

s_mov_b32 s0, s18 // (op 1823) 
s_mul_hi_i32 s47, s0, s12 // (op 1823) 
s_add_i32 s22, s47, s0 // (op 1823) 


s_ashr_i32 s0, s22, 31 // (op 1823) 
s_cmp_eq_i32 s12, 0 // (op 1823) 
s_mov_b32 s47, scc // (op 1823) 
s_cmp_eq_i32 s47, 0 // (op 1823) 
s_cselect_b32 s23, 0, -1 // (op 1823) 

s_and_b32 s47, s44, 31 // (op 1823) 
s_lshl_b32 s24, 1, s47 // (op 1823) 
s_add_i32 s25, s24, s23 // (op 1823) 


s_and_b32 s23, s0, s25 // (op 1823) 


s_add_i32 s0, s22, s23 // (op 1823) 


s_ashr_i32 s22, s0, s47 // (op 1823) 


s_lshl_b32 s0, s44, 24 // (op 1823) 
s_ashr_i32 s47, s0, 31 // (op 1823) 

s_xor_b32 s0, s22, s47 // (op 1823) 

s_sub_i32 s22, s0, s47 // (op 1823) 


s_mul_i32 s0, s46, s11 // (op 1823) 
s_add_u32 s24, s6, 128 // (op 1823) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1823) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 1823) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 1823) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 7 // (op 1823) 

s_sub_i32 s23, s24, s0 // (op 1823) 

s_mul_i32 s47, s22, s23 // (op 1823) 


s_sub_u32 s22, s18, s47 // (op 1823) 


s_add_u32 s18, s22, s0 // (op 1823) 


s_cmp_eq_i32 s19, 0 // (op 1823) 
s_mov_b32 s0, scc // (op 1823) 

s_cmp_eq_i32 s0, 0 // (op 1823) 
s_cselect_b32 s19, s18, s1 // (op 1823) 



v_lshl_add_u32 v18, s19, 7, v14 // (op 1823) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1823) 
s_lshr_b32 s47, s0, 1 // (op 1823) 

v_lshl_add_u32 v14, s47, 8, v15 // (op 1823) 


v_mov_b32 v23, 0 // (op 1823) convert
v_mov_b32 v22, v14 // (op 1823) convert

v_mul_lo_u32 v15, v18, s27 // (op 1823) most significant: low of low * high
v_mul_hi_u32 v19, v18, s26 // (op 1823) most significant: high of low * low
v_mul_lo_u32 v14, v18, s26 // (op 1823) least significant: low of low * low
v_add_u32 v15, v15, v19 // (op 1823) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v22 // (op 1823) least significant half
v_addc_co_u32 v15, s[0:1], v15, v23, s[0:1] // (op 1823) most significant half



s_mul_i32 s1, s26, s7 // (op 1825) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 1825) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 1825) least significant
s_mul_hi_u32 s18, s26, s6 // (op 1825) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1825) most significant: sum
s_add_u32 s1, s1, s18 // (op 1825) most significant: sum


s_mov_b32 s48, s28 // (op 1825) reg expression
s_mov_b32 s49, s29 // (op 1825) reg expression
s_mov_b32 s50, s0 // (op 1825) reg expression
s_mov_b32 s51, 131072 // (op 1825) call()

s_mul_hi_i32 s47, s16, s10 // (op 1893) 
v_lshrrev_b32 v18, 3, v1 // (op 1893) 
s_add_i32 s0, s47, s16 // (op 1893) 

v_and_b32 v19, -8, v1 // (op 1893) 
s_ashr_i32 s1, s0, 31 // (op 1893) 
v_sub_u32 v22, v1, v19 // (op 1893) 

s_cmp_eq_i32 s10, 0 // (op 1893) 
s_mov_b32 s47, scc // (op 1893) 
v_lshlrev_b32 v19, 4, v22 // (op 1893) 

s_cmp_eq_i32 s47, 0 // (op 1893) 
s_cselect_b32 s18, 0, -1 // (op 1893) 

s_and_b32 s19, s13, 31 // (op 1893) 
s_lshl_b32 s47, 1, s19 // (op 1893) 
s_add_i32 s22, s47, s18 // (op 1893) 


s_and_b32 s18, s1, s22 // (op 1893) 


s_add_i32 s1, s0, s18 // (op 1893) 


s_ashr_i32 s0, s1, s19 // (op 1893) 


s_lshl_b32 s1, s13, 24 // (op 1893) 
s_ashr_i32 s47, s1, 31 // (op 1893) 

s_xor_b32 s1, s0, s47 // (op 1893) 

s_sub_i32 s0, s1, s47 // (op 1893) 


s_mul_hi_i32 s1, s0, s14 // (op 1893) 
s_add_i32 s47, s1, s0 // (op 1893) 

s_ashr_i32 s1, s47, 31 // (op 1893) 
s_cmp_eq_i32 s14, 0 // (op 1893) 
s_mov_b32 s18, scc // (op 1893) 
s_cmp_eq_i32 s18, 0 // (op 1893) 
s_cselect_b32 s19, 0, -1 // (op 1893) 

s_and_b32 s18, s15, 31 // (op 1893) 
s_lshl_b32 s22, 1, s18 // (op 1893) 
s_add_i32 s23, s22, s19 // (op 1893) 


s_and_b32 s19, s1, s23 // (op 1893) 


s_add_i32 s1, s47, s19 // (op 1893) 


s_ashr_i32 s19, s1, s18 // (op 1893) 


s_lshl_b32 s1, s15, 24 // (op 1893) 
s_ashr_i32 s18, s1, 31 // (op 1893) 

s_xor_b32 s1, s19, s18 // (op 1893) 

s_sub_i32 s19, s1, s18 // (op 1893) 


s_add_u32 s22, s4, 32 // (op 1893) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1893) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1893) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1893) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1893) 

s_mul_i32 s1, s19, s22 // (op 1893) 

s_sub_i32 s47, s0, s1 // (op 1893) 


s_mul_hi_i32 s0, s16, s9 // (op 1893) 
s_add_i32 s1, s0, s16 // (op 1893) 

s_ashr_i32 s0, s1, 31 // (op 1893) 
s_cmp_eq_i32 s9, 0 // (op 1893) 
s_mov_b32 s18, scc // (op 1893) 
s_cmp_eq_i32 s18, 0 // (op 1893) 
s_cselect_b32 s19, 0, -1 // (op 1893) 

s_and_b32 s18, s45, 31 // (op 1893) 
s_lshl_b32 s24, 1, s18 // (op 1893) 
s_add_i32 s25, s24, s19 // (op 1893) 


s_and_b32 s19, s0, s25 // (op 1893) 


s_add_i32 s0, s1, s19 // (op 1893) 


s_ashr_i32 s1, s0, s18 // (op 1893) 


s_lshl_b32 s0, s45, 24 // (op 1893) 
s_ashr_i32 s18, s0, 31 // (op 1893) 

s_xor_b32 s0, s1, s18 // (op 1893) 

s_sub_i32 s1, s0, s18 // (op 1893) 


s_mul_i32 s0, s11, s22 // (op 1893) 

s_mul_i32 s18, s8, s0 // (op 1893) 

s_mul_i32 s0, s1, s18 // (op 1893) 

s_sub_u32 s18, s16, s0 // (op 1893) 

s_mul_hi_i32 s0, s18, s12 // (op 1893) 
s_add_i32 s19, s0, s18 // (op 1893) 


s_ashr_i32 s0, s19, 31 // (op 1893) 
s_cmp_eq_i32 s12, 0 // (op 1893) 
s_mov_b32 s18, scc // (op 1893) 
s_cmp_eq_i32 s18, 0 // (op 1893) 
s_cselect_b32 s22, 0, -1 // (op 1893) 

s_and_b32 s18, s44, 31 // (op 1893) 
s_lshl_b32 s23, 1, s18 // (op 1893) 
s_add_i32 s24, s23, s22 // (op 1893) 


s_and_b32 s25, s0, s24 // (op 1893) 


s_add_i32 s0, s19, s25 // (op 1893) 


s_ashr_i32 s19, s0, s18 // (op 1893) 


s_lshl_b32 s0, s44, 24 // (op 1893) 
s_ashr_i32 s18, s0, 31 // (op 1893) 

s_xor_b32 s0, s19, s18 // (op 1893) 

s_sub_i32 s19, s0, s18 // (op 1893) 


s_cmp_eq_i32 s1, 0 // (op 1893) 
s_mov_b32 s0, scc // (op 1893) 

s_cmp_eq_i32 s0, 0 // (op 1893) 
s_cselect_b32 s1, s19, s47 // (op 1893) 



v_lshl_add_u32 v22, s1, 5, v18 // (op 1893) 


v_mul_lo_u32 v27, v22, s31 // (op 1893) most significant: low of low * high
v_mul_hi_u32 v18, v22, s30 // (op 1893) most significant: high of low * low
v_mul_lo_u32 v26, v22, s30 // (op 1893) least significant: low of low * low
v_add_u32 v27, v27, v18 // (op 1893) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1893) 
s_lshr_b32 s47, s0, 1 // (op 1893) 

v_lshl_add_u32 v18, s47, 8, v19 // (op 1893) 


v_mov_b32 v29, 0 // (op 1893) convert
v_mov_b32 v28, v18 // (op 1893) convert

v_add_co_u32 v18, s[0:1], v28, v26 // (op 1893) least significant half
v_addc_co_u32 v19, s[0:1], v29, v27, s[0:1] // (op 1893) most significant half



s_mul_i32 s1, s30, s5 // (op 1895) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 1895) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 1895) least significant
s_mul_hi_u32 s18, s30, s4 // (op 1895) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1895) most significant: sum
s_add_u32 s1, s1, s18 // (op 1895) most significant: sum


s_mov_b32 s60, s32 // (op 1895) reg expression
s_mov_b32 s61, s33 // (op 1895) reg expression
s_mov_b32 s62, s0 // (op 1895) reg expression
s_mov_b32 s63, 131072 // (op 1895) call()

s_mul_hi_i32 s47, s16, s10 // (op 1957) 
v_lshrrev_b32 v22, 2, v1 // (op 1957) 
s_add_i32 s0, s47, s16 // (op 1957) 

v_lshlrev_b32 v23, 1, v22 // (op 1957) 

s_ashr_i32 s1, s0, 31 // (op 1957) 
v_and_b32 v22, -4, v1 // (op 1957) 
s_cmp_eq_i32 s10, 0 // (op 1957) 
s_mov_b32 s47, scc // (op 1957) 
v_sub_u32 v24, v1, v22 // (op 1957) 

s_cmp_eq_i32 s47, 0 // (op 1957) 
s_cselect_b32 s18, 0, -1 // (op 1957) 

s_and_b32 s19, s13, 31 // (op 1957) 
s_lshl_b32 s47, 1, s19 // (op 1957) 
s_add_i32 s22, s47, s18 // (op 1957) 


s_and_b32 s18, s1, s22 // (op 1957) 


s_add_i32 s1, s0, s18 // (op 1957) 


s_ashr_i32 s0, s1, s19 // (op 1957) 


s_lshl_b32 s1, s13, 24 // (op 1957) 
s_ashr_i32 s47, s1, 31 // (op 1957) 

s_xor_b32 s1, s0, s47 // (op 1957) 

s_sub_i32 s0, s1, s47 // (op 1957) 


s_mul_hi_i32 s1, s0, s14 // (op 1957) 
s_add_i32 s47, s1, s0 // (op 1957) 

s_ashr_i32 s1, s47, 31 // (op 1957) 
s_cmp_eq_i32 s14, 0 // (op 1957) 
s_mov_b32 s18, scc // (op 1957) 
s_cmp_eq_i32 s18, 0 // (op 1957) 
s_cselect_b32 s19, 0, -1 // (op 1957) 

s_and_b32 s18, s15, 31 // (op 1957) 
s_lshl_b32 s22, 1, s18 // (op 1957) 
s_add_i32 s23, s22, s19 // (op 1957) 


s_and_b32 s19, s1, s23 // (op 1957) 


s_add_i32 s1, s47, s19 // (op 1957) 


s_ashr_i32 s19, s1, s18 // (op 1957) 


s_lshl_b32 s1, s15, 24 // (op 1957) 
s_ashr_i32 s18, s1, 31 // (op 1957) 

s_xor_b32 s1, s19, s18 // (op 1957) 

s_sub_i32 s19, s1, s18 // (op 1957) 


s_mul_i32 s1, s0, s11 // (op 1957) 

s_sub_u32 s0, s16, s1 // (op 1957) 

s_mul_i32 s1, s19, s11 // (op 1957) 
s_add_u32 s1, s1, s0 // (op 1957) 


s_mul_hi_i32 s0, s16, s9 // (op 1957) 
s_add_i32 s47, s0, s16 // (op 1957) 

s_ashr_i32 s0, s47, 31 // (op 1957) 
s_cmp_eq_i32 s9, 0 // (op 1957) 
s_mov_b32 s18, scc // (op 1957) 
s_cmp_eq_i32 s18, 0 // (op 1957) 
s_cselect_b32 s19, 0, -1 // (op 1957) 

s_and_b32 s18, s45, 31 // (op 1957) 
s_lshl_b32 s22, 1, s18 // (op 1957) 
s_add_i32 s23, s22, s19 // (op 1957) 


s_and_b32 s19, s0, s23 // (op 1957) 


s_add_i32 s0, s47, s19 // (op 1957) 


s_ashr_i32 s19, s0, s18 // (op 1957) 


s_lshl_b32 s0, s45, 24 // (op 1957) 
s_ashr_i32 s18, s0, 31 // (op 1957) 

s_xor_b32 s0, s19, s18 // (op 1957) 

s_sub_i32 s19, s0, s18 // (op 1957) 


s_add_u32 s22, s4, 32 // (op 1957) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1957) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1957) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1957) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1957) 

s_mul_i32 s0, s11, s22 // (op 1957) 

s_mul_i32 s18, s8, s0 // (op 1957) 

s_mul_i32 s0, s19, s18 // (op 1957) 

s_sub_u32 s18, s16, s0 // (op 1957) 

s_mov_b32 s0, s18 // (op 1957) 
s_mul_hi_i32 s47, s0, s12 // (op 1957) 
s_add_i32 s22, s47, s0 // (op 1957) 


s_ashr_i32 s0, s22, 31 // (op 1957) 
s_cmp_eq_i32 s12, 0 // (op 1957) 
s_mov_b32 s47, scc // (op 1957) 
s_cmp_eq_i32 s47, 0 // (op 1957) 
s_cselect_b32 s23, 0, -1 // (op 1957) 

s_and_b32 s47, s44, 31 // (op 1957) 
s_lshl_b32 s24, 1, s47 // (op 1957) 
s_add_i32 s25, s24, s23 // (op 1957) 


s_and_b32 s23, s0, s25 // (op 1957) 


s_add_i32 s0, s22, s23 // (op 1957) 


s_ashr_i32 s22, s0, s47 // (op 1957) 


s_lshl_b32 s0, s44, 24 // (op 1957) 
s_ashr_i32 s47, s0, 31 // (op 1957) 

s_xor_b32 s0, s22, s47 // (op 1957) 

s_sub_i32 s22, s0, s47 // (op 1957) 


s_mul_i32 s0, s46, s11 // (op 1957) 
s_add_u32 s24, s6, 128 // (op 1957) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1957) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 1957) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 1957) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 7 // (op 1957) 

s_sub_i32 s23, s24, s0 // (op 1957) 

s_mul_i32 s47, s22, s23 // (op 1957) 


s_sub_u32 s22, s18, s47 // (op 1957) 


s_add_u32 s18, s22, s0 // (op 1957) 


s_cmp_eq_i32 s19, 0 // (op 1957) 
s_mov_b32 s0, scc // (op 1957) 

s_cmp_eq_i32 s0, 0 // (op 1957) 
s_cselect_b32 s19, s18, s1 // (op 1957) 



v_lshl_add_u32 v22, s19, 7, v23 // (op 1957) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1957) 
s_lshr_b32 s47, s0, 1 // (op 1957) 

v_lshl_add_u32 v23, s47, 3, v24 // (op 1957) 


v_mul_lo_u32 v24, v22, s34 // (op 1957) 
v_add_u32 v24, v24, v23 // (op 1957) 


s_mul_i32 s1, s34, s7 // (op 1959) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1959) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1959) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1959) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1959) most significant: sum
s_add_u32 s1, s1, s18 // (op 1959) most significant: sum


s_mov_b32 s64, s36 // (op 1959) reg expression
s_mov_b32 s65, s37 // (op 1959) reg expression
s_mov_b32 s66, s0 // (op 1959) reg expression
s_mov_b32 s67, 131072 // (op 1959) call()

s_mul_hi_i32 s47, s16, s10 // (op 2021) 
v_lshrrev_b32 v22, 6, v1 // (op 2021) 
s_add_i32 s0, s47, s16 // (op 2021) 

v_and_b32 v23, 1, v22 // (op 2021) 

s_ashr_i32 s1, s0, 31 // (op 2021) 
v_and_b32 v22, 15, v1 // (op 2021) 
s_cmp_eq_i32 s10, 0 // (op 2021) 
s_mov_b32 s47, scc // (op 2021) 
v_lshl_add_u32 v26, v23, 4, v22 // (op 2021) 


s_cmp_eq_i32 s47, 0 // (op 2021) 
s_cselect_b32 s18, 0, -1 // (op 2021) 

v_and_b32 v22, 63, v1 // (op 2021) 
s_and_b32 s19, s13, 31 // (op 2021) 
v_lshrrev_b32 v23, 4, v22 // (op 2021) 

s_lshl_b32 s47, 1, s19 // (op 2021) 
s_add_i32 s22, s47, s18 // (op 2021) 


s_and_b32 s18, s1, s22 // (op 2021) 


s_add_i32 s1, s0, s18 // (op 2021) 


s_ashr_i32 s0, s1, s19 // (op 2021) 


s_lshl_b32 s1, s13, 24 // (op 2021) 
s_ashr_i32 s47, s1, 31 // (op 2021) 

s_xor_b32 s1, s0, s47 // (op 2021) 

s_sub_i32 s0, s1, s47 // (op 2021) 


s_mul_hi_i32 s1, s0, s14 // (op 2021) 
s_add_i32 s47, s1, s0 // (op 2021) 

s_ashr_i32 s1, s47, 31 // (op 2021) 
s_cmp_eq_i32 s14, 0 // (op 2021) 
s_mov_b32 s18, scc // (op 2021) 
s_cmp_eq_i32 s18, 0 // (op 2021) 
s_cselect_b32 s19, 0, -1 // (op 2021) 

s_and_b32 s18, s15, 31 // (op 2021) 
s_lshl_b32 s22, 1, s18 // (op 2021) 
s_add_i32 s23, s22, s19 // (op 2021) 


s_and_b32 s19, s1, s23 // (op 2021) 


s_add_i32 s1, s47, s19 // (op 2021) 


s_ashr_i32 s19, s1, s18 // (op 2021) 


s_lshl_b32 s1, s15, 24 // (op 2021) 
s_ashr_i32 s18, s1, 31 // (op 2021) 

s_xor_b32 s1, s19, s18 // (op 2021) 

s_sub_i32 s19, s1, s18 // (op 2021) 


s_add_u32 s22, s4, 32 // (op 2021) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 2021) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2021) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2021) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 2021) 

s_mul_i32 s1, s19, s22 // (op 2021) 

s_sub_i32 s47, s0, s1 // (op 2021) 


s_mul_hi_i32 s0, s16, s9 // (op 2021) 
s_add_i32 s1, s0, s16 // (op 2021) 

s_ashr_i32 s0, s1, 31 // (op 2021) 
s_cmp_eq_i32 s9, 0 // (op 2021) 
s_mov_b32 s18, scc // (op 2021) 
s_cmp_eq_i32 s18, 0 // (op 2021) 
s_cselect_b32 s19, 0, -1 // (op 2021) 

s_and_b32 s18, s45, 31 // (op 2021) 
s_lshl_b32 s24, 1, s18 // (op 2021) 
s_add_i32 s25, s24, s19 // (op 2021) 


s_and_b32 s19, s0, s25 // (op 2021) 


s_add_i32 s0, s1, s19 // (op 2021) 


s_ashr_i32 s1, s0, s18 // (op 2021) 


s_lshl_b32 s0, s45, 24 // (op 2021) 
s_ashr_i32 s18, s0, 31 // (op 2021) 

s_xor_b32 s0, s1, s18 // (op 2021) 

s_sub_i32 s1, s0, s18 // (op 2021) 


s_mul_i32 s0, s11, s22 // (op 2021) 

s_mul_i32 s18, s8, s0 // (op 2021) 

s_mul_i32 s0, s1, s18 // (op 2021) 

s_sub_u32 s18, s16, s0 // (op 2021) 

s_mul_hi_i32 s0, s18, s12 // (op 2021) 
s_add_i32 s19, s0, s18 // (op 2021) 


s_ashr_i32 s0, s19, 31 // (op 2021) 
s_cmp_eq_i32 s12, 0 // (op 2021) 
s_mov_b32 s18, scc // (op 2021) 
s_cmp_eq_i32 s18, 0 // (op 2021) 
s_cselect_b32 s22, 0, -1 // (op 2021) 

s_and_b32 s18, s44, 31 // (op 2021) 
s_lshl_b32 s23, 1, s18 // (op 2021) 
s_add_i32 s24, s23, s22 // (op 2021) 


s_and_b32 s25, s0, s24 // (op 2021) 


s_add_i32 s0, s19, s25 // (op 2021) 


s_ashr_i32 s19, s0, s18 // (op 2021) 


s_lshl_b32 s0, s44, 24 // (op 2021) 
s_ashr_i32 s18, s0, 31 // (op 2021) 

s_xor_b32 s0, s19, s18 // (op 2021) 

s_sub_i32 s19, s0, s18 // (op 2021) 


s_cmp_eq_i32 s1, 0 // (op 2021) 
s_mov_b32 s0, scc // (op 2021) 

s_cmp_eq_i32 s0, 0 // (op 2021) 
s_cselect_b32 s1, s19, s47 // (op 2021) 



v_lshl_add_u32 v22, s1, 5, v26 // (op 2021) 


v_mul_lo_u32 v26, v22, s38 // (op 2021) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 2021) 
s_lshr_b32 s47, s0, 1 // (op 2021) 

v_lshl_add_u32 v22, s47, 3, v23 // (op 2021) 


v_add_u32 v23, v22, v26 // (op 2021) 


s_mul_i32 s1, s38, s5 // (op 2023) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 2023) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 2023) least significant
s_mul_hi_u32 s18, s38, s4 // (op 2023) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2023) most significant: sum
s_add_u32 s1, s1, s18 // (op 2023) most significant: sum


s_mov_b32 s68, s40 // (op 2023) reg expression
s_mov_b32 s69, s41 // (op 2023) reg expression
s_mov_b32 s70, s0 // (op 2023) reg expression
s_mov_b32 s71, 131072 // (op 2023) call()

v_and_b32 v22, -8, v1 // (op 2114) 
v_sub_u32 v26, v1, v22 // (op 2114) 

v_lshlrev_b32 v22, 4, v26 // (op 2114) 

v_lshrrev_b32 v26, 3, v1 // (op 2114) 
v_lshl_add_u32 v27, v26, 7, v22 // (op 2114) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 2114) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 2171) 
v_sub_u32 v26, v1, v22 // (op 2171) 

v_lshlrev_b32 v22, 4, v26 // (op 2171) 

v_lshrrev_b32 v26, 3, v1 // (op 2171) 
v_lshl_add_u32 v27, v26, 7, v22 // (op 2171) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2171) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 2223) 
v_sub_u32 v26, v1, v22 // (op 2223) 

v_lshrrev_b32 v22, 2, v1 // (op 2223) 
v_lshl_add_u32 v27, v22, 3, v26 // (op 2223) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 644) 
s_and_b32 s1, s18, 4294967294 // (op 644) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 643) 
s_mov_b32 s22, s18 // (op 643) convert

s_cmp_lt_u32 s1, s22 // (op 643) 












s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_643 // (op 643) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_643 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoopTail_643:
s_barrier  // (op 2439) 
s_add_u32 s1, s1, 1 // (op 645) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 1775) 
v_mov_b64 v[12:13], s[18:19] // (op 1775) convert


v_mov_b64 v[16:17], v[14:15] // (op 1775) 
v_mov_b64 v[20:21], v[16:17] // (op 1775) 
s_add_u32 s18, 4096, s47 // (op 1775) 
s_mov_b32 m0, s18 // (op 1775) reg expression
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 1775) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v12 // (op 1775) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 1775) most significant half

s_add_u32 m0, m0, 4096 // (op 1775) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 1775) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v12 // (op 1775) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 1775) most significant half

s_add_u32 m0, m0, 4096 // (op 1775) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 1775) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v12 // (op 1775) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 1775) most significant half

s_add_u32 m0, m0, 4096 // (op 1775) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 1775) Load value direct to lds




v_add_co_u32 v14, vcc, 128, v14 // (op 1813) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1813) most significant half
s_barrier  // (op 2455) 


v_mov_b64 v[12:13], v[18:19] // (op 1781) 

s_mov_b32 m0, s0 // (op 1781) reg expression
buffer_load_dwordx4 v12, s[60:63], 0 offen offset: 0 lds // (op 1781) Load value direct to lds

v_add_co_u32 v18, vcc, 128, v18 // (op 1885) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1885) most significant half

v_mov_b64 v[12:13], s[34:35] // (op 184) convert

v_mov_b32 v16, v24 // (op 184) 
v_mov_b32 v17, v16 // (op 184) 
buffer_load_ubyte v20, v17, s[64:67], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v17, v17, v12 // (op 184) 
buffer_load_ubyte v21, v17, s[64:67], 0 offen offset: 0 // (op 184) Load value



v_add_u32 v24, 4, v24 // (op 1949) 
s_barrier  // (op 2471) 
s_waitcnt vmcnt(3)
s_barrier  // (op 2445) 






ds_read_b128 v[28:31], v7 offset:4096 // (op 173) Load local data 
ds_read_b128 v[32:35], v7 offset:4160 // (op 173) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v27, v20 offset:40960 // (op 182) Store local data 
s_barrier  // (op 2461) 
s_waitcnt vmcnt(0)
ds_write_b8 v27, v21 offset:40964 // (op 182) Store local data 




ds_read_b128 v[36:39], v3 // (op 175) Load local data 
ds_read_b128 v[40:43], v3 offset:64 // (op 175) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2478) 



ds_read_u8 v12, v0 offset:40960 // (op 177) Load local data 


v_mov_b32 v13, v23 // (op 179) 

buffer_load_ubyte v16, v13, s[68:71], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v23, 4, v23 // (op 2013) 



ds_read_b128 v[44:47], v7 offset:8192 // (op 325) Load local data 
ds_read_b128 v[48:51], v7 offset:8256 // (op 325) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:43], a[0:3], v12, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 180) 





ds_read_u8 v17, v0 offset:41088 // (op 329) Load local data 



ds_read_b128 v[28:31], v7 offset:12288 // (op 487) Load local data 
ds_read_b128 v[32:35], v7 offset:12352 // (op 487) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[36:43], a[4:7], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 331) 





ds_read_u8 v17, v0 offset:41216 // (op 491) Load local data 



ds_read_b128 v[44:47], v7 offset:16384 // (op 649) Load local data 
ds_read_b128 v[48:51], v7 offset:16448 // (op 649) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:35], v[36:43], a[8:11], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 493) 





ds_read_u8 v17, v0 offset:41344 // (op 653) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[44:51], v[36:43], a[12:15], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 655) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 643) 
s_mov_b32 s22, s18 // (op 643) convert

s_cmp_lt_u32 s1, s22 // (op 643) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoopTail_643 // (op 643) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoopTail_643 if true)
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_643:





















s_barrier  // (op 2548) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 

v_lshrrev_b32 v10, 16, v14 // (op 52) 
v_lshrrev_b32 v11, 16, v15 // (op 52) 
v_lshrrev_b32 v12, 16, v16 // (op 52) 
v_lshrrev_b32 v13, 16, v17 // (op 52) 


v_pack_b32_f16 v14, v10, v11 // (op 54) 
v_pack_b32_f16 v15, v12, v13 // (op 54) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen // (op 54) Store value


v_accvgpr_read v10, a4 // (op 885) 
v_accvgpr_read v11, a5 // (op 885) 
v_accvgpr_read v12, a6 // (op 885) 
v_accvgpr_read v13, a7 // (op 885) 
v_mul_f32 v14, v8, v10 // (op 885) 
v_mul_f32 v15, v8, v11 // (op 885) 
v_mul_f32 v16, v8, v12 // (op 885) 
v_mul_f32 v17, v8, v13 // (op 885) 

v_lshrrev_b32 v10, 16, v14 // (op 887) 
v_lshrrev_b32 v11, 16, v15 // (op 887) 
v_lshrrev_b32 v12, 16, v16 // (op 887) 
v_lshrrev_b32 v13, 16, v17 // (op 887) 


v_pack_b32_f16 v14, v10, v11 // (op 888) 
v_pack_b32_f16 v15, v12, v13 // (op 888) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen offset:64 // (op 888) Store value


v_accvgpr_read v10, a8 // (op 905) 
v_accvgpr_read v11, a9 // (op 905) 
v_accvgpr_read v12, a10 // (op 905) 
v_accvgpr_read v13, a11 // (op 905) 
v_mul_f32 v14, v8, v10 // (op 905) 
v_mul_f32 v15, v8, v11 // (op 905) 
v_mul_f32 v16, v8, v12 // (op 905) 
v_mul_f32 v17, v8, v13 // (op 905) 

v_lshrrev_b32 v10, 16, v14 // (op 907) 
v_lshrrev_b32 v11, 16, v15 // (op 907) 
v_lshrrev_b32 v12, 16, v16 // (op 907) 
v_lshrrev_b32 v13, 16, v17 // (op 907) 


v_pack_b32_f16 v14, v10, v11 // (op 908) 
v_pack_b32_f16 v15, v12, v13 // (op 908) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen offset:128 // (op 908) Store value


v_accvgpr_read v10, a12 // (op 925) 
v_accvgpr_read v11, a13 // (op 925) 
v_accvgpr_read v12, a14 // (op 925) 
v_accvgpr_read v13, a15 // (op 925) 
v_mul_f32 v14, v8, v10 // (op 925) 
v_mul_f32 v15, v8, v11 // (op 925) 
v_mul_f32 v16, v8, v12 // (op 925) 
v_mul_f32 v17, v8, v13 // (op 925) 

v_lshrrev_b32 v10, 16, v14 // (op 927) 
v_lshrrev_b32 v11, 16, v15 // (op 927) 
v_lshrrev_b32 v12, 16, v16 // (op 927) 
v_lshrrev_b32 v13, 16, v17 // (op 927) 


v_pack_b32_f16 v14, v10, v11 // (op 928) 
v_pack_b32_f16 v15, v12, v13 // (op 928) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen offset:192 // (op 928) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:192 // (op 1008) Load value
v_accvgpr_read v12, a12 // (op 1004) 
v_accvgpr_read v13, a13 // (op 1004) 
v_accvgpr_read v14, a14 // (op 1004) 
v_accvgpr_read v15, a15 // (op 1004) 
v_mul_f32 v16, v8, v12 // (op 1004) 
v_mul_f32 v17, v8, v13 // (op 1004) 
v_mul_f32 v18, v8, v14 // (op 1004) 
v_mul_f32 v19, v8, v15 // (op 1004) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 1009) 
v_lshrrev_b32 v13, 16, v10 // (op 1009) 
v_lshlrev_b32 v12, 16, v12 // (op 1009) 
v_lshlrev_b32 v13, 16, v13 // (op 1009) 
v_mul_f32 v20, v5, v12 // (op 1009) 
v_mul_f32 v21, v5, v13 // (op 1009) 
v_and_b32 v12, 65535, v11 // (op 1009) 
v_lshrrev_b32 v13, 16, v11 // (op 1009) 
v_lshlrev_b32 v12, 16, v12 // (op 1009) 
v_lshlrev_b32 v13, 16, v13 // (op 1009) 
v_mul_f32 v22, v5, v12 // (op 1009) 
v_mul_f32 v23, v5, v13 // (op 1009) 


v_add_f32 v10, v20, v16 // (op 1005) 
v_add_f32 v11, v21, v17 // (op 1005) 
v_add_f32 v12, v22, v18 // (op 1005) 
v_add_f32 v13, v23, v19 // (op 1005) 


v_lshrrev_b32 v14, 16, v10 // (op 1006) 
v_lshrrev_b32 v15, 16, v11 // (op 1006) 
v_lshrrev_b32 v16, 16, v12 // (op 1006) 
v_lshrrev_b32 v17, 16, v13 // (op 1006) 


v_pack_b32_f16 v10, v14, v15 // (op 1007) 
v_pack_b32_f16 v11, v16, v17 // (op 1007) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen offset:192 // (op 1007) Store value





buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:128 // (op 988) Load value
v_accvgpr_read v12, a8 // (op 984) 
v_accvgpr_read v13, a9 // (op 984) 
v_accvgpr_read v14, a10 // (op 984) 
v_accvgpr_read v15, a11 // (op 984) 
v_mul_f32 v16, v8, v12 // (op 984) 
v_mul_f32 v17, v8, v13 // (op 984) 
v_mul_f32 v18, v8, v14 // (op 984) 
v_mul_f32 v19, v8, v15 // (op 984) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 989) 
v_lshrrev_b32 v13, 16, v10 // (op 989) 
v_lshlrev_b32 v12, 16, v12 // (op 989) 
v_lshlrev_b32 v13, 16, v13 // (op 989) 
v_mul_f32 v20, v5, v12 // (op 989) 
v_mul_f32 v21, v5, v13 // (op 989) 
v_and_b32 v12, 65535, v11 // (op 989) 
v_lshrrev_b32 v13, 16, v11 // (op 989) 
v_lshlrev_b32 v12, 16, v12 // (op 989) 
v_lshlrev_b32 v13, 16, v13 // (op 989) 
v_mul_f32 v22, v5, v12 // (op 989) 
v_mul_f32 v23, v5, v13 // (op 989) 


v_add_f32 v10, v20, v16 // (op 985) 
v_add_f32 v11, v21, v17 // (op 985) 
v_add_f32 v12, v22, v18 // (op 985) 
v_add_f32 v13, v23, v19 // (op 985) 


v_lshrrev_b32 v14, 16, v10 // (op 986) 
v_lshrrev_b32 v15, 16, v11 // (op 986) 
v_lshrrev_b32 v16, 16, v12 // (op 986) 
v_lshrrev_b32 v17, 16, v13 // (op 986) 


v_pack_b32_f16 v10, v14, v15 // (op 987) 
v_pack_b32_f16 v11, v16, v17 // (op 987) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen offset:128 // (op 987) Store value





buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:64 // (op 968) Load value
v_accvgpr_read v12, a4 // (op 964) 
v_accvgpr_read v13, a5 // (op 964) 
v_accvgpr_read v14, a6 // (op 964) 
v_accvgpr_read v15, a7 // (op 964) 
v_mul_f32 v16, v8, v12 // (op 964) 
v_mul_f32 v17, v8, v13 // (op 964) 
v_mul_f32 v18, v8, v14 // (op 964) 
v_mul_f32 v19, v8, v15 // (op 964) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 969) 
v_lshrrev_b32 v13, 16, v10 // (op 969) 
v_lshlrev_b32 v12, 16, v12 // (op 969) 
v_lshlrev_b32 v13, 16, v13 // (op 969) 
v_mul_f32 v20, v5, v12 // (op 969) 
v_mul_f32 v21, v5, v13 // (op 969) 
v_and_b32 v12, 65535, v11 // (op 969) 
v_lshrrev_b32 v13, 16, v11 // (op 969) 
v_lshlrev_b32 v12, 16, v12 // (op 969) 
v_lshlrev_b32 v13, 16, v13 // (op 969) 
v_mul_f32 v22, v5, v12 // (op 969) 
v_mul_f32 v23, v5, v13 // (op 969) 


v_add_f32 v10, v20, v16 // (op 965) 
v_add_f32 v11, v21, v17 // (op 965) 
v_add_f32 v12, v22, v18 // (op 965) 
v_add_f32 v13, v23, v19 // (op 965) 


v_lshrrev_b32 v14, 16, v10 // (op 966) 
v_lshrrev_b32 v15, 16, v11 // (op 966) 
v_lshrrev_b32 v16, 16, v12 // (op 966) 
v_lshrrev_b32 v17, 16, v13 // (op 966) 


v_pack_b32_f16 v10, v14, v15 // (op 967) 
v_pack_b32_f16 v11, v16, v17 // (op 967) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen offset:64 // (op 967) Store value





buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen // (op 138) Load value
v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v8, v12 // (op 134) 
v_mul_f32 v17, v8, v13 // (op 134) 
v_mul_f32 v18, v8, v14 // (op 134) 
v_mul_f32 v19, v8, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 139) 
v_lshrrev_b32 v13, 16, v10 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v20, v5, v12 // (op 139) 
v_mul_f32 v21, v5, v13 // (op 139) 
v_and_b32 v12, 65535, v11 // (op 139) 
v_lshrrev_b32 v13, 16, v11 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v22, v5, v12 // (op 139) 
v_mul_f32 v23, v5, v13 // (op 139) 


v_add_f32 v10, v20, v16 // (op 135) 
v_add_f32 v11, v21, v17 // (op 135) 
v_add_f32 v12, v22, v18 // (op 135) 
v_add_f32 v13, v23, v19 // (op 135) 


v_lshrrev_b32 v14, 16, v10 // (op 136) 
v_lshrrev_b32 v15, 16, v11 // (op 136) 
v_lshrrev_b32 v16, 16, v12 // (op 136) 
v_lshrrev_b32 v17, 16, v13 // (op 136) 


v_pack_b32_f16 v10, v14, v15 // (op 137) 
v_pack_b32_f16 v11, v16, v17 // (op 137) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:












s_endpgm  // End of RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 41984 // lds bytes
  .amdhsa_accum_offset 80
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
    macrotile: [128, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 41984
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 80
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}}
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

