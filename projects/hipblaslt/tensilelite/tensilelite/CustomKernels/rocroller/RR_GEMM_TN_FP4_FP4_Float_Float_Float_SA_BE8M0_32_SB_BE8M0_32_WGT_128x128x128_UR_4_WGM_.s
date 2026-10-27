// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 10410) 
s_mul_hi_i32 s1, s16, s10 // (op 10549) 
v_lshrrev_b32 v0, 7, v1 // (op 10471) 
v_and_b32 v3, 15, v1 // (op 10471) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 10471) 


v_and_b32 v0, 63, v1 // (op 10471) 
v_lshrrev_b32 v3, 4, v0 // (op 10471) 

v_lshlrev_b32 v0, 5, v3 // (op 10471) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 10471) 


v_lshlrev_b32 v0, 2, v3 // (op 10471) 

v_lshrrev_b32 v3, 3, v0 // (op 10471) 

v_lshrrev_b32 v0, 6, v1 // (op 10528) 
v_and_b32 v4, 1, v0 // (op 10528) 

v_and_b32 v0, 15, v1 // (op 10528) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 10528) 


v_and_b32 v0, 63, v1 // (op 10528) 
v_lshrrev_b32 v4, 4, v0 // (op 10528) 

v_lshlrev_b32 v0, 5, v4 // (op 10528) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 10528) 


v_lshlrev_b32 v0, 2, v4 // (op 10528) 

v_lshrrev_b32 v4, 3, v0 // (op 10528) 

v_lshrrev_b32 v0, 7, v1 // (op 10595) 
v_and_b32 v5, 15, v1 // (op 10595) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 10595) 


v_and_b32 v0, 63, v1 // (op 10595) 
v_lshrrev_b32 v5, 4, v0 // (op 10595) 

v_lshlrev_b32 v0, 5, v5 // (op 10595) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 10595) 


v_lshlrev_b32 v0, 2, v5 // (op 10595) 

v_lshrrev_b32 v5, 3, v0 // (op 10595) 

v_lshrrev_b32 v0, 6, v1 // (op 10639) 
v_and_b32 v6, 1, v0 // (op 10639) 

v_and_b32 v0, 15, v1 // (op 10639) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 10639) 


v_and_b32 v0, 63, v1 // (op 10639) 
v_lshrrev_b32 v6, 4, v0 // (op 10639) 

v_lshlrev_b32 v0, 5, v6 // (op 10639) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 10639) 


v_lshlrev_b32 v0, 2, v6 // (op 10639) 

v_lshrrev_b32 v6, 3, v0 // (op 10639) 

v_lshrrev_b32 v0, 7, v1 // (op 10683) 
v_and_b32 v7, 15, v1 // (op 10683) 
v_lshl_add_u32 v8, v0, 6, v7 // (op 10683) 


v_and_b32 v0, 63, v1 // (op 10683) 
v_lshrrev_b32 v7, 4, v0 // (op 10683) 

v_lshlrev_b32 v0, 5, v7 // (op 10683) 

v_lshl_add_u32 v7, v8, 7, v0 // (op 10683) 


v_lshlrev_b32 v0, 2, v7 // (op 10683) 

v_lshrrev_b32 v7, 3, v0 // (op 10683) 

v_lshrrev_b32 v0, 6, v1 // (op 10727) 
v_and_b32 v8, 1, v0 // (op 10727) 

v_and_b32 v0, 15, v1 // (op 10727) 
v_lshl_add_u32 v9, v8, 6, v0 // (op 10727) 


v_and_b32 v0, 63, v1 // (op 10727) 
v_lshrrev_b32 v8, 4, v0 // (op 10727) 

v_lshlrev_b32 v0, 5, v8 // (op 10727) 

v_lshl_add_u32 v8, v9, 7, v0 // (op 10727) 


v_lshlrev_b32 v0, 2, v8 // (op 10727) 

v_lshrrev_b32 v8, 3, v0 // (op 10727) 

v_lshrrev_b32 v0, 7, v1 // (op 10771) 
v_and_b32 v9, 15, v1 // (op 10771) 
v_lshl_add_u32 v10, v0, 6, v9 // (op 10771) 


v_and_b32 v0, 63, v1 // (op 10771) 
v_lshrrev_b32 v9, 4, v0 // (op 10771) 

v_lshlrev_b32 v0, 5, v9 // (op 10771) 

v_lshl_add_u32 v9, v10, 7, v0 // (op 10771) 


v_lshlrev_b32 v0, 2, v9 // (op 10771) 

v_lshrrev_b32 v9, 3, v0 // (op 10771) 

v_lshrrev_b32 v0, 6, v1 // (op 10815) 
v_and_b32 v10, 1, v0 // (op 10815) 

v_and_b32 v0, 15, v1 // (op 10815) 
v_lshl_add_u32 v11, v10, 6, v0 // (op 10815) 


v_and_b32 v0, 63, v1 // (op 10815) 
v_lshrrev_b32 v10, 4, v0 // (op 10815) 

v_lshlrev_b32 v0, 5, v10 // (op 10815) 

v_lshl_add_u32 v10, v11, 7, v0 // (op 10815) 


v_lshlrev_b32 v0, 2, v10 // (op 10815) 

v_lshrrev_b32 v10, 3, v0 // (op 10815) 

v_lshrrev_b32 v0, 7, v1 // (op 10410) 
s_add_i32 s18, s0, s16 // (op 10410) 

v_and_b32 v11, 63, v1 // (op 10410) 
s_ashr_i32 s0, s18, 31 // (op 10410) 
v_lshrrev_b32 v12, 4, v11 // (op 10410) 
s_cmp_eq_i32 s10, 0 // (op 10410) 
s_mov_b32 s19, scc // (op 10410) 
v_lshlrev_b32 v13, 2, v12 // (op 10410) 

s_cmp_eq_i32 s19, 0 // (op 10410) 
s_cselect_b32 s49, 0, -1 // (op 10410) 

v_lshl_add_u32 v12, v0, 6, v13 // (op 10410) 


s_and_b32 s19, s13, 31 // (op 10410) 
v_lshrrev_b32 v0, 2, v11 // (op 10410) 

v_and_b32 v11, 3, v0 // (op 10410) 

v_lshrrev_b32 v0, 7, v1 // (op 10549) 
v_and_b32 v13, 63, v1 // (op 10549) 
s_lshl_b32 s50, 1, s19 // (op 10410) 
v_and_b32 v14, 3, v1 // (op 10410) 
s_add_i32 s51, s50, s49 // (op 10410) 


s_and_b32 s49, s0, s51 // (op 10410) 


s_add_i32 s0, s1, s16 // (op 10549) 

s_ashr_i32 s1, s0, 31 // (op 10549) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 10410) 


s_add_i32 s50, s18, s49 // (op 10410) 


v_lshrrev_b32 v11, 6, v1 // (op 10410) 
s_ashr_i32 s18, s50, s19 // (op 10410) 


v_and_b32 v14, 1, v11 // (op 10410) 

s_lshl_b32 s19, s13, 24 // (op 10410) 
v_lshl_add_u32 v11, v14, 6, v15 // (op 10410) 


s_ashr_i32 s49, s19, 31 // (op 10410) 

s_xor_b32 s19, s18, s49 // (op 10410) 

s_sub_i32 s18, s19, s49 // (op 10410) 


s_mul_hi_i32 s19, s18, s14 // (op 10410) 
s_add_i32 s49, s19, s18 // (op 10410) 

s_ashr_i32 s19, s49, 31 // (op 10410) 
s_cmp_eq_i32 s14, 0 // (op 10410) 
s_mov_b32 s50, scc // (op 10410) 
s_cmp_eq_i32 s50, 0 // (op 10410) 
s_cselect_b32 s51, 0, -1 // (op 10410) 

s_and_b32 s50, s15, 31 // (op 10410) 
v_lshrrev_b32 v14, 4, v13 // (op 10549) 
s_cmp_eq_i32 s10, 0 // (op 10549) 
s_mov_b32 s52, scc // (op 10549) 
v_lshlrev_b32 v15, 2, v14 // (op 10549) 

s_cmp_eq_i32 s52, 0 // (op 10549) 
s_cselect_b32 s53, 0, -1 // (op 10549) 

s_lshl_b32 s52, 1, s50 // (op 10410) 
v_lshl_add_u32 v14, v0, 6, v15 // (op 10549) 


v_lshrrev_b32 v0, 2, v13 // (op 10549) 

s_add_i32 s54, s52, s51 // (op 10410) 


s_and_b32 s51, s19, s54 // (op 10410) 


s_add_i32 s19, s49, s51 // (op 10410) 


s_ashr_i32 s49, s19, s50 // (op 10410) 


s_lshl_b32 s19, s15, 24 // (op 10410) 
s_ashr_i32 s50, s19, 31 // (op 10410) 

s_xor_b32 s19, s49, s50 // (op 10410) 

s_sub_i32 s49, s19, s50 // (op 10410) 


s_mul_i32 s19, s18, s11 // (op 10410) 
s_sub_u32 s50, s16, s19 // (op 10410) 

s_mul_i32 s19, s49, s11 // (op 10410) 
s_add_u32 s19, s19, s50 // (op 10410) 

s_mul_hi_i32 s50, s16, s9 // (op 10410) 
s_add_i32 s51, s50, s16 // (op 10410) 

s_ashr_i32 s50, s51, 31 // (op 10410) 
s_cmp_eq_i32 s9, 0 // (op 10410) 
s_mov_b32 s52, scc // (op 10410) 
s_cmp_eq_i32 s52, 0 // (op 10410) 
s_cselect_b32 s54, 0, -1 // (op 10410) 

s_and_b32 s52, s13, 31 // (op 10549) 
v_and_b32 v13, 3, v0 // (op 10549) 

s_lshl_b32 s55, 1, s52 // (op 10549) 
v_and_b32 v0, 3, v1 // (op 10549) 
s_add_i32 s56, s55, s53 // (op 10549) 


s_and_b32 s53, s1, s56 // (op 10549) 


v_lshl_add_u32 v15, v13, 2, v0 // (op 10549) 


s_add_i32 s1, s0, s53 // (op 10549) 


v_lshrrev_b32 v0, 6, v1 // (op 10549) 
s_ashr_i32 s0, s1, s52 // (op 10549) 


v_and_b32 v13, 1, v0 // (op 10549) 

s_lshl_b32 s1, s13, 24 // (op 10549) 
v_lshl_add_u32 v0, v13, 6, v15 // (op 10549) 


s_ashr_i32 s52, s1, 31 // (op 10549) 

s_xor_b32 s1, s0, s52 // (op 10549) 

s_sub_i32 s0, s1, s52 // (op 10549) 


s_mul_hi_i32 s1, s0, s14 // (op 10549) 
s_add_i32 s52, s1, s0 // (op 10549) 

s_ashr_i32 s1, s52, 31 // (op 10549) 
s_cmp_eq_i32 s14, 0 // (op 10549) 
s_mov_b32 s53, scc // (op 10549) 
s_cmp_eq_i32 s53, 0 // (op 10549) 
s_cselect_b32 s55, 0, -1 // (op 10549) 

s_and_b32 s53, s15, 31 // (op 10549) 
s_lshl_b32 s56, 1, s53 // (op 10549) 
s_add_i32 s57, s56, s55 // (op 10549) 


s_and_b32 s55, s1, s57 // (op 10549) 


s_add_i32 s1, s52, s55 // (op 10549) 


s_ashr_i32 s52, s1, s53 // (op 10549) 


s_lshl_b32 s1, s15, 24 // (op 10549) 
s_ashr_i32 s53, s1, 31 // (op 10549) 

s_xor_b32 s1, s52, s53 // (op 10549) 

s_sub_i32 s52, s1, s53 // (op 10549) 


s_mul_i32 s1, s0, s11 // (op 10549) 
s_sub_u32 s53, s16, s1 // (op 10549) 

s_mul_i32 s1, s52, s11 // (op 10549) 
s_add_u32 s1, s1, s53 // (op 10549) 

s_mul_hi_i32 s53, s16, s9 // (op 10549) 
s_add_i32 s55, s53, s16 // (op 10549) 

s_ashr_i32 s53, s55, 31 // (op 10549) 
s_cmp_eq_i32 s9, 0 // (op 10549) 
s_mov_b32 s56, scc // (op 10549) 
s_cmp_eq_i32 s56, 0 // (op 10549) 
s_cselect_b32 s57, 0, -1 // (op 10549) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 10410) 
s_lshl_b32 s58, 1, s56 // (op 10410) 
s_add_i32 s59, s58, s54 // (op 10410) 


s_and_b32 s54, s50, s59 // (op 10410) 


s_add_i32 s50, s51, s54 // (op 10410) 


s_ashr_i32 s51, s50, s56 // (op 10410) 


s_lshl_b32 s50, s45, 24 // (op 10410) 
s_ashr_i32 s54, s50, 31 // (op 10410) 

s_xor_b32 s50, s51, s54 // (op 10410) 

s_sub_i32 s51, s50, s54 // (op 10410) 


s_and_b32 s50, s45, 31 // (op 10549) 
s_lshl_b32 s54, 1, s50 // (op 10549) 
s_add_i32 s56, s54, s57 // (op 10549) 


s_and_b32 s54, s53, s56 // (op 10549) 


s_add_i32 s53, s55, s54 // (op 10549) 


s_ashr_i32 s54, s53, s50 // (op 10549) 


s_lshl_b32 s50, s45, 24 // (op 10549) 
s_ashr_i32 s53, s50, 31 // (op 10549) 

s_xor_b32 s50, s54, s53 // (op 10549) 

s_sub_i32 s54, s50, s53 // (op 10549) 


s_add_u32 s56, s6, 128 // (op 10410) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 10410) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 10410) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 10410) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 10410) 

s_mul_i32 s50, s11, s56 // (op 10410) 
s_mul_i32 s53, s8, s50 // (op 10410) 

s_mul_i32 s50, s51, s53 // (op 10410) 

s_sub_u32 s53, s16, s50 // (op 10410) 

s_mov_b32 s50, s53 // (op 10410) 
s_mul_hi_i32 s55, s50, s12 // (op 10410) 
s_add_i32 s58, s55, s50 // (op 10410) 


s_ashr_i32 s50, s58, 31 // (op 10410) 
s_cmp_eq_i32 s12, 0 // (op 10410) 
s_mov_b32 s55, scc // (op 10410) 
s_cmp_eq_i32 s55, 0 // (op 10410) 
s_cselect_b32 s59, 0, -1 // (op 10410) 

s_and_b32 s55, s44, 31 // (op 10410) 
s_lshl_b32 s60, 1, s55 // (op 10410) 
s_add_i32 s61, s60, s59 // (op 10410) 


s_and_b32 s59, s50, s61 // (op 10410) 


s_add_i32 s50, s58, s59 // (op 10410) 


s_ashr_i32 s58, s50, s55 // (op 10410) 


s_lshl_b32 s50, s44, 24 // (op 10410) 
s_ashr_i32 s55, s50, 31 // (op 10410) 

s_xor_b32 s50, s58, s55 // (op 10410) 

s_sub_i32 s58, s50, s55 // (op 10410) 


s_mul_i32 s50, s46, s11 // (op 10410) 
s_add_u32 s60, s20, 128 // (op 10410) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 10410) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 10410) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 10410) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 7 // (op 10410) 

s_sub_i32 s55, s60, s50 // (op 10410) 

s_mul_i32 s59, s58, s55 // (op 10410) 

s_sub_u32 s55, s53, s59 // (op 10410) 


s_add_u32 s53, s55, s50 // (op 10410) 


s_cmp_eq_i32 s51, 0 // (op 10410) 
s_mov_b32 s50, scc // (op 10410) 

s_cmp_eq_i32 s50, 0 // (op 10410) 
s_cselect_b32 s51, s53, s19 // (op 10410) 


s_mul_i32 s19, s49, s56 // (op 10410) 


v_lshl_add_u32 v13, s51, 7, v12 // (op 10410) 


s_sub_i32 s49, s18, s19 // (op 10410) 


s_cmp_eq_i32 s50, 0 // (op 10410) 
s_cselect_b32 s51, s58, s49 // (op 10410) 



v_lshl_add_u32 v12, s51, 7, v11 // (op 10410) 


v_mul_lo_u32 v11, v12, s2 // (op 10410) 

v_add_lshl_u32 v12, v13, v11, 2 // (op 10410) 


s_add_u32 s18, s6, 128 // (op 10549) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 10549) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 10549) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 10549) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 7 // (op 10549) 

s_mul_i32 s53, s11, s18 // (op 10549) 
s_mul_i32 s49, s8, s53 // (op 10549) 

s_mul_i32 s53, s54, s49 // (op 10549) 

s_sub_u32 s49, s16, s53 // (op 10549) 

s_mov_b32 s53, s49 // (op 10549) 
s_mul_hi_i32 s50, s53, s12 // (op 10549) 
s_add_i32 s51, s50, s53 // (op 10549) 


s_ashr_i32 s50, s51, 31 // (op 10549) 
s_cmp_eq_i32 s12, 0 // (op 10549) 
s_mov_b32 s53, scc // (op 10549) 
s_cmp_eq_i32 s53, 0 // (op 10549) 
s_cselect_b32 s55, 0, -1 // (op 10549) 

s_and_b32 s53, s44, 31 // (op 10549) 
s_lshl_b32 s56, 1, s53 // (op 10549) 
s_add_i32 s57, s56, s55 // (op 10549) 


s_and_b32 s55, s50, s57 // (op 10549) 


s_add_i32 s50, s51, s55 // (op 10549) 


s_ashr_i32 s51, s50, s53 // (op 10549) 


s_lshl_b32 s50, s44, 24 // (op 10549) 
s_ashr_i32 s53, s50, 31 // (op 10549) 

s_xor_b32 s50, s51, s53 // (op 10549) 

s_sub_i32 s51, s50, s53 // (op 10549) 


s_mul_i32 s50, s46, s11 // (op 10549) 
s_add_u32 s56, s20, 128 // (op 10549) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 10549) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 10549) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 10549) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 10549) 

s_sub_i32 s53, s56, s50 // (op 10549) 

s_mul_i32 s55, s51, s53 // (op 10549) 

s_sub_u32 s53, s49, s55 // (op 10549) 


s_add_u32 s49, s53, s50 // (op 10549) 


s_cmp_eq_i32 s54, 0 // (op 10549) 
s_mov_b32 s50, scc // (op 10549) 

s_cmp_eq_i32 s50, 0 // (op 10549) 
s_cselect_b32 s53, s49, s1 // (op 10549) 


s_mul_i32 s1, s52, s18 // (op 10549) 


v_lshl_add_u32 v11, s53, 7, v14 // (op 10549) 


s_sub_i32 s49, s0, s1 // (op 10549) 


s_cmp_eq_i32 s50, 0 // (op 10549) 
s_cselect_b32 s0, s51, s49 // (op 10549) 



v_lshl_add_u32 v13, s0, 7, v0 // (op 10549) 


v_mul_lo_u32 v0, v13, s4 // (op 10549) 

v_add_lshl_u32 v13, v11, v0, 2 // (op 10549) 


s_mul_i32 s1, s2, s7 // (op 10412) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 10412) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 10412) least significant
s_mul_hi_u32 s19, s2, s6 // (op 10412) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10412) most significant: sum
s_add_u32 s1, s1, s19 // (op 10412) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10412) 

s_mul_i32 s1, s4, s7 // (op 10551) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 10551) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 10551) least significant
s_mul_hi_u32 s50, s4, s6 // (op 10551) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 10551) most significant: sum
s_add_u32 s1, s1, s50 // (op 10551) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 2 // (op 10551) 

s_mov_b32 s52, s22 // (op 10412) reg expression
s_mov_b32 s53, s23 // (op 10412) reg expression
s_mov_b32 s54, s18 // (op 10412) reg expression
s_mov_b32 s55, 131072 // (op 10412) call()

s_mov_b32 s56, s24 // (op 10551) reg expression

s_mov_b32 s57, s25 // (op 10551) reg expression
s_mov_b32 s58, s50 // (op 10551) reg expression
s_mov_b32 s59, 131072 // (op 10551) call()


v_mov_b32 v0, s47 // (op 2) Move value
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
s_mul_hi_i32 s0, s16, s10 // (op 10300) 
v_lshrrev_b32 v14, 2, v1 // (op 10300) 
s_add_i32 s1, s0, s16 // (op 10300) 

v_and_b32 v15, -4, v1 // (op 10300) 
s_ashr_i32 s0, s1, 31 // (op 10300) 
v_sub_u32 v16, v1, v15 // (op 10300) 

s_cmp_eq_i32 s10, 0 // (op 10300) 
s_mov_b32 s18, scc // (op 10300) 
v_lshlrev_b32 v15, 5, v16 // (op 10300) 

s_cmp_eq_i32 s18, 0 // (op 10300) 
s_cselect_b32 s19, 0, -1 // (op 10300) 

v_mov_b32 v17, 0 // (op 10300) convert
v_mov_b32 v16, v15 // (op 10300) convert

s_and_b32 s18, s13, 31 // (op 10300) 
s_lshl_b32 s22, 1, s18 // (op 10300) 
s_add_i32 s23, s22, s19 // (op 10300) 


s_and_b32 s19, s0, s23 // (op 10300) 


s_add_i32 s0, s1, s19 // (op 10300) 


s_ashr_i32 s1, s0, s18 // (op 10300) 


s_lshl_b32 s0, s13, 24 // (op 10300) 
s_ashr_i32 s18, s0, 31 // (op 10300) 

s_xor_b32 s0, s1, s18 // (op 10300) 

s_sub_i32 s1, s0, s18 // (op 10300) 


s_mul_hi_i32 s0, s1, s14 // (op 10300) 
s_add_i32 s18, s0, s1 // (op 10300) 

s_ashr_i32 s0, s18, 31 // (op 10300) 
s_cmp_eq_i32 s14, 0 // (op 10300) 
s_mov_b32 s19, scc // (op 10300) 
s_cmp_eq_i32 s19, 0 // (op 10300) 
s_cselect_b32 s22, 0, -1 // (op 10300) 

s_and_b32 s19, s15, 31 // (op 10300) 
s_lshl_b32 s23, 1, s19 // (op 10300) 
s_add_i32 s24, s23, s22 // (op 10300) 


s_and_b32 s25, s0, s24 // (op 10300) 


s_add_i32 s0, s18, s25 // (op 10300) 


s_ashr_i32 s18, s0, s19 // (op 10300) 


s_lshl_b32 s0, s15, 24 // (op 10300) 
s_ashr_i32 s19, s0, 31 // (op 10300) 

s_xor_b32 s0, s18, s19 // (op 10300) 

s_sub_i32 s18, s0, s19 // (op 10300) 


s_mul_i32 s0, s1, s11 // (op 10300) 

s_sub_u32 s1, s16, s0 // (op 10300) 

s_mul_i32 s0, s18, s11 // (op 10300) 
s_add_u32 s0, s0, s1 // (op 10300) 


s_mul_hi_i32 s1, s16, s9 // (op 10300) 
s_add_i32 s18, s1, s16 // (op 10300) 

s_ashr_i32 s1, s18, 31 // (op 10300) 
s_cmp_eq_i32 s9, 0 // (op 10300) 
s_mov_b32 s19, scc // (op 10300) 
s_cmp_eq_i32 s19, 0 // (op 10300) 
s_cselect_b32 s22, 0, -1 // (op 10300) 

s_and_b32 s19, s45, 31 // (op 10300) 
s_lshl_b32 s23, 1, s19 // (op 10300) 
s_add_i32 s24, s23, s22 // (op 10300) 


s_and_b32 s25, s1, s24 // (op 10300) 


s_add_i32 s1, s18, s25 // (op 10300) 


s_ashr_i32 s18, s1, s19 // (op 10300) 


s_lshl_b32 s1, s45, 24 // (op 10300) 
s_ashr_i32 s19, s1, 31 // (op 10300) 

s_xor_b32 s1, s18, s19 // (op 10300) 

s_sub_i32 s18, s1, s19 // (op 10300) 


s_add_u32 s22, s6, 128 // (op 10300) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10300) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10300) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10300) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10300) 

s_mul_i32 s1, s11, s22 // (op 10300) 

s_mul_i32 s19, s8, s1 // (op 10300) 

s_mul_i32 s1, s18, s19 // (op 10300) 

s_sub_u32 s19, s16, s1 // (op 10300) 

s_mov_b32 s1, s19 // (op 10300) 
s_mul_hi_i32 s22, s1, s12 // (op 10300) 
s_add_i32 s23, s22, s1 // (op 10300) 


s_ashr_i32 s1, s23, 31 // (op 10300) 
s_cmp_eq_i32 s12, 0 // (op 10300) 
s_mov_b32 s22, scc // (op 10300) 
s_cmp_eq_i32 s22, 0 // (op 10300) 
s_cselect_b32 s24, 0, -1 // (op 10300) 

s_and_b32 s22, s44, 31 // (op 10300) 
s_lshl_b32 s25, 1, s22 // (op 10300) 
s_add_i32 s47, s25, s24 // (op 10300) 


s_and_b32 s24, s1, s47 // (op 10300) 


s_add_i32 s1, s23, s24 // (op 10300) 


s_ashr_i32 s23, s1, s22 // (op 10300) 


s_lshl_b32 s1, s44, 24 // (op 10300) 
s_ashr_i32 s22, s1, 31 // (op 10300) 

s_xor_b32 s1, s23, s22 // (op 10300) 

s_sub_i32 s23, s1, s22 // (op 10300) 


s_mul_i32 s1, s46, s11 // (op 10300) 
s_add_u32 s24, s20, 128 // (op 10300) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 10300) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 10300) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 10300) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 10300) 

s_sub_i32 s22, s24, s1 // (op 10300) 

s_mul_i32 s24, s23, s22 // (op 10300) 


s_sub_u32 s25, s19, s24 // (op 10300) 


s_add_u32 s19, s25, s1 // (op 10300) 


s_cmp_eq_i32 s18, 0 // (op 10300) 
s_mov_b32 s1, scc // (op 10300) 

s_cmp_eq_i32 s1, 0 // (op 10300) 
s_cselect_b32 s18, s19, s0 // (op 10300) 



v_lshl_add_u32 v15, s18, 7, v14 // (op 10300) 


v_mul_lo_u32 v19, v15, s27 // (op 10300) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 10300) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 10300) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 10300) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 10300) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 10300) most significant half



v_lshlrev_b64 v[14:15], 2, v[18:19] // (op 10300) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 10300) 

s_mul_i32 s1, s26, s21 // (op 10302) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 10302) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 10302) least significant
s_mul_hi_u32 s19, s26, s20 // (op 10302) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10302) most significant: sum
s_add_u32 s1, s1, s19 // (op 10302) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10302) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10302) 

s_mov_b32 s48, s28 // (op 10302) reg expression
s_mov_b32 s49, s29 // (op 10302) reg expression
s_mov_b32 s50, s0 // (op 10302) reg expression
s_mov_b32 s51, 131072 // (op 10302) call()

s_mul_hi_i32 s47, s16, s10 // (op 10319) 
v_lshrrev_b32 v14, 2, v1 // (op 10319) 
s_add_i32 s0, s47, s16 // (op 10319) 

v_and_b32 v15, -4, v1 // (op 10319) 
s_ashr_i32 s1, s0, 31 // (op 10319) 
v_sub_u32 v18, v1, v15 // (op 10319) 

s_cmp_eq_i32 s10, 0 // (op 10319) 
s_mov_b32 s47, scc // (op 10319) 
v_lshlrev_b32 v15, 5, v18 // (op 10319) 

s_cmp_eq_i32 s47, 0 // (op 10319) 
s_cselect_b32 s18, 0, -1 // (op 10319) 

v_mov_b32 v19, 0 // (op 10319) convert
v_mov_b32 v18, v15 // (op 10319) convert

s_and_b32 s19, s13, 31 // (op 10319) 
s_lshl_b32 s47, 1, s19 // (op 10319) 
s_add_i32 s22, s47, s18 // (op 10319) 


s_and_b32 s18, s1, s22 // (op 10319) 


s_add_i32 s1, s0, s18 // (op 10319) 


s_ashr_i32 s0, s1, s19 // (op 10319) 


s_lshl_b32 s1, s13, 24 // (op 10319) 
s_ashr_i32 s47, s1, 31 // (op 10319) 

s_xor_b32 s1, s0, s47 // (op 10319) 

s_sub_i32 s0, s1, s47 // (op 10319) 


s_mul_hi_i32 s1, s0, s14 // (op 10319) 
s_add_i32 s47, s1, s0 // (op 10319) 

s_ashr_i32 s1, s47, 31 // (op 10319) 
s_cmp_eq_i32 s14, 0 // (op 10319) 
s_mov_b32 s18, scc // (op 10319) 
s_cmp_eq_i32 s18, 0 // (op 10319) 
s_cselect_b32 s19, 0, -1 // (op 10319) 

s_and_b32 s18, s15, 31 // (op 10319) 
s_lshl_b32 s22, 1, s18 // (op 10319) 
s_add_i32 s23, s22, s19 // (op 10319) 


s_and_b32 s19, s1, s23 // (op 10319) 


s_add_i32 s1, s47, s19 // (op 10319) 


s_ashr_i32 s19, s1, s18 // (op 10319) 


s_lshl_b32 s1, s15, 24 // (op 10319) 
s_ashr_i32 s18, s1, 31 // (op 10319) 

s_xor_b32 s1, s19, s18 // (op 10319) 

s_sub_i32 s19, s1, s18 // (op 10319) 


s_mul_i32 s1, s0, s11 // (op 10319) 

s_sub_u32 s0, s16, s1 // (op 10319) 

s_mul_i32 s1, s19, s11 // (op 10319) 
s_add_u32 s1, s1, s0 // (op 10319) 


s_mul_hi_i32 s0, s16, s9 // (op 10319) 
s_add_i32 s47, s0, s16 // (op 10319) 

s_ashr_i32 s0, s47, 31 // (op 10319) 
s_cmp_eq_i32 s9, 0 // (op 10319) 
s_mov_b32 s18, scc // (op 10319) 
s_cmp_eq_i32 s18, 0 // (op 10319) 
s_cselect_b32 s19, 0, -1 // (op 10319) 

s_and_b32 s18, s45, 31 // (op 10319) 
s_lshl_b32 s22, 1, s18 // (op 10319) 
s_add_i32 s23, s22, s19 // (op 10319) 


s_and_b32 s19, s0, s23 // (op 10319) 


s_add_i32 s0, s47, s19 // (op 10319) 


s_ashr_i32 s19, s0, s18 // (op 10319) 


s_lshl_b32 s0, s45, 24 // (op 10319) 
s_ashr_i32 s18, s0, 31 // (op 10319) 

s_xor_b32 s0, s19, s18 // (op 10319) 

s_sub_i32 s19, s0, s18 // (op 10319) 


s_add_u32 s22, s6, 128 // (op 10319) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10319) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10319) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10319) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10319) 

s_mul_i32 s0, s11, s22 // (op 10319) 

s_mul_i32 s18, s8, s0 // (op 10319) 

s_mul_i32 s0, s19, s18 // (op 10319) 

s_sub_u32 s18, s16, s0 // (op 10319) 

s_mov_b32 s0, s18 // (op 10319) 
s_mul_hi_i32 s47, s0, s12 // (op 10319) 
s_add_i32 s22, s47, s0 // (op 10319) 


s_ashr_i32 s0, s22, 31 // (op 10319) 
s_cmp_eq_i32 s12, 0 // (op 10319) 
s_mov_b32 s47, scc // (op 10319) 
s_cmp_eq_i32 s47, 0 // (op 10319) 
s_cselect_b32 s23, 0, -1 // (op 10319) 

s_and_b32 s47, s44, 31 // (op 10319) 
s_lshl_b32 s24, 1, s47 // (op 10319) 
s_add_i32 s25, s24, s23 // (op 10319) 


s_and_b32 s23, s0, s25 // (op 10319) 


s_add_i32 s0, s22, s23 // (op 10319) 


s_ashr_i32 s22, s0, s47 // (op 10319) 


s_lshl_b32 s0, s44, 24 // (op 10319) 
s_ashr_i32 s47, s0, 31 // (op 10319) 

s_xor_b32 s0, s22, s47 // (op 10319) 

s_sub_i32 s22, s0, s47 // (op 10319) 


s_mul_i32 s0, s46, s11 // (op 10319) 
s_add_u32 s24, s20, 128 // (op 10319) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 10319) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 10319) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 10319) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 7 // (op 10319) 

s_sub_i32 s23, s24, s0 // (op 10319) 

s_mul_i32 s47, s22, s23 // (op 10319) 


s_sub_u32 s22, s18, s47 // (op 10319) 


s_add_u32 s18, s22, s0 // (op 10319) 


s_cmp_eq_i32 s19, 0 // (op 10319) 
s_mov_b32 s0, scc // (op 10319) 

s_cmp_eq_i32 s0, 0 // (op 10319) 
s_cselect_b32 s19, s18, s1 // (op 10319) 



v_lshl_add_u32 v15, s19, 7, v14 // (op 10319) 


v_mul_lo_u32 v21, v15, s27 // (op 10319) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 10319) most significant: high of low * low
v_mul_lo_u32 v20, v15, s26 // (op 10319) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 10319) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 10319) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 10319) most significant half



v_lshlrev_b64 v[14:15], 2, v[20:21] // (op 10319) 

v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 10319) 

s_mul_i32 s1, s26, s21 // (op 10321) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 10321) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 10321) least significant
s_mul_hi_u32 s18, s26, s20 // (op 10321) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10321) most significant: sum
s_add_u32 s1, s1, s18 // (op 10321) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10321) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10321) 

s_mov_b32 s48, s28 // (op 10321) reg expression
s_mov_b32 s49, s29 // (op 10321) reg expression
s_mov_b32 s50, s0 // (op 10321) reg expression
s_mov_b32 s51, 131072 // (op 10321) call()

s_mul_hi_i32 s47, s16, s10 // (op 10364) 
v_lshrrev_b32 v14, 2, v1 // (op 10364) 
s_add_i32 s0, s47, s16 // (op 10364) 

v_and_b32 v15, -4, v1 // (op 10364) 
s_ashr_i32 s1, s0, 31 // (op 10364) 
v_sub_u32 v20, v1, v15 // (op 10364) 

s_cmp_eq_i32 s10, 0 // (op 10364) 
s_mov_b32 s47, scc // (op 10364) 
v_lshlrev_b32 v15, 5, v20 // (op 10364) 

s_cmp_eq_i32 s47, 0 // (op 10364) 
s_cselect_b32 s18, 0, -1 // (op 10364) 

v_mov_b32 v21, 0 // (op 10364) convert
v_mov_b32 v20, v15 // (op 10364) convert

s_and_b32 s19, s13, 31 // (op 10364) 
s_lshl_b32 s47, 1, s19 // (op 10364) 
s_add_i32 s22, s47, s18 // (op 10364) 


s_and_b32 s18, s1, s22 // (op 10364) 


s_add_i32 s1, s0, s18 // (op 10364) 


s_ashr_i32 s0, s1, s19 // (op 10364) 


s_lshl_b32 s1, s13, 24 // (op 10364) 
s_ashr_i32 s47, s1, 31 // (op 10364) 

s_xor_b32 s1, s0, s47 // (op 10364) 

s_sub_i32 s0, s1, s47 // (op 10364) 


s_mul_hi_i32 s1, s0, s14 // (op 10364) 
s_add_i32 s47, s1, s0 // (op 10364) 

s_ashr_i32 s1, s47, 31 // (op 10364) 
s_cmp_eq_i32 s14, 0 // (op 10364) 
s_mov_b32 s18, scc // (op 10364) 
s_cmp_eq_i32 s18, 0 // (op 10364) 
s_cselect_b32 s19, 0, -1 // (op 10364) 

s_and_b32 s18, s15, 31 // (op 10364) 
s_lshl_b32 s22, 1, s18 // (op 10364) 
s_add_i32 s23, s22, s19 // (op 10364) 


s_and_b32 s19, s1, s23 // (op 10364) 


s_add_i32 s1, s47, s19 // (op 10364) 


s_ashr_i32 s19, s1, s18 // (op 10364) 


s_lshl_b32 s1, s15, 24 // (op 10364) 
s_ashr_i32 s18, s1, 31 // (op 10364) 

s_xor_b32 s1, s19, s18 // (op 10364) 

s_sub_i32 s19, s1, s18 // (op 10364) 


s_add_u32 s22, s6, 128 // (op 10364) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10364) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10364) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10364) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10364) 

s_mul_i32 s1, s19, s22 // (op 10364) 

s_sub_i32 s47, s0, s1 // (op 10364) 


s_mul_hi_i32 s0, s16, s9 // (op 10364) 
s_add_i32 s1, s0, s16 // (op 10364) 

s_ashr_i32 s0, s1, 31 // (op 10364) 
s_cmp_eq_i32 s9, 0 // (op 10364) 
s_mov_b32 s18, scc // (op 10364) 
s_cmp_eq_i32 s18, 0 // (op 10364) 
s_cselect_b32 s19, 0, -1 // (op 10364) 

s_and_b32 s18, s45, 31 // (op 10364) 
s_lshl_b32 s24, 1, s18 // (op 10364) 
s_add_i32 s25, s24, s19 // (op 10364) 


s_and_b32 s19, s0, s25 // (op 10364) 


s_add_i32 s0, s1, s19 // (op 10364) 


s_ashr_i32 s1, s0, s18 // (op 10364) 


s_lshl_b32 s0, s45, 24 // (op 10364) 
s_ashr_i32 s18, s0, 31 // (op 10364) 

s_xor_b32 s0, s1, s18 // (op 10364) 

s_sub_i32 s1, s0, s18 // (op 10364) 


s_mul_i32 s0, s11, s22 // (op 10364) 

s_mul_i32 s18, s8, s0 // (op 10364) 

s_mul_i32 s0, s1, s18 // (op 10364) 

s_sub_u32 s18, s16, s0 // (op 10364) 

s_mul_hi_i32 s0, s18, s12 // (op 10364) 
s_add_i32 s19, s0, s18 // (op 10364) 


s_ashr_i32 s0, s19, 31 // (op 10364) 
s_cmp_eq_i32 s12, 0 // (op 10364) 
s_mov_b32 s18, scc // (op 10364) 
s_cmp_eq_i32 s18, 0 // (op 10364) 
s_cselect_b32 s22, 0, -1 // (op 10364) 

s_and_b32 s18, s44, 31 // (op 10364) 
s_lshl_b32 s23, 1, s18 // (op 10364) 
s_add_i32 s24, s23, s22 // (op 10364) 


s_and_b32 s25, s0, s24 // (op 10364) 


s_add_i32 s0, s19, s25 // (op 10364) 


s_ashr_i32 s19, s0, s18 // (op 10364) 


s_lshl_b32 s0, s44, 24 // (op 10364) 
s_ashr_i32 s18, s0, 31 // (op 10364) 

s_xor_b32 s0, s19, s18 // (op 10364) 

s_sub_i32 s19, s0, s18 // (op 10364) 


s_cmp_eq_i32 s1, 0 // (op 10364) 
s_mov_b32 s0, scc // (op 10364) 

s_cmp_eq_i32 s0, 0 // (op 10364) 
s_cselect_b32 s1, s19, s47 // (op 10364) 



v_lshl_add_u32 v15, s1, 7, v14 // (op 10364) 


v_mul_lo_u32 v23, v15, s31 // (op 10364) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 10364) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 10364) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 10364) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 10364) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 10364) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 10364) 


v_lshrrev_b64 v[20:21], 3, v[14:15] // (op 10364) 

s_mul_i32 s1, s30, s7 // (op 10366) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 10366) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 10366) least significant
s_mul_hi_u32 s18, s30, s6 // (op 10366) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10366) most significant: sum
s_add_u32 s1, s1, s18 // (op 10366) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10366) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10366) 

s_mov_b32 s60, s32 // (op 10366) reg expression
s_mov_b32 s61, s33 // (op 10366) reg expression
s_mov_b32 s62, s0 // (op 10366) reg expression
s_mov_b32 s63, 131072 // (op 10366) call()

s_mul_hi_i32 s47, s16, s10 // (op 10383) 
v_lshrrev_b32 v14, 2, v1 // (op 10383) 
s_add_i32 s0, s47, s16 // (op 10383) 

v_and_b32 v15, -4, v1 // (op 10383) 
s_ashr_i32 s1, s0, 31 // (op 10383) 
v_sub_u32 v22, v1, v15 // (op 10383) 

s_cmp_eq_i32 s10, 0 // (op 10383) 
s_mov_b32 s47, scc // (op 10383) 
v_lshlrev_b32 v15, 5, v22 // (op 10383) 

s_cmp_eq_i32 s47, 0 // (op 10383) 
s_cselect_b32 s18, 0, -1 // (op 10383) 

v_mov_b32 v23, 0 // (op 10383) convert
v_mov_b32 v22, v15 // (op 10383) convert

s_and_b32 s19, s13, 31 // (op 10383) 
s_lshl_b32 s47, 1, s19 // (op 10383) 
s_add_i32 s22, s47, s18 // (op 10383) 


s_and_b32 s18, s1, s22 // (op 10383) 


s_add_i32 s1, s0, s18 // (op 10383) 


s_ashr_i32 s0, s1, s19 // (op 10383) 


s_lshl_b32 s1, s13, 24 // (op 10383) 
s_ashr_i32 s47, s1, 31 // (op 10383) 

s_xor_b32 s1, s0, s47 // (op 10383) 

s_sub_i32 s0, s1, s47 // (op 10383) 


s_mul_hi_i32 s1, s0, s14 // (op 10383) 
s_add_i32 s47, s1, s0 // (op 10383) 

s_ashr_i32 s1, s47, 31 // (op 10383) 
s_cmp_eq_i32 s14, 0 // (op 10383) 
s_mov_b32 s18, scc // (op 10383) 
s_cmp_eq_i32 s18, 0 // (op 10383) 
s_cselect_b32 s19, 0, -1 // (op 10383) 

s_and_b32 s18, s15, 31 // (op 10383) 
s_lshl_b32 s22, 1, s18 // (op 10383) 
s_add_i32 s23, s22, s19 // (op 10383) 


s_and_b32 s19, s1, s23 // (op 10383) 


s_add_i32 s1, s47, s19 // (op 10383) 


s_ashr_i32 s19, s1, s18 // (op 10383) 


s_lshl_b32 s1, s15, 24 // (op 10383) 
s_ashr_i32 s18, s1, 31 // (op 10383) 

s_xor_b32 s1, s19, s18 // (op 10383) 

s_sub_i32 s19, s1, s18 // (op 10383) 


s_add_u32 s22, s6, 128 // (op 10383) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10383) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10383) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10383) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10383) 

s_mul_i32 s1, s19, s22 // (op 10383) 

s_sub_i32 s47, s0, s1 // (op 10383) 


s_mul_hi_i32 s0, s16, s9 // (op 10383) 
s_add_i32 s1, s0, s16 // (op 10383) 

s_ashr_i32 s0, s1, 31 // (op 10383) 
s_cmp_eq_i32 s9, 0 // (op 10383) 
s_mov_b32 s18, scc // (op 10383) 
s_cmp_eq_i32 s18, 0 // (op 10383) 
s_cselect_b32 s19, 0, -1 // (op 10383) 

s_and_b32 s18, s45, 31 // (op 10383) 
s_lshl_b32 s24, 1, s18 // (op 10383) 
s_add_i32 s25, s24, s19 // (op 10383) 


s_and_b32 s19, s0, s25 // (op 10383) 


s_add_i32 s0, s1, s19 // (op 10383) 


s_ashr_i32 s1, s0, s18 // (op 10383) 


s_lshl_b32 s0, s45, 24 // (op 10383) 
s_ashr_i32 s18, s0, 31 // (op 10383) 

s_xor_b32 s0, s1, s18 // (op 10383) 

s_sub_i32 s1, s0, s18 // (op 10383) 


s_mul_i32 s0, s11, s22 // (op 10383) 

s_mul_i32 s18, s8, s0 // (op 10383) 

s_mul_i32 s0, s1, s18 // (op 10383) 

s_sub_u32 s18, s16, s0 // (op 10383) 

s_mul_hi_i32 s0, s18, s12 // (op 10383) 
s_add_i32 s19, s0, s18 // (op 10383) 


s_ashr_i32 s0, s19, 31 // (op 10383) 
s_cmp_eq_i32 s12, 0 // (op 10383) 
s_mov_b32 s18, scc // (op 10383) 
s_cmp_eq_i32 s18, 0 // (op 10383) 
s_cselect_b32 s22, 0, -1 // (op 10383) 

s_and_b32 s18, s44, 31 // (op 10383) 
s_lshl_b32 s23, 1, s18 // (op 10383) 
s_add_i32 s24, s23, s22 // (op 10383) 


s_and_b32 s25, s0, s24 // (op 10383) 


s_add_i32 s0, s19, s25 // (op 10383) 


s_ashr_i32 s19, s0, s18 // (op 10383) 


s_lshl_b32 s0, s44, 24 // (op 10383) 
s_ashr_i32 s18, s0, 31 // (op 10383) 

s_xor_b32 s0, s19, s18 // (op 10383) 

s_sub_i32 s19, s0, s18 // (op 10383) 


s_cmp_eq_i32 s1, 0 // (op 10383) 
s_mov_b32 s0, scc // (op 10383) 

s_cmp_eq_i32 s0, 0 // (op 10383) 
s_cselect_b32 s1, s19, s47 // (op 10383) 



v_lshl_add_u32 v15, s1, 7, v14 // (op 10383) 


v_mul_lo_u32 v25, v15, s31 // (op 10383) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 10383) most significant: high of low * low
v_mul_lo_u32 v24, v15, s30 // (op 10383) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 10383) most significant: sum


v_add_co_u32 v14, s[0:1], v22, v24 // (op 10383) least significant half
v_addc_co_u32 v15, s[0:1], v23, v25, s[0:1] // (op 10383) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 10383) 


v_lshrrev_b64 v[22:23], 3, v[14:15] // (op 10383) 

s_mul_i32 s1, s30, s7 // (op 10385) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 10385) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 10385) least significant
s_mul_hi_u32 s18, s30, s6 // (op 10385) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10385) most significant: sum
s_add_u32 s1, s1, s18 // (op 10385) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10385) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10385) 

s_mov_b32 s60, s32 // (op 10385) reg expression
s_mov_b32 s61, s33 // (op 10385) reg expression
s_mov_b32 s62, s0 // (op 10385) reg expression
s_mov_b32 s63, 131072 // (op 10385) call()

s_mul_hi_i32 s47, s16, s10 // (op 10837) 
v_and_b32 v14, 63, v1 // (op 10837) 
s_add_i32 s0, s47, s16 // (op 10837) 

v_lshrrev_b32 v15, 4, v14 // (op 10837) 
s_ashr_i32 s1, s0, 31 // (op 10837) 
v_and_b32 v24, 3, v15 // (op 10837) 

s_cmp_eq_i32 s10, 0 // (op 10837) 
s_mov_b32 s47, scc // (op 10837) 
v_and_b32 v15, 15, v1 // (op 10837) 
s_cmp_eq_i32 s47, 0 // (op 10837) 
s_cselect_b32 s18, 0, -1 // (op 10837) 

v_lshl_add_u32 v25, v24, 4, v15 // (op 10837) 


s_and_b32 s19, s13, 31 // (op 10837) 
v_lshrrev_b32 v15, 7, v1 // (op 10837) 
s_lshl_b32 s47, 1, s19 // (op 10837) 
v_lshl_add_u32 v24, v15, 6, v25 // (op 10837) 


s_add_i32 s22, s47, s18 // (op 10837) 


v_lshrrev_b32 v15, 6, v14 // (op 10837) 

s_and_b32 s18, s1, s22 // (op 10837) 


v_lshlrev_b32 v14, 2, v15 // (op 10837) 

s_add_i32 s1, s0, s18 // (op 10837) 


s_ashr_i32 s0, s1, s19 // (op 10837) 


s_lshl_b32 s1, s13, 24 // (op 10837) 
s_ashr_i32 s47, s1, 31 // (op 10837) 

s_xor_b32 s1, s0, s47 // (op 10837) 

s_sub_i32 s0, s1, s47 // (op 10837) 


s_mul_hi_i32 s1, s0, s14 // (op 10837) 
s_add_i32 s47, s1, s0 // (op 10837) 

s_ashr_i32 s1, s47, 31 // (op 10837) 
s_cmp_eq_i32 s14, 0 // (op 10837) 
s_mov_b32 s18, scc // (op 10837) 
s_cmp_eq_i32 s18, 0 // (op 10837) 
s_cselect_b32 s19, 0, -1 // (op 10837) 

s_and_b32 s18, s15, 31 // (op 10837) 
s_lshl_b32 s22, 1, s18 // (op 10837) 
s_add_i32 s23, s22, s19 // (op 10837) 


s_and_b32 s19, s1, s23 // (op 10837) 


s_add_i32 s1, s47, s19 // (op 10837) 


s_ashr_i32 s19, s1, s18 // (op 10837) 


s_lshl_b32 s1, s15, 24 // (op 10837) 
s_ashr_i32 s18, s1, 31 // (op 10837) 

s_xor_b32 s1, s19, s18 // (op 10837) 

s_sub_i32 s19, s1, s18 // (op 10837) 


s_mul_i32 s1, s0, s11 // (op 10837) 

s_sub_u32 s0, s16, s1 // (op 10837) 

s_mul_i32 s1, s19, s11 // (op 10837) 
s_add_u32 s1, s1, s0 // (op 10837) 


s_mul_hi_i32 s0, s16, s9 // (op 10837) 
s_add_i32 s47, s0, s16 // (op 10837) 

s_ashr_i32 s0, s47, 31 // (op 10837) 
s_cmp_eq_i32 s9, 0 // (op 10837) 
s_mov_b32 s18, scc // (op 10837) 
s_cmp_eq_i32 s18, 0 // (op 10837) 
s_cselect_b32 s19, 0, -1 // (op 10837) 

s_and_b32 s18, s45, 31 // (op 10837) 
s_lshl_b32 s22, 1, s18 // (op 10837) 
s_add_i32 s23, s22, s19 // (op 10837) 


s_and_b32 s19, s0, s23 // (op 10837) 


s_add_i32 s0, s47, s19 // (op 10837) 


s_ashr_i32 s19, s0, s18 // (op 10837) 


s_lshl_b32 s0, s45, 24 // (op 10837) 
s_ashr_i32 s18, s0, 31 // (op 10837) 

s_xor_b32 s0, s19, s18 // (op 10837) 

s_sub_i32 s19, s0, s18 // (op 10837) 


s_add_u32 s22, s6, 128 // (op 10837) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10837) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10837) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10837) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10837) 

s_mul_i32 s0, s11, s22 // (op 10837) 

s_mul_i32 s18, s8, s0 // (op 10837) 

s_mul_i32 s0, s19, s18 // (op 10837) 

s_sub_u32 s18, s16, s0 // (op 10837) 

s_mov_b32 s0, s18 // (op 10837) 
s_mul_hi_i32 s47, s0, s12 // (op 10837) 
s_add_i32 s22, s47, s0 // (op 10837) 


s_ashr_i32 s0, s22, 31 // (op 10837) 
s_cmp_eq_i32 s12, 0 // (op 10837) 
s_mov_b32 s47, scc // (op 10837) 
s_cmp_eq_i32 s47, 0 // (op 10837) 
s_cselect_b32 s23, 0, -1 // (op 10837) 

s_and_b32 s47, s44, 31 // (op 10837) 
s_lshl_b32 s24, 1, s47 // (op 10837) 
s_add_i32 s25, s24, s23 // (op 10837) 


s_and_b32 s23, s0, s25 // (op 10837) 


s_add_i32 s0, s22, s23 // (op 10837) 


s_ashr_i32 s22, s0, s47 // (op 10837) 


s_lshl_b32 s0, s44, 24 // (op 10837) 
s_ashr_i32 s47, s0, 31 // (op 10837) 

s_xor_b32 s0, s22, s47 // (op 10837) 

s_sub_i32 s22, s0, s47 // (op 10837) 


s_mul_i32 s0, s46, s11 // (op 10837) 
s_add_u32 s24, s20, 128 // (op 10837) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 10837) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 10837) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 10837) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 10837) 

s_sub_i32 s23, s24, s0 // (op 10837) 

s_mul_i32 s47, s22, s23 // (op 10837) 


s_sub_u32 s22, s18, s47 // (op 10837) 


s_add_u32 s18, s22, s0 // (op 10837) 


s_cmp_eq_i32 s19, 0 // (op 10837) 
s_mov_b32 s0, scc // (op 10837) 

s_cmp_eq_i32 s0, 0 // (op 10837) 
s_cselect_b32 s19, s18, s1 // (op 10837) 



v_lshl_add_u32 v15, s19, 7, v24 // (op 10837) 


v_mul_lo_u32 v24, v15, s34 // (op 10837) 
v_add_u32 v24, v24, v14 // (op 10837) 


s_mul_i32 s1, s34, s21 // (op 10839) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 10839) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 10839) least significant
s_mul_hi_u32 s18, s34, s20 // (op 10839) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10839) most significant: sum
s_add_u32 s1, s1, s18 // (op 10839) most significant: sum


s_mov_b32 s64, s36 // (op 10839) reg expression
s_mov_b32 s65, s37 // (op 10839) reg expression
s_mov_b32 s66, s0 // (op 10839) reg expression
s_mov_b32 s67, 131072 // (op 10839) call()

s_mul_hi_i32 s47, s16, s10 // (op 10858) 
v_and_b32 v14, 63, v1 // (op 10858) 
s_add_i32 s0, s47, s16 // (op 10858) 

v_lshrrev_b32 v15, 4, v14 // (op 10858) 
s_ashr_i32 s1, s0, 31 // (op 10858) 
v_and_b32 v25, 3, v15 // (op 10858) 

s_cmp_eq_i32 s10, 0 // (op 10858) 
s_mov_b32 s47, scc // (op 10858) 
v_and_b32 v15, 15, v1 // (op 10858) 
s_cmp_eq_i32 s47, 0 // (op 10858) 
s_cselect_b32 s18, 0, -1 // (op 10858) 

v_lshl_add_u32 v26, v25, 4, v15 // (op 10858) 


s_and_b32 s19, s13, 31 // (op 10858) 
v_lshrrev_b32 v15, 7, v1 // (op 10858) 
s_lshl_b32 s47, 1, s19 // (op 10858) 
v_lshl_add_u32 v25, v15, 6, v26 // (op 10858) 


s_add_i32 s22, s47, s18 // (op 10858) 


v_lshrrev_b32 v15, 6, v14 // (op 10858) 

s_and_b32 s18, s1, s22 // (op 10858) 


v_lshlrev_b32 v14, 2, v15 // (op 10858) 

s_add_i32 s1, s0, s18 // (op 10858) 


s_ashr_i32 s0, s1, s19 // (op 10858) 


s_lshl_b32 s1, s13, 24 // (op 10858) 
s_ashr_i32 s47, s1, 31 // (op 10858) 

s_xor_b32 s1, s0, s47 // (op 10858) 

s_sub_i32 s0, s1, s47 // (op 10858) 


s_mul_hi_i32 s1, s0, s14 // (op 10858) 
s_add_i32 s47, s1, s0 // (op 10858) 

s_ashr_i32 s1, s47, 31 // (op 10858) 
s_cmp_eq_i32 s14, 0 // (op 10858) 
s_mov_b32 s18, scc // (op 10858) 
s_cmp_eq_i32 s18, 0 // (op 10858) 
s_cselect_b32 s19, 0, -1 // (op 10858) 

s_and_b32 s18, s15, 31 // (op 10858) 
s_lshl_b32 s22, 1, s18 // (op 10858) 
s_add_i32 s23, s22, s19 // (op 10858) 


s_and_b32 s19, s1, s23 // (op 10858) 


s_add_i32 s1, s47, s19 // (op 10858) 


s_ashr_i32 s19, s1, s18 // (op 10858) 


s_lshl_b32 s1, s15, 24 // (op 10858) 
s_ashr_i32 s18, s1, 31 // (op 10858) 

s_xor_b32 s1, s19, s18 // (op 10858) 

s_sub_i32 s19, s1, s18 // (op 10858) 


s_mul_i32 s1, s0, s11 // (op 10858) 

s_sub_u32 s0, s16, s1 // (op 10858) 

s_mul_i32 s1, s19, s11 // (op 10858) 
s_add_u32 s1, s1, s0 // (op 10858) 


s_mul_hi_i32 s0, s16, s9 // (op 10858) 
s_add_i32 s47, s0, s16 // (op 10858) 

s_ashr_i32 s0, s47, 31 // (op 10858) 
s_cmp_eq_i32 s9, 0 // (op 10858) 
s_mov_b32 s18, scc // (op 10858) 
s_cmp_eq_i32 s18, 0 // (op 10858) 
s_cselect_b32 s19, 0, -1 // (op 10858) 

s_and_b32 s18, s45, 31 // (op 10858) 
s_lshl_b32 s22, 1, s18 // (op 10858) 
s_add_i32 s23, s22, s19 // (op 10858) 


s_and_b32 s19, s0, s23 // (op 10858) 


s_add_i32 s0, s47, s19 // (op 10858) 


s_ashr_i32 s19, s0, s18 // (op 10858) 


s_lshl_b32 s0, s45, 24 // (op 10858) 
s_ashr_i32 s18, s0, 31 // (op 10858) 

s_xor_b32 s0, s19, s18 // (op 10858) 

s_sub_i32 s19, s0, s18 // (op 10858) 


s_add_u32 s22, s6, 128 // (op 10858) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10858) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10858) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10858) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10858) 

s_mul_i32 s0, s11, s22 // (op 10858) 

s_mul_i32 s18, s8, s0 // (op 10858) 

s_mul_i32 s0, s19, s18 // (op 10858) 

s_sub_u32 s18, s16, s0 // (op 10858) 

s_mov_b32 s0, s18 // (op 10858) 
s_mul_hi_i32 s47, s0, s12 // (op 10858) 
s_add_i32 s22, s47, s0 // (op 10858) 


s_ashr_i32 s0, s22, 31 // (op 10858) 
s_cmp_eq_i32 s12, 0 // (op 10858) 
s_mov_b32 s47, scc // (op 10858) 
s_cmp_eq_i32 s47, 0 // (op 10858) 
s_cselect_b32 s23, 0, -1 // (op 10858) 

s_and_b32 s47, s44, 31 // (op 10858) 
s_lshl_b32 s24, 1, s47 // (op 10858) 
s_add_i32 s25, s24, s23 // (op 10858) 


s_and_b32 s23, s0, s25 // (op 10858) 


s_add_i32 s0, s22, s23 // (op 10858) 


s_ashr_i32 s22, s0, s47 // (op 10858) 


s_lshl_b32 s0, s44, 24 // (op 10858) 
s_ashr_i32 s47, s0, 31 // (op 10858) 

s_xor_b32 s0, s22, s47 // (op 10858) 

s_sub_i32 s22, s0, s47 // (op 10858) 


s_mul_i32 s0, s46, s11 // (op 10858) 
s_add_u32 s24, s20, 128 // (op 10858) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 10858) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 10858) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 10858) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 7 // (op 10858) 

s_sub_i32 s23, s24, s0 // (op 10858) 

s_mul_i32 s47, s22, s23 // (op 10858) 


s_sub_u32 s22, s18, s47 // (op 10858) 


s_add_u32 s18, s22, s0 // (op 10858) 


s_cmp_eq_i32 s19, 0 // (op 10858) 
s_mov_b32 s0, scc // (op 10858) 

s_cmp_eq_i32 s0, 0 // (op 10858) 
s_cselect_b32 s19, s18, s1 // (op 10858) 



v_lshl_add_u32 v15, s19, 7, v25 // (op 10858) 


v_mul_lo_u32 v25, v15, s34 // (op 10858) 
v_add_u32 v25, v25, v14 // (op 10858) 


s_mul_i32 s1, s34, s21 // (op 10860) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 10860) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 10860) least significant
s_mul_hi_u32 s18, s34, s20 // (op 10860) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10860) most significant: sum
s_add_u32 s1, s1, s18 // (op 10860) most significant: sum


s_mov_b32 s64, s36 // (op 10860) reg expression
s_mov_b32 s65, s37 // (op 10860) reg expression
s_mov_b32 s66, s0 // (op 10860) reg expression
s_mov_b32 s67, 131072 // (op 10860) call()

s_mul_hi_i32 s47, s16, s10 // (op 10889) 
v_and_b32 v14, 63, v1 // (op 10889) 
s_add_i32 s0, s47, s16 // (op 10889) 

v_lshrrev_b32 v15, 4, v14 // (op 10889) 
s_ashr_i32 s1, s0, 31 // (op 10889) 
v_and_b32 v26, 3, v15 // (op 10889) 

s_cmp_eq_i32 s10, 0 // (op 10889) 
s_mov_b32 s47, scc // (op 10889) 
v_and_b32 v15, 15, v1 // (op 10889) 
s_cmp_eq_i32 s47, 0 // (op 10889) 
s_cselect_b32 s18, 0, -1 // (op 10889) 

v_lshl_add_u32 v27, v26, 4, v15 // (op 10889) 


s_and_b32 s19, s13, 31 // (op 10889) 
v_lshrrev_b32 v15, 6, v1 // (op 10889) 
s_lshl_b32 s47, 1, s19 // (op 10889) 
v_and_b32 v26, 1, v15 // (op 10889) 

s_add_i32 s22, s47, s18 // (op 10889) 


v_lshl_add_u32 v15, v26, 6, v27 // (op 10889) 


s_and_b32 s18, s1, s22 // (op 10889) 


v_lshrrev_b32 v26, 6, v14 // (op 10889) 

s_add_i32 s1, s0, s18 // (op 10889) 


s_ashr_i32 s0, s1, s19 // (op 10889) 


s_lshl_b32 s1, s13, 24 // (op 10889) 
s_ashr_i32 s47, s1, 31 // (op 10889) 

s_xor_b32 s1, s0, s47 // (op 10889) 

s_sub_i32 s0, s1, s47 // (op 10889) 


s_mul_hi_i32 s1, s0, s14 // (op 10889) 
s_add_i32 s47, s1, s0 // (op 10889) 

s_ashr_i32 s1, s47, 31 // (op 10889) 
s_cmp_eq_i32 s14, 0 // (op 10889) 
s_mov_b32 s18, scc // (op 10889) 
s_cmp_eq_i32 s18, 0 // (op 10889) 
s_cselect_b32 s19, 0, -1 // (op 10889) 

s_and_b32 s18, s15, 31 // (op 10889) 
s_lshl_b32 s22, 1, s18 // (op 10889) 
s_add_i32 s23, s22, s19 // (op 10889) 


s_and_b32 s19, s1, s23 // (op 10889) 


s_add_i32 s1, s47, s19 // (op 10889) 


s_ashr_i32 s19, s1, s18 // (op 10889) 


s_lshl_b32 s1, s15, 24 // (op 10889) 
s_ashr_i32 s18, s1, 31 // (op 10889) 

s_xor_b32 s1, s19, s18 // (op 10889) 

s_sub_i32 s19, s1, s18 // (op 10889) 


s_add_u32 s22, s6, 128 // (op 10889) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10889) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10889) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10889) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10889) 

s_mul_i32 s1, s19, s22 // (op 10889) 

s_sub_i32 s47, s0, s1 // (op 10889) 


s_mul_hi_i32 s0, s16, s9 // (op 10889) 
s_add_i32 s1, s0, s16 // (op 10889) 

s_ashr_i32 s0, s1, 31 // (op 10889) 
s_cmp_eq_i32 s9, 0 // (op 10889) 
s_mov_b32 s18, scc // (op 10889) 
s_cmp_eq_i32 s18, 0 // (op 10889) 
s_cselect_b32 s19, 0, -1 // (op 10889) 

s_and_b32 s18, s45, 31 // (op 10889) 
s_lshl_b32 s24, 1, s18 // (op 10889) 
s_add_i32 s25, s24, s19 // (op 10889) 


s_and_b32 s19, s0, s25 // (op 10889) 


s_add_i32 s0, s1, s19 // (op 10889) 


s_ashr_i32 s1, s0, s18 // (op 10889) 


s_lshl_b32 s0, s45, 24 // (op 10889) 
s_ashr_i32 s18, s0, 31 // (op 10889) 

s_xor_b32 s0, s1, s18 // (op 10889) 

s_sub_i32 s1, s0, s18 // (op 10889) 


s_mul_i32 s0, s11, s22 // (op 10889) 

s_mul_i32 s18, s8, s0 // (op 10889) 

s_mul_i32 s0, s1, s18 // (op 10889) 

s_sub_u32 s18, s16, s0 // (op 10889) 

s_mul_hi_i32 s0, s18, s12 // (op 10889) 
s_add_i32 s19, s0, s18 // (op 10889) 


s_ashr_i32 s0, s19, 31 // (op 10889) 
s_cmp_eq_i32 s12, 0 // (op 10889) 
s_mov_b32 s18, scc // (op 10889) 
s_cmp_eq_i32 s18, 0 // (op 10889) 
s_cselect_b32 s22, 0, -1 // (op 10889) 

s_and_b32 s18, s44, 31 // (op 10889) 
s_lshl_b32 s23, 1, s18 // (op 10889) 
s_add_i32 s24, s23, s22 // (op 10889) 


s_and_b32 s25, s0, s24 // (op 10889) 


s_add_i32 s0, s19, s25 // (op 10889) 


s_ashr_i32 s19, s0, s18 // (op 10889) 


s_lshl_b32 s0, s44, 24 // (op 10889) 
s_ashr_i32 s18, s0, 31 // (op 10889) 

s_xor_b32 s0, s19, s18 // (op 10889) 

s_sub_i32 s19, s0, s18 // (op 10889) 


s_cmp_eq_i32 s1, 0 // (op 10889) 
s_mov_b32 s0, scc // (op 10889) 

s_cmp_eq_i32 s0, 0 // (op 10889) 
s_cselect_b32 s1, s19, s47 // (op 10889) 



v_lshl_add_u32 v14, s1, 7, v15 // (op 10889) 


v_mul_lo_u32 v15, v14, s38 // (op 10889) 

v_lshl_add_u32 v14, v26, 2, v15 // (op 10889) 


s_mul_i32 s1, s38, s7 // (op 10891) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 10891) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 10891) least significant
s_mul_hi_u32 s18, s38, s6 // (op 10891) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10891) most significant: sum
s_add_u32 s1, s1, s18 // (op 10891) most significant: sum


s_mov_b32 s68, s40 // (op 10891) reg expression
s_mov_b32 s69, s41 // (op 10891) reg expression
s_mov_b32 s70, s0 // (op 10891) reg expression
s_mov_b32 s71, 131072 // (op 10891) call()

s_mul_hi_i32 s47, s16, s10 // (op 10910) 
v_and_b32 v15, 63, v1 // (op 10910) 
s_add_i32 s0, s47, s16 // (op 10910) 

v_lshrrev_b32 v26, 4, v15 // (op 10910) 
s_ashr_i32 s1, s0, 31 // (op 10910) 
v_and_b32 v27, 3, v26 // (op 10910) 

s_cmp_eq_i32 s10, 0 // (op 10910) 
s_mov_b32 s47, scc // (op 10910) 
v_and_b32 v26, 15, v1 // (op 10910) 
s_cmp_eq_i32 s47, 0 // (op 10910) 
s_cselect_b32 s18, 0, -1 // (op 10910) 

v_lshl_add_u32 v28, v27, 4, v26 // (op 10910) 


s_and_b32 s19, s13, 31 // (op 10910) 
v_lshrrev_b32 v26, 6, v1 // (op 10910) 
s_lshl_b32 s47, 1, s19 // (op 10910) 
v_and_b32 v27, 1, v26 // (op 10910) 

s_add_i32 s22, s47, s18 // (op 10910) 


v_lshl_add_u32 v26, v27, 6, v28 // (op 10910) 


s_and_b32 s18, s1, s22 // (op 10910) 


v_lshrrev_b32 v27, 6, v15 // (op 10910) 

s_add_i32 s1, s0, s18 // (op 10910) 


s_ashr_i32 s0, s1, s19 // (op 10910) 


s_lshl_b32 s1, s13, 24 // (op 10910) 
s_ashr_i32 s47, s1, 31 // (op 10910) 

s_xor_b32 s1, s0, s47 // (op 10910) 

s_sub_i32 s0, s1, s47 // (op 10910) 


s_mul_hi_i32 s1, s0, s14 // (op 10910) 
s_add_i32 s47, s1, s0 // (op 10910) 

s_ashr_i32 s1, s47, 31 // (op 10910) 
s_cmp_eq_i32 s14, 0 // (op 10910) 
s_mov_b32 s18, scc // (op 10910) 
s_cmp_eq_i32 s18, 0 // (op 10910) 
s_cselect_b32 s19, 0, -1 // (op 10910) 

s_and_b32 s18, s15, 31 // (op 10910) 
s_lshl_b32 s22, 1, s18 // (op 10910) 
s_add_i32 s23, s22, s19 // (op 10910) 


s_and_b32 s19, s1, s23 // (op 10910) 


s_add_i32 s1, s47, s19 // (op 10910) 


s_ashr_i32 s19, s1, s18 // (op 10910) 


s_lshl_b32 s1, s15, 24 // (op 10910) 
s_ashr_i32 s18, s1, 31 // (op 10910) 

s_xor_b32 s1, s19, s18 // (op 10910) 

s_sub_i32 s19, s1, s18 // (op 10910) 


s_add_u32 s22, s6, 128 // (op 10910) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10910) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10910) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10910) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10910) 

s_mul_i32 s1, s19, s22 // (op 10910) 

s_sub_i32 s47, s0, s1 // (op 10910) 


s_mul_hi_i32 s0, s16, s9 // (op 10910) 
s_add_i32 s1, s0, s16 // (op 10910) 

s_ashr_i32 s0, s1, 31 // (op 10910) 
s_cmp_eq_i32 s9, 0 // (op 10910) 
s_mov_b32 s18, scc // (op 10910) 
s_cmp_eq_i32 s18, 0 // (op 10910) 
s_cselect_b32 s19, 0, -1 // (op 10910) 

s_and_b32 s18, s45, 31 // (op 10910) 
s_lshl_b32 s24, 1, s18 // (op 10910) 
s_add_i32 s25, s24, s19 // (op 10910) 


s_and_b32 s19, s0, s25 // (op 10910) 


s_add_i32 s0, s1, s19 // (op 10910) 


s_ashr_i32 s1, s0, s18 // (op 10910) 


s_lshl_b32 s0, s45, 24 // (op 10910) 
s_ashr_i32 s18, s0, 31 // (op 10910) 

s_xor_b32 s0, s1, s18 // (op 10910) 

s_sub_i32 s1, s0, s18 // (op 10910) 


s_mul_i32 s0, s11, s22 // (op 10910) 

s_mul_i32 s18, s8, s0 // (op 10910) 

s_mul_i32 s0, s1, s18 // (op 10910) 

s_sub_u32 s18, s16, s0 // (op 10910) 

s_mul_hi_i32 s0, s18, s12 // (op 10910) 
s_add_i32 s19, s0, s18 // (op 10910) 


s_ashr_i32 s0, s19, 31 // (op 10910) 
s_cmp_eq_i32 s12, 0 // (op 10910) 
s_mov_b32 s18, scc // (op 10910) 
s_cmp_eq_i32 s18, 0 // (op 10910) 
s_cselect_b32 s22, 0, -1 // (op 10910) 

s_and_b32 s18, s44, 31 // (op 10910) 
s_lshl_b32 s23, 1, s18 // (op 10910) 
s_add_i32 s24, s23, s22 // (op 10910) 


s_and_b32 s25, s0, s24 // (op 10910) 


s_add_i32 s0, s19, s25 // (op 10910) 


s_ashr_i32 s19, s0, s18 // (op 10910) 


s_lshl_b32 s0, s44, 24 // (op 10910) 
s_ashr_i32 s18, s0, 31 // (op 10910) 

s_xor_b32 s0, s19, s18 // (op 10910) 

s_sub_i32 s19, s0, s18 // (op 10910) 


s_cmp_eq_i32 s1, 0 // (op 10910) 
s_mov_b32 s0, scc // (op 10910) 

s_cmp_eq_i32 s0, 0 // (op 10910) 
s_cselect_b32 s1, s19, s47 // (op 10910) 



v_lshl_add_u32 v15, s1, 7, v26 // (op 10910) 


v_mul_lo_u32 v26, v15, s38 // (op 10910) 

v_lshl_add_u32 v15, v27, 2, v26 // (op 10910) 


s_mul_i32 s1, s38, s7 // (op 10912) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 10912) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 10912) least significant
s_mul_hi_u32 s18, s38, s6 // (op 10912) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10912) most significant: sum
s_add_u32 s1, s1, s18 // (op 10912) most significant: sum


s_mov_b32 s68, s40 // (op 10912) reg expression
s_mov_b32 s69, s41 // (op 10912) reg expression
s_mov_b32 s70, s0 // (op 10912) reg expression
s_mov_b32 s71, 131072 // (op 10912) call()

v_and_b32 v26, -4, v1 // (op 10511) 
v_sub_u32 v27, v1, v26 // (op 10511) 

v_lshlrev_b32 v26, 5, v27 // (op 10511) 

v_lshrrev_b32 v27, 2, v1 // (op 10511) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 10511) 


v_lshlrev_b32 v26, 2, v28 // (op 10511) 

v_lshrrev_b32 v27, 3, v26 // (op 10511) 

s_nop 0
v_readfirstlane_b32 s47, v27 // (op 10511) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 10226) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10226) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 10226) 


v_mov_b64 v[28:29], v[20:21] // (op 10226) 
s_mov_b32 m0, s47 // (op 10226) reg expression
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 10226) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 10226) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 10226) most significant half

s_add_u32 m0, m0, 4096 // (op 10226) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 10226) Load value direct to lds



v_and_b32 v26, -4, v1 // (op 10454) 
v_sub_u32 v27, v1, v26 // (op 10454) 

v_lshlrev_b32 v26, 5, v27 // (op 10454) 

v_lshrrev_b32 v27, 2, v1 // (op 10454) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 10454) 


v_lshlrev_b32 v26, 2, v28 // (op 10454) 

v_lshrrev_b32 v27, 3, v26 // (op 10454) 

s_nop 0
v_readfirstlane_b32 s47, v27 // (op 10454) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 10231) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10231) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 10231) 


v_mov_b64 v[28:29], v[16:17] // (op 10231) 
s_add_u32 s0, 8192, s47 // (op 10231) 
s_mov_b32 m0, s0 // (op 10231) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 10231) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 10231) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 10231) most significant half

s_add_u32 m0, m0, 4096 // (op 10231) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 10231) Load value direct to lds






v_mov_b32 v26, v14 // (op 74) reg expression

buffer_load_dwordx4 v[28:31], v26, s[68:71], 0 offen // (op 74) Load value




v_mov_b32 v14, v24 // (op 70) reg expression

buffer_load_dwordx4 v[32:35], v14, s[64:67], 0 offen // (op 70) Load value


v_and_b32 v14, -4, v1 // (op 10622) 
v_sub_u32 v24, v1, v14 // (op 10622) 

v_lshlrev_b32 v14, 5, v24 // (op 10622) 

v_lshrrev_b32 v24, 2, v1 // (op 10622) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 10622) 


v_lshlrev_b32 v14, 2, v26 // (op 10622) 

v_lshrrev_b32 v24, 3, v14 // (op 10622) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 10622) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 10236) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10236) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 10236) 


v_mov_b64 v[36:37], v[20:21] // (op 10236) 
s_add_u32 s0, 16384, s47 // (op 10236) 
s_mov_b32 m0, s0 // (op 10236) reg expression
buffer_load_dwordx4 v36, s[60:63], 64 offen offset: 0 lds // (op 10236) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v26 // (op 10236) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 10236) most significant half

s_add_u32 m0, m0, 4096 // (op 10236) 
buffer_load_dwordx4 v36, s[60:63], 64 offen offset: 0 lds // (op 10236) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 10578) 
v_sub_u32 v24, v1, v14 // (op 10578) 

v_lshlrev_b32 v14, 5, v24 // (op 10578) 

v_lshrrev_b32 v24, 2, v1 // (op 10578) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 10578) 


v_lshlrev_b32 v14, 2, v26 // (op 10578) 

v_lshrrev_b32 v24, 3, v14 // (op 10578) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 10578) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 10238) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10238) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 10238) 


v_mov_b64 v[36:37], v[16:17] // (op 10238) 
s_add_u32 s0, 24576, s47 // (op 10238) 
s_mov_b32 m0, s0 // (op 10238) reg expression
buffer_load_dwordx4 v36, s[48:51], 64 offen offset: 0 lds // (op 10238) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v26 // (op 10238) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 10238) most significant half

s_add_u32 m0, m0, 4096 // (op 10238) 
buffer_load_dwordx4 v36, s[48:51], 64 offen offset: 0 lds // (op 10238) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 10710) 
v_sub_u32 v24, v1, v14 // (op 10710) 

v_lshlrev_b32 v14, 5, v24 // (op 10710) 

v_lshrrev_b32 v24, 2, v1 // (op 10710) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 10710) 


v_lshlrev_b32 v14, 2, v26 // (op 10710) 

v_lshrrev_b32 v24, 3, v14 // (op 10710) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 10710) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 10240) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10240) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 10240) 


v_mov_b64 v[36:37], v[20:21] // (op 10240) 
s_add_u32 s0, 32768, s47 // (op 10240) 
s_mov_b32 m0, s0 // (op 10240) reg expression
s_mov_b32 s1, 128 // (op 10240) call()
buffer_load_dwordx4 v36, s[60:63], s1 offen offset: 0 lds // (op 10240) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v26 // (op 10240) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 10240) most significant half

s_add_u32 m0, m0, 4096 // (op 10240) 
buffer_load_dwordx4 v36, s[60:63], s1 offen offset: 0 lds // (op 10240) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 10666) 
v_sub_u32 v24, v1, v14 // (op 10666) 

v_lshlrev_b32 v14, 5, v24 // (op 10666) 

v_lshrrev_b32 v24, 2, v1 // (op 10666) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 10666) 


v_lshlrev_b32 v14, 2, v26 // (op 10666) 

v_lshrrev_b32 v24, 3, v14 // (op 10666) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 10666) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 10242) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10242) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 10242) 


v_mov_b64 v[36:37], v[16:17] // (op 10242) 
s_add_u32 s0, 40960, s47 // (op 10242) 
s_mov_b32 m0, s0 // (op 10242) reg expression
s_mov_b32 s1, 128 // (op 10242) call()
buffer_load_dwordx4 v36, s[48:51], s1 offen offset: 0 lds // (op 10242) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v26 // (op 10242) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 10242) most significant half

s_add_u32 m0, m0, 4096 // (op 10242) 
buffer_load_dwordx4 v36, s[48:51], s1 offen offset: 0 lds // (op 10242) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 10798) 
v_sub_u32 v24, v1, v14 // (op 10798) 

v_lshlrev_b32 v14, 5, v24 // (op 10798) 

v_lshrrev_b32 v24, 2, v1 // (op 10798) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 10798) 


v_lshlrev_b32 v14, 2, v26 // (op 10798) 

v_lshrrev_b32 v24, 3, v14 // (op 10798) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 10798) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 10244) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10244) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 10244) 


v_mov_b64 v[36:37], v[20:21] // (op 10244) 
s_add_u32 s0, 49152, s47 // (op 10244) 
s_mov_b32 m0, s0 // (op 10244) reg expression
s_mov_b32 s1, 192 // (op 10244) call()
buffer_load_dwordx4 v36, s[60:63], s1 offen offset: 0 lds // (op 10244) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v26 // (op 10244) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 10244) most significant half

s_add_u32 m0, m0, 4096 // (op 10244) 
buffer_load_dwordx4 v36, s[60:63], s1 offen offset: 0 lds // (op 10244) Load value direct to lds






v_and_b32 v14, -4, v1 // (op 10754) 
v_sub_u32 v24, v1, v14 // (op 10754) 

v_lshlrev_b32 v14, 5, v24 // (op 10754) 

v_lshrrev_b32 v24, 2, v1 // (op 10754) 
v_lshl_add_u32 v20, v24, 7, v14 // (op 10754) 


v_lshlrev_b32 v14, 2, v20 // (op 10754) 

v_lshrrev_b32 v24, 3, v14 // (op 10754) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 10754) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 10246) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10246) 

v_lshrrev_b64 v[20:21], 3, s[18:19] // (op 10246) 


v_mov_b64 v[26:27], v[16:17] // (op 10246) 
s_add_u32 s0, 57344, s47 // (op 10246) 
s_mov_b32 m0, s0 // (op 10246) reg expression
s_mov_b32 s1, 192 // (op 10246) call()
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 10246) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v20 // (op 10246) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 10246) most significant half

s_add_u32 m0, m0, 4096 // (op 10246) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 10246) Load value direct to lds






s_waitcnt vmcnt(16)
s_barrier  // (op 11199) 



ds_read_b128 v[36:39], v4 // (op 9808) Load local data 



ds_read_b128 v[40:43], v4 offset:1024 // (op 9813) Load local data 



ds_read_b128 v[44:47], v4 offset:2048 // (op 9818) Load local data 



ds_read_b128 v[48:51], v4 offset:3072 // (op 9823) Load local data 
s_waitcnt vmcnt(14)
s_barrier  // (op 11182) 



ds_read_b128 v[52:55], v3 offset:8192 // (op 9788) Load local data 



ds_read_b128 v[56:59], v3 offset:9216 // (op 9793) Load local data 



ds_read_b128 v[60:63], v3 offset:10240 // (op 9798) Load local data 



ds_read_b128 v[64:67], v3 offset:11264 // (op 9803) Load local data 
v_and_b32 v14, -4, v1 // (op 10426) 
v_sub_u32 v24, v1, v14 // (op 10426) 

v_lshlrev_b32 v14, 5, v24 // (op 10426) 

v_lshrrev_b32 v24, 2, v1 // (op 10426) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10426) 


v_lshlrev_b32 v14, 2, v16 // (op 10426) 

v_lshrrev_b32 v24, 3, v14 // (op 10426) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 10426) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 10483) 
v_sub_u32 v24, v1, v14 // (op 10483) 

v_lshlrev_b32 v14, 5, v24 // (op 10483) 

v_lshrrev_b32 v24, 2, v1 // (op 10483) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10483) 


v_lshlrev_b32 v14, 2, v16 // (op 10483) 

v_lshrrev_b32 v24, 3, v14 // (op 10483) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 10483) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 10563) 
v_sub_u32 v24, v1, v14 // (op 10563) 

v_lshlrev_b32 v14, 5, v24 // (op 10563) 

v_lshrrev_b32 v24, 2, v1 // (op 10563) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10563) 


v_lshlrev_b32 v14, 2, v16 // (op 10563) 

v_lshrrev_b32 v24, 3, v14 // (op 10563) 

s_nop 0
v_readfirstlane_b32 s1, v24 // (op 10563) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 10607) 
v_sub_u32 v24, v1, v14 // (op 10607) 

v_lshlrev_b32 v14, 5, v24 // (op 10607) 

v_lshrrev_b32 v24, 2, v1 // (op 10607) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10607) 


v_lshlrev_b32 v14, 2, v16 // (op 10607) 

v_lshrrev_b32 v24, 3, v14 // (op 10607) 

s_nop 0
v_readfirstlane_b32 s18, v24 // (op 10607) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 10651) 
v_sub_u32 v24, v1, v14 // (op 10651) 

v_lshlrev_b32 v14, 5, v24 // (op 10651) 

v_lshrrev_b32 v24, 2, v1 // (op 10651) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10651) 


v_lshlrev_b32 v14, 2, v16 // (op 10651) 

v_lshrrev_b32 v24, 3, v14 // (op 10651) 

s_nop 0
v_readfirstlane_b32 s19, v24 // (op 10651) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 10695) 
v_sub_u32 v24, v1, v14 // (op 10695) 

v_lshlrev_b32 v14, 5, v24 // (op 10695) 

v_lshrrev_b32 v24, 2, v1 // (op 10695) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10695) 


v_lshlrev_b32 v14, 2, v16 // (op 10695) 

v_lshrrev_b32 v24, 3, v14 // (op 10695) 

s_nop 0
v_readfirstlane_b32 s22, v24 // (op 10695) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 10739) 
v_sub_u32 v24, v1, v14 // (op 10739) 

v_lshlrev_b32 v14, 5, v24 // (op 10739) 

v_lshrrev_b32 v24, 2, v1 // (op 10739) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10739) 


v_lshlrev_b32 v14, 2, v16 // (op 10739) 

v_lshrrev_b32 v24, 3, v14 // (op 10739) 

s_nop 0
v_readfirstlane_b32 s23, v24 // (op 10739) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 10783) 
v_sub_u32 v24, v1, v14 // (op 10783) 

v_lshlrev_b32 v14, 5, v24 // (op 10783) 

v_lshrrev_b32 v24, 2, v1 // (op 10783) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10783) 


v_lshlrev_b32 v14, 2, v16 // (op 10783) 

v_lshrrev_b32 v24, 3, v14 // (op 10783) 

s_nop 0
v_readfirstlane_b32 s24, v24 // (op 10783) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s25, 0 // (op 2282) call()
s_ashr_i64 s[72:73], s[42:43], 7 // (op 2281) 
s_mov_b32 s74, s72 // (op 2281) convert

s_and_b32 s72, s74, 4294967292 // (op 2281) 

s_cmp_lt_u32 s25, s72 // (op 2281) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoop_2281 // (op 2281) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoop_2281 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoop_2281:
s_add_u32 s25, s25, 4 // (op 2283) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 11202) 
s_waitcnt vmcnt(13)
v_mov_b32 v68, v28 // (op 10126) reg expression

v_mov_b32 v69, v29 // (op 10126) reg expression
v_mov_b32 v70, v30 // (op 10126) reg expression
v_mov_b32 v71, v31 // (op 10126) reg expression
s_lshl_b64 s[72:73], s[30:31], 6 // (op 10179) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10179) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10179) 


v_mov_b64 v[20:21], v[22:23] // (op 10179) 
v_mov_b64 v[26:27], v[20:21] // (op 10179) 
s_mov_b32 m0, s0 // (op 10179) reg expression
s_mov_b32 s72, 256 // (op 10179) call()
s_waitcnt vmcnt(12)
v_mov_b32 v72, v32 // (op 10139) reg expression
v_mov_b32 v73, v33 // (op 10139) reg expression
v_mov_b32 v74, v34 // (op 10139) reg expression
v_mov_b32 v75, v35 // (op 10139) reg expression
buffer_load_dwordx4 v26, s[60:63], s72 offen offset: 0 lds // (op 10179) Load value direct to lds
v_add_co_u32 v26, s[74:75], v26, v16 // (op 10179) least significant half
v_addc_co_u32 v27, s[74:75], 0, v27, s[74:75] // (op 10179) most significant half

s_add_u32 m0, m0, 4096 // (op 10179) 
v_bfe_u32 v76, v72, 0, 8 // (op 9141) 
v_bfe_u32 v77, v72, 8, 8 // (op 9141) 
v_bfe_u32 v78, v72, 16, 8 // (op 9141) 
v_bfe_u32 v79, v72, 24, 8 // (op 9141) 
v_permlane16_swap_b32 v76, v77 // (op 9141) 
s_nop 0
v_permlane16_swap_b32 v78, v79 // (op 9141) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v76, v78 // (op 9141) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v77, v79 // (op 9141) 
buffer_load_dwordx4 v26, s[60:63], s72 offen offset: 0 lds // (op 10179) Load value direct to lds



v_bfe_u32 v80, v68, 0, 8 // (op 9270) 
v_bfe_u32 v81, v68, 8, 8 // (op 9270) 
v_bfe_u32 v82, v68, 16, 8 // (op 9270) 
v_bfe_u32 v83, v68, 24, 8 // (op 9270) 
v_permlane16_swap_b32 v80, v81 // (op 9270) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 9270) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 9270) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 9270) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 11185) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[36:39], a[0:3], v76, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 76) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 10174) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[40:43], a[4:7], v76, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 417) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10174) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:55], v[44:47], a[8:11], v76, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 643) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10174) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:55], v[48:51], a[12:15], v76, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 869) 
v_mov_b64 v[26:27], v[18:19] // (op 10174) 
v_mov_b64 v[84:85], v[26:27] // (op 10174) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[36:39], a[16:19], v77, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1541) 
s_add_u32 s72, 8192, s47 // (op 10174) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[40:43], a[20:23], v77, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1465) 
s_mov_b32 m0, s72 // (op 10174) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[44:47], a[24:27], v77, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1389) 
s_mov_b32 s73, 256 // (op 10174) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[48:51], a[28:31], v77, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1313) 
buffer_load_dwordx4 v84, s[48:51], s73 offen offset: 0 lds // (op 10174) Load value direct to lds
v_add_co_u32 v84, s[74:75], v84, v16 // (op 10174) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[60:63], v[36:39], a[32:35], v78, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2543) 
v_addc_co_u32 v85, s[74:75], 0, v85, s[74:75] // (op 10174) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:63], v[40:43], a[36:39], v78, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2467) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:63], v[44:47], a[40:43], v78, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2391) 
s_add_u32 m0, m0, 4096 // (op 10174) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[60:63], v[48:51], a[44:47], v78, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2315) 
s_waitcnt vmcnt(13)
s_barrier  // (op 11222) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[64:67], v[36:39], a[48:51], v79, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3545) 
ds_read_b128 v[86:89], v6 offset:16384 // (op 212) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[64:67], v[40:43], a[52:55], v79, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3469) 
ds_read_b128 v[90:93], v6 offset:17408 // (op 428) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[64:67], v[44:47], a[56:59], v79, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3393) 
ds_read_b128 v[94:97], v6 offset:18432 // (op 654) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[64:67], v[48:51], a[60:63], v79, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3317) 
ds_read_b128 v[98:101], v6 offset:19456 // (op 880) Load local data 


buffer_load_dwordx4 v84, s[48:51], s73 offen offset: 0 lds // (op 10174) Load value direct to lds






v_mov_b32 v14, v15 // (op 10134) 
v_add_u32 v24, 16, v14 // (op 10134) 

buffer_load_dwordx4 v[28:31], v24, s[68:71], 0 offen // (op 10134) Load value


v_add_u32 v15, 16, v15 // (op 10900) 
s_waitcnt vmcnt(13)
s_barrier  // (op 11215) 





ds_read_b128 v[52:55], v5 offset:24576 // (op 210) Load local data 
v_mov_b32 v14, v25 // (op 10147) 



ds_read_b128 v[56:59], v5 offset:25600 // (op 1323) Load local data 
v_add_u32 v24, 16, v14 // (op 10147) 




ds_read_b128 v[60:63], v5 offset:26624 // (op 2325) Load local data 
buffer_load_dwordx4 v[32:35], v24, s[64:67], 0 offen // (op 10147) Load value





v_add_u32 v25, 16, v25 // (op 10848) 
ds_read_b128 v[76:79], v5 offset:27648 // (op 3327) Load local data 
v_bfe_u32 v80, v73, 0, 8 // (op 9156) 
v_bfe_u32 v81, v73, 8, 8 // (op 9156) 
v_bfe_u32 v82, v73, 16, 8 // (op 9156) 
v_bfe_u32 v83, v73, 24, 8 // (op 9156) 
v_permlane16_swap_b32 v80, v81 // (op 9156) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 9156) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 9156) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 9156) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 11226) 

s_lshl_b64 s[72:73], s[30:31], 6 // (op 10201) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10201) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10201) 


v_mov_b64 v[84:85], v[20:21] // (op 10201) 
s_add_u32 s72, 16384, s18 // (op 10201) 
s_mov_b32 m0, s72 // (op 10201) reg expression
s_mov_b32 s73, 320 // (op 10201) call()
v_bfe_u32 v102, v69, 0, 8 // (op 9285) 
v_bfe_u32 v103, v69, 8, 8 // (op 9285) 
v_bfe_u32 v104, v69, 16, 8 // (op 9285) 
v_bfe_u32 v105, v69, 24, 8 // (op 9285) 
v_permlane16_swap_b32 v102, v103 // (op 9285) 
s_nop 0
v_permlane16_swap_b32 v104, v105 // (op 9285) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v102, v104 // (op 9285) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v103, v105 // (op 9285) 
buffer_load_dwordx4 v84, s[60:63], s73 offen offset: 0 lds // (op 10201) Load value direct to lds
v_add_co_u32 v84, s[74:75], v84, v16 // (op 10201) least significant half
v_addc_co_u32 v85, s[74:75], 0, v85, s[74:75] // (op 10201) most significant half

s_add_u32 m0, m0, 4096 // (op 10201) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[86:89], a[0:3], v80, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 217) 
buffer_load_dwordx4 v84, s[60:63], s73 offen offset: 0 lds // (op 10201) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[90:93], a[4:7], v80, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 418) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 11219) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:55], v[94:97], a[8:11], v80, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 644) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 10196) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:55], v[98:101], a[12:15], v80, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 870) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10196) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10196) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[86:89], a[16:19], v81, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1542) 
v_mov_b64 v[84:85], v[26:27] // (op 10196) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[90:93], a[20:23], v81, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1466) 
s_add_u32 s72, 24576, s1 // (op 10196) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[94:97], a[24:27], v81, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1390) 
s_mov_b32 m0, s72 // (op 10196) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[98:101], a[28:31], v81, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1314) 
s_mov_b32 s73, 320 // (op 10196) call()
buffer_load_dwordx4 v84, s[48:51], s73 offen offset: 0 lds // (op 10196) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[60:63], v[86:89], a[32:35], v82, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2544) 
v_add_co_u32 v84, s[74:75], v84, v16 // (op 10196) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:63], v[90:93], a[36:39], v82, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2468) 
v_addc_co_u32 v85, s[74:75], 0, v85, s[74:75] // (op 10196) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:63], v[94:97], a[40:43], v82, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2392) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[60:63], v[98:101], a[44:47], v82, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2316) 
s_add_u32 m0, m0, 4096 // (op 10196) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[86:89], a[48:51], v83, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3546) 
s_waitcnt vmcnt(15)
s_barrier  // (op 11237) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[90:93], a[52:55], v83, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3470) 
ds_read_b128 v[36:39], v8 offset:32768 // (op 240) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[94:97], a[56:59], v83, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3394) 
ds_read_b128 v[40:43], v8 offset:33792 // (op 438) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[98:101], a[60:63], v83, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3318) 
ds_read_b128 v[44:47], v8 offset:34816 // (op 664) Load local data 







ds_read_b128 v[48:51], v8 offset:35840 // (op 890) Load local data 
buffer_load_dwordx4 v84, s[48:51], s73 offen offset: 0 lds // (op 10196) Load value direct to lds




s_waitcnt vmcnt(14)
s_barrier  // (op 11230) 



ds_read_b128 v[52:55], v7 offset:40960 // (op 238) Load local data 



ds_read_b128 v[56:59], v7 offset:41984 // (op 1333) Load local data 



ds_read_b128 v[76:79], v7 offset:43008 // (op 2335) Load local data 



ds_read_b128 v[80:83], v7 offset:44032 // (op 3337) Load local data 
v_bfe_u32 v84, v74, 0, 8 // (op 9173) 
v_bfe_u32 v85, v74, 8, 8 // (op 9173) 
v_bfe_u32 v86, v74, 16, 8 // (op 9173) 
v_bfe_u32 v87, v74, 24, 8 // (op 9173) 
v_permlane16_swap_b32 v84, v85 // (op 9173) 
s_nop 0
v_permlane16_swap_b32 v86, v87 // (op 9173) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v84, v86 // (op 9173) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v85, v87 // (op 9173) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 11240) 

s_lshl_b64 s[72:73], s[30:31], 6 // (op 10211) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10211) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10211) 


v_mov_b64 v[88:89], v[20:21] // (op 10211) 
s_add_u32 s72, 32768, s22 // (op 10211) 
s_mov_b32 m0, s72 // (op 10211) reg expression
s_mov_b32 s73, 384 // (op 10211) call()
buffer_load_dwordx4 v88, s[60:63], s73 offen offset: 0 lds // (op 10211) Load value direct to lds
v_add_co_u32 v88, s[74:75], v88, v16 // (op 10211) least significant half
v_addc_co_u32 v89, s[74:75], 0, v89, s[74:75] // (op 10211) most significant half

s_add_u32 m0, m0, 4096 // (op 10211) 
buffer_load_dwordx4 v88, s[60:63], s73 offen offset: 0 lds // (op 10211) Load value direct to lds




v_bfe_u32 v88, v70, 0, 8 // (op 9303) 
v_bfe_u32 v89, v70, 8, 8 // (op 9303) 
v_bfe_u32 v90, v70, 16, 8 // (op 9303) 
v_bfe_u32 v91, v70, 24, 8 // (op 9303) 
v_permlane16_swap_b32 v88, v89 // (op 9303) 
s_nop 0
v_permlane16_swap_b32 v90, v91 // (op 9303) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v88, v90 // (op 9303) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v89, v91 // (op 9303) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 11234) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[36:39], a[0:3], v84, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 245) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 10206) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[40:43], a[4:7], v84, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 419) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10206) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:55], v[44:47], a[8:11], v84, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 645) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10206) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:55], v[48:51], a[12:15], v84, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 871) 
v_mov_b64 v[92:93], v[26:27] // (op 10206) 
s_add_u32 s72, 40960, s19 // (op 10206) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[36:39], a[16:19], v85, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1543) 
s_mov_b32 m0, s72 // (op 10206) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[40:43], a[20:23], v85, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1467) 
s_mov_b32 s73, 384 // (op 10206) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[44:47], a[24:27], v85, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1391) 
s_waitcnt vmcnt(14)
s_barrier  // (op 11250) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[48:51], a[28:31], v85, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1315) 
ds_read_b128 v[60:63], v10 offset:49152 // (op 268) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[36:39], a[32:35], v86, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2545) 
ds_read_b128 v[64:67], v10 offset:50176 // (op 448) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[40:43], a[36:39], v86, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2469) 
ds_read_b128 v[94:97], v10 offset:51200 // (op 674) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[44:47], a[40:43], v86, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2393) 
ds_read_b128 v[98:101], v10 offset:52224 // (op 900) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[48:51], a[44:47], v86, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2317) 

s_waitcnt vmcnt(12)
s_barrier  // (op 11243) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[36:39], a[48:51], v87, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3547) 
ds_read_b128 v[52:55], v9 offset:57344 // (op 266) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[40:43], a[52:55], v87, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3471) 
ds_read_b128 v[56:59], v9 offset:58368 // (op 1343) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[44:47], a[56:59], v87, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3395) 
ds_read_b128 v[76:79], v9 offset:59392 // (op 2345) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[48:51], a[60:63], v87, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3319) 
ds_read_b128 v[102:105], v9 offset:60416 // (op 3347) Load local data 



buffer_load_dwordx4 v92, s[48:51], s73 offen offset: 0 lds // (op 10206) Load value direct to lds
v_add_co_u32 v92, s[74:75], v92, v16 // (op 10206) least significant half
v_addc_co_u32 v93, s[74:75], 0, v93, s[74:75] // (op 10206) most significant half

s_add_u32 m0, m0, 4096 // (op 10206) 
buffer_load_dwordx4 v92, s[48:51], s73 offen offset: 0 lds // (op 10206) Load value direct to lds




v_bfe_u32 v80, v75, 0, 8 // (op 9190) 
v_bfe_u32 v81, v75, 8, 8 // (op 9190) 
v_bfe_u32 v82, v75, 16, 8 // (op 9190) 
v_bfe_u32 v83, v75, 24, 8 // (op 9190) 
v_permlane16_swap_b32 v80, v81 // (op 9190) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 9190) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 9190) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 9190) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 11253) 


s_lshl_b64 s[72:73], s[30:31], 6 // (op 10221) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10221) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10221) 


v_mov_b64 v[72:73], v[20:21] // (op 10221) 
s_add_u32 s72, 49152, s24 // (op 10221) 
s_mov_b32 m0, s72 // (op 10221) reg expression
s_mov_b32 s73, 448 // (op 10221) call()
buffer_load_dwordx4 v72, s[60:63], s73 offen offset: 0 lds // (op 10221) Load value direct to lds
v_add_co_u32 v72, s[74:75], v72, v16 // (op 10221) least significant half
v_addc_co_u32 v73, s[74:75], 0, v73, s[74:75] // (op 10221) most significant half

s_add_u32 m0, m0, 4096 // (op 10221) 
v_bfe_u32 v84, v71, 0, 8 // (op 9321) 
v_bfe_u32 v85, v71, 8, 8 // (op 9321) 
v_bfe_u32 v86, v71, 16, 8 // (op 9321) 
v_bfe_u32 v87, v71, 24, 8 // (op 9321) 
v_permlane16_swap_b32 v84, v85 // (op 9321) 
s_nop 0
v_permlane16_swap_b32 v86, v87 // (op 9321) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v84, v86 // (op 9321) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v85, v87 // (op 9321) 
buffer_load_dwordx4 v72, s[60:63], s73 offen offset: 0 lds // (op 10221) Load value direct to lds






v_add_co_u32 v22, vcc, 256, v22 // (op 10375) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 10375) most significant half
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[60:63], a[0:3], v80, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 273) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 11247) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[64:67], a[4:7], v80, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 420) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 10216) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:55], v[94:97], a[8:11], v80, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 646) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10216) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:55], v[98:101], a[12:15], v80, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 872) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10216) 


v_mov_b64 v[20:21], v[26:27] // (op 10216) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[60:63], a[16:19], v81, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1544) 
s_add_u32 s72, 57344, s23 // (op 10216) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[64:67], a[20:23], v81, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1468) 
s_mov_b32 m0, s72 // (op 10216) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[94:97], a[24:27], v81, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1392) 
s_mov_b32 s73, 448 // (op 10216) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[98:101], a[28:31], v81, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1316) 
s_waitcnt vmcnt(14)
s_barrier  // (op 11208) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[60:63], a[32:35], v82, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2546) 
ds_read_b128 v[36:39], v4 // (op 65) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[64:67], a[36:39], v82, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2470) 
ds_read_b128 v[40:43], v4 offset:1024 // (op 414) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[94:97], a[40:43], v82, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2394) 
ds_read_b128 v[44:47], v4 offset:2048 // (op 640) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[98:101], a[44:47], v82, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2318) 
buffer_load_dwordx4 v20, s[48:51], s73 offen offset: 0 lds // (op 10216) Load value direct to lds

v_add_co_u32 v20, s[74:75], v20, v16 // (op 10216) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[102:105], v[60:63], a[48:51], v83, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3548) 
v_addc_co_u32 v21, s[74:75], 0, v21, s[74:75] // (op 10216) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[102:105], v[64:67], a[52:55], v83, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3472) 
s_add_u32 m0, m0, 4096 // (op 10216) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[102:105], v[94:97], a[56:59], v83, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3396) 
ds_read_b128 v[48:51], v4 offset:3072 // (op 866) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[102:105], v[98:101], a[60:63], v83, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3320) 
buffer_load_dwordx4 v20, s[48:51], s73 offen offset: 0 lds // (op 10216) Load value direct to lds









s_waitcnt vmcnt(14)
s_barrier  // (op 11191) 



ds_read_b128 v[52:55], v3 offset:8192 // (op 60) Load local data 
v_add_co_u32 v18, vcc, 256, v18 // (op 10311) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 10311) most significant half
ds_read_b128 v[56:59], v3 offset:9216 // (op 1309) Load local data 



ds_read_b128 v[60:63], v3 offset:10240 // (op 2311) Load local data 



ds_read_b128 v[64:67], v3 offset:11264 // (op 3313) Load local data 
s_ashr_i64 s[72:73], s[42:43], 7 // (op 2281) 
s_mov_b32 s74, s72 // (op 2281) convert

s_and_b32 s72, s74, 4294967292 // (op 2281) 

s_cmp_lt_u32 s25, s72 // (op 2281) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoop_2281 // (op 2281) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoop_2281 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoop_2281:















s_barrier  // (op 11308) 
s_mul_hi_i32 s47, s16, s10 // (op 10259) 
v_lshrrev_b32 v14, 2, v1 // (op 10259) 
s_add_i32 s0, s47, s16 // (op 10259) 

v_and_b32 v15, -4, v1 // (op 10259) 
s_ashr_i32 s1, s0, 31 // (op 10259) 
v_sub_u32 v16, v1, v15 // (op 10259) 

s_cmp_eq_i32 s10, 0 // (op 10259) 
s_mov_b32 s47, scc // (op 10259) 
v_lshlrev_b32 v15, 5, v16 // (op 10259) 

s_cmp_eq_i32 s47, 0 // (op 10259) 
s_cselect_b32 s18, 0, -1 // (op 10259) 

s_and_b32 s19, s13, 31 // (op 10259) 
s_lshl_b32 s47, 1, s19 // (op 10259) 
s_add_i32 s22, s47, s18 // (op 10259) 


s_and_b32 s18, s1, s22 // (op 10259) 


s_add_i32 s1, s0, s18 // (op 10259) 


s_ashr_i32 s0, s1, s19 // (op 10259) 


s_lshl_b32 s1, s13, 24 // (op 10259) 
s_ashr_i32 s47, s1, 31 // (op 10259) 

s_xor_b32 s1, s0, s47 // (op 10259) 

s_sub_i32 s0, s1, s47 // (op 10259) 


s_mul_hi_i32 s1, s0, s14 // (op 10259) 
s_add_i32 s47, s1, s0 // (op 10259) 

s_ashr_i32 s1, s47, 31 // (op 10259) 
s_cmp_eq_i32 s14, 0 // (op 10259) 
s_mov_b32 s18, scc // (op 10259) 
s_cmp_eq_i32 s18, 0 // (op 10259) 
s_cselect_b32 s19, 0, -1 // (op 10259) 

s_and_b32 s18, s15, 31 // (op 10259) 
s_lshl_b32 s22, 1, s18 // (op 10259) 
s_add_i32 s23, s22, s19 // (op 10259) 


s_and_b32 s19, s1, s23 // (op 10259) 


s_add_i32 s1, s47, s19 // (op 10259) 


s_ashr_i32 s19, s1, s18 // (op 10259) 


s_lshl_b32 s1, s15, 24 // (op 10259) 
s_ashr_i32 s18, s1, 31 // (op 10259) 

s_xor_b32 s1, s19, s18 // (op 10259) 

s_sub_i32 s19, s1, s18 // (op 10259) 


s_mul_i32 s1, s0, s11 // (op 10259) 

s_sub_u32 s0, s16, s1 // (op 10259) 

s_mul_i32 s1, s19, s11 // (op 10259) 
s_add_u32 s1, s1, s0 // (op 10259) 


s_mul_hi_i32 s0, s16, s9 // (op 10259) 
s_add_i32 s47, s0, s16 // (op 10259) 

s_ashr_i32 s0, s47, 31 // (op 10259) 
s_cmp_eq_i32 s9, 0 // (op 10259) 
s_mov_b32 s18, scc // (op 10259) 
s_cmp_eq_i32 s18, 0 // (op 10259) 
s_cselect_b32 s19, 0, -1 // (op 10259) 

s_and_b32 s18, s45, 31 // (op 10259) 
s_lshl_b32 s22, 1, s18 // (op 10259) 
s_add_i32 s23, s22, s19 // (op 10259) 


s_and_b32 s19, s0, s23 // (op 10259) 


s_add_i32 s0, s47, s19 // (op 10259) 


s_ashr_i32 s19, s0, s18 // (op 10259) 


s_lshl_b32 s0, s45, 24 // (op 10259) 
s_ashr_i32 s18, s0, 31 // (op 10259) 

s_xor_b32 s0, s19, s18 // (op 10259) 

s_sub_i32 s19, s0, s18 // (op 10259) 


s_add_u32 s22, s6, 128 // (op 10259) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10259) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10259) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10259) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10259) 

s_mul_i32 s0, s11, s22 // (op 10259) 

s_mul_i32 s18, s8, s0 // (op 10259) 

s_mul_i32 s0, s19, s18 // (op 10259) 

s_sub_u32 s18, s16, s0 // (op 10259) 

s_mov_b32 s0, s18 // (op 10259) 
s_mul_hi_i32 s47, s0, s12 // (op 10259) 
s_add_i32 s22, s47, s0 // (op 10259) 


s_ashr_i32 s0, s22, 31 // (op 10259) 
s_cmp_eq_i32 s12, 0 // (op 10259) 
s_mov_b32 s47, scc // (op 10259) 
s_cmp_eq_i32 s47, 0 // (op 10259) 
s_cselect_b32 s23, 0, -1 // (op 10259) 

s_and_b32 s47, s44, 31 // (op 10259) 
s_lshl_b32 s24, 1, s47 // (op 10259) 
s_add_i32 s25, s24, s23 // (op 10259) 


s_and_b32 s23, s0, s25 // (op 10259) 


s_add_i32 s0, s22, s23 // (op 10259) 


s_ashr_i32 s22, s0, s47 // (op 10259) 


s_lshl_b32 s0, s44, 24 // (op 10259) 
s_ashr_i32 s47, s0, 31 // (op 10259) 

s_xor_b32 s0, s22, s47 // (op 10259) 

s_sub_i32 s22, s0, s47 // (op 10259) 


s_mul_i32 s0, s46, s11 // (op 10259) 
s_add_u32 s24, s20, 128 // (op 10259) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 10259) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 10259) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 10259) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 10259) 

s_sub_i32 s23, s24, s0 // (op 10259) 

s_mul_i32 s47, s22, s23 // (op 10259) 


s_sub_u32 s22, s18, s47 // (op 10259) 


s_add_u32 s18, s22, s0 // (op 10259) 


s_cmp_eq_i32 s19, 0 // (op 10259) 
s_mov_b32 s0, scc // (op 10259) 

s_cmp_eq_i32 s0, 0 // (op 10259) 
s_cselect_b32 s19, s18, s1 // (op 10259) 



v_lshl_add_u32 v16, s19, 7, v14 // (op 10259) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 10259) 
s_lshr_b32 s47, s0, 2 // (op 10259) 

v_lshl_add_u32 v14, s47, 9, v15 // (op 10259) 


v_mov_b32 v27, 0 // (op 10259) convert
v_mov_b32 v26, v14 // (op 10259) convert

v_mul_lo_u32 v15, v16, s27 // (op 10259) most significant: low of low * high
v_mul_hi_u32 v17, v16, s26 // (op 10259) most significant: high of low * low
v_mul_lo_u32 v14, v16, s26 // (op 10259) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 10259) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v26 // (op 10259) least significant half
v_addc_co_u32 v15, s[0:1], v15, v27, s[0:1] // (op 10259) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 10259) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 10259) 

s_mul_i32 s1, s26, s21 // (op 10261) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 10261) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 10261) least significant
s_mul_hi_u32 s18, s26, s20 // (op 10261) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10261) most significant: sum
s_add_u32 s1, s1, s18 // (op 10261) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10261) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10261) 

s_mov_b32 s48, s28 // (op 10261) reg expression
s_mov_b32 s49, s29 // (op 10261) reg expression
s_mov_b32 s50, s0 // (op 10261) reg expression
s_mov_b32 s51, 131072 // (op 10261) call()

s_mul_hi_i32 s47, s16, s10 // (op 10341) 
v_lshrrev_b32 v16, 2, v1 // (op 10341) 
s_add_i32 s0, s47, s16 // (op 10341) 

v_and_b32 v17, -4, v1 // (op 10341) 
s_ashr_i32 s1, s0, 31 // (op 10341) 
v_sub_u32 v18, v1, v17 // (op 10341) 

s_cmp_eq_i32 s10, 0 // (op 10341) 
s_mov_b32 s47, scc // (op 10341) 
v_lshlrev_b32 v17, 5, v18 // (op 10341) 

s_cmp_eq_i32 s47, 0 // (op 10341) 
s_cselect_b32 s18, 0, -1 // (op 10341) 

s_and_b32 s19, s13, 31 // (op 10341) 
s_lshl_b32 s47, 1, s19 // (op 10341) 
s_add_i32 s22, s47, s18 // (op 10341) 


s_and_b32 s18, s1, s22 // (op 10341) 


s_add_i32 s1, s0, s18 // (op 10341) 


s_ashr_i32 s0, s1, s19 // (op 10341) 


s_lshl_b32 s1, s13, 24 // (op 10341) 
s_ashr_i32 s47, s1, 31 // (op 10341) 

s_xor_b32 s1, s0, s47 // (op 10341) 

s_sub_i32 s0, s1, s47 // (op 10341) 


s_mul_hi_i32 s1, s0, s14 // (op 10341) 
s_add_i32 s47, s1, s0 // (op 10341) 

s_ashr_i32 s1, s47, 31 // (op 10341) 
s_cmp_eq_i32 s14, 0 // (op 10341) 
s_mov_b32 s18, scc // (op 10341) 
s_cmp_eq_i32 s18, 0 // (op 10341) 
s_cselect_b32 s19, 0, -1 // (op 10341) 

s_and_b32 s18, s15, 31 // (op 10341) 
s_lshl_b32 s22, 1, s18 // (op 10341) 
s_add_i32 s23, s22, s19 // (op 10341) 


s_and_b32 s19, s1, s23 // (op 10341) 


s_add_i32 s1, s47, s19 // (op 10341) 


s_ashr_i32 s19, s1, s18 // (op 10341) 


s_lshl_b32 s1, s15, 24 // (op 10341) 
s_ashr_i32 s18, s1, 31 // (op 10341) 

s_xor_b32 s1, s19, s18 // (op 10341) 

s_sub_i32 s19, s1, s18 // (op 10341) 


s_add_u32 s22, s6, 128 // (op 10341) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10341) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10341) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10341) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10341) 

s_mul_i32 s1, s19, s22 // (op 10341) 

s_sub_i32 s47, s0, s1 // (op 10341) 


s_mul_hi_i32 s0, s16, s9 // (op 10341) 
s_add_i32 s1, s0, s16 // (op 10341) 

s_ashr_i32 s0, s1, 31 // (op 10341) 
s_cmp_eq_i32 s9, 0 // (op 10341) 
s_mov_b32 s18, scc // (op 10341) 
s_cmp_eq_i32 s18, 0 // (op 10341) 
s_cselect_b32 s19, 0, -1 // (op 10341) 

s_and_b32 s18, s45, 31 // (op 10341) 
s_lshl_b32 s24, 1, s18 // (op 10341) 
s_add_i32 s25, s24, s19 // (op 10341) 


s_and_b32 s19, s0, s25 // (op 10341) 


s_add_i32 s0, s1, s19 // (op 10341) 


s_ashr_i32 s1, s0, s18 // (op 10341) 


s_lshl_b32 s0, s45, 24 // (op 10341) 
s_ashr_i32 s18, s0, 31 // (op 10341) 

s_xor_b32 s0, s1, s18 // (op 10341) 

s_sub_i32 s1, s0, s18 // (op 10341) 


s_mul_i32 s0, s11, s22 // (op 10341) 

s_mul_i32 s18, s8, s0 // (op 10341) 

s_mul_i32 s0, s1, s18 // (op 10341) 

s_sub_u32 s18, s16, s0 // (op 10341) 

s_mul_hi_i32 s0, s18, s12 // (op 10341) 
s_add_i32 s19, s0, s18 // (op 10341) 


s_ashr_i32 s0, s19, 31 // (op 10341) 
s_cmp_eq_i32 s12, 0 // (op 10341) 
s_mov_b32 s18, scc // (op 10341) 
s_cmp_eq_i32 s18, 0 // (op 10341) 
s_cselect_b32 s22, 0, -1 // (op 10341) 

s_and_b32 s18, s44, 31 // (op 10341) 
s_lshl_b32 s23, 1, s18 // (op 10341) 
s_add_i32 s24, s23, s22 // (op 10341) 


s_and_b32 s25, s0, s24 // (op 10341) 


s_add_i32 s0, s19, s25 // (op 10341) 


s_ashr_i32 s19, s0, s18 // (op 10341) 


s_lshl_b32 s0, s44, 24 // (op 10341) 
s_ashr_i32 s18, s0, 31 // (op 10341) 

s_xor_b32 s0, s19, s18 // (op 10341) 

s_sub_i32 s19, s0, s18 // (op 10341) 


s_cmp_eq_i32 s1, 0 // (op 10341) 
s_mov_b32 s0, scc // (op 10341) 

s_cmp_eq_i32 s0, 0 // (op 10341) 
s_cselect_b32 s1, s19, s47 // (op 10341) 



v_lshl_add_u32 v18, s1, 7, v16 // (op 10341) 


v_mul_lo_u32 v27, v18, s31 // (op 10341) most significant: low of low * high
v_mul_hi_u32 v16, v18, s30 // (op 10341) most significant: high of low * low
v_mul_lo_u32 v26, v18, s30 // (op 10341) least significant: low of low * low
v_add_u32 v27, v27, v16 // (op 10341) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 10341) 
s_lshr_b32 s47, s0, 2 // (op 10341) 

v_lshl_add_u32 v16, s47, 9, v17 // (op 10341) 


v_mov_b32 v25, 0 // (op 10341) convert
v_mov_b32 v24, v16 // (op 10341) convert

v_add_co_u32 v16, s[0:1], v24, v26 // (op 10341) least significant half
v_addc_co_u32 v17, s[0:1], v25, v27, s[0:1] // (op 10341) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 10341) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 10341) 

s_mul_i32 s1, s30, s7 // (op 10343) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 10343) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 10343) least significant
s_mul_hi_u32 s18, s30, s6 // (op 10343) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10343) most significant: sum
s_add_u32 s1, s1, s18 // (op 10343) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10343) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10343) 

s_mov_b32 s60, s32 // (op 10343) reg expression
s_mov_b32 s61, s33 // (op 10343) reg expression
s_mov_b32 s62, s0 // (op 10343) reg expression
s_mov_b32 s63, 131072 // (op 10343) call()

v_and_b32 v16, -4, v1 // (op 10439) 
v_sub_u32 v17, v1, v16 // (op 10439) 

v_lshlrev_b32 v16, 5, v17 // (op 10439) 

v_lshrrev_b32 v17, 2, v1 // (op 10439) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 10439) 


v_lshlrev_b32 v16, 2, v20 // (op 10439) 

v_lshrrev_b32 v17, 3, v16 // (op 10439) 

s_nop 0
v_readfirstlane_b32 s47, v17 // (op 10439) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 10496) 
v_sub_u32 v17, v1, v16 // (op 10496) 

v_lshlrev_b32 v16, 5, v17 // (op 10496) 

v_lshrrev_b32 v17, 2, v1 // (op 10496) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 10496) 


v_lshlrev_b32 v16, 2, v20 // (op 10496) 

v_lshrrev_b32 v17, 3, v16 // (op 10496) 

s_nop 0
v_readfirstlane_b32 s0, v17 // (op 10496) Wait state hazard: VALU Write Readlane Hazard

s_mul_hi_i32 s1, s16, s10 // (op 10937) 
v_and_b32 v16, 63, v1 // (op 10937) 
s_add_i32 s18, s1, s16 // (op 10937) 

v_lshrrev_b32 v17, 4, v16 // (op 10937) 
s_ashr_i32 s1, s18, 31 // (op 10937) 
v_and_b32 v20, 3, v17 // (op 10937) 

s_cmp_eq_i32 s10, 0 // (op 10937) 
s_mov_b32 s19, scc // (op 10937) 
v_and_b32 v17, 15, v1 // (op 10937) 
s_cmp_eq_i32 s19, 0 // (op 10937) 
s_cselect_b32 s22, 0, -1 // (op 10937) 

v_lshl_add_u32 v21, v20, 4, v17 // (op 10937) 


s_and_b32 s19, s13, 31 // (op 10937) 
v_lshrrev_b32 v17, 7, v1 // (op 10937) 
s_lshl_b32 s23, 1, s19 // (op 10937) 
v_lshl_add_u32 v20, v17, 6, v21 // (op 10937) 


s_add_i32 s24, s23, s22 // (op 10937) 


v_lshrrev_b32 v17, 6, v16 // (op 10937) 

s_and_b32 s25, s1, s24 // (op 10937) 


v_lshlrev_b32 v16, 2, v17 // (op 10937) 

s_add_i32 s1, s18, s25 // (op 10937) 


s_ashr_i32 s18, s1, s19 // (op 10937) 


s_lshl_b32 s1, s13, 24 // (op 10937) 
s_ashr_i32 s19, s1, 31 // (op 10937) 

s_xor_b32 s1, s18, s19 // (op 10937) 

s_sub_i32 s18, s1, s19 // (op 10937) 


s_mul_hi_i32 s1, s18, s14 // (op 10937) 
s_add_i32 s19, s1, s18 // (op 10937) 

s_ashr_i32 s1, s19, 31 // (op 10937) 
s_cmp_eq_i32 s14, 0 // (op 10937) 
s_mov_b32 s22, scc // (op 10937) 
s_cmp_eq_i32 s22, 0 // (op 10937) 
s_cselect_b32 s23, 0, -1 // (op 10937) 

s_and_b32 s22, s15, 31 // (op 10937) 
s_lshl_b32 s24, 1, s22 // (op 10937) 
s_add_i32 s25, s24, s23 // (op 10937) 


s_and_b32 s23, s1, s25 // (op 10937) 


s_add_i32 s1, s19, s23 // (op 10937) 


s_ashr_i32 s19, s1, s22 // (op 10937) 


s_lshl_b32 s1, s15, 24 // (op 10937) 
s_ashr_i32 s22, s1, 31 // (op 10937) 

s_xor_b32 s1, s19, s22 // (op 10937) 

s_sub_i32 s19, s1, s22 // (op 10937) 


s_mul_i32 s1, s18, s11 // (op 10937) 

s_sub_u32 s18, s16, s1 // (op 10937) 

s_mul_i32 s1, s19, s11 // (op 10937) 
s_add_u32 s1, s1, s18 // (op 10937) 


s_mul_hi_i32 s18, s16, s9 // (op 10937) 
s_add_i32 s19, s18, s16 // (op 10937) 

s_ashr_i32 s18, s19, 31 // (op 10937) 
s_cmp_eq_i32 s9, 0 // (op 10937) 
s_mov_b32 s22, scc // (op 10937) 
s_cmp_eq_i32 s22, 0 // (op 10937) 
s_cselect_b32 s23, 0, -1 // (op 10937) 

s_and_b32 s22, s45, 31 // (op 10937) 
s_lshl_b32 s24, 1, s22 // (op 10937) 
s_add_i32 s25, s24, s23 // (op 10937) 


s_and_b32 s23, s18, s25 // (op 10937) 


s_add_i32 s18, s19, s23 // (op 10937) 


s_ashr_i32 s19, s18, s22 // (op 10937) 


s_lshl_b32 s18, s45, 24 // (op 10937) 
s_ashr_i32 s22, s18, 31 // (op 10937) 

s_xor_b32 s18, s19, s22 // (op 10937) 

s_sub_i32 s19, s18, s22 // (op 10937) 


s_add_u32 s22, s6, 128 // (op 10937) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10937) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10937) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10937) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10937) 

s_mul_i32 s18, s11, s22 // (op 10937) 

s_mul_i32 s22, s8, s18 // (op 10937) 

s_mul_i32 s18, s19, s22 // (op 10937) 

s_sub_u32 s22, s16, s18 // (op 10937) 

s_mov_b32 s18, s22 // (op 10937) 
s_mul_hi_i32 s23, s18, s12 // (op 10937) 
s_add_i32 s24, s23, s18 // (op 10937) 


s_ashr_i32 s18, s24, 31 // (op 10937) 
s_cmp_eq_i32 s12, 0 // (op 10937) 
s_mov_b32 s23, scc // (op 10937) 
s_cmp_eq_i32 s23, 0 // (op 10937) 
s_cselect_b32 s25, 0, -1 // (op 10937) 

s_and_b32 s23, s44, 31 // (op 10937) 
s_lshl_b32 s64, 1, s23 // (op 10937) 
s_add_i32 s65, s64, s25 // (op 10937) 


s_and_b32 s25, s18, s65 // (op 10937) 


s_add_i32 s18, s24, s25 // (op 10937) 


s_ashr_i32 s24, s18, s23 // (op 10937) 


s_lshl_b32 s18, s44, 24 // (op 10937) 
s_ashr_i32 s23, s18, 31 // (op 10937) 

s_xor_b32 s18, s24, s23 // (op 10937) 

s_sub_i32 s24, s18, s23 // (op 10937) 


s_mul_i32 s18, s46, s11 // (op 10937) 
s_add_u32 s64, s20, 128 // (op 10937) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 10937) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 10937) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 10937) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 7 // (op 10937) 

s_sub_i32 s23, s64, s18 // (op 10937) 

s_mul_i32 s25, s24, s23 // (op 10937) 


s_sub_u32 s23, s22, s25 // (op 10937) 


s_add_u32 s22, s23, s18 // (op 10937) 


s_cmp_eq_i32 s19, 0 // (op 10937) 
s_mov_b32 s18, scc // (op 10937) 

s_cmp_eq_i32 s18, 0 // (op 10937) 
s_cselect_b32 s19, s22, s1 // (op 10937) 



v_lshl_add_u32 v17, s19, 7, v20 // (op 10937) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 10937) 
s_lshr_b32 s1, s18, 2 // (op 10937) 

v_lshl_add_u32 v20, s1, 4, v16 // (op 10937) 


v_mul_lo_u32 v16, v17, s34 // (op 10937) 
v_add_u32 v16, v16, v20 // (op 10937) 


s_mul_i32 s19, s34, s21 // (op 10939) most significant: low of low * high
s_mul_i32 s1, s35, s20 // (op 10939) most significant: low of high * low
s_mul_i32 s18, s34, s20 // (op 10939) least significant
s_mul_hi_u32 s22, s34, s20 // (op 10939) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 10939) most significant: sum
s_add_u32 s19, s19, s22 // (op 10939) most significant: sum


s_mov_b32 s64, s36 // (op 10939) reg expression
s_mov_b32 s65, s37 // (op 10939) reg expression
s_mov_b32 s66, s18 // (op 10939) reg expression
s_mov_b32 s67, 131072 // (op 10939) call()

s_mul_hi_i32 s1, s16, s10 // (op 10964) 
v_and_b32 v17, 63, v1 // (op 10964) 
s_add_i32 s18, s1, s16 // (op 10964) 

v_lshrrev_b32 v20, 4, v17 // (op 10964) 
s_ashr_i32 s1, s18, 31 // (op 10964) 
v_and_b32 v21, 3, v20 // (op 10964) 

s_cmp_eq_i32 s10, 0 // (op 10964) 
s_mov_b32 s19, scc // (op 10964) 
v_and_b32 v20, 15, v1 // (op 10964) 
s_cmp_eq_i32 s19, 0 // (op 10964) 
s_cselect_b32 s22, 0, -1 // (op 10964) 

v_lshl_add_u32 v22, v21, 4, v20 // (op 10964) 


s_and_b32 s19, s13, 31 // (op 10964) 
v_lshrrev_b32 v20, 6, v1 // (op 10964) 
s_lshl_b32 s23, 1, s19 // (op 10964) 
v_and_b32 v21, 1, v20 // (op 10964) 

s_add_i32 s24, s23, s22 // (op 10964) 


v_lshl_add_u32 v20, v21, 6, v22 // (op 10964) 


s_and_b32 s25, s1, s24 // (op 10964) 


v_lshrrev_b32 v21, 6, v17 // (op 10964) 

s_add_i32 s1, s18, s25 // (op 10964) 


v_lshlrev_b32 v17, 2, v21 // (op 10964) 

s_ashr_i32 s18, s1, s19 // (op 10964) 


s_lshl_b32 s1, s13, 24 // (op 10964) 
s_ashr_i32 s19, s1, 31 // (op 10964) 

s_xor_b32 s1, s18, s19 // (op 10964) 

s_sub_i32 s18, s1, s19 // (op 10964) 


s_mul_hi_i32 s1, s18, s14 // (op 10964) 
s_add_i32 s19, s1, s18 // (op 10964) 

s_ashr_i32 s1, s19, 31 // (op 10964) 
s_cmp_eq_i32 s14, 0 // (op 10964) 
s_mov_b32 s22, scc // (op 10964) 
s_cmp_eq_i32 s22, 0 // (op 10964) 
s_cselect_b32 s23, 0, -1 // (op 10964) 

s_and_b32 s22, s15, 31 // (op 10964) 
s_lshl_b32 s24, 1, s22 // (op 10964) 
s_add_i32 s25, s24, s23 // (op 10964) 


s_and_b32 s23, s1, s25 // (op 10964) 


s_add_i32 s1, s19, s23 // (op 10964) 


s_ashr_i32 s19, s1, s22 // (op 10964) 


s_lshl_b32 s1, s15, 24 // (op 10964) 
s_ashr_i32 s22, s1, 31 // (op 10964) 

s_xor_b32 s1, s19, s22 // (op 10964) 

s_sub_i32 s19, s1, s22 // (op 10964) 


s_add_u32 s22, s6, 128 // (op 10964) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10964) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10964) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10964) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10964) 

s_mul_i32 s1, s19, s22 // (op 10964) 

s_sub_i32 s19, s18, s1 // (op 10964) 


s_mul_hi_i32 s1, s16, s9 // (op 10964) 
s_add_i32 s18, s1, s16 // (op 10964) 

s_ashr_i32 s1, s18, 31 // (op 10964) 
s_cmp_eq_i32 s9, 0 // (op 10964) 
s_mov_b32 s24, scc // (op 10964) 
s_cmp_eq_i32 s24, 0 // (op 10964) 
s_cselect_b32 s25, 0, -1 // (op 10964) 

s_and_b32 s24, s45, 31 // (op 10964) 
s_lshl_b32 s68, 1, s24 // (op 10964) 
s_add_i32 s69, s68, s25 // (op 10964) 


s_and_b32 s25, s1, s69 // (op 10964) 


s_add_i32 s1, s18, s25 // (op 10964) 


s_ashr_i32 s18, s1, s24 // (op 10964) 


s_lshl_b32 s1, s45, 24 // (op 10964) 
s_ashr_i32 s24, s1, 31 // (op 10964) 

s_xor_b32 s1, s18, s24 // (op 10964) 

s_sub_i32 s18, s1, s24 // (op 10964) 


s_mul_i32 s1, s11, s22 // (op 10964) 

s_mul_i32 s22, s8, s1 // (op 10964) 

s_mul_i32 s1, s18, s22 // (op 10964) 

s_sub_u32 s22, s16, s1 // (op 10964) 

s_mul_hi_i32 s1, s22, s12 // (op 10964) 
s_add_i32 s23, s1, s22 // (op 10964) 


s_ashr_i32 s1, s23, 31 // (op 10964) 
s_cmp_eq_i32 s12, 0 // (op 10964) 
s_mov_b32 s22, scc // (op 10964) 
s_cmp_eq_i32 s22, 0 // (op 10964) 
s_cselect_b32 s24, 0, -1 // (op 10964) 

s_and_b32 s22, s44, 31 // (op 10964) 
s_lshl_b32 s25, 1, s22 // (op 10964) 
s_add_i32 s68, s25, s24 // (op 10964) 


s_and_b32 s24, s1, s68 // (op 10964) 


s_add_i32 s1, s23, s24 // (op 10964) 


s_ashr_i32 s23, s1, s22 // (op 10964) 


s_lshl_b32 s1, s44, 24 // (op 10964) 
s_ashr_i32 s22, s1, 31 // (op 10964) 

s_xor_b32 s1, s23, s22 // (op 10964) 

s_sub_i32 s23, s1, s22 // (op 10964) 


s_cmp_eq_i32 s18, 0 // (op 10964) 
s_mov_b32 s1, scc // (op 10964) 

s_cmp_eq_i32 s1, 0 // (op 10964) 
s_cselect_b32 s18, s23, s19 // (op 10964) 



v_lshl_add_u32 v21, s18, 7, v20 // (op 10964) 


v_mul_lo_u32 v20, v21, s38 // (op 10964) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 10964) 
s_lshr_b32 s1, s18, 2 // (op 10964) 

v_lshl_add_u32 v21, s1, 4, v17 // (op 10964) 


v_add_u32 v17, v21, v20 // (op 10964) 


s_mul_i32 s19, s38, s7 // (op 10966) most significant: low of low * high
s_mul_i32 s1, s39, s6 // (op 10966) most significant: low of high * low
s_mul_i32 s18, s38, s6 // (op 10966) least significant
s_mul_hi_u32 s22, s38, s6 // (op 10966) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 10966) most significant: sum
s_add_u32 s19, s19, s22 // (op 10966) most significant: sum


s_mov_b32 s68, s40 // (op 10966) reg expression
s_mov_b32 s69, s41 // (op 10966) reg expression
s_mov_b32 s70, s18 // (op 10966) reg expression
s_mov_b32 s71, 131072 // (op 10966) call()

s_ashr_i64 s[18:19], s[42:43], 7 // (op 3289) 
s_and_b32 s1, s18, 4294967292 // (op 3289) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 3288) 
s_mov_b32 s22, s18 // (op 3288) convert

s_cmp_lt_u32 s1, s22 // (op 3288) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_3288 // (op 3288) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_3288 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoopTail_3288:
s_barrier  // (op 11188) 
s_add_u32 s1, s1, 1 // (op 3290) 

s_lshl_b64 s[18:19], s[26:27], 6 // (op 10184) 
s_lshl_b64 s[22:23], s[18:19], 2 // (op 10184) 

v_lshrrev_b64 v[20:21], 3, s[22:23] // (op 10184) 


v_mov_b64 v[22:23], v[14:15] // (op 10184) 
v_mov_b64 v[24:25], v[22:23] // (op 10184) 
s_add_u32 s18, 8192, s47 // (op 10184) 
s_mov_b32 m0, s18 // (op 10184) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 10184) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v20 // (op 10184) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 10184) most significant half

s_add_u32 m0, m0, 4096 // (op 10184) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 10184) Load value direct to lds




v_add_co_u32 v14, s[18:19], 64, v14 // (op 10249) least significant half
v_addc_co_u32 v15, s[18:19], 0, v15, s[18:19] // (op 10249) most significant half

s_barrier  // (op 11205) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 10190) 
s_lshl_b64 s[22:23], s[18:19], 2 // (op 10190) 

v_lshrrev_b64 v[20:21], 3, s[22:23] // (op 10190) 


v_mov_b64 v[22:23], v[18:19] // (op 10190) 
v_mov_b64 v[24:25], v[22:23] // (op 10190) 
s_mov_b32 m0, s0 // (op 10190) reg expression
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 10190) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 10190) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 10190) most significant half

s_add_u32 m0, m0, 4096 // (op 10190) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 10190) Load value direct to lds



v_add_co_u32 v18, s[18:19], 64, v18 // (op 10333) least significant half
v_addc_co_u32 v19, s[18:19], 0, v19, s[18:19] // (op 10333) most significant half

s_waitcnt vmcnt(2)
s_barrier  // (op 11195) 



ds_read_b128 v[20:23], v3 offset:8192 // (op 160) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 11211) 



ds_read_b128 v[24:27], v4 // (op 162) Load local data 


v_mov_b32 v28, v16 // (op 164) 
v_mov_b32 v29, v28 // (op 164) reg expression

buffer_load_dwordx4 v[30:33], v29, s[64:67], 0 offen // (op 164) Load value


v_add_u32 v16, 4, v16 // (op 10927) 


v_mov_b32 v28, v17 // (op 166) 
v_mov_b32 v29, v28 // (op 166) reg expression

buffer_load_dwordx4 v[34:37], v29, s[68:71], 0 offen // (op 166) Load value


v_add_u32 v17, 4, v17 // (op 10954) 
s_waitcnt vmcnt(1)
v_bfe_u32 v38, v30, 0, 8 // (op 9402) 



v_bfe_u32 v39, v30, 8, 8 // (op 9402) 
v_bfe_u32 v40, v30, 16, 8 // (op 9402) 
v_bfe_u32 v41, v30, 24, 8 // (op 9402) 
v_permlane16_swap_b32 v38, v39 // (op 9402) 
ds_read_b128 v[42:45], v4 offset:1024 // (op 397) Load local data 
v_permlane16_swap_b32 v40, v41 // (op 9402) 



s_nop 1
v_permlane32_swap_b32 v38, v40 // (op 9402) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v39, v41 // (op 9402) 
ds_read_b128 v[46:49], v4 offset:2048 // (op 623) Load local data 




ds_read_b128 v[28:31], v4 offset:3072 // (op 849) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v50, v34, 0, 8 // (op 9432) 



v_bfe_u32 v51, v34, 8, 8 // (op 9432) 
v_bfe_u32 v52, v34, 16, 8 // (op 9432) 
v_bfe_u32 v53, v34, 24, 8 // (op 9432) 
v_permlane16_swap_b32 v50, v51 // (op 9432) 
s_nop 0
v_permlane16_swap_b32 v52, v53 // (op 9432) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v50, v52 // (op 9432) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v51, v53 // (op 9432) 
ds_read_b128 v[54:57], v3 offset:9216 // (op 1209) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:27], a[0:3], v38, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 167) 
ds_read_b128 v[32:35], v3 offset:10240 // (op 2211) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:23], v[42:45], a[4:7], v38, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 400) 
ds_read_b128 v[58:61], v3 offset:11264 // (op 3213) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:23], v[46:49], a[8:11], v38, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 626) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:23], v[28:31], a[12:15], v38, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 852) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:57], v[24:27], a[16:19], v39, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1213) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[54:57], v[46:49], a[24:27], v39, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1267) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[54:57], v[28:31], a[28:31], v39, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1294) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[54:57], v[42:45], a[20:23], v39, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1240) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[32:35], v[24:27], a[32:35], v40, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2215) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[32:35], v[46:49], a[40:43], v40, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2269) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[32:35], v[28:31], a[44:47], v40, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2296) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[32:35], v[42:45], a[36:39], v40, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2242) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[58:61], v[24:27], a[48:51], v41, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3217) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[58:61], v[46:49], a[56:59], v41, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3271) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[58:61], v[28:31], a[60:63], v41, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3298) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[58:61], v[42:45], a[52:55], v41, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3244) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 3288) 
s_mov_b32 s22, s18 // (op 3288) convert

s_cmp_lt_u32 s1, s22 // (op 3288) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoopTail_3288 // (op 3288) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoopTail_3288 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_3288:




















s_barrier  // (op 11326) 





















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalFalse_0_beta_114
v_accvgpr_read v4, a0 // (op 42) 
v_accvgpr_read v5, a1 // (op 42) 
v_accvgpr_read v6, a2 // (op 42) 
v_accvgpr_read v7, a3 // (op 42) 
v_mul_f32 v14, v11, v4 // (op 42) 
v_mul_f32 v15, v11, v5 // (op 42) 
v_mul_f32 v16, v11, v6 // (op 42) 
v_mul_f32 v17, v11, v7 // (op 42) 



v_mov_b32 v3, v13 // (op 48) reg expression

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 48) Store value


v_accvgpr_read v4, a4 // (op 4694) 
v_accvgpr_read v5, a5 // (op 4694) 
v_accvgpr_read v6, a6 // (op 4694) 
v_accvgpr_read v7, a7 // (op 4694) 
v_mul_f32 v14, v11, v4 // (op 4694) 
v_mul_f32 v15, v11, v5 // (op 4694) 
v_mul_f32 v16, v11, v6 // (op 4694) 
v_mul_f32 v17, v11, v7 // (op 4694) 



s_lshl_b32 s0, s4, 6 // (op 4696) 
v_add_u32 v3, v13, s0 // (op 4696) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4696) Store value


v_accvgpr_read v4, a8 // (op 4706) 
v_accvgpr_read v5, a9 // (op 4706) 
v_accvgpr_read v6, a10 // (op 4706) 
v_accvgpr_read v7, a11 // (op 4706) 
v_mul_f32 v14, v11, v4 // (op 4706) 
v_mul_f32 v15, v11, v5 // (op 4706) 
v_mul_f32 v16, v11, v6 // (op 4706) 
v_mul_f32 v17, v11, v7 // (op 4706) 



s_lshl_b32 s0, s4, 7 // (op 4708) 
v_add_u32 v3, v13, s0 // (op 4708) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4708) Store value


v_accvgpr_read v4, a12 // (op 4718) 
v_accvgpr_read v5, a13 // (op 4718) 
v_accvgpr_read v6, a14 // (op 4718) 
v_accvgpr_read v7, a15 // (op 4718) 
v_mul_f32 v14, v11, v4 // (op 4718) 
v_mul_f32 v15, v11, v5 // (op 4718) 
v_mul_f32 v16, v11, v6 // (op 4718) 
v_mul_f32 v17, v11, v7 // (op 4718) 



s_lshl_b32 s0, s4, 6 // (op 4720) 
s_mul_i32 s1, 3, s0 // (op 4720) 

v_add_u32 v3, v13, s1 // (op 4720) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4720) Store value


v_accvgpr_read v4, a16 // (op 4752) 
v_accvgpr_read v5, a17 // (op 4752) 
v_accvgpr_read v6, a18 // (op 4752) 
v_accvgpr_read v7, a19 // (op 4752) 
v_mul_f32 v14, v11, v4 // (op 4752) 
v_mul_f32 v15, v11, v5 // (op 4752) 
v_mul_f32 v16, v11, v6 // (op 4752) 
v_mul_f32 v17, v11, v7 // (op 4752) 



v_add_u32 v3, 64, v13 // (op 4755) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4755) Store value


v_accvgpr_read v4, a20 // (op 4765) 
v_accvgpr_read v5, a21 // (op 4765) 
v_accvgpr_read v6, a22 // (op 4765) 
v_accvgpr_read v7, a23 // (op 4765) 
v_mul_f32 v14, v11, v4 // (op 4765) 
v_mul_f32 v15, v11, v5 // (op 4765) 
v_mul_f32 v16, v11, v6 // (op 4765) 
v_mul_f32 v17, v11, v7 // (op 4765) 



s_lshl_b32 s0, s4, 6 // (op 4757) 
s_add_u32 s0, s0, 64 // (op 4757) 
v_add_u32 v3, v13, s0 // (op 4757) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4757) Store value


v_accvgpr_read v4, a24 // (op 4770) 
v_accvgpr_read v5, a25 // (op 4770) 
v_accvgpr_read v6, a26 // (op 4770) 
v_accvgpr_read v7, a27 // (op 4770) 
v_mul_f32 v14, v11, v4 // (op 4770) 
v_mul_f32 v15, v11, v5 // (op 4770) 
v_mul_f32 v16, v11, v6 // (op 4770) 
v_mul_f32 v17, v11, v7 // (op 4770) 



s_lshl_b32 s0, s4, 7 // (op 4759) 
s_add_u32 s0, s0, 64 // (op 4759) 
v_add_u32 v3, v13, s0 // (op 4759) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4759) Store value


v_accvgpr_read v4, a28 // (op 4775) 
v_accvgpr_read v5, a29 // (op 4775) 
v_accvgpr_read v6, a30 // (op 4775) 
v_accvgpr_read v7, a31 // (op 4775) 
v_mul_f32 v14, v11, v4 // (op 4775) 
v_mul_f32 v15, v11, v5 // (op 4775) 
v_mul_f32 v16, v11, v6 // (op 4775) 
v_mul_f32 v17, v11, v7 // (op 4775) 



s_lshl_b32 s0, s4, 6 // (op 4761) 
s_mul_i32 s1, 3, s0 // (op 4761) 
s_add_u32 s1, s1, 64 // (op 4761) 

v_add_u32 v3, v13, s1 // (op 4761) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4761) Store value


v_accvgpr_read v4, a32 // (op 4820) 
v_accvgpr_read v5, a33 // (op 4820) 
v_accvgpr_read v6, a34 // (op 4820) 
v_accvgpr_read v7, a35 // (op 4820) 
v_mul_f32 v14, v11, v4 // (op 4820) 
v_mul_f32 v15, v11, v5 // (op 4820) 
v_mul_f32 v16, v11, v6 // (op 4820) 
v_mul_f32 v17, v11, v7 // (op 4820) 



v_add_u32 v3, 128, v13 // (op 4823) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4823) Store value


v_accvgpr_read v4, a36 // (op 4833) 
v_accvgpr_read v5, a37 // (op 4833) 
v_accvgpr_read v6, a38 // (op 4833) 
v_accvgpr_read v7, a39 // (op 4833) 
v_mul_f32 v14, v11, v4 // (op 4833) 
v_mul_f32 v15, v11, v5 // (op 4833) 
v_mul_f32 v16, v11, v6 // (op 4833) 
v_mul_f32 v17, v11, v7 // (op 4833) 



s_lshl_b32 s0, s4, 6 // (op 4825) 
s_add_u32 s0, s0, 128 // (op 4825) 
v_add_u32 v3, v13, s0 // (op 4825) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4825) Store value


v_accvgpr_read v4, a40 // (op 4838) 
v_accvgpr_read v5, a41 // (op 4838) 
v_accvgpr_read v6, a42 // (op 4838) 
v_accvgpr_read v7, a43 // (op 4838) 
v_mul_f32 v14, v11, v4 // (op 4838) 
v_mul_f32 v15, v11, v5 // (op 4838) 
v_mul_f32 v16, v11, v6 // (op 4838) 
v_mul_f32 v17, v11, v7 // (op 4838) 



s_lshl_b32 s0, s4, 7 // (op 4827) 
s_add_u32 s0, s0, 128 // (op 4827) 
v_add_u32 v3, v13, s0 // (op 4827) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4827) Store value


v_accvgpr_read v4, a44 // (op 4843) 
v_accvgpr_read v5, a45 // (op 4843) 
v_accvgpr_read v6, a46 // (op 4843) 
v_accvgpr_read v7, a47 // (op 4843) 
v_mul_f32 v14, v11, v4 // (op 4843) 
v_mul_f32 v15, v11, v5 // (op 4843) 
v_mul_f32 v16, v11, v6 // (op 4843) 
v_mul_f32 v17, v11, v7 // (op 4843) 



s_lshl_b32 s0, s4, 6 // (op 4829) 
s_mul_i32 s1, 3, s0 // (op 4829) 
s_add_u32 s1, s1, 128 // (op 4829) 

v_add_u32 v3, v13, s1 // (op 4829) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4829) Store value


v_accvgpr_read v4, a48 // (op 4888) 
v_accvgpr_read v5, a49 // (op 4888) 
v_accvgpr_read v6, a50 // (op 4888) 
v_accvgpr_read v7, a51 // (op 4888) 
v_mul_f32 v14, v11, v4 // (op 4888) 
v_mul_f32 v15, v11, v5 // (op 4888) 
v_mul_f32 v16, v11, v6 // (op 4888) 
v_mul_f32 v17, v11, v7 // (op 4888) 



v_add_u32 v3, 192, v13 // (op 4891) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4891) Store value


v_accvgpr_read v4, a52 // (op 4901) 
v_accvgpr_read v5, a53 // (op 4901) 
v_accvgpr_read v6, a54 // (op 4901) 
v_accvgpr_read v7, a55 // (op 4901) 
v_mul_f32 v14, v11, v4 // (op 4901) 
v_mul_f32 v15, v11, v5 // (op 4901) 
v_mul_f32 v16, v11, v6 // (op 4901) 
v_mul_f32 v17, v11, v7 // (op 4901) 



s_lshl_b32 s0, s4, 6 // (op 4893) 
s_add_u32 s0, s0, 192 // (op 4893) 
v_add_u32 v3, v13, s0 // (op 4893) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4893) Store value


v_accvgpr_read v4, a56 // (op 4906) 
v_accvgpr_read v5, a57 // (op 4906) 
v_accvgpr_read v6, a58 // (op 4906) 
v_accvgpr_read v7, a59 // (op 4906) 
v_mul_f32 v14, v11, v4 // (op 4906) 
v_mul_f32 v15, v11, v5 // (op 4906) 
v_mul_f32 v16, v11, v6 // (op 4906) 
v_mul_f32 v17, v11, v7 // (op 4906) 



s_lshl_b32 s0, s4, 7 // (op 4895) 
s_add_u32 s0, s0, 192 // (op 4895) 
v_add_u32 v3, v13, s0 // (op 4895) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4895) Store value


v_accvgpr_read v4, a60 // (op 4911) 
v_accvgpr_read v5, a61 // (op 4911) 
v_accvgpr_read v6, a62 // (op 4911) 
v_accvgpr_read v7, a63 // (op 4911) 
v_mul_f32 v14, v11, v4 // (op 4911) 
v_mul_f32 v15, v11, v5 // (op 4911) 
v_mul_f32 v16, v11, v6 // (op 4911) 
v_mul_f32 v17, v11, v7 // (op 4911) 



s_lshl_b32 s0, s4, 6 // (op 4897) 
s_mul_i32 s1, 3, s0 // (op 4897) 
s_add_u32 s1, s1, 192 // (op 4897) 

v_add_u32 v3, v13, s1 // (op 4897) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 4897) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalBottom_0_beta_114
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalFalse_0_beta_114:


v_add_u32 v3, 192, v12 // (op 5255) 

buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5255) Load value

v_accvgpr_read v14, a48 // (op 5243) 
v_accvgpr_read v15, a49 // (op 5243) 
v_accvgpr_read v16, a50 // (op 5243) 
v_accvgpr_read v17, a51 // (op 5243) 
v_mul_f32 v18, v11, v14 // (op 5243) 
v_mul_f32 v19, v11, v15 // (op 5243) 
v_mul_f32 v20, v11, v16 // (op 5243) 
v_mul_f32 v21, v11, v17 // (op 5243) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5254) 
v_mul_f32 v15, v0, v5 // (op 5254) 
v_mul_f32 v16, v0, v6 // (op 5254) 
v_mul_f32 v17, v0, v7 // (op 5254) 

v_add_f32 v4, v14, v18 // (op 5244) 
v_add_f32 v5, v15, v19 // (op 5244) 
v_add_f32 v6, v16, v20 // (op 5244) 
v_add_f32 v7, v17, v21 // (op 5244) 




v_add_u32 v3, 192, v13 // (op 5246) 

buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5246) Store value




s_lshl_b32 s0, s2, 6 // (op 5259) 
s_add_u32 s0, s0, 192 // (op 5259) 
v_add_u32 v3, v12, s0 // (op 5259) 


buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5259) Load value

v_accvgpr_read v14, a52 // (op 5269) 
v_accvgpr_read v15, a53 // (op 5269) 
v_accvgpr_read v16, a54 // (op 5269) 
v_accvgpr_read v17, a55 // (op 5269) 
v_mul_f32 v18, v11, v14 // (op 5269) 
v_mul_f32 v19, v11, v15 // (op 5269) 
v_mul_f32 v20, v11, v16 // (op 5269) 
v_mul_f32 v21, v11, v17 // (op 5269) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5257) 
v_mul_f32 v15, v0, v5 // (op 5257) 
v_mul_f32 v16, v0, v6 // (op 5257) 
v_mul_f32 v17, v0, v7 // (op 5257) 

v_add_f32 v4, v14, v18 // (op 5258) 
v_add_f32 v5, v15, v19 // (op 5258) 
v_add_f32 v6, v16, v20 // (op 5258) 
v_add_f32 v7, v17, v21 // (op 5258) 




s_lshl_b32 s0, s4, 6 // (op 5248) 
s_add_u32 s0, s0, 192 // (op 5248) 
v_add_u32 v3, v13, s0 // (op 5248) 


buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5248) Store value




s_lshl_b32 s0, s2, 7 // (op 5263) 
s_add_u32 s0, s0, 192 // (op 5263) 
v_add_u32 v3, v12, s0 // (op 5263) 


buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5263) Load value

v_accvgpr_read v14, a56 // (op 5271) 
v_accvgpr_read v15, a57 // (op 5271) 
v_accvgpr_read v16, a58 // (op 5271) 
v_accvgpr_read v17, a59 // (op 5271) 
v_mul_f32 v18, v11, v14 // (op 5271) 
v_mul_f32 v19, v11, v15 // (op 5271) 
v_mul_f32 v20, v11, v16 // (op 5271) 
v_mul_f32 v21, v11, v17 // (op 5271) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5261) 
v_mul_f32 v15, v0, v5 // (op 5261) 
v_mul_f32 v16, v0, v6 // (op 5261) 
v_mul_f32 v17, v0, v7 // (op 5261) 

v_add_f32 v4, v14, v18 // (op 5262) 
v_add_f32 v5, v15, v19 // (op 5262) 
v_add_f32 v6, v16, v20 // (op 5262) 
v_add_f32 v7, v17, v21 // (op 5262) 




s_lshl_b32 s0, s4, 7 // (op 5250) 
s_add_u32 s0, s0, 192 // (op 5250) 
v_add_u32 v3, v13, s0 // (op 5250) 


buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5250) Store value




s_lshl_b32 s0, s2, 6 // (op 5267) 
s_mul_i32 s1, 3, s0 // (op 5267) 
s_add_u32 s1, s1, 192 // (op 5267) 

v_add_u32 v3, v12, s1 // (op 5267) 


buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5267) Load value

v_accvgpr_read v14, a60 // (op 5273) 
v_accvgpr_read v15, a61 // (op 5273) 
v_accvgpr_read v16, a62 // (op 5273) 
v_accvgpr_read v17, a63 // (op 5273) 
v_mul_f32 v18, v11, v14 // (op 5273) 
v_mul_f32 v19, v11, v15 // (op 5273) 
v_mul_f32 v20, v11, v16 // (op 5273) 
v_mul_f32 v21, v11, v17 // (op 5273) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5265) 
v_mul_f32 v15, v0, v5 // (op 5265) 
v_mul_f32 v16, v0, v6 // (op 5265) 
v_mul_f32 v17, v0, v7 // (op 5265) 

v_add_f32 v4, v14, v18 // (op 5266) 
v_add_f32 v5, v15, v19 // (op 5266) 
v_add_f32 v6, v16, v20 // (op 5266) 
v_add_f32 v7, v17, v21 // (op 5266) 




s_lshl_b32 s0, s4, 6 // (op 5252) 
s_mul_i32 s1, 3, s0 // (op 5252) 
s_add_u32 s1, s1, 192 // (op 5252) 

v_add_u32 v3, v13, s1 // (op 5252) 


buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5252) Store value




v_add_u32 v3, 128, v12 // (op 5176) 

s_nop 0
buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5176) Load value

v_accvgpr_read v14, a32 // (op 5164) 
v_accvgpr_read v15, a33 // (op 5164) 
v_accvgpr_read v16, a34 // (op 5164) 
v_accvgpr_read v17, a35 // (op 5164) 
v_mul_f32 v18, v11, v14 // (op 5164) 
v_mul_f32 v19, v11, v15 // (op 5164) 
v_mul_f32 v20, v11, v16 // (op 5164) 
v_mul_f32 v21, v11, v17 // (op 5164) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5175) 
v_mul_f32 v15, v0, v5 // (op 5175) 
v_mul_f32 v16, v0, v6 // (op 5175) 
v_mul_f32 v17, v0, v7 // (op 5175) 

v_add_f32 v4, v14, v18 // (op 5165) 
v_add_f32 v5, v15, v19 // (op 5165) 
v_add_f32 v6, v16, v20 // (op 5165) 
v_add_f32 v7, v17, v21 // (op 5165) 




v_add_u32 v3, 128, v13 // (op 5167) 

buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5167) Store value




s_lshl_b32 s0, s2, 6 // (op 5180) 
s_add_u32 s0, s0, 128 // (op 5180) 
v_add_u32 v3, v12, s0 // (op 5180) 


buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5180) Load value

v_accvgpr_read v14, a36 // (op 5190) 
v_accvgpr_read v15, a37 // (op 5190) 
v_accvgpr_read v16, a38 // (op 5190) 
v_accvgpr_read v17, a39 // (op 5190) 
v_mul_f32 v18, v11, v14 // (op 5190) 
v_mul_f32 v19, v11, v15 // (op 5190) 
v_mul_f32 v20, v11, v16 // (op 5190) 
v_mul_f32 v21, v11, v17 // (op 5190) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5178) 
v_mul_f32 v15, v0, v5 // (op 5178) 
v_mul_f32 v16, v0, v6 // (op 5178) 
v_mul_f32 v17, v0, v7 // (op 5178) 

v_add_f32 v4, v14, v18 // (op 5179) 
v_add_f32 v5, v15, v19 // (op 5179) 
v_add_f32 v6, v16, v20 // (op 5179) 
v_add_f32 v7, v17, v21 // (op 5179) 




s_lshl_b32 s0, s4, 6 // (op 5169) 
s_add_u32 s0, s0, 128 // (op 5169) 
v_add_u32 v3, v13, s0 // (op 5169) 


buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5169) Store value




s_lshl_b32 s0, s2, 7 // (op 5184) 
s_add_u32 s0, s0, 128 // (op 5184) 
v_add_u32 v3, v12, s0 // (op 5184) 


buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5184) Load value

v_accvgpr_read v14, a40 // (op 5192) 
v_accvgpr_read v15, a41 // (op 5192) 
v_accvgpr_read v16, a42 // (op 5192) 
v_accvgpr_read v17, a43 // (op 5192) 
v_mul_f32 v18, v11, v14 // (op 5192) 
v_mul_f32 v19, v11, v15 // (op 5192) 
v_mul_f32 v20, v11, v16 // (op 5192) 
v_mul_f32 v21, v11, v17 // (op 5192) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5182) 
v_mul_f32 v15, v0, v5 // (op 5182) 
v_mul_f32 v16, v0, v6 // (op 5182) 
v_mul_f32 v17, v0, v7 // (op 5182) 

v_add_f32 v4, v14, v18 // (op 5183) 
v_add_f32 v5, v15, v19 // (op 5183) 
v_add_f32 v6, v16, v20 // (op 5183) 
v_add_f32 v7, v17, v21 // (op 5183) 




s_lshl_b32 s0, s4, 7 // (op 5171) 
s_add_u32 s0, s0, 128 // (op 5171) 
v_add_u32 v3, v13, s0 // (op 5171) 


buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5171) Store value




s_lshl_b32 s0, s2, 6 // (op 5188) 
s_mul_i32 s1, 3, s0 // (op 5188) 
s_add_u32 s1, s1, 128 // (op 5188) 

v_add_u32 v3, v12, s1 // (op 5188) 


buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5188) Load value

v_accvgpr_read v14, a44 // (op 5194) 
v_accvgpr_read v15, a45 // (op 5194) 
v_accvgpr_read v16, a46 // (op 5194) 
v_accvgpr_read v17, a47 // (op 5194) 
v_mul_f32 v18, v11, v14 // (op 5194) 
v_mul_f32 v19, v11, v15 // (op 5194) 
v_mul_f32 v20, v11, v16 // (op 5194) 
v_mul_f32 v21, v11, v17 // (op 5194) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5186) 
v_mul_f32 v15, v0, v5 // (op 5186) 
v_mul_f32 v16, v0, v6 // (op 5186) 
v_mul_f32 v17, v0, v7 // (op 5186) 

v_add_f32 v4, v14, v18 // (op 5187) 
v_add_f32 v5, v15, v19 // (op 5187) 
v_add_f32 v6, v16, v20 // (op 5187) 
v_add_f32 v7, v17, v21 // (op 5187) 




s_lshl_b32 s0, s4, 6 // (op 5173) 
s_mul_i32 s1, 3, s0 // (op 5173) 
s_add_u32 s1, s1, 128 // (op 5173) 

v_add_u32 v3, v13, s1 // (op 5173) 


buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5173) Store value




v_add_u32 v3, 64, v12 // (op 5097) 

s_nop 0
buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5097) Load value

v_accvgpr_read v14, a16 // (op 5085) 
v_accvgpr_read v15, a17 // (op 5085) 
v_accvgpr_read v16, a18 // (op 5085) 
v_accvgpr_read v17, a19 // (op 5085) 
v_mul_f32 v18, v11, v14 // (op 5085) 
v_mul_f32 v19, v11, v15 // (op 5085) 
v_mul_f32 v20, v11, v16 // (op 5085) 
v_mul_f32 v21, v11, v17 // (op 5085) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5096) 
v_mul_f32 v15, v0, v5 // (op 5096) 
v_mul_f32 v16, v0, v6 // (op 5096) 
v_mul_f32 v17, v0, v7 // (op 5096) 

v_add_f32 v4, v14, v18 // (op 5086) 
v_add_f32 v5, v15, v19 // (op 5086) 
v_add_f32 v6, v16, v20 // (op 5086) 
v_add_f32 v7, v17, v21 // (op 5086) 




v_add_u32 v3, 64, v13 // (op 5088) 

buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5088) Store value




s_lshl_b32 s0, s2, 6 // (op 5101) 
s_add_u32 s0, s0, 64 // (op 5101) 
v_add_u32 v3, v12, s0 // (op 5101) 


buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5101) Load value

v_accvgpr_read v14, a20 // (op 5111) 
v_accvgpr_read v15, a21 // (op 5111) 
v_accvgpr_read v16, a22 // (op 5111) 
v_accvgpr_read v17, a23 // (op 5111) 
v_mul_f32 v18, v11, v14 // (op 5111) 
v_mul_f32 v19, v11, v15 // (op 5111) 
v_mul_f32 v20, v11, v16 // (op 5111) 
v_mul_f32 v21, v11, v17 // (op 5111) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5099) 
v_mul_f32 v15, v0, v5 // (op 5099) 
v_mul_f32 v16, v0, v6 // (op 5099) 
v_mul_f32 v17, v0, v7 // (op 5099) 

v_add_f32 v4, v14, v18 // (op 5100) 
v_add_f32 v5, v15, v19 // (op 5100) 
v_add_f32 v6, v16, v20 // (op 5100) 
v_add_f32 v7, v17, v21 // (op 5100) 




s_lshl_b32 s0, s4, 6 // (op 5090) 
s_add_u32 s0, s0, 64 // (op 5090) 
v_add_u32 v3, v13, s0 // (op 5090) 


buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5090) Store value




s_lshl_b32 s0, s2, 7 // (op 5105) 
s_add_u32 s0, s0, 64 // (op 5105) 
v_add_u32 v3, v12, s0 // (op 5105) 


buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5105) Load value

v_accvgpr_read v14, a24 // (op 5113) 
v_accvgpr_read v15, a25 // (op 5113) 
v_accvgpr_read v16, a26 // (op 5113) 
v_accvgpr_read v17, a27 // (op 5113) 
v_mul_f32 v18, v11, v14 // (op 5113) 
v_mul_f32 v19, v11, v15 // (op 5113) 
v_mul_f32 v20, v11, v16 // (op 5113) 
v_mul_f32 v21, v11, v17 // (op 5113) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5103) 
v_mul_f32 v15, v0, v5 // (op 5103) 
v_mul_f32 v16, v0, v6 // (op 5103) 
v_mul_f32 v17, v0, v7 // (op 5103) 

v_add_f32 v4, v14, v18 // (op 5104) 
v_add_f32 v5, v15, v19 // (op 5104) 
v_add_f32 v6, v16, v20 // (op 5104) 
v_add_f32 v7, v17, v21 // (op 5104) 




s_lshl_b32 s0, s4, 7 // (op 5092) 
s_add_u32 s0, s0, 64 // (op 5092) 
v_add_u32 v3, v13, s0 // (op 5092) 


buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5092) Store value




s_lshl_b32 s0, s2, 6 // (op 5109) 
s_mul_i32 s1, 3, s0 // (op 5109) 
s_add_u32 s1, s1, 64 // (op 5109) 

v_add_u32 v3, v12, s1 // (op 5109) 


buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5109) Load value

v_accvgpr_read v14, a28 // (op 5115) 
v_accvgpr_read v15, a29 // (op 5115) 
v_accvgpr_read v16, a30 // (op 5115) 
v_accvgpr_read v17, a31 // (op 5115) 
v_mul_f32 v18, v11, v14 // (op 5115) 
v_mul_f32 v19, v11, v15 // (op 5115) 
v_mul_f32 v20, v11, v16 // (op 5115) 
v_mul_f32 v21, v11, v17 // (op 5115) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5107) 
v_mul_f32 v15, v0, v5 // (op 5107) 
v_mul_f32 v16, v0, v6 // (op 5107) 
v_mul_f32 v17, v0, v7 // (op 5107) 

v_add_f32 v4, v14, v18 // (op 5108) 
v_add_f32 v5, v15, v19 // (op 5108) 
v_add_f32 v6, v16, v20 // (op 5108) 
v_add_f32 v7, v17, v21 // (op 5108) 




s_lshl_b32 s0, s4, 6 // (op 5094) 
s_mul_i32 s1, 3, s0 // (op 5094) 
s_add_u32 s1, s1, 64 // (op 5094) 

v_add_u32 v3, v13, s1 // (op 5094) 


buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5094) Store value




v_mov_b32 v3, v12 // (op 126) reg expression

s_nop 0
buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 126) Load value

v_accvgpr_read v14, a0 // (op 123) 
v_accvgpr_read v15, a1 // (op 123) 
v_accvgpr_read v16, a2 // (op 123) 
v_accvgpr_read v17, a3 // (op 123) 
v_mul_f32 v18, v11, v14 // (op 123) 
v_mul_f32 v19, v11, v15 // (op 123) 
v_mul_f32 v20, v11, v16 // (op 123) 
v_mul_f32 v21, v11, v17 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 127) 
v_mul_f32 v15, v0, v5 // (op 127) 
v_mul_f32 v16, v0, v6 // (op 127) 
v_mul_f32 v17, v0, v7 // (op 127) 

v_add_f32 v4, v14, v18 // (op 124) 
v_add_f32 v5, v15, v19 // (op 124) 
v_add_f32 v6, v16, v20 // (op 124) 
v_add_f32 v7, v17, v21 // (op 124) 




v_mov_b32 v3, v13 // (op 125) reg expression

buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 125) Store value




s_lshl_b32 s0, s2, 6 // (op 5011) 
v_add_u32 v3, v12, s0 // (op 5011) 


buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5011) Load value

v_accvgpr_read v14, a4 // (op 5008) 
v_accvgpr_read v15, a5 // (op 5008) 
v_accvgpr_read v16, a6 // (op 5008) 
v_accvgpr_read v17, a7 // (op 5008) 
v_mul_f32 v18, v11, v14 // (op 5008) 
v_mul_f32 v19, v11, v15 // (op 5008) 
v_mul_f32 v20, v11, v16 // (op 5008) 
v_mul_f32 v21, v11, v17 // (op 5008) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5012) 
v_mul_f32 v15, v0, v5 // (op 5012) 
v_mul_f32 v16, v0, v6 // (op 5012) 
v_mul_f32 v17, v0, v7 // (op 5012) 

v_add_f32 v4, v14, v18 // (op 5009) 
v_add_f32 v5, v15, v19 // (op 5009) 
v_add_f32 v6, v16, v20 // (op 5009) 
v_add_f32 v7, v17, v21 // (op 5009) 




s_lshl_b32 s0, s4, 6 // (op 5010) 
v_add_u32 v3, v13, s0 // (op 5010) 


buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5010) Store value




s_lshl_b32 s0, s2, 7 // (op 5023) 
v_add_u32 v3, v12, s0 // (op 5023) 


buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5023) Load value

v_accvgpr_read v14, a8 // (op 5020) 
v_accvgpr_read v15, a9 // (op 5020) 
v_accvgpr_read v16, a10 // (op 5020) 
v_accvgpr_read v17, a11 // (op 5020) 
v_mul_f32 v18, v11, v14 // (op 5020) 
v_mul_f32 v19, v11, v15 // (op 5020) 
v_mul_f32 v20, v11, v16 // (op 5020) 
v_mul_f32 v21, v11, v17 // (op 5020) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5024) 
v_mul_f32 v15, v0, v5 // (op 5024) 
v_mul_f32 v16, v0, v6 // (op 5024) 
v_mul_f32 v17, v0, v7 // (op 5024) 

v_add_f32 v4, v14, v18 // (op 5021) 
v_add_f32 v5, v15, v19 // (op 5021) 
v_add_f32 v6, v16, v20 // (op 5021) 
v_add_f32 v7, v17, v21 // (op 5021) 




s_lshl_b32 s0, s4, 7 // (op 5022) 
v_add_u32 v3, v13, s0 // (op 5022) 


buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5022) Store value




s_lshl_b32 s0, s2, 6 // (op 5035) 
s_mul_i32 s1, 3, s0 // (op 5035) 

v_add_u32 v3, v12, s1 // (op 5035) 


buffer_load_dwordx4 v[4:7], v3, s[52:55], 0 offen // (op 5035) Load value

v_accvgpr_read v14, a12 // (op 5032) 
v_accvgpr_read v15, a13 // (op 5032) 
v_accvgpr_read v16, a14 // (op 5032) 
v_accvgpr_read v17, a15 // (op 5032) 
v_mul_f32 v18, v11, v14 // (op 5032) 
v_mul_f32 v19, v11, v15 // (op 5032) 
v_mul_f32 v20, v11, v16 // (op 5032) 
v_mul_f32 v21, v11, v17 // (op 5032) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 5036) 
v_mul_f32 v15, v0, v5 // (op 5036) 
v_mul_f32 v16, v0, v6 // (op 5036) 
v_mul_f32 v17, v0, v7 // (op 5036) 

v_add_f32 v4, v14, v18 // (op 5033) 
v_add_f32 v5, v15, v19 // (op 5033) 
v_add_f32 v6, v16, v20 // (op 5033) 
v_add_f32 v7, v17, v21 // (op 5033) 




s_lshl_b32 s0, s4, 6 // (op 5034) 
s_mul_i32 s1, 3, s0 // (op 5034) 

v_add_u32 v3, v13, s1 // (op 5034) 


buffer_store_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 5034) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalBottom_0_beta_114:
























s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
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
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_.kd
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

