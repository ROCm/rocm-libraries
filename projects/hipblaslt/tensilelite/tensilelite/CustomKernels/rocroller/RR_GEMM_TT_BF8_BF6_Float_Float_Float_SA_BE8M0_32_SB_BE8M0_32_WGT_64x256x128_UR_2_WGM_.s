// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_,@function
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__exec_begin:


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
s_load_dwordx2 s[48:49], s[0:1], 168 // Load scalar value
s_load_dword s50, s[0:1], 176 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


s_mul_hi_i32 s0, s16, s11 // (op 2671) 
s_mul_hi_i32 s1, s16, s11 // (op 2912) 
v_and_b32 v0, 63, v1 // (op 2727) 
v_lshrrev_b32 v3, 4, v0 // (op 2727) 
v_and_b32 v4, 1, v3 // (op 2727) 

v_and_b32 v3, 15, v1 // (op 2727) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 2727) 


v_lshrrev_b32 v3, 7, v1 // (op 2727) 
v_lshl_add_u32 v4, v3, 5, v5 // (op 2727) 


v_lshrrev_b32 v3, 5, v0 // (op 2727) 

v_lshlrev_b32 v0, 4, v3 // (op 2727) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 2727) 


v_lshrrev_b32 v0, 7, v1 // (op 2838) 
v_and_b32 v4, 31, v1 // (op 2838) 
v_lshl_add_u32 v5, v0, 5, v4 // (op 2838) 


v_and_b32 v0, 63, v1 // (op 2838) 
v_lshrrev_b32 v4, 5, v0 // (op 2838) 

v_lshl_add_u32 v0, v5, 2, v4 // (op 2838) 


v_and_b32 v4, 63, v1 // (op 2953) 
v_lshrrev_b32 v5, 4, v4 // (op 2953) 
v_and_b32 v6, 1, v5 // (op 2953) 

v_and_b32 v5, 15, v1 // (op 2953) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 2953) 


v_lshrrev_b32 v5, 7, v1 // (op 2953) 
v_lshl_add_u32 v6, v5, 5, v7 // (op 2953) 


v_lshrrev_b32 v5, 5, v4 // (op 2953) 

v_lshlrev_b32 v4, 4, v5 // (op 2953) 

v_lshl_add_u32 v5, v6, 7, v4 // (op 2953) 


v_lshrrev_b32 v4, 7, v1 // (op 3006) 
v_and_b32 v6, 31, v1 // (op 3006) 
v_lshl_add_u32 v7, v4, 5, v6 // (op 3006) 


v_and_b32 v4, 63, v1 // (op 3006) 
v_lshrrev_b32 v6, 5, v4 // (op 3006) 

v_lshl_add_u32 v4, v7, 2, v6 // (op 3006) 


v_lshrrev_b32 v6, 7, v1 // (op 2671) 
s_add_i32 s18, s0, s16 // (op 2671) 

v_and_b32 v7, 63, v1 // (op 2671) 
s_ashr_i32 s0, s18, 31 // (op 2671) 
v_lshrrev_b32 v8, 5, v7 // (op 2671) 
s_cmp_eq_i32 s11, 0 // (op 2671) 
s_mov_b32 s19, scc // (op 2671) 
v_lshlrev_b32 v9, 2, v8 // (op 2671) 

s_cmp_eq_i32 s19, 0 // (op 2671) 
s_cselect_b32 s51, 0, -1 // (op 2671) 

v_lshl_add_u32 v8, v6, 5, v9 // (op 2671) 


s_and_b32 s19, s12, 31 // (op 2671) 
v_lshrrev_b32 v6, 2, v7 // (op 2671) 

v_and_b32 v7, 7, v6 // (op 2671) 

v_and_b32 v6, 63, v1 // (op 2785) 
v_and_b32 v9, 48, v1 // (op 2785) 
v_sub_u32 v10, v6, v9 // (op 2785) 

v_lshrrev_b32 v9, 5, v6 // (op 2785) 
v_lshl_add_u32 v11, v9, 5, v10 // (op 2785) 


v_lshrrev_b32 v9, 4, v6 // (op 2785) 

v_and_b32 v6, -2, v9 // (op 2785) 
v_sub_u32 v10, v9, v6 // (op 2785) 


v_lshlrev_b32 v6, 4, v10 // (op 2785) 

v_lshrrev_b32 v9, 6, v1 // (op 2785) 
v_and_b32 v10, 1, v9 // (op 2785) 

v_lshl_add_u32 v9, v10, 5, v6 // (op 2785) 


v_lshl_add_u32 v6, v11, 8, v9 // (op 2785) 


v_mul_lo_u32 v9, v6, 6 // (op 2785) 
v_lshrrev_b32 v10, 3, v9 // (op 2785) 

v_lshrrev_b32 v9, 4, v6 // (op 2785) 

v_lshlrev_b32 v6, 2, v9 // (op 2785) 

v_add_u32 v9, v10, v6 // (op 2785) 


v_lshrrev_b32 v6, 6, v1 // (op 2896) 
v_and_b32 v10, 1, v6 // (op 2896) 

v_and_b32 v6, 31, v1 // (op 2896) 
v_lshl_add_u32 v11, v10, 5, v6 // (op 2896) 


v_and_b32 v6, 63, v1 // (op 2896) 
v_lshrrev_b32 v10, 5, v6 // (op 2896) 

v_lshl_add_u32 v6, v10, 8, v11 // (op 2896) 


v_lshrrev_b32 v10, 7, v1 // (op 2912) 
v_and_b32 v11, 63, v1 // (op 2912) 
v_and_b32 v12, 63, v1 // (op 2982) 
v_and_b32 v13, 48, v1 // (op 2982) 
v_sub_u32 v14, v12, v13 // (op 2982) 

v_lshrrev_b32 v13, 5, v12 // (op 2982) 
v_lshl_add_u32 v15, v13, 5, v14 // (op 2982) 


v_lshrrev_b32 v13, 4, v12 // (op 2982) 

v_and_b32 v12, -2, v13 // (op 2982) 
v_sub_u32 v14, v13, v12 // (op 2982) 


v_lshlrev_b32 v12, 4, v14 // (op 2982) 

v_lshrrev_b32 v13, 6, v1 // (op 2982) 
v_and_b32 v14, 1, v13 // (op 2982) 

v_lshl_add_u32 v13, v14, 5, v12 // (op 2982) 


v_lshl_add_u32 v12, v15, 8, v13 // (op 2982) 


v_mul_lo_u32 v13, v12, 6 // (op 2982) 
v_lshrrev_b32 v14, 3, v13 // (op 2982) 

v_lshrrev_b32 v13, 4, v12 // (op 2982) 

v_lshlrev_b32 v12, 2, v13 // (op 2982) 

v_add_u32 v13, v14, v12 // (op 2982) 


v_lshrrev_b32 v12, 6, v1 // (op 3035) 
v_and_b32 v14, 1, v12 // (op 3035) 

v_and_b32 v12, 31, v1 // (op 3035) 
v_lshl_add_u32 v15, v14, 5, v12 // (op 3035) 


v_and_b32 v12, 63, v1 // (op 3035) 
v_lshrrev_b32 v14, 5, v12 // (op 3035) 

v_lshl_add_u32 v12, v14, 8, v15 // (op 3035) 


s_lshl_b32 s52, 1, s19 // (op 2671) 
v_and_b32 v14, 3, v1 // (op 2671) 
s_add_i32 s53, s52, s51 // (op 2671) 


s_and_b32 s51, s0, s53 // (op 2671) 


s_add_i32 s0, s1, s16 // (op 2912) 

s_ashr_i32 s1, s0, 31 // (op 2912) 
v_lshl_add_u32 v15, v7, 2, v14 // (op 2671) 


s_add_i32 s52, s18, s51 // (op 2671) 


v_lshrrev_b32 v7, 6, v1 // (op 2671) 
s_ashr_i32 s18, s52, s19 // (op 2671) 


v_and_b32 v14, 1, v7 // (op 2671) 

s_lshl_b32 s19, s12, 24 // (op 2671) 
v_lshl_add_u32 v7, v14, 5, v15 // (op 2671) 


s_ashr_i32 s51, s19, 31 // (op 2671) 

s_xor_b32 s19, s18, s51 // (op 2671) 

s_sub_i32 s18, s19, s51 // (op 2671) 


s_mul_hi_i32 s19, s18, s13 // (op 2671) 
s_add_i32 s51, s19, s18 // (op 2671) 

s_ashr_i32 s19, s51, 31 // (op 2671) 
s_cmp_eq_i32 s13, 0 // (op 2671) 
s_mov_b32 s52, scc // (op 2671) 
s_cmp_eq_i32 s52, 0 // (op 2671) 
s_cselect_b32 s53, 0, -1 // (op 2671) 

s_and_b32 s52, s14, 31 // (op 2671) 
v_lshrrev_b32 v14, 5, v11 // (op 2912) 
s_cmp_eq_i32 s11, 0 // (op 2912) 
s_mov_b32 s54, scc // (op 2912) 
v_lshlrev_b32 v15, 2, v14 // (op 2912) 

s_cmp_eq_i32 s54, 0 // (op 2912) 
s_cselect_b32 s55, 0, -1 // (op 2912) 

s_lshl_b32 s54, 1, s52 // (op 2671) 
v_lshl_add_u32 v14, v10, 5, v15 // (op 2912) 


v_lshrrev_b32 v10, 2, v11 // (op 2912) 

s_add_i32 s56, s54, s53 // (op 2671) 


s_and_b32 s53, s19, s56 // (op 2671) 


s_add_i32 s19, s51, s53 // (op 2671) 


s_ashr_i32 s51, s19, s52 // (op 2671) 


s_lshl_b32 s19, s14, 24 // (op 2671) 
s_ashr_i32 s52, s19, 31 // (op 2671) 

s_xor_b32 s19, s51, s52 // (op 2671) 

s_sub_i32 s51, s19, s52 // (op 2671) 


s_and_b32 s19, s12, 31 // (op 2912) 
s_lshl_b32 s52, 1, s19 // (op 2912) 
v_and_b32 v11, 7, v10 // (op 2912) 

s_add_i32 s53, s52, s55 // (op 2912) 


v_and_b32 v10, 3, v1 // (op 2912) 
s_and_b32 s52, s1, s53 // (op 2912) 


v_lshl_add_u32 v15, v11, 2, v10 // (op 2912) 


s_add_i32 s1, s0, s52 // (op 2912) 


v_lshrrev_b32 v10, 6, v1 // (op 2912) 
s_ashr_i32 s0, s1, s19 // (op 2912) 


v_and_b32 v11, 1, v10 // (op 2912) 

s_lshl_b32 s1, s12, 24 // (op 2912) 
v_lshl_add_u32 v10, v11, 5, v15 // (op 2912) 


s_ashr_i32 s19, s1, 31 // (op 2912) 

s_xor_b32 s1, s0, s19 // (op 2912) 

s_sub_i32 s0, s1, s19 // (op 2912) 


s_mul_hi_i32 s1, s0, s13 // (op 2912) 
s_add_i32 s19, s1, s0 // (op 2912) 

s_ashr_i32 s1, s19, 31 // (op 2912) 
s_cmp_eq_i32 s13, 0 // (op 2912) 
s_mov_b32 s52, scc // (op 2912) 
s_cmp_eq_i32 s52, 0 // (op 2912) 
s_cselect_b32 s53, 0, -1 // (op 2912) 

s_and_b32 s52, s14, 31 // (op 2912) 
s_lshl_b32 s54, 1, s52 // (op 2912) 
s_add_i32 s55, s54, s53 // (op 2912) 


s_and_b32 s53, s1, s55 // (op 2912) 


s_add_i32 s1, s19, s53 // (op 2912) 


s_ashr_i32 s19, s1, s52 // (op 2912) 


s_lshl_b32 s1, s14, 24 // (op 2912) 
s_ashr_i32 s52, s1, 31 // (op 2912) 

s_xor_b32 s1, s19, s52 // (op 2912) 

s_sub_i32 s19, s1, s52 // (op 2912) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 2671) 
s_sub_u32 s52, s16, s1 // (op 2671) 

s_mul_i32 s1, s51, s44 // (op 2671) 
s_add_u32 s1, s1, s52 // (op 2671) 

s_mul_hi_i32 s52, s16, s10 // (op 2671) 
s_add_i32 s53, s52, s16 // (op 2671) 

s_ashr_i32 s52, s53, 31 // (op 2671) 
s_cmp_eq_i32 s10, 0 // (op 2671) 
s_mov_b32 s54, scc // (op 2671) 
s_cmp_eq_i32 s54, 0 // (op 2671) 
s_cselect_b32 s55, 0, -1 // (op 2671) 

s_and_b32 s54, s47, 31 // (op 2671) 
s_lshl_b32 s56, 1, s54 // (op 2671) 
s_add_i32 s57, s56, s55 // (op 2671) 


s_and_b32 s55, s52, s57 // (op 2671) 


s_add_i32 s52, s53, s55 // (op 2671) 


s_ashr_i32 s53, s52, s54 // (op 2671) 


s_lshl_b32 s52, s47, 24 // (op 2671) 
s_ashr_i32 s54, s52, 31 // (op 2671) 

s_xor_b32 s52, s53, s54 // (op 2671) 

s_sub_i32 s53, s52, s54 // (op 2671) 


s_mul_i32 s52, s0, s44 // (op 2912) 
s_sub_u32 s54, s16, s52 // (op 2912) 

s_mul_i32 s52, s19, s44 // (op 2912) 
s_add_u32 s52, s52, s54 // (op 2912) 

s_mul_hi_i32 s54, s16, s10 // (op 2912) 
s_add_i32 s55, s54, s16 // (op 2912) 

s_ashr_i32 s54, s55, 31 // (op 2912) 
s_cmp_eq_i32 s10, 0 // (op 2912) 
s_mov_b32 s56, scc // (op 2912) 
s_cmp_eq_i32 s56, 0 // (op 2912) 
s_cselect_b32 s57, 0, -1 // (op 2912) 

s_and_b32 s56, s47, 31 // (op 2912) 
s_lshl_b32 s58, 1, s56 // (op 2912) 
s_add_i32 s59, s58, s57 // (op 2912) 


s_and_b32 s57, s54, s59 // (op 2912) 


s_add_i32 s54, s55, s57 // (op 2912) 


s_ashr_i32 s55, s54, s56 // (op 2912) 


s_lshl_b32 s54, s47, 24 // (op 2912) 
s_ashr_i32 s56, s54, 31 // (op 2912) 

s_xor_b32 s54, s55, s56 // (op 2912) 

s_sub_i32 s55, s54, s56 // (op 2912) 


s_add_u32 s56, s6, 256 // (op 2671) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 2671) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2671) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2671) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 2671) 

s_mul_i32 s54, s44, s56 // (op 2671) 
s_mul_i32 s58, s48, s54 // (op 2671) 

s_mul_i32 s54, s53, s58 // (op 2671) 

s_sub_u32 s58, s16, s54 // (op 2671) 

s_mov_b32 s54, s58 // (op 2671) 
s_mul_hi_i32 s59, s54, s15 // (op 2671) 
s_add_i32 s60, s59, s54 // (op 2671) 


s_ashr_i32 s54, s60, 31 // (op 2671) 
s_cmp_eq_i32 s15, 0 // (op 2671) 
s_mov_b32 s59, scc // (op 2671) 
s_cmp_eq_i32 s59, 0 // (op 2671) 
s_cselect_b32 s61, 0, -1 // (op 2671) 

s_and_b32 s59, s46, 31 // (op 2671) 
s_lshl_b32 s62, 1, s59 // (op 2671) 
s_add_i32 s63, s62, s61 // (op 2671) 


s_and_b32 s61, s54, s63 // (op 2671) 


s_add_i32 s54, s60, s61 // (op 2671) 


s_ashr_i32 s60, s54, s59 // (op 2671) 


s_lshl_b32 s54, s46, 24 // (op 2671) 
s_ashr_i32 s59, s54, 31 // (op 2671) 

s_xor_b32 s54, s60, s59 // (op 2671) 

s_sub_i32 s60, s54, s59 // (op 2671) 


s_mul_i32 s54, s45, s44 // (op 2671) 
s_add_u32 s62, s8, 64 // (op 2671) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 2671) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 2671) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 2671) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 2671) 

s_sub_i32 s59, s62, s54 // (op 2671) 

s_mul_i32 s61, s60, s59 // (op 2671) 

s_sub_u32 s59, s58, s61 // (op 2671) 


s_add_u32 s58, s59, s54 // (op 2671) 


s_cmp_eq_i32 s53, 0 // (op 2671) 
s_mov_b32 s54, scc // (op 2671) 

s_cmp_eq_i32 s54, 0 // (op 2671) 
s_cselect_b32 s53, s58, s1 // (op 2671) 


s_mul_i32 s1, s51, s56 // (op 2671) 


v_lshl_add_u32 v11, s53, 6, v8 // (op 2671) 


s_sub_i32 s51, s18, s1 // (op 2671) 


s_cmp_eq_i32 s54, 0 // (op 2671) 
s_cselect_b32 s1, s60, s51 // (op 2671) 



v_lshl_add_u32 v8, s1, 8, v7 // (op 2671) 


v_mul_lo_u32 v7, v8, s2 // (op 2671) 

v_add_lshl_u32 v8, v11, v7, 2 // (op 2671) 


s_add_u32 s56, s6, 256 // (op 2912) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 2912) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2912) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2912) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 2912) 

s_mul_i32 s1, s44, s56 // (op 2912) 
s_mul_i32 s18, s48, s1 // (op 2912) 

s_mul_i32 s1, s55, s18 // (op 2912) 

s_sub_u32 s18, s16, s1 // (op 2912) 

s_mov_b32 s1, s18 // (op 2912) 
s_mul_hi_i32 s51, s1, s15 // (op 2912) 
s_add_i32 s53, s51, s1 // (op 2912) 


s_ashr_i32 s1, s53, 31 // (op 2912) 
s_cmp_eq_i32 s15, 0 // (op 2912) 
s_mov_b32 s51, scc // (op 2912) 
s_cmp_eq_i32 s51, 0 // (op 2912) 
s_cselect_b32 s54, 0, -1 // (op 2912) 

s_and_b32 s51, s46, 31 // (op 2912) 
s_lshl_b32 s58, 1, s51 // (op 2912) 
s_add_i32 s59, s58, s54 // (op 2912) 


s_and_b32 s54, s1, s59 // (op 2912) 


s_add_i32 s1, s53, s54 // (op 2912) 


s_ashr_i32 s53, s1, s51 // (op 2912) 


s_lshl_b32 s1, s46, 24 // (op 2912) 
s_ashr_i32 s51, s1, 31 // (op 2912) 

s_xor_b32 s1, s53, s51 // (op 2912) 

s_sub_i32 s53, s1, s51 // (op 2912) 


s_mul_i32 s1, s45, s44 // (op 2912) 
s_add_u32 s58, s8, 64 // (op 2912) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 2912) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 2912) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 2912) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 2912) 

s_sub_i32 s51, s58, s1 // (op 2912) 

s_mul_i32 s54, s53, s51 // (op 2912) 

s_sub_u32 s51, s18, s54 // (op 2912) 


s_add_u32 s18, s51, s1 // (op 2912) 


s_cmp_eq_i32 s55, 0 // (op 2912) 
s_mov_b32 s1, scc // (op 2912) 

s_cmp_eq_i32 s1, 0 // (op 2912) 
s_cselect_b32 s51, s18, s52 // (op 2912) 


s_mul_i32 s18, s19, s56 // (op 2912) 


v_lshl_add_u32 v7, s51, 6, v14 // (op 2912) 


s_sub_i32 s19, s0, s18 // (op 2912) 


s_cmp_eq_i32 s1, 0 // (op 2912) 
s_cselect_b32 s0, s53, s19 // (op 2912) 



v_lshl_add_u32 v11, s0, 8, v10 // (op 2912) 


v_mul_lo_u32 v10, v11, s4 // (op 2912) 

v_add_lshl_u32 v11, v7, v10, 2 // (op 2912) 


s_mul_i32 s1, s2, s7 // (op 2673) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 2673) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 2673) least significant
s_mul_hi_u32 s19, s2, s6 // (op 2673) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2673) most significant: sum
s_add_u32 s1, s1, s19 // (op 2673) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2673) 

s_mul_i32 s1, s4, s7 // (op 2914) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 2914) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 2914) least significant
s_mul_hi_u32 s52, s4, s6 // (op 2914) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 2914) most significant: sum
s_add_u32 s1, s1, s52 // (op 2914) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 2914) 

s_mov_b32 s56, s20 // (op 2673) reg expression
s_mov_b32 s57, s21 // (op 2673) reg expression
s_mov_b32 s58, s18 // (op 2673) reg expression
s_mov_b32 s59, 131072 // (op 2673) call()

s_mov_b32 s60, s22 // (op 2914) reg expression

s_mov_b32 s61, s23 // (op 2914) reg expression
s_mov_b32 s62, s52 // (op 2914) reg expression
s_mov_b32 s63, 131072 // (op 2914) call()


v_mov_b32 v7, s49 // (op 2) Move value
v_mov_b32 v10, s50 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 63) call()
v_accvgpr_write a5, 0.00000 // (op 63) call()
v_accvgpr_write a6, 0.00000 // (op 63) call()
v_accvgpr_write a7, 0.00000 // (op 63) call()
v_accvgpr_write a8, 0.00000 // (op 63) call()
v_accvgpr_write a9, 0.00000 // (op 63) call()
v_accvgpr_write a10, 0.00000 // (op 63) call()
v_accvgpr_write a11, 0.00000 // (op 63) call()
v_accvgpr_write a12, 0.00000 // (op 63) call()
v_accvgpr_write a13, 0.00000 // (op 63) call()
v_accvgpr_write a14, 0.00000 // (op 63) call()
v_accvgpr_write a15, 0.00000 // (op 63) call()
v_accvgpr_write a16, 0.00000 // (op 398) call()
v_accvgpr_write a17, 0.00000 // (op 398) call()
v_accvgpr_write a18, 0.00000 // (op 398) call()
v_accvgpr_write a19, 0.00000 // (op 398) call()
v_accvgpr_write a20, 0.00000 // (op 398) call()
v_accvgpr_write a21, 0.00000 // (op 398) call()
v_accvgpr_write a22, 0.00000 // (op 398) call()
v_accvgpr_write a23, 0.00000 // (op 398) call()
v_accvgpr_write a24, 0.00000 // (op 398) call()
v_accvgpr_write a25, 0.00000 // (op 398) call()
v_accvgpr_write a26, 0.00000 // (op 398) call()
v_accvgpr_write a27, 0.00000 // (op 398) call()
v_accvgpr_write a28, 0.00000 // (op 398) call()
v_accvgpr_write a29, 0.00000 // (op 398) call()
v_accvgpr_write a30, 0.00000 // (op 398) call()
v_accvgpr_write a31, 0.00000 // (op 398) call()
v_accvgpr_write a32, 0.00000 // (op 632) call()
v_accvgpr_write a33, 0.00000 // (op 632) call()
v_accvgpr_write a34, 0.00000 // (op 632) call()
v_accvgpr_write a35, 0.00000 // (op 632) call()
v_accvgpr_write a36, 0.00000 // (op 632) call()
v_accvgpr_write a37, 0.00000 // (op 632) call()
v_accvgpr_write a38, 0.00000 // (op 632) call()
v_accvgpr_write a39, 0.00000 // (op 632) call()
v_accvgpr_write a40, 0.00000 // (op 632) call()
v_accvgpr_write a41, 0.00000 // (op 632) call()
v_accvgpr_write a42, 0.00000 // (op 632) call()
v_accvgpr_write a43, 0.00000 // (op 632) call()
v_accvgpr_write a44, 0.00000 // (op 632) call()
v_accvgpr_write a45, 0.00000 // (op 632) call()
v_accvgpr_write a46, 0.00000 // (op 632) call()
v_accvgpr_write a47, 0.00000 // (op 632) call()
v_accvgpr_write a48, 0.00000 // (op 866) call()
v_accvgpr_write a49, 0.00000 // (op 866) call()
v_accvgpr_write a50, 0.00000 // (op 866) call()
v_accvgpr_write a51, 0.00000 // (op 866) call()
v_accvgpr_write a52, 0.00000 // (op 866) call()
v_accvgpr_write a53, 0.00000 // (op 866) call()
v_accvgpr_write a54, 0.00000 // (op 866) call()
v_accvgpr_write a55, 0.00000 // (op 866) call()
v_accvgpr_write a56, 0.00000 // (op 866) call()
v_accvgpr_write a57, 0.00000 // (op 866) call()
v_accvgpr_write a58, 0.00000 // (op 866) call()
v_accvgpr_write a59, 0.00000 // (op 866) call()
v_accvgpr_write a60, 0.00000 // (op 866) call()
v_accvgpr_write a61, 0.00000 // (op 866) call()
v_accvgpr_write a62, 0.00000 // (op 866) call()
v_accvgpr_write a63, 0.00000 // (op 866) call()
s_mul_hi_i32 s0, s16, s11 // (op 2438) 
v_lshrrev_b32 v14, 3, v1 // (op 2438) 
s_add_i32 s1, s0, s16 // (op 2438) 

v_and_b32 v15, -8, v1 // (op 2438) 
s_ashr_i32 s0, s1, 31 // (op 2438) 
v_sub_u32 v16, v1, v15 // (op 2438) 

s_cmp_eq_i32 s11, 0 // (op 2438) 
s_mov_b32 s18, scc // (op 2438) 
v_lshlrev_b32 v15, 4, v16 // (op 2438) 

s_cmp_eq_i32 s18, 0 // (op 2438) 
s_cselect_b32 s19, 0, -1 // (op 2438) 

v_mov_b32 v17, 0 // (op 2438) convert
v_mov_b32 v16, v15 // (op 2438) convert

s_and_b32 s18, s12, 31 // (op 2438) 
s_lshl_b32 s20, 1, s18 // (op 2438) 
s_add_i32 s21, s20, s19 // (op 2438) 


s_and_b32 s19, s0, s21 // (op 2438) 


s_add_i32 s0, s1, s19 // (op 2438) 


s_ashr_i32 s1, s0, s18 // (op 2438) 


s_lshl_b32 s0, s12, 24 // (op 2438) 
s_ashr_i32 s18, s0, 31 // (op 2438) 

s_xor_b32 s0, s1, s18 // (op 2438) 

s_sub_i32 s1, s0, s18 // (op 2438) 


s_mul_hi_i32 s0, s1, s13 // (op 2438) 
s_add_i32 s18, s0, s1 // (op 2438) 

s_ashr_i32 s0, s18, 31 // (op 2438) 
s_cmp_eq_i32 s13, 0 // (op 2438) 
s_mov_b32 s19, scc // (op 2438) 
s_cmp_eq_i32 s19, 0 // (op 2438) 
s_cselect_b32 s20, 0, -1 // (op 2438) 

s_and_b32 s19, s14, 31 // (op 2438) 
s_lshl_b32 s21, 1, s19 // (op 2438) 
s_add_i32 s22, s21, s20 // (op 2438) 


s_and_b32 s23, s0, s22 // (op 2438) 


s_add_i32 s0, s18, s23 // (op 2438) 


s_ashr_i32 s18, s0, s19 // (op 2438) 


s_lshl_b32 s0, s14, 24 // (op 2438) 
s_ashr_i32 s19, s0, 31 // (op 2438) 

s_xor_b32 s0, s18, s19 // (op 2438) 

s_sub_i32 s18, s0, s19 // (op 2438) 


s_mul_i32 s0, s1, s44 // (op 2438) 

s_sub_u32 s1, s16, s0 // (op 2438) 

s_mul_i32 s0, s18, s44 // (op 2438) 
s_add_u32 s0, s0, s1 // (op 2438) 


s_mul_hi_i32 s1, s16, s10 // (op 2438) 
s_add_i32 s18, s1, s16 // (op 2438) 

s_ashr_i32 s1, s18, 31 // (op 2438) 
s_cmp_eq_i32 s10, 0 // (op 2438) 
s_mov_b32 s19, scc // (op 2438) 
s_cmp_eq_i32 s19, 0 // (op 2438) 
s_cselect_b32 s20, 0, -1 // (op 2438) 

s_and_b32 s19, s47, 31 // (op 2438) 
s_lshl_b32 s21, 1, s19 // (op 2438) 
s_add_i32 s22, s21, s20 // (op 2438) 


s_and_b32 s23, s1, s22 // (op 2438) 


s_add_i32 s1, s18, s23 // (op 2438) 


s_ashr_i32 s18, s1, s19 // (op 2438) 


s_lshl_b32 s1, s47, 24 // (op 2438) 
s_ashr_i32 s19, s1, 31 // (op 2438) 

s_xor_b32 s1, s18, s19 // (op 2438) 

s_sub_i32 s18, s1, s19 // (op 2438) 


s_add_u32 s22, s6, 256 // (op 2438) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2438) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 2438) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 2438) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 8 // (op 2438) 

s_mul_i32 s1, s44, s22 // (op 2438) 

s_mul_i32 s19, s48, s1 // (op 2438) 

s_mul_i32 s1, s18, s19 // (op 2438) 

s_sub_u32 s19, s16, s1 // (op 2438) 

s_mov_b32 s1, s19 // (op 2438) 
s_mul_hi_i32 s20, s1, s15 // (op 2438) 
s_add_i32 s21, s20, s1 // (op 2438) 


s_ashr_i32 s1, s21, 31 // (op 2438) 
s_cmp_eq_i32 s15, 0 // (op 2438) 
s_mov_b32 s20, scc // (op 2438) 
s_cmp_eq_i32 s20, 0 // (op 2438) 
s_cselect_b32 s22, 0, -1 // (op 2438) 

s_and_b32 s20, s46, 31 // (op 2438) 
s_lshl_b32 s23, 1, s20 // (op 2438) 
s_add_i32 s49, s23, s22 // (op 2438) 


s_and_b32 s22, s1, s49 // (op 2438) 


s_add_i32 s1, s21, s22 // (op 2438) 


s_ashr_i32 s21, s1, s20 // (op 2438) 


s_lshl_b32 s1, s46, 24 // (op 2438) 
s_ashr_i32 s20, s1, 31 // (op 2438) 

s_xor_b32 s1, s21, s20 // (op 2438) 

s_sub_i32 s21, s1, s20 // (op 2438) 


s_mul_i32 s1, s45, s44 // (op 2438) 
s_add_u32 s22, s8, 64 // (op 2438) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 2438) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 2438) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 2438) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 6 // (op 2438) 

s_sub_i32 s20, s22, s1 // (op 2438) 

s_mul_i32 s22, s21, s20 // (op 2438) 


s_sub_u32 s23, s19, s22 // (op 2438) 


s_add_u32 s19, s23, s1 // (op 2438) 


s_cmp_eq_i32 s18, 0 // (op 2438) 
s_mov_b32 s1, scc // (op 2438) 

s_cmp_eq_i32 s1, 0 // (op 2438) 
s_cselect_b32 s18, s19, s0 // (op 2438) 



v_lshl_add_u32 v15, s18, 6, v14 // (op 2438) 


v_mul_lo_u32 v19, v15, s25 // (op 2438) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 2438) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 2438) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 2438) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 2438) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 2438) most significant half



s_mul_i32 s1, s24, s9 // (op 2440) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 2440) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 2440) least significant
s_mul_hi_u32 s19, s24, s8 // (op 2440) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2440) most significant: sum
s_add_u32 s1, s1, s19 // (op 2440) most significant: sum


s_mov_b32 s20, s26 // (op 2440) reg expression
s_mov_b32 s21, s27 // (op 2440) reg expression
s_mov_b32 s22, s0 // (op 2440) reg expression
s_mov_b32 s23, 131072 // (op 2440) call()

s_mul_hi_i32 s0, s16, s11 // (op 2457) 
v_lshrrev_b32 v14, 3, v1 // (op 2457) 
s_add_i32 s1, s0, s16 // (op 2457) 

v_and_b32 v15, -8, v1 // (op 2457) 
s_ashr_i32 s0, s1, 31 // (op 2457) 
v_sub_u32 v16, v1, v15 // (op 2457) 

s_cmp_eq_i32 s11, 0 // (op 2457) 
s_mov_b32 s18, scc // (op 2457) 
v_lshlrev_b32 v15, 4, v16 // (op 2457) 

s_cmp_eq_i32 s18, 0 // (op 2457) 
s_cselect_b32 s19, 0, -1 // (op 2457) 

v_mov_b32 v17, 0 // (op 2457) convert
v_mov_b32 v16, v15 // (op 2457) convert

s_and_b32 s18, s12, 31 // (op 2457) 
s_lshl_b32 s49, 1, s18 // (op 2457) 
s_add_i32 s50, s49, s19 // (op 2457) 


s_and_b32 s19, s0, s50 // (op 2457) 


s_add_i32 s0, s1, s19 // (op 2457) 


s_ashr_i32 s1, s0, s18 // (op 2457) 


s_lshl_b32 s0, s12, 24 // (op 2457) 
s_ashr_i32 s18, s0, 31 // (op 2457) 

s_xor_b32 s0, s1, s18 // (op 2457) 

s_sub_i32 s1, s0, s18 // (op 2457) 


s_mul_hi_i32 s0, s1, s13 // (op 2457) 
s_add_i32 s18, s0, s1 // (op 2457) 

s_ashr_i32 s0, s18, 31 // (op 2457) 
s_cmp_eq_i32 s13, 0 // (op 2457) 
s_mov_b32 s19, scc // (op 2457) 
s_cmp_eq_i32 s19, 0 // (op 2457) 
s_cselect_b32 s49, 0, -1 // (op 2457) 

s_and_b32 s19, s14, 31 // (op 2457) 
s_lshl_b32 s50, 1, s19 // (op 2457) 
s_add_i32 s51, s50, s49 // (op 2457) 


s_and_b32 s49, s0, s51 // (op 2457) 


s_add_i32 s0, s18, s49 // (op 2457) 


s_ashr_i32 s18, s0, s19 // (op 2457) 


s_lshl_b32 s0, s14, 24 // (op 2457) 
s_ashr_i32 s19, s0, 31 // (op 2457) 

s_xor_b32 s0, s18, s19 // (op 2457) 

s_sub_i32 s18, s0, s19 // (op 2457) 


s_mul_i32 s0, s1, s44 // (op 2457) 

s_sub_u32 s1, s16, s0 // (op 2457) 

s_mul_i32 s0, s18, s44 // (op 2457) 
s_add_u32 s0, s0, s1 // (op 2457) 


s_mul_hi_i32 s1, s16, s10 // (op 2457) 
s_add_i32 s18, s1, s16 // (op 2457) 

s_ashr_i32 s1, s18, 31 // (op 2457) 
s_cmp_eq_i32 s10, 0 // (op 2457) 
s_mov_b32 s19, scc // (op 2457) 
s_cmp_eq_i32 s19, 0 // (op 2457) 
s_cselect_b32 s49, 0, -1 // (op 2457) 

s_and_b32 s19, s47, 31 // (op 2457) 
s_lshl_b32 s50, 1, s19 // (op 2457) 
s_add_i32 s51, s50, s49 // (op 2457) 


s_and_b32 s49, s1, s51 // (op 2457) 


s_add_i32 s1, s18, s49 // (op 2457) 


s_ashr_i32 s18, s1, s19 // (op 2457) 


s_lshl_b32 s1, s47, 24 // (op 2457) 
s_ashr_i32 s19, s1, 31 // (op 2457) 

s_xor_b32 s1, s18, s19 // (op 2457) 

s_sub_i32 s18, s1, s19 // (op 2457) 


s_add_u32 s54, s6, 256 // (op 2457) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 2457) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 2457) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 2457) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 8 // (op 2457) 

s_mul_i32 s1, s44, s50 // (op 2457) 

s_mul_i32 s19, s48, s1 // (op 2457) 

s_mul_i32 s1, s18, s19 // (op 2457) 

s_sub_u32 s19, s16, s1 // (op 2457) 

s_mov_b32 s1, s19 // (op 2457) 
s_mul_hi_i32 s49, s1, s15 // (op 2457) 
s_add_i32 s50, s49, s1 // (op 2457) 


s_ashr_i32 s1, s50, 31 // (op 2457) 
s_cmp_eq_i32 s15, 0 // (op 2457) 
s_mov_b32 s49, scc // (op 2457) 
s_cmp_eq_i32 s49, 0 // (op 2457) 
s_cselect_b32 s51, 0, -1 // (op 2457) 

s_and_b32 s49, s46, 31 // (op 2457) 
s_lshl_b32 s52, 1, s49 // (op 2457) 
s_add_i32 s53, s52, s51 // (op 2457) 


s_and_b32 s51, s1, s53 // (op 2457) 


s_add_i32 s1, s50, s51 // (op 2457) 


s_ashr_i32 s50, s1, s49 // (op 2457) 


s_lshl_b32 s1, s46, 24 // (op 2457) 
s_ashr_i32 s49, s1, 31 // (op 2457) 

s_xor_b32 s1, s50, s49 // (op 2457) 

s_sub_i32 s50, s1, s49 // (op 2457) 


s_mul_i32 s1, s45, s44 // (op 2457) 
s_add_u32 s54, s8, 64 // (op 2457) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 2457) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 2457) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 2457) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 6 // (op 2457) 

s_sub_i32 s49, s54, s1 // (op 2457) 

s_mul_i32 s51, s50, s49 // (op 2457) 


s_sub_u32 s49, s19, s51 // (op 2457) 


s_add_u32 s19, s49, s1 // (op 2457) 


s_cmp_eq_i32 s18, 0 // (op 2457) 
s_mov_b32 s1, scc // (op 2457) 

s_cmp_eq_i32 s1, 0 // (op 2457) 
s_cselect_b32 s18, s19, s0 // (op 2457) 



v_lshl_add_u32 v15, s18, 6, v14 // (op 2457) 


v_mul_lo_u32 v21, v15, s25 // (op 2457) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 2457) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 2457) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 2457) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 2457) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 2457) most significant half



s_mul_i32 s1, s24, s9 // (op 2459) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 2459) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 2459) least significant
s_mul_hi_u32 s19, s24, s8 // (op 2459) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2459) most significant: sum
s_add_u32 s1, s1, s19 // (op 2459) most significant: sum


s_mov_b32 s20, s26 // (op 2459) reg expression
s_mov_b32 s21, s27 // (op 2459) reg expression
s_mov_b32 s22, s0 // (op 2459) reg expression
s_mov_b32 s23, 131072 // (op 2459) call()

s_mul_hi_i32 s0, s16, s11 // (op 2502) 
v_and_b32 v14, -4, v1 // (op 2502) 
s_add_i32 s1, s0, s16 // (op 2502) 

v_sub_u32 v15, v1, v14 // (op 2502) 

s_ashr_i32 s0, s1, 31 // (op 2502) 
v_lshlrev_b32 v14, 6, v15 // (op 2502) 

s_cmp_eq_i32 s11, 0 // (op 2502) 
s_mov_b32 s18, scc // (op 2502) 
v_lshrrev_b32 v15, 2, v1 // (op 2502) 
s_cmp_eq_i32 s18, 0 // (op 2502) 
s_cselect_b32 s19, 0, -1 // (op 2502) 

v_lshlrev_b32 v16, 1, v15 // (op 2502) 

s_and_b32 s18, s12, 31 // (op 2502) 
s_lshl_b32 s49, 1, s18 // (op 2502) 
s_add_i32 s50, s49, s19 // (op 2502) 


s_and_b32 s19, s0, s50 // (op 2502) 


s_add_i32 s0, s1, s19 // (op 2502) 


s_ashr_i32 s1, s0, s18 // (op 2502) 


s_lshl_b32 s0, s12, 24 // (op 2502) 
s_ashr_i32 s18, s0, 31 // (op 2502) 

s_xor_b32 s0, s1, s18 // (op 2502) 

s_sub_i32 s1, s0, s18 // (op 2502) 


s_mul_hi_i32 s0, s1, s13 // (op 2502) 
s_add_i32 s18, s0, s1 // (op 2502) 

s_ashr_i32 s0, s18, 31 // (op 2502) 
s_cmp_eq_i32 s13, 0 // (op 2502) 
s_mov_b32 s19, scc // (op 2502) 
s_cmp_eq_i32 s19, 0 // (op 2502) 
s_cselect_b32 s49, 0, -1 // (op 2502) 

s_and_b32 s19, s14, 31 // (op 2502) 
s_lshl_b32 s50, 1, s19 // (op 2502) 
s_add_i32 s51, s50, s49 // (op 2502) 


s_and_b32 s49, s0, s51 // (op 2502) 


s_add_i32 s0, s18, s49 // (op 2502) 


s_ashr_i32 s18, s0, s19 // (op 2502) 


s_lshl_b32 s0, s14, 24 // (op 2502) 
s_ashr_i32 s19, s0, 31 // (op 2502) 

s_xor_b32 s0, s18, s19 // (op 2502) 

s_sub_i32 s18, s0, s19 // (op 2502) 


s_add_u32 s54, s6, 256 // (op 2502) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 2502) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 2502) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 2502) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 8 // (op 2502) 

s_mul_i32 s0, s18, s50 // (op 2502) 

s_sub_i32 s49, s1, s0 // (op 2502) 


s_mul_hi_i32 s0, s16, s10 // (op 2502) 
s_add_i32 s1, s0, s16 // (op 2502) 

s_ashr_i32 s0, s1, 31 // (op 2502) 
s_cmp_eq_i32 s10, 0 // (op 2502) 
s_mov_b32 s18, scc // (op 2502) 
s_cmp_eq_i32 s18, 0 // (op 2502) 
s_cselect_b32 s19, 0, -1 // (op 2502) 

s_and_b32 s18, s47, 31 // (op 2502) 
s_lshl_b32 s52, 1, s18 // (op 2502) 
s_add_i32 s53, s52, s19 // (op 2502) 


s_and_b32 s19, s0, s53 // (op 2502) 


s_add_i32 s0, s1, s19 // (op 2502) 


s_ashr_i32 s1, s0, s18 // (op 2502) 


s_lshl_b32 s0, s47, 24 // (op 2502) 
s_ashr_i32 s18, s0, 31 // (op 2502) 

s_xor_b32 s0, s1, s18 // (op 2502) 

s_sub_i32 s1, s0, s18 // (op 2502) 


s_mul_i32 s0, s44, s50 // (op 2502) 

s_mul_i32 s18, s48, s0 // (op 2502) 

s_mul_i32 s0, s1, s18 // (op 2502) 

s_sub_u32 s18, s16, s0 // (op 2502) 

s_mul_hi_i32 s0, s18, s15 // (op 2502) 
s_add_i32 s19, s0, s18 // (op 2502) 


s_ashr_i32 s0, s19, 31 // (op 2502) 
s_cmp_eq_i32 s15, 0 // (op 2502) 
s_mov_b32 s18, scc // (op 2502) 
s_cmp_eq_i32 s18, 0 // (op 2502) 
s_cselect_b32 s50, 0, -1 // (op 2502) 

s_and_b32 s18, s46, 31 // (op 2502) 
s_lshl_b32 s51, 1, s18 // (op 2502) 
s_add_i32 s52, s51, s50 // (op 2502) 


s_and_b32 s50, s0, s52 // (op 2502) 


s_add_i32 s0, s19, s50 // (op 2502) 


s_ashr_i32 s19, s0, s18 // (op 2502) 


s_lshl_b32 s0, s46, 24 // (op 2502) 
s_ashr_i32 s18, s0, 31 // (op 2502) 

s_xor_b32 s0, s19, s18 // (op 2502) 

s_sub_i32 s19, s0, s18 // (op 2502) 


s_cmp_eq_i32 s1, 0 // (op 2502) 
s_mov_b32 s0, scc // (op 2502) 

s_cmp_eq_i32 s0, 0 // (op 2502) 
s_cselect_b32 s1, s19, s49 // (op 2502) 



v_lshl_add_u32 v15, s1, 8, v14 // (op 2502) 


v_mov_b32 v23, 0 // (op 2502) convert
v_mov_b32 v22, v15 // (op 2502) convert

v_mul_lo_u32 v15, v16, s29 // (op 2502) most significant: low of low * high
v_mul_hi_u32 v17, v16, s28 // (op 2502) most significant: high of low * low
v_mul_lo_u32 v14, v16, s28 // (op 2502) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 2502) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v22 // (op 2502) least significant half
v_addc_co_u32 v15, s[0:1], v15, v23, s[0:1] // (op 2502) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 2502) most significant: low of high * low
v_mul_hi_u32 v22, v14, 6 // (op 2502) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 2502) least significant: low of low * low
v_add_u32 v17, v17, v22 // (op 2502) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 2502) 

v_mov_b32 v16, v14 // (op 2502) convert

s_mul_i32 s1, s33, 6 // (op 2504) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 2504) least significant
s_mul_hi_u32 s18, s32, 6 // (op 2504) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2504) most significant: sum

s_mul_i32 s19, s28, s1 // (op 2504) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 2504) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 2504) least significant
s_mul_hi_u32 s50, s28, s0 // (op 2504) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 2504) most significant: sum
s_add_u32 s19, s19, s50 // (op 2504) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2504) 

s_mov_b32 s52, s30 // (op 2504) reg expression
s_mov_b32 s53, s31 // (op 2504) reg expression
s_mov_b32 s54, s0 // (op 2504) reg expression
s_mov_b32 s55, 131072 // (op 2504) call()

s_mul_hi_i32 s0, s16, s11 // (op 2521) 
v_and_b32 v17, -4, v1 // (op 2521) 
s_add_i32 s1, s0, s16 // (op 2521) 

v_sub_u32 v14, v1, v17 // (op 2521) 

s_ashr_i32 s0, s1, 31 // (op 2521) 
v_lshlrev_b32 v15, 6, v14 // (op 2521) 

s_cmp_eq_i32 s11, 0 // (op 2521) 
s_mov_b32 s18, scc // (op 2521) 
v_lshrrev_b32 v14, 2, v1 // (op 2521) 
s_cmp_eq_i32 s18, 0 // (op 2521) 
s_cselect_b32 s19, 0, -1 // (op 2521) 

v_lshlrev_b32 v17, 1, v14 // (op 2521) 

s_and_b32 s18, s12, 31 // (op 2521) 
s_lshl_b32 s49, 1, s18 // (op 2521) 
s_add_i32 s50, s49, s19 // (op 2521) 


s_and_b32 s19, s0, s50 // (op 2521) 


s_add_i32 s0, s1, s19 // (op 2521) 


s_ashr_i32 s1, s0, s18 // (op 2521) 


s_lshl_b32 s0, s12, 24 // (op 2521) 
s_ashr_i32 s18, s0, 31 // (op 2521) 

s_xor_b32 s0, s1, s18 // (op 2521) 

s_sub_i32 s1, s0, s18 // (op 2521) 


s_mul_hi_i32 s0, s1, s13 // (op 2521) 
s_add_i32 s18, s0, s1 // (op 2521) 

s_ashr_i32 s0, s18, 31 // (op 2521) 
s_cmp_eq_i32 s13, 0 // (op 2521) 
s_mov_b32 s19, scc // (op 2521) 
s_cmp_eq_i32 s19, 0 // (op 2521) 
s_cselect_b32 s49, 0, -1 // (op 2521) 

s_and_b32 s19, s14, 31 // (op 2521) 
s_lshl_b32 s50, 1, s19 // (op 2521) 
s_add_i32 s51, s50, s49 // (op 2521) 


s_and_b32 s49, s0, s51 // (op 2521) 


s_add_i32 s0, s18, s49 // (op 2521) 


s_ashr_i32 s18, s0, s19 // (op 2521) 


s_lshl_b32 s0, s14, 24 // (op 2521) 
s_ashr_i32 s19, s0, 31 // (op 2521) 

s_xor_b32 s0, s18, s19 // (op 2521) 

s_sub_i32 s18, s0, s19 // (op 2521) 


s_add_u32 s50, s6, 256 // (op 2521) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2521) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 2521) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 2521) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 2521) 

s_mul_i32 s0, s18, s50 // (op 2521) 

s_sub_i32 s49, s1, s0 // (op 2521) 


s_mul_hi_i32 s0, s16, s10 // (op 2521) 
s_add_i32 s1, s0, s16 // (op 2521) 

s_ashr_i32 s0, s1, 31 // (op 2521) 
s_cmp_eq_i32 s10, 0 // (op 2521) 
s_mov_b32 s18, scc // (op 2521) 
s_cmp_eq_i32 s18, 0 // (op 2521) 
s_cselect_b32 s19, 0, -1 // (op 2521) 

s_and_b32 s18, s47, 31 // (op 2521) 
s_lshl_b32 s64, 1, s18 // (op 2521) 
s_add_i32 s65, s64, s19 // (op 2521) 


s_and_b32 s19, s0, s65 // (op 2521) 


s_add_i32 s0, s1, s19 // (op 2521) 


s_ashr_i32 s1, s0, s18 // (op 2521) 


s_lshl_b32 s0, s47, 24 // (op 2521) 
s_ashr_i32 s18, s0, 31 // (op 2521) 

s_xor_b32 s0, s1, s18 // (op 2521) 

s_sub_i32 s1, s0, s18 // (op 2521) 


s_mul_i32 s0, s44, s50 // (op 2521) 

s_mul_i32 s18, s48, s0 // (op 2521) 

s_mul_i32 s0, s1, s18 // (op 2521) 

s_sub_u32 s18, s16, s0 // (op 2521) 

s_mul_hi_i32 s0, s18, s15 // (op 2521) 
s_add_i32 s19, s0, s18 // (op 2521) 


s_ashr_i32 s0, s19, 31 // (op 2521) 
s_cmp_eq_i32 s15, 0 // (op 2521) 
s_mov_b32 s18, scc // (op 2521) 
s_cmp_eq_i32 s18, 0 // (op 2521) 
s_cselect_b32 s50, 0, -1 // (op 2521) 

s_and_b32 s18, s46, 31 // (op 2521) 
s_lshl_b32 s51, 1, s18 // (op 2521) 
s_add_i32 s64, s51, s50 // (op 2521) 


s_and_b32 s50, s0, s64 // (op 2521) 


s_add_i32 s0, s19, s50 // (op 2521) 


s_ashr_i32 s19, s0, s18 // (op 2521) 


s_lshl_b32 s0, s46, 24 // (op 2521) 
s_ashr_i32 s18, s0, 31 // (op 2521) 

s_xor_b32 s0, s19, s18 // (op 2521) 

s_sub_i32 s19, s0, s18 // (op 2521) 


s_cmp_eq_i32 s1, 0 // (op 2521) 
s_mov_b32 s0, scc // (op 2521) 

s_cmp_eq_i32 s0, 0 // (op 2521) 
s_cselect_b32 s1, s19, s49 // (op 2521) 



v_lshl_add_u32 v14, s1, 8, v15 // (op 2521) 


v_mov_b32 v23, 0 // (op 2521) convert
v_mov_b32 v22, v14 // (op 2521) convert

v_mul_lo_u32 v15, v17, s29 // (op 2521) most significant: low of low * high
v_mul_hi_u32 v24, v17, s28 // (op 2521) most significant: high of low * low
v_mul_lo_u32 v14, v17, s28 // (op 2521) least significant: low of low * low
v_add_u32 v15, v15, v24 // (op 2521) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v22 // (op 2521) least significant half
v_addc_co_u32 v15, s[0:1], v15, v23, s[0:1] // (op 2521) most significant half



v_mul_lo_u32 v23, v15, 6 // (op 2521) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 2521) most significant: high of low * low
v_mul_lo_u32 v22, v14, 6 // (op 2521) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 2521) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[22:23] // (op 2521) 

v_mov_b32 v17, v14 // (op 2521) convert

s_mul_i32 s1, s33, 6 // (op 2523) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 2523) least significant
s_mul_hi_u32 s18, s32, 6 // (op 2523) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2523) most significant: sum

s_mul_i32 s19, s28, s1 // (op 2523) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 2523) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 2523) least significant
s_mul_hi_u32 s50, s28, s0 // (op 2523) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 2523) most significant: sum
s_add_u32 s19, s19, s50 // (op 2523) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2523) 

s_mov_b32 s52, s30 // (op 2523) reg expression
s_mov_b32 s53, s31 // (op 2523) reg expression
s_mov_b32 s54, s0 // (op 2523) reg expression
s_mov_b32 s55, 131072 // (op 2523) call()

s_mul_hi_i32 s0, s16, s11 // (op 2566) 
v_lshrrev_b32 v14, 2, v1 // (op 2566) 
s_add_i32 s1, s0, s16 // (op 2566) 

v_and_b32 v15, -4, v1 // (op 2566) 
s_ashr_i32 s0, s1, 31 // (op 2566) 
v_sub_u32 v22, v1, v15 // (op 2566) 

s_cmp_eq_i32 s11, 0 // (op 2566) 
s_mov_b32 s18, scc // (op 2566) 
s_cmp_eq_i32 s18, 0 // (op 2566) 
s_cselect_b32 s19, 0, -1 // (op 2566) 

s_and_b32 s18, s12, 31 // (op 2566) 
s_lshl_b32 s49, 1, s18 // (op 2566) 
s_add_i32 s50, s49, s19 // (op 2566) 


s_and_b32 s19, s0, s50 // (op 2566) 


s_add_i32 s0, s1, s19 // (op 2566) 


s_ashr_i32 s1, s0, s18 // (op 2566) 


s_lshl_b32 s0, s12, 24 // (op 2566) 
s_ashr_i32 s18, s0, 31 // (op 2566) 

s_xor_b32 s0, s1, s18 // (op 2566) 

s_sub_i32 s1, s0, s18 // (op 2566) 


s_mul_hi_i32 s0, s1, s13 // (op 2566) 
s_add_i32 s18, s0, s1 // (op 2566) 

s_ashr_i32 s0, s18, 31 // (op 2566) 
s_cmp_eq_i32 s13, 0 // (op 2566) 
s_mov_b32 s19, scc // (op 2566) 
s_cmp_eq_i32 s19, 0 // (op 2566) 
s_cselect_b32 s49, 0, -1 // (op 2566) 

s_and_b32 s19, s14, 31 // (op 2566) 
s_lshl_b32 s50, 1, s19 // (op 2566) 
s_add_i32 s51, s50, s49 // (op 2566) 


s_and_b32 s49, s0, s51 // (op 2566) 


s_add_i32 s0, s18, s49 // (op 2566) 


s_ashr_i32 s18, s0, s19 // (op 2566) 


s_lshl_b32 s0, s14, 24 // (op 2566) 
s_ashr_i32 s19, s0, 31 // (op 2566) 

s_xor_b32 s0, s18, s19 // (op 2566) 

s_sub_i32 s18, s0, s19 // (op 2566) 


s_mul_i32 s0, s1, s44 // (op 2566) 

s_sub_u32 s1, s16, s0 // (op 2566) 

s_mul_i32 s0, s18, s44 // (op 2566) 
s_add_u32 s0, s0, s1 // (op 2566) 


s_mul_hi_i32 s1, s16, s10 // (op 2566) 
s_add_i32 s18, s1, s16 // (op 2566) 

s_ashr_i32 s1, s18, 31 // (op 2566) 
s_cmp_eq_i32 s10, 0 // (op 2566) 
s_mov_b32 s19, scc // (op 2566) 
s_cmp_eq_i32 s19, 0 // (op 2566) 
s_cselect_b32 s49, 0, -1 // (op 2566) 

s_and_b32 s19, s47, 31 // (op 2566) 
s_lshl_b32 s50, 1, s19 // (op 2566) 
s_add_i32 s51, s50, s49 // (op 2566) 


s_and_b32 s49, s1, s51 // (op 2566) 


s_add_i32 s1, s18, s49 // (op 2566) 


s_ashr_i32 s18, s1, s19 // (op 2566) 


s_lshl_b32 s1, s47, 24 // (op 2566) 
s_ashr_i32 s19, s1, 31 // (op 2566) 

s_xor_b32 s1, s18, s19 // (op 2566) 

s_sub_i32 s18, s1, s19 // (op 2566) 


s_add_u32 s50, s6, 256 // (op 2566) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2566) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 2566) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 2566) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 2566) 

s_mul_i32 s1, s44, s50 // (op 2566) 

s_mul_i32 s19, s48, s1 // (op 2566) 

s_mul_i32 s1, s18, s19 // (op 2566) 

s_sub_u32 s19, s16, s1 // (op 2566) 

s_mov_b32 s1, s19 // (op 2566) 
s_mul_hi_i32 s49, s1, s15 // (op 2566) 
s_add_i32 s50, s49, s1 // (op 2566) 


s_ashr_i32 s1, s50, 31 // (op 2566) 
s_cmp_eq_i32 s15, 0 // (op 2566) 
s_mov_b32 s49, scc // (op 2566) 
s_cmp_eq_i32 s49, 0 // (op 2566) 
s_cselect_b32 s51, 0, -1 // (op 2566) 

s_and_b32 s49, s46, 31 // (op 2566) 
s_lshl_b32 s64, 1, s49 // (op 2566) 
s_add_i32 s65, s64, s51 // (op 2566) 


s_and_b32 s51, s1, s65 // (op 2566) 


s_add_i32 s1, s50, s51 // (op 2566) 


s_ashr_i32 s50, s1, s49 // (op 2566) 


s_lshl_b32 s1, s46, 24 // (op 2566) 
s_ashr_i32 s49, s1, 31 // (op 2566) 

s_xor_b32 s1, s50, s49 // (op 2566) 

s_sub_i32 s50, s1, s49 // (op 2566) 


s_mul_i32 s1, s45, s44 // (op 2566) 
s_add_u32 s64, s8, 64 // (op 2566) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 2566) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 2566) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 2566) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 6 // (op 2566) 

s_sub_i32 s49, s64, s1 // (op 2566) 

s_mul_i32 s51, s50, s49 // (op 2566) 


s_sub_u32 s49, s19, s51 // (op 2566) 


s_add_u32 s19, s49, s1 // (op 2566) 


s_cmp_eq_i32 s18, 0 // (op 2566) 
s_mov_b32 s1, scc // (op 2566) 

s_cmp_eq_i32 s1, 0 // (op 2566) 
s_cselect_b32 s18, s19, s0 // (op 2566) 



v_lshl_add_u32 v15, s18, 6, v14 // (op 2566) 


v_mul_lo_u32 v14, v15, s34 // (op 2566) 
v_add_u32 v14, v14, v22 // (op 2566) 


s_mul_i32 s1, s34, s9 // (op 2568) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 2568) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2568) least significant
s_mul_hi_u32 s19, s34, s8 // (op 2568) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2568) most significant: sum
s_add_u32 s1, s1, s19 // (op 2568) most significant: sum


s_mov_b32 s64, s36 // (op 2568) reg expression
s_mov_b32 s65, s37 // (op 2568) reg expression
s_mov_b32 s66, s0 // (op 2568) reg expression
s_mov_b32 s67, 131072 // (op 2568) call()

s_mul_hi_i32 s0, s16, s11 // (op 2585) 
v_lshrrev_b32 v15, 2, v1 // (op 2585) 
s_add_i32 s1, s0, s16 // (op 2585) 

v_and_b32 v22, -4, v1 // (op 2585) 
s_ashr_i32 s0, s1, 31 // (op 2585) 
v_sub_u32 v23, v1, v22 // (op 2585) 

s_cmp_eq_i32 s11, 0 // (op 2585) 
s_mov_b32 s18, scc // (op 2585) 
s_cmp_eq_i32 s18, 0 // (op 2585) 
s_cselect_b32 s19, 0, -1 // (op 2585) 

s_and_b32 s18, s12, 31 // (op 2585) 
s_lshl_b32 s49, 1, s18 // (op 2585) 
s_add_i32 s50, s49, s19 // (op 2585) 


s_and_b32 s19, s0, s50 // (op 2585) 


s_add_i32 s0, s1, s19 // (op 2585) 


s_ashr_i32 s1, s0, s18 // (op 2585) 


s_lshl_b32 s0, s12, 24 // (op 2585) 
s_ashr_i32 s18, s0, 31 // (op 2585) 

s_xor_b32 s0, s1, s18 // (op 2585) 

s_sub_i32 s1, s0, s18 // (op 2585) 


s_mul_hi_i32 s0, s1, s13 // (op 2585) 
s_add_i32 s18, s0, s1 // (op 2585) 

s_ashr_i32 s0, s18, 31 // (op 2585) 
s_cmp_eq_i32 s13, 0 // (op 2585) 
s_mov_b32 s19, scc // (op 2585) 
s_cmp_eq_i32 s19, 0 // (op 2585) 
s_cselect_b32 s49, 0, -1 // (op 2585) 

s_and_b32 s19, s14, 31 // (op 2585) 
s_lshl_b32 s50, 1, s19 // (op 2585) 
s_add_i32 s51, s50, s49 // (op 2585) 


s_and_b32 s49, s0, s51 // (op 2585) 


s_add_i32 s0, s18, s49 // (op 2585) 


s_ashr_i32 s18, s0, s19 // (op 2585) 


s_lshl_b32 s0, s14, 24 // (op 2585) 
s_ashr_i32 s19, s0, 31 // (op 2585) 

s_xor_b32 s0, s18, s19 // (op 2585) 

s_sub_i32 s18, s0, s19 // (op 2585) 


s_mul_i32 s0, s1, s44 // (op 2585) 

s_sub_u32 s1, s16, s0 // (op 2585) 

s_mul_i32 s0, s18, s44 // (op 2585) 
s_add_u32 s0, s0, s1 // (op 2585) 


s_mul_hi_i32 s1, s16, s10 // (op 2585) 
s_add_i32 s18, s1, s16 // (op 2585) 

s_ashr_i32 s1, s18, 31 // (op 2585) 
s_cmp_eq_i32 s10, 0 // (op 2585) 
s_mov_b32 s19, scc // (op 2585) 
s_cmp_eq_i32 s19, 0 // (op 2585) 
s_cselect_b32 s49, 0, -1 // (op 2585) 

s_and_b32 s19, s47, 31 // (op 2585) 
s_lshl_b32 s50, 1, s19 // (op 2585) 
s_add_i32 s51, s50, s49 // (op 2585) 


s_and_b32 s49, s1, s51 // (op 2585) 


s_add_i32 s1, s18, s49 // (op 2585) 


s_ashr_i32 s18, s1, s19 // (op 2585) 


s_lshl_b32 s1, s47, 24 // (op 2585) 
s_ashr_i32 s19, s1, 31 // (op 2585) 

s_xor_b32 s1, s18, s19 // (op 2585) 

s_sub_i32 s18, s1, s19 // (op 2585) 


s_add_u32 s50, s6, 256 // (op 2585) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2585) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 2585) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 2585) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 2585) 

s_mul_i32 s1, s44, s50 // (op 2585) 

s_mul_i32 s19, s48, s1 // (op 2585) 

s_mul_i32 s1, s18, s19 // (op 2585) 

s_sub_u32 s19, s16, s1 // (op 2585) 

s_mov_b32 s1, s19 // (op 2585) 
s_mul_hi_i32 s49, s1, s15 // (op 2585) 
s_add_i32 s50, s49, s1 // (op 2585) 


s_ashr_i32 s1, s50, 31 // (op 2585) 
s_cmp_eq_i32 s15, 0 // (op 2585) 
s_mov_b32 s49, scc // (op 2585) 
s_cmp_eq_i32 s49, 0 // (op 2585) 
s_cselect_b32 s51, 0, -1 // (op 2585) 

s_and_b32 s49, s46, 31 // (op 2585) 
s_lshl_b32 s68, 1, s49 // (op 2585) 
s_add_i32 s69, s68, s51 // (op 2585) 


s_and_b32 s51, s1, s69 // (op 2585) 


s_add_i32 s1, s50, s51 // (op 2585) 


s_ashr_i32 s50, s1, s49 // (op 2585) 


s_lshl_b32 s1, s46, 24 // (op 2585) 
s_ashr_i32 s49, s1, 31 // (op 2585) 

s_xor_b32 s1, s50, s49 // (op 2585) 

s_sub_i32 s50, s1, s49 // (op 2585) 


s_mul_i32 s1, s45, s44 // (op 2585) 
s_add_u32 s68, s8, 64 // (op 2585) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 2585) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 2585) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 2585) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 6 // (op 2585) 

s_sub_i32 s49, s68, s1 // (op 2585) 

s_mul_i32 s51, s50, s49 // (op 2585) 


s_sub_u32 s49, s19, s51 // (op 2585) 


s_add_u32 s19, s49, s1 // (op 2585) 


s_cmp_eq_i32 s18, 0 // (op 2585) 
s_mov_b32 s1, scc // (op 2585) 

s_cmp_eq_i32 s1, 0 // (op 2585) 
s_cselect_b32 s18, s19, s0 // (op 2585) 



v_lshl_add_u32 v22, s18, 6, v15 // (op 2585) 


v_mul_lo_u32 v15, v22, s34 // (op 2585) 
v_add_u32 v15, v15, v23 // (op 2585) 


s_mul_i32 s1, s34, s9 // (op 2587) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 2587) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2587) least significant
s_mul_hi_u32 s19, s34, s8 // (op 2587) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2587) most significant: sum
s_add_u32 s1, s1, s19 // (op 2587) most significant: sum


s_mov_b32 s64, s36 // (op 2587) reg expression
s_mov_b32 s65, s37 // (op 2587) reg expression
s_mov_b32 s66, s0 // (op 2587) reg expression
s_mov_b32 s67, 131072 // (op 2587) call()

s_mul_hi_i32 s0, s16, s11 // (op 2630) 
v_and_b32 v22, -64, v1 // (op 2630) 
s_add_i32 s1, s0, s16 // (op 2630) 

v_sub_u32 v23, v1, v22 // (op 2630) 

s_ashr_i32 s0, s1, 31 // (op 2630) 
v_lshlrev_b32 v22, 2, v23 // (op 2630) 

s_cmp_eq_i32 s11, 0 // (op 2630) 
s_mov_b32 s18, scc // (op 2630) 
v_lshrrev_b32 v23, 6, v1 // (op 2630) 
s_cmp_eq_i32 s18, 0 // (op 2630) 
s_cselect_b32 s19, 0, -1 // (op 2630) 

s_and_b32 s18, s12, 31 // (op 2630) 
s_lshl_b32 s49, 1, s18 // (op 2630) 
s_add_i32 s50, s49, s19 // (op 2630) 


s_and_b32 s19, s0, s50 // (op 2630) 


s_add_i32 s0, s1, s19 // (op 2630) 


s_ashr_i32 s1, s0, s18 // (op 2630) 


s_lshl_b32 s0, s12, 24 // (op 2630) 
s_ashr_i32 s18, s0, 31 // (op 2630) 

s_xor_b32 s0, s1, s18 // (op 2630) 

s_sub_i32 s1, s0, s18 // (op 2630) 


s_mul_hi_i32 s0, s1, s13 // (op 2630) 
s_add_i32 s18, s0, s1 // (op 2630) 

s_ashr_i32 s0, s18, 31 // (op 2630) 
s_cmp_eq_i32 s13, 0 // (op 2630) 
s_mov_b32 s19, scc // (op 2630) 
s_cmp_eq_i32 s19, 0 // (op 2630) 
s_cselect_b32 s49, 0, -1 // (op 2630) 

s_and_b32 s19, s14, 31 // (op 2630) 
s_lshl_b32 s50, 1, s19 // (op 2630) 
s_add_i32 s51, s50, s49 // (op 2630) 


s_and_b32 s49, s0, s51 // (op 2630) 


s_add_i32 s0, s18, s49 // (op 2630) 


s_ashr_i32 s18, s0, s19 // (op 2630) 


s_lshl_b32 s0, s14, 24 // (op 2630) 
s_ashr_i32 s19, s0, 31 // (op 2630) 

s_xor_b32 s0, s18, s19 // (op 2630) 

s_sub_i32 s18, s0, s19 // (op 2630) 


s_add_u32 s50, s6, 256 // (op 2630) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2630) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 2630) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 2630) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 2630) 

s_mul_i32 s0, s18, s50 // (op 2630) 

s_sub_i32 s49, s1, s0 // (op 2630) 


s_mul_hi_i32 s0, s16, s10 // (op 2630) 
s_add_i32 s1, s0, s16 // (op 2630) 

s_ashr_i32 s0, s1, 31 // (op 2630) 
s_cmp_eq_i32 s10, 0 // (op 2630) 
s_mov_b32 s18, scc // (op 2630) 
s_cmp_eq_i32 s18, 0 // (op 2630) 
s_cselect_b32 s19, 0, -1 // (op 2630) 

s_and_b32 s18, s47, 31 // (op 2630) 
s_lshl_b32 s68, 1, s18 // (op 2630) 
s_add_i32 s69, s68, s19 // (op 2630) 


s_and_b32 s19, s0, s69 // (op 2630) 


s_add_i32 s0, s1, s19 // (op 2630) 


s_ashr_i32 s1, s0, s18 // (op 2630) 


s_lshl_b32 s0, s47, 24 // (op 2630) 
s_ashr_i32 s18, s0, 31 // (op 2630) 

s_xor_b32 s0, s1, s18 // (op 2630) 

s_sub_i32 s1, s0, s18 // (op 2630) 


s_mul_i32 s0, s44, s50 // (op 2630) 

s_mul_i32 s18, s48, s0 // (op 2630) 

s_mul_i32 s0, s1, s18 // (op 2630) 

s_sub_u32 s18, s16, s0 // (op 2630) 

s_mul_hi_i32 s0, s18, s15 // (op 2630) 
s_add_i32 s19, s0, s18 // (op 2630) 


s_ashr_i32 s0, s19, 31 // (op 2630) 
s_cmp_eq_i32 s15, 0 // (op 2630) 
s_mov_b32 s18, scc // (op 2630) 
s_cmp_eq_i32 s18, 0 // (op 2630) 
s_cselect_b32 s50, 0, -1 // (op 2630) 

s_and_b32 s18, s46, 31 // (op 2630) 
s_lshl_b32 s51, 1, s18 // (op 2630) 
s_add_i32 s68, s51, s50 // (op 2630) 


s_and_b32 s50, s0, s68 // (op 2630) 


s_add_i32 s0, s19, s50 // (op 2630) 


s_ashr_i32 s19, s0, s18 // (op 2630) 


s_lshl_b32 s0, s46, 24 // (op 2630) 
s_ashr_i32 s18, s0, 31 // (op 2630) 

s_xor_b32 s0, s19, s18 // (op 2630) 

s_sub_i32 s19, s0, s18 // (op 2630) 


s_cmp_eq_i32 s1, 0 // (op 2630) 
s_mov_b32 s0, scc // (op 2630) 

s_cmp_eq_i32 s0, 0 // (op 2630) 
s_cselect_b32 s1, s19, s49 // (op 2630) 



v_lshl_add_u32 v24, s1, 8, v22 // (op 2630) 


v_mul_lo_u32 v22, v23, s38 // (op 2630) 
v_add_u32 v22, v22, v24 // (op 2630) 


s_mul_i32 s1, s38, s43 // (op 2632) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 2632) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 2632) least significant
s_mul_hi_u32 s19, s38, s42 // (op 2632) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2632) most significant: sum
s_add_u32 s1, s1, s19 // (op 2632) most significant: sum


s_mov_b32 s68, s40 // (op 2632) reg expression
s_mov_b32 s69, s41 // (op 2632) reg expression
s_mov_b32 s70, s0 // (op 2632) reg expression
s_mov_b32 s71, 131072 // (op 2632) call()

s_mul_hi_i32 s0, s16, s11 // (op 2649) 
v_and_b32 v23, -64, v1 // (op 2649) 
s_add_i32 s1, s0, s16 // (op 2649) 

v_sub_u32 v24, v1, v23 // (op 2649) 

s_ashr_i32 s0, s1, 31 // (op 2649) 
v_lshlrev_b32 v23, 2, v24 // (op 2649) 

s_cmp_eq_i32 s11, 0 // (op 2649) 
s_mov_b32 s18, scc // (op 2649) 
v_lshrrev_b32 v24, 6, v1 // (op 2649) 
s_cmp_eq_i32 s18, 0 // (op 2649) 
s_cselect_b32 s19, 0, -1 // (op 2649) 

s_and_b32 s18, s12, 31 // (op 2649) 
s_lshl_b32 s49, 1, s18 // (op 2649) 
s_add_i32 s50, s49, s19 // (op 2649) 


s_and_b32 s19, s0, s50 // (op 2649) 


s_add_i32 s0, s1, s19 // (op 2649) 


s_ashr_i32 s1, s0, s18 // (op 2649) 


s_lshl_b32 s0, s12, 24 // (op 2649) 
s_ashr_i32 s18, s0, 31 // (op 2649) 

s_xor_b32 s0, s1, s18 // (op 2649) 

s_sub_i32 s1, s0, s18 // (op 2649) 


s_mul_hi_i32 s0, s1, s13 // (op 2649) 
s_add_i32 s18, s0, s1 // (op 2649) 

s_ashr_i32 s0, s18, 31 // (op 2649) 
s_cmp_eq_i32 s13, 0 // (op 2649) 
s_mov_b32 s19, scc // (op 2649) 
s_cmp_eq_i32 s19, 0 // (op 2649) 
s_cselect_b32 s49, 0, -1 // (op 2649) 

s_and_b32 s19, s14, 31 // (op 2649) 
s_lshl_b32 s50, 1, s19 // (op 2649) 
s_add_i32 s51, s50, s49 // (op 2649) 


s_and_b32 s49, s0, s51 // (op 2649) 


s_add_i32 s0, s18, s49 // (op 2649) 


s_ashr_i32 s18, s0, s19 // (op 2649) 


s_lshl_b32 s0, s14, 24 // (op 2649) 
s_ashr_i32 s19, s0, 31 // (op 2649) 

s_xor_b32 s0, s18, s19 // (op 2649) 

s_sub_i32 s18, s0, s19 // (op 2649) 


s_add_u32 s50, s6, 256 // (op 2649) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2649) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2649) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2649) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 2649) 

s_mul_i32 s0, s18, s50 // (op 2649) 

s_sub_i32 s49, s1, s0 // (op 2649) 


s_mul_hi_i32 s0, s16, s10 // (op 2649) 
s_add_i32 s1, s0, s16 // (op 2649) 

s_ashr_i32 s0, s1, 31 // (op 2649) 
s_cmp_eq_i32 s10, 0 // (op 2649) 
s_mov_b32 s18, scc // (op 2649) 
s_cmp_eq_i32 s18, 0 // (op 2649) 
s_cselect_b32 s19, 0, -1 // (op 2649) 

s_and_b32 s18, s47, 31 // (op 2649) 
s_lshl_b32 s72, 1, s18 // (op 2649) 
s_add_i32 s73, s72, s19 // (op 2649) 


s_and_b32 s19, s0, s73 // (op 2649) 


s_add_i32 s0, s1, s19 // (op 2649) 


s_ashr_i32 s1, s0, s18 // (op 2649) 


s_lshl_b32 s0, s47, 24 // (op 2649) 
s_ashr_i32 s18, s0, 31 // (op 2649) 

s_xor_b32 s0, s1, s18 // (op 2649) 

s_sub_i32 s1, s0, s18 // (op 2649) 


s_mul_i32 s0, s44, s50 // (op 2649) 

s_mul_i32 s18, s48, s0 // (op 2649) 

s_mul_i32 s0, s1, s18 // (op 2649) 

s_sub_u32 s18, s16, s0 // (op 2649) 

s_mul_hi_i32 s0, s18, s15 // (op 2649) 
s_add_i32 s19, s0, s18 // (op 2649) 


s_ashr_i32 s0, s19, 31 // (op 2649) 
s_cmp_eq_i32 s15, 0 // (op 2649) 
s_mov_b32 s18, scc // (op 2649) 
s_cmp_eq_i32 s18, 0 // (op 2649) 
s_cselect_b32 s50, 0, -1 // (op 2649) 

s_and_b32 s18, s46, 31 // (op 2649) 
s_lshl_b32 s51, 1, s18 // (op 2649) 
s_add_i32 s72, s51, s50 // (op 2649) 


s_and_b32 s50, s0, s72 // (op 2649) 


s_add_i32 s0, s19, s50 // (op 2649) 


s_ashr_i32 s19, s0, s18 // (op 2649) 


s_lshl_b32 s0, s46, 24 // (op 2649) 
s_ashr_i32 s18, s0, 31 // (op 2649) 

s_xor_b32 s0, s19, s18 // (op 2649) 

s_sub_i32 s19, s0, s18 // (op 2649) 


s_cmp_eq_i32 s1, 0 // (op 2649) 
s_mov_b32 s0, scc // (op 2649) 

s_cmp_eq_i32 s0, 0 // (op 2649) 
s_cselect_b32 s1, s19, s49 // (op 2649) 



v_lshl_add_u32 v25, s1, 8, v23 // (op 2649) 


v_mul_lo_u32 v23, v24, s38 // (op 2649) 
v_add_u32 v23, v23, v25 // (op 2649) 


s_mul_i32 s1, s38, s43 // (op 2651) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 2651) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 2651) least significant
s_mul_hi_u32 s19, s38, s42 // (op 2651) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2651) most significant: sum
s_add_u32 s1, s1, s19 // (op 2651) most significant: sum


s_mov_b32 s68, s40 // (op 2651) reg expression
s_mov_b32 s69, s41 // (op 2651) reg expression
s_mov_b32 s70, s0 // (op 2651) reg expression
s_mov_b32 s71, 131072 // (op 2651) call()


s_mul_i32 s1, s29, 6 // (op 2153) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 2153) least significant
s_mul_hi_u32 s18, s28, 6 // (op 2153) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2153) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 2153) 


v_mov_b32 v26, v16 // (op 2153) reg expression
buffer_load_dwordx3 v[28:30], v26, s[52:55], 0 offen // (op 2153) Load value
buffer_load_dwordx3 v[32:34], v26, s[52:55], 0 offen offset:12 // (op 2153) Load value
buffer_load_dwordx3 v[36:38], v26, s[52:55], 0 offen offset:24 // (op 2153) Load value
buffer_load_dwordx3 v[40:42], v26, s[52:55], 0 offen offset:36 // (op 2153) Load value
v_add_u32 v26, v26, v24 // (op 2153) 
buffer_load_dwordx3 v[44:46], v26, s[52:55], 0 offen // (op 2153) Load value
buffer_load_dwordx3 v[48:50], v26, s[52:55], 0 offen offset:12 // (op 2153) Load value
buffer_load_dwordx3 v[52:54], v26, s[52:55], 0 offen offset:24 // (op 2153) Load value
buffer_load_dwordx3 v[56:58], v26, s[52:55], 0 offen offset:36 // (op 2153) Load value




v_mov_b32 v31, v22 // (op 2156) reg expression

buffer_load_dword v35, v31, s[68:71], 0 offen // (op 2156) Load value

v_and_b32 v31, -8, v1 // (op 2715) 
v_sub_u32 v39, v1, v31 // (op 2715) 

v_lshlrev_b32 v31, 4, v39 // (op 2715) 

v_lshrrev_b32 v39, 3, v1 // (op 2715) 
v_lshl_add_u32 v43, v39, 7, v31 // (op 2715) 


s_nop 0
v_readfirstlane_b32 s0, v43 // (op 2715) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 5 // (op 2391) 
v_mov_b64 v[24:25], s[18:19] // (op 2391) convert


v_mov_b64 v[26:27], v[18:19] // (op 2391) 
s_mov_b32 m0, s0 // (op 2391) reg expression
buffer_load_dwordx4 v26, s[20:23], 0 offen offset: 0 lds // (op 2391) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 2391) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 2391) most significant half

s_add_u32 m0, m0, 4096 // (op 2391) 
buffer_load_dwordx4 v26, s[20:23], 0 offen offset: 0 lds // (op 2391) Load value direct to lds






buffer_load_ubyte v31, v14, s[64:67], 0 offen offset: 0 // (op 2162) Load value

s_mul_i32 s1, s29, 6 // (op 2165) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 2165) least significant
s_mul_hi_u32 s18, s28, 6 // (op 2165) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2165) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 2165) 


s_lshl_b64 s[0:1], s[28:29], 7 // (op 2165) 
s_mul_i32 s19, s1, 6 // (op 2165) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 2165) least significant
s_mul_hi_u32 s49, s0, 6 // (op 2165) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 2165) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 2165) 

v_add_u32 v39, v16, s0 // (op 2165) 

buffer_load_dwordx3 v[60:62], v39, s[52:55], 0 offen // (op 2165) Load value
buffer_load_dwordx3 v[64:66], v39, s[52:55], 0 offen offset:12 // (op 2165) Load value
buffer_load_dwordx3 v[68:70], v39, s[52:55], 0 offen offset:24 // (op 2165) Load value
buffer_load_dwordx3 v[72:74], v39, s[52:55], 0 offen offset:36 // (op 2165) Load value
v_add_u32 v39, v39, v24 // (op 2165) 
buffer_load_dwordx3 v[76:78], v39, s[52:55], 0 offen // (op 2165) Load value
buffer_load_dwordx3 v[80:82], v39, s[52:55], 0 offen offset:12 // (op 2165) Load value
buffer_load_dwordx3 v[84:86], v39, s[52:55], 0 offen offset:24 // (op 2165) Load value
buffer_load_dwordx3 v[88:90], v39, s[52:55], 0 offen offset:36 // (op 2165) Load value





s_lshl_b32 s0, s38, 2 // (op 2168) 
v_add_u32 v16, v22, s0 // (op 2168) 


buffer_load_dword v39, v16, s[68:71], 0 offen // (op 2168) Load value


v_and_b32 v16, -8, v1 // (op 2941) 
v_sub_u32 v22, v1, v16 // (op 2941) 

v_lshlrev_b32 v16, 4, v22 // (op 2941) 

v_lshrrev_b32 v22, 3, v1 // (op 2941) 
v_lshl_add_u32 v43, v22, 7, v16 // (op 2941) 


s_nop 0
v_readfirstlane_b32 s0, v43 // (op 2941) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 5 // (op 2396) 
v_mov_b64 v[24:25], s[18:19] // (op 2396) convert


v_mov_b64 v[26:27], v[18:19] // (op 2396) 
s_add_u32 s1, 8192, s0 // (op 2396) 
s_mov_b32 m0, s1 // (op 2396) reg expression
s_mov_b32 s18, 128 // (op 2396) call()
buffer_load_dwordx4 v26, s[20:23], s18 offen offset: 0 lds // (op 2396) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v24 // (op 2396) least significant half
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 2396) most significant half

s_add_u32 m0, m0, 4096 // (op 2396) 
buffer_load_dwordx4 v26, s[20:23], s18 offen offset: 0 lds // (op 2396) Load value direct to lds









buffer_load_ubyte v16, v14, s[64:67], 0 offen offset:4 // (op 2174) Load value

v_and_b32 v14, -4, v1 // (op 2768) 
v_sub_u32 v22, v1, v14 // (op 2768) 

v_lshlrev_b32 v14, 6, v22 // (op 2768) 

v_lshrrev_b32 v22, 2, v1 // (op 2768) 
v_lshl_add_u32 v43, v22, 9, v14 // (op 2768) 


v_mul_lo_u32 v14, v43, 6 // (op 2768) 
v_lshrrev_b32 v22, 3, v14 // (op 2768) 

v_lshrrev_b32 v14, 4, v43 // (op 2768) 

v_lshlrev_b32 v43, 2, v14 // (op 2768) 

v_add_u32 v14, v22, v43 // (op 2768) 





s_waitcnt vmcnt(23)
ds_write_b96 v14, v[28:30] offset:16384 // (op 2176) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v14, v[32:34] offset:16400 // (op 2176) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v14, v[36:38] offset:16416 // (op 2176) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v14, v[40:42] offset:16432 // (op 2176) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v14, v[44:46] offset:16640 // (op 2176) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v14, v[48:50] offset:16656 // (op 2176) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v14, v[52:54] offset:16672 // (op 2176) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v14, v[56:58] offset:16688 // (op 2176) Store local data 

v_and_b32 v14, -64, v1 // (op 2879) 
v_sub_u32 v22, v1, v14 // (op 2879) 

v_lshlrev_b32 v14, 2, v22 // (op 2879) 

v_lshrrev_b32 v22, 6, v1 // (op 2879) 
v_lshl_add_u32 v43, v22, 8, v14 // (op 2879) 





s_waitcnt vmcnt(15)
ds_write_b32 v43, v35 offset:49152 // (op 2177) Store local data 

v_and_b32 v14, -4, v1 // (op 2826) 
v_sub_u32 v22, v1, v14 // (op 2826) 

v_and_b32 v14, 4294967292, v1 // (op 2826) 
v_add_u32 v43, v14, v22 // (op 2826) 





s_waitcnt vmcnt(12)
ds_write_b8 v43, v31 offset:50176 // (op 2179) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 3117) 



ds_read_b96_tr_b6 v[92:94], v9 offset:16384 // (op 2197) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v9 offset:20480 // (op 2197) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3158) 


ds_read_u8 v14, v6 offset:49152 // (op 2220) Load local data 



ds_read_b96_tr_b6 v[98:100], v9 offset:16448 // (op 2202) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v9 offset:20544 // (op 2202) Transpose load local data 


ds_read_u8 v22, v6 offset:49216 // (op 2225) Load local data 



ds_read_b96_tr_b6 v[104:106], v9 offset:16512 // (op 2207) Transpose load local data 
ds_read_b96_tr_b6 v[107:109], v9 offset:20608 // (op 2207) Transpose load local data 


ds_read_u8 v43, v6 offset:49280 // (op 2230) Load local data 



ds_read_b96_tr_b6 v[110:112], v9 offset:16576 // (op 2212) Transpose load local data 
ds_read_b96_tr_b6 v[113:115], v9 offset:20672 // (op 2212) Transpose load local data 


ds_read_u8 v47, v6 offset:49344 // (op 2235) Load local data 
s_barrier  // (op 3098) 



ds_read_b128 v[116:119], v3 // (op 2193) Load local data 
ds_read_b128 v[120:123], v3 offset:32 // (op 2193) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3138) 



ds_read_u8 v51, v0 offset:50176 // (op 2216) Load local data 
v_and_b32 v55, -8, v1 // (op 2687) 
v_sub_u32 v59, v1, v55 // (op 2687) 

v_lshlrev_b32 v55, 4, v59 // (op 2687) 

v_lshrrev_b32 v59, 3, v1 // (op 2687) 
v_lshl_add_u32 v63, v59, 7, v55 // (op 2687) 


s_nop 0
v_readfirstlane_b32 s0, v63 // (op 2687) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v55, -4, v1 // (op 2739) 
v_sub_u32 v59, v1, v55 // (op 2739) 

v_lshlrev_b32 v55, 6, v59 // (op 2739) 

v_lshrrev_b32 v59, 2, v1 // (op 2739) 
v_lshl_add_u32 v63, v59, 9, v55 // (op 2739) 


v_mul_lo_u32 v55, v63, 6 // (op 2739) 
v_lshrrev_b32 v59, 3, v55 // (op 2739) 

v_lshrrev_b32 v55, 4, v63 // (op 2739) 

v_lshlrev_b32 v63, 2, v55 // (op 2739) 

v_add_u32 v55, v59, v63 // (op 2739) 


v_and_b32 v59, -4, v1 // (op 2797) 
v_sub_u32 v63, v1, v59 // (op 2797) 

v_and_b32 v59, 4294967292, v1 // (op 2797) 
v_add_u32 v67, v59, v63 // (op 2797) 


v_and_b32 v59, -64, v1 // (op 2850) 
v_sub_u32 v63, v1, v59 // (op 2850) 

v_lshlrev_b32 v59, 2, v63 // (op 2850) 

v_lshrrev_b32 v63, 6, v1 // (op 2850) 
v_lshl_add_u32 v71, v63, 8, v59 // (op 2850) 


v_and_b32 v59, -8, v1 // (op 2926) 
v_sub_u32 v63, v1, v59 // (op 2926) 

v_lshlrev_b32 v59, 4, v63 // (op 2926) 

v_lshrrev_b32 v63, 3, v1 // (op 2926) 
v_lshl_add_u32 v75, v63, 7, v59 // (op 2926) 


s_nop 0
v_readfirstlane_b32 s1, v75 // (op 2926) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v59, -4, v1 // (op 2965) 
v_sub_u32 v63, v1, v59 // (op 2965) 

v_lshlrev_b32 v59, 6, v63 // (op 2965) 

v_lshrrev_b32 v63, 2, v1 // (op 2965) 
v_lshl_add_u32 v75, v63, 9, v59 // (op 2965) 


v_mul_lo_u32 v59, v75, 6 // (op 2965) 
v_lshrrev_b32 v63, 3, v59 // (op 2965) 

v_lshrrev_b32 v59, 4, v75 // (op 2965) 

v_lshlrev_b32 v75, 2, v59 // (op 2965) 

v_add_u32 v59, v63, v75 // (op 2965) 


v_and_b32 v63, -4, v1 // (op 2994) 
v_sub_u32 v75, v1, v63 // (op 2994) 

v_and_b32 v63, 4294967292, v1 // (op 2994) 
v_add_u32 v79, v63, v75 // (op 2994) 


v_and_b32 v63, -64, v1 // (op 3018) 
v_sub_u32 v75, v1, v63 // (op 3018) 

v_lshlrev_b32 v63, 2, v75 // (op 3018) 

v_lshrrev_b32 v75, 6, v1 // (op 3018) 
v_lshl_add_u32 v83, v75, 8, v63 // (op 3018) 


s_mov_b32 s18, 0 // (op 868) call()
s_ashr_i64 s[50:51], s[32:33], 7 // (op 867) 
s_mov_b32 s19, s50 // (op 867) convert

s_and_b32 s49, s19, 4294967294 // (op 867) 

s_cmp_lt_u32 s18, s49 // (op 867) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_867 // (op 867) Condition: Top (jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_867 if false)
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_867:

s_add_u32 s18, s18, 2 // (op 869) 
s_mul_i32 s51, s29, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s28, 6 // (op 10) least significant
s_mul_hi_u32 s19, s28, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 10) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[50:51] // (op 10) 


v_mov_b32 v63, v17 // (op 10) 
s_lshl_b64 s[50:51], s[28:29], 7 // (op 10) 
s_mul_i32 s73, s51, 6 // (op 10) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 10) least significant
s_mul_hi_u32 s19, s50, 6 // (op 10) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 10) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 10) 

s_lshl_b32 s19, s50, 1 // (op 10) 

v_add_u32 v75, v63, s19 // (op 10) 

buffer_load_dwordx3 v[28:30], v75, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[32:34], v75, s[52:55], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[36:38], v75, s[52:55], 0 offen offset:24 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[116:123], v[92:97], a[0:15], v51, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 88) 
buffer_load_dwordx3 v[40:42], v75, s[52:55], 0 offen offset:36 // (op 10) Load value
s_barrier  // (op 3107) 
v_add_u32 v75, v75, v18 // (op 10) 



buffer_load_dwordx3 v[44:46], v75, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[48:50], v75, s[52:55], 0 offen offset:12 // (op 10) Load value
ds_read_b128 v[124:127], v3 offset:64 // (op 91) Load local data 
ds_read_b128 v[128:131], v3 offset:96 // (op 91) Load local data 
buffer_load_dwordx3 v[52:54], v75, s[52:55], 0 offen offset:24 // (op 10) Load value
s_barrier  // (op 3127) 
buffer_load_dwordx3 v[56:58], v75, s[52:55], 0 offen offset:36 // (op 10) Load value







ds_read_b96_tr_b6 v[92:94], v9 offset:32768 // (op 95) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v9 offset:36864 // (op 95) Transpose load local data 
v_mov_b32 v75, v23 // (op 25) 
s_barrier  // (op 3148) 
s_lshl_b32 s19, s38, 3 // (op 25) 



ds_read_u8 v87, v0 offset:50178 // (op 99) Load local data 
v_add_u32 v91, v75, s19 // (op 25) 


s_barrier  // (op 3168) 
buffer_load_dword v35, v91, s[68:71], 0 offen // (op 25) Load value



s_waitcnt lgkmcnt(3)
s_barrier  // (op 3101) 
ds_read_u8 v14, v6 offset:49664 // (op 103) Load local data 

s_lshl_b64 s[50:51], s[24:25], 5 // (op 2375) 
v_mov_b64 v[18:19], s[50:51] // (op 2375) convert


v_mov_b64 v[24:25], v[20:21] // (op 2375) 
v_mov_b64 v[26:27], v[24:25] // (op 2375) 
s_mov_b32 m0, s0 // (op 2375) reg expression
s_mov_b32 s19, 256 // (op 2375) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[124:131], v[92:97], a[0:15], v87, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 105) 
buffer_load_dwordx4 v26, s[20:23], s19 offen offset: 0 lds // (op 2375) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v18 // (op 2375) least significant half
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[116:123], v[98:103], a[16:31], v51, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 447) 
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 2375) most significant half




s_add_u32 m0, m0, 4096 // (op 2375) 
ds_read_b96_tr_b6 v[92:94], v9 offset:32832 // (op 451) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v9 offset:36928 // (op 451) Transpose load local data 
buffer_load_dwordx4 v26, s[20:23], s19 offen offset: 0 lds // (op 2375) Load value direct to lds







ds_read_u8 v14, v6 offset:49728 // (op 455) Load local data 
v_mov_b32 v91, v15 // (op 17) 

buffer_load_ubyte v31, v91, s[64:67], 0 offen offset:8 // (op 17) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[124:131], v[92:97], a[16:31], v87, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 456) 
s_barrier  // (op 3188) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[116:123], v[104:109], a[32:47], v51, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 681) 
s_waitcnt vmcnt(23)
ds_write_b96 v59, v[60:62] offset:50432 // (op 302) Store local data 



ds_read_b96_tr_b6 v[92:94], v9 offset:32896 // (op 685) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v9 offset:36992 // (op 685) Transpose load local data 
s_waitcnt vmcnt(22)
ds_write_b96 v59, v[64:66] offset:50448 // (op 302) Store local data 


ds_read_u8 v14, v6 offset:49792 // (op 689) Load local data 
s_waitcnt vmcnt(21)
ds_write_b96 v59, v[68:70] offset:50464 // (op 302) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[124:131], v[92:97], a[32:47], v87, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 690) 
s_waitcnt vmcnt(20)
ds_write_b96 v59, v[72:74] offset:50480 // (op 302) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v59, v[76:78] offset:50688 // (op 302) Store local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[116:123], v[110:115], a[48:63], v51, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 915) 
s_waitcnt vmcnt(18)
ds_write_b96 v59, v[80:82] offset:50704 // (op 302) Store local data 



ds_read_b96_tr_b6 v[110:112], v9 offset:32960 // (op 919) Transpose load local data 
ds_read_b96_tr_b6 v[113:115], v9 offset:37056 // (op 919) Transpose load local data 
s_waitcnt vmcnt(17)
ds_write_b96 v59, v[84:86] offset:50720 // (op 302) Store local data 


ds_read_u8 v47, v6 offset:49856 // (op 923) Load local data 
s_waitcnt vmcnt(16)
ds_write_b96 v59, v[88:90] offset:50736 // (op 302) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[124:131], v[110:115], a[48:63], v87, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 924) 


s_barrier  // (op 3200) 



v_add_u32 v87, 83200, v83 // (op 304) 
s_waitcnt vmcnt(15)
ds_write_b32 v87, v39 // (op 304) Store local data 

s_barrier  // (op 3194) 



v_add_u32 v87, 84224, v79 // (op 303) 
s_waitcnt vmcnt(12)
ds_write_b8 v87, v16 // (op 303) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 3184) 



ds_read_b96_tr_b6 v[92:94], v13 offset:50432 // (op 287) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v13 offset:54528 // (op 287) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3197) 


v_add_u32 v87, 83200, v12 // (op 291) 
ds_read_u8 v14, v87 // (op 291) Load local data 




ds_read_b96_tr_b6 v[98:100], v13 offset:50496 // (op 470) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v13 offset:54592 // (op 470) Transpose load local data 


v_add_u32 v87, 83264, v12 // (op 474) 
ds_read_u8 v22, v87 // (op 474) Load local data 




ds_read_b96_tr_b6 v[104:106], v13 offset:50560 // (op 704) Transpose load local data 
ds_read_b96_tr_b6 v[107:109], v13 offset:54656 // (op 704) Transpose load local data 


v_add_u32 v87, 83328, v12 // (op 708) 
ds_read_u8 v43, v87 // (op 708) Load local data 




ds_read_b96_tr_b6 v[124:126], v13 offset:50624 // (op 938) Transpose load local data 
ds_read_b96_tr_b6 v[127:129], v13 offset:54720 // (op 938) Transpose load local data 


v_add_u32 v87, 83392, v12 // (op 942) 
ds_read_u8 v18, v87 // (op 942) Load local data 

s_barrier  // (op 3178) 



ds_read_b128 v[116:119], v5 offset:8192 // (op 285) Load local data 
ds_read_b128 v[120:123], v5 offset:8224 // (op 285) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3191) 



v_add_u32 v19, 84224, v4 // (op 289) 
ds_read_u8 v51, v19 // (op 289) Load local data 


s_mul_i32 s51, s29, 6 // (op 305) most significant: low of high * low
s_mul_i32 s50, s28, 6 // (op 305) least significant
s_mul_hi_u32 s19, s28, 6 // (op 305) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 305) most significant: sum

v_lshrrev_b64 v[26:27], 3, s[50:51] // (op 305) 


s_lshl_b64 s[50:51], s[28:29], 7 // (op 305) 
s_mul_i32 s73, s51, 6 // (op 305) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 305) least significant
s_mul_hi_u32 s19, s50, 6 // (op 305) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 305) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 305) 

s_mul_i32 s19, 3, s50 // (op 305) 

v_add_u32 v19, v63, s19 // (op 305) 

buffer_load_dwordx3 v[60:62], v19, s[52:55], 0 offen // (op 305) Load value
buffer_load_dwordx3 v[64:66], v19, s[52:55], 0 offen offset:12 // (op 305) Load value
buffer_load_dwordx3 v[68:70], v19, s[52:55], 0 offen offset:24 // (op 305) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[116:123], v[92:97], a[0:15], v51, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 292) 
buffer_load_dwordx3 v[72:74], v19, s[52:55], 0 offen offset:36 // (op 305) Load value
v_add_u32 v19, v19, v26 // (op 305) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[116:123], v[98:103], a[16:31], v51, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 457) 
buffer_load_dwordx3 v[76:78], v19, s[52:55], 0 offen // (op 305) Load value
buffer_load_dwordx3 v[80:82], v19, s[52:55], 0 offen offset:12 // (op 305) Load value
ds_read_b128 v[130:133], v5 offset:8256 // (op 294) Load local data 
ds_read_b128 v[134:137], v5 offset:8288 // (op 294) Load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[116:123], v[104:109], a[32:47], v51, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 691) 



v_add_u32 v87, 66816, v13 // (op 296) 
ds_read_b96_tr_b6 v[138:140], v87 // (op 296) Transpose load local data 

v_add_u32 v87, 70912, v13 // (op 296) 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[116:123], v[124:129], a[48:63], v51, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 925) 
ds_read_b96_tr_b6 v[141:143], v87 // (op 296) Transpose load local data 






v_add_u32 v18, 84226, v4 // (op 298) 
ds_read_u8 v87, v18 // (op 298) Load local data 

buffer_load_dwordx3 v[84:86], v19, s[52:55], 0 offen offset:24 // (op 305) Load value
buffer_load_dwordx3 v[88:90], v19, s[52:55], 0 offen offset:36 // (op 305) Load value





s_barrier  // (op 3239) 


s_lshl_b32 s19, s38, 2 // (op 307) 
s_mul_i32 s49, 3, s19 // (op 307) 

v_add_u32 v63, 83712, v12 // (op 300) 
ds_read_u8 v18, v63 // (op 300) Load local data 

v_add_u32 v19, v75, s49 // (op 307) 





buffer_load_dword v39, v19, s[68:71], 0 offen // (op 307) Load value

v_add_u32 v19, 66880, v13 // (op 461) 

s_lshl_b64 s[50:51], s[28:29], 7 // (op 2513) 
s_lshl_b32 s19, s38, 3 // (op 2641) 
v_add_u32 v23, v23, s19 // (op 2641) 

s_mul_i32 s73, s51, 6 // (op 2513) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 2513) least significant
s_mul_hi_u32 s19, s50, 6 // (op 2513) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 2513) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 2513) 

s_lshl_b32 s19, s50, 1 // (op 2513) 

v_add_u32 v17, v17, s19 // (op 2513) 

ds_read_b96_tr_b6 v[124:126], v19 // (op 461) Transpose load local data 

v_add_u32 v19, 70976, v13 // (op 461) 
ds_read_b96_tr_b6 v[127:129], v19 // (op 461) Transpose load local data 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[130:137], v[138:143], a[0:15], v87, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 301) 




s_barrier  // (op 3181) 
v_add_u32 v63, 83776, v12 // (op 465) 

ds_read_u8 v75, v63 // (op 465) Load local data 

s_lshl_b64 s[50:51], s[24:25], 5 // (op 2386) 



v_add_u32 v63, 66944, v13 // (op 695) 
v_mov_b64 v[18:19], s[50:51] // (op 2386) convert


v_mov_b64 v[26:27], v[24:25] // (op 2386) 
s_add_u32 s19, 8192, s1 // (op 2386) 
s_mov_b32 m0, s19 // (op 2386) reg expression
s_mov_b32 s49, 384 // (op 2386) call()
ds_read_b96_tr_b6 v[138:140], v63 // (op 695) Transpose load local data 

v_add_u32 v63, 71040, v13 // (op 695) 
ds_read_b96_tr_b6 v[141:143], v63 // (op 695) Transpose load local data 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[130:137], v[124:129], a[16:31], v87, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 466) 




v_add_u32 v63, 83840, v12 // (op 699) 
ds_read_u8 v75, v63 // (op 699) Load local data 

buffer_load_dwordx4 v26, s[20:23], s49 offen offset: 0 lds // (op 2386) Load value direct to lds



v_add_u32 v63, 67008, v13 // (op 929) 
v_add_co_u32 v26, s[50:51], v26, v18 // (op 2386) least significant half
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 2386) most significant half

s_add_u32 m0, m0, 4096 // (op 2386) 
buffer_load_dwordx4 v26, s[20:23], s49 offen offset: 0 lds // (op 2386) Load value direct to lds




ds_read_b96_tr_b6 v[124:126], v63 // (op 929) Transpose load local data 


v_add_u32 v63, 71104, v13 // (op 929) 



ds_read_b96_tr_b6 v[127:129], v63 // (op 929) Transpose load local data 

v_add_co_u32 v20, vcc, 256, v20 // (op 2449) least significant half
buffer_load_ubyte v16, v91, s[64:67], 0 offen offset:12 // (op 306) Load value



s_barrier  // (op 3297) 
s_barrier  // (op 3120) 
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 2449) most significant half



v_add_u32 v15, 8, v15 // (op 2577) 
v_add_u32 v63, 83904, v12 // (op 933) 
ds_read_u8 v91, v63 // (op 933) Load local data 

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[130:137], v[138:143], a[32:47], v87, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 700) 


s_barrier  // (op 3301) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[130:137], v[124:129], a[48:63], v87, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 934) 
s_waitcnt vmcnt(23)
ds_write_b96 v55, v[28:30] offset:16384 // (op 13) Store local data 




s_waitcnt vmcnt(22)
ds_write_b96 v55, v[32:34] offset:16400 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v55, v[36:38] offset:16416 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v55, v[40:42] offset:16432 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v55, v[44:46] offset:16640 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v55, v[48:50] offset:16656 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v55, v[52:54] offset:16672 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v55, v[56:58] offset:16688 // (op 13) Store local data 
s_barrier  // (op 3161) 



s_waitcnt vmcnt(15)
ds_write_b32 v71, v35 offset:49152 // (op 30) Store local data 
s_barrier  // (op 3141) 



s_waitcnt vmcnt(12)
ds_write_b8 v67, v31 offset:50176 // (op 21) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3130) 



ds_read_b96_tr_b6 v[92:94], v9 offset:16384 // (op 75) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v9 offset:20480 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3171) 


ds_read_u8 v14, v6 offset:49152 // (op 85) Load local data 



ds_read_b96_tr_b6 v[98:100], v9 offset:16448 // (op 442) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v9 offset:20544 // (op 442) Transpose load local data 


ds_read_u8 v22, v6 offset:49216 // (op 446) Load local data 



ds_read_b96_tr_b6 v[104:106], v9 offset:16512 // (op 676) Transpose load local data 
ds_read_b96_tr_b6 v[107:109], v9 offset:20608 // (op 676) Transpose load local data 


ds_read_u8 v43, v6 offset:49280 // (op 680) Load local data 



ds_read_b96_tr_b6 v[110:112], v9 offset:16576 // (op 910) Transpose load local data 
ds_read_b96_tr_b6 v[113:115], v9 offset:20672 // (op 910) Transpose load local data 


ds_read_u8 v47, v6 offset:49344 // (op 914) Load local data 
s_barrier  // (op 3110) 



ds_read_b128 v[116:119], v3 // (op 70) Load local data 
ds_read_b128 v[120:123], v3 offset:32 // (op 70) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3151) 



ds_read_u8 v51, v0 offset:50176 // (op 80) Load local data 
s_ashr_i64 s[50:51], s[32:33], 7 // (op 867) 
s_mov_b32 s19, s50 // (op 867) convert

s_and_b32 s49, s19, 4294967294 // (op 867) 

s_cmp_lt_u32 s18, s49 // (op 867) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_867 // (op 867) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_867 if true)
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_867:
















s_barrier  // (op 3279) 
s_mul_hi_i32 s0, s16, s11 // (op 2409) 
v_lshrrev_b32 v15, 3, v1 // (op 2409) 
s_add_i32 s1, s0, s16 // (op 2409) 

v_and_b32 v63, -8, v1 // (op 2409) 
s_ashr_i32 s0, s1, 31 // (op 2409) 
v_sub_u32 v67, v1, v63 // (op 2409) 

s_cmp_eq_i32 s11, 0 // (op 2409) 
s_mov_b32 s18, scc // (op 2409) 
v_lshlrev_b32 v63, 4, v67 // (op 2409) 

s_cmp_eq_i32 s18, 0 // (op 2409) 
s_cselect_b32 s19, 0, -1 // (op 2409) 

s_and_b32 s18, s12, 31 // (op 2409) 
s_lshl_b32 s49, 1, s18 // (op 2409) 
s_add_i32 s50, s49, s19 // (op 2409) 


s_and_b32 s19, s0, s50 // (op 2409) 


s_add_i32 s0, s1, s19 // (op 2409) 


s_ashr_i32 s1, s0, s18 // (op 2409) 


s_lshl_b32 s0, s12, 24 // (op 2409) 
s_ashr_i32 s18, s0, 31 // (op 2409) 

s_xor_b32 s0, s1, s18 // (op 2409) 

s_sub_i32 s1, s0, s18 // (op 2409) 


s_mul_hi_i32 s0, s1, s13 // (op 2409) 
s_add_i32 s18, s0, s1 // (op 2409) 

s_ashr_i32 s0, s18, 31 // (op 2409) 
s_cmp_eq_i32 s13, 0 // (op 2409) 
s_mov_b32 s19, scc // (op 2409) 
s_cmp_eq_i32 s19, 0 // (op 2409) 
s_cselect_b32 s49, 0, -1 // (op 2409) 

s_and_b32 s19, s14, 31 // (op 2409) 
s_lshl_b32 s50, 1, s19 // (op 2409) 
s_add_i32 s51, s50, s49 // (op 2409) 


s_and_b32 s49, s0, s51 // (op 2409) 


s_add_i32 s0, s18, s49 // (op 2409) 


s_ashr_i32 s18, s0, s19 // (op 2409) 


s_lshl_b32 s0, s14, 24 // (op 2409) 
s_ashr_i32 s19, s0, 31 // (op 2409) 

s_xor_b32 s0, s18, s19 // (op 2409) 

s_sub_i32 s18, s0, s19 // (op 2409) 


s_mul_i32 s0, s1, s44 // (op 2409) 

s_sub_u32 s1, s16, s0 // (op 2409) 

s_mul_i32 s0, s18, s44 // (op 2409) 
s_add_u32 s0, s0, s1 // (op 2409) 


s_mul_hi_i32 s1, s16, s10 // (op 2409) 
s_add_i32 s18, s1, s16 // (op 2409) 

s_ashr_i32 s1, s18, 31 // (op 2409) 
s_cmp_eq_i32 s10, 0 // (op 2409) 
s_mov_b32 s19, scc // (op 2409) 
s_cmp_eq_i32 s19, 0 // (op 2409) 
s_cselect_b32 s49, 0, -1 // (op 2409) 

s_and_b32 s19, s47, 31 // (op 2409) 
s_lshl_b32 s50, 1, s19 // (op 2409) 
s_add_i32 s51, s50, s49 // (op 2409) 


s_and_b32 s49, s1, s51 // (op 2409) 


s_add_i32 s1, s18, s49 // (op 2409) 


s_ashr_i32 s18, s1, s19 // (op 2409) 


s_lshl_b32 s1, s47, 24 // (op 2409) 
s_ashr_i32 s19, s1, 31 // (op 2409) 

s_xor_b32 s1, s18, s19 // (op 2409) 

s_sub_i32 s18, s1, s19 // (op 2409) 


s_add_u32 s50, s6, 256 // (op 2409) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2409) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2409) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2409) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 2409) 

s_mul_i32 s1, s44, s50 // (op 2409) 

s_mul_i32 s19, s48, s1 // (op 2409) 

s_mul_i32 s1, s18, s19 // (op 2409) 

s_sub_u32 s19, s16, s1 // (op 2409) 

s_mov_b32 s1, s19 // (op 2409) 
s_mul_hi_i32 s49, s1, s15 // (op 2409) 
s_add_i32 s50, s49, s1 // (op 2409) 


s_ashr_i32 s1, s50, 31 // (op 2409) 
s_cmp_eq_i32 s15, 0 // (op 2409) 
s_mov_b32 s49, scc // (op 2409) 
s_cmp_eq_i32 s49, 0 // (op 2409) 
s_cselect_b32 s51, 0, -1 // (op 2409) 

s_and_b32 s49, s46, 31 // (op 2409) 
s_lshl_b32 s72, 1, s49 // (op 2409) 
s_add_i32 s73, s72, s51 // (op 2409) 


s_and_b32 s51, s1, s73 // (op 2409) 


s_add_i32 s1, s50, s51 // (op 2409) 


s_ashr_i32 s50, s1, s49 // (op 2409) 


s_lshl_b32 s1, s46, 24 // (op 2409) 
s_ashr_i32 s49, s1, 31 // (op 2409) 

s_xor_b32 s1, s50, s49 // (op 2409) 

s_sub_i32 s50, s1, s49 // (op 2409) 


s_mul_i32 s1, s45, s44 // (op 2409) 
s_add_u32 s72, s8, 64 // (op 2409) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 2409) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 2409) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 2409) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 2409) 

s_sub_i32 s49, s72, s1 // (op 2409) 

s_mul_i32 s51, s50, s49 // (op 2409) 


s_sub_u32 s49, s19, s51 // (op 2409) 


s_add_u32 s19, s49, s1 // (op 2409) 


s_cmp_eq_i32 s18, 0 // (op 2409) 
s_mov_b32 s1, scc // (op 2409) 

s_cmp_eq_i32 s1, 0 // (op 2409) 
s_cselect_b32 s18, s19, s0 // (op 2409) 



v_lshl_add_u32 v67, s18, 6, v15 // (op 2409) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 2409) 
s_lshr_b32 s18, s0, 1 // (op 2409) 

v_lshl_add_u32 v15, s18, 8, v63 // (op 2409) 


v_mov_b32 v21, 0 // (op 2409) convert
v_mov_b32 v20, v15 // (op 2409) convert

v_mul_lo_u32 v19, v67, s25 // (op 2409) most significant: low of low * high
v_mul_hi_u32 v15, v67, s24 // (op 2409) most significant: high of low * low
v_mul_lo_u32 v18, v67, s24 // (op 2409) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 2409) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v20 // (op 2409) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 2409) most significant half



s_mul_i32 s1, s24, s9 // (op 2411) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 2411) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 2411) least significant
s_mul_hi_u32 s19, s24, s8 // (op 2411) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2411) most significant: sum
s_add_u32 s1, s1, s19 // (op 2411) most significant: sum


s_mov_b32 s20, s26 // (op 2411) reg expression
s_mov_b32 s21, s27 // (op 2411) reg expression
s_mov_b32 s22, s0 // (op 2411) reg expression
s_mov_b32 s23, 131072 // (op 2411) call()

s_mul_hi_i32 s0, s16, s11 // (op 2479) 
v_and_b32 v15, -4, v1 // (op 2479) 
s_add_i32 s1, s0, s16 // (op 2479) 

v_sub_u32 v17, v1, v15 // (op 2479) 

s_ashr_i32 s0, s1, 31 // (op 2479) 
v_lshlrev_b32 v15, 6, v17 // (op 2479) 

s_cmp_eq_i32 s11, 0 // (op 2479) 
s_mov_b32 s18, scc // (op 2479) 
v_lshrrev_b32 v17, 2, v1 // (op 2479) 
s_cmp_eq_i32 s18, 0 // (op 2479) 
s_cselect_b32 s19, 0, -1 // (op 2479) 

v_lshlrev_b32 v63, 1, v17 // (op 2479) 

s_and_b32 s18, s12, 31 // (op 2479) 
s_lshl_b32 s49, 1, s18 // (op 2479) 
s_add_i32 s50, s49, s19 // (op 2479) 


s_and_b32 s19, s0, s50 // (op 2479) 


s_add_i32 s0, s1, s19 // (op 2479) 


s_ashr_i32 s1, s0, s18 // (op 2479) 


s_lshl_b32 s0, s12, 24 // (op 2479) 
s_ashr_i32 s18, s0, 31 // (op 2479) 

s_xor_b32 s0, s1, s18 // (op 2479) 

s_sub_i32 s1, s0, s18 // (op 2479) 


s_mul_hi_i32 s0, s1, s13 // (op 2479) 
s_add_i32 s18, s0, s1 // (op 2479) 

s_ashr_i32 s0, s18, 31 // (op 2479) 
s_cmp_eq_i32 s13, 0 // (op 2479) 
s_mov_b32 s19, scc // (op 2479) 
s_cmp_eq_i32 s19, 0 // (op 2479) 
s_cselect_b32 s49, 0, -1 // (op 2479) 

s_and_b32 s19, s14, 31 // (op 2479) 
s_lshl_b32 s50, 1, s19 // (op 2479) 
s_add_i32 s51, s50, s49 // (op 2479) 


s_and_b32 s49, s0, s51 // (op 2479) 


s_add_i32 s0, s18, s49 // (op 2479) 


s_ashr_i32 s18, s0, s19 // (op 2479) 


s_lshl_b32 s0, s14, 24 // (op 2479) 
s_ashr_i32 s19, s0, 31 // (op 2479) 

s_xor_b32 s0, s18, s19 // (op 2479) 

s_sub_i32 s18, s0, s19 // (op 2479) 


s_add_u32 s50, s6, 256 // (op 2479) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2479) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2479) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2479) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 2479) 

s_mul_i32 s0, s18, s50 // (op 2479) 

s_sub_i32 s49, s1, s0 // (op 2479) 


s_mul_hi_i32 s0, s16, s10 // (op 2479) 
s_add_i32 s1, s0, s16 // (op 2479) 

s_ashr_i32 s0, s1, 31 // (op 2479) 
s_cmp_eq_i32 s10, 0 // (op 2479) 
s_mov_b32 s18, scc // (op 2479) 
s_cmp_eq_i32 s18, 0 // (op 2479) 
s_cselect_b32 s19, 0, -1 // (op 2479) 

s_and_b32 s18, s47, 31 // (op 2479) 
s_lshl_b32 s72, 1, s18 // (op 2479) 
s_add_i32 s73, s72, s19 // (op 2479) 


s_and_b32 s19, s0, s73 // (op 2479) 


s_add_i32 s0, s1, s19 // (op 2479) 


s_ashr_i32 s1, s0, s18 // (op 2479) 


s_lshl_b32 s0, s47, 24 // (op 2479) 
s_ashr_i32 s18, s0, 31 // (op 2479) 

s_xor_b32 s0, s1, s18 // (op 2479) 

s_sub_i32 s1, s0, s18 // (op 2479) 


s_mul_i32 s0, s44, s50 // (op 2479) 

s_mul_i32 s18, s48, s0 // (op 2479) 

s_mul_i32 s0, s1, s18 // (op 2479) 

s_sub_u32 s18, s16, s0 // (op 2479) 

s_mul_hi_i32 s0, s18, s15 // (op 2479) 
s_add_i32 s19, s0, s18 // (op 2479) 


s_ashr_i32 s0, s19, 31 // (op 2479) 
s_cmp_eq_i32 s15, 0 // (op 2479) 
s_mov_b32 s18, scc // (op 2479) 
s_cmp_eq_i32 s18, 0 // (op 2479) 
s_cselect_b32 s50, 0, -1 // (op 2479) 

s_and_b32 s18, s46, 31 // (op 2479) 
s_lshl_b32 s51, 1, s18 // (op 2479) 
s_add_i32 s72, s51, s50 // (op 2479) 


s_and_b32 s50, s0, s72 // (op 2479) 


s_add_i32 s0, s19, s50 // (op 2479) 


s_ashr_i32 s19, s0, s18 // (op 2479) 


s_lshl_b32 s0, s46, 24 // (op 2479) 
s_ashr_i32 s18, s0, 31 // (op 2479) 

s_xor_b32 s0, s19, s18 // (op 2479) 

s_sub_i32 s19, s0, s18 // (op 2479) 


s_cmp_eq_i32 s1, 0 // (op 2479) 
s_mov_b32 s0, scc // (op 2479) 

s_cmp_eq_i32 s0, 0 // (op 2479) 
s_cselect_b32 s1, s19, s49 // (op 2479) 



v_lshl_add_u32 v17, s1, 8, v15 // (op 2479) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 2479) 
v_mov_b32 v21, 0 // (op 2479) convert
v_mov_b32 v20, v17 // (op 2479) convert

s_lshr_b32 s18, s0, 1 // (op 2479) 

v_lshl_add_u32 v15, s18, 8, v63 // (op 2479) 


v_mul_lo_u32 v41, v15, s29 // (op 2479) most significant: low of low * high
v_mul_hi_u32 v17, v15, s28 // (op 2479) most significant: high of low * low
v_mul_lo_u32 v40, v15, s28 // (op 2479) least significant: low of low * low
v_add_u32 v41, v41, v17 // (op 2479) most significant: sum

v_add_co_u32 v40, s[0:1], v40, v20 // (op 2479) least significant half
v_addc_co_u32 v41, s[0:1], v41, v21, s[0:1] // (op 2479) most significant half



v_mul_lo_u32 v21, v41, 6 // (op 2479) most significant: low of high * low
v_mul_hi_u32 v15, v40, 6 // (op 2479) most significant: high of low * low
v_mul_lo_u32 v20, v40, 6 // (op 2479) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 2479) most significant: sum


v_lshrrev_b64 v[40:41], 3, v[20:21] // (op 2479) 

v_mov_b32 v15, v40 // (op 2479) convert

s_mul_i32 s1, s33, 6 // (op 2481) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 2481) least significant
s_mul_hi_u32 s18, s32, 6 // (op 2481) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2481) most significant: sum

s_mul_i32 s19, s28, s1 // (op 2481) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 2481) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 2481) least significant
s_mul_hi_u32 s50, s28, s0 // (op 2481) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 2481) most significant: sum
s_add_u32 s19, s19, s50 // (op 2481) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2481) 

s_mov_b32 s52, s30 // (op 2481) reg expression
s_mov_b32 s53, s31 // (op 2481) reg expression
s_mov_b32 s54, s0 // (op 2481) reg expression
s_mov_b32 s55, 131072 // (op 2481) call()

s_mul_hi_i32 s0, s16, s11 // (op 2543) 
v_lshrrev_b32 v17, 2, v1 // (op 2543) 
s_add_i32 s1, s0, s16 // (op 2543) 

v_and_b32 v63, -4, v1 // (op 2543) 
s_ashr_i32 s0, s1, 31 // (op 2543) 
v_sub_u32 v67, v1, v63 // (op 2543) 

s_cmp_eq_i32 s11, 0 // (op 2543) 
s_mov_b32 s18, scc // (op 2543) 
s_cmp_eq_i32 s18, 0 // (op 2543) 
s_cselect_b32 s19, 0, -1 // (op 2543) 

s_and_b32 s18, s12, 31 // (op 2543) 
s_lshl_b32 s49, 1, s18 // (op 2543) 
s_add_i32 s50, s49, s19 // (op 2543) 


s_and_b32 s19, s0, s50 // (op 2543) 


s_add_i32 s0, s1, s19 // (op 2543) 


s_ashr_i32 s1, s0, s18 // (op 2543) 


s_lshl_b32 s0, s12, 24 // (op 2543) 
s_ashr_i32 s18, s0, 31 // (op 2543) 

s_xor_b32 s0, s1, s18 // (op 2543) 

s_sub_i32 s1, s0, s18 // (op 2543) 


s_mul_hi_i32 s0, s1, s13 // (op 2543) 
s_add_i32 s18, s0, s1 // (op 2543) 

s_ashr_i32 s0, s18, 31 // (op 2543) 
s_cmp_eq_i32 s13, 0 // (op 2543) 
s_mov_b32 s19, scc // (op 2543) 
s_cmp_eq_i32 s19, 0 // (op 2543) 
s_cselect_b32 s49, 0, -1 // (op 2543) 

s_and_b32 s19, s14, 31 // (op 2543) 
s_lshl_b32 s50, 1, s19 // (op 2543) 
s_add_i32 s51, s50, s49 // (op 2543) 


s_and_b32 s49, s0, s51 // (op 2543) 


s_add_i32 s0, s18, s49 // (op 2543) 


s_ashr_i32 s18, s0, s19 // (op 2543) 


s_lshl_b32 s0, s14, 24 // (op 2543) 
s_ashr_i32 s19, s0, 31 // (op 2543) 

s_xor_b32 s0, s18, s19 // (op 2543) 

s_sub_i32 s18, s0, s19 // (op 2543) 


s_mul_i32 s0, s1, s44 // (op 2543) 

s_sub_u32 s1, s16, s0 // (op 2543) 

s_mul_i32 s0, s18, s44 // (op 2543) 
s_add_u32 s0, s0, s1 // (op 2543) 


s_mul_hi_i32 s1, s16, s10 // (op 2543) 
s_add_i32 s18, s1, s16 // (op 2543) 

s_ashr_i32 s1, s18, 31 // (op 2543) 
s_cmp_eq_i32 s10, 0 // (op 2543) 
s_mov_b32 s19, scc // (op 2543) 
s_cmp_eq_i32 s19, 0 // (op 2543) 
s_cselect_b32 s49, 0, -1 // (op 2543) 

s_and_b32 s19, s47, 31 // (op 2543) 
s_lshl_b32 s50, 1, s19 // (op 2543) 
s_add_i32 s51, s50, s49 // (op 2543) 


s_and_b32 s49, s1, s51 // (op 2543) 


s_add_i32 s1, s18, s49 // (op 2543) 


s_ashr_i32 s18, s1, s19 // (op 2543) 


s_lshl_b32 s1, s47, 24 // (op 2543) 
s_ashr_i32 s19, s1, 31 // (op 2543) 

s_xor_b32 s1, s18, s19 // (op 2543) 

s_sub_i32 s18, s1, s19 // (op 2543) 


s_add_u32 s50, s6, 256 // (op 2543) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2543) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2543) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2543) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 2543) 

s_mul_i32 s1, s44, s50 // (op 2543) 

s_mul_i32 s19, s48, s1 // (op 2543) 

s_mul_i32 s1, s18, s19 // (op 2543) 

s_sub_u32 s19, s16, s1 // (op 2543) 

s_mov_b32 s1, s19 // (op 2543) 
s_mul_hi_i32 s49, s1, s15 // (op 2543) 
s_add_i32 s50, s49, s1 // (op 2543) 


s_ashr_i32 s1, s50, 31 // (op 2543) 
s_cmp_eq_i32 s15, 0 // (op 2543) 
s_mov_b32 s49, scc // (op 2543) 
s_cmp_eq_i32 s49, 0 // (op 2543) 
s_cselect_b32 s51, 0, -1 // (op 2543) 

s_and_b32 s49, s46, 31 // (op 2543) 
s_lshl_b32 s72, 1, s49 // (op 2543) 
s_add_i32 s73, s72, s51 // (op 2543) 


s_and_b32 s51, s1, s73 // (op 2543) 


s_add_i32 s1, s50, s51 // (op 2543) 


s_ashr_i32 s50, s1, s49 // (op 2543) 


s_lshl_b32 s1, s46, 24 // (op 2543) 
s_ashr_i32 s49, s1, 31 // (op 2543) 

s_xor_b32 s1, s50, s49 // (op 2543) 

s_sub_i32 s50, s1, s49 // (op 2543) 


s_mul_i32 s1, s45, s44 // (op 2543) 
s_add_u32 s72, s8, 64 // (op 2543) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 2543) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 2543) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 2543) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 2543) 

s_sub_i32 s49, s72, s1 // (op 2543) 

s_mul_i32 s51, s50, s49 // (op 2543) 


s_sub_u32 s49, s19, s51 // (op 2543) 


s_add_u32 s19, s49, s1 // (op 2543) 


s_cmp_eq_i32 s18, 0 // (op 2543) 
s_mov_b32 s1, scc // (op 2543) 

s_cmp_eq_i32 s1, 0 // (op 2543) 
s_cselect_b32 s18, s19, s0 // (op 2543) 



v_lshl_add_u32 v63, s18, 6, v17 // (op 2543) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 2543) 
s_lshr_b32 s18, s0, 1 // (op 2543) 

v_lshl_add_u32 v17, s18, 3, v67 // (op 2543) 


v_mul_lo_u32 v67, v63, s34 // (op 2543) 
v_add_u32 v67, v67, v17 // (op 2543) 


s_mul_i32 s1, s34, s9 // (op 2545) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 2545) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2545) least significant
s_mul_hi_u32 s19, s34, s8 // (op 2545) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2545) most significant: sum
s_add_u32 s1, s1, s19 // (op 2545) most significant: sum


s_mov_b32 s64, s36 // (op 2545) reg expression
s_mov_b32 s65, s37 // (op 2545) reg expression
s_mov_b32 s66, s0 // (op 2545) reg expression
s_mov_b32 s67, 131072 // (op 2545) call()

s_mul_hi_i32 s0, s16, s11 // (op 2607) 
v_and_b32 v17, -64, v1 // (op 2607) 
s_add_i32 s1, s0, s16 // (op 2607) 

v_sub_u32 v63, v1, v17 // (op 2607) 

s_ashr_i32 s0, s1, 31 // (op 2607) 
v_lshlrev_b32 v17, 2, v63 // (op 2607) 

s_cmp_eq_i32 s11, 0 // (op 2607) 
s_mov_b32 s18, scc // (op 2607) 
v_lshrrev_b32 v63, 6, v1 // (op 2607) 
s_cmp_eq_i32 s18, 0 // (op 2607) 
s_cselect_b32 s19, 0, -1 // (op 2607) 

s_and_b32 s18, s12, 31 // (op 2607) 
s_lshl_b32 s49, 1, s18 // (op 2607) 
s_add_i32 s50, s49, s19 // (op 2607) 


s_and_b32 s19, s0, s50 // (op 2607) 


s_add_i32 s0, s1, s19 // (op 2607) 


s_ashr_i32 s1, s0, s18 // (op 2607) 


s_lshl_b32 s0, s12, 24 // (op 2607) 
s_ashr_i32 s18, s0, 31 // (op 2607) 

s_xor_b32 s0, s1, s18 // (op 2607) 

s_sub_i32 s1, s0, s18 // (op 2607) 


s_mul_hi_i32 s0, s1, s13 // (op 2607) 
s_add_i32 s18, s0, s1 // (op 2607) 

s_ashr_i32 s0, s18, 31 // (op 2607) 
s_cmp_eq_i32 s13, 0 // (op 2607) 
s_mov_b32 s19, scc // (op 2607) 
s_cmp_eq_i32 s19, 0 // (op 2607) 
s_cselect_b32 s49, 0, -1 // (op 2607) 

s_and_b32 s19, s14, 31 // (op 2607) 
s_lshl_b32 s50, 1, s19 // (op 2607) 
s_add_i32 s51, s50, s49 // (op 2607) 


s_and_b32 s49, s0, s51 // (op 2607) 


s_add_i32 s0, s18, s49 // (op 2607) 


s_ashr_i32 s18, s0, s19 // (op 2607) 


s_lshl_b32 s0, s14, 24 // (op 2607) 
s_ashr_i32 s19, s0, 31 // (op 2607) 

s_xor_b32 s0, s18, s19 // (op 2607) 

s_sub_i32 s18, s0, s19 // (op 2607) 


s_add_u32 s50, s6, 256 // (op 2607) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2607) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2607) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2607) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 2607) 

s_mul_i32 s0, s18, s50 // (op 2607) 

s_sub_i32 s49, s1, s0 // (op 2607) 


s_mul_hi_i32 s0, s16, s10 // (op 2607) 
s_add_i32 s1, s0, s16 // (op 2607) 

s_ashr_i32 s0, s1, 31 // (op 2607) 
s_cmp_eq_i32 s10, 0 // (op 2607) 
s_mov_b32 s18, scc // (op 2607) 
s_cmp_eq_i32 s18, 0 // (op 2607) 
s_cselect_b32 s19, 0, -1 // (op 2607) 

s_and_b32 s18, s47, 31 // (op 2607) 
s_lshl_b32 s72, 1, s18 // (op 2607) 
s_add_i32 s73, s72, s19 // (op 2607) 


s_and_b32 s19, s0, s73 // (op 2607) 


s_add_i32 s0, s1, s19 // (op 2607) 


s_ashr_i32 s1, s0, s18 // (op 2607) 


s_lshl_b32 s0, s47, 24 // (op 2607) 
s_ashr_i32 s18, s0, 31 // (op 2607) 

s_xor_b32 s0, s1, s18 // (op 2607) 

s_sub_i32 s1, s0, s18 // (op 2607) 


s_mul_i32 s0, s44, s50 // (op 2607) 

s_mul_i32 s18, s48, s0 // (op 2607) 

s_mul_i32 s0, s1, s18 // (op 2607) 

s_sub_u32 s18, s16, s0 // (op 2607) 

s_mul_hi_i32 s0, s18, s15 // (op 2607) 
s_add_i32 s19, s0, s18 // (op 2607) 


s_ashr_i32 s0, s19, 31 // (op 2607) 
s_cmp_eq_i32 s15, 0 // (op 2607) 
s_mov_b32 s18, scc // (op 2607) 
s_cmp_eq_i32 s18, 0 // (op 2607) 
s_cselect_b32 s50, 0, -1 // (op 2607) 

s_and_b32 s18, s46, 31 // (op 2607) 
s_lshl_b32 s51, 1, s18 // (op 2607) 
s_add_i32 s72, s51, s50 // (op 2607) 


s_and_b32 s50, s0, s72 // (op 2607) 


s_add_i32 s0, s19, s50 // (op 2607) 


s_ashr_i32 s19, s0, s18 // (op 2607) 


s_lshl_b32 s0, s46, 24 // (op 2607) 
s_ashr_i32 s18, s0, 31 // (op 2607) 

s_xor_b32 s0, s19, s18 // (op 2607) 

s_sub_i32 s19, s0, s18 // (op 2607) 


s_cmp_eq_i32 s1, 0 // (op 2607) 
s_mov_b32 s0, scc // (op 2607) 

s_cmp_eq_i32 s0, 0 // (op 2607) 
s_cselect_b32 s1, s19, s49 // (op 2607) 



v_lshl_add_u32 v71, s1, 8, v17 // (op 2607) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 2607) 
s_lshr_b32 s18, s0, 1 // (op 2607) 

v_lshl_add_u32 v17, s18, 3, v63 // (op 2607) 


v_mul_lo_u32 v63, v17, s38 // (op 2607) 
v_add_u32 v63, v63, v71 // (op 2607) 


s_mul_i32 s1, s38, s43 // (op 2609) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 2609) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 2609) least significant
s_mul_hi_u32 s19, s38, s42 // (op 2609) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2609) most significant: sum
s_add_u32 s1, s1, s19 // (op 2609) most significant: sum


s_mov_b32 s68, s40 // (op 2609) reg expression
s_mov_b32 s69, s41 // (op 2609) reg expression
s_mov_b32 s70, s0 // (op 2609) reg expression
s_mov_b32 s71, 131072 // (op 2609) call()

v_and_b32 v17, -8, v1 // (op 2700) 
v_sub_u32 v71, v1, v17 // (op 2700) 

v_lshlrev_b32 v17, 4, v71 // (op 2700) 

v_lshrrev_b32 v71, 3, v1 // (op 2700) 
v_lshl_add_u32 v75, v71, 7, v17 // (op 2700) 


s_nop 0
v_readfirstlane_b32 s0, v75 // (op 2700) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -4, v1 // (op 2752) 
v_sub_u32 v71, v1, v17 // (op 2752) 

v_lshlrev_b32 v17, 6, v71 // (op 2752) 

v_lshrrev_b32 v71, 2, v1 // (op 2752) 
v_lshl_add_u32 v75, v71, 9, v17 // (op 2752) 


v_mul_lo_u32 v17, v75, 6 // (op 2752) 
v_lshrrev_b32 v71, 3, v17 // (op 2752) 

v_lshrrev_b32 v17, 4, v75 // (op 2752) 

v_lshlrev_b32 v75, 2, v17 // (op 2752) 

v_add_u32 v17, v71, v75 // (op 2752) 


v_and_b32 v71, -4, v1 // (op 2810) 
v_sub_u32 v75, v1, v71 // (op 2810) 

v_and_b32 v71, 4294967292, v1 // (op 2810) 
v_add_u32 v79, v71, v75 // (op 2810) 


v_and_b32 v71, -64, v1 // (op 2863) 
v_sub_u32 v75, v1, v71 // (op 2863) 

v_lshlrev_b32 v71, 2, v75 // (op 2863) 

v_lshrrev_b32 v75, 6, v1 // (op 2863) 
v_lshl_add_u32 v83, v75, 8, v71 // (op 2863) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 873) 
s_and_b32 s1, s18, 4294967294 // (op 873) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 872) 
s_mov_b32 s49, s18 // (op 872) convert

s_cmp_lt_u32 s1, s49 // (op 872) 














s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_872 // (op 872) Condition: Top (jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_872 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_872:
s_barrier  // (op 3104) 
s_add_u32 s1, s1, 1 // (op 874) 

s_lshl_b64 s[18:19], s[24:25], 5 // (op 2380) 
v_mov_b64 v[20:21], s[18:19] // (op 2380) convert


v_mov_b64 v[22:23], v[18:19] // (op 2380) 
v_mov_b64 v[24:25], v[22:23] // (op 2380) 
s_mov_b32 m0, s0 // (op 2380) reg expression
buffer_load_dwordx4 v24, s[20:23], 0 offen offset: 0 lds // (op 2380) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 2380) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 2380) most significant half

s_add_u32 m0, m0, 4096 // (op 2380) 
buffer_load_dwordx4 v24, s[20:23], 0 offen offset: 0 lds // (op 2380) Load value direct to lds



v_add_co_u32 v18, vcc, 128, v18 // (op 2399) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 2399) most significant half

s_mul_i32 s19, s29, 6 // (op 221) most significant: low of high * low
s_mul_i32 s18, s28, 6 // (op 221) least significant
s_mul_hi_u32 s49, s28, 6 // (op 221) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 221) most significant: sum

v_lshrrev_b64 v[20:21], 3, s[18:19] // (op 221) 


v_mov_b32 v14, v15 // (op 221) 
v_mov_b32 v16, v14 // (op 221) reg expression
buffer_load_dwordx3 v[22:24], v16, s[52:55], 0 offen // (op 221) Load value
buffer_load_dwordx3 v[26:28], v16, s[52:55], 0 offen offset:12 // (op 221) Load value
buffer_load_dwordx3 v[30:32], v16, s[52:55], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx3 v[34:36], v16, s[52:55], 0 offen offset:36 // (op 221) Load value
v_add_u32 v16, v16, v20 // (op 221) 
buffer_load_dwordx3 v[38:40], v16, s[52:55], 0 offen // (op 221) Load value
buffer_load_dwordx3 v[42:44], v16, s[52:55], 0 offen offset:12 // (op 221) Load value
buffer_load_dwordx3 v[46:48], v16, s[52:55], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx3 v[50:52], v16, s[52:55], 0 offen offset:36 // (op 221) Load value



s_lshl_b64 s[18:19], s[28:29], 7 // (op 2471) 
s_mul_i32 s51, s19, 6 // (op 2471) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 2471) least significant
s_mul_hi_u32 s49, s18, 6 // (op 2471) most significant: high of low * low
s_add_u32 s51, s51, s49 // (op 2471) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 2471) 

v_add_u32 v15, v15, s18 // (op 2471) 



s_barrier  // (op 3123) 
v_mov_b32 v14, v67 // (op 222) 




buffer_load_ubyte v16, v14, s[64:67], 0 offen offset: 0 // (op 222) Load value

v_add_u32 v67, 4, v67 // (op 2535) 
s_waitcnt vmcnt(8)
ds_write_b96 v17, v[22:24] offset:16384 // (op 218) Store local data 


v_mov_b32 v14, v63 // (op 223) 
v_mov_b32 v25, v14 // (op 223) reg expression

buffer_load_dword v29, v25, s[68:71], 0 offen // (op 223) Load value


s_lshl_b32 s18, s38, 2 // (op 2599) 
v_add_u32 v63, v63, s18 // (op 2599) 

s_waitcnt vmcnt(8)
ds_write_b96 v17, v[26:28] offset:16400 // (op 218) Store local data 
s_barrier  // (op 3113) 
s_waitcnt vmcnt(7)
ds_write_b96 v17, v[30:32] offset:16416 // (op 218) Store local data 



ds_read_b128 v[68:71], v3 // (op 201) Load local data 
ds_read_b128 v[72:75], v3 offset:32 // (op 201) Load local data 
s_waitcnt vmcnt(6)
ds_write_b96 v17, v[34:36] offset:16432 // (op 218) Store local data 
s_waitcnt vmcnt(5)
ds_write_b96 v17, v[38:40] offset:16640 // (op 218) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v17, v[42:44] offset:16656 // (op 218) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v17, v[46:48] offset:16672 // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v17, v[50:52] offset:16688 // (op 218) Store local data 

s_barrier  // (op 3144) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3134) 






ds_read_b96_tr_b6 v[20:22], v9 offset:16384 // (op 203) Transpose load local data 
ds_read_b96_tr_b6 v[23:25], v9 offset:20480 // (op 203) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v79, v16 offset:50176 // (op 219) Store local data 

s_barrier  // (op 3164) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3154) 






ds_read_u8 v14, v0 offset:50176 // (op 205) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v83, v29 offset:49152 // (op 220) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3174) 


ds_read_u8 v16, v6 offset:49152 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[68:75], v[20:25], a[0:15], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 208) 





ds_read_b128 v[20:23], v3 offset:64 // (op 210) Load local data 
ds_read_b128 v[24:27], v3 offset:96 // (op 210) Load local data 



ds_read_b96_tr_b6 v[28:30], v9 offset:32768 // (op 212) Transpose load local data 
ds_read_b96_tr_b6 v[31:33], v9 offset:36864 // (op 212) Transpose load local data 



ds_read_u8 v16, v0 offset:50178 // (op 214) Load local data 


ds_read_u8 v34, v6 offset:49664 // (op 216) Load local data 



ds_read_b96_tr_b6 v[36:38], v9 offset:16448 // (op 412) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v9 offset:20544 // (op 412) Transpose load local data 


ds_read_u8 v35, v6 offset:49216 // (op 416) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[68:75], v[36:41], a[16:31], v14, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 417) 





ds_read_b96_tr_b6 v[36:38], v9 offset:32832 // (op 421) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v9 offset:36928 // (op 421) Transpose load local data 


ds_read_u8 v35, v6 offset:49728 // (op 425) Load local data 



ds_read_b96_tr_b6 v[42:44], v9 offset:16512 // (op 646) Transpose load local data 
ds_read_b96_tr_b6 v[45:47], v9 offset:20608 // (op 646) Transpose load local data 


ds_read_u8 v48, v6 offset:49280 // (op 650) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[68:75], v[42:47], a[32:47], v14, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 651) 





ds_read_b96_tr_b6 v[42:44], v9 offset:32896 // (op 655) Transpose load local data 
ds_read_b96_tr_b6 v[45:47], v9 offset:36992 // (op 655) Transpose load local data 


ds_read_u8 v48, v6 offset:49792 // (op 659) Load local data 



ds_read_b96_tr_b6 v[50:52], v9 offset:16576 // (op 880) Transpose load local data 
ds_read_b96_tr_b6 v[53:55], v9 offset:20672 // (op 880) Transpose load local data 


ds_read_u8 v49, v6 offset:49344 // (op 884) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[68:75], v[50:55], a[48:63], v14, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 885) 







ds_read_b96_tr_b6 v[68:70], v9 offset:32960 // (op 889) Transpose load local data 
ds_read_b96_tr_b6 v[71:73], v9 offset:37056 // (op 889) Transpose load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[20:27], v[28:33], a[0:15], v16, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 217) 




ds_read_u8 v14, v6 offset:49856 // (op 893) Load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[20:27], v[36:41], a[16:31], v16, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 426) 


v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[20:27], v[42:47], a[32:47], v16, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 660) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[20:27], v[68:73], a[48:63], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 894) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 872) 
s_mov_b32 s49, s18 // (op 872) convert

s_cmp_lt_u32 s1, s49 // (op 872) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_872 // (op 872) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_872 if true)
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_872:























s_barrier  // (op 3283) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_155
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_accvgpr_read v16, a4 // (op 50) 
v_accvgpr_read v17, a5 // (op 50) 
v_accvgpr_read v18, a6 // (op 50) 
v_accvgpr_read v19, a7 // (op 50) 
v_accvgpr_read v20, a8 // (op 50) 
v_accvgpr_read v21, a9 // (op 50) 
v_accvgpr_read v22, a10 // (op 50) 
v_accvgpr_read v23, a11 // (op 50) 
v_accvgpr_read v24, a12 // (op 50) 
v_accvgpr_read v25, a13 // (op 50) 
v_accvgpr_read v26, a14 // (op 50) 
v_accvgpr_read v27, a15 // (op 50) 
v_mul_f32 v28, v10, v12 // (op 50) 
v_mul_f32 v29, v10, v13 // (op 50) 
v_mul_f32 v30, v10, v14 // (op 50) 
v_mul_f32 v31, v10, v15 // (op 50) 
v_mul_f32 v32, v10, v16 // (op 50) 
v_mul_f32 v33, v10, v17 // (op 50) 
v_mul_f32 v34, v10, v18 // (op 50) 
v_mul_f32 v35, v10, v19 // (op 50) 
v_mul_f32 v36, v10, v20 // (op 50) 
v_mul_f32 v37, v10, v21 // (op 50) 
v_mul_f32 v38, v10, v22 // (op 50) 
v_mul_f32 v39, v10, v23 // (op 50) 
v_mul_f32 v40, v10, v24 // (op 50) 
v_mul_f32 v41, v10, v25 // (op 50) 
v_mul_f32 v42, v10, v26 // (op 50) 
v_mul_f32 v43, v10, v27 // (op 50) 



v_mov_b32 v0, v11 // (op 56) reg expression

buffer_store_dwordx4 v[28:31], v0, s[60:63], 0 offen // (op 56) Store value
buffer_store_dwordx4 v[32:35], v0, s[60:63], 0 offen offset:32 // (op 56) Store value
buffer_store_dwordx4 v[36:39], v0, s[60:63], 0 offen offset:64 // (op 56) Store value
buffer_store_dwordx4 v[40:43], v0, s[60:63], 0 offen offset:96 // (op 56) Store value


v_accvgpr_read v12, a16 // (op 1258) 
v_accvgpr_read v13, a17 // (op 1258) 
v_accvgpr_read v14, a18 // (op 1258) 
v_accvgpr_read v15, a19 // (op 1258) 
v_accvgpr_read v16, a20 // (op 1258) 
v_accvgpr_read v17, a21 // (op 1258) 
v_accvgpr_read v18, a22 // (op 1258) 
v_accvgpr_read v19, a23 // (op 1258) 
v_accvgpr_read v20, a24 // (op 1258) 
v_accvgpr_read v21, a25 // (op 1258) 
v_accvgpr_read v22, a26 // (op 1258) 
v_accvgpr_read v23, a27 // (op 1258) 
v_accvgpr_read v24, a28 // (op 1258) 
v_accvgpr_read v25, a29 // (op 1258) 
v_accvgpr_read v26, a30 // (op 1258) 
v_accvgpr_read v27, a31 // (op 1258) 
v_mul_f32 v28, v10, v12 // (op 1258) 
v_mul_f32 v29, v10, v13 // (op 1258) 
v_mul_f32 v30, v10, v14 // (op 1258) 
v_mul_f32 v31, v10, v15 // (op 1258) 
v_mul_f32 v32, v10, v16 // (op 1258) 
v_mul_f32 v33, v10, v17 // (op 1258) 
v_mul_f32 v34, v10, v18 // (op 1258) 
v_mul_f32 v35, v10, v19 // (op 1258) 
v_mul_f32 v36, v10, v20 // (op 1258) 
v_mul_f32 v37, v10, v21 // (op 1258) 
v_mul_f32 v38, v10, v22 // (op 1258) 
v_mul_f32 v39, v10, v23 // (op 1258) 
v_mul_f32 v40, v10, v24 // (op 1258) 
v_mul_f32 v41, v10, v25 // (op 1258) 
v_mul_f32 v42, v10, v26 // (op 1258) 
v_mul_f32 v43, v10, v27 // (op 1258) 



s_lshl_b32 s0, s4, 8 // (op 1260) 
v_add_u32 v0, v11, s0 // (op 1260) 


buffer_store_dwordx4 v[28:31], v0, s[60:63], 0 offen // (op 1260) Store value
buffer_store_dwordx4 v[32:35], v0, s[60:63], 0 offen offset:32 // (op 1260) Store value
buffer_store_dwordx4 v[36:39], v0, s[60:63], 0 offen offset:64 // (op 1260) Store value
buffer_store_dwordx4 v[40:43], v0, s[60:63], 0 offen offset:96 // (op 1260) Store value


v_accvgpr_read v12, a32 // (op 1270) 
v_accvgpr_read v13, a33 // (op 1270) 
v_accvgpr_read v14, a34 // (op 1270) 
v_accvgpr_read v15, a35 // (op 1270) 
v_accvgpr_read v16, a36 // (op 1270) 
v_accvgpr_read v17, a37 // (op 1270) 
v_accvgpr_read v18, a38 // (op 1270) 
v_accvgpr_read v19, a39 // (op 1270) 
v_accvgpr_read v20, a40 // (op 1270) 
v_accvgpr_read v21, a41 // (op 1270) 
v_accvgpr_read v22, a42 // (op 1270) 
v_accvgpr_read v23, a43 // (op 1270) 
v_accvgpr_read v24, a44 // (op 1270) 
v_accvgpr_read v25, a45 // (op 1270) 
v_accvgpr_read v26, a46 // (op 1270) 
v_accvgpr_read v27, a47 // (op 1270) 
v_mul_f32 v28, v10, v12 // (op 1270) 
v_mul_f32 v29, v10, v13 // (op 1270) 
v_mul_f32 v30, v10, v14 // (op 1270) 
v_mul_f32 v31, v10, v15 // (op 1270) 
v_mul_f32 v32, v10, v16 // (op 1270) 
v_mul_f32 v33, v10, v17 // (op 1270) 
v_mul_f32 v34, v10, v18 // (op 1270) 
v_mul_f32 v35, v10, v19 // (op 1270) 
v_mul_f32 v36, v10, v20 // (op 1270) 
v_mul_f32 v37, v10, v21 // (op 1270) 
v_mul_f32 v38, v10, v22 // (op 1270) 
v_mul_f32 v39, v10, v23 // (op 1270) 
v_mul_f32 v40, v10, v24 // (op 1270) 
v_mul_f32 v41, v10, v25 // (op 1270) 
v_mul_f32 v42, v10, v26 // (op 1270) 
v_mul_f32 v43, v10, v27 // (op 1270) 



s_lshl_b32 s0, s4, 9 // (op 1272) 
v_add_u32 v0, v11, s0 // (op 1272) 


buffer_store_dwordx4 v[28:31], v0, s[60:63], 0 offen // (op 1272) Store value
buffer_store_dwordx4 v[32:35], v0, s[60:63], 0 offen offset:32 // (op 1272) Store value
buffer_store_dwordx4 v[36:39], v0, s[60:63], 0 offen offset:64 // (op 1272) Store value
buffer_store_dwordx4 v[40:43], v0, s[60:63], 0 offen offset:96 // (op 1272) Store value


v_accvgpr_read v12, a48 // (op 1282) 
v_accvgpr_read v13, a49 // (op 1282) 
v_accvgpr_read v14, a50 // (op 1282) 
v_accvgpr_read v15, a51 // (op 1282) 
v_accvgpr_read v16, a52 // (op 1282) 
v_accvgpr_read v17, a53 // (op 1282) 
v_accvgpr_read v18, a54 // (op 1282) 
v_accvgpr_read v19, a55 // (op 1282) 
v_accvgpr_read v20, a56 // (op 1282) 
v_accvgpr_read v21, a57 // (op 1282) 
v_accvgpr_read v22, a58 // (op 1282) 
v_accvgpr_read v23, a59 // (op 1282) 
v_accvgpr_read v24, a60 // (op 1282) 
v_accvgpr_read v25, a61 // (op 1282) 
v_accvgpr_read v26, a62 // (op 1282) 
v_accvgpr_read v27, a63 // (op 1282) 
v_mul_f32 v28, v10, v12 // (op 1282) 
v_mul_f32 v29, v10, v13 // (op 1282) 
v_mul_f32 v30, v10, v14 // (op 1282) 
v_mul_f32 v31, v10, v15 // (op 1282) 
v_mul_f32 v32, v10, v16 // (op 1282) 
v_mul_f32 v33, v10, v17 // (op 1282) 
v_mul_f32 v34, v10, v18 // (op 1282) 
v_mul_f32 v35, v10, v19 // (op 1282) 
v_mul_f32 v36, v10, v20 // (op 1282) 
v_mul_f32 v37, v10, v21 // (op 1282) 
v_mul_f32 v38, v10, v22 // (op 1282) 
v_mul_f32 v39, v10, v23 // (op 1282) 
v_mul_f32 v40, v10, v24 // (op 1282) 
v_mul_f32 v41, v10, v25 // (op 1282) 
v_mul_f32 v42, v10, v26 // (op 1282) 
v_mul_f32 v43, v10, v27 // (op 1282) 



s_lshl_b32 s0, s4, 8 // (op 1284) 
s_mul_i32 s1, 3, s0 // (op 1284) 

v_add_u32 v0, v11, s1 // (op 1284) 


buffer_store_dwordx4 v[28:31], v0, s[60:63], 0 offen // (op 1284) Store value
buffer_store_dwordx4 v[32:35], v0, s[60:63], 0 offen offset:32 // (op 1284) Store value
buffer_store_dwordx4 v[36:39], v0, s[60:63], 0 offen offset:64 // (op 1284) Store value
buffer_store_dwordx4 v[40:43], v0, s[60:63], 0 offen offset:96 // (op 1284) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_155
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_155:


v_mov_b32 v0, v8 // (op 167) reg expression

buffer_load_dwordx4 v[12:15], v0, s[56:59], 0 offen // (op 167) Load value
buffer_load_dwordx4 v[16:19], v0, s[56:59], 0 offen offset:32 // (op 167) Load value
buffer_load_dwordx4 v[20:23], v0, s[56:59], 0 offen offset:64 // (op 167) Load value
v_accvgpr_read v28, a0 // (op 164) 
v_accvgpr_read v29, a1 // (op 164) 
v_accvgpr_read v30, a2 // (op 164) 
v_accvgpr_read v31, a3 // (op 164) 
v_accvgpr_read v32, a4 // (op 164) 
v_accvgpr_read v33, a5 // (op 164) 
v_accvgpr_read v34, a6 // (op 164) 
v_accvgpr_read v35, a7 // (op 164) 
v_accvgpr_read v36, a8 // (op 164) 
v_accvgpr_read v37, a9 // (op 164) 
v_accvgpr_read v38, a10 // (op 164) 
v_accvgpr_read v39, a11 // (op 164) 
v_accvgpr_read v40, a12 // (op 164) 
v_accvgpr_read v41, a13 // (op 164) 
v_accvgpr_read v42, a14 // (op 164) 
v_accvgpr_read v43, a15 // (op 164) 
v_mul_f32 v44, v10, v28 // (op 164) 
v_mul_f32 v45, v10, v29 // (op 164) 
v_mul_f32 v46, v10, v30 // (op 164) 
v_mul_f32 v47, v10, v31 // (op 164) 
v_mul_f32 v48, v10, v32 // (op 164) 
v_mul_f32 v49, v10, v33 // (op 164) 
v_mul_f32 v50, v10, v34 // (op 164) 
v_mul_f32 v51, v10, v35 // (op 164) 
v_mul_f32 v52, v10, v36 // (op 164) 
v_mul_f32 v53, v10, v37 // (op 164) 
v_mul_f32 v54, v10, v38 // (op 164) 
v_mul_f32 v55, v10, v39 // (op 164) 
v_mul_f32 v56, v10, v40 // (op 164) 
v_mul_f32 v57, v10, v41 // (op 164) 
v_mul_f32 v58, v10, v42 // (op 164) 
v_mul_f32 v59, v10, v43 // (op 164) 

buffer_load_dwordx4 v[24:27], v0, s[56:59], 0 offen offset:96 // (op 167) Load value

s_waitcnt vmcnt(3)
v_mul_f32 v28, v7, v12 // (op 168) 
v_mul_f32 v29, v7, v13 // (op 168) 
v_mul_f32 v30, v7, v14 // (op 168) 
v_mul_f32 v31, v7, v15 // (op 168) 
s_waitcnt vmcnt(2)
v_mul_f32 v32, v7, v16 // (op 168) 
v_mul_f32 v33, v7, v17 // (op 168) 
v_mul_f32 v34, v7, v18 // (op 168) 
v_mul_f32 v35, v7, v19 // (op 168) 
s_waitcnt vmcnt(1)
v_mul_f32 v36, v7, v20 // (op 168) 
v_mul_f32 v37, v7, v21 // (op 168) 
v_mul_f32 v38, v7, v22 // (op 168) 
v_mul_f32 v39, v7, v23 // (op 168) 
s_waitcnt vmcnt(0)
v_mul_f32 v40, v7, v24 // (op 168) 
v_mul_f32 v41, v7, v25 // (op 168) 
v_mul_f32 v42, v7, v26 // (op 168) 
v_mul_f32 v43, v7, v27 // (op 168) 

v_add_f32 v12, v28, v44 // (op 165) 
v_add_f32 v13, v29, v45 // (op 165) 
v_add_f32 v14, v30, v46 // (op 165) 
v_add_f32 v15, v31, v47 // (op 165) 
v_add_f32 v16, v32, v48 // (op 165) 
v_add_f32 v17, v33, v49 // (op 165) 
v_add_f32 v18, v34, v50 // (op 165) 
v_add_f32 v19, v35, v51 // (op 165) 
v_add_f32 v20, v36, v52 // (op 165) 
v_add_f32 v21, v37, v53 // (op 165) 
v_add_f32 v22, v38, v54 // (op 165) 
v_add_f32 v23, v39, v55 // (op 165) 
v_add_f32 v24, v40, v56 // (op 165) 
v_add_f32 v25, v41, v57 // (op 165) 
v_add_f32 v26, v42, v58 // (op 165) 
v_add_f32 v27, v43, v59 // (op 165) 




v_mov_b32 v0, v11 // (op 166) reg expression

buffer_store_dwordx4 v[12:15], v0, s[60:63], 0 offen // (op 166) Store value
buffer_store_dwordx4 v[16:19], v0, s[60:63], 0 offen offset:32 // (op 166) Store value
buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen offset:64 // (op 166) Store value
buffer_store_dwordx4 v[24:27], v0, s[60:63], 0 offen offset:96 // (op 166) Store value




s_lshl_b32 s0, s2, 8 // (op 1316) 
v_add_u32 v0, v8, s0 // (op 1316) 


buffer_load_dwordx4 v[12:15], v0, s[56:59], 0 offen // (op 1316) Load value
buffer_load_dwordx4 v[16:19], v0, s[56:59], 0 offen offset:32 // (op 1316) Load value
v_accvgpr_read v28, a16 // (op 1313) 
v_accvgpr_read v29, a17 // (op 1313) 
v_accvgpr_read v30, a18 // (op 1313) 
v_accvgpr_read v31, a19 // (op 1313) 
v_accvgpr_read v32, a20 // (op 1313) 
v_accvgpr_read v33, a21 // (op 1313) 
v_accvgpr_read v34, a22 // (op 1313) 
v_accvgpr_read v35, a23 // (op 1313) 
v_accvgpr_read v36, a24 // (op 1313) 
v_accvgpr_read v37, a25 // (op 1313) 
v_accvgpr_read v38, a26 // (op 1313) 
v_accvgpr_read v39, a27 // (op 1313) 
v_accvgpr_read v40, a28 // (op 1313) 
v_accvgpr_read v41, a29 // (op 1313) 
v_accvgpr_read v42, a30 // (op 1313) 
v_accvgpr_read v43, a31 // (op 1313) 
v_mul_f32 v44, v10, v28 // (op 1313) 
v_mul_f32 v45, v10, v29 // (op 1313) 
v_mul_f32 v46, v10, v30 // (op 1313) 
v_mul_f32 v47, v10, v31 // (op 1313) 
v_mul_f32 v48, v10, v32 // (op 1313) 
v_mul_f32 v49, v10, v33 // (op 1313) 
v_mul_f32 v50, v10, v34 // (op 1313) 
v_mul_f32 v51, v10, v35 // (op 1313) 
v_mul_f32 v52, v10, v36 // (op 1313) 
v_mul_f32 v53, v10, v37 // (op 1313) 
v_mul_f32 v54, v10, v38 // (op 1313) 
v_mul_f32 v55, v10, v39 // (op 1313) 
v_mul_f32 v56, v10, v40 // (op 1313) 
v_mul_f32 v57, v10, v41 // (op 1313) 
v_mul_f32 v58, v10, v42 // (op 1313) 
v_mul_f32 v59, v10, v43 // (op 1313) 

buffer_load_dwordx4 v[20:23], v0, s[56:59], 0 offen offset:64 // (op 1316) Load value
buffer_load_dwordx4 v[24:27], v0, s[56:59], 0 offen offset:96 // (op 1316) Load value

s_waitcnt vmcnt(3)
v_mul_f32 v28, v7, v12 // (op 1317) 
v_mul_f32 v29, v7, v13 // (op 1317) 
v_mul_f32 v30, v7, v14 // (op 1317) 
v_mul_f32 v31, v7, v15 // (op 1317) 
s_waitcnt vmcnt(2)
v_mul_f32 v32, v7, v16 // (op 1317) 
v_mul_f32 v33, v7, v17 // (op 1317) 
v_mul_f32 v34, v7, v18 // (op 1317) 
v_mul_f32 v35, v7, v19 // (op 1317) 
s_waitcnt vmcnt(1)
v_mul_f32 v36, v7, v20 // (op 1317) 
v_mul_f32 v37, v7, v21 // (op 1317) 
v_mul_f32 v38, v7, v22 // (op 1317) 
v_mul_f32 v39, v7, v23 // (op 1317) 
s_waitcnt vmcnt(0)
v_mul_f32 v40, v7, v24 // (op 1317) 
v_mul_f32 v41, v7, v25 // (op 1317) 
v_mul_f32 v42, v7, v26 // (op 1317) 
v_mul_f32 v43, v7, v27 // (op 1317) 

v_add_f32 v12, v28, v44 // (op 1314) 
v_add_f32 v13, v29, v45 // (op 1314) 
v_add_f32 v14, v30, v46 // (op 1314) 
v_add_f32 v15, v31, v47 // (op 1314) 
v_add_f32 v16, v32, v48 // (op 1314) 
v_add_f32 v17, v33, v49 // (op 1314) 
v_add_f32 v18, v34, v50 // (op 1314) 
v_add_f32 v19, v35, v51 // (op 1314) 
v_add_f32 v20, v36, v52 // (op 1314) 
v_add_f32 v21, v37, v53 // (op 1314) 
v_add_f32 v22, v38, v54 // (op 1314) 
v_add_f32 v23, v39, v55 // (op 1314) 
v_add_f32 v24, v40, v56 // (op 1314) 
v_add_f32 v25, v41, v57 // (op 1314) 
v_add_f32 v26, v42, v58 // (op 1314) 
v_add_f32 v27, v43, v59 // (op 1314) 




s_lshl_b32 s0, s4, 8 // (op 1315) 
v_add_u32 v0, v11, s0 // (op 1315) 


buffer_store_dwordx4 v[12:15], v0, s[60:63], 0 offen // (op 1315) Store value
buffer_store_dwordx4 v[16:19], v0, s[60:63], 0 offen offset:32 // (op 1315) Store value
buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen offset:64 // (op 1315) Store value
buffer_store_dwordx4 v[24:27], v0, s[60:63], 0 offen offset:96 // (op 1315) Store value




s_lshl_b32 s0, s2, 9 // (op 1328) 
v_add_u32 v0, v8, s0 // (op 1328) 


buffer_load_dwordx4 v[12:15], v0, s[56:59], 0 offen // (op 1328) Load value
buffer_load_dwordx4 v[16:19], v0, s[56:59], 0 offen offset:32 // (op 1328) Load value
v_accvgpr_read v28, a32 // (op 1325) 
v_accvgpr_read v29, a33 // (op 1325) 
v_accvgpr_read v30, a34 // (op 1325) 
v_accvgpr_read v31, a35 // (op 1325) 
v_accvgpr_read v32, a36 // (op 1325) 
v_accvgpr_read v33, a37 // (op 1325) 
v_accvgpr_read v34, a38 // (op 1325) 
v_accvgpr_read v35, a39 // (op 1325) 
v_accvgpr_read v36, a40 // (op 1325) 
v_accvgpr_read v37, a41 // (op 1325) 
v_accvgpr_read v38, a42 // (op 1325) 
v_accvgpr_read v39, a43 // (op 1325) 
v_accvgpr_read v40, a44 // (op 1325) 
v_accvgpr_read v41, a45 // (op 1325) 
v_accvgpr_read v42, a46 // (op 1325) 
v_accvgpr_read v43, a47 // (op 1325) 
v_mul_f32 v44, v10, v28 // (op 1325) 
v_mul_f32 v45, v10, v29 // (op 1325) 
v_mul_f32 v46, v10, v30 // (op 1325) 
v_mul_f32 v47, v10, v31 // (op 1325) 
v_mul_f32 v48, v10, v32 // (op 1325) 
v_mul_f32 v49, v10, v33 // (op 1325) 
v_mul_f32 v50, v10, v34 // (op 1325) 
v_mul_f32 v51, v10, v35 // (op 1325) 
v_mul_f32 v52, v10, v36 // (op 1325) 
v_mul_f32 v53, v10, v37 // (op 1325) 
v_mul_f32 v54, v10, v38 // (op 1325) 
v_mul_f32 v55, v10, v39 // (op 1325) 
v_mul_f32 v56, v10, v40 // (op 1325) 
v_mul_f32 v57, v10, v41 // (op 1325) 
v_mul_f32 v58, v10, v42 // (op 1325) 
v_mul_f32 v59, v10, v43 // (op 1325) 

buffer_load_dwordx4 v[20:23], v0, s[56:59], 0 offen offset:64 // (op 1328) Load value
buffer_load_dwordx4 v[24:27], v0, s[56:59], 0 offen offset:96 // (op 1328) Load value

s_waitcnt vmcnt(3)
v_mul_f32 v28, v7, v12 // (op 1329) 
v_mul_f32 v29, v7, v13 // (op 1329) 
v_mul_f32 v30, v7, v14 // (op 1329) 
v_mul_f32 v31, v7, v15 // (op 1329) 
s_waitcnt vmcnt(2)
v_mul_f32 v32, v7, v16 // (op 1329) 
v_mul_f32 v33, v7, v17 // (op 1329) 
v_mul_f32 v34, v7, v18 // (op 1329) 
v_mul_f32 v35, v7, v19 // (op 1329) 
s_waitcnt vmcnt(1)
v_mul_f32 v36, v7, v20 // (op 1329) 
v_mul_f32 v37, v7, v21 // (op 1329) 
v_mul_f32 v38, v7, v22 // (op 1329) 
v_mul_f32 v39, v7, v23 // (op 1329) 
s_waitcnt vmcnt(0)
v_mul_f32 v40, v7, v24 // (op 1329) 
v_mul_f32 v41, v7, v25 // (op 1329) 
v_mul_f32 v42, v7, v26 // (op 1329) 
v_mul_f32 v43, v7, v27 // (op 1329) 

v_add_f32 v12, v28, v44 // (op 1326) 
v_add_f32 v13, v29, v45 // (op 1326) 
v_add_f32 v14, v30, v46 // (op 1326) 
v_add_f32 v15, v31, v47 // (op 1326) 
v_add_f32 v16, v32, v48 // (op 1326) 
v_add_f32 v17, v33, v49 // (op 1326) 
v_add_f32 v18, v34, v50 // (op 1326) 
v_add_f32 v19, v35, v51 // (op 1326) 
v_add_f32 v20, v36, v52 // (op 1326) 
v_add_f32 v21, v37, v53 // (op 1326) 
v_add_f32 v22, v38, v54 // (op 1326) 
v_add_f32 v23, v39, v55 // (op 1326) 
v_add_f32 v24, v40, v56 // (op 1326) 
v_add_f32 v25, v41, v57 // (op 1326) 
v_add_f32 v26, v42, v58 // (op 1326) 
v_add_f32 v27, v43, v59 // (op 1326) 




s_lshl_b32 s0, s4, 9 // (op 1327) 
v_add_u32 v0, v11, s0 // (op 1327) 


buffer_store_dwordx4 v[12:15], v0, s[60:63], 0 offen // (op 1327) Store value
buffer_store_dwordx4 v[16:19], v0, s[60:63], 0 offen offset:32 // (op 1327) Store value
buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen offset:64 // (op 1327) Store value
buffer_store_dwordx4 v[24:27], v0, s[60:63], 0 offen offset:96 // (op 1327) Store value




s_lshl_b32 s0, s2, 8 // (op 1340) 
s_mul_i32 s1, 3, s0 // (op 1340) 

v_add_u32 v0, v8, s1 // (op 1340) 


buffer_load_dwordx4 v[12:15], v0, s[56:59], 0 offen // (op 1340) Load value
buffer_load_dwordx4 v[16:19], v0, s[56:59], 0 offen offset:32 // (op 1340) Load value
v_accvgpr_read v28, a48 // (op 1337) 
v_accvgpr_read v29, a49 // (op 1337) 
v_accvgpr_read v30, a50 // (op 1337) 
v_accvgpr_read v31, a51 // (op 1337) 
v_accvgpr_read v32, a52 // (op 1337) 
v_accvgpr_read v33, a53 // (op 1337) 
v_accvgpr_read v34, a54 // (op 1337) 
v_accvgpr_read v35, a55 // (op 1337) 
v_accvgpr_read v36, a56 // (op 1337) 
v_accvgpr_read v37, a57 // (op 1337) 
v_accvgpr_read v38, a58 // (op 1337) 
v_accvgpr_read v39, a59 // (op 1337) 
v_accvgpr_read v40, a60 // (op 1337) 
v_accvgpr_read v41, a61 // (op 1337) 
v_accvgpr_read v42, a62 // (op 1337) 
v_accvgpr_read v43, a63 // (op 1337) 
v_mul_f32 v44, v10, v28 // (op 1337) 
v_mul_f32 v45, v10, v29 // (op 1337) 
v_mul_f32 v46, v10, v30 // (op 1337) 
v_mul_f32 v47, v10, v31 // (op 1337) 
v_mul_f32 v48, v10, v32 // (op 1337) 
v_mul_f32 v49, v10, v33 // (op 1337) 
v_mul_f32 v50, v10, v34 // (op 1337) 
v_mul_f32 v51, v10, v35 // (op 1337) 
v_mul_f32 v52, v10, v36 // (op 1337) 
v_mul_f32 v53, v10, v37 // (op 1337) 
v_mul_f32 v54, v10, v38 // (op 1337) 
v_mul_f32 v55, v10, v39 // (op 1337) 
v_mul_f32 v56, v10, v40 // (op 1337) 
v_mul_f32 v57, v10, v41 // (op 1337) 
v_mul_f32 v58, v10, v42 // (op 1337) 
v_mul_f32 v59, v10, v43 // (op 1337) 

buffer_load_dwordx4 v[20:23], v0, s[56:59], 0 offen offset:64 // (op 1340) Load value
buffer_load_dwordx4 v[24:27], v0, s[56:59], 0 offen offset:96 // (op 1340) Load value

s_waitcnt vmcnt(3)
v_mul_f32 v28, v7, v12 // (op 1341) 
v_mul_f32 v29, v7, v13 // (op 1341) 
v_mul_f32 v30, v7, v14 // (op 1341) 
v_mul_f32 v31, v7, v15 // (op 1341) 
s_waitcnt vmcnt(2)
v_mul_f32 v32, v7, v16 // (op 1341) 
v_mul_f32 v33, v7, v17 // (op 1341) 
v_mul_f32 v34, v7, v18 // (op 1341) 
v_mul_f32 v35, v7, v19 // (op 1341) 
s_waitcnt vmcnt(1)
v_mul_f32 v36, v7, v20 // (op 1341) 
v_mul_f32 v37, v7, v21 // (op 1341) 
v_mul_f32 v38, v7, v22 // (op 1341) 
v_mul_f32 v39, v7, v23 // (op 1341) 
s_waitcnt vmcnt(0)
v_mul_f32 v40, v7, v24 // (op 1341) 
v_mul_f32 v41, v7, v25 // (op 1341) 
v_mul_f32 v42, v7, v26 // (op 1341) 
v_mul_f32 v43, v7, v27 // (op 1341) 

v_add_f32 v12, v28, v44 // (op 1338) 
v_add_f32 v13, v29, v45 // (op 1338) 
v_add_f32 v14, v30, v46 // (op 1338) 
v_add_f32 v15, v31, v47 // (op 1338) 
v_add_f32 v16, v32, v48 // (op 1338) 
v_add_f32 v17, v33, v49 // (op 1338) 
v_add_f32 v18, v34, v50 // (op 1338) 
v_add_f32 v19, v35, v51 // (op 1338) 
v_add_f32 v20, v36, v52 // (op 1338) 
v_add_f32 v21, v37, v53 // (op 1338) 
v_add_f32 v22, v38, v54 // (op 1338) 
v_add_f32 v23, v39, v55 // (op 1338) 
v_add_f32 v24, v40, v56 // (op 1338) 
v_add_f32 v25, v41, v57 // (op 1338) 
v_add_f32 v26, v42, v58 // (op 1338) 
v_add_f32 v27, v43, v59 // (op 1338) 




s_lshl_b32 s0, s4, 8 // (op 1339) 
s_mul_i32 s1, 3, s0 // (op 1339) 

v_add_u32 v0, v11, s1 // (op 1339) 


buffer_store_dwordx4 v[12:15], v0, s[60:63], 0 offen // (op 1339) Store value
buffer_store_dwordx4 v[16:19], v0, s[60:63], 0 offen offset:32 // (op 1339) Store value
buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen offset:64 // (op 1339) Store value
buffer_store_dwordx4 v[24:27], v0, s[60:63], 0 offen offset:96 // (op 1339) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_155:












s_endpgm  // End of RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.LRR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__end:
.size RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_, .LRR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__end-RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 208
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 84480 // lds bytes
  .amdhsa_accum_offset 144
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
  MXScaleFormat: NoSwizzle
  ProblemType:
    OperationType: GEMM
    DataType: B8
    DataTypeA: B8
    DataTypeB: B6
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: True
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
            { type: int64, semantic: SizeFree0 },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSumScaleA },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [64, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 84480
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 144
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_11
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_2_size_1_12
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_4
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_9
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 152
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 156
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 160
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 164
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 168
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 172
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 176
        .value_kind: by_value
...

.end_amdgpu_metadata

