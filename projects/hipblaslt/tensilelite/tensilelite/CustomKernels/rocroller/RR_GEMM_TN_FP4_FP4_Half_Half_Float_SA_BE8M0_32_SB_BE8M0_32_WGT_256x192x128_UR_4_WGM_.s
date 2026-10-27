// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM_
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM_,@function
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__exec_begin:


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


v_and_b32 v0, 4294967232, v1 // (op 30885) 
v_and_b32 v3, 15, v1 // (op 30885) 
v_add_lshl_u32 v4, v0, v3, 7 // (op 30885) 


v_and_b32 v0, 63, v1 // (op 30885) 
v_lshrrev_b32 v3, 4, v0 // (op 30885) 

v_lshlrev_b32 v0, 5, v3 // (op 30885) 

v_add_lshl_u32 v3, v4, v0, 2 // (op 30885) 


v_lshrrev_b32 v0, 3, v3 // (op 30885) 

v_and_b32 v3, 15, v1 // (op 30942) 
v_and_b32 v4, 63, v1 // (op 30942) 
v_lshrrev_b32 v5, 4, v4 // (op 30942) 

v_lshlrev_b32 v4, 5, v5 // (op 30942) 

v_lshl_add_u32 v5, v3, 7, v4 // (op 30942) 


v_lshlrev_b32 v3, 2, v5 // (op 30942) 

v_lshrrev_b32 v4, 3, v3 // (op 30942) 

v_and_b32 v3, 4294967232, v1 // (op 31009) 
v_and_b32 v5, 15, v1 // (op 31009) 
v_add_lshl_u32 v6, v3, v5, 7 // (op 31009) 


v_and_b32 v3, 63, v1 // (op 31009) 
v_lshrrev_b32 v5, 4, v3 // (op 31009) 

v_lshlrev_b32 v3, 5, v5 // (op 31009) 

v_add_lshl_u32 v5, v6, v3, 2 // (op 31009) 


v_lshrrev_b32 v3, 3, v5 // (op 31009) 

v_and_b32 v5, 15, v1 // (op 31053) 
v_and_b32 v6, 63, v1 // (op 31053) 
v_lshrrev_b32 v7, 4, v6 // (op 31053) 

v_lshlrev_b32 v6, 5, v7 // (op 31053) 

v_lshl_add_u32 v7, v5, 7, v6 // (op 31053) 


v_lshlrev_b32 v5, 2, v7 // (op 31053) 

v_lshrrev_b32 v6, 3, v5 // (op 31053) 

v_and_b32 v5, 4294967232, v1 // (op 31097) 
v_and_b32 v7, 15, v1 // (op 31097) 
v_add_lshl_u32 v8, v5, v7, 7 // (op 31097) 


v_and_b32 v5, 63, v1 // (op 31097) 
v_lshrrev_b32 v7, 4, v5 // (op 31097) 

v_lshlrev_b32 v5, 5, v7 // (op 31097) 

v_add_lshl_u32 v7, v8, v5, 2 // (op 31097) 


v_lshrrev_b32 v5, 3, v7 // (op 31097) 

v_and_b32 v7, 15, v1 // (op 31141) 
v_and_b32 v8, 63, v1 // (op 31141) 
v_lshrrev_b32 v9, 4, v8 // (op 31141) 

v_lshlrev_b32 v8, 5, v9 // (op 31141) 

v_lshl_add_u32 v9, v7, 7, v8 // (op 31141) 


v_lshlrev_b32 v7, 2, v9 // (op 31141) 

v_lshrrev_b32 v8, 3, v7 // (op 31141) 

v_and_b32 v7, 4294967232, v1 // (op 31185) 
v_and_b32 v9, 15, v1 // (op 31185) 
v_add_lshl_u32 v10, v7, v9, 7 // (op 31185) 


v_and_b32 v7, 63, v1 // (op 31185) 
v_lshrrev_b32 v9, 4, v7 // (op 31185) 

v_lshlrev_b32 v7, 5, v9 // (op 31185) 

v_add_lshl_u32 v9, v10, v7, 2 // (op 31185) 


v_lshrrev_b32 v7, 3, v9 // (op 31185) 

v_and_b32 v9, 15, v1 // (op 31229) 
v_and_b32 v10, 63, v1 // (op 31229) 
v_lshrrev_b32 v11, 4, v10 // (op 31229) 

v_lshlrev_b32 v10, 5, v11 // (op 31229) 

v_lshl_add_u32 v11, v9, 7, v10 // (op 31229) 


v_lshlrev_b32 v9, 2, v11 // (op 31229) 

v_lshrrev_b32 v10, 3, v9 // (op 31229) 

v_and_b32 v9, 4294967232, v1 // (op 30824) 
v_and_b32 v11, 4294967232, v1 // (op 30963) 

s_waitcnt lgkmcnt(0)
s_mul_hi_i32 s0, s16, s47 // (op 30824) 
s_add_i32 s1, s0, s16 // (op 30824) 

s_mul_hi_i32 s0, s16, s47 // (op 30963) 
s_add_i32 s18, s0, s16 // (op 30963) 

v_and_b32 v12, 63, v1 // (op 30824) 
s_ashr_i32 s0, s1, 31 // (op 30824) 
v_lshrrev_b32 v13, 4, v12 // (op 30824) 
s_cmp_eq_i32 s47, 0 // (op 30824) 
s_mov_b32 s19, scc // (op 30824) 
v_lshlrev_b32 v14, 2, v13 // (op 30824) 

s_cmp_eq_i32 s19, 0 // (op 30824) 
s_cselect_b32 s51, 0, -1 // (op 30824) 

v_add_u32 v13, v9, v14 // (op 30824) 


s_and_b32 s19, s11, 31 // (op 30824) 
v_lshrrev_b32 v9, 2, v12 // (op 30824) 

v_and_b32 v12, 3, v9 // (op 30824) 

v_and_b32 v9, 63, v1 // (op 30963) 
v_lshrrev_b32 v14, 4, v9 // (op 30963) 
s_lshl_b32 s52, 1, s19 // (op 30824) 
v_and_b32 v15, 3, v1 // (op 30824) 
s_add_i32 s53, s52, s51 // (op 30824) 


s_and_b32 s51, s0, s53 // (op 30824) 


s_ashr_i32 s0, s18, 31 // (op 30963) 
s_cmp_eq_i32 s47, 0 // (op 30963) 
s_mov_b32 s52, scc // (op 30963) 
v_lshl_add_u32 v16, v12, 2, v15 // (op 30824) 


s_add_i32 s53, s1, s51 // (op 30824) 


s_ashr_i32 s1, s53, s19 // (op 30824) 


s_lshl_b32 s19, s11, 24 // (op 30824) 
s_ashr_i32 s51, s19, 31 // (op 30824) 

s_xor_b32 s19, s1, s51 // (op 30824) 

s_sub_i32 s1, s19, s51 // (op 30824) 


s_mul_hi_i32 s19, s1, s12 // (op 30824) 
s_add_i32 s51, s19, s1 // (op 30824) 

s_ashr_i32 s19, s51, 31 // (op 30824) 
s_cmp_eq_i32 s12, 0 // (op 30824) 
s_mov_b32 s53, scc // (op 30824) 
s_cmp_eq_i32 s53, 0 // (op 30824) 
s_cselect_b32 s54, 0, -1 // (op 30824) 

s_and_b32 s53, s13, 31 // (op 30824) 
v_lshlrev_b32 v12, 2, v14 // (op 30963) 

s_cmp_eq_i32 s52, 0 // (op 30963) 
s_cselect_b32 s55, 0, -1 // (op 30963) 

s_lshl_b32 s52, 1, s53 // (op 30824) 
v_add_u32 v14, v11, v12 // (op 30963) 


v_lshrrev_b32 v15, 2, v9 // (op 30963) 

s_add_i32 s56, s52, s54 // (op 30824) 


s_and_b32 s52, s19, s56 // (op 30824) 


s_add_i32 s19, s51, s52 // (op 30824) 


s_ashr_i32 s54, s19, s53 // (op 30824) 


s_lshl_b32 s19, s13, 24 // (op 30824) 
s_ashr_i32 s51, s19, 31 // (op 30824) 

s_xor_b32 s19, s54, s51 // (op 30824) 

s_sub_i32 s52, s19, s51 // (op 30824) 


s_mul_i32 s19, s1, s48 // (op 30824) 
s_sub_u32 s51, s16, s19 // (op 30824) 

s_mul_i32 s19, s52, s48 // (op 30824) 
s_add_u32 s19, s19, s51 // (op 30824) 

s_mul_hi_i32 s51, s16, s10 // (op 30824) 
s_add_i32 s53, s51, s16 // (op 30824) 

s_ashr_i32 s51, s53, 31 // (op 30824) 
s_cmp_eq_i32 s10, 0 // (op 30824) 
s_mov_b32 s54, scc // (op 30824) 
s_cmp_eq_i32 s54, 0 // (op 30824) 
s_cselect_b32 s56, 0, -1 // (op 30824) 

s_and_b32 s54, s44, 31 // (op 30824) 
s_lshl_b32 s57, 1, s54 // (op 30824) 
s_add_i32 s58, s57, s56 // (op 30824) 


s_and_b32 s56, s51, s58 // (op 30824) 


s_add_i32 s51, s53, s56 // (op 30824) 


s_ashr_i32 s53, s51, s54 // (op 30824) 


s_lshl_b32 s51, s44, 24 // (op 30824) 
s_ashr_i32 s54, s51, 31 // (op 30824) 

s_xor_b32 s51, s53, s54 // (op 30824) 

s_sub_i32 s53, s51, s54 // (op 30824) 


s_mul_i32 s51, s48, s6 // (op 30824) 
s_mul_i32 s54, s14, s51 // (op 30824) 

s_mul_i32 s51, s53, s54 // (op 30824) 

s_sub_u32 s54, s16, s51 // (op 30824) 

s_mov_b32 s51, s54 // (op 30824) 
s_mul_hi_i32 s56, s51, s45 // (op 30824) 
s_add_i32 s57, s56, s51 // (op 30824) 


s_ashr_i32 s51, s57, 31 // (op 30824) 
s_cmp_eq_i32 s45, 0 // (op 30824) 
s_mov_b32 s56, scc // (op 30824) 
s_cmp_eq_i32 s56, 0 // (op 30824) 
s_cselect_b32 s58, 0, -1 // (op 30824) 

s_and_b32 s56, s15, 31 // (op 30824) 
s_lshl_b32 s59, 1, s56 // (op 30824) 
s_add_i32 s60, s59, s58 // (op 30824) 


s_and_b32 s58, s51, s60 // (op 30824) 


s_add_i32 s51, s57, s58 // (op 30824) 


s_ashr_i32 s57, s51, s56 // (op 30824) 


s_lshl_b32 s51, s15, 24 // (op 30824) 
s_ashr_i32 s56, s51, 31 // (op 30824) 

s_xor_b32 s51, s57, s56 // (op 30824) 

s_sub_i32 s57, s51, s56 // (op 30824) 


s_mul_i32 s51, s46, s48 // (op 30824) 
s_and_b32 s56, s11, 31 // (op 30963) 
s_lshl_b32 s58, 1, s56 // (op 30963) 
v_and_b32 v9, 3, v15 // (op 30963) 

s_add_i32 s59, s58, s55 // (op 30963) 


v_and_b32 v15, 3, v1 // (op 30963) 
s_and_b32 s55, s0, s59 // (op 30963) 


v_lshl_add_u32 v11, v9, 2, v15 // (op 30963) 


s_add_i32 s0, s18, s55 // (op 30963) 


s_ashr_i32 s18, s0, s56 // (op 30963) 


s_lshl_b32 s0, s11, 24 // (op 30963) 
s_ashr_i32 s55, s0, 31 // (op 30963) 

s_xor_b32 s0, s18, s55 // (op 30963) 

s_sub_i32 s18, s0, s55 // (op 30963) 


s_mul_hi_i32 s0, s18, s12 // (op 30963) 
s_add_i32 s55, s0, s18 // (op 30963) 

s_ashr_i32 s0, s55, 31 // (op 30963) 
s_cmp_eq_i32 s12, 0 // (op 30963) 
s_mov_b32 s56, scc // (op 30963) 
s_cmp_eq_i32 s56, 0 // (op 30963) 
s_cselect_b32 s58, 0, -1 // (op 30963) 

s_and_b32 s56, s13, 31 // (op 30963) 
s_lshl_b32 s59, 1, s56 // (op 30963) 
s_add_i32 s60, s59, s58 // (op 30963) 


s_and_b32 s58, s0, s60 // (op 30963) 


s_add_i32 s0, s55, s58 // (op 30963) 


s_ashr_i32 s55, s0, s56 // (op 30963) 


s_lshl_b32 s0, s13, 24 // (op 30963) 
s_ashr_i32 s56, s0, 31 // (op 30963) 

s_xor_b32 s0, s55, s56 // (op 30963) 

s_sub_i32 s55, s0, s56 // (op 30963) 


s_mul_i32 s0, s18, s48 // (op 30963) 
s_sub_u32 s56, s16, s0 // (op 30963) 

s_mul_i32 s0, s55, s48 // (op 30963) 
s_add_u32 s0, s0, s56 // (op 30963) 

s_mul_hi_i32 s56, s16, s10 // (op 30963) 
s_add_i32 s58, s56, s16 // (op 30963) 

s_ashr_i32 s56, s58, 31 // (op 30963) 
s_cmp_eq_i32 s10, 0 // (op 30963) 
s_mov_b32 s59, scc // (op 30963) 
s_cmp_eq_i32 s59, 0 // (op 30963) 
s_cselect_b32 s60, 0, -1 // (op 30963) 

s_and_b32 s59, s44, 31 // (op 30963) 
s_lshl_b32 s61, 1, s59 // (op 30963) 
s_add_i32 s62, s61, s60 // (op 30963) 


s_and_b32 s60, s56, s62 // (op 30963) 


s_add_i32 s56, s58, s60 // (op 30963) 


s_ashr_i32 s58, s56, s59 // (op 30963) 


s_lshl_b32 s56, s44, 24 // (op 30963) 
s_ashr_i32 s59, s56, 31 // (op 30963) 

s_xor_b32 s56, s58, s59 // (op 30963) 

s_sub_i32 s58, s56, s59 // (op 30963) 


s_mul_i32 s56, s48, s6 // (op 30963) 
s_mul_i32 s59, s14, s56 // (op 30963) 

s_mul_i32 s56, s58, s59 // (op 30963) 

s_sub_u32 s59, s16, s56 // (op 30963) 

s_mov_b32 s56, s59 // (op 30963) 
s_mul_hi_i32 s60, s56, s45 // (op 30963) 
s_add_i32 s61, s60, s56 // (op 30963) 


s_ashr_i32 s56, s61, 31 // (op 30963) 
s_cmp_eq_i32 s45, 0 // (op 30963) 
s_mov_b32 s60, scc // (op 30963) 
s_cmp_eq_i32 s60, 0 // (op 30963) 
s_cselect_b32 s62, 0, -1 // (op 30963) 

s_and_b32 s60, s15, 31 // (op 30963) 
s_lshl_b32 s63, 1, s60 // (op 30963) 
s_add_i32 s64, s63, s62 // (op 30963) 


s_and_b32 s62, s56, s64 // (op 30963) 


s_add_i32 s56, s61, s62 // (op 30963) 


s_ashr_i32 s61, s56, s60 // (op 30963) 


s_lshl_b32 s56, s15, 24 // (op 30963) 
s_ashr_i32 s60, s56, 31 // (op 30963) 

s_xor_b32 s56, s61, s60 // (op 30963) 

s_sub_i32 s61, s56, s60 // (op 30963) 


s_mul_i32 s56, s46, s48 // (op 30963) 
s_add_u32 s62, s8, 256 // (op 30824) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 30824) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 30824) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 30824) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 8 // (op 30824) 

s_sub_i32 s60, s62, s51 // (op 30824) 

s_mul_i32 s62, s57, s60 // (op 30824) 

s_sub_u32 s60, s54, s62 // (op 30824) 


s_add_u32 s54, s60, s51 // (op 30824) 


s_cmp_eq_i32 s53, 0 // (op 30824) 
s_mov_b32 s51, scc // (op 30824) 

s_cmp_eq_i32 s51, 0 // (op 30824) 
s_cselect_b32 s53, s54, s19 // (op 30824) 


s_mul_i32 s19, s52, s6 // (op 30824) 

v_lshl_add_u32 v9, s53, 8, v13 // (op 30824) 


s_sub_i32 s60, s1, s19 // (op 30824) 


s_cmp_eq_i32 s51, 0 // (op 30824) 
s_cselect_b32 s1, s57, s60 // (op 30824) 



v_mov_b32 v15, 192 // (op 30824) 
v_mul_lo_u32 v12, s1, v15 // (op 30824) 

v_add_u32 v12, v12, v16 // (op 30824) 


v_mul_lo_u32 v13, v12, s2 // (op 30824) 

v_add_lshl_u32 v12, v9, v13, 1 // (op 30824) 


s_add_u32 s52, s8, 256 // (op 30963) Start of Int64 add, locking SCC
s_addc_u32 s53, s9, 0 // (op 30963) End of Int64 add, unlocking SCC
s_sub_u32 s62, s52, 1 // (op 30963) least significant half; sets scc
s_subb_u32 s63, s53, 0 // (op 30963) most significant half; uses scc

s_ashr_i64 s[52:53], s[62:63], 8 // (op 30963) 

s_sub_i32 s1, s52, s56 // (op 30963) 

s_mul_i32 s19, s61, s1 // (op 30963) 

s_sub_u32 s1, s59, s19 // (op 30963) 


s_add_u32 s19, s1, s56 // (op 30963) 


s_cmp_eq_i32 s58, 0 // (op 30963) 
s_mov_b32 s1, scc // (op 30963) 

s_cmp_eq_i32 s1, 0 // (op 30963) 
s_cselect_b32 s51, s19, s0 // (op 30963) 


s_mul_i32 s0, s55, s6 // (op 30963) 

v_lshl_add_u32 v9, s51, 8, v14 // (op 30963) 


s_sub_i32 s19, s18, s0 // (op 30963) 


s_cmp_eq_i32 s1, 0 // (op 30963) 
s_cselect_b32 s0, s61, s19 // (op 30963) 



v_mov_b32 v13, 192 // (op 30963) 
v_mul_lo_u32 v14, s0, v13 // (op 30963) 

v_add_u32 v14, v14, v11 // (op 30963) 


v_mul_lo_u32 v11, v14, s4 // (op 30963) 

v_add_lshl_u32 v13, v9, v11, 1 // (op 30963) 


s_mul_i32 s1, s2, s23 // (op 30826) most significant: low of low * high
s_mul_i32 s18, s3, s22 // (op 30826) most significant: low of high * low
s_mul_i32 s0, s2, s22 // (op 30826) least significant
s_mul_hi_u32 s19, s2, s22 // (op 30826) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30826) most significant: sum
s_add_u32 s1, s1, s19 // (op 30826) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 30826) 

s_mul_i32 s1, s4, s23 // (op 30965) most significant: low of low * high
s_mul_i32 s51, s5, s22 // (op 30965) most significant: low of high * low
s_mul_i32 s0, s4, s22 // (op 30965) least significant
s_mul_hi_u32 s52, s4, s22 // (op 30965) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 30965) most significant: sum
s_add_u32 s1, s1, s52 // (op 30965) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 30965) 

s_mov_b32 s56, s20 // (op 30826) reg expression
s_mov_b32 s57, s21 // (op 30826) reg expression
s_mov_b32 s58, s18 // (op 30826) reg expression
s_mov_b32 s59, 131072 // (op 30826) call()

s_mov_b32 s60, s24 // (op 30965) reg expression

s_mov_b32 s61, s25 // (op 30965) reg expression
s_mov_b32 s62, s52 // (op 30965) reg expression
s_mov_b32 s63, 131072 // (op 30965) call()


v_mov_b32 v9, s49 // (op 2) Move value
v_mov_b32 v11, s50 // (op 35) Move value


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
v_accvgpr_write a32, 0.00000 // (op 1969) call()
v_accvgpr_write a33, 0.00000 // (op 1969) call()
v_accvgpr_write a34, 0.00000 // (op 1969) call()
v_accvgpr_write a35, 0.00000 // (op 1969) call()
v_accvgpr_write a36, 0.00000 // (op 2195) call()
v_accvgpr_write a37, 0.00000 // (op 2195) call()
v_accvgpr_write a38, 0.00000 // (op 2195) call()
v_accvgpr_write a39, 0.00000 // (op 2195) call()
v_accvgpr_write a40, 0.00000 // (op 2421) call()
v_accvgpr_write a41, 0.00000 // (op 2421) call()
v_accvgpr_write a42, 0.00000 // (op 2421) call()
v_accvgpr_write a43, 0.00000 // (op 2421) call()
v_accvgpr_write a44, 0.00000 // (op 2647) call()
v_accvgpr_write a45, 0.00000 // (op 2647) call()
v_accvgpr_write a46, 0.00000 // (op 2647) call()
v_accvgpr_write a47, 0.00000 // (op 2647) call()
v_accvgpr_write a48, 0.00000 // (op 3281) call()
v_accvgpr_write a49, 0.00000 // (op 3281) call()
v_accvgpr_write a50, 0.00000 // (op 3281) call()
v_accvgpr_write a51, 0.00000 // (op 3281) call()
v_accvgpr_write a52, 0.00000 // (op 4518) call()
v_accvgpr_write a53, 0.00000 // (op 4518) call()
v_accvgpr_write a54, 0.00000 // (op 4518) call()
v_accvgpr_write a55, 0.00000 // (op 4518) call()
v_accvgpr_write a56, 0.00000 // (op 4519) call()
v_accvgpr_write a57, 0.00000 // (op 4519) call()
v_accvgpr_write a58, 0.00000 // (op 4519) call()
v_accvgpr_write a59, 0.00000 // (op 4519) call()
v_accvgpr_write a60, 0.00000 // (op 4520) call()
v_accvgpr_write a61, 0.00000 // (op 4520) call()
v_accvgpr_write a62, 0.00000 // (op 4520) call()
v_accvgpr_write a63, 0.00000 // (op 4520) call()
v_accvgpr_write a64, 0.00000 // (op 4521) call()
v_accvgpr_write a65, 0.00000 // (op 4521) call()
v_accvgpr_write a66, 0.00000 // (op 4521) call()
v_accvgpr_write a67, 0.00000 // (op 4521) call()
v_accvgpr_write a68, 0.00000 // (op 4522) call()
v_accvgpr_write a69, 0.00000 // (op 4522) call()
v_accvgpr_write a70, 0.00000 // (op 4522) call()
v_accvgpr_write a71, 0.00000 // (op 4522) call()
v_accvgpr_write a72, 0.00000 // (op 4523) call()
v_accvgpr_write a73, 0.00000 // (op 4523) call()
v_accvgpr_write a74, 0.00000 // (op 4523) call()
v_accvgpr_write a75, 0.00000 // (op 4523) call()
v_accvgpr_write a76, 0.00000 // (op 4524) call()
v_accvgpr_write a77, 0.00000 // (op 4524) call()
v_accvgpr_write a78, 0.00000 // (op 4524) call()
v_accvgpr_write a79, 0.00000 // (op 4524) call()
v_accvgpr_write a80, 0.00000 // (op 4525) call()
v_accvgpr_write a81, 0.00000 // (op 4525) call()
v_accvgpr_write a82, 0.00000 // (op 4525) call()
v_accvgpr_write a83, 0.00000 // (op 4525) call()
v_accvgpr_write a84, 0.00000 // (op 4526) call()
v_accvgpr_write a85, 0.00000 // (op 4526) call()
v_accvgpr_write a86, 0.00000 // (op 4526) call()
v_accvgpr_write a87, 0.00000 // (op 4526) call()
v_accvgpr_write a88, 0.00000 // (op 4527) call()
v_accvgpr_write a89, 0.00000 // (op 4527) call()
v_accvgpr_write a90, 0.00000 // (op 4527) call()
v_accvgpr_write a91, 0.00000 // (op 4527) call()
v_accvgpr_write a92, 0.00000 // (op 4528) call()
v_accvgpr_write a93, 0.00000 // (op 4528) call()
v_accvgpr_write a94, 0.00000 // (op 4528) call()
v_accvgpr_write a95, 0.00000 // (op 4528) call()
v_accvgpr_write a96, 0.00000 // (op 6283) call()
v_accvgpr_write a97, 0.00000 // (op 6283) call()
v_accvgpr_write a98, 0.00000 // (op 6283) call()
v_accvgpr_write a99, 0.00000 // (op 6283) call()
v_accvgpr_write a100, 0.00000 // (op 7520) call()
v_accvgpr_write a101, 0.00000 // (op 7520) call()
v_accvgpr_write a102, 0.00000 // (op 7520) call()
v_accvgpr_write a103, 0.00000 // (op 7520) call()
v_accvgpr_write a104, 0.00000 // (op 7521) call()
v_accvgpr_write a105, 0.00000 // (op 7521) call()
v_accvgpr_write a106, 0.00000 // (op 7521) call()
v_accvgpr_write a107, 0.00000 // (op 7521) call()
v_accvgpr_write a108, 0.00000 // (op 7522) call()
v_accvgpr_write a109, 0.00000 // (op 7522) call()
v_accvgpr_write a110, 0.00000 // (op 7522) call()
v_accvgpr_write a111, 0.00000 // (op 7522) call()
v_accvgpr_write a112, 0.00000 // (op 7523) call()
v_accvgpr_write a113, 0.00000 // (op 7523) call()
v_accvgpr_write a114, 0.00000 // (op 7523) call()
v_accvgpr_write a115, 0.00000 // (op 7523) call()
v_accvgpr_write a116, 0.00000 // (op 7524) call()
v_accvgpr_write a117, 0.00000 // (op 7524) call()
v_accvgpr_write a118, 0.00000 // (op 7524) call()
v_accvgpr_write a119, 0.00000 // (op 7524) call()
v_accvgpr_write a120, 0.00000 // (op 7525) call()
v_accvgpr_write a121, 0.00000 // (op 7525) call()
v_accvgpr_write a122, 0.00000 // (op 7525) call()
v_accvgpr_write a123, 0.00000 // (op 7525) call()
v_accvgpr_write a124, 0.00000 // (op 7526) call()
v_accvgpr_write a125, 0.00000 // (op 7526) call()
v_accvgpr_write a126, 0.00000 // (op 7526) call()
v_accvgpr_write a127, 0.00000 // (op 7526) call()
v_accvgpr_write a128, 0.00000 // (op 7527) call()
v_accvgpr_write a129, 0.00000 // (op 7527) call()
v_accvgpr_write a130, 0.00000 // (op 7527) call()
v_accvgpr_write a131, 0.00000 // (op 7527) call()
v_accvgpr_write a132, 0.00000 // (op 7528) call()
v_accvgpr_write a133, 0.00000 // (op 7528) call()
v_accvgpr_write a134, 0.00000 // (op 7528) call()
v_accvgpr_write a135, 0.00000 // (op 7528) call()
v_accvgpr_write a136, 0.00000 // (op 7529) call()
v_accvgpr_write a137, 0.00000 // (op 7529) call()
v_accvgpr_write a138, 0.00000 // (op 7529) call()
v_accvgpr_write a139, 0.00000 // (op 7529) call()
v_accvgpr_write a140, 0.00000 // (op 7530) call()
v_accvgpr_write a141, 0.00000 // (op 7530) call()
v_accvgpr_write a142, 0.00000 // (op 7530) call()
v_accvgpr_write a143, 0.00000 // (op 7530) call()
v_accvgpr_write a144, 0.00000 // (op 9285) call()
v_accvgpr_write a145, 0.00000 // (op 9285) call()
v_accvgpr_write a146, 0.00000 // (op 9285) call()
v_accvgpr_write a147, 0.00000 // (op 9285) call()
v_accvgpr_write a148, 0.00000 // (op 10522) call()
v_accvgpr_write a149, 0.00000 // (op 10522) call()
v_accvgpr_write a150, 0.00000 // (op 10522) call()
v_accvgpr_write a151, 0.00000 // (op 10522) call()
v_accvgpr_write a152, 0.00000 // (op 10523) call()
v_accvgpr_write a153, 0.00000 // (op 10523) call()
v_accvgpr_write a154, 0.00000 // (op 10523) call()
v_accvgpr_write a155, 0.00000 // (op 10523) call()
v_accvgpr_write a156, 0.00000 // (op 10524) call()
v_accvgpr_write a157, 0.00000 // (op 10524) call()
v_accvgpr_write a158, 0.00000 // (op 10524) call()
v_accvgpr_write a159, 0.00000 // (op 10524) call()
v_accvgpr_write a160, 0.00000 // (op 10525) call()
v_accvgpr_write a161, 0.00000 // (op 10525) call()
v_accvgpr_write a162, 0.00000 // (op 10525) call()
v_accvgpr_write a163, 0.00000 // (op 10525) call()
v_accvgpr_write a164, 0.00000 // (op 10526) call()
v_accvgpr_write a165, 0.00000 // (op 10526) call()
v_accvgpr_write a166, 0.00000 // (op 10526) call()
v_accvgpr_write a167, 0.00000 // (op 10526) call()
v_accvgpr_write a168, 0.00000 // (op 10527) call()
v_accvgpr_write a169, 0.00000 // (op 10527) call()
v_accvgpr_write a170, 0.00000 // (op 10527) call()
v_accvgpr_write a171, 0.00000 // (op 10527) call()
v_accvgpr_write a172, 0.00000 // (op 10528) call()
v_accvgpr_write a173, 0.00000 // (op 10528) call()
v_accvgpr_write a174, 0.00000 // (op 10528) call()
v_accvgpr_write a175, 0.00000 // (op 10528) call()
v_accvgpr_write a176, 0.00000 // (op 10529) call()
v_accvgpr_write a177, 0.00000 // (op 10529) call()
v_accvgpr_write a178, 0.00000 // (op 10529) call()
v_accvgpr_write a179, 0.00000 // (op 10529) call()
v_accvgpr_write a180, 0.00000 // (op 10530) call()
v_accvgpr_write a181, 0.00000 // (op 10530) call()
v_accvgpr_write a182, 0.00000 // (op 10530) call()
v_accvgpr_write a183, 0.00000 // (op 10530) call()
v_accvgpr_write a184, 0.00000 // (op 10531) call()
v_accvgpr_write a185, 0.00000 // (op 10531) call()
v_accvgpr_write a186, 0.00000 // (op 10531) call()
v_accvgpr_write a187, 0.00000 // (op 10531) call()
v_accvgpr_write a188, 0.00000 // (op 10532) call()
v_accvgpr_write a189, 0.00000 // (op 10532) call()
v_accvgpr_write a190, 0.00000 // (op 10532) call()
v_accvgpr_write a191, 0.00000 // (op 10532) call()
s_mul_hi_i32 s0, s16, s47 // (op 30714) 
v_lshrrev_b32 v14, 2, v1 // (op 30714) 
s_add_i32 s1, s0, s16 // (op 30714) 

v_and_b32 v15, -4, v1 // (op 30714) 
s_ashr_i32 s0, s1, 31 // (op 30714) 
v_sub_u32 v16, v1, v15 // (op 30714) 

s_cmp_eq_i32 s47, 0 // (op 30714) 
s_mov_b32 s18, scc // (op 30714) 
v_lshlrev_b32 v15, 5, v16 // (op 30714) 

s_cmp_eq_i32 s18, 0 // (op 30714) 
s_cselect_b32 s19, 0, -1 // (op 30714) 

v_mov_b32 v17, 0 // (op 30714) convert
v_mov_b32 v16, v15 // (op 30714) convert

s_and_b32 s18, s11, 31 // (op 30714) 
s_lshl_b32 s20, 1, s18 // (op 30714) 
s_add_i32 s21, s20, s19 // (op 30714) 


s_and_b32 s19, s0, s21 // (op 30714) 


s_add_i32 s0, s1, s19 // (op 30714) 


s_ashr_i32 s1, s0, s18 // (op 30714) 


s_lshl_b32 s0, s11, 24 // (op 30714) 
s_ashr_i32 s18, s0, 31 // (op 30714) 

s_xor_b32 s0, s1, s18 // (op 30714) 

s_sub_i32 s1, s0, s18 // (op 30714) 


s_mul_hi_i32 s0, s1, s12 // (op 30714) 
s_add_i32 s18, s0, s1 // (op 30714) 

s_ashr_i32 s0, s18, 31 // (op 30714) 
s_cmp_eq_i32 s12, 0 // (op 30714) 
s_mov_b32 s19, scc // (op 30714) 
s_cmp_eq_i32 s19, 0 // (op 30714) 
s_cselect_b32 s20, 0, -1 // (op 30714) 

s_and_b32 s19, s13, 31 // (op 30714) 
s_lshl_b32 s21, 1, s19 // (op 30714) 
s_add_i32 s24, s21, s20 // (op 30714) 


s_and_b32 s25, s0, s24 // (op 30714) 


s_add_i32 s0, s18, s25 // (op 30714) 


s_ashr_i32 s18, s0, s19 // (op 30714) 


s_lshl_b32 s0, s13, 24 // (op 30714) 
s_ashr_i32 s19, s0, 31 // (op 30714) 

s_xor_b32 s0, s18, s19 // (op 30714) 

s_sub_i32 s18, s0, s19 // (op 30714) 


s_mul_i32 s0, s1, s48 // (op 30714) 

s_sub_u32 s1, s16, s0 // (op 30714) 

s_mul_i32 s0, s18, s48 // (op 30714) 
s_add_u32 s0, s0, s1 // (op 30714) 


s_mul_hi_i32 s1, s16, s10 // (op 30714) 
s_add_i32 s18, s1, s16 // (op 30714) 

s_ashr_i32 s1, s18, 31 // (op 30714) 
s_cmp_eq_i32 s10, 0 // (op 30714) 
s_mov_b32 s19, scc // (op 30714) 
s_cmp_eq_i32 s19, 0 // (op 30714) 
s_cselect_b32 s20, 0, -1 // (op 30714) 

s_and_b32 s19, s44, 31 // (op 30714) 
s_lshl_b32 s21, 1, s19 // (op 30714) 
s_add_i32 s24, s21, s20 // (op 30714) 


s_and_b32 s25, s1, s24 // (op 30714) 


s_add_i32 s1, s18, s25 // (op 30714) 


s_ashr_i32 s18, s1, s19 // (op 30714) 


s_lshl_b32 s1, s44, 24 // (op 30714) 
s_ashr_i32 s19, s1, 31 // (op 30714) 

s_xor_b32 s1, s18, s19 // (op 30714) 

s_sub_i32 s18, s1, s19 // (op 30714) 


s_mul_i32 s1, s48, s6 // (op 30714) 
s_mul_i32 s19, s14, s1 // (op 30714) 

s_mul_i32 s1, s18, s19 // (op 30714) 

s_sub_u32 s19, s16, s1 // (op 30714) 

s_mov_b32 s1, s19 // (op 30714) 
s_mul_hi_i32 s20, s1, s45 // (op 30714) 
s_add_i32 s21, s20, s1 // (op 30714) 


s_ashr_i32 s1, s21, 31 // (op 30714) 
s_cmp_eq_i32 s45, 0 // (op 30714) 
s_mov_b32 s20, scc // (op 30714) 
s_cmp_eq_i32 s20, 0 // (op 30714) 
s_cselect_b32 s24, 0, -1 // (op 30714) 

s_and_b32 s20, s15, 31 // (op 30714) 
s_lshl_b32 s25, 1, s20 // (op 30714) 
s_add_i32 s49, s25, s24 // (op 30714) 


s_and_b32 s24, s1, s49 // (op 30714) 


s_add_i32 s1, s21, s24 // (op 30714) 


s_ashr_i32 s21, s1, s20 // (op 30714) 


s_lshl_b32 s1, s15, 24 // (op 30714) 
s_ashr_i32 s20, s1, 31 // (op 30714) 

s_xor_b32 s1, s21, s20 // (op 30714) 

s_sub_i32 s21, s1, s20 // (op 30714) 


s_mul_i32 s1, s46, s48 // (op 30714) 
s_add_u32 s24, s8, 256 // (op 30714) Start of Int64 add, locking SCC
s_addc_u32 s25, s9, 0 // (op 30714) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 30714) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 30714) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 8 // (op 30714) 

s_sub_i32 s20, s24, s1 // (op 30714) 

s_mul_i32 s24, s21, s20 // (op 30714) 


s_sub_u32 s25, s19, s24 // (op 30714) 


s_add_u32 s24, s25, s1 // (op 30714) 


s_cmp_eq_i32 s18, 0 // (op 30714) 
s_mov_b32 s1, scc // (op 30714) 

s_cmp_eq_i32 s1, 0 // (op 30714) 
s_cselect_b32 s25, s24, s0 // (op 30714) 



v_lshl_add_u32 v15, s25, 8, v14 // (op 30714) 


v_mul_lo_u32 v19, v15, s27 // (op 30714) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 30714) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 30714) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 30714) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 30714) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 30714) most significant half



v_lshlrev_b64 v[14:15], 2, v[18:19] // (op 30714) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 30714) 

s_mul_i32 s1, s26, s9 // (op 30716) most significant: low of low * high
s_mul_i32 s18, s27, s8 // (op 30716) most significant: low of high * low
s_mul_i32 s0, s26, s8 // (op 30716) least significant
s_mul_hi_u32 s19, s26, s8 // (op 30716) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30716) most significant: sum
s_add_u32 s1, s1, s19 // (op 30716) most significant: sum


s_lshl_b64 s[24:25], s[0:1], 2 // (op 30716) 

s_ashr_i64 s[0:1], s[24:25], 3 // (op 30716) 

s_mov_b32 s52, s28 // (op 30716) reg expression
s_mov_b32 s53, s29 // (op 30716) reg expression
s_mov_b32 s54, s0 // (op 30716) reg expression
s_mov_b32 s55, 131072 // (op 30716) call()

s_mul_hi_i32 s0, s16, s47 // (op 30733) 
v_lshrrev_b32 v14, 2, v1 // (op 30733) 
s_add_i32 s1, s0, s16 // (op 30733) 

v_and_b32 v15, -4, v1 // (op 30733) 
s_ashr_i32 s0, s1, 31 // (op 30733) 
v_sub_u32 v18, v1, v15 // (op 30733) 

s_cmp_eq_i32 s47, 0 // (op 30733) 
s_mov_b32 s18, scc // (op 30733) 
v_lshlrev_b32 v15, 5, v18 // (op 30733) 

s_cmp_eq_i32 s18, 0 // (op 30733) 
s_cselect_b32 s19, 0, -1 // (op 30733) 

v_mov_b32 v19, 0 // (op 30733) convert
v_mov_b32 v18, v15 // (op 30733) convert

s_and_b32 s18, s11, 31 // (op 30733) 
s_lshl_b32 s20, 1, s18 // (op 30733) 
s_add_i32 s21, s20, s19 // (op 30733) 


s_and_b32 s19, s0, s21 // (op 30733) 


s_add_i32 s0, s1, s19 // (op 30733) 


s_ashr_i32 s1, s0, s18 // (op 30733) 


s_lshl_b32 s0, s11, 24 // (op 30733) 
s_ashr_i32 s18, s0, 31 // (op 30733) 

s_xor_b32 s0, s1, s18 // (op 30733) 

s_sub_i32 s1, s0, s18 // (op 30733) 


s_mul_hi_i32 s0, s1, s12 // (op 30733) 
s_add_i32 s18, s0, s1 // (op 30733) 

s_ashr_i32 s0, s18, 31 // (op 30733) 
s_cmp_eq_i32 s12, 0 // (op 30733) 
s_mov_b32 s19, scc // (op 30733) 
s_cmp_eq_i32 s19, 0 // (op 30733) 
s_cselect_b32 s20, 0, -1 // (op 30733) 

s_and_b32 s19, s13, 31 // (op 30733) 
s_lshl_b32 s21, 1, s19 // (op 30733) 
s_add_i32 s24, s21, s20 // (op 30733) 


s_and_b32 s25, s0, s24 // (op 30733) 


s_add_i32 s0, s18, s25 // (op 30733) 


s_ashr_i32 s18, s0, s19 // (op 30733) 


s_lshl_b32 s0, s13, 24 // (op 30733) 
s_ashr_i32 s19, s0, 31 // (op 30733) 

s_xor_b32 s0, s18, s19 // (op 30733) 

s_sub_i32 s18, s0, s19 // (op 30733) 


s_mul_i32 s0, s1, s48 // (op 30733) 

s_sub_u32 s1, s16, s0 // (op 30733) 

s_mul_i32 s0, s18, s48 // (op 30733) 
s_add_u32 s0, s0, s1 // (op 30733) 


s_mul_hi_i32 s1, s16, s10 // (op 30733) 
s_add_i32 s18, s1, s16 // (op 30733) 

s_ashr_i32 s1, s18, 31 // (op 30733) 
s_cmp_eq_i32 s10, 0 // (op 30733) 
s_mov_b32 s19, scc // (op 30733) 
s_cmp_eq_i32 s19, 0 // (op 30733) 
s_cselect_b32 s20, 0, -1 // (op 30733) 

s_and_b32 s19, s44, 31 // (op 30733) 
s_lshl_b32 s21, 1, s19 // (op 30733) 
s_add_i32 s24, s21, s20 // (op 30733) 


s_and_b32 s25, s1, s24 // (op 30733) 


s_add_i32 s1, s18, s25 // (op 30733) 


s_ashr_i32 s18, s1, s19 // (op 30733) 


s_lshl_b32 s1, s44, 24 // (op 30733) 
s_ashr_i32 s19, s1, 31 // (op 30733) 

s_xor_b32 s1, s18, s19 // (op 30733) 

s_sub_i32 s18, s1, s19 // (op 30733) 


s_mul_i32 s1, s48, s6 // (op 30733) 
s_mul_i32 s19, s14, s1 // (op 30733) 

s_mul_i32 s1, s18, s19 // (op 30733) 

s_sub_u32 s19, s16, s1 // (op 30733) 

s_mov_b32 s1, s19 // (op 30733) 
s_mul_hi_i32 s20, s1, s45 // (op 30733) 
s_add_i32 s21, s20, s1 // (op 30733) 


s_ashr_i32 s1, s21, 31 // (op 30733) 
s_cmp_eq_i32 s45, 0 // (op 30733) 
s_mov_b32 s20, scc // (op 30733) 
s_cmp_eq_i32 s20, 0 // (op 30733) 
s_cselect_b32 s24, 0, -1 // (op 30733) 

s_and_b32 s20, s15, 31 // (op 30733) 
s_lshl_b32 s25, 1, s20 // (op 30733) 
s_add_i32 s49, s25, s24 // (op 30733) 


s_and_b32 s24, s1, s49 // (op 30733) 


s_add_i32 s1, s21, s24 // (op 30733) 


s_ashr_i32 s21, s1, s20 // (op 30733) 


s_lshl_b32 s1, s15, 24 // (op 30733) 
s_ashr_i32 s20, s1, 31 // (op 30733) 

s_xor_b32 s1, s21, s20 // (op 30733) 

s_sub_i32 s21, s1, s20 // (op 30733) 


s_mul_i32 s1, s46, s48 // (op 30733) 
s_add_u32 s24, s8, 256 // (op 30733) Start of Int64 add, locking SCC
s_addc_u32 s25, s9, 0 // (op 30733) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 30733) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 30733) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 30733) 

s_sub_i32 s20, s24, s1 // (op 30733) 

s_mul_i32 s24, s21, s20 // (op 30733) 


s_sub_u32 s25, s19, s24 // (op 30733) 


s_add_u32 s24, s25, s1 // (op 30733) 


s_cmp_eq_i32 s18, 0 // (op 30733) 
s_mov_b32 s1, scc // (op 30733) 

s_cmp_eq_i32 s1, 0 // (op 30733) 
s_cselect_b32 s25, s24, s0 // (op 30733) 



v_lshl_add_u32 v15, s25, 8, v14 // (op 30733) 


v_mul_lo_u32 v21, v15, s27 // (op 30733) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 30733) most significant: high of low * low
v_mul_lo_u32 v20, v15, s26 // (op 30733) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 30733) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 30733) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 30733) most significant half



v_lshlrev_b64 v[14:15], 2, v[20:21] // (op 30733) 

v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 30733) 

s_mul_i32 s1, s26, s9 // (op 30735) most significant: low of low * high
s_mul_i32 s18, s27, s8 // (op 30735) most significant: low of high * low
s_mul_i32 s0, s26, s8 // (op 30735) least significant
s_mul_hi_u32 s19, s26, s8 // (op 30735) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30735) most significant: sum
s_add_u32 s1, s1, s19 // (op 30735) most significant: sum


s_lshl_b64 s[24:25], s[0:1], 2 // (op 30735) 

s_ashr_i64 s[0:1], s[24:25], 3 // (op 30735) 

s_mov_b32 s52, s28 // (op 30735) reg expression
s_mov_b32 s53, s29 // (op 30735) reg expression
s_mov_b32 s54, s0 // (op 30735) reg expression
s_mov_b32 s55, 131072 // (op 30735) call()

s_mul_hi_i32 s0, s16, s47 // (op 30778) 
v_lshrrev_b32 v14, 2, v1 // (op 30778) 
s_add_i32 s1, s0, s16 // (op 30778) 

v_and_b32 v15, -4, v1 // (op 30778) 
s_ashr_i32 s0, s1, 31 // (op 30778) 
v_sub_u32 v20, v1, v15 // (op 30778) 

s_cmp_eq_i32 s47, 0 // (op 30778) 
s_mov_b32 s18, scc // (op 30778) 
v_lshlrev_b32 v15, 5, v20 // (op 30778) 

s_cmp_eq_i32 s18, 0 // (op 30778) 
s_cselect_b32 s19, 0, -1 // (op 30778) 

v_mov_b32 v21, 0 // (op 30778) convert
v_mov_b32 v20, v15 // (op 30778) convert

s_and_b32 s18, s11, 31 // (op 30778) 
s_lshl_b32 s20, 1, s18 // (op 30778) 
s_add_i32 s21, s20, s19 // (op 30778) 


s_and_b32 s19, s0, s21 // (op 30778) 


s_add_i32 s0, s1, s19 // (op 30778) 


s_ashr_i32 s1, s0, s18 // (op 30778) 


s_lshl_b32 s0, s11, 24 // (op 30778) 
s_ashr_i32 s18, s0, 31 // (op 30778) 

s_xor_b32 s0, s1, s18 // (op 30778) 

s_sub_i32 s1, s0, s18 // (op 30778) 


s_mul_hi_i32 s0, s1, s12 // (op 30778) 
s_add_i32 s18, s0, s1 // (op 30778) 

s_ashr_i32 s0, s18, 31 // (op 30778) 
s_cmp_eq_i32 s12, 0 // (op 30778) 
s_mov_b32 s19, scc // (op 30778) 
s_cmp_eq_i32 s19, 0 // (op 30778) 
s_cselect_b32 s20, 0, -1 // (op 30778) 

s_and_b32 s19, s13, 31 // (op 30778) 
s_lshl_b32 s21, 1, s19 // (op 30778) 
s_add_i32 s24, s21, s20 // (op 30778) 


s_and_b32 s25, s0, s24 // (op 30778) 


s_add_i32 s0, s18, s25 // (op 30778) 


s_ashr_i32 s18, s0, s19 // (op 30778) 


s_lshl_b32 s0, s13, 24 // (op 30778) 
s_ashr_i32 s19, s0, 31 // (op 30778) 

s_xor_b32 s0, s18, s19 // (op 30778) 

s_sub_i32 s18, s0, s19 // (op 30778) 


s_mul_i32 s0, s18, s6 // (op 30778) 

s_sub_i32 s18, s1, s0 // (op 30778) 


s_mul_hi_i32 s0, s16, s10 // (op 30778) 
s_add_i32 s1, s0, s16 // (op 30778) 

s_ashr_i32 s0, s1, 31 // (op 30778) 
s_cmp_eq_i32 s10, 0 // (op 30778) 
s_mov_b32 s19, scc // (op 30778) 
s_cmp_eq_i32 s19, 0 // (op 30778) 
s_cselect_b32 s20, 0, -1 // (op 30778) 

s_and_b32 s19, s44, 31 // (op 30778) 
s_lshl_b32 s21, 1, s19 // (op 30778) 
s_add_i32 s24, s21, s20 // (op 30778) 


s_and_b32 s25, s0, s24 // (op 30778) 


s_add_i32 s0, s1, s25 // (op 30778) 


s_ashr_i32 s1, s0, s19 // (op 30778) 


s_lshl_b32 s0, s44, 24 // (op 30778) 
s_ashr_i32 s19, s0, 31 // (op 30778) 

s_xor_b32 s0, s1, s19 // (op 30778) 

s_sub_i32 s1, s0, s19 // (op 30778) 


s_mul_i32 s0, s48, s6 // (op 30778) 
s_mul_i32 s19, s14, s0 // (op 30778) 

s_mul_i32 s0, s1, s19 // (op 30778) 

s_sub_u32 s19, s16, s0 // (op 30778) 

s_mul_hi_i32 s0, s19, s45 // (op 30778) 
s_add_i32 s20, s0, s19 // (op 30778) 


s_ashr_i32 s0, s20, 31 // (op 30778) 
s_cmp_eq_i32 s45, 0 // (op 30778) 
s_mov_b32 s19, scc // (op 30778) 
s_cmp_eq_i32 s19, 0 // (op 30778) 
s_cselect_b32 s21, 0, -1 // (op 30778) 

s_and_b32 s19, s15, 31 // (op 30778) 
s_lshl_b32 s24, 1, s19 // (op 30778) 
s_add_i32 s25, s24, s21 // (op 30778) 


s_and_b32 s21, s0, s25 // (op 30778) 


s_add_i32 s0, s20, s21 // (op 30778) 


s_ashr_i32 s20, s0, s19 // (op 30778) 


s_lshl_b32 s0, s15, 24 // (op 30778) 
s_ashr_i32 s19, s0, 31 // (op 30778) 

s_xor_b32 s0, s20, s19 // (op 30778) 

s_sub_i32 s20, s0, s19 // (op 30778) 


s_cmp_eq_i32 s1, 0 // (op 30778) 
s_mov_b32 s0, scc // (op 30778) 

s_cmp_eq_i32 s0, 0 // (op 30778) 
s_cselect_b32 s1, s20, s18 // (op 30778) 



v_mov_b32 v15, 192 // (op 30778) 
v_mul_lo_u32 v22, s1, v15 // (op 30778) 

v_add_u32 v22, v22, v14 // (op 30778) 


v_mul_lo_u32 v15, v22, s31 // (op 30778) most significant: low of low * high
v_mul_hi_u32 v23, v22, s30 // (op 30778) most significant: high of low * low
v_mul_lo_u32 v14, v22, s30 // (op 30778) least significant: low of low * low
v_add_u32 v15, v15, v23 // (op 30778) most significant: sum


v_add_co_u32 v22, s[0:1], v20, v14 // (op 30778) least significant half
v_addc_co_u32 v23, s[0:1], v21, v15, s[0:1] // (op 30778) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 30778) 


v_lshrrev_b64 v[14:15], 3, v[22:23] // (op 30778) 

s_mul_i32 s1, s30, s23 // (op 30780) most significant: low of low * high
s_mul_i32 s18, s31, s22 // (op 30780) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 30780) least significant
s_mul_hi_u32 s19, s30, s22 // (op 30780) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30780) most significant: sum
s_add_u32 s1, s1, s19 // (op 30780) most significant: sum


s_lshl_b64 s[24:25], s[0:1], 2 // (op 30780) 

s_ashr_i64 s[0:1], s[24:25], 3 // (op 30780) 

s_mov_b32 s64, s32 // (op 30780) reg expression
s_mov_b32 s65, s33 // (op 30780) reg expression
s_mov_b32 s66, s0 // (op 30780) reg expression
s_mov_b32 s67, 131072 // (op 30780) call()

s_mul_hi_i32 s0, s16, s47 // (op 30797) 
v_lshrrev_b32 v20, 2, v1 // (op 30797) 
s_add_i32 s1, s0, s16 // (op 30797) 

v_and_b32 v21, -4, v1 // (op 30797) 
s_ashr_i32 s0, s1, 31 // (op 30797) 
v_sub_u32 v22, v1, v21 // (op 30797) 

s_cmp_eq_i32 s47, 0 // (op 30797) 
s_mov_b32 s18, scc // (op 30797) 
v_lshlrev_b32 v21, 5, v22 // (op 30797) 

s_cmp_eq_i32 s18, 0 // (op 30797) 
s_cselect_b32 s19, 0, -1 // (op 30797) 

v_mov_b32 v23, 0 // (op 30797) convert
v_mov_b32 v22, v21 // (op 30797) convert

s_and_b32 s18, s11, 31 // (op 30797) 
s_lshl_b32 s20, 1, s18 // (op 30797) 
s_add_i32 s21, s20, s19 // (op 30797) 


s_and_b32 s19, s0, s21 // (op 30797) 


s_add_i32 s0, s1, s19 // (op 30797) 


s_ashr_i32 s1, s0, s18 // (op 30797) 


s_lshl_b32 s0, s11, 24 // (op 30797) 
s_ashr_i32 s18, s0, 31 // (op 30797) 

s_xor_b32 s0, s1, s18 // (op 30797) 

s_sub_i32 s1, s0, s18 // (op 30797) 


s_mul_hi_i32 s0, s1, s12 // (op 30797) 
s_add_i32 s18, s0, s1 // (op 30797) 

s_ashr_i32 s0, s18, 31 // (op 30797) 
s_cmp_eq_i32 s12, 0 // (op 30797) 
s_mov_b32 s19, scc // (op 30797) 
s_cmp_eq_i32 s19, 0 // (op 30797) 
s_cselect_b32 s20, 0, -1 // (op 30797) 

s_and_b32 s19, s13, 31 // (op 30797) 
s_lshl_b32 s21, 1, s19 // (op 30797) 
s_add_i32 s24, s21, s20 // (op 30797) 


s_and_b32 s25, s0, s24 // (op 30797) 


s_add_i32 s0, s18, s25 // (op 30797) 


s_ashr_i32 s18, s0, s19 // (op 30797) 


s_lshl_b32 s0, s13, 24 // (op 30797) 
s_ashr_i32 s19, s0, 31 // (op 30797) 

s_xor_b32 s0, s18, s19 // (op 30797) 

s_sub_i32 s18, s0, s19 // (op 30797) 


s_mul_i32 s0, s18, s6 // (op 30797) 

s_sub_i32 s18, s1, s0 // (op 30797) 


s_mul_hi_i32 s0, s16, s10 // (op 30797) 
s_add_i32 s1, s0, s16 // (op 30797) 

s_ashr_i32 s0, s1, 31 // (op 30797) 
s_cmp_eq_i32 s10, 0 // (op 30797) 
s_mov_b32 s19, scc // (op 30797) 
s_cmp_eq_i32 s19, 0 // (op 30797) 
s_cselect_b32 s20, 0, -1 // (op 30797) 

s_and_b32 s19, s44, 31 // (op 30797) 
s_lshl_b32 s21, 1, s19 // (op 30797) 
s_add_i32 s24, s21, s20 // (op 30797) 


s_and_b32 s25, s0, s24 // (op 30797) 


s_add_i32 s0, s1, s25 // (op 30797) 


s_ashr_i32 s1, s0, s19 // (op 30797) 


s_lshl_b32 s0, s44, 24 // (op 30797) 
s_ashr_i32 s19, s0, 31 // (op 30797) 

s_xor_b32 s0, s1, s19 // (op 30797) 

s_sub_i32 s1, s0, s19 // (op 30797) 


s_mul_i32 s0, s48, s6 // (op 30797) 
s_mul_i32 s19, s14, s0 // (op 30797) 

s_mul_i32 s0, s1, s19 // (op 30797) 

s_sub_u32 s19, s16, s0 // (op 30797) 

s_mul_hi_i32 s0, s19, s45 // (op 30797) 
s_add_i32 s20, s0, s19 // (op 30797) 


s_ashr_i32 s0, s20, 31 // (op 30797) 
s_cmp_eq_i32 s45, 0 // (op 30797) 
s_mov_b32 s19, scc // (op 30797) 
s_cmp_eq_i32 s19, 0 // (op 30797) 
s_cselect_b32 s21, 0, -1 // (op 30797) 

s_and_b32 s19, s15, 31 // (op 30797) 
s_lshl_b32 s24, 1, s19 // (op 30797) 
s_add_i32 s25, s24, s21 // (op 30797) 


s_and_b32 s21, s0, s25 // (op 30797) 


s_add_i32 s0, s20, s21 // (op 30797) 


s_ashr_i32 s20, s0, s19 // (op 30797) 


s_lshl_b32 s0, s15, 24 // (op 30797) 
s_ashr_i32 s19, s0, 31 // (op 30797) 

s_xor_b32 s0, s20, s19 // (op 30797) 

s_sub_i32 s20, s0, s19 // (op 30797) 


s_cmp_eq_i32 s1, 0 // (op 30797) 
s_mov_b32 s0, scc // (op 30797) 

s_cmp_eq_i32 s0, 0 // (op 30797) 
s_cselect_b32 s1, s20, s18 // (op 30797) 



v_mov_b32 v21, 192 // (op 30797) 
v_mul_lo_u32 v24, s1, v21 // (op 30797) 

v_add_u32 v24, v24, v20 // (op 30797) 


v_mul_lo_u32 v21, v24, s31 // (op 30797) most significant: low of low * high
v_mul_hi_u32 v25, v24, s30 // (op 30797) most significant: high of low * low
v_mul_lo_u32 v20, v24, s30 // (op 30797) least significant: low of low * low
v_add_u32 v21, v21, v25 // (op 30797) most significant: sum


v_add_co_u32 v24, s[0:1], v22, v20 // (op 30797) least significant half
v_addc_co_u32 v25, s[0:1], v23, v21, s[0:1] // (op 30797) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 30797) 


v_lshrrev_b64 v[20:21], 3, v[24:25] // (op 30797) 

s_mul_i32 s1, s30, s23 // (op 30799) most significant: low of low * high
s_mul_i32 s18, s31, s22 // (op 30799) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 30799) least significant
s_mul_hi_u32 s19, s30, s22 // (op 30799) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30799) most significant: sum
s_add_u32 s1, s1, s19 // (op 30799) most significant: sum


s_lshl_b64 s[24:25], s[0:1], 2 // (op 30799) 

s_ashr_i64 s[0:1], s[24:25], 3 // (op 30799) 

s_mov_b32 s64, s32 // (op 30799) reg expression
s_mov_b32 s65, s33 // (op 30799) reg expression
s_mov_b32 s66, s0 // (op 30799) reg expression
s_mov_b32 s67, 131072 // (op 30799) call()

s_mul_hi_i32 s0, s16, s47 // (op 31251) 
v_and_b32 v22, 63, v1 // (op 31251) 
s_add_i32 s1, s0, s16 // (op 31251) 

v_lshrrev_b32 v23, 4, v22 // (op 31251) 
s_ashr_i32 s0, s1, 31 // (op 31251) 
v_and_b32 v24, 3, v23 // (op 31251) 

s_cmp_eq_i32 s47, 0 // (op 31251) 
s_mov_b32 s18, scc // (op 31251) 
v_and_b32 v23, 15, v1 // (op 31251) 
s_cmp_eq_i32 s18, 0 // (op 31251) 
s_cselect_b32 s19, 0, -1 // (op 31251) 

v_lshl_add_u32 v25, v24, 4, v23 // (op 31251) 


s_and_b32 s18, s11, 31 // (op 31251) 
v_and_b32 v23, 4294967232, v1 // (op 31251) 
s_lshl_b32 s20, 1, s18 // (op 31251) 
v_add_u32 v24, v23, v25 // (op 31251) 


s_add_i32 s21, s20, s19 // (op 31251) 


v_lshrrev_b32 v23, 6, v22 // (op 31251) 

s_and_b32 s19, s0, s21 // (op 31251) 


v_lshlrev_b32 v22, 2, v23 // (op 31251) 

s_add_i32 s0, s1, s19 // (op 31251) 


s_ashr_i32 s1, s0, s18 // (op 31251) 


s_lshl_b32 s0, s11, 24 // (op 31251) 
s_ashr_i32 s18, s0, 31 // (op 31251) 

s_xor_b32 s0, s1, s18 // (op 31251) 

s_sub_i32 s1, s0, s18 // (op 31251) 


s_mul_hi_i32 s0, s1, s12 // (op 31251) 
s_add_i32 s18, s0, s1 // (op 31251) 

s_ashr_i32 s0, s18, 31 // (op 31251) 
s_cmp_eq_i32 s12, 0 // (op 31251) 
s_mov_b32 s19, scc // (op 31251) 
s_cmp_eq_i32 s19, 0 // (op 31251) 
s_cselect_b32 s20, 0, -1 // (op 31251) 

s_and_b32 s19, s13, 31 // (op 31251) 
s_lshl_b32 s21, 1, s19 // (op 31251) 
s_add_i32 s24, s21, s20 // (op 31251) 


s_and_b32 s25, s0, s24 // (op 31251) 


s_add_i32 s0, s18, s25 // (op 31251) 


s_ashr_i32 s18, s0, s19 // (op 31251) 


s_lshl_b32 s0, s13, 24 // (op 31251) 
s_ashr_i32 s19, s0, 31 // (op 31251) 

s_xor_b32 s0, s18, s19 // (op 31251) 

s_sub_i32 s18, s0, s19 // (op 31251) 


s_mul_i32 s0, s1, s48 // (op 31251) 

s_sub_u32 s1, s16, s0 // (op 31251) 

s_mul_i32 s0, s18, s48 // (op 31251) 
s_add_u32 s0, s0, s1 // (op 31251) 


s_mul_hi_i32 s1, s16, s10 // (op 31251) 
s_add_i32 s18, s1, s16 // (op 31251) 

s_ashr_i32 s1, s18, 31 // (op 31251) 
s_cmp_eq_i32 s10, 0 // (op 31251) 
s_mov_b32 s19, scc // (op 31251) 
s_cmp_eq_i32 s19, 0 // (op 31251) 
s_cselect_b32 s20, 0, -1 // (op 31251) 

s_and_b32 s19, s44, 31 // (op 31251) 
s_lshl_b32 s21, 1, s19 // (op 31251) 
s_add_i32 s24, s21, s20 // (op 31251) 


s_and_b32 s25, s1, s24 // (op 31251) 


s_add_i32 s1, s18, s25 // (op 31251) 


s_ashr_i32 s18, s1, s19 // (op 31251) 


s_lshl_b32 s1, s44, 24 // (op 31251) 
s_ashr_i32 s19, s1, 31 // (op 31251) 

s_xor_b32 s1, s18, s19 // (op 31251) 

s_sub_i32 s18, s1, s19 // (op 31251) 


s_mul_i32 s1, s48, s6 // (op 31251) 
s_mul_i32 s19, s14, s1 // (op 31251) 

s_mul_i32 s1, s18, s19 // (op 31251) 

s_sub_u32 s19, s16, s1 // (op 31251) 

s_mov_b32 s1, s19 // (op 31251) 
s_mul_hi_i32 s20, s1, s45 // (op 31251) 
s_add_i32 s21, s20, s1 // (op 31251) 


s_ashr_i32 s1, s21, 31 // (op 31251) 
s_cmp_eq_i32 s45, 0 // (op 31251) 
s_mov_b32 s20, scc // (op 31251) 
s_cmp_eq_i32 s20, 0 // (op 31251) 
s_cselect_b32 s24, 0, -1 // (op 31251) 

s_and_b32 s20, s15, 31 // (op 31251) 
s_lshl_b32 s25, 1, s20 // (op 31251) 
s_add_i32 s49, s25, s24 // (op 31251) 


s_and_b32 s24, s1, s49 // (op 31251) 


s_add_i32 s1, s21, s24 // (op 31251) 


s_ashr_i32 s21, s1, s20 // (op 31251) 


s_lshl_b32 s1, s15, 24 // (op 31251) 
s_ashr_i32 s20, s1, 31 // (op 31251) 

s_xor_b32 s1, s21, s20 // (op 31251) 

s_sub_i32 s21, s1, s20 // (op 31251) 


s_mul_i32 s1, s46, s48 // (op 31251) 
s_add_u32 s24, s8, 256 // (op 31251) Start of Int64 add, locking SCC
s_addc_u32 s25, s9, 0 // (op 31251) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 31251) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 31251) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 31251) 

s_sub_i32 s20, s24, s1 // (op 31251) 

s_mul_i32 s24, s21, s20 // (op 31251) 


s_sub_u32 s25, s19, s24 // (op 31251) 


s_add_u32 s24, s25, s1 // (op 31251) 


s_cmp_eq_i32 s18, 0 // (op 31251) 
s_mov_b32 s1, scc // (op 31251) 

s_cmp_eq_i32 s1, 0 // (op 31251) 
s_cselect_b32 s25, s24, s0 // (op 31251) 



v_lshl_add_u32 v23, s25, 8, v24 // (op 31251) 


v_mul_lo_u32 v24, v23, s34 // (op 31251) 
v_add_u32 v24, v24, v22 // (op 31251) 


s_mul_i32 s1, s34, s9 // (op 31253) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 31253) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 31253) least significant
s_mul_hi_u32 s19, s34, s8 // (op 31253) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 31253) most significant: sum
s_add_u32 s1, s1, s19 // (op 31253) most significant: sum


s_mov_b32 s68, s36 // (op 31253) reg expression
s_mov_b32 s69, s37 // (op 31253) reg expression
s_mov_b32 s70, s0 // (op 31253) reg expression
s_mov_b32 s71, 131072 // (op 31253) call()

s_mul_hi_i32 s0, s16, s47 // (op 31272) 
v_and_b32 v22, 63, v1 // (op 31272) 
s_add_i32 s1, s0, s16 // (op 31272) 

v_lshrrev_b32 v23, 4, v22 // (op 31272) 
s_ashr_i32 s0, s1, 31 // (op 31272) 
v_and_b32 v25, 3, v23 // (op 31272) 

s_cmp_eq_i32 s47, 0 // (op 31272) 
s_mov_b32 s18, scc // (op 31272) 
v_and_b32 v23, 15, v1 // (op 31272) 
s_cmp_eq_i32 s18, 0 // (op 31272) 
s_cselect_b32 s19, 0, -1 // (op 31272) 

v_lshl_add_u32 v26, v25, 4, v23 // (op 31272) 


s_and_b32 s18, s11, 31 // (op 31272) 
v_and_b32 v23, 4294967232, v1 // (op 31272) 
s_lshl_b32 s20, 1, s18 // (op 31272) 
v_add_u32 v25, v23, v26 // (op 31272) 


s_add_i32 s21, s20, s19 // (op 31272) 


v_lshrrev_b32 v23, 6, v22 // (op 31272) 

s_and_b32 s19, s0, s21 // (op 31272) 


v_lshlrev_b32 v22, 2, v23 // (op 31272) 

s_add_i32 s0, s1, s19 // (op 31272) 


s_ashr_i32 s1, s0, s18 // (op 31272) 


s_lshl_b32 s0, s11, 24 // (op 31272) 
s_ashr_i32 s18, s0, 31 // (op 31272) 

s_xor_b32 s0, s1, s18 // (op 31272) 

s_sub_i32 s1, s0, s18 // (op 31272) 


s_mul_hi_i32 s0, s1, s12 // (op 31272) 
s_add_i32 s18, s0, s1 // (op 31272) 

s_ashr_i32 s0, s18, 31 // (op 31272) 
s_cmp_eq_i32 s12, 0 // (op 31272) 
s_mov_b32 s19, scc // (op 31272) 
s_cmp_eq_i32 s19, 0 // (op 31272) 
s_cselect_b32 s20, 0, -1 // (op 31272) 

s_and_b32 s19, s13, 31 // (op 31272) 
s_lshl_b32 s21, 1, s19 // (op 31272) 
s_add_i32 s24, s21, s20 // (op 31272) 


s_and_b32 s25, s0, s24 // (op 31272) 


s_add_i32 s0, s18, s25 // (op 31272) 


s_ashr_i32 s18, s0, s19 // (op 31272) 


s_lshl_b32 s0, s13, 24 // (op 31272) 
s_ashr_i32 s19, s0, 31 // (op 31272) 

s_xor_b32 s0, s18, s19 // (op 31272) 

s_sub_i32 s18, s0, s19 // (op 31272) 


s_mul_i32 s0, s1, s48 // (op 31272) 

s_sub_u32 s1, s16, s0 // (op 31272) 

s_mul_i32 s0, s18, s48 // (op 31272) 
s_add_u32 s0, s0, s1 // (op 31272) 


s_mul_hi_i32 s1, s16, s10 // (op 31272) 
s_add_i32 s18, s1, s16 // (op 31272) 

s_ashr_i32 s1, s18, 31 // (op 31272) 
s_cmp_eq_i32 s10, 0 // (op 31272) 
s_mov_b32 s19, scc // (op 31272) 
s_cmp_eq_i32 s19, 0 // (op 31272) 
s_cselect_b32 s20, 0, -1 // (op 31272) 

s_and_b32 s19, s44, 31 // (op 31272) 
s_lshl_b32 s21, 1, s19 // (op 31272) 
s_add_i32 s24, s21, s20 // (op 31272) 


s_and_b32 s25, s1, s24 // (op 31272) 


s_add_i32 s1, s18, s25 // (op 31272) 


s_ashr_i32 s18, s1, s19 // (op 31272) 


s_lshl_b32 s1, s44, 24 // (op 31272) 
s_ashr_i32 s19, s1, 31 // (op 31272) 

s_xor_b32 s1, s18, s19 // (op 31272) 

s_sub_i32 s18, s1, s19 // (op 31272) 


s_mul_i32 s1, s48, s6 // (op 31272) 
s_mul_i32 s19, s14, s1 // (op 31272) 

s_mul_i32 s1, s18, s19 // (op 31272) 

s_sub_u32 s19, s16, s1 // (op 31272) 

s_mov_b32 s1, s19 // (op 31272) 
s_mul_hi_i32 s20, s1, s45 // (op 31272) 
s_add_i32 s21, s20, s1 // (op 31272) 


s_ashr_i32 s1, s21, 31 // (op 31272) 
s_cmp_eq_i32 s45, 0 // (op 31272) 
s_mov_b32 s20, scc // (op 31272) 
s_cmp_eq_i32 s20, 0 // (op 31272) 
s_cselect_b32 s24, 0, -1 // (op 31272) 

s_and_b32 s20, s15, 31 // (op 31272) 
s_lshl_b32 s25, 1, s20 // (op 31272) 
s_add_i32 s49, s25, s24 // (op 31272) 


s_and_b32 s24, s1, s49 // (op 31272) 


s_add_i32 s1, s21, s24 // (op 31272) 


s_ashr_i32 s21, s1, s20 // (op 31272) 


s_lshl_b32 s1, s15, 24 // (op 31272) 
s_ashr_i32 s20, s1, 31 // (op 31272) 

s_xor_b32 s1, s21, s20 // (op 31272) 

s_sub_i32 s21, s1, s20 // (op 31272) 


s_mul_i32 s1, s46, s48 // (op 31272) 
s_add_u32 s24, s8, 256 // (op 31272) Start of Int64 add, locking SCC
s_addc_u32 s25, s9, 0 // (op 31272) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 31272) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 31272) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 31272) 

s_sub_i32 s20, s24, s1 // (op 31272) 

s_mul_i32 s24, s21, s20 // (op 31272) 


s_sub_u32 s25, s19, s24 // (op 31272) 


s_add_u32 s24, s25, s1 // (op 31272) 


s_cmp_eq_i32 s18, 0 // (op 31272) 
s_mov_b32 s1, scc // (op 31272) 

s_cmp_eq_i32 s1, 0 // (op 31272) 
s_cselect_b32 s25, s24, s0 // (op 31272) 



v_lshl_add_u32 v23, s25, 8, v25 // (op 31272) 


v_mul_lo_u32 v25, v23, s34 // (op 31272) 
v_add_u32 v25, v25, v22 // (op 31272) 


s_mul_i32 s1, s34, s9 // (op 31274) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 31274) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 31274) least significant
s_mul_hi_u32 s19, s34, s8 // (op 31274) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 31274) most significant: sum
s_add_u32 s1, s1, s19 // (op 31274) most significant: sum


s_mov_b32 s68, s36 // (op 31274) reg expression
s_mov_b32 s69, s37 // (op 31274) reg expression
s_mov_b32 s70, s0 // (op 31274) reg expression
s_mov_b32 s71, 131072 // (op 31274) call()

s_mul_hi_i32 s0, s16, s47 // (op 31303) 
v_and_b32 v22, 63, v1 // (op 31303) 
s_add_i32 s1, s0, s16 // (op 31303) 

v_lshrrev_b32 v23, 4, v22 // (op 31303) 
s_ashr_i32 s0, s1, 31 // (op 31303) 
v_and_b32 v26, 3, v23 // (op 31303) 

s_cmp_eq_i32 s47, 0 // (op 31303) 
s_mov_b32 s18, scc // (op 31303) 
v_and_b32 v23, 15, v1 // (op 31303) 
s_cmp_eq_i32 s18, 0 // (op 31303) 
s_cselect_b32 s19, 0, -1 // (op 31303) 

v_lshl_add_u32 v27, v26, 4, v23 // (op 31303) 


s_and_b32 s18, s11, 31 // (op 31303) 
v_lshrrev_b32 v23, 6, v22 // (op 31303) 

s_lshl_b32 s20, 1, s18 // (op 31303) 
s_add_i32 s21, s20, s19 // (op 31303) 


s_and_b32 s19, s0, s21 // (op 31303) 


s_add_i32 s0, s1, s19 // (op 31303) 


s_ashr_i32 s1, s0, s18 // (op 31303) 


s_lshl_b32 s0, s11, 24 // (op 31303) 
s_ashr_i32 s18, s0, 31 // (op 31303) 

s_xor_b32 s0, s1, s18 // (op 31303) 

s_sub_i32 s1, s0, s18 // (op 31303) 


s_mul_hi_i32 s0, s1, s12 // (op 31303) 
s_add_i32 s18, s0, s1 // (op 31303) 

s_ashr_i32 s0, s18, 31 // (op 31303) 
s_cmp_eq_i32 s12, 0 // (op 31303) 
s_mov_b32 s19, scc // (op 31303) 
s_cmp_eq_i32 s19, 0 // (op 31303) 
s_cselect_b32 s20, 0, -1 // (op 31303) 

s_and_b32 s19, s13, 31 // (op 31303) 
s_lshl_b32 s21, 1, s19 // (op 31303) 
s_add_i32 s24, s21, s20 // (op 31303) 


s_and_b32 s25, s0, s24 // (op 31303) 


s_add_i32 s0, s18, s25 // (op 31303) 


s_ashr_i32 s18, s0, s19 // (op 31303) 


s_lshl_b32 s0, s13, 24 // (op 31303) 
s_ashr_i32 s19, s0, 31 // (op 31303) 

s_xor_b32 s0, s18, s19 // (op 31303) 

s_sub_i32 s18, s0, s19 // (op 31303) 


s_mul_i32 s0, s18, s6 // (op 31303) 

s_sub_i32 s18, s1, s0 // (op 31303) 


s_mul_hi_i32 s0, s16, s10 // (op 31303) 
s_add_i32 s1, s0, s16 // (op 31303) 

s_ashr_i32 s0, s1, 31 // (op 31303) 
s_cmp_eq_i32 s10, 0 // (op 31303) 
s_mov_b32 s19, scc // (op 31303) 
s_cmp_eq_i32 s19, 0 // (op 31303) 
s_cselect_b32 s20, 0, -1 // (op 31303) 

s_and_b32 s19, s44, 31 // (op 31303) 
s_lshl_b32 s21, 1, s19 // (op 31303) 
s_add_i32 s24, s21, s20 // (op 31303) 


s_and_b32 s25, s0, s24 // (op 31303) 


s_add_i32 s0, s1, s25 // (op 31303) 


s_ashr_i32 s1, s0, s19 // (op 31303) 


s_lshl_b32 s0, s44, 24 // (op 31303) 
s_ashr_i32 s19, s0, 31 // (op 31303) 

s_xor_b32 s0, s1, s19 // (op 31303) 

s_sub_i32 s1, s0, s19 // (op 31303) 


s_mul_i32 s0, s48, s6 // (op 31303) 
s_mul_i32 s19, s14, s0 // (op 31303) 

s_mul_i32 s0, s1, s19 // (op 31303) 

s_sub_u32 s19, s16, s0 // (op 31303) 

s_mul_hi_i32 s0, s19, s45 // (op 31303) 
s_add_i32 s20, s0, s19 // (op 31303) 


s_ashr_i32 s0, s20, 31 // (op 31303) 
s_cmp_eq_i32 s45, 0 // (op 31303) 
s_mov_b32 s19, scc // (op 31303) 
s_cmp_eq_i32 s19, 0 // (op 31303) 
s_cselect_b32 s21, 0, -1 // (op 31303) 

s_and_b32 s19, s15, 31 // (op 31303) 
s_lshl_b32 s24, 1, s19 // (op 31303) 
s_add_i32 s25, s24, s21 // (op 31303) 


s_and_b32 s21, s0, s25 // (op 31303) 


s_add_i32 s0, s20, s21 // (op 31303) 


s_ashr_i32 s20, s0, s19 // (op 31303) 


s_lshl_b32 s0, s15, 24 // (op 31303) 
s_ashr_i32 s19, s0, 31 // (op 31303) 

s_xor_b32 s0, s20, s19 // (op 31303) 

s_sub_i32 s20, s0, s19 // (op 31303) 


s_cmp_eq_i32 s1, 0 // (op 31303) 
s_mov_b32 s0, scc // (op 31303) 

s_cmp_eq_i32 s0, 0 // (op 31303) 
s_cselect_b32 s1, s20, s18 // (op 31303) 



v_mov_b32 v22, 192 // (op 31303) 
v_mul_lo_u32 v26, s1, v22 // (op 31303) 

v_add_u32 v26, v26, v27 // (op 31303) 


v_mul_lo_u32 v22, v26, s38 // (op 31303) 

v_lshl_add_u32 v26, v23, 2, v22 // (op 31303) 


s_mul_i32 s1, s38, s23 // (op 31305) most significant: low of low * high
s_mul_i32 s18, s39, s22 // (op 31305) most significant: low of high * low
s_mul_i32 s0, s38, s22 // (op 31305) least significant
s_mul_hi_u32 s19, s38, s22 // (op 31305) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 31305) most significant: sum
s_add_u32 s1, s1, s19 // (op 31305) most significant: sum


s_mov_b32 s72, s40 // (op 31305) reg expression
s_mov_b32 s73, s41 // (op 31305) reg expression
s_mov_b32 s74, s0 // (op 31305) reg expression
s_mov_b32 s75, 131072 // (op 31305) call()

s_mul_hi_i32 s0, s16, s47 // (op 31324) 
v_and_b32 v22, 63, v1 // (op 31324) 
s_add_i32 s1, s0, s16 // (op 31324) 

v_lshrrev_b32 v23, 4, v22 // (op 31324) 
s_ashr_i32 s0, s1, 31 // (op 31324) 
v_and_b32 v27, 3, v23 // (op 31324) 

s_cmp_eq_i32 s47, 0 // (op 31324) 
s_mov_b32 s18, scc // (op 31324) 
v_and_b32 v23, 15, v1 // (op 31324) 
s_cmp_eq_i32 s18, 0 // (op 31324) 
s_cselect_b32 s19, 0, -1 // (op 31324) 

v_lshl_add_u32 v28, v27, 4, v23 // (op 31324) 


s_and_b32 s18, s11, 31 // (op 31324) 
v_lshrrev_b32 v23, 6, v22 // (op 31324) 

s_lshl_b32 s20, 1, s18 // (op 31324) 
s_add_i32 s21, s20, s19 // (op 31324) 


s_and_b32 s19, s0, s21 // (op 31324) 


s_add_i32 s0, s1, s19 // (op 31324) 


s_ashr_i32 s1, s0, s18 // (op 31324) 


s_lshl_b32 s0, s11, 24 // (op 31324) 
s_ashr_i32 s18, s0, 31 // (op 31324) 

s_xor_b32 s0, s1, s18 // (op 31324) 

s_sub_i32 s1, s0, s18 // (op 31324) 


s_mul_hi_i32 s0, s1, s12 // (op 31324) 
s_add_i32 s18, s0, s1 // (op 31324) 

s_ashr_i32 s0, s18, 31 // (op 31324) 
s_cmp_eq_i32 s12, 0 // (op 31324) 
s_mov_b32 s19, scc // (op 31324) 
s_cmp_eq_i32 s19, 0 // (op 31324) 
s_cselect_b32 s20, 0, -1 // (op 31324) 

s_and_b32 s19, s13, 31 // (op 31324) 
s_lshl_b32 s21, 1, s19 // (op 31324) 
s_add_i32 s24, s21, s20 // (op 31324) 


s_and_b32 s25, s0, s24 // (op 31324) 


s_add_i32 s0, s18, s25 // (op 31324) 


s_ashr_i32 s18, s0, s19 // (op 31324) 


s_lshl_b32 s0, s13, 24 // (op 31324) 
s_ashr_i32 s19, s0, 31 // (op 31324) 

s_xor_b32 s0, s18, s19 // (op 31324) 

s_sub_i32 s18, s0, s19 // (op 31324) 


s_mul_i32 s0, s18, s6 // (op 31324) 

s_sub_i32 s18, s1, s0 // (op 31324) 


s_mul_hi_i32 s0, s16, s10 // (op 31324) 
s_add_i32 s1, s0, s16 // (op 31324) 

s_ashr_i32 s0, s1, 31 // (op 31324) 
s_cmp_eq_i32 s10, 0 // (op 31324) 
s_mov_b32 s19, scc // (op 31324) 
s_cmp_eq_i32 s19, 0 // (op 31324) 
s_cselect_b32 s20, 0, -1 // (op 31324) 

s_and_b32 s19, s44, 31 // (op 31324) 
s_lshl_b32 s21, 1, s19 // (op 31324) 
s_add_i32 s24, s21, s20 // (op 31324) 


s_and_b32 s25, s0, s24 // (op 31324) 


s_add_i32 s0, s1, s25 // (op 31324) 


s_ashr_i32 s1, s0, s19 // (op 31324) 


s_lshl_b32 s0, s44, 24 // (op 31324) 
s_ashr_i32 s19, s0, 31 // (op 31324) 

s_xor_b32 s0, s1, s19 // (op 31324) 

s_sub_i32 s1, s0, s19 // (op 31324) 


s_mul_i32 s0, s48, s6 // (op 31324) 
s_mul_i32 s19, s14, s0 // (op 31324) 

s_mul_i32 s0, s1, s19 // (op 31324) 

s_sub_u32 s19, s16, s0 // (op 31324) 

s_mul_hi_i32 s0, s19, s45 // (op 31324) 
s_add_i32 s20, s0, s19 // (op 31324) 


s_ashr_i32 s0, s20, 31 // (op 31324) 
s_cmp_eq_i32 s45, 0 // (op 31324) 
s_mov_b32 s19, scc // (op 31324) 
s_cmp_eq_i32 s19, 0 // (op 31324) 
s_cselect_b32 s21, 0, -1 // (op 31324) 

s_and_b32 s19, s15, 31 // (op 31324) 
s_lshl_b32 s24, 1, s19 // (op 31324) 
s_add_i32 s25, s24, s21 // (op 31324) 


s_and_b32 s21, s0, s25 // (op 31324) 


s_add_i32 s0, s20, s21 // (op 31324) 


s_ashr_i32 s20, s0, s19 // (op 31324) 


s_lshl_b32 s0, s15, 24 // (op 31324) 
s_ashr_i32 s19, s0, 31 // (op 31324) 

s_xor_b32 s0, s20, s19 // (op 31324) 

s_sub_i32 s20, s0, s19 // (op 31324) 


s_cmp_eq_i32 s1, 0 // (op 31324) 
s_mov_b32 s0, scc // (op 31324) 

s_cmp_eq_i32 s0, 0 // (op 31324) 
s_cselect_b32 s1, s20, s18 // (op 31324) 



v_mov_b32 v22, 192 // (op 31324) 
v_mul_lo_u32 v27, s1, v22 // (op 31324) 

v_add_u32 v27, v27, v28 // (op 31324) 


v_mul_lo_u32 v22, v27, s38 // (op 31324) 

v_lshl_add_u32 v27, v23, 2, v22 // (op 31324) 


s_mul_i32 s1, s38, s23 // (op 31326) most significant: low of low * high
s_mul_i32 s18, s39, s22 // (op 31326) most significant: low of high * low
s_mul_i32 s0, s38, s22 // (op 31326) least significant
s_mul_hi_u32 s19, s38, s22 // (op 31326) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 31326) most significant: sum
s_add_u32 s1, s1, s19 // (op 31326) most significant: sum


s_mov_b32 s72, s40 // (op 31326) reg expression
s_mov_b32 s73, s41 // (op 31326) reg expression
s_mov_b32 s74, s0 // (op 31326) reg expression
s_mov_b32 s75, 131072 // (op 31326) call()

v_and_b32 v22, -4, v1 // (op 30925) 
v_sub_u32 v23, v1, v22 // (op 30925) 

v_lshlrev_b32 v22, 5, v23 // (op 30925) 

v_lshrrev_b32 v23, 2, v1 // (op 30925) 
v_lshl_add_u32 v28, v23, 7, v22 // (op 30925) 


v_lshlrev_b32 v22, 2, v28 // (op 30925) 

v_lshrrev_b32 v23, 3, v22 // (op 30925) 

s_nop 0
v_readfirstlane_b32 s0, v23 // (op 30925) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[24:25], s[30:31], 6 // (op 30608) 
s_lshl_b64 s[50:51], s[24:25], 2 // (op 30608) 

v_lshrrev_b64 v[22:23], 3, s[50:51] // (op 30608) 


v_mov_b64 v[28:29], v[14:15] // (op 30608) 
s_mov_b32 m0, s0 // (op 30608) reg expression
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 30608) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v22 // (op 30608) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 30608) most significant half

s_add_u32 m0, m0, 4096 // (op 30608) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 30608) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v22 // (op 30608) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 30608) most significant half

s_add_u32 m0, m0, 4096 // (op 30608) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 30608) Load value direct to lds



v_and_b32 v22, -4, v1 // (op 30868) 
v_sub_u32 v23, v1, v22 // (op 30868) 

v_lshlrev_b32 v22, 5, v23 // (op 30868) 

v_lshrrev_b32 v23, 2, v1 // (op 30868) 
v_lshl_add_u32 v28, v23, 7, v22 // (op 30868) 


v_lshlrev_b32 v22, 2, v28 // (op 30868) 

v_lshrrev_b32 v23, 3, v22 // (op 30868) 

s_nop 0
v_readfirstlane_b32 s0, v23 // (op 30868) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[24:25], s[26:27], 6 // (op 30613) 
s_lshl_b64 s[50:51], s[24:25], 2 // (op 30613) 

v_lshrrev_b64 v[22:23], 3, s[50:51] // (op 30613) 


v_mov_b64 v[28:29], v[16:17] // (op 30613) 
s_add_u32 s1, 12288, s0 // (op 30613) 
s_mov_b32 m0, s1 // (op 30613) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30613) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v22 // (op 30613) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 30613) most significant half

s_add_u32 m0, m0, 4096 // (op 30613) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30613) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v22 // (op 30613) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 30613) most significant half

s_add_u32 m0, m0, 4096 // (op 30613) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30613) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v22 // (op 30613) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 30613) most significant half

s_add_u32 m0, m0, 4096 // (op 30613) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30613) Load value direct to lds






v_mov_b32 v22, v26 // (op 76) reg expression

buffer_load_dwordx4 v[28:31], v22, s[72:75], 0 offen // (op 76) Load value



s_lshl_b32 s0, s38, 6 // (op 1104) 
v_add_u32 v22, v26, s0 // (op 1104) 


buffer_load_dwordx4 v[32:35], v22, s[72:75], 0 offen // (op 1104) Load value



s_lshl_b32 s0, s38, 7 // (op 2008) 
v_add_u32 v22, v26, s0 // (op 2008) 


buffer_load_dwordx4 v[36:39], v22, s[72:75], 0 offen // (op 2008) Load value




v_mov_b32 v26, v24 // (op 72) reg expression

buffer_load_dwordx4 v[40:43], v26, s[68:71], 0 offen // (op 72) Load value


v_and_b32 v26, -4, v1 // (op 31036) 
v_sub_u32 v22, v1, v26 // (op 31036) 

v_lshlrev_b32 v26, 5, v22 // (op 31036) 

v_lshrrev_b32 v22, 2, v1 // (op 31036) 
v_lshl_add_u32 v23, v22, 7, v26 // (op 31036) 


v_lshlrev_b32 v22, 2, v23 // (op 31036) 

v_lshrrev_b32 v26, 3, v22 // (op 31036) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31036) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[24:25], s[30:31], 6 // (op 30618) 
s_lshl_b64 s[50:51], s[24:25], 2 // (op 30618) 

v_lshrrev_b64 v[22:23], 3, s[50:51] // (op 30618) 


v_mov_b64 v[44:45], v[14:15] // (op 30618) 
s_add_u32 s1, 28672, s0 // (op 30618) 
s_mov_b32 m0, s1 // (op 30618) reg expression
buffer_load_dwordx4 v44, s[64:67], 64 offen offset: 0 lds // (op 30618) Load value direct to lds
v_add_co_u32 v44, s[24:25], v44, v22 // (op 30618) least significant half
v_addc_co_u32 v45, s[24:25], 0, v45, s[24:25] // (op 30618) most significant half

s_add_u32 m0, m0, 4096 // (op 30618) 
buffer_load_dwordx4 v44, s[64:67], 64 offen offset: 0 lds // (op 30618) Load value direct to lds
v_add_co_u32 v44, s[24:25], v44, v22 // (op 30618) least significant half
v_addc_co_u32 v45, s[24:25], 0, v45, s[24:25] // (op 30618) most significant half

s_add_u32 m0, m0, 4096 // (op 30618) 
buffer_load_dwordx4 v44, s[64:67], 64 offen offset: 0 lds // (op 30618) Load value direct to lds




v_and_b32 v26, -4, v1 // (op 30992) 
v_sub_u32 v22, v1, v26 // (op 30992) 

v_lshlrev_b32 v26, 5, v22 // (op 30992) 

v_lshrrev_b32 v22, 2, v1 // (op 30992) 
v_lshl_add_u32 v23, v22, 7, v26 // (op 30992) 


v_lshlrev_b32 v22, 2, v23 // (op 30992) 

v_lshrrev_b32 v26, 3, v22 // (op 30992) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 30992) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[24:25], s[26:27], 6 // (op 30620) 
s_lshl_b64 s[50:51], s[24:25], 2 // (op 30620) 

v_lshrrev_b64 v[22:23], 3, s[50:51] // (op 30620) 


v_mov_b64 v[44:45], v[16:17] // (op 30620) 
s_add_u32 s1, 40960, s0 // (op 30620) 
s_mov_b32 m0, s1 // (op 30620) reg expression
buffer_load_dwordx4 v44, s[52:55], 64 offen offset: 0 lds // (op 30620) Load value direct to lds
v_add_co_u32 v44, s[24:25], v44, v22 // (op 30620) least significant half
v_addc_co_u32 v45, s[24:25], 0, v45, s[24:25] // (op 30620) most significant half

s_add_u32 m0, m0, 4096 // (op 30620) 
buffer_load_dwordx4 v44, s[52:55], 64 offen offset: 0 lds // (op 30620) Load value direct to lds
v_add_co_u32 v44, s[24:25], v44, v22 // (op 30620) least significant half
v_addc_co_u32 v45, s[24:25], 0, v45, s[24:25] // (op 30620) most significant half

s_add_u32 m0, m0, 4096 // (op 30620) 
buffer_load_dwordx4 v44, s[52:55], 64 offen offset: 0 lds // (op 30620) Load value direct to lds
v_add_co_u32 v44, s[24:25], v44, v22 // (op 30620) least significant half
v_addc_co_u32 v45, s[24:25], 0, v45, s[24:25] // (op 30620) most significant half

s_add_u32 m0, m0, 4096 // (op 30620) 
buffer_load_dwordx4 v44, s[52:55], 64 offen offset: 0 lds // (op 30620) Load value direct to lds




v_and_b32 v26, -4, v1 // (op 31124) 
v_sub_u32 v22, v1, v26 // (op 31124) 

v_lshlrev_b32 v26, 5, v22 // (op 31124) 

v_lshrrev_b32 v22, 2, v1 // (op 31124) 
v_lshl_add_u32 v23, v22, 7, v26 // (op 31124) 


v_lshlrev_b32 v22, 2, v23 // (op 31124) 

v_lshrrev_b32 v26, 3, v22 // (op 31124) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31124) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[24:25], s[30:31], 6 // (op 30622) 
s_lshl_b64 s[50:51], s[24:25], 2 // (op 30622) 

v_lshrrev_b64 v[22:23], 3, s[50:51] // (op 30622) 


v_mov_b64 v[44:45], v[14:15] // (op 30622) 
s_add_u32 s1, 57344, s0 // (op 30622) 
s_mov_b32 m0, s1 // (op 30622) reg expression
s_mov_b32 s18, 128 // (op 30622) call()
buffer_load_dwordx4 v44, s[64:67], s18 offen offset: 0 lds // (op 30622) Load value direct to lds
v_add_co_u32 v44, s[20:21], v44, v22 // (op 30622) least significant half
v_addc_co_u32 v45, s[20:21], 0, v45, s[20:21] // (op 30622) most significant half

s_add_u32 m0, m0, 4096 // (op 30622) 
buffer_load_dwordx4 v44, s[64:67], s18 offen offset: 0 lds // (op 30622) Load value direct to lds
v_add_co_u32 v44, s[20:21], v44, v22 // (op 30622) least significant half
v_addc_co_u32 v45, s[20:21], 0, v45, s[20:21] // (op 30622) most significant half

s_add_u32 m0, m0, 4096 // (op 30622) 
buffer_load_dwordx4 v44, s[64:67], s18 offen offset: 0 lds // (op 30622) Load value direct to lds





v_and_b32 v26, -4, v1 // (op 31080) 
v_sub_u32 v22, v1, v26 // (op 31080) 

v_lshlrev_b32 v26, 5, v22 // (op 31080) 

v_lshrrev_b32 v22, 2, v1 // (op 31080) 
v_lshl_add_u32 v23, v22, 7, v26 // (op 31080) 


v_lshlrev_b32 v22, 2, v23 // (op 31080) 

v_lshrrev_b32 v26, 3, v22 // (op 31080) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31080) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[24:25], s[26:27], 6 // (op 30624) 
s_lshl_b64 s[50:51], s[24:25], 2 // (op 30624) 

v_lshrrev_b64 v[22:23], 3, s[50:51] // (op 30624) 


v_mov_b64 v[44:45], v[16:17] // (op 30624) 
s_add_u32 s1, 69632, s0 // (op 30624) 
s_mov_b32 m0, s1 // (op 30624) reg expression
s_mov_b32 s18, 128 // (op 30624) call()
buffer_load_dwordx4 v44, s[52:55], s18 offen offset: 0 lds // (op 30624) Load value direct to lds
v_add_co_u32 v44, s[20:21], v44, v22 // (op 30624) least significant half
v_addc_co_u32 v45, s[20:21], 0, v45, s[20:21] // (op 30624) most significant half

s_add_u32 m0, m0, 4096 // (op 30624) 
buffer_load_dwordx4 v44, s[52:55], s18 offen offset: 0 lds // (op 30624) Load value direct to lds
v_add_co_u32 v44, s[20:21], v44, v22 // (op 30624) least significant half
v_addc_co_u32 v45, s[20:21], 0, v45, s[20:21] // (op 30624) most significant half

s_add_u32 m0, m0, 4096 // (op 30624) 
buffer_load_dwordx4 v44, s[52:55], s18 offen offset: 0 lds // (op 30624) Load value direct to lds
v_add_co_u32 v44, s[20:21], v44, v22 // (op 30624) least significant half
v_addc_co_u32 v45, s[20:21], 0, v45, s[20:21] // (op 30624) most significant half

s_add_u32 m0, m0, 4096 // (op 30624) 
buffer_load_dwordx4 v44, s[52:55], s18 offen offset: 0 lds // (op 30624) Load value direct to lds





v_and_b32 v26, -4, v1 // (op 31212) 
v_sub_u32 v22, v1, v26 // (op 31212) 

v_lshlrev_b32 v26, 5, v22 // (op 31212) 

v_lshrrev_b32 v22, 2, v1 // (op 31212) 
v_lshl_add_u32 v23, v22, 7, v26 // (op 31212) 


v_lshlrev_b32 v22, 2, v23 // (op 31212) 

v_lshrrev_b32 v26, 3, v22 // (op 31212) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31212) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[24:25], s[30:31], 6 // (op 30626) 
s_lshl_b64 s[50:51], s[24:25], 2 // (op 30626) 

v_lshrrev_b64 v[22:23], 3, s[50:51] // (op 30626) 


v_mov_b64 v[44:45], v[14:15] // (op 30626) 
s_add_u32 s1, 86016, s0 // (op 30626) 
s_mov_b32 m0, s1 // (op 30626) reg expression
s_mov_b32 s18, 192 // (op 30626) call()
buffer_load_dwordx4 v44, s[64:67], s18 offen offset: 0 lds // (op 30626) Load value direct to lds
v_add_co_u32 v44, s[20:21], v44, v22 // (op 30626) least significant half
v_addc_co_u32 v45, s[20:21], 0, v45, s[20:21] // (op 30626) most significant half

s_add_u32 m0, m0, 4096 // (op 30626) 
buffer_load_dwordx4 v44, s[64:67], s18 offen offset: 0 lds // (op 30626) Load value direct to lds
v_add_co_u32 v44, s[20:21], v44, v22 // (op 30626) least significant half
v_addc_co_u32 v45, s[20:21], 0, v45, s[20:21] // (op 30626) most significant half

s_add_u32 m0, m0, 4096 // (op 30626) 
buffer_load_dwordx4 v44, s[64:67], s18 offen offset: 0 lds // (op 30626) Load value direct to lds






v_and_b32 v26, -4, v1 // (op 31168) 
v_sub_u32 v14, v1, v26 // (op 31168) 

v_lshlrev_b32 v15, 5, v14 // (op 31168) 

v_lshrrev_b32 v14, 2, v1 // (op 31168) 
v_lshl_add_u32 v26, v14, 7, v15 // (op 31168) 


v_lshlrev_b32 v14, 2, v26 // (op 31168) 

v_lshrrev_b32 v15, 3, v14 // (op 31168) 

s_nop 0
v_readfirstlane_b32 s0, v15 // (op 31168) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[24:25], s[26:27], 6 // (op 30628) 
s_lshl_b64 s[50:51], s[24:25], 2 // (op 30628) 

v_lshrrev_b64 v[14:15], 3, s[50:51] // (op 30628) 


v_mov_b64 v[22:23], v[16:17] // (op 30628) 
s_add_u32 s1, 98304, s0 // (op 30628) 
s_mov_b32 m0, s1 // (op 30628) reg expression
s_mov_b32 s18, 192 // (op 30628) call()
buffer_load_dwordx4 v22, s[52:55], s18 offen offset: 0 lds // (op 30628) Load value direct to lds
v_add_co_u32 v22, s[20:21], v22, v14 // (op 30628) least significant half
v_addc_co_u32 v23, s[20:21], 0, v23, s[20:21] // (op 30628) most significant half

s_add_u32 m0, m0, 4096 // (op 30628) 
buffer_load_dwordx4 v22, s[52:55], s18 offen offset: 0 lds // (op 30628) Load value direct to lds
v_add_co_u32 v22, s[20:21], v22, v14 // (op 30628) least significant half
v_addc_co_u32 v23, s[20:21], 0, v23, s[20:21] // (op 30628) most significant half

s_add_u32 m0, m0, 4096 // (op 30628) 
buffer_load_dwordx4 v22, s[52:55], s18 offen offset: 0 lds // (op 30628) Load value direct to lds
v_add_co_u32 v22, s[20:21], v22, v14 // (op 30628) least significant half
v_addc_co_u32 v23, s[20:21], 0, v23, s[20:21] // (op 30628) most significant half

s_add_u32 m0, m0, 4096 // (op 30628) 
buffer_load_dwordx4 v22, s[52:55], s18 offen offset: 0 lds // (op 30628) Load value direct to lds






s_waitcnt vmcnt(29)
s_barrier  // (op 31925) 



ds_read_b128 v[14:17], v4 // (op 29641) Load local data 



ds_read_b128 v[44:47], v4 offset:1024 // (op 29646) Load local data 



ds_read_b128 v[48:51], v4 offset:2048 // (op 29651) Load local data 



ds_read_b128 v[52:55], v4 offset:3072 // (op 29656) Load local data 



ds_read_b128 v[56:59], v4 offset:4096 // (op 29661) Load local data 



ds_read_b128 v[60:63], v4 offset:5120 // (op 29666) Load local data 



ds_read_b128 v[64:67], v4 offset:6144 // (op 29671) Load local data 



ds_read_b128 v[68:71], v4 offset:7168 // (op 29676) Load local data 



ds_read_b128 v[72:75], v4 offset:8192 // (op 29681) Load local data 



ds_read_b128 v[76:79], v4 offset:9216 // (op 29686) Load local data 



ds_read_b128 v[80:83], v4 offset:10240 // (op 29691) Load local data 



ds_read_b128 v[84:87], v4 offset:11264 // (op 29696) Load local data 
s_waitcnt vmcnt(25)
s_barrier  // (op 31908) 



ds_read_b128 v[88:91], v0 offset:12288 // (op 29621) Load local data 



ds_read_b128 v[92:95], v0 offset:13312 // (op 29626) Load local data 



ds_read_b128 v[96:99], v0 offset:14336 // (op 29631) Load local data 



ds_read_b128 v[100:103], v0 offset:15360 // (op 29636) Load local data 
v_and_b32 v26, -4, v1 // (op 30840) 
v_sub_u32 v22, v1, v26 // (op 30840) 

v_lshlrev_b32 v26, 5, v22 // (op 30840) 

v_lshrrev_b32 v22, 2, v1 // (op 30840) 
v_lshl_add_u32 v23, v22, 7, v26 // (op 30840) 


v_lshlrev_b32 v22, 2, v23 // (op 30840) 

v_lshrrev_b32 v26, 3, v22 // (op 30840) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 30840) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 30897) 
v_sub_u32 v22, v1, v26 // (op 30897) 

v_lshlrev_b32 v26, 5, v22 // (op 30897) 

v_lshrrev_b32 v22, 2, v1 // (op 30897) 
v_lshl_add_u32 v23, v22, 7, v26 // (op 30897) 


v_lshlrev_b32 v22, 2, v23 // (op 30897) 

v_lshrrev_b32 v26, 3, v22 // (op 30897) 

s_nop 0
v_readfirstlane_b32 s1, v26 // (op 30897) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 30977) 
v_sub_u32 v22, v1, v26 // (op 30977) 

v_lshlrev_b32 v26, 5, v22 // (op 30977) 

v_lshrrev_b32 v22, 2, v1 // (op 30977) 
v_lshl_add_u32 v23, v22, 7, v26 // (op 30977) 


v_lshlrev_b32 v22, 2, v23 // (op 30977) 

v_lshrrev_b32 v26, 3, v22 // (op 30977) 

s_nop 0
v_readfirstlane_b32 s18, v26 // (op 30977) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 31021) 
v_sub_u32 v22, v1, v26 // (op 31021) 

v_lshlrev_b32 v26, 5, v22 // (op 31021) 

v_lshrrev_b32 v22, 2, v1 // (op 31021) 
v_lshl_add_u32 v23, v22, 7, v26 // (op 31021) 


v_lshlrev_b32 v22, 2, v23 // (op 31021) 

v_lshrrev_b32 v26, 3, v22 // (op 31021) 

s_nop 0
v_readfirstlane_b32 s19, v26 // (op 31021) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 31065) 
v_sub_u32 v22, v1, v26 // (op 31065) 

v_lshlrev_b32 v26, 5, v22 // (op 31065) 

v_lshrrev_b32 v22, 2, v1 // (op 31065) 
v_lshl_add_u32 v23, v22, 7, v26 // (op 31065) 


v_lshlrev_b32 v22, 2, v23 // (op 31065) 

v_lshrrev_b32 v26, 3, v22 // (op 31065) 

s_nop 0
v_readfirstlane_b32 s20, v26 // (op 31065) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 31109) 
v_sub_u32 v22, v1, v26 // (op 31109) 

v_lshlrev_b32 v26, 5, v22 // (op 31109) 

v_lshrrev_b32 v22, 2, v1 // (op 31109) 
v_lshl_add_u32 v23, v22, 7, v26 // (op 31109) 


v_lshlrev_b32 v22, 2, v23 // (op 31109) 

v_lshrrev_b32 v26, 3, v22 // (op 31109) 

s_nop 0
v_readfirstlane_b32 s21, v26 // (op 31109) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 31153) 
v_sub_u32 v22, v1, v26 // (op 31153) 

v_lshlrev_b32 v26, 5, v22 // (op 31153) 

v_lshrrev_b32 v22, 2, v1 // (op 31153) 
v_lshl_add_u32 v23, v22, 7, v26 // (op 31153) 


v_lshlrev_b32 v22, 2, v23 // (op 31153) 

v_lshrrev_b32 v26, 3, v22 // (op 31153) 

s_nop 0
v_readfirstlane_b32 s24, v26 // (op 31153) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 31197) 
v_sub_u32 v22, v1, v26 // (op 31197) 

v_lshlrev_b32 v26, 5, v22 // (op 31197) 

v_lshrrev_b32 v22, 2, v1 // (op 31197) 
v_lshl_add_u32 v23, v22, 7, v26 // (op 31197) 


v_lshlrev_b32 v22, 2, v23 // (op 31197) 

v_lshrrev_b32 v26, 3, v22 // (op 31197) 

s_nop 0
v_readfirstlane_b32 s25, v26 // (op 31197) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s49, 0 // (op 9476) call()
s_ashr_i64 s[50:51], s[42:43], 7 // (op 9475) 
s_mov_b32 s76, s50 // (op 9475) convert

s_and_b32 s50, s76, 4294967292 // (op 9475) 

s_cmp_lt_u32 s49, s50 // (op 9475) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ForLoopBottom_KLoop_9475 // (op 9475) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ForLoopBottom_KLoop_9475 if false)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ForLoopTop_KLoop_9475:
s_add_u32 s49, s49, 4 // (op 9477) 
s_waitcnt vmcnt(24)
v_mov_b32 v104, v28 // (op 30466) reg expression
v_mov_b32 v105, v29 // (op 30466) reg expression
v_mov_b32 v106, v30 // (op 30466) reg expression
v_mov_b32 v107, v31 // (op 30466) reg expression
s_waitcnt lgkmcnt(4)
s_barrier  // (op 31928) 
s_waitcnt vmcnt(23)
v_mov_b32 v108, v32 // (op 30479) reg expression

v_mov_b32 v109, v33 // (op 30479) reg expression
v_mov_b32 v110, v34 // (op 30479) reg expression
v_mov_b32 v111, v35 // (op 30479) reg expression
s_lshl_b64 s[50:51], s[30:31], 6 // (op 30561) 
s_lshl_b64 s[76:77], s[50:51], 2 // (op 30561) 

v_lshrrev_b64 v[22:23], 3, s[76:77] // (op 30561) 


v_mov_b64 v[112:113], v[20:21] // (op 30561) 
v_mov_b64 v[114:115], v[112:113] // (op 30561) 
s_mov_b32 m0, s1 // (op 30561) reg expression
s_mov_b32 s50, 256 // (op 30561) call()
buffer_load_dwordx4 v114, s[64:67], s50 offen offset: 0 lds // (op 30561) Load value direct to lds
v_add_co_u32 v114, s[76:77], v114, v22 // (op 30561) least significant half
v_addc_co_u32 v115, s[76:77], 0, v115, s[76:77] // (op 30561) most significant half

s_add_u32 m0, m0, 4096 // (op 30561) 
buffer_load_dwordx4 v114, s[64:67], s50 offen offset: 0 lds // (op 30561) Load value direct to lds
v_add_co_u32 v114, s[76:77], v114, v22 // (op 30561) least significant half
v_addc_co_u32 v115, s[76:77], 0, v115, s[76:77] // (op 30561) most significant half

s_add_u32 m0, m0, 4096 // (op 30561) 
buffer_load_dwordx4 v114, s[64:67], s50 offen offset: 0 lds // (op 30561) Load value direct to lds



s_waitcnt vmcnt(25)
v_mov_b32 v114, v36 // (op 30492) reg expression
v_mov_b32 v115, v37 // (op 30492) reg expression
v_mov_b32 v116, v38 // (op 30492) reg expression
v_mov_b32 v117, v39 // (op 30492) reg expression
s_waitcnt lgkmcnt(0)
s_barrier  // (op 31911) 
s_waitcnt vmcnt(24)
v_mov_b32 v118, v40 // (op 30505) reg expression

v_mov_b32 v119, v41 // (op 30505) reg expression
v_mov_b32 v120, v42 // (op 30505) reg expression
v_mov_b32 v121, v43 // (op 30505) reg expression
s_lshl_b64 s[50:51], s[26:27], 6 // (op 30556) 
s_lshl_b64 s[76:77], s[50:51], 2 // (op 30556) 

v_lshrrev_b64 v[22:23], 3, s[76:77] // (op 30556) 


v_mov_b64 v[122:123], v[18:19] // (op 30556) 
v_mov_b64 v[124:125], v[122:123] // (op 30556) 
s_add_u32 s50, 12288, s0 // (op 30556) 
s_mov_b32 m0, s50 // (op 30556) reg expression
s_mov_b32 s51, 256 // (op 30556) call()
v_bfe_u32 v126, v118, 0, 8 // (op 27783) 
v_bfe_u32 v127, v118, 8, 8 // (op 27783) 
v_bfe_u32 v128, v118, 16, 8 // (op 27783) 
v_bfe_u32 v129, v118, 24, 8 // (op 27783) 
v_permlane16_swap_b32 v126, v127 // (op 27783) 
s_nop 0
v_permlane16_swap_b32 v128, v129 // (op 27783) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v126, v128 // (op 27783) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v127, v129 // (op 27783) 
buffer_load_dwordx4 v124, s[52:55], s51 offen offset: 0 lds // (op 30556) Load value direct to lds
v_add_co_u32 v124, s[76:77], v124, v22 // (op 30556) least significant half
v_addc_co_u32 v125, s[76:77], 0, v125, s[76:77] // (op 30556) most significant half

s_add_u32 m0, m0, 4096 // (op 30556) 
buffer_load_dwordx4 v124, s[52:55], s51 offen offset: 0 lds // (op 30556) Load value direct to lds
v_add_co_u32 v124, s[76:77], v124, v22 // (op 30556) least significant half
v_addc_co_u32 v125, s[76:77], 0, v125, s[76:77] // (op 30556) most significant half

s_add_u32 m0, m0, 4096 // (op 30556) 
buffer_load_dwordx4 v124, s[52:55], s51 offen offset: 0 lds // (op 30556) Load value direct to lds
v_add_co_u32 v124, s[76:77], v124, v22 // (op 30556) least significant half
v_addc_co_u32 v125, s[76:77], 0, v125, s[76:77] // (op 30556) most significant half

s_add_u32 m0, m0, 4096 // (op 30556) 
v_bfe_u32 v130, v104, 0, 8 // (op 28040) 
v_bfe_u32 v131, v104, 8, 8 // (op 28040) 
v_bfe_u32 v132, v104, 16, 8 // (op 28040) 
v_bfe_u32 v133, v104, 24, 8 // (op 28040) 
v_permlane16_swap_b32 v130, v131 // (op 28040) 
s_nop 0
v_permlane16_swap_b32 v132, v133 // (op 28040) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v130, v132 // (op 28040) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v131, v133 // (op 28040) 
buffer_load_dwordx4 v124, s[52:55], s51 offen offset: 0 lds // (op 30556) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[14:17], a[0:3], v126, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 78) 
v_mov_b32 v26, v27 // (op 30474) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[44:47], a[4:7], v126, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 421) 
v_add_u32 v22, 16, v26 // (op 30474) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[48:51], a[8:11], v126, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 647) 
buffer_load_dwordx4 v[28:31], v22, s[72:75], 0 offen // (op 30474) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[52:55], a[12:15], v126, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 873) 
s_lshl_b32 s50, s38, 6 // (op 30487) 
s_add_u32 s50, s50, 16 // (op 30487) 
v_add_u32 v22, v26, s50 // (op 30487) 


buffer_load_dwordx4 v[32:35], v22, s[72:75], 0 offen // (op 30487) Load value

v_bfe_u32 v134, v108, 0, 8 // (op 28108) 


v_bfe_u32 v135, v108, 8, 8 // (op 28108) 
v_bfe_u32 v136, v108, 16, 8 // (op 28108) 
v_bfe_u32 v137, v108, 24, 8 // (op 28108) 
v_permlane16_swap_b32 v134, v135 // (op 28108) 
s_lshl_b32 s50, s38, 7 // (op 30500) 
s_add_u32 s50, s50, 16 // (op 30500) 
v_permlane16_swap_b32 v136, v137 // (op 28108) 
v_add_u32 v22, v26, s50 // (op 30500) 


s_nop 0
v_permlane32_swap_b32 v134, v136 // (op 28108) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v135, v137 // (op 28108) 
buffer_load_dwordx4 v[36:39], v22, s[72:75], 0 offen // (op 30500) Load value




v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[56:59], a[16:19], v126, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1099) 
v_add_u32 v27, 16, v27 // (op 31314) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[60:63], a[20:23], v126, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1325) 
v_mov_b32 v26, v25 // (op 30513) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:91], v[64:67], a[24:27], v126, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1551) 
v_add_u32 v22, 16, v26 // (op 30513) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[88:91], v[68:71], a[28:31], v126, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1777) 
buffer_load_dwordx4 v[40:43], v22, s[68:71], 0 offen // (op 30513) Load value


v_add_u32 v25, 16, v25 // (op 31262) 
v_bfe_u32 v138, v114, 0, 8 // (op 28176) 
v_bfe_u32 v139, v114, 8, 8 // (op 28176) 
v_bfe_u32 v140, v114, 16, 8 // (op 28176) 
v_bfe_u32 v141, v114, 24, 8 // (op 28176) 
v_permlane16_swap_b32 v138, v139 // (op 28176) 
s_nop 0
v_permlane16_swap_b32 v140, v141 // (op 28176) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v138, v140 // (op 28176) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v139, v141 // (op 28176) 
s_nop 0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[72:75], a[32:35], v126, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2003) Wait state hazard: VALU Write Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[76:79], a[36:39], v126, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2229) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[80:83], a[40:43], v126, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2455) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[84:87], a[44:47], v126, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2681) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[14:17], a[48:51], v127, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4449) 
s_waitcnt vmcnt(29)
s_barrier  // (op 31948) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[44:47], a[52:55], v127, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4373) 
ds_read_b128 v[142:145], v6 offset:28672 // (op 216) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[48:51], a[56:59], v127, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4297) 
ds_read_b128 v[146:149], v6 offset:29696 // (op 432) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[52:55], a[60:63], v127, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4221) 
ds_read_b128 v[150:153], v6 offset:30720 // (op 658) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[92:95], v[56:59], a[64:67], v127, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4145) 
ds_read_b128 v[154:157], v6 offset:31744 // (op 884) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[92:95], v[60:63], a[68:71], v127, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4069) 
ds_read_b128 v[158:161], v6 offset:32768 // (op 1110) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:95], v[64:67], a[72:75], v127, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3993) 
ds_read_b128 v[162:165], v6 offset:33792 // (op 1336) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:95], v[68:71], a[76:79], v127, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3917) 
ds_read_b128 v[166:169], v6 offset:34816 // (op 1562) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:95], v[72:75], a[80:83], v127, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3841) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[92:95], v[76:79], a[84:87], v127, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3765) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[92:95], v[80:83], a[88:91], v127, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3689) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[92:95], v[84:87], a[92:95], v127, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3613) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:99], v[14:17], a[96:99], v128, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7451) 
ds_read_b128 v[170:173], v6 offset:35840 // (op 1788) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[96:99], v[44:47], a[100:103], v128, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7375) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[96:99], v[48:51], a[104:107], v128, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7299) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[96:99], v[52:55], a[108:111], v128, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7223) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[96:99], v[56:59], a[112:115], v128, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7147) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[96:99], v[60:63], a[116:119], v128, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7071) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[96:99], v[64:67], a[120:123], v128, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6995) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[96:99], v[68:71], a[124:127], v128, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6919) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:99], v[72:75], a[128:131], v128, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6843) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[96:99], v[76:79], a[132:135], v128, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6767) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[96:99], v[80:83], a[136:139], v128, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6691) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[96:99], v[84:87], a[140:143], v128, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6615) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[100:103], v[14:17], a[144:147], v129, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10453) 
ds_read_b128 v[174:177], v6 offset:36864 // (op 2014) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[100:103], v[44:47], a[148:151], v129, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10377) 
ds_read_b128 v[178:181], v6 offset:37888 // (op 2240) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[100:103], v[48:51], a[152:155], v129, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10301) 
ds_read_b128 v[182:185], v6 offset:38912 // (op 2466) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[100:103], v[52:55], a[156:159], v129, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10225) 
ds_read_b128 v[186:189], v6 offset:39936 // (op 2692) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[100:103], v[56:59], a[160:163], v129, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10149) 
s_waitcnt vmcnt(25)
s_barrier  // (op 31941) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[100:103], v[60:63], a[164:167], v129, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10073) 
ds_read_b128 v[88:91], v3 offset:40960 // (op 214) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[100:103], v[64:67], a[168:171], v129, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9997) 
ds_read_b128 v[92:95], v3 offset:41984 // (op 3623) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[100:103], v[68:71], a[172:175], v129, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9921) 
ds_read_b128 v[96:99], v3 offset:43008 // (op 6625) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[100:103], v[72:75], a[176:179], v129, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9845) 
ds_read_b128 v[130:133], v3 offset:44032 // (op 9627) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[100:103], v[76:79], a[180:183], v129, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9769) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[100:103], v[80:83], a[184:187], v129, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9693) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[100:103], v[84:87], a[188:191], v129, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9617) 


v_bfe_u32 v124, v119, 0, 8 // (op 27798) 
v_bfe_u32 v125, v119, 8, 8 // (op 27798) 
v_bfe_u32 v126, v119, 16, 8 // (op 27798) 
v_bfe_u32 v127, v119, 24, 8 // (op 27798) 
v_permlane16_swap_b32 v124, v125 // (op 27798) 
s_nop 0
v_permlane16_swap_b32 v126, v127 // (op 27798) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v124, v126 // (op 27798) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v125, v127 // (op 27798) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 31952) 

s_lshl_b64 s[50:51], s[30:31], 6 // (op 30583) 
s_lshl_b64 s[76:77], s[50:51], 2 // (op 30583) 

v_lshrrev_b64 v[128:129], 3, s[76:77] // (op 30583) 


v_mov_b64 v[22:23], v[112:113] // (op 30583) 
s_add_u32 s50, 28672, s19 // (op 30583) 
s_mov_b32 m0, s50 // (op 30583) reg expression
s_mov_b32 s51, 320 // (op 30583) call()
buffer_load_dwordx4 v22, s[64:67], s51 offen offset: 0 lds // (op 30583) Load value direct to lds
v_add_co_u32 v22, s[76:77], v22, v128 // (op 30583) least significant half
v_addc_co_u32 v23, s[76:77], 0, v23, s[76:77] // (op 30583) most significant half

s_add_u32 m0, m0, 4096 // (op 30583) 
buffer_load_dwordx4 v22, s[64:67], s51 offen offset: 0 lds // (op 30583) Load value direct to lds
v_add_co_u32 v22, s[76:77], v22, v128 // (op 30583) least significant half
v_addc_co_u32 v23, s[76:77], 0, v23, s[76:77] // (op 30583) most significant half

s_add_u32 m0, m0, 4096 // (op 30583) 
buffer_load_dwordx4 v22, s[64:67], s51 offen offset: 0 lds // (op 30583) Load value direct to lds




v_bfe_u32 v134, v105, 0, 8 // (op 28055) 
v_bfe_u32 v135, v105, 8, 8 // (op 28055) 
v_bfe_u32 v136, v105, 16, 8 // (op 28055) 
v_bfe_u32 v137, v105, 24, 8 // (op 28055) 
v_permlane16_swap_b32 v134, v135 // (op 28055) 
s_nop 0
v_permlane16_swap_b32 v136, v137 // (op 28055) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v134, v136 // (op 28055) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v135, v137 // (op 28055) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 31945) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[142:145], a[0:3], v124, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 221) 
s_lshl_b64 s[50:51], s[26:27], 6 // (op 30578) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[146:149], a[4:7], v124, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 422) 
s_lshl_b64 s[76:77], s[50:51], 2 // (op 30578) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[150:153], a[8:11], v124, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 648) 
v_lshrrev_b64 v[128:129], 3, s[76:77] // (op 30578) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[154:157], a[12:15], v124, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 874) 
v_mov_b64 v[22:23], v[122:123] // (op 30578) 
s_add_u32 s50, 40960, s18 // (op 30578) 
s_mov_b32 m0, s50 // (op 30578) reg expression
s_mov_b32 s51, 320 // (op 30578) call()
v_bfe_u32 v138, v109, 0, 8 // (op 28123) 
v_bfe_u32 v139, v109, 8, 8 // (op 28123) 
v_bfe_u32 v140, v109, 16, 8 // (op 28123) 
v_bfe_u32 v141, v109, 24, 8 // (op 28123) 
v_permlane16_swap_b32 v138, v139 // (op 28123) 
s_nop 0
v_permlane16_swap_b32 v140, v141 // (op 28123) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v138, v140 // (op 28123) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v139, v141 // (op 28123) 
buffer_load_dwordx4 v22, s[52:55], s51 offen offset: 0 lds // (op 30578) Load value direct to lds
v_add_co_u32 v22, s[76:77], v22, v128 // (op 30578) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[158:161], a[16:19], v124, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1100) 
v_addc_co_u32 v23, s[76:77], 0, v23, s[76:77] // (op 30578) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[162:165], a[20:23], v124, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1326) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:91], v[166:169], a[24:27], v124, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1552) 
s_add_u32 m0, m0, 4096 // (op 30578) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[88:91], v[170:173], a[28:31], v124, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1778) 
buffer_load_dwordx4 v22, s[52:55], s51 offen offset: 0 lds // (op 30578) Load value direct to lds
v_add_co_u32 v22, s[76:77], v22, v128 // (op 30578) least significant half
v_addc_co_u32 v23, s[76:77], 0, v23, s[76:77] // (op 30578) most significant half

s_add_u32 m0, m0, 4096 // (op 30578) 
buffer_load_dwordx4 v22, s[52:55], s51 offen offset: 0 lds // (op 30578) Load value direct to lds
v_add_co_u32 v22, s[76:77], v22, v128 // (op 30578) least significant half
v_addc_co_u32 v23, s[76:77], 0, v23, s[76:77] // (op 30578) most significant half

s_add_u32 m0, m0, 4096 // (op 30578) 
v_bfe_u32 v190, v115, 0, 8 // (op 28191) 
v_bfe_u32 v191, v115, 8, 8 // (op 28191) 
v_bfe_u32 v192, v115, 16, 8 // (op 28191) 
v_bfe_u32 v193, v115, 24, 8 // (op 28191) 
v_permlane16_swap_b32 v190, v191 // (op 28191) 
s_nop 0
v_permlane16_swap_b32 v192, v193 // (op 28191) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v190, v192 // (op 28191) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v191, v193 // (op 28191) 
buffer_load_dwordx4 v22, s[52:55], s51 offen offset: 0 lds // (op 30578) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[174:177], a[32:35], v124, v190 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2004) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[178:181], a[36:39], v124, v191 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2230) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[182:185], a[40:43], v124, v192 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2456) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[186:189], a[44:47], v124, v193 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2682) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[142:145], a[48:51], v125, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4450) 
s_waitcnt vmcnt(29)
s_barrier  // (op 31963) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[146:149], a[52:55], v125, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4374) 
ds_read_b128 v[14:17], v8 offset:57344 // (op 244) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[150:153], a[56:59], v125, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4298) 
ds_read_b128 v[44:47], v8 offset:58368 // (op 442) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[154:157], a[60:63], v125, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4222) 
ds_read_b128 v[48:51], v8 offset:59392 // (op 668) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[92:95], v[158:161], a[64:67], v125, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4146) 
ds_read_b128 v[52:55], v8 offset:60416 // (op 894) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[92:95], v[162:165], a[68:71], v125, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4070) 
ds_read_b128 v[56:59], v8 offset:61440 // (op 1120) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:95], v[166:169], a[72:75], v125, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3994) 
ds_read_b128 v[60:63], v8 offset:62464 // (op 1346) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:95], v[170:173], a[76:79], v125, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3918) 
ds_read_b128 v[64:67], v8 offset:63488 // (op 1572) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:95], v[174:177], a[80:83], v125, v190 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3842) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[92:95], v[178:181], a[84:87], v125, v191 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3766) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[92:95], v[182:185], a[88:91], v125, v192 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3690) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[92:95], v[186:189], a[92:95], v125, v193 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3614) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:99], v[142:145], a[96:99], v126, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7452) 
ds_read_b128 v[68:71], v8 offset:64512 // (op 1798) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[96:99], v[146:149], a[100:103], v126, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7376) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[96:99], v[150:153], a[104:107], v126, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7300) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[96:99], v[154:157], a[108:111], v126, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7224) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[96:99], v[158:161], a[112:115], v126, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7148) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[96:99], v[162:165], a[116:119], v126, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7072) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[96:99], v[166:169], a[120:123], v126, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6996) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[96:99], v[170:173], a[124:127], v126, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6920) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:99], v[174:177], a[128:131], v126, v190 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6844) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[96:99], v[178:181], a[132:135], v126, v191 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6768) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[96:99], v[182:185], a[136:139], v126, v192 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6692) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[96:99], v[186:189], a[140:143], v126, v193 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6616) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[130:133], v[142:145], a[144:147], v127, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10454) 
v_add_u32 v26, 65536, v8 // (op 2024) 

ds_read_b128 v[72:75], v26 // (op 2024) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[130:133], v[146:149], a[148:151], v127, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10378) 
v_add_u32 v26, 66560, v8 // (op 2250) 

ds_read_b128 v[142:145], v26 // (op 2250) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[130:133], v[150:153], a[152:155], v127, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10302) 
v_add_u32 v26, 67584, v8 // (op 2476) 

ds_read_b128 v[146:149], v26 // (op 2476) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[130:133], v[154:157], a[156:159], v127, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10226) 
v_add_u32 v26, 68608, v8 // (op 2702) 


ds_read_b128 v[134:137], v26 // (op 2702) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[130:133], v[158:161], a[160:163], v127, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10150) 

s_waitcnt vmcnt(25)
s_barrier  // (op 31956) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[130:133], v[162:165], a[164:167], v127, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10074) 
v_add_u32 v26, 69632, v5 // (op 242) 

ds_read_b128 v[88:91], v26 // (op 242) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[130:133], v[166:169], a[168:171], v127, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9998) 
v_add_u32 v26, 70656, v5 // (op 3633) 

ds_read_b128 v[92:95], v26 // (op 3633) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[130:133], v[170:173], a[172:175], v127, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9922) 
v_add_u32 v26, 71680, v5 // (op 6635) 


ds_read_b128 v[96:99], v26 // (op 6635) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[130:133], v[174:177], a[176:179], v127, v190 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9846) 
v_add_u32 v26, 72704, v5 // (op 9637) 

ds_read_b128 v[100:103], v26 // (op 9637) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[130:133], v[178:181], a[180:183], v127, v191 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9770) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[130:133], v[182:185], a[184:187], v127, v192 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9694) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[130:133], v[186:189], a[188:191], v127, v193 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9618) 




v_bfe_u32 v138, v120, 0, 8 // (op 27815) 
v_bfe_u32 v139, v120, 8, 8 // (op 27815) 
v_bfe_u32 v140, v120, 16, 8 // (op 27815) 
v_bfe_u32 v141, v120, 24, 8 // (op 27815) 
v_permlane16_swap_b32 v138, v139 // (op 27815) 
s_nop 0
v_permlane16_swap_b32 v140, v141 // (op 27815) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v138, v140 // (op 27815) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v139, v141 // (op 27815) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 31966) 

s_lshl_b64 s[50:51], s[30:31], 6 // (op 30593) 
s_lshl_b64 s[76:77], s[50:51], 2 // (op 30593) 

v_lshrrev_b64 v[22:23], 3, s[76:77] // (op 30593) 


v_mov_b64 v[124:125], v[112:113] // (op 30593) 
s_add_u32 s50, 57344, s21 // (op 30593) 
s_mov_b32 m0, s50 // (op 30593) reg expression
s_mov_b32 s51, 384 // (op 30593) call()
buffer_load_dwordx4 v124, s[64:67], s51 offen offset: 0 lds // (op 30593) Load value direct to lds
v_add_co_u32 v124, s[76:77], v124, v22 // (op 30593) least significant half
v_addc_co_u32 v125, s[76:77], 0, v125, s[76:77] // (op 30593) most significant half

s_add_u32 m0, m0, 4096 // (op 30593) 
buffer_load_dwordx4 v124, s[64:67], s51 offen offset: 0 lds // (op 30593) Load value direct to lds
v_add_co_u32 v124, s[76:77], v124, v22 // (op 30593) least significant half
v_addc_co_u32 v125, s[76:77], 0, v125, s[76:77] // (op 30593) most significant half

s_add_u32 m0, m0, 4096 // (op 30593) 
buffer_load_dwordx4 v124, s[64:67], s51 offen offset: 0 lds // (op 30593) Load value direct to lds




v_bfe_u32 v124, v106, 0, 8 // (op 28073) 
v_bfe_u32 v125, v106, 8, 8 // (op 28073) 
v_bfe_u32 v126, v106, 16, 8 // (op 28073) 
v_bfe_u32 v127, v106, 24, 8 // (op 28073) 
v_permlane16_swap_b32 v124, v125 // (op 28073) 
s_nop 0
v_permlane16_swap_b32 v126, v127 // (op 28073) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v124, v126 // (op 28073) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v125, v127 // (op 28073) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 31960) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[14:17], a[0:3], v138, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 249) 
s_lshl_b64 s[50:51], s[26:27], 6 // (op 30588) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[44:47], a[4:7], v138, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 423) 
s_lshl_b64 s[76:77], s[50:51], 2 // (op 30588) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[48:51], a[8:11], v138, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 649) 
v_lshrrev_b64 v[22:23], 3, s[76:77] // (op 30588) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[52:55], a[12:15], v138, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 875) 
v_mov_b64 v[128:129], v[122:123] // (op 30588) 
s_add_u32 s50, 69632, s20 // (op 30588) 
s_mov_b32 m0, s50 // (op 30588) reg expression
s_mov_b32 s51, 384 // (op 30588) call()
v_bfe_u32 v130, v110, 0, 8 // (op 28141) 
v_bfe_u32 v131, v110, 8, 8 // (op 28141) 
v_bfe_u32 v132, v110, 16, 8 // (op 28141) 
v_bfe_u32 v133, v110, 24, 8 // (op 28141) 
v_permlane16_swap_b32 v130, v131 // (op 28141) 
s_nop 0
v_permlane16_swap_b32 v132, v133 // (op 28141) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v130, v132 // (op 28141) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v131, v133 // (op 28141) 
buffer_load_dwordx4 v128, s[52:55], s51 offen offset: 0 lds // (op 30588) Load value direct to lds
v_add_co_u32 v128, s[76:77], v128, v22 // (op 30588) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[56:59], a[16:19], v138, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1101) 
v_addc_co_u32 v129, s[76:77], 0, v129, s[76:77] // (op 30588) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[60:63], a[20:23], v138, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1327) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:91], v[64:67], a[24:27], v138, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1553) 
s_add_u32 m0, m0, 4096 // (op 30588) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[88:91], v[68:71], a[28:31], v138, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1779) 
buffer_load_dwordx4 v128, s[52:55], s51 offen offset: 0 lds // (op 30588) Load value direct to lds
v_add_co_u32 v128, s[76:77], v128, v22 // (op 30588) least significant half
v_addc_co_u32 v129, s[76:77], 0, v129, s[76:77] // (op 30588) most significant half

s_add_u32 m0, m0, 4096 // (op 30588) 
buffer_load_dwordx4 v128, s[52:55], s51 offen offset: 0 lds // (op 30588) Load value direct to lds
v_add_co_u32 v128, s[76:77], v128, v22 // (op 30588) least significant half
v_addc_co_u32 v129, s[76:77], 0, v129, s[76:77] // (op 30588) most significant half

s_add_u32 m0, m0, 4096 // (op 30588) 
v_bfe_u32 v150, v116, 0, 8 // (op 28209) 
v_bfe_u32 v151, v116, 8, 8 // (op 28209) 
v_bfe_u32 v152, v116, 16, 8 // (op 28209) 
v_bfe_u32 v153, v116, 24, 8 // (op 28209) 
v_permlane16_swap_b32 v150, v151 // (op 28209) 
s_nop 0
v_permlane16_swap_b32 v152, v153 // (op 28209) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v150, v152 // (op 28209) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v151, v153 // (op 28209) 
buffer_load_dwordx4 v128, s[52:55], s51 offen offset: 0 lds // (op 30588) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[72:75], a[32:35], v138, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2005) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[142:145], a[36:39], v138, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2231) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[146:149], a[40:43], v138, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2457) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[134:137], a[44:47], v138, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2683) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[14:17], a[48:51], v139, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4451) 
s_waitcnt vmcnt(29)
s_barrier  // (op 31976) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[44:47], a[52:55], v139, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4375) 
v_add_u32 v26, 86016, v10 // (op 272) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[48:51], a[56:59], v139, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4299) 
ds_read_b128 v[76:79], v26 // (op 272) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[52:55], a[60:63], v139, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4223) 
v_add_u32 v26, 87040, v10 // (op 452) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[92:95], v[56:59], a[64:67], v139, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4147) 
ds_read_b128 v[80:83], v26 // (op 452) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[92:95], v[60:63], a[68:71], v139, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4071) 
v_add_u32 v26, 88064, v10 // (op 678) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:95], v[64:67], a[72:75], v139, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3995) 
ds_read_b128 v[84:87], v26 // (op 678) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:95], v[68:71], a[76:79], v139, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3919) 
v_add_u32 v26, 89088, v10 // (op 904) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:95], v[72:75], a[80:83], v139, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3843) 
ds_read_b128 v[154:157], v26 // (op 904) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[92:95], v[142:145], a[84:87], v139, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3767) 
v_add_u32 v26, 90112, v10 // (op 1130) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[92:95], v[146:149], a[88:91], v139, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3691) 
ds_read_b128 v[158:161], v26 // (op 1130) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[92:95], v[134:137], a[92:95], v139, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3615) 
v_add_u32 v26, 91136, v10 // (op 1356) 
ds_read_b128 v[162:165], v26 // (op 1356) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:99], v[14:17], a[96:99], v140, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7453) 
v_add_u32 v26, 92160, v10 // (op 1582) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[96:99], v[44:47], a[100:103], v140, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7377) 
ds_read_b128 v[166:169], v26 // (op 1582) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[96:99], v[48:51], a[104:107], v140, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7301) 
v_add_u32 v26, 93184, v10 // (op 1808) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[96:99], v[52:55], a[108:111], v140, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7225) 
ds_read_b128 v[170:173], v26 // (op 1808) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[96:99], v[56:59], a[112:115], v140, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7149) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[96:99], v[60:63], a[116:119], v140, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7073) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[96:99], v[64:67], a[120:123], v140, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6997) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[96:99], v[68:71], a[124:127], v140, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6921) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:99], v[72:75], a[128:131], v140, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6845) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[96:99], v[142:145], a[132:135], v140, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6769) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[96:99], v[146:149], a[136:139], v140, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6693) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[96:99], v[134:137], a[140:143], v140, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6617) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[100:103], v[14:17], a[144:147], v141, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10455) 
v_add_u32 v26, 94208, v10 // (op 2034) 
ds_read_b128 v[174:177], v26 // (op 2034) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[100:103], v[44:47], a[148:151], v141, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10379) 
v_add_u32 v26, 95232, v10 // (op 2260) 
ds_read_b128 v[178:181], v26 // (op 2260) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[100:103], v[48:51], a[152:155], v141, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10303) 
v_add_u32 v26, 96256, v10 // (op 2486) 
ds_read_b128 v[182:185], v26 // (op 2486) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[100:103], v[52:55], a[156:159], v141, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10227) 
v_add_u32 v26, 97280, v10 // (op 2712) 

ds_read_b128 v[124:127], v26 // (op 2712) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[100:103], v[56:59], a[160:163], v141, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10151) 
s_waitcnt vmcnt(25)
s_barrier  // (op 31969) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[100:103], v[60:63], a[164:167], v141, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10075) 
v_add_u32 v26, 98304, v7 // (op 270) 
ds_read_b128 v[88:91], v26 // (op 270) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[100:103], v[64:67], a[168:171], v141, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9999) 
v_add_u32 v26, 99328, v7 // (op 3643) 
ds_read_b128 v[92:95], v26 // (op 3643) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[100:103], v[68:71], a[172:175], v141, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9923) 
v_add_u32 v26, 100352, v7 // (op 6645) 

ds_read_b128 v[96:99], v26 // (op 6645) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[100:103], v[72:75], a[176:179], v141, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9847) 
v_add_u32 v26, 101376, v7 // (op 9647) 
ds_read_b128 v[128:131], v26 // (op 9647) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[100:103], v[142:145], a[180:183], v141, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9771) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[100:103], v[146:149], a[184:187], v141, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9695) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[100:103], v[134:137], a[188:191], v141, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9619) 



v_bfe_u32 v132, v121, 0, 8 // (op 27832) 
v_bfe_u32 v133, v121, 8, 8 // (op 27832) 
v_bfe_u32 v134, v121, 16, 8 // (op 27832) 
v_bfe_u32 v135, v121, 24, 8 // (op 27832) 
v_permlane16_swap_b32 v132, v133 // (op 27832) 
s_nop 0
v_permlane16_swap_b32 v134, v135 // (op 27832) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v132, v134 // (op 27832) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v133, v135 // (op 27832) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 31979) 


s_lshl_b64 s[50:51], s[30:31], 6 // (op 30603) 
s_lshl_b64 s[76:77], s[50:51], 2 // (op 30603) 

v_lshrrev_b64 v[22:23], 3, s[76:77] // (op 30603) 


v_mov_b64 v[118:119], v[112:113] // (op 30603) 
s_add_u32 s50, 86016, s25 // (op 30603) 
s_mov_b32 m0, s50 // (op 30603) reg expression
s_mov_b32 s51, 448 // (op 30603) call()
buffer_load_dwordx4 v118, s[64:67], s51 offen offset: 0 lds // (op 30603) Load value direct to lds
v_add_co_u32 v118, s[76:77], v118, v22 // (op 30603) least significant half
v_addc_co_u32 v119, s[76:77], 0, v119, s[76:77] // (op 30603) most significant half

s_add_u32 m0, m0, 4096 // (op 30603) 
buffer_load_dwordx4 v118, s[64:67], s51 offen offset: 0 lds // (op 30603) Load value direct to lds
v_add_co_u32 v118, s[76:77], v118, v22 // (op 30603) least significant half
v_addc_co_u32 v119, s[76:77], 0, v119, s[76:77] // (op 30603) most significant half

s_add_u32 m0, m0, 4096 // (op 30603) 
buffer_load_dwordx4 v118, s[64:67], s51 offen offset: 0 lds // (op 30603) Load value direct to lds




v_bfe_u32 v118, v107, 0, 8 // (op 28091) 

v_bfe_u32 v119, v107, 8, 8 // (op 28091) 
v_bfe_u32 v120, v107, 16, 8 // (op 28091) 
v_bfe_u32 v121, v107, 24, 8 // (op 28091) 
v_permlane16_swap_b32 v118, v119 // (op 28091) 
v_add_co_u32 v20, vcc, 256, v20 // (op 30789) least significant half
v_permlane16_swap_b32 v120, v121 // (op 28091) 
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 30789) most significant half
s_nop 0
v_permlane32_swap_b32 v118, v120 // (op 28091) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v119, v121 // (op 28091) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 31973) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[76:79], a[0:3], v132, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 277) 
s_lshl_b64 s[50:51], s[26:27], 6 // (op 30598) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[80:83], a[4:7], v132, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 424) 
s_lshl_b64 s[76:77], s[50:51], 2 // (op 30598) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[84:87], a[8:11], v132, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 650) 
v_lshrrev_b64 v[112:113], 3, s[76:77] // (op 30598) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[154:157], a[12:15], v132, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 876) 
v_mov_b64 v[22:23], v[122:123] // (op 30598) 
s_add_u32 s50, 98304, s24 // (op 30598) 
s_mov_b32 m0, s50 // (op 30598) reg expression
s_mov_b32 s51, 448 // (op 30598) call()
v_bfe_u32 v104, v111, 0, 8 // (op 28159) 
v_bfe_u32 v105, v111, 8, 8 // (op 28159) 
v_bfe_u32 v106, v111, 16, 8 // (op 28159) 
v_bfe_u32 v107, v111, 24, 8 // (op 28159) 
v_permlane16_swap_b32 v104, v105 // (op 28159) 
s_nop 0
v_permlane16_swap_b32 v106, v107 // (op 28159) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v104, v106 // (op 28159) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v105, v107 // (op 28159) 
buffer_load_dwordx4 v22, s[52:55], s51 offen offset: 0 lds // (op 30598) Load value direct to lds

v_add_co_u32 v22, s[76:77], v22, v112 // (op 30598) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[158:161], a[16:19], v132, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1102) 
v_addc_co_u32 v23, s[76:77], 0, v23, s[76:77] // (op 30598) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[162:165], a[20:23], v132, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1328) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:91], v[166:169], a[24:27], v132, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1554) 
s_add_u32 m0, m0, 4096 // (op 30598) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[88:91], v[170:173], a[28:31], v132, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1780) 
buffer_load_dwordx4 v22, s[52:55], s51 offen offset: 0 lds // (op 30598) Load value direct to lds
v_add_co_u32 v22, s[76:77], v22, v112 // (op 30598) least significant half
v_addc_co_u32 v23, s[76:77], 0, v23, s[76:77] // (op 30598) most significant half

s_add_u32 m0, m0, 4096 // (op 30598) 
buffer_load_dwordx4 v22, s[52:55], s51 offen offset: 0 lds // (op 30598) Load value direct to lds
v_add_co_u32 v22, s[76:77], v22, v112 // (op 30598) least significant half
v_addc_co_u32 v23, s[76:77], 0, v23, s[76:77] // (op 30598) most significant half

s_add_u32 m0, m0, 4096 // (op 30598) 
v_bfe_u32 v108, v117, 0, 8 // (op 28227) 
v_bfe_u32 v109, v117, 8, 8 // (op 28227) 
v_bfe_u32 v110, v117, 16, 8 // (op 28227) 
v_bfe_u32 v111, v117, 24, 8 // (op 28227) 
v_permlane16_swap_b32 v108, v109 // (op 28227) 
s_nop 0
v_permlane16_swap_b32 v110, v111 // (op 28227) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v108, v110 // (op 28227) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v109, v111 // (op 28227) 
buffer_load_dwordx4 v22, s[52:55], s51 offen offset: 0 lds // (op 30598) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[174:177], a[32:35], v132, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2006) 
v_add_co_u32 v18, vcc, 256, v18 // (op 30725) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[178:181], a[36:39], v132, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2232) 
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 30725) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[182:185], a[40:43], v132, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2458) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[124:127], a[44:47], v132, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2684) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[76:79], a[48:51], v133, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4452) 
s_waitcnt vmcnt(29)
s_barrier  // (op 31934) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[80:83], a[52:55], v133, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4376) 
ds_read_b128 v[14:17], v4 // (op 67) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[84:87], a[56:59], v133, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4300) 
ds_read_b128 v[44:47], v4 offset:1024 // (op 418) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[154:157], a[60:63], v133, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4224) 
ds_read_b128 v[48:51], v4 offset:2048 // (op 644) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[92:95], v[158:161], a[64:67], v133, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4148) 
ds_read_b128 v[52:55], v4 offset:3072 // (op 870) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[92:95], v[162:165], a[68:71], v133, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4072) 
ds_read_b128 v[56:59], v4 offset:4096 // (op 1096) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:95], v[166:169], a[72:75], v133, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3996) 
ds_read_b128 v[60:63], v4 offset:5120 // (op 1322) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:95], v[170:173], a[76:79], v133, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3920) 
ds_read_b128 v[64:67], v4 offset:6144 // (op 1548) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:95], v[174:177], a[80:83], v133, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3844) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[92:95], v[178:181], a[84:87], v133, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3768) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[92:95], v[182:185], a[88:91], v133, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3692) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[92:95], v[124:127], a[92:95], v133, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3616) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:99], v[76:79], a[96:99], v134, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7454) 
ds_read_b128 v[68:71], v4 offset:7168 // (op 1774) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[96:99], v[80:83], a[100:103], v134, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7378) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[96:99], v[84:87], a[104:107], v134, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7302) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[96:99], v[154:157], a[108:111], v134, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7226) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[96:99], v[158:161], a[112:115], v134, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7150) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[96:99], v[162:165], a[116:119], v134, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7074) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[96:99], v[166:169], a[120:123], v134, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6998) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[96:99], v[170:173], a[124:127], v134, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6922) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:99], v[174:177], a[128:131], v134, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6846) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[96:99], v[178:181], a[132:135], v134, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6770) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[96:99], v[182:185], a[136:139], v134, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6694) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[96:99], v[124:127], a[140:143], v134, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6618) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[128:131], v[76:79], a[144:147], v135, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10456) 
ds_read_b128 v[72:75], v4 offset:8192 // (op 2000) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[128:131], v[80:83], a[148:151], v135, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10380) 
ds_read_b128 v[76:79], v4 offset:9216 // (op 2226) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[128:131], v[84:87], a[152:155], v135, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10304) 
ds_read_b128 v[80:83], v4 offset:10240 // (op 2452) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[128:131], v[154:157], a[156:159], v135, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10228) 
ds_read_b128 v[84:87], v4 offset:11264 // (op 2678) Load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[128:131], v[158:161], a[160:163], v135, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10152) 
s_waitcnt vmcnt(25)
s_barrier  // (op 31917) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[128:131], v[162:165], a[164:167], v135, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10076) 
ds_read_b128 v[88:91], v0 offset:12288 // (op 62) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[128:131], v[166:169], a[168:171], v135, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10000) 
ds_read_b128 v[92:95], v0 offset:13312 // (op 3609) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[128:131], v[170:173], a[172:175], v135, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9924) 
ds_read_b128 v[96:99], v0 offset:14336 // (op 6611) Load local data 





v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[128:131], v[174:177], a[176:179], v135, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9848) 
ds_read_b128 v[100:103], v0 offset:15360 // (op 9613) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[178:181], a[180:183], v135, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9772) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[128:131], v[182:185], a[184:187], v135, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9696) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[128:131], v[124:127], a[188:191], v135, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9620) 




s_ashr_i64 s[50:51], s[42:43], 7 // (op 9475) 
s_mov_b32 s76, s50 // (op 9475) convert

s_and_b32 s50, s76, 4294967292 // (op 9475) 

s_cmp_lt_u32 s49, s50 // (op 9475) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ForLoopTop_KLoop_9475 // (op 9475) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ForLoopTop_KLoop_9475 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ForLoopBottom_KLoop_9475:
















s_barrier  // (op 32058) 
s_mul_hi_i32 s0, s16, s47 // (op 30641) 
v_lshrrev_b32 v14, 2, v1 // (op 30641) 
s_add_i32 s1, s0, s16 // (op 30641) 

v_and_b32 v15, -4, v1 // (op 30641) 
s_ashr_i32 s0, s1, 31 // (op 30641) 
v_sub_u32 v16, v1, v15 // (op 30641) 

s_cmp_eq_i32 s47, 0 // (op 30641) 
s_mov_b32 s18, scc // (op 30641) 
v_lshlrev_b32 v15, 5, v16 // (op 30641) 

s_cmp_eq_i32 s18, 0 // (op 30641) 
s_cselect_b32 s19, 0, -1 // (op 30641) 

s_and_b32 s18, s11, 31 // (op 30641) 
s_lshl_b32 s20, 1, s18 // (op 30641) 
s_add_i32 s21, s20, s19 // (op 30641) 


s_and_b32 s19, s0, s21 // (op 30641) 


s_add_i32 s0, s1, s19 // (op 30641) 


s_ashr_i32 s1, s0, s18 // (op 30641) 


s_lshl_b32 s0, s11, 24 // (op 30641) 
s_ashr_i32 s18, s0, 31 // (op 30641) 

s_xor_b32 s0, s1, s18 // (op 30641) 

s_sub_i32 s1, s0, s18 // (op 30641) 


s_mul_hi_i32 s0, s1, s12 // (op 30641) 
s_add_i32 s18, s0, s1 // (op 30641) 

s_ashr_i32 s0, s18, 31 // (op 30641) 
s_cmp_eq_i32 s12, 0 // (op 30641) 
s_mov_b32 s19, scc // (op 30641) 
s_cmp_eq_i32 s19, 0 // (op 30641) 
s_cselect_b32 s20, 0, -1 // (op 30641) 

s_and_b32 s19, s13, 31 // (op 30641) 
s_lshl_b32 s21, 1, s19 // (op 30641) 
s_add_i32 s24, s21, s20 // (op 30641) 


s_and_b32 s25, s0, s24 // (op 30641) 


s_add_i32 s0, s18, s25 // (op 30641) 


s_ashr_i32 s18, s0, s19 // (op 30641) 


s_lshl_b32 s0, s13, 24 // (op 30641) 
s_ashr_i32 s19, s0, 31 // (op 30641) 

s_xor_b32 s0, s18, s19 // (op 30641) 

s_sub_i32 s18, s0, s19 // (op 30641) 


s_mul_i32 s0, s1, s48 // (op 30641) 

s_sub_u32 s1, s16, s0 // (op 30641) 

s_mul_i32 s0, s18, s48 // (op 30641) 
s_add_u32 s0, s0, s1 // (op 30641) 


s_mul_hi_i32 s1, s16, s10 // (op 30641) 
s_add_i32 s18, s1, s16 // (op 30641) 

s_ashr_i32 s1, s18, 31 // (op 30641) 
s_cmp_eq_i32 s10, 0 // (op 30641) 
s_mov_b32 s19, scc // (op 30641) 
s_cmp_eq_i32 s19, 0 // (op 30641) 
s_cselect_b32 s20, 0, -1 // (op 30641) 

s_and_b32 s19, s44, 31 // (op 30641) 
s_lshl_b32 s21, 1, s19 // (op 30641) 
s_add_i32 s24, s21, s20 // (op 30641) 


s_and_b32 s25, s1, s24 // (op 30641) 


s_add_i32 s1, s18, s25 // (op 30641) 


s_ashr_i32 s18, s1, s19 // (op 30641) 


s_lshl_b32 s1, s44, 24 // (op 30641) 
s_ashr_i32 s19, s1, 31 // (op 30641) 

s_xor_b32 s1, s18, s19 // (op 30641) 

s_sub_i32 s18, s1, s19 // (op 30641) 


s_mul_i32 s1, s48, s6 // (op 30641) 
s_mul_i32 s19, s14, s1 // (op 30641) 

s_mul_i32 s1, s18, s19 // (op 30641) 

s_sub_u32 s19, s16, s1 // (op 30641) 

s_mov_b32 s1, s19 // (op 30641) 
s_mul_hi_i32 s20, s1, s45 // (op 30641) 
s_add_i32 s21, s20, s1 // (op 30641) 


s_ashr_i32 s1, s21, 31 // (op 30641) 
s_cmp_eq_i32 s45, 0 // (op 30641) 
s_mov_b32 s20, scc // (op 30641) 
s_cmp_eq_i32 s20, 0 // (op 30641) 
s_cselect_b32 s24, 0, -1 // (op 30641) 

s_and_b32 s20, s15, 31 // (op 30641) 
s_lshl_b32 s25, 1, s20 // (op 30641) 
s_add_i32 s49, s25, s24 // (op 30641) 


s_and_b32 s24, s1, s49 // (op 30641) 


s_add_i32 s1, s21, s24 // (op 30641) 


s_ashr_i32 s21, s1, s20 // (op 30641) 


s_lshl_b32 s1, s15, 24 // (op 30641) 
s_ashr_i32 s20, s1, 31 // (op 30641) 

s_xor_b32 s1, s21, s20 // (op 30641) 

s_sub_i32 s21, s1, s20 // (op 30641) 


s_mul_i32 s1, s46, s48 // (op 30641) 
s_add_u32 s24, s8, 256 // (op 30641) Start of Int64 add, locking SCC
s_addc_u32 s25, s9, 0 // (op 30641) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 30641) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 30641) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 30641) 

s_sub_i32 s20, s24, s1 // (op 30641) 

s_mul_i32 s24, s21, s20 // (op 30641) 


s_sub_u32 s25, s19, s24 // (op 30641) 


s_add_u32 s24, s25, s1 // (op 30641) 


s_cmp_eq_i32 s18, 0 // (op 30641) 
s_mov_b32 s1, scc // (op 30641) 

s_cmp_eq_i32 s1, 0 // (op 30641) 
s_cselect_b32 s25, s24, s0 // (op 30641) 



v_lshl_add_u32 v16, s25, 8, v14 // (op 30641) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 30641) 
s_lshr_b32 s18, s0, 2 // (op 30641) 

v_lshl_add_u32 v14, s18, 9, v15 // (op 30641) 


v_mov_b32 v27, 0 // (op 30641) convert
v_mov_b32 v26, v14 // (op 30641) convert

v_mul_lo_u32 v15, v16, s27 // (op 30641) most significant: low of low * high
v_mul_hi_u32 v17, v16, s26 // (op 30641) most significant: high of low * low
v_mul_lo_u32 v14, v16, s26 // (op 30641) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 30641) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v26 // (op 30641) least significant half
v_addc_co_u32 v15, s[0:1], v15, v27, s[0:1] // (op 30641) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 30641) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 30641) 

s_mul_i32 s1, s26, s9 // (op 30643) most significant: low of low * high
s_mul_i32 s18, s27, s8 // (op 30643) most significant: low of high * low
s_mul_i32 s0, s26, s8 // (op 30643) least significant
s_mul_hi_u32 s19, s26, s8 // (op 30643) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30643) most significant: sum
s_add_u32 s1, s1, s19 // (op 30643) most significant: sum


s_lshl_b64 s[24:25], s[0:1], 2 // (op 30643) 

s_ashr_i64 s[0:1], s[24:25], 3 // (op 30643) 

s_mov_b32 s52, s28 // (op 30643) reg expression
s_mov_b32 s53, s29 // (op 30643) reg expression
s_mov_b32 s54, s0 // (op 30643) reg expression
s_mov_b32 s55, 131072 // (op 30643) call()

s_mul_hi_i32 s0, s16, s47 // (op 30755) 
v_lshrrev_b32 v16, 2, v1 // (op 30755) 
s_add_i32 s1, s0, s16 // (op 30755) 

v_and_b32 v17, -4, v1 // (op 30755) 
s_ashr_i32 s0, s1, 31 // (op 30755) 
v_sub_u32 v18, v1, v17 // (op 30755) 

s_cmp_eq_i32 s47, 0 // (op 30755) 
s_mov_b32 s18, scc // (op 30755) 
v_lshlrev_b32 v17, 5, v18 // (op 30755) 

s_cmp_eq_i32 s18, 0 // (op 30755) 
s_cselect_b32 s19, 0, -1 // (op 30755) 

s_and_b32 s18, s11, 31 // (op 30755) 
s_lshl_b32 s20, 1, s18 // (op 30755) 
s_add_i32 s21, s20, s19 // (op 30755) 


s_and_b32 s19, s0, s21 // (op 30755) 


s_add_i32 s0, s1, s19 // (op 30755) 


s_ashr_i32 s1, s0, s18 // (op 30755) 


s_lshl_b32 s0, s11, 24 // (op 30755) 
s_ashr_i32 s18, s0, 31 // (op 30755) 

s_xor_b32 s0, s1, s18 // (op 30755) 

s_sub_i32 s1, s0, s18 // (op 30755) 


s_mul_hi_i32 s0, s1, s12 // (op 30755) 
s_add_i32 s18, s0, s1 // (op 30755) 

s_ashr_i32 s0, s18, 31 // (op 30755) 
s_cmp_eq_i32 s12, 0 // (op 30755) 
s_mov_b32 s19, scc // (op 30755) 
s_cmp_eq_i32 s19, 0 // (op 30755) 
s_cselect_b32 s20, 0, -1 // (op 30755) 

s_and_b32 s19, s13, 31 // (op 30755) 
s_lshl_b32 s21, 1, s19 // (op 30755) 
s_add_i32 s24, s21, s20 // (op 30755) 


s_and_b32 s25, s0, s24 // (op 30755) 


s_add_i32 s0, s18, s25 // (op 30755) 


s_ashr_i32 s18, s0, s19 // (op 30755) 


s_lshl_b32 s0, s13, 24 // (op 30755) 
s_ashr_i32 s19, s0, 31 // (op 30755) 

s_xor_b32 s0, s18, s19 // (op 30755) 

s_sub_i32 s18, s0, s19 // (op 30755) 


s_mul_i32 s0, s18, s6 // (op 30755) 

s_sub_i32 s18, s1, s0 // (op 30755) 


s_mul_hi_i32 s0, s16, s10 // (op 30755) 
s_add_i32 s1, s0, s16 // (op 30755) 

s_ashr_i32 s0, s1, 31 // (op 30755) 
s_cmp_eq_i32 s10, 0 // (op 30755) 
s_mov_b32 s19, scc // (op 30755) 
s_cmp_eq_i32 s19, 0 // (op 30755) 
s_cselect_b32 s20, 0, -1 // (op 30755) 

s_and_b32 s19, s44, 31 // (op 30755) 
s_lshl_b32 s21, 1, s19 // (op 30755) 
s_add_i32 s24, s21, s20 // (op 30755) 


s_and_b32 s25, s0, s24 // (op 30755) 


s_add_i32 s0, s1, s25 // (op 30755) 


s_ashr_i32 s1, s0, s19 // (op 30755) 


s_lshl_b32 s0, s44, 24 // (op 30755) 
s_ashr_i32 s19, s0, 31 // (op 30755) 

s_xor_b32 s0, s1, s19 // (op 30755) 

s_sub_i32 s1, s0, s19 // (op 30755) 


s_mul_i32 s0, s48, s6 // (op 30755) 
s_mul_i32 s19, s14, s0 // (op 30755) 

s_mul_i32 s0, s1, s19 // (op 30755) 

s_sub_u32 s19, s16, s0 // (op 30755) 

s_mul_hi_i32 s0, s19, s45 // (op 30755) 
s_add_i32 s20, s0, s19 // (op 30755) 


s_ashr_i32 s0, s20, 31 // (op 30755) 
s_cmp_eq_i32 s45, 0 // (op 30755) 
s_mov_b32 s19, scc // (op 30755) 
s_cmp_eq_i32 s19, 0 // (op 30755) 
s_cselect_b32 s21, 0, -1 // (op 30755) 

s_and_b32 s19, s15, 31 // (op 30755) 
s_lshl_b32 s24, 1, s19 // (op 30755) 
s_add_i32 s25, s24, s21 // (op 30755) 


s_and_b32 s21, s0, s25 // (op 30755) 


s_add_i32 s0, s20, s21 // (op 30755) 


s_ashr_i32 s20, s0, s19 // (op 30755) 


s_lshl_b32 s0, s15, 24 // (op 30755) 
s_ashr_i32 s19, s0, 31 // (op 30755) 

s_xor_b32 s0, s20, s19 // (op 30755) 

s_sub_i32 s20, s0, s19 // (op 30755) 


s_cmp_eq_i32 s1, 0 // (op 30755) 
s_mov_b32 s0, scc // (op 30755) 

s_cmp_eq_i32 s0, 0 // (op 30755) 
s_cselect_b32 s1, s20, s18 // (op 30755) 



v_mov_b32 v18, 192 // (op 30755) 
v_mul_lo_u32 v19, s1, v18 // (op 30755) 

v_add_u32 v19, v19, v16 // (op 30755) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 30755) 
v_mul_lo_u32 v21, v19, s31 // (op 30755) most significant: low of low * high
s_lshr_b32 s18, s0, 2 // (op 30755) 

v_mul_hi_u32 v16, v19, s30 // (op 30755) most significant: high of low * low
v_mul_lo_u32 v20, v19, s30 // (op 30755) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 30755) most significant: sum


v_lshl_add_u32 v16, s18, 9, v17 // (op 30755) 


v_mov_b32 v19, 0 // (op 30755) convert
v_mov_b32 v18, v16 // (op 30755) convert

v_add_co_u32 v16, s[0:1], v18, v20 // (op 30755) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 30755) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 30755) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 30755) 

s_mul_i32 s1, s30, s23 // (op 30757) most significant: low of low * high
s_mul_i32 s18, s31, s22 // (op 30757) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 30757) least significant
s_mul_hi_u32 s19, s30, s22 // (op 30757) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30757) most significant: sum
s_add_u32 s1, s1, s19 // (op 30757) most significant: sum


s_lshl_b64 s[24:25], s[0:1], 2 // (op 30757) 

s_ashr_i64 s[0:1], s[24:25], 3 // (op 30757) 

s_mov_b32 s64, s32 // (op 30757) reg expression
s_mov_b32 s65, s33 // (op 30757) reg expression
s_mov_b32 s66, s0 // (op 30757) reg expression
s_mov_b32 s67, 131072 // (op 30757) call()

v_and_b32 v16, -4, v1 // (op 30853) 
v_sub_u32 v17, v1, v16 // (op 30853) 

v_lshlrev_b32 v16, 5, v17 // (op 30853) 

v_lshrrev_b32 v17, 2, v1 // (op 30853) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 30853) 


v_lshlrev_b32 v16, 2, v20 // (op 30853) 

v_lshrrev_b32 v17, 3, v16 // (op 30853) 

s_nop 0
v_readfirstlane_b32 s0, v17 // (op 30853) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 30910) 
v_sub_u32 v17, v1, v16 // (op 30910) 

v_lshlrev_b32 v16, 5, v17 // (op 30910) 

v_lshrrev_b32 v17, 2, v1 // (op 30910) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 30910) 


v_lshlrev_b32 v16, 2, v20 // (op 30910) 

v_lshrrev_b32 v17, 3, v16 // (op 30910) 

s_nop 0
v_readfirstlane_b32 s1, v17 // (op 30910) Wait state hazard: VALU Write Readlane Hazard

s_mul_hi_i32 s18, s16, s47 // (op 31351) 
v_and_b32 v16, 63, v1 // (op 31351) 
s_add_i32 s19, s18, s16 // (op 31351) 

v_lshrrev_b32 v17, 4, v16 // (op 31351) 
s_ashr_i32 s18, s19, 31 // (op 31351) 
v_and_b32 v20, 3, v17 // (op 31351) 

s_cmp_eq_i32 s47, 0 // (op 31351) 
s_mov_b32 s20, scc // (op 31351) 
v_and_b32 v17, 15, v1 // (op 31351) 
s_cmp_eq_i32 s20, 0 // (op 31351) 
s_cselect_b32 s21, 0, -1 // (op 31351) 

v_lshl_add_u32 v21, v20, 4, v17 // (op 31351) 


s_and_b32 s20, s11, 31 // (op 31351) 
v_and_b32 v17, 4294967232, v1 // (op 31351) 
s_lshl_b32 s24, 1, s20 // (op 31351) 
v_add_u32 v20, v17, v21 // (op 31351) 


s_add_i32 s25, s24, s21 // (op 31351) 


v_lshrrev_b32 v17, 6, v16 // (op 31351) 

s_and_b32 s21, s18, s25 // (op 31351) 


v_lshlrev_b32 v16, 2, v17 // (op 31351) 

s_add_i32 s18, s19, s21 // (op 31351) 


s_ashr_i32 s19, s18, s20 // (op 31351) 


s_lshl_b32 s18, s11, 24 // (op 31351) 
s_ashr_i32 s20, s18, 31 // (op 31351) 

s_xor_b32 s18, s19, s20 // (op 31351) 

s_sub_i32 s19, s18, s20 // (op 31351) 


s_mul_hi_i32 s18, s19, s12 // (op 31351) 
s_add_i32 s20, s18, s19 // (op 31351) 

s_ashr_i32 s18, s20, 31 // (op 31351) 
s_cmp_eq_i32 s12, 0 // (op 31351) 
s_mov_b32 s21, scc // (op 31351) 
s_cmp_eq_i32 s21, 0 // (op 31351) 
s_cselect_b32 s24, 0, -1 // (op 31351) 

s_and_b32 s21, s13, 31 // (op 31351) 
s_lshl_b32 s25, 1, s21 // (op 31351) 
s_add_i32 s49, s25, s24 // (op 31351) 


s_and_b32 s24, s18, s49 // (op 31351) 


s_add_i32 s18, s20, s24 // (op 31351) 


s_ashr_i32 s20, s18, s21 // (op 31351) 


s_lshl_b32 s18, s13, 24 // (op 31351) 
s_ashr_i32 s21, s18, 31 // (op 31351) 

s_xor_b32 s18, s20, s21 // (op 31351) 

s_sub_i32 s20, s18, s21 // (op 31351) 


s_mul_i32 s18, s19, s48 // (op 31351) 

s_sub_u32 s19, s16, s18 // (op 31351) 

s_mul_i32 s18, s20, s48 // (op 31351) 
s_add_u32 s18, s18, s19 // (op 31351) 


s_mul_hi_i32 s19, s16, s10 // (op 31351) 
s_add_i32 s20, s19, s16 // (op 31351) 

s_ashr_i32 s19, s20, 31 // (op 31351) 
s_cmp_eq_i32 s10, 0 // (op 31351) 
s_mov_b32 s21, scc // (op 31351) 
s_cmp_eq_i32 s21, 0 // (op 31351) 
s_cselect_b32 s24, 0, -1 // (op 31351) 

s_and_b32 s21, s44, 31 // (op 31351) 
s_lshl_b32 s25, 1, s21 // (op 31351) 
s_add_i32 s49, s25, s24 // (op 31351) 


s_and_b32 s24, s19, s49 // (op 31351) 


s_add_i32 s19, s20, s24 // (op 31351) 


s_ashr_i32 s20, s19, s21 // (op 31351) 


s_lshl_b32 s19, s44, 24 // (op 31351) 
s_ashr_i32 s21, s19, 31 // (op 31351) 

s_xor_b32 s19, s20, s21 // (op 31351) 

s_sub_i32 s20, s19, s21 // (op 31351) 


s_mul_i32 s19, s48, s6 // (op 31351) 
s_mul_i32 s21, s14, s19 // (op 31351) 

s_mul_i32 s19, s20, s21 // (op 31351) 

s_sub_u32 s21, s16, s19 // (op 31351) 

s_mov_b32 s19, s21 // (op 31351) 
s_mul_hi_i32 s24, s19, s45 // (op 31351) 
s_add_i32 s25, s24, s19 // (op 31351) 


s_ashr_i32 s19, s25, 31 // (op 31351) 
s_cmp_eq_i32 s45, 0 // (op 31351) 
s_mov_b32 s24, scc // (op 31351) 
s_cmp_eq_i32 s24, 0 // (op 31351) 
s_cselect_b32 s49, 0, -1 // (op 31351) 

s_and_b32 s24, s15, 31 // (op 31351) 
s_lshl_b32 s50, 1, s24 // (op 31351) 
s_add_i32 s51, s50, s49 // (op 31351) 


s_and_b32 s49, s19, s51 // (op 31351) 


s_add_i32 s19, s25, s49 // (op 31351) 


s_ashr_i32 s25, s19, s24 // (op 31351) 


s_lshl_b32 s19, s15, 24 // (op 31351) 
s_ashr_i32 s24, s19, 31 // (op 31351) 

s_xor_b32 s19, s25, s24 // (op 31351) 

s_sub_i32 s25, s19, s24 // (op 31351) 


s_mul_i32 s19, s46, s48 // (op 31351) 
s_add_u32 s50, s8, 256 // (op 31351) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 31351) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 31351) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 31351) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 31351) 

s_sub_i32 s24, s50, s19 // (op 31351) 

s_mul_i32 s49, s25, s24 // (op 31351) 


s_sub_u32 s24, s21, s49 // (op 31351) 


s_add_u32 s21, s24, s19 // (op 31351) 


s_cmp_eq_i32 s20, 0 // (op 31351) 
s_mov_b32 s19, scc // (op 31351) 

s_cmp_eq_i32 s19, 0 // (op 31351) 
s_cselect_b32 s20, s21, s18 // (op 31351) 



v_lshl_add_u32 v17, s20, 8, v20 // (op 31351) 


s_ashr_i64 s[24:25], s[42:43], 7 // (op 31351) 
s_lshr_b32 s18, s24, 2 // (op 31351) 

v_lshl_add_u32 v20, s18, 4, v16 // (op 31351) 


v_mul_lo_u32 v16, v17, s34 // (op 31351) 
v_add_u32 v16, v16, v20 // (op 31351) 


s_mul_i32 s25, s34, s9 // (op 31353) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 31353) most significant: low of high * low
s_mul_i32 s24, s34, s8 // (op 31353) least significant
s_mul_hi_u32 s19, s34, s8 // (op 31353) most significant: high of low * low
s_add_u32 s25, s25, s18 // (op 31353) most significant: sum
s_add_u32 s25, s25, s19 // (op 31353) most significant: sum


s_mov_b32 s68, s36 // (op 31353) reg expression
s_mov_b32 s69, s37 // (op 31353) reg expression
s_mov_b32 s70, s24 // (op 31353) reg expression
s_mov_b32 s71, 131072 // (op 31353) call()

s_mul_hi_i32 s18, s16, s47 // (op 31378) 
v_and_b32 v17, 63, v1 // (op 31378) 
s_add_i32 s19, s18, s16 // (op 31378) 

v_lshrrev_b32 v20, 4, v17 // (op 31378) 
s_ashr_i32 s18, s19, 31 // (op 31378) 
v_and_b32 v21, 3, v20 // (op 31378) 

s_cmp_eq_i32 s47, 0 // (op 31378) 
s_mov_b32 s20, scc // (op 31378) 
v_and_b32 v20, 15, v1 // (op 31378) 
s_cmp_eq_i32 s20, 0 // (op 31378) 
s_cselect_b32 s21, 0, -1 // (op 31378) 

v_lshl_add_u32 v22, v21, 4, v20 // (op 31378) 


s_and_b32 s20, s11, 31 // (op 31378) 
v_lshrrev_b32 v20, 6, v17 // (op 31378) 

s_lshl_b32 s24, 1, s20 // (op 31378) 
v_lshlrev_b32 v17, 2, v20 // (op 31378) 

s_add_i32 s25, s24, s21 // (op 31378) 


s_and_b32 s21, s18, s25 // (op 31378) 


s_add_i32 s18, s19, s21 // (op 31378) 


s_ashr_i32 s19, s18, s20 // (op 31378) 


s_lshl_b32 s18, s11, 24 // (op 31378) 
s_ashr_i32 s20, s18, 31 // (op 31378) 

s_xor_b32 s18, s19, s20 // (op 31378) 

s_sub_i32 s19, s18, s20 // (op 31378) 


s_mul_hi_i32 s18, s19, s12 // (op 31378) 
s_add_i32 s20, s18, s19 // (op 31378) 

s_ashr_i32 s18, s20, 31 // (op 31378) 
s_cmp_eq_i32 s12, 0 // (op 31378) 
s_mov_b32 s21, scc // (op 31378) 
s_cmp_eq_i32 s21, 0 // (op 31378) 
s_cselect_b32 s24, 0, -1 // (op 31378) 

s_and_b32 s21, s13, 31 // (op 31378) 
s_lshl_b32 s25, 1, s21 // (op 31378) 
s_add_i32 s49, s25, s24 // (op 31378) 


s_and_b32 s24, s18, s49 // (op 31378) 


s_add_i32 s18, s20, s24 // (op 31378) 


s_ashr_i32 s20, s18, s21 // (op 31378) 


s_lshl_b32 s18, s13, 24 // (op 31378) 
s_ashr_i32 s21, s18, 31 // (op 31378) 

s_xor_b32 s18, s20, s21 // (op 31378) 

s_sub_i32 s20, s18, s21 // (op 31378) 


s_mul_i32 s18, s20, s6 // (op 31378) 

s_sub_i32 s20, s19, s18 // (op 31378) 


s_mul_hi_i32 s21, s16, s10 // (op 31378) 
s_add_i32 s18, s21, s16 // (op 31378) 

s_ashr_i32 s19, s18, 31 // (op 31378) 
s_cmp_eq_i32 s10, 0 // (op 31378) 
s_mov_b32 s21, scc // (op 31378) 
s_cmp_eq_i32 s21, 0 // (op 31378) 
s_cselect_b32 s24, 0, -1 // (op 31378) 

s_and_b32 s21, s44, 31 // (op 31378) 
s_lshl_b32 s25, 1, s21 // (op 31378) 
s_add_i32 s49, s25, s24 // (op 31378) 


s_and_b32 s24, s19, s49 // (op 31378) 


s_add_i32 s19, s18, s24 // (op 31378) 


s_ashr_i32 s18, s19, s21 // (op 31378) 


s_lshl_b32 s19, s44, 24 // (op 31378) 
s_ashr_i32 s21, s19, 31 // (op 31378) 

s_xor_b32 s19, s18, s21 // (op 31378) 

s_sub_i32 s18, s19, s21 // (op 31378) 


s_mul_i32 s19, s48, s6 // (op 31378) 
s_mul_i32 s21, s14, s19 // (op 31378) 

s_mul_i32 s19, s18, s21 // (op 31378) 

s_sub_u32 s21, s16, s19 // (op 31378) 

s_mul_hi_i32 s19, s21, s45 // (op 31378) 
s_add_i32 s24, s19, s21 // (op 31378) 


s_ashr_i32 s19, s24, 31 // (op 31378) 
s_cmp_eq_i32 s45, 0 // (op 31378) 
s_mov_b32 s21, scc // (op 31378) 
s_cmp_eq_i32 s21, 0 // (op 31378) 
s_cselect_b32 s25, 0, -1 // (op 31378) 

s_and_b32 s21, s15, 31 // (op 31378) 
s_lshl_b32 s49, 1, s21 // (op 31378) 
s_add_i32 s50, s49, s25 // (op 31378) 


s_and_b32 s25, s19, s50 // (op 31378) 


s_add_i32 s19, s24, s25 // (op 31378) 


s_ashr_i32 s24, s19, s21 // (op 31378) 


s_lshl_b32 s19, s15, 24 // (op 31378) 
s_ashr_i32 s21, s19, 31 // (op 31378) 

s_xor_b32 s19, s24, s21 // (op 31378) 

s_sub_i32 s24, s19, s21 // (op 31378) 


s_cmp_eq_i32 s18, 0 // (op 31378) 
s_mov_b32 s19, scc // (op 31378) 

s_cmp_eq_i32 s19, 0 // (op 31378) 
s_cselect_b32 s18, s24, s20 // (op 31378) 



v_mov_b32 v20, 192 // (op 31378) 
v_mul_lo_u32 v21, s18, v20 // (op 31378) 

v_add_u32 v21, v21, v22 // (op 31378) 


s_ashr_i64 s[24:25], s[42:43], 7 // (op 31378) 
v_mul_lo_u32 v20, v21, s38 // (op 31378) 

s_lshr_b32 s18, s24, 2 // (op 31378) 

v_lshl_add_u32 v21, s18, 4, v17 // (op 31378) 


v_add_u32 v17, v21, v20 // (op 31378) 


s_mul_i32 s25, s38, s23 // (op 31380) most significant: low of low * high
s_mul_i32 s18, s39, s22 // (op 31380) most significant: low of high * low
s_mul_i32 s24, s38, s22 // (op 31380) least significant
s_mul_hi_u32 s19, s38, s22 // (op 31380) most significant: high of low * low
s_add_u32 s25, s25, s18 // (op 31380) most significant: sum
s_add_u32 s25, s25, s19 // (op 31380) most significant: sum


s_mov_b32 s72, s40 // (op 31380) reg expression
s_mov_b32 s73, s41 // (op 31380) reg expression
s_mov_b32 s74, s24 // (op 31380) reg expression
s_mov_b32 s75, 131072 // (op 31380) call()

s_ashr_i64 s[24:25], s[42:43], 7 // (op 9481) 
s_and_b32 s18, s24, 4294967292 // (op 9481) 

s_ashr_i64 s[20:21], s[42:43], 7 // (op 9480) 
s_mov_b32 s19, s20 // (op 9480) convert

s_cmp_lt_u32 s18, s19 // (op 9480) 




















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ForLoopBottom_KLoopTail_9480 // (op 9480) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ForLoopBottom_KLoopTail_9480 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ForLoopTop_KLoopTail_9480:
s_barrier  // (op 31914) 
s_add_u32 s18, s18, 1 // (op 9482) 

s_lshl_b64 s[20:21], s[26:27], 6 // (op 30566) 
s_lshl_b64 s[24:25], s[20:21], 2 // (op 30566) 

v_lshrrev_b64 v[20:21], 3, s[24:25] // (op 30566) 


v_mov_b64 v[22:23], v[14:15] // (op 30566) 
v_mov_b64 v[24:25], v[22:23] // (op 30566) 
s_add_u32 s19, 12288, s0 // (op 30566) 
s_mov_b32 m0, s19 // (op 30566) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 30566) Load value direct to lds
v_add_co_u32 v24, s[20:21], v24, v20 // (op 30566) least significant half
v_addc_co_u32 v25, s[20:21], 0, v25, s[20:21] // (op 30566) most significant half

s_add_u32 m0, m0, 4096 // (op 30566) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 30566) Load value direct to lds
v_add_co_u32 v24, s[20:21], v24, v20 // (op 30566) least significant half
v_addc_co_u32 v25, s[20:21], 0, v25, s[20:21] // (op 30566) most significant half

s_add_u32 m0, m0, 4096 // (op 30566) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 30566) Load value direct to lds
v_add_co_u32 v24, s[20:21], v24, v20 // (op 30566) least significant half
v_addc_co_u32 v25, s[20:21], 0, v25, s[20:21] // (op 30566) most significant half

s_add_u32 m0, m0, 4096 // (op 30566) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 30566) Load value direct to lds




v_add_co_u32 v14, s[20:21], 64, v14 // (op 30631) least significant half
v_addc_co_u32 v15, s[20:21], 0, v15, s[20:21] // (op 30631) most significant half

s_barrier  // (op 31931) 

s_lshl_b64 s[20:21], s[30:31], 6 // (op 30572) 
s_lshl_b64 s[24:25], s[20:21], 2 // (op 30572) 

v_lshrrev_b64 v[20:21], 3, s[24:25] // (op 30572) 


v_mov_b64 v[22:23], v[18:19] // (op 30572) 
v_mov_b64 v[24:25], v[22:23] // (op 30572) 
s_mov_b32 m0, s1 // (op 30572) reg expression
buffer_load_dwordx4 v24, s[64:67], 0 offen offset: 0 lds // (op 30572) Load value direct to lds
v_add_co_u32 v24, s[20:21], v24, v20 // (op 30572) least significant half
v_addc_co_u32 v25, s[20:21], 0, v25, s[20:21] // (op 30572) most significant half

s_add_u32 m0, m0, 4096 // (op 30572) 
buffer_load_dwordx4 v24, s[64:67], 0 offen offset: 0 lds // (op 30572) Load value direct to lds
v_add_co_u32 v24, s[20:21], v24, v20 // (op 30572) least significant half
v_addc_co_u32 v25, s[20:21], 0, v25, s[20:21] // (op 30572) most significant half

s_add_u32 m0, m0, 4096 // (op 30572) 
buffer_load_dwordx4 v24, s[64:67], 0 offen offset: 0 lds // (op 30572) Load value direct to lds



v_add_co_u32 v18, s[20:21], 64, v18 // (op 30747) least significant half
v_addc_co_u32 v19, s[20:21], 0, v19, s[20:21] // (op 30747) most significant half

s_waitcnt vmcnt(3)
s_barrier  // (op 31921) 



ds_read_b128 v[20:23], v0 offset:12288 // (op 164) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 31937) 



ds_read_b128 v[24:27], v4 // (op 166) Load local data 


v_mov_b32 v28, v16 // (op 168) 
v_mov_b32 v29, v28 // (op 168) reg expression

buffer_load_dwordx4 v[30:33], v29, s[68:71], 0 offen // (op 168) Load value


v_add_u32 v16, 4, v16 // (op 31341) 


v_mov_b32 v28, v17 // (op 170) 
v_mov_b32 v29, v28 // (op 170) reg expression

buffer_load_dwordx4 v[34:37], v29, s[72:75], 0 offen // (op 170) Load value

s_waitcnt vmcnt(1)
v_bfe_u32 v38, v30, 0, 8 // (op 28436) 



v_bfe_u32 v39, v30, 8, 8 // (op 28436) 
v_bfe_u32 v40, v30, 16, 8 // (op 28436) 
v_bfe_u32 v41, v30, 24, 8 // (op 28436) 
v_permlane16_swap_b32 v38, v39 // (op 28436) 
ds_read_b128 v[42:45], v4 offset:1024 // (op 401) Load local data 
v_permlane16_swap_b32 v40, v41 // (op 28436) 



s_nop 1
v_permlane32_swap_b32 v38, v40 // (op 28436) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v39, v41 // (op 28436) 
ds_read_b128 v[46:49], v4 offset:2048 // (op 627) Load local data 




ds_read_b128 v[30:33], v4 offset:3072 // (op 853) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v50, v34, 0, 8 // (op 28498) 



v_bfe_u32 v51, v34, 8, 8 // (op 28498) 
v_bfe_u32 v52, v34, 16, 8 // (op 28498) 
v_bfe_u32 v53, v34, 24, 8 // (op 28498) 
v_permlane16_swap_b32 v50, v51 // (op 28498) 
s_nop 0
v_permlane16_swap_b32 v52, v53 // (op 28498) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v50, v52 // (op 28498) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v51, v53 // (op 28498) 
ds_read_b128 v[54:57], v4 offset:4096 // (op 1079) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:27], a[0:3], v38, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 171) 
s_lshl_b32 s19, s38, 6 // (op 1084) 
v_add_u32 v29, v28, s19 // (op 1084) 


buffer_load_dwordx4 v[34:37], v29, s[72:75], 0 offen // (op 1084) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:23], v[42:45], a[4:7], v38, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 404) 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:23], v[46:49], a[8:11], v38, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 630) 
ds_read_b128 v[58:61], v4 offset:5120 // (op 1305) Load local data 



ds_read_b128 v[62:65], v4 offset:6144 // (op 1531) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:23], v[30:33], a[12:15], v38, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 856) 



ds_read_b128 v[66:69], v4 offset:7168 // (op 1757) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v70, v34, 0, 8 // (op 28512) 



v_bfe_u32 v71, v34, 8, 8 // (op 28512) 
v_bfe_u32 v72, v34, 16, 8 // (op 28512) 
v_bfe_u32 v73, v34, 24, 8 // (op 28512) 
v_permlane16_swap_b32 v70, v71 // (op 28512) 
s_nop 0
v_permlane16_swap_b32 v72, v73 // (op 28512) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v70, v72 // (op 28512) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v71, v73 // (op 28512) 
ds_read_b128 v[74:77], v4 offset:8192 // (op 1983) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:23], v[54:57], a[16:19], v38, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1082) 
s_lshl_b32 s19, s38, 7 // (op 1988) 
v_add_u32 v29, v28, s19 // (op 1988) 


buffer_load_dwordx4 v[34:37], v29, s[72:75], 0 offen // (op 1988) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:23], v[58:61], a[20:23], v38, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1308) 




v_add_u32 v17, 4, v17 // (op 31368) 
ds_read_b128 v[78:81], v4 offset:9216 // (op 2209) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:23], v[62:65], a[24:27], v38, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1534) 



ds_read_b128 v[82:85], v4 offset:10240 // (op 2435) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:23], v[66:69], a[28:31], v38, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1760) 



s_waitcnt vmcnt(0)
v_bfe_u32 v86, v34, 0, 8 // (op 28526) 
v_bfe_u32 v87, v34, 8, 8 // (op 28526) 
v_bfe_u32 v88, v34, 16, 8 // (op 28526) 
v_bfe_u32 v89, v34, 24, 8 // (op 28526) 
v_permlane16_swap_b32 v86, v87 // (op 28526) 
s_nop 0
v_permlane16_swap_b32 v88, v89 // (op 28526) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v86, v88 // (op 28526) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v87, v89 // (op 28526) 
ds_read_b128 v[90:93], v4 offset:11264 // (op 2661) Load local data 




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[20:23], v[74:77], a[32:35], v38, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1986) 
ds_read_b128 v[34:37], v0 offset:13312 // (op 3293) Load local data 



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[20:23], v[78:81], a[36:39], v38, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2212) 
ds_read_b128 v[94:97], v0 offset:14336 // (op 6295) Load local data 



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[20:23], v[82:85], a[40:43], v38, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2438) 
ds_read_b128 v[98:101], v0 offset:15360 // (op 9297) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[20:23], v[90:93], a[44:47], v38, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2664) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[34:37], v[24:27], a[48:51], v39, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3297) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[34:37], v[46:49], a[56:59], v39, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3351) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[34:37], v[30:33], a[60:63], v39, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3378) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[34:37], v[42:45], a[52:55], v39, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3324) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[34:37], v[54:57], a[64:67], v39, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3405) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[34:37], v[62:65], a[72:75], v39, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3459) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[34:37], v[66:69], a[76:79], v39, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3486) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[34:37], v[58:61], a[68:71], v39, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3432) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[34:37], v[74:77], a[80:83], v39, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3513) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[34:37], v[82:85], a[88:91], v39, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3567) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:37], v[90:93], a[92:95], v39, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3594) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[34:37], v[78:81], a[84:87], v39, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3540) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[94:97], v[24:27], a[96:99], v40, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6299) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[94:97], v[46:49], a[104:107], v40, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6353) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:97], v[30:33], a[108:111], v40, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6380) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[94:97], v[42:45], a[100:103], v40, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6326) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[94:97], v[54:57], a[112:115], v40, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6407) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[94:97], v[62:65], a[120:123], v40, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6461) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[94:97], v[66:69], a[124:127], v40, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6488) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[94:97], v[58:61], a[116:119], v40, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6434) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[94:97], v[74:77], a[128:131], v40, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6515) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[94:97], v[82:85], a[136:139], v40, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6569) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[94:97], v[90:93], a[140:143], v40, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6596) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[94:97], v[78:81], a[132:135], v40, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6542) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[98:101], v[24:27], a[144:147], v41, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9301) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[98:101], v[46:49], a[152:155], v41, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9355) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[98:101], v[30:33], a[156:159], v41, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9382) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[98:101], v[42:45], a[148:151], v41, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9328) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[98:101], v[54:57], a[160:163], v41, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9409) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[98:101], v[62:65], a[168:171], v41, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9463) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[98:101], v[66:69], a[172:175], v41, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9490) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[98:101], v[58:61], a[164:167], v41, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9436) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[98:101], v[74:77], a[176:179], v41, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9517) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[98:101], v[82:85], a[184:187], v41, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9571) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[98:101], v[90:93], a[188:191], v41, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9598) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[98:101], v[78:81], a[180:183], v41, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9544) 




s_ashr_i64 s[20:21], s[42:43], 7 // (op 9480) 
s_mov_b32 s19, s20 // (op 9480) convert

s_cmp_lt_u32 s18, s19 // (op 9480) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ForLoopTop_KLoopTail_9480 // (op 9480) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ForLoopTop_KLoopTail_9480 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ForLoopBottom_KLoopTail_9480:




















s_barrier  // (op 32062) 






















v_cmp_eq_f32 vcc, 0.00000, v9 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ConditionalFalse_0_beta_116
v_accvgpr_read v4, a0 // (op 42) 
v_accvgpr_read v5, a1 // (op 42) 
v_accvgpr_read v6, a2 // (op 42) 
v_accvgpr_read v7, a3 // (op 42) 
v_mul_f32 v14, v11, v4 // (op 42) 
v_mul_f32 v15, v11, v5 // (op 42) 
v_mul_f32 v16, v11, v6 // (op 42) 
v_mul_f32 v17, v11, v7 // (op 42) 

v_cvt_f16_f32 v4, v14 // (op 48) 
v_cvt_f16_f32 v5, v15 // (op 48) 
v_cvt_f16_f32 v6, v16 // (op 48) 
v_cvt_f16_f32 v7, v17 // (op 48) 


v_pack_b32_f16 v14, v4, v5 // (op 50) 
v_pack_b32_f16 v15, v6, v7 // (op 50) 

v_mov_b32 v0, v13 // (op 50) reg expression

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 50) Store value



v_accvgpr_read v4, a4 // (op 13738) 
v_accvgpr_read v5, a5 // (op 13738) 
v_accvgpr_read v6, a6 // (op 13738) 
v_accvgpr_read v7, a7 // (op 13738) 
v_mul_f32 v14, v11, v4 // (op 13738) 
v_mul_f32 v15, v11, v5 // (op 13738) 
v_mul_f32 v16, v11, v6 // (op 13738) 
v_mul_f32 v17, v11, v7 // (op 13738) 

v_cvt_f16_f32 v4, v14 // (op 13740) 
v_cvt_f16_f32 v5, v15 // (op 13740) 
v_cvt_f16_f32 v6, v16 // (op 13740) 
v_cvt_f16_f32 v7, v17 // (op 13740) 


v_pack_b32_f16 v14, v4, v5 // (op 13741) 
v_pack_b32_f16 v15, v6, v7 // (op 13741) 

s_lshl_b32 s0, s4, 5 // (op 13741) 
v_add_u32 v0, v13, s0 // (op 13741) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13741) Store value



v_accvgpr_read v4, a8 // (op 13752) 
v_accvgpr_read v5, a9 // (op 13752) 
v_accvgpr_read v6, a10 // (op 13752) 
v_accvgpr_read v7, a11 // (op 13752) 
v_mul_f32 v14, v11, v4 // (op 13752) 
v_mul_f32 v15, v11, v5 // (op 13752) 
v_mul_f32 v16, v11, v6 // (op 13752) 
v_mul_f32 v17, v11, v7 // (op 13752) 

v_cvt_f16_f32 v4, v14 // (op 13754) 
v_cvt_f16_f32 v5, v15 // (op 13754) 
v_cvt_f16_f32 v6, v16 // (op 13754) 
v_cvt_f16_f32 v7, v17 // (op 13754) 


v_pack_b32_f16 v14, v4, v5 // (op 13755) 
v_pack_b32_f16 v15, v6, v7 // (op 13755) 

s_lshl_b32 s0, s4, 6 // (op 13755) 
v_add_u32 v0, v13, s0 // (op 13755) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13755) Store value



v_accvgpr_read v4, a12 // (op 13766) 
v_accvgpr_read v5, a13 // (op 13766) 
v_accvgpr_read v6, a14 // (op 13766) 
v_accvgpr_read v7, a15 // (op 13766) 
v_mul_f32 v14, v11, v4 // (op 13766) 
v_mul_f32 v15, v11, v5 // (op 13766) 
v_mul_f32 v16, v11, v6 // (op 13766) 
v_mul_f32 v17, v11, v7 // (op 13766) 

v_cvt_f16_f32 v4, v14 // (op 13768) 
v_cvt_f16_f32 v5, v15 // (op 13768) 
v_cvt_f16_f32 v6, v16 // (op 13768) 
v_cvt_f16_f32 v7, v17 // (op 13768) 


v_pack_b32_f16 v14, v4, v5 // (op 13769) 
v_pack_b32_f16 v15, v6, v7 // (op 13769) 

s_lshl_b32 s0, s4, 5 // (op 13769) 
s_mul_i32 s1, 3, s0 // (op 13769) 

v_add_u32 v0, v13, s1 // (op 13769) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13769) Store value



v_accvgpr_read v4, a16 // (op 13780) 
v_accvgpr_read v5, a17 // (op 13780) 
v_accvgpr_read v6, a18 // (op 13780) 
v_accvgpr_read v7, a19 // (op 13780) 
v_mul_f32 v14, v11, v4 // (op 13780) 
v_mul_f32 v15, v11, v5 // (op 13780) 
v_mul_f32 v16, v11, v6 // (op 13780) 
v_mul_f32 v17, v11, v7 // (op 13780) 

v_cvt_f16_f32 v4, v14 // (op 13782) 
v_cvt_f16_f32 v5, v15 // (op 13782) 
v_cvt_f16_f32 v6, v16 // (op 13782) 
v_cvt_f16_f32 v7, v17 // (op 13782) 


v_pack_b32_f16 v14, v4, v5 // (op 13783) 
v_pack_b32_f16 v15, v6, v7 // (op 13783) 

s_lshl_b32 s0, s4, 7 // (op 13783) 
v_add_u32 v0, v13, s0 // (op 13783) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13783) Store value



v_accvgpr_read v4, a20 // (op 13794) 
v_accvgpr_read v5, a21 // (op 13794) 
v_accvgpr_read v6, a22 // (op 13794) 
v_accvgpr_read v7, a23 // (op 13794) 
v_mul_f32 v14, v11, v4 // (op 13794) 
v_mul_f32 v15, v11, v5 // (op 13794) 
v_mul_f32 v16, v11, v6 // (op 13794) 
v_mul_f32 v17, v11, v7 // (op 13794) 

v_cvt_f16_f32 v4, v14 // (op 13796) 
v_cvt_f16_f32 v5, v15 // (op 13796) 
v_cvt_f16_f32 v6, v16 // (op 13796) 
v_cvt_f16_f32 v7, v17 // (op 13796) 


v_pack_b32_f16 v14, v4, v5 // (op 13797) 
v_pack_b32_f16 v15, v6, v7 // (op 13797) 

s_lshl_b32 s0, s4, 5 // (op 13797) 
s_mul_i32 s1, 5, s0 // (op 13797) 

v_add_u32 v0, v13, s1 // (op 13797) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13797) Store value



v_accvgpr_read v4, a24 // (op 13808) 
v_accvgpr_read v5, a25 // (op 13808) 
v_accvgpr_read v6, a26 // (op 13808) 
v_accvgpr_read v7, a27 // (op 13808) 
v_mul_f32 v14, v11, v4 // (op 13808) 
v_mul_f32 v15, v11, v5 // (op 13808) 
v_mul_f32 v16, v11, v6 // (op 13808) 
v_mul_f32 v17, v11, v7 // (op 13808) 

v_cvt_f16_f32 v4, v14 // (op 13810) 
v_cvt_f16_f32 v5, v15 // (op 13810) 
v_cvt_f16_f32 v6, v16 // (op 13810) 
v_cvt_f16_f32 v7, v17 // (op 13810) 


v_pack_b32_f16 v14, v4, v5 // (op 13811) 
v_pack_b32_f16 v15, v6, v7 // (op 13811) 

s_lshl_b32 s0, s4, 5 // (op 13811) 
s_mul_i32 s1, 6, s0 // (op 13811) 

v_add_u32 v0, v13, s1 // (op 13811) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13811) Store value



v_accvgpr_read v4, a28 // (op 13822) 
v_accvgpr_read v5, a29 // (op 13822) 
v_accvgpr_read v6, a30 // (op 13822) 
v_accvgpr_read v7, a31 // (op 13822) 
v_mul_f32 v14, v11, v4 // (op 13822) 
v_mul_f32 v15, v11, v5 // (op 13822) 
v_mul_f32 v16, v11, v6 // (op 13822) 
v_mul_f32 v17, v11, v7 // (op 13822) 

v_cvt_f16_f32 v4, v14 // (op 13824) 
v_cvt_f16_f32 v5, v15 // (op 13824) 
v_cvt_f16_f32 v6, v16 // (op 13824) 
v_cvt_f16_f32 v7, v17 // (op 13824) 


v_pack_b32_f16 v14, v4, v5 // (op 13825) 
v_pack_b32_f16 v15, v6, v7 // (op 13825) 

s_lshl_b32 s0, s4, 5 // (op 13825) 
s_mul_i32 s1, 7, s0 // (op 13825) 

v_add_u32 v0, v13, s1 // (op 13825) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13825) Store value



v_accvgpr_read v4, a32 // (op 13836) 
v_accvgpr_read v5, a33 // (op 13836) 
v_accvgpr_read v6, a34 // (op 13836) 
v_accvgpr_read v7, a35 // (op 13836) 
v_mul_f32 v14, v11, v4 // (op 13836) 
v_mul_f32 v15, v11, v5 // (op 13836) 
v_mul_f32 v16, v11, v6 // (op 13836) 
v_mul_f32 v17, v11, v7 // (op 13836) 

v_cvt_f16_f32 v4, v14 // (op 13838) 
v_cvt_f16_f32 v5, v15 // (op 13838) 
v_cvt_f16_f32 v6, v16 // (op 13838) 
v_cvt_f16_f32 v7, v17 // (op 13838) 


v_pack_b32_f16 v14, v4, v5 // (op 13839) 
v_pack_b32_f16 v15, v6, v7 // (op 13839) 

s_lshl_b32 s0, s4, 8 // (op 13839) 
v_add_u32 v0, v13, s0 // (op 13839) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13839) Store value



v_accvgpr_read v4, a36 // (op 13850) 
v_accvgpr_read v5, a37 // (op 13850) 
v_accvgpr_read v6, a38 // (op 13850) 
v_accvgpr_read v7, a39 // (op 13850) 
v_mul_f32 v14, v11, v4 // (op 13850) 
v_mul_f32 v15, v11, v5 // (op 13850) 
v_mul_f32 v16, v11, v6 // (op 13850) 
v_mul_f32 v17, v11, v7 // (op 13850) 

v_cvt_f16_f32 v4, v14 // (op 13852) 
v_cvt_f16_f32 v5, v15 // (op 13852) 
v_cvt_f16_f32 v6, v16 // (op 13852) 
v_cvt_f16_f32 v7, v17 // (op 13852) 


v_pack_b32_f16 v14, v4, v5 // (op 13853) 
v_pack_b32_f16 v15, v6, v7 // (op 13853) 

s_lshl_b32 s0, s4, 5 // (op 13853) 
s_mul_i32 s1, 9, s0 // (op 13853) 

v_add_u32 v0, v13, s1 // (op 13853) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13853) Store value



v_accvgpr_read v4, a40 // (op 13864) 
v_accvgpr_read v5, a41 // (op 13864) 
v_accvgpr_read v6, a42 // (op 13864) 
v_accvgpr_read v7, a43 // (op 13864) 
v_mul_f32 v14, v11, v4 // (op 13864) 
v_mul_f32 v15, v11, v5 // (op 13864) 
v_mul_f32 v16, v11, v6 // (op 13864) 
v_mul_f32 v17, v11, v7 // (op 13864) 

v_cvt_f16_f32 v4, v14 // (op 13866) 
v_cvt_f16_f32 v5, v15 // (op 13866) 
v_cvt_f16_f32 v6, v16 // (op 13866) 
v_cvt_f16_f32 v7, v17 // (op 13866) 


v_pack_b32_f16 v14, v4, v5 // (op 13867) 
v_pack_b32_f16 v15, v6, v7 // (op 13867) 

s_lshl_b32 s0, s4, 5 // (op 13867) 
s_mul_i32 s1, 10, s0 // (op 13867) 

v_add_u32 v0, v13, s1 // (op 13867) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13867) Store value



v_accvgpr_read v4, a44 // (op 13878) 
v_accvgpr_read v5, a45 // (op 13878) 
v_accvgpr_read v6, a46 // (op 13878) 
v_accvgpr_read v7, a47 // (op 13878) 
v_mul_f32 v14, v11, v4 // (op 13878) 
v_mul_f32 v15, v11, v5 // (op 13878) 
v_mul_f32 v16, v11, v6 // (op 13878) 
v_mul_f32 v17, v11, v7 // (op 13878) 

v_cvt_f16_f32 v4, v14 // (op 13880) 
v_cvt_f16_f32 v5, v15 // (op 13880) 
v_cvt_f16_f32 v6, v16 // (op 13880) 
v_cvt_f16_f32 v7, v17 // (op 13880) 


v_pack_b32_f16 v14, v4, v5 // (op 13881) 
v_pack_b32_f16 v15, v6, v7 // (op 13881) 

s_lshl_b32 s0, s4, 5 // (op 13881) 
s_mul_i32 s1, 11, s0 // (op 13881) 

v_add_u32 v0, v13, s1 // (op 13881) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13881) Store value



v_accvgpr_read v4, a48 // (op 13954) 
v_accvgpr_read v5, a49 // (op 13954) 
v_accvgpr_read v6, a50 // (op 13954) 
v_accvgpr_read v7, a51 // (op 13954) 
v_mul_f32 v14, v11, v4 // (op 13954) 
v_mul_f32 v15, v11, v5 // (op 13954) 
v_mul_f32 v16, v11, v6 // (op 13954) 
v_mul_f32 v17, v11, v7 // (op 13954) 

v_cvt_f16_f32 v4, v14 // (op 13956) 
v_cvt_f16_f32 v5, v15 // (op 13956) 
v_cvt_f16_f32 v6, v16 // (op 13956) 
v_cvt_f16_f32 v7, v17 // (op 13956) 


v_pack_b32_f16 v14, v4, v5 // (op 13958) 
v_pack_b32_f16 v15, v6, v7 // (op 13958) 

v_add_u32 v0, 32, v13 // (op 13958) 

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13958) Store value



v_accvgpr_read v4, a52 // (op 13984) 
v_accvgpr_read v5, a53 // (op 13984) 
v_accvgpr_read v6, a54 // (op 13984) 
v_accvgpr_read v7, a55 // (op 13984) 
v_mul_f32 v14, v11, v4 // (op 13984) 
v_mul_f32 v15, v11, v5 // (op 13984) 
v_mul_f32 v16, v11, v6 // (op 13984) 
v_mul_f32 v17, v11, v7 // (op 13984) 

v_cvt_f16_f32 v4, v14 // (op 13986) 
v_cvt_f16_f32 v5, v15 // (op 13986) 
v_cvt_f16_f32 v6, v16 // (op 13986) 
v_cvt_f16_f32 v7, v17 // (op 13986) 


v_pack_b32_f16 v14, v4, v5 // (op 13960) 
v_pack_b32_f16 v15, v6, v7 // (op 13960) 

s_lshl_b32 s0, s4, 5 // (op 13960) 
s_add_u32 s0, s0, 32 // (op 13960) 
v_add_u32 v0, v13, s0 // (op 13960) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13960) Store value



v_accvgpr_read v4, a56 // (op 13990) 
v_accvgpr_read v5, a57 // (op 13990) 
v_accvgpr_read v6, a58 // (op 13990) 
v_accvgpr_read v7, a59 // (op 13990) 
v_mul_f32 v14, v11, v4 // (op 13990) 
v_mul_f32 v15, v11, v5 // (op 13990) 
v_mul_f32 v16, v11, v6 // (op 13990) 
v_mul_f32 v17, v11, v7 // (op 13990) 

v_cvt_f16_f32 v4, v14 // (op 13992) 
v_cvt_f16_f32 v5, v15 // (op 13992) 
v_cvt_f16_f32 v6, v16 // (op 13992) 
v_cvt_f16_f32 v7, v17 // (op 13992) 


v_pack_b32_f16 v14, v4, v5 // (op 13962) 
v_pack_b32_f16 v15, v6, v7 // (op 13962) 

s_lshl_b32 s0, s4, 6 // (op 13962) 
s_add_u32 s0, s0, 32 // (op 13962) 
v_add_u32 v0, v13, s0 // (op 13962) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13962) Store value



v_accvgpr_read v4, a60 // (op 13996) 
v_accvgpr_read v5, a61 // (op 13996) 
v_accvgpr_read v6, a62 // (op 13996) 
v_accvgpr_read v7, a63 // (op 13996) 
v_mul_f32 v14, v11, v4 // (op 13996) 
v_mul_f32 v15, v11, v5 // (op 13996) 
v_mul_f32 v16, v11, v6 // (op 13996) 
v_mul_f32 v17, v11, v7 // (op 13996) 

v_cvt_f16_f32 v4, v14 // (op 13998) 
v_cvt_f16_f32 v5, v15 // (op 13998) 
v_cvt_f16_f32 v6, v16 // (op 13998) 
v_cvt_f16_f32 v7, v17 // (op 13998) 


v_pack_b32_f16 v14, v4, v5 // (op 13964) 
v_pack_b32_f16 v15, v6, v7 // (op 13964) 

s_lshl_b32 s0, s4, 5 // (op 13964) 
s_mul_i32 s1, 3, s0 // (op 13964) 
s_add_u32 s1, s1, 32 // (op 13964) 

v_add_u32 v0, v13, s1 // (op 13964) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13964) Store value



v_accvgpr_read v4, a64 // (op 14002) 
v_accvgpr_read v5, a65 // (op 14002) 
v_accvgpr_read v6, a66 // (op 14002) 
v_accvgpr_read v7, a67 // (op 14002) 
v_mul_f32 v14, v11, v4 // (op 14002) 
v_mul_f32 v15, v11, v5 // (op 14002) 
v_mul_f32 v16, v11, v6 // (op 14002) 
v_mul_f32 v17, v11, v7 // (op 14002) 

v_cvt_f16_f32 v4, v14 // (op 14004) 
v_cvt_f16_f32 v5, v15 // (op 14004) 
v_cvt_f16_f32 v6, v16 // (op 14004) 
v_cvt_f16_f32 v7, v17 // (op 14004) 


v_pack_b32_f16 v14, v4, v5 // (op 13966) 
v_pack_b32_f16 v15, v6, v7 // (op 13966) 

s_lshl_b32 s0, s4, 7 // (op 13966) 
s_add_u32 s0, s0, 32 // (op 13966) 
v_add_u32 v0, v13, s0 // (op 13966) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13966) Store value



v_accvgpr_read v4, a68 // (op 14008) 
v_accvgpr_read v5, a69 // (op 14008) 
v_accvgpr_read v6, a70 // (op 14008) 
v_accvgpr_read v7, a71 // (op 14008) 
v_mul_f32 v14, v11, v4 // (op 14008) 
v_mul_f32 v15, v11, v5 // (op 14008) 
v_mul_f32 v16, v11, v6 // (op 14008) 
v_mul_f32 v17, v11, v7 // (op 14008) 

v_cvt_f16_f32 v4, v14 // (op 14010) 
v_cvt_f16_f32 v5, v15 // (op 14010) 
v_cvt_f16_f32 v6, v16 // (op 14010) 
v_cvt_f16_f32 v7, v17 // (op 14010) 


v_pack_b32_f16 v14, v4, v5 // (op 13968) 
v_pack_b32_f16 v15, v6, v7 // (op 13968) 

s_lshl_b32 s0, s4, 5 // (op 13968) 
s_mul_i32 s1, 5, s0 // (op 13968) 
s_add_u32 s1, s1, 32 // (op 13968) 

v_add_u32 v0, v13, s1 // (op 13968) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13968) Store value



v_accvgpr_read v4, a72 // (op 14014) 
v_accvgpr_read v5, a73 // (op 14014) 
v_accvgpr_read v6, a74 // (op 14014) 
v_accvgpr_read v7, a75 // (op 14014) 
v_mul_f32 v14, v11, v4 // (op 14014) 
v_mul_f32 v15, v11, v5 // (op 14014) 
v_mul_f32 v16, v11, v6 // (op 14014) 
v_mul_f32 v17, v11, v7 // (op 14014) 

v_cvt_f16_f32 v4, v14 // (op 14016) 
v_cvt_f16_f32 v5, v15 // (op 14016) 
v_cvt_f16_f32 v6, v16 // (op 14016) 
v_cvt_f16_f32 v7, v17 // (op 14016) 


v_pack_b32_f16 v14, v4, v5 // (op 13970) 
v_pack_b32_f16 v15, v6, v7 // (op 13970) 

s_lshl_b32 s0, s4, 5 // (op 13970) 
s_mul_i32 s1, 6, s0 // (op 13970) 
s_add_u32 s1, s1, 32 // (op 13970) 

v_add_u32 v0, v13, s1 // (op 13970) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13970) Store value



v_accvgpr_read v4, a76 // (op 14020) 
v_accvgpr_read v5, a77 // (op 14020) 
v_accvgpr_read v6, a78 // (op 14020) 
v_accvgpr_read v7, a79 // (op 14020) 
v_mul_f32 v14, v11, v4 // (op 14020) 
v_mul_f32 v15, v11, v5 // (op 14020) 
v_mul_f32 v16, v11, v6 // (op 14020) 
v_mul_f32 v17, v11, v7 // (op 14020) 

v_cvt_f16_f32 v4, v14 // (op 14022) 
v_cvt_f16_f32 v5, v15 // (op 14022) 
v_cvt_f16_f32 v6, v16 // (op 14022) 
v_cvt_f16_f32 v7, v17 // (op 14022) 


v_pack_b32_f16 v14, v4, v5 // (op 13972) 
v_pack_b32_f16 v15, v6, v7 // (op 13972) 

s_lshl_b32 s0, s4, 5 // (op 13972) 
s_mul_i32 s1, 7, s0 // (op 13972) 
s_add_u32 s1, s1, 32 // (op 13972) 

v_add_u32 v0, v13, s1 // (op 13972) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13972) Store value



v_accvgpr_read v4, a80 // (op 14026) 
v_accvgpr_read v5, a81 // (op 14026) 
v_accvgpr_read v6, a82 // (op 14026) 
v_accvgpr_read v7, a83 // (op 14026) 
v_mul_f32 v14, v11, v4 // (op 14026) 
v_mul_f32 v15, v11, v5 // (op 14026) 
v_mul_f32 v16, v11, v6 // (op 14026) 
v_mul_f32 v17, v11, v7 // (op 14026) 

v_cvt_f16_f32 v4, v14 // (op 14028) 
v_cvt_f16_f32 v5, v15 // (op 14028) 
v_cvt_f16_f32 v6, v16 // (op 14028) 
v_cvt_f16_f32 v7, v17 // (op 14028) 


v_pack_b32_f16 v14, v4, v5 // (op 13974) 
v_pack_b32_f16 v15, v6, v7 // (op 13974) 

s_lshl_b32 s0, s4, 8 // (op 13974) 
s_add_u32 s0, s0, 32 // (op 13974) 
v_add_u32 v0, v13, s0 // (op 13974) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13974) Store value



v_accvgpr_read v4, a84 // (op 14032) 
v_accvgpr_read v5, a85 // (op 14032) 
v_accvgpr_read v6, a86 // (op 14032) 
v_accvgpr_read v7, a87 // (op 14032) 
v_mul_f32 v14, v11, v4 // (op 14032) 
v_mul_f32 v15, v11, v5 // (op 14032) 
v_mul_f32 v16, v11, v6 // (op 14032) 
v_mul_f32 v17, v11, v7 // (op 14032) 

v_cvt_f16_f32 v4, v14 // (op 14034) 
v_cvt_f16_f32 v5, v15 // (op 14034) 
v_cvt_f16_f32 v6, v16 // (op 14034) 
v_cvt_f16_f32 v7, v17 // (op 14034) 


v_pack_b32_f16 v14, v4, v5 // (op 13976) 
v_pack_b32_f16 v15, v6, v7 // (op 13976) 

s_lshl_b32 s0, s4, 5 // (op 13976) 
s_mul_i32 s1, 9, s0 // (op 13976) 
s_add_u32 s1, s1, 32 // (op 13976) 

v_add_u32 v0, v13, s1 // (op 13976) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13976) Store value



v_accvgpr_read v4, a88 // (op 14038) 
v_accvgpr_read v5, a89 // (op 14038) 
v_accvgpr_read v6, a90 // (op 14038) 
v_accvgpr_read v7, a91 // (op 14038) 
v_mul_f32 v14, v11, v4 // (op 14038) 
v_mul_f32 v15, v11, v5 // (op 14038) 
v_mul_f32 v16, v11, v6 // (op 14038) 
v_mul_f32 v17, v11, v7 // (op 14038) 

v_cvt_f16_f32 v4, v14 // (op 14040) 
v_cvt_f16_f32 v5, v15 // (op 14040) 
v_cvt_f16_f32 v6, v16 // (op 14040) 
v_cvt_f16_f32 v7, v17 // (op 14040) 


v_pack_b32_f16 v14, v4, v5 // (op 13978) 
v_pack_b32_f16 v15, v6, v7 // (op 13978) 

s_lshl_b32 s0, s4, 5 // (op 13978) 
s_mul_i32 s1, 10, s0 // (op 13978) 
s_add_u32 s1, s1, 32 // (op 13978) 

v_add_u32 v0, v13, s1 // (op 13978) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13978) Store value



v_accvgpr_read v4, a92 // (op 14044) 
v_accvgpr_read v5, a93 // (op 14044) 
v_accvgpr_read v6, a94 // (op 14044) 
v_accvgpr_read v7, a95 // (op 14044) 
v_mul_f32 v14, v11, v4 // (op 14044) 
v_mul_f32 v15, v11, v5 // (op 14044) 
v_mul_f32 v16, v11, v6 // (op 14044) 
v_mul_f32 v17, v11, v7 // (op 14044) 

v_cvt_f16_f32 v4, v14 // (op 14046) 
v_cvt_f16_f32 v5, v15 // (op 14046) 
v_cvt_f16_f32 v6, v16 // (op 14046) 
v_cvt_f16_f32 v7, v17 // (op 14046) 


v_pack_b32_f16 v14, v4, v5 // (op 13980) 
v_pack_b32_f16 v15, v6, v7 // (op 13980) 

s_lshl_b32 s0, s4, 5 // (op 13980) 
s_mul_i32 s1, 11, s0 // (op 13980) 
s_add_u32 s1, s1, 32 // (op 13980) 

v_add_u32 v0, v13, s1 // (op 13980) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 13980) Store value



v_accvgpr_read v4, a96 // (op 14174) 
v_accvgpr_read v5, a97 // (op 14174) 
v_accvgpr_read v6, a98 // (op 14174) 
v_accvgpr_read v7, a99 // (op 14174) 
v_mul_f32 v14, v11, v4 // (op 14174) 
v_mul_f32 v15, v11, v5 // (op 14174) 
v_mul_f32 v16, v11, v6 // (op 14174) 
v_mul_f32 v17, v11, v7 // (op 14174) 

v_cvt_f16_f32 v4, v14 // (op 14176) 
v_cvt_f16_f32 v5, v15 // (op 14176) 
v_cvt_f16_f32 v6, v16 // (op 14176) 
v_cvt_f16_f32 v7, v17 // (op 14176) 


v_pack_b32_f16 v14, v4, v5 // (op 14178) 
v_pack_b32_f16 v15, v6, v7 // (op 14178) 

v_add_u32 v0, 64, v13 // (op 14178) 

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14178) Store value



v_accvgpr_read v4, a100 // (op 14204) 
v_accvgpr_read v5, a101 // (op 14204) 
v_accvgpr_read v6, a102 // (op 14204) 
v_accvgpr_read v7, a103 // (op 14204) 
v_mul_f32 v14, v11, v4 // (op 14204) 
v_mul_f32 v15, v11, v5 // (op 14204) 
v_mul_f32 v16, v11, v6 // (op 14204) 
v_mul_f32 v17, v11, v7 // (op 14204) 

v_cvt_f16_f32 v4, v14 // (op 14206) 
v_cvt_f16_f32 v5, v15 // (op 14206) 
v_cvt_f16_f32 v6, v16 // (op 14206) 
v_cvt_f16_f32 v7, v17 // (op 14206) 


v_pack_b32_f16 v14, v4, v5 // (op 14180) 
v_pack_b32_f16 v15, v6, v7 // (op 14180) 

s_lshl_b32 s0, s4, 5 // (op 14180) 
s_add_u32 s0, s0, 64 // (op 14180) 
v_add_u32 v0, v13, s0 // (op 14180) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14180) Store value



v_accvgpr_read v4, a104 // (op 14210) 
v_accvgpr_read v5, a105 // (op 14210) 
v_accvgpr_read v6, a106 // (op 14210) 
v_accvgpr_read v7, a107 // (op 14210) 
v_mul_f32 v14, v11, v4 // (op 14210) 
v_mul_f32 v15, v11, v5 // (op 14210) 
v_mul_f32 v16, v11, v6 // (op 14210) 
v_mul_f32 v17, v11, v7 // (op 14210) 

v_cvt_f16_f32 v4, v14 // (op 14212) 
v_cvt_f16_f32 v5, v15 // (op 14212) 
v_cvt_f16_f32 v6, v16 // (op 14212) 
v_cvt_f16_f32 v7, v17 // (op 14212) 


v_pack_b32_f16 v14, v4, v5 // (op 14182) 
v_pack_b32_f16 v15, v6, v7 // (op 14182) 

s_lshl_b32 s0, s4, 6 // (op 14182) 
s_add_u32 s0, s0, 64 // (op 14182) 
v_add_u32 v0, v13, s0 // (op 14182) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14182) Store value



v_accvgpr_read v4, a108 // (op 14216) 
v_accvgpr_read v5, a109 // (op 14216) 
v_accvgpr_read v6, a110 // (op 14216) 
v_accvgpr_read v7, a111 // (op 14216) 
v_mul_f32 v14, v11, v4 // (op 14216) 
v_mul_f32 v15, v11, v5 // (op 14216) 
v_mul_f32 v16, v11, v6 // (op 14216) 
v_mul_f32 v17, v11, v7 // (op 14216) 

v_cvt_f16_f32 v4, v14 // (op 14218) 
v_cvt_f16_f32 v5, v15 // (op 14218) 
v_cvt_f16_f32 v6, v16 // (op 14218) 
v_cvt_f16_f32 v7, v17 // (op 14218) 


v_pack_b32_f16 v14, v4, v5 // (op 14184) 
v_pack_b32_f16 v15, v6, v7 // (op 14184) 

s_lshl_b32 s0, s4, 5 // (op 14184) 
s_mul_i32 s1, 3, s0 // (op 14184) 
s_add_u32 s1, s1, 64 // (op 14184) 

v_add_u32 v0, v13, s1 // (op 14184) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14184) Store value



v_accvgpr_read v4, a112 // (op 14222) 
v_accvgpr_read v5, a113 // (op 14222) 
v_accvgpr_read v6, a114 // (op 14222) 
v_accvgpr_read v7, a115 // (op 14222) 
v_mul_f32 v14, v11, v4 // (op 14222) 
v_mul_f32 v15, v11, v5 // (op 14222) 
v_mul_f32 v16, v11, v6 // (op 14222) 
v_mul_f32 v17, v11, v7 // (op 14222) 

v_cvt_f16_f32 v4, v14 // (op 14224) 
v_cvt_f16_f32 v5, v15 // (op 14224) 
v_cvt_f16_f32 v6, v16 // (op 14224) 
v_cvt_f16_f32 v7, v17 // (op 14224) 


v_pack_b32_f16 v14, v4, v5 // (op 14186) 
v_pack_b32_f16 v15, v6, v7 // (op 14186) 

s_lshl_b32 s0, s4, 7 // (op 14186) 
s_add_u32 s0, s0, 64 // (op 14186) 
v_add_u32 v0, v13, s0 // (op 14186) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14186) Store value



v_accvgpr_read v4, a116 // (op 14228) 
v_accvgpr_read v5, a117 // (op 14228) 
v_accvgpr_read v6, a118 // (op 14228) 
v_accvgpr_read v7, a119 // (op 14228) 
v_mul_f32 v14, v11, v4 // (op 14228) 
v_mul_f32 v15, v11, v5 // (op 14228) 
v_mul_f32 v16, v11, v6 // (op 14228) 
v_mul_f32 v17, v11, v7 // (op 14228) 

v_cvt_f16_f32 v4, v14 // (op 14230) 
v_cvt_f16_f32 v5, v15 // (op 14230) 
v_cvt_f16_f32 v6, v16 // (op 14230) 
v_cvt_f16_f32 v7, v17 // (op 14230) 


v_pack_b32_f16 v14, v4, v5 // (op 14188) 
v_pack_b32_f16 v15, v6, v7 // (op 14188) 

s_lshl_b32 s0, s4, 5 // (op 14188) 
s_mul_i32 s1, 5, s0 // (op 14188) 
s_add_u32 s1, s1, 64 // (op 14188) 

v_add_u32 v0, v13, s1 // (op 14188) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14188) Store value



v_accvgpr_read v4, a120 // (op 14234) 
v_accvgpr_read v5, a121 // (op 14234) 
v_accvgpr_read v6, a122 // (op 14234) 
v_accvgpr_read v7, a123 // (op 14234) 
v_mul_f32 v14, v11, v4 // (op 14234) 
v_mul_f32 v15, v11, v5 // (op 14234) 
v_mul_f32 v16, v11, v6 // (op 14234) 
v_mul_f32 v17, v11, v7 // (op 14234) 

v_cvt_f16_f32 v4, v14 // (op 14236) 
v_cvt_f16_f32 v5, v15 // (op 14236) 
v_cvt_f16_f32 v6, v16 // (op 14236) 
v_cvt_f16_f32 v7, v17 // (op 14236) 


v_pack_b32_f16 v14, v4, v5 // (op 14190) 
v_pack_b32_f16 v15, v6, v7 // (op 14190) 

s_lshl_b32 s0, s4, 5 // (op 14190) 
s_mul_i32 s1, 6, s0 // (op 14190) 
s_add_u32 s1, s1, 64 // (op 14190) 

v_add_u32 v0, v13, s1 // (op 14190) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14190) Store value



v_accvgpr_read v4, a124 // (op 14240) 
v_accvgpr_read v5, a125 // (op 14240) 
v_accvgpr_read v6, a126 // (op 14240) 
v_accvgpr_read v7, a127 // (op 14240) 
v_mul_f32 v14, v11, v4 // (op 14240) 
v_mul_f32 v15, v11, v5 // (op 14240) 
v_mul_f32 v16, v11, v6 // (op 14240) 
v_mul_f32 v17, v11, v7 // (op 14240) 

v_cvt_f16_f32 v4, v14 // (op 14242) 
v_cvt_f16_f32 v5, v15 // (op 14242) 
v_cvt_f16_f32 v6, v16 // (op 14242) 
v_cvt_f16_f32 v7, v17 // (op 14242) 


v_pack_b32_f16 v14, v4, v5 // (op 14192) 
v_pack_b32_f16 v15, v6, v7 // (op 14192) 

s_lshl_b32 s0, s4, 5 // (op 14192) 
s_mul_i32 s1, 7, s0 // (op 14192) 
s_add_u32 s1, s1, 64 // (op 14192) 

v_add_u32 v0, v13, s1 // (op 14192) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14192) Store value



v_accvgpr_read v4, a128 // (op 14246) 
v_accvgpr_read v5, a129 // (op 14246) 
v_accvgpr_read v6, a130 // (op 14246) 
v_accvgpr_read v7, a131 // (op 14246) 
v_mul_f32 v14, v11, v4 // (op 14246) 
v_mul_f32 v15, v11, v5 // (op 14246) 
v_mul_f32 v16, v11, v6 // (op 14246) 
v_mul_f32 v17, v11, v7 // (op 14246) 

v_cvt_f16_f32 v4, v14 // (op 14248) 
v_cvt_f16_f32 v5, v15 // (op 14248) 
v_cvt_f16_f32 v6, v16 // (op 14248) 
v_cvt_f16_f32 v7, v17 // (op 14248) 


v_pack_b32_f16 v14, v4, v5 // (op 14194) 
v_pack_b32_f16 v15, v6, v7 // (op 14194) 

s_lshl_b32 s0, s4, 8 // (op 14194) 
s_add_u32 s0, s0, 64 // (op 14194) 
v_add_u32 v0, v13, s0 // (op 14194) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14194) Store value



v_accvgpr_read v4, a132 // (op 14252) 
v_accvgpr_read v5, a133 // (op 14252) 
v_accvgpr_read v6, a134 // (op 14252) 
v_accvgpr_read v7, a135 // (op 14252) 
v_mul_f32 v14, v11, v4 // (op 14252) 
v_mul_f32 v15, v11, v5 // (op 14252) 
v_mul_f32 v16, v11, v6 // (op 14252) 
v_mul_f32 v17, v11, v7 // (op 14252) 

v_cvt_f16_f32 v4, v14 // (op 14254) 
v_cvt_f16_f32 v5, v15 // (op 14254) 
v_cvt_f16_f32 v6, v16 // (op 14254) 
v_cvt_f16_f32 v7, v17 // (op 14254) 


v_pack_b32_f16 v14, v4, v5 // (op 14196) 
v_pack_b32_f16 v15, v6, v7 // (op 14196) 

s_lshl_b32 s0, s4, 5 // (op 14196) 
s_mul_i32 s1, 9, s0 // (op 14196) 
s_add_u32 s1, s1, 64 // (op 14196) 

v_add_u32 v0, v13, s1 // (op 14196) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14196) Store value



v_accvgpr_read v4, a136 // (op 14258) 
v_accvgpr_read v5, a137 // (op 14258) 
v_accvgpr_read v6, a138 // (op 14258) 
v_accvgpr_read v7, a139 // (op 14258) 
v_mul_f32 v14, v11, v4 // (op 14258) 
v_mul_f32 v15, v11, v5 // (op 14258) 
v_mul_f32 v16, v11, v6 // (op 14258) 
v_mul_f32 v17, v11, v7 // (op 14258) 

v_cvt_f16_f32 v4, v14 // (op 14260) 
v_cvt_f16_f32 v5, v15 // (op 14260) 
v_cvt_f16_f32 v6, v16 // (op 14260) 
v_cvt_f16_f32 v7, v17 // (op 14260) 


v_pack_b32_f16 v14, v4, v5 // (op 14198) 
v_pack_b32_f16 v15, v6, v7 // (op 14198) 

s_lshl_b32 s0, s4, 5 // (op 14198) 
s_mul_i32 s1, 10, s0 // (op 14198) 
s_add_u32 s1, s1, 64 // (op 14198) 

v_add_u32 v0, v13, s1 // (op 14198) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14198) Store value



v_accvgpr_read v4, a140 // (op 14264) 
v_accvgpr_read v5, a141 // (op 14264) 
v_accvgpr_read v6, a142 // (op 14264) 
v_accvgpr_read v7, a143 // (op 14264) 
v_mul_f32 v14, v11, v4 // (op 14264) 
v_mul_f32 v15, v11, v5 // (op 14264) 
v_mul_f32 v16, v11, v6 // (op 14264) 
v_mul_f32 v17, v11, v7 // (op 14264) 

v_cvt_f16_f32 v4, v14 // (op 14266) 
v_cvt_f16_f32 v5, v15 // (op 14266) 
v_cvt_f16_f32 v6, v16 // (op 14266) 
v_cvt_f16_f32 v7, v17 // (op 14266) 


v_pack_b32_f16 v14, v4, v5 // (op 14200) 
v_pack_b32_f16 v15, v6, v7 // (op 14200) 

s_lshl_b32 s0, s4, 5 // (op 14200) 
s_mul_i32 s1, 11, s0 // (op 14200) 
s_add_u32 s1, s1, 64 // (op 14200) 

v_add_u32 v0, v13, s1 // (op 14200) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14200) Store value



v_accvgpr_read v4, a144 // (op 14394) 
v_accvgpr_read v5, a145 // (op 14394) 
v_accvgpr_read v6, a146 // (op 14394) 
v_accvgpr_read v7, a147 // (op 14394) 
v_mul_f32 v14, v11, v4 // (op 14394) 
v_mul_f32 v15, v11, v5 // (op 14394) 
v_mul_f32 v16, v11, v6 // (op 14394) 
v_mul_f32 v17, v11, v7 // (op 14394) 

v_cvt_f16_f32 v4, v14 // (op 14396) 
v_cvt_f16_f32 v5, v15 // (op 14396) 
v_cvt_f16_f32 v6, v16 // (op 14396) 
v_cvt_f16_f32 v7, v17 // (op 14396) 


v_pack_b32_f16 v14, v4, v5 // (op 14398) 
v_pack_b32_f16 v15, v6, v7 // (op 14398) 

v_add_u32 v0, 96, v13 // (op 14398) 

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14398) Store value



v_accvgpr_read v4, a148 // (op 14424) 
v_accvgpr_read v5, a149 // (op 14424) 
v_accvgpr_read v6, a150 // (op 14424) 
v_accvgpr_read v7, a151 // (op 14424) 
v_mul_f32 v14, v11, v4 // (op 14424) 
v_mul_f32 v15, v11, v5 // (op 14424) 
v_mul_f32 v16, v11, v6 // (op 14424) 
v_mul_f32 v17, v11, v7 // (op 14424) 

v_cvt_f16_f32 v4, v14 // (op 14426) 
v_cvt_f16_f32 v5, v15 // (op 14426) 
v_cvt_f16_f32 v6, v16 // (op 14426) 
v_cvt_f16_f32 v7, v17 // (op 14426) 


v_pack_b32_f16 v14, v4, v5 // (op 14400) 
v_pack_b32_f16 v15, v6, v7 // (op 14400) 

s_lshl_b32 s0, s4, 5 // (op 14400) 
s_add_u32 s0, s0, 96 // (op 14400) 
v_add_u32 v0, v13, s0 // (op 14400) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14400) Store value



v_accvgpr_read v4, a152 // (op 14430) 
v_accvgpr_read v5, a153 // (op 14430) 
v_accvgpr_read v6, a154 // (op 14430) 
v_accvgpr_read v7, a155 // (op 14430) 
v_mul_f32 v14, v11, v4 // (op 14430) 
v_mul_f32 v15, v11, v5 // (op 14430) 
v_mul_f32 v16, v11, v6 // (op 14430) 
v_mul_f32 v17, v11, v7 // (op 14430) 

v_cvt_f16_f32 v4, v14 // (op 14432) 
v_cvt_f16_f32 v5, v15 // (op 14432) 
v_cvt_f16_f32 v6, v16 // (op 14432) 
v_cvt_f16_f32 v7, v17 // (op 14432) 


v_pack_b32_f16 v14, v4, v5 // (op 14402) 
v_pack_b32_f16 v15, v6, v7 // (op 14402) 

s_lshl_b32 s0, s4, 6 // (op 14402) 
s_add_u32 s0, s0, 96 // (op 14402) 
v_add_u32 v0, v13, s0 // (op 14402) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14402) Store value



v_accvgpr_read v4, a156 // (op 14436) 
v_accvgpr_read v5, a157 // (op 14436) 
v_accvgpr_read v6, a158 // (op 14436) 
v_accvgpr_read v7, a159 // (op 14436) 
v_mul_f32 v14, v11, v4 // (op 14436) 
v_mul_f32 v15, v11, v5 // (op 14436) 
v_mul_f32 v16, v11, v6 // (op 14436) 
v_mul_f32 v17, v11, v7 // (op 14436) 

v_cvt_f16_f32 v4, v14 // (op 14438) 
v_cvt_f16_f32 v5, v15 // (op 14438) 
v_cvt_f16_f32 v6, v16 // (op 14438) 
v_cvt_f16_f32 v7, v17 // (op 14438) 


v_pack_b32_f16 v14, v4, v5 // (op 14404) 
v_pack_b32_f16 v15, v6, v7 // (op 14404) 

s_lshl_b32 s0, s4, 5 // (op 14404) 
s_mul_i32 s1, 3, s0 // (op 14404) 
s_add_u32 s1, s1, 96 // (op 14404) 

v_add_u32 v0, v13, s1 // (op 14404) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14404) Store value



v_accvgpr_read v4, a160 // (op 14442) 
v_accvgpr_read v5, a161 // (op 14442) 
v_accvgpr_read v6, a162 // (op 14442) 
v_accvgpr_read v7, a163 // (op 14442) 
v_mul_f32 v14, v11, v4 // (op 14442) 
v_mul_f32 v15, v11, v5 // (op 14442) 
v_mul_f32 v16, v11, v6 // (op 14442) 
v_mul_f32 v17, v11, v7 // (op 14442) 

v_cvt_f16_f32 v4, v14 // (op 14444) 
v_cvt_f16_f32 v5, v15 // (op 14444) 
v_cvt_f16_f32 v6, v16 // (op 14444) 
v_cvt_f16_f32 v7, v17 // (op 14444) 


v_pack_b32_f16 v14, v4, v5 // (op 14406) 
v_pack_b32_f16 v15, v6, v7 // (op 14406) 

s_lshl_b32 s0, s4, 7 // (op 14406) 
s_add_u32 s0, s0, 96 // (op 14406) 
v_add_u32 v0, v13, s0 // (op 14406) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14406) Store value



v_accvgpr_read v4, a164 // (op 14448) 
v_accvgpr_read v5, a165 // (op 14448) 
v_accvgpr_read v6, a166 // (op 14448) 
v_accvgpr_read v7, a167 // (op 14448) 
v_mul_f32 v14, v11, v4 // (op 14448) 
v_mul_f32 v15, v11, v5 // (op 14448) 
v_mul_f32 v16, v11, v6 // (op 14448) 
v_mul_f32 v17, v11, v7 // (op 14448) 

v_cvt_f16_f32 v4, v14 // (op 14450) 
v_cvt_f16_f32 v5, v15 // (op 14450) 
v_cvt_f16_f32 v6, v16 // (op 14450) 
v_cvt_f16_f32 v7, v17 // (op 14450) 


v_pack_b32_f16 v14, v4, v5 // (op 14408) 
v_pack_b32_f16 v15, v6, v7 // (op 14408) 

s_lshl_b32 s0, s4, 5 // (op 14408) 
s_mul_i32 s1, 5, s0 // (op 14408) 
s_add_u32 s1, s1, 96 // (op 14408) 

v_add_u32 v0, v13, s1 // (op 14408) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14408) Store value



v_accvgpr_read v4, a168 // (op 14454) 
v_accvgpr_read v5, a169 // (op 14454) 
v_accvgpr_read v6, a170 // (op 14454) 
v_accvgpr_read v7, a171 // (op 14454) 
v_mul_f32 v14, v11, v4 // (op 14454) 
v_mul_f32 v15, v11, v5 // (op 14454) 
v_mul_f32 v16, v11, v6 // (op 14454) 
v_mul_f32 v17, v11, v7 // (op 14454) 

v_cvt_f16_f32 v4, v14 // (op 14456) 
v_cvt_f16_f32 v5, v15 // (op 14456) 
v_cvt_f16_f32 v6, v16 // (op 14456) 
v_cvt_f16_f32 v7, v17 // (op 14456) 


v_pack_b32_f16 v14, v4, v5 // (op 14410) 
v_pack_b32_f16 v15, v6, v7 // (op 14410) 

s_lshl_b32 s0, s4, 5 // (op 14410) 
s_mul_i32 s1, 6, s0 // (op 14410) 
s_add_u32 s1, s1, 96 // (op 14410) 

v_add_u32 v0, v13, s1 // (op 14410) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14410) Store value



v_accvgpr_read v4, a172 // (op 14460) 
v_accvgpr_read v5, a173 // (op 14460) 
v_accvgpr_read v6, a174 // (op 14460) 
v_accvgpr_read v7, a175 // (op 14460) 
v_mul_f32 v14, v11, v4 // (op 14460) 
v_mul_f32 v15, v11, v5 // (op 14460) 
v_mul_f32 v16, v11, v6 // (op 14460) 
v_mul_f32 v17, v11, v7 // (op 14460) 

v_cvt_f16_f32 v4, v14 // (op 14462) 
v_cvt_f16_f32 v5, v15 // (op 14462) 
v_cvt_f16_f32 v6, v16 // (op 14462) 
v_cvt_f16_f32 v7, v17 // (op 14462) 


v_pack_b32_f16 v14, v4, v5 // (op 14412) 
v_pack_b32_f16 v15, v6, v7 // (op 14412) 

s_lshl_b32 s0, s4, 5 // (op 14412) 
s_mul_i32 s1, 7, s0 // (op 14412) 
s_add_u32 s1, s1, 96 // (op 14412) 

v_add_u32 v0, v13, s1 // (op 14412) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14412) Store value



v_accvgpr_read v4, a176 // (op 14466) 
v_accvgpr_read v5, a177 // (op 14466) 
v_accvgpr_read v6, a178 // (op 14466) 
v_accvgpr_read v7, a179 // (op 14466) 
v_mul_f32 v14, v11, v4 // (op 14466) 
v_mul_f32 v15, v11, v5 // (op 14466) 
v_mul_f32 v16, v11, v6 // (op 14466) 
v_mul_f32 v17, v11, v7 // (op 14466) 

v_cvt_f16_f32 v4, v14 // (op 14468) 
v_cvt_f16_f32 v5, v15 // (op 14468) 
v_cvt_f16_f32 v6, v16 // (op 14468) 
v_cvt_f16_f32 v7, v17 // (op 14468) 


v_pack_b32_f16 v14, v4, v5 // (op 14414) 
v_pack_b32_f16 v15, v6, v7 // (op 14414) 

s_lshl_b32 s0, s4, 8 // (op 14414) 
s_add_u32 s0, s0, 96 // (op 14414) 
v_add_u32 v0, v13, s0 // (op 14414) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14414) Store value



v_accvgpr_read v4, a180 // (op 14472) 
v_accvgpr_read v5, a181 // (op 14472) 
v_accvgpr_read v6, a182 // (op 14472) 
v_accvgpr_read v7, a183 // (op 14472) 
v_mul_f32 v14, v11, v4 // (op 14472) 
v_mul_f32 v15, v11, v5 // (op 14472) 
v_mul_f32 v16, v11, v6 // (op 14472) 
v_mul_f32 v17, v11, v7 // (op 14472) 

v_cvt_f16_f32 v4, v14 // (op 14474) 
v_cvt_f16_f32 v5, v15 // (op 14474) 
v_cvt_f16_f32 v6, v16 // (op 14474) 
v_cvt_f16_f32 v7, v17 // (op 14474) 


v_pack_b32_f16 v14, v4, v5 // (op 14416) 
v_pack_b32_f16 v15, v6, v7 // (op 14416) 

s_lshl_b32 s0, s4, 5 // (op 14416) 
s_mul_i32 s1, 9, s0 // (op 14416) 
s_add_u32 s1, s1, 96 // (op 14416) 

v_add_u32 v0, v13, s1 // (op 14416) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14416) Store value



v_accvgpr_read v4, a184 // (op 14478) 
v_accvgpr_read v5, a185 // (op 14478) 
v_accvgpr_read v6, a186 // (op 14478) 
v_accvgpr_read v7, a187 // (op 14478) 
v_mul_f32 v14, v11, v4 // (op 14478) 
v_mul_f32 v15, v11, v5 // (op 14478) 
v_mul_f32 v16, v11, v6 // (op 14478) 
v_mul_f32 v17, v11, v7 // (op 14478) 

v_cvt_f16_f32 v4, v14 // (op 14480) 
v_cvt_f16_f32 v5, v15 // (op 14480) 
v_cvt_f16_f32 v6, v16 // (op 14480) 
v_cvt_f16_f32 v7, v17 // (op 14480) 


v_pack_b32_f16 v14, v4, v5 // (op 14418) 
v_pack_b32_f16 v15, v6, v7 // (op 14418) 

s_lshl_b32 s0, s4, 5 // (op 14418) 
s_mul_i32 s1, 10, s0 // (op 14418) 
s_add_u32 s1, s1, 96 // (op 14418) 

v_add_u32 v0, v13, s1 // (op 14418) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14418) Store value



v_accvgpr_read v4, a188 // (op 14484) 
v_accvgpr_read v5, a189 // (op 14484) 
v_accvgpr_read v6, a190 // (op 14484) 
v_accvgpr_read v7, a191 // (op 14484) 
v_mul_f32 v14, v11, v4 // (op 14484) 
v_mul_f32 v15, v11, v5 // (op 14484) 
v_mul_f32 v16, v11, v6 // (op 14484) 
v_mul_f32 v17, v11, v7 // (op 14484) 

v_cvt_f16_f32 v4, v14 // (op 14486) 
v_cvt_f16_f32 v5, v15 // (op 14486) 
v_cvt_f16_f32 v6, v16 // (op 14486) 
v_cvt_f16_f32 v7, v17 // (op 14486) 


v_pack_b32_f16 v14, v4, v5 // (op 14420) 
v_pack_b32_f16 v15, v6, v7 // (op 14420) 

s_lshl_b32 s0, s4, 5 // (op 14420) 
s_mul_i32 s1, 11, s0 // (op 14420) 
s_add_u32 s1, s1, 96 // (op 14420) 

v_add_u32 v0, v13, s1 // (op 14420) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 14420) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ConditionalBottom_0_beta_116
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ConditionalFalse_0_beta_116:


v_add_u32 v0, 96, v12 // (op 15576) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15576) Load value

v_accvgpr_read v14, a144 // (op 15547) 
v_accvgpr_read v15, a145 // (op 15547) 
v_accvgpr_read v16, a146 // (op 15547) 
v_accvgpr_read v17, a147 // (op 15547) 
v_mul_f32 v18, v11, v14 // (op 15547) 
v_mul_f32 v19, v11, v15 // (op 15547) 
v_mul_f32 v20, v11, v16 // (op 15547) 
v_mul_f32 v21, v11, v17 // (op 15547) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15575) 
v_lshrrev_b32 v7, 16, v4 // (op 15575) 
v_cvt_f32_f16 v7, v7 // (op 15575) 
v_mul_f32 v14, v9, v6 // (op 15575) 
v_mul_f32 v15, v9, v7 // (op 15575) 
v_cvt_f32_f16 v6, v5 // (op 15575) 
v_lshrrev_b32 v7, 16, v5 // (op 15575) 
v_cvt_f32_f16 v7, v7 // (op 15575) 
v_mul_f32 v16, v9, v6 // (op 15575) 
v_mul_f32 v17, v9, v7 // (op 15575) 


v_add_f32 v4, v14, v18 // (op 15548) 
v_add_f32 v5, v15, v19 // (op 15548) 
v_add_f32 v6, v16, v20 // (op 15548) 
v_add_f32 v7, v17, v21 // (op 15548) 


v_cvt_f16_f32 v14, v4 // (op 15549) 
v_cvt_f16_f32 v15, v5 // (op 15549) 
v_cvt_f16_f32 v16, v6 // (op 15549) 
v_cvt_f16_f32 v17, v7 // (op 15549) 


v_pack_b32_f16 v4, v14, v15 // (op 15551) 
v_pack_b32_f16 v5, v16, v17 // (op 15551) 

v_add_u32 v0, 96, v13 // (op 15551) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15551) Store value





s_lshl_b32 s0, s2, 5 // (op 15581) 
s_add_u32 s0, s0, 96 // (op 15581) 
v_add_u32 v0, v12, s0 // (op 15581) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15581) Load value

v_accvgpr_read v14, a148 // (op 15633) 
v_accvgpr_read v15, a149 // (op 15633) 
v_accvgpr_read v16, a150 // (op 15633) 
v_accvgpr_read v17, a151 // (op 15633) 
v_mul_f32 v18, v11, v14 // (op 15633) 
v_mul_f32 v19, v11, v15 // (op 15633) 
v_mul_f32 v20, v11, v16 // (op 15633) 
v_mul_f32 v21, v11, v17 // (op 15633) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15578) 
v_lshrrev_b32 v7, 16, v4 // (op 15578) 
v_cvt_f32_f16 v7, v7 // (op 15578) 
v_mul_f32 v14, v9, v6 // (op 15578) 
v_mul_f32 v15, v9, v7 // (op 15578) 
v_cvt_f32_f16 v6, v5 // (op 15578) 
v_lshrrev_b32 v7, 16, v5 // (op 15578) 
v_cvt_f32_f16 v7, v7 // (op 15578) 
v_mul_f32 v16, v9, v6 // (op 15578) 
v_mul_f32 v17, v9, v7 // (op 15578) 


v_add_f32 v4, v14, v18 // (op 15579) 
v_add_f32 v5, v15, v19 // (op 15579) 
v_add_f32 v6, v16, v20 // (op 15579) 
v_add_f32 v7, v17, v21 // (op 15579) 


v_cvt_f16_f32 v14, v4 // (op 15580) 
v_cvt_f16_f32 v15, v5 // (op 15580) 
v_cvt_f16_f32 v16, v6 // (op 15580) 
v_cvt_f16_f32 v17, v7 // (op 15580) 


v_pack_b32_f16 v4, v14, v15 // (op 15553) 
v_pack_b32_f16 v5, v16, v17 // (op 15553) 

s_lshl_b32 s0, s4, 5 // (op 15553) 
s_add_u32 s0, s0, 96 // (op 15553) 
v_add_u32 v0, v13, s0 // (op 15553) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15553) Store value





s_lshl_b32 s0, s2, 6 // (op 15586) 
s_add_u32 s0, s0, 96 // (op 15586) 
v_add_u32 v0, v12, s0 // (op 15586) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15586) Load value

v_accvgpr_read v14, a152 // (op 15635) 
v_accvgpr_read v15, a153 // (op 15635) 
v_accvgpr_read v16, a154 // (op 15635) 
v_accvgpr_read v17, a155 // (op 15635) 
v_mul_f32 v18, v11, v14 // (op 15635) 
v_mul_f32 v19, v11, v15 // (op 15635) 
v_mul_f32 v20, v11, v16 // (op 15635) 
v_mul_f32 v21, v11, v17 // (op 15635) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15583) 
v_lshrrev_b32 v7, 16, v4 // (op 15583) 
v_cvt_f32_f16 v7, v7 // (op 15583) 
v_mul_f32 v14, v9, v6 // (op 15583) 
v_mul_f32 v15, v9, v7 // (op 15583) 
v_cvt_f32_f16 v6, v5 // (op 15583) 
v_lshrrev_b32 v7, 16, v5 // (op 15583) 
v_cvt_f32_f16 v7, v7 // (op 15583) 
v_mul_f32 v16, v9, v6 // (op 15583) 
v_mul_f32 v17, v9, v7 // (op 15583) 


v_add_f32 v4, v14, v18 // (op 15584) 
v_add_f32 v5, v15, v19 // (op 15584) 
v_add_f32 v6, v16, v20 // (op 15584) 
v_add_f32 v7, v17, v21 // (op 15584) 


v_cvt_f16_f32 v14, v4 // (op 15585) 
v_cvt_f16_f32 v15, v5 // (op 15585) 
v_cvt_f16_f32 v16, v6 // (op 15585) 
v_cvt_f16_f32 v17, v7 // (op 15585) 


v_pack_b32_f16 v4, v14, v15 // (op 15555) 
v_pack_b32_f16 v5, v16, v17 // (op 15555) 

s_lshl_b32 s0, s4, 6 // (op 15555) 
s_add_u32 s0, s0, 96 // (op 15555) 
v_add_u32 v0, v13, s0 // (op 15555) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15555) Store value





s_lshl_b32 s0, s2, 5 // (op 15591) 
s_mul_i32 s1, 3, s0 // (op 15591) 
s_add_u32 s1, s1, 96 // (op 15591) 

v_add_u32 v0, v12, s1 // (op 15591) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15591) Load value

v_accvgpr_read v14, a156 // (op 15637) 
v_accvgpr_read v15, a157 // (op 15637) 
v_accvgpr_read v16, a158 // (op 15637) 
v_accvgpr_read v17, a159 // (op 15637) 
v_mul_f32 v18, v11, v14 // (op 15637) 
v_mul_f32 v19, v11, v15 // (op 15637) 
v_mul_f32 v20, v11, v16 // (op 15637) 
v_mul_f32 v21, v11, v17 // (op 15637) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15588) 
v_lshrrev_b32 v7, 16, v4 // (op 15588) 
v_cvt_f32_f16 v7, v7 // (op 15588) 
v_mul_f32 v14, v9, v6 // (op 15588) 
v_mul_f32 v15, v9, v7 // (op 15588) 
v_cvt_f32_f16 v6, v5 // (op 15588) 
v_lshrrev_b32 v7, 16, v5 // (op 15588) 
v_cvt_f32_f16 v7, v7 // (op 15588) 
v_mul_f32 v16, v9, v6 // (op 15588) 
v_mul_f32 v17, v9, v7 // (op 15588) 


v_add_f32 v4, v14, v18 // (op 15589) 
v_add_f32 v5, v15, v19 // (op 15589) 
v_add_f32 v6, v16, v20 // (op 15589) 
v_add_f32 v7, v17, v21 // (op 15589) 


v_cvt_f16_f32 v14, v4 // (op 15590) 
v_cvt_f16_f32 v15, v5 // (op 15590) 
v_cvt_f16_f32 v16, v6 // (op 15590) 
v_cvt_f16_f32 v17, v7 // (op 15590) 


v_pack_b32_f16 v4, v14, v15 // (op 15557) 
v_pack_b32_f16 v5, v16, v17 // (op 15557) 

s_lshl_b32 s0, s4, 5 // (op 15557) 
s_mul_i32 s1, 3, s0 // (op 15557) 
s_add_u32 s1, s1, 96 // (op 15557) 

v_add_u32 v0, v13, s1 // (op 15557) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15557) Store value





s_lshl_b32 s0, s2, 7 // (op 15596) 
s_add_u32 s0, s0, 96 // (op 15596) 
v_add_u32 v0, v12, s0 // (op 15596) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15596) Load value

v_accvgpr_read v14, a160 // (op 15639) 
v_accvgpr_read v15, a161 // (op 15639) 
v_accvgpr_read v16, a162 // (op 15639) 
v_accvgpr_read v17, a163 // (op 15639) 
v_mul_f32 v18, v11, v14 // (op 15639) 
v_mul_f32 v19, v11, v15 // (op 15639) 
v_mul_f32 v20, v11, v16 // (op 15639) 
v_mul_f32 v21, v11, v17 // (op 15639) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15593) 
v_lshrrev_b32 v7, 16, v4 // (op 15593) 
v_cvt_f32_f16 v7, v7 // (op 15593) 
v_mul_f32 v14, v9, v6 // (op 15593) 
v_mul_f32 v15, v9, v7 // (op 15593) 
v_cvt_f32_f16 v6, v5 // (op 15593) 
v_lshrrev_b32 v7, 16, v5 // (op 15593) 
v_cvt_f32_f16 v7, v7 // (op 15593) 
v_mul_f32 v16, v9, v6 // (op 15593) 
v_mul_f32 v17, v9, v7 // (op 15593) 


v_add_f32 v4, v14, v18 // (op 15594) 
v_add_f32 v5, v15, v19 // (op 15594) 
v_add_f32 v6, v16, v20 // (op 15594) 
v_add_f32 v7, v17, v21 // (op 15594) 


v_cvt_f16_f32 v14, v4 // (op 15595) 
v_cvt_f16_f32 v15, v5 // (op 15595) 
v_cvt_f16_f32 v16, v6 // (op 15595) 
v_cvt_f16_f32 v17, v7 // (op 15595) 


v_pack_b32_f16 v4, v14, v15 // (op 15559) 
v_pack_b32_f16 v5, v16, v17 // (op 15559) 

s_lshl_b32 s0, s4, 7 // (op 15559) 
s_add_u32 s0, s0, 96 // (op 15559) 
v_add_u32 v0, v13, s0 // (op 15559) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15559) Store value





s_lshl_b32 s0, s2, 5 // (op 15601) 
s_mul_i32 s1, 5, s0 // (op 15601) 
s_add_u32 s1, s1, 96 // (op 15601) 

v_add_u32 v0, v12, s1 // (op 15601) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15601) Load value

v_accvgpr_read v14, a164 // (op 15641) 
v_accvgpr_read v15, a165 // (op 15641) 
v_accvgpr_read v16, a166 // (op 15641) 
v_accvgpr_read v17, a167 // (op 15641) 
v_mul_f32 v18, v11, v14 // (op 15641) 
v_mul_f32 v19, v11, v15 // (op 15641) 
v_mul_f32 v20, v11, v16 // (op 15641) 
v_mul_f32 v21, v11, v17 // (op 15641) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15598) 
v_lshrrev_b32 v7, 16, v4 // (op 15598) 
v_cvt_f32_f16 v7, v7 // (op 15598) 
v_mul_f32 v14, v9, v6 // (op 15598) 
v_mul_f32 v15, v9, v7 // (op 15598) 
v_cvt_f32_f16 v6, v5 // (op 15598) 
v_lshrrev_b32 v7, 16, v5 // (op 15598) 
v_cvt_f32_f16 v7, v7 // (op 15598) 
v_mul_f32 v16, v9, v6 // (op 15598) 
v_mul_f32 v17, v9, v7 // (op 15598) 


v_add_f32 v4, v14, v18 // (op 15599) 
v_add_f32 v5, v15, v19 // (op 15599) 
v_add_f32 v6, v16, v20 // (op 15599) 
v_add_f32 v7, v17, v21 // (op 15599) 


v_cvt_f16_f32 v14, v4 // (op 15600) 
v_cvt_f16_f32 v15, v5 // (op 15600) 
v_cvt_f16_f32 v16, v6 // (op 15600) 
v_cvt_f16_f32 v17, v7 // (op 15600) 


v_pack_b32_f16 v4, v14, v15 // (op 15561) 
v_pack_b32_f16 v5, v16, v17 // (op 15561) 

s_lshl_b32 s0, s4, 5 // (op 15561) 
s_mul_i32 s1, 5, s0 // (op 15561) 
s_add_u32 s1, s1, 96 // (op 15561) 

v_add_u32 v0, v13, s1 // (op 15561) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15561) Store value





s_lshl_b32 s0, s2, 5 // (op 15606) 
s_mul_i32 s1, 6, s0 // (op 15606) 
s_add_u32 s1, s1, 96 // (op 15606) 

v_add_u32 v0, v12, s1 // (op 15606) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15606) Load value

v_accvgpr_read v14, a168 // (op 15643) 
v_accvgpr_read v15, a169 // (op 15643) 
v_accvgpr_read v16, a170 // (op 15643) 
v_accvgpr_read v17, a171 // (op 15643) 
v_mul_f32 v18, v11, v14 // (op 15643) 
v_mul_f32 v19, v11, v15 // (op 15643) 
v_mul_f32 v20, v11, v16 // (op 15643) 
v_mul_f32 v21, v11, v17 // (op 15643) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15603) 
v_lshrrev_b32 v7, 16, v4 // (op 15603) 
v_cvt_f32_f16 v7, v7 // (op 15603) 
v_mul_f32 v14, v9, v6 // (op 15603) 
v_mul_f32 v15, v9, v7 // (op 15603) 
v_cvt_f32_f16 v6, v5 // (op 15603) 
v_lshrrev_b32 v7, 16, v5 // (op 15603) 
v_cvt_f32_f16 v7, v7 // (op 15603) 
v_mul_f32 v16, v9, v6 // (op 15603) 
v_mul_f32 v17, v9, v7 // (op 15603) 


v_add_f32 v4, v14, v18 // (op 15604) 
v_add_f32 v5, v15, v19 // (op 15604) 
v_add_f32 v6, v16, v20 // (op 15604) 
v_add_f32 v7, v17, v21 // (op 15604) 


v_cvt_f16_f32 v14, v4 // (op 15605) 
v_cvt_f16_f32 v15, v5 // (op 15605) 
v_cvt_f16_f32 v16, v6 // (op 15605) 
v_cvt_f16_f32 v17, v7 // (op 15605) 


v_pack_b32_f16 v4, v14, v15 // (op 15563) 
v_pack_b32_f16 v5, v16, v17 // (op 15563) 

s_lshl_b32 s0, s4, 5 // (op 15563) 
s_mul_i32 s1, 6, s0 // (op 15563) 
s_add_u32 s1, s1, 96 // (op 15563) 

v_add_u32 v0, v13, s1 // (op 15563) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15563) Store value





s_lshl_b32 s0, s2, 5 // (op 15611) 
s_mul_i32 s1, 7, s0 // (op 15611) 
s_add_u32 s1, s1, 96 // (op 15611) 

v_add_u32 v0, v12, s1 // (op 15611) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15611) Load value

v_accvgpr_read v14, a172 // (op 15645) 
v_accvgpr_read v15, a173 // (op 15645) 
v_accvgpr_read v16, a174 // (op 15645) 
v_accvgpr_read v17, a175 // (op 15645) 
v_mul_f32 v18, v11, v14 // (op 15645) 
v_mul_f32 v19, v11, v15 // (op 15645) 
v_mul_f32 v20, v11, v16 // (op 15645) 
v_mul_f32 v21, v11, v17 // (op 15645) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15608) 
v_lshrrev_b32 v7, 16, v4 // (op 15608) 
v_cvt_f32_f16 v7, v7 // (op 15608) 
v_mul_f32 v14, v9, v6 // (op 15608) 
v_mul_f32 v15, v9, v7 // (op 15608) 
v_cvt_f32_f16 v6, v5 // (op 15608) 
v_lshrrev_b32 v7, 16, v5 // (op 15608) 
v_cvt_f32_f16 v7, v7 // (op 15608) 
v_mul_f32 v16, v9, v6 // (op 15608) 
v_mul_f32 v17, v9, v7 // (op 15608) 


v_add_f32 v4, v14, v18 // (op 15609) 
v_add_f32 v5, v15, v19 // (op 15609) 
v_add_f32 v6, v16, v20 // (op 15609) 
v_add_f32 v7, v17, v21 // (op 15609) 


v_cvt_f16_f32 v14, v4 // (op 15610) 
v_cvt_f16_f32 v15, v5 // (op 15610) 
v_cvt_f16_f32 v16, v6 // (op 15610) 
v_cvt_f16_f32 v17, v7 // (op 15610) 


v_pack_b32_f16 v4, v14, v15 // (op 15565) 
v_pack_b32_f16 v5, v16, v17 // (op 15565) 

s_lshl_b32 s0, s4, 5 // (op 15565) 
s_mul_i32 s1, 7, s0 // (op 15565) 
s_add_u32 s1, s1, 96 // (op 15565) 

v_add_u32 v0, v13, s1 // (op 15565) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15565) Store value





s_lshl_b32 s0, s2, 8 // (op 15616) 
s_add_u32 s0, s0, 96 // (op 15616) 
v_add_u32 v0, v12, s0 // (op 15616) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15616) Load value

v_accvgpr_read v14, a176 // (op 15647) 
v_accvgpr_read v15, a177 // (op 15647) 
v_accvgpr_read v16, a178 // (op 15647) 
v_accvgpr_read v17, a179 // (op 15647) 
v_mul_f32 v18, v11, v14 // (op 15647) 
v_mul_f32 v19, v11, v15 // (op 15647) 
v_mul_f32 v20, v11, v16 // (op 15647) 
v_mul_f32 v21, v11, v17 // (op 15647) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15613) 
v_lshrrev_b32 v7, 16, v4 // (op 15613) 
v_cvt_f32_f16 v7, v7 // (op 15613) 
v_mul_f32 v14, v9, v6 // (op 15613) 
v_mul_f32 v15, v9, v7 // (op 15613) 
v_cvt_f32_f16 v6, v5 // (op 15613) 
v_lshrrev_b32 v7, 16, v5 // (op 15613) 
v_cvt_f32_f16 v7, v7 // (op 15613) 
v_mul_f32 v16, v9, v6 // (op 15613) 
v_mul_f32 v17, v9, v7 // (op 15613) 


v_add_f32 v4, v14, v18 // (op 15614) 
v_add_f32 v5, v15, v19 // (op 15614) 
v_add_f32 v6, v16, v20 // (op 15614) 
v_add_f32 v7, v17, v21 // (op 15614) 


v_cvt_f16_f32 v14, v4 // (op 15615) 
v_cvt_f16_f32 v15, v5 // (op 15615) 
v_cvt_f16_f32 v16, v6 // (op 15615) 
v_cvt_f16_f32 v17, v7 // (op 15615) 


v_pack_b32_f16 v4, v14, v15 // (op 15567) 
v_pack_b32_f16 v5, v16, v17 // (op 15567) 

s_lshl_b32 s0, s4, 8 // (op 15567) 
s_add_u32 s0, s0, 96 // (op 15567) 
v_add_u32 v0, v13, s0 // (op 15567) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15567) Store value





s_lshl_b32 s0, s2, 5 // (op 15621) 
s_mul_i32 s1, 9, s0 // (op 15621) 
s_add_u32 s1, s1, 96 // (op 15621) 

v_add_u32 v0, v12, s1 // (op 15621) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15621) Load value

v_accvgpr_read v14, a180 // (op 15649) 
v_accvgpr_read v15, a181 // (op 15649) 
v_accvgpr_read v16, a182 // (op 15649) 
v_accvgpr_read v17, a183 // (op 15649) 
v_mul_f32 v18, v11, v14 // (op 15649) 
v_mul_f32 v19, v11, v15 // (op 15649) 
v_mul_f32 v20, v11, v16 // (op 15649) 
v_mul_f32 v21, v11, v17 // (op 15649) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15618) 
v_lshrrev_b32 v7, 16, v4 // (op 15618) 
v_cvt_f32_f16 v7, v7 // (op 15618) 
v_mul_f32 v14, v9, v6 // (op 15618) 
v_mul_f32 v15, v9, v7 // (op 15618) 
v_cvt_f32_f16 v6, v5 // (op 15618) 
v_lshrrev_b32 v7, 16, v5 // (op 15618) 
v_cvt_f32_f16 v7, v7 // (op 15618) 
v_mul_f32 v16, v9, v6 // (op 15618) 
v_mul_f32 v17, v9, v7 // (op 15618) 


v_add_f32 v4, v14, v18 // (op 15619) 
v_add_f32 v5, v15, v19 // (op 15619) 
v_add_f32 v6, v16, v20 // (op 15619) 
v_add_f32 v7, v17, v21 // (op 15619) 


v_cvt_f16_f32 v14, v4 // (op 15620) 
v_cvt_f16_f32 v15, v5 // (op 15620) 
v_cvt_f16_f32 v16, v6 // (op 15620) 
v_cvt_f16_f32 v17, v7 // (op 15620) 


v_pack_b32_f16 v4, v14, v15 // (op 15569) 
v_pack_b32_f16 v5, v16, v17 // (op 15569) 

s_lshl_b32 s0, s4, 5 // (op 15569) 
s_mul_i32 s1, 9, s0 // (op 15569) 
s_add_u32 s1, s1, 96 // (op 15569) 

v_add_u32 v0, v13, s1 // (op 15569) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15569) Store value





s_lshl_b32 s0, s2, 5 // (op 15626) 
s_mul_i32 s1, 10, s0 // (op 15626) 
s_add_u32 s1, s1, 96 // (op 15626) 

v_add_u32 v0, v12, s1 // (op 15626) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15626) Load value

v_accvgpr_read v14, a184 // (op 15651) 
v_accvgpr_read v15, a185 // (op 15651) 
v_accvgpr_read v16, a186 // (op 15651) 
v_accvgpr_read v17, a187 // (op 15651) 
v_mul_f32 v18, v11, v14 // (op 15651) 
v_mul_f32 v19, v11, v15 // (op 15651) 
v_mul_f32 v20, v11, v16 // (op 15651) 
v_mul_f32 v21, v11, v17 // (op 15651) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15623) 
v_lshrrev_b32 v7, 16, v4 // (op 15623) 
v_cvt_f32_f16 v7, v7 // (op 15623) 
v_mul_f32 v14, v9, v6 // (op 15623) 
v_mul_f32 v15, v9, v7 // (op 15623) 
v_cvt_f32_f16 v6, v5 // (op 15623) 
v_lshrrev_b32 v7, 16, v5 // (op 15623) 
v_cvt_f32_f16 v7, v7 // (op 15623) 
v_mul_f32 v16, v9, v6 // (op 15623) 
v_mul_f32 v17, v9, v7 // (op 15623) 


v_add_f32 v4, v14, v18 // (op 15624) 
v_add_f32 v5, v15, v19 // (op 15624) 
v_add_f32 v6, v16, v20 // (op 15624) 
v_add_f32 v7, v17, v21 // (op 15624) 


v_cvt_f16_f32 v14, v4 // (op 15625) 
v_cvt_f16_f32 v15, v5 // (op 15625) 
v_cvt_f16_f32 v16, v6 // (op 15625) 
v_cvt_f16_f32 v17, v7 // (op 15625) 


v_pack_b32_f16 v4, v14, v15 // (op 15571) 
v_pack_b32_f16 v5, v16, v17 // (op 15571) 

s_lshl_b32 s0, s4, 5 // (op 15571) 
s_mul_i32 s1, 10, s0 // (op 15571) 
s_add_u32 s1, s1, 96 // (op 15571) 

v_add_u32 v0, v13, s1 // (op 15571) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15571) Store value





s_lshl_b32 s0, s2, 5 // (op 15631) 
s_mul_i32 s1, 11, s0 // (op 15631) 
s_add_u32 s1, s1, 96 // (op 15631) 

v_add_u32 v0, v12, s1 // (op 15631) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15631) Load value

v_accvgpr_read v14, a188 // (op 15653) 
v_accvgpr_read v15, a189 // (op 15653) 
v_accvgpr_read v16, a190 // (op 15653) 
v_accvgpr_read v17, a191 // (op 15653) 
v_mul_f32 v18, v11, v14 // (op 15653) 
v_mul_f32 v19, v11, v15 // (op 15653) 
v_mul_f32 v20, v11, v16 // (op 15653) 
v_mul_f32 v21, v11, v17 // (op 15653) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15628) 
v_lshrrev_b32 v7, 16, v4 // (op 15628) 
v_cvt_f32_f16 v7, v7 // (op 15628) 
v_mul_f32 v14, v9, v6 // (op 15628) 
v_mul_f32 v15, v9, v7 // (op 15628) 
v_cvt_f32_f16 v6, v5 // (op 15628) 
v_lshrrev_b32 v7, 16, v5 // (op 15628) 
v_cvt_f32_f16 v7, v7 // (op 15628) 
v_mul_f32 v16, v9, v6 // (op 15628) 
v_mul_f32 v17, v9, v7 // (op 15628) 


v_add_f32 v4, v14, v18 // (op 15629) 
v_add_f32 v5, v15, v19 // (op 15629) 
v_add_f32 v6, v16, v20 // (op 15629) 
v_add_f32 v7, v17, v21 // (op 15629) 


v_cvt_f16_f32 v14, v4 // (op 15630) 
v_cvt_f16_f32 v15, v5 // (op 15630) 
v_cvt_f16_f32 v16, v6 // (op 15630) 
v_cvt_f16_f32 v17, v7 // (op 15630) 


v_pack_b32_f16 v4, v14, v15 // (op 15573) 
v_pack_b32_f16 v5, v16, v17 // (op 15573) 

s_lshl_b32 s0, s4, 5 // (op 15573) 
s_mul_i32 s1, 11, s0 // (op 15573) 
s_add_u32 s1, s1, 96 // (op 15573) 

v_add_u32 v0, v13, s1 // (op 15573) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15573) Store value





v_add_u32 v0, 64, v12 // (op 15321) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15321) Load value

v_accvgpr_read v14, a96 // (op 15292) 
v_accvgpr_read v15, a97 // (op 15292) 
v_accvgpr_read v16, a98 // (op 15292) 
v_accvgpr_read v17, a99 // (op 15292) 
v_mul_f32 v18, v11, v14 // (op 15292) 
v_mul_f32 v19, v11, v15 // (op 15292) 
v_mul_f32 v20, v11, v16 // (op 15292) 
v_mul_f32 v21, v11, v17 // (op 15292) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15320) 
v_lshrrev_b32 v7, 16, v4 // (op 15320) 
v_cvt_f32_f16 v7, v7 // (op 15320) 
v_mul_f32 v14, v9, v6 // (op 15320) 
v_mul_f32 v15, v9, v7 // (op 15320) 
v_cvt_f32_f16 v6, v5 // (op 15320) 
v_lshrrev_b32 v7, 16, v5 // (op 15320) 
v_cvt_f32_f16 v7, v7 // (op 15320) 
v_mul_f32 v16, v9, v6 // (op 15320) 
v_mul_f32 v17, v9, v7 // (op 15320) 


v_add_f32 v4, v14, v18 // (op 15293) 
v_add_f32 v5, v15, v19 // (op 15293) 
v_add_f32 v6, v16, v20 // (op 15293) 
v_add_f32 v7, v17, v21 // (op 15293) 


v_cvt_f16_f32 v14, v4 // (op 15294) 
v_cvt_f16_f32 v15, v5 // (op 15294) 
v_cvt_f16_f32 v16, v6 // (op 15294) 
v_cvt_f16_f32 v17, v7 // (op 15294) 


v_pack_b32_f16 v4, v14, v15 // (op 15296) 
v_pack_b32_f16 v5, v16, v17 // (op 15296) 

v_add_u32 v0, 64, v13 // (op 15296) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15296) Store value





s_lshl_b32 s0, s2, 5 // (op 15326) 
s_add_u32 s0, s0, 64 // (op 15326) 
v_add_u32 v0, v12, s0 // (op 15326) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15326) Load value

v_accvgpr_read v14, a100 // (op 15378) 
v_accvgpr_read v15, a101 // (op 15378) 
v_accvgpr_read v16, a102 // (op 15378) 
v_accvgpr_read v17, a103 // (op 15378) 
v_mul_f32 v18, v11, v14 // (op 15378) 
v_mul_f32 v19, v11, v15 // (op 15378) 
v_mul_f32 v20, v11, v16 // (op 15378) 
v_mul_f32 v21, v11, v17 // (op 15378) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15323) 
v_lshrrev_b32 v7, 16, v4 // (op 15323) 
v_cvt_f32_f16 v7, v7 // (op 15323) 
v_mul_f32 v14, v9, v6 // (op 15323) 
v_mul_f32 v15, v9, v7 // (op 15323) 
v_cvt_f32_f16 v6, v5 // (op 15323) 
v_lshrrev_b32 v7, 16, v5 // (op 15323) 
v_cvt_f32_f16 v7, v7 // (op 15323) 
v_mul_f32 v16, v9, v6 // (op 15323) 
v_mul_f32 v17, v9, v7 // (op 15323) 


v_add_f32 v4, v14, v18 // (op 15324) 
v_add_f32 v5, v15, v19 // (op 15324) 
v_add_f32 v6, v16, v20 // (op 15324) 
v_add_f32 v7, v17, v21 // (op 15324) 


v_cvt_f16_f32 v14, v4 // (op 15325) 
v_cvt_f16_f32 v15, v5 // (op 15325) 
v_cvt_f16_f32 v16, v6 // (op 15325) 
v_cvt_f16_f32 v17, v7 // (op 15325) 


v_pack_b32_f16 v4, v14, v15 // (op 15298) 
v_pack_b32_f16 v5, v16, v17 // (op 15298) 

s_lshl_b32 s0, s4, 5 // (op 15298) 
s_add_u32 s0, s0, 64 // (op 15298) 
v_add_u32 v0, v13, s0 // (op 15298) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15298) Store value





s_lshl_b32 s0, s2, 6 // (op 15331) 
s_add_u32 s0, s0, 64 // (op 15331) 
v_add_u32 v0, v12, s0 // (op 15331) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15331) Load value

v_accvgpr_read v14, a104 // (op 15380) 
v_accvgpr_read v15, a105 // (op 15380) 
v_accvgpr_read v16, a106 // (op 15380) 
v_accvgpr_read v17, a107 // (op 15380) 
v_mul_f32 v18, v11, v14 // (op 15380) 
v_mul_f32 v19, v11, v15 // (op 15380) 
v_mul_f32 v20, v11, v16 // (op 15380) 
v_mul_f32 v21, v11, v17 // (op 15380) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15328) 
v_lshrrev_b32 v7, 16, v4 // (op 15328) 
v_cvt_f32_f16 v7, v7 // (op 15328) 
v_mul_f32 v14, v9, v6 // (op 15328) 
v_mul_f32 v15, v9, v7 // (op 15328) 
v_cvt_f32_f16 v6, v5 // (op 15328) 
v_lshrrev_b32 v7, 16, v5 // (op 15328) 
v_cvt_f32_f16 v7, v7 // (op 15328) 
v_mul_f32 v16, v9, v6 // (op 15328) 
v_mul_f32 v17, v9, v7 // (op 15328) 


v_add_f32 v4, v14, v18 // (op 15329) 
v_add_f32 v5, v15, v19 // (op 15329) 
v_add_f32 v6, v16, v20 // (op 15329) 
v_add_f32 v7, v17, v21 // (op 15329) 


v_cvt_f16_f32 v14, v4 // (op 15330) 
v_cvt_f16_f32 v15, v5 // (op 15330) 
v_cvt_f16_f32 v16, v6 // (op 15330) 
v_cvt_f16_f32 v17, v7 // (op 15330) 


v_pack_b32_f16 v4, v14, v15 // (op 15300) 
v_pack_b32_f16 v5, v16, v17 // (op 15300) 

s_lshl_b32 s0, s4, 6 // (op 15300) 
s_add_u32 s0, s0, 64 // (op 15300) 
v_add_u32 v0, v13, s0 // (op 15300) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15300) Store value





s_lshl_b32 s0, s2, 5 // (op 15336) 
s_mul_i32 s1, 3, s0 // (op 15336) 
s_add_u32 s1, s1, 64 // (op 15336) 

v_add_u32 v0, v12, s1 // (op 15336) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15336) Load value

v_accvgpr_read v14, a108 // (op 15382) 
v_accvgpr_read v15, a109 // (op 15382) 
v_accvgpr_read v16, a110 // (op 15382) 
v_accvgpr_read v17, a111 // (op 15382) 
v_mul_f32 v18, v11, v14 // (op 15382) 
v_mul_f32 v19, v11, v15 // (op 15382) 
v_mul_f32 v20, v11, v16 // (op 15382) 
v_mul_f32 v21, v11, v17 // (op 15382) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15333) 
v_lshrrev_b32 v7, 16, v4 // (op 15333) 
v_cvt_f32_f16 v7, v7 // (op 15333) 
v_mul_f32 v14, v9, v6 // (op 15333) 
v_mul_f32 v15, v9, v7 // (op 15333) 
v_cvt_f32_f16 v6, v5 // (op 15333) 
v_lshrrev_b32 v7, 16, v5 // (op 15333) 
v_cvt_f32_f16 v7, v7 // (op 15333) 
v_mul_f32 v16, v9, v6 // (op 15333) 
v_mul_f32 v17, v9, v7 // (op 15333) 


v_add_f32 v4, v14, v18 // (op 15334) 
v_add_f32 v5, v15, v19 // (op 15334) 
v_add_f32 v6, v16, v20 // (op 15334) 
v_add_f32 v7, v17, v21 // (op 15334) 


v_cvt_f16_f32 v14, v4 // (op 15335) 
v_cvt_f16_f32 v15, v5 // (op 15335) 
v_cvt_f16_f32 v16, v6 // (op 15335) 
v_cvt_f16_f32 v17, v7 // (op 15335) 


v_pack_b32_f16 v4, v14, v15 // (op 15302) 
v_pack_b32_f16 v5, v16, v17 // (op 15302) 

s_lshl_b32 s0, s4, 5 // (op 15302) 
s_mul_i32 s1, 3, s0 // (op 15302) 
s_add_u32 s1, s1, 64 // (op 15302) 

v_add_u32 v0, v13, s1 // (op 15302) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15302) Store value





s_lshl_b32 s0, s2, 7 // (op 15341) 
s_add_u32 s0, s0, 64 // (op 15341) 
v_add_u32 v0, v12, s0 // (op 15341) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15341) Load value

v_accvgpr_read v14, a112 // (op 15384) 
v_accvgpr_read v15, a113 // (op 15384) 
v_accvgpr_read v16, a114 // (op 15384) 
v_accvgpr_read v17, a115 // (op 15384) 
v_mul_f32 v18, v11, v14 // (op 15384) 
v_mul_f32 v19, v11, v15 // (op 15384) 
v_mul_f32 v20, v11, v16 // (op 15384) 
v_mul_f32 v21, v11, v17 // (op 15384) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15338) 
v_lshrrev_b32 v7, 16, v4 // (op 15338) 
v_cvt_f32_f16 v7, v7 // (op 15338) 
v_mul_f32 v14, v9, v6 // (op 15338) 
v_mul_f32 v15, v9, v7 // (op 15338) 
v_cvt_f32_f16 v6, v5 // (op 15338) 
v_lshrrev_b32 v7, 16, v5 // (op 15338) 
v_cvt_f32_f16 v7, v7 // (op 15338) 
v_mul_f32 v16, v9, v6 // (op 15338) 
v_mul_f32 v17, v9, v7 // (op 15338) 


v_add_f32 v4, v14, v18 // (op 15339) 
v_add_f32 v5, v15, v19 // (op 15339) 
v_add_f32 v6, v16, v20 // (op 15339) 
v_add_f32 v7, v17, v21 // (op 15339) 


v_cvt_f16_f32 v14, v4 // (op 15340) 
v_cvt_f16_f32 v15, v5 // (op 15340) 
v_cvt_f16_f32 v16, v6 // (op 15340) 
v_cvt_f16_f32 v17, v7 // (op 15340) 


v_pack_b32_f16 v4, v14, v15 // (op 15304) 
v_pack_b32_f16 v5, v16, v17 // (op 15304) 

s_lshl_b32 s0, s4, 7 // (op 15304) 
s_add_u32 s0, s0, 64 // (op 15304) 
v_add_u32 v0, v13, s0 // (op 15304) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15304) Store value





s_lshl_b32 s0, s2, 5 // (op 15346) 
s_mul_i32 s1, 5, s0 // (op 15346) 
s_add_u32 s1, s1, 64 // (op 15346) 

v_add_u32 v0, v12, s1 // (op 15346) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15346) Load value

v_accvgpr_read v14, a116 // (op 15386) 
v_accvgpr_read v15, a117 // (op 15386) 
v_accvgpr_read v16, a118 // (op 15386) 
v_accvgpr_read v17, a119 // (op 15386) 
v_mul_f32 v18, v11, v14 // (op 15386) 
v_mul_f32 v19, v11, v15 // (op 15386) 
v_mul_f32 v20, v11, v16 // (op 15386) 
v_mul_f32 v21, v11, v17 // (op 15386) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15343) 
v_lshrrev_b32 v7, 16, v4 // (op 15343) 
v_cvt_f32_f16 v7, v7 // (op 15343) 
v_mul_f32 v14, v9, v6 // (op 15343) 
v_mul_f32 v15, v9, v7 // (op 15343) 
v_cvt_f32_f16 v6, v5 // (op 15343) 
v_lshrrev_b32 v7, 16, v5 // (op 15343) 
v_cvt_f32_f16 v7, v7 // (op 15343) 
v_mul_f32 v16, v9, v6 // (op 15343) 
v_mul_f32 v17, v9, v7 // (op 15343) 


v_add_f32 v4, v14, v18 // (op 15344) 
v_add_f32 v5, v15, v19 // (op 15344) 
v_add_f32 v6, v16, v20 // (op 15344) 
v_add_f32 v7, v17, v21 // (op 15344) 


v_cvt_f16_f32 v14, v4 // (op 15345) 
v_cvt_f16_f32 v15, v5 // (op 15345) 
v_cvt_f16_f32 v16, v6 // (op 15345) 
v_cvt_f16_f32 v17, v7 // (op 15345) 


v_pack_b32_f16 v4, v14, v15 // (op 15306) 
v_pack_b32_f16 v5, v16, v17 // (op 15306) 

s_lshl_b32 s0, s4, 5 // (op 15306) 
s_mul_i32 s1, 5, s0 // (op 15306) 
s_add_u32 s1, s1, 64 // (op 15306) 

v_add_u32 v0, v13, s1 // (op 15306) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15306) Store value





s_lshl_b32 s0, s2, 5 // (op 15351) 
s_mul_i32 s1, 6, s0 // (op 15351) 
s_add_u32 s1, s1, 64 // (op 15351) 

v_add_u32 v0, v12, s1 // (op 15351) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15351) Load value

v_accvgpr_read v14, a120 // (op 15388) 
v_accvgpr_read v15, a121 // (op 15388) 
v_accvgpr_read v16, a122 // (op 15388) 
v_accvgpr_read v17, a123 // (op 15388) 
v_mul_f32 v18, v11, v14 // (op 15388) 
v_mul_f32 v19, v11, v15 // (op 15388) 
v_mul_f32 v20, v11, v16 // (op 15388) 
v_mul_f32 v21, v11, v17 // (op 15388) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15348) 
v_lshrrev_b32 v7, 16, v4 // (op 15348) 
v_cvt_f32_f16 v7, v7 // (op 15348) 
v_mul_f32 v14, v9, v6 // (op 15348) 
v_mul_f32 v15, v9, v7 // (op 15348) 
v_cvt_f32_f16 v6, v5 // (op 15348) 
v_lshrrev_b32 v7, 16, v5 // (op 15348) 
v_cvt_f32_f16 v7, v7 // (op 15348) 
v_mul_f32 v16, v9, v6 // (op 15348) 
v_mul_f32 v17, v9, v7 // (op 15348) 


v_add_f32 v4, v14, v18 // (op 15349) 
v_add_f32 v5, v15, v19 // (op 15349) 
v_add_f32 v6, v16, v20 // (op 15349) 
v_add_f32 v7, v17, v21 // (op 15349) 


v_cvt_f16_f32 v14, v4 // (op 15350) 
v_cvt_f16_f32 v15, v5 // (op 15350) 
v_cvt_f16_f32 v16, v6 // (op 15350) 
v_cvt_f16_f32 v17, v7 // (op 15350) 


v_pack_b32_f16 v4, v14, v15 // (op 15308) 
v_pack_b32_f16 v5, v16, v17 // (op 15308) 

s_lshl_b32 s0, s4, 5 // (op 15308) 
s_mul_i32 s1, 6, s0 // (op 15308) 
s_add_u32 s1, s1, 64 // (op 15308) 

v_add_u32 v0, v13, s1 // (op 15308) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15308) Store value





s_lshl_b32 s0, s2, 5 // (op 15356) 
s_mul_i32 s1, 7, s0 // (op 15356) 
s_add_u32 s1, s1, 64 // (op 15356) 

v_add_u32 v0, v12, s1 // (op 15356) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15356) Load value

v_accvgpr_read v14, a124 // (op 15390) 
v_accvgpr_read v15, a125 // (op 15390) 
v_accvgpr_read v16, a126 // (op 15390) 
v_accvgpr_read v17, a127 // (op 15390) 
v_mul_f32 v18, v11, v14 // (op 15390) 
v_mul_f32 v19, v11, v15 // (op 15390) 
v_mul_f32 v20, v11, v16 // (op 15390) 
v_mul_f32 v21, v11, v17 // (op 15390) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15353) 
v_lshrrev_b32 v7, 16, v4 // (op 15353) 
v_cvt_f32_f16 v7, v7 // (op 15353) 
v_mul_f32 v14, v9, v6 // (op 15353) 
v_mul_f32 v15, v9, v7 // (op 15353) 
v_cvt_f32_f16 v6, v5 // (op 15353) 
v_lshrrev_b32 v7, 16, v5 // (op 15353) 
v_cvt_f32_f16 v7, v7 // (op 15353) 
v_mul_f32 v16, v9, v6 // (op 15353) 
v_mul_f32 v17, v9, v7 // (op 15353) 


v_add_f32 v4, v14, v18 // (op 15354) 
v_add_f32 v5, v15, v19 // (op 15354) 
v_add_f32 v6, v16, v20 // (op 15354) 
v_add_f32 v7, v17, v21 // (op 15354) 


v_cvt_f16_f32 v14, v4 // (op 15355) 
v_cvt_f16_f32 v15, v5 // (op 15355) 
v_cvt_f16_f32 v16, v6 // (op 15355) 
v_cvt_f16_f32 v17, v7 // (op 15355) 


v_pack_b32_f16 v4, v14, v15 // (op 15310) 
v_pack_b32_f16 v5, v16, v17 // (op 15310) 

s_lshl_b32 s0, s4, 5 // (op 15310) 
s_mul_i32 s1, 7, s0 // (op 15310) 
s_add_u32 s1, s1, 64 // (op 15310) 

v_add_u32 v0, v13, s1 // (op 15310) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15310) Store value





s_lshl_b32 s0, s2, 8 // (op 15361) 
s_add_u32 s0, s0, 64 // (op 15361) 
v_add_u32 v0, v12, s0 // (op 15361) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15361) Load value

v_accvgpr_read v14, a128 // (op 15392) 
v_accvgpr_read v15, a129 // (op 15392) 
v_accvgpr_read v16, a130 // (op 15392) 
v_accvgpr_read v17, a131 // (op 15392) 
v_mul_f32 v18, v11, v14 // (op 15392) 
v_mul_f32 v19, v11, v15 // (op 15392) 
v_mul_f32 v20, v11, v16 // (op 15392) 
v_mul_f32 v21, v11, v17 // (op 15392) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15358) 
v_lshrrev_b32 v7, 16, v4 // (op 15358) 
v_cvt_f32_f16 v7, v7 // (op 15358) 
v_mul_f32 v14, v9, v6 // (op 15358) 
v_mul_f32 v15, v9, v7 // (op 15358) 
v_cvt_f32_f16 v6, v5 // (op 15358) 
v_lshrrev_b32 v7, 16, v5 // (op 15358) 
v_cvt_f32_f16 v7, v7 // (op 15358) 
v_mul_f32 v16, v9, v6 // (op 15358) 
v_mul_f32 v17, v9, v7 // (op 15358) 


v_add_f32 v4, v14, v18 // (op 15359) 
v_add_f32 v5, v15, v19 // (op 15359) 
v_add_f32 v6, v16, v20 // (op 15359) 
v_add_f32 v7, v17, v21 // (op 15359) 


v_cvt_f16_f32 v14, v4 // (op 15360) 
v_cvt_f16_f32 v15, v5 // (op 15360) 
v_cvt_f16_f32 v16, v6 // (op 15360) 
v_cvt_f16_f32 v17, v7 // (op 15360) 


v_pack_b32_f16 v4, v14, v15 // (op 15312) 
v_pack_b32_f16 v5, v16, v17 // (op 15312) 

s_lshl_b32 s0, s4, 8 // (op 15312) 
s_add_u32 s0, s0, 64 // (op 15312) 
v_add_u32 v0, v13, s0 // (op 15312) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15312) Store value





s_lshl_b32 s0, s2, 5 // (op 15366) 
s_mul_i32 s1, 9, s0 // (op 15366) 
s_add_u32 s1, s1, 64 // (op 15366) 

v_add_u32 v0, v12, s1 // (op 15366) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15366) Load value

v_accvgpr_read v14, a132 // (op 15394) 
v_accvgpr_read v15, a133 // (op 15394) 
v_accvgpr_read v16, a134 // (op 15394) 
v_accvgpr_read v17, a135 // (op 15394) 
v_mul_f32 v18, v11, v14 // (op 15394) 
v_mul_f32 v19, v11, v15 // (op 15394) 
v_mul_f32 v20, v11, v16 // (op 15394) 
v_mul_f32 v21, v11, v17 // (op 15394) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15363) 
v_lshrrev_b32 v7, 16, v4 // (op 15363) 
v_cvt_f32_f16 v7, v7 // (op 15363) 
v_mul_f32 v14, v9, v6 // (op 15363) 
v_mul_f32 v15, v9, v7 // (op 15363) 
v_cvt_f32_f16 v6, v5 // (op 15363) 
v_lshrrev_b32 v7, 16, v5 // (op 15363) 
v_cvt_f32_f16 v7, v7 // (op 15363) 
v_mul_f32 v16, v9, v6 // (op 15363) 
v_mul_f32 v17, v9, v7 // (op 15363) 


v_add_f32 v4, v14, v18 // (op 15364) 
v_add_f32 v5, v15, v19 // (op 15364) 
v_add_f32 v6, v16, v20 // (op 15364) 
v_add_f32 v7, v17, v21 // (op 15364) 


v_cvt_f16_f32 v14, v4 // (op 15365) 
v_cvt_f16_f32 v15, v5 // (op 15365) 
v_cvt_f16_f32 v16, v6 // (op 15365) 
v_cvt_f16_f32 v17, v7 // (op 15365) 


v_pack_b32_f16 v4, v14, v15 // (op 15314) 
v_pack_b32_f16 v5, v16, v17 // (op 15314) 

s_lshl_b32 s0, s4, 5 // (op 15314) 
s_mul_i32 s1, 9, s0 // (op 15314) 
s_add_u32 s1, s1, 64 // (op 15314) 

v_add_u32 v0, v13, s1 // (op 15314) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15314) Store value





s_lshl_b32 s0, s2, 5 // (op 15371) 
s_mul_i32 s1, 10, s0 // (op 15371) 
s_add_u32 s1, s1, 64 // (op 15371) 

v_add_u32 v0, v12, s1 // (op 15371) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15371) Load value

v_accvgpr_read v14, a136 // (op 15396) 
v_accvgpr_read v15, a137 // (op 15396) 
v_accvgpr_read v16, a138 // (op 15396) 
v_accvgpr_read v17, a139 // (op 15396) 
v_mul_f32 v18, v11, v14 // (op 15396) 
v_mul_f32 v19, v11, v15 // (op 15396) 
v_mul_f32 v20, v11, v16 // (op 15396) 
v_mul_f32 v21, v11, v17 // (op 15396) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15368) 
v_lshrrev_b32 v7, 16, v4 // (op 15368) 
v_cvt_f32_f16 v7, v7 // (op 15368) 
v_mul_f32 v14, v9, v6 // (op 15368) 
v_mul_f32 v15, v9, v7 // (op 15368) 
v_cvt_f32_f16 v6, v5 // (op 15368) 
v_lshrrev_b32 v7, 16, v5 // (op 15368) 
v_cvt_f32_f16 v7, v7 // (op 15368) 
v_mul_f32 v16, v9, v6 // (op 15368) 
v_mul_f32 v17, v9, v7 // (op 15368) 


v_add_f32 v4, v14, v18 // (op 15369) 
v_add_f32 v5, v15, v19 // (op 15369) 
v_add_f32 v6, v16, v20 // (op 15369) 
v_add_f32 v7, v17, v21 // (op 15369) 


v_cvt_f16_f32 v14, v4 // (op 15370) 
v_cvt_f16_f32 v15, v5 // (op 15370) 
v_cvt_f16_f32 v16, v6 // (op 15370) 
v_cvt_f16_f32 v17, v7 // (op 15370) 


v_pack_b32_f16 v4, v14, v15 // (op 15316) 
v_pack_b32_f16 v5, v16, v17 // (op 15316) 

s_lshl_b32 s0, s4, 5 // (op 15316) 
s_mul_i32 s1, 10, s0 // (op 15316) 
s_add_u32 s1, s1, 64 // (op 15316) 

v_add_u32 v0, v13, s1 // (op 15316) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15316) Store value





s_lshl_b32 s0, s2, 5 // (op 15376) 
s_mul_i32 s1, 11, s0 // (op 15376) 
s_add_u32 s1, s1, 64 // (op 15376) 

v_add_u32 v0, v12, s1 // (op 15376) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15376) Load value

v_accvgpr_read v14, a140 // (op 15398) 
v_accvgpr_read v15, a141 // (op 15398) 
v_accvgpr_read v16, a142 // (op 15398) 
v_accvgpr_read v17, a143 // (op 15398) 
v_mul_f32 v18, v11, v14 // (op 15398) 
v_mul_f32 v19, v11, v15 // (op 15398) 
v_mul_f32 v20, v11, v16 // (op 15398) 
v_mul_f32 v21, v11, v17 // (op 15398) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15373) 
v_lshrrev_b32 v7, 16, v4 // (op 15373) 
v_cvt_f32_f16 v7, v7 // (op 15373) 
v_mul_f32 v14, v9, v6 // (op 15373) 
v_mul_f32 v15, v9, v7 // (op 15373) 
v_cvt_f32_f16 v6, v5 // (op 15373) 
v_lshrrev_b32 v7, 16, v5 // (op 15373) 
v_cvt_f32_f16 v7, v7 // (op 15373) 
v_mul_f32 v16, v9, v6 // (op 15373) 
v_mul_f32 v17, v9, v7 // (op 15373) 


v_add_f32 v4, v14, v18 // (op 15374) 
v_add_f32 v5, v15, v19 // (op 15374) 
v_add_f32 v6, v16, v20 // (op 15374) 
v_add_f32 v7, v17, v21 // (op 15374) 


v_cvt_f16_f32 v14, v4 // (op 15375) 
v_cvt_f16_f32 v15, v5 // (op 15375) 
v_cvt_f16_f32 v16, v6 // (op 15375) 
v_cvt_f16_f32 v17, v7 // (op 15375) 


v_pack_b32_f16 v4, v14, v15 // (op 15318) 
v_pack_b32_f16 v5, v16, v17 // (op 15318) 

s_lshl_b32 s0, s4, 5 // (op 15318) 
s_mul_i32 s1, 11, s0 // (op 15318) 
s_add_u32 s1, s1, 64 // (op 15318) 

v_add_u32 v0, v13, s1 // (op 15318) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15318) Store value





v_add_u32 v0, 32, v12 // (op 15066) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15066) Load value

v_accvgpr_read v14, a48 // (op 15037) 
v_accvgpr_read v15, a49 // (op 15037) 
v_accvgpr_read v16, a50 // (op 15037) 
v_accvgpr_read v17, a51 // (op 15037) 
v_mul_f32 v18, v11, v14 // (op 15037) 
v_mul_f32 v19, v11, v15 // (op 15037) 
v_mul_f32 v20, v11, v16 // (op 15037) 
v_mul_f32 v21, v11, v17 // (op 15037) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15065) 
v_lshrrev_b32 v7, 16, v4 // (op 15065) 
v_cvt_f32_f16 v7, v7 // (op 15065) 
v_mul_f32 v14, v9, v6 // (op 15065) 
v_mul_f32 v15, v9, v7 // (op 15065) 
v_cvt_f32_f16 v6, v5 // (op 15065) 
v_lshrrev_b32 v7, 16, v5 // (op 15065) 
v_cvt_f32_f16 v7, v7 // (op 15065) 
v_mul_f32 v16, v9, v6 // (op 15065) 
v_mul_f32 v17, v9, v7 // (op 15065) 


v_add_f32 v4, v14, v18 // (op 15038) 
v_add_f32 v5, v15, v19 // (op 15038) 
v_add_f32 v6, v16, v20 // (op 15038) 
v_add_f32 v7, v17, v21 // (op 15038) 


v_cvt_f16_f32 v14, v4 // (op 15039) 
v_cvt_f16_f32 v15, v5 // (op 15039) 
v_cvt_f16_f32 v16, v6 // (op 15039) 
v_cvt_f16_f32 v17, v7 // (op 15039) 


v_pack_b32_f16 v4, v14, v15 // (op 15041) 
v_pack_b32_f16 v5, v16, v17 // (op 15041) 

v_add_u32 v0, 32, v13 // (op 15041) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15041) Store value





s_lshl_b32 s0, s2, 5 // (op 15071) 
s_add_u32 s0, s0, 32 // (op 15071) 
v_add_u32 v0, v12, s0 // (op 15071) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15071) Load value

v_accvgpr_read v14, a52 // (op 15123) 
v_accvgpr_read v15, a53 // (op 15123) 
v_accvgpr_read v16, a54 // (op 15123) 
v_accvgpr_read v17, a55 // (op 15123) 
v_mul_f32 v18, v11, v14 // (op 15123) 
v_mul_f32 v19, v11, v15 // (op 15123) 
v_mul_f32 v20, v11, v16 // (op 15123) 
v_mul_f32 v21, v11, v17 // (op 15123) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15068) 
v_lshrrev_b32 v7, 16, v4 // (op 15068) 
v_cvt_f32_f16 v7, v7 // (op 15068) 
v_mul_f32 v14, v9, v6 // (op 15068) 
v_mul_f32 v15, v9, v7 // (op 15068) 
v_cvt_f32_f16 v6, v5 // (op 15068) 
v_lshrrev_b32 v7, 16, v5 // (op 15068) 
v_cvt_f32_f16 v7, v7 // (op 15068) 
v_mul_f32 v16, v9, v6 // (op 15068) 
v_mul_f32 v17, v9, v7 // (op 15068) 


v_add_f32 v4, v14, v18 // (op 15069) 
v_add_f32 v5, v15, v19 // (op 15069) 
v_add_f32 v6, v16, v20 // (op 15069) 
v_add_f32 v7, v17, v21 // (op 15069) 


v_cvt_f16_f32 v14, v4 // (op 15070) 
v_cvt_f16_f32 v15, v5 // (op 15070) 
v_cvt_f16_f32 v16, v6 // (op 15070) 
v_cvt_f16_f32 v17, v7 // (op 15070) 


v_pack_b32_f16 v4, v14, v15 // (op 15043) 
v_pack_b32_f16 v5, v16, v17 // (op 15043) 

s_lshl_b32 s0, s4, 5 // (op 15043) 
s_add_u32 s0, s0, 32 // (op 15043) 
v_add_u32 v0, v13, s0 // (op 15043) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15043) Store value





s_lshl_b32 s0, s2, 6 // (op 15076) 
s_add_u32 s0, s0, 32 // (op 15076) 
v_add_u32 v0, v12, s0 // (op 15076) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15076) Load value

v_accvgpr_read v14, a56 // (op 15125) 
v_accvgpr_read v15, a57 // (op 15125) 
v_accvgpr_read v16, a58 // (op 15125) 
v_accvgpr_read v17, a59 // (op 15125) 
v_mul_f32 v18, v11, v14 // (op 15125) 
v_mul_f32 v19, v11, v15 // (op 15125) 
v_mul_f32 v20, v11, v16 // (op 15125) 
v_mul_f32 v21, v11, v17 // (op 15125) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15073) 
v_lshrrev_b32 v7, 16, v4 // (op 15073) 
v_cvt_f32_f16 v7, v7 // (op 15073) 
v_mul_f32 v14, v9, v6 // (op 15073) 
v_mul_f32 v15, v9, v7 // (op 15073) 
v_cvt_f32_f16 v6, v5 // (op 15073) 
v_lshrrev_b32 v7, 16, v5 // (op 15073) 
v_cvt_f32_f16 v7, v7 // (op 15073) 
v_mul_f32 v16, v9, v6 // (op 15073) 
v_mul_f32 v17, v9, v7 // (op 15073) 


v_add_f32 v4, v14, v18 // (op 15074) 
v_add_f32 v5, v15, v19 // (op 15074) 
v_add_f32 v6, v16, v20 // (op 15074) 
v_add_f32 v7, v17, v21 // (op 15074) 


v_cvt_f16_f32 v14, v4 // (op 15075) 
v_cvt_f16_f32 v15, v5 // (op 15075) 
v_cvt_f16_f32 v16, v6 // (op 15075) 
v_cvt_f16_f32 v17, v7 // (op 15075) 


v_pack_b32_f16 v4, v14, v15 // (op 15045) 
v_pack_b32_f16 v5, v16, v17 // (op 15045) 

s_lshl_b32 s0, s4, 6 // (op 15045) 
s_add_u32 s0, s0, 32 // (op 15045) 
v_add_u32 v0, v13, s0 // (op 15045) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15045) Store value





s_lshl_b32 s0, s2, 5 // (op 15081) 
s_mul_i32 s1, 3, s0 // (op 15081) 
s_add_u32 s1, s1, 32 // (op 15081) 

v_add_u32 v0, v12, s1 // (op 15081) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15081) Load value

v_accvgpr_read v14, a60 // (op 15127) 
v_accvgpr_read v15, a61 // (op 15127) 
v_accvgpr_read v16, a62 // (op 15127) 
v_accvgpr_read v17, a63 // (op 15127) 
v_mul_f32 v18, v11, v14 // (op 15127) 
v_mul_f32 v19, v11, v15 // (op 15127) 
v_mul_f32 v20, v11, v16 // (op 15127) 
v_mul_f32 v21, v11, v17 // (op 15127) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15078) 
v_lshrrev_b32 v7, 16, v4 // (op 15078) 
v_cvt_f32_f16 v7, v7 // (op 15078) 
v_mul_f32 v14, v9, v6 // (op 15078) 
v_mul_f32 v15, v9, v7 // (op 15078) 
v_cvt_f32_f16 v6, v5 // (op 15078) 
v_lshrrev_b32 v7, 16, v5 // (op 15078) 
v_cvt_f32_f16 v7, v7 // (op 15078) 
v_mul_f32 v16, v9, v6 // (op 15078) 
v_mul_f32 v17, v9, v7 // (op 15078) 


v_add_f32 v4, v14, v18 // (op 15079) 
v_add_f32 v5, v15, v19 // (op 15079) 
v_add_f32 v6, v16, v20 // (op 15079) 
v_add_f32 v7, v17, v21 // (op 15079) 


v_cvt_f16_f32 v14, v4 // (op 15080) 
v_cvt_f16_f32 v15, v5 // (op 15080) 
v_cvt_f16_f32 v16, v6 // (op 15080) 
v_cvt_f16_f32 v17, v7 // (op 15080) 


v_pack_b32_f16 v4, v14, v15 // (op 15047) 
v_pack_b32_f16 v5, v16, v17 // (op 15047) 

s_lshl_b32 s0, s4, 5 // (op 15047) 
s_mul_i32 s1, 3, s0 // (op 15047) 
s_add_u32 s1, s1, 32 // (op 15047) 

v_add_u32 v0, v13, s1 // (op 15047) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15047) Store value





s_lshl_b32 s0, s2, 7 // (op 15086) 
s_add_u32 s0, s0, 32 // (op 15086) 
v_add_u32 v0, v12, s0 // (op 15086) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15086) Load value

v_accvgpr_read v14, a64 // (op 15129) 
v_accvgpr_read v15, a65 // (op 15129) 
v_accvgpr_read v16, a66 // (op 15129) 
v_accvgpr_read v17, a67 // (op 15129) 
v_mul_f32 v18, v11, v14 // (op 15129) 
v_mul_f32 v19, v11, v15 // (op 15129) 
v_mul_f32 v20, v11, v16 // (op 15129) 
v_mul_f32 v21, v11, v17 // (op 15129) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15083) 
v_lshrrev_b32 v7, 16, v4 // (op 15083) 
v_cvt_f32_f16 v7, v7 // (op 15083) 
v_mul_f32 v14, v9, v6 // (op 15083) 
v_mul_f32 v15, v9, v7 // (op 15083) 
v_cvt_f32_f16 v6, v5 // (op 15083) 
v_lshrrev_b32 v7, 16, v5 // (op 15083) 
v_cvt_f32_f16 v7, v7 // (op 15083) 
v_mul_f32 v16, v9, v6 // (op 15083) 
v_mul_f32 v17, v9, v7 // (op 15083) 


v_add_f32 v4, v14, v18 // (op 15084) 
v_add_f32 v5, v15, v19 // (op 15084) 
v_add_f32 v6, v16, v20 // (op 15084) 
v_add_f32 v7, v17, v21 // (op 15084) 


v_cvt_f16_f32 v14, v4 // (op 15085) 
v_cvt_f16_f32 v15, v5 // (op 15085) 
v_cvt_f16_f32 v16, v6 // (op 15085) 
v_cvt_f16_f32 v17, v7 // (op 15085) 


v_pack_b32_f16 v4, v14, v15 // (op 15049) 
v_pack_b32_f16 v5, v16, v17 // (op 15049) 

s_lshl_b32 s0, s4, 7 // (op 15049) 
s_add_u32 s0, s0, 32 // (op 15049) 
v_add_u32 v0, v13, s0 // (op 15049) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15049) Store value





s_lshl_b32 s0, s2, 5 // (op 15091) 
s_mul_i32 s1, 5, s0 // (op 15091) 
s_add_u32 s1, s1, 32 // (op 15091) 

v_add_u32 v0, v12, s1 // (op 15091) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15091) Load value

v_accvgpr_read v14, a68 // (op 15131) 
v_accvgpr_read v15, a69 // (op 15131) 
v_accvgpr_read v16, a70 // (op 15131) 
v_accvgpr_read v17, a71 // (op 15131) 
v_mul_f32 v18, v11, v14 // (op 15131) 
v_mul_f32 v19, v11, v15 // (op 15131) 
v_mul_f32 v20, v11, v16 // (op 15131) 
v_mul_f32 v21, v11, v17 // (op 15131) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15088) 
v_lshrrev_b32 v7, 16, v4 // (op 15088) 
v_cvt_f32_f16 v7, v7 // (op 15088) 
v_mul_f32 v14, v9, v6 // (op 15088) 
v_mul_f32 v15, v9, v7 // (op 15088) 
v_cvt_f32_f16 v6, v5 // (op 15088) 
v_lshrrev_b32 v7, 16, v5 // (op 15088) 
v_cvt_f32_f16 v7, v7 // (op 15088) 
v_mul_f32 v16, v9, v6 // (op 15088) 
v_mul_f32 v17, v9, v7 // (op 15088) 


v_add_f32 v4, v14, v18 // (op 15089) 
v_add_f32 v5, v15, v19 // (op 15089) 
v_add_f32 v6, v16, v20 // (op 15089) 
v_add_f32 v7, v17, v21 // (op 15089) 


v_cvt_f16_f32 v14, v4 // (op 15090) 
v_cvt_f16_f32 v15, v5 // (op 15090) 
v_cvt_f16_f32 v16, v6 // (op 15090) 
v_cvt_f16_f32 v17, v7 // (op 15090) 


v_pack_b32_f16 v4, v14, v15 // (op 15051) 
v_pack_b32_f16 v5, v16, v17 // (op 15051) 

s_lshl_b32 s0, s4, 5 // (op 15051) 
s_mul_i32 s1, 5, s0 // (op 15051) 
s_add_u32 s1, s1, 32 // (op 15051) 

v_add_u32 v0, v13, s1 // (op 15051) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15051) Store value





s_lshl_b32 s0, s2, 5 // (op 15096) 
s_mul_i32 s1, 6, s0 // (op 15096) 
s_add_u32 s1, s1, 32 // (op 15096) 

v_add_u32 v0, v12, s1 // (op 15096) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15096) Load value

v_accvgpr_read v14, a72 // (op 15133) 
v_accvgpr_read v15, a73 // (op 15133) 
v_accvgpr_read v16, a74 // (op 15133) 
v_accvgpr_read v17, a75 // (op 15133) 
v_mul_f32 v18, v11, v14 // (op 15133) 
v_mul_f32 v19, v11, v15 // (op 15133) 
v_mul_f32 v20, v11, v16 // (op 15133) 
v_mul_f32 v21, v11, v17 // (op 15133) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15093) 
v_lshrrev_b32 v7, 16, v4 // (op 15093) 
v_cvt_f32_f16 v7, v7 // (op 15093) 
v_mul_f32 v14, v9, v6 // (op 15093) 
v_mul_f32 v15, v9, v7 // (op 15093) 
v_cvt_f32_f16 v6, v5 // (op 15093) 
v_lshrrev_b32 v7, 16, v5 // (op 15093) 
v_cvt_f32_f16 v7, v7 // (op 15093) 
v_mul_f32 v16, v9, v6 // (op 15093) 
v_mul_f32 v17, v9, v7 // (op 15093) 


v_add_f32 v4, v14, v18 // (op 15094) 
v_add_f32 v5, v15, v19 // (op 15094) 
v_add_f32 v6, v16, v20 // (op 15094) 
v_add_f32 v7, v17, v21 // (op 15094) 


v_cvt_f16_f32 v14, v4 // (op 15095) 
v_cvt_f16_f32 v15, v5 // (op 15095) 
v_cvt_f16_f32 v16, v6 // (op 15095) 
v_cvt_f16_f32 v17, v7 // (op 15095) 


v_pack_b32_f16 v4, v14, v15 // (op 15053) 
v_pack_b32_f16 v5, v16, v17 // (op 15053) 

s_lshl_b32 s0, s4, 5 // (op 15053) 
s_mul_i32 s1, 6, s0 // (op 15053) 
s_add_u32 s1, s1, 32 // (op 15053) 

v_add_u32 v0, v13, s1 // (op 15053) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15053) Store value





s_lshl_b32 s0, s2, 5 // (op 15101) 
s_mul_i32 s1, 7, s0 // (op 15101) 
s_add_u32 s1, s1, 32 // (op 15101) 

v_add_u32 v0, v12, s1 // (op 15101) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15101) Load value

v_accvgpr_read v14, a76 // (op 15135) 
v_accvgpr_read v15, a77 // (op 15135) 
v_accvgpr_read v16, a78 // (op 15135) 
v_accvgpr_read v17, a79 // (op 15135) 
v_mul_f32 v18, v11, v14 // (op 15135) 
v_mul_f32 v19, v11, v15 // (op 15135) 
v_mul_f32 v20, v11, v16 // (op 15135) 
v_mul_f32 v21, v11, v17 // (op 15135) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15098) 
v_lshrrev_b32 v7, 16, v4 // (op 15098) 
v_cvt_f32_f16 v7, v7 // (op 15098) 
v_mul_f32 v14, v9, v6 // (op 15098) 
v_mul_f32 v15, v9, v7 // (op 15098) 
v_cvt_f32_f16 v6, v5 // (op 15098) 
v_lshrrev_b32 v7, 16, v5 // (op 15098) 
v_cvt_f32_f16 v7, v7 // (op 15098) 
v_mul_f32 v16, v9, v6 // (op 15098) 
v_mul_f32 v17, v9, v7 // (op 15098) 


v_add_f32 v4, v14, v18 // (op 15099) 
v_add_f32 v5, v15, v19 // (op 15099) 
v_add_f32 v6, v16, v20 // (op 15099) 
v_add_f32 v7, v17, v21 // (op 15099) 


v_cvt_f16_f32 v14, v4 // (op 15100) 
v_cvt_f16_f32 v15, v5 // (op 15100) 
v_cvt_f16_f32 v16, v6 // (op 15100) 
v_cvt_f16_f32 v17, v7 // (op 15100) 


v_pack_b32_f16 v4, v14, v15 // (op 15055) 
v_pack_b32_f16 v5, v16, v17 // (op 15055) 

s_lshl_b32 s0, s4, 5 // (op 15055) 
s_mul_i32 s1, 7, s0 // (op 15055) 
s_add_u32 s1, s1, 32 // (op 15055) 

v_add_u32 v0, v13, s1 // (op 15055) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15055) Store value





s_lshl_b32 s0, s2, 8 // (op 15106) 
s_add_u32 s0, s0, 32 // (op 15106) 
v_add_u32 v0, v12, s0 // (op 15106) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15106) Load value

v_accvgpr_read v14, a80 // (op 15137) 
v_accvgpr_read v15, a81 // (op 15137) 
v_accvgpr_read v16, a82 // (op 15137) 
v_accvgpr_read v17, a83 // (op 15137) 
v_mul_f32 v18, v11, v14 // (op 15137) 
v_mul_f32 v19, v11, v15 // (op 15137) 
v_mul_f32 v20, v11, v16 // (op 15137) 
v_mul_f32 v21, v11, v17 // (op 15137) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15103) 
v_lshrrev_b32 v7, 16, v4 // (op 15103) 
v_cvt_f32_f16 v7, v7 // (op 15103) 
v_mul_f32 v14, v9, v6 // (op 15103) 
v_mul_f32 v15, v9, v7 // (op 15103) 
v_cvt_f32_f16 v6, v5 // (op 15103) 
v_lshrrev_b32 v7, 16, v5 // (op 15103) 
v_cvt_f32_f16 v7, v7 // (op 15103) 
v_mul_f32 v16, v9, v6 // (op 15103) 
v_mul_f32 v17, v9, v7 // (op 15103) 


v_add_f32 v4, v14, v18 // (op 15104) 
v_add_f32 v5, v15, v19 // (op 15104) 
v_add_f32 v6, v16, v20 // (op 15104) 
v_add_f32 v7, v17, v21 // (op 15104) 


v_cvt_f16_f32 v14, v4 // (op 15105) 
v_cvt_f16_f32 v15, v5 // (op 15105) 
v_cvt_f16_f32 v16, v6 // (op 15105) 
v_cvt_f16_f32 v17, v7 // (op 15105) 


v_pack_b32_f16 v4, v14, v15 // (op 15057) 
v_pack_b32_f16 v5, v16, v17 // (op 15057) 

s_lshl_b32 s0, s4, 8 // (op 15057) 
s_add_u32 s0, s0, 32 // (op 15057) 
v_add_u32 v0, v13, s0 // (op 15057) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15057) Store value





s_lshl_b32 s0, s2, 5 // (op 15111) 
s_mul_i32 s1, 9, s0 // (op 15111) 
s_add_u32 s1, s1, 32 // (op 15111) 

v_add_u32 v0, v12, s1 // (op 15111) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15111) Load value

v_accvgpr_read v14, a84 // (op 15139) 
v_accvgpr_read v15, a85 // (op 15139) 
v_accvgpr_read v16, a86 // (op 15139) 
v_accvgpr_read v17, a87 // (op 15139) 
v_mul_f32 v18, v11, v14 // (op 15139) 
v_mul_f32 v19, v11, v15 // (op 15139) 
v_mul_f32 v20, v11, v16 // (op 15139) 
v_mul_f32 v21, v11, v17 // (op 15139) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15108) 
v_lshrrev_b32 v7, 16, v4 // (op 15108) 
v_cvt_f32_f16 v7, v7 // (op 15108) 
v_mul_f32 v14, v9, v6 // (op 15108) 
v_mul_f32 v15, v9, v7 // (op 15108) 
v_cvt_f32_f16 v6, v5 // (op 15108) 
v_lshrrev_b32 v7, 16, v5 // (op 15108) 
v_cvt_f32_f16 v7, v7 // (op 15108) 
v_mul_f32 v16, v9, v6 // (op 15108) 
v_mul_f32 v17, v9, v7 // (op 15108) 


v_add_f32 v4, v14, v18 // (op 15109) 
v_add_f32 v5, v15, v19 // (op 15109) 
v_add_f32 v6, v16, v20 // (op 15109) 
v_add_f32 v7, v17, v21 // (op 15109) 


v_cvt_f16_f32 v14, v4 // (op 15110) 
v_cvt_f16_f32 v15, v5 // (op 15110) 
v_cvt_f16_f32 v16, v6 // (op 15110) 
v_cvt_f16_f32 v17, v7 // (op 15110) 


v_pack_b32_f16 v4, v14, v15 // (op 15059) 
v_pack_b32_f16 v5, v16, v17 // (op 15059) 

s_lshl_b32 s0, s4, 5 // (op 15059) 
s_mul_i32 s1, 9, s0 // (op 15059) 
s_add_u32 s1, s1, 32 // (op 15059) 

v_add_u32 v0, v13, s1 // (op 15059) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15059) Store value





s_lshl_b32 s0, s2, 5 // (op 15116) 
s_mul_i32 s1, 10, s0 // (op 15116) 
s_add_u32 s1, s1, 32 // (op 15116) 

v_add_u32 v0, v12, s1 // (op 15116) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15116) Load value

v_accvgpr_read v14, a88 // (op 15141) 
v_accvgpr_read v15, a89 // (op 15141) 
v_accvgpr_read v16, a90 // (op 15141) 
v_accvgpr_read v17, a91 // (op 15141) 
v_mul_f32 v18, v11, v14 // (op 15141) 
v_mul_f32 v19, v11, v15 // (op 15141) 
v_mul_f32 v20, v11, v16 // (op 15141) 
v_mul_f32 v21, v11, v17 // (op 15141) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15113) 
v_lshrrev_b32 v7, 16, v4 // (op 15113) 
v_cvt_f32_f16 v7, v7 // (op 15113) 
v_mul_f32 v14, v9, v6 // (op 15113) 
v_mul_f32 v15, v9, v7 // (op 15113) 
v_cvt_f32_f16 v6, v5 // (op 15113) 
v_lshrrev_b32 v7, 16, v5 // (op 15113) 
v_cvt_f32_f16 v7, v7 // (op 15113) 
v_mul_f32 v16, v9, v6 // (op 15113) 
v_mul_f32 v17, v9, v7 // (op 15113) 


v_add_f32 v4, v14, v18 // (op 15114) 
v_add_f32 v5, v15, v19 // (op 15114) 
v_add_f32 v6, v16, v20 // (op 15114) 
v_add_f32 v7, v17, v21 // (op 15114) 


v_cvt_f16_f32 v14, v4 // (op 15115) 
v_cvt_f16_f32 v15, v5 // (op 15115) 
v_cvt_f16_f32 v16, v6 // (op 15115) 
v_cvt_f16_f32 v17, v7 // (op 15115) 


v_pack_b32_f16 v4, v14, v15 // (op 15061) 
v_pack_b32_f16 v5, v16, v17 // (op 15061) 

s_lshl_b32 s0, s4, 5 // (op 15061) 
s_mul_i32 s1, 10, s0 // (op 15061) 
s_add_u32 s1, s1, 32 // (op 15061) 

v_add_u32 v0, v13, s1 // (op 15061) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15061) Store value





s_lshl_b32 s0, s2, 5 // (op 15121) 
s_mul_i32 s1, 11, s0 // (op 15121) 
s_add_u32 s1, s1, 32 // (op 15121) 

v_add_u32 v0, v12, s1 // (op 15121) 


v_accvgpr_read v4, a92 // (op 15143) 
v_accvgpr_read v5, a93 // (op 15143) 
v_accvgpr_read v6, a94 // (op 15143) 
v_accvgpr_read v7, a95 // (op 15143) 
v_mul_f32 v14, v11, v4 // (op 15143) 
v_mul_f32 v15, v11, v5 // (op 15143) 
v_mul_f32 v16, v11, v6 // (op 15143) 
v_mul_f32 v17, v11, v7 // (op 15143) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 15121) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15118) 
v_lshrrev_b32 v7, 16, v4 // (op 15118) 
v_cvt_f32_f16 v7, v7 // (op 15118) 
v_mul_f32 v18, v9, v6 // (op 15118) 
v_mul_f32 v19, v9, v7 // (op 15118) 
v_cvt_f32_f16 v6, v5 // (op 15118) 
v_lshrrev_b32 v7, 16, v5 // (op 15118) 
v_cvt_f32_f16 v7, v7 // (op 15118) 
v_mul_f32 v20, v9, v6 // (op 15118) 
v_mul_f32 v21, v9, v7 // (op 15118) 


v_add_f32 v4, v18, v14 // (op 15119) 
v_add_f32 v5, v19, v15 // (op 15119) 
v_add_f32 v6, v20, v16 // (op 15119) 
v_add_f32 v7, v21, v17 // (op 15119) 


v_cvt_f16_f32 v14, v4 // (op 15120) 
v_cvt_f16_f32 v15, v5 // (op 15120) 
v_cvt_f16_f32 v16, v6 // (op 15120) 
v_cvt_f16_f32 v17, v7 // (op 15120) 


v_pack_b32_f16 v4, v14, v15 // (op 15063) 
v_pack_b32_f16 v5, v16, v17 // (op 15063) 

s_lshl_b32 s0, s4, 5 // (op 15063) 
s_mul_i32 s1, 11, s0 // (op 15063) 
s_add_u32 s1, s1, 32 // (op 15063) 

v_add_u32 v0, v13, s1 // (op 15063) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 15063) Store value





v_mov_b32 v0, v12 // (op 129) reg expression

v_accvgpr_read v4, a0 // (op 125) 
v_accvgpr_read v5, a1 // (op 125) 
v_accvgpr_read v6, a2 // (op 125) 
v_accvgpr_read v7, a3 // (op 125) 
v_mul_f32 v14, v11, v4 // (op 125) 
v_mul_f32 v15, v11, v5 // (op 125) 
v_mul_f32 v16, v11, v6 // (op 125) 
v_mul_f32 v17, v11, v7 // (op 125) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 129) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 130) 
v_lshrrev_b32 v7, 16, v4 // (op 130) 
v_cvt_f32_f16 v7, v7 // (op 130) 
v_mul_f32 v18, v9, v6 // (op 130) 
v_mul_f32 v19, v9, v7 // (op 130) 
v_cvt_f32_f16 v6, v5 // (op 130) 
v_lshrrev_b32 v7, 16, v5 // (op 130) 
v_cvt_f32_f16 v7, v7 // (op 130) 
v_mul_f32 v20, v9, v6 // (op 130) 
v_mul_f32 v21, v9, v7 // (op 130) 


v_add_f32 v4, v18, v14 // (op 126) 
v_add_f32 v5, v19, v15 // (op 126) 
v_add_f32 v6, v20, v16 // (op 126) 
v_add_f32 v7, v21, v17 // (op 126) 


v_cvt_f16_f32 v14, v4 // (op 127) 
v_cvt_f16_f32 v15, v5 // (op 127) 
v_cvt_f16_f32 v16, v6 // (op 127) 
v_cvt_f16_f32 v17, v7 // (op 127) 


v_pack_b32_f16 v4, v14, v15 // (op 128) 
v_pack_b32_f16 v5, v16, v17 // (op 128) 

v_mov_b32 v0, v13 // (op 128) reg expression

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 128) Store value





s_lshl_b32 s0, s2, 5 // (op 14766) 
v_add_u32 v0, v12, s0 // (op 14766) 


v_accvgpr_read v4, a4 // (op 14762) 
v_accvgpr_read v5, a5 // (op 14762) 
v_accvgpr_read v6, a6 // (op 14762) 
v_accvgpr_read v7, a7 // (op 14762) 
v_mul_f32 v14, v11, v4 // (op 14762) 
v_mul_f32 v15, v11, v5 // (op 14762) 
v_mul_f32 v16, v11, v6 // (op 14762) 
v_mul_f32 v17, v11, v7 // (op 14762) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 14766) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14767) 
v_lshrrev_b32 v7, 16, v4 // (op 14767) 
v_cvt_f32_f16 v7, v7 // (op 14767) 
v_mul_f32 v18, v9, v6 // (op 14767) 
v_mul_f32 v19, v9, v7 // (op 14767) 
v_cvt_f32_f16 v6, v5 // (op 14767) 
v_lshrrev_b32 v7, 16, v5 // (op 14767) 
v_cvt_f32_f16 v7, v7 // (op 14767) 
v_mul_f32 v20, v9, v6 // (op 14767) 
v_mul_f32 v21, v9, v7 // (op 14767) 


v_add_f32 v4, v18, v14 // (op 14763) 
v_add_f32 v5, v19, v15 // (op 14763) 
v_add_f32 v6, v20, v16 // (op 14763) 
v_add_f32 v7, v21, v17 // (op 14763) 


v_cvt_f16_f32 v14, v4 // (op 14764) 
v_cvt_f16_f32 v15, v5 // (op 14764) 
v_cvt_f16_f32 v16, v6 // (op 14764) 
v_cvt_f16_f32 v17, v7 // (op 14764) 


v_pack_b32_f16 v4, v14, v15 // (op 14765) 
v_pack_b32_f16 v5, v16, v17 // (op 14765) 

s_lshl_b32 s0, s4, 5 // (op 14765) 
v_add_u32 v0, v13, s0 // (op 14765) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14765) Store value





s_lshl_b32 s0, s2, 6 // (op 14780) 
v_add_u32 v0, v12, s0 // (op 14780) 


v_accvgpr_read v4, a8 // (op 14776) 
v_accvgpr_read v5, a9 // (op 14776) 
v_accvgpr_read v6, a10 // (op 14776) 
v_accvgpr_read v7, a11 // (op 14776) 
v_mul_f32 v14, v11, v4 // (op 14776) 
v_mul_f32 v15, v11, v5 // (op 14776) 
v_mul_f32 v16, v11, v6 // (op 14776) 
v_mul_f32 v17, v11, v7 // (op 14776) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 14780) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14781) 
v_lshrrev_b32 v7, 16, v4 // (op 14781) 
v_cvt_f32_f16 v7, v7 // (op 14781) 
v_mul_f32 v18, v9, v6 // (op 14781) 
v_mul_f32 v19, v9, v7 // (op 14781) 
v_cvt_f32_f16 v6, v5 // (op 14781) 
v_lshrrev_b32 v7, 16, v5 // (op 14781) 
v_cvt_f32_f16 v7, v7 // (op 14781) 
v_mul_f32 v20, v9, v6 // (op 14781) 
v_mul_f32 v21, v9, v7 // (op 14781) 


v_add_f32 v4, v18, v14 // (op 14777) 
v_add_f32 v5, v19, v15 // (op 14777) 
v_add_f32 v6, v20, v16 // (op 14777) 
v_add_f32 v7, v21, v17 // (op 14777) 


v_cvt_f16_f32 v14, v4 // (op 14778) 
v_cvt_f16_f32 v15, v5 // (op 14778) 
v_cvt_f16_f32 v16, v6 // (op 14778) 
v_cvt_f16_f32 v17, v7 // (op 14778) 


v_pack_b32_f16 v4, v14, v15 // (op 14779) 
v_pack_b32_f16 v5, v16, v17 // (op 14779) 

s_lshl_b32 s0, s4, 6 // (op 14779) 
v_add_u32 v0, v13, s0 // (op 14779) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14779) Store value





s_lshl_b32 s0, s2, 5 // (op 14794) 
s_mul_i32 s1, 3, s0 // (op 14794) 

v_add_u32 v0, v12, s1 // (op 14794) 


v_accvgpr_read v4, a12 // (op 14790) 
v_accvgpr_read v5, a13 // (op 14790) 
v_accvgpr_read v6, a14 // (op 14790) 
v_accvgpr_read v7, a15 // (op 14790) 
v_mul_f32 v14, v11, v4 // (op 14790) 
v_mul_f32 v15, v11, v5 // (op 14790) 
v_mul_f32 v16, v11, v6 // (op 14790) 
v_mul_f32 v17, v11, v7 // (op 14790) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 14794) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14795) 
v_lshrrev_b32 v7, 16, v4 // (op 14795) 
v_cvt_f32_f16 v7, v7 // (op 14795) 
v_mul_f32 v18, v9, v6 // (op 14795) 
v_mul_f32 v19, v9, v7 // (op 14795) 
v_cvt_f32_f16 v6, v5 // (op 14795) 
v_lshrrev_b32 v7, 16, v5 // (op 14795) 
v_cvt_f32_f16 v7, v7 // (op 14795) 
v_mul_f32 v20, v9, v6 // (op 14795) 
v_mul_f32 v21, v9, v7 // (op 14795) 


v_add_f32 v4, v18, v14 // (op 14791) 
v_add_f32 v5, v19, v15 // (op 14791) 
v_add_f32 v6, v20, v16 // (op 14791) 
v_add_f32 v7, v21, v17 // (op 14791) 


v_cvt_f16_f32 v14, v4 // (op 14792) 
v_cvt_f16_f32 v15, v5 // (op 14792) 
v_cvt_f16_f32 v16, v6 // (op 14792) 
v_cvt_f16_f32 v17, v7 // (op 14792) 


v_pack_b32_f16 v4, v14, v15 // (op 14793) 
v_pack_b32_f16 v5, v16, v17 // (op 14793) 

s_lshl_b32 s0, s4, 5 // (op 14793) 
s_mul_i32 s1, 3, s0 // (op 14793) 

v_add_u32 v0, v13, s1 // (op 14793) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14793) Store value





s_lshl_b32 s0, s2, 7 // (op 14808) 
v_add_u32 v0, v12, s0 // (op 14808) 


v_accvgpr_read v4, a16 // (op 14804) 
v_accvgpr_read v5, a17 // (op 14804) 
v_accvgpr_read v6, a18 // (op 14804) 
v_accvgpr_read v7, a19 // (op 14804) 
v_mul_f32 v14, v11, v4 // (op 14804) 
v_mul_f32 v15, v11, v5 // (op 14804) 
v_mul_f32 v16, v11, v6 // (op 14804) 
v_mul_f32 v17, v11, v7 // (op 14804) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 14808) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14809) 
v_lshrrev_b32 v7, 16, v4 // (op 14809) 
v_cvt_f32_f16 v7, v7 // (op 14809) 
v_mul_f32 v18, v9, v6 // (op 14809) 
v_mul_f32 v19, v9, v7 // (op 14809) 
v_cvt_f32_f16 v6, v5 // (op 14809) 
v_lshrrev_b32 v7, 16, v5 // (op 14809) 
v_cvt_f32_f16 v7, v7 // (op 14809) 
v_mul_f32 v20, v9, v6 // (op 14809) 
v_mul_f32 v21, v9, v7 // (op 14809) 


v_add_f32 v4, v18, v14 // (op 14805) 
v_add_f32 v5, v19, v15 // (op 14805) 
v_add_f32 v6, v20, v16 // (op 14805) 
v_add_f32 v7, v21, v17 // (op 14805) 


v_cvt_f16_f32 v14, v4 // (op 14806) 
v_cvt_f16_f32 v15, v5 // (op 14806) 
v_cvt_f16_f32 v16, v6 // (op 14806) 
v_cvt_f16_f32 v17, v7 // (op 14806) 


v_pack_b32_f16 v4, v14, v15 // (op 14807) 
v_pack_b32_f16 v5, v16, v17 // (op 14807) 

s_lshl_b32 s0, s4, 7 // (op 14807) 
v_add_u32 v0, v13, s0 // (op 14807) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14807) Store value





s_lshl_b32 s0, s2, 5 // (op 14822) 
s_mul_i32 s1, 5, s0 // (op 14822) 

v_add_u32 v0, v12, s1 // (op 14822) 


v_accvgpr_read v4, a20 // (op 14818) 
v_accvgpr_read v5, a21 // (op 14818) 
v_accvgpr_read v6, a22 // (op 14818) 
v_accvgpr_read v7, a23 // (op 14818) 
v_mul_f32 v14, v11, v4 // (op 14818) 
v_mul_f32 v15, v11, v5 // (op 14818) 
v_mul_f32 v16, v11, v6 // (op 14818) 
v_mul_f32 v17, v11, v7 // (op 14818) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 14822) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14823) 
v_lshrrev_b32 v7, 16, v4 // (op 14823) 
v_cvt_f32_f16 v7, v7 // (op 14823) 
v_mul_f32 v18, v9, v6 // (op 14823) 
v_mul_f32 v19, v9, v7 // (op 14823) 
v_cvt_f32_f16 v6, v5 // (op 14823) 
v_lshrrev_b32 v7, 16, v5 // (op 14823) 
v_cvt_f32_f16 v7, v7 // (op 14823) 
v_mul_f32 v20, v9, v6 // (op 14823) 
v_mul_f32 v21, v9, v7 // (op 14823) 


v_add_f32 v4, v18, v14 // (op 14819) 
v_add_f32 v5, v19, v15 // (op 14819) 
v_add_f32 v6, v20, v16 // (op 14819) 
v_add_f32 v7, v21, v17 // (op 14819) 


v_cvt_f16_f32 v14, v4 // (op 14820) 
v_cvt_f16_f32 v15, v5 // (op 14820) 
v_cvt_f16_f32 v16, v6 // (op 14820) 
v_cvt_f16_f32 v17, v7 // (op 14820) 


v_pack_b32_f16 v4, v14, v15 // (op 14821) 
v_pack_b32_f16 v5, v16, v17 // (op 14821) 

s_lshl_b32 s0, s4, 5 // (op 14821) 
s_mul_i32 s1, 5, s0 // (op 14821) 

v_add_u32 v0, v13, s1 // (op 14821) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14821) Store value





s_lshl_b32 s0, s2, 5 // (op 14836) 
s_mul_i32 s1, 6, s0 // (op 14836) 

v_add_u32 v0, v12, s1 // (op 14836) 


v_accvgpr_read v4, a24 // (op 14832) 
v_accvgpr_read v5, a25 // (op 14832) 
v_accvgpr_read v6, a26 // (op 14832) 
v_accvgpr_read v7, a27 // (op 14832) 
v_mul_f32 v14, v11, v4 // (op 14832) 
v_mul_f32 v15, v11, v5 // (op 14832) 
v_mul_f32 v16, v11, v6 // (op 14832) 
v_mul_f32 v17, v11, v7 // (op 14832) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 14836) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14837) 
v_lshrrev_b32 v7, 16, v4 // (op 14837) 
v_cvt_f32_f16 v7, v7 // (op 14837) 
v_mul_f32 v18, v9, v6 // (op 14837) 
v_mul_f32 v19, v9, v7 // (op 14837) 
v_cvt_f32_f16 v6, v5 // (op 14837) 
v_lshrrev_b32 v7, 16, v5 // (op 14837) 
v_cvt_f32_f16 v7, v7 // (op 14837) 
v_mul_f32 v20, v9, v6 // (op 14837) 
v_mul_f32 v21, v9, v7 // (op 14837) 


v_add_f32 v4, v18, v14 // (op 14833) 
v_add_f32 v5, v19, v15 // (op 14833) 
v_add_f32 v6, v20, v16 // (op 14833) 
v_add_f32 v7, v21, v17 // (op 14833) 


v_cvt_f16_f32 v14, v4 // (op 14834) 
v_cvt_f16_f32 v15, v5 // (op 14834) 
v_cvt_f16_f32 v16, v6 // (op 14834) 
v_cvt_f16_f32 v17, v7 // (op 14834) 


v_pack_b32_f16 v4, v14, v15 // (op 14835) 
v_pack_b32_f16 v5, v16, v17 // (op 14835) 

s_lshl_b32 s0, s4, 5 // (op 14835) 
s_mul_i32 s1, 6, s0 // (op 14835) 

v_add_u32 v0, v13, s1 // (op 14835) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14835) Store value





s_lshl_b32 s0, s2, 5 // (op 14850) 
s_mul_i32 s1, 7, s0 // (op 14850) 

v_add_u32 v0, v12, s1 // (op 14850) 


v_accvgpr_read v4, a28 // (op 14846) 
v_accvgpr_read v5, a29 // (op 14846) 
v_accvgpr_read v6, a30 // (op 14846) 
v_accvgpr_read v7, a31 // (op 14846) 
v_mul_f32 v14, v11, v4 // (op 14846) 
v_mul_f32 v15, v11, v5 // (op 14846) 
v_mul_f32 v16, v11, v6 // (op 14846) 
v_mul_f32 v17, v11, v7 // (op 14846) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 14850) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14851) 
v_lshrrev_b32 v7, 16, v4 // (op 14851) 
v_cvt_f32_f16 v7, v7 // (op 14851) 
v_mul_f32 v18, v9, v6 // (op 14851) 
v_mul_f32 v19, v9, v7 // (op 14851) 
v_cvt_f32_f16 v6, v5 // (op 14851) 
v_lshrrev_b32 v7, 16, v5 // (op 14851) 
v_cvt_f32_f16 v7, v7 // (op 14851) 
v_mul_f32 v20, v9, v6 // (op 14851) 
v_mul_f32 v21, v9, v7 // (op 14851) 


v_add_f32 v4, v18, v14 // (op 14847) 
v_add_f32 v5, v19, v15 // (op 14847) 
v_add_f32 v6, v20, v16 // (op 14847) 
v_add_f32 v7, v21, v17 // (op 14847) 


v_cvt_f16_f32 v14, v4 // (op 14848) 
v_cvt_f16_f32 v15, v5 // (op 14848) 
v_cvt_f16_f32 v16, v6 // (op 14848) 
v_cvt_f16_f32 v17, v7 // (op 14848) 


v_pack_b32_f16 v4, v14, v15 // (op 14849) 
v_pack_b32_f16 v5, v16, v17 // (op 14849) 

s_lshl_b32 s0, s4, 5 // (op 14849) 
s_mul_i32 s1, 7, s0 // (op 14849) 

v_add_u32 v0, v13, s1 // (op 14849) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14849) Store value





s_lshl_b32 s0, s2, 8 // (op 14864) 
v_add_u32 v0, v12, s0 // (op 14864) 


v_accvgpr_read v4, a32 // (op 14860) 
v_accvgpr_read v5, a33 // (op 14860) 
v_accvgpr_read v6, a34 // (op 14860) 
v_accvgpr_read v7, a35 // (op 14860) 
v_mul_f32 v14, v11, v4 // (op 14860) 
v_mul_f32 v15, v11, v5 // (op 14860) 
v_mul_f32 v16, v11, v6 // (op 14860) 
v_mul_f32 v17, v11, v7 // (op 14860) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 14864) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14865) 
v_lshrrev_b32 v7, 16, v4 // (op 14865) 
v_cvt_f32_f16 v7, v7 // (op 14865) 
v_mul_f32 v18, v9, v6 // (op 14865) 
v_mul_f32 v19, v9, v7 // (op 14865) 
v_cvt_f32_f16 v6, v5 // (op 14865) 
v_lshrrev_b32 v7, 16, v5 // (op 14865) 
v_cvt_f32_f16 v7, v7 // (op 14865) 
v_mul_f32 v20, v9, v6 // (op 14865) 
v_mul_f32 v21, v9, v7 // (op 14865) 


v_add_f32 v4, v18, v14 // (op 14861) 
v_add_f32 v5, v19, v15 // (op 14861) 
v_add_f32 v6, v20, v16 // (op 14861) 
v_add_f32 v7, v21, v17 // (op 14861) 


v_cvt_f16_f32 v14, v4 // (op 14862) 
v_cvt_f16_f32 v15, v5 // (op 14862) 
v_cvt_f16_f32 v16, v6 // (op 14862) 
v_cvt_f16_f32 v17, v7 // (op 14862) 


v_pack_b32_f16 v4, v14, v15 // (op 14863) 
v_pack_b32_f16 v5, v16, v17 // (op 14863) 

s_lshl_b32 s0, s4, 8 // (op 14863) 
v_add_u32 v0, v13, s0 // (op 14863) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14863) Store value





s_lshl_b32 s0, s2, 5 // (op 14878) 
s_mul_i32 s1, 9, s0 // (op 14878) 

v_add_u32 v0, v12, s1 // (op 14878) 


v_accvgpr_read v4, a36 // (op 14874) 
v_accvgpr_read v5, a37 // (op 14874) 
v_accvgpr_read v6, a38 // (op 14874) 
v_accvgpr_read v7, a39 // (op 14874) 
v_mul_f32 v14, v11, v4 // (op 14874) 
v_mul_f32 v15, v11, v5 // (op 14874) 
v_mul_f32 v16, v11, v6 // (op 14874) 
v_mul_f32 v17, v11, v7 // (op 14874) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 14878) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14879) 
v_lshrrev_b32 v7, 16, v4 // (op 14879) 
v_cvt_f32_f16 v7, v7 // (op 14879) 
v_mul_f32 v18, v9, v6 // (op 14879) 
v_mul_f32 v19, v9, v7 // (op 14879) 
v_cvt_f32_f16 v6, v5 // (op 14879) 
v_lshrrev_b32 v7, 16, v5 // (op 14879) 
v_cvt_f32_f16 v7, v7 // (op 14879) 
v_mul_f32 v20, v9, v6 // (op 14879) 
v_mul_f32 v21, v9, v7 // (op 14879) 


v_add_f32 v4, v18, v14 // (op 14875) 
v_add_f32 v5, v19, v15 // (op 14875) 
v_add_f32 v6, v20, v16 // (op 14875) 
v_add_f32 v7, v21, v17 // (op 14875) 


v_cvt_f16_f32 v14, v4 // (op 14876) 
v_cvt_f16_f32 v15, v5 // (op 14876) 
v_cvt_f16_f32 v16, v6 // (op 14876) 
v_cvt_f16_f32 v17, v7 // (op 14876) 


v_pack_b32_f16 v4, v14, v15 // (op 14877) 
v_pack_b32_f16 v5, v16, v17 // (op 14877) 

s_lshl_b32 s0, s4, 5 // (op 14877) 
s_mul_i32 s1, 9, s0 // (op 14877) 

v_add_u32 v0, v13, s1 // (op 14877) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14877) Store value





s_lshl_b32 s0, s2, 5 // (op 14892) 
s_mul_i32 s1, 10, s0 // (op 14892) 

v_add_u32 v0, v12, s1 // (op 14892) 


v_accvgpr_read v4, a40 // (op 14888) 
v_accvgpr_read v5, a41 // (op 14888) 
v_accvgpr_read v6, a42 // (op 14888) 
v_accvgpr_read v7, a43 // (op 14888) 
v_mul_f32 v14, v11, v4 // (op 14888) 
v_mul_f32 v15, v11, v5 // (op 14888) 
v_mul_f32 v16, v11, v6 // (op 14888) 
v_mul_f32 v17, v11, v7 // (op 14888) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 14892) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14893) 
v_lshrrev_b32 v7, 16, v4 // (op 14893) 
v_cvt_f32_f16 v7, v7 // (op 14893) 
v_mul_f32 v18, v9, v6 // (op 14893) 
v_mul_f32 v19, v9, v7 // (op 14893) 
v_cvt_f32_f16 v6, v5 // (op 14893) 
v_lshrrev_b32 v7, 16, v5 // (op 14893) 
v_cvt_f32_f16 v7, v7 // (op 14893) 
v_mul_f32 v20, v9, v6 // (op 14893) 
v_mul_f32 v21, v9, v7 // (op 14893) 


v_add_f32 v4, v18, v14 // (op 14889) 
v_add_f32 v5, v19, v15 // (op 14889) 
v_add_f32 v6, v20, v16 // (op 14889) 
v_add_f32 v7, v21, v17 // (op 14889) 


v_cvt_f16_f32 v14, v4 // (op 14890) 
v_cvt_f16_f32 v15, v5 // (op 14890) 
v_cvt_f16_f32 v16, v6 // (op 14890) 
v_cvt_f16_f32 v17, v7 // (op 14890) 


v_pack_b32_f16 v4, v14, v15 // (op 14891) 
v_pack_b32_f16 v5, v16, v17 // (op 14891) 

s_lshl_b32 s0, s4, 5 // (op 14891) 
s_mul_i32 s1, 10, s0 // (op 14891) 

v_add_u32 v0, v13, s1 // (op 14891) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14891) Store value





s_lshl_b32 s0, s2, 5 // (op 14906) 
s_mul_i32 s1, 11, s0 // (op 14906) 

v_add_u32 v0, v12, s1 // (op 14906) 


v_accvgpr_read v4, a44 // (op 14902) 
v_accvgpr_read v5, a45 // (op 14902) 
v_accvgpr_read v6, a46 // (op 14902) 
v_accvgpr_read v7, a47 // (op 14902) 
v_mul_f32 v14, v11, v4 // (op 14902) 
v_mul_f32 v15, v11, v5 // (op 14902) 
v_mul_f32 v16, v11, v6 // (op 14902) 
v_mul_f32 v17, v11, v7 // (op 14902) 

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 14906) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14907) 
v_lshrrev_b32 v7, 16, v4 // (op 14907) 
v_cvt_f32_f16 v7, v7 // (op 14907) 
v_mul_f32 v18, v9, v6 // (op 14907) 
v_mul_f32 v19, v9, v7 // (op 14907) 
v_cvt_f32_f16 v6, v5 // (op 14907) 
v_lshrrev_b32 v7, 16, v5 // (op 14907) 
v_cvt_f32_f16 v7, v7 // (op 14907) 
v_mul_f32 v20, v9, v6 // (op 14907) 
v_mul_f32 v21, v9, v7 // (op 14907) 


v_add_f32 v4, v18, v14 // (op 14903) 
v_add_f32 v5, v19, v15 // (op 14903) 
v_add_f32 v6, v20, v16 // (op 14903) 
v_add_f32 v7, v21, v17 // (op 14903) 


v_cvt_f16_f32 v14, v4 // (op 14904) 
v_cvt_f16_f32 v15, v5 // (op 14904) 
v_cvt_f16_f32 v16, v6 // (op 14904) 
v_cvt_f16_f32 v17, v7 // (op 14904) 


v_pack_b32_f16 v4, v14, v15 // (op 14905) 
v_pack_b32_f16 v5, v16, v17 // (op 14905) 

s_lshl_b32 s0, s4, 5 // (op 14905) 
s_mul_i32 s1, 11, s0 // (op 14905) 

v_add_u32 v0, v13, s1 // (op 14905) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14905) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__ConditionalBottom_0_beta_116:
























































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM_
.LRR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__end:
.size RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM_, .LRR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM__end-RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM_
  .amdhsa_next_free_vgpr 388
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 114688 // lds bytes
  .amdhsa_accum_offset 196
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
            { type: int64, semantic: RR_NumTilesN },
            { type: int64, semantic: SizeFree0 },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: address, semantic: AddressC },
            { type: int64, semantic: SizeFree1 },
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
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [256, 192, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 12]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM_
    .symbol: RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x192x128_UR_4_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 114688
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 78
    .vgpr_count: 194
    .agpr_count: 192
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
      - .name: BitwiseXor_18
        .size: 8
        .offset: 16
        .expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 32
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 156, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_24
        .size: 4
        .offset: 36
        .expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 40
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 44
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_22
        .size: 4
        .offset: 48
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_25, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 168, size: 4}, rhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
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
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 152
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 156, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 156
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_25, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 168, size: 4}, rhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_25
        .size: 4
        .offset: 160
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_23
        .size: 4
        .offset: 164
        .expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: WGM_19
        .size: 4
        .offset: 168
        .expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 172
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 176
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

