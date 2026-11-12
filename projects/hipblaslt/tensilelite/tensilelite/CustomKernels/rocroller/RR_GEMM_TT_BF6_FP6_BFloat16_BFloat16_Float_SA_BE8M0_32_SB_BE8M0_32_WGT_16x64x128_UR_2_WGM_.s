// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_,@function
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s11 // (op 1135) 
s_mul_hi_i32 s1, s16, s11 // (op 1324) 
v_lshrrev_b32 v0, 7, v1 // (op 1192) 
v_and_b32 v3, 15, v1 // (op 1192) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1192) 


v_and_b32 v0, 63, v1 // (op 1192) 
v_lshrrev_b32 v3, 4, v0 // (op 1192) 

v_lshlrev_b32 v0, 5, v3 // (op 1192) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 1192) 


v_mul_lo_u32 v0, v3, 6 // (op 1192) 

v_lshrrev_b32 v3, 3, v0 // (op 1192) 

v_lshrrev_b32 v0, 7, v1 // (op 1350) 
v_and_b32 v4, 15, v1 // (op 1350) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1350) 


v_and_b32 v0, 63, v1 // (op 1350) 
v_lshrrev_b32 v4, 4, v0 // (op 1350) 

v_lshlrev_b32 v0, 5, v4 // (op 1350) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 1350) 


v_mul_lo_u32 v0, v4, 6 // (op 1350) 

v_lshrrev_b32 v4, 3, v0 // (op 1350) 

v_lshrrev_b32 v0, 7, v1 // (op 1135) 
s_add_i32 s18, s0, s16 // (op 1135) 

v_and_b32 v5, 63, v1 // (op 1135) 
s_ashr_i32 s0, s18, 31 // (op 1135) 
v_lshrrev_b32 v6, 4, v5 // (op 1135) 
s_cmp_eq_i32 s11, 0 // (op 1135) 
s_mov_b32 s19, scc // (op 1135) 
v_lshlrev_b32 v7, 2, v6 // (op 1135) 

s_cmp_eq_i32 s19, 0 // (op 1135) 
s_cselect_b32 s51, 0, -1 // (op 1135) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1135) 


s_and_b32 s19, s12, 31 // (op 1135) 
v_lshrrev_b32 v0, 2, v5 // (op 1135) 

v_and_b32 v5, 3, v0 // (op 1135) 

v_and_b32 v0, 63, v1 // (op 1250) 
v_and_b32 v7, 48, v1 // (op 1250) 
v_sub_u32 v8, v0, v7 // (op 1250) 

v_lshrrev_b32 v7, 4, v0 // (op 1250) 

v_lshl_add_u32 v0, v7, 5, v8 // (op 1250) 


v_lshrrev_b32 v7, 6, v1 // (op 1250) 
v_and_b32 v8, 1, v7 // (op 1250) 

v_lshlrev_b32 v7, 4, v8 // (op 1250) 

v_lshl_add_u32 v8, v0, 6, v7 // (op 1250) 


v_mul_lo_u32 v0, v8, 6 // (op 1250) 
v_lshrrev_b32 v7, 3, v0 // (op 1250) 

v_lshrrev_b32 v0, 4, v8 // (op 1250) 

v_lshlrev_b32 v8, 2, v0 // (op 1250) 

v_add_u32 v0, v7, v8 // (op 1250) 


v_lshrrev_b32 v7, 6, v1 // (op 1308) 
v_and_b32 v8, 1, v7 // (op 1308) 

v_and_b32 v7, 15, v1 // (op 1308) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 1308) 


v_and_b32 v7, 63, v1 // (op 1308) 
v_lshrrev_b32 v8, 4, v7 // (op 1308) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 1308) 


v_lshrrev_b32 v8, 7, v1 // (op 1324) 
v_and_b32 v9, 63, v1 // (op 1324) 
v_and_b32 v10, 63, v1 // (op 1379) 
v_and_b32 v11, 48, v1 // (op 1379) 
v_sub_u32 v12, v10, v11 // (op 1379) 

v_lshrrev_b32 v11, 4, v10 // (op 1379) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 1379) 


v_lshrrev_b32 v11, 6, v1 // (op 1379) 
v_and_b32 v12, 1, v11 // (op 1379) 

v_lshlrev_b32 v11, 4, v12 // (op 1379) 

v_lshl_add_u32 v12, v10, 6, v11 // (op 1379) 


v_mul_lo_u32 v10, v12, 6 // (op 1379) 
v_lshrrev_b32 v11, 3, v10 // (op 1379) 

v_lshrrev_b32 v10, 4, v12 // (op 1379) 

v_lshlrev_b32 v12, 2, v10 // (op 1379) 

v_add_u32 v10, v11, v12 // (op 1379) 


v_lshrrev_b32 v11, 6, v1 // (op 1408) 
v_and_b32 v12, 1, v11 // (op 1408) 

v_and_b32 v11, 15, v1 // (op 1408) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 1408) 


v_and_b32 v11, 63, v1 // (op 1408) 
v_lshrrev_b32 v12, 4, v11 // (op 1408) 

v_lshl_add_u32 v11, v12, 6, v13 // (op 1408) 


s_lshl_b32 s52, 1, s19 // (op 1135) 
v_and_b32 v12, 3, v1 // (op 1135) 
s_add_i32 s53, s52, s51 // (op 1135) 


s_and_b32 s51, s0, s53 // (op 1135) 


s_add_i32 s0, s1, s16 // (op 1324) 

s_ashr_i32 s1, s0, 31 // (op 1324) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1135) 


s_add_i32 s52, s18, s51 // (op 1135) 


v_lshrrev_b32 v5, 6, v1 // (op 1135) 
s_ashr_i32 s18, s52, s19 // (op 1135) 


v_and_b32 v12, 1, v5 // (op 1135) 

s_lshl_b32 s19, s12, 24 // (op 1135) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1135) 


s_ashr_i32 s51, s19, 31 // (op 1135) 

s_xor_b32 s19, s18, s51 // (op 1135) 

s_sub_i32 s18, s19, s51 // (op 1135) 


s_mul_hi_i32 s19, s18, s13 // (op 1135) 
s_add_i32 s51, s19, s18 // (op 1135) 

s_ashr_i32 s19, s51, 31 // (op 1135) 
s_cmp_eq_i32 s13, 0 // (op 1135) 
s_mov_b32 s52, scc // (op 1135) 
s_cmp_eq_i32 s52, 0 // (op 1135) 
s_cselect_b32 s53, 0, -1 // (op 1135) 

s_and_b32 s52, s14, 31 // (op 1135) 
v_lshrrev_b32 v12, 4, v9 // (op 1324) 
s_cmp_eq_i32 s11, 0 // (op 1324) 
s_mov_b32 s54, scc // (op 1324) 
v_lshlrev_b32 v13, 2, v12 // (op 1324) 

s_cmp_eq_i32 s54, 0 // (op 1324) 
s_cselect_b32 s55, 0, -1 // (op 1324) 

s_lshl_b32 s54, 1, s52 // (op 1135) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 1324) 


v_lshrrev_b32 v8, 2, v9 // (op 1324) 

s_add_i32 s56, s54, s53 // (op 1135) 


s_and_b32 s53, s19, s56 // (op 1135) 


s_add_i32 s19, s51, s53 // (op 1135) 


s_ashr_i32 s51, s19, s52 // (op 1135) 


s_lshl_b32 s19, s14, 24 // (op 1135) 
s_ashr_i32 s52, s19, 31 // (op 1135) 

s_xor_b32 s19, s51, s52 // (op 1135) 

s_sub_i32 s51, s19, s52 // (op 1135) 


s_and_b32 s19, s12, 31 // (op 1324) 
s_lshl_b32 s52, 1, s19 // (op 1324) 
v_and_b32 v9, 3, v8 // (op 1324) 

s_add_i32 s53, s52, s55 // (op 1324) 


v_and_b32 v8, 3, v1 // (op 1324) 
s_and_b32 s52, s1, s53 // (op 1324) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1324) 


s_add_i32 s1, s0, s52 // (op 1324) 


v_lshrrev_b32 v8, 6, v1 // (op 1324) 
s_ashr_i32 s0, s1, s19 // (op 1324) 


v_and_b32 v9, 1, v8 // (op 1324) 

s_lshl_b32 s1, s12, 24 // (op 1324) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1324) 


s_ashr_i32 s19, s1, 31 // (op 1324) 

s_xor_b32 s1, s0, s19 // (op 1324) 

s_sub_i32 s0, s1, s19 // (op 1324) 


s_mul_hi_i32 s1, s0, s13 // (op 1324) 
s_add_i32 s19, s1, s0 // (op 1324) 

s_ashr_i32 s1, s19, 31 // (op 1324) 
s_cmp_eq_i32 s13, 0 // (op 1324) 
s_mov_b32 s52, scc // (op 1324) 
s_cmp_eq_i32 s52, 0 // (op 1324) 
s_cselect_b32 s53, 0, -1 // (op 1324) 

s_and_b32 s52, s14, 31 // (op 1324) 
s_lshl_b32 s54, 1, s52 // (op 1324) 
s_add_i32 s55, s54, s53 // (op 1324) 


s_and_b32 s53, s1, s55 // (op 1324) 


s_add_i32 s1, s19, s53 // (op 1324) 


s_ashr_i32 s19, s1, s52 // (op 1324) 


s_lshl_b32 s1, s14, 24 // (op 1324) 
s_ashr_i32 s52, s1, 31 // (op 1324) 

s_xor_b32 s1, s19, s52 // (op 1324) 

s_sub_i32 s19, s1, s52 // (op 1324) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 1135) 
s_sub_u32 s52, s16, s1 // (op 1135) 

s_mul_i32 s1, s51, s44 // (op 1135) 
s_add_u32 s1, s1, s52 // (op 1135) 

s_mul_hi_i32 s52, s16, s10 // (op 1135) 
s_add_i32 s53, s52, s16 // (op 1135) 

s_ashr_i32 s52, s53, 31 // (op 1135) 
s_cmp_eq_i32 s10, 0 // (op 1135) 
s_mov_b32 s54, scc // (op 1135) 
s_cmp_eq_i32 s54, 0 // (op 1135) 
s_cselect_b32 s55, 0, -1 // (op 1135) 

s_and_b32 s54, s47, 31 // (op 1135) 
s_lshl_b32 s56, 1, s54 // (op 1135) 
s_add_i32 s57, s56, s55 // (op 1135) 


s_and_b32 s55, s52, s57 // (op 1135) 


s_add_i32 s52, s53, s55 // (op 1135) 


s_ashr_i32 s53, s52, s54 // (op 1135) 


s_lshl_b32 s52, s47, 24 // (op 1135) 
s_ashr_i32 s54, s52, 31 // (op 1135) 

s_xor_b32 s52, s53, s54 // (op 1135) 

s_sub_i32 s53, s52, s54 // (op 1135) 


s_mul_i32 s52, s0, s44 // (op 1324) 
s_sub_u32 s54, s16, s52 // (op 1324) 

s_mul_i32 s52, s19, s44 // (op 1324) 
s_add_u32 s52, s52, s54 // (op 1324) 

s_mul_hi_i32 s54, s16, s10 // (op 1324) 
s_add_i32 s55, s54, s16 // (op 1324) 

s_ashr_i32 s54, s55, 31 // (op 1324) 
s_cmp_eq_i32 s10, 0 // (op 1324) 
s_mov_b32 s56, scc // (op 1324) 
s_cmp_eq_i32 s56, 0 // (op 1324) 
s_cselect_b32 s57, 0, -1 // (op 1324) 

s_and_b32 s56, s47, 31 // (op 1324) 
s_lshl_b32 s58, 1, s56 // (op 1324) 
s_add_i32 s59, s58, s57 // (op 1324) 


s_and_b32 s57, s54, s59 // (op 1324) 


s_add_i32 s54, s55, s57 // (op 1324) 


s_ashr_i32 s55, s54, s56 // (op 1324) 


s_lshl_b32 s54, s47, 24 // (op 1324) 
s_ashr_i32 s56, s54, 31 // (op 1324) 

s_xor_b32 s54, s55, s56 // (op 1324) 

s_sub_i32 s55, s54, s56 // (op 1324) 


s_add_u32 s56, s6, 64 // (op 1135) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1135) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1135) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1135) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1135) 

s_mul_i32 s54, s44, s56 // (op 1135) 
s_mul_i32 s58, s48, s54 // (op 1135) 

s_mul_i32 s54, s53, s58 // (op 1135) 

s_sub_u32 s58, s16, s54 // (op 1135) 

s_mov_b32 s54, s58 // (op 1135) 
s_mul_hi_i32 s59, s54, s15 // (op 1135) 
s_add_i32 s60, s59, s54 // (op 1135) 


s_ashr_i32 s54, s60, 31 // (op 1135) 
s_cmp_eq_i32 s15, 0 // (op 1135) 
s_mov_b32 s59, scc // (op 1135) 
s_cmp_eq_i32 s59, 0 // (op 1135) 
s_cselect_b32 s61, 0, -1 // (op 1135) 

s_and_b32 s59, s46, 31 // (op 1135) 
s_lshl_b32 s62, 1, s59 // (op 1135) 
s_add_i32 s63, s62, s61 // (op 1135) 


s_and_b32 s61, s54, s63 // (op 1135) 


s_add_i32 s54, s60, s61 // (op 1135) 


s_ashr_i32 s60, s54, s59 // (op 1135) 


s_lshl_b32 s54, s46, 24 // (op 1135) 
s_ashr_i32 s59, s54, 31 // (op 1135) 

s_xor_b32 s54, s60, s59 // (op 1135) 

s_sub_i32 s60, s54, s59 // (op 1135) 


s_mul_i32 s54, s45, s44 // (op 1135) 
s_add_u32 s62, s8, 16 // (op 1135) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 1135) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1135) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1135) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 1135) 

s_sub_i32 s59, s62, s54 // (op 1135) 

s_mul_i32 s61, s60, s59 // (op 1135) 

s_sub_u32 s59, s58, s61 // (op 1135) 


s_add_u32 s58, s59, s54 // (op 1135) 


s_cmp_eq_i32 s53, 0 // (op 1135) 
s_mov_b32 s54, scc // (op 1135) 

s_cmp_eq_i32 s54, 0 // (op 1135) 
s_cselect_b32 s53, s58, s1 // (op 1135) 


s_mul_i32 s1, s51, s56 // (op 1135) 


v_lshl_add_u32 v9, s53, 4, v6 // (op 1135) 


s_sub_i32 s51, s18, s1 // (op 1135) 


s_cmp_eq_i32 s54, 0 // (op 1135) 
s_cselect_b32 s1, s60, s51 // (op 1135) 



v_lshl_add_u32 v6, s1, 6, v5 // (op 1135) 


v_mul_lo_u32 v5, v6, s2 // (op 1135) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 1135) 


s_add_u32 s56, s6, 64 // (op 1324) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1324) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1324) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1324) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1324) 

s_mul_i32 s1, s44, s56 // (op 1324) 
s_mul_i32 s18, s48, s1 // (op 1324) 

s_mul_i32 s1, s55, s18 // (op 1324) 

s_sub_u32 s18, s16, s1 // (op 1324) 

s_mov_b32 s1, s18 // (op 1324) 
s_mul_hi_i32 s51, s1, s15 // (op 1324) 
s_add_i32 s53, s51, s1 // (op 1324) 


s_ashr_i32 s1, s53, 31 // (op 1324) 
s_cmp_eq_i32 s15, 0 // (op 1324) 
s_mov_b32 s51, scc // (op 1324) 
s_cmp_eq_i32 s51, 0 // (op 1324) 
s_cselect_b32 s54, 0, -1 // (op 1324) 

s_and_b32 s51, s46, 31 // (op 1324) 
s_lshl_b32 s58, 1, s51 // (op 1324) 
s_add_i32 s59, s58, s54 // (op 1324) 


s_and_b32 s54, s1, s59 // (op 1324) 


s_add_i32 s1, s53, s54 // (op 1324) 


s_ashr_i32 s53, s1, s51 // (op 1324) 


s_lshl_b32 s1, s46, 24 // (op 1324) 
s_ashr_i32 s51, s1, 31 // (op 1324) 

s_xor_b32 s1, s53, s51 // (op 1324) 

s_sub_i32 s53, s1, s51 // (op 1324) 


s_mul_i32 s1, s45, s44 // (op 1324) 
s_add_u32 s58, s8, 16 // (op 1324) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 1324) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1324) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1324) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1324) 

s_sub_i32 s51, s58, s1 // (op 1324) 

s_mul_i32 s54, s53, s51 // (op 1324) 

s_sub_u32 s51, s18, s54 // (op 1324) 


s_add_u32 s18, s51, s1 // (op 1324) 


s_cmp_eq_i32 s55, 0 // (op 1324) 
s_mov_b32 s1, scc // (op 1324) 

s_cmp_eq_i32 s1, 0 // (op 1324) 
s_cselect_b32 s51, s18, s52 // (op 1324) 


s_mul_i32 s18, s19, s56 // (op 1324) 


v_lshl_add_u32 v5, s51, 4, v12 // (op 1324) 


s_sub_i32 s19, s0, s18 // (op 1324) 


s_cmp_eq_i32 s1, 0 // (op 1324) 
s_cselect_b32 s0, s53, s19 // (op 1324) 



v_lshl_add_u32 v9, s0, 6, v8 // (op 1324) 


v_mul_lo_u32 v8, v9, s4 // (op 1324) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 1324) 


s_mul_i32 s1, s2, s7 // (op 1137) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1137) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1137) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1137) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1137) most significant: sum
s_add_u32 s1, s1, s19 // (op 1137) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1137) 

s_mul_i32 s1, s4, s7 // (op 1326) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 1326) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 1326) least significant
s_mul_hi_u32 s52, s4, s6 // (op 1326) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1326) most significant: sum
s_add_u32 s1, s1, s52 // (op 1326) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 1326) 

s_mov_b32 s56, s20 // (op 1137) reg expression
s_mov_b32 s57, s21 // (op 1137) reg expression
s_mov_b32 s58, s18 // (op 1137) reg expression
s_mov_b32 s59, 131072 // (op 1137) call()

s_mov_b32 s60, s22 // (op 1326) reg expression

s_mov_b32 s61, s23 // (op 1326) reg expression
s_mov_b32 s62, s52 // (op 1326) reg expression
s_mov_b32 s63, 131072 // (op 1326) call()


v_mov_b32 v5, s49 // (op 2) Move value
v_mov_b32 v8, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 310) call()
v_accvgpr_write a5, 0.00000 // (op 310) call()
v_accvgpr_write a6, 0.00000 // (op 310) call()
v_accvgpr_write a7, 0.00000 // (op 310) call()
s_mul_hi_i32 s0, s16, s11 // (op 902) 
v_lshrrev_b32 v12, 3, v1 // (op 902) 
s_add_i32 s1, s0, s16 // (op 902) 

v_and_b32 v13, -8, v1 // (op 902) 
s_ashr_i32 s0, s1, 31 // (op 902) 
v_sub_u32 v14, v1, v13 // (op 902) 

s_cmp_eq_i32 s11, 0 // (op 902) 
s_mov_b32 s18, scc // (op 902) 
v_lshlrev_b32 v13, 4, v14 // (op 902) 

s_cmp_eq_i32 s18, 0 // (op 902) 
s_cselect_b32 s19, 0, -1 // (op 902) 

v_mov_b32 v15, 0 // (op 902) convert
v_mov_b32 v14, v13 // (op 902) convert

s_and_b32 s18, s12, 31 // (op 902) 
s_lshl_b32 s20, 1, s18 // (op 902) 
s_add_i32 s21, s20, s19 // (op 902) 


s_and_b32 s19, s0, s21 // (op 902) 


s_add_i32 s0, s1, s19 // (op 902) 


s_ashr_i32 s1, s0, s18 // (op 902) 


s_lshl_b32 s0, s12, 24 // (op 902) 
s_ashr_i32 s18, s0, 31 // (op 902) 

s_xor_b32 s0, s1, s18 // (op 902) 

s_sub_i32 s1, s0, s18 // (op 902) 


s_mul_hi_i32 s0, s1, s13 // (op 902) 
s_add_i32 s18, s0, s1 // (op 902) 

s_ashr_i32 s0, s18, 31 // (op 902) 
s_cmp_eq_i32 s13, 0 // (op 902) 
s_mov_b32 s19, scc // (op 902) 
s_cmp_eq_i32 s19, 0 // (op 902) 
s_cselect_b32 s20, 0, -1 // (op 902) 

s_and_b32 s19, s14, 31 // (op 902) 
s_lshl_b32 s21, 1, s19 // (op 902) 
s_add_i32 s22, s21, s20 // (op 902) 


s_and_b32 s23, s0, s22 // (op 902) 


s_add_i32 s0, s18, s23 // (op 902) 


s_ashr_i32 s18, s0, s19 // (op 902) 


s_lshl_b32 s0, s14, 24 // (op 902) 
s_ashr_i32 s19, s0, 31 // (op 902) 

s_xor_b32 s0, s18, s19 // (op 902) 

s_sub_i32 s18, s0, s19 // (op 902) 


s_mul_i32 s0, s1, s44 // (op 902) 

s_sub_u32 s1, s16, s0 // (op 902) 

s_mul_i32 s0, s18, s44 // (op 902) 
s_add_u32 s0, s0, s1 // (op 902) 


s_mul_hi_i32 s1, s16, s10 // (op 902) 
s_add_i32 s18, s1, s16 // (op 902) 

s_ashr_i32 s1, s18, 31 // (op 902) 
s_cmp_eq_i32 s10, 0 // (op 902) 
s_mov_b32 s19, scc // (op 902) 
s_cmp_eq_i32 s19, 0 // (op 902) 
s_cselect_b32 s20, 0, -1 // (op 902) 

s_and_b32 s19, s47, 31 // (op 902) 
s_lshl_b32 s21, 1, s19 // (op 902) 
s_add_i32 s22, s21, s20 // (op 902) 


s_and_b32 s23, s1, s22 // (op 902) 


s_add_i32 s1, s18, s23 // (op 902) 


s_ashr_i32 s18, s1, s19 // (op 902) 


s_lshl_b32 s1, s47, 24 // (op 902) 
s_ashr_i32 s19, s1, 31 // (op 902) 

s_xor_b32 s1, s18, s19 // (op 902) 

s_sub_i32 s18, s1, s19 // (op 902) 


s_add_u32 s22, s6, 64 // (op 902) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 902) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 902) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 902) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 6 // (op 902) 

s_mul_i32 s1, s44, s22 // (op 902) 

s_mul_i32 s19, s48, s1 // (op 902) 

s_mul_i32 s1, s18, s19 // (op 902) 

s_sub_u32 s19, s16, s1 // (op 902) 

s_mov_b32 s1, s19 // (op 902) 
s_mul_hi_i32 s20, s1, s15 // (op 902) 
s_add_i32 s21, s20, s1 // (op 902) 


s_ashr_i32 s1, s21, 31 // (op 902) 
s_cmp_eq_i32 s15, 0 // (op 902) 
s_mov_b32 s20, scc // (op 902) 
s_cmp_eq_i32 s20, 0 // (op 902) 
s_cselect_b32 s22, 0, -1 // (op 902) 

s_and_b32 s20, s46, 31 // (op 902) 
s_lshl_b32 s23, 1, s20 // (op 902) 
s_add_i32 s49, s23, s22 // (op 902) 


s_and_b32 s22, s1, s49 // (op 902) 


s_add_i32 s1, s21, s22 // (op 902) 


s_ashr_i32 s21, s1, s20 // (op 902) 


s_lshl_b32 s1, s46, 24 // (op 902) 
s_ashr_i32 s20, s1, 31 // (op 902) 

s_xor_b32 s1, s21, s20 // (op 902) 

s_sub_i32 s21, s1, s20 // (op 902) 


s_mul_i32 s1, s45, s44 // (op 902) 
s_add_u32 s22, s8, 16 // (op 902) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 902) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 902) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 902) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 4 // (op 902) 

s_sub_i32 s20, s22, s1 // (op 902) 

s_mul_i32 s22, s21, s20 // (op 902) 


s_sub_u32 s23, s19, s22 // (op 902) 


s_add_u32 s19, s23, s1 // (op 902) 


s_cmp_eq_i32 s18, 0 // (op 902) 
s_mov_b32 s1, scc // (op 902) 

s_cmp_eq_i32 s1, 0 // (op 902) 
s_cselect_b32 s18, s19, s0 // (op 902) 



v_lshl_add_u32 v13, s18, 4, v12 // (op 902) 


v_mul_lo_u32 v17, v13, s25 // (op 902) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 902) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 902) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 902) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 902) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 902) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 902) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 902) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 902) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 902) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 902) 

v_mov_b32 v12, v14 // (op 902) convert

s_mul_i32 s1, s9, 6 // (op 904) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 904) least significant
s_mul_hi_u32 s18, s8, 6 // (op 904) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 904) most significant: sum

s_mul_i32 s19, s24, s1 // (op 904) most significant: low of low * high
s_mul_i32 s20, s25, s0 // (op 904) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 904) least significant
s_mul_hi_u32 s21, s24, s0 // (op 904) most significant: high of low * low
s_add_u32 s19, s19, s20 // (op 904) most significant: sum
s_add_u32 s19, s19, s21 // (op 904) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 904) 

s_mov_b32 s20, s26 // (op 904) reg expression
s_mov_b32 s21, s27 // (op 904) reg expression
s_mov_b32 s22, s0 // (op 904) reg expression
s_mov_b32 s23, 131072 // (op 904) call()

s_mul_hi_i32 s0, s16, s11 // (op 921) 
v_lshrrev_b32 v13, 3, v1 // (op 921) 
s_add_i32 s1, s0, s16 // (op 921) 

v_and_b32 v14, -8, v1 // (op 921) 
s_ashr_i32 s0, s1, 31 // (op 921) 
v_sub_u32 v15, v1, v14 // (op 921) 

s_cmp_eq_i32 s11, 0 // (op 921) 
s_mov_b32 s18, scc // (op 921) 
v_lshlrev_b32 v14, 4, v15 // (op 921) 

s_cmp_eq_i32 s18, 0 // (op 921) 
s_cselect_b32 s19, 0, -1 // (op 921) 

v_mov_b32 v17, 0 // (op 921) convert
v_mov_b32 v16, v14 // (op 921) convert

s_and_b32 s18, s12, 31 // (op 921) 
s_lshl_b32 s49, 1, s18 // (op 921) 
s_add_i32 s50, s49, s19 // (op 921) 


s_and_b32 s19, s0, s50 // (op 921) 


s_add_i32 s0, s1, s19 // (op 921) 


s_ashr_i32 s1, s0, s18 // (op 921) 


s_lshl_b32 s0, s12, 24 // (op 921) 
s_ashr_i32 s18, s0, 31 // (op 921) 

s_xor_b32 s0, s1, s18 // (op 921) 

s_sub_i32 s1, s0, s18 // (op 921) 


s_mul_hi_i32 s0, s1, s13 // (op 921) 
s_add_i32 s18, s0, s1 // (op 921) 

s_ashr_i32 s0, s18, 31 // (op 921) 
s_cmp_eq_i32 s13, 0 // (op 921) 
s_mov_b32 s19, scc // (op 921) 
s_cmp_eq_i32 s19, 0 // (op 921) 
s_cselect_b32 s49, 0, -1 // (op 921) 

s_and_b32 s19, s14, 31 // (op 921) 
s_lshl_b32 s50, 1, s19 // (op 921) 
s_add_i32 s51, s50, s49 // (op 921) 


s_and_b32 s49, s0, s51 // (op 921) 


s_add_i32 s0, s18, s49 // (op 921) 


s_ashr_i32 s18, s0, s19 // (op 921) 


s_lshl_b32 s0, s14, 24 // (op 921) 
s_ashr_i32 s19, s0, 31 // (op 921) 

s_xor_b32 s0, s18, s19 // (op 921) 

s_sub_i32 s18, s0, s19 // (op 921) 


s_mul_i32 s0, s1, s44 // (op 921) 

s_sub_u32 s1, s16, s0 // (op 921) 

s_mul_i32 s0, s18, s44 // (op 921) 
s_add_u32 s0, s0, s1 // (op 921) 


s_mul_hi_i32 s1, s16, s10 // (op 921) 
s_add_i32 s18, s1, s16 // (op 921) 

s_ashr_i32 s1, s18, 31 // (op 921) 
s_cmp_eq_i32 s10, 0 // (op 921) 
s_mov_b32 s19, scc // (op 921) 
s_cmp_eq_i32 s19, 0 // (op 921) 
s_cselect_b32 s49, 0, -1 // (op 921) 

s_and_b32 s19, s47, 31 // (op 921) 
s_lshl_b32 s50, 1, s19 // (op 921) 
s_add_i32 s51, s50, s49 // (op 921) 


s_and_b32 s49, s1, s51 // (op 921) 


s_add_i32 s1, s18, s49 // (op 921) 


s_ashr_i32 s18, s1, s19 // (op 921) 


s_lshl_b32 s1, s47, 24 // (op 921) 
s_ashr_i32 s19, s1, 31 // (op 921) 

s_xor_b32 s1, s18, s19 // (op 921) 

s_sub_i32 s18, s1, s19 // (op 921) 


s_add_u32 s54, s6, 64 // (op 921) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 921) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 921) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 921) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 921) 

s_mul_i32 s1, s44, s50 // (op 921) 

s_mul_i32 s19, s48, s1 // (op 921) 

s_mul_i32 s1, s18, s19 // (op 921) 

s_sub_u32 s19, s16, s1 // (op 921) 

s_mov_b32 s1, s19 // (op 921) 
s_mul_hi_i32 s49, s1, s15 // (op 921) 
s_add_i32 s50, s49, s1 // (op 921) 


s_ashr_i32 s1, s50, 31 // (op 921) 
s_cmp_eq_i32 s15, 0 // (op 921) 
s_mov_b32 s49, scc // (op 921) 
s_cmp_eq_i32 s49, 0 // (op 921) 
s_cselect_b32 s51, 0, -1 // (op 921) 

s_and_b32 s49, s46, 31 // (op 921) 
s_lshl_b32 s52, 1, s49 // (op 921) 
s_add_i32 s53, s52, s51 // (op 921) 


s_and_b32 s51, s1, s53 // (op 921) 


s_add_i32 s1, s50, s51 // (op 921) 


s_ashr_i32 s50, s1, s49 // (op 921) 


s_lshl_b32 s1, s46, 24 // (op 921) 
s_ashr_i32 s49, s1, 31 // (op 921) 

s_xor_b32 s1, s50, s49 // (op 921) 

s_sub_i32 s50, s1, s49 // (op 921) 


s_mul_i32 s1, s45, s44 // (op 921) 
s_add_u32 s54, s8, 16 // (op 921) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 921) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 921) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 921) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 4 // (op 921) 

s_sub_i32 s49, s54, s1 // (op 921) 

s_mul_i32 s51, s50, s49 // (op 921) 


s_sub_u32 s49, s19, s51 // (op 921) 


s_add_u32 s19, s49, s1 // (op 921) 


s_cmp_eq_i32 s18, 0 // (op 921) 
s_mov_b32 s1, scc // (op 921) 

s_cmp_eq_i32 s1, 0 // (op 921) 
s_cselect_b32 s18, s19, s0 // (op 921) 



v_lshl_add_u32 v14, s18, 4, v13 // (op 921) 


v_mul_lo_u32 v19, v14, s25 // (op 921) most significant: low of low * high
v_mul_hi_u32 v13, v14, s24 // (op 921) most significant: high of low * low
v_mul_lo_u32 v18, v14, s24 // (op 921) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 921) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 921) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 921) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 921) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 921) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 921) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 921) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 921) 

v_mov_b32 v13, v14 // (op 921) convert

s_mul_i32 s1, s9, 6 // (op 923) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 923) least significant
s_mul_hi_u32 s18, s8, 6 // (op 923) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 923) most significant: sum

s_mul_i32 s19, s24, s1 // (op 923) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 923) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 923) least significant
s_mul_hi_u32 s50, s24, s0 // (op 923) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 923) most significant: sum
s_add_u32 s19, s19, s50 // (op 923) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 923) 

s_mov_b32 s20, s26 // (op 923) reg expression
s_mov_b32 s21, s27 // (op 923) reg expression
s_mov_b32 s22, s0 // (op 923) reg expression
s_mov_b32 s23, 131072 // (op 923) call()

s_mul_hi_i32 s0, s16, s11 // (op 966) 
s_add_i32 s1, s0, s16 // (op 966) 

s_ashr_i32 s0, s1, 31 // (op 966) 
s_cmp_eq_i32 s11, 0 // (op 966) 
s_mov_b32 s18, scc // (op 966) 
s_cmp_eq_i32 s18, 0 // (op 966) 
s_cselect_b32 s19, 0, -1 // (op 966) 

s_and_b32 s18, s12, 31 // (op 966) 
s_lshl_b32 s49, 1, s18 // (op 966) 
s_add_i32 s50, s49, s19 // (op 966) 


s_and_b32 s19, s0, s50 // (op 966) 


s_add_i32 s0, s1, s19 // (op 966) 


s_ashr_i32 s1, s0, s18 // (op 966) 


s_lshl_b32 s0, s12, 24 // (op 966) 
s_ashr_i32 s18, s0, 31 // (op 966) 

s_xor_b32 s0, s1, s18 // (op 966) 

s_sub_i32 s1, s0, s18 // (op 966) 


s_mul_hi_i32 s0, s1, s13 // (op 966) 
s_add_i32 s18, s0, s1 // (op 966) 

s_ashr_i32 s0, s18, 31 // (op 966) 
s_cmp_eq_i32 s13, 0 // (op 966) 
s_mov_b32 s19, scc // (op 966) 
s_cmp_eq_i32 s19, 0 // (op 966) 
s_cselect_b32 s49, 0, -1 // (op 966) 

s_and_b32 s19, s14, 31 // (op 966) 
s_lshl_b32 s50, 1, s19 // (op 966) 
s_add_i32 s51, s50, s49 // (op 966) 


s_and_b32 s49, s0, s51 // (op 966) 


s_add_i32 s0, s18, s49 // (op 966) 


s_ashr_i32 s18, s0, s19 // (op 966) 


s_lshl_b32 s0, s14, 24 // (op 966) 
s_ashr_i32 s19, s0, 31 // (op 966) 

s_xor_b32 s0, s18, s19 // (op 966) 

s_sub_i32 s18, s0, s19 // (op 966) 


s_add_u32 s54, s6, 64 // (op 966) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 966) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 966) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 966) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 966) 

s_mul_i32 s0, s18, s50 // (op 966) 

s_sub_i32 s49, s1, s0 // (op 966) 


s_mul_hi_i32 s0, s16, s10 // (op 966) 
s_add_i32 s1, s0, s16 // (op 966) 

s_ashr_i32 s0, s1, 31 // (op 966) 
s_cmp_eq_i32 s10, 0 // (op 966) 
s_mov_b32 s18, scc // (op 966) 
s_cmp_eq_i32 s18, 0 // (op 966) 
s_cselect_b32 s19, 0, -1 // (op 966) 

s_and_b32 s18, s47, 31 // (op 966) 
s_lshl_b32 s52, 1, s18 // (op 966) 
s_add_i32 s53, s52, s19 // (op 966) 


s_and_b32 s19, s0, s53 // (op 966) 


s_add_i32 s0, s1, s19 // (op 966) 


s_ashr_i32 s1, s0, s18 // (op 966) 


s_lshl_b32 s0, s47, 24 // (op 966) 
s_ashr_i32 s18, s0, 31 // (op 966) 

s_xor_b32 s0, s1, s18 // (op 966) 

s_sub_i32 s1, s0, s18 // (op 966) 


s_mul_i32 s0, s44, s50 // (op 966) 

s_mul_i32 s18, s48, s0 // (op 966) 

s_mul_i32 s0, s1, s18 // (op 966) 

s_sub_u32 s18, s16, s0 // (op 966) 

s_mul_hi_i32 s0, s18, s15 // (op 966) 
s_add_i32 s19, s0, s18 // (op 966) 


s_ashr_i32 s0, s19, 31 // (op 966) 
s_cmp_eq_i32 s15, 0 // (op 966) 
s_mov_b32 s18, scc // (op 966) 
s_cmp_eq_i32 s18, 0 // (op 966) 
s_cselect_b32 s50, 0, -1 // (op 966) 

s_and_b32 s18, s46, 31 // (op 966) 
s_lshl_b32 s51, 1, s18 // (op 966) 
s_add_i32 s52, s51, s50 // (op 966) 


s_and_b32 s50, s0, s52 // (op 966) 


s_add_i32 s0, s19, s50 // (op 966) 


s_ashr_i32 s19, s0, s18 // (op 966) 


s_lshl_b32 s0, s46, 24 // (op 966) 
s_ashr_i32 s18, s0, 31 // (op 966) 

s_xor_b32 s0, s19, s18 // (op 966) 

s_sub_i32 s19, s0, s18 // (op 966) 


s_cmp_eq_i32 s1, 0 // (op 966) 
s_mov_b32 s0, scc // (op 966) 

s_cmp_eq_i32 s0, 0 // (op 966) 
s_cselect_b32 s1, s19, s49 // (op 966) 



s_lshl_b32 s0, s1, 6 // (op 966) 

s_mov_b32 s19, 0 // (op 966) convert
s_mov_b32 s18, s0 // (op 966) convert

v_mul_lo_u32 v15, v1, s29 // (op 966) most significant: low of low * high
v_mul_hi_u32 v16, v1, s28 // (op 966) most significant: high of low * low
v_mul_lo_u32 v14, v1, s28 // (op 966) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 966) most significant: sum

v_mov_b32 v16, s18 // (op 966) 
v_mov_b32 v17, s19 // (op 966) 
v_add_co_u32 v14, s[0:1], v14, v16 // (op 966) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 966) most significant half




v_mul_lo_u32 v17, v15, 6 // (op 966) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 966) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 966) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 966) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 966) 

v_mov_b32 v16, v14 // (op 966) convert

s_mul_i32 s1, s33, 6 // (op 968) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 968) least significant
s_mul_hi_u32 s18, s32, 6 // (op 968) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 968) most significant: sum

s_mul_i32 s19, s28, s1 // (op 968) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 968) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 968) least significant
s_mul_hi_u32 s50, s28, s0 // (op 968) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 968) most significant: sum
s_add_u32 s19, s19, s50 // (op 968) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 968) 

s_mov_b32 s52, s30 // (op 968) reg expression
s_mov_b32 s53, s31 // (op 968) reg expression
s_mov_b32 s54, s0 // (op 968) reg expression
s_mov_b32 s55, 131072 // (op 968) call()

s_mul_hi_i32 s0, s16, s11 // (op 985) 
s_add_i32 s1, s0, s16 // (op 985) 

s_ashr_i32 s0, s1, 31 // (op 985) 
s_cmp_eq_i32 s11, 0 // (op 985) 
s_mov_b32 s18, scc // (op 985) 
s_cmp_eq_i32 s18, 0 // (op 985) 
s_cselect_b32 s19, 0, -1 // (op 985) 

s_and_b32 s18, s12, 31 // (op 985) 
s_lshl_b32 s49, 1, s18 // (op 985) 
s_add_i32 s50, s49, s19 // (op 985) 


s_and_b32 s19, s0, s50 // (op 985) 


s_add_i32 s0, s1, s19 // (op 985) 


s_ashr_i32 s1, s0, s18 // (op 985) 


s_lshl_b32 s0, s12, 24 // (op 985) 
s_ashr_i32 s18, s0, 31 // (op 985) 

s_xor_b32 s0, s1, s18 // (op 985) 

s_sub_i32 s1, s0, s18 // (op 985) 


s_mul_hi_i32 s0, s1, s13 // (op 985) 
s_add_i32 s18, s0, s1 // (op 985) 

s_ashr_i32 s0, s18, 31 // (op 985) 
s_cmp_eq_i32 s13, 0 // (op 985) 
s_mov_b32 s19, scc // (op 985) 
s_cmp_eq_i32 s19, 0 // (op 985) 
s_cselect_b32 s49, 0, -1 // (op 985) 

s_and_b32 s19, s14, 31 // (op 985) 
s_lshl_b32 s50, 1, s19 // (op 985) 
s_add_i32 s51, s50, s49 // (op 985) 


s_and_b32 s49, s0, s51 // (op 985) 


s_add_i32 s0, s18, s49 // (op 985) 


s_ashr_i32 s18, s0, s19 // (op 985) 


s_lshl_b32 s0, s14, 24 // (op 985) 
s_ashr_i32 s19, s0, 31 // (op 985) 

s_xor_b32 s0, s18, s19 // (op 985) 

s_sub_i32 s18, s0, s19 // (op 985) 


s_add_u32 s50, s6, 64 // (op 985) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 985) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 985) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 985) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 985) 

s_mul_i32 s0, s18, s50 // (op 985) 

s_sub_i32 s49, s1, s0 // (op 985) 


s_mul_hi_i32 s0, s16, s10 // (op 985) 
s_add_i32 s1, s0, s16 // (op 985) 

s_ashr_i32 s0, s1, 31 // (op 985) 
s_cmp_eq_i32 s10, 0 // (op 985) 
s_mov_b32 s18, scc // (op 985) 
s_cmp_eq_i32 s18, 0 // (op 985) 
s_cselect_b32 s19, 0, -1 // (op 985) 

s_and_b32 s18, s47, 31 // (op 985) 
s_lshl_b32 s64, 1, s18 // (op 985) 
s_add_i32 s65, s64, s19 // (op 985) 


s_and_b32 s19, s0, s65 // (op 985) 


s_add_i32 s0, s1, s19 // (op 985) 


s_ashr_i32 s1, s0, s18 // (op 985) 


s_lshl_b32 s0, s47, 24 // (op 985) 
s_ashr_i32 s18, s0, 31 // (op 985) 

s_xor_b32 s0, s1, s18 // (op 985) 

s_sub_i32 s1, s0, s18 // (op 985) 


s_mul_i32 s0, s44, s50 // (op 985) 

s_mul_i32 s18, s48, s0 // (op 985) 

s_mul_i32 s0, s1, s18 // (op 985) 

s_sub_u32 s18, s16, s0 // (op 985) 

s_mul_hi_i32 s0, s18, s15 // (op 985) 
s_add_i32 s19, s0, s18 // (op 985) 


s_ashr_i32 s0, s19, 31 // (op 985) 
s_cmp_eq_i32 s15, 0 // (op 985) 
s_mov_b32 s18, scc // (op 985) 
s_cmp_eq_i32 s18, 0 // (op 985) 
s_cselect_b32 s50, 0, -1 // (op 985) 

s_and_b32 s18, s46, 31 // (op 985) 
s_lshl_b32 s51, 1, s18 // (op 985) 
s_add_i32 s64, s51, s50 // (op 985) 


s_and_b32 s50, s0, s64 // (op 985) 


s_add_i32 s0, s19, s50 // (op 985) 


s_ashr_i32 s19, s0, s18 // (op 985) 


s_lshl_b32 s0, s46, 24 // (op 985) 
s_ashr_i32 s18, s0, 31 // (op 985) 

s_xor_b32 s0, s19, s18 // (op 985) 

s_sub_i32 s19, s0, s18 // (op 985) 


s_cmp_eq_i32 s1, 0 // (op 985) 
s_mov_b32 s0, scc // (op 985) 

s_cmp_eq_i32 s0, 0 // (op 985) 
s_cselect_b32 s1, s19, s49 // (op 985) 



s_lshl_b32 s0, s1, 6 // (op 985) 

s_mov_b32 s19, 0 // (op 985) convert
s_mov_b32 s18, s0 // (op 985) convert

v_mul_lo_u32 v15, v1, s29 // (op 985) most significant: low of low * high
v_mul_hi_u32 v17, v1, s28 // (op 985) most significant: high of low * low
v_mul_lo_u32 v14, v1, s28 // (op 985) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 985) most significant: sum

v_mov_b32 v17, s18 // (op 985) 
v_mov_b32 v18, s19 // (op 985) 
v_add_co_u32 v14, s[0:1], v14, v17 // (op 985) least significant half
v_addc_co_u32 v15, s[0:1], v15, v18, s[0:1] // (op 985) most significant half




v_mul_lo_u32 v19, v15, 6 // (op 985) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 985) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 985) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 985) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 985) 

v_mov_b32 v17, v14 // (op 985) convert

s_mul_i32 s1, s33, 6 // (op 987) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 987) least significant
s_mul_hi_u32 s18, s32, 6 // (op 987) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 987) most significant: sum

s_mul_i32 s19, s28, s1 // (op 987) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 987) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 987) least significant
s_mul_hi_u32 s50, s28, s0 // (op 987) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 987) most significant: sum
s_add_u32 s19, s19, s50 // (op 987) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 987) 

s_mov_b32 s52, s30 // (op 987) reg expression
s_mov_b32 s53, s31 // (op 987) reg expression
s_mov_b32 s54, s0 // (op 987) reg expression
s_mov_b32 s55, 131072 // (op 987) call()

s_mul_hi_i32 s0, s16, s11 // (op 1030) 
v_lshrrev_b32 v14, 7, v1 // (op 1030) 
s_add_i32 s1, s0, s16 // (op 1030) 

v_and_b32 v15, 15, v1 // (op 1030) 
s_ashr_i32 s0, s1, 31 // (op 1030) 
v_lshl_add_u32 v18, v14, 4, v15 // (op 1030) 


s_cmp_eq_i32 s11, 0 // (op 1030) 
s_mov_b32 s18, scc // (op 1030) 
v_and_b32 v14, 63, v1 // (op 1030) 
s_cmp_eq_i32 s18, 0 // (op 1030) 
s_cselect_b32 s19, 0, -1 // (op 1030) 

v_lshrrev_b32 v15, 4, v14 // (op 1030) 

s_and_b32 s18, s12, 31 // (op 1030) 
s_lshl_b32 s49, 1, s18 // (op 1030) 
s_add_i32 s50, s49, s19 // (op 1030) 


s_and_b32 s19, s0, s50 // (op 1030) 


s_add_i32 s0, s1, s19 // (op 1030) 


s_ashr_i32 s1, s0, s18 // (op 1030) 


s_lshl_b32 s0, s12, 24 // (op 1030) 
s_ashr_i32 s18, s0, 31 // (op 1030) 

s_xor_b32 s0, s1, s18 // (op 1030) 

s_sub_i32 s1, s0, s18 // (op 1030) 


s_mul_hi_i32 s0, s1, s13 // (op 1030) 
s_add_i32 s18, s0, s1 // (op 1030) 

s_ashr_i32 s0, s18, 31 // (op 1030) 
s_cmp_eq_i32 s13, 0 // (op 1030) 
s_mov_b32 s19, scc // (op 1030) 
s_cmp_eq_i32 s19, 0 // (op 1030) 
s_cselect_b32 s49, 0, -1 // (op 1030) 

s_and_b32 s19, s14, 31 // (op 1030) 
s_lshl_b32 s50, 1, s19 // (op 1030) 
s_add_i32 s51, s50, s49 // (op 1030) 


s_and_b32 s49, s0, s51 // (op 1030) 


s_add_i32 s0, s18, s49 // (op 1030) 


s_ashr_i32 s18, s0, s19 // (op 1030) 


s_lshl_b32 s0, s14, 24 // (op 1030) 
s_ashr_i32 s19, s0, 31 // (op 1030) 

s_xor_b32 s0, s18, s19 // (op 1030) 

s_sub_i32 s18, s0, s19 // (op 1030) 


s_mul_i32 s0, s1, s44 // (op 1030) 

s_sub_u32 s1, s16, s0 // (op 1030) 

s_mul_i32 s0, s18, s44 // (op 1030) 
s_add_u32 s0, s0, s1 // (op 1030) 


s_mul_hi_i32 s1, s16, s10 // (op 1030) 
s_add_i32 s18, s1, s16 // (op 1030) 

s_ashr_i32 s1, s18, 31 // (op 1030) 
s_cmp_eq_i32 s10, 0 // (op 1030) 
s_mov_b32 s19, scc // (op 1030) 
s_cmp_eq_i32 s19, 0 // (op 1030) 
s_cselect_b32 s49, 0, -1 // (op 1030) 

s_and_b32 s19, s47, 31 // (op 1030) 
s_lshl_b32 s50, 1, s19 // (op 1030) 
s_add_i32 s51, s50, s49 // (op 1030) 


s_and_b32 s49, s1, s51 // (op 1030) 


s_add_i32 s1, s18, s49 // (op 1030) 


s_ashr_i32 s18, s1, s19 // (op 1030) 


s_lshl_b32 s1, s47, 24 // (op 1030) 
s_ashr_i32 s19, s1, 31 // (op 1030) 

s_xor_b32 s1, s18, s19 // (op 1030) 

s_sub_i32 s18, s1, s19 // (op 1030) 


s_add_u32 s50, s6, 64 // (op 1030) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1030) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1030) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1030) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1030) 

s_mul_i32 s1, s44, s50 // (op 1030) 

s_mul_i32 s19, s48, s1 // (op 1030) 

s_mul_i32 s1, s18, s19 // (op 1030) 

s_sub_u32 s19, s16, s1 // (op 1030) 

s_mov_b32 s1, s19 // (op 1030) 
s_mul_hi_i32 s49, s1, s15 // (op 1030) 
s_add_i32 s50, s49, s1 // (op 1030) 


s_ashr_i32 s1, s50, 31 // (op 1030) 
s_cmp_eq_i32 s15, 0 // (op 1030) 
s_mov_b32 s49, scc // (op 1030) 
s_cmp_eq_i32 s49, 0 // (op 1030) 
s_cselect_b32 s51, 0, -1 // (op 1030) 

s_and_b32 s49, s46, 31 // (op 1030) 
s_lshl_b32 s64, 1, s49 // (op 1030) 
s_add_i32 s65, s64, s51 // (op 1030) 


s_and_b32 s51, s1, s65 // (op 1030) 


s_add_i32 s1, s50, s51 // (op 1030) 


s_ashr_i32 s50, s1, s49 // (op 1030) 


s_lshl_b32 s1, s46, 24 // (op 1030) 
s_ashr_i32 s49, s1, 31 // (op 1030) 

s_xor_b32 s1, s50, s49 // (op 1030) 

s_sub_i32 s50, s1, s49 // (op 1030) 


s_mul_i32 s1, s45, s44 // (op 1030) 
s_add_u32 s64, s8, 16 // (op 1030) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 1030) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1030) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1030) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 1030) 

s_sub_i32 s49, s64, s1 // (op 1030) 

s_mul_i32 s51, s50, s49 // (op 1030) 


s_sub_u32 s49, s19, s51 // (op 1030) 


s_add_u32 s19, s49, s1 // (op 1030) 


s_cmp_eq_i32 s18, 0 // (op 1030) 
s_mov_b32 s1, scc // (op 1030) 

s_cmp_eq_i32 s1, 0 // (op 1030) 
s_cselect_b32 s18, s19, s0 // (op 1030) 



v_lshl_add_u32 v14, s18, 4, v18 // (op 1030) 


v_mul_lo_u32 v18, v14, s34 // (op 1030) 
v_add_u32 v18, v18, v15 // (op 1030) 


s_mul_i32 s1, s34, s9 // (op 1032) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1032) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1032) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1032) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1032) most significant: sum
s_add_u32 s1, s1, s19 // (op 1032) most significant: sum


s_mov_b32 s64, s36 // (op 1032) reg expression
s_mov_b32 s65, s37 // (op 1032) reg expression
s_mov_b32 s66, s0 // (op 1032) reg expression
s_mov_b32 s67, 131072 // (op 1032) call()

s_mul_hi_i32 s0, s16, s11 // (op 1049) 
v_lshrrev_b32 v14, 7, v1 // (op 1049) 
s_add_i32 s1, s0, s16 // (op 1049) 

v_and_b32 v15, 15, v1 // (op 1049) 
s_ashr_i32 s0, s1, 31 // (op 1049) 
v_lshl_add_u32 v19, v14, 4, v15 // (op 1049) 


s_cmp_eq_i32 s11, 0 // (op 1049) 
s_mov_b32 s18, scc // (op 1049) 
v_and_b32 v14, 63, v1 // (op 1049) 
s_cmp_eq_i32 s18, 0 // (op 1049) 
s_cselect_b32 s19, 0, -1 // (op 1049) 

v_lshrrev_b32 v15, 4, v14 // (op 1049) 

s_and_b32 s18, s12, 31 // (op 1049) 
s_lshl_b32 s49, 1, s18 // (op 1049) 
s_add_i32 s50, s49, s19 // (op 1049) 


s_and_b32 s19, s0, s50 // (op 1049) 


s_add_i32 s0, s1, s19 // (op 1049) 


s_ashr_i32 s1, s0, s18 // (op 1049) 


s_lshl_b32 s0, s12, 24 // (op 1049) 
s_ashr_i32 s18, s0, 31 // (op 1049) 

s_xor_b32 s0, s1, s18 // (op 1049) 

s_sub_i32 s1, s0, s18 // (op 1049) 


s_mul_hi_i32 s0, s1, s13 // (op 1049) 
s_add_i32 s18, s0, s1 // (op 1049) 

s_ashr_i32 s0, s18, 31 // (op 1049) 
s_cmp_eq_i32 s13, 0 // (op 1049) 
s_mov_b32 s19, scc // (op 1049) 
s_cmp_eq_i32 s19, 0 // (op 1049) 
s_cselect_b32 s49, 0, -1 // (op 1049) 

s_and_b32 s19, s14, 31 // (op 1049) 
s_lshl_b32 s50, 1, s19 // (op 1049) 
s_add_i32 s51, s50, s49 // (op 1049) 


s_and_b32 s49, s0, s51 // (op 1049) 


s_add_i32 s0, s18, s49 // (op 1049) 


s_ashr_i32 s18, s0, s19 // (op 1049) 


s_lshl_b32 s0, s14, 24 // (op 1049) 
s_ashr_i32 s19, s0, 31 // (op 1049) 

s_xor_b32 s0, s18, s19 // (op 1049) 

s_sub_i32 s18, s0, s19 // (op 1049) 


s_mul_i32 s0, s1, s44 // (op 1049) 

s_sub_u32 s1, s16, s0 // (op 1049) 

s_mul_i32 s0, s18, s44 // (op 1049) 
s_add_u32 s0, s0, s1 // (op 1049) 


s_mul_hi_i32 s1, s16, s10 // (op 1049) 
s_add_i32 s18, s1, s16 // (op 1049) 

s_ashr_i32 s1, s18, 31 // (op 1049) 
s_cmp_eq_i32 s10, 0 // (op 1049) 
s_mov_b32 s19, scc // (op 1049) 
s_cmp_eq_i32 s19, 0 // (op 1049) 
s_cselect_b32 s49, 0, -1 // (op 1049) 

s_and_b32 s19, s47, 31 // (op 1049) 
s_lshl_b32 s50, 1, s19 // (op 1049) 
s_add_i32 s51, s50, s49 // (op 1049) 


s_and_b32 s49, s1, s51 // (op 1049) 


s_add_i32 s1, s18, s49 // (op 1049) 


s_ashr_i32 s18, s1, s19 // (op 1049) 


s_lshl_b32 s1, s47, 24 // (op 1049) 
s_ashr_i32 s19, s1, 31 // (op 1049) 

s_xor_b32 s1, s18, s19 // (op 1049) 

s_sub_i32 s18, s1, s19 // (op 1049) 


s_add_u32 s50, s6, 64 // (op 1049) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1049) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1049) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1049) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1049) 

s_mul_i32 s1, s44, s50 // (op 1049) 

s_mul_i32 s19, s48, s1 // (op 1049) 

s_mul_i32 s1, s18, s19 // (op 1049) 

s_sub_u32 s19, s16, s1 // (op 1049) 

s_mov_b32 s1, s19 // (op 1049) 
s_mul_hi_i32 s49, s1, s15 // (op 1049) 
s_add_i32 s50, s49, s1 // (op 1049) 


s_ashr_i32 s1, s50, 31 // (op 1049) 
s_cmp_eq_i32 s15, 0 // (op 1049) 
s_mov_b32 s49, scc // (op 1049) 
s_cmp_eq_i32 s49, 0 // (op 1049) 
s_cselect_b32 s51, 0, -1 // (op 1049) 

s_and_b32 s49, s46, 31 // (op 1049) 
s_lshl_b32 s68, 1, s49 // (op 1049) 
s_add_i32 s69, s68, s51 // (op 1049) 


s_and_b32 s51, s1, s69 // (op 1049) 


s_add_i32 s1, s50, s51 // (op 1049) 


s_ashr_i32 s50, s1, s49 // (op 1049) 


s_lshl_b32 s1, s46, 24 // (op 1049) 
s_ashr_i32 s49, s1, 31 // (op 1049) 

s_xor_b32 s1, s50, s49 // (op 1049) 

s_sub_i32 s50, s1, s49 // (op 1049) 


s_mul_i32 s1, s45, s44 // (op 1049) 
s_add_u32 s68, s8, 16 // (op 1049) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 1049) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1049) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1049) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 1049) 

s_sub_i32 s49, s68, s1 // (op 1049) 

s_mul_i32 s51, s50, s49 // (op 1049) 


s_sub_u32 s49, s19, s51 // (op 1049) 


s_add_u32 s19, s49, s1 // (op 1049) 


s_cmp_eq_i32 s18, 0 // (op 1049) 
s_mov_b32 s1, scc // (op 1049) 

s_cmp_eq_i32 s1, 0 // (op 1049) 
s_cselect_b32 s18, s19, s0 // (op 1049) 



v_lshl_add_u32 v14, s18, 4, v19 // (op 1049) 


v_mul_lo_u32 v19, v14, s34 // (op 1049) 
v_add_u32 v19, v19, v15 // (op 1049) 


s_mul_i32 s1, s34, s9 // (op 1051) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1051) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1051) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1051) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1051) most significant: sum
s_add_u32 s1, s1, s19 // (op 1051) most significant: sum


s_mov_b32 s64, s36 // (op 1051) reg expression
s_mov_b32 s65, s37 // (op 1051) reg expression
s_mov_b32 s66, s0 // (op 1051) reg expression
s_mov_b32 s67, 131072 // (op 1051) call()

s_mul_hi_i32 s0, s16, s11 // (op 1094) 
v_and_b32 v14, -64, v1 // (op 1094) 
s_add_i32 s1, s0, s16 // (op 1094) 

v_sub_u32 v15, v1, v14 // (op 1094) 

s_ashr_i32 s0, s1, 31 // (op 1094) 
v_lshrrev_b32 v14, 6, v1 // (op 1094) 
s_cmp_eq_i32 s11, 0 // (op 1094) 
s_mov_b32 s18, scc // (op 1094) 
v_lshlrev_b32 v20, 1, v14 // (op 1094) 

s_cmp_eq_i32 s18, 0 // (op 1094) 
s_cselect_b32 s19, 0, -1 // (op 1094) 

s_and_b32 s18, s12, 31 // (op 1094) 
s_lshl_b32 s49, 1, s18 // (op 1094) 
s_add_i32 s50, s49, s19 // (op 1094) 


s_and_b32 s19, s0, s50 // (op 1094) 


s_add_i32 s0, s1, s19 // (op 1094) 


s_ashr_i32 s1, s0, s18 // (op 1094) 


s_lshl_b32 s0, s12, 24 // (op 1094) 
s_ashr_i32 s18, s0, 31 // (op 1094) 

s_xor_b32 s0, s1, s18 // (op 1094) 

s_sub_i32 s1, s0, s18 // (op 1094) 


s_mul_hi_i32 s0, s1, s13 // (op 1094) 
s_add_i32 s18, s0, s1 // (op 1094) 

s_ashr_i32 s0, s18, 31 // (op 1094) 
s_cmp_eq_i32 s13, 0 // (op 1094) 
s_mov_b32 s19, scc // (op 1094) 
s_cmp_eq_i32 s19, 0 // (op 1094) 
s_cselect_b32 s49, 0, -1 // (op 1094) 

s_and_b32 s19, s14, 31 // (op 1094) 
s_lshl_b32 s50, 1, s19 // (op 1094) 
s_add_i32 s51, s50, s49 // (op 1094) 


s_and_b32 s49, s0, s51 // (op 1094) 


s_add_i32 s0, s18, s49 // (op 1094) 


s_ashr_i32 s18, s0, s19 // (op 1094) 


s_lshl_b32 s0, s14, 24 // (op 1094) 
s_ashr_i32 s19, s0, 31 // (op 1094) 

s_xor_b32 s0, s18, s19 // (op 1094) 

s_sub_i32 s18, s0, s19 // (op 1094) 


s_add_u32 s50, s6, 64 // (op 1094) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1094) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1094) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1094) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1094) 

s_mul_i32 s0, s18, s50 // (op 1094) 

s_sub_i32 s49, s1, s0 // (op 1094) 


s_mul_hi_i32 s0, s16, s10 // (op 1094) 
s_add_i32 s1, s0, s16 // (op 1094) 

s_ashr_i32 s0, s1, 31 // (op 1094) 
s_cmp_eq_i32 s10, 0 // (op 1094) 
s_mov_b32 s18, scc // (op 1094) 
s_cmp_eq_i32 s18, 0 // (op 1094) 
s_cselect_b32 s19, 0, -1 // (op 1094) 

s_and_b32 s18, s47, 31 // (op 1094) 
s_lshl_b32 s68, 1, s18 // (op 1094) 
s_add_i32 s69, s68, s19 // (op 1094) 


s_and_b32 s19, s0, s69 // (op 1094) 


s_add_i32 s0, s1, s19 // (op 1094) 


s_ashr_i32 s1, s0, s18 // (op 1094) 


s_lshl_b32 s0, s47, 24 // (op 1094) 
s_ashr_i32 s18, s0, 31 // (op 1094) 

s_xor_b32 s0, s1, s18 // (op 1094) 

s_sub_i32 s1, s0, s18 // (op 1094) 


s_mul_i32 s0, s44, s50 // (op 1094) 

s_mul_i32 s18, s48, s0 // (op 1094) 

s_mul_i32 s0, s1, s18 // (op 1094) 

s_sub_u32 s18, s16, s0 // (op 1094) 

s_mul_hi_i32 s0, s18, s15 // (op 1094) 
s_add_i32 s19, s0, s18 // (op 1094) 


s_ashr_i32 s0, s19, 31 // (op 1094) 
s_cmp_eq_i32 s15, 0 // (op 1094) 
s_mov_b32 s18, scc // (op 1094) 
s_cmp_eq_i32 s18, 0 // (op 1094) 
s_cselect_b32 s50, 0, -1 // (op 1094) 

s_and_b32 s18, s46, 31 // (op 1094) 
s_lshl_b32 s51, 1, s18 // (op 1094) 
s_add_i32 s68, s51, s50 // (op 1094) 


s_and_b32 s50, s0, s68 // (op 1094) 


s_add_i32 s0, s19, s50 // (op 1094) 


s_ashr_i32 s19, s0, s18 // (op 1094) 


s_lshl_b32 s0, s46, 24 // (op 1094) 
s_ashr_i32 s18, s0, 31 // (op 1094) 

s_xor_b32 s0, s19, s18 // (op 1094) 

s_sub_i32 s19, s0, s18 // (op 1094) 


s_cmp_eq_i32 s1, 0 // (op 1094) 
s_mov_b32 s0, scc // (op 1094) 

s_cmp_eq_i32 s0, 0 // (op 1094) 
s_cselect_b32 s1, s19, s49 // (op 1094) 



v_lshl_add_u32 v14, s1, 6, v15 // (op 1094) 


v_mul_lo_u32 v15, v20, s38 // (op 1094) 
v_add_u32 v15, v15, v14 // (op 1094) 


s_mul_i32 s1, s38, s43 // (op 1096) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1096) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1096) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1096) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1096) most significant: sum
s_add_u32 s1, s1, s19 // (op 1096) most significant: sum


s_mov_b32 s68, s40 // (op 1096) reg expression
s_mov_b32 s69, s41 // (op 1096) reg expression
s_mov_b32 s70, s0 // (op 1096) reg expression
s_mov_b32 s71, 131072 // (op 1096) call()

s_mul_hi_i32 s0, s16, s11 // (op 1113) 
v_and_b32 v14, -64, v1 // (op 1113) 
s_add_i32 s1, s0, s16 // (op 1113) 

v_sub_u32 v20, v1, v14 // (op 1113) 

s_ashr_i32 s0, s1, 31 // (op 1113) 
v_lshrrev_b32 v14, 6, v1 // (op 1113) 
s_cmp_eq_i32 s11, 0 // (op 1113) 
s_mov_b32 s18, scc // (op 1113) 
v_lshlrev_b32 v21, 1, v14 // (op 1113) 

s_cmp_eq_i32 s18, 0 // (op 1113) 
s_cselect_b32 s19, 0, -1 // (op 1113) 

s_and_b32 s18, s12, 31 // (op 1113) 
s_lshl_b32 s49, 1, s18 // (op 1113) 
s_add_i32 s50, s49, s19 // (op 1113) 


s_and_b32 s19, s0, s50 // (op 1113) 


s_add_i32 s0, s1, s19 // (op 1113) 


s_ashr_i32 s1, s0, s18 // (op 1113) 


s_lshl_b32 s0, s12, 24 // (op 1113) 
s_ashr_i32 s18, s0, 31 // (op 1113) 

s_xor_b32 s0, s1, s18 // (op 1113) 

s_sub_i32 s1, s0, s18 // (op 1113) 


s_mul_hi_i32 s0, s1, s13 // (op 1113) 
s_add_i32 s18, s0, s1 // (op 1113) 

s_ashr_i32 s0, s18, 31 // (op 1113) 
s_cmp_eq_i32 s13, 0 // (op 1113) 
s_mov_b32 s19, scc // (op 1113) 
s_cmp_eq_i32 s19, 0 // (op 1113) 
s_cselect_b32 s49, 0, -1 // (op 1113) 

s_and_b32 s19, s14, 31 // (op 1113) 
s_lshl_b32 s50, 1, s19 // (op 1113) 
s_add_i32 s51, s50, s49 // (op 1113) 


s_and_b32 s49, s0, s51 // (op 1113) 


s_add_i32 s0, s18, s49 // (op 1113) 


s_ashr_i32 s18, s0, s19 // (op 1113) 


s_lshl_b32 s0, s14, 24 // (op 1113) 
s_ashr_i32 s19, s0, 31 // (op 1113) 

s_xor_b32 s0, s18, s19 // (op 1113) 

s_sub_i32 s18, s0, s19 // (op 1113) 


s_add_u32 s50, s6, 64 // (op 1113) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1113) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1113) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1113) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1113) 

s_mul_i32 s0, s18, s50 // (op 1113) 

s_sub_i32 s49, s1, s0 // (op 1113) 


s_mul_hi_i32 s0, s16, s10 // (op 1113) 
s_add_i32 s1, s0, s16 // (op 1113) 

s_ashr_i32 s0, s1, 31 // (op 1113) 
s_cmp_eq_i32 s10, 0 // (op 1113) 
s_mov_b32 s18, scc // (op 1113) 
s_cmp_eq_i32 s18, 0 // (op 1113) 
s_cselect_b32 s19, 0, -1 // (op 1113) 

s_and_b32 s18, s47, 31 // (op 1113) 
s_lshl_b32 s72, 1, s18 // (op 1113) 
s_add_i32 s73, s72, s19 // (op 1113) 


s_and_b32 s19, s0, s73 // (op 1113) 


s_add_i32 s0, s1, s19 // (op 1113) 


s_ashr_i32 s1, s0, s18 // (op 1113) 


s_lshl_b32 s0, s47, 24 // (op 1113) 
s_ashr_i32 s18, s0, 31 // (op 1113) 

s_xor_b32 s0, s1, s18 // (op 1113) 

s_sub_i32 s1, s0, s18 // (op 1113) 


s_mul_i32 s0, s44, s50 // (op 1113) 

s_mul_i32 s18, s48, s0 // (op 1113) 

s_mul_i32 s0, s1, s18 // (op 1113) 

s_sub_u32 s18, s16, s0 // (op 1113) 

s_mul_hi_i32 s0, s18, s15 // (op 1113) 
s_add_i32 s19, s0, s18 // (op 1113) 


s_ashr_i32 s0, s19, 31 // (op 1113) 
s_cmp_eq_i32 s15, 0 // (op 1113) 
s_mov_b32 s18, scc // (op 1113) 
s_cmp_eq_i32 s18, 0 // (op 1113) 
s_cselect_b32 s50, 0, -1 // (op 1113) 

s_and_b32 s18, s46, 31 // (op 1113) 
s_lshl_b32 s51, 1, s18 // (op 1113) 
s_add_i32 s72, s51, s50 // (op 1113) 


s_and_b32 s50, s0, s72 // (op 1113) 


s_add_i32 s0, s19, s50 // (op 1113) 


s_ashr_i32 s19, s0, s18 // (op 1113) 


s_lshl_b32 s0, s46, 24 // (op 1113) 
s_ashr_i32 s18, s0, 31 // (op 1113) 

s_xor_b32 s0, s19, s18 // (op 1113) 

s_sub_i32 s19, s0, s18 // (op 1113) 


s_cmp_eq_i32 s1, 0 // (op 1113) 
s_mov_b32 s0, scc // (op 1113) 

s_cmp_eq_i32 s0, 0 // (op 1113) 
s_cselect_b32 s1, s19, s49 // (op 1113) 



v_lshl_add_u32 v14, s1, 6, v20 // (op 1113) 


v_mul_lo_u32 v20, v21, s38 // (op 1113) 
v_add_u32 v20, v20, v14 // (op 1113) 


s_mul_i32 s1, s38, s43 // (op 1115) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1115) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1115) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1115) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1115) most significant: sum
s_add_u32 s1, s1, s19 // (op 1115) most significant: sum


s_mov_b32 s68, s40 // (op 1115) reg expression
s_mov_b32 s69, s41 // (op 1115) reg expression
s_mov_b32 s70, s0 // (op 1115) reg expression
s_mov_b32 s71, 131072 // (op 1115) call()




buffer_load_ubyte v14, v18, s[64:67], 0 offen offset: 0 // (op 769) Load value


v_mov_b32 v21, v16 // (op 773) reg expression

buffer_load_dwordx3 v[22:24], v21, s[52:55], 0 offen // (op 773) Load value
buffer_load_dwordx3 v[26:28], v21, s[52:55], 0 offen offset:12 // (op 773) Load value
buffer_load_dwordx3 v[30:32], v21, s[52:55], 0 offen offset:24 // (op 773) Load value
buffer_load_dwordx3 v[34:36], v21, s[52:55], 0 offen offset:36 // (op 773) Load value


v_mov_b64 v[38:39], s[38:39] // (op 776) convert

v_mov_b32 v21, v15 // (op 776) reg expression
buffer_load_ubyte v40, v21, s[68:71], 0 offen offset: 0 // (op 776) Load value
v_add_u32 v21, v21, v38 // (op 776) 
buffer_load_ubyte v41, v21, s[68:71], 0 offen offset: 0 // (op 776) Load value





buffer_load_dwordx3 v[42:44], v12, s[20:23], 0 offen // (op 779) Load value



buffer_load_ubyte v21, v18, s[64:67], 0 offen offset:4 // (op 783) Load value



s_lshl_b64 s[0:1], s[28:29], 7 // (op 787) 
s_mul_i32 s19, s1, 6 // (op 787) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 787) least significant
s_mul_hi_u32 s49, s0, 6 // (op 787) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 787) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 787) 

v_add_u32 v18, v16, s0 // (op 787) 


buffer_load_dwordx3 v[46:48], v18, s[52:55], 0 offen // (op 787) Load value
buffer_load_dwordx3 v[50:52], v18, s[52:55], 0 offen offset:12 // (op 787) Load value
buffer_load_dwordx3 v[54:56], v18, s[52:55], 0 offen offset:24 // (op 787) Load value
buffer_load_dwordx3 v[58:60], v18, s[52:55], 0 offen offset:36 // (op 787) Load value



v_mov_b64 v[38:39], s[38:39] // (op 790) convert

s_lshl_b32 s0, s38, 2 // (op 790) 
v_add_u32 v16, v15, s0 // (op 790) 

buffer_load_ubyte v62, v16, s[68:71], 0 offen offset: 0 // (op 790) Load value
v_add_u32 v16, v16, v38 // (op 790) 
buffer_load_ubyte v63, v16, s[68:71], 0 offen offset: 0 // (op 790) Load value






buffer_load_dwordx3 v[64:66], v12, s[20:23], 0 offen offset:96 // (op 793) Load value

v_lshlrev_b32 v12, 6, v1 // (op 1233) 
v_mul_lo_u32 v18, v12, 6 // (op 1233) 
v_lshrrev_b32 v25, 3, v18 // (op 1233) 

v_lshrrev_b32 v18, 4, v12 // (op 1233) 

v_lshlrev_b32 v12, 2, v18 // (op 1233) 

v_add_u32 v18, v25, v12 // (op 1233) 





s_waitcnt vmcnt(14)
ds_write_b96 v18, v[22:24] // (op 795) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v18, v[26:28] offset:16 // (op 795) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v18, v[30:32] offset:32 // (op 795) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v18, v[34:36] offset:48 // (op 795) Store local data 

v_and_b32 v12, -64, v1 // (op 1291) 
v_sub_u32 v18, v1, v12 // (op 1291) 

v_lshrrev_b32 v12, 6, v1 // (op 1291) 
v_lshl_add_u32 v25, v12, 7, v18 // (op 1291) 





s_waitcnt vmcnt(10)
ds_write_b8 v25, v40 offset:8192 // (op 796) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v25, v41 offset:8256 // (op 796) Store local data 

v_and_b32 v12, -8, v1 // (op 1180) 
v_sub_u32 v18, v1, v12 // (op 1180) 

v_lshlrev_b32 v12, 4, v18 // (op 1180) 

v_lshrrev_b32 v18, 3, v1 // (op 1180) 
v_lshl_add_u32 v25, v18, 7, v12 // (op 1180) 


v_mul_lo_u32 v12, v25, 6 // (op 1180) 

v_lshrrev_b32 v18, 3, v12 // (op 1180) 




s_waitcnt vmcnt(8)
ds_write_b96 v18, v[42:44] offset:8448 // (op 797) Store local data 

v_and_b32 v12, -8, v1 // (op 1151) 
v_sub_u32 v18, v1, v12 // (op 1151) 

v_lshlrev_b32 v12, 4, v18 // (op 1151) 

v_lshrrev_b32 v18, 3, v1 // (op 1151) 
v_lshl_add_u32 v25, v18, 7, v12 // (op 1151) 


v_mul_lo_u32 v12, v25, 6 // (op 1151) 

v_lshrrev_b32 v18, 3, v12 // (op 1151) 

v_lshlrev_b32 v12, 6, v1 // (op 1204) 
v_mul_lo_u32 v25, v12, 6 // (op 1204) 
v_lshrrev_b32 v29, 3, v25 // (op 1204) 

v_lshrrev_b32 v25, 4, v12 // (op 1204) 

v_lshlrev_b32 v12, 2, v25 // (op 1204) 

v_add_u32 v25, v29, v12 // (op 1204) 


v_and_b32 v12, -64, v1 // (op 1262) 
v_sub_u32 v29, v1, v12 // (op 1262) 

v_lshrrev_b32 v12, 6, v1 // (op 1262) 
v_lshl_add_u32 v33, v12, 7, v29 // (op 1262) 


v_and_b32 v12, -8, v1 // (op 1338) 
v_sub_u32 v29, v1, v12 // (op 1338) 

v_lshlrev_b32 v12, 4, v29 // (op 1338) 

v_lshrrev_b32 v29, 3, v1 // (op 1338) 
v_lshl_add_u32 v45, v29, 7, v12 // (op 1338) 


v_mul_lo_u32 v12, v45, 6 // (op 1338) 

v_lshrrev_b32 v29, 3, v12 // (op 1338) 

v_lshlrev_b32 v12, 6, v1 // (op 1362) 
v_mul_lo_u32 v45, v12, 6 // (op 1362) 
v_lshrrev_b32 v49, 3, v45 // (op 1362) 

v_lshrrev_b32 v45, 4, v12 // (op 1362) 

v_lshlrev_b32 v12, 2, v45 // (op 1362) 

v_add_u32 v45, v49, v12 // (op 1362) 


v_and_b32 v12, -64, v1 // (op 1391) 
v_sub_u32 v49, v1, v12 // (op 1391) 

v_lshrrev_b32 v12, 6, v1 // (op 1391) 
v_lshl_add_u32 v53, v12, 7, v49 // (op 1391) 


s_mov_b32 s0, 0 // (op 312) call()
s_ashr_i64 s[18:19], s[32:33], 7 // (op 311) 
s_mov_b32 s1, s18 // (op 311) convert

s_and_b32 s18, s1, 4294967294 // (op 311) 

s_cmp_lt_u32 s0, s18 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_311 // (op 311) Condition: Top (jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_311 if false)
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_311:
s_add_u32 s0, s0, 2 // (op 313) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1449) 



ds_read_b128 v[68:71], v3 offset:8448 // (op 67) Load local data 
ds_read_b64 v[72:73], v3 offset:8464 // (op 67) Load local data 
s_barrier  // (op 1462) 



ds_read_b96_tr_b6 v[74:76], v0 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[77:79], v0 offset:1024 // (op 72) Transpose load local data 
s_barrier  // (op 1476) 


ds_read_u8 v12, v7 offset:8192 // (op 81) Load local data 



ds_read_b96_tr_b6 v[80:82], v0 offset:32 // (op 343) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:73], v[74:79], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 84) 
ds_read_b96_tr_b6 v[83:85], v0 offset:1056 // (op 343) Transpose load local data 




ds_read_u8 v12, v7 offset:8224 // (op 346) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:73], v[80:85], a[4:7], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 347) 





v_mov_b32 v12, v19 // (op 77) 

buffer_load_ubyte v14, v12, s[64:67], 0 offen offset:8 // (op 77) Load value


v_mov_b32 v49, v17 // (op 10) 
s_lshl_b64 s[18:19], s[28:29], 7 // (op 10) 
s_mul_i32 s51, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 10) least significant
s_mul_hi_u32 s1, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 10) 

s_lshl_b32 s1, s18, 1 // (op 10) 

v_add_u32 v57, v49, s1 // (op 10) 


buffer_load_dwordx3 v[22:24], v57, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[26:28], v57, s[52:55], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[30:32], v57, s[52:55], 0 offen offset:24 // (op 10) Load value
buffer_load_dwordx3 v[34:36], v57, s[52:55], 0 offen offset:36 // (op 10) Load value


v_mov_b64 v[38:39], s[38:39] // (op 21) convert

v_mov_b32 v37, v20 // (op 21) 
s_lshl_b32 s1, s38, 3 // (op 21) 
v_add_u32 v57, v37, s1 // (op 21) 

buffer_load_ubyte v40, v57, s[68:71], 0 offen offset: 0 // (op 21) Load value
v_add_u32 v57, v57, v38 // (op 21) 
buffer_load_ubyte v41, v57, s[68:71], 0 offen offset: 0 // (op 21) Load value




v_mov_b32 v57, v13 // (op 4) 

buffer_load_dwordx3 v[42:44], v57, s[20:23], 0 offen offset:192 // (op 4) Load value
s_barrier  // (op 1499) 



s_waitcnt vmcnt(14)
ds_write_b96 v45, v[46:48] offset:9984 // (op 239) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v45, v[50:52] offset:10000 // (op 239) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v45, v[54:56] offset:10016 // (op 239) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v45, v[58:60] offset:10032 // (op 239) Store local data 
s_barrier  // (op 1505) 



s_waitcnt vmcnt(10)
ds_write_b8 v53, v62 offset:18176 // (op 240) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v53, v63 offset:18240 // (op 240) Store local data 
s_barrier  // (op 1493) 



s_waitcnt vmcnt(8)
ds_write_b96 v29, v[64:66] offset:18432 // (op 229) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1490) 



ds_read_b128 v[68:71], v4 offset:18432 // (op 231) Load local data 
ds_read_b64 v[72:73], v4 offset:18448 // (op 231) Load local data 
s_barrier  // (op 1496) 
s_barrier  // (op 1546) 



ds_read_b96_tr_b6 v[74:76], v10 offset:9984 // (op 233) Transpose load local data 
ds_read_b96_tr_b6 v[77:79], v10 offset:11008 // (op 233) Transpose load local data 
s_barrier  // (op 1502) 


ds_read_u8 v61, v11 offset:18176 // (op 237) Load local data 



ds_read_b96_tr_b6 v[80:82], v10 offset:10016 // (op 352) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:73], v[74:79], a[0:3], v21, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 238) 
ds_read_b96_tr_b6 v[83:85], v10 offset:11040 // (op 352) Transpose load local data 




s_barrier  // (op 1580) 
ds_read_u8 v61, v11 offset:18208 // (op 355) Load local data 
s_barrier  // (op 1584) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:73], v[80:85], a[4:7], v21, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 348) 






buffer_load_ubyte v21, v12, s[64:67], 0 offen offset:12 // (op 235) Load value



v_add_u32 v19, 8, v19 // (op 1041) 
s_lshl_b64 s[18:19], s[28:29], 7 // (op 241) 
s_mul_i32 s51, s19, 6 // (op 241) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 241) least significant
s_mul_hi_u32 s1, s18, 6 // (op 241) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 241) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 241) 

s_mul_i32 s1, 3, s18 // (op 241) 

v_add_u32 v12, v49, s1 // (op 241) 


buffer_load_dwordx3 v[46:48], v12, s[52:55], 0 offen // (op 241) Load value
buffer_load_dwordx3 v[50:52], v12, s[52:55], 0 offen offset:12 // (op 241) Load value
buffer_load_dwordx3 v[54:56], v12, s[52:55], 0 offen offset:24 // (op 241) Load value
buffer_load_dwordx3 v[58:60], v12, s[52:55], 0 offen offset:36 // (op 241) Load value



v_mov_b64 v[38:39], s[38:39] // (op 242) convert

s_lshl_b32 s1, s38, 2 // (op 242) 
s_mul_i32 s18, 3, s1 // (op 242) 

v_add_u32 v12, v37, s18 // (op 242) 

s_lshl_b64 s[18:19], s[28:29], 7 // (op 977) 
s_mul_i32 s51, s19, 6 // (op 977) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 977) least significant
s_mul_hi_u32 s1, s18, 6 // (op 977) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 977) most significant: sum


buffer_load_ubyte v62, v12, s[68:71], 0 offen offset: 0 // (op 242) Load value
v_add_u32 v12, v12, v38 // (op 242) 
s_lshr_b64 s[18:19], s[50:51], 3 // (op 977) 

s_lshl_b32 s1, s18, 1 // (op 977) 

v_add_u32 v17, v17, s1 // (op 977) 

buffer_load_ubyte v63, v12, s[68:71], 0 offen offset: 0 // (op 242) Load value






s_lshl_b32 s1, s38, 3 // (op 1105) 
v_add_u32 v20, v20, s1 // (op 1105) 

buffer_load_dwordx3 v[64:66], v57, s[20:23], 0 offen offset:288 // (op 228) Load value

s_barrier  // (op 1465) 
v_add_u32 v13, 192, v13 // (op 913) 



s_waitcnt vmcnt(14)
ds_write_b96 v25, v[22:24] // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v25, v[26:28] offset:16 // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v25, v[30:32] offset:32 // (op 13) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v25, v[34:36] offset:48 // (op 13) Store local data 
s_barrier  // (op 1479) 



s_waitcnt vmcnt(10)
ds_write_b8 v33, v40 offset:8192 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v33, v41 offset:8256 // (op 26) Store local data 
s_barrier  // (op 1452) 



s_waitcnt vmcnt(8)
ds_write_b96 v18, v[42:44] offset:8448 // (op 6) Store local data 
s_ashr_i64 s[18:19], s[32:33], 7 // (op 311) 
s_mov_b32 s1, s18 // (op 311) convert

s_and_b32 s18, s1, 4294967294 // (op 311) 

s_cmp_lt_u32 s0, s18 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_311 // (op 311) Condition: Bottom (jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_311 if true)
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_311:















s_mul_hi_i32 s0, s16, s11 // (op 875) 
v_lshrrev_b32 v49, 3, v1 // (op 875) 
s_add_i32 s1, s0, s16 // (op 875) 

v_and_b32 v53, -8, v1 // (op 875) 
s_ashr_i32 s0, s1, 31 // (op 875) 
v_sub_u32 v57, v1, v53 // (op 875) 

s_cmp_eq_i32 s11, 0 // (op 875) 
s_mov_b32 s18, scc // (op 875) 
v_lshlrev_b32 v53, 4, v57 // (op 875) 

s_cmp_eq_i32 s18, 0 // (op 875) 
s_cselect_b32 s19, 0, -1 // (op 875) 

s_and_b32 s18, s12, 31 // (op 875) 
s_lshl_b32 s49, 1, s18 // (op 875) 
s_add_i32 s50, s49, s19 // (op 875) 


s_and_b32 s19, s0, s50 // (op 875) 


s_add_i32 s0, s1, s19 // (op 875) 


s_ashr_i32 s1, s0, s18 // (op 875) 


s_lshl_b32 s0, s12, 24 // (op 875) 
s_ashr_i32 s18, s0, 31 // (op 875) 

s_xor_b32 s0, s1, s18 // (op 875) 

s_sub_i32 s1, s0, s18 // (op 875) 


s_mul_hi_i32 s0, s1, s13 // (op 875) 
s_add_i32 s18, s0, s1 // (op 875) 

s_ashr_i32 s0, s18, 31 // (op 875) 
s_cmp_eq_i32 s13, 0 // (op 875) 
s_mov_b32 s19, scc // (op 875) 
s_cmp_eq_i32 s19, 0 // (op 875) 
s_cselect_b32 s49, 0, -1 // (op 875) 

s_and_b32 s19, s14, 31 // (op 875) 
s_lshl_b32 s50, 1, s19 // (op 875) 
s_add_i32 s51, s50, s49 // (op 875) 


s_and_b32 s49, s0, s51 // (op 875) 


s_add_i32 s0, s18, s49 // (op 875) 


s_ashr_i32 s18, s0, s19 // (op 875) 


s_lshl_b32 s0, s14, 24 // (op 875) 
s_ashr_i32 s19, s0, 31 // (op 875) 

s_xor_b32 s0, s18, s19 // (op 875) 

s_sub_i32 s18, s0, s19 // (op 875) 


s_mul_i32 s0, s1, s44 // (op 875) 

s_sub_u32 s1, s16, s0 // (op 875) 

s_mul_i32 s0, s18, s44 // (op 875) 
s_add_u32 s0, s0, s1 // (op 875) 


s_mul_hi_i32 s1, s16, s10 // (op 875) 
s_add_i32 s18, s1, s16 // (op 875) 

s_ashr_i32 s1, s18, 31 // (op 875) 
s_cmp_eq_i32 s10, 0 // (op 875) 
s_mov_b32 s19, scc // (op 875) 
s_cmp_eq_i32 s19, 0 // (op 875) 
s_cselect_b32 s49, 0, -1 // (op 875) 

s_and_b32 s19, s47, 31 // (op 875) 
s_lshl_b32 s50, 1, s19 // (op 875) 
s_add_i32 s51, s50, s49 // (op 875) 


s_and_b32 s49, s1, s51 // (op 875) 


s_add_i32 s1, s18, s49 // (op 875) 


s_ashr_i32 s18, s1, s19 // (op 875) 


s_lshl_b32 s1, s47, 24 // (op 875) 
s_ashr_i32 s19, s1, 31 // (op 875) 

s_xor_b32 s1, s18, s19 // (op 875) 

s_sub_i32 s18, s1, s19 // (op 875) 


s_add_u32 s50, s6, 64 // (op 875) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 875) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 875) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 875) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 875) 

s_mul_i32 s1, s44, s50 // (op 875) 

s_mul_i32 s19, s48, s1 // (op 875) 

s_mul_i32 s1, s18, s19 // (op 875) 

s_sub_u32 s19, s16, s1 // (op 875) 

s_mov_b32 s1, s19 // (op 875) 
s_mul_hi_i32 s49, s1, s15 // (op 875) 
s_add_i32 s50, s49, s1 // (op 875) 


s_ashr_i32 s1, s50, 31 // (op 875) 
s_cmp_eq_i32 s15, 0 // (op 875) 
s_mov_b32 s49, scc // (op 875) 
s_cmp_eq_i32 s49, 0 // (op 875) 
s_cselect_b32 s51, 0, -1 // (op 875) 

s_and_b32 s49, s46, 31 // (op 875) 
s_lshl_b32 s72, 1, s49 // (op 875) 
s_add_i32 s73, s72, s51 // (op 875) 


s_and_b32 s51, s1, s73 // (op 875) 


s_add_i32 s1, s50, s51 // (op 875) 


s_ashr_i32 s50, s1, s49 // (op 875) 


s_lshl_b32 s1, s46, 24 // (op 875) 
s_ashr_i32 s49, s1, 31 // (op 875) 

s_xor_b32 s1, s50, s49 // (op 875) 

s_sub_i32 s50, s1, s49 // (op 875) 


s_mul_i32 s1, s45, s44 // (op 875) 
s_add_u32 s72, s8, 16 // (op 875) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 875) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 875) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 875) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 875) 

s_sub_i32 s49, s72, s1 // (op 875) 

s_mul_i32 s51, s50, s49 // (op 875) 


s_sub_u32 s49, s19, s51 // (op 875) 


s_add_u32 s19, s49, s1 // (op 875) 


s_cmp_eq_i32 s18, 0 // (op 875) 
s_mov_b32 s1, scc // (op 875) 

s_cmp_eq_i32 s1, 0 // (op 875) 
s_cselect_b32 s18, s19, s0 // (op 875) 



v_lshl_add_u32 v57, s18, 4, v49 // (op 875) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 875) 
s_lshr_b32 s18, s0, 1 // (op 875) 

v_lshl_add_u32 v49, s18, 8, v53 // (op 875) 


v_mov_b32 v13, 0 // (op 875) convert
v_mov_b32 v12, v49 // (op 875) convert

v_mul_lo_u32 v15, v57, s25 // (op 875) most significant: low of low * high
v_mul_hi_u32 v49, v57, s24 // (op 875) most significant: high of low * low
v_mul_lo_u32 v14, v57, s24 // (op 875) least significant: low of low * low
v_add_u32 v15, v15, v49 // (op 875) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 875) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 875) most significant half



v_mul_lo_u32 v13, v15, 6 // (op 875) most significant: low of high * low
v_mul_hi_u32 v49, v14, 6 // (op 875) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 875) least significant: low of low * low
v_add_u32 v13, v13, v49 // (op 875) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 875) 

v_mov_b32 v49, v14 // (op 875) convert

s_mul_i32 s1, s9, 6 // (op 877) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 877) least significant
s_mul_hi_u32 s18, s8, 6 // (op 877) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 877) most significant: sum

s_mul_i32 s19, s24, s1 // (op 877) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 877) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 877) least significant
s_mul_hi_u32 s50, s24, s0 // (op 877) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 877) most significant: sum
s_add_u32 s19, s19, s50 // (op 877) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 877) 

s_mov_b32 s20, s26 // (op 877) reg expression
s_mov_b32 s21, s27 // (op 877) reg expression
s_mov_b32 s22, s0 // (op 877) reg expression
s_mov_b32 s23, 131072 // (op 877) call()

s_mul_hi_i32 s0, s16, s11 // (op 943) 
s_add_i32 s1, s0, s16 // (op 943) 

s_ashr_i32 s0, s1, 31 // (op 943) 
s_cmp_eq_i32 s11, 0 // (op 943) 
s_mov_b32 s18, scc // (op 943) 
s_cmp_eq_i32 s18, 0 // (op 943) 
s_cselect_b32 s19, 0, -1 // (op 943) 

s_and_b32 s18, s12, 31 // (op 943) 
s_lshl_b32 s49, 1, s18 // (op 943) 
s_add_i32 s50, s49, s19 // (op 943) 


s_and_b32 s19, s0, s50 // (op 943) 


s_add_i32 s0, s1, s19 // (op 943) 


s_ashr_i32 s1, s0, s18 // (op 943) 


s_lshl_b32 s0, s12, 24 // (op 943) 
s_ashr_i32 s18, s0, 31 // (op 943) 

s_xor_b32 s0, s1, s18 // (op 943) 

s_sub_i32 s1, s0, s18 // (op 943) 


s_mul_hi_i32 s0, s1, s13 // (op 943) 
s_add_i32 s18, s0, s1 // (op 943) 

s_ashr_i32 s0, s18, 31 // (op 943) 
s_cmp_eq_i32 s13, 0 // (op 943) 
s_mov_b32 s19, scc // (op 943) 
s_cmp_eq_i32 s19, 0 // (op 943) 
s_cselect_b32 s49, 0, -1 // (op 943) 

s_and_b32 s19, s14, 31 // (op 943) 
s_lshl_b32 s50, 1, s19 // (op 943) 
s_add_i32 s51, s50, s49 // (op 943) 


s_and_b32 s49, s0, s51 // (op 943) 


s_add_i32 s0, s18, s49 // (op 943) 


s_ashr_i32 s18, s0, s19 // (op 943) 


s_lshl_b32 s0, s14, 24 // (op 943) 
s_ashr_i32 s19, s0, 31 // (op 943) 

s_xor_b32 s0, s18, s19 // (op 943) 

s_sub_i32 s18, s0, s19 // (op 943) 


s_add_u32 s50, s6, 64 // (op 943) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 943) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 943) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 943) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 943) 

s_mul_i32 s0, s18, s50 // (op 943) 

s_sub_i32 s49, s1, s0 // (op 943) 


s_mul_hi_i32 s0, s16, s10 // (op 943) 
s_add_i32 s1, s0, s16 // (op 943) 

s_ashr_i32 s0, s1, 31 // (op 943) 
s_cmp_eq_i32 s10, 0 // (op 943) 
s_mov_b32 s18, scc // (op 943) 
s_cmp_eq_i32 s18, 0 // (op 943) 
s_cselect_b32 s19, 0, -1 // (op 943) 

s_and_b32 s18, s47, 31 // (op 943) 
s_lshl_b32 s72, 1, s18 // (op 943) 
s_add_i32 s73, s72, s19 // (op 943) 


s_and_b32 s19, s0, s73 // (op 943) 


s_add_i32 s0, s1, s19 // (op 943) 


s_ashr_i32 s1, s0, s18 // (op 943) 


s_lshl_b32 s0, s47, 24 // (op 943) 
s_ashr_i32 s18, s0, 31 // (op 943) 

s_xor_b32 s0, s1, s18 // (op 943) 

s_sub_i32 s1, s0, s18 // (op 943) 


s_mul_i32 s0, s44, s50 // (op 943) 

s_mul_i32 s18, s48, s0 // (op 943) 

s_mul_i32 s0, s1, s18 // (op 943) 

s_sub_u32 s18, s16, s0 // (op 943) 

s_mul_hi_i32 s0, s18, s15 // (op 943) 
s_add_i32 s19, s0, s18 // (op 943) 


s_ashr_i32 s0, s19, 31 // (op 943) 
s_cmp_eq_i32 s15, 0 // (op 943) 
s_mov_b32 s18, scc // (op 943) 
s_cmp_eq_i32 s18, 0 // (op 943) 
s_cselect_b32 s50, 0, -1 // (op 943) 

s_and_b32 s18, s46, 31 // (op 943) 
s_lshl_b32 s51, 1, s18 // (op 943) 
s_add_i32 s72, s51, s50 // (op 943) 


s_and_b32 s50, s0, s72 // (op 943) 


s_add_i32 s0, s19, s50 // (op 943) 


s_ashr_i32 s19, s0, s18 // (op 943) 


s_lshl_b32 s0, s46, 24 // (op 943) 
s_ashr_i32 s18, s0, 31 // (op 943) 

s_xor_b32 s0, s19, s18 // (op 943) 

s_sub_i32 s19, s0, s18 // (op 943) 


s_cmp_eq_i32 s1, 0 // (op 943) 
s_mov_b32 s0, scc // (op 943) 

s_cmp_eq_i32 s0, 0 // (op 943) 
s_cselect_b32 s1, s19, s49 // (op 943) 



s_lshl_b32 s0, s1, 6 // (op 943) 

s_mov_b32 s19, 0 // (op 943) convert
s_mov_b32 s18, s0 // (op 943) convert

s_ashr_i64 s[0:1], s[32:33], 7 // (op 943) 
s_lshr_b32 s49, s0, 1 // (op 943) 

v_lshl_add_u32 v53, s49, 8, v1 // (op 943) 

v_mul_lo_u32 v13, v53, s29 // (op 943) most significant: low of low * high
v_mul_hi_u32 v57, v53, s28 // (op 943) most significant: high of low * low
v_mul_lo_u32 v12, v53, s28 // (op 943) least significant: low of low * low
v_add_u32 v13, v13, v57 // (op 943) most significant: sum

v_mov_b32 v57, s18 // (op 943) 
v_mov_b32 v61, s19 // (op 943) 
v_add_co_u32 v12, s[0:1], v12, v57 // (op 943) least significant half
v_addc_co_u32 v13, s[0:1], v13, v61, s[0:1] // (op 943) most significant half





v_mul_lo_u32 v15, v13, 6 // (op 943) most significant: low of high * low
v_mul_hi_u32 v53, v12, 6 // (op 943) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 943) least significant: low of low * low
v_add_u32 v15, v15, v53 // (op 943) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 943) 

v_mov_b32 v53, v12 // (op 943) convert

s_mul_i32 s1, s33, 6 // (op 945) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 945) least significant
s_mul_hi_u32 s18, s32, 6 // (op 945) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 945) most significant: sum

s_mul_i32 s19, s28, s1 // (op 945) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 945) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 945) least significant
s_mul_hi_u32 s50, s28, s0 // (op 945) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 945) most significant: sum
s_add_u32 s19, s19, s50 // (op 945) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 945) 

s_mov_b32 s52, s30 // (op 945) reg expression
s_mov_b32 s53, s31 // (op 945) reg expression
s_mov_b32 s54, s0 // (op 945) reg expression
s_mov_b32 s55, 131072 // (op 945) call()

s_mul_hi_i32 s0, s16, s11 // (op 1007) 
v_lshrrev_b32 v57, 7, v1 // (op 1007) 
s_add_i32 s1, s0, s16 // (op 1007) 

v_and_b32 v61, 15, v1 // (op 1007) 
s_ashr_i32 s0, s1, 31 // (op 1007) 
v_lshl_add_u32 v12, v57, 4, v61 // (op 1007) 


s_cmp_eq_i32 s11, 0 // (op 1007) 
s_mov_b32 s18, scc // (op 1007) 
v_and_b32 v57, 63, v1 // (op 1007) 
s_cmp_eq_i32 s18, 0 // (op 1007) 
s_cselect_b32 s19, 0, -1 // (op 1007) 

v_lshrrev_b32 v61, 4, v57 // (op 1007) 

s_and_b32 s18, s12, 31 // (op 1007) 
s_lshl_b32 s49, 1, s18 // (op 1007) 
s_add_i32 s50, s49, s19 // (op 1007) 


s_and_b32 s19, s0, s50 // (op 1007) 


s_add_i32 s0, s1, s19 // (op 1007) 


s_ashr_i32 s1, s0, s18 // (op 1007) 


s_lshl_b32 s0, s12, 24 // (op 1007) 
s_ashr_i32 s18, s0, 31 // (op 1007) 

s_xor_b32 s0, s1, s18 // (op 1007) 

s_sub_i32 s1, s0, s18 // (op 1007) 


s_mul_hi_i32 s0, s1, s13 // (op 1007) 
s_add_i32 s18, s0, s1 // (op 1007) 

s_ashr_i32 s0, s18, 31 // (op 1007) 
s_cmp_eq_i32 s13, 0 // (op 1007) 
s_mov_b32 s19, scc // (op 1007) 
s_cmp_eq_i32 s19, 0 // (op 1007) 
s_cselect_b32 s49, 0, -1 // (op 1007) 

s_and_b32 s19, s14, 31 // (op 1007) 
s_lshl_b32 s50, 1, s19 // (op 1007) 
s_add_i32 s51, s50, s49 // (op 1007) 


s_and_b32 s49, s0, s51 // (op 1007) 


s_add_i32 s0, s18, s49 // (op 1007) 


s_ashr_i32 s18, s0, s19 // (op 1007) 


s_lshl_b32 s0, s14, 24 // (op 1007) 
s_ashr_i32 s19, s0, 31 // (op 1007) 

s_xor_b32 s0, s18, s19 // (op 1007) 

s_sub_i32 s18, s0, s19 // (op 1007) 


s_mul_i32 s0, s1, s44 // (op 1007) 

s_sub_u32 s1, s16, s0 // (op 1007) 

s_mul_i32 s0, s18, s44 // (op 1007) 
s_add_u32 s0, s0, s1 // (op 1007) 


s_mul_hi_i32 s1, s16, s10 // (op 1007) 
s_add_i32 s18, s1, s16 // (op 1007) 

s_ashr_i32 s1, s18, 31 // (op 1007) 
s_cmp_eq_i32 s10, 0 // (op 1007) 
s_mov_b32 s19, scc // (op 1007) 
s_cmp_eq_i32 s19, 0 // (op 1007) 
s_cselect_b32 s49, 0, -1 // (op 1007) 

s_and_b32 s19, s47, 31 // (op 1007) 
s_lshl_b32 s50, 1, s19 // (op 1007) 
s_add_i32 s51, s50, s49 // (op 1007) 


s_and_b32 s49, s1, s51 // (op 1007) 


s_add_i32 s1, s18, s49 // (op 1007) 


s_ashr_i32 s18, s1, s19 // (op 1007) 


s_lshl_b32 s1, s47, 24 // (op 1007) 
s_ashr_i32 s19, s1, 31 // (op 1007) 

s_xor_b32 s1, s18, s19 // (op 1007) 

s_sub_i32 s18, s1, s19 // (op 1007) 


s_add_u32 s50, s6, 64 // (op 1007) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1007) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1007) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1007) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1007) 

s_mul_i32 s1, s44, s50 // (op 1007) 

s_mul_i32 s19, s48, s1 // (op 1007) 

s_mul_i32 s1, s18, s19 // (op 1007) 

s_sub_u32 s19, s16, s1 // (op 1007) 

s_mov_b32 s1, s19 // (op 1007) 
s_mul_hi_i32 s49, s1, s15 // (op 1007) 
s_add_i32 s50, s49, s1 // (op 1007) 


s_ashr_i32 s1, s50, 31 // (op 1007) 
s_cmp_eq_i32 s15, 0 // (op 1007) 
s_mov_b32 s49, scc // (op 1007) 
s_cmp_eq_i32 s49, 0 // (op 1007) 
s_cselect_b32 s51, 0, -1 // (op 1007) 

s_and_b32 s49, s46, 31 // (op 1007) 
s_lshl_b32 s72, 1, s49 // (op 1007) 
s_add_i32 s73, s72, s51 // (op 1007) 


s_and_b32 s51, s1, s73 // (op 1007) 


s_add_i32 s1, s50, s51 // (op 1007) 


s_ashr_i32 s50, s1, s49 // (op 1007) 


s_lshl_b32 s1, s46, 24 // (op 1007) 
s_ashr_i32 s49, s1, 31 // (op 1007) 

s_xor_b32 s1, s50, s49 // (op 1007) 

s_sub_i32 s50, s1, s49 // (op 1007) 


s_mul_i32 s1, s45, s44 // (op 1007) 
s_add_u32 s72, s8, 16 // (op 1007) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 1007) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1007) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1007) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 1007) 

s_sub_i32 s49, s72, s1 // (op 1007) 

s_mul_i32 s51, s50, s49 // (op 1007) 


s_sub_u32 s49, s19, s51 // (op 1007) 


s_add_u32 s19, s49, s1 // (op 1007) 


s_cmp_eq_i32 s18, 0 // (op 1007) 
s_mov_b32 s1, scc // (op 1007) 

s_cmp_eq_i32 s1, 0 // (op 1007) 
s_cselect_b32 s18, s19, s0 // (op 1007) 



v_lshl_add_u32 v57, s18, 4, v12 // (op 1007) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1007) 
s_lshr_b32 s18, s0, 1 // (op 1007) 

v_lshl_add_u32 v12, s18, 3, v61 // (op 1007) 


v_mul_lo_u32 v61, v57, s34 // (op 1007) 
v_add_u32 v61, v61, v12 // (op 1007) 


s_mul_i32 s1, s34, s9 // (op 1009) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1009) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1009) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1009) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1009) most significant: sum
s_add_u32 s1, s1, s19 // (op 1009) most significant: sum


s_mov_b32 s64, s36 // (op 1009) reg expression
s_mov_b32 s65, s37 // (op 1009) reg expression
s_mov_b32 s66, s0 // (op 1009) reg expression
s_mov_b32 s67, 131072 // (op 1009) call()

s_mul_hi_i32 s0, s16, s11 // (op 1071) 
v_and_b32 v57, -64, v1 // (op 1071) 
s_add_i32 s1, s0, s16 // (op 1071) 

v_sub_u32 v12, v1, v57 // (op 1071) 

s_ashr_i32 s0, s1, 31 // (op 1071) 
v_lshrrev_b32 v57, 6, v1 // (op 1071) 
s_cmp_eq_i32 s11, 0 // (op 1071) 
s_mov_b32 s18, scc // (op 1071) 
v_lshlrev_b32 v13, 1, v57 // (op 1071) 

s_cmp_eq_i32 s18, 0 // (op 1071) 
s_cselect_b32 s19, 0, -1 // (op 1071) 

s_and_b32 s18, s12, 31 // (op 1071) 
s_lshl_b32 s49, 1, s18 // (op 1071) 
s_add_i32 s50, s49, s19 // (op 1071) 


s_and_b32 s19, s0, s50 // (op 1071) 


s_add_i32 s0, s1, s19 // (op 1071) 


s_ashr_i32 s1, s0, s18 // (op 1071) 


s_lshl_b32 s0, s12, 24 // (op 1071) 
s_ashr_i32 s18, s0, 31 // (op 1071) 

s_xor_b32 s0, s1, s18 // (op 1071) 

s_sub_i32 s1, s0, s18 // (op 1071) 


s_mul_hi_i32 s0, s1, s13 // (op 1071) 
s_add_i32 s18, s0, s1 // (op 1071) 

s_ashr_i32 s0, s18, 31 // (op 1071) 
s_cmp_eq_i32 s13, 0 // (op 1071) 
s_mov_b32 s19, scc // (op 1071) 
s_cmp_eq_i32 s19, 0 // (op 1071) 
s_cselect_b32 s49, 0, -1 // (op 1071) 

s_and_b32 s19, s14, 31 // (op 1071) 
s_lshl_b32 s50, 1, s19 // (op 1071) 
s_add_i32 s51, s50, s49 // (op 1071) 


s_and_b32 s49, s0, s51 // (op 1071) 


s_add_i32 s0, s18, s49 // (op 1071) 


s_ashr_i32 s18, s0, s19 // (op 1071) 


s_lshl_b32 s0, s14, 24 // (op 1071) 
s_ashr_i32 s19, s0, 31 // (op 1071) 

s_xor_b32 s0, s18, s19 // (op 1071) 

s_sub_i32 s18, s0, s19 // (op 1071) 


s_add_u32 s50, s6, 64 // (op 1071) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1071) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1071) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1071) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1071) 

s_mul_i32 s0, s18, s50 // (op 1071) 

s_sub_i32 s49, s1, s0 // (op 1071) 


s_mul_hi_i32 s0, s16, s10 // (op 1071) 
s_add_i32 s1, s0, s16 // (op 1071) 

s_ashr_i32 s0, s1, 31 // (op 1071) 
s_cmp_eq_i32 s10, 0 // (op 1071) 
s_mov_b32 s18, scc // (op 1071) 
s_cmp_eq_i32 s18, 0 // (op 1071) 
s_cselect_b32 s19, 0, -1 // (op 1071) 

s_and_b32 s18, s47, 31 // (op 1071) 
s_lshl_b32 s72, 1, s18 // (op 1071) 
s_add_i32 s73, s72, s19 // (op 1071) 


s_and_b32 s19, s0, s73 // (op 1071) 


s_add_i32 s0, s1, s19 // (op 1071) 


s_ashr_i32 s1, s0, s18 // (op 1071) 


s_lshl_b32 s0, s47, 24 // (op 1071) 
s_ashr_i32 s18, s0, 31 // (op 1071) 

s_xor_b32 s0, s1, s18 // (op 1071) 

s_sub_i32 s1, s0, s18 // (op 1071) 


s_mul_i32 s0, s44, s50 // (op 1071) 

s_mul_i32 s18, s48, s0 // (op 1071) 

s_mul_i32 s0, s1, s18 // (op 1071) 

s_sub_u32 s18, s16, s0 // (op 1071) 

s_mul_hi_i32 s0, s18, s15 // (op 1071) 
s_add_i32 s19, s0, s18 // (op 1071) 


s_ashr_i32 s0, s19, 31 // (op 1071) 
s_cmp_eq_i32 s15, 0 // (op 1071) 
s_mov_b32 s18, scc // (op 1071) 
s_cmp_eq_i32 s18, 0 // (op 1071) 
s_cselect_b32 s50, 0, -1 // (op 1071) 

s_and_b32 s18, s46, 31 // (op 1071) 
s_lshl_b32 s51, 1, s18 // (op 1071) 
s_add_i32 s72, s51, s50 // (op 1071) 


s_and_b32 s50, s0, s72 // (op 1071) 


s_add_i32 s0, s19, s50 // (op 1071) 


s_ashr_i32 s19, s0, s18 // (op 1071) 


s_lshl_b32 s0, s46, 24 // (op 1071) 
s_ashr_i32 s18, s0, 31 // (op 1071) 

s_xor_b32 s0, s19, s18 // (op 1071) 

s_sub_i32 s19, s0, s18 // (op 1071) 


s_cmp_eq_i32 s1, 0 // (op 1071) 
s_mov_b32 s0, scc // (op 1071) 

s_cmp_eq_i32 s0, 0 // (op 1071) 
s_cselect_b32 s1, s19, s49 // (op 1071) 



v_lshl_add_u32 v57, s1, 6, v12 // (op 1071) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1071) 
s_lshr_b32 s18, s0, 1 // (op 1071) 

v_lshl_add_u32 v12, s18, 3, v13 // (op 1071) 


v_mul_lo_u32 v13, v12, s38 // (op 1071) 
v_add_u32 v13, v13, v57 // (op 1071) 


s_mul_i32 s1, s38, s43 // (op 1073) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1073) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1073) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1073) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1073) most significant: sum
s_add_u32 s1, s1, s19 // (op 1073) most significant: sum


s_mov_b32 s68, s40 // (op 1073) reg expression
s_mov_b32 s69, s41 // (op 1073) reg expression
s_mov_b32 s70, s0 // (op 1073) reg expression
s_mov_b32 s71, 131072 // (op 1073) call()

v_and_b32 v12, -8, v1 // (op 1164) 
v_sub_u32 v57, v1, v12 // (op 1164) 

v_lshlrev_b32 v12, 4, v57 // (op 1164) 

v_lshrrev_b32 v57, 3, v1 // (op 1164) 
v_lshl_add_u32 v14, v57, 7, v12 // (op 1164) 


v_mul_lo_u32 v12, v14, 6 // (op 1164) 

v_lshrrev_b32 v57, 3, v12 // (op 1164) 

v_lshlrev_b32 v12, 6, v1 // (op 1217) 
v_mul_lo_u32 v14, v12, 6 // (op 1217) 
v_lshrrev_b32 v15, 3, v14 // (op 1217) 

v_lshrrev_b32 v14, 4, v12 // (op 1217) 

v_lshlrev_b32 v12, 2, v14 // (op 1217) 

v_add_u32 v14, v15, v12 // (op 1217) 


v_and_b32 v12, -64, v1 // (op 1275) 
v_sub_u32 v15, v1, v12 // (op 1275) 

v_lshrrev_b32 v12, 6, v1 // (op 1275) 
v_lshl_add_u32 v16, v12, 7, v15 // (op 1275) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 317) 
s_and_b32 s18, s0, 4294967294 // (op 317) 

s_ashr_i64 s[0:1], s[32:33], 7 // (op 316) 
s_mov_b32 s19, s0 // (op 316) convert

s_cmp_lt_u32 s18, s19 // (op 316) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316 // (op 316) Condition: Top (jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316:


s_add_u32 s18, s18, 1 // (op 318) 
v_mov_b32 v12, v49 // (op 170) 

buffer_load_dwordx3 v[18:20], v12, s[20:23], 0 offen // (op 170) Load value

v_add_u32 v49, 96, v49 // (op 865) 


s_barrier  // (op 1455) 
v_mov_b32 v12, v53 // (op 183) 



v_mov_b32 v15, v12 // (op 183) reg expression

buffer_load_dwordx3 v[22:24], v15, s[52:55], 0 offen // (op 183) Load value
buffer_load_dwordx3 v[26:28], v15, s[52:55], 0 offen offset:12 // (op 183) Load value
s_waitcnt vmcnt(2)
ds_write_b96 v57, v[18:20] offset:8448 // (op 171) Store local data 
buffer_load_dwordx3 v[30:32], v15, s[52:55], 0 offen offset:24 // (op 183) Load value

buffer_load_dwordx3 v[34:36], v15, s[52:55], 0 offen offset:36 // (op 183) Load value


s_lshl_b64 s[0:1], s[28:29], 7 // (op 935) 
s_mul_i32 s51, s1, 6 // (op 935) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 935) least significant
s_mul_hi_u32 s19, s0, 6 // (op 935) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 935) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 935) 

v_add_u32 v53, v53, s0 // (op 935) 


s_barrier  // (op 1468) 
v_mov_b64 v[20:21], s[38:39] // (op 184) convert




v_mov_b32 v12, v13 // (op 184) 
v_mov_b32 v15, v12 // (op 184) reg expression
buffer_load_ubyte v18, v15, s[68:71], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v15, v15, v20 // (op 184) 
s_waitcnt vmcnt(4)
ds_write_b96 v14, v[22:24] // (op 181) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v14, v[26:28] offset:16 // (op 181) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v14, v[30:32] offset:32 // (op 181) Store local data 
buffer_load_ubyte v19, v15, s[68:71], 0 offen offset: 0 // (op 184) Load value



s_lshl_b32 s19, s38, 2 // (op 1063) 
v_add_u32 v13, v13, s19 // (op 1063) 

s_waitcnt vmcnt(2)
ds_write_b96 v14, v[34:36] offset:48 // (op 181) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1458) 



s_barrier  // (op 1482) 
ds_read_b128 v[20:23], v3 offset:8448 // (op 173) Load local data 



ds_read_b64 v[24:25], v3 offset:8464 // (op 173) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v16, v18 offset:8192 // (op 182) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1472) 
s_waitcnt vmcnt(0)
ds_write_b8 v16, v19 offset:8256 // (op 182) Store local data 




ds_read_b96_tr_b6 v[26:28], v0 // (op 175) Transpose load local data 
ds_read_b96_tr_b6 v[29:31], v0 offset:1024 // (op 175) Transpose load local data 


v_mov_b32 v12, v61 // (op 177) 

buffer_load_ubyte v15, v12, s[64:67], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v61, 4, v61 // (op 999) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1486) 


ds_read_u8 v12, v7 offset:8192 // (op 179) Load local data 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v15, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 180) 
ds_read_b96_tr_b6 v[32:34], v0 offset:32 // (op 324) Transpose load local data 


ds_read_b96_tr_b6 v[35:37], v0 offset:1056 // (op 324) Transpose load local data 


ds_read_u8 v12, v7 offset:8224 // (op 327) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v15, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 328) 




s_ashr_i64 s[0:1], s[32:33], 7 // (op 316) 
s_mov_b32 s19, s0 // (op 316) convert

s_cmp_lt_u32 s18, s19 // (op 316) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316 // (op 316) Condition: Bottom (jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316 if true)
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316:






















s_barrier  // (op 1568) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_125
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

v_mov_b32 v0, v9 // (op 54) reg expression

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 54) Store value



v_accvgpr_read v10, a4 // (op 508) 
v_accvgpr_read v11, a5 // (op 508) 
v_accvgpr_read v12, a6 // (op 508) 
v_accvgpr_read v13, a7 // (op 508) 
v_mul_f32 v14, v8, v10 // (op 508) 
v_mul_f32 v15, v8, v11 // (op 508) 
v_mul_f32 v16, v8, v12 // (op 508) 
v_mul_f32 v17, v8, v13 // (op 508) 

v_lshrrev_b32 v10, 16, v14 // (op 510) 
v_lshrrev_b32 v11, 16, v15 // (op 510) 
v_lshrrev_b32 v12, 16, v16 // (op 510) 
v_lshrrev_b32 v13, 16, v17 // (op 510) 


v_pack_b32_f16 v14, v10, v11 // (op 511) 
v_pack_b32_f16 v15, v12, v13 // (op 511) 

s_lshl_b32 s0, s4, 6 // (op 511) 
v_add_u32 v0, v9, s0 // (op 511) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 511) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_125:


v_mov_b32 v0, v6 // (op 138) reg expression

buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 138) Load value

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

v_mov_b32 v0, v9 // (op 137) reg expression

buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 137) Store value





s_lshl_b32 s0, s2, 6 // (op 535) 
v_add_u32 v0, v6, s0 // (op 535) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 535) Load value

v_accvgpr_read v12, a4 // (op 531) 
v_accvgpr_read v13, a5 // (op 531) 
v_accvgpr_read v14, a6 // (op 531) 
v_accvgpr_read v15, a7 // (op 531) 
v_mul_f32 v16, v8, v12 // (op 531) 
v_mul_f32 v17, v8, v13 // (op 531) 
v_mul_f32 v18, v8, v14 // (op 531) 
v_mul_f32 v19, v8, v15 // (op 531) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 536) 
v_lshrrev_b32 v13, 16, v10 // (op 536) 
v_lshlrev_b32 v12, 16, v12 // (op 536) 
v_lshlrev_b32 v13, 16, v13 // (op 536) 
v_mul_f32 v20, v5, v12 // (op 536) 
v_mul_f32 v21, v5, v13 // (op 536) 
v_and_b32 v12, 65535, v11 // (op 536) 
v_lshrrev_b32 v13, 16, v11 // (op 536) 
v_lshlrev_b32 v12, 16, v12 // (op 536) 
v_lshlrev_b32 v13, 16, v13 // (op 536) 
v_mul_f32 v22, v5, v12 // (op 536) 
v_mul_f32 v23, v5, v13 // (op 536) 


v_add_f32 v10, v20, v16 // (op 532) 
v_add_f32 v11, v21, v17 // (op 532) 
v_add_f32 v12, v22, v18 // (op 532) 
v_add_f32 v13, v23, v19 // (op 532) 


v_lshrrev_b32 v14, 16, v10 // (op 533) 
v_lshrrev_b32 v15, 16, v11 // (op 533) 
v_lshrrev_b32 v16, 16, v12 // (op 533) 
v_lshrrev_b32 v17, 16, v13 // (op 533) 


v_pack_b32_f16 v10, v14, v15 // (op 534) 
v_pack_b32_f16 v11, v16, v17 // (op 534) 

s_lshl_b32 s0, s4, 6 // (op 534) 
v_add_u32 v0, v9, s0 // (op 534) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 534) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.LRR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__end:
.size RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_, .LRR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__end-RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 19968 // lds bytes
  .amdhsa_accum_offset 88
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: F6
    DestDataType: b
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
    macrotile: [16, 64, 128]
    threads: [128, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 19968
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 86
    .agpr_count: 8
    .max_flat_workgroup_size: 128
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

