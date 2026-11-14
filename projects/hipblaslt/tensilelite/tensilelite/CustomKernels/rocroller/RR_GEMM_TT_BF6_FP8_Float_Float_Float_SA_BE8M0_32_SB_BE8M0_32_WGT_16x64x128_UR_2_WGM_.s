// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_,@function
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s11 // (op 1151) 
s_mul_hi_i32 s1, s16, s11 // (op 1339) 
v_lshrrev_b32 v0, 7, v1 // (op 1208) 
v_and_b32 v3, 15, v1 // (op 1208) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1208) 


v_and_b32 v0, 63, v1 // (op 1208) 
v_lshrrev_b32 v3, 4, v0 // (op 1208) 

v_lshlrev_b32 v0, 5, v3 // (op 1208) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 1208) 


v_mul_lo_u32 v0, v3, 6 // (op 1208) 

v_lshrrev_b32 v3, 3, v0 // (op 1208) 

v_lshrrev_b32 v0, 7, v1 // (op 1365) 
v_and_b32 v4, 15, v1 // (op 1365) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1365) 


v_and_b32 v0, 63, v1 // (op 1365) 
v_lshrrev_b32 v4, 4, v0 // (op 1365) 

v_lshlrev_b32 v0, 5, v4 // (op 1365) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 1365) 


v_mul_lo_u32 v0, v4, 6 // (op 1365) 

v_lshrrev_b32 v4, 3, v0 // (op 1365) 

v_lshrrev_b32 v0, 7, v1 // (op 1151) 
s_add_i32 s18, s0, s16 // (op 1151) 

v_and_b32 v5, 63, v1 // (op 1151) 
s_ashr_i32 s0, s18, 31 // (op 1151) 
v_lshrrev_b32 v6, 4, v5 // (op 1151) 
s_cmp_eq_i32 s11, 0 // (op 1151) 
s_mov_b32 s19, scc // (op 1151) 
v_lshlrev_b32 v7, 2, v6 // (op 1151) 

s_cmp_eq_i32 s19, 0 // (op 1151) 
s_cselect_b32 s51, 0, -1 // (op 1151) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1151) 


s_and_b32 s19, s12, 31 // (op 1151) 
v_lshrrev_b32 v0, 2, v5 // (op 1151) 

v_and_b32 v5, 3, v0 // (op 1151) 

v_and_b32 v0, 48, v1 // (op 1265) 
v_and_b32 v7, 63, v1 // (op 1265) 
v_sub_u32 v8, v7, v0 // (op 1265) 

v_lshrrev_b32 v7, 1, v8 // (op 1265) 
v_add_lshl_u32 v9, v0, v7, 6 // (op 1265) 


v_and_b32 v0, 1, v8 // (op 1265) 

v_lshlrev_b32 v7, 3, v0 // (op 1265) 

v_lshrrev_b32 v0, 6, v1 // (op 1265) 
v_and_b32 v8, 1, v0 // (op 1265) 

v_lshl_add_u32 v0, v8, 4, v7 // (op 1265) 


v_add_u32 v7, v9, v0 // (op 1265) 


v_lshrrev_b32 v0, 6, v1 // (op 1323) 
v_and_b32 v8, 1, v0 // (op 1323) 

v_and_b32 v0, 15, v1 // (op 1323) 
v_lshl_add_u32 v9, v8, 4, v0 // (op 1323) 


v_and_b32 v0, 63, v1 // (op 1323) 
v_lshrrev_b32 v8, 4, v0 // (op 1323) 

v_lshl_add_u32 v0, v8, 6, v9 // (op 1323) 


v_lshrrev_b32 v8, 7, v1 // (op 1339) 
v_and_b32 v9, 63, v1 // (op 1339) 
v_and_b32 v10, 48, v1 // (op 1409) 
v_and_b32 v11, 63, v1 // (op 1409) 
v_sub_u32 v12, v11, v10 // (op 1409) 

v_lshrrev_b32 v11, 1, v12 // (op 1409) 
v_add_lshl_u32 v13, v10, v11, 6 // (op 1409) 


v_and_b32 v10, 1, v12 // (op 1409) 

v_lshlrev_b32 v11, 3, v10 // (op 1409) 

v_lshrrev_b32 v10, 6, v1 // (op 1409) 
v_and_b32 v12, 1, v10 // (op 1409) 

v_lshl_add_u32 v10, v12, 4, v11 // (op 1409) 


v_add_u32 v11, v13, v10 // (op 1409) 


v_lshrrev_b32 v10, 6, v1 // (op 1438) 
v_and_b32 v12, 1, v10 // (op 1438) 

v_and_b32 v10, 15, v1 // (op 1438) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 1438) 


v_and_b32 v10, 63, v1 // (op 1438) 
v_lshrrev_b32 v12, 4, v10 // (op 1438) 

v_lshl_add_u32 v10, v12, 6, v13 // (op 1438) 


s_lshl_b32 s52, 1, s19 // (op 1151) 
v_and_b32 v12, 3, v1 // (op 1151) 
s_add_i32 s53, s52, s51 // (op 1151) 


s_and_b32 s51, s0, s53 // (op 1151) 


s_add_i32 s0, s1, s16 // (op 1339) 

s_ashr_i32 s1, s0, 31 // (op 1339) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1151) 


s_add_i32 s52, s18, s51 // (op 1151) 


v_lshrrev_b32 v5, 6, v1 // (op 1151) 
s_ashr_i32 s18, s52, s19 // (op 1151) 


v_and_b32 v12, 1, v5 // (op 1151) 

s_lshl_b32 s19, s12, 24 // (op 1151) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1151) 


s_ashr_i32 s51, s19, 31 // (op 1151) 

s_xor_b32 s19, s18, s51 // (op 1151) 

s_sub_i32 s18, s19, s51 // (op 1151) 


s_mul_hi_i32 s19, s18, s13 // (op 1151) 
s_add_i32 s51, s19, s18 // (op 1151) 

s_ashr_i32 s19, s51, 31 // (op 1151) 
s_cmp_eq_i32 s13, 0 // (op 1151) 
s_mov_b32 s52, scc // (op 1151) 
s_cmp_eq_i32 s52, 0 // (op 1151) 
s_cselect_b32 s53, 0, -1 // (op 1151) 

s_and_b32 s52, s14, 31 // (op 1151) 
v_lshrrev_b32 v12, 4, v9 // (op 1339) 
s_cmp_eq_i32 s11, 0 // (op 1339) 
s_mov_b32 s54, scc // (op 1339) 
v_lshlrev_b32 v13, 2, v12 // (op 1339) 

s_cmp_eq_i32 s54, 0 // (op 1339) 
s_cselect_b32 s55, 0, -1 // (op 1339) 

s_lshl_b32 s54, 1, s52 // (op 1151) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 1339) 


v_lshrrev_b32 v8, 2, v9 // (op 1339) 

s_add_i32 s56, s54, s53 // (op 1151) 


s_and_b32 s53, s19, s56 // (op 1151) 


s_add_i32 s19, s51, s53 // (op 1151) 


s_ashr_i32 s51, s19, s52 // (op 1151) 


s_lshl_b32 s19, s14, 24 // (op 1151) 
s_ashr_i32 s52, s19, 31 // (op 1151) 

s_xor_b32 s19, s51, s52 // (op 1151) 

s_sub_i32 s51, s19, s52 // (op 1151) 


s_and_b32 s19, s12, 31 // (op 1339) 
s_lshl_b32 s52, 1, s19 // (op 1339) 
v_and_b32 v9, 3, v8 // (op 1339) 

s_add_i32 s53, s52, s55 // (op 1339) 


v_and_b32 v8, 3, v1 // (op 1339) 
s_and_b32 s52, s1, s53 // (op 1339) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1339) 


s_add_i32 s1, s0, s52 // (op 1339) 


v_lshrrev_b32 v8, 6, v1 // (op 1339) 
s_ashr_i32 s0, s1, s19 // (op 1339) 


v_and_b32 v9, 1, v8 // (op 1339) 

s_lshl_b32 s1, s12, 24 // (op 1339) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1339) 


s_ashr_i32 s19, s1, 31 // (op 1339) 

s_xor_b32 s1, s0, s19 // (op 1339) 

s_sub_i32 s0, s1, s19 // (op 1339) 


s_mul_hi_i32 s1, s0, s13 // (op 1339) 
s_add_i32 s19, s1, s0 // (op 1339) 

s_ashr_i32 s1, s19, 31 // (op 1339) 
s_cmp_eq_i32 s13, 0 // (op 1339) 
s_mov_b32 s52, scc // (op 1339) 
s_cmp_eq_i32 s52, 0 // (op 1339) 
s_cselect_b32 s53, 0, -1 // (op 1339) 

s_and_b32 s52, s14, 31 // (op 1339) 
s_lshl_b32 s54, 1, s52 // (op 1339) 
s_add_i32 s55, s54, s53 // (op 1339) 


s_and_b32 s53, s1, s55 // (op 1339) 


s_add_i32 s1, s19, s53 // (op 1339) 


s_ashr_i32 s19, s1, s52 // (op 1339) 


s_lshl_b32 s1, s14, 24 // (op 1339) 
s_ashr_i32 s52, s1, 31 // (op 1339) 

s_xor_b32 s1, s19, s52 // (op 1339) 

s_sub_i32 s19, s1, s52 // (op 1339) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 1151) 
s_sub_u32 s52, s16, s1 // (op 1151) 

s_mul_i32 s1, s51, s44 // (op 1151) 
s_add_u32 s1, s1, s52 // (op 1151) 

s_mul_hi_i32 s52, s16, s10 // (op 1151) 
s_add_i32 s53, s52, s16 // (op 1151) 

s_ashr_i32 s52, s53, 31 // (op 1151) 
s_cmp_eq_i32 s10, 0 // (op 1151) 
s_mov_b32 s54, scc // (op 1151) 
s_cmp_eq_i32 s54, 0 // (op 1151) 
s_cselect_b32 s55, 0, -1 // (op 1151) 

s_and_b32 s54, s47, 31 // (op 1151) 
s_lshl_b32 s56, 1, s54 // (op 1151) 
s_add_i32 s57, s56, s55 // (op 1151) 


s_and_b32 s55, s52, s57 // (op 1151) 


s_add_i32 s52, s53, s55 // (op 1151) 


s_ashr_i32 s53, s52, s54 // (op 1151) 


s_lshl_b32 s52, s47, 24 // (op 1151) 
s_ashr_i32 s54, s52, 31 // (op 1151) 

s_xor_b32 s52, s53, s54 // (op 1151) 

s_sub_i32 s53, s52, s54 // (op 1151) 


s_mul_i32 s52, s0, s44 // (op 1339) 
s_sub_u32 s54, s16, s52 // (op 1339) 

s_mul_i32 s52, s19, s44 // (op 1339) 
s_add_u32 s52, s52, s54 // (op 1339) 

s_mul_hi_i32 s54, s16, s10 // (op 1339) 
s_add_i32 s55, s54, s16 // (op 1339) 

s_ashr_i32 s54, s55, 31 // (op 1339) 
s_cmp_eq_i32 s10, 0 // (op 1339) 
s_mov_b32 s56, scc // (op 1339) 
s_cmp_eq_i32 s56, 0 // (op 1339) 
s_cselect_b32 s57, 0, -1 // (op 1339) 

s_and_b32 s56, s47, 31 // (op 1339) 
s_lshl_b32 s58, 1, s56 // (op 1339) 
s_add_i32 s59, s58, s57 // (op 1339) 


s_and_b32 s57, s54, s59 // (op 1339) 


s_add_i32 s54, s55, s57 // (op 1339) 


s_ashr_i32 s55, s54, s56 // (op 1339) 


s_lshl_b32 s54, s47, 24 // (op 1339) 
s_ashr_i32 s56, s54, 31 // (op 1339) 

s_xor_b32 s54, s55, s56 // (op 1339) 

s_sub_i32 s55, s54, s56 // (op 1339) 


s_add_u32 s56, s6, 64 // (op 1151) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1151) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1151) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1151) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1151) 

s_mul_i32 s54, s44, s56 // (op 1151) 
s_mul_i32 s58, s48, s54 // (op 1151) 

s_mul_i32 s54, s53, s58 // (op 1151) 

s_sub_u32 s58, s16, s54 // (op 1151) 

s_mov_b32 s54, s58 // (op 1151) 
s_mul_hi_i32 s59, s54, s15 // (op 1151) 
s_add_i32 s60, s59, s54 // (op 1151) 


s_ashr_i32 s54, s60, 31 // (op 1151) 
s_cmp_eq_i32 s15, 0 // (op 1151) 
s_mov_b32 s59, scc // (op 1151) 
s_cmp_eq_i32 s59, 0 // (op 1151) 
s_cselect_b32 s61, 0, -1 // (op 1151) 

s_and_b32 s59, s46, 31 // (op 1151) 
s_lshl_b32 s62, 1, s59 // (op 1151) 
s_add_i32 s63, s62, s61 // (op 1151) 


s_and_b32 s61, s54, s63 // (op 1151) 


s_add_i32 s54, s60, s61 // (op 1151) 


s_ashr_i32 s60, s54, s59 // (op 1151) 


s_lshl_b32 s54, s46, 24 // (op 1151) 
s_ashr_i32 s59, s54, 31 // (op 1151) 

s_xor_b32 s54, s60, s59 // (op 1151) 

s_sub_i32 s60, s54, s59 // (op 1151) 


s_mul_i32 s54, s45, s44 // (op 1151) 
s_add_u32 s62, s8, 16 // (op 1151) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 1151) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1151) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1151) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 1151) 

s_sub_i32 s59, s62, s54 // (op 1151) 

s_mul_i32 s61, s60, s59 // (op 1151) 

s_sub_u32 s59, s58, s61 // (op 1151) 


s_add_u32 s58, s59, s54 // (op 1151) 


s_cmp_eq_i32 s53, 0 // (op 1151) 
s_mov_b32 s54, scc // (op 1151) 

s_cmp_eq_i32 s54, 0 // (op 1151) 
s_cselect_b32 s53, s58, s1 // (op 1151) 


s_mul_i32 s1, s51, s56 // (op 1151) 


v_lshl_add_u32 v9, s53, 4, v6 // (op 1151) 


s_sub_i32 s51, s18, s1 // (op 1151) 


s_cmp_eq_i32 s54, 0 // (op 1151) 
s_cselect_b32 s1, s60, s51 // (op 1151) 



v_lshl_add_u32 v6, s1, 6, v5 // (op 1151) 


v_mul_lo_u32 v5, v6, s2 // (op 1151) 

v_add_lshl_u32 v6, v9, v5, 2 // (op 1151) 


s_add_u32 s56, s6, 64 // (op 1339) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1339) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1339) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1339) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1339) 

s_mul_i32 s1, s44, s56 // (op 1339) 
s_mul_i32 s18, s48, s1 // (op 1339) 

s_mul_i32 s1, s55, s18 // (op 1339) 

s_sub_u32 s18, s16, s1 // (op 1339) 

s_mov_b32 s1, s18 // (op 1339) 
s_mul_hi_i32 s51, s1, s15 // (op 1339) 
s_add_i32 s53, s51, s1 // (op 1339) 


s_ashr_i32 s1, s53, 31 // (op 1339) 
s_cmp_eq_i32 s15, 0 // (op 1339) 
s_mov_b32 s51, scc // (op 1339) 
s_cmp_eq_i32 s51, 0 // (op 1339) 
s_cselect_b32 s54, 0, -1 // (op 1339) 

s_and_b32 s51, s46, 31 // (op 1339) 
s_lshl_b32 s58, 1, s51 // (op 1339) 
s_add_i32 s59, s58, s54 // (op 1339) 


s_and_b32 s54, s1, s59 // (op 1339) 


s_add_i32 s1, s53, s54 // (op 1339) 


s_ashr_i32 s53, s1, s51 // (op 1339) 


s_lshl_b32 s1, s46, 24 // (op 1339) 
s_ashr_i32 s51, s1, 31 // (op 1339) 

s_xor_b32 s1, s53, s51 // (op 1339) 

s_sub_i32 s53, s1, s51 // (op 1339) 


s_mul_i32 s1, s45, s44 // (op 1339) 
s_add_u32 s58, s8, 16 // (op 1339) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 1339) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1339) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1339) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1339) 

s_sub_i32 s51, s58, s1 // (op 1339) 

s_mul_i32 s54, s53, s51 // (op 1339) 

s_sub_u32 s51, s18, s54 // (op 1339) 


s_add_u32 s18, s51, s1 // (op 1339) 


s_cmp_eq_i32 s55, 0 // (op 1339) 
s_mov_b32 s1, scc // (op 1339) 

s_cmp_eq_i32 s1, 0 // (op 1339) 
s_cselect_b32 s51, s18, s52 // (op 1339) 


s_mul_i32 s18, s19, s56 // (op 1339) 


v_lshl_add_u32 v5, s51, 4, v12 // (op 1339) 


s_sub_i32 s19, s0, s18 // (op 1339) 


s_cmp_eq_i32 s1, 0 // (op 1339) 
s_cselect_b32 s0, s53, s19 // (op 1339) 



v_lshl_add_u32 v9, s0, 6, v8 // (op 1339) 


v_mul_lo_u32 v8, v9, s4 // (op 1339) 

v_add_lshl_u32 v9, v5, v8, 2 // (op 1339) 


s_mul_i32 s1, s2, s7 // (op 1153) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1153) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1153) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1153) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1153) most significant: sum
s_add_u32 s1, s1, s19 // (op 1153) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1153) 

s_mul_i32 s1, s4, s7 // (op 1341) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 1341) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 1341) least significant
s_mul_hi_u32 s52, s4, s6 // (op 1341) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1341) most significant: sum
s_add_u32 s1, s1, s52 // (op 1341) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 1341) 

s_mov_b32 s56, s20 // (op 1153) reg expression
s_mov_b32 s57, s21 // (op 1153) reg expression
s_mov_b32 s58, s18 // (op 1153) reg expression
s_mov_b32 s59, 131072 // (op 1153) call()

s_mov_b32 s60, s22 // (op 1341) reg expression

s_mov_b32 s61, s23 // (op 1341) reg expression
s_mov_b32 s62, s52 // (op 1341) reg expression
s_mov_b32 s63, 131072 // (op 1341) call()


v_mov_b32 v5, s49 // (op 2) Move value
v_mov_b32 v8, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 306) call()
v_accvgpr_write a5, 0.00000 // (op 306) call()
v_accvgpr_write a6, 0.00000 // (op 306) call()
v_accvgpr_write a7, 0.00000 // (op 306) call()
s_mul_hi_i32 s0, s16, s11 // (op 918) 
v_lshrrev_b32 v12, 3, v1 // (op 918) 
s_add_i32 s1, s0, s16 // (op 918) 

v_and_b32 v13, -8, v1 // (op 918) 
s_ashr_i32 s0, s1, 31 // (op 918) 
v_sub_u32 v14, v1, v13 // (op 918) 

s_cmp_eq_i32 s11, 0 // (op 918) 
s_mov_b32 s18, scc // (op 918) 
v_lshlrev_b32 v13, 4, v14 // (op 918) 

s_cmp_eq_i32 s18, 0 // (op 918) 
s_cselect_b32 s19, 0, -1 // (op 918) 

v_mov_b32 v15, 0 // (op 918) convert
v_mov_b32 v14, v13 // (op 918) convert

s_and_b32 s18, s12, 31 // (op 918) 
s_lshl_b32 s20, 1, s18 // (op 918) 
s_add_i32 s21, s20, s19 // (op 918) 


s_and_b32 s19, s0, s21 // (op 918) 


s_add_i32 s0, s1, s19 // (op 918) 


s_ashr_i32 s1, s0, s18 // (op 918) 


s_lshl_b32 s0, s12, 24 // (op 918) 
s_ashr_i32 s18, s0, 31 // (op 918) 

s_xor_b32 s0, s1, s18 // (op 918) 

s_sub_i32 s1, s0, s18 // (op 918) 


s_mul_hi_i32 s0, s1, s13 // (op 918) 
s_add_i32 s18, s0, s1 // (op 918) 

s_ashr_i32 s0, s18, 31 // (op 918) 
s_cmp_eq_i32 s13, 0 // (op 918) 
s_mov_b32 s19, scc // (op 918) 
s_cmp_eq_i32 s19, 0 // (op 918) 
s_cselect_b32 s20, 0, -1 // (op 918) 

s_and_b32 s19, s14, 31 // (op 918) 
s_lshl_b32 s21, 1, s19 // (op 918) 
s_add_i32 s22, s21, s20 // (op 918) 


s_and_b32 s23, s0, s22 // (op 918) 


s_add_i32 s0, s18, s23 // (op 918) 


s_ashr_i32 s18, s0, s19 // (op 918) 


s_lshl_b32 s0, s14, 24 // (op 918) 
s_ashr_i32 s19, s0, 31 // (op 918) 

s_xor_b32 s0, s18, s19 // (op 918) 

s_sub_i32 s18, s0, s19 // (op 918) 


s_mul_i32 s0, s1, s44 // (op 918) 

s_sub_u32 s1, s16, s0 // (op 918) 

s_mul_i32 s0, s18, s44 // (op 918) 
s_add_u32 s0, s0, s1 // (op 918) 


s_mul_hi_i32 s1, s16, s10 // (op 918) 
s_add_i32 s18, s1, s16 // (op 918) 

s_ashr_i32 s1, s18, 31 // (op 918) 
s_cmp_eq_i32 s10, 0 // (op 918) 
s_mov_b32 s19, scc // (op 918) 
s_cmp_eq_i32 s19, 0 // (op 918) 
s_cselect_b32 s20, 0, -1 // (op 918) 

s_and_b32 s19, s47, 31 // (op 918) 
s_lshl_b32 s21, 1, s19 // (op 918) 
s_add_i32 s22, s21, s20 // (op 918) 


s_and_b32 s23, s1, s22 // (op 918) 


s_add_i32 s1, s18, s23 // (op 918) 


s_ashr_i32 s18, s1, s19 // (op 918) 


s_lshl_b32 s1, s47, 24 // (op 918) 
s_ashr_i32 s19, s1, 31 // (op 918) 

s_xor_b32 s1, s18, s19 // (op 918) 

s_sub_i32 s18, s1, s19 // (op 918) 


s_add_u32 s22, s6, 64 // (op 918) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 918) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 918) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 918) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 6 // (op 918) 

s_mul_i32 s1, s44, s22 // (op 918) 

s_mul_i32 s19, s48, s1 // (op 918) 

s_mul_i32 s1, s18, s19 // (op 918) 

s_sub_u32 s19, s16, s1 // (op 918) 

s_mov_b32 s1, s19 // (op 918) 
s_mul_hi_i32 s20, s1, s15 // (op 918) 
s_add_i32 s21, s20, s1 // (op 918) 


s_ashr_i32 s1, s21, 31 // (op 918) 
s_cmp_eq_i32 s15, 0 // (op 918) 
s_mov_b32 s20, scc // (op 918) 
s_cmp_eq_i32 s20, 0 // (op 918) 
s_cselect_b32 s22, 0, -1 // (op 918) 

s_and_b32 s20, s46, 31 // (op 918) 
s_lshl_b32 s23, 1, s20 // (op 918) 
s_add_i32 s49, s23, s22 // (op 918) 


s_and_b32 s22, s1, s49 // (op 918) 


s_add_i32 s1, s21, s22 // (op 918) 


s_ashr_i32 s21, s1, s20 // (op 918) 


s_lshl_b32 s1, s46, 24 // (op 918) 
s_ashr_i32 s20, s1, 31 // (op 918) 

s_xor_b32 s1, s21, s20 // (op 918) 

s_sub_i32 s21, s1, s20 // (op 918) 


s_mul_i32 s1, s45, s44 // (op 918) 
s_add_u32 s22, s8, 16 // (op 918) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 918) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 918) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 918) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 4 // (op 918) 

s_sub_i32 s20, s22, s1 // (op 918) 

s_mul_i32 s22, s21, s20 // (op 918) 


s_sub_u32 s23, s19, s22 // (op 918) 


s_add_u32 s19, s23, s1 // (op 918) 


s_cmp_eq_i32 s18, 0 // (op 918) 
s_mov_b32 s1, scc // (op 918) 

s_cmp_eq_i32 s1, 0 // (op 918) 
s_cselect_b32 s18, s19, s0 // (op 918) 



v_lshl_add_u32 v13, s18, 4, v12 // (op 918) 


v_mul_lo_u32 v17, v13, s25 // (op 918) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 918) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 918) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 918) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 918) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 918) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 918) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 918) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 918) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 918) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 918) 

v_mov_b32 v12, v14 // (op 918) convert

s_mul_i32 s1, s9, 6 // (op 920) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 920) least significant
s_mul_hi_u32 s18, s8, 6 // (op 920) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 920) most significant: sum

s_mul_i32 s19, s24, s1 // (op 920) most significant: low of low * high
s_mul_i32 s20, s25, s0 // (op 920) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 920) least significant
s_mul_hi_u32 s21, s24, s0 // (op 920) most significant: high of low * low
s_add_u32 s19, s19, s20 // (op 920) most significant: sum
s_add_u32 s19, s19, s21 // (op 920) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 920) 

s_mov_b32 s20, s26 // (op 920) reg expression
s_mov_b32 s21, s27 // (op 920) reg expression
s_mov_b32 s22, s0 // (op 920) reg expression
s_mov_b32 s23, 131072 // (op 920) call()

s_mul_hi_i32 s0, s16, s11 // (op 937) 
v_lshrrev_b32 v13, 3, v1 // (op 937) 
s_add_i32 s1, s0, s16 // (op 937) 

v_and_b32 v14, -8, v1 // (op 937) 
s_ashr_i32 s0, s1, 31 // (op 937) 
v_sub_u32 v15, v1, v14 // (op 937) 

s_cmp_eq_i32 s11, 0 // (op 937) 
s_mov_b32 s18, scc // (op 937) 
v_lshlrev_b32 v14, 4, v15 // (op 937) 

s_cmp_eq_i32 s18, 0 // (op 937) 
s_cselect_b32 s19, 0, -1 // (op 937) 

v_mov_b32 v17, 0 // (op 937) convert
v_mov_b32 v16, v14 // (op 937) convert

s_and_b32 s18, s12, 31 // (op 937) 
s_lshl_b32 s49, 1, s18 // (op 937) 
s_add_i32 s50, s49, s19 // (op 937) 


s_and_b32 s19, s0, s50 // (op 937) 


s_add_i32 s0, s1, s19 // (op 937) 


s_ashr_i32 s1, s0, s18 // (op 937) 


s_lshl_b32 s0, s12, 24 // (op 937) 
s_ashr_i32 s18, s0, 31 // (op 937) 

s_xor_b32 s0, s1, s18 // (op 937) 

s_sub_i32 s1, s0, s18 // (op 937) 


s_mul_hi_i32 s0, s1, s13 // (op 937) 
s_add_i32 s18, s0, s1 // (op 937) 

s_ashr_i32 s0, s18, 31 // (op 937) 
s_cmp_eq_i32 s13, 0 // (op 937) 
s_mov_b32 s19, scc // (op 937) 
s_cmp_eq_i32 s19, 0 // (op 937) 
s_cselect_b32 s49, 0, -1 // (op 937) 

s_and_b32 s19, s14, 31 // (op 937) 
s_lshl_b32 s50, 1, s19 // (op 937) 
s_add_i32 s51, s50, s49 // (op 937) 


s_and_b32 s49, s0, s51 // (op 937) 


s_add_i32 s0, s18, s49 // (op 937) 


s_ashr_i32 s18, s0, s19 // (op 937) 


s_lshl_b32 s0, s14, 24 // (op 937) 
s_ashr_i32 s19, s0, 31 // (op 937) 

s_xor_b32 s0, s18, s19 // (op 937) 

s_sub_i32 s18, s0, s19 // (op 937) 


s_mul_i32 s0, s1, s44 // (op 937) 

s_sub_u32 s1, s16, s0 // (op 937) 

s_mul_i32 s0, s18, s44 // (op 937) 
s_add_u32 s0, s0, s1 // (op 937) 


s_mul_hi_i32 s1, s16, s10 // (op 937) 
s_add_i32 s18, s1, s16 // (op 937) 

s_ashr_i32 s1, s18, 31 // (op 937) 
s_cmp_eq_i32 s10, 0 // (op 937) 
s_mov_b32 s19, scc // (op 937) 
s_cmp_eq_i32 s19, 0 // (op 937) 
s_cselect_b32 s49, 0, -1 // (op 937) 

s_and_b32 s19, s47, 31 // (op 937) 
s_lshl_b32 s50, 1, s19 // (op 937) 
s_add_i32 s51, s50, s49 // (op 937) 


s_and_b32 s49, s1, s51 // (op 937) 


s_add_i32 s1, s18, s49 // (op 937) 


s_ashr_i32 s18, s1, s19 // (op 937) 


s_lshl_b32 s1, s47, 24 // (op 937) 
s_ashr_i32 s19, s1, 31 // (op 937) 

s_xor_b32 s1, s18, s19 // (op 937) 

s_sub_i32 s18, s1, s19 // (op 937) 


s_add_u32 s54, s6, 64 // (op 937) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 937) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 937) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 937) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 937) 

s_mul_i32 s1, s44, s50 // (op 937) 

s_mul_i32 s19, s48, s1 // (op 937) 

s_mul_i32 s1, s18, s19 // (op 937) 

s_sub_u32 s19, s16, s1 // (op 937) 

s_mov_b32 s1, s19 // (op 937) 
s_mul_hi_i32 s49, s1, s15 // (op 937) 
s_add_i32 s50, s49, s1 // (op 937) 


s_ashr_i32 s1, s50, 31 // (op 937) 
s_cmp_eq_i32 s15, 0 // (op 937) 
s_mov_b32 s49, scc // (op 937) 
s_cmp_eq_i32 s49, 0 // (op 937) 
s_cselect_b32 s51, 0, -1 // (op 937) 

s_and_b32 s49, s46, 31 // (op 937) 
s_lshl_b32 s52, 1, s49 // (op 937) 
s_add_i32 s53, s52, s51 // (op 937) 


s_and_b32 s51, s1, s53 // (op 937) 


s_add_i32 s1, s50, s51 // (op 937) 


s_ashr_i32 s50, s1, s49 // (op 937) 


s_lshl_b32 s1, s46, 24 // (op 937) 
s_ashr_i32 s49, s1, 31 // (op 937) 

s_xor_b32 s1, s50, s49 // (op 937) 

s_sub_i32 s50, s1, s49 // (op 937) 


s_mul_i32 s1, s45, s44 // (op 937) 
s_add_u32 s54, s8, 16 // (op 937) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 937) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 937) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 937) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 4 // (op 937) 

s_sub_i32 s49, s54, s1 // (op 937) 

s_mul_i32 s51, s50, s49 // (op 937) 


s_sub_u32 s49, s19, s51 // (op 937) 


s_add_u32 s19, s49, s1 // (op 937) 


s_cmp_eq_i32 s18, 0 // (op 937) 
s_mov_b32 s1, scc // (op 937) 

s_cmp_eq_i32 s1, 0 // (op 937) 
s_cselect_b32 s18, s19, s0 // (op 937) 



v_lshl_add_u32 v14, s18, 4, v13 // (op 937) 


v_mul_lo_u32 v19, v14, s25 // (op 937) most significant: low of low * high
v_mul_hi_u32 v13, v14, s24 // (op 937) most significant: high of low * low
v_mul_lo_u32 v18, v14, s24 // (op 937) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 937) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 937) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 937) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 937) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 937) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 937) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 937) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 937) 

v_mov_b32 v13, v14 // (op 937) convert

s_mul_i32 s1, s9, 6 // (op 939) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 939) least significant
s_mul_hi_u32 s18, s8, 6 // (op 939) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 939) most significant: sum

s_mul_i32 s19, s24, s1 // (op 939) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 939) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 939) least significant
s_mul_hi_u32 s50, s24, s0 // (op 939) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 939) most significant: sum
s_add_u32 s19, s19, s50 // (op 939) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 939) 

s_mov_b32 s20, s26 // (op 939) reg expression
s_mov_b32 s21, s27 // (op 939) reg expression
s_mov_b32 s22, s0 // (op 939) reg expression
s_mov_b32 s23, 131072 // (op 939) call()

s_mul_hi_i32 s0, s16, s11 // (op 982) 
v_and_b32 v14, -4, v1 // (op 982) 
s_add_i32 s1, s0, s16 // (op 982) 

v_sub_u32 v15, v1, v14 // (op 982) 

s_ashr_i32 s0, s1, 31 // (op 982) 
v_lshlrev_b32 v14, 4, v15 // (op 982) 

s_cmp_eq_i32 s11, 0 // (op 982) 
s_mov_b32 s18, scc // (op 982) 
v_lshrrev_b32 v15, 2, v1 // (op 982) 
s_cmp_eq_i32 s18, 0 // (op 982) 
s_cselect_b32 s19, 0, -1 // (op 982) 

s_and_b32 s18, s12, 31 // (op 982) 
s_lshl_b32 s49, 1, s18 // (op 982) 
s_add_i32 s50, s49, s19 // (op 982) 


s_and_b32 s19, s0, s50 // (op 982) 


s_add_i32 s0, s1, s19 // (op 982) 


s_ashr_i32 s1, s0, s18 // (op 982) 


s_lshl_b32 s0, s12, 24 // (op 982) 
s_ashr_i32 s18, s0, 31 // (op 982) 

s_xor_b32 s0, s1, s18 // (op 982) 

s_sub_i32 s1, s0, s18 // (op 982) 


s_mul_hi_i32 s0, s1, s13 // (op 982) 
s_add_i32 s18, s0, s1 // (op 982) 

s_ashr_i32 s0, s18, 31 // (op 982) 
s_cmp_eq_i32 s13, 0 // (op 982) 
s_mov_b32 s19, scc // (op 982) 
s_cmp_eq_i32 s19, 0 // (op 982) 
s_cselect_b32 s49, 0, -1 // (op 982) 

s_and_b32 s19, s14, 31 // (op 982) 
s_lshl_b32 s50, 1, s19 // (op 982) 
s_add_i32 s51, s50, s49 // (op 982) 


s_and_b32 s49, s0, s51 // (op 982) 


s_add_i32 s0, s18, s49 // (op 982) 


s_ashr_i32 s18, s0, s19 // (op 982) 


s_lshl_b32 s0, s14, 24 // (op 982) 
s_ashr_i32 s19, s0, 31 // (op 982) 

s_xor_b32 s0, s18, s19 // (op 982) 

s_sub_i32 s18, s0, s19 // (op 982) 


s_add_u32 s54, s6, 64 // (op 982) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 982) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 982) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 982) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 982) 

s_mul_i32 s0, s18, s50 // (op 982) 

s_sub_i32 s49, s1, s0 // (op 982) 


s_mul_hi_i32 s0, s16, s10 // (op 982) 
s_add_i32 s1, s0, s16 // (op 982) 

s_ashr_i32 s0, s1, 31 // (op 982) 
s_cmp_eq_i32 s10, 0 // (op 982) 
s_mov_b32 s18, scc // (op 982) 
s_cmp_eq_i32 s18, 0 // (op 982) 
s_cselect_b32 s19, 0, -1 // (op 982) 

s_and_b32 s18, s47, 31 // (op 982) 
s_lshl_b32 s52, 1, s18 // (op 982) 
s_add_i32 s53, s52, s19 // (op 982) 


s_and_b32 s19, s0, s53 // (op 982) 


s_add_i32 s0, s1, s19 // (op 982) 


s_ashr_i32 s1, s0, s18 // (op 982) 


s_lshl_b32 s0, s47, 24 // (op 982) 
s_ashr_i32 s18, s0, 31 // (op 982) 

s_xor_b32 s0, s1, s18 // (op 982) 

s_sub_i32 s1, s0, s18 // (op 982) 


s_mul_i32 s0, s44, s50 // (op 982) 

s_mul_i32 s18, s48, s0 // (op 982) 

s_mul_i32 s0, s1, s18 // (op 982) 

s_sub_u32 s18, s16, s0 // (op 982) 

s_mul_hi_i32 s0, s18, s15 // (op 982) 
s_add_i32 s19, s0, s18 // (op 982) 


s_ashr_i32 s0, s19, 31 // (op 982) 
s_cmp_eq_i32 s15, 0 // (op 982) 
s_mov_b32 s18, scc // (op 982) 
s_cmp_eq_i32 s18, 0 // (op 982) 
s_cselect_b32 s50, 0, -1 // (op 982) 

s_and_b32 s18, s46, 31 // (op 982) 
s_lshl_b32 s51, 1, s18 // (op 982) 
s_add_i32 s52, s51, s50 // (op 982) 


s_and_b32 s50, s0, s52 // (op 982) 


s_add_i32 s0, s19, s50 // (op 982) 


s_ashr_i32 s19, s0, s18 // (op 982) 


s_lshl_b32 s0, s46, 24 // (op 982) 
s_ashr_i32 s18, s0, 31 // (op 982) 

s_xor_b32 s0, s19, s18 // (op 982) 

s_sub_i32 s19, s0, s18 // (op 982) 


s_cmp_eq_i32 s1, 0 // (op 982) 
s_mov_b32 s0, scc // (op 982) 

s_cmp_eq_i32 s0, 0 // (op 982) 
s_cselect_b32 s1, s19, s49 // (op 982) 



v_lshl_add_u32 v16, s1, 6, v14 // (op 982) 


v_mov_b32 v19, 0 // (op 982) convert
v_mov_b32 v18, v16 // (op 982) convert

v_mul_lo_u32 v17, v15, s29 // (op 982) most significant: low of low * high
v_mul_hi_u32 v14, v15, s28 // (op 982) most significant: high of low * low
v_mul_lo_u32 v16, v15, s28 // (op 982) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 982) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v18 // (op 982) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 982) most significant half



s_mul_i32 s1, s28, s33 // (op 984) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 984) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 984) least significant
s_mul_hi_u32 s19, s28, s32 // (op 984) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 984) most significant: sum
s_add_u32 s1, s1, s19 // (op 984) most significant: sum


s_mov_b32 s52, s30 // (op 984) reg expression
s_mov_b32 s53, s31 // (op 984) reg expression
s_mov_b32 s54, s0 // (op 984) reg expression
s_mov_b32 s55, 131072 // (op 984) call()

s_mul_hi_i32 s0, s16, s11 // (op 1001) 
v_and_b32 v14, -4, v1 // (op 1001) 
s_add_i32 s1, s0, s16 // (op 1001) 

v_sub_u32 v15, v1, v14 // (op 1001) 

s_ashr_i32 s0, s1, 31 // (op 1001) 
v_lshlrev_b32 v14, 4, v15 // (op 1001) 

s_cmp_eq_i32 s11, 0 // (op 1001) 
s_mov_b32 s18, scc // (op 1001) 
v_lshrrev_b32 v15, 2, v1 // (op 1001) 
s_cmp_eq_i32 s18, 0 // (op 1001) 
s_cselect_b32 s19, 0, -1 // (op 1001) 

s_and_b32 s18, s12, 31 // (op 1001) 
s_lshl_b32 s49, 1, s18 // (op 1001) 
s_add_i32 s50, s49, s19 // (op 1001) 


s_and_b32 s19, s0, s50 // (op 1001) 


s_add_i32 s0, s1, s19 // (op 1001) 


s_ashr_i32 s1, s0, s18 // (op 1001) 


s_lshl_b32 s0, s12, 24 // (op 1001) 
s_ashr_i32 s18, s0, 31 // (op 1001) 

s_xor_b32 s0, s1, s18 // (op 1001) 

s_sub_i32 s1, s0, s18 // (op 1001) 


s_mul_hi_i32 s0, s1, s13 // (op 1001) 
s_add_i32 s18, s0, s1 // (op 1001) 

s_ashr_i32 s0, s18, 31 // (op 1001) 
s_cmp_eq_i32 s13, 0 // (op 1001) 
s_mov_b32 s19, scc // (op 1001) 
s_cmp_eq_i32 s19, 0 // (op 1001) 
s_cselect_b32 s49, 0, -1 // (op 1001) 

s_and_b32 s19, s14, 31 // (op 1001) 
s_lshl_b32 s50, 1, s19 // (op 1001) 
s_add_i32 s51, s50, s49 // (op 1001) 


s_and_b32 s49, s0, s51 // (op 1001) 


s_add_i32 s0, s18, s49 // (op 1001) 


s_ashr_i32 s18, s0, s19 // (op 1001) 


s_lshl_b32 s0, s14, 24 // (op 1001) 
s_ashr_i32 s19, s0, 31 // (op 1001) 

s_xor_b32 s0, s18, s19 // (op 1001) 

s_sub_i32 s18, s0, s19 // (op 1001) 


s_add_u32 s50, s6, 64 // (op 1001) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1001) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1001) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1001) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1001) 

s_mul_i32 s0, s18, s50 // (op 1001) 

s_sub_i32 s49, s1, s0 // (op 1001) 


s_mul_hi_i32 s0, s16, s10 // (op 1001) 
s_add_i32 s1, s0, s16 // (op 1001) 

s_ashr_i32 s0, s1, 31 // (op 1001) 
s_cmp_eq_i32 s10, 0 // (op 1001) 
s_mov_b32 s18, scc // (op 1001) 
s_cmp_eq_i32 s18, 0 // (op 1001) 
s_cselect_b32 s19, 0, -1 // (op 1001) 

s_and_b32 s18, s47, 31 // (op 1001) 
s_lshl_b32 s64, 1, s18 // (op 1001) 
s_add_i32 s65, s64, s19 // (op 1001) 


s_and_b32 s19, s0, s65 // (op 1001) 


s_add_i32 s0, s1, s19 // (op 1001) 


s_ashr_i32 s1, s0, s18 // (op 1001) 


s_lshl_b32 s0, s47, 24 // (op 1001) 
s_ashr_i32 s18, s0, 31 // (op 1001) 

s_xor_b32 s0, s1, s18 // (op 1001) 

s_sub_i32 s1, s0, s18 // (op 1001) 


s_mul_i32 s0, s44, s50 // (op 1001) 

s_mul_i32 s18, s48, s0 // (op 1001) 

s_mul_i32 s0, s1, s18 // (op 1001) 

s_sub_u32 s18, s16, s0 // (op 1001) 

s_mul_hi_i32 s0, s18, s15 // (op 1001) 
s_add_i32 s19, s0, s18 // (op 1001) 


s_ashr_i32 s0, s19, 31 // (op 1001) 
s_cmp_eq_i32 s15, 0 // (op 1001) 
s_mov_b32 s18, scc // (op 1001) 
s_cmp_eq_i32 s18, 0 // (op 1001) 
s_cselect_b32 s50, 0, -1 // (op 1001) 

s_and_b32 s18, s46, 31 // (op 1001) 
s_lshl_b32 s51, 1, s18 // (op 1001) 
s_add_i32 s64, s51, s50 // (op 1001) 


s_and_b32 s50, s0, s64 // (op 1001) 


s_add_i32 s0, s19, s50 // (op 1001) 


s_ashr_i32 s19, s0, s18 // (op 1001) 


s_lshl_b32 s0, s46, 24 // (op 1001) 
s_ashr_i32 s18, s0, 31 // (op 1001) 

s_xor_b32 s0, s19, s18 // (op 1001) 

s_sub_i32 s19, s0, s18 // (op 1001) 


s_cmp_eq_i32 s1, 0 // (op 1001) 
s_mov_b32 s0, scc // (op 1001) 

s_cmp_eq_i32 s0, 0 // (op 1001) 
s_cselect_b32 s1, s19, s49 // (op 1001) 



v_lshl_add_u32 v18, s1, 6, v14 // (op 1001) 


v_mov_b32 v21, 0 // (op 1001) convert
v_mov_b32 v20, v18 // (op 1001) convert

v_mul_lo_u32 v19, v15, s29 // (op 1001) most significant: low of low * high
v_mul_hi_u32 v14, v15, s28 // (op 1001) most significant: high of low * low
v_mul_lo_u32 v18, v15, s28 // (op 1001) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1001) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v20 // (op 1001) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 1001) most significant half



s_mul_i32 s1, s28, s33 // (op 1003) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 1003) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1003) least significant
s_mul_hi_u32 s19, s28, s32 // (op 1003) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1003) most significant: sum
s_add_u32 s1, s1, s19 // (op 1003) most significant: sum


s_mov_b32 s52, s30 // (op 1003) reg expression
s_mov_b32 s53, s31 // (op 1003) reg expression
s_mov_b32 s54, s0 // (op 1003) reg expression
s_mov_b32 s55, 131072 // (op 1003) call()

s_mul_hi_i32 s0, s16, s11 // (op 1046) 
v_lshrrev_b32 v14, 7, v1 // (op 1046) 
s_add_i32 s1, s0, s16 // (op 1046) 

v_and_b32 v15, 15, v1 // (op 1046) 
s_ashr_i32 s0, s1, 31 // (op 1046) 
v_lshl_add_u32 v20, v14, 4, v15 // (op 1046) 


s_cmp_eq_i32 s11, 0 // (op 1046) 
s_mov_b32 s18, scc // (op 1046) 
v_and_b32 v14, 63, v1 // (op 1046) 
s_cmp_eq_i32 s18, 0 // (op 1046) 
s_cselect_b32 s19, 0, -1 // (op 1046) 

v_lshrrev_b32 v15, 4, v14 // (op 1046) 

s_and_b32 s18, s12, 31 // (op 1046) 
s_lshl_b32 s49, 1, s18 // (op 1046) 
s_add_i32 s50, s49, s19 // (op 1046) 


s_and_b32 s19, s0, s50 // (op 1046) 


s_add_i32 s0, s1, s19 // (op 1046) 


s_ashr_i32 s1, s0, s18 // (op 1046) 


s_lshl_b32 s0, s12, 24 // (op 1046) 
s_ashr_i32 s18, s0, 31 // (op 1046) 

s_xor_b32 s0, s1, s18 // (op 1046) 

s_sub_i32 s1, s0, s18 // (op 1046) 


s_mul_hi_i32 s0, s1, s13 // (op 1046) 
s_add_i32 s18, s0, s1 // (op 1046) 

s_ashr_i32 s0, s18, 31 // (op 1046) 
s_cmp_eq_i32 s13, 0 // (op 1046) 
s_mov_b32 s19, scc // (op 1046) 
s_cmp_eq_i32 s19, 0 // (op 1046) 
s_cselect_b32 s49, 0, -1 // (op 1046) 

s_and_b32 s19, s14, 31 // (op 1046) 
s_lshl_b32 s50, 1, s19 // (op 1046) 
s_add_i32 s51, s50, s49 // (op 1046) 


s_and_b32 s49, s0, s51 // (op 1046) 


s_add_i32 s0, s18, s49 // (op 1046) 


s_ashr_i32 s18, s0, s19 // (op 1046) 


s_lshl_b32 s0, s14, 24 // (op 1046) 
s_ashr_i32 s19, s0, 31 // (op 1046) 

s_xor_b32 s0, s18, s19 // (op 1046) 

s_sub_i32 s18, s0, s19 // (op 1046) 


s_mul_i32 s0, s1, s44 // (op 1046) 

s_sub_u32 s1, s16, s0 // (op 1046) 

s_mul_i32 s0, s18, s44 // (op 1046) 
s_add_u32 s0, s0, s1 // (op 1046) 


s_mul_hi_i32 s1, s16, s10 // (op 1046) 
s_add_i32 s18, s1, s16 // (op 1046) 

s_ashr_i32 s1, s18, 31 // (op 1046) 
s_cmp_eq_i32 s10, 0 // (op 1046) 
s_mov_b32 s19, scc // (op 1046) 
s_cmp_eq_i32 s19, 0 // (op 1046) 
s_cselect_b32 s49, 0, -1 // (op 1046) 

s_and_b32 s19, s47, 31 // (op 1046) 
s_lshl_b32 s50, 1, s19 // (op 1046) 
s_add_i32 s51, s50, s49 // (op 1046) 


s_and_b32 s49, s1, s51 // (op 1046) 


s_add_i32 s1, s18, s49 // (op 1046) 


s_ashr_i32 s18, s1, s19 // (op 1046) 


s_lshl_b32 s1, s47, 24 // (op 1046) 
s_ashr_i32 s19, s1, 31 // (op 1046) 

s_xor_b32 s1, s18, s19 // (op 1046) 

s_sub_i32 s18, s1, s19 // (op 1046) 


s_add_u32 s50, s6, 64 // (op 1046) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1046) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1046) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1046) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1046) 

s_mul_i32 s1, s44, s50 // (op 1046) 

s_mul_i32 s19, s48, s1 // (op 1046) 

s_mul_i32 s1, s18, s19 // (op 1046) 

s_sub_u32 s19, s16, s1 // (op 1046) 

s_mov_b32 s1, s19 // (op 1046) 
s_mul_hi_i32 s49, s1, s15 // (op 1046) 
s_add_i32 s50, s49, s1 // (op 1046) 


s_ashr_i32 s1, s50, 31 // (op 1046) 
s_cmp_eq_i32 s15, 0 // (op 1046) 
s_mov_b32 s49, scc // (op 1046) 
s_cmp_eq_i32 s49, 0 // (op 1046) 
s_cselect_b32 s51, 0, -1 // (op 1046) 

s_and_b32 s49, s46, 31 // (op 1046) 
s_lshl_b32 s64, 1, s49 // (op 1046) 
s_add_i32 s65, s64, s51 // (op 1046) 


s_and_b32 s51, s1, s65 // (op 1046) 


s_add_i32 s1, s50, s51 // (op 1046) 


s_ashr_i32 s50, s1, s49 // (op 1046) 


s_lshl_b32 s1, s46, 24 // (op 1046) 
s_ashr_i32 s49, s1, 31 // (op 1046) 

s_xor_b32 s1, s50, s49 // (op 1046) 

s_sub_i32 s50, s1, s49 // (op 1046) 


s_mul_i32 s1, s45, s44 // (op 1046) 
s_add_u32 s64, s8, 16 // (op 1046) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 1046) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1046) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1046) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 1046) 

s_sub_i32 s49, s64, s1 // (op 1046) 

s_mul_i32 s51, s50, s49 // (op 1046) 


s_sub_u32 s49, s19, s51 // (op 1046) 


s_add_u32 s19, s49, s1 // (op 1046) 


s_cmp_eq_i32 s18, 0 // (op 1046) 
s_mov_b32 s1, scc // (op 1046) 

s_cmp_eq_i32 s1, 0 // (op 1046) 
s_cselect_b32 s18, s19, s0 // (op 1046) 



v_lshl_add_u32 v14, s18, 4, v20 // (op 1046) 


v_mul_lo_u32 v20, v14, s34 // (op 1046) 
v_add_u32 v20, v20, v15 // (op 1046) 


s_mul_i32 s1, s34, s9 // (op 1048) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1048) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1048) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1048) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1048) most significant: sum
s_add_u32 s1, s1, s19 // (op 1048) most significant: sum


s_mov_b32 s64, s36 // (op 1048) reg expression
s_mov_b32 s65, s37 // (op 1048) reg expression
s_mov_b32 s66, s0 // (op 1048) reg expression
s_mov_b32 s67, 131072 // (op 1048) call()

s_mul_hi_i32 s0, s16, s11 // (op 1065) 
v_lshrrev_b32 v14, 7, v1 // (op 1065) 
s_add_i32 s1, s0, s16 // (op 1065) 

v_and_b32 v15, 15, v1 // (op 1065) 
s_ashr_i32 s0, s1, 31 // (op 1065) 
v_lshl_add_u32 v21, v14, 4, v15 // (op 1065) 


s_cmp_eq_i32 s11, 0 // (op 1065) 
s_mov_b32 s18, scc // (op 1065) 
v_and_b32 v14, 63, v1 // (op 1065) 
s_cmp_eq_i32 s18, 0 // (op 1065) 
s_cselect_b32 s19, 0, -1 // (op 1065) 

v_lshrrev_b32 v15, 4, v14 // (op 1065) 

s_and_b32 s18, s12, 31 // (op 1065) 
s_lshl_b32 s49, 1, s18 // (op 1065) 
s_add_i32 s50, s49, s19 // (op 1065) 


s_and_b32 s19, s0, s50 // (op 1065) 


s_add_i32 s0, s1, s19 // (op 1065) 


s_ashr_i32 s1, s0, s18 // (op 1065) 


s_lshl_b32 s0, s12, 24 // (op 1065) 
s_ashr_i32 s18, s0, 31 // (op 1065) 

s_xor_b32 s0, s1, s18 // (op 1065) 

s_sub_i32 s1, s0, s18 // (op 1065) 


s_mul_hi_i32 s0, s1, s13 // (op 1065) 
s_add_i32 s18, s0, s1 // (op 1065) 

s_ashr_i32 s0, s18, 31 // (op 1065) 
s_cmp_eq_i32 s13, 0 // (op 1065) 
s_mov_b32 s19, scc // (op 1065) 
s_cmp_eq_i32 s19, 0 // (op 1065) 
s_cselect_b32 s49, 0, -1 // (op 1065) 

s_and_b32 s19, s14, 31 // (op 1065) 
s_lshl_b32 s50, 1, s19 // (op 1065) 
s_add_i32 s51, s50, s49 // (op 1065) 


s_and_b32 s49, s0, s51 // (op 1065) 


s_add_i32 s0, s18, s49 // (op 1065) 


s_ashr_i32 s18, s0, s19 // (op 1065) 


s_lshl_b32 s0, s14, 24 // (op 1065) 
s_ashr_i32 s19, s0, 31 // (op 1065) 

s_xor_b32 s0, s18, s19 // (op 1065) 

s_sub_i32 s18, s0, s19 // (op 1065) 


s_mul_i32 s0, s1, s44 // (op 1065) 

s_sub_u32 s1, s16, s0 // (op 1065) 

s_mul_i32 s0, s18, s44 // (op 1065) 
s_add_u32 s0, s0, s1 // (op 1065) 


s_mul_hi_i32 s1, s16, s10 // (op 1065) 
s_add_i32 s18, s1, s16 // (op 1065) 

s_ashr_i32 s1, s18, 31 // (op 1065) 
s_cmp_eq_i32 s10, 0 // (op 1065) 
s_mov_b32 s19, scc // (op 1065) 
s_cmp_eq_i32 s19, 0 // (op 1065) 
s_cselect_b32 s49, 0, -1 // (op 1065) 

s_and_b32 s19, s47, 31 // (op 1065) 
s_lshl_b32 s50, 1, s19 // (op 1065) 
s_add_i32 s51, s50, s49 // (op 1065) 


s_and_b32 s49, s1, s51 // (op 1065) 


s_add_i32 s1, s18, s49 // (op 1065) 


s_ashr_i32 s18, s1, s19 // (op 1065) 


s_lshl_b32 s1, s47, 24 // (op 1065) 
s_ashr_i32 s19, s1, 31 // (op 1065) 

s_xor_b32 s1, s18, s19 // (op 1065) 

s_sub_i32 s18, s1, s19 // (op 1065) 


s_add_u32 s50, s6, 64 // (op 1065) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1065) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1065) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1065) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1065) 

s_mul_i32 s1, s44, s50 // (op 1065) 

s_mul_i32 s19, s48, s1 // (op 1065) 

s_mul_i32 s1, s18, s19 // (op 1065) 

s_sub_u32 s19, s16, s1 // (op 1065) 

s_mov_b32 s1, s19 // (op 1065) 
s_mul_hi_i32 s49, s1, s15 // (op 1065) 
s_add_i32 s50, s49, s1 // (op 1065) 


s_ashr_i32 s1, s50, 31 // (op 1065) 
s_cmp_eq_i32 s15, 0 // (op 1065) 
s_mov_b32 s49, scc // (op 1065) 
s_cmp_eq_i32 s49, 0 // (op 1065) 
s_cselect_b32 s51, 0, -1 // (op 1065) 

s_and_b32 s49, s46, 31 // (op 1065) 
s_lshl_b32 s68, 1, s49 // (op 1065) 
s_add_i32 s69, s68, s51 // (op 1065) 


s_and_b32 s51, s1, s69 // (op 1065) 


s_add_i32 s1, s50, s51 // (op 1065) 


s_ashr_i32 s50, s1, s49 // (op 1065) 


s_lshl_b32 s1, s46, 24 // (op 1065) 
s_ashr_i32 s49, s1, 31 // (op 1065) 

s_xor_b32 s1, s50, s49 // (op 1065) 

s_sub_i32 s50, s1, s49 // (op 1065) 


s_mul_i32 s1, s45, s44 // (op 1065) 
s_add_u32 s68, s8, 16 // (op 1065) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 1065) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1065) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1065) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 1065) 

s_sub_i32 s49, s68, s1 // (op 1065) 

s_mul_i32 s51, s50, s49 // (op 1065) 


s_sub_u32 s49, s19, s51 // (op 1065) 


s_add_u32 s19, s49, s1 // (op 1065) 


s_cmp_eq_i32 s18, 0 // (op 1065) 
s_mov_b32 s1, scc // (op 1065) 

s_cmp_eq_i32 s1, 0 // (op 1065) 
s_cselect_b32 s18, s19, s0 // (op 1065) 



v_lshl_add_u32 v14, s18, 4, v21 // (op 1065) 


v_mul_lo_u32 v21, v14, s34 // (op 1065) 
v_add_u32 v21, v21, v15 // (op 1065) 


s_mul_i32 s1, s34, s9 // (op 1067) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1067) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1067) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1067) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1067) most significant: sum
s_add_u32 s1, s1, s19 // (op 1067) most significant: sum


s_mov_b32 s64, s36 // (op 1067) reg expression
s_mov_b32 s65, s37 // (op 1067) reg expression
s_mov_b32 s66, s0 // (op 1067) reg expression
s_mov_b32 s67, 131072 // (op 1067) call()

s_mul_hi_i32 s0, s16, s11 // (op 1110) 
v_and_b32 v14, -64, v1 // (op 1110) 
s_add_i32 s1, s0, s16 // (op 1110) 

v_sub_u32 v15, v1, v14 // (op 1110) 

s_ashr_i32 s0, s1, 31 // (op 1110) 
v_lshrrev_b32 v14, 6, v1 // (op 1110) 
s_cmp_eq_i32 s11, 0 // (op 1110) 
s_mov_b32 s18, scc // (op 1110) 
v_lshlrev_b32 v22, 1, v14 // (op 1110) 

s_cmp_eq_i32 s18, 0 // (op 1110) 
s_cselect_b32 s19, 0, -1 // (op 1110) 

s_and_b32 s18, s12, 31 // (op 1110) 
s_lshl_b32 s49, 1, s18 // (op 1110) 
s_add_i32 s50, s49, s19 // (op 1110) 


s_and_b32 s19, s0, s50 // (op 1110) 


s_add_i32 s0, s1, s19 // (op 1110) 


s_ashr_i32 s1, s0, s18 // (op 1110) 


s_lshl_b32 s0, s12, 24 // (op 1110) 
s_ashr_i32 s18, s0, 31 // (op 1110) 

s_xor_b32 s0, s1, s18 // (op 1110) 

s_sub_i32 s1, s0, s18 // (op 1110) 


s_mul_hi_i32 s0, s1, s13 // (op 1110) 
s_add_i32 s18, s0, s1 // (op 1110) 

s_ashr_i32 s0, s18, 31 // (op 1110) 
s_cmp_eq_i32 s13, 0 // (op 1110) 
s_mov_b32 s19, scc // (op 1110) 
s_cmp_eq_i32 s19, 0 // (op 1110) 
s_cselect_b32 s49, 0, -1 // (op 1110) 

s_and_b32 s19, s14, 31 // (op 1110) 
s_lshl_b32 s50, 1, s19 // (op 1110) 
s_add_i32 s51, s50, s49 // (op 1110) 


s_and_b32 s49, s0, s51 // (op 1110) 


s_add_i32 s0, s18, s49 // (op 1110) 


s_ashr_i32 s18, s0, s19 // (op 1110) 


s_lshl_b32 s0, s14, 24 // (op 1110) 
s_ashr_i32 s19, s0, 31 // (op 1110) 

s_xor_b32 s0, s18, s19 // (op 1110) 

s_sub_i32 s18, s0, s19 // (op 1110) 


s_add_u32 s50, s6, 64 // (op 1110) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1110) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1110) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1110) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1110) 

s_mul_i32 s0, s18, s50 // (op 1110) 

s_sub_i32 s49, s1, s0 // (op 1110) 


s_mul_hi_i32 s0, s16, s10 // (op 1110) 
s_add_i32 s1, s0, s16 // (op 1110) 

s_ashr_i32 s0, s1, 31 // (op 1110) 
s_cmp_eq_i32 s10, 0 // (op 1110) 
s_mov_b32 s18, scc // (op 1110) 
s_cmp_eq_i32 s18, 0 // (op 1110) 
s_cselect_b32 s19, 0, -1 // (op 1110) 

s_and_b32 s18, s47, 31 // (op 1110) 
s_lshl_b32 s68, 1, s18 // (op 1110) 
s_add_i32 s69, s68, s19 // (op 1110) 


s_and_b32 s19, s0, s69 // (op 1110) 


s_add_i32 s0, s1, s19 // (op 1110) 


s_ashr_i32 s1, s0, s18 // (op 1110) 


s_lshl_b32 s0, s47, 24 // (op 1110) 
s_ashr_i32 s18, s0, 31 // (op 1110) 

s_xor_b32 s0, s1, s18 // (op 1110) 

s_sub_i32 s1, s0, s18 // (op 1110) 


s_mul_i32 s0, s44, s50 // (op 1110) 

s_mul_i32 s18, s48, s0 // (op 1110) 

s_mul_i32 s0, s1, s18 // (op 1110) 

s_sub_u32 s18, s16, s0 // (op 1110) 

s_mul_hi_i32 s0, s18, s15 // (op 1110) 
s_add_i32 s19, s0, s18 // (op 1110) 


s_ashr_i32 s0, s19, 31 // (op 1110) 
s_cmp_eq_i32 s15, 0 // (op 1110) 
s_mov_b32 s18, scc // (op 1110) 
s_cmp_eq_i32 s18, 0 // (op 1110) 
s_cselect_b32 s50, 0, -1 // (op 1110) 

s_and_b32 s18, s46, 31 // (op 1110) 
s_lshl_b32 s51, 1, s18 // (op 1110) 
s_add_i32 s68, s51, s50 // (op 1110) 


s_and_b32 s50, s0, s68 // (op 1110) 


s_add_i32 s0, s19, s50 // (op 1110) 


s_ashr_i32 s19, s0, s18 // (op 1110) 


s_lshl_b32 s0, s46, 24 // (op 1110) 
s_ashr_i32 s18, s0, 31 // (op 1110) 

s_xor_b32 s0, s19, s18 // (op 1110) 

s_sub_i32 s19, s0, s18 // (op 1110) 


s_cmp_eq_i32 s1, 0 // (op 1110) 
s_mov_b32 s0, scc // (op 1110) 

s_cmp_eq_i32 s0, 0 // (op 1110) 
s_cselect_b32 s1, s19, s49 // (op 1110) 



v_lshl_add_u32 v14, s1, 6, v15 // (op 1110) 


v_mul_lo_u32 v15, v22, s38 // (op 1110) 
v_add_u32 v15, v15, v14 // (op 1110) 


s_mul_i32 s1, s38, s43 // (op 1112) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1112) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1112) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1112) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1112) most significant: sum
s_add_u32 s1, s1, s19 // (op 1112) most significant: sum


s_mov_b32 s68, s40 // (op 1112) reg expression
s_mov_b32 s69, s41 // (op 1112) reg expression
s_mov_b32 s70, s0 // (op 1112) reg expression
s_mov_b32 s71, 131072 // (op 1112) call()

s_mul_hi_i32 s0, s16, s11 // (op 1129) 
v_and_b32 v14, -64, v1 // (op 1129) 
s_add_i32 s1, s0, s16 // (op 1129) 

v_sub_u32 v22, v1, v14 // (op 1129) 

s_ashr_i32 s0, s1, 31 // (op 1129) 
v_lshrrev_b32 v14, 6, v1 // (op 1129) 
s_cmp_eq_i32 s11, 0 // (op 1129) 
s_mov_b32 s18, scc // (op 1129) 
v_lshlrev_b32 v23, 1, v14 // (op 1129) 

s_cmp_eq_i32 s18, 0 // (op 1129) 
s_cselect_b32 s19, 0, -1 // (op 1129) 

s_and_b32 s18, s12, 31 // (op 1129) 
s_lshl_b32 s49, 1, s18 // (op 1129) 
s_add_i32 s50, s49, s19 // (op 1129) 


s_and_b32 s19, s0, s50 // (op 1129) 


s_add_i32 s0, s1, s19 // (op 1129) 


s_ashr_i32 s1, s0, s18 // (op 1129) 


s_lshl_b32 s0, s12, 24 // (op 1129) 
s_ashr_i32 s18, s0, 31 // (op 1129) 

s_xor_b32 s0, s1, s18 // (op 1129) 

s_sub_i32 s1, s0, s18 // (op 1129) 


s_mul_hi_i32 s0, s1, s13 // (op 1129) 
s_add_i32 s18, s0, s1 // (op 1129) 

s_ashr_i32 s0, s18, 31 // (op 1129) 
s_cmp_eq_i32 s13, 0 // (op 1129) 
s_mov_b32 s19, scc // (op 1129) 
s_cmp_eq_i32 s19, 0 // (op 1129) 
s_cselect_b32 s49, 0, -1 // (op 1129) 

s_and_b32 s19, s14, 31 // (op 1129) 
s_lshl_b32 s50, 1, s19 // (op 1129) 
s_add_i32 s51, s50, s49 // (op 1129) 


s_and_b32 s49, s0, s51 // (op 1129) 


s_add_i32 s0, s18, s49 // (op 1129) 


s_ashr_i32 s18, s0, s19 // (op 1129) 


s_lshl_b32 s0, s14, 24 // (op 1129) 
s_ashr_i32 s19, s0, 31 // (op 1129) 

s_xor_b32 s0, s18, s19 // (op 1129) 

s_sub_i32 s18, s0, s19 // (op 1129) 


s_add_u32 s50, s6, 64 // (op 1129) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1129) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1129) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1129) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1129) 

s_mul_i32 s0, s18, s50 // (op 1129) 

s_sub_i32 s49, s1, s0 // (op 1129) 


s_mul_hi_i32 s0, s16, s10 // (op 1129) 
s_add_i32 s1, s0, s16 // (op 1129) 

s_ashr_i32 s0, s1, 31 // (op 1129) 
s_cmp_eq_i32 s10, 0 // (op 1129) 
s_mov_b32 s18, scc // (op 1129) 
s_cmp_eq_i32 s18, 0 // (op 1129) 
s_cselect_b32 s19, 0, -1 // (op 1129) 

s_and_b32 s18, s47, 31 // (op 1129) 
s_lshl_b32 s72, 1, s18 // (op 1129) 
s_add_i32 s73, s72, s19 // (op 1129) 


s_and_b32 s19, s0, s73 // (op 1129) 


s_add_i32 s0, s1, s19 // (op 1129) 


s_ashr_i32 s1, s0, s18 // (op 1129) 


s_lshl_b32 s0, s47, 24 // (op 1129) 
s_ashr_i32 s18, s0, 31 // (op 1129) 

s_xor_b32 s0, s1, s18 // (op 1129) 

s_sub_i32 s1, s0, s18 // (op 1129) 


s_mul_i32 s0, s44, s50 // (op 1129) 

s_mul_i32 s18, s48, s0 // (op 1129) 

s_mul_i32 s0, s1, s18 // (op 1129) 

s_sub_u32 s18, s16, s0 // (op 1129) 

s_mul_hi_i32 s0, s18, s15 // (op 1129) 
s_add_i32 s19, s0, s18 // (op 1129) 


s_ashr_i32 s0, s19, 31 // (op 1129) 
s_cmp_eq_i32 s15, 0 // (op 1129) 
s_mov_b32 s18, scc // (op 1129) 
s_cmp_eq_i32 s18, 0 // (op 1129) 
s_cselect_b32 s50, 0, -1 // (op 1129) 

s_and_b32 s18, s46, 31 // (op 1129) 
s_lshl_b32 s51, 1, s18 // (op 1129) 
s_add_i32 s72, s51, s50 // (op 1129) 


s_and_b32 s50, s0, s72 // (op 1129) 


s_add_i32 s0, s19, s50 // (op 1129) 


s_ashr_i32 s19, s0, s18 // (op 1129) 


s_lshl_b32 s0, s46, 24 // (op 1129) 
s_ashr_i32 s18, s0, 31 // (op 1129) 

s_xor_b32 s0, s19, s18 // (op 1129) 

s_sub_i32 s19, s0, s18 // (op 1129) 


s_cmp_eq_i32 s1, 0 // (op 1129) 
s_mov_b32 s0, scc // (op 1129) 

s_cmp_eq_i32 s0, 0 // (op 1129) 
s_cselect_b32 s1, s19, s49 // (op 1129) 



v_lshl_add_u32 v14, s1, 6, v22 // (op 1129) 


v_mul_lo_u32 v22, v23, s38 // (op 1129) 
v_add_u32 v22, v22, v14 // (op 1129) 


s_mul_i32 s1, s38, s43 // (op 1131) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1131) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1131) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1131) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1131) most significant: sum
s_add_u32 s1, s1, s19 // (op 1131) most significant: sum


s_mov_b32 s68, s40 // (op 1131) reg expression
s_mov_b32 s69, s41 // (op 1131) reg expression
s_mov_b32 s70, s0 // (op 1131) reg expression
s_mov_b32 s71, 131072 // (op 1131) call()




buffer_load_ubyte v14, v20, s[64:67], 0 offen offset: 0 // (op 761) Load value
v_and_b32 v23, -4, v1 // (op 1248) 
v_sub_u32 v24, v1, v23 // (op 1248) 

v_lshlrev_b32 v23, 4, v24 // (op 1248) 

v_lshrrev_b32 v24, 2, v1 // (op 1248) 
v_lshl_add_u32 v25, v24, 6, v23 // (op 1248) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 1248) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 5 // (op 873) 
v_mov_b64 v[24:25], s[18:19] // (op 873) convert


v_mov_b64 v[26:27], v[16:17] // (op 873) reg expression
s_mov_b32 m0, s0 // (op 873) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 873) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 873) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 873) most significant half

s_add_u32 m0, m0, 2048 // (op 873) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 873) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 873) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 873) most significant half

s_add_u32 m0, m0, 2048 // (op 873) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 873) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 873) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 873) most significant half

s_add_u32 m0, m0, 2048 // (op 873) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 873) Load value direct to lds




v_mov_b64 v[24:25], s[38:39] // (op 768) convert

v_mov_b32 v23, v15 // (op 768) reg expression
buffer_load_ubyte v26, v23, s[68:71], 0 offen offset: 0 // (op 768) Load value
v_add_u32 v23, v23, v24 // (op 768) 
buffer_load_ubyte v27, v23, s[68:71], 0 offen offset: 0 // (op 768) Load value





buffer_load_dwordx3 v[28:30], v12, s[20:23], 0 offen // (op 771) Load value



buffer_load_ubyte v23, v20, s[64:67], 0 offen offset:4 // (op 775) Load value

v_and_b32 v20, -4, v1 // (op 1392) 
v_sub_u32 v24, v1, v20 // (op 1392) 

v_lshlrev_b32 v20, 4, v24 // (op 1392) 

v_lshrrev_b32 v24, 2, v1 // (op 1392) 
v_lshl_add_u32 v25, v24, 6, v20 // (op 1392) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 1392) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 5 // (op 878) 
v_mov_b64 v[24:25], s[18:19] // (op 878) convert


s_lshl_b64 s[18:19], s[28:29], 7 // (op 878) 
v_mov_b32 v20, s18 // (op 878) 
v_mov_b32 v31, s19 // (op 878) 
v_add_co_u32 v32, s[50:51], v16, v20 // (op 878) least significant half
v_addc_co_u32 v33, s[50:51], v17, v31, s[50:51] // (op 878) most significant half




s_add_u32 s1, 8192, s0 // (op 878) 
s_mov_b32 m0, s1 // (op 878) reg expression
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 878) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v24 // (op 878) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 878) most significant half

s_add_u32 m0, m0, 2048 // (op 878) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 878) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v24 // (op 878) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 878) most significant half

s_add_u32 m0, m0, 2048 // (op 878) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 878) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v24 // (op 878) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 878) most significant half

s_add_u32 m0, m0, 2048 // (op 878) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 878) Load value direct to lds






v_mov_b64 v[16:17], s[38:39] // (op 782) convert

s_lshl_b32 s0, s38, 2 // (op 782) 
v_add_u32 v20, v15, s0 // (op 782) 

buffer_load_ubyte v24, v20, s[68:71], 0 offen offset: 0 // (op 782) Load value
v_add_u32 v20, v20, v16 // (op 782) 
buffer_load_ubyte v25, v20, s[68:71], 0 offen offset: 0 // (op 782) Load value






buffer_load_dwordx3 v[32:34], v12, s[20:23], 0 offen offset:96 // (op 785) Load value

v_and_b32 v12, -64, v1 // (op 1306) 
v_sub_u32 v20, v1, v12 // (op 1306) 

v_lshrrev_b32 v12, 6, v1 // (op 1306) 
v_lshl_add_u32 v31, v12, 7, v20 // (op 1306) 





s_waitcnt vmcnt(10)
ds_write_b8 v31, v26 offset:16384 // (op 788) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v31, v27 offset:16448 // (op 788) Store local data 

v_and_b32 v12, -8, v1 // (op 1196) 
v_sub_u32 v20, v1, v12 // (op 1196) 

v_lshlrev_b32 v12, 4, v20 // (op 1196) 

v_lshrrev_b32 v20, 3, v1 // (op 1196) 
v_lshl_add_u32 v31, v20, 7, v12 // (op 1196) 


v_mul_lo_u32 v12, v31, 6 // (op 1196) 

v_lshrrev_b32 v20, 3, v12 // (op 1196) 




s_waitcnt vmcnt(8)
ds_write_b96 v20, v[28:30] offset:16640 // (op 789) Store local data 

v_and_b32 v12, -8, v1 // (op 1167) 
v_sub_u32 v20, v1, v12 // (op 1167) 

v_lshlrev_b32 v12, 4, v20 // (op 1167) 

v_lshrrev_b32 v20, 3, v1 // (op 1167) 
v_lshl_add_u32 v31, v20, 7, v12 // (op 1167) 


v_mul_lo_u32 v12, v31, 6 // (op 1167) 

v_lshrrev_b32 v20, 3, v12 // (op 1167) 

v_and_b32 v12, -4, v1 // (op 1220) 
v_sub_u32 v31, v1, v12 // (op 1220) 

v_lshlrev_b32 v12, 4, v31 // (op 1220) 

v_lshrrev_b32 v31, 2, v1 // (op 1220) 
v_lshl_add_u32 v15, v31, 6, v12 // (op 1220) 


s_nop 0
v_readfirstlane_b32 s0, v15 // (op 1220) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1277) 
v_sub_u32 v31, v1, v12 // (op 1277) 

v_lshrrev_b32 v12, 6, v1 // (op 1277) 
v_lshl_add_u32 v15, v12, 7, v31 // (op 1277) 


v_and_b32 v12, -8, v1 // (op 1353) 
v_sub_u32 v31, v1, v12 // (op 1353) 

v_lshlrev_b32 v12, 4, v31 // (op 1353) 

v_lshrrev_b32 v31, 3, v1 // (op 1353) 
v_lshl_add_u32 v16, v31, 7, v12 // (op 1353) 


v_mul_lo_u32 v12, v16, 6 // (op 1353) 

v_lshrrev_b32 v31, 3, v12 // (op 1353) 

v_and_b32 v12, -4, v1 // (op 1377) 
v_sub_u32 v16, v1, v12 // (op 1377) 

v_lshlrev_b32 v12, 4, v16 // (op 1377) 

v_lshrrev_b32 v16, 2, v1 // (op 1377) 
v_lshl_add_u32 v17, v16, 6, v12 // (op 1377) 


s_nop 0
v_readfirstlane_b32 s1, v17 // (op 1377) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1421) 
v_sub_u32 v16, v1, v12 // (op 1421) 

v_lshrrev_b32 v12, 6, v1 // (op 1421) 
v_lshl_add_u32 v17, v12, 7, v16 // (op 1421) 


s_mov_b32 s18, 0 // (op 308) call()
s_ashr_i64 s[50:51], s[32:33], 7 // (op 307) 
s_mov_b32 s19, s50 // (op 307) convert

s_and_b32 s49, s19, 4294967294 // (op 307) 

s_cmp_lt_u32 s18, s49 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_307 // (op 307) Condition: Top (jump to RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_307 if false)
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_307:
s_add_u32 s18, s18, 2 // (op 309) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1479) 



ds_read_b128 v[36:39], v3 offset:16640 // (op 65) Load local data 
ds_read_b64 v[40:41], v3 offset:16656 // (op 65) Load local data 
s_barrier  // (op 1492) 



ds_read_b64_tr_b8 v[42:43], v7 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:512 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:4096 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:4608 // (op 70) Transpose load local data 
s_barrier  // (op 1505) 


ds_read_u8 v12, v0 offset:16384 // (op 79) Load local data 



ds_read_b64_tr_b8 v[50:51], v7 offset:32 // (op 339) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v7 offset:544 // (op 339) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:49], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 82) 
ds_read_b64_tr_b8 v[54:55], v7 offset:4128 // (op 339) Transpose load local data 


ds_read_b64_tr_b8 v[56:57], v7 offset:4640 // (op 339) Transpose load local data 


ds_read_u8 v12, v0 offset:16416 // (op 342) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[50:57], a[4:7], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 343) 





v_mov_b32 v12, v21 // (op 75) 

buffer_load_ubyte v14, v12, s[64:67], 0 offen offset:8 // (op 75) Load value
s_barrier  // (op 1495) 

s_lshl_b64 s[50:51], s[28:29], 5 // (op 856) 
v_mov_b64 v[36:37], s[50:51] // (op 856) convert


v_mov_b64 v[38:39], v[18:19] // (op 856) 
s_lshl_b64 s[50:51], s[28:29], 7 // (op 856) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 856) 

v_mov_b32 v16, s72 // (op 856) 
v_mov_b32 v35, s73 // (op 856) 
v_add_co_u32 v40, s[50:51], v38, v16 // (op 856) least significant half
v_addc_co_u32 v41, s[50:51], v39, v35, s[50:51] // (op 856) most significant half




s_mov_b32 m0, s0 // (op 856) reg expression
buffer_load_dwordx4 v40, s[52:55], 0 offen offset: 0 lds // (op 856) Load value direct to lds
v_add_co_u32 v40, s[50:51], v40, v36 // (op 856) least significant half
v_addc_co_u32 v41, s[50:51], 0, v41, s[50:51] // (op 856) most significant half

s_add_u32 m0, m0, 2048 // (op 856) 
buffer_load_dwordx4 v40, s[52:55], 0 offen offset: 0 lds // (op 856) Load value direct to lds
v_add_co_u32 v40, s[50:51], v40, v36 // (op 856) least significant half
v_addc_co_u32 v41, s[50:51], 0, v41, s[50:51] // (op 856) most significant half

s_add_u32 m0, m0, 2048 // (op 856) 
buffer_load_dwordx4 v40, s[52:55], 0 offen offset: 0 lds // (op 856) Load value direct to lds
v_add_co_u32 v40, s[50:51], v40, v36 // (op 856) least significant half
v_addc_co_u32 v41, s[50:51], 0, v41, s[50:51] // (op 856) most significant half

s_add_u32 m0, m0, 2048 // (op 856) 
buffer_load_dwordx4 v40, s[52:55], 0 offen offset: 0 lds // (op 856) Load value direct to lds



v_mov_b64 v[36:37], s[38:39] // (op 21) convert

v_mov_b32 v16, v22 // (op 21) 
s_lshl_b32 s19, s38, 3 // (op 21) 
v_add_u32 v35, v16, s19 // (op 21) 

buffer_load_ubyte v26, v35, s[68:71], 0 offen offset: 0 // (op 21) Load value
v_add_u32 v35, v35, v36 // (op 21) 
buffer_load_ubyte v27, v35, s[68:71], 0 offen offset: 0 // (op 21) Load value




v_mov_b32 v35, v13 // (op 4) 

buffer_load_dwordx3 v[28:30], v35, s[20:23], 0 offen offset:192 // (op 4) Load value
s_barrier  // (op 1534) 



s_waitcnt vmcnt(10)
ds_write_b8 v17, v24 offset:18176 // (op 236) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v17, v25 offset:18240 // (op 236) Store local data 
s_barrier  // (op 1522) 



s_waitcnt vmcnt(8)
ds_write_b96 v31, v[32:34] offset:18432 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1519) 



ds_read_b128 v[40:43], v4 offset:18432 // (op 227) Load local data 
ds_read_b64 v[44:45], v4 offset:18448 // (op 227) Load local data 
s_barrier  // (op 1525) 
s_barrier  // (op 1565) 



ds_read_b64_tr_b8 v[46:47], v11 offset:8192 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v11 offset:8704 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v11 offset:12288 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v11 offset:12800 // (op 229) Transpose load local data 
s_barrier  // (op 1531) 


ds_read_u8 v36, v10 offset:18176 // (op 233) Load local data 



ds_read_b64_tr_b8 v[54:55], v11 offset:8224 // (op 348) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v11 offset:8736 // (op 348) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:45], v[46:53], a[0:3], v23, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 234) 
ds_read_b64_tr_b8 v[58:59], v11 offset:12320 // (op 348) Transpose load local data 


ds_read_b64_tr_b8 v[60:61], v11 offset:12832 // (op 348) Transpose load local data 


ds_read_u8 v36, v10 offset:18208 // (op 351) Load local data 
s_barrier  // (op 1601) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:45], v[54:61], a[4:7], v23, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 344) 






buffer_load_ubyte v23, v12, s[64:67], 0 offen offset:12 // (op 231) Load value

s_barrier  // (op 1528) 
v_add_u32 v21, 8, v21 // (op 1057) 

s_lshl_b64 s[50:51], s[28:29], 5 // (op 868) 
v_mov_b64 v[36:37], s[50:51] // (op 868) convert


s_lshl_b64 s[50:51], s[28:29], 7 // (op 868) 
s_mul_i32 s73, 3, s51 // (op 868) most significant: low of low * high
s_mul_i32 s72, 3, s50 // (op 868) least significant
s_mul_hi_u32 s19, 3, s50 // (op 868) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 868) most significant: sum


v_mov_b32 v12, s72 // (op 868) 
v_mov_b32 v40, s73 // (op 868) 
v_add_co_u32 v42, s[50:51], v38, v12 // (op 868) least significant half
v_addc_co_u32 v43, s[50:51], v39, v40, s[50:51] // (op 868) most significant half




s_add_u32 s19, 8192, s1 // (op 868) 
s_mov_b32 m0, s19 // (op 868) reg expression
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 868) Load value direct to lds
v_add_co_u32 v42, s[50:51], v42, v36 // (op 868) least significant half
v_addc_co_u32 v43, s[50:51], 0, v43, s[50:51] // (op 868) most significant half

s_add_u32 m0, m0, 2048 // (op 868) 
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 868) Load value direct to lds
v_add_co_u32 v42, s[50:51], v42, v36 // (op 868) least significant half
v_addc_co_u32 v43, s[50:51], 0, v43, s[50:51] // (op 868) most significant half

s_add_u32 m0, m0, 2048 // (op 868) 
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 868) Load value direct to lds
v_add_co_u32 v42, s[50:51], v42, v36 // (op 868) least significant half
v_addc_co_u32 v43, s[50:51], 0, v43, s[50:51] // (op 868) most significant half

s_add_u32 m0, m0, 2048 // (op 868) 
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 868) Load value direct to lds





v_mov_b64 v[36:37], s[38:39] // (op 238) convert

s_lshl_b32 s19, s38, 2 // (op 238) 
s_mul_i32 s49, 3, s19 // (op 238) 

v_add_u32 v12, v16, s49 // (op 238) 

s_lshl_b64 s[50:51], s[28:29], 7 // (op 993) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 993) 

v_mov_b32 v38, s72 // (op 993) 
v_mov_b32 v39, s73 // (op 993) 
v_add_co_u32 v18, s[50:51], v18, v38 // (op 993) least significant half
v_addc_co_u32 v19, s[50:51], v19, v39, s[50:51] // (op 993) most significant half




buffer_load_ubyte v24, v12, s[68:71], 0 offen offset: 0 // (op 238) Load value
v_add_u32 v12, v12, v36 // (op 238) 
buffer_load_ubyte v25, v12, s[68:71], 0 offen offset: 0 // (op 238) Load value






s_lshl_b32 s19, s38, 3 // (op 1121) 
v_add_u32 v22, v22, s19 // (op 1121) 

buffer_load_dwordx3 v[32:34], v35, s[20:23], 0 offen offset:288 // (op 224) Load value

s_barrier  // (op 1508) 
v_add_u32 v13, 192, v13 // (op 929) 



s_waitcnt vmcnt(10)
ds_write_b8 v15, v26 offset:16384 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v15, v27 offset:16448 // (op 26) Store local data 
s_barrier  // (op 1482) 



s_waitcnt vmcnt(8)
ds_write_b96 v20, v[28:30] offset:16640 // (op 6) Store local data 
s_ashr_i64 s[50:51], s[32:33], 7 // (op 307) 
s_mov_b32 s19, s50 // (op 307) convert

s_and_b32 s49, s19, 4294967294 // (op 307) 

s_cmp_lt_u32 s18, s49 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_307 // (op 307) Condition: Bottom (jump to RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_307 if true)
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_307:














s_barrier  // (op 1583) 
s_mul_hi_i32 s0, s16, s11 // (op 891) 
v_lshrrev_b32 v12, 3, v1 // (op 891) 
s_add_i32 s1, s0, s16 // (op 891) 

v_and_b32 v13, -8, v1 // (op 891) 
s_ashr_i32 s0, s1, 31 // (op 891) 
v_sub_u32 v14, v1, v13 // (op 891) 

s_cmp_eq_i32 s11, 0 // (op 891) 
s_mov_b32 s18, scc // (op 891) 
v_lshlrev_b32 v13, 4, v14 // (op 891) 

s_cmp_eq_i32 s18, 0 // (op 891) 
s_cselect_b32 s19, 0, -1 // (op 891) 

s_and_b32 s18, s12, 31 // (op 891) 
s_lshl_b32 s49, 1, s18 // (op 891) 
s_add_i32 s50, s49, s19 // (op 891) 


s_and_b32 s19, s0, s50 // (op 891) 


s_add_i32 s0, s1, s19 // (op 891) 


s_ashr_i32 s1, s0, s18 // (op 891) 


s_lshl_b32 s0, s12, 24 // (op 891) 
s_ashr_i32 s18, s0, 31 // (op 891) 

s_xor_b32 s0, s1, s18 // (op 891) 

s_sub_i32 s1, s0, s18 // (op 891) 


s_mul_hi_i32 s0, s1, s13 // (op 891) 
s_add_i32 s18, s0, s1 // (op 891) 

s_ashr_i32 s0, s18, 31 // (op 891) 
s_cmp_eq_i32 s13, 0 // (op 891) 
s_mov_b32 s19, scc // (op 891) 
s_cmp_eq_i32 s19, 0 // (op 891) 
s_cselect_b32 s49, 0, -1 // (op 891) 

s_and_b32 s19, s14, 31 // (op 891) 
s_lshl_b32 s50, 1, s19 // (op 891) 
s_add_i32 s51, s50, s49 // (op 891) 


s_and_b32 s49, s0, s51 // (op 891) 


s_add_i32 s0, s18, s49 // (op 891) 


s_ashr_i32 s18, s0, s19 // (op 891) 


s_lshl_b32 s0, s14, 24 // (op 891) 
s_ashr_i32 s19, s0, 31 // (op 891) 

s_xor_b32 s0, s18, s19 // (op 891) 

s_sub_i32 s18, s0, s19 // (op 891) 


s_mul_i32 s0, s1, s44 // (op 891) 

s_sub_u32 s1, s16, s0 // (op 891) 

s_mul_i32 s0, s18, s44 // (op 891) 
s_add_u32 s0, s0, s1 // (op 891) 


s_mul_hi_i32 s1, s16, s10 // (op 891) 
s_add_i32 s18, s1, s16 // (op 891) 

s_ashr_i32 s1, s18, 31 // (op 891) 
s_cmp_eq_i32 s10, 0 // (op 891) 
s_mov_b32 s19, scc // (op 891) 
s_cmp_eq_i32 s19, 0 // (op 891) 
s_cselect_b32 s49, 0, -1 // (op 891) 

s_and_b32 s19, s47, 31 // (op 891) 
s_lshl_b32 s50, 1, s19 // (op 891) 
s_add_i32 s51, s50, s49 // (op 891) 


s_and_b32 s49, s1, s51 // (op 891) 


s_add_i32 s1, s18, s49 // (op 891) 


s_ashr_i32 s18, s1, s19 // (op 891) 


s_lshl_b32 s1, s47, 24 // (op 891) 
s_ashr_i32 s19, s1, 31 // (op 891) 

s_xor_b32 s1, s18, s19 // (op 891) 

s_sub_i32 s18, s1, s19 // (op 891) 


s_add_u32 s50, s6, 64 // (op 891) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 891) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 891) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 891) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 891) 

s_mul_i32 s1, s44, s50 // (op 891) 

s_mul_i32 s19, s48, s1 // (op 891) 

s_mul_i32 s1, s18, s19 // (op 891) 

s_sub_u32 s19, s16, s1 // (op 891) 

s_mov_b32 s1, s19 // (op 891) 
s_mul_hi_i32 s49, s1, s15 // (op 891) 
s_add_i32 s50, s49, s1 // (op 891) 


s_ashr_i32 s1, s50, 31 // (op 891) 
s_cmp_eq_i32 s15, 0 // (op 891) 
s_mov_b32 s49, scc // (op 891) 
s_cmp_eq_i32 s49, 0 // (op 891) 
s_cselect_b32 s51, 0, -1 // (op 891) 

s_and_b32 s49, s46, 31 // (op 891) 
s_lshl_b32 s72, 1, s49 // (op 891) 
s_add_i32 s73, s72, s51 // (op 891) 


s_and_b32 s51, s1, s73 // (op 891) 


s_add_i32 s1, s50, s51 // (op 891) 


s_ashr_i32 s50, s1, s49 // (op 891) 


s_lshl_b32 s1, s46, 24 // (op 891) 
s_ashr_i32 s49, s1, 31 // (op 891) 

s_xor_b32 s1, s50, s49 // (op 891) 

s_sub_i32 s50, s1, s49 // (op 891) 


s_mul_i32 s1, s45, s44 // (op 891) 
s_add_u32 s72, s8, 16 // (op 891) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 891) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 891) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 891) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 891) 

s_sub_i32 s49, s72, s1 // (op 891) 

s_mul_i32 s51, s50, s49 // (op 891) 


s_sub_u32 s49, s19, s51 // (op 891) 


s_add_u32 s19, s49, s1 // (op 891) 


s_cmp_eq_i32 s18, 0 // (op 891) 
s_mov_b32 s1, scc // (op 891) 

s_cmp_eq_i32 s1, 0 // (op 891) 
s_cselect_b32 s18, s19, s0 // (op 891) 



v_lshl_add_u32 v14, s18, 4, v12 // (op 891) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 891) 
s_lshr_b32 s18, s0, 1 // (op 891) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 891) 


v_mov_b32 v27, 0 // (op 891) convert
v_mov_b32 v26, v12 // (op 891) convert

v_mul_lo_u32 v13, v14, s25 // (op 891) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 891) most significant: high of low * low
v_mul_lo_u32 v12, v14, s24 // (op 891) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 891) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v26 // (op 891) least significant half
v_addc_co_u32 v13, s[0:1], v13, v27, s[0:1] // (op 891) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 891) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 891) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 891) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 891) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 891) 

v_mov_b32 v14, v12 // (op 891) convert

s_mul_i32 s1, s9, 6 // (op 893) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 893) least significant
s_mul_hi_u32 s18, s8, 6 // (op 893) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 893) most significant: sum

s_mul_i32 s19, s24, s1 // (op 893) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 893) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 893) least significant
s_mul_hi_u32 s50, s24, s0 // (op 893) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 893) most significant: sum
s_add_u32 s19, s19, s50 // (op 893) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 893) 

s_mov_b32 s20, s26 // (op 893) reg expression
s_mov_b32 s21, s27 // (op 893) reg expression
s_mov_b32 s22, s0 // (op 893) reg expression
s_mov_b32 s23, 131072 // (op 893) call()

s_mul_hi_i32 s0, s16, s11 // (op 959) 
v_and_b32 v12, -4, v1 // (op 959) 
s_add_i32 s1, s0, s16 // (op 959) 

v_sub_u32 v13, v1, v12 // (op 959) 

s_ashr_i32 s0, s1, 31 // (op 959) 
v_lshlrev_b32 v12, 4, v13 // (op 959) 

s_cmp_eq_i32 s11, 0 // (op 959) 
s_mov_b32 s18, scc // (op 959) 
v_lshrrev_b32 v13, 2, v1 // (op 959) 
s_cmp_eq_i32 s18, 0 // (op 959) 
s_cselect_b32 s19, 0, -1 // (op 959) 

s_and_b32 s18, s12, 31 // (op 959) 
s_lshl_b32 s49, 1, s18 // (op 959) 
s_add_i32 s50, s49, s19 // (op 959) 


s_and_b32 s19, s0, s50 // (op 959) 


s_add_i32 s0, s1, s19 // (op 959) 


s_ashr_i32 s1, s0, s18 // (op 959) 


s_lshl_b32 s0, s12, 24 // (op 959) 
s_ashr_i32 s18, s0, 31 // (op 959) 

s_xor_b32 s0, s1, s18 // (op 959) 

s_sub_i32 s1, s0, s18 // (op 959) 


s_mul_hi_i32 s0, s1, s13 // (op 959) 
s_add_i32 s18, s0, s1 // (op 959) 

s_ashr_i32 s0, s18, 31 // (op 959) 
s_cmp_eq_i32 s13, 0 // (op 959) 
s_mov_b32 s19, scc // (op 959) 
s_cmp_eq_i32 s19, 0 // (op 959) 
s_cselect_b32 s49, 0, -1 // (op 959) 

s_and_b32 s19, s14, 31 // (op 959) 
s_lshl_b32 s50, 1, s19 // (op 959) 
s_add_i32 s51, s50, s49 // (op 959) 


s_and_b32 s49, s0, s51 // (op 959) 


s_add_i32 s0, s18, s49 // (op 959) 


s_ashr_i32 s18, s0, s19 // (op 959) 


s_lshl_b32 s0, s14, 24 // (op 959) 
s_ashr_i32 s19, s0, 31 // (op 959) 

s_xor_b32 s0, s18, s19 // (op 959) 

s_sub_i32 s18, s0, s19 // (op 959) 


s_add_u32 s50, s6, 64 // (op 959) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 959) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 959) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 959) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 959) 

s_mul_i32 s0, s18, s50 // (op 959) 

s_sub_i32 s49, s1, s0 // (op 959) 


s_mul_hi_i32 s0, s16, s10 // (op 959) 
s_add_i32 s1, s0, s16 // (op 959) 

s_ashr_i32 s0, s1, 31 // (op 959) 
s_cmp_eq_i32 s10, 0 // (op 959) 
s_mov_b32 s18, scc // (op 959) 
s_cmp_eq_i32 s18, 0 // (op 959) 
s_cselect_b32 s19, 0, -1 // (op 959) 

s_and_b32 s18, s47, 31 // (op 959) 
s_lshl_b32 s72, 1, s18 // (op 959) 
s_add_i32 s73, s72, s19 // (op 959) 


s_and_b32 s19, s0, s73 // (op 959) 


s_add_i32 s0, s1, s19 // (op 959) 


s_ashr_i32 s1, s0, s18 // (op 959) 


s_lshl_b32 s0, s47, 24 // (op 959) 
s_ashr_i32 s18, s0, 31 // (op 959) 

s_xor_b32 s0, s1, s18 // (op 959) 

s_sub_i32 s1, s0, s18 // (op 959) 


s_mul_i32 s0, s44, s50 // (op 959) 

s_mul_i32 s18, s48, s0 // (op 959) 

s_mul_i32 s0, s1, s18 // (op 959) 

s_sub_u32 s18, s16, s0 // (op 959) 

s_mul_hi_i32 s0, s18, s15 // (op 959) 
s_add_i32 s19, s0, s18 // (op 959) 


s_ashr_i32 s0, s19, 31 // (op 959) 
s_cmp_eq_i32 s15, 0 // (op 959) 
s_mov_b32 s18, scc // (op 959) 
s_cmp_eq_i32 s18, 0 // (op 959) 
s_cselect_b32 s50, 0, -1 // (op 959) 

s_and_b32 s18, s46, 31 // (op 959) 
s_lshl_b32 s51, 1, s18 // (op 959) 
s_add_i32 s72, s51, s50 // (op 959) 


s_and_b32 s50, s0, s72 // (op 959) 


s_add_i32 s0, s19, s50 // (op 959) 


s_ashr_i32 s19, s0, s18 // (op 959) 


s_lshl_b32 s0, s46, 24 // (op 959) 
s_ashr_i32 s18, s0, 31 // (op 959) 

s_xor_b32 s0, s19, s18 // (op 959) 

s_sub_i32 s19, s0, s18 // (op 959) 


s_cmp_eq_i32 s1, 0 // (op 959) 
s_mov_b32 s0, scc // (op 959) 

s_cmp_eq_i32 s0, 0 // (op 959) 
s_cselect_b32 s1, s19, s49 // (op 959) 



v_lshl_add_u32 v15, s1, 6, v12 // (op 959) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 959) 
v_mov_b32 v17, 0 // (op 959) convert
v_mov_b32 v16, v15 // (op 959) convert

s_lshr_b32 s18, s0, 1 // (op 959) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 959) 


v_mul_lo_u32 v19, v12, s29 // (op 959) most significant: low of low * high
v_mul_hi_u32 v13, v12, s28 // (op 959) most significant: high of low * low
v_mul_lo_u32 v18, v12, s28 // (op 959) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 959) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 959) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 959) most significant half



s_mul_i32 s1, s28, s33 // (op 961) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 961) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 961) least significant
s_mul_hi_u32 s19, s28, s32 // (op 961) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 961) most significant: sum
s_add_u32 s1, s1, s19 // (op 961) most significant: sum


s_mov_b32 s52, s30 // (op 961) reg expression
s_mov_b32 s53, s31 // (op 961) reg expression
s_mov_b32 s54, s0 // (op 961) reg expression
s_mov_b32 s55, 131072 // (op 961) call()

s_mul_hi_i32 s0, s16, s11 // (op 1023) 
v_lshrrev_b32 v12, 7, v1 // (op 1023) 
s_add_i32 s1, s0, s16 // (op 1023) 

v_and_b32 v13, 15, v1 // (op 1023) 
s_ashr_i32 s0, s1, 31 // (op 1023) 
v_lshl_add_u32 v15, v12, 4, v13 // (op 1023) 


s_cmp_eq_i32 s11, 0 // (op 1023) 
s_mov_b32 s18, scc // (op 1023) 
v_and_b32 v12, 63, v1 // (op 1023) 
s_cmp_eq_i32 s18, 0 // (op 1023) 
s_cselect_b32 s19, 0, -1 // (op 1023) 

v_lshrrev_b32 v13, 4, v12 // (op 1023) 

s_and_b32 s18, s12, 31 // (op 1023) 
s_lshl_b32 s49, 1, s18 // (op 1023) 
s_add_i32 s50, s49, s19 // (op 1023) 


s_and_b32 s19, s0, s50 // (op 1023) 


s_add_i32 s0, s1, s19 // (op 1023) 


s_ashr_i32 s1, s0, s18 // (op 1023) 


s_lshl_b32 s0, s12, 24 // (op 1023) 
s_ashr_i32 s18, s0, 31 // (op 1023) 

s_xor_b32 s0, s1, s18 // (op 1023) 

s_sub_i32 s1, s0, s18 // (op 1023) 


s_mul_hi_i32 s0, s1, s13 // (op 1023) 
s_add_i32 s18, s0, s1 // (op 1023) 

s_ashr_i32 s0, s18, 31 // (op 1023) 
s_cmp_eq_i32 s13, 0 // (op 1023) 
s_mov_b32 s19, scc // (op 1023) 
s_cmp_eq_i32 s19, 0 // (op 1023) 
s_cselect_b32 s49, 0, -1 // (op 1023) 

s_and_b32 s19, s14, 31 // (op 1023) 
s_lshl_b32 s50, 1, s19 // (op 1023) 
s_add_i32 s51, s50, s49 // (op 1023) 


s_and_b32 s49, s0, s51 // (op 1023) 


s_add_i32 s0, s18, s49 // (op 1023) 


s_ashr_i32 s18, s0, s19 // (op 1023) 


s_lshl_b32 s0, s14, 24 // (op 1023) 
s_ashr_i32 s19, s0, 31 // (op 1023) 

s_xor_b32 s0, s18, s19 // (op 1023) 

s_sub_i32 s18, s0, s19 // (op 1023) 


s_mul_i32 s0, s1, s44 // (op 1023) 

s_sub_u32 s1, s16, s0 // (op 1023) 

s_mul_i32 s0, s18, s44 // (op 1023) 
s_add_u32 s0, s0, s1 // (op 1023) 


s_mul_hi_i32 s1, s16, s10 // (op 1023) 
s_add_i32 s18, s1, s16 // (op 1023) 

s_ashr_i32 s1, s18, 31 // (op 1023) 
s_cmp_eq_i32 s10, 0 // (op 1023) 
s_mov_b32 s19, scc // (op 1023) 
s_cmp_eq_i32 s19, 0 // (op 1023) 
s_cselect_b32 s49, 0, -1 // (op 1023) 

s_and_b32 s19, s47, 31 // (op 1023) 
s_lshl_b32 s50, 1, s19 // (op 1023) 
s_add_i32 s51, s50, s49 // (op 1023) 


s_and_b32 s49, s1, s51 // (op 1023) 


s_add_i32 s1, s18, s49 // (op 1023) 


s_ashr_i32 s18, s1, s19 // (op 1023) 


s_lshl_b32 s1, s47, 24 // (op 1023) 
s_ashr_i32 s19, s1, 31 // (op 1023) 

s_xor_b32 s1, s18, s19 // (op 1023) 

s_sub_i32 s18, s1, s19 // (op 1023) 


s_add_u32 s50, s6, 64 // (op 1023) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1023) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1023) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1023) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1023) 

s_mul_i32 s1, s44, s50 // (op 1023) 

s_mul_i32 s19, s48, s1 // (op 1023) 

s_mul_i32 s1, s18, s19 // (op 1023) 

s_sub_u32 s19, s16, s1 // (op 1023) 

s_mov_b32 s1, s19 // (op 1023) 
s_mul_hi_i32 s49, s1, s15 // (op 1023) 
s_add_i32 s50, s49, s1 // (op 1023) 


s_ashr_i32 s1, s50, 31 // (op 1023) 
s_cmp_eq_i32 s15, 0 // (op 1023) 
s_mov_b32 s49, scc // (op 1023) 
s_cmp_eq_i32 s49, 0 // (op 1023) 
s_cselect_b32 s51, 0, -1 // (op 1023) 

s_and_b32 s49, s46, 31 // (op 1023) 
s_lshl_b32 s72, 1, s49 // (op 1023) 
s_add_i32 s73, s72, s51 // (op 1023) 


s_and_b32 s51, s1, s73 // (op 1023) 


s_add_i32 s1, s50, s51 // (op 1023) 


s_ashr_i32 s50, s1, s49 // (op 1023) 


s_lshl_b32 s1, s46, 24 // (op 1023) 
s_ashr_i32 s49, s1, 31 // (op 1023) 

s_xor_b32 s1, s50, s49 // (op 1023) 

s_sub_i32 s50, s1, s49 // (op 1023) 


s_mul_i32 s1, s45, s44 // (op 1023) 
s_add_u32 s72, s8, 16 // (op 1023) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 1023) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1023) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1023) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 1023) 

s_sub_i32 s49, s72, s1 // (op 1023) 

s_mul_i32 s51, s50, s49 // (op 1023) 


s_sub_u32 s49, s19, s51 // (op 1023) 


s_add_u32 s19, s49, s1 // (op 1023) 


s_cmp_eq_i32 s18, 0 // (op 1023) 
s_mov_b32 s1, scc // (op 1023) 

s_cmp_eq_i32 s1, 0 // (op 1023) 
s_cselect_b32 s18, s19, s0 // (op 1023) 



v_lshl_add_u32 v12, s18, 4, v15 // (op 1023) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1023) 
s_lshr_b32 s18, s0, 1 // (op 1023) 

v_lshl_add_u32 v15, s18, 3, v13 // (op 1023) 


v_mul_lo_u32 v13, v12, s34 // (op 1023) 
v_add_u32 v13, v13, v15 // (op 1023) 


s_mul_i32 s1, s34, s9 // (op 1025) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1025) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1025) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1025) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1025) most significant: sum
s_add_u32 s1, s1, s19 // (op 1025) most significant: sum


s_mov_b32 s64, s36 // (op 1025) reg expression
s_mov_b32 s65, s37 // (op 1025) reg expression
s_mov_b32 s66, s0 // (op 1025) reg expression
s_mov_b32 s67, 131072 // (op 1025) call()

s_mul_hi_i32 s0, s16, s11 // (op 1087) 
v_and_b32 v12, -64, v1 // (op 1087) 
s_add_i32 s1, s0, s16 // (op 1087) 

v_sub_u32 v15, v1, v12 // (op 1087) 

s_ashr_i32 s0, s1, 31 // (op 1087) 
v_lshrrev_b32 v12, 6, v1 // (op 1087) 
s_cmp_eq_i32 s11, 0 // (op 1087) 
s_mov_b32 s18, scc // (op 1087) 
v_lshlrev_b32 v16, 1, v12 // (op 1087) 

s_cmp_eq_i32 s18, 0 // (op 1087) 
s_cselect_b32 s19, 0, -1 // (op 1087) 

s_and_b32 s18, s12, 31 // (op 1087) 
s_lshl_b32 s49, 1, s18 // (op 1087) 
s_add_i32 s50, s49, s19 // (op 1087) 


s_and_b32 s19, s0, s50 // (op 1087) 


s_add_i32 s0, s1, s19 // (op 1087) 


s_ashr_i32 s1, s0, s18 // (op 1087) 


s_lshl_b32 s0, s12, 24 // (op 1087) 
s_ashr_i32 s18, s0, 31 // (op 1087) 

s_xor_b32 s0, s1, s18 // (op 1087) 

s_sub_i32 s1, s0, s18 // (op 1087) 


s_mul_hi_i32 s0, s1, s13 // (op 1087) 
s_add_i32 s18, s0, s1 // (op 1087) 

s_ashr_i32 s0, s18, 31 // (op 1087) 
s_cmp_eq_i32 s13, 0 // (op 1087) 
s_mov_b32 s19, scc // (op 1087) 
s_cmp_eq_i32 s19, 0 // (op 1087) 
s_cselect_b32 s49, 0, -1 // (op 1087) 

s_and_b32 s19, s14, 31 // (op 1087) 
s_lshl_b32 s50, 1, s19 // (op 1087) 
s_add_i32 s51, s50, s49 // (op 1087) 


s_and_b32 s49, s0, s51 // (op 1087) 


s_add_i32 s0, s18, s49 // (op 1087) 


s_ashr_i32 s18, s0, s19 // (op 1087) 


s_lshl_b32 s0, s14, 24 // (op 1087) 
s_ashr_i32 s19, s0, 31 // (op 1087) 

s_xor_b32 s0, s18, s19 // (op 1087) 

s_sub_i32 s18, s0, s19 // (op 1087) 


s_add_u32 s50, s6, 64 // (op 1087) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1087) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1087) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1087) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1087) 

s_mul_i32 s0, s18, s50 // (op 1087) 

s_sub_i32 s49, s1, s0 // (op 1087) 


s_mul_hi_i32 s0, s16, s10 // (op 1087) 
s_add_i32 s1, s0, s16 // (op 1087) 

s_ashr_i32 s0, s1, 31 // (op 1087) 
s_cmp_eq_i32 s10, 0 // (op 1087) 
s_mov_b32 s18, scc // (op 1087) 
s_cmp_eq_i32 s18, 0 // (op 1087) 
s_cselect_b32 s19, 0, -1 // (op 1087) 

s_and_b32 s18, s47, 31 // (op 1087) 
s_lshl_b32 s72, 1, s18 // (op 1087) 
s_add_i32 s73, s72, s19 // (op 1087) 


s_and_b32 s19, s0, s73 // (op 1087) 


s_add_i32 s0, s1, s19 // (op 1087) 


s_ashr_i32 s1, s0, s18 // (op 1087) 


s_lshl_b32 s0, s47, 24 // (op 1087) 
s_ashr_i32 s18, s0, 31 // (op 1087) 

s_xor_b32 s0, s1, s18 // (op 1087) 

s_sub_i32 s1, s0, s18 // (op 1087) 


s_mul_i32 s0, s44, s50 // (op 1087) 

s_mul_i32 s18, s48, s0 // (op 1087) 

s_mul_i32 s0, s1, s18 // (op 1087) 

s_sub_u32 s18, s16, s0 // (op 1087) 

s_mul_hi_i32 s0, s18, s15 // (op 1087) 
s_add_i32 s19, s0, s18 // (op 1087) 


s_ashr_i32 s0, s19, 31 // (op 1087) 
s_cmp_eq_i32 s15, 0 // (op 1087) 
s_mov_b32 s18, scc // (op 1087) 
s_cmp_eq_i32 s18, 0 // (op 1087) 
s_cselect_b32 s50, 0, -1 // (op 1087) 

s_and_b32 s18, s46, 31 // (op 1087) 
s_lshl_b32 s51, 1, s18 // (op 1087) 
s_add_i32 s72, s51, s50 // (op 1087) 


s_and_b32 s50, s0, s72 // (op 1087) 


s_add_i32 s0, s19, s50 // (op 1087) 


s_ashr_i32 s19, s0, s18 // (op 1087) 


s_lshl_b32 s0, s46, 24 // (op 1087) 
s_ashr_i32 s18, s0, 31 // (op 1087) 

s_xor_b32 s0, s19, s18 // (op 1087) 

s_sub_i32 s19, s0, s18 // (op 1087) 


s_cmp_eq_i32 s1, 0 // (op 1087) 
s_mov_b32 s0, scc // (op 1087) 

s_cmp_eq_i32 s0, 0 // (op 1087) 
s_cselect_b32 s1, s19, s49 // (op 1087) 



v_lshl_add_u32 v12, s1, 6, v15 // (op 1087) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1087) 
s_lshr_b32 s18, s0, 1 // (op 1087) 

v_lshl_add_u32 v15, s18, 3, v16 // (op 1087) 


v_mul_lo_u32 v16, v15, s38 // (op 1087) 
v_add_u32 v16, v16, v12 // (op 1087) 


s_mul_i32 s1, s38, s43 // (op 1089) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1089) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1089) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1089) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1089) most significant: sum
s_add_u32 s1, s1, s19 // (op 1089) most significant: sum


s_mov_b32 s68, s40 // (op 1089) reg expression
s_mov_b32 s69, s41 // (op 1089) reg expression
s_mov_b32 s70, s0 // (op 1089) reg expression
s_mov_b32 s71, 131072 // (op 1089) call()

v_and_b32 v12, -8, v1 // (op 1180) 
v_sub_u32 v15, v1, v12 // (op 1180) 

v_lshlrev_b32 v12, 4, v15 // (op 1180) 

v_lshrrev_b32 v15, 3, v1 // (op 1180) 
v_lshl_add_u32 v17, v15, 7, v12 // (op 1180) 


v_mul_lo_u32 v12, v17, 6 // (op 1180) 

v_lshrrev_b32 v15, 3, v12 // (op 1180) 

v_and_b32 v12, -4, v1 // (op 1233) 
v_sub_u32 v17, v1, v12 // (op 1233) 

v_lshlrev_b32 v12, 4, v17 // (op 1233) 

v_lshrrev_b32 v17, 2, v1 // (op 1233) 
v_lshl_add_u32 v20, v17, 6, v12 // (op 1233) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1233) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1290) 
v_sub_u32 v17, v1, v12 // (op 1290) 

v_lshrrev_b32 v12, 6, v1 // (op 1290) 
v_lshl_add_u32 v20, v12, 7, v17 // (op 1290) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 313) 
s_and_b32 s1, s18, 4294967294 // (op 313) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 312) 
s_mov_b32 s49, s18 // (op 312) convert

s_cmp_lt_u32 s1, s49 // (op 312) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_312 // (op 312) Condition: Top (jump to RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_312 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_312:


s_add_u32 s1, s1, 1 // (op 314) 
v_mov_b32 v12, v14 // (op 166) 

buffer_load_dwordx3 v[22:24], v12, s[20:23], 0 offen // (op 166) Load value

v_add_u32 v14, 96, v14 // (op 881) 
s_barrier  // (op 1485) 
s_barrier  // (op 1498) 




s_lshl_b64 s[18:19], s[28:29], 5 // (op 861) 
v_mov_b64 v[26:27], s[18:19] // (op 861) convert


v_mov_b64 v[28:29], v[18:19] // (op 861) 
v_mov_b64 v[30:31], v[28:29] // (op 861) reg expression
s_mov_b32 m0, s0 // (op 861) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 861) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v26 // (op 861) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 861) most significant half

s_add_u32 m0, m0, 2048 // (op 861) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 861) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v26 // (op 861) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 861) most significant half

s_add_u32 m0, m0, 2048 // (op 861) 
s_waitcnt vmcnt(2)
ds_write_b96 v15, v[22:24] offset:16640 // (op 167) Store local data 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 861) Load value direct to lds

v_add_co_u32 v30, s[18:19], v30, v26 // (op 861) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 861) most significant half

s_add_u32 m0, m0, 2048 // (op 861) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 861) Load value direct to lds



s_lshl_b64 s[18:19], s[28:29], 7 // (op 951) 
v_mov_b32 v12, s18 // (op 951) 
v_mov_b32 v17, s19 // (op 951) 
v_add_co_u32 v18, s[50:51], v18, v12 // (op 951) least significant half
v_addc_co_u32 v19, s[50:51], v19, v17, s[50:51] // (op 951) most significant half





v_mov_b64 v[22:23], s[38:39] // (op 180) convert

v_mov_b32 v12, v16 // (op 180) 
v_mov_b32 v17, v12 // (op 180) reg expression
buffer_load_ubyte v24, v17, s[68:71], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v17, v17, v22 // (op 180) 
buffer_load_ubyte v25, v17, s[68:71], 0 offen offset: 0 // (op 180) Load value



s_lshl_b32 s18, s38, 2 // (op 1079) 
v_add_u32 v16, v16, s18 // (op 1079) 

s_barrier  // (op 1511) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1488) 






ds_read_b128 v[26:29], v3 offset:16640 // (op 169) Load local data 
ds_read_b64 v[30:31], v3 offset:16656 // (op 169) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v24 offset:16384 // (op 178) Store local data 
s_barrier  // (op 1501) 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v25 offset:16448 // (op 178) Store local data 




ds_read_b64_tr_b8 v[32:33], v7 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v7 offset:512 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v7 offset:4096 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v7 offset:4608 // (op 171) Transpose load local data 


v_mov_b32 v12, v13 // (op 173) 

buffer_load_ubyte v17, v12, s[64:67], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v13, 4, v13 // (op 1015) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1515) 


ds_read_u8 v12, v0 offset:16384 // (op 175) Load local data 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[32:39], a[0:3], v17, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 176) 
ds_read_b64_tr_b8 v[40:41], v7 offset:32 // (op 320) Transpose load local data 


ds_read_b64_tr_b8 v[42:43], v7 offset:544 // (op 320) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:4128 // (op 320) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:4640 // (op 320) Transpose load local data 


ds_read_u8 v12, v0 offset:16416 // (op 323) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:31], v[40:47], a[4:7], v17, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 324) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 312) 
s_mov_b32 s49, s18 // (op 312) convert

s_cmp_lt_u32 s1, s49 // (op 312) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_312 // (op 312) Condition: Bottom (jump to RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_312 if true)
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_312:






















s_barrier  // (op 1587) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_123
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 



v_mov_b32 v0, v9 // (op 52) reg expression

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 52) Store value


v_accvgpr_read v10, a4 // (op 504) 
v_accvgpr_read v11, a5 // (op 504) 
v_accvgpr_read v12, a6 // (op 504) 
v_accvgpr_read v13, a7 // (op 504) 
v_mul_f32 v14, v8, v10 // (op 504) 
v_mul_f32 v15, v8, v11 // (op 504) 
v_mul_f32 v16, v8, v12 // (op 504) 
v_mul_f32 v17, v8, v13 // (op 504) 



s_lshl_b32 s0, s4, 7 // (op 506) 
v_add_u32 v0, v9, s0 // (op 506) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 506) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_123:


v_mov_b32 v0, v6 // (op 135) reg expression

buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 135) Load value

v_accvgpr_read v14, a0 // (op 132) 
v_accvgpr_read v15, a1 // (op 132) 
v_accvgpr_read v16, a2 // (op 132) 
v_accvgpr_read v17, a3 // (op 132) 
v_mul_f32 v18, v8, v14 // (op 132) 
v_mul_f32 v19, v8, v15 // (op 132) 
v_mul_f32 v20, v8, v16 // (op 132) 
v_mul_f32 v21, v8, v17 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 136) 
v_mul_f32 v15, v5, v11 // (op 136) 
v_mul_f32 v16, v5, v12 // (op 136) 
v_mul_f32 v17, v5, v13 // (op 136) 

v_add_f32 v10, v14, v18 // (op 133) 
v_add_f32 v11, v15, v19 // (op 133) 
v_add_f32 v12, v16, v20 // (op 133) 
v_add_f32 v13, v17, v21 // (op 133) 




v_mov_b32 v0, v9 // (op 134) reg expression

buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 134) Store value




s_lshl_b32 s0, s2, 7 // (op 528) 
v_add_u32 v0, v6, s0 // (op 528) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 528) Load value

v_accvgpr_read v14, a4 // (op 525) 
v_accvgpr_read v15, a5 // (op 525) 
v_accvgpr_read v16, a6 // (op 525) 
v_accvgpr_read v17, a7 // (op 525) 
v_mul_f32 v18, v8, v14 // (op 525) 
v_mul_f32 v19, v8, v15 // (op 525) 
v_mul_f32 v20, v8, v16 // (op 525) 
v_mul_f32 v21, v8, v17 // (op 525) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 529) 
v_mul_f32 v15, v5, v11 // (op 529) 
v_mul_f32 v16, v5, v12 // (op 529) 
v_mul_f32 v17, v5, v13 // (op 529) 

v_add_f32 v10, v14, v18 // (op 526) 
v_add_f32 v11, v15, v19 // (op 526) 
v_add_f32 v12, v16, v20 // (op 526) 
v_add_f32 v13, v17, v21 // (op 526) 




s_lshl_b32 s0, s4, 7 // (op 527) 
v_add_u32 v0, v9, s0 // (op 527) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 527) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.LRR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__end:
.size RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_, .LRR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__end-RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 19968 // lds bytes
  .amdhsa_accum_offset 64
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
    DataTypeB: F8
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
  - .name: RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 19968
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 62
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

