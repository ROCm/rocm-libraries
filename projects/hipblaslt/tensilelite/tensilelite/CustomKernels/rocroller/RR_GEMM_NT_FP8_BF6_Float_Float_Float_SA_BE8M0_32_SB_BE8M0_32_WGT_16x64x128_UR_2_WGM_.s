// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_,@function
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s12 // (op 1150) 
s_mul_hi_i32 s1, s16, s12 // (op 1338) 
v_and_b32 v0, 48, v1 // (op 1206) 
v_and_b32 v3, 63, v1 // (op 1206) 
v_sub_u32 v4, v3, v0 // (op 1206) 

v_lshrrev_b32 v3, 1, v4 // (op 1206) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 1206) 


v_and_b32 v0, 1, v4 // (op 1206) 

v_lshlrev_b32 v3, 3, v0 // (op 1206) 

v_lshrrev_b32 v0, 7, v1 // (op 1206) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1206) 


v_add_u32 v0, v5, v4 // (op 1206) 


v_and_b32 v3, 48, v1 // (op 1379) 
v_and_b32 v4, 63, v1 // (op 1379) 
v_sub_u32 v5, v4, v3 // (op 1379) 

v_lshrrev_b32 v4, 1, v5 // (op 1379) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 1379) 


v_and_b32 v3, 1, v5 // (op 1379) 

v_lshlrev_b32 v4, 3, v3 // (op 1379) 

v_lshrrev_b32 v3, 7, v1 // (op 1379) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 1379) 


v_add_u32 v3, v6, v5 // (op 1379) 


v_lshrrev_b32 v4, 7, v1 // (op 1150) 
s_add_i32 s18, s0, s16 // (op 1150) 

v_and_b32 v5, 63, v1 // (op 1150) 
s_ashr_i32 s0, s18, 31 // (op 1150) 
v_lshrrev_b32 v6, 4, v5 // (op 1150) 
s_cmp_eq_i32 s12, 0 // (op 1150) 
s_mov_b32 s19, scc // (op 1150) 
v_lshlrev_b32 v7, 2, v6 // (op 1150) 

s_cmp_eq_i32 s19, 0 // (op 1150) 
s_cselect_b32 s51, 0, -1 // (op 1150) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1150) 


s_and_b32 s19, s15, 31 // (op 1150) 
v_lshrrev_b32 v4, 2, v5 // (op 1150) 

v_and_b32 v5, 3, v4 // (op 1150) 

v_and_b32 v4, 63, v1 // (op 1264) 
v_and_b32 v7, 48, v1 // (op 1264) 
v_sub_u32 v8, v4, v7 // (op 1264) 

v_lshrrev_b32 v7, 4, v4 // (op 1264) 

v_lshl_add_u32 v4, v7, 5, v8 // (op 1264) 


v_lshrrev_b32 v7, 6, v1 // (op 1264) 
v_and_b32 v8, 1, v7 // (op 1264) 

v_lshlrev_b32 v7, 4, v8 // (op 1264) 

v_lshl_add_u32 v8, v4, 6, v7 // (op 1264) 


v_mul_lo_u32 v4, v8, 6 // (op 1264) 
v_lshrrev_b32 v7, 3, v4 // (op 1264) 

v_lshrrev_b32 v4, 4, v8 // (op 1264) 

v_lshlrev_b32 v8, 2, v4 // (op 1264) 

v_add_u32 v4, v7, v8 // (op 1264) 


v_lshrrev_b32 v7, 6, v1 // (op 1322) 
v_and_b32 v8, 1, v7 // (op 1322) 

v_and_b32 v7, 15, v1 // (op 1322) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 1322) 


v_and_b32 v7, 63, v1 // (op 1322) 
v_lshrrev_b32 v8, 4, v7 // (op 1322) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 1322) 


v_lshrrev_b32 v8, 7, v1 // (op 1338) 
v_and_b32 v9, 63, v1 // (op 1338) 
v_and_b32 v10, 63, v1 // (op 1408) 
v_and_b32 v11, 48, v1 // (op 1408) 
v_sub_u32 v12, v10, v11 // (op 1408) 

v_lshrrev_b32 v11, 4, v10 // (op 1408) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 1408) 


v_lshrrev_b32 v11, 6, v1 // (op 1408) 
v_and_b32 v12, 1, v11 // (op 1408) 

v_lshlrev_b32 v11, 4, v12 // (op 1408) 

v_lshl_add_u32 v12, v10, 6, v11 // (op 1408) 


v_mul_lo_u32 v10, v12, 6 // (op 1408) 
v_lshrrev_b32 v11, 3, v10 // (op 1408) 

v_lshrrev_b32 v10, 4, v12 // (op 1408) 

v_lshlrev_b32 v12, 2, v10 // (op 1408) 

v_add_u32 v10, v11, v12 // (op 1408) 


v_lshrrev_b32 v11, 6, v1 // (op 1437) 
v_and_b32 v12, 1, v11 // (op 1437) 

v_and_b32 v11, 15, v1 // (op 1437) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 1437) 


v_and_b32 v11, 63, v1 // (op 1437) 
v_lshrrev_b32 v12, 4, v11 // (op 1437) 

v_lshl_add_u32 v11, v12, 6, v13 // (op 1437) 


s_lshl_b32 s52, 1, s19 // (op 1150) 
v_and_b32 v12, 3, v1 // (op 1150) 
s_add_i32 s53, s52, s51 // (op 1150) 


s_and_b32 s51, s0, s53 // (op 1150) 


s_add_i32 s0, s1, s16 // (op 1338) 

s_ashr_i32 s1, s0, 31 // (op 1338) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1150) 


s_add_i32 s52, s18, s51 // (op 1150) 


v_lshrrev_b32 v5, 6, v1 // (op 1150) 
s_ashr_i32 s18, s52, s19 // (op 1150) 


v_and_b32 v12, 1, v5 // (op 1150) 

s_lshl_b32 s19, s15, 24 // (op 1150) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1150) 


s_ashr_i32 s51, s19, 31 // (op 1150) 

s_xor_b32 s19, s18, s51 // (op 1150) 

s_sub_i32 s18, s19, s51 // (op 1150) 


v_lshrrev_b32 v12, 4, v9 // (op 1338) 
s_cmp_eq_i32 s12, 0 // (op 1338) 
s_mov_b32 s19, scc // (op 1338) 
v_lshlrev_b32 v13, 2, v12 // (op 1338) 

s_cmp_eq_i32 s19, 0 // (op 1338) 
s_cselect_b32 s51, 0, -1 // (op 1338) 

v_lshl_add_u32 v12, v8, 4, v13 // (op 1338) 


s_and_b32 s19, s15, 31 // (op 1338) 
v_lshrrev_b32 v8, 2, v9 // (op 1338) 

s_lshl_b32 s52, 1, s19 // (op 1338) 
v_and_b32 v9, 3, v8 // (op 1338) 

s_add_i32 s53, s52, s51 // (op 1338) 


v_and_b32 v8, 3, v1 // (op 1338) 
s_and_b32 s51, s1, s53 // (op 1338) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1338) 


s_add_i32 s1, s0, s51 // (op 1338) 


v_lshrrev_b32 v8, 6, v1 // (op 1338) 
s_ashr_i32 s0, s1, s19 // (op 1338) 


v_and_b32 v9, 1, v8 // (op 1338) 

s_lshl_b32 s1, s15, 24 // (op 1338) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1338) 


s_ashr_i32 s19, s1, 31 // (op 1338) 

s_xor_b32 s1, s0, s19 // (op 1338) 

s_sub_i32 s0, s1, s19 // (op 1338) 



s_waitcnt lgkmcnt(0)
s_mul_hi_i32 s1, s18, s44 // (op 1150) 
s_add_i32 s19, s1, s18 // (op 1150) 

s_ashr_i32 s1, s19, 31 // (op 1150) 
s_cmp_eq_i32 s44, 0 // (op 1150) 
s_mov_b32 s51, scc // (op 1150) 
s_cmp_eq_i32 s51, 0 // (op 1150) 
s_cselect_b32 s52, 0, -1 // (op 1150) 

s_and_b32 s51, s45, 31 // (op 1150) 
s_lshl_b32 s53, 1, s51 // (op 1150) 
s_add_i32 s54, s53, s52 // (op 1150) 


s_and_b32 s52, s1, s54 // (op 1150) 


s_add_i32 s1, s19, s52 // (op 1150) 


s_ashr_i32 s19, s1, s51 // (op 1150) 


s_lshl_b32 s1, s45, 24 // (op 1150) 
s_ashr_i32 s51, s1, 31 // (op 1150) 

s_xor_b32 s1, s19, s51 // (op 1150) 

s_sub_i32 s19, s1, s51 // (op 1150) 


s_mul_i32 s1, s18, s13 // (op 1150) 
s_sub_u32 s51, s16, s1 // (op 1150) 

s_mul_i32 s1, s19, s13 // (op 1150) 
s_add_u32 s1, s1, s51 // (op 1150) 

s_mul_hi_i32 s51, s16, s11 // (op 1150) 
s_add_i32 s52, s51, s16 // (op 1150) 

s_ashr_i32 s51, s52, 31 // (op 1150) 
s_cmp_eq_i32 s11, 0 // (op 1150) 
s_mov_b32 s53, scc // (op 1150) 
s_cmp_eq_i32 s53, 0 // (op 1150) 
s_cselect_b32 s54, 0, -1 // (op 1150) 

s_and_b32 s53, s47, 31 // (op 1150) 
s_lshl_b32 s55, 1, s53 // (op 1150) 
s_add_i32 s56, s55, s54 // (op 1150) 


s_and_b32 s54, s51, s56 // (op 1150) 


s_add_i32 s51, s52, s54 // (op 1150) 


s_ashr_i32 s52, s51, s53 // (op 1150) 


s_lshl_b32 s51, s47, 24 // (op 1150) 
s_ashr_i32 s53, s51, 31 // (op 1150) 

s_xor_b32 s51, s52, s53 // (op 1150) 

s_sub_i32 s52, s51, s53 // (op 1150) 


s_mul_hi_i32 s51, s0, s44 // (op 1338) 
s_add_i32 s53, s51, s0 // (op 1338) 

s_ashr_i32 s51, s53, 31 // (op 1338) 
s_cmp_eq_i32 s44, 0 // (op 1338) 
s_mov_b32 s54, scc // (op 1338) 
s_cmp_eq_i32 s54, 0 // (op 1338) 
s_cselect_b32 s55, 0, -1 // (op 1338) 

s_and_b32 s54, s45, 31 // (op 1338) 
s_lshl_b32 s56, 1, s54 // (op 1338) 
s_add_i32 s57, s56, s55 // (op 1338) 


s_and_b32 s55, s51, s57 // (op 1338) 


s_add_i32 s51, s53, s55 // (op 1338) 


s_ashr_i32 s53, s51, s54 // (op 1338) 


s_lshl_b32 s51, s45, 24 // (op 1338) 
s_ashr_i32 s54, s51, 31 // (op 1338) 

s_xor_b32 s51, s53, s54 // (op 1338) 

s_sub_i32 s53, s51, s54 // (op 1338) 


s_mul_i32 s51, s0, s13 // (op 1338) 
s_sub_u32 s54, s16, s51 // (op 1338) 

s_mul_i32 s51, s53, s13 // (op 1338) 
s_add_u32 s51, s51, s54 // (op 1338) 

s_mul_hi_i32 s54, s16, s11 // (op 1338) 
s_add_i32 s55, s54, s16 // (op 1338) 

s_ashr_i32 s54, s55, 31 // (op 1338) 
s_cmp_eq_i32 s11, 0 // (op 1338) 
s_mov_b32 s56, scc // (op 1338) 
s_cmp_eq_i32 s56, 0 // (op 1338) 
s_cselect_b32 s57, 0, -1 // (op 1338) 

s_and_b32 s56, s47, 31 // (op 1338) 
s_lshl_b32 s58, 1, s56 // (op 1338) 
s_add_i32 s59, s58, s57 // (op 1338) 


s_and_b32 s57, s54, s59 // (op 1338) 


s_add_i32 s54, s55, s57 // (op 1338) 


s_ashr_i32 s55, s54, s56 // (op 1338) 


s_lshl_b32 s54, s47, 24 // (op 1338) 
s_ashr_i32 s56, s54, 31 // (op 1338) 

s_xor_b32 s54, s55, s56 // (op 1338) 

s_sub_i32 s55, s54, s56 // (op 1338) 


s_add_u32 s56, s8, 64 // (op 1150) Start of Int64 add, locking SCC
s_addc_u32 s57, s9, 0 // (op 1150) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1150) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1150) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1150) 

s_mul_i32 s54, s13, s56 // (op 1150) 
s_mul_i32 s58, s10, s54 // (op 1150) 

s_mul_i32 s54, s52, s58 // (op 1150) 

s_sub_u32 s58, s16, s54 // (op 1150) 

s_mov_b32 s54, s58 // (op 1150) 
s_mul_hi_i32 s59, s54, s14 // (op 1150) 
s_add_i32 s60, s59, s54 // (op 1150) 


s_ashr_i32 s54, s60, 31 // (op 1150) 
s_cmp_eq_i32 s14, 0 // (op 1150) 
s_mov_b32 s59, scc // (op 1150) 
s_cmp_eq_i32 s59, 0 // (op 1150) 
s_cselect_b32 s61, 0, -1 // (op 1150) 

s_and_b32 s59, s46, 31 // (op 1150) 
s_lshl_b32 s62, 1, s59 // (op 1150) 
s_add_i32 s63, s62, s61 // (op 1150) 


s_and_b32 s61, s54, s63 // (op 1150) 


s_add_i32 s54, s60, s61 // (op 1150) 


s_ashr_i32 s60, s54, s59 // (op 1150) 


s_lshl_b32 s54, s46, 24 // (op 1150) 
s_ashr_i32 s59, s54, 31 // (op 1150) 

s_xor_b32 s54, s60, s59 // (op 1150) 

s_sub_i32 s60, s54, s59 // (op 1150) 


s_mul_i32 s54, s48, s13 // (op 1150) 
s_add_u32 s62, s6, 16 // (op 1150) Start of Int64 add, locking SCC
s_addc_u32 s63, s7, 0 // (op 1150) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1150) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1150) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 1150) 

s_sub_i32 s59, s62, s54 // (op 1150) 

s_mul_i32 s61, s60, s59 // (op 1150) 

s_sub_u32 s59, s58, s61 // (op 1150) 


s_add_u32 s58, s59, s54 // (op 1150) 


s_cmp_eq_i32 s52, 0 // (op 1150) 
s_mov_b32 s54, scc // (op 1150) 

s_cmp_eq_i32 s54, 0 // (op 1150) 
s_cselect_b32 s52, s58, s1 // (op 1150) 


s_mul_i32 s1, s19, s56 // (op 1150) 


v_lshl_add_u32 v9, s52, 4, v6 // (op 1150) 


s_sub_i32 s19, s18, s1 // (op 1150) 


s_cmp_eq_i32 s54, 0 // (op 1150) 
s_cselect_b32 s1, s60, s19 // (op 1150) 



v_lshl_add_u32 v6, s1, 6, v5 // (op 1150) 


v_mul_lo_u32 v5, v6, s2 // (op 1150) 

v_add_lshl_u32 v6, v9, v5, 2 // (op 1150) 


s_add_u32 s18, s8, 64 // (op 1338) Start of Int64 add, locking SCC
s_addc_u32 s19, s9, 0 // (op 1338) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 1338) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 1338) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 6 // (op 1338) 

s_mul_i32 s1, s13, s18 // (op 1338) 
s_mul_i32 s52, s10, s1 // (op 1338) 

s_mul_i32 s1, s55, s52 // (op 1338) 

s_sub_u32 s52, s16, s1 // (op 1338) 

s_mov_b32 s1, s52 // (op 1338) 
s_mul_hi_i32 s54, s1, s14 // (op 1338) 
s_add_i32 s56, s54, s1 // (op 1338) 


s_ashr_i32 s1, s56, 31 // (op 1338) 
s_cmp_eq_i32 s14, 0 // (op 1338) 
s_mov_b32 s54, scc // (op 1338) 
s_cmp_eq_i32 s54, 0 // (op 1338) 
s_cselect_b32 s57, 0, -1 // (op 1338) 

s_and_b32 s54, s46, 31 // (op 1338) 
s_lshl_b32 s58, 1, s54 // (op 1338) 
s_add_i32 s59, s58, s57 // (op 1338) 


s_and_b32 s57, s1, s59 // (op 1338) 


s_add_i32 s1, s56, s57 // (op 1338) 


s_ashr_i32 s56, s1, s54 // (op 1338) 


s_lshl_b32 s1, s46, 24 // (op 1338) 
s_ashr_i32 s54, s1, 31 // (op 1338) 

s_xor_b32 s1, s56, s54 // (op 1338) 

s_sub_i32 s56, s1, s54 // (op 1338) 


s_mul_i32 s1, s48, s13 // (op 1338) 
s_add_u32 s58, s6, 16 // (op 1338) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 1338) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1338) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1338) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1338) 

s_sub_i32 s54, s58, s1 // (op 1338) 

s_mul_i32 s57, s56, s54 // (op 1338) 

s_sub_u32 s54, s52, s57 // (op 1338) 


s_add_u32 s52, s54, s1 // (op 1338) 


s_cmp_eq_i32 s55, 0 // (op 1338) 
s_mov_b32 s1, scc // (op 1338) 

s_cmp_eq_i32 s1, 0 // (op 1338) 
s_cselect_b32 s54, s52, s51 // (op 1338) 


s_mul_i32 s55, s53, s18 // (op 1338) 


v_lshl_add_u32 v5, s54, 4, v12 // (op 1338) 


s_sub_i32 s18, s0, s55 // (op 1338) 


s_cmp_eq_i32 s1, 0 // (op 1338) 
s_cselect_b32 s0, s56, s18 // (op 1338) 



v_lshl_add_u32 v9, s0, 6, v8 // (op 1338) 


v_mul_lo_u32 v8, v9, s4 // (op 1338) 

v_add_lshl_u32 v9, v5, v8, 2 // (op 1338) 


s_mul_i32 s1, s2, s9 // (op 1152) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1152) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1152) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1152) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1152) most significant: sum
s_add_u32 s1, s1, s19 // (op 1152) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1152) 

s_mul_i32 s1, s4, s9 // (op 1340) most significant: low of low * high
s_mul_i32 s51, s5, s8 // (op 1340) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1340) least significant
s_mul_hi_u32 s52, s4, s8 // (op 1340) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1340) most significant: sum
s_add_u32 s1, s1, s52 // (op 1340) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 1340) 

s_mov_b32 s56, s20 // (op 1152) reg expression
s_mov_b32 s57, s21 // (op 1152) reg expression
s_mov_b32 s58, s18 // (op 1152) reg expression
s_mov_b32 s59, 131072 // (op 1152) call()

s_mov_b32 s60, s22 // (op 1340) reg expression

s_mov_b32 s61, s23 // (op 1340) reg expression
s_mov_b32 s62, s52 // (op 1340) reg expression
s_mov_b32 s63, 131072 // (op 1340) call()


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
s_mul_hi_i32 s0, s16, s12 // (op 917) 
s_add_i32 s1, s0, s16 // (op 917) 

s_ashr_i32 s0, s1, 31 // (op 917) 
s_cmp_eq_i32 s12, 0 // (op 917) 
s_mov_b32 s18, scc // (op 917) 
s_cmp_eq_i32 s18, 0 // (op 917) 
s_cselect_b32 s19, 0, -1 // (op 917) 

s_and_b32 s18, s15, 31 // (op 917) 
s_lshl_b32 s20, 1, s18 // (op 917) 
s_add_i32 s21, s20, s19 // (op 917) 


s_and_b32 s19, s0, s21 // (op 917) 


s_add_i32 s0, s1, s19 // (op 917) 


s_ashr_i32 s1, s0, s18 // (op 917) 


s_lshl_b32 s0, s15, 24 // (op 917) 
s_ashr_i32 s18, s0, 31 // (op 917) 

s_xor_b32 s0, s1, s18 // (op 917) 

s_sub_i32 s1, s0, s18 // (op 917) 


s_mul_hi_i32 s0, s1, s44 // (op 917) 
s_add_i32 s18, s0, s1 // (op 917) 

s_ashr_i32 s0, s18, 31 // (op 917) 
s_cmp_eq_i32 s44, 0 // (op 917) 
s_mov_b32 s19, scc // (op 917) 
s_cmp_eq_i32 s19, 0 // (op 917) 
s_cselect_b32 s20, 0, -1 // (op 917) 

s_and_b32 s19, s45, 31 // (op 917) 
s_lshl_b32 s21, 1, s19 // (op 917) 
s_add_i32 s22, s21, s20 // (op 917) 


s_and_b32 s23, s0, s22 // (op 917) 


s_add_i32 s0, s18, s23 // (op 917) 


s_ashr_i32 s18, s0, s19 // (op 917) 


s_lshl_b32 s0, s45, 24 // (op 917) 
s_ashr_i32 s19, s0, 31 // (op 917) 

s_xor_b32 s0, s18, s19 // (op 917) 

s_sub_i32 s18, s0, s19 // (op 917) 


s_mul_i32 s0, s1, s13 // (op 917) 

s_sub_u32 s1, s16, s0 // (op 917) 

s_mul_i32 s0, s18, s13 // (op 917) 
s_add_u32 s0, s0, s1 // (op 917) 


s_mul_hi_i32 s1, s16, s11 // (op 917) 
s_add_i32 s18, s1, s16 // (op 917) 

s_ashr_i32 s1, s18, 31 // (op 917) 
s_cmp_eq_i32 s11, 0 // (op 917) 
s_mov_b32 s19, scc // (op 917) 
s_cmp_eq_i32 s19, 0 // (op 917) 
s_cselect_b32 s20, 0, -1 // (op 917) 

s_and_b32 s19, s47, 31 // (op 917) 
s_lshl_b32 s21, 1, s19 // (op 917) 
s_add_i32 s22, s21, s20 // (op 917) 


s_and_b32 s23, s1, s22 // (op 917) 


s_add_i32 s1, s18, s23 // (op 917) 


s_ashr_i32 s18, s1, s19 // (op 917) 


s_lshl_b32 s1, s47, 24 // (op 917) 
s_ashr_i32 s19, s1, 31 // (op 917) 

s_xor_b32 s1, s18, s19 // (op 917) 

s_sub_i32 s18, s1, s19 // (op 917) 


s_add_u32 s22, s8, 64 // (op 917) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 917) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 917) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 917) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 6 // (op 917) 

s_mul_i32 s1, s13, s22 // (op 917) 

s_mul_i32 s19, s10, s1 // (op 917) 

s_mul_i32 s1, s18, s19 // (op 917) 

s_sub_u32 s19, s16, s1 // (op 917) 

s_mov_b32 s1, s19 // (op 917) 
s_mul_hi_i32 s20, s1, s14 // (op 917) 
s_add_i32 s21, s20, s1 // (op 917) 


s_ashr_i32 s1, s21, 31 // (op 917) 
s_cmp_eq_i32 s14, 0 // (op 917) 
s_mov_b32 s20, scc // (op 917) 
s_cmp_eq_i32 s20, 0 // (op 917) 
s_cselect_b32 s22, 0, -1 // (op 917) 

s_and_b32 s20, s46, 31 // (op 917) 
s_lshl_b32 s23, 1, s20 // (op 917) 
s_add_i32 s49, s23, s22 // (op 917) 


s_and_b32 s22, s1, s49 // (op 917) 


s_add_i32 s1, s21, s22 // (op 917) 


s_ashr_i32 s21, s1, s20 // (op 917) 


s_lshl_b32 s1, s46, 24 // (op 917) 
s_ashr_i32 s20, s1, 31 // (op 917) 

s_xor_b32 s1, s21, s20 // (op 917) 

s_sub_i32 s21, s1, s20 // (op 917) 


s_mul_i32 s1, s48, s13 // (op 917) 
s_add_u32 s22, s6, 16 // (op 917) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 917) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 917) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 917) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 4 // (op 917) 

s_sub_i32 s20, s22, s1 // (op 917) 

s_mul_i32 s22, s21, s20 // (op 917) 


s_sub_u32 s23, s19, s22 // (op 917) 


s_add_u32 s19, s23, s1 // (op 917) 


s_cmp_eq_i32 s18, 0 // (op 917) 
s_mov_b32 s1, scc // (op 917) 

s_cmp_eq_i32 s1, 0 // (op 917) 
s_cselect_b32 s18, s19, s0 // (op 917) 



s_lshl_b32 s0, s18, 4 // (op 917) 

s_mov_b32 s19, 0 // (op 917) convert
s_mov_b32 s18, s0 // (op 917) convert

v_mul_lo_u32 v13, v1, s25 // (op 917) most significant: low of low * high
v_mul_hi_u32 v14, v1, s24 // (op 917) most significant: high of low * low
v_mul_lo_u32 v12, v1, s24 // (op 917) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 917) most significant: sum

v_mov_b32 v14, s18 // (op 917) 
v_mov_b32 v15, s19 // (op 917) 
v_add_co_u32 v16, s[0:1], v14, v12 // (op 917) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 917) most significant half





s_mul_i32 s1, s24, s29 // (op 919) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 919) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 919) least significant
s_mul_hi_u32 s19, s24, s28 // (op 919) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 919) most significant: sum
s_add_u32 s1, s1, s19 // (op 919) most significant: sum


s_mov_b32 s20, s26 // (op 919) reg expression
s_mov_b32 s21, s27 // (op 919) reg expression
s_mov_b32 s22, s0 // (op 919) reg expression
s_mov_b32 s23, 131072 // (op 919) call()

s_mul_hi_i32 s0, s16, s12 // (op 936) 
s_add_i32 s1, s0, s16 // (op 936) 

s_ashr_i32 s0, s1, 31 // (op 936) 
s_cmp_eq_i32 s12, 0 // (op 936) 
s_mov_b32 s18, scc // (op 936) 
s_cmp_eq_i32 s18, 0 // (op 936) 
s_cselect_b32 s19, 0, -1 // (op 936) 

s_and_b32 s18, s15, 31 // (op 936) 
s_lshl_b32 s49, 1, s18 // (op 936) 
s_add_i32 s50, s49, s19 // (op 936) 


s_and_b32 s19, s0, s50 // (op 936) 


s_add_i32 s0, s1, s19 // (op 936) 


s_ashr_i32 s1, s0, s18 // (op 936) 


s_lshl_b32 s0, s15, 24 // (op 936) 
s_ashr_i32 s18, s0, 31 // (op 936) 

s_xor_b32 s0, s1, s18 // (op 936) 

s_sub_i32 s1, s0, s18 // (op 936) 


s_mul_hi_i32 s0, s1, s44 // (op 936) 
s_add_i32 s18, s0, s1 // (op 936) 

s_ashr_i32 s0, s18, 31 // (op 936) 
s_cmp_eq_i32 s44, 0 // (op 936) 
s_mov_b32 s19, scc // (op 936) 
s_cmp_eq_i32 s19, 0 // (op 936) 
s_cselect_b32 s49, 0, -1 // (op 936) 

s_and_b32 s19, s45, 31 // (op 936) 
s_lshl_b32 s50, 1, s19 // (op 936) 
s_add_i32 s51, s50, s49 // (op 936) 


s_and_b32 s49, s0, s51 // (op 936) 


s_add_i32 s0, s18, s49 // (op 936) 


s_ashr_i32 s18, s0, s19 // (op 936) 


s_lshl_b32 s0, s45, 24 // (op 936) 
s_ashr_i32 s19, s0, 31 // (op 936) 

s_xor_b32 s0, s18, s19 // (op 936) 

s_sub_i32 s18, s0, s19 // (op 936) 


s_mul_i32 s0, s1, s13 // (op 936) 

s_sub_u32 s1, s16, s0 // (op 936) 

s_mul_i32 s0, s18, s13 // (op 936) 
s_add_u32 s0, s0, s1 // (op 936) 


s_mul_hi_i32 s1, s16, s11 // (op 936) 
s_add_i32 s18, s1, s16 // (op 936) 

s_ashr_i32 s1, s18, 31 // (op 936) 
s_cmp_eq_i32 s11, 0 // (op 936) 
s_mov_b32 s19, scc // (op 936) 
s_cmp_eq_i32 s19, 0 // (op 936) 
s_cselect_b32 s49, 0, -1 // (op 936) 

s_and_b32 s19, s47, 31 // (op 936) 
s_lshl_b32 s50, 1, s19 // (op 936) 
s_add_i32 s51, s50, s49 // (op 936) 


s_and_b32 s49, s1, s51 // (op 936) 


s_add_i32 s1, s18, s49 // (op 936) 


s_ashr_i32 s18, s1, s19 // (op 936) 


s_lshl_b32 s1, s47, 24 // (op 936) 
s_ashr_i32 s19, s1, 31 // (op 936) 

s_xor_b32 s1, s18, s19 // (op 936) 

s_sub_i32 s18, s1, s19 // (op 936) 


s_add_u32 s54, s8, 64 // (op 936) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 936) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 936) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 936) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 936) 

s_mul_i32 s1, s13, s50 // (op 936) 

s_mul_i32 s19, s10, s1 // (op 936) 

s_mul_i32 s1, s18, s19 // (op 936) 

s_sub_u32 s19, s16, s1 // (op 936) 

s_mov_b32 s1, s19 // (op 936) 
s_mul_hi_i32 s49, s1, s14 // (op 936) 
s_add_i32 s50, s49, s1 // (op 936) 


s_ashr_i32 s1, s50, 31 // (op 936) 
s_cmp_eq_i32 s14, 0 // (op 936) 
s_mov_b32 s49, scc // (op 936) 
s_cmp_eq_i32 s49, 0 // (op 936) 
s_cselect_b32 s51, 0, -1 // (op 936) 

s_and_b32 s49, s46, 31 // (op 936) 
s_lshl_b32 s52, 1, s49 // (op 936) 
s_add_i32 s53, s52, s51 // (op 936) 


s_and_b32 s51, s1, s53 // (op 936) 


s_add_i32 s1, s50, s51 // (op 936) 


s_ashr_i32 s50, s1, s49 // (op 936) 


s_lshl_b32 s1, s46, 24 // (op 936) 
s_ashr_i32 s49, s1, 31 // (op 936) 

s_xor_b32 s1, s50, s49 // (op 936) 

s_sub_i32 s50, s1, s49 // (op 936) 


s_mul_i32 s1, s48, s13 // (op 936) 
s_add_u32 s54, s6, 16 // (op 936) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 936) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 936) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 936) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 4 // (op 936) 

s_sub_i32 s49, s54, s1 // (op 936) 

s_mul_i32 s51, s50, s49 // (op 936) 


s_sub_u32 s49, s19, s51 // (op 936) 


s_add_u32 s19, s49, s1 // (op 936) 


s_cmp_eq_i32 s18, 0 // (op 936) 
s_mov_b32 s1, scc // (op 936) 

s_cmp_eq_i32 s1, 0 // (op 936) 
s_cselect_b32 s18, s19, s0 // (op 936) 



s_lshl_b32 s19, s18, 4 // (op 936) 

s_mov_b32 s1, 0 // (op 936) convert
s_mov_b32 s0, s19 // (op 936) convert

v_mul_lo_u32 v13, v1, s25 // (op 936) most significant: low of low * high
v_mul_hi_u32 v14, v1, s24 // (op 936) most significant: high of low * low
v_mul_lo_u32 v12, v1, s24 // (op 936) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 936) most significant: sum

v_mov_b32 v14, s0 // (op 936) 
v_mov_b32 v15, s1 // (op 936) 
v_add_co_u32 v18, s[18:19], v14, v12 // (op 936) least significant half
v_addc_co_u32 v19, s[18:19], v15, v13, s[18:19] // (op 936) most significant half





s_mul_i32 s1, s24, s29 // (op 938) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 938) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 938) least significant
s_mul_hi_u32 s19, s24, s28 // (op 938) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 938) most significant: sum
s_add_u32 s1, s1, s19 // (op 938) most significant: sum


s_mov_b32 s20, s26 // (op 938) reg expression
s_mov_b32 s21, s27 // (op 938) reg expression
s_mov_b32 s22, s0 // (op 938) reg expression
s_mov_b32 s23, 131072 // (op 938) call()

s_mul_hi_i32 s0, s16, s12 // (op 981) 
s_add_i32 s1, s0, s16 // (op 981) 

s_ashr_i32 s0, s1, 31 // (op 981) 
s_cmp_eq_i32 s12, 0 // (op 981) 
s_mov_b32 s18, scc // (op 981) 
s_cmp_eq_i32 s18, 0 // (op 981) 
s_cselect_b32 s19, 0, -1 // (op 981) 

s_and_b32 s18, s15, 31 // (op 981) 
s_lshl_b32 s49, 1, s18 // (op 981) 
s_add_i32 s50, s49, s19 // (op 981) 


s_and_b32 s19, s0, s50 // (op 981) 


s_add_i32 s0, s1, s19 // (op 981) 


s_ashr_i32 s1, s0, s18 // (op 981) 


s_lshl_b32 s0, s15, 24 // (op 981) 
s_ashr_i32 s18, s0, 31 // (op 981) 

s_xor_b32 s0, s1, s18 // (op 981) 

s_sub_i32 s1, s0, s18 // (op 981) 


s_mul_hi_i32 s0, s1, s44 // (op 981) 
s_add_i32 s18, s0, s1 // (op 981) 

s_ashr_i32 s0, s18, 31 // (op 981) 
s_cmp_eq_i32 s44, 0 // (op 981) 
s_mov_b32 s19, scc // (op 981) 
s_cmp_eq_i32 s19, 0 // (op 981) 
s_cselect_b32 s49, 0, -1 // (op 981) 

s_and_b32 s19, s45, 31 // (op 981) 
s_lshl_b32 s50, 1, s19 // (op 981) 
s_add_i32 s51, s50, s49 // (op 981) 


s_and_b32 s49, s0, s51 // (op 981) 


s_add_i32 s0, s18, s49 // (op 981) 


s_ashr_i32 s18, s0, s19 // (op 981) 


s_lshl_b32 s0, s45, 24 // (op 981) 
s_ashr_i32 s19, s0, 31 // (op 981) 

s_xor_b32 s0, s18, s19 // (op 981) 

s_sub_i32 s18, s0, s19 // (op 981) 


s_add_u32 s54, s8, 64 // (op 981) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 981) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 981) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 981) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 981) 

s_mul_i32 s0, s18, s50 // (op 981) 

s_sub_i32 s49, s1, s0 // (op 981) 


s_mul_hi_i32 s0, s16, s11 // (op 981) 
s_add_i32 s1, s0, s16 // (op 981) 

s_ashr_i32 s0, s1, 31 // (op 981) 
s_cmp_eq_i32 s11, 0 // (op 981) 
s_mov_b32 s18, scc // (op 981) 
s_cmp_eq_i32 s18, 0 // (op 981) 
s_cselect_b32 s19, 0, -1 // (op 981) 

s_and_b32 s18, s47, 31 // (op 981) 
s_lshl_b32 s52, 1, s18 // (op 981) 
s_add_i32 s53, s52, s19 // (op 981) 


s_and_b32 s19, s0, s53 // (op 981) 


s_add_i32 s0, s1, s19 // (op 981) 


s_ashr_i32 s1, s0, s18 // (op 981) 


s_lshl_b32 s0, s47, 24 // (op 981) 
s_ashr_i32 s18, s0, 31 // (op 981) 

s_xor_b32 s0, s1, s18 // (op 981) 

s_sub_i32 s1, s0, s18 // (op 981) 


s_mul_i32 s0, s13, s50 // (op 981) 

s_mul_i32 s18, s10, s0 // (op 981) 

s_mul_i32 s0, s1, s18 // (op 981) 

s_sub_u32 s18, s16, s0 // (op 981) 

s_mul_hi_i32 s0, s18, s14 // (op 981) 
s_add_i32 s19, s0, s18 // (op 981) 


s_ashr_i32 s0, s19, 31 // (op 981) 
s_cmp_eq_i32 s14, 0 // (op 981) 
s_mov_b32 s18, scc // (op 981) 
s_cmp_eq_i32 s18, 0 // (op 981) 
s_cselect_b32 s50, 0, -1 // (op 981) 

s_and_b32 s18, s46, 31 // (op 981) 
s_lshl_b32 s51, 1, s18 // (op 981) 
s_add_i32 s52, s51, s50 // (op 981) 


s_and_b32 s50, s0, s52 // (op 981) 


s_add_i32 s0, s19, s50 // (op 981) 


s_ashr_i32 s19, s0, s18 // (op 981) 


s_lshl_b32 s0, s46, 24 // (op 981) 
s_ashr_i32 s18, s0, 31 // (op 981) 

s_xor_b32 s0, s19, s18 // (op 981) 

s_sub_i32 s19, s0, s18 // (op 981) 


s_cmp_eq_i32 s1, 0 // (op 981) 
s_mov_b32 s0, scc // (op 981) 

s_cmp_eq_i32 s0, 0 // (op 981) 
s_cselect_b32 s1, s19, s49 // (op 981) 



s_lshl_b32 s0, s1, 6 // (op 981) 

s_mov_b32 s19, 0 // (op 981) convert
s_mov_b32 s18, s0 // (op 981) convert

v_mul_lo_u32 v13, v1, s31 // (op 981) most significant: low of low * high
v_mul_hi_u32 v14, v1, s30 // (op 981) most significant: high of low * low
v_mul_lo_u32 v12, v1, s30 // (op 981) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 981) most significant: sum

v_mov_b32 v14, s18 // (op 981) 
v_mov_b32 v15, s19 // (op 981) 
v_add_co_u32 v12, s[0:1], v12, v14 // (op 981) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 981) most significant half




v_mul_lo_u32 v15, v13, 6 // (op 981) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 981) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 981) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 981) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 981) 

v_mov_b32 v14, v12 // (op 981) convert

s_mul_i32 s1, s29, 6 // (op 983) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 983) least significant
s_mul_hi_u32 s18, s28, 6 // (op 983) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 983) most significant: sum

s_mul_i32 s19, s30, s1 // (op 983) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 983) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 983) least significant
s_mul_hi_u32 s50, s30, s0 // (op 983) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 983) most significant: sum
s_add_u32 s19, s19, s50 // (op 983) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 983) 

s_mov_b32 s52, s32 // (op 983) reg expression
s_mov_b32 s53, s33 // (op 983) reg expression
s_mov_b32 s54, s0 // (op 983) reg expression
s_mov_b32 s55, 131072 // (op 983) call()

s_mul_hi_i32 s0, s16, s12 // (op 1000) 
s_add_i32 s1, s0, s16 // (op 1000) 

s_ashr_i32 s0, s1, 31 // (op 1000) 
s_cmp_eq_i32 s12, 0 // (op 1000) 
s_mov_b32 s18, scc // (op 1000) 
s_cmp_eq_i32 s18, 0 // (op 1000) 
s_cselect_b32 s19, 0, -1 // (op 1000) 

s_and_b32 s18, s15, 31 // (op 1000) 
s_lshl_b32 s49, 1, s18 // (op 1000) 
s_add_i32 s50, s49, s19 // (op 1000) 


s_and_b32 s19, s0, s50 // (op 1000) 


s_add_i32 s0, s1, s19 // (op 1000) 


s_ashr_i32 s1, s0, s18 // (op 1000) 


s_lshl_b32 s0, s15, 24 // (op 1000) 
s_ashr_i32 s18, s0, 31 // (op 1000) 

s_xor_b32 s0, s1, s18 // (op 1000) 

s_sub_i32 s1, s0, s18 // (op 1000) 


s_mul_hi_i32 s0, s1, s44 // (op 1000) 
s_add_i32 s18, s0, s1 // (op 1000) 

s_ashr_i32 s0, s18, 31 // (op 1000) 
s_cmp_eq_i32 s44, 0 // (op 1000) 
s_mov_b32 s19, scc // (op 1000) 
s_cmp_eq_i32 s19, 0 // (op 1000) 
s_cselect_b32 s49, 0, -1 // (op 1000) 

s_and_b32 s19, s45, 31 // (op 1000) 
s_lshl_b32 s50, 1, s19 // (op 1000) 
s_add_i32 s51, s50, s49 // (op 1000) 


s_and_b32 s49, s0, s51 // (op 1000) 


s_add_i32 s0, s18, s49 // (op 1000) 


s_ashr_i32 s18, s0, s19 // (op 1000) 


s_lshl_b32 s0, s45, 24 // (op 1000) 
s_ashr_i32 s19, s0, 31 // (op 1000) 

s_xor_b32 s0, s18, s19 // (op 1000) 

s_sub_i32 s18, s0, s19 // (op 1000) 


s_add_u32 s50, s8, 64 // (op 1000) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1000) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1000) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1000) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1000) 

s_mul_i32 s0, s18, s50 // (op 1000) 

s_sub_i32 s49, s1, s0 // (op 1000) 


s_mul_hi_i32 s0, s16, s11 // (op 1000) 
s_add_i32 s1, s0, s16 // (op 1000) 

s_ashr_i32 s0, s1, 31 // (op 1000) 
s_cmp_eq_i32 s11, 0 // (op 1000) 
s_mov_b32 s18, scc // (op 1000) 
s_cmp_eq_i32 s18, 0 // (op 1000) 
s_cselect_b32 s19, 0, -1 // (op 1000) 

s_and_b32 s18, s47, 31 // (op 1000) 
s_lshl_b32 s64, 1, s18 // (op 1000) 
s_add_i32 s65, s64, s19 // (op 1000) 


s_and_b32 s19, s0, s65 // (op 1000) 


s_add_i32 s0, s1, s19 // (op 1000) 


s_ashr_i32 s1, s0, s18 // (op 1000) 


s_lshl_b32 s0, s47, 24 // (op 1000) 
s_ashr_i32 s18, s0, 31 // (op 1000) 

s_xor_b32 s0, s1, s18 // (op 1000) 

s_sub_i32 s1, s0, s18 // (op 1000) 


s_mul_i32 s0, s13, s50 // (op 1000) 

s_mul_i32 s18, s10, s0 // (op 1000) 

s_mul_i32 s0, s1, s18 // (op 1000) 

s_sub_u32 s18, s16, s0 // (op 1000) 

s_mul_hi_i32 s0, s18, s14 // (op 1000) 
s_add_i32 s19, s0, s18 // (op 1000) 


s_ashr_i32 s0, s19, 31 // (op 1000) 
s_cmp_eq_i32 s14, 0 // (op 1000) 
s_mov_b32 s18, scc // (op 1000) 
s_cmp_eq_i32 s18, 0 // (op 1000) 
s_cselect_b32 s50, 0, -1 // (op 1000) 

s_and_b32 s18, s46, 31 // (op 1000) 
s_lshl_b32 s51, 1, s18 // (op 1000) 
s_add_i32 s64, s51, s50 // (op 1000) 


s_and_b32 s50, s0, s64 // (op 1000) 


s_add_i32 s0, s19, s50 // (op 1000) 


s_ashr_i32 s19, s0, s18 // (op 1000) 


s_lshl_b32 s0, s46, 24 // (op 1000) 
s_ashr_i32 s18, s0, 31 // (op 1000) 

s_xor_b32 s0, s19, s18 // (op 1000) 

s_sub_i32 s19, s0, s18 // (op 1000) 


s_cmp_eq_i32 s1, 0 // (op 1000) 
s_mov_b32 s0, scc // (op 1000) 

s_cmp_eq_i32 s0, 0 // (op 1000) 
s_cselect_b32 s1, s19, s49 // (op 1000) 



s_lshl_b32 s0, s1, 6 // (op 1000) 

s_mov_b32 s19, 0 // (op 1000) convert
s_mov_b32 s18, s0 // (op 1000) convert

v_mul_lo_u32 v13, v1, s31 // (op 1000) most significant: low of low * high
v_mul_hi_u32 v15, v1, s30 // (op 1000) most significant: high of low * low
v_mul_lo_u32 v12, v1, s30 // (op 1000) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 1000) most significant: sum

v_mov_b32 v15, s18 // (op 1000) 
v_mov_b32 v20, s19 // (op 1000) 
v_add_co_u32 v12, s[0:1], v12, v15 // (op 1000) least significant half
v_addc_co_u32 v13, s[0:1], v13, v20, s[0:1] // (op 1000) most significant half




v_mul_lo_u32 v21, v13, 6 // (op 1000) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 1000) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 1000) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1000) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 1000) 

v_mov_b32 v15, v12 // (op 1000) convert

s_mul_i32 s1, s29, 6 // (op 1002) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1002) least significant
s_mul_hi_u32 s18, s28, 6 // (op 1002) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1002) most significant: sum

s_mul_i32 s19, s30, s1 // (op 1002) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 1002) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 1002) least significant
s_mul_hi_u32 s50, s30, s0 // (op 1002) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1002) most significant: sum
s_add_u32 s19, s19, s50 // (op 1002) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1002) 

s_mov_b32 s52, s32 // (op 1002) reg expression
s_mov_b32 s53, s33 // (op 1002) reg expression
s_mov_b32 s54, s0 // (op 1002) reg expression
s_mov_b32 s55, 131072 // (op 1002) call()

s_mul_hi_i32 s0, s16, s12 // (op 1045) 
v_lshrrev_b32 v12, 7, v1 // (op 1045) 
s_add_i32 s1, s0, s16 // (op 1045) 

v_and_b32 v13, 15, v1 // (op 1045) 
s_ashr_i32 s0, s1, 31 // (op 1045) 
v_lshl_add_u32 v20, v12, 4, v13 // (op 1045) 


s_cmp_eq_i32 s12, 0 // (op 1045) 
s_mov_b32 s18, scc // (op 1045) 
v_and_b32 v12, 63, v1 // (op 1045) 
s_cmp_eq_i32 s18, 0 // (op 1045) 
s_cselect_b32 s19, 0, -1 // (op 1045) 

v_lshrrev_b32 v13, 4, v12 // (op 1045) 

s_and_b32 s18, s15, 31 // (op 1045) 
s_lshl_b32 s49, 1, s18 // (op 1045) 
s_add_i32 s50, s49, s19 // (op 1045) 


s_and_b32 s19, s0, s50 // (op 1045) 


s_add_i32 s0, s1, s19 // (op 1045) 


s_ashr_i32 s1, s0, s18 // (op 1045) 


s_lshl_b32 s0, s15, 24 // (op 1045) 
s_ashr_i32 s18, s0, 31 // (op 1045) 

s_xor_b32 s0, s1, s18 // (op 1045) 

s_sub_i32 s1, s0, s18 // (op 1045) 


s_mul_hi_i32 s0, s1, s44 // (op 1045) 
s_add_i32 s18, s0, s1 // (op 1045) 

s_ashr_i32 s0, s18, 31 // (op 1045) 
s_cmp_eq_i32 s44, 0 // (op 1045) 
s_mov_b32 s19, scc // (op 1045) 
s_cmp_eq_i32 s19, 0 // (op 1045) 
s_cselect_b32 s49, 0, -1 // (op 1045) 

s_and_b32 s19, s45, 31 // (op 1045) 
s_lshl_b32 s50, 1, s19 // (op 1045) 
s_add_i32 s51, s50, s49 // (op 1045) 


s_and_b32 s49, s0, s51 // (op 1045) 


s_add_i32 s0, s18, s49 // (op 1045) 


s_ashr_i32 s18, s0, s19 // (op 1045) 


s_lshl_b32 s0, s45, 24 // (op 1045) 
s_ashr_i32 s19, s0, 31 // (op 1045) 

s_xor_b32 s0, s18, s19 // (op 1045) 

s_sub_i32 s18, s0, s19 // (op 1045) 


s_mul_i32 s0, s1, s13 // (op 1045) 

s_sub_u32 s1, s16, s0 // (op 1045) 

s_mul_i32 s0, s18, s13 // (op 1045) 
s_add_u32 s0, s0, s1 // (op 1045) 


s_mul_hi_i32 s1, s16, s11 // (op 1045) 
s_add_i32 s18, s1, s16 // (op 1045) 

s_ashr_i32 s1, s18, 31 // (op 1045) 
s_cmp_eq_i32 s11, 0 // (op 1045) 
s_mov_b32 s19, scc // (op 1045) 
s_cmp_eq_i32 s19, 0 // (op 1045) 
s_cselect_b32 s49, 0, -1 // (op 1045) 

s_and_b32 s19, s47, 31 // (op 1045) 
s_lshl_b32 s50, 1, s19 // (op 1045) 
s_add_i32 s51, s50, s49 // (op 1045) 


s_and_b32 s49, s1, s51 // (op 1045) 


s_add_i32 s1, s18, s49 // (op 1045) 


s_ashr_i32 s18, s1, s19 // (op 1045) 


s_lshl_b32 s1, s47, 24 // (op 1045) 
s_ashr_i32 s19, s1, 31 // (op 1045) 

s_xor_b32 s1, s18, s19 // (op 1045) 

s_sub_i32 s18, s1, s19 // (op 1045) 


s_add_u32 s50, s8, 64 // (op 1045) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1045) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1045) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1045) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1045) 

s_mul_i32 s1, s13, s50 // (op 1045) 

s_mul_i32 s19, s10, s1 // (op 1045) 

s_mul_i32 s1, s18, s19 // (op 1045) 

s_sub_u32 s19, s16, s1 // (op 1045) 

s_mov_b32 s1, s19 // (op 1045) 
s_mul_hi_i32 s49, s1, s14 // (op 1045) 
s_add_i32 s50, s49, s1 // (op 1045) 


s_ashr_i32 s1, s50, 31 // (op 1045) 
s_cmp_eq_i32 s14, 0 // (op 1045) 
s_mov_b32 s49, scc // (op 1045) 
s_cmp_eq_i32 s49, 0 // (op 1045) 
s_cselect_b32 s51, 0, -1 // (op 1045) 

s_and_b32 s49, s46, 31 // (op 1045) 
s_lshl_b32 s64, 1, s49 // (op 1045) 
s_add_i32 s65, s64, s51 // (op 1045) 


s_and_b32 s51, s1, s65 // (op 1045) 


s_add_i32 s1, s50, s51 // (op 1045) 


s_ashr_i32 s50, s1, s49 // (op 1045) 


s_lshl_b32 s1, s46, 24 // (op 1045) 
s_ashr_i32 s49, s1, 31 // (op 1045) 

s_xor_b32 s1, s50, s49 // (op 1045) 

s_sub_i32 s50, s1, s49 // (op 1045) 


s_mul_i32 s1, s48, s13 // (op 1045) 
s_add_u32 s64, s6, 16 // (op 1045) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 1045) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1045) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1045) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 1045) 

s_sub_i32 s49, s64, s1 // (op 1045) 

s_mul_i32 s51, s50, s49 // (op 1045) 


s_sub_u32 s49, s19, s51 // (op 1045) 


s_add_u32 s19, s49, s1 // (op 1045) 


s_cmp_eq_i32 s18, 0 // (op 1045) 
s_mov_b32 s1, scc // (op 1045) 

s_cmp_eq_i32 s1, 0 // (op 1045) 
s_cselect_b32 s18, s19, s0 // (op 1045) 



v_lshl_add_u32 v12, s18, 4, v20 // (op 1045) 


v_mul_lo_u32 v20, v13, s34 // (op 1045) 

v_add_u32 v13, v12, v20 // (op 1045) 


s_mul_i32 s1, s34, s39 // (op 1047) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 1047) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1047) least significant
s_mul_hi_u32 s19, s34, s38 // (op 1047) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1047) most significant: sum
s_add_u32 s1, s1, s19 // (op 1047) most significant: sum


s_mov_b32 s64, s36 // (op 1047) reg expression
s_mov_b32 s65, s37 // (op 1047) reg expression
s_mov_b32 s66, s0 // (op 1047) reg expression
s_mov_b32 s67, 131072 // (op 1047) call()

s_mul_hi_i32 s0, s16, s12 // (op 1064) 
v_lshrrev_b32 v12, 7, v1 // (op 1064) 
s_add_i32 s1, s0, s16 // (op 1064) 

v_and_b32 v20, 15, v1 // (op 1064) 
s_ashr_i32 s0, s1, 31 // (op 1064) 
v_lshl_add_u32 v21, v12, 4, v20 // (op 1064) 


s_cmp_eq_i32 s12, 0 // (op 1064) 
s_mov_b32 s18, scc // (op 1064) 
v_and_b32 v12, 63, v1 // (op 1064) 
s_cmp_eq_i32 s18, 0 // (op 1064) 
s_cselect_b32 s19, 0, -1 // (op 1064) 

v_lshrrev_b32 v20, 4, v12 // (op 1064) 

s_and_b32 s18, s15, 31 // (op 1064) 
s_lshl_b32 s49, 1, s18 // (op 1064) 
s_add_i32 s50, s49, s19 // (op 1064) 


s_and_b32 s19, s0, s50 // (op 1064) 


s_add_i32 s0, s1, s19 // (op 1064) 


s_ashr_i32 s1, s0, s18 // (op 1064) 


s_lshl_b32 s0, s15, 24 // (op 1064) 
s_ashr_i32 s18, s0, 31 // (op 1064) 

s_xor_b32 s0, s1, s18 // (op 1064) 

s_sub_i32 s1, s0, s18 // (op 1064) 


s_mul_hi_i32 s0, s1, s44 // (op 1064) 
s_add_i32 s18, s0, s1 // (op 1064) 

s_ashr_i32 s0, s18, 31 // (op 1064) 
s_cmp_eq_i32 s44, 0 // (op 1064) 
s_mov_b32 s19, scc // (op 1064) 
s_cmp_eq_i32 s19, 0 // (op 1064) 
s_cselect_b32 s49, 0, -1 // (op 1064) 

s_and_b32 s19, s45, 31 // (op 1064) 
s_lshl_b32 s50, 1, s19 // (op 1064) 
s_add_i32 s51, s50, s49 // (op 1064) 


s_and_b32 s49, s0, s51 // (op 1064) 


s_add_i32 s0, s18, s49 // (op 1064) 


s_ashr_i32 s18, s0, s19 // (op 1064) 


s_lshl_b32 s0, s45, 24 // (op 1064) 
s_ashr_i32 s19, s0, 31 // (op 1064) 

s_xor_b32 s0, s18, s19 // (op 1064) 

s_sub_i32 s18, s0, s19 // (op 1064) 


s_mul_i32 s0, s1, s13 // (op 1064) 

s_sub_u32 s1, s16, s0 // (op 1064) 

s_mul_i32 s0, s18, s13 // (op 1064) 
s_add_u32 s0, s0, s1 // (op 1064) 


s_mul_hi_i32 s1, s16, s11 // (op 1064) 
s_add_i32 s18, s1, s16 // (op 1064) 

s_ashr_i32 s1, s18, 31 // (op 1064) 
s_cmp_eq_i32 s11, 0 // (op 1064) 
s_mov_b32 s19, scc // (op 1064) 
s_cmp_eq_i32 s19, 0 // (op 1064) 
s_cselect_b32 s49, 0, -1 // (op 1064) 

s_and_b32 s19, s47, 31 // (op 1064) 
s_lshl_b32 s50, 1, s19 // (op 1064) 
s_add_i32 s51, s50, s49 // (op 1064) 


s_and_b32 s49, s1, s51 // (op 1064) 


s_add_i32 s1, s18, s49 // (op 1064) 


s_ashr_i32 s18, s1, s19 // (op 1064) 


s_lshl_b32 s1, s47, 24 // (op 1064) 
s_ashr_i32 s19, s1, 31 // (op 1064) 

s_xor_b32 s1, s18, s19 // (op 1064) 

s_sub_i32 s18, s1, s19 // (op 1064) 


s_add_u32 s50, s8, 64 // (op 1064) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1064) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1064) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1064) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1064) 

s_mul_i32 s1, s13, s50 // (op 1064) 

s_mul_i32 s19, s10, s1 // (op 1064) 

s_mul_i32 s1, s18, s19 // (op 1064) 

s_sub_u32 s19, s16, s1 // (op 1064) 

s_mov_b32 s1, s19 // (op 1064) 
s_mul_hi_i32 s49, s1, s14 // (op 1064) 
s_add_i32 s50, s49, s1 // (op 1064) 


s_ashr_i32 s1, s50, 31 // (op 1064) 
s_cmp_eq_i32 s14, 0 // (op 1064) 
s_mov_b32 s49, scc // (op 1064) 
s_cmp_eq_i32 s49, 0 // (op 1064) 
s_cselect_b32 s51, 0, -1 // (op 1064) 

s_and_b32 s49, s46, 31 // (op 1064) 
s_lshl_b32 s68, 1, s49 // (op 1064) 
s_add_i32 s69, s68, s51 // (op 1064) 


s_and_b32 s51, s1, s69 // (op 1064) 


s_add_i32 s1, s50, s51 // (op 1064) 


s_ashr_i32 s50, s1, s49 // (op 1064) 


s_lshl_b32 s1, s46, 24 // (op 1064) 
s_ashr_i32 s49, s1, 31 // (op 1064) 

s_xor_b32 s1, s50, s49 // (op 1064) 

s_sub_i32 s50, s1, s49 // (op 1064) 


s_mul_i32 s1, s48, s13 // (op 1064) 
s_add_u32 s68, s6, 16 // (op 1064) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 1064) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1064) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1064) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 1064) 

s_sub_i32 s49, s68, s1 // (op 1064) 

s_mul_i32 s51, s50, s49 // (op 1064) 


s_sub_u32 s49, s19, s51 // (op 1064) 


s_add_u32 s19, s49, s1 // (op 1064) 


s_cmp_eq_i32 s18, 0 // (op 1064) 
s_mov_b32 s1, scc // (op 1064) 

s_cmp_eq_i32 s1, 0 // (op 1064) 
s_cselect_b32 s18, s19, s0 // (op 1064) 



v_lshl_add_u32 v12, s18, 4, v21 // (op 1064) 


v_mul_lo_u32 v21, v20, s34 // (op 1064) 

v_add_u32 v20, v12, v21 // (op 1064) 


s_mul_i32 s1, s34, s39 // (op 1066) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 1066) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1066) least significant
s_mul_hi_u32 s19, s34, s38 // (op 1066) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1066) most significant: sum
s_add_u32 s1, s1, s19 // (op 1066) most significant: sum


s_mov_b32 s64, s36 // (op 1066) reg expression
s_mov_b32 s65, s37 // (op 1066) reg expression
s_mov_b32 s66, s0 // (op 1066) reg expression
s_mov_b32 s67, 131072 // (op 1066) call()

s_mul_hi_i32 s0, s16, s12 // (op 1109) 
v_and_b32 v12, -64, v1 // (op 1109) 
s_add_i32 s1, s0, s16 // (op 1109) 

v_sub_u32 v21, v1, v12 // (op 1109) 

s_ashr_i32 s0, s1, 31 // (op 1109) 
v_lshrrev_b32 v12, 6, v1 // (op 1109) 
s_cmp_eq_i32 s12, 0 // (op 1109) 
s_mov_b32 s18, scc // (op 1109) 
v_lshlrev_b32 v22, 1, v12 // (op 1109) 

s_cmp_eq_i32 s18, 0 // (op 1109) 
s_cselect_b32 s19, 0, -1 // (op 1109) 

s_and_b32 s18, s15, 31 // (op 1109) 
s_lshl_b32 s49, 1, s18 // (op 1109) 
s_add_i32 s50, s49, s19 // (op 1109) 


s_and_b32 s19, s0, s50 // (op 1109) 


s_add_i32 s0, s1, s19 // (op 1109) 


s_ashr_i32 s1, s0, s18 // (op 1109) 


s_lshl_b32 s0, s15, 24 // (op 1109) 
s_ashr_i32 s18, s0, 31 // (op 1109) 

s_xor_b32 s0, s1, s18 // (op 1109) 

s_sub_i32 s1, s0, s18 // (op 1109) 


s_mul_hi_i32 s0, s1, s44 // (op 1109) 
s_add_i32 s18, s0, s1 // (op 1109) 

s_ashr_i32 s0, s18, 31 // (op 1109) 
s_cmp_eq_i32 s44, 0 // (op 1109) 
s_mov_b32 s19, scc // (op 1109) 
s_cmp_eq_i32 s19, 0 // (op 1109) 
s_cselect_b32 s49, 0, -1 // (op 1109) 

s_and_b32 s19, s45, 31 // (op 1109) 
s_lshl_b32 s50, 1, s19 // (op 1109) 
s_add_i32 s51, s50, s49 // (op 1109) 


s_and_b32 s49, s0, s51 // (op 1109) 


s_add_i32 s0, s18, s49 // (op 1109) 


s_ashr_i32 s18, s0, s19 // (op 1109) 


s_lshl_b32 s0, s45, 24 // (op 1109) 
s_ashr_i32 s19, s0, 31 // (op 1109) 

s_xor_b32 s0, s18, s19 // (op 1109) 

s_sub_i32 s18, s0, s19 // (op 1109) 


s_add_u32 s50, s8, 64 // (op 1109) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1109) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1109) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1109) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1109) 

s_mul_i32 s0, s18, s50 // (op 1109) 

s_sub_i32 s49, s1, s0 // (op 1109) 


s_mul_hi_i32 s0, s16, s11 // (op 1109) 
s_add_i32 s1, s0, s16 // (op 1109) 

s_ashr_i32 s0, s1, 31 // (op 1109) 
s_cmp_eq_i32 s11, 0 // (op 1109) 
s_mov_b32 s18, scc // (op 1109) 
s_cmp_eq_i32 s18, 0 // (op 1109) 
s_cselect_b32 s19, 0, -1 // (op 1109) 

s_and_b32 s18, s47, 31 // (op 1109) 
s_lshl_b32 s68, 1, s18 // (op 1109) 
s_add_i32 s69, s68, s19 // (op 1109) 


s_and_b32 s19, s0, s69 // (op 1109) 


s_add_i32 s0, s1, s19 // (op 1109) 


s_ashr_i32 s1, s0, s18 // (op 1109) 


s_lshl_b32 s0, s47, 24 // (op 1109) 
s_ashr_i32 s18, s0, 31 // (op 1109) 

s_xor_b32 s0, s1, s18 // (op 1109) 

s_sub_i32 s1, s0, s18 // (op 1109) 


s_mul_i32 s0, s13, s50 // (op 1109) 

s_mul_i32 s18, s10, s0 // (op 1109) 

s_mul_i32 s0, s1, s18 // (op 1109) 

s_sub_u32 s18, s16, s0 // (op 1109) 

s_mul_hi_i32 s0, s18, s14 // (op 1109) 
s_add_i32 s19, s0, s18 // (op 1109) 


s_ashr_i32 s0, s19, 31 // (op 1109) 
s_cmp_eq_i32 s14, 0 // (op 1109) 
s_mov_b32 s18, scc // (op 1109) 
s_cmp_eq_i32 s18, 0 // (op 1109) 
s_cselect_b32 s50, 0, -1 // (op 1109) 

s_and_b32 s18, s46, 31 // (op 1109) 
s_lshl_b32 s51, 1, s18 // (op 1109) 
s_add_i32 s68, s51, s50 // (op 1109) 


s_and_b32 s50, s0, s68 // (op 1109) 


s_add_i32 s0, s19, s50 // (op 1109) 


s_ashr_i32 s19, s0, s18 // (op 1109) 


s_lshl_b32 s0, s46, 24 // (op 1109) 
s_ashr_i32 s18, s0, 31 // (op 1109) 

s_xor_b32 s0, s19, s18 // (op 1109) 

s_sub_i32 s19, s0, s18 // (op 1109) 


s_cmp_eq_i32 s1, 0 // (op 1109) 
s_mov_b32 s0, scc // (op 1109) 

s_cmp_eq_i32 s0, 0 // (op 1109) 
s_cselect_b32 s1, s19, s49 // (op 1109) 



v_lshl_add_u32 v12, s1, 6, v21 // (op 1109) 


v_mul_lo_u32 v21, v22, s40 // (op 1109) 
v_add_u32 v21, v21, v12 // (op 1109) 


s_mul_i32 s1, s40, s39 // (op 1111) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 1111) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 1111) least significant
s_mul_hi_u32 s19, s40, s38 // (op 1111) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1111) most significant: sum
s_add_u32 s1, s1, s19 // (op 1111) most significant: sum


s_mov_b32 s68, s42 // (op 1111) reg expression
s_mov_b32 s69, s43 // (op 1111) reg expression
s_mov_b32 s70, s0 // (op 1111) reg expression
s_mov_b32 s71, 131072 // (op 1111) call()

s_mul_hi_i32 s0, s16, s12 // (op 1128) 
v_and_b32 v12, -64, v1 // (op 1128) 
s_add_i32 s1, s0, s16 // (op 1128) 

v_sub_u32 v22, v1, v12 // (op 1128) 

s_ashr_i32 s0, s1, 31 // (op 1128) 
v_lshrrev_b32 v12, 6, v1 // (op 1128) 
s_cmp_eq_i32 s12, 0 // (op 1128) 
s_mov_b32 s18, scc // (op 1128) 
v_lshlrev_b32 v23, 1, v12 // (op 1128) 

s_cmp_eq_i32 s18, 0 // (op 1128) 
s_cselect_b32 s19, 0, -1 // (op 1128) 

s_and_b32 s18, s15, 31 // (op 1128) 
s_lshl_b32 s49, 1, s18 // (op 1128) 
s_add_i32 s50, s49, s19 // (op 1128) 


s_and_b32 s19, s0, s50 // (op 1128) 


s_add_i32 s0, s1, s19 // (op 1128) 


s_ashr_i32 s1, s0, s18 // (op 1128) 


s_lshl_b32 s0, s15, 24 // (op 1128) 
s_ashr_i32 s18, s0, 31 // (op 1128) 

s_xor_b32 s0, s1, s18 // (op 1128) 

s_sub_i32 s1, s0, s18 // (op 1128) 


s_mul_hi_i32 s0, s1, s44 // (op 1128) 
s_add_i32 s18, s0, s1 // (op 1128) 

s_ashr_i32 s0, s18, 31 // (op 1128) 
s_cmp_eq_i32 s44, 0 // (op 1128) 
s_mov_b32 s19, scc // (op 1128) 
s_cmp_eq_i32 s19, 0 // (op 1128) 
s_cselect_b32 s49, 0, -1 // (op 1128) 

s_and_b32 s19, s45, 31 // (op 1128) 
s_lshl_b32 s50, 1, s19 // (op 1128) 
s_add_i32 s51, s50, s49 // (op 1128) 


s_and_b32 s49, s0, s51 // (op 1128) 


s_add_i32 s0, s18, s49 // (op 1128) 


s_ashr_i32 s18, s0, s19 // (op 1128) 


s_lshl_b32 s0, s45, 24 // (op 1128) 
s_ashr_i32 s19, s0, 31 // (op 1128) 

s_xor_b32 s0, s18, s19 // (op 1128) 

s_sub_i32 s18, s0, s19 // (op 1128) 


s_add_u32 s50, s8, 64 // (op 1128) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1128) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1128) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1128) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1128) 

s_mul_i32 s0, s18, s50 // (op 1128) 

s_sub_i32 s49, s1, s0 // (op 1128) 


s_mul_hi_i32 s0, s16, s11 // (op 1128) 
s_add_i32 s1, s0, s16 // (op 1128) 

s_ashr_i32 s0, s1, 31 // (op 1128) 
s_cmp_eq_i32 s11, 0 // (op 1128) 
s_mov_b32 s18, scc // (op 1128) 
s_cmp_eq_i32 s18, 0 // (op 1128) 
s_cselect_b32 s19, 0, -1 // (op 1128) 

s_and_b32 s18, s47, 31 // (op 1128) 
s_lshl_b32 s72, 1, s18 // (op 1128) 
s_add_i32 s73, s72, s19 // (op 1128) 


s_and_b32 s19, s0, s73 // (op 1128) 


s_add_i32 s0, s1, s19 // (op 1128) 


s_ashr_i32 s1, s0, s18 // (op 1128) 


s_lshl_b32 s0, s47, 24 // (op 1128) 
s_ashr_i32 s18, s0, 31 // (op 1128) 

s_xor_b32 s0, s1, s18 // (op 1128) 

s_sub_i32 s1, s0, s18 // (op 1128) 


s_mul_i32 s0, s13, s50 // (op 1128) 

s_mul_i32 s18, s10, s0 // (op 1128) 

s_mul_i32 s0, s1, s18 // (op 1128) 

s_sub_u32 s18, s16, s0 // (op 1128) 

s_mul_hi_i32 s0, s18, s14 // (op 1128) 
s_add_i32 s19, s0, s18 // (op 1128) 


s_ashr_i32 s0, s19, 31 // (op 1128) 
s_cmp_eq_i32 s14, 0 // (op 1128) 
s_mov_b32 s18, scc // (op 1128) 
s_cmp_eq_i32 s18, 0 // (op 1128) 
s_cselect_b32 s50, 0, -1 // (op 1128) 

s_and_b32 s18, s46, 31 // (op 1128) 
s_lshl_b32 s51, 1, s18 // (op 1128) 
s_add_i32 s72, s51, s50 // (op 1128) 


s_and_b32 s50, s0, s72 // (op 1128) 


s_add_i32 s0, s19, s50 // (op 1128) 


s_ashr_i32 s19, s0, s18 // (op 1128) 


s_lshl_b32 s0, s46, 24 // (op 1128) 
s_ashr_i32 s18, s0, 31 // (op 1128) 

s_xor_b32 s0, s19, s18 // (op 1128) 

s_sub_i32 s19, s0, s18 // (op 1128) 


s_cmp_eq_i32 s1, 0 // (op 1128) 
s_mov_b32 s0, scc // (op 1128) 

s_cmp_eq_i32 s0, 0 // (op 1128) 
s_cselect_b32 s1, s19, s49 // (op 1128) 



v_lshl_add_u32 v12, s1, 6, v22 // (op 1128) 


v_mul_lo_u32 v22, v23, s40 // (op 1128) 
v_add_u32 v22, v22, v12 // (op 1128) 


s_mul_i32 s1, s40, s39 // (op 1130) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 1130) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 1130) least significant
s_mul_hi_u32 s19, s40, s38 // (op 1130) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1130) most significant: sum
s_add_u32 s1, s1, s19 // (op 1130) most significant: sum


s_mov_b32 s68, s42 // (op 1130) reg expression
s_mov_b32 s69, s43 // (op 1130) reg expression
s_mov_b32 s70, s0 // (op 1130) reg expression
s_mov_b32 s71, 131072 // (op 1130) call()


v_mov_b64 v[24:25], s[34:35] // (op 761) convert

v_mov_b32 v12, v13 // (op 761) reg expression
v_mov_b32 v23, v12 // (op 761) 
buffer_load_ubyte v26, v23, s[64:67], 0 offen offset: 0 // (op 761) Load value





v_mov_b32 v12, v14 // (op 765) reg expression

buffer_load_dwordx3 v[28:30], v12, s[52:55], 0 offen // (op 765) Load value
buffer_load_dwordx3 v[32:34], v12, s[52:55], 0 offen offset:12 // (op 765) Load value
buffer_load_dwordx3 v[36:38], v12, s[52:55], 0 offen offset:24 // (op 765) Load value
buffer_load_dwordx3 v[40:42], v12, s[52:55], 0 offen offset:36 // (op 765) Load value


v_mov_b64 v[24:25], s[40:41] // (op 768) convert

v_mov_b32 v12, v21 // (op 768) reg expression
buffer_load_ubyte v44, v12, s[68:71], 0 offen offset: 0 // (op 768) Load value
v_add_u32 v12, v12, v24 // (op 768) 
buffer_load_ubyte v45, v12, s[68:71], 0 offen offset: 0 // (op 768) Load value


v_lshlrev_b32 v12, 4, v1 // (op 1194) 
s_nop 0
v_readfirstlane_b32 s0, v12 // (op 1194) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[24:25], v[16:17] // (op 872) reg expression

s_nop 0
s_mov_b32 m0, s0 // (op 872) reg expression
buffer_load_dwordx4 v24, s[20:23], 0 offen offset: 0 lds // (op 872) Load value direct to lds



v_mov_b64 v[24:25], s[34:35] // (op 775) convert

s_lshl_b32 s0, s34, 2 // (op 775) 
v_add_u32 v12, v13, s0 // (op 775) 

v_mov_b32 v23, v12 // (op 775) 
buffer_load_ubyte v27, v23, s[64:67], 0 offen offset: 0 // (op 775) Load value






s_lshl_b64 s[0:1], s[30:31], 7 // (op 779) 
s_mul_i32 s19, s1, 6 // (op 779) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 779) least significant
s_mul_hi_u32 s49, s0, 6 // (op 779) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 779) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 779) 

v_add_u32 v31, v14, s0 // (op 779) 


buffer_load_dwordx3 v[46:48], v31, s[52:55], 0 offen // (op 779) Load value
buffer_load_dwordx3 v[50:52], v31, s[52:55], 0 offen offset:12 // (op 779) Load value
buffer_load_dwordx3 v[54:56], v31, s[52:55], 0 offen offset:24 // (op 779) Load value
buffer_load_dwordx3 v[58:60], v31, s[52:55], 0 offen offset:36 // (op 779) Load value



v_mov_b64 v[24:25], s[40:41] // (op 782) convert

s_lshl_b32 s0, s40, 2 // (op 782) 
v_add_u32 v23, v21, s0 // (op 782) 

buffer_load_ubyte v12, v23, s[68:71], 0 offen offset: 0 // (op 782) Load value
v_add_u32 v23, v23, v24 // (op 782) 
buffer_load_ubyte v13, v23, s[68:71], 0 offen offset: 0 // (op 782) Load value



v_lshlrev_b32 v14, 4, v1 // (op 1367) 
s_nop 0
v_readfirstlane_b32 s0, v14 // (op 1367) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[18:19], s[24:25], 7 // (op 877) 
v_mov_b32 v14, s18 // (op 877) 
v_mov_b32 v21, s19 // (op 877) 
v_add_co_u32 v24, s[50:51], v16, v14 // (op 877) least significant half
v_addc_co_u32 v25, s[50:51], v17, v21, s[50:51] // (op 877) most significant half




s_add_u32 s1, 2048, s0 // (op 877) 

s_mov_b32 m0, s1 // (op 877) reg expression
buffer_load_dwordx4 v24, s[20:23], 0 offen offset: 0 lds // (op 877) Load value direct to lds




v_lshlrev_b32 v14, 6, v1 // (op 1247) 
v_mul_lo_u32 v21, v14, 6 // (op 1247) 
v_lshrrev_b32 v31, 3, v21 // (op 1247) 

v_lshrrev_b32 v21, 4, v14 // (op 1247) 

v_lshlrev_b32 v14, 2, v21 // (op 1247) 

v_add_u32 v21, v31, v14 // (op 1247) 





s_waitcnt vmcnt(14)
ds_write_b96 v21, v[28:30] offset:4096 // (op 787) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v21, v[32:34] offset:4112 // (op 787) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v21, v[36:38] offset:4128 // (op 787) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v21, v[40:42] offset:4144 // (op 787) Store local data 

v_and_b32 v14, -64, v1 // (op 1305) 
v_sub_u32 v21, v1, v14 // (op 1305) 

v_lshrrev_b32 v14, 6, v1 // (op 1305) 
v_lshl_add_u32 v31, v14, 7, v21 // (op 1305) 





s_waitcnt vmcnt(10)
ds_write_b8 v31, v44 offset:12288 // (op 788) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v31, v45 offset:12352 // (op 788) Store local data 

v_lshlrev_b32 v14, 4, v1 // (op 1166) 
s_nop 0
v_readfirstlane_b32 s0, v14 // (op 1166) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v14, 6, v1 // (op 1218) 
v_mul_lo_u32 v21, v14, 6 // (op 1218) 
v_lshrrev_b32 v31, 3, v21 // (op 1218) 

v_lshrrev_b32 v21, 4, v14 // (op 1218) 

v_lshlrev_b32 v14, 2, v21 // (op 1218) 

v_add_u32 v21, v31, v14 // (op 1218) 


v_and_b32 v14, -64, v1 // (op 1276) 
v_sub_u32 v31, v1, v14 // (op 1276) 

v_lshrrev_b32 v14, 6, v1 // (op 1276) 
v_lshl_add_u32 v35, v14, 7, v31 // (op 1276) 


v_lshlrev_b32 v14, 4, v1 // (op 1352) 
s_nop 0
v_readfirstlane_b32 s1, v14 // (op 1352) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v14, 6, v1 // (op 1391) 
v_mul_lo_u32 v31, v14, 6 // (op 1391) 
v_lshrrev_b32 v39, 3, v31 // (op 1391) 

v_lshrrev_b32 v31, 4, v14 // (op 1391) 

v_lshlrev_b32 v14, 2, v31 // (op 1391) 

v_add_u32 v31, v39, v14 // (op 1391) 


v_and_b32 v14, -64, v1 // (op 1420) 
v_sub_u32 v39, v1, v14 // (op 1420) 

v_lshrrev_b32 v14, 6, v1 // (op 1420) 
v_lshl_add_u32 v43, v14, 7, v39 // (op 1420) 


s_mov_b32 s18, 0 // (op 308) call()
s_ashr_i64 s[50:51], s[28:29], 7 // (op 307) 
s_mov_b32 s19, s50 // (op 307) convert

s_and_b32 s49, s19, 4294967294 // (op 307) 

s_cmp_lt_u32 s18, s49 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_307 // (op 307) Condition: Top (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_307 if false)
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_307:
s_add_u32 s18, s18, 2 // (op 309) 
s_waitcnt vmcnt(8)
s_barrier  // (op 1478) 



ds_read_b64_tr_b8 v[62:63], v0 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v0 offset:128 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v0 offset:1024 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v0 offset:1152 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1491) 



ds_read_b96_tr_b6 v[70:72], v4 offset:4096 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[73:75], v4 offset:5120 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1505) 


ds_read_u8 v14, v7 offset:12288 // (op 79) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[70:75], a[0:3], v26, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 82) 
ds_read_b96_tr_b6 v[76:78], v4 offset:4128 // (op 339) Transpose load local data 


ds_read_b96_tr_b6 v[79:81], v4 offset:5152 // (op 339) Transpose load local data 


ds_read_u8 v14, v7 offset:12320 // (op 342) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[76:81], a[4:7], v26, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 343) 




v_mov_b64 v[16:17], s[34:35] // (op 75) convert

v_mov_b32 v14, v20 // (op 75) 
s_lshl_b32 s19, s34, 3 // (op 75) 
v_add_u32 v39, v14, s19 // (op 75) 

v_mov_b32 v49, v39 // (op 75) 
buffer_load_ubyte v26, v49, s[64:67], 0 offen offset: 0 // (op 75) Load value





v_mov_b32 v39, v15 // (op 10) 
s_lshl_b64 s[50:51], s[30:31], 7 // (op 10) 
s_mul_i32 s73, s51, 6 // (op 10) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 10) least significant
s_mul_hi_u32 s19, s50, 6 // (op 10) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 10) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 10) 

s_lshl_b32 s19, s50, 1 // (op 10) 

v_add_u32 v49, v39, s19 // (op 10) 


buffer_load_dwordx3 v[28:30], v49, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[32:34], v49, s[52:55], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[36:38], v49, s[52:55], 0 offen offset:24 // (op 10) Load value
buffer_load_dwordx3 v[40:42], v49, s[52:55], 0 offen offset:36 // (op 10) Load value


v_mov_b64 v[16:17], s[40:41] // (op 21) convert

v_mov_b32 v49, v22 // (op 21) 
s_lshl_b32 s19, s40, 3 // (op 21) 
v_add_u32 v53, v49, s19 // (op 21) 

buffer_load_ubyte v44, v53, s[68:71], 0 offen offset: 0 // (op 21) Load value
v_add_u32 v53, v53, v16 // (op 21) 
buffer_load_ubyte v45, v53, s[68:71], 0 offen offset: 0 // (op 21) Load value


s_barrier  // (op 1481) 


v_mov_b64 v[16:17], v[18:19] // (op 856) 
s_lshl_b64 s[50:51], s[24:25], 7 // (op 856) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 856) 

v_mov_b32 v53, s72 // (op 856) 
v_mov_b32 v57, s73 // (op 856) 
v_add_co_u32 v24, s[50:51], v16, v53 // (op 856) least significant half
v_addc_co_u32 v25, s[50:51], v17, v57, s[50:51] // (op 856) most significant half





s_mov_b32 m0, s0 // (op 856) reg expression
buffer_load_dwordx4 v24, s[20:23], 0 offen offset: 0 lds // (op 856) Load value direct to lds

s_barrier  // (op 1528) 



s_waitcnt vmcnt(14)
ds_write_b96 v31, v[46:48] offset:12544 // (op 235) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v31, v[50:52] offset:12560 // (op 235) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v31, v[54:56] offset:12576 // (op 235) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v31, v[58:60] offset:12592 // (op 235) Store local data 
s_barrier  // (op 1534) 



s_waitcnt vmcnt(10)
ds_write_b8 v43, v12 offset:20736 // (op 236) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v43, v13 offset:20800 // (op 236) Store local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 1519) 



ds_read_b64_tr_b8 v[62:63], v3 offset:2048 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v3 offset:2176 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v3 offset:3072 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v3 offset:3200 // (op 227) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1525) 



ds_read_b96_tr_b6 v[70:72], v10 offset:12544 // (op 229) Transpose load local data 
ds_read_b96_tr_b6 v[73:75], v10 offset:13568 // (op 229) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1531) 


ds_read_u8 v53, v11 offset:20736 // (op 233) Load local data 



ds_read_b96_tr_b6 v[76:78], v10 offset:12576 // (op 348) Transpose load local data 
ds_read_b96_tr_b6 v[79:81], v10 offset:13600 // (op 348) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[70:75], a[0:3], v27, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 234) 




s_barrier  // (op 1595) 
ds_read_u8 v53, v11 offset:20768 // (op 351) Load local data 
s_barrier  // (op 1599) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[76:81], a[4:7], v27, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 344) 




v_mov_b64 v[24:25], s[34:35] // (op 231) convert

s_lshl_b32 s19, s34, 2 // (op 231) 
s_mul_i32 s49, 3, s19 // (op 231) 

v_add_u32 v23, v14, s49 // (op 231) 

v_mov_b32 v53, v23 // (op 231) 
buffer_load_ubyte v27, v53, s[64:67], 0 offen offset: 0 // (op 231) Load value






s_lshl_b32 s19, s34, 3 // (op 1056) 
v_add_u32 v20, v20, s19 // (op 1056) 

s_lshl_b64 s[50:51], s[30:31], 7 // (op 237) 
s_mul_i32 s73, s51, 6 // (op 237) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 237) least significant
s_mul_hi_u32 s19, s50, 6 // (op 237) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 237) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 237) 

s_mul_i32 s19, 3, s50 // (op 237) 

v_add_u32 v14, v39, s19 // (op 237) 


buffer_load_dwordx3 v[46:48], v14, s[52:55], 0 offen // (op 237) Load value
buffer_load_dwordx3 v[50:52], v14, s[52:55], 0 offen offset:12 // (op 237) Load value
buffer_load_dwordx3 v[54:56], v14, s[52:55], 0 offen offset:24 // (op 237) Load value
buffer_load_dwordx3 v[58:60], v14, s[52:55], 0 offen offset:36 // (op 237) Load value



v_mov_b64 v[24:25], s[40:41] // (op 238) convert

s_lshl_b32 s19, s40, 2 // (op 238) 
s_mul_i32 s49, 3, s19 // (op 238) 

v_add_u32 v14, v49, s49 // (op 238) 

s_lshl_b64 s[50:51], s[30:31], 7 // (op 992) 
s_mul_i32 s73, s51, 6 // (op 992) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 992) least significant
s_mul_hi_u32 s19, s50, 6 // (op 992) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 992) most significant: sum


buffer_load_ubyte v12, v14, s[68:71], 0 offen offset: 0 // (op 238) Load value
v_add_u32 v14, v14, v24 // (op 238) 
s_lshr_b64 s[50:51], s[72:73], 3 // (op 992) 

s_lshl_b32 s19, s50, 1 // (op 992) 

v_add_u32 v15, v15, s19 // (op 992) 

buffer_load_ubyte v13, v14, s[68:71], 0 offen offset: 0 // (op 238) Load value



s_barrier  // (op 1522) 
s_lshl_b32 s19, s40, 3 // (op 1120) 


v_add_u32 v22, v22, s19 // (op 1120) 

s_lshl_b64 s[50:51], s[24:25], 7 // (op 867) 
s_mul_i32 s73, 3, s51 // (op 867) most significant: low of low * high
s_mul_i32 s72, 3, s50 // (op 867) least significant
s_mul_hi_u32 s19, 3, s50 // (op 867) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 867) most significant: sum


v_mov_b32 v14, s72 // (op 867) 
v_mov_b32 v39, s73 // (op 867) 
v_add_co_u32 v24, s[50:51], v16, v14 // (op 867) least significant half
v_addc_co_u32 v25, s[50:51], v17, v39, s[50:51] // (op 867) most significant half




s_add_u32 s19, 2048, s1 // (op 867) 

s_mov_b32 m0, s19 // (op 867) reg expression
buffer_load_dwordx4 v24, s[20:23], 0 offen offset: 0 lds // (op 867) Load value direct to lds



s_barrier  // (op 1494) 
s_lshl_b64 s[50:51], s[24:25], 7 // (op 928) 



s_lshl_b64 s[72:73], s[50:51], 1 // (op 928) 

v_mov_b32 v14, s72 // (op 928) 
v_mov_b32 v39, s73 // (op 928) 
v_add_co_u32 v18, s[50:51], v18, v14 // (op 928) least significant half
v_addc_co_u32 v19, s[50:51], v19, v39, s[50:51] // (op 928) most significant half




s_waitcnt vmcnt(14)
ds_write_b96 v21, v[28:30] offset:4096 // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v21, v[32:34] offset:4112 // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v21, v[36:38] offset:4128 // (op 13) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v21, v[40:42] offset:4144 // (op 13) Store local data 
s_barrier  // (op 1508) 



s_waitcnt vmcnt(10)
ds_write_b8 v35, v44 offset:12288 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v35, v45 offset:12352 // (op 26) Store local data 
s_ashr_i64 s[50:51], s[28:29], 7 // (op 307) 
s_mov_b32 s19, s50 // (op 307) convert

s_and_b32 s49, s19, 4294967294 // (op 307) 

s_cmp_lt_u32 s18, s49 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_307 // (op 307) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_307 if true)
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_307:














s_barrier  // (op 1579) 
s_mul_hi_i32 s0, s16, s12 // (op 890) 
s_add_i32 s1, s0, s16 // (op 890) 

s_ashr_i32 s0, s1, 31 // (op 890) 
s_cmp_eq_i32 s12, 0 // (op 890) 
s_mov_b32 s18, scc // (op 890) 
s_cmp_eq_i32 s18, 0 // (op 890) 
s_cselect_b32 s19, 0, -1 // (op 890) 

s_and_b32 s18, s15, 31 // (op 890) 
s_lshl_b32 s49, 1, s18 // (op 890) 
s_add_i32 s50, s49, s19 // (op 890) 


s_and_b32 s19, s0, s50 // (op 890) 


s_add_i32 s0, s1, s19 // (op 890) 


s_ashr_i32 s1, s0, s18 // (op 890) 


s_lshl_b32 s0, s15, 24 // (op 890) 
s_ashr_i32 s18, s0, 31 // (op 890) 

s_xor_b32 s0, s1, s18 // (op 890) 

s_sub_i32 s1, s0, s18 // (op 890) 


s_mul_hi_i32 s0, s1, s44 // (op 890) 
s_add_i32 s18, s0, s1 // (op 890) 

s_ashr_i32 s0, s18, 31 // (op 890) 
s_cmp_eq_i32 s44, 0 // (op 890) 
s_mov_b32 s19, scc // (op 890) 
s_cmp_eq_i32 s19, 0 // (op 890) 
s_cselect_b32 s49, 0, -1 // (op 890) 

s_and_b32 s19, s45, 31 // (op 890) 
s_lshl_b32 s50, 1, s19 // (op 890) 
s_add_i32 s51, s50, s49 // (op 890) 


s_and_b32 s49, s0, s51 // (op 890) 


s_add_i32 s0, s18, s49 // (op 890) 


s_ashr_i32 s18, s0, s19 // (op 890) 


s_lshl_b32 s0, s45, 24 // (op 890) 
s_ashr_i32 s19, s0, 31 // (op 890) 

s_xor_b32 s0, s18, s19 // (op 890) 

s_sub_i32 s18, s0, s19 // (op 890) 


s_mul_i32 s0, s1, s13 // (op 890) 

s_sub_u32 s1, s16, s0 // (op 890) 

s_mul_i32 s0, s18, s13 // (op 890) 
s_add_u32 s0, s0, s1 // (op 890) 


s_mul_hi_i32 s1, s16, s11 // (op 890) 
s_add_i32 s18, s1, s16 // (op 890) 

s_ashr_i32 s1, s18, 31 // (op 890) 
s_cmp_eq_i32 s11, 0 // (op 890) 
s_mov_b32 s19, scc // (op 890) 
s_cmp_eq_i32 s19, 0 // (op 890) 
s_cselect_b32 s49, 0, -1 // (op 890) 

s_and_b32 s19, s47, 31 // (op 890) 
s_lshl_b32 s50, 1, s19 // (op 890) 
s_add_i32 s51, s50, s49 // (op 890) 


s_and_b32 s49, s1, s51 // (op 890) 


s_add_i32 s1, s18, s49 // (op 890) 


s_ashr_i32 s18, s1, s19 // (op 890) 


s_lshl_b32 s1, s47, 24 // (op 890) 
s_ashr_i32 s19, s1, 31 // (op 890) 

s_xor_b32 s1, s18, s19 // (op 890) 

s_sub_i32 s18, s1, s19 // (op 890) 


s_add_u32 s50, s8, 64 // (op 890) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 890) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 890) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 890) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 890) 

s_mul_i32 s1, s13, s50 // (op 890) 

s_mul_i32 s19, s10, s1 // (op 890) 

s_mul_i32 s1, s18, s19 // (op 890) 

s_sub_u32 s19, s16, s1 // (op 890) 

s_mov_b32 s1, s19 // (op 890) 
s_mul_hi_i32 s49, s1, s14 // (op 890) 
s_add_i32 s50, s49, s1 // (op 890) 


s_ashr_i32 s1, s50, 31 // (op 890) 
s_cmp_eq_i32 s14, 0 // (op 890) 
s_mov_b32 s49, scc // (op 890) 
s_cmp_eq_i32 s49, 0 // (op 890) 
s_cselect_b32 s51, 0, -1 // (op 890) 

s_and_b32 s49, s46, 31 // (op 890) 
s_lshl_b32 s72, 1, s49 // (op 890) 
s_add_i32 s73, s72, s51 // (op 890) 


s_and_b32 s51, s1, s73 // (op 890) 


s_add_i32 s1, s50, s51 // (op 890) 


s_ashr_i32 s50, s1, s49 // (op 890) 


s_lshl_b32 s1, s46, 24 // (op 890) 
s_ashr_i32 s49, s1, 31 // (op 890) 

s_xor_b32 s1, s50, s49 // (op 890) 

s_sub_i32 s50, s1, s49 // (op 890) 


s_mul_i32 s1, s48, s13 // (op 890) 
s_add_u32 s72, s6, 16 // (op 890) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 890) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 890) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 890) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 890) 

s_sub_i32 s49, s72, s1 // (op 890) 

s_mul_i32 s51, s50, s49 // (op 890) 


s_sub_u32 s49, s19, s51 // (op 890) 


s_add_u32 s19, s49, s1 // (op 890) 


s_cmp_eq_i32 s18, 0 // (op 890) 
s_mov_b32 s1, scc // (op 890) 

s_cmp_eq_i32 s1, 0 // (op 890) 
s_cselect_b32 s18, s19, s0 // (op 890) 



s_lshl_b32 s19, s18, 4 // (op 890) 

s_mov_b32 s1, 0 // (op 890) convert
s_mov_b32 s0, s19 // (op 890) convert

s_ashr_i64 s[18:19], s[28:29], 7 // (op 890) 
s_lshr_b32 s49, s18, 1 // (op 890) 

v_lshl_add_u32 v49, s49, 8, v1 // (op 890) 

v_mul_lo_u32 v15, v49, s25 // (op 890) most significant: low of low * high
v_mul_hi_u32 v53, v49, s24 // (op 890) most significant: high of low * low
v_mul_lo_u32 v14, v49, s24 // (op 890) least significant: low of low * low
v_add_u32 v15, v15, v53 // (op 890) most significant: sum


v_mov_b32 v49, s0 // (op 890) 
v_mov_b32 v53, s1 // (op 890) 
v_add_co_u32 v16, s[18:19], v49, v14 // (op 890) least significant half
v_addc_co_u32 v17, s[18:19], v53, v15, s[18:19] // (op 890) most significant half





s_mul_i32 s1, s24, s29 // (op 892) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 892) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 892) least significant
s_mul_hi_u32 s19, s24, s28 // (op 892) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 892) most significant: sum
s_add_u32 s1, s1, s19 // (op 892) most significant: sum


s_mov_b32 s20, s26 // (op 892) reg expression
s_mov_b32 s21, s27 // (op 892) reg expression
s_mov_b32 s22, s0 // (op 892) reg expression
s_mov_b32 s23, 131072 // (op 892) call()

s_mul_hi_i32 s0, s16, s12 // (op 958) 
s_add_i32 s1, s0, s16 // (op 958) 

s_ashr_i32 s0, s1, 31 // (op 958) 
s_cmp_eq_i32 s12, 0 // (op 958) 
s_mov_b32 s18, scc // (op 958) 
s_cmp_eq_i32 s18, 0 // (op 958) 
s_cselect_b32 s19, 0, -1 // (op 958) 

s_and_b32 s18, s15, 31 // (op 958) 
s_lshl_b32 s49, 1, s18 // (op 958) 
s_add_i32 s50, s49, s19 // (op 958) 


s_and_b32 s19, s0, s50 // (op 958) 


s_add_i32 s0, s1, s19 // (op 958) 


s_ashr_i32 s1, s0, s18 // (op 958) 


s_lshl_b32 s0, s15, 24 // (op 958) 
s_ashr_i32 s18, s0, 31 // (op 958) 

s_xor_b32 s0, s1, s18 // (op 958) 

s_sub_i32 s1, s0, s18 // (op 958) 


s_mul_hi_i32 s0, s1, s44 // (op 958) 
s_add_i32 s18, s0, s1 // (op 958) 

s_ashr_i32 s0, s18, 31 // (op 958) 
s_cmp_eq_i32 s44, 0 // (op 958) 
s_mov_b32 s19, scc // (op 958) 
s_cmp_eq_i32 s19, 0 // (op 958) 
s_cselect_b32 s49, 0, -1 // (op 958) 

s_and_b32 s19, s45, 31 // (op 958) 
s_lshl_b32 s50, 1, s19 // (op 958) 
s_add_i32 s51, s50, s49 // (op 958) 


s_and_b32 s49, s0, s51 // (op 958) 


s_add_i32 s0, s18, s49 // (op 958) 


s_ashr_i32 s18, s0, s19 // (op 958) 


s_lshl_b32 s0, s45, 24 // (op 958) 
s_ashr_i32 s19, s0, 31 // (op 958) 

s_xor_b32 s0, s18, s19 // (op 958) 

s_sub_i32 s18, s0, s19 // (op 958) 


s_add_u32 s50, s8, 64 // (op 958) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 958) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 958) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 958) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 958) 

s_mul_i32 s0, s18, s50 // (op 958) 

s_sub_i32 s49, s1, s0 // (op 958) 


s_mul_hi_i32 s0, s16, s11 // (op 958) 
s_add_i32 s1, s0, s16 // (op 958) 

s_ashr_i32 s0, s1, 31 // (op 958) 
s_cmp_eq_i32 s11, 0 // (op 958) 
s_mov_b32 s18, scc // (op 958) 
s_cmp_eq_i32 s18, 0 // (op 958) 
s_cselect_b32 s19, 0, -1 // (op 958) 

s_and_b32 s18, s47, 31 // (op 958) 
s_lshl_b32 s72, 1, s18 // (op 958) 
s_add_i32 s73, s72, s19 // (op 958) 


s_and_b32 s19, s0, s73 // (op 958) 


s_add_i32 s0, s1, s19 // (op 958) 


s_ashr_i32 s1, s0, s18 // (op 958) 


s_lshl_b32 s0, s47, 24 // (op 958) 
s_ashr_i32 s18, s0, 31 // (op 958) 

s_xor_b32 s0, s1, s18 // (op 958) 

s_sub_i32 s1, s0, s18 // (op 958) 


s_mul_i32 s0, s13, s50 // (op 958) 

s_mul_i32 s18, s10, s0 // (op 958) 

s_mul_i32 s0, s1, s18 // (op 958) 

s_sub_u32 s18, s16, s0 // (op 958) 

s_mul_hi_i32 s0, s18, s14 // (op 958) 
s_add_i32 s19, s0, s18 // (op 958) 


s_ashr_i32 s0, s19, 31 // (op 958) 
s_cmp_eq_i32 s14, 0 // (op 958) 
s_mov_b32 s18, scc // (op 958) 
s_cmp_eq_i32 s18, 0 // (op 958) 
s_cselect_b32 s50, 0, -1 // (op 958) 

s_and_b32 s18, s46, 31 // (op 958) 
s_lshl_b32 s51, 1, s18 // (op 958) 
s_add_i32 s72, s51, s50 // (op 958) 


s_and_b32 s50, s0, s72 // (op 958) 


s_add_i32 s0, s19, s50 // (op 958) 


s_ashr_i32 s19, s0, s18 // (op 958) 


s_lshl_b32 s0, s46, 24 // (op 958) 
s_ashr_i32 s18, s0, 31 // (op 958) 

s_xor_b32 s0, s19, s18 // (op 958) 

s_sub_i32 s19, s0, s18 // (op 958) 


s_cmp_eq_i32 s1, 0 // (op 958) 
s_mov_b32 s0, scc // (op 958) 

s_cmp_eq_i32 s0, 0 // (op 958) 
s_cselect_b32 s1, s19, s49 // (op 958) 



s_lshl_b32 s0, s1, 6 // (op 958) 

s_mov_b32 s19, 0 // (op 958) convert
s_mov_b32 s18, s0 // (op 958) convert

s_ashr_i64 s[0:1], s[28:29], 7 // (op 958) 
s_lshr_b32 s49, s0, 1 // (op 958) 

v_lshl_add_u32 v49, s49, 8, v1 // (op 958) 

v_mul_lo_u32 v15, v49, s31 // (op 958) most significant: low of low * high
v_mul_hi_u32 v53, v49, s30 // (op 958) most significant: high of low * low
v_mul_lo_u32 v14, v49, s30 // (op 958) least significant: low of low * low
v_add_u32 v15, v15, v53 // (op 958) most significant: sum

v_mov_b32 v53, s18 // (op 958) 
v_mov_b32 v57, s19 // (op 958) 
v_add_co_u32 v14, s[0:1], v14, v53 // (op 958) least significant half
v_addc_co_u32 v15, s[0:1], v15, v57, s[0:1] // (op 958) most significant half





v_mul_lo_u32 v19, v15, 6 // (op 958) most significant: low of high * low
v_mul_hi_u32 v49, v14, 6 // (op 958) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 958) least significant: low of low * low
v_add_u32 v19, v19, v49 // (op 958) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 958) 

v_mov_b32 v49, v14 // (op 958) convert

s_mul_i32 s1, s29, 6 // (op 960) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 960) least significant
s_mul_hi_u32 s18, s28, 6 // (op 960) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 960) most significant: sum

s_mul_i32 s19, s30, s1 // (op 960) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 960) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 960) least significant
s_mul_hi_u32 s50, s30, s0 // (op 960) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 960) most significant: sum
s_add_u32 s19, s19, s50 // (op 960) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 960) 

s_mov_b32 s52, s32 // (op 960) reg expression
s_mov_b32 s53, s33 // (op 960) reg expression
s_mov_b32 s54, s0 // (op 960) reg expression
s_mov_b32 s55, 131072 // (op 960) call()

s_mul_hi_i32 s0, s16, s12 // (op 1022) 
v_lshrrev_b32 v53, 7, v1 // (op 1022) 
s_add_i32 s1, s0, s16 // (op 1022) 

v_and_b32 v57, 15, v1 // (op 1022) 
s_ashr_i32 s0, s1, 31 // (op 1022) 
v_lshl_add_u32 v14, v53, 4, v57 // (op 1022) 


s_cmp_eq_i32 s12, 0 // (op 1022) 
s_mov_b32 s18, scc // (op 1022) 
v_and_b32 v15, 63, v1 // (op 1022) 
s_cmp_eq_i32 s18, 0 // (op 1022) 
s_cselect_b32 s19, 0, -1 // (op 1022) 

v_lshrrev_b32 v53, 4, v15 // (op 1022) 

s_and_b32 s18, s15, 31 // (op 1022) 
s_lshl_b32 s49, 1, s18 // (op 1022) 
s_add_i32 s50, s49, s19 // (op 1022) 


s_and_b32 s19, s0, s50 // (op 1022) 


s_add_i32 s0, s1, s19 // (op 1022) 


s_ashr_i32 s1, s0, s18 // (op 1022) 


s_lshl_b32 s0, s15, 24 // (op 1022) 
s_ashr_i32 s18, s0, 31 // (op 1022) 

s_xor_b32 s0, s1, s18 // (op 1022) 

s_sub_i32 s1, s0, s18 // (op 1022) 


s_mul_hi_i32 s0, s1, s44 // (op 1022) 
s_add_i32 s18, s0, s1 // (op 1022) 

s_ashr_i32 s0, s18, 31 // (op 1022) 
s_cmp_eq_i32 s44, 0 // (op 1022) 
s_mov_b32 s19, scc // (op 1022) 
s_cmp_eq_i32 s19, 0 // (op 1022) 
s_cselect_b32 s49, 0, -1 // (op 1022) 

s_and_b32 s19, s45, 31 // (op 1022) 
s_lshl_b32 s50, 1, s19 // (op 1022) 
s_add_i32 s51, s50, s49 // (op 1022) 


s_and_b32 s49, s0, s51 // (op 1022) 


s_add_i32 s0, s18, s49 // (op 1022) 


s_ashr_i32 s18, s0, s19 // (op 1022) 


s_lshl_b32 s0, s45, 24 // (op 1022) 
s_ashr_i32 s19, s0, 31 // (op 1022) 

s_xor_b32 s0, s18, s19 // (op 1022) 

s_sub_i32 s18, s0, s19 // (op 1022) 


s_mul_i32 s0, s1, s13 // (op 1022) 

s_sub_u32 s1, s16, s0 // (op 1022) 

s_mul_i32 s0, s18, s13 // (op 1022) 
s_add_u32 s0, s0, s1 // (op 1022) 


s_mul_hi_i32 s1, s16, s11 // (op 1022) 
s_add_i32 s18, s1, s16 // (op 1022) 

s_ashr_i32 s1, s18, 31 // (op 1022) 
s_cmp_eq_i32 s11, 0 // (op 1022) 
s_mov_b32 s19, scc // (op 1022) 
s_cmp_eq_i32 s19, 0 // (op 1022) 
s_cselect_b32 s49, 0, -1 // (op 1022) 

s_and_b32 s19, s47, 31 // (op 1022) 
s_lshl_b32 s50, 1, s19 // (op 1022) 
s_add_i32 s51, s50, s49 // (op 1022) 


s_and_b32 s49, s1, s51 // (op 1022) 


s_add_i32 s1, s18, s49 // (op 1022) 


s_ashr_i32 s18, s1, s19 // (op 1022) 


s_lshl_b32 s1, s47, 24 // (op 1022) 
s_ashr_i32 s19, s1, 31 // (op 1022) 

s_xor_b32 s1, s18, s19 // (op 1022) 

s_sub_i32 s18, s1, s19 // (op 1022) 


s_add_u32 s50, s8, 64 // (op 1022) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1022) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1022) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1022) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1022) 

s_mul_i32 s1, s13, s50 // (op 1022) 

s_mul_i32 s19, s10, s1 // (op 1022) 

s_mul_i32 s1, s18, s19 // (op 1022) 

s_sub_u32 s19, s16, s1 // (op 1022) 

s_mov_b32 s1, s19 // (op 1022) 
s_mul_hi_i32 s49, s1, s14 // (op 1022) 
s_add_i32 s50, s49, s1 // (op 1022) 


s_ashr_i32 s1, s50, 31 // (op 1022) 
s_cmp_eq_i32 s14, 0 // (op 1022) 
s_mov_b32 s49, scc // (op 1022) 
s_cmp_eq_i32 s49, 0 // (op 1022) 
s_cselect_b32 s51, 0, -1 // (op 1022) 

s_and_b32 s49, s46, 31 // (op 1022) 
s_lshl_b32 s72, 1, s49 // (op 1022) 
s_add_i32 s73, s72, s51 // (op 1022) 


s_and_b32 s51, s1, s73 // (op 1022) 


s_add_i32 s1, s50, s51 // (op 1022) 


s_ashr_i32 s50, s1, s49 // (op 1022) 


s_lshl_b32 s1, s46, 24 // (op 1022) 
s_ashr_i32 s49, s1, 31 // (op 1022) 

s_xor_b32 s1, s50, s49 // (op 1022) 

s_sub_i32 s50, s1, s49 // (op 1022) 


s_mul_i32 s1, s48, s13 // (op 1022) 
s_add_u32 s72, s6, 16 // (op 1022) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 1022) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1022) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1022) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 1022) 

s_sub_i32 s49, s72, s1 // (op 1022) 

s_mul_i32 s51, s50, s49 // (op 1022) 


s_sub_u32 s49, s19, s51 // (op 1022) 


s_add_u32 s19, s49, s1 // (op 1022) 


s_cmp_eq_i32 s18, 0 // (op 1022) 
s_mov_b32 s1, scc // (op 1022) 

s_cmp_eq_i32 s1, 0 // (op 1022) 
s_cselect_b32 s18, s19, s0 // (op 1022) 



v_lshl_add_u32 v15, s18, 4, v14 // (op 1022) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 1022) 
s_lshr_b32 s18, s0, 1 // (op 1022) 

v_lshl_add_u32 v14, s18, 3, v53 // (op 1022) 


v_mul_lo_u32 v53, v14, s34 // (op 1022) 

v_add_u32 v14, v15, v53 // (op 1022) 


s_mul_i32 s1, s34, s39 // (op 1024) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 1024) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1024) least significant
s_mul_hi_u32 s19, s34, s38 // (op 1024) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1024) most significant: sum
s_add_u32 s1, s1, s19 // (op 1024) most significant: sum


s_mov_b32 s64, s36 // (op 1024) reg expression
s_mov_b32 s65, s37 // (op 1024) reg expression
s_mov_b32 s66, s0 // (op 1024) reg expression
s_mov_b32 s67, 131072 // (op 1024) call()

s_mul_hi_i32 s0, s16, s12 // (op 1086) 
v_and_b32 v15, -64, v1 // (op 1086) 
s_add_i32 s1, s0, s16 // (op 1086) 

v_sub_u32 v53, v1, v15 // (op 1086) 

s_ashr_i32 s0, s1, 31 // (op 1086) 
v_lshrrev_b32 v15, 6, v1 // (op 1086) 
s_cmp_eq_i32 s12, 0 // (op 1086) 
s_mov_b32 s18, scc // (op 1086) 
v_lshlrev_b32 v57, 1, v15 // (op 1086) 

s_cmp_eq_i32 s18, 0 // (op 1086) 
s_cselect_b32 s19, 0, -1 // (op 1086) 

s_and_b32 s18, s15, 31 // (op 1086) 
s_lshl_b32 s49, 1, s18 // (op 1086) 
s_add_i32 s50, s49, s19 // (op 1086) 


s_and_b32 s19, s0, s50 // (op 1086) 


s_add_i32 s0, s1, s19 // (op 1086) 


s_ashr_i32 s1, s0, s18 // (op 1086) 


s_lshl_b32 s0, s15, 24 // (op 1086) 
s_ashr_i32 s18, s0, 31 // (op 1086) 

s_xor_b32 s0, s1, s18 // (op 1086) 

s_sub_i32 s1, s0, s18 // (op 1086) 


s_mul_hi_i32 s0, s1, s44 // (op 1086) 
s_add_i32 s18, s0, s1 // (op 1086) 

s_ashr_i32 s0, s18, 31 // (op 1086) 
s_cmp_eq_i32 s44, 0 // (op 1086) 
s_mov_b32 s19, scc // (op 1086) 
s_cmp_eq_i32 s19, 0 // (op 1086) 
s_cselect_b32 s49, 0, -1 // (op 1086) 

s_and_b32 s19, s45, 31 // (op 1086) 
s_lshl_b32 s50, 1, s19 // (op 1086) 
s_add_i32 s51, s50, s49 // (op 1086) 


s_and_b32 s49, s0, s51 // (op 1086) 


s_add_i32 s0, s18, s49 // (op 1086) 


s_ashr_i32 s18, s0, s19 // (op 1086) 


s_lshl_b32 s0, s45, 24 // (op 1086) 
s_ashr_i32 s19, s0, 31 // (op 1086) 

s_xor_b32 s0, s18, s19 // (op 1086) 

s_sub_i32 s18, s0, s19 // (op 1086) 


s_add_u32 s50, s8, 64 // (op 1086) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1086) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1086) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1086) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1086) 

s_mul_i32 s0, s18, s50 // (op 1086) 

s_sub_i32 s49, s1, s0 // (op 1086) 


s_mul_hi_i32 s0, s16, s11 // (op 1086) 
s_add_i32 s1, s0, s16 // (op 1086) 

s_ashr_i32 s0, s1, 31 // (op 1086) 
s_cmp_eq_i32 s11, 0 // (op 1086) 
s_mov_b32 s18, scc // (op 1086) 
s_cmp_eq_i32 s18, 0 // (op 1086) 
s_cselect_b32 s19, 0, -1 // (op 1086) 

s_and_b32 s18, s47, 31 // (op 1086) 
s_lshl_b32 s72, 1, s18 // (op 1086) 
s_add_i32 s73, s72, s19 // (op 1086) 


s_and_b32 s19, s0, s73 // (op 1086) 


s_add_i32 s0, s1, s19 // (op 1086) 


s_ashr_i32 s1, s0, s18 // (op 1086) 


s_lshl_b32 s0, s47, 24 // (op 1086) 
s_ashr_i32 s18, s0, 31 // (op 1086) 

s_xor_b32 s0, s1, s18 // (op 1086) 

s_sub_i32 s1, s0, s18 // (op 1086) 


s_mul_i32 s0, s13, s50 // (op 1086) 

s_mul_i32 s18, s10, s0 // (op 1086) 

s_mul_i32 s0, s1, s18 // (op 1086) 

s_sub_u32 s18, s16, s0 // (op 1086) 

s_mul_hi_i32 s0, s18, s14 // (op 1086) 
s_add_i32 s19, s0, s18 // (op 1086) 


s_ashr_i32 s0, s19, 31 // (op 1086) 
s_cmp_eq_i32 s14, 0 // (op 1086) 
s_mov_b32 s18, scc // (op 1086) 
s_cmp_eq_i32 s18, 0 // (op 1086) 
s_cselect_b32 s50, 0, -1 // (op 1086) 

s_and_b32 s18, s46, 31 // (op 1086) 
s_lshl_b32 s51, 1, s18 // (op 1086) 
s_add_i32 s72, s51, s50 // (op 1086) 


s_and_b32 s50, s0, s72 // (op 1086) 


s_add_i32 s0, s19, s50 // (op 1086) 


s_ashr_i32 s19, s0, s18 // (op 1086) 


s_lshl_b32 s0, s46, 24 // (op 1086) 
s_ashr_i32 s18, s0, 31 // (op 1086) 

s_xor_b32 s0, s19, s18 // (op 1086) 

s_sub_i32 s19, s0, s18 // (op 1086) 


s_cmp_eq_i32 s1, 0 // (op 1086) 
s_mov_b32 s0, scc // (op 1086) 

s_cmp_eq_i32 s0, 0 // (op 1086) 
s_cselect_b32 s1, s19, s49 // (op 1086) 



v_lshl_add_u32 v15, s1, 6, v53 // (op 1086) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 1086) 
s_lshr_b32 s18, s0, 1 // (op 1086) 

v_lshl_add_u32 v53, s18, 3, v57 // (op 1086) 


v_mul_lo_u32 v57, v53, s40 // (op 1086) 
v_add_u32 v57, v57, v15 // (op 1086) 


s_mul_i32 s1, s40, s39 // (op 1088) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 1088) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 1088) least significant
s_mul_hi_u32 s19, s40, s38 // (op 1088) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1088) most significant: sum
s_add_u32 s1, s1, s19 // (op 1088) most significant: sum


s_mov_b32 s68, s42 // (op 1088) reg expression
s_mov_b32 s69, s43 // (op 1088) reg expression
s_mov_b32 s70, s0 // (op 1088) reg expression
s_mov_b32 s71, 131072 // (op 1088) call()

v_lshlrev_b32 v15, 4, v1 // (op 1179) 
s_nop 0
v_readfirstlane_b32 s0, v15 // (op 1179) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v15, 6, v1 // (op 1231) 
v_mul_lo_u32 v53, v15, 6 // (op 1231) 
v_lshrrev_b32 v18, 3, v53 // (op 1231) 

v_lshrrev_b32 v53, 4, v15 // (op 1231) 

v_lshlrev_b32 v15, 2, v53 // (op 1231) 

v_add_u32 v53, v18, v15 // (op 1231) 


v_and_b32 v15, -64, v1 // (op 1289) 
v_sub_u32 v18, v1, v15 // (op 1289) 

v_lshrrev_b32 v15, 6, v1 // (op 1289) 
v_lshl_add_u32 v19, v15, 7, v18 // (op 1289) 


s_ashr_i64 s[18:19], s[28:29], 7 // (op 313) 
s_and_b32 s1, s18, 4294967294 // (op 313) 

s_ashr_i64 s[18:19], s[28:29], 7 // (op 312) 
s_mov_b32 s49, s18 // (op 312) convert

s_cmp_lt_u32 s1, s49 // (op 312) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_312 // (op 312) Condition: Top (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_312 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_312:
s_barrier  // (op 1484) 
s_add_u32 s1, s1, 1 // (op 314) 


v_mov_b64 v[12:13], v[16:17] // (op 861) 
v_mov_b64 v[20:21], v[12:13] // (op 861) reg expression

s_mov_b32 m0, s0 // (op 861) reg expression
buffer_load_dwordx4 v20, s[20:23], 0 offen offset: 0 lds // (op 861) Load value direct to lds


s_lshl_b64 s[18:19], s[24:25], 7 // (op 880) 
v_mov_b32 v15, s18 // (op 880) 
v_mov_b32 v18, s19 // (op 880) 
v_add_co_u32 v16, s[50:51], v16, v15 // (op 880) least significant half
v_addc_co_u32 v17, s[50:51], v17, v18, s[50:51] // (op 880) most significant half






v_mov_b32 v15, v49 // (op 179) 
v_mov_b32 v18, v15 // (op 179) reg expression

buffer_load_dwordx3 v[20:22], v18, s[52:55], 0 offen // (op 179) Load value
buffer_load_dwordx3 v[24:26], v18, s[52:55], 0 offen offset:12 // (op 179) Load value
buffer_load_dwordx3 v[28:30], v18, s[52:55], 0 offen offset:24 // (op 179) Load value
buffer_load_dwordx3 v[32:34], v18, s[52:55], 0 offen offset:36 // (op 179) Load value


s_lshl_b64 s[18:19], s[30:31], 7 // (op 950) 
s_mul_i32 s51, s19, 6 // (op 950) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 950) least significant
s_mul_hi_u32 s49, s18, 6 // (op 950) most significant: high of low * low
s_add_u32 s51, s51, s49 // (op 950) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 950) 

v_add_u32 v49, v49, s18 // (op 950) 


s_barrier  // (op 1497) 
v_mov_b64 v[12:13], s[40:41] // (op 180) convert




v_mov_b32 v15, v57 // (op 180) 
v_mov_b32 v18, v15 // (op 180) reg expression
buffer_load_ubyte v50, v18, s[68:71], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v18, v18, v12 // (op 180) 
s_waitcnt vmcnt(4)
ds_write_b96 v53, v[20:22] offset:4096 // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v53, v[24:26] offset:4112 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v53, v[28:30] offset:4128 // (op 177) Store local data 
buffer_load_ubyte v51, v18, s[68:71], 0 offen offset: 0 // (op 180) Load value



s_lshl_b32 s18, s40, 2 // (op 1078) 
v_add_u32 v57, v57, s18 // (op 1078) 

s_waitcnt vmcnt(2)
ds_write_b96 v53, v[32:34] offset:4144 // (op 177) Store local data 

s_barrier  // (op 1487) 



s_barrier  // (op 1511) 
ds_read_b64_tr_b8 v[20:21], v0 // (op 169) Transpose load local data 



ds_read_b64_tr_b8 v[22:23], v0 offset:128 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:1024 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:1152 // (op 169) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v50 offset:12288 // (op 178) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1501) 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v51 offset:12352 // (op 178) Store local data 




ds_read_b96_tr_b6 v[28:30], v4 offset:4096 // (op 171) Transpose load local data 
ds_read_b96_tr_b6 v[31:33], v4 offset:5120 // (op 171) Transpose load local data 

v_mov_b64 v[12:13], s[34:35] // (op 173) convert

v_mov_b32 v15, v14 // (op 173) 
v_mov_b32 v18, v15 // (op 173) reg expression
v_mov_b32 v34, v18 // (op 173) 
buffer_load_ubyte v35, v34, s[64:67], 0 offen offset: 0 // (op 173) Load value




s_lshl_b32 s18, s34, 2 // (op 1014) 
v_add_u32 v14, v14, s18 // (op 1014) 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1515) 


ds_read_u8 v15, v7 offset:12288 // (op 175) Load local data 



ds_read_b96_tr_b6 v[36:38], v4 offset:4128 // (op 320) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v4 offset:5152 // (op 320) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[28:33], a[0:3], v35, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 176) 




ds_read_u8 v15, v7 offset:12320 // (op 323) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:27], v[36:41], a[4:7], v35, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 324) 




s_ashr_i64 s[18:19], s[28:29], 7 // (op 312) 
s_mov_b32 s49, s18 // (op 312) convert

s_cmp_lt_u32 s1, s49 // (op 312) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_312 // (op 312) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_312 if true)
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_312:






















s_barrier  // (op 1583) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_123:


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
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.LRR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__end:
.size RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_, .LRR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__end-RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 92
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 20992 // lds bytes
  .amdhsa_accum_offset 84
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: B6
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
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
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: SizeFree1 },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
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
  - .name: RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 82
    .agpr_count: 8
    .max_flat_workgroup_size: 128
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_0_size_0_15
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Tensor_2_size_1_11
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
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
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_5
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 136
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 144
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: MagicMultiple_28
        .size: 4
        .offset: 152
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
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
      - .name: Subtract_24
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

