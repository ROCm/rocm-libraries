// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_,@function
RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s11 // (op 1159) 
s_mul_hi_i32 s1, s16, s11 // (op 1347) 
v_lshrrev_b32 v0, 7, v1 // (op 1216) 
v_and_b32 v3, 15, v1 // (op 1216) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1216) 


v_and_b32 v0, 63, v1 // (op 1216) 
v_lshrrev_b32 v3, 4, v0 // (op 1216) 

v_lshlrev_b32 v0, 5, v3 // (op 1216) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 1216) 


v_mul_lo_u32 v0, v3, 6 // (op 1216) 

v_lshrrev_b32 v3, 3, v0 // (op 1216) 

v_lshrrev_b32 v0, 7, v1 // (op 1373) 
v_and_b32 v4, 15, v1 // (op 1373) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1373) 


v_and_b32 v0, 63, v1 // (op 1373) 
v_lshrrev_b32 v4, 4, v0 // (op 1373) 

v_lshlrev_b32 v0, 5, v4 // (op 1373) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 1373) 


v_mul_lo_u32 v0, v4, 6 // (op 1373) 

v_lshrrev_b32 v4, 3, v0 // (op 1373) 

v_lshrrev_b32 v0, 7, v1 // (op 1159) 
s_add_i32 s18, s0, s16 // (op 1159) 

v_and_b32 v5, 63, v1 // (op 1159) 
s_ashr_i32 s0, s18, 31 // (op 1159) 
v_lshrrev_b32 v6, 4, v5 // (op 1159) 
s_cmp_eq_i32 s11, 0 // (op 1159) 
s_mov_b32 s19, scc // (op 1159) 
v_lshlrev_b32 v7, 2, v6 // (op 1159) 

s_cmp_eq_i32 s19, 0 // (op 1159) 
s_cselect_b32 s51, 0, -1 // (op 1159) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1159) 


s_and_b32 s19, s12, 31 // (op 1159) 
v_lshrrev_b32 v0, 2, v5 // (op 1159) 

v_and_b32 v5, 3, v0 // (op 1159) 

v_and_b32 v0, 48, v1 // (op 1273) 
v_and_b32 v7, 63, v1 // (op 1273) 
v_sub_u32 v8, v7, v0 // (op 1273) 

v_lshrrev_b32 v7, 1, v8 // (op 1273) 
v_add_lshl_u32 v9, v0, v7, 6 // (op 1273) 


v_and_b32 v0, 1, v8 // (op 1273) 

v_lshlrev_b32 v7, 3, v0 // (op 1273) 

v_lshrrev_b32 v0, 6, v1 // (op 1273) 
v_and_b32 v8, 1, v0 // (op 1273) 

v_lshl_add_u32 v0, v8, 4, v7 // (op 1273) 


v_add_u32 v7, v9, v0 // (op 1273) 


v_lshrrev_b32 v0, 6, v1 // (op 1331) 
v_and_b32 v8, 1, v0 // (op 1331) 

v_and_b32 v0, 15, v1 // (op 1331) 
v_lshl_add_u32 v9, v8, 4, v0 // (op 1331) 


v_and_b32 v0, 63, v1 // (op 1331) 
v_lshrrev_b32 v8, 4, v0 // (op 1331) 

v_lshl_add_u32 v0, v8, 6, v9 // (op 1331) 


v_lshrrev_b32 v8, 7, v1 // (op 1347) 
v_and_b32 v9, 63, v1 // (op 1347) 
v_and_b32 v10, 48, v1 // (op 1417) 
v_and_b32 v11, 63, v1 // (op 1417) 
v_sub_u32 v12, v11, v10 // (op 1417) 

v_lshrrev_b32 v11, 1, v12 // (op 1417) 
v_add_lshl_u32 v13, v10, v11, 6 // (op 1417) 


v_and_b32 v10, 1, v12 // (op 1417) 

v_lshlrev_b32 v11, 3, v10 // (op 1417) 

v_lshrrev_b32 v10, 6, v1 // (op 1417) 
v_and_b32 v12, 1, v10 // (op 1417) 

v_lshl_add_u32 v10, v12, 4, v11 // (op 1417) 


v_add_u32 v11, v13, v10 // (op 1417) 


v_lshrrev_b32 v10, 6, v1 // (op 1446) 
v_and_b32 v12, 1, v10 // (op 1446) 

v_and_b32 v10, 15, v1 // (op 1446) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 1446) 


v_and_b32 v10, 63, v1 // (op 1446) 
v_lshrrev_b32 v12, 4, v10 // (op 1446) 

v_lshl_add_u32 v10, v12, 6, v13 // (op 1446) 


s_lshl_b32 s52, 1, s19 // (op 1159) 
v_and_b32 v12, 3, v1 // (op 1159) 
s_add_i32 s53, s52, s51 // (op 1159) 


s_and_b32 s51, s0, s53 // (op 1159) 


s_add_i32 s0, s1, s16 // (op 1347) 

s_ashr_i32 s1, s0, 31 // (op 1347) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1159) 


s_add_i32 s52, s18, s51 // (op 1159) 


v_lshrrev_b32 v5, 6, v1 // (op 1159) 
s_ashr_i32 s18, s52, s19 // (op 1159) 


v_and_b32 v12, 1, v5 // (op 1159) 

s_lshl_b32 s19, s12, 24 // (op 1159) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1159) 


s_ashr_i32 s51, s19, 31 // (op 1159) 

s_xor_b32 s19, s18, s51 // (op 1159) 

s_sub_i32 s18, s19, s51 // (op 1159) 


s_mul_hi_i32 s19, s18, s13 // (op 1159) 
s_add_i32 s51, s19, s18 // (op 1159) 

s_ashr_i32 s19, s51, 31 // (op 1159) 
s_cmp_eq_i32 s13, 0 // (op 1159) 
s_mov_b32 s52, scc // (op 1159) 
s_cmp_eq_i32 s52, 0 // (op 1159) 
s_cselect_b32 s53, 0, -1 // (op 1159) 

s_and_b32 s52, s14, 31 // (op 1159) 
v_lshrrev_b32 v12, 4, v9 // (op 1347) 
s_cmp_eq_i32 s11, 0 // (op 1347) 
s_mov_b32 s54, scc // (op 1347) 
v_lshlrev_b32 v13, 2, v12 // (op 1347) 

s_cmp_eq_i32 s54, 0 // (op 1347) 
s_cselect_b32 s55, 0, -1 // (op 1347) 

s_lshl_b32 s54, 1, s52 // (op 1159) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 1347) 


v_lshrrev_b32 v8, 2, v9 // (op 1347) 

s_add_i32 s56, s54, s53 // (op 1159) 


s_and_b32 s53, s19, s56 // (op 1159) 


s_add_i32 s19, s51, s53 // (op 1159) 


s_ashr_i32 s51, s19, s52 // (op 1159) 


s_lshl_b32 s19, s14, 24 // (op 1159) 
s_ashr_i32 s52, s19, 31 // (op 1159) 

s_xor_b32 s19, s51, s52 // (op 1159) 

s_sub_i32 s51, s19, s52 // (op 1159) 


s_and_b32 s19, s12, 31 // (op 1347) 
s_lshl_b32 s52, 1, s19 // (op 1347) 
v_and_b32 v9, 3, v8 // (op 1347) 

s_add_i32 s53, s52, s55 // (op 1347) 


v_and_b32 v8, 3, v1 // (op 1347) 
s_and_b32 s52, s1, s53 // (op 1347) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1347) 


s_add_i32 s1, s0, s52 // (op 1347) 


v_lshrrev_b32 v8, 6, v1 // (op 1347) 
s_ashr_i32 s0, s1, s19 // (op 1347) 


v_and_b32 v9, 1, v8 // (op 1347) 

s_lshl_b32 s1, s12, 24 // (op 1347) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1347) 


s_ashr_i32 s19, s1, 31 // (op 1347) 

s_xor_b32 s1, s0, s19 // (op 1347) 

s_sub_i32 s0, s1, s19 // (op 1347) 


s_mul_hi_i32 s1, s0, s13 // (op 1347) 
s_add_i32 s19, s1, s0 // (op 1347) 

s_ashr_i32 s1, s19, 31 // (op 1347) 
s_cmp_eq_i32 s13, 0 // (op 1347) 
s_mov_b32 s52, scc // (op 1347) 
s_cmp_eq_i32 s52, 0 // (op 1347) 
s_cselect_b32 s53, 0, -1 // (op 1347) 

s_and_b32 s52, s14, 31 // (op 1347) 
s_lshl_b32 s54, 1, s52 // (op 1347) 
s_add_i32 s55, s54, s53 // (op 1347) 


s_and_b32 s53, s1, s55 // (op 1347) 


s_add_i32 s1, s19, s53 // (op 1347) 


s_ashr_i32 s19, s1, s52 // (op 1347) 


s_lshl_b32 s1, s14, 24 // (op 1347) 
s_ashr_i32 s52, s1, 31 // (op 1347) 

s_xor_b32 s1, s19, s52 // (op 1347) 

s_sub_i32 s19, s1, s52 // (op 1347) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 1159) 
s_sub_u32 s52, s16, s1 // (op 1159) 

s_mul_i32 s1, s51, s44 // (op 1159) 
s_add_u32 s1, s1, s52 // (op 1159) 

s_mul_hi_i32 s52, s16, s10 // (op 1159) 
s_add_i32 s53, s52, s16 // (op 1159) 

s_ashr_i32 s52, s53, 31 // (op 1159) 
s_cmp_eq_i32 s10, 0 // (op 1159) 
s_mov_b32 s54, scc // (op 1159) 
s_cmp_eq_i32 s54, 0 // (op 1159) 
s_cselect_b32 s55, 0, -1 // (op 1159) 

s_and_b32 s54, s47, 31 // (op 1159) 
s_lshl_b32 s56, 1, s54 // (op 1159) 
s_add_i32 s57, s56, s55 // (op 1159) 


s_and_b32 s55, s52, s57 // (op 1159) 


s_add_i32 s52, s53, s55 // (op 1159) 


s_ashr_i32 s53, s52, s54 // (op 1159) 


s_lshl_b32 s52, s47, 24 // (op 1159) 
s_ashr_i32 s54, s52, 31 // (op 1159) 

s_xor_b32 s52, s53, s54 // (op 1159) 

s_sub_i32 s53, s52, s54 // (op 1159) 


s_mul_i32 s52, s0, s44 // (op 1347) 
s_sub_u32 s54, s16, s52 // (op 1347) 

s_mul_i32 s52, s19, s44 // (op 1347) 
s_add_u32 s52, s52, s54 // (op 1347) 

s_mul_hi_i32 s54, s16, s10 // (op 1347) 
s_add_i32 s55, s54, s16 // (op 1347) 

s_ashr_i32 s54, s55, 31 // (op 1347) 
s_cmp_eq_i32 s10, 0 // (op 1347) 
s_mov_b32 s56, scc // (op 1347) 
s_cmp_eq_i32 s56, 0 // (op 1347) 
s_cselect_b32 s57, 0, -1 // (op 1347) 

s_and_b32 s56, s47, 31 // (op 1347) 
s_lshl_b32 s58, 1, s56 // (op 1347) 
s_add_i32 s59, s58, s57 // (op 1347) 


s_and_b32 s57, s54, s59 // (op 1347) 


s_add_i32 s54, s55, s57 // (op 1347) 


s_ashr_i32 s55, s54, s56 // (op 1347) 


s_lshl_b32 s54, s47, 24 // (op 1347) 
s_ashr_i32 s56, s54, 31 // (op 1347) 

s_xor_b32 s54, s55, s56 // (op 1347) 

s_sub_i32 s55, s54, s56 // (op 1347) 


s_add_u32 s56, s6, 64 // (op 1159) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1159) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1159) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1159) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1159) 

s_mul_i32 s54, s44, s56 // (op 1159) 
s_mul_i32 s58, s48, s54 // (op 1159) 

s_mul_i32 s54, s53, s58 // (op 1159) 

s_sub_u32 s58, s16, s54 // (op 1159) 

s_mov_b32 s54, s58 // (op 1159) 
s_mul_hi_i32 s59, s54, s15 // (op 1159) 
s_add_i32 s60, s59, s54 // (op 1159) 


s_ashr_i32 s54, s60, 31 // (op 1159) 
s_cmp_eq_i32 s15, 0 // (op 1159) 
s_mov_b32 s59, scc // (op 1159) 
s_cmp_eq_i32 s59, 0 // (op 1159) 
s_cselect_b32 s61, 0, -1 // (op 1159) 

s_and_b32 s59, s46, 31 // (op 1159) 
s_lshl_b32 s62, 1, s59 // (op 1159) 
s_add_i32 s63, s62, s61 // (op 1159) 


s_and_b32 s61, s54, s63 // (op 1159) 


s_add_i32 s54, s60, s61 // (op 1159) 


s_ashr_i32 s60, s54, s59 // (op 1159) 


s_lshl_b32 s54, s46, 24 // (op 1159) 
s_ashr_i32 s59, s54, 31 // (op 1159) 

s_xor_b32 s54, s60, s59 // (op 1159) 

s_sub_i32 s60, s54, s59 // (op 1159) 


s_mul_i32 s54, s45, s44 // (op 1159) 
s_add_u32 s62, s8, 16 // (op 1159) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 1159) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1159) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1159) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 1159) 

s_sub_i32 s59, s62, s54 // (op 1159) 

s_mul_i32 s61, s60, s59 // (op 1159) 

s_sub_u32 s59, s58, s61 // (op 1159) 


s_add_u32 s58, s59, s54 // (op 1159) 


s_cmp_eq_i32 s53, 0 // (op 1159) 
s_mov_b32 s54, scc // (op 1159) 

s_cmp_eq_i32 s54, 0 // (op 1159) 
s_cselect_b32 s53, s58, s1 // (op 1159) 


s_mul_i32 s1, s51, s56 // (op 1159) 


v_lshl_add_u32 v9, s53, 4, v6 // (op 1159) 


s_sub_i32 s51, s18, s1 // (op 1159) 


s_cmp_eq_i32 s54, 0 // (op 1159) 
s_cselect_b32 s1, s60, s51 // (op 1159) 



v_lshl_add_u32 v6, s1, 6, v5 // (op 1159) 


v_mul_lo_u32 v5, v6, s2 // (op 1159) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 1159) 


s_add_u32 s56, s6, 64 // (op 1347) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1347) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1347) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1347) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1347) 

s_mul_i32 s1, s44, s56 // (op 1347) 
s_mul_i32 s18, s48, s1 // (op 1347) 

s_mul_i32 s1, s55, s18 // (op 1347) 

s_sub_u32 s18, s16, s1 // (op 1347) 

s_mov_b32 s1, s18 // (op 1347) 
s_mul_hi_i32 s51, s1, s15 // (op 1347) 
s_add_i32 s53, s51, s1 // (op 1347) 


s_ashr_i32 s1, s53, 31 // (op 1347) 
s_cmp_eq_i32 s15, 0 // (op 1347) 
s_mov_b32 s51, scc // (op 1347) 
s_cmp_eq_i32 s51, 0 // (op 1347) 
s_cselect_b32 s54, 0, -1 // (op 1347) 

s_and_b32 s51, s46, 31 // (op 1347) 
s_lshl_b32 s58, 1, s51 // (op 1347) 
s_add_i32 s59, s58, s54 // (op 1347) 


s_and_b32 s54, s1, s59 // (op 1347) 


s_add_i32 s1, s53, s54 // (op 1347) 


s_ashr_i32 s53, s1, s51 // (op 1347) 


s_lshl_b32 s1, s46, 24 // (op 1347) 
s_ashr_i32 s51, s1, 31 // (op 1347) 

s_xor_b32 s1, s53, s51 // (op 1347) 

s_sub_i32 s53, s1, s51 // (op 1347) 


s_mul_i32 s1, s45, s44 // (op 1347) 
s_add_u32 s58, s8, 16 // (op 1347) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 1347) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1347) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1347) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1347) 

s_sub_i32 s51, s58, s1 // (op 1347) 

s_mul_i32 s54, s53, s51 // (op 1347) 

s_sub_u32 s51, s18, s54 // (op 1347) 


s_add_u32 s18, s51, s1 // (op 1347) 


s_cmp_eq_i32 s55, 0 // (op 1347) 
s_mov_b32 s1, scc // (op 1347) 

s_cmp_eq_i32 s1, 0 // (op 1347) 
s_cselect_b32 s51, s18, s52 // (op 1347) 


s_mul_i32 s18, s19, s56 // (op 1347) 


v_lshl_add_u32 v5, s51, 4, v12 // (op 1347) 


s_sub_i32 s19, s0, s18 // (op 1347) 


s_cmp_eq_i32 s1, 0 // (op 1347) 
s_cselect_b32 s0, s53, s19 // (op 1347) 



v_lshl_add_u32 v9, s0, 6, v8 // (op 1347) 


v_mul_lo_u32 v8, v9, s4 // (op 1347) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 1347) 


s_mul_i32 s1, s2, s7 // (op 1161) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1161) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1161) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1161) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1161) most significant: sum
s_add_u32 s1, s1, s19 // (op 1161) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1161) 

s_mul_i32 s1, s4, s7 // (op 1349) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 1349) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 1349) least significant
s_mul_hi_u32 s52, s4, s6 // (op 1349) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1349) most significant: sum
s_add_u32 s1, s1, s52 // (op 1349) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 1349) 

s_mov_b32 s56, s20 // (op 1161) reg expression
s_mov_b32 s57, s21 // (op 1161) reg expression
s_mov_b32 s58, s18 // (op 1161) reg expression
s_mov_b32 s59, 131072 // (op 1161) call()

s_mov_b32 s60, s22 // (op 1349) reg expression

s_mov_b32 s61, s23 // (op 1349) reg expression
s_mov_b32 s62, s52 // (op 1349) reg expression
s_mov_b32 s63, 131072 // (op 1349) call()


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
s_mul_hi_i32 s0, s16, s11 // (op 926) 
v_lshrrev_b32 v12, 3, v1 // (op 926) 
s_add_i32 s1, s0, s16 // (op 926) 

v_and_b32 v13, -8, v1 // (op 926) 
s_ashr_i32 s0, s1, 31 // (op 926) 
v_sub_u32 v14, v1, v13 // (op 926) 

s_cmp_eq_i32 s11, 0 // (op 926) 
s_mov_b32 s18, scc // (op 926) 
v_lshlrev_b32 v13, 4, v14 // (op 926) 

s_cmp_eq_i32 s18, 0 // (op 926) 
s_cselect_b32 s19, 0, -1 // (op 926) 

v_mov_b32 v15, 0 // (op 926) convert
v_mov_b32 v14, v13 // (op 926) convert

s_and_b32 s18, s12, 31 // (op 926) 
s_lshl_b32 s20, 1, s18 // (op 926) 
s_add_i32 s21, s20, s19 // (op 926) 


s_and_b32 s19, s0, s21 // (op 926) 


s_add_i32 s0, s1, s19 // (op 926) 


s_ashr_i32 s1, s0, s18 // (op 926) 


s_lshl_b32 s0, s12, 24 // (op 926) 
s_ashr_i32 s18, s0, 31 // (op 926) 

s_xor_b32 s0, s1, s18 // (op 926) 

s_sub_i32 s1, s0, s18 // (op 926) 


s_mul_hi_i32 s0, s1, s13 // (op 926) 
s_add_i32 s18, s0, s1 // (op 926) 

s_ashr_i32 s0, s18, 31 // (op 926) 
s_cmp_eq_i32 s13, 0 // (op 926) 
s_mov_b32 s19, scc // (op 926) 
s_cmp_eq_i32 s19, 0 // (op 926) 
s_cselect_b32 s20, 0, -1 // (op 926) 

s_and_b32 s19, s14, 31 // (op 926) 
s_lshl_b32 s21, 1, s19 // (op 926) 
s_add_i32 s22, s21, s20 // (op 926) 


s_and_b32 s23, s0, s22 // (op 926) 


s_add_i32 s0, s18, s23 // (op 926) 


s_ashr_i32 s18, s0, s19 // (op 926) 


s_lshl_b32 s0, s14, 24 // (op 926) 
s_ashr_i32 s19, s0, 31 // (op 926) 

s_xor_b32 s0, s18, s19 // (op 926) 

s_sub_i32 s18, s0, s19 // (op 926) 


s_mul_i32 s0, s1, s44 // (op 926) 

s_sub_u32 s1, s16, s0 // (op 926) 

s_mul_i32 s0, s18, s44 // (op 926) 
s_add_u32 s0, s0, s1 // (op 926) 


s_mul_hi_i32 s1, s16, s10 // (op 926) 
s_add_i32 s18, s1, s16 // (op 926) 

s_ashr_i32 s1, s18, 31 // (op 926) 
s_cmp_eq_i32 s10, 0 // (op 926) 
s_mov_b32 s19, scc // (op 926) 
s_cmp_eq_i32 s19, 0 // (op 926) 
s_cselect_b32 s20, 0, -1 // (op 926) 

s_and_b32 s19, s47, 31 // (op 926) 
s_lshl_b32 s21, 1, s19 // (op 926) 
s_add_i32 s22, s21, s20 // (op 926) 


s_and_b32 s23, s1, s22 // (op 926) 


s_add_i32 s1, s18, s23 // (op 926) 


s_ashr_i32 s18, s1, s19 // (op 926) 


s_lshl_b32 s1, s47, 24 // (op 926) 
s_ashr_i32 s19, s1, 31 // (op 926) 

s_xor_b32 s1, s18, s19 // (op 926) 

s_sub_i32 s18, s1, s19 // (op 926) 


s_add_u32 s22, s6, 64 // (op 926) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 926) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 926) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 926) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 6 // (op 926) 

s_mul_i32 s1, s44, s22 // (op 926) 

s_mul_i32 s19, s48, s1 // (op 926) 

s_mul_i32 s1, s18, s19 // (op 926) 

s_sub_u32 s19, s16, s1 // (op 926) 

s_mov_b32 s1, s19 // (op 926) 
s_mul_hi_i32 s20, s1, s15 // (op 926) 
s_add_i32 s21, s20, s1 // (op 926) 


s_ashr_i32 s1, s21, 31 // (op 926) 
s_cmp_eq_i32 s15, 0 // (op 926) 
s_mov_b32 s20, scc // (op 926) 
s_cmp_eq_i32 s20, 0 // (op 926) 
s_cselect_b32 s22, 0, -1 // (op 926) 

s_and_b32 s20, s46, 31 // (op 926) 
s_lshl_b32 s23, 1, s20 // (op 926) 
s_add_i32 s49, s23, s22 // (op 926) 


s_and_b32 s22, s1, s49 // (op 926) 


s_add_i32 s1, s21, s22 // (op 926) 


s_ashr_i32 s21, s1, s20 // (op 926) 


s_lshl_b32 s1, s46, 24 // (op 926) 
s_ashr_i32 s20, s1, 31 // (op 926) 

s_xor_b32 s1, s21, s20 // (op 926) 

s_sub_i32 s21, s1, s20 // (op 926) 


s_mul_i32 s1, s45, s44 // (op 926) 
s_add_u32 s22, s8, 16 // (op 926) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 926) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 926) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 926) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 4 // (op 926) 

s_sub_i32 s20, s22, s1 // (op 926) 

s_mul_i32 s22, s21, s20 // (op 926) 


s_sub_u32 s23, s19, s22 // (op 926) 


s_add_u32 s19, s23, s1 // (op 926) 


s_cmp_eq_i32 s18, 0 // (op 926) 
s_mov_b32 s1, scc // (op 926) 

s_cmp_eq_i32 s1, 0 // (op 926) 
s_cselect_b32 s18, s19, s0 // (op 926) 



v_lshl_add_u32 v13, s18, 4, v12 // (op 926) 


v_mul_lo_u32 v17, v13, s25 // (op 926) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 926) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 926) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 926) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 926) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 926) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 926) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 926) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 926) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 926) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 926) 

v_mov_b32 v12, v14 // (op 926) convert

s_mul_i32 s1, s9, 6 // (op 928) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 928) least significant
s_mul_hi_u32 s18, s8, 6 // (op 928) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 928) most significant: sum

s_mul_i32 s19, s24, s1 // (op 928) most significant: low of low * high
s_mul_i32 s20, s25, s0 // (op 928) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 928) least significant
s_mul_hi_u32 s21, s24, s0 // (op 928) most significant: high of low * low
s_add_u32 s19, s19, s20 // (op 928) most significant: sum
s_add_u32 s19, s19, s21 // (op 928) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 928) 

s_mov_b32 s20, s26 // (op 928) reg expression
s_mov_b32 s21, s27 // (op 928) reg expression
s_mov_b32 s22, s0 // (op 928) reg expression
s_mov_b32 s23, 131072 // (op 928) call()

s_mul_hi_i32 s0, s16, s11 // (op 945) 
v_lshrrev_b32 v13, 3, v1 // (op 945) 
s_add_i32 s1, s0, s16 // (op 945) 

v_and_b32 v14, -8, v1 // (op 945) 
s_ashr_i32 s0, s1, 31 // (op 945) 
v_sub_u32 v15, v1, v14 // (op 945) 

s_cmp_eq_i32 s11, 0 // (op 945) 
s_mov_b32 s18, scc // (op 945) 
v_lshlrev_b32 v14, 4, v15 // (op 945) 

s_cmp_eq_i32 s18, 0 // (op 945) 
s_cselect_b32 s19, 0, -1 // (op 945) 

v_mov_b32 v17, 0 // (op 945) convert
v_mov_b32 v16, v14 // (op 945) convert

s_and_b32 s18, s12, 31 // (op 945) 
s_lshl_b32 s49, 1, s18 // (op 945) 
s_add_i32 s50, s49, s19 // (op 945) 


s_and_b32 s19, s0, s50 // (op 945) 


s_add_i32 s0, s1, s19 // (op 945) 


s_ashr_i32 s1, s0, s18 // (op 945) 


s_lshl_b32 s0, s12, 24 // (op 945) 
s_ashr_i32 s18, s0, 31 // (op 945) 

s_xor_b32 s0, s1, s18 // (op 945) 

s_sub_i32 s1, s0, s18 // (op 945) 


s_mul_hi_i32 s0, s1, s13 // (op 945) 
s_add_i32 s18, s0, s1 // (op 945) 

s_ashr_i32 s0, s18, 31 // (op 945) 
s_cmp_eq_i32 s13, 0 // (op 945) 
s_mov_b32 s19, scc // (op 945) 
s_cmp_eq_i32 s19, 0 // (op 945) 
s_cselect_b32 s49, 0, -1 // (op 945) 

s_and_b32 s19, s14, 31 // (op 945) 
s_lshl_b32 s50, 1, s19 // (op 945) 
s_add_i32 s51, s50, s49 // (op 945) 


s_and_b32 s49, s0, s51 // (op 945) 


s_add_i32 s0, s18, s49 // (op 945) 


s_ashr_i32 s18, s0, s19 // (op 945) 


s_lshl_b32 s0, s14, 24 // (op 945) 
s_ashr_i32 s19, s0, 31 // (op 945) 

s_xor_b32 s0, s18, s19 // (op 945) 

s_sub_i32 s18, s0, s19 // (op 945) 


s_mul_i32 s0, s1, s44 // (op 945) 

s_sub_u32 s1, s16, s0 // (op 945) 

s_mul_i32 s0, s18, s44 // (op 945) 
s_add_u32 s0, s0, s1 // (op 945) 


s_mul_hi_i32 s1, s16, s10 // (op 945) 
s_add_i32 s18, s1, s16 // (op 945) 

s_ashr_i32 s1, s18, 31 // (op 945) 
s_cmp_eq_i32 s10, 0 // (op 945) 
s_mov_b32 s19, scc // (op 945) 
s_cmp_eq_i32 s19, 0 // (op 945) 
s_cselect_b32 s49, 0, -1 // (op 945) 

s_and_b32 s19, s47, 31 // (op 945) 
s_lshl_b32 s50, 1, s19 // (op 945) 
s_add_i32 s51, s50, s49 // (op 945) 


s_and_b32 s49, s1, s51 // (op 945) 


s_add_i32 s1, s18, s49 // (op 945) 


s_ashr_i32 s18, s1, s19 // (op 945) 


s_lshl_b32 s1, s47, 24 // (op 945) 
s_ashr_i32 s19, s1, 31 // (op 945) 

s_xor_b32 s1, s18, s19 // (op 945) 

s_sub_i32 s18, s1, s19 // (op 945) 


s_add_u32 s54, s6, 64 // (op 945) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 945) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 945) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 945) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 945) 

s_mul_i32 s1, s44, s50 // (op 945) 

s_mul_i32 s19, s48, s1 // (op 945) 

s_mul_i32 s1, s18, s19 // (op 945) 

s_sub_u32 s19, s16, s1 // (op 945) 

s_mov_b32 s1, s19 // (op 945) 
s_mul_hi_i32 s49, s1, s15 // (op 945) 
s_add_i32 s50, s49, s1 // (op 945) 


s_ashr_i32 s1, s50, 31 // (op 945) 
s_cmp_eq_i32 s15, 0 // (op 945) 
s_mov_b32 s49, scc // (op 945) 
s_cmp_eq_i32 s49, 0 // (op 945) 
s_cselect_b32 s51, 0, -1 // (op 945) 

s_and_b32 s49, s46, 31 // (op 945) 
s_lshl_b32 s52, 1, s49 // (op 945) 
s_add_i32 s53, s52, s51 // (op 945) 


s_and_b32 s51, s1, s53 // (op 945) 


s_add_i32 s1, s50, s51 // (op 945) 


s_ashr_i32 s50, s1, s49 // (op 945) 


s_lshl_b32 s1, s46, 24 // (op 945) 
s_ashr_i32 s49, s1, 31 // (op 945) 

s_xor_b32 s1, s50, s49 // (op 945) 

s_sub_i32 s50, s1, s49 // (op 945) 


s_mul_i32 s1, s45, s44 // (op 945) 
s_add_u32 s54, s8, 16 // (op 945) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 945) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 945) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 945) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 4 // (op 945) 

s_sub_i32 s49, s54, s1 // (op 945) 

s_mul_i32 s51, s50, s49 // (op 945) 


s_sub_u32 s49, s19, s51 // (op 945) 


s_add_u32 s19, s49, s1 // (op 945) 


s_cmp_eq_i32 s18, 0 // (op 945) 
s_mov_b32 s1, scc // (op 945) 

s_cmp_eq_i32 s1, 0 // (op 945) 
s_cselect_b32 s18, s19, s0 // (op 945) 



v_lshl_add_u32 v14, s18, 4, v13 // (op 945) 


v_mul_lo_u32 v19, v14, s25 // (op 945) most significant: low of low * high
v_mul_hi_u32 v13, v14, s24 // (op 945) most significant: high of low * low
v_mul_lo_u32 v18, v14, s24 // (op 945) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 945) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 945) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 945) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 945) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 945) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 945) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 945) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 945) 

v_mov_b32 v13, v14 // (op 945) convert

s_mul_i32 s1, s9, 6 // (op 947) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 947) least significant
s_mul_hi_u32 s18, s8, 6 // (op 947) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 947) most significant: sum

s_mul_i32 s19, s24, s1 // (op 947) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 947) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 947) least significant
s_mul_hi_u32 s50, s24, s0 // (op 947) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 947) most significant: sum
s_add_u32 s19, s19, s50 // (op 947) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 947) 

s_mov_b32 s20, s26 // (op 947) reg expression
s_mov_b32 s21, s27 // (op 947) reg expression
s_mov_b32 s22, s0 // (op 947) reg expression
s_mov_b32 s23, 131072 // (op 947) call()

s_mul_hi_i32 s0, s16, s11 // (op 990) 
v_and_b32 v14, -4, v1 // (op 990) 
s_add_i32 s1, s0, s16 // (op 990) 

v_sub_u32 v15, v1, v14 // (op 990) 

s_ashr_i32 s0, s1, 31 // (op 990) 
v_lshlrev_b32 v14, 4, v15 // (op 990) 

s_cmp_eq_i32 s11, 0 // (op 990) 
s_mov_b32 s18, scc // (op 990) 
v_lshrrev_b32 v15, 2, v1 // (op 990) 
s_cmp_eq_i32 s18, 0 // (op 990) 
s_cselect_b32 s19, 0, -1 // (op 990) 

s_and_b32 s18, s12, 31 // (op 990) 
s_lshl_b32 s49, 1, s18 // (op 990) 
s_add_i32 s50, s49, s19 // (op 990) 


s_and_b32 s19, s0, s50 // (op 990) 


s_add_i32 s0, s1, s19 // (op 990) 


s_ashr_i32 s1, s0, s18 // (op 990) 


s_lshl_b32 s0, s12, 24 // (op 990) 
s_ashr_i32 s18, s0, 31 // (op 990) 

s_xor_b32 s0, s1, s18 // (op 990) 

s_sub_i32 s1, s0, s18 // (op 990) 


s_mul_hi_i32 s0, s1, s13 // (op 990) 
s_add_i32 s18, s0, s1 // (op 990) 

s_ashr_i32 s0, s18, 31 // (op 990) 
s_cmp_eq_i32 s13, 0 // (op 990) 
s_mov_b32 s19, scc // (op 990) 
s_cmp_eq_i32 s19, 0 // (op 990) 
s_cselect_b32 s49, 0, -1 // (op 990) 

s_and_b32 s19, s14, 31 // (op 990) 
s_lshl_b32 s50, 1, s19 // (op 990) 
s_add_i32 s51, s50, s49 // (op 990) 


s_and_b32 s49, s0, s51 // (op 990) 


s_add_i32 s0, s18, s49 // (op 990) 


s_ashr_i32 s18, s0, s19 // (op 990) 


s_lshl_b32 s0, s14, 24 // (op 990) 
s_ashr_i32 s19, s0, 31 // (op 990) 

s_xor_b32 s0, s18, s19 // (op 990) 

s_sub_i32 s18, s0, s19 // (op 990) 


s_add_u32 s54, s6, 64 // (op 990) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 990) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 990) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 990) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 990) 

s_mul_i32 s0, s18, s50 // (op 990) 

s_sub_i32 s49, s1, s0 // (op 990) 


s_mul_hi_i32 s0, s16, s10 // (op 990) 
s_add_i32 s1, s0, s16 // (op 990) 

s_ashr_i32 s0, s1, 31 // (op 990) 
s_cmp_eq_i32 s10, 0 // (op 990) 
s_mov_b32 s18, scc // (op 990) 
s_cmp_eq_i32 s18, 0 // (op 990) 
s_cselect_b32 s19, 0, -1 // (op 990) 

s_and_b32 s18, s47, 31 // (op 990) 
s_lshl_b32 s52, 1, s18 // (op 990) 
s_add_i32 s53, s52, s19 // (op 990) 


s_and_b32 s19, s0, s53 // (op 990) 


s_add_i32 s0, s1, s19 // (op 990) 


s_ashr_i32 s1, s0, s18 // (op 990) 


s_lshl_b32 s0, s47, 24 // (op 990) 
s_ashr_i32 s18, s0, 31 // (op 990) 

s_xor_b32 s0, s1, s18 // (op 990) 

s_sub_i32 s1, s0, s18 // (op 990) 


s_mul_i32 s0, s44, s50 // (op 990) 

s_mul_i32 s18, s48, s0 // (op 990) 

s_mul_i32 s0, s1, s18 // (op 990) 

s_sub_u32 s18, s16, s0 // (op 990) 

s_mul_hi_i32 s0, s18, s15 // (op 990) 
s_add_i32 s19, s0, s18 // (op 990) 


s_ashr_i32 s0, s19, 31 // (op 990) 
s_cmp_eq_i32 s15, 0 // (op 990) 
s_mov_b32 s18, scc // (op 990) 
s_cmp_eq_i32 s18, 0 // (op 990) 
s_cselect_b32 s50, 0, -1 // (op 990) 

s_and_b32 s18, s46, 31 // (op 990) 
s_lshl_b32 s51, 1, s18 // (op 990) 
s_add_i32 s52, s51, s50 // (op 990) 


s_and_b32 s50, s0, s52 // (op 990) 


s_add_i32 s0, s19, s50 // (op 990) 


s_ashr_i32 s19, s0, s18 // (op 990) 


s_lshl_b32 s0, s46, 24 // (op 990) 
s_ashr_i32 s18, s0, 31 // (op 990) 

s_xor_b32 s0, s19, s18 // (op 990) 

s_sub_i32 s19, s0, s18 // (op 990) 


s_cmp_eq_i32 s1, 0 // (op 990) 
s_mov_b32 s0, scc // (op 990) 

s_cmp_eq_i32 s0, 0 // (op 990) 
s_cselect_b32 s1, s19, s49 // (op 990) 



v_lshl_add_u32 v16, s1, 6, v14 // (op 990) 


v_mov_b32 v19, 0 // (op 990) convert
v_mov_b32 v18, v16 // (op 990) convert

v_mul_lo_u32 v17, v15, s29 // (op 990) most significant: low of low * high
v_mul_hi_u32 v14, v15, s28 // (op 990) most significant: high of low * low
v_mul_lo_u32 v16, v15, s28 // (op 990) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 990) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v18 // (op 990) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 990) most significant half



s_mul_i32 s1, s28, s33 // (op 992) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 992) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 992) least significant
s_mul_hi_u32 s19, s28, s32 // (op 992) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 992) most significant: sum
s_add_u32 s1, s1, s19 // (op 992) most significant: sum


s_mov_b32 s52, s30 // (op 992) reg expression
s_mov_b32 s53, s31 // (op 992) reg expression
s_mov_b32 s54, s0 // (op 992) reg expression
s_mov_b32 s55, 131072 // (op 992) call()

s_mul_hi_i32 s0, s16, s11 // (op 1009) 
v_and_b32 v14, -4, v1 // (op 1009) 
s_add_i32 s1, s0, s16 // (op 1009) 

v_sub_u32 v15, v1, v14 // (op 1009) 

s_ashr_i32 s0, s1, 31 // (op 1009) 
v_lshlrev_b32 v14, 4, v15 // (op 1009) 

s_cmp_eq_i32 s11, 0 // (op 1009) 
s_mov_b32 s18, scc // (op 1009) 
v_lshrrev_b32 v15, 2, v1 // (op 1009) 
s_cmp_eq_i32 s18, 0 // (op 1009) 
s_cselect_b32 s19, 0, -1 // (op 1009) 

s_and_b32 s18, s12, 31 // (op 1009) 
s_lshl_b32 s49, 1, s18 // (op 1009) 
s_add_i32 s50, s49, s19 // (op 1009) 


s_and_b32 s19, s0, s50 // (op 1009) 


s_add_i32 s0, s1, s19 // (op 1009) 


s_ashr_i32 s1, s0, s18 // (op 1009) 


s_lshl_b32 s0, s12, 24 // (op 1009) 
s_ashr_i32 s18, s0, 31 // (op 1009) 

s_xor_b32 s0, s1, s18 // (op 1009) 

s_sub_i32 s1, s0, s18 // (op 1009) 


s_mul_hi_i32 s0, s1, s13 // (op 1009) 
s_add_i32 s18, s0, s1 // (op 1009) 

s_ashr_i32 s0, s18, 31 // (op 1009) 
s_cmp_eq_i32 s13, 0 // (op 1009) 
s_mov_b32 s19, scc // (op 1009) 
s_cmp_eq_i32 s19, 0 // (op 1009) 
s_cselect_b32 s49, 0, -1 // (op 1009) 

s_and_b32 s19, s14, 31 // (op 1009) 
s_lshl_b32 s50, 1, s19 // (op 1009) 
s_add_i32 s51, s50, s49 // (op 1009) 


s_and_b32 s49, s0, s51 // (op 1009) 


s_add_i32 s0, s18, s49 // (op 1009) 


s_ashr_i32 s18, s0, s19 // (op 1009) 


s_lshl_b32 s0, s14, 24 // (op 1009) 
s_ashr_i32 s19, s0, 31 // (op 1009) 

s_xor_b32 s0, s18, s19 // (op 1009) 

s_sub_i32 s18, s0, s19 // (op 1009) 


s_add_u32 s50, s6, 64 // (op 1009) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1009) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1009) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1009) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1009) 

s_mul_i32 s0, s18, s50 // (op 1009) 

s_sub_i32 s49, s1, s0 // (op 1009) 


s_mul_hi_i32 s0, s16, s10 // (op 1009) 
s_add_i32 s1, s0, s16 // (op 1009) 

s_ashr_i32 s0, s1, 31 // (op 1009) 
s_cmp_eq_i32 s10, 0 // (op 1009) 
s_mov_b32 s18, scc // (op 1009) 
s_cmp_eq_i32 s18, 0 // (op 1009) 
s_cselect_b32 s19, 0, -1 // (op 1009) 

s_and_b32 s18, s47, 31 // (op 1009) 
s_lshl_b32 s64, 1, s18 // (op 1009) 
s_add_i32 s65, s64, s19 // (op 1009) 


s_and_b32 s19, s0, s65 // (op 1009) 


s_add_i32 s0, s1, s19 // (op 1009) 


s_ashr_i32 s1, s0, s18 // (op 1009) 


s_lshl_b32 s0, s47, 24 // (op 1009) 
s_ashr_i32 s18, s0, 31 // (op 1009) 

s_xor_b32 s0, s1, s18 // (op 1009) 

s_sub_i32 s1, s0, s18 // (op 1009) 


s_mul_i32 s0, s44, s50 // (op 1009) 

s_mul_i32 s18, s48, s0 // (op 1009) 

s_mul_i32 s0, s1, s18 // (op 1009) 

s_sub_u32 s18, s16, s0 // (op 1009) 

s_mul_hi_i32 s0, s18, s15 // (op 1009) 
s_add_i32 s19, s0, s18 // (op 1009) 


s_ashr_i32 s0, s19, 31 // (op 1009) 
s_cmp_eq_i32 s15, 0 // (op 1009) 
s_mov_b32 s18, scc // (op 1009) 
s_cmp_eq_i32 s18, 0 // (op 1009) 
s_cselect_b32 s50, 0, -1 // (op 1009) 

s_and_b32 s18, s46, 31 // (op 1009) 
s_lshl_b32 s51, 1, s18 // (op 1009) 
s_add_i32 s64, s51, s50 // (op 1009) 


s_and_b32 s50, s0, s64 // (op 1009) 


s_add_i32 s0, s19, s50 // (op 1009) 


s_ashr_i32 s19, s0, s18 // (op 1009) 


s_lshl_b32 s0, s46, 24 // (op 1009) 
s_ashr_i32 s18, s0, 31 // (op 1009) 

s_xor_b32 s0, s19, s18 // (op 1009) 

s_sub_i32 s19, s0, s18 // (op 1009) 


s_cmp_eq_i32 s1, 0 // (op 1009) 
s_mov_b32 s0, scc // (op 1009) 

s_cmp_eq_i32 s0, 0 // (op 1009) 
s_cselect_b32 s1, s19, s49 // (op 1009) 



v_lshl_add_u32 v18, s1, 6, v14 // (op 1009) 


v_mov_b32 v21, 0 // (op 1009) convert
v_mov_b32 v20, v18 // (op 1009) convert

v_mul_lo_u32 v19, v15, s29 // (op 1009) most significant: low of low * high
v_mul_hi_u32 v14, v15, s28 // (op 1009) most significant: high of low * low
v_mul_lo_u32 v18, v15, s28 // (op 1009) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1009) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v20 // (op 1009) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 1009) most significant half



s_mul_i32 s1, s28, s33 // (op 1011) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 1011) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1011) least significant
s_mul_hi_u32 s19, s28, s32 // (op 1011) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1011) most significant: sum
s_add_u32 s1, s1, s19 // (op 1011) most significant: sum


s_mov_b32 s52, s30 // (op 1011) reg expression
s_mov_b32 s53, s31 // (op 1011) reg expression
s_mov_b32 s54, s0 // (op 1011) reg expression
s_mov_b32 s55, 131072 // (op 1011) call()

s_mul_hi_i32 s0, s16, s11 // (op 1054) 
v_lshrrev_b32 v14, 7, v1 // (op 1054) 
s_add_i32 s1, s0, s16 // (op 1054) 

v_and_b32 v15, 15, v1 // (op 1054) 
s_ashr_i32 s0, s1, 31 // (op 1054) 
v_lshl_add_u32 v20, v14, 4, v15 // (op 1054) 


s_cmp_eq_i32 s11, 0 // (op 1054) 
s_mov_b32 s18, scc // (op 1054) 
v_and_b32 v14, 63, v1 // (op 1054) 
s_cmp_eq_i32 s18, 0 // (op 1054) 
s_cselect_b32 s19, 0, -1 // (op 1054) 

v_lshrrev_b32 v15, 4, v14 // (op 1054) 

s_and_b32 s18, s12, 31 // (op 1054) 
s_lshl_b32 s49, 1, s18 // (op 1054) 
s_add_i32 s50, s49, s19 // (op 1054) 


s_and_b32 s19, s0, s50 // (op 1054) 


s_add_i32 s0, s1, s19 // (op 1054) 


s_ashr_i32 s1, s0, s18 // (op 1054) 


s_lshl_b32 s0, s12, 24 // (op 1054) 
s_ashr_i32 s18, s0, 31 // (op 1054) 

s_xor_b32 s0, s1, s18 // (op 1054) 

s_sub_i32 s1, s0, s18 // (op 1054) 


s_mul_hi_i32 s0, s1, s13 // (op 1054) 
s_add_i32 s18, s0, s1 // (op 1054) 

s_ashr_i32 s0, s18, 31 // (op 1054) 
s_cmp_eq_i32 s13, 0 // (op 1054) 
s_mov_b32 s19, scc // (op 1054) 
s_cmp_eq_i32 s19, 0 // (op 1054) 
s_cselect_b32 s49, 0, -1 // (op 1054) 

s_and_b32 s19, s14, 31 // (op 1054) 
s_lshl_b32 s50, 1, s19 // (op 1054) 
s_add_i32 s51, s50, s49 // (op 1054) 


s_and_b32 s49, s0, s51 // (op 1054) 


s_add_i32 s0, s18, s49 // (op 1054) 


s_ashr_i32 s18, s0, s19 // (op 1054) 


s_lshl_b32 s0, s14, 24 // (op 1054) 
s_ashr_i32 s19, s0, 31 // (op 1054) 

s_xor_b32 s0, s18, s19 // (op 1054) 

s_sub_i32 s18, s0, s19 // (op 1054) 


s_mul_i32 s0, s1, s44 // (op 1054) 

s_sub_u32 s1, s16, s0 // (op 1054) 

s_mul_i32 s0, s18, s44 // (op 1054) 
s_add_u32 s0, s0, s1 // (op 1054) 


s_mul_hi_i32 s1, s16, s10 // (op 1054) 
s_add_i32 s18, s1, s16 // (op 1054) 

s_ashr_i32 s1, s18, 31 // (op 1054) 
s_cmp_eq_i32 s10, 0 // (op 1054) 
s_mov_b32 s19, scc // (op 1054) 
s_cmp_eq_i32 s19, 0 // (op 1054) 
s_cselect_b32 s49, 0, -1 // (op 1054) 

s_and_b32 s19, s47, 31 // (op 1054) 
s_lshl_b32 s50, 1, s19 // (op 1054) 
s_add_i32 s51, s50, s49 // (op 1054) 


s_and_b32 s49, s1, s51 // (op 1054) 


s_add_i32 s1, s18, s49 // (op 1054) 


s_ashr_i32 s18, s1, s19 // (op 1054) 


s_lshl_b32 s1, s47, 24 // (op 1054) 
s_ashr_i32 s19, s1, 31 // (op 1054) 

s_xor_b32 s1, s18, s19 // (op 1054) 

s_sub_i32 s18, s1, s19 // (op 1054) 


s_add_u32 s50, s6, 64 // (op 1054) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1054) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1054) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1054) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1054) 

s_mul_i32 s1, s44, s50 // (op 1054) 

s_mul_i32 s19, s48, s1 // (op 1054) 

s_mul_i32 s1, s18, s19 // (op 1054) 

s_sub_u32 s19, s16, s1 // (op 1054) 

s_mov_b32 s1, s19 // (op 1054) 
s_mul_hi_i32 s49, s1, s15 // (op 1054) 
s_add_i32 s50, s49, s1 // (op 1054) 


s_ashr_i32 s1, s50, 31 // (op 1054) 
s_cmp_eq_i32 s15, 0 // (op 1054) 
s_mov_b32 s49, scc // (op 1054) 
s_cmp_eq_i32 s49, 0 // (op 1054) 
s_cselect_b32 s51, 0, -1 // (op 1054) 

s_and_b32 s49, s46, 31 // (op 1054) 
s_lshl_b32 s64, 1, s49 // (op 1054) 
s_add_i32 s65, s64, s51 // (op 1054) 


s_and_b32 s51, s1, s65 // (op 1054) 


s_add_i32 s1, s50, s51 // (op 1054) 


s_ashr_i32 s50, s1, s49 // (op 1054) 


s_lshl_b32 s1, s46, 24 // (op 1054) 
s_ashr_i32 s49, s1, 31 // (op 1054) 

s_xor_b32 s1, s50, s49 // (op 1054) 

s_sub_i32 s50, s1, s49 // (op 1054) 


s_mul_i32 s1, s45, s44 // (op 1054) 
s_add_u32 s64, s8, 16 // (op 1054) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 1054) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1054) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1054) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 1054) 

s_sub_i32 s49, s64, s1 // (op 1054) 

s_mul_i32 s51, s50, s49 // (op 1054) 


s_sub_u32 s49, s19, s51 // (op 1054) 


s_add_u32 s19, s49, s1 // (op 1054) 


s_cmp_eq_i32 s18, 0 // (op 1054) 
s_mov_b32 s1, scc // (op 1054) 

s_cmp_eq_i32 s1, 0 // (op 1054) 
s_cselect_b32 s18, s19, s0 // (op 1054) 



v_lshl_add_u32 v14, s18, 4, v20 // (op 1054) 


v_mul_lo_u32 v20, v14, s34 // (op 1054) 
v_add_u32 v20, v20, v15 // (op 1054) 


s_mul_i32 s1, s34, s9 // (op 1056) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1056) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1056) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1056) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1056) most significant: sum
s_add_u32 s1, s1, s19 // (op 1056) most significant: sum


s_mov_b32 s64, s36 // (op 1056) reg expression
s_mov_b32 s65, s37 // (op 1056) reg expression
s_mov_b32 s66, s0 // (op 1056) reg expression
s_mov_b32 s67, 131072 // (op 1056) call()

s_mul_hi_i32 s0, s16, s11 // (op 1073) 
v_lshrrev_b32 v14, 7, v1 // (op 1073) 
s_add_i32 s1, s0, s16 // (op 1073) 

v_and_b32 v15, 15, v1 // (op 1073) 
s_ashr_i32 s0, s1, 31 // (op 1073) 
v_lshl_add_u32 v21, v14, 4, v15 // (op 1073) 


s_cmp_eq_i32 s11, 0 // (op 1073) 
s_mov_b32 s18, scc // (op 1073) 
v_and_b32 v14, 63, v1 // (op 1073) 
s_cmp_eq_i32 s18, 0 // (op 1073) 
s_cselect_b32 s19, 0, -1 // (op 1073) 

v_lshrrev_b32 v15, 4, v14 // (op 1073) 

s_and_b32 s18, s12, 31 // (op 1073) 
s_lshl_b32 s49, 1, s18 // (op 1073) 
s_add_i32 s50, s49, s19 // (op 1073) 


s_and_b32 s19, s0, s50 // (op 1073) 


s_add_i32 s0, s1, s19 // (op 1073) 


s_ashr_i32 s1, s0, s18 // (op 1073) 


s_lshl_b32 s0, s12, 24 // (op 1073) 
s_ashr_i32 s18, s0, 31 // (op 1073) 

s_xor_b32 s0, s1, s18 // (op 1073) 

s_sub_i32 s1, s0, s18 // (op 1073) 


s_mul_hi_i32 s0, s1, s13 // (op 1073) 
s_add_i32 s18, s0, s1 // (op 1073) 

s_ashr_i32 s0, s18, 31 // (op 1073) 
s_cmp_eq_i32 s13, 0 // (op 1073) 
s_mov_b32 s19, scc // (op 1073) 
s_cmp_eq_i32 s19, 0 // (op 1073) 
s_cselect_b32 s49, 0, -1 // (op 1073) 

s_and_b32 s19, s14, 31 // (op 1073) 
s_lshl_b32 s50, 1, s19 // (op 1073) 
s_add_i32 s51, s50, s49 // (op 1073) 


s_and_b32 s49, s0, s51 // (op 1073) 


s_add_i32 s0, s18, s49 // (op 1073) 


s_ashr_i32 s18, s0, s19 // (op 1073) 


s_lshl_b32 s0, s14, 24 // (op 1073) 
s_ashr_i32 s19, s0, 31 // (op 1073) 

s_xor_b32 s0, s18, s19 // (op 1073) 

s_sub_i32 s18, s0, s19 // (op 1073) 


s_mul_i32 s0, s1, s44 // (op 1073) 

s_sub_u32 s1, s16, s0 // (op 1073) 

s_mul_i32 s0, s18, s44 // (op 1073) 
s_add_u32 s0, s0, s1 // (op 1073) 


s_mul_hi_i32 s1, s16, s10 // (op 1073) 
s_add_i32 s18, s1, s16 // (op 1073) 

s_ashr_i32 s1, s18, 31 // (op 1073) 
s_cmp_eq_i32 s10, 0 // (op 1073) 
s_mov_b32 s19, scc // (op 1073) 
s_cmp_eq_i32 s19, 0 // (op 1073) 
s_cselect_b32 s49, 0, -1 // (op 1073) 

s_and_b32 s19, s47, 31 // (op 1073) 
s_lshl_b32 s50, 1, s19 // (op 1073) 
s_add_i32 s51, s50, s49 // (op 1073) 


s_and_b32 s49, s1, s51 // (op 1073) 


s_add_i32 s1, s18, s49 // (op 1073) 


s_ashr_i32 s18, s1, s19 // (op 1073) 


s_lshl_b32 s1, s47, 24 // (op 1073) 
s_ashr_i32 s19, s1, 31 // (op 1073) 

s_xor_b32 s1, s18, s19 // (op 1073) 

s_sub_i32 s18, s1, s19 // (op 1073) 


s_add_u32 s50, s6, 64 // (op 1073) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1073) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1073) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1073) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1073) 

s_mul_i32 s1, s44, s50 // (op 1073) 

s_mul_i32 s19, s48, s1 // (op 1073) 

s_mul_i32 s1, s18, s19 // (op 1073) 

s_sub_u32 s19, s16, s1 // (op 1073) 

s_mov_b32 s1, s19 // (op 1073) 
s_mul_hi_i32 s49, s1, s15 // (op 1073) 
s_add_i32 s50, s49, s1 // (op 1073) 


s_ashr_i32 s1, s50, 31 // (op 1073) 
s_cmp_eq_i32 s15, 0 // (op 1073) 
s_mov_b32 s49, scc // (op 1073) 
s_cmp_eq_i32 s49, 0 // (op 1073) 
s_cselect_b32 s51, 0, -1 // (op 1073) 

s_and_b32 s49, s46, 31 // (op 1073) 
s_lshl_b32 s68, 1, s49 // (op 1073) 
s_add_i32 s69, s68, s51 // (op 1073) 


s_and_b32 s51, s1, s69 // (op 1073) 


s_add_i32 s1, s50, s51 // (op 1073) 


s_ashr_i32 s50, s1, s49 // (op 1073) 


s_lshl_b32 s1, s46, 24 // (op 1073) 
s_ashr_i32 s49, s1, 31 // (op 1073) 

s_xor_b32 s1, s50, s49 // (op 1073) 

s_sub_i32 s50, s1, s49 // (op 1073) 


s_mul_i32 s1, s45, s44 // (op 1073) 
s_add_u32 s68, s8, 16 // (op 1073) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 1073) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1073) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1073) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 1073) 

s_sub_i32 s49, s68, s1 // (op 1073) 

s_mul_i32 s51, s50, s49 // (op 1073) 


s_sub_u32 s49, s19, s51 // (op 1073) 


s_add_u32 s19, s49, s1 // (op 1073) 


s_cmp_eq_i32 s18, 0 // (op 1073) 
s_mov_b32 s1, scc // (op 1073) 

s_cmp_eq_i32 s1, 0 // (op 1073) 
s_cselect_b32 s18, s19, s0 // (op 1073) 



v_lshl_add_u32 v14, s18, 4, v21 // (op 1073) 


v_mul_lo_u32 v21, v14, s34 // (op 1073) 
v_add_u32 v21, v21, v15 // (op 1073) 


s_mul_i32 s1, s34, s9 // (op 1075) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1075) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1075) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1075) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1075) most significant: sum
s_add_u32 s1, s1, s19 // (op 1075) most significant: sum


s_mov_b32 s64, s36 // (op 1075) reg expression
s_mov_b32 s65, s37 // (op 1075) reg expression
s_mov_b32 s66, s0 // (op 1075) reg expression
s_mov_b32 s67, 131072 // (op 1075) call()

s_mul_hi_i32 s0, s16, s11 // (op 1118) 
v_and_b32 v14, -64, v1 // (op 1118) 
s_add_i32 s1, s0, s16 // (op 1118) 

v_sub_u32 v15, v1, v14 // (op 1118) 

s_ashr_i32 s0, s1, 31 // (op 1118) 
v_lshrrev_b32 v14, 6, v1 // (op 1118) 
s_cmp_eq_i32 s11, 0 // (op 1118) 
s_mov_b32 s18, scc // (op 1118) 
v_lshlrev_b32 v22, 1, v14 // (op 1118) 

s_cmp_eq_i32 s18, 0 // (op 1118) 
s_cselect_b32 s19, 0, -1 // (op 1118) 

s_and_b32 s18, s12, 31 // (op 1118) 
s_lshl_b32 s49, 1, s18 // (op 1118) 
s_add_i32 s50, s49, s19 // (op 1118) 


s_and_b32 s19, s0, s50 // (op 1118) 


s_add_i32 s0, s1, s19 // (op 1118) 


s_ashr_i32 s1, s0, s18 // (op 1118) 


s_lshl_b32 s0, s12, 24 // (op 1118) 
s_ashr_i32 s18, s0, 31 // (op 1118) 

s_xor_b32 s0, s1, s18 // (op 1118) 

s_sub_i32 s1, s0, s18 // (op 1118) 


s_mul_hi_i32 s0, s1, s13 // (op 1118) 
s_add_i32 s18, s0, s1 // (op 1118) 

s_ashr_i32 s0, s18, 31 // (op 1118) 
s_cmp_eq_i32 s13, 0 // (op 1118) 
s_mov_b32 s19, scc // (op 1118) 
s_cmp_eq_i32 s19, 0 // (op 1118) 
s_cselect_b32 s49, 0, -1 // (op 1118) 

s_and_b32 s19, s14, 31 // (op 1118) 
s_lshl_b32 s50, 1, s19 // (op 1118) 
s_add_i32 s51, s50, s49 // (op 1118) 


s_and_b32 s49, s0, s51 // (op 1118) 


s_add_i32 s0, s18, s49 // (op 1118) 


s_ashr_i32 s18, s0, s19 // (op 1118) 


s_lshl_b32 s0, s14, 24 // (op 1118) 
s_ashr_i32 s19, s0, 31 // (op 1118) 

s_xor_b32 s0, s18, s19 // (op 1118) 

s_sub_i32 s18, s0, s19 // (op 1118) 


s_add_u32 s50, s6, 64 // (op 1118) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1118) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1118) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1118) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1118) 

s_mul_i32 s0, s18, s50 // (op 1118) 

s_sub_i32 s49, s1, s0 // (op 1118) 


s_mul_hi_i32 s0, s16, s10 // (op 1118) 
s_add_i32 s1, s0, s16 // (op 1118) 

s_ashr_i32 s0, s1, 31 // (op 1118) 
s_cmp_eq_i32 s10, 0 // (op 1118) 
s_mov_b32 s18, scc // (op 1118) 
s_cmp_eq_i32 s18, 0 // (op 1118) 
s_cselect_b32 s19, 0, -1 // (op 1118) 

s_and_b32 s18, s47, 31 // (op 1118) 
s_lshl_b32 s68, 1, s18 // (op 1118) 
s_add_i32 s69, s68, s19 // (op 1118) 


s_and_b32 s19, s0, s69 // (op 1118) 


s_add_i32 s0, s1, s19 // (op 1118) 


s_ashr_i32 s1, s0, s18 // (op 1118) 


s_lshl_b32 s0, s47, 24 // (op 1118) 
s_ashr_i32 s18, s0, 31 // (op 1118) 

s_xor_b32 s0, s1, s18 // (op 1118) 

s_sub_i32 s1, s0, s18 // (op 1118) 


s_mul_i32 s0, s44, s50 // (op 1118) 

s_mul_i32 s18, s48, s0 // (op 1118) 

s_mul_i32 s0, s1, s18 // (op 1118) 

s_sub_u32 s18, s16, s0 // (op 1118) 

s_mul_hi_i32 s0, s18, s15 // (op 1118) 
s_add_i32 s19, s0, s18 // (op 1118) 


s_ashr_i32 s0, s19, 31 // (op 1118) 
s_cmp_eq_i32 s15, 0 // (op 1118) 
s_mov_b32 s18, scc // (op 1118) 
s_cmp_eq_i32 s18, 0 // (op 1118) 
s_cselect_b32 s50, 0, -1 // (op 1118) 

s_and_b32 s18, s46, 31 // (op 1118) 
s_lshl_b32 s51, 1, s18 // (op 1118) 
s_add_i32 s68, s51, s50 // (op 1118) 


s_and_b32 s50, s0, s68 // (op 1118) 


s_add_i32 s0, s19, s50 // (op 1118) 


s_ashr_i32 s19, s0, s18 // (op 1118) 


s_lshl_b32 s0, s46, 24 // (op 1118) 
s_ashr_i32 s18, s0, 31 // (op 1118) 

s_xor_b32 s0, s19, s18 // (op 1118) 

s_sub_i32 s19, s0, s18 // (op 1118) 


s_cmp_eq_i32 s1, 0 // (op 1118) 
s_mov_b32 s0, scc // (op 1118) 

s_cmp_eq_i32 s0, 0 // (op 1118) 
s_cselect_b32 s1, s19, s49 // (op 1118) 



v_lshl_add_u32 v14, s1, 6, v15 // (op 1118) 


v_mul_lo_u32 v15, v22, s38 // (op 1118) 
v_add_u32 v15, v15, v14 // (op 1118) 


s_mul_i32 s1, s38, s43 // (op 1120) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1120) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1120) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1120) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1120) most significant: sum
s_add_u32 s1, s1, s19 // (op 1120) most significant: sum


s_mov_b32 s68, s40 // (op 1120) reg expression
s_mov_b32 s69, s41 // (op 1120) reg expression
s_mov_b32 s70, s0 // (op 1120) reg expression
s_mov_b32 s71, 131072 // (op 1120) call()

s_mul_hi_i32 s0, s16, s11 // (op 1137) 
v_and_b32 v14, -64, v1 // (op 1137) 
s_add_i32 s1, s0, s16 // (op 1137) 

v_sub_u32 v22, v1, v14 // (op 1137) 

s_ashr_i32 s0, s1, 31 // (op 1137) 
v_lshrrev_b32 v14, 6, v1 // (op 1137) 
s_cmp_eq_i32 s11, 0 // (op 1137) 
s_mov_b32 s18, scc // (op 1137) 
v_lshlrev_b32 v23, 1, v14 // (op 1137) 

s_cmp_eq_i32 s18, 0 // (op 1137) 
s_cselect_b32 s19, 0, -1 // (op 1137) 

s_and_b32 s18, s12, 31 // (op 1137) 
s_lshl_b32 s49, 1, s18 // (op 1137) 
s_add_i32 s50, s49, s19 // (op 1137) 


s_and_b32 s19, s0, s50 // (op 1137) 


s_add_i32 s0, s1, s19 // (op 1137) 


s_ashr_i32 s1, s0, s18 // (op 1137) 


s_lshl_b32 s0, s12, 24 // (op 1137) 
s_ashr_i32 s18, s0, 31 // (op 1137) 

s_xor_b32 s0, s1, s18 // (op 1137) 

s_sub_i32 s1, s0, s18 // (op 1137) 


s_mul_hi_i32 s0, s1, s13 // (op 1137) 
s_add_i32 s18, s0, s1 // (op 1137) 

s_ashr_i32 s0, s18, 31 // (op 1137) 
s_cmp_eq_i32 s13, 0 // (op 1137) 
s_mov_b32 s19, scc // (op 1137) 
s_cmp_eq_i32 s19, 0 // (op 1137) 
s_cselect_b32 s49, 0, -1 // (op 1137) 

s_and_b32 s19, s14, 31 // (op 1137) 
s_lshl_b32 s50, 1, s19 // (op 1137) 
s_add_i32 s51, s50, s49 // (op 1137) 


s_and_b32 s49, s0, s51 // (op 1137) 


s_add_i32 s0, s18, s49 // (op 1137) 


s_ashr_i32 s18, s0, s19 // (op 1137) 


s_lshl_b32 s0, s14, 24 // (op 1137) 
s_ashr_i32 s19, s0, 31 // (op 1137) 

s_xor_b32 s0, s18, s19 // (op 1137) 

s_sub_i32 s18, s0, s19 // (op 1137) 


s_add_u32 s50, s6, 64 // (op 1137) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1137) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1137) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1137) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1137) 

s_mul_i32 s0, s18, s50 // (op 1137) 

s_sub_i32 s49, s1, s0 // (op 1137) 


s_mul_hi_i32 s0, s16, s10 // (op 1137) 
s_add_i32 s1, s0, s16 // (op 1137) 

s_ashr_i32 s0, s1, 31 // (op 1137) 
s_cmp_eq_i32 s10, 0 // (op 1137) 
s_mov_b32 s18, scc // (op 1137) 
s_cmp_eq_i32 s18, 0 // (op 1137) 
s_cselect_b32 s19, 0, -1 // (op 1137) 

s_and_b32 s18, s47, 31 // (op 1137) 
s_lshl_b32 s72, 1, s18 // (op 1137) 
s_add_i32 s73, s72, s19 // (op 1137) 


s_and_b32 s19, s0, s73 // (op 1137) 


s_add_i32 s0, s1, s19 // (op 1137) 


s_ashr_i32 s1, s0, s18 // (op 1137) 


s_lshl_b32 s0, s47, 24 // (op 1137) 
s_ashr_i32 s18, s0, 31 // (op 1137) 

s_xor_b32 s0, s1, s18 // (op 1137) 

s_sub_i32 s1, s0, s18 // (op 1137) 


s_mul_i32 s0, s44, s50 // (op 1137) 

s_mul_i32 s18, s48, s0 // (op 1137) 

s_mul_i32 s0, s1, s18 // (op 1137) 

s_sub_u32 s18, s16, s0 // (op 1137) 

s_mul_hi_i32 s0, s18, s15 // (op 1137) 
s_add_i32 s19, s0, s18 // (op 1137) 


s_ashr_i32 s0, s19, 31 // (op 1137) 
s_cmp_eq_i32 s15, 0 // (op 1137) 
s_mov_b32 s18, scc // (op 1137) 
s_cmp_eq_i32 s18, 0 // (op 1137) 
s_cselect_b32 s50, 0, -1 // (op 1137) 

s_and_b32 s18, s46, 31 // (op 1137) 
s_lshl_b32 s51, 1, s18 // (op 1137) 
s_add_i32 s72, s51, s50 // (op 1137) 


s_and_b32 s50, s0, s72 // (op 1137) 


s_add_i32 s0, s19, s50 // (op 1137) 


s_ashr_i32 s19, s0, s18 // (op 1137) 


s_lshl_b32 s0, s46, 24 // (op 1137) 
s_ashr_i32 s18, s0, 31 // (op 1137) 

s_xor_b32 s0, s19, s18 // (op 1137) 

s_sub_i32 s19, s0, s18 // (op 1137) 


s_cmp_eq_i32 s1, 0 // (op 1137) 
s_mov_b32 s0, scc // (op 1137) 

s_cmp_eq_i32 s0, 0 // (op 1137) 
s_cselect_b32 s1, s19, s49 // (op 1137) 



v_lshl_add_u32 v14, s1, 6, v22 // (op 1137) 


v_mul_lo_u32 v22, v23, s38 // (op 1137) 
v_add_u32 v22, v22, v14 // (op 1137) 


s_mul_i32 s1, s38, s43 // (op 1139) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1139) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1139) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1139) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1139) most significant: sum
s_add_u32 s1, s1, s19 // (op 1139) most significant: sum


s_mov_b32 s68, s40 // (op 1139) reg expression
s_mov_b32 s69, s41 // (op 1139) reg expression
s_mov_b32 s70, s0 // (op 1139) reg expression
s_mov_b32 s71, 131072 // (op 1139) call()




buffer_load_ubyte v14, v20, s[64:67], 0 offen offset: 0 // (op 769) Load value
v_and_b32 v23, -4, v1 // (op 1256) 
v_sub_u32 v24, v1, v23 // (op 1256) 

v_lshlrev_b32 v23, 4, v24 // (op 1256) 

v_lshrrev_b32 v24, 2, v1 // (op 1256) 
v_lshl_add_u32 v25, v24, 6, v23 // (op 1256) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 1256) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 5 // (op 881) 
v_mov_b64 v[24:25], s[18:19] // (op 881) convert


v_mov_b64 v[26:27], v[16:17] // (op 881) reg expression
s_mov_b32 m0, s0 // (op 881) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 881) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 881) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 881) most significant half

s_add_u32 m0, m0, 2048 // (op 881) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 881) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 881) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 881) most significant half

s_add_u32 m0, m0, 2048 // (op 881) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 881) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 881) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 881) most significant half

s_add_u32 m0, m0, 2048 // (op 881) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 881) Load value direct to lds




v_mov_b64 v[24:25], s[38:39] // (op 776) convert

v_mov_b32 v23, v15 // (op 776) reg expression
buffer_load_ubyte v26, v23, s[68:71], 0 offen offset: 0 // (op 776) Load value
v_add_u32 v23, v23, v24 // (op 776) 
buffer_load_ubyte v27, v23, s[68:71], 0 offen offset: 0 // (op 776) Load value





buffer_load_dwordx3 v[28:30], v12, s[20:23], 0 offen // (op 779) Load value



buffer_load_ubyte v23, v20, s[64:67], 0 offen offset:4 // (op 783) Load value

v_and_b32 v20, -4, v1 // (op 1400) 
v_sub_u32 v24, v1, v20 // (op 1400) 

v_lshlrev_b32 v20, 4, v24 // (op 1400) 

v_lshrrev_b32 v24, 2, v1 // (op 1400) 
v_lshl_add_u32 v25, v24, 6, v20 // (op 1400) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 1400) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 5 // (op 886) 
v_mov_b64 v[24:25], s[18:19] // (op 886) convert


s_lshl_b64 s[18:19], s[28:29], 7 // (op 886) 
v_mov_b32 v20, s18 // (op 886) 
v_mov_b32 v31, s19 // (op 886) 
v_add_co_u32 v32, s[50:51], v16, v20 // (op 886) least significant half
v_addc_co_u32 v33, s[50:51], v17, v31, s[50:51] // (op 886) most significant half




s_add_u32 s1, 8192, s0 // (op 886) 
s_mov_b32 m0, s1 // (op 886) reg expression
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 886) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v24 // (op 886) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 886) most significant half

s_add_u32 m0, m0, 2048 // (op 886) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 886) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v24 // (op 886) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 886) most significant half

s_add_u32 m0, m0, 2048 // (op 886) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 886) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v24 // (op 886) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 886) most significant half

s_add_u32 m0, m0, 2048 // (op 886) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 886) Load value direct to lds






v_mov_b64 v[16:17], s[38:39] // (op 790) convert

s_lshl_b32 s0, s38, 2 // (op 790) 
v_add_u32 v20, v15, s0 // (op 790) 

buffer_load_ubyte v24, v20, s[68:71], 0 offen offset: 0 // (op 790) Load value
v_add_u32 v20, v20, v16 // (op 790) 
buffer_load_ubyte v25, v20, s[68:71], 0 offen offset: 0 // (op 790) Load value






buffer_load_dwordx3 v[32:34], v12, s[20:23], 0 offen offset:96 // (op 793) Load value

v_and_b32 v12, -64, v1 // (op 1314) 
v_sub_u32 v20, v1, v12 // (op 1314) 

v_lshrrev_b32 v12, 6, v1 // (op 1314) 
v_lshl_add_u32 v31, v12, 7, v20 // (op 1314) 





s_waitcnt vmcnt(10)
ds_write_b8 v31, v26 offset:16384 // (op 796) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v31, v27 offset:16448 // (op 796) Store local data 

v_and_b32 v12, -8, v1 // (op 1204) 
v_sub_u32 v20, v1, v12 // (op 1204) 

v_lshlrev_b32 v12, 4, v20 // (op 1204) 

v_lshrrev_b32 v20, 3, v1 // (op 1204) 
v_lshl_add_u32 v31, v20, 7, v12 // (op 1204) 


v_mul_lo_u32 v12, v31, 6 // (op 1204) 

v_lshrrev_b32 v20, 3, v12 // (op 1204) 




s_waitcnt vmcnt(8)
ds_write_b96 v20, v[28:30] offset:16640 // (op 797) Store local data 

v_and_b32 v12, -8, v1 // (op 1175) 
v_sub_u32 v20, v1, v12 // (op 1175) 

v_lshlrev_b32 v12, 4, v20 // (op 1175) 

v_lshrrev_b32 v20, 3, v1 // (op 1175) 
v_lshl_add_u32 v31, v20, 7, v12 // (op 1175) 


v_mul_lo_u32 v12, v31, 6 // (op 1175) 

v_lshrrev_b32 v20, 3, v12 // (op 1175) 

v_and_b32 v12, -4, v1 // (op 1228) 
v_sub_u32 v31, v1, v12 // (op 1228) 

v_lshlrev_b32 v12, 4, v31 // (op 1228) 

v_lshrrev_b32 v31, 2, v1 // (op 1228) 
v_lshl_add_u32 v15, v31, 6, v12 // (op 1228) 


s_nop 0
v_readfirstlane_b32 s0, v15 // (op 1228) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1285) 
v_sub_u32 v31, v1, v12 // (op 1285) 

v_lshrrev_b32 v12, 6, v1 // (op 1285) 
v_lshl_add_u32 v15, v12, 7, v31 // (op 1285) 


v_and_b32 v12, -8, v1 // (op 1361) 
v_sub_u32 v31, v1, v12 // (op 1361) 

v_lshlrev_b32 v12, 4, v31 // (op 1361) 

v_lshrrev_b32 v31, 3, v1 // (op 1361) 
v_lshl_add_u32 v16, v31, 7, v12 // (op 1361) 


v_mul_lo_u32 v12, v16, 6 // (op 1361) 

v_lshrrev_b32 v31, 3, v12 // (op 1361) 

v_and_b32 v12, -4, v1 // (op 1385) 
v_sub_u32 v16, v1, v12 // (op 1385) 

v_lshlrev_b32 v12, 4, v16 // (op 1385) 

v_lshrrev_b32 v16, 2, v1 // (op 1385) 
v_lshl_add_u32 v17, v16, 6, v12 // (op 1385) 


s_nop 0
v_readfirstlane_b32 s1, v17 // (op 1385) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1429) 
v_sub_u32 v16, v1, v12 // (op 1429) 

v_lshrrev_b32 v12, 6, v1 // (op 1429) 
v_lshl_add_u32 v17, v12, 7, v16 // (op 1429) 


s_mov_b32 s18, 0 // (op 312) call()
s_ashr_i64 s[50:51], s[32:33], 7 // (op 311) 
s_mov_b32 s19, s50 // (op 311) convert

s_and_b32 s49, s19, 4294967294 // (op 311) 

s_cmp_lt_u32 s18, s49 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_311 // (op 311) Condition: Top (jump to RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_311 if false)
RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_311:
s_add_u32 s18, s18, 2 // (op 313) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1487) 



ds_read_b128 v[36:39], v3 offset:16640 // (op 67) Load local data 
ds_read_b64 v[40:41], v3 offset:16656 // (op 67) Load local data 
s_barrier  // (op 1500) 



ds_read_b64_tr_b8 v[42:43], v7 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:512 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:4096 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:4608 // (op 72) Transpose load local data 
s_barrier  // (op 1513) 


ds_read_u8 v12, v0 offset:16384 // (op 81) Load local data 



ds_read_b64_tr_b8 v[50:51], v7 offset:32 // (op 343) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v7 offset:544 // (op 343) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:49], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 84) 
ds_read_b64_tr_b8 v[54:55], v7 offset:4128 // (op 343) Transpose load local data 


ds_read_b64_tr_b8 v[56:57], v7 offset:4640 // (op 343) Transpose load local data 


ds_read_u8 v12, v0 offset:16416 // (op 346) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[50:57], a[4:7], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 347) 





v_mov_b32 v12, v21 // (op 77) 

buffer_load_ubyte v14, v12, s[64:67], 0 offen offset:8 // (op 77) Load value
s_barrier  // (op 1503) 

s_lshl_b64 s[50:51], s[28:29], 5 // (op 864) 
v_mov_b64 v[36:37], s[50:51] // (op 864) convert


v_mov_b64 v[38:39], v[18:19] // (op 864) 
s_lshl_b64 s[50:51], s[28:29], 7 // (op 864) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 864) 

v_mov_b32 v16, s72 // (op 864) 
v_mov_b32 v35, s73 // (op 864) 
v_add_co_u32 v40, s[50:51], v38, v16 // (op 864) least significant half
v_addc_co_u32 v41, s[50:51], v39, v35, s[50:51] // (op 864) most significant half




s_mov_b32 m0, s0 // (op 864) reg expression
buffer_load_dwordx4 v40, s[52:55], 0 offen offset: 0 lds // (op 864) Load value direct to lds
v_add_co_u32 v40, s[50:51], v40, v36 // (op 864) least significant half
v_addc_co_u32 v41, s[50:51], 0, v41, s[50:51] // (op 864) most significant half

s_add_u32 m0, m0, 2048 // (op 864) 
buffer_load_dwordx4 v40, s[52:55], 0 offen offset: 0 lds // (op 864) Load value direct to lds
v_add_co_u32 v40, s[50:51], v40, v36 // (op 864) least significant half
v_addc_co_u32 v41, s[50:51], 0, v41, s[50:51] // (op 864) most significant half

s_add_u32 m0, m0, 2048 // (op 864) 
buffer_load_dwordx4 v40, s[52:55], 0 offen offset: 0 lds // (op 864) Load value direct to lds
v_add_co_u32 v40, s[50:51], v40, v36 // (op 864) least significant half
v_addc_co_u32 v41, s[50:51], 0, v41, s[50:51] // (op 864) most significant half

s_add_u32 m0, m0, 2048 // (op 864) 
buffer_load_dwordx4 v40, s[52:55], 0 offen offset: 0 lds // (op 864) Load value direct to lds



v_mov_b64 v[36:37], s[38:39] // (op 21) convert

v_mov_b32 v16, v22 // (op 21) 
s_lshl_b32 s19, s38, 3 // (op 21) 
v_add_u32 v35, v16, s19 // (op 21) 

buffer_load_ubyte v26, v35, s[68:71], 0 offen offset: 0 // (op 21) Load value
v_add_u32 v35, v35, v36 // (op 21) 
buffer_load_ubyte v27, v35, s[68:71], 0 offen offset: 0 // (op 21) Load value




v_mov_b32 v35, v13 // (op 4) 

buffer_load_dwordx3 v[28:30], v35, s[20:23], 0 offen offset:192 // (op 4) Load value
s_barrier  // (op 1542) 



s_waitcnt vmcnt(10)
ds_write_b8 v17, v24 offset:18176 // (op 240) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v17, v25 offset:18240 // (op 240) Store local data 
s_barrier  // (op 1530) 



s_waitcnt vmcnt(8)
ds_write_b96 v31, v[32:34] offset:18432 // (op 229) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1527) 



ds_read_b128 v[40:43], v4 offset:18432 // (op 231) Load local data 
ds_read_b64 v[44:45], v4 offset:18448 // (op 231) Load local data 
s_barrier  // (op 1533) 
s_barrier  // (op 1577) 



ds_read_b64_tr_b8 v[46:47], v11 offset:8192 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v11 offset:8704 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v11 offset:12288 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v11 offset:12800 // (op 233) Transpose load local data 
s_barrier  // (op 1539) 


ds_read_u8 v36, v10 offset:18176 // (op 237) Load local data 



ds_read_b64_tr_b8 v[54:55], v11 offset:8224 // (op 352) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v11 offset:8736 // (op 352) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:45], v[46:53], a[0:3], v23, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 238) 
ds_read_b64_tr_b8 v[58:59], v11 offset:12320 // (op 352) Transpose load local data 


ds_read_b64_tr_b8 v[60:61], v11 offset:12832 // (op 352) Transpose load local data 


ds_read_u8 v36, v10 offset:18208 // (op 355) Load local data 
s_barrier  // (op 1613) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:45], v[54:61], a[4:7], v23, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 348) 






buffer_load_ubyte v23, v12, s[64:67], 0 offen offset:12 // (op 235) Load value

s_barrier  // (op 1536) 
v_add_u32 v21, 8, v21 // (op 1065) 

s_lshl_b64 s[50:51], s[28:29], 5 // (op 876) 
v_mov_b64 v[36:37], s[50:51] // (op 876) convert


s_lshl_b64 s[50:51], s[28:29], 7 // (op 876) 
s_mul_i32 s73, 3, s51 // (op 876) most significant: low of low * high
s_mul_i32 s72, 3, s50 // (op 876) least significant
s_mul_hi_u32 s19, 3, s50 // (op 876) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 876) most significant: sum


v_mov_b32 v12, s72 // (op 876) 
v_mov_b32 v40, s73 // (op 876) 
v_add_co_u32 v42, s[50:51], v38, v12 // (op 876) least significant half
v_addc_co_u32 v43, s[50:51], v39, v40, s[50:51] // (op 876) most significant half




s_add_u32 s19, 8192, s1 // (op 876) 
s_mov_b32 m0, s19 // (op 876) reg expression
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 876) Load value direct to lds
v_add_co_u32 v42, s[50:51], v42, v36 // (op 876) least significant half
v_addc_co_u32 v43, s[50:51], 0, v43, s[50:51] // (op 876) most significant half

s_add_u32 m0, m0, 2048 // (op 876) 
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 876) Load value direct to lds
v_add_co_u32 v42, s[50:51], v42, v36 // (op 876) least significant half
v_addc_co_u32 v43, s[50:51], 0, v43, s[50:51] // (op 876) most significant half

s_add_u32 m0, m0, 2048 // (op 876) 
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 876) Load value direct to lds
v_add_co_u32 v42, s[50:51], v42, v36 // (op 876) least significant half
v_addc_co_u32 v43, s[50:51], 0, v43, s[50:51] // (op 876) most significant half

s_add_u32 m0, m0, 2048 // (op 876) 
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 876) Load value direct to lds





v_mov_b64 v[36:37], s[38:39] // (op 242) convert

s_lshl_b32 s19, s38, 2 // (op 242) 
s_mul_i32 s49, 3, s19 // (op 242) 

v_add_u32 v12, v16, s49 // (op 242) 

s_lshl_b64 s[50:51], s[28:29], 7 // (op 1001) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 1001) 

v_mov_b32 v38, s72 // (op 1001) 
v_mov_b32 v39, s73 // (op 1001) 
v_add_co_u32 v18, s[50:51], v18, v38 // (op 1001) least significant half
v_addc_co_u32 v19, s[50:51], v19, v39, s[50:51] // (op 1001) most significant half




buffer_load_ubyte v24, v12, s[68:71], 0 offen offset: 0 // (op 242) Load value
v_add_u32 v12, v12, v36 // (op 242) 
buffer_load_ubyte v25, v12, s[68:71], 0 offen offset: 0 // (op 242) Load value






s_lshl_b32 s19, s38, 3 // (op 1129) 
v_add_u32 v22, v22, s19 // (op 1129) 

buffer_load_dwordx3 v[32:34], v35, s[20:23], 0 offen offset:288 // (op 228) Load value

s_barrier  // (op 1516) 
v_add_u32 v13, 192, v13 // (op 937) 



s_waitcnt vmcnt(10)
ds_write_b8 v15, v26 offset:16384 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v15, v27 offset:16448 // (op 26) Store local data 
s_barrier  // (op 1490) 



s_waitcnt vmcnt(8)
ds_write_b96 v20, v[28:30] offset:16640 // (op 6) Store local data 
s_ashr_i64 s[50:51], s[32:33], 7 // (op 311) 
s_mov_b32 s19, s50 // (op 311) convert

s_and_b32 s49, s19, 4294967294 // (op 311) 

s_cmp_lt_u32 s18, s49 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_311 // (op 311) Condition: Bottom (jump to RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_311 if true)
RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_311:














s_barrier  // (op 1595) 
s_mul_hi_i32 s0, s16, s11 // (op 899) 
v_lshrrev_b32 v12, 3, v1 // (op 899) 
s_add_i32 s1, s0, s16 // (op 899) 

v_and_b32 v13, -8, v1 // (op 899) 
s_ashr_i32 s0, s1, 31 // (op 899) 
v_sub_u32 v14, v1, v13 // (op 899) 

s_cmp_eq_i32 s11, 0 // (op 899) 
s_mov_b32 s18, scc // (op 899) 
v_lshlrev_b32 v13, 4, v14 // (op 899) 

s_cmp_eq_i32 s18, 0 // (op 899) 
s_cselect_b32 s19, 0, -1 // (op 899) 

s_and_b32 s18, s12, 31 // (op 899) 
s_lshl_b32 s49, 1, s18 // (op 899) 
s_add_i32 s50, s49, s19 // (op 899) 


s_and_b32 s19, s0, s50 // (op 899) 


s_add_i32 s0, s1, s19 // (op 899) 


s_ashr_i32 s1, s0, s18 // (op 899) 


s_lshl_b32 s0, s12, 24 // (op 899) 
s_ashr_i32 s18, s0, 31 // (op 899) 

s_xor_b32 s0, s1, s18 // (op 899) 

s_sub_i32 s1, s0, s18 // (op 899) 


s_mul_hi_i32 s0, s1, s13 // (op 899) 
s_add_i32 s18, s0, s1 // (op 899) 

s_ashr_i32 s0, s18, 31 // (op 899) 
s_cmp_eq_i32 s13, 0 // (op 899) 
s_mov_b32 s19, scc // (op 899) 
s_cmp_eq_i32 s19, 0 // (op 899) 
s_cselect_b32 s49, 0, -1 // (op 899) 

s_and_b32 s19, s14, 31 // (op 899) 
s_lshl_b32 s50, 1, s19 // (op 899) 
s_add_i32 s51, s50, s49 // (op 899) 


s_and_b32 s49, s0, s51 // (op 899) 


s_add_i32 s0, s18, s49 // (op 899) 


s_ashr_i32 s18, s0, s19 // (op 899) 


s_lshl_b32 s0, s14, 24 // (op 899) 
s_ashr_i32 s19, s0, 31 // (op 899) 

s_xor_b32 s0, s18, s19 // (op 899) 

s_sub_i32 s18, s0, s19 // (op 899) 


s_mul_i32 s0, s1, s44 // (op 899) 

s_sub_u32 s1, s16, s0 // (op 899) 

s_mul_i32 s0, s18, s44 // (op 899) 
s_add_u32 s0, s0, s1 // (op 899) 


s_mul_hi_i32 s1, s16, s10 // (op 899) 
s_add_i32 s18, s1, s16 // (op 899) 

s_ashr_i32 s1, s18, 31 // (op 899) 
s_cmp_eq_i32 s10, 0 // (op 899) 
s_mov_b32 s19, scc // (op 899) 
s_cmp_eq_i32 s19, 0 // (op 899) 
s_cselect_b32 s49, 0, -1 // (op 899) 

s_and_b32 s19, s47, 31 // (op 899) 
s_lshl_b32 s50, 1, s19 // (op 899) 
s_add_i32 s51, s50, s49 // (op 899) 


s_and_b32 s49, s1, s51 // (op 899) 


s_add_i32 s1, s18, s49 // (op 899) 


s_ashr_i32 s18, s1, s19 // (op 899) 


s_lshl_b32 s1, s47, 24 // (op 899) 
s_ashr_i32 s19, s1, 31 // (op 899) 

s_xor_b32 s1, s18, s19 // (op 899) 

s_sub_i32 s18, s1, s19 // (op 899) 


s_add_u32 s50, s6, 64 // (op 899) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 899) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 899) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 899) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 899) 

s_mul_i32 s1, s44, s50 // (op 899) 

s_mul_i32 s19, s48, s1 // (op 899) 

s_mul_i32 s1, s18, s19 // (op 899) 

s_sub_u32 s19, s16, s1 // (op 899) 

s_mov_b32 s1, s19 // (op 899) 
s_mul_hi_i32 s49, s1, s15 // (op 899) 
s_add_i32 s50, s49, s1 // (op 899) 


s_ashr_i32 s1, s50, 31 // (op 899) 
s_cmp_eq_i32 s15, 0 // (op 899) 
s_mov_b32 s49, scc // (op 899) 
s_cmp_eq_i32 s49, 0 // (op 899) 
s_cselect_b32 s51, 0, -1 // (op 899) 

s_and_b32 s49, s46, 31 // (op 899) 
s_lshl_b32 s72, 1, s49 // (op 899) 
s_add_i32 s73, s72, s51 // (op 899) 


s_and_b32 s51, s1, s73 // (op 899) 


s_add_i32 s1, s50, s51 // (op 899) 


s_ashr_i32 s50, s1, s49 // (op 899) 


s_lshl_b32 s1, s46, 24 // (op 899) 
s_ashr_i32 s49, s1, 31 // (op 899) 

s_xor_b32 s1, s50, s49 // (op 899) 

s_sub_i32 s50, s1, s49 // (op 899) 


s_mul_i32 s1, s45, s44 // (op 899) 
s_add_u32 s72, s8, 16 // (op 899) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 899) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 899) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 899) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 899) 

s_sub_i32 s49, s72, s1 // (op 899) 

s_mul_i32 s51, s50, s49 // (op 899) 


s_sub_u32 s49, s19, s51 // (op 899) 


s_add_u32 s19, s49, s1 // (op 899) 


s_cmp_eq_i32 s18, 0 // (op 899) 
s_mov_b32 s1, scc // (op 899) 

s_cmp_eq_i32 s1, 0 // (op 899) 
s_cselect_b32 s18, s19, s0 // (op 899) 



v_lshl_add_u32 v14, s18, 4, v12 // (op 899) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 899) 
s_lshr_b32 s18, s0, 1 // (op 899) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 899) 


v_mov_b32 v27, 0 // (op 899) convert
v_mov_b32 v26, v12 // (op 899) convert

v_mul_lo_u32 v13, v14, s25 // (op 899) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 899) most significant: high of low * low
v_mul_lo_u32 v12, v14, s24 // (op 899) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 899) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v26 // (op 899) least significant half
v_addc_co_u32 v13, s[0:1], v13, v27, s[0:1] // (op 899) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 899) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 899) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 899) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 899) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 899) 

v_mov_b32 v14, v12 // (op 899) convert

s_mul_i32 s1, s9, 6 // (op 901) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 901) least significant
s_mul_hi_u32 s18, s8, 6 // (op 901) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 901) most significant: sum

s_mul_i32 s19, s24, s1 // (op 901) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 901) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 901) least significant
s_mul_hi_u32 s50, s24, s0 // (op 901) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 901) most significant: sum
s_add_u32 s19, s19, s50 // (op 901) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 901) 

s_mov_b32 s20, s26 // (op 901) reg expression
s_mov_b32 s21, s27 // (op 901) reg expression
s_mov_b32 s22, s0 // (op 901) reg expression
s_mov_b32 s23, 131072 // (op 901) call()

s_mul_hi_i32 s0, s16, s11 // (op 967) 
v_and_b32 v12, -4, v1 // (op 967) 
s_add_i32 s1, s0, s16 // (op 967) 

v_sub_u32 v13, v1, v12 // (op 967) 

s_ashr_i32 s0, s1, 31 // (op 967) 
v_lshlrev_b32 v12, 4, v13 // (op 967) 

s_cmp_eq_i32 s11, 0 // (op 967) 
s_mov_b32 s18, scc // (op 967) 
v_lshrrev_b32 v13, 2, v1 // (op 967) 
s_cmp_eq_i32 s18, 0 // (op 967) 
s_cselect_b32 s19, 0, -1 // (op 967) 

s_and_b32 s18, s12, 31 // (op 967) 
s_lshl_b32 s49, 1, s18 // (op 967) 
s_add_i32 s50, s49, s19 // (op 967) 


s_and_b32 s19, s0, s50 // (op 967) 


s_add_i32 s0, s1, s19 // (op 967) 


s_ashr_i32 s1, s0, s18 // (op 967) 


s_lshl_b32 s0, s12, 24 // (op 967) 
s_ashr_i32 s18, s0, 31 // (op 967) 

s_xor_b32 s0, s1, s18 // (op 967) 

s_sub_i32 s1, s0, s18 // (op 967) 


s_mul_hi_i32 s0, s1, s13 // (op 967) 
s_add_i32 s18, s0, s1 // (op 967) 

s_ashr_i32 s0, s18, 31 // (op 967) 
s_cmp_eq_i32 s13, 0 // (op 967) 
s_mov_b32 s19, scc // (op 967) 
s_cmp_eq_i32 s19, 0 // (op 967) 
s_cselect_b32 s49, 0, -1 // (op 967) 

s_and_b32 s19, s14, 31 // (op 967) 
s_lshl_b32 s50, 1, s19 // (op 967) 
s_add_i32 s51, s50, s49 // (op 967) 


s_and_b32 s49, s0, s51 // (op 967) 


s_add_i32 s0, s18, s49 // (op 967) 


s_ashr_i32 s18, s0, s19 // (op 967) 


s_lshl_b32 s0, s14, 24 // (op 967) 
s_ashr_i32 s19, s0, 31 // (op 967) 

s_xor_b32 s0, s18, s19 // (op 967) 

s_sub_i32 s18, s0, s19 // (op 967) 


s_add_u32 s50, s6, 64 // (op 967) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 967) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 967) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 967) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 967) 

s_mul_i32 s0, s18, s50 // (op 967) 

s_sub_i32 s49, s1, s0 // (op 967) 


s_mul_hi_i32 s0, s16, s10 // (op 967) 
s_add_i32 s1, s0, s16 // (op 967) 

s_ashr_i32 s0, s1, 31 // (op 967) 
s_cmp_eq_i32 s10, 0 // (op 967) 
s_mov_b32 s18, scc // (op 967) 
s_cmp_eq_i32 s18, 0 // (op 967) 
s_cselect_b32 s19, 0, -1 // (op 967) 

s_and_b32 s18, s47, 31 // (op 967) 
s_lshl_b32 s72, 1, s18 // (op 967) 
s_add_i32 s73, s72, s19 // (op 967) 


s_and_b32 s19, s0, s73 // (op 967) 


s_add_i32 s0, s1, s19 // (op 967) 


s_ashr_i32 s1, s0, s18 // (op 967) 


s_lshl_b32 s0, s47, 24 // (op 967) 
s_ashr_i32 s18, s0, 31 // (op 967) 

s_xor_b32 s0, s1, s18 // (op 967) 

s_sub_i32 s1, s0, s18 // (op 967) 


s_mul_i32 s0, s44, s50 // (op 967) 

s_mul_i32 s18, s48, s0 // (op 967) 

s_mul_i32 s0, s1, s18 // (op 967) 

s_sub_u32 s18, s16, s0 // (op 967) 

s_mul_hi_i32 s0, s18, s15 // (op 967) 
s_add_i32 s19, s0, s18 // (op 967) 


s_ashr_i32 s0, s19, 31 // (op 967) 
s_cmp_eq_i32 s15, 0 // (op 967) 
s_mov_b32 s18, scc // (op 967) 
s_cmp_eq_i32 s18, 0 // (op 967) 
s_cselect_b32 s50, 0, -1 // (op 967) 

s_and_b32 s18, s46, 31 // (op 967) 
s_lshl_b32 s51, 1, s18 // (op 967) 
s_add_i32 s72, s51, s50 // (op 967) 


s_and_b32 s50, s0, s72 // (op 967) 


s_add_i32 s0, s19, s50 // (op 967) 


s_ashr_i32 s19, s0, s18 // (op 967) 


s_lshl_b32 s0, s46, 24 // (op 967) 
s_ashr_i32 s18, s0, 31 // (op 967) 

s_xor_b32 s0, s19, s18 // (op 967) 

s_sub_i32 s19, s0, s18 // (op 967) 


s_cmp_eq_i32 s1, 0 // (op 967) 
s_mov_b32 s0, scc // (op 967) 

s_cmp_eq_i32 s0, 0 // (op 967) 
s_cselect_b32 s1, s19, s49 // (op 967) 



v_lshl_add_u32 v15, s1, 6, v12 // (op 967) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 967) 
v_mov_b32 v17, 0 // (op 967) convert
v_mov_b32 v16, v15 // (op 967) convert

s_lshr_b32 s18, s0, 1 // (op 967) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 967) 


v_mul_lo_u32 v19, v12, s29 // (op 967) most significant: low of low * high
v_mul_hi_u32 v13, v12, s28 // (op 967) most significant: high of low * low
v_mul_lo_u32 v18, v12, s28 // (op 967) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 967) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 967) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 967) most significant half



s_mul_i32 s1, s28, s33 // (op 969) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 969) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 969) least significant
s_mul_hi_u32 s19, s28, s32 // (op 969) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 969) most significant: sum
s_add_u32 s1, s1, s19 // (op 969) most significant: sum


s_mov_b32 s52, s30 // (op 969) reg expression
s_mov_b32 s53, s31 // (op 969) reg expression
s_mov_b32 s54, s0 // (op 969) reg expression
s_mov_b32 s55, 131072 // (op 969) call()

s_mul_hi_i32 s0, s16, s11 // (op 1031) 
v_lshrrev_b32 v12, 7, v1 // (op 1031) 
s_add_i32 s1, s0, s16 // (op 1031) 

v_and_b32 v13, 15, v1 // (op 1031) 
s_ashr_i32 s0, s1, 31 // (op 1031) 
v_lshl_add_u32 v15, v12, 4, v13 // (op 1031) 


s_cmp_eq_i32 s11, 0 // (op 1031) 
s_mov_b32 s18, scc // (op 1031) 
v_and_b32 v12, 63, v1 // (op 1031) 
s_cmp_eq_i32 s18, 0 // (op 1031) 
s_cselect_b32 s19, 0, -1 // (op 1031) 

v_lshrrev_b32 v13, 4, v12 // (op 1031) 

s_and_b32 s18, s12, 31 // (op 1031) 
s_lshl_b32 s49, 1, s18 // (op 1031) 
s_add_i32 s50, s49, s19 // (op 1031) 


s_and_b32 s19, s0, s50 // (op 1031) 


s_add_i32 s0, s1, s19 // (op 1031) 


s_ashr_i32 s1, s0, s18 // (op 1031) 


s_lshl_b32 s0, s12, 24 // (op 1031) 
s_ashr_i32 s18, s0, 31 // (op 1031) 

s_xor_b32 s0, s1, s18 // (op 1031) 

s_sub_i32 s1, s0, s18 // (op 1031) 


s_mul_hi_i32 s0, s1, s13 // (op 1031) 
s_add_i32 s18, s0, s1 // (op 1031) 

s_ashr_i32 s0, s18, 31 // (op 1031) 
s_cmp_eq_i32 s13, 0 // (op 1031) 
s_mov_b32 s19, scc // (op 1031) 
s_cmp_eq_i32 s19, 0 // (op 1031) 
s_cselect_b32 s49, 0, -1 // (op 1031) 

s_and_b32 s19, s14, 31 // (op 1031) 
s_lshl_b32 s50, 1, s19 // (op 1031) 
s_add_i32 s51, s50, s49 // (op 1031) 


s_and_b32 s49, s0, s51 // (op 1031) 


s_add_i32 s0, s18, s49 // (op 1031) 


s_ashr_i32 s18, s0, s19 // (op 1031) 


s_lshl_b32 s0, s14, 24 // (op 1031) 
s_ashr_i32 s19, s0, 31 // (op 1031) 

s_xor_b32 s0, s18, s19 // (op 1031) 

s_sub_i32 s18, s0, s19 // (op 1031) 


s_mul_i32 s0, s1, s44 // (op 1031) 

s_sub_u32 s1, s16, s0 // (op 1031) 

s_mul_i32 s0, s18, s44 // (op 1031) 
s_add_u32 s0, s0, s1 // (op 1031) 


s_mul_hi_i32 s1, s16, s10 // (op 1031) 
s_add_i32 s18, s1, s16 // (op 1031) 

s_ashr_i32 s1, s18, 31 // (op 1031) 
s_cmp_eq_i32 s10, 0 // (op 1031) 
s_mov_b32 s19, scc // (op 1031) 
s_cmp_eq_i32 s19, 0 // (op 1031) 
s_cselect_b32 s49, 0, -1 // (op 1031) 

s_and_b32 s19, s47, 31 // (op 1031) 
s_lshl_b32 s50, 1, s19 // (op 1031) 
s_add_i32 s51, s50, s49 // (op 1031) 


s_and_b32 s49, s1, s51 // (op 1031) 


s_add_i32 s1, s18, s49 // (op 1031) 


s_ashr_i32 s18, s1, s19 // (op 1031) 


s_lshl_b32 s1, s47, 24 // (op 1031) 
s_ashr_i32 s19, s1, 31 // (op 1031) 

s_xor_b32 s1, s18, s19 // (op 1031) 

s_sub_i32 s18, s1, s19 // (op 1031) 


s_add_u32 s50, s6, 64 // (op 1031) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1031) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1031) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1031) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1031) 

s_mul_i32 s1, s44, s50 // (op 1031) 

s_mul_i32 s19, s48, s1 // (op 1031) 

s_mul_i32 s1, s18, s19 // (op 1031) 

s_sub_u32 s19, s16, s1 // (op 1031) 

s_mov_b32 s1, s19 // (op 1031) 
s_mul_hi_i32 s49, s1, s15 // (op 1031) 
s_add_i32 s50, s49, s1 // (op 1031) 


s_ashr_i32 s1, s50, 31 // (op 1031) 
s_cmp_eq_i32 s15, 0 // (op 1031) 
s_mov_b32 s49, scc // (op 1031) 
s_cmp_eq_i32 s49, 0 // (op 1031) 
s_cselect_b32 s51, 0, -1 // (op 1031) 

s_and_b32 s49, s46, 31 // (op 1031) 
s_lshl_b32 s72, 1, s49 // (op 1031) 
s_add_i32 s73, s72, s51 // (op 1031) 


s_and_b32 s51, s1, s73 // (op 1031) 


s_add_i32 s1, s50, s51 // (op 1031) 


s_ashr_i32 s50, s1, s49 // (op 1031) 


s_lshl_b32 s1, s46, 24 // (op 1031) 
s_ashr_i32 s49, s1, 31 // (op 1031) 

s_xor_b32 s1, s50, s49 // (op 1031) 

s_sub_i32 s50, s1, s49 // (op 1031) 


s_mul_i32 s1, s45, s44 // (op 1031) 
s_add_u32 s72, s8, 16 // (op 1031) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 1031) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1031) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1031) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 1031) 

s_sub_i32 s49, s72, s1 // (op 1031) 

s_mul_i32 s51, s50, s49 // (op 1031) 


s_sub_u32 s49, s19, s51 // (op 1031) 


s_add_u32 s19, s49, s1 // (op 1031) 


s_cmp_eq_i32 s18, 0 // (op 1031) 
s_mov_b32 s1, scc // (op 1031) 

s_cmp_eq_i32 s1, 0 // (op 1031) 
s_cselect_b32 s18, s19, s0 // (op 1031) 



v_lshl_add_u32 v12, s18, 4, v15 // (op 1031) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1031) 
s_lshr_b32 s18, s0, 1 // (op 1031) 

v_lshl_add_u32 v15, s18, 3, v13 // (op 1031) 


v_mul_lo_u32 v13, v12, s34 // (op 1031) 
v_add_u32 v13, v13, v15 // (op 1031) 


s_mul_i32 s1, s34, s9 // (op 1033) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1033) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1033) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1033) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1033) most significant: sum
s_add_u32 s1, s1, s19 // (op 1033) most significant: sum


s_mov_b32 s64, s36 // (op 1033) reg expression
s_mov_b32 s65, s37 // (op 1033) reg expression
s_mov_b32 s66, s0 // (op 1033) reg expression
s_mov_b32 s67, 131072 // (op 1033) call()

s_mul_hi_i32 s0, s16, s11 // (op 1095) 
v_and_b32 v12, -64, v1 // (op 1095) 
s_add_i32 s1, s0, s16 // (op 1095) 

v_sub_u32 v15, v1, v12 // (op 1095) 

s_ashr_i32 s0, s1, 31 // (op 1095) 
v_lshrrev_b32 v12, 6, v1 // (op 1095) 
s_cmp_eq_i32 s11, 0 // (op 1095) 
s_mov_b32 s18, scc // (op 1095) 
v_lshlrev_b32 v16, 1, v12 // (op 1095) 

s_cmp_eq_i32 s18, 0 // (op 1095) 
s_cselect_b32 s19, 0, -1 // (op 1095) 

s_and_b32 s18, s12, 31 // (op 1095) 
s_lshl_b32 s49, 1, s18 // (op 1095) 
s_add_i32 s50, s49, s19 // (op 1095) 


s_and_b32 s19, s0, s50 // (op 1095) 


s_add_i32 s0, s1, s19 // (op 1095) 


s_ashr_i32 s1, s0, s18 // (op 1095) 


s_lshl_b32 s0, s12, 24 // (op 1095) 
s_ashr_i32 s18, s0, 31 // (op 1095) 

s_xor_b32 s0, s1, s18 // (op 1095) 

s_sub_i32 s1, s0, s18 // (op 1095) 


s_mul_hi_i32 s0, s1, s13 // (op 1095) 
s_add_i32 s18, s0, s1 // (op 1095) 

s_ashr_i32 s0, s18, 31 // (op 1095) 
s_cmp_eq_i32 s13, 0 // (op 1095) 
s_mov_b32 s19, scc // (op 1095) 
s_cmp_eq_i32 s19, 0 // (op 1095) 
s_cselect_b32 s49, 0, -1 // (op 1095) 

s_and_b32 s19, s14, 31 // (op 1095) 
s_lshl_b32 s50, 1, s19 // (op 1095) 
s_add_i32 s51, s50, s49 // (op 1095) 


s_and_b32 s49, s0, s51 // (op 1095) 


s_add_i32 s0, s18, s49 // (op 1095) 


s_ashr_i32 s18, s0, s19 // (op 1095) 


s_lshl_b32 s0, s14, 24 // (op 1095) 
s_ashr_i32 s19, s0, 31 // (op 1095) 

s_xor_b32 s0, s18, s19 // (op 1095) 

s_sub_i32 s18, s0, s19 // (op 1095) 


s_add_u32 s50, s6, 64 // (op 1095) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1095) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1095) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1095) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1095) 

s_mul_i32 s0, s18, s50 // (op 1095) 

s_sub_i32 s49, s1, s0 // (op 1095) 


s_mul_hi_i32 s0, s16, s10 // (op 1095) 
s_add_i32 s1, s0, s16 // (op 1095) 

s_ashr_i32 s0, s1, 31 // (op 1095) 
s_cmp_eq_i32 s10, 0 // (op 1095) 
s_mov_b32 s18, scc // (op 1095) 
s_cmp_eq_i32 s18, 0 // (op 1095) 
s_cselect_b32 s19, 0, -1 // (op 1095) 

s_and_b32 s18, s47, 31 // (op 1095) 
s_lshl_b32 s72, 1, s18 // (op 1095) 
s_add_i32 s73, s72, s19 // (op 1095) 


s_and_b32 s19, s0, s73 // (op 1095) 


s_add_i32 s0, s1, s19 // (op 1095) 


s_ashr_i32 s1, s0, s18 // (op 1095) 


s_lshl_b32 s0, s47, 24 // (op 1095) 
s_ashr_i32 s18, s0, 31 // (op 1095) 

s_xor_b32 s0, s1, s18 // (op 1095) 

s_sub_i32 s1, s0, s18 // (op 1095) 


s_mul_i32 s0, s44, s50 // (op 1095) 

s_mul_i32 s18, s48, s0 // (op 1095) 

s_mul_i32 s0, s1, s18 // (op 1095) 

s_sub_u32 s18, s16, s0 // (op 1095) 

s_mul_hi_i32 s0, s18, s15 // (op 1095) 
s_add_i32 s19, s0, s18 // (op 1095) 


s_ashr_i32 s0, s19, 31 // (op 1095) 
s_cmp_eq_i32 s15, 0 // (op 1095) 
s_mov_b32 s18, scc // (op 1095) 
s_cmp_eq_i32 s18, 0 // (op 1095) 
s_cselect_b32 s50, 0, -1 // (op 1095) 

s_and_b32 s18, s46, 31 // (op 1095) 
s_lshl_b32 s51, 1, s18 // (op 1095) 
s_add_i32 s72, s51, s50 // (op 1095) 


s_and_b32 s50, s0, s72 // (op 1095) 


s_add_i32 s0, s19, s50 // (op 1095) 


s_ashr_i32 s19, s0, s18 // (op 1095) 


s_lshl_b32 s0, s46, 24 // (op 1095) 
s_ashr_i32 s18, s0, 31 // (op 1095) 

s_xor_b32 s0, s19, s18 // (op 1095) 

s_sub_i32 s19, s0, s18 // (op 1095) 


s_cmp_eq_i32 s1, 0 // (op 1095) 
s_mov_b32 s0, scc // (op 1095) 

s_cmp_eq_i32 s0, 0 // (op 1095) 
s_cselect_b32 s1, s19, s49 // (op 1095) 



v_lshl_add_u32 v12, s1, 6, v15 // (op 1095) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1095) 
s_lshr_b32 s18, s0, 1 // (op 1095) 

v_lshl_add_u32 v15, s18, 3, v16 // (op 1095) 


v_mul_lo_u32 v16, v15, s38 // (op 1095) 
v_add_u32 v16, v16, v12 // (op 1095) 


s_mul_i32 s1, s38, s43 // (op 1097) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1097) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1097) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1097) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1097) most significant: sum
s_add_u32 s1, s1, s19 // (op 1097) most significant: sum


s_mov_b32 s68, s40 // (op 1097) reg expression
s_mov_b32 s69, s41 // (op 1097) reg expression
s_mov_b32 s70, s0 // (op 1097) reg expression
s_mov_b32 s71, 131072 // (op 1097) call()

v_and_b32 v12, -8, v1 // (op 1188) 
v_sub_u32 v15, v1, v12 // (op 1188) 

v_lshlrev_b32 v12, 4, v15 // (op 1188) 

v_lshrrev_b32 v15, 3, v1 // (op 1188) 
v_lshl_add_u32 v17, v15, 7, v12 // (op 1188) 


v_mul_lo_u32 v12, v17, 6 // (op 1188) 

v_lshrrev_b32 v15, 3, v12 // (op 1188) 

v_and_b32 v12, -4, v1 // (op 1241) 
v_sub_u32 v17, v1, v12 // (op 1241) 

v_lshlrev_b32 v12, 4, v17 // (op 1241) 

v_lshrrev_b32 v17, 2, v1 // (op 1241) 
v_lshl_add_u32 v20, v17, 6, v12 // (op 1241) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1241) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1298) 
v_sub_u32 v17, v1, v12 // (op 1298) 

v_lshrrev_b32 v12, 6, v1 // (op 1298) 
v_lshl_add_u32 v20, v12, 7, v17 // (op 1298) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 317) 
s_and_b32 s1, s18, 4294967294 // (op 317) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 316) 
s_mov_b32 s49, s18 // (op 316) convert

s_cmp_lt_u32 s1, s49 // (op 316) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316 // (op 316) Condition: Top (jump to RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316:


s_add_u32 s1, s1, 1 // (op 318) 
v_mov_b32 v12, v14 // (op 170) 

buffer_load_dwordx3 v[22:24], v12, s[20:23], 0 offen // (op 170) Load value

v_add_u32 v14, 96, v14 // (op 889) 
s_barrier  // (op 1493) 
s_barrier  // (op 1506) 




s_lshl_b64 s[18:19], s[28:29], 5 // (op 869) 
v_mov_b64 v[26:27], s[18:19] // (op 869) convert


v_mov_b64 v[28:29], v[18:19] // (op 869) 
v_mov_b64 v[30:31], v[28:29] // (op 869) reg expression
s_mov_b32 m0, s0 // (op 869) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 869) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v26 // (op 869) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 869) most significant half

s_add_u32 m0, m0, 2048 // (op 869) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 869) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v26 // (op 869) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 869) most significant half

s_add_u32 m0, m0, 2048 // (op 869) 
s_waitcnt vmcnt(2)
ds_write_b96 v15, v[22:24] offset:16640 // (op 171) Store local data 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 869) Load value direct to lds

v_add_co_u32 v30, s[18:19], v30, v26 // (op 869) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 869) most significant half

s_add_u32 m0, m0, 2048 // (op 869) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 869) Load value direct to lds



s_lshl_b64 s[18:19], s[28:29], 7 // (op 959) 
v_mov_b32 v12, s18 // (op 959) 
v_mov_b32 v17, s19 // (op 959) 
v_add_co_u32 v18, s[50:51], v18, v12 // (op 959) least significant half
v_addc_co_u32 v19, s[50:51], v19, v17, s[50:51] // (op 959) most significant half





v_mov_b64 v[22:23], s[38:39] // (op 184) convert

v_mov_b32 v12, v16 // (op 184) 
v_mov_b32 v17, v12 // (op 184) reg expression
buffer_load_ubyte v24, v17, s[68:71], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v17, v17, v22 // (op 184) 
buffer_load_ubyte v25, v17, s[68:71], 0 offen offset: 0 // (op 184) Load value



s_lshl_b32 s18, s38, 2 // (op 1087) 
v_add_u32 v16, v16, s18 // (op 1087) 

s_barrier  // (op 1519) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1496) 






ds_read_b128 v[26:29], v3 offset:16640 // (op 173) Load local data 
ds_read_b64 v[30:31], v3 offset:16656 // (op 173) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v24 offset:16384 // (op 182) Store local data 
s_barrier  // (op 1509) 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v25 offset:16448 // (op 182) Store local data 




ds_read_b64_tr_b8 v[32:33], v7 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v7 offset:512 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v7 offset:4096 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v7 offset:4608 // (op 175) Transpose load local data 


v_mov_b32 v12, v13 // (op 177) 

buffer_load_ubyte v17, v12, s[64:67], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v13, 4, v13 // (op 1023) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1523) 


ds_read_u8 v12, v0 offset:16384 // (op 179) Load local data 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[32:39], a[0:3], v17, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 180) 
ds_read_b64_tr_b8 v[40:41], v7 offset:32 // (op 324) Transpose load local data 


ds_read_b64_tr_b8 v[42:43], v7 offset:544 // (op 324) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:4128 // (op 324) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:4640 // (op 324) Transpose load local data 


ds_read_u8 v12, v0 offset:16416 // (op 327) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:31], v[40:47], a[4:7], v17, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 328) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 316) 
s_mov_b32 s49, s18 // (op 316) convert

s_cmp_lt_u32 s1, s49 // (op 316) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316 // (op 316) Condition: Bottom (jump to RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316 if true)
RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316:






















s_barrier  // (op 1599) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_125
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 

v_cvt_f16_f32 v10, v14 // (op 52) 
v_cvt_f16_f32 v11, v15 // (op 52) 
v_cvt_f16_f32 v12, v16 // (op 52) 
v_cvt_f16_f32 v13, v17 // (op 52) 


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

v_cvt_f16_f32 v10, v14 // (op 510) 
v_cvt_f16_f32 v11, v15 // (op 510) 
v_cvt_f16_f32 v12, v16 // (op 510) 
v_cvt_f16_f32 v13, v17 // (op 510) 


v_pack_b32_f16 v14, v10, v11 // (op 511) 
v_pack_b32_f16 v15, v12, v13 // (op 511) 

s_lshl_b32 s0, s4, 6 // (op 511) 
v_add_u32 v0, v9, s0 // (op 511) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 511) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_125:


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
v_cvt_f32_f16 v12, v10 // (op 139) 
v_lshrrev_b32 v13, 16, v10 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v20, v5, v12 // (op 139) 
v_mul_f32 v21, v5, v13 // (op 139) 
v_cvt_f32_f16 v12, v11 // (op 139) 
v_lshrrev_b32 v13, 16, v11 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v22, v5, v12 // (op 139) 
v_mul_f32 v23, v5, v13 // (op 139) 


v_add_f32 v10, v20, v16 // (op 135) 
v_add_f32 v11, v21, v17 // (op 135) 
v_add_f32 v12, v22, v18 // (op 135) 
v_add_f32 v13, v23, v19 // (op 135) 


v_cvt_f16_f32 v14, v10 // (op 136) 
v_cvt_f16_f32 v15, v11 // (op 136) 
v_cvt_f16_f32 v16, v12 // (op 136) 
v_cvt_f16_f32 v17, v13 // (op 136) 


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
v_cvt_f32_f16 v12, v10 // (op 536) 
v_lshrrev_b32 v13, 16, v10 // (op 536) 
v_cvt_f32_f16 v13, v13 // (op 536) 
v_mul_f32 v20, v5, v12 // (op 536) 
v_mul_f32 v21, v5, v13 // (op 536) 
v_cvt_f32_f16 v12, v11 // (op 536) 
v_lshrrev_b32 v13, 16, v11 // (op 536) 
v_cvt_f32_f16 v13, v13 // (op 536) 
v_mul_f32 v22, v5, v12 // (op 536) 
v_mul_f32 v23, v5, v13 // (op 536) 


v_add_f32 v10, v20, v16 // (op 532) 
v_add_f32 v11, v21, v17 // (op 532) 
v_add_f32 v12, v22, v18 // (op 532) 
v_add_f32 v13, v23, v19 // (op 532) 


v_cvt_f16_f32 v14, v10 // (op 533) 
v_cvt_f16_f32 v15, v11 // (op 533) 
v_cvt_f16_f32 v16, v12 // (op 533) 
v_cvt_f16_f32 v17, v13 // (op 533) 


v_pack_b32_f16 v10, v14, v15 // (op 534) 
v_pack_b32_f16 v11, v16, v17 // (op 534) 

s_lshl_b32 s0, s4, 6 // (op 534) 
v_add_u32 v0, v9, s0 // (op 534) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 534) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.LRR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__end:
.size RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_, .LRR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__end-RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: F8
    DestDataType: h
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
  - .name: RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
    .symbol: RR_GEMM_TT_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_.kd
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

