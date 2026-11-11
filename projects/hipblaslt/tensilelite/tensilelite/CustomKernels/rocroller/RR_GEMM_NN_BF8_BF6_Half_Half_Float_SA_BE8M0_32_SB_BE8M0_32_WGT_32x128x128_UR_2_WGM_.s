// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM_,@function
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1995) 
s_mul_hi_i32 s1, s16, s10 // (op 2183) 
v_and_b32 v0, 48, v1 // (op 2051) 
v_and_b32 v3, 63, v1 // (op 2051) 
v_sub_u32 v4, v3, v0 // (op 2051) 

v_lshrrev_b32 v3, 1, v4 // (op 2051) 
v_add_lshl_u32 v5, v0, v3, 5 // (op 2051) 


v_and_b32 v0, 1, v4 // (op 2051) 

v_lshlrev_b32 v3, 3, v0 // (op 2051) 

v_lshrrev_b32 v0, 7, v1 // (op 2051) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2051) 


v_add_u32 v0, v5, v4 // (op 2051) 


v_and_b32 v3, 48, v1 // (op 2224) 
v_and_b32 v4, 63, v1 // (op 2224) 
v_sub_u32 v5, v4, v3 // (op 2224) 

v_lshrrev_b32 v4, 1, v5 // (op 2224) 
v_add_lshl_u32 v6, v3, v4, 5 // (op 2224) 


v_and_b32 v3, 1, v5 // (op 2224) 

v_lshlrev_b32 v4, 3, v3 // (op 2224) 

v_lshrrev_b32 v3, 7, v1 // (op 2224) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 2224) 


v_add_u32 v3, v6, v5 // (op 2224) 


v_lshrrev_b32 v4, 7, v1 // (op 1995) 
s_add_i32 s18, s0, s16 // (op 1995) 

v_and_b32 v5, 63, v1 // (op 1995) 
s_ashr_i32 s0, s18, 31 // (op 1995) 
v_lshrrev_b32 v6, 4, v5 // (op 1995) 
s_cmp_eq_i32 s10, 0 // (op 1995) 
s_mov_b32 s19, scc // (op 1995) 
v_lshlrev_b32 v7, 2, v6 // (op 1995) 

s_cmp_eq_i32 s19, 0 // (op 1995) 
s_cselect_b32 s51, 0, -1 // (op 1995) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1995) 


s_and_b32 s19, s11, 31 // (op 1995) 
v_lshrrev_b32 v4, 2, v5 // (op 1995) 

v_and_b32 v5, 3, v4 // (op 1995) 

v_lshrrev_b32 v4, 6, v1 // (op 2109) 
v_and_b32 v7, 1, v4 // (op 2109) 

v_and_b32 v4, 15, v1 // (op 2109) 
v_lshl_add_u32 v8, v7, 4, v4 // (op 2109) 


v_and_b32 v4, 63, v1 // (op 2109) 
v_lshrrev_b32 v7, 4, v4 // (op 2109) 

v_lshlrev_b32 v4, 5, v7 // (op 2109) 

v_lshl_add_u32 v7, v8, 7, v4 // (op 2109) 


v_mul_lo_u32 v4, v7, 6 // (op 2109) 

v_lshrrev_b32 v7, 3, v4 // (op 2109) 

v_lshrrev_b32 v4, 6, v1 // (op 2167) 
v_and_b32 v8, 1, v4 // (op 2167) 

v_and_b32 v4, 15, v1 // (op 2167) 
v_lshl_add_u32 v9, v8, 4, v4 // (op 2167) 


v_and_b32 v4, 63, v1 // (op 2167) 
v_lshrrev_b32 v8, 4, v4 // (op 2167) 

v_lshl_add_u32 v4, v9, 2, v8 // (op 2167) 


v_lshrrev_b32 v8, 7, v1 // (op 2183) 
v_and_b32 v9, 63, v1 // (op 2183) 
v_lshrrev_b32 v10, 6, v1 // (op 2253) 
v_and_b32 v11, 1, v10 // (op 2253) 

v_and_b32 v10, 15, v1 // (op 2253) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 2253) 


v_and_b32 v10, 63, v1 // (op 2253) 
v_lshrrev_b32 v11, 4, v10 // (op 2253) 

v_lshlrev_b32 v10, 5, v11 // (op 2253) 

v_lshl_add_u32 v11, v12, 7, v10 // (op 2253) 


v_mul_lo_u32 v10, v11, 6 // (op 2253) 

v_lshrrev_b32 v11, 3, v10 // (op 2253) 

v_lshrrev_b32 v10, 6, v1 // (op 2282) 
v_and_b32 v12, 1, v10 // (op 2282) 

v_and_b32 v10, 15, v1 // (op 2282) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 2282) 


v_and_b32 v10, 63, v1 // (op 2282) 
v_lshrrev_b32 v12, 4, v10 // (op 2282) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 2282) 


s_lshl_b32 s52, 1, s19 // (op 1995) 
v_and_b32 v12, 3, v1 // (op 1995) 
s_add_i32 s53, s52, s51 // (op 1995) 


s_and_b32 s51, s0, s53 // (op 1995) 


s_add_i32 s0, s1, s16 // (op 2183) 

s_ashr_i32 s1, s0, 31 // (op 2183) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1995) 


s_add_i32 s52, s18, s51 // (op 1995) 


v_lshrrev_b32 v5, 6, v1 // (op 1995) 
s_ashr_i32 s18, s52, s19 // (op 1995) 


v_and_b32 v12, 1, v5 // (op 1995) 

s_lshl_b32 s19, s11, 24 // (op 1995) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1995) 


s_ashr_i32 s51, s19, 31 // (op 1995) 

s_xor_b32 s19, s18, s51 // (op 1995) 

s_sub_i32 s18, s19, s51 // (op 1995) 


s_mul_hi_i32 s19, s18, s12 // (op 1995) 
s_add_i32 s51, s19, s18 // (op 1995) 

s_ashr_i32 s19, s51, 31 // (op 1995) 
s_cmp_eq_i32 s12, 0 // (op 1995) 
s_mov_b32 s52, scc // (op 1995) 
s_cmp_eq_i32 s52, 0 // (op 1995) 
s_cselect_b32 s53, 0, -1 // (op 1995) 

s_and_b32 s52, s13, 31 // (op 1995) 
v_lshrrev_b32 v12, 4, v9 // (op 2183) 
s_cmp_eq_i32 s10, 0 // (op 2183) 
s_mov_b32 s54, scc // (op 2183) 
v_lshlrev_b32 v13, 2, v12 // (op 2183) 

s_cmp_eq_i32 s54, 0 // (op 2183) 
s_cselect_b32 s55, 0, -1 // (op 2183) 

s_lshl_b32 s54, 1, s52 // (op 1995) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 2183) 


v_lshrrev_b32 v8, 2, v9 // (op 2183) 

s_add_i32 s56, s54, s53 // (op 1995) 


s_and_b32 s53, s19, s56 // (op 1995) 


s_add_i32 s19, s51, s53 // (op 1995) 


s_ashr_i32 s51, s19, s52 // (op 1995) 


s_lshl_b32 s19, s13, 24 // (op 1995) 
s_ashr_i32 s52, s19, 31 // (op 1995) 

s_xor_b32 s19, s51, s52 // (op 1995) 

s_sub_i32 s51, s19, s52 // (op 1995) 


s_mul_i32 s19, s18, s15 // (op 1995) 
s_sub_u32 s52, s16, s19 // (op 1995) 

s_mul_i32 s19, s51, s15 // (op 1995) 
s_add_u32 s19, s19, s52 // (op 1995) 

s_mul_hi_i32 s52, s16, s9 // (op 1995) 
s_add_i32 s53, s52, s16 // (op 1995) 

s_ashr_i32 s52, s53, 31 // (op 1995) 
s_cmp_eq_i32 s9, 0 // (op 1995) 
s_mov_b32 s54, scc // (op 1995) 
s_cmp_eq_i32 s54, 0 // (op 1995) 
s_cselect_b32 s56, 0, -1 // (op 1995) 

s_and_b32 s54, s11, 31 // (op 2183) 
v_and_b32 v9, 3, v8 // (op 2183) 

s_lshl_b32 s57, 1, s54 // (op 2183) 
v_and_b32 v8, 3, v1 // (op 2183) 
s_add_i32 s58, s57, s55 // (op 2183) 


s_and_b32 s55, s1, s58 // (op 2183) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 2183) 


s_add_i32 s1, s0, s55 // (op 2183) 


v_lshrrev_b32 v8, 6, v1 // (op 2183) 
s_ashr_i32 s0, s1, s54 // (op 2183) 


v_and_b32 v9, 1, v8 // (op 2183) 

s_lshl_b32 s1, s11, 24 // (op 2183) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 2183) 


s_ashr_i32 s54, s1, 31 // (op 2183) 

s_xor_b32 s1, s0, s54 // (op 2183) 

s_sub_i32 s0, s1, s54 // (op 2183) 


s_mul_hi_i32 s1, s0, s12 // (op 2183) 
s_add_i32 s54, s1, s0 // (op 2183) 

s_ashr_i32 s1, s54, 31 // (op 2183) 
s_cmp_eq_i32 s12, 0 // (op 2183) 
s_mov_b32 s55, scc // (op 2183) 
s_cmp_eq_i32 s55, 0 // (op 2183) 
s_cselect_b32 s57, 0, -1 // (op 2183) 

s_and_b32 s55, s13, 31 // (op 2183) 
s_lshl_b32 s58, 1, s55 // (op 2183) 
s_add_i32 s59, s58, s57 // (op 2183) 


s_and_b32 s57, s1, s59 // (op 2183) 


s_add_i32 s1, s54, s57 // (op 2183) 


s_ashr_i32 s54, s1, s55 // (op 2183) 


s_lshl_b32 s1, s13, 24 // (op 2183) 
s_ashr_i32 s55, s1, 31 // (op 2183) 

s_xor_b32 s1, s54, s55 // (op 2183) 

s_sub_i32 s54, s1, s55 // (op 2183) 


s_mul_i32 s1, s0, s15 // (op 2183) 
s_sub_u32 s55, s16, s1 // (op 2183) 

s_mul_i32 s1, s54, s15 // (op 2183) 
s_add_u32 s1, s1, s55 // (op 2183) 

s_mul_hi_i32 s55, s16, s9 // (op 2183) 
s_add_i32 s57, s55, s16 // (op 2183) 

s_ashr_i32 s55, s57, 31 // (op 2183) 
s_cmp_eq_i32 s9, 0 // (op 2183) 
s_mov_b32 s58, scc // (op 2183) 
s_cmp_eq_i32 s58, 0 // (op 2183) 
s_cselect_b32 s59, 0, -1 // (op 2183) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 1995) 
s_lshl_b32 s60, 1, s58 // (op 1995) 
s_add_i32 s61, s60, s56 // (op 1995) 


s_and_b32 s56, s52, s61 // (op 1995) 


s_add_i32 s52, s53, s56 // (op 1995) 


s_ashr_i32 s53, s52, s58 // (op 1995) 


s_lshl_b32 s52, s46, 24 // (op 1995) 
s_ashr_i32 s56, s52, 31 // (op 1995) 

s_xor_b32 s52, s53, s56 // (op 1995) 

s_sub_i32 s53, s52, s56 // (op 1995) 


s_and_b32 s52, s46, 31 // (op 2183) 
s_lshl_b32 s56, 1, s52 // (op 2183) 
s_add_i32 s58, s56, s59 // (op 2183) 


s_and_b32 s56, s55, s58 // (op 2183) 


s_add_i32 s55, s57, s56 // (op 2183) 


s_ashr_i32 s56, s55, s52 // (op 2183) 


s_lshl_b32 s52, s46, 24 // (op 2183) 
s_ashr_i32 s55, s52, 31 // (op 2183) 

s_xor_b32 s52, s56, s55 // (op 2183) 

s_sub_i32 s56, s52, s55 // (op 2183) 


s_add_u32 s58, s6, 128 // (op 1995) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 1995) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1995) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1995) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 7 // (op 1995) 

s_mul_i32 s52, s15, s58 // (op 1995) 
s_mul_i32 s55, s8, s52 // (op 1995) 

s_mul_i32 s52, s53, s55 // (op 1995) 

s_sub_u32 s55, s16, s52 // (op 1995) 

s_mov_b32 s52, s55 // (op 1995) 
s_mul_hi_i32 s57, s52, s14 // (op 1995) 
s_add_i32 s60, s57, s52 // (op 1995) 


s_ashr_i32 s52, s60, 31 // (op 1995) 
s_cmp_eq_i32 s14, 0 // (op 1995) 
s_mov_b32 s57, scc // (op 1995) 
s_cmp_eq_i32 s57, 0 // (op 1995) 
s_cselect_b32 s61, 0, -1 // (op 1995) 

s_and_b32 s57, s48, 31 // (op 1995) 
s_lshl_b32 s62, 1, s57 // (op 1995) 
s_add_i32 s63, s62, s61 // (op 1995) 


s_and_b32 s61, s52, s63 // (op 1995) 


s_add_i32 s52, s60, s61 // (op 1995) 


s_ashr_i32 s60, s52, s57 // (op 1995) 


s_lshl_b32 s52, s48, 24 // (op 1995) 
s_ashr_i32 s57, s52, 31 // (op 1995) 

s_xor_b32 s52, s60, s57 // (op 1995) 

s_sub_i32 s60, s52, s57 // (op 1995) 


s_mul_i32 s52, s47, s15 // (op 1995) 
s_add_u32 s62, s20, 32 // (op 1995) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 1995) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1995) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1995) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 5 // (op 1995) 

s_sub_i32 s57, s62, s52 // (op 1995) 

s_mul_i32 s61, s60, s57 // (op 1995) 

s_sub_u32 s57, s55, s61 // (op 1995) 


s_add_u32 s55, s57, s52 // (op 1995) 


s_cmp_eq_i32 s53, 0 // (op 1995) 
s_mov_b32 s52, scc // (op 1995) 

s_cmp_eq_i32 s52, 0 // (op 1995) 
s_cselect_b32 s53, s55, s19 // (op 1995) 


s_mul_i32 s19, s51, s58 // (op 1995) 


v_lshl_add_u32 v9, s53, 5, v6 // (op 1995) 


s_sub_i32 s51, s18, s19 // (op 1995) 


s_cmp_eq_i32 s52, 0 // (op 1995) 
s_cselect_b32 s53, s60, s51 // (op 1995) 



v_lshl_add_u32 v6, s53, 7, v5 // (op 1995) 


v_mul_lo_u32 v5, v6, s2 // (op 1995) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 1995) 


s_add_u32 s18, s6, 128 // (op 2183) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 2183) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 2183) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 2183) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 7 // (op 2183) 

s_mul_i32 s55, s15, s18 // (op 2183) 
s_mul_i32 s51, s8, s55 // (op 2183) 

s_mul_i32 s55, s56, s51 // (op 2183) 

s_sub_u32 s51, s16, s55 // (op 2183) 

s_mov_b32 s55, s51 // (op 2183) 
s_mul_hi_i32 s52, s55, s14 // (op 2183) 
s_add_i32 s53, s52, s55 // (op 2183) 


s_ashr_i32 s52, s53, 31 // (op 2183) 
s_cmp_eq_i32 s14, 0 // (op 2183) 
s_mov_b32 s55, scc // (op 2183) 
s_cmp_eq_i32 s55, 0 // (op 2183) 
s_cselect_b32 s57, 0, -1 // (op 2183) 

s_and_b32 s55, s48, 31 // (op 2183) 
s_lshl_b32 s58, 1, s55 // (op 2183) 
s_add_i32 s59, s58, s57 // (op 2183) 


s_and_b32 s57, s52, s59 // (op 2183) 


s_add_i32 s52, s53, s57 // (op 2183) 


s_ashr_i32 s53, s52, s55 // (op 2183) 


s_lshl_b32 s52, s48, 24 // (op 2183) 
s_ashr_i32 s55, s52, 31 // (op 2183) 

s_xor_b32 s52, s53, s55 // (op 2183) 

s_sub_i32 s53, s52, s55 // (op 2183) 


s_mul_i32 s52, s47, s15 // (op 2183) 
s_add_u32 s58, s20, 32 // (op 2183) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 2183) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 2183) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 2183) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 5 // (op 2183) 

s_sub_i32 s55, s58, s52 // (op 2183) 

s_mul_i32 s57, s53, s55 // (op 2183) 

s_sub_u32 s55, s51, s57 // (op 2183) 


s_add_u32 s51, s55, s52 // (op 2183) 


s_cmp_eq_i32 s56, 0 // (op 2183) 
s_mov_b32 s52, scc // (op 2183) 

s_cmp_eq_i32 s52, 0 // (op 2183) 
s_cselect_b32 s55, s51, s1 // (op 2183) 


s_mul_i32 s1, s54, s18 // (op 2183) 


v_lshl_add_u32 v5, s55, 5, v12 // (op 2183) 


s_sub_i32 s51, s0, s1 // (op 2183) 


s_cmp_eq_i32 s52, 0 // (op 2183) 
s_cselect_b32 s0, s53, s51 // (op 2183) 



v_lshl_add_u32 v9, s0, 7, v8 // (op 2183) 


v_mul_lo_u32 v8, v9, s4 // (op 2183) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 2183) 


s_mul_i32 s1, s2, s7 // (op 1997) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1997) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1997) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1997) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1997) most significant: sum
s_add_u32 s1, s1, s19 // (op 1997) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1997) 

s_mul_i32 s1, s4, s7 // (op 2185) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 2185) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 2185) least significant
s_mul_hi_u32 s52, s4, s6 // (op 2185) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 2185) most significant: sum
s_add_u32 s1, s1, s52 // (op 2185) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 2185) 

s_mov_b32 s56, s22 // (op 1997) reg expression
s_mov_b32 s57, s23 // (op 1997) reg expression
s_mov_b32 s58, s18 // (op 1997) reg expression
s_mov_b32 s59, 131072 // (op 1997) call()

s_mov_b32 s60, s24 // (op 2185) reg expression

s_mov_b32 s61, s25 // (op 2185) reg expression
s_mov_b32 s62, s52 // (op 2185) reg expression
s_mov_b32 s63, 131072 // (op 2185) call()


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
v_accvgpr_write a8, 0.00000 // (op 466) call()
v_accvgpr_write a9, 0.00000 // (op 466) call()
v_accvgpr_write a10, 0.00000 // (op 466) call()
v_accvgpr_write a11, 0.00000 // (op 466) call()
v_accvgpr_write a12, 0.00000 // (op 622) call()
v_accvgpr_write a13, 0.00000 // (op 622) call()
v_accvgpr_write a14, 0.00000 // (op 622) call()
v_accvgpr_write a15, 0.00000 // (op 622) call()
s_mul_hi_i32 s0, s16, s10 // (op 1762) 
v_and_b32 v12, -2, v1 // (op 1762) 
s_add_i32 s1, s0, s16 // (op 1762) 

v_sub_u32 v13, v1, v12 // (op 1762) 

s_ashr_i32 s0, s1, 31 // (op 1762) 
v_lshlrev_b32 v12, 4, v13 // (op 1762) 

s_cmp_eq_i32 s10, 0 // (op 1762) 
s_mov_b32 s18, scc // (op 1762) 
v_lshrrev_b32 v13, 1, v1 // (op 1762) 
s_cmp_eq_i32 s18, 0 // (op 1762) 
s_cselect_b32 s19, 0, -1 // (op 1762) 

s_and_b32 s18, s11, 31 // (op 1762) 
s_lshl_b32 s22, 1, s18 // (op 1762) 
s_add_i32 s23, s22, s19 // (op 1762) 


s_and_b32 s19, s0, s23 // (op 1762) 


s_add_i32 s0, s1, s19 // (op 1762) 


s_ashr_i32 s1, s0, s18 // (op 1762) 


s_lshl_b32 s0, s11, 24 // (op 1762) 
s_ashr_i32 s18, s0, 31 // (op 1762) 

s_xor_b32 s0, s1, s18 // (op 1762) 

s_sub_i32 s1, s0, s18 // (op 1762) 


s_mul_hi_i32 s0, s1, s12 // (op 1762) 
s_add_i32 s18, s0, s1 // (op 1762) 

s_ashr_i32 s0, s18, 31 // (op 1762) 
s_cmp_eq_i32 s12, 0 // (op 1762) 
s_mov_b32 s19, scc // (op 1762) 
s_cmp_eq_i32 s19, 0 // (op 1762) 
s_cselect_b32 s22, 0, -1 // (op 1762) 

s_and_b32 s19, s13, 31 // (op 1762) 
s_lshl_b32 s23, 1, s19 // (op 1762) 
s_add_i32 s24, s23, s22 // (op 1762) 


s_and_b32 s25, s0, s24 // (op 1762) 


s_add_i32 s0, s18, s25 // (op 1762) 


s_ashr_i32 s18, s0, s19 // (op 1762) 


s_lshl_b32 s0, s13, 24 // (op 1762) 
s_ashr_i32 s19, s0, 31 // (op 1762) 

s_xor_b32 s0, s18, s19 // (op 1762) 

s_sub_i32 s18, s0, s19 // (op 1762) 


s_mul_i32 s0, s1, s15 // (op 1762) 

s_sub_u32 s1, s16, s0 // (op 1762) 

s_mul_i32 s0, s18, s15 // (op 1762) 
s_add_u32 s0, s0, s1 // (op 1762) 


s_mul_hi_i32 s1, s16, s9 // (op 1762) 
s_add_i32 s18, s1, s16 // (op 1762) 

s_ashr_i32 s1, s18, 31 // (op 1762) 
s_cmp_eq_i32 s9, 0 // (op 1762) 
s_mov_b32 s19, scc // (op 1762) 
s_cmp_eq_i32 s19, 0 // (op 1762) 
s_cselect_b32 s22, 0, -1 // (op 1762) 

s_and_b32 s19, s46, 31 // (op 1762) 
s_lshl_b32 s23, 1, s19 // (op 1762) 
s_add_i32 s24, s23, s22 // (op 1762) 


s_and_b32 s25, s1, s24 // (op 1762) 


s_add_i32 s1, s18, s25 // (op 1762) 


s_ashr_i32 s18, s1, s19 // (op 1762) 


s_lshl_b32 s1, s46, 24 // (op 1762) 
s_ashr_i32 s19, s1, 31 // (op 1762) 

s_xor_b32 s1, s18, s19 // (op 1762) 

s_sub_i32 s18, s1, s19 // (op 1762) 


s_add_u32 s22, s6, 128 // (op 1762) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1762) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1762) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1762) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 1762) 

s_mul_i32 s1, s15, s22 // (op 1762) 

s_mul_i32 s19, s8, s1 // (op 1762) 

s_mul_i32 s1, s18, s19 // (op 1762) 

s_sub_u32 s19, s16, s1 // (op 1762) 

s_mov_b32 s1, s19 // (op 1762) 
s_mul_hi_i32 s22, s1, s14 // (op 1762) 
s_add_i32 s23, s22, s1 // (op 1762) 


s_ashr_i32 s1, s23, 31 // (op 1762) 
s_cmp_eq_i32 s14, 0 // (op 1762) 
s_mov_b32 s22, scc // (op 1762) 
s_cmp_eq_i32 s22, 0 // (op 1762) 
s_cselect_b32 s24, 0, -1 // (op 1762) 

s_and_b32 s22, s48, 31 // (op 1762) 
s_lshl_b32 s25, 1, s22 // (op 1762) 
s_add_i32 s49, s25, s24 // (op 1762) 


s_and_b32 s24, s1, s49 // (op 1762) 


s_add_i32 s1, s23, s24 // (op 1762) 


s_ashr_i32 s23, s1, s22 // (op 1762) 


s_lshl_b32 s1, s48, 24 // (op 1762) 
s_ashr_i32 s22, s1, 31 // (op 1762) 

s_xor_b32 s1, s23, s22 // (op 1762) 

s_sub_i32 s23, s1, s22 // (op 1762) 


s_mul_i32 s1, s47, s15 // (op 1762) 
s_add_u32 s24, s20, 32 // (op 1762) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1762) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 1762) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 1762) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 5 // (op 1762) 

s_sub_i32 s22, s24, s1 // (op 1762) 

s_mul_i32 s24, s23, s22 // (op 1762) 


s_sub_u32 s25, s19, s24 // (op 1762) 


s_add_u32 s19, s25, s1 // (op 1762) 


s_cmp_eq_i32 s18, 0 // (op 1762) 
s_mov_b32 s1, scc // (op 1762) 

s_cmp_eq_i32 s1, 0 // (op 1762) 
s_cselect_b32 s18, s19, s0 // (op 1762) 



v_lshl_add_u32 v14, s18, 5, v12 // (op 1762) 


v_mov_b32 v17, 0 // (op 1762) convert
v_mov_b32 v16, v14 // (op 1762) convert

v_mul_lo_u32 v15, v13, s27 // (op 1762) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 1762) most significant: high of low * low
v_mul_lo_u32 v14, v13, s26 // (op 1762) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 1762) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v14 // (op 1762) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 1762) most significant half



s_mul_i32 s1, s26, s31 // (op 1764) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1764) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1764) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1764) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1764) most significant: sum
s_add_u32 s1, s1, s19 // (op 1764) most significant: sum


s_mov_b32 s52, s28 // (op 1764) reg expression
s_mov_b32 s53, s29 // (op 1764) reg expression
s_mov_b32 s54, s0 // (op 1764) reg expression
s_mov_b32 s55, 131072 // (op 1764) call()

s_mul_hi_i32 s0, s16, s10 // (op 1781) 
v_and_b32 v14, -2, v1 // (op 1781) 
s_add_i32 s1, s0, s16 // (op 1781) 

v_sub_u32 v15, v1, v14 // (op 1781) 

s_ashr_i32 s0, s1, 31 // (op 1781) 
v_lshlrev_b32 v14, 4, v15 // (op 1781) 

s_cmp_eq_i32 s10, 0 // (op 1781) 
s_mov_b32 s18, scc // (op 1781) 
v_lshrrev_b32 v15, 1, v1 // (op 1781) 
s_cmp_eq_i32 s18, 0 // (op 1781) 
s_cselect_b32 s19, 0, -1 // (op 1781) 

s_and_b32 s18, s11, 31 // (op 1781) 
s_lshl_b32 s22, 1, s18 // (op 1781) 
s_add_i32 s23, s22, s19 // (op 1781) 


s_and_b32 s19, s0, s23 // (op 1781) 


s_add_i32 s0, s1, s19 // (op 1781) 


s_ashr_i32 s1, s0, s18 // (op 1781) 


s_lshl_b32 s0, s11, 24 // (op 1781) 
s_ashr_i32 s18, s0, 31 // (op 1781) 

s_xor_b32 s0, s1, s18 // (op 1781) 

s_sub_i32 s1, s0, s18 // (op 1781) 


s_mul_hi_i32 s0, s1, s12 // (op 1781) 
s_add_i32 s18, s0, s1 // (op 1781) 

s_ashr_i32 s0, s18, 31 // (op 1781) 
s_cmp_eq_i32 s12, 0 // (op 1781) 
s_mov_b32 s19, scc // (op 1781) 
s_cmp_eq_i32 s19, 0 // (op 1781) 
s_cselect_b32 s22, 0, -1 // (op 1781) 

s_and_b32 s19, s13, 31 // (op 1781) 
s_lshl_b32 s23, 1, s19 // (op 1781) 
s_add_i32 s24, s23, s22 // (op 1781) 


s_and_b32 s25, s0, s24 // (op 1781) 


s_add_i32 s0, s18, s25 // (op 1781) 


s_ashr_i32 s18, s0, s19 // (op 1781) 


s_lshl_b32 s0, s13, 24 // (op 1781) 
s_ashr_i32 s19, s0, 31 // (op 1781) 

s_xor_b32 s0, s18, s19 // (op 1781) 

s_sub_i32 s18, s0, s19 // (op 1781) 


s_mul_i32 s0, s1, s15 // (op 1781) 

s_sub_u32 s1, s16, s0 // (op 1781) 

s_mul_i32 s0, s18, s15 // (op 1781) 
s_add_u32 s0, s0, s1 // (op 1781) 


s_mul_hi_i32 s1, s16, s9 // (op 1781) 
s_add_i32 s18, s1, s16 // (op 1781) 

s_ashr_i32 s1, s18, 31 // (op 1781) 
s_cmp_eq_i32 s9, 0 // (op 1781) 
s_mov_b32 s19, scc // (op 1781) 
s_cmp_eq_i32 s19, 0 // (op 1781) 
s_cselect_b32 s22, 0, -1 // (op 1781) 

s_and_b32 s19, s46, 31 // (op 1781) 
s_lshl_b32 s23, 1, s19 // (op 1781) 
s_add_i32 s24, s23, s22 // (op 1781) 


s_and_b32 s25, s1, s24 // (op 1781) 


s_add_i32 s1, s18, s25 // (op 1781) 


s_ashr_i32 s18, s1, s19 // (op 1781) 


s_lshl_b32 s1, s46, 24 // (op 1781) 
s_ashr_i32 s19, s1, 31 // (op 1781) 

s_xor_b32 s1, s18, s19 // (op 1781) 

s_sub_i32 s18, s1, s19 // (op 1781) 


s_add_u32 s50, s6, 128 // (op 1781) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1781) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1781) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1781) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 1781) 

s_mul_i32 s1, s15, s24 // (op 1781) 

s_mul_i32 s19, s8, s1 // (op 1781) 

s_mul_i32 s1, s18, s19 // (op 1781) 

s_sub_u32 s19, s16, s1 // (op 1781) 

s_mov_b32 s1, s19 // (op 1781) 
s_mul_hi_i32 s22, s1, s14 // (op 1781) 
s_add_i32 s23, s22, s1 // (op 1781) 


s_ashr_i32 s1, s23, 31 // (op 1781) 
s_cmp_eq_i32 s14, 0 // (op 1781) 
s_mov_b32 s22, scc // (op 1781) 
s_cmp_eq_i32 s22, 0 // (op 1781) 
s_cselect_b32 s24, 0, -1 // (op 1781) 

s_and_b32 s22, s48, 31 // (op 1781) 
s_lshl_b32 s25, 1, s22 // (op 1781) 
s_add_i32 s49, s25, s24 // (op 1781) 


s_and_b32 s24, s1, s49 // (op 1781) 


s_add_i32 s1, s23, s24 // (op 1781) 


s_ashr_i32 s23, s1, s22 // (op 1781) 


s_lshl_b32 s1, s48, 24 // (op 1781) 
s_ashr_i32 s22, s1, 31 // (op 1781) 

s_xor_b32 s1, s23, s22 // (op 1781) 

s_sub_i32 s23, s1, s22 // (op 1781) 


s_mul_i32 s1, s47, s15 // (op 1781) 
s_add_u32 s24, s20, 32 // (op 1781) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1781) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1781) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1781) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 1781) 

s_sub_i32 s22, s24, s1 // (op 1781) 

s_mul_i32 s24, s23, s22 // (op 1781) 


s_sub_u32 s25, s19, s24 // (op 1781) 


s_add_u32 s19, s25, s1 // (op 1781) 


s_cmp_eq_i32 s18, 0 // (op 1781) 
s_mov_b32 s1, scc // (op 1781) 

s_cmp_eq_i32 s1, 0 // (op 1781) 
s_cselect_b32 s18, s19, s0 // (op 1781) 



v_lshl_add_u32 v16, s18, 5, v14 // (op 1781) 


v_mov_b32 v19, 0 // (op 1781) convert
v_mov_b32 v18, v16 // (op 1781) convert

v_mul_lo_u32 v17, v15, s27 // (op 1781) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 1781) most significant: high of low * low
v_mul_lo_u32 v16, v15, s26 // (op 1781) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 1781) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 1781) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 1781) most significant half



s_mul_i32 s1, s26, s31 // (op 1783) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1783) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1783) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1783) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1783) most significant: sum
s_add_u32 s1, s1, s19 // (op 1783) most significant: sum


s_mov_b32 s52, s28 // (op 1783) reg expression
s_mov_b32 s53, s29 // (op 1783) reg expression
s_mov_b32 s54, s0 // (op 1783) reg expression
s_mov_b32 s55, 131072 // (op 1783) call()

s_mul_hi_i32 s0, s16, s10 // (op 1826) 
v_lshrrev_b32 v16, 1, v1 // (op 1826) 
s_add_i32 s1, s0, s16 // (op 1826) 

v_and_b32 v17, -2, v1 // (op 1826) 
s_ashr_i32 s0, s1, 31 // (op 1826) 
v_sub_u32 v18, v1, v17 // (op 1826) 

s_cmp_eq_i32 s10, 0 // (op 1826) 
s_mov_b32 s18, scc // (op 1826) 
v_lshlrev_b32 v17, 6, v18 // (op 1826) 

s_cmp_eq_i32 s18, 0 // (op 1826) 
s_cselect_b32 s19, 0, -1 // (op 1826) 

v_mov_b32 v19, 0 // (op 1826) convert
v_mov_b32 v18, v17 // (op 1826) convert

s_and_b32 s18, s11, 31 // (op 1826) 
s_lshl_b32 s22, 1, s18 // (op 1826) 
s_add_i32 s23, s22, s19 // (op 1826) 


s_and_b32 s19, s0, s23 // (op 1826) 


s_add_i32 s0, s1, s19 // (op 1826) 


s_ashr_i32 s1, s0, s18 // (op 1826) 


s_lshl_b32 s0, s11, 24 // (op 1826) 
s_ashr_i32 s18, s0, 31 // (op 1826) 

s_xor_b32 s0, s1, s18 // (op 1826) 

s_sub_i32 s1, s0, s18 // (op 1826) 


s_mul_hi_i32 s0, s1, s12 // (op 1826) 
s_add_i32 s18, s0, s1 // (op 1826) 

s_ashr_i32 s0, s18, 31 // (op 1826) 
s_cmp_eq_i32 s12, 0 // (op 1826) 
s_mov_b32 s19, scc // (op 1826) 
s_cmp_eq_i32 s19, 0 // (op 1826) 
s_cselect_b32 s22, 0, -1 // (op 1826) 

s_and_b32 s19, s13, 31 // (op 1826) 
s_lshl_b32 s23, 1, s19 // (op 1826) 
s_add_i32 s24, s23, s22 // (op 1826) 


s_and_b32 s25, s0, s24 // (op 1826) 


s_add_i32 s0, s18, s25 // (op 1826) 


s_ashr_i32 s18, s0, s19 // (op 1826) 


s_lshl_b32 s0, s13, 24 // (op 1826) 
s_ashr_i32 s19, s0, 31 // (op 1826) 

s_xor_b32 s0, s18, s19 // (op 1826) 

s_sub_i32 s18, s0, s19 // (op 1826) 


s_add_u32 s50, s6, 128 // (op 1826) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1826) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1826) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1826) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 1826) 

s_mul_i32 s0, s18, s24 // (op 1826) 

s_sub_i32 s18, s1, s0 // (op 1826) 


s_mul_hi_i32 s19, s16, s9 // (op 1826) 
s_add_i32 s0, s19, s16 // (op 1826) 

s_ashr_i32 s1, s0, 31 // (op 1826) 
s_cmp_eq_i32 s9, 0 // (op 1826) 
s_mov_b32 s19, scc // (op 1826) 
s_cmp_eq_i32 s19, 0 // (op 1826) 
s_cselect_b32 s22, 0, -1 // (op 1826) 

s_and_b32 s19, s46, 31 // (op 1826) 
s_lshl_b32 s23, 1, s19 // (op 1826) 
s_add_i32 s49, s23, s22 // (op 1826) 


s_and_b32 s22, s1, s49 // (op 1826) 


s_add_i32 s1, s0, s22 // (op 1826) 


s_ashr_i32 s0, s1, s19 // (op 1826) 


s_lshl_b32 s1, s46, 24 // (op 1826) 
s_ashr_i32 s19, s1, 31 // (op 1826) 

s_xor_b32 s1, s0, s19 // (op 1826) 

s_sub_i32 s0, s1, s19 // (op 1826) 


s_mul_i32 s1, s15, s24 // (op 1826) 

s_mul_i32 s19, s8, s1 // (op 1826) 

s_mul_i32 s1, s0, s19 // (op 1826) 

s_sub_u32 s19, s16, s1 // (op 1826) 

s_mul_hi_i32 s1, s19, s14 // (op 1826) 
s_add_i32 s22, s1, s19 // (op 1826) 


s_ashr_i32 s1, s22, 31 // (op 1826) 
s_cmp_eq_i32 s14, 0 // (op 1826) 
s_mov_b32 s19, scc // (op 1826) 
s_cmp_eq_i32 s19, 0 // (op 1826) 
s_cselect_b32 s23, 0, -1 // (op 1826) 

s_and_b32 s19, s48, 31 // (op 1826) 
s_lshl_b32 s24, 1, s19 // (op 1826) 
s_add_i32 s25, s24, s23 // (op 1826) 


s_and_b32 s23, s1, s25 // (op 1826) 


s_add_i32 s1, s22, s23 // (op 1826) 


s_ashr_i32 s22, s1, s19 // (op 1826) 


s_lshl_b32 s1, s48, 24 // (op 1826) 
s_ashr_i32 s19, s1, 31 // (op 1826) 

s_xor_b32 s1, s22, s19 // (op 1826) 

s_sub_i32 s22, s1, s19 // (op 1826) 


s_cmp_eq_i32 s0, 0 // (op 1826) 
s_mov_b32 s1, scc // (op 1826) 

s_cmp_eq_i32 s1, 0 // (op 1826) 
s_cselect_b32 s0, s22, s18 // (op 1826) 



v_lshl_add_u32 v17, s0, 7, v16 // (op 1826) 


v_mul_lo_u32 v21, v17, s33 // (op 1826) most significant: low of low * high
v_mul_hi_u32 v16, v17, s32 // (op 1826) most significant: high of low * low
v_mul_lo_u32 v20, v17, s32 // (op 1826) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 1826) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 1826) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 1826) most significant half



v_mul_lo_u32 v19, v17, 6 // (op 1826) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 1826) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 1826) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 1826) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 1826) 

v_mov_b32 v18, v16 // (op 1826) convert

s_mul_i32 s1, s7, 6 // (op 1828) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 1828) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1828) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1828) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1828) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1828) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1828) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1828) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1828) most significant: sum
s_add_u32 s19, s19, s23 // (op 1828) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1828) 

s_mov_b32 s64, s34 // (op 1828) reg expression
s_mov_b32 s65, s35 // (op 1828) reg expression
s_mov_b32 s66, s0 // (op 1828) reg expression
s_mov_b32 s67, 131072 // (op 1828) call()

s_mul_hi_i32 s0, s16, s10 // (op 1845) 
v_lshrrev_b32 v16, 1, v1 // (op 1845) 
s_add_i32 s1, s0, s16 // (op 1845) 

v_and_b32 v17, -2, v1 // (op 1845) 
s_ashr_i32 s0, s1, 31 // (op 1845) 
v_sub_u32 v19, v1, v17 // (op 1845) 

s_cmp_eq_i32 s10, 0 // (op 1845) 
s_mov_b32 s18, scc // (op 1845) 
v_lshlrev_b32 v17, 6, v19 // (op 1845) 

s_cmp_eq_i32 s18, 0 // (op 1845) 
s_cselect_b32 s19, 0, -1 // (op 1845) 

v_mov_b32 v21, 0 // (op 1845) convert
v_mov_b32 v20, v17 // (op 1845) convert

s_and_b32 s18, s11, 31 // (op 1845) 
s_lshl_b32 s22, 1, s18 // (op 1845) 
s_add_i32 s23, s22, s19 // (op 1845) 


s_and_b32 s19, s0, s23 // (op 1845) 


s_add_i32 s0, s1, s19 // (op 1845) 


s_ashr_i32 s1, s0, s18 // (op 1845) 


s_lshl_b32 s0, s11, 24 // (op 1845) 
s_ashr_i32 s18, s0, 31 // (op 1845) 

s_xor_b32 s0, s1, s18 // (op 1845) 

s_sub_i32 s1, s0, s18 // (op 1845) 


s_mul_hi_i32 s0, s1, s12 // (op 1845) 
s_add_i32 s18, s0, s1 // (op 1845) 

s_ashr_i32 s0, s18, 31 // (op 1845) 
s_cmp_eq_i32 s12, 0 // (op 1845) 
s_mov_b32 s19, scc // (op 1845) 
s_cmp_eq_i32 s19, 0 // (op 1845) 
s_cselect_b32 s22, 0, -1 // (op 1845) 

s_and_b32 s19, s13, 31 // (op 1845) 
s_lshl_b32 s23, 1, s19 // (op 1845) 
s_add_i32 s24, s23, s22 // (op 1845) 


s_and_b32 s25, s0, s24 // (op 1845) 


s_add_i32 s0, s18, s25 // (op 1845) 


s_ashr_i32 s18, s0, s19 // (op 1845) 


s_lshl_b32 s0, s13, 24 // (op 1845) 
s_ashr_i32 s19, s0, 31 // (op 1845) 

s_xor_b32 s0, s18, s19 // (op 1845) 

s_sub_i32 s18, s0, s19 // (op 1845) 


s_add_u32 s50, s6, 128 // (op 1845) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1845) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1845) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1845) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 1845) 

s_mul_i32 s0, s18, s24 // (op 1845) 

s_sub_i32 s18, s1, s0 // (op 1845) 


s_mul_hi_i32 s19, s16, s9 // (op 1845) 
s_add_i32 s0, s19, s16 // (op 1845) 

s_ashr_i32 s1, s0, 31 // (op 1845) 
s_cmp_eq_i32 s9, 0 // (op 1845) 
s_mov_b32 s19, scc // (op 1845) 
s_cmp_eq_i32 s19, 0 // (op 1845) 
s_cselect_b32 s22, 0, -1 // (op 1845) 

s_and_b32 s19, s46, 31 // (op 1845) 
s_lshl_b32 s23, 1, s19 // (op 1845) 
s_add_i32 s49, s23, s22 // (op 1845) 


s_and_b32 s22, s1, s49 // (op 1845) 


s_add_i32 s1, s0, s22 // (op 1845) 


s_ashr_i32 s0, s1, s19 // (op 1845) 


s_lshl_b32 s1, s46, 24 // (op 1845) 
s_ashr_i32 s19, s1, 31 // (op 1845) 

s_xor_b32 s1, s0, s19 // (op 1845) 

s_sub_i32 s0, s1, s19 // (op 1845) 


s_mul_i32 s1, s15, s24 // (op 1845) 

s_mul_i32 s19, s8, s1 // (op 1845) 

s_mul_i32 s1, s0, s19 // (op 1845) 

s_sub_u32 s19, s16, s1 // (op 1845) 

s_mul_hi_i32 s1, s19, s14 // (op 1845) 
s_add_i32 s22, s1, s19 // (op 1845) 


s_ashr_i32 s1, s22, 31 // (op 1845) 
s_cmp_eq_i32 s14, 0 // (op 1845) 
s_mov_b32 s19, scc // (op 1845) 
s_cmp_eq_i32 s19, 0 // (op 1845) 
s_cselect_b32 s23, 0, -1 // (op 1845) 

s_and_b32 s19, s48, 31 // (op 1845) 
s_lshl_b32 s24, 1, s19 // (op 1845) 
s_add_i32 s25, s24, s23 // (op 1845) 


s_and_b32 s23, s1, s25 // (op 1845) 


s_add_i32 s1, s22, s23 // (op 1845) 


s_ashr_i32 s22, s1, s19 // (op 1845) 


s_lshl_b32 s1, s48, 24 // (op 1845) 
s_ashr_i32 s19, s1, 31 // (op 1845) 

s_xor_b32 s1, s22, s19 // (op 1845) 

s_sub_i32 s22, s1, s19 // (op 1845) 


s_cmp_eq_i32 s0, 0 // (op 1845) 
s_mov_b32 s1, scc // (op 1845) 

s_cmp_eq_i32 s1, 0 // (op 1845) 
s_cselect_b32 s0, s22, s18 // (op 1845) 



v_lshl_add_u32 v17, s0, 7, v16 // (op 1845) 


v_mul_lo_u32 v23, v17, s33 // (op 1845) most significant: low of low * high
v_mul_hi_u32 v16, v17, s32 // (op 1845) most significant: high of low * low
v_mul_lo_u32 v22, v17, s32 // (op 1845) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 1845) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v22 // (op 1845) least significant half
v_addc_co_u32 v17, s[0:1], v21, v23, s[0:1] // (op 1845) most significant half



v_mul_lo_u32 v21, v17, 6 // (op 1845) most significant: low of high * low
v_mul_hi_u32 v19, v16, 6 // (op 1845) most significant: high of low * low
v_mul_lo_u32 v20, v16, 6 // (op 1845) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 1845) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 1845) 

v_mov_b32 v19, v16 // (op 1845) convert

s_mul_i32 s1, s7, 6 // (op 1847) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 1847) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1847) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1847) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1847) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1847) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1847) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1847) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1847) most significant: sum
s_add_u32 s19, s19, s23 // (op 1847) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1847) 

s_mov_b32 s64, s34 // (op 1847) reg expression
s_mov_b32 s65, s35 // (op 1847) reg expression
s_mov_b32 s66, s0 // (op 1847) reg expression
s_mov_b32 s67, 131072 // (op 1847) call()

s_mul_hi_i32 s0, s16, s10 // (op 1890) 
v_lshrrev_b32 v16, 7, v1 // (op 1890) 
s_add_i32 s1, s0, s16 // (op 1890) 

v_and_b32 v17, 15, v1 // (op 1890) 
s_ashr_i32 s0, s1, 31 // (op 1890) 
v_lshl_add_u32 v20, v16, 4, v17 // (op 1890) 


s_cmp_eq_i32 s10, 0 // (op 1890) 
s_mov_b32 s18, scc // (op 1890) 
v_and_b32 v16, 63, v1 // (op 1890) 
s_cmp_eq_i32 s18, 0 // (op 1890) 
s_cselect_b32 s19, 0, -1 // (op 1890) 

v_lshrrev_b32 v17, 4, v16 // (op 1890) 

s_and_b32 s18, s11, 31 // (op 1890) 
s_lshl_b32 s22, 1, s18 // (op 1890) 
s_add_i32 s23, s22, s19 // (op 1890) 


s_and_b32 s19, s0, s23 // (op 1890) 


s_add_i32 s0, s1, s19 // (op 1890) 


s_ashr_i32 s1, s0, s18 // (op 1890) 


s_lshl_b32 s0, s11, 24 // (op 1890) 
s_ashr_i32 s18, s0, 31 // (op 1890) 

s_xor_b32 s0, s1, s18 // (op 1890) 

s_sub_i32 s1, s0, s18 // (op 1890) 


s_mul_hi_i32 s0, s1, s12 // (op 1890) 
s_add_i32 s18, s0, s1 // (op 1890) 

s_ashr_i32 s0, s18, 31 // (op 1890) 
s_cmp_eq_i32 s12, 0 // (op 1890) 
s_mov_b32 s19, scc // (op 1890) 
s_cmp_eq_i32 s19, 0 // (op 1890) 
s_cselect_b32 s22, 0, -1 // (op 1890) 

s_and_b32 s19, s13, 31 // (op 1890) 
s_lshl_b32 s23, 1, s19 // (op 1890) 
s_add_i32 s24, s23, s22 // (op 1890) 


s_and_b32 s25, s0, s24 // (op 1890) 


s_add_i32 s0, s18, s25 // (op 1890) 


s_ashr_i32 s18, s0, s19 // (op 1890) 


s_lshl_b32 s0, s13, 24 // (op 1890) 
s_ashr_i32 s19, s0, 31 // (op 1890) 

s_xor_b32 s0, s18, s19 // (op 1890) 

s_sub_i32 s18, s0, s19 // (op 1890) 


s_mul_i32 s0, s1, s15 // (op 1890) 

s_sub_u32 s1, s16, s0 // (op 1890) 

s_mul_i32 s0, s18, s15 // (op 1890) 
s_add_u32 s0, s0, s1 // (op 1890) 


s_mul_hi_i32 s1, s16, s9 // (op 1890) 
s_add_i32 s18, s1, s16 // (op 1890) 

s_ashr_i32 s1, s18, 31 // (op 1890) 
s_cmp_eq_i32 s9, 0 // (op 1890) 
s_mov_b32 s19, scc // (op 1890) 
s_cmp_eq_i32 s19, 0 // (op 1890) 
s_cselect_b32 s22, 0, -1 // (op 1890) 

s_and_b32 s19, s46, 31 // (op 1890) 
s_lshl_b32 s23, 1, s19 // (op 1890) 
s_add_i32 s24, s23, s22 // (op 1890) 


s_and_b32 s25, s1, s24 // (op 1890) 


s_add_i32 s1, s18, s25 // (op 1890) 


s_ashr_i32 s18, s1, s19 // (op 1890) 


s_lshl_b32 s1, s46, 24 // (op 1890) 
s_ashr_i32 s19, s1, 31 // (op 1890) 

s_xor_b32 s1, s18, s19 // (op 1890) 

s_sub_i32 s18, s1, s19 // (op 1890) 


s_add_u32 s50, s6, 128 // (op 1890) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1890) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1890) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1890) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 1890) 

s_mul_i32 s1, s15, s24 // (op 1890) 

s_mul_i32 s19, s8, s1 // (op 1890) 

s_mul_i32 s1, s18, s19 // (op 1890) 

s_sub_u32 s19, s16, s1 // (op 1890) 

s_mov_b32 s1, s19 // (op 1890) 
s_mul_hi_i32 s22, s1, s14 // (op 1890) 
s_add_i32 s23, s22, s1 // (op 1890) 


s_ashr_i32 s1, s23, 31 // (op 1890) 
s_cmp_eq_i32 s14, 0 // (op 1890) 
s_mov_b32 s22, scc // (op 1890) 
s_cmp_eq_i32 s22, 0 // (op 1890) 
s_cselect_b32 s24, 0, -1 // (op 1890) 

s_and_b32 s22, s48, 31 // (op 1890) 
s_lshl_b32 s25, 1, s22 // (op 1890) 
s_add_i32 s49, s25, s24 // (op 1890) 


s_and_b32 s24, s1, s49 // (op 1890) 


s_add_i32 s1, s23, s24 // (op 1890) 


s_ashr_i32 s23, s1, s22 // (op 1890) 


s_lshl_b32 s1, s48, 24 // (op 1890) 
s_ashr_i32 s22, s1, 31 // (op 1890) 

s_xor_b32 s1, s23, s22 // (op 1890) 

s_sub_i32 s23, s1, s22 // (op 1890) 


s_mul_i32 s1, s47, s15 // (op 1890) 
s_add_u32 s24, s20, 32 // (op 1890) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1890) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1890) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1890) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 1890) 

s_sub_i32 s22, s24, s1 // (op 1890) 

s_mul_i32 s24, s23, s22 // (op 1890) 


s_sub_u32 s25, s19, s24 // (op 1890) 


s_add_u32 s19, s25, s1 // (op 1890) 


s_cmp_eq_i32 s18, 0 // (op 1890) 
s_mov_b32 s1, scc // (op 1890) 

s_cmp_eq_i32 s1, 0 // (op 1890) 
s_cselect_b32 s18, s19, s0 // (op 1890) 



v_lshl_add_u32 v16, s18, 5, v20 // (op 1890) 


v_mul_lo_u32 v20, v17, s36 // (op 1890) 

v_add_u32 v17, v16, v20 // (op 1890) 


s_mul_i32 s1, s36, s41 // (op 1892) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1892) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1892) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1892) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1892) most significant: sum
s_add_u32 s1, s1, s19 // (op 1892) most significant: sum


s_mov_b32 s68, s38 // (op 1892) reg expression
s_mov_b32 s69, s39 // (op 1892) reg expression
s_mov_b32 s70, s0 // (op 1892) reg expression
s_mov_b32 s71, 131072 // (op 1892) call()

s_mul_hi_i32 s0, s16, s10 // (op 1909) 
v_lshrrev_b32 v16, 7, v1 // (op 1909) 
s_add_i32 s1, s0, s16 // (op 1909) 

v_and_b32 v20, 15, v1 // (op 1909) 
s_ashr_i32 s0, s1, 31 // (op 1909) 
v_lshl_add_u32 v21, v16, 4, v20 // (op 1909) 


s_cmp_eq_i32 s10, 0 // (op 1909) 
s_mov_b32 s18, scc // (op 1909) 
v_and_b32 v16, 63, v1 // (op 1909) 
s_cmp_eq_i32 s18, 0 // (op 1909) 
s_cselect_b32 s19, 0, -1 // (op 1909) 

v_lshrrev_b32 v20, 4, v16 // (op 1909) 

s_and_b32 s18, s11, 31 // (op 1909) 
s_lshl_b32 s22, 1, s18 // (op 1909) 
s_add_i32 s23, s22, s19 // (op 1909) 


s_and_b32 s19, s0, s23 // (op 1909) 


s_add_i32 s0, s1, s19 // (op 1909) 


s_ashr_i32 s1, s0, s18 // (op 1909) 


s_lshl_b32 s0, s11, 24 // (op 1909) 
s_ashr_i32 s18, s0, 31 // (op 1909) 

s_xor_b32 s0, s1, s18 // (op 1909) 

s_sub_i32 s1, s0, s18 // (op 1909) 


s_mul_hi_i32 s0, s1, s12 // (op 1909) 
s_add_i32 s18, s0, s1 // (op 1909) 

s_ashr_i32 s0, s18, 31 // (op 1909) 
s_cmp_eq_i32 s12, 0 // (op 1909) 
s_mov_b32 s19, scc // (op 1909) 
s_cmp_eq_i32 s19, 0 // (op 1909) 
s_cselect_b32 s22, 0, -1 // (op 1909) 

s_and_b32 s19, s13, 31 // (op 1909) 
s_lshl_b32 s23, 1, s19 // (op 1909) 
s_add_i32 s24, s23, s22 // (op 1909) 


s_and_b32 s25, s0, s24 // (op 1909) 


s_add_i32 s0, s18, s25 // (op 1909) 


s_ashr_i32 s18, s0, s19 // (op 1909) 


s_lshl_b32 s0, s13, 24 // (op 1909) 
s_ashr_i32 s19, s0, 31 // (op 1909) 

s_xor_b32 s0, s18, s19 // (op 1909) 

s_sub_i32 s18, s0, s19 // (op 1909) 


s_mul_i32 s0, s1, s15 // (op 1909) 

s_sub_u32 s1, s16, s0 // (op 1909) 

s_mul_i32 s0, s18, s15 // (op 1909) 
s_add_u32 s0, s0, s1 // (op 1909) 


s_mul_hi_i32 s1, s16, s9 // (op 1909) 
s_add_i32 s18, s1, s16 // (op 1909) 

s_ashr_i32 s1, s18, 31 // (op 1909) 
s_cmp_eq_i32 s9, 0 // (op 1909) 
s_mov_b32 s19, scc // (op 1909) 
s_cmp_eq_i32 s19, 0 // (op 1909) 
s_cselect_b32 s22, 0, -1 // (op 1909) 

s_and_b32 s19, s46, 31 // (op 1909) 
s_lshl_b32 s23, 1, s19 // (op 1909) 
s_add_i32 s24, s23, s22 // (op 1909) 


s_and_b32 s25, s1, s24 // (op 1909) 


s_add_i32 s1, s18, s25 // (op 1909) 


s_ashr_i32 s18, s1, s19 // (op 1909) 


s_lshl_b32 s1, s46, 24 // (op 1909) 
s_ashr_i32 s19, s1, 31 // (op 1909) 

s_xor_b32 s1, s18, s19 // (op 1909) 

s_sub_i32 s18, s1, s19 // (op 1909) 


s_add_u32 s50, s6, 128 // (op 1909) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1909) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1909) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1909) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 1909) 

s_mul_i32 s1, s15, s24 // (op 1909) 

s_mul_i32 s19, s8, s1 // (op 1909) 

s_mul_i32 s1, s18, s19 // (op 1909) 

s_sub_u32 s19, s16, s1 // (op 1909) 

s_mov_b32 s1, s19 // (op 1909) 
s_mul_hi_i32 s22, s1, s14 // (op 1909) 
s_add_i32 s23, s22, s1 // (op 1909) 


s_ashr_i32 s1, s23, 31 // (op 1909) 
s_cmp_eq_i32 s14, 0 // (op 1909) 
s_mov_b32 s22, scc // (op 1909) 
s_cmp_eq_i32 s22, 0 // (op 1909) 
s_cselect_b32 s24, 0, -1 // (op 1909) 

s_and_b32 s22, s48, 31 // (op 1909) 
s_lshl_b32 s25, 1, s22 // (op 1909) 
s_add_i32 s49, s25, s24 // (op 1909) 


s_and_b32 s24, s1, s49 // (op 1909) 


s_add_i32 s1, s23, s24 // (op 1909) 


s_ashr_i32 s23, s1, s22 // (op 1909) 


s_lshl_b32 s1, s48, 24 // (op 1909) 
s_ashr_i32 s22, s1, 31 // (op 1909) 

s_xor_b32 s1, s23, s22 // (op 1909) 

s_sub_i32 s23, s1, s22 // (op 1909) 


s_mul_i32 s1, s47, s15 // (op 1909) 
s_add_u32 s24, s20, 32 // (op 1909) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1909) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1909) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1909) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 1909) 

s_sub_i32 s22, s24, s1 // (op 1909) 

s_mul_i32 s24, s23, s22 // (op 1909) 


s_sub_u32 s25, s19, s24 // (op 1909) 


s_add_u32 s19, s25, s1 // (op 1909) 


s_cmp_eq_i32 s18, 0 // (op 1909) 
s_mov_b32 s1, scc // (op 1909) 

s_cmp_eq_i32 s1, 0 // (op 1909) 
s_cselect_b32 s18, s19, s0 // (op 1909) 



v_lshl_add_u32 v16, s18, 5, v21 // (op 1909) 


v_mul_lo_u32 v21, v20, s36 // (op 1909) 

v_add_u32 v20, v16, v21 // (op 1909) 


s_mul_i32 s1, s36, s41 // (op 1911) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1911) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1911) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1911) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1911) most significant: sum
s_add_u32 s1, s1, s19 // (op 1911) most significant: sum


s_mov_b32 s68, s38 // (op 1911) reg expression
s_mov_b32 s69, s39 // (op 1911) reg expression
s_mov_b32 s70, s0 // (op 1911) reg expression
s_mov_b32 s71, 131072 // (op 1911) call()

s_mul_hi_i32 s0, s16, s10 // (op 1954) 
v_lshrrev_b32 v16, 2, v1 // (op 1954) 
s_add_i32 s1, s0, s16 // (op 1954) 

v_lshlrev_b32 v21, 1, v16 // (op 1954) 

s_ashr_i32 s0, s1, 31 // (op 1954) 
v_and_b32 v16, -4, v1 // (op 1954) 
s_cmp_eq_i32 s10, 0 // (op 1954) 
s_mov_b32 s18, scc // (op 1954) 
v_sub_u32 v22, v1, v16 // (op 1954) 

s_cmp_eq_i32 s18, 0 // (op 1954) 
s_cselect_b32 s19, 0, -1 // (op 1954) 

s_and_b32 s18, s11, 31 // (op 1954) 
s_lshl_b32 s22, 1, s18 // (op 1954) 
s_add_i32 s23, s22, s19 // (op 1954) 


s_and_b32 s19, s0, s23 // (op 1954) 


s_add_i32 s0, s1, s19 // (op 1954) 


s_ashr_i32 s1, s0, s18 // (op 1954) 


s_lshl_b32 s0, s11, 24 // (op 1954) 
s_ashr_i32 s18, s0, 31 // (op 1954) 

s_xor_b32 s0, s1, s18 // (op 1954) 

s_sub_i32 s1, s0, s18 // (op 1954) 


s_mul_hi_i32 s0, s1, s12 // (op 1954) 
s_add_i32 s18, s0, s1 // (op 1954) 

s_ashr_i32 s0, s18, 31 // (op 1954) 
s_cmp_eq_i32 s12, 0 // (op 1954) 
s_mov_b32 s19, scc // (op 1954) 
s_cmp_eq_i32 s19, 0 // (op 1954) 
s_cselect_b32 s22, 0, -1 // (op 1954) 

s_and_b32 s19, s13, 31 // (op 1954) 
s_lshl_b32 s23, 1, s19 // (op 1954) 
s_add_i32 s24, s23, s22 // (op 1954) 


s_and_b32 s25, s0, s24 // (op 1954) 


s_add_i32 s0, s18, s25 // (op 1954) 


s_ashr_i32 s18, s0, s19 // (op 1954) 


s_lshl_b32 s0, s13, 24 // (op 1954) 
s_ashr_i32 s19, s0, 31 // (op 1954) 

s_xor_b32 s0, s18, s19 // (op 1954) 

s_sub_i32 s18, s0, s19 // (op 1954) 


s_add_u32 s50, s6, 128 // (op 1954) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1954) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1954) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1954) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 1954) 

s_mul_i32 s0, s18, s24 // (op 1954) 

s_sub_i32 s18, s1, s0 // (op 1954) 


s_mul_hi_i32 s19, s16, s9 // (op 1954) 
s_add_i32 s0, s19, s16 // (op 1954) 

s_ashr_i32 s1, s0, 31 // (op 1954) 
s_cmp_eq_i32 s9, 0 // (op 1954) 
s_mov_b32 s19, scc // (op 1954) 
s_cmp_eq_i32 s19, 0 // (op 1954) 
s_cselect_b32 s22, 0, -1 // (op 1954) 

s_and_b32 s19, s46, 31 // (op 1954) 
s_lshl_b32 s23, 1, s19 // (op 1954) 
s_add_i32 s49, s23, s22 // (op 1954) 


s_and_b32 s22, s1, s49 // (op 1954) 


s_add_i32 s1, s0, s22 // (op 1954) 


s_ashr_i32 s0, s1, s19 // (op 1954) 


s_lshl_b32 s1, s46, 24 // (op 1954) 
s_ashr_i32 s19, s1, 31 // (op 1954) 

s_xor_b32 s1, s0, s19 // (op 1954) 

s_sub_i32 s0, s1, s19 // (op 1954) 


s_mul_i32 s1, s15, s24 // (op 1954) 

s_mul_i32 s19, s8, s1 // (op 1954) 

s_mul_i32 s1, s0, s19 // (op 1954) 

s_sub_u32 s19, s16, s1 // (op 1954) 

s_mul_hi_i32 s1, s19, s14 // (op 1954) 
s_add_i32 s22, s1, s19 // (op 1954) 


s_ashr_i32 s1, s22, 31 // (op 1954) 
s_cmp_eq_i32 s14, 0 // (op 1954) 
s_mov_b32 s19, scc // (op 1954) 
s_cmp_eq_i32 s19, 0 // (op 1954) 
s_cselect_b32 s23, 0, -1 // (op 1954) 

s_and_b32 s19, s48, 31 // (op 1954) 
s_lshl_b32 s24, 1, s19 // (op 1954) 
s_add_i32 s25, s24, s23 // (op 1954) 


s_and_b32 s23, s1, s25 // (op 1954) 


s_add_i32 s1, s22, s23 // (op 1954) 


s_ashr_i32 s22, s1, s19 // (op 1954) 


s_lshl_b32 s1, s48, 24 // (op 1954) 
s_ashr_i32 s19, s1, 31 // (op 1954) 

s_xor_b32 s1, s22, s19 // (op 1954) 

s_sub_i32 s22, s1, s19 // (op 1954) 


s_cmp_eq_i32 s0, 0 // (op 1954) 
s_mov_b32 s1, scc // (op 1954) 

s_cmp_eq_i32 s1, 0 // (op 1954) 
s_cselect_b32 s0, s22, s18 // (op 1954) 



v_lshl_add_u32 v16, s0, 7, v21 // (op 1954) 


v_mul_lo_u32 v21, v16, s42 // (op 1954) 

v_add_u32 v16, v22, v21 // (op 1954) 


s_mul_i32 s1, s42, s7 // (op 1956) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1956) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1956) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1956) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1956) most significant: sum
s_add_u32 s1, s1, s19 // (op 1956) most significant: sum


s_mov_b32 s72, s44 // (op 1956) reg expression
s_mov_b32 s73, s45 // (op 1956) reg expression
s_mov_b32 s74, s0 // (op 1956) reg expression
s_mov_b32 s75, 131072 // (op 1956) call()

s_mul_hi_i32 s0, s16, s10 // (op 1973) 
v_lshrrev_b32 v21, 2, v1 // (op 1973) 
s_add_i32 s1, s0, s16 // (op 1973) 

v_lshlrev_b32 v22, 1, v21 // (op 1973) 

s_ashr_i32 s0, s1, 31 // (op 1973) 
v_and_b32 v21, -4, v1 // (op 1973) 
s_cmp_eq_i32 s10, 0 // (op 1973) 
s_mov_b32 s18, scc // (op 1973) 
v_sub_u32 v23, v1, v21 // (op 1973) 

s_cmp_eq_i32 s18, 0 // (op 1973) 
s_cselect_b32 s19, 0, -1 // (op 1973) 

s_and_b32 s18, s11, 31 // (op 1973) 
s_lshl_b32 s22, 1, s18 // (op 1973) 
s_add_i32 s23, s22, s19 // (op 1973) 


s_and_b32 s19, s0, s23 // (op 1973) 


s_add_i32 s0, s1, s19 // (op 1973) 


s_ashr_i32 s1, s0, s18 // (op 1973) 


s_lshl_b32 s0, s11, 24 // (op 1973) 
s_ashr_i32 s18, s0, 31 // (op 1973) 

s_xor_b32 s0, s1, s18 // (op 1973) 

s_sub_i32 s1, s0, s18 // (op 1973) 


s_mul_hi_i32 s0, s1, s12 // (op 1973) 
s_add_i32 s18, s0, s1 // (op 1973) 

s_ashr_i32 s0, s18, 31 // (op 1973) 
s_cmp_eq_i32 s12, 0 // (op 1973) 
s_mov_b32 s19, scc // (op 1973) 
s_cmp_eq_i32 s19, 0 // (op 1973) 
s_cselect_b32 s22, 0, -1 // (op 1973) 

s_and_b32 s19, s13, 31 // (op 1973) 
s_lshl_b32 s23, 1, s19 // (op 1973) 
s_add_i32 s24, s23, s22 // (op 1973) 


s_and_b32 s25, s0, s24 // (op 1973) 


s_add_i32 s0, s18, s25 // (op 1973) 


s_ashr_i32 s18, s0, s19 // (op 1973) 


s_lshl_b32 s0, s13, 24 // (op 1973) 
s_ashr_i32 s19, s0, 31 // (op 1973) 

s_xor_b32 s0, s18, s19 // (op 1973) 

s_sub_i32 s18, s0, s19 // (op 1973) 


s_add_u32 s50, s6, 128 // (op 1973) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1973) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1973) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1973) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 1973) 

s_mul_i32 s0, s18, s24 // (op 1973) 

s_sub_i32 s18, s1, s0 // (op 1973) 


s_mul_hi_i32 s19, s16, s9 // (op 1973) 
s_add_i32 s0, s19, s16 // (op 1973) 

s_ashr_i32 s1, s0, 31 // (op 1973) 
s_cmp_eq_i32 s9, 0 // (op 1973) 
s_mov_b32 s19, scc // (op 1973) 
s_cmp_eq_i32 s19, 0 // (op 1973) 
s_cselect_b32 s22, 0, -1 // (op 1973) 

s_and_b32 s19, s46, 31 // (op 1973) 
s_lshl_b32 s23, 1, s19 // (op 1973) 
s_add_i32 s49, s23, s22 // (op 1973) 


s_and_b32 s22, s1, s49 // (op 1973) 


s_add_i32 s1, s0, s22 // (op 1973) 


s_ashr_i32 s0, s1, s19 // (op 1973) 


s_lshl_b32 s1, s46, 24 // (op 1973) 
s_ashr_i32 s19, s1, 31 // (op 1973) 

s_xor_b32 s1, s0, s19 // (op 1973) 

s_sub_i32 s0, s1, s19 // (op 1973) 


s_mul_i32 s1, s15, s24 // (op 1973) 

s_mul_i32 s19, s8, s1 // (op 1973) 

s_mul_i32 s1, s0, s19 // (op 1973) 

s_sub_u32 s19, s16, s1 // (op 1973) 

s_mul_hi_i32 s1, s19, s14 // (op 1973) 
s_add_i32 s22, s1, s19 // (op 1973) 


s_ashr_i32 s1, s22, 31 // (op 1973) 
s_cmp_eq_i32 s14, 0 // (op 1973) 
s_mov_b32 s19, scc // (op 1973) 
s_cmp_eq_i32 s19, 0 // (op 1973) 
s_cselect_b32 s23, 0, -1 // (op 1973) 

s_and_b32 s19, s48, 31 // (op 1973) 
s_lshl_b32 s24, 1, s19 // (op 1973) 
s_add_i32 s25, s24, s23 // (op 1973) 


s_and_b32 s23, s1, s25 // (op 1973) 


s_add_i32 s1, s22, s23 // (op 1973) 


s_ashr_i32 s22, s1, s19 // (op 1973) 


s_lshl_b32 s1, s48, 24 // (op 1973) 
s_ashr_i32 s19, s1, 31 // (op 1973) 

s_xor_b32 s1, s22, s19 // (op 1973) 

s_sub_i32 s22, s1, s19 // (op 1973) 


s_cmp_eq_i32 s0, 0 // (op 1973) 
s_mov_b32 s1, scc // (op 1973) 

s_cmp_eq_i32 s1, 0 // (op 1973) 
s_cselect_b32 s0, s22, s18 // (op 1973) 



v_lshl_add_u32 v21, s0, 7, v22 // (op 1973) 


v_mul_lo_u32 v22, v21, s42 // (op 1973) 

v_add_u32 v21, v23, v22 // (op 1973) 


s_mul_i32 s1, s42, s7 // (op 1975) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1975) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1975) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1975) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1975) most significant: sum
s_add_u32 s1, s1, s19 // (op 1975) most significant: sum


s_mov_b32 s72, s44 // (op 1975) reg expression
s_mov_b32 s73, s45 // (op 1975) reg expression
s_mov_b32 s74, s0 // (op 1975) reg expression
s_mov_b32 s75, 131072 // (op 1975) call()


v_mov_b64 v[22:23], s[36:37] // (op 1560) convert

v_mov_b32 v24, v17 // (op 1560) reg expression
v_mov_b32 v25, v24 // (op 1560) 
buffer_load_ubyte v26, v25, s[68:71], 0 offen offset: 0 // (op 1560) Load value






buffer_load_dwordx4 v[28:31], v18, s[64:67], 0 offen // (op 1564) Load value
buffer_load_dwordx2 v[32:33], v18, s[64:67], 0 offen offset:16 // (op 1564) Load value
buffer_load_dwordx4 v[34:37], v18, s[64:67], 0 offen offset:24 // (op 1564) Load value
buffer_load_dwordx2 v[38:39], v18, s[64:67], 0 offen offset:40 // (op 1564) Load value

v_mov_b64 v[22:23], s[42:43] // (op 1567) convert

v_mov_b32 v27, v16 // (op 1567) 
buffer_load_ubyte v24, v27, s[72:75], 0 offen offset: 0 // (op 1567) Load value
v_add_u32 v27, v27, v22 // (op 1567) 
buffer_load_ubyte v25, v27, s[72:75], 0 offen offset: 0 // (op 1567) Load value


v_and_b32 v27, -2, v1 // (op 2039) 
v_sub_u32 v22, v1, v27 // (op 2039) 

v_lshlrev_b32 v23, 4, v22 // (op 2039) 

v_lshrrev_b32 v22, 1, v1 // (op 2039) 
v_lshl_add_u32 v27, v22, 5, v23 // (op 2039) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2039) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[22:23], v[12:13] // (op 1715) reg expression

s_nop 0
s_mov_b32 m0, s0 // (op 1715) reg expression
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 1715) Load value direct to lds



v_mov_b64 v[22:23], s[36:37] // (op 1574) convert

s_lshl_b32 s0, s36, 2 // (op 1574) 
v_add_u32 v27, v17, s0 // (op 1574) 

v_mov_b32 v40, v27 // (op 1574) 
buffer_load_ubyte v41, v40, s[68:71], 0 offen offset: 0 // (op 1574) Load value







buffer_load_dwordx4 v[42:45], v18, s[64:67], 0 offen offset:96 // (op 1578) Load value
buffer_load_dwordx2 v[46:47], v18, s[64:67], 0 offen offset:112 // (op 1578) Load value
buffer_load_dwordx4 v[48:51], v18, s[64:67], 0 offen offset:120 // (op 1578) Load value
buffer_load_dwordx2 v[52:53], v18, s[64:67], 0 offen offset:136 // (op 1578) Load value


v_mov_b64 v[22:23], s[42:43] // (op 1581) convert

v_mov_b32 v27, v16 // (op 1581) 
buffer_load_ubyte v54, v27, s[72:75], 0 offen offset:4 // (op 1581) Load value
v_add_u32 v27, v27, v22 // (op 1581) 
buffer_load_ubyte v55, v27, s[72:75], 0 offen offset:4 // (op 1581) Load value



v_and_b32 v27, -2, v1 // (op 2212) 
v_sub_u32 v40, v1, v27 // (op 2212) 

v_lshlrev_b32 v27, 4, v40 // (op 2212) 

v_lshrrev_b32 v40, 1, v1 // (op 2212) 
v_lshl_add_u32 v16, v40, 5, v27 // (op 2212) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 2212) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[18:19], s[26:27], 7 // (op 1720) 
v_mov_b32 v27, s18 // (op 1720) 
v_mov_b32 v40, s19 // (op 1720) 
v_add_co_u32 v22, s[50:51], v12, v27 // (op 1720) least significant half
v_addc_co_u32 v23, s[50:51], v13, v40, s[50:51] // (op 1720) most significant half




s_add_u32 s1, 4096, s0 // (op 1720) 

s_mov_b32 m0, s1 // (op 1720) reg expression
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 1720) Load value direct to lds




v_and_b32 v27, -2, v1 // (op 2092) 
v_sub_u32 v40, v1, v27 // (op 2092) 

v_lshlrev_b32 v27, 6, v40 // (op 2092) 

v_lshrrev_b32 v40, 1, v1 // (op 2092) 
v_lshl_add_u32 v12, v40, 7, v27 // (op 2092) 


v_mul_lo_u32 v13, v12, 6 // (op 2092) 

v_lshrrev_b32 v12, 3, v13 // (op 2092) 




s_waitcnt vmcnt(14)
ds_write_b128 v12, v[28:31] offset:8192 // (op 1586) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v12, v[32:33] offset:8208 // (op 1586) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v12, v[34:37] offset:8216 // (op 1586) Store local data 
s_waitcnt vmcnt(11)
ds_write_b64 v12, v[38:39] offset:8232 // (op 1586) Store local data 

v_and_b32 v27, -4, v1 // (op 2150) 
v_sub_u32 v40, v1, v27 // (op 2150) 

v_lshrrev_b32 v27, 2, v1 // (op 2150) 
v_lshl_add_u32 v12, v27, 3, v40 // (op 2150) 





s_waitcnt vmcnt(10)
ds_write_b8 v12, v24 offset:20480 // (op 1587) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v12, v25 offset:20484 // (op 1587) Store local data 

v_and_b32 v27, -2, v1 // (op 2011) 
v_sub_u32 v40, v1, v27 // (op 2011) 

v_lshlrev_b32 v27, 4, v40 // (op 2011) 

v_lshrrev_b32 v40, 1, v1 // (op 2011) 
v_lshl_add_u32 v12, v40, 5, v27 // (op 2011) 


s_nop 0
v_readfirstlane_b32 s0, v12 // (op 2011) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -2, v1 // (op 2063) 
v_sub_u32 v40, v1, v27 // (op 2063) 

v_lshlrev_b32 v27, 6, v40 // (op 2063) 

v_lshrrev_b32 v40, 1, v1 // (op 2063) 
v_lshl_add_u32 v12, v40, 7, v27 // (op 2063) 


v_mul_lo_u32 v13, v12, 6 // (op 2063) 

v_lshrrev_b32 v12, 3, v13 // (op 2063) 

v_and_b32 v13, -4, v1 // (op 2121) 
v_sub_u32 v27, v1, v13 // (op 2121) 

v_lshrrev_b32 v13, 2, v1 // (op 2121) 
v_lshl_add_u32 v40, v13, 3, v27 // (op 2121) 


v_and_b32 v13, -2, v1 // (op 2197) 
v_sub_u32 v27, v1, v13 // (op 2197) 

v_lshlrev_b32 v13, 4, v27 // (op 2197) 

v_lshrrev_b32 v27, 1, v1 // (op 2197) 
v_lshl_add_u32 v16, v27, 5, v13 // (op 2197) 


s_nop 0
v_readfirstlane_b32 s1, v16 // (op 2197) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -2, v1 // (op 2236) 
v_sub_u32 v27, v1, v13 // (op 2236) 

v_lshlrev_b32 v13, 6, v27 // (op 2236) 

v_lshrrev_b32 v27, 1, v1 // (op 2236) 
v_lshl_add_u32 v16, v27, 7, v13 // (op 2236) 


v_mul_lo_u32 v13, v16, 6 // (op 2236) 

v_lshrrev_b32 v27, 3, v13 // (op 2236) 

v_and_b32 v13, -4, v1 // (op 2265) 
v_sub_u32 v16, v1, v13 // (op 2265) 

v_lshrrev_b32 v13, 2, v1 // (op 2265) 
v_lshl_add_u32 v17, v13, 3, v16 // (op 2265) 


s_mov_b32 s18, 0 // (op 624) call()
s_ashr_i64 s[50:51], s[30:31], 7 // (op 623) 
s_mov_b32 s19, s50 // (op 623) convert

s_and_b32 s22, s19, 4294967294 // (op 623) 

s_cmp_lt_u32 s18, s22 // (op 623) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ForLoopBottom_KLoop_623 // (op 623) Condition: Top (jump to RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ForLoopBottom_KLoop_623 if false)
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ForLoopTop_KLoop_623:
s_add_u32 s18, s18, 2 // (op 625) 
s_waitcnt vmcnt(8)
s_barrier  // (op 2329) 



ds_read_b64_tr_b8 v[56:57], v0 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:256 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:2048 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:2304 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2342) 



ds_read_b128 v[64:67], v7 offset:8192 // (op 72) Load local data 
ds_read_b64 v[68:69], v7 offset:8208 // (op 72) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2356) 



ds_read_u8 v13, v4 offset:20480 // (op 81) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[64:69], a[0:3], v26, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 84) 
ds_read_b128 v[70:73], v7 offset:11264 // (op 343) Load local data 


ds_read_b64 v[74:75], v7 offset:11280 // (op 343) Load local data 



ds_read_u8 v13, v4 offset:20608 // (op 346) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[70:75], a[4:7], v26, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 347) 
ds_read_b128 v[64:67], v7 offset:14336 // (op 499) Load local data 


ds_read_b64 v[68:69], v7 offset:14352 // (op 499) Load local data 



ds_read_u8 v13, v4 offset:20736 // (op 502) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:63], v[64:69], a[8:11], v26, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 503) 
ds_read_b128 v[70:73], v7 offset:17408 // (op 655) Load local data 


ds_read_b64 v[74:75], v7 offset:17424 // (op 655) Load local data 



ds_read_u8 v13, v4 offset:20864 // (op 658) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[70:75], a[12:15], v26, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 659) 




v_mov_b64 v[22:23], s[36:37] // (op 77) convert

v_mov_b32 v13, v20 // (op 77) 
s_lshl_b32 s19, s36, 3 // (op 77) 
v_add_u32 v16, v13, s19 // (op 77) 

v_mov_b32 v18, v16 // (op 77) 
buffer_load_ubyte v26, v18, s[68:71], 0 offen offset: 0 // (op 77) Load value





v_mov_b32 v16, v19 // (op 10) 

buffer_load_dwordx4 v[28:31], v16, s[64:67], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[32:33], v16, s[64:67], 0 offen offset:208 // (op 10) Load value
buffer_load_dwordx4 v[34:37], v16, s[64:67], 0 offen offset:216 // (op 10) Load value
buffer_load_dwordx2 v[38:39], v16, s[64:67], 0 offen offset:232 // (op 10) Load value

v_mov_b64 v[22:23], s[42:43] // (op 21) convert

v_mov_b32 v18, v21 // (op 21) 
v_mov_b32 v56, v18 // (op 21) 
buffer_load_ubyte v24, v56, s[72:75], 0 offen offset:8 // (op 21) Load value
v_add_u32 v56, v56, v22 // (op 21) 
buffer_load_ubyte v25, v56, s[72:75], 0 offen offset:8 // (op 21) Load value


s_barrier  // (op 2332) 


v_mov_b64 v[22:23], v[14:15] // (op 1699) 
s_lshl_b64 s[50:51], s[26:27], 7 // (op 1699) 
s_lshl_b64 s[22:23], s[50:51], 1 // (op 1699) 

v_mov_b32 v56, s22 // (op 1699) 
v_mov_b32 v57, s23 // (op 1699) 
v_add_co_u32 v58, s[24:25], v22, v56 // (op 1699) least significant half
v_addc_co_u32 v59, s[24:25], v23, v57, s[24:25] // (op 1699) most significant half





s_mov_b32 m0, s0 // (op 1699) reg expression
buffer_load_dwordx4 v58, s[52:55], 0 offen offset: 0 lds // (op 1699) Load value direct to lds

s_barrier  // (op 2379) 



s_waitcnt vmcnt(14)
ds_write_b128 v27, v[42:45] offset:20992 // (op 239) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v27, v[46:47] offset:21008 // (op 239) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v27, v[48:51] offset:21016 // (op 239) Store local data 
s_waitcnt vmcnt(11)
ds_write_b64 v27, v[52:53] offset:21032 // (op 239) Store local data 
s_barrier  // (op 2385) 



s_waitcnt vmcnt(10)
ds_write_b8 v17, v54 offset:33280 // (op 240) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v17, v55 offset:33284 // (op 240) Store local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 2370) 



ds_read_b64_tr_b8 v[56:57], v3 offset:4096 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v3 offset:4352 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v3 offset:6144 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v3 offset:6400 // (op 231) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2376) 



ds_read_b128 v[64:67], v11 offset:20992 // (op 233) Load local data 
ds_read_b64 v[68:69], v11 offset:21008 // (op 233) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2382) 



ds_read_u8 v70, v10 offset:33280 // (op 237) Load local data 



ds_read_b128 v[72:75], v11 offset:24064 // (op 352) Load local data 
ds_read_b64 v[76:77], v11 offset:24080 // (op 352) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[64:69], a[0:3], v41, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 238) 





ds_read_u8 v64, v10 offset:33408 // (op 355) Load local data 



ds_read_b128 v[66:69], v11 offset:27136 // (op 508) Load local data 
ds_read_b64 v[70:71], v11 offset:27152 // (op 508) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[72:77], a[4:7], v41, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 348) 





ds_read_u8 v64, v10 offset:33536 // (op 511) Load local data 



ds_read_b128 v[72:75], v11 offset:30208 // (op 664) Load local data 
ds_read_b64 v[76:77], v11 offset:30224 // (op 664) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:63], v[66:71], a[8:11], v41, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 504) 





s_barrier  // (op 2462) 
ds_read_u8 v64, v10 offset:33664 // (op 667) Load local data 
s_barrier  // (op 2466) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[72:77], a[12:15], v41, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 660) 




v_mov_b64 v[56:57], s[36:37] // (op 235) convert

s_lshl_b32 s19, s36, 2 // (op 235) 
s_mul_i32 s22, 3, s19 // (op 235) 

v_add_u32 v58, v13, s22 // (op 235) 

v_mov_b32 v59, v58 // (op 235) 
buffer_load_ubyte v41, v59, s[68:71], 0 offen offset: 0 // (op 235) Load value







s_lshl_b32 s19, s36, 3 // (op 1901) 
v_add_u32 v20, v20, s19 // (op 1901) 

buffer_load_dwordx4 v[42:45], v16, s[64:67], 0 offen offset:288 // (op 241) Load value
buffer_load_dwordx2 v[46:47], v16, s[64:67], 0 offen offset:304 // (op 241) Load value
buffer_load_dwordx4 v[48:51], v16, s[64:67], 0 offen offset:312 // (op 241) Load value
buffer_load_dwordx2 v[52:53], v16, s[64:67], 0 offen offset:328 // (op 241) Load value


v_add_u32 v19, 192, v19 // (op 1837) 
v_mov_b64 v[56:57], s[42:43] // (op 242) convert

v_mov_b32 v13, v18 // (op 242) 
buffer_load_ubyte v54, v13, s[72:75], 0 offen offset:12 // (op 242) Load value
v_add_u32 v13, v13, v56 // (op 242) 
buffer_load_ubyte v55, v13, s[72:75], 0 offen offset:12 // (op 242) Load value



s_barrier  // (op 2373) 
v_add_u32 v21, 8, v21 // (op 1965) 


s_lshl_b64 s[50:51], s[26:27], 7 // (op 1710) 
s_mul_i32 s23, 3, s51 // (op 1710) most significant: low of low * high
s_mul_i32 s22, 3, s50 // (op 1710) least significant
s_mul_hi_u32 s19, 3, s50 // (op 1710) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 1710) most significant: sum


v_mov_b32 v13, s22 // (op 1710) 
v_mov_b32 v16, s23 // (op 1710) 
v_add_co_u32 v56, s[24:25], v22, v13 // (op 1710) least significant half
v_addc_co_u32 v57, s[24:25], v23, v16, s[24:25] // (op 1710) most significant half




s_add_u32 s19, 4096, s1 // (op 1710) 

s_mov_b32 m0, s19 // (op 1710) reg expression
buffer_load_dwordx4 v56, s[52:55], 0 offen offset: 0 lds // (op 1710) Load value direct to lds



s_barrier  // (op 2345) 
s_lshl_b64 s[50:51], s[26:27], 7 // (op 1773) 



s_lshl_b64 s[22:23], s[50:51], 1 // (op 1773) 

v_mov_b32 v13, s22 // (op 1773) 
v_mov_b32 v16, s23 // (op 1773) 
v_add_co_u32 v14, s[24:25], v14, v13 // (op 1773) least significant half
v_addc_co_u32 v15, s[24:25], v15, v16, s[24:25] // (op 1773) most significant half




s_waitcnt vmcnt(14)
ds_write_b128 v12, v[28:31] offset:8192 // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v12, v[32:33] offset:8208 // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v12, v[34:37] offset:8216 // (op 13) Store local data 
s_waitcnt vmcnt(11)
ds_write_b64 v12, v[38:39] offset:8232 // (op 13) Store local data 
s_barrier  // (op 2359) 



s_waitcnt vmcnt(10)
ds_write_b8 v40, v24 offset:20480 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v40, v25 offset:20484 // (op 26) Store local data 
s_ashr_i64 s[50:51], s[30:31], 7 // (op 623) 
s_mov_b32 s19, s50 // (op 623) convert

s_and_b32 s22, s19, 4294967294 // (op 623) 

s_cmp_lt_u32 s18, s22 // (op 623) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ForLoopTop_KLoop_623 // (op 623) Condition: Bottom (jump to RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ForLoopTop_KLoop_623 if true)
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ForLoopBottom_KLoop_623:














s_barrier  // (op 2446) 
s_mul_hi_i32 s0, s16, s10 // (op 1733) 
v_and_b32 v12, -2, v1 // (op 1733) 
s_add_i32 s1, s0, s16 // (op 1733) 

v_sub_u32 v13, v1, v12 // (op 1733) 

s_ashr_i32 s0, s1, 31 // (op 1733) 
v_lshlrev_b32 v12, 4, v13 // (op 1733) 

s_cmp_eq_i32 s10, 0 // (op 1733) 
s_mov_b32 s18, scc // (op 1733) 
v_lshrrev_b32 v13, 1, v1 // (op 1733) 
s_cmp_eq_i32 s18, 0 // (op 1733) 
s_cselect_b32 s19, 0, -1 // (op 1733) 

s_and_b32 s18, s11, 31 // (op 1733) 
s_lshl_b32 s22, 1, s18 // (op 1733) 
s_add_i32 s23, s22, s19 // (op 1733) 


s_and_b32 s19, s0, s23 // (op 1733) 


s_add_i32 s0, s1, s19 // (op 1733) 


s_ashr_i32 s1, s0, s18 // (op 1733) 


s_lshl_b32 s0, s11, 24 // (op 1733) 
s_ashr_i32 s18, s0, 31 // (op 1733) 

s_xor_b32 s0, s1, s18 // (op 1733) 

s_sub_i32 s1, s0, s18 // (op 1733) 


s_mul_hi_i32 s0, s1, s12 // (op 1733) 
s_add_i32 s18, s0, s1 // (op 1733) 

s_ashr_i32 s0, s18, 31 // (op 1733) 
s_cmp_eq_i32 s12, 0 // (op 1733) 
s_mov_b32 s19, scc // (op 1733) 
s_cmp_eq_i32 s19, 0 // (op 1733) 
s_cselect_b32 s22, 0, -1 // (op 1733) 

s_and_b32 s19, s13, 31 // (op 1733) 
s_lshl_b32 s23, 1, s19 // (op 1733) 
s_add_i32 s24, s23, s22 // (op 1733) 


s_and_b32 s25, s0, s24 // (op 1733) 


s_add_i32 s0, s18, s25 // (op 1733) 


s_ashr_i32 s18, s0, s19 // (op 1733) 


s_lshl_b32 s0, s13, 24 // (op 1733) 
s_ashr_i32 s19, s0, 31 // (op 1733) 

s_xor_b32 s0, s18, s19 // (op 1733) 

s_sub_i32 s18, s0, s19 // (op 1733) 


s_mul_i32 s0, s1, s15 // (op 1733) 

s_sub_u32 s1, s16, s0 // (op 1733) 

s_mul_i32 s0, s18, s15 // (op 1733) 
s_add_u32 s0, s0, s1 // (op 1733) 


s_mul_hi_i32 s1, s16, s9 // (op 1733) 
s_add_i32 s18, s1, s16 // (op 1733) 

s_ashr_i32 s1, s18, 31 // (op 1733) 
s_cmp_eq_i32 s9, 0 // (op 1733) 
s_mov_b32 s19, scc // (op 1733) 
s_cmp_eq_i32 s19, 0 // (op 1733) 
s_cselect_b32 s22, 0, -1 // (op 1733) 

s_and_b32 s19, s46, 31 // (op 1733) 
s_lshl_b32 s23, 1, s19 // (op 1733) 
s_add_i32 s24, s23, s22 // (op 1733) 


s_and_b32 s25, s1, s24 // (op 1733) 


s_add_i32 s1, s18, s25 // (op 1733) 


s_ashr_i32 s18, s1, s19 // (op 1733) 


s_lshl_b32 s1, s46, 24 // (op 1733) 
s_ashr_i32 s19, s1, 31 // (op 1733) 

s_xor_b32 s1, s18, s19 // (op 1733) 

s_sub_i32 s18, s1, s19 // (op 1733) 


s_add_u32 s50, s6, 128 // (op 1733) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1733) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1733) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1733) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 1733) 

s_mul_i32 s1, s15, s24 // (op 1733) 

s_mul_i32 s19, s8, s1 // (op 1733) 

s_mul_i32 s1, s18, s19 // (op 1733) 

s_sub_u32 s19, s16, s1 // (op 1733) 

s_mov_b32 s1, s19 // (op 1733) 
s_mul_hi_i32 s22, s1, s14 // (op 1733) 
s_add_i32 s23, s22, s1 // (op 1733) 


s_ashr_i32 s1, s23, 31 // (op 1733) 
s_cmp_eq_i32 s14, 0 // (op 1733) 
s_mov_b32 s22, scc // (op 1733) 
s_cmp_eq_i32 s22, 0 // (op 1733) 
s_cselect_b32 s24, 0, -1 // (op 1733) 

s_and_b32 s22, s48, 31 // (op 1733) 
s_lshl_b32 s25, 1, s22 // (op 1733) 
s_add_i32 s49, s25, s24 // (op 1733) 


s_and_b32 s24, s1, s49 // (op 1733) 


s_add_i32 s1, s23, s24 // (op 1733) 


s_ashr_i32 s23, s1, s22 // (op 1733) 


s_lshl_b32 s1, s48, 24 // (op 1733) 
s_ashr_i32 s22, s1, 31 // (op 1733) 

s_xor_b32 s1, s23, s22 // (op 1733) 

s_sub_i32 s23, s1, s22 // (op 1733) 


s_mul_i32 s1, s47, s15 // (op 1733) 
s_add_u32 s24, s20, 32 // (op 1733) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1733) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1733) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1733) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 1733) 

s_sub_i32 s22, s24, s1 // (op 1733) 

s_mul_i32 s24, s23, s22 // (op 1733) 


s_sub_u32 s25, s19, s24 // (op 1733) 


s_add_u32 s19, s25, s1 // (op 1733) 


s_cmp_eq_i32 s18, 0 // (op 1733) 
s_mov_b32 s1, scc // (op 1733) 

s_cmp_eq_i32 s1, 0 // (op 1733) 
s_cselect_b32 s18, s19, s0 // (op 1733) 



v_lshl_add_u32 v14, s18, 5, v12 // (op 1733) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 1733) 
v_mov_b32 v17, 0 // (op 1733) convert
v_mov_b32 v16, v14 // (op 1733) convert

s_lshr_b32 s18, s0, 1 // (op 1733) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 1733) 


v_mul_lo_u32 v15, v12, s27 // (op 1733) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 1733) most significant: high of low * low
v_mul_lo_u32 v14, v12, s26 // (op 1733) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 1733) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v14 // (op 1733) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 1733) most significant half



s_mul_i32 s1, s26, s31 // (op 1735) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1735) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1735) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1735) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1735) most significant: sum
s_add_u32 s1, s1, s19 // (op 1735) most significant: sum


s_mov_b32 s52, s28 // (op 1735) reg expression
s_mov_b32 s53, s29 // (op 1735) reg expression
s_mov_b32 s54, s0 // (op 1735) reg expression
s_mov_b32 s55, 131072 // (op 1735) call()

s_mul_hi_i32 s0, s16, s10 // (op 1803) 
v_lshrrev_b32 v14, 1, v1 // (op 1803) 
s_add_i32 s1, s0, s16 // (op 1803) 

v_and_b32 v15, -2, v1 // (op 1803) 
s_ashr_i32 s0, s1, 31 // (op 1803) 
v_sub_u32 v16, v1, v15 // (op 1803) 

s_cmp_eq_i32 s10, 0 // (op 1803) 
s_mov_b32 s18, scc // (op 1803) 
v_lshlrev_b32 v15, 6, v16 // (op 1803) 

s_cmp_eq_i32 s18, 0 // (op 1803) 
s_cselect_b32 s19, 0, -1 // (op 1803) 

s_and_b32 s18, s11, 31 // (op 1803) 
s_lshl_b32 s22, 1, s18 // (op 1803) 
s_add_i32 s23, s22, s19 // (op 1803) 


s_and_b32 s19, s0, s23 // (op 1803) 


s_add_i32 s0, s1, s19 // (op 1803) 


s_ashr_i32 s1, s0, s18 // (op 1803) 


s_lshl_b32 s0, s11, 24 // (op 1803) 
s_ashr_i32 s18, s0, 31 // (op 1803) 

s_xor_b32 s0, s1, s18 // (op 1803) 

s_sub_i32 s1, s0, s18 // (op 1803) 


s_mul_hi_i32 s0, s1, s12 // (op 1803) 
s_add_i32 s18, s0, s1 // (op 1803) 

s_ashr_i32 s0, s18, 31 // (op 1803) 
s_cmp_eq_i32 s12, 0 // (op 1803) 
s_mov_b32 s19, scc // (op 1803) 
s_cmp_eq_i32 s19, 0 // (op 1803) 
s_cselect_b32 s22, 0, -1 // (op 1803) 

s_and_b32 s19, s13, 31 // (op 1803) 
s_lshl_b32 s23, 1, s19 // (op 1803) 
s_add_i32 s24, s23, s22 // (op 1803) 


s_and_b32 s25, s0, s24 // (op 1803) 


s_add_i32 s0, s18, s25 // (op 1803) 


s_ashr_i32 s18, s0, s19 // (op 1803) 


s_lshl_b32 s0, s13, 24 // (op 1803) 
s_ashr_i32 s19, s0, 31 // (op 1803) 

s_xor_b32 s0, s18, s19 // (op 1803) 

s_sub_i32 s18, s0, s19 // (op 1803) 


s_add_u32 s50, s6, 128 // (op 1803) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1803) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1803) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1803) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 1803) 

s_mul_i32 s0, s18, s24 // (op 1803) 

s_sub_i32 s18, s1, s0 // (op 1803) 


s_mul_hi_i32 s19, s16, s9 // (op 1803) 
s_add_i32 s0, s19, s16 // (op 1803) 

s_ashr_i32 s1, s0, 31 // (op 1803) 
s_cmp_eq_i32 s9, 0 // (op 1803) 
s_mov_b32 s19, scc // (op 1803) 
s_cmp_eq_i32 s19, 0 // (op 1803) 
s_cselect_b32 s22, 0, -1 // (op 1803) 

s_and_b32 s19, s46, 31 // (op 1803) 
s_lshl_b32 s23, 1, s19 // (op 1803) 
s_add_i32 s49, s23, s22 // (op 1803) 


s_and_b32 s22, s1, s49 // (op 1803) 


s_add_i32 s1, s0, s22 // (op 1803) 


s_ashr_i32 s0, s1, s19 // (op 1803) 


s_lshl_b32 s1, s46, 24 // (op 1803) 
s_ashr_i32 s19, s1, 31 // (op 1803) 

s_xor_b32 s1, s0, s19 // (op 1803) 

s_sub_i32 s0, s1, s19 // (op 1803) 


s_mul_i32 s1, s15, s24 // (op 1803) 

s_mul_i32 s19, s8, s1 // (op 1803) 

s_mul_i32 s1, s0, s19 // (op 1803) 

s_sub_u32 s19, s16, s1 // (op 1803) 

s_mul_hi_i32 s1, s19, s14 // (op 1803) 
s_add_i32 s22, s1, s19 // (op 1803) 


s_ashr_i32 s1, s22, 31 // (op 1803) 
s_cmp_eq_i32 s14, 0 // (op 1803) 
s_mov_b32 s19, scc // (op 1803) 
s_cmp_eq_i32 s19, 0 // (op 1803) 
s_cselect_b32 s23, 0, -1 // (op 1803) 

s_and_b32 s19, s48, 31 // (op 1803) 
s_lshl_b32 s24, 1, s19 // (op 1803) 
s_add_i32 s25, s24, s23 // (op 1803) 


s_and_b32 s23, s1, s25 // (op 1803) 


s_add_i32 s1, s22, s23 // (op 1803) 


s_ashr_i32 s22, s1, s19 // (op 1803) 


s_lshl_b32 s1, s48, 24 // (op 1803) 
s_ashr_i32 s19, s1, 31 // (op 1803) 

s_xor_b32 s1, s22, s19 // (op 1803) 

s_sub_i32 s22, s1, s19 // (op 1803) 


s_cmp_eq_i32 s0, 0 // (op 1803) 
s_mov_b32 s1, scc // (op 1803) 

s_cmp_eq_i32 s1, 0 // (op 1803) 
s_cselect_b32 s0, s22, s18 // (op 1803) 



v_lshl_add_u32 v16, s0, 7, v14 // (op 1803) 


v_mul_lo_u32 v19, v16, s33 // (op 1803) most significant: low of low * high
v_mul_hi_u32 v14, v16, s32 // (op 1803) most significant: high of low * low
v_mul_lo_u32 v18, v16, s32 // (op 1803) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1803) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 7 // (op 1803) 
s_lshr_b32 s18, s0, 1 // (op 1803) 

v_lshl_add_u32 v14, s18, 8, v15 // (op 1803) 


v_mov_b32 v17, 0 // (op 1803) convert
v_mov_b32 v16, v14 // (op 1803) convert

v_add_co_u32 v14, s[0:1], v16, v18 // (op 1803) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 1803) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 1803) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 1803) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 1803) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1803) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1803) 

v_mov_b32 v16, v14 // (op 1803) convert

s_mul_i32 s1, s7, 6 // (op 1805) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 1805) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1805) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1805) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1805) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1805) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1805) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1805) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1805) most significant: sum
s_add_u32 s19, s19, s23 // (op 1805) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1805) 

s_mov_b32 s64, s34 // (op 1805) reg expression
s_mov_b32 s65, s35 // (op 1805) reg expression
s_mov_b32 s66, s0 // (op 1805) reg expression
s_mov_b32 s67, 131072 // (op 1805) call()

s_mul_hi_i32 s0, s16, s10 // (op 1867) 
v_lshrrev_b32 v14, 7, v1 // (op 1867) 
s_add_i32 s1, s0, s16 // (op 1867) 

v_and_b32 v15, 15, v1 // (op 1867) 
s_ashr_i32 s0, s1, 31 // (op 1867) 
v_lshl_add_u32 v17, v14, 4, v15 // (op 1867) 


s_cmp_eq_i32 s10, 0 // (op 1867) 
s_mov_b32 s18, scc // (op 1867) 
v_and_b32 v14, 63, v1 // (op 1867) 
s_cmp_eq_i32 s18, 0 // (op 1867) 
s_cselect_b32 s19, 0, -1 // (op 1867) 

v_lshrrev_b32 v15, 4, v14 // (op 1867) 

s_and_b32 s18, s11, 31 // (op 1867) 
s_lshl_b32 s22, 1, s18 // (op 1867) 
s_add_i32 s23, s22, s19 // (op 1867) 


s_and_b32 s19, s0, s23 // (op 1867) 


s_add_i32 s0, s1, s19 // (op 1867) 


s_ashr_i32 s1, s0, s18 // (op 1867) 


s_lshl_b32 s0, s11, 24 // (op 1867) 
s_ashr_i32 s18, s0, 31 // (op 1867) 

s_xor_b32 s0, s1, s18 // (op 1867) 

s_sub_i32 s1, s0, s18 // (op 1867) 


s_mul_hi_i32 s0, s1, s12 // (op 1867) 
s_add_i32 s18, s0, s1 // (op 1867) 

s_ashr_i32 s0, s18, 31 // (op 1867) 
s_cmp_eq_i32 s12, 0 // (op 1867) 
s_mov_b32 s19, scc // (op 1867) 
s_cmp_eq_i32 s19, 0 // (op 1867) 
s_cselect_b32 s22, 0, -1 // (op 1867) 

s_and_b32 s19, s13, 31 // (op 1867) 
s_lshl_b32 s23, 1, s19 // (op 1867) 
s_add_i32 s24, s23, s22 // (op 1867) 


s_and_b32 s25, s0, s24 // (op 1867) 


s_add_i32 s0, s18, s25 // (op 1867) 


s_ashr_i32 s18, s0, s19 // (op 1867) 


s_lshl_b32 s0, s13, 24 // (op 1867) 
s_ashr_i32 s19, s0, 31 // (op 1867) 

s_xor_b32 s0, s18, s19 // (op 1867) 

s_sub_i32 s18, s0, s19 // (op 1867) 


s_mul_i32 s0, s1, s15 // (op 1867) 

s_sub_u32 s1, s16, s0 // (op 1867) 

s_mul_i32 s0, s18, s15 // (op 1867) 
s_add_u32 s0, s0, s1 // (op 1867) 


s_mul_hi_i32 s1, s16, s9 // (op 1867) 
s_add_i32 s18, s1, s16 // (op 1867) 

s_ashr_i32 s1, s18, 31 // (op 1867) 
s_cmp_eq_i32 s9, 0 // (op 1867) 
s_mov_b32 s19, scc // (op 1867) 
s_cmp_eq_i32 s19, 0 // (op 1867) 
s_cselect_b32 s22, 0, -1 // (op 1867) 

s_and_b32 s19, s46, 31 // (op 1867) 
s_lshl_b32 s23, 1, s19 // (op 1867) 
s_add_i32 s24, s23, s22 // (op 1867) 


s_and_b32 s25, s1, s24 // (op 1867) 


s_add_i32 s1, s18, s25 // (op 1867) 


s_ashr_i32 s18, s1, s19 // (op 1867) 


s_lshl_b32 s1, s46, 24 // (op 1867) 
s_ashr_i32 s19, s1, 31 // (op 1867) 

s_xor_b32 s1, s18, s19 // (op 1867) 

s_sub_i32 s18, s1, s19 // (op 1867) 


s_add_u32 s50, s6, 128 // (op 1867) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1867) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1867) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1867) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 1867) 

s_mul_i32 s1, s15, s24 // (op 1867) 

s_mul_i32 s19, s8, s1 // (op 1867) 

s_mul_i32 s1, s18, s19 // (op 1867) 

s_sub_u32 s19, s16, s1 // (op 1867) 

s_mov_b32 s1, s19 // (op 1867) 
s_mul_hi_i32 s22, s1, s14 // (op 1867) 
s_add_i32 s23, s22, s1 // (op 1867) 


s_ashr_i32 s1, s23, 31 // (op 1867) 
s_cmp_eq_i32 s14, 0 // (op 1867) 
s_mov_b32 s22, scc // (op 1867) 
s_cmp_eq_i32 s22, 0 // (op 1867) 
s_cselect_b32 s24, 0, -1 // (op 1867) 

s_and_b32 s22, s48, 31 // (op 1867) 
s_lshl_b32 s25, 1, s22 // (op 1867) 
s_add_i32 s49, s25, s24 // (op 1867) 


s_and_b32 s24, s1, s49 // (op 1867) 


s_add_i32 s1, s23, s24 // (op 1867) 


s_ashr_i32 s23, s1, s22 // (op 1867) 


s_lshl_b32 s1, s48, 24 // (op 1867) 
s_ashr_i32 s22, s1, 31 // (op 1867) 

s_xor_b32 s1, s23, s22 // (op 1867) 

s_sub_i32 s23, s1, s22 // (op 1867) 


s_mul_i32 s1, s47, s15 // (op 1867) 
s_add_u32 s24, s20, 32 // (op 1867) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1867) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1867) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1867) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 1867) 

s_sub_i32 s22, s24, s1 // (op 1867) 

s_mul_i32 s24, s23, s22 // (op 1867) 


s_sub_u32 s25, s19, s24 // (op 1867) 


s_add_u32 s19, s25, s1 // (op 1867) 


s_cmp_eq_i32 s18, 0 // (op 1867) 
s_mov_b32 s1, scc // (op 1867) 

s_cmp_eq_i32 s1, 0 // (op 1867) 
s_cselect_b32 s18, s19, s0 // (op 1867) 



v_lshl_add_u32 v14, s18, 5, v17 // (op 1867) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 1867) 
s_lshr_b32 s18, s0, 1 // (op 1867) 

v_lshl_add_u32 v17, s18, 3, v15 // (op 1867) 


v_mul_lo_u32 v15, v17, s36 // (op 1867) 

v_add_u32 v17, v14, v15 // (op 1867) 


s_mul_i32 s1, s36, s41 // (op 1869) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1869) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1869) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1869) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1869) most significant: sum
s_add_u32 s1, s1, s19 // (op 1869) most significant: sum


s_mov_b32 s68, s38 // (op 1869) reg expression
s_mov_b32 s69, s39 // (op 1869) reg expression
s_mov_b32 s70, s0 // (op 1869) reg expression
s_mov_b32 s71, 131072 // (op 1869) call()

s_mul_hi_i32 s0, s16, s10 // (op 1931) 
v_lshrrev_b32 v14, 2, v1 // (op 1931) 
s_add_i32 s1, s0, s16 // (op 1931) 

v_lshlrev_b32 v15, 1, v14 // (op 1931) 

s_ashr_i32 s0, s1, 31 // (op 1931) 
v_and_b32 v14, -4, v1 // (op 1931) 
s_cmp_eq_i32 s10, 0 // (op 1931) 
s_mov_b32 s18, scc // (op 1931) 
v_sub_u32 v18, v1, v14 // (op 1931) 

s_cmp_eq_i32 s18, 0 // (op 1931) 
s_cselect_b32 s19, 0, -1 // (op 1931) 

s_and_b32 s18, s11, 31 // (op 1931) 
s_lshl_b32 s22, 1, s18 // (op 1931) 
s_add_i32 s23, s22, s19 // (op 1931) 


s_and_b32 s19, s0, s23 // (op 1931) 


s_add_i32 s0, s1, s19 // (op 1931) 


s_ashr_i32 s1, s0, s18 // (op 1931) 


s_lshl_b32 s0, s11, 24 // (op 1931) 
s_ashr_i32 s18, s0, 31 // (op 1931) 

s_xor_b32 s0, s1, s18 // (op 1931) 

s_sub_i32 s1, s0, s18 // (op 1931) 


s_mul_hi_i32 s0, s1, s12 // (op 1931) 
s_add_i32 s18, s0, s1 // (op 1931) 

s_ashr_i32 s0, s18, 31 // (op 1931) 
s_cmp_eq_i32 s12, 0 // (op 1931) 
s_mov_b32 s19, scc // (op 1931) 
s_cmp_eq_i32 s19, 0 // (op 1931) 
s_cselect_b32 s22, 0, -1 // (op 1931) 

s_and_b32 s19, s13, 31 // (op 1931) 
s_lshl_b32 s23, 1, s19 // (op 1931) 
s_add_i32 s24, s23, s22 // (op 1931) 


s_and_b32 s25, s0, s24 // (op 1931) 


s_add_i32 s0, s18, s25 // (op 1931) 


s_ashr_i32 s18, s0, s19 // (op 1931) 


s_lshl_b32 s0, s13, 24 // (op 1931) 
s_ashr_i32 s19, s0, 31 // (op 1931) 

s_xor_b32 s0, s18, s19 // (op 1931) 

s_sub_i32 s18, s0, s19 // (op 1931) 


s_add_u32 s50, s6, 128 // (op 1931) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1931) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1931) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1931) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 1931) 

s_mul_i32 s0, s18, s24 // (op 1931) 

s_sub_i32 s18, s1, s0 // (op 1931) 


s_mul_hi_i32 s19, s16, s9 // (op 1931) 
s_add_i32 s0, s19, s16 // (op 1931) 

s_ashr_i32 s1, s0, 31 // (op 1931) 
s_cmp_eq_i32 s9, 0 // (op 1931) 
s_mov_b32 s19, scc // (op 1931) 
s_cmp_eq_i32 s19, 0 // (op 1931) 
s_cselect_b32 s22, 0, -1 // (op 1931) 

s_and_b32 s19, s46, 31 // (op 1931) 
s_lshl_b32 s23, 1, s19 // (op 1931) 
s_add_i32 s49, s23, s22 // (op 1931) 


s_and_b32 s22, s1, s49 // (op 1931) 


s_add_i32 s1, s0, s22 // (op 1931) 


s_ashr_i32 s0, s1, s19 // (op 1931) 


s_lshl_b32 s1, s46, 24 // (op 1931) 
s_ashr_i32 s19, s1, 31 // (op 1931) 

s_xor_b32 s1, s0, s19 // (op 1931) 

s_sub_i32 s0, s1, s19 // (op 1931) 


s_mul_i32 s1, s15, s24 // (op 1931) 

s_mul_i32 s19, s8, s1 // (op 1931) 

s_mul_i32 s1, s0, s19 // (op 1931) 

s_sub_u32 s19, s16, s1 // (op 1931) 

s_mul_hi_i32 s1, s19, s14 // (op 1931) 
s_add_i32 s22, s1, s19 // (op 1931) 


s_ashr_i32 s1, s22, 31 // (op 1931) 
s_cmp_eq_i32 s14, 0 // (op 1931) 
s_mov_b32 s19, scc // (op 1931) 
s_cmp_eq_i32 s19, 0 // (op 1931) 
s_cselect_b32 s23, 0, -1 // (op 1931) 

s_and_b32 s19, s48, 31 // (op 1931) 
s_lshl_b32 s24, 1, s19 // (op 1931) 
s_add_i32 s25, s24, s23 // (op 1931) 


s_and_b32 s23, s1, s25 // (op 1931) 


s_add_i32 s1, s22, s23 // (op 1931) 


s_ashr_i32 s22, s1, s19 // (op 1931) 


s_lshl_b32 s1, s48, 24 // (op 1931) 
s_ashr_i32 s19, s1, 31 // (op 1931) 

s_xor_b32 s1, s22, s19 // (op 1931) 

s_sub_i32 s22, s1, s19 // (op 1931) 


s_cmp_eq_i32 s0, 0 // (op 1931) 
s_mov_b32 s1, scc // (op 1931) 

s_cmp_eq_i32 s1, 0 // (op 1931) 
s_cselect_b32 s0, s22, s18 // (op 1931) 



v_lshl_add_u32 v14, s0, 7, v15 // (op 1931) 


v_mul_lo_u32 v15, v14, s42 // (op 1931) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 1931) 
s_lshr_b32 s18, s0, 1 // (op 1931) 

v_lshl_add_u32 v14, s18, 3, v18 // (op 1931) 


v_add_u32 v18, v14, v15 // (op 1931) 


s_mul_i32 s1, s42, s7 // (op 1933) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1933) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1933) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1933) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1933) most significant: sum
s_add_u32 s1, s1, s19 // (op 1933) most significant: sum


s_mov_b32 s72, s44 // (op 1933) reg expression
s_mov_b32 s73, s45 // (op 1933) reg expression
s_mov_b32 s74, s0 // (op 1933) reg expression
s_mov_b32 s75, 131072 // (op 1933) call()

v_and_b32 v14, -2, v1 // (op 2024) 
v_sub_u32 v15, v1, v14 // (op 2024) 

v_lshlrev_b32 v14, 4, v15 // (op 2024) 

v_lshrrev_b32 v15, 1, v1 // (op 2024) 
v_lshl_add_u32 v19, v15, 5, v14 // (op 2024) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 2024) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -2, v1 // (op 2076) 
v_sub_u32 v15, v1, v14 // (op 2076) 

v_lshlrev_b32 v14, 6, v15 // (op 2076) 

v_lshrrev_b32 v15, 1, v1 // (op 2076) 
v_lshl_add_u32 v19, v15, 7, v14 // (op 2076) 


v_mul_lo_u32 v14, v19, 6 // (op 2076) 

v_lshrrev_b32 v15, 3, v14 // (op 2076) 

v_and_b32 v14, -4, v1 // (op 2134) 
v_sub_u32 v19, v1, v14 // (op 2134) 

v_lshrrev_b32 v14, 2, v1 // (op 2134) 
v_lshl_add_u32 v20, v14, 3, v19 // (op 2134) 


s_ashr_i64 s[18:19], s[30:31], 7 // (op 629) 
s_and_b32 s1, s18, 4294967294 // (op 629) 

s_ashr_i64 s[18:19], s[30:31], 7 // (op 628) 
s_mov_b32 s22, s18 // (op 628) convert

s_cmp_lt_u32 s1, s22 // (op 628) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_628 // (op 628) Condition: Top (jump to RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_628 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ForLoopTop_KLoopTail_628:
s_barrier  // (op 2335) 
s_add_u32 s1, s1, 1 // (op 630) 


v_mov_b64 v[22:23], v[12:13] // (op 1704) 
v_mov_b64 v[24:25], v[22:23] // (op 1704) reg expression

s_mov_b32 m0, s0 // (op 1704) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 1704) Load value direct to lds


s_lshl_b64 s[18:19], s[26:27], 7 // (op 1723) 
v_mov_b32 v14, s18 // (op 1723) 
v_mov_b32 v19, s19 // (op 1723) 
v_add_co_u32 v12, s[50:51], v12, v14 // (op 1723) least significant half
v_addc_co_u32 v13, s[50:51], v13, v19, s[50:51] // (op 1723) most significant half






v_mov_b32 v14, v16 // (op 183) 

buffer_load_dwordx4 v[22:25], v14, s[64:67], 0 offen // (op 183) Load value
buffer_load_dwordx2 v[26:27], v14, s[64:67], 0 offen offset:16 // (op 183) Load value
buffer_load_dwordx4 v[28:31], v14, s[64:67], 0 offen offset:24 // (op 183) Load value
buffer_load_dwordx2 v[32:33], v14, s[64:67], 0 offen offset:40 // (op 183) Load value

v_add_u32 v16, 96, v16 // (op 1795) 

s_barrier  // (op 2348) 
v_mov_b64 v[34:35], s[42:43] // (op 184) convert




v_mov_b32 v14, v18 // (op 184) 
v_mov_b32 v19, v14 // (op 184) 
buffer_load_ubyte v36, v19, s[72:75], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v19, v19, v34 // (op 184) 
s_waitcnt vmcnt(4)
ds_write_b128 v15, v[22:25] offset:8192 // (op 181) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v15, v[26:27] offset:8208 // (op 181) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v15, v[28:31] offset:8216 // (op 181) Store local data 
buffer_load_ubyte v37, v19, s[72:75], 0 offen offset: 0 // (op 184) Load value



v_add_u32 v18, 4, v18 // (op 1923) 
s_waitcnt vmcnt(2)
ds_write_b64 v15, v[32:33] offset:8232 // (op 181) Store local data 

s_barrier  // (op 2338) 



s_barrier  // (op 2362) 
ds_read_b64_tr_b8 v[28:29], v0 // (op 173) Transpose load local data 



ds_read_b64_tr_b8 v[30:31], v0 offset:256 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:2048 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:2304 // (op 173) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v36 offset:20480 // (op 182) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2352) 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v37 offset:20484 // (op 182) Store local data 




ds_read_b128 v[22:25], v7 offset:8192 // (op 175) Load local data 
ds_read_b64 v[26:27], v7 offset:8208 // (op 175) Load local data 

v_mov_b64 v[36:37], s[36:37] // (op 177) convert

v_mov_b32 v14, v17 // (op 177) 
v_mov_b32 v19, v14 // (op 177) reg expression
v_mov_b32 v21, v19 // (op 177) 
buffer_load_ubyte v38, v21, s[68:71], 0 offen offset: 0 // (op 177) Load value




s_lshl_b32 s18, s36, 2 // (op 1859) 
v_add_u32 v17, v17, s18 // (op 1859) 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 2366) 



ds_read_u8 v14, v4 offset:20480 // (op 179) Load local data 



ds_read_b128 v[40:43], v7 offset:11264 // (op 324) Load local data 
ds_read_b64 v[44:45], v7 offset:11280 // (op 324) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[22:27], a[0:3], v38, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 180) 





ds_read_u8 v14, v4 offset:20608 // (op 327) Load local data 



ds_read_b128 v[22:25], v7 offset:14336 // (op 480) Load local data 
ds_read_b64 v[26:27], v7 offset:14352 // (op 480) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:35], v[40:45], a[4:7], v38, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 328) 





ds_read_u8 v14, v4 offset:20736 // (op 483) Load local data 



ds_read_b128 v[40:43], v7 offset:17408 // (op 636) Load local data 
ds_read_b64 v[44:45], v7 offset:17424 // (op 636) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:35], v[22:27], a[8:11], v38, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 484) 





ds_read_u8 v14, v4 offset:20864 // (op 639) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[28:35], v[40:45], a[12:15], v38, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 640) 




s_ashr_i64 s[18:19], s[30:31], 7 // (op 628) 
s_mov_b32 s22, s18 // (op 628) convert

s_cmp_lt_u32 s1, s22 // (op 628) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ForLoopTop_KLoopTail_628 // (op 628) Condition: Bottom (jump to RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ForLoopTop_KLoopTail_628 if true)
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_628:






















s_barrier  // (op 2450) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ConditionalFalse_0_beta_125
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



v_accvgpr_read v10, a4 // (op 864) 
v_accvgpr_read v11, a5 // (op 864) 
v_accvgpr_read v12, a6 // (op 864) 
v_accvgpr_read v13, a7 // (op 864) 
v_mul_f32 v14, v8, v10 // (op 864) 
v_mul_f32 v15, v8, v11 // (op 864) 
v_mul_f32 v16, v8, v12 // (op 864) 
v_mul_f32 v17, v8, v13 // (op 864) 

v_cvt_f16_f32 v10, v14 // (op 866) 
v_cvt_f16_f32 v11, v15 // (op 866) 
v_cvt_f16_f32 v12, v16 // (op 866) 
v_cvt_f16_f32 v13, v17 // (op 866) 


v_pack_b32_f16 v14, v10, v11 // (op 867) 
v_pack_b32_f16 v15, v12, v13 // (op 867) 

s_lshl_b32 s0, s4, 6 // (op 867) 
v_add_u32 v0, v9, s0 // (op 867) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 867) Store value



v_accvgpr_read v10, a8 // (op 878) 
v_accvgpr_read v11, a9 // (op 878) 
v_accvgpr_read v12, a10 // (op 878) 
v_accvgpr_read v13, a11 // (op 878) 
v_mul_f32 v14, v8, v10 // (op 878) 
v_mul_f32 v15, v8, v11 // (op 878) 
v_mul_f32 v16, v8, v12 // (op 878) 
v_mul_f32 v17, v8, v13 // (op 878) 

v_cvt_f16_f32 v10, v14 // (op 880) 
v_cvt_f16_f32 v11, v15 // (op 880) 
v_cvt_f16_f32 v12, v16 // (op 880) 
v_cvt_f16_f32 v13, v17 // (op 880) 


v_pack_b32_f16 v14, v10, v11 // (op 881) 
v_pack_b32_f16 v15, v12, v13 // (op 881) 

s_lshl_b32 s0, s4, 7 // (op 881) 
v_add_u32 v0, v9, s0 // (op 881) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 881) Store value



v_accvgpr_read v10, a12 // (op 892) 
v_accvgpr_read v11, a13 // (op 892) 
v_accvgpr_read v12, a14 // (op 892) 
v_accvgpr_read v13, a15 // (op 892) 
v_mul_f32 v14, v8, v10 // (op 892) 
v_mul_f32 v15, v8, v11 // (op 892) 
v_mul_f32 v16, v8, v12 // (op 892) 
v_mul_f32 v17, v8, v13 // (op 892) 

v_cvt_f16_f32 v10, v14 // (op 894) 
v_cvt_f16_f32 v11, v15 // (op 894) 
v_cvt_f16_f32 v12, v16 // (op 894) 
v_cvt_f16_f32 v13, v17 // (op 894) 


v_pack_b32_f16 v14, v10, v11 // (op 895) 
v_pack_b32_f16 v15, v12, v13 // (op 895) 

s_lshl_b32 s0, s4, 6 // (op 895) 
s_mul_i32 s1, 3, s0 // (op 895) 

v_add_u32 v0, v9, s1 // (op 895) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 895) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ConditionalFalse_0_beta_125:


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





s_lshl_b32 s0, s2, 6 // (op 929) 
v_add_u32 v0, v6, s0 // (op 929) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 929) Load value

v_accvgpr_read v12, a4 // (op 925) 
v_accvgpr_read v13, a5 // (op 925) 
v_accvgpr_read v14, a6 // (op 925) 
v_accvgpr_read v15, a7 // (op 925) 
v_mul_f32 v16, v8, v12 // (op 925) 
v_mul_f32 v17, v8, v13 // (op 925) 
v_mul_f32 v18, v8, v14 // (op 925) 
v_mul_f32 v19, v8, v15 // (op 925) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 930) 
v_lshrrev_b32 v13, 16, v10 // (op 930) 
v_cvt_f32_f16 v13, v13 // (op 930) 
v_mul_f32 v20, v5, v12 // (op 930) 
v_mul_f32 v21, v5, v13 // (op 930) 
v_cvt_f32_f16 v12, v11 // (op 930) 
v_lshrrev_b32 v13, 16, v11 // (op 930) 
v_cvt_f32_f16 v13, v13 // (op 930) 
v_mul_f32 v22, v5, v12 // (op 930) 
v_mul_f32 v23, v5, v13 // (op 930) 


v_add_f32 v10, v20, v16 // (op 926) 
v_add_f32 v11, v21, v17 // (op 926) 
v_add_f32 v12, v22, v18 // (op 926) 
v_add_f32 v13, v23, v19 // (op 926) 


v_cvt_f16_f32 v14, v10 // (op 927) 
v_cvt_f16_f32 v15, v11 // (op 927) 
v_cvt_f16_f32 v16, v12 // (op 927) 
v_cvt_f16_f32 v17, v13 // (op 927) 


v_pack_b32_f16 v10, v14, v15 // (op 928) 
v_pack_b32_f16 v11, v16, v17 // (op 928) 

s_lshl_b32 s0, s4, 6 // (op 928) 
v_add_u32 v0, v9, s0 // (op 928) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 928) Store value





s_lshl_b32 s0, s2, 7 // (op 943) 
v_add_u32 v0, v6, s0 // (op 943) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 943) Load value

v_accvgpr_read v12, a8 // (op 939) 
v_accvgpr_read v13, a9 // (op 939) 
v_accvgpr_read v14, a10 // (op 939) 
v_accvgpr_read v15, a11 // (op 939) 
v_mul_f32 v16, v8, v12 // (op 939) 
v_mul_f32 v17, v8, v13 // (op 939) 
v_mul_f32 v18, v8, v14 // (op 939) 
v_mul_f32 v19, v8, v15 // (op 939) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 944) 
v_lshrrev_b32 v13, 16, v10 // (op 944) 
v_cvt_f32_f16 v13, v13 // (op 944) 
v_mul_f32 v20, v5, v12 // (op 944) 
v_mul_f32 v21, v5, v13 // (op 944) 
v_cvt_f32_f16 v12, v11 // (op 944) 
v_lshrrev_b32 v13, 16, v11 // (op 944) 
v_cvt_f32_f16 v13, v13 // (op 944) 
v_mul_f32 v22, v5, v12 // (op 944) 
v_mul_f32 v23, v5, v13 // (op 944) 


v_add_f32 v10, v20, v16 // (op 940) 
v_add_f32 v11, v21, v17 // (op 940) 
v_add_f32 v12, v22, v18 // (op 940) 
v_add_f32 v13, v23, v19 // (op 940) 


v_cvt_f16_f32 v14, v10 // (op 941) 
v_cvt_f16_f32 v15, v11 // (op 941) 
v_cvt_f16_f32 v16, v12 // (op 941) 
v_cvt_f16_f32 v17, v13 // (op 941) 


v_pack_b32_f16 v10, v14, v15 // (op 942) 
v_pack_b32_f16 v11, v16, v17 // (op 942) 

s_lshl_b32 s0, s4, 7 // (op 942) 
v_add_u32 v0, v9, s0 // (op 942) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 942) Store value





s_lshl_b32 s0, s2, 6 // (op 957) 
s_mul_i32 s1, 3, s0 // (op 957) 

v_add_u32 v0, v6, s1 // (op 957) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 957) Load value

v_accvgpr_read v12, a12 // (op 953) 
v_accvgpr_read v13, a13 // (op 953) 
v_accvgpr_read v14, a14 // (op 953) 
v_accvgpr_read v15, a15 // (op 953) 
v_mul_f32 v16, v8, v12 // (op 953) 
v_mul_f32 v17, v8, v13 // (op 953) 
v_mul_f32 v18, v8, v14 // (op 953) 
v_mul_f32 v19, v8, v15 // (op 953) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 958) 
v_lshrrev_b32 v13, 16, v10 // (op 958) 
v_cvt_f32_f16 v13, v13 // (op 958) 
v_mul_f32 v20, v5, v12 // (op 958) 
v_mul_f32 v21, v5, v13 // (op 958) 
v_cvt_f32_f16 v12, v11 // (op 958) 
v_lshrrev_b32 v13, 16, v11 // (op 958) 
v_cvt_f32_f16 v13, v13 // (op 958) 
v_mul_f32 v22, v5, v12 // (op 958) 
v_mul_f32 v23, v5, v13 // (op 958) 


v_add_f32 v10, v20, v16 // (op 954) 
v_add_f32 v11, v21, v17 // (op 954) 
v_add_f32 v12, v22, v18 // (op 954) 
v_add_f32 v13, v23, v19 // (op 954) 


v_cvt_f16_f32 v14, v10 // (op 955) 
v_cvt_f16_f32 v15, v11 // (op 955) 
v_cvt_f16_f32 v16, v12 // (op 955) 
v_cvt_f16_f32 v17, v13 // (op 955) 


v_pack_b32_f16 v10, v14, v15 // (op 956) 
v_pack_b32_f16 v11, v16, v17 // (op 956) 

s_lshl_b32 s0, s4, 6 // (op 956) 
s_mul_i32 s1, 3, s0 // (op 956) 

v_add_u32 v0, v9, s1 // (op 956) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 956) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__ConditionalBottom_0_beta_125:












s_endpgm  // End of RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM_
.LRR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__end:
.size RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM_, .LRR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM__end-RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 33792 // lds bytes
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
  MXScaleFormat: NoSwizzle
  ProblemType:
    OperationType: GEMM
    DataType: B8
    DataTypeA: B8
    DataTypeB: B6
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
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
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int64, semantic: SizeFree0 },
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
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [32, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 33792
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 78
    .agpr_count: 16
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
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 24
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_0_size_0_15
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 72
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 80
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 112
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 128
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 136
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 152
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 160
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 164
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
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

