// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_,@function
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s11 // (op 1511) 
s_mul_hi_i32 s1, s16, s11 // (op 1751) 
v_lshrrev_b32 v0, 7, v1 // (op 1567) 
v_and_b32 v3, 15, v1 // (op 1567) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1567) 


v_and_b32 v0, 48, v1 // (op 1567) 
v_lshl_add_u32 v3, v4, 8, v0 // (op 1567) 


v_lshrrev_b32 v0, 7, v1 // (op 1677) 
v_and_b32 v4, 15, v1 // (op 1677) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1677) 


v_and_b32 v0, 63, v1 // (op 1677) 
v_lshrrev_b32 v4, 4, v0 // (op 1677) 

v_lshl_add_u32 v0, v5, 3, v4 // (op 1677) 


v_lshrrev_b32 v4, 7, v1 // (op 1792) 
v_and_b32 v5, 15, v1 // (op 1792) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 1792) 


v_and_b32 v4, 48, v1 // (op 1792) 
v_lshl_add_u32 v5, v6, 8, v4 // (op 1792) 


v_lshrrev_b32 v4, 7, v1 // (op 1860) 
v_and_b32 v6, 15, v1 // (op 1860) 
v_lshl_add_u32 v7, v4, 4, v6 // (op 1860) 


v_and_b32 v4, 63, v1 // (op 1860) 
v_lshrrev_b32 v6, 4, v4 // (op 1860) 

v_lshl_add_u32 v4, v7, 3, v6 // (op 1860) 


v_lshrrev_b32 v6, 7, v1 // (op 1511) 
s_add_i32 s18, s0, s16 // (op 1511) 

v_and_b32 v7, 63, v1 // (op 1511) 
s_ashr_i32 s0, s18, 31 // (op 1511) 
v_lshrrev_b32 v8, 4, v7 // (op 1511) 
s_cmp_eq_i32 s11, 0 // (op 1511) 
s_mov_b32 s19, scc // (op 1511) 
v_lshlrev_b32 v9, 2, v8 // (op 1511) 

s_cmp_eq_i32 s19, 0 // (op 1511) 
s_cselect_b32 s51, 0, -1 // (op 1511) 

v_lshl_add_u32 v8, v6, 4, v9 // (op 1511) 


s_and_b32 s19, s12, 31 // (op 1511) 
v_lshrrev_b32 v6, 2, v7 // (op 1511) 

v_and_b32 v7, 3, v6 // (op 1511) 

v_and_b32 v6, 48, v1 // (op 1624) 
v_and_b32 v9, 63, v1 // (op 1624) 
v_sub_u32 v10, v9, v6 // (op 1624) 

v_lshrrev_b32 v9, 1, v10 // (op 1624) 
v_add_lshl_u32 v11, v6, v9, 6 // (op 1624) 


v_and_b32 v6, 1, v10 // (op 1624) 

v_lshlrev_b32 v9, 3, v6 // (op 1624) 

v_lshrrev_b32 v6, 6, v1 // (op 1624) 
v_and_b32 v10, 1, v6 // (op 1624) 

v_lshl_add_u32 v6, v10, 4, v9 // (op 1624) 


v_add_u32 v9, v11, v6 // (op 1624) 


v_lshrrev_b32 v6, 6, v1 // (op 1735) 
v_and_b32 v10, 1, v6 // (op 1735) 

v_and_b32 v6, 15, v1 // (op 1735) 
v_lshl_add_u32 v11, v10, 4, v6 // (op 1735) 


v_and_b32 v6, 63, v1 // (op 1735) 
v_lshrrev_b32 v10, 4, v6 // (op 1735) 

v_lshl_add_u32 v6, v10, 6, v11 // (op 1735) 


v_lshrrev_b32 v10, 7, v1 // (op 1751) 
v_and_b32 v11, 63, v1 // (op 1751) 
v_and_b32 v12, 48, v1 // (op 1836) 
v_and_b32 v13, 63, v1 // (op 1836) 
v_sub_u32 v14, v13, v12 // (op 1836) 

v_lshrrev_b32 v13, 1, v14 // (op 1836) 
v_add_lshl_u32 v15, v12, v13, 6 // (op 1836) 


v_and_b32 v12, 1, v14 // (op 1836) 

v_lshlrev_b32 v13, 3, v12 // (op 1836) 

v_lshrrev_b32 v12, 6, v1 // (op 1836) 
v_and_b32 v14, 1, v12 // (op 1836) 

v_lshl_add_u32 v12, v14, 4, v13 // (op 1836) 


v_add_u32 v13, v15, v12 // (op 1836) 


v_lshrrev_b32 v12, 6, v1 // (op 1889) 
v_and_b32 v14, 1, v12 // (op 1889) 

v_and_b32 v12, 15, v1 // (op 1889) 
v_lshl_add_u32 v15, v14, 4, v12 // (op 1889) 


v_and_b32 v12, 63, v1 // (op 1889) 
v_lshrrev_b32 v14, 4, v12 // (op 1889) 

v_lshl_add_u32 v12, v14, 6, v15 // (op 1889) 


s_lshl_b32 s52, 1, s19 // (op 1511) 
v_and_b32 v14, 3, v1 // (op 1511) 
s_add_i32 s53, s52, s51 // (op 1511) 


s_and_b32 s51, s0, s53 // (op 1511) 


s_add_i32 s0, s1, s16 // (op 1751) 

s_ashr_i32 s1, s0, 31 // (op 1751) 
v_lshl_add_u32 v15, v7, 2, v14 // (op 1511) 


s_add_i32 s52, s18, s51 // (op 1511) 


v_lshrrev_b32 v7, 6, v1 // (op 1511) 
s_ashr_i32 s18, s52, s19 // (op 1511) 


v_and_b32 v14, 1, v7 // (op 1511) 

s_lshl_b32 s19, s12, 24 // (op 1511) 
v_lshl_add_u32 v7, v14, 4, v15 // (op 1511) 


s_ashr_i32 s51, s19, 31 // (op 1511) 

s_xor_b32 s19, s18, s51 // (op 1511) 

s_sub_i32 s18, s19, s51 // (op 1511) 


s_mul_hi_i32 s19, s18, s13 // (op 1511) 
s_add_i32 s51, s19, s18 // (op 1511) 

s_ashr_i32 s19, s51, 31 // (op 1511) 
s_cmp_eq_i32 s13, 0 // (op 1511) 
s_mov_b32 s52, scc // (op 1511) 
s_cmp_eq_i32 s52, 0 // (op 1511) 
s_cselect_b32 s53, 0, -1 // (op 1511) 

s_and_b32 s52, s14, 31 // (op 1511) 
v_lshrrev_b32 v14, 4, v11 // (op 1751) 
s_cmp_eq_i32 s11, 0 // (op 1751) 
s_mov_b32 s54, scc // (op 1751) 
v_lshlrev_b32 v15, 2, v14 // (op 1751) 

s_cmp_eq_i32 s54, 0 // (op 1751) 
s_cselect_b32 s55, 0, -1 // (op 1751) 

s_lshl_b32 s54, 1, s52 // (op 1511) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 1751) 


v_lshrrev_b32 v10, 2, v11 // (op 1751) 

s_add_i32 s56, s54, s53 // (op 1511) 


s_and_b32 s53, s19, s56 // (op 1511) 


s_add_i32 s19, s51, s53 // (op 1511) 


s_ashr_i32 s51, s19, s52 // (op 1511) 


s_lshl_b32 s19, s14, 24 // (op 1511) 
s_ashr_i32 s52, s19, 31 // (op 1511) 

s_xor_b32 s19, s51, s52 // (op 1511) 

s_sub_i32 s51, s19, s52 // (op 1511) 


s_and_b32 s19, s12, 31 // (op 1751) 
s_lshl_b32 s52, 1, s19 // (op 1751) 
v_and_b32 v11, 3, v10 // (op 1751) 

s_add_i32 s53, s52, s55 // (op 1751) 


v_and_b32 v10, 3, v1 // (op 1751) 
s_and_b32 s52, s1, s53 // (op 1751) 


v_lshl_add_u32 v15, v11, 2, v10 // (op 1751) 


s_add_i32 s1, s0, s52 // (op 1751) 


v_lshrrev_b32 v10, 6, v1 // (op 1751) 
s_ashr_i32 s0, s1, s19 // (op 1751) 


v_and_b32 v11, 1, v10 // (op 1751) 

s_lshl_b32 s1, s12, 24 // (op 1751) 
v_lshl_add_u32 v10, v11, 4, v15 // (op 1751) 


s_ashr_i32 s19, s1, 31 // (op 1751) 

s_xor_b32 s1, s0, s19 // (op 1751) 

s_sub_i32 s0, s1, s19 // (op 1751) 


s_mul_hi_i32 s1, s0, s13 // (op 1751) 
s_add_i32 s19, s1, s0 // (op 1751) 

s_ashr_i32 s1, s19, 31 // (op 1751) 
s_cmp_eq_i32 s13, 0 // (op 1751) 
s_mov_b32 s52, scc // (op 1751) 
s_cmp_eq_i32 s52, 0 // (op 1751) 
s_cselect_b32 s53, 0, -1 // (op 1751) 

s_and_b32 s52, s14, 31 // (op 1751) 
s_lshl_b32 s54, 1, s52 // (op 1751) 
s_add_i32 s55, s54, s53 // (op 1751) 


s_and_b32 s53, s1, s55 // (op 1751) 


s_add_i32 s1, s19, s53 // (op 1751) 


s_ashr_i32 s19, s1, s52 // (op 1751) 


s_lshl_b32 s1, s14, 24 // (op 1751) 
s_ashr_i32 s52, s1, 31 // (op 1751) 

s_xor_b32 s1, s19, s52 // (op 1751) 

s_sub_i32 s19, s1, s52 // (op 1751) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 1511) 
s_sub_u32 s52, s16, s1 // (op 1511) 

s_mul_i32 s1, s51, s44 // (op 1511) 
s_add_u32 s1, s1, s52 // (op 1511) 

s_mul_hi_i32 s52, s16, s10 // (op 1511) 
s_add_i32 s53, s52, s16 // (op 1511) 

s_ashr_i32 s52, s53, 31 // (op 1511) 
s_cmp_eq_i32 s10, 0 // (op 1511) 
s_mov_b32 s54, scc // (op 1511) 
s_cmp_eq_i32 s54, 0 // (op 1511) 
s_cselect_b32 s55, 0, -1 // (op 1511) 

s_and_b32 s54, s47, 31 // (op 1511) 
s_lshl_b32 s56, 1, s54 // (op 1511) 
s_add_i32 s57, s56, s55 // (op 1511) 


s_and_b32 s55, s52, s57 // (op 1511) 


s_add_i32 s52, s53, s55 // (op 1511) 


s_ashr_i32 s53, s52, s54 // (op 1511) 


s_lshl_b32 s52, s47, 24 // (op 1511) 
s_ashr_i32 s54, s52, 31 // (op 1511) 

s_xor_b32 s52, s53, s54 // (op 1511) 

s_sub_i32 s53, s52, s54 // (op 1511) 


s_mul_i32 s52, s0, s44 // (op 1751) 
s_sub_u32 s54, s16, s52 // (op 1751) 

s_mul_i32 s52, s19, s44 // (op 1751) 
s_add_u32 s52, s52, s54 // (op 1751) 

s_mul_hi_i32 s54, s16, s10 // (op 1751) 
s_add_i32 s55, s54, s16 // (op 1751) 

s_ashr_i32 s54, s55, 31 // (op 1751) 
s_cmp_eq_i32 s10, 0 // (op 1751) 
s_mov_b32 s56, scc // (op 1751) 
s_cmp_eq_i32 s56, 0 // (op 1751) 
s_cselect_b32 s57, 0, -1 // (op 1751) 

s_and_b32 s56, s47, 31 // (op 1751) 
s_lshl_b32 s58, 1, s56 // (op 1751) 
s_add_i32 s59, s58, s57 // (op 1751) 


s_and_b32 s57, s54, s59 // (op 1751) 


s_add_i32 s54, s55, s57 // (op 1751) 


s_ashr_i32 s55, s54, s56 // (op 1751) 


s_lshl_b32 s54, s47, 24 // (op 1751) 
s_ashr_i32 s56, s54, 31 // (op 1751) 

s_xor_b32 s54, s55, s56 // (op 1751) 

s_sub_i32 s55, s54, s56 // (op 1751) 


s_add_u32 s56, s6, 64 // (op 1511) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1511) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1511) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1511) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1511) 

s_mul_i32 s54, s44, s56 // (op 1511) 
s_mul_i32 s58, s48, s54 // (op 1511) 

s_mul_i32 s54, s53, s58 // (op 1511) 

s_sub_u32 s58, s16, s54 // (op 1511) 

s_mov_b32 s54, s58 // (op 1511) 
s_mul_hi_i32 s59, s54, s15 // (op 1511) 
s_add_i32 s60, s59, s54 // (op 1511) 


s_ashr_i32 s54, s60, 31 // (op 1511) 
s_cmp_eq_i32 s15, 0 // (op 1511) 
s_mov_b32 s59, scc // (op 1511) 
s_cmp_eq_i32 s59, 0 // (op 1511) 
s_cselect_b32 s61, 0, -1 // (op 1511) 

s_and_b32 s59, s46, 31 // (op 1511) 
s_lshl_b32 s62, 1, s59 // (op 1511) 
s_add_i32 s63, s62, s61 // (op 1511) 


s_and_b32 s61, s54, s63 // (op 1511) 


s_add_i32 s54, s60, s61 // (op 1511) 


s_ashr_i32 s60, s54, s59 // (op 1511) 


s_lshl_b32 s54, s46, 24 // (op 1511) 
s_ashr_i32 s59, s54, 31 // (op 1511) 

s_xor_b32 s54, s60, s59 // (op 1511) 

s_sub_i32 s60, s54, s59 // (op 1511) 


s_mul_i32 s54, s45, s44 // (op 1511) 
s_add_u32 s62, s8, 16 // (op 1511) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 1511) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1511) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1511) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 1511) 

s_sub_i32 s59, s62, s54 // (op 1511) 

s_mul_i32 s61, s60, s59 // (op 1511) 

s_sub_u32 s59, s58, s61 // (op 1511) 


s_add_u32 s58, s59, s54 // (op 1511) 


s_cmp_eq_i32 s53, 0 // (op 1511) 
s_mov_b32 s54, scc // (op 1511) 

s_cmp_eq_i32 s54, 0 // (op 1511) 
s_cselect_b32 s53, s58, s1 // (op 1511) 


s_mul_i32 s1, s51, s56 // (op 1511) 


v_lshl_add_u32 v11, s53, 4, v8 // (op 1511) 


s_sub_i32 s51, s18, s1 // (op 1511) 


s_cmp_eq_i32 s54, 0 // (op 1511) 
s_cselect_b32 s1, s60, s51 // (op 1511) 



v_lshl_add_u32 v8, s1, 6, v7 // (op 1511) 


v_mul_lo_u32 v7, v8, s2 // (op 1511) 

v_add_lshl_u32 v8, v11, v7, 2 // (op 1511) 


s_add_u32 s56, s6, 64 // (op 1751) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1751) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1751) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1751) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1751) 

s_mul_i32 s1, s44, s56 // (op 1751) 
s_mul_i32 s18, s48, s1 // (op 1751) 

s_mul_i32 s1, s55, s18 // (op 1751) 

s_sub_u32 s18, s16, s1 // (op 1751) 

s_mov_b32 s1, s18 // (op 1751) 
s_mul_hi_i32 s51, s1, s15 // (op 1751) 
s_add_i32 s53, s51, s1 // (op 1751) 


s_ashr_i32 s1, s53, 31 // (op 1751) 
s_cmp_eq_i32 s15, 0 // (op 1751) 
s_mov_b32 s51, scc // (op 1751) 
s_cmp_eq_i32 s51, 0 // (op 1751) 
s_cselect_b32 s54, 0, -1 // (op 1751) 

s_and_b32 s51, s46, 31 // (op 1751) 
s_lshl_b32 s58, 1, s51 // (op 1751) 
s_add_i32 s59, s58, s54 // (op 1751) 


s_and_b32 s54, s1, s59 // (op 1751) 


s_add_i32 s1, s53, s54 // (op 1751) 


s_ashr_i32 s53, s1, s51 // (op 1751) 


s_lshl_b32 s1, s46, 24 // (op 1751) 
s_ashr_i32 s51, s1, 31 // (op 1751) 

s_xor_b32 s1, s53, s51 // (op 1751) 

s_sub_i32 s53, s1, s51 // (op 1751) 


s_mul_i32 s1, s45, s44 // (op 1751) 
s_add_u32 s58, s8, 16 // (op 1751) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 1751) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1751) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1751) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1751) 

s_sub_i32 s51, s58, s1 // (op 1751) 

s_mul_i32 s54, s53, s51 // (op 1751) 

s_sub_u32 s51, s18, s54 // (op 1751) 


s_add_u32 s18, s51, s1 // (op 1751) 


s_cmp_eq_i32 s55, 0 // (op 1751) 
s_mov_b32 s1, scc // (op 1751) 

s_cmp_eq_i32 s1, 0 // (op 1751) 
s_cselect_b32 s51, s18, s52 // (op 1751) 


s_mul_i32 s18, s19, s56 // (op 1751) 


v_lshl_add_u32 v7, s51, 4, v14 // (op 1751) 


s_sub_i32 s19, s0, s18 // (op 1751) 


s_cmp_eq_i32 s1, 0 // (op 1751) 
s_cselect_b32 s0, s53, s19 // (op 1751) 



v_lshl_add_u32 v11, s0, 6, v10 // (op 1751) 


v_mul_lo_u32 v10, v11, s4 // (op 1751) 

v_add_lshl_u32 v11, v7, v10, 2 // (op 1751) 


s_mul_i32 s1, s2, s7 // (op 1513) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1513) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1513) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1513) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1513) most significant: sum
s_add_u32 s1, s1, s19 // (op 1513) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1513) 

s_mul_i32 s1, s4, s7 // (op 1753) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 1753) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 1753) least significant
s_mul_hi_u32 s52, s4, s6 // (op 1753) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1753) most significant: sum
s_add_u32 s1, s1, s52 // (op 1753) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 1753) 

s_mov_b32 s56, s20 // (op 1513) reg expression
s_mov_b32 s57, s21 // (op 1513) reg expression
s_mov_b32 s58, s18 // (op 1513) reg expression
s_mov_b32 s59, 131072 // (op 1513) call()

s_mov_b32 s60, s22 // (op 1753) reg expression

s_mov_b32 s61, s23 // (op 1753) reg expression
s_mov_b32 s62, s52 // (op 1753) reg expression
s_mov_b32 s63, 131072 // (op 1753) call()


v_mov_b32 v7, s49 // (op 2) Move value
v_mov_b32 v10, s50 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 398) call()
v_accvgpr_write a5, 0.00000 // (op 398) call()
v_accvgpr_write a6, 0.00000 // (op 398) call()
v_accvgpr_write a7, 0.00000 // (op 398) call()
s_mul_hi_i32 s0, s16, s11 // (op 1278) 
v_lshrrev_b32 v14, 4, v1 // (op 1278) 
s_add_i32 s1, s0, s16 // (op 1278) 

v_and_b32 v15, -16, v1 // (op 1278) 
s_ashr_i32 s0, s1, 31 // (op 1278) 
v_sub_u32 v16, v1, v15 // (op 1278) 

s_cmp_eq_i32 s11, 0 // (op 1278) 
s_mov_b32 s18, scc // (op 1278) 
v_lshlrev_b32 v15, 4, v16 // (op 1278) 

s_cmp_eq_i32 s18, 0 // (op 1278) 
s_cselect_b32 s19, 0, -1 // (op 1278) 

v_mov_b32 v17, 0 // (op 1278) convert
v_mov_b32 v16, v15 // (op 1278) convert

s_and_b32 s18, s12, 31 // (op 1278) 
s_lshl_b32 s20, 1, s18 // (op 1278) 
s_add_i32 s21, s20, s19 // (op 1278) 


s_and_b32 s19, s0, s21 // (op 1278) 


s_add_i32 s0, s1, s19 // (op 1278) 


s_ashr_i32 s1, s0, s18 // (op 1278) 


s_lshl_b32 s0, s12, 24 // (op 1278) 
s_ashr_i32 s18, s0, 31 // (op 1278) 

s_xor_b32 s0, s1, s18 // (op 1278) 

s_sub_i32 s1, s0, s18 // (op 1278) 


s_mul_hi_i32 s0, s1, s13 // (op 1278) 
s_add_i32 s18, s0, s1 // (op 1278) 

s_ashr_i32 s0, s18, 31 // (op 1278) 
s_cmp_eq_i32 s13, 0 // (op 1278) 
s_mov_b32 s19, scc // (op 1278) 
s_cmp_eq_i32 s19, 0 // (op 1278) 
s_cselect_b32 s20, 0, -1 // (op 1278) 

s_and_b32 s19, s14, 31 // (op 1278) 
s_lshl_b32 s21, 1, s19 // (op 1278) 
s_add_i32 s22, s21, s20 // (op 1278) 


s_and_b32 s23, s0, s22 // (op 1278) 


s_add_i32 s0, s18, s23 // (op 1278) 


s_ashr_i32 s18, s0, s19 // (op 1278) 


s_lshl_b32 s0, s14, 24 // (op 1278) 
s_ashr_i32 s19, s0, 31 // (op 1278) 

s_xor_b32 s0, s18, s19 // (op 1278) 

s_sub_i32 s18, s0, s19 // (op 1278) 


s_mul_i32 s0, s1, s44 // (op 1278) 

s_sub_u32 s1, s16, s0 // (op 1278) 

s_mul_i32 s0, s18, s44 // (op 1278) 
s_add_u32 s0, s0, s1 // (op 1278) 


s_mul_hi_i32 s1, s16, s10 // (op 1278) 
s_add_i32 s18, s1, s16 // (op 1278) 

s_ashr_i32 s1, s18, 31 // (op 1278) 
s_cmp_eq_i32 s10, 0 // (op 1278) 
s_mov_b32 s19, scc // (op 1278) 
s_cmp_eq_i32 s19, 0 // (op 1278) 
s_cselect_b32 s20, 0, -1 // (op 1278) 

s_and_b32 s19, s47, 31 // (op 1278) 
s_lshl_b32 s21, 1, s19 // (op 1278) 
s_add_i32 s22, s21, s20 // (op 1278) 


s_and_b32 s23, s1, s22 // (op 1278) 


s_add_i32 s1, s18, s23 // (op 1278) 


s_ashr_i32 s18, s1, s19 // (op 1278) 


s_lshl_b32 s1, s47, 24 // (op 1278) 
s_ashr_i32 s19, s1, 31 // (op 1278) 

s_xor_b32 s1, s18, s19 // (op 1278) 

s_sub_i32 s18, s1, s19 // (op 1278) 


s_add_u32 s22, s6, 64 // (op 1278) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1278) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 1278) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 1278) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 6 // (op 1278) 

s_mul_i32 s1, s44, s22 // (op 1278) 

s_mul_i32 s19, s48, s1 // (op 1278) 

s_mul_i32 s1, s18, s19 // (op 1278) 

s_sub_u32 s19, s16, s1 // (op 1278) 

s_mov_b32 s1, s19 // (op 1278) 
s_mul_hi_i32 s20, s1, s15 // (op 1278) 
s_add_i32 s21, s20, s1 // (op 1278) 


s_ashr_i32 s1, s21, 31 // (op 1278) 
s_cmp_eq_i32 s15, 0 // (op 1278) 
s_mov_b32 s20, scc // (op 1278) 
s_cmp_eq_i32 s20, 0 // (op 1278) 
s_cselect_b32 s22, 0, -1 // (op 1278) 

s_and_b32 s20, s46, 31 // (op 1278) 
s_lshl_b32 s23, 1, s20 // (op 1278) 
s_add_i32 s49, s23, s22 // (op 1278) 


s_and_b32 s22, s1, s49 // (op 1278) 


s_add_i32 s1, s21, s22 // (op 1278) 


s_ashr_i32 s21, s1, s20 // (op 1278) 


s_lshl_b32 s1, s46, 24 // (op 1278) 
s_ashr_i32 s20, s1, 31 // (op 1278) 

s_xor_b32 s1, s21, s20 // (op 1278) 

s_sub_i32 s21, s1, s20 // (op 1278) 


s_mul_i32 s1, s45, s44 // (op 1278) 
s_add_u32 s22, s8, 16 // (op 1278) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 1278) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 1278) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 1278) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 4 // (op 1278) 

s_sub_i32 s20, s22, s1 // (op 1278) 

s_mul_i32 s22, s21, s20 // (op 1278) 


s_sub_u32 s23, s19, s22 // (op 1278) 


s_add_u32 s19, s23, s1 // (op 1278) 


s_cmp_eq_i32 s18, 0 // (op 1278) 
s_mov_b32 s1, scc // (op 1278) 

s_cmp_eq_i32 s1, 0 // (op 1278) 
s_cselect_b32 s18, s19, s0 // (op 1278) 



v_lshl_add_u32 v15, s18, 4, v14 // (op 1278) 


v_mul_lo_u32 v19, v15, s25 // (op 1278) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1278) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 1278) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1278) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 1278) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1278) most significant half



s_mul_i32 s1, s24, s9 // (op 1280) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 1280) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1280) least significant
s_mul_hi_u32 s19, s24, s8 // (op 1280) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1280) most significant: sum
s_add_u32 s1, s1, s19 // (op 1280) most significant: sum


s_mov_b32 s20, s26 // (op 1280) reg expression
s_mov_b32 s21, s27 // (op 1280) reg expression
s_mov_b32 s22, s0 // (op 1280) reg expression
s_mov_b32 s23, 131072 // (op 1280) call()

s_mul_hi_i32 s0, s16, s11 // (op 1297) 
v_lshrrev_b32 v14, 4, v1 // (op 1297) 
s_add_i32 s1, s0, s16 // (op 1297) 

v_and_b32 v15, -16, v1 // (op 1297) 
s_ashr_i32 s0, s1, 31 // (op 1297) 
v_sub_u32 v16, v1, v15 // (op 1297) 

s_cmp_eq_i32 s11, 0 // (op 1297) 
s_mov_b32 s18, scc // (op 1297) 
v_lshlrev_b32 v15, 4, v16 // (op 1297) 

s_cmp_eq_i32 s18, 0 // (op 1297) 
s_cselect_b32 s19, 0, -1 // (op 1297) 

v_mov_b32 v17, 0 // (op 1297) convert
v_mov_b32 v16, v15 // (op 1297) convert

s_and_b32 s18, s12, 31 // (op 1297) 
s_lshl_b32 s49, 1, s18 // (op 1297) 
s_add_i32 s50, s49, s19 // (op 1297) 


s_and_b32 s19, s0, s50 // (op 1297) 


s_add_i32 s0, s1, s19 // (op 1297) 


s_ashr_i32 s1, s0, s18 // (op 1297) 


s_lshl_b32 s0, s12, 24 // (op 1297) 
s_ashr_i32 s18, s0, 31 // (op 1297) 

s_xor_b32 s0, s1, s18 // (op 1297) 

s_sub_i32 s1, s0, s18 // (op 1297) 


s_mul_hi_i32 s0, s1, s13 // (op 1297) 
s_add_i32 s18, s0, s1 // (op 1297) 

s_ashr_i32 s0, s18, 31 // (op 1297) 
s_cmp_eq_i32 s13, 0 // (op 1297) 
s_mov_b32 s19, scc // (op 1297) 
s_cmp_eq_i32 s19, 0 // (op 1297) 
s_cselect_b32 s49, 0, -1 // (op 1297) 

s_and_b32 s19, s14, 31 // (op 1297) 
s_lshl_b32 s50, 1, s19 // (op 1297) 
s_add_i32 s51, s50, s49 // (op 1297) 


s_and_b32 s49, s0, s51 // (op 1297) 


s_add_i32 s0, s18, s49 // (op 1297) 


s_ashr_i32 s18, s0, s19 // (op 1297) 


s_lshl_b32 s0, s14, 24 // (op 1297) 
s_ashr_i32 s19, s0, 31 // (op 1297) 

s_xor_b32 s0, s18, s19 // (op 1297) 

s_sub_i32 s18, s0, s19 // (op 1297) 


s_mul_i32 s0, s1, s44 // (op 1297) 

s_sub_u32 s1, s16, s0 // (op 1297) 

s_mul_i32 s0, s18, s44 // (op 1297) 
s_add_u32 s0, s0, s1 // (op 1297) 


s_mul_hi_i32 s1, s16, s10 // (op 1297) 
s_add_i32 s18, s1, s16 // (op 1297) 

s_ashr_i32 s1, s18, 31 // (op 1297) 
s_cmp_eq_i32 s10, 0 // (op 1297) 
s_mov_b32 s19, scc // (op 1297) 
s_cmp_eq_i32 s19, 0 // (op 1297) 
s_cselect_b32 s49, 0, -1 // (op 1297) 

s_and_b32 s19, s47, 31 // (op 1297) 
s_lshl_b32 s50, 1, s19 // (op 1297) 
s_add_i32 s51, s50, s49 // (op 1297) 


s_and_b32 s49, s1, s51 // (op 1297) 


s_add_i32 s1, s18, s49 // (op 1297) 


s_ashr_i32 s18, s1, s19 // (op 1297) 


s_lshl_b32 s1, s47, 24 // (op 1297) 
s_ashr_i32 s19, s1, 31 // (op 1297) 

s_xor_b32 s1, s18, s19 // (op 1297) 

s_sub_i32 s18, s1, s19 // (op 1297) 


s_add_u32 s54, s6, 64 // (op 1297) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 1297) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1297) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1297) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 1297) 

s_mul_i32 s1, s44, s50 // (op 1297) 

s_mul_i32 s19, s48, s1 // (op 1297) 

s_mul_i32 s1, s18, s19 // (op 1297) 

s_sub_u32 s19, s16, s1 // (op 1297) 

s_mov_b32 s1, s19 // (op 1297) 
s_mul_hi_i32 s49, s1, s15 // (op 1297) 
s_add_i32 s50, s49, s1 // (op 1297) 


s_ashr_i32 s1, s50, 31 // (op 1297) 
s_cmp_eq_i32 s15, 0 // (op 1297) 
s_mov_b32 s49, scc // (op 1297) 
s_cmp_eq_i32 s49, 0 // (op 1297) 
s_cselect_b32 s51, 0, -1 // (op 1297) 

s_and_b32 s49, s46, 31 // (op 1297) 
s_lshl_b32 s52, 1, s49 // (op 1297) 
s_add_i32 s53, s52, s51 // (op 1297) 


s_and_b32 s51, s1, s53 // (op 1297) 


s_add_i32 s1, s50, s51 // (op 1297) 


s_ashr_i32 s50, s1, s49 // (op 1297) 


s_lshl_b32 s1, s46, 24 // (op 1297) 
s_ashr_i32 s49, s1, 31 // (op 1297) 

s_xor_b32 s1, s50, s49 // (op 1297) 

s_sub_i32 s50, s1, s49 // (op 1297) 


s_mul_i32 s1, s45, s44 // (op 1297) 
s_add_u32 s54, s8, 16 // (op 1297) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 1297) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1297) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1297) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 4 // (op 1297) 

s_sub_i32 s49, s54, s1 // (op 1297) 

s_mul_i32 s51, s50, s49 // (op 1297) 


s_sub_u32 s49, s19, s51 // (op 1297) 


s_add_u32 s19, s49, s1 // (op 1297) 


s_cmp_eq_i32 s18, 0 // (op 1297) 
s_mov_b32 s1, scc // (op 1297) 

s_cmp_eq_i32 s1, 0 // (op 1297) 
s_cselect_b32 s18, s19, s0 // (op 1297) 



v_lshl_add_u32 v15, s18, 4, v14 // (op 1297) 


v_mul_lo_u32 v21, v15, s25 // (op 1297) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1297) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 1297) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1297) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 1297) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 1297) most significant half



s_mul_i32 s1, s24, s9 // (op 1299) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 1299) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1299) least significant
s_mul_hi_u32 s19, s24, s8 // (op 1299) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1299) most significant: sum
s_add_u32 s1, s1, s19 // (op 1299) most significant: sum


s_mov_b32 s20, s26 // (op 1299) reg expression
s_mov_b32 s21, s27 // (op 1299) reg expression
s_mov_b32 s22, s0 // (op 1299) reg expression
s_mov_b32 s23, 131072 // (op 1299) call()

s_mul_hi_i32 s0, s16, s11 // (op 1342) 
v_and_b32 v14, -4, v1 // (op 1342) 
s_add_i32 s1, s0, s16 // (op 1342) 

v_sub_u32 v15, v1, v14 // (op 1342) 

s_ashr_i32 s0, s1, 31 // (op 1342) 
v_lshlrev_b32 v14, 4, v15 // (op 1342) 

s_cmp_eq_i32 s11, 0 // (op 1342) 
s_mov_b32 s18, scc // (op 1342) 
v_lshrrev_b32 v15, 2, v1 // (op 1342) 
s_cmp_eq_i32 s18, 0 // (op 1342) 
s_cselect_b32 s19, 0, -1 // (op 1342) 

s_and_b32 s18, s12, 31 // (op 1342) 
s_lshl_b32 s49, 1, s18 // (op 1342) 
s_add_i32 s50, s49, s19 // (op 1342) 


s_and_b32 s19, s0, s50 // (op 1342) 


s_add_i32 s0, s1, s19 // (op 1342) 


s_ashr_i32 s1, s0, s18 // (op 1342) 


s_lshl_b32 s0, s12, 24 // (op 1342) 
s_ashr_i32 s18, s0, 31 // (op 1342) 

s_xor_b32 s0, s1, s18 // (op 1342) 

s_sub_i32 s1, s0, s18 // (op 1342) 


s_mul_hi_i32 s0, s1, s13 // (op 1342) 
s_add_i32 s18, s0, s1 // (op 1342) 

s_ashr_i32 s0, s18, 31 // (op 1342) 
s_cmp_eq_i32 s13, 0 // (op 1342) 
s_mov_b32 s19, scc // (op 1342) 
s_cmp_eq_i32 s19, 0 // (op 1342) 
s_cselect_b32 s49, 0, -1 // (op 1342) 

s_and_b32 s19, s14, 31 // (op 1342) 
s_lshl_b32 s50, 1, s19 // (op 1342) 
s_add_i32 s51, s50, s49 // (op 1342) 


s_and_b32 s49, s0, s51 // (op 1342) 


s_add_i32 s0, s18, s49 // (op 1342) 


s_ashr_i32 s18, s0, s19 // (op 1342) 


s_lshl_b32 s0, s14, 24 // (op 1342) 
s_ashr_i32 s19, s0, 31 // (op 1342) 

s_xor_b32 s0, s18, s19 // (op 1342) 

s_sub_i32 s18, s0, s19 // (op 1342) 


s_add_u32 s54, s6, 64 // (op 1342) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 1342) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1342) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1342) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 1342) 

s_mul_i32 s0, s18, s50 // (op 1342) 

s_sub_i32 s49, s1, s0 // (op 1342) 


s_mul_hi_i32 s0, s16, s10 // (op 1342) 
s_add_i32 s1, s0, s16 // (op 1342) 

s_ashr_i32 s0, s1, 31 // (op 1342) 
s_cmp_eq_i32 s10, 0 // (op 1342) 
s_mov_b32 s18, scc // (op 1342) 
s_cmp_eq_i32 s18, 0 // (op 1342) 
s_cselect_b32 s19, 0, -1 // (op 1342) 

s_and_b32 s18, s47, 31 // (op 1342) 
s_lshl_b32 s52, 1, s18 // (op 1342) 
s_add_i32 s53, s52, s19 // (op 1342) 


s_and_b32 s19, s0, s53 // (op 1342) 


s_add_i32 s0, s1, s19 // (op 1342) 


s_ashr_i32 s1, s0, s18 // (op 1342) 


s_lshl_b32 s0, s47, 24 // (op 1342) 
s_ashr_i32 s18, s0, 31 // (op 1342) 

s_xor_b32 s0, s1, s18 // (op 1342) 

s_sub_i32 s1, s0, s18 // (op 1342) 


s_mul_i32 s0, s44, s50 // (op 1342) 

s_mul_i32 s18, s48, s0 // (op 1342) 

s_mul_i32 s0, s1, s18 // (op 1342) 

s_sub_u32 s18, s16, s0 // (op 1342) 

s_mul_hi_i32 s0, s18, s15 // (op 1342) 
s_add_i32 s19, s0, s18 // (op 1342) 


s_ashr_i32 s0, s19, 31 // (op 1342) 
s_cmp_eq_i32 s15, 0 // (op 1342) 
s_mov_b32 s18, scc // (op 1342) 
s_cmp_eq_i32 s18, 0 // (op 1342) 
s_cselect_b32 s50, 0, -1 // (op 1342) 

s_and_b32 s18, s46, 31 // (op 1342) 
s_lshl_b32 s51, 1, s18 // (op 1342) 
s_add_i32 s52, s51, s50 // (op 1342) 


s_and_b32 s50, s0, s52 // (op 1342) 


s_add_i32 s0, s19, s50 // (op 1342) 


s_ashr_i32 s19, s0, s18 // (op 1342) 


s_lshl_b32 s0, s46, 24 // (op 1342) 
s_ashr_i32 s18, s0, 31 // (op 1342) 

s_xor_b32 s0, s19, s18 // (op 1342) 

s_sub_i32 s19, s0, s18 // (op 1342) 


s_cmp_eq_i32 s1, 0 // (op 1342) 
s_mov_b32 s0, scc // (op 1342) 

s_cmp_eq_i32 s0, 0 // (op 1342) 
s_cselect_b32 s1, s19, s49 // (op 1342) 



v_lshl_add_u32 v16, s1, 6, v14 // (op 1342) 


v_mov_b32 v23, 0 // (op 1342) convert
v_mov_b32 v22, v16 // (op 1342) convert

v_mul_lo_u32 v17, v15, s29 // (op 1342) most significant: low of low * high
v_mul_hi_u32 v14, v15, s28 // (op 1342) most significant: high of low * low
v_mul_lo_u32 v16, v15, s28 // (op 1342) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 1342) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v22 // (op 1342) least significant half
v_addc_co_u32 v17, s[0:1], v17, v23, s[0:1] // (op 1342) most significant half



s_mul_i32 s1, s28, s33 // (op 1344) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 1344) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1344) least significant
s_mul_hi_u32 s19, s28, s32 // (op 1344) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1344) most significant: sum
s_add_u32 s1, s1, s19 // (op 1344) most significant: sum


s_mov_b32 s52, s30 // (op 1344) reg expression
s_mov_b32 s53, s31 // (op 1344) reg expression
s_mov_b32 s54, s0 // (op 1344) reg expression
s_mov_b32 s55, 131072 // (op 1344) call()

s_mul_hi_i32 s0, s16, s11 // (op 1361) 
v_and_b32 v14, -4, v1 // (op 1361) 
s_add_i32 s1, s0, s16 // (op 1361) 

v_sub_u32 v15, v1, v14 // (op 1361) 

s_ashr_i32 s0, s1, 31 // (op 1361) 
v_lshlrev_b32 v14, 4, v15 // (op 1361) 

s_cmp_eq_i32 s11, 0 // (op 1361) 
s_mov_b32 s18, scc // (op 1361) 
v_lshrrev_b32 v15, 2, v1 // (op 1361) 
s_cmp_eq_i32 s18, 0 // (op 1361) 
s_cselect_b32 s19, 0, -1 // (op 1361) 

s_and_b32 s18, s12, 31 // (op 1361) 
s_lshl_b32 s49, 1, s18 // (op 1361) 
s_add_i32 s50, s49, s19 // (op 1361) 


s_and_b32 s19, s0, s50 // (op 1361) 


s_add_i32 s0, s1, s19 // (op 1361) 


s_ashr_i32 s1, s0, s18 // (op 1361) 


s_lshl_b32 s0, s12, 24 // (op 1361) 
s_ashr_i32 s18, s0, 31 // (op 1361) 

s_xor_b32 s0, s1, s18 // (op 1361) 

s_sub_i32 s1, s0, s18 // (op 1361) 


s_mul_hi_i32 s0, s1, s13 // (op 1361) 
s_add_i32 s18, s0, s1 // (op 1361) 

s_ashr_i32 s0, s18, 31 // (op 1361) 
s_cmp_eq_i32 s13, 0 // (op 1361) 
s_mov_b32 s19, scc // (op 1361) 
s_cmp_eq_i32 s19, 0 // (op 1361) 
s_cselect_b32 s49, 0, -1 // (op 1361) 

s_and_b32 s19, s14, 31 // (op 1361) 
s_lshl_b32 s50, 1, s19 // (op 1361) 
s_add_i32 s51, s50, s49 // (op 1361) 


s_and_b32 s49, s0, s51 // (op 1361) 


s_add_i32 s0, s18, s49 // (op 1361) 


s_ashr_i32 s18, s0, s19 // (op 1361) 


s_lshl_b32 s0, s14, 24 // (op 1361) 
s_ashr_i32 s19, s0, 31 // (op 1361) 

s_xor_b32 s0, s18, s19 // (op 1361) 

s_sub_i32 s18, s0, s19 // (op 1361) 


s_add_u32 s50, s6, 64 // (op 1361) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1361) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1361) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1361) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1361) 

s_mul_i32 s0, s18, s50 // (op 1361) 

s_sub_i32 s49, s1, s0 // (op 1361) 


s_mul_hi_i32 s0, s16, s10 // (op 1361) 
s_add_i32 s1, s0, s16 // (op 1361) 

s_ashr_i32 s0, s1, 31 // (op 1361) 
s_cmp_eq_i32 s10, 0 // (op 1361) 
s_mov_b32 s18, scc // (op 1361) 
s_cmp_eq_i32 s18, 0 // (op 1361) 
s_cselect_b32 s19, 0, -1 // (op 1361) 

s_and_b32 s18, s47, 31 // (op 1361) 
s_lshl_b32 s64, 1, s18 // (op 1361) 
s_add_i32 s65, s64, s19 // (op 1361) 


s_and_b32 s19, s0, s65 // (op 1361) 


s_add_i32 s0, s1, s19 // (op 1361) 


s_ashr_i32 s1, s0, s18 // (op 1361) 


s_lshl_b32 s0, s47, 24 // (op 1361) 
s_ashr_i32 s18, s0, 31 // (op 1361) 

s_xor_b32 s0, s1, s18 // (op 1361) 

s_sub_i32 s1, s0, s18 // (op 1361) 


s_mul_i32 s0, s44, s50 // (op 1361) 

s_mul_i32 s18, s48, s0 // (op 1361) 

s_mul_i32 s0, s1, s18 // (op 1361) 

s_sub_u32 s18, s16, s0 // (op 1361) 

s_mul_hi_i32 s0, s18, s15 // (op 1361) 
s_add_i32 s19, s0, s18 // (op 1361) 


s_ashr_i32 s0, s19, 31 // (op 1361) 
s_cmp_eq_i32 s15, 0 // (op 1361) 
s_mov_b32 s18, scc // (op 1361) 
s_cmp_eq_i32 s18, 0 // (op 1361) 
s_cselect_b32 s50, 0, -1 // (op 1361) 

s_and_b32 s18, s46, 31 // (op 1361) 
s_lshl_b32 s51, 1, s18 // (op 1361) 
s_add_i32 s64, s51, s50 // (op 1361) 


s_and_b32 s50, s0, s64 // (op 1361) 


s_add_i32 s0, s19, s50 // (op 1361) 


s_ashr_i32 s19, s0, s18 // (op 1361) 


s_lshl_b32 s0, s46, 24 // (op 1361) 
s_ashr_i32 s18, s0, 31 // (op 1361) 

s_xor_b32 s0, s19, s18 // (op 1361) 

s_sub_i32 s19, s0, s18 // (op 1361) 


s_cmp_eq_i32 s1, 0 // (op 1361) 
s_mov_b32 s0, scc // (op 1361) 

s_cmp_eq_i32 s0, 0 // (op 1361) 
s_cselect_b32 s1, s19, s49 // (op 1361) 



v_lshl_add_u32 v22, s1, 6, v14 // (op 1361) 


v_mov_b32 v25, 0 // (op 1361) convert
v_mov_b32 v24, v22 // (op 1361) convert

v_mul_lo_u32 v23, v15, s29 // (op 1361) most significant: low of low * high
v_mul_hi_u32 v14, v15, s28 // (op 1361) most significant: high of low * low
v_mul_lo_u32 v22, v15, s28 // (op 1361) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 1361) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v24 // (op 1361) least significant half
v_addc_co_u32 v23, s[0:1], v23, v25, s[0:1] // (op 1361) most significant half



s_mul_i32 s1, s28, s33 // (op 1363) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 1363) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1363) least significant
s_mul_hi_u32 s19, s28, s32 // (op 1363) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1363) most significant: sum
s_add_u32 s1, s1, s19 // (op 1363) most significant: sum


s_mov_b32 s52, s30 // (op 1363) reg expression
s_mov_b32 s53, s31 // (op 1363) reg expression
s_mov_b32 s54, s0 // (op 1363) reg expression
s_mov_b32 s55, 131072 // (op 1363) call()

s_mul_hi_i32 s0, s16, s11 // (op 1406) 
v_lshrrev_b32 v14, 3, v1 // (op 1406) 
s_add_i32 s1, s0, s16 // (op 1406) 

v_and_b32 v15, -8, v1 // (op 1406) 
s_ashr_i32 s0, s1, 31 // (op 1406) 
v_sub_u32 v24, v1, v15 // (op 1406) 

s_cmp_eq_i32 s11, 0 // (op 1406) 
s_mov_b32 s18, scc // (op 1406) 
s_cmp_eq_i32 s18, 0 // (op 1406) 
s_cselect_b32 s19, 0, -1 // (op 1406) 

s_and_b32 s18, s12, 31 // (op 1406) 
s_lshl_b32 s49, 1, s18 // (op 1406) 
s_add_i32 s50, s49, s19 // (op 1406) 


s_and_b32 s19, s0, s50 // (op 1406) 


s_add_i32 s0, s1, s19 // (op 1406) 


s_ashr_i32 s1, s0, s18 // (op 1406) 


s_lshl_b32 s0, s12, 24 // (op 1406) 
s_ashr_i32 s18, s0, 31 // (op 1406) 

s_xor_b32 s0, s1, s18 // (op 1406) 

s_sub_i32 s1, s0, s18 // (op 1406) 


s_mul_hi_i32 s0, s1, s13 // (op 1406) 
s_add_i32 s18, s0, s1 // (op 1406) 

s_ashr_i32 s0, s18, 31 // (op 1406) 
s_cmp_eq_i32 s13, 0 // (op 1406) 
s_mov_b32 s19, scc // (op 1406) 
s_cmp_eq_i32 s19, 0 // (op 1406) 
s_cselect_b32 s49, 0, -1 // (op 1406) 

s_and_b32 s19, s14, 31 // (op 1406) 
s_lshl_b32 s50, 1, s19 // (op 1406) 
s_add_i32 s51, s50, s49 // (op 1406) 


s_and_b32 s49, s0, s51 // (op 1406) 


s_add_i32 s0, s18, s49 // (op 1406) 


s_ashr_i32 s18, s0, s19 // (op 1406) 


s_lshl_b32 s0, s14, 24 // (op 1406) 
s_ashr_i32 s19, s0, 31 // (op 1406) 

s_xor_b32 s0, s18, s19 // (op 1406) 

s_sub_i32 s18, s0, s19 // (op 1406) 


s_mul_i32 s0, s1, s44 // (op 1406) 

s_sub_u32 s1, s16, s0 // (op 1406) 

s_mul_i32 s0, s18, s44 // (op 1406) 
s_add_u32 s0, s0, s1 // (op 1406) 


s_mul_hi_i32 s1, s16, s10 // (op 1406) 
s_add_i32 s18, s1, s16 // (op 1406) 

s_ashr_i32 s1, s18, 31 // (op 1406) 
s_cmp_eq_i32 s10, 0 // (op 1406) 
s_mov_b32 s19, scc // (op 1406) 
s_cmp_eq_i32 s19, 0 // (op 1406) 
s_cselect_b32 s49, 0, -1 // (op 1406) 

s_and_b32 s19, s47, 31 // (op 1406) 
s_lshl_b32 s50, 1, s19 // (op 1406) 
s_add_i32 s51, s50, s49 // (op 1406) 


s_and_b32 s49, s1, s51 // (op 1406) 


s_add_i32 s1, s18, s49 // (op 1406) 


s_ashr_i32 s18, s1, s19 // (op 1406) 


s_lshl_b32 s1, s47, 24 // (op 1406) 
s_ashr_i32 s19, s1, 31 // (op 1406) 

s_xor_b32 s1, s18, s19 // (op 1406) 

s_sub_i32 s18, s1, s19 // (op 1406) 


s_add_u32 s50, s6, 64 // (op 1406) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1406) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1406) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1406) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1406) 

s_mul_i32 s1, s44, s50 // (op 1406) 

s_mul_i32 s19, s48, s1 // (op 1406) 

s_mul_i32 s1, s18, s19 // (op 1406) 

s_sub_u32 s19, s16, s1 // (op 1406) 

s_mov_b32 s1, s19 // (op 1406) 
s_mul_hi_i32 s49, s1, s15 // (op 1406) 
s_add_i32 s50, s49, s1 // (op 1406) 


s_ashr_i32 s1, s50, 31 // (op 1406) 
s_cmp_eq_i32 s15, 0 // (op 1406) 
s_mov_b32 s49, scc // (op 1406) 
s_cmp_eq_i32 s49, 0 // (op 1406) 
s_cselect_b32 s51, 0, -1 // (op 1406) 

s_and_b32 s49, s46, 31 // (op 1406) 
s_lshl_b32 s64, 1, s49 // (op 1406) 
s_add_i32 s65, s64, s51 // (op 1406) 


s_and_b32 s51, s1, s65 // (op 1406) 


s_add_i32 s1, s50, s51 // (op 1406) 


s_ashr_i32 s50, s1, s49 // (op 1406) 


s_lshl_b32 s1, s46, 24 // (op 1406) 
s_ashr_i32 s49, s1, 31 // (op 1406) 

s_xor_b32 s1, s50, s49 // (op 1406) 

s_sub_i32 s50, s1, s49 // (op 1406) 


s_mul_i32 s1, s45, s44 // (op 1406) 
s_add_u32 s64, s8, 16 // (op 1406) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 1406) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1406) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1406) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 1406) 

s_sub_i32 s49, s64, s1 // (op 1406) 

s_mul_i32 s51, s50, s49 // (op 1406) 


s_sub_u32 s49, s19, s51 // (op 1406) 


s_add_u32 s19, s49, s1 // (op 1406) 


s_cmp_eq_i32 s18, 0 // (op 1406) 
s_mov_b32 s1, scc // (op 1406) 

s_cmp_eq_i32 s1, 0 // (op 1406) 
s_cselect_b32 s18, s19, s0 // (op 1406) 



v_lshl_add_u32 v15, s18, 4, v14 // (op 1406) 


v_mul_lo_u32 v14, v15, s34 // (op 1406) 
v_add_u32 v14, v14, v24 // (op 1406) 


s_mul_i32 s1, s34, s9 // (op 1408) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1408) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1408) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1408) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1408) most significant: sum
s_add_u32 s1, s1, s19 // (op 1408) most significant: sum


s_mov_b32 s64, s36 // (op 1408) reg expression
s_mov_b32 s65, s37 // (op 1408) reg expression
s_mov_b32 s66, s0 // (op 1408) reg expression
s_mov_b32 s67, 131072 // (op 1408) call()

s_mul_hi_i32 s0, s16, s11 // (op 1425) 
v_lshrrev_b32 v15, 3, v1 // (op 1425) 
s_add_i32 s1, s0, s16 // (op 1425) 

v_and_b32 v24, -8, v1 // (op 1425) 
s_ashr_i32 s0, s1, 31 // (op 1425) 
v_sub_u32 v25, v1, v24 // (op 1425) 

s_cmp_eq_i32 s11, 0 // (op 1425) 
s_mov_b32 s18, scc // (op 1425) 
s_cmp_eq_i32 s18, 0 // (op 1425) 
s_cselect_b32 s19, 0, -1 // (op 1425) 

s_and_b32 s18, s12, 31 // (op 1425) 
s_lshl_b32 s49, 1, s18 // (op 1425) 
s_add_i32 s50, s49, s19 // (op 1425) 


s_and_b32 s19, s0, s50 // (op 1425) 


s_add_i32 s0, s1, s19 // (op 1425) 


s_ashr_i32 s1, s0, s18 // (op 1425) 


s_lshl_b32 s0, s12, 24 // (op 1425) 
s_ashr_i32 s18, s0, 31 // (op 1425) 

s_xor_b32 s0, s1, s18 // (op 1425) 

s_sub_i32 s1, s0, s18 // (op 1425) 


s_mul_hi_i32 s0, s1, s13 // (op 1425) 
s_add_i32 s18, s0, s1 // (op 1425) 

s_ashr_i32 s0, s18, 31 // (op 1425) 
s_cmp_eq_i32 s13, 0 // (op 1425) 
s_mov_b32 s19, scc // (op 1425) 
s_cmp_eq_i32 s19, 0 // (op 1425) 
s_cselect_b32 s49, 0, -1 // (op 1425) 

s_and_b32 s19, s14, 31 // (op 1425) 
s_lshl_b32 s50, 1, s19 // (op 1425) 
s_add_i32 s51, s50, s49 // (op 1425) 


s_and_b32 s49, s0, s51 // (op 1425) 


s_add_i32 s0, s18, s49 // (op 1425) 


s_ashr_i32 s18, s0, s19 // (op 1425) 


s_lshl_b32 s0, s14, 24 // (op 1425) 
s_ashr_i32 s19, s0, 31 // (op 1425) 

s_xor_b32 s0, s18, s19 // (op 1425) 

s_sub_i32 s18, s0, s19 // (op 1425) 


s_mul_i32 s0, s1, s44 // (op 1425) 

s_sub_u32 s1, s16, s0 // (op 1425) 

s_mul_i32 s0, s18, s44 // (op 1425) 
s_add_u32 s0, s0, s1 // (op 1425) 


s_mul_hi_i32 s1, s16, s10 // (op 1425) 
s_add_i32 s18, s1, s16 // (op 1425) 

s_ashr_i32 s1, s18, 31 // (op 1425) 
s_cmp_eq_i32 s10, 0 // (op 1425) 
s_mov_b32 s19, scc // (op 1425) 
s_cmp_eq_i32 s19, 0 // (op 1425) 
s_cselect_b32 s49, 0, -1 // (op 1425) 

s_and_b32 s19, s47, 31 // (op 1425) 
s_lshl_b32 s50, 1, s19 // (op 1425) 
s_add_i32 s51, s50, s49 // (op 1425) 


s_and_b32 s49, s1, s51 // (op 1425) 


s_add_i32 s1, s18, s49 // (op 1425) 


s_ashr_i32 s18, s1, s19 // (op 1425) 


s_lshl_b32 s1, s47, 24 // (op 1425) 
s_ashr_i32 s19, s1, 31 // (op 1425) 

s_xor_b32 s1, s18, s19 // (op 1425) 

s_sub_i32 s18, s1, s19 // (op 1425) 


s_add_u32 s50, s6, 64 // (op 1425) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1425) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1425) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1425) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1425) 

s_mul_i32 s1, s44, s50 // (op 1425) 

s_mul_i32 s19, s48, s1 // (op 1425) 

s_mul_i32 s1, s18, s19 // (op 1425) 

s_sub_u32 s19, s16, s1 // (op 1425) 

s_mov_b32 s1, s19 // (op 1425) 
s_mul_hi_i32 s49, s1, s15 // (op 1425) 
s_add_i32 s50, s49, s1 // (op 1425) 


s_ashr_i32 s1, s50, 31 // (op 1425) 
s_cmp_eq_i32 s15, 0 // (op 1425) 
s_mov_b32 s49, scc // (op 1425) 
s_cmp_eq_i32 s49, 0 // (op 1425) 
s_cselect_b32 s51, 0, -1 // (op 1425) 

s_and_b32 s49, s46, 31 // (op 1425) 
s_lshl_b32 s68, 1, s49 // (op 1425) 
s_add_i32 s69, s68, s51 // (op 1425) 


s_and_b32 s51, s1, s69 // (op 1425) 


s_add_i32 s1, s50, s51 // (op 1425) 


s_ashr_i32 s50, s1, s49 // (op 1425) 


s_lshl_b32 s1, s46, 24 // (op 1425) 
s_ashr_i32 s49, s1, 31 // (op 1425) 

s_xor_b32 s1, s50, s49 // (op 1425) 

s_sub_i32 s50, s1, s49 // (op 1425) 


s_mul_i32 s1, s45, s44 // (op 1425) 
s_add_u32 s68, s8, 16 // (op 1425) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 1425) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1425) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1425) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 1425) 

s_sub_i32 s49, s68, s1 // (op 1425) 

s_mul_i32 s51, s50, s49 // (op 1425) 


s_sub_u32 s49, s19, s51 // (op 1425) 


s_add_u32 s19, s49, s1 // (op 1425) 


s_cmp_eq_i32 s18, 0 // (op 1425) 
s_mov_b32 s1, scc // (op 1425) 

s_cmp_eq_i32 s1, 0 // (op 1425) 
s_cselect_b32 s18, s19, s0 // (op 1425) 



v_lshl_add_u32 v24, s18, 4, v15 // (op 1425) 


v_mul_lo_u32 v15, v24, s34 // (op 1425) 
v_add_u32 v15, v15, v25 // (op 1425) 


s_mul_i32 s1, s34, s9 // (op 1427) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1427) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1427) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1427) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1427) most significant: sum
s_add_u32 s1, s1, s19 // (op 1427) most significant: sum


s_mov_b32 s64, s36 // (op 1427) reg expression
s_mov_b32 s65, s37 // (op 1427) reg expression
s_mov_b32 s66, s0 // (op 1427) reg expression
s_mov_b32 s67, 131072 // (op 1427) call()

s_mul_hi_i32 s0, s16, s11 // (op 1470) 
v_and_b32 v24, -16, v1 // (op 1470) 
s_add_i32 s1, s0, s16 // (op 1470) 

v_sub_u32 v25, v1, v24 // (op 1470) 

s_ashr_i32 s0, s1, 31 // (op 1470) 
v_lshlrev_b32 v24, 2, v25 // (op 1470) 

s_cmp_eq_i32 s11, 0 // (op 1470) 
s_mov_b32 s18, scc // (op 1470) 
v_lshrrev_b32 v25, 4, v1 // (op 1470) 
s_cmp_eq_i32 s18, 0 // (op 1470) 
s_cselect_b32 s19, 0, -1 // (op 1470) 

s_and_b32 s18, s12, 31 // (op 1470) 
s_lshl_b32 s49, 1, s18 // (op 1470) 
s_add_i32 s50, s49, s19 // (op 1470) 


s_and_b32 s19, s0, s50 // (op 1470) 


s_add_i32 s0, s1, s19 // (op 1470) 


s_ashr_i32 s1, s0, s18 // (op 1470) 


s_lshl_b32 s0, s12, 24 // (op 1470) 
s_ashr_i32 s18, s0, 31 // (op 1470) 

s_xor_b32 s0, s1, s18 // (op 1470) 

s_sub_i32 s1, s0, s18 // (op 1470) 


s_mul_hi_i32 s0, s1, s13 // (op 1470) 
s_add_i32 s18, s0, s1 // (op 1470) 

s_ashr_i32 s0, s18, 31 // (op 1470) 
s_cmp_eq_i32 s13, 0 // (op 1470) 
s_mov_b32 s19, scc // (op 1470) 
s_cmp_eq_i32 s19, 0 // (op 1470) 
s_cselect_b32 s49, 0, -1 // (op 1470) 

s_and_b32 s19, s14, 31 // (op 1470) 
s_lshl_b32 s50, 1, s19 // (op 1470) 
s_add_i32 s51, s50, s49 // (op 1470) 


s_and_b32 s49, s0, s51 // (op 1470) 


s_add_i32 s0, s18, s49 // (op 1470) 


s_ashr_i32 s18, s0, s19 // (op 1470) 


s_lshl_b32 s0, s14, 24 // (op 1470) 
s_ashr_i32 s19, s0, 31 // (op 1470) 

s_xor_b32 s0, s18, s19 // (op 1470) 

s_sub_i32 s18, s0, s19 // (op 1470) 


s_add_u32 s50, s6, 64 // (op 1470) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1470) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1470) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1470) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1470) 

s_mul_i32 s0, s18, s50 // (op 1470) 

s_sub_i32 s49, s1, s0 // (op 1470) 


s_mul_hi_i32 s0, s16, s10 // (op 1470) 
s_add_i32 s1, s0, s16 // (op 1470) 

s_ashr_i32 s0, s1, 31 // (op 1470) 
s_cmp_eq_i32 s10, 0 // (op 1470) 
s_mov_b32 s18, scc // (op 1470) 
s_cmp_eq_i32 s18, 0 // (op 1470) 
s_cselect_b32 s19, 0, -1 // (op 1470) 

s_and_b32 s18, s47, 31 // (op 1470) 
s_lshl_b32 s68, 1, s18 // (op 1470) 
s_add_i32 s69, s68, s19 // (op 1470) 


s_and_b32 s19, s0, s69 // (op 1470) 


s_add_i32 s0, s1, s19 // (op 1470) 


s_ashr_i32 s1, s0, s18 // (op 1470) 


s_lshl_b32 s0, s47, 24 // (op 1470) 
s_ashr_i32 s18, s0, 31 // (op 1470) 

s_xor_b32 s0, s1, s18 // (op 1470) 

s_sub_i32 s1, s0, s18 // (op 1470) 


s_mul_i32 s0, s44, s50 // (op 1470) 

s_mul_i32 s18, s48, s0 // (op 1470) 

s_mul_i32 s0, s1, s18 // (op 1470) 

s_sub_u32 s18, s16, s0 // (op 1470) 

s_mul_hi_i32 s0, s18, s15 // (op 1470) 
s_add_i32 s19, s0, s18 // (op 1470) 


s_ashr_i32 s0, s19, 31 // (op 1470) 
s_cmp_eq_i32 s15, 0 // (op 1470) 
s_mov_b32 s18, scc // (op 1470) 
s_cmp_eq_i32 s18, 0 // (op 1470) 
s_cselect_b32 s50, 0, -1 // (op 1470) 

s_and_b32 s18, s46, 31 // (op 1470) 
s_lshl_b32 s51, 1, s18 // (op 1470) 
s_add_i32 s68, s51, s50 // (op 1470) 


s_and_b32 s50, s0, s68 // (op 1470) 


s_add_i32 s0, s19, s50 // (op 1470) 


s_ashr_i32 s19, s0, s18 // (op 1470) 


s_lshl_b32 s0, s46, 24 // (op 1470) 
s_ashr_i32 s18, s0, 31 // (op 1470) 

s_xor_b32 s0, s19, s18 // (op 1470) 

s_sub_i32 s19, s0, s18 // (op 1470) 


s_cmp_eq_i32 s1, 0 // (op 1470) 
s_mov_b32 s0, scc // (op 1470) 

s_cmp_eq_i32 s0, 0 // (op 1470) 
s_cselect_b32 s1, s19, s49 // (op 1470) 



v_lshl_add_u32 v26, s1, 6, v24 // (op 1470) 


v_mul_lo_u32 v24, v25, s38 // (op 1470) 
v_add_u32 v24, v24, v26 // (op 1470) 


s_mul_i32 s1, s38, s43 // (op 1472) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1472) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1472) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1472) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1472) most significant: sum
s_add_u32 s1, s1, s19 // (op 1472) most significant: sum


s_mov_b32 s68, s40 // (op 1472) reg expression
s_mov_b32 s69, s41 // (op 1472) reg expression
s_mov_b32 s70, s0 // (op 1472) reg expression
s_mov_b32 s71, 131072 // (op 1472) call()

s_mul_hi_i32 s0, s16, s11 // (op 1489) 
v_and_b32 v25, -16, v1 // (op 1489) 
s_add_i32 s1, s0, s16 // (op 1489) 

v_sub_u32 v26, v1, v25 // (op 1489) 

s_ashr_i32 s0, s1, 31 // (op 1489) 
v_lshlrev_b32 v25, 2, v26 // (op 1489) 

s_cmp_eq_i32 s11, 0 // (op 1489) 
s_mov_b32 s18, scc // (op 1489) 
v_lshrrev_b32 v26, 4, v1 // (op 1489) 
s_cmp_eq_i32 s18, 0 // (op 1489) 
s_cselect_b32 s19, 0, -1 // (op 1489) 

s_and_b32 s18, s12, 31 // (op 1489) 
s_lshl_b32 s49, 1, s18 // (op 1489) 
s_add_i32 s50, s49, s19 // (op 1489) 


s_and_b32 s19, s0, s50 // (op 1489) 


s_add_i32 s0, s1, s19 // (op 1489) 


s_ashr_i32 s1, s0, s18 // (op 1489) 


s_lshl_b32 s0, s12, 24 // (op 1489) 
s_ashr_i32 s18, s0, 31 // (op 1489) 

s_xor_b32 s0, s1, s18 // (op 1489) 

s_sub_i32 s1, s0, s18 // (op 1489) 


s_mul_hi_i32 s0, s1, s13 // (op 1489) 
s_add_i32 s18, s0, s1 // (op 1489) 

s_ashr_i32 s0, s18, 31 // (op 1489) 
s_cmp_eq_i32 s13, 0 // (op 1489) 
s_mov_b32 s19, scc // (op 1489) 
s_cmp_eq_i32 s19, 0 // (op 1489) 
s_cselect_b32 s49, 0, -1 // (op 1489) 

s_and_b32 s19, s14, 31 // (op 1489) 
s_lshl_b32 s50, 1, s19 // (op 1489) 
s_add_i32 s51, s50, s49 // (op 1489) 


s_and_b32 s49, s0, s51 // (op 1489) 


s_add_i32 s0, s18, s49 // (op 1489) 


s_ashr_i32 s18, s0, s19 // (op 1489) 


s_lshl_b32 s0, s14, 24 // (op 1489) 
s_ashr_i32 s19, s0, 31 // (op 1489) 

s_xor_b32 s0, s18, s19 // (op 1489) 

s_sub_i32 s18, s0, s19 // (op 1489) 


s_add_u32 s50, s6, 64 // (op 1489) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1489) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1489) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1489) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1489) 

s_mul_i32 s0, s18, s50 // (op 1489) 

s_sub_i32 s49, s1, s0 // (op 1489) 


s_mul_hi_i32 s0, s16, s10 // (op 1489) 
s_add_i32 s1, s0, s16 // (op 1489) 

s_ashr_i32 s0, s1, 31 // (op 1489) 
s_cmp_eq_i32 s10, 0 // (op 1489) 
s_mov_b32 s18, scc // (op 1489) 
s_cmp_eq_i32 s18, 0 // (op 1489) 
s_cselect_b32 s19, 0, -1 // (op 1489) 

s_and_b32 s18, s47, 31 // (op 1489) 
s_lshl_b32 s72, 1, s18 // (op 1489) 
s_add_i32 s73, s72, s19 // (op 1489) 


s_and_b32 s19, s0, s73 // (op 1489) 


s_add_i32 s0, s1, s19 // (op 1489) 


s_ashr_i32 s1, s0, s18 // (op 1489) 


s_lshl_b32 s0, s47, 24 // (op 1489) 
s_ashr_i32 s18, s0, 31 // (op 1489) 

s_xor_b32 s0, s1, s18 // (op 1489) 

s_sub_i32 s1, s0, s18 // (op 1489) 


s_mul_i32 s0, s44, s50 // (op 1489) 

s_mul_i32 s18, s48, s0 // (op 1489) 

s_mul_i32 s0, s1, s18 // (op 1489) 

s_sub_u32 s18, s16, s0 // (op 1489) 

s_mul_hi_i32 s0, s18, s15 // (op 1489) 
s_add_i32 s19, s0, s18 // (op 1489) 


s_ashr_i32 s0, s19, 31 // (op 1489) 
s_cmp_eq_i32 s15, 0 // (op 1489) 
s_mov_b32 s18, scc // (op 1489) 
s_cmp_eq_i32 s18, 0 // (op 1489) 
s_cselect_b32 s50, 0, -1 // (op 1489) 

s_and_b32 s18, s46, 31 // (op 1489) 
s_lshl_b32 s51, 1, s18 // (op 1489) 
s_add_i32 s72, s51, s50 // (op 1489) 


s_and_b32 s50, s0, s72 // (op 1489) 


s_add_i32 s0, s19, s50 // (op 1489) 


s_ashr_i32 s19, s0, s18 // (op 1489) 


s_lshl_b32 s0, s46, 24 // (op 1489) 
s_ashr_i32 s18, s0, 31 // (op 1489) 

s_xor_b32 s0, s19, s18 // (op 1489) 

s_sub_i32 s19, s0, s18 // (op 1489) 


s_cmp_eq_i32 s1, 0 // (op 1489) 
s_mov_b32 s0, scc // (op 1489) 

s_cmp_eq_i32 s0, 0 // (op 1489) 
s_cselect_b32 s1, s19, s49 // (op 1489) 



v_lshl_add_u32 v27, s1, 6, v25 // (op 1489) 


v_mul_lo_u32 v25, v26, s38 // (op 1489) 
v_add_u32 v25, v25, v27 // (op 1489) 


s_mul_i32 s1, s38, s43 // (op 1491) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1491) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1491) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1491) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1491) most significant: sum
s_add_u32 s1, s1, s19 // (op 1491) most significant: sum


s_mov_b32 s68, s40 // (op 1491) reg expression
s_mov_b32 s69, s41 // (op 1491) reg expression
s_mov_b32 s70, s0 // (op 1491) reg expression
s_mov_b32 s71, 131072 // (op 1491) call()

v_and_b32 v26, -4, v1 // (op 1607) 
v_sub_u32 v27, v1, v26 // (op 1607) 

v_lshlrev_b32 v26, 4, v27 // (op 1607) 

v_lshrrev_b32 v27, 2, v1 // (op 1607) 
v_lshl_add_u32 v28, v27, 6, v26 // (op 1607) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1607) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 5 // (op 1226) 
v_mov_b64 v[26:27], s[18:19] // (op 1226) convert


v_mov_b64 v[28:29], v[16:17] // (op 1226) reg expression
s_mov_b32 m0, s0 // (op 1226) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 1226) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1226) most significant half

s_add_u32 m0, m0, 2048 // (op 1226) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 1226) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1226) most significant half

s_add_u32 m0, m0, 2048 // (op 1226) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 1226) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1226) most significant half

s_add_u32 m0, m0, 2048 // (op 1226) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 1226) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1226) most significant half

s_add_u32 m0, m0, 2048 // (op 1226) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 1226) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1226) most significant half

s_add_u32 m0, m0, 2048 // (op 1226) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 1226) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1226) most significant half

s_add_u32 m0, m0, 2048 // (op 1226) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 1226) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1226) most significant half

s_add_u32 m0, m0, 2048 // (op 1226) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1226) Load value direct to lds





v_mov_b32 v26, v24 // (op 1032) reg expression

buffer_load_dword v27, v26, s[68:71], 0 offen // (op 1032) Load value

v_and_b32 v26, -16, v1 // (op 1555) 
v_sub_u32 v28, v1, v26 // (op 1555) 

v_lshlrev_b32 v26, 4, v28 // (op 1555) 

v_lshrrev_b32 v28, 4, v1 // (op 1555) 
v_lshl_add_u32 v29, v28, 8, v26 // (op 1555) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 1555) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 3 // (op 1231) 
v_mov_b64 v[28:29], s[18:19] // (op 1231) convert


v_mov_b64 v[30:31], v[18:19] // (op 1231) 
s_add_u32 s1, 16384, s0 // (op 1231) 
s_mov_b32 m0, s1 // (op 1231) reg expression
buffer_load_dwordx4 v30, s[20:23], 0 offen offset: 0 lds // (op 1231) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 1231) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 1231) most significant half

s_add_u32 m0, m0, 2048 // (op 1231) 
buffer_load_dwordx4 v30, s[20:23], 0 offen offset: 0 lds // (op 1231) Load value direct to lds







buffer_load_ubyte v26, v14, s[64:67], 0 offen offset: 0 // (op 1038) Load value
v_and_b32 v28, -4, v1 // (op 1819) 
v_sub_u32 v29, v1, v28 // (op 1819) 

v_lshlrev_b32 v28, 4, v29 // (op 1819) 

v_lshrrev_b32 v29, 2, v1 // (op 1819) 
v_lshl_add_u32 v30, v29, 6, v28 // (op 1819) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 1819) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 5 // (op 1236) 
v_mov_b64 v[28:29], s[18:19] // (op 1236) convert


s_lshl_b64 s[18:19], s[28:29], 8 // (op 1236) 
v_mov_b32 v30, s18 // (op 1236) 
v_mov_b32 v31, s19 // (op 1236) 
v_add_co_u32 v32, s[50:51], v16, v30 // (op 1236) least significant half
v_addc_co_u32 v33, s[50:51], v17, v31, s[50:51] // (op 1236) most significant half




s_add_u32 s1, 20480, s0 // (op 1236) 
s_mov_b32 m0, s1 // (op 1236) reg expression
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 1236) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1236) most significant half

s_add_u32 m0, m0, 2048 // (op 1236) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 1236) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1236) most significant half

s_add_u32 m0, m0, 2048 // (op 1236) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 1236) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1236) most significant half

s_add_u32 m0, m0, 2048 // (op 1236) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 1236) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1236) most significant half

s_add_u32 m0, m0, 2048 // (op 1236) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 1236) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1236) most significant half

s_add_u32 m0, m0, 2048 // (op 1236) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 1236) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1236) most significant half

s_add_u32 m0, m0, 2048 // (op 1236) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 1236) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1236) most significant half

s_add_u32 m0, m0, 2048 // (op 1236) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 1236) Load value direct to lds







s_lshl_b32 s0, s38, 3 // (op 1044) 
v_add_u32 v16, v24, s0 // (op 1044) 


buffer_load_dword v17, v16, s[68:71], 0 offen // (op 1044) Load value


v_and_b32 v16, -16, v1 // (op 1780) 
v_sub_u32 v24, v1, v16 // (op 1780) 

v_lshlrev_b32 v16, 4, v24 // (op 1780) 

v_lshrrev_b32 v24, 4, v1 // (op 1780) 
v_lshl_add_u32 v28, v24, 8, v16 // (op 1780) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1780) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 3 // (op 1238) 
v_mov_b64 v[28:29], s[18:19] // (op 1238) convert


v_mov_b64 v[30:31], v[18:19] // (op 1238) 
s_add_u32 s1, 36864, s0 // (op 1238) 
s_mov_b32 m0, s1 // (op 1238) reg expression
s_mov_b32 s18, 256 // (op 1238) call()
buffer_load_dwordx4 v30, s[20:23], s18 offen offset: 0 lds // (op 1238) Load value direct to lds
v_add_co_u32 v30, s[50:51], v30, v28 // (op 1238) least significant half
v_addc_co_u32 v31, s[50:51], 0, v31, s[50:51] // (op 1238) most significant half

s_add_u32 m0, m0, 2048 // (op 1238) 
buffer_load_dwordx4 v30, s[20:23], s18 offen offset: 0 lds // (op 1238) Load value direct to lds









buffer_load_ubyte v16, v14, s[64:67], 0 offen offset:8 // (op 1050) Load value

v_and_b32 v14, -16, v1 // (op 1718) 
v_sub_u32 v24, v1, v14 // (op 1718) 

v_lshlrev_b32 v14, 2, v24 // (op 1718) 

v_lshrrev_b32 v24, 4, v1 // (op 1718) 
v_lshl_add_u32 v18, v24, 6, v14 // (op 1718) 





s_waitcnt vmcnt(15)
ds_write_b32 v18, v27 offset:40960 // (op 1053) Store local data 

v_and_b32 v14, -8, v1 // (op 1665) 
v_sub_u32 v24, v1, v14 // (op 1665) 

v_and_b32 v14, 4294967288, v1 // (op 1665) 
v_add_u32 v18, v14, v24 // (op 1665) 





s_waitcnt vmcnt(12)
ds_write_b8 v18, v26 offset:41472 // (op 1055) Store local data 

s_barrier  // (op 1979) 



ds_read_b64_tr_b8 v[28:29], v9 // (op 1076) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v9 offset:512 // (op 1076) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v9 offset:4096 // (op 1076) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v9 offset:4608 // (op 1076) Transpose load local data 



ds_read_b64_tr_b8 v[36:37], v9 offset:8192 // (op 1086) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v9 offset:8704 // (op 1086) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v9 offset:12288 // (op 1086) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v9 offset:12800 // (op 1086) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2014) 


ds_read_u8 v14, v6 offset:40960 // (op 1102) Load local data 


ds_read_u8 v24, v6 offset:41216 // (op 1112) Load local data 



ds_read_b64_tr_b8 v[44:45], v9 offset:32 // (op 1081) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v9 offset:544 // (op 1081) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v9 offset:4128 // (op 1081) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v9 offset:4640 // (op 1081) Transpose load local data 



ds_read_b64_tr_b8 v[52:53], v9 offset:8224 // (op 1091) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v9 offset:8736 // (op 1091) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v9 offset:12320 // (op 1091) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v9 offset:12832 // (op 1091) Transpose load local data 


ds_read_u8 v18, v6 offset:40992 // (op 1107) Load local data 


ds_read_u8 v19, v6 offset:41248 // (op 1117) Load local data 
s_barrier  // (op 1963) 



ds_read_b128 v[60:63], v3 offset:16384 // (op 1069) Load local data 
ds_read_b128 v[64:67], v3 offset:16448 // (op 1069) Load local data 



ds_read_b128 v[68:71], v3 offset:16512 // (op 1072) Load local data 
ds_read_b128 v[72:75], v3 offset:16576 // (op 1072) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 1996) 



ds_read_u8 v76, v0 offset:41472 // (op 1095) Load local data 



ds_read_u8 v77, v0 offset:41476 // (op 1098) Load local data 
v_and_b32 v78, -16, v1 // (op 1527) 
v_sub_u32 v79, v1, v78 // (op 1527) 

v_lshlrev_b32 v78, 4, v79 // (op 1527) 

v_lshrrev_b32 v79, 4, v1 // (op 1527) 
v_lshl_add_u32 v80, v79, 8, v78 // (op 1527) 


s_nop 0
v_readfirstlane_b32 s0, v80 // (op 1527) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v78, -4, v1 // (op 1579) 
v_sub_u32 v79, v1, v78 // (op 1579) 

v_lshlrev_b32 v78, 4, v79 // (op 1579) 

v_lshrrev_b32 v79, 2, v1 // (op 1579) 
v_lshl_add_u32 v80, v79, 6, v78 // (op 1579) 


s_nop 0
v_readfirstlane_b32 s1, v80 // (op 1579) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v78, -8, v1 // (op 1636) 
v_sub_u32 v79, v1, v78 // (op 1636) 

v_and_b32 v78, 4294967288, v1 // (op 1636) 
v_add_u32 v80, v78, v79 // (op 1636) 


v_and_b32 v78, -16, v1 // (op 1689) 
v_sub_u32 v79, v1, v78 // (op 1689) 

v_lshlrev_b32 v78, 2, v79 // (op 1689) 

v_lshrrev_b32 v79, 4, v1 // (op 1689) 
v_lshl_add_u32 v81, v79, 6, v78 // (op 1689) 


v_and_b32 v78, -16, v1 // (op 1765) 
v_sub_u32 v79, v1, v78 // (op 1765) 

v_lshlrev_b32 v78, 4, v79 // (op 1765) 

v_lshrrev_b32 v79, 4, v1 // (op 1765) 
v_lshl_add_u32 v82, v79, 8, v78 // (op 1765) 


s_nop 0
v_readfirstlane_b32 s18, v82 // (op 1765) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v78, -4, v1 // (op 1804) 
v_sub_u32 v79, v1, v78 // (op 1804) 

v_lshlrev_b32 v78, 4, v79 // (op 1804) 

v_lshrrev_b32 v79, 2, v1 // (op 1804) 
v_lshl_add_u32 v82, v79, 6, v78 // (op 1804) 


s_nop 0
v_readfirstlane_b32 s19, v82 // (op 1804) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v78, -8, v1 // (op 1848) 
v_sub_u32 v79, v1, v78 // (op 1848) 

v_and_b32 v78, 4294967288, v1 // (op 1848) 
v_add_u32 v82, v78, v79 // (op 1848) 


v_and_b32 v78, -16, v1 // (op 1872) 
v_sub_u32 v79, v1, v78 // (op 1872) 

v_lshlrev_b32 v78, 2, v79 // (op 1872) 

v_lshrrev_b32 v79, 4, v1 // (op 1872) 
v_lshl_add_u32 v83, v79, 6, v78 // (op 1872) 


s_mov_b32 s49, 0 // (op 400) call()
s_ashr_i64 s[50:51], s[32:33], 8 // (op 399) 
s_mov_b32 s72, s50 // (op 399) convert

s_and_b32 s50, s72, 4294967294 // (op 399) 

s_cmp_lt_u32 s49, s50 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_399 // (op 399) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_399 if false)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_399:
s_add_u32 s49, s49, 2 // (op 401) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:67], v[28:35], a[0:3], v76, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 88) 
s_barrier  // (op 1982) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[60:67], v[44:51], a[4:7], v76, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 447) 
s_lshl_b64 s[50:51], s[28:29], 5 // (op 1198) 
v_mov_b64 v[78:79], s[50:51] // (op 1198) convert


v_mov_b64 v[84:85], v[22:23] // (op 1198) 
s_lshl_b64 s[50:51], s[28:29], 8 // (op 1198) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 1198) 

v_mov_b32 v86, s72 // (op 1198) 
v_mov_b32 v87, s73 // (op 1198) 
v_add_co_u32 v88, s[50:51], v84, v86 // (op 1198) least significant half
v_addc_co_u32 v89, s[50:51], v85, v87, s[50:51] // (op 1198) most significant half




s_mov_b32 m0, s1 // (op 1198) reg expression
buffer_load_dwordx4 v88, s[52:55], 0 offen offset: 0 lds // (op 1198) Load value direct to lds
v_add_co_u32 v88, s[50:51], v88, v78 // (op 1198) least significant half
v_addc_co_u32 v89, s[50:51], 0, v89, s[50:51] // (op 1198) most significant half

s_add_u32 m0, m0, 2048 // (op 1198) 
buffer_load_dwordx4 v88, s[52:55], 0 offen offset: 0 lds // (op 1198) Load value direct to lds
v_add_co_u32 v88, s[50:51], v88, v78 // (op 1198) least significant half
v_addc_co_u32 v89, s[50:51], 0, v89, s[50:51] // (op 1198) most significant half

s_add_u32 m0, m0, 2048 // (op 1198) 
buffer_load_dwordx4 v88, s[52:55], 0 offen offset: 0 lds // (op 1198) Load value direct to lds
v_add_co_u32 v88, s[50:51], v88, v78 // (op 1198) least significant half
v_addc_co_u32 v89, s[50:51], 0, v89, s[50:51] // (op 1198) most significant half

s_add_u32 m0, m0, 2048 // (op 1198) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:75], v[36:43], a[0:3], v77, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 105) 
buffer_load_dwordx4 v88, s[52:55], 0 offen offset: 0 lds // (op 1198) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:75], v[52:59], a[4:7], v77, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 456) 
v_add_co_u32 v88, s[50:51], v88, v78 // (op 1198) least significant half
v_addc_co_u32 v89, s[50:51], 0, v89, s[50:51] // (op 1198) most significant half

s_add_u32 m0, m0, 2048 // (op 1198) 
buffer_load_dwordx4 v88, s[52:55], 0 offen offset: 0 lds // (op 1198) Load value direct to lds
v_add_co_u32 v88, s[50:51], v88, v78 // (op 1198) least significant half
v_addc_co_u32 v89, s[50:51], 0, v89, s[50:51] // (op 1198) most significant half

s_add_u32 m0, m0, 2048 // (op 1198) 
buffer_load_dwordx4 v88, s[52:55], 0 offen offset: 0 lds // (op 1198) Load value direct to lds
v_add_co_u32 v88, s[50:51], v88, v78 // (op 1198) least significant half
v_addc_co_u32 v89, s[50:51], 0, v89, s[50:51] // (op 1198) most significant half

s_add_u32 m0, m0, 2048 // (op 1198) 
buffer_load_dwordx4 v88, s[52:55], 0 offen offset: 0 lds // (op 1198) Load value direct to lds
v_add_co_u32 v88, s[50:51], v88, v78 // (op 1198) least significant half
v_addc_co_u32 v89, s[50:51], 0, v89, s[50:51] // (op 1198) most significant half

s_add_u32 m0, m0, 2048 // (op 1198) 
buffer_load_dwordx4 v88, s[52:55], 0 offen offset: 0 lds // (op 1198) Load value direct to lds




v_mov_b32 v78, v25 // (op 25) 
s_lshl_b32 s50, s38, 4 // (op 25) 
v_add_u32 v79, v78, s50 // (op 25) 


buffer_load_dword v27, v79, s[68:71], 0 offen // (op 25) Load value

s_barrier  // (op 1966) 

s_lshl_b64 s[50:51], s[24:25], 3 // (op 1193) 
v_mov_b64 v[86:87], s[50:51] // (op 1193) convert


v_mov_b64 v[88:89], v[20:21] // (op 1193) 
v_mov_b64 v[90:91], v[88:89] // (op 1193) 
s_add_u32 s50, 16384, s0 // (op 1193) 
s_mov_b32 m0, s50 // (op 1193) reg expression
s_mov_b32 s51, 512 // (op 1193) call()
buffer_load_dwordx4 v90, s[20:23], s51 offen offset: 0 lds // (op 1193) Load value direct to lds
v_add_co_u32 v90, s[72:73], v90, v86 // (op 1193) least significant half
v_addc_co_u32 v91, s[72:73], 0, v91, s[72:73] // (op 1193) most significant half

s_add_u32 m0, m0, 2048 // (op 1193) 
buffer_load_dwordx4 v90, s[20:23], s51 offen offset: 0 lds // (op 1193) Load value direct to lds






v_mov_b32 v79, v15 // (op 17) 

buffer_load_ubyte v26, v79, s[64:67], 0 offen offset:16 // (op 17) Load value
s_barrier  // (op 2054) 



s_waitcnt vmcnt(15)
ds_write_b32 v83, v17 offset:41600 // (op 304) Store local data 
s_barrier  // (op 2048) 



s_waitcnt vmcnt(12)
ds_write_b8 v82, v16 offset:42112 // (op 303) Store local data 
s_barrier  // (op 2037) 



ds_read_b64_tr_b8 v[28:29], v13 offset:20480 // (op 287) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v13 offset:20992 // (op 287) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v13 offset:24576 // (op 287) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v13 offset:25088 // (op 287) Transpose load local data 



ds_read_b64_tr_b8 v[90:91], v13 offset:28672 // (op 296) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v13 offset:29184 // (op 296) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v13 offset:32768 // (op 296) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v13 offset:33280 // (op 296) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2051) 


ds_read_u8 v14, v12 offset:41600 // (op 291) Load local data 


ds_read_u8 v86, v12 offset:41856 // (op 300) Load local data 



ds_read_b64_tr_b8 v[44:45], v13 offset:20512 // (op 470) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v13 offset:21024 // (op 470) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v13 offset:24608 // (op 470) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v13 offset:25120 // (op 470) Transpose load local data 



ds_read_b64_tr_b8 v[98:99], v13 offset:28704 // (op 461) Transpose load local data 
ds_read_b64_tr_b8 v[100:101], v13 offset:29216 // (op 461) Transpose load local data 
ds_read_b64_tr_b8 v[102:103], v13 offset:32800 // (op 461) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v13 offset:33312 // (op 461) Transpose load local data 


ds_read_u8 v18, v12 offset:41632 // (op 474) Load local data 


ds_read_u8 v87, v12 offset:41888 // (op 465) Load local data 
s_barrier  // (op 2031) 
s_barrier  // (op 2121) 



ds_read_b128 v[60:63], v5 offset:36864 // (op 285) Load local data 
ds_read_b128 v[64:67], v5 offset:36928 // (op 285) Load local data 



ds_read_b128 v[106:109], v5 offset:36992 // (op 294) Load local data 
ds_read_b128 v[110:113], v5 offset:37056 // (op 294) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 2044) 



ds_read_u8 v76, v4 offset:42112 // (op 289) Load local data 



ds_read_u8 v77, v4 offset:42116 // (op 298) Load local data 
s_barrier  // (op 2085) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:67], v[28:35], a[0:3], v76, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 292) 
s_barrier  // (op 2041) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[60:67], v[44:51], a[4:7], v76, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 457) 
s_lshl_b64 s[50:51], s[28:29], 5 // (op 1221) 
v_mov_b64 v[114:115], s[50:51] // (op 1221) convert


s_lshl_b64 s[50:51], s[28:29], 8 // (op 1221) 
s_mul_i32 s73, 3, s51 // (op 1221) most significant: low of low * high
s_mul_i32 s72, 3, s50 // (op 1221) least significant
s_mul_hi_u32 s74, 3, s50 // (op 1221) most significant: high of low * low
s_add_u32 s73, s73, s74 // (op 1221) most significant: sum


v_mov_b32 v116, s72 // (op 1221) 
v_mov_b32 v117, s73 // (op 1221) 
v_add_co_u32 v118, s[50:51], v84, v116 // (op 1221) least significant half
v_addc_co_u32 v119, s[50:51], v85, v117, s[50:51] // (op 1221) most significant half




s_add_u32 s50, 20480, s19 // (op 1221) 
s_nop 0
s_mov_b32 m0, s50 // (op 1221) reg expression
buffer_load_dwordx4 v118, s[52:55], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v118, s[72:73], v118, v114 // (op 1221) least significant half
v_addc_co_u32 v119, s[72:73], 0, v119, s[72:73] // (op 1221) most significant half

s_add_u32 m0, m0, 2048 // (op 1221) 
buffer_load_dwordx4 v118, s[52:55], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v118, s[72:73], v118, v114 // (op 1221) least significant half
v_addc_co_u32 v119, s[72:73], 0, v119, s[72:73] // (op 1221) most significant half

s_add_u32 m0, m0, 2048 // (op 1221) 
buffer_load_dwordx4 v118, s[52:55], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v118, s[72:73], v118, v114 // (op 1221) least significant half
v_addc_co_u32 v119, s[72:73], 0, v119, s[72:73] // (op 1221) most significant half

s_add_u32 m0, m0, 2048 // (op 1221) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[106:113], v[90:97], a[0:3], v77, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 301) 
buffer_load_dwordx4 v118, s[52:55], 0 offen offset: 0 lds // (op 1221) Load value direct to lds


v_add_co_u32 v118, s[72:73], v118, v114 // (op 1221) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[106:113], v[98:105], a[4:7], v77, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 466) 
v_addc_co_u32 v119, s[72:73], 0, v119, s[72:73] // (op 1221) most significant half




s_add_u32 m0, m0, 2048 // (op 1221) 
buffer_load_dwordx4 v118, s[52:55], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v118, s[72:73], v118, v114 // (op 1221) least significant half
v_addc_co_u32 v119, s[72:73], 0, v119, s[72:73] // (op 1221) most significant half

s_add_u32 m0, m0, 2048 // (op 1221) 
buffer_load_dwordx4 v118, s[52:55], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v118, s[72:73], v118, v114 // (op 1221) least significant half
v_addc_co_u32 v119, s[72:73], 0, v119, s[72:73] // (op 1221) most significant half

s_add_u32 m0, m0, 2048 // (op 1221) 
buffer_load_dwordx4 v118, s[52:55], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v118, s[72:73], v118, v114 // (op 1221) least significant half
v_addc_co_u32 v119, s[72:73], 0, v119, s[72:73] // (op 1221) most significant half

s_add_u32 m0, m0, 2048 // (op 1221) 
buffer_load_dwordx4 v118, s[52:55], 0 offen offset: 0 lds // (op 1221) Load value direct to lds






s_lshl_b32 s50, s38, 3 // (op 307) 
s_mul_i32 s51, 3, s50 // (op 307) 

v_add_u32 v84, v78, s51 // (op 307) 


buffer_load_dword v17, v84, s[68:71], 0 offen // (op 307) Load value

s_lshl_b64 s[50:51], s[28:29], 8 // (op 1353) 

s_lshl_b64 s[72:73], s[50:51], 1 // (op 1353) 

s_barrier  // (op 2034) 
s_lshl_b32 s50, s38, 4 // (op 1481) 

v_add_u32 v25, v25, s50 // (op 1481) 

v_mov_b32 v78, s72 // (op 1353) 
v_mov_b32 v84, s73 // (op 1353) 
s_lshl_b64 s[50:51], s[24:25], 3 // (op 1216) 
v_mov_b64 v[86:87], s[50:51] // (op 1216) convert


v_mov_b64 v[90:91], v[88:89] // (op 1216) 
s_add_u32 s50, 36864, s18 // (op 1216) 
s_mov_b32 m0, s50 // (op 1216) reg expression
s_mov_b32 s51, 768 // (op 1216) call()
v_add_co_u32 v22, s[74:75], v22, v78 // (op 1353) least significant half
v_addc_co_u32 v23, s[74:75], v23, v84, s[74:75] // (op 1353) most significant half




buffer_load_dwordx4 v90, s[20:23], s51 offen offset: 0 lds // (op 1216) Load value direct to lds
v_add_co_u32 v90, s[72:73], v90, v86 // (op 1216) least significant half
v_addc_co_u32 v91, s[72:73], 0, v91, s[72:73] // (op 1216) most significant half

s_add_u32 m0, m0, 2048 // (op 1216) 
buffer_load_dwordx4 v90, s[20:23], s51 offen offset: 0 lds // (op 1216) Load value direct to lds








v_add_co_u32 v20, vcc, 512, v20 // (op 1289) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 1289) most significant half
buffer_load_ubyte v16, v79, s[64:67], 0 offen offset:24 // (op 306) Load value

s_barrier  // (op 2017) 
v_add_u32 v15, 16, v15 // (op 1417) 



s_waitcnt vmcnt(15)
ds_write_b32 v81, v27 offset:40960 // (op 30) Store local data 
s_barrier  // (op 1999) 



s_waitcnt vmcnt(12)
ds_write_b8 v80, v26 offset:41472 // (op 21) Store local data 
s_barrier  // (op 1988) 



ds_read_b64_tr_b8 v[28:29], v9 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v9 offset:512 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v9 offset:4096 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v9 offset:4608 // (op 75) Transpose load local data 



ds_read_b64_tr_b8 v[36:37], v9 offset:8192 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v9 offset:8704 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v9 offset:12288 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v9 offset:12800 // (op 95) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2024) 


ds_read_u8 v14, v6 offset:40960 // (op 85) Load local data 


ds_read_u8 v24, v6 offset:41216 // (op 103) Load local data 



ds_read_b64_tr_b8 v[44:45], v9 offset:32 // (op 442) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v9 offset:544 // (op 442) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v9 offset:4128 // (op 442) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v9 offset:4640 // (op 442) Transpose load local data 



ds_read_b64_tr_b8 v[52:53], v9 offset:8224 // (op 451) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v9 offset:8736 // (op 451) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v9 offset:12320 // (op 451) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v9 offset:12832 // (op 451) Transpose load local data 


ds_read_u8 v18, v6 offset:40992 // (op 446) Load local data 


ds_read_u8 v19, v6 offset:41248 // (op 455) Load local data 
s_barrier  // (op 1972) 



ds_read_b128 v[60:63], v3 offset:16384 // (op 70) Load local data 
ds_read_b128 v[64:67], v3 offset:16448 // (op 70) Load local data 



ds_read_b128 v[68:71], v3 offset:16512 // (op 91) Load local data 
ds_read_b128 v[72:75], v3 offset:16576 // (op 91) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 2006) 



ds_read_u8 v76, v0 offset:41472 // (op 80) Load local data 



ds_read_u8 v77, v0 offset:41476 // (op 99) Load local data 
s_ashr_i64 s[50:51], s[32:33], 8 // (op 399) 
s_mov_b32 s72, s50 // (op 399) convert

s_and_b32 s50, s72, 4294967294 // (op 399) 

s_cmp_lt_u32 s49, s50 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_399 // (op 399) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_399 if true)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_399:



















s_barrier  // (op 2103) 
s_mul_hi_i32 s0, s16, s11 // (op 1251) 
v_lshrrev_b32 v14, 4, v1 // (op 1251) 
s_add_i32 s1, s0, s16 // (op 1251) 

v_and_b32 v15, -16, v1 // (op 1251) 
s_ashr_i32 s0, s1, 31 // (op 1251) 
v_sub_u32 v20, v1, v15 // (op 1251) 

s_cmp_eq_i32 s11, 0 // (op 1251) 
s_mov_b32 s18, scc // (op 1251) 
v_lshlrev_b32 v15, 4, v20 // (op 1251) 

s_cmp_eq_i32 s18, 0 // (op 1251) 
s_cselect_b32 s19, 0, -1 // (op 1251) 

s_and_b32 s18, s12, 31 // (op 1251) 
s_lshl_b32 s49, 1, s18 // (op 1251) 
s_add_i32 s50, s49, s19 // (op 1251) 


s_and_b32 s19, s0, s50 // (op 1251) 


s_add_i32 s0, s1, s19 // (op 1251) 


s_ashr_i32 s1, s0, s18 // (op 1251) 


s_lshl_b32 s0, s12, 24 // (op 1251) 
s_ashr_i32 s18, s0, 31 // (op 1251) 

s_xor_b32 s0, s1, s18 // (op 1251) 

s_sub_i32 s1, s0, s18 // (op 1251) 


s_mul_hi_i32 s0, s1, s13 // (op 1251) 
s_add_i32 s18, s0, s1 // (op 1251) 

s_ashr_i32 s0, s18, 31 // (op 1251) 
s_cmp_eq_i32 s13, 0 // (op 1251) 
s_mov_b32 s19, scc // (op 1251) 
s_cmp_eq_i32 s19, 0 // (op 1251) 
s_cselect_b32 s49, 0, -1 // (op 1251) 

s_and_b32 s19, s14, 31 // (op 1251) 
s_lshl_b32 s50, 1, s19 // (op 1251) 
s_add_i32 s51, s50, s49 // (op 1251) 


s_and_b32 s49, s0, s51 // (op 1251) 


s_add_i32 s0, s18, s49 // (op 1251) 


s_ashr_i32 s18, s0, s19 // (op 1251) 


s_lshl_b32 s0, s14, 24 // (op 1251) 
s_ashr_i32 s19, s0, 31 // (op 1251) 

s_xor_b32 s0, s18, s19 // (op 1251) 

s_sub_i32 s18, s0, s19 // (op 1251) 


s_mul_i32 s0, s1, s44 // (op 1251) 

s_sub_u32 s1, s16, s0 // (op 1251) 

s_mul_i32 s0, s18, s44 // (op 1251) 
s_add_u32 s0, s0, s1 // (op 1251) 


s_mul_hi_i32 s1, s16, s10 // (op 1251) 
s_add_i32 s18, s1, s16 // (op 1251) 

s_ashr_i32 s1, s18, 31 // (op 1251) 
s_cmp_eq_i32 s10, 0 // (op 1251) 
s_mov_b32 s19, scc // (op 1251) 
s_cmp_eq_i32 s19, 0 // (op 1251) 
s_cselect_b32 s49, 0, -1 // (op 1251) 

s_and_b32 s19, s47, 31 // (op 1251) 
s_lshl_b32 s50, 1, s19 // (op 1251) 
s_add_i32 s51, s50, s49 // (op 1251) 


s_and_b32 s49, s1, s51 // (op 1251) 


s_add_i32 s1, s18, s49 // (op 1251) 


s_ashr_i32 s18, s1, s19 // (op 1251) 


s_lshl_b32 s1, s47, 24 // (op 1251) 
s_ashr_i32 s19, s1, 31 // (op 1251) 

s_xor_b32 s1, s18, s19 // (op 1251) 

s_sub_i32 s18, s1, s19 // (op 1251) 


s_add_u32 s50, s6, 64 // (op 1251) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1251) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1251) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1251) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1251) 

s_mul_i32 s1, s44, s50 // (op 1251) 

s_mul_i32 s19, s48, s1 // (op 1251) 

s_mul_i32 s1, s18, s19 // (op 1251) 

s_sub_u32 s19, s16, s1 // (op 1251) 

s_mov_b32 s1, s19 // (op 1251) 
s_mul_hi_i32 s49, s1, s15 // (op 1251) 
s_add_i32 s50, s49, s1 // (op 1251) 


s_ashr_i32 s1, s50, 31 // (op 1251) 
s_cmp_eq_i32 s15, 0 // (op 1251) 
s_mov_b32 s49, scc // (op 1251) 
s_cmp_eq_i32 s49, 0 // (op 1251) 
s_cselect_b32 s51, 0, -1 // (op 1251) 

s_and_b32 s49, s46, 31 // (op 1251) 
s_lshl_b32 s72, 1, s49 // (op 1251) 
s_add_i32 s73, s72, s51 // (op 1251) 


s_and_b32 s51, s1, s73 // (op 1251) 


s_add_i32 s1, s50, s51 // (op 1251) 


s_ashr_i32 s50, s1, s49 // (op 1251) 


s_lshl_b32 s1, s46, 24 // (op 1251) 
s_ashr_i32 s49, s1, 31 // (op 1251) 

s_xor_b32 s1, s50, s49 // (op 1251) 

s_sub_i32 s50, s1, s49 // (op 1251) 


s_mul_i32 s1, s45, s44 // (op 1251) 
s_add_u32 s72, s8, 16 // (op 1251) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 1251) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1251) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1251) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 1251) 

s_sub_i32 s49, s72, s1 // (op 1251) 

s_mul_i32 s51, s50, s49 // (op 1251) 


s_sub_u32 s49, s19, s51 // (op 1251) 


s_add_u32 s19, s49, s1 // (op 1251) 


s_cmp_eq_i32 s18, 0 // (op 1251) 
s_mov_b32 s1, scc // (op 1251) 

s_cmp_eq_i32 s1, 0 // (op 1251) 
s_cselect_b32 s18, s19, s0 // (op 1251) 



v_lshl_add_u32 v20, s18, 4, v14 // (op 1251) 


s_ashr_i64 s[0:1], s[32:33], 8 // (op 1251) 
s_lshr_b32 s18, s0, 1 // (op 1251) 

v_lshl_add_u32 v14, s18, 9, v15 // (op 1251) 


v_mov_b32 v43, 0 // (op 1251) convert
v_mov_b32 v42, v14 // (op 1251) convert

v_mul_lo_u32 v15, v20, s25 // (op 1251) most significant: low of low * high
v_mul_hi_u32 v21, v20, s24 // (op 1251) most significant: high of low * low
v_mul_lo_u32 v14, v20, s24 // (op 1251) least significant: low of low * low
v_add_u32 v15, v15, v21 // (op 1251) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v42 // (op 1251) least significant half
v_addc_co_u32 v15, s[0:1], v15, v43, s[0:1] // (op 1251) most significant half



s_mul_i32 s1, s24, s9 // (op 1253) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 1253) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1253) least significant
s_mul_hi_u32 s19, s24, s8 // (op 1253) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1253) most significant: sum
s_add_u32 s1, s1, s19 // (op 1253) most significant: sum


s_mov_b32 s20, s26 // (op 1253) reg expression
s_mov_b32 s21, s27 // (op 1253) reg expression
s_mov_b32 s22, s0 // (op 1253) reg expression
s_mov_b32 s23, 131072 // (op 1253) call()

s_mul_hi_i32 s0, s16, s11 // (op 1319) 
v_and_b32 v20, -4, v1 // (op 1319) 
s_add_i32 s1, s0, s16 // (op 1319) 

v_sub_u32 v21, v1, v20 // (op 1319) 

s_ashr_i32 s0, s1, 31 // (op 1319) 
v_lshlrev_b32 v20, 4, v21 // (op 1319) 

s_cmp_eq_i32 s11, 0 // (op 1319) 
s_mov_b32 s18, scc // (op 1319) 
v_lshrrev_b32 v21, 2, v1 // (op 1319) 
s_cmp_eq_i32 s18, 0 // (op 1319) 
s_cselect_b32 s19, 0, -1 // (op 1319) 

s_and_b32 s18, s12, 31 // (op 1319) 
s_lshl_b32 s49, 1, s18 // (op 1319) 
s_add_i32 s50, s49, s19 // (op 1319) 


s_and_b32 s19, s0, s50 // (op 1319) 


s_add_i32 s0, s1, s19 // (op 1319) 


s_ashr_i32 s1, s0, s18 // (op 1319) 


s_lshl_b32 s0, s12, 24 // (op 1319) 
s_ashr_i32 s18, s0, 31 // (op 1319) 

s_xor_b32 s0, s1, s18 // (op 1319) 

s_sub_i32 s1, s0, s18 // (op 1319) 


s_mul_hi_i32 s0, s1, s13 // (op 1319) 
s_add_i32 s18, s0, s1 // (op 1319) 

s_ashr_i32 s0, s18, 31 // (op 1319) 
s_cmp_eq_i32 s13, 0 // (op 1319) 
s_mov_b32 s19, scc // (op 1319) 
s_cmp_eq_i32 s19, 0 // (op 1319) 
s_cselect_b32 s49, 0, -1 // (op 1319) 

s_and_b32 s19, s14, 31 // (op 1319) 
s_lshl_b32 s50, 1, s19 // (op 1319) 
s_add_i32 s51, s50, s49 // (op 1319) 


s_and_b32 s49, s0, s51 // (op 1319) 


s_add_i32 s0, s18, s49 // (op 1319) 


s_ashr_i32 s18, s0, s19 // (op 1319) 


s_lshl_b32 s0, s14, 24 // (op 1319) 
s_ashr_i32 s19, s0, 31 // (op 1319) 

s_xor_b32 s0, s18, s19 // (op 1319) 

s_sub_i32 s18, s0, s19 // (op 1319) 


s_add_u32 s50, s6, 64 // (op 1319) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1319) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1319) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1319) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1319) 

s_mul_i32 s0, s18, s50 // (op 1319) 

s_sub_i32 s49, s1, s0 // (op 1319) 


s_mul_hi_i32 s0, s16, s10 // (op 1319) 
s_add_i32 s1, s0, s16 // (op 1319) 

s_ashr_i32 s0, s1, 31 // (op 1319) 
s_cmp_eq_i32 s10, 0 // (op 1319) 
s_mov_b32 s18, scc // (op 1319) 
s_cmp_eq_i32 s18, 0 // (op 1319) 
s_cselect_b32 s19, 0, -1 // (op 1319) 

s_and_b32 s18, s47, 31 // (op 1319) 
s_lshl_b32 s72, 1, s18 // (op 1319) 
s_add_i32 s73, s72, s19 // (op 1319) 


s_and_b32 s19, s0, s73 // (op 1319) 


s_add_i32 s0, s1, s19 // (op 1319) 


s_ashr_i32 s1, s0, s18 // (op 1319) 


s_lshl_b32 s0, s47, 24 // (op 1319) 
s_ashr_i32 s18, s0, 31 // (op 1319) 

s_xor_b32 s0, s1, s18 // (op 1319) 

s_sub_i32 s1, s0, s18 // (op 1319) 


s_mul_i32 s0, s44, s50 // (op 1319) 

s_mul_i32 s18, s48, s0 // (op 1319) 

s_mul_i32 s0, s1, s18 // (op 1319) 

s_sub_u32 s18, s16, s0 // (op 1319) 

s_mul_hi_i32 s0, s18, s15 // (op 1319) 
s_add_i32 s19, s0, s18 // (op 1319) 


s_ashr_i32 s0, s19, 31 // (op 1319) 
s_cmp_eq_i32 s15, 0 // (op 1319) 
s_mov_b32 s18, scc // (op 1319) 
s_cmp_eq_i32 s18, 0 // (op 1319) 
s_cselect_b32 s50, 0, -1 // (op 1319) 

s_and_b32 s18, s46, 31 // (op 1319) 
s_lshl_b32 s51, 1, s18 // (op 1319) 
s_add_i32 s72, s51, s50 // (op 1319) 


s_and_b32 s50, s0, s72 // (op 1319) 


s_add_i32 s0, s19, s50 // (op 1319) 


s_ashr_i32 s19, s0, s18 // (op 1319) 


s_lshl_b32 s0, s46, 24 // (op 1319) 
s_ashr_i32 s18, s0, 31 // (op 1319) 

s_xor_b32 s0, s19, s18 // (op 1319) 

s_sub_i32 s19, s0, s18 // (op 1319) 


s_cmp_eq_i32 s1, 0 // (op 1319) 
s_mov_b32 s0, scc // (op 1319) 

s_cmp_eq_i32 s0, 0 // (op 1319) 
s_cselect_b32 s1, s19, s49 // (op 1319) 



v_lshl_add_u32 v22, s1, 6, v20 // (op 1319) 


s_ashr_i64 s[0:1], s[32:33], 8 // (op 1319) 
v_mov_b32 v43, 0 // (op 1319) convert
v_mov_b32 v42, v22 // (op 1319) convert

s_lshr_b32 s18, s0, 1 // (op 1319) 

v_lshl_add_u32 v20, s18, 9, v21 // (op 1319) 


v_mul_lo_u32 v41, v20, s29 // (op 1319) most significant: low of low * high
v_mul_hi_u32 v21, v20, s28 // (op 1319) most significant: high of low * low
v_mul_lo_u32 v40, v20, s28 // (op 1319) least significant: low of low * low
v_add_u32 v41, v41, v21 // (op 1319) most significant: sum

v_add_co_u32 v40, s[0:1], v40, v42 // (op 1319) least significant half
v_addc_co_u32 v41, s[0:1], v41, v43, s[0:1] // (op 1319) most significant half



s_mul_i32 s1, s28, s33 // (op 1321) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 1321) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1321) least significant
s_mul_hi_u32 s19, s28, s32 // (op 1321) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1321) most significant: sum
s_add_u32 s1, s1, s19 // (op 1321) most significant: sum


s_mov_b32 s52, s30 // (op 1321) reg expression
s_mov_b32 s53, s31 // (op 1321) reg expression
s_mov_b32 s54, s0 // (op 1321) reg expression
s_mov_b32 s55, 131072 // (op 1321) call()

s_mul_hi_i32 s0, s16, s11 // (op 1383) 
v_lshrrev_b32 v20, 3, v1 // (op 1383) 
s_add_i32 s1, s0, s16 // (op 1383) 

v_and_b32 v21, -8, v1 // (op 1383) 
s_ashr_i32 s0, s1, 31 // (op 1383) 
v_sub_u32 v22, v1, v21 // (op 1383) 

s_cmp_eq_i32 s11, 0 // (op 1383) 
s_mov_b32 s18, scc // (op 1383) 
s_cmp_eq_i32 s18, 0 // (op 1383) 
s_cselect_b32 s19, 0, -1 // (op 1383) 

s_and_b32 s18, s12, 31 // (op 1383) 
s_lshl_b32 s49, 1, s18 // (op 1383) 
s_add_i32 s50, s49, s19 // (op 1383) 


s_and_b32 s19, s0, s50 // (op 1383) 


s_add_i32 s0, s1, s19 // (op 1383) 


s_ashr_i32 s1, s0, s18 // (op 1383) 


s_lshl_b32 s0, s12, 24 // (op 1383) 
s_ashr_i32 s18, s0, 31 // (op 1383) 

s_xor_b32 s0, s1, s18 // (op 1383) 

s_sub_i32 s1, s0, s18 // (op 1383) 


s_mul_hi_i32 s0, s1, s13 // (op 1383) 
s_add_i32 s18, s0, s1 // (op 1383) 

s_ashr_i32 s0, s18, 31 // (op 1383) 
s_cmp_eq_i32 s13, 0 // (op 1383) 
s_mov_b32 s19, scc // (op 1383) 
s_cmp_eq_i32 s19, 0 // (op 1383) 
s_cselect_b32 s49, 0, -1 // (op 1383) 

s_and_b32 s19, s14, 31 // (op 1383) 
s_lshl_b32 s50, 1, s19 // (op 1383) 
s_add_i32 s51, s50, s49 // (op 1383) 


s_and_b32 s49, s0, s51 // (op 1383) 


s_add_i32 s0, s18, s49 // (op 1383) 


s_ashr_i32 s18, s0, s19 // (op 1383) 


s_lshl_b32 s0, s14, 24 // (op 1383) 
s_ashr_i32 s19, s0, 31 // (op 1383) 

s_xor_b32 s0, s18, s19 // (op 1383) 

s_sub_i32 s18, s0, s19 // (op 1383) 


s_mul_i32 s0, s1, s44 // (op 1383) 

s_sub_u32 s1, s16, s0 // (op 1383) 

s_mul_i32 s0, s18, s44 // (op 1383) 
s_add_u32 s0, s0, s1 // (op 1383) 


s_mul_hi_i32 s1, s16, s10 // (op 1383) 
s_add_i32 s18, s1, s16 // (op 1383) 

s_ashr_i32 s1, s18, 31 // (op 1383) 
s_cmp_eq_i32 s10, 0 // (op 1383) 
s_mov_b32 s19, scc // (op 1383) 
s_cmp_eq_i32 s19, 0 // (op 1383) 
s_cselect_b32 s49, 0, -1 // (op 1383) 

s_and_b32 s19, s47, 31 // (op 1383) 
s_lshl_b32 s50, 1, s19 // (op 1383) 
s_add_i32 s51, s50, s49 // (op 1383) 


s_and_b32 s49, s1, s51 // (op 1383) 


s_add_i32 s1, s18, s49 // (op 1383) 


s_ashr_i32 s18, s1, s19 // (op 1383) 


s_lshl_b32 s1, s47, 24 // (op 1383) 
s_ashr_i32 s19, s1, 31 // (op 1383) 

s_xor_b32 s1, s18, s19 // (op 1383) 

s_sub_i32 s18, s1, s19 // (op 1383) 


s_add_u32 s50, s6, 64 // (op 1383) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1383) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1383) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1383) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1383) 

s_mul_i32 s1, s44, s50 // (op 1383) 

s_mul_i32 s19, s48, s1 // (op 1383) 

s_mul_i32 s1, s18, s19 // (op 1383) 

s_sub_u32 s19, s16, s1 // (op 1383) 

s_mov_b32 s1, s19 // (op 1383) 
s_mul_hi_i32 s49, s1, s15 // (op 1383) 
s_add_i32 s50, s49, s1 // (op 1383) 


s_ashr_i32 s1, s50, 31 // (op 1383) 
s_cmp_eq_i32 s15, 0 // (op 1383) 
s_mov_b32 s49, scc // (op 1383) 
s_cmp_eq_i32 s49, 0 // (op 1383) 
s_cselect_b32 s51, 0, -1 // (op 1383) 

s_and_b32 s49, s46, 31 // (op 1383) 
s_lshl_b32 s72, 1, s49 // (op 1383) 
s_add_i32 s73, s72, s51 // (op 1383) 


s_and_b32 s51, s1, s73 // (op 1383) 


s_add_i32 s1, s50, s51 // (op 1383) 


s_ashr_i32 s50, s1, s49 // (op 1383) 


s_lshl_b32 s1, s46, 24 // (op 1383) 
s_ashr_i32 s49, s1, 31 // (op 1383) 

s_xor_b32 s1, s50, s49 // (op 1383) 

s_sub_i32 s50, s1, s49 // (op 1383) 


s_mul_i32 s1, s45, s44 // (op 1383) 
s_add_u32 s72, s8, 16 // (op 1383) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 1383) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1383) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1383) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 1383) 

s_sub_i32 s49, s72, s1 // (op 1383) 

s_mul_i32 s51, s50, s49 // (op 1383) 


s_sub_u32 s49, s19, s51 // (op 1383) 


s_add_u32 s19, s49, s1 // (op 1383) 


s_cmp_eq_i32 s18, 0 // (op 1383) 
s_mov_b32 s1, scc // (op 1383) 

s_cmp_eq_i32 s1, 0 // (op 1383) 
s_cselect_b32 s18, s19, s0 // (op 1383) 



v_lshl_add_u32 v21, s18, 4, v20 // (op 1383) 


s_ashr_i64 s[0:1], s[32:33], 8 // (op 1383) 
s_lshr_b32 s18, s0, 1 // (op 1383) 

v_lshl_add_u32 v20, s18, 4, v22 // (op 1383) 


v_mul_lo_u32 v22, v21, s34 // (op 1383) 
v_add_u32 v22, v22, v20 // (op 1383) 


s_mul_i32 s1, s34, s9 // (op 1385) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1385) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1385) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1385) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1385) most significant: sum
s_add_u32 s1, s1, s19 // (op 1385) most significant: sum


s_mov_b32 s64, s36 // (op 1385) reg expression
s_mov_b32 s65, s37 // (op 1385) reg expression
s_mov_b32 s66, s0 // (op 1385) reg expression
s_mov_b32 s67, 131072 // (op 1385) call()

s_mul_hi_i32 s0, s16, s11 // (op 1447) 
v_and_b32 v20, -16, v1 // (op 1447) 
s_add_i32 s1, s0, s16 // (op 1447) 

v_sub_u32 v21, v1, v20 // (op 1447) 

s_ashr_i32 s0, s1, 31 // (op 1447) 
v_lshlrev_b32 v20, 2, v21 // (op 1447) 

s_cmp_eq_i32 s11, 0 // (op 1447) 
s_mov_b32 s18, scc // (op 1447) 
v_lshrrev_b32 v21, 4, v1 // (op 1447) 
s_cmp_eq_i32 s18, 0 // (op 1447) 
s_cselect_b32 s19, 0, -1 // (op 1447) 

s_and_b32 s18, s12, 31 // (op 1447) 
s_lshl_b32 s49, 1, s18 // (op 1447) 
s_add_i32 s50, s49, s19 // (op 1447) 


s_and_b32 s19, s0, s50 // (op 1447) 


s_add_i32 s0, s1, s19 // (op 1447) 


s_ashr_i32 s1, s0, s18 // (op 1447) 


s_lshl_b32 s0, s12, 24 // (op 1447) 
s_ashr_i32 s18, s0, 31 // (op 1447) 

s_xor_b32 s0, s1, s18 // (op 1447) 

s_sub_i32 s1, s0, s18 // (op 1447) 


s_mul_hi_i32 s0, s1, s13 // (op 1447) 
s_add_i32 s18, s0, s1 // (op 1447) 

s_ashr_i32 s0, s18, 31 // (op 1447) 
s_cmp_eq_i32 s13, 0 // (op 1447) 
s_mov_b32 s19, scc // (op 1447) 
s_cmp_eq_i32 s19, 0 // (op 1447) 
s_cselect_b32 s49, 0, -1 // (op 1447) 

s_and_b32 s19, s14, 31 // (op 1447) 
s_lshl_b32 s50, 1, s19 // (op 1447) 
s_add_i32 s51, s50, s49 // (op 1447) 


s_and_b32 s49, s0, s51 // (op 1447) 


s_add_i32 s0, s18, s49 // (op 1447) 


s_ashr_i32 s18, s0, s19 // (op 1447) 


s_lshl_b32 s0, s14, 24 // (op 1447) 
s_ashr_i32 s19, s0, 31 // (op 1447) 

s_xor_b32 s0, s18, s19 // (op 1447) 

s_sub_i32 s18, s0, s19 // (op 1447) 


s_add_u32 s50, s6, 64 // (op 1447) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1447) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1447) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1447) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1447) 

s_mul_i32 s0, s18, s50 // (op 1447) 

s_sub_i32 s49, s1, s0 // (op 1447) 


s_mul_hi_i32 s0, s16, s10 // (op 1447) 
s_add_i32 s1, s0, s16 // (op 1447) 

s_ashr_i32 s0, s1, 31 // (op 1447) 
s_cmp_eq_i32 s10, 0 // (op 1447) 
s_mov_b32 s18, scc // (op 1447) 
s_cmp_eq_i32 s18, 0 // (op 1447) 
s_cselect_b32 s19, 0, -1 // (op 1447) 

s_and_b32 s18, s47, 31 // (op 1447) 
s_lshl_b32 s72, 1, s18 // (op 1447) 
s_add_i32 s73, s72, s19 // (op 1447) 


s_and_b32 s19, s0, s73 // (op 1447) 


s_add_i32 s0, s1, s19 // (op 1447) 


s_ashr_i32 s1, s0, s18 // (op 1447) 


s_lshl_b32 s0, s47, 24 // (op 1447) 
s_ashr_i32 s18, s0, 31 // (op 1447) 

s_xor_b32 s0, s1, s18 // (op 1447) 

s_sub_i32 s1, s0, s18 // (op 1447) 


s_mul_i32 s0, s44, s50 // (op 1447) 

s_mul_i32 s18, s48, s0 // (op 1447) 

s_mul_i32 s0, s1, s18 // (op 1447) 

s_sub_u32 s18, s16, s0 // (op 1447) 

s_mul_hi_i32 s0, s18, s15 // (op 1447) 
s_add_i32 s19, s0, s18 // (op 1447) 


s_ashr_i32 s0, s19, 31 // (op 1447) 
s_cmp_eq_i32 s15, 0 // (op 1447) 
s_mov_b32 s18, scc // (op 1447) 
s_cmp_eq_i32 s18, 0 // (op 1447) 
s_cselect_b32 s50, 0, -1 // (op 1447) 

s_and_b32 s18, s46, 31 // (op 1447) 
s_lshl_b32 s51, 1, s18 // (op 1447) 
s_add_i32 s72, s51, s50 // (op 1447) 


s_and_b32 s50, s0, s72 // (op 1447) 


s_add_i32 s0, s19, s50 // (op 1447) 


s_ashr_i32 s19, s0, s18 // (op 1447) 


s_lshl_b32 s0, s46, 24 // (op 1447) 
s_ashr_i32 s18, s0, 31 // (op 1447) 

s_xor_b32 s0, s19, s18 // (op 1447) 

s_sub_i32 s19, s0, s18 // (op 1447) 


s_cmp_eq_i32 s1, 0 // (op 1447) 
s_mov_b32 s0, scc // (op 1447) 

s_cmp_eq_i32 s0, 0 // (op 1447) 
s_cselect_b32 s1, s19, s49 // (op 1447) 



v_lshl_add_u32 v23, s1, 6, v20 // (op 1447) 


s_ashr_i64 s[0:1], s[32:33], 8 // (op 1447) 
s_lshr_b32 s18, s0, 1 // (op 1447) 

v_lshl_add_u32 v20, s18, 4, v21 // (op 1447) 


v_mul_lo_u32 v21, v20, s38 // (op 1447) 
v_add_u32 v21, v21, v23 // (op 1447) 


s_mul_i32 s1, s38, s43 // (op 1449) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1449) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1449) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1449) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1449) most significant: sum
s_add_u32 s1, s1, s19 // (op 1449) most significant: sum


s_mov_b32 s68, s40 // (op 1449) reg expression
s_mov_b32 s69, s41 // (op 1449) reg expression
s_mov_b32 s70, s0 // (op 1449) reg expression
s_mov_b32 s71, 131072 // (op 1449) call()

v_and_b32 v20, -16, v1 // (op 1540) 
v_sub_u32 v23, v1, v20 // (op 1540) 

v_lshlrev_b32 v20, 4, v23 // (op 1540) 

v_lshrrev_b32 v23, 4, v1 // (op 1540) 
v_lshl_add_u32 v24, v23, 8, v20 // (op 1540) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 1540) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -4, v1 // (op 1592) 
v_sub_u32 v23, v1, v20 // (op 1592) 

v_lshlrev_b32 v20, 4, v23 // (op 1592) 

v_lshrrev_b32 v23, 2, v1 // (op 1592) 
v_lshl_add_u32 v24, v23, 6, v20 // (op 1592) 


s_nop 0
v_readfirstlane_b32 s1, v24 // (op 1592) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 1649) 
v_sub_u32 v23, v1, v20 // (op 1649) 

v_and_b32 v20, 4294967288, v1 // (op 1649) 
v_add_u32 v24, v20, v23 // (op 1649) 


v_and_b32 v20, -16, v1 // (op 1702) 
v_sub_u32 v23, v1, v20 // (op 1702) 

v_lshlrev_b32 v20, 2, v23 // (op 1702) 

v_lshrrev_b32 v23, 4, v1 // (op 1702) 
v_lshl_add_u32 v25, v23, 6, v20 // (op 1702) 


s_ashr_i64 s[18:19], s[32:33], 8 // (op 405) 
s_and_b32 s49, s18, 4294967294 // (op 405) 

s_ashr_i64 s[18:19], s[32:33], 8 // (op 404) 
s_mov_b32 s50, s18 // (op 404) convert

s_cmp_lt_u32 s49, s50 // (op 404) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_404 // (op 404) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_404 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_404:
s_barrier  // (op 1969) 
s_add_u32 s49, s49, 1 // (op 406) 

s_lshl_b64 s[18:19], s[24:25], 3 // (op 1203) 
v_mov_b64 v[16:17], s[18:19] // (op 1203) convert


v_mov_b64 v[18:19], v[14:15] // (op 1203) 
v_mov_b64 v[26:27], v[18:19] // (op 1203) 
s_add_u32 s18, 16384, s0 // (op 1203) 
s_mov_b32 m0, s18 // (op 1203) reg expression
buffer_load_dwordx4 v26, s[20:23], 0 offen offset: 0 lds // (op 1203) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v16 // (op 1203) least significant half
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 1203) most significant half

s_add_u32 m0, m0, 2048 // (op 1203) 
buffer_load_dwordx4 v26, s[20:23], 0 offen offset: 0 lds // (op 1203) Load value direct to lds




v_add_co_u32 v14, vcc, 256, v14 // (op 1241) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1241) most significant half
s_barrier  // (op 1985) 

s_lshl_b64 s[18:19], s[28:29], 5 // (op 1209) 
v_mov_b64 v[16:17], s[18:19] // (op 1209) convert


v_mov_b64 v[18:19], v[40:41] // (op 1209) 
v_mov_b64 v[26:27], v[18:19] // (op 1209) reg expression
s_mov_b32 m0, s1 // (op 1209) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1209) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 1209) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 1209) most significant half

s_add_u32 m0, m0, 2048 // (op 1209) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1209) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 1209) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 1209) most significant half

s_add_u32 m0, m0, 2048 // (op 1209) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1209) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 1209) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 1209) most significant half

s_add_u32 m0, m0, 2048 // (op 1209) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1209) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 1209) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 1209) most significant half

s_add_u32 m0, m0, 2048 // (op 1209) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1209) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 1209) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 1209) most significant half

s_add_u32 m0, m0, 2048 // (op 1209) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1209) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 1209) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 1209) most significant half

s_add_u32 m0, m0, 2048 // (op 1209) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1209) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 1209) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 1209) most significant half

s_add_u32 m0, m0, 2048 // (op 1209) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1209) Load value direct to lds



s_lshl_b64 s[18:19], s[28:29], 8 // (op 1311) 
v_mov_b32 v23, s18 // (op 1311) 
v_mov_b32 v16, s19 // (op 1311) 
v_add_co_u32 v40, s[50:51], v40, v23 // (op 1311) least significant half
v_addc_co_u32 v41, s[50:51], v41, v16, s[50:51] // (op 1311) most significant half






v_mov_b32 v23, v22 // (op 222) 

buffer_load_ubyte v16, v23, s[64:67], 0 offen offset: 0 // (op 222) Load value

v_add_u32 v22, 8, v22 // (op 1375) 


s_barrier  // (op 2002) 
v_mov_b32 v23, v21 // (op 223) 



v_mov_b32 v17, v23 // (op 223) reg expression

buffer_load_dword v18, v17, s[68:71], 0 offen // (op 223) Load value


s_lshl_b32 s18, s38, 3 // (op 1439) 
v_add_u32 v21, v21, s18 // (op 1439) 

s_waitcnt vmcnt(1)
ds_write_b8 v24, v16 offset:41472 // (op 219) Store local data 

s_barrier  // (op 1975) 



s_barrier  // (op 2020) 
ds_read_b128 v[26:29], v3 offset:16384 // (op 201) Load local data 



ds_read_b128 v[30:33], v3 offset:16448 // (op 201) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v25, v18 offset:40960 // (op 220) Store local data 

s_barrier  // (op 1992) 



ds_read_b64_tr_b8 v[42:43], v9 // (op 203) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v9 offset:512 // (op 203) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v9 offset:4096 // (op 203) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v9 offset:4608 // (op 203) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 2010) 



ds_read_u8 v23, v0 offset:41472 // (op 205) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2027) 


ds_read_u8 v16, v6 offset:40960 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[42:49], a[0:3], v23, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 208) 





ds_read_b128 v[42:45], v3 offset:16512 // (op 210) Load local data 
ds_read_b128 v[46:49], v3 offset:16576 // (op 210) Load local data 



ds_read_b64_tr_b8 v[50:51], v9 offset:8192 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v9 offset:8704 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v9 offset:12288 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v9 offset:12800 // (op 212) Transpose load local data 



ds_read_u8 v16, v0 offset:41476 // (op 214) Load local data 


ds_read_u8 v17, v6 offset:41216 // (op 216) Load local data 



ds_read_b64_tr_b8 v[58:59], v9 offset:32 // (op 412) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v9 offset:544 // (op 412) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v9 offset:4128 // (op 412) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v9 offset:4640 // (op 412) Transpose load local data 


ds_read_u8 v18, v6 offset:40992 // (op 416) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[58:65], a[4:7], v23, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 417) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:57], a[0:3], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 217) 
ds_read_b64_tr_b8 v[26:27], v9 offset:8224 // (op 421) Transpose load local data 


ds_read_b64_tr_b8 v[28:29], v9 offset:8736 // (op 421) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v9 offset:12320 // (op 421) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v9 offset:12832 // (op 421) Transpose load local data 


ds_read_u8 v23, v6 offset:41248 // (op 425) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[26:33], a[4:7], v16, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 426) 




s_ashr_i64 s[18:19], s[32:33], 8 // (op 404) 
s_mov_b32 s50, s18 // (op 404) convert

s_cmp_lt_u32 s49, s50 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_404 // (op 404) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_404 if true)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_404:























s_barrier  // (op 2107) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_155
v_accvgpr_read v12, a0 // (op 50) 
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v10, v12 // (op 50) 
v_mul_f32 v17, v10, v13 // (op 50) 
v_mul_f32 v18, v10, v14 // (op 50) 
v_mul_f32 v19, v10, v15 // (op 50) 



v_mov_b32 v0, v11 // (op 56) reg expression

buffer_store_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 56) Store value


v_accvgpr_read v12, a4 // (op 710) 
v_accvgpr_read v13, a5 // (op 710) 
v_accvgpr_read v14, a6 // (op 710) 
v_accvgpr_read v15, a7 // (op 710) 
v_mul_f32 v16, v10, v12 // (op 710) 
v_mul_f32 v17, v10, v13 // (op 710) 
v_mul_f32 v18, v10, v14 // (op 710) 
v_mul_f32 v19, v10, v15 // (op 710) 



s_lshl_b32 s0, s4, 7 // (op 712) 
v_add_u32 v0, v11, s0 // (op 712) 


buffer_store_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 712) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_155
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_155:


v_mov_b32 v0, v8 // (op 167) reg expression

buffer_load_dwordx4 v[12:15], v0, s[56:59], 0 offen // (op 167) Load value

v_accvgpr_read v16, a0 // (op 164) 
v_accvgpr_read v17, a1 // (op 164) 
v_accvgpr_read v18, a2 // (op 164) 
v_accvgpr_read v19, a3 // (op 164) 
v_mul_f32 v20, v10, v16 // (op 164) 
v_mul_f32 v21, v10, v17 // (op 164) 
v_mul_f32 v22, v10, v18 // (op 164) 
v_mul_f32 v23, v10, v19 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v7, v12 // (op 168) 
v_mul_f32 v17, v7, v13 // (op 168) 
v_mul_f32 v18, v7, v14 // (op 168) 
v_mul_f32 v19, v7, v15 // (op 168) 

v_add_f32 v12, v16, v20 // (op 165) 
v_add_f32 v13, v17, v21 // (op 165) 
v_add_f32 v14, v18, v22 // (op 165) 
v_add_f32 v15, v19, v23 // (op 165) 




v_mov_b32 v0, v11 // (op 166) reg expression

buffer_store_dwordx4 v[12:15], v0, s[60:63], 0 offen // (op 166) Store value




s_lshl_b32 s0, s2, 7 // (op 734) 
v_add_u32 v0, v8, s0 // (op 734) 


buffer_load_dwordx4 v[12:15], v0, s[56:59], 0 offen // (op 734) Load value

v_accvgpr_read v16, a4 // (op 731) 
v_accvgpr_read v17, a5 // (op 731) 
v_accvgpr_read v18, a6 // (op 731) 
v_accvgpr_read v19, a7 // (op 731) 
v_mul_f32 v20, v10, v16 // (op 731) 
v_mul_f32 v21, v10, v17 // (op 731) 
v_mul_f32 v22, v10, v18 // (op 731) 
v_mul_f32 v23, v10, v19 // (op 731) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v7, v12 // (op 735) 
v_mul_f32 v17, v7, v13 // (op 735) 
v_mul_f32 v18, v7, v14 // (op 735) 
v_mul_f32 v19, v7, v15 // (op 735) 

v_add_f32 v12, v16, v20 // (op 732) 
v_add_f32 v13, v17, v21 // (op 732) 
v_add_f32 v14, v18, v22 // (op 732) 
v_add_f32 v15, v19, v23 // (op 732) 




s_lshl_b32 s0, s4, 7 // (op 733) 
v_add_u32 v0, v11, s0 // (op 733) 


buffer_store_dwordx4 v[12:15], v0, s[60:63], 0 offen // (op 733) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_155:










s_endpgm  // End of RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.LRR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end:
.size RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_, .LRR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end-RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 128
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 42240 // lds bytes
  .amdhsa_accum_offset 120
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
    DataTypeB: B8
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
    macrotile: [16, 64, 256]
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
  - .name: RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 42240
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 120
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

