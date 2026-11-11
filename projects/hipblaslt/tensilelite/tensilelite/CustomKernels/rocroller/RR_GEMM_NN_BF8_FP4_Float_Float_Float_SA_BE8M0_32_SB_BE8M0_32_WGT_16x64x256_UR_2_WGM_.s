// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_,@function
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1438) 
s_mul_hi_i32 s1, s16, s10 // (op 1680) 
v_and_b32 v0, 48, v1 // (op 1495) 
v_and_b32 v3, 63, v1 // (op 1495) 
v_sub_u32 v4, v3, v0 // (op 1495) 

v_lshrrev_b32 v3, 1, v4 // (op 1495) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 1495) 


v_and_b32 v0, 1, v4 // (op 1495) 

v_lshlrev_b32 v3, 3, v0 // (op 1495) 

v_lshrrev_b32 v0, 7, v1 // (op 1495) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1495) 


v_add_u32 v0, v5, v4 // (op 1495) 


v_lshrrev_b32 v3, 7, v1 // (op 1606) 
v_and_b32 v4, 15, v1 // (op 1606) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 1606) 


v_and_b32 v3, 48, v1 // (op 1606) 
v_add_u32 v4, v3, v5 // (op 1606) 


v_and_b32 v3, 48, v1 // (op 1706) 
v_and_b32 v5, 63, v1 // (op 1706) 
v_sub_u32 v6, v5, v3 // (op 1706) 

v_lshrrev_b32 v5, 1, v6 // (op 1706) 
v_add_lshl_u32 v7, v3, v5, 4 // (op 1706) 


v_and_b32 v3, 1, v6 // (op 1706) 

v_lshlrev_b32 v5, 3, v3 // (op 1706) 

v_lshrrev_b32 v3, 7, v1 // (op 1706) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 1706) 


v_add_u32 v3, v7, v6 // (op 1706) 


v_lshrrev_b32 v5, 7, v1 // (op 1759) 
v_and_b32 v6, 15, v1 // (op 1759) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 1759) 


v_and_b32 v5, 48, v1 // (op 1759) 
v_add_u32 v6, v5, v7 // (op 1759) 


v_lshrrev_b32 v5, 7, v1 // (op 1438) 
s_add_i32 s18, s0, s16 // (op 1438) 

v_and_b32 v7, 63, v1 // (op 1438) 
s_ashr_i32 s0, s18, 31 // (op 1438) 
v_lshrrev_b32 v8, 4, v7 // (op 1438) 
s_cmp_eq_i32 s10, 0 // (op 1438) 
s_mov_b32 s19, scc // (op 1438) 
v_lshlrev_b32 v9, 2, v8 // (op 1438) 

s_cmp_eq_i32 s19, 0 // (op 1438) 
s_cselect_b32 s51, 0, -1 // (op 1438) 

v_lshl_add_u32 v8, v5, 4, v9 // (op 1438) 


s_and_b32 s19, s11, 31 // (op 1438) 
v_lshrrev_b32 v5, 2, v7 // (op 1438) 

v_and_b32 v7, 3, v5 // (op 1438) 

v_lshrrev_b32 v5, 6, v1 // (op 1553) 
v_and_b32 v9, 1, v5 // (op 1553) 

v_and_b32 v5, 15, v1 // (op 1553) 
v_lshl_add_u32 v10, v9, 4, v5 // (op 1553) 


v_and_b32 v5, 63, v1 // (op 1553) 
v_lshrrev_b32 v9, 4, v5 // (op 1553) 

v_lshlrev_b32 v5, 5, v9 // (op 1553) 

v_lshl_add_u32 v9, v10, 8, v5 // (op 1553) 


v_lshlrev_b32 v5, 2, v9 // (op 1553) 

v_lshrrev_b32 v9, 3, v5 // (op 1553) 

v_lshrrev_b32 v5, 6, v1 // (op 1664) 
v_and_b32 v10, 1, v5 // (op 1664) 

v_and_b32 v5, 15, v1 // (op 1664) 
v_lshl_add_u32 v11, v10, 4, v5 // (op 1664) 


v_and_b32 v5, 63, v1 // (op 1664) 
v_lshrrev_b32 v10, 4, v5 // (op 1664) 

v_lshl_add_u32 v5, v11, 3, v10 // (op 1664) 


v_lshrrev_b32 v10, 7, v1 // (op 1680) 
v_and_b32 v11, 63, v1 // (op 1680) 
v_lshrrev_b32 v12, 6, v1 // (op 1735) 
v_and_b32 v13, 1, v12 // (op 1735) 

v_and_b32 v12, 15, v1 // (op 1735) 
v_lshl_add_u32 v14, v13, 4, v12 // (op 1735) 


v_and_b32 v12, 63, v1 // (op 1735) 
v_lshrrev_b32 v13, 4, v12 // (op 1735) 

v_lshlrev_b32 v12, 5, v13 // (op 1735) 

v_lshl_add_u32 v13, v14, 8, v12 // (op 1735) 


v_lshlrev_b32 v12, 2, v13 // (op 1735) 

v_lshrrev_b32 v13, 3, v12 // (op 1735) 

v_lshrrev_b32 v12, 6, v1 // (op 1788) 
v_and_b32 v14, 1, v12 // (op 1788) 

v_and_b32 v12, 15, v1 // (op 1788) 
v_lshl_add_u32 v15, v14, 4, v12 // (op 1788) 


v_and_b32 v12, 63, v1 // (op 1788) 
v_lshrrev_b32 v14, 4, v12 // (op 1788) 

v_lshl_add_u32 v12, v15, 3, v14 // (op 1788) 


s_lshl_b32 s52, 1, s19 // (op 1438) 
v_and_b32 v14, 3, v1 // (op 1438) 
s_add_i32 s53, s52, s51 // (op 1438) 


s_and_b32 s51, s0, s53 // (op 1438) 


s_add_i32 s0, s1, s16 // (op 1680) 

s_ashr_i32 s1, s0, 31 // (op 1680) 
v_lshl_add_u32 v15, v7, 2, v14 // (op 1438) 


s_add_i32 s52, s18, s51 // (op 1438) 


v_lshrrev_b32 v7, 6, v1 // (op 1438) 
s_ashr_i32 s18, s52, s19 // (op 1438) 


v_and_b32 v14, 1, v7 // (op 1438) 

s_lshl_b32 s19, s11, 24 // (op 1438) 
v_lshl_add_u32 v7, v14, 4, v15 // (op 1438) 


s_ashr_i32 s51, s19, 31 // (op 1438) 

s_xor_b32 s19, s18, s51 // (op 1438) 

s_sub_i32 s18, s19, s51 // (op 1438) 


s_mul_hi_i32 s19, s18, s12 // (op 1438) 
s_add_i32 s51, s19, s18 // (op 1438) 

s_ashr_i32 s19, s51, 31 // (op 1438) 
s_cmp_eq_i32 s12, 0 // (op 1438) 
s_mov_b32 s52, scc // (op 1438) 
s_cmp_eq_i32 s52, 0 // (op 1438) 
s_cselect_b32 s53, 0, -1 // (op 1438) 

s_and_b32 s52, s13, 31 // (op 1438) 
v_lshrrev_b32 v14, 4, v11 // (op 1680) 
s_cmp_eq_i32 s10, 0 // (op 1680) 
s_mov_b32 s54, scc // (op 1680) 
v_lshlrev_b32 v15, 2, v14 // (op 1680) 

s_cmp_eq_i32 s54, 0 // (op 1680) 
s_cselect_b32 s55, 0, -1 // (op 1680) 

s_lshl_b32 s54, 1, s52 // (op 1438) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 1680) 


v_lshrrev_b32 v10, 2, v11 // (op 1680) 

s_add_i32 s56, s54, s53 // (op 1438) 


s_and_b32 s53, s19, s56 // (op 1438) 


s_add_i32 s19, s51, s53 // (op 1438) 


s_ashr_i32 s51, s19, s52 // (op 1438) 


s_lshl_b32 s19, s13, 24 // (op 1438) 
s_ashr_i32 s52, s19, 31 // (op 1438) 

s_xor_b32 s19, s51, s52 // (op 1438) 

s_sub_i32 s51, s19, s52 // (op 1438) 


s_mul_i32 s19, s18, s15 // (op 1438) 
s_sub_u32 s52, s16, s19 // (op 1438) 

s_mul_i32 s19, s51, s15 // (op 1438) 
s_add_u32 s19, s19, s52 // (op 1438) 

s_mul_hi_i32 s52, s16, s9 // (op 1438) 
s_add_i32 s53, s52, s16 // (op 1438) 

s_ashr_i32 s52, s53, 31 // (op 1438) 
s_cmp_eq_i32 s9, 0 // (op 1438) 
s_mov_b32 s54, scc // (op 1438) 
s_cmp_eq_i32 s54, 0 // (op 1438) 
s_cselect_b32 s56, 0, -1 // (op 1438) 

s_and_b32 s54, s11, 31 // (op 1680) 
v_and_b32 v11, 3, v10 // (op 1680) 

s_lshl_b32 s57, 1, s54 // (op 1680) 
v_and_b32 v10, 3, v1 // (op 1680) 
s_add_i32 s58, s57, s55 // (op 1680) 


s_and_b32 s55, s1, s58 // (op 1680) 


v_lshl_add_u32 v15, v11, 2, v10 // (op 1680) 


s_add_i32 s1, s0, s55 // (op 1680) 


v_lshrrev_b32 v10, 6, v1 // (op 1680) 
s_ashr_i32 s0, s1, s54 // (op 1680) 


v_and_b32 v11, 1, v10 // (op 1680) 

s_lshl_b32 s1, s11, 24 // (op 1680) 
v_lshl_add_u32 v10, v11, 4, v15 // (op 1680) 


s_ashr_i32 s54, s1, 31 // (op 1680) 

s_xor_b32 s1, s0, s54 // (op 1680) 

s_sub_i32 s0, s1, s54 // (op 1680) 


s_mul_hi_i32 s1, s0, s12 // (op 1680) 
s_add_i32 s54, s1, s0 // (op 1680) 

s_ashr_i32 s1, s54, 31 // (op 1680) 
s_cmp_eq_i32 s12, 0 // (op 1680) 
s_mov_b32 s55, scc // (op 1680) 
s_cmp_eq_i32 s55, 0 // (op 1680) 
s_cselect_b32 s57, 0, -1 // (op 1680) 

s_and_b32 s55, s13, 31 // (op 1680) 
s_lshl_b32 s58, 1, s55 // (op 1680) 
s_add_i32 s59, s58, s57 // (op 1680) 


s_and_b32 s57, s1, s59 // (op 1680) 


s_add_i32 s1, s54, s57 // (op 1680) 


s_ashr_i32 s54, s1, s55 // (op 1680) 


s_lshl_b32 s1, s13, 24 // (op 1680) 
s_ashr_i32 s55, s1, 31 // (op 1680) 

s_xor_b32 s1, s54, s55 // (op 1680) 

s_sub_i32 s54, s1, s55 // (op 1680) 


s_mul_i32 s1, s0, s15 // (op 1680) 
s_sub_u32 s55, s16, s1 // (op 1680) 

s_mul_i32 s1, s54, s15 // (op 1680) 
s_add_u32 s1, s1, s55 // (op 1680) 

s_mul_hi_i32 s55, s16, s9 // (op 1680) 
s_add_i32 s57, s55, s16 // (op 1680) 

s_ashr_i32 s55, s57, 31 // (op 1680) 
s_cmp_eq_i32 s9, 0 // (op 1680) 
s_mov_b32 s58, scc // (op 1680) 
s_cmp_eq_i32 s58, 0 // (op 1680) 
s_cselect_b32 s59, 0, -1 // (op 1680) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 1438) 
s_lshl_b32 s60, 1, s58 // (op 1438) 
s_add_i32 s61, s60, s56 // (op 1438) 


s_and_b32 s56, s52, s61 // (op 1438) 


s_add_i32 s52, s53, s56 // (op 1438) 


s_ashr_i32 s53, s52, s58 // (op 1438) 


s_lshl_b32 s52, s46, 24 // (op 1438) 
s_ashr_i32 s56, s52, 31 // (op 1438) 

s_xor_b32 s52, s53, s56 // (op 1438) 

s_sub_i32 s53, s52, s56 // (op 1438) 


s_and_b32 s52, s46, 31 // (op 1680) 
s_lshl_b32 s56, 1, s52 // (op 1680) 
s_add_i32 s58, s56, s59 // (op 1680) 


s_and_b32 s56, s55, s58 // (op 1680) 


s_add_i32 s55, s57, s56 // (op 1680) 


s_ashr_i32 s56, s55, s52 // (op 1680) 


s_lshl_b32 s52, s46, 24 // (op 1680) 
s_ashr_i32 s55, s52, 31 // (op 1680) 

s_xor_b32 s52, s56, s55 // (op 1680) 

s_sub_i32 s56, s52, s55 // (op 1680) 


s_add_u32 s58, s6, 64 // (op 1438) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 1438) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1438) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1438) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 1438) 

s_mul_i32 s52, s15, s58 // (op 1438) 
s_mul_i32 s55, s8, s52 // (op 1438) 

s_mul_i32 s52, s53, s55 // (op 1438) 

s_sub_u32 s55, s16, s52 // (op 1438) 

s_mov_b32 s52, s55 // (op 1438) 
s_mul_hi_i32 s57, s52, s14 // (op 1438) 
s_add_i32 s60, s57, s52 // (op 1438) 


s_ashr_i32 s52, s60, 31 // (op 1438) 
s_cmp_eq_i32 s14, 0 // (op 1438) 
s_mov_b32 s57, scc // (op 1438) 
s_cmp_eq_i32 s57, 0 // (op 1438) 
s_cselect_b32 s61, 0, -1 // (op 1438) 

s_and_b32 s57, s48, 31 // (op 1438) 
s_lshl_b32 s62, 1, s57 // (op 1438) 
s_add_i32 s63, s62, s61 // (op 1438) 


s_and_b32 s61, s52, s63 // (op 1438) 


s_add_i32 s52, s60, s61 // (op 1438) 


s_ashr_i32 s60, s52, s57 // (op 1438) 


s_lshl_b32 s52, s48, 24 // (op 1438) 
s_ashr_i32 s57, s52, 31 // (op 1438) 

s_xor_b32 s52, s60, s57 // (op 1438) 

s_sub_i32 s60, s52, s57 // (op 1438) 


s_mul_i32 s52, s47, s15 // (op 1438) 
s_add_u32 s62, s20, 16 // (op 1438) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 1438) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1438) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1438) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 1438) 

s_sub_i32 s57, s62, s52 // (op 1438) 

s_mul_i32 s61, s60, s57 // (op 1438) 

s_sub_u32 s57, s55, s61 // (op 1438) 


s_add_u32 s55, s57, s52 // (op 1438) 


s_cmp_eq_i32 s53, 0 // (op 1438) 
s_mov_b32 s52, scc // (op 1438) 

s_cmp_eq_i32 s52, 0 // (op 1438) 
s_cselect_b32 s53, s55, s19 // (op 1438) 


s_mul_i32 s19, s51, s58 // (op 1438) 


v_lshl_add_u32 v11, s53, 4, v8 // (op 1438) 


s_sub_i32 s51, s18, s19 // (op 1438) 


s_cmp_eq_i32 s52, 0 // (op 1438) 
s_cselect_b32 s53, s60, s51 // (op 1438) 



v_lshl_add_u32 v8, s53, 6, v7 // (op 1438) 


v_mul_lo_u32 v7, v8, s2 // (op 1438) 

v_add_lshl_u32 v8, v11, v7, 2 // (op 1438) 


s_add_u32 s18, s6, 64 // (op 1680) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 1680) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 1680) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 1680) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 6 // (op 1680) 

s_mul_i32 s55, s15, s18 // (op 1680) 
s_mul_i32 s51, s8, s55 // (op 1680) 

s_mul_i32 s55, s56, s51 // (op 1680) 

s_sub_u32 s51, s16, s55 // (op 1680) 

s_mov_b32 s55, s51 // (op 1680) 
s_mul_hi_i32 s52, s55, s14 // (op 1680) 
s_add_i32 s53, s52, s55 // (op 1680) 


s_ashr_i32 s52, s53, 31 // (op 1680) 
s_cmp_eq_i32 s14, 0 // (op 1680) 
s_mov_b32 s55, scc // (op 1680) 
s_cmp_eq_i32 s55, 0 // (op 1680) 
s_cselect_b32 s57, 0, -1 // (op 1680) 

s_and_b32 s55, s48, 31 // (op 1680) 
s_lshl_b32 s58, 1, s55 // (op 1680) 
s_add_i32 s59, s58, s57 // (op 1680) 


s_and_b32 s57, s52, s59 // (op 1680) 


s_add_i32 s52, s53, s57 // (op 1680) 


s_ashr_i32 s53, s52, s55 // (op 1680) 


s_lshl_b32 s52, s48, 24 // (op 1680) 
s_ashr_i32 s55, s52, 31 // (op 1680) 

s_xor_b32 s52, s53, s55 // (op 1680) 

s_sub_i32 s53, s52, s55 // (op 1680) 


s_mul_i32 s52, s47, s15 // (op 1680) 
s_add_u32 s58, s20, 16 // (op 1680) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 1680) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1680) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1680) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1680) 

s_sub_i32 s55, s58, s52 // (op 1680) 

s_mul_i32 s57, s53, s55 // (op 1680) 

s_sub_u32 s55, s51, s57 // (op 1680) 


s_add_u32 s51, s55, s52 // (op 1680) 


s_cmp_eq_i32 s56, 0 // (op 1680) 
s_mov_b32 s52, scc // (op 1680) 

s_cmp_eq_i32 s52, 0 // (op 1680) 
s_cselect_b32 s55, s51, s1 // (op 1680) 


s_mul_i32 s1, s54, s18 // (op 1680) 


v_lshl_add_u32 v7, s55, 4, v14 // (op 1680) 


s_sub_i32 s51, s0, s1 // (op 1680) 


s_cmp_eq_i32 s52, 0 // (op 1680) 
s_cselect_b32 s0, s53, s51 // (op 1680) 



v_lshl_add_u32 v11, s0, 6, v10 // (op 1680) 


v_mul_lo_u32 v10, v11, s4 // (op 1680) 

v_add_lshl_u32 v11, v7, v10, 2 // (op 1680) 


s_mul_i32 s1, s2, s7 // (op 1440) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1440) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1440) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1440) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1440) most significant: sum
s_add_u32 s1, s1, s19 // (op 1440) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1440) 

s_mul_i32 s1, s4, s7 // (op 1682) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 1682) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 1682) least significant
s_mul_hi_u32 s52, s4, s6 // (op 1682) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1682) most significant: sum
s_add_u32 s1, s1, s52 // (op 1682) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 1682) 

s_mov_b32 s56, s22 // (op 1440) reg expression
s_mov_b32 s57, s23 // (op 1440) reg expression
s_mov_b32 s58, s18 // (op 1440) reg expression
s_mov_b32 s59, 131072 // (op 1440) call()

s_mov_b32 s60, s24 // (op 1682) reg expression

s_mov_b32 s61, s25 // (op 1682) reg expression
s_mov_b32 s62, s52 // (op 1682) reg expression
s_mov_b32 s63, 131072 // (op 1682) call()


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
s_mul_hi_i32 s0, s16, s10 // (op 1205) 
v_lshlrev_b32 v14, 1, v1 // (op 1205) 
s_add_i32 s1, s0, s16 // (op 1205) 

s_ashr_i32 s0, s1, 31 // (op 1205) 
s_cmp_eq_i32 s10, 0 // (op 1205) 
s_mov_b32 s18, scc // (op 1205) 
s_cmp_eq_i32 s18, 0 // (op 1205) 
s_cselect_b32 s19, 0, -1 // (op 1205) 

s_and_b32 s18, s11, 31 // (op 1205) 
s_lshl_b32 s22, 1, s18 // (op 1205) 
s_add_i32 s23, s22, s19 // (op 1205) 


s_and_b32 s19, s0, s23 // (op 1205) 


s_add_i32 s0, s1, s19 // (op 1205) 


s_ashr_i32 s1, s0, s18 // (op 1205) 


s_lshl_b32 s0, s11, 24 // (op 1205) 
s_ashr_i32 s18, s0, 31 // (op 1205) 

s_xor_b32 s0, s1, s18 // (op 1205) 

s_sub_i32 s1, s0, s18 // (op 1205) 


s_mul_hi_i32 s0, s1, s12 // (op 1205) 
s_add_i32 s18, s0, s1 // (op 1205) 

s_ashr_i32 s0, s18, 31 // (op 1205) 
s_cmp_eq_i32 s12, 0 // (op 1205) 
s_mov_b32 s19, scc // (op 1205) 
s_cmp_eq_i32 s19, 0 // (op 1205) 
s_cselect_b32 s22, 0, -1 // (op 1205) 

s_and_b32 s19, s13, 31 // (op 1205) 
s_lshl_b32 s23, 1, s19 // (op 1205) 
s_add_i32 s24, s23, s22 // (op 1205) 


s_and_b32 s25, s0, s24 // (op 1205) 


s_add_i32 s0, s18, s25 // (op 1205) 


s_ashr_i32 s18, s0, s19 // (op 1205) 


s_lshl_b32 s0, s13, 24 // (op 1205) 
s_ashr_i32 s19, s0, 31 // (op 1205) 

s_xor_b32 s0, s18, s19 // (op 1205) 

s_sub_i32 s18, s0, s19 // (op 1205) 


s_mul_i32 s0, s1, s15 // (op 1205) 

s_sub_u32 s1, s16, s0 // (op 1205) 

s_mul_i32 s0, s18, s15 // (op 1205) 
s_add_u32 s0, s0, s1 // (op 1205) 


s_mul_hi_i32 s1, s16, s9 // (op 1205) 
s_add_i32 s18, s1, s16 // (op 1205) 

s_ashr_i32 s1, s18, 31 // (op 1205) 
s_cmp_eq_i32 s9, 0 // (op 1205) 
s_mov_b32 s19, scc // (op 1205) 
s_cmp_eq_i32 s19, 0 // (op 1205) 
s_cselect_b32 s22, 0, -1 // (op 1205) 

s_and_b32 s19, s46, 31 // (op 1205) 
s_lshl_b32 s23, 1, s19 // (op 1205) 
s_add_i32 s24, s23, s22 // (op 1205) 


s_and_b32 s25, s1, s24 // (op 1205) 


s_add_i32 s1, s18, s25 // (op 1205) 


s_ashr_i32 s18, s1, s19 // (op 1205) 


s_lshl_b32 s1, s46, 24 // (op 1205) 
s_ashr_i32 s19, s1, 31 // (op 1205) 

s_xor_b32 s1, s18, s19 // (op 1205) 

s_sub_i32 s18, s1, s19 // (op 1205) 


s_add_u32 s22, s6, 64 // (op 1205) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1205) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1205) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1205) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1205) 

s_mul_i32 s1, s15, s22 // (op 1205) 

s_mul_i32 s19, s8, s1 // (op 1205) 

s_mul_i32 s1, s18, s19 // (op 1205) 

s_sub_u32 s19, s16, s1 // (op 1205) 

s_mov_b32 s1, s19 // (op 1205) 
s_mul_hi_i32 s22, s1, s14 // (op 1205) 
s_add_i32 s23, s22, s1 // (op 1205) 


s_ashr_i32 s1, s23, 31 // (op 1205) 
s_cmp_eq_i32 s14, 0 // (op 1205) 
s_mov_b32 s22, scc // (op 1205) 
s_cmp_eq_i32 s22, 0 // (op 1205) 
s_cselect_b32 s24, 0, -1 // (op 1205) 

s_and_b32 s22, s48, 31 // (op 1205) 
s_lshl_b32 s25, 1, s22 // (op 1205) 
s_add_i32 s49, s25, s24 // (op 1205) 


s_and_b32 s24, s1, s49 // (op 1205) 


s_add_i32 s1, s23, s24 // (op 1205) 


s_ashr_i32 s23, s1, s22 // (op 1205) 


s_lshl_b32 s1, s48, 24 // (op 1205) 
s_ashr_i32 s22, s1, 31 // (op 1205) 

s_xor_b32 s1, s23, s22 // (op 1205) 

s_sub_i32 s23, s1, s22 // (op 1205) 


s_mul_i32 s1, s47, s15 // (op 1205) 
s_add_u32 s24, s20, 16 // (op 1205) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1205) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 1205) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 1205) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 4 // (op 1205) 

s_sub_i32 s22, s24, s1 // (op 1205) 

s_mul_i32 s24, s23, s22 // (op 1205) 


s_sub_u32 s25, s19, s24 // (op 1205) 


s_add_u32 s19, s25, s1 // (op 1205) 


s_cmp_eq_i32 s18, 0 // (op 1205) 
s_mov_b32 s1, scc // (op 1205) 

s_cmp_eq_i32 s1, 0 // (op 1205) 
s_cselect_b32 s18, s19, s0 // (op 1205) 



v_mul_lo_u32 v15, v14, s26 // (op 1205) 

v_lshl_add_u32 v14, s18, 4, v15 // (op 1205) 


s_mul_i32 s1, s26, s31 // (op 1207) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1207) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1207) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1207) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1207) most significant: sum
s_add_u32 s1, s1, s19 // (op 1207) most significant: sum


s_mov_b32 s52, s28 // (op 1207) reg expression
s_mov_b32 s53, s29 // (op 1207) reg expression
s_mov_b32 s54, s0 // (op 1207) reg expression
s_mov_b32 s55, 131072 // (op 1207) call()

s_mul_hi_i32 s0, s16, s10 // (op 1224) 
v_lshlrev_b32 v15, 1, v1 // (op 1224) 
s_add_i32 s1, s0, s16 // (op 1224) 

s_ashr_i32 s0, s1, 31 // (op 1224) 
s_cmp_eq_i32 s10, 0 // (op 1224) 
s_mov_b32 s18, scc // (op 1224) 
s_cmp_eq_i32 s18, 0 // (op 1224) 
s_cselect_b32 s19, 0, -1 // (op 1224) 

s_and_b32 s18, s11, 31 // (op 1224) 
s_lshl_b32 s22, 1, s18 // (op 1224) 
s_add_i32 s23, s22, s19 // (op 1224) 


s_and_b32 s19, s0, s23 // (op 1224) 


s_add_i32 s0, s1, s19 // (op 1224) 


s_ashr_i32 s1, s0, s18 // (op 1224) 


s_lshl_b32 s0, s11, 24 // (op 1224) 
s_ashr_i32 s18, s0, 31 // (op 1224) 

s_xor_b32 s0, s1, s18 // (op 1224) 

s_sub_i32 s1, s0, s18 // (op 1224) 


s_mul_hi_i32 s0, s1, s12 // (op 1224) 
s_add_i32 s18, s0, s1 // (op 1224) 

s_ashr_i32 s0, s18, 31 // (op 1224) 
s_cmp_eq_i32 s12, 0 // (op 1224) 
s_mov_b32 s19, scc // (op 1224) 
s_cmp_eq_i32 s19, 0 // (op 1224) 
s_cselect_b32 s22, 0, -1 // (op 1224) 

s_and_b32 s19, s13, 31 // (op 1224) 
s_lshl_b32 s23, 1, s19 // (op 1224) 
s_add_i32 s24, s23, s22 // (op 1224) 


s_and_b32 s25, s0, s24 // (op 1224) 


s_add_i32 s0, s18, s25 // (op 1224) 


s_ashr_i32 s18, s0, s19 // (op 1224) 


s_lshl_b32 s0, s13, 24 // (op 1224) 
s_ashr_i32 s19, s0, 31 // (op 1224) 

s_xor_b32 s0, s18, s19 // (op 1224) 

s_sub_i32 s18, s0, s19 // (op 1224) 


s_mul_i32 s0, s1, s15 // (op 1224) 

s_sub_u32 s1, s16, s0 // (op 1224) 

s_mul_i32 s0, s18, s15 // (op 1224) 
s_add_u32 s0, s0, s1 // (op 1224) 


s_mul_hi_i32 s1, s16, s9 // (op 1224) 
s_add_i32 s18, s1, s16 // (op 1224) 

s_ashr_i32 s1, s18, 31 // (op 1224) 
s_cmp_eq_i32 s9, 0 // (op 1224) 
s_mov_b32 s19, scc // (op 1224) 
s_cmp_eq_i32 s19, 0 // (op 1224) 
s_cselect_b32 s22, 0, -1 // (op 1224) 

s_and_b32 s19, s46, 31 // (op 1224) 
s_lshl_b32 s23, 1, s19 // (op 1224) 
s_add_i32 s24, s23, s22 // (op 1224) 


s_and_b32 s25, s1, s24 // (op 1224) 


s_add_i32 s1, s18, s25 // (op 1224) 


s_ashr_i32 s18, s1, s19 // (op 1224) 


s_lshl_b32 s1, s46, 24 // (op 1224) 
s_ashr_i32 s19, s1, 31 // (op 1224) 

s_xor_b32 s1, s18, s19 // (op 1224) 

s_sub_i32 s18, s1, s19 // (op 1224) 


s_add_u32 s50, s6, 64 // (op 1224) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1224) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1224) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1224) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1224) 

s_mul_i32 s1, s15, s24 // (op 1224) 

s_mul_i32 s19, s8, s1 // (op 1224) 

s_mul_i32 s1, s18, s19 // (op 1224) 

s_sub_u32 s19, s16, s1 // (op 1224) 

s_mov_b32 s1, s19 // (op 1224) 
s_mul_hi_i32 s22, s1, s14 // (op 1224) 
s_add_i32 s23, s22, s1 // (op 1224) 


s_ashr_i32 s1, s23, 31 // (op 1224) 
s_cmp_eq_i32 s14, 0 // (op 1224) 
s_mov_b32 s22, scc // (op 1224) 
s_cmp_eq_i32 s22, 0 // (op 1224) 
s_cselect_b32 s24, 0, -1 // (op 1224) 

s_and_b32 s22, s48, 31 // (op 1224) 
s_lshl_b32 s25, 1, s22 // (op 1224) 
s_add_i32 s49, s25, s24 // (op 1224) 


s_and_b32 s24, s1, s49 // (op 1224) 


s_add_i32 s1, s23, s24 // (op 1224) 


s_ashr_i32 s23, s1, s22 // (op 1224) 


s_lshl_b32 s1, s48, 24 // (op 1224) 
s_ashr_i32 s22, s1, 31 // (op 1224) 

s_xor_b32 s1, s23, s22 // (op 1224) 

s_sub_i32 s23, s1, s22 // (op 1224) 


s_mul_i32 s1, s47, s15 // (op 1224) 
s_add_u32 s24, s20, 16 // (op 1224) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1224) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1224) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1224) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1224) 

s_sub_i32 s22, s24, s1 // (op 1224) 

s_mul_i32 s24, s23, s22 // (op 1224) 


s_sub_u32 s25, s19, s24 // (op 1224) 


s_add_u32 s19, s25, s1 // (op 1224) 


s_cmp_eq_i32 s18, 0 // (op 1224) 
s_mov_b32 s1, scc // (op 1224) 

s_cmp_eq_i32 s1, 0 // (op 1224) 
s_cselect_b32 s18, s19, s0 // (op 1224) 



v_mul_lo_u32 v16, v15, s26 // (op 1224) 

v_lshl_add_u32 v15, s18, 4, v16 // (op 1224) 


s_mul_i32 s1, s26, s31 // (op 1226) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1226) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1226) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1226) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1226) most significant: sum
s_add_u32 s1, s1, s19 // (op 1226) most significant: sum


s_mov_b32 s52, s28 // (op 1226) reg expression
s_mov_b32 s53, s29 // (op 1226) reg expression
s_mov_b32 s54, s0 // (op 1226) reg expression
s_mov_b32 s55, 131072 // (op 1226) call()

s_mul_hi_i32 s0, s16, s10 // (op 1269) 
v_lshrrev_b32 v16, 3, v1 // (op 1269) 
s_add_i32 s1, s0, s16 // (op 1269) 

v_lshlrev_b32 v17, 2, v16 // (op 1269) 

s_ashr_i32 s0, s1, 31 // (op 1269) 
v_and_b32 v16, -8, v1 // (op 1269) 
s_cmp_eq_i32 s10, 0 // (op 1269) 
s_mov_b32 s18, scc // (op 1269) 
v_sub_u32 v18, v1, v16 // (op 1269) 

s_cmp_eq_i32 s18, 0 // (op 1269) 
s_cselect_b32 s19, 0, -1 // (op 1269) 

v_lshlrev_b32 v16, 5, v18 // (op 1269) 

v_mov_b32 v19, 0 // (op 1269) convert
v_mov_b32 v18, v16 // (op 1269) convert

s_and_b32 s18, s11, 31 // (op 1269) 
s_lshl_b32 s22, 1, s18 // (op 1269) 
s_add_i32 s23, s22, s19 // (op 1269) 


s_and_b32 s19, s0, s23 // (op 1269) 


s_add_i32 s0, s1, s19 // (op 1269) 


s_ashr_i32 s1, s0, s18 // (op 1269) 


s_lshl_b32 s0, s11, 24 // (op 1269) 
s_ashr_i32 s18, s0, 31 // (op 1269) 

s_xor_b32 s0, s1, s18 // (op 1269) 

s_sub_i32 s1, s0, s18 // (op 1269) 


s_mul_hi_i32 s0, s1, s12 // (op 1269) 
s_add_i32 s18, s0, s1 // (op 1269) 

s_ashr_i32 s0, s18, 31 // (op 1269) 
s_cmp_eq_i32 s12, 0 // (op 1269) 
s_mov_b32 s19, scc // (op 1269) 
s_cmp_eq_i32 s19, 0 // (op 1269) 
s_cselect_b32 s22, 0, -1 // (op 1269) 

s_and_b32 s19, s13, 31 // (op 1269) 
s_lshl_b32 s23, 1, s19 // (op 1269) 
s_add_i32 s24, s23, s22 // (op 1269) 


s_and_b32 s25, s0, s24 // (op 1269) 


s_add_i32 s0, s18, s25 // (op 1269) 


s_ashr_i32 s18, s0, s19 // (op 1269) 


s_lshl_b32 s0, s13, 24 // (op 1269) 
s_ashr_i32 s19, s0, 31 // (op 1269) 

s_xor_b32 s0, s18, s19 // (op 1269) 

s_sub_i32 s18, s0, s19 // (op 1269) 


s_add_u32 s50, s6, 64 // (op 1269) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1269) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1269) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1269) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1269) 

s_mul_i32 s0, s18, s24 // (op 1269) 

s_sub_i32 s18, s1, s0 // (op 1269) 


s_mul_hi_i32 s19, s16, s9 // (op 1269) 
s_add_i32 s0, s19, s16 // (op 1269) 

s_ashr_i32 s1, s0, 31 // (op 1269) 
s_cmp_eq_i32 s9, 0 // (op 1269) 
s_mov_b32 s19, scc // (op 1269) 
s_cmp_eq_i32 s19, 0 // (op 1269) 
s_cselect_b32 s22, 0, -1 // (op 1269) 

s_and_b32 s19, s46, 31 // (op 1269) 
s_lshl_b32 s23, 1, s19 // (op 1269) 
s_add_i32 s49, s23, s22 // (op 1269) 


s_and_b32 s22, s1, s49 // (op 1269) 


s_add_i32 s1, s0, s22 // (op 1269) 


s_ashr_i32 s0, s1, s19 // (op 1269) 


s_lshl_b32 s1, s46, 24 // (op 1269) 
s_ashr_i32 s19, s1, 31 // (op 1269) 

s_xor_b32 s1, s0, s19 // (op 1269) 

s_sub_i32 s0, s1, s19 // (op 1269) 


s_mul_i32 s1, s15, s24 // (op 1269) 

s_mul_i32 s19, s8, s1 // (op 1269) 

s_mul_i32 s1, s0, s19 // (op 1269) 

s_sub_u32 s19, s16, s1 // (op 1269) 

s_mul_hi_i32 s1, s19, s14 // (op 1269) 
s_add_i32 s22, s1, s19 // (op 1269) 


s_ashr_i32 s1, s22, 31 // (op 1269) 
s_cmp_eq_i32 s14, 0 // (op 1269) 
s_mov_b32 s19, scc // (op 1269) 
s_cmp_eq_i32 s19, 0 // (op 1269) 
s_cselect_b32 s23, 0, -1 // (op 1269) 

s_and_b32 s19, s48, 31 // (op 1269) 
s_lshl_b32 s24, 1, s19 // (op 1269) 
s_add_i32 s25, s24, s23 // (op 1269) 


s_and_b32 s23, s1, s25 // (op 1269) 


s_add_i32 s1, s22, s23 // (op 1269) 


s_ashr_i32 s22, s1, s19 // (op 1269) 


s_lshl_b32 s1, s48, 24 // (op 1269) 
s_ashr_i32 s19, s1, 31 // (op 1269) 

s_xor_b32 s1, s22, s19 // (op 1269) 

s_sub_i32 s22, s1, s19 // (op 1269) 


s_cmp_eq_i32 s0, 0 // (op 1269) 
s_mov_b32 s1, scc // (op 1269) 

s_cmp_eq_i32 s1, 0 // (op 1269) 
s_cselect_b32 s0, s22, s18 // (op 1269) 



v_lshl_add_u32 v16, s0, 6, v17 // (op 1269) 


v_mul_lo_u32 v21, v16, s33 // (op 1269) most significant: low of low * high
v_mul_hi_u32 v17, v16, s32 // (op 1269) most significant: high of low * low
v_mul_lo_u32 v20, v16, s32 // (op 1269) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 1269) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 1269) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 1269) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 1269) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 1269) 

v_mov_b32 v16, v18 // (op 1269) convert

s_mul_i32 s1, s32, s7 // (op 1271) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 1271) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 1271) least significant
s_mul_hi_u32 s19, s32, s6 // (op 1271) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1271) most significant: sum
s_add_u32 s1, s1, s19 // (op 1271) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1271) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 1271) 

s_mov_b32 s64, s34 // (op 1271) reg expression
s_mov_b32 s65, s35 // (op 1271) reg expression
s_mov_b32 s66, s0 // (op 1271) reg expression
s_mov_b32 s67, 131072 // (op 1271) call()

s_mul_hi_i32 s0, s16, s10 // (op 1288) 
v_lshrrev_b32 v17, 3, v1 // (op 1288) 
s_add_i32 s1, s0, s16 // (op 1288) 

v_lshlrev_b32 v18, 2, v17 // (op 1288) 

s_ashr_i32 s0, s1, 31 // (op 1288) 
v_and_b32 v17, -8, v1 // (op 1288) 
s_cmp_eq_i32 s10, 0 // (op 1288) 
s_mov_b32 s18, scc // (op 1288) 
v_sub_u32 v19, v1, v17 // (op 1288) 

s_cmp_eq_i32 s18, 0 // (op 1288) 
s_cselect_b32 s19, 0, -1 // (op 1288) 

v_lshlrev_b32 v17, 5, v19 // (op 1288) 

v_mov_b32 v21, 0 // (op 1288) convert
v_mov_b32 v20, v17 // (op 1288) convert

s_and_b32 s18, s11, 31 // (op 1288) 
s_lshl_b32 s22, 1, s18 // (op 1288) 
s_add_i32 s23, s22, s19 // (op 1288) 


s_and_b32 s19, s0, s23 // (op 1288) 


s_add_i32 s0, s1, s19 // (op 1288) 


s_ashr_i32 s1, s0, s18 // (op 1288) 


s_lshl_b32 s0, s11, 24 // (op 1288) 
s_ashr_i32 s18, s0, 31 // (op 1288) 

s_xor_b32 s0, s1, s18 // (op 1288) 

s_sub_i32 s1, s0, s18 // (op 1288) 


s_mul_hi_i32 s0, s1, s12 // (op 1288) 
s_add_i32 s18, s0, s1 // (op 1288) 

s_ashr_i32 s0, s18, 31 // (op 1288) 
s_cmp_eq_i32 s12, 0 // (op 1288) 
s_mov_b32 s19, scc // (op 1288) 
s_cmp_eq_i32 s19, 0 // (op 1288) 
s_cselect_b32 s22, 0, -1 // (op 1288) 

s_and_b32 s19, s13, 31 // (op 1288) 
s_lshl_b32 s23, 1, s19 // (op 1288) 
s_add_i32 s24, s23, s22 // (op 1288) 


s_and_b32 s25, s0, s24 // (op 1288) 


s_add_i32 s0, s18, s25 // (op 1288) 


s_ashr_i32 s18, s0, s19 // (op 1288) 


s_lshl_b32 s0, s13, 24 // (op 1288) 
s_ashr_i32 s19, s0, 31 // (op 1288) 

s_xor_b32 s0, s18, s19 // (op 1288) 

s_sub_i32 s18, s0, s19 // (op 1288) 


s_add_u32 s50, s6, 64 // (op 1288) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1288) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1288) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1288) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1288) 

s_mul_i32 s0, s18, s24 // (op 1288) 

s_sub_i32 s18, s1, s0 // (op 1288) 


s_mul_hi_i32 s19, s16, s9 // (op 1288) 
s_add_i32 s0, s19, s16 // (op 1288) 

s_ashr_i32 s1, s0, 31 // (op 1288) 
s_cmp_eq_i32 s9, 0 // (op 1288) 
s_mov_b32 s19, scc // (op 1288) 
s_cmp_eq_i32 s19, 0 // (op 1288) 
s_cselect_b32 s22, 0, -1 // (op 1288) 

s_and_b32 s19, s46, 31 // (op 1288) 
s_lshl_b32 s23, 1, s19 // (op 1288) 
s_add_i32 s49, s23, s22 // (op 1288) 


s_and_b32 s22, s1, s49 // (op 1288) 


s_add_i32 s1, s0, s22 // (op 1288) 


s_ashr_i32 s0, s1, s19 // (op 1288) 


s_lshl_b32 s1, s46, 24 // (op 1288) 
s_ashr_i32 s19, s1, 31 // (op 1288) 

s_xor_b32 s1, s0, s19 // (op 1288) 

s_sub_i32 s0, s1, s19 // (op 1288) 


s_mul_i32 s1, s15, s24 // (op 1288) 

s_mul_i32 s19, s8, s1 // (op 1288) 

s_mul_i32 s1, s0, s19 // (op 1288) 

s_sub_u32 s19, s16, s1 // (op 1288) 

s_mul_hi_i32 s1, s19, s14 // (op 1288) 
s_add_i32 s22, s1, s19 // (op 1288) 


s_ashr_i32 s1, s22, 31 // (op 1288) 
s_cmp_eq_i32 s14, 0 // (op 1288) 
s_mov_b32 s19, scc // (op 1288) 
s_cmp_eq_i32 s19, 0 // (op 1288) 
s_cselect_b32 s23, 0, -1 // (op 1288) 

s_and_b32 s19, s48, 31 // (op 1288) 
s_lshl_b32 s24, 1, s19 // (op 1288) 
s_add_i32 s25, s24, s23 // (op 1288) 


s_and_b32 s23, s1, s25 // (op 1288) 


s_add_i32 s1, s22, s23 // (op 1288) 


s_ashr_i32 s22, s1, s19 // (op 1288) 


s_lshl_b32 s1, s48, 24 // (op 1288) 
s_ashr_i32 s19, s1, 31 // (op 1288) 

s_xor_b32 s1, s22, s19 // (op 1288) 

s_sub_i32 s22, s1, s19 // (op 1288) 


s_cmp_eq_i32 s0, 0 // (op 1288) 
s_mov_b32 s1, scc // (op 1288) 

s_cmp_eq_i32 s1, 0 // (op 1288) 
s_cselect_b32 s0, s22, s18 // (op 1288) 



v_lshl_add_u32 v17, s0, 6, v18 // (op 1288) 


v_mul_lo_u32 v19, v17, s33 // (op 1288) most significant: low of low * high
v_mul_hi_u32 v22, v17, s32 // (op 1288) most significant: high of low * low
v_mul_lo_u32 v18, v17, s32 // (op 1288) least significant: low of low * low
v_add_u32 v19, v19, v22 // (op 1288) most significant: sum


v_add_co_u32 v22, s[0:1], v20, v18 // (op 1288) least significant half
v_addc_co_u32 v23, s[0:1], v21, v19, s[0:1] // (op 1288) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 1288) 


v_lshrrev_b64 v[20:21], 3, v[22:23] // (op 1288) 

v_mov_b32 v17, v20 // (op 1288) convert

s_mul_i32 s1, s32, s7 // (op 1290) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 1290) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 1290) least significant
s_mul_hi_u32 s19, s32, s6 // (op 1290) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1290) most significant: sum
s_add_u32 s1, s1, s19 // (op 1290) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1290) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 1290) 

s_mov_b32 s64, s34 // (op 1290) reg expression
s_mov_b32 s65, s35 // (op 1290) reg expression
s_mov_b32 s66, s0 // (op 1290) reg expression
s_mov_b32 s67, 131072 // (op 1290) call()

s_mul_hi_i32 s0, s16, s10 // (op 1333) 
v_and_b32 v18, -16, v1 // (op 1333) 
s_add_i32 s1, s0, s16 // (op 1333) 

v_sub_u32 v19, v1, v18 // (op 1333) 

s_ashr_i32 s0, s1, 31 // (op 1333) 
v_lshrrev_b32 v18, 4, v1 // (op 1333) 
s_cmp_eq_i32 s10, 0 // (op 1333) 
s_mov_b32 s18, scc // (op 1333) 
s_cmp_eq_i32 s18, 0 // (op 1333) 
s_cselect_b32 s19, 0, -1 // (op 1333) 

s_and_b32 s18, s11, 31 // (op 1333) 
s_lshl_b32 s22, 1, s18 // (op 1333) 
s_add_i32 s23, s22, s19 // (op 1333) 


s_and_b32 s19, s0, s23 // (op 1333) 


s_add_i32 s0, s1, s19 // (op 1333) 


s_ashr_i32 s1, s0, s18 // (op 1333) 


s_lshl_b32 s0, s11, 24 // (op 1333) 
s_ashr_i32 s18, s0, 31 // (op 1333) 

s_xor_b32 s0, s1, s18 // (op 1333) 

s_sub_i32 s1, s0, s18 // (op 1333) 


s_mul_hi_i32 s0, s1, s12 // (op 1333) 
s_add_i32 s18, s0, s1 // (op 1333) 

s_ashr_i32 s0, s18, 31 // (op 1333) 
s_cmp_eq_i32 s12, 0 // (op 1333) 
s_mov_b32 s19, scc // (op 1333) 
s_cmp_eq_i32 s19, 0 // (op 1333) 
s_cselect_b32 s22, 0, -1 // (op 1333) 

s_and_b32 s19, s13, 31 // (op 1333) 
s_lshl_b32 s23, 1, s19 // (op 1333) 
s_add_i32 s24, s23, s22 // (op 1333) 


s_and_b32 s25, s0, s24 // (op 1333) 


s_add_i32 s0, s18, s25 // (op 1333) 


s_ashr_i32 s18, s0, s19 // (op 1333) 


s_lshl_b32 s0, s13, 24 // (op 1333) 
s_ashr_i32 s19, s0, 31 // (op 1333) 

s_xor_b32 s0, s18, s19 // (op 1333) 

s_sub_i32 s18, s0, s19 // (op 1333) 


s_mul_i32 s0, s1, s15 // (op 1333) 

s_sub_u32 s1, s16, s0 // (op 1333) 

s_mul_i32 s0, s18, s15 // (op 1333) 
s_add_u32 s0, s0, s1 // (op 1333) 


s_mul_hi_i32 s1, s16, s9 // (op 1333) 
s_add_i32 s18, s1, s16 // (op 1333) 

s_ashr_i32 s1, s18, 31 // (op 1333) 
s_cmp_eq_i32 s9, 0 // (op 1333) 
s_mov_b32 s19, scc // (op 1333) 
s_cmp_eq_i32 s19, 0 // (op 1333) 
s_cselect_b32 s22, 0, -1 // (op 1333) 

s_and_b32 s19, s46, 31 // (op 1333) 
s_lshl_b32 s23, 1, s19 // (op 1333) 
s_add_i32 s24, s23, s22 // (op 1333) 


s_and_b32 s25, s1, s24 // (op 1333) 


s_add_i32 s1, s18, s25 // (op 1333) 


s_ashr_i32 s18, s1, s19 // (op 1333) 


s_lshl_b32 s1, s46, 24 // (op 1333) 
s_ashr_i32 s19, s1, 31 // (op 1333) 

s_xor_b32 s1, s18, s19 // (op 1333) 

s_sub_i32 s18, s1, s19 // (op 1333) 


s_add_u32 s50, s6, 64 // (op 1333) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1333) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1333) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1333) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1333) 

s_mul_i32 s1, s15, s24 // (op 1333) 

s_mul_i32 s19, s8, s1 // (op 1333) 

s_mul_i32 s1, s18, s19 // (op 1333) 

s_sub_u32 s19, s16, s1 // (op 1333) 

s_mov_b32 s1, s19 // (op 1333) 
s_mul_hi_i32 s22, s1, s14 // (op 1333) 
s_add_i32 s23, s22, s1 // (op 1333) 


s_ashr_i32 s1, s23, 31 // (op 1333) 
s_cmp_eq_i32 s14, 0 // (op 1333) 
s_mov_b32 s22, scc // (op 1333) 
s_cmp_eq_i32 s22, 0 // (op 1333) 
s_cselect_b32 s24, 0, -1 // (op 1333) 

s_and_b32 s22, s48, 31 // (op 1333) 
s_lshl_b32 s25, 1, s22 // (op 1333) 
s_add_i32 s49, s25, s24 // (op 1333) 


s_and_b32 s24, s1, s49 // (op 1333) 


s_add_i32 s1, s23, s24 // (op 1333) 


s_ashr_i32 s23, s1, s22 // (op 1333) 


s_lshl_b32 s1, s48, 24 // (op 1333) 
s_ashr_i32 s22, s1, 31 // (op 1333) 

s_xor_b32 s1, s23, s22 // (op 1333) 

s_sub_i32 s23, s1, s22 // (op 1333) 


s_mul_i32 s1, s47, s15 // (op 1333) 
s_add_u32 s24, s20, 16 // (op 1333) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1333) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1333) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1333) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1333) 

s_sub_i32 s22, s24, s1 // (op 1333) 

s_mul_i32 s24, s23, s22 // (op 1333) 


s_sub_u32 s25, s19, s24 // (op 1333) 


s_add_u32 s19, s25, s1 // (op 1333) 


s_cmp_eq_i32 s18, 0 // (op 1333) 
s_mov_b32 s1, scc // (op 1333) 

s_cmp_eq_i32 s1, 0 // (op 1333) 
s_cselect_b32 s18, s19, s0 // (op 1333) 



v_lshl_add_u32 v20, s18, 4, v19 // (op 1333) 


v_mul_lo_u32 v19, v18, s36 // (op 1333) 

v_add_u32 v18, v20, v19 // (op 1333) 


s_mul_i32 s1, s36, s41 // (op 1335) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1335) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1335) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1335) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1335) most significant: sum
s_add_u32 s1, s1, s19 // (op 1335) most significant: sum


s_mov_b32 s68, s38 // (op 1335) reg expression
s_mov_b32 s69, s39 // (op 1335) reg expression
s_mov_b32 s70, s0 // (op 1335) reg expression
s_mov_b32 s71, 131072 // (op 1335) call()

s_mul_hi_i32 s0, s16, s10 // (op 1352) 
v_and_b32 v19, -16, v1 // (op 1352) 
s_add_i32 s1, s0, s16 // (op 1352) 

v_sub_u32 v20, v1, v19 // (op 1352) 

s_ashr_i32 s0, s1, 31 // (op 1352) 
v_lshrrev_b32 v19, 4, v1 // (op 1352) 
s_cmp_eq_i32 s10, 0 // (op 1352) 
s_mov_b32 s18, scc // (op 1352) 
s_cmp_eq_i32 s18, 0 // (op 1352) 
s_cselect_b32 s19, 0, -1 // (op 1352) 

s_and_b32 s18, s11, 31 // (op 1352) 
s_lshl_b32 s22, 1, s18 // (op 1352) 
s_add_i32 s23, s22, s19 // (op 1352) 


s_and_b32 s19, s0, s23 // (op 1352) 


s_add_i32 s0, s1, s19 // (op 1352) 


s_ashr_i32 s1, s0, s18 // (op 1352) 


s_lshl_b32 s0, s11, 24 // (op 1352) 
s_ashr_i32 s18, s0, 31 // (op 1352) 

s_xor_b32 s0, s1, s18 // (op 1352) 

s_sub_i32 s1, s0, s18 // (op 1352) 


s_mul_hi_i32 s0, s1, s12 // (op 1352) 
s_add_i32 s18, s0, s1 // (op 1352) 

s_ashr_i32 s0, s18, 31 // (op 1352) 
s_cmp_eq_i32 s12, 0 // (op 1352) 
s_mov_b32 s19, scc // (op 1352) 
s_cmp_eq_i32 s19, 0 // (op 1352) 
s_cselect_b32 s22, 0, -1 // (op 1352) 

s_and_b32 s19, s13, 31 // (op 1352) 
s_lshl_b32 s23, 1, s19 // (op 1352) 
s_add_i32 s24, s23, s22 // (op 1352) 


s_and_b32 s25, s0, s24 // (op 1352) 


s_add_i32 s0, s18, s25 // (op 1352) 


s_ashr_i32 s18, s0, s19 // (op 1352) 


s_lshl_b32 s0, s13, 24 // (op 1352) 
s_ashr_i32 s19, s0, 31 // (op 1352) 

s_xor_b32 s0, s18, s19 // (op 1352) 

s_sub_i32 s18, s0, s19 // (op 1352) 


s_mul_i32 s0, s1, s15 // (op 1352) 

s_sub_u32 s1, s16, s0 // (op 1352) 

s_mul_i32 s0, s18, s15 // (op 1352) 
s_add_u32 s0, s0, s1 // (op 1352) 


s_mul_hi_i32 s1, s16, s9 // (op 1352) 
s_add_i32 s18, s1, s16 // (op 1352) 

s_ashr_i32 s1, s18, 31 // (op 1352) 
s_cmp_eq_i32 s9, 0 // (op 1352) 
s_mov_b32 s19, scc // (op 1352) 
s_cmp_eq_i32 s19, 0 // (op 1352) 
s_cselect_b32 s22, 0, -1 // (op 1352) 

s_and_b32 s19, s46, 31 // (op 1352) 
s_lshl_b32 s23, 1, s19 // (op 1352) 
s_add_i32 s24, s23, s22 // (op 1352) 


s_and_b32 s25, s1, s24 // (op 1352) 


s_add_i32 s1, s18, s25 // (op 1352) 


s_ashr_i32 s18, s1, s19 // (op 1352) 


s_lshl_b32 s1, s46, 24 // (op 1352) 
s_ashr_i32 s19, s1, 31 // (op 1352) 

s_xor_b32 s1, s18, s19 // (op 1352) 

s_sub_i32 s18, s1, s19 // (op 1352) 


s_add_u32 s50, s6, 64 // (op 1352) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1352) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1352) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1352) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1352) 

s_mul_i32 s1, s15, s24 // (op 1352) 

s_mul_i32 s19, s8, s1 // (op 1352) 

s_mul_i32 s1, s18, s19 // (op 1352) 

s_sub_u32 s19, s16, s1 // (op 1352) 

s_mov_b32 s1, s19 // (op 1352) 
s_mul_hi_i32 s22, s1, s14 // (op 1352) 
s_add_i32 s23, s22, s1 // (op 1352) 


s_ashr_i32 s1, s23, 31 // (op 1352) 
s_cmp_eq_i32 s14, 0 // (op 1352) 
s_mov_b32 s22, scc // (op 1352) 
s_cmp_eq_i32 s22, 0 // (op 1352) 
s_cselect_b32 s24, 0, -1 // (op 1352) 

s_and_b32 s22, s48, 31 // (op 1352) 
s_lshl_b32 s25, 1, s22 // (op 1352) 
s_add_i32 s49, s25, s24 // (op 1352) 


s_and_b32 s24, s1, s49 // (op 1352) 


s_add_i32 s1, s23, s24 // (op 1352) 


s_ashr_i32 s23, s1, s22 // (op 1352) 


s_lshl_b32 s1, s48, 24 // (op 1352) 
s_ashr_i32 s22, s1, 31 // (op 1352) 

s_xor_b32 s1, s23, s22 // (op 1352) 

s_sub_i32 s23, s1, s22 // (op 1352) 


s_mul_i32 s1, s47, s15 // (op 1352) 
s_add_u32 s24, s20, 16 // (op 1352) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1352) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1352) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1352) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1352) 

s_sub_i32 s22, s24, s1 // (op 1352) 

s_mul_i32 s24, s23, s22 // (op 1352) 


s_sub_u32 s25, s19, s24 // (op 1352) 


s_add_u32 s19, s25, s1 // (op 1352) 


s_cmp_eq_i32 s18, 0 // (op 1352) 
s_mov_b32 s1, scc // (op 1352) 

s_cmp_eq_i32 s1, 0 // (op 1352) 
s_cselect_b32 s18, s19, s0 // (op 1352) 



v_lshl_add_u32 v21, s18, 4, v20 // (op 1352) 


v_mul_lo_u32 v20, v19, s36 // (op 1352) 

v_add_u32 v19, v21, v20 // (op 1352) 


s_mul_i32 s1, s36, s41 // (op 1354) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1354) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1354) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1354) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1354) most significant: sum
s_add_u32 s1, s1, s19 // (op 1354) most significant: sum


s_mov_b32 s68, s38 // (op 1354) reg expression
s_mov_b32 s69, s39 // (op 1354) reg expression
s_mov_b32 s70, s0 // (op 1354) reg expression
s_mov_b32 s71, 131072 // (op 1354) call()

s_mul_hi_i32 s0, s16, s10 // (op 1397) 
v_lshrrev_b32 v20, 1, v1 // (op 1397) 
s_add_i32 s1, s0, s16 // (op 1397) 

v_and_b32 v21, -2, v1 // (op 1397) 
s_ashr_i32 s0, s1, 31 // (op 1397) 
v_sub_u32 v22, v1, v21 // (op 1397) 

s_cmp_eq_i32 s10, 0 // (op 1397) 
s_mov_b32 s18, scc // (op 1397) 
s_cmp_eq_i32 s18, 0 // (op 1397) 
s_cselect_b32 s19, 0, -1 // (op 1397) 

s_and_b32 s18, s11, 31 // (op 1397) 
s_lshl_b32 s22, 1, s18 // (op 1397) 
s_add_i32 s23, s22, s19 // (op 1397) 


s_and_b32 s19, s0, s23 // (op 1397) 


s_add_i32 s0, s1, s19 // (op 1397) 


s_ashr_i32 s1, s0, s18 // (op 1397) 


s_lshl_b32 s0, s11, 24 // (op 1397) 
s_ashr_i32 s18, s0, 31 // (op 1397) 

s_xor_b32 s0, s1, s18 // (op 1397) 

s_sub_i32 s1, s0, s18 // (op 1397) 


s_mul_hi_i32 s0, s1, s12 // (op 1397) 
s_add_i32 s18, s0, s1 // (op 1397) 

s_ashr_i32 s0, s18, 31 // (op 1397) 
s_cmp_eq_i32 s12, 0 // (op 1397) 
s_mov_b32 s19, scc // (op 1397) 
s_cmp_eq_i32 s19, 0 // (op 1397) 
s_cselect_b32 s22, 0, -1 // (op 1397) 

s_and_b32 s19, s13, 31 // (op 1397) 
s_lshl_b32 s23, 1, s19 // (op 1397) 
s_add_i32 s24, s23, s22 // (op 1397) 


s_and_b32 s25, s0, s24 // (op 1397) 


s_add_i32 s0, s18, s25 // (op 1397) 


s_ashr_i32 s18, s0, s19 // (op 1397) 


s_lshl_b32 s0, s13, 24 // (op 1397) 
s_ashr_i32 s19, s0, 31 // (op 1397) 

s_xor_b32 s0, s18, s19 // (op 1397) 

s_sub_i32 s18, s0, s19 // (op 1397) 


s_add_u32 s50, s6, 64 // (op 1397) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1397) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1397) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1397) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1397) 

s_mul_i32 s0, s18, s24 // (op 1397) 

s_sub_i32 s18, s1, s0 // (op 1397) 


s_mul_hi_i32 s19, s16, s9 // (op 1397) 
s_add_i32 s0, s19, s16 // (op 1397) 

s_ashr_i32 s1, s0, 31 // (op 1397) 
s_cmp_eq_i32 s9, 0 // (op 1397) 
s_mov_b32 s19, scc // (op 1397) 
s_cmp_eq_i32 s19, 0 // (op 1397) 
s_cselect_b32 s22, 0, -1 // (op 1397) 

s_and_b32 s19, s46, 31 // (op 1397) 
s_lshl_b32 s23, 1, s19 // (op 1397) 
s_add_i32 s49, s23, s22 // (op 1397) 


s_and_b32 s22, s1, s49 // (op 1397) 


s_add_i32 s1, s0, s22 // (op 1397) 


s_ashr_i32 s0, s1, s19 // (op 1397) 


s_lshl_b32 s1, s46, 24 // (op 1397) 
s_ashr_i32 s19, s1, 31 // (op 1397) 

s_xor_b32 s1, s0, s19 // (op 1397) 

s_sub_i32 s0, s1, s19 // (op 1397) 


s_mul_i32 s1, s15, s24 // (op 1397) 

s_mul_i32 s19, s8, s1 // (op 1397) 

s_mul_i32 s1, s0, s19 // (op 1397) 

s_sub_u32 s19, s16, s1 // (op 1397) 

s_mul_hi_i32 s1, s19, s14 // (op 1397) 
s_add_i32 s22, s1, s19 // (op 1397) 


s_ashr_i32 s1, s22, 31 // (op 1397) 
s_cmp_eq_i32 s14, 0 // (op 1397) 
s_mov_b32 s19, scc // (op 1397) 
s_cmp_eq_i32 s19, 0 // (op 1397) 
s_cselect_b32 s23, 0, -1 // (op 1397) 

s_and_b32 s19, s48, 31 // (op 1397) 
s_lshl_b32 s24, 1, s19 // (op 1397) 
s_add_i32 s25, s24, s23 // (op 1397) 


s_and_b32 s23, s1, s25 // (op 1397) 


s_add_i32 s1, s22, s23 // (op 1397) 


s_ashr_i32 s22, s1, s19 // (op 1397) 


s_lshl_b32 s1, s48, 24 // (op 1397) 
s_ashr_i32 s19, s1, 31 // (op 1397) 

s_xor_b32 s1, s22, s19 // (op 1397) 

s_sub_i32 s22, s1, s19 // (op 1397) 


s_cmp_eq_i32 s0, 0 // (op 1397) 
s_mov_b32 s1, scc // (op 1397) 

s_cmp_eq_i32 s1, 0 // (op 1397) 
s_cselect_b32 s0, s22, s18 // (op 1397) 



v_lshl_add_u32 v21, s0, 6, v20 // (op 1397) 


v_mul_lo_u32 v20, v21, s42 // (op 1397) 

v_lshl_add_u32 v21, v22, 2, v20 // (op 1397) 


s_mul_i32 s1, s42, s7 // (op 1399) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1399) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1399) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1399) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1399) most significant: sum
s_add_u32 s1, s1, s19 // (op 1399) most significant: sum


s_mov_b32 s72, s44 // (op 1399) reg expression
s_mov_b32 s73, s45 // (op 1399) reg expression
s_mov_b32 s74, s0 // (op 1399) reg expression
s_mov_b32 s75, 131072 // (op 1399) call()

s_mul_hi_i32 s0, s16, s10 // (op 1416) 
v_lshrrev_b32 v20, 1, v1 // (op 1416) 
s_add_i32 s1, s0, s16 // (op 1416) 

v_and_b32 v22, -2, v1 // (op 1416) 
s_ashr_i32 s0, s1, 31 // (op 1416) 
v_sub_u32 v23, v1, v22 // (op 1416) 

s_cmp_eq_i32 s10, 0 // (op 1416) 
s_mov_b32 s18, scc // (op 1416) 
s_cmp_eq_i32 s18, 0 // (op 1416) 
s_cselect_b32 s19, 0, -1 // (op 1416) 

s_and_b32 s18, s11, 31 // (op 1416) 
s_lshl_b32 s22, 1, s18 // (op 1416) 
s_add_i32 s23, s22, s19 // (op 1416) 


s_and_b32 s19, s0, s23 // (op 1416) 


s_add_i32 s0, s1, s19 // (op 1416) 


s_ashr_i32 s1, s0, s18 // (op 1416) 


s_lshl_b32 s0, s11, 24 // (op 1416) 
s_ashr_i32 s18, s0, 31 // (op 1416) 

s_xor_b32 s0, s1, s18 // (op 1416) 

s_sub_i32 s1, s0, s18 // (op 1416) 


s_mul_hi_i32 s0, s1, s12 // (op 1416) 
s_add_i32 s18, s0, s1 // (op 1416) 

s_ashr_i32 s0, s18, 31 // (op 1416) 
s_cmp_eq_i32 s12, 0 // (op 1416) 
s_mov_b32 s19, scc // (op 1416) 
s_cmp_eq_i32 s19, 0 // (op 1416) 
s_cselect_b32 s22, 0, -1 // (op 1416) 

s_and_b32 s19, s13, 31 // (op 1416) 
s_lshl_b32 s23, 1, s19 // (op 1416) 
s_add_i32 s24, s23, s22 // (op 1416) 


s_and_b32 s25, s0, s24 // (op 1416) 


s_add_i32 s0, s18, s25 // (op 1416) 


s_ashr_i32 s18, s0, s19 // (op 1416) 


s_lshl_b32 s0, s13, 24 // (op 1416) 
s_ashr_i32 s19, s0, 31 // (op 1416) 

s_xor_b32 s0, s18, s19 // (op 1416) 

s_sub_i32 s18, s0, s19 // (op 1416) 


s_add_u32 s50, s6, 64 // (op 1416) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1416) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1416) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1416) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1416) 

s_mul_i32 s0, s18, s24 // (op 1416) 

s_sub_i32 s18, s1, s0 // (op 1416) 


s_mul_hi_i32 s19, s16, s9 // (op 1416) 
s_add_i32 s0, s19, s16 // (op 1416) 

s_ashr_i32 s1, s0, 31 // (op 1416) 
s_cmp_eq_i32 s9, 0 // (op 1416) 
s_mov_b32 s19, scc // (op 1416) 
s_cmp_eq_i32 s19, 0 // (op 1416) 
s_cselect_b32 s22, 0, -1 // (op 1416) 

s_and_b32 s19, s46, 31 // (op 1416) 
s_lshl_b32 s23, 1, s19 // (op 1416) 
s_add_i32 s49, s23, s22 // (op 1416) 


s_and_b32 s22, s1, s49 // (op 1416) 


s_add_i32 s1, s0, s22 // (op 1416) 


s_ashr_i32 s0, s1, s19 // (op 1416) 


s_lshl_b32 s1, s46, 24 // (op 1416) 
s_ashr_i32 s19, s1, 31 // (op 1416) 

s_xor_b32 s1, s0, s19 // (op 1416) 

s_sub_i32 s0, s1, s19 // (op 1416) 


s_mul_i32 s1, s15, s24 // (op 1416) 

s_mul_i32 s19, s8, s1 // (op 1416) 

s_mul_i32 s1, s0, s19 // (op 1416) 

s_sub_u32 s19, s16, s1 // (op 1416) 

s_mul_hi_i32 s1, s19, s14 // (op 1416) 
s_add_i32 s22, s1, s19 // (op 1416) 


s_ashr_i32 s1, s22, 31 // (op 1416) 
s_cmp_eq_i32 s14, 0 // (op 1416) 
s_mov_b32 s19, scc // (op 1416) 
s_cmp_eq_i32 s19, 0 // (op 1416) 
s_cselect_b32 s23, 0, -1 // (op 1416) 

s_and_b32 s19, s48, 31 // (op 1416) 
s_lshl_b32 s24, 1, s19 // (op 1416) 
s_add_i32 s25, s24, s23 // (op 1416) 


s_and_b32 s23, s1, s25 // (op 1416) 


s_add_i32 s1, s22, s23 // (op 1416) 


s_ashr_i32 s22, s1, s19 // (op 1416) 


s_lshl_b32 s1, s48, 24 // (op 1416) 
s_ashr_i32 s19, s1, 31 // (op 1416) 

s_xor_b32 s1, s22, s19 // (op 1416) 

s_sub_i32 s22, s1, s19 // (op 1416) 


s_cmp_eq_i32 s0, 0 // (op 1416) 
s_mov_b32 s1, scc // (op 1416) 

s_cmp_eq_i32 s1, 0 // (op 1416) 
s_cselect_b32 s0, s22, s18 // (op 1416) 



v_lshl_add_u32 v22, s0, 6, v20 // (op 1416) 


v_mul_lo_u32 v20, v22, s42 // (op 1416) 

v_lshl_add_u32 v22, v23, 2, v20 // (op 1416) 


s_mul_i32 s1, s42, s7 // (op 1418) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1418) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1418) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1418) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1418) most significant: sum
s_add_u32 s1, s1, s19 // (op 1418) most significant: sum


s_mov_b32 s72, s44 // (op 1418) reg expression
s_mov_b32 s73, s45 // (op 1418) reg expression
s_mov_b32 s74, s0 // (op 1418) reg expression
s_mov_b32 s75, 131072 // (op 1418) call()


s_lshl_b64 s[0:1], s[32:33], 2 // (op 1029) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1029) 


v_mov_b32 v20, v16 // (op 1029) 
buffer_load_dwordx4 v[26:29], v20, s[64:67], 0 offen // (op 1029) Load value
v_add_u32 v20, v20, v24 // (op 1029) 
buffer_load_dwordx4 v[30:33], v20, s[64:67], 0 offen // (op 1029) Load value
v_add_u32 v20, v20, v24 // (op 1029) 
buffer_load_dwordx4 v[34:37], v20, s[64:67], 0 offen // (op 1029) Load value
v_add_u32 v20, v20, v24 // (op 1029) 
buffer_load_dwordx4 v[38:41], v20, s[64:67], 0 offen // (op 1029) Load value





buffer_load_dword v20, v21, s[72:75], 0 offen // (op 1032) Load value

v_mov_b64 v[24:25], s[26:27] // (op 1035) convert

v_mov_b32 v23, v14 // (op 1035) reg expression
buffer_load_dwordx4 v[42:45], v23, s[52:55], 0 offen // (op 1035) Load value
v_add_u32 v23, v23, v24 // (op 1035) 
buffer_load_dwordx4 v[46:49], v23, s[52:55], 0 offen // (op 1035) Load value




v_mov_b32 v23, v18 // (op 1038) reg expression

buffer_load_ubyte v24, v23, s[68:71], 0 offen offset: 0 // (op 1038) Load value


s_lshl_b64 s[0:1], s[32:33], 2 // (op 1041) 
v_lshrrev_b64 v[50:51], 3, s[0:1] // (op 1041) 


v_mov_b32 v23, v16 // (op 1041) 
buffer_load_dwordx4 v[52:55], v23, s[64:67], 0 offen offset:128 // (op 1041) Load value
v_add_u32 v23, v23, v50 // (op 1041) 
buffer_load_dwordx4 v[56:59], v23, s[64:67], 0 offen offset:128 // (op 1041) Load value
v_add_u32 v23, v23, v50 // (op 1041) 
buffer_load_dwordx4 v[60:63], v23, s[64:67], 0 offen offset:128 // (op 1041) Load value
v_add_u32 v23, v23, v50 // (op 1041) 
buffer_load_dwordx4 v[64:67], v23, s[64:67], 0 offen offset:128 // (op 1041) Load value






buffer_load_dword v16, v21, s[72:75], 0 offen offset:8 // (op 1044) Load value


v_mov_b64 v[50:51], s[26:27] // (op 1047) convert

s_lshl_b32 s0, s26, 8 // (op 1047) 
v_add_u32 v21, v14, s0 // (op 1047) 

buffer_load_dwordx4 v[68:71], v21, s[52:55], 0 offen // (op 1047) Load value
v_add_u32 v21, v21, v50 // (op 1047) 
buffer_load_dwordx4 v[72:75], v21, s[52:55], 0 offen // (op 1047) Load value





s_lshl_b32 s0, s36, 3 // (op 1050) 
v_add_u32 v14, v18, s0 // (op 1050) 


buffer_load_ubyte v21, v14, s[68:71], 0 offen offset: 0 // (op 1050) Load value


v_and_b32 v14, -8, v1 // (op 1536) 
v_sub_u32 v18, v1, v14 // (op 1536) 

v_lshlrev_b32 v14, 5, v18 // (op 1536) 

v_lshrrev_b32 v18, 3, v1 // (op 1536) 
v_lshl_add_u32 v23, v18, 10, v14 // (op 1536) 


v_lshlrev_b32 v14, 2, v23 // (op 1536) 

v_lshrrev_b32 v18, 3, v14 // (op 1536) 




s_waitcnt vmcnt(15)
ds_write_b128 v18, v[26:29] // (op 1052) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v18, v[30:33] offset:128 // (op 1052) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v18, v[34:37] offset:256 // (op 1052) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v18, v[38:41] offset:384 // (op 1052) Store local data 

v_and_b32 v14, -2, v1 // (op 1647) 
v_sub_u32 v18, v1, v14 // (op 1647) 

v_lshlrev_b32 v14, 2, v18 // (op 1647) 

v_lshrrev_b32 v18, 1, v1 // (op 1647) 
v_lshl_add_u32 v23, v18, 3, v14 // (op 1647) 





s_waitcnt vmcnt(11)
ds_write_b32 v23, v20 offset:8192 // (op 1053) Store local data 

v_lshlrev_b32 v14, 5, v1 // (op 1483) 



s_waitcnt vmcnt(10)
ds_write_b128 v14, v[42:45] offset:8704 // (op 1054) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v14, v[46:49] offset:8720 // (op 1054) Store local data 

v_and_b32 v14, -16, v1 // (op 1594) 
v_sub_u32 v18, v1, v14 // (op 1594) 

v_and_b32 v14, 4294967280, v1 // (op 1594) 
v_add_u32 v23, v14, v18 // (op 1594) 





s_waitcnt vmcnt(8)
ds_write_b8 v23, v24 offset:12800 // (op 1055) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1858) 



ds_read_b128 v[76:79], v9 // (op 1073) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1899) 



ds_read_u8 v14, v5 offset:8192 // (op 1086) Load local data 



ds_read_b128 v[80:83], v9 offset:4096 // (op 1078) Load local data 



ds_read_u8 v18, v5 offset:8448 // (op 1091) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1839) 



ds_read_b64_tr_b8 v[84:85], v0 offset:8704 // (op 1069) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v0 offset:8832 // (op 1069) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v0 offset:9728 // (op 1069) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v0 offset:9856 // (op 1069) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1879) 


ds_read_u8 v23, v4 offset:12800 // (op 1082) Load local data 
v_lshlrev_b32 v25, 5, v1 // (op 1454) 
v_and_b32 v50, -8, v1 // (op 1507) 
v_sub_u32 v51, v1, v50 // (op 1507) 

v_lshlrev_b32 v50, 5, v51 // (op 1507) 

v_lshrrev_b32 v51, 3, v1 // (op 1507) 
v_lshl_add_u32 v92, v51, 10, v50 // (op 1507) 


v_lshlrev_b32 v50, 2, v92 // (op 1507) 

v_lshrrev_b32 v51, 3, v50 // (op 1507) 

v_and_b32 v50, -16, v1 // (op 1565) 
v_sub_u32 v92, v1, v50 // (op 1565) 

v_and_b32 v50, 4294967280, v1 // (op 1565) 
v_add_u32 v93, v50, v92 // (op 1565) 


v_and_b32 v50, -2, v1 // (op 1618) 
v_sub_u32 v92, v1, v50 // (op 1618) 

v_lshlrev_b32 v50, 2, v92 // (op 1618) 

v_lshrrev_b32 v92, 1, v1 // (op 1618) 
v_lshl_add_u32 v94, v92, 3, v50 // (op 1618) 


v_lshlrev_b32 v50, 5, v1 // (op 1694) 
v_and_b32 v92, -8, v1 // (op 1718) 
v_sub_u32 v95, v1, v92 // (op 1718) 

v_lshlrev_b32 v92, 5, v95 // (op 1718) 

v_lshrrev_b32 v95, 3, v1 // (op 1718) 
v_lshl_add_u32 v96, v95, 10, v92 // (op 1718) 


v_lshlrev_b32 v92, 2, v96 // (op 1718) 

v_lshrrev_b32 v95, 3, v92 // (op 1718) 

v_and_b32 v92, -16, v1 // (op 1747) 
v_sub_u32 v96, v1, v92 // (op 1747) 

v_and_b32 v92, 4294967280, v1 // (op 1747) 
v_add_u32 v97, v92, v96 // (op 1747) 


v_and_b32 v92, -2, v1 // (op 1771) 
v_sub_u32 v96, v1, v92 // (op 1771) 

v_lshlrev_b32 v92, 2, v96 // (op 1771) 

v_lshrrev_b32 v96, 1, v1 // (op 1771) 
v_lshl_add_u32 v98, v96, 3, v92 // (op 1771) 


s_mov_b32 s0, 0 // (op 400) call()
s_ashr_i64 s[18:19], s[30:31], 8 // (op 399) 
s_mov_b32 s1, s18 // (op 399) convert

s_and_b32 s18, s1, 4294967294 // (op 399) 

s_cmp_lt_u32 s0, s18 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_399 // (op 399) Condition: Top (jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_399 if false)
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_399:

s_add_u32 s0, s0, 2 // (op 401) 
s_lshl_b64 s[18:19], s[32:33], 2 // (op 10) 
v_lshrrev_b64 v[100:101], 3, s[18:19] // (op 10) 


v_mov_b32 v92, v17 // (op 10) 
v_mov_b32 v96, v92 // (op 10) 
buffer_load_dwordx4 v[26:29], v96, s[64:67], 0 offen offset:256 // (op 10) Load value
v_add_u32 v96, v96, v100 // (op 10) 
buffer_load_dwordx4 v[30:33], v96, s[64:67], 0 offen offset:256 // (op 10) Load value
v_add_u32 v96, v96, v100 // (op 10) 
buffer_load_dwordx4 v[34:37], v96, s[64:67], 0 offen offset:256 // (op 10) Load value
v_add_u32 v96, v96, v100 // (op 10) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:91], v[76:79], a[0:3], v23, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 88) 
buffer_load_dwordx4 v[38:41], v96, s[64:67], 0 offen offset:256 // (op 10) Load value




s_barrier  // (op 1848) 
v_mov_b32 v96, v22 // (op 25) 




buffer_load_dword v20, v96, s[72:75], 0 offen offset:16 // (op 25) Load value
ds_read_b64_tr_b8 v[100:101], v0 offset:10752 // (op 91) Transpose load local data 

ds_read_b64_tr_b8 v[102:103], v0 offset:10880 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v0 offset:11776 // (op 91) Transpose load local data 
v_mov_b64 v[108:109], s[26:27] // (op 4) convert

v_mov_b32 v99, v15 // (op 4) 
s_lshl_b32 s1, s26, 9 // (op 4) 
v_add_u32 v110, v99, s1 // (op 4) 

buffer_load_dwordx4 v[42:45], v110, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v110, v110, v108 // (op 4) 
ds_read_b64_tr_b8 v[106:107], v0 offset:11904 // (op 91) Transpose load local data 
buffer_load_dwordx4 v[46:49], v110, s[52:55], 0 offen // (op 4) Load value




s_barrier  // (op 1868) 
v_mov_b32 v108, v19 // (op 17) 



ds_read_b128 v[76:79], v9 offset:64 // (op 95) Load local data 
s_lshl_b32 s1, s36, 4 // (op 17) 
s_barrier  // (op 1889) 
v_add_u32 v109, v108, s1 // (op 17) 




buffer_load_ubyte v24, v109, s[68:71], 0 offen offset: 0 // (op 17) Load value

ds_read_u8 v109, v4 offset:12864 // (op 99) Load local data 
s_barrier  // (op 1929) 
s_barrier  // (op 1909) 






ds_read_u8 v14, v5 offset:8196 // (op 103) Load local data 
s_waitcnt vmcnt(15)
ds_write_b128 v95, v[52:55] offset:12928 // (op 302) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[100:107], v[76:79], a[0:3], v109, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 105) 
s_waitcnt vmcnt(14)
ds_write_b128 v95, v[56:59] offset:13056 // (op 302) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v95, v[60:63] offset:13184 // (op 302) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[84:91], v[80:83], a[4:7], v23, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 447) 
s_waitcnt vmcnt(12)
ds_write_b128 v95, v[64:67] offset:13312 // (op 302) Store local data 



ds_read_b128 v[80:83], v9 offset:4160 // (op 451) Load local data 



s_barrier  // (op 1941) 
ds_read_u8 v18, v5 offset:8452 // (op 455) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[100:107], v[80:83], a[4:7], v109, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 456) 
s_waitcnt vmcnt(11)
ds_write_b32 v98, v16 offset:21120 // (op 304) Store local data 


s_barrier  // (op 1922) 



s_waitcnt vmcnt(10)
ds_write_b128 v50, v[68:71] offset:21632 // (op 283) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v50, v[72:75] offset:21648 // (op 283) Store local data 
s_barrier  // (op 1935) 



s_waitcnt vmcnt(8)
ds_write_b8 v97, v21 offset:25728 // (op 303) Store local data 
s_barrier  // (op 1925) 



ds_read_b128 v[76:79], v13 offset:12928 // (op 287) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1938) 



ds_read_u8 v14, v12 offset:21120 // (op 291) Load local data 



ds_read_b128 v[100:103], v13 offset:17024 // (op 470) Load local data 



ds_read_u8 v104, v12 offset:21376 // (op 474) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1919) 



ds_read_b64_tr_b8 v[84:85], v3 offset:21632 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v3 offset:21760 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v3 offset:22656 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v3 offset:22784 // (op 285) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1932) 


ds_read_u8 v23, v6 offset:25728 // (op 289) Load local data 

s_lshl_b64 s[18:19], s[32:33], 2 // (op 305) 
v_lshrrev_b64 v[106:107], 3, s[18:19] // (op 305) 


v_mov_b32 v105, v92 // (op 305) 
buffer_load_dwordx4 v[52:55], v105, s[64:67], 0 offen offset:384 // (op 305) Load value
v_add_u32 v105, v105, v106 // (op 305) 
buffer_load_dwordx4 v[56:59], v105, s[64:67], 0 offen offset:384 // (op 305) Load value
v_add_u32 v105, v105, v106 // (op 305) 
buffer_load_dwordx4 v[60:63], v105, s[64:67], 0 offen offset:384 // (op 305) Load value
v_add_u32 v105, v105, v106 // (op 305) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:91], v[76:79], a[0:3], v23, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 292) 
buffer_load_dwordx4 v[64:67], v105, s[64:67], 0 offen offset:384 // (op 305) Load value









v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[84:91], v[100:103], a[4:7], v23, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 457) 
buffer_load_dword v16, v96, s[72:75], 0 offen offset:24 // (op 307) Load value



v_add_u32 v17, 256, v17 // (op 1280) 

v_add_u32 v22, 16, v22 // (op 1408) 
v_mov_b64 v[100:101], s[26:27] // (op 282) convert

s_lshl_b32 s1, s26, 8 // (op 282) 
s_mul_i32 s18, 3, s1 // (op 282) 

v_add_u32 v92, v99, s18 // (op 282) 

buffer_load_dwordx4 v[68:71], v92, s[52:55], 0 offen // (op 282) Load value
v_add_u32 v92, v92, v100 // (op 282) 
ds_read_b64_tr_b8 v[110:111], v3 offset:23680 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[112:113], v3 offset:23808 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[114:115], v3 offset:24704 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[116:117], v3 offset:24832 // (op 294) Transpose load local data 
buffer_load_dwordx4 v[72:75], v92, s[52:55], 0 offen // (op 282) Load value








s_barrier  // (op 1986) 
s_lshl_b32 s1, s36, 3 // (op 306) 
s_mul_i32 s18, 3, s1 // (op 306) 

s_lshl_b32 s1, s26, 9 // (op 1216) 
v_add_u32 v15, v15, s1 // (op 1216) 

v_add_u32 v92, v108, s18 // (op 306) 


buffer_load_ubyte v21, v92, s[68:71], 0 offen offset: 0 // (op 306) Load value

ds_read_b128 v[104:107], v13 offset:12992 // (op 296) Load local data 



s_barrier  // (op 1861) 
ds_read_u8 v92, v6 offset:25792 // (op 298) Load local data 



s_lshl_b32 s1, s36, 4 // (op 1344) 



s_barrier  // (op 1990) 
v_add_u32 v19, v19, s1 // (op 1344) 

ds_read_u8 v96, v12 offset:21124 // (op 300) Load local data 
s_waitcnt vmcnt(15)
ds_write_b128 v51, v[26:29] // (op 13) Store local data 



ds_read_b128 v[100:103], v13 offset:17088 // (op 461) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[110:117], v[104:107], a[0:3], v92, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 301) 





s_barrier  // (op 2026) 
ds_read_u8 v96, v12 offset:21380 // (op 465) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v51, v[30:33] offset:128 // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v51, v[34:37] offset:256 // (op 13) Store local data 
s_barrier  // (op 2030) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[110:117], v[100:103], a[4:7], v92, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 466) 
s_waitcnt vmcnt(12)
ds_write_b128 v51, v[38:41] offset:384 // (op 13) Store local data 




s_barrier  // (op 1902) 



s_waitcnt vmcnt(11)
ds_write_b32 v94, v20 offset:8192 // (op 30) Store local data 
s_barrier  // (op 1842) 



s_waitcnt vmcnt(10)
ds_write_b128 v25, v[42:45] offset:8704 // (op 6) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v25, v[46:49] offset:8720 // (op 6) Store local data 
s_barrier  // (op 1882) 



s_waitcnt vmcnt(8)
ds_write_b8 v93, v24 offset:12800 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1871) 



ds_read_b128 v[76:79], v9 // (op 75) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1912) 



ds_read_u8 v14, v5 offset:8192 // (op 85) Load local data 



ds_read_b128 v[80:83], v9 offset:4096 // (op 442) Load local data 



ds_read_u8 v18, v5 offset:8448 // (op 446) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1851) 



ds_read_b64_tr_b8 v[84:85], v0 offset:8704 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v0 offset:8832 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v0 offset:9728 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v0 offset:9856 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1892) 


ds_read_u8 v23, v4 offset:12800 // (op 80) Load local data 
s_ashr_i64 s[18:19], s[30:31], 8 // (op 399) 
s_mov_b32 s1, s18 // (op 399) convert

s_and_b32 s18, s1, 4294967294 // (op 399) 

s_cmp_lt_u32 s0, s18 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_399 // (op 399) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_399 if true)
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_399:

















s_mul_hi_i32 s0, s16, s10 // (op 1178) 
v_lshlrev_b32 v15, 1, v1 // (op 1178) 
s_add_i32 s1, s0, s16 // (op 1178) 

s_ashr_i32 s0, s1, 31 // (op 1178) 
s_cmp_eq_i32 s10, 0 // (op 1178) 
s_mov_b32 s18, scc // (op 1178) 
s_cmp_eq_i32 s18, 0 // (op 1178) 
s_cselect_b32 s19, 0, -1 // (op 1178) 

s_and_b32 s18, s11, 31 // (op 1178) 
s_lshl_b32 s22, 1, s18 // (op 1178) 
s_add_i32 s23, s22, s19 // (op 1178) 


s_and_b32 s19, s0, s23 // (op 1178) 


s_add_i32 s0, s1, s19 // (op 1178) 


s_ashr_i32 s1, s0, s18 // (op 1178) 


s_lshl_b32 s0, s11, 24 // (op 1178) 
s_ashr_i32 s18, s0, 31 // (op 1178) 

s_xor_b32 s0, s1, s18 // (op 1178) 

s_sub_i32 s1, s0, s18 // (op 1178) 


s_mul_hi_i32 s0, s1, s12 // (op 1178) 
s_add_i32 s18, s0, s1 // (op 1178) 

s_ashr_i32 s0, s18, 31 // (op 1178) 
s_cmp_eq_i32 s12, 0 // (op 1178) 
s_mov_b32 s19, scc // (op 1178) 
s_cmp_eq_i32 s19, 0 // (op 1178) 
s_cselect_b32 s22, 0, -1 // (op 1178) 

s_and_b32 s19, s13, 31 // (op 1178) 
s_lshl_b32 s23, 1, s19 // (op 1178) 
s_add_i32 s24, s23, s22 // (op 1178) 


s_and_b32 s25, s0, s24 // (op 1178) 


s_add_i32 s0, s18, s25 // (op 1178) 


s_ashr_i32 s18, s0, s19 // (op 1178) 


s_lshl_b32 s0, s13, 24 // (op 1178) 
s_ashr_i32 s19, s0, 31 // (op 1178) 

s_xor_b32 s0, s18, s19 // (op 1178) 

s_sub_i32 s18, s0, s19 // (op 1178) 


s_mul_i32 s0, s1, s15 // (op 1178) 

s_sub_u32 s1, s16, s0 // (op 1178) 

s_mul_i32 s0, s18, s15 // (op 1178) 
s_add_u32 s0, s0, s1 // (op 1178) 


s_mul_hi_i32 s1, s16, s9 // (op 1178) 
s_add_i32 s18, s1, s16 // (op 1178) 

s_ashr_i32 s1, s18, 31 // (op 1178) 
s_cmp_eq_i32 s9, 0 // (op 1178) 
s_mov_b32 s19, scc // (op 1178) 
s_cmp_eq_i32 s19, 0 // (op 1178) 
s_cselect_b32 s22, 0, -1 // (op 1178) 

s_and_b32 s19, s46, 31 // (op 1178) 
s_lshl_b32 s23, 1, s19 // (op 1178) 
s_add_i32 s24, s23, s22 // (op 1178) 


s_and_b32 s25, s1, s24 // (op 1178) 


s_add_i32 s1, s18, s25 // (op 1178) 


s_ashr_i32 s18, s1, s19 // (op 1178) 


s_lshl_b32 s1, s46, 24 // (op 1178) 
s_ashr_i32 s19, s1, 31 // (op 1178) 

s_xor_b32 s1, s18, s19 // (op 1178) 

s_sub_i32 s18, s1, s19 // (op 1178) 


s_add_u32 s50, s6, 64 // (op 1178) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1178) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1178) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1178) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1178) 

s_mul_i32 s1, s15, s24 // (op 1178) 

s_mul_i32 s19, s8, s1 // (op 1178) 

s_mul_i32 s1, s18, s19 // (op 1178) 

s_sub_u32 s19, s16, s1 // (op 1178) 

s_mov_b32 s1, s19 // (op 1178) 
s_mul_hi_i32 s22, s1, s14 // (op 1178) 
s_add_i32 s23, s22, s1 // (op 1178) 


s_ashr_i32 s1, s23, 31 // (op 1178) 
s_cmp_eq_i32 s14, 0 // (op 1178) 
s_mov_b32 s22, scc // (op 1178) 
s_cmp_eq_i32 s22, 0 // (op 1178) 
s_cselect_b32 s24, 0, -1 // (op 1178) 

s_and_b32 s22, s48, 31 // (op 1178) 
s_lshl_b32 s25, 1, s22 // (op 1178) 
s_add_i32 s49, s25, s24 // (op 1178) 


s_and_b32 s24, s1, s49 // (op 1178) 


s_add_i32 s1, s23, s24 // (op 1178) 


s_ashr_i32 s23, s1, s22 // (op 1178) 


s_lshl_b32 s1, s48, 24 // (op 1178) 
s_ashr_i32 s22, s1, 31 // (op 1178) 

s_xor_b32 s1, s23, s22 // (op 1178) 

s_sub_i32 s23, s1, s22 // (op 1178) 


s_mul_i32 s1, s47, s15 // (op 1178) 
s_add_u32 s24, s20, 16 // (op 1178) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1178) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1178) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1178) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1178) 

s_sub_i32 s22, s24, s1 // (op 1178) 

s_mul_i32 s24, s23, s22 // (op 1178) 


s_sub_u32 s25, s19, s24 // (op 1178) 


s_add_u32 s19, s25, s1 // (op 1178) 


s_cmp_eq_i32 s18, 0 // (op 1178) 
s_mov_b32 s1, scc // (op 1178) 

s_cmp_eq_i32 s1, 0 // (op 1178) 
s_cselect_b32 s18, s19, s0 // (op 1178) 



s_ashr_i64 s[0:1], s[30:31], 8 // (op 1178) 
s_lshr_b32 s19, s0, 1 // (op 1178) 

v_lshl_add_u32 v17, s19, 9, v15 // (op 1178) 


v_mul_lo_u32 v15, v17, s26 // (op 1178) 

v_lshl_add_u32 v17, s18, 4, v15 // (op 1178) 


s_mul_i32 s1, s26, s31 // (op 1180) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1180) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1180) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1180) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1180) most significant: sum
s_add_u32 s1, s1, s19 // (op 1180) most significant: sum


s_mov_b32 s52, s28 // (op 1180) reg expression
s_mov_b32 s53, s29 // (op 1180) reg expression
s_mov_b32 s54, s0 // (op 1180) reg expression
s_mov_b32 s55, 131072 // (op 1180) call()

s_mul_hi_i32 s0, s16, s10 // (op 1246) 
v_lshrrev_b32 v15, 3, v1 // (op 1246) 
s_add_i32 s1, s0, s16 // (op 1246) 

v_lshlrev_b32 v22, 2, v15 // (op 1246) 

s_ashr_i32 s0, s1, 31 // (op 1246) 
v_and_b32 v15, -8, v1 // (op 1246) 
s_cmp_eq_i32 s10, 0 // (op 1246) 
s_mov_b32 s18, scc // (op 1246) 
v_sub_u32 v19, v1, v15 // (op 1246) 

s_cmp_eq_i32 s18, 0 // (op 1246) 
s_cselect_b32 s19, 0, -1 // (op 1246) 

v_lshlrev_b32 v15, 5, v19 // (op 1246) 

s_and_b32 s18, s11, 31 // (op 1246) 
s_lshl_b32 s22, 1, s18 // (op 1246) 
s_add_i32 s23, s22, s19 // (op 1246) 


s_and_b32 s19, s0, s23 // (op 1246) 


s_add_i32 s0, s1, s19 // (op 1246) 


s_ashr_i32 s1, s0, s18 // (op 1246) 


s_lshl_b32 s0, s11, 24 // (op 1246) 
s_ashr_i32 s18, s0, 31 // (op 1246) 

s_xor_b32 s0, s1, s18 // (op 1246) 

s_sub_i32 s1, s0, s18 // (op 1246) 


s_mul_hi_i32 s0, s1, s12 // (op 1246) 
s_add_i32 s18, s0, s1 // (op 1246) 

s_ashr_i32 s0, s18, 31 // (op 1246) 
s_cmp_eq_i32 s12, 0 // (op 1246) 
s_mov_b32 s19, scc // (op 1246) 
s_cmp_eq_i32 s19, 0 // (op 1246) 
s_cselect_b32 s22, 0, -1 // (op 1246) 

s_and_b32 s19, s13, 31 // (op 1246) 
s_lshl_b32 s23, 1, s19 // (op 1246) 
s_add_i32 s24, s23, s22 // (op 1246) 


s_and_b32 s25, s0, s24 // (op 1246) 


s_add_i32 s0, s18, s25 // (op 1246) 


s_ashr_i32 s18, s0, s19 // (op 1246) 


s_lshl_b32 s0, s13, 24 // (op 1246) 
s_ashr_i32 s19, s0, 31 // (op 1246) 

s_xor_b32 s0, s18, s19 // (op 1246) 

s_sub_i32 s18, s0, s19 // (op 1246) 


s_add_u32 s50, s6, 64 // (op 1246) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1246) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1246) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1246) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1246) 

s_mul_i32 s0, s18, s24 // (op 1246) 

s_sub_i32 s18, s1, s0 // (op 1246) 


s_mul_hi_i32 s19, s16, s9 // (op 1246) 
s_add_i32 s0, s19, s16 // (op 1246) 

s_ashr_i32 s1, s0, 31 // (op 1246) 
s_cmp_eq_i32 s9, 0 // (op 1246) 
s_mov_b32 s19, scc // (op 1246) 
s_cmp_eq_i32 s19, 0 // (op 1246) 
s_cselect_b32 s22, 0, -1 // (op 1246) 

s_and_b32 s19, s46, 31 // (op 1246) 
s_lshl_b32 s23, 1, s19 // (op 1246) 
s_add_i32 s49, s23, s22 // (op 1246) 


s_and_b32 s22, s1, s49 // (op 1246) 


s_add_i32 s1, s0, s22 // (op 1246) 


s_ashr_i32 s0, s1, s19 // (op 1246) 


s_lshl_b32 s1, s46, 24 // (op 1246) 
s_ashr_i32 s19, s1, 31 // (op 1246) 

s_xor_b32 s1, s0, s19 // (op 1246) 

s_sub_i32 s0, s1, s19 // (op 1246) 


s_mul_i32 s1, s15, s24 // (op 1246) 

s_mul_i32 s19, s8, s1 // (op 1246) 

s_mul_i32 s1, s0, s19 // (op 1246) 

s_sub_u32 s19, s16, s1 // (op 1246) 

s_mul_hi_i32 s1, s19, s14 // (op 1246) 
s_add_i32 s22, s1, s19 // (op 1246) 


s_ashr_i32 s1, s22, 31 // (op 1246) 
s_cmp_eq_i32 s14, 0 // (op 1246) 
s_mov_b32 s19, scc // (op 1246) 
s_cmp_eq_i32 s19, 0 // (op 1246) 
s_cselect_b32 s23, 0, -1 // (op 1246) 

s_and_b32 s19, s48, 31 // (op 1246) 
s_lshl_b32 s24, 1, s19 // (op 1246) 
s_add_i32 s25, s24, s23 // (op 1246) 


s_and_b32 s23, s1, s25 // (op 1246) 


s_add_i32 s1, s22, s23 // (op 1246) 


s_ashr_i32 s22, s1, s19 // (op 1246) 


s_lshl_b32 s1, s48, 24 // (op 1246) 
s_ashr_i32 s19, s1, 31 // (op 1246) 

s_xor_b32 s1, s22, s19 // (op 1246) 

s_sub_i32 s22, s1, s19 // (op 1246) 


s_cmp_eq_i32 s0, 0 // (op 1246) 
s_mov_b32 s1, scc // (op 1246) 

s_cmp_eq_i32 s1, 0 // (op 1246) 
s_cselect_b32 s0, s22, s18 // (op 1246) 



v_lshl_add_u32 v19, s0, 6, v22 // (op 1246) 


v_mul_lo_u32 v25, v19, s33 // (op 1246) most significant: low of low * high
v_mul_hi_u32 v20, v19, s32 // (op 1246) most significant: high of low * low
v_mul_lo_u32 v24, v19, s32 // (op 1246) least significant: low of low * low
v_add_u32 v25, v25, v20 // (op 1246) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 8 // (op 1246) 
s_lshr_b32 s18, s0, 1 // (op 1246) 

v_lshl_add_u32 v22, s18, 9, v15 // (op 1246) 


v_mov_b32 v27, 0 // (op 1246) convert
v_mov_b32 v26, v22 // (op 1246) convert

v_add_co_u32 v28, s[0:1], v26, v24 // (op 1246) least significant half
v_addc_co_u32 v29, s[0:1], v27, v25, s[0:1] // (op 1246) most significant half

v_lshlrev_b64 v[28:29], 2, v[28:29] // (op 1246) 


v_lshrrev_b64 v[24:25], 3, v[28:29] // (op 1246) 

v_mov_b32 v15, v24 // (op 1246) convert

s_mul_i32 s1, s32, s7 // (op 1248) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 1248) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 1248) least significant
s_mul_hi_u32 s19, s32, s6 // (op 1248) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1248) most significant: sum
s_add_u32 s1, s1, s19 // (op 1248) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1248) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 1248) 

s_mov_b32 s64, s34 // (op 1248) reg expression
s_mov_b32 s65, s35 // (op 1248) reg expression
s_mov_b32 s66, s0 // (op 1248) reg expression
s_mov_b32 s67, 131072 // (op 1248) call()

s_mul_hi_i32 s0, s16, s10 // (op 1310) 
v_and_b32 v22, -16, v1 // (op 1310) 
s_add_i32 s1, s0, s16 // (op 1310) 

v_sub_u32 v19, v1, v22 // (op 1310) 

s_ashr_i32 s0, s1, 31 // (op 1310) 
v_lshrrev_b32 v20, 4, v1 // (op 1310) 
s_cmp_eq_i32 s10, 0 // (op 1310) 
s_mov_b32 s18, scc // (op 1310) 
s_cmp_eq_i32 s18, 0 // (op 1310) 
s_cselect_b32 s19, 0, -1 // (op 1310) 

s_and_b32 s18, s11, 31 // (op 1310) 
s_lshl_b32 s22, 1, s18 // (op 1310) 
s_add_i32 s23, s22, s19 // (op 1310) 


s_and_b32 s19, s0, s23 // (op 1310) 


s_add_i32 s0, s1, s19 // (op 1310) 


s_ashr_i32 s1, s0, s18 // (op 1310) 


s_lshl_b32 s0, s11, 24 // (op 1310) 
s_ashr_i32 s18, s0, 31 // (op 1310) 

s_xor_b32 s0, s1, s18 // (op 1310) 

s_sub_i32 s1, s0, s18 // (op 1310) 


s_mul_hi_i32 s0, s1, s12 // (op 1310) 
s_add_i32 s18, s0, s1 // (op 1310) 

s_ashr_i32 s0, s18, 31 // (op 1310) 
s_cmp_eq_i32 s12, 0 // (op 1310) 
s_mov_b32 s19, scc // (op 1310) 
s_cmp_eq_i32 s19, 0 // (op 1310) 
s_cselect_b32 s22, 0, -1 // (op 1310) 

s_and_b32 s19, s13, 31 // (op 1310) 
s_lshl_b32 s23, 1, s19 // (op 1310) 
s_add_i32 s24, s23, s22 // (op 1310) 


s_and_b32 s25, s0, s24 // (op 1310) 


s_add_i32 s0, s18, s25 // (op 1310) 


s_ashr_i32 s18, s0, s19 // (op 1310) 


s_lshl_b32 s0, s13, 24 // (op 1310) 
s_ashr_i32 s19, s0, 31 // (op 1310) 

s_xor_b32 s0, s18, s19 // (op 1310) 

s_sub_i32 s18, s0, s19 // (op 1310) 


s_mul_i32 s0, s1, s15 // (op 1310) 

s_sub_u32 s1, s16, s0 // (op 1310) 

s_mul_i32 s0, s18, s15 // (op 1310) 
s_add_u32 s0, s0, s1 // (op 1310) 


s_mul_hi_i32 s1, s16, s9 // (op 1310) 
s_add_i32 s18, s1, s16 // (op 1310) 

s_ashr_i32 s1, s18, 31 // (op 1310) 
s_cmp_eq_i32 s9, 0 // (op 1310) 
s_mov_b32 s19, scc // (op 1310) 
s_cmp_eq_i32 s19, 0 // (op 1310) 
s_cselect_b32 s22, 0, -1 // (op 1310) 

s_and_b32 s19, s46, 31 // (op 1310) 
s_lshl_b32 s23, 1, s19 // (op 1310) 
s_add_i32 s24, s23, s22 // (op 1310) 


s_and_b32 s25, s1, s24 // (op 1310) 


s_add_i32 s1, s18, s25 // (op 1310) 


s_ashr_i32 s18, s1, s19 // (op 1310) 


s_lshl_b32 s1, s46, 24 // (op 1310) 
s_ashr_i32 s19, s1, 31 // (op 1310) 

s_xor_b32 s1, s18, s19 // (op 1310) 

s_sub_i32 s18, s1, s19 // (op 1310) 


s_add_u32 s50, s6, 64 // (op 1310) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1310) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1310) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1310) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1310) 

s_mul_i32 s1, s15, s24 // (op 1310) 

s_mul_i32 s19, s8, s1 // (op 1310) 

s_mul_i32 s1, s18, s19 // (op 1310) 

s_sub_u32 s19, s16, s1 // (op 1310) 

s_mov_b32 s1, s19 // (op 1310) 
s_mul_hi_i32 s22, s1, s14 // (op 1310) 
s_add_i32 s23, s22, s1 // (op 1310) 


s_ashr_i32 s1, s23, 31 // (op 1310) 
s_cmp_eq_i32 s14, 0 // (op 1310) 
s_mov_b32 s22, scc // (op 1310) 
s_cmp_eq_i32 s22, 0 // (op 1310) 
s_cselect_b32 s24, 0, -1 // (op 1310) 

s_and_b32 s22, s48, 31 // (op 1310) 
s_lshl_b32 s25, 1, s22 // (op 1310) 
s_add_i32 s49, s25, s24 // (op 1310) 


s_and_b32 s24, s1, s49 // (op 1310) 


s_add_i32 s1, s23, s24 // (op 1310) 


s_ashr_i32 s23, s1, s22 // (op 1310) 


s_lshl_b32 s1, s48, 24 // (op 1310) 
s_ashr_i32 s22, s1, 31 // (op 1310) 

s_xor_b32 s1, s23, s22 // (op 1310) 

s_sub_i32 s23, s1, s22 // (op 1310) 


s_mul_i32 s1, s47, s15 // (op 1310) 
s_add_u32 s24, s20, 16 // (op 1310) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1310) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1310) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1310) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1310) 

s_sub_i32 s22, s24, s1 // (op 1310) 

s_mul_i32 s24, s23, s22 // (op 1310) 


s_sub_u32 s25, s19, s24 // (op 1310) 


s_add_u32 s19, s25, s1 // (op 1310) 


s_cmp_eq_i32 s18, 0 // (op 1310) 
s_mov_b32 s1, scc // (op 1310) 

s_cmp_eq_i32 s1, 0 // (op 1310) 
s_cselect_b32 s18, s19, s0 // (op 1310) 



v_lshl_add_u32 v22, s18, 4, v19 // (op 1310) 


s_ashr_i64 s[0:1], s[30:31], 8 // (op 1310) 
s_lshr_b32 s18, s0, 1 // (op 1310) 

v_lshl_add_u32 v19, s18, 4, v20 // (op 1310) 


v_mul_lo_u32 v20, v19, s36 // (op 1310) 

v_add_u32 v19, v22, v20 // (op 1310) 


s_mul_i32 s1, s36, s41 // (op 1312) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1312) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1312) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1312) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1312) most significant: sum
s_add_u32 s1, s1, s19 // (op 1312) most significant: sum


s_mov_b32 s68, s38 // (op 1312) reg expression
s_mov_b32 s69, s39 // (op 1312) reg expression
s_mov_b32 s70, s0 // (op 1312) reg expression
s_mov_b32 s71, 131072 // (op 1312) call()

s_mul_hi_i32 s0, s16, s10 // (op 1374) 
v_lshrrev_b32 v20, 1, v1 // (op 1374) 
s_add_i32 s1, s0, s16 // (op 1374) 

v_and_b32 v22, -2, v1 // (op 1374) 
s_ashr_i32 s0, s1, 31 // (op 1374) 
v_sub_u32 v24, v1, v22 // (op 1374) 

s_cmp_eq_i32 s10, 0 // (op 1374) 
s_mov_b32 s18, scc // (op 1374) 
v_lshlrev_b32 v22, 2, v24 // (op 1374) 

s_cmp_eq_i32 s18, 0 // (op 1374) 
s_cselect_b32 s19, 0, -1 // (op 1374) 

s_and_b32 s18, s11, 31 // (op 1374) 
s_lshl_b32 s22, 1, s18 // (op 1374) 
s_add_i32 s23, s22, s19 // (op 1374) 


s_and_b32 s19, s0, s23 // (op 1374) 


s_add_i32 s0, s1, s19 // (op 1374) 


s_ashr_i32 s1, s0, s18 // (op 1374) 


s_lshl_b32 s0, s11, 24 // (op 1374) 
s_ashr_i32 s18, s0, 31 // (op 1374) 

s_xor_b32 s0, s1, s18 // (op 1374) 

s_sub_i32 s1, s0, s18 // (op 1374) 


s_mul_hi_i32 s0, s1, s12 // (op 1374) 
s_add_i32 s18, s0, s1 // (op 1374) 

s_ashr_i32 s0, s18, 31 // (op 1374) 
s_cmp_eq_i32 s12, 0 // (op 1374) 
s_mov_b32 s19, scc // (op 1374) 
s_cmp_eq_i32 s19, 0 // (op 1374) 
s_cselect_b32 s22, 0, -1 // (op 1374) 

s_and_b32 s19, s13, 31 // (op 1374) 
s_lshl_b32 s23, 1, s19 // (op 1374) 
s_add_i32 s24, s23, s22 // (op 1374) 


s_and_b32 s25, s0, s24 // (op 1374) 


s_add_i32 s0, s18, s25 // (op 1374) 


s_ashr_i32 s18, s0, s19 // (op 1374) 


s_lshl_b32 s0, s13, 24 // (op 1374) 
s_ashr_i32 s19, s0, 31 // (op 1374) 

s_xor_b32 s0, s18, s19 // (op 1374) 

s_sub_i32 s18, s0, s19 // (op 1374) 


s_add_u32 s50, s6, 64 // (op 1374) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1374) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1374) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1374) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1374) 

s_mul_i32 s0, s18, s24 // (op 1374) 

s_sub_i32 s18, s1, s0 // (op 1374) 


s_mul_hi_i32 s19, s16, s9 // (op 1374) 
s_add_i32 s0, s19, s16 // (op 1374) 

s_ashr_i32 s1, s0, 31 // (op 1374) 
s_cmp_eq_i32 s9, 0 // (op 1374) 
s_mov_b32 s19, scc // (op 1374) 
s_cmp_eq_i32 s19, 0 // (op 1374) 
s_cselect_b32 s22, 0, -1 // (op 1374) 

s_and_b32 s19, s46, 31 // (op 1374) 
s_lshl_b32 s23, 1, s19 // (op 1374) 
s_add_i32 s49, s23, s22 // (op 1374) 


s_and_b32 s22, s1, s49 // (op 1374) 


s_add_i32 s1, s0, s22 // (op 1374) 


s_ashr_i32 s0, s1, s19 // (op 1374) 


s_lshl_b32 s1, s46, 24 // (op 1374) 
s_ashr_i32 s19, s1, 31 // (op 1374) 

s_xor_b32 s1, s0, s19 // (op 1374) 

s_sub_i32 s0, s1, s19 // (op 1374) 


s_mul_i32 s1, s15, s24 // (op 1374) 

s_mul_i32 s19, s8, s1 // (op 1374) 

s_mul_i32 s1, s0, s19 // (op 1374) 

s_sub_u32 s19, s16, s1 // (op 1374) 

s_mul_hi_i32 s1, s19, s14 // (op 1374) 
s_add_i32 s22, s1, s19 // (op 1374) 


s_ashr_i32 s1, s22, 31 // (op 1374) 
s_cmp_eq_i32 s14, 0 // (op 1374) 
s_mov_b32 s19, scc // (op 1374) 
s_cmp_eq_i32 s19, 0 // (op 1374) 
s_cselect_b32 s23, 0, -1 // (op 1374) 

s_and_b32 s19, s48, 31 // (op 1374) 
s_lshl_b32 s24, 1, s19 // (op 1374) 
s_add_i32 s25, s24, s23 // (op 1374) 


s_and_b32 s23, s1, s25 // (op 1374) 


s_add_i32 s1, s22, s23 // (op 1374) 


s_ashr_i32 s22, s1, s19 // (op 1374) 


s_lshl_b32 s1, s48, 24 // (op 1374) 
s_ashr_i32 s19, s1, 31 // (op 1374) 

s_xor_b32 s1, s22, s19 // (op 1374) 

s_sub_i32 s22, s1, s19 // (op 1374) 


s_cmp_eq_i32 s0, 0 // (op 1374) 
s_mov_b32 s1, scc // (op 1374) 

s_cmp_eq_i32 s1, 0 // (op 1374) 
s_cselect_b32 s0, s22, s18 // (op 1374) 



v_lshl_add_u32 v24, s0, 6, v20 // (op 1374) 


v_mul_lo_u32 v20, v24, s42 // (op 1374) 

s_ashr_i64 s[0:1], s[30:31], 8 // (op 1374) 
s_lshr_b32 s18, s0, 1 // (op 1374) 

v_lshl_add_u32 v24, s18, 4, v22 // (op 1374) 


v_add_u32 v22, v24, v20 // (op 1374) 


s_mul_i32 s1, s42, s7 // (op 1376) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1376) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1376) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1376) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1376) most significant: sum
s_add_u32 s1, s1, s19 // (op 1376) most significant: sum


s_mov_b32 s72, s44 // (op 1376) reg expression
s_mov_b32 s73, s45 // (op 1376) reg expression
s_mov_b32 s74, s0 // (op 1376) reg expression
s_mov_b32 s75, 131072 // (op 1376) call()

v_lshlrev_b32 v20, 5, v1 // (op 1467) 
v_and_b32 v24, -8, v1 // (op 1520) 
v_sub_u32 v25, v1, v24 // (op 1520) 

v_lshlrev_b32 v24, 5, v25 // (op 1520) 

v_lshrrev_b32 v25, 3, v1 // (op 1520) 
v_lshl_add_u32 v26, v25, 10, v24 // (op 1520) 


v_lshlrev_b32 v24, 2, v26 // (op 1520) 

v_lshrrev_b32 v25, 3, v24 // (op 1520) 

v_and_b32 v24, -16, v1 // (op 1578) 
v_sub_u32 v26, v1, v24 // (op 1578) 

v_and_b32 v24, 4294967280, v1 // (op 1578) 
v_add_u32 v27, v24, v26 // (op 1578) 


v_and_b32 v24, -2, v1 // (op 1631) 
v_sub_u32 v26, v1, v24 // (op 1631) 

v_lshlrev_b32 v24, 2, v26 // (op 1631) 

v_lshrrev_b32 v26, 1, v1 // (op 1631) 
v_lshl_add_u32 v28, v26, 3, v24 // (op 1631) 


s_ashr_i64 s[0:1], s[30:31], 8 // (op 405) 
s_and_b32 s18, s0, 4294967294 // (op 405) 

s_ashr_i64 s[0:1], s[30:31], 8 // (op 404) 
s_mov_b32 s19, s0 // (op 404) convert

s_cmp_lt_u32 s18, s19 // (op 404) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_404 // (op 404) Condition: Top (jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_404 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_404:

s_add_u32 s18, s18, 1 // (op 406) 
v_mov_b64 v[30:31], s[26:27] // (op 198) convert

v_mov_b32 v14, v17 // (op 198) 
v_mov_b32 v16, v14 // (op 198) reg expression
buffer_load_dwordx4 v[32:35], v16, s[52:55], 0 offen // (op 198) Load value
v_add_u32 v16, v16, v30 // (op 198) 
buffer_load_dwordx4 v[36:39], v16, s[52:55], 0 offen // (op 198) Load value



s_lshl_b32 s19, s26, 8 // (op 1168) 
v_add_u32 v17, v17, s19 // (op 1168) 


s_barrier  // (op 1845) 
s_lshl_b64 s[0:1], s[32:33], 2 // (op 221) 



v_lshrrev_b64 v[30:31], 3, s[0:1] // (op 221) 


v_mov_b32 v14, v15 // (op 221) 
v_mov_b32 v16, v14 // (op 221) 
buffer_load_dwordx4 v[40:43], v16, s[64:67], 0 offen // (op 221) Load value
v_add_u32 v16, v16, v30 // (op 221) 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[32:35] offset:8704 // (op 199) Store local data 
buffer_load_dwordx4 v[44:47], v16, s[64:67], 0 offen // (op 221) Load value
v_add_u32 v16, v16, v30 // (op 221) 
buffer_load_dwordx4 v[48:51], v16, s[64:67], 0 offen // (op 221) Load value
v_add_u32 v16, v16, v30 // (op 221) 
buffer_load_dwordx4 v[52:55], v16, s[64:67], 0 offen // (op 221) Load value



v_add_u32 v15, 128, v15 // (op 1238) 
s_waitcnt vmcnt(4)
ds_write_b128 v20, v[36:39] offset:8720 // (op 199) Store local data 



v_mov_b32 v14, v19 // (op 222) 
s_barrier  // (op 1864) 
v_mov_b32 v16, v14 // (op 222) reg expression




s_waitcnt vmcnt(3)
ds_write_b128 v25, v[40:43] // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v25, v[44:47] offset:128 // (op 218) Store local data 
buffer_load_ubyte v18, v16, s[68:71], 0 offen offset: 0 // (op 222) Load value


s_lshl_b32 s19, s36, 3 // (op 1302) 
v_add_u32 v19, v19, s19 // (op 1302) 

s_waitcnt vmcnt(2)
ds_write_b128 v25, v[48:51] offset:256 // (op 218) Store local data 


v_mov_b32 v14, v22 // (op 223) 

buffer_load_dword v16, v14, s[72:75], 0 offen // (op 223) Load value

v_add_u32 v22, 8, v22 // (op 1366) 
s_waitcnt vmcnt(2)
ds_write_b128 v25, v[52:55] offset:384 // (op 218) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1854) 



s_barrier  // (op 1885) 
ds_read_b64_tr_b8 v[30:31], v0 offset:8704 // (op 201) Transpose load local data 



ds_read_b64_tr_b8 v[32:33], v0 offset:8832 // (op 201) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:9728 // (op 201) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:9856 // (op 201) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v27, v18 offset:12800 // (op 219) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 1875) 



s_barrier  // (op 1905) 
ds_read_b128 v[38:41], v9 // (op 203) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 1895) 
s_waitcnt vmcnt(0)
ds_write_b32 v28, v16 offset:8192 // (op 220) Store local data 



ds_read_u8 v14, v4 offset:12800 // (op 205) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1915) 



ds_read_u8 v16, v5 offset:8192 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:41], a[0:3], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 208) 





ds_read_b64_tr_b8 v[38:39], v0 offset:10752 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:10880 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:11776 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:11904 // (op 210) Transpose load local data 



ds_read_b128 v[46:49], v9 offset:64 // (op 212) Load local data 


ds_read_u8 v16, v4 offset:12864 // (op 214) Load local data 



ds_read_u8 v18, v5 offset:8196 // (op 216) Load local data 



ds_read_b128 v[50:53], v9 offset:4096 // (op 412) Load local data 



ds_read_u8 v21, v5 offset:8448 // (op 416) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:37], v[50:53], a[4:7], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 417) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[46:49], a[0:3], v16, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 217) 
ds_read_b128 v[34:37], v9 offset:4160 // (op 421) Load local data 





ds_read_u8 v14, v5 offset:8452 // (op 425) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:45], v[34:37], a[4:7], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 426) 




s_ashr_i64 s[0:1], s[30:31], 8 // (op 404) 
s_mov_b32 s19, s0 // (op 404) convert

s_cmp_lt_u32 s18, s19 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_404 // (op 404) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_404 if true)
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_404:























s_barrier  // (op 2012) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_155
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
s_branch RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_155
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_155:


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
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_155:










s_endpgm  // End of RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.LRR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end:
.size RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_, .LRR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end-RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 128
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25856 // lds bytes
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
    DataTypeB: F4
    DestDataType: s
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
  - .name: RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
    .symbol: RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 25856
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 118
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

