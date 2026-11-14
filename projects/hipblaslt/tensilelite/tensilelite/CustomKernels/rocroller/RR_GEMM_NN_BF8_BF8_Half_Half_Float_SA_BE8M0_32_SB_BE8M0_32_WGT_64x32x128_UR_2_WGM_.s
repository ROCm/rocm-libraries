// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_,@function
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s8 // (op 1209) 
s_mul_hi_i32 s1, s16, s8 // (op 1396) 
v_lshrrev_b32 v0, 6, v1 // (op 1322) 
v_and_b32 v3, 1, v0 // (op 1322) 

v_and_b32 v0, 15, v1 // (op 1322) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 1322) 


v_and_b32 v0, 48, v1 // (op 1322) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 1322) 


v_lshrrev_b32 v0, 6, v1 // (op 1481) 
v_and_b32 v4, 1, v0 // (op 1481) 

v_and_b32 v0, 15, v1 // (op 1481) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 1481) 


v_and_b32 v0, 48, v1 // (op 1481) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 1481) 


v_lshrrev_b32 v0, 7, v1 // (op 1209) 
s_add_i32 s18, s0, s16 // (op 1209) 

v_and_b32 v5, 63, v1 // (op 1209) 
s_ashr_i32 s0, s18, 31 // (op 1209) 
v_lshrrev_b32 v6, 4, v5 // (op 1209) 
s_cmp_eq_i32 s8, 0 // (op 1209) 
s_mov_b32 s19, scc // (op 1209) 
v_lshlrev_b32 v7, 2, v6 // (op 1209) 

s_cmp_eq_i32 s19, 0 // (op 1209) 
s_cselect_b32 s51, 0, -1 // (op 1209) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1209) 


s_and_b32 s19, s9, 31 // (op 1209) 
v_lshrrev_b32 v0, 2, v5 // (op 1209) 

v_and_b32 v5, 3, v0 // (op 1209) 

v_and_b32 v0, 48, v1 // (op 1270) 
v_and_b32 v7, 63, v1 // (op 1270) 
v_sub_u32 v8, v7, v0 // (op 1270) 

v_lshrrev_b32 v7, 1, v8 // (op 1270) 
v_add_lshl_u32 v9, v0, v7, 6 // (op 1270) 


v_and_b32 v0, 1, v8 // (op 1270) 

v_lshlrev_b32 v7, 3, v0 // (op 1270) 

v_lshrrev_b32 v0, 7, v1 // (op 1270) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 1270) 


v_add_u32 v0, v9, v8 // (op 1270) 


v_lshrrev_b32 v7, 7, v1 // (op 1380) 
v_and_b32 v8, 15, v1 // (op 1380) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 1380) 


v_and_b32 v7, 63, v1 // (op 1380) 
v_lshrrev_b32 v8, 4, v7 // (op 1380) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 1380) 


v_lshrrev_b32 v8, 7, v1 // (op 1396) 
v_and_b32 v9, 63, v1 // (op 1396) 
v_and_b32 v10, 48, v1 // (op 1442) 
v_and_b32 v11, 63, v1 // (op 1442) 
v_sub_u32 v12, v11, v10 // (op 1442) 

v_lshrrev_b32 v11, 1, v12 // (op 1442) 
v_add_lshl_u32 v13, v10, v11, 6 // (op 1442) 


v_and_b32 v10, 1, v12 // (op 1442) 

v_lshlrev_b32 v11, 3, v10 // (op 1442) 

v_lshrrev_b32 v10, 7, v1 // (op 1442) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 1442) 


v_add_u32 v10, v13, v12 // (op 1442) 


v_lshrrev_b32 v11, 7, v1 // (op 1510) 
v_and_b32 v12, 15, v1 // (op 1510) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 1510) 


v_and_b32 v11, 63, v1 // (op 1510) 
v_lshrrev_b32 v12, 4, v11 // (op 1510) 

v_lshl_add_u32 v11, v12, 6, v13 // (op 1510) 


s_lshl_b32 s52, 1, s19 // (op 1209) 
v_and_b32 v12, 3, v1 // (op 1209) 
s_add_i32 s53, s52, s51 // (op 1209) 


s_and_b32 s51, s0, s53 // (op 1209) 


s_add_i32 s0, s1, s16 // (op 1396) 

s_ashr_i32 s1, s0, 31 // (op 1396) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1209) 


s_add_i32 s52, s18, s51 // (op 1209) 


v_lshrrev_b32 v5, 6, v1 // (op 1209) 
s_ashr_i32 s18, s52, s19 // (op 1209) 


v_and_b32 v12, 1, v5 // (op 1209) 

s_lshl_b32 s19, s9, 24 // (op 1209) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1209) 


s_ashr_i32 s51, s19, 31 // (op 1209) 

s_xor_b32 s19, s18, s51 // (op 1209) 

s_sub_i32 s18, s19, s51 // (op 1209) 


s_mul_hi_i32 s19, s18, s10 // (op 1209) 
s_add_i32 s51, s19, s18 // (op 1209) 

s_ashr_i32 s19, s51, 31 // (op 1209) 
s_cmp_eq_i32 s10, 0 // (op 1209) 
s_mov_b32 s52, scc // (op 1209) 
s_cmp_eq_i32 s52, 0 // (op 1209) 
s_cselect_b32 s53, 0, -1 // (op 1209) 

s_and_b32 s52, s11, 31 // (op 1209) 
v_lshrrev_b32 v12, 4, v9 // (op 1396) 
s_cmp_eq_i32 s8, 0 // (op 1396) 
s_mov_b32 s54, scc // (op 1396) 
v_lshlrev_b32 v13, 2, v12 // (op 1396) 

s_cmp_eq_i32 s54, 0 // (op 1396) 
s_cselect_b32 s55, 0, -1 // (op 1396) 

s_lshl_b32 s54, 1, s52 // (op 1209) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 1396) 


v_lshrrev_b32 v8, 2, v9 // (op 1396) 

s_add_i32 s56, s54, s53 // (op 1209) 


s_and_b32 s53, s19, s56 // (op 1209) 


s_add_i32 s19, s51, s53 // (op 1209) 


s_ashr_i32 s51, s19, s52 // (op 1209) 


s_lshl_b32 s19, s11, 24 // (op 1209) 
s_ashr_i32 s52, s19, 31 // (op 1209) 

s_xor_b32 s19, s51, s52 // (op 1209) 

s_sub_i32 s51, s19, s52 // (op 1209) 


s_mul_i32 s19, s18, s13 // (op 1209) 
s_sub_u32 s52, s16, s19 // (op 1209) 

s_mul_i32 s19, s51, s13 // (op 1209) 
s_add_u32 s19, s19, s52 // (op 1209) 

s_mul_hi_i32 s52, s16, s7 // (op 1209) 
s_add_i32 s53, s52, s16 // (op 1209) 

s_ashr_i32 s52, s53, 31 // (op 1209) 
s_cmp_eq_i32 s7, 0 // (op 1209) 
s_mov_b32 s54, scc // (op 1209) 
s_cmp_eq_i32 s54, 0 // (op 1209) 
s_cselect_b32 s56, 0, -1 // (op 1209) 

s_and_b32 s54, s14, 31 // (op 1209) 
s_lshl_b32 s57, 1, s54 // (op 1209) 
s_add_i32 s58, s57, s56 // (op 1209) 


s_and_b32 s56, s52, s58 // (op 1209) 


s_add_i32 s52, s53, s56 // (op 1209) 


s_ashr_i32 s53, s52, s54 // (op 1209) 


s_lshl_b32 s52, s14, 24 // (op 1209) 
s_ashr_i32 s54, s52, 31 // (op 1209) 

s_xor_b32 s52, s53, s54 // (op 1209) 

s_sub_i32 s53, s52, s54 // (op 1209) 


s_and_b32 s52, s9, 31 // (op 1396) 
s_lshl_b32 s54, 1, s52 // (op 1396) 
v_and_b32 v9, 3, v8 // (op 1396) 

s_add_i32 s56, s54, s55 // (op 1396) 


v_and_b32 v8, 3, v1 // (op 1396) 
s_and_b32 s54, s1, s56 // (op 1396) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1396) 


s_add_i32 s1, s0, s54 // (op 1396) 


v_lshrrev_b32 v8, 6, v1 // (op 1396) 
s_ashr_i32 s0, s1, s52 // (op 1396) 


v_and_b32 v9, 1, v8 // (op 1396) 

s_lshl_b32 s1, s9, 24 // (op 1396) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1396) 


s_ashr_i32 s52, s1, 31 // (op 1396) 

s_xor_b32 s1, s0, s52 // (op 1396) 

s_sub_i32 s0, s1, s52 // (op 1396) 


s_mul_hi_i32 s1, s0, s10 // (op 1396) 
s_add_i32 s52, s1, s0 // (op 1396) 

s_ashr_i32 s1, s52, 31 // (op 1396) 
s_cmp_eq_i32 s10, 0 // (op 1396) 
s_mov_b32 s54, scc // (op 1396) 
s_cmp_eq_i32 s54, 0 // (op 1396) 
s_cselect_b32 s55, 0, -1 // (op 1396) 

s_and_b32 s54, s11, 31 // (op 1396) 
s_lshl_b32 s56, 1, s54 // (op 1396) 
s_add_i32 s57, s56, s55 // (op 1396) 


s_and_b32 s55, s1, s57 // (op 1396) 


s_add_i32 s1, s52, s55 // (op 1396) 


s_ashr_i32 s52, s1, s54 // (op 1396) 


s_lshl_b32 s1, s11, 24 // (op 1396) 
s_ashr_i32 s54, s1, 31 // (op 1396) 

s_xor_b32 s1, s52, s54 // (op 1396) 

s_sub_i32 s52, s1, s54 // (op 1396) 


s_mul_i32 s1, s0, s13 // (op 1396) 
s_sub_u32 s54, s16, s1 // (op 1396) 

s_mul_i32 s1, s52, s13 // (op 1396) 
s_add_u32 s1, s1, s54 // (op 1396) 

s_mul_hi_i32 s54, s16, s7 // (op 1396) 
s_add_i32 s55, s54, s16 // (op 1396) 

s_ashr_i32 s54, s55, 31 // (op 1396) 
s_cmp_eq_i32 s7, 0 // (op 1396) 
s_mov_b32 s56, scc // (op 1396) 
s_cmp_eq_i32 s56, 0 // (op 1396) 
s_cselect_b32 s57, 0, -1 // (op 1396) 

s_and_b32 s56, s14, 31 // (op 1396) 
s_lshl_b32 s58, 1, s56 // (op 1396) 
s_add_i32 s59, s58, s57 // (op 1396) 


s_and_b32 s57, s54, s59 // (op 1396) 


s_add_i32 s54, s55, s57 // (op 1396) 


s_ashr_i32 s55, s54, s56 // (op 1396) 


s_lshl_b32 s54, s14, 24 // (op 1396) 
s_ashr_i32 s56, s54, 31 // (op 1396) 

s_xor_b32 s54, s55, s56 // (op 1396) 

s_sub_i32 s55, s54, s56 // (op 1396) 


s_add_u32 s56, s4, 32 // (op 1209) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 1209) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1209) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1209) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 1209) 

s_mul_i32 s54, s13, s56 // (op 1209) 
s_mul_i32 s58, s6, s54 // (op 1209) 

s_mul_i32 s54, s53, s58 // (op 1209) 

s_sub_u32 s58, s16, s54 // (op 1209) 

s_mov_b32 s54, s58 // (op 1209) 
s_mul_hi_i32 s59, s54, s12 // (op 1209) 
s_add_i32 s60, s59, s54 // (op 1209) 


s_ashr_i32 s54, s60, 31 // (op 1209) 
s_cmp_eq_i32 s12, 0 // (op 1209) 
s_mov_b32 s59, scc // (op 1209) 
s_cmp_eq_i32 s59, 0 // (op 1209) 
s_cselect_b32 s61, 0, -1 // (op 1209) 

s_add_u32 s62, s4, 32 // (op 1396) Start of Int64 add, locking SCC
s_addc_u32 s63, s5, 0 // (op 1396) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1396) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1396) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 5 // (op 1396) 

s_mul_i32 s59, s13, s62 // (op 1396) 
s_mul_i32 s64, s6, s59 // (op 1396) 

s_mul_i32 s59, s55, s64 // (op 1396) 

s_sub_u32 s64, s16, s59 // (op 1396) 

s_mov_b32 s59, s64 // (op 1396) 
s_mul_hi_i32 s65, s59, s12 // (op 1396) 
s_add_i32 s66, s65, s59 // (op 1396) 


s_ashr_i32 s59, s66, 31 // (op 1396) 
s_cmp_eq_i32 s12, 0 // (op 1396) 
s_mov_b32 s65, scc // (op 1396) 
s_cmp_eq_i32 s65, 0 // (op 1396) 
s_cselect_b32 s67, 0, -1 // (op 1396) 


s_waitcnt lgkmcnt(0)
s_and_b32 s65, s48, 31 // (op 1209) 
s_lshl_b32 s68, 1, s65 // (op 1209) 
s_add_i32 s69, s68, s61 // (op 1209) 


s_and_b32 s61, s54, s69 // (op 1209) 


s_add_i32 s54, s60, s61 // (op 1209) 


s_ashr_i32 s60, s54, s65 // (op 1209) 


s_lshl_b32 s54, s48, 24 // (op 1209) 
s_ashr_i32 s61, s54, 31 // (op 1209) 

s_xor_b32 s54, s60, s61 // (op 1209) 

s_sub_i32 s60, s54, s61 // (op 1209) 


s_mul_i32 s54, s15, s13 // (op 1209) 
s_and_b32 s61, s48, 31 // (op 1396) 
s_lshl_b32 s65, 1, s61 // (op 1396) 
s_add_i32 s68, s65, s67 // (op 1396) 


s_and_b32 s65, s59, s68 // (op 1396) 


s_add_i32 s59, s66, s65 // (op 1396) 


s_ashr_i32 s65, s59, s61 // (op 1396) 


s_lshl_b32 s59, s48, 24 // (op 1396) 
s_ashr_i32 s61, s59, 31 // (op 1396) 

s_xor_b32 s59, s65, s61 // (op 1396) 

s_sub_i32 s65, s59, s61 // (op 1396) 


s_mul_i32 s59, s15, s13 // (op 1396) 
s_add_u32 s66, s20, 64 // (op 1209) Start of Int64 add, locking SCC
s_addc_u32 s67, s21, 0 // (op 1209) End of Int64 add, unlocking SCC
s_sub_u32 s68, s66, 1 // (op 1209) least significant half; sets scc
s_subb_u32 s69, s67, 0 // (op 1209) most significant half; uses scc

s_ashr_i64 s[66:67], s[68:69], 6 // (op 1209) 

s_sub_i32 s61, s66, s54 // (op 1209) 

s_mul_i32 s66, s60, s61 // (op 1209) 

s_sub_u32 s61, s58, s66 // (op 1209) 


s_add_u32 s58, s61, s54 // (op 1209) 


s_cmp_eq_i32 s53, 0 // (op 1209) 
s_mov_b32 s54, scc // (op 1209) 

s_cmp_eq_i32 s54, 0 // (op 1209) 
s_cselect_b32 s53, s58, s19 // (op 1209) 


s_mul_i32 s19, s51, s56 // (op 1209) 


v_lshl_add_u32 v9, s53, 6, v6 // (op 1209) 


s_sub_i32 s51, s18, s19 // (op 1209) 


s_cmp_eq_i32 s54, 0 // (op 1209) 
s_cselect_b32 s53, s60, s51 // (op 1209) 



v_lshl_add_u32 v6, s53, 5, v5 // (op 1209) 


v_mul_lo_u32 v5, v6, s2 // (op 1209) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 1209) 


s_add_u32 s18, s20, 64 // (op 1396) Start of Int64 add, locking SCC
s_addc_u32 s19, s21, 0 // (op 1396) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 1396) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 1396) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 6 // (op 1396) 

s_sub_i32 s51, s18, s59 // (op 1396) 

s_mul_i32 s18, s65, s51 // (op 1396) 

s_sub_u32 s19, s64, s18 // (op 1396) 


s_add_u32 s18, s19, s59 // (op 1396) 


s_cmp_eq_i32 s55, 0 // (op 1396) 
s_mov_b32 s19, scc // (op 1396) 

s_cmp_eq_i32 s19, 0 // (op 1396) 
s_cselect_b32 s51, s18, s1 // (op 1396) 


s_mul_i32 s1, s52, s62 // (op 1396) 


v_lshl_add_u32 v5, s51, 6, v12 // (op 1396) 


s_sub_i32 s18, s0, s1 // (op 1396) 


s_cmp_eq_i32 s19, 0 // (op 1396) 
s_cselect_b32 s0, s65, s18 // (op 1396) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 1396) 


v_mul_lo_u32 v8, v9, s22 // (op 1396) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 1396) 


s_mul_i32 s1, s2, s5 // (op 1211) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 1211) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 1211) least significant
s_mul_hi_u32 s19, s2, s4 // (op 1211) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1211) most significant: sum
s_add_u32 s1, s1, s19 // (op 1211) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1211) 

s_mul_i32 s1, s22, s5 // (op 1398) most significant: low of low * high
s_mul_i32 s51, s23, s4 // (op 1398) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 1398) least significant
s_mul_hi_u32 s52, s22, s4 // (op 1398) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1398) most significant: sum
s_add_u32 s1, s1, s52 // (op 1398) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 1398) 

s_mov_b32 s56, s24 // (op 1211) reg expression
s_mov_b32 s57, s25 // (op 1211) reg expression
s_mov_b32 s58, s18 // (op 1211) reg expression
s_mov_b32 s59, 131072 // (op 1211) call()

s_mov_b32 s60, s26 // (op 1398) reg expression

s_mov_b32 s61, s27 // (op 1398) reg expression
s_mov_b32 s62, s52 // (op 1398) reg expression
s_mov_b32 s63, 131072 // (op 1398) call()


v_mov_b32 v5, s49 // (op 2) Move value
v_mov_b32 v8, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 313) call()
v_accvgpr_write a5, 0.00000 // (op 313) call()
v_accvgpr_write a6, 0.00000 // (op 313) call()
v_accvgpr_write a7, 0.00000 // (op 313) call()
s_mul_hi_i32 s0, s16, s8 // (op 976) 
v_and_b32 v12, -4, v1 // (op 976) 
s_add_i32 s1, s0, s16 // (op 976) 

v_sub_u32 v13, v1, v12 // (op 976) 

s_ashr_i32 s0, s1, 31 // (op 976) 
v_lshlrev_b32 v12, 4, v13 // (op 976) 

s_cmp_eq_i32 s8, 0 // (op 976) 
s_mov_b32 s18, scc // (op 976) 
v_lshrrev_b32 v13, 2, v1 // (op 976) 
s_cmp_eq_i32 s18, 0 // (op 976) 
s_cselect_b32 s19, 0, -1 // (op 976) 

s_and_b32 s18, s9, 31 // (op 976) 
s_lshl_b32 s24, 1, s18 // (op 976) 
s_add_i32 s25, s24, s19 // (op 976) 


s_and_b32 s19, s0, s25 // (op 976) 


s_add_i32 s0, s1, s19 // (op 976) 


s_ashr_i32 s1, s0, s18 // (op 976) 


s_lshl_b32 s0, s9, 24 // (op 976) 
s_ashr_i32 s18, s0, 31 // (op 976) 

s_xor_b32 s0, s1, s18 // (op 976) 

s_sub_i32 s1, s0, s18 // (op 976) 


s_mul_hi_i32 s0, s1, s10 // (op 976) 
s_add_i32 s18, s0, s1 // (op 976) 

s_ashr_i32 s0, s18, 31 // (op 976) 
s_cmp_eq_i32 s10, 0 // (op 976) 
s_mov_b32 s19, scc // (op 976) 
s_cmp_eq_i32 s19, 0 // (op 976) 
s_cselect_b32 s24, 0, -1 // (op 976) 

s_and_b32 s19, s11, 31 // (op 976) 
s_lshl_b32 s25, 1, s19 // (op 976) 
s_add_i32 s26, s25, s24 // (op 976) 


s_and_b32 s27, s0, s26 // (op 976) 


s_add_i32 s0, s18, s27 // (op 976) 


s_ashr_i32 s18, s0, s19 // (op 976) 


s_lshl_b32 s0, s11, 24 // (op 976) 
s_ashr_i32 s19, s0, 31 // (op 976) 

s_xor_b32 s0, s18, s19 // (op 976) 

s_sub_i32 s18, s0, s19 // (op 976) 


s_mul_i32 s0, s1, s13 // (op 976) 

s_sub_u32 s1, s16, s0 // (op 976) 

s_mul_i32 s0, s18, s13 // (op 976) 
s_add_u32 s0, s0, s1 // (op 976) 


s_mul_hi_i32 s1, s16, s7 // (op 976) 
s_add_i32 s18, s1, s16 // (op 976) 

s_ashr_i32 s1, s18, 31 // (op 976) 
s_cmp_eq_i32 s7, 0 // (op 976) 
s_mov_b32 s19, scc // (op 976) 
s_cmp_eq_i32 s19, 0 // (op 976) 
s_cselect_b32 s24, 0, -1 // (op 976) 

s_and_b32 s19, s14, 31 // (op 976) 
s_lshl_b32 s25, 1, s19 // (op 976) 
s_add_i32 s26, s25, s24 // (op 976) 


s_and_b32 s27, s1, s26 // (op 976) 


s_add_i32 s1, s18, s27 // (op 976) 


s_ashr_i32 s18, s1, s19 // (op 976) 


s_lshl_b32 s1, s14, 24 // (op 976) 
s_ashr_i32 s19, s1, 31 // (op 976) 

s_xor_b32 s1, s18, s19 // (op 976) 

s_sub_i32 s18, s1, s19 // (op 976) 


s_add_u32 s24, s4, 32 // (op 976) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 976) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 976) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 976) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 5 // (op 976) 

s_mul_i32 s1, s13, s24 // (op 976) 

s_mul_i32 s19, s6, s1 // (op 976) 

s_mul_i32 s1, s18, s19 // (op 976) 

s_sub_u32 s19, s16, s1 // (op 976) 

s_mov_b32 s1, s19 // (op 976) 
s_mul_hi_i32 s24, s1, s12 // (op 976) 
s_add_i32 s25, s24, s1 // (op 976) 


s_ashr_i32 s1, s25, 31 // (op 976) 
s_cmp_eq_i32 s12, 0 // (op 976) 
s_mov_b32 s24, scc // (op 976) 
s_cmp_eq_i32 s24, 0 // (op 976) 
s_cselect_b32 s26, 0, -1 // (op 976) 

s_and_b32 s24, s48, 31 // (op 976) 
s_lshl_b32 s27, 1, s24 // (op 976) 
s_add_i32 s49, s27, s26 // (op 976) 


s_and_b32 s26, s1, s49 // (op 976) 


s_add_i32 s1, s25, s26 // (op 976) 


s_ashr_i32 s25, s1, s24 // (op 976) 


s_lshl_b32 s1, s48, 24 // (op 976) 
s_ashr_i32 s24, s1, 31 // (op 976) 

s_xor_b32 s1, s25, s24 // (op 976) 

s_sub_i32 s25, s1, s24 // (op 976) 


s_mul_i32 s1, s15, s13 // (op 976) 
s_add_u32 s26, s20, 64 // (op 976) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 976) End of Int64 add, unlocking SCC
s_sub_u32 s54, s26, 1 // (op 976) least significant half; sets scc
s_subb_u32 s55, s27, 0 // (op 976) most significant half; uses scc

s_ashr_i64 s[26:27], s[54:55], 6 // (op 976) 

s_sub_i32 s24, s26, s1 // (op 976) 

s_mul_i32 s26, s25, s24 // (op 976) 


s_sub_u32 s27, s19, s26 // (op 976) 


s_add_u32 s19, s27, s1 // (op 976) 


s_cmp_eq_i32 s18, 0 // (op 976) 
s_mov_b32 s1, scc // (op 976) 

s_cmp_eq_i32 s1, 0 // (op 976) 
s_cselect_b32 s18, s19, s0 // (op 976) 



v_lshl_add_u32 v14, s18, 6, v12 // (op 976) 


v_mov_b32 v17, 0 // (op 976) convert
v_mov_b32 v16, v14 // (op 976) convert

v_mul_lo_u32 v15, v13, s29 // (op 976) most significant: low of low * high
v_mul_hi_u32 v12, v13, s28 // (op 976) most significant: high of low * low
v_mul_lo_u32 v14, v13, s28 // (op 976) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 976) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v14 // (op 976) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 976) most significant half



s_mul_i32 s1, s28, s33 // (op 978) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 978) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 978) least significant
s_mul_hi_u32 s19, s28, s32 // (op 978) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 978) most significant: sum
s_add_u32 s1, s1, s19 // (op 978) most significant: sum


s_mov_b32 s24, s30 // (op 978) reg expression
s_mov_b32 s25, s31 // (op 978) reg expression
s_mov_b32 s26, s0 // (op 978) reg expression
s_mov_b32 s27, 131072 // (op 978) call()

s_mul_hi_i32 s0, s16, s8 // (op 995) 
v_and_b32 v14, -4, v1 // (op 995) 
s_add_i32 s1, s0, s16 // (op 995) 

v_sub_u32 v15, v1, v14 // (op 995) 

s_ashr_i32 s0, s1, 31 // (op 995) 
v_lshlrev_b32 v14, 4, v15 // (op 995) 

s_cmp_eq_i32 s8, 0 // (op 995) 
s_mov_b32 s18, scc // (op 995) 
v_lshrrev_b32 v15, 2, v1 // (op 995) 
s_cmp_eq_i32 s18, 0 // (op 995) 
s_cselect_b32 s19, 0, -1 // (op 995) 

s_and_b32 s18, s9, 31 // (op 995) 
s_lshl_b32 s49, 1, s18 // (op 995) 
s_add_i32 s50, s49, s19 // (op 995) 


s_and_b32 s19, s0, s50 // (op 995) 


s_add_i32 s0, s1, s19 // (op 995) 


s_ashr_i32 s1, s0, s18 // (op 995) 


s_lshl_b32 s0, s9, 24 // (op 995) 
s_ashr_i32 s18, s0, 31 // (op 995) 

s_xor_b32 s0, s1, s18 // (op 995) 

s_sub_i32 s1, s0, s18 // (op 995) 


s_mul_hi_i32 s0, s1, s10 // (op 995) 
s_add_i32 s18, s0, s1 // (op 995) 

s_ashr_i32 s0, s18, 31 // (op 995) 
s_cmp_eq_i32 s10, 0 // (op 995) 
s_mov_b32 s19, scc // (op 995) 
s_cmp_eq_i32 s19, 0 // (op 995) 
s_cselect_b32 s49, 0, -1 // (op 995) 

s_and_b32 s19, s11, 31 // (op 995) 
s_lshl_b32 s50, 1, s19 // (op 995) 
s_add_i32 s51, s50, s49 // (op 995) 


s_and_b32 s49, s0, s51 // (op 995) 


s_add_i32 s0, s18, s49 // (op 995) 


s_ashr_i32 s18, s0, s19 // (op 995) 


s_lshl_b32 s0, s11, 24 // (op 995) 
s_ashr_i32 s19, s0, 31 // (op 995) 

s_xor_b32 s0, s18, s19 // (op 995) 

s_sub_i32 s18, s0, s19 // (op 995) 


s_mul_i32 s0, s1, s13 // (op 995) 

s_sub_u32 s1, s16, s0 // (op 995) 

s_mul_i32 s0, s18, s13 // (op 995) 
s_add_u32 s0, s0, s1 // (op 995) 


s_mul_hi_i32 s1, s16, s7 // (op 995) 
s_add_i32 s18, s1, s16 // (op 995) 

s_ashr_i32 s1, s18, 31 // (op 995) 
s_cmp_eq_i32 s7, 0 // (op 995) 
s_mov_b32 s19, scc // (op 995) 
s_cmp_eq_i32 s19, 0 // (op 995) 
s_cselect_b32 s49, 0, -1 // (op 995) 

s_and_b32 s19, s14, 31 // (op 995) 
s_lshl_b32 s50, 1, s19 // (op 995) 
s_add_i32 s51, s50, s49 // (op 995) 


s_and_b32 s49, s1, s51 // (op 995) 


s_add_i32 s1, s18, s49 // (op 995) 


s_ashr_i32 s18, s1, s19 // (op 995) 


s_lshl_b32 s1, s14, 24 // (op 995) 
s_ashr_i32 s19, s1, 31 // (op 995) 

s_xor_b32 s1, s18, s19 // (op 995) 

s_sub_i32 s18, s1, s19 // (op 995) 


s_add_u32 s54, s4, 32 // (op 995) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 995) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 995) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 995) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 995) 

s_mul_i32 s1, s13, s50 // (op 995) 

s_mul_i32 s19, s6, s1 // (op 995) 

s_mul_i32 s1, s18, s19 // (op 995) 

s_sub_u32 s19, s16, s1 // (op 995) 

s_mov_b32 s1, s19 // (op 995) 
s_mul_hi_i32 s49, s1, s12 // (op 995) 
s_add_i32 s50, s49, s1 // (op 995) 


s_ashr_i32 s1, s50, 31 // (op 995) 
s_cmp_eq_i32 s12, 0 // (op 995) 
s_mov_b32 s49, scc // (op 995) 
s_cmp_eq_i32 s49, 0 // (op 995) 
s_cselect_b32 s51, 0, -1 // (op 995) 

s_and_b32 s49, s48, 31 // (op 995) 
s_lshl_b32 s52, 1, s49 // (op 995) 
s_add_i32 s53, s52, s51 // (op 995) 


s_and_b32 s51, s1, s53 // (op 995) 


s_add_i32 s1, s50, s51 // (op 995) 


s_ashr_i32 s50, s1, s49 // (op 995) 


s_lshl_b32 s1, s48, 24 // (op 995) 
s_ashr_i32 s49, s1, 31 // (op 995) 

s_xor_b32 s1, s50, s49 // (op 995) 

s_sub_i32 s50, s1, s49 // (op 995) 


s_mul_i32 s1, s15, s13 // (op 995) 
s_add_u32 s54, s20, 64 // (op 995) Start of Int64 add, locking SCC
s_addc_u32 s55, s21, 0 // (op 995) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 995) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 995) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 6 // (op 995) 

s_sub_i32 s49, s54, s1 // (op 995) 

s_mul_i32 s51, s50, s49 // (op 995) 


s_sub_u32 s49, s19, s51 // (op 995) 


s_add_u32 s19, s49, s1 // (op 995) 


s_cmp_eq_i32 s18, 0 // (op 995) 
s_mov_b32 s1, scc // (op 995) 

s_cmp_eq_i32 s1, 0 // (op 995) 
s_cselect_b32 s18, s19, s0 // (op 995) 



v_lshl_add_u32 v16, s18, 6, v14 // (op 995) 


v_mov_b32 v19, 0 // (op 995) convert
v_mov_b32 v18, v16 // (op 995) convert

v_mul_lo_u32 v17, v15, s29 // (op 995) most significant: low of low * high
v_mul_hi_u32 v14, v15, s28 // (op 995) most significant: high of low * low
v_mul_lo_u32 v16, v15, s28 // (op 995) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 995) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 995) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 995) most significant half



s_mul_i32 s1, s28, s33 // (op 997) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 997) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 997) least significant
s_mul_hi_u32 s19, s28, s32 // (op 997) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 997) most significant: sum
s_add_u32 s1, s1, s19 // (op 997) most significant: sum


s_mov_b32 s24, s30 // (op 997) reg expression
s_mov_b32 s25, s31 // (op 997) reg expression
s_mov_b32 s26, s0 // (op 997) reg expression
s_mov_b32 s27, 131072 // (op 997) call()

s_mul_hi_i32 s0, s16, s8 // (op 1040) 
v_lshrrev_b32 v16, 3, v1 // (op 1040) 
s_add_i32 s1, s0, s16 // (op 1040) 

v_and_b32 v17, -8, v1 // (op 1040) 
s_ashr_i32 s0, s1, 31 // (op 1040) 
v_sub_u32 v18, v1, v17 // (op 1040) 

s_cmp_eq_i32 s8, 0 // (op 1040) 
s_mov_b32 s18, scc // (op 1040) 
v_lshlrev_b32 v17, 4, v18 // (op 1040) 

s_cmp_eq_i32 s18, 0 // (op 1040) 
s_cselect_b32 s19, 0, -1 // (op 1040) 

v_mov_b32 v19, 0 // (op 1040) convert
v_mov_b32 v18, v17 // (op 1040) convert

s_and_b32 s18, s9, 31 // (op 1040) 
s_lshl_b32 s49, 1, s18 // (op 1040) 
s_add_i32 s50, s49, s19 // (op 1040) 


s_and_b32 s19, s0, s50 // (op 1040) 


s_add_i32 s0, s1, s19 // (op 1040) 


s_ashr_i32 s1, s0, s18 // (op 1040) 


s_lshl_b32 s0, s9, 24 // (op 1040) 
s_ashr_i32 s18, s0, 31 // (op 1040) 

s_xor_b32 s0, s1, s18 // (op 1040) 

s_sub_i32 s1, s0, s18 // (op 1040) 


s_mul_hi_i32 s0, s1, s10 // (op 1040) 
s_add_i32 s18, s0, s1 // (op 1040) 

s_ashr_i32 s0, s18, 31 // (op 1040) 
s_cmp_eq_i32 s10, 0 // (op 1040) 
s_mov_b32 s19, scc // (op 1040) 
s_cmp_eq_i32 s19, 0 // (op 1040) 
s_cselect_b32 s49, 0, -1 // (op 1040) 

s_and_b32 s19, s11, 31 // (op 1040) 
s_lshl_b32 s50, 1, s19 // (op 1040) 
s_add_i32 s51, s50, s49 // (op 1040) 


s_and_b32 s49, s0, s51 // (op 1040) 


s_add_i32 s0, s18, s49 // (op 1040) 


s_ashr_i32 s18, s0, s19 // (op 1040) 


s_lshl_b32 s0, s11, 24 // (op 1040) 
s_ashr_i32 s19, s0, 31 // (op 1040) 

s_xor_b32 s0, s18, s19 // (op 1040) 

s_sub_i32 s18, s0, s19 // (op 1040) 


s_add_u32 s54, s4, 32 // (op 1040) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 1040) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1040) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1040) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 1040) 

s_mul_i32 s0, s18, s50 // (op 1040) 

s_sub_i32 s49, s1, s0 // (op 1040) 


s_mul_hi_i32 s0, s16, s7 // (op 1040) 
s_add_i32 s1, s0, s16 // (op 1040) 

s_ashr_i32 s0, s1, 31 // (op 1040) 
s_cmp_eq_i32 s7, 0 // (op 1040) 
s_mov_b32 s18, scc // (op 1040) 
s_cmp_eq_i32 s18, 0 // (op 1040) 
s_cselect_b32 s19, 0, -1 // (op 1040) 

s_and_b32 s18, s14, 31 // (op 1040) 
s_lshl_b32 s52, 1, s18 // (op 1040) 
s_add_i32 s53, s52, s19 // (op 1040) 


s_and_b32 s19, s0, s53 // (op 1040) 


s_add_i32 s0, s1, s19 // (op 1040) 


s_ashr_i32 s1, s0, s18 // (op 1040) 


s_lshl_b32 s0, s14, 24 // (op 1040) 
s_ashr_i32 s18, s0, 31 // (op 1040) 

s_xor_b32 s0, s1, s18 // (op 1040) 

s_sub_i32 s1, s0, s18 // (op 1040) 


s_mul_i32 s0, s13, s50 // (op 1040) 

s_mul_i32 s18, s6, s0 // (op 1040) 

s_mul_i32 s0, s1, s18 // (op 1040) 

s_sub_u32 s18, s16, s0 // (op 1040) 

s_mul_hi_i32 s0, s18, s12 // (op 1040) 
s_add_i32 s19, s0, s18 // (op 1040) 


s_ashr_i32 s0, s19, 31 // (op 1040) 
s_cmp_eq_i32 s12, 0 // (op 1040) 
s_mov_b32 s18, scc // (op 1040) 
s_cmp_eq_i32 s18, 0 // (op 1040) 
s_cselect_b32 s50, 0, -1 // (op 1040) 

s_and_b32 s18, s48, 31 // (op 1040) 
s_lshl_b32 s51, 1, s18 // (op 1040) 
s_add_i32 s52, s51, s50 // (op 1040) 


s_and_b32 s50, s0, s52 // (op 1040) 


s_add_i32 s0, s19, s50 // (op 1040) 


s_ashr_i32 s19, s0, s18 // (op 1040) 


s_lshl_b32 s0, s48, 24 // (op 1040) 
s_ashr_i32 s18, s0, 31 // (op 1040) 

s_xor_b32 s0, s19, s18 // (op 1040) 

s_sub_i32 s19, s0, s18 // (op 1040) 


s_cmp_eq_i32 s1, 0 // (op 1040) 
s_mov_b32 s0, scc // (op 1040) 

s_cmp_eq_i32 s0, 0 // (op 1040) 
s_cselect_b32 s1, s19, s49 // (op 1040) 



v_lshl_add_u32 v17, s1, 5, v16 // (op 1040) 


v_mul_lo_u32 v21, v17, s35 // (op 1040) most significant: low of low * high
v_mul_hi_u32 v16, v17, s34 // (op 1040) most significant: high of low * low
v_mul_lo_u32 v20, v17, s34 // (op 1040) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 1040) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 1040) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 1040) most significant half



s_mul_i32 s1, s34, s5 // (op 1042) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 1042) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 1042) least significant
s_mul_hi_u32 s19, s34, s4 // (op 1042) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1042) most significant: sum
s_add_u32 s1, s1, s19 // (op 1042) most significant: sum


s_mov_b32 s52, s36 // (op 1042) reg expression
s_mov_b32 s53, s37 // (op 1042) reg expression
s_mov_b32 s54, s0 // (op 1042) reg expression
s_mov_b32 s55, 131072 // (op 1042) call()

s_mul_hi_i32 s0, s16, s8 // (op 1059) 
v_lshrrev_b32 v18, 3, v1 // (op 1059) 
s_add_i32 s1, s0, s16 // (op 1059) 

v_and_b32 v19, -8, v1 // (op 1059) 
s_ashr_i32 s0, s1, 31 // (op 1059) 
v_sub_u32 v20, v1, v19 // (op 1059) 

s_cmp_eq_i32 s8, 0 // (op 1059) 
s_mov_b32 s18, scc // (op 1059) 
v_lshlrev_b32 v19, 4, v20 // (op 1059) 

s_cmp_eq_i32 s18, 0 // (op 1059) 
s_cselect_b32 s19, 0, -1 // (op 1059) 

v_mov_b32 v21, 0 // (op 1059) convert
v_mov_b32 v20, v19 // (op 1059) convert

s_and_b32 s18, s9, 31 // (op 1059) 
s_lshl_b32 s49, 1, s18 // (op 1059) 
s_add_i32 s50, s49, s19 // (op 1059) 


s_and_b32 s19, s0, s50 // (op 1059) 


s_add_i32 s0, s1, s19 // (op 1059) 


s_ashr_i32 s1, s0, s18 // (op 1059) 


s_lshl_b32 s0, s9, 24 // (op 1059) 
s_ashr_i32 s18, s0, 31 // (op 1059) 

s_xor_b32 s0, s1, s18 // (op 1059) 

s_sub_i32 s1, s0, s18 // (op 1059) 


s_mul_hi_i32 s0, s1, s10 // (op 1059) 
s_add_i32 s18, s0, s1 // (op 1059) 

s_ashr_i32 s0, s18, 31 // (op 1059) 
s_cmp_eq_i32 s10, 0 // (op 1059) 
s_mov_b32 s19, scc // (op 1059) 
s_cmp_eq_i32 s19, 0 // (op 1059) 
s_cselect_b32 s49, 0, -1 // (op 1059) 

s_and_b32 s19, s11, 31 // (op 1059) 
s_lshl_b32 s50, 1, s19 // (op 1059) 
s_add_i32 s51, s50, s49 // (op 1059) 


s_and_b32 s49, s0, s51 // (op 1059) 


s_add_i32 s0, s18, s49 // (op 1059) 


s_ashr_i32 s18, s0, s19 // (op 1059) 


s_lshl_b32 s0, s11, 24 // (op 1059) 
s_ashr_i32 s19, s0, 31 // (op 1059) 

s_xor_b32 s0, s18, s19 // (op 1059) 

s_sub_i32 s18, s0, s19 // (op 1059) 


s_add_u32 s50, s4, 32 // (op 1059) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1059) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1059) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1059) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 1059) 

s_mul_i32 s0, s18, s50 // (op 1059) 

s_sub_i32 s49, s1, s0 // (op 1059) 


s_mul_hi_i32 s0, s16, s7 // (op 1059) 
s_add_i32 s1, s0, s16 // (op 1059) 

s_ashr_i32 s0, s1, 31 // (op 1059) 
s_cmp_eq_i32 s7, 0 // (op 1059) 
s_mov_b32 s18, scc // (op 1059) 
s_cmp_eq_i32 s18, 0 // (op 1059) 
s_cselect_b32 s19, 0, -1 // (op 1059) 

s_and_b32 s18, s14, 31 // (op 1059) 
s_lshl_b32 s64, 1, s18 // (op 1059) 
s_add_i32 s65, s64, s19 // (op 1059) 


s_and_b32 s19, s0, s65 // (op 1059) 


s_add_i32 s0, s1, s19 // (op 1059) 


s_ashr_i32 s1, s0, s18 // (op 1059) 


s_lshl_b32 s0, s14, 24 // (op 1059) 
s_ashr_i32 s18, s0, 31 // (op 1059) 

s_xor_b32 s0, s1, s18 // (op 1059) 

s_sub_i32 s1, s0, s18 // (op 1059) 


s_mul_i32 s0, s13, s50 // (op 1059) 

s_mul_i32 s18, s6, s0 // (op 1059) 

s_mul_i32 s0, s1, s18 // (op 1059) 

s_sub_u32 s18, s16, s0 // (op 1059) 

s_mul_hi_i32 s0, s18, s12 // (op 1059) 
s_add_i32 s19, s0, s18 // (op 1059) 


s_ashr_i32 s0, s19, 31 // (op 1059) 
s_cmp_eq_i32 s12, 0 // (op 1059) 
s_mov_b32 s18, scc // (op 1059) 
s_cmp_eq_i32 s18, 0 // (op 1059) 
s_cselect_b32 s50, 0, -1 // (op 1059) 

s_and_b32 s18, s48, 31 // (op 1059) 
s_lshl_b32 s51, 1, s18 // (op 1059) 
s_add_i32 s64, s51, s50 // (op 1059) 


s_and_b32 s50, s0, s64 // (op 1059) 


s_add_i32 s0, s19, s50 // (op 1059) 


s_ashr_i32 s19, s0, s18 // (op 1059) 


s_lshl_b32 s0, s48, 24 // (op 1059) 
s_ashr_i32 s18, s0, 31 // (op 1059) 

s_xor_b32 s0, s19, s18 // (op 1059) 

s_sub_i32 s19, s0, s18 // (op 1059) 


s_cmp_eq_i32 s1, 0 // (op 1059) 
s_mov_b32 s0, scc // (op 1059) 

s_cmp_eq_i32 s0, 0 // (op 1059) 
s_cselect_b32 s1, s19, s49 // (op 1059) 



v_lshl_add_u32 v19, s1, 5, v18 // (op 1059) 


v_mul_lo_u32 v23, v19, s35 // (op 1059) most significant: low of low * high
v_mul_hi_u32 v18, v19, s34 // (op 1059) most significant: high of low * low
v_mul_lo_u32 v22, v19, s34 // (op 1059) least significant: low of low * low
v_add_u32 v23, v23, v18 // (op 1059) most significant: sum


v_add_co_u32 v18, s[0:1], v20, v22 // (op 1059) least significant half
v_addc_co_u32 v19, s[0:1], v21, v23, s[0:1] // (op 1059) most significant half



s_mul_i32 s1, s34, s5 // (op 1061) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 1061) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 1061) least significant
s_mul_hi_u32 s19, s34, s4 // (op 1061) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1061) most significant: sum
s_add_u32 s1, s1, s19 // (op 1061) most significant: sum


s_mov_b32 s52, s36 // (op 1061) reg expression
s_mov_b32 s53, s37 // (op 1061) reg expression
s_mov_b32 s54, s0 // (op 1061) reg expression
s_mov_b32 s55, 131072 // (op 1061) call()

s_mul_hi_i32 s0, s16, s8 // (op 1104) 
v_and_b32 v20, -64, v1 // (op 1104) 
s_add_i32 s1, s0, s16 // (op 1104) 

v_sub_u32 v21, v1, v20 // (op 1104) 

s_ashr_i32 s0, s1, 31 // (op 1104) 
v_lshrrev_b32 v20, 6, v1 // (op 1104) 
s_cmp_eq_i32 s8, 0 // (op 1104) 
s_mov_b32 s18, scc // (op 1104) 
s_cmp_eq_i32 s18, 0 // (op 1104) 
s_cselect_b32 s19, 0, -1 // (op 1104) 

s_and_b32 s18, s9, 31 // (op 1104) 
s_lshl_b32 s49, 1, s18 // (op 1104) 
s_add_i32 s50, s49, s19 // (op 1104) 


s_and_b32 s19, s0, s50 // (op 1104) 


s_add_i32 s0, s1, s19 // (op 1104) 


s_ashr_i32 s1, s0, s18 // (op 1104) 


s_lshl_b32 s0, s9, 24 // (op 1104) 
s_ashr_i32 s18, s0, 31 // (op 1104) 

s_xor_b32 s0, s1, s18 // (op 1104) 

s_sub_i32 s1, s0, s18 // (op 1104) 


s_mul_hi_i32 s0, s1, s10 // (op 1104) 
s_add_i32 s18, s0, s1 // (op 1104) 

s_ashr_i32 s0, s18, 31 // (op 1104) 
s_cmp_eq_i32 s10, 0 // (op 1104) 
s_mov_b32 s19, scc // (op 1104) 
s_cmp_eq_i32 s19, 0 // (op 1104) 
s_cselect_b32 s49, 0, -1 // (op 1104) 

s_and_b32 s19, s11, 31 // (op 1104) 
s_lshl_b32 s50, 1, s19 // (op 1104) 
s_add_i32 s51, s50, s49 // (op 1104) 


s_and_b32 s49, s0, s51 // (op 1104) 


s_add_i32 s0, s18, s49 // (op 1104) 


s_ashr_i32 s18, s0, s19 // (op 1104) 


s_lshl_b32 s0, s11, 24 // (op 1104) 
s_ashr_i32 s19, s0, 31 // (op 1104) 

s_xor_b32 s0, s18, s19 // (op 1104) 

s_sub_i32 s18, s0, s19 // (op 1104) 


s_mul_i32 s0, s1, s13 // (op 1104) 

s_sub_u32 s1, s16, s0 // (op 1104) 

s_mul_i32 s0, s18, s13 // (op 1104) 
s_add_u32 s0, s0, s1 // (op 1104) 


s_mul_hi_i32 s1, s16, s7 // (op 1104) 
s_add_i32 s18, s1, s16 // (op 1104) 

s_ashr_i32 s1, s18, 31 // (op 1104) 
s_cmp_eq_i32 s7, 0 // (op 1104) 
s_mov_b32 s19, scc // (op 1104) 
s_cmp_eq_i32 s19, 0 // (op 1104) 
s_cselect_b32 s49, 0, -1 // (op 1104) 

s_and_b32 s19, s14, 31 // (op 1104) 
s_lshl_b32 s50, 1, s19 // (op 1104) 
s_add_i32 s51, s50, s49 // (op 1104) 


s_and_b32 s49, s1, s51 // (op 1104) 


s_add_i32 s1, s18, s49 // (op 1104) 


s_ashr_i32 s18, s1, s19 // (op 1104) 


s_lshl_b32 s1, s14, 24 // (op 1104) 
s_ashr_i32 s19, s1, 31 // (op 1104) 

s_xor_b32 s1, s18, s19 // (op 1104) 

s_sub_i32 s18, s1, s19 // (op 1104) 


s_add_u32 s50, s4, 32 // (op 1104) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1104) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1104) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1104) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 1104) 

s_mul_i32 s1, s13, s50 // (op 1104) 

s_mul_i32 s19, s6, s1 // (op 1104) 

s_mul_i32 s1, s18, s19 // (op 1104) 

s_sub_u32 s19, s16, s1 // (op 1104) 

s_mov_b32 s1, s19 // (op 1104) 
s_mul_hi_i32 s49, s1, s12 // (op 1104) 
s_add_i32 s50, s49, s1 // (op 1104) 


s_ashr_i32 s1, s50, 31 // (op 1104) 
s_cmp_eq_i32 s12, 0 // (op 1104) 
s_mov_b32 s49, scc // (op 1104) 
s_cmp_eq_i32 s49, 0 // (op 1104) 
s_cselect_b32 s51, 0, -1 // (op 1104) 

s_and_b32 s49, s48, 31 // (op 1104) 
s_lshl_b32 s64, 1, s49 // (op 1104) 
s_add_i32 s65, s64, s51 // (op 1104) 


s_and_b32 s51, s1, s65 // (op 1104) 


s_add_i32 s1, s50, s51 // (op 1104) 


s_ashr_i32 s50, s1, s49 // (op 1104) 


s_lshl_b32 s1, s48, 24 // (op 1104) 
s_ashr_i32 s49, s1, 31 // (op 1104) 

s_xor_b32 s1, s50, s49 // (op 1104) 

s_sub_i32 s50, s1, s49 // (op 1104) 


s_mul_i32 s1, s15, s13 // (op 1104) 
s_add_u32 s64, s20, 64 // (op 1104) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 1104) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1104) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1104) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 6 // (op 1104) 

s_sub_i32 s49, s64, s1 // (op 1104) 

s_mul_i32 s51, s50, s49 // (op 1104) 


s_sub_u32 s49, s19, s51 // (op 1104) 


s_add_u32 s19, s49, s1 // (op 1104) 


s_cmp_eq_i32 s18, 0 // (op 1104) 
s_mov_b32 s1, scc // (op 1104) 

s_cmp_eq_i32 s1, 0 // (op 1104) 
s_cselect_b32 s18, s19, s0 // (op 1104) 



v_lshl_add_u32 v22, s18, 6, v21 // (op 1104) 


v_mul_lo_u32 v21, v20, s38 // (op 1104) 

v_add_u32 v20, v22, v21 // (op 1104) 


s_mul_i32 s1, s38, s43 // (op 1106) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1106) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1106) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1106) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1106) most significant: sum
s_add_u32 s1, s1, s19 // (op 1106) most significant: sum


s_mov_b32 s64, s40 // (op 1106) reg expression
s_mov_b32 s65, s41 // (op 1106) reg expression
s_mov_b32 s66, s0 // (op 1106) reg expression
s_mov_b32 s67, 131072 // (op 1106) call()

s_mul_hi_i32 s0, s16, s8 // (op 1123) 
v_and_b32 v21, -64, v1 // (op 1123) 
s_add_i32 s1, s0, s16 // (op 1123) 

v_sub_u32 v22, v1, v21 // (op 1123) 

s_ashr_i32 s0, s1, 31 // (op 1123) 
v_lshrrev_b32 v21, 6, v1 // (op 1123) 
s_cmp_eq_i32 s8, 0 // (op 1123) 
s_mov_b32 s18, scc // (op 1123) 
s_cmp_eq_i32 s18, 0 // (op 1123) 
s_cselect_b32 s19, 0, -1 // (op 1123) 

s_and_b32 s18, s9, 31 // (op 1123) 
s_lshl_b32 s49, 1, s18 // (op 1123) 
s_add_i32 s50, s49, s19 // (op 1123) 


s_and_b32 s19, s0, s50 // (op 1123) 


s_add_i32 s0, s1, s19 // (op 1123) 


s_ashr_i32 s1, s0, s18 // (op 1123) 


s_lshl_b32 s0, s9, 24 // (op 1123) 
s_ashr_i32 s18, s0, 31 // (op 1123) 

s_xor_b32 s0, s1, s18 // (op 1123) 

s_sub_i32 s1, s0, s18 // (op 1123) 


s_mul_hi_i32 s0, s1, s10 // (op 1123) 
s_add_i32 s18, s0, s1 // (op 1123) 

s_ashr_i32 s0, s18, 31 // (op 1123) 
s_cmp_eq_i32 s10, 0 // (op 1123) 
s_mov_b32 s19, scc // (op 1123) 
s_cmp_eq_i32 s19, 0 // (op 1123) 
s_cselect_b32 s49, 0, -1 // (op 1123) 

s_and_b32 s19, s11, 31 // (op 1123) 
s_lshl_b32 s50, 1, s19 // (op 1123) 
s_add_i32 s51, s50, s49 // (op 1123) 


s_and_b32 s49, s0, s51 // (op 1123) 


s_add_i32 s0, s18, s49 // (op 1123) 


s_ashr_i32 s18, s0, s19 // (op 1123) 


s_lshl_b32 s0, s11, 24 // (op 1123) 
s_ashr_i32 s19, s0, 31 // (op 1123) 

s_xor_b32 s0, s18, s19 // (op 1123) 

s_sub_i32 s18, s0, s19 // (op 1123) 


s_mul_i32 s0, s1, s13 // (op 1123) 

s_sub_u32 s1, s16, s0 // (op 1123) 

s_mul_i32 s0, s18, s13 // (op 1123) 
s_add_u32 s0, s0, s1 // (op 1123) 


s_mul_hi_i32 s1, s16, s7 // (op 1123) 
s_add_i32 s18, s1, s16 // (op 1123) 

s_ashr_i32 s1, s18, 31 // (op 1123) 
s_cmp_eq_i32 s7, 0 // (op 1123) 
s_mov_b32 s19, scc // (op 1123) 
s_cmp_eq_i32 s19, 0 // (op 1123) 
s_cselect_b32 s49, 0, -1 // (op 1123) 

s_and_b32 s19, s14, 31 // (op 1123) 
s_lshl_b32 s50, 1, s19 // (op 1123) 
s_add_i32 s51, s50, s49 // (op 1123) 


s_and_b32 s49, s1, s51 // (op 1123) 


s_add_i32 s1, s18, s49 // (op 1123) 


s_ashr_i32 s18, s1, s19 // (op 1123) 


s_lshl_b32 s1, s14, 24 // (op 1123) 
s_ashr_i32 s19, s1, 31 // (op 1123) 

s_xor_b32 s1, s18, s19 // (op 1123) 

s_sub_i32 s18, s1, s19 // (op 1123) 


s_add_u32 s50, s4, 32 // (op 1123) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1123) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1123) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1123) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 1123) 

s_mul_i32 s1, s13, s50 // (op 1123) 

s_mul_i32 s19, s6, s1 // (op 1123) 

s_mul_i32 s1, s18, s19 // (op 1123) 

s_sub_u32 s19, s16, s1 // (op 1123) 

s_mov_b32 s1, s19 // (op 1123) 
s_mul_hi_i32 s49, s1, s12 // (op 1123) 
s_add_i32 s50, s49, s1 // (op 1123) 


s_ashr_i32 s1, s50, 31 // (op 1123) 
s_cmp_eq_i32 s12, 0 // (op 1123) 
s_mov_b32 s49, scc // (op 1123) 
s_cmp_eq_i32 s49, 0 // (op 1123) 
s_cselect_b32 s51, 0, -1 // (op 1123) 

s_and_b32 s49, s48, 31 // (op 1123) 
s_lshl_b32 s68, 1, s49 // (op 1123) 
s_add_i32 s69, s68, s51 // (op 1123) 


s_and_b32 s51, s1, s69 // (op 1123) 


s_add_i32 s1, s50, s51 // (op 1123) 


s_ashr_i32 s50, s1, s49 // (op 1123) 


s_lshl_b32 s1, s48, 24 // (op 1123) 
s_ashr_i32 s49, s1, 31 // (op 1123) 

s_xor_b32 s1, s50, s49 // (op 1123) 

s_sub_i32 s50, s1, s49 // (op 1123) 


s_mul_i32 s1, s15, s13 // (op 1123) 
s_add_u32 s68, s20, 64 // (op 1123) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 1123) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1123) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1123) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 6 // (op 1123) 

s_sub_i32 s49, s68, s1 // (op 1123) 

s_mul_i32 s51, s50, s49 // (op 1123) 


s_sub_u32 s49, s19, s51 // (op 1123) 


s_add_u32 s19, s49, s1 // (op 1123) 


s_cmp_eq_i32 s18, 0 // (op 1123) 
s_mov_b32 s1, scc // (op 1123) 

s_cmp_eq_i32 s1, 0 // (op 1123) 
s_cselect_b32 s18, s19, s0 // (op 1123) 



v_lshl_add_u32 v23, s18, 6, v22 // (op 1123) 


v_mul_lo_u32 v22, v21, s38 // (op 1123) 

v_add_u32 v21, v23, v22 // (op 1123) 


s_mul_i32 s1, s38, s43 // (op 1125) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1125) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1125) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1125) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1125) most significant: sum
s_add_u32 s1, s1, s19 // (op 1125) most significant: sum


s_mov_b32 s64, s40 // (op 1125) reg expression
s_mov_b32 s65, s41 // (op 1125) reg expression
s_mov_b32 s66, s0 // (op 1125) reg expression
s_mov_b32 s67, 131072 // (op 1125) call()

s_mul_hi_i32 s0, s16, s8 // (op 1168) 
v_lshrrev_b32 v22, 6, v1 // (op 1168) 
s_add_i32 s1, s0, s16 // (op 1168) 

v_and_b32 v23, 1, v22 // (op 1168) 

s_ashr_i32 s0, s1, 31 // (op 1168) 
v_and_b32 v22, 15, v1 // (op 1168) 
s_cmp_eq_i32 s8, 0 // (op 1168) 
s_mov_b32 s18, scc // (op 1168) 
v_lshl_add_u32 v24, v23, 4, v22 // (op 1168) 


s_cmp_eq_i32 s18, 0 // (op 1168) 
s_cselect_b32 s19, 0, -1 // (op 1168) 

v_and_b32 v22, 63, v1 // (op 1168) 
s_and_b32 s18, s9, 31 // (op 1168) 
v_lshrrev_b32 v23, 4, v22 // (op 1168) 

s_lshl_b32 s49, 1, s18 // (op 1168) 
s_add_i32 s50, s49, s19 // (op 1168) 


s_and_b32 s19, s0, s50 // (op 1168) 


s_add_i32 s0, s1, s19 // (op 1168) 


s_ashr_i32 s1, s0, s18 // (op 1168) 


s_lshl_b32 s0, s9, 24 // (op 1168) 
s_ashr_i32 s18, s0, 31 // (op 1168) 

s_xor_b32 s0, s1, s18 // (op 1168) 

s_sub_i32 s1, s0, s18 // (op 1168) 


s_mul_hi_i32 s0, s1, s10 // (op 1168) 
s_add_i32 s18, s0, s1 // (op 1168) 

s_ashr_i32 s0, s18, 31 // (op 1168) 
s_cmp_eq_i32 s10, 0 // (op 1168) 
s_mov_b32 s19, scc // (op 1168) 
s_cmp_eq_i32 s19, 0 // (op 1168) 
s_cselect_b32 s49, 0, -1 // (op 1168) 

s_and_b32 s19, s11, 31 // (op 1168) 
s_lshl_b32 s50, 1, s19 // (op 1168) 
s_add_i32 s51, s50, s49 // (op 1168) 


s_and_b32 s49, s0, s51 // (op 1168) 


s_add_i32 s0, s18, s49 // (op 1168) 


s_ashr_i32 s18, s0, s19 // (op 1168) 


s_lshl_b32 s0, s11, 24 // (op 1168) 
s_ashr_i32 s19, s0, 31 // (op 1168) 

s_xor_b32 s0, s18, s19 // (op 1168) 

s_sub_i32 s18, s0, s19 // (op 1168) 


s_add_u32 s50, s4, 32 // (op 1168) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1168) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1168) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1168) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 1168) 

s_mul_i32 s0, s18, s50 // (op 1168) 

s_sub_i32 s49, s1, s0 // (op 1168) 


s_mul_hi_i32 s0, s16, s7 // (op 1168) 
s_add_i32 s1, s0, s16 // (op 1168) 

s_ashr_i32 s0, s1, 31 // (op 1168) 
s_cmp_eq_i32 s7, 0 // (op 1168) 
s_mov_b32 s18, scc // (op 1168) 
s_cmp_eq_i32 s18, 0 // (op 1168) 
s_cselect_b32 s19, 0, -1 // (op 1168) 

s_and_b32 s18, s14, 31 // (op 1168) 
s_lshl_b32 s68, 1, s18 // (op 1168) 
s_add_i32 s69, s68, s19 // (op 1168) 


s_and_b32 s19, s0, s69 // (op 1168) 


s_add_i32 s0, s1, s19 // (op 1168) 


s_ashr_i32 s1, s0, s18 // (op 1168) 


s_lshl_b32 s0, s14, 24 // (op 1168) 
s_ashr_i32 s18, s0, 31 // (op 1168) 

s_xor_b32 s0, s1, s18 // (op 1168) 

s_sub_i32 s1, s0, s18 // (op 1168) 


s_mul_i32 s0, s13, s50 // (op 1168) 

s_mul_i32 s18, s6, s0 // (op 1168) 

s_mul_i32 s0, s1, s18 // (op 1168) 

s_sub_u32 s18, s16, s0 // (op 1168) 

s_mul_hi_i32 s0, s18, s12 // (op 1168) 
s_add_i32 s19, s0, s18 // (op 1168) 


s_ashr_i32 s0, s19, 31 // (op 1168) 
s_cmp_eq_i32 s12, 0 // (op 1168) 
s_mov_b32 s18, scc // (op 1168) 
s_cmp_eq_i32 s18, 0 // (op 1168) 
s_cselect_b32 s50, 0, -1 // (op 1168) 

s_and_b32 s18, s48, 31 // (op 1168) 
s_lshl_b32 s51, 1, s18 // (op 1168) 
s_add_i32 s68, s51, s50 // (op 1168) 


s_and_b32 s50, s0, s68 // (op 1168) 


s_add_i32 s0, s19, s50 // (op 1168) 


s_ashr_i32 s19, s0, s18 // (op 1168) 


s_lshl_b32 s0, s48, 24 // (op 1168) 
s_ashr_i32 s18, s0, 31 // (op 1168) 

s_xor_b32 s0, s19, s18 // (op 1168) 

s_sub_i32 s19, s0, s18 // (op 1168) 


s_cmp_eq_i32 s1, 0 // (op 1168) 
s_mov_b32 s0, scc // (op 1168) 

s_cmp_eq_i32 s0, 0 // (op 1168) 
s_cselect_b32 s1, s19, s49 // (op 1168) 



v_lshl_add_u32 v22, s1, 5, v24 // (op 1168) 


v_mul_lo_u32 v24, v22, s44 // (op 1168) 

v_add_u32 v22, v23, v24 // (op 1168) 


s_mul_i32 s1, s44, s5 // (op 1170) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 1170) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 1170) least significant
s_mul_hi_u32 s19, s44, s4 // (op 1170) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1170) most significant: sum
s_add_u32 s1, s1, s19 // (op 1170) most significant: sum


s_mov_b32 s68, s46 // (op 1170) reg expression
s_mov_b32 s69, s47 // (op 1170) reg expression
s_mov_b32 s70, s0 // (op 1170) reg expression
s_mov_b32 s71, 131072 // (op 1170) call()

s_mul_hi_i32 s0, s16, s8 // (op 1187) 
v_lshrrev_b32 v23, 6, v1 // (op 1187) 
s_add_i32 s1, s0, s16 // (op 1187) 

v_and_b32 v24, 1, v23 // (op 1187) 

s_ashr_i32 s0, s1, 31 // (op 1187) 
v_and_b32 v23, 15, v1 // (op 1187) 
s_cmp_eq_i32 s8, 0 // (op 1187) 
s_mov_b32 s18, scc // (op 1187) 
v_lshl_add_u32 v25, v24, 4, v23 // (op 1187) 


s_cmp_eq_i32 s18, 0 // (op 1187) 
s_cselect_b32 s19, 0, -1 // (op 1187) 

v_and_b32 v23, 63, v1 // (op 1187) 
s_and_b32 s18, s9, 31 // (op 1187) 
v_lshrrev_b32 v24, 4, v23 // (op 1187) 

s_lshl_b32 s49, 1, s18 // (op 1187) 
s_add_i32 s50, s49, s19 // (op 1187) 


s_and_b32 s19, s0, s50 // (op 1187) 


s_add_i32 s0, s1, s19 // (op 1187) 


s_ashr_i32 s1, s0, s18 // (op 1187) 


s_lshl_b32 s0, s9, 24 // (op 1187) 
s_ashr_i32 s18, s0, 31 // (op 1187) 

s_xor_b32 s0, s1, s18 // (op 1187) 

s_sub_i32 s1, s0, s18 // (op 1187) 


s_mul_hi_i32 s0, s1, s10 // (op 1187) 
s_add_i32 s18, s0, s1 // (op 1187) 

s_ashr_i32 s0, s18, 31 // (op 1187) 
s_cmp_eq_i32 s10, 0 // (op 1187) 
s_mov_b32 s19, scc // (op 1187) 
s_cmp_eq_i32 s19, 0 // (op 1187) 
s_cselect_b32 s49, 0, -1 // (op 1187) 

s_and_b32 s19, s11, 31 // (op 1187) 
s_lshl_b32 s50, 1, s19 // (op 1187) 
s_add_i32 s51, s50, s49 // (op 1187) 


s_and_b32 s49, s0, s51 // (op 1187) 


s_add_i32 s0, s18, s49 // (op 1187) 


s_ashr_i32 s18, s0, s19 // (op 1187) 


s_lshl_b32 s0, s11, 24 // (op 1187) 
s_ashr_i32 s19, s0, 31 // (op 1187) 

s_xor_b32 s0, s18, s19 // (op 1187) 

s_sub_i32 s18, s0, s19 // (op 1187) 


s_add_u32 s50, s4, 32 // (op 1187) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1187) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1187) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1187) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1187) 

s_mul_i32 s0, s18, s50 // (op 1187) 

s_sub_i32 s49, s1, s0 // (op 1187) 


s_mul_hi_i32 s0, s16, s7 // (op 1187) 
s_add_i32 s1, s0, s16 // (op 1187) 

s_ashr_i32 s0, s1, 31 // (op 1187) 
s_cmp_eq_i32 s7, 0 // (op 1187) 
s_mov_b32 s18, scc // (op 1187) 
s_cmp_eq_i32 s18, 0 // (op 1187) 
s_cselect_b32 s19, 0, -1 // (op 1187) 

s_and_b32 s18, s14, 31 // (op 1187) 
s_lshl_b32 s72, 1, s18 // (op 1187) 
s_add_i32 s73, s72, s19 // (op 1187) 


s_and_b32 s19, s0, s73 // (op 1187) 


s_add_i32 s0, s1, s19 // (op 1187) 


s_ashr_i32 s1, s0, s18 // (op 1187) 


s_lshl_b32 s0, s14, 24 // (op 1187) 
s_ashr_i32 s18, s0, 31 // (op 1187) 

s_xor_b32 s0, s1, s18 // (op 1187) 

s_sub_i32 s1, s0, s18 // (op 1187) 


s_mul_i32 s0, s13, s50 // (op 1187) 

s_mul_i32 s18, s6, s0 // (op 1187) 

s_mul_i32 s0, s1, s18 // (op 1187) 

s_sub_u32 s18, s16, s0 // (op 1187) 

s_mul_hi_i32 s0, s18, s12 // (op 1187) 
s_add_i32 s19, s0, s18 // (op 1187) 


s_ashr_i32 s0, s19, 31 // (op 1187) 
s_cmp_eq_i32 s12, 0 // (op 1187) 
s_mov_b32 s18, scc // (op 1187) 
s_cmp_eq_i32 s18, 0 // (op 1187) 
s_cselect_b32 s50, 0, -1 // (op 1187) 

s_and_b32 s18, s48, 31 // (op 1187) 
s_lshl_b32 s51, 1, s18 // (op 1187) 
s_add_i32 s72, s51, s50 // (op 1187) 


s_and_b32 s50, s0, s72 // (op 1187) 


s_add_i32 s0, s19, s50 // (op 1187) 


s_ashr_i32 s19, s0, s18 // (op 1187) 


s_lshl_b32 s0, s48, 24 // (op 1187) 
s_ashr_i32 s18, s0, 31 // (op 1187) 

s_xor_b32 s0, s19, s18 // (op 1187) 

s_sub_i32 s19, s0, s18 // (op 1187) 


s_cmp_eq_i32 s1, 0 // (op 1187) 
s_mov_b32 s0, scc // (op 1187) 

s_cmp_eq_i32 s0, 0 // (op 1187) 
s_cselect_b32 s1, s19, s49 // (op 1187) 



v_lshl_add_u32 v23, s1, 5, v25 // (op 1187) 


v_mul_lo_u32 v25, v23, s44 // (op 1187) 

v_add_u32 v23, v24, v25 // (op 1187) 


s_mul_i32 s1, s44, s5 // (op 1189) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 1189) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 1189) least significant
s_mul_hi_u32 s19, s44, s4 // (op 1189) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1189) most significant: sum
s_add_u32 s1, s1, s19 // (op 1189) most significant: sum


s_mov_b32 s68, s46 // (op 1189) reg expression
s_mov_b32 s69, s47 // (op 1189) reg expression
s_mov_b32 s70, s0 // (op 1189) reg expression
s_mov_b32 s71, 131072 // (op 1189) call()




buffer_load_ubyte v24, v22, s[68:71], 0 offen offset: 0 // (op 782) Load value
v_and_b32 v25, -8, v1 // (op 1310) 
v_sub_u32 v26, v1, v25 // (op 1310) 

v_lshlrev_b32 v25, 4, v26 // (op 1310) 

v_lshrrev_b32 v26, 3, v1 // (op 1310) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 1310) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1310) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 924) reg expression
buffer_load_dwordx4 v16, s[52:55], 0 offen offset: 0 lds // (op 924) Load value direct to lds

v_and_b32 v25, -4, v1 // (op 1253) 
v_sub_u32 v26, v1, v25 // (op 1253) 

v_lshlrev_b32 v25, 4, v26 // (op 1253) 

v_lshrrev_b32 v26, 2, v1 // (op 1253) 
v_lshl_add_u32 v27, v26, 6, v25 // (op 1253) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1253) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 6 // (op 929) 
v_mov_b64 v[26:27], s[18:19] // (op 929) convert


v_mov_b64 v[28:29], v[12:13] // (op 929) reg expression
s_add_u32 s1, 4096, s0 // (op 929) 
s_mov_b32 m0, s1 // (op 929) reg expression
buffer_load_dwordx4 v28, s[24:27], 0 offen offset: 0 lds // (op 929) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 929) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 929) most significant half

s_add_u32 m0, m0, 4096 // (op 929) 
buffer_load_dwordx4 v28, s[24:27], 0 offen offset: 0 lds // (op 929) Load value direct to lds






v_mov_b32 v25, v20 // (op 792) reg expression

buffer_load_ubyte v26, v25, s[64:67], 0 offen offset: 0 // (op 792) Load value




buffer_load_ubyte v25, v22, s[68:71], 0 offen offset:4 // (op 796) Load value

v_and_b32 v22, -8, v1 // (op 1469) 
v_sub_u32 v27, v1, v22 // (op 1469) 

v_lshlrev_b32 v22, 4, v27 // (op 1469) 

v_lshrrev_b32 v27, 3, v1 // (op 1469) 
v_lshl_add_u32 v28, v27, 7, v22 // (op 1469) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1469) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 12288, s0 // (op 934) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 934) reg expression
s_mov_b32 s18, 128 // (op 934) call()
buffer_load_dwordx4 v16, s[52:55], s18 offen offset: 0 lds // (op 934) Load value direct to lds




v_and_b32 v22, -4, v1 // (op 1425) 
v_sub_u32 v16, v1, v22 // (op 1425) 

v_lshlrev_b32 v17, 4, v16 // (op 1425) 

v_lshrrev_b32 v16, 2, v1 // (op 1425) 
v_lshl_add_u32 v22, v16, 6, v17 // (op 1425) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 1425) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 6 // (op 936) 
v_mov_b64 v[16:17], s[18:19] // (op 936) convert


s_lshl_b64 s[18:19], s[28:29], 7 // (op 936) 
v_mov_b32 v22, s18 // (op 936) 
v_mov_b32 v27, s19 // (op 936) 
v_add_co_u32 v28, s[50:51], v12, v22 // (op 936) least significant half
v_addc_co_u32 v29, s[50:51], v13, v27, s[50:51] // (op 936) most significant half




s_add_u32 s1, 16384, s0 // (op 936) 
s_mov_b32 m0, s1 // (op 936) reg expression
buffer_load_dwordx4 v28, s[24:27], 0 offen offset: 0 lds // (op 936) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v16 // (op 936) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 936) most significant half

s_add_u32 m0, m0, 4096 // (op 936) 
buffer_load_dwordx4 v28, s[24:27], 0 offen offset: 0 lds // (op 936) Load value direct to lds







s_lshl_b32 s0, s38, 2 // (op 806) 
v_add_u32 v22, v20, s0 // (op 806) 


buffer_load_ubyte v12, v22, s[64:67], 0 offen offset: 0 // (op 806) Load value


v_and_b32 v13, -64, v1 // (op 1363) 
v_sub_u32 v20, v1, v13 // (op 1363) 

v_and_b32 v13, 4294967232, v1 // (op 1363) 
v_add_u32 v22, v13, v20 // (op 1363) 





s_waitcnt vmcnt(5)
ds_write_b8 v22, v26 offset:24576 // (op 810) Store local data 

s_barrier  // (op 1567) 



ds_read_b128 v[28:31], v3 // (op 833) Load local data 
ds_read_b128 v[32:35], v3 offset:64 // (op 833) Load local data 
s_barrier  // (op 1551) 



ds_read_b64_tr_b8 v[36:37], v0 offset:4096 // (op 824) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:4608 // (op 824) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:8192 // (op 824) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:8704 // (op 824) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1583) 


ds_read_u8 v13, v7 offset:24576 // (op 837) Load local data 



ds_read_b64_tr_b8 v[44:45], v0 offset:4128 // (op 829) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:4640 // (op 829) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:8224 // (op 829) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:8736 // (op 829) Transpose load local data 


ds_read_u8 v20, v7 offset:24608 // (op 842) Load local data 
v_and_b32 v22, -4, v1 // (op 1225) 
v_sub_u32 v27, v1, v22 // (op 1225) 

v_lshlrev_b32 v22, 4, v27 // (op 1225) 

v_lshrrev_b32 v27, 2, v1 // (op 1225) 
v_lshl_add_u32 v16, v27, 6, v22 // (op 1225) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 1225) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 1282) 
v_sub_u32 v27, v1, v22 // (op 1282) 

v_lshlrev_b32 v22, 4, v27 // (op 1282) 

v_lshrrev_b32 v27, 3, v1 // (op 1282) 
v_lshl_add_u32 v16, v27, 7, v22 // (op 1282) 


s_nop 0
v_readfirstlane_b32 s1, v16 // (op 1282) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -64, v1 // (op 1334) 
v_sub_u32 v27, v1, v22 // (op 1334) 

v_and_b32 v22, 4294967232, v1 // (op 1334) 
v_add_u32 v16, v22, v27 // (op 1334) 


v_and_b32 v17, -4, v1 // (op 1410) 
v_sub_u32 v22, v1, v17 // (op 1410) 

v_lshlrev_b32 v17, 4, v22 // (op 1410) 

v_lshrrev_b32 v22, 2, v1 // (op 1410) 
v_lshl_add_u32 v27, v22, 6, v17 // (op 1410) 


s_nop 0
v_readfirstlane_b32 s18, v27 // (op 1410) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -8, v1 // (op 1454) 
v_sub_u32 v22, v1, v17 // (op 1454) 

v_lshlrev_b32 v17, 4, v22 // (op 1454) 

v_lshrrev_b32 v22, 3, v1 // (op 1454) 
v_lshl_add_u32 v27, v22, 7, v17 // (op 1454) 


s_nop 0
v_readfirstlane_b32 s19, v27 // (op 1454) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -64, v1 // (op 1493) 
v_sub_u32 v22, v1, v17 // (op 1493) 

v_and_b32 v17, 4294967232, v1 // (op 1493) 
v_add_u32 v27, v17, v22 // (op 1493) 


s_mov_b32 s49, 0 // (op 315) call()
s_ashr_i64 s[50:51], s[32:33], 7 // (op 314) 
s_mov_b32 s72, s50 // (op 314) convert

s_and_b32 s50, s72, 4294967294 // (op 314) 

s_cmp_lt_u32 s49, s50 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_314 // (op 314) Condition: Top (jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_314 if false)
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_314:
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[28:35], a[0:3], v13, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 84) 
s_add_u32 s49, s49, 2 // (op 316) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[28:35], a[4:7], v20, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 350) 


v_mov_b32 v17, v23 // (op 82) 

buffer_load_ubyte v24, v17, s[68:71], 0 offen offset:8 // (op 82) Load value
s_barrier  // (op 1570) 


v_mov_b64 v[52:53], v[18:19] // (op 896) 

s_mov_b32 m0, s1 // (op 896) reg expression
s_mov_b32 s50, 256 // (op 896) call()
buffer_load_dwordx4 v52, s[52:55], s50 offen offset: 0 lds // (op 896) Load value direct to lds

s_barrier  // (op 1554) 

s_lshl_b64 s[50:51], s[28:29], 6 // (op 891) 
v_mov_b64 v[54:55], s[50:51] // (op 891) convert


v_mov_b64 v[56:57], v[14:15] // (op 891) 
s_lshl_b64 s[50:51], s[28:29], 7 // (op 891) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 891) 

v_mov_b32 v22, s72 // (op 891) 
v_mov_b32 v58, s73 // (op 891) 
v_add_co_u32 v60, s[50:51], v56, v22 // (op 891) least significant half
v_addc_co_u32 v61, s[50:51], v57, v58, s[50:51] // (op 891) most significant half




s_add_u32 s50, 4096, s0 // (op 891) 
s_nop 0
s_mov_b32 m0, s50 // (op 891) reg expression
buffer_load_dwordx4 v60, s[24:27], 0 offen offset: 0 lds // (op 891) Load value direct to lds
v_add_co_u32 v60, s[72:73], v60, v54 // (op 891) least significant half
v_addc_co_u32 v61, s[72:73], 0, v61, s[72:73] // (op 891) most significant half

s_add_u32 m0, m0, 4096 // (op 891) 
buffer_load_dwordx4 v60, s[24:27], 0 offen offset: 0 lds // (op 891) Load value direct to lds





v_mov_b32 v22, v21 // (op 17) 
s_lshl_b32 s50, s38, 3 // (op 17) 
v_add_u32 v54, v22, s50 // (op 17) 


buffer_load_ubyte v26, v54, s[64:67], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1615) 



s_waitcnt vmcnt(5)
ds_write_b8 v27, v12 offset:24832 // (op 240) Store local data 
s_barrier  // (op 1606) 



ds_read_b128 v[28:31], v4 offset:12288 // (op 233) Load local data 
ds_read_b128 v[32:35], v4 offset:12352 // (op 233) Load local data 
s_barrier  // (op 1600) 



ds_read_b64_tr_b8 v[36:37], v10 offset:16384 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v10 offset:16896 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v10 offset:20480 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v10 offset:20992 // (op 231) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1612) 


ds_read_u8 v13, v11 offset:24832 // (op 235) Load local data 



ds_read_b64_tr_b8 v[44:45], v10 offset:16416 // (op 357) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v10 offset:16928 // (op 357) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v10 offset:20512 // (op 357) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v10 offset:21024 // (op 357) Transpose load local data 


ds_read_u8 v20, v11 offset:24864 // (op 361) Load local data 
s_barrier  // (op 1670) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[28:35], a[0:3], v13, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 238) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[28:35], a[4:7], v20, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 351) 



buffer_load_ubyte v25, v17, s[68:71], 0 offen offset:12 // (op 237) Load value

s_barrier  // (op 1609) 
v_add_u32 v23, 8, v23 // (op 1179) 


s_add_u32 s50, 12288, s19 // (op 919) 

s_mov_b32 m0, s50 // (op 919) reg expression
s_mov_b32 s51, 384 // (op 919) call()
buffer_load_dwordx4 v52, s[52:55], s51 offen offset: 0 lds // (op 919) Load value direct to lds



s_barrier  // (op 1603) 
v_add_co_u32 v18, vcc, 256, v18 // (op 1051) least significant half

v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1051) most significant half
s_lshl_b64 s[50:51], s[28:29], 6 // (op 914) 
v_mov_b64 v[52:53], s[50:51] // (op 914) convert


s_lshl_b64 s[50:51], s[28:29], 7 // (op 914) 
s_mul_i32 s73, 3, s51 // (op 914) most significant: low of low * high
s_mul_i32 s72, 3, s50 // (op 914) least significant
s_mul_hi_u32 s74, 3, s50 // (op 914) most significant: high of low * low
s_add_u32 s73, s73, s74 // (op 914) most significant: sum


v_mov_b32 v17, s72 // (op 914) 
v_mov_b32 v54, s73 // (op 914) 
v_add_co_u32 v58, s[50:51], v56, v17 // (op 914) least significant half
v_addc_co_u32 v59, s[50:51], v57, v54, s[50:51] // (op 914) most significant half




s_add_u32 s50, 16384, s18 // (op 914) 
s_nop 0
s_mov_b32 m0, s50 // (op 914) reg expression
buffer_load_dwordx4 v58, s[24:27], 0 offen offset: 0 lds // (op 914) Load value direct to lds
v_add_co_u32 v58, s[72:73], v58, v52 // (op 914) least significant half
v_addc_co_u32 v59, s[72:73], 0, v59, s[72:73] // (op 914) most significant half

s_add_u32 m0, m0, 4096 // (op 914) 
buffer_load_dwordx4 v58, s[24:27], 0 offen offset: 0 lds // (op 914) Load value direct to lds






s_lshl_b32 s50, s38, 2 // (op 242) 
s_mul_i32 s51, 3, s50 // (op 242) 

v_add_u32 v17, v22, s51 // (op 242) 


s_lshl_b64 s[50:51], s[28:29], 7 // (op 987) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 987) 

v_mov_b32 v52, s72 // (op 987) 
v_mov_b32 v53, s73 // (op 987) 
v_add_co_u32 v14, s[50:51], v14, v52 // (op 987) least significant half
v_addc_co_u32 v15, s[50:51], v15, v53, s[50:51] // (op 987) most significant half




buffer_load_ubyte v12, v17, s[64:67], 0 offen offset: 0 // (op 242) Load value


s_lshl_b32 s50, s38, 3 // (op 1115) 
s_barrier  // (op 1586) 
v_add_u32 v21, v21, s50 // (op 1115) 




s_waitcnt vmcnt(5)
ds_write_b8 v16, v26 offset:24576 // (op 21) Store local data 
s_barrier  // (op 1576) 



ds_read_b128 v[28:31], v3 // (op 72) Load local data 
ds_read_b128 v[32:35], v3 offset:64 // (op 72) Load local data 
s_barrier  // (op 1560) 



ds_read_b64_tr_b8 v[36:37], v0 offset:4096 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:4608 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:8192 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:8704 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1593) 


ds_read_u8 v13, v7 offset:24576 // (op 77) Load local data 



ds_read_b64_tr_b8 v[44:45], v0 offset:4128 // (op 344) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:4640 // (op 344) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:8224 // (op 344) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:8736 // (op 344) Transpose load local data 


ds_read_u8 v20, v7 offset:24608 // (op 348) Load local data 
s_ashr_i64 s[50:51], s[32:33], 7 // (op 314) 
s_mov_b32 s72, s50 // (op 314) convert

s_and_b32 s50, s72, 4294967294 // (op 314) 

s_cmp_lt_u32 s49, s50 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_314 // (op 314) Condition: Bottom (jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_314 if true)
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_314:













s_barrier  // (op 1654) 
s_mul_hi_i32 s0, s16, s8 // (op 949) 
v_and_b32 v14, -4, v1 // (op 949) 
s_add_i32 s1, s0, s16 // (op 949) 

v_sub_u32 v15, v1, v14 // (op 949) 

s_ashr_i32 s0, s1, 31 // (op 949) 
v_lshlrev_b32 v14, 4, v15 // (op 949) 

s_cmp_eq_i32 s8, 0 // (op 949) 
s_mov_b32 s18, scc // (op 949) 
v_lshrrev_b32 v15, 2, v1 // (op 949) 
s_cmp_eq_i32 s18, 0 // (op 949) 
s_cselect_b32 s19, 0, -1 // (op 949) 

s_and_b32 s18, s9, 31 // (op 949) 
s_lshl_b32 s49, 1, s18 // (op 949) 
s_add_i32 s50, s49, s19 // (op 949) 


s_and_b32 s19, s0, s50 // (op 949) 


s_add_i32 s0, s1, s19 // (op 949) 


s_ashr_i32 s1, s0, s18 // (op 949) 


s_lshl_b32 s0, s9, 24 // (op 949) 
s_ashr_i32 s18, s0, 31 // (op 949) 

s_xor_b32 s0, s1, s18 // (op 949) 

s_sub_i32 s1, s0, s18 // (op 949) 


s_mul_hi_i32 s0, s1, s10 // (op 949) 
s_add_i32 s18, s0, s1 // (op 949) 

s_ashr_i32 s0, s18, 31 // (op 949) 
s_cmp_eq_i32 s10, 0 // (op 949) 
s_mov_b32 s19, scc // (op 949) 
s_cmp_eq_i32 s19, 0 // (op 949) 
s_cselect_b32 s49, 0, -1 // (op 949) 

s_and_b32 s19, s11, 31 // (op 949) 
s_lshl_b32 s50, 1, s19 // (op 949) 
s_add_i32 s51, s50, s49 // (op 949) 


s_and_b32 s49, s0, s51 // (op 949) 


s_add_i32 s0, s18, s49 // (op 949) 


s_ashr_i32 s18, s0, s19 // (op 949) 


s_lshl_b32 s0, s11, 24 // (op 949) 
s_ashr_i32 s19, s0, 31 // (op 949) 

s_xor_b32 s0, s18, s19 // (op 949) 

s_sub_i32 s18, s0, s19 // (op 949) 


s_mul_i32 s0, s1, s13 // (op 949) 

s_sub_u32 s1, s16, s0 // (op 949) 

s_mul_i32 s0, s18, s13 // (op 949) 
s_add_u32 s0, s0, s1 // (op 949) 


s_mul_hi_i32 s1, s16, s7 // (op 949) 
s_add_i32 s18, s1, s16 // (op 949) 

s_ashr_i32 s1, s18, 31 // (op 949) 
s_cmp_eq_i32 s7, 0 // (op 949) 
s_mov_b32 s19, scc // (op 949) 
s_cmp_eq_i32 s19, 0 // (op 949) 
s_cselect_b32 s49, 0, -1 // (op 949) 

s_and_b32 s19, s14, 31 // (op 949) 
s_lshl_b32 s50, 1, s19 // (op 949) 
s_add_i32 s51, s50, s49 // (op 949) 


s_and_b32 s49, s1, s51 // (op 949) 


s_add_i32 s1, s18, s49 // (op 949) 


s_ashr_i32 s18, s1, s19 // (op 949) 


s_lshl_b32 s1, s14, 24 // (op 949) 
s_ashr_i32 s19, s1, 31 // (op 949) 

s_xor_b32 s1, s18, s19 // (op 949) 

s_sub_i32 s18, s1, s19 // (op 949) 


s_add_u32 s50, s4, 32 // (op 949) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 949) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 949) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 949) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 949) 

s_mul_i32 s1, s13, s50 // (op 949) 

s_mul_i32 s19, s6, s1 // (op 949) 

s_mul_i32 s1, s18, s19 // (op 949) 

s_sub_u32 s19, s16, s1 // (op 949) 

s_mov_b32 s1, s19 // (op 949) 
s_mul_hi_i32 s49, s1, s12 // (op 949) 
s_add_i32 s50, s49, s1 // (op 949) 


s_ashr_i32 s1, s50, 31 // (op 949) 
s_cmp_eq_i32 s12, 0 // (op 949) 
s_mov_b32 s49, scc // (op 949) 
s_cmp_eq_i32 s49, 0 // (op 949) 
s_cselect_b32 s51, 0, -1 // (op 949) 

s_and_b32 s49, s48, 31 // (op 949) 
s_lshl_b32 s72, 1, s49 // (op 949) 
s_add_i32 s73, s72, s51 // (op 949) 


s_and_b32 s51, s1, s73 // (op 949) 


s_add_i32 s1, s50, s51 // (op 949) 


s_ashr_i32 s50, s1, s49 // (op 949) 


s_lshl_b32 s1, s48, 24 // (op 949) 
s_ashr_i32 s49, s1, 31 // (op 949) 

s_xor_b32 s1, s50, s49 // (op 949) 

s_sub_i32 s50, s1, s49 // (op 949) 


s_mul_i32 s1, s15, s13 // (op 949) 
s_add_u32 s72, s20, 64 // (op 949) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 949) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 949) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 949) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 949) 

s_sub_i32 s49, s72, s1 // (op 949) 

s_mul_i32 s51, s50, s49 // (op 949) 


s_sub_u32 s49, s19, s51 // (op 949) 


s_add_u32 s19, s49, s1 // (op 949) 


s_cmp_eq_i32 s18, 0 // (op 949) 
s_mov_b32 s1, scc // (op 949) 

s_cmp_eq_i32 s1, 0 // (op 949) 
s_cselect_b32 s18, s19, s0 // (op 949) 



v_lshl_add_u32 v16, s18, 6, v14 // (op 949) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 949) 
v_mov_b32 v19, 0 // (op 949) convert
v_mov_b32 v18, v16 // (op 949) convert

s_lshr_b32 s18, s0, 1 // (op 949) 

v_lshl_add_u32 v14, s18, 8, v15 // (op 949) 


v_mul_lo_u32 v17, v14, s29 // (op 949) most significant: low of low * high
v_mul_hi_u32 v15, v14, s28 // (op 949) most significant: high of low * low
v_mul_lo_u32 v16, v14, s28 // (op 949) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 949) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 949) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 949) most significant half



s_mul_i32 s1, s28, s33 // (op 951) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 951) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 951) least significant
s_mul_hi_u32 s19, s28, s32 // (op 951) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 951) most significant: sum
s_add_u32 s1, s1, s19 // (op 951) most significant: sum


s_mov_b32 s24, s30 // (op 951) reg expression
s_mov_b32 s25, s31 // (op 951) reg expression
s_mov_b32 s26, s0 // (op 951) reg expression
s_mov_b32 s27, 131072 // (op 951) call()

s_mul_hi_i32 s0, s16, s8 // (op 1017) 
v_lshrrev_b32 v16, 3, v1 // (op 1017) 
s_add_i32 s1, s0, s16 // (op 1017) 

v_and_b32 v17, -8, v1 // (op 1017) 
s_ashr_i32 s0, s1, 31 // (op 1017) 
v_sub_u32 v18, v1, v17 // (op 1017) 

s_cmp_eq_i32 s8, 0 // (op 1017) 
s_mov_b32 s18, scc // (op 1017) 
v_lshlrev_b32 v17, 4, v18 // (op 1017) 

s_cmp_eq_i32 s18, 0 // (op 1017) 
s_cselect_b32 s19, 0, -1 // (op 1017) 

s_and_b32 s18, s9, 31 // (op 1017) 
s_lshl_b32 s49, 1, s18 // (op 1017) 
s_add_i32 s50, s49, s19 // (op 1017) 


s_and_b32 s19, s0, s50 // (op 1017) 


s_add_i32 s0, s1, s19 // (op 1017) 


s_ashr_i32 s1, s0, s18 // (op 1017) 


s_lshl_b32 s0, s9, 24 // (op 1017) 
s_ashr_i32 s18, s0, 31 // (op 1017) 

s_xor_b32 s0, s1, s18 // (op 1017) 

s_sub_i32 s1, s0, s18 // (op 1017) 


s_mul_hi_i32 s0, s1, s10 // (op 1017) 
s_add_i32 s18, s0, s1 // (op 1017) 

s_ashr_i32 s0, s18, 31 // (op 1017) 
s_cmp_eq_i32 s10, 0 // (op 1017) 
s_mov_b32 s19, scc // (op 1017) 
s_cmp_eq_i32 s19, 0 // (op 1017) 
s_cselect_b32 s49, 0, -1 // (op 1017) 

s_and_b32 s19, s11, 31 // (op 1017) 
s_lshl_b32 s50, 1, s19 // (op 1017) 
s_add_i32 s51, s50, s49 // (op 1017) 


s_and_b32 s49, s0, s51 // (op 1017) 


s_add_i32 s0, s18, s49 // (op 1017) 


s_ashr_i32 s18, s0, s19 // (op 1017) 


s_lshl_b32 s0, s11, 24 // (op 1017) 
s_ashr_i32 s19, s0, 31 // (op 1017) 

s_xor_b32 s0, s18, s19 // (op 1017) 

s_sub_i32 s18, s0, s19 // (op 1017) 


s_add_u32 s50, s4, 32 // (op 1017) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1017) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1017) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1017) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1017) 

s_mul_i32 s0, s18, s50 // (op 1017) 

s_sub_i32 s49, s1, s0 // (op 1017) 


s_mul_hi_i32 s0, s16, s7 // (op 1017) 
s_add_i32 s1, s0, s16 // (op 1017) 

s_ashr_i32 s0, s1, 31 // (op 1017) 
s_cmp_eq_i32 s7, 0 // (op 1017) 
s_mov_b32 s18, scc // (op 1017) 
s_cmp_eq_i32 s18, 0 // (op 1017) 
s_cselect_b32 s19, 0, -1 // (op 1017) 

s_and_b32 s18, s14, 31 // (op 1017) 
s_lshl_b32 s72, 1, s18 // (op 1017) 
s_add_i32 s73, s72, s19 // (op 1017) 


s_and_b32 s19, s0, s73 // (op 1017) 


s_add_i32 s0, s1, s19 // (op 1017) 


s_ashr_i32 s1, s0, s18 // (op 1017) 


s_lshl_b32 s0, s14, 24 // (op 1017) 
s_ashr_i32 s18, s0, 31 // (op 1017) 

s_xor_b32 s0, s1, s18 // (op 1017) 

s_sub_i32 s1, s0, s18 // (op 1017) 


s_mul_i32 s0, s13, s50 // (op 1017) 

s_mul_i32 s18, s6, s0 // (op 1017) 

s_mul_i32 s0, s1, s18 // (op 1017) 

s_sub_u32 s18, s16, s0 // (op 1017) 

s_mul_hi_i32 s0, s18, s12 // (op 1017) 
s_add_i32 s19, s0, s18 // (op 1017) 


s_ashr_i32 s0, s19, 31 // (op 1017) 
s_cmp_eq_i32 s12, 0 // (op 1017) 
s_mov_b32 s18, scc // (op 1017) 
s_cmp_eq_i32 s18, 0 // (op 1017) 
s_cselect_b32 s50, 0, -1 // (op 1017) 

s_and_b32 s18, s48, 31 // (op 1017) 
s_lshl_b32 s51, 1, s18 // (op 1017) 
s_add_i32 s72, s51, s50 // (op 1017) 


s_and_b32 s50, s0, s72 // (op 1017) 


s_add_i32 s0, s19, s50 // (op 1017) 


s_ashr_i32 s19, s0, s18 // (op 1017) 


s_lshl_b32 s0, s48, 24 // (op 1017) 
s_ashr_i32 s18, s0, 31 // (op 1017) 

s_xor_b32 s0, s19, s18 // (op 1017) 

s_sub_i32 s19, s0, s18 // (op 1017) 


s_cmp_eq_i32 s1, 0 // (op 1017) 
s_mov_b32 s0, scc // (op 1017) 

s_cmp_eq_i32 s0, 0 // (op 1017) 
s_cselect_b32 s1, s19, s49 // (op 1017) 



v_lshl_add_u32 v18, s1, 5, v16 // (op 1017) 


v_mul_lo_u32 v27, v18, s35 // (op 1017) most significant: low of low * high
v_mul_hi_u32 v16, v18, s34 // (op 1017) most significant: high of low * low
v_mul_lo_u32 v26, v18, s34 // (op 1017) least significant: low of low * low
v_add_u32 v27, v27, v16 // (op 1017) most significant: sum


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1017) 
s_lshr_b32 s18, s0, 1 // (op 1017) 

v_lshl_add_u32 v16, s18, 8, v17 // (op 1017) 


v_mov_b32 v19, 0 // (op 1017) convert
v_mov_b32 v18, v16 // (op 1017) convert

v_add_co_u32 v16, s[0:1], v18, v26 // (op 1017) least significant half
v_addc_co_u32 v17, s[0:1], v19, v27, s[0:1] // (op 1017) most significant half



s_mul_i32 s1, s34, s5 // (op 1019) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 1019) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 1019) least significant
s_mul_hi_u32 s19, s34, s4 // (op 1019) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1019) most significant: sum
s_add_u32 s1, s1, s19 // (op 1019) most significant: sum


s_mov_b32 s52, s36 // (op 1019) reg expression
s_mov_b32 s53, s37 // (op 1019) reg expression
s_mov_b32 s54, s0 // (op 1019) reg expression
s_mov_b32 s55, 131072 // (op 1019) call()

s_mul_hi_i32 s0, s16, s8 // (op 1081) 
v_and_b32 v18, -64, v1 // (op 1081) 
s_add_i32 s1, s0, s16 // (op 1081) 

v_sub_u32 v19, v1, v18 // (op 1081) 

s_ashr_i32 s0, s1, 31 // (op 1081) 
v_lshrrev_b32 v18, 6, v1 // (op 1081) 
s_cmp_eq_i32 s8, 0 // (op 1081) 
s_mov_b32 s18, scc // (op 1081) 
s_cmp_eq_i32 s18, 0 // (op 1081) 
s_cselect_b32 s19, 0, -1 // (op 1081) 

s_and_b32 s18, s9, 31 // (op 1081) 
s_lshl_b32 s49, 1, s18 // (op 1081) 
s_add_i32 s50, s49, s19 // (op 1081) 


s_and_b32 s19, s0, s50 // (op 1081) 


s_add_i32 s0, s1, s19 // (op 1081) 


s_ashr_i32 s1, s0, s18 // (op 1081) 


s_lshl_b32 s0, s9, 24 // (op 1081) 
s_ashr_i32 s18, s0, 31 // (op 1081) 

s_xor_b32 s0, s1, s18 // (op 1081) 

s_sub_i32 s1, s0, s18 // (op 1081) 


s_mul_hi_i32 s0, s1, s10 // (op 1081) 
s_add_i32 s18, s0, s1 // (op 1081) 

s_ashr_i32 s0, s18, 31 // (op 1081) 
s_cmp_eq_i32 s10, 0 // (op 1081) 
s_mov_b32 s19, scc // (op 1081) 
s_cmp_eq_i32 s19, 0 // (op 1081) 
s_cselect_b32 s49, 0, -1 // (op 1081) 

s_and_b32 s19, s11, 31 // (op 1081) 
s_lshl_b32 s50, 1, s19 // (op 1081) 
s_add_i32 s51, s50, s49 // (op 1081) 


s_and_b32 s49, s0, s51 // (op 1081) 


s_add_i32 s0, s18, s49 // (op 1081) 


s_ashr_i32 s18, s0, s19 // (op 1081) 


s_lshl_b32 s0, s11, 24 // (op 1081) 
s_ashr_i32 s19, s0, 31 // (op 1081) 

s_xor_b32 s0, s18, s19 // (op 1081) 

s_sub_i32 s18, s0, s19 // (op 1081) 


s_mul_i32 s0, s1, s13 // (op 1081) 

s_sub_u32 s1, s16, s0 // (op 1081) 

s_mul_i32 s0, s18, s13 // (op 1081) 
s_add_u32 s0, s0, s1 // (op 1081) 


s_mul_hi_i32 s1, s16, s7 // (op 1081) 
s_add_i32 s18, s1, s16 // (op 1081) 

s_ashr_i32 s1, s18, 31 // (op 1081) 
s_cmp_eq_i32 s7, 0 // (op 1081) 
s_mov_b32 s19, scc // (op 1081) 
s_cmp_eq_i32 s19, 0 // (op 1081) 
s_cselect_b32 s49, 0, -1 // (op 1081) 

s_and_b32 s19, s14, 31 // (op 1081) 
s_lshl_b32 s50, 1, s19 // (op 1081) 
s_add_i32 s51, s50, s49 // (op 1081) 


s_and_b32 s49, s1, s51 // (op 1081) 


s_add_i32 s1, s18, s49 // (op 1081) 


s_ashr_i32 s18, s1, s19 // (op 1081) 


s_lshl_b32 s1, s14, 24 // (op 1081) 
s_ashr_i32 s19, s1, 31 // (op 1081) 

s_xor_b32 s1, s18, s19 // (op 1081) 

s_sub_i32 s18, s1, s19 // (op 1081) 


s_add_u32 s50, s4, 32 // (op 1081) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1081) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1081) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1081) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1081) 

s_mul_i32 s1, s13, s50 // (op 1081) 

s_mul_i32 s19, s6, s1 // (op 1081) 

s_mul_i32 s1, s18, s19 // (op 1081) 

s_sub_u32 s19, s16, s1 // (op 1081) 

s_mov_b32 s1, s19 // (op 1081) 
s_mul_hi_i32 s49, s1, s12 // (op 1081) 
s_add_i32 s50, s49, s1 // (op 1081) 


s_ashr_i32 s1, s50, 31 // (op 1081) 
s_cmp_eq_i32 s12, 0 // (op 1081) 
s_mov_b32 s49, scc // (op 1081) 
s_cmp_eq_i32 s49, 0 // (op 1081) 
s_cselect_b32 s51, 0, -1 // (op 1081) 

s_and_b32 s49, s48, 31 // (op 1081) 
s_lshl_b32 s72, 1, s49 // (op 1081) 
s_add_i32 s73, s72, s51 // (op 1081) 


s_and_b32 s51, s1, s73 // (op 1081) 


s_add_i32 s1, s50, s51 // (op 1081) 


s_ashr_i32 s50, s1, s49 // (op 1081) 


s_lshl_b32 s1, s48, 24 // (op 1081) 
s_ashr_i32 s49, s1, 31 // (op 1081) 

s_xor_b32 s1, s50, s49 // (op 1081) 

s_sub_i32 s50, s1, s49 // (op 1081) 


s_mul_i32 s1, s15, s13 // (op 1081) 
s_add_u32 s72, s20, 64 // (op 1081) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 1081) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1081) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1081) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 1081) 

s_sub_i32 s49, s72, s1 // (op 1081) 

s_mul_i32 s51, s50, s49 // (op 1081) 


s_sub_u32 s49, s19, s51 // (op 1081) 


s_add_u32 s19, s49, s1 // (op 1081) 


s_cmp_eq_i32 s18, 0 // (op 1081) 
s_mov_b32 s1, scc // (op 1081) 

s_cmp_eq_i32 s1, 0 // (op 1081) 
s_cselect_b32 s18, s19, s0 // (op 1081) 



v_lshl_add_u32 v21, s18, 6, v19 // (op 1081) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1081) 
s_lshr_b32 s18, s0, 1 // (op 1081) 

v_lshl_add_u32 v19, s18, 3, v18 // (op 1081) 


v_mul_lo_u32 v18, v19, s38 // (op 1081) 

v_add_u32 v19, v21, v18 // (op 1081) 


s_mul_i32 s1, s38, s43 // (op 1083) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1083) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1083) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1083) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1083) most significant: sum
s_add_u32 s1, s1, s19 // (op 1083) most significant: sum


s_mov_b32 s64, s40 // (op 1083) reg expression
s_mov_b32 s65, s41 // (op 1083) reg expression
s_mov_b32 s66, s0 // (op 1083) reg expression
s_mov_b32 s67, 131072 // (op 1083) call()

s_mul_hi_i32 s0, s16, s8 // (op 1145) 
v_lshrrev_b32 v18, 6, v1 // (op 1145) 
s_add_i32 s1, s0, s16 // (op 1145) 

v_and_b32 v21, 1, v18 // (op 1145) 

s_ashr_i32 s0, s1, 31 // (op 1145) 
v_and_b32 v18, 15, v1 // (op 1145) 
s_cmp_eq_i32 s8, 0 // (op 1145) 
s_mov_b32 s18, scc // (op 1145) 
v_lshl_add_u32 v22, v21, 4, v18 // (op 1145) 


s_cmp_eq_i32 s18, 0 // (op 1145) 
s_cselect_b32 s19, 0, -1 // (op 1145) 

v_and_b32 v18, 63, v1 // (op 1145) 
s_and_b32 s18, s9, 31 // (op 1145) 
v_lshrrev_b32 v21, 4, v18 // (op 1145) 

s_lshl_b32 s49, 1, s18 // (op 1145) 
s_add_i32 s50, s49, s19 // (op 1145) 


s_and_b32 s19, s0, s50 // (op 1145) 


s_add_i32 s0, s1, s19 // (op 1145) 


s_ashr_i32 s1, s0, s18 // (op 1145) 


s_lshl_b32 s0, s9, 24 // (op 1145) 
s_ashr_i32 s18, s0, 31 // (op 1145) 

s_xor_b32 s0, s1, s18 // (op 1145) 

s_sub_i32 s1, s0, s18 // (op 1145) 


s_mul_hi_i32 s0, s1, s10 // (op 1145) 
s_add_i32 s18, s0, s1 // (op 1145) 

s_ashr_i32 s0, s18, 31 // (op 1145) 
s_cmp_eq_i32 s10, 0 // (op 1145) 
s_mov_b32 s19, scc // (op 1145) 
s_cmp_eq_i32 s19, 0 // (op 1145) 
s_cselect_b32 s49, 0, -1 // (op 1145) 

s_and_b32 s19, s11, 31 // (op 1145) 
s_lshl_b32 s50, 1, s19 // (op 1145) 
s_add_i32 s51, s50, s49 // (op 1145) 


s_and_b32 s49, s0, s51 // (op 1145) 


s_add_i32 s0, s18, s49 // (op 1145) 


s_ashr_i32 s18, s0, s19 // (op 1145) 


s_lshl_b32 s0, s11, 24 // (op 1145) 
s_ashr_i32 s19, s0, 31 // (op 1145) 

s_xor_b32 s0, s18, s19 // (op 1145) 

s_sub_i32 s18, s0, s19 // (op 1145) 


s_add_u32 s50, s4, 32 // (op 1145) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1145) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1145) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1145) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1145) 

s_mul_i32 s0, s18, s50 // (op 1145) 

s_sub_i32 s49, s1, s0 // (op 1145) 


s_mul_hi_i32 s0, s16, s7 // (op 1145) 
s_add_i32 s1, s0, s16 // (op 1145) 

s_ashr_i32 s0, s1, 31 // (op 1145) 
s_cmp_eq_i32 s7, 0 // (op 1145) 
s_mov_b32 s18, scc // (op 1145) 
s_cmp_eq_i32 s18, 0 // (op 1145) 
s_cselect_b32 s19, 0, -1 // (op 1145) 

s_and_b32 s18, s14, 31 // (op 1145) 
s_lshl_b32 s72, 1, s18 // (op 1145) 
s_add_i32 s73, s72, s19 // (op 1145) 


s_and_b32 s19, s0, s73 // (op 1145) 


s_add_i32 s0, s1, s19 // (op 1145) 


s_ashr_i32 s1, s0, s18 // (op 1145) 


s_lshl_b32 s0, s14, 24 // (op 1145) 
s_ashr_i32 s18, s0, 31 // (op 1145) 

s_xor_b32 s0, s1, s18 // (op 1145) 

s_sub_i32 s1, s0, s18 // (op 1145) 


s_mul_i32 s0, s13, s50 // (op 1145) 

s_mul_i32 s18, s6, s0 // (op 1145) 

s_mul_i32 s0, s1, s18 // (op 1145) 

s_sub_u32 s18, s16, s0 // (op 1145) 

s_mul_hi_i32 s0, s18, s12 // (op 1145) 
s_add_i32 s19, s0, s18 // (op 1145) 


s_ashr_i32 s0, s19, 31 // (op 1145) 
s_cmp_eq_i32 s12, 0 // (op 1145) 
s_mov_b32 s18, scc // (op 1145) 
s_cmp_eq_i32 s18, 0 // (op 1145) 
s_cselect_b32 s50, 0, -1 // (op 1145) 

s_and_b32 s18, s48, 31 // (op 1145) 
s_lshl_b32 s51, 1, s18 // (op 1145) 
s_add_i32 s72, s51, s50 // (op 1145) 


s_and_b32 s50, s0, s72 // (op 1145) 


s_add_i32 s0, s19, s50 // (op 1145) 


s_ashr_i32 s19, s0, s18 // (op 1145) 


s_lshl_b32 s0, s48, 24 // (op 1145) 
s_ashr_i32 s18, s0, 31 // (op 1145) 

s_xor_b32 s0, s19, s18 // (op 1145) 

s_sub_i32 s19, s0, s18 // (op 1145) 


s_cmp_eq_i32 s1, 0 // (op 1145) 
s_mov_b32 s0, scc // (op 1145) 

s_cmp_eq_i32 s0, 0 // (op 1145) 
s_cselect_b32 s1, s19, s49 // (op 1145) 



v_lshl_add_u32 v18, s1, 5, v22 // (op 1145) 


v_mul_lo_u32 v22, v18, s44 // (op 1145) 

s_ashr_i64 s[0:1], s[32:33], 7 // (op 1145) 
s_lshr_b32 s18, s0, 1 // (op 1145) 

v_lshl_add_u32 v18, s18, 3, v21 // (op 1145) 


v_add_u32 v21, v18, v22 // (op 1145) 


s_mul_i32 s1, s44, s5 // (op 1147) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 1147) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 1147) least significant
s_mul_hi_u32 s19, s44, s4 // (op 1147) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1147) most significant: sum
s_add_u32 s1, s1, s19 // (op 1147) most significant: sum


s_mov_b32 s68, s46 // (op 1147) reg expression
s_mov_b32 s69, s47 // (op 1147) reg expression
s_mov_b32 s70, s0 // (op 1147) reg expression
s_mov_b32 s71, 131072 // (op 1147) call()

v_and_b32 v18, -4, v1 // (op 1238) 
v_sub_u32 v22, v1, v18 // (op 1238) 

v_lshlrev_b32 v18, 4, v22 // (op 1238) 

v_lshrrev_b32 v22, 2, v1 // (op 1238) 
v_lshl_add_u32 v23, v22, 6, v18 // (op 1238) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 1238) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 1295) 
v_sub_u32 v22, v1, v18 // (op 1295) 

v_lshlrev_b32 v18, 4, v22 // (op 1295) 

v_lshrrev_b32 v22, 3, v1 // (op 1295) 
v_lshl_add_u32 v23, v22, 7, v18 // (op 1295) 


s_nop 0
v_readfirstlane_b32 s1, v23 // (op 1295) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -64, v1 // (op 1347) 
v_sub_u32 v22, v1, v18 // (op 1347) 

v_and_b32 v18, 4294967232, v1 // (op 1347) 
v_add_u32 v23, v18, v22 // (op 1347) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 320) 
s_and_b32 s49, s18, 4294967294 // (op 320) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 319) 
s_mov_b32 s50, s18 // (op 319) convert

s_cmp_lt_u32 s49, s50 // (op 319) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_319 // (op 319) Condition: Top (jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_319 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_319:
s_barrier  // (op 1557) 
s_add_u32 s49, s49, 1 // (op 321) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 901) 
v_mov_b64 v[12:13], s[18:19] // (op 901) convert


v_mov_b64 v[24:25], v[14:15] // (op 901) 
v_mov_b64 v[26:27], v[24:25] // (op 901) reg expression
s_add_u32 s18, 4096, s0 // (op 901) 
s_mov_b32 m0, s18 // (op 901) reg expression
buffer_load_dwordx4 v26, s[24:27], 0 offen offset: 0 lds // (op 901) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v12 // (op 901) least significant half
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 901) most significant half

s_add_u32 m0, m0, 4096 // (op 901) 
buffer_load_dwordx4 v26, s[24:27], 0 offen offset: 0 lds // (op 901) Load value direct to lds




s_lshl_b64 s[18:19], s[28:29], 7 // (op 939) 
v_mov_b32 v18, s18 // (op 939) 
v_mov_b32 v20, s19 // (op 939) 
v_add_co_u32 v14, s[50:51], v14, v18 // (op 939) least significant half
v_addc_co_u32 v15, s[50:51], v15, v20, s[50:51] // (op 939) most significant half




s_barrier  // (op 1573) 


v_mov_b64 v[12:13], v[16:17] // (op 907) 

s_mov_b32 m0, s1 // (op 907) reg expression
buffer_load_dwordx4 v12, s[52:55], 0 offen offset: 0 lds // (op 907) Load value direct to lds

v_add_co_u32 v16, vcc, 128, v16 // (op 1009) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1009) most significant half


v_mov_b32 v18, v19 // (op 184) 
v_mov_b32 v20, v18 // (op 184) reg expression

buffer_load_ubyte v22, v20, s[64:67], 0 offen offset: 0 // (op 184) Load value


s_lshl_b32 s18, s38, 2 // (op 1073) 
v_add_u32 v19, v19, s18 // (op 1073) 

s_barrier  // (op 1589) 
s_waitcnt vmcnt(2)
s_barrier  // (op 1563) 






ds_read_b64_tr_b8 v[24:25], v0 offset:4096 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:4608 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:8192 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:8704 // (op 173) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v22 offset:24576 // (op 182) Store local data 

s_barrier  // (op 1579) 



ds_read_b128 v[32:35], v3 // (op 175) Load local data 
ds_read_b128 v[36:39], v3 offset:64 // (op 175) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1596) 


ds_read_u8 v18, v7 offset:24576 // (op 177) Load local data 


v_mov_b32 v20, v21 // (op 179) 

buffer_load_ubyte v22, v20, s[68:71], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v21, 4, v21 // (op 1137) 



ds_read_b64_tr_b8 v[40:41], v0 offset:4128 // (op 325) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[32:39], a[0:3], v18, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 180) 
ds_read_b64_tr_b8 v[42:43], v0 offset:4640 // (op 325) Transpose load local data 


ds_read_b64_tr_b8 v[44:45], v0 offset:8224 // (op 325) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:8736 // (op 325) Transpose load local data 


ds_read_u8 v18, v7 offset:24608 // (op 329) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:47], v[32:39], a[4:7], v18, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 331) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 319) 
s_mov_b32 s50, s18 // (op 319) convert

s_cmp_lt_u32 s49, s50 // (op 319) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_319 // (op 319) Condition: Bottom (jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_319 if true)
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_319:






















s_barrier  // (op 1658) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
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


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen // (op 54) Store value


v_accvgpr_read v10, a4 // (op 517) 
v_accvgpr_read v11, a5 // (op 517) 
v_accvgpr_read v12, a6 // (op 517) 
v_accvgpr_read v13, a7 // (op 517) 
v_mul_f32 v14, v8, v10 // (op 517) 
v_mul_f32 v15, v8, v11 // (op 517) 
v_mul_f32 v16, v8, v12 // (op 517) 
v_mul_f32 v17, v8, v13 // (op 517) 

v_cvt_f16_f32 v10, v14 // (op 519) 
v_cvt_f16_f32 v11, v15 // (op 519) 
v_cvt_f16_f32 v12, v16 // (op 519) 
v_cvt_f16_f32 v13, v17 // (op 519) 


v_pack_b32_f16 v14, v10, v11 // (op 520) 
v_pack_b32_f16 v15, v12, v13 // (op 520) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:64 // (op 520) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[10:11], v6, s[56:59], 0 offen offset:64 // (op 550) Load value
v_accvgpr_read v12, a4 // (op 546) 
v_accvgpr_read v13, a5 // (op 546) 
v_accvgpr_read v14, a6 // (op 546) 
v_accvgpr_read v15, a7 // (op 546) 
v_mul_f32 v16, v8, v12 // (op 546) 
v_mul_f32 v17, v8, v13 // (op 546) 
v_mul_f32 v18, v8, v14 // (op 546) 
v_mul_f32 v19, v8, v15 // (op 546) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 551) 
v_lshrrev_b32 v13, 16, v10 // (op 551) 
v_cvt_f32_f16 v13, v13 // (op 551) 
v_mul_f32 v20, v5, v12 // (op 551) 
v_mul_f32 v21, v5, v13 // (op 551) 
v_cvt_f32_f16 v12, v11 // (op 551) 
v_lshrrev_b32 v13, 16, v11 // (op 551) 
v_cvt_f32_f16 v13, v13 // (op 551) 
v_mul_f32 v22, v5, v12 // (op 551) 
v_mul_f32 v23, v5, v13 // (op 551) 


v_add_f32 v10, v20, v16 // (op 547) 
v_add_f32 v11, v21, v17 // (op 547) 
v_add_f32 v12, v22, v18 // (op 547) 
v_add_f32 v13, v23, v19 // (op 547) 


v_cvt_f16_f32 v14, v10 // (op 548) 
v_cvt_f16_f32 v15, v11 // (op 548) 
v_cvt_f16_f32 v16, v12 // (op 548) 
v_cvt_f16_f32 v17, v13 // (op 548) 


v_pack_b32_f16 v10, v14, v15 // (op 549) 
v_pack_b32_f16 v11, v16, v17 // (op 549) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:64 // (op 549) Store value





buffer_load_dwordx2 v[10:11], v6, s[56:59], 0 offen // (op 138) Load value
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


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.LRR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__end:
.size RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_, .LRR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__end-RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25088 // lds bytes
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: B8
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
            { type: int64, semantic: SizeFree1 },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideD0 },
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
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [64, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 25088
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 62
    .agpr_count: 8
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_11
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 16
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 20
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 24
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 28
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_0_size_0_15
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 72
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
        .size: 8
        .offset: 152
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 160
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
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

