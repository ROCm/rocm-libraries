// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_,@function
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s8 // (op 1160) 
s_mul_hi_i32 s1, s16, s8 // (op 1348) 
v_lshrrev_b32 v0, 6, v1 // (op 1274) 
v_and_b32 v3, 1, v0 // (op 1274) 

v_and_b32 v0, 15, v1 // (op 1274) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 1274) 


v_and_b32 v0, 48, v1 // (op 1274) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 1274) 


v_lshrrev_b32 v0, 6, v1 // (op 1418) 
v_and_b32 v4, 1, v0 // (op 1418) 

v_and_b32 v0, 15, v1 // (op 1418) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 1418) 


v_and_b32 v0, 48, v1 // (op 1418) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 1418) 


v_lshrrev_b32 v0, 7, v1 // (op 1160) 
s_add_i32 s18, s0, s16 // (op 1160) 

v_and_b32 v5, 63, v1 // (op 1160) 
s_ashr_i32 s0, s18, 31 // (op 1160) 
v_lshrrev_b32 v6, 4, v5 // (op 1160) 
s_cmp_eq_i32 s8, 0 // (op 1160) 
s_mov_b32 s19, scc // (op 1160) 
v_lshlrev_b32 v7, 2, v6 // (op 1160) 

s_cmp_eq_i32 s19, 0 // (op 1160) 
s_cselect_b32 s51, 0, -1 // (op 1160) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1160) 


s_and_b32 s19, s9, 31 // (op 1160) 
v_lshrrev_b32 v0, 2, v5 // (op 1160) 

v_and_b32 v5, 3, v0 // (op 1160) 

v_and_b32 v0, 63, v1 // (op 1222) 
v_and_b32 v7, 48, v1 // (op 1222) 
v_sub_u32 v8, v0, v7 // (op 1222) 

v_lshrrev_b32 v7, 4, v0 // (op 1222) 

v_lshl_add_u32 v0, v7, 5, v8 // (op 1222) 


v_lshrrev_b32 v7, 7, v1 // (op 1222) 
v_lshlrev_b32 v8, 4, v7 // (op 1222) 

v_lshl_add_u32 v7, v0, 6, v8 // (op 1222) 


v_mul_lo_u32 v0, v7, 6 // (op 1222) 
v_lshrrev_b32 v8, 3, v0 // (op 1222) 

v_lshrrev_b32 v0, 4, v7 // (op 1222) 

v_lshlrev_b32 v7, 2, v0 // (op 1222) 

v_add_u32 v0, v8, v7 // (op 1222) 


v_lshrrev_b32 v7, 7, v1 // (op 1332) 
v_and_b32 v8, 15, v1 // (op 1332) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 1332) 


v_and_b32 v7, 63, v1 // (op 1332) 
v_lshrrev_b32 v8, 4, v7 // (op 1332) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 1332) 


v_lshrrev_b32 v8, 7, v1 // (op 1348) 
v_and_b32 v9, 63, v1 // (op 1348) 
v_and_b32 v10, 63, v1 // (op 1379) 
v_and_b32 v11, 48, v1 // (op 1379) 
v_sub_u32 v12, v10, v11 // (op 1379) 

v_lshrrev_b32 v11, 4, v10 // (op 1379) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 1379) 


v_lshrrev_b32 v11, 7, v1 // (op 1379) 
v_lshlrev_b32 v12, 4, v11 // (op 1379) 

v_lshl_add_u32 v11, v10, 6, v12 // (op 1379) 


v_mul_lo_u32 v10, v11, 6 // (op 1379) 
v_lshrrev_b32 v12, 3, v10 // (op 1379) 

v_lshrrev_b32 v10, 4, v11 // (op 1379) 

v_lshlrev_b32 v11, 2, v10 // (op 1379) 

v_add_u32 v10, v12, v11 // (op 1379) 


v_lshrrev_b32 v11, 7, v1 // (op 1447) 
v_and_b32 v12, 15, v1 // (op 1447) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 1447) 


v_and_b32 v11, 63, v1 // (op 1447) 
v_lshrrev_b32 v12, 4, v11 // (op 1447) 

v_lshl_add_u32 v11, v12, 6, v13 // (op 1447) 


s_lshl_b32 s52, 1, s19 // (op 1160) 
v_and_b32 v12, 3, v1 // (op 1160) 
s_add_i32 s53, s52, s51 // (op 1160) 


s_and_b32 s51, s0, s53 // (op 1160) 


s_add_i32 s0, s1, s16 // (op 1348) 

s_ashr_i32 s1, s0, 31 // (op 1348) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1160) 


s_add_i32 s52, s18, s51 // (op 1160) 


v_lshrrev_b32 v5, 6, v1 // (op 1160) 
s_ashr_i32 s18, s52, s19 // (op 1160) 


v_and_b32 v12, 1, v5 // (op 1160) 

s_lshl_b32 s19, s9, 24 // (op 1160) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1160) 


s_ashr_i32 s51, s19, 31 // (op 1160) 

s_xor_b32 s19, s18, s51 // (op 1160) 

s_sub_i32 s18, s19, s51 // (op 1160) 


s_mul_hi_i32 s19, s18, s10 // (op 1160) 
s_add_i32 s51, s19, s18 // (op 1160) 

s_ashr_i32 s19, s51, 31 // (op 1160) 
s_cmp_eq_i32 s10, 0 // (op 1160) 
s_mov_b32 s52, scc // (op 1160) 
s_cmp_eq_i32 s52, 0 // (op 1160) 
s_cselect_b32 s53, 0, -1 // (op 1160) 

s_and_b32 s52, s11, 31 // (op 1160) 
v_lshrrev_b32 v12, 4, v9 // (op 1348) 
s_cmp_eq_i32 s8, 0 // (op 1348) 
s_mov_b32 s54, scc // (op 1348) 
v_lshlrev_b32 v13, 2, v12 // (op 1348) 

s_cmp_eq_i32 s54, 0 // (op 1348) 
s_cselect_b32 s55, 0, -1 // (op 1348) 

s_lshl_b32 s54, 1, s52 // (op 1160) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 1348) 


v_lshrrev_b32 v8, 2, v9 // (op 1348) 

s_add_i32 s56, s54, s53 // (op 1160) 


s_and_b32 s53, s19, s56 // (op 1160) 


s_add_i32 s19, s51, s53 // (op 1160) 


s_ashr_i32 s51, s19, s52 // (op 1160) 


s_lshl_b32 s19, s11, 24 // (op 1160) 
s_ashr_i32 s52, s19, 31 // (op 1160) 

s_xor_b32 s19, s51, s52 // (op 1160) 

s_sub_i32 s51, s19, s52 // (op 1160) 


s_mul_i32 s19, s18, s13 // (op 1160) 
s_sub_u32 s52, s16, s19 // (op 1160) 

s_mul_i32 s19, s51, s13 // (op 1160) 
s_add_u32 s19, s19, s52 // (op 1160) 

s_mul_hi_i32 s52, s16, s7 // (op 1160) 
s_add_i32 s53, s52, s16 // (op 1160) 

s_ashr_i32 s52, s53, 31 // (op 1160) 
s_cmp_eq_i32 s7, 0 // (op 1160) 
s_mov_b32 s54, scc // (op 1160) 
s_cmp_eq_i32 s54, 0 // (op 1160) 
s_cselect_b32 s56, 0, -1 // (op 1160) 

s_and_b32 s54, s14, 31 // (op 1160) 
s_lshl_b32 s57, 1, s54 // (op 1160) 
s_add_i32 s58, s57, s56 // (op 1160) 


s_and_b32 s56, s52, s58 // (op 1160) 


s_add_i32 s52, s53, s56 // (op 1160) 


s_ashr_i32 s53, s52, s54 // (op 1160) 


s_lshl_b32 s52, s14, 24 // (op 1160) 
s_ashr_i32 s54, s52, 31 // (op 1160) 

s_xor_b32 s52, s53, s54 // (op 1160) 

s_sub_i32 s53, s52, s54 // (op 1160) 


s_and_b32 s52, s9, 31 // (op 1348) 
s_lshl_b32 s54, 1, s52 // (op 1348) 
v_and_b32 v9, 3, v8 // (op 1348) 

s_add_i32 s56, s54, s55 // (op 1348) 


v_and_b32 v8, 3, v1 // (op 1348) 
s_and_b32 s54, s1, s56 // (op 1348) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1348) 


s_add_i32 s1, s0, s54 // (op 1348) 


v_lshrrev_b32 v8, 6, v1 // (op 1348) 
s_ashr_i32 s0, s1, s52 // (op 1348) 


v_and_b32 v9, 1, v8 // (op 1348) 

s_lshl_b32 s1, s9, 24 // (op 1348) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1348) 


s_ashr_i32 s52, s1, 31 // (op 1348) 

s_xor_b32 s1, s0, s52 // (op 1348) 

s_sub_i32 s0, s1, s52 // (op 1348) 


s_mul_hi_i32 s1, s0, s10 // (op 1348) 
s_add_i32 s52, s1, s0 // (op 1348) 

s_ashr_i32 s1, s52, 31 // (op 1348) 
s_cmp_eq_i32 s10, 0 // (op 1348) 
s_mov_b32 s54, scc // (op 1348) 
s_cmp_eq_i32 s54, 0 // (op 1348) 
s_cselect_b32 s55, 0, -1 // (op 1348) 

s_and_b32 s54, s11, 31 // (op 1348) 
s_lshl_b32 s56, 1, s54 // (op 1348) 
s_add_i32 s57, s56, s55 // (op 1348) 


s_and_b32 s55, s1, s57 // (op 1348) 


s_add_i32 s1, s52, s55 // (op 1348) 


s_ashr_i32 s52, s1, s54 // (op 1348) 


s_lshl_b32 s1, s11, 24 // (op 1348) 
s_ashr_i32 s54, s1, 31 // (op 1348) 

s_xor_b32 s1, s52, s54 // (op 1348) 

s_sub_i32 s52, s1, s54 // (op 1348) 


s_mul_i32 s1, s0, s13 // (op 1348) 
s_sub_u32 s54, s16, s1 // (op 1348) 

s_mul_i32 s1, s52, s13 // (op 1348) 
s_add_u32 s1, s1, s54 // (op 1348) 

s_mul_hi_i32 s54, s16, s7 // (op 1348) 
s_add_i32 s55, s54, s16 // (op 1348) 

s_ashr_i32 s54, s55, 31 // (op 1348) 
s_cmp_eq_i32 s7, 0 // (op 1348) 
s_mov_b32 s56, scc // (op 1348) 
s_cmp_eq_i32 s56, 0 // (op 1348) 
s_cselect_b32 s57, 0, -1 // (op 1348) 

s_and_b32 s56, s14, 31 // (op 1348) 
s_lshl_b32 s58, 1, s56 // (op 1348) 
s_add_i32 s59, s58, s57 // (op 1348) 


s_and_b32 s57, s54, s59 // (op 1348) 


s_add_i32 s54, s55, s57 // (op 1348) 


s_ashr_i32 s55, s54, s56 // (op 1348) 


s_lshl_b32 s54, s14, 24 // (op 1348) 
s_ashr_i32 s56, s54, 31 // (op 1348) 

s_xor_b32 s54, s55, s56 // (op 1348) 

s_sub_i32 s55, s54, s56 // (op 1348) 


s_add_u32 s56, s4, 32 // (op 1160) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 1160) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1160) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1160) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 1160) 

s_mul_i32 s54, s13, s56 // (op 1160) 
s_mul_i32 s58, s6, s54 // (op 1160) 

s_mul_i32 s54, s53, s58 // (op 1160) 

s_sub_u32 s58, s16, s54 // (op 1160) 

s_mov_b32 s54, s58 // (op 1160) 
s_mul_hi_i32 s59, s54, s12 // (op 1160) 
s_add_i32 s60, s59, s54 // (op 1160) 


s_ashr_i32 s54, s60, 31 // (op 1160) 
s_cmp_eq_i32 s12, 0 // (op 1160) 
s_mov_b32 s59, scc // (op 1160) 
s_cmp_eq_i32 s59, 0 // (op 1160) 
s_cselect_b32 s61, 0, -1 // (op 1160) 

s_add_u32 s62, s4, 32 // (op 1348) Start of Int64 add, locking SCC
s_addc_u32 s63, s5, 0 // (op 1348) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1348) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1348) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 5 // (op 1348) 

s_mul_i32 s59, s13, s62 // (op 1348) 
s_mul_i32 s64, s6, s59 // (op 1348) 

s_mul_i32 s59, s55, s64 // (op 1348) 

s_sub_u32 s64, s16, s59 // (op 1348) 

s_mov_b32 s59, s64 // (op 1348) 
s_mul_hi_i32 s65, s59, s12 // (op 1348) 
s_add_i32 s66, s65, s59 // (op 1348) 


s_ashr_i32 s59, s66, 31 // (op 1348) 
s_cmp_eq_i32 s12, 0 // (op 1348) 
s_mov_b32 s65, scc // (op 1348) 
s_cmp_eq_i32 s65, 0 // (op 1348) 
s_cselect_b32 s67, 0, -1 // (op 1348) 


s_waitcnt lgkmcnt(0)
s_and_b32 s65, s48, 31 // (op 1160) 
s_lshl_b32 s68, 1, s65 // (op 1160) 
s_add_i32 s69, s68, s61 // (op 1160) 


s_and_b32 s61, s54, s69 // (op 1160) 


s_add_i32 s54, s60, s61 // (op 1160) 


s_ashr_i32 s60, s54, s65 // (op 1160) 


s_lshl_b32 s54, s48, 24 // (op 1160) 
s_ashr_i32 s61, s54, 31 // (op 1160) 

s_xor_b32 s54, s60, s61 // (op 1160) 

s_sub_i32 s60, s54, s61 // (op 1160) 


s_mul_i32 s54, s15, s13 // (op 1160) 
s_and_b32 s61, s48, 31 // (op 1348) 
s_lshl_b32 s65, 1, s61 // (op 1348) 
s_add_i32 s68, s65, s67 // (op 1348) 


s_and_b32 s65, s59, s68 // (op 1348) 


s_add_i32 s59, s66, s65 // (op 1348) 


s_ashr_i32 s65, s59, s61 // (op 1348) 


s_lshl_b32 s59, s48, 24 // (op 1348) 
s_ashr_i32 s61, s59, 31 // (op 1348) 

s_xor_b32 s59, s65, s61 // (op 1348) 

s_sub_i32 s65, s59, s61 // (op 1348) 


s_mul_i32 s59, s15, s13 // (op 1348) 
s_add_u32 s66, s20, 64 // (op 1160) Start of Int64 add, locking SCC
s_addc_u32 s67, s21, 0 // (op 1160) End of Int64 add, unlocking SCC
s_sub_u32 s68, s66, 1 // (op 1160) least significant half; sets scc
s_subb_u32 s69, s67, 0 // (op 1160) most significant half; uses scc

s_ashr_i64 s[66:67], s[68:69], 6 // (op 1160) 

s_sub_i32 s61, s66, s54 // (op 1160) 

s_mul_i32 s66, s60, s61 // (op 1160) 

s_sub_u32 s61, s58, s66 // (op 1160) 


s_add_u32 s58, s61, s54 // (op 1160) 


s_cmp_eq_i32 s53, 0 // (op 1160) 
s_mov_b32 s54, scc // (op 1160) 

s_cmp_eq_i32 s54, 0 // (op 1160) 
s_cselect_b32 s53, s58, s19 // (op 1160) 


s_mul_i32 s19, s51, s56 // (op 1160) 


v_lshl_add_u32 v9, s53, 6, v6 // (op 1160) 


s_sub_i32 s51, s18, s19 // (op 1160) 


s_cmp_eq_i32 s54, 0 // (op 1160) 
s_cselect_b32 s53, s60, s51 // (op 1160) 



v_lshl_add_u32 v6, s53, 5, v5 // (op 1160) 


v_mul_lo_u32 v5, v6, s2 // (op 1160) 

v_add_lshl_u32 v6, v9, v5, 2 // (op 1160) 


s_add_u32 s18, s20, 64 // (op 1348) Start of Int64 add, locking SCC
s_addc_u32 s19, s21, 0 // (op 1348) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 1348) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 1348) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 6 // (op 1348) 

s_sub_i32 s51, s18, s59 // (op 1348) 

s_mul_i32 s18, s65, s51 // (op 1348) 

s_sub_u32 s19, s64, s18 // (op 1348) 


s_add_u32 s18, s19, s59 // (op 1348) 


s_cmp_eq_i32 s55, 0 // (op 1348) 
s_mov_b32 s19, scc // (op 1348) 

s_cmp_eq_i32 s19, 0 // (op 1348) 
s_cselect_b32 s51, s18, s1 // (op 1348) 


s_mul_i32 s1, s52, s62 // (op 1348) 


v_lshl_add_u32 v5, s51, 6, v12 // (op 1348) 


s_sub_i32 s18, s0, s1 // (op 1348) 


s_cmp_eq_i32 s19, 0 // (op 1348) 
s_cselect_b32 s0, s65, s18 // (op 1348) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 1348) 


v_mul_lo_u32 v8, v9, s22 // (op 1348) 

v_add_lshl_u32 v9, v5, v8, 2 // (op 1348) 


s_mul_i32 s1, s2, s5 // (op 1162) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 1162) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 1162) least significant
s_mul_hi_u32 s19, s2, s4 // (op 1162) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1162) most significant: sum
s_add_u32 s1, s1, s19 // (op 1162) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1162) 

s_mul_i32 s1, s22, s5 // (op 1350) most significant: low of low * high
s_mul_i32 s51, s23, s4 // (op 1350) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 1350) least significant
s_mul_hi_u32 s52, s22, s4 // (op 1350) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1350) most significant: sum
s_add_u32 s1, s1, s52 // (op 1350) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 1350) 

s_mov_b32 s56, s24 // (op 1162) reg expression
s_mov_b32 s57, s25 // (op 1162) reg expression
s_mov_b32 s58, s18 // (op 1162) reg expression
s_mov_b32 s59, 131072 // (op 1162) call()

s_mov_b32 s60, s26 // (op 1350) reg expression

s_mov_b32 s61, s27 // (op 1350) reg expression
s_mov_b32 s62, s52 // (op 1350) reg expression
s_mov_b32 s63, 131072 // (op 1350) call()


v_mov_b32 v5, s49 // (op 2) Move value
v_mov_b32 v8, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 309) call()
v_accvgpr_write a5, 0.00000 // (op 309) call()
v_accvgpr_write a6, 0.00000 // (op 309) call()
v_accvgpr_write a7, 0.00000 // (op 309) call()
s_mul_hi_i32 s0, s16, s8 // (op 927) 
v_and_b32 v12, -2, v1 // (op 927) 
s_add_i32 s1, s0, s16 // (op 927) 

v_sub_u32 v13, v1, v12 // (op 927) 

s_ashr_i32 s0, s1, 31 // (op 927) 
v_lshlrev_b32 v12, 5, v13 // (op 927) 

s_cmp_eq_i32 s8, 0 // (op 927) 
s_mov_b32 s18, scc // (op 927) 
v_lshrrev_b32 v13, 1, v1 // (op 927) 
s_cmp_eq_i32 s18, 0 // (op 927) 
s_cselect_b32 s19, 0, -1 // (op 927) 

s_and_b32 s18, s9, 31 // (op 927) 
s_lshl_b32 s24, 1, s18 // (op 927) 
s_add_i32 s25, s24, s19 // (op 927) 


s_and_b32 s19, s0, s25 // (op 927) 


s_add_i32 s0, s1, s19 // (op 927) 


s_ashr_i32 s1, s0, s18 // (op 927) 


s_lshl_b32 s0, s9, 24 // (op 927) 
s_ashr_i32 s18, s0, 31 // (op 927) 

s_xor_b32 s0, s1, s18 // (op 927) 

s_sub_i32 s1, s0, s18 // (op 927) 


s_mul_hi_i32 s0, s1, s10 // (op 927) 
s_add_i32 s18, s0, s1 // (op 927) 

s_ashr_i32 s0, s18, 31 // (op 927) 
s_cmp_eq_i32 s10, 0 // (op 927) 
s_mov_b32 s19, scc // (op 927) 
s_cmp_eq_i32 s19, 0 // (op 927) 
s_cselect_b32 s24, 0, -1 // (op 927) 

s_and_b32 s19, s11, 31 // (op 927) 
s_lshl_b32 s25, 1, s19 // (op 927) 
s_add_i32 s26, s25, s24 // (op 927) 


s_and_b32 s27, s0, s26 // (op 927) 


s_add_i32 s0, s18, s27 // (op 927) 


s_ashr_i32 s18, s0, s19 // (op 927) 


s_lshl_b32 s0, s11, 24 // (op 927) 
s_ashr_i32 s19, s0, 31 // (op 927) 

s_xor_b32 s0, s18, s19 // (op 927) 

s_sub_i32 s18, s0, s19 // (op 927) 


s_mul_i32 s0, s1, s13 // (op 927) 

s_sub_u32 s1, s16, s0 // (op 927) 

s_mul_i32 s0, s18, s13 // (op 927) 
s_add_u32 s0, s0, s1 // (op 927) 


s_mul_hi_i32 s1, s16, s7 // (op 927) 
s_add_i32 s18, s1, s16 // (op 927) 

s_ashr_i32 s1, s18, 31 // (op 927) 
s_cmp_eq_i32 s7, 0 // (op 927) 
s_mov_b32 s19, scc // (op 927) 
s_cmp_eq_i32 s19, 0 // (op 927) 
s_cselect_b32 s24, 0, -1 // (op 927) 

s_and_b32 s19, s14, 31 // (op 927) 
s_lshl_b32 s25, 1, s19 // (op 927) 
s_add_i32 s26, s25, s24 // (op 927) 


s_and_b32 s27, s1, s26 // (op 927) 


s_add_i32 s1, s18, s27 // (op 927) 


s_ashr_i32 s18, s1, s19 // (op 927) 


s_lshl_b32 s1, s14, 24 // (op 927) 
s_ashr_i32 s19, s1, 31 // (op 927) 

s_xor_b32 s1, s18, s19 // (op 927) 

s_sub_i32 s18, s1, s19 // (op 927) 


s_add_u32 s24, s4, 32 // (op 927) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 927) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 927) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 927) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 5 // (op 927) 

s_mul_i32 s1, s13, s24 // (op 927) 

s_mul_i32 s19, s6, s1 // (op 927) 

s_mul_i32 s1, s18, s19 // (op 927) 

s_sub_u32 s19, s16, s1 // (op 927) 

s_mov_b32 s1, s19 // (op 927) 
s_mul_hi_i32 s24, s1, s12 // (op 927) 
s_add_i32 s25, s24, s1 // (op 927) 


s_ashr_i32 s1, s25, 31 // (op 927) 
s_cmp_eq_i32 s12, 0 // (op 927) 
s_mov_b32 s24, scc // (op 927) 
s_cmp_eq_i32 s24, 0 // (op 927) 
s_cselect_b32 s26, 0, -1 // (op 927) 

s_and_b32 s24, s48, 31 // (op 927) 
s_lshl_b32 s27, 1, s24 // (op 927) 
s_add_i32 s49, s27, s26 // (op 927) 


s_and_b32 s26, s1, s49 // (op 927) 


s_add_i32 s1, s25, s26 // (op 927) 


s_ashr_i32 s25, s1, s24 // (op 927) 


s_lshl_b32 s1, s48, 24 // (op 927) 
s_ashr_i32 s24, s1, 31 // (op 927) 

s_xor_b32 s1, s25, s24 // (op 927) 

s_sub_i32 s25, s1, s24 // (op 927) 


s_mul_i32 s1, s15, s13 // (op 927) 
s_add_u32 s26, s20, 64 // (op 927) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 927) End of Int64 add, unlocking SCC
s_sub_u32 s54, s26, 1 // (op 927) least significant half; sets scc
s_subb_u32 s55, s27, 0 // (op 927) most significant half; uses scc

s_ashr_i64 s[26:27], s[54:55], 6 // (op 927) 

s_sub_i32 s24, s26, s1 // (op 927) 

s_mul_i32 s26, s25, s24 // (op 927) 


s_sub_u32 s27, s19, s26 // (op 927) 


s_add_u32 s19, s27, s1 // (op 927) 


s_cmp_eq_i32 s18, 0 // (op 927) 
s_mov_b32 s1, scc // (op 927) 

s_cmp_eq_i32 s1, 0 // (op 927) 
s_cselect_b32 s18, s19, s0 // (op 927) 



v_lshl_add_u32 v14, s18, 6, v12 // (op 927) 


v_mov_b32 v17, 0 // (op 927) convert
v_mov_b32 v16, v14 // (op 927) convert

v_mul_lo_u32 v15, v13, s29 // (op 927) most significant: low of low * high
v_mul_hi_u32 v12, v13, s28 // (op 927) most significant: high of low * low
v_mul_lo_u32 v14, v13, s28 // (op 927) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 927) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v14 // (op 927) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 927) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 927) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 927) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 927) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 927) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 927) 

v_mov_b32 v14, v12 // (op 927) convert

s_mul_i32 s1, s33, 6 // (op 929) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 929) least significant
s_mul_hi_u32 s18, s32, 6 // (op 929) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 929) most significant: sum

s_mul_i32 s19, s28, s1 // (op 929) most significant: low of low * high
s_mul_i32 s24, s29, s0 // (op 929) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 929) least significant
s_mul_hi_u32 s25, s28, s0 // (op 929) most significant: high of low * low
s_add_u32 s19, s19, s24 // (op 929) most significant: sum
s_add_u32 s19, s19, s25 // (op 929) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 929) 

s_mov_b32 s24, s30 // (op 929) reg expression
s_mov_b32 s25, s31 // (op 929) reg expression
s_mov_b32 s26, s0 // (op 929) reg expression
s_mov_b32 s27, 131072 // (op 929) call()

s_mul_hi_i32 s0, s16, s8 // (op 946) 
v_and_b32 v12, -2, v1 // (op 946) 
s_add_i32 s1, s0, s16 // (op 946) 

v_sub_u32 v13, v1, v12 // (op 946) 

s_ashr_i32 s0, s1, 31 // (op 946) 
v_lshlrev_b32 v12, 5, v13 // (op 946) 

s_cmp_eq_i32 s8, 0 // (op 946) 
s_mov_b32 s18, scc // (op 946) 
v_lshrrev_b32 v13, 1, v1 // (op 946) 
s_cmp_eq_i32 s18, 0 // (op 946) 
s_cselect_b32 s19, 0, -1 // (op 946) 

s_and_b32 s18, s9, 31 // (op 946) 
s_lshl_b32 s49, 1, s18 // (op 946) 
s_add_i32 s50, s49, s19 // (op 946) 


s_and_b32 s19, s0, s50 // (op 946) 


s_add_i32 s0, s1, s19 // (op 946) 


s_ashr_i32 s1, s0, s18 // (op 946) 


s_lshl_b32 s0, s9, 24 // (op 946) 
s_ashr_i32 s18, s0, 31 // (op 946) 

s_xor_b32 s0, s1, s18 // (op 946) 

s_sub_i32 s1, s0, s18 // (op 946) 


s_mul_hi_i32 s0, s1, s10 // (op 946) 
s_add_i32 s18, s0, s1 // (op 946) 

s_ashr_i32 s0, s18, 31 // (op 946) 
s_cmp_eq_i32 s10, 0 // (op 946) 
s_mov_b32 s19, scc // (op 946) 
s_cmp_eq_i32 s19, 0 // (op 946) 
s_cselect_b32 s49, 0, -1 // (op 946) 

s_and_b32 s19, s11, 31 // (op 946) 
s_lshl_b32 s50, 1, s19 // (op 946) 
s_add_i32 s51, s50, s49 // (op 946) 


s_and_b32 s49, s0, s51 // (op 946) 


s_add_i32 s0, s18, s49 // (op 946) 


s_ashr_i32 s18, s0, s19 // (op 946) 


s_lshl_b32 s0, s11, 24 // (op 946) 
s_ashr_i32 s19, s0, 31 // (op 946) 

s_xor_b32 s0, s18, s19 // (op 946) 

s_sub_i32 s18, s0, s19 // (op 946) 


s_mul_i32 s0, s1, s13 // (op 946) 

s_sub_u32 s1, s16, s0 // (op 946) 

s_mul_i32 s0, s18, s13 // (op 946) 
s_add_u32 s0, s0, s1 // (op 946) 


s_mul_hi_i32 s1, s16, s7 // (op 946) 
s_add_i32 s18, s1, s16 // (op 946) 

s_ashr_i32 s1, s18, 31 // (op 946) 
s_cmp_eq_i32 s7, 0 // (op 946) 
s_mov_b32 s19, scc // (op 946) 
s_cmp_eq_i32 s19, 0 // (op 946) 
s_cselect_b32 s49, 0, -1 // (op 946) 

s_and_b32 s19, s14, 31 // (op 946) 
s_lshl_b32 s50, 1, s19 // (op 946) 
s_add_i32 s51, s50, s49 // (op 946) 


s_and_b32 s49, s1, s51 // (op 946) 


s_add_i32 s1, s18, s49 // (op 946) 


s_ashr_i32 s18, s1, s19 // (op 946) 


s_lshl_b32 s1, s14, 24 // (op 946) 
s_ashr_i32 s19, s1, 31 // (op 946) 

s_xor_b32 s1, s18, s19 // (op 946) 

s_sub_i32 s18, s1, s19 // (op 946) 


s_add_u32 s54, s4, 32 // (op 946) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 946) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 946) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 946) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 946) 

s_mul_i32 s1, s13, s50 // (op 946) 

s_mul_i32 s19, s6, s1 // (op 946) 

s_mul_i32 s1, s18, s19 // (op 946) 

s_sub_u32 s19, s16, s1 // (op 946) 

s_mov_b32 s1, s19 // (op 946) 
s_mul_hi_i32 s49, s1, s12 // (op 946) 
s_add_i32 s50, s49, s1 // (op 946) 


s_ashr_i32 s1, s50, 31 // (op 946) 
s_cmp_eq_i32 s12, 0 // (op 946) 
s_mov_b32 s49, scc // (op 946) 
s_cmp_eq_i32 s49, 0 // (op 946) 
s_cselect_b32 s51, 0, -1 // (op 946) 

s_and_b32 s49, s48, 31 // (op 946) 
s_lshl_b32 s52, 1, s49 // (op 946) 
s_add_i32 s53, s52, s51 // (op 946) 


s_and_b32 s51, s1, s53 // (op 946) 


s_add_i32 s1, s50, s51 // (op 946) 


s_ashr_i32 s50, s1, s49 // (op 946) 


s_lshl_b32 s1, s48, 24 // (op 946) 
s_ashr_i32 s49, s1, 31 // (op 946) 

s_xor_b32 s1, s50, s49 // (op 946) 

s_sub_i32 s50, s1, s49 // (op 946) 


s_mul_i32 s1, s15, s13 // (op 946) 
s_add_u32 s54, s20, 64 // (op 946) Start of Int64 add, locking SCC
s_addc_u32 s55, s21, 0 // (op 946) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 946) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 946) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 6 // (op 946) 

s_sub_i32 s49, s54, s1 // (op 946) 

s_mul_i32 s51, s50, s49 // (op 946) 


s_sub_u32 s49, s19, s51 // (op 946) 


s_add_u32 s19, s49, s1 // (op 946) 


s_cmp_eq_i32 s18, 0 // (op 946) 
s_mov_b32 s1, scc // (op 946) 

s_cmp_eq_i32 s1, 0 // (op 946) 
s_cselect_b32 s18, s19, s0 // (op 946) 



v_lshl_add_u32 v15, s18, 6, v12 // (op 946) 


v_mov_b32 v17, 0 // (op 946) convert
v_mov_b32 v16, v15 // (op 946) convert

v_mul_lo_u32 v19, v13, s29 // (op 946) most significant: low of low * high
v_mul_hi_u32 v12, v13, s28 // (op 946) most significant: high of low * low
v_mul_lo_u32 v18, v13, s28 // (op 946) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 946) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 946) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 946) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 946) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 946) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 946) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 946) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 946) 

v_mov_b32 v15, v12 // (op 946) convert

s_mul_i32 s1, s33, 6 // (op 948) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 948) least significant
s_mul_hi_u32 s18, s32, 6 // (op 948) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 948) most significant: sum

s_mul_i32 s19, s28, s1 // (op 948) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 948) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 948) least significant
s_mul_hi_u32 s50, s28, s0 // (op 948) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 948) most significant: sum
s_add_u32 s19, s19, s50 // (op 948) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 948) 

s_mov_b32 s24, s30 // (op 948) reg expression
s_mov_b32 s25, s31 // (op 948) reg expression
s_mov_b32 s26, s0 // (op 948) reg expression
s_mov_b32 s27, 131072 // (op 948) call()

s_mul_hi_i32 s0, s16, s8 // (op 991) 
v_lshrrev_b32 v12, 3, v1 // (op 991) 
s_add_i32 s1, s0, s16 // (op 991) 

v_and_b32 v13, -8, v1 // (op 991) 
s_ashr_i32 s0, s1, 31 // (op 991) 
v_sub_u32 v16, v1, v13 // (op 991) 

s_cmp_eq_i32 s8, 0 // (op 991) 
s_mov_b32 s18, scc // (op 991) 
v_lshlrev_b32 v13, 4, v16 // (op 991) 

s_cmp_eq_i32 s18, 0 // (op 991) 
s_cselect_b32 s19, 0, -1 // (op 991) 

v_mov_b32 v17, 0 // (op 991) convert
v_mov_b32 v16, v13 // (op 991) convert

s_and_b32 s18, s9, 31 // (op 991) 
s_lshl_b32 s49, 1, s18 // (op 991) 
s_add_i32 s50, s49, s19 // (op 991) 


s_and_b32 s19, s0, s50 // (op 991) 


s_add_i32 s0, s1, s19 // (op 991) 


s_ashr_i32 s1, s0, s18 // (op 991) 


s_lshl_b32 s0, s9, 24 // (op 991) 
s_ashr_i32 s18, s0, 31 // (op 991) 

s_xor_b32 s0, s1, s18 // (op 991) 

s_sub_i32 s1, s0, s18 // (op 991) 


s_mul_hi_i32 s0, s1, s10 // (op 991) 
s_add_i32 s18, s0, s1 // (op 991) 

s_ashr_i32 s0, s18, 31 // (op 991) 
s_cmp_eq_i32 s10, 0 // (op 991) 
s_mov_b32 s19, scc // (op 991) 
s_cmp_eq_i32 s19, 0 // (op 991) 
s_cselect_b32 s49, 0, -1 // (op 991) 

s_and_b32 s19, s11, 31 // (op 991) 
s_lshl_b32 s50, 1, s19 // (op 991) 
s_add_i32 s51, s50, s49 // (op 991) 


s_and_b32 s49, s0, s51 // (op 991) 


s_add_i32 s0, s18, s49 // (op 991) 


s_ashr_i32 s18, s0, s19 // (op 991) 


s_lshl_b32 s0, s11, 24 // (op 991) 
s_ashr_i32 s19, s0, 31 // (op 991) 

s_xor_b32 s0, s18, s19 // (op 991) 

s_sub_i32 s18, s0, s19 // (op 991) 


s_add_u32 s54, s4, 32 // (op 991) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 991) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 991) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 991) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 991) 

s_mul_i32 s0, s18, s50 // (op 991) 

s_sub_i32 s49, s1, s0 // (op 991) 


s_mul_hi_i32 s0, s16, s7 // (op 991) 
s_add_i32 s1, s0, s16 // (op 991) 

s_ashr_i32 s0, s1, 31 // (op 991) 
s_cmp_eq_i32 s7, 0 // (op 991) 
s_mov_b32 s18, scc // (op 991) 
s_cmp_eq_i32 s18, 0 // (op 991) 
s_cselect_b32 s19, 0, -1 // (op 991) 

s_and_b32 s18, s14, 31 // (op 991) 
s_lshl_b32 s52, 1, s18 // (op 991) 
s_add_i32 s53, s52, s19 // (op 991) 


s_and_b32 s19, s0, s53 // (op 991) 


s_add_i32 s0, s1, s19 // (op 991) 


s_ashr_i32 s1, s0, s18 // (op 991) 


s_lshl_b32 s0, s14, 24 // (op 991) 
s_ashr_i32 s18, s0, 31 // (op 991) 

s_xor_b32 s0, s1, s18 // (op 991) 

s_sub_i32 s1, s0, s18 // (op 991) 


s_mul_i32 s0, s13, s50 // (op 991) 

s_mul_i32 s18, s6, s0 // (op 991) 

s_mul_i32 s0, s1, s18 // (op 991) 

s_sub_u32 s18, s16, s0 // (op 991) 

s_mul_hi_i32 s0, s18, s12 // (op 991) 
s_add_i32 s19, s0, s18 // (op 991) 


s_ashr_i32 s0, s19, 31 // (op 991) 
s_cmp_eq_i32 s12, 0 // (op 991) 
s_mov_b32 s18, scc // (op 991) 
s_cmp_eq_i32 s18, 0 // (op 991) 
s_cselect_b32 s50, 0, -1 // (op 991) 

s_and_b32 s18, s48, 31 // (op 991) 
s_lshl_b32 s51, 1, s18 // (op 991) 
s_add_i32 s52, s51, s50 // (op 991) 


s_and_b32 s50, s0, s52 // (op 991) 


s_add_i32 s0, s19, s50 // (op 991) 


s_ashr_i32 s19, s0, s18 // (op 991) 


s_lshl_b32 s0, s48, 24 // (op 991) 
s_ashr_i32 s18, s0, 31 // (op 991) 

s_xor_b32 s0, s19, s18 // (op 991) 

s_sub_i32 s19, s0, s18 // (op 991) 


s_cmp_eq_i32 s1, 0 // (op 991) 
s_mov_b32 s0, scc // (op 991) 

s_cmp_eq_i32 s0, 0 // (op 991) 
s_cselect_b32 s1, s19, s49 // (op 991) 



v_lshl_add_u32 v13, s1, 5, v12 // (op 991) 


v_mul_lo_u32 v19, v13, s35 // (op 991) most significant: low of low * high
v_mul_hi_u32 v12, v13, s34 // (op 991) most significant: high of low * low
v_mul_lo_u32 v18, v13, s34 // (op 991) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 991) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 991) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 991) most significant half



s_mul_i32 s1, s34, s5 // (op 993) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 993) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 993) least significant
s_mul_hi_u32 s19, s34, s4 // (op 993) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 993) most significant: sum
s_add_u32 s1, s1, s19 // (op 993) most significant: sum


s_mov_b32 s52, s36 // (op 993) reg expression
s_mov_b32 s53, s37 // (op 993) reg expression
s_mov_b32 s54, s0 // (op 993) reg expression
s_mov_b32 s55, 131072 // (op 993) call()

s_mul_hi_i32 s0, s16, s8 // (op 1010) 
v_lshrrev_b32 v16, 3, v1 // (op 1010) 
s_add_i32 s1, s0, s16 // (op 1010) 

v_and_b32 v17, -8, v1 // (op 1010) 
s_ashr_i32 s0, s1, 31 // (op 1010) 
v_sub_u32 v18, v1, v17 // (op 1010) 

s_cmp_eq_i32 s8, 0 // (op 1010) 
s_mov_b32 s18, scc // (op 1010) 
v_lshlrev_b32 v17, 4, v18 // (op 1010) 

s_cmp_eq_i32 s18, 0 // (op 1010) 
s_cselect_b32 s19, 0, -1 // (op 1010) 

v_mov_b32 v19, 0 // (op 1010) convert
v_mov_b32 v18, v17 // (op 1010) convert

s_and_b32 s18, s9, 31 // (op 1010) 
s_lshl_b32 s49, 1, s18 // (op 1010) 
s_add_i32 s50, s49, s19 // (op 1010) 


s_and_b32 s19, s0, s50 // (op 1010) 


s_add_i32 s0, s1, s19 // (op 1010) 


s_ashr_i32 s1, s0, s18 // (op 1010) 


s_lshl_b32 s0, s9, 24 // (op 1010) 
s_ashr_i32 s18, s0, 31 // (op 1010) 

s_xor_b32 s0, s1, s18 // (op 1010) 

s_sub_i32 s1, s0, s18 // (op 1010) 


s_mul_hi_i32 s0, s1, s10 // (op 1010) 
s_add_i32 s18, s0, s1 // (op 1010) 

s_ashr_i32 s0, s18, 31 // (op 1010) 
s_cmp_eq_i32 s10, 0 // (op 1010) 
s_mov_b32 s19, scc // (op 1010) 
s_cmp_eq_i32 s19, 0 // (op 1010) 
s_cselect_b32 s49, 0, -1 // (op 1010) 

s_and_b32 s19, s11, 31 // (op 1010) 
s_lshl_b32 s50, 1, s19 // (op 1010) 
s_add_i32 s51, s50, s49 // (op 1010) 


s_and_b32 s49, s0, s51 // (op 1010) 


s_add_i32 s0, s18, s49 // (op 1010) 


s_ashr_i32 s18, s0, s19 // (op 1010) 


s_lshl_b32 s0, s11, 24 // (op 1010) 
s_ashr_i32 s19, s0, 31 // (op 1010) 

s_xor_b32 s0, s18, s19 // (op 1010) 

s_sub_i32 s18, s0, s19 // (op 1010) 


s_add_u32 s50, s4, 32 // (op 1010) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1010) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1010) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1010) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 1010) 

s_mul_i32 s0, s18, s50 // (op 1010) 

s_sub_i32 s49, s1, s0 // (op 1010) 


s_mul_hi_i32 s0, s16, s7 // (op 1010) 
s_add_i32 s1, s0, s16 // (op 1010) 

s_ashr_i32 s0, s1, 31 // (op 1010) 
s_cmp_eq_i32 s7, 0 // (op 1010) 
s_mov_b32 s18, scc // (op 1010) 
s_cmp_eq_i32 s18, 0 // (op 1010) 
s_cselect_b32 s19, 0, -1 // (op 1010) 

s_and_b32 s18, s14, 31 // (op 1010) 
s_lshl_b32 s64, 1, s18 // (op 1010) 
s_add_i32 s65, s64, s19 // (op 1010) 


s_and_b32 s19, s0, s65 // (op 1010) 


s_add_i32 s0, s1, s19 // (op 1010) 


s_ashr_i32 s1, s0, s18 // (op 1010) 


s_lshl_b32 s0, s14, 24 // (op 1010) 
s_ashr_i32 s18, s0, 31 // (op 1010) 

s_xor_b32 s0, s1, s18 // (op 1010) 

s_sub_i32 s1, s0, s18 // (op 1010) 


s_mul_i32 s0, s13, s50 // (op 1010) 

s_mul_i32 s18, s6, s0 // (op 1010) 

s_mul_i32 s0, s1, s18 // (op 1010) 

s_sub_u32 s18, s16, s0 // (op 1010) 

s_mul_hi_i32 s0, s18, s12 // (op 1010) 
s_add_i32 s19, s0, s18 // (op 1010) 


s_ashr_i32 s0, s19, 31 // (op 1010) 
s_cmp_eq_i32 s12, 0 // (op 1010) 
s_mov_b32 s18, scc // (op 1010) 
s_cmp_eq_i32 s18, 0 // (op 1010) 
s_cselect_b32 s50, 0, -1 // (op 1010) 

s_and_b32 s18, s48, 31 // (op 1010) 
s_lshl_b32 s51, 1, s18 // (op 1010) 
s_add_i32 s64, s51, s50 // (op 1010) 


s_and_b32 s50, s0, s64 // (op 1010) 


s_add_i32 s0, s19, s50 // (op 1010) 


s_ashr_i32 s19, s0, s18 // (op 1010) 


s_lshl_b32 s0, s48, 24 // (op 1010) 
s_ashr_i32 s18, s0, 31 // (op 1010) 

s_xor_b32 s0, s19, s18 // (op 1010) 

s_sub_i32 s19, s0, s18 // (op 1010) 


s_cmp_eq_i32 s1, 0 // (op 1010) 
s_mov_b32 s0, scc // (op 1010) 

s_cmp_eq_i32 s0, 0 // (op 1010) 
s_cselect_b32 s1, s19, s49 // (op 1010) 



v_lshl_add_u32 v17, s1, 5, v16 // (op 1010) 


v_mul_lo_u32 v21, v17, s35 // (op 1010) most significant: low of low * high
v_mul_hi_u32 v16, v17, s34 // (op 1010) most significant: high of low * low
v_mul_lo_u32 v20, v17, s34 // (op 1010) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 1010) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 1010) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 1010) most significant half



s_mul_i32 s1, s34, s5 // (op 1012) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 1012) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 1012) least significant
s_mul_hi_u32 s19, s34, s4 // (op 1012) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1012) most significant: sum
s_add_u32 s1, s1, s19 // (op 1012) most significant: sum


s_mov_b32 s52, s36 // (op 1012) reg expression
s_mov_b32 s53, s37 // (op 1012) reg expression
s_mov_b32 s54, s0 // (op 1012) reg expression
s_mov_b32 s55, 131072 // (op 1012) call()

s_mul_hi_i32 s0, s16, s8 // (op 1055) 
v_and_b32 v18, -64, v1 // (op 1055) 
s_add_i32 s1, s0, s16 // (op 1055) 

v_sub_u32 v19, v1, v18 // (op 1055) 

s_ashr_i32 s0, s1, 31 // (op 1055) 
v_lshrrev_b32 v18, 6, v1 // (op 1055) 
s_cmp_eq_i32 s8, 0 // (op 1055) 
s_mov_b32 s18, scc // (op 1055) 
s_cmp_eq_i32 s18, 0 // (op 1055) 
s_cselect_b32 s19, 0, -1 // (op 1055) 

s_and_b32 s18, s9, 31 // (op 1055) 
s_lshl_b32 s49, 1, s18 // (op 1055) 
s_add_i32 s50, s49, s19 // (op 1055) 


s_and_b32 s19, s0, s50 // (op 1055) 


s_add_i32 s0, s1, s19 // (op 1055) 


s_ashr_i32 s1, s0, s18 // (op 1055) 


s_lshl_b32 s0, s9, 24 // (op 1055) 
s_ashr_i32 s18, s0, 31 // (op 1055) 

s_xor_b32 s0, s1, s18 // (op 1055) 

s_sub_i32 s1, s0, s18 // (op 1055) 


s_mul_hi_i32 s0, s1, s10 // (op 1055) 
s_add_i32 s18, s0, s1 // (op 1055) 

s_ashr_i32 s0, s18, 31 // (op 1055) 
s_cmp_eq_i32 s10, 0 // (op 1055) 
s_mov_b32 s19, scc // (op 1055) 
s_cmp_eq_i32 s19, 0 // (op 1055) 
s_cselect_b32 s49, 0, -1 // (op 1055) 

s_and_b32 s19, s11, 31 // (op 1055) 
s_lshl_b32 s50, 1, s19 // (op 1055) 
s_add_i32 s51, s50, s49 // (op 1055) 


s_and_b32 s49, s0, s51 // (op 1055) 


s_add_i32 s0, s18, s49 // (op 1055) 


s_ashr_i32 s18, s0, s19 // (op 1055) 


s_lshl_b32 s0, s11, 24 // (op 1055) 
s_ashr_i32 s19, s0, 31 // (op 1055) 

s_xor_b32 s0, s18, s19 // (op 1055) 

s_sub_i32 s18, s0, s19 // (op 1055) 


s_mul_i32 s0, s1, s13 // (op 1055) 

s_sub_u32 s1, s16, s0 // (op 1055) 

s_mul_i32 s0, s18, s13 // (op 1055) 
s_add_u32 s0, s0, s1 // (op 1055) 


s_mul_hi_i32 s1, s16, s7 // (op 1055) 
s_add_i32 s18, s1, s16 // (op 1055) 

s_ashr_i32 s1, s18, 31 // (op 1055) 
s_cmp_eq_i32 s7, 0 // (op 1055) 
s_mov_b32 s19, scc // (op 1055) 
s_cmp_eq_i32 s19, 0 // (op 1055) 
s_cselect_b32 s49, 0, -1 // (op 1055) 

s_and_b32 s19, s14, 31 // (op 1055) 
s_lshl_b32 s50, 1, s19 // (op 1055) 
s_add_i32 s51, s50, s49 // (op 1055) 


s_and_b32 s49, s1, s51 // (op 1055) 


s_add_i32 s1, s18, s49 // (op 1055) 


s_ashr_i32 s18, s1, s19 // (op 1055) 


s_lshl_b32 s1, s14, 24 // (op 1055) 
s_ashr_i32 s19, s1, 31 // (op 1055) 

s_xor_b32 s1, s18, s19 // (op 1055) 

s_sub_i32 s18, s1, s19 // (op 1055) 


s_add_u32 s50, s4, 32 // (op 1055) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1055) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1055) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1055) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 1055) 

s_mul_i32 s1, s13, s50 // (op 1055) 

s_mul_i32 s19, s6, s1 // (op 1055) 

s_mul_i32 s1, s18, s19 // (op 1055) 

s_sub_u32 s19, s16, s1 // (op 1055) 

s_mov_b32 s1, s19 // (op 1055) 
s_mul_hi_i32 s49, s1, s12 // (op 1055) 
s_add_i32 s50, s49, s1 // (op 1055) 


s_ashr_i32 s1, s50, 31 // (op 1055) 
s_cmp_eq_i32 s12, 0 // (op 1055) 
s_mov_b32 s49, scc // (op 1055) 
s_cmp_eq_i32 s49, 0 // (op 1055) 
s_cselect_b32 s51, 0, -1 // (op 1055) 

s_and_b32 s49, s48, 31 // (op 1055) 
s_lshl_b32 s64, 1, s49 // (op 1055) 
s_add_i32 s65, s64, s51 // (op 1055) 


s_and_b32 s51, s1, s65 // (op 1055) 


s_add_i32 s1, s50, s51 // (op 1055) 


s_ashr_i32 s50, s1, s49 // (op 1055) 


s_lshl_b32 s1, s48, 24 // (op 1055) 
s_ashr_i32 s49, s1, 31 // (op 1055) 

s_xor_b32 s1, s50, s49 // (op 1055) 

s_sub_i32 s50, s1, s49 // (op 1055) 


s_mul_i32 s1, s15, s13 // (op 1055) 
s_add_u32 s64, s20, 64 // (op 1055) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 1055) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1055) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1055) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 6 // (op 1055) 

s_sub_i32 s49, s64, s1 // (op 1055) 

s_mul_i32 s51, s50, s49 // (op 1055) 


s_sub_u32 s49, s19, s51 // (op 1055) 


s_add_u32 s19, s49, s1 // (op 1055) 


s_cmp_eq_i32 s18, 0 // (op 1055) 
s_mov_b32 s1, scc // (op 1055) 

s_cmp_eq_i32 s1, 0 // (op 1055) 
s_cselect_b32 s18, s19, s0 // (op 1055) 



v_lshl_add_u32 v20, s18, 6, v19 // (op 1055) 


v_mul_lo_u32 v19, v18, s38 // (op 1055) 

v_add_u32 v18, v20, v19 // (op 1055) 


s_mul_i32 s1, s38, s43 // (op 1057) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1057) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1057) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1057) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1057) most significant: sum
s_add_u32 s1, s1, s19 // (op 1057) most significant: sum


s_mov_b32 s64, s40 // (op 1057) reg expression
s_mov_b32 s65, s41 // (op 1057) reg expression
s_mov_b32 s66, s0 // (op 1057) reg expression
s_mov_b32 s67, 131072 // (op 1057) call()

s_mul_hi_i32 s0, s16, s8 // (op 1074) 
v_and_b32 v19, -64, v1 // (op 1074) 
s_add_i32 s1, s0, s16 // (op 1074) 

v_sub_u32 v20, v1, v19 // (op 1074) 

s_ashr_i32 s0, s1, 31 // (op 1074) 
v_lshrrev_b32 v19, 6, v1 // (op 1074) 
s_cmp_eq_i32 s8, 0 // (op 1074) 
s_mov_b32 s18, scc // (op 1074) 
s_cmp_eq_i32 s18, 0 // (op 1074) 
s_cselect_b32 s19, 0, -1 // (op 1074) 

s_and_b32 s18, s9, 31 // (op 1074) 
s_lshl_b32 s49, 1, s18 // (op 1074) 
s_add_i32 s50, s49, s19 // (op 1074) 


s_and_b32 s19, s0, s50 // (op 1074) 


s_add_i32 s0, s1, s19 // (op 1074) 


s_ashr_i32 s1, s0, s18 // (op 1074) 


s_lshl_b32 s0, s9, 24 // (op 1074) 
s_ashr_i32 s18, s0, 31 // (op 1074) 

s_xor_b32 s0, s1, s18 // (op 1074) 

s_sub_i32 s1, s0, s18 // (op 1074) 


s_mul_hi_i32 s0, s1, s10 // (op 1074) 
s_add_i32 s18, s0, s1 // (op 1074) 

s_ashr_i32 s0, s18, 31 // (op 1074) 
s_cmp_eq_i32 s10, 0 // (op 1074) 
s_mov_b32 s19, scc // (op 1074) 
s_cmp_eq_i32 s19, 0 // (op 1074) 
s_cselect_b32 s49, 0, -1 // (op 1074) 

s_and_b32 s19, s11, 31 // (op 1074) 
s_lshl_b32 s50, 1, s19 // (op 1074) 
s_add_i32 s51, s50, s49 // (op 1074) 


s_and_b32 s49, s0, s51 // (op 1074) 


s_add_i32 s0, s18, s49 // (op 1074) 


s_ashr_i32 s18, s0, s19 // (op 1074) 


s_lshl_b32 s0, s11, 24 // (op 1074) 
s_ashr_i32 s19, s0, 31 // (op 1074) 

s_xor_b32 s0, s18, s19 // (op 1074) 

s_sub_i32 s18, s0, s19 // (op 1074) 


s_mul_i32 s0, s1, s13 // (op 1074) 

s_sub_u32 s1, s16, s0 // (op 1074) 

s_mul_i32 s0, s18, s13 // (op 1074) 
s_add_u32 s0, s0, s1 // (op 1074) 


s_mul_hi_i32 s1, s16, s7 // (op 1074) 
s_add_i32 s18, s1, s16 // (op 1074) 

s_ashr_i32 s1, s18, 31 // (op 1074) 
s_cmp_eq_i32 s7, 0 // (op 1074) 
s_mov_b32 s19, scc // (op 1074) 
s_cmp_eq_i32 s19, 0 // (op 1074) 
s_cselect_b32 s49, 0, -1 // (op 1074) 

s_and_b32 s19, s14, 31 // (op 1074) 
s_lshl_b32 s50, 1, s19 // (op 1074) 
s_add_i32 s51, s50, s49 // (op 1074) 


s_and_b32 s49, s1, s51 // (op 1074) 


s_add_i32 s1, s18, s49 // (op 1074) 


s_ashr_i32 s18, s1, s19 // (op 1074) 


s_lshl_b32 s1, s14, 24 // (op 1074) 
s_ashr_i32 s19, s1, 31 // (op 1074) 

s_xor_b32 s1, s18, s19 // (op 1074) 

s_sub_i32 s18, s1, s19 // (op 1074) 


s_add_u32 s50, s4, 32 // (op 1074) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1074) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1074) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1074) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 1074) 

s_mul_i32 s1, s13, s50 // (op 1074) 

s_mul_i32 s19, s6, s1 // (op 1074) 

s_mul_i32 s1, s18, s19 // (op 1074) 

s_sub_u32 s19, s16, s1 // (op 1074) 

s_mov_b32 s1, s19 // (op 1074) 
s_mul_hi_i32 s49, s1, s12 // (op 1074) 
s_add_i32 s50, s49, s1 // (op 1074) 


s_ashr_i32 s1, s50, 31 // (op 1074) 
s_cmp_eq_i32 s12, 0 // (op 1074) 
s_mov_b32 s49, scc // (op 1074) 
s_cmp_eq_i32 s49, 0 // (op 1074) 
s_cselect_b32 s51, 0, -1 // (op 1074) 

s_and_b32 s49, s48, 31 // (op 1074) 
s_lshl_b32 s68, 1, s49 // (op 1074) 
s_add_i32 s69, s68, s51 // (op 1074) 


s_and_b32 s51, s1, s69 // (op 1074) 


s_add_i32 s1, s50, s51 // (op 1074) 


s_ashr_i32 s50, s1, s49 // (op 1074) 


s_lshl_b32 s1, s48, 24 // (op 1074) 
s_ashr_i32 s49, s1, 31 // (op 1074) 

s_xor_b32 s1, s50, s49 // (op 1074) 

s_sub_i32 s50, s1, s49 // (op 1074) 


s_mul_i32 s1, s15, s13 // (op 1074) 
s_add_u32 s68, s20, 64 // (op 1074) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 1074) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1074) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1074) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 6 // (op 1074) 

s_sub_i32 s49, s68, s1 // (op 1074) 

s_mul_i32 s51, s50, s49 // (op 1074) 


s_sub_u32 s49, s19, s51 // (op 1074) 


s_add_u32 s19, s49, s1 // (op 1074) 


s_cmp_eq_i32 s18, 0 // (op 1074) 
s_mov_b32 s1, scc // (op 1074) 

s_cmp_eq_i32 s1, 0 // (op 1074) 
s_cselect_b32 s18, s19, s0 // (op 1074) 



v_lshl_add_u32 v21, s18, 6, v20 // (op 1074) 


v_mul_lo_u32 v20, v19, s38 // (op 1074) 

v_add_u32 v19, v21, v20 // (op 1074) 


s_mul_i32 s1, s38, s43 // (op 1076) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1076) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1076) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1076) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1076) most significant: sum
s_add_u32 s1, s1, s19 // (op 1076) most significant: sum


s_mov_b32 s64, s40 // (op 1076) reg expression
s_mov_b32 s65, s41 // (op 1076) reg expression
s_mov_b32 s66, s0 // (op 1076) reg expression
s_mov_b32 s67, 131072 // (op 1076) call()

s_mul_hi_i32 s0, s16, s8 // (op 1119) 
v_lshrrev_b32 v20, 6, v1 // (op 1119) 
s_add_i32 s1, s0, s16 // (op 1119) 

v_and_b32 v21, 1, v20 // (op 1119) 

s_ashr_i32 s0, s1, 31 // (op 1119) 
v_and_b32 v20, 15, v1 // (op 1119) 
s_cmp_eq_i32 s8, 0 // (op 1119) 
s_mov_b32 s18, scc // (op 1119) 
v_lshl_add_u32 v22, v21, 4, v20 // (op 1119) 


s_cmp_eq_i32 s18, 0 // (op 1119) 
s_cselect_b32 s19, 0, -1 // (op 1119) 

v_and_b32 v20, 63, v1 // (op 1119) 
s_and_b32 s18, s9, 31 // (op 1119) 
v_lshrrev_b32 v21, 4, v20 // (op 1119) 

s_lshl_b32 s49, 1, s18 // (op 1119) 
s_add_i32 s50, s49, s19 // (op 1119) 


s_and_b32 s19, s0, s50 // (op 1119) 


s_add_i32 s0, s1, s19 // (op 1119) 


s_ashr_i32 s1, s0, s18 // (op 1119) 


s_lshl_b32 s0, s9, 24 // (op 1119) 
s_ashr_i32 s18, s0, 31 // (op 1119) 

s_xor_b32 s0, s1, s18 // (op 1119) 

s_sub_i32 s1, s0, s18 // (op 1119) 


s_mul_hi_i32 s0, s1, s10 // (op 1119) 
s_add_i32 s18, s0, s1 // (op 1119) 

s_ashr_i32 s0, s18, 31 // (op 1119) 
s_cmp_eq_i32 s10, 0 // (op 1119) 
s_mov_b32 s19, scc // (op 1119) 
s_cmp_eq_i32 s19, 0 // (op 1119) 
s_cselect_b32 s49, 0, -1 // (op 1119) 

s_and_b32 s19, s11, 31 // (op 1119) 
s_lshl_b32 s50, 1, s19 // (op 1119) 
s_add_i32 s51, s50, s49 // (op 1119) 


s_and_b32 s49, s0, s51 // (op 1119) 


s_add_i32 s0, s18, s49 // (op 1119) 


s_ashr_i32 s18, s0, s19 // (op 1119) 


s_lshl_b32 s0, s11, 24 // (op 1119) 
s_ashr_i32 s19, s0, 31 // (op 1119) 

s_xor_b32 s0, s18, s19 // (op 1119) 

s_sub_i32 s18, s0, s19 // (op 1119) 


s_add_u32 s50, s4, 32 // (op 1119) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1119) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1119) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1119) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 1119) 

s_mul_i32 s0, s18, s50 // (op 1119) 

s_sub_i32 s49, s1, s0 // (op 1119) 


s_mul_hi_i32 s0, s16, s7 // (op 1119) 
s_add_i32 s1, s0, s16 // (op 1119) 

s_ashr_i32 s0, s1, 31 // (op 1119) 
s_cmp_eq_i32 s7, 0 // (op 1119) 
s_mov_b32 s18, scc // (op 1119) 
s_cmp_eq_i32 s18, 0 // (op 1119) 
s_cselect_b32 s19, 0, -1 // (op 1119) 

s_and_b32 s18, s14, 31 // (op 1119) 
s_lshl_b32 s68, 1, s18 // (op 1119) 
s_add_i32 s69, s68, s19 // (op 1119) 


s_and_b32 s19, s0, s69 // (op 1119) 


s_add_i32 s0, s1, s19 // (op 1119) 


s_ashr_i32 s1, s0, s18 // (op 1119) 


s_lshl_b32 s0, s14, 24 // (op 1119) 
s_ashr_i32 s18, s0, 31 // (op 1119) 

s_xor_b32 s0, s1, s18 // (op 1119) 

s_sub_i32 s1, s0, s18 // (op 1119) 


s_mul_i32 s0, s13, s50 // (op 1119) 

s_mul_i32 s18, s6, s0 // (op 1119) 

s_mul_i32 s0, s1, s18 // (op 1119) 

s_sub_u32 s18, s16, s0 // (op 1119) 

s_mul_hi_i32 s0, s18, s12 // (op 1119) 
s_add_i32 s19, s0, s18 // (op 1119) 


s_ashr_i32 s0, s19, 31 // (op 1119) 
s_cmp_eq_i32 s12, 0 // (op 1119) 
s_mov_b32 s18, scc // (op 1119) 
s_cmp_eq_i32 s18, 0 // (op 1119) 
s_cselect_b32 s50, 0, -1 // (op 1119) 

s_and_b32 s18, s48, 31 // (op 1119) 
s_lshl_b32 s51, 1, s18 // (op 1119) 
s_add_i32 s68, s51, s50 // (op 1119) 


s_and_b32 s50, s0, s68 // (op 1119) 


s_add_i32 s0, s19, s50 // (op 1119) 


s_ashr_i32 s19, s0, s18 // (op 1119) 


s_lshl_b32 s0, s48, 24 // (op 1119) 
s_ashr_i32 s18, s0, 31 // (op 1119) 

s_xor_b32 s0, s19, s18 // (op 1119) 

s_sub_i32 s19, s0, s18 // (op 1119) 


s_cmp_eq_i32 s1, 0 // (op 1119) 
s_mov_b32 s0, scc // (op 1119) 

s_cmp_eq_i32 s0, 0 // (op 1119) 
s_cselect_b32 s1, s19, s49 // (op 1119) 



v_lshl_add_u32 v20, s1, 5, v22 // (op 1119) 


v_mul_lo_u32 v22, v20, s44 // (op 1119) 

v_add_u32 v20, v21, v22 // (op 1119) 


s_mul_i32 s1, s44, s5 // (op 1121) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 1121) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 1121) least significant
s_mul_hi_u32 s19, s44, s4 // (op 1121) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1121) most significant: sum
s_add_u32 s1, s1, s19 // (op 1121) most significant: sum


s_mov_b32 s68, s46 // (op 1121) reg expression
s_mov_b32 s69, s47 // (op 1121) reg expression
s_mov_b32 s70, s0 // (op 1121) reg expression
s_mov_b32 s71, 131072 // (op 1121) call()

s_mul_hi_i32 s0, s16, s8 // (op 1138) 
v_lshrrev_b32 v21, 6, v1 // (op 1138) 
s_add_i32 s1, s0, s16 // (op 1138) 

v_and_b32 v22, 1, v21 // (op 1138) 

s_ashr_i32 s0, s1, 31 // (op 1138) 
v_and_b32 v21, 15, v1 // (op 1138) 
s_cmp_eq_i32 s8, 0 // (op 1138) 
s_mov_b32 s18, scc // (op 1138) 
v_lshl_add_u32 v23, v22, 4, v21 // (op 1138) 


s_cmp_eq_i32 s18, 0 // (op 1138) 
s_cselect_b32 s19, 0, -1 // (op 1138) 

v_and_b32 v21, 63, v1 // (op 1138) 
s_and_b32 s18, s9, 31 // (op 1138) 
v_lshrrev_b32 v22, 4, v21 // (op 1138) 

s_lshl_b32 s49, 1, s18 // (op 1138) 
s_add_i32 s50, s49, s19 // (op 1138) 


s_and_b32 s19, s0, s50 // (op 1138) 


s_add_i32 s0, s1, s19 // (op 1138) 


s_ashr_i32 s1, s0, s18 // (op 1138) 


s_lshl_b32 s0, s9, 24 // (op 1138) 
s_ashr_i32 s18, s0, 31 // (op 1138) 

s_xor_b32 s0, s1, s18 // (op 1138) 

s_sub_i32 s1, s0, s18 // (op 1138) 


s_mul_hi_i32 s0, s1, s10 // (op 1138) 
s_add_i32 s18, s0, s1 // (op 1138) 

s_ashr_i32 s0, s18, 31 // (op 1138) 
s_cmp_eq_i32 s10, 0 // (op 1138) 
s_mov_b32 s19, scc // (op 1138) 
s_cmp_eq_i32 s19, 0 // (op 1138) 
s_cselect_b32 s49, 0, -1 // (op 1138) 

s_and_b32 s19, s11, 31 // (op 1138) 
s_lshl_b32 s50, 1, s19 // (op 1138) 
s_add_i32 s51, s50, s49 // (op 1138) 


s_and_b32 s49, s0, s51 // (op 1138) 


s_add_i32 s0, s18, s49 // (op 1138) 


s_ashr_i32 s18, s0, s19 // (op 1138) 


s_lshl_b32 s0, s11, 24 // (op 1138) 
s_ashr_i32 s19, s0, 31 // (op 1138) 

s_xor_b32 s0, s18, s19 // (op 1138) 

s_sub_i32 s18, s0, s19 // (op 1138) 


s_add_u32 s50, s4, 32 // (op 1138) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1138) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1138) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1138) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1138) 

s_mul_i32 s0, s18, s50 // (op 1138) 

s_sub_i32 s49, s1, s0 // (op 1138) 


s_mul_hi_i32 s0, s16, s7 // (op 1138) 
s_add_i32 s1, s0, s16 // (op 1138) 

s_ashr_i32 s0, s1, 31 // (op 1138) 
s_cmp_eq_i32 s7, 0 // (op 1138) 
s_mov_b32 s18, scc // (op 1138) 
s_cmp_eq_i32 s18, 0 // (op 1138) 
s_cselect_b32 s19, 0, -1 // (op 1138) 

s_and_b32 s18, s14, 31 // (op 1138) 
s_lshl_b32 s72, 1, s18 // (op 1138) 
s_add_i32 s73, s72, s19 // (op 1138) 


s_and_b32 s19, s0, s73 // (op 1138) 


s_add_i32 s0, s1, s19 // (op 1138) 


s_ashr_i32 s1, s0, s18 // (op 1138) 


s_lshl_b32 s0, s14, 24 // (op 1138) 
s_ashr_i32 s18, s0, 31 // (op 1138) 

s_xor_b32 s0, s1, s18 // (op 1138) 

s_sub_i32 s1, s0, s18 // (op 1138) 


s_mul_i32 s0, s13, s50 // (op 1138) 

s_mul_i32 s18, s6, s0 // (op 1138) 

s_mul_i32 s0, s1, s18 // (op 1138) 

s_sub_u32 s18, s16, s0 // (op 1138) 

s_mul_hi_i32 s0, s18, s12 // (op 1138) 
s_add_i32 s19, s0, s18 // (op 1138) 


s_ashr_i32 s0, s19, 31 // (op 1138) 
s_cmp_eq_i32 s12, 0 // (op 1138) 
s_mov_b32 s18, scc // (op 1138) 
s_cmp_eq_i32 s18, 0 // (op 1138) 
s_cselect_b32 s50, 0, -1 // (op 1138) 

s_and_b32 s18, s48, 31 // (op 1138) 
s_lshl_b32 s51, 1, s18 // (op 1138) 
s_add_i32 s72, s51, s50 // (op 1138) 


s_and_b32 s50, s0, s72 // (op 1138) 


s_add_i32 s0, s19, s50 // (op 1138) 


s_ashr_i32 s19, s0, s18 // (op 1138) 


s_lshl_b32 s0, s48, 24 // (op 1138) 
s_ashr_i32 s18, s0, 31 // (op 1138) 

s_xor_b32 s0, s19, s18 // (op 1138) 

s_sub_i32 s19, s0, s18 // (op 1138) 


s_cmp_eq_i32 s1, 0 // (op 1138) 
s_mov_b32 s0, scc // (op 1138) 

s_cmp_eq_i32 s0, 0 // (op 1138) 
s_cselect_b32 s1, s19, s49 // (op 1138) 



v_lshl_add_u32 v21, s1, 5, v23 // (op 1138) 


v_mul_lo_u32 v23, v21, s44 // (op 1138) 

v_add_u32 v21, v22, v23 // (op 1138) 


s_mul_i32 s1, s44, s5 // (op 1140) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 1140) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 1140) least significant
s_mul_hi_u32 s19, s44, s4 // (op 1140) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1140) most significant: sum
s_add_u32 s1, s1, s19 // (op 1140) most significant: sum


s_mov_b32 s68, s46 // (op 1140) reg expression
s_mov_b32 s69, s47 // (op 1140) reg expression
s_mov_b32 s70, s0 // (op 1140) reg expression
s_mov_b32 s71, 131072 // (op 1140) call()




buffer_load_ubyte v22, v20, s[68:71], 0 offen offset: 0 // (op 774) Load value
v_and_b32 v23, -8, v1 // (op 1262) 
v_sub_u32 v24, v1, v23 // (op 1262) 

v_lshlrev_b32 v23, 4, v24 // (op 1262) 

v_lshrrev_b32 v24, 3, v1 // (op 1262) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 1262) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 1262) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 882) reg expression
buffer_load_dwordx4 v12, s[52:55], 0 offen offset: 0 lds // (op 882) Load value direct to lds



v_mov_b32 v23, v14 // (op 781) reg expression

buffer_load_dwordx3 v[24:26], v23, s[24:27], 0 offen // (op 781) Load value
buffer_load_dwordx3 v[28:30], v23, s[24:27], 0 offen offset:12 // (op 781) Load value



v_mov_b32 v23, v18 // (op 784) reg expression

buffer_load_ubyte v27, v23, s[64:67], 0 offen offset: 0 // (op 784) Load value




buffer_load_ubyte v23, v20, s[68:71], 0 offen offset:4 // (op 788) Load value

v_and_b32 v20, -8, v1 // (op 1406) 
v_sub_u32 v31, v1, v20 // (op 1406) 

v_lshlrev_b32 v20, 4, v31 // (op 1406) 

v_lshrrev_b32 v31, 3, v1 // (op 1406) 
v_lshl_add_u32 v32, v31, 7, v20 // (op 1406) 


s_nop 0
v_readfirstlane_b32 s0, v32 // (op 1406) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 887) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 887) reg expression
s_mov_b32 s18, 128 // (op 887) call()
buffer_load_dwordx4 v12, s[52:55], s18 offen offset: 0 lds // (op 887) Load value direct to lds






s_lshl_b64 s[0:1], s[28:29], 7 // (op 795) 
s_mul_i32 s19, s1, 6 // (op 795) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 795) least significant
s_mul_hi_u32 s49, s0, 6 // (op 795) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 795) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 795) 

v_add_u32 v20, v14, s0 // (op 795) 


buffer_load_dwordx3 v[32:34], v20, s[24:27], 0 offen // (op 795) Load value
buffer_load_dwordx3 v[36:38], v20, s[24:27], 0 offen offset:12 // (op 795) Load value




s_lshl_b32 s0, s38, 2 // (op 798) 
v_add_u32 v20, v18, s0 // (op 798) 


buffer_load_ubyte v31, v20, s[64:67], 0 offen offset: 0 // (op 798) Load value


v_and_b32 v18, -2, v1 // (op 1205) 
v_sub_u32 v20, v1, v18 // (op 1205) 

v_lshlrev_b32 v18, 5, v20 // (op 1205) 

v_lshrrev_b32 v20, 1, v1 // (op 1205) 
v_lshl_add_u32 v35, v20, 6, v18 // (op 1205) 


v_mul_lo_u32 v18, v35, 6 // (op 1205) 
v_lshrrev_b32 v20, 3, v18 // (op 1205) 

v_lshrrev_b32 v18, 4, v35 // (op 1205) 

v_lshlrev_b32 v35, 2, v18 // (op 1205) 

v_add_u32 v18, v20, v35 // (op 1205) 





s_waitcnt vmcnt(7)
ds_write_b96 v18, v[24:26] offset:8192 // (op 801) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v18, v[28:30] offset:8208 // (op 801) Store local data 

v_and_b32 v18, -64, v1 // (op 1315) 
v_sub_u32 v20, v1, v18 // (op 1315) 

v_and_b32 v18, 4294967232, v1 // (op 1315) 
v_add_u32 v35, v18, v20 // (op 1315) 





s_waitcnt vmcnt(5)
ds_write_b8 v35, v27 offset:16384 // (op 802) Store local data 

v_and_b32 v18, -2, v1 // (op 1176) 
v_sub_u32 v20, v1, v18 // (op 1176) 

v_lshlrev_b32 v18, 5, v20 // (op 1176) 

v_lshrrev_b32 v20, 1, v1 // (op 1176) 
v_lshl_add_u32 v35, v20, 6, v18 // (op 1176) 


v_mul_lo_u32 v18, v35, 6 // (op 1176) 
v_lshrrev_b32 v20, 3, v18 // (op 1176) 

v_lshrrev_b32 v18, 4, v35 // (op 1176) 

v_lshlrev_b32 v35, 2, v18 // (op 1176) 

v_add_u32 v18, v20, v35 // (op 1176) 


v_and_b32 v20, -8, v1 // (op 1234) 
v_sub_u32 v35, v1, v20 // (op 1234) 

v_lshlrev_b32 v20, 4, v35 // (op 1234) 

v_lshrrev_b32 v35, 3, v1 // (op 1234) 
v_lshl_add_u32 v12, v35, 7, v20 // (op 1234) 


s_nop 0
v_readfirstlane_b32 s0, v12 // (op 1234) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -64, v1 // (op 1286) 
v_sub_u32 v35, v1, v20 // (op 1286) 

v_and_b32 v20, 4294967232, v1 // (op 1286) 
v_add_u32 v12, v20, v35 // (op 1286) 


v_and_b32 v20, -2, v1 // (op 1362) 
v_sub_u32 v35, v1, v20 // (op 1362) 

v_lshlrev_b32 v20, 5, v35 // (op 1362) 

v_lshrrev_b32 v35, 1, v1 // (op 1362) 
v_lshl_add_u32 v13, v35, 6, v20 // (op 1362) 


v_mul_lo_u32 v14, v13, 6 // (op 1362) 
v_lshrrev_b32 v20, 3, v14 // (op 1362) 

v_lshrrev_b32 v14, 4, v13 // (op 1362) 

v_lshlrev_b32 v13, 2, v14 // (op 1362) 

v_add_u32 v14, v20, v13 // (op 1362) 


v_and_b32 v13, -8, v1 // (op 1391) 
v_sub_u32 v20, v1, v13 // (op 1391) 

v_lshlrev_b32 v13, 4, v20 // (op 1391) 

v_lshrrev_b32 v20, 3, v1 // (op 1391) 
v_lshl_add_u32 v35, v20, 7, v13 // (op 1391) 


s_nop 0
v_readfirstlane_b32 s1, v35 // (op 1391) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -64, v1 // (op 1430) 
v_sub_u32 v20, v1, v13 // (op 1430) 

v_and_b32 v13, 4294967232, v1 // (op 1430) 
v_add_u32 v35, v13, v20 // (op 1430) 


s_mov_b32 s18, 0 // (op 311) call()
s_ashr_i64 s[50:51], s[32:33], 7 // (op 310) 
s_mov_b32 s19, s50 // (op 310) convert

s_and_b32 s49, s19, 4294967294 // (op 310) 

s_cmp_lt_u32 s18, s49 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_310 // (op 310) Condition: Top (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_310 if false)
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_310:
s_add_u32 s18, s18, 2 // (op 312) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1488) 



ds_read_b96_tr_b6 v[40:42], v0 offset:8192 // (op 65) Transpose load local data 
ds_read_b96_tr_b6 v[43:45], v0 offset:9216 // (op 65) Transpose load local data 
s_barrier  // (op 1501) 



ds_read_b128 v[46:49], v3 // (op 70) Load local data 
ds_read_b128 v[50:53], v3 offset:64 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1514) 


ds_read_u8 v13, v7 offset:16384 // (op 75) Load local data 



ds_read_b96_tr_b6 v[54:56], v0 offset:8224 // (op 340) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:45], v[46:53], a[0:3], v13, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 82) 
ds_read_b96_tr_b6 v[57:59], v0 offset:9248 // (op 340) Transpose load local data 




ds_read_u8 v13, v7 offset:16416 // (op 344) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[46:53], a[4:7], v13, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 346) 





v_mov_b32 v13, v21 // (op 80) 

buffer_load_ubyte v22, v13, s[68:71], 0 offen offset:8 // (op 80) Load value
s_barrier  // (op 1504) 


v_mov_b64 v[40:41], v[16:17] // (op 865) 

s_mov_b32 m0, s0 // (op 865) reg expression
s_mov_b32 s19, 256 // (op 865) call()
buffer_load_dwordx4 v40, s[52:55], s19 offen offset: 0 lds // (op 865) Load value direct to lds



v_mov_b32 v20, v15 // (op 4) 
s_lshl_b64 s[50:51], s[28:29], 7 // (op 4) 
s_mul_i32 s73, s51, 6 // (op 4) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 4) least significant
s_mul_hi_u32 s19, s50, 6 // (op 4) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 4) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 4) 

s_lshl_b32 s19, s50, 1 // (op 4) 

v_add_u32 v39, v20, s19 // (op 4) 


buffer_load_dwordx3 v[24:26], v39, s[24:27], 0 offen // (op 4) Load value
buffer_load_dwordx3 v[28:30], v39, s[24:27], 0 offen offset:12 // (op 4) Load value



v_mov_b32 v39, v19 // (op 17) 
s_lshl_b32 s19, s38, 3 // (op 17) 
v_add_u32 v42, v39, s19 // (op 17) 


buffer_load_ubyte v27, v42, s[64:67], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1531) 



s_waitcnt vmcnt(7)
ds_write_b96 v14, v[32:34] offset:16640 // (op 225) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v14, v[36:38] offset:16656 // (op 225) Store local data 
s_barrier  // (op 1543) 



s_waitcnt vmcnt(5)
ds_write_b8 v35, v31 offset:24832 // (op 236) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1528) 



ds_read_b96_tr_b6 v[42:44], v10 offset:16640 // (op 227) Transpose load local data 
ds_read_b96_tr_b6 v[45:47], v10 offset:17664 // (op 227) Transpose load local data 
s_barrier  // (op 1534) 



ds_read_b128 v[48:51], v4 offset:4096 // (op 229) Load local data 
ds_read_b128 v[52:55], v4 offset:4160 // (op 229) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1540) 


ds_read_u8 v56, v11 offset:24832 // (op 231) Load local data 



ds_read_b96_tr_b6 v[58:60], v10 offset:16672 // (op 353) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:47], v[48:55], a[0:3], v56, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 234) 
ds_read_b96_tr_b6 v[61:63], v10 offset:17696 // (op 353) Transpose load local data 




s_barrier  // (op 1604) 
ds_read_u8 v42, v11 offset:24864 // (op 357) Load local data 
s_barrier  // (op 1608) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:63], v[48:55], a[4:7], v42, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 347) 






buffer_load_ubyte v23, v13, s[68:71], 0 offen offset:12 // (op 233) Load value

s_barrier  // (op 1537) 
v_add_u32 v21, 8, v21 // (op 1130) 


s_add_u32 s19, 4096, s1 // (op 877) 

s_mov_b32 m0, s19 // (op 877) reg expression
s_mov_b32 s49, 384 // (op 877) call()
buffer_load_dwordx4 v40, s[52:55], s49 offen offset: 0 lds // (op 877) Load value direct to lds





v_add_co_u32 v16, vcc, 256, v16 // (op 1002) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1002) most significant half
s_lshl_b64 s[50:51], s[28:29], 7 // (op 224) 
s_mul_i32 s73, s51, 6 // (op 224) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 224) least significant
s_mul_hi_u32 s19, s50, 6 // (op 224) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 224) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 224) 

s_mul_i32 s19, 3, s50 // (op 224) 

v_add_u32 v13, v20, s19 // (op 224) 


buffer_load_dwordx3 v[32:34], v13, s[24:27], 0 offen // (op 224) Load value
buffer_load_dwordx3 v[36:38], v13, s[24:27], 0 offen offset:12 // (op 224) Load value




s_lshl_b32 s19, s38, 2 // (op 238) 
s_mul_i32 s49, 3, s19 // (op 238) 

v_add_u32 v13, v39, s49 // (op 238) 


s_lshl_b64 s[50:51], s[28:29], 7 // (op 938) 
s_mul_i32 s73, s51, 6 // (op 938) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 938) least significant
s_mul_hi_u32 s19, s50, 6 // (op 938) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 938) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 938) 

s_lshl_b32 s19, s50, 1 // (op 938) 

v_add_u32 v15, v15, s19 // (op 938) 

buffer_load_ubyte v31, v13, s[64:67], 0 offen offset: 0 // (op 238) Load value


s_barrier  // (op 1491) 
s_lshl_b32 s19, s38, 3 // (op 1066) 



v_add_u32 v19, v19, s19 // (op 1066) 

s_waitcnt vmcnt(7)
ds_write_b96 v18, v[24:26] offset:8192 // (op 6) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v18, v[28:30] offset:8208 // (op 6) Store local data 
s_barrier  // (op 1517) 



s_waitcnt vmcnt(5)
ds_write_b8 v12, v27 offset:16384 // (op 21) Store local data 
s_ashr_i64 s[50:51], s[32:33], 7 // (op 310) 
s_mov_b32 s19, s50 // (op 310) convert

s_and_b32 s49, s19, 4294967294 // (op 310) 

s_cmp_lt_u32 s18, s49 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_310 // (op 310) Condition: Bottom (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_310 if true)
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_310:














s_barrier  // (op 1588) 
s_mul_hi_i32 s0, s16, s8 // (op 900) 
v_and_b32 v35, -2, v1 // (op 900) 
s_add_i32 s1, s0, s16 // (op 900) 

v_sub_u32 v12, v1, v35 // (op 900) 

s_ashr_i32 s0, s1, 31 // (op 900) 
v_lshlrev_b32 v35, 5, v12 // (op 900) 

s_cmp_eq_i32 s8, 0 // (op 900) 
s_mov_b32 s18, scc // (op 900) 
v_lshrrev_b32 v12, 1, v1 // (op 900) 
s_cmp_eq_i32 s18, 0 // (op 900) 
s_cselect_b32 s19, 0, -1 // (op 900) 

s_and_b32 s18, s9, 31 // (op 900) 
s_lshl_b32 s49, 1, s18 // (op 900) 
s_add_i32 s50, s49, s19 // (op 900) 


s_and_b32 s19, s0, s50 // (op 900) 


s_add_i32 s0, s1, s19 // (op 900) 


s_ashr_i32 s1, s0, s18 // (op 900) 


s_lshl_b32 s0, s9, 24 // (op 900) 
s_ashr_i32 s18, s0, 31 // (op 900) 

s_xor_b32 s0, s1, s18 // (op 900) 

s_sub_i32 s1, s0, s18 // (op 900) 


s_mul_hi_i32 s0, s1, s10 // (op 900) 
s_add_i32 s18, s0, s1 // (op 900) 

s_ashr_i32 s0, s18, 31 // (op 900) 
s_cmp_eq_i32 s10, 0 // (op 900) 
s_mov_b32 s19, scc // (op 900) 
s_cmp_eq_i32 s19, 0 // (op 900) 
s_cselect_b32 s49, 0, -1 // (op 900) 

s_and_b32 s19, s11, 31 // (op 900) 
s_lshl_b32 s50, 1, s19 // (op 900) 
s_add_i32 s51, s50, s49 // (op 900) 


s_and_b32 s49, s0, s51 // (op 900) 


s_add_i32 s0, s18, s49 // (op 900) 


s_ashr_i32 s18, s0, s19 // (op 900) 


s_lshl_b32 s0, s11, 24 // (op 900) 
s_ashr_i32 s19, s0, 31 // (op 900) 

s_xor_b32 s0, s18, s19 // (op 900) 

s_sub_i32 s18, s0, s19 // (op 900) 


s_mul_i32 s0, s1, s13 // (op 900) 

s_sub_u32 s1, s16, s0 // (op 900) 

s_mul_i32 s0, s18, s13 // (op 900) 
s_add_u32 s0, s0, s1 // (op 900) 


s_mul_hi_i32 s1, s16, s7 // (op 900) 
s_add_i32 s18, s1, s16 // (op 900) 

s_ashr_i32 s1, s18, 31 // (op 900) 
s_cmp_eq_i32 s7, 0 // (op 900) 
s_mov_b32 s19, scc // (op 900) 
s_cmp_eq_i32 s19, 0 // (op 900) 
s_cselect_b32 s49, 0, -1 // (op 900) 

s_and_b32 s19, s14, 31 // (op 900) 
s_lshl_b32 s50, 1, s19 // (op 900) 
s_add_i32 s51, s50, s49 // (op 900) 


s_and_b32 s49, s1, s51 // (op 900) 


s_add_i32 s1, s18, s49 // (op 900) 


s_ashr_i32 s18, s1, s19 // (op 900) 


s_lshl_b32 s1, s14, 24 // (op 900) 
s_ashr_i32 s19, s1, 31 // (op 900) 

s_xor_b32 s1, s18, s19 // (op 900) 

s_sub_i32 s18, s1, s19 // (op 900) 


s_add_u32 s50, s4, 32 // (op 900) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 900) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 900) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 900) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 900) 

s_mul_i32 s1, s13, s50 // (op 900) 

s_mul_i32 s19, s6, s1 // (op 900) 

s_mul_i32 s1, s18, s19 // (op 900) 

s_sub_u32 s19, s16, s1 // (op 900) 

s_mov_b32 s1, s19 // (op 900) 
s_mul_hi_i32 s49, s1, s12 // (op 900) 
s_add_i32 s50, s49, s1 // (op 900) 


s_ashr_i32 s1, s50, 31 // (op 900) 
s_cmp_eq_i32 s12, 0 // (op 900) 
s_mov_b32 s49, scc // (op 900) 
s_cmp_eq_i32 s49, 0 // (op 900) 
s_cselect_b32 s51, 0, -1 // (op 900) 

s_and_b32 s49, s48, 31 // (op 900) 
s_lshl_b32 s72, 1, s49 // (op 900) 
s_add_i32 s73, s72, s51 // (op 900) 


s_and_b32 s51, s1, s73 // (op 900) 


s_add_i32 s1, s50, s51 // (op 900) 


s_ashr_i32 s50, s1, s49 // (op 900) 


s_lshl_b32 s1, s48, 24 // (op 900) 
s_ashr_i32 s49, s1, 31 // (op 900) 

s_xor_b32 s1, s50, s49 // (op 900) 

s_sub_i32 s50, s1, s49 // (op 900) 


s_mul_i32 s1, s15, s13 // (op 900) 
s_add_u32 s72, s20, 64 // (op 900) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 900) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 900) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 900) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 900) 

s_sub_i32 s49, s72, s1 // (op 900) 

s_mul_i32 s51, s50, s49 // (op 900) 


s_sub_u32 s49, s19, s51 // (op 900) 


s_add_u32 s19, s49, s1 // (op 900) 


s_cmp_eq_i32 s18, 0 // (op 900) 
s_mov_b32 s1, scc // (op 900) 

s_cmp_eq_i32 s1, 0 // (op 900) 
s_cselect_b32 s18, s19, s0 // (op 900) 



v_lshl_add_u32 v13, s18, 6, v35 // (op 900) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 900) 
v_mov_b32 v15, 0 // (op 900) convert
v_mov_b32 v14, v13 // (op 900) convert

s_lshr_b32 s18, s0, 1 // (op 900) 

v_lshl_add_u32 v13, s18, 8, v12 // (op 900) 


v_mul_lo_u32 v17, v13, s29 // (op 900) most significant: low of low * high
v_mul_hi_u32 v12, v13, s28 // (op 900) most significant: high of low * low
v_mul_lo_u32 v16, v13, s28 // (op 900) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 900) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 900) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 900) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 900) most significant: low of high * low
v_mul_hi_u32 v35, v12, 6 // (op 900) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 900) least significant: low of low * low
v_add_u32 v15, v15, v35 // (op 900) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 900) 

v_mov_b32 v35, v12 // (op 900) convert

s_mul_i32 s1, s33, 6 // (op 902) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 902) least significant
s_mul_hi_u32 s18, s32, 6 // (op 902) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 902) most significant: sum

s_mul_i32 s19, s28, s1 // (op 902) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 902) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 902) least significant
s_mul_hi_u32 s50, s28, s0 // (op 902) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 902) most significant: sum
s_add_u32 s19, s19, s50 // (op 902) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 902) 

s_mov_b32 s24, s30 // (op 902) reg expression
s_mov_b32 s25, s31 // (op 902) reg expression
s_mov_b32 s26, s0 // (op 902) reg expression
s_mov_b32 s27, 131072 // (op 902) call()

s_mul_hi_i32 s0, s16, s8 // (op 968) 
v_lshrrev_b32 v12, 3, v1 // (op 968) 
s_add_i32 s1, s0, s16 // (op 968) 

v_and_b32 v13, -8, v1 // (op 968) 
s_ashr_i32 s0, s1, 31 // (op 968) 
v_sub_u32 v14, v1, v13 // (op 968) 

s_cmp_eq_i32 s8, 0 // (op 968) 
s_mov_b32 s18, scc // (op 968) 
v_lshlrev_b32 v13, 4, v14 // (op 968) 

s_cmp_eq_i32 s18, 0 // (op 968) 
s_cselect_b32 s19, 0, -1 // (op 968) 

s_and_b32 s18, s9, 31 // (op 968) 
s_lshl_b32 s49, 1, s18 // (op 968) 
s_add_i32 s50, s49, s19 // (op 968) 


s_and_b32 s19, s0, s50 // (op 968) 


s_add_i32 s0, s1, s19 // (op 968) 


s_ashr_i32 s1, s0, s18 // (op 968) 


s_lshl_b32 s0, s9, 24 // (op 968) 
s_ashr_i32 s18, s0, 31 // (op 968) 

s_xor_b32 s0, s1, s18 // (op 968) 

s_sub_i32 s1, s0, s18 // (op 968) 


s_mul_hi_i32 s0, s1, s10 // (op 968) 
s_add_i32 s18, s0, s1 // (op 968) 

s_ashr_i32 s0, s18, 31 // (op 968) 
s_cmp_eq_i32 s10, 0 // (op 968) 
s_mov_b32 s19, scc // (op 968) 
s_cmp_eq_i32 s19, 0 // (op 968) 
s_cselect_b32 s49, 0, -1 // (op 968) 

s_and_b32 s19, s11, 31 // (op 968) 
s_lshl_b32 s50, 1, s19 // (op 968) 
s_add_i32 s51, s50, s49 // (op 968) 


s_and_b32 s49, s0, s51 // (op 968) 


s_add_i32 s0, s18, s49 // (op 968) 


s_ashr_i32 s18, s0, s19 // (op 968) 


s_lshl_b32 s0, s11, 24 // (op 968) 
s_ashr_i32 s19, s0, 31 // (op 968) 

s_xor_b32 s0, s18, s19 // (op 968) 

s_sub_i32 s18, s0, s19 // (op 968) 


s_add_u32 s50, s4, 32 // (op 968) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 968) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 968) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 968) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 968) 

s_mul_i32 s0, s18, s50 // (op 968) 

s_sub_i32 s49, s1, s0 // (op 968) 


s_mul_hi_i32 s0, s16, s7 // (op 968) 
s_add_i32 s1, s0, s16 // (op 968) 

s_ashr_i32 s0, s1, 31 // (op 968) 
s_cmp_eq_i32 s7, 0 // (op 968) 
s_mov_b32 s18, scc // (op 968) 
s_cmp_eq_i32 s18, 0 // (op 968) 
s_cselect_b32 s19, 0, -1 // (op 968) 

s_and_b32 s18, s14, 31 // (op 968) 
s_lshl_b32 s72, 1, s18 // (op 968) 
s_add_i32 s73, s72, s19 // (op 968) 


s_and_b32 s19, s0, s73 // (op 968) 


s_add_i32 s0, s1, s19 // (op 968) 


s_ashr_i32 s1, s0, s18 // (op 968) 


s_lshl_b32 s0, s14, 24 // (op 968) 
s_ashr_i32 s18, s0, 31 // (op 968) 

s_xor_b32 s0, s1, s18 // (op 968) 

s_sub_i32 s1, s0, s18 // (op 968) 


s_mul_i32 s0, s13, s50 // (op 968) 

s_mul_i32 s18, s6, s0 // (op 968) 

s_mul_i32 s0, s1, s18 // (op 968) 

s_sub_u32 s18, s16, s0 // (op 968) 

s_mul_hi_i32 s0, s18, s12 // (op 968) 
s_add_i32 s19, s0, s18 // (op 968) 


s_ashr_i32 s0, s19, 31 // (op 968) 
s_cmp_eq_i32 s12, 0 // (op 968) 
s_mov_b32 s18, scc // (op 968) 
s_cmp_eq_i32 s18, 0 // (op 968) 
s_cselect_b32 s50, 0, -1 // (op 968) 

s_and_b32 s18, s48, 31 // (op 968) 
s_lshl_b32 s51, 1, s18 // (op 968) 
s_add_i32 s72, s51, s50 // (op 968) 


s_and_b32 s50, s0, s72 // (op 968) 


s_add_i32 s0, s19, s50 // (op 968) 


s_ashr_i32 s19, s0, s18 // (op 968) 


s_lshl_b32 s0, s48, 24 // (op 968) 
s_ashr_i32 s18, s0, 31 // (op 968) 

s_xor_b32 s0, s19, s18 // (op 968) 

s_sub_i32 s19, s0, s18 // (op 968) 


s_cmp_eq_i32 s1, 0 // (op 968) 
s_mov_b32 s0, scc // (op 968) 

s_cmp_eq_i32 s0, 0 // (op 968) 
s_cselect_b32 s1, s19, s49 // (op 968) 



v_lshl_add_u32 v14, s1, 5, v12 // (op 968) 


v_mul_lo_u32 v25, v14, s35 // (op 968) most significant: low of low * high
v_mul_hi_u32 v12, v14, s34 // (op 968) most significant: high of low * low
v_mul_lo_u32 v24, v14, s34 // (op 968) least significant: low of low * low
v_add_u32 v25, v25, v12 // (op 968) most significant: sum


s_ashr_i64 s[0:1], s[32:33], 7 // (op 968) 
s_lshr_b32 s18, s0, 1 // (op 968) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 968) 


v_mov_b32 v27, 0 // (op 968) convert
v_mov_b32 v26, v12 // (op 968) convert

v_add_co_u32 v12, s[0:1], v26, v24 // (op 968) least significant half
v_addc_co_u32 v13, s[0:1], v27, v25, s[0:1] // (op 968) most significant half



s_mul_i32 s1, s34, s5 // (op 970) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 970) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 970) least significant
s_mul_hi_u32 s19, s34, s4 // (op 970) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 970) most significant: sum
s_add_u32 s1, s1, s19 // (op 970) most significant: sum


s_mov_b32 s52, s36 // (op 970) reg expression
s_mov_b32 s53, s37 // (op 970) reg expression
s_mov_b32 s54, s0 // (op 970) reg expression
s_mov_b32 s55, 131072 // (op 970) call()

s_mul_hi_i32 s0, s16, s8 // (op 1032) 
v_and_b32 v14, -64, v1 // (op 1032) 
s_add_i32 s1, s0, s16 // (op 1032) 

v_sub_u32 v15, v1, v14 // (op 1032) 

s_ashr_i32 s0, s1, 31 // (op 1032) 
v_lshrrev_b32 v14, 6, v1 // (op 1032) 
s_cmp_eq_i32 s8, 0 // (op 1032) 
s_mov_b32 s18, scc // (op 1032) 
s_cmp_eq_i32 s18, 0 // (op 1032) 
s_cselect_b32 s19, 0, -1 // (op 1032) 

s_and_b32 s18, s9, 31 // (op 1032) 
s_lshl_b32 s49, 1, s18 // (op 1032) 
s_add_i32 s50, s49, s19 // (op 1032) 


s_and_b32 s19, s0, s50 // (op 1032) 


s_add_i32 s0, s1, s19 // (op 1032) 


s_ashr_i32 s1, s0, s18 // (op 1032) 


s_lshl_b32 s0, s9, 24 // (op 1032) 
s_ashr_i32 s18, s0, 31 // (op 1032) 

s_xor_b32 s0, s1, s18 // (op 1032) 

s_sub_i32 s1, s0, s18 // (op 1032) 


s_mul_hi_i32 s0, s1, s10 // (op 1032) 
s_add_i32 s18, s0, s1 // (op 1032) 

s_ashr_i32 s0, s18, 31 // (op 1032) 
s_cmp_eq_i32 s10, 0 // (op 1032) 
s_mov_b32 s19, scc // (op 1032) 
s_cmp_eq_i32 s19, 0 // (op 1032) 
s_cselect_b32 s49, 0, -1 // (op 1032) 

s_and_b32 s19, s11, 31 // (op 1032) 
s_lshl_b32 s50, 1, s19 // (op 1032) 
s_add_i32 s51, s50, s49 // (op 1032) 


s_and_b32 s49, s0, s51 // (op 1032) 


s_add_i32 s0, s18, s49 // (op 1032) 


s_ashr_i32 s18, s0, s19 // (op 1032) 


s_lshl_b32 s0, s11, 24 // (op 1032) 
s_ashr_i32 s19, s0, 31 // (op 1032) 

s_xor_b32 s0, s18, s19 // (op 1032) 

s_sub_i32 s18, s0, s19 // (op 1032) 


s_mul_i32 s0, s1, s13 // (op 1032) 

s_sub_u32 s1, s16, s0 // (op 1032) 

s_mul_i32 s0, s18, s13 // (op 1032) 
s_add_u32 s0, s0, s1 // (op 1032) 


s_mul_hi_i32 s1, s16, s7 // (op 1032) 
s_add_i32 s18, s1, s16 // (op 1032) 

s_ashr_i32 s1, s18, 31 // (op 1032) 
s_cmp_eq_i32 s7, 0 // (op 1032) 
s_mov_b32 s19, scc // (op 1032) 
s_cmp_eq_i32 s19, 0 // (op 1032) 
s_cselect_b32 s49, 0, -1 // (op 1032) 

s_and_b32 s19, s14, 31 // (op 1032) 
s_lshl_b32 s50, 1, s19 // (op 1032) 
s_add_i32 s51, s50, s49 // (op 1032) 


s_and_b32 s49, s1, s51 // (op 1032) 


s_add_i32 s1, s18, s49 // (op 1032) 


s_ashr_i32 s18, s1, s19 // (op 1032) 


s_lshl_b32 s1, s14, 24 // (op 1032) 
s_ashr_i32 s19, s1, 31 // (op 1032) 

s_xor_b32 s1, s18, s19 // (op 1032) 

s_sub_i32 s18, s1, s19 // (op 1032) 


s_add_u32 s50, s4, 32 // (op 1032) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1032) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1032) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1032) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1032) 

s_mul_i32 s1, s13, s50 // (op 1032) 

s_mul_i32 s19, s6, s1 // (op 1032) 

s_mul_i32 s1, s18, s19 // (op 1032) 

s_sub_u32 s19, s16, s1 // (op 1032) 

s_mov_b32 s1, s19 // (op 1032) 
s_mul_hi_i32 s49, s1, s12 // (op 1032) 
s_add_i32 s50, s49, s1 // (op 1032) 


s_ashr_i32 s1, s50, 31 // (op 1032) 
s_cmp_eq_i32 s12, 0 // (op 1032) 
s_mov_b32 s49, scc // (op 1032) 
s_cmp_eq_i32 s49, 0 // (op 1032) 
s_cselect_b32 s51, 0, -1 // (op 1032) 

s_and_b32 s49, s48, 31 // (op 1032) 
s_lshl_b32 s72, 1, s49 // (op 1032) 
s_add_i32 s73, s72, s51 // (op 1032) 


s_and_b32 s51, s1, s73 // (op 1032) 


s_add_i32 s1, s50, s51 // (op 1032) 


s_ashr_i32 s50, s1, s49 // (op 1032) 


s_lshl_b32 s1, s48, 24 // (op 1032) 
s_ashr_i32 s49, s1, 31 // (op 1032) 

s_xor_b32 s1, s50, s49 // (op 1032) 

s_sub_i32 s50, s1, s49 // (op 1032) 


s_mul_i32 s1, s15, s13 // (op 1032) 
s_add_u32 s72, s20, 64 // (op 1032) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 1032) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1032) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1032) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 1032) 

s_sub_i32 s49, s72, s1 // (op 1032) 

s_mul_i32 s51, s50, s49 // (op 1032) 


s_sub_u32 s49, s19, s51 // (op 1032) 


s_add_u32 s19, s49, s1 // (op 1032) 


s_cmp_eq_i32 s18, 0 // (op 1032) 
s_mov_b32 s1, scc // (op 1032) 

s_cmp_eq_i32 s1, 0 // (op 1032) 
s_cselect_b32 s18, s19, s0 // (op 1032) 



v_lshl_add_u32 v16, s18, 6, v15 // (op 1032) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1032) 
s_lshr_b32 s18, s0, 1 // (op 1032) 

v_lshl_add_u32 v15, s18, 3, v14 // (op 1032) 


v_mul_lo_u32 v14, v15, s38 // (op 1032) 

v_add_u32 v15, v16, v14 // (op 1032) 


s_mul_i32 s1, s38, s43 // (op 1034) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1034) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1034) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1034) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1034) most significant: sum
s_add_u32 s1, s1, s19 // (op 1034) most significant: sum


s_mov_b32 s64, s40 // (op 1034) reg expression
s_mov_b32 s65, s41 // (op 1034) reg expression
s_mov_b32 s66, s0 // (op 1034) reg expression
s_mov_b32 s67, 131072 // (op 1034) call()

s_mul_hi_i32 s0, s16, s8 // (op 1096) 
v_lshrrev_b32 v14, 6, v1 // (op 1096) 
s_add_i32 s1, s0, s16 // (op 1096) 

v_and_b32 v16, 1, v14 // (op 1096) 

s_ashr_i32 s0, s1, 31 // (op 1096) 
v_and_b32 v14, 15, v1 // (op 1096) 
s_cmp_eq_i32 s8, 0 // (op 1096) 
s_mov_b32 s18, scc // (op 1096) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 1096) 


s_cmp_eq_i32 s18, 0 // (op 1096) 
s_cselect_b32 s19, 0, -1 // (op 1096) 

v_and_b32 v14, 63, v1 // (op 1096) 
s_and_b32 s18, s9, 31 // (op 1096) 
v_lshrrev_b32 v16, 4, v14 // (op 1096) 

s_lshl_b32 s49, 1, s18 // (op 1096) 
s_add_i32 s50, s49, s19 // (op 1096) 


s_and_b32 s19, s0, s50 // (op 1096) 


s_add_i32 s0, s1, s19 // (op 1096) 


s_ashr_i32 s1, s0, s18 // (op 1096) 


s_lshl_b32 s0, s9, 24 // (op 1096) 
s_ashr_i32 s18, s0, 31 // (op 1096) 

s_xor_b32 s0, s1, s18 // (op 1096) 

s_sub_i32 s1, s0, s18 // (op 1096) 


s_mul_hi_i32 s0, s1, s10 // (op 1096) 
s_add_i32 s18, s0, s1 // (op 1096) 

s_ashr_i32 s0, s18, 31 // (op 1096) 
s_cmp_eq_i32 s10, 0 // (op 1096) 
s_mov_b32 s19, scc // (op 1096) 
s_cmp_eq_i32 s19, 0 // (op 1096) 
s_cselect_b32 s49, 0, -1 // (op 1096) 

s_and_b32 s19, s11, 31 // (op 1096) 
s_lshl_b32 s50, 1, s19 // (op 1096) 
s_add_i32 s51, s50, s49 // (op 1096) 


s_and_b32 s49, s0, s51 // (op 1096) 


s_add_i32 s0, s18, s49 // (op 1096) 


s_ashr_i32 s18, s0, s19 // (op 1096) 


s_lshl_b32 s0, s11, 24 // (op 1096) 
s_ashr_i32 s19, s0, 31 // (op 1096) 

s_xor_b32 s0, s18, s19 // (op 1096) 

s_sub_i32 s18, s0, s19 // (op 1096) 


s_add_u32 s50, s4, 32 // (op 1096) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1096) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1096) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1096) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1096) 

s_mul_i32 s0, s18, s50 // (op 1096) 

s_sub_i32 s49, s1, s0 // (op 1096) 


s_mul_hi_i32 s0, s16, s7 // (op 1096) 
s_add_i32 s1, s0, s16 // (op 1096) 

s_ashr_i32 s0, s1, 31 // (op 1096) 
s_cmp_eq_i32 s7, 0 // (op 1096) 
s_mov_b32 s18, scc // (op 1096) 
s_cmp_eq_i32 s18, 0 // (op 1096) 
s_cselect_b32 s19, 0, -1 // (op 1096) 

s_and_b32 s18, s14, 31 // (op 1096) 
s_lshl_b32 s72, 1, s18 // (op 1096) 
s_add_i32 s73, s72, s19 // (op 1096) 


s_and_b32 s19, s0, s73 // (op 1096) 


s_add_i32 s0, s1, s19 // (op 1096) 


s_ashr_i32 s1, s0, s18 // (op 1096) 


s_lshl_b32 s0, s14, 24 // (op 1096) 
s_ashr_i32 s18, s0, 31 // (op 1096) 

s_xor_b32 s0, s1, s18 // (op 1096) 

s_sub_i32 s1, s0, s18 // (op 1096) 


s_mul_i32 s0, s13, s50 // (op 1096) 

s_mul_i32 s18, s6, s0 // (op 1096) 

s_mul_i32 s0, s1, s18 // (op 1096) 

s_sub_u32 s18, s16, s0 // (op 1096) 

s_mul_hi_i32 s0, s18, s12 // (op 1096) 
s_add_i32 s19, s0, s18 // (op 1096) 


s_ashr_i32 s0, s19, 31 // (op 1096) 
s_cmp_eq_i32 s12, 0 // (op 1096) 
s_mov_b32 s18, scc // (op 1096) 
s_cmp_eq_i32 s18, 0 // (op 1096) 
s_cselect_b32 s50, 0, -1 // (op 1096) 

s_and_b32 s18, s48, 31 // (op 1096) 
s_lshl_b32 s51, 1, s18 // (op 1096) 
s_add_i32 s72, s51, s50 // (op 1096) 


s_and_b32 s50, s0, s72 // (op 1096) 


s_add_i32 s0, s19, s50 // (op 1096) 


s_ashr_i32 s19, s0, s18 // (op 1096) 


s_lshl_b32 s0, s48, 24 // (op 1096) 
s_ashr_i32 s18, s0, 31 // (op 1096) 

s_xor_b32 s0, s19, s18 // (op 1096) 

s_sub_i32 s19, s0, s18 // (op 1096) 


s_cmp_eq_i32 s1, 0 // (op 1096) 
s_mov_b32 s0, scc // (op 1096) 

s_cmp_eq_i32 s0, 0 // (op 1096) 
s_cselect_b32 s1, s19, s49 // (op 1096) 



v_lshl_add_u32 v14, s1, 5, v17 // (op 1096) 


v_mul_lo_u32 v17, v14, s44 // (op 1096) 

s_ashr_i64 s[0:1], s[32:33], 7 // (op 1096) 
s_lshr_b32 s18, s0, 1 // (op 1096) 

v_lshl_add_u32 v14, s18, 3, v16 // (op 1096) 


v_add_u32 v16, v14, v17 // (op 1096) 


s_mul_i32 s1, s44, s5 // (op 1098) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 1098) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 1098) least significant
s_mul_hi_u32 s19, s44, s4 // (op 1098) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1098) most significant: sum
s_add_u32 s1, s1, s19 // (op 1098) most significant: sum


s_mov_b32 s68, s46 // (op 1098) reg expression
s_mov_b32 s69, s47 // (op 1098) reg expression
s_mov_b32 s70, s0 // (op 1098) reg expression
s_mov_b32 s71, 131072 // (op 1098) call()

v_and_b32 v14, -2, v1 // (op 1189) 
v_sub_u32 v17, v1, v14 // (op 1189) 

v_lshlrev_b32 v14, 5, v17 // (op 1189) 

v_lshrrev_b32 v17, 1, v1 // (op 1189) 
v_lshl_add_u32 v18, v17, 6, v14 // (op 1189) 


v_mul_lo_u32 v14, v18, 6 // (op 1189) 
v_lshrrev_b32 v17, 3, v14 // (op 1189) 

v_lshrrev_b32 v14, 4, v18 // (op 1189) 

v_lshlrev_b32 v18, 2, v14 // (op 1189) 

v_add_u32 v14, v17, v18 // (op 1189) 


v_and_b32 v17, -8, v1 // (op 1247) 
v_sub_u32 v18, v1, v17 // (op 1247) 

v_lshlrev_b32 v17, 4, v18 // (op 1247) 

v_lshrrev_b32 v18, 3, v1 // (op 1247) 
v_lshl_add_u32 v19, v18, 7, v17 // (op 1247) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 1247) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -64, v1 // (op 1299) 
v_sub_u32 v18, v1, v17 // (op 1299) 

v_and_b32 v17, 4294967232, v1 // (op 1299) 
v_add_u32 v19, v17, v18 // (op 1299) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 316) 
s_and_b32 s1, s18, 4294967294 // (op 316) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 315) 
s_mov_b32 s49, s18 // (op 315) convert

s_cmp_lt_u32 s1, s49 // (op 315) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_315 // (op 315) Condition: Top (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_315 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_315:


s_add_u32 s1, s1, 1 // (op 317) 
v_mov_b32 v17, v35 // (op 166) 
v_mov_b32 v18, v17 // (op 166) reg expression

buffer_load_dwordx3 v[20:22], v18, s[24:27], 0 offen // (op 166) Load value
buffer_load_dwordx3 v[24:26], v18, s[24:27], 0 offen offset:12 // (op 166) Load value


s_lshl_b64 s[18:19], s[28:29], 7 // (op 890) 
s_mul_i32 s51, s19, 6 // (op 890) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 890) least significant
s_mul_hi_u32 s49, s18, 6 // (op 890) most significant: high of low * low
s_add_u32 s51, s51, s49 // (op 890) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 890) 

v_add_u32 v35, v35, s18 // (op 890) 

s_barrier  // (op 1494) 
s_barrier  // (op 1507) 





v_mov_b64 v[28:29], v[12:13] // (op 870) 

s_mov_b32 m0, s0 // (op 870) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 870) Load value direct to lds

v_add_co_u32 v12, vcc, 128, v12 // (op 960) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 960) most significant half
s_waitcnt vmcnt(2)
ds_write_b96 v14, v[20:22] offset:8192 // (op 167) Store local data 


v_mov_b32 v23, v15 // (op 180) 
v_mov_b32 v17, v23 // (op 180) reg expression

buffer_load_ubyte v18, v17, s[64:67], 0 offen offset: 0 // (op 180) Load value


s_lshl_b32 s18, s38, 2 // (op 1024) 
v_add_u32 v15, v15, s18 // (op 1024) 

s_waitcnt vmcnt(2)
ds_write_b96 v14, v[24:26] offset:8208 // (op 167) Store local data 

s_barrier  // (op 1520) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1497) 






ds_read_b96_tr_b6 v[20:22], v0 offset:8192 // (op 169) Transpose load local data 
ds_read_b96_tr_b6 v[23:25], v0 offset:9216 // (op 169) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v18 offset:16384 // (op 178) Store local data 

s_barrier  // (op 1510) 



ds_read_b128 v[26:29], v3 // (op 171) Load local data 
ds_read_b128 v[30:33], v3 offset:64 // (op 171) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1524) 


ds_read_u8 v34, v7 offset:16384 // (op 173) Load local data 


v_mov_b32 v17, v16 // (op 175) 

buffer_load_ubyte v18, v17, s[68:71], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v16, 4, v16 // (op 1088) 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:33], a[0:3], v34, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 176) 
ds_read_b96_tr_b6 v[36:38], v0 offset:8224 // (op 321) Transpose load local data 


ds_read_b96_tr_b6 v[39:41], v0 offset:9248 // (op 321) Transpose load local data 


ds_read_u8 v17, v7 offset:16416 // (op 325) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[26:33], a[4:7], v17, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 327) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 315) 
s_mov_b32 s49, s18 // (op 315) convert

s_cmp_lt_u32 s1, s49 // (op 315) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_315 // (op 315) Condition: Bottom (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_315 if true)
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_315:






















s_barrier  // (op 1592) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_123
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen // (op 52) Store value

v_accvgpr_read v10, a4 // (op 513) 
v_accvgpr_read v11, a5 // (op 513) 
v_accvgpr_read v12, a6 // (op 513) 
v_accvgpr_read v13, a7 // (op 513) 
v_mul_f32 v14, v8, v10 // (op 513) 
v_mul_f32 v15, v8, v11 // (op 513) 
v_mul_f32 v16, v8, v12 // (op 513) 
v_mul_f32 v17, v8, v13 // (op 513) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:128 // (op 515) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:128 // (op 543) Load value
v_accvgpr_read v14, a4 // (op 540) 
v_accvgpr_read v15, a5 // (op 540) 
v_accvgpr_read v16, a6 // (op 540) 
v_accvgpr_read v17, a7 // (op 540) 
v_mul_f32 v18, v8, v14 // (op 540) 
v_mul_f32 v19, v8, v15 // (op 540) 
v_mul_f32 v20, v8, v16 // (op 540) 
v_mul_f32 v21, v8, v17 // (op 540) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 544) 
v_mul_f32 v15, v5, v11 // (op 544) 
v_mul_f32 v16, v5, v12 // (op 544) 
v_mul_f32 v17, v5, v13 // (op 544) 

v_add_f32 v10, v14, v18 // (op 541) 
v_add_f32 v11, v15, v19 // (op 541) 
v_add_f32 v12, v16, v20 // (op 541) 
v_add_f32 v13, v17, v21 // (op 541) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:128 // (op 542) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen // (op 135) Load value
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





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen // (op 134) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.LRR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__end:
.size RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_, .LRR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__end-RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: B8
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
  - .name: RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 25088
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 64
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

