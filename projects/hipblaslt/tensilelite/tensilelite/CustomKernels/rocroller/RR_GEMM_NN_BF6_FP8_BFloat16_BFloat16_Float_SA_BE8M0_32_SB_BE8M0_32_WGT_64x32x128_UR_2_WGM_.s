// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_,@function
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s8 // (op 1168) 
s_mul_hi_i32 s1, s16, s8 // (op 1356) 
v_lshrrev_b32 v0, 6, v1 // (op 1282) 
v_and_b32 v3, 1, v0 // (op 1282) 

v_and_b32 v0, 15, v1 // (op 1282) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 1282) 


v_and_b32 v0, 48, v1 // (op 1282) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 1282) 


v_lshrrev_b32 v0, 6, v1 // (op 1426) 
v_and_b32 v4, 1, v0 // (op 1426) 

v_and_b32 v0, 15, v1 // (op 1426) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 1426) 


v_and_b32 v0, 48, v1 // (op 1426) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 1426) 


v_lshrrev_b32 v0, 7, v1 // (op 1168) 
s_add_i32 s18, s0, s16 // (op 1168) 

v_and_b32 v5, 63, v1 // (op 1168) 
s_ashr_i32 s0, s18, 31 // (op 1168) 
v_lshrrev_b32 v6, 4, v5 // (op 1168) 
s_cmp_eq_i32 s8, 0 // (op 1168) 
s_mov_b32 s19, scc // (op 1168) 
v_lshlrev_b32 v7, 2, v6 // (op 1168) 

s_cmp_eq_i32 s19, 0 // (op 1168) 
s_cselect_b32 s51, 0, -1 // (op 1168) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1168) 


s_and_b32 s19, s9, 31 // (op 1168) 
v_lshrrev_b32 v0, 2, v5 // (op 1168) 

v_and_b32 v5, 3, v0 // (op 1168) 

v_and_b32 v0, 63, v1 // (op 1230) 
v_and_b32 v7, 48, v1 // (op 1230) 
v_sub_u32 v8, v0, v7 // (op 1230) 

v_lshrrev_b32 v7, 4, v0 // (op 1230) 

v_lshl_add_u32 v0, v7, 5, v8 // (op 1230) 


v_lshrrev_b32 v7, 7, v1 // (op 1230) 
v_lshlrev_b32 v8, 4, v7 // (op 1230) 

v_lshl_add_u32 v7, v0, 6, v8 // (op 1230) 


v_mul_lo_u32 v0, v7, 6 // (op 1230) 
v_lshrrev_b32 v8, 3, v0 // (op 1230) 

v_lshrrev_b32 v0, 4, v7 // (op 1230) 

v_lshlrev_b32 v7, 2, v0 // (op 1230) 

v_add_u32 v0, v8, v7 // (op 1230) 


v_lshrrev_b32 v7, 7, v1 // (op 1340) 
v_and_b32 v8, 15, v1 // (op 1340) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 1340) 


v_and_b32 v7, 63, v1 // (op 1340) 
v_lshrrev_b32 v8, 4, v7 // (op 1340) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 1340) 


v_lshrrev_b32 v8, 7, v1 // (op 1356) 
v_and_b32 v9, 63, v1 // (op 1356) 
v_and_b32 v10, 63, v1 // (op 1387) 
v_and_b32 v11, 48, v1 // (op 1387) 
v_sub_u32 v12, v10, v11 // (op 1387) 

v_lshrrev_b32 v11, 4, v10 // (op 1387) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 1387) 


v_lshrrev_b32 v11, 7, v1 // (op 1387) 
v_lshlrev_b32 v12, 4, v11 // (op 1387) 

v_lshl_add_u32 v11, v10, 6, v12 // (op 1387) 


v_mul_lo_u32 v10, v11, 6 // (op 1387) 
v_lshrrev_b32 v12, 3, v10 // (op 1387) 

v_lshrrev_b32 v10, 4, v11 // (op 1387) 

v_lshlrev_b32 v11, 2, v10 // (op 1387) 

v_add_u32 v10, v12, v11 // (op 1387) 


v_lshrrev_b32 v11, 7, v1 // (op 1455) 
v_and_b32 v12, 15, v1 // (op 1455) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 1455) 


v_and_b32 v11, 63, v1 // (op 1455) 
v_lshrrev_b32 v12, 4, v11 // (op 1455) 

v_lshl_add_u32 v11, v12, 6, v13 // (op 1455) 


s_lshl_b32 s52, 1, s19 // (op 1168) 
v_and_b32 v12, 3, v1 // (op 1168) 
s_add_i32 s53, s52, s51 // (op 1168) 


s_and_b32 s51, s0, s53 // (op 1168) 


s_add_i32 s0, s1, s16 // (op 1356) 

s_ashr_i32 s1, s0, 31 // (op 1356) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1168) 


s_add_i32 s52, s18, s51 // (op 1168) 


v_lshrrev_b32 v5, 6, v1 // (op 1168) 
s_ashr_i32 s18, s52, s19 // (op 1168) 


v_and_b32 v12, 1, v5 // (op 1168) 

s_lshl_b32 s19, s9, 24 // (op 1168) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1168) 


s_ashr_i32 s51, s19, 31 // (op 1168) 

s_xor_b32 s19, s18, s51 // (op 1168) 

s_sub_i32 s18, s19, s51 // (op 1168) 


s_mul_hi_i32 s19, s18, s10 // (op 1168) 
s_add_i32 s51, s19, s18 // (op 1168) 

s_ashr_i32 s19, s51, 31 // (op 1168) 
s_cmp_eq_i32 s10, 0 // (op 1168) 
s_mov_b32 s52, scc // (op 1168) 
s_cmp_eq_i32 s52, 0 // (op 1168) 
s_cselect_b32 s53, 0, -1 // (op 1168) 

s_and_b32 s52, s11, 31 // (op 1168) 
v_lshrrev_b32 v12, 4, v9 // (op 1356) 
s_cmp_eq_i32 s8, 0 // (op 1356) 
s_mov_b32 s54, scc // (op 1356) 
v_lshlrev_b32 v13, 2, v12 // (op 1356) 

s_cmp_eq_i32 s54, 0 // (op 1356) 
s_cselect_b32 s55, 0, -1 // (op 1356) 

s_lshl_b32 s54, 1, s52 // (op 1168) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 1356) 


v_lshrrev_b32 v8, 2, v9 // (op 1356) 

s_add_i32 s56, s54, s53 // (op 1168) 


s_and_b32 s53, s19, s56 // (op 1168) 


s_add_i32 s19, s51, s53 // (op 1168) 


s_ashr_i32 s51, s19, s52 // (op 1168) 


s_lshl_b32 s19, s11, 24 // (op 1168) 
s_ashr_i32 s52, s19, 31 // (op 1168) 

s_xor_b32 s19, s51, s52 // (op 1168) 

s_sub_i32 s51, s19, s52 // (op 1168) 


s_mul_i32 s19, s18, s13 // (op 1168) 
s_sub_u32 s52, s16, s19 // (op 1168) 

s_mul_i32 s19, s51, s13 // (op 1168) 
s_add_u32 s19, s19, s52 // (op 1168) 

s_mul_hi_i32 s52, s16, s7 // (op 1168) 
s_add_i32 s53, s52, s16 // (op 1168) 

s_ashr_i32 s52, s53, 31 // (op 1168) 
s_cmp_eq_i32 s7, 0 // (op 1168) 
s_mov_b32 s54, scc // (op 1168) 
s_cmp_eq_i32 s54, 0 // (op 1168) 
s_cselect_b32 s56, 0, -1 // (op 1168) 

s_and_b32 s54, s14, 31 // (op 1168) 
s_lshl_b32 s57, 1, s54 // (op 1168) 
s_add_i32 s58, s57, s56 // (op 1168) 


s_and_b32 s56, s52, s58 // (op 1168) 


s_add_i32 s52, s53, s56 // (op 1168) 


s_ashr_i32 s53, s52, s54 // (op 1168) 


s_lshl_b32 s52, s14, 24 // (op 1168) 
s_ashr_i32 s54, s52, 31 // (op 1168) 

s_xor_b32 s52, s53, s54 // (op 1168) 

s_sub_i32 s53, s52, s54 // (op 1168) 


s_and_b32 s52, s9, 31 // (op 1356) 
s_lshl_b32 s54, 1, s52 // (op 1356) 
v_and_b32 v9, 3, v8 // (op 1356) 

s_add_i32 s56, s54, s55 // (op 1356) 


v_and_b32 v8, 3, v1 // (op 1356) 
s_and_b32 s54, s1, s56 // (op 1356) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1356) 


s_add_i32 s1, s0, s54 // (op 1356) 


v_lshrrev_b32 v8, 6, v1 // (op 1356) 
s_ashr_i32 s0, s1, s52 // (op 1356) 


v_and_b32 v9, 1, v8 // (op 1356) 

s_lshl_b32 s1, s9, 24 // (op 1356) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1356) 


s_ashr_i32 s52, s1, 31 // (op 1356) 

s_xor_b32 s1, s0, s52 // (op 1356) 

s_sub_i32 s0, s1, s52 // (op 1356) 


s_mul_hi_i32 s1, s0, s10 // (op 1356) 
s_add_i32 s52, s1, s0 // (op 1356) 

s_ashr_i32 s1, s52, 31 // (op 1356) 
s_cmp_eq_i32 s10, 0 // (op 1356) 
s_mov_b32 s54, scc // (op 1356) 
s_cmp_eq_i32 s54, 0 // (op 1356) 
s_cselect_b32 s55, 0, -1 // (op 1356) 

s_and_b32 s54, s11, 31 // (op 1356) 
s_lshl_b32 s56, 1, s54 // (op 1356) 
s_add_i32 s57, s56, s55 // (op 1356) 


s_and_b32 s55, s1, s57 // (op 1356) 


s_add_i32 s1, s52, s55 // (op 1356) 


s_ashr_i32 s52, s1, s54 // (op 1356) 


s_lshl_b32 s1, s11, 24 // (op 1356) 
s_ashr_i32 s54, s1, 31 // (op 1356) 

s_xor_b32 s1, s52, s54 // (op 1356) 

s_sub_i32 s52, s1, s54 // (op 1356) 


s_mul_i32 s1, s0, s13 // (op 1356) 
s_sub_u32 s54, s16, s1 // (op 1356) 

s_mul_i32 s1, s52, s13 // (op 1356) 
s_add_u32 s1, s1, s54 // (op 1356) 

s_mul_hi_i32 s54, s16, s7 // (op 1356) 
s_add_i32 s55, s54, s16 // (op 1356) 

s_ashr_i32 s54, s55, 31 // (op 1356) 
s_cmp_eq_i32 s7, 0 // (op 1356) 
s_mov_b32 s56, scc // (op 1356) 
s_cmp_eq_i32 s56, 0 // (op 1356) 
s_cselect_b32 s57, 0, -1 // (op 1356) 

s_and_b32 s56, s14, 31 // (op 1356) 
s_lshl_b32 s58, 1, s56 // (op 1356) 
s_add_i32 s59, s58, s57 // (op 1356) 


s_and_b32 s57, s54, s59 // (op 1356) 


s_add_i32 s54, s55, s57 // (op 1356) 


s_ashr_i32 s55, s54, s56 // (op 1356) 


s_lshl_b32 s54, s14, 24 // (op 1356) 
s_ashr_i32 s56, s54, 31 // (op 1356) 

s_xor_b32 s54, s55, s56 // (op 1356) 

s_sub_i32 s55, s54, s56 // (op 1356) 


s_add_u32 s56, s4, 32 // (op 1168) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 1168) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1168) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1168) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 1168) 

s_mul_i32 s54, s13, s56 // (op 1168) 
s_mul_i32 s58, s6, s54 // (op 1168) 

s_mul_i32 s54, s53, s58 // (op 1168) 

s_sub_u32 s58, s16, s54 // (op 1168) 

s_mov_b32 s54, s58 // (op 1168) 
s_mul_hi_i32 s59, s54, s12 // (op 1168) 
s_add_i32 s60, s59, s54 // (op 1168) 


s_ashr_i32 s54, s60, 31 // (op 1168) 
s_cmp_eq_i32 s12, 0 // (op 1168) 
s_mov_b32 s59, scc // (op 1168) 
s_cmp_eq_i32 s59, 0 // (op 1168) 
s_cselect_b32 s61, 0, -1 // (op 1168) 

s_add_u32 s62, s4, 32 // (op 1356) Start of Int64 add, locking SCC
s_addc_u32 s63, s5, 0 // (op 1356) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1356) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1356) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 5 // (op 1356) 

s_mul_i32 s59, s13, s62 // (op 1356) 
s_mul_i32 s64, s6, s59 // (op 1356) 

s_mul_i32 s59, s55, s64 // (op 1356) 

s_sub_u32 s64, s16, s59 // (op 1356) 

s_mov_b32 s59, s64 // (op 1356) 
s_mul_hi_i32 s65, s59, s12 // (op 1356) 
s_add_i32 s66, s65, s59 // (op 1356) 


s_ashr_i32 s59, s66, 31 // (op 1356) 
s_cmp_eq_i32 s12, 0 // (op 1356) 
s_mov_b32 s65, scc // (op 1356) 
s_cmp_eq_i32 s65, 0 // (op 1356) 
s_cselect_b32 s67, 0, -1 // (op 1356) 


s_waitcnt lgkmcnt(0)
s_and_b32 s65, s48, 31 // (op 1168) 
s_lshl_b32 s68, 1, s65 // (op 1168) 
s_add_i32 s69, s68, s61 // (op 1168) 


s_and_b32 s61, s54, s69 // (op 1168) 


s_add_i32 s54, s60, s61 // (op 1168) 


s_ashr_i32 s60, s54, s65 // (op 1168) 


s_lshl_b32 s54, s48, 24 // (op 1168) 
s_ashr_i32 s61, s54, 31 // (op 1168) 

s_xor_b32 s54, s60, s61 // (op 1168) 

s_sub_i32 s60, s54, s61 // (op 1168) 


s_mul_i32 s54, s15, s13 // (op 1168) 
s_and_b32 s61, s48, 31 // (op 1356) 
s_lshl_b32 s65, 1, s61 // (op 1356) 
s_add_i32 s68, s65, s67 // (op 1356) 


s_and_b32 s65, s59, s68 // (op 1356) 


s_add_i32 s59, s66, s65 // (op 1356) 


s_ashr_i32 s65, s59, s61 // (op 1356) 


s_lshl_b32 s59, s48, 24 // (op 1356) 
s_ashr_i32 s61, s59, 31 // (op 1356) 

s_xor_b32 s59, s65, s61 // (op 1356) 

s_sub_i32 s65, s59, s61 // (op 1356) 


s_mul_i32 s59, s15, s13 // (op 1356) 
s_add_u32 s66, s20, 64 // (op 1168) Start of Int64 add, locking SCC
s_addc_u32 s67, s21, 0 // (op 1168) End of Int64 add, unlocking SCC
s_sub_u32 s68, s66, 1 // (op 1168) least significant half; sets scc
s_subb_u32 s69, s67, 0 // (op 1168) most significant half; uses scc

s_ashr_i64 s[66:67], s[68:69], 6 // (op 1168) 

s_sub_i32 s61, s66, s54 // (op 1168) 

s_mul_i32 s66, s60, s61 // (op 1168) 

s_sub_u32 s61, s58, s66 // (op 1168) 


s_add_u32 s58, s61, s54 // (op 1168) 


s_cmp_eq_i32 s53, 0 // (op 1168) 
s_mov_b32 s54, scc // (op 1168) 

s_cmp_eq_i32 s54, 0 // (op 1168) 
s_cselect_b32 s53, s58, s19 // (op 1168) 


s_mul_i32 s19, s51, s56 // (op 1168) 


v_lshl_add_u32 v9, s53, 6, v6 // (op 1168) 


s_sub_i32 s51, s18, s19 // (op 1168) 


s_cmp_eq_i32 s54, 0 // (op 1168) 
s_cselect_b32 s53, s60, s51 // (op 1168) 



v_lshl_add_u32 v6, s53, 5, v5 // (op 1168) 


v_mul_lo_u32 v5, v6, s2 // (op 1168) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 1168) 


s_add_u32 s18, s20, 64 // (op 1356) Start of Int64 add, locking SCC
s_addc_u32 s19, s21, 0 // (op 1356) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 1356) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 1356) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 6 // (op 1356) 

s_sub_i32 s51, s18, s59 // (op 1356) 

s_mul_i32 s18, s65, s51 // (op 1356) 

s_sub_u32 s19, s64, s18 // (op 1356) 


s_add_u32 s18, s19, s59 // (op 1356) 


s_cmp_eq_i32 s55, 0 // (op 1356) 
s_mov_b32 s19, scc // (op 1356) 

s_cmp_eq_i32 s19, 0 // (op 1356) 
s_cselect_b32 s51, s18, s1 // (op 1356) 


s_mul_i32 s1, s52, s62 // (op 1356) 


v_lshl_add_u32 v5, s51, 6, v12 // (op 1356) 


s_sub_i32 s18, s0, s1 // (op 1356) 


s_cmp_eq_i32 s19, 0 // (op 1356) 
s_cselect_b32 s0, s65, s18 // (op 1356) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 1356) 


v_mul_lo_u32 v8, v9, s22 // (op 1356) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 1356) 


s_mul_i32 s1, s2, s5 // (op 1170) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 1170) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 1170) least significant
s_mul_hi_u32 s19, s2, s4 // (op 1170) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1170) most significant: sum
s_add_u32 s1, s1, s19 // (op 1170) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1170) 

s_mul_i32 s1, s22, s5 // (op 1358) most significant: low of low * high
s_mul_i32 s51, s23, s4 // (op 1358) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 1358) least significant
s_mul_hi_u32 s52, s22, s4 // (op 1358) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1358) most significant: sum
s_add_u32 s1, s1, s52 // (op 1358) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 1358) 

s_mov_b32 s56, s24 // (op 1170) reg expression
s_mov_b32 s57, s25 // (op 1170) reg expression
s_mov_b32 s58, s18 // (op 1170) reg expression
s_mov_b32 s59, 131072 // (op 1170) call()

s_mov_b32 s60, s26 // (op 1358) reg expression

s_mov_b32 s61, s27 // (op 1358) reg expression
s_mov_b32 s62, s52 // (op 1358) reg expression
s_mov_b32 s63, 131072 // (op 1358) call()


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
s_mul_hi_i32 s0, s16, s8 // (op 935) 
v_and_b32 v12, -2, v1 // (op 935) 
s_add_i32 s1, s0, s16 // (op 935) 

v_sub_u32 v13, v1, v12 // (op 935) 

s_ashr_i32 s0, s1, 31 // (op 935) 
v_lshlrev_b32 v12, 5, v13 // (op 935) 

s_cmp_eq_i32 s8, 0 // (op 935) 
s_mov_b32 s18, scc // (op 935) 
v_lshrrev_b32 v13, 1, v1 // (op 935) 
s_cmp_eq_i32 s18, 0 // (op 935) 
s_cselect_b32 s19, 0, -1 // (op 935) 

s_and_b32 s18, s9, 31 // (op 935) 
s_lshl_b32 s24, 1, s18 // (op 935) 
s_add_i32 s25, s24, s19 // (op 935) 


s_and_b32 s19, s0, s25 // (op 935) 


s_add_i32 s0, s1, s19 // (op 935) 


s_ashr_i32 s1, s0, s18 // (op 935) 


s_lshl_b32 s0, s9, 24 // (op 935) 
s_ashr_i32 s18, s0, 31 // (op 935) 

s_xor_b32 s0, s1, s18 // (op 935) 

s_sub_i32 s1, s0, s18 // (op 935) 


s_mul_hi_i32 s0, s1, s10 // (op 935) 
s_add_i32 s18, s0, s1 // (op 935) 

s_ashr_i32 s0, s18, 31 // (op 935) 
s_cmp_eq_i32 s10, 0 // (op 935) 
s_mov_b32 s19, scc // (op 935) 
s_cmp_eq_i32 s19, 0 // (op 935) 
s_cselect_b32 s24, 0, -1 // (op 935) 

s_and_b32 s19, s11, 31 // (op 935) 
s_lshl_b32 s25, 1, s19 // (op 935) 
s_add_i32 s26, s25, s24 // (op 935) 


s_and_b32 s27, s0, s26 // (op 935) 


s_add_i32 s0, s18, s27 // (op 935) 


s_ashr_i32 s18, s0, s19 // (op 935) 


s_lshl_b32 s0, s11, 24 // (op 935) 
s_ashr_i32 s19, s0, 31 // (op 935) 

s_xor_b32 s0, s18, s19 // (op 935) 

s_sub_i32 s18, s0, s19 // (op 935) 


s_mul_i32 s0, s1, s13 // (op 935) 

s_sub_u32 s1, s16, s0 // (op 935) 

s_mul_i32 s0, s18, s13 // (op 935) 
s_add_u32 s0, s0, s1 // (op 935) 


s_mul_hi_i32 s1, s16, s7 // (op 935) 
s_add_i32 s18, s1, s16 // (op 935) 

s_ashr_i32 s1, s18, 31 // (op 935) 
s_cmp_eq_i32 s7, 0 // (op 935) 
s_mov_b32 s19, scc // (op 935) 
s_cmp_eq_i32 s19, 0 // (op 935) 
s_cselect_b32 s24, 0, -1 // (op 935) 

s_and_b32 s19, s14, 31 // (op 935) 
s_lshl_b32 s25, 1, s19 // (op 935) 
s_add_i32 s26, s25, s24 // (op 935) 


s_and_b32 s27, s1, s26 // (op 935) 


s_add_i32 s1, s18, s27 // (op 935) 


s_ashr_i32 s18, s1, s19 // (op 935) 


s_lshl_b32 s1, s14, 24 // (op 935) 
s_ashr_i32 s19, s1, 31 // (op 935) 

s_xor_b32 s1, s18, s19 // (op 935) 

s_sub_i32 s18, s1, s19 // (op 935) 


s_add_u32 s24, s4, 32 // (op 935) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 935) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 935) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 935) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 5 // (op 935) 

s_mul_i32 s1, s13, s24 // (op 935) 

s_mul_i32 s19, s6, s1 // (op 935) 

s_mul_i32 s1, s18, s19 // (op 935) 

s_sub_u32 s19, s16, s1 // (op 935) 

s_mov_b32 s1, s19 // (op 935) 
s_mul_hi_i32 s24, s1, s12 // (op 935) 
s_add_i32 s25, s24, s1 // (op 935) 


s_ashr_i32 s1, s25, 31 // (op 935) 
s_cmp_eq_i32 s12, 0 // (op 935) 
s_mov_b32 s24, scc // (op 935) 
s_cmp_eq_i32 s24, 0 // (op 935) 
s_cselect_b32 s26, 0, -1 // (op 935) 

s_and_b32 s24, s48, 31 // (op 935) 
s_lshl_b32 s27, 1, s24 // (op 935) 
s_add_i32 s49, s27, s26 // (op 935) 


s_and_b32 s26, s1, s49 // (op 935) 


s_add_i32 s1, s25, s26 // (op 935) 


s_ashr_i32 s25, s1, s24 // (op 935) 


s_lshl_b32 s1, s48, 24 // (op 935) 
s_ashr_i32 s24, s1, 31 // (op 935) 

s_xor_b32 s1, s25, s24 // (op 935) 

s_sub_i32 s25, s1, s24 // (op 935) 


s_mul_i32 s1, s15, s13 // (op 935) 
s_add_u32 s26, s20, 64 // (op 935) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 935) End of Int64 add, unlocking SCC
s_sub_u32 s54, s26, 1 // (op 935) least significant half; sets scc
s_subb_u32 s55, s27, 0 // (op 935) most significant half; uses scc

s_ashr_i64 s[26:27], s[54:55], 6 // (op 935) 

s_sub_i32 s24, s26, s1 // (op 935) 

s_mul_i32 s26, s25, s24 // (op 935) 


s_sub_u32 s27, s19, s26 // (op 935) 


s_add_u32 s19, s27, s1 // (op 935) 


s_cmp_eq_i32 s18, 0 // (op 935) 
s_mov_b32 s1, scc // (op 935) 

s_cmp_eq_i32 s1, 0 // (op 935) 
s_cselect_b32 s18, s19, s0 // (op 935) 



v_lshl_add_u32 v14, s18, 6, v12 // (op 935) 


v_mov_b32 v17, 0 // (op 935) convert
v_mov_b32 v16, v14 // (op 935) convert

v_mul_lo_u32 v15, v13, s29 // (op 935) most significant: low of low * high
v_mul_hi_u32 v12, v13, s28 // (op 935) most significant: high of low * low
v_mul_lo_u32 v14, v13, s28 // (op 935) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 935) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v14 // (op 935) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 935) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 935) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 935) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 935) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 935) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 935) 

v_mov_b32 v14, v12 // (op 935) convert

s_mul_i32 s1, s33, 6 // (op 937) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 937) least significant
s_mul_hi_u32 s18, s32, 6 // (op 937) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 937) most significant: sum

s_mul_i32 s19, s28, s1 // (op 937) most significant: low of low * high
s_mul_i32 s24, s29, s0 // (op 937) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 937) least significant
s_mul_hi_u32 s25, s28, s0 // (op 937) most significant: high of low * low
s_add_u32 s19, s19, s24 // (op 937) most significant: sum
s_add_u32 s19, s19, s25 // (op 937) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 937) 

s_mov_b32 s24, s30 // (op 937) reg expression
s_mov_b32 s25, s31 // (op 937) reg expression
s_mov_b32 s26, s0 // (op 937) reg expression
s_mov_b32 s27, 131072 // (op 937) call()

s_mul_hi_i32 s0, s16, s8 // (op 954) 
v_and_b32 v12, -2, v1 // (op 954) 
s_add_i32 s1, s0, s16 // (op 954) 

v_sub_u32 v13, v1, v12 // (op 954) 

s_ashr_i32 s0, s1, 31 // (op 954) 
v_lshlrev_b32 v12, 5, v13 // (op 954) 

s_cmp_eq_i32 s8, 0 // (op 954) 
s_mov_b32 s18, scc // (op 954) 
v_lshrrev_b32 v13, 1, v1 // (op 954) 
s_cmp_eq_i32 s18, 0 // (op 954) 
s_cselect_b32 s19, 0, -1 // (op 954) 

s_and_b32 s18, s9, 31 // (op 954) 
s_lshl_b32 s49, 1, s18 // (op 954) 
s_add_i32 s50, s49, s19 // (op 954) 


s_and_b32 s19, s0, s50 // (op 954) 


s_add_i32 s0, s1, s19 // (op 954) 


s_ashr_i32 s1, s0, s18 // (op 954) 


s_lshl_b32 s0, s9, 24 // (op 954) 
s_ashr_i32 s18, s0, 31 // (op 954) 

s_xor_b32 s0, s1, s18 // (op 954) 

s_sub_i32 s1, s0, s18 // (op 954) 


s_mul_hi_i32 s0, s1, s10 // (op 954) 
s_add_i32 s18, s0, s1 // (op 954) 

s_ashr_i32 s0, s18, 31 // (op 954) 
s_cmp_eq_i32 s10, 0 // (op 954) 
s_mov_b32 s19, scc // (op 954) 
s_cmp_eq_i32 s19, 0 // (op 954) 
s_cselect_b32 s49, 0, -1 // (op 954) 

s_and_b32 s19, s11, 31 // (op 954) 
s_lshl_b32 s50, 1, s19 // (op 954) 
s_add_i32 s51, s50, s49 // (op 954) 


s_and_b32 s49, s0, s51 // (op 954) 


s_add_i32 s0, s18, s49 // (op 954) 


s_ashr_i32 s18, s0, s19 // (op 954) 


s_lshl_b32 s0, s11, 24 // (op 954) 
s_ashr_i32 s19, s0, 31 // (op 954) 

s_xor_b32 s0, s18, s19 // (op 954) 

s_sub_i32 s18, s0, s19 // (op 954) 


s_mul_i32 s0, s1, s13 // (op 954) 

s_sub_u32 s1, s16, s0 // (op 954) 

s_mul_i32 s0, s18, s13 // (op 954) 
s_add_u32 s0, s0, s1 // (op 954) 


s_mul_hi_i32 s1, s16, s7 // (op 954) 
s_add_i32 s18, s1, s16 // (op 954) 

s_ashr_i32 s1, s18, 31 // (op 954) 
s_cmp_eq_i32 s7, 0 // (op 954) 
s_mov_b32 s19, scc // (op 954) 
s_cmp_eq_i32 s19, 0 // (op 954) 
s_cselect_b32 s49, 0, -1 // (op 954) 

s_and_b32 s19, s14, 31 // (op 954) 
s_lshl_b32 s50, 1, s19 // (op 954) 
s_add_i32 s51, s50, s49 // (op 954) 


s_and_b32 s49, s1, s51 // (op 954) 


s_add_i32 s1, s18, s49 // (op 954) 


s_ashr_i32 s18, s1, s19 // (op 954) 


s_lshl_b32 s1, s14, 24 // (op 954) 
s_ashr_i32 s19, s1, 31 // (op 954) 

s_xor_b32 s1, s18, s19 // (op 954) 

s_sub_i32 s18, s1, s19 // (op 954) 


s_add_u32 s54, s4, 32 // (op 954) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 954) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 954) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 954) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 954) 

s_mul_i32 s1, s13, s50 // (op 954) 

s_mul_i32 s19, s6, s1 // (op 954) 

s_mul_i32 s1, s18, s19 // (op 954) 

s_sub_u32 s19, s16, s1 // (op 954) 

s_mov_b32 s1, s19 // (op 954) 
s_mul_hi_i32 s49, s1, s12 // (op 954) 
s_add_i32 s50, s49, s1 // (op 954) 


s_ashr_i32 s1, s50, 31 // (op 954) 
s_cmp_eq_i32 s12, 0 // (op 954) 
s_mov_b32 s49, scc // (op 954) 
s_cmp_eq_i32 s49, 0 // (op 954) 
s_cselect_b32 s51, 0, -1 // (op 954) 

s_and_b32 s49, s48, 31 // (op 954) 
s_lshl_b32 s52, 1, s49 // (op 954) 
s_add_i32 s53, s52, s51 // (op 954) 


s_and_b32 s51, s1, s53 // (op 954) 


s_add_i32 s1, s50, s51 // (op 954) 


s_ashr_i32 s50, s1, s49 // (op 954) 


s_lshl_b32 s1, s48, 24 // (op 954) 
s_ashr_i32 s49, s1, 31 // (op 954) 

s_xor_b32 s1, s50, s49 // (op 954) 

s_sub_i32 s50, s1, s49 // (op 954) 


s_mul_i32 s1, s15, s13 // (op 954) 
s_add_u32 s54, s20, 64 // (op 954) Start of Int64 add, locking SCC
s_addc_u32 s55, s21, 0 // (op 954) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 954) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 954) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 6 // (op 954) 

s_sub_i32 s49, s54, s1 // (op 954) 

s_mul_i32 s51, s50, s49 // (op 954) 


s_sub_u32 s49, s19, s51 // (op 954) 


s_add_u32 s19, s49, s1 // (op 954) 


s_cmp_eq_i32 s18, 0 // (op 954) 
s_mov_b32 s1, scc // (op 954) 

s_cmp_eq_i32 s1, 0 // (op 954) 
s_cselect_b32 s18, s19, s0 // (op 954) 



v_lshl_add_u32 v15, s18, 6, v12 // (op 954) 


v_mov_b32 v17, 0 // (op 954) convert
v_mov_b32 v16, v15 // (op 954) convert

v_mul_lo_u32 v19, v13, s29 // (op 954) most significant: low of low * high
v_mul_hi_u32 v12, v13, s28 // (op 954) most significant: high of low * low
v_mul_lo_u32 v18, v13, s28 // (op 954) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 954) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 954) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 954) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 954) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 954) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 954) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 954) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 954) 

v_mov_b32 v15, v12 // (op 954) convert

s_mul_i32 s1, s33, 6 // (op 956) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 956) least significant
s_mul_hi_u32 s18, s32, 6 // (op 956) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 956) most significant: sum

s_mul_i32 s19, s28, s1 // (op 956) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 956) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 956) least significant
s_mul_hi_u32 s50, s28, s0 // (op 956) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 956) most significant: sum
s_add_u32 s19, s19, s50 // (op 956) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 956) 

s_mov_b32 s24, s30 // (op 956) reg expression
s_mov_b32 s25, s31 // (op 956) reg expression
s_mov_b32 s26, s0 // (op 956) reg expression
s_mov_b32 s27, 131072 // (op 956) call()

s_mul_hi_i32 s0, s16, s8 // (op 999) 
v_lshrrev_b32 v12, 3, v1 // (op 999) 
s_add_i32 s1, s0, s16 // (op 999) 

v_and_b32 v13, -8, v1 // (op 999) 
s_ashr_i32 s0, s1, 31 // (op 999) 
v_sub_u32 v16, v1, v13 // (op 999) 

s_cmp_eq_i32 s8, 0 // (op 999) 
s_mov_b32 s18, scc // (op 999) 
v_lshlrev_b32 v13, 4, v16 // (op 999) 

s_cmp_eq_i32 s18, 0 // (op 999) 
s_cselect_b32 s19, 0, -1 // (op 999) 

v_mov_b32 v17, 0 // (op 999) convert
v_mov_b32 v16, v13 // (op 999) convert

s_and_b32 s18, s9, 31 // (op 999) 
s_lshl_b32 s49, 1, s18 // (op 999) 
s_add_i32 s50, s49, s19 // (op 999) 


s_and_b32 s19, s0, s50 // (op 999) 


s_add_i32 s0, s1, s19 // (op 999) 


s_ashr_i32 s1, s0, s18 // (op 999) 


s_lshl_b32 s0, s9, 24 // (op 999) 
s_ashr_i32 s18, s0, 31 // (op 999) 

s_xor_b32 s0, s1, s18 // (op 999) 

s_sub_i32 s1, s0, s18 // (op 999) 


s_mul_hi_i32 s0, s1, s10 // (op 999) 
s_add_i32 s18, s0, s1 // (op 999) 

s_ashr_i32 s0, s18, 31 // (op 999) 
s_cmp_eq_i32 s10, 0 // (op 999) 
s_mov_b32 s19, scc // (op 999) 
s_cmp_eq_i32 s19, 0 // (op 999) 
s_cselect_b32 s49, 0, -1 // (op 999) 

s_and_b32 s19, s11, 31 // (op 999) 
s_lshl_b32 s50, 1, s19 // (op 999) 
s_add_i32 s51, s50, s49 // (op 999) 


s_and_b32 s49, s0, s51 // (op 999) 


s_add_i32 s0, s18, s49 // (op 999) 


s_ashr_i32 s18, s0, s19 // (op 999) 


s_lshl_b32 s0, s11, 24 // (op 999) 
s_ashr_i32 s19, s0, 31 // (op 999) 

s_xor_b32 s0, s18, s19 // (op 999) 

s_sub_i32 s18, s0, s19 // (op 999) 


s_add_u32 s54, s4, 32 // (op 999) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 999) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 999) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 999) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 999) 

s_mul_i32 s0, s18, s50 // (op 999) 

s_sub_i32 s49, s1, s0 // (op 999) 


s_mul_hi_i32 s0, s16, s7 // (op 999) 
s_add_i32 s1, s0, s16 // (op 999) 

s_ashr_i32 s0, s1, 31 // (op 999) 
s_cmp_eq_i32 s7, 0 // (op 999) 
s_mov_b32 s18, scc // (op 999) 
s_cmp_eq_i32 s18, 0 // (op 999) 
s_cselect_b32 s19, 0, -1 // (op 999) 

s_and_b32 s18, s14, 31 // (op 999) 
s_lshl_b32 s52, 1, s18 // (op 999) 
s_add_i32 s53, s52, s19 // (op 999) 


s_and_b32 s19, s0, s53 // (op 999) 


s_add_i32 s0, s1, s19 // (op 999) 


s_ashr_i32 s1, s0, s18 // (op 999) 


s_lshl_b32 s0, s14, 24 // (op 999) 
s_ashr_i32 s18, s0, 31 // (op 999) 

s_xor_b32 s0, s1, s18 // (op 999) 

s_sub_i32 s1, s0, s18 // (op 999) 


s_mul_i32 s0, s13, s50 // (op 999) 

s_mul_i32 s18, s6, s0 // (op 999) 

s_mul_i32 s0, s1, s18 // (op 999) 

s_sub_u32 s18, s16, s0 // (op 999) 

s_mul_hi_i32 s0, s18, s12 // (op 999) 
s_add_i32 s19, s0, s18 // (op 999) 


s_ashr_i32 s0, s19, 31 // (op 999) 
s_cmp_eq_i32 s12, 0 // (op 999) 
s_mov_b32 s18, scc // (op 999) 
s_cmp_eq_i32 s18, 0 // (op 999) 
s_cselect_b32 s50, 0, -1 // (op 999) 

s_and_b32 s18, s48, 31 // (op 999) 
s_lshl_b32 s51, 1, s18 // (op 999) 
s_add_i32 s52, s51, s50 // (op 999) 


s_and_b32 s50, s0, s52 // (op 999) 


s_add_i32 s0, s19, s50 // (op 999) 


s_ashr_i32 s19, s0, s18 // (op 999) 


s_lshl_b32 s0, s48, 24 // (op 999) 
s_ashr_i32 s18, s0, 31 // (op 999) 

s_xor_b32 s0, s19, s18 // (op 999) 

s_sub_i32 s19, s0, s18 // (op 999) 


s_cmp_eq_i32 s1, 0 // (op 999) 
s_mov_b32 s0, scc // (op 999) 

s_cmp_eq_i32 s0, 0 // (op 999) 
s_cselect_b32 s1, s19, s49 // (op 999) 



v_lshl_add_u32 v13, s1, 5, v12 // (op 999) 


v_mul_lo_u32 v19, v13, s35 // (op 999) most significant: low of low * high
v_mul_hi_u32 v12, v13, s34 // (op 999) most significant: high of low * low
v_mul_lo_u32 v18, v13, s34 // (op 999) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 999) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 999) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 999) most significant half



s_mul_i32 s1, s34, s5 // (op 1001) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 1001) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 1001) least significant
s_mul_hi_u32 s19, s34, s4 // (op 1001) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1001) most significant: sum
s_add_u32 s1, s1, s19 // (op 1001) most significant: sum


s_mov_b32 s52, s36 // (op 1001) reg expression
s_mov_b32 s53, s37 // (op 1001) reg expression
s_mov_b32 s54, s0 // (op 1001) reg expression
s_mov_b32 s55, 131072 // (op 1001) call()

s_mul_hi_i32 s0, s16, s8 // (op 1018) 
v_lshrrev_b32 v16, 3, v1 // (op 1018) 
s_add_i32 s1, s0, s16 // (op 1018) 

v_and_b32 v17, -8, v1 // (op 1018) 
s_ashr_i32 s0, s1, 31 // (op 1018) 
v_sub_u32 v18, v1, v17 // (op 1018) 

s_cmp_eq_i32 s8, 0 // (op 1018) 
s_mov_b32 s18, scc // (op 1018) 
v_lshlrev_b32 v17, 4, v18 // (op 1018) 

s_cmp_eq_i32 s18, 0 // (op 1018) 
s_cselect_b32 s19, 0, -1 // (op 1018) 

v_mov_b32 v19, 0 // (op 1018) convert
v_mov_b32 v18, v17 // (op 1018) convert

s_and_b32 s18, s9, 31 // (op 1018) 
s_lshl_b32 s49, 1, s18 // (op 1018) 
s_add_i32 s50, s49, s19 // (op 1018) 


s_and_b32 s19, s0, s50 // (op 1018) 


s_add_i32 s0, s1, s19 // (op 1018) 


s_ashr_i32 s1, s0, s18 // (op 1018) 


s_lshl_b32 s0, s9, 24 // (op 1018) 
s_ashr_i32 s18, s0, 31 // (op 1018) 

s_xor_b32 s0, s1, s18 // (op 1018) 

s_sub_i32 s1, s0, s18 // (op 1018) 


s_mul_hi_i32 s0, s1, s10 // (op 1018) 
s_add_i32 s18, s0, s1 // (op 1018) 

s_ashr_i32 s0, s18, 31 // (op 1018) 
s_cmp_eq_i32 s10, 0 // (op 1018) 
s_mov_b32 s19, scc // (op 1018) 
s_cmp_eq_i32 s19, 0 // (op 1018) 
s_cselect_b32 s49, 0, -1 // (op 1018) 

s_and_b32 s19, s11, 31 // (op 1018) 
s_lshl_b32 s50, 1, s19 // (op 1018) 
s_add_i32 s51, s50, s49 // (op 1018) 


s_and_b32 s49, s0, s51 // (op 1018) 


s_add_i32 s0, s18, s49 // (op 1018) 


s_ashr_i32 s18, s0, s19 // (op 1018) 


s_lshl_b32 s0, s11, 24 // (op 1018) 
s_ashr_i32 s19, s0, 31 // (op 1018) 

s_xor_b32 s0, s18, s19 // (op 1018) 

s_sub_i32 s18, s0, s19 // (op 1018) 


s_add_u32 s50, s4, 32 // (op 1018) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1018) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1018) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1018) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 1018) 

s_mul_i32 s0, s18, s50 // (op 1018) 

s_sub_i32 s49, s1, s0 // (op 1018) 


s_mul_hi_i32 s0, s16, s7 // (op 1018) 
s_add_i32 s1, s0, s16 // (op 1018) 

s_ashr_i32 s0, s1, 31 // (op 1018) 
s_cmp_eq_i32 s7, 0 // (op 1018) 
s_mov_b32 s18, scc // (op 1018) 
s_cmp_eq_i32 s18, 0 // (op 1018) 
s_cselect_b32 s19, 0, -1 // (op 1018) 

s_and_b32 s18, s14, 31 // (op 1018) 
s_lshl_b32 s64, 1, s18 // (op 1018) 
s_add_i32 s65, s64, s19 // (op 1018) 


s_and_b32 s19, s0, s65 // (op 1018) 


s_add_i32 s0, s1, s19 // (op 1018) 


s_ashr_i32 s1, s0, s18 // (op 1018) 


s_lshl_b32 s0, s14, 24 // (op 1018) 
s_ashr_i32 s18, s0, 31 // (op 1018) 

s_xor_b32 s0, s1, s18 // (op 1018) 

s_sub_i32 s1, s0, s18 // (op 1018) 


s_mul_i32 s0, s13, s50 // (op 1018) 

s_mul_i32 s18, s6, s0 // (op 1018) 

s_mul_i32 s0, s1, s18 // (op 1018) 

s_sub_u32 s18, s16, s0 // (op 1018) 

s_mul_hi_i32 s0, s18, s12 // (op 1018) 
s_add_i32 s19, s0, s18 // (op 1018) 


s_ashr_i32 s0, s19, 31 // (op 1018) 
s_cmp_eq_i32 s12, 0 // (op 1018) 
s_mov_b32 s18, scc // (op 1018) 
s_cmp_eq_i32 s18, 0 // (op 1018) 
s_cselect_b32 s50, 0, -1 // (op 1018) 

s_and_b32 s18, s48, 31 // (op 1018) 
s_lshl_b32 s51, 1, s18 // (op 1018) 
s_add_i32 s64, s51, s50 // (op 1018) 


s_and_b32 s50, s0, s64 // (op 1018) 


s_add_i32 s0, s19, s50 // (op 1018) 


s_ashr_i32 s19, s0, s18 // (op 1018) 


s_lshl_b32 s0, s48, 24 // (op 1018) 
s_ashr_i32 s18, s0, 31 // (op 1018) 

s_xor_b32 s0, s19, s18 // (op 1018) 

s_sub_i32 s19, s0, s18 // (op 1018) 


s_cmp_eq_i32 s1, 0 // (op 1018) 
s_mov_b32 s0, scc // (op 1018) 

s_cmp_eq_i32 s0, 0 // (op 1018) 
s_cselect_b32 s1, s19, s49 // (op 1018) 



v_lshl_add_u32 v17, s1, 5, v16 // (op 1018) 


v_mul_lo_u32 v21, v17, s35 // (op 1018) most significant: low of low * high
v_mul_hi_u32 v16, v17, s34 // (op 1018) most significant: high of low * low
v_mul_lo_u32 v20, v17, s34 // (op 1018) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 1018) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 1018) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 1018) most significant half



s_mul_i32 s1, s34, s5 // (op 1020) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 1020) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 1020) least significant
s_mul_hi_u32 s19, s34, s4 // (op 1020) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1020) most significant: sum
s_add_u32 s1, s1, s19 // (op 1020) most significant: sum


s_mov_b32 s52, s36 // (op 1020) reg expression
s_mov_b32 s53, s37 // (op 1020) reg expression
s_mov_b32 s54, s0 // (op 1020) reg expression
s_mov_b32 s55, 131072 // (op 1020) call()

s_mul_hi_i32 s0, s16, s8 // (op 1063) 
v_and_b32 v18, -64, v1 // (op 1063) 
s_add_i32 s1, s0, s16 // (op 1063) 

v_sub_u32 v19, v1, v18 // (op 1063) 

s_ashr_i32 s0, s1, 31 // (op 1063) 
v_lshrrev_b32 v18, 6, v1 // (op 1063) 
s_cmp_eq_i32 s8, 0 // (op 1063) 
s_mov_b32 s18, scc // (op 1063) 
s_cmp_eq_i32 s18, 0 // (op 1063) 
s_cselect_b32 s19, 0, -1 // (op 1063) 

s_and_b32 s18, s9, 31 // (op 1063) 
s_lshl_b32 s49, 1, s18 // (op 1063) 
s_add_i32 s50, s49, s19 // (op 1063) 


s_and_b32 s19, s0, s50 // (op 1063) 


s_add_i32 s0, s1, s19 // (op 1063) 


s_ashr_i32 s1, s0, s18 // (op 1063) 


s_lshl_b32 s0, s9, 24 // (op 1063) 
s_ashr_i32 s18, s0, 31 // (op 1063) 

s_xor_b32 s0, s1, s18 // (op 1063) 

s_sub_i32 s1, s0, s18 // (op 1063) 


s_mul_hi_i32 s0, s1, s10 // (op 1063) 
s_add_i32 s18, s0, s1 // (op 1063) 

s_ashr_i32 s0, s18, 31 // (op 1063) 
s_cmp_eq_i32 s10, 0 // (op 1063) 
s_mov_b32 s19, scc // (op 1063) 
s_cmp_eq_i32 s19, 0 // (op 1063) 
s_cselect_b32 s49, 0, -1 // (op 1063) 

s_and_b32 s19, s11, 31 // (op 1063) 
s_lshl_b32 s50, 1, s19 // (op 1063) 
s_add_i32 s51, s50, s49 // (op 1063) 


s_and_b32 s49, s0, s51 // (op 1063) 


s_add_i32 s0, s18, s49 // (op 1063) 


s_ashr_i32 s18, s0, s19 // (op 1063) 


s_lshl_b32 s0, s11, 24 // (op 1063) 
s_ashr_i32 s19, s0, 31 // (op 1063) 

s_xor_b32 s0, s18, s19 // (op 1063) 

s_sub_i32 s18, s0, s19 // (op 1063) 


s_mul_i32 s0, s1, s13 // (op 1063) 

s_sub_u32 s1, s16, s0 // (op 1063) 

s_mul_i32 s0, s18, s13 // (op 1063) 
s_add_u32 s0, s0, s1 // (op 1063) 


s_mul_hi_i32 s1, s16, s7 // (op 1063) 
s_add_i32 s18, s1, s16 // (op 1063) 

s_ashr_i32 s1, s18, 31 // (op 1063) 
s_cmp_eq_i32 s7, 0 // (op 1063) 
s_mov_b32 s19, scc // (op 1063) 
s_cmp_eq_i32 s19, 0 // (op 1063) 
s_cselect_b32 s49, 0, -1 // (op 1063) 

s_and_b32 s19, s14, 31 // (op 1063) 
s_lshl_b32 s50, 1, s19 // (op 1063) 
s_add_i32 s51, s50, s49 // (op 1063) 


s_and_b32 s49, s1, s51 // (op 1063) 


s_add_i32 s1, s18, s49 // (op 1063) 


s_ashr_i32 s18, s1, s19 // (op 1063) 


s_lshl_b32 s1, s14, 24 // (op 1063) 
s_ashr_i32 s19, s1, 31 // (op 1063) 

s_xor_b32 s1, s18, s19 // (op 1063) 

s_sub_i32 s18, s1, s19 // (op 1063) 


s_add_u32 s50, s4, 32 // (op 1063) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1063) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1063) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1063) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 1063) 

s_mul_i32 s1, s13, s50 // (op 1063) 

s_mul_i32 s19, s6, s1 // (op 1063) 

s_mul_i32 s1, s18, s19 // (op 1063) 

s_sub_u32 s19, s16, s1 // (op 1063) 

s_mov_b32 s1, s19 // (op 1063) 
s_mul_hi_i32 s49, s1, s12 // (op 1063) 
s_add_i32 s50, s49, s1 // (op 1063) 


s_ashr_i32 s1, s50, 31 // (op 1063) 
s_cmp_eq_i32 s12, 0 // (op 1063) 
s_mov_b32 s49, scc // (op 1063) 
s_cmp_eq_i32 s49, 0 // (op 1063) 
s_cselect_b32 s51, 0, -1 // (op 1063) 

s_and_b32 s49, s48, 31 // (op 1063) 
s_lshl_b32 s64, 1, s49 // (op 1063) 
s_add_i32 s65, s64, s51 // (op 1063) 


s_and_b32 s51, s1, s65 // (op 1063) 


s_add_i32 s1, s50, s51 // (op 1063) 


s_ashr_i32 s50, s1, s49 // (op 1063) 


s_lshl_b32 s1, s48, 24 // (op 1063) 
s_ashr_i32 s49, s1, 31 // (op 1063) 

s_xor_b32 s1, s50, s49 // (op 1063) 

s_sub_i32 s50, s1, s49 // (op 1063) 


s_mul_i32 s1, s15, s13 // (op 1063) 
s_add_u32 s64, s20, 64 // (op 1063) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 1063) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1063) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1063) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 6 // (op 1063) 

s_sub_i32 s49, s64, s1 // (op 1063) 

s_mul_i32 s51, s50, s49 // (op 1063) 


s_sub_u32 s49, s19, s51 // (op 1063) 


s_add_u32 s19, s49, s1 // (op 1063) 


s_cmp_eq_i32 s18, 0 // (op 1063) 
s_mov_b32 s1, scc // (op 1063) 

s_cmp_eq_i32 s1, 0 // (op 1063) 
s_cselect_b32 s18, s19, s0 // (op 1063) 



v_lshl_add_u32 v20, s18, 6, v19 // (op 1063) 


v_mul_lo_u32 v19, v18, s38 // (op 1063) 

v_add_u32 v18, v20, v19 // (op 1063) 


s_mul_i32 s1, s38, s43 // (op 1065) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1065) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1065) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1065) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1065) most significant: sum
s_add_u32 s1, s1, s19 // (op 1065) most significant: sum


s_mov_b32 s64, s40 // (op 1065) reg expression
s_mov_b32 s65, s41 // (op 1065) reg expression
s_mov_b32 s66, s0 // (op 1065) reg expression
s_mov_b32 s67, 131072 // (op 1065) call()

s_mul_hi_i32 s0, s16, s8 // (op 1082) 
v_and_b32 v19, -64, v1 // (op 1082) 
s_add_i32 s1, s0, s16 // (op 1082) 

v_sub_u32 v20, v1, v19 // (op 1082) 

s_ashr_i32 s0, s1, 31 // (op 1082) 
v_lshrrev_b32 v19, 6, v1 // (op 1082) 
s_cmp_eq_i32 s8, 0 // (op 1082) 
s_mov_b32 s18, scc // (op 1082) 
s_cmp_eq_i32 s18, 0 // (op 1082) 
s_cselect_b32 s19, 0, -1 // (op 1082) 

s_and_b32 s18, s9, 31 // (op 1082) 
s_lshl_b32 s49, 1, s18 // (op 1082) 
s_add_i32 s50, s49, s19 // (op 1082) 


s_and_b32 s19, s0, s50 // (op 1082) 


s_add_i32 s0, s1, s19 // (op 1082) 


s_ashr_i32 s1, s0, s18 // (op 1082) 


s_lshl_b32 s0, s9, 24 // (op 1082) 
s_ashr_i32 s18, s0, 31 // (op 1082) 

s_xor_b32 s0, s1, s18 // (op 1082) 

s_sub_i32 s1, s0, s18 // (op 1082) 


s_mul_hi_i32 s0, s1, s10 // (op 1082) 
s_add_i32 s18, s0, s1 // (op 1082) 

s_ashr_i32 s0, s18, 31 // (op 1082) 
s_cmp_eq_i32 s10, 0 // (op 1082) 
s_mov_b32 s19, scc // (op 1082) 
s_cmp_eq_i32 s19, 0 // (op 1082) 
s_cselect_b32 s49, 0, -1 // (op 1082) 

s_and_b32 s19, s11, 31 // (op 1082) 
s_lshl_b32 s50, 1, s19 // (op 1082) 
s_add_i32 s51, s50, s49 // (op 1082) 


s_and_b32 s49, s0, s51 // (op 1082) 


s_add_i32 s0, s18, s49 // (op 1082) 


s_ashr_i32 s18, s0, s19 // (op 1082) 


s_lshl_b32 s0, s11, 24 // (op 1082) 
s_ashr_i32 s19, s0, 31 // (op 1082) 

s_xor_b32 s0, s18, s19 // (op 1082) 

s_sub_i32 s18, s0, s19 // (op 1082) 


s_mul_i32 s0, s1, s13 // (op 1082) 

s_sub_u32 s1, s16, s0 // (op 1082) 

s_mul_i32 s0, s18, s13 // (op 1082) 
s_add_u32 s0, s0, s1 // (op 1082) 


s_mul_hi_i32 s1, s16, s7 // (op 1082) 
s_add_i32 s18, s1, s16 // (op 1082) 

s_ashr_i32 s1, s18, 31 // (op 1082) 
s_cmp_eq_i32 s7, 0 // (op 1082) 
s_mov_b32 s19, scc // (op 1082) 
s_cmp_eq_i32 s19, 0 // (op 1082) 
s_cselect_b32 s49, 0, -1 // (op 1082) 

s_and_b32 s19, s14, 31 // (op 1082) 
s_lshl_b32 s50, 1, s19 // (op 1082) 
s_add_i32 s51, s50, s49 // (op 1082) 


s_and_b32 s49, s1, s51 // (op 1082) 


s_add_i32 s1, s18, s49 // (op 1082) 


s_ashr_i32 s18, s1, s19 // (op 1082) 


s_lshl_b32 s1, s14, 24 // (op 1082) 
s_ashr_i32 s19, s1, 31 // (op 1082) 

s_xor_b32 s1, s18, s19 // (op 1082) 

s_sub_i32 s18, s1, s19 // (op 1082) 


s_add_u32 s50, s4, 32 // (op 1082) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1082) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1082) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1082) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 1082) 

s_mul_i32 s1, s13, s50 // (op 1082) 

s_mul_i32 s19, s6, s1 // (op 1082) 

s_mul_i32 s1, s18, s19 // (op 1082) 

s_sub_u32 s19, s16, s1 // (op 1082) 

s_mov_b32 s1, s19 // (op 1082) 
s_mul_hi_i32 s49, s1, s12 // (op 1082) 
s_add_i32 s50, s49, s1 // (op 1082) 


s_ashr_i32 s1, s50, 31 // (op 1082) 
s_cmp_eq_i32 s12, 0 // (op 1082) 
s_mov_b32 s49, scc // (op 1082) 
s_cmp_eq_i32 s49, 0 // (op 1082) 
s_cselect_b32 s51, 0, -1 // (op 1082) 

s_and_b32 s49, s48, 31 // (op 1082) 
s_lshl_b32 s68, 1, s49 // (op 1082) 
s_add_i32 s69, s68, s51 // (op 1082) 


s_and_b32 s51, s1, s69 // (op 1082) 


s_add_i32 s1, s50, s51 // (op 1082) 


s_ashr_i32 s50, s1, s49 // (op 1082) 


s_lshl_b32 s1, s48, 24 // (op 1082) 
s_ashr_i32 s49, s1, 31 // (op 1082) 

s_xor_b32 s1, s50, s49 // (op 1082) 

s_sub_i32 s50, s1, s49 // (op 1082) 


s_mul_i32 s1, s15, s13 // (op 1082) 
s_add_u32 s68, s20, 64 // (op 1082) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 1082) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1082) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1082) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 6 // (op 1082) 

s_sub_i32 s49, s68, s1 // (op 1082) 

s_mul_i32 s51, s50, s49 // (op 1082) 


s_sub_u32 s49, s19, s51 // (op 1082) 


s_add_u32 s19, s49, s1 // (op 1082) 


s_cmp_eq_i32 s18, 0 // (op 1082) 
s_mov_b32 s1, scc // (op 1082) 

s_cmp_eq_i32 s1, 0 // (op 1082) 
s_cselect_b32 s18, s19, s0 // (op 1082) 



v_lshl_add_u32 v21, s18, 6, v20 // (op 1082) 


v_mul_lo_u32 v20, v19, s38 // (op 1082) 

v_add_u32 v19, v21, v20 // (op 1082) 


s_mul_i32 s1, s38, s43 // (op 1084) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1084) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1084) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1084) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1084) most significant: sum
s_add_u32 s1, s1, s19 // (op 1084) most significant: sum


s_mov_b32 s64, s40 // (op 1084) reg expression
s_mov_b32 s65, s41 // (op 1084) reg expression
s_mov_b32 s66, s0 // (op 1084) reg expression
s_mov_b32 s67, 131072 // (op 1084) call()

s_mul_hi_i32 s0, s16, s8 // (op 1127) 
v_lshrrev_b32 v20, 6, v1 // (op 1127) 
s_add_i32 s1, s0, s16 // (op 1127) 

v_and_b32 v21, 1, v20 // (op 1127) 

s_ashr_i32 s0, s1, 31 // (op 1127) 
v_and_b32 v20, 15, v1 // (op 1127) 
s_cmp_eq_i32 s8, 0 // (op 1127) 
s_mov_b32 s18, scc // (op 1127) 
v_lshl_add_u32 v22, v21, 4, v20 // (op 1127) 


s_cmp_eq_i32 s18, 0 // (op 1127) 
s_cselect_b32 s19, 0, -1 // (op 1127) 

v_and_b32 v20, 63, v1 // (op 1127) 
s_and_b32 s18, s9, 31 // (op 1127) 
v_lshrrev_b32 v21, 4, v20 // (op 1127) 

s_lshl_b32 s49, 1, s18 // (op 1127) 
s_add_i32 s50, s49, s19 // (op 1127) 


s_and_b32 s19, s0, s50 // (op 1127) 


s_add_i32 s0, s1, s19 // (op 1127) 


s_ashr_i32 s1, s0, s18 // (op 1127) 


s_lshl_b32 s0, s9, 24 // (op 1127) 
s_ashr_i32 s18, s0, 31 // (op 1127) 

s_xor_b32 s0, s1, s18 // (op 1127) 

s_sub_i32 s1, s0, s18 // (op 1127) 


s_mul_hi_i32 s0, s1, s10 // (op 1127) 
s_add_i32 s18, s0, s1 // (op 1127) 

s_ashr_i32 s0, s18, 31 // (op 1127) 
s_cmp_eq_i32 s10, 0 // (op 1127) 
s_mov_b32 s19, scc // (op 1127) 
s_cmp_eq_i32 s19, 0 // (op 1127) 
s_cselect_b32 s49, 0, -1 // (op 1127) 

s_and_b32 s19, s11, 31 // (op 1127) 
s_lshl_b32 s50, 1, s19 // (op 1127) 
s_add_i32 s51, s50, s49 // (op 1127) 


s_and_b32 s49, s0, s51 // (op 1127) 


s_add_i32 s0, s18, s49 // (op 1127) 


s_ashr_i32 s18, s0, s19 // (op 1127) 


s_lshl_b32 s0, s11, 24 // (op 1127) 
s_ashr_i32 s19, s0, 31 // (op 1127) 

s_xor_b32 s0, s18, s19 // (op 1127) 

s_sub_i32 s18, s0, s19 // (op 1127) 


s_add_u32 s50, s4, 32 // (op 1127) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1127) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1127) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1127) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 1127) 

s_mul_i32 s0, s18, s50 // (op 1127) 

s_sub_i32 s49, s1, s0 // (op 1127) 


s_mul_hi_i32 s0, s16, s7 // (op 1127) 
s_add_i32 s1, s0, s16 // (op 1127) 

s_ashr_i32 s0, s1, 31 // (op 1127) 
s_cmp_eq_i32 s7, 0 // (op 1127) 
s_mov_b32 s18, scc // (op 1127) 
s_cmp_eq_i32 s18, 0 // (op 1127) 
s_cselect_b32 s19, 0, -1 // (op 1127) 

s_and_b32 s18, s14, 31 // (op 1127) 
s_lshl_b32 s68, 1, s18 // (op 1127) 
s_add_i32 s69, s68, s19 // (op 1127) 


s_and_b32 s19, s0, s69 // (op 1127) 


s_add_i32 s0, s1, s19 // (op 1127) 


s_ashr_i32 s1, s0, s18 // (op 1127) 


s_lshl_b32 s0, s14, 24 // (op 1127) 
s_ashr_i32 s18, s0, 31 // (op 1127) 

s_xor_b32 s0, s1, s18 // (op 1127) 

s_sub_i32 s1, s0, s18 // (op 1127) 


s_mul_i32 s0, s13, s50 // (op 1127) 

s_mul_i32 s18, s6, s0 // (op 1127) 

s_mul_i32 s0, s1, s18 // (op 1127) 

s_sub_u32 s18, s16, s0 // (op 1127) 

s_mul_hi_i32 s0, s18, s12 // (op 1127) 
s_add_i32 s19, s0, s18 // (op 1127) 


s_ashr_i32 s0, s19, 31 // (op 1127) 
s_cmp_eq_i32 s12, 0 // (op 1127) 
s_mov_b32 s18, scc // (op 1127) 
s_cmp_eq_i32 s18, 0 // (op 1127) 
s_cselect_b32 s50, 0, -1 // (op 1127) 

s_and_b32 s18, s48, 31 // (op 1127) 
s_lshl_b32 s51, 1, s18 // (op 1127) 
s_add_i32 s68, s51, s50 // (op 1127) 


s_and_b32 s50, s0, s68 // (op 1127) 


s_add_i32 s0, s19, s50 // (op 1127) 


s_ashr_i32 s19, s0, s18 // (op 1127) 


s_lshl_b32 s0, s48, 24 // (op 1127) 
s_ashr_i32 s18, s0, 31 // (op 1127) 

s_xor_b32 s0, s19, s18 // (op 1127) 

s_sub_i32 s19, s0, s18 // (op 1127) 


s_cmp_eq_i32 s1, 0 // (op 1127) 
s_mov_b32 s0, scc // (op 1127) 

s_cmp_eq_i32 s0, 0 // (op 1127) 
s_cselect_b32 s1, s19, s49 // (op 1127) 



v_lshl_add_u32 v20, s1, 5, v22 // (op 1127) 


v_mul_lo_u32 v22, v20, s44 // (op 1127) 

v_add_u32 v20, v21, v22 // (op 1127) 


s_mul_i32 s1, s44, s5 // (op 1129) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 1129) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 1129) least significant
s_mul_hi_u32 s19, s44, s4 // (op 1129) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1129) most significant: sum
s_add_u32 s1, s1, s19 // (op 1129) most significant: sum


s_mov_b32 s68, s46 // (op 1129) reg expression
s_mov_b32 s69, s47 // (op 1129) reg expression
s_mov_b32 s70, s0 // (op 1129) reg expression
s_mov_b32 s71, 131072 // (op 1129) call()

s_mul_hi_i32 s0, s16, s8 // (op 1146) 
v_lshrrev_b32 v21, 6, v1 // (op 1146) 
s_add_i32 s1, s0, s16 // (op 1146) 

v_and_b32 v22, 1, v21 // (op 1146) 

s_ashr_i32 s0, s1, 31 // (op 1146) 
v_and_b32 v21, 15, v1 // (op 1146) 
s_cmp_eq_i32 s8, 0 // (op 1146) 
s_mov_b32 s18, scc // (op 1146) 
v_lshl_add_u32 v23, v22, 4, v21 // (op 1146) 


s_cmp_eq_i32 s18, 0 // (op 1146) 
s_cselect_b32 s19, 0, -1 // (op 1146) 

v_and_b32 v21, 63, v1 // (op 1146) 
s_and_b32 s18, s9, 31 // (op 1146) 
v_lshrrev_b32 v22, 4, v21 // (op 1146) 

s_lshl_b32 s49, 1, s18 // (op 1146) 
s_add_i32 s50, s49, s19 // (op 1146) 


s_and_b32 s19, s0, s50 // (op 1146) 


s_add_i32 s0, s1, s19 // (op 1146) 


s_ashr_i32 s1, s0, s18 // (op 1146) 


s_lshl_b32 s0, s9, 24 // (op 1146) 
s_ashr_i32 s18, s0, 31 // (op 1146) 

s_xor_b32 s0, s1, s18 // (op 1146) 

s_sub_i32 s1, s0, s18 // (op 1146) 


s_mul_hi_i32 s0, s1, s10 // (op 1146) 
s_add_i32 s18, s0, s1 // (op 1146) 

s_ashr_i32 s0, s18, 31 // (op 1146) 
s_cmp_eq_i32 s10, 0 // (op 1146) 
s_mov_b32 s19, scc // (op 1146) 
s_cmp_eq_i32 s19, 0 // (op 1146) 
s_cselect_b32 s49, 0, -1 // (op 1146) 

s_and_b32 s19, s11, 31 // (op 1146) 
s_lshl_b32 s50, 1, s19 // (op 1146) 
s_add_i32 s51, s50, s49 // (op 1146) 


s_and_b32 s49, s0, s51 // (op 1146) 


s_add_i32 s0, s18, s49 // (op 1146) 


s_ashr_i32 s18, s0, s19 // (op 1146) 


s_lshl_b32 s0, s11, 24 // (op 1146) 
s_ashr_i32 s19, s0, 31 // (op 1146) 

s_xor_b32 s0, s18, s19 // (op 1146) 

s_sub_i32 s18, s0, s19 // (op 1146) 


s_add_u32 s50, s4, 32 // (op 1146) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1146) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1146) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1146) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1146) 

s_mul_i32 s0, s18, s50 // (op 1146) 

s_sub_i32 s49, s1, s0 // (op 1146) 


s_mul_hi_i32 s0, s16, s7 // (op 1146) 
s_add_i32 s1, s0, s16 // (op 1146) 

s_ashr_i32 s0, s1, 31 // (op 1146) 
s_cmp_eq_i32 s7, 0 // (op 1146) 
s_mov_b32 s18, scc // (op 1146) 
s_cmp_eq_i32 s18, 0 // (op 1146) 
s_cselect_b32 s19, 0, -1 // (op 1146) 

s_and_b32 s18, s14, 31 // (op 1146) 
s_lshl_b32 s72, 1, s18 // (op 1146) 
s_add_i32 s73, s72, s19 // (op 1146) 


s_and_b32 s19, s0, s73 // (op 1146) 


s_add_i32 s0, s1, s19 // (op 1146) 


s_ashr_i32 s1, s0, s18 // (op 1146) 


s_lshl_b32 s0, s14, 24 // (op 1146) 
s_ashr_i32 s18, s0, 31 // (op 1146) 

s_xor_b32 s0, s1, s18 // (op 1146) 

s_sub_i32 s1, s0, s18 // (op 1146) 


s_mul_i32 s0, s13, s50 // (op 1146) 

s_mul_i32 s18, s6, s0 // (op 1146) 

s_mul_i32 s0, s1, s18 // (op 1146) 

s_sub_u32 s18, s16, s0 // (op 1146) 

s_mul_hi_i32 s0, s18, s12 // (op 1146) 
s_add_i32 s19, s0, s18 // (op 1146) 


s_ashr_i32 s0, s19, 31 // (op 1146) 
s_cmp_eq_i32 s12, 0 // (op 1146) 
s_mov_b32 s18, scc // (op 1146) 
s_cmp_eq_i32 s18, 0 // (op 1146) 
s_cselect_b32 s50, 0, -1 // (op 1146) 

s_and_b32 s18, s48, 31 // (op 1146) 
s_lshl_b32 s51, 1, s18 // (op 1146) 
s_add_i32 s72, s51, s50 // (op 1146) 


s_and_b32 s50, s0, s72 // (op 1146) 


s_add_i32 s0, s19, s50 // (op 1146) 


s_ashr_i32 s19, s0, s18 // (op 1146) 


s_lshl_b32 s0, s48, 24 // (op 1146) 
s_ashr_i32 s18, s0, 31 // (op 1146) 

s_xor_b32 s0, s19, s18 // (op 1146) 

s_sub_i32 s19, s0, s18 // (op 1146) 


s_cmp_eq_i32 s1, 0 // (op 1146) 
s_mov_b32 s0, scc // (op 1146) 

s_cmp_eq_i32 s0, 0 // (op 1146) 
s_cselect_b32 s1, s19, s49 // (op 1146) 



v_lshl_add_u32 v21, s1, 5, v23 // (op 1146) 


v_mul_lo_u32 v23, v21, s44 // (op 1146) 

v_add_u32 v21, v22, v23 // (op 1146) 


s_mul_i32 s1, s44, s5 // (op 1148) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 1148) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 1148) least significant
s_mul_hi_u32 s19, s44, s4 // (op 1148) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1148) most significant: sum
s_add_u32 s1, s1, s19 // (op 1148) most significant: sum


s_mov_b32 s68, s46 // (op 1148) reg expression
s_mov_b32 s69, s47 // (op 1148) reg expression
s_mov_b32 s70, s0 // (op 1148) reg expression
s_mov_b32 s71, 131072 // (op 1148) call()




buffer_load_ubyte v22, v20, s[68:71], 0 offen offset: 0 // (op 782) Load value
v_and_b32 v23, -8, v1 // (op 1270) 
v_sub_u32 v24, v1, v23 // (op 1270) 

v_lshlrev_b32 v23, 4, v24 // (op 1270) 

v_lshrrev_b32 v24, 3, v1 // (op 1270) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 1270) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 1270) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 890) reg expression
buffer_load_dwordx4 v12, s[52:55], 0 offen offset: 0 lds // (op 890) Load value direct to lds



v_mov_b32 v23, v14 // (op 789) reg expression

buffer_load_dwordx3 v[24:26], v23, s[24:27], 0 offen // (op 789) Load value
buffer_load_dwordx3 v[28:30], v23, s[24:27], 0 offen offset:12 // (op 789) Load value



v_mov_b32 v23, v18 // (op 792) reg expression

buffer_load_ubyte v27, v23, s[64:67], 0 offen offset: 0 // (op 792) Load value




buffer_load_ubyte v23, v20, s[68:71], 0 offen offset:4 // (op 796) Load value

v_and_b32 v20, -8, v1 // (op 1414) 
v_sub_u32 v31, v1, v20 // (op 1414) 

v_lshlrev_b32 v20, 4, v31 // (op 1414) 

v_lshrrev_b32 v31, 3, v1 // (op 1414) 
v_lshl_add_u32 v32, v31, 7, v20 // (op 1414) 


s_nop 0
v_readfirstlane_b32 s0, v32 // (op 1414) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 895) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 895) reg expression
s_mov_b32 s18, 128 // (op 895) call()
buffer_load_dwordx4 v12, s[52:55], s18 offen offset: 0 lds // (op 895) Load value direct to lds






s_lshl_b64 s[0:1], s[28:29], 7 // (op 803) 
s_mul_i32 s19, s1, 6 // (op 803) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 803) least significant
s_mul_hi_u32 s49, s0, 6 // (op 803) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 803) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 803) 

v_add_u32 v20, v14, s0 // (op 803) 


buffer_load_dwordx3 v[32:34], v20, s[24:27], 0 offen // (op 803) Load value
buffer_load_dwordx3 v[36:38], v20, s[24:27], 0 offen offset:12 // (op 803) Load value




s_lshl_b32 s0, s38, 2 // (op 806) 
v_add_u32 v20, v18, s0 // (op 806) 


buffer_load_ubyte v31, v20, s[64:67], 0 offen offset: 0 // (op 806) Load value


v_and_b32 v18, -2, v1 // (op 1213) 
v_sub_u32 v20, v1, v18 // (op 1213) 

v_lshlrev_b32 v18, 5, v20 // (op 1213) 

v_lshrrev_b32 v20, 1, v1 // (op 1213) 
v_lshl_add_u32 v35, v20, 6, v18 // (op 1213) 


v_mul_lo_u32 v18, v35, 6 // (op 1213) 
v_lshrrev_b32 v20, 3, v18 // (op 1213) 

v_lshrrev_b32 v18, 4, v35 // (op 1213) 

v_lshlrev_b32 v35, 2, v18 // (op 1213) 

v_add_u32 v18, v20, v35 // (op 1213) 





s_waitcnt vmcnt(7)
ds_write_b96 v18, v[24:26] offset:8192 // (op 809) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v18, v[28:30] offset:8208 // (op 809) Store local data 

v_and_b32 v18, -64, v1 // (op 1323) 
v_sub_u32 v20, v1, v18 // (op 1323) 

v_and_b32 v18, 4294967232, v1 // (op 1323) 
v_add_u32 v35, v18, v20 // (op 1323) 





s_waitcnt vmcnt(5)
ds_write_b8 v35, v27 offset:16384 // (op 810) Store local data 

v_and_b32 v18, -2, v1 // (op 1184) 
v_sub_u32 v20, v1, v18 // (op 1184) 

v_lshlrev_b32 v18, 5, v20 // (op 1184) 

v_lshrrev_b32 v20, 1, v1 // (op 1184) 
v_lshl_add_u32 v35, v20, 6, v18 // (op 1184) 


v_mul_lo_u32 v18, v35, 6 // (op 1184) 
v_lshrrev_b32 v20, 3, v18 // (op 1184) 

v_lshrrev_b32 v18, 4, v35 // (op 1184) 

v_lshlrev_b32 v35, 2, v18 // (op 1184) 

v_add_u32 v18, v20, v35 // (op 1184) 


v_and_b32 v20, -8, v1 // (op 1242) 
v_sub_u32 v35, v1, v20 // (op 1242) 

v_lshlrev_b32 v20, 4, v35 // (op 1242) 

v_lshrrev_b32 v35, 3, v1 // (op 1242) 
v_lshl_add_u32 v12, v35, 7, v20 // (op 1242) 


s_nop 0
v_readfirstlane_b32 s0, v12 // (op 1242) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -64, v1 // (op 1294) 
v_sub_u32 v35, v1, v20 // (op 1294) 

v_and_b32 v20, 4294967232, v1 // (op 1294) 
v_add_u32 v12, v20, v35 // (op 1294) 


v_and_b32 v20, -2, v1 // (op 1370) 
v_sub_u32 v35, v1, v20 // (op 1370) 

v_lshlrev_b32 v20, 5, v35 // (op 1370) 

v_lshrrev_b32 v35, 1, v1 // (op 1370) 
v_lshl_add_u32 v13, v35, 6, v20 // (op 1370) 


v_mul_lo_u32 v14, v13, 6 // (op 1370) 
v_lshrrev_b32 v20, 3, v14 // (op 1370) 

v_lshrrev_b32 v14, 4, v13 // (op 1370) 

v_lshlrev_b32 v13, 2, v14 // (op 1370) 

v_add_u32 v14, v20, v13 // (op 1370) 


v_and_b32 v13, -8, v1 // (op 1399) 
v_sub_u32 v20, v1, v13 // (op 1399) 

v_lshlrev_b32 v13, 4, v20 // (op 1399) 

v_lshrrev_b32 v20, 3, v1 // (op 1399) 
v_lshl_add_u32 v35, v20, 7, v13 // (op 1399) 


s_nop 0
v_readfirstlane_b32 s1, v35 // (op 1399) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -64, v1 // (op 1438) 
v_sub_u32 v20, v1, v13 // (op 1438) 

v_and_b32 v13, 4294967232, v1 // (op 1438) 
v_add_u32 v35, v13, v20 // (op 1438) 


s_mov_b32 s18, 0 // (op 315) call()
s_ashr_i64 s[50:51], s[32:33], 7 // (op 314) 
s_mov_b32 s19, s50 // (op 314) convert

s_and_b32 s49, s19, 4294967294 // (op 314) 

s_cmp_lt_u32 s18, s49 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_314 // (op 314) Condition: Top (jump to RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_314 if false)
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_314:
s_add_u32 s18, s18, 2 // (op 316) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1496) 



ds_read_b96_tr_b6 v[40:42], v0 offset:8192 // (op 67) Transpose load local data 
ds_read_b96_tr_b6 v[43:45], v0 offset:9216 // (op 67) Transpose load local data 
s_barrier  // (op 1509) 



ds_read_b128 v[46:49], v3 // (op 72) Load local data 
ds_read_b128 v[50:53], v3 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1522) 


ds_read_u8 v13, v7 offset:16384 // (op 77) Load local data 



ds_read_b96_tr_b6 v[54:56], v0 offset:8224 // (op 344) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:45], v[46:53], a[0:3], v13, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 84) 
ds_read_b96_tr_b6 v[57:59], v0 offset:9248 // (op 344) Transpose load local data 




ds_read_u8 v13, v7 offset:16416 // (op 348) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[46:53], a[4:7], v13, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 350) 





v_mov_b32 v13, v21 // (op 82) 

buffer_load_ubyte v22, v13, s[68:71], 0 offen offset:8 // (op 82) Load value
s_barrier  // (op 1512) 


v_mov_b64 v[40:41], v[16:17] // (op 873) 

s_mov_b32 m0, s0 // (op 873) reg expression
s_mov_b32 s19, 256 // (op 873) call()
buffer_load_dwordx4 v40, s[52:55], s19 offen offset: 0 lds // (op 873) Load value direct to lds



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

s_barrier  // (op 1539) 



s_waitcnt vmcnt(7)
ds_write_b96 v14, v[32:34] offset:16640 // (op 229) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v14, v[36:38] offset:16656 // (op 229) Store local data 
s_barrier  // (op 1551) 



s_waitcnt vmcnt(5)
ds_write_b8 v35, v31 offset:24832 // (op 240) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1536) 



ds_read_b96_tr_b6 v[42:44], v10 offset:16640 // (op 231) Transpose load local data 
ds_read_b96_tr_b6 v[45:47], v10 offset:17664 // (op 231) Transpose load local data 
s_barrier  // (op 1542) 



ds_read_b128 v[48:51], v4 offset:4096 // (op 233) Load local data 
ds_read_b128 v[52:55], v4 offset:4160 // (op 233) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1548) 


ds_read_u8 v56, v11 offset:24832 // (op 235) Load local data 



ds_read_b96_tr_b6 v[58:60], v10 offset:16672 // (op 357) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:47], v[48:55], a[0:3], v56, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 238) 
ds_read_b96_tr_b6 v[61:63], v10 offset:17696 // (op 357) Transpose load local data 




s_barrier  // (op 1616) 
ds_read_u8 v42, v11 offset:24864 // (op 361) Load local data 
s_barrier  // (op 1620) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:63], v[48:55], a[4:7], v42, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 351) 






buffer_load_ubyte v23, v13, s[68:71], 0 offen offset:12 // (op 237) Load value

s_barrier  // (op 1545) 
v_add_u32 v21, 8, v21 // (op 1138) 


s_add_u32 s19, 4096, s1 // (op 885) 

s_mov_b32 m0, s19 // (op 885) reg expression
s_mov_b32 s49, 384 // (op 885) call()
buffer_load_dwordx4 v40, s[52:55], s49 offen offset: 0 lds // (op 885) Load value direct to lds





v_add_co_u32 v16, vcc, 256, v16 // (op 1010) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1010) most significant half
s_lshl_b64 s[50:51], s[28:29], 7 // (op 228) 
s_mul_i32 s73, s51, 6 // (op 228) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 228) least significant
s_mul_hi_u32 s19, s50, 6 // (op 228) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 228) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 228) 

s_mul_i32 s19, 3, s50 // (op 228) 

v_add_u32 v13, v20, s19 // (op 228) 


buffer_load_dwordx3 v[32:34], v13, s[24:27], 0 offen // (op 228) Load value
buffer_load_dwordx3 v[36:38], v13, s[24:27], 0 offen offset:12 // (op 228) Load value




s_lshl_b32 s19, s38, 2 // (op 242) 
s_mul_i32 s49, 3, s19 // (op 242) 

v_add_u32 v13, v39, s49 // (op 242) 


s_lshl_b64 s[50:51], s[28:29], 7 // (op 946) 
s_mul_i32 s73, s51, 6 // (op 946) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 946) least significant
s_mul_hi_u32 s19, s50, 6 // (op 946) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 946) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 946) 

s_lshl_b32 s19, s50, 1 // (op 946) 

v_add_u32 v15, v15, s19 // (op 946) 

buffer_load_ubyte v31, v13, s[64:67], 0 offen offset: 0 // (op 242) Load value


s_barrier  // (op 1499) 
s_lshl_b32 s19, s38, 3 // (op 1074) 



v_add_u32 v19, v19, s19 // (op 1074) 

s_waitcnt vmcnt(7)
ds_write_b96 v18, v[24:26] offset:8192 // (op 6) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v18, v[28:30] offset:8208 // (op 6) Store local data 
s_barrier  // (op 1525) 



s_waitcnt vmcnt(5)
ds_write_b8 v12, v27 offset:16384 // (op 21) Store local data 
s_ashr_i64 s[50:51], s[32:33], 7 // (op 314) 
s_mov_b32 s19, s50 // (op 314) convert

s_and_b32 s49, s19, 4294967294 // (op 314) 

s_cmp_lt_u32 s18, s49 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_314 // (op 314) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_314 if true)
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_314:














s_barrier  // (op 1600) 
s_mul_hi_i32 s0, s16, s8 // (op 908) 
v_and_b32 v35, -2, v1 // (op 908) 
s_add_i32 s1, s0, s16 // (op 908) 

v_sub_u32 v12, v1, v35 // (op 908) 

s_ashr_i32 s0, s1, 31 // (op 908) 
v_lshlrev_b32 v35, 5, v12 // (op 908) 

s_cmp_eq_i32 s8, 0 // (op 908) 
s_mov_b32 s18, scc // (op 908) 
v_lshrrev_b32 v12, 1, v1 // (op 908) 
s_cmp_eq_i32 s18, 0 // (op 908) 
s_cselect_b32 s19, 0, -1 // (op 908) 

s_and_b32 s18, s9, 31 // (op 908) 
s_lshl_b32 s49, 1, s18 // (op 908) 
s_add_i32 s50, s49, s19 // (op 908) 


s_and_b32 s19, s0, s50 // (op 908) 


s_add_i32 s0, s1, s19 // (op 908) 


s_ashr_i32 s1, s0, s18 // (op 908) 


s_lshl_b32 s0, s9, 24 // (op 908) 
s_ashr_i32 s18, s0, 31 // (op 908) 

s_xor_b32 s0, s1, s18 // (op 908) 

s_sub_i32 s1, s0, s18 // (op 908) 


s_mul_hi_i32 s0, s1, s10 // (op 908) 
s_add_i32 s18, s0, s1 // (op 908) 

s_ashr_i32 s0, s18, 31 // (op 908) 
s_cmp_eq_i32 s10, 0 // (op 908) 
s_mov_b32 s19, scc // (op 908) 
s_cmp_eq_i32 s19, 0 // (op 908) 
s_cselect_b32 s49, 0, -1 // (op 908) 

s_and_b32 s19, s11, 31 // (op 908) 
s_lshl_b32 s50, 1, s19 // (op 908) 
s_add_i32 s51, s50, s49 // (op 908) 


s_and_b32 s49, s0, s51 // (op 908) 


s_add_i32 s0, s18, s49 // (op 908) 


s_ashr_i32 s18, s0, s19 // (op 908) 


s_lshl_b32 s0, s11, 24 // (op 908) 
s_ashr_i32 s19, s0, 31 // (op 908) 

s_xor_b32 s0, s18, s19 // (op 908) 

s_sub_i32 s18, s0, s19 // (op 908) 


s_mul_i32 s0, s1, s13 // (op 908) 

s_sub_u32 s1, s16, s0 // (op 908) 

s_mul_i32 s0, s18, s13 // (op 908) 
s_add_u32 s0, s0, s1 // (op 908) 


s_mul_hi_i32 s1, s16, s7 // (op 908) 
s_add_i32 s18, s1, s16 // (op 908) 

s_ashr_i32 s1, s18, 31 // (op 908) 
s_cmp_eq_i32 s7, 0 // (op 908) 
s_mov_b32 s19, scc // (op 908) 
s_cmp_eq_i32 s19, 0 // (op 908) 
s_cselect_b32 s49, 0, -1 // (op 908) 

s_and_b32 s19, s14, 31 // (op 908) 
s_lshl_b32 s50, 1, s19 // (op 908) 
s_add_i32 s51, s50, s49 // (op 908) 


s_and_b32 s49, s1, s51 // (op 908) 


s_add_i32 s1, s18, s49 // (op 908) 


s_ashr_i32 s18, s1, s19 // (op 908) 


s_lshl_b32 s1, s14, 24 // (op 908) 
s_ashr_i32 s19, s1, 31 // (op 908) 

s_xor_b32 s1, s18, s19 // (op 908) 

s_sub_i32 s18, s1, s19 // (op 908) 


s_add_u32 s50, s4, 32 // (op 908) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 908) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 908) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 908) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 908) 

s_mul_i32 s1, s13, s50 // (op 908) 

s_mul_i32 s19, s6, s1 // (op 908) 

s_mul_i32 s1, s18, s19 // (op 908) 

s_sub_u32 s19, s16, s1 // (op 908) 

s_mov_b32 s1, s19 // (op 908) 
s_mul_hi_i32 s49, s1, s12 // (op 908) 
s_add_i32 s50, s49, s1 // (op 908) 


s_ashr_i32 s1, s50, 31 // (op 908) 
s_cmp_eq_i32 s12, 0 // (op 908) 
s_mov_b32 s49, scc // (op 908) 
s_cmp_eq_i32 s49, 0 // (op 908) 
s_cselect_b32 s51, 0, -1 // (op 908) 

s_and_b32 s49, s48, 31 // (op 908) 
s_lshl_b32 s72, 1, s49 // (op 908) 
s_add_i32 s73, s72, s51 // (op 908) 


s_and_b32 s51, s1, s73 // (op 908) 


s_add_i32 s1, s50, s51 // (op 908) 


s_ashr_i32 s50, s1, s49 // (op 908) 


s_lshl_b32 s1, s48, 24 // (op 908) 
s_ashr_i32 s49, s1, 31 // (op 908) 

s_xor_b32 s1, s50, s49 // (op 908) 

s_sub_i32 s50, s1, s49 // (op 908) 


s_mul_i32 s1, s15, s13 // (op 908) 
s_add_u32 s72, s20, 64 // (op 908) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 908) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 908) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 908) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 908) 

s_sub_i32 s49, s72, s1 // (op 908) 

s_mul_i32 s51, s50, s49 // (op 908) 


s_sub_u32 s49, s19, s51 // (op 908) 


s_add_u32 s19, s49, s1 // (op 908) 


s_cmp_eq_i32 s18, 0 // (op 908) 
s_mov_b32 s1, scc // (op 908) 

s_cmp_eq_i32 s1, 0 // (op 908) 
s_cselect_b32 s18, s19, s0 // (op 908) 



v_lshl_add_u32 v13, s18, 6, v35 // (op 908) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 908) 
v_mov_b32 v15, 0 // (op 908) convert
v_mov_b32 v14, v13 // (op 908) convert

s_lshr_b32 s18, s0, 1 // (op 908) 

v_lshl_add_u32 v13, s18, 8, v12 // (op 908) 


v_mul_lo_u32 v17, v13, s29 // (op 908) most significant: low of low * high
v_mul_hi_u32 v12, v13, s28 // (op 908) most significant: high of low * low
v_mul_lo_u32 v16, v13, s28 // (op 908) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 908) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 908) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 908) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 908) most significant: low of high * low
v_mul_hi_u32 v35, v12, 6 // (op 908) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 908) least significant: low of low * low
v_add_u32 v15, v15, v35 // (op 908) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 908) 

v_mov_b32 v35, v12 // (op 908) convert

s_mul_i32 s1, s33, 6 // (op 910) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 910) least significant
s_mul_hi_u32 s18, s32, 6 // (op 910) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 910) most significant: sum

s_mul_i32 s19, s28, s1 // (op 910) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 910) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 910) least significant
s_mul_hi_u32 s50, s28, s0 // (op 910) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 910) most significant: sum
s_add_u32 s19, s19, s50 // (op 910) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 910) 

s_mov_b32 s24, s30 // (op 910) reg expression
s_mov_b32 s25, s31 // (op 910) reg expression
s_mov_b32 s26, s0 // (op 910) reg expression
s_mov_b32 s27, 131072 // (op 910) call()

s_mul_hi_i32 s0, s16, s8 // (op 976) 
v_lshrrev_b32 v12, 3, v1 // (op 976) 
s_add_i32 s1, s0, s16 // (op 976) 

v_and_b32 v13, -8, v1 // (op 976) 
s_ashr_i32 s0, s1, 31 // (op 976) 
v_sub_u32 v14, v1, v13 // (op 976) 

s_cmp_eq_i32 s8, 0 // (op 976) 
s_mov_b32 s18, scc // (op 976) 
v_lshlrev_b32 v13, 4, v14 // (op 976) 

s_cmp_eq_i32 s18, 0 // (op 976) 
s_cselect_b32 s19, 0, -1 // (op 976) 

s_and_b32 s18, s9, 31 // (op 976) 
s_lshl_b32 s49, 1, s18 // (op 976) 
s_add_i32 s50, s49, s19 // (op 976) 


s_and_b32 s19, s0, s50 // (op 976) 


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
s_cselect_b32 s49, 0, -1 // (op 976) 

s_and_b32 s19, s11, 31 // (op 976) 
s_lshl_b32 s50, 1, s19 // (op 976) 
s_add_i32 s51, s50, s49 // (op 976) 


s_and_b32 s49, s0, s51 // (op 976) 


s_add_i32 s0, s18, s49 // (op 976) 


s_ashr_i32 s18, s0, s19 // (op 976) 


s_lshl_b32 s0, s11, 24 // (op 976) 
s_ashr_i32 s19, s0, 31 // (op 976) 

s_xor_b32 s0, s18, s19 // (op 976) 

s_sub_i32 s18, s0, s19 // (op 976) 


s_add_u32 s50, s4, 32 // (op 976) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 976) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 976) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 976) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 976) 

s_mul_i32 s0, s18, s50 // (op 976) 

s_sub_i32 s49, s1, s0 // (op 976) 


s_mul_hi_i32 s0, s16, s7 // (op 976) 
s_add_i32 s1, s0, s16 // (op 976) 

s_ashr_i32 s0, s1, 31 // (op 976) 
s_cmp_eq_i32 s7, 0 // (op 976) 
s_mov_b32 s18, scc // (op 976) 
s_cmp_eq_i32 s18, 0 // (op 976) 
s_cselect_b32 s19, 0, -1 // (op 976) 

s_and_b32 s18, s14, 31 // (op 976) 
s_lshl_b32 s72, 1, s18 // (op 976) 
s_add_i32 s73, s72, s19 // (op 976) 


s_and_b32 s19, s0, s73 // (op 976) 


s_add_i32 s0, s1, s19 // (op 976) 


s_ashr_i32 s1, s0, s18 // (op 976) 


s_lshl_b32 s0, s14, 24 // (op 976) 
s_ashr_i32 s18, s0, 31 // (op 976) 

s_xor_b32 s0, s1, s18 // (op 976) 

s_sub_i32 s1, s0, s18 // (op 976) 


s_mul_i32 s0, s13, s50 // (op 976) 

s_mul_i32 s18, s6, s0 // (op 976) 

s_mul_i32 s0, s1, s18 // (op 976) 

s_sub_u32 s18, s16, s0 // (op 976) 

s_mul_hi_i32 s0, s18, s12 // (op 976) 
s_add_i32 s19, s0, s18 // (op 976) 


s_ashr_i32 s0, s19, 31 // (op 976) 
s_cmp_eq_i32 s12, 0 // (op 976) 
s_mov_b32 s18, scc // (op 976) 
s_cmp_eq_i32 s18, 0 // (op 976) 
s_cselect_b32 s50, 0, -1 // (op 976) 

s_and_b32 s18, s48, 31 // (op 976) 
s_lshl_b32 s51, 1, s18 // (op 976) 
s_add_i32 s72, s51, s50 // (op 976) 


s_and_b32 s50, s0, s72 // (op 976) 


s_add_i32 s0, s19, s50 // (op 976) 


s_ashr_i32 s19, s0, s18 // (op 976) 


s_lshl_b32 s0, s48, 24 // (op 976) 
s_ashr_i32 s18, s0, 31 // (op 976) 

s_xor_b32 s0, s19, s18 // (op 976) 

s_sub_i32 s19, s0, s18 // (op 976) 


s_cmp_eq_i32 s1, 0 // (op 976) 
s_mov_b32 s0, scc // (op 976) 

s_cmp_eq_i32 s0, 0 // (op 976) 
s_cselect_b32 s1, s19, s49 // (op 976) 



v_lshl_add_u32 v14, s1, 5, v12 // (op 976) 


v_mul_lo_u32 v25, v14, s35 // (op 976) most significant: low of low * high
v_mul_hi_u32 v12, v14, s34 // (op 976) most significant: high of low * low
v_mul_lo_u32 v24, v14, s34 // (op 976) least significant: low of low * low
v_add_u32 v25, v25, v12 // (op 976) most significant: sum


s_ashr_i64 s[0:1], s[32:33], 7 // (op 976) 
s_lshr_b32 s18, s0, 1 // (op 976) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 976) 


v_mov_b32 v27, 0 // (op 976) convert
v_mov_b32 v26, v12 // (op 976) convert

v_add_co_u32 v12, s[0:1], v26, v24 // (op 976) least significant half
v_addc_co_u32 v13, s[0:1], v27, v25, s[0:1] // (op 976) most significant half



s_mul_i32 s1, s34, s5 // (op 978) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 978) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 978) least significant
s_mul_hi_u32 s19, s34, s4 // (op 978) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 978) most significant: sum
s_add_u32 s1, s1, s19 // (op 978) most significant: sum


s_mov_b32 s52, s36 // (op 978) reg expression
s_mov_b32 s53, s37 // (op 978) reg expression
s_mov_b32 s54, s0 // (op 978) reg expression
s_mov_b32 s55, 131072 // (op 978) call()

s_mul_hi_i32 s0, s16, s8 // (op 1040) 
v_and_b32 v14, -64, v1 // (op 1040) 
s_add_i32 s1, s0, s16 // (op 1040) 

v_sub_u32 v15, v1, v14 // (op 1040) 

s_ashr_i32 s0, s1, 31 // (op 1040) 
v_lshrrev_b32 v14, 6, v1 // (op 1040) 
s_cmp_eq_i32 s8, 0 // (op 1040) 
s_mov_b32 s18, scc // (op 1040) 
s_cmp_eq_i32 s18, 0 // (op 1040) 
s_cselect_b32 s19, 0, -1 // (op 1040) 

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


s_mul_i32 s0, s1, s13 // (op 1040) 

s_sub_u32 s1, s16, s0 // (op 1040) 

s_mul_i32 s0, s18, s13 // (op 1040) 
s_add_u32 s0, s0, s1 // (op 1040) 


s_mul_hi_i32 s1, s16, s7 // (op 1040) 
s_add_i32 s18, s1, s16 // (op 1040) 

s_ashr_i32 s1, s18, 31 // (op 1040) 
s_cmp_eq_i32 s7, 0 // (op 1040) 
s_mov_b32 s19, scc // (op 1040) 
s_cmp_eq_i32 s19, 0 // (op 1040) 
s_cselect_b32 s49, 0, -1 // (op 1040) 

s_and_b32 s19, s14, 31 // (op 1040) 
s_lshl_b32 s50, 1, s19 // (op 1040) 
s_add_i32 s51, s50, s49 // (op 1040) 


s_and_b32 s49, s1, s51 // (op 1040) 


s_add_i32 s1, s18, s49 // (op 1040) 


s_ashr_i32 s18, s1, s19 // (op 1040) 


s_lshl_b32 s1, s14, 24 // (op 1040) 
s_ashr_i32 s19, s1, 31 // (op 1040) 

s_xor_b32 s1, s18, s19 // (op 1040) 

s_sub_i32 s18, s1, s19 // (op 1040) 


s_add_u32 s50, s4, 32 // (op 1040) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1040) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1040) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1040) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1040) 

s_mul_i32 s1, s13, s50 // (op 1040) 

s_mul_i32 s19, s6, s1 // (op 1040) 

s_mul_i32 s1, s18, s19 // (op 1040) 

s_sub_u32 s19, s16, s1 // (op 1040) 

s_mov_b32 s1, s19 // (op 1040) 
s_mul_hi_i32 s49, s1, s12 // (op 1040) 
s_add_i32 s50, s49, s1 // (op 1040) 


s_ashr_i32 s1, s50, 31 // (op 1040) 
s_cmp_eq_i32 s12, 0 // (op 1040) 
s_mov_b32 s49, scc // (op 1040) 
s_cmp_eq_i32 s49, 0 // (op 1040) 
s_cselect_b32 s51, 0, -1 // (op 1040) 

s_and_b32 s49, s48, 31 // (op 1040) 
s_lshl_b32 s72, 1, s49 // (op 1040) 
s_add_i32 s73, s72, s51 // (op 1040) 


s_and_b32 s51, s1, s73 // (op 1040) 


s_add_i32 s1, s50, s51 // (op 1040) 


s_ashr_i32 s50, s1, s49 // (op 1040) 


s_lshl_b32 s1, s48, 24 // (op 1040) 
s_ashr_i32 s49, s1, 31 // (op 1040) 

s_xor_b32 s1, s50, s49 // (op 1040) 

s_sub_i32 s50, s1, s49 // (op 1040) 


s_mul_i32 s1, s15, s13 // (op 1040) 
s_add_u32 s72, s20, 64 // (op 1040) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 1040) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1040) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1040) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 1040) 

s_sub_i32 s49, s72, s1 // (op 1040) 

s_mul_i32 s51, s50, s49 // (op 1040) 


s_sub_u32 s49, s19, s51 // (op 1040) 


s_add_u32 s19, s49, s1 // (op 1040) 


s_cmp_eq_i32 s18, 0 // (op 1040) 
s_mov_b32 s1, scc // (op 1040) 

s_cmp_eq_i32 s1, 0 // (op 1040) 
s_cselect_b32 s18, s19, s0 // (op 1040) 



v_lshl_add_u32 v16, s18, 6, v15 // (op 1040) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1040) 
s_lshr_b32 s18, s0, 1 // (op 1040) 

v_lshl_add_u32 v15, s18, 3, v14 // (op 1040) 


v_mul_lo_u32 v14, v15, s38 // (op 1040) 

v_add_u32 v15, v16, v14 // (op 1040) 


s_mul_i32 s1, s38, s43 // (op 1042) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1042) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1042) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1042) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1042) most significant: sum
s_add_u32 s1, s1, s19 // (op 1042) most significant: sum


s_mov_b32 s64, s40 // (op 1042) reg expression
s_mov_b32 s65, s41 // (op 1042) reg expression
s_mov_b32 s66, s0 // (op 1042) reg expression
s_mov_b32 s67, 131072 // (op 1042) call()

s_mul_hi_i32 s0, s16, s8 // (op 1104) 
v_lshrrev_b32 v14, 6, v1 // (op 1104) 
s_add_i32 s1, s0, s16 // (op 1104) 

v_and_b32 v16, 1, v14 // (op 1104) 

s_ashr_i32 s0, s1, 31 // (op 1104) 
v_and_b32 v14, 15, v1 // (op 1104) 
s_cmp_eq_i32 s8, 0 // (op 1104) 
s_mov_b32 s18, scc // (op 1104) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 1104) 


s_cmp_eq_i32 s18, 0 // (op 1104) 
s_cselect_b32 s19, 0, -1 // (op 1104) 

v_and_b32 v14, 63, v1 // (op 1104) 
s_and_b32 s18, s9, 31 // (op 1104) 
v_lshrrev_b32 v16, 4, v14 // (op 1104) 

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


s_add_u32 s50, s4, 32 // (op 1104) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1104) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1104) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1104) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1104) 

s_mul_i32 s0, s18, s50 // (op 1104) 

s_sub_i32 s49, s1, s0 // (op 1104) 


s_mul_hi_i32 s0, s16, s7 // (op 1104) 
s_add_i32 s1, s0, s16 // (op 1104) 

s_ashr_i32 s0, s1, 31 // (op 1104) 
s_cmp_eq_i32 s7, 0 // (op 1104) 
s_mov_b32 s18, scc // (op 1104) 
s_cmp_eq_i32 s18, 0 // (op 1104) 
s_cselect_b32 s19, 0, -1 // (op 1104) 

s_and_b32 s18, s14, 31 // (op 1104) 
s_lshl_b32 s72, 1, s18 // (op 1104) 
s_add_i32 s73, s72, s19 // (op 1104) 


s_and_b32 s19, s0, s73 // (op 1104) 


s_add_i32 s0, s1, s19 // (op 1104) 


s_ashr_i32 s1, s0, s18 // (op 1104) 


s_lshl_b32 s0, s14, 24 // (op 1104) 
s_ashr_i32 s18, s0, 31 // (op 1104) 

s_xor_b32 s0, s1, s18 // (op 1104) 

s_sub_i32 s1, s0, s18 // (op 1104) 


s_mul_i32 s0, s13, s50 // (op 1104) 

s_mul_i32 s18, s6, s0 // (op 1104) 

s_mul_i32 s0, s1, s18 // (op 1104) 

s_sub_u32 s18, s16, s0 // (op 1104) 

s_mul_hi_i32 s0, s18, s12 // (op 1104) 
s_add_i32 s19, s0, s18 // (op 1104) 


s_ashr_i32 s0, s19, 31 // (op 1104) 
s_cmp_eq_i32 s12, 0 // (op 1104) 
s_mov_b32 s18, scc // (op 1104) 
s_cmp_eq_i32 s18, 0 // (op 1104) 
s_cselect_b32 s50, 0, -1 // (op 1104) 

s_and_b32 s18, s48, 31 // (op 1104) 
s_lshl_b32 s51, 1, s18 // (op 1104) 
s_add_i32 s72, s51, s50 // (op 1104) 


s_and_b32 s50, s0, s72 // (op 1104) 


s_add_i32 s0, s19, s50 // (op 1104) 


s_ashr_i32 s19, s0, s18 // (op 1104) 


s_lshl_b32 s0, s48, 24 // (op 1104) 
s_ashr_i32 s18, s0, 31 // (op 1104) 

s_xor_b32 s0, s19, s18 // (op 1104) 

s_sub_i32 s19, s0, s18 // (op 1104) 


s_cmp_eq_i32 s1, 0 // (op 1104) 
s_mov_b32 s0, scc // (op 1104) 

s_cmp_eq_i32 s0, 0 // (op 1104) 
s_cselect_b32 s1, s19, s49 // (op 1104) 



v_lshl_add_u32 v14, s1, 5, v17 // (op 1104) 


v_mul_lo_u32 v17, v14, s44 // (op 1104) 

s_ashr_i64 s[0:1], s[32:33], 7 // (op 1104) 
s_lshr_b32 s18, s0, 1 // (op 1104) 

v_lshl_add_u32 v14, s18, 3, v16 // (op 1104) 


v_add_u32 v16, v14, v17 // (op 1104) 


s_mul_i32 s1, s44, s5 // (op 1106) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 1106) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 1106) least significant
s_mul_hi_u32 s19, s44, s4 // (op 1106) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1106) most significant: sum
s_add_u32 s1, s1, s19 // (op 1106) most significant: sum


s_mov_b32 s68, s46 // (op 1106) reg expression
s_mov_b32 s69, s47 // (op 1106) reg expression
s_mov_b32 s70, s0 // (op 1106) reg expression
s_mov_b32 s71, 131072 // (op 1106) call()

v_and_b32 v14, -2, v1 // (op 1197) 
v_sub_u32 v17, v1, v14 // (op 1197) 

v_lshlrev_b32 v14, 5, v17 // (op 1197) 

v_lshrrev_b32 v17, 1, v1 // (op 1197) 
v_lshl_add_u32 v18, v17, 6, v14 // (op 1197) 


v_mul_lo_u32 v14, v18, 6 // (op 1197) 
v_lshrrev_b32 v17, 3, v14 // (op 1197) 

v_lshrrev_b32 v14, 4, v18 // (op 1197) 

v_lshlrev_b32 v18, 2, v14 // (op 1197) 

v_add_u32 v14, v17, v18 // (op 1197) 


v_and_b32 v17, -8, v1 // (op 1255) 
v_sub_u32 v18, v1, v17 // (op 1255) 

v_lshlrev_b32 v17, 4, v18 // (op 1255) 

v_lshrrev_b32 v18, 3, v1 // (op 1255) 
v_lshl_add_u32 v19, v18, 7, v17 // (op 1255) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 1255) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -64, v1 // (op 1307) 
v_sub_u32 v18, v1, v17 // (op 1307) 

v_and_b32 v17, 4294967232, v1 // (op 1307) 
v_add_u32 v19, v17, v18 // (op 1307) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 320) 
s_and_b32 s1, s18, 4294967294 // (op 320) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 319) 
s_mov_b32 s49, s18 // (op 319) convert

s_cmp_lt_u32 s1, s49 // (op 319) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_319 // (op 319) Condition: Top (jump to RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_319 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_319:


s_add_u32 s1, s1, 1 // (op 321) 
v_mov_b32 v17, v35 // (op 170) 
v_mov_b32 v18, v17 // (op 170) reg expression

buffer_load_dwordx3 v[20:22], v18, s[24:27], 0 offen // (op 170) Load value
buffer_load_dwordx3 v[24:26], v18, s[24:27], 0 offen offset:12 // (op 170) Load value


s_lshl_b64 s[18:19], s[28:29], 7 // (op 898) 
s_mul_i32 s51, s19, 6 // (op 898) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 898) least significant
s_mul_hi_u32 s49, s18, 6 // (op 898) most significant: high of low * low
s_add_u32 s51, s51, s49 // (op 898) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 898) 

v_add_u32 v35, v35, s18 // (op 898) 

s_barrier  // (op 1502) 
s_barrier  // (op 1515) 





v_mov_b64 v[28:29], v[12:13] // (op 878) 

s_mov_b32 m0, s0 // (op 878) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 878) Load value direct to lds

v_add_co_u32 v12, vcc, 128, v12 // (op 968) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 968) most significant half
s_waitcnt vmcnt(2)
ds_write_b96 v14, v[20:22] offset:8192 // (op 171) Store local data 


v_mov_b32 v23, v15 // (op 184) 
v_mov_b32 v17, v23 // (op 184) reg expression

buffer_load_ubyte v18, v17, s[64:67], 0 offen offset: 0 // (op 184) Load value


s_lshl_b32 s18, s38, 2 // (op 1032) 
v_add_u32 v15, v15, s18 // (op 1032) 

s_waitcnt vmcnt(2)
ds_write_b96 v14, v[24:26] offset:8208 // (op 171) Store local data 

s_barrier  // (op 1528) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1505) 






ds_read_b96_tr_b6 v[20:22], v0 offset:8192 // (op 173) Transpose load local data 
ds_read_b96_tr_b6 v[23:25], v0 offset:9216 // (op 173) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v18 offset:16384 // (op 182) Store local data 

s_barrier  // (op 1518) 



ds_read_b128 v[26:29], v3 // (op 175) Load local data 
ds_read_b128 v[30:33], v3 offset:64 // (op 175) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1532) 


ds_read_u8 v34, v7 offset:16384 // (op 177) Load local data 


v_mov_b32 v17, v16 // (op 179) 

buffer_load_ubyte v18, v17, s[68:71], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v16, 4, v16 // (op 1096) 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:33], a[0:3], v34, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 180) 
ds_read_b96_tr_b6 v[36:38], v0 offset:8224 // (op 325) Transpose load local data 


ds_read_b96_tr_b6 v[39:41], v0 offset:9248 // (op 325) Transpose load local data 


ds_read_u8 v17, v7 offset:16416 // (op 329) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[26:33], a[4:7], v17, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 331) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 319) 
s_mov_b32 s49, s18 // (op 319) convert

s_cmp_lt_u32 s1, s49 // (op 319) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_319 // (op 319) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_319 if true)
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_319:






















s_barrier  // (op 1604) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
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


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen // (op 54) Store value


v_accvgpr_read v10, a4 // (op 517) 
v_accvgpr_read v11, a5 // (op 517) 
v_accvgpr_read v12, a6 // (op 517) 
v_accvgpr_read v13, a7 // (op 517) 
v_mul_f32 v14, v8, v10 // (op 517) 
v_mul_f32 v15, v8, v11 // (op 517) 
v_mul_f32 v16, v8, v12 // (op 517) 
v_mul_f32 v17, v8, v13 // (op 517) 

v_lshrrev_b32 v10, 16, v14 // (op 519) 
v_lshrrev_b32 v11, 16, v15 // (op 519) 
v_lshrrev_b32 v12, 16, v16 // (op 519) 
v_lshrrev_b32 v13, 16, v17 // (op 519) 


v_pack_b32_f16 v14, v10, v11 // (op 520) 
v_pack_b32_f16 v15, v12, v13 // (op 520) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:64 // (op 520) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:



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
v_and_b32 v12, 65535, v10 // (op 551) 
v_lshrrev_b32 v13, 16, v10 // (op 551) 
v_lshlrev_b32 v12, 16, v12 // (op 551) 
v_lshlrev_b32 v13, 16, v13 // (op 551) 
v_mul_f32 v20, v5, v12 // (op 551) 
v_mul_f32 v21, v5, v13 // (op 551) 
v_and_b32 v12, 65535, v11 // (op 551) 
v_lshrrev_b32 v13, 16, v11 // (op 551) 
v_lshlrev_b32 v12, 16, v12 // (op 551) 
v_lshlrev_b32 v13, 16, v13 // (op 551) 
v_mul_f32 v22, v5, v12 // (op 551) 
v_mul_f32 v23, v5, v13 // (op 551) 


v_add_f32 v10, v20, v16 // (op 547) 
v_add_f32 v11, v21, v17 // (op 547) 
v_add_f32 v12, v22, v18 // (op 547) 
v_add_f32 v13, v23, v19 // (op 547) 


v_lshrrev_b32 v14, 16, v10 // (op 548) 
v_lshrrev_b32 v15, 16, v11 // (op 548) 
v_lshrrev_b32 v16, 16, v12 // (op 548) 
v_lshrrev_b32 v17, 16, v13 // (op 548) 


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


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.LRR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__end:
.size RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_, .LRR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__end-RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
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
    DataTypeB: F8
    DestDataType: b
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
  - .name: RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_.kd
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

