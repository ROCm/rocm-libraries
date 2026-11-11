// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_,@function
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s8 // (op 2048) 
s_mul_hi_i32 s1, s16, s8 // (op 2236) 
v_lshrrev_b32 v0, 6, v1 // (op 2162) 
v_and_b32 v3, 1, v0 // (op 2162) 

v_and_b32 v0, 15, v1 // (op 2162) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 2162) 


v_and_b32 v0, 48, v1 // (op 2162) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 2162) 


v_lshrrev_b32 v0, 6, v1 // (op 2306) 
v_and_b32 v4, 1, v0 // (op 2306) 

v_and_b32 v0, 15, v1 // (op 2306) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 2306) 


v_and_b32 v0, 48, v1 // (op 2306) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 2306) 


v_lshrrev_b32 v0, 7, v1 // (op 2048) 
s_add_i32 s18, s0, s16 // (op 2048) 

v_and_b32 v5, 63, v1 // (op 2048) 
s_ashr_i32 s0, s18, 31 // (op 2048) 
v_lshrrev_b32 v6, 4, v5 // (op 2048) 
s_cmp_eq_i32 s8, 0 // (op 2048) 
s_mov_b32 s19, scc // (op 2048) 
v_lshlrev_b32 v7, 2, v6 // (op 2048) 

s_cmp_eq_i32 s19, 0 // (op 2048) 
s_cselect_b32 s51, 0, -1 // (op 2048) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 2048) 


s_and_b32 s19, s9, 31 // (op 2048) 
v_lshrrev_b32 v0, 2, v5 // (op 2048) 

v_and_b32 v5, 3, v0 // (op 2048) 

v_and_b32 v0, 63, v1 // (op 2110) 
v_and_b32 v7, 48, v1 // (op 2110) 
v_sub_u32 v8, v0, v7 // (op 2110) 

v_lshrrev_b32 v7, 4, v0 // (op 2110) 

v_lshl_add_u32 v0, v7, 5, v8 // (op 2110) 


v_lshrrev_b32 v7, 7, v1 // (op 2110) 
v_lshlrev_b32 v8, 4, v7 // (op 2110) 

v_lshl_add_u32 v7, v0, 7, v8 // (op 2110) 


v_mul_lo_u32 v0, v7, 6 // (op 2110) 
v_lshrrev_b32 v8, 3, v0 // (op 2110) 

v_lshrrev_b32 v0, 4, v7 // (op 2110) 

v_lshlrev_b32 v7, 2, v0 // (op 2110) 

v_add_u32 v0, v8, v7 // (op 2110) 


v_lshrrev_b32 v7, 7, v1 // (op 2220) 
v_and_b32 v8, 15, v1 // (op 2220) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 2220) 


v_and_b32 v7, 63, v1 // (op 2220) 
v_lshrrev_b32 v8, 4, v7 // (op 2220) 

v_lshl_add_u32 v7, v8, 7, v9 // (op 2220) 


v_lshrrev_b32 v8, 7, v1 // (op 2236) 
v_and_b32 v9, 63, v1 // (op 2236) 
v_and_b32 v10, 63, v1 // (op 2267) 
v_and_b32 v11, 48, v1 // (op 2267) 
v_sub_u32 v12, v10, v11 // (op 2267) 

v_lshrrev_b32 v11, 4, v10 // (op 2267) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 2267) 


v_lshrrev_b32 v11, 7, v1 // (op 2267) 
v_lshlrev_b32 v12, 4, v11 // (op 2267) 

v_lshl_add_u32 v11, v10, 7, v12 // (op 2267) 


v_mul_lo_u32 v10, v11, 6 // (op 2267) 
v_lshrrev_b32 v12, 3, v10 // (op 2267) 

v_lshrrev_b32 v10, 4, v11 // (op 2267) 

v_lshlrev_b32 v11, 2, v10 // (op 2267) 

v_add_u32 v10, v12, v11 // (op 2267) 


v_lshrrev_b32 v11, 7, v1 // (op 2335) 
v_and_b32 v12, 15, v1 // (op 2335) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 2335) 


v_and_b32 v11, 63, v1 // (op 2335) 
v_lshrrev_b32 v12, 4, v11 // (op 2335) 

v_lshl_add_u32 v11, v12, 7, v13 // (op 2335) 


s_lshl_b32 s52, 1, s19 // (op 2048) 
v_and_b32 v12, 3, v1 // (op 2048) 
s_add_i32 s53, s52, s51 // (op 2048) 


s_and_b32 s51, s0, s53 // (op 2048) 


s_add_i32 s0, s1, s16 // (op 2236) 

s_ashr_i32 s1, s0, 31 // (op 2236) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 2048) 


s_add_i32 s52, s18, s51 // (op 2048) 


v_lshrrev_b32 v5, 6, v1 // (op 2048) 
s_ashr_i32 s18, s52, s19 // (op 2048) 


v_and_b32 v12, 1, v5 // (op 2048) 

s_lshl_b32 s19, s9, 24 // (op 2048) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 2048) 


s_ashr_i32 s51, s19, 31 // (op 2048) 

s_xor_b32 s19, s18, s51 // (op 2048) 

s_sub_i32 s18, s19, s51 // (op 2048) 


s_mul_hi_i32 s19, s18, s10 // (op 2048) 
s_add_i32 s51, s19, s18 // (op 2048) 

s_ashr_i32 s19, s51, 31 // (op 2048) 
s_cmp_eq_i32 s10, 0 // (op 2048) 
s_mov_b32 s52, scc // (op 2048) 
s_cmp_eq_i32 s52, 0 // (op 2048) 
s_cselect_b32 s53, 0, -1 // (op 2048) 

s_and_b32 s52, s11, 31 // (op 2048) 
v_lshrrev_b32 v12, 4, v9 // (op 2236) 
s_cmp_eq_i32 s8, 0 // (op 2236) 
s_mov_b32 s54, scc // (op 2236) 
v_lshlrev_b32 v13, 2, v12 // (op 2236) 

s_cmp_eq_i32 s54, 0 // (op 2236) 
s_cselect_b32 s55, 0, -1 // (op 2236) 

s_lshl_b32 s54, 1, s52 // (op 2048) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 2236) 


v_lshrrev_b32 v8, 2, v9 // (op 2236) 

s_add_i32 s56, s54, s53 // (op 2048) 


s_and_b32 s53, s19, s56 // (op 2048) 


s_add_i32 s19, s51, s53 // (op 2048) 


s_ashr_i32 s51, s19, s52 // (op 2048) 


s_lshl_b32 s19, s11, 24 // (op 2048) 
s_ashr_i32 s52, s19, 31 // (op 2048) 

s_xor_b32 s19, s51, s52 // (op 2048) 

s_sub_i32 s51, s19, s52 // (op 2048) 


s_mul_i32 s19, s18, s13 // (op 2048) 
s_sub_u32 s52, s16, s19 // (op 2048) 

s_mul_i32 s19, s51, s13 // (op 2048) 
s_add_u32 s19, s19, s52 // (op 2048) 

s_mul_hi_i32 s52, s16, s7 // (op 2048) 
s_add_i32 s53, s52, s16 // (op 2048) 

s_ashr_i32 s52, s53, 31 // (op 2048) 
s_cmp_eq_i32 s7, 0 // (op 2048) 
s_mov_b32 s54, scc // (op 2048) 
s_cmp_eq_i32 s54, 0 // (op 2048) 
s_cselect_b32 s56, 0, -1 // (op 2048) 

s_and_b32 s54, s14, 31 // (op 2048) 
s_lshl_b32 s57, 1, s54 // (op 2048) 
s_add_i32 s58, s57, s56 // (op 2048) 


s_and_b32 s56, s52, s58 // (op 2048) 


s_add_i32 s52, s53, s56 // (op 2048) 


s_ashr_i32 s53, s52, s54 // (op 2048) 


s_lshl_b32 s52, s14, 24 // (op 2048) 
s_ashr_i32 s54, s52, 31 // (op 2048) 

s_xor_b32 s52, s53, s54 // (op 2048) 

s_sub_i32 s53, s52, s54 // (op 2048) 


s_and_b32 s52, s9, 31 // (op 2236) 
s_lshl_b32 s54, 1, s52 // (op 2236) 
v_and_b32 v9, 3, v8 // (op 2236) 

s_add_i32 s56, s54, s55 // (op 2236) 


v_and_b32 v8, 3, v1 // (op 2236) 
s_and_b32 s54, s1, s56 // (op 2236) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 2236) 


s_add_i32 s1, s0, s54 // (op 2236) 


v_lshrrev_b32 v8, 6, v1 // (op 2236) 
s_ashr_i32 s0, s1, s52 // (op 2236) 


v_and_b32 v9, 1, v8 // (op 2236) 

s_lshl_b32 s1, s9, 24 // (op 2236) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 2236) 


s_ashr_i32 s52, s1, 31 // (op 2236) 

s_xor_b32 s1, s0, s52 // (op 2236) 

s_sub_i32 s0, s1, s52 // (op 2236) 


s_mul_hi_i32 s1, s0, s10 // (op 2236) 
s_add_i32 s52, s1, s0 // (op 2236) 

s_ashr_i32 s1, s52, 31 // (op 2236) 
s_cmp_eq_i32 s10, 0 // (op 2236) 
s_mov_b32 s54, scc // (op 2236) 
s_cmp_eq_i32 s54, 0 // (op 2236) 
s_cselect_b32 s55, 0, -1 // (op 2236) 

s_and_b32 s54, s11, 31 // (op 2236) 
s_lshl_b32 s56, 1, s54 // (op 2236) 
s_add_i32 s57, s56, s55 // (op 2236) 


s_and_b32 s55, s1, s57 // (op 2236) 


s_add_i32 s1, s52, s55 // (op 2236) 


s_ashr_i32 s52, s1, s54 // (op 2236) 


s_lshl_b32 s1, s11, 24 // (op 2236) 
s_ashr_i32 s54, s1, 31 // (op 2236) 

s_xor_b32 s1, s52, s54 // (op 2236) 

s_sub_i32 s52, s1, s54 // (op 2236) 


s_mul_i32 s1, s0, s13 // (op 2236) 
s_sub_u32 s54, s16, s1 // (op 2236) 

s_mul_i32 s1, s52, s13 // (op 2236) 
s_add_u32 s1, s1, s54 // (op 2236) 

s_mul_hi_i32 s54, s16, s7 // (op 2236) 
s_add_i32 s55, s54, s16 // (op 2236) 

s_ashr_i32 s54, s55, 31 // (op 2236) 
s_cmp_eq_i32 s7, 0 // (op 2236) 
s_mov_b32 s56, scc // (op 2236) 
s_cmp_eq_i32 s56, 0 // (op 2236) 
s_cselect_b32 s57, 0, -1 // (op 2236) 

s_and_b32 s56, s14, 31 // (op 2236) 
s_lshl_b32 s58, 1, s56 // (op 2236) 
s_add_i32 s59, s58, s57 // (op 2236) 


s_and_b32 s57, s54, s59 // (op 2236) 


s_add_i32 s54, s55, s57 // (op 2236) 


s_ashr_i32 s55, s54, s56 // (op 2236) 


s_lshl_b32 s54, s14, 24 // (op 2236) 
s_ashr_i32 s56, s54, 31 // (op 2236) 

s_xor_b32 s54, s55, s56 // (op 2236) 

s_sub_i32 s55, s54, s56 // (op 2236) 


s_add_u32 s56, s4, 32 // (op 2048) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 2048) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2048) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2048) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 2048) 

s_mul_i32 s54, s13, s56 // (op 2048) 
s_mul_i32 s58, s6, s54 // (op 2048) 

s_mul_i32 s54, s53, s58 // (op 2048) 

s_sub_u32 s58, s16, s54 // (op 2048) 

s_mov_b32 s54, s58 // (op 2048) 
s_mul_hi_i32 s59, s54, s12 // (op 2048) 
s_add_i32 s60, s59, s54 // (op 2048) 


s_ashr_i32 s54, s60, 31 // (op 2048) 
s_cmp_eq_i32 s12, 0 // (op 2048) 
s_mov_b32 s59, scc // (op 2048) 
s_cmp_eq_i32 s59, 0 // (op 2048) 
s_cselect_b32 s61, 0, -1 // (op 2048) 

s_add_u32 s62, s4, 32 // (op 2236) Start of Int64 add, locking SCC
s_addc_u32 s63, s5, 0 // (op 2236) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 2236) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 2236) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 5 // (op 2236) 

s_mul_i32 s59, s13, s62 // (op 2236) 
s_mul_i32 s64, s6, s59 // (op 2236) 

s_mul_i32 s59, s55, s64 // (op 2236) 

s_sub_u32 s64, s16, s59 // (op 2236) 

s_mov_b32 s59, s64 // (op 2236) 
s_mul_hi_i32 s65, s59, s12 // (op 2236) 
s_add_i32 s66, s65, s59 // (op 2236) 


s_ashr_i32 s59, s66, 31 // (op 2236) 
s_cmp_eq_i32 s12, 0 // (op 2236) 
s_mov_b32 s65, scc // (op 2236) 
s_cmp_eq_i32 s65, 0 // (op 2236) 
s_cselect_b32 s67, 0, -1 // (op 2236) 


s_waitcnt lgkmcnt(0)
s_and_b32 s65, s48, 31 // (op 2048) 
s_lshl_b32 s68, 1, s65 // (op 2048) 
s_add_i32 s69, s68, s61 // (op 2048) 


s_and_b32 s61, s54, s69 // (op 2048) 


s_add_i32 s54, s60, s61 // (op 2048) 


s_ashr_i32 s60, s54, s65 // (op 2048) 


s_lshl_b32 s54, s48, 24 // (op 2048) 
s_ashr_i32 s61, s54, 31 // (op 2048) 

s_xor_b32 s54, s60, s61 // (op 2048) 

s_sub_i32 s60, s54, s61 // (op 2048) 


s_mul_i32 s54, s15, s13 // (op 2048) 
s_and_b32 s61, s48, 31 // (op 2236) 
s_lshl_b32 s65, 1, s61 // (op 2236) 
s_add_i32 s68, s65, s67 // (op 2236) 


s_and_b32 s65, s59, s68 // (op 2236) 


s_add_i32 s59, s66, s65 // (op 2236) 


s_ashr_i32 s65, s59, s61 // (op 2236) 


s_lshl_b32 s59, s48, 24 // (op 2236) 
s_ashr_i32 s61, s59, 31 // (op 2236) 

s_xor_b32 s59, s65, s61 // (op 2236) 

s_sub_i32 s65, s59, s61 // (op 2236) 


s_mul_i32 s59, s15, s13 // (op 2236) 
s_add_u32 s66, s20, 128 // (op 2048) Start of Int64 add, locking SCC
s_addc_u32 s67, s21, 0 // (op 2048) End of Int64 add, unlocking SCC
s_sub_u32 s68, s66, 1 // (op 2048) least significant half; sets scc
s_subb_u32 s69, s67, 0 // (op 2048) most significant half; uses scc

s_ashr_i64 s[66:67], s[68:69], 7 // (op 2048) 

s_sub_i32 s61, s66, s54 // (op 2048) 

s_mul_i32 s66, s60, s61 // (op 2048) 

s_sub_u32 s61, s58, s66 // (op 2048) 


s_add_u32 s58, s61, s54 // (op 2048) 


s_cmp_eq_i32 s53, 0 // (op 2048) 
s_mov_b32 s54, scc // (op 2048) 

s_cmp_eq_i32 s54, 0 // (op 2048) 
s_cselect_b32 s53, s58, s19 // (op 2048) 


s_mul_i32 s19, s51, s56 // (op 2048) 


v_lshl_add_u32 v9, s53, 7, v6 // (op 2048) 


s_sub_i32 s51, s18, s19 // (op 2048) 


s_cmp_eq_i32 s54, 0 // (op 2048) 
s_cselect_b32 s53, s60, s51 // (op 2048) 



v_lshl_add_u32 v6, s53, 5, v5 // (op 2048) 


v_mul_lo_u32 v5, v6, s2 // (op 2048) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 2048) 


s_add_u32 s18, s20, 128 // (op 2236) Start of Int64 add, locking SCC
s_addc_u32 s19, s21, 0 // (op 2236) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 2236) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 2236) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 7 // (op 2236) 

s_sub_i32 s51, s18, s59 // (op 2236) 

s_mul_i32 s18, s65, s51 // (op 2236) 

s_sub_u32 s19, s64, s18 // (op 2236) 


s_add_u32 s18, s19, s59 // (op 2236) 


s_cmp_eq_i32 s55, 0 // (op 2236) 
s_mov_b32 s19, scc // (op 2236) 

s_cmp_eq_i32 s19, 0 // (op 2236) 
s_cselect_b32 s51, s18, s1 // (op 2236) 


s_mul_i32 s1, s52, s62 // (op 2236) 


v_lshl_add_u32 v5, s51, 7, v12 // (op 2236) 


s_sub_i32 s18, s0, s1 // (op 2236) 


s_cmp_eq_i32 s19, 0 // (op 2236) 
s_cselect_b32 s0, s65, s18 // (op 2236) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 2236) 


v_mul_lo_u32 v8, v9, s22 // (op 2236) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 2236) 


s_mul_i32 s1, s2, s5 // (op 2050) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 2050) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 2050) least significant
s_mul_hi_u32 s19, s2, s4 // (op 2050) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2050) most significant: sum
s_add_u32 s1, s1, s19 // (op 2050) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2050) 

s_mul_i32 s1, s22, s5 // (op 2238) most significant: low of low * high
s_mul_i32 s51, s23, s4 // (op 2238) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 2238) least significant
s_mul_hi_u32 s52, s22, s4 // (op 2238) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 2238) most significant: sum
s_add_u32 s1, s1, s52 // (op 2238) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 2238) 

s_mov_b32 s56, s24 // (op 2050) reg expression
s_mov_b32 s57, s25 // (op 2050) reg expression
s_mov_b32 s58, s18 // (op 2050) reg expression
s_mov_b32 s59, 131072 // (op 2050) call()

s_mov_b32 s60, s26 // (op 2238) reg expression

s_mov_b32 s61, s27 // (op 2238) reg expression
s_mov_b32 s62, s52 // (op 2238) reg expression
s_mov_b32 s63, 131072 // (op 2238) call()


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
v_accvgpr_write a8, 0.00000 // (op 475) call()
v_accvgpr_write a9, 0.00000 // (op 475) call()
v_accvgpr_write a10, 0.00000 // (op 475) call()
v_accvgpr_write a11, 0.00000 // (op 475) call()
v_accvgpr_write a12, 0.00000 // (op 637) call()
v_accvgpr_write a13, 0.00000 // (op 637) call()
v_accvgpr_write a14, 0.00000 // (op 637) call()
v_accvgpr_write a15, 0.00000 // (op 637) call()
s_mul_hi_i32 s0, s16, s8 // (op 1815) 
v_and_b32 v12, -2, v1 // (op 1815) 
s_add_i32 s1, s0, s16 // (op 1815) 

v_sub_u32 v13, v1, v12 // (op 1815) 

s_ashr_i32 s0, s1, 31 // (op 1815) 
v_lshlrev_b32 v12, 6, v13 // (op 1815) 

s_cmp_eq_i32 s8, 0 // (op 1815) 
s_mov_b32 s18, scc // (op 1815) 
v_lshrrev_b32 v13, 1, v1 // (op 1815) 
s_cmp_eq_i32 s18, 0 // (op 1815) 
s_cselect_b32 s19, 0, -1 // (op 1815) 

s_and_b32 s18, s9, 31 // (op 1815) 
s_lshl_b32 s24, 1, s18 // (op 1815) 
s_add_i32 s25, s24, s19 // (op 1815) 


s_and_b32 s19, s0, s25 // (op 1815) 


s_add_i32 s0, s1, s19 // (op 1815) 


s_ashr_i32 s1, s0, s18 // (op 1815) 


s_lshl_b32 s0, s9, 24 // (op 1815) 
s_ashr_i32 s18, s0, 31 // (op 1815) 

s_xor_b32 s0, s1, s18 // (op 1815) 

s_sub_i32 s1, s0, s18 // (op 1815) 


s_mul_hi_i32 s0, s1, s10 // (op 1815) 
s_add_i32 s18, s0, s1 // (op 1815) 

s_ashr_i32 s0, s18, 31 // (op 1815) 
s_cmp_eq_i32 s10, 0 // (op 1815) 
s_mov_b32 s19, scc // (op 1815) 
s_cmp_eq_i32 s19, 0 // (op 1815) 
s_cselect_b32 s24, 0, -1 // (op 1815) 

s_and_b32 s19, s11, 31 // (op 1815) 
s_lshl_b32 s25, 1, s19 // (op 1815) 
s_add_i32 s26, s25, s24 // (op 1815) 


s_and_b32 s27, s0, s26 // (op 1815) 


s_add_i32 s0, s18, s27 // (op 1815) 


s_ashr_i32 s18, s0, s19 // (op 1815) 


s_lshl_b32 s0, s11, 24 // (op 1815) 
s_ashr_i32 s19, s0, 31 // (op 1815) 

s_xor_b32 s0, s18, s19 // (op 1815) 

s_sub_i32 s18, s0, s19 // (op 1815) 


s_mul_i32 s0, s1, s13 // (op 1815) 

s_sub_u32 s1, s16, s0 // (op 1815) 

s_mul_i32 s0, s18, s13 // (op 1815) 
s_add_u32 s0, s0, s1 // (op 1815) 


s_mul_hi_i32 s1, s16, s7 // (op 1815) 
s_add_i32 s18, s1, s16 // (op 1815) 

s_ashr_i32 s1, s18, 31 // (op 1815) 
s_cmp_eq_i32 s7, 0 // (op 1815) 
s_mov_b32 s19, scc // (op 1815) 
s_cmp_eq_i32 s19, 0 // (op 1815) 
s_cselect_b32 s24, 0, -1 // (op 1815) 

s_and_b32 s19, s14, 31 // (op 1815) 
s_lshl_b32 s25, 1, s19 // (op 1815) 
s_add_i32 s26, s25, s24 // (op 1815) 


s_and_b32 s27, s1, s26 // (op 1815) 


s_add_i32 s1, s18, s27 // (op 1815) 


s_ashr_i32 s18, s1, s19 // (op 1815) 


s_lshl_b32 s1, s14, 24 // (op 1815) 
s_ashr_i32 s19, s1, 31 // (op 1815) 

s_xor_b32 s1, s18, s19 // (op 1815) 

s_sub_i32 s18, s1, s19 // (op 1815) 


s_add_u32 s24, s4, 32 // (op 1815) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1815) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 1815) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 1815) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 5 // (op 1815) 

s_mul_i32 s1, s13, s24 // (op 1815) 

s_mul_i32 s19, s6, s1 // (op 1815) 

s_mul_i32 s1, s18, s19 // (op 1815) 

s_sub_u32 s19, s16, s1 // (op 1815) 

s_mov_b32 s1, s19 // (op 1815) 
s_mul_hi_i32 s24, s1, s12 // (op 1815) 
s_add_i32 s25, s24, s1 // (op 1815) 


s_ashr_i32 s1, s25, 31 // (op 1815) 
s_cmp_eq_i32 s12, 0 // (op 1815) 
s_mov_b32 s24, scc // (op 1815) 
s_cmp_eq_i32 s24, 0 // (op 1815) 
s_cselect_b32 s26, 0, -1 // (op 1815) 

s_and_b32 s24, s48, 31 // (op 1815) 
s_lshl_b32 s27, 1, s24 // (op 1815) 
s_add_i32 s49, s27, s26 // (op 1815) 


s_and_b32 s26, s1, s49 // (op 1815) 


s_add_i32 s1, s25, s26 // (op 1815) 


s_ashr_i32 s25, s1, s24 // (op 1815) 


s_lshl_b32 s1, s48, 24 // (op 1815) 
s_ashr_i32 s24, s1, 31 // (op 1815) 

s_xor_b32 s1, s25, s24 // (op 1815) 

s_sub_i32 s25, s1, s24 // (op 1815) 


s_mul_i32 s1, s15, s13 // (op 1815) 
s_add_u32 s26, s20, 128 // (op 1815) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 1815) End of Int64 add, unlocking SCC
s_sub_u32 s54, s26, 1 // (op 1815) least significant half; sets scc
s_subb_u32 s55, s27, 0 // (op 1815) most significant half; uses scc

s_ashr_i64 s[26:27], s[54:55], 7 // (op 1815) 

s_sub_i32 s24, s26, s1 // (op 1815) 

s_mul_i32 s26, s25, s24 // (op 1815) 


s_sub_u32 s27, s19, s26 // (op 1815) 


s_add_u32 s19, s27, s1 // (op 1815) 


s_cmp_eq_i32 s18, 0 // (op 1815) 
s_mov_b32 s1, scc // (op 1815) 

s_cmp_eq_i32 s1, 0 // (op 1815) 
s_cselect_b32 s18, s19, s0 // (op 1815) 



v_lshl_add_u32 v14, s18, 7, v12 // (op 1815) 


v_mov_b32 v17, 0 // (op 1815) convert
v_mov_b32 v16, v14 // (op 1815) convert

v_mul_lo_u32 v15, v13, s29 // (op 1815) most significant: low of low * high
v_mul_hi_u32 v12, v13, s28 // (op 1815) most significant: high of low * low
v_mul_lo_u32 v14, v13, s28 // (op 1815) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 1815) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v14 // (op 1815) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 1815) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 1815) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 1815) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 1815) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 1815) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 1815) 

v_mov_b32 v14, v12 // (op 1815) convert

s_mul_i32 s1, s33, 6 // (op 1817) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1817) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1817) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1817) most significant: sum

s_mul_i32 s19, s28, s1 // (op 1817) most significant: low of low * high
s_mul_i32 s24, s29, s0 // (op 1817) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 1817) least significant
s_mul_hi_u32 s25, s28, s0 // (op 1817) most significant: high of low * low
s_add_u32 s19, s19, s24 // (op 1817) most significant: sum
s_add_u32 s19, s19, s25 // (op 1817) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1817) 

s_mov_b32 s24, s30 // (op 1817) reg expression
s_mov_b32 s25, s31 // (op 1817) reg expression
s_mov_b32 s26, s0 // (op 1817) reg expression
s_mov_b32 s27, 131072 // (op 1817) call()

s_mul_hi_i32 s0, s16, s8 // (op 1834) 
v_and_b32 v12, -2, v1 // (op 1834) 
s_add_i32 s1, s0, s16 // (op 1834) 

v_sub_u32 v13, v1, v12 // (op 1834) 

s_ashr_i32 s0, s1, 31 // (op 1834) 
v_lshlrev_b32 v12, 6, v13 // (op 1834) 

s_cmp_eq_i32 s8, 0 // (op 1834) 
s_mov_b32 s18, scc // (op 1834) 
v_lshrrev_b32 v13, 1, v1 // (op 1834) 
s_cmp_eq_i32 s18, 0 // (op 1834) 
s_cselect_b32 s19, 0, -1 // (op 1834) 

s_and_b32 s18, s9, 31 // (op 1834) 
s_lshl_b32 s49, 1, s18 // (op 1834) 
s_add_i32 s50, s49, s19 // (op 1834) 


s_and_b32 s19, s0, s50 // (op 1834) 


s_add_i32 s0, s1, s19 // (op 1834) 


s_ashr_i32 s1, s0, s18 // (op 1834) 


s_lshl_b32 s0, s9, 24 // (op 1834) 
s_ashr_i32 s18, s0, 31 // (op 1834) 

s_xor_b32 s0, s1, s18 // (op 1834) 

s_sub_i32 s1, s0, s18 // (op 1834) 


s_mul_hi_i32 s0, s1, s10 // (op 1834) 
s_add_i32 s18, s0, s1 // (op 1834) 

s_ashr_i32 s0, s18, 31 // (op 1834) 
s_cmp_eq_i32 s10, 0 // (op 1834) 
s_mov_b32 s19, scc // (op 1834) 
s_cmp_eq_i32 s19, 0 // (op 1834) 
s_cselect_b32 s49, 0, -1 // (op 1834) 

s_and_b32 s19, s11, 31 // (op 1834) 
s_lshl_b32 s50, 1, s19 // (op 1834) 
s_add_i32 s51, s50, s49 // (op 1834) 


s_and_b32 s49, s0, s51 // (op 1834) 


s_add_i32 s0, s18, s49 // (op 1834) 


s_ashr_i32 s18, s0, s19 // (op 1834) 


s_lshl_b32 s0, s11, 24 // (op 1834) 
s_ashr_i32 s19, s0, 31 // (op 1834) 

s_xor_b32 s0, s18, s19 // (op 1834) 

s_sub_i32 s18, s0, s19 // (op 1834) 


s_mul_i32 s0, s1, s13 // (op 1834) 

s_sub_u32 s1, s16, s0 // (op 1834) 

s_mul_i32 s0, s18, s13 // (op 1834) 
s_add_u32 s0, s0, s1 // (op 1834) 


s_mul_hi_i32 s1, s16, s7 // (op 1834) 
s_add_i32 s18, s1, s16 // (op 1834) 

s_ashr_i32 s1, s18, 31 // (op 1834) 
s_cmp_eq_i32 s7, 0 // (op 1834) 
s_mov_b32 s19, scc // (op 1834) 
s_cmp_eq_i32 s19, 0 // (op 1834) 
s_cselect_b32 s49, 0, -1 // (op 1834) 

s_and_b32 s19, s14, 31 // (op 1834) 
s_lshl_b32 s50, 1, s19 // (op 1834) 
s_add_i32 s51, s50, s49 // (op 1834) 


s_and_b32 s49, s1, s51 // (op 1834) 


s_add_i32 s1, s18, s49 // (op 1834) 


s_ashr_i32 s18, s1, s19 // (op 1834) 


s_lshl_b32 s1, s14, 24 // (op 1834) 
s_ashr_i32 s19, s1, 31 // (op 1834) 

s_xor_b32 s1, s18, s19 // (op 1834) 

s_sub_i32 s18, s1, s19 // (op 1834) 


s_add_u32 s54, s4, 32 // (op 1834) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 1834) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1834) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1834) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 1834) 

s_mul_i32 s1, s13, s50 // (op 1834) 

s_mul_i32 s19, s6, s1 // (op 1834) 

s_mul_i32 s1, s18, s19 // (op 1834) 

s_sub_u32 s19, s16, s1 // (op 1834) 

s_mov_b32 s1, s19 // (op 1834) 
s_mul_hi_i32 s49, s1, s12 // (op 1834) 
s_add_i32 s50, s49, s1 // (op 1834) 


s_ashr_i32 s1, s50, 31 // (op 1834) 
s_cmp_eq_i32 s12, 0 // (op 1834) 
s_mov_b32 s49, scc // (op 1834) 
s_cmp_eq_i32 s49, 0 // (op 1834) 
s_cselect_b32 s51, 0, -1 // (op 1834) 

s_and_b32 s49, s48, 31 // (op 1834) 
s_lshl_b32 s52, 1, s49 // (op 1834) 
s_add_i32 s53, s52, s51 // (op 1834) 


s_and_b32 s51, s1, s53 // (op 1834) 


s_add_i32 s1, s50, s51 // (op 1834) 


s_ashr_i32 s50, s1, s49 // (op 1834) 


s_lshl_b32 s1, s48, 24 // (op 1834) 
s_ashr_i32 s49, s1, 31 // (op 1834) 

s_xor_b32 s1, s50, s49 // (op 1834) 

s_sub_i32 s50, s1, s49 // (op 1834) 


s_mul_i32 s1, s15, s13 // (op 1834) 
s_add_u32 s54, s20, 128 // (op 1834) Start of Int64 add, locking SCC
s_addc_u32 s55, s21, 0 // (op 1834) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1834) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1834) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 7 // (op 1834) 

s_sub_i32 s49, s54, s1 // (op 1834) 

s_mul_i32 s51, s50, s49 // (op 1834) 


s_sub_u32 s49, s19, s51 // (op 1834) 


s_add_u32 s19, s49, s1 // (op 1834) 


s_cmp_eq_i32 s18, 0 // (op 1834) 
s_mov_b32 s1, scc // (op 1834) 

s_cmp_eq_i32 s1, 0 // (op 1834) 
s_cselect_b32 s18, s19, s0 // (op 1834) 



v_lshl_add_u32 v15, s18, 7, v12 // (op 1834) 


v_mov_b32 v17, 0 // (op 1834) convert
v_mov_b32 v16, v15 // (op 1834) convert

v_mul_lo_u32 v19, v13, s29 // (op 1834) most significant: low of low * high
v_mul_hi_u32 v12, v13, s28 // (op 1834) most significant: high of low * low
v_mul_lo_u32 v18, v13, s28 // (op 1834) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 1834) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 1834) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 1834) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 1834) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 1834) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 1834) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 1834) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 1834) 

v_mov_b32 v15, v12 // (op 1834) convert

s_mul_i32 s1, s33, 6 // (op 1836) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1836) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1836) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1836) most significant: sum

s_mul_i32 s19, s28, s1 // (op 1836) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 1836) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 1836) least significant
s_mul_hi_u32 s50, s28, s0 // (op 1836) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1836) most significant: sum
s_add_u32 s19, s19, s50 // (op 1836) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1836) 

s_mov_b32 s24, s30 // (op 1836) reg expression
s_mov_b32 s25, s31 // (op 1836) reg expression
s_mov_b32 s26, s0 // (op 1836) reg expression
s_mov_b32 s27, 131072 // (op 1836) call()

s_mul_hi_i32 s0, s16, s8 // (op 1879) 
v_lshrrev_b32 v12, 3, v1 // (op 1879) 
s_add_i32 s1, s0, s16 // (op 1879) 

v_and_b32 v13, -8, v1 // (op 1879) 
s_ashr_i32 s0, s1, 31 // (op 1879) 
v_sub_u32 v16, v1, v13 // (op 1879) 

s_cmp_eq_i32 s8, 0 // (op 1879) 
s_mov_b32 s18, scc // (op 1879) 
v_lshlrev_b32 v13, 4, v16 // (op 1879) 

s_cmp_eq_i32 s18, 0 // (op 1879) 
s_cselect_b32 s19, 0, -1 // (op 1879) 

v_mov_b32 v17, 0 // (op 1879) convert
v_mov_b32 v16, v13 // (op 1879) convert

s_and_b32 s18, s9, 31 // (op 1879) 
s_lshl_b32 s49, 1, s18 // (op 1879) 
s_add_i32 s50, s49, s19 // (op 1879) 


s_and_b32 s19, s0, s50 // (op 1879) 


s_add_i32 s0, s1, s19 // (op 1879) 


s_ashr_i32 s1, s0, s18 // (op 1879) 


s_lshl_b32 s0, s9, 24 // (op 1879) 
s_ashr_i32 s18, s0, 31 // (op 1879) 

s_xor_b32 s0, s1, s18 // (op 1879) 

s_sub_i32 s1, s0, s18 // (op 1879) 


s_mul_hi_i32 s0, s1, s10 // (op 1879) 
s_add_i32 s18, s0, s1 // (op 1879) 

s_ashr_i32 s0, s18, 31 // (op 1879) 
s_cmp_eq_i32 s10, 0 // (op 1879) 
s_mov_b32 s19, scc // (op 1879) 
s_cmp_eq_i32 s19, 0 // (op 1879) 
s_cselect_b32 s49, 0, -1 // (op 1879) 

s_and_b32 s19, s11, 31 // (op 1879) 
s_lshl_b32 s50, 1, s19 // (op 1879) 
s_add_i32 s51, s50, s49 // (op 1879) 


s_and_b32 s49, s0, s51 // (op 1879) 


s_add_i32 s0, s18, s49 // (op 1879) 


s_ashr_i32 s18, s0, s19 // (op 1879) 


s_lshl_b32 s0, s11, 24 // (op 1879) 
s_ashr_i32 s19, s0, 31 // (op 1879) 

s_xor_b32 s0, s18, s19 // (op 1879) 

s_sub_i32 s18, s0, s19 // (op 1879) 


s_add_u32 s54, s4, 32 // (op 1879) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 1879) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1879) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1879) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 1879) 

s_mul_i32 s0, s18, s50 // (op 1879) 

s_sub_i32 s49, s1, s0 // (op 1879) 


s_mul_hi_i32 s0, s16, s7 // (op 1879) 
s_add_i32 s1, s0, s16 // (op 1879) 

s_ashr_i32 s0, s1, 31 // (op 1879) 
s_cmp_eq_i32 s7, 0 // (op 1879) 
s_mov_b32 s18, scc // (op 1879) 
s_cmp_eq_i32 s18, 0 // (op 1879) 
s_cselect_b32 s19, 0, -1 // (op 1879) 

s_and_b32 s18, s14, 31 // (op 1879) 
s_lshl_b32 s52, 1, s18 // (op 1879) 
s_add_i32 s53, s52, s19 // (op 1879) 


s_and_b32 s19, s0, s53 // (op 1879) 


s_add_i32 s0, s1, s19 // (op 1879) 


s_ashr_i32 s1, s0, s18 // (op 1879) 


s_lshl_b32 s0, s14, 24 // (op 1879) 
s_ashr_i32 s18, s0, 31 // (op 1879) 

s_xor_b32 s0, s1, s18 // (op 1879) 

s_sub_i32 s1, s0, s18 // (op 1879) 


s_mul_i32 s0, s13, s50 // (op 1879) 

s_mul_i32 s18, s6, s0 // (op 1879) 

s_mul_i32 s0, s1, s18 // (op 1879) 

s_sub_u32 s18, s16, s0 // (op 1879) 

s_mul_hi_i32 s0, s18, s12 // (op 1879) 
s_add_i32 s19, s0, s18 // (op 1879) 


s_ashr_i32 s0, s19, 31 // (op 1879) 
s_cmp_eq_i32 s12, 0 // (op 1879) 
s_mov_b32 s18, scc // (op 1879) 
s_cmp_eq_i32 s18, 0 // (op 1879) 
s_cselect_b32 s50, 0, -1 // (op 1879) 

s_and_b32 s18, s48, 31 // (op 1879) 
s_lshl_b32 s51, 1, s18 // (op 1879) 
s_add_i32 s52, s51, s50 // (op 1879) 


s_and_b32 s50, s0, s52 // (op 1879) 


s_add_i32 s0, s19, s50 // (op 1879) 


s_ashr_i32 s19, s0, s18 // (op 1879) 


s_lshl_b32 s0, s48, 24 // (op 1879) 
s_ashr_i32 s18, s0, 31 // (op 1879) 

s_xor_b32 s0, s19, s18 // (op 1879) 

s_sub_i32 s19, s0, s18 // (op 1879) 


s_cmp_eq_i32 s1, 0 // (op 1879) 
s_mov_b32 s0, scc // (op 1879) 

s_cmp_eq_i32 s0, 0 // (op 1879) 
s_cselect_b32 s1, s19, s49 // (op 1879) 



v_lshl_add_u32 v13, s1, 5, v12 // (op 1879) 


v_mul_lo_u32 v19, v13, s35 // (op 1879) most significant: low of low * high
v_mul_hi_u32 v12, v13, s34 // (op 1879) most significant: high of low * low
v_mul_lo_u32 v18, v13, s34 // (op 1879) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 1879) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 1879) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 1879) most significant half



s_mul_i32 s1, s34, s5 // (op 1881) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 1881) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 1881) least significant
s_mul_hi_u32 s19, s34, s4 // (op 1881) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1881) most significant: sum
s_add_u32 s1, s1, s19 // (op 1881) most significant: sum


s_mov_b32 s52, s36 // (op 1881) reg expression
s_mov_b32 s53, s37 // (op 1881) reg expression
s_mov_b32 s54, s0 // (op 1881) reg expression
s_mov_b32 s55, 131072 // (op 1881) call()

s_mul_hi_i32 s0, s16, s8 // (op 1898) 
v_lshrrev_b32 v16, 3, v1 // (op 1898) 
s_add_i32 s1, s0, s16 // (op 1898) 

v_and_b32 v17, -8, v1 // (op 1898) 
s_ashr_i32 s0, s1, 31 // (op 1898) 
v_sub_u32 v18, v1, v17 // (op 1898) 

s_cmp_eq_i32 s8, 0 // (op 1898) 
s_mov_b32 s18, scc // (op 1898) 
v_lshlrev_b32 v17, 4, v18 // (op 1898) 

s_cmp_eq_i32 s18, 0 // (op 1898) 
s_cselect_b32 s19, 0, -1 // (op 1898) 

v_mov_b32 v19, 0 // (op 1898) convert
v_mov_b32 v18, v17 // (op 1898) convert

s_and_b32 s18, s9, 31 // (op 1898) 
s_lshl_b32 s49, 1, s18 // (op 1898) 
s_add_i32 s50, s49, s19 // (op 1898) 


s_and_b32 s19, s0, s50 // (op 1898) 


s_add_i32 s0, s1, s19 // (op 1898) 


s_ashr_i32 s1, s0, s18 // (op 1898) 


s_lshl_b32 s0, s9, 24 // (op 1898) 
s_ashr_i32 s18, s0, 31 // (op 1898) 

s_xor_b32 s0, s1, s18 // (op 1898) 

s_sub_i32 s1, s0, s18 // (op 1898) 


s_mul_hi_i32 s0, s1, s10 // (op 1898) 
s_add_i32 s18, s0, s1 // (op 1898) 

s_ashr_i32 s0, s18, 31 // (op 1898) 
s_cmp_eq_i32 s10, 0 // (op 1898) 
s_mov_b32 s19, scc // (op 1898) 
s_cmp_eq_i32 s19, 0 // (op 1898) 
s_cselect_b32 s49, 0, -1 // (op 1898) 

s_and_b32 s19, s11, 31 // (op 1898) 
s_lshl_b32 s50, 1, s19 // (op 1898) 
s_add_i32 s51, s50, s49 // (op 1898) 


s_and_b32 s49, s0, s51 // (op 1898) 


s_add_i32 s0, s18, s49 // (op 1898) 


s_ashr_i32 s18, s0, s19 // (op 1898) 


s_lshl_b32 s0, s11, 24 // (op 1898) 
s_ashr_i32 s19, s0, 31 // (op 1898) 

s_xor_b32 s0, s18, s19 // (op 1898) 

s_sub_i32 s18, s0, s19 // (op 1898) 


s_add_u32 s50, s4, 32 // (op 1898) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1898) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1898) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1898) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 1898) 

s_mul_i32 s0, s18, s50 // (op 1898) 

s_sub_i32 s49, s1, s0 // (op 1898) 


s_mul_hi_i32 s0, s16, s7 // (op 1898) 
s_add_i32 s1, s0, s16 // (op 1898) 

s_ashr_i32 s0, s1, 31 // (op 1898) 
s_cmp_eq_i32 s7, 0 // (op 1898) 
s_mov_b32 s18, scc // (op 1898) 
s_cmp_eq_i32 s18, 0 // (op 1898) 
s_cselect_b32 s19, 0, -1 // (op 1898) 

s_and_b32 s18, s14, 31 // (op 1898) 
s_lshl_b32 s64, 1, s18 // (op 1898) 
s_add_i32 s65, s64, s19 // (op 1898) 


s_and_b32 s19, s0, s65 // (op 1898) 


s_add_i32 s0, s1, s19 // (op 1898) 


s_ashr_i32 s1, s0, s18 // (op 1898) 


s_lshl_b32 s0, s14, 24 // (op 1898) 
s_ashr_i32 s18, s0, 31 // (op 1898) 

s_xor_b32 s0, s1, s18 // (op 1898) 

s_sub_i32 s1, s0, s18 // (op 1898) 


s_mul_i32 s0, s13, s50 // (op 1898) 

s_mul_i32 s18, s6, s0 // (op 1898) 

s_mul_i32 s0, s1, s18 // (op 1898) 

s_sub_u32 s18, s16, s0 // (op 1898) 

s_mul_hi_i32 s0, s18, s12 // (op 1898) 
s_add_i32 s19, s0, s18 // (op 1898) 


s_ashr_i32 s0, s19, 31 // (op 1898) 
s_cmp_eq_i32 s12, 0 // (op 1898) 
s_mov_b32 s18, scc // (op 1898) 
s_cmp_eq_i32 s18, 0 // (op 1898) 
s_cselect_b32 s50, 0, -1 // (op 1898) 

s_and_b32 s18, s48, 31 // (op 1898) 
s_lshl_b32 s51, 1, s18 // (op 1898) 
s_add_i32 s64, s51, s50 // (op 1898) 


s_and_b32 s50, s0, s64 // (op 1898) 


s_add_i32 s0, s19, s50 // (op 1898) 


s_ashr_i32 s19, s0, s18 // (op 1898) 


s_lshl_b32 s0, s48, 24 // (op 1898) 
s_ashr_i32 s18, s0, 31 // (op 1898) 

s_xor_b32 s0, s19, s18 // (op 1898) 

s_sub_i32 s19, s0, s18 // (op 1898) 


s_cmp_eq_i32 s1, 0 // (op 1898) 
s_mov_b32 s0, scc // (op 1898) 

s_cmp_eq_i32 s0, 0 // (op 1898) 
s_cselect_b32 s1, s19, s49 // (op 1898) 



v_lshl_add_u32 v17, s1, 5, v16 // (op 1898) 


v_mul_lo_u32 v21, v17, s35 // (op 1898) most significant: low of low * high
v_mul_hi_u32 v16, v17, s34 // (op 1898) most significant: high of low * low
v_mul_lo_u32 v20, v17, s34 // (op 1898) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 1898) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 1898) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 1898) most significant half



s_mul_i32 s1, s34, s5 // (op 1900) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 1900) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 1900) least significant
s_mul_hi_u32 s19, s34, s4 // (op 1900) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1900) most significant: sum
s_add_u32 s1, s1, s19 // (op 1900) most significant: sum


s_mov_b32 s52, s36 // (op 1900) reg expression
s_mov_b32 s53, s37 // (op 1900) reg expression
s_mov_b32 s54, s0 // (op 1900) reg expression
s_mov_b32 s55, 131072 // (op 1900) call()

s_mul_hi_i32 s0, s16, s8 // (op 1943) 
v_and_b32 v18, -128, v1 // (op 1943) 
s_add_i32 s1, s0, s16 // (op 1943) 

v_sub_u32 v19, v1, v18 // (op 1943) 

s_ashr_i32 s0, s1, 31 // (op 1943) 
v_lshrrev_b32 v18, 7, v1 // (op 1943) 
s_cmp_eq_i32 s8, 0 // (op 1943) 
s_mov_b32 s18, scc // (op 1943) 
v_lshlrev_b32 v20, 1, v18 // (op 1943) 

s_cmp_eq_i32 s18, 0 // (op 1943) 
s_cselect_b32 s19, 0, -1 // (op 1943) 

s_and_b32 s18, s9, 31 // (op 1943) 
s_lshl_b32 s49, 1, s18 // (op 1943) 
s_add_i32 s50, s49, s19 // (op 1943) 


s_and_b32 s19, s0, s50 // (op 1943) 


s_add_i32 s0, s1, s19 // (op 1943) 


s_ashr_i32 s1, s0, s18 // (op 1943) 


s_lshl_b32 s0, s9, 24 // (op 1943) 
s_ashr_i32 s18, s0, 31 // (op 1943) 

s_xor_b32 s0, s1, s18 // (op 1943) 

s_sub_i32 s1, s0, s18 // (op 1943) 


s_mul_hi_i32 s0, s1, s10 // (op 1943) 
s_add_i32 s18, s0, s1 // (op 1943) 

s_ashr_i32 s0, s18, 31 // (op 1943) 
s_cmp_eq_i32 s10, 0 // (op 1943) 
s_mov_b32 s19, scc // (op 1943) 
s_cmp_eq_i32 s19, 0 // (op 1943) 
s_cselect_b32 s49, 0, -1 // (op 1943) 

s_and_b32 s19, s11, 31 // (op 1943) 
s_lshl_b32 s50, 1, s19 // (op 1943) 
s_add_i32 s51, s50, s49 // (op 1943) 


s_and_b32 s49, s0, s51 // (op 1943) 


s_add_i32 s0, s18, s49 // (op 1943) 


s_ashr_i32 s18, s0, s19 // (op 1943) 


s_lshl_b32 s0, s11, 24 // (op 1943) 
s_ashr_i32 s19, s0, 31 // (op 1943) 

s_xor_b32 s0, s18, s19 // (op 1943) 

s_sub_i32 s18, s0, s19 // (op 1943) 


s_mul_i32 s0, s1, s13 // (op 1943) 

s_sub_u32 s1, s16, s0 // (op 1943) 

s_mul_i32 s0, s18, s13 // (op 1943) 
s_add_u32 s0, s0, s1 // (op 1943) 


s_mul_hi_i32 s1, s16, s7 // (op 1943) 
s_add_i32 s18, s1, s16 // (op 1943) 

s_ashr_i32 s1, s18, 31 // (op 1943) 
s_cmp_eq_i32 s7, 0 // (op 1943) 
s_mov_b32 s19, scc // (op 1943) 
s_cmp_eq_i32 s19, 0 // (op 1943) 
s_cselect_b32 s49, 0, -1 // (op 1943) 

s_and_b32 s19, s14, 31 // (op 1943) 
s_lshl_b32 s50, 1, s19 // (op 1943) 
s_add_i32 s51, s50, s49 // (op 1943) 


s_and_b32 s49, s1, s51 // (op 1943) 


s_add_i32 s1, s18, s49 // (op 1943) 


s_ashr_i32 s18, s1, s19 // (op 1943) 


s_lshl_b32 s1, s14, 24 // (op 1943) 
s_ashr_i32 s19, s1, 31 // (op 1943) 

s_xor_b32 s1, s18, s19 // (op 1943) 

s_sub_i32 s18, s1, s19 // (op 1943) 


s_add_u32 s50, s4, 32 // (op 1943) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1943) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1943) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1943) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 1943) 

s_mul_i32 s1, s13, s50 // (op 1943) 

s_mul_i32 s19, s6, s1 // (op 1943) 

s_mul_i32 s1, s18, s19 // (op 1943) 

s_sub_u32 s19, s16, s1 // (op 1943) 

s_mov_b32 s1, s19 // (op 1943) 
s_mul_hi_i32 s49, s1, s12 // (op 1943) 
s_add_i32 s50, s49, s1 // (op 1943) 


s_ashr_i32 s1, s50, 31 // (op 1943) 
s_cmp_eq_i32 s12, 0 // (op 1943) 
s_mov_b32 s49, scc // (op 1943) 
s_cmp_eq_i32 s49, 0 // (op 1943) 
s_cselect_b32 s51, 0, -1 // (op 1943) 

s_and_b32 s49, s48, 31 // (op 1943) 
s_lshl_b32 s64, 1, s49 // (op 1943) 
s_add_i32 s65, s64, s51 // (op 1943) 


s_and_b32 s51, s1, s65 // (op 1943) 


s_add_i32 s1, s50, s51 // (op 1943) 


s_ashr_i32 s50, s1, s49 // (op 1943) 


s_lshl_b32 s1, s48, 24 // (op 1943) 
s_ashr_i32 s49, s1, 31 // (op 1943) 

s_xor_b32 s1, s50, s49 // (op 1943) 

s_sub_i32 s50, s1, s49 // (op 1943) 


s_mul_i32 s1, s15, s13 // (op 1943) 
s_add_u32 s64, s20, 128 // (op 1943) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 1943) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1943) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1943) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 7 // (op 1943) 

s_sub_i32 s49, s64, s1 // (op 1943) 

s_mul_i32 s51, s50, s49 // (op 1943) 


s_sub_u32 s49, s19, s51 // (op 1943) 


s_add_u32 s19, s49, s1 // (op 1943) 


s_cmp_eq_i32 s18, 0 // (op 1943) 
s_mov_b32 s1, scc // (op 1943) 

s_cmp_eq_i32 s1, 0 // (op 1943) 
s_cselect_b32 s18, s19, s0 // (op 1943) 



v_lshl_add_u32 v18, s18, 7, v19 // (op 1943) 


v_mul_lo_u32 v19, v20, s38 // (op 1943) 

v_add_u32 v20, v18, v19 // (op 1943) 


s_mul_i32 s1, s38, s43 // (op 1945) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1945) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1945) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1945) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1945) most significant: sum
s_add_u32 s1, s1, s19 // (op 1945) most significant: sum


s_mov_b32 s64, s40 // (op 1945) reg expression
s_mov_b32 s65, s41 // (op 1945) reg expression
s_mov_b32 s66, s0 // (op 1945) reg expression
s_mov_b32 s67, 131072 // (op 1945) call()

s_mul_hi_i32 s0, s16, s8 // (op 1962) 
v_and_b32 v18, -128, v1 // (op 1962) 
s_add_i32 s1, s0, s16 // (op 1962) 

v_sub_u32 v19, v1, v18 // (op 1962) 

s_ashr_i32 s0, s1, 31 // (op 1962) 
v_lshrrev_b32 v18, 7, v1 // (op 1962) 
s_cmp_eq_i32 s8, 0 // (op 1962) 
s_mov_b32 s18, scc // (op 1962) 
v_lshlrev_b32 v21, 1, v18 // (op 1962) 

s_cmp_eq_i32 s18, 0 // (op 1962) 
s_cselect_b32 s19, 0, -1 // (op 1962) 

s_and_b32 s18, s9, 31 // (op 1962) 
s_lshl_b32 s49, 1, s18 // (op 1962) 
s_add_i32 s50, s49, s19 // (op 1962) 


s_and_b32 s19, s0, s50 // (op 1962) 


s_add_i32 s0, s1, s19 // (op 1962) 


s_ashr_i32 s1, s0, s18 // (op 1962) 


s_lshl_b32 s0, s9, 24 // (op 1962) 
s_ashr_i32 s18, s0, 31 // (op 1962) 

s_xor_b32 s0, s1, s18 // (op 1962) 

s_sub_i32 s1, s0, s18 // (op 1962) 


s_mul_hi_i32 s0, s1, s10 // (op 1962) 
s_add_i32 s18, s0, s1 // (op 1962) 

s_ashr_i32 s0, s18, 31 // (op 1962) 
s_cmp_eq_i32 s10, 0 // (op 1962) 
s_mov_b32 s19, scc // (op 1962) 
s_cmp_eq_i32 s19, 0 // (op 1962) 
s_cselect_b32 s49, 0, -1 // (op 1962) 

s_and_b32 s19, s11, 31 // (op 1962) 
s_lshl_b32 s50, 1, s19 // (op 1962) 
s_add_i32 s51, s50, s49 // (op 1962) 


s_and_b32 s49, s0, s51 // (op 1962) 


s_add_i32 s0, s18, s49 // (op 1962) 


s_ashr_i32 s18, s0, s19 // (op 1962) 


s_lshl_b32 s0, s11, 24 // (op 1962) 
s_ashr_i32 s19, s0, 31 // (op 1962) 

s_xor_b32 s0, s18, s19 // (op 1962) 

s_sub_i32 s18, s0, s19 // (op 1962) 


s_mul_i32 s0, s1, s13 // (op 1962) 

s_sub_u32 s1, s16, s0 // (op 1962) 

s_mul_i32 s0, s18, s13 // (op 1962) 
s_add_u32 s0, s0, s1 // (op 1962) 


s_mul_hi_i32 s1, s16, s7 // (op 1962) 
s_add_i32 s18, s1, s16 // (op 1962) 

s_ashr_i32 s1, s18, 31 // (op 1962) 
s_cmp_eq_i32 s7, 0 // (op 1962) 
s_mov_b32 s19, scc // (op 1962) 
s_cmp_eq_i32 s19, 0 // (op 1962) 
s_cselect_b32 s49, 0, -1 // (op 1962) 

s_and_b32 s19, s14, 31 // (op 1962) 
s_lshl_b32 s50, 1, s19 // (op 1962) 
s_add_i32 s51, s50, s49 // (op 1962) 


s_and_b32 s49, s1, s51 // (op 1962) 


s_add_i32 s1, s18, s49 // (op 1962) 


s_ashr_i32 s18, s1, s19 // (op 1962) 


s_lshl_b32 s1, s14, 24 // (op 1962) 
s_ashr_i32 s19, s1, 31 // (op 1962) 

s_xor_b32 s1, s18, s19 // (op 1962) 

s_sub_i32 s18, s1, s19 // (op 1962) 


s_add_u32 s50, s4, 32 // (op 1962) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1962) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1962) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1962) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 1962) 

s_mul_i32 s1, s13, s50 // (op 1962) 

s_mul_i32 s19, s6, s1 // (op 1962) 

s_mul_i32 s1, s18, s19 // (op 1962) 

s_sub_u32 s19, s16, s1 // (op 1962) 

s_mov_b32 s1, s19 // (op 1962) 
s_mul_hi_i32 s49, s1, s12 // (op 1962) 
s_add_i32 s50, s49, s1 // (op 1962) 


s_ashr_i32 s1, s50, 31 // (op 1962) 
s_cmp_eq_i32 s12, 0 // (op 1962) 
s_mov_b32 s49, scc // (op 1962) 
s_cmp_eq_i32 s49, 0 // (op 1962) 
s_cselect_b32 s51, 0, -1 // (op 1962) 

s_and_b32 s49, s48, 31 // (op 1962) 
s_lshl_b32 s68, 1, s49 // (op 1962) 
s_add_i32 s69, s68, s51 // (op 1962) 


s_and_b32 s51, s1, s69 // (op 1962) 


s_add_i32 s1, s50, s51 // (op 1962) 


s_ashr_i32 s50, s1, s49 // (op 1962) 


s_lshl_b32 s1, s48, 24 // (op 1962) 
s_ashr_i32 s49, s1, 31 // (op 1962) 

s_xor_b32 s1, s50, s49 // (op 1962) 

s_sub_i32 s50, s1, s49 // (op 1962) 


s_mul_i32 s1, s15, s13 // (op 1962) 
s_add_u32 s68, s20, 128 // (op 1962) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 1962) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1962) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1962) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 7 // (op 1962) 

s_sub_i32 s49, s68, s1 // (op 1962) 

s_mul_i32 s51, s50, s49 // (op 1962) 


s_sub_u32 s49, s19, s51 // (op 1962) 


s_add_u32 s19, s49, s1 // (op 1962) 


s_cmp_eq_i32 s18, 0 // (op 1962) 
s_mov_b32 s1, scc // (op 1962) 

s_cmp_eq_i32 s1, 0 // (op 1962) 
s_cselect_b32 s18, s19, s0 // (op 1962) 



v_lshl_add_u32 v18, s18, 7, v19 // (op 1962) 


v_mul_lo_u32 v19, v21, s38 // (op 1962) 

v_add_u32 v21, v18, v19 // (op 1962) 


s_mul_i32 s1, s38, s43 // (op 1964) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1964) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1964) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1964) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1964) most significant: sum
s_add_u32 s1, s1, s19 // (op 1964) most significant: sum


s_mov_b32 s64, s40 // (op 1964) reg expression
s_mov_b32 s65, s41 // (op 1964) reg expression
s_mov_b32 s66, s0 // (op 1964) reg expression
s_mov_b32 s67, 131072 // (op 1964) call()

s_mul_hi_i32 s0, s16, s8 // (op 2007) 
v_lshrrev_b32 v18, 6, v1 // (op 2007) 
s_add_i32 s1, s0, s16 // (op 2007) 

v_and_b32 v19, 1, v18 // (op 2007) 

s_ashr_i32 s0, s1, 31 // (op 2007) 
v_and_b32 v18, 15, v1 // (op 2007) 
s_cmp_eq_i32 s8, 0 // (op 2007) 
s_mov_b32 s18, scc // (op 2007) 
v_lshl_add_u32 v22, v19, 4, v18 // (op 2007) 


s_cmp_eq_i32 s18, 0 // (op 2007) 
s_cselect_b32 s19, 0, -1 // (op 2007) 

v_and_b32 v18, 63, v1 // (op 2007) 
s_and_b32 s18, s9, 31 // (op 2007) 
v_lshrrev_b32 v19, 4, v18 // (op 2007) 

s_lshl_b32 s49, 1, s18 // (op 2007) 
s_add_i32 s50, s49, s19 // (op 2007) 


s_and_b32 s19, s0, s50 // (op 2007) 


s_add_i32 s0, s1, s19 // (op 2007) 


s_ashr_i32 s1, s0, s18 // (op 2007) 


s_lshl_b32 s0, s9, 24 // (op 2007) 
s_ashr_i32 s18, s0, 31 // (op 2007) 

s_xor_b32 s0, s1, s18 // (op 2007) 

s_sub_i32 s1, s0, s18 // (op 2007) 


s_mul_hi_i32 s0, s1, s10 // (op 2007) 
s_add_i32 s18, s0, s1 // (op 2007) 

s_ashr_i32 s0, s18, 31 // (op 2007) 
s_cmp_eq_i32 s10, 0 // (op 2007) 
s_mov_b32 s19, scc // (op 2007) 
s_cmp_eq_i32 s19, 0 // (op 2007) 
s_cselect_b32 s49, 0, -1 // (op 2007) 

s_and_b32 s19, s11, 31 // (op 2007) 
s_lshl_b32 s50, 1, s19 // (op 2007) 
s_add_i32 s51, s50, s49 // (op 2007) 


s_and_b32 s49, s0, s51 // (op 2007) 


s_add_i32 s0, s18, s49 // (op 2007) 


s_ashr_i32 s18, s0, s19 // (op 2007) 


s_lshl_b32 s0, s11, 24 // (op 2007) 
s_ashr_i32 s19, s0, 31 // (op 2007) 

s_xor_b32 s0, s18, s19 // (op 2007) 

s_sub_i32 s18, s0, s19 // (op 2007) 


s_add_u32 s50, s4, 32 // (op 2007) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 2007) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 2007) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 2007) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 2007) 

s_mul_i32 s0, s18, s50 // (op 2007) 

s_sub_i32 s49, s1, s0 // (op 2007) 


s_mul_hi_i32 s0, s16, s7 // (op 2007) 
s_add_i32 s1, s0, s16 // (op 2007) 

s_ashr_i32 s0, s1, 31 // (op 2007) 
s_cmp_eq_i32 s7, 0 // (op 2007) 
s_mov_b32 s18, scc // (op 2007) 
s_cmp_eq_i32 s18, 0 // (op 2007) 
s_cselect_b32 s19, 0, -1 // (op 2007) 

s_and_b32 s18, s14, 31 // (op 2007) 
s_lshl_b32 s68, 1, s18 // (op 2007) 
s_add_i32 s69, s68, s19 // (op 2007) 


s_and_b32 s19, s0, s69 // (op 2007) 


s_add_i32 s0, s1, s19 // (op 2007) 


s_ashr_i32 s1, s0, s18 // (op 2007) 


s_lshl_b32 s0, s14, 24 // (op 2007) 
s_ashr_i32 s18, s0, 31 // (op 2007) 

s_xor_b32 s0, s1, s18 // (op 2007) 

s_sub_i32 s1, s0, s18 // (op 2007) 


s_mul_i32 s0, s13, s50 // (op 2007) 

s_mul_i32 s18, s6, s0 // (op 2007) 

s_mul_i32 s0, s1, s18 // (op 2007) 

s_sub_u32 s18, s16, s0 // (op 2007) 

s_mul_hi_i32 s0, s18, s12 // (op 2007) 
s_add_i32 s19, s0, s18 // (op 2007) 


s_ashr_i32 s0, s19, 31 // (op 2007) 
s_cmp_eq_i32 s12, 0 // (op 2007) 
s_mov_b32 s18, scc // (op 2007) 
s_cmp_eq_i32 s18, 0 // (op 2007) 
s_cselect_b32 s50, 0, -1 // (op 2007) 

s_and_b32 s18, s48, 31 // (op 2007) 
s_lshl_b32 s51, 1, s18 // (op 2007) 
s_add_i32 s68, s51, s50 // (op 2007) 


s_and_b32 s50, s0, s68 // (op 2007) 


s_add_i32 s0, s19, s50 // (op 2007) 


s_ashr_i32 s19, s0, s18 // (op 2007) 


s_lshl_b32 s0, s48, 24 // (op 2007) 
s_ashr_i32 s18, s0, 31 // (op 2007) 

s_xor_b32 s0, s19, s18 // (op 2007) 

s_sub_i32 s19, s0, s18 // (op 2007) 


s_cmp_eq_i32 s1, 0 // (op 2007) 
s_mov_b32 s0, scc // (op 2007) 

s_cmp_eq_i32 s0, 0 // (op 2007) 
s_cselect_b32 s1, s19, s49 // (op 2007) 



v_lshl_add_u32 v18, s1, 5, v22 // (op 2007) 


v_mul_lo_u32 v22, v18, s44 // (op 2007) 

v_add_u32 v18, v19, v22 // (op 2007) 


s_mul_i32 s1, s44, s5 // (op 2009) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 2009) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 2009) least significant
s_mul_hi_u32 s19, s44, s4 // (op 2009) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2009) most significant: sum
s_add_u32 s1, s1, s19 // (op 2009) most significant: sum


s_mov_b32 s68, s46 // (op 2009) reg expression
s_mov_b32 s69, s47 // (op 2009) reg expression
s_mov_b32 s70, s0 // (op 2009) reg expression
s_mov_b32 s71, 131072 // (op 2009) call()

s_mul_hi_i32 s0, s16, s8 // (op 2026) 
v_lshrrev_b32 v19, 6, v1 // (op 2026) 
s_add_i32 s1, s0, s16 // (op 2026) 

v_and_b32 v22, 1, v19 // (op 2026) 

s_ashr_i32 s0, s1, 31 // (op 2026) 
v_and_b32 v19, 15, v1 // (op 2026) 
s_cmp_eq_i32 s8, 0 // (op 2026) 
s_mov_b32 s18, scc // (op 2026) 
v_lshl_add_u32 v23, v22, 4, v19 // (op 2026) 


s_cmp_eq_i32 s18, 0 // (op 2026) 
s_cselect_b32 s19, 0, -1 // (op 2026) 

v_and_b32 v19, 63, v1 // (op 2026) 
s_and_b32 s18, s9, 31 // (op 2026) 
v_lshrrev_b32 v22, 4, v19 // (op 2026) 

s_lshl_b32 s49, 1, s18 // (op 2026) 
s_add_i32 s50, s49, s19 // (op 2026) 


s_and_b32 s19, s0, s50 // (op 2026) 


s_add_i32 s0, s1, s19 // (op 2026) 


s_ashr_i32 s1, s0, s18 // (op 2026) 


s_lshl_b32 s0, s9, 24 // (op 2026) 
s_ashr_i32 s18, s0, 31 // (op 2026) 

s_xor_b32 s0, s1, s18 // (op 2026) 

s_sub_i32 s1, s0, s18 // (op 2026) 


s_mul_hi_i32 s0, s1, s10 // (op 2026) 
s_add_i32 s18, s0, s1 // (op 2026) 

s_ashr_i32 s0, s18, 31 // (op 2026) 
s_cmp_eq_i32 s10, 0 // (op 2026) 
s_mov_b32 s19, scc // (op 2026) 
s_cmp_eq_i32 s19, 0 // (op 2026) 
s_cselect_b32 s49, 0, -1 // (op 2026) 

s_and_b32 s19, s11, 31 // (op 2026) 
s_lshl_b32 s50, 1, s19 // (op 2026) 
s_add_i32 s51, s50, s49 // (op 2026) 


s_and_b32 s49, s0, s51 // (op 2026) 


s_add_i32 s0, s18, s49 // (op 2026) 


s_ashr_i32 s18, s0, s19 // (op 2026) 


s_lshl_b32 s0, s11, 24 // (op 2026) 
s_ashr_i32 s19, s0, 31 // (op 2026) 

s_xor_b32 s0, s18, s19 // (op 2026) 

s_sub_i32 s18, s0, s19 // (op 2026) 


s_add_u32 s50, s4, 32 // (op 2026) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 2026) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2026) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2026) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 2026) 

s_mul_i32 s0, s18, s50 // (op 2026) 

s_sub_i32 s49, s1, s0 // (op 2026) 


s_mul_hi_i32 s0, s16, s7 // (op 2026) 
s_add_i32 s1, s0, s16 // (op 2026) 

s_ashr_i32 s0, s1, 31 // (op 2026) 
s_cmp_eq_i32 s7, 0 // (op 2026) 
s_mov_b32 s18, scc // (op 2026) 
s_cmp_eq_i32 s18, 0 // (op 2026) 
s_cselect_b32 s19, 0, -1 // (op 2026) 

s_and_b32 s18, s14, 31 // (op 2026) 
s_lshl_b32 s72, 1, s18 // (op 2026) 
s_add_i32 s73, s72, s19 // (op 2026) 


s_and_b32 s19, s0, s73 // (op 2026) 


s_add_i32 s0, s1, s19 // (op 2026) 


s_ashr_i32 s1, s0, s18 // (op 2026) 


s_lshl_b32 s0, s14, 24 // (op 2026) 
s_ashr_i32 s18, s0, 31 // (op 2026) 

s_xor_b32 s0, s1, s18 // (op 2026) 

s_sub_i32 s1, s0, s18 // (op 2026) 


s_mul_i32 s0, s13, s50 // (op 2026) 

s_mul_i32 s18, s6, s0 // (op 2026) 

s_mul_i32 s0, s1, s18 // (op 2026) 

s_sub_u32 s18, s16, s0 // (op 2026) 

s_mul_hi_i32 s0, s18, s12 // (op 2026) 
s_add_i32 s19, s0, s18 // (op 2026) 


s_ashr_i32 s0, s19, 31 // (op 2026) 
s_cmp_eq_i32 s12, 0 // (op 2026) 
s_mov_b32 s18, scc // (op 2026) 
s_cmp_eq_i32 s18, 0 // (op 2026) 
s_cselect_b32 s50, 0, -1 // (op 2026) 

s_and_b32 s18, s48, 31 // (op 2026) 
s_lshl_b32 s51, 1, s18 // (op 2026) 
s_add_i32 s72, s51, s50 // (op 2026) 


s_and_b32 s50, s0, s72 // (op 2026) 


s_add_i32 s0, s19, s50 // (op 2026) 


s_ashr_i32 s19, s0, s18 // (op 2026) 


s_lshl_b32 s0, s48, 24 // (op 2026) 
s_ashr_i32 s18, s0, 31 // (op 2026) 

s_xor_b32 s0, s19, s18 // (op 2026) 

s_sub_i32 s19, s0, s18 // (op 2026) 


s_cmp_eq_i32 s1, 0 // (op 2026) 
s_mov_b32 s0, scc // (op 2026) 

s_cmp_eq_i32 s0, 0 // (op 2026) 
s_cselect_b32 s1, s19, s49 // (op 2026) 



v_lshl_add_u32 v19, s1, 5, v23 // (op 2026) 


v_mul_lo_u32 v23, v19, s44 // (op 2026) 

v_add_u32 v19, v22, v23 // (op 2026) 


s_mul_i32 s1, s44, s5 // (op 2028) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 2028) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 2028) least significant
s_mul_hi_u32 s19, s44, s4 // (op 2028) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2028) most significant: sum
s_add_u32 s1, s1, s19 // (op 2028) most significant: sum


s_mov_b32 s68, s46 // (op 2028) reg expression
s_mov_b32 s69, s47 // (op 2028) reg expression
s_mov_b32 s70, s0 // (op 2028) reg expression
s_mov_b32 s71, 131072 // (op 2028) call()




buffer_load_ubyte v22, v18, s[68:71], 0 offen offset: 0 // (op 1616) Load value
v_and_b32 v23, -8, v1 // (op 2150) 
v_sub_u32 v24, v1, v23 // (op 2150) 

v_lshlrev_b32 v23, 4, v24 // (op 2150) 

v_lshrrev_b32 v24, 3, v1 // (op 2150) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 2150) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 2150) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 1768) reg expression
buffer_load_dwordx4 v12, s[52:55], 0 offen offset: 0 lds // (op 1768) Load value direct to lds



v_mov_b32 v23, v14 // (op 1623) reg expression

buffer_load_dwordx3 v[24:26], v23, s[24:27], 0 offen // (op 1623) Load value
buffer_load_dwordx3 v[28:30], v23, s[24:27], 0 offen offset:12 // (op 1623) Load value
buffer_load_dwordx3 v[32:34], v23, s[24:27], 0 offen offset:24 // (op 1623) Load value
buffer_load_dwordx3 v[36:38], v23, s[24:27], 0 offen offset:36 // (op 1623) Load value


v_mov_b64 v[40:41], s[38:39] // (op 1626) convert

v_mov_b32 v23, v20 // (op 1626) reg expression
buffer_load_ubyte v42, v23, s[64:67], 0 offen offset: 0 // (op 1626) Load value
v_add_u32 v23, v23, v40 // (op 1626) 
buffer_load_ubyte v43, v23, s[64:67], 0 offen offset: 0 // (op 1626) Load value





buffer_load_ubyte v23, v18, s[68:71], 0 offen offset:4 // (op 1630) Load value

v_and_b32 v18, -8, v1 // (op 2294) 
v_sub_u32 v27, v1, v18 // (op 2294) 

v_lshlrev_b32 v18, 4, v27 // (op 2294) 

v_lshrrev_b32 v27, 3, v1 // (op 2294) 
v_lshl_add_u32 v31, v27, 7, v18 // (op 2294) 


s_nop 0
v_readfirstlane_b32 s0, v31 // (op 2294) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 1773) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 1773) reg expression
s_mov_b32 s18, 128 // (op 1773) call()
buffer_load_dwordx4 v12, s[52:55], s18 offen offset: 0 lds // (op 1773) Load value direct to lds






s_lshl_b64 s[0:1], s[28:29], 7 // (op 1637) 
s_mul_i32 s19, s1, 6 // (op 1637) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 1637) least significant
s_mul_hi_u32 s49, s0, 6 // (op 1637) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1637) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 1637) 

v_add_u32 v18, v14, s0 // (op 1637) 


buffer_load_dwordx3 v[44:46], v18, s[24:27], 0 offen // (op 1637) Load value
buffer_load_dwordx3 v[48:50], v18, s[24:27], 0 offen offset:12 // (op 1637) Load value
buffer_load_dwordx3 v[52:54], v18, s[24:27], 0 offen offset:24 // (op 1637) Load value
buffer_load_dwordx3 v[56:58], v18, s[24:27], 0 offen offset:36 // (op 1637) Load value



v_mov_b64 v[40:41], s[38:39] // (op 1640) convert

s_lshl_b32 s0, s38, 2 // (op 1640) 
v_add_u32 v18, v20, s0 // (op 1640) 

buffer_load_ubyte v12, v18, s[64:67], 0 offen offset: 0 // (op 1640) Load value
v_add_u32 v18, v18, v40 // (op 1640) 
buffer_load_ubyte v13, v18, s[64:67], 0 offen offset: 0 // (op 1640) Load value



v_and_b32 v14, -2, v1 // (op 2093) 
v_sub_u32 v18, v1, v14 // (op 2093) 

v_lshlrev_b32 v14, 6, v18 // (op 2093) 

v_lshrrev_b32 v18, 1, v1 // (op 2093) 
v_lshl_add_u32 v20, v18, 7, v14 // (op 2093) 


v_mul_lo_u32 v14, v20, 6 // (op 2093) 
v_lshrrev_b32 v18, 3, v14 // (op 2093) 

v_lshrrev_b32 v14, 4, v20 // (op 2093) 

v_lshlrev_b32 v20, 2, v14 // (op 2093) 

v_add_u32 v14, v18, v20 // (op 2093) 





s_waitcnt vmcnt(13)
ds_write_b96 v14, v[24:26] offset:8192 // (op 1643) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v14, v[28:30] offset:8208 // (op 1643) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v14, v[32:34] offset:8224 // (op 1643) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v14, v[36:38] offset:8240 // (op 1643) Store local data 

v_and_b32 v14, -128, v1 // (op 2203) 
v_sub_u32 v18, v1, v14 // (op 2203) 

v_lshrrev_b32 v14, 7, v1 // (op 2203) 
v_lshl_add_u32 v20, v14, 8, v18 // (op 2203) 





s_waitcnt vmcnt(9)
ds_write_b8 v20, v42 offset:24576 // (op 1644) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v20, v43 offset:24704 // (op 1644) Store local data 

v_and_b32 v14, -2, v1 // (op 2064) 
v_sub_u32 v18, v1, v14 // (op 2064) 

v_lshlrev_b32 v14, 6, v18 // (op 2064) 

v_lshrrev_b32 v18, 1, v1 // (op 2064) 
v_lshl_add_u32 v20, v18, 7, v14 // (op 2064) 


v_mul_lo_u32 v14, v20, 6 // (op 2064) 
v_lshrrev_b32 v18, 3, v14 // (op 2064) 

v_lshrrev_b32 v14, 4, v20 // (op 2064) 

v_lshlrev_b32 v20, 2, v14 // (op 2064) 

v_add_u32 v14, v18, v20 // (op 2064) 


v_and_b32 v18, -8, v1 // (op 2122) 
v_sub_u32 v20, v1, v18 // (op 2122) 

v_lshlrev_b32 v18, 4, v20 // (op 2122) 

v_lshrrev_b32 v20, 3, v1 // (op 2122) 
v_lshl_add_u32 v27, v20, 7, v18 // (op 2122) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2122) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -128, v1 // (op 2174) 
v_sub_u32 v20, v1, v18 // (op 2174) 

v_lshrrev_b32 v18, 7, v1 // (op 2174) 
v_lshl_add_u32 v27, v18, 8, v20 // (op 2174) 


v_and_b32 v18, -2, v1 // (op 2250) 
v_sub_u32 v20, v1, v18 // (op 2250) 

v_lshlrev_b32 v18, 6, v20 // (op 2250) 

v_lshrrev_b32 v20, 1, v1 // (op 2250) 
v_lshl_add_u32 v31, v20, 7, v18 // (op 2250) 


v_mul_lo_u32 v18, v31, 6 // (op 2250) 
v_lshrrev_b32 v20, 3, v18 // (op 2250) 

v_lshrrev_b32 v18, 4, v31 // (op 2250) 

v_lshlrev_b32 v31, 2, v18 // (op 2250) 

v_add_u32 v18, v20, v31 // (op 2250) 


v_and_b32 v20, -8, v1 // (op 2279) 
v_sub_u32 v31, v1, v20 // (op 2279) 

v_lshlrev_b32 v20, 4, v31 // (op 2279) 

v_lshrrev_b32 v31, 3, v1 // (op 2279) 
v_lshl_add_u32 v35, v31, 7, v20 // (op 2279) 


s_nop 0
v_readfirstlane_b32 s1, v35 // (op 2279) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -128, v1 // (op 2318) 
v_sub_u32 v31, v1, v20 // (op 2318) 

v_lshrrev_b32 v20, 7, v1 // (op 2318) 
v_lshl_add_u32 v35, v20, 8, v31 // (op 2318) 


s_mov_b32 s18, 0 // (op 639) call()
s_ashr_i64 s[50:51], s[32:33], 7 // (op 638) 
s_mov_b32 s19, s50 // (op 638) convert

s_and_b32 s49, s19, 4294967294 // (op 638) 

s_cmp_lt_u32 s18, s49 // (op 638) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoop_638 // (op 638) Condition: Top (jump to RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoop_638 if false)
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoop_638:
s_add_u32 s18, s18, 2 // (op 640) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2382) 



ds_read_b96_tr_b6 v[60:62], v0 offset:8192 // (op 67) Transpose load local data 
ds_read_b96_tr_b6 v[63:65], v0 offset:10240 // (op 67) Transpose load local data 
s_barrier  // (op 2395) 



ds_read_b128 v[66:69], v3 // (op 72) Load local data 
ds_read_b128 v[70:73], v3 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2408) 


ds_read_u8 v20, v7 offset:24576 // (op 77) Load local data 



ds_read_b96_tr_b6 v[74:76], v0 offset:8224 // (op 344) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:65], v[66:73], a[0:3], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 84) 
ds_read_b96_tr_b6 v[77:79], v0 offset:10272 // (op 344) Transpose load local data 




ds_read_u8 v20, v7 offset:24608 // (op 348) Load local data 



ds_read_b96_tr_b6 v[60:62], v0 offset:8256 // (op 506) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:79], v[66:73], a[4:7], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 350) 
ds_read_b96_tr_b6 v[63:65], v0 offset:10304 // (op 506) Transpose load local data 




ds_read_u8 v20, v7 offset:24640 // (op 510) Load local data 



ds_read_b96_tr_b6 v[74:76], v0 offset:8288 // (op 668) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[66:73], a[8:11], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 512) 
ds_read_b96_tr_b6 v[77:79], v0 offset:10336 // (op 668) Transpose load local data 




ds_read_u8 v20, v7 offset:24672 // (op 672) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[74:79], v[66:73], a[12:15], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 674) 





v_mov_b32 v20, v19 // (op 82) 

buffer_load_ubyte v22, v20, s[68:71], 0 offen offset:8 // (op 82) Load value
s_barrier  // (op 2398) 


v_mov_b64 v[40:41], v[16:17] // (op 1751) 

s_mov_b32 m0, s0 // (op 1751) reg expression
s_mov_b32 s19, 256 // (op 1751) call()
buffer_load_dwordx4 v40, s[52:55], s19 offen offset: 0 lds // (op 1751) Load value direct to lds



v_mov_b32 v31, v15 // (op 4) 
s_lshl_b64 s[50:51], s[28:29], 7 // (op 4) 
s_mul_i32 s73, s51, 6 // (op 4) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 4) least significant
s_mul_hi_u32 s19, s50, 6 // (op 4) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 4) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 4) 

s_lshl_b32 s19, s50, 1 // (op 4) 

v_add_u32 v39, v31, s19 // (op 4) 


buffer_load_dwordx3 v[24:26], v39, s[24:27], 0 offen // (op 4) Load value
buffer_load_dwordx3 v[28:30], v39, s[24:27], 0 offen offset:12 // (op 4) Load value
buffer_load_dwordx3 v[32:34], v39, s[24:27], 0 offen offset:24 // (op 4) Load value
buffer_load_dwordx3 v[36:38], v39, s[24:27], 0 offen offset:36 // (op 4) Load value


v_mov_b64 v[60:61], s[38:39] // (op 17) convert

v_mov_b32 v39, v21 // (op 17) 
s_lshl_b32 s19, s38, 3 // (op 17) 
v_add_u32 v47, v39, s19 // (op 17) 

buffer_load_ubyte v42, v47, s[64:67], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v47, v47, v60 // (op 17) 
buffer_load_ubyte v43, v47, s[64:67], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 2425) 



s_waitcnt vmcnt(13)
ds_write_b96 v18, v[44:46] offset:25088 // (op 229) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v18, v[48:50] offset:25104 // (op 229) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v18, v[52:54] offset:25120 // (op 229) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v18, v[56:58] offset:25136 // (op 229) Store local data 
s_barrier  // (op 2437) 



s_waitcnt vmcnt(9)
ds_write_b8 v35, v12 offset:41472 // (op 240) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v35, v13 offset:41600 // (op 240) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2422) 



ds_read_b96_tr_b6 v[60:62], v10 offset:25088 // (op 231) Transpose load local data 
ds_read_b96_tr_b6 v[63:65], v10 offset:27136 // (op 231) Transpose load local data 
s_barrier  // (op 2428) 



ds_read_b128 v[66:69], v4 offset:4096 // (op 233) Load local data 
ds_read_b128 v[70:73], v4 offset:4160 // (op 233) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2434) 


ds_read_u8 v47, v11 offset:41472 // (op 235) Load local data 



ds_read_b96_tr_b6 v[74:76], v10 offset:25120 // (op 357) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:65], v[66:73], a[0:3], v47, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 238) 
ds_read_b96_tr_b6 v[77:79], v10 offset:27168 // (op 357) Transpose load local data 




ds_read_u8 v47, v11 offset:41504 // (op 361) Load local data 



ds_read_b96_tr_b6 v[60:62], v10 offset:25152 // (op 519) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:79], v[66:73], a[4:7], v47, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 351) 
ds_read_b96_tr_b6 v[63:65], v10 offset:27200 // (op 519) Transpose load local data 




ds_read_u8 v47, v11 offset:41536 // (op 523) Load local data 



ds_read_b96_tr_b6 v[74:76], v10 offset:25184 // (op 681) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[66:73], a[8:11], v47, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 513) 
ds_read_b96_tr_b6 v[77:79], v10 offset:27232 // (op 681) Transpose load local data 




s_barrier  // (op 2514) 
ds_read_u8 v47, v11 offset:41568 // (op 685) Load local data 
s_barrier  // (op 2518) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[74:79], v[66:73], a[12:15], v47, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 675) 






buffer_load_ubyte v23, v20, s[68:71], 0 offen offset:12 // (op 237) Load value

s_barrier  // (op 2431) 
v_add_u32 v19, 8, v19 // (op 2018) 


s_add_u32 s19, 4096, s1 // (op 1763) 

s_mov_b32 m0, s19 // (op 1763) reg expression
s_mov_b32 s49, 384 // (op 1763) call()
buffer_load_dwordx4 v40, s[52:55], s49 offen offset: 0 lds // (op 1763) Load value direct to lds





v_add_co_u32 v16, vcc, 256, v16 // (op 1890) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1890) most significant half
s_lshl_b64 s[50:51], s[28:29], 7 // (op 228) 
s_mul_i32 s73, s51, 6 // (op 228) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 228) least significant
s_mul_hi_u32 s19, s50, 6 // (op 228) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 228) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 228) 

s_mul_i32 s19, 3, s50 // (op 228) 

v_add_u32 v20, v31, s19 // (op 228) 


buffer_load_dwordx3 v[44:46], v20, s[24:27], 0 offen // (op 228) Load value
buffer_load_dwordx3 v[48:50], v20, s[24:27], 0 offen offset:12 // (op 228) Load value
buffer_load_dwordx3 v[52:54], v20, s[24:27], 0 offen offset:24 // (op 228) Load value
buffer_load_dwordx3 v[56:58], v20, s[24:27], 0 offen offset:36 // (op 228) Load value



v_mov_b64 v[40:41], s[38:39] // (op 242) convert

s_lshl_b32 s19, s38, 2 // (op 242) 
s_mul_i32 s49, 3, s19 // (op 242) 

v_add_u32 v20, v39, s49 // (op 242) 

s_lshl_b64 s[50:51], s[28:29], 7 // (op 1826) 
s_mul_i32 s73, s51, 6 // (op 1826) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 1826) least significant
s_mul_hi_u32 s19, s50, 6 // (op 1826) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 1826) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 1826) 

s_lshl_b32 s19, s50, 1 // (op 1826) 

v_add_u32 v15, v15, s19 // (op 1826) 

buffer_load_ubyte v12, v20, s[64:67], 0 offen offset: 0 // (op 242) Load value
v_add_u32 v20, v20, v40 // (op 242) 
buffer_load_ubyte v13, v20, s[64:67], 0 offen offset: 0 // (op 242) Load value



s_barrier  // (op 2385) 
s_lshl_b32 s19, s38, 3 // (op 1954) 



v_add_u32 v21, v21, s19 // (op 1954) 

s_waitcnt vmcnt(13)
ds_write_b96 v14, v[24:26] offset:8192 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v14, v[28:30] offset:8208 // (op 6) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v14, v[32:34] offset:8224 // (op 6) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v14, v[36:38] offset:8240 // (op 6) Store local data 
s_barrier  // (op 2411) 



s_waitcnt vmcnt(9)
ds_write_b8 v27, v42 offset:24576 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v27, v43 offset:24704 // (op 21) Store local data 
s_ashr_i64 s[50:51], s[32:33], 7 // (op 638) 
s_mov_b32 s19, s50 // (op 638) convert

s_and_b32 s49, s19, 4294967294 // (op 638) 

s_cmp_lt_u32 s18, s49 // (op 638) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoop_638 // (op 638) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoop_638 if true)
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoop_638:














s_barrier  // (op 2498) 
s_mul_hi_i32 s0, s16, s8 // (op 1786) 
v_and_b32 v47, -2, v1 // (op 1786) 
s_add_i32 s1, s0, s16 // (op 1786) 

v_sub_u32 v51, v1, v47 // (op 1786) 

s_ashr_i32 s0, s1, 31 // (op 1786) 
v_lshlrev_b32 v47, 6, v51 // (op 1786) 

s_cmp_eq_i32 s8, 0 // (op 1786) 
s_mov_b32 s18, scc // (op 1786) 
v_lshrrev_b32 v51, 1, v1 // (op 1786) 
s_cmp_eq_i32 s18, 0 // (op 1786) 
s_cselect_b32 s19, 0, -1 // (op 1786) 

s_and_b32 s18, s9, 31 // (op 1786) 
s_lshl_b32 s49, 1, s18 // (op 1786) 
s_add_i32 s50, s49, s19 // (op 1786) 


s_and_b32 s19, s0, s50 // (op 1786) 


s_add_i32 s0, s1, s19 // (op 1786) 


s_ashr_i32 s1, s0, s18 // (op 1786) 


s_lshl_b32 s0, s9, 24 // (op 1786) 
s_ashr_i32 s18, s0, 31 // (op 1786) 

s_xor_b32 s0, s1, s18 // (op 1786) 

s_sub_i32 s1, s0, s18 // (op 1786) 


s_mul_hi_i32 s0, s1, s10 // (op 1786) 
s_add_i32 s18, s0, s1 // (op 1786) 

s_ashr_i32 s0, s18, 31 // (op 1786) 
s_cmp_eq_i32 s10, 0 // (op 1786) 
s_mov_b32 s19, scc // (op 1786) 
s_cmp_eq_i32 s19, 0 // (op 1786) 
s_cselect_b32 s49, 0, -1 // (op 1786) 

s_and_b32 s19, s11, 31 // (op 1786) 
s_lshl_b32 s50, 1, s19 // (op 1786) 
s_add_i32 s51, s50, s49 // (op 1786) 


s_and_b32 s49, s0, s51 // (op 1786) 


s_add_i32 s0, s18, s49 // (op 1786) 


s_ashr_i32 s18, s0, s19 // (op 1786) 


s_lshl_b32 s0, s11, 24 // (op 1786) 
s_ashr_i32 s19, s0, 31 // (op 1786) 

s_xor_b32 s0, s18, s19 // (op 1786) 

s_sub_i32 s18, s0, s19 // (op 1786) 


s_mul_i32 s0, s1, s13 // (op 1786) 

s_sub_u32 s1, s16, s0 // (op 1786) 

s_mul_i32 s0, s18, s13 // (op 1786) 
s_add_u32 s0, s0, s1 // (op 1786) 


s_mul_hi_i32 s1, s16, s7 // (op 1786) 
s_add_i32 s18, s1, s16 // (op 1786) 

s_ashr_i32 s1, s18, 31 // (op 1786) 
s_cmp_eq_i32 s7, 0 // (op 1786) 
s_mov_b32 s19, scc // (op 1786) 
s_cmp_eq_i32 s19, 0 // (op 1786) 
s_cselect_b32 s49, 0, -1 // (op 1786) 

s_and_b32 s19, s14, 31 // (op 1786) 
s_lshl_b32 s50, 1, s19 // (op 1786) 
s_add_i32 s51, s50, s49 // (op 1786) 


s_and_b32 s49, s1, s51 // (op 1786) 


s_add_i32 s1, s18, s49 // (op 1786) 


s_ashr_i32 s18, s1, s19 // (op 1786) 


s_lshl_b32 s1, s14, 24 // (op 1786) 
s_ashr_i32 s19, s1, 31 // (op 1786) 

s_xor_b32 s1, s18, s19 // (op 1786) 

s_sub_i32 s18, s1, s19 // (op 1786) 


s_add_u32 s50, s4, 32 // (op 1786) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1786) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1786) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1786) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1786) 

s_mul_i32 s1, s13, s50 // (op 1786) 

s_mul_i32 s19, s6, s1 // (op 1786) 

s_mul_i32 s1, s18, s19 // (op 1786) 

s_sub_u32 s19, s16, s1 // (op 1786) 

s_mov_b32 s1, s19 // (op 1786) 
s_mul_hi_i32 s49, s1, s12 // (op 1786) 
s_add_i32 s50, s49, s1 // (op 1786) 


s_ashr_i32 s1, s50, 31 // (op 1786) 
s_cmp_eq_i32 s12, 0 // (op 1786) 
s_mov_b32 s49, scc // (op 1786) 
s_cmp_eq_i32 s49, 0 // (op 1786) 
s_cselect_b32 s51, 0, -1 // (op 1786) 

s_and_b32 s49, s48, 31 // (op 1786) 
s_lshl_b32 s72, 1, s49 // (op 1786) 
s_add_i32 s73, s72, s51 // (op 1786) 


s_and_b32 s51, s1, s73 // (op 1786) 


s_add_i32 s1, s50, s51 // (op 1786) 


s_ashr_i32 s50, s1, s49 // (op 1786) 


s_lshl_b32 s1, s48, 24 // (op 1786) 
s_ashr_i32 s49, s1, 31 // (op 1786) 

s_xor_b32 s1, s50, s49 // (op 1786) 

s_sub_i32 s50, s1, s49 // (op 1786) 


s_mul_i32 s1, s15, s13 // (op 1786) 
s_add_u32 s72, s20, 128 // (op 1786) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 1786) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1786) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1786) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 7 // (op 1786) 

s_sub_i32 s49, s72, s1 // (op 1786) 

s_mul_i32 s51, s50, s49 // (op 1786) 


s_sub_u32 s49, s19, s51 // (op 1786) 


s_add_u32 s19, s49, s1 // (op 1786) 


s_cmp_eq_i32 s18, 0 // (op 1786) 
s_mov_b32 s1, scc // (op 1786) 

s_cmp_eq_i32 s1, 0 // (op 1786) 
s_cselect_b32 s18, s19, s0 // (op 1786) 



v_lshl_add_u32 v55, s18, 7, v47 // (op 1786) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1786) 
v_mov_b32 v15, 0 // (op 1786) convert
v_mov_b32 v14, v55 // (op 1786) convert

s_lshr_b32 s18, s0, 1 // (op 1786) 

v_lshl_add_u32 v47, s18, 8, v51 // (op 1786) 


v_mul_lo_u32 v17, v47, s29 // (op 1786) most significant: low of low * high
v_mul_hi_u32 v51, v47, s28 // (op 1786) most significant: high of low * low
v_mul_lo_u32 v16, v47, s28 // (op 1786) least significant: low of low * low
v_add_u32 v17, v17, v51 // (op 1786) most significant: sum


v_add_co_u32 v18, s[0:1], v14, v16 // (op 1786) least significant half
v_addc_co_u32 v19, s[0:1], v15, v17, s[0:1] // (op 1786) most significant half



v_mul_lo_u32 v15, v19, 6 // (op 1786) most significant: low of high * low
v_mul_hi_u32 v47, v18, 6 // (op 1786) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 1786) least significant: low of low * low
v_add_u32 v15, v15, v47 // (op 1786) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1786) 

v_mov_b32 v47, v16 // (op 1786) convert

s_mul_i32 s1, s33, 6 // (op 1788) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1788) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1788) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1788) most significant: sum

s_mul_i32 s19, s28, s1 // (op 1788) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 1788) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 1788) least significant
s_mul_hi_u32 s50, s28, s0 // (op 1788) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1788) most significant: sum
s_add_u32 s19, s19, s50 // (op 1788) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1788) 

s_mov_b32 s24, s30 // (op 1788) reg expression
s_mov_b32 s25, s31 // (op 1788) reg expression
s_mov_b32 s26, s0 // (op 1788) reg expression
s_mov_b32 s27, 131072 // (op 1788) call()

s_mul_hi_i32 s0, s16, s8 // (op 1856) 
v_lshrrev_b32 v51, 3, v1 // (op 1856) 
s_add_i32 s1, s0, s16 // (op 1856) 

v_and_b32 v55, -8, v1 // (op 1856) 
s_ashr_i32 s0, s1, 31 // (op 1856) 
v_sub_u32 v14, v1, v55 // (op 1856) 

s_cmp_eq_i32 s8, 0 // (op 1856) 
s_mov_b32 s18, scc // (op 1856) 
v_lshlrev_b32 v55, 4, v14 // (op 1856) 

s_cmp_eq_i32 s18, 0 // (op 1856) 
s_cselect_b32 s19, 0, -1 // (op 1856) 

s_and_b32 s18, s9, 31 // (op 1856) 
s_lshl_b32 s49, 1, s18 // (op 1856) 
s_add_i32 s50, s49, s19 // (op 1856) 


s_and_b32 s19, s0, s50 // (op 1856) 


s_add_i32 s0, s1, s19 // (op 1856) 


s_ashr_i32 s1, s0, s18 // (op 1856) 


s_lshl_b32 s0, s9, 24 // (op 1856) 
s_ashr_i32 s18, s0, 31 // (op 1856) 

s_xor_b32 s0, s1, s18 // (op 1856) 

s_sub_i32 s1, s0, s18 // (op 1856) 


s_mul_hi_i32 s0, s1, s10 // (op 1856) 
s_add_i32 s18, s0, s1 // (op 1856) 

s_ashr_i32 s0, s18, 31 // (op 1856) 
s_cmp_eq_i32 s10, 0 // (op 1856) 
s_mov_b32 s19, scc // (op 1856) 
s_cmp_eq_i32 s19, 0 // (op 1856) 
s_cselect_b32 s49, 0, -1 // (op 1856) 

s_and_b32 s19, s11, 31 // (op 1856) 
s_lshl_b32 s50, 1, s19 // (op 1856) 
s_add_i32 s51, s50, s49 // (op 1856) 


s_and_b32 s49, s0, s51 // (op 1856) 


s_add_i32 s0, s18, s49 // (op 1856) 


s_ashr_i32 s18, s0, s19 // (op 1856) 


s_lshl_b32 s0, s11, 24 // (op 1856) 
s_ashr_i32 s19, s0, 31 // (op 1856) 

s_xor_b32 s0, s18, s19 // (op 1856) 

s_sub_i32 s18, s0, s19 // (op 1856) 


s_add_u32 s50, s4, 32 // (op 1856) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1856) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1856) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1856) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1856) 

s_mul_i32 s0, s18, s50 // (op 1856) 

s_sub_i32 s49, s1, s0 // (op 1856) 


s_mul_hi_i32 s0, s16, s7 // (op 1856) 
s_add_i32 s1, s0, s16 // (op 1856) 

s_ashr_i32 s0, s1, 31 // (op 1856) 
s_cmp_eq_i32 s7, 0 // (op 1856) 
s_mov_b32 s18, scc // (op 1856) 
s_cmp_eq_i32 s18, 0 // (op 1856) 
s_cselect_b32 s19, 0, -1 // (op 1856) 

s_and_b32 s18, s14, 31 // (op 1856) 
s_lshl_b32 s72, 1, s18 // (op 1856) 
s_add_i32 s73, s72, s19 // (op 1856) 


s_and_b32 s19, s0, s73 // (op 1856) 


s_add_i32 s0, s1, s19 // (op 1856) 


s_ashr_i32 s1, s0, s18 // (op 1856) 


s_lshl_b32 s0, s14, 24 // (op 1856) 
s_ashr_i32 s18, s0, 31 // (op 1856) 

s_xor_b32 s0, s1, s18 // (op 1856) 

s_sub_i32 s1, s0, s18 // (op 1856) 


s_mul_i32 s0, s13, s50 // (op 1856) 

s_mul_i32 s18, s6, s0 // (op 1856) 

s_mul_i32 s0, s1, s18 // (op 1856) 

s_sub_u32 s18, s16, s0 // (op 1856) 

s_mul_hi_i32 s0, s18, s12 // (op 1856) 
s_add_i32 s19, s0, s18 // (op 1856) 


s_ashr_i32 s0, s19, 31 // (op 1856) 
s_cmp_eq_i32 s12, 0 // (op 1856) 
s_mov_b32 s18, scc // (op 1856) 
s_cmp_eq_i32 s18, 0 // (op 1856) 
s_cselect_b32 s50, 0, -1 // (op 1856) 

s_and_b32 s18, s48, 31 // (op 1856) 
s_lshl_b32 s51, 1, s18 // (op 1856) 
s_add_i32 s72, s51, s50 // (op 1856) 


s_and_b32 s50, s0, s72 // (op 1856) 


s_add_i32 s0, s19, s50 // (op 1856) 


s_ashr_i32 s19, s0, s18 // (op 1856) 


s_lshl_b32 s0, s48, 24 // (op 1856) 
s_ashr_i32 s18, s0, 31 // (op 1856) 

s_xor_b32 s0, s19, s18 // (op 1856) 

s_sub_i32 s19, s0, s18 // (op 1856) 


s_cmp_eq_i32 s1, 0 // (op 1856) 
s_mov_b32 s0, scc // (op 1856) 

s_cmp_eq_i32 s0, 0 // (op 1856) 
s_cselect_b32 s1, s19, s49 // (op 1856) 



v_lshl_add_u32 v14, s1, 5, v51 // (op 1856) 


v_mul_lo_u32 v25, v14, s35 // (op 1856) most significant: low of low * high
v_mul_hi_u32 v51, v14, s34 // (op 1856) most significant: high of low * low
v_mul_lo_u32 v24, v14, s34 // (op 1856) least significant: low of low * low
v_add_u32 v25, v25, v51 // (op 1856) most significant: sum


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1856) 
s_lshr_b32 s18, s0, 1 // (op 1856) 

v_lshl_add_u32 v51, s18, 8, v55 // (op 1856) 


v_mov_b32 v15, 0 // (op 1856) convert
v_mov_b32 v14, v51 // (op 1856) convert

v_add_co_u32 v16, s[0:1], v14, v24 // (op 1856) least significant half
v_addc_co_u32 v17, s[0:1], v15, v25, s[0:1] // (op 1856) most significant half



s_mul_i32 s1, s34, s5 // (op 1858) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 1858) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 1858) least significant
s_mul_hi_u32 s19, s34, s4 // (op 1858) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1858) most significant: sum
s_add_u32 s1, s1, s19 // (op 1858) most significant: sum


s_mov_b32 s52, s36 // (op 1858) reg expression
s_mov_b32 s53, s37 // (op 1858) reg expression
s_mov_b32 s54, s0 // (op 1858) reg expression
s_mov_b32 s55, 131072 // (op 1858) call()

s_mul_hi_i32 s0, s16, s8 // (op 1920) 
v_and_b32 v51, -128, v1 // (op 1920) 
s_add_i32 s1, s0, s16 // (op 1920) 

v_sub_u32 v55, v1, v51 // (op 1920) 

s_ashr_i32 s0, s1, 31 // (op 1920) 
v_lshrrev_b32 v51, 7, v1 // (op 1920) 
s_cmp_eq_i32 s8, 0 // (op 1920) 
s_mov_b32 s18, scc // (op 1920) 
v_lshlrev_b32 v14, 1, v51 // (op 1920) 

s_cmp_eq_i32 s18, 0 // (op 1920) 
s_cselect_b32 s19, 0, -1 // (op 1920) 

s_and_b32 s18, s9, 31 // (op 1920) 
s_lshl_b32 s49, 1, s18 // (op 1920) 
s_add_i32 s50, s49, s19 // (op 1920) 


s_and_b32 s19, s0, s50 // (op 1920) 


s_add_i32 s0, s1, s19 // (op 1920) 


s_ashr_i32 s1, s0, s18 // (op 1920) 


s_lshl_b32 s0, s9, 24 // (op 1920) 
s_ashr_i32 s18, s0, 31 // (op 1920) 

s_xor_b32 s0, s1, s18 // (op 1920) 

s_sub_i32 s1, s0, s18 // (op 1920) 


s_mul_hi_i32 s0, s1, s10 // (op 1920) 
s_add_i32 s18, s0, s1 // (op 1920) 

s_ashr_i32 s0, s18, 31 // (op 1920) 
s_cmp_eq_i32 s10, 0 // (op 1920) 
s_mov_b32 s19, scc // (op 1920) 
s_cmp_eq_i32 s19, 0 // (op 1920) 
s_cselect_b32 s49, 0, -1 // (op 1920) 

s_and_b32 s19, s11, 31 // (op 1920) 
s_lshl_b32 s50, 1, s19 // (op 1920) 
s_add_i32 s51, s50, s49 // (op 1920) 


s_and_b32 s49, s0, s51 // (op 1920) 


s_add_i32 s0, s18, s49 // (op 1920) 


s_ashr_i32 s18, s0, s19 // (op 1920) 


s_lshl_b32 s0, s11, 24 // (op 1920) 
s_ashr_i32 s19, s0, 31 // (op 1920) 

s_xor_b32 s0, s18, s19 // (op 1920) 

s_sub_i32 s18, s0, s19 // (op 1920) 


s_mul_i32 s0, s1, s13 // (op 1920) 

s_sub_u32 s1, s16, s0 // (op 1920) 

s_mul_i32 s0, s18, s13 // (op 1920) 
s_add_u32 s0, s0, s1 // (op 1920) 


s_mul_hi_i32 s1, s16, s7 // (op 1920) 
s_add_i32 s18, s1, s16 // (op 1920) 

s_ashr_i32 s1, s18, 31 // (op 1920) 
s_cmp_eq_i32 s7, 0 // (op 1920) 
s_mov_b32 s19, scc // (op 1920) 
s_cmp_eq_i32 s19, 0 // (op 1920) 
s_cselect_b32 s49, 0, -1 // (op 1920) 

s_and_b32 s19, s14, 31 // (op 1920) 
s_lshl_b32 s50, 1, s19 // (op 1920) 
s_add_i32 s51, s50, s49 // (op 1920) 


s_and_b32 s49, s1, s51 // (op 1920) 


s_add_i32 s1, s18, s49 // (op 1920) 


s_ashr_i32 s18, s1, s19 // (op 1920) 


s_lshl_b32 s1, s14, 24 // (op 1920) 
s_ashr_i32 s19, s1, 31 // (op 1920) 

s_xor_b32 s1, s18, s19 // (op 1920) 

s_sub_i32 s18, s1, s19 // (op 1920) 


s_add_u32 s50, s4, 32 // (op 1920) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1920) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1920) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1920) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1920) 

s_mul_i32 s1, s13, s50 // (op 1920) 

s_mul_i32 s19, s6, s1 // (op 1920) 

s_mul_i32 s1, s18, s19 // (op 1920) 

s_sub_u32 s19, s16, s1 // (op 1920) 

s_mov_b32 s1, s19 // (op 1920) 
s_mul_hi_i32 s49, s1, s12 // (op 1920) 
s_add_i32 s50, s49, s1 // (op 1920) 


s_ashr_i32 s1, s50, 31 // (op 1920) 
s_cmp_eq_i32 s12, 0 // (op 1920) 
s_mov_b32 s49, scc // (op 1920) 
s_cmp_eq_i32 s49, 0 // (op 1920) 
s_cselect_b32 s51, 0, -1 // (op 1920) 

s_and_b32 s49, s48, 31 // (op 1920) 
s_lshl_b32 s72, 1, s49 // (op 1920) 
s_add_i32 s73, s72, s51 // (op 1920) 


s_and_b32 s51, s1, s73 // (op 1920) 


s_add_i32 s1, s50, s51 // (op 1920) 


s_ashr_i32 s50, s1, s49 // (op 1920) 


s_lshl_b32 s1, s48, 24 // (op 1920) 
s_ashr_i32 s49, s1, 31 // (op 1920) 

s_xor_b32 s1, s50, s49 // (op 1920) 

s_sub_i32 s50, s1, s49 // (op 1920) 


s_mul_i32 s1, s15, s13 // (op 1920) 
s_add_u32 s72, s20, 128 // (op 1920) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 1920) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1920) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1920) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 7 // (op 1920) 

s_sub_i32 s49, s72, s1 // (op 1920) 

s_mul_i32 s51, s50, s49 // (op 1920) 


s_sub_u32 s49, s19, s51 // (op 1920) 


s_add_u32 s19, s49, s1 // (op 1920) 


s_cmp_eq_i32 s18, 0 // (op 1920) 
s_mov_b32 s1, scc // (op 1920) 

s_cmp_eq_i32 s1, 0 // (op 1920) 
s_cselect_b32 s18, s19, s0 // (op 1920) 



v_lshl_add_u32 v15, s18, 7, v55 // (op 1920) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1920) 
s_lshr_b32 s18, s0, 1 // (op 1920) 

v_lshl_add_u32 v51, s18, 3, v14 // (op 1920) 


v_mul_lo_u32 v14, v51, s38 // (op 1920) 

v_add_u32 v51, v15, v14 // (op 1920) 


s_mul_i32 s1, s38, s43 // (op 1922) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1922) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1922) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1922) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1922) most significant: sum
s_add_u32 s1, s1, s19 // (op 1922) most significant: sum


s_mov_b32 s64, s40 // (op 1922) reg expression
s_mov_b32 s65, s41 // (op 1922) reg expression
s_mov_b32 s66, s0 // (op 1922) reg expression
s_mov_b32 s67, 131072 // (op 1922) call()

s_mul_hi_i32 s0, s16, s8 // (op 1984) 
v_lshrrev_b32 v55, 6, v1 // (op 1984) 
s_add_i32 s1, s0, s16 // (op 1984) 

v_and_b32 v14, 1, v55 // (op 1984) 

s_ashr_i32 s0, s1, 31 // (op 1984) 
v_and_b32 v15, 15, v1 // (op 1984) 
s_cmp_eq_i32 s8, 0 // (op 1984) 
s_mov_b32 s18, scc // (op 1984) 
v_lshl_add_u32 v55, v14, 4, v15 // (op 1984) 


s_cmp_eq_i32 s18, 0 // (op 1984) 
s_cselect_b32 s19, 0, -1 // (op 1984) 

v_and_b32 v14, 63, v1 // (op 1984) 
s_and_b32 s18, s9, 31 // (op 1984) 
v_lshrrev_b32 v15, 4, v14 // (op 1984) 

s_lshl_b32 s49, 1, s18 // (op 1984) 
s_add_i32 s50, s49, s19 // (op 1984) 


s_and_b32 s19, s0, s50 // (op 1984) 


s_add_i32 s0, s1, s19 // (op 1984) 


s_ashr_i32 s1, s0, s18 // (op 1984) 


s_lshl_b32 s0, s9, 24 // (op 1984) 
s_ashr_i32 s18, s0, 31 // (op 1984) 

s_xor_b32 s0, s1, s18 // (op 1984) 

s_sub_i32 s1, s0, s18 // (op 1984) 


s_mul_hi_i32 s0, s1, s10 // (op 1984) 
s_add_i32 s18, s0, s1 // (op 1984) 

s_ashr_i32 s0, s18, 31 // (op 1984) 
s_cmp_eq_i32 s10, 0 // (op 1984) 
s_mov_b32 s19, scc // (op 1984) 
s_cmp_eq_i32 s19, 0 // (op 1984) 
s_cselect_b32 s49, 0, -1 // (op 1984) 

s_and_b32 s19, s11, 31 // (op 1984) 
s_lshl_b32 s50, 1, s19 // (op 1984) 
s_add_i32 s51, s50, s49 // (op 1984) 


s_and_b32 s49, s0, s51 // (op 1984) 


s_add_i32 s0, s18, s49 // (op 1984) 


s_ashr_i32 s18, s0, s19 // (op 1984) 


s_lshl_b32 s0, s11, 24 // (op 1984) 
s_ashr_i32 s19, s0, 31 // (op 1984) 

s_xor_b32 s0, s18, s19 // (op 1984) 

s_sub_i32 s18, s0, s19 // (op 1984) 


s_add_u32 s50, s4, 32 // (op 1984) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1984) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1984) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1984) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1984) 

s_mul_i32 s0, s18, s50 // (op 1984) 

s_sub_i32 s49, s1, s0 // (op 1984) 


s_mul_hi_i32 s0, s16, s7 // (op 1984) 
s_add_i32 s1, s0, s16 // (op 1984) 

s_ashr_i32 s0, s1, 31 // (op 1984) 
s_cmp_eq_i32 s7, 0 // (op 1984) 
s_mov_b32 s18, scc // (op 1984) 
s_cmp_eq_i32 s18, 0 // (op 1984) 
s_cselect_b32 s19, 0, -1 // (op 1984) 

s_and_b32 s18, s14, 31 // (op 1984) 
s_lshl_b32 s72, 1, s18 // (op 1984) 
s_add_i32 s73, s72, s19 // (op 1984) 


s_and_b32 s19, s0, s73 // (op 1984) 


s_add_i32 s0, s1, s19 // (op 1984) 


s_ashr_i32 s1, s0, s18 // (op 1984) 


s_lshl_b32 s0, s14, 24 // (op 1984) 
s_ashr_i32 s18, s0, 31 // (op 1984) 

s_xor_b32 s0, s1, s18 // (op 1984) 

s_sub_i32 s1, s0, s18 // (op 1984) 


s_mul_i32 s0, s13, s50 // (op 1984) 

s_mul_i32 s18, s6, s0 // (op 1984) 

s_mul_i32 s0, s1, s18 // (op 1984) 

s_sub_u32 s18, s16, s0 // (op 1984) 

s_mul_hi_i32 s0, s18, s12 // (op 1984) 
s_add_i32 s19, s0, s18 // (op 1984) 


s_ashr_i32 s0, s19, 31 // (op 1984) 
s_cmp_eq_i32 s12, 0 // (op 1984) 
s_mov_b32 s18, scc // (op 1984) 
s_cmp_eq_i32 s18, 0 // (op 1984) 
s_cselect_b32 s50, 0, -1 // (op 1984) 

s_and_b32 s18, s48, 31 // (op 1984) 
s_lshl_b32 s51, 1, s18 // (op 1984) 
s_add_i32 s72, s51, s50 // (op 1984) 


s_and_b32 s50, s0, s72 // (op 1984) 


s_add_i32 s0, s19, s50 // (op 1984) 


s_ashr_i32 s19, s0, s18 // (op 1984) 


s_lshl_b32 s0, s48, 24 // (op 1984) 
s_ashr_i32 s18, s0, 31 // (op 1984) 

s_xor_b32 s0, s19, s18 // (op 1984) 

s_sub_i32 s19, s0, s18 // (op 1984) 


s_cmp_eq_i32 s1, 0 // (op 1984) 
s_mov_b32 s0, scc // (op 1984) 

s_cmp_eq_i32 s0, 0 // (op 1984) 
s_cselect_b32 s1, s19, s49 // (op 1984) 



v_lshl_add_u32 v14, s1, 5, v55 // (op 1984) 


v_mul_lo_u32 v55, v14, s44 // (op 1984) 

s_ashr_i64 s[0:1], s[32:33], 7 // (op 1984) 
s_lshr_b32 s18, s0, 1 // (op 1984) 

v_lshl_add_u32 v14, s18, 3, v15 // (op 1984) 


v_add_u32 v15, v14, v55 // (op 1984) 


s_mul_i32 s1, s44, s5 // (op 1986) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 1986) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 1986) least significant
s_mul_hi_u32 s19, s44, s4 // (op 1986) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1986) most significant: sum
s_add_u32 s1, s1, s19 // (op 1986) most significant: sum


s_mov_b32 s68, s46 // (op 1986) reg expression
s_mov_b32 s69, s47 // (op 1986) reg expression
s_mov_b32 s70, s0 // (op 1986) reg expression
s_mov_b32 s71, 131072 // (op 1986) call()

v_and_b32 v14, -2, v1 // (op 2077) 
v_sub_u32 v55, v1, v14 // (op 2077) 

v_lshlrev_b32 v14, 6, v55 // (op 2077) 

v_lshrrev_b32 v55, 1, v1 // (op 2077) 
v_lshl_add_u32 v18, v55, 7, v14 // (op 2077) 


v_mul_lo_u32 v14, v18, 6 // (op 2077) 
v_lshrrev_b32 v55, 3, v14 // (op 2077) 

v_lshrrev_b32 v14, 4, v18 // (op 2077) 

v_lshlrev_b32 v18, 2, v14 // (op 2077) 

v_add_u32 v14, v55, v18 // (op 2077) 


v_and_b32 v55, -8, v1 // (op 2135) 
v_sub_u32 v18, v1, v55 // (op 2135) 

v_lshlrev_b32 v55, 4, v18 // (op 2135) 

v_lshrrev_b32 v18, 3, v1 // (op 2135) 
v_lshl_add_u32 v19, v18, 7, v55 // (op 2135) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 2135) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v55, -128, v1 // (op 2187) 
v_sub_u32 v18, v1, v55 // (op 2187) 

v_lshrrev_b32 v55, 7, v1 // (op 2187) 
v_lshl_add_u32 v19, v55, 8, v18 // (op 2187) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 644) 
s_and_b32 s1, s18, 4294967294 // (op 644) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 643) 
s_mov_b32 s49, s18 // (op 643) convert

s_cmp_lt_u32 s1, s49 // (op 643) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_643 // (op 643) Condition: Top (jump to RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_643 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoopTail_643:


s_add_u32 s1, s1, 1 // (op 645) 
v_mov_b32 v18, v47 // (op 170) 
v_mov_b32 v12, v18 // (op 170) reg expression

buffer_load_dwordx3 v[20:22], v12, s[24:27], 0 offen // (op 170) Load value
buffer_load_dwordx3 v[24:26], v12, s[24:27], 0 offen offset:12 // (op 170) Load value
buffer_load_dwordx3 v[28:30], v12, s[24:27], 0 offen offset:24 // (op 170) Load value
buffer_load_dwordx3 v[32:34], v12, s[24:27], 0 offen offset:36 // (op 170) Load value


s_lshl_b64 s[18:19], s[28:29], 7 // (op 1776) 
s_mul_i32 s51, s19, 6 // (op 1776) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 1776) least significant
s_mul_hi_u32 s49, s18, 6 // (op 1776) most significant: high of low * low
s_add_u32 s51, s51, s49 // (op 1776) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 1776) 

v_add_u32 v47, v47, s18 // (op 1776) 

s_barrier  // (op 2388) 
s_barrier  // (op 2401) 





v_mov_b64 v[12:13], v[16:17] // (op 1756) 

s_mov_b32 m0, s0 // (op 1756) reg expression
buffer_load_dwordx4 v12, s[52:55], 0 offen offset: 0 lds // (op 1756) Load value direct to lds

v_add_co_u32 v16, vcc, 128, v16 // (op 1848) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1848) most significant half
s_waitcnt vmcnt(4)
ds_write_b96 v14, v[20:22] offset:8192 // (op 171) Store local data 

v_mov_b64 v[12:13], s[38:39] // (op 184) convert

v_mov_b32 v18, v51 // (op 184) 
v_mov_b32 v23, v18 // (op 184) reg expression
buffer_load_ubyte v48, v23, s[64:67], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v23, v23, v12 // (op 184) 
s_waitcnt vmcnt(4)
ds_write_b96 v14, v[24:26] offset:8208 // (op 171) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v14, v[28:30] offset:8224 // (op 171) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v14, v[32:34] offset:8240 // (op 171) Store local data 
buffer_load_ubyte v49, v23, s[64:67], 0 offen offset: 0 // (op 184) Load value




s_lshl_b32 s18, s38, 2 // (op 1912) 
v_add_u32 v51, v51, s18 // (op 1912) 

s_barrier  // (op 2414) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2391) 






ds_read_b96_tr_b6 v[20:22], v0 offset:8192 // (op 173) Transpose load local data 
ds_read_b96_tr_b6 v[23:25], v0 offset:10240 // (op 173) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v48 offset:24576 // (op 182) Store local data 
s_barrier  // (op 2404) 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v49 offset:24704 // (op 182) Store local data 




ds_read_b128 v[26:29], v3 // (op 175) Load local data 
ds_read_b128 v[30:33], v3 offset:64 // (op 175) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2418) 


ds_read_u8 v18, v7 offset:24576 // (op 177) Load local data 


v_mov_b32 v12, v15 // (op 179) 

buffer_load_ubyte v13, v12, s[68:71], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v15, 4, v15 // (op 1976) 



ds_read_b96_tr_b6 v[34:36], v0 offset:8224 // (op 325) Transpose load local data 
ds_read_b96_tr_b6 v[37:39], v0 offset:10272 // (op 325) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:33], a[0:3], v18, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 180) 




ds_read_u8 v12, v7 offset:24608 // (op 329) Load local data 



ds_read_b96_tr_b6 v[20:22], v0 offset:8256 // (op 487) Transpose load local data 
ds_read_b96_tr_b6 v[23:25], v0 offset:10304 // (op 487) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:39], v[26:33], a[4:7], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 331) 




ds_read_u8 v12, v7 offset:24640 // (op 491) Load local data 



ds_read_b96_tr_b6 v[34:36], v0 offset:8288 // (op 649) Transpose load local data 
ds_read_b96_tr_b6 v[37:39], v0 offset:10336 // (op 649) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[26:33], a[8:11], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 493) 




ds_read_u8 v12, v7 offset:24672 // (op 653) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:39], v[26:33], a[12:15], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 655) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 643) 
s_mov_b32 s49, s18 // (op 643) convert

s_cmp_lt_u32 s1, s49 // (op 643) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoopTail_643 // (op 643) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoopTail_643 if true)
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_643:






















s_barrier  // (op 2502) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
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


v_accvgpr_read v10, a4 // (op 885) 
v_accvgpr_read v11, a5 // (op 885) 
v_accvgpr_read v12, a6 // (op 885) 
v_accvgpr_read v13, a7 // (op 885) 
v_mul_f32 v14, v8, v10 // (op 885) 
v_mul_f32 v15, v8, v11 // (op 885) 
v_mul_f32 v16, v8, v12 // (op 885) 
v_mul_f32 v17, v8, v13 // (op 885) 

v_lshrrev_b32 v10, 16, v14 // (op 887) 
v_lshrrev_b32 v11, 16, v15 // (op 887) 
v_lshrrev_b32 v12, 16, v16 // (op 887) 
v_lshrrev_b32 v13, 16, v17 // (op 887) 


v_pack_b32_f16 v14, v10, v11 // (op 888) 
v_pack_b32_f16 v15, v12, v13 // (op 888) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:64 // (op 888) Store value


v_accvgpr_read v10, a8 // (op 905) 
v_accvgpr_read v11, a9 // (op 905) 
v_accvgpr_read v12, a10 // (op 905) 
v_accvgpr_read v13, a11 // (op 905) 
v_mul_f32 v14, v8, v10 // (op 905) 
v_mul_f32 v15, v8, v11 // (op 905) 
v_mul_f32 v16, v8, v12 // (op 905) 
v_mul_f32 v17, v8, v13 // (op 905) 

v_lshrrev_b32 v10, 16, v14 // (op 907) 
v_lshrrev_b32 v11, 16, v15 // (op 907) 
v_lshrrev_b32 v12, 16, v16 // (op 907) 
v_lshrrev_b32 v13, 16, v17 // (op 907) 


v_pack_b32_f16 v14, v10, v11 // (op 908) 
v_pack_b32_f16 v15, v12, v13 // (op 908) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:128 // (op 908) Store value


v_accvgpr_read v10, a12 // (op 925) 
v_accvgpr_read v11, a13 // (op 925) 
v_accvgpr_read v12, a14 // (op 925) 
v_accvgpr_read v13, a15 // (op 925) 
v_mul_f32 v14, v8, v10 // (op 925) 
v_mul_f32 v15, v8, v11 // (op 925) 
v_mul_f32 v16, v8, v12 // (op 925) 
v_mul_f32 v17, v8, v13 // (op 925) 

v_lshrrev_b32 v10, 16, v14 // (op 927) 
v_lshrrev_b32 v11, 16, v15 // (op 927) 
v_lshrrev_b32 v12, 16, v16 // (op 927) 
v_lshrrev_b32 v13, 16, v17 // (op 927) 


v_pack_b32_f16 v14, v10, v11 // (op 928) 
v_pack_b32_f16 v15, v12, v13 // (op 928) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:192 // (op 928) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[10:11], v6, s[56:59], 0 offen offset:192 // (op 1008) Load value
v_accvgpr_read v12, a12 // (op 1004) 
v_accvgpr_read v13, a13 // (op 1004) 
v_accvgpr_read v14, a14 // (op 1004) 
v_accvgpr_read v15, a15 // (op 1004) 
v_mul_f32 v16, v8, v12 // (op 1004) 
v_mul_f32 v17, v8, v13 // (op 1004) 
v_mul_f32 v18, v8, v14 // (op 1004) 
v_mul_f32 v19, v8, v15 // (op 1004) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 1009) 
v_lshrrev_b32 v13, 16, v10 // (op 1009) 
v_lshlrev_b32 v12, 16, v12 // (op 1009) 
v_lshlrev_b32 v13, 16, v13 // (op 1009) 
v_mul_f32 v20, v5, v12 // (op 1009) 
v_mul_f32 v21, v5, v13 // (op 1009) 
v_and_b32 v12, 65535, v11 // (op 1009) 
v_lshrrev_b32 v13, 16, v11 // (op 1009) 
v_lshlrev_b32 v12, 16, v12 // (op 1009) 
v_lshlrev_b32 v13, 16, v13 // (op 1009) 
v_mul_f32 v22, v5, v12 // (op 1009) 
v_mul_f32 v23, v5, v13 // (op 1009) 


v_add_f32 v10, v20, v16 // (op 1005) 
v_add_f32 v11, v21, v17 // (op 1005) 
v_add_f32 v12, v22, v18 // (op 1005) 
v_add_f32 v13, v23, v19 // (op 1005) 


v_lshrrev_b32 v14, 16, v10 // (op 1006) 
v_lshrrev_b32 v15, 16, v11 // (op 1006) 
v_lshrrev_b32 v16, 16, v12 // (op 1006) 
v_lshrrev_b32 v17, 16, v13 // (op 1006) 


v_pack_b32_f16 v10, v14, v15 // (op 1007) 
v_pack_b32_f16 v11, v16, v17 // (op 1007) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:192 // (op 1007) Store value





buffer_load_dwordx2 v[10:11], v6, s[56:59], 0 offen offset:128 // (op 988) Load value
v_accvgpr_read v12, a8 // (op 984) 
v_accvgpr_read v13, a9 // (op 984) 
v_accvgpr_read v14, a10 // (op 984) 
v_accvgpr_read v15, a11 // (op 984) 
v_mul_f32 v16, v8, v12 // (op 984) 
v_mul_f32 v17, v8, v13 // (op 984) 
v_mul_f32 v18, v8, v14 // (op 984) 
v_mul_f32 v19, v8, v15 // (op 984) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 989) 
v_lshrrev_b32 v13, 16, v10 // (op 989) 
v_lshlrev_b32 v12, 16, v12 // (op 989) 
v_lshlrev_b32 v13, 16, v13 // (op 989) 
v_mul_f32 v20, v5, v12 // (op 989) 
v_mul_f32 v21, v5, v13 // (op 989) 
v_and_b32 v12, 65535, v11 // (op 989) 
v_lshrrev_b32 v13, 16, v11 // (op 989) 
v_lshlrev_b32 v12, 16, v12 // (op 989) 
v_lshlrev_b32 v13, 16, v13 // (op 989) 
v_mul_f32 v22, v5, v12 // (op 989) 
v_mul_f32 v23, v5, v13 // (op 989) 


v_add_f32 v10, v20, v16 // (op 985) 
v_add_f32 v11, v21, v17 // (op 985) 
v_add_f32 v12, v22, v18 // (op 985) 
v_add_f32 v13, v23, v19 // (op 985) 


v_lshrrev_b32 v14, 16, v10 // (op 986) 
v_lshrrev_b32 v15, 16, v11 // (op 986) 
v_lshrrev_b32 v16, 16, v12 // (op 986) 
v_lshrrev_b32 v17, 16, v13 // (op 986) 


v_pack_b32_f16 v10, v14, v15 // (op 987) 
v_pack_b32_f16 v11, v16, v17 // (op 987) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:128 // (op 987) Store value





buffer_load_dwordx2 v[10:11], v6, s[56:59], 0 offen offset:64 // (op 968) Load value
v_accvgpr_read v12, a4 // (op 964) 
v_accvgpr_read v13, a5 // (op 964) 
v_accvgpr_read v14, a6 // (op 964) 
v_accvgpr_read v15, a7 // (op 964) 
v_mul_f32 v16, v8, v12 // (op 964) 
v_mul_f32 v17, v8, v13 // (op 964) 
v_mul_f32 v18, v8, v14 // (op 964) 
v_mul_f32 v19, v8, v15 // (op 964) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 969) 
v_lshrrev_b32 v13, 16, v10 // (op 969) 
v_lshlrev_b32 v12, 16, v12 // (op 969) 
v_lshlrev_b32 v13, 16, v13 // (op 969) 
v_mul_f32 v20, v5, v12 // (op 969) 
v_mul_f32 v21, v5, v13 // (op 969) 
v_and_b32 v12, 65535, v11 // (op 969) 
v_lshrrev_b32 v13, 16, v11 // (op 969) 
v_lshlrev_b32 v12, 16, v12 // (op 969) 
v_lshlrev_b32 v13, 16, v13 // (op 969) 
v_mul_f32 v22, v5, v12 // (op 969) 
v_mul_f32 v23, v5, v13 // (op 969) 


v_add_f32 v10, v20, v16 // (op 965) 
v_add_f32 v11, v21, v17 // (op 965) 
v_add_f32 v12, v22, v18 // (op 965) 
v_add_f32 v13, v23, v19 // (op 965) 


v_lshrrev_b32 v14, 16, v10 // (op 966) 
v_lshrrev_b32 v15, 16, v11 // (op 966) 
v_lshrrev_b32 v16, 16, v12 // (op 966) 
v_lshrrev_b32 v17, 16, v13 // (op 966) 


v_pack_b32_f16 v10, v14, v15 // (op 967) 
v_pack_b32_f16 v11, v16, v17 // (op 967) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:64 // (op 967) Store value





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
RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:












s_endpgm  // End of RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
.LRR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__end:
.size RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_, .LRR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__end-RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 41984 // lds bytes
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
    macrotile: [128, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 41984
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 80
    .agpr_count: 16
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

