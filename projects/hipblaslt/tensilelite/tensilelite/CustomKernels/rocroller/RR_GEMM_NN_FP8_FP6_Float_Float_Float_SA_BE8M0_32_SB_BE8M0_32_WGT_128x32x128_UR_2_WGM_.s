// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_,@function
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s8 // (op 2029) 
s_mul_hi_i32 s1, s16, s8 // (op 2217) 
v_lshrrev_b32 v0, 6, v1 // (op 2143) 
v_and_b32 v3, 1, v0 // (op 2143) 

v_and_b32 v0, 15, v1 // (op 2143) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 2143) 


v_and_b32 v0, 63, v1 // (op 2143) 
v_lshrrev_b32 v3, 4, v0 // (op 2143) 

v_lshlrev_b32 v0, 5, v3 // (op 2143) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 2143) 


v_mul_lo_u32 v0, v3, 6 // (op 2143) 

v_lshrrev_b32 v3, 3, v0 // (op 2143) 

v_lshrrev_b32 v0, 6, v1 // (op 2287) 
v_and_b32 v4, 1, v0 // (op 2287) 

v_and_b32 v0, 15, v1 // (op 2287) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 2287) 


v_and_b32 v0, 63, v1 // (op 2287) 
v_lshrrev_b32 v4, 4, v0 // (op 2287) 

v_lshlrev_b32 v0, 5, v4 // (op 2287) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 2287) 


v_mul_lo_u32 v0, v4, 6 // (op 2287) 

v_lshrrev_b32 v4, 3, v0 // (op 2287) 

v_lshrrev_b32 v0, 7, v1 // (op 2029) 
s_add_i32 s18, s0, s16 // (op 2029) 

v_and_b32 v5, 63, v1 // (op 2029) 
s_ashr_i32 s0, s18, 31 // (op 2029) 
v_lshrrev_b32 v6, 4, v5 // (op 2029) 
s_cmp_eq_i32 s8, 0 // (op 2029) 
s_mov_b32 s19, scc // (op 2029) 
v_lshlrev_b32 v7, 2, v6 // (op 2029) 

s_cmp_eq_i32 s19, 0 // (op 2029) 
s_cselect_b32 s51, 0, -1 // (op 2029) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 2029) 


s_and_b32 s19, s9, 31 // (op 2029) 
v_lshrrev_b32 v0, 2, v5 // (op 2029) 

v_and_b32 v5, 3, v0 // (op 2029) 

v_and_b32 v0, 48, v1 // (op 2090) 
v_and_b32 v7, 63, v1 // (op 2090) 
v_sub_u32 v8, v7, v0 // (op 2090) 

v_lshrrev_b32 v7, 1, v8 // (op 2090) 
v_add_lshl_u32 v9, v0, v7, 7 // (op 2090) 


v_and_b32 v0, 1, v8 // (op 2090) 

v_lshlrev_b32 v7, 3, v0 // (op 2090) 

v_lshrrev_b32 v0, 7, v1 // (op 2090) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 2090) 


v_add_u32 v0, v9, v8 // (op 2090) 


v_lshrrev_b32 v7, 7, v1 // (op 2201) 
v_and_b32 v8, 15, v1 // (op 2201) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 2201) 


v_and_b32 v7, 63, v1 // (op 2201) 
v_lshrrev_b32 v8, 4, v7 // (op 2201) 

v_lshl_add_u32 v7, v8, 7, v9 // (op 2201) 


v_lshrrev_b32 v8, 7, v1 // (op 2217) 
v_and_b32 v9, 63, v1 // (op 2217) 
v_and_b32 v10, 48, v1 // (op 2263) 
v_and_b32 v11, 63, v1 // (op 2263) 
v_sub_u32 v12, v11, v10 // (op 2263) 

v_lshrrev_b32 v11, 1, v12 // (op 2263) 
v_add_lshl_u32 v13, v10, v11, 7 // (op 2263) 


v_and_b32 v10, 1, v12 // (op 2263) 

v_lshlrev_b32 v11, 3, v10 // (op 2263) 

v_lshrrev_b32 v10, 7, v1 // (op 2263) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 2263) 


v_add_u32 v10, v13, v12 // (op 2263) 


v_lshrrev_b32 v11, 7, v1 // (op 2316) 
v_and_b32 v12, 15, v1 // (op 2316) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 2316) 


v_and_b32 v11, 63, v1 // (op 2316) 
v_lshrrev_b32 v12, 4, v11 // (op 2316) 

v_lshl_add_u32 v11, v12, 7, v13 // (op 2316) 


s_lshl_b32 s52, 1, s19 // (op 2029) 
v_and_b32 v12, 3, v1 // (op 2029) 
s_add_i32 s53, s52, s51 // (op 2029) 


s_and_b32 s51, s0, s53 // (op 2029) 


s_add_i32 s0, s1, s16 // (op 2217) 

s_ashr_i32 s1, s0, 31 // (op 2217) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 2029) 


s_add_i32 s52, s18, s51 // (op 2029) 


v_lshrrev_b32 v5, 6, v1 // (op 2029) 
s_ashr_i32 s18, s52, s19 // (op 2029) 


v_and_b32 v12, 1, v5 // (op 2029) 

s_lshl_b32 s19, s9, 24 // (op 2029) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 2029) 


s_ashr_i32 s51, s19, 31 // (op 2029) 

s_xor_b32 s19, s18, s51 // (op 2029) 

s_sub_i32 s18, s19, s51 // (op 2029) 


s_mul_hi_i32 s19, s18, s10 // (op 2029) 
s_add_i32 s51, s19, s18 // (op 2029) 

s_ashr_i32 s19, s51, 31 // (op 2029) 
s_cmp_eq_i32 s10, 0 // (op 2029) 
s_mov_b32 s52, scc // (op 2029) 
s_cmp_eq_i32 s52, 0 // (op 2029) 
s_cselect_b32 s53, 0, -1 // (op 2029) 

s_and_b32 s52, s11, 31 // (op 2029) 
v_lshrrev_b32 v12, 4, v9 // (op 2217) 
s_cmp_eq_i32 s8, 0 // (op 2217) 
s_mov_b32 s54, scc // (op 2217) 
v_lshlrev_b32 v13, 2, v12 // (op 2217) 

s_cmp_eq_i32 s54, 0 // (op 2217) 
s_cselect_b32 s55, 0, -1 // (op 2217) 

s_lshl_b32 s54, 1, s52 // (op 2029) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 2217) 


v_lshrrev_b32 v8, 2, v9 // (op 2217) 

s_add_i32 s56, s54, s53 // (op 2029) 


s_and_b32 s53, s19, s56 // (op 2029) 


s_add_i32 s19, s51, s53 // (op 2029) 


s_ashr_i32 s51, s19, s52 // (op 2029) 


s_lshl_b32 s19, s11, 24 // (op 2029) 
s_ashr_i32 s52, s19, 31 // (op 2029) 

s_xor_b32 s19, s51, s52 // (op 2029) 

s_sub_i32 s51, s19, s52 // (op 2029) 


s_mul_i32 s19, s18, s13 // (op 2029) 
s_sub_u32 s52, s16, s19 // (op 2029) 

s_mul_i32 s19, s51, s13 // (op 2029) 
s_add_u32 s19, s19, s52 // (op 2029) 

s_mul_hi_i32 s52, s16, s7 // (op 2029) 
s_add_i32 s53, s52, s16 // (op 2029) 

s_ashr_i32 s52, s53, 31 // (op 2029) 
s_cmp_eq_i32 s7, 0 // (op 2029) 
s_mov_b32 s54, scc // (op 2029) 
s_cmp_eq_i32 s54, 0 // (op 2029) 
s_cselect_b32 s56, 0, -1 // (op 2029) 

s_and_b32 s54, s14, 31 // (op 2029) 
s_lshl_b32 s57, 1, s54 // (op 2029) 
s_add_i32 s58, s57, s56 // (op 2029) 


s_and_b32 s56, s52, s58 // (op 2029) 


s_add_i32 s52, s53, s56 // (op 2029) 


s_ashr_i32 s53, s52, s54 // (op 2029) 


s_lshl_b32 s52, s14, 24 // (op 2029) 
s_ashr_i32 s54, s52, 31 // (op 2029) 

s_xor_b32 s52, s53, s54 // (op 2029) 

s_sub_i32 s53, s52, s54 // (op 2029) 


s_and_b32 s52, s9, 31 // (op 2217) 
s_lshl_b32 s54, 1, s52 // (op 2217) 
v_and_b32 v9, 3, v8 // (op 2217) 

s_add_i32 s56, s54, s55 // (op 2217) 


v_and_b32 v8, 3, v1 // (op 2217) 
s_and_b32 s54, s1, s56 // (op 2217) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 2217) 


s_add_i32 s1, s0, s54 // (op 2217) 


v_lshrrev_b32 v8, 6, v1 // (op 2217) 
s_ashr_i32 s0, s1, s52 // (op 2217) 


v_and_b32 v9, 1, v8 // (op 2217) 

s_lshl_b32 s1, s9, 24 // (op 2217) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 2217) 


s_ashr_i32 s52, s1, 31 // (op 2217) 

s_xor_b32 s1, s0, s52 // (op 2217) 

s_sub_i32 s0, s1, s52 // (op 2217) 


s_mul_hi_i32 s1, s0, s10 // (op 2217) 
s_add_i32 s52, s1, s0 // (op 2217) 

s_ashr_i32 s1, s52, 31 // (op 2217) 
s_cmp_eq_i32 s10, 0 // (op 2217) 
s_mov_b32 s54, scc // (op 2217) 
s_cmp_eq_i32 s54, 0 // (op 2217) 
s_cselect_b32 s55, 0, -1 // (op 2217) 

s_and_b32 s54, s11, 31 // (op 2217) 
s_lshl_b32 s56, 1, s54 // (op 2217) 
s_add_i32 s57, s56, s55 // (op 2217) 


s_and_b32 s55, s1, s57 // (op 2217) 


s_add_i32 s1, s52, s55 // (op 2217) 


s_ashr_i32 s52, s1, s54 // (op 2217) 


s_lshl_b32 s1, s11, 24 // (op 2217) 
s_ashr_i32 s54, s1, 31 // (op 2217) 

s_xor_b32 s1, s52, s54 // (op 2217) 

s_sub_i32 s52, s1, s54 // (op 2217) 


s_mul_i32 s1, s0, s13 // (op 2217) 
s_sub_u32 s54, s16, s1 // (op 2217) 

s_mul_i32 s1, s52, s13 // (op 2217) 
s_add_u32 s1, s1, s54 // (op 2217) 

s_mul_hi_i32 s54, s16, s7 // (op 2217) 
s_add_i32 s55, s54, s16 // (op 2217) 

s_ashr_i32 s54, s55, 31 // (op 2217) 
s_cmp_eq_i32 s7, 0 // (op 2217) 
s_mov_b32 s56, scc // (op 2217) 
s_cmp_eq_i32 s56, 0 // (op 2217) 
s_cselect_b32 s57, 0, -1 // (op 2217) 

s_and_b32 s56, s14, 31 // (op 2217) 
s_lshl_b32 s58, 1, s56 // (op 2217) 
s_add_i32 s59, s58, s57 // (op 2217) 


s_and_b32 s57, s54, s59 // (op 2217) 


s_add_i32 s54, s55, s57 // (op 2217) 


s_ashr_i32 s55, s54, s56 // (op 2217) 


s_lshl_b32 s54, s14, 24 // (op 2217) 
s_ashr_i32 s56, s54, 31 // (op 2217) 

s_xor_b32 s54, s55, s56 // (op 2217) 

s_sub_i32 s55, s54, s56 // (op 2217) 


s_add_u32 s56, s4, 32 // (op 2029) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 2029) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2029) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2029) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 2029) 

s_mul_i32 s54, s13, s56 // (op 2029) 
s_mul_i32 s58, s6, s54 // (op 2029) 

s_mul_i32 s54, s53, s58 // (op 2029) 

s_sub_u32 s58, s16, s54 // (op 2029) 

s_mov_b32 s54, s58 // (op 2029) 
s_mul_hi_i32 s59, s54, s12 // (op 2029) 
s_add_i32 s60, s59, s54 // (op 2029) 


s_ashr_i32 s54, s60, 31 // (op 2029) 
s_cmp_eq_i32 s12, 0 // (op 2029) 
s_mov_b32 s59, scc // (op 2029) 
s_cmp_eq_i32 s59, 0 // (op 2029) 
s_cselect_b32 s61, 0, -1 // (op 2029) 

s_add_u32 s62, s4, 32 // (op 2217) Start of Int64 add, locking SCC
s_addc_u32 s63, s5, 0 // (op 2217) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 2217) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 2217) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 5 // (op 2217) 

s_mul_i32 s59, s13, s62 // (op 2217) 
s_mul_i32 s64, s6, s59 // (op 2217) 

s_mul_i32 s59, s55, s64 // (op 2217) 

s_sub_u32 s64, s16, s59 // (op 2217) 

s_mov_b32 s59, s64 // (op 2217) 
s_mul_hi_i32 s65, s59, s12 // (op 2217) 
s_add_i32 s66, s65, s59 // (op 2217) 


s_ashr_i32 s59, s66, 31 // (op 2217) 
s_cmp_eq_i32 s12, 0 // (op 2217) 
s_mov_b32 s65, scc // (op 2217) 
s_cmp_eq_i32 s65, 0 // (op 2217) 
s_cselect_b32 s67, 0, -1 // (op 2217) 


s_waitcnt lgkmcnt(0)
s_and_b32 s65, s48, 31 // (op 2029) 
s_lshl_b32 s68, 1, s65 // (op 2029) 
s_add_i32 s69, s68, s61 // (op 2029) 


s_and_b32 s61, s54, s69 // (op 2029) 


s_add_i32 s54, s60, s61 // (op 2029) 


s_ashr_i32 s60, s54, s65 // (op 2029) 


s_lshl_b32 s54, s48, 24 // (op 2029) 
s_ashr_i32 s61, s54, 31 // (op 2029) 

s_xor_b32 s54, s60, s61 // (op 2029) 

s_sub_i32 s60, s54, s61 // (op 2029) 


s_mul_i32 s54, s15, s13 // (op 2029) 
s_and_b32 s61, s48, 31 // (op 2217) 
s_lshl_b32 s65, 1, s61 // (op 2217) 
s_add_i32 s68, s65, s67 // (op 2217) 


s_and_b32 s65, s59, s68 // (op 2217) 


s_add_i32 s59, s66, s65 // (op 2217) 


s_ashr_i32 s65, s59, s61 // (op 2217) 


s_lshl_b32 s59, s48, 24 // (op 2217) 
s_ashr_i32 s61, s59, 31 // (op 2217) 

s_xor_b32 s59, s65, s61 // (op 2217) 

s_sub_i32 s65, s59, s61 // (op 2217) 


s_mul_i32 s59, s15, s13 // (op 2217) 
s_add_u32 s66, s20, 128 // (op 2029) Start of Int64 add, locking SCC
s_addc_u32 s67, s21, 0 // (op 2029) End of Int64 add, unlocking SCC
s_sub_u32 s68, s66, 1 // (op 2029) least significant half; sets scc
s_subb_u32 s69, s67, 0 // (op 2029) most significant half; uses scc

s_ashr_i64 s[66:67], s[68:69], 7 // (op 2029) 

s_sub_i32 s61, s66, s54 // (op 2029) 

s_mul_i32 s66, s60, s61 // (op 2029) 

s_sub_u32 s61, s58, s66 // (op 2029) 


s_add_u32 s58, s61, s54 // (op 2029) 


s_cmp_eq_i32 s53, 0 // (op 2029) 
s_mov_b32 s54, scc // (op 2029) 

s_cmp_eq_i32 s54, 0 // (op 2029) 
s_cselect_b32 s53, s58, s19 // (op 2029) 


s_mul_i32 s19, s51, s56 // (op 2029) 


v_lshl_add_u32 v9, s53, 7, v6 // (op 2029) 


s_sub_i32 s51, s18, s19 // (op 2029) 


s_cmp_eq_i32 s54, 0 // (op 2029) 
s_cselect_b32 s53, s60, s51 // (op 2029) 



v_lshl_add_u32 v6, s53, 5, v5 // (op 2029) 


v_mul_lo_u32 v5, v6, s2 // (op 2029) 

v_add_lshl_u32 v6, v9, v5, 2 // (op 2029) 


s_add_u32 s18, s20, 128 // (op 2217) Start of Int64 add, locking SCC
s_addc_u32 s19, s21, 0 // (op 2217) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 2217) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 2217) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 7 // (op 2217) 

s_sub_i32 s51, s18, s59 // (op 2217) 

s_mul_i32 s18, s65, s51 // (op 2217) 

s_sub_u32 s19, s64, s18 // (op 2217) 


s_add_u32 s18, s19, s59 // (op 2217) 


s_cmp_eq_i32 s55, 0 // (op 2217) 
s_mov_b32 s19, scc // (op 2217) 

s_cmp_eq_i32 s19, 0 // (op 2217) 
s_cselect_b32 s51, s18, s1 // (op 2217) 


s_mul_i32 s1, s52, s62 // (op 2217) 


v_lshl_add_u32 v5, s51, 7, v12 // (op 2217) 


s_sub_i32 s18, s0, s1 // (op 2217) 


s_cmp_eq_i32 s19, 0 // (op 2217) 
s_cselect_b32 s0, s65, s18 // (op 2217) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 2217) 


v_mul_lo_u32 v8, v9, s22 // (op 2217) 

v_add_lshl_u32 v9, v5, v8, 2 // (op 2217) 


s_mul_i32 s1, s2, s5 // (op 2031) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 2031) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 2031) least significant
s_mul_hi_u32 s19, s2, s4 // (op 2031) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2031) most significant: sum
s_add_u32 s1, s1, s19 // (op 2031) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2031) 

s_mul_i32 s1, s22, s5 // (op 2219) most significant: low of low * high
s_mul_i32 s51, s23, s4 // (op 2219) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 2219) least significant
s_mul_hi_u32 s52, s22, s4 // (op 2219) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 2219) most significant: sum
s_add_u32 s1, s1, s52 // (op 2219) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 2219) 

s_mov_b32 s56, s24 // (op 2031) reg expression
s_mov_b32 s57, s25 // (op 2031) reg expression
s_mov_b32 s58, s18 // (op 2031) reg expression
s_mov_b32 s59, 131072 // (op 2031) call()

s_mov_b32 s60, s26 // (op 2219) reg expression

s_mov_b32 s61, s27 // (op 2219) reg expression
s_mov_b32 s62, s52 // (op 2219) reg expression
s_mov_b32 s63, 131072 // (op 2219) call()


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
v_accvgpr_write a8, 0.00000 // (op 471) call()
v_accvgpr_write a9, 0.00000 // (op 471) call()
v_accvgpr_write a10, 0.00000 // (op 471) call()
v_accvgpr_write a11, 0.00000 // (op 471) call()
v_accvgpr_write a12, 0.00000 // (op 633) call()
v_accvgpr_write a13, 0.00000 // (op 633) call()
v_accvgpr_write a14, 0.00000 // (op 633) call()
v_accvgpr_write a15, 0.00000 // (op 633) call()
s_mul_hi_i32 s0, s16, s8 // (op 1796) 
v_and_b32 v12, -8, v1 // (op 1796) 
s_add_i32 s1, s0, s16 // (op 1796) 

v_sub_u32 v13, v1, v12 // (op 1796) 

s_ashr_i32 s0, s1, 31 // (op 1796) 
v_lshlrev_b32 v12, 4, v13 // (op 1796) 

s_cmp_eq_i32 s8, 0 // (op 1796) 
s_mov_b32 s18, scc // (op 1796) 
v_lshrrev_b32 v13, 3, v1 // (op 1796) 
s_cmp_eq_i32 s18, 0 // (op 1796) 
s_cselect_b32 s19, 0, -1 // (op 1796) 

s_and_b32 s18, s9, 31 // (op 1796) 
s_lshl_b32 s24, 1, s18 // (op 1796) 
s_add_i32 s25, s24, s19 // (op 1796) 


s_and_b32 s19, s0, s25 // (op 1796) 


s_add_i32 s0, s1, s19 // (op 1796) 


s_ashr_i32 s1, s0, s18 // (op 1796) 


s_lshl_b32 s0, s9, 24 // (op 1796) 
s_ashr_i32 s18, s0, 31 // (op 1796) 

s_xor_b32 s0, s1, s18 // (op 1796) 

s_sub_i32 s1, s0, s18 // (op 1796) 


s_mul_hi_i32 s0, s1, s10 // (op 1796) 
s_add_i32 s18, s0, s1 // (op 1796) 

s_ashr_i32 s0, s18, 31 // (op 1796) 
s_cmp_eq_i32 s10, 0 // (op 1796) 
s_mov_b32 s19, scc // (op 1796) 
s_cmp_eq_i32 s19, 0 // (op 1796) 
s_cselect_b32 s24, 0, -1 // (op 1796) 

s_and_b32 s19, s11, 31 // (op 1796) 
s_lshl_b32 s25, 1, s19 // (op 1796) 
s_add_i32 s26, s25, s24 // (op 1796) 


s_and_b32 s27, s0, s26 // (op 1796) 


s_add_i32 s0, s18, s27 // (op 1796) 


s_ashr_i32 s18, s0, s19 // (op 1796) 


s_lshl_b32 s0, s11, 24 // (op 1796) 
s_ashr_i32 s19, s0, 31 // (op 1796) 

s_xor_b32 s0, s18, s19 // (op 1796) 

s_sub_i32 s18, s0, s19 // (op 1796) 


s_mul_i32 s0, s1, s13 // (op 1796) 

s_sub_u32 s1, s16, s0 // (op 1796) 

s_mul_i32 s0, s18, s13 // (op 1796) 
s_add_u32 s0, s0, s1 // (op 1796) 


s_mul_hi_i32 s1, s16, s7 // (op 1796) 
s_add_i32 s18, s1, s16 // (op 1796) 

s_ashr_i32 s1, s18, 31 // (op 1796) 
s_cmp_eq_i32 s7, 0 // (op 1796) 
s_mov_b32 s19, scc // (op 1796) 
s_cmp_eq_i32 s19, 0 // (op 1796) 
s_cselect_b32 s24, 0, -1 // (op 1796) 

s_and_b32 s19, s14, 31 // (op 1796) 
s_lshl_b32 s25, 1, s19 // (op 1796) 
s_add_i32 s26, s25, s24 // (op 1796) 


s_and_b32 s27, s1, s26 // (op 1796) 


s_add_i32 s1, s18, s27 // (op 1796) 


s_ashr_i32 s18, s1, s19 // (op 1796) 


s_lshl_b32 s1, s14, 24 // (op 1796) 
s_ashr_i32 s19, s1, 31 // (op 1796) 

s_xor_b32 s1, s18, s19 // (op 1796) 

s_sub_i32 s18, s1, s19 // (op 1796) 


s_add_u32 s24, s4, 32 // (op 1796) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1796) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 1796) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 1796) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 5 // (op 1796) 

s_mul_i32 s1, s13, s24 // (op 1796) 

s_mul_i32 s19, s6, s1 // (op 1796) 

s_mul_i32 s1, s18, s19 // (op 1796) 

s_sub_u32 s19, s16, s1 // (op 1796) 

s_mov_b32 s1, s19 // (op 1796) 
s_mul_hi_i32 s24, s1, s12 // (op 1796) 
s_add_i32 s25, s24, s1 // (op 1796) 


s_ashr_i32 s1, s25, 31 // (op 1796) 
s_cmp_eq_i32 s12, 0 // (op 1796) 
s_mov_b32 s24, scc // (op 1796) 
s_cmp_eq_i32 s24, 0 // (op 1796) 
s_cselect_b32 s26, 0, -1 // (op 1796) 

s_and_b32 s24, s48, 31 // (op 1796) 
s_lshl_b32 s27, 1, s24 // (op 1796) 
s_add_i32 s49, s27, s26 // (op 1796) 


s_and_b32 s26, s1, s49 // (op 1796) 


s_add_i32 s1, s25, s26 // (op 1796) 


s_ashr_i32 s25, s1, s24 // (op 1796) 


s_lshl_b32 s1, s48, 24 // (op 1796) 
s_ashr_i32 s24, s1, 31 // (op 1796) 

s_xor_b32 s1, s25, s24 // (op 1796) 

s_sub_i32 s25, s1, s24 // (op 1796) 


s_mul_i32 s1, s15, s13 // (op 1796) 
s_add_u32 s26, s20, 128 // (op 1796) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 1796) End of Int64 add, unlocking SCC
s_sub_u32 s54, s26, 1 // (op 1796) least significant half; sets scc
s_subb_u32 s55, s27, 0 // (op 1796) most significant half; uses scc

s_ashr_i64 s[26:27], s[54:55], 7 // (op 1796) 

s_sub_i32 s24, s26, s1 // (op 1796) 

s_mul_i32 s26, s25, s24 // (op 1796) 


s_sub_u32 s27, s19, s26 // (op 1796) 


s_add_u32 s19, s27, s1 // (op 1796) 


s_cmp_eq_i32 s18, 0 // (op 1796) 
s_mov_b32 s1, scc // (op 1796) 

s_cmp_eq_i32 s1, 0 // (op 1796) 
s_cselect_b32 s18, s19, s0 // (op 1796) 



v_lshl_add_u32 v14, s18, 7, v12 // (op 1796) 


v_mov_b32 v17, 0 // (op 1796) convert
v_mov_b32 v16, v14 // (op 1796) convert

v_mul_lo_u32 v15, v13, s29 // (op 1796) most significant: low of low * high
v_mul_hi_u32 v12, v13, s28 // (op 1796) most significant: high of low * low
v_mul_lo_u32 v14, v13, s28 // (op 1796) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 1796) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v14 // (op 1796) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 1796) most significant half



s_mul_i32 s1, s28, s33 // (op 1798) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 1798) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1798) least significant
s_mul_hi_u32 s19, s28, s32 // (op 1798) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1798) most significant: sum
s_add_u32 s1, s1, s19 // (op 1798) most significant: sum


s_mov_b32 s24, s30 // (op 1798) reg expression
s_mov_b32 s25, s31 // (op 1798) reg expression
s_mov_b32 s26, s0 // (op 1798) reg expression
s_mov_b32 s27, 131072 // (op 1798) call()

s_mul_hi_i32 s0, s16, s8 // (op 1815) 
v_and_b32 v14, -8, v1 // (op 1815) 
s_add_i32 s1, s0, s16 // (op 1815) 

v_sub_u32 v15, v1, v14 // (op 1815) 

s_ashr_i32 s0, s1, 31 // (op 1815) 
v_lshlrev_b32 v14, 4, v15 // (op 1815) 

s_cmp_eq_i32 s8, 0 // (op 1815) 
s_mov_b32 s18, scc // (op 1815) 
v_lshrrev_b32 v15, 3, v1 // (op 1815) 
s_cmp_eq_i32 s18, 0 // (op 1815) 
s_cselect_b32 s19, 0, -1 // (op 1815) 

s_and_b32 s18, s9, 31 // (op 1815) 
s_lshl_b32 s49, 1, s18 // (op 1815) 
s_add_i32 s50, s49, s19 // (op 1815) 


s_and_b32 s19, s0, s50 // (op 1815) 


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
s_cselect_b32 s49, 0, -1 // (op 1815) 

s_and_b32 s19, s11, 31 // (op 1815) 
s_lshl_b32 s50, 1, s19 // (op 1815) 
s_add_i32 s51, s50, s49 // (op 1815) 


s_and_b32 s49, s0, s51 // (op 1815) 


s_add_i32 s0, s18, s49 // (op 1815) 


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
s_cselect_b32 s49, 0, -1 // (op 1815) 

s_and_b32 s19, s14, 31 // (op 1815) 
s_lshl_b32 s50, 1, s19 // (op 1815) 
s_add_i32 s51, s50, s49 // (op 1815) 


s_and_b32 s49, s1, s51 // (op 1815) 


s_add_i32 s1, s18, s49 // (op 1815) 


s_ashr_i32 s18, s1, s19 // (op 1815) 


s_lshl_b32 s1, s14, 24 // (op 1815) 
s_ashr_i32 s19, s1, 31 // (op 1815) 

s_xor_b32 s1, s18, s19 // (op 1815) 

s_sub_i32 s18, s1, s19 // (op 1815) 


s_add_u32 s54, s4, 32 // (op 1815) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 1815) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1815) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1815) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 1815) 

s_mul_i32 s1, s13, s50 // (op 1815) 

s_mul_i32 s19, s6, s1 // (op 1815) 

s_mul_i32 s1, s18, s19 // (op 1815) 

s_sub_u32 s19, s16, s1 // (op 1815) 

s_mov_b32 s1, s19 // (op 1815) 
s_mul_hi_i32 s49, s1, s12 // (op 1815) 
s_add_i32 s50, s49, s1 // (op 1815) 


s_ashr_i32 s1, s50, 31 // (op 1815) 
s_cmp_eq_i32 s12, 0 // (op 1815) 
s_mov_b32 s49, scc // (op 1815) 
s_cmp_eq_i32 s49, 0 // (op 1815) 
s_cselect_b32 s51, 0, -1 // (op 1815) 

s_and_b32 s49, s48, 31 // (op 1815) 
s_lshl_b32 s52, 1, s49 // (op 1815) 
s_add_i32 s53, s52, s51 // (op 1815) 


s_and_b32 s51, s1, s53 // (op 1815) 


s_add_i32 s1, s50, s51 // (op 1815) 


s_ashr_i32 s50, s1, s49 // (op 1815) 


s_lshl_b32 s1, s48, 24 // (op 1815) 
s_ashr_i32 s49, s1, 31 // (op 1815) 

s_xor_b32 s1, s50, s49 // (op 1815) 

s_sub_i32 s50, s1, s49 // (op 1815) 


s_mul_i32 s1, s15, s13 // (op 1815) 
s_add_u32 s54, s20, 128 // (op 1815) Start of Int64 add, locking SCC
s_addc_u32 s55, s21, 0 // (op 1815) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1815) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1815) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 7 // (op 1815) 

s_sub_i32 s49, s54, s1 // (op 1815) 

s_mul_i32 s51, s50, s49 // (op 1815) 


s_sub_u32 s49, s19, s51 // (op 1815) 


s_add_u32 s19, s49, s1 // (op 1815) 


s_cmp_eq_i32 s18, 0 // (op 1815) 
s_mov_b32 s1, scc // (op 1815) 

s_cmp_eq_i32 s1, 0 // (op 1815) 
s_cselect_b32 s18, s19, s0 // (op 1815) 



v_lshl_add_u32 v16, s18, 7, v14 // (op 1815) 


v_mov_b32 v19, 0 // (op 1815) convert
v_mov_b32 v18, v16 // (op 1815) convert

v_mul_lo_u32 v17, v15, s29 // (op 1815) most significant: low of low * high
v_mul_hi_u32 v14, v15, s28 // (op 1815) most significant: high of low * low
v_mul_lo_u32 v16, v15, s28 // (op 1815) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 1815) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 1815) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 1815) most significant half



s_mul_i32 s1, s28, s33 // (op 1817) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 1817) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1817) least significant
s_mul_hi_u32 s19, s28, s32 // (op 1817) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1817) most significant: sum
s_add_u32 s1, s1, s19 // (op 1817) most significant: sum


s_mov_b32 s24, s30 // (op 1817) reg expression
s_mov_b32 s25, s31 // (op 1817) reg expression
s_mov_b32 s26, s0 // (op 1817) reg expression
s_mov_b32 s27, 131072 // (op 1817) call()

s_mul_hi_i32 s0, s16, s8 // (op 1860) 
v_lshrrev_b32 v16, 3, v1 // (op 1860) 
s_add_i32 s1, s0, s16 // (op 1860) 

v_and_b32 v17, -8, v1 // (op 1860) 
s_ashr_i32 s0, s1, 31 // (op 1860) 
v_sub_u32 v18, v1, v17 // (op 1860) 

s_cmp_eq_i32 s8, 0 // (op 1860) 
s_mov_b32 s18, scc // (op 1860) 
v_lshlrev_b32 v17, 4, v18 // (op 1860) 

s_cmp_eq_i32 s18, 0 // (op 1860) 
s_cselect_b32 s19, 0, -1 // (op 1860) 

v_mov_b32 v19, 0 // (op 1860) convert
v_mov_b32 v18, v17 // (op 1860) convert

s_and_b32 s18, s9, 31 // (op 1860) 
s_lshl_b32 s49, 1, s18 // (op 1860) 
s_add_i32 s50, s49, s19 // (op 1860) 


s_and_b32 s19, s0, s50 // (op 1860) 


s_add_i32 s0, s1, s19 // (op 1860) 


s_ashr_i32 s1, s0, s18 // (op 1860) 


s_lshl_b32 s0, s9, 24 // (op 1860) 
s_ashr_i32 s18, s0, 31 // (op 1860) 

s_xor_b32 s0, s1, s18 // (op 1860) 

s_sub_i32 s1, s0, s18 // (op 1860) 


s_mul_hi_i32 s0, s1, s10 // (op 1860) 
s_add_i32 s18, s0, s1 // (op 1860) 

s_ashr_i32 s0, s18, 31 // (op 1860) 
s_cmp_eq_i32 s10, 0 // (op 1860) 
s_mov_b32 s19, scc // (op 1860) 
s_cmp_eq_i32 s19, 0 // (op 1860) 
s_cselect_b32 s49, 0, -1 // (op 1860) 

s_and_b32 s19, s11, 31 // (op 1860) 
s_lshl_b32 s50, 1, s19 // (op 1860) 
s_add_i32 s51, s50, s49 // (op 1860) 


s_and_b32 s49, s0, s51 // (op 1860) 


s_add_i32 s0, s18, s49 // (op 1860) 


s_ashr_i32 s18, s0, s19 // (op 1860) 


s_lshl_b32 s0, s11, 24 // (op 1860) 
s_ashr_i32 s19, s0, 31 // (op 1860) 

s_xor_b32 s0, s18, s19 // (op 1860) 

s_sub_i32 s18, s0, s19 // (op 1860) 


s_add_u32 s54, s4, 32 // (op 1860) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 1860) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1860) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1860) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 1860) 

s_mul_i32 s0, s18, s50 // (op 1860) 

s_sub_i32 s49, s1, s0 // (op 1860) 


s_mul_hi_i32 s0, s16, s7 // (op 1860) 
s_add_i32 s1, s0, s16 // (op 1860) 

s_ashr_i32 s0, s1, 31 // (op 1860) 
s_cmp_eq_i32 s7, 0 // (op 1860) 
s_mov_b32 s18, scc // (op 1860) 
s_cmp_eq_i32 s18, 0 // (op 1860) 
s_cselect_b32 s19, 0, -1 // (op 1860) 

s_and_b32 s18, s14, 31 // (op 1860) 
s_lshl_b32 s52, 1, s18 // (op 1860) 
s_add_i32 s53, s52, s19 // (op 1860) 


s_and_b32 s19, s0, s53 // (op 1860) 


s_add_i32 s0, s1, s19 // (op 1860) 


s_ashr_i32 s1, s0, s18 // (op 1860) 


s_lshl_b32 s0, s14, 24 // (op 1860) 
s_ashr_i32 s18, s0, 31 // (op 1860) 

s_xor_b32 s0, s1, s18 // (op 1860) 

s_sub_i32 s1, s0, s18 // (op 1860) 


s_mul_i32 s0, s13, s50 // (op 1860) 

s_mul_i32 s18, s6, s0 // (op 1860) 

s_mul_i32 s0, s1, s18 // (op 1860) 

s_sub_u32 s18, s16, s0 // (op 1860) 

s_mul_hi_i32 s0, s18, s12 // (op 1860) 
s_add_i32 s19, s0, s18 // (op 1860) 


s_ashr_i32 s0, s19, 31 // (op 1860) 
s_cmp_eq_i32 s12, 0 // (op 1860) 
s_mov_b32 s18, scc // (op 1860) 
s_cmp_eq_i32 s18, 0 // (op 1860) 
s_cselect_b32 s50, 0, -1 // (op 1860) 

s_and_b32 s18, s48, 31 // (op 1860) 
s_lshl_b32 s51, 1, s18 // (op 1860) 
s_add_i32 s52, s51, s50 // (op 1860) 


s_and_b32 s50, s0, s52 // (op 1860) 


s_add_i32 s0, s19, s50 // (op 1860) 


s_ashr_i32 s19, s0, s18 // (op 1860) 


s_lshl_b32 s0, s48, 24 // (op 1860) 
s_ashr_i32 s18, s0, 31 // (op 1860) 

s_xor_b32 s0, s19, s18 // (op 1860) 

s_sub_i32 s19, s0, s18 // (op 1860) 


s_cmp_eq_i32 s1, 0 // (op 1860) 
s_mov_b32 s0, scc // (op 1860) 

s_cmp_eq_i32 s0, 0 // (op 1860) 
s_cselect_b32 s1, s19, s49 // (op 1860) 



v_lshl_add_u32 v17, s1, 5, v16 // (op 1860) 


v_mul_lo_u32 v21, v17, s35 // (op 1860) most significant: low of low * high
v_mul_hi_u32 v16, v17, s34 // (op 1860) most significant: high of low * low
v_mul_lo_u32 v20, v17, s34 // (op 1860) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 1860) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 1860) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 1860) most significant half



v_mul_lo_u32 v19, v17, 6 // (op 1860) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 1860) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 1860) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 1860) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 1860) 

v_mov_b32 v18, v16 // (op 1860) convert

s_mul_i32 s1, s5, 6 // (op 1862) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 1862) least significant
s_mul_hi_u32 s18, s4, 6 // (op 1862) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1862) most significant: sum

s_mul_i32 s19, s34, s1 // (op 1862) most significant: low of low * high
s_mul_i32 s49, s35, s0 // (op 1862) most significant: low of high * low
s_mul_i32 s18, s34, s0 // (op 1862) least significant
s_mul_hi_u32 s50, s34, s0 // (op 1862) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1862) most significant: sum
s_add_u32 s19, s19, s50 // (op 1862) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1862) 

s_mov_b32 s52, s36 // (op 1862) reg expression
s_mov_b32 s53, s37 // (op 1862) reg expression
s_mov_b32 s54, s0 // (op 1862) reg expression
s_mov_b32 s55, 131072 // (op 1862) call()

s_mul_hi_i32 s0, s16, s8 // (op 1879) 
v_lshrrev_b32 v16, 3, v1 // (op 1879) 
s_add_i32 s1, s0, s16 // (op 1879) 

v_and_b32 v17, -8, v1 // (op 1879) 
s_ashr_i32 s0, s1, 31 // (op 1879) 
v_sub_u32 v19, v1, v17 // (op 1879) 

s_cmp_eq_i32 s8, 0 // (op 1879) 
s_mov_b32 s18, scc // (op 1879) 
v_lshlrev_b32 v17, 4, v19 // (op 1879) 

s_cmp_eq_i32 s18, 0 // (op 1879) 
s_cselect_b32 s19, 0, -1 // (op 1879) 

v_mov_b32 v21, 0 // (op 1879) convert
v_mov_b32 v20, v17 // (op 1879) convert

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


s_add_u32 s50, s4, 32 // (op 1879) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1879) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1879) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1879) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 1879) 

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
s_lshl_b32 s64, 1, s18 // (op 1879) 
s_add_i32 s65, s64, s19 // (op 1879) 


s_and_b32 s19, s0, s65 // (op 1879) 


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
s_add_i32 s64, s51, s50 // (op 1879) 


s_and_b32 s50, s0, s64 // (op 1879) 


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



v_lshl_add_u32 v17, s1, 5, v16 // (op 1879) 


v_mul_lo_u32 v23, v17, s35 // (op 1879) most significant: low of low * high
v_mul_hi_u32 v16, v17, s34 // (op 1879) most significant: high of low * low
v_mul_lo_u32 v22, v17, s34 // (op 1879) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 1879) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v22 // (op 1879) least significant half
v_addc_co_u32 v17, s[0:1], v21, v23, s[0:1] // (op 1879) most significant half



v_mul_lo_u32 v21, v17, 6 // (op 1879) most significant: low of high * low
v_mul_hi_u32 v19, v16, 6 // (op 1879) most significant: high of low * low
v_mul_lo_u32 v20, v16, 6 // (op 1879) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 1879) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 1879) 

v_mov_b32 v19, v16 // (op 1879) convert

s_mul_i32 s1, s5, 6 // (op 1881) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 1881) least significant
s_mul_hi_u32 s18, s4, 6 // (op 1881) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1881) most significant: sum

s_mul_i32 s19, s34, s1 // (op 1881) most significant: low of low * high
s_mul_i32 s49, s35, s0 // (op 1881) most significant: low of high * low
s_mul_i32 s18, s34, s0 // (op 1881) least significant
s_mul_hi_u32 s50, s34, s0 // (op 1881) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1881) most significant: sum
s_add_u32 s19, s19, s50 // (op 1881) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1881) 

s_mov_b32 s52, s36 // (op 1881) reg expression
s_mov_b32 s53, s37 // (op 1881) reg expression
s_mov_b32 s54, s0 // (op 1881) reg expression
s_mov_b32 s55, 131072 // (op 1881) call()

s_mul_hi_i32 s0, s16, s8 // (op 1924) 
v_and_b32 v16, -128, v1 // (op 1924) 
s_add_i32 s1, s0, s16 // (op 1924) 

v_sub_u32 v17, v1, v16 // (op 1924) 

s_ashr_i32 s0, s1, 31 // (op 1924) 
v_lshrrev_b32 v16, 7, v1 // (op 1924) 
s_cmp_eq_i32 s8, 0 // (op 1924) 
s_mov_b32 s18, scc // (op 1924) 
v_lshlrev_b32 v20, 1, v16 // (op 1924) 

s_cmp_eq_i32 s18, 0 // (op 1924) 
s_cselect_b32 s19, 0, -1 // (op 1924) 

s_and_b32 s18, s9, 31 // (op 1924) 
s_lshl_b32 s49, 1, s18 // (op 1924) 
s_add_i32 s50, s49, s19 // (op 1924) 


s_and_b32 s19, s0, s50 // (op 1924) 


s_add_i32 s0, s1, s19 // (op 1924) 


s_ashr_i32 s1, s0, s18 // (op 1924) 


s_lshl_b32 s0, s9, 24 // (op 1924) 
s_ashr_i32 s18, s0, 31 // (op 1924) 

s_xor_b32 s0, s1, s18 // (op 1924) 

s_sub_i32 s1, s0, s18 // (op 1924) 


s_mul_hi_i32 s0, s1, s10 // (op 1924) 
s_add_i32 s18, s0, s1 // (op 1924) 

s_ashr_i32 s0, s18, 31 // (op 1924) 
s_cmp_eq_i32 s10, 0 // (op 1924) 
s_mov_b32 s19, scc // (op 1924) 
s_cmp_eq_i32 s19, 0 // (op 1924) 
s_cselect_b32 s49, 0, -1 // (op 1924) 

s_and_b32 s19, s11, 31 // (op 1924) 
s_lshl_b32 s50, 1, s19 // (op 1924) 
s_add_i32 s51, s50, s49 // (op 1924) 


s_and_b32 s49, s0, s51 // (op 1924) 


s_add_i32 s0, s18, s49 // (op 1924) 


s_ashr_i32 s18, s0, s19 // (op 1924) 


s_lshl_b32 s0, s11, 24 // (op 1924) 
s_ashr_i32 s19, s0, 31 // (op 1924) 

s_xor_b32 s0, s18, s19 // (op 1924) 

s_sub_i32 s18, s0, s19 // (op 1924) 


s_mul_i32 s0, s1, s13 // (op 1924) 

s_sub_u32 s1, s16, s0 // (op 1924) 

s_mul_i32 s0, s18, s13 // (op 1924) 
s_add_u32 s0, s0, s1 // (op 1924) 


s_mul_hi_i32 s1, s16, s7 // (op 1924) 
s_add_i32 s18, s1, s16 // (op 1924) 

s_ashr_i32 s1, s18, 31 // (op 1924) 
s_cmp_eq_i32 s7, 0 // (op 1924) 
s_mov_b32 s19, scc // (op 1924) 
s_cmp_eq_i32 s19, 0 // (op 1924) 
s_cselect_b32 s49, 0, -1 // (op 1924) 

s_and_b32 s19, s14, 31 // (op 1924) 
s_lshl_b32 s50, 1, s19 // (op 1924) 
s_add_i32 s51, s50, s49 // (op 1924) 


s_and_b32 s49, s1, s51 // (op 1924) 


s_add_i32 s1, s18, s49 // (op 1924) 


s_ashr_i32 s18, s1, s19 // (op 1924) 


s_lshl_b32 s1, s14, 24 // (op 1924) 
s_ashr_i32 s19, s1, 31 // (op 1924) 

s_xor_b32 s1, s18, s19 // (op 1924) 

s_sub_i32 s18, s1, s19 // (op 1924) 


s_add_u32 s50, s4, 32 // (op 1924) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1924) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1924) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1924) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 1924) 

s_mul_i32 s1, s13, s50 // (op 1924) 

s_mul_i32 s19, s6, s1 // (op 1924) 

s_mul_i32 s1, s18, s19 // (op 1924) 

s_sub_u32 s19, s16, s1 // (op 1924) 

s_mov_b32 s1, s19 // (op 1924) 
s_mul_hi_i32 s49, s1, s12 // (op 1924) 
s_add_i32 s50, s49, s1 // (op 1924) 


s_ashr_i32 s1, s50, 31 // (op 1924) 
s_cmp_eq_i32 s12, 0 // (op 1924) 
s_mov_b32 s49, scc // (op 1924) 
s_cmp_eq_i32 s49, 0 // (op 1924) 
s_cselect_b32 s51, 0, -1 // (op 1924) 

s_and_b32 s49, s48, 31 // (op 1924) 
s_lshl_b32 s64, 1, s49 // (op 1924) 
s_add_i32 s65, s64, s51 // (op 1924) 


s_and_b32 s51, s1, s65 // (op 1924) 


s_add_i32 s1, s50, s51 // (op 1924) 


s_ashr_i32 s50, s1, s49 // (op 1924) 


s_lshl_b32 s1, s48, 24 // (op 1924) 
s_ashr_i32 s49, s1, 31 // (op 1924) 

s_xor_b32 s1, s50, s49 // (op 1924) 

s_sub_i32 s50, s1, s49 // (op 1924) 


s_mul_i32 s1, s15, s13 // (op 1924) 
s_add_u32 s64, s20, 128 // (op 1924) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 1924) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1924) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1924) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 7 // (op 1924) 

s_sub_i32 s49, s64, s1 // (op 1924) 

s_mul_i32 s51, s50, s49 // (op 1924) 


s_sub_u32 s49, s19, s51 // (op 1924) 


s_add_u32 s19, s49, s1 // (op 1924) 


s_cmp_eq_i32 s18, 0 // (op 1924) 
s_mov_b32 s1, scc // (op 1924) 

s_cmp_eq_i32 s1, 0 // (op 1924) 
s_cselect_b32 s18, s19, s0 // (op 1924) 



v_lshl_add_u32 v16, s18, 7, v17 // (op 1924) 


v_mul_lo_u32 v17, v20, s38 // (op 1924) 

v_add_u32 v20, v16, v17 // (op 1924) 


s_mul_i32 s1, s38, s43 // (op 1926) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1926) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1926) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1926) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1926) most significant: sum
s_add_u32 s1, s1, s19 // (op 1926) most significant: sum


s_mov_b32 s64, s40 // (op 1926) reg expression
s_mov_b32 s65, s41 // (op 1926) reg expression
s_mov_b32 s66, s0 // (op 1926) reg expression
s_mov_b32 s67, 131072 // (op 1926) call()

s_mul_hi_i32 s0, s16, s8 // (op 1943) 
v_and_b32 v16, -128, v1 // (op 1943) 
s_add_i32 s1, s0, s16 // (op 1943) 

v_sub_u32 v17, v1, v16 // (op 1943) 

s_ashr_i32 s0, s1, 31 // (op 1943) 
v_lshrrev_b32 v16, 7, v1 // (op 1943) 
s_cmp_eq_i32 s8, 0 // (op 1943) 
s_mov_b32 s18, scc // (op 1943) 
v_lshlrev_b32 v21, 1, v16 // (op 1943) 

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
s_sub_u32 s68, s50, 1 // (op 1943) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1943) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 1943) 

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
s_lshl_b32 s68, 1, s49 // (op 1943) 
s_add_i32 s69, s68, s51 // (op 1943) 


s_and_b32 s51, s1, s69 // (op 1943) 


s_add_i32 s1, s50, s51 // (op 1943) 


s_ashr_i32 s50, s1, s49 // (op 1943) 


s_lshl_b32 s1, s48, 24 // (op 1943) 
s_ashr_i32 s49, s1, 31 // (op 1943) 

s_xor_b32 s1, s50, s49 // (op 1943) 

s_sub_i32 s50, s1, s49 // (op 1943) 


s_mul_i32 s1, s15, s13 // (op 1943) 
s_add_u32 s68, s20, 128 // (op 1943) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 1943) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1943) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1943) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 7 // (op 1943) 

s_sub_i32 s49, s68, s1 // (op 1943) 

s_mul_i32 s51, s50, s49 // (op 1943) 


s_sub_u32 s49, s19, s51 // (op 1943) 


s_add_u32 s19, s49, s1 // (op 1943) 


s_cmp_eq_i32 s18, 0 // (op 1943) 
s_mov_b32 s1, scc // (op 1943) 

s_cmp_eq_i32 s1, 0 // (op 1943) 
s_cselect_b32 s18, s19, s0 // (op 1943) 



v_lshl_add_u32 v16, s18, 7, v17 // (op 1943) 


v_mul_lo_u32 v17, v21, s38 // (op 1943) 

v_add_u32 v21, v16, v17 // (op 1943) 


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

s_mul_hi_i32 s0, s16, s8 // (op 1988) 
v_lshrrev_b32 v16, 6, v1 // (op 1988) 
s_add_i32 s1, s0, s16 // (op 1988) 

v_and_b32 v17, 1, v16 // (op 1988) 

s_ashr_i32 s0, s1, 31 // (op 1988) 
v_and_b32 v16, 15, v1 // (op 1988) 
s_cmp_eq_i32 s8, 0 // (op 1988) 
s_mov_b32 s18, scc // (op 1988) 
v_lshl_add_u32 v22, v17, 4, v16 // (op 1988) 


s_cmp_eq_i32 s18, 0 // (op 1988) 
s_cselect_b32 s19, 0, -1 // (op 1988) 

v_and_b32 v16, 63, v1 // (op 1988) 
s_and_b32 s18, s9, 31 // (op 1988) 
v_lshrrev_b32 v17, 4, v16 // (op 1988) 

s_lshl_b32 s49, 1, s18 // (op 1988) 
s_add_i32 s50, s49, s19 // (op 1988) 


s_and_b32 s19, s0, s50 // (op 1988) 


s_add_i32 s0, s1, s19 // (op 1988) 


s_ashr_i32 s1, s0, s18 // (op 1988) 


s_lshl_b32 s0, s9, 24 // (op 1988) 
s_ashr_i32 s18, s0, 31 // (op 1988) 

s_xor_b32 s0, s1, s18 // (op 1988) 

s_sub_i32 s1, s0, s18 // (op 1988) 


s_mul_hi_i32 s0, s1, s10 // (op 1988) 
s_add_i32 s18, s0, s1 // (op 1988) 

s_ashr_i32 s0, s18, 31 // (op 1988) 
s_cmp_eq_i32 s10, 0 // (op 1988) 
s_mov_b32 s19, scc // (op 1988) 
s_cmp_eq_i32 s19, 0 // (op 1988) 
s_cselect_b32 s49, 0, -1 // (op 1988) 

s_and_b32 s19, s11, 31 // (op 1988) 
s_lshl_b32 s50, 1, s19 // (op 1988) 
s_add_i32 s51, s50, s49 // (op 1988) 


s_and_b32 s49, s0, s51 // (op 1988) 


s_add_i32 s0, s18, s49 // (op 1988) 


s_ashr_i32 s18, s0, s19 // (op 1988) 


s_lshl_b32 s0, s11, 24 // (op 1988) 
s_ashr_i32 s19, s0, 31 // (op 1988) 

s_xor_b32 s0, s18, s19 // (op 1988) 

s_sub_i32 s18, s0, s19 // (op 1988) 


s_add_u32 s50, s4, 32 // (op 1988) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1988) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1988) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1988) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 1988) 

s_mul_i32 s0, s18, s50 // (op 1988) 

s_sub_i32 s49, s1, s0 // (op 1988) 


s_mul_hi_i32 s0, s16, s7 // (op 1988) 
s_add_i32 s1, s0, s16 // (op 1988) 

s_ashr_i32 s0, s1, 31 // (op 1988) 
s_cmp_eq_i32 s7, 0 // (op 1988) 
s_mov_b32 s18, scc // (op 1988) 
s_cmp_eq_i32 s18, 0 // (op 1988) 
s_cselect_b32 s19, 0, -1 // (op 1988) 

s_and_b32 s18, s14, 31 // (op 1988) 
s_lshl_b32 s68, 1, s18 // (op 1988) 
s_add_i32 s69, s68, s19 // (op 1988) 


s_and_b32 s19, s0, s69 // (op 1988) 


s_add_i32 s0, s1, s19 // (op 1988) 


s_ashr_i32 s1, s0, s18 // (op 1988) 


s_lshl_b32 s0, s14, 24 // (op 1988) 
s_ashr_i32 s18, s0, 31 // (op 1988) 

s_xor_b32 s0, s1, s18 // (op 1988) 

s_sub_i32 s1, s0, s18 // (op 1988) 


s_mul_i32 s0, s13, s50 // (op 1988) 

s_mul_i32 s18, s6, s0 // (op 1988) 

s_mul_i32 s0, s1, s18 // (op 1988) 

s_sub_u32 s18, s16, s0 // (op 1988) 

s_mul_hi_i32 s0, s18, s12 // (op 1988) 
s_add_i32 s19, s0, s18 // (op 1988) 


s_ashr_i32 s0, s19, 31 // (op 1988) 
s_cmp_eq_i32 s12, 0 // (op 1988) 
s_mov_b32 s18, scc // (op 1988) 
s_cmp_eq_i32 s18, 0 // (op 1988) 
s_cselect_b32 s50, 0, -1 // (op 1988) 

s_and_b32 s18, s48, 31 // (op 1988) 
s_lshl_b32 s51, 1, s18 // (op 1988) 
s_add_i32 s68, s51, s50 // (op 1988) 


s_and_b32 s50, s0, s68 // (op 1988) 


s_add_i32 s0, s19, s50 // (op 1988) 


s_ashr_i32 s19, s0, s18 // (op 1988) 


s_lshl_b32 s0, s48, 24 // (op 1988) 
s_ashr_i32 s18, s0, 31 // (op 1988) 

s_xor_b32 s0, s19, s18 // (op 1988) 

s_sub_i32 s19, s0, s18 // (op 1988) 


s_cmp_eq_i32 s1, 0 // (op 1988) 
s_mov_b32 s0, scc // (op 1988) 

s_cmp_eq_i32 s0, 0 // (op 1988) 
s_cselect_b32 s1, s19, s49 // (op 1988) 



v_lshl_add_u32 v16, s1, 5, v22 // (op 1988) 


v_mul_lo_u32 v22, v16, s44 // (op 1988) 

v_add_u32 v16, v17, v22 // (op 1988) 


s_mul_i32 s1, s44, s5 // (op 1990) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 1990) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 1990) least significant
s_mul_hi_u32 s19, s44, s4 // (op 1990) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1990) most significant: sum
s_add_u32 s1, s1, s19 // (op 1990) most significant: sum


s_mov_b32 s68, s46 // (op 1990) reg expression
s_mov_b32 s69, s47 // (op 1990) reg expression
s_mov_b32 s70, s0 // (op 1990) reg expression
s_mov_b32 s71, 131072 // (op 1990) call()

s_mul_hi_i32 s0, s16, s8 // (op 2007) 
v_lshrrev_b32 v17, 6, v1 // (op 2007) 
s_add_i32 s1, s0, s16 // (op 2007) 

v_and_b32 v22, 1, v17 // (op 2007) 

s_ashr_i32 s0, s1, 31 // (op 2007) 
v_and_b32 v17, 15, v1 // (op 2007) 
s_cmp_eq_i32 s8, 0 // (op 2007) 
s_mov_b32 s18, scc // (op 2007) 
v_lshl_add_u32 v23, v22, 4, v17 // (op 2007) 


s_cmp_eq_i32 s18, 0 // (op 2007) 
s_cselect_b32 s19, 0, -1 // (op 2007) 

v_and_b32 v17, 63, v1 // (op 2007) 
s_and_b32 s18, s9, 31 // (op 2007) 
v_lshrrev_b32 v22, 4, v17 // (op 2007) 

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
s_sub_u32 s72, s50, 1 // (op 2007) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2007) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 2007) 

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
s_lshl_b32 s72, 1, s18 // (op 2007) 
s_add_i32 s73, s72, s19 // (op 2007) 


s_and_b32 s19, s0, s73 // (op 2007) 


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
s_add_i32 s72, s51, s50 // (op 2007) 


s_and_b32 s50, s0, s72 // (op 2007) 


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



v_lshl_add_u32 v17, s1, 5, v23 // (op 2007) 


v_mul_lo_u32 v23, v17, s44 // (op 2007) 

v_add_u32 v17, v22, v23 // (op 2007) 


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




buffer_load_ubyte v22, v16, s[68:71], 0 offen offset: 0 // (op 1598) Load value



buffer_load_dwordx3 v[24:26], v18, s[52:55], 0 offen // (op 1602) Load value
v_and_b32 v23, -8, v1 // (op 2073) 
v_sub_u32 v27, v1, v23 // (op 2073) 

v_lshlrev_b32 v23, 4, v27 // (op 2073) 

v_lshrrev_b32 v27, 3, v1 // (op 2073) 
v_lshl_add_u32 v28, v27, 7, v23 // (op 2073) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 2073) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 5 // (op 1749) 
v_mov_b64 v[28:29], s[18:19] // (op 1749) convert


v_mov_b64 v[30:31], v[12:13] // (op 1749) reg expression
s_mov_b32 m0, s0 // (op 1749) reg expression
buffer_load_dwordx4 v30, s[24:27], 0 offen offset: 0 lds // (op 1749) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 1749) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 1749) most significant half

s_add_u32 m0, m0, 4096 // (op 1749) 
buffer_load_dwordx4 v30, s[24:27], 0 offen offset: 0 lds // (op 1749) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 1749) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 1749) most significant half

s_add_u32 m0, m0, 4096 // (op 1749) 
buffer_load_dwordx4 v30, s[24:27], 0 offen offset: 0 lds // (op 1749) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 1749) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 1749) most significant half

s_add_u32 m0, m0, 4096 // (op 1749) 
buffer_load_dwordx4 v30, s[24:27], 0 offen offset: 0 lds // (op 1749) Load value direct to lds




v_mov_b64 v[28:29], s[38:39] // (op 1608) convert

v_mov_b32 v23, v20 // (op 1608) reg expression
buffer_load_ubyte v30, v23, s[64:67], 0 offen offset: 0 // (op 1608) Load value
v_add_u32 v23, v23, v28 // (op 1608) 
buffer_load_ubyte v31, v23, s[64:67], 0 offen offset: 0 // (op 1608) Load value





buffer_load_ubyte v23, v16, s[68:71], 0 offen offset:4 // (op 1612) Load value




buffer_load_dwordx3 v[32:34], v18, s[52:55], 0 offen offset:96 // (op 1616) Load value

v_and_b32 v16, -8, v1 // (op 2246) 
v_sub_u32 v18, v1, v16 // (op 2246) 

v_lshlrev_b32 v16, 4, v18 // (op 2246) 

v_lshrrev_b32 v18, 3, v1 // (op 2246) 
v_lshl_add_u32 v27, v18, 7, v16 // (op 2246) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2246) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 5 // (op 1754) 
v_mov_b64 v[28:29], s[18:19] // (op 1754) convert


s_lshl_b64 s[18:19], s[28:29], 7 // (op 1754) 
v_mov_b32 v16, s18 // (op 1754) 
v_mov_b32 v18, s19 // (op 1754) 
v_add_co_u32 v36, s[50:51], v12, v16 // (op 1754) least significant half
v_addc_co_u32 v37, s[50:51], v13, v18, s[50:51] // (op 1754) most significant half




s_add_u32 s1, 16384, s0 // (op 1754) 
s_mov_b32 m0, s1 // (op 1754) reg expression
buffer_load_dwordx4 v36, s[24:27], 0 offen offset: 0 lds // (op 1754) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v28 // (op 1754) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1754) most significant half

s_add_u32 m0, m0, 4096 // (op 1754) 
buffer_load_dwordx4 v36, s[24:27], 0 offen offset: 0 lds // (op 1754) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v28 // (op 1754) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1754) most significant half

s_add_u32 m0, m0, 4096 // (op 1754) 
buffer_load_dwordx4 v36, s[24:27], 0 offen offset: 0 lds // (op 1754) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v28 // (op 1754) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1754) most significant half

s_add_u32 m0, m0, 4096 // (op 1754) 
buffer_load_dwordx4 v36, s[24:27], 0 offen offset: 0 lds // (op 1754) Load value direct to lds






v_mov_b64 v[12:13], s[38:39] // (op 1622) convert

s_lshl_b32 s0, s38, 2 // (op 1622) 
v_add_u32 v16, v20, s0 // (op 1622) 

buffer_load_ubyte v28, v16, s[64:67], 0 offen offset: 0 // (op 1622) Load value
v_add_u32 v16, v16, v12 // (op 1622) 
buffer_load_ubyte v29, v16, s[64:67], 0 offen offset: 0 // (op 1622) Load value



v_and_b32 v16, -8, v1 // (op 2131) 
v_sub_u32 v18, v1, v16 // (op 2131) 

v_lshlrev_b32 v16, 4, v18 // (op 2131) 

v_lshrrev_b32 v18, 3, v1 // (op 2131) 
v_lshl_add_u32 v20, v18, 7, v16 // (op 2131) 


v_mul_lo_u32 v16, v20, 6 // (op 2131) 

v_lshrrev_b32 v18, 3, v16 // (op 2131) 




s_waitcnt vmcnt(14)
ds_write_b96 v18, v[24:26] offset:32768 // (op 1624) Store local data 

v_and_b32 v16, -128, v1 // (op 2184) 
v_sub_u32 v18, v1, v16 // (op 2184) 

v_lshrrev_b32 v16, 7, v1 // (op 2184) 
v_lshl_add_u32 v20, v16, 8, v18 // (op 2184) 





s_waitcnt vmcnt(9)
ds_write_b8 v20, v30 offset:35840 // (op 1626) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v20, v31 offset:35968 // (op 1626) Store local data 

v_and_b32 v16, -8, v1 // (op 2045) 
v_sub_u32 v18, v1, v16 // (op 2045) 

v_lshlrev_b32 v16, 4, v18 // (op 2045) 

v_lshrrev_b32 v18, 3, v1 // (op 2045) 
v_lshl_add_u32 v20, v18, 7, v16 // (op 2045) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 2045) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 2102) 
v_sub_u32 v18, v1, v16 // (op 2102) 

v_lshlrev_b32 v16, 4, v18 // (op 2102) 

v_lshrrev_b32 v18, 3, v1 // (op 2102) 
v_lshl_add_u32 v20, v18, 7, v16 // (op 2102) 


v_mul_lo_u32 v16, v20, 6 // (op 2102) 

v_lshrrev_b32 v18, 3, v16 // (op 2102) 

v_and_b32 v16, -128, v1 // (op 2155) 
v_sub_u32 v20, v1, v16 // (op 2155) 

v_lshrrev_b32 v16, 7, v1 // (op 2155) 
v_lshl_add_u32 v27, v16, 8, v20 // (op 2155) 


v_and_b32 v16, -8, v1 // (op 2231) 
v_sub_u32 v20, v1, v16 // (op 2231) 

v_lshlrev_b32 v16, 4, v20 // (op 2231) 

v_lshrrev_b32 v20, 3, v1 // (op 2231) 
v_lshl_add_u32 v12, v20, 7, v16 // (op 2231) 


s_nop 0
v_readfirstlane_b32 s1, v12 // (op 2231) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 2275) 
v_sub_u32 v20, v1, v16 // (op 2275) 

v_lshlrev_b32 v16, 4, v20 // (op 2275) 

v_lshrrev_b32 v20, 3, v1 // (op 2275) 
v_lshl_add_u32 v12, v20, 7, v16 // (op 2275) 


v_mul_lo_u32 v13, v12, 6 // (op 2275) 

v_lshrrev_b32 v12, 3, v13 // (op 2275) 

v_and_b32 v13, -128, v1 // (op 2299) 
v_sub_u32 v16, v1, v13 // (op 2299) 

v_lshrrev_b32 v13, 7, v1 // (op 2299) 
v_lshl_add_u32 v20, v13, 8, v16 // (op 2299) 


s_mov_b32 s18, 0 // (op 635) call()
s_ashr_i64 s[50:51], s[32:33], 7 // (op 634) 
s_mov_b32 s19, s50 // (op 634) convert

s_and_b32 s49, s19, 4294967294 // (op 634) 

s_cmp_lt_u32 s18, s49 // (op 634) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoop_634 // (op 634) Condition: Top (jump to RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoop_634 if false)
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoop_634:
s_barrier  // (op 2363) 
s_add_u32 s18, s18, 2 // (op 636) 



ds_read_b64_tr_b8 v[36:37], v0 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:1024 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:8192 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:9216 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2376) 



ds_read_b128 v[44:47], v3 offset:32768 // (op 70) Load local data 
ds_read_b64 v[48:49], v3 offset:32784 // (op 70) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2390) 


ds_read_u8 v13, v7 offset:35840 // (op 75) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:49], a[0:3], v13, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 82) 
ds_read_b64_tr_b8 v[50:51], v0 offset:32 // (op 340) Transpose load local data 


ds_read_b64_tr_b8 v[52:53], v0 offset:1056 // (op 340) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:8224 // (op 340) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:9248 // (op 340) Transpose load local data 


ds_read_u8 v13, v7 offset:35872 // (op 344) Load local data 



ds_read_b64_tr_b8 v[36:37], v0 offset:64 // (op 502) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[44:49], a[4:7], v13, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 346) 
ds_read_b64_tr_b8 v[38:39], v0 offset:1088 // (op 502) Transpose load local data 


ds_read_b64_tr_b8 v[40:41], v0 offset:8256 // (op 502) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:9280 // (op 502) Transpose load local data 


ds_read_u8 v13, v7 offset:35904 // (op 506) Load local data 



ds_read_b64_tr_b8 v[50:51], v0 offset:96 // (op 664) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:1120 // (op 664) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:43], v[44:49], a[8:11], v13, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 508) 
ds_read_b64_tr_b8 v[54:55], v0 offset:8288 // (op 664) Transpose load local data 


ds_read_b64_tr_b8 v[56:57], v0 offset:9312 // (op 664) Transpose load local data 


ds_read_u8 v13, v7 offset:35936 // (op 668) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[44:49], a[12:15], v13, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 670) 





v_mov_b32 v13, v17 // (op 80) 

buffer_load_ubyte v22, v13, s[68:71], 0 offen offset:8 // (op 80) Load value


v_mov_b32 v16, v19 // (op 10) 

buffer_load_dwordx3 v[24:26], v16, s[52:55], 0 offen offset:192 // (op 10) Load value
s_barrier  // (op 2366) 

s_lshl_b64 s[50:51], s[28:29], 5 // (op 1733) 
v_mov_b64 v[36:37], s[50:51] // (op 1733) convert


v_mov_b64 v[38:39], v[14:15] // (op 1733) 
s_lshl_b64 s[50:51], s[28:29], 7 // (op 1733) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 1733) 

v_mov_b32 v35, s72 // (op 1733) 
v_mov_b32 v40, s73 // (op 1733) 
v_add_co_u32 v42, s[50:51], v38, v35 // (op 1733) least significant half
v_addc_co_u32 v43, s[50:51], v39, v40, s[50:51] // (op 1733) most significant half




s_mov_b32 m0, s0 // (op 1733) reg expression
buffer_load_dwordx4 v42, s[24:27], 0 offen offset: 0 lds // (op 1733) Load value direct to lds
v_add_co_u32 v42, s[50:51], v42, v36 // (op 1733) least significant half
v_addc_co_u32 v43, s[50:51], 0, v43, s[50:51] // (op 1733) most significant half

s_add_u32 m0, m0, 4096 // (op 1733) 
buffer_load_dwordx4 v42, s[24:27], 0 offen offset: 0 lds // (op 1733) Load value direct to lds
v_add_co_u32 v42, s[50:51], v42, v36 // (op 1733) least significant half
v_addc_co_u32 v43, s[50:51], 0, v43, s[50:51] // (op 1733) most significant half

s_add_u32 m0, m0, 4096 // (op 1733) 
buffer_load_dwordx4 v42, s[24:27], 0 offen offset: 0 lds // (op 1733) Load value direct to lds
v_add_co_u32 v42, s[50:51], v42, v36 // (op 1733) least significant half
v_addc_co_u32 v43, s[50:51], 0, v43, s[50:51] // (op 1733) most significant half

s_add_u32 m0, m0, 4096 // (op 1733) 
buffer_load_dwordx4 v42, s[24:27], 0 offen offset: 0 lds // (op 1733) Load value direct to lds



v_mov_b64 v[36:37], s[38:39] // (op 17) convert

v_mov_b32 v35, v21 // (op 17) 
s_lshl_b32 s19, s38, 3 // (op 17) 
v_add_u32 v40, v35, s19 // (op 17) 

buffer_load_ubyte v30, v40, s[64:67], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v40, v40, v36 // (op 17) 
buffer_load_ubyte v31, v40, s[64:67], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 2413) 



s_waitcnt vmcnt(14)
ds_write_b96 v12, v[32:34] offset:36352 // (op 235) Store local data 
s_barrier  // (op 2419) 



s_waitcnt vmcnt(9)
ds_write_b8 v20, v28 offset:39424 // (op 236) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v20, v29 offset:39552 // (op 236) Store local data 
s_barrier  // (op 2404) 



ds_read_b64_tr_b8 v[40:41], v10 offset:16384 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v10 offset:17408 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v10 offset:24576 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v10 offset:25600 // (op 227) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2410) 



ds_read_b128 v[48:51], v4 offset:36352 // (op 229) Load local data 
ds_read_b64 v[52:53], v4 offset:36368 // (op 229) Load local data 
s_barrier  // (op 2448) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2416) 


ds_read_u8 v36, v11 offset:39424 // (op 231) Load local data 



ds_read_b64_tr_b8 v[54:55], v10 offset:16416 // (op 353) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v10 offset:17440 // (op 353) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:53], a[0:3], v36, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 234) 
ds_read_b64_tr_b8 v[58:59], v10 offset:24608 // (op 353) Transpose load local data 


ds_read_b64_tr_b8 v[60:61], v10 offset:25632 // (op 353) Transpose load local data 


ds_read_u8 v36, v11 offset:39456 // (op 357) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[48:53], a[4:7], v36, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 347) 
ds_read_b64_tr_b8 v[40:41], v10 offset:16448 // (op 515) Transpose load local data 


ds_read_b64_tr_b8 v[42:43], v10 offset:17472 // (op 515) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v10 offset:24640 // (op 515) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v10 offset:25664 // (op 515) Transpose load local data 


ds_read_u8 v36, v11 offset:39488 // (op 519) Load local data 



ds_read_b64_tr_b8 v[54:55], v10 offset:16480 // (op 677) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:47], v[48:53], a[8:11], v36, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 509) 
ds_read_b64_tr_b8 v[56:57], v10 offset:17504 // (op 677) Transpose load local data 


ds_read_b64_tr_b8 v[58:59], v10 offset:24672 // (op 677) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v10 offset:25696 // (op 677) Transpose load local data 


ds_read_u8 v36, v11 offset:39520 // (op 681) Load local data 
s_barrier  // (op 2498) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:61], v[48:53], a[12:15], v36, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 671) 






buffer_load_ubyte v23, v13, s[68:71], 0 offen offset:12 // (op 233) Load value




buffer_load_dwordx3 v[32:34], v16, s[52:55], 0 offen offset:288 // (op 237) Load value
v_add_u32 v17, 8, v17 // (op 1999) 

s_barrier  // (op 2407) 
v_add_u32 v19, 192, v19 // (op 1871) 

s_lshl_b64 s[50:51], s[28:29], 5 // (op 1744) 
v_mov_b64 v[36:37], s[50:51] // (op 1744) convert


s_lshl_b64 s[50:51], s[28:29], 7 // (op 1744) 
s_mul_i32 s73, 3, s51 // (op 1744) most significant: low of low * high
s_mul_i32 s72, 3, s50 // (op 1744) least significant
s_mul_hi_u32 s19, 3, s50 // (op 1744) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 1744) most significant: sum


v_mov_b32 v13, s72 // (op 1744) 
v_mov_b32 v16, s73 // (op 1744) 
v_add_co_u32 v40, s[50:51], v38, v13 // (op 1744) least significant half
v_addc_co_u32 v41, s[50:51], v39, v16, s[50:51] // (op 1744) most significant half




s_add_u32 s19, 16384, s1 // (op 1744) 
s_mov_b32 m0, s19 // (op 1744) reg expression
buffer_load_dwordx4 v40, s[24:27], 0 offen offset: 0 lds // (op 1744) Load value direct to lds
v_add_co_u32 v40, s[50:51], v40, v36 // (op 1744) least significant half
v_addc_co_u32 v41, s[50:51], 0, v41, s[50:51] // (op 1744) most significant half

s_add_u32 m0, m0, 4096 // (op 1744) 
buffer_load_dwordx4 v40, s[24:27], 0 offen offset: 0 lds // (op 1744) Load value direct to lds
v_add_co_u32 v40, s[50:51], v40, v36 // (op 1744) least significant half
v_addc_co_u32 v41, s[50:51], 0, v41, s[50:51] // (op 1744) most significant half

s_add_u32 m0, m0, 4096 // (op 1744) 
buffer_load_dwordx4 v40, s[24:27], 0 offen offset: 0 lds // (op 1744) Load value direct to lds
v_add_co_u32 v40, s[50:51], v40, v36 // (op 1744) least significant half
v_addc_co_u32 v41, s[50:51], 0, v41, s[50:51] // (op 1744) most significant half

s_add_u32 m0, m0, 4096 // (op 1744) 
buffer_load_dwordx4 v40, s[24:27], 0 offen offset: 0 lds // (op 1744) Load value direct to lds





v_mov_b64 v[36:37], s[38:39] // (op 238) convert

s_lshl_b32 s19, s38, 2 // (op 238) 
s_mul_i32 s49, 3, s19 // (op 238) 

v_add_u32 v13, v35, s49 // (op 238) 

s_lshl_b64 s[50:51], s[28:29], 7 // (op 1807) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 1807) 

v_mov_b32 v16, s72 // (op 1807) 
v_mov_b32 v38, s73 // (op 1807) 
v_add_co_u32 v14, s[50:51], v14, v16 // (op 1807) least significant half
v_addc_co_u32 v15, s[50:51], v15, v38, s[50:51] // (op 1807) most significant half




buffer_load_ubyte v28, v13, s[64:67], 0 offen offset: 0 // (op 238) Load value
v_add_u32 v13, v13, v36 // (op 238) 
buffer_load_ubyte v29, v13, s[64:67], 0 offen offset: 0 // (op 238) Load value



s_barrier  // (op 2379) 
s_lshl_b32 s19, s38, 3 // (op 1935) 



v_add_u32 v21, v21, s19 // (op 1935) 

s_waitcnt vmcnt(14)
ds_write_b96 v18, v[24:26] offset:32768 // (op 13) Store local data 
s_barrier  // (op 2393) 



s_waitcnt vmcnt(9)
ds_write_b8 v27, v30 offset:35840 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v27, v31 offset:35968 // (op 21) Store local data 
s_ashr_i64 s[50:51], s[32:33], 7 // (op 634) 
s_mov_b32 s19, s50 // (op 634) convert

s_and_b32 s49, s19, 4294967294 // (op 634) 

s_cmp_lt_u32 s18, s49 // (op 634) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoop_634 // (op 634) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoop_634 if true)
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoop_634:














s_barrier  // (op 2480) 
s_mul_hi_i32 s0, s16, s8 // (op 1767) 
v_and_b32 v12, -8, v1 // (op 1767) 
s_add_i32 s1, s0, s16 // (op 1767) 

v_sub_u32 v13, v1, v12 // (op 1767) 

s_ashr_i32 s0, s1, 31 // (op 1767) 
v_lshlrev_b32 v12, 4, v13 // (op 1767) 

s_cmp_eq_i32 s8, 0 // (op 1767) 
s_mov_b32 s18, scc // (op 1767) 
v_lshrrev_b32 v13, 3, v1 // (op 1767) 
s_cmp_eq_i32 s18, 0 // (op 1767) 
s_cselect_b32 s19, 0, -1 // (op 1767) 

s_and_b32 s18, s9, 31 // (op 1767) 
s_lshl_b32 s49, 1, s18 // (op 1767) 
s_add_i32 s50, s49, s19 // (op 1767) 


s_and_b32 s19, s0, s50 // (op 1767) 


s_add_i32 s0, s1, s19 // (op 1767) 


s_ashr_i32 s1, s0, s18 // (op 1767) 


s_lshl_b32 s0, s9, 24 // (op 1767) 
s_ashr_i32 s18, s0, 31 // (op 1767) 

s_xor_b32 s0, s1, s18 // (op 1767) 

s_sub_i32 s1, s0, s18 // (op 1767) 


s_mul_hi_i32 s0, s1, s10 // (op 1767) 
s_add_i32 s18, s0, s1 // (op 1767) 

s_ashr_i32 s0, s18, 31 // (op 1767) 
s_cmp_eq_i32 s10, 0 // (op 1767) 
s_mov_b32 s19, scc // (op 1767) 
s_cmp_eq_i32 s19, 0 // (op 1767) 
s_cselect_b32 s49, 0, -1 // (op 1767) 

s_and_b32 s19, s11, 31 // (op 1767) 
s_lshl_b32 s50, 1, s19 // (op 1767) 
s_add_i32 s51, s50, s49 // (op 1767) 


s_and_b32 s49, s0, s51 // (op 1767) 


s_add_i32 s0, s18, s49 // (op 1767) 


s_ashr_i32 s18, s0, s19 // (op 1767) 


s_lshl_b32 s0, s11, 24 // (op 1767) 
s_ashr_i32 s19, s0, 31 // (op 1767) 

s_xor_b32 s0, s18, s19 // (op 1767) 

s_sub_i32 s18, s0, s19 // (op 1767) 


s_mul_i32 s0, s1, s13 // (op 1767) 

s_sub_u32 s1, s16, s0 // (op 1767) 

s_mul_i32 s0, s18, s13 // (op 1767) 
s_add_u32 s0, s0, s1 // (op 1767) 


s_mul_hi_i32 s1, s16, s7 // (op 1767) 
s_add_i32 s18, s1, s16 // (op 1767) 

s_ashr_i32 s1, s18, 31 // (op 1767) 
s_cmp_eq_i32 s7, 0 // (op 1767) 
s_mov_b32 s19, scc // (op 1767) 
s_cmp_eq_i32 s19, 0 // (op 1767) 
s_cselect_b32 s49, 0, -1 // (op 1767) 

s_and_b32 s19, s14, 31 // (op 1767) 
s_lshl_b32 s50, 1, s19 // (op 1767) 
s_add_i32 s51, s50, s49 // (op 1767) 


s_and_b32 s49, s1, s51 // (op 1767) 


s_add_i32 s1, s18, s49 // (op 1767) 


s_ashr_i32 s18, s1, s19 // (op 1767) 


s_lshl_b32 s1, s14, 24 // (op 1767) 
s_ashr_i32 s19, s1, 31 // (op 1767) 

s_xor_b32 s1, s18, s19 // (op 1767) 

s_sub_i32 s18, s1, s19 // (op 1767) 


s_add_u32 s50, s4, 32 // (op 1767) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1767) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1767) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1767) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1767) 

s_mul_i32 s1, s13, s50 // (op 1767) 

s_mul_i32 s19, s6, s1 // (op 1767) 

s_mul_i32 s1, s18, s19 // (op 1767) 

s_sub_u32 s19, s16, s1 // (op 1767) 

s_mov_b32 s1, s19 // (op 1767) 
s_mul_hi_i32 s49, s1, s12 // (op 1767) 
s_add_i32 s50, s49, s1 // (op 1767) 


s_ashr_i32 s1, s50, 31 // (op 1767) 
s_cmp_eq_i32 s12, 0 // (op 1767) 
s_mov_b32 s49, scc // (op 1767) 
s_cmp_eq_i32 s49, 0 // (op 1767) 
s_cselect_b32 s51, 0, -1 // (op 1767) 

s_and_b32 s49, s48, 31 // (op 1767) 
s_lshl_b32 s72, 1, s49 // (op 1767) 
s_add_i32 s73, s72, s51 // (op 1767) 


s_and_b32 s51, s1, s73 // (op 1767) 


s_add_i32 s1, s50, s51 // (op 1767) 


s_ashr_i32 s50, s1, s49 // (op 1767) 


s_lshl_b32 s1, s48, 24 // (op 1767) 
s_ashr_i32 s49, s1, 31 // (op 1767) 

s_xor_b32 s1, s50, s49 // (op 1767) 

s_sub_i32 s50, s1, s49 // (op 1767) 


s_mul_i32 s1, s15, s13 // (op 1767) 
s_add_u32 s72, s20, 128 // (op 1767) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 1767) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1767) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1767) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 7 // (op 1767) 

s_sub_i32 s49, s72, s1 // (op 1767) 

s_mul_i32 s51, s50, s49 // (op 1767) 


s_sub_u32 s49, s19, s51 // (op 1767) 


s_add_u32 s19, s49, s1 // (op 1767) 


s_cmp_eq_i32 s18, 0 // (op 1767) 
s_mov_b32 s1, scc // (op 1767) 

s_cmp_eq_i32 s1, 0 // (op 1767) 
s_cselect_b32 s18, s19, s0 // (op 1767) 



v_lshl_add_u32 v14, s18, 7, v12 // (op 1767) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1767) 
v_mov_b32 v31, 0 // (op 1767) convert
v_mov_b32 v30, v14 // (op 1767) convert

s_lshr_b32 s18, s0, 1 // (op 1767) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 1767) 


v_mul_lo_u32 v25, v12, s29 // (op 1767) most significant: low of low * high
v_mul_hi_u32 v13, v12, s28 // (op 1767) most significant: high of low * low
v_mul_lo_u32 v24, v12, s28 // (op 1767) least significant: low of low * low
v_add_u32 v25, v25, v13 // (op 1767) most significant: sum


v_add_co_u32 v26, s[0:1], v30, v24 // (op 1767) least significant half
v_addc_co_u32 v27, s[0:1], v31, v25, s[0:1] // (op 1767) most significant half



s_mul_i32 s1, s28, s33 // (op 1769) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 1769) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1769) least significant
s_mul_hi_u32 s19, s28, s32 // (op 1769) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1769) most significant: sum
s_add_u32 s1, s1, s19 // (op 1769) most significant: sum


s_mov_b32 s24, s30 // (op 1769) reg expression
s_mov_b32 s25, s31 // (op 1769) reg expression
s_mov_b32 s26, s0 // (op 1769) reg expression
s_mov_b32 s27, 131072 // (op 1769) call()

s_mul_hi_i32 s0, s16, s8 // (op 1837) 
v_lshrrev_b32 v12, 3, v1 // (op 1837) 
s_add_i32 s1, s0, s16 // (op 1837) 

v_and_b32 v13, -8, v1 // (op 1837) 
s_ashr_i32 s0, s1, 31 // (op 1837) 
v_sub_u32 v14, v1, v13 // (op 1837) 

s_cmp_eq_i32 s8, 0 // (op 1837) 
s_mov_b32 s18, scc // (op 1837) 
v_lshlrev_b32 v13, 4, v14 // (op 1837) 

s_cmp_eq_i32 s18, 0 // (op 1837) 
s_cselect_b32 s19, 0, -1 // (op 1837) 

s_and_b32 s18, s9, 31 // (op 1837) 
s_lshl_b32 s49, 1, s18 // (op 1837) 
s_add_i32 s50, s49, s19 // (op 1837) 


s_and_b32 s19, s0, s50 // (op 1837) 


s_add_i32 s0, s1, s19 // (op 1837) 


s_ashr_i32 s1, s0, s18 // (op 1837) 


s_lshl_b32 s0, s9, 24 // (op 1837) 
s_ashr_i32 s18, s0, 31 // (op 1837) 

s_xor_b32 s0, s1, s18 // (op 1837) 

s_sub_i32 s1, s0, s18 // (op 1837) 


s_mul_hi_i32 s0, s1, s10 // (op 1837) 
s_add_i32 s18, s0, s1 // (op 1837) 

s_ashr_i32 s0, s18, 31 // (op 1837) 
s_cmp_eq_i32 s10, 0 // (op 1837) 
s_mov_b32 s19, scc // (op 1837) 
s_cmp_eq_i32 s19, 0 // (op 1837) 
s_cselect_b32 s49, 0, -1 // (op 1837) 

s_and_b32 s19, s11, 31 // (op 1837) 
s_lshl_b32 s50, 1, s19 // (op 1837) 
s_add_i32 s51, s50, s49 // (op 1837) 


s_and_b32 s49, s0, s51 // (op 1837) 


s_add_i32 s0, s18, s49 // (op 1837) 


s_ashr_i32 s18, s0, s19 // (op 1837) 


s_lshl_b32 s0, s11, 24 // (op 1837) 
s_ashr_i32 s19, s0, 31 // (op 1837) 

s_xor_b32 s0, s18, s19 // (op 1837) 

s_sub_i32 s18, s0, s19 // (op 1837) 


s_add_u32 s50, s4, 32 // (op 1837) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1837) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1837) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1837) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1837) 

s_mul_i32 s0, s18, s50 // (op 1837) 

s_sub_i32 s49, s1, s0 // (op 1837) 


s_mul_hi_i32 s0, s16, s7 // (op 1837) 
s_add_i32 s1, s0, s16 // (op 1837) 

s_ashr_i32 s0, s1, 31 // (op 1837) 
s_cmp_eq_i32 s7, 0 // (op 1837) 
s_mov_b32 s18, scc // (op 1837) 
s_cmp_eq_i32 s18, 0 // (op 1837) 
s_cselect_b32 s19, 0, -1 // (op 1837) 

s_and_b32 s18, s14, 31 // (op 1837) 
s_lshl_b32 s72, 1, s18 // (op 1837) 
s_add_i32 s73, s72, s19 // (op 1837) 


s_and_b32 s19, s0, s73 // (op 1837) 


s_add_i32 s0, s1, s19 // (op 1837) 


s_ashr_i32 s1, s0, s18 // (op 1837) 


s_lshl_b32 s0, s14, 24 // (op 1837) 
s_ashr_i32 s18, s0, 31 // (op 1837) 

s_xor_b32 s0, s1, s18 // (op 1837) 

s_sub_i32 s1, s0, s18 // (op 1837) 


s_mul_i32 s0, s13, s50 // (op 1837) 

s_mul_i32 s18, s6, s0 // (op 1837) 

s_mul_i32 s0, s1, s18 // (op 1837) 

s_sub_u32 s18, s16, s0 // (op 1837) 

s_mul_hi_i32 s0, s18, s12 // (op 1837) 
s_add_i32 s19, s0, s18 // (op 1837) 


s_ashr_i32 s0, s19, 31 // (op 1837) 
s_cmp_eq_i32 s12, 0 // (op 1837) 
s_mov_b32 s18, scc // (op 1837) 
s_cmp_eq_i32 s18, 0 // (op 1837) 
s_cselect_b32 s50, 0, -1 // (op 1837) 

s_and_b32 s18, s48, 31 // (op 1837) 
s_lshl_b32 s51, 1, s18 // (op 1837) 
s_add_i32 s72, s51, s50 // (op 1837) 


s_and_b32 s50, s0, s72 // (op 1837) 


s_add_i32 s0, s19, s50 // (op 1837) 


s_ashr_i32 s19, s0, s18 // (op 1837) 


s_lshl_b32 s0, s48, 24 // (op 1837) 
s_ashr_i32 s18, s0, 31 // (op 1837) 

s_xor_b32 s0, s19, s18 // (op 1837) 

s_sub_i32 s19, s0, s18 // (op 1837) 


s_cmp_eq_i32 s1, 0 // (op 1837) 
s_mov_b32 s0, scc // (op 1837) 

s_cmp_eq_i32 s0, 0 // (op 1837) 
s_cselect_b32 s1, s19, s49 // (op 1837) 



v_lshl_add_u32 v14, s1, 5, v12 // (op 1837) 


v_mul_lo_u32 v25, v14, s35 // (op 1837) most significant: low of low * high
v_mul_hi_u32 v12, v14, s34 // (op 1837) most significant: high of low * low
v_mul_lo_u32 v24, v14, s34 // (op 1837) least significant: low of low * low
v_add_u32 v25, v25, v12 // (op 1837) most significant: sum


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1837) 
s_lshr_b32 s18, s0, 1 // (op 1837) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 1837) 


v_mov_b32 v31, 0 // (op 1837) convert
v_mov_b32 v30, v12 // (op 1837) convert

v_add_co_u32 v12, s[0:1], v30, v24 // (op 1837) least significant half
v_addc_co_u32 v13, s[0:1], v31, v25, s[0:1] // (op 1837) most significant half



v_mul_lo_u32 v25, v13, 6 // (op 1837) most significant: low of high * low
v_mul_hi_u32 v14, v12, 6 // (op 1837) most significant: high of low * low
v_mul_lo_u32 v24, v12, 6 // (op 1837) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 1837) most significant: sum


v_lshrrev_b64 v[30:31], 3, v[24:25] // (op 1837) 

v_mov_b32 v12, v30 // (op 1837) convert

s_mul_i32 s1, s5, 6 // (op 1839) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 1839) least significant
s_mul_hi_u32 s18, s4, 6 // (op 1839) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1839) most significant: sum

s_mul_i32 s19, s34, s1 // (op 1839) most significant: low of low * high
s_mul_i32 s49, s35, s0 // (op 1839) most significant: low of high * low
s_mul_i32 s18, s34, s0 // (op 1839) least significant
s_mul_hi_u32 s50, s34, s0 // (op 1839) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1839) most significant: sum
s_add_u32 s19, s19, s50 // (op 1839) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1839) 

s_mov_b32 s52, s36 // (op 1839) reg expression
s_mov_b32 s53, s37 // (op 1839) reg expression
s_mov_b32 s54, s0 // (op 1839) reg expression
s_mov_b32 s55, 131072 // (op 1839) call()

s_mul_hi_i32 s0, s16, s8 // (op 1901) 
v_and_b32 v13, -128, v1 // (op 1901) 
s_add_i32 s1, s0, s16 // (op 1901) 

v_sub_u32 v14, v1, v13 // (op 1901) 

s_ashr_i32 s0, s1, 31 // (op 1901) 
v_lshrrev_b32 v13, 7, v1 // (op 1901) 
s_cmp_eq_i32 s8, 0 // (op 1901) 
s_mov_b32 s18, scc // (op 1901) 
v_lshlrev_b32 v15, 1, v13 // (op 1901) 

s_cmp_eq_i32 s18, 0 // (op 1901) 
s_cselect_b32 s19, 0, -1 // (op 1901) 

s_and_b32 s18, s9, 31 // (op 1901) 
s_lshl_b32 s49, 1, s18 // (op 1901) 
s_add_i32 s50, s49, s19 // (op 1901) 


s_and_b32 s19, s0, s50 // (op 1901) 


s_add_i32 s0, s1, s19 // (op 1901) 


s_ashr_i32 s1, s0, s18 // (op 1901) 


s_lshl_b32 s0, s9, 24 // (op 1901) 
s_ashr_i32 s18, s0, 31 // (op 1901) 

s_xor_b32 s0, s1, s18 // (op 1901) 

s_sub_i32 s1, s0, s18 // (op 1901) 


s_mul_hi_i32 s0, s1, s10 // (op 1901) 
s_add_i32 s18, s0, s1 // (op 1901) 

s_ashr_i32 s0, s18, 31 // (op 1901) 
s_cmp_eq_i32 s10, 0 // (op 1901) 
s_mov_b32 s19, scc // (op 1901) 
s_cmp_eq_i32 s19, 0 // (op 1901) 
s_cselect_b32 s49, 0, -1 // (op 1901) 

s_and_b32 s19, s11, 31 // (op 1901) 
s_lshl_b32 s50, 1, s19 // (op 1901) 
s_add_i32 s51, s50, s49 // (op 1901) 


s_and_b32 s49, s0, s51 // (op 1901) 


s_add_i32 s0, s18, s49 // (op 1901) 


s_ashr_i32 s18, s0, s19 // (op 1901) 


s_lshl_b32 s0, s11, 24 // (op 1901) 
s_ashr_i32 s19, s0, 31 // (op 1901) 

s_xor_b32 s0, s18, s19 // (op 1901) 

s_sub_i32 s18, s0, s19 // (op 1901) 


s_mul_i32 s0, s1, s13 // (op 1901) 

s_sub_u32 s1, s16, s0 // (op 1901) 

s_mul_i32 s0, s18, s13 // (op 1901) 
s_add_u32 s0, s0, s1 // (op 1901) 


s_mul_hi_i32 s1, s16, s7 // (op 1901) 
s_add_i32 s18, s1, s16 // (op 1901) 

s_ashr_i32 s1, s18, 31 // (op 1901) 
s_cmp_eq_i32 s7, 0 // (op 1901) 
s_mov_b32 s19, scc // (op 1901) 
s_cmp_eq_i32 s19, 0 // (op 1901) 
s_cselect_b32 s49, 0, -1 // (op 1901) 

s_and_b32 s19, s14, 31 // (op 1901) 
s_lshl_b32 s50, 1, s19 // (op 1901) 
s_add_i32 s51, s50, s49 // (op 1901) 


s_and_b32 s49, s1, s51 // (op 1901) 


s_add_i32 s1, s18, s49 // (op 1901) 


s_ashr_i32 s18, s1, s19 // (op 1901) 


s_lshl_b32 s1, s14, 24 // (op 1901) 
s_ashr_i32 s19, s1, 31 // (op 1901) 

s_xor_b32 s1, s18, s19 // (op 1901) 

s_sub_i32 s18, s1, s19 // (op 1901) 


s_add_u32 s50, s4, 32 // (op 1901) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1901) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1901) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1901) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1901) 

s_mul_i32 s1, s13, s50 // (op 1901) 

s_mul_i32 s19, s6, s1 // (op 1901) 

s_mul_i32 s1, s18, s19 // (op 1901) 

s_sub_u32 s19, s16, s1 // (op 1901) 

s_mov_b32 s1, s19 // (op 1901) 
s_mul_hi_i32 s49, s1, s12 // (op 1901) 
s_add_i32 s50, s49, s1 // (op 1901) 


s_ashr_i32 s1, s50, 31 // (op 1901) 
s_cmp_eq_i32 s12, 0 // (op 1901) 
s_mov_b32 s49, scc // (op 1901) 
s_cmp_eq_i32 s49, 0 // (op 1901) 
s_cselect_b32 s51, 0, -1 // (op 1901) 

s_and_b32 s49, s48, 31 // (op 1901) 
s_lshl_b32 s72, 1, s49 // (op 1901) 
s_add_i32 s73, s72, s51 // (op 1901) 


s_and_b32 s51, s1, s73 // (op 1901) 


s_add_i32 s1, s50, s51 // (op 1901) 


s_ashr_i32 s50, s1, s49 // (op 1901) 


s_lshl_b32 s1, s48, 24 // (op 1901) 
s_ashr_i32 s49, s1, 31 // (op 1901) 

s_xor_b32 s1, s50, s49 // (op 1901) 

s_sub_i32 s50, s1, s49 // (op 1901) 


s_mul_i32 s1, s15, s13 // (op 1901) 
s_add_u32 s72, s20, 128 // (op 1901) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 1901) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1901) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1901) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 7 // (op 1901) 

s_sub_i32 s49, s72, s1 // (op 1901) 

s_mul_i32 s51, s50, s49 // (op 1901) 


s_sub_u32 s49, s19, s51 // (op 1901) 


s_add_u32 s19, s49, s1 // (op 1901) 


s_cmp_eq_i32 s18, 0 // (op 1901) 
s_mov_b32 s1, scc // (op 1901) 

s_cmp_eq_i32 s1, 0 // (op 1901) 
s_cselect_b32 s18, s19, s0 // (op 1901) 



v_lshl_add_u32 v13, s18, 7, v14 // (op 1901) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1901) 
s_lshr_b32 s18, s0, 1 // (op 1901) 

v_lshl_add_u32 v14, s18, 3, v15 // (op 1901) 


v_mul_lo_u32 v15, v14, s38 // (op 1901) 

v_add_u32 v14, v13, v15 // (op 1901) 


s_mul_i32 s1, s38, s43 // (op 1903) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1903) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1903) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1903) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1903) most significant: sum
s_add_u32 s1, s1, s19 // (op 1903) most significant: sum


s_mov_b32 s64, s40 // (op 1903) reg expression
s_mov_b32 s65, s41 // (op 1903) reg expression
s_mov_b32 s66, s0 // (op 1903) reg expression
s_mov_b32 s67, 131072 // (op 1903) call()

s_mul_hi_i32 s0, s16, s8 // (op 1965) 
v_lshrrev_b32 v13, 6, v1 // (op 1965) 
s_add_i32 s1, s0, s16 // (op 1965) 

v_and_b32 v15, 1, v13 // (op 1965) 

s_ashr_i32 s0, s1, 31 // (op 1965) 
v_and_b32 v13, 15, v1 // (op 1965) 
s_cmp_eq_i32 s8, 0 // (op 1965) 
s_mov_b32 s18, scc // (op 1965) 
v_lshl_add_u32 v16, v15, 4, v13 // (op 1965) 


s_cmp_eq_i32 s18, 0 // (op 1965) 
s_cselect_b32 s19, 0, -1 // (op 1965) 

v_and_b32 v13, 63, v1 // (op 1965) 
s_and_b32 s18, s9, 31 // (op 1965) 
v_lshrrev_b32 v15, 4, v13 // (op 1965) 

s_lshl_b32 s49, 1, s18 // (op 1965) 
s_add_i32 s50, s49, s19 // (op 1965) 


s_and_b32 s19, s0, s50 // (op 1965) 


s_add_i32 s0, s1, s19 // (op 1965) 


s_ashr_i32 s1, s0, s18 // (op 1965) 


s_lshl_b32 s0, s9, 24 // (op 1965) 
s_ashr_i32 s18, s0, 31 // (op 1965) 

s_xor_b32 s0, s1, s18 // (op 1965) 

s_sub_i32 s1, s0, s18 // (op 1965) 


s_mul_hi_i32 s0, s1, s10 // (op 1965) 
s_add_i32 s18, s0, s1 // (op 1965) 

s_ashr_i32 s0, s18, 31 // (op 1965) 
s_cmp_eq_i32 s10, 0 // (op 1965) 
s_mov_b32 s19, scc // (op 1965) 
s_cmp_eq_i32 s19, 0 // (op 1965) 
s_cselect_b32 s49, 0, -1 // (op 1965) 

s_and_b32 s19, s11, 31 // (op 1965) 
s_lshl_b32 s50, 1, s19 // (op 1965) 
s_add_i32 s51, s50, s49 // (op 1965) 


s_and_b32 s49, s0, s51 // (op 1965) 


s_add_i32 s0, s18, s49 // (op 1965) 


s_ashr_i32 s18, s0, s19 // (op 1965) 


s_lshl_b32 s0, s11, 24 // (op 1965) 
s_ashr_i32 s19, s0, 31 // (op 1965) 

s_xor_b32 s0, s18, s19 // (op 1965) 

s_sub_i32 s18, s0, s19 // (op 1965) 


s_add_u32 s50, s4, 32 // (op 1965) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 1965) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1965) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1965) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 1965) 

s_mul_i32 s0, s18, s50 // (op 1965) 

s_sub_i32 s49, s1, s0 // (op 1965) 


s_mul_hi_i32 s0, s16, s7 // (op 1965) 
s_add_i32 s1, s0, s16 // (op 1965) 

s_ashr_i32 s0, s1, 31 // (op 1965) 
s_cmp_eq_i32 s7, 0 // (op 1965) 
s_mov_b32 s18, scc // (op 1965) 
s_cmp_eq_i32 s18, 0 // (op 1965) 
s_cselect_b32 s19, 0, -1 // (op 1965) 

s_and_b32 s18, s14, 31 // (op 1965) 
s_lshl_b32 s72, 1, s18 // (op 1965) 
s_add_i32 s73, s72, s19 // (op 1965) 


s_and_b32 s19, s0, s73 // (op 1965) 


s_add_i32 s0, s1, s19 // (op 1965) 


s_ashr_i32 s1, s0, s18 // (op 1965) 


s_lshl_b32 s0, s14, 24 // (op 1965) 
s_ashr_i32 s18, s0, 31 // (op 1965) 

s_xor_b32 s0, s1, s18 // (op 1965) 

s_sub_i32 s1, s0, s18 // (op 1965) 


s_mul_i32 s0, s13, s50 // (op 1965) 

s_mul_i32 s18, s6, s0 // (op 1965) 

s_mul_i32 s0, s1, s18 // (op 1965) 

s_sub_u32 s18, s16, s0 // (op 1965) 

s_mul_hi_i32 s0, s18, s12 // (op 1965) 
s_add_i32 s19, s0, s18 // (op 1965) 


s_ashr_i32 s0, s19, 31 // (op 1965) 
s_cmp_eq_i32 s12, 0 // (op 1965) 
s_mov_b32 s18, scc // (op 1965) 
s_cmp_eq_i32 s18, 0 // (op 1965) 
s_cselect_b32 s50, 0, -1 // (op 1965) 

s_and_b32 s18, s48, 31 // (op 1965) 
s_lshl_b32 s51, 1, s18 // (op 1965) 
s_add_i32 s72, s51, s50 // (op 1965) 


s_and_b32 s50, s0, s72 // (op 1965) 


s_add_i32 s0, s19, s50 // (op 1965) 


s_ashr_i32 s19, s0, s18 // (op 1965) 


s_lshl_b32 s0, s48, 24 // (op 1965) 
s_ashr_i32 s18, s0, 31 // (op 1965) 

s_xor_b32 s0, s19, s18 // (op 1965) 

s_sub_i32 s19, s0, s18 // (op 1965) 


s_cmp_eq_i32 s1, 0 // (op 1965) 
s_mov_b32 s0, scc // (op 1965) 

s_cmp_eq_i32 s0, 0 // (op 1965) 
s_cselect_b32 s1, s19, s49 // (op 1965) 



v_lshl_add_u32 v13, s1, 5, v16 // (op 1965) 


v_mul_lo_u32 v16, v13, s44 // (op 1965) 

s_ashr_i64 s[0:1], s[32:33], 7 // (op 1965) 
s_lshr_b32 s18, s0, 1 // (op 1965) 

v_lshl_add_u32 v13, s18, 3, v15 // (op 1965) 


v_add_u32 v15, v13, v16 // (op 1965) 


s_mul_i32 s1, s44, s5 // (op 1967) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 1967) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 1967) least significant
s_mul_hi_u32 s19, s44, s4 // (op 1967) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1967) most significant: sum
s_add_u32 s1, s1, s19 // (op 1967) most significant: sum


s_mov_b32 s68, s46 // (op 1967) reg expression
s_mov_b32 s69, s47 // (op 1967) reg expression
s_mov_b32 s70, s0 // (op 1967) reg expression
s_mov_b32 s71, 131072 // (op 1967) call()

v_and_b32 v13, -8, v1 // (op 2058) 
v_sub_u32 v16, v1, v13 // (op 2058) 

v_lshlrev_b32 v13, 4, v16 // (op 2058) 

v_lshrrev_b32 v16, 3, v1 // (op 2058) 
v_lshl_add_u32 v17, v16, 7, v13 // (op 2058) 


s_nop 0
v_readfirstlane_b32 s0, v17 // (op 2058) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -8, v1 // (op 2115) 
v_sub_u32 v16, v1, v13 // (op 2115) 

v_lshlrev_b32 v13, 4, v16 // (op 2115) 

v_lshrrev_b32 v16, 3, v1 // (op 2115) 
v_lshl_add_u32 v17, v16, 7, v13 // (op 2115) 


v_mul_lo_u32 v13, v17, 6 // (op 2115) 

v_lshrrev_b32 v16, 3, v13 // (op 2115) 

v_and_b32 v13, -128, v1 // (op 2168) 
v_sub_u32 v17, v1, v13 // (op 2168) 

v_lshrrev_b32 v13, 7, v1 // (op 2168) 
v_lshl_add_u32 v18, v13, 8, v17 // (op 2168) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 640) 
s_and_b32 s1, s18, 4294967294 // (op 640) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 639) 
s_mov_b32 s49, s18 // (op 639) convert

s_cmp_lt_u32 s1, s49 // (op 639) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_639 // (op 639) Condition: Top (jump to RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_639 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoopTail_639:
s_barrier  // (op 2369) 
s_add_u32 s1, s1, 1 // (op 641) 

s_lshl_b64 s[18:19], s[28:29], 5 // (op 1738) 
v_mov_b64 v[24:25], s[18:19] // (op 1738) convert


v_mov_b64 v[22:23], v[26:27] // (op 1738) 
v_mov_b64 v[20:21], v[22:23] // (op 1738) reg expression
s_mov_b32 m0, s0 // (op 1738) reg expression
buffer_load_dwordx4 v20, s[24:27], 0 offen offset: 0 lds // (op 1738) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v24 // (op 1738) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 1738) most significant half

s_add_u32 m0, m0, 4096 // (op 1738) 
buffer_load_dwordx4 v20, s[24:27], 0 offen offset: 0 lds // (op 1738) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v24 // (op 1738) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 1738) most significant half

s_add_u32 m0, m0, 4096 // (op 1738) 
buffer_load_dwordx4 v20, s[24:27], 0 offen offset: 0 lds // (op 1738) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v24 // (op 1738) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 1738) most significant half

s_add_u32 m0, m0, 4096 // (op 1738) 
buffer_load_dwordx4 v20, s[24:27], 0 offen offset: 0 lds // (op 1738) Load value direct to lds



s_lshl_b64 s[18:19], s[28:29], 7 // (op 1757) 
v_mov_b32 v13, s18 // (op 1757) 
v_mov_b32 v17, s19 // (op 1757) 
v_add_co_u32 v26, s[50:51], v26, v13 // (op 1757) least significant half
v_addc_co_u32 v27, s[50:51], v27, v17, s[50:51] // (op 1757) most significant half






v_mov_b32 v13, v12 // (op 179) 

buffer_load_dwordx3 v[20:22], v13, s[52:55], 0 offen // (op 179) Load value

v_add_u32 v12, 96, v12 // (op 1829) 

s_barrier  // (op 2382) 
v_mov_b64 v[24:25], s[38:39] // (op 180) convert




v_mov_b32 v13, v14 // (op 180) 
v_mov_b32 v17, v13 // (op 180) reg expression
buffer_load_ubyte v28, v17, s[64:67], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v17, v17, v24 // (op 180) 
s_waitcnt vmcnt(1)
ds_write_b96 v16, v[20:22] offset:32768 // (op 177) Store local data 
buffer_load_ubyte v29, v17, s[64:67], 0 offen offset: 0 // (op 180) Load value




s_lshl_b32 s18, s38, 2 // (op 1893) 
v_add_u32 v14, v14, s18 // (op 1893) 

s_barrier  // (op 2372) 
s_barrier  // (op 2396) 






ds_read_b64_tr_b8 v[30:31], v0 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:1024 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:8192 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:9216 // (op 169) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v18, v28 offset:35840 // (op 178) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2386) 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v29 offset:35968 // (op 178) Store local data 




ds_read_b128 v[20:23], v3 offset:32768 // (op 171) Load local data 
ds_read_b64 v[24:25], v3 offset:32784 // (op 171) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2400) 


ds_read_u8 v13, v7 offset:35840 // (op 173) Load local data 


v_mov_b32 v17, v15 // (op 175) 

buffer_load_ubyte v19, v17, s[68:71], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v15, 4, v15 // (op 1957) 



ds_read_b64_tr_b8 v[38:39], v0 offset:32 // (op 321) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:1056 // (op 321) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[20:25], a[0:3], v13, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 176) 
ds_read_b64_tr_b8 v[42:43], v0 offset:8224 // (op 321) Transpose load local data 


ds_read_b64_tr_b8 v[44:45], v0 offset:9248 // (op 321) Transpose load local data 


ds_read_u8 v13, v7 offset:35872 // (op 325) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:45], v[20:25], a[4:7], v13, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 327) 
ds_read_b64_tr_b8 v[28:29], v0 offset:64 // (op 483) Transpose load local data 


ds_read_b64_tr_b8 v[30:31], v0 offset:1088 // (op 483) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:8256 // (op 483) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:9280 // (op 483) Transpose load local data 


ds_read_u8 v13, v7 offset:35904 // (op 487) Load local data 



ds_read_b64_tr_b8 v[36:37], v0 offset:96 // (op 645) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:35], v[20:25], a[8:11], v13, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 489) 
ds_read_b64_tr_b8 v[38:39], v0 offset:1120 // (op 645) Transpose load local data 


ds_read_b64_tr_b8 v[40:41], v0 offset:8288 // (op 645) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:9312 // (op 645) Transpose load local data 


ds_read_u8 v13, v7 offset:35936 // (op 649) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:43], v[20:25], a[12:15], v13, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 651) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 639) 
s_mov_b32 s49, s18 // (op 639) convert

s_cmp_lt_u32 s1, s49 // (op 639) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoopTail_639 // (op 639) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoopTail_639 if true)
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_639:






















s_barrier  // (op 2484) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalFalse_0_beta_123
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen // (op 52) Store value

v_accvgpr_read v10, a4 // (op 881) 
v_accvgpr_read v11, a5 // (op 881) 
v_accvgpr_read v12, a6 // (op 881) 
v_accvgpr_read v13, a7 // (op 881) 
v_mul_f32 v14, v8, v10 // (op 881) 
v_mul_f32 v15, v8, v11 // (op 881) 
v_mul_f32 v16, v8, v12 // (op 881) 
v_mul_f32 v17, v8, v13 // (op 881) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:128 // (op 883) Store value

v_accvgpr_read v10, a8 // (op 899) 
v_accvgpr_read v11, a9 // (op 899) 
v_accvgpr_read v12, a10 // (op 899) 
v_accvgpr_read v13, a11 // (op 899) 
v_mul_f32 v14, v8, v10 // (op 899) 
v_mul_f32 v15, v8, v11 // (op 899) 
v_mul_f32 v16, v8, v12 // (op 899) 
v_mul_f32 v17, v8, v13 // (op 899) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:256 // (op 901) Store value

v_accvgpr_read v10, a12 // (op 917) 
v_accvgpr_read v11, a13 // (op 917) 
v_accvgpr_read v12, a14 // (op 917) 
v_accvgpr_read v13, a15 // (op 917) 
v_mul_f32 v14, v8, v10 // (op 917) 
v_mul_f32 v15, v8, v11 // (op 917) 
v_mul_f32 v16, v8, v12 // (op 917) 
v_mul_f32 v17, v8, v13 // (op 917) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:384 // (op 919) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:256 // (op 975) Load value
v_accvgpr_read v14, a8 // (op 972) 
v_accvgpr_read v15, a9 // (op 972) 
v_accvgpr_read v16, a10 // (op 972) 
v_accvgpr_read v17, a11 // (op 972) 
v_mul_f32 v18, v8, v14 // (op 972) 
v_mul_f32 v19, v8, v15 // (op 972) 
v_mul_f32 v20, v8, v16 // (op 972) 
v_mul_f32 v21, v8, v17 // (op 972) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 976) 
v_mul_f32 v15, v5, v11 // (op 976) 
v_mul_f32 v16, v5, v12 // (op 976) 
v_mul_f32 v17, v5, v13 // (op 976) 

v_add_f32 v10, v14, v18 // (op 973) 
v_add_f32 v11, v15, v19 // (op 973) 
v_add_f32 v12, v16, v20 // (op 973) 
v_add_f32 v13, v17, v21 // (op 973) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:256 // (op 974) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:128 // (op 957) Load value
v_accvgpr_read v14, a4 // (op 954) 
v_accvgpr_read v15, a5 // (op 954) 
v_accvgpr_read v16, a6 // (op 954) 
v_accvgpr_read v17, a7 // (op 954) 
v_mul_f32 v18, v8, v14 // (op 954) 
v_mul_f32 v19, v8, v15 // (op 954) 
v_mul_f32 v20, v8, v16 // (op 954) 
v_mul_f32 v21, v8, v17 // (op 954) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 958) 
v_mul_f32 v15, v5, v11 // (op 958) 
v_mul_f32 v16, v5, v12 // (op 958) 
v_mul_f32 v17, v5, v13 // (op 958) 

v_add_f32 v10, v14, v18 // (op 955) 
v_add_f32 v11, v15, v19 // (op 955) 
v_add_f32 v12, v16, v20 // (op 955) 
v_add_f32 v13, v17, v21 // (op 955) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:128 // (op 956) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:384 // (op 993) Load value
v_accvgpr_read v14, a12 // (op 990) 
v_accvgpr_read v15, a13 // (op 990) 
v_accvgpr_read v16, a14 // (op 990) 
v_accvgpr_read v17, a15 // (op 990) 
v_mul_f32 v18, v8, v14 // (op 990) 
v_mul_f32 v19, v8, v15 // (op 990) 
v_mul_f32 v20, v8, v16 // (op 990) 
v_mul_f32 v21, v8, v17 // (op 990) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 994) 
v_mul_f32 v15, v5, v11 // (op 994) 
v_mul_f32 v16, v5, v12 // (op 994) 
v_mul_f32 v17, v5, v13 // (op 994) 

v_add_f32 v10, v14, v18 // (op 991) 
v_add_f32 v11, v15, v19 // (op 991) 
v_add_f32 v12, v16, v20 // (op 991) 
v_add_f32 v13, v17, v21 // (op 991) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:384 // (op 992) Store value




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
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalBottom_0_beta_123:












s_endpgm  // End of RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
.LRR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__end:
.size RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_, .LRR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__end-RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 80
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 39936 // lds bytes
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: F6
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
  - .name: RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 39936
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 62
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

