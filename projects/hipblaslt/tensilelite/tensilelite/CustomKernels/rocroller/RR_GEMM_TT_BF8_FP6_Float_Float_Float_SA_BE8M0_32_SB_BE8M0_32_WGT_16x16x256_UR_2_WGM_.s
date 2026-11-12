// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_,@function
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 871) 
v_and_b32 v3, 15, v1 // (op 871) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 871) 


v_and_b32 v0, 48, v1 // (op 871) 
v_lshl_add_u32 v3, v4, 8, v0 // (op 871) 


v_and_b32 v0, 63, v1 // (op 924) 
v_and_b32 v4, 48, v1 // (op 924) 
v_sub_u32 v5, v0, v4 // (op 924) 

v_lshrrev_b32 v4, 4, v0 // (op 924) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 924) 


v_lshlrev_b32 v4, 4, v0 // (op 924) 
v_mul_lo_u32 v5, v4, 6 // (op 924) 

v_lshrrev_b32 v4, 3, v5 // (op 924) 

v_and_b32 v5, 268435455, v0 // (op 924) 

v_lshlrev_b32 v0, 2, v5 // (op 924) 

v_add_u32 v5, v4, v0 // (op 924) 


v_lshrrev_b32 v0, 6, v1 // (op 977) 
v_and_b32 v4, 15, v1 // (op 977) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 977) 


v_and_b32 v0, 63, v1 // (op 977) 
v_lshrrev_b32 v4, 4, v0 // (op 977) 

v_lshl_add_u32 v0, v6, 3, v4 // (op 977) 


v_and_b32 v4, 48, v1 // (op 1030) 
v_and_b32 v6, 15, v1 // (op 1030) 
v_add_u32 v7, v4, v6 // (op 1030) 


v_lshrrev_b32 v4, 6, v1 // (op 1103) 
v_and_b32 v6, 15, v1 // (op 1103) 
v_lshl_add_u32 v8, v4, 4, v6 // (op 1103) 


v_and_b32 v4, 48, v1 // (op 1103) 
v_lshl_add_u32 v6, v8, 8, v4 // (op 1103) 


v_and_b32 v4, 63, v1 // (op 1127) 
v_and_b32 v8, 48, v1 // (op 1127) 
v_sub_u32 v9, v4, v8 // (op 1127) 

v_lshrrev_b32 v8, 4, v4 // (op 1127) 

v_lshl_add_u32 v4, v8, 5, v9 // (op 1127) 


v_lshlrev_b32 v8, 4, v4 // (op 1127) 
v_mul_lo_u32 v9, v8, 6 // (op 1127) 

v_lshrrev_b32 v8, 3, v9 // (op 1127) 

v_and_b32 v9, 268435455, v4 // (op 1127) 

v_lshlrev_b32 v4, 2, v9 // (op 1127) 

v_add_u32 v9, v8, v4 // (op 1127) 


v_lshrrev_b32 v4, 6, v1 // (op 1151) 
v_and_b32 v8, 15, v1 // (op 1151) 
v_lshl_add_u32 v10, v4, 4, v8 // (op 1151) 


v_and_b32 v4, 63, v1 // (op 1151) 
v_lshrrev_b32 v8, 4, v4 // (op 1151) 

v_lshl_add_u32 v4, v10, 3, v8 // (op 1151) 


v_and_b32 v8, 48, v1 // (op 1175) 
v_and_b32 v10, 15, v1 // (op 1175) 
v_add_u32 v11, v8, v10 // (op 1175) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v10, s9 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_mul_hi_i32 s0, s16, s11 // (op 582) 
v_lshrrev_b32 v12, 4, v1 // (op 582) 
s_add_i32 s1, s0, s16 // (op 582) 

v_and_b32 v13, -16, v1 // (op 582) 
s_ashr_i32 s0, s1, 31 // (op 582) 
v_sub_u32 v14, v1, v13 // (op 582) 

s_cmp_eq_i32 s11, 0 // (op 582) 
s_mov_b32 s8, scc // (op 582) 
v_lshlrev_b32 v13, 4, v14 // (op 582) 

s_cmp_eq_i32 s8, 0 // (op 582) 
s_cselect_b32 s9, 0, -1 // (op 582) 

v_mov_b32 v15, 0 // (op 582) convert
v_mov_b32 v14, v13 // (op 582) convert

s_and_b32 s8, s12, 31 // (op 582) 
s_lshl_b32 s18, 1, s8 // (op 582) 
s_add_i32 s19, s18, s9 // (op 582) 


s_and_b32 s9, s0, s19 // (op 582) 


s_add_i32 s0, s1, s9 // (op 582) 


s_ashr_i32 s1, s0, s8 // (op 582) 


s_lshl_b32 s0, s12, 24 // (op 582) 
s_ashr_i32 s8, s0, 31 // (op 582) 

s_xor_b32 s0, s1, s8 // (op 582) 

s_sub_i32 s1, s0, s8 // (op 582) 


s_mul_hi_i32 s0, s1, s13 // (op 582) 
s_add_i32 s8, s0, s1 // (op 582) 

s_ashr_i32 s0, s8, 31 // (op 582) 
s_cmp_eq_i32 s13, 0 // (op 582) 
s_mov_b32 s9, scc // (op 582) 
s_cmp_eq_i32 s9, 0 // (op 582) 
s_cselect_b32 s18, 0, -1 // (op 582) 

s_and_b32 s9, s14, 31 // (op 582) 
s_lshl_b32 s19, 1, s9 // (op 582) 
s_add_i32 s51, s19, s18 // (op 582) 


s_and_b32 s18, s0, s51 // (op 582) 


s_add_i32 s0, s8, s18 // (op 582) 


s_ashr_i32 s8, s0, s9 // (op 582) 


s_lshl_b32 s0, s14, 24 // (op 582) 
s_ashr_i32 s9, s0, 31 // (op 582) 

s_xor_b32 s0, s8, s9 // (op 582) 

s_sub_i32 s8, s0, s9 // (op 582) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s46 // (op 582) 

s_sub_u32 s1, s16, s0 // (op 582) 

s_mul_i32 s0, s8, s46 // (op 582) 
s_add_u32 s0, s0, s1 // (op 582) 


s_mul_hi_i32 s1, s16, s10 // (op 582) 
s_add_i32 s8, s1, s16 // (op 582) 

s_ashr_i32 s1, s8, 31 // (op 582) 
s_cmp_eq_i32 s10, 0 // (op 582) 
s_mov_b32 s9, scc // (op 582) 
s_cmp_eq_i32 s9, 0 // (op 582) 
s_cselect_b32 s18, 0, -1 // (op 582) 

s_and_b32 s9, s49, 31 // (op 582) 
s_lshl_b32 s19, 1, s9 // (op 582) 
s_add_i32 s51, s19, s18 // (op 582) 


s_and_b32 s18, s1, s51 // (op 582) 


s_add_i32 s1, s8, s18 // (op 582) 


s_ashr_i32 s8, s1, s9 // (op 582) 


s_lshl_b32 s1, s49, 24 // (op 582) 
s_ashr_i32 s9, s1, 31 // (op 582) 

s_xor_b32 s1, s8, s9 // (op 582) 

s_sub_i32 s8, s1, s9 // (op 582) 


s_add_u32 s18, s2, 16 // (op 582) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 582) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 582) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 582) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 582) 

s_mul_i32 s1, s46, s18 // (op 582) 

s_mul_i32 s9, s50, s1 // (op 582) 

s_mul_i32 s1, s8, s9 // (op 582) 

s_sub_u32 s9, s16, s1 // (op 582) 

s_mov_b32 s1, s9 // (op 582) 
s_mul_hi_i32 s18, s1, s15 // (op 582) 
s_add_i32 s19, s18, s1 // (op 582) 


s_ashr_i32 s1, s19, 31 // (op 582) 
s_cmp_eq_i32 s15, 0 // (op 582) 
s_mov_b32 s18, scc // (op 582) 
s_cmp_eq_i32 s18, 0 // (op 582) 
s_cselect_b32 s51, 0, -1 // (op 582) 

s_and_b32 s18, s48, 31 // (op 582) 
s_lshl_b32 s52, 1, s18 // (op 582) 
s_add_i32 s53, s52, s51 // (op 582) 


s_and_b32 s51, s1, s53 // (op 582) 


s_add_i32 s1, s19, s51 // (op 582) 


s_ashr_i32 s19, s1, s18 // (op 582) 


s_lshl_b32 s1, s48, 24 // (op 582) 
s_ashr_i32 s18, s1, 31 // (op 582) 

s_xor_b32 s1, s19, s18 // (op 582) 

s_sub_i32 s19, s1, s18 // (op 582) 


s_mul_i32 s1, s47, s46 // (op 582) 
s_add_u32 s52, s6, 16 // (op 582) Start of Int64 add, locking SCC
s_addc_u32 s53, s7, 0 // (op 582) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 582) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 582) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 582) 

s_sub_i32 s18, s52, s1 // (op 582) 

s_mul_i32 s51, s19, s18 // (op 582) 


s_sub_u32 s18, s9, s51 // (op 582) 


s_add_u32 s9, s18, s1 // (op 582) 


s_cmp_eq_i32 s8, 0 // (op 582) 
s_mov_b32 s1, scc // (op 582) 

s_cmp_eq_i32 s1, 0 // (op 582) 
s_cselect_b32 s8, s9, s0 // (op 582) 



v_lshl_add_u32 v13, s8, 4, v12 // (op 582) 


v_mul_lo_u32 v17, v13, s5 // (op 582) most significant: low of low * high
v_mul_hi_u32 v12, v13, s4 // (op 582) most significant: high of low * low
v_mul_lo_u32 v16, v13, s4 // (op 582) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 582) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 582) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 582) most significant half



s_mul_i32 s1, s4, s7 // (op 584) most significant: low of low * high
s_mul_i32 s8, s5, s6 // (op 584) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 584) least significant
s_mul_hi_u32 s9, s4, s6 // (op 584) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 584) most significant: sum
s_add_u32 s1, s1, s9 // (op 584) most significant: sum


s_mov_b32 s52, s20 // (op 584) reg expression
s_mov_b32 s53, s21 // (op 584) reg expression
s_mov_b32 s54, s0 // (op 584) reg expression
s_mov_b32 s55, 131072 // (op 584) call()

s_mul_hi_i32 s51, s16, s11 // (op 601) 
v_lshrrev_b32 v12, 4, v1 // (op 601) 
s_add_i32 s0, s51, s16 // (op 601) 

v_and_b32 v13, -16, v1 // (op 601) 
s_ashr_i32 s1, s0, 31 // (op 601) 
v_sub_u32 v14, v1, v13 // (op 601) 

s_cmp_eq_i32 s11, 0 // (op 601) 
s_mov_b32 s51, scc // (op 601) 
v_lshlrev_b32 v13, 4, v14 // (op 601) 

s_cmp_eq_i32 s51, 0 // (op 601) 
s_cselect_b32 s8, 0, -1 // (op 601) 

v_mov_b32 v15, 0 // (op 601) convert
v_mov_b32 v14, v13 // (op 601) convert

s_and_b32 s9, s12, 31 // (op 601) 
s_lshl_b32 s51, 1, s9 // (op 601) 
s_add_i32 s18, s51, s8 // (op 601) 


s_and_b32 s8, s1, s18 // (op 601) 


s_add_i32 s1, s0, s8 // (op 601) 


s_ashr_i32 s0, s1, s9 // (op 601) 


s_lshl_b32 s1, s12, 24 // (op 601) 
s_ashr_i32 s51, s1, 31 // (op 601) 

s_xor_b32 s1, s0, s51 // (op 601) 

s_sub_i32 s0, s1, s51 // (op 601) 


s_mul_hi_i32 s1, s0, s13 // (op 601) 
s_add_i32 s51, s1, s0 // (op 601) 

s_ashr_i32 s1, s51, 31 // (op 601) 
s_cmp_eq_i32 s13, 0 // (op 601) 
s_mov_b32 s8, scc // (op 601) 
s_cmp_eq_i32 s8, 0 // (op 601) 
s_cselect_b32 s9, 0, -1 // (op 601) 

s_and_b32 s8, s14, 31 // (op 601) 
s_lshl_b32 s18, 1, s8 // (op 601) 
s_add_i32 s19, s18, s9 // (op 601) 


s_and_b32 s9, s1, s19 // (op 601) 


s_add_i32 s1, s51, s9 // (op 601) 


s_ashr_i32 s9, s1, s8 // (op 601) 


s_lshl_b32 s1, s14, 24 // (op 601) 
s_ashr_i32 s8, s1, 31 // (op 601) 

s_xor_b32 s1, s9, s8 // (op 601) 

s_sub_i32 s9, s1, s8 // (op 601) 


s_mul_i32 s1, s0, s46 // (op 601) 

s_sub_u32 s0, s16, s1 // (op 601) 

s_mul_i32 s1, s9, s46 // (op 601) 
s_add_u32 s1, s1, s0 // (op 601) 


s_mul_hi_i32 s0, s16, s10 // (op 601) 
s_add_i32 s51, s0, s16 // (op 601) 

s_ashr_i32 s0, s51, 31 // (op 601) 
s_cmp_eq_i32 s10, 0 // (op 601) 
s_mov_b32 s8, scc // (op 601) 
s_cmp_eq_i32 s8, 0 // (op 601) 
s_cselect_b32 s9, 0, -1 // (op 601) 

s_and_b32 s8, s49, 31 // (op 601) 
s_lshl_b32 s18, 1, s8 // (op 601) 
s_add_i32 s19, s18, s9 // (op 601) 


s_and_b32 s9, s0, s19 // (op 601) 


s_add_i32 s0, s51, s9 // (op 601) 


s_ashr_i32 s9, s0, s8 // (op 601) 


s_lshl_b32 s0, s49, 24 // (op 601) 
s_ashr_i32 s8, s0, 31 // (op 601) 

s_xor_b32 s0, s9, s8 // (op 601) 

s_sub_i32 s9, s0, s8 // (op 601) 


s_add_u32 s18, s2, 16 // (op 601) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 601) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 601) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 601) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 601) 

s_mul_i32 s0, s46, s18 // (op 601) 

s_mul_i32 s8, s50, s0 // (op 601) 

s_mul_i32 s0, s9, s8 // (op 601) 

s_sub_u32 s8, s16, s0 // (op 601) 

s_mov_b32 s0, s8 // (op 601) 
s_mul_hi_i32 s51, s0, s15 // (op 601) 
s_add_i32 s18, s51, s0 // (op 601) 


s_ashr_i32 s0, s18, 31 // (op 601) 
s_cmp_eq_i32 s15, 0 // (op 601) 
s_mov_b32 s19, scc // (op 601) 
s_cmp_eq_i32 s19, 0 // (op 601) 
s_cselect_b32 s51, 0, -1 // (op 601) 

s_and_b32 s19, s48, 31 // (op 601) 
s_lshl_b32 s56, 1, s19 // (op 601) 
s_add_i32 s57, s56, s51 // (op 601) 


s_and_b32 s51, s0, s57 // (op 601) 


s_add_i32 s0, s18, s51 // (op 601) 


s_ashr_i32 s18, s0, s19 // (op 601) 


s_lshl_b32 s0, s48, 24 // (op 601) 
s_ashr_i32 s19, s0, 31 // (op 601) 

s_xor_b32 s0, s18, s19 // (op 601) 

s_sub_i32 s18, s0, s19 // (op 601) 


s_mul_i32 s0, s47, s46 // (op 601) 
s_add_u32 s56, s6, 16 // (op 601) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 601) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 601) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 601) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 601) 

s_sub_i32 s19, s56, s0 // (op 601) 

s_mul_i32 s51, s18, s19 // (op 601) 


s_sub_u32 s18, s8, s51 // (op 601) 


s_add_u32 s8, s18, s0 // (op 601) 


s_cmp_eq_i32 s9, 0 // (op 601) 
s_mov_b32 s0, scc // (op 601) 

s_cmp_eq_i32 s0, 0 // (op 601) 
s_cselect_b32 s9, s8, s1 // (op 601) 



v_lshl_add_u32 v13, s9, 4, v12 // (op 601) 


v_mul_lo_u32 v19, v13, s5 // (op 601) most significant: low of low * high
v_mul_hi_u32 v12, v13, s4 // (op 601) most significant: high of low * low
v_mul_lo_u32 v18, v13, s4 // (op 601) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 601) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 601) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 601) most significant half



s_mul_i32 s1, s4, s7 // (op 603) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 603) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 603) least significant
s_mul_hi_u32 s8, s4, s6 // (op 603) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 603) most significant: sum
s_add_u32 s1, s1, s8 // (op 603) most significant: sum


s_mov_b32 s52, s20 // (op 603) reg expression
s_mov_b32 s53, s21 // (op 603) reg expression
s_mov_b32 s54, s0 // (op 603) reg expression
s_mov_b32 s55, 131072 // (op 603) call()

s_mul_hi_i32 s51, s16, s11 // (op 646) 
v_lshlrev_b32 v12, 2, v1 // (op 646) 
s_add_i32 s0, s51, s16 // (op 646) 

s_ashr_i32 s1, s0, 31 // (op 646) 
s_cmp_eq_i32 s11, 0 // (op 646) 
s_mov_b32 s51, scc // (op 646) 
s_cmp_eq_i32 s51, 0 // (op 646) 
s_cselect_b32 s8, 0, -1 // (op 646) 

s_and_b32 s9, s12, 31 // (op 646) 
s_lshl_b32 s51, 1, s9 // (op 646) 
s_add_i32 s18, s51, s8 // (op 646) 


s_and_b32 s8, s1, s18 // (op 646) 


s_add_i32 s1, s0, s8 // (op 646) 


s_ashr_i32 s0, s1, s9 // (op 646) 


s_lshl_b32 s1, s12, 24 // (op 646) 
s_ashr_i32 s51, s1, 31 // (op 646) 

s_xor_b32 s1, s0, s51 // (op 646) 

s_sub_i32 s0, s1, s51 // (op 646) 


s_mul_hi_i32 s1, s0, s13 // (op 646) 
s_add_i32 s51, s1, s0 // (op 646) 

s_ashr_i32 s1, s51, 31 // (op 646) 
s_cmp_eq_i32 s13, 0 // (op 646) 
s_mov_b32 s8, scc // (op 646) 
s_cmp_eq_i32 s8, 0 // (op 646) 
s_cselect_b32 s9, 0, -1 // (op 646) 

s_and_b32 s8, s14, 31 // (op 646) 
s_lshl_b32 s18, 1, s8 // (op 646) 
s_add_i32 s19, s18, s9 // (op 646) 


s_and_b32 s9, s1, s19 // (op 646) 


s_add_i32 s1, s51, s9 // (op 646) 


s_ashr_i32 s9, s1, s8 // (op 646) 


s_lshl_b32 s1, s14, 24 // (op 646) 
s_ashr_i32 s8, s1, 31 // (op 646) 

s_xor_b32 s1, s9, s8 // (op 646) 

s_sub_i32 s9, s1, s8 // (op 646) 


s_add_u32 s18, s2, 16 // (op 646) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 646) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 646) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 646) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 646) 

s_mul_i32 s1, s9, s18 // (op 646) 

s_sub_i32 s51, s0, s1 // (op 646) 


s_mul_hi_i32 s0, s16, s10 // (op 646) 
s_add_i32 s1, s0, s16 // (op 646) 

s_ashr_i32 s0, s1, 31 // (op 646) 
s_cmp_eq_i32 s10, 0 // (op 646) 
s_mov_b32 s8, scc // (op 646) 
s_cmp_eq_i32 s8, 0 // (op 646) 
s_cselect_b32 s9, 0, -1 // (op 646) 

s_and_b32 s8, s49, 31 // (op 646) 
s_lshl_b32 s56, 1, s8 // (op 646) 
s_add_i32 s57, s56, s9 // (op 646) 


s_and_b32 s9, s0, s57 // (op 646) 


s_add_i32 s0, s1, s9 // (op 646) 


s_ashr_i32 s1, s0, s8 // (op 646) 


s_lshl_b32 s0, s49, 24 // (op 646) 
s_ashr_i32 s8, s0, 31 // (op 646) 

s_xor_b32 s0, s1, s8 // (op 646) 

s_sub_i32 s1, s0, s8 // (op 646) 


s_mul_i32 s0, s46, s18 // (op 646) 

s_mul_i32 s8, s50, s0 // (op 646) 

s_mul_i32 s0, s1, s8 // (op 646) 

s_sub_u32 s8, s16, s0 // (op 646) 

s_mul_hi_i32 s0, s8, s15 // (op 646) 
s_add_i32 s9, s0, s8 // (op 646) 


s_ashr_i32 s0, s9, 31 // (op 646) 
s_cmp_eq_i32 s15, 0 // (op 646) 
s_mov_b32 s8, scc // (op 646) 
s_cmp_eq_i32 s8, 0 // (op 646) 
s_cselect_b32 s18, 0, -1 // (op 646) 

s_and_b32 s8, s48, 31 // (op 646) 
s_lshl_b32 s19, 1, s8 // (op 646) 
s_add_i32 s56, s19, s18 // (op 646) 


s_and_b32 s18, s0, s56 // (op 646) 


s_add_i32 s0, s9, s18 // (op 646) 


s_ashr_i32 s9, s0, s8 // (op 646) 


s_lshl_b32 s0, s48, 24 // (op 646) 
s_ashr_i32 s8, s0, 31 // (op 646) 

s_xor_b32 s0, s9, s8 // (op 646) 

s_sub_i32 s9, s0, s8 // (op 646) 


s_cmp_eq_i32 s1, 0 // (op 646) 
s_mov_b32 s0, scc // (op 646) 

s_cmp_eq_i32 s0, 0 // (op 646) 
s_cselect_b32 s1, s9, s51 // (op 646) 



s_lshl_b32 s0, s1, 4 // (op 646) 

s_mov_b32 s9, 0 // (op 646) convert
s_mov_b32 s8, s0 // (op 646) convert

v_mul_lo_u32 v15, v12, s23 // (op 646) most significant: low of low * high
v_mul_hi_u32 v13, v12, s22 // (op 646) most significant: high of low * low
v_mul_lo_u32 v14, v12, s22 // (op 646) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 646) most significant: sum

v_mov_b32 v13, s8 // (op 646) 
v_mov_b32 v20, s9 // (op 646) 
v_add_co_u32 v14, s[0:1], v14, v13 // (op 646) least significant half
v_addc_co_u32 v15, s[0:1], v15, v20, s[0:1] // (op 646) most significant half





v_mul_lo_u32 v13, v15, 6 // (op 646) most significant: low of high * low
v_mul_hi_u32 v20, v14, 6 // (op 646) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 646) least significant: low of low * low
v_add_u32 v13, v13, v20 // (op 646) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 646) 

v_mov_b32 v12, v14 // (op 646) convert

s_mul_i32 s1, s27, 6 // (op 648) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 648) least significant
s_mul_hi_u32 s51, s26, 6 // (op 648) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 648) most significant: sum

s_mul_i32 s9, s22, s1 // (op 648) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 648) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 648) least significant
s_mul_hi_u32 s18, s22, s0 // (op 648) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 648) most significant: sum
s_add_u32 s9, s9, s18 // (op 648) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 648) 

s_mov_b32 s56, s24 // (op 648) reg expression
s_mov_b32 s57, s25 // (op 648) reg expression
s_mov_b32 s58, s0 // (op 648) reg expression
s_mov_b32 s59, 131072 // (op 648) call()

s_mul_hi_i32 s51, s16, s11 // (op 665) 
v_lshlrev_b32 v13, 2, v1 // (op 665) 
s_add_i32 s0, s51, s16 // (op 665) 

s_ashr_i32 s1, s0, 31 // (op 665) 
s_cmp_eq_i32 s11, 0 // (op 665) 
s_mov_b32 s51, scc // (op 665) 
s_cmp_eq_i32 s51, 0 // (op 665) 
s_cselect_b32 s8, 0, -1 // (op 665) 

s_and_b32 s9, s12, 31 // (op 665) 
s_lshl_b32 s51, 1, s9 // (op 665) 
s_add_i32 s18, s51, s8 // (op 665) 


s_and_b32 s8, s1, s18 // (op 665) 


s_add_i32 s1, s0, s8 // (op 665) 


s_ashr_i32 s0, s1, s9 // (op 665) 


s_lshl_b32 s1, s12, 24 // (op 665) 
s_ashr_i32 s51, s1, 31 // (op 665) 

s_xor_b32 s1, s0, s51 // (op 665) 

s_sub_i32 s0, s1, s51 // (op 665) 


s_mul_hi_i32 s1, s0, s13 // (op 665) 
s_add_i32 s51, s1, s0 // (op 665) 

s_ashr_i32 s1, s51, 31 // (op 665) 
s_cmp_eq_i32 s13, 0 // (op 665) 
s_mov_b32 s8, scc // (op 665) 
s_cmp_eq_i32 s8, 0 // (op 665) 
s_cselect_b32 s9, 0, -1 // (op 665) 

s_and_b32 s8, s14, 31 // (op 665) 
s_lshl_b32 s18, 1, s8 // (op 665) 
s_add_i32 s19, s18, s9 // (op 665) 


s_and_b32 s9, s1, s19 // (op 665) 


s_add_i32 s1, s51, s9 // (op 665) 


s_ashr_i32 s9, s1, s8 // (op 665) 


s_lshl_b32 s1, s14, 24 // (op 665) 
s_ashr_i32 s8, s1, 31 // (op 665) 

s_xor_b32 s1, s9, s8 // (op 665) 

s_sub_i32 s9, s1, s8 // (op 665) 


s_add_u32 s18, s2, 16 // (op 665) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 665) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 665) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 665) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 665) 

s_mul_i32 s1, s9, s18 // (op 665) 

s_sub_i32 s51, s0, s1 // (op 665) 


s_mul_hi_i32 s0, s16, s10 // (op 665) 
s_add_i32 s1, s0, s16 // (op 665) 

s_ashr_i32 s0, s1, 31 // (op 665) 
s_cmp_eq_i32 s10, 0 // (op 665) 
s_mov_b32 s8, scc // (op 665) 
s_cmp_eq_i32 s8, 0 // (op 665) 
s_cselect_b32 s9, 0, -1 // (op 665) 

s_and_b32 s8, s49, 31 // (op 665) 
s_lshl_b32 s60, 1, s8 // (op 665) 
s_add_i32 s61, s60, s9 // (op 665) 


s_and_b32 s9, s0, s61 // (op 665) 


s_add_i32 s0, s1, s9 // (op 665) 


s_ashr_i32 s1, s0, s8 // (op 665) 


s_lshl_b32 s0, s49, 24 // (op 665) 
s_ashr_i32 s8, s0, 31 // (op 665) 

s_xor_b32 s0, s1, s8 // (op 665) 

s_sub_i32 s1, s0, s8 // (op 665) 


s_mul_i32 s0, s46, s18 // (op 665) 

s_mul_i32 s8, s50, s0 // (op 665) 

s_mul_i32 s0, s1, s8 // (op 665) 

s_sub_u32 s8, s16, s0 // (op 665) 

s_mul_hi_i32 s0, s8, s15 // (op 665) 
s_add_i32 s9, s0, s8 // (op 665) 


s_ashr_i32 s0, s9, 31 // (op 665) 
s_cmp_eq_i32 s15, 0 // (op 665) 
s_mov_b32 s8, scc // (op 665) 
s_cmp_eq_i32 s8, 0 // (op 665) 
s_cselect_b32 s18, 0, -1 // (op 665) 

s_and_b32 s8, s48, 31 // (op 665) 
s_lshl_b32 s19, 1, s8 // (op 665) 
s_add_i32 s60, s19, s18 // (op 665) 


s_and_b32 s18, s0, s60 // (op 665) 


s_add_i32 s0, s9, s18 // (op 665) 


s_ashr_i32 s9, s0, s8 // (op 665) 


s_lshl_b32 s0, s48, 24 // (op 665) 
s_ashr_i32 s8, s0, 31 // (op 665) 

s_xor_b32 s0, s9, s8 // (op 665) 

s_sub_i32 s9, s0, s8 // (op 665) 


s_cmp_eq_i32 s1, 0 // (op 665) 
s_mov_b32 s0, scc // (op 665) 

s_cmp_eq_i32 s0, 0 // (op 665) 
s_cselect_b32 s1, s9, s51 // (op 665) 



s_lshl_b32 s0, s1, 4 // (op 665) 

s_mov_b32 s9, 0 // (op 665) convert
s_mov_b32 s8, s0 // (op 665) convert

v_mul_lo_u32 v15, v13, s23 // (op 665) most significant: low of low * high
v_mul_hi_u32 v20, v13, s22 // (op 665) most significant: high of low * low
v_mul_lo_u32 v14, v13, s22 // (op 665) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 665) most significant: sum

v_mov_b32 v20, s8 // (op 665) 
v_mov_b32 v21, s9 // (op 665) 
v_add_co_u32 v14, s[0:1], v14, v20 // (op 665) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 665) most significant half





v_mul_lo_u32 v21, v15, 6 // (op 665) most significant: low of high * low
v_mul_hi_u32 v13, v14, 6 // (op 665) most significant: high of low * low
v_mul_lo_u32 v20, v14, 6 // (op 665) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 665) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 665) 

v_mov_b32 v13, v14 // (op 665) convert

s_mul_i32 s1, s27, 6 // (op 667) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 667) least significant
s_mul_hi_u32 s51, s26, 6 // (op 667) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 667) most significant: sum

s_mul_i32 s9, s22, s1 // (op 667) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 667) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 667) least significant
s_mul_hi_u32 s18, s22, s0 // (op 667) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 667) most significant: sum
s_add_u32 s9, s9, s18 // (op 667) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 667) 

s_mov_b32 s56, s24 // (op 667) reg expression
s_mov_b32 s57, s25 // (op 667) reg expression
s_mov_b32 s58, s0 // (op 667) reg expression
s_mov_b32 s59, 131072 // (op 667) call()

s_mul_hi_i32 s51, s16, s11 // (op 710) 
v_lshrrev_b32 v14, 3, v1 // (op 710) 
s_add_i32 s0, s51, s16 // (op 710) 

v_lshlrev_b32 v15, 1, v14 // (op 710) 

s_ashr_i32 s1, s0, 31 // (op 710) 
v_and_b32 v14, -8, v1 // (op 710) 
s_cmp_eq_i32 s11, 0 // (op 710) 
s_mov_b32 s51, scc // (op 710) 
v_sub_u32 v20, v1, v14 // (op 710) 

s_cmp_eq_i32 s51, 0 // (op 710) 
s_cselect_b32 s8, 0, -1 // (op 710) 

s_and_b32 s9, s12, 31 // (op 710) 
s_lshl_b32 s51, 1, s9 // (op 710) 
s_add_i32 s18, s51, s8 // (op 710) 


s_and_b32 s8, s1, s18 // (op 710) 


s_add_i32 s1, s0, s8 // (op 710) 


s_ashr_i32 s0, s1, s9 // (op 710) 


s_lshl_b32 s1, s12, 24 // (op 710) 
s_ashr_i32 s51, s1, 31 // (op 710) 

s_xor_b32 s1, s0, s51 // (op 710) 

s_sub_i32 s0, s1, s51 // (op 710) 


s_mul_hi_i32 s1, s0, s13 // (op 710) 
s_add_i32 s51, s1, s0 // (op 710) 

s_ashr_i32 s1, s51, 31 // (op 710) 
s_cmp_eq_i32 s13, 0 // (op 710) 
s_mov_b32 s8, scc // (op 710) 
s_cmp_eq_i32 s8, 0 // (op 710) 
s_cselect_b32 s9, 0, -1 // (op 710) 

s_and_b32 s8, s14, 31 // (op 710) 
s_lshl_b32 s18, 1, s8 // (op 710) 
s_add_i32 s19, s18, s9 // (op 710) 


s_and_b32 s9, s1, s19 // (op 710) 


s_add_i32 s1, s51, s9 // (op 710) 


s_ashr_i32 s9, s1, s8 // (op 710) 


s_lshl_b32 s1, s14, 24 // (op 710) 
s_ashr_i32 s8, s1, 31 // (op 710) 

s_xor_b32 s1, s9, s8 // (op 710) 

s_sub_i32 s9, s1, s8 // (op 710) 


s_mul_i32 s1, s0, s46 // (op 710) 

s_sub_u32 s0, s16, s1 // (op 710) 

s_mul_i32 s1, s9, s46 // (op 710) 
s_add_u32 s1, s1, s0 // (op 710) 


s_mul_hi_i32 s0, s16, s10 // (op 710) 
s_add_i32 s51, s0, s16 // (op 710) 

s_ashr_i32 s0, s51, 31 // (op 710) 
s_cmp_eq_i32 s10, 0 // (op 710) 
s_mov_b32 s8, scc // (op 710) 
s_cmp_eq_i32 s8, 0 // (op 710) 
s_cselect_b32 s9, 0, -1 // (op 710) 

s_and_b32 s8, s49, 31 // (op 710) 
s_lshl_b32 s18, 1, s8 // (op 710) 
s_add_i32 s19, s18, s9 // (op 710) 


s_and_b32 s9, s0, s19 // (op 710) 


s_add_i32 s0, s51, s9 // (op 710) 


s_ashr_i32 s9, s0, s8 // (op 710) 


s_lshl_b32 s0, s49, 24 // (op 710) 
s_ashr_i32 s8, s0, 31 // (op 710) 

s_xor_b32 s0, s9, s8 // (op 710) 

s_sub_i32 s9, s0, s8 // (op 710) 


s_add_u32 s18, s2, 16 // (op 710) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 710) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 710) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 710) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 710) 

s_mul_i32 s0, s46, s18 // (op 710) 

s_mul_i32 s8, s50, s0 // (op 710) 

s_mul_i32 s0, s9, s8 // (op 710) 

s_sub_u32 s8, s16, s0 // (op 710) 

s_mov_b32 s0, s8 // (op 710) 
s_mul_hi_i32 s51, s0, s15 // (op 710) 
s_add_i32 s18, s51, s0 // (op 710) 


s_ashr_i32 s0, s18, 31 // (op 710) 
s_cmp_eq_i32 s15, 0 // (op 710) 
s_mov_b32 s19, scc // (op 710) 
s_cmp_eq_i32 s19, 0 // (op 710) 
s_cselect_b32 s51, 0, -1 // (op 710) 

s_and_b32 s19, s48, 31 // (op 710) 
s_lshl_b32 s60, 1, s19 // (op 710) 
s_add_i32 s61, s60, s51 // (op 710) 


s_and_b32 s51, s0, s61 // (op 710) 


s_add_i32 s0, s18, s51 // (op 710) 


s_ashr_i32 s18, s0, s19 // (op 710) 


s_lshl_b32 s0, s48, 24 // (op 710) 
s_ashr_i32 s19, s0, 31 // (op 710) 

s_xor_b32 s0, s18, s19 // (op 710) 

s_sub_i32 s18, s0, s19 // (op 710) 


s_mul_i32 s0, s47, s46 // (op 710) 
s_add_u32 s60, s6, 16 // (op 710) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 710) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 710) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 710) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 710) 

s_sub_i32 s19, s60, s0 // (op 710) 

s_mul_i32 s51, s18, s19 // (op 710) 


s_sub_u32 s18, s8, s51 // (op 710) 


s_add_u32 s8, s18, s0 // (op 710) 


s_cmp_eq_i32 s9, 0 // (op 710) 
s_mov_b32 s0, scc // (op 710) 

s_cmp_eq_i32 s0, 0 // (op 710) 
s_cselect_b32 s9, s8, s1 // (op 710) 



v_lshl_add_u32 v14, s9, 4, v15 // (op 710) 


v_mul_lo_u32 v15, v14, s28 // (op 710) 
v_add_u32 v15, v15, v20 // (op 710) 


s_mul_i32 s1, s28, s7 // (op 712) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 712) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 712) least significant
s_mul_hi_u32 s8, s28, s6 // (op 712) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 712) most significant: sum
s_add_u32 s1, s1, s8 // (op 712) most significant: sum


s_mov_b32 s60, s30 // (op 712) reg expression
s_mov_b32 s61, s31 // (op 712) reg expression
s_mov_b32 s62, s0 // (op 712) reg expression
s_mov_b32 s63, 131072 // (op 712) call()

s_mul_hi_i32 s51, s16, s11 // (op 729) 
v_lshrrev_b32 v14, 3, v1 // (op 729) 
s_add_i32 s0, s51, s16 // (op 729) 

v_lshlrev_b32 v20, 1, v14 // (op 729) 

s_ashr_i32 s1, s0, 31 // (op 729) 
v_and_b32 v14, -8, v1 // (op 729) 
s_cmp_eq_i32 s11, 0 // (op 729) 
s_mov_b32 s51, scc // (op 729) 
v_sub_u32 v21, v1, v14 // (op 729) 

s_cmp_eq_i32 s51, 0 // (op 729) 
s_cselect_b32 s8, 0, -1 // (op 729) 

s_and_b32 s9, s12, 31 // (op 729) 
s_lshl_b32 s51, 1, s9 // (op 729) 
s_add_i32 s18, s51, s8 // (op 729) 


s_and_b32 s8, s1, s18 // (op 729) 


s_add_i32 s1, s0, s8 // (op 729) 


s_ashr_i32 s0, s1, s9 // (op 729) 


s_lshl_b32 s1, s12, 24 // (op 729) 
s_ashr_i32 s51, s1, 31 // (op 729) 

s_xor_b32 s1, s0, s51 // (op 729) 

s_sub_i32 s0, s1, s51 // (op 729) 


s_mul_hi_i32 s1, s0, s13 // (op 729) 
s_add_i32 s51, s1, s0 // (op 729) 

s_ashr_i32 s1, s51, 31 // (op 729) 
s_cmp_eq_i32 s13, 0 // (op 729) 
s_mov_b32 s8, scc // (op 729) 
s_cmp_eq_i32 s8, 0 // (op 729) 
s_cselect_b32 s9, 0, -1 // (op 729) 

s_and_b32 s8, s14, 31 // (op 729) 
s_lshl_b32 s18, 1, s8 // (op 729) 
s_add_i32 s19, s18, s9 // (op 729) 


s_and_b32 s9, s1, s19 // (op 729) 


s_add_i32 s1, s51, s9 // (op 729) 


s_ashr_i32 s9, s1, s8 // (op 729) 


s_lshl_b32 s1, s14, 24 // (op 729) 
s_ashr_i32 s8, s1, 31 // (op 729) 

s_xor_b32 s1, s9, s8 // (op 729) 

s_sub_i32 s9, s1, s8 // (op 729) 


s_mul_i32 s1, s0, s46 // (op 729) 

s_sub_u32 s0, s16, s1 // (op 729) 

s_mul_i32 s1, s9, s46 // (op 729) 
s_add_u32 s1, s1, s0 // (op 729) 


s_mul_hi_i32 s0, s16, s10 // (op 729) 
s_add_i32 s51, s0, s16 // (op 729) 

s_ashr_i32 s0, s51, 31 // (op 729) 
s_cmp_eq_i32 s10, 0 // (op 729) 
s_mov_b32 s8, scc // (op 729) 
s_cmp_eq_i32 s8, 0 // (op 729) 
s_cselect_b32 s9, 0, -1 // (op 729) 

s_and_b32 s8, s49, 31 // (op 729) 
s_lshl_b32 s18, 1, s8 // (op 729) 
s_add_i32 s19, s18, s9 // (op 729) 


s_and_b32 s9, s0, s19 // (op 729) 


s_add_i32 s0, s51, s9 // (op 729) 


s_ashr_i32 s9, s0, s8 // (op 729) 


s_lshl_b32 s0, s49, 24 // (op 729) 
s_ashr_i32 s8, s0, 31 // (op 729) 

s_xor_b32 s0, s9, s8 // (op 729) 

s_sub_i32 s9, s0, s8 // (op 729) 


s_add_u32 s18, s2, 16 // (op 729) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 729) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 729) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 729) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 729) 

s_mul_i32 s0, s46, s18 // (op 729) 

s_mul_i32 s8, s50, s0 // (op 729) 

s_mul_i32 s0, s9, s8 // (op 729) 

s_sub_u32 s8, s16, s0 // (op 729) 

s_mov_b32 s0, s8 // (op 729) 
s_mul_hi_i32 s51, s0, s15 // (op 729) 
s_add_i32 s18, s51, s0 // (op 729) 


s_ashr_i32 s0, s18, 31 // (op 729) 
s_cmp_eq_i32 s15, 0 // (op 729) 
s_mov_b32 s19, scc // (op 729) 
s_cmp_eq_i32 s19, 0 // (op 729) 
s_cselect_b32 s51, 0, -1 // (op 729) 

s_and_b32 s19, s48, 31 // (op 729) 
s_lshl_b32 s64, 1, s19 // (op 729) 
s_add_i32 s65, s64, s51 // (op 729) 


s_and_b32 s51, s0, s65 // (op 729) 


s_add_i32 s0, s18, s51 // (op 729) 


s_ashr_i32 s18, s0, s19 // (op 729) 


s_lshl_b32 s0, s48, 24 // (op 729) 
s_ashr_i32 s19, s0, 31 // (op 729) 

s_xor_b32 s0, s18, s19 // (op 729) 

s_sub_i32 s18, s0, s19 // (op 729) 


s_mul_i32 s0, s47, s46 // (op 729) 
s_add_u32 s64, s6, 16 // (op 729) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 729) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 729) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 729) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 729) 

s_sub_i32 s19, s64, s0 // (op 729) 

s_mul_i32 s51, s18, s19 // (op 729) 


s_sub_u32 s18, s8, s51 // (op 729) 


s_add_u32 s8, s18, s0 // (op 729) 


s_cmp_eq_i32 s9, 0 // (op 729) 
s_mov_b32 s0, scc // (op 729) 

s_cmp_eq_i32 s0, 0 // (op 729) 
s_cselect_b32 s9, s8, s1 // (op 729) 



v_lshl_add_u32 v14, s9, 4, v20 // (op 729) 


v_mul_lo_u32 v20, v14, s28 // (op 729) 
v_add_u32 v20, v20, v21 // (op 729) 


s_mul_i32 s1, s28, s7 // (op 731) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 731) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 731) least significant
s_mul_hi_u32 s8, s28, s6 // (op 731) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 731) most significant: sum
s_add_u32 s1, s1, s8 // (op 731) most significant: sum


s_mov_b32 s60, s30 // (op 731) reg expression
s_mov_b32 s61, s31 // (op 731) reg expression
s_mov_b32 s62, s0 // (op 731) reg expression
s_mov_b32 s63, 131072 // (op 731) call()

s_mul_hi_i32 s51, s16, s11 // (op 774) 
v_and_b32 v14, -16, v1 // (op 774) 
s_add_i32 s0, s51, s16 // (op 774) 

v_sub_u32 v21, v1, v14 // (op 774) 

s_ashr_i32 s1, s0, 31 // (op 774) 
v_lshrrev_b32 v14, 4, v1 // (op 774) 
s_cmp_eq_i32 s11, 0 // (op 774) 
s_mov_b32 s51, scc // (op 774) 
v_lshlrev_b32 v22, 1, v14 // (op 774) 

s_cmp_eq_i32 s51, 0 // (op 774) 
s_cselect_b32 s8, 0, -1 // (op 774) 

s_and_b32 s9, s12, 31 // (op 774) 
s_lshl_b32 s51, 1, s9 // (op 774) 
s_add_i32 s18, s51, s8 // (op 774) 


s_and_b32 s8, s1, s18 // (op 774) 


s_add_i32 s1, s0, s8 // (op 774) 


s_ashr_i32 s0, s1, s9 // (op 774) 


s_lshl_b32 s1, s12, 24 // (op 774) 
s_ashr_i32 s51, s1, 31 // (op 774) 

s_xor_b32 s1, s0, s51 // (op 774) 

s_sub_i32 s0, s1, s51 // (op 774) 


s_mul_hi_i32 s1, s0, s13 // (op 774) 
s_add_i32 s51, s1, s0 // (op 774) 

s_ashr_i32 s1, s51, 31 // (op 774) 
s_cmp_eq_i32 s13, 0 // (op 774) 
s_mov_b32 s8, scc // (op 774) 
s_cmp_eq_i32 s8, 0 // (op 774) 
s_cselect_b32 s9, 0, -1 // (op 774) 

s_and_b32 s8, s14, 31 // (op 774) 
s_lshl_b32 s18, 1, s8 // (op 774) 
s_add_i32 s19, s18, s9 // (op 774) 


s_and_b32 s9, s1, s19 // (op 774) 


s_add_i32 s1, s51, s9 // (op 774) 


s_ashr_i32 s9, s1, s8 // (op 774) 


s_lshl_b32 s1, s14, 24 // (op 774) 
s_ashr_i32 s8, s1, 31 // (op 774) 

s_xor_b32 s1, s9, s8 // (op 774) 

s_sub_i32 s9, s1, s8 // (op 774) 


s_add_u32 s18, s2, 16 // (op 774) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 774) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 774) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 774) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 774) 

s_mul_i32 s1, s9, s18 // (op 774) 

s_sub_i32 s51, s0, s1 // (op 774) 


s_mul_hi_i32 s0, s16, s10 // (op 774) 
s_add_i32 s1, s0, s16 // (op 774) 

s_ashr_i32 s0, s1, 31 // (op 774) 
s_cmp_eq_i32 s10, 0 // (op 774) 
s_mov_b32 s8, scc // (op 774) 
s_cmp_eq_i32 s8, 0 // (op 774) 
s_cselect_b32 s9, 0, -1 // (op 774) 

s_and_b32 s8, s49, 31 // (op 774) 
s_lshl_b32 s64, 1, s8 // (op 774) 
s_add_i32 s65, s64, s9 // (op 774) 


s_and_b32 s9, s0, s65 // (op 774) 


s_add_i32 s0, s1, s9 // (op 774) 


s_ashr_i32 s1, s0, s8 // (op 774) 


s_lshl_b32 s0, s49, 24 // (op 774) 
s_ashr_i32 s8, s0, 31 // (op 774) 

s_xor_b32 s0, s1, s8 // (op 774) 

s_sub_i32 s1, s0, s8 // (op 774) 


s_mul_i32 s0, s46, s18 // (op 774) 

s_mul_i32 s8, s50, s0 // (op 774) 

s_mul_i32 s0, s1, s8 // (op 774) 

s_sub_u32 s8, s16, s0 // (op 774) 

s_mul_hi_i32 s0, s8, s15 // (op 774) 
s_add_i32 s9, s0, s8 // (op 774) 


s_ashr_i32 s0, s9, 31 // (op 774) 
s_cmp_eq_i32 s15, 0 // (op 774) 
s_mov_b32 s8, scc // (op 774) 
s_cmp_eq_i32 s8, 0 // (op 774) 
s_cselect_b32 s18, 0, -1 // (op 774) 

s_and_b32 s8, s48, 31 // (op 774) 
s_lshl_b32 s19, 1, s8 // (op 774) 
s_add_i32 s64, s19, s18 // (op 774) 


s_and_b32 s18, s0, s64 // (op 774) 


s_add_i32 s0, s9, s18 // (op 774) 


s_ashr_i32 s9, s0, s8 // (op 774) 


s_lshl_b32 s0, s48, 24 // (op 774) 
s_ashr_i32 s8, s0, 31 // (op 774) 

s_xor_b32 s0, s9, s8 // (op 774) 

s_sub_i32 s9, s0, s8 // (op 774) 


s_cmp_eq_i32 s1, 0 // (op 774) 
s_mov_b32 s0, scc // (op 774) 

s_cmp_eq_i32 s0, 0 // (op 774) 
s_cselect_b32 s1, s9, s51 // (op 774) 



v_lshl_add_u32 v14, s1, 4, v21 // (op 774) 


v_mul_lo_u32 v21, v22, s32 // (op 774) 
v_add_u32 v21, v21, v14 // (op 774) 


s_mul_i32 s1, s32, s37 // (op 776) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 776) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 776) least significant
s_mul_hi_u32 s8, s32, s36 // (op 776) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 776) most significant: sum
s_add_u32 s1, s1, s8 // (op 776) most significant: sum


s_mov_b32 s64, s34 // (op 776) reg expression
s_mov_b32 s65, s35 // (op 776) reg expression
s_mov_b32 s66, s0 // (op 776) reg expression
s_mov_b32 s67, 131072 // (op 776) call()

s_mul_hi_i32 s51, s16, s11 // (op 793) 
v_and_b32 v14, -16, v1 // (op 793) 
s_add_i32 s0, s51, s16 // (op 793) 

v_sub_u32 v22, v1, v14 // (op 793) 

s_ashr_i32 s1, s0, 31 // (op 793) 
v_lshrrev_b32 v14, 4, v1 // (op 793) 
s_cmp_eq_i32 s11, 0 // (op 793) 
s_mov_b32 s51, scc // (op 793) 
v_lshlrev_b32 v23, 1, v14 // (op 793) 

s_cmp_eq_i32 s51, 0 // (op 793) 
s_cselect_b32 s8, 0, -1 // (op 793) 

s_and_b32 s9, s12, 31 // (op 793) 
s_lshl_b32 s51, 1, s9 // (op 793) 
s_add_i32 s18, s51, s8 // (op 793) 


s_and_b32 s8, s1, s18 // (op 793) 


s_add_i32 s1, s0, s8 // (op 793) 


s_ashr_i32 s0, s1, s9 // (op 793) 


s_lshl_b32 s1, s12, 24 // (op 793) 
s_ashr_i32 s51, s1, 31 // (op 793) 

s_xor_b32 s1, s0, s51 // (op 793) 

s_sub_i32 s0, s1, s51 // (op 793) 


s_mul_hi_i32 s1, s0, s13 // (op 793) 
s_add_i32 s51, s1, s0 // (op 793) 

s_ashr_i32 s1, s51, 31 // (op 793) 
s_cmp_eq_i32 s13, 0 // (op 793) 
s_mov_b32 s8, scc // (op 793) 
s_cmp_eq_i32 s8, 0 // (op 793) 
s_cselect_b32 s9, 0, -1 // (op 793) 

s_and_b32 s8, s14, 31 // (op 793) 
s_lshl_b32 s18, 1, s8 // (op 793) 
s_add_i32 s19, s18, s9 // (op 793) 


s_and_b32 s9, s1, s19 // (op 793) 


s_add_i32 s1, s51, s9 // (op 793) 


s_ashr_i32 s9, s1, s8 // (op 793) 


s_lshl_b32 s1, s14, 24 // (op 793) 
s_ashr_i32 s8, s1, 31 // (op 793) 

s_xor_b32 s1, s9, s8 // (op 793) 

s_sub_i32 s9, s1, s8 // (op 793) 


s_add_u32 s18, s2, 16 // (op 793) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 793) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 793) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 793) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 793) 

s_mul_i32 s1, s9, s18 // (op 793) 

s_sub_i32 s51, s0, s1 // (op 793) 


s_mul_hi_i32 s0, s16, s10 // (op 793) 
s_add_i32 s1, s0, s16 // (op 793) 

s_ashr_i32 s0, s1, 31 // (op 793) 
s_cmp_eq_i32 s10, 0 // (op 793) 
s_mov_b32 s8, scc // (op 793) 
s_cmp_eq_i32 s8, 0 // (op 793) 
s_cselect_b32 s9, 0, -1 // (op 793) 

s_and_b32 s8, s49, 31 // (op 793) 
s_lshl_b32 s68, 1, s8 // (op 793) 
s_add_i32 s69, s68, s9 // (op 793) 


s_and_b32 s9, s0, s69 // (op 793) 


s_add_i32 s0, s1, s9 // (op 793) 


s_ashr_i32 s1, s0, s8 // (op 793) 


s_lshl_b32 s0, s49, 24 // (op 793) 
s_ashr_i32 s8, s0, 31 // (op 793) 

s_xor_b32 s0, s1, s8 // (op 793) 

s_sub_i32 s1, s0, s8 // (op 793) 


s_mul_i32 s0, s46, s18 // (op 793) 

s_mul_i32 s8, s50, s0 // (op 793) 

s_mul_i32 s0, s1, s8 // (op 793) 

s_sub_u32 s8, s16, s0 // (op 793) 

s_mul_hi_i32 s0, s8, s15 // (op 793) 
s_add_i32 s9, s0, s8 // (op 793) 


s_ashr_i32 s0, s9, 31 // (op 793) 
s_cmp_eq_i32 s15, 0 // (op 793) 
s_mov_b32 s8, scc // (op 793) 
s_cmp_eq_i32 s8, 0 // (op 793) 
s_cselect_b32 s18, 0, -1 // (op 793) 

s_and_b32 s8, s48, 31 // (op 793) 
s_lshl_b32 s19, 1, s8 // (op 793) 
s_add_i32 s68, s19, s18 // (op 793) 


s_and_b32 s18, s0, s68 // (op 793) 


s_add_i32 s0, s9, s18 // (op 793) 


s_ashr_i32 s9, s0, s8 // (op 793) 


s_lshl_b32 s0, s48, 24 // (op 793) 
s_ashr_i32 s8, s0, 31 // (op 793) 

s_xor_b32 s0, s9, s8 // (op 793) 

s_sub_i32 s9, s0, s8 // (op 793) 


s_cmp_eq_i32 s1, 0 // (op 793) 
s_mov_b32 s0, scc // (op 793) 

s_cmp_eq_i32 s0, 0 // (op 793) 
s_cselect_b32 s1, s9, s51 // (op 793) 



v_lshl_add_u32 v14, s1, 4, v22 // (op 793) 


v_mul_lo_u32 v22, v23, s32 // (op 793) 
v_add_u32 v22, v22, v14 // (op 793) 


s_mul_i32 s1, s32, s37 // (op 795) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 795) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 795) least significant
s_mul_hi_u32 s8, s32, s36 // (op 795) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 795) most significant: sum
s_add_u32 s1, s1, s8 // (op 795) most significant: sum


s_mov_b32 s64, s34 // (op 795) reg expression
s_mov_b32 s65, s35 // (op 795) reg expression
s_mov_b32 s66, s0 // (op 795) reg expression
s_mov_b32 s67, 131072 // (op 795) call()


s_mul_i32 s1, s23, 6 // (op 424) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 424) least significant
s_mul_hi_u32 s51, s22, 6 // (op 424) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 424) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 424) 


v_mov_b32 v14, v12 // (op 424) reg expression
buffer_load_dwordx3 v[26:28], v14, s[56:59], 0 offen // (op 424) Load value
v_add_u32 v14, v14, v24 // (op 424) 
buffer_load_dwordx3 v[30:32], v14, s[56:59], 0 offen // (op 424) Load value
v_add_u32 v14, v14, v24 // (op 424) 
buffer_load_dwordx3 v[34:36], v14, s[56:59], 0 offen // (op 424) Load value
v_add_u32 v14, v14, v24 // (op 424) 
buffer_load_dwordx3 v[38:40], v14, s[56:59], 0 offen // (op 424) Load value



v_mov_b64 v[24:25], s[32:33] // (op 427) convert

v_mov_b32 v14, v21 // (op 427) reg expression
buffer_load_ubyte v42, v14, s[64:67], 0 offen offset: 0 // (op 427) Load value
v_add_u32 v14, v14, v24 // (op 427) 
buffer_load_ubyte v43, v14, s[64:67], 0 offen offset: 0 // (op 427) Load value


v_and_b32 v14, -16, v1 // (op 859) 
v_sub_u32 v29, v1, v14 // (op 859) 

v_lshlrev_b32 v14, 4, v29 // (op 859) 

v_lshrrev_b32 v29, 4, v1 // (op 859) 
v_lshl_add_u32 v33, v29, 8, v14 // (op 859) 


s_nop 0
v_readfirstlane_b32 s51, v33 // (op 859) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[4:5], 2 // (op 538) 
v_mov_b64 v[24:25], s[0:1] // (op 538) convert


v_mov_b64 v[44:45], v[16:17] // (op 538) 
s_mov_b32 m0, s51 // (op 538) reg expression
buffer_load_dwordx4 v44, s[52:55], 0 offen offset: 0 lds // (op 538) Load value direct to lds
v_add_co_u32 v44, s[0:1], v44, v24 // (op 538) least significant half
v_addc_co_u32 v45, s[0:1], 0, v45, s[0:1] // (op 538) most significant half

s_add_u32 m0, m0, 1024 // (op 538) 
buffer_load_dwordx4 v44, s[52:55], 0 offen offset: 0 lds // (op 538) Load value direct to lds
v_add_co_u32 v44, s[0:1], v44, v24 // (op 538) least significant half
v_addc_co_u32 v45, s[0:1], 0, v45, s[0:1] // (op 538) most significant half

s_add_u32 m0, m0, 1024 // (op 538) 
buffer_load_dwordx4 v44, s[52:55], 0 offen offset: 0 lds // (op 538) Load value direct to lds
v_add_co_u32 v44, s[0:1], v44, v24 // (op 538) least significant half
v_addc_co_u32 v45, s[0:1], 0, v45, s[0:1] // (op 538) most significant half

s_add_u32 m0, m0, 1024 // (op 538) 
buffer_load_dwordx4 v44, s[52:55], 0 offen offset: 0 lds // (op 538) Load value direct to lds




v_mov_b64 v[24:25], s[28:29] // (op 433) convert

v_mov_b32 v14, v15 // (op 433) 
buffer_load_ubyte v44, v14, s[60:63], 0 offen offset: 0 // (op 433) Load value
v_add_u32 v14, v14, v24 // (op 433) 
buffer_load_ubyte v45, v14, s[60:63], 0 offen offset: 0 // (op 433) Load value



s_mul_i32 s1, s23, 6 // (op 436) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 436) least significant
s_mul_hi_u32 s51, s22, 6 // (op 436) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 436) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 436) 


s_lshl_b64 s[0:1], s[22:23], 8 // (op 436) 
s_mul_i32 s9, s1, 6 // (op 436) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 436) least significant
s_mul_hi_u32 s51, s0, 6 // (op 436) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 436) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 436) 

v_add_u32 v14, v12, s0 // (op 436) 

buffer_load_dwordx3 v[46:48], v14, s[56:59], 0 offen // (op 436) Load value
v_add_u32 v14, v14, v24 // (op 436) 
buffer_load_dwordx3 v[50:52], v14, s[56:59], 0 offen // (op 436) Load value
v_add_u32 v14, v14, v24 // (op 436) 
buffer_load_dwordx3 v[54:56], v14, s[56:59], 0 offen // (op 436) Load value
v_add_u32 v14, v14, v24 // (op 436) 
buffer_load_dwordx3 v[58:60], v14, s[56:59], 0 offen // (op 436) Load value




v_mov_b64 v[24:25], s[32:33] // (op 439) convert

s_lshl_b32 s51, s32, 3 // (op 439) 
v_add_u32 v12, v21, s51 // (op 439) 

buffer_load_ubyte v62, v12, s[64:67], 0 offen offset: 0 // (op 439) Load value
v_add_u32 v12, v12, v24 // (op 439) 
buffer_load_ubyte v63, v12, s[64:67], 0 offen offset: 0 // (op 439) Load value



v_and_b32 v12, -16, v1 // (op 1091) 
v_sub_u32 v14, v1, v12 // (op 1091) 

v_lshlrev_b32 v12, 4, v14 // (op 1091) 

v_lshrrev_b32 v14, 4, v1 // (op 1091) 
v_lshl_add_u32 v21, v14, 8, v12 // (op 1091) 


s_nop 0
v_readfirstlane_b32 s51, v21 // (op 1091) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[4:5], 2 // (op 543) 
v_mov_b64 v[24:25], s[0:1] // (op 543) convert


v_mov_b64 v[64:65], v[16:17] // (op 543) 
s_add_u32 s0, 4096, s51 // (op 543) 
s_mov_b32 m0, s0 // (op 543) reg expression
s_mov_b32 s1, 256 // (op 543) call()
buffer_load_dwordx4 v64, s[52:55], s1 offen offset: 0 lds // (op 543) Load value direct to lds
v_add_co_u32 v64, s[8:9], v64, v24 // (op 543) least significant half
v_addc_co_u32 v65, s[8:9], 0, v65, s[8:9] // (op 543) most significant half

s_add_u32 m0, m0, 1024 // (op 543) 
buffer_load_dwordx4 v64, s[52:55], s1 offen offset: 0 lds // (op 543) Load value direct to lds
v_add_co_u32 v64, s[8:9], v64, v24 // (op 543) least significant half
v_addc_co_u32 v65, s[8:9], 0, v65, s[8:9] // (op 543) most significant half

s_add_u32 m0, m0, 1024 // (op 543) 
buffer_load_dwordx4 v64, s[52:55], s1 offen offset: 0 lds // (op 543) Load value direct to lds
v_add_co_u32 v64, s[8:9], v64, v24 // (op 543) least significant half
v_addc_co_u32 v65, s[8:9], 0, v65, s[8:9] // (op 543) most significant half

s_add_u32 m0, m0, 1024 // (op 543) 
buffer_load_dwordx4 v64, s[52:55], s1 offen offset: 0 lds // (op 543) Load value direct to lds







v_mov_b64 v[16:17], s[28:29] // (op 445) convert

v_mov_b32 v12, v15 // (op 445) 
buffer_load_ubyte v24, v12, s[60:63], 0 offen offset:8 // (op 445) Load value
v_add_u32 v12, v12, v16 // (op 445) 
buffer_load_ubyte v25, v12, s[60:63], 0 offen offset:8 // (op 445) Load value



v_lshlrev_b32 v12, 6, v1 // (op 912) 
v_mul_lo_u32 v21, v12, 6 // (op 912) 
v_lshrrev_b32 v23, 3, v21 // (op 912) 

v_lshrrev_b32 v21, 4, v12 // (op 912) 

v_lshlrev_b32 v12, 2, v21 // (op 912) 

v_add_u32 v21, v23, v12 // (op 912) 





s_waitcnt vmcnt(23)
ds_write_b96 v21, v[26:28] offset:8192 // (op 447) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v21, v[30:32] offset:8208 // (op 447) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v21, v[34:36] offset:8224 // (op 447) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v21, v[38:40] offset:8240 // (op 447) Store local data 

v_and_b32 v12, -16, v1 // (op 1018) 
v_sub_u32 v21, v1, v12 // (op 1018) 

v_lshrrev_b32 v12, 4, v1 // (op 1018) 
v_lshl_add_u32 v23, v12, 5, v21 // (op 1018) 





s_waitcnt vmcnt(19)
ds_write_b8 v23, v42 offset:12288 // (op 448) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v23, v43 offset:12304 // (op 448) Store local data 

v_and_b32 v12, -8, v1 // (op 965) 
v_sub_u32 v21, v1, v12 // (op 965) 

v_lshrrev_b32 v12, 3, v1 // (op 965) 
v_lshl_add_u32 v23, v12, 4, v21 // (op 965) 





s_waitcnt vmcnt(13)
ds_write_b8 v23, v44 offset:12416 // (op 450) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v23, v45 offset:12424 // (op 450) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1231) 



ds_read_b96_tr_b6 v[64:66], v5 offset:8192 // (op 467) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v5 offset:8448 // (op 467) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1272) 


ds_read_u8 v12, v7 offset:12288 // (op 473) Load local data 
s_barrier  // (op 1212) 



ds_read_b128 v[70:73], v3 // (op 464) Load local data 
ds_read_b128 v[74:77], v3 offset:64 // (op 464) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1252) 



ds_read_u8 v21, v0 offset:12416 // (op 470) Load local data 
v_and_b32 v23, -16, v1 // (op 831) 
v_sub_u32 v29, v1, v23 // (op 831) 

v_lshlrev_b32 v23, 4, v29 // (op 831) 

v_lshrrev_b32 v29, 4, v1 // (op 831) 
v_lshl_add_u32 v33, v29, 8, v23 // (op 831) 


s_nop 0
v_readfirstlane_b32 s51, v33 // (op 831) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v23, 6, v1 // (op 883) 
v_mul_lo_u32 v29, v23, 6 // (op 883) 
v_lshrrev_b32 v33, 3, v29 // (op 883) 

v_lshrrev_b32 v29, 4, v23 // (op 883) 

v_lshlrev_b32 v23, 2, v29 // (op 883) 

v_add_u32 v29, v33, v23 // (op 883) 


v_and_b32 v23, -8, v1 // (op 936) 
v_sub_u32 v33, v1, v23 // (op 936) 

v_lshrrev_b32 v23, 3, v1 // (op 936) 
v_lshl_add_u32 v37, v23, 4, v33 // (op 936) 


v_and_b32 v23, -16, v1 // (op 989) 
v_sub_u32 v33, v1, v23 // (op 989) 

v_lshrrev_b32 v23, 4, v1 // (op 989) 
v_lshl_add_u32 v41, v23, 5, v33 // (op 989) 


v_and_b32 v23, -16, v1 // (op 1076) 
v_sub_u32 v33, v1, v23 // (op 1076) 

v_lshlrev_b32 v23, 4, v33 // (op 1076) 

v_lshrrev_b32 v33, 4, v1 // (op 1076) 
v_lshl_add_u32 v49, v33, 8, v23 // (op 1076) 


s_nop 0
v_readfirstlane_b32 s0, v49 // (op 1076) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v23, 6, v1 // (op 1115) 
v_mul_lo_u32 v33, v23, 6 // (op 1115) 
v_lshrrev_b32 v49, 3, v33 // (op 1115) 

v_lshrrev_b32 v33, 4, v23 // (op 1115) 

v_lshlrev_b32 v23, 2, v33 // (op 1115) 

v_add_u32 v33, v49, v23 // (op 1115) 


v_and_b32 v23, -8, v1 // (op 1139) 
v_sub_u32 v49, v1, v23 // (op 1139) 

v_lshrrev_b32 v23, 3, v1 // (op 1139) 
v_lshl_add_u32 v53, v23, 4, v49 // (op 1139) 


v_and_b32 v23, -16, v1 // (op 1163) 
v_sub_u32 v49, v1, v23 // (op 1163) 

v_lshrrev_b32 v23, 4, v1 // (op 1163) 
v_lshl_add_u32 v57, v23, 5, v49 // (op 1163) 


s_mov_b32 s1, 0 // (op 59) call()
s_ashr_i64 s[8:9], s[26:27], 8 // (op 58) 
s_mov_b32 s18, s8 // (op 58) convert

s_and_b32 s19, s18, 4294967294 // (op 58) 

s_cmp_lt_u32 s1, s19 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58 if false)
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58:

s_add_u32 s1, s1, 2 // (op 60) 
s_mul_i32 s9, s23, 6 // (op 10) most significant: low of high * low
s_mul_i32 s8, s22, 6 // (op 10) least significant
s_mul_hi_u32 s18, s22, 6 // (op 10) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 10) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[8:9] // (op 10) 


v_mov_b32 v23, v13 // (op 10) 
s_lshl_b64 s[8:9], s[22:23], 8 // (op 10) 
s_mul_i32 s19, s9, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s8, 6 // (op 10) least significant
s_mul_hi_u32 s68, s8, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 10) most significant: sum


s_lshr_b64 s[8:9], s[18:19], 3 // (op 10) 

s_lshl_b32 s18, s8, 1 // (op 10) 

v_add_u32 v49, v23, s18 // (op 10) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[64:69], a[0:3], v21, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 88) 
buffer_load_dwordx3 v[26:28], v49, s[56:59], 0 offen // (op 10) Load value
s_barrier  // (op 1221) 
v_add_u32 v49, v49, v14 // (op 10) 



ds_read_b128 v[70:73], v3 offset:128 // (op 91) Load local data 
ds_read_b128 v[74:77], v3 offset:192 // (op 91) Load local data 
buffer_load_dwordx3 v[30:32], v49, s[56:59], 0 offen // (op 10) Load value
s_barrier  // (op 1241) 
v_add_u32 v49, v49, v14 // (op 10) 



ds_read_b96_tr_b6 v[64:66], v5 offset:10240 // (op 95) Transpose load local data 
buffer_load_dwordx3 v[34:36], v49, s[56:59], 0 offen // (op 10) Load value
v_add_u32 v49, v49, v14 // (op 10) 
ds_read_b96_tr_b6 v[67:69], v5 offset:10496 // (op 95) Transpose load local data 
buffer_load_dwordx3 v[38:40], v49, s[56:59], 0 offen // (op 10) Load value



s_barrier  // (op 1262) 
v_mov_b64 v[14:15], s[32:33] // (op 25) convert




ds_read_u8 v21, v0 offset:12420 // (op 99) Load local data 
v_mov_b32 v49, v22 // (op 25) 
s_barrier  // (op 1282) 
s_lshl_b32 s8, s32, 4 // (op 25) 


ds_read_u8 v12, v7 offset:12352 // (op 103) Load local data 
v_add_u32 v61, v49, s8 // (op 25) 

buffer_load_ubyte v42, v61, s[64:67], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v61, v61, v14 // (op 25) 
buffer_load_ubyte v43, v61, s[64:67], 0 offen offset: 0 // (op 25) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[64:69], a[0:3], v21, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 105) 
s_barrier  // (op 1215) 

s_lshl_b64 s[8:9], s[4:5], 2 // (op 522) 
v_mov_b64 v[14:15], s[8:9] // (op 522) convert


v_mov_b64 v[16:17], v[18:19] // (op 522) 
v_mov_b64 v[78:79], v[16:17] // (op 522) 
s_mov_b32 m0, s51 // (op 522) reg expression
s_mov_b32 s8, 512 // (op 522) call()
buffer_load_dwordx4 v78, s[52:55], s8 offen offset: 0 lds // (op 522) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 522) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 522) most significant half

s_add_u32 m0, m0, 1024 // (op 522) 
buffer_load_dwordx4 v78, s[52:55], s8 offen offset: 0 lds // (op 522) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 522) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 522) most significant half

s_add_u32 m0, m0, 1024 // (op 522) 
buffer_load_dwordx4 v78, s[52:55], s8 offen offset: 0 lds // (op 522) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 522) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 522) most significant half

s_add_u32 m0, m0, 1024 // (op 522) 
buffer_load_dwordx4 v78, s[52:55], s8 offen offset: 0 lds // (op 522) Load value direct to lds




v_mov_b64 v[14:15], s[28:29] // (op 17) convert

v_mov_b32 v61, v20 // (op 17) 
v_mov_b32 v78, v61 // (op 17) 
buffer_load_ubyte v44, v78, s[60:63], 0 offen offset:16 // (op 17) Load value
v_add_u32 v78, v78, v14 // (op 17) 
buffer_load_ubyte v45, v78, s[60:63], 0 offen offset:16 // (op 17) Load value


s_barrier  // (op 1302) 



s_waitcnt vmcnt(23)
ds_write_b96 v33, v[46:48] offset:12544 // (op 302) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v33, v[50:52] offset:12560 // (op 302) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v33, v[54:56] offset:12576 // (op 302) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v33, v[58:60] offset:12592 // (op 302) Store local data 
s_barrier  // (op 1314) 



s_waitcnt vmcnt(19)
ds_write_b8 v57, v62 offset:16640 // (op 304) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v57, v63 offset:16656 // (op 304) Store local data 
s_barrier  // (op 1308) 



s_waitcnt vmcnt(13)
ds_write_b8 v53, v24 offset:16768 // (op 303) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v53, v25 offset:16776 // (op 303) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1298) 



ds_read_b96_tr_b6 v[78:80], v9 offset:12544 // (op 287) Transpose load local data 
ds_read_b96_tr_b6 v[81:83], v9 offset:12800 // (op 287) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1311) 


ds_read_u8 v14, v11 offset:16640 // (op 291) Load local data 
s_barrier  // (op 1292) 



ds_read_b128 v[84:87], v6 offset:4096 // (op 285) Load local data 
ds_read_b128 v[88:91], v6 offset:4160 // (op 285) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1305) 



ds_read_u8 v15, v4 offset:16768 // (op 289) Load local data 

s_mul_i32 s9, s23, 6 // (op 305) most significant: low of high * low
s_mul_i32 s8, s22, 6 // (op 305) least significant
s_mul_hi_u32 s18, s22, 6 // (op 305) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 305) most significant: sum

v_lshrrev_b64 v[92:93], 3, s[8:9] // (op 305) 


s_lshl_b64 s[8:9], s[22:23], 8 // (op 305) 
s_mul_i32 s19, s9, 6 // (op 305) most significant: low of high * low
s_mul_i32 s18, s8, 6 // (op 305) least significant
s_mul_hi_u32 s68, s8, 6 // (op 305) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 305) most significant: sum


s_lshr_b64 s[8:9], s[18:19], 3 // (op 305) 

s_mul_i32 s18, 3, s8 // (op 305) 

v_add_u32 v94, v23, s18 // (op 305) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:91], v[78:83], a[0:3], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 292) 
buffer_load_dwordx3 v[46:48], v94, s[56:59], 0 offen // (op 305) Load value




v_add_u32 v94, v94, v92 // (op 305) 



ds_read_b128 v[78:81], v6 offset:4224 // (op 294) Load local data 
ds_read_b128 v[82:85], v6 offset:4288 // (op 294) Load local data 
buffer_load_dwordx3 v[50:52], v94, s[56:59], 0 offen // (op 305) Load value
v_add_u32 v94, v94, v92 // (op 305) 



ds_read_b96_tr_b6 v[86:88], v9 offset:14592 // (op 296) Transpose load local data 
buffer_load_dwordx3 v[54:56], v94, s[56:59], 0 offen // (op 305) Load value
v_add_u32 v94, v94, v92 // (op 305) 
ds_read_b96_tr_b6 v[89:91], v9 offset:14848 // (op 296) Transpose load local data 
buffer_load_dwordx3 v[58:60], v94, s[56:59], 0 offen // (op 305) Load value







s_barrier  // (op 1369) 
ds_read_u8 v23, v4 offset:16772 // (op 298) Load local data 
v_mov_b64 v[14:15], s[32:33] // (op 307) convert

s_lshl_b32 s8, s32, 3 // (op 307) 


s_barrier  // (op 1373) 
ds_read_u8 v92, v11 offset:16704 // (op 300) Load local data 
s_mul_i32 s9, 3, s8 // (op 307) 

v_add_u32 v93, v49, s9 // (op 307) 

s_barrier  // (op 1377) 
buffer_load_ubyte v62, v93, s[64:67], 0 offen offset: 0 // (op 307) Load value
v_add_u32 v93, v93, v14 // (op 307) 
s_lshl_b64 s[8:9], s[22:23], 8 // (op 657) 
buffer_load_ubyte v63, v93, s[64:67], 0 offen offset: 0 // (op 307) Load value


s_mul_i32 s19, s9, 6 // (op 657) most significant: low of high * low

s_mul_i32 s18, s8, 6 // (op 657) least significant
s_mul_hi_u32 s68, s8, 6 // (op 657) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 657) most significant: sum


s_lshl_b32 s8, s32, 4 // (op 785) 
v_add_u32 v22, v22, s8 // (op 785) 

s_lshr_b64 s[8:9], s[18:19], 3 // (op 657) 

s_lshl_b32 s18, s8, 1 // (op 657) 

v_add_u32 v13, v13, s18 // (op 657) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:85], v[86:91], a[0:3], v23, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 301) 
s_barrier  // (op 1295) 





s_lshl_b64 s[8:9], s[4:5], 2 // (op 533) 
v_mov_b64 v[14:15], s[8:9] // (op 533) convert


v_mov_b64 v[78:79], v[16:17] // (op 533) 
s_add_u32 s8, 4096, s0 // (op 533) 
s_mov_b32 m0, s8 // (op 533) reg expression
s_mov_b32 s9, 768 // (op 533) call()
buffer_load_dwordx4 v78, s[52:55], s9 offen offset: 0 lds // (op 533) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 533) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 533) most significant half

s_add_u32 m0, m0, 1024 // (op 533) 
buffer_load_dwordx4 v78, s[52:55], s9 offen offset: 0 lds // (op 533) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 533) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 533) most significant half

s_add_u32 m0, m0, 1024 // (op 533) 
buffer_load_dwordx4 v78, s[52:55], s9 offen offset: 0 lds // (op 533) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 533) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 533) most significant half

s_add_u32 m0, m0, 1024 // (op 533) 
buffer_load_dwordx4 v78, s[52:55], s9 offen offset: 0 lds // (op 533) Load value direct to lds






v_add_co_u32 v18, vcc, 512, v18 // (op 593) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 593) most significant half
v_mov_b64 v[14:15], s[28:29] // (op 306) convert

v_mov_b32 v23, v61 // (op 306) 
buffer_load_ubyte v24, v23, s[60:63], 0 offen offset:24 // (op 306) Load value
v_add_u32 v23, v23, v14 // (op 306) 
buffer_load_ubyte v25, v23, s[60:63], 0 offen offset:24 // (op 306) Load value



s_barrier  // (op 1234) 
v_add_u32 v20, 16, v20 // (op 721) 



s_waitcnt vmcnt(23)
ds_write_b96 v29, v[26:28] offset:8192 // (op 13) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v29, v[30:32] offset:8208 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v29, v[34:36] offset:8224 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v29, v[38:40] offset:8240 // (op 13) Store local data 
s_barrier  // (op 1275) 



s_waitcnt vmcnt(19)
ds_write_b8 v41, v42 offset:12288 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v41, v43 offset:12304 // (op 30) Store local data 
s_barrier  // (op 1255) 



s_waitcnt vmcnt(13)
ds_write_b8 v37, v44 offset:12416 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v37, v45 offset:12424 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1244) 



ds_read_b96_tr_b6 v[64:66], v5 offset:8192 // (op 75) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v5 offset:8448 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1285) 


ds_read_u8 v12, v7 offset:12288 // (op 85) Load local data 
s_barrier  // (op 1224) 



ds_read_b128 v[70:73], v3 // (op 70) Load local data 
ds_read_b128 v[74:77], v3 offset:64 // (op 70) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1265) 



ds_read_u8 v21, v0 offset:12416 // (op 80) Load local data 
s_ashr_i64 s[8:9], s[26:27], 8 // (op 58) 
s_mov_b32 s18, s8 // (op 58) convert

s_and_b32 s19, s18, 4294967294 // (op 58) 

s_cmp_lt_u32 s1, s19 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58 if true)
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58:
















s_barrier  // (op 1327) 
s_mul_hi_i32 s51, s16, s11 // (op 556) 
v_lshrrev_b32 v49, 4, v1 // (op 556) 
s_add_i32 s0, s51, s16 // (op 556) 

v_and_b32 v53, -16, v1 // (op 556) 
s_ashr_i32 s1, s0, 31 // (op 556) 
v_sub_u32 v57, v1, v53 // (op 556) 

s_cmp_eq_i32 s11, 0 // (op 556) 
s_mov_b32 s51, scc // (op 556) 
v_lshlrev_b32 v53, 4, v57 // (op 556) 

s_cmp_eq_i32 s51, 0 // (op 556) 
s_cselect_b32 s8, 0, -1 // (op 556) 

s_and_b32 s9, s12, 31 // (op 556) 
s_lshl_b32 s51, 1, s9 // (op 556) 
s_add_i32 s18, s51, s8 // (op 556) 


s_and_b32 s8, s1, s18 // (op 556) 


s_add_i32 s1, s0, s8 // (op 556) 


s_ashr_i32 s0, s1, s9 // (op 556) 


s_lshl_b32 s1, s12, 24 // (op 556) 
s_ashr_i32 s51, s1, 31 // (op 556) 

s_xor_b32 s1, s0, s51 // (op 556) 

s_sub_i32 s0, s1, s51 // (op 556) 


s_mul_hi_i32 s1, s0, s13 // (op 556) 
s_add_i32 s51, s1, s0 // (op 556) 

s_ashr_i32 s1, s51, 31 // (op 556) 
s_cmp_eq_i32 s13, 0 // (op 556) 
s_mov_b32 s8, scc // (op 556) 
s_cmp_eq_i32 s8, 0 // (op 556) 
s_cselect_b32 s9, 0, -1 // (op 556) 

s_and_b32 s8, s14, 31 // (op 556) 
s_lshl_b32 s18, 1, s8 // (op 556) 
s_add_i32 s19, s18, s9 // (op 556) 


s_and_b32 s9, s1, s19 // (op 556) 


s_add_i32 s1, s51, s9 // (op 556) 


s_ashr_i32 s9, s1, s8 // (op 556) 


s_lshl_b32 s1, s14, 24 // (op 556) 
s_ashr_i32 s8, s1, 31 // (op 556) 

s_xor_b32 s1, s9, s8 // (op 556) 

s_sub_i32 s9, s1, s8 // (op 556) 


s_mul_i32 s1, s0, s46 // (op 556) 

s_sub_u32 s0, s16, s1 // (op 556) 

s_mul_i32 s1, s9, s46 // (op 556) 
s_add_u32 s1, s1, s0 // (op 556) 


s_mul_hi_i32 s0, s16, s10 // (op 556) 
s_add_i32 s51, s0, s16 // (op 556) 

s_ashr_i32 s0, s51, 31 // (op 556) 
s_cmp_eq_i32 s10, 0 // (op 556) 
s_mov_b32 s8, scc // (op 556) 
s_cmp_eq_i32 s8, 0 // (op 556) 
s_cselect_b32 s9, 0, -1 // (op 556) 

s_and_b32 s8, s49, 31 // (op 556) 
s_lshl_b32 s18, 1, s8 // (op 556) 
s_add_i32 s19, s18, s9 // (op 556) 


s_and_b32 s9, s0, s19 // (op 556) 


s_add_i32 s0, s51, s9 // (op 556) 


s_ashr_i32 s9, s0, s8 // (op 556) 


s_lshl_b32 s0, s49, 24 // (op 556) 
s_ashr_i32 s8, s0, 31 // (op 556) 

s_xor_b32 s0, s9, s8 // (op 556) 

s_sub_i32 s9, s0, s8 // (op 556) 


s_add_u32 s18, s2, 16 // (op 556) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 556) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 556) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 556) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 556) 

s_mul_i32 s0, s46, s18 // (op 556) 

s_mul_i32 s8, s50, s0 // (op 556) 

s_mul_i32 s0, s9, s8 // (op 556) 

s_sub_u32 s8, s16, s0 // (op 556) 

s_mov_b32 s0, s8 // (op 556) 
s_mul_hi_i32 s51, s0, s15 // (op 556) 
s_add_i32 s18, s51, s0 // (op 556) 


s_ashr_i32 s0, s18, 31 // (op 556) 
s_cmp_eq_i32 s15, 0 // (op 556) 
s_mov_b32 s19, scc // (op 556) 
s_cmp_eq_i32 s19, 0 // (op 556) 
s_cselect_b32 s51, 0, -1 // (op 556) 

s_and_b32 s19, s48, 31 // (op 556) 
s_lshl_b32 s68, 1, s19 // (op 556) 
s_add_i32 s69, s68, s51 // (op 556) 


s_and_b32 s51, s0, s69 // (op 556) 


s_add_i32 s0, s18, s51 // (op 556) 


s_ashr_i32 s18, s0, s19 // (op 556) 


s_lshl_b32 s0, s48, 24 // (op 556) 
s_ashr_i32 s19, s0, 31 // (op 556) 

s_xor_b32 s0, s18, s19 // (op 556) 

s_sub_i32 s18, s0, s19 // (op 556) 


s_mul_i32 s0, s47, s46 // (op 556) 
s_add_u32 s68, s6, 16 // (op 556) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 556) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 556) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 556) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 556) 

s_sub_i32 s19, s68, s0 // (op 556) 

s_mul_i32 s51, s18, s19 // (op 556) 


s_sub_u32 s18, s8, s51 // (op 556) 


s_add_u32 s8, s18, s0 // (op 556) 


s_cmp_eq_i32 s9, 0 // (op 556) 
s_mov_b32 s0, scc // (op 556) 

s_cmp_eq_i32 s0, 0 // (op 556) 
s_cselect_b32 s9, s8, s1 // (op 556) 



v_lshl_add_u32 v57, s9, 4, v49 // (op 556) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 556) 
s_lshr_b32 s51, s0, 1 // (op 556) 

v_lshl_add_u32 v49, s51, 9, v53 // (op 556) 


v_mov_b32 v23, 0 // (op 556) convert
v_mov_b32 v22, v49 // (op 556) convert

v_mul_lo_u32 v27, v57, s5 // (op 556) most significant: low of low * high
v_mul_hi_u32 v49, v57, s4 // (op 556) most significant: high of low * low
v_mul_lo_u32 v26, v57, s4 // (op 556) least significant: low of low * low
v_add_u32 v27, v27, v49 // (op 556) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v22 // (op 556) least significant half
v_addc_co_u32 v27, s[0:1], v27, v23, s[0:1] // (op 556) most significant half



s_mul_i32 s1, s4, s7 // (op 558) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 558) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 558) least significant
s_mul_hi_u32 s8, s4, s6 // (op 558) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 558) most significant: sum
s_add_u32 s1, s1, s8 // (op 558) most significant: sum


s_mov_b32 s52, s20 // (op 558) reg expression
s_mov_b32 s53, s21 // (op 558) reg expression
s_mov_b32 s54, s0 // (op 558) reg expression
s_mov_b32 s55, 131072 // (op 558) call()

s_mul_hi_i32 s51, s16, s11 // (op 623) 
v_lshlrev_b32 v49, 2, v1 // (op 623) 
s_add_i32 s0, s51, s16 // (op 623) 

s_ashr_i32 s1, s0, 31 // (op 623) 
s_cmp_eq_i32 s11, 0 // (op 623) 
s_mov_b32 s51, scc // (op 623) 
s_cmp_eq_i32 s51, 0 // (op 623) 
s_cselect_b32 s8, 0, -1 // (op 623) 

s_and_b32 s9, s12, 31 // (op 623) 
s_lshl_b32 s51, 1, s9 // (op 623) 
s_add_i32 s18, s51, s8 // (op 623) 


s_and_b32 s8, s1, s18 // (op 623) 


s_add_i32 s1, s0, s8 // (op 623) 


s_ashr_i32 s0, s1, s9 // (op 623) 


s_lshl_b32 s1, s12, 24 // (op 623) 
s_ashr_i32 s51, s1, 31 // (op 623) 

s_xor_b32 s1, s0, s51 // (op 623) 

s_sub_i32 s0, s1, s51 // (op 623) 


s_mul_hi_i32 s1, s0, s13 // (op 623) 
s_add_i32 s51, s1, s0 // (op 623) 

s_ashr_i32 s1, s51, 31 // (op 623) 
s_cmp_eq_i32 s13, 0 // (op 623) 
s_mov_b32 s8, scc // (op 623) 
s_cmp_eq_i32 s8, 0 // (op 623) 
s_cselect_b32 s9, 0, -1 // (op 623) 

s_and_b32 s8, s14, 31 // (op 623) 
s_lshl_b32 s18, 1, s8 // (op 623) 
s_add_i32 s19, s18, s9 // (op 623) 


s_and_b32 s9, s1, s19 // (op 623) 


s_add_i32 s1, s51, s9 // (op 623) 


s_ashr_i32 s9, s1, s8 // (op 623) 


s_lshl_b32 s1, s14, 24 // (op 623) 
s_ashr_i32 s8, s1, 31 // (op 623) 

s_xor_b32 s1, s9, s8 // (op 623) 

s_sub_i32 s9, s1, s8 // (op 623) 


s_add_u32 s18, s2, 16 // (op 623) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 623) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 623) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 623) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 623) 

s_mul_i32 s1, s9, s18 // (op 623) 

s_sub_i32 s51, s0, s1 // (op 623) 


s_mul_hi_i32 s0, s16, s10 // (op 623) 
s_add_i32 s1, s0, s16 // (op 623) 

s_ashr_i32 s0, s1, 31 // (op 623) 
s_cmp_eq_i32 s10, 0 // (op 623) 
s_mov_b32 s8, scc // (op 623) 
s_cmp_eq_i32 s8, 0 // (op 623) 
s_cselect_b32 s9, 0, -1 // (op 623) 

s_and_b32 s8, s49, 31 // (op 623) 
s_lshl_b32 s68, 1, s8 // (op 623) 
s_add_i32 s69, s68, s9 // (op 623) 


s_and_b32 s9, s0, s69 // (op 623) 


s_add_i32 s0, s1, s9 // (op 623) 


s_ashr_i32 s1, s0, s8 // (op 623) 


s_lshl_b32 s0, s49, 24 // (op 623) 
s_ashr_i32 s8, s0, 31 // (op 623) 

s_xor_b32 s0, s1, s8 // (op 623) 

s_sub_i32 s1, s0, s8 // (op 623) 


s_mul_i32 s0, s46, s18 // (op 623) 

s_mul_i32 s8, s50, s0 // (op 623) 

s_mul_i32 s0, s1, s8 // (op 623) 

s_sub_u32 s8, s16, s0 // (op 623) 

s_mul_hi_i32 s0, s8, s15 // (op 623) 
s_add_i32 s9, s0, s8 // (op 623) 


s_ashr_i32 s0, s9, 31 // (op 623) 
s_cmp_eq_i32 s15, 0 // (op 623) 
s_mov_b32 s8, scc // (op 623) 
s_cmp_eq_i32 s8, 0 // (op 623) 
s_cselect_b32 s18, 0, -1 // (op 623) 

s_and_b32 s8, s48, 31 // (op 623) 
s_lshl_b32 s19, 1, s8 // (op 623) 
s_add_i32 s68, s19, s18 // (op 623) 


s_and_b32 s18, s0, s68 // (op 623) 


s_add_i32 s0, s9, s18 // (op 623) 


s_ashr_i32 s9, s0, s8 // (op 623) 


s_lshl_b32 s0, s48, 24 // (op 623) 
s_ashr_i32 s8, s0, 31 // (op 623) 

s_xor_b32 s0, s9, s8 // (op 623) 

s_sub_i32 s9, s0, s8 // (op 623) 


s_cmp_eq_i32 s1, 0 // (op 623) 
s_mov_b32 s0, scc // (op 623) 

s_cmp_eq_i32 s0, 0 // (op 623) 
s_cselect_b32 s1, s9, s51 // (op 623) 



s_lshl_b32 s0, s1, 4 // (op 623) 

s_mov_b32 s9, 0 // (op 623) convert
s_mov_b32 s8, s0 // (op 623) convert

s_ashr_i64 s[0:1], s[26:27], 8 // (op 623) 
s_lshr_b32 s51, s0, 1 // (op 623) 

v_lshl_add_u32 v53, s51, 9, v49 // (op 623) 


v_mul_lo_u32 v23, v53, s23 // (op 623) most significant: low of low * high
v_mul_hi_u32 v49, v53, s22 // (op 623) most significant: high of low * low
v_mul_lo_u32 v22, v53, s22 // (op 623) least significant: low of low * low
v_add_u32 v23, v23, v49 // (op 623) most significant: sum

v_mov_b32 v49, s8 // (op 623) 
v_mov_b32 v57, s9 // (op 623) 
v_add_co_u32 v22, s[0:1], v22, v49 // (op 623) least significant half
v_addc_co_u32 v23, s[0:1], v23, v57, s[0:1] // (op 623) most significant half





v_mul_lo_u32 v29, v23, 6 // (op 623) most significant: low of high * low
v_mul_hi_u32 v49, v22, 6 // (op 623) most significant: high of low * low
v_mul_lo_u32 v28, v22, 6 // (op 623) least significant: low of low * low
v_add_u32 v29, v29, v49 // (op 623) most significant: sum


v_lshrrev_b64 v[22:23], 3, v[28:29] // (op 623) 

v_mov_b32 v49, v22 // (op 623) convert

s_mul_i32 s1, s27, 6 // (op 625) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 625) least significant
s_mul_hi_u32 s51, s26, 6 // (op 625) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 625) most significant: sum

s_mul_i32 s9, s22, s1 // (op 625) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 625) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 625) least significant
s_mul_hi_u32 s18, s22, s0 // (op 625) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 625) most significant: sum
s_add_u32 s9, s9, s18 // (op 625) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 625) 

s_mov_b32 s56, s24 // (op 625) reg expression
s_mov_b32 s57, s25 // (op 625) reg expression
s_mov_b32 s58, s0 // (op 625) reg expression
s_mov_b32 s59, 131072 // (op 625) call()

s_mul_hi_i32 s51, s16, s11 // (op 687) 
v_lshrrev_b32 v53, 3, v1 // (op 687) 
s_add_i32 s0, s51, s16 // (op 687) 

v_lshlrev_b32 v57, 1, v53 // (op 687) 

s_ashr_i32 s1, s0, 31 // (op 687) 
v_and_b32 v53, -8, v1 // (op 687) 
s_cmp_eq_i32 s11, 0 // (op 687) 
s_mov_b32 s51, scc // (op 687) 
v_sub_u32 v61, v1, v53 // (op 687) 

s_cmp_eq_i32 s51, 0 // (op 687) 
s_cselect_b32 s8, 0, -1 // (op 687) 

s_and_b32 s9, s12, 31 // (op 687) 
s_lshl_b32 s51, 1, s9 // (op 687) 
s_add_i32 s18, s51, s8 // (op 687) 


s_and_b32 s8, s1, s18 // (op 687) 


s_add_i32 s1, s0, s8 // (op 687) 


s_ashr_i32 s0, s1, s9 // (op 687) 


s_lshl_b32 s1, s12, 24 // (op 687) 
s_ashr_i32 s51, s1, 31 // (op 687) 

s_xor_b32 s1, s0, s51 // (op 687) 

s_sub_i32 s0, s1, s51 // (op 687) 


s_mul_hi_i32 s1, s0, s13 // (op 687) 
s_add_i32 s51, s1, s0 // (op 687) 

s_ashr_i32 s1, s51, 31 // (op 687) 
s_cmp_eq_i32 s13, 0 // (op 687) 
s_mov_b32 s8, scc // (op 687) 
s_cmp_eq_i32 s8, 0 // (op 687) 
s_cselect_b32 s9, 0, -1 // (op 687) 

s_and_b32 s8, s14, 31 // (op 687) 
s_lshl_b32 s18, 1, s8 // (op 687) 
s_add_i32 s19, s18, s9 // (op 687) 


s_and_b32 s9, s1, s19 // (op 687) 


s_add_i32 s1, s51, s9 // (op 687) 


s_ashr_i32 s9, s1, s8 // (op 687) 


s_lshl_b32 s1, s14, 24 // (op 687) 
s_ashr_i32 s8, s1, 31 // (op 687) 

s_xor_b32 s1, s9, s8 // (op 687) 

s_sub_i32 s9, s1, s8 // (op 687) 


s_mul_i32 s1, s0, s46 // (op 687) 

s_sub_u32 s0, s16, s1 // (op 687) 

s_mul_i32 s1, s9, s46 // (op 687) 
s_add_u32 s1, s1, s0 // (op 687) 


s_mul_hi_i32 s0, s16, s10 // (op 687) 
s_add_i32 s51, s0, s16 // (op 687) 

s_ashr_i32 s0, s51, 31 // (op 687) 
s_cmp_eq_i32 s10, 0 // (op 687) 
s_mov_b32 s8, scc // (op 687) 
s_cmp_eq_i32 s8, 0 // (op 687) 
s_cselect_b32 s9, 0, -1 // (op 687) 

s_and_b32 s8, s49, 31 // (op 687) 
s_lshl_b32 s18, 1, s8 // (op 687) 
s_add_i32 s19, s18, s9 // (op 687) 


s_and_b32 s9, s0, s19 // (op 687) 


s_add_i32 s0, s51, s9 // (op 687) 


s_ashr_i32 s9, s0, s8 // (op 687) 


s_lshl_b32 s0, s49, 24 // (op 687) 
s_ashr_i32 s8, s0, 31 // (op 687) 

s_xor_b32 s0, s9, s8 // (op 687) 

s_sub_i32 s9, s0, s8 // (op 687) 


s_add_u32 s18, s2, 16 // (op 687) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 687) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 687) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 687) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 687) 

s_mul_i32 s0, s46, s18 // (op 687) 

s_mul_i32 s8, s50, s0 // (op 687) 

s_mul_i32 s0, s9, s8 // (op 687) 

s_sub_u32 s8, s16, s0 // (op 687) 

s_mov_b32 s0, s8 // (op 687) 
s_mul_hi_i32 s51, s0, s15 // (op 687) 
s_add_i32 s18, s51, s0 // (op 687) 


s_ashr_i32 s0, s18, 31 // (op 687) 
s_cmp_eq_i32 s15, 0 // (op 687) 
s_mov_b32 s19, scc // (op 687) 
s_cmp_eq_i32 s19, 0 // (op 687) 
s_cselect_b32 s51, 0, -1 // (op 687) 

s_and_b32 s19, s48, 31 // (op 687) 
s_lshl_b32 s68, 1, s19 // (op 687) 
s_add_i32 s69, s68, s51 // (op 687) 


s_and_b32 s51, s0, s69 // (op 687) 


s_add_i32 s0, s18, s51 // (op 687) 


s_ashr_i32 s18, s0, s19 // (op 687) 


s_lshl_b32 s0, s48, 24 // (op 687) 
s_ashr_i32 s19, s0, 31 // (op 687) 

s_xor_b32 s0, s18, s19 // (op 687) 

s_sub_i32 s18, s0, s19 // (op 687) 


s_mul_i32 s0, s47, s46 // (op 687) 
s_add_u32 s68, s6, 16 // (op 687) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 687) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 687) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 687) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 687) 

s_sub_i32 s19, s68, s0 // (op 687) 

s_mul_i32 s51, s18, s19 // (op 687) 


s_sub_u32 s18, s8, s51 // (op 687) 


s_add_u32 s8, s18, s0 // (op 687) 


s_cmp_eq_i32 s9, 0 // (op 687) 
s_mov_b32 s0, scc // (op 687) 

s_cmp_eq_i32 s0, 0 // (op 687) 
s_cselect_b32 s9, s8, s1 // (op 687) 



v_lshl_add_u32 v53, s9, 4, v57 // (op 687) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 687) 
s_lshr_b32 s51, s0, 1 // (op 687) 

v_lshl_add_u32 v57, s51, 4, v61 // (op 687) 


v_mul_lo_u32 v61, v53, s28 // (op 687) 
v_add_u32 v61, v61, v57 // (op 687) 


s_mul_i32 s1, s28, s7 // (op 689) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 689) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 689) least significant
s_mul_hi_u32 s8, s28, s6 // (op 689) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 689) most significant: sum
s_add_u32 s1, s1, s8 // (op 689) most significant: sum


s_mov_b32 s60, s30 // (op 689) reg expression
s_mov_b32 s61, s31 // (op 689) reg expression
s_mov_b32 s62, s0 // (op 689) reg expression
s_mov_b32 s63, 131072 // (op 689) call()

s_mul_hi_i32 s51, s16, s11 // (op 751) 
v_and_b32 v53, -16, v1 // (op 751) 
s_add_i32 s0, s51, s16 // (op 751) 

v_sub_u32 v57, v1, v53 // (op 751) 

s_ashr_i32 s1, s0, 31 // (op 751) 
v_lshrrev_b32 v53, 4, v1 // (op 751) 
s_cmp_eq_i32 s11, 0 // (op 751) 
s_mov_b32 s51, scc // (op 751) 
v_lshlrev_b32 v13, 1, v53 // (op 751) 

s_cmp_eq_i32 s51, 0 // (op 751) 
s_cselect_b32 s8, 0, -1 // (op 751) 

s_and_b32 s9, s12, 31 // (op 751) 
s_lshl_b32 s51, 1, s9 // (op 751) 
s_add_i32 s18, s51, s8 // (op 751) 


s_and_b32 s8, s1, s18 // (op 751) 


s_add_i32 s1, s0, s8 // (op 751) 


s_ashr_i32 s0, s1, s9 // (op 751) 


s_lshl_b32 s1, s12, 24 // (op 751) 
s_ashr_i32 s51, s1, 31 // (op 751) 

s_xor_b32 s1, s0, s51 // (op 751) 

s_sub_i32 s0, s1, s51 // (op 751) 


s_mul_hi_i32 s1, s0, s13 // (op 751) 
s_add_i32 s51, s1, s0 // (op 751) 

s_ashr_i32 s1, s51, 31 // (op 751) 
s_cmp_eq_i32 s13, 0 // (op 751) 
s_mov_b32 s8, scc // (op 751) 
s_cmp_eq_i32 s8, 0 // (op 751) 
s_cselect_b32 s9, 0, -1 // (op 751) 

s_and_b32 s8, s14, 31 // (op 751) 
s_lshl_b32 s18, 1, s8 // (op 751) 
s_add_i32 s19, s18, s9 // (op 751) 


s_and_b32 s9, s1, s19 // (op 751) 


s_add_i32 s1, s51, s9 // (op 751) 


s_ashr_i32 s9, s1, s8 // (op 751) 


s_lshl_b32 s1, s14, 24 // (op 751) 
s_ashr_i32 s8, s1, 31 // (op 751) 

s_xor_b32 s1, s9, s8 // (op 751) 

s_sub_i32 s9, s1, s8 // (op 751) 


s_add_u32 s18, s2, 16 // (op 751) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 751) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 751) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 751) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 751) 

s_mul_i32 s1, s9, s18 // (op 751) 

s_sub_i32 s51, s0, s1 // (op 751) 


s_mul_hi_i32 s0, s16, s10 // (op 751) 
s_add_i32 s1, s0, s16 // (op 751) 

s_ashr_i32 s0, s1, 31 // (op 751) 
s_cmp_eq_i32 s10, 0 // (op 751) 
s_mov_b32 s8, scc // (op 751) 
s_cmp_eq_i32 s8, 0 // (op 751) 
s_cselect_b32 s9, 0, -1 // (op 751) 

s_and_b32 s8, s49, 31 // (op 751) 
s_lshl_b32 s68, 1, s8 // (op 751) 
s_add_i32 s69, s68, s9 // (op 751) 


s_and_b32 s9, s0, s69 // (op 751) 


s_add_i32 s0, s1, s9 // (op 751) 


s_ashr_i32 s1, s0, s8 // (op 751) 


s_lshl_b32 s0, s49, 24 // (op 751) 
s_ashr_i32 s8, s0, 31 // (op 751) 

s_xor_b32 s0, s1, s8 // (op 751) 

s_sub_i32 s1, s0, s8 // (op 751) 


s_mul_i32 s0, s46, s18 // (op 751) 

s_mul_i32 s8, s50, s0 // (op 751) 

s_mul_i32 s0, s1, s8 // (op 751) 

s_sub_u32 s8, s16, s0 // (op 751) 

s_mul_hi_i32 s0, s8, s15 // (op 751) 
s_add_i32 s9, s0, s8 // (op 751) 


s_ashr_i32 s0, s9, 31 // (op 751) 
s_cmp_eq_i32 s15, 0 // (op 751) 
s_mov_b32 s8, scc // (op 751) 
s_cmp_eq_i32 s8, 0 // (op 751) 
s_cselect_b32 s18, 0, -1 // (op 751) 

s_and_b32 s8, s48, 31 // (op 751) 
s_lshl_b32 s19, 1, s8 // (op 751) 
s_add_i32 s68, s19, s18 // (op 751) 


s_and_b32 s18, s0, s68 // (op 751) 


s_add_i32 s0, s9, s18 // (op 751) 


s_ashr_i32 s9, s0, s8 // (op 751) 


s_lshl_b32 s0, s48, 24 // (op 751) 
s_ashr_i32 s8, s0, 31 // (op 751) 

s_xor_b32 s0, s9, s8 // (op 751) 

s_sub_i32 s9, s0, s8 // (op 751) 


s_cmp_eq_i32 s1, 0 // (op 751) 
s_mov_b32 s0, scc // (op 751) 

s_cmp_eq_i32 s0, 0 // (op 751) 
s_cselect_b32 s1, s9, s51 // (op 751) 



v_lshl_add_u32 v53, s1, 4, v57 // (op 751) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 751) 
s_lshr_b32 s51, s0, 1 // (op 751) 

v_lshl_add_u32 v57, s51, 4, v13 // (op 751) 


v_mul_lo_u32 v13, v57, s32 // (op 751) 
v_add_u32 v13, v13, v53 // (op 751) 


s_mul_i32 s1, s32, s37 // (op 753) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 753) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 753) least significant
s_mul_hi_u32 s8, s32, s36 // (op 753) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 753) most significant: sum
s_add_u32 s1, s1, s8 // (op 753) most significant: sum


s_mov_b32 s64, s34 // (op 753) reg expression
s_mov_b32 s65, s35 // (op 753) reg expression
s_mov_b32 s66, s0 // (op 753) reg expression
s_mov_b32 s67, 131072 // (op 753) call()

v_and_b32 v53, -16, v1 // (op 844) 
v_sub_u32 v57, v1, v53 // (op 844) 

v_lshlrev_b32 v53, 4, v57 // (op 844) 

v_lshrrev_b32 v57, 4, v1 // (op 844) 
v_lshl_add_u32 v14, v57, 8, v53 // (op 844) 


s_nop 0
v_readfirstlane_b32 s51, v14 // (op 844) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v53, 6, v1 // (op 896) 
v_mul_lo_u32 v57, v53, 6 // (op 896) 
v_lshrrev_b32 v14, 3, v57 // (op 896) 

v_lshrrev_b32 v57, 4, v53 // (op 896) 

v_lshlrev_b32 v53, 2, v57 // (op 896) 

v_add_u32 v57, v14, v53 // (op 896) 


v_and_b32 v53, -8, v1 // (op 949) 
v_sub_u32 v14, v1, v53 // (op 949) 

v_lshrrev_b32 v53, 3, v1 // (op 949) 
v_lshl_add_u32 v15, v53, 4, v14 // (op 949) 


v_and_b32 v14, -16, v1 // (op 1002) 
v_sub_u32 v53, v1, v14 // (op 1002) 

v_lshrrev_b32 v14, 4, v1 // (op 1002) 
v_lshl_add_u32 v16, v14, 5, v53 // (op 1002) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 194) 
s_and_b32 s8, s0, 4294967294 // (op 194) 

s_ashr_i64 s[0:1], s[26:27], 8 // (op 193) 
s_mov_b32 s9, s0 // (op 193) convert

s_cmp_lt_u32 s8, s9 // (op 193) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193 // (op 193) Condition: Top (jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193:
s_barrier  // (op 1218) 
s_add_u32 s8, s8, 1 // (op 195) 

s_lshl_b64 s[0:1], s[4:5], 2 // (op 527) 
v_mov_b64 v[24:25], s[0:1] // (op 527) convert


v_mov_b64 v[22:23], v[26:27] // (op 527) 
v_mov_b64 v[20:21], v[22:23] // (op 527) 
s_mov_b32 m0, s51 // (op 527) reg expression
buffer_load_dwordx4 v20, s[52:55], 0 offen offset: 0 lds // (op 527) Load value direct to lds
v_add_co_u32 v20, s[0:1], v20, v24 // (op 527) least significant half
v_addc_co_u32 v21, s[0:1], 0, v21, s[0:1] // (op 527) most significant half

s_add_u32 m0, m0, 1024 // (op 527) 
buffer_load_dwordx4 v20, s[52:55], 0 offen offset: 0 lds // (op 527) Load value direct to lds
v_add_co_u32 v20, s[0:1], v20, v24 // (op 527) least significant half
v_addc_co_u32 v21, s[0:1], 0, v21, s[0:1] // (op 527) most significant half

s_add_u32 m0, m0, 1024 // (op 527) 
buffer_load_dwordx4 v20, s[52:55], 0 offen offset: 0 lds // (op 527) Load value direct to lds
v_add_co_u32 v20, s[0:1], v20, v24 // (op 527) least significant half
v_addc_co_u32 v21, s[0:1], 0, v21, s[0:1] // (op 527) most significant half

s_add_u32 m0, m0, 1024 // (op 527) 
buffer_load_dwordx4 v20, s[52:55], 0 offen offset: 0 lds // (op 527) Load value direct to lds



v_add_co_u32 v26, vcc, 256, v26 // (op 546) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 546) most significant half

s_mul_i32 s1, s23, 6 // (op 221) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 221) least significant
s_mul_hi_u32 s9, s22, 6 // (op 221) most significant: high of low * low
s_add_u32 s1, s1, s9 // (op 221) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 221) 


v_mov_b32 v12, v49 // (op 221) 
v_mov_b32 v14, v12 // (op 221) reg expression
buffer_load_dwordx3 v[18:20], v14, s[56:59], 0 offen // (op 221) Load value
v_add_u32 v14, v14, v24 // (op 221) 
buffer_load_dwordx3 v[28:30], v14, s[56:59], 0 offen // (op 221) Load value
v_add_u32 v14, v14, v24 // (op 221) 
buffer_load_dwordx3 v[32:34], v14, s[56:59], 0 offen // (op 221) Load value
v_add_u32 v14, v14, v24 // (op 221) 
buffer_load_dwordx3 v[36:38], v14, s[56:59], 0 offen // (op 221) Load value



s_lshl_b64 s[0:1], s[22:23], 8 // (op 615) 
s_mul_i32 s19, s1, 6 // (op 615) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 615) least significant
s_mul_hi_u32 s9, s0, 6 // (op 615) most significant: high of low * low
s_add_u32 s19, s19, s9 // (op 615) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 615) 

v_add_u32 v49, v49, s0 // (op 615) 


s_barrier  // (op 1237) 
v_mov_b64 v[24:25], s[28:29] // (op 222) convert




v_mov_b32 v12, v61 // (op 222) 
v_mov_b32 v14, v12 // (op 222) 
buffer_load_ubyte v22, v14, s[60:63], 0 offen offset: 0 // (op 222) Load value
v_add_u32 v14, v14, v24 // (op 222) 
s_waitcnt vmcnt(4)
ds_write_b96 v57, v[18:20] offset:8192 // (op 218) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v57, v[28:30] offset:8208 // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v57, v[32:34] offset:8224 // (op 218) Store local data 
buffer_load_ubyte v23, v14, s[60:63], 0 offen offset: 0 // (op 222) Load value



v_add_u32 v61, 8, v61 // (op 679) 
s_waitcnt vmcnt(2)
ds_write_b96 v57, v[36:38] offset:8240 // (op 218) Store local data 


v_mov_b64 v[24:25], s[32:33] // (op 223) convert

s_barrier  // (op 1258) 
v_mov_b32 v12, v13 // (op 223) 



v_mov_b32 v14, v12 // (op 223) reg expression
buffer_load_ubyte v20, v14, s[64:67], 0 offen offset: 0 // (op 223) Load value
v_add_u32 v14, v14, v24 // (op 223) 
buffer_load_ubyte v21, v14, s[64:67], 0 offen offset: 0 // (op 223) Load value



s_lshl_b32 s9, s32, 3 // (op 743) 
v_add_u32 v13, v13, s9 // (op 743) 

s_waitcnt vmcnt(3)
ds_write_b8 v15, v22 offset:12416 // (op 219) Store local data 
s_barrier  // (op 1227) 
s_waitcnt vmcnt(2)
ds_write_b8 v15, v23 offset:12424 // (op 219) Store local data 




ds_read_b128 v[28:31], v3 // (op 201) Load local data 
s_barrier  // (op 1278) 
ds_read_b128 v[32:35], v3 offset:64 // (op 201) Load local data 



s_waitcnt lgkmcnt(4)
s_barrier  // (op 1248) 
s_waitcnt vmcnt(1)
ds_write_b8 v16, v20 offset:12288 // (op 220) Store local data 



ds_read_b96_tr_b6 v[36:38], v5 offset:8192 // (op 203) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v5 offset:8448 // (op 203) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v16, v21 offset:12304 // (op 220) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 1268) 



ds_read_u8 v12, v0 offset:12416 // (op 205) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1288) 


ds_read_u8 v14, v7 offset:12288 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:41], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 208) 







ds_read_b128 v[18:21], v3 offset:128 // (op 210) Load local data 
ds_read_b128 v[22:25], v3 offset:192 // (op 210) Load local data 



ds_read_b96_tr_b6 v[28:30], v5 offset:10240 // (op 212) Transpose load local data 
ds_read_b96_tr_b6 v[31:33], v5 offset:10496 // (op 212) Transpose load local data 



ds_read_u8 v12, v0 offset:12420 // (op 214) Load local data 


ds_read_u8 v14, v7 offset:12352 // (op 216) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[28:33], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 217) 




s_ashr_i64 s[0:1], s[26:27], 8 // (op 193) 
s_mov_b32 s9, s0 // (op 193) convert

s_cmp_lt_u32 s8, s9 // (op 193) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193 // (op 193) Condition: Bottom (jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193 if true)
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193:























s_barrier  // (op 1361) 








v_cmp_eq_f32 vcc, 0.00000, v8 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v10, v4 // (op 50) 
v_mul_f32 v13, v10, v5 // (op 50) 
v_mul_f32 v14, v10, v6 // (op 50) 
v_mul_f32 v15, v10, v7 // (op 50) 

s_mul_hi_i32 s0, s16, s11 // (op 1044) 
v_lshrrev_b32 v0, 6, v1 // (op 1044) 
s_add_i32 s1, s0, s16 // (op 1044) 

v_and_b32 v9, 63, v1 // (op 1044) 
s_ashr_i32 s0, s1, 31 // (op 1044) 
v_lshrrev_b32 v11, 4, v9 // (op 1044) 
s_cmp_eq_i32 s11, 0 // (op 1044) 
s_mov_b32 s4, scc // (op 1044) 
v_lshlrev_b32 v3, 2, v11 // (op 1044) 

s_cmp_eq_i32 s4, 0 // (op 1044) 
s_cselect_b32 s5, 0, -1 // (op 1044) 

v_lshl_add_u32 v11, v0, 4, v3 // (op 1044) 


s_and_b32 s4, s12, 31 // (op 1044) 
v_lshrrev_b32 v0, 2, v9 // (op 1044) 

s_lshl_b32 s8, 1, s4 // (op 1044) 
v_and_b32 v9, 3, v0 // (op 1044) 

s_add_i32 s9, s8, s5 // (op 1044) 


v_and_b32 v0, 3, v1 // (op 1044) 
s_and_b32 s5, s0, s9 // (op 1044) 


v_lshl_add_u32 v3, v9, 2, v0 // (op 1044) 


s_add_i32 s0, s1, s5 // (op 1044) 


s_ashr_i32 s1, s0, s4 // (op 1044) 


s_lshl_b32 s0, s12, 24 // (op 1044) 
s_ashr_i32 s4, s0, 31 // (op 1044) 

s_xor_b32 s0, s1, s4 // (op 1044) 

s_sub_i32 s1, s0, s4 // (op 1044) 


s_mul_hi_i32 s0, s1, s13 // (op 1044) 
s_add_i32 s4, s0, s1 // (op 1044) 

s_ashr_i32 s0, s4, 31 // (op 1044) 
s_cmp_eq_i32 s13, 0 // (op 1044) 
s_mov_b32 s5, scc // (op 1044) 
s_cmp_eq_i32 s5, 0 // (op 1044) 
s_cselect_b32 s8, 0, -1 // (op 1044) 

s_and_b32 s5, s14, 31 // (op 1044) 
s_lshl_b32 s9, 1, s5 // (op 1044) 
s_add_i32 s18, s9, s8 // (op 1044) 


s_and_b32 s8, s0, s18 // (op 1044) 


s_add_i32 s0, s4, s8 // (op 1044) 


s_ashr_i32 s4, s0, s5 // (op 1044) 


s_lshl_b32 s0, s14, 24 // (op 1044) 
s_ashr_i32 s5, s0, 31 // (op 1044) 

s_xor_b32 s0, s4, s5 // (op 1044) 

s_sub_i32 s4, s0, s5 // (op 1044) 


s_mul_i32 s0, s1, s46 // (op 1044) 
s_sub_u32 s5, s16, s0 // (op 1044) 

s_mul_i32 s0, s4, s46 // (op 1044) 
s_add_u32 s0, s0, s5 // (op 1044) 

s_mul_hi_i32 s5, s16, s10 // (op 1044) 
s_add_i32 s8, s5, s16 // (op 1044) 

s_ashr_i32 s5, s8, 31 // (op 1044) 
s_cmp_eq_i32 s10, 0 // (op 1044) 
s_mov_b32 s9, scc // (op 1044) 
s_cmp_eq_i32 s9, 0 // (op 1044) 
s_cselect_b32 s18, 0, -1 // (op 1044) 

s_and_b32 s9, s49, 31 // (op 1044) 
s_lshl_b32 s19, 1, s9 // (op 1044) 
s_add_i32 s20, s19, s18 // (op 1044) 


s_and_b32 s18, s5, s20 // (op 1044) 


s_add_i32 s5, s8, s18 // (op 1044) 


s_ashr_i32 s8, s5, s9 // (op 1044) 


s_lshl_b32 s5, s49, 24 // (op 1044) 
s_ashr_i32 s9, s5, 31 // (op 1044) 

s_xor_b32 s5, s8, s9 // (op 1044) 

s_sub_i32 s8, s5, s9 // (op 1044) 


s_add_u32 s18, s2, 16 // (op 1044) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 1044) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 1044) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 1044) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 1044) 

s_mul_i32 s5, s46, s18 // (op 1044) 
s_mul_i32 s9, s50, s5 // (op 1044) 

s_mul_i32 s5, s8, s9 // (op 1044) 

s_sub_u32 s9, s16, s5 // (op 1044) 

s_mov_b32 s5, s9 // (op 1044) 
s_mul_hi_i32 s20, s5, s15 // (op 1044) 
s_add_i32 s21, s20, s5 // (op 1044) 


s_ashr_i32 s5, s21, 31 // (op 1044) 
s_cmp_eq_i32 s15, 0 // (op 1044) 
s_mov_b32 s20, scc // (op 1044) 
s_cmp_eq_i32 s20, 0 // (op 1044) 
s_cselect_b32 s22, 0, -1 // (op 1044) 

s_and_b32 s20, s48, 31 // (op 1044) 
s_lshl_b32 s23, 1, s20 // (op 1044) 
s_add_i32 s24, s23, s22 // (op 1044) 


s_and_b32 s22, s5, s24 // (op 1044) 


s_add_i32 s5, s21, s22 // (op 1044) 


s_ashr_i32 s21, s5, s20 // (op 1044) 


s_lshl_b32 s5, s48, 24 // (op 1044) 
s_ashr_i32 s20, s5, 31 // (op 1044) 

s_xor_b32 s5, s21, s20 // (op 1044) 

s_sub_i32 s21, s5, s20 // (op 1044) 


s_mul_i32 s5, s47, s46 // (op 1044) 
s_add_u32 s22, s6, 16 // (op 1044) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1044) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1044) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1044) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1044) 

s_sub_i32 s20, s22, s5 // (op 1044) 

s_mul_i32 s22, s21, s20 // (op 1044) 

s_sub_u32 s20, s9, s22 // (op 1044) 


s_add_u32 s9, s20, s5 // (op 1044) 


s_cmp_eq_i32 s8, 0 // (op 1044) 
s_mov_b32 s5, scc // (op 1044) 

s_cmp_eq_i32 s5, 0 // (op 1044) 
s_cselect_b32 s8, s9, s0 // (op 1044) 


s_mul_i32 s0, s4, s18 // (op 1044) 


v_lshl_add_u32 v0, s8, 4, v11 // (op 1044) 


s_sub_i32 s4, s1, s0 // (op 1044) 


s_cmp_eq_i32 s5, 0 // (op 1044) 
s_cselect_b32 s0, s21, s4 // (op 1044) 



v_lshl_add_u32 v9, s0, 4, v3 // (op 1044) 


v_mul_lo_u32 v11, v9, s38 // (op 1044) 

v_add_lshl_u32 v9, v0, v11, 2 // (op 1044) 


s_mul_i32 s1, s38, s3 // (op 1046) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 1046) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 1046) least significant
s_mul_hi_u32 s5, s38, s2 // (op 1046) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 1046) most significant: sum
s_add_u32 s1, s1, s5 // (op 1046) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 1046) 

s_mov_b32 s20, s40 // (op 1046) reg expression
s_mov_b32 s21, s41 // (op 1046) reg expression
s_mov_b32 s22, s4 // (op 1046) reg expression
s_mov_b32 s23, 131072 // (op 1046) call()




buffer_store_dwordx4 v[12:15], v9, s[20:23], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155:
s_mul_hi_i32 s0, s16, s11 // (op 814) 
v_lshrrev_b32 v0, 6, v1 // (op 814) 
s_add_i32 s1, s0, s16 // (op 814) 

v_and_b32 v9, 63, v1 // (op 814) 
s_ashr_i32 s0, s1, 31 // (op 814) 
v_lshrrev_b32 v3, 4, v9 // (op 814) 
s_cmp_eq_i32 s11, 0 // (op 814) 
s_mov_b32 s4, scc // (op 814) 
v_lshlrev_b32 v4, 2, v3 // (op 814) 

s_cmp_eq_i32 s4, 0 // (op 814) 
s_cselect_b32 s5, 0, -1 // (op 814) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 814) 


s_and_b32 s4, s12, 31 // (op 814) 
v_lshrrev_b32 v0, 2, v9 // (op 814) 

s_lshl_b32 s8, 1, s4 // (op 814) 
v_and_b32 v9, 3, v0 // (op 814) 

s_add_i32 s9, s8, s5 // (op 814) 


v_and_b32 v0, 3, v1 // (op 814) 
s_and_b32 s5, s0, s9 // (op 814) 


v_lshl_add_u32 v4, v9, 2, v0 // (op 814) 


s_add_i32 s0, s1, s5 // (op 814) 


s_ashr_i32 s1, s0, s4 // (op 814) 


s_lshl_b32 s0, s12, 24 // (op 814) 
s_ashr_i32 s4, s0, 31 // (op 814) 

s_xor_b32 s0, s1, s4 // (op 814) 

s_sub_i32 s1, s0, s4 // (op 814) 


s_mul_hi_i32 s0, s1, s13 // (op 814) 
s_add_i32 s4, s0, s1 // (op 814) 

s_ashr_i32 s0, s4, 31 // (op 814) 
s_cmp_eq_i32 s13, 0 // (op 814) 
s_mov_b32 s5, scc // (op 814) 
s_cmp_eq_i32 s5, 0 // (op 814) 
s_cselect_b32 s8, 0, -1 // (op 814) 

s_and_b32 s5, s14, 31 // (op 814) 
s_lshl_b32 s9, 1, s5 // (op 814) 
s_add_i32 s18, s9, s8 // (op 814) 


s_and_b32 s19, s0, s18 // (op 814) 


s_add_i32 s0, s4, s19 // (op 814) 


s_ashr_i32 s4, s0, s5 // (op 814) 


s_lshl_b32 s0, s14, 24 // (op 814) 
s_ashr_i32 s5, s0, 31 // (op 814) 

s_xor_b32 s0, s4, s5 // (op 814) 

s_sub_i32 s4, s0, s5 // (op 814) 


s_mul_i32 s0, s1, s46 // (op 814) 
s_sub_u32 s5, s16, s0 // (op 814) 

s_mul_i32 s0, s4, s46 // (op 814) 
s_add_u32 s0, s0, s5 // (op 814) 

s_mul_hi_i32 s5, s16, s10 // (op 814) 
s_add_i32 s8, s5, s16 // (op 814) 

s_ashr_i32 s5, s8, 31 // (op 814) 
s_cmp_eq_i32 s10, 0 // (op 814) 
s_mov_b32 s9, scc // (op 814) 
s_cmp_eq_i32 s9, 0 // (op 814) 
s_cselect_b32 s18, 0, -1 // (op 814) 

s_and_b32 s9, s49, 31 // (op 814) 
s_lshl_b32 s19, 1, s9 // (op 814) 
s_add_i32 s24, s19, s18 // (op 814) 


s_and_b32 s18, s5, s24 // (op 814) 


s_add_i32 s5, s8, s18 // (op 814) 


s_ashr_i32 s8, s5, s9 // (op 814) 


s_lshl_b32 s5, s49, 24 // (op 814) 
s_ashr_i32 s9, s5, 31 // (op 814) 

s_xor_b32 s5, s8, s9 // (op 814) 

s_sub_i32 s8, s5, s9 // (op 814) 


s_add_u32 s18, s2, 16 // (op 814) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 814) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 814) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 814) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 4 // (op 814) 

s_mul_i32 s5, s46, s18 // (op 814) 
s_mul_i32 s9, s50, s5 // (op 814) 

s_mul_i32 s5, s8, s9 // (op 814) 

s_sub_u32 s9, s16, s5 // (op 814) 

s_mov_b32 s5, s9 // (op 814) 
s_mul_hi_i32 s24, s5, s15 // (op 814) 
s_add_i32 s25, s24, s5 // (op 814) 


s_ashr_i32 s5, s25, 31 // (op 814) 
s_cmp_eq_i32 s15, 0 // (op 814) 
s_mov_b32 s24, scc // (op 814) 
s_cmp_eq_i32 s24, 0 // (op 814) 
s_cselect_b32 s26, 0, -1 // (op 814) 

s_and_b32 s24, s48, 31 // (op 814) 
s_lshl_b32 s27, 1, s24 // (op 814) 
s_add_i32 s28, s27, s26 // (op 814) 


s_and_b32 s26, s5, s28 // (op 814) 


s_add_i32 s5, s25, s26 // (op 814) 


s_ashr_i32 s25, s5, s24 // (op 814) 


s_lshl_b32 s5, s48, 24 // (op 814) 
s_ashr_i32 s24, s5, 31 // (op 814) 

s_xor_b32 s5, s25, s24 // (op 814) 

s_sub_i32 s25, s5, s24 // (op 814) 


s_mul_i32 s5, s47, s46 // (op 814) 
s_add_u32 s26, s6, 16 // (op 814) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 814) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 814) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 814) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 814) 

s_sub_i32 s24, s26, s5 // (op 814) 

s_mul_i32 s26, s25, s24 // (op 814) 

s_sub_u32 s24, s9, s26 // (op 814) 


s_add_u32 s9, s24, s5 // (op 814) 


s_cmp_eq_i32 s8, 0 // (op 814) 
s_mov_b32 s5, scc // (op 814) 

s_cmp_eq_i32 s5, 0 // (op 814) 
s_cselect_b32 s8, s9, s0 // (op 814) 


s_mul_i32 s0, s4, s18 // (op 814) 


v_lshl_add_u32 v0, s8, 4, v3 // (op 814) 


s_sub_i32 s4, s1, s0 // (op 814) 


s_cmp_eq_i32 s5, 0 // (op 814) 
s_cselect_b32 s24, s25, s4 // (op 814) 



v_lshl_add_u32 v3, s24, 4, v4 // (op 814) 


v_mul_lo_u32 v9, v3, s42 // (op 814) 

v_add_lshl_u32 v3, v0, v9, 2 // (op 814) 


s_mul_i32 s1, s42, s3 // (op 816) most significant: low of low * high
s_mul_i32 s4, s43, s2 // (op 816) most significant: low of high * low
s_mul_i32 s0, s42, s2 // (op 816) least significant
s_mul_hi_u32 s5, s42, s2 // (op 816) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 816) most significant: sum
s_add_u32 s1, s1, s5 // (op 816) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 816) 

s_mov_b32 s24, s44 // (op 816) reg expression
s_mov_b32 s25, s45 // (op 816) reg expression
s_mov_b32 s26, s4 // (op 816) reg expression
s_mov_b32 s27, 131072 // (op 816) call()





buffer_load_dwordx4 v[4:7], v3, s[24:27], 0 offen // (op 167) Load value



v_accvgpr_read v12, a0 // (op 164) 
v_accvgpr_read v13, a1 // (op 164) 
v_accvgpr_read v14, a2 // (op 164) 
v_accvgpr_read v15, a3 // (op 164) 
v_mul_f32 v16, v10, v12 // (op 164) 
v_mul_f32 v17, v10, v13 // (op 164) 
v_mul_f32 v18, v10, v14 // (op 164) 
v_mul_f32 v19, v10, v15 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v8, v4 // (op 168) 
v_mul_f32 v13, v8, v5 // (op 168) 
v_mul_f32 v14, v8, v6 // (op 168) 
v_mul_f32 v15, v8, v7 // (op 168) 

v_add_f32 v4, v12, v16 // (op 165) 
v_add_f32 v5, v13, v17 // (op 165) 
v_add_f32 v6, v14, v18 // (op 165) 
v_add_f32 v7, v15, v19 // (op 165) 


s_mul_hi_i32 s0, s16, s11 // (op 1061) 
v_lshrrev_b32 v0, 6, v1 // (op 1061) 
s_add_i32 s1, s0, s16 // (op 1061) 

v_and_b32 v3, 63, v1 // (op 1061) 
s_ashr_i32 s0, s1, 31 // (op 1061) 
v_lshrrev_b32 v9, 4, v3 // (op 1061) 
s_cmp_eq_i32 s11, 0 // (op 1061) 
s_mov_b32 s4, scc // (op 1061) 
v_lshlrev_b32 v11, 2, v9 // (op 1061) 

s_cmp_eq_i32 s4, 0 // (op 1061) 
s_cselect_b32 s5, 0, -1 // (op 1061) 

v_lshl_add_u32 v9, v0, 4, v11 // (op 1061) 


s_and_b32 s4, s12, 31 // (op 1061) 
v_lshrrev_b32 v0, 2, v3 // (op 1061) 

s_lshl_b32 s8, 1, s4 // (op 1061) 
v_and_b32 v3, 3, v0 // (op 1061) 

s_add_i32 s9, s8, s5 // (op 1061) 


v_and_b32 v0, 3, v1 // (op 1061) 
s_and_b32 s5, s0, s9 // (op 1061) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 1061) 


s_add_i32 s0, s1, s5 // (op 1061) 


s_ashr_i32 s1, s0, s4 // (op 1061) 


s_lshl_b32 s0, s12, 24 // (op 1061) 
s_ashr_i32 s4, s0, 31 // (op 1061) 

s_xor_b32 s0, s1, s4 // (op 1061) 

s_sub_i32 s1, s0, s4 // (op 1061) 


s_mul_hi_i32 s0, s1, s13 // (op 1061) 
s_add_i32 s4, s0, s1 // (op 1061) 

s_ashr_i32 s0, s4, 31 // (op 1061) 
s_cmp_eq_i32 s13, 0 // (op 1061) 
s_mov_b32 s5, scc // (op 1061) 
s_cmp_eq_i32 s5, 0 // (op 1061) 
s_cselect_b32 s8, 0, -1 // (op 1061) 

s_and_b32 s5, s14, 31 // (op 1061) 
s_lshl_b32 s9, 1, s5 // (op 1061) 
s_add_i32 s18, s9, s8 // (op 1061) 


s_and_b32 s19, s0, s18 // (op 1061) 


s_add_i32 s0, s4, s19 // (op 1061) 


s_ashr_i32 s4, s0, s5 // (op 1061) 


s_lshl_b32 s0, s14, 24 // (op 1061) 
s_ashr_i32 s5, s0, 31 // (op 1061) 

s_xor_b32 s0, s4, s5 // (op 1061) 

s_sub_i32 s4, s0, s5 // (op 1061) 


s_mul_i32 s0, s1, s46 // (op 1061) 
s_sub_u32 s5, s16, s0 // (op 1061) 

s_mul_i32 s0, s4, s46 // (op 1061) 
s_add_u32 s0, s0, s5 // (op 1061) 

s_mul_hi_i32 s5, s16, s10 // (op 1061) 
s_add_i32 s8, s5, s16 // (op 1061) 

s_ashr_i32 s5, s8, 31 // (op 1061) 
s_cmp_eq_i32 s10, 0 // (op 1061) 
s_mov_b32 s9, scc // (op 1061) 
s_cmp_eq_i32 s9, 0 // (op 1061) 
s_cselect_b32 s18, 0, -1 // (op 1061) 

s_and_b32 s9, s49, 31 // (op 1061) 
s_lshl_b32 s19, 1, s9 // (op 1061) 
s_add_i32 s24, s19, s18 // (op 1061) 


s_and_b32 s18, s5, s24 // (op 1061) 


s_add_i32 s5, s8, s18 // (op 1061) 


s_ashr_i32 s8, s5, s9 // (op 1061) 


s_lshl_b32 s5, s49, 24 // (op 1061) 
s_ashr_i32 s9, s5, 31 // (op 1061) 

s_xor_b32 s5, s8, s9 // (op 1061) 

s_sub_i32 s8, s5, s9 // (op 1061) 


s_add_u32 s18, s2, 16 // (op 1061) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 1061) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 1061) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 1061) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 4 // (op 1061) 

s_mul_i32 s5, s46, s18 // (op 1061) 
s_mul_i32 s9, s50, s5 // (op 1061) 

s_mul_i32 s5, s8, s9 // (op 1061) 

s_sub_u32 s9, s16, s5 // (op 1061) 

s_mov_b32 s5, s9 // (op 1061) 
s_mul_hi_i32 s24, s5, s15 // (op 1061) 
s_add_i32 s25, s24, s5 // (op 1061) 


s_ashr_i32 s5, s25, 31 // (op 1061) 
s_cmp_eq_i32 s15, 0 // (op 1061) 
s_mov_b32 s24, scc // (op 1061) 
s_cmp_eq_i32 s24, 0 // (op 1061) 
s_cselect_b32 s26, 0, -1 // (op 1061) 

s_and_b32 s24, s48, 31 // (op 1061) 
s_lshl_b32 s27, 1, s24 // (op 1061) 
s_add_i32 s28, s27, s26 // (op 1061) 


s_and_b32 s26, s5, s28 // (op 1061) 


s_add_i32 s5, s25, s26 // (op 1061) 


s_ashr_i32 s25, s5, s24 // (op 1061) 


s_lshl_b32 s5, s48, 24 // (op 1061) 
s_ashr_i32 s24, s5, 31 // (op 1061) 

s_xor_b32 s5, s25, s24 // (op 1061) 

s_sub_i32 s25, s5, s24 // (op 1061) 


s_mul_i32 s5, s47, s46 // (op 1061) 
s_add_u32 s26, s6, 16 // (op 1061) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 1061) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 1061) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 1061) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 1061) 

s_sub_i32 s24, s26, s5 // (op 1061) 

s_mul_i32 s26, s25, s24 // (op 1061) 

s_sub_u32 s24, s9, s26 // (op 1061) 


s_add_u32 s9, s24, s5 // (op 1061) 


s_cmp_eq_i32 s8, 0 // (op 1061) 
s_mov_b32 s5, scc // (op 1061) 

s_cmp_eq_i32 s5, 0 // (op 1061) 
s_cselect_b32 s8, s9, s0 // (op 1061) 


s_mul_i32 s0, s4, s18 // (op 1061) 


v_lshl_add_u32 v0, s8, 4, v9 // (op 1061) 


s_sub_i32 s4, s1, s0 // (op 1061) 


s_cmp_eq_i32 s5, 0 // (op 1061) 
s_cselect_b32 s24, s25, s4 // (op 1061) 



v_lshl_add_u32 v3, s24, 4, v11 // (op 1061) 


v_mul_lo_u32 v9, v3, s38 // (op 1061) 

v_add_lshl_u32 v3, v0, v9, 2 // (op 1061) 


s_mul_i32 s1, s38, s3 // (op 1063) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 1063) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 1063) least significant
s_mul_hi_u32 s5, s38, s2 // (op 1063) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 1063) most significant: sum
s_add_u32 s1, s1, s5 // (op 1063) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 1063) 

s_mov_b32 s20, s40 // (op 1063) reg expression
s_mov_b32 s21, s41 // (op 1063) reg expression
s_mov_b32 s22, s4 // (op 1063) reg expression
s_mov_b32 s23, 131072 // (op 1063) call()




buffer_store_dwordx4 v[4:7], v3, s[20:23], 0 offen // (op 166) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155:
















s_endpgm  // End of RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.LRR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end:
.size RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_, .LRR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end-RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16896 // lds bytes
  .amdhsa_accum_offset 96
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
    DataTypeB: F6
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
    args: [ { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: StrideA0 },
            { type: int64, semantic: SizeFree0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: address, semantic: AddressA },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesByBlock } ]
    macrotile: [16, 16, 256]
    threads: [64, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 95
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .args:
      - .name: Tensor_2_size_1_12
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 24
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 28
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
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 72
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_4
        .size: 8
        .offset: 80
        .value_kind: by_value
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 112
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_9
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_11
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 152
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: WGM_18
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
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 172
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 176
        .value_kind: by_value
...

.end_amdgpu_metadata

