// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__exec_begin:


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
s_load_dword s48, s[0:1], 168 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


v_lshrrev_b32 v0, 6, v1 // (op 827) 
v_and_b32 v3, 15, v1 // (op 827) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 827) 


v_and_b32 v0, 48, v1 // (op 827) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 827) 


v_and_b32 v0, 15, v1 // (op 879) 
v_and_b32 v4, 48, v1 // (op 879) 
v_lshl_add_u32 v5, v0, 7, v4 // (op 879) 


v_lshrrev_b32 v0, 6, v1 // (op 932) 
v_and_b32 v4, 15, v1 // (op 932) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 932) 


v_and_b32 v0, 63, v1 // (op 932) 
v_lshrrev_b32 v4, 4, v0 // (op 932) 

v_lshl_add_u32 v0, v6, 2, v4 // (op 932) 


v_and_b32 v4, 15, v1 // (op 985) 
v_and_b32 v6, 63, v1 // (op 985) 
v_lshrrev_b32 v7, 4, v6 // (op 985) 

v_lshl_add_u32 v6, v4, 2, v7 // (op 985) 


v_lshrrev_b32 v4, 6, v1 // (op 1058) 
v_and_b32 v7, 15, v1 // (op 1058) 
v_lshl_add_u32 v8, v4, 4, v7 // (op 1058) 


v_and_b32 v4, 48, v1 // (op 1058) 
v_lshl_add_u32 v7, v8, 7, v4 // (op 1058) 


v_and_b32 v4, 15, v1 // (op 1097) 
v_and_b32 v8, 48, v1 // (op 1097) 
v_lshl_add_u32 v9, v4, 7, v8 // (op 1097) 


v_lshrrev_b32 v4, 6, v1 // (op 1121) 
v_and_b32 v8, 15, v1 // (op 1121) 
v_lshl_add_u32 v10, v4, 4, v8 // (op 1121) 


v_and_b32 v4, 63, v1 // (op 1121) 
v_lshrrev_b32 v8, 4, v4 // (op 1121) 

v_lshl_add_u32 v4, v10, 2, v8 // (op 1121) 


v_and_b32 v8, 15, v1 // (op 1145) 
v_and_b32 v10, 63, v1 // (op 1145) 
v_lshrrev_b32 v11, 4, v10 // (op 1145) 

v_lshl_add_u32 v10, v8, 2, v11 // (op 1145) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v11, s9 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_mul_hi_i32 s0, s16, s12 // (op 538) 
v_lshrrev_b32 v12, 3, v1 // (op 538) 
s_add_i32 s1, s0, s16 // (op 538) 

v_and_b32 v13, -8, v1 // (op 538) 
s_ashr_i32 s0, s1, 31 // (op 538) 
v_sub_u32 v14, v1, v13 // (op 538) 

s_cmp_eq_i32 s12, 0 // (op 538) 
s_mov_b32 s8, scc // (op 538) 
v_lshlrev_b32 v13, 4, v14 // (op 538) 

s_cmp_eq_i32 s8, 0 // (op 538) 
s_cselect_b32 s9, 0, -1 // (op 538) 

v_mov_b32 v15, 0 // (op 538) convert
v_mov_b32 v14, v13 // (op 538) convert

s_and_b32 s8, s13, 31 // (op 538) 
s_lshl_b32 s18, 1, s8 // (op 538) 
s_add_i32 s19, s18, s9 // (op 538) 


s_and_b32 s9, s0, s19 // (op 538) 


s_add_i32 s0, s1, s9 // (op 538) 


s_ashr_i32 s1, s0, s8 // (op 538) 


s_lshl_b32 s0, s13, 24 // (op 538) 
s_ashr_i32 s8, s0, 31 // (op 538) 

s_xor_b32 s0, s1, s8 // (op 538) 

s_sub_i32 s1, s0, s8 // (op 538) 


s_mul_hi_i32 s0, s1, s14 // (op 538) 
s_add_i32 s8, s0, s1 // (op 538) 

s_ashr_i32 s0, s8, 31 // (op 538) 
s_cmp_eq_i32 s14, 0 // (op 538) 
s_mov_b32 s9, scc // (op 538) 
s_cmp_eq_i32 s9, 0 // (op 538) 
s_cselect_b32 s18, 0, -1 // (op 538) 

s_and_b32 s9, s15, 31 // (op 538) 
s_lshl_b32 s19, 1, s9 // (op 538) 
s_add_i32 s49, s19, s18 // (op 538) 


s_and_b32 s18, s0, s49 // (op 538) 


s_add_i32 s0, s8, s18 // (op 538) 


s_ashr_i32 s8, s0, s9 // (op 538) 


s_lshl_b32 s0, s15, 24 // (op 538) 
s_ashr_i32 s9, s0, 31 // (op 538) 

s_xor_b32 s0, s8, s9 // (op 538) 

s_sub_i32 s8, s0, s9 // (op 538) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s45 // (op 538) 

s_sub_u32 s1, s16, s0 // (op 538) 

s_mul_i32 s0, s8, s45 // (op 538) 
s_add_u32 s0, s0, s1 // (op 538) 


s_mul_hi_i32 s1, s16, s11 // (op 538) 
s_add_i32 s8, s1, s16 // (op 538) 

s_ashr_i32 s1, s8, 31 // (op 538) 
s_cmp_eq_i32 s11, 0 // (op 538) 
s_mov_b32 s9, scc // (op 538) 
s_cmp_eq_i32 s9, 0 // (op 538) 
s_cselect_b32 s18, 0, -1 // (op 538) 

s_and_b32 s9, s46, 31 // (op 538) 
s_lshl_b32 s19, 1, s9 // (op 538) 
s_add_i32 s49, s19, s18 // (op 538) 


s_and_b32 s18, s1, s49 // (op 538) 


s_add_i32 s1, s8, s18 // (op 538) 


s_ashr_i32 s8, s1, s9 // (op 538) 


s_lshl_b32 s1, s46, 24 // (op 538) 
s_ashr_i32 s9, s1, 31 // (op 538) 

s_xor_b32 s1, s8, s9 // (op 538) 

s_sub_i32 s8, s1, s9 // (op 538) 


s_add_u32 s18, s4, 16 // (op 538) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 538) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 538) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 538) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 538) 

s_mul_i32 s1, s45, s18 // (op 538) 

s_mul_i32 s9, s10, s1 // (op 538) 

s_mul_i32 s1, s8, s9 // (op 538) 

s_sub_u32 s9, s16, s1 // (op 538) 

s_mov_b32 s1, s9 // (op 538) 
s_mul_hi_i32 s18, s1, s44 // (op 538) 
s_add_i32 s19, s18, s1 // (op 538) 


s_ashr_i32 s1, s19, 31 // (op 538) 
s_cmp_eq_i32 s44, 0 // (op 538) 
s_mov_b32 s18, scc // (op 538) 
s_cmp_eq_i32 s18, 0 // (op 538) 
s_cselect_b32 s49, 0, -1 // (op 538) 

s_and_b32 s18, s48, 31 // (op 538) 
s_lshl_b32 s50, 1, s18 // (op 538) 
s_add_i32 s51, s50, s49 // (op 538) 


s_and_b32 s49, s1, s51 // (op 538) 


s_add_i32 s1, s19, s49 // (op 538) 


s_ashr_i32 s19, s1, s18 // (op 538) 


s_lshl_b32 s1, s48, 24 // (op 538) 
s_ashr_i32 s18, s1, 31 // (op 538) 

s_xor_b32 s1, s19, s18 // (op 538) 

s_sub_i32 s19, s1, s18 // (op 538) 


s_mul_i32 s1, s47, s45 // (op 538) 
s_add_u32 s50, s6, 16 // (op 538) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 538) End of Int64 add, unlocking SCC
s_sub_u32 s52, s50, 1 // (op 538) least significant half; sets scc
s_subb_u32 s53, s51, 0 // (op 538) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 4 // (op 538) 

s_sub_i32 s18, s50, s1 // (op 538) 

s_mul_i32 s49, s19, s18 // (op 538) 


s_sub_u32 s18, s9, s49 // (op 538) 


s_add_u32 s9, s18, s1 // (op 538) 


s_cmp_eq_i32 s8, 0 // (op 538) 
s_mov_b32 s1, scc // (op 538) 

s_cmp_eq_i32 s1, 0 // (op 538) 
s_cselect_b32 s8, s9, s0 // (op 538) 



v_lshl_add_u32 v13, s8, 4, v12 // (op 538) 


v_mul_lo_u32 v17, v13, s3 // (op 538) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 538) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 538) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 538) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 538) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 538) most significant half



s_mul_i32 s1, s2, s7 // (op 540) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 540) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 540) least significant
s_mul_hi_u32 s9, s2, s6 // (op 540) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 540) most significant: sum
s_add_u32 s1, s1, s9 // (op 540) most significant: sum


s_mov_b32 s52, s20 // (op 540) reg expression
s_mov_b32 s53, s21 // (op 540) reg expression
s_mov_b32 s54, s0 // (op 540) reg expression
s_mov_b32 s55, 131072 // (op 540) call()

s_mul_hi_i32 s0, s16, s12 // (op 557) 
v_lshrrev_b32 v12, 3, v1 // (op 557) 
s_add_i32 s1, s0, s16 // (op 557) 

v_and_b32 v13, -8, v1 // (op 557) 
s_ashr_i32 s0, s1, 31 // (op 557) 
v_sub_u32 v14, v1, v13 // (op 557) 

s_cmp_eq_i32 s12, 0 // (op 557) 
s_mov_b32 s8, scc // (op 557) 
v_lshlrev_b32 v13, 4, v14 // (op 557) 

s_cmp_eq_i32 s8, 0 // (op 557) 
s_cselect_b32 s9, 0, -1 // (op 557) 

v_mov_b32 v15, 0 // (op 557) convert
v_mov_b32 v14, v13 // (op 557) convert

s_and_b32 s8, s13, 31 // (op 557) 
s_lshl_b32 s18, 1, s8 // (op 557) 
s_add_i32 s19, s18, s9 // (op 557) 


s_and_b32 s9, s0, s19 // (op 557) 


s_add_i32 s0, s1, s9 // (op 557) 


s_ashr_i32 s1, s0, s8 // (op 557) 


s_lshl_b32 s0, s13, 24 // (op 557) 
s_ashr_i32 s8, s0, 31 // (op 557) 

s_xor_b32 s0, s1, s8 // (op 557) 

s_sub_i32 s1, s0, s8 // (op 557) 


s_mul_hi_i32 s0, s1, s14 // (op 557) 
s_add_i32 s8, s0, s1 // (op 557) 

s_ashr_i32 s0, s8, 31 // (op 557) 
s_cmp_eq_i32 s14, 0 // (op 557) 
s_mov_b32 s9, scc // (op 557) 
s_cmp_eq_i32 s9, 0 // (op 557) 
s_cselect_b32 s18, 0, -1 // (op 557) 

s_and_b32 s9, s15, 31 // (op 557) 
s_lshl_b32 s19, 1, s9 // (op 557) 
s_add_i32 s49, s19, s18 // (op 557) 


s_and_b32 s18, s0, s49 // (op 557) 


s_add_i32 s0, s8, s18 // (op 557) 


s_ashr_i32 s8, s0, s9 // (op 557) 


s_lshl_b32 s0, s15, 24 // (op 557) 
s_ashr_i32 s9, s0, 31 // (op 557) 

s_xor_b32 s0, s8, s9 // (op 557) 

s_sub_i32 s8, s0, s9 // (op 557) 


s_mul_i32 s0, s1, s45 // (op 557) 

s_sub_u32 s1, s16, s0 // (op 557) 

s_mul_i32 s0, s8, s45 // (op 557) 
s_add_u32 s0, s0, s1 // (op 557) 


s_mul_hi_i32 s1, s16, s11 // (op 557) 
s_add_i32 s8, s1, s16 // (op 557) 

s_ashr_i32 s1, s8, 31 // (op 557) 
s_cmp_eq_i32 s11, 0 // (op 557) 
s_mov_b32 s9, scc // (op 557) 
s_cmp_eq_i32 s9, 0 // (op 557) 
s_cselect_b32 s18, 0, -1 // (op 557) 

s_and_b32 s9, s46, 31 // (op 557) 
s_lshl_b32 s19, 1, s9 // (op 557) 
s_add_i32 s49, s19, s18 // (op 557) 


s_and_b32 s18, s1, s49 // (op 557) 


s_add_i32 s1, s8, s18 // (op 557) 


s_ashr_i32 s8, s1, s9 // (op 557) 


s_lshl_b32 s1, s46, 24 // (op 557) 
s_ashr_i32 s9, s1, 31 // (op 557) 

s_xor_b32 s1, s8, s9 // (op 557) 

s_sub_i32 s8, s1, s9 // (op 557) 


s_add_u32 s18, s4, 16 // (op 557) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 557) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 557) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 557) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 557) 

s_mul_i32 s1, s45, s18 // (op 557) 

s_mul_i32 s9, s10, s1 // (op 557) 

s_mul_i32 s1, s8, s9 // (op 557) 

s_sub_u32 s9, s16, s1 // (op 557) 

s_mov_b32 s1, s9 // (op 557) 
s_mul_hi_i32 s18, s1, s44 // (op 557) 
s_add_i32 s19, s18, s1 // (op 557) 


s_ashr_i32 s1, s19, 31 // (op 557) 
s_cmp_eq_i32 s44, 0 // (op 557) 
s_mov_b32 s18, scc // (op 557) 
s_cmp_eq_i32 s18, 0 // (op 557) 
s_cselect_b32 s49, 0, -1 // (op 557) 

s_and_b32 s18, s48, 31 // (op 557) 
s_lshl_b32 s50, 1, s18 // (op 557) 
s_add_i32 s51, s50, s49 // (op 557) 


s_and_b32 s49, s1, s51 // (op 557) 


s_add_i32 s1, s19, s49 // (op 557) 


s_ashr_i32 s19, s1, s18 // (op 557) 


s_lshl_b32 s1, s48, 24 // (op 557) 
s_ashr_i32 s18, s1, 31 // (op 557) 

s_xor_b32 s1, s19, s18 // (op 557) 

s_sub_i32 s19, s1, s18 // (op 557) 


s_mul_i32 s1, s47, s45 // (op 557) 
s_add_u32 s50, s6, 16 // (op 557) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 557) End of Int64 add, unlocking SCC
s_sub_u32 s56, s50, 1 // (op 557) least significant half; sets scc
s_subb_u32 s57, s51, 0 // (op 557) most significant half; uses scc

s_ashr_i64 s[50:51], s[56:57], 4 // (op 557) 

s_sub_i32 s18, s50, s1 // (op 557) 

s_mul_i32 s49, s19, s18 // (op 557) 


s_sub_u32 s18, s9, s49 // (op 557) 


s_add_u32 s9, s18, s1 // (op 557) 


s_cmp_eq_i32 s8, 0 // (op 557) 
s_mov_b32 s1, scc // (op 557) 

s_cmp_eq_i32 s1, 0 // (op 557) 
s_cselect_b32 s8, s9, s0 // (op 557) 



v_lshl_add_u32 v13, s8, 4, v12 // (op 557) 


v_mul_lo_u32 v19, v13, s3 // (op 557) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 557) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 557) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 557) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 557) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 557) most significant half



s_mul_i32 s1, s2, s7 // (op 559) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 559) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 559) least significant
s_mul_hi_u32 s9, s2, s6 // (op 559) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 559) most significant: sum
s_add_u32 s1, s1, s9 // (op 559) most significant: sum


s_mov_b32 s52, s20 // (op 559) reg expression
s_mov_b32 s53, s21 // (op 559) reg expression
s_mov_b32 s54, s0 // (op 559) reg expression
s_mov_b32 s55, 131072 // (op 559) call()

s_mul_hi_i32 s0, s16, s12 // (op 602) 
v_lshrrev_b32 v12, 3, v1 // (op 602) 
s_add_i32 s1, s0, s16 // (op 602) 

v_and_b32 v13, -8, v1 // (op 602) 
s_ashr_i32 s0, s1, 31 // (op 602) 
v_sub_u32 v14, v1, v13 // (op 602) 

s_cmp_eq_i32 s12, 0 // (op 602) 
s_mov_b32 s8, scc // (op 602) 
v_lshlrev_b32 v13, 4, v14 // (op 602) 

s_cmp_eq_i32 s8, 0 // (op 602) 
s_cselect_b32 s9, 0, -1 // (op 602) 

v_mov_b32 v15, 0 // (op 602) convert
v_mov_b32 v14, v13 // (op 602) convert

s_and_b32 s8, s13, 31 // (op 602) 
s_lshl_b32 s18, 1, s8 // (op 602) 
s_add_i32 s19, s18, s9 // (op 602) 


s_and_b32 s9, s0, s19 // (op 602) 


s_add_i32 s0, s1, s9 // (op 602) 


s_ashr_i32 s1, s0, s8 // (op 602) 


s_lshl_b32 s0, s13, 24 // (op 602) 
s_ashr_i32 s8, s0, 31 // (op 602) 

s_xor_b32 s0, s1, s8 // (op 602) 

s_sub_i32 s1, s0, s8 // (op 602) 


s_mul_hi_i32 s0, s1, s14 // (op 602) 
s_add_i32 s8, s0, s1 // (op 602) 

s_ashr_i32 s0, s8, 31 // (op 602) 
s_cmp_eq_i32 s14, 0 // (op 602) 
s_mov_b32 s9, scc // (op 602) 
s_cmp_eq_i32 s9, 0 // (op 602) 
s_cselect_b32 s18, 0, -1 // (op 602) 

s_and_b32 s9, s15, 31 // (op 602) 
s_lshl_b32 s19, 1, s9 // (op 602) 
s_add_i32 s49, s19, s18 // (op 602) 


s_and_b32 s18, s0, s49 // (op 602) 


s_add_i32 s0, s8, s18 // (op 602) 


s_ashr_i32 s8, s0, s9 // (op 602) 


s_lshl_b32 s0, s15, 24 // (op 602) 
s_ashr_i32 s9, s0, 31 // (op 602) 

s_xor_b32 s0, s8, s9 // (op 602) 

s_sub_i32 s8, s0, s9 // (op 602) 


s_add_u32 s18, s4, 16 // (op 602) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 602) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 602) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 602) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 602) 

s_mul_i32 s0, s8, s18 // (op 602) 

s_sub_i32 s8, s1, s0 // (op 602) 


s_mul_hi_i32 s9, s16, s11 // (op 602) 
s_add_i32 s0, s9, s16 // (op 602) 

s_ashr_i32 s1, s0, 31 // (op 602) 
s_cmp_eq_i32 s11, 0 // (op 602) 
s_mov_b32 s9, scc // (op 602) 
s_cmp_eq_i32 s9, 0 // (op 602) 
s_cselect_b32 s49, 0, -1 // (op 602) 

s_and_b32 s9, s46, 31 // (op 602) 
s_lshl_b32 s50, 1, s9 // (op 602) 
s_add_i32 s51, s50, s49 // (op 602) 


s_and_b32 s49, s1, s51 // (op 602) 


s_add_i32 s1, s0, s49 // (op 602) 


s_ashr_i32 s0, s1, s9 // (op 602) 


s_lshl_b32 s1, s46, 24 // (op 602) 
s_ashr_i32 s9, s1, 31 // (op 602) 

s_xor_b32 s1, s0, s9 // (op 602) 

s_sub_i32 s0, s1, s9 // (op 602) 


s_mul_i32 s1, s45, s18 // (op 602) 

s_mul_i32 s9, s10, s1 // (op 602) 

s_mul_i32 s1, s0, s9 // (op 602) 

s_sub_u32 s9, s16, s1 // (op 602) 

s_mul_hi_i32 s1, s9, s44 // (op 602) 
s_add_i32 s18, s1, s9 // (op 602) 


s_ashr_i32 s1, s18, 31 // (op 602) 
s_cmp_eq_i32 s44, 0 // (op 602) 
s_mov_b32 s9, scc // (op 602) 
s_cmp_eq_i32 s9, 0 // (op 602) 
s_cselect_b32 s19, 0, -1 // (op 602) 

s_and_b32 s9, s48, 31 // (op 602) 
s_lshl_b32 s49, 1, s9 // (op 602) 
s_add_i32 s50, s49, s19 // (op 602) 


s_and_b32 s19, s1, s50 // (op 602) 


s_add_i32 s1, s18, s19 // (op 602) 


s_ashr_i32 s18, s1, s9 // (op 602) 


s_lshl_b32 s1, s48, 24 // (op 602) 
s_ashr_i32 s9, s1, 31 // (op 602) 

s_xor_b32 s1, s18, s9 // (op 602) 

s_sub_i32 s18, s1, s9 // (op 602) 


s_cmp_eq_i32 s0, 0 // (op 602) 
s_mov_b32 s1, scc // (op 602) 

s_cmp_eq_i32 s1, 0 // (op 602) 
s_cselect_b32 s0, s18, s8 // (op 602) 



v_lshl_add_u32 v13, s0, 4, v12 // (op 602) 


v_mul_lo_u32 v21, v13, s23 // (op 602) most significant: low of low * high
v_mul_hi_u32 v12, v13, s22 // (op 602) most significant: high of low * low
v_mul_lo_u32 v20, v13, s22 // (op 602) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 602) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 602) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 602) most significant half



s_mul_i32 s1, s22, s5 // (op 604) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 604) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 604) least significant
s_mul_hi_u32 s9, s22, s4 // (op 604) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 604) most significant: sum
s_add_u32 s1, s1, s9 // (op 604) most significant: sum


s_mov_b32 s56, s24 // (op 604) reg expression
s_mov_b32 s57, s25 // (op 604) reg expression
s_mov_b32 s58, s0 // (op 604) reg expression
s_mov_b32 s59, 131072 // (op 604) call()

s_mul_hi_i32 s0, s16, s12 // (op 621) 
v_lshrrev_b32 v14, 3, v1 // (op 621) 
s_add_i32 s1, s0, s16 // (op 621) 

v_and_b32 v15, -8, v1 // (op 621) 
s_ashr_i32 s0, s1, 31 // (op 621) 
v_sub_u32 v20, v1, v15 // (op 621) 

s_cmp_eq_i32 s12, 0 // (op 621) 
s_mov_b32 s8, scc // (op 621) 
v_lshlrev_b32 v15, 4, v20 // (op 621) 

s_cmp_eq_i32 s8, 0 // (op 621) 
s_cselect_b32 s9, 0, -1 // (op 621) 

v_mov_b32 v21, 0 // (op 621) convert
v_mov_b32 v20, v15 // (op 621) convert

s_and_b32 s8, s13, 31 // (op 621) 
s_lshl_b32 s18, 1, s8 // (op 621) 
s_add_i32 s19, s18, s9 // (op 621) 


s_and_b32 s9, s0, s19 // (op 621) 


s_add_i32 s0, s1, s9 // (op 621) 


s_ashr_i32 s1, s0, s8 // (op 621) 


s_lshl_b32 s0, s13, 24 // (op 621) 
s_ashr_i32 s8, s0, 31 // (op 621) 

s_xor_b32 s0, s1, s8 // (op 621) 

s_sub_i32 s1, s0, s8 // (op 621) 


s_mul_hi_i32 s0, s1, s14 // (op 621) 
s_add_i32 s8, s0, s1 // (op 621) 

s_ashr_i32 s0, s8, 31 // (op 621) 
s_cmp_eq_i32 s14, 0 // (op 621) 
s_mov_b32 s9, scc // (op 621) 
s_cmp_eq_i32 s9, 0 // (op 621) 
s_cselect_b32 s18, 0, -1 // (op 621) 

s_and_b32 s9, s15, 31 // (op 621) 
s_lshl_b32 s19, 1, s9 // (op 621) 
s_add_i32 s49, s19, s18 // (op 621) 


s_and_b32 s18, s0, s49 // (op 621) 


s_add_i32 s0, s8, s18 // (op 621) 


s_ashr_i32 s8, s0, s9 // (op 621) 


s_lshl_b32 s0, s15, 24 // (op 621) 
s_ashr_i32 s9, s0, 31 // (op 621) 

s_xor_b32 s0, s8, s9 // (op 621) 

s_sub_i32 s8, s0, s9 // (op 621) 


s_add_u32 s18, s4, 16 // (op 621) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 621) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 621) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 621) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 621) 

s_mul_i32 s0, s8, s18 // (op 621) 

s_sub_i32 s8, s1, s0 // (op 621) 


s_mul_hi_i32 s9, s16, s11 // (op 621) 
s_add_i32 s0, s9, s16 // (op 621) 

s_ashr_i32 s1, s0, 31 // (op 621) 
s_cmp_eq_i32 s11, 0 // (op 621) 
s_mov_b32 s9, scc // (op 621) 
s_cmp_eq_i32 s9, 0 // (op 621) 
s_cselect_b32 s49, 0, -1 // (op 621) 

s_and_b32 s9, s46, 31 // (op 621) 
s_lshl_b32 s50, 1, s9 // (op 621) 
s_add_i32 s51, s50, s49 // (op 621) 


s_and_b32 s49, s1, s51 // (op 621) 


s_add_i32 s1, s0, s49 // (op 621) 


s_ashr_i32 s0, s1, s9 // (op 621) 


s_lshl_b32 s1, s46, 24 // (op 621) 
s_ashr_i32 s9, s1, 31 // (op 621) 

s_xor_b32 s1, s0, s9 // (op 621) 

s_sub_i32 s0, s1, s9 // (op 621) 


s_mul_i32 s1, s45, s18 // (op 621) 

s_mul_i32 s9, s10, s1 // (op 621) 

s_mul_i32 s1, s0, s9 // (op 621) 

s_sub_u32 s9, s16, s1 // (op 621) 

s_mul_hi_i32 s1, s9, s44 // (op 621) 
s_add_i32 s18, s1, s9 // (op 621) 


s_ashr_i32 s1, s18, 31 // (op 621) 
s_cmp_eq_i32 s44, 0 // (op 621) 
s_mov_b32 s9, scc // (op 621) 
s_cmp_eq_i32 s9, 0 // (op 621) 
s_cselect_b32 s19, 0, -1 // (op 621) 

s_and_b32 s9, s48, 31 // (op 621) 
s_lshl_b32 s49, 1, s9 // (op 621) 
s_add_i32 s50, s49, s19 // (op 621) 


s_and_b32 s19, s1, s50 // (op 621) 


s_add_i32 s1, s18, s19 // (op 621) 


s_ashr_i32 s18, s1, s9 // (op 621) 


s_lshl_b32 s1, s48, 24 // (op 621) 
s_ashr_i32 s9, s1, 31 // (op 621) 

s_xor_b32 s1, s18, s9 // (op 621) 

s_sub_i32 s18, s1, s9 // (op 621) 


s_cmp_eq_i32 s0, 0 // (op 621) 
s_mov_b32 s1, scc // (op 621) 

s_cmp_eq_i32 s1, 0 // (op 621) 
s_cselect_b32 s0, s18, s8 // (op 621) 



v_lshl_add_u32 v15, s0, 4, v14 // (op 621) 


v_mul_lo_u32 v23, v15, s23 // (op 621) most significant: low of low * high
v_mul_hi_u32 v14, v15, s22 // (op 621) most significant: high of low * low
v_mul_lo_u32 v22, v15, s22 // (op 621) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 621) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 621) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 621) most significant half



s_mul_i32 s1, s22, s5 // (op 623) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 623) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 623) least significant
s_mul_hi_u32 s9, s22, s4 // (op 623) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 623) most significant: sum
s_add_u32 s1, s1, s9 // (op 623) most significant: sum


s_mov_b32 s56, s24 // (op 623) reg expression
s_mov_b32 s57, s25 // (op 623) reg expression
s_mov_b32 s58, s0 // (op 623) reg expression
s_mov_b32 s59, 131072 // (op 623) call()

s_mul_hi_i32 s0, s16, s12 // (op 666) 
v_lshrrev_b32 v20, 2, v1 // (op 666) 
s_add_i32 s1, s0, s16 // (op 666) 

v_and_b32 v21, -4, v1 // (op 666) 
s_ashr_i32 s0, s1, 31 // (op 666) 
v_sub_u32 v22, v1, v21 // (op 666) 

s_cmp_eq_i32 s12, 0 // (op 666) 
s_mov_b32 s8, scc // (op 666) 
s_cmp_eq_i32 s8, 0 // (op 666) 
s_cselect_b32 s9, 0, -1 // (op 666) 

s_and_b32 s8, s13, 31 // (op 666) 
s_lshl_b32 s18, 1, s8 // (op 666) 
s_add_i32 s19, s18, s9 // (op 666) 


s_and_b32 s9, s0, s19 // (op 666) 


s_add_i32 s0, s1, s9 // (op 666) 


s_ashr_i32 s1, s0, s8 // (op 666) 


s_lshl_b32 s0, s13, 24 // (op 666) 
s_ashr_i32 s8, s0, 31 // (op 666) 

s_xor_b32 s0, s1, s8 // (op 666) 

s_sub_i32 s1, s0, s8 // (op 666) 


s_mul_hi_i32 s0, s1, s14 // (op 666) 
s_add_i32 s8, s0, s1 // (op 666) 

s_ashr_i32 s0, s8, 31 // (op 666) 
s_cmp_eq_i32 s14, 0 // (op 666) 
s_mov_b32 s9, scc // (op 666) 
s_cmp_eq_i32 s9, 0 // (op 666) 
s_cselect_b32 s18, 0, -1 // (op 666) 

s_and_b32 s9, s15, 31 // (op 666) 
s_lshl_b32 s19, 1, s9 // (op 666) 
s_add_i32 s49, s19, s18 // (op 666) 


s_and_b32 s18, s0, s49 // (op 666) 


s_add_i32 s0, s8, s18 // (op 666) 


s_ashr_i32 s8, s0, s9 // (op 666) 


s_lshl_b32 s0, s15, 24 // (op 666) 
s_ashr_i32 s9, s0, 31 // (op 666) 

s_xor_b32 s0, s8, s9 // (op 666) 

s_sub_i32 s8, s0, s9 // (op 666) 


s_mul_i32 s0, s1, s45 // (op 666) 

s_sub_u32 s1, s16, s0 // (op 666) 

s_mul_i32 s0, s8, s45 // (op 666) 
s_add_u32 s0, s0, s1 // (op 666) 


s_mul_hi_i32 s1, s16, s11 // (op 666) 
s_add_i32 s8, s1, s16 // (op 666) 

s_ashr_i32 s1, s8, 31 // (op 666) 
s_cmp_eq_i32 s11, 0 // (op 666) 
s_mov_b32 s9, scc // (op 666) 
s_cmp_eq_i32 s9, 0 // (op 666) 
s_cselect_b32 s18, 0, -1 // (op 666) 

s_and_b32 s9, s46, 31 // (op 666) 
s_lshl_b32 s19, 1, s9 // (op 666) 
s_add_i32 s49, s19, s18 // (op 666) 


s_and_b32 s18, s1, s49 // (op 666) 


s_add_i32 s1, s8, s18 // (op 666) 


s_ashr_i32 s8, s1, s9 // (op 666) 


s_lshl_b32 s1, s46, 24 // (op 666) 
s_ashr_i32 s9, s1, 31 // (op 666) 

s_xor_b32 s1, s8, s9 // (op 666) 

s_sub_i32 s8, s1, s9 // (op 666) 


s_add_u32 s18, s4, 16 // (op 666) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 666) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 666) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 666) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 666) 

s_mul_i32 s1, s45, s18 // (op 666) 

s_mul_i32 s9, s10, s1 // (op 666) 

s_mul_i32 s1, s8, s9 // (op 666) 

s_sub_u32 s9, s16, s1 // (op 666) 

s_mov_b32 s1, s9 // (op 666) 
s_mul_hi_i32 s18, s1, s44 // (op 666) 
s_add_i32 s19, s18, s1 // (op 666) 


s_ashr_i32 s1, s19, 31 // (op 666) 
s_cmp_eq_i32 s44, 0 // (op 666) 
s_mov_b32 s18, scc // (op 666) 
s_cmp_eq_i32 s18, 0 // (op 666) 
s_cselect_b32 s49, 0, -1 // (op 666) 

s_and_b32 s18, s48, 31 // (op 666) 
s_lshl_b32 s50, 1, s18 // (op 666) 
s_add_i32 s51, s50, s49 // (op 666) 


s_and_b32 s49, s1, s51 // (op 666) 


s_add_i32 s1, s19, s49 // (op 666) 


s_ashr_i32 s19, s1, s18 // (op 666) 


s_lshl_b32 s1, s48, 24 // (op 666) 
s_ashr_i32 s18, s1, 31 // (op 666) 

s_xor_b32 s1, s19, s18 // (op 666) 

s_sub_i32 s19, s1, s18 // (op 666) 


s_mul_i32 s1, s47, s45 // (op 666) 
s_add_u32 s50, s6, 16 // (op 666) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 666) End of Int64 add, unlocking SCC
s_sub_u32 s60, s50, 1 // (op 666) least significant half; sets scc
s_subb_u32 s61, s51, 0 // (op 666) most significant half; uses scc

s_ashr_i64 s[50:51], s[60:61], 4 // (op 666) 

s_sub_i32 s18, s50, s1 // (op 666) 

s_mul_i32 s49, s19, s18 // (op 666) 


s_sub_u32 s18, s9, s49 // (op 666) 


s_add_u32 s9, s18, s1 // (op 666) 


s_cmp_eq_i32 s8, 0 // (op 666) 
s_mov_b32 s1, scc // (op 666) 

s_cmp_eq_i32 s1, 0 // (op 666) 
s_cselect_b32 s8, s9, s0 // (op 666) 



v_lshl_add_u32 v21, s8, 4, v20 // (op 666) 


v_mul_lo_u32 v20, v21, s26 // (op 666) 
v_add_u32 v20, v20, v22 // (op 666) 


s_mul_i32 s1, s26, s7 // (op 668) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 668) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 668) least significant
s_mul_hi_u32 s9, s26, s6 // (op 668) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 668) most significant: sum
s_add_u32 s1, s1, s9 // (op 668) most significant: sum


s_mov_b32 s60, s28 // (op 668) reg expression
s_mov_b32 s61, s29 // (op 668) reg expression
s_mov_b32 s62, s0 // (op 668) reg expression
s_mov_b32 s63, 131072 // (op 668) call()

s_mul_hi_i32 s0, s16, s12 // (op 685) 
v_lshrrev_b32 v21, 2, v1 // (op 685) 
s_add_i32 s1, s0, s16 // (op 685) 

v_and_b32 v22, -4, v1 // (op 685) 
s_ashr_i32 s0, s1, 31 // (op 685) 
v_sub_u32 v23, v1, v22 // (op 685) 

s_cmp_eq_i32 s12, 0 // (op 685) 
s_mov_b32 s8, scc // (op 685) 
s_cmp_eq_i32 s8, 0 // (op 685) 
s_cselect_b32 s9, 0, -1 // (op 685) 

s_and_b32 s8, s13, 31 // (op 685) 
s_lshl_b32 s18, 1, s8 // (op 685) 
s_add_i32 s19, s18, s9 // (op 685) 


s_and_b32 s9, s0, s19 // (op 685) 


s_add_i32 s0, s1, s9 // (op 685) 


s_ashr_i32 s1, s0, s8 // (op 685) 


s_lshl_b32 s0, s13, 24 // (op 685) 
s_ashr_i32 s8, s0, 31 // (op 685) 

s_xor_b32 s0, s1, s8 // (op 685) 

s_sub_i32 s1, s0, s8 // (op 685) 


s_mul_hi_i32 s0, s1, s14 // (op 685) 
s_add_i32 s8, s0, s1 // (op 685) 

s_ashr_i32 s0, s8, 31 // (op 685) 
s_cmp_eq_i32 s14, 0 // (op 685) 
s_mov_b32 s9, scc // (op 685) 
s_cmp_eq_i32 s9, 0 // (op 685) 
s_cselect_b32 s18, 0, -1 // (op 685) 

s_and_b32 s9, s15, 31 // (op 685) 
s_lshl_b32 s19, 1, s9 // (op 685) 
s_add_i32 s49, s19, s18 // (op 685) 


s_and_b32 s18, s0, s49 // (op 685) 


s_add_i32 s0, s8, s18 // (op 685) 


s_ashr_i32 s8, s0, s9 // (op 685) 


s_lshl_b32 s0, s15, 24 // (op 685) 
s_ashr_i32 s9, s0, 31 // (op 685) 

s_xor_b32 s0, s8, s9 // (op 685) 

s_sub_i32 s8, s0, s9 // (op 685) 


s_mul_i32 s0, s1, s45 // (op 685) 

s_sub_u32 s1, s16, s0 // (op 685) 

s_mul_i32 s0, s8, s45 // (op 685) 
s_add_u32 s0, s0, s1 // (op 685) 


s_mul_hi_i32 s1, s16, s11 // (op 685) 
s_add_i32 s8, s1, s16 // (op 685) 

s_ashr_i32 s1, s8, 31 // (op 685) 
s_cmp_eq_i32 s11, 0 // (op 685) 
s_mov_b32 s9, scc // (op 685) 
s_cmp_eq_i32 s9, 0 // (op 685) 
s_cselect_b32 s18, 0, -1 // (op 685) 

s_and_b32 s9, s46, 31 // (op 685) 
s_lshl_b32 s19, 1, s9 // (op 685) 
s_add_i32 s49, s19, s18 // (op 685) 


s_and_b32 s18, s1, s49 // (op 685) 


s_add_i32 s1, s8, s18 // (op 685) 


s_ashr_i32 s8, s1, s9 // (op 685) 


s_lshl_b32 s1, s46, 24 // (op 685) 
s_ashr_i32 s9, s1, 31 // (op 685) 

s_xor_b32 s1, s8, s9 // (op 685) 

s_sub_i32 s8, s1, s9 // (op 685) 


s_add_u32 s18, s4, 16 // (op 685) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 685) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 685) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 685) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 685) 

s_mul_i32 s1, s45, s18 // (op 685) 

s_mul_i32 s9, s10, s1 // (op 685) 

s_mul_i32 s1, s8, s9 // (op 685) 

s_sub_u32 s9, s16, s1 // (op 685) 

s_mov_b32 s1, s9 // (op 685) 
s_mul_hi_i32 s18, s1, s44 // (op 685) 
s_add_i32 s19, s18, s1 // (op 685) 


s_ashr_i32 s1, s19, 31 // (op 685) 
s_cmp_eq_i32 s44, 0 // (op 685) 
s_mov_b32 s18, scc // (op 685) 
s_cmp_eq_i32 s18, 0 // (op 685) 
s_cselect_b32 s49, 0, -1 // (op 685) 

s_and_b32 s18, s48, 31 // (op 685) 
s_lshl_b32 s50, 1, s18 // (op 685) 
s_add_i32 s51, s50, s49 // (op 685) 


s_and_b32 s49, s1, s51 // (op 685) 


s_add_i32 s1, s19, s49 // (op 685) 


s_ashr_i32 s19, s1, s18 // (op 685) 


s_lshl_b32 s1, s48, 24 // (op 685) 
s_ashr_i32 s18, s1, 31 // (op 685) 

s_xor_b32 s1, s19, s18 // (op 685) 

s_sub_i32 s19, s1, s18 // (op 685) 


s_mul_i32 s1, s47, s45 // (op 685) 
s_add_u32 s50, s6, 16 // (op 685) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 685) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 685) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 685) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 4 // (op 685) 

s_sub_i32 s18, s50, s1 // (op 685) 

s_mul_i32 s49, s19, s18 // (op 685) 


s_sub_u32 s18, s9, s49 // (op 685) 


s_add_u32 s9, s18, s1 // (op 685) 


s_cmp_eq_i32 s8, 0 // (op 685) 
s_mov_b32 s1, scc // (op 685) 

s_cmp_eq_i32 s1, 0 // (op 685) 
s_cselect_b32 s8, s9, s0 // (op 685) 



v_lshl_add_u32 v22, s8, 4, v21 // (op 685) 


v_mul_lo_u32 v21, v22, s26 // (op 685) 
v_add_u32 v21, v21, v23 // (op 685) 


s_mul_i32 s1, s26, s7 // (op 687) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 687) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 687) least significant
s_mul_hi_u32 s9, s26, s6 // (op 687) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 687) most significant: sum
s_add_u32 s1, s1, s9 // (op 687) most significant: sum


s_mov_b32 s60, s28 // (op 687) reg expression
s_mov_b32 s61, s29 // (op 687) reg expression
s_mov_b32 s62, s0 // (op 687) reg expression
s_mov_b32 s63, 131072 // (op 687) call()

s_mul_hi_i32 s0, s16, s12 // (op 730) 
v_lshrrev_b32 v22, 2, v1 // (op 730) 
s_add_i32 s1, s0, s16 // (op 730) 

v_and_b32 v23, -4, v1 // (op 730) 
s_ashr_i32 s0, s1, 31 // (op 730) 
v_sub_u32 v24, v1, v23 // (op 730) 

s_cmp_eq_i32 s12, 0 // (op 730) 
s_mov_b32 s8, scc // (op 730) 
s_cmp_eq_i32 s8, 0 // (op 730) 
s_cselect_b32 s9, 0, -1 // (op 730) 

s_and_b32 s8, s13, 31 // (op 730) 
s_lshl_b32 s18, 1, s8 // (op 730) 
s_add_i32 s19, s18, s9 // (op 730) 


s_and_b32 s9, s0, s19 // (op 730) 


s_add_i32 s0, s1, s9 // (op 730) 


s_ashr_i32 s1, s0, s8 // (op 730) 


s_lshl_b32 s0, s13, 24 // (op 730) 
s_ashr_i32 s8, s0, 31 // (op 730) 

s_xor_b32 s0, s1, s8 // (op 730) 

s_sub_i32 s1, s0, s8 // (op 730) 


s_mul_hi_i32 s0, s1, s14 // (op 730) 
s_add_i32 s8, s0, s1 // (op 730) 

s_ashr_i32 s0, s8, 31 // (op 730) 
s_cmp_eq_i32 s14, 0 // (op 730) 
s_mov_b32 s9, scc // (op 730) 
s_cmp_eq_i32 s9, 0 // (op 730) 
s_cselect_b32 s18, 0, -1 // (op 730) 

s_and_b32 s9, s15, 31 // (op 730) 
s_lshl_b32 s19, 1, s9 // (op 730) 
s_add_i32 s49, s19, s18 // (op 730) 


s_and_b32 s18, s0, s49 // (op 730) 


s_add_i32 s0, s8, s18 // (op 730) 


s_ashr_i32 s8, s0, s9 // (op 730) 


s_lshl_b32 s0, s15, 24 // (op 730) 
s_ashr_i32 s9, s0, 31 // (op 730) 

s_xor_b32 s0, s8, s9 // (op 730) 

s_sub_i32 s8, s0, s9 // (op 730) 


s_add_u32 s18, s4, 16 // (op 730) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 730) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 730) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 730) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 730) 

s_mul_i32 s0, s8, s18 // (op 730) 

s_sub_i32 s8, s1, s0 // (op 730) 


s_mul_hi_i32 s9, s16, s11 // (op 730) 
s_add_i32 s0, s9, s16 // (op 730) 

s_ashr_i32 s1, s0, 31 // (op 730) 
s_cmp_eq_i32 s11, 0 // (op 730) 
s_mov_b32 s9, scc // (op 730) 
s_cmp_eq_i32 s9, 0 // (op 730) 
s_cselect_b32 s49, 0, -1 // (op 730) 

s_and_b32 s9, s46, 31 // (op 730) 
s_lshl_b32 s50, 1, s9 // (op 730) 
s_add_i32 s51, s50, s49 // (op 730) 


s_and_b32 s49, s1, s51 // (op 730) 


s_add_i32 s1, s0, s49 // (op 730) 


s_ashr_i32 s0, s1, s9 // (op 730) 


s_lshl_b32 s1, s46, 24 // (op 730) 
s_ashr_i32 s9, s1, 31 // (op 730) 

s_xor_b32 s1, s0, s9 // (op 730) 

s_sub_i32 s0, s1, s9 // (op 730) 


s_mul_i32 s1, s45, s18 // (op 730) 

s_mul_i32 s9, s10, s1 // (op 730) 

s_mul_i32 s1, s0, s9 // (op 730) 

s_sub_u32 s9, s16, s1 // (op 730) 

s_mul_hi_i32 s1, s9, s44 // (op 730) 
s_add_i32 s18, s1, s9 // (op 730) 


s_ashr_i32 s1, s18, 31 // (op 730) 
s_cmp_eq_i32 s44, 0 // (op 730) 
s_mov_b32 s9, scc // (op 730) 
s_cmp_eq_i32 s9, 0 // (op 730) 
s_cselect_b32 s19, 0, -1 // (op 730) 

s_and_b32 s9, s48, 31 // (op 730) 
s_lshl_b32 s49, 1, s9 // (op 730) 
s_add_i32 s50, s49, s19 // (op 730) 


s_and_b32 s19, s1, s50 // (op 730) 


s_add_i32 s1, s18, s19 // (op 730) 


s_ashr_i32 s18, s1, s9 // (op 730) 


s_lshl_b32 s1, s48, 24 // (op 730) 
s_ashr_i32 s9, s1, 31 // (op 730) 

s_xor_b32 s1, s18, s9 // (op 730) 

s_sub_i32 s18, s1, s9 // (op 730) 


s_cmp_eq_i32 s0, 0 // (op 730) 
s_mov_b32 s1, scc // (op 730) 

s_cmp_eq_i32 s1, 0 // (op 730) 
s_cselect_b32 s0, s18, s8 // (op 730) 



v_lshl_add_u32 v23, s0, 4, v22 // (op 730) 


v_mul_lo_u32 v22, v23, s30 // (op 730) 

v_add_u32 v23, v24, v22 // (op 730) 


s_mul_i32 s1, s30, s5 // (op 732) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 732) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 732) least significant
s_mul_hi_u32 s9, s30, s4 // (op 732) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 732) most significant: sum
s_add_u32 s1, s1, s9 // (op 732) most significant: sum


s_mov_b32 s64, s32 // (op 732) reg expression
s_mov_b32 s65, s33 // (op 732) reg expression
s_mov_b32 s66, s0 // (op 732) reg expression
s_mov_b32 s67, 131072 // (op 732) call()

s_mul_hi_i32 s0, s16, s12 // (op 749) 
v_lshrrev_b32 v22, 2, v1 // (op 749) 
s_add_i32 s1, s0, s16 // (op 749) 

v_and_b32 v24, -4, v1 // (op 749) 
s_ashr_i32 s0, s1, 31 // (op 749) 
v_sub_u32 v25, v1, v24 // (op 749) 

s_cmp_eq_i32 s12, 0 // (op 749) 
s_mov_b32 s8, scc // (op 749) 
s_cmp_eq_i32 s8, 0 // (op 749) 
s_cselect_b32 s9, 0, -1 // (op 749) 

s_and_b32 s8, s13, 31 // (op 749) 
s_lshl_b32 s18, 1, s8 // (op 749) 
s_add_i32 s19, s18, s9 // (op 749) 


s_and_b32 s9, s0, s19 // (op 749) 


s_add_i32 s0, s1, s9 // (op 749) 


s_ashr_i32 s1, s0, s8 // (op 749) 


s_lshl_b32 s0, s13, 24 // (op 749) 
s_ashr_i32 s8, s0, 31 // (op 749) 

s_xor_b32 s0, s1, s8 // (op 749) 

s_sub_i32 s1, s0, s8 // (op 749) 


s_mul_hi_i32 s0, s1, s14 // (op 749) 
s_add_i32 s8, s0, s1 // (op 749) 

s_ashr_i32 s0, s8, 31 // (op 749) 
s_cmp_eq_i32 s14, 0 // (op 749) 
s_mov_b32 s9, scc // (op 749) 
s_cmp_eq_i32 s9, 0 // (op 749) 
s_cselect_b32 s18, 0, -1 // (op 749) 

s_and_b32 s9, s15, 31 // (op 749) 
s_lshl_b32 s19, 1, s9 // (op 749) 
s_add_i32 s49, s19, s18 // (op 749) 


s_and_b32 s18, s0, s49 // (op 749) 


s_add_i32 s0, s8, s18 // (op 749) 


s_ashr_i32 s8, s0, s9 // (op 749) 


s_lshl_b32 s0, s15, 24 // (op 749) 
s_ashr_i32 s9, s0, 31 // (op 749) 

s_xor_b32 s0, s8, s9 // (op 749) 

s_sub_i32 s8, s0, s9 // (op 749) 


s_add_u32 s18, s4, 16 // (op 749) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 749) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 749) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 749) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 749) 

s_mul_i32 s0, s8, s18 // (op 749) 

s_sub_i32 s8, s1, s0 // (op 749) 


s_mul_hi_i32 s9, s16, s11 // (op 749) 
s_add_i32 s0, s9, s16 // (op 749) 

s_ashr_i32 s1, s0, 31 // (op 749) 
s_cmp_eq_i32 s11, 0 // (op 749) 
s_mov_b32 s9, scc // (op 749) 
s_cmp_eq_i32 s9, 0 // (op 749) 
s_cselect_b32 s49, 0, -1 // (op 749) 

s_and_b32 s9, s46, 31 // (op 749) 
s_lshl_b32 s50, 1, s9 // (op 749) 
s_add_i32 s51, s50, s49 // (op 749) 


s_and_b32 s49, s1, s51 // (op 749) 


s_add_i32 s1, s0, s49 // (op 749) 


s_ashr_i32 s0, s1, s9 // (op 749) 


s_lshl_b32 s1, s46, 24 // (op 749) 
s_ashr_i32 s9, s1, 31 // (op 749) 

s_xor_b32 s1, s0, s9 // (op 749) 

s_sub_i32 s0, s1, s9 // (op 749) 


s_mul_i32 s1, s45, s18 // (op 749) 

s_mul_i32 s9, s10, s1 // (op 749) 

s_mul_i32 s1, s0, s9 // (op 749) 

s_sub_u32 s9, s16, s1 // (op 749) 

s_mul_hi_i32 s1, s9, s44 // (op 749) 
s_add_i32 s18, s1, s9 // (op 749) 


s_ashr_i32 s1, s18, 31 // (op 749) 
s_cmp_eq_i32 s44, 0 // (op 749) 
s_mov_b32 s9, scc // (op 749) 
s_cmp_eq_i32 s9, 0 // (op 749) 
s_cselect_b32 s19, 0, -1 // (op 749) 

s_and_b32 s9, s48, 31 // (op 749) 
s_lshl_b32 s49, 1, s9 // (op 749) 
s_add_i32 s50, s49, s19 // (op 749) 


s_and_b32 s19, s1, s50 // (op 749) 


s_add_i32 s1, s18, s19 // (op 749) 


s_ashr_i32 s18, s1, s9 // (op 749) 


s_lshl_b32 s1, s48, 24 // (op 749) 
s_ashr_i32 s9, s1, 31 // (op 749) 

s_xor_b32 s1, s18, s9 // (op 749) 

s_sub_i32 s18, s1, s9 // (op 749) 


s_cmp_eq_i32 s0, 0 // (op 749) 
s_mov_b32 s1, scc // (op 749) 

s_cmp_eq_i32 s1, 0 // (op 749) 
s_cselect_b32 s0, s18, s8 // (op 749) 



v_lshl_add_u32 v24, s0, 4, v22 // (op 749) 


v_mul_lo_u32 v22, v24, s30 // (op 749) 

v_add_u32 v24, v25, v22 // (op 749) 


s_mul_i32 s1, s30, s5 // (op 751) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 751) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 751) least significant
s_mul_hi_u32 s9, s30, s4 // (op 751) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 751) most significant: sum
s_add_u32 s1, s1, s9 // (op 751) most significant: sum


s_mov_b32 s64, s32 // (op 751) reg expression
s_mov_b32 s65, s33 // (op 751) reg expression
s_mov_b32 s66, s0 // (op 751) reg expression
s_mov_b32 s67, 131072 // (op 751) call()

v_and_b32 v22, -8, v1 // (op 867) 
v_sub_u32 v25, v1, v22 // (op 867) 

v_lshlrev_b32 v22, 4, v25 // (op 867) 

v_lshrrev_b32 v25, 3, v1 // (op 867) 
v_lshl_add_u32 v26, v25, 7, v22 // (op 867) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 867) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[8:9], s[22:23], 3 // (op 487) 
v_mov_b64 v[26:27], s[8:9] // (op 487) convert


v_mov_b64 v[28:29], v[12:13] // (op 487) 
s_mov_b32 m0, s0 // (op 487) reg expression
buffer_load_dwordx4 v28, s[56:59], 0 offen offset: 0 lds // (op 487) Load value direct to lds
v_add_co_u32 v28, s[8:9], v28, v26 // (op 487) least significant half
v_addc_co_u32 v29, s[8:9], 0, v29, s[8:9] // (op 487) most significant half

s_add_u32 m0, m0, 1024 // (op 487) 
buffer_load_dwordx4 v28, s[56:59], 0 offen offset: 0 lds // (op 487) Load value direct to lds






buffer_load_ubyte v22, v23, s[64:67], 0 offen offset: 0 // (op 369) Load value
v_and_b32 v25, -8, v1 // (op 815) 
v_sub_u32 v26, v1, v25 // (op 815) 

v_lshlrev_b32 v25, 4, v26 // (op 815) 

v_lshrrev_b32 v26, 3, v1 // (op 815) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 815) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 815) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[8:9], s[2:3], 3 // (op 492) 
v_mov_b64 v[26:27], s[8:9] // (op 492) convert


v_mov_b64 v[28:29], v[16:17] // (op 492) 
s_add_u32 s1, 2048, s0 // (op 492) 
s_mov_b32 m0, s1 // (op 492) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 492) Load value direct to lds
v_add_co_u32 v28, s[8:9], v28, v26 // (op 492) least significant half
v_addc_co_u32 v29, s[8:9], 0, v29, s[8:9] // (op 492) most significant half

s_add_u32 m0, m0, 1024 // (op 492) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 492) Load value direct to lds







buffer_load_ubyte v25, v20, s[60:63], 0 offen offset: 0 // (op 375) Load value
v_and_b32 v26, -8, v1 // (op 1085) 
v_sub_u32 v27, v1, v26 // (op 1085) 

v_lshlrev_b32 v26, 4, v27 // (op 1085) 

v_lshrrev_b32 v27, 3, v1 // (op 1085) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 1085) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1085) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[8:9], s[22:23], 3 // (op 497) 
v_mov_b64 v[26:27], s[8:9] // (op 497) convert


v_mov_b64 v[28:29], v[12:13] // (op 497) 
s_add_u32 s1, 4096, s0 // (op 497) 
s_mov_b32 m0, s1 // (op 497) reg expression
s_mov_b32 s8, 128 // (op 497) call()
buffer_load_dwordx4 v28, s[56:59], s8 offen offset: 0 lds // (op 497) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 497) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 497) most significant half

s_add_u32 m0, m0, 1024 // (op 497) 
buffer_load_dwordx4 v28, s[56:59], s8 offen offset: 0 lds // (op 497) Load value direct to lds









buffer_load_ubyte v12, v23, s[64:67], 0 offen offset:4 // (op 381) Load value

v_and_b32 v13, -8, v1 // (op 1046) 
v_sub_u32 v23, v1, v13 // (op 1046) 

v_lshlrev_b32 v13, 4, v23 // (op 1046) 

v_lshrrev_b32 v23, 3, v1 // (op 1046) 
v_lshl_add_u32 v26, v23, 7, v13 // (op 1046) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1046) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[8:9], s[2:3], 3 // (op 499) 
v_mov_b64 v[26:27], s[8:9] // (op 499) convert


v_mov_b64 v[28:29], v[16:17] // (op 499) 
s_add_u32 s1, 6144, s0 // (op 499) 
s_mov_b32 m0, s1 // (op 499) reg expression
s_mov_b32 s8, 128 // (op 499) call()
buffer_load_dwordx4 v28, s[52:55], s8 offen offset: 0 lds // (op 499) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 499) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 499) most significant half

s_add_u32 m0, m0, 1024 // (op 499) 
buffer_load_dwordx4 v28, s[52:55], s8 offen offset: 0 lds // (op 499) Load value direct to lds









buffer_load_ubyte v13, v20, s[60:63], 0 offen offset:4 // (op 387) Load value

v_and_b32 v20, -4, v1 // (op 973) 
v_sub_u32 v23, v1, v20 // (op 973) 

v_and_b32 v20, 4294967292, v1 // (op 973) 
v_add_u32 v16, v20, v23 // (op 973) 





s_waitcnt vmcnt(9)
ds_write_b8 v16, v22 offset:8192 // (op 390) Store local data 

v_and_b32 v20, -4, v1 // (op 920) 
v_sub_u32 v23, v1, v20 // (op 920) 

v_and_b32 v20, 4294967292, v1 // (op 920) 
v_add_u32 v16, v20, v23 // (op 920) 





s_waitcnt vmcnt(6)
ds_write_b8 v16, v25 offset:8256 // (op 392) Store local data 

s_barrier  // (op 1203) 



ds_read_b128 v[26:29], v5 // (op 409) Load local data 
ds_read_b128 v[30:33], v5 offset:64 // (op 409) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1237) 



ds_read_u8 v20, v6 offset:8192 // (op 415) Load local data 
s_barrier  // (op 1187) 



ds_read_b128 v[34:37], v3 offset:2048 // (op 406) Load local data 
ds_read_b128 v[38:41], v3 offset:2112 // (op 406) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1220) 



ds_read_u8 v23, v0 offset:8256 // (op 412) Load local data 
v_and_b32 v16, -8, v1 // (op 787) 
v_sub_u32 v17, v1, v16 // (op 787) 

v_lshlrev_b32 v16, 4, v17 // (op 787) 

v_lshrrev_b32 v17, 3, v1 // (op 787) 
v_lshl_add_u32 v42, v17, 7, v16 // (op 787) 


s_nop 0
v_readfirstlane_b32 s0, v42 // (op 787) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 839) 
v_sub_u32 v17, v1, v16 // (op 839) 

v_lshlrev_b32 v16, 4, v17 // (op 839) 

v_lshrrev_b32 v17, 3, v1 // (op 839) 
v_lshl_add_u32 v42, v17, 7, v16 // (op 839) 


s_nop 0
v_readfirstlane_b32 s1, v42 // (op 839) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 891) 
v_sub_u32 v17, v1, v16 // (op 891) 

v_and_b32 v16, 4294967292, v1 // (op 891) 
v_add_u32 v42, v16, v17 // (op 891) 


v_and_b32 v16, -4, v1 // (op 944) 
v_sub_u32 v17, v1, v16 // (op 944) 

v_and_b32 v16, 4294967292, v1 // (op 944) 
v_add_u32 v43, v16, v17 // (op 944) 


v_and_b32 v16, -8, v1 // (op 1031) 
v_sub_u32 v17, v1, v16 // (op 1031) 

v_lshlrev_b32 v16, 4, v17 // (op 1031) 

v_lshrrev_b32 v17, 3, v1 // (op 1031) 
v_lshl_add_u32 v44, v17, 7, v16 // (op 1031) 


s_nop 0
v_readfirstlane_b32 s8, v44 // (op 1031) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 1070) 
v_sub_u32 v17, v1, v16 // (op 1070) 

v_lshlrev_b32 v16, 4, v17 // (op 1070) 

v_lshrrev_b32 v17, 3, v1 // (op 1070) 
v_lshl_add_u32 v44, v17, 7, v16 // (op 1070) 


s_nop 0
v_readfirstlane_b32 s9, v44 // (op 1070) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 1109) 
v_sub_u32 v17, v1, v16 // (op 1109) 

v_and_b32 v16, 4294967292, v1 // (op 1109) 
v_add_u32 v44, v16, v17 // (op 1109) 


v_and_b32 v16, -4, v1 // (op 1133) 
v_sub_u32 v17, v1, v16 // (op 1133) 

v_and_b32 v16, 4294967292, v1 // (op 1133) 
v_add_u32 v45, v16, v17 // (op 1133) 


s_mov_b32 s18, 0 // (op 59) call()
s_ashr_i64 s[50:51], s[34:35], 7 // (op 58) 
s_mov_b32 s19, s50 // (op 58) convert

s_and_b32 s49, s19, 4294967294 // (op 58) 

s_cmp_lt_u32 s18, s49 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_58 if false)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_58:
s_add_u32 s18, s18, 2 // (op 60) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v23, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 88) 
s_barrier  // (op 1206) 

s_lshl_b64 s[50:51], s[22:23], 3 // (op 459) 
v_mov_b64 v[16:17], s[50:51] // (op 459) convert


v_mov_b64 v[46:47], v[14:15] // (op 459) 
v_mov_b64 v[48:49], v[46:47] // (op 459) 
s_mov_b32 m0, s1 // (op 459) reg expression
s_mov_b32 s19, 256 // (op 459) call()
buffer_load_dwordx4 v48, s[56:59], s19 offen offset: 0 lds // (op 459) Load value direct to lds
v_add_co_u32 v48, s[50:51], v48, v16 // (op 459) least significant half
v_addc_co_u32 v49, s[50:51], 0, v49, s[50:51] // (op 459) most significant half

s_add_u32 m0, m0, 1024 // (op 459) 
buffer_load_dwordx4 v48, s[56:59], s19 offen offset: 0 lds // (op 459) Load value direct to lds





v_mov_b32 v16, v24 // (op 25) 

buffer_load_ubyte v22, v16, s[64:67], 0 offen offset:8 // (op 25) Load value
s_barrier  // (op 1190) 

s_lshl_b64 s[50:51], s[2:3], 3 // (op 454) 
v_mov_b64 v[48:49], s[50:51] // (op 454) convert


v_mov_b64 v[50:51], v[18:19] // (op 454) 
v_mov_b64 v[52:53], v[50:51] // (op 454) 
s_add_u32 s19, 2048, s0 // (op 454) 
s_mov_b32 m0, s19 // (op 454) reg expression
s_mov_b32 s49, 256 // (op 454) call()
buffer_load_dwordx4 v52, s[52:55], s49 offen offset: 0 lds // (op 454) Load value direct to lds
v_add_co_u32 v52, s[50:51], v52, v48 // (op 454) least significant half
v_addc_co_u32 v53, s[50:51], 0, v53, s[50:51] // (op 454) most significant half

s_add_u32 m0, m0, 1024 // (op 454) 
buffer_load_dwordx4 v52, s[52:55], s49 offen offset: 0 lds // (op 454) Load value direct to lds






v_mov_b32 v17, v21 // (op 17) 

buffer_load_ubyte v25, v17, s[60:63], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 1276) 



s_waitcnt vmcnt(9)
ds_write_b8 v45, v12 offset:8320 // (op 255) Store local data 
s_barrier  // (op 1270) 



s_waitcnt vmcnt(6)
ds_write_b8 v44, v13 offset:8384 // (op 254) Store local data 
s_barrier  // (op 1260) 



ds_read_b128 v[26:29], v9 offset:4096 // (op 247) Load local data 
ds_read_b128 v[30:33], v9 offset:4160 // (op 247) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1273) 



ds_read_u8 v20, v10 offset:8320 // (op 251) Load local data 
s_barrier  // (op 1254) 
s_barrier  // (op 1327) 



ds_read_b128 v[34:37], v7 offset:6144 // (op 245) Load local data 
ds_read_b128 v[38:41], v7 offset:6208 // (op 245) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1267) 



ds_read_u8 v23, v4 offset:8384 // (op 249) Load local data 
s_barrier  // (op 1321) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v23, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 252) 
s_barrier  // (op 1264) 

s_lshl_b64 s[50:51], s[22:23], 3 // (op 482) 
v_mov_b64 v[48:49], s[50:51] // (op 482) convert


v_mov_b64 v[52:53], v[46:47] // (op 482) 
s_add_u32 s19, 4096, s9 // (op 482) 
s_mov_b32 m0, s19 // (op 482) reg expression
s_mov_b32 s49, 384 // (op 482) call()
buffer_load_dwordx4 v52, s[56:59], s49 offen offset: 0 lds // (op 482) Load value direct to lds
v_add_co_u32 v52, s[50:51], v52, v48 // (op 482) least significant half
v_addc_co_u32 v53, s[50:51], 0, v53, s[50:51] // (op 482) most significant half

s_add_u32 m0, m0, 1024 // (op 482) 
buffer_load_dwordx4 v52, s[56:59], s49 offen offset: 0 lds // (op 482) Load value direct to lds








v_add_co_u32 v14, vcc, 256, v14 // (op 613) least significant half
buffer_load_ubyte v12, v16, s[64:67], 0 offen offset:12 // (op 258) Load value
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 613) most significant half

s_barrier  // (op 1257) 
v_add_u32 v24, 8, v24 // (op 741) 

s_lshl_b64 s[50:51], s[2:3], 3 // (op 477) 
v_mov_b64 v[46:47], s[50:51] // (op 477) convert


v_mov_b64 v[48:49], v[50:51] // (op 477) 
s_add_u32 s19, 6144, s8 // (op 477) 
s_mov_b32 m0, s19 // (op 477) reg expression
s_mov_b32 s49, 384 // (op 477) call()
buffer_load_dwordx4 v48, s[52:55], s49 offen offset: 0 lds // (op 477) Load value direct to lds
v_add_co_u32 v48, s[50:51], v48, v46 // (op 477) least significant half
v_addc_co_u32 v49, s[50:51], 0, v49, s[50:51] // (op 477) most significant half

s_add_u32 m0, m0, 1024 // (op 477) 
buffer_load_dwordx4 v48, s[52:55], s49 offen offset: 0 lds // (op 477) Load value direct to lds








v_add_co_u32 v18, vcc, 256, v18 // (op 549) least significant half
buffer_load_ubyte v13, v17, s[60:63], 0 offen offset:12 // (op 257) Load value
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 549) most significant half

s_barrier  // (op 1240) 
v_add_u32 v21, 8, v21 // (op 677) 



s_waitcnt vmcnt(9)
ds_write_b8 v43, v22 offset:8192 // (op 30) Store local data 
s_barrier  // (op 1223) 



s_waitcnt vmcnt(6)
ds_write_b8 v42, v25 offset:8256 // (op 21) Store local data 
s_barrier  // (op 1212) 



ds_read_b128 v[26:29], v5 // (op 75) Load local data 
ds_read_b128 v[30:33], v5 offset:64 // (op 75) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1247) 



ds_read_u8 v20, v6 offset:8192 // (op 85) Load local data 
s_barrier  // (op 1196) 



ds_read_b128 v[34:37], v3 offset:2048 // (op 70) Load local data 
ds_read_b128 v[38:41], v3 offset:2112 // (op 70) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1230) 



ds_read_u8 v23, v0 offset:8256 // (op 80) Load local data 
s_ashr_i64 s[50:51], s[34:35], 7 // (op 58) 
s_mov_b32 s19, s50 // (op 58) convert

s_and_b32 s49, s19, 4294967294 // (op 58) 

s_cmp_lt_u32 s18, s49 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_58 if true)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_58:















s_barrier  // (op 1287) 
s_mul_hi_i32 s0, s16, s12 // (op 512) 
v_lshrrev_b32 v14, 3, v1 // (op 512) 
s_add_i32 s1, s0, s16 // (op 512) 

v_and_b32 v15, -8, v1 // (op 512) 
s_ashr_i32 s0, s1, 31 // (op 512) 
v_sub_u32 v16, v1, v15 // (op 512) 

s_cmp_eq_i32 s12, 0 // (op 512) 
s_mov_b32 s8, scc // (op 512) 
v_lshlrev_b32 v15, 4, v16 // (op 512) 

s_cmp_eq_i32 s8, 0 // (op 512) 
s_cselect_b32 s9, 0, -1 // (op 512) 

s_and_b32 s8, s13, 31 // (op 512) 
s_lshl_b32 s18, 1, s8 // (op 512) 
s_add_i32 s19, s18, s9 // (op 512) 


s_and_b32 s9, s0, s19 // (op 512) 


s_add_i32 s0, s1, s9 // (op 512) 


s_ashr_i32 s1, s0, s8 // (op 512) 


s_lshl_b32 s0, s13, 24 // (op 512) 
s_ashr_i32 s8, s0, 31 // (op 512) 

s_xor_b32 s0, s1, s8 // (op 512) 

s_sub_i32 s1, s0, s8 // (op 512) 


s_mul_hi_i32 s0, s1, s14 // (op 512) 
s_add_i32 s8, s0, s1 // (op 512) 

s_ashr_i32 s0, s8, 31 // (op 512) 
s_cmp_eq_i32 s14, 0 // (op 512) 
s_mov_b32 s9, scc // (op 512) 
s_cmp_eq_i32 s9, 0 // (op 512) 
s_cselect_b32 s18, 0, -1 // (op 512) 

s_and_b32 s9, s15, 31 // (op 512) 
s_lshl_b32 s19, 1, s9 // (op 512) 
s_add_i32 s49, s19, s18 // (op 512) 


s_and_b32 s18, s0, s49 // (op 512) 


s_add_i32 s0, s8, s18 // (op 512) 


s_ashr_i32 s8, s0, s9 // (op 512) 


s_lshl_b32 s0, s15, 24 // (op 512) 
s_ashr_i32 s9, s0, 31 // (op 512) 

s_xor_b32 s0, s8, s9 // (op 512) 

s_sub_i32 s8, s0, s9 // (op 512) 


s_mul_i32 s0, s1, s45 // (op 512) 

s_sub_u32 s1, s16, s0 // (op 512) 

s_mul_i32 s0, s8, s45 // (op 512) 
s_add_u32 s0, s0, s1 // (op 512) 


s_mul_hi_i32 s1, s16, s11 // (op 512) 
s_add_i32 s8, s1, s16 // (op 512) 

s_ashr_i32 s1, s8, 31 // (op 512) 
s_cmp_eq_i32 s11, 0 // (op 512) 
s_mov_b32 s9, scc // (op 512) 
s_cmp_eq_i32 s9, 0 // (op 512) 
s_cselect_b32 s18, 0, -1 // (op 512) 

s_and_b32 s9, s46, 31 // (op 512) 
s_lshl_b32 s19, 1, s9 // (op 512) 
s_add_i32 s49, s19, s18 // (op 512) 


s_and_b32 s18, s1, s49 // (op 512) 


s_add_i32 s1, s8, s18 // (op 512) 


s_ashr_i32 s8, s1, s9 // (op 512) 


s_lshl_b32 s1, s46, 24 // (op 512) 
s_ashr_i32 s9, s1, 31 // (op 512) 

s_xor_b32 s1, s8, s9 // (op 512) 

s_sub_i32 s8, s1, s9 // (op 512) 


s_add_u32 s18, s4, 16 // (op 512) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 512) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 512) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 512) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 512) 

s_mul_i32 s1, s45, s18 // (op 512) 

s_mul_i32 s9, s10, s1 // (op 512) 

s_mul_i32 s1, s8, s9 // (op 512) 

s_sub_u32 s9, s16, s1 // (op 512) 

s_mov_b32 s1, s9 // (op 512) 
s_mul_hi_i32 s18, s1, s44 // (op 512) 
s_add_i32 s19, s18, s1 // (op 512) 


s_ashr_i32 s1, s19, 31 // (op 512) 
s_cmp_eq_i32 s44, 0 // (op 512) 
s_mov_b32 s18, scc // (op 512) 
s_cmp_eq_i32 s18, 0 // (op 512) 
s_cselect_b32 s49, 0, -1 // (op 512) 

s_and_b32 s18, s48, 31 // (op 512) 
s_lshl_b32 s50, 1, s18 // (op 512) 
s_add_i32 s51, s50, s49 // (op 512) 


s_and_b32 s49, s1, s51 // (op 512) 


s_add_i32 s1, s19, s49 // (op 512) 


s_ashr_i32 s19, s1, s18 // (op 512) 


s_lshl_b32 s1, s48, 24 // (op 512) 
s_ashr_i32 s18, s1, 31 // (op 512) 

s_xor_b32 s1, s19, s18 // (op 512) 

s_sub_i32 s19, s1, s18 // (op 512) 


s_mul_i32 s1, s47, s45 // (op 512) 
s_add_u32 s50, s6, 16 // (op 512) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 512) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 512) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 512) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 512) 

s_sub_i32 s18, s50, s1 // (op 512) 

s_mul_i32 s49, s19, s18 // (op 512) 


s_sub_u32 s18, s9, s49 // (op 512) 


s_add_u32 s9, s18, s1 // (op 512) 


s_cmp_eq_i32 s8, 0 // (op 512) 
s_mov_b32 s1, scc // (op 512) 

s_cmp_eq_i32 s1, 0 // (op 512) 
s_cselect_b32 s8, s9, s0 // (op 512) 



v_lshl_add_u32 v16, s8, 4, v14 // (op 512) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 512) 
s_lshr_b32 s8, s0, 1 // (op 512) 

v_lshl_add_u32 v14, s8, 8, v15 // (op 512) 


v_mov_b32 v19, 0 // (op 512) convert
v_mov_b32 v18, v14 // (op 512) convert

v_mul_lo_u32 v15, v16, s3 // (op 512) most significant: low of low * high
v_mul_hi_u32 v17, v16, s2 // (op 512) most significant: high of low * low
v_mul_lo_u32 v14, v16, s2 // (op 512) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 512) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v18 // (op 512) least significant half
v_addc_co_u32 v15, s[0:1], v15, v19, s[0:1] // (op 512) most significant half



s_mul_i32 s1, s2, s7 // (op 514) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 514) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 514) least significant
s_mul_hi_u32 s9, s2, s6 // (op 514) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 514) most significant: sum
s_add_u32 s1, s1, s9 // (op 514) most significant: sum


s_mov_b32 s52, s20 // (op 514) reg expression
s_mov_b32 s53, s21 // (op 514) reg expression
s_mov_b32 s54, s0 // (op 514) reg expression
s_mov_b32 s55, 131072 // (op 514) call()

s_mul_hi_i32 s0, s16, s12 // (op 579) 
v_lshrrev_b32 v16, 3, v1 // (op 579) 
s_add_i32 s1, s0, s16 // (op 579) 

v_and_b32 v17, -8, v1 // (op 579) 
s_ashr_i32 s0, s1, 31 // (op 579) 
v_sub_u32 v18, v1, v17 // (op 579) 

s_cmp_eq_i32 s12, 0 // (op 579) 
s_mov_b32 s8, scc // (op 579) 
v_lshlrev_b32 v17, 4, v18 // (op 579) 

s_cmp_eq_i32 s8, 0 // (op 579) 
s_cselect_b32 s9, 0, -1 // (op 579) 

s_and_b32 s8, s13, 31 // (op 579) 
s_lshl_b32 s18, 1, s8 // (op 579) 
s_add_i32 s19, s18, s9 // (op 579) 


s_and_b32 s9, s0, s19 // (op 579) 


s_add_i32 s0, s1, s9 // (op 579) 


s_ashr_i32 s1, s0, s8 // (op 579) 


s_lshl_b32 s0, s13, 24 // (op 579) 
s_ashr_i32 s8, s0, 31 // (op 579) 

s_xor_b32 s0, s1, s8 // (op 579) 

s_sub_i32 s1, s0, s8 // (op 579) 


s_mul_hi_i32 s0, s1, s14 // (op 579) 
s_add_i32 s8, s0, s1 // (op 579) 

s_ashr_i32 s0, s8, 31 // (op 579) 
s_cmp_eq_i32 s14, 0 // (op 579) 
s_mov_b32 s9, scc // (op 579) 
s_cmp_eq_i32 s9, 0 // (op 579) 
s_cselect_b32 s18, 0, -1 // (op 579) 

s_and_b32 s9, s15, 31 // (op 579) 
s_lshl_b32 s19, 1, s9 // (op 579) 
s_add_i32 s49, s19, s18 // (op 579) 


s_and_b32 s18, s0, s49 // (op 579) 


s_add_i32 s0, s8, s18 // (op 579) 


s_ashr_i32 s8, s0, s9 // (op 579) 


s_lshl_b32 s0, s15, 24 // (op 579) 
s_ashr_i32 s9, s0, 31 // (op 579) 

s_xor_b32 s0, s8, s9 // (op 579) 

s_sub_i32 s8, s0, s9 // (op 579) 


s_add_u32 s18, s4, 16 // (op 579) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 579) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 579) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 579) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 579) 

s_mul_i32 s0, s8, s18 // (op 579) 

s_sub_i32 s8, s1, s0 // (op 579) 


s_mul_hi_i32 s9, s16, s11 // (op 579) 
s_add_i32 s0, s9, s16 // (op 579) 

s_ashr_i32 s1, s0, 31 // (op 579) 
s_cmp_eq_i32 s11, 0 // (op 579) 
s_mov_b32 s9, scc // (op 579) 
s_cmp_eq_i32 s9, 0 // (op 579) 
s_cselect_b32 s49, 0, -1 // (op 579) 

s_and_b32 s9, s46, 31 // (op 579) 
s_lshl_b32 s50, 1, s9 // (op 579) 
s_add_i32 s51, s50, s49 // (op 579) 


s_and_b32 s49, s1, s51 // (op 579) 


s_add_i32 s1, s0, s49 // (op 579) 


s_ashr_i32 s0, s1, s9 // (op 579) 


s_lshl_b32 s1, s46, 24 // (op 579) 
s_ashr_i32 s9, s1, 31 // (op 579) 

s_xor_b32 s1, s0, s9 // (op 579) 

s_sub_i32 s0, s1, s9 // (op 579) 


s_mul_i32 s1, s45, s18 // (op 579) 

s_mul_i32 s9, s10, s1 // (op 579) 

s_mul_i32 s1, s0, s9 // (op 579) 

s_sub_u32 s9, s16, s1 // (op 579) 

s_mul_hi_i32 s1, s9, s44 // (op 579) 
s_add_i32 s18, s1, s9 // (op 579) 


s_ashr_i32 s1, s18, 31 // (op 579) 
s_cmp_eq_i32 s44, 0 // (op 579) 
s_mov_b32 s9, scc // (op 579) 
s_cmp_eq_i32 s9, 0 // (op 579) 
s_cselect_b32 s19, 0, -1 // (op 579) 

s_and_b32 s9, s48, 31 // (op 579) 
s_lshl_b32 s49, 1, s9 // (op 579) 
s_add_i32 s50, s49, s19 // (op 579) 


s_and_b32 s19, s1, s50 // (op 579) 


s_add_i32 s1, s18, s19 // (op 579) 


s_ashr_i32 s18, s1, s9 // (op 579) 


s_lshl_b32 s1, s48, 24 // (op 579) 
s_ashr_i32 s9, s1, 31 // (op 579) 

s_xor_b32 s1, s18, s9 // (op 579) 

s_sub_i32 s18, s1, s9 // (op 579) 


s_cmp_eq_i32 s0, 0 // (op 579) 
s_mov_b32 s1, scc // (op 579) 

s_cmp_eq_i32 s1, 0 // (op 579) 
s_cselect_b32 s0, s18, s8 // (op 579) 



v_lshl_add_u32 v18, s0, 4, v16 // (op 579) 


v_mul_lo_u32 v25, v18, s23 // (op 579) most significant: low of low * high
v_mul_hi_u32 v16, v18, s22 // (op 579) most significant: high of low * low
v_mul_lo_u32 v24, v18, s22 // (op 579) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 579) most significant: sum


s_ashr_i64 s[0:1], s[34:35], 7 // (op 579) 
s_lshr_b32 s8, s0, 1 // (op 579) 

v_lshl_add_u32 v16, s8, 8, v17 // (op 579) 


v_mov_b32 v19, 0 // (op 579) convert
v_mov_b32 v18, v16 // (op 579) convert

v_add_co_u32 v16, s[0:1], v18, v24 // (op 579) least significant half
v_addc_co_u32 v17, s[0:1], v19, v25, s[0:1] // (op 579) most significant half



s_mul_i32 s1, s22, s5 // (op 581) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 581) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 581) least significant
s_mul_hi_u32 s9, s22, s4 // (op 581) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 581) most significant: sum
s_add_u32 s1, s1, s9 // (op 581) most significant: sum


s_mov_b32 s56, s24 // (op 581) reg expression
s_mov_b32 s57, s25 // (op 581) reg expression
s_mov_b32 s58, s0 // (op 581) reg expression
s_mov_b32 s59, 131072 // (op 581) call()

s_mul_hi_i32 s0, s16, s12 // (op 643) 
v_lshrrev_b32 v18, 2, v1 // (op 643) 
s_add_i32 s1, s0, s16 // (op 643) 

v_and_b32 v19, -4, v1 // (op 643) 
s_ashr_i32 s0, s1, 31 // (op 643) 
v_sub_u32 v21, v1, v19 // (op 643) 

s_cmp_eq_i32 s12, 0 // (op 643) 
s_mov_b32 s8, scc // (op 643) 
s_cmp_eq_i32 s8, 0 // (op 643) 
s_cselect_b32 s9, 0, -1 // (op 643) 

s_and_b32 s8, s13, 31 // (op 643) 
s_lshl_b32 s18, 1, s8 // (op 643) 
s_add_i32 s19, s18, s9 // (op 643) 


s_and_b32 s9, s0, s19 // (op 643) 


s_add_i32 s0, s1, s9 // (op 643) 


s_ashr_i32 s1, s0, s8 // (op 643) 


s_lshl_b32 s0, s13, 24 // (op 643) 
s_ashr_i32 s8, s0, 31 // (op 643) 

s_xor_b32 s0, s1, s8 // (op 643) 

s_sub_i32 s1, s0, s8 // (op 643) 


s_mul_hi_i32 s0, s1, s14 // (op 643) 
s_add_i32 s8, s0, s1 // (op 643) 

s_ashr_i32 s0, s8, 31 // (op 643) 
s_cmp_eq_i32 s14, 0 // (op 643) 
s_mov_b32 s9, scc // (op 643) 
s_cmp_eq_i32 s9, 0 // (op 643) 
s_cselect_b32 s18, 0, -1 // (op 643) 

s_and_b32 s9, s15, 31 // (op 643) 
s_lshl_b32 s19, 1, s9 // (op 643) 
s_add_i32 s49, s19, s18 // (op 643) 


s_and_b32 s18, s0, s49 // (op 643) 


s_add_i32 s0, s8, s18 // (op 643) 


s_ashr_i32 s8, s0, s9 // (op 643) 


s_lshl_b32 s0, s15, 24 // (op 643) 
s_ashr_i32 s9, s0, 31 // (op 643) 

s_xor_b32 s0, s8, s9 // (op 643) 

s_sub_i32 s8, s0, s9 // (op 643) 


s_mul_i32 s0, s1, s45 // (op 643) 

s_sub_u32 s1, s16, s0 // (op 643) 

s_mul_i32 s0, s8, s45 // (op 643) 
s_add_u32 s0, s0, s1 // (op 643) 


s_mul_hi_i32 s1, s16, s11 // (op 643) 
s_add_i32 s8, s1, s16 // (op 643) 

s_ashr_i32 s1, s8, 31 // (op 643) 
s_cmp_eq_i32 s11, 0 // (op 643) 
s_mov_b32 s9, scc // (op 643) 
s_cmp_eq_i32 s9, 0 // (op 643) 
s_cselect_b32 s18, 0, -1 // (op 643) 

s_and_b32 s9, s46, 31 // (op 643) 
s_lshl_b32 s19, 1, s9 // (op 643) 
s_add_i32 s49, s19, s18 // (op 643) 


s_and_b32 s18, s1, s49 // (op 643) 


s_add_i32 s1, s8, s18 // (op 643) 


s_ashr_i32 s8, s1, s9 // (op 643) 


s_lshl_b32 s1, s46, 24 // (op 643) 
s_ashr_i32 s9, s1, 31 // (op 643) 

s_xor_b32 s1, s8, s9 // (op 643) 

s_sub_i32 s8, s1, s9 // (op 643) 


s_add_u32 s18, s4, 16 // (op 643) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 643) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 643) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 643) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 643) 

s_mul_i32 s1, s45, s18 // (op 643) 

s_mul_i32 s9, s10, s1 // (op 643) 

s_mul_i32 s1, s8, s9 // (op 643) 

s_sub_u32 s9, s16, s1 // (op 643) 

s_mov_b32 s1, s9 // (op 643) 
s_mul_hi_i32 s18, s1, s44 // (op 643) 
s_add_i32 s19, s18, s1 // (op 643) 


s_ashr_i32 s1, s19, 31 // (op 643) 
s_cmp_eq_i32 s44, 0 // (op 643) 
s_mov_b32 s18, scc // (op 643) 
s_cmp_eq_i32 s18, 0 // (op 643) 
s_cselect_b32 s49, 0, -1 // (op 643) 

s_and_b32 s18, s48, 31 // (op 643) 
s_lshl_b32 s50, 1, s18 // (op 643) 
s_add_i32 s51, s50, s49 // (op 643) 


s_and_b32 s49, s1, s51 // (op 643) 


s_add_i32 s1, s19, s49 // (op 643) 


s_ashr_i32 s19, s1, s18 // (op 643) 


s_lshl_b32 s1, s48, 24 // (op 643) 
s_ashr_i32 s18, s1, 31 // (op 643) 

s_xor_b32 s1, s19, s18 // (op 643) 

s_sub_i32 s19, s1, s18 // (op 643) 


s_mul_i32 s1, s47, s45 // (op 643) 
s_add_u32 s50, s6, 16 // (op 643) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 643) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 643) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 643) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 643) 

s_sub_i32 s18, s50, s1 // (op 643) 

s_mul_i32 s49, s19, s18 // (op 643) 


s_sub_u32 s18, s9, s49 // (op 643) 


s_add_u32 s9, s18, s1 // (op 643) 


s_cmp_eq_i32 s8, 0 // (op 643) 
s_mov_b32 s1, scc // (op 643) 

s_cmp_eq_i32 s1, 0 // (op 643) 
s_cselect_b32 s8, s9, s0 // (op 643) 



v_lshl_add_u32 v19, s8, 4, v18 // (op 643) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 643) 
s_lshr_b32 s8, s0, 1 // (op 643) 

v_lshl_add_u32 v18, s8, 3, v21 // (op 643) 


v_mul_lo_u32 v21, v19, s26 // (op 643) 
v_add_u32 v21, v21, v18 // (op 643) 


s_mul_i32 s1, s26, s7 // (op 645) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 645) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 645) least significant
s_mul_hi_u32 s9, s26, s6 // (op 645) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 645) most significant: sum
s_add_u32 s1, s1, s9 // (op 645) most significant: sum


s_mov_b32 s60, s28 // (op 645) reg expression
s_mov_b32 s61, s29 // (op 645) reg expression
s_mov_b32 s62, s0 // (op 645) reg expression
s_mov_b32 s63, 131072 // (op 645) call()

s_mul_hi_i32 s0, s16, s12 // (op 707) 
v_lshrrev_b32 v22, 2, v1 // (op 707) 
s_add_i32 s1, s0, s16 // (op 707) 

v_and_b32 v18, -4, v1 // (op 707) 
s_ashr_i32 s0, s1, 31 // (op 707) 
v_sub_u32 v19, v1, v18 // (op 707) 

s_cmp_eq_i32 s12, 0 // (op 707) 
s_mov_b32 s8, scc // (op 707) 
s_cmp_eq_i32 s8, 0 // (op 707) 
s_cselect_b32 s9, 0, -1 // (op 707) 

s_and_b32 s8, s13, 31 // (op 707) 
s_lshl_b32 s18, 1, s8 // (op 707) 
s_add_i32 s19, s18, s9 // (op 707) 


s_and_b32 s9, s0, s19 // (op 707) 


s_add_i32 s0, s1, s9 // (op 707) 


s_ashr_i32 s1, s0, s8 // (op 707) 


s_lshl_b32 s0, s13, 24 // (op 707) 
s_ashr_i32 s8, s0, 31 // (op 707) 

s_xor_b32 s0, s1, s8 // (op 707) 

s_sub_i32 s1, s0, s8 // (op 707) 


s_mul_hi_i32 s0, s1, s14 // (op 707) 
s_add_i32 s8, s0, s1 // (op 707) 

s_ashr_i32 s0, s8, 31 // (op 707) 
s_cmp_eq_i32 s14, 0 // (op 707) 
s_mov_b32 s9, scc // (op 707) 
s_cmp_eq_i32 s9, 0 // (op 707) 
s_cselect_b32 s18, 0, -1 // (op 707) 

s_and_b32 s9, s15, 31 // (op 707) 
s_lshl_b32 s19, 1, s9 // (op 707) 
s_add_i32 s49, s19, s18 // (op 707) 


s_and_b32 s18, s0, s49 // (op 707) 


s_add_i32 s0, s8, s18 // (op 707) 


s_ashr_i32 s8, s0, s9 // (op 707) 


s_lshl_b32 s0, s15, 24 // (op 707) 
s_ashr_i32 s9, s0, 31 // (op 707) 

s_xor_b32 s0, s8, s9 // (op 707) 

s_sub_i32 s8, s0, s9 // (op 707) 


s_add_u32 s18, s4, 16 // (op 707) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 707) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 707) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 707) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 707) 

s_mul_i32 s0, s8, s18 // (op 707) 

s_sub_i32 s8, s1, s0 // (op 707) 


s_mul_hi_i32 s9, s16, s11 // (op 707) 
s_add_i32 s0, s9, s16 // (op 707) 

s_ashr_i32 s1, s0, 31 // (op 707) 
s_cmp_eq_i32 s11, 0 // (op 707) 
s_mov_b32 s9, scc // (op 707) 
s_cmp_eq_i32 s9, 0 // (op 707) 
s_cselect_b32 s49, 0, -1 // (op 707) 

s_and_b32 s9, s46, 31 // (op 707) 
s_lshl_b32 s50, 1, s9 // (op 707) 
s_add_i32 s51, s50, s49 // (op 707) 


s_and_b32 s49, s1, s51 // (op 707) 


s_add_i32 s1, s0, s49 // (op 707) 


s_ashr_i32 s0, s1, s9 // (op 707) 


s_lshl_b32 s1, s46, 24 // (op 707) 
s_ashr_i32 s9, s1, 31 // (op 707) 

s_xor_b32 s1, s0, s9 // (op 707) 

s_sub_i32 s0, s1, s9 // (op 707) 


s_mul_i32 s1, s45, s18 // (op 707) 

s_mul_i32 s9, s10, s1 // (op 707) 

s_mul_i32 s1, s0, s9 // (op 707) 

s_sub_u32 s9, s16, s1 // (op 707) 

s_mul_hi_i32 s1, s9, s44 // (op 707) 
s_add_i32 s18, s1, s9 // (op 707) 


s_ashr_i32 s1, s18, 31 // (op 707) 
s_cmp_eq_i32 s44, 0 // (op 707) 
s_mov_b32 s9, scc // (op 707) 
s_cmp_eq_i32 s9, 0 // (op 707) 
s_cselect_b32 s19, 0, -1 // (op 707) 

s_and_b32 s9, s48, 31 // (op 707) 
s_lshl_b32 s49, 1, s9 // (op 707) 
s_add_i32 s50, s49, s19 // (op 707) 


s_and_b32 s19, s1, s50 // (op 707) 


s_add_i32 s1, s18, s19 // (op 707) 


s_ashr_i32 s18, s1, s9 // (op 707) 


s_lshl_b32 s1, s48, 24 // (op 707) 
s_ashr_i32 s9, s1, 31 // (op 707) 

s_xor_b32 s1, s18, s9 // (op 707) 

s_sub_i32 s18, s1, s9 // (op 707) 


s_cmp_eq_i32 s0, 0 // (op 707) 
s_mov_b32 s1, scc // (op 707) 

s_cmp_eq_i32 s1, 0 // (op 707) 
s_cselect_b32 s0, s18, s8 // (op 707) 



v_lshl_add_u32 v18, s0, 4, v22 // (op 707) 


v_mul_lo_u32 v22, v18, s30 // (op 707) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 707) 
s_lshr_b32 s8, s0, 1 // (op 707) 

v_lshl_add_u32 v18, s8, 3, v19 // (op 707) 


v_add_u32 v19, v18, v22 // (op 707) 


s_mul_i32 s1, s30, s5 // (op 709) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 709) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 709) least significant
s_mul_hi_u32 s9, s30, s4 // (op 709) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 709) most significant: sum
s_add_u32 s1, s1, s9 // (op 709) most significant: sum


s_mov_b32 s64, s32 // (op 709) reg expression
s_mov_b32 s65, s33 // (op 709) reg expression
s_mov_b32 s66, s0 // (op 709) reg expression
s_mov_b32 s67, 131072 // (op 709) call()

v_and_b32 v18, -8, v1 // (op 800) 
v_sub_u32 v22, v1, v18 // (op 800) 

v_lshlrev_b32 v18, 4, v22 // (op 800) 

v_lshrrev_b32 v22, 3, v1 // (op 800) 
v_lshl_add_u32 v24, v22, 7, v18 // (op 800) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 800) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 852) 
v_sub_u32 v22, v1, v18 // (op 852) 

v_lshlrev_b32 v18, 4, v22 // (op 852) 

v_lshrrev_b32 v22, 3, v1 // (op 852) 
v_lshl_add_u32 v24, v22, 7, v18 // (op 852) 


s_nop 0
v_readfirstlane_b32 s1, v24 // (op 852) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -4, v1 // (op 904) 
v_sub_u32 v22, v1, v18 // (op 904) 

v_and_b32 v18, 4294967292, v1 // (op 904) 
v_add_u32 v24, v18, v22 // (op 904) 


v_and_b32 v18, -4, v1 // (op 957) 
v_sub_u32 v22, v1, v18 // (op 957) 

v_and_b32 v18, 4294967292, v1 // (op 957) 
v_add_u32 v25, v18, v22 // (op 957) 


s_ashr_i64 s[8:9], s[34:35], 7 // (op 172) 
s_and_b32 s18, s8, 4294967294 // (op 172) 

s_ashr_i64 s[8:9], s[34:35], 7 // (op 171) 
s_mov_b32 s19, s8 // (op 171) convert

s_cmp_lt_u32 s18, s19 // (op 171) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_171 // (op 171) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_171 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_171:
s_barrier  // (op 1193) 
s_add_u32 s18, s18, 1 // (op 173) 

s_lshl_b64 s[8:9], s[2:3], 3 // (op 464) 
v_mov_b64 v[12:13], s[8:9] // (op 464) convert


v_mov_b64 v[22:23], v[14:15] // (op 464) 
v_mov_b64 v[26:27], v[22:23] // (op 464) 
s_add_u32 s19, 2048, s0 // (op 464) 
s_mov_b32 m0, s19 // (op 464) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 464) Load value direct to lds
v_add_co_u32 v26, s[8:9], v26, v12 // (op 464) least significant half
v_addc_co_u32 v27, s[8:9], 0, v27, s[8:9] // (op 464) most significant half

s_add_u32 m0, m0, 1024 // (op 464) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 464) Load value direct to lds




v_add_co_u32 v14, vcc, 128, v14 // (op 502) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 502) most significant half
s_barrier  // (op 1209) 

s_lshl_b64 s[8:9], s[22:23], 3 // (op 470) 
v_mov_b64 v[12:13], s[8:9] // (op 470) convert


v_mov_b64 v[22:23], v[16:17] // (op 470) 
v_mov_b64 v[26:27], v[22:23] // (op 470) 
s_mov_b32 m0, s1 // (op 470) reg expression
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 470) Load value direct to lds
v_add_co_u32 v26, s[8:9], v26, v12 // (op 470) least significant half
v_addc_co_u32 v27, s[8:9], 0, v27, s[8:9] // (op 470) most significant half

s_add_u32 m0, m0, 1024 // (op 470) 
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 470) Load value direct to lds



v_add_co_u32 v16, vcc, 128, v16 // (op 571) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 571) most significant half


v_mov_b32 v18, v21 // (op 191) 

buffer_load_ubyte v20, v18, s[60:63], 0 offen offset: 0 // (op 191) Load value

v_add_u32 v21, 4, v21 // (op 635) 


s_barrier  // (op 1226) 
v_mov_b32 v18, v19 // (op 192) 




buffer_load_ubyte v12, v18, s[64:67], 0 offen offset: 0 // (op 192) Load value

v_add_u32 v19, 4, v19 // (op 699) 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v20 offset:8256 // (op 188) Store local data 

s_barrier  // (op 1199) 



s_barrier  // (op 1243) 
ds_read_b128 v[26:29], v3 offset:2048 // (op 179) Load local data 



ds_read_b128 v[30:33], v3 offset:2112 // (op 179) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v25, v12 offset:8192 // (op 189) Store local data 

s_barrier  // (op 1216) 



ds_read_b128 v[34:37], v5 // (op 181) Load local data 
ds_read_b128 v[38:41], v5 offset:64 // (op 181) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1233) 



ds_read_u8 v18, v0 offset:8256 // (op 183) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1250) 



ds_read_u8 v20, v6 offset:8192 // (op 185) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v18, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 186) 




s_ashr_i64 s[8:9], s[34:35], 7 // (op 171) 
s_mov_b32 s19, s8 // (op 171) convert

s_cmp_lt_u32 s18, s19 // (op 171) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_171 // (op 171) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_171 if true)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_171:






















s_barrier  // (op 1313) 








v_cmp_eq_f32 vcc, 0.00000, v8 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_133
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v11, v4 // (op 50) 
v_mul_f32 v13, v11, v5 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 

s_mul_hi_i32 s0, s16, s12 // (op 999) 
v_lshrrev_b32 v0, 6, v1 // (op 999) 
s_add_i32 s1, s0, s16 // (op 999) 

v_and_b32 v3, 63, v1 // (op 999) 
s_ashr_i32 s0, s1, 31 // (op 999) 
v_lshrrev_b32 v4, 4, v3 // (op 999) 
s_cmp_eq_i32 s12, 0 // (op 999) 
s_mov_b32 s2, scc // (op 999) 
v_lshlrev_b32 v5, 2, v4 // (op 999) 

s_cmp_eq_i32 s2, 0 // (op 999) 
s_cselect_b32 s3, 0, -1 // (op 999) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 999) 


s_and_b32 s2, s13, 31 // (op 999) 
v_lshrrev_b32 v0, 2, v3 // (op 999) 

s_lshl_b32 s8, 1, s2 // (op 999) 
v_and_b32 v3, 3, v0 // (op 999) 

s_add_i32 s9, s8, s3 // (op 999) 


v_and_b32 v0, 3, v1 // (op 999) 
s_and_b32 s3, s0, s9 // (op 999) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 999) 


s_add_i32 s0, s1, s3 // (op 999) 


s_ashr_i32 s1, s0, s2 // (op 999) 


s_lshl_b32 s0, s13, 24 // (op 999) 
s_ashr_i32 s2, s0, 31 // (op 999) 

s_xor_b32 s0, s1, s2 // (op 999) 

s_sub_i32 s1, s0, s2 // (op 999) 


s_mul_hi_i32 s0, s1, s14 // (op 999) 
s_add_i32 s2, s0, s1 // (op 999) 

s_ashr_i32 s0, s2, 31 // (op 999) 
s_cmp_eq_i32 s14, 0 // (op 999) 
s_mov_b32 s3, scc // (op 999) 
s_cmp_eq_i32 s3, 0 // (op 999) 
s_cselect_b32 s8, 0, -1 // (op 999) 

s_and_b32 s3, s15, 31 // (op 999) 
s_lshl_b32 s9, 1, s3 // (op 999) 
s_add_i32 s18, s9, s8 // (op 999) 


s_and_b32 s8, s0, s18 // (op 999) 


s_add_i32 s0, s2, s8 // (op 999) 


s_ashr_i32 s2, s0, s3 // (op 999) 


s_lshl_b32 s0, s15, 24 // (op 999) 
s_ashr_i32 s3, s0, 31 // (op 999) 

s_xor_b32 s0, s2, s3 // (op 999) 

s_sub_i32 s2, s0, s3 // (op 999) 


s_mul_i32 s0, s1, s45 // (op 999) 
s_sub_u32 s3, s16, s0 // (op 999) 

s_mul_i32 s0, s2, s45 // (op 999) 
s_add_u32 s0, s0, s3 // (op 999) 

s_mul_hi_i32 s3, s16, s11 // (op 999) 
s_add_i32 s8, s3, s16 // (op 999) 

s_ashr_i32 s3, s8, 31 // (op 999) 
s_cmp_eq_i32 s11, 0 // (op 999) 
s_mov_b32 s9, scc // (op 999) 
s_cmp_eq_i32 s9, 0 // (op 999) 
s_cselect_b32 s18, 0, -1 // (op 999) 

s_and_b32 s9, s46, 31 // (op 999) 
s_lshl_b32 s19, 1, s9 // (op 999) 
s_add_i32 s20, s19, s18 // (op 999) 


s_and_b32 s18, s3, s20 // (op 999) 


s_add_i32 s3, s8, s18 // (op 999) 


s_ashr_i32 s8, s3, s9 // (op 999) 


s_lshl_b32 s3, s46, 24 // (op 999) 
s_ashr_i32 s9, s3, 31 // (op 999) 

s_xor_b32 s3, s8, s9 // (op 999) 

s_sub_i32 s8, s3, s9 // (op 999) 


s_add_u32 s18, s4, 16 // (op 999) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 999) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 999) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 999) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 999) 

s_mul_i32 s3, s45, s18 // (op 999) 
s_mul_i32 s9, s10, s3 // (op 999) 

s_mul_i32 s3, s8, s9 // (op 999) 

s_sub_u32 s9, s16, s3 // (op 999) 

s_mov_b32 s3, s9 // (op 999) 
s_mul_hi_i32 s20, s3, s44 // (op 999) 
s_add_i32 s21, s20, s3 // (op 999) 


s_ashr_i32 s3, s21, 31 // (op 999) 
s_cmp_eq_i32 s44, 0 // (op 999) 
s_mov_b32 s20, scc // (op 999) 
s_cmp_eq_i32 s20, 0 // (op 999) 
s_cselect_b32 s22, 0, -1 // (op 999) 

s_and_b32 s20, s48, 31 // (op 999) 
s_lshl_b32 s23, 1, s20 // (op 999) 
s_add_i32 s24, s23, s22 // (op 999) 


s_and_b32 s22, s3, s24 // (op 999) 


s_add_i32 s3, s21, s22 // (op 999) 


s_ashr_i32 s21, s3, s20 // (op 999) 


s_lshl_b32 s3, s48, 24 // (op 999) 
s_ashr_i32 s20, s3, 31 // (op 999) 

s_xor_b32 s3, s21, s20 // (op 999) 

s_sub_i32 s21, s3, s20 // (op 999) 


s_mul_i32 s3, s47, s45 // (op 999) 
s_add_u32 s22, s6, 16 // (op 999) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 999) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 999) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 999) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 999) 

s_sub_i32 s20, s22, s3 // (op 999) 

s_mul_i32 s22, s21, s20 // (op 999) 

s_sub_u32 s20, s9, s22 // (op 999) 


s_add_u32 s9, s20, s3 // (op 999) 


s_cmp_eq_i32 s8, 0 // (op 999) 
s_mov_b32 s3, scc // (op 999) 

s_cmp_eq_i32 s3, 0 // (op 999) 
s_cselect_b32 s8, s9, s0 // (op 999) 


s_mul_i32 s0, s2, s18 // (op 999) 


v_lshl_add_u32 v0, s8, 4, v4 // (op 999) 


s_sub_i32 s2, s1, s0 // (op 999) 


s_cmp_eq_i32 s3, 0 // (op 999) 
s_cselect_b32 s0, s21, s2 // (op 999) 



v_lshl_add_u32 v3, s0, 4, v5 // (op 999) 


v_mul_lo_u32 v4, v3, s36 // (op 999) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 999) 


s_mul_i32 s9, s36, s5 // (op 1001) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 1001) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 1001) least significant
s_mul_hi_u32 s1, s36, s4 // (op 1001) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 1001) most significant: sum
s_add_u32 s9, s9, s1 // (op 1001) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 1001) 

s_mov_b32 s32, s38 // (op 1001) reg expression
s_mov_b32 s33, s39 // (op 1001) reg expression
s_mov_b32 s34, s0 // (op 1001) reg expression
s_mov_b32 s35, 131072 // (op 1001) call()




buffer_store_dwordx4 v[12:15], v3, s[32:35], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_133
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_133:
s_mul_hi_i32 s0, s16, s12 // (op 770) 
v_lshrrev_b32 v0, 6, v1 // (op 770) 
s_add_i32 s1, s0, s16 // (op 770) 

v_and_b32 v3, 63, v1 // (op 770) 
s_ashr_i32 s0, s1, 31 // (op 770) 
v_lshrrev_b32 v4, 4, v3 // (op 770) 
s_cmp_eq_i32 s12, 0 // (op 770) 
s_mov_b32 s2, scc // (op 770) 
v_lshlrev_b32 v5, 2, v4 // (op 770) 

s_cmp_eq_i32 s2, 0 // (op 770) 
s_cselect_b32 s3, 0, -1 // (op 770) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 770) 


s_and_b32 s2, s13, 31 // (op 770) 
v_lshrrev_b32 v0, 2, v3 // (op 770) 

s_lshl_b32 s8, 1, s2 // (op 770) 
v_and_b32 v3, 3, v0 // (op 770) 

s_add_i32 s9, s8, s3 // (op 770) 


v_and_b32 v0, 3, v1 // (op 770) 
s_and_b32 s3, s0, s9 // (op 770) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 770) 


s_add_i32 s0, s1, s3 // (op 770) 


s_ashr_i32 s1, s0, s2 // (op 770) 


s_lshl_b32 s0, s13, 24 // (op 770) 
s_ashr_i32 s2, s0, 31 // (op 770) 

s_xor_b32 s0, s1, s2 // (op 770) 

s_sub_i32 s1, s0, s2 // (op 770) 


s_mul_hi_i32 s0, s1, s14 // (op 770) 
s_add_i32 s2, s0, s1 // (op 770) 

s_ashr_i32 s0, s2, 31 // (op 770) 
s_cmp_eq_i32 s14, 0 // (op 770) 
s_mov_b32 s3, scc // (op 770) 
s_cmp_eq_i32 s3, 0 // (op 770) 
s_cselect_b32 s8, 0, -1 // (op 770) 

s_and_b32 s3, s15, 31 // (op 770) 
s_lshl_b32 s9, 1, s3 // (op 770) 
s_add_i32 s18, s9, s8 // (op 770) 


s_and_b32 s8, s0, s18 // (op 770) 


s_add_i32 s0, s2, s8 // (op 770) 


s_ashr_i32 s2, s0, s3 // (op 770) 


s_lshl_b32 s0, s15, 24 // (op 770) 
s_ashr_i32 s3, s0, 31 // (op 770) 

s_xor_b32 s0, s2, s3 // (op 770) 

s_sub_i32 s2, s0, s3 // (op 770) 


s_mul_i32 s0, s1, s45 // (op 770) 
s_sub_u32 s3, s16, s0 // (op 770) 

s_mul_i32 s0, s2, s45 // (op 770) 
s_add_u32 s0, s0, s3 // (op 770) 

s_mul_hi_i32 s3, s16, s11 // (op 770) 
s_add_i32 s8, s3, s16 // (op 770) 

s_ashr_i32 s3, s8, 31 // (op 770) 
s_cmp_eq_i32 s11, 0 // (op 770) 
s_mov_b32 s9, scc // (op 770) 
s_cmp_eq_i32 s9, 0 // (op 770) 
s_cselect_b32 s18, 0, -1 // (op 770) 

s_and_b32 s9, s46, 31 // (op 770) 
s_lshl_b32 s19, 1, s9 // (op 770) 
s_add_i32 s20, s19, s18 // (op 770) 


s_and_b32 s18, s3, s20 // (op 770) 


s_add_i32 s3, s8, s18 // (op 770) 


s_ashr_i32 s8, s3, s9 // (op 770) 


s_lshl_b32 s3, s46, 24 // (op 770) 
s_ashr_i32 s9, s3, 31 // (op 770) 

s_xor_b32 s3, s8, s9 // (op 770) 

s_sub_i32 s8, s3, s9 // (op 770) 


s_add_u32 s18, s4, 16 // (op 770) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 770) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 770) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 770) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 770) 

s_mul_i32 s3, s45, s18 // (op 770) 
s_mul_i32 s9, s10, s3 // (op 770) 

s_mul_i32 s3, s8, s9 // (op 770) 

s_sub_u32 s9, s16, s3 // (op 770) 

s_mov_b32 s3, s9 // (op 770) 
s_mul_hi_i32 s20, s3, s44 // (op 770) 
s_add_i32 s21, s20, s3 // (op 770) 


s_ashr_i32 s3, s21, 31 // (op 770) 
s_cmp_eq_i32 s44, 0 // (op 770) 
s_mov_b32 s20, scc // (op 770) 
s_cmp_eq_i32 s20, 0 // (op 770) 
s_cselect_b32 s22, 0, -1 // (op 770) 

s_and_b32 s20, s48, 31 // (op 770) 
s_lshl_b32 s23, 1, s20 // (op 770) 
s_add_i32 s24, s23, s22 // (op 770) 


s_and_b32 s22, s3, s24 // (op 770) 


s_add_i32 s3, s21, s22 // (op 770) 


s_ashr_i32 s21, s3, s20 // (op 770) 


s_lshl_b32 s3, s48, 24 // (op 770) 
s_ashr_i32 s20, s3, 31 // (op 770) 

s_xor_b32 s3, s21, s20 // (op 770) 

s_sub_i32 s21, s3, s20 // (op 770) 


s_mul_i32 s3, s47, s45 // (op 770) 
s_add_u32 s22, s6, 16 // (op 770) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 770) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 770) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 770) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 770) 

s_sub_i32 s20, s22, s3 // (op 770) 

s_mul_i32 s22, s21, s20 // (op 770) 

s_sub_u32 s20, s9, s22 // (op 770) 


s_add_u32 s9, s20, s3 // (op 770) 


s_cmp_eq_i32 s8, 0 // (op 770) 
s_mov_b32 s3, scc // (op 770) 

s_cmp_eq_i32 s3, 0 // (op 770) 
s_cselect_b32 s8, s9, s0 // (op 770) 


s_mul_i32 s0, s2, s18 // (op 770) 


v_lshl_add_u32 v0, s8, 4, v4 // (op 770) 


s_sub_i32 s2, s1, s0 // (op 770) 


s_cmp_eq_i32 s3, 0 // (op 770) 
s_cselect_b32 s0, s21, s2 // (op 770) 



v_lshl_add_u32 v3, s0, 4, v5 // (op 770) 


v_mul_lo_u32 v4, v3, s40 // (op 770) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 770) 


s_mul_i32 s9, s40, s5 // (op 772) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 772) most significant: low of high * low
s_mul_i32 s8, s40, s4 // (op 772) least significant
s_mul_hi_u32 s1, s40, s4 // (op 772) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 772) most significant: sum
s_add_u32 s9, s9, s1 // (op 772) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 772) 

s_mov_b32 s28, s42 // (op 772) reg expression
s_mov_b32 s29, s43 // (op 772) reg expression
s_mov_b32 s30, s0 // (op 772) reg expression
s_mov_b32 s31, 131072 // (op 772) call()





buffer_load_dwordx4 v[4:7], v3, s[28:31], 0 offen // (op 145) Load value



v_accvgpr_read v12, a0 // (op 142) 
v_accvgpr_read v13, a1 // (op 142) 
v_accvgpr_read v14, a2 // (op 142) 
v_accvgpr_read v15, a3 // (op 142) 
v_mul_f32 v16, v11, v12 // (op 142) 
v_mul_f32 v17, v11, v13 // (op 142) 
v_mul_f32 v18, v11, v14 // (op 142) 
v_mul_f32 v19, v11, v15 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v8, v4 // (op 146) 
v_mul_f32 v13, v8, v5 // (op 146) 
v_mul_f32 v14, v8, v6 // (op 146) 
v_mul_f32 v15, v8, v7 // (op 146) 

v_add_f32 v4, v12, v16 // (op 143) 
v_add_f32 v5, v13, v17 // (op 143) 
v_add_f32 v6, v14, v18 // (op 143) 
v_add_f32 v7, v15, v19 // (op 143) 


s_mul_hi_i32 s0, s16, s12 // (op 1016) 
v_lshrrev_b32 v0, 6, v1 // (op 1016) 
s_add_i32 s1, s0, s16 // (op 1016) 

v_and_b32 v3, 63, v1 // (op 1016) 
s_ashr_i32 s0, s1, 31 // (op 1016) 
v_lshrrev_b32 v9, 4, v3 // (op 1016) 
s_cmp_eq_i32 s12, 0 // (op 1016) 
s_mov_b32 s2, scc // (op 1016) 
v_lshlrev_b32 v10, 2, v9 // (op 1016) 

s_cmp_eq_i32 s2, 0 // (op 1016) 
s_cselect_b32 s3, 0, -1 // (op 1016) 

v_lshl_add_u32 v9, v0, 4, v10 // (op 1016) 


s_and_b32 s2, s13, 31 // (op 1016) 
v_lshrrev_b32 v0, 2, v3 // (op 1016) 

s_lshl_b32 s8, 1, s2 // (op 1016) 
v_and_b32 v3, 3, v0 // (op 1016) 

s_add_i32 s9, s8, s3 // (op 1016) 


v_and_b32 v0, 3, v1 // (op 1016) 
s_and_b32 s3, s0, s9 // (op 1016) 


v_lshl_add_u32 v10, v3, 2, v0 // (op 1016) 


s_add_i32 s0, s1, s3 // (op 1016) 


s_ashr_i32 s1, s0, s2 // (op 1016) 


s_lshl_b32 s0, s13, 24 // (op 1016) 
s_ashr_i32 s2, s0, 31 // (op 1016) 

s_xor_b32 s0, s1, s2 // (op 1016) 

s_sub_i32 s1, s0, s2 // (op 1016) 


s_mul_hi_i32 s0, s1, s14 // (op 1016) 
s_add_i32 s2, s0, s1 // (op 1016) 

s_ashr_i32 s0, s2, 31 // (op 1016) 
s_cmp_eq_i32 s14, 0 // (op 1016) 
s_mov_b32 s3, scc // (op 1016) 
s_cmp_eq_i32 s3, 0 // (op 1016) 
s_cselect_b32 s8, 0, -1 // (op 1016) 

s_and_b32 s3, s15, 31 // (op 1016) 
s_lshl_b32 s9, 1, s3 // (op 1016) 
s_add_i32 s18, s9, s8 // (op 1016) 


s_and_b32 s8, s0, s18 // (op 1016) 


s_add_i32 s0, s2, s8 // (op 1016) 


s_ashr_i32 s2, s0, s3 // (op 1016) 


s_lshl_b32 s0, s15, 24 // (op 1016) 
s_ashr_i32 s3, s0, 31 // (op 1016) 

s_xor_b32 s0, s2, s3 // (op 1016) 

s_sub_i32 s2, s0, s3 // (op 1016) 


s_mul_i32 s0, s1, s45 // (op 1016) 
s_sub_u32 s3, s16, s0 // (op 1016) 

s_mul_i32 s0, s2, s45 // (op 1016) 
s_add_u32 s0, s0, s3 // (op 1016) 

s_mul_hi_i32 s3, s16, s11 // (op 1016) 
s_add_i32 s8, s3, s16 // (op 1016) 

s_ashr_i32 s3, s8, 31 // (op 1016) 
s_cmp_eq_i32 s11, 0 // (op 1016) 
s_mov_b32 s9, scc // (op 1016) 
s_cmp_eq_i32 s9, 0 // (op 1016) 
s_cselect_b32 s18, 0, -1 // (op 1016) 

s_and_b32 s9, s46, 31 // (op 1016) 
s_lshl_b32 s19, 1, s9 // (op 1016) 
s_add_i32 s20, s19, s18 // (op 1016) 


s_and_b32 s18, s3, s20 // (op 1016) 


s_add_i32 s3, s8, s18 // (op 1016) 


s_ashr_i32 s8, s3, s9 // (op 1016) 


s_lshl_b32 s3, s46, 24 // (op 1016) 
s_ashr_i32 s9, s3, 31 // (op 1016) 

s_xor_b32 s3, s8, s9 // (op 1016) 

s_sub_i32 s8, s3, s9 // (op 1016) 


s_add_u32 s18, s4, 16 // (op 1016) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 1016) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 1016) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 1016) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 1016) 

s_mul_i32 s3, s45, s18 // (op 1016) 
s_mul_i32 s9, s10, s3 // (op 1016) 

s_mul_i32 s3, s8, s9 // (op 1016) 

s_sub_u32 s9, s16, s3 // (op 1016) 

s_mov_b32 s3, s9 // (op 1016) 
s_mul_hi_i32 s20, s3, s44 // (op 1016) 
s_add_i32 s21, s20, s3 // (op 1016) 


s_ashr_i32 s3, s21, 31 // (op 1016) 
s_cmp_eq_i32 s44, 0 // (op 1016) 
s_mov_b32 s20, scc // (op 1016) 
s_cmp_eq_i32 s20, 0 // (op 1016) 
s_cselect_b32 s22, 0, -1 // (op 1016) 

s_and_b32 s20, s48, 31 // (op 1016) 
s_lshl_b32 s23, 1, s20 // (op 1016) 
s_add_i32 s24, s23, s22 // (op 1016) 


s_and_b32 s22, s3, s24 // (op 1016) 


s_add_i32 s3, s21, s22 // (op 1016) 


s_ashr_i32 s21, s3, s20 // (op 1016) 


s_lshl_b32 s3, s48, 24 // (op 1016) 
s_ashr_i32 s20, s3, 31 // (op 1016) 

s_xor_b32 s3, s21, s20 // (op 1016) 

s_sub_i32 s21, s3, s20 // (op 1016) 


s_mul_i32 s3, s47, s45 // (op 1016) 
s_add_u32 s22, s6, 16 // (op 1016) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1016) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1016) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1016) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1016) 

s_sub_i32 s20, s22, s3 // (op 1016) 

s_mul_i32 s22, s21, s20 // (op 1016) 

s_sub_u32 s20, s9, s22 // (op 1016) 


s_add_u32 s9, s20, s3 // (op 1016) 


s_cmp_eq_i32 s8, 0 // (op 1016) 
s_mov_b32 s3, scc // (op 1016) 

s_cmp_eq_i32 s3, 0 // (op 1016) 
s_cselect_b32 s8, s9, s0 // (op 1016) 


s_mul_i32 s0, s2, s18 // (op 1016) 


v_lshl_add_u32 v0, s8, 4, v9 // (op 1016) 


s_sub_i32 s2, s1, s0 // (op 1016) 


s_cmp_eq_i32 s3, 0 // (op 1016) 
s_cselect_b32 s0, s21, s2 // (op 1016) 



v_lshl_add_u32 v3, s0, 4, v10 // (op 1016) 


v_mul_lo_u32 v9, v3, s36 // (op 1016) 

v_add_lshl_u32 v3, v0, v9, 2 // (op 1016) 


s_mul_i32 s9, s36, s5 // (op 1018) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 1018) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 1018) least significant
s_mul_hi_u32 s1, s36, s4 // (op 1018) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 1018) most significant: sum
s_add_u32 s9, s9, s1 // (op 1018) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 1018) 

s_mov_b32 s32, s38 // (op 1018) reg expression
s_mov_b32 s33, s39 // (op 1018) reg expression
s_mov_b32 s34, s0 // (op 1018) reg expression
s_mov_b32 s35, 131072 // (op 1018) call()




buffer_store_dwordx4 v[4:7], v3, s[32:35], 0 offen // (op 144) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_133:
















s_endpgm  // End of RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end-RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 60
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 8448 // lds bytes
  .amdhsa_accum_offset 56
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
  ProblemType:
    OperationType: GEMM
    DataType: F8
    DataTypeA: F8
    DataTypeB: B8
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
    MXBlockA: 32
    MXBlockB: 32
  CustomKernel:
    args: [ { type: int64, semantic: StrideA0 },
            { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: SizeFree0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: address, semantic: AddressA },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail } ]
    macrotile: [16, 16, 128]
    threads: [64, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 8448
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 70
    .vgpr_count: 54
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .workgroup_size: [64, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, {is-null: true}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 24
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 28
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 32
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 36
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 40
        .expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 44
        .expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 48
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 128
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 136
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 144
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: MagicMultiple_30
        .size: 4
        .offset: 152
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 156
        .expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 160
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 164
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 168
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

