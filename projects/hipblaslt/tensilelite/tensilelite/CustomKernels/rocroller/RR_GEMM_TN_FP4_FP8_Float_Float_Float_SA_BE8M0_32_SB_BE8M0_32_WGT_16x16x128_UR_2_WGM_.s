// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_,@function
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 765) 
v_and_b32 v3, 15, v1 // (op 765) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 765) 


v_and_b32 v0, 63, v1 // (op 765) 
v_lshrrev_b32 v3, 4, v0 // (op 765) 

v_lshlrev_b32 v0, 5, v3 // (op 765) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 765) 


v_lshlrev_b32 v0, 2, v3 // (op 765) 

v_lshrrev_b32 v3, 3, v0 // (op 765) 

v_and_b32 v0, 15, v1 // (op 818) 
v_and_b32 v4, 48, v1 // (op 818) 
v_lshl_add_u32 v5, v0, 7, v4 // (op 818) 


v_lshrrev_b32 v0, 6, v1 // (op 871) 
v_and_b32 v4, 15, v1 // (op 871) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 871) 


v_and_b32 v0, 63, v1 // (op 871) 
v_lshrrev_b32 v4, 4, v0 // (op 871) 

v_lshl_add_u32 v0, v6, 2, v4 // (op 871) 


v_and_b32 v4, 15, v1 // (op 924) 
v_and_b32 v6, 63, v1 // (op 924) 
v_lshrrev_b32 v7, 4, v6 // (op 924) 

v_lshl_add_u32 v6, v4, 2, v7 // (op 924) 


v_lshrrev_b32 v4, 6, v1 // (op 982) 
v_and_b32 v7, 15, v1 // (op 982) 
v_lshl_add_u32 v8, v4, 4, v7 // (op 982) 


v_and_b32 v4, 63, v1 // (op 982) 
v_lshrrev_b32 v7, 4, v4 // (op 982) 

v_lshlrev_b32 v4, 5, v7 // (op 982) 

v_lshl_add_u32 v7, v8, 7, v4 // (op 982) 


v_lshlrev_b32 v4, 2, v7 // (op 982) 

v_lshrrev_b32 v7, 3, v4 // (op 982) 

v_and_b32 v4, 15, v1 // (op 1006) 
v_and_b32 v8, 48, v1 // (op 1006) 
v_lshl_add_u32 v9, v4, 7, v8 // (op 1006) 


v_lshrrev_b32 v4, 6, v1 // (op 1030) 
v_and_b32 v8, 15, v1 // (op 1030) 
v_lshl_add_u32 v10, v4, 4, v8 // (op 1030) 


v_and_b32 v4, 63, v1 // (op 1030) 
v_lshrrev_b32 v8, 4, v4 // (op 1030) 

v_lshl_add_u32 v4, v10, 2, v8 // (op 1030) 


v_and_b32 v8, 15, v1 // (op 1054) 
v_and_b32 v10, 63, v1 // (op 1054) 
v_lshrrev_b32 v11, 4, v10 // (op 1054) 

v_lshl_add_u32 v10, v8, 2, v11 // (op 1054) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v11, s9 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_mul_hi_i32 s0, s16, s12 // (op 475) 
v_lshrrev_b32 v12, 2, v1 // (op 475) 
s_add_i32 s1, s0, s16 // (op 475) 

v_and_b32 v13, -4, v1 // (op 475) 
s_ashr_i32 s0, s1, 31 // (op 475) 
v_sub_u32 v14, v1, v13 // (op 475) 

s_cmp_eq_i32 s12, 0 // (op 475) 
s_mov_b32 s8, scc // (op 475) 
v_lshlrev_b32 v13, 5, v14 // (op 475) 

s_cmp_eq_i32 s8, 0 // (op 475) 
s_cselect_b32 s9, 0, -1 // (op 475) 

v_mov_b32 v15, 0 // (op 475) convert
v_mov_b32 v14, v13 // (op 475) convert

s_and_b32 s8, s13, 31 // (op 475) 
s_lshl_b32 s18, 1, s8 // (op 475) 
s_add_i32 s19, s18, s9 // (op 475) 


s_and_b32 s9, s0, s19 // (op 475) 


s_add_i32 s0, s1, s9 // (op 475) 


s_ashr_i32 s1, s0, s8 // (op 475) 


s_lshl_b32 s0, s13, 24 // (op 475) 
s_ashr_i32 s8, s0, 31 // (op 475) 

s_xor_b32 s0, s1, s8 // (op 475) 

s_sub_i32 s1, s0, s8 // (op 475) 


s_mul_hi_i32 s0, s1, s14 // (op 475) 
s_add_i32 s8, s0, s1 // (op 475) 

s_ashr_i32 s0, s8, 31 // (op 475) 
s_cmp_eq_i32 s14, 0 // (op 475) 
s_mov_b32 s9, scc // (op 475) 
s_cmp_eq_i32 s9, 0 // (op 475) 
s_cselect_b32 s18, 0, -1 // (op 475) 

s_and_b32 s9, s15, 31 // (op 475) 
s_lshl_b32 s19, 1, s9 // (op 475) 
s_add_i32 s49, s19, s18 // (op 475) 


s_and_b32 s18, s0, s49 // (op 475) 


s_add_i32 s0, s8, s18 // (op 475) 


s_ashr_i32 s8, s0, s9 // (op 475) 


s_lshl_b32 s0, s15, 24 // (op 475) 
s_ashr_i32 s9, s0, 31 // (op 475) 

s_xor_b32 s0, s8, s9 // (op 475) 

s_sub_i32 s8, s0, s9 // (op 475) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s45 // (op 475) 

s_sub_u32 s1, s16, s0 // (op 475) 

s_mul_i32 s0, s8, s45 // (op 475) 
s_add_u32 s0, s0, s1 // (op 475) 


s_mul_hi_i32 s1, s16, s11 // (op 475) 
s_add_i32 s8, s1, s16 // (op 475) 

s_ashr_i32 s1, s8, 31 // (op 475) 
s_cmp_eq_i32 s11, 0 // (op 475) 
s_mov_b32 s9, scc // (op 475) 
s_cmp_eq_i32 s9, 0 // (op 475) 
s_cselect_b32 s18, 0, -1 // (op 475) 

s_and_b32 s9, s46, 31 // (op 475) 
s_lshl_b32 s19, 1, s9 // (op 475) 
s_add_i32 s49, s19, s18 // (op 475) 


s_and_b32 s18, s1, s49 // (op 475) 


s_add_i32 s1, s8, s18 // (op 475) 


s_ashr_i32 s8, s1, s9 // (op 475) 


s_lshl_b32 s1, s46, 24 // (op 475) 
s_ashr_i32 s9, s1, 31 // (op 475) 

s_xor_b32 s1, s8, s9 // (op 475) 

s_sub_i32 s8, s1, s9 // (op 475) 


s_add_u32 s18, s4, 16 // (op 475) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 475) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 475) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 475) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 475) 

s_mul_i32 s1, s45, s18 // (op 475) 

s_mul_i32 s9, s10, s1 // (op 475) 

s_mul_i32 s1, s8, s9 // (op 475) 

s_sub_u32 s9, s16, s1 // (op 475) 

s_mov_b32 s1, s9 // (op 475) 
s_mul_hi_i32 s18, s1, s44 // (op 475) 
s_add_i32 s19, s18, s1 // (op 475) 


s_ashr_i32 s1, s19, 31 // (op 475) 
s_cmp_eq_i32 s44, 0 // (op 475) 
s_mov_b32 s18, scc // (op 475) 
s_cmp_eq_i32 s18, 0 // (op 475) 
s_cselect_b32 s49, 0, -1 // (op 475) 

s_and_b32 s18, s48, 31 // (op 475) 
s_lshl_b32 s50, 1, s18 // (op 475) 
s_add_i32 s51, s50, s49 // (op 475) 


s_and_b32 s49, s1, s51 // (op 475) 


s_add_i32 s1, s19, s49 // (op 475) 


s_ashr_i32 s19, s1, s18 // (op 475) 


s_lshl_b32 s1, s48, 24 // (op 475) 
s_ashr_i32 s18, s1, 31 // (op 475) 

s_xor_b32 s1, s19, s18 // (op 475) 

s_sub_i32 s19, s1, s18 // (op 475) 


s_mul_i32 s1, s47, s45 // (op 475) 
s_add_u32 s50, s6, 16 // (op 475) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 475) End of Int64 add, unlocking SCC
s_sub_u32 s52, s50, 1 // (op 475) least significant half; sets scc
s_subb_u32 s53, s51, 0 // (op 475) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 4 // (op 475) 

s_sub_i32 s18, s50, s1 // (op 475) 

s_mul_i32 s49, s19, s18 // (op 475) 


s_sub_u32 s18, s9, s49 // (op 475) 


s_add_u32 s9, s18, s1 // (op 475) 


s_cmp_eq_i32 s8, 0 // (op 475) 
s_mov_b32 s1, scc // (op 475) 

s_cmp_eq_i32 s1, 0 // (op 475) 
s_cselect_b32 s8, s9, s0 // (op 475) 



v_lshl_add_u32 v13, s8, 4, v12 // (op 475) 


v_mul_lo_u32 v17, v13, s3 // (op 475) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 475) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 475) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 475) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 475) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 475) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 475) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 475) 

v_mov_b32 v12, v14 // (op 475) convert

s_mul_i32 s1, s2, s7 // (op 477) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 477) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 477) least significant
s_mul_hi_u32 s9, s2, s6 // (op 477) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 477) most significant: sum
s_add_u32 s1, s1, s9 // (op 477) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 477) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 477) 

s_mov_b32 s52, s20 // (op 477) reg expression
s_mov_b32 s53, s21 // (op 477) reg expression
s_mov_b32 s54, s0 // (op 477) reg expression
s_mov_b32 s55, 131072 // (op 477) call()

s_mul_hi_i32 s0, s16, s12 // (op 494) 
v_lshrrev_b32 v13, 2, v1 // (op 494) 
s_add_i32 s1, s0, s16 // (op 494) 

v_and_b32 v14, -4, v1 // (op 494) 
s_ashr_i32 s0, s1, 31 // (op 494) 
v_sub_u32 v15, v1, v14 // (op 494) 

s_cmp_eq_i32 s12, 0 // (op 494) 
s_mov_b32 s8, scc // (op 494) 
v_lshlrev_b32 v14, 5, v15 // (op 494) 

s_cmp_eq_i32 s8, 0 // (op 494) 
s_cselect_b32 s9, 0, -1 // (op 494) 

v_mov_b32 v17, 0 // (op 494) convert
v_mov_b32 v16, v14 // (op 494) convert

s_and_b32 s8, s13, 31 // (op 494) 
s_lshl_b32 s18, 1, s8 // (op 494) 
s_add_i32 s19, s18, s9 // (op 494) 


s_and_b32 s9, s0, s19 // (op 494) 


s_add_i32 s0, s1, s9 // (op 494) 


s_ashr_i32 s1, s0, s8 // (op 494) 


s_lshl_b32 s0, s13, 24 // (op 494) 
s_ashr_i32 s8, s0, 31 // (op 494) 

s_xor_b32 s0, s1, s8 // (op 494) 

s_sub_i32 s1, s0, s8 // (op 494) 


s_mul_hi_i32 s0, s1, s14 // (op 494) 
s_add_i32 s8, s0, s1 // (op 494) 

s_ashr_i32 s0, s8, 31 // (op 494) 
s_cmp_eq_i32 s14, 0 // (op 494) 
s_mov_b32 s9, scc // (op 494) 
s_cmp_eq_i32 s9, 0 // (op 494) 
s_cselect_b32 s18, 0, -1 // (op 494) 

s_and_b32 s9, s15, 31 // (op 494) 
s_lshl_b32 s19, 1, s9 // (op 494) 
s_add_i32 s49, s19, s18 // (op 494) 


s_and_b32 s18, s0, s49 // (op 494) 


s_add_i32 s0, s8, s18 // (op 494) 


s_ashr_i32 s8, s0, s9 // (op 494) 


s_lshl_b32 s0, s15, 24 // (op 494) 
s_ashr_i32 s9, s0, 31 // (op 494) 

s_xor_b32 s0, s8, s9 // (op 494) 

s_sub_i32 s8, s0, s9 // (op 494) 


s_mul_i32 s0, s1, s45 // (op 494) 

s_sub_u32 s1, s16, s0 // (op 494) 

s_mul_i32 s0, s8, s45 // (op 494) 
s_add_u32 s0, s0, s1 // (op 494) 


s_mul_hi_i32 s1, s16, s11 // (op 494) 
s_add_i32 s8, s1, s16 // (op 494) 

s_ashr_i32 s1, s8, 31 // (op 494) 
s_cmp_eq_i32 s11, 0 // (op 494) 
s_mov_b32 s9, scc // (op 494) 
s_cmp_eq_i32 s9, 0 // (op 494) 
s_cselect_b32 s18, 0, -1 // (op 494) 

s_and_b32 s9, s46, 31 // (op 494) 
s_lshl_b32 s19, 1, s9 // (op 494) 
s_add_i32 s49, s19, s18 // (op 494) 


s_and_b32 s18, s1, s49 // (op 494) 


s_add_i32 s1, s8, s18 // (op 494) 


s_ashr_i32 s8, s1, s9 // (op 494) 


s_lshl_b32 s1, s46, 24 // (op 494) 
s_ashr_i32 s9, s1, 31 // (op 494) 

s_xor_b32 s1, s8, s9 // (op 494) 

s_sub_i32 s8, s1, s9 // (op 494) 


s_add_u32 s18, s4, 16 // (op 494) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 494) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 494) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 494) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 494) 

s_mul_i32 s1, s45, s18 // (op 494) 

s_mul_i32 s9, s10, s1 // (op 494) 

s_mul_i32 s1, s8, s9 // (op 494) 

s_sub_u32 s9, s16, s1 // (op 494) 

s_mov_b32 s1, s9 // (op 494) 
s_mul_hi_i32 s18, s1, s44 // (op 494) 
s_add_i32 s19, s18, s1 // (op 494) 


s_ashr_i32 s1, s19, 31 // (op 494) 
s_cmp_eq_i32 s44, 0 // (op 494) 
s_mov_b32 s18, scc // (op 494) 
s_cmp_eq_i32 s18, 0 // (op 494) 
s_cselect_b32 s49, 0, -1 // (op 494) 

s_and_b32 s18, s48, 31 // (op 494) 
s_lshl_b32 s50, 1, s18 // (op 494) 
s_add_i32 s51, s50, s49 // (op 494) 


s_and_b32 s49, s1, s51 // (op 494) 


s_add_i32 s1, s19, s49 // (op 494) 


s_ashr_i32 s19, s1, s18 // (op 494) 


s_lshl_b32 s1, s48, 24 // (op 494) 
s_ashr_i32 s18, s1, 31 // (op 494) 

s_xor_b32 s1, s19, s18 // (op 494) 

s_sub_i32 s19, s1, s18 // (op 494) 


s_mul_i32 s1, s47, s45 // (op 494) 
s_add_u32 s50, s6, 16 // (op 494) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 494) End of Int64 add, unlocking SCC
s_sub_u32 s56, s50, 1 // (op 494) least significant half; sets scc
s_subb_u32 s57, s51, 0 // (op 494) most significant half; uses scc

s_ashr_i64 s[50:51], s[56:57], 4 // (op 494) 

s_sub_i32 s18, s50, s1 // (op 494) 

s_mul_i32 s49, s19, s18 // (op 494) 


s_sub_u32 s18, s9, s49 // (op 494) 


s_add_u32 s9, s18, s1 // (op 494) 


s_cmp_eq_i32 s8, 0 // (op 494) 
s_mov_b32 s1, scc // (op 494) 

s_cmp_eq_i32 s1, 0 // (op 494) 
s_cselect_b32 s8, s9, s0 // (op 494) 



v_lshl_add_u32 v14, s8, 4, v13 // (op 494) 


v_mul_lo_u32 v19, v14, s3 // (op 494) most significant: low of low * high
v_mul_hi_u32 v13, v14, s2 // (op 494) most significant: high of low * low
v_mul_lo_u32 v18, v14, s2 // (op 494) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 494) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 494) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 494) most significant half



v_lshlrev_b64 v[16:17], 2, v[18:19] // (op 494) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 494) 

v_mov_b32 v13, v14 // (op 494) convert

s_mul_i32 s1, s2, s7 // (op 496) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 496) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 496) least significant
s_mul_hi_u32 s9, s2, s6 // (op 496) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 496) most significant: sum
s_add_u32 s1, s1, s9 // (op 496) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 496) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 496) 

s_mov_b32 s52, s20 // (op 496) reg expression
s_mov_b32 s53, s21 // (op 496) reg expression
s_mov_b32 s54, s0 // (op 496) reg expression
s_mov_b32 s55, 131072 // (op 496) call()

s_mul_hi_i32 s0, s16, s12 // (op 539) 
v_lshrrev_b32 v14, 3, v1 // (op 539) 
s_add_i32 s1, s0, s16 // (op 539) 

v_lshlrev_b32 v15, 1, v14 // (op 539) 

s_ashr_i32 s0, s1, 31 // (op 539) 
v_and_b32 v14, -8, v1 // (op 539) 
s_cmp_eq_i32 s12, 0 // (op 539) 
s_mov_b32 s8, scc // (op 539) 
v_sub_u32 v16, v1, v14 // (op 539) 

s_cmp_eq_i32 s8, 0 // (op 539) 
s_cselect_b32 s9, 0, -1 // (op 539) 

s_and_b32 s8, s13, 31 // (op 539) 
s_lshl_b32 s18, 1, s8 // (op 539) 
s_add_i32 s19, s18, s9 // (op 539) 


s_and_b32 s9, s0, s19 // (op 539) 


s_add_i32 s0, s1, s9 // (op 539) 


s_ashr_i32 s1, s0, s8 // (op 539) 


s_lshl_b32 s0, s13, 24 // (op 539) 
s_ashr_i32 s8, s0, 31 // (op 539) 

s_xor_b32 s0, s1, s8 // (op 539) 

s_sub_i32 s1, s0, s8 // (op 539) 


s_mul_hi_i32 s0, s1, s14 // (op 539) 
s_add_i32 s8, s0, s1 // (op 539) 

s_ashr_i32 s0, s8, 31 // (op 539) 
s_cmp_eq_i32 s14, 0 // (op 539) 
s_mov_b32 s9, scc // (op 539) 
s_cmp_eq_i32 s9, 0 // (op 539) 
s_cselect_b32 s18, 0, -1 // (op 539) 

s_and_b32 s9, s15, 31 // (op 539) 
s_lshl_b32 s19, 1, s9 // (op 539) 
s_add_i32 s49, s19, s18 // (op 539) 


s_and_b32 s18, s0, s49 // (op 539) 


s_add_i32 s0, s8, s18 // (op 539) 


s_ashr_i32 s8, s0, s9 // (op 539) 


s_lshl_b32 s0, s15, 24 // (op 539) 
s_ashr_i32 s9, s0, 31 // (op 539) 

s_xor_b32 s0, s8, s9 // (op 539) 

s_sub_i32 s8, s0, s9 // (op 539) 


s_add_u32 s18, s4, 16 // (op 539) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 539) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 539) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 539) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 539) 

s_mul_i32 s0, s8, s18 // (op 539) 

s_sub_i32 s8, s1, s0 // (op 539) 


s_mul_hi_i32 s9, s16, s11 // (op 539) 
s_add_i32 s0, s9, s16 // (op 539) 

s_ashr_i32 s1, s0, 31 // (op 539) 
s_cmp_eq_i32 s11, 0 // (op 539) 
s_mov_b32 s9, scc // (op 539) 
s_cmp_eq_i32 s9, 0 // (op 539) 
s_cselect_b32 s49, 0, -1 // (op 539) 

s_and_b32 s9, s46, 31 // (op 539) 
s_lshl_b32 s50, 1, s9 // (op 539) 
s_add_i32 s51, s50, s49 // (op 539) 


s_and_b32 s49, s1, s51 // (op 539) 


s_add_i32 s1, s0, s49 // (op 539) 


s_ashr_i32 s0, s1, s9 // (op 539) 


s_lshl_b32 s1, s46, 24 // (op 539) 
s_ashr_i32 s9, s1, 31 // (op 539) 

s_xor_b32 s1, s0, s9 // (op 539) 

s_sub_i32 s0, s1, s9 // (op 539) 


s_mul_i32 s1, s45, s18 // (op 539) 

s_mul_i32 s9, s10, s1 // (op 539) 

s_mul_i32 s1, s0, s9 // (op 539) 

s_sub_u32 s9, s16, s1 // (op 539) 

s_mul_hi_i32 s1, s9, s44 // (op 539) 
s_add_i32 s18, s1, s9 // (op 539) 


s_ashr_i32 s1, s18, 31 // (op 539) 
s_cmp_eq_i32 s44, 0 // (op 539) 
s_mov_b32 s9, scc // (op 539) 
s_cmp_eq_i32 s9, 0 // (op 539) 
s_cselect_b32 s19, 0, -1 // (op 539) 

s_and_b32 s9, s48, 31 // (op 539) 
s_lshl_b32 s49, 1, s9 // (op 539) 
s_add_i32 s50, s49, s19 // (op 539) 


s_and_b32 s19, s1, s50 // (op 539) 


s_add_i32 s1, s18, s19 // (op 539) 


s_ashr_i32 s18, s1, s9 // (op 539) 


s_lshl_b32 s1, s48, 24 // (op 539) 
s_ashr_i32 s9, s1, 31 // (op 539) 

s_xor_b32 s1, s18, s9 // (op 539) 

s_sub_i32 s18, s1, s9 // (op 539) 


s_cmp_eq_i32 s0, 0 // (op 539) 
s_mov_b32 s1, scc // (op 539) 

s_cmp_eq_i32 s1, 0 // (op 539) 
s_cselect_b32 s0, s18, s8 // (op 539) 



v_lshl_add_u32 v14, s0, 4, v15 // (op 539) 


v_mul_lo_u32 v15, v14, s22 // (op 539) 

v_lshl_add_u32 v14, v16, 4, v15 // (op 539) 


s_mul_i32 s1, s22, s5 // (op 541) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 541) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 541) least significant
s_mul_hi_u32 s9, s22, s4 // (op 541) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 541) most significant: sum
s_add_u32 s1, s1, s9 // (op 541) most significant: sum


s_mov_b32 s56, s24 // (op 541) reg expression
s_mov_b32 s57, s25 // (op 541) reg expression
s_mov_b32 s58, s0 // (op 541) reg expression
s_mov_b32 s59, 131072 // (op 541) call()

s_mul_hi_i32 s0, s16, s12 // (op 558) 
v_lshrrev_b32 v15, 3, v1 // (op 558) 
s_add_i32 s1, s0, s16 // (op 558) 

v_lshlrev_b32 v16, 1, v15 // (op 558) 

s_ashr_i32 s0, s1, 31 // (op 558) 
v_and_b32 v15, -8, v1 // (op 558) 
s_cmp_eq_i32 s12, 0 // (op 558) 
s_mov_b32 s8, scc // (op 558) 
v_sub_u32 v17, v1, v15 // (op 558) 

s_cmp_eq_i32 s8, 0 // (op 558) 
s_cselect_b32 s9, 0, -1 // (op 558) 

s_and_b32 s8, s13, 31 // (op 558) 
s_lshl_b32 s18, 1, s8 // (op 558) 
s_add_i32 s19, s18, s9 // (op 558) 


s_and_b32 s9, s0, s19 // (op 558) 


s_add_i32 s0, s1, s9 // (op 558) 


s_ashr_i32 s1, s0, s8 // (op 558) 


s_lshl_b32 s0, s13, 24 // (op 558) 
s_ashr_i32 s8, s0, 31 // (op 558) 

s_xor_b32 s0, s1, s8 // (op 558) 

s_sub_i32 s1, s0, s8 // (op 558) 


s_mul_hi_i32 s0, s1, s14 // (op 558) 
s_add_i32 s8, s0, s1 // (op 558) 

s_ashr_i32 s0, s8, 31 // (op 558) 
s_cmp_eq_i32 s14, 0 // (op 558) 
s_mov_b32 s9, scc // (op 558) 
s_cmp_eq_i32 s9, 0 // (op 558) 
s_cselect_b32 s18, 0, -1 // (op 558) 

s_and_b32 s9, s15, 31 // (op 558) 
s_lshl_b32 s19, 1, s9 // (op 558) 
s_add_i32 s49, s19, s18 // (op 558) 


s_and_b32 s18, s0, s49 // (op 558) 


s_add_i32 s0, s8, s18 // (op 558) 


s_ashr_i32 s8, s0, s9 // (op 558) 


s_lshl_b32 s0, s15, 24 // (op 558) 
s_ashr_i32 s9, s0, 31 // (op 558) 

s_xor_b32 s0, s8, s9 // (op 558) 

s_sub_i32 s8, s0, s9 // (op 558) 


s_add_u32 s18, s4, 16 // (op 558) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 558) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 558) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 558) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 558) 

s_mul_i32 s0, s8, s18 // (op 558) 

s_sub_i32 s8, s1, s0 // (op 558) 


s_mul_hi_i32 s9, s16, s11 // (op 558) 
s_add_i32 s0, s9, s16 // (op 558) 

s_ashr_i32 s1, s0, 31 // (op 558) 
s_cmp_eq_i32 s11, 0 // (op 558) 
s_mov_b32 s9, scc // (op 558) 
s_cmp_eq_i32 s9, 0 // (op 558) 
s_cselect_b32 s49, 0, -1 // (op 558) 

s_and_b32 s9, s46, 31 // (op 558) 
s_lshl_b32 s50, 1, s9 // (op 558) 
s_add_i32 s51, s50, s49 // (op 558) 


s_and_b32 s49, s1, s51 // (op 558) 


s_add_i32 s1, s0, s49 // (op 558) 


s_ashr_i32 s0, s1, s9 // (op 558) 


s_lshl_b32 s1, s46, 24 // (op 558) 
s_ashr_i32 s9, s1, 31 // (op 558) 

s_xor_b32 s1, s0, s9 // (op 558) 

s_sub_i32 s0, s1, s9 // (op 558) 


s_mul_i32 s1, s45, s18 // (op 558) 

s_mul_i32 s9, s10, s1 // (op 558) 

s_mul_i32 s1, s0, s9 // (op 558) 

s_sub_u32 s9, s16, s1 // (op 558) 

s_mul_hi_i32 s1, s9, s44 // (op 558) 
s_add_i32 s18, s1, s9 // (op 558) 


s_ashr_i32 s1, s18, 31 // (op 558) 
s_cmp_eq_i32 s44, 0 // (op 558) 
s_mov_b32 s9, scc // (op 558) 
s_cmp_eq_i32 s9, 0 // (op 558) 
s_cselect_b32 s19, 0, -1 // (op 558) 

s_and_b32 s9, s48, 31 // (op 558) 
s_lshl_b32 s49, 1, s9 // (op 558) 
s_add_i32 s50, s49, s19 // (op 558) 


s_and_b32 s19, s1, s50 // (op 558) 


s_add_i32 s1, s18, s19 // (op 558) 


s_ashr_i32 s18, s1, s9 // (op 558) 


s_lshl_b32 s1, s48, 24 // (op 558) 
s_ashr_i32 s9, s1, 31 // (op 558) 

s_xor_b32 s1, s18, s9 // (op 558) 

s_sub_i32 s18, s1, s9 // (op 558) 


s_cmp_eq_i32 s0, 0 // (op 558) 
s_mov_b32 s1, scc // (op 558) 

s_cmp_eq_i32 s1, 0 // (op 558) 
s_cselect_b32 s0, s18, s8 // (op 558) 



v_lshl_add_u32 v15, s0, 4, v16 // (op 558) 


v_mul_lo_u32 v16, v15, s22 // (op 558) 

v_lshl_add_u32 v15, v17, 4, v16 // (op 558) 


s_mul_i32 s1, s22, s5 // (op 560) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 560) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 560) least significant
s_mul_hi_u32 s9, s22, s4 // (op 560) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 560) most significant: sum
s_add_u32 s1, s1, s9 // (op 560) most significant: sum


s_mov_b32 s56, s24 // (op 560) reg expression
s_mov_b32 s57, s25 // (op 560) reg expression
s_mov_b32 s58, s0 // (op 560) reg expression
s_mov_b32 s59, 131072 // (op 560) call()

s_mul_hi_i32 s0, s16, s12 // (op 603) 
v_lshrrev_b32 v16, 2, v1 // (op 603) 
s_add_i32 s1, s0, s16 // (op 603) 

v_and_b32 v17, -4, v1 // (op 603) 
s_ashr_i32 s0, s1, 31 // (op 603) 
v_sub_u32 v18, v1, v17 // (op 603) 

s_cmp_eq_i32 s12, 0 // (op 603) 
s_mov_b32 s8, scc // (op 603) 
s_cmp_eq_i32 s8, 0 // (op 603) 
s_cselect_b32 s9, 0, -1 // (op 603) 

s_and_b32 s8, s13, 31 // (op 603) 
s_lshl_b32 s18, 1, s8 // (op 603) 
s_add_i32 s19, s18, s9 // (op 603) 


s_and_b32 s9, s0, s19 // (op 603) 


s_add_i32 s0, s1, s9 // (op 603) 


s_ashr_i32 s1, s0, s8 // (op 603) 


s_lshl_b32 s0, s13, 24 // (op 603) 
s_ashr_i32 s8, s0, 31 // (op 603) 

s_xor_b32 s0, s1, s8 // (op 603) 

s_sub_i32 s1, s0, s8 // (op 603) 


s_mul_hi_i32 s0, s1, s14 // (op 603) 
s_add_i32 s8, s0, s1 // (op 603) 

s_ashr_i32 s0, s8, 31 // (op 603) 
s_cmp_eq_i32 s14, 0 // (op 603) 
s_mov_b32 s9, scc // (op 603) 
s_cmp_eq_i32 s9, 0 // (op 603) 
s_cselect_b32 s18, 0, -1 // (op 603) 

s_and_b32 s9, s15, 31 // (op 603) 
s_lshl_b32 s19, 1, s9 // (op 603) 
s_add_i32 s49, s19, s18 // (op 603) 


s_and_b32 s18, s0, s49 // (op 603) 


s_add_i32 s0, s8, s18 // (op 603) 


s_ashr_i32 s8, s0, s9 // (op 603) 


s_lshl_b32 s0, s15, 24 // (op 603) 
s_ashr_i32 s9, s0, 31 // (op 603) 

s_xor_b32 s0, s8, s9 // (op 603) 

s_sub_i32 s8, s0, s9 // (op 603) 


s_mul_i32 s0, s1, s45 // (op 603) 

s_sub_u32 s1, s16, s0 // (op 603) 

s_mul_i32 s0, s8, s45 // (op 603) 
s_add_u32 s0, s0, s1 // (op 603) 


s_mul_hi_i32 s1, s16, s11 // (op 603) 
s_add_i32 s8, s1, s16 // (op 603) 

s_ashr_i32 s1, s8, 31 // (op 603) 
s_cmp_eq_i32 s11, 0 // (op 603) 
s_mov_b32 s9, scc // (op 603) 
s_cmp_eq_i32 s9, 0 // (op 603) 
s_cselect_b32 s18, 0, -1 // (op 603) 

s_and_b32 s9, s46, 31 // (op 603) 
s_lshl_b32 s19, 1, s9 // (op 603) 
s_add_i32 s49, s19, s18 // (op 603) 


s_and_b32 s18, s1, s49 // (op 603) 


s_add_i32 s1, s8, s18 // (op 603) 


s_ashr_i32 s8, s1, s9 // (op 603) 


s_lshl_b32 s1, s46, 24 // (op 603) 
s_ashr_i32 s9, s1, 31 // (op 603) 

s_xor_b32 s1, s8, s9 // (op 603) 

s_sub_i32 s8, s1, s9 // (op 603) 


s_add_u32 s18, s4, 16 // (op 603) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 603) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 603) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 603) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 603) 

s_mul_i32 s1, s45, s18 // (op 603) 

s_mul_i32 s9, s10, s1 // (op 603) 

s_mul_i32 s1, s8, s9 // (op 603) 

s_sub_u32 s9, s16, s1 // (op 603) 

s_mov_b32 s1, s9 // (op 603) 
s_mul_hi_i32 s18, s1, s44 // (op 603) 
s_add_i32 s19, s18, s1 // (op 603) 


s_ashr_i32 s1, s19, 31 // (op 603) 
s_cmp_eq_i32 s44, 0 // (op 603) 
s_mov_b32 s18, scc // (op 603) 
s_cmp_eq_i32 s18, 0 // (op 603) 
s_cselect_b32 s49, 0, -1 // (op 603) 

s_and_b32 s18, s48, 31 // (op 603) 
s_lshl_b32 s50, 1, s18 // (op 603) 
s_add_i32 s51, s50, s49 // (op 603) 


s_and_b32 s49, s1, s51 // (op 603) 


s_add_i32 s1, s19, s49 // (op 603) 


s_ashr_i32 s19, s1, s18 // (op 603) 


s_lshl_b32 s1, s48, 24 // (op 603) 
s_ashr_i32 s18, s1, 31 // (op 603) 

s_xor_b32 s1, s19, s18 // (op 603) 

s_sub_i32 s19, s1, s18 // (op 603) 


s_mul_i32 s1, s47, s45 // (op 603) 
s_add_u32 s50, s6, 16 // (op 603) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 603) End of Int64 add, unlocking SCC
s_sub_u32 s60, s50, 1 // (op 603) least significant half; sets scc
s_subb_u32 s61, s51, 0 // (op 603) most significant half; uses scc

s_ashr_i64 s[50:51], s[60:61], 4 // (op 603) 

s_sub_i32 s18, s50, s1 // (op 603) 

s_mul_i32 s49, s19, s18 // (op 603) 


s_sub_u32 s18, s9, s49 // (op 603) 


s_add_u32 s9, s18, s1 // (op 603) 


s_cmp_eq_i32 s8, 0 // (op 603) 
s_mov_b32 s1, scc // (op 603) 

s_cmp_eq_i32 s1, 0 // (op 603) 
s_cselect_b32 s8, s9, s0 // (op 603) 



v_lshl_add_u32 v17, s8, 4, v16 // (op 603) 


v_mul_lo_u32 v16, v17, s26 // (op 603) 
v_add_u32 v16, v16, v18 // (op 603) 


s_mul_i32 s1, s26, s7 // (op 605) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 605) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 605) least significant
s_mul_hi_u32 s9, s26, s6 // (op 605) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 605) most significant: sum
s_add_u32 s1, s1, s9 // (op 605) most significant: sum


s_mov_b32 s60, s28 // (op 605) reg expression
s_mov_b32 s61, s29 // (op 605) reg expression
s_mov_b32 s62, s0 // (op 605) reg expression
s_mov_b32 s63, 131072 // (op 605) call()

s_mul_hi_i32 s0, s16, s12 // (op 622) 
v_lshrrev_b32 v17, 2, v1 // (op 622) 
s_add_i32 s1, s0, s16 // (op 622) 

v_and_b32 v18, -4, v1 // (op 622) 
s_ashr_i32 s0, s1, 31 // (op 622) 
v_sub_u32 v19, v1, v18 // (op 622) 

s_cmp_eq_i32 s12, 0 // (op 622) 
s_mov_b32 s8, scc // (op 622) 
s_cmp_eq_i32 s8, 0 // (op 622) 
s_cselect_b32 s9, 0, -1 // (op 622) 

s_and_b32 s8, s13, 31 // (op 622) 
s_lshl_b32 s18, 1, s8 // (op 622) 
s_add_i32 s19, s18, s9 // (op 622) 


s_and_b32 s9, s0, s19 // (op 622) 


s_add_i32 s0, s1, s9 // (op 622) 


s_ashr_i32 s1, s0, s8 // (op 622) 


s_lshl_b32 s0, s13, 24 // (op 622) 
s_ashr_i32 s8, s0, 31 // (op 622) 

s_xor_b32 s0, s1, s8 // (op 622) 

s_sub_i32 s1, s0, s8 // (op 622) 


s_mul_hi_i32 s0, s1, s14 // (op 622) 
s_add_i32 s8, s0, s1 // (op 622) 

s_ashr_i32 s0, s8, 31 // (op 622) 
s_cmp_eq_i32 s14, 0 // (op 622) 
s_mov_b32 s9, scc // (op 622) 
s_cmp_eq_i32 s9, 0 // (op 622) 
s_cselect_b32 s18, 0, -1 // (op 622) 

s_and_b32 s9, s15, 31 // (op 622) 
s_lshl_b32 s19, 1, s9 // (op 622) 
s_add_i32 s49, s19, s18 // (op 622) 


s_and_b32 s18, s0, s49 // (op 622) 


s_add_i32 s0, s8, s18 // (op 622) 


s_ashr_i32 s8, s0, s9 // (op 622) 


s_lshl_b32 s0, s15, 24 // (op 622) 
s_ashr_i32 s9, s0, 31 // (op 622) 

s_xor_b32 s0, s8, s9 // (op 622) 

s_sub_i32 s8, s0, s9 // (op 622) 


s_mul_i32 s0, s1, s45 // (op 622) 

s_sub_u32 s1, s16, s0 // (op 622) 

s_mul_i32 s0, s8, s45 // (op 622) 
s_add_u32 s0, s0, s1 // (op 622) 


s_mul_hi_i32 s1, s16, s11 // (op 622) 
s_add_i32 s8, s1, s16 // (op 622) 

s_ashr_i32 s1, s8, 31 // (op 622) 
s_cmp_eq_i32 s11, 0 // (op 622) 
s_mov_b32 s9, scc // (op 622) 
s_cmp_eq_i32 s9, 0 // (op 622) 
s_cselect_b32 s18, 0, -1 // (op 622) 

s_and_b32 s9, s46, 31 // (op 622) 
s_lshl_b32 s19, 1, s9 // (op 622) 
s_add_i32 s49, s19, s18 // (op 622) 


s_and_b32 s18, s1, s49 // (op 622) 


s_add_i32 s1, s8, s18 // (op 622) 


s_ashr_i32 s8, s1, s9 // (op 622) 


s_lshl_b32 s1, s46, 24 // (op 622) 
s_ashr_i32 s9, s1, 31 // (op 622) 

s_xor_b32 s1, s8, s9 // (op 622) 

s_sub_i32 s8, s1, s9 // (op 622) 


s_add_u32 s18, s4, 16 // (op 622) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 622) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 622) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 622) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 622) 

s_mul_i32 s1, s45, s18 // (op 622) 

s_mul_i32 s9, s10, s1 // (op 622) 

s_mul_i32 s1, s8, s9 // (op 622) 

s_sub_u32 s9, s16, s1 // (op 622) 

s_mov_b32 s1, s9 // (op 622) 
s_mul_hi_i32 s18, s1, s44 // (op 622) 
s_add_i32 s19, s18, s1 // (op 622) 


s_ashr_i32 s1, s19, 31 // (op 622) 
s_cmp_eq_i32 s44, 0 // (op 622) 
s_mov_b32 s18, scc // (op 622) 
s_cmp_eq_i32 s18, 0 // (op 622) 
s_cselect_b32 s49, 0, -1 // (op 622) 

s_and_b32 s18, s48, 31 // (op 622) 
s_lshl_b32 s50, 1, s18 // (op 622) 
s_add_i32 s51, s50, s49 // (op 622) 


s_and_b32 s49, s1, s51 // (op 622) 


s_add_i32 s1, s19, s49 // (op 622) 


s_ashr_i32 s19, s1, s18 // (op 622) 


s_lshl_b32 s1, s48, 24 // (op 622) 
s_ashr_i32 s18, s1, 31 // (op 622) 

s_xor_b32 s1, s19, s18 // (op 622) 

s_sub_i32 s19, s1, s18 // (op 622) 


s_mul_i32 s1, s47, s45 // (op 622) 
s_add_u32 s50, s6, 16 // (op 622) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 622) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 622) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 622) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 4 // (op 622) 

s_sub_i32 s18, s50, s1 // (op 622) 

s_mul_i32 s49, s19, s18 // (op 622) 


s_sub_u32 s18, s9, s49 // (op 622) 


s_add_u32 s9, s18, s1 // (op 622) 


s_cmp_eq_i32 s8, 0 // (op 622) 
s_mov_b32 s1, scc // (op 622) 

s_cmp_eq_i32 s1, 0 // (op 622) 
s_cselect_b32 s8, s9, s0 // (op 622) 



v_lshl_add_u32 v18, s8, 4, v17 // (op 622) 


v_mul_lo_u32 v17, v18, s26 // (op 622) 
v_add_u32 v17, v17, v19 // (op 622) 


s_mul_i32 s1, s26, s7 // (op 624) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 624) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 624) least significant
s_mul_hi_u32 s9, s26, s6 // (op 624) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 624) most significant: sum
s_add_u32 s1, s1, s9 // (op 624) most significant: sum


s_mov_b32 s60, s28 // (op 624) reg expression
s_mov_b32 s61, s29 // (op 624) reg expression
s_mov_b32 s62, s0 // (op 624) reg expression
s_mov_b32 s63, 131072 // (op 624) call()

s_mul_hi_i32 s0, s16, s12 // (op 667) 
v_lshrrev_b32 v18, 2, v1 // (op 667) 
s_add_i32 s1, s0, s16 // (op 667) 

v_and_b32 v19, -4, v1 // (op 667) 
s_ashr_i32 s0, s1, 31 // (op 667) 
v_sub_u32 v20, v1, v19 // (op 667) 

s_cmp_eq_i32 s12, 0 // (op 667) 
s_mov_b32 s8, scc // (op 667) 
s_cmp_eq_i32 s8, 0 // (op 667) 
s_cselect_b32 s9, 0, -1 // (op 667) 

s_and_b32 s8, s13, 31 // (op 667) 
s_lshl_b32 s18, 1, s8 // (op 667) 
s_add_i32 s19, s18, s9 // (op 667) 


s_and_b32 s9, s0, s19 // (op 667) 


s_add_i32 s0, s1, s9 // (op 667) 


s_ashr_i32 s1, s0, s8 // (op 667) 


s_lshl_b32 s0, s13, 24 // (op 667) 
s_ashr_i32 s8, s0, 31 // (op 667) 

s_xor_b32 s0, s1, s8 // (op 667) 

s_sub_i32 s1, s0, s8 // (op 667) 


s_mul_hi_i32 s0, s1, s14 // (op 667) 
s_add_i32 s8, s0, s1 // (op 667) 

s_ashr_i32 s0, s8, 31 // (op 667) 
s_cmp_eq_i32 s14, 0 // (op 667) 
s_mov_b32 s9, scc // (op 667) 
s_cmp_eq_i32 s9, 0 // (op 667) 
s_cselect_b32 s18, 0, -1 // (op 667) 

s_and_b32 s9, s15, 31 // (op 667) 
s_lshl_b32 s19, 1, s9 // (op 667) 
s_add_i32 s49, s19, s18 // (op 667) 


s_and_b32 s18, s0, s49 // (op 667) 


s_add_i32 s0, s8, s18 // (op 667) 


s_ashr_i32 s8, s0, s9 // (op 667) 


s_lshl_b32 s0, s15, 24 // (op 667) 
s_ashr_i32 s9, s0, 31 // (op 667) 

s_xor_b32 s0, s8, s9 // (op 667) 

s_sub_i32 s8, s0, s9 // (op 667) 


s_add_u32 s18, s4, 16 // (op 667) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 667) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 667) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 667) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 667) 

s_mul_i32 s0, s8, s18 // (op 667) 

s_sub_i32 s8, s1, s0 // (op 667) 


s_mul_hi_i32 s9, s16, s11 // (op 667) 
s_add_i32 s0, s9, s16 // (op 667) 

s_ashr_i32 s1, s0, 31 // (op 667) 
s_cmp_eq_i32 s11, 0 // (op 667) 
s_mov_b32 s9, scc // (op 667) 
s_cmp_eq_i32 s9, 0 // (op 667) 
s_cselect_b32 s49, 0, -1 // (op 667) 

s_and_b32 s9, s46, 31 // (op 667) 
s_lshl_b32 s50, 1, s9 // (op 667) 
s_add_i32 s51, s50, s49 // (op 667) 


s_and_b32 s49, s1, s51 // (op 667) 


s_add_i32 s1, s0, s49 // (op 667) 


s_ashr_i32 s0, s1, s9 // (op 667) 


s_lshl_b32 s1, s46, 24 // (op 667) 
s_ashr_i32 s9, s1, 31 // (op 667) 

s_xor_b32 s1, s0, s9 // (op 667) 

s_sub_i32 s0, s1, s9 // (op 667) 


s_mul_i32 s1, s45, s18 // (op 667) 

s_mul_i32 s9, s10, s1 // (op 667) 

s_mul_i32 s1, s0, s9 // (op 667) 

s_sub_u32 s9, s16, s1 // (op 667) 

s_mul_hi_i32 s1, s9, s44 // (op 667) 
s_add_i32 s18, s1, s9 // (op 667) 


s_ashr_i32 s1, s18, 31 // (op 667) 
s_cmp_eq_i32 s44, 0 // (op 667) 
s_mov_b32 s9, scc // (op 667) 
s_cmp_eq_i32 s9, 0 // (op 667) 
s_cselect_b32 s19, 0, -1 // (op 667) 

s_and_b32 s9, s48, 31 // (op 667) 
s_lshl_b32 s49, 1, s9 // (op 667) 
s_add_i32 s50, s49, s19 // (op 667) 


s_and_b32 s19, s1, s50 // (op 667) 


s_add_i32 s1, s18, s19 // (op 667) 


s_ashr_i32 s18, s1, s9 // (op 667) 


s_lshl_b32 s1, s48, 24 // (op 667) 
s_ashr_i32 s9, s1, 31 // (op 667) 

s_xor_b32 s1, s18, s9 // (op 667) 

s_sub_i32 s18, s1, s9 // (op 667) 


s_cmp_eq_i32 s0, 0 // (op 667) 
s_mov_b32 s1, scc // (op 667) 

s_cmp_eq_i32 s1, 0 // (op 667) 
s_cselect_b32 s0, s18, s8 // (op 667) 



v_lshl_add_u32 v19, s0, 4, v18 // (op 667) 


v_mul_lo_u32 v18, v19, s30 // (op 667) 

v_add_u32 v19, v20, v18 // (op 667) 


s_mul_i32 s1, s30, s5 // (op 669) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 669) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 669) least significant
s_mul_hi_u32 s9, s30, s4 // (op 669) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 669) most significant: sum
s_add_u32 s1, s1, s9 // (op 669) most significant: sum


s_mov_b32 s64, s32 // (op 669) reg expression
s_mov_b32 s65, s33 // (op 669) reg expression
s_mov_b32 s66, s0 // (op 669) reg expression
s_mov_b32 s67, 131072 // (op 669) call()

s_mul_hi_i32 s0, s16, s12 // (op 686) 
v_lshrrev_b32 v18, 2, v1 // (op 686) 
s_add_i32 s1, s0, s16 // (op 686) 

v_and_b32 v20, -4, v1 // (op 686) 
s_ashr_i32 s0, s1, 31 // (op 686) 
v_sub_u32 v21, v1, v20 // (op 686) 

s_cmp_eq_i32 s12, 0 // (op 686) 
s_mov_b32 s8, scc // (op 686) 
s_cmp_eq_i32 s8, 0 // (op 686) 
s_cselect_b32 s9, 0, -1 // (op 686) 

s_and_b32 s8, s13, 31 // (op 686) 
s_lshl_b32 s18, 1, s8 // (op 686) 
s_add_i32 s19, s18, s9 // (op 686) 


s_and_b32 s9, s0, s19 // (op 686) 


s_add_i32 s0, s1, s9 // (op 686) 


s_ashr_i32 s1, s0, s8 // (op 686) 


s_lshl_b32 s0, s13, 24 // (op 686) 
s_ashr_i32 s8, s0, 31 // (op 686) 

s_xor_b32 s0, s1, s8 // (op 686) 

s_sub_i32 s1, s0, s8 // (op 686) 


s_mul_hi_i32 s0, s1, s14 // (op 686) 
s_add_i32 s8, s0, s1 // (op 686) 

s_ashr_i32 s0, s8, 31 // (op 686) 
s_cmp_eq_i32 s14, 0 // (op 686) 
s_mov_b32 s9, scc // (op 686) 
s_cmp_eq_i32 s9, 0 // (op 686) 
s_cselect_b32 s18, 0, -1 // (op 686) 

s_and_b32 s9, s15, 31 // (op 686) 
s_lshl_b32 s19, 1, s9 // (op 686) 
s_add_i32 s49, s19, s18 // (op 686) 


s_and_b32 s18, s0, s49 // (op 686) 


s_add_i32 s0, s8, s18 // (op 686) 


s_ashr_i32 s8, s0, s9 // (op 686) 


s_lshl_b32 s0, s15, 24 // (op 686) 
s_ashr_i32 s9, s0, 31 // (op 686) 

s_xor_b32 s0, s8, s9 // (op 686) 

s_sub_i32 s8, s0, s9 // (op 686) 


s_add_u32 s18, s4, 16 // (op 686) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 686) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 686) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 686) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 686) 

s_mul_i32 s0, s8, s18 // (op 686) 

s_sub_i32 s8, s1, s0 // (op 686) 


s_mul_hi_i32 s9, s16, s11 // (op 686) 
s_add_i32 s0, s9, s16 // (op 686) 

s_ashr_i32 s1, s0, 31 // (op 686) 
s_cmp_eq_i32 s11, 0 // (op 686) 
s_mov_b32 s9, scc // (op 686) 
s_cmp_eq_i32 s9, 0 // (op 686) 
s_cselect_b32 s49, 0, -1 // (op 686) 

s_and_b32 s9, s46, 31 // (op 686) 
s_lshl_b32 s50, 1, s9 // (op 686) 
s_add_i32 s51, s50, s49 // (op 686) 


s_and_b32 s49, s1, s51 // (op 686) 


s_add_i32 s1, s0, s49 // (op 686) 


s_ashr_i32 s0, s1, s9 // (op 686) 


s_lshl_b32 s1, s46, 24 // (op 686) 
s_ashr_i32 s9, s1, 31 // (op 686) 

s_xor_b32 s1, s0, s9 // (op 686) 

s_sub_i32 s0, s1, s9 // (op 686) 


s_mul_i32 s1, s45, s18 // (op 686) 

s_mul_i32 s9, s10, s1 // (op 686) 

s_mul_i32 s1, s0, s9 // (op 686) 

s_sub_u32 s9, s16, s1 // (op 686) 

s_mul_hi_i32 s1, s9, s44 // (op 686) 
s_add_i32 s18, s1, s9 // (op 686) 


s_ashr_i32 s1, s18, 31 // (op 686) 
s_cmp_eq_i32 s44, 0 // (op 686) 
s_mov_b32 s9, scc // (op 686) 
s_cmp_eq_i32 s9, 0 // (op 686) 
s_cselect_b32 s19, 0, -1 // (op 686) 

s_and_b32 s9, s48, 31 // (op 686) 
s_lshl_b32 s49, 1, s9 // (op 686) 
s_add_i32 s50, s49, s19 // (op 686) 


s_and_b32 s19, s1, s50 // (op 686) 


s_add_i32 s1, s18, s19 // (op 686) 


s_ashr_i32 s18, s1, s9 // (op 686) 


s_lshl_b32 s1, s48, 24 // (op 686) 
s_ashr_i32 s9, s1, 31 // (op 686) 

s_xor_b32 s1, s18, s9 // (op 686) 

s_sub_i32 s18, s1, s9 // (op 686) 


s_cmp_eq_i32 s0, 0 // (op 686) 
s_mov_b32 s1, scc // (op 686) 

s_cmp_eq_i32 s1, 0 // (op 686) 
s_cselect_b32 s0, s18, s8 // (op 686) 



v_lshl_add_u32 v20, s0, 4, v18 // (op 686) 


v_mul_lo_u32 v18, v20, s30 // (op 686) 

v_add_u32 v20, v21, v18 // (op 686) 


s_mul_i32 s1, s30, s5 // (op 688) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 688) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 688) least significant
s_mul_hi_u32 s9, s30, s4 // (op 688) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 688) most significant: sum
s_add_u32 s1, s1, s9 // (op 688) most significant: sum


s_mov_b32 s64, s32 // (op 688) reg expression
s_mov_b32 s65, s33 // (op 688) reg expression
s_mov_b32 s66, s0 // (op 688) reg expression
s_mov_b32 s67, 131072 // (op 688) call()


v_mov_b64 v[22:23], s[22:23] // (op 366) convert

v_mov_b32 v18, v14 // (op 366) 
buffer_load_dwordx4 v[24:27], v18, s[56:59], 0 offen // (op 366) Load value
v_add_u32 v18, v18, v22 // (op 366) 
buffer_load_dwordx4 v[28:31], v18, s[56:59], 0 offen // (op 366) Load value





buffer_load_ubyte v18, v19, s[64:67], 0 offen offset: 0 // (op 369) Load value



buffer_load_dwordx4 v[32:35], v12, s[52:55], 0 offen // (op 372) Load value



buffer_load_ubyte v21, v16, s[60:63], 0 offen offset: 0 // (op 375) Load value

v_mov_b64 v[22:23], s[22:23] // (op 378) convert

v_mov_b32 v36, v14 // (op 378) 
buffer_load_dwordx4 v[38:41], v36, s[56:59], 0 offen offset:128 // (op 378) Load value
v_add_u32 v36, v36, v22 // (op 378) 
buffer_load_dwordx4 v[42:45], v36, s[56:59], 0 offen offset:128 // (op 378) Load value






buffer_load_ubyte v14, v19, s[64:67], 0 offen offset:4 // (op 381) Load value




buffer_load_dwordx4 v[46:49], v12, s[52:55], 0 offen offset:64 // (op 384) Load value




buffer_load_ubyte v12, v16, s[60:63], 0 offen offset:4 // (op 387) Load value

v_and_b32 v16, -8, v1 // (op 806) 
v_sub_u32 v19, v1, v16 // (op 806) 

v_lshlrev_b32 v16, 4, v19 // (op 806) 

v_lshrrev_b32 v19, 3, v1 // (op 806) 
v_lshl_add_u32 v22, v19, 8, v16 // (op 806) 





s_waitcnt vmcnt(9)
ds_write_b128 v22, v[24:27] // (op 389) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v22, v[28:31] offset:128 // (op 389) Store local data 

v_and_b32 v16, -4, v1 // (op 912) 
v_sub_u32 v19, v1, v16 // (op 912) 

v_and_b32 v16, 4294967292, v1 // (op 912) 
v_add_u32 v22, v16, v19 // (op 912) 





s_waitcnt vmcnt(7)
ds_write_b8 v22, v18 offset:2048 // (op 390) Store local data 

v_and_b32 v16, -4, v1 // (op 753) 
v_sub_u32 v19, v1, v16 // (op 753) 

v_lshlrev_b32 v16, 5, v19 // (op 753) 

v_lshrrev_b32 v19, 2, v1 // (op 753) 
v_lshl_add_u32 v22, v19, 7, v16 // (op 753) 


v_lshlrev_b32 v16, 2, v22 // (op 753) 

v_lshrrev_b32 v19, 3, v16 // (op 753) 




s_waitcnt vmcnt(6)
ds_write_b128 v19, v[32:35] offset:2112 // (op 391) Store local data 

v_and_b32 v16, -4, v1 // (op 859) 
v_sub_u32 v19, v1, v16 // (op 859) 

v_and_b32 v16, 4294967292, v1 // (op 859) 
v_add_u32 v22, v16, v19 // (op 859) 





s_waitcnt vmcnt(5)
ds_write_b8 v22, v21 offset:3136 // (op 392) Store local data 

v_and_b32 v16, -4, v1 // (op 724) 
v_sub_u32 v19, v1, v16 // (op 724) 

v_lshlrev_b32 v16, 5, v19 // (op 724) 

v_lshrrev_b32 v19, 2, v1 // (op 724) 
v_lshl_add_u32 v22, v19, 7, v16 // (op 724) 


v_lshlrev_b32 v16, 2, v22 // (op 724) 

v_lshrrev_b32 v19, 3, v16 // (op 724) 

v_and_b32 v16, -8, v1 // (op 777) 
v_sub_u32 v22, v1, v16 // (op 777) 

v_lshlrev_b32 v16, 4, v22 // (op 777) 

v_lshrrev_b32 v22, 3, v1 // (op 777) 
v_lshl_add_u32 v23, v22, 8, v16 // (op 777) 


v_and_b32 v16, -4, v1 // (op 830) 
v_sub_u32 v22, v1, v16 // (op 830) 

v_and_b32 v16, 4294967292, v1 // (op 830) 
v_add_u32 v36, v16, v22 // (op 830) 


v_and_b32 v16, -4, v1 // (op 883) 
v_sub_u32 v22, v1, v16 // (op 883) 

v_and_b32 v16, 4294967292, v1 // (op 883) 
v_add_u32 v37, v16, v22 // (op 883) 


v_and_b32 v16, -4, v1 // (op 970) 
v_sub_u32 v22, v1, v16 // (op 970) 

v_lshlrev_b32 v16, 5, v22 // (op 970) 

v_lshrrev_b32 v22, 2, v1 // (op 970) 
v_lshl_add_u32 v50, v22, 7, v16 // (op 970) 


v_lshlrev_b32 v16, 2, v50 // (op 970) 

v_lshrrev_b32 v22, 3, v16 // (op 970) 

v_and_b32 v16, -8, v1 // (op 994) 
v_sub_u32 v50, v1, v16 // (op 994) 

v_lshlrev_b32 v16, 4, v50 // (op 994) 

v_lshrrev_b32 v50, 3, v1 // (op 994) 
v_lshl_add_u32 v51, v50, 8, v16 // (op 994) 


v_and_b32 v16, -4, v1 // (op 1018) 
v_sub_u32 v50, v1, v16 // (op 1018) 

v_and_b32 v16, 4294967292, v1 // (op 1018) 
v_add_u32 v52, v16, v50 // (op 1018) 


v_and_b32 v16, -4, v1 // (op 1042) 
v_sub_u32 v50, v1, v16 // (op 1042) 

v_and_b32 v16, 4294967292, v1 // (op 1042) 
v_add_u32 v53, v16, v50 // (op 1042) 


s_mov_b32 s0, 0 // (op 59) call()
s_ashr_i64 s[8:9], s[34:35], 7 // (op 58) 
s_mov_b32 s1, s8 // (op 58) convert

s_and_b32 s8, s1, 4294967294 // (op 58) 

s_cmp_lt_u32 s0, s8 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_58 if false)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_58:

s_add_u32 s0, s0, 2 // (op 60) 
v_mov_b64 v[54:55], s[22:23] // (op 10) convert

v_mov_b32 v16, v15 // (op 10) 
v_mov_b32 v50, v16 // (op 10) 
buffer_load_dwordx4 v[24:27], v50, s[56:59], 0 offen offset:256 // (op 10) Load value
v_add_u32 v50, v50, v54 // (op 10) 
buffer_load_dwordx4 v[28:31], v50, s[56:59], 0 offen offset:256 // (op 10) Load value


s_waitcnt lgkmcnt(1)
s_barrier  // (op 1091) 





v_mov_b32 v50, v20 // (op 25) 

ds_read_b128 v[54:57], v3 offset:2112 // (op 70) Load local data 
buffer_load_ubyte v18, v50, s[64:67], 0 offen offset:8 // (op 25) Load value


s_barrier  // (op 1104) 
v_mov_b32 v58, v13 // (op 4) 




ds_read_b128 v[60:63], v5 // (op 75) Load local data 
ds_read_b128 v[64:67], v5 offset:64 // (op 75) Load local data 
buffer_load_dwordx4 v[32:35], v58, s[52:55], 0 offen offset:128 // (op 4) Load value


v_mov_b32 v59, v17 // (op 17) 

buffer_load_ubyte v21, v59, s[60:63], 0 offen offset:8 // (op 17) Load value
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1118) 



s_barrier  // (op 1155) 
ds_read_u8 v68, v0 offset:3136 // (op 80) Load local data 



s_barrier  // (op 1132) 
s_waitcnt vmcnt(9)
ds_write_b128 v51, v[38:41] offset:3200 // (op 253) Store local data 



ds_read_u8 v69, v6 offset:2048 // (op 85) Load local data 
s_waitcnt vmcnt(8)
ds_write_b128 v51, v[42:45] offset:3328 // (op 253) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:57], v[60:67], a[0:3], v68, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 88) 




s_barrier  // (op 1167) 



s_waitcnt vmcnt(7)
ds_write_b8 v53, v14 offset:5248 // (op 255) Store local data 
s_barrier  // (op 1149) 



s_waitcnt vmcnt(6)
ds_write_b128 v22, v[46:49] offset:5312 // (op 243) Store local data 
s_barrier  // (op 1161) 



s_waitcnt vmcnt(5)
ds_write_b8 v52, v12 offset:6336 // (op 254) Store local data 

v_mov_b64 v[54:55], s[22:23] // (op 256) convert

v_mov_b32 v56, v16 // (op 256) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1146) 
buffer_load_dwordx4 v[38:41], v56, s[56:59], 0 offen offset:384 // (op 256) Load value



v_add_u32 v56, v56, v54 // (op 256) 
ds_read_b128 v[60:63], v7 offset:5312 // (op 245) Load local data 
buffer_load_dwordx4 v[42:45], v56, s[56:59], 0 offen offset:384 // (op 256) Load value






s_barrier  // (op 1152) 
s_barrier  // (op 1222) 



buffer_load_ubyte v14, v50, s[64:67], 0 offen offset:12 // (op 258) Load value
v_add_u32 v15, 256, v15 // (op 550) 

ds_read_b128 v[64:67], v9 offset:3200 // (op 247) Load local data 



ds_read_b128 v[68:71], v9 offset:3264 // (op 247) Load local data 
v_add_u32 v20, 8, v20 // (op 678) 
buffer_load_dwordx4 v[46:49], v58, s[52:55], 0 offen offset:192 // (op 242) Load value

s_barrier  // (op 1228) 



v_add_u32 v13, 128, v13 // (op 486) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1158) 
buffer_load_ubyte v12, v59, s[60:63], 0 offen offset:12 // (op 257) Load value




ds_read_u8 v16, v4 offset:6336 // (op 249) Load local data 
s_barrier  // (op 1107) 



s_barrier  // (op 1164) 
s_barrier  // (op 1234) 



v_add_u32 v17, 8, v17 // (op 614) 
ds_read_u8 v50, v10 offset:5248 // (op 251) Load local data 
s_waitcnt vmcnt(9)
ds_write_b128 v23, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v23, v[28:31] offset:128 // (op 13) Store local data 
s_barrier  // (op 1240) 
s_barrier  // (op 1135) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:63], v[64:71], a[0:3], v16, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 252) 







s_waitcnt vmcnt(7)
ds_write_b8 v37, v18 offset:2048 // (op 30) Store local data 
s_barrier  // (op 1094) 



s_waitcnt vmcnt(6)
ds_write_b128 v19, v[32:35] offset:2112 // (op 6) Store local data 
s_barrier  // (op 1121) 



s_waitcnt vmcnt(5)
ds_write_b8 v36, v21 offset:3136 // (op 21) Store local data 
s_ashr_i64 s[8:9], s[34:35], 7 // (op 58) 
s_mov_b32 s1, s8 // (op 58) convert

s_and_b32 s8, s1, 4294967294 // (op 58) 

s_cmp_lt_u32 s0, s8 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_58 if true)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_58:

















s_mul_hi_i32 s0, s16, s12 // (op 449) 
v_lshrrev_b32 v13, 2, v1 // (op 449) 
s_add_i32 s1, s0, s16 // (op 449) 

v_and_b32 v15, -4, v1 // (op 449) 
s_ashr_i32 s0, s1, 31 // (op 449) 
v_sub_u32 v16, v1, v15 // (op 449) 

s_cmp_eq_i32 s12, 0 // (op 449) 
s_mov_b32 s8, scc // (op 449) 
v_lshlrev_b32 v15, 5, v16 // (op 449) 

s_cmp_eq_i32 s8, 0 // (op 449) 
s_cselect_b32 s9, 0, -1 // (op 449) 

s_and_b32 s8, s13, 31 // (op 449) 
s_lshl_b32 s18, 1, s8 // (op 449) 
s_add_i32 s19, s18, s9 // (op 449) 


s_and_b32 s9, s0, s19 // (op 449) 


s_add_i32 s0, s1, s9 // (op 449) 


s_ashr_i32 s1, s0, s8 // (op 449) 


s_lshl_b32 s0, s13, 24 // (op 449) 
s_ashr_i32 s8, s0, 31 // (op 449) 

s_xor_b32 s0, s1, s8 // (op 449) 

s_sub_i32 s1, s0, s8 // (op 449) 


s_mul_hi_i32 s0, s1, s14 // (op 449) 
s_add_i32 s8, s0, s1 // (op 449) 

s_ashr_i32 s0, s8, 31 // (op 449) 
s_cmp_eq_i32 s14, 0 // (op 449) 
s_mov_b32 s9, scc // (op 449) 
s_cmp_eq_i32 s9, 0 // (op 449) 
s_cselect_b32 s18, 0, -1 // (op 449) 

s_and_b32 s9, s15, 31 // (op 449) 
s_lshl_b32 s19, 1, s9 // (op 449) 
s_add_i32 s49, s19, s18 // (op 449) 


s_and_b32 s18, s0, s49 // (op 449) 


s_add_i32 s0, s8, s18 // (op 449) 


s_ashr_i32 s8, s0, s9 // (op 449) 


s_lshl_b32 s0, s15, 24 // (op 449) 
s_ashr_i32 s9, s0, 31 // (op 449) 

s_xor_b32 s0, s8, s9 // (op 449) 

s_sub_i32 s8, s0, s9 // (op 449) 


s_mul_i32 s0, s1, s45 // (op 449) 

s_sub_u32 s1, s16, s0 // (op 449) 

s_mul_i32 s0, s8, s45 // (op 449) 
s_add_u32 s0, s0, s1 // (op 449) 


s_mul_hi_i32 s1, s16, s11 // (op 449) 
s_add_i32 s8, s1, s16 // (op 449) 

s_ashr_i32 s1, s8, 31 // (op 449) 
s_cmp_eq_i32 s11, 0 // (op 449) 
s_mov_b32 s9, scc // (op 449) 
s_cmp_eq_i32 s9, 0 // (op 449) 
s_cselect_b32 s18, 0, -1 // (op 449) 

s_and_b32 s9, s46, 31 // (op 449) 
s_lshl_b32 s19, 1, s9 // (op 449) 
s_add_i32 s49, s19, s18 // (op 449) 


s_and_b32 s18, s1, s49 // (op 449) 


s_add_i32 s1, s8, s18 // (op 449) 


s_ashr_i32 s8, s1, s9 // (op 449) 


s_lshl_b32 s1, s46, 24 // (op 449) 
s_ashr_i32 s9, s1, 31 // (op 449) 

s_xor_b32 s1, s8, s9 // (op 449) 

s_sub_i32 s8, s1, s9 // (op 449) 


s_add_u32 s18, s4, 16 // (op 449) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 449) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 449) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 449) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 449) 

s_mul_i32 s1, s45, s18 // (op 449) 

s_mul_i32 s9, s10, s1 // (op 449) 

s_mul_i32 s1, s8, s9 // (op 449) 

s_sub_u32 s9, s16, s1 // (op 449) 

s_mov_b32 s1, s9 // (op 449) 
s_mul_hi_i32 s18, s1, s44 // (op 449) 
s_add_i32 s19, s18, s1 // (op 449) 


s_ashr_i32 s1, s19, 31 // (op 449) 
s_cmp_eq_i32 s44, 0 // (op 449) 
s_mov_b32 s18, scc // (op 449) 
s_cmp_eq_i32 s18, 0 // (op 449) 
s_cselect_b32 s49, 0, -1 // (op 449) 

s_and_b32 s18, s48, 31 // (op 449) 
s_lshl_b32 s50, 1, s18 // (op 449) 
s_add_i32 s51, s50, s49 // (op 449) 


s_and_b32 s49, s1, s51 // (op 449) 


s_add_i32 s1, s19, s49 // (op 449) 


s_ashr_i32 s19, s1, s18 // (op 449) 


s_lshl_b32 s1, s48, 24 // (op 449) 
s_ashr_i32 s18, s1, 31 // (op 449) 

s_xor_b32 s1, s19, s18 // (op 449) 

s_sub_i32 s19, s1, s18 // (op 449) 


s_mul_i32 s1, s47, s45 // (op 449) 
s_add_u32 s50, s6, 16 // (op 449) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 449) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 449) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 449) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 449) 

s_sub_i32 s18, s50, s1 // (op 449) 

s_mul_i32 s49, s19, s18 // (op 449) 


s_sub_u32 s18, s9, s49 // (op 449) 


s_add_u32 s9, s18, s1 // (op 449) 


s_cmp_eq_i32 s8, 0 // (op 449) 
s_mov_b32 s1, scc // (op 449) 

s_cmp_eq_i32 s1, 0 // (op 449) 
s_cselect_b32 s8, s9, s0 // (op 449) 



v_lshl_add_u32 v16, s8, 4, v13 // (op 449) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 449) 
s_lshr_b32 s8, s0, 1 // (op 449) 

v_lshl_add_u32 v13, s8, 8, v15 // (op 449) 


v_mov_b32 v37, 0 // (op 449) convert
v_mov_b32 v36, v13 // (op 449) convert

v_mul_lo_u32 v35, v16, s3 // (op 449) most significant: low of low * high
v_mul_hi_u32 v13, v16, s2 // (op 449) most significant: high of low * low
v_mul_lo_u32 v34, v16, s2 // (op 449) least significant: low of low * low
v_add_u32 v35, v35, v13 // (op 449) most significant: sum

v_add_co_u32 v34, s[0:1], v34, v36 // (op 449) least significant half
v_addc_co_u32 v35, s[0:1], v35, v37, s[0:1] // (op 449) most significant half



v_lshlrev_b64 v[36:37], 2, v[34:35] // (op 449) 

v_lshrrev_b64 v[34:35], 3, v[36:37] // (op 449) 

v_mov_b32 v13, v34 // (op 449) convert

s_mul_i32 s1, s2, s7 // (op 451) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 451) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 451) least significant
s_mul_hi_u32 s9, s2, s6 // (op 451) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 451) most significant: sum
s_add_u32 s1, s1, s9 // (op 451) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 451) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 451) 

s_mov_b32 s52, s20 // (op 451) reg expression
s_mov_b32 s53, s21 // (op 451) reg expression
s_mov_b32 s54, s0 // (op 451) reg expression
s_mov_b32 s55, 131072 // (op 451) call()

s_mul_hi_i32 s0, s16, s12 // (op 516) 
v_lshrrev_b32 v15, 3, v1 // (op 516) 
s_add_i32 s1, s0, s16 // (op 516) 

v_lshlrev_b32 v16, 1, v15 // (op 516) 

s_ashr_i32 s0, s1, 31 // (op 516) 
v_and_b32 v15, -8, v1 // (op 516) 
s_cmp_eq_i32 s12, 0 // (op 516) 
s_mov_b32 s8, scc // (op 516) 
v_sub_u32 v17, v1, v15 // (op 516) 

s_cmp_eq_i32 s8, 0 // (op 516) 
s_cselect_b32 s9, 0, -1 // (op 516) 

v_lshlrev_b32 v15, 4, v17 // (op 516) 

s_and_b32 s8, s13, 31 // (op 516) 
s_lshl_b32 s18, 1, s8 // (op 516) 
s_add_i32 s19, s18, s9 // (op 516) 


s_and_b32 s9, s0, s19 // (op 516) 


s_add_i32 s0, s1, s9 // (op 516) 


s_ashr_i32 s1, s0, s8 // (op 516) 


s_lshl_b32 s0, s13, 24 // (op 516) 
s_ashr_i32 s8, s0, 31 // (op 516) 

s_xor_b32 s0, s1, s8 // (op 516) 

s_sub_i32 s1, s0, s8 // (op 516) 


s_mul_hi_i32 s0, s1, s14 // (op 516) 
s_add_i32 s8, s0, s1 // (op 516) 

s_ashr_i32 s0, s8, 31 // (op 516) 
s_cmp_eq_i32 s14, 0 // (op 516) 
s_mov_b32 s9, scc // (op 516) 
s_cmp_eq_i32 s9, 0 // (op 516) 
s_cselect_b32 s18, 0, -1 // (op 516) 

s_and_b32 s9, s15, 31 // (op 516) 
s_lshl_b32 s19, 1, s9 // (op 516) 
s_add_i32 s49, s19, s18 // (op 516) 


s_and_b32 s18, s0, s49 // (op 516) 


s_add_i32 s0, s8, s18 // (op 516) 


s_ashr_i32 s8, s0, s9 // (op 516) 


s_lshl_b32 s0, s15, 24 // (op 516) 
s_ashr_i32 s9, s0, 31 // (op 516) 

s_xor_b32 s0, s8, s9 // (op 516) 

s_sub_i32 s8, s0, s9 // (op 516) 


s_add_u32 s18, s4, 16 // (op 516) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 516) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 516) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 516) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 516) 

s_mul_i32 s0, s8, s18 // (op 516) 

s_sub_i32 s8, s1, s0 // (op 516) 


s_mul_hi_i32 s9, s16, s11 // (op 516) 
s_add_i32 s0, s9, s16 // (op 516) 

s_ashr_i32 s1, s0, 31 // (op 516) 
s_cmp_eq_i32 s11, 0 // (op 516) 
s_mov_b32 s9, scc // (op 516) 
s_cmp_eq_i32 s9, 0 // (op 516) 
s_cselect_b32 s49, 0, -1 // (op 516) 

s_and_b32 s9, s46, 31 // (op 516) 
s_lshl_b32 s50, 1, s9 // (op 516) 
s_add_i32 s51, s50, s49 // (op 516) 


s_and_b32 s49, s1, s51 // (op 516) 


s_add_i32 s1, s0, s49 // (op 516) 


s_ashr_i32 s0, s1, s9 // (op 516) 


s_lshl_b32 s1, s46, 24 // (op 516) 
s_ashr_i32 s9, s1, 31 // (op 516) 

s_xor_b32 s1, s0, s9 // (op 516) 

s_sub_i32 s0, s1, s9 // (op 516) 


s_mul_i32 s1, s45, s18 // (op 516) 

s_mul_i32 s9, s10, s1 // (op 516) 

s_mul_i32 s1, s0, s9 // (op 516) 

s_sub_u32 s9, s16, s1 // (op 516) 

s_mul_hi_i32 s1, s9, s44 // (op 516) 
s_add_i32 s18, s1, s9 // (op 516) 


s_ashr_i32 s1, s18, 31 // (op 516) 
s_cmp_eq_i32 s44, 0 // (op 516) 
s_mov_b32 s9, scc // (op 516) 
s_cmp_eq_i32 s9, 0 // (op 516) 
s_cselect_b32 s19, 0, -1 // (op 516) 

s_and_b32 s9, s48, 31 // (op 516) 
s_lshl_b32 s49, 1, s9 // (op 516) 
s_add_i32 s50, s49, s19 // (op 516) 


s_and_b32 s19, s1, s50 // (op 516) 


s_add_i32 s1, s18, s19 // (op 516) 


s_ashr_i32 s18, s1, s9 // (op 516) 


s_lshl_b32 s1, s48, 24 // (op 516) 
s_ashr_i32 s9, s1, 31 // (op 516) 

s_xor_b32 s1, s18, s9 // (op 516) 

s_sub_i32 s18, s1, s9 // (op 516) 


s_cmp_eq_i32 s0, 0 // (op 516) 
s_mov_b32 s1, scc // (op 516) 

s_cmp_eq_i32 s1, 0 // (op 516) 
s_cselect_b32 s0, s18, s8 // (op 516) 



v_lshl_add_u32 v17, s0, 4, v16 // (op 516) 


v_mul_lo_u32 v16, v17, s22 // (op 516) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 516) 
s_lshr_b32 s8, s0, 1 // (op 516) 

v_lshl_add_u32 v17, s8, 8, v15 // (op 516) 


v_add_u32 v15, v17, v16 // (op 516) 


s_mul_i32 s1, s22, s5 // (op 518) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 518) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 518) least significant
s_mul_hi_u32 s9, s22, s4 // (op 518) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 518) most significant: sum
s_add_u32 s1, s1, s9 // (op 518) most significant: sum


s_mov_b32 s56, s24 // (op 518) reg expression
s_mov_b32 s57, s25 // (op 518) reg expression
s_mov_b32 s58, s0 // (op 518) reg expression
s_mov_b32 s59, 131072 // (op 518) call()

s_mul_hi_i32 s0, s16, s12 // (op 580) 
v_lshrrev_b32 v16, 2, v1 // (op 580) 
s_add_i32 s1, s0, s16 // (op 580) 

v_and_b32 v17, -4, v1 // (op 580) 
s_ashr_i32 s0, s1, 31 // (op 580) 
v_sub_u32 v18, v1, v17 // (op 580) 

s_cmp_eq_i32 s12, 0 // (op 580) 
s_mov_b32 s8, scc // (op 580) 
s_cmp_eq_i32 s8, 0 // (op 580) 
s_cselect_b32 s9, 0, -1 // (op 580) 

s_and_b32 s8, s13, 31 // (op 580) 
s_lshl_b32 s18, 1, s8 // (op 580) 
s_add_i32 s19, s18, s9 // (op 580) 


s_and_b32 s9, s0, s19 // (op 580) 


s_add_i32 s0, s1, s9 // (op 580) 


s_ashr_i32 s1, s0, s8 // (op 580) 


s_lshl_b32 s0, s13, 24 // (op 580) 
s_ashr_i32 s8, s0, 31 // (op 580) 

s_xor_b32 s0, s1, s8 // (op 580) 

s_sub_i32 s1, s0, s8 // (op 580) 


s_mul_hi_i32 s0, s1, s14 // (op 580) 
s_add_i32 s8, s0, s1 // (op 580) 

s_ashr_i32 s0, s8, 31 // (op 580) 
s_cmp_eq_i32 s14, 0 // (op 580) 
s_mov_b32 s9, scc // (op 580) 
s_cmp_eq_i32 s9, 0 // (op 580) 
s_cselect_b32 s18, 0, -1 // (op 580) 

s_and_b32 s9, s15, 31 // (op 580) 
s_lshl_b32 s19, 1, s9 // (op 580) 
s_add_i32 s49, s19, s18 // (op 580) 


s_and_b32 s18, s0, s49 // (op 580) 


s_add_i32 s0, s8, s18 // (op 580) 


s_ashr_i32 s8, s0, s9 // (op 580) 


s_lshl_b32 s0, s15, 24 // (op 580) 
s_ashr_i32 s9, s0, 31 // (op 580) 

s_xor_b32 s0, s8, s9 // (op 580) 

s_sub_i32 s8, s0, s9 // (op 580) 


s_mul_i32 s0, s1, s45 // (op 580) 

s_sub_u32 s1, s16, s0 // (op 580) 

s_mul_i32 s0, s8, s45 // (op 580) 
s_add_u32 s0, s0, s1 // (op 580) 


s_mul_hi_i32 s1, s16, s11 // (op 580) 
s_add_i32 s8, s1, s16 // (op 580) 

s_ashr_i32 s1, s8, 31 // (op 580) 
s_cmp_eq_i32 s11, 0 // (op 580) 
s_mov_b32 s9, scc // (op 580) 
s_cmp_eq_i32 s9, 0 // (op 580) 
s_cselect_b32 s18, 0, -1 // (op 580) 

s_and_b32 s9, s46, 31 // (op 580) 
s_lshl_b32 s19, 1, s9 // (op 580) 
s_add_i32 s49, s19, s18 // (op 580) 


s_and_b32 s18, s1, s49 // (op 580) 


s_add_i32 s1, s8, s18 // (op 580) 


s_ashr_i32 s8, s1, s9 // (op 580) 


s_lshl_b32 s1, s46, 24 // (op 580) 
s_ashr_i32 s9, s1, 31 // (op 580) 

s_xor_b32 s1, s8, s9 // (op 580) 

s_sub_i32 s8, s1, s9 // (op 580) 


s_add_u32 s18, s4, 16 // (op 580) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 580) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 580) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 580) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 580) 

s_mul_i32 s1, s45, s18 // (op 580) 

s_mul_i32 s9, s10, s1 // (op 580) 

s_mul_i32 s1, s8, s9 // (op 580) 

s_sub_u32 s9, s16, s1 // (op 580) 

s_mov_b32 s1, s9 // (op 580) 
s_mul_hi_i32 s18, s1, s44 // (op 580) 
s_add_i32 s19, s18, s1 // (op 580) 


s_ashr_i32 s1, s19, 31 // (op 580) 
s_cmp_eq_i32 s44, 0 // (op 580) 
s_mov_b32 s18, scc // (op 580) 
s_cmp_eq_i32 s18, 0 // (op 580) 
s_cselect_b32 s49, 0, -1 // (op 580) 

s_and_b32 s18, s48, 31 // (op 580) 
s_lshl_b32 s50, 1, s18 // (op 580) 
s_add_i32 s51, s50, s49 // (op 580) 


s_and_b32 s49, s1, s51 // (op 580) 


s_add_i32 s1, s19, s49 // (op 580) 


s_ashr_i32 s19, s1, s18 // (op 580) 


s_lshl_b32 s1, s48, 24 // (op 580) 
s_ashr_i32 s18, s1, 31 // (op 580) 

s_xor_b32 s1, s19, s18 // (op 580) 

s_sub_i32 s19, s1, s18 // (op 580) 


s_mul_i32 s1, s47, s45 // (op 580) 
s_add_u32 s50, s6, 16 // (op 580) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 580) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 580) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 580) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 580) 

s_sub_i32 s18, s50, s1 // (op 580) 

s_mul_i32 s49, s19, s18 // (op 580) 


s_sub_u32 s18, s9, s49 // (op 580) 


s_add_u32 s9, s18, s1 // (op 580) 


s_cmp_eq_i32 s8, 0 // (op 580) 
s_mov_b32 s1, scc // (op 580) 

s_cmp_eq_i32 s1, 0 // (op 580) 
s_cselect_b32 s8, s9, s0 // (op 580) 



v_lshl_add_u32 v17, s8, 4, v16 // (op 580) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 580) 
s_lshr_b32 s8, s0, 1 // (op 580) 

v_lshl_add_u32 v16, s8, 3, v18 // (op 580) 


v_mul_lo_u32 v18, v17, s26 // (op 580) 
v_add_u32 v18, v18, v16 // (op 580) 


s_mul_i32 s1, s26, s7 // (op 582) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 582) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 582) least significant
s_mul_hi_u32 s9, s26, s6 // (op 582) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 582) most significant: sum
s_add_u32 s1, s1, s9 // (op 582) most significant: sum


s_mov_b32 s60, s28 // (op 582) reg expression
s_mov_b32 s61, s29 // (op 582) reg expression
s_mov_b32 s62, s0 // (op 582) reg expression
s_mov_b32 s63, 131072 // (op 582) call()

s_mul_hi_i32 s0, s16, s12 // (op 644) 
v_lshrrev_b32 v16, 2, v1 // (op 644) 
s_add_i32 s1, s0, s16 // (op 644) 

v_and_b32 v17, -4, v1 // (op 644) 
s_ashr_i32 s0, s1, 31 // (op 644) 
v_sub_u32 v19, v1, v17 // (op 644) 

s_cmp_eq_i32 s12, 0 // (op 644) 
s_mov_b32 s8, scc // (op 644) 
s_cmp_eq_i32 s8, 0 // (op 644) 
s_cselect_b32 s9, 0, -1 // (op 644) 

s_and_b32 s8, s13, 31 // (op 644) 
s_lshl_b32 s18, 1, s8 // (op 644) 
s_add_i32 s19, s18, s9 // (op 644) 


s_and_b32 s9, s0, s19 // (op 644) 


s_add_i32 s0, s1, s9 // (op 644) 


s_ashr_i32 s1, s0, s8 // (op 644) 


s_lshl_b32 s0, s13, 24 // (op 644) 
s_ashr_i32 s8, s0, 31 // (op 644) 

s_xor_b32 s0, s1, s8 // (op 644) 

s_sub_i32 s1, s0, s8 // (op 644) 


s_mul_hi_i32 s0, s1, s14 // (op 644) 
s_add_i32 s8, s0, s1 // (op 644) 

s_ashr_i32 s0, s8, 31 // (op 644) 
s_cmp_eq_i32 s14, 0 // (op 644) 
s_mov_b32 s9, scc // (op 644) 
s_cmp_eq_i32 s9, 0 // (op 644) 
s_cselect_b32 s18, 0, -1 // (op 644) 

s_and_b32 s9, s15, 31 // (op 644) 
s_lshl_b32 s19, 1, s9 // (op 644) 
s_add_i32 s49, s19, s18 // (op 644) 


s_and_b32 s18, s0, s49 // (op 644) 


s_add_i32 s0, s8, s18 // (op 644) 


s_ashr_i32 s8, s0, s9 // (op 644) 


s_lshl_b32 s0, s15, 24 // (op 644) 
s_ashr_i32 s9, s0, 31 // (op 644) 

s_xor_b32 s0, s8, s9 // (op 644) 

s_sub_i32 s8, s0, s9 // (op 644) 


s_add_u32 s18, s4, 16 // (op 644) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 644) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 644) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 644) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 644) 

s_mul_i32 s0, s8, s18 // (op 644) 

s_sub_i32 s8, s1, s0 // (op 644) 


s_mul_hi_i32 s9, s16, s11 // (op 644) 
s_add_i32 s0, s9, s16 // (op 644) 

s_ashr_i32 s1, s0, 31 // (op 644) 
s_cmp_eq_i32 s11, 0 // (op 644) 
s_mov_b32 s9, scc // (op 644) 
s_cmp_eq_i32 s9, 0 // (op 644) 
s_cselect_b32 s49, 0, -1 // (op 644) 

s_and_b32 s9, s46, 31 // (op 644) 
s_lshl_b32 s50, 1, s9 // (op 644) 
s_add_i32 s51, s50, s49 // (op 644) 


s_and_b32 s49, s1, s51 // (op 644) 


s_add_i32 s1, s0, s49 // (op 644) 


s_ashr_i32 s0, s1, s9 // (op 644) 


s_lshl_b32 s1, s46, 24 // (op 644) 
s_ashr_i32 s9, s1, 31 // (op 644) 

s_xor_b32 s1, s0, s9 // (op 644) 

s_sub_i32 s0, s1, s9 // (op 644) 


s_mul_i32 s1, s45, s18 // (op 644) 

s_mul_i32 s9, s10, s1 // (op 644) 

s_mul_i32 s1, s0, s9 // (op 644) 

s_sub_u32 s9, s16, s1 // (op 644) 

s_mul_hi_i32 s1, s9, s44 // (op 644) 
s_add_i32 s18, s1, s9 // (op 644) 


s_ashr_i32 s1, s18, 31 // (op 644) 
s_cmp_eq_i32 s44, 0 // (op 644) 
s_mov_b32 s9, scc // (op 644) 
s_cmp_eq_i32 s9, 0 // (op 644) 
s_cselect_b32 s19, 0, -1 // (op 644) 

s_and_b32 s9, s48, 31 // (op 644) 
s_lshl_b32 s49, 1, s9 // (op 644) 
s_add_i32 s50, s49, s19 // (op 644) 


s_and_b32 s19, s1, s50 // (op 644) 


s_add_i32 s1, s18, s19 // (op 644) 


s_ashr_i32 s18, s1, s9 // (op 644) 


s_lshl_b32 s1, s48, 24 // (op 644) 
s_ashr_i32 s9, s1, 31 // (op 644) 

s_xor_b32 s1, s18, s9 // (op 644) 

s_sub_i32 s18, s1, s9 // (op 644) 


s_cmp_eq_i32 s0, 0 // (op 644) 
s_mov_b32 s1, scc // (op 644) 

s_cmp_eq_i32 s1, 0 // (op 644) 
s_cselect_b32 s0, s18, s8 // (op 644) 



v_lshl_add_u32 v17, s0, 4, v16 // (op 644) 


v_mul_lo_u32 v16, v17, s30 // (op 644) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 644) 
s_lshr_b32 s8, s0, 1 // (op 644) 

v_lshl_add_u32 v17, s8, 3, v19 // (op 644) 


v_add_u32 v19, v17, v16 // (op 644) 


s_mul_i32 s1, s30, s5 // (op 646) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 646) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 646) least significant
s_mul_hi_u32 s9, s30, s4 // (op 646) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 646) most significant: sum
s_add_u32 s1, s1, s9 // (op 646) most significant: sum


s_mov_b32 s64, s32 // (op 646) reg expression
s_mov_b32 s65, s33 // (op 646) reg expression
s_mov_b32 s66, s0 // (op 646) reg expression
s_mov_b32 s67, 131072 // (op 646) call()

v_and_b32 v16, -4, v1 // (op 737) 
v_sub_u32 v17, v1, v16 // (op 737) 

v_lshlrev_b32 v16, 5, v17 // (op 737) 

v_lshrrev_b32 v17, 2, v1 // (op 737) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 737) 


v_lshlrev_b32 v16, 2, v20 // (op 737) 

v_lshrrev_b32 v17, 3, v16 // (op 737) 

v_and_b32 v16, -8, v1 // (op 790) 
v_sub_u32 v20, v1, v16 // (op 790) 

v_lshlrev_b32 v16, 4, v20 // (op 790) 

v_lshrrev_b32 v20, 3, v1 // (op 790) 
v_lshl_add_u32 v21, v20, 8, v16 // (op 790) 


v_and_b32 v16, -4, v1 // (op 843) 
v_sub_u32 v20, v1, v16 // (op 843) 

v_and_b32 v16, 4294967292, v1 // (op 843) 
v_add_u32 v22, v16, v20 // (op 843) 


v_and_b32 v16, -4, v1 // (op 896) 
v_sub_u32 v20, v1, v16 // (op 896) 

v_and_b32 v16, 4294967292, v1 // (op 896) 
v_add_u32 v23, v16, v20 // (op 896) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 172) 
s_and_b32 s8, s0, 4294967294 // (op 172) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 171) 
s_mov_b32 s9, s0 // (op 171) convert

s_cmp_lt_u32 s8, s9 // (op 171) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_171 // (op 171) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_171 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_171:


s_add_u32 s8, s8, 1 // (op 173) 
v_mov_b32 v12, v13 // (op 176) 

buffer_load_dwordx4 v[24:27], v12, s[52:55], 0 offen // (op 176) Load value

v_add_u32 v13, 64, v13 // (op 439) 

s_barrier  // (op 1097) 
v_mov_b64 v[28:29], s[22:23] // (op 190) convert




v_mov_b32 v12, v15 // (op 190) 
v_mov_b32 v14, v12 // (op 190) 
buffer_load_dwordx4 v[30:33], v14, s[56:59], 0 offen // (op 190) Load value
v_add_u32 v14, v14, v28 // (op 190) 
buffer_load_dwordx4 v[34:37], v14, s[56:59], 0 offen // (op 190) Load value



v_add_u32 v15, 128, v15 // (op 508) 
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[24:27] offset:2112 // (op 177) Store local data 



v_mov_b32 v12, v18 // (op 191) 

s_barrier  // (op 1110) 
buffer_load_ubyte v14, v12, s[60:63], 0 offen offset: 0 // (op 191) Load value




v_add_u32 v18, 4, v18 // (op 572) 
s_waitcnt vmcnt(2)
ds_write_b128 v21, v[30:33] // (op 187) Store local data 


v_mov_b32 v12, v19 // (op 192) 

buffer_load_ubyte v16, v12, s[64:67], 0 offen offset: 0 // (op 192) Load value

v_add_u32 v19, 4, v19 // (op 636) 
s_waitcnt vmcnt(2)
ds_write_b128 v21, v[34:37] offset:128 // (op 187) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1100) 



s_barrier  // (op 1124) 
ds_read_b128 v[24:27], v3 offset:2112 // (op 179) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 1114) 
s_waitcnt vmcnt(1)
ds_write_b8 v22, v14 offset:3136 // (op 188) Store local data 




ds_read_b128 v[28:31], v5 // (op 181) Load local data 
s_barrier  // (op 1138) 
ds_read_b128 v[32:35], v5 offset:64 // (op 181) Load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 1128) 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v16 offset:2048 // (op 189) Store local data 




ds_read_u8 v12, v0 offset:3136 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1142) 



ds_read_u8 v14, v6 offset:2048 // (op 185) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[28:35], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 186) 




s_ashr_i64 s[0:1], s[34:35], 7 // (op 171) 
s_mov_b32 s9, s0 // (op 171) convert

s_cmp_lt_u32 s8, s9 // (op 171) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_171 // (op 171) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_171 if true)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_171:






















s_barrier  // (op 1216) 








v_cmp_eq_f32 vcc, 0.00000, v8 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_133
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v11, v4 // (op 50) 
v_mul_f32 v13, v11, v5 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 

s_mul_hi_i32 s0, s16, s12 // (op 938) 
v_lshrrev_b32 v0, 6, v1 // (op 938) 
s_add_i32 s1, s0, s16 // (op 938) 

v_and_b32 v3, 63, v1 // (op 938) 
s_ashr_i32 s0, s1, 31 // (op 938) 
v_lshrrev_b32 v4, 4, v3 // (op 938) 
s_cmp_eq_i32 s12, 0 // (op 938) 
s_mov_b32 s2, scc // (op 938) 
v_lshlrev_b32 v5, 2, v4 // (op 938) 

s_cmp_eq_i32 s2, 0 // (op 938) 
s_cselect_b32 s3, 0, -1 // (op 938) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 938) 


s_and_b32 s2, s13, 31 // (op 938) 
v_lshrrev_b32 v0, 2, v3 // (op 938) 

s_lshl_b32 s8, 1, s2 // (op 938) 
v_and_b32 v3, 3, v0 // (op 938) 

s_add_i32 s9, s8, s3 // (op 938) 


v_and_b32 v0, 3, v1 // (op 938) 
s_and_b32 s3, s0, s9 // (op 938) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 938) 


s_add_i32 s0, s1, s3 // (op 938) 


s_ashr_i32 s1, s0, s2 // (op 938) 


s_lshl_b32 s0, s13, 24 // (op 938) 
s_ashr_i32 s2, s0, 31 // (op 938) 

s_xor_b32 s0, s1, s2 // (op 938) 

s_sub_i32 s1, s0, s2 // (op 938) 


s_mul_hi_i32 s0, s1, s14 // (op 938) 
s_add_i32 s2, s0, s1 // (op 938) 

s_ashr_i32 s0, s2, 31 // (op 938) 
s_cmp_eq_i32 s14, 0 // (op 938) 
s_mov_b32 s3, scc // (op 938) 
s_cmp_eq_i32 s3, 0 // (op 938) 
s_cselect_b32 s8, 0, -1 // (op 938) 

s_and_b32 s3, s15, 31 // (op 938) 
s_lshl_b32 s9, 1, s3 // (op 938) 
s_add_i32 s18, s9, s8 // (op 938) 


s_and_b32 s8, s0, s18 // (op 938) 


s_add_i32 s0, s2, s8 // (op 938) 


s_ashr_i32 s2, s0, s3 // (op 938) 


s_lshl_b32 s0, s15, 24 // (op 938) 
s_ashr_i32 s3, s0, 31 // (op 938) 

s_xor_b32 s0, s2, s3 // (op 938) 

s_sub_i32 s2, s0, s3 // (op 938) 


s_mul_i32 s0, s1, s45 // (op 938) 
s_sub_u32 s3, s16, s0 // (op 938) 

s_mul_i32 s0, s2, s45 // (op 938) 
s_add_u32 s0, s0, s3 // (op 938) 

s_mul_hi_i32 s3, s16, s11 // (op 938) 
s_add_i32 s8, s3, s16 // (op 938) 

s_ashr_i32 s3, s8, 31 // (op 938) 
s_cmp_eq_i32 s11, 0 // (op 938) 
s_mov_b32 s9, scc // (op 938) 
s_cmp_eq_i32 s9, 0 // (op 938) 
s_cselect_b32 s18, 0, -1 // (op 938) 

s_and_b32 s9, s46, 31 // (op 938) 
s_lshl_b32 s19, 1, s9 // (op 938) 
s_add_i32 s20, s19, s18 // (op 938) 


s_and_b32 s18, s3, s20 // (op 938) 


s_add_i32 s3, s8, s18 // (op 938) 


s_ashr_i32 s8, s3, s9 // (op 938) 


s_lshl_b32 s3, s46, 24 // (op 938) 
s_ashr_i32 s9, s3, 31 // (op 938) 

s_xor_b32 s3, s8, s9 // (op 938) 

s_sub_i32 s8, s3, s9 // (op 938) 


s_add_u32 s18, s4, 16 // (op 938) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 938) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 938) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 938) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 938) 

s_mul_i32 s3, s45, s18 // (op 938) 
s_mul_i32 s9, s10, s3 // (op 938) 

s_mul_i32 s3, s8, s9 // (op 938) 

s_sub_u32 s9, s16, s3 // (op 938) 

s_mov_b32 s3, s9 // (op 938) 
s_mul_hi_i32 s20, s3, s44 // (op 938) 
s_add_i32 s21, s20, s3 // (op 938) 


s_ashr_i32 s3, s21, 31 // (op 938) 
s_cmp_eq_i32 s44, 0 // (op 938) 
s_mov_b32 s20, scc // (op 938) 
s_cmp_eq_i32 s20, 0 // (op 938) 
s_cselect_b32 s22, 0, -1 // (op 938) 

s_and_b32 s20, s48, 31 // (op 938) 
s_lshl_b32 s23, 1, s20 // (op 938) 
s_add_i32 s24, s23, s22 // (op 938) 


s_and_b32 s22, s3, s24 // (op 938) 


s_add_i32 s3, s21, s22 // (op 938) 


s_ashr_i32 s21, s3, s20 // (op 938) 


s_lshl_b32 s3, s48, 24 // (op 938) 
s_ashr_i32 s20, s3, 31 // (op 938) 

s_xor_b32 s3, s21, s20 // (op 938) 

s_sub_i32 s21, s3, s20 // (op 938) 


s_mul_i32 s3, s47, s45 // (op 938) 
s_add_u32 s22, s6, 16 // (op 938) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 938) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 938) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 938) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 938) 

s_sub_i32 s20, s22, s3 // (op 938) 

s_mul_i32 s22, s21, s20 // (op 938) 

s_sub_u32 s20, s9, s22 // (op 938) 


s_add_u32 s9, s20, s3 // (op 938) 


s_cmp_eq_i32 s8, 0 // (op 938) 
s_mov_b32 s3, scc // (op 938) 

s_cmp_eq_i32 s3, 0 // (op 938) 
s_cselect_b32 s8, s9, s0 // (op 938) 


s_mul_i32 s0, s2, s18 // (op 938) 


v_lshl_add_u32 v0, s8, 4, v4 // (op 938) 


s_sub_i32 s2, s1, s0 // (op 938) 


s_cmp_eq_i32 s3, 0 // (op 938) 
s_cselect_b32 s0, s21, s2 // (op 938) 



v_lshl_add_u32 v3, s0, 4, v5 // (op 938) 


v_mul_lo_u32 v4, v3, s36 // (op 938) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 938) 


s_mul_i32 s9, s36, s5 // (op 940) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 940) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 940) least significant
s_mul_hi_u32 s1, s36, s4 // (op 940) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 940) most significant: sum
s_add_u32 s9, s9, s1 // (op 940) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 940) 

s_mov_b32 s32, s38 // (op 940) reg expression
s_mov_b32 s33, s39 // (op 940) reg expression
s_mov_b32 s34, s0 // (op 940) reg expression
s_mov_b32 s35, 131072 // (op 940) call()




buffer_store_dwordx4 v[12:15], v3, s[32:35], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_133
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_133:
s_mul_hi_i32 s0, s16, s12 // (op 707) 
v_lshrrev_b32 v0, 6, v1 // (op 707) 
s_add_i32 s1, s0, s16 // (op 707) 

v_and_b32 v3, 63, v1 // (op 707) 
s_ashr_i32 s0, s1, 31 // (op 707) 
v_lshrrev_b32 v4, 4, v3 // (op 707) 
s_cmp_eq_i32 s12, 0 // (op 707) 
s_mov_b32 s2, scc // (op 707) 
v_lshlrev_b32 v5, 2, v4 // (op 707) 

s_cmp_eq_i32 s2, 0 // (op 707) 
s_cselect_b32 s3, 0, -1 // (op 707) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 707) 


s_and_b32 s2, s13, 31 // (op 707) 
v_lshrrev_b32 v0, 2, v3 // (op 707) 

s_lshl_b32 s8, 1, s2 // (op 707) 
v_and_b32 v3, 3, v0 // (op 707) 

s_add_i32 s9, s8, s3 // (op 707) 


v_and_b32 v0, 3, v1 // (op 707) 
s_and_b32 s3, s0, s9 // (op 707) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 707) 


s_add_i32 s0, s1, s3 // (op 707) 


s_ashr_i32 s1, s0, s2 // (op 707) 


s_lshl_b32 s0, s13, 24 // (op 707) 
s_ashr_i32 s2, s0, 31 // (op 707) 

s_xor_b32 s0, s1, s2 // (op 707) 

s_sub_i32 s1, s0, s2 // (op 707) 


s_mul_hi_i32 s0, s1, s14 // (op 707) 
s_add_i32 s2, s0, s1 // (op 707) 

s_ashr_i32 s0, s2, 31 // (op 707) 
s_cmp_eq_i32 s14, 0 // (op 707) 
s_mov_b32 s3, scc // (op 707) 
s_cmp_eq_i32 s3, 0 // (op 707) 
s_cselect_b32 s8, 0, -1 // (op 707) 

s_and_b32 s3, s15, 31 // (op 707) 
s_lshl_b32 s9, 1, s3 // (op 707) 
s_add_i32 s18, s9, s8 // (op 707) 


s_and_b32 s8, s0, s18 // (op 707) 


s_add_i32 s0, s2, s8 // (op 707) 


s_ashr_i32 s2, s0, s3 // (op 707) 


s_lshl_b32 s0, s15, 24 // (op 707) 
s_ashr_i32 s3, s0, 31 // (op 707) 

s_xor_b32 s0, s2, s3 // (op 707) 

s_sub_i32 s2, s0, s3 // (op 707) 


s_mul_i32 s0, s1, s45 // (op 707) 
s_sub_u32 s3, s16, s0 // (op 707) 

s_mul_i32 s0, s2, s45 // (op 707) 
s_add_u32 s0, s0, s3 // (op 707) 

s_mul_hi_i32 s3, s16, s11 // (op 707) 
s_add_i32 s8, s3, s16 // (op 707) 

s_ashr_i32 s3, s8, 31 // (op 707) 
s_cmp_eq_i32 s11, 0 // (op 707) 
s_mov_b32 s9, scc // (op 707) 
s_cmp_eq_i32 s9, 0 // (op 707) 
s_cselect_b32 s18, 0, -1 // (op 707) 

s_and_b32 s9, s46, 31 // (op 707) 
s_lshl_b32 s19, 1, s9 // (op 707) 
s_add_i32 s20, s19, s18 // (op 707) 


s_and_b32 s18, s3, s20 // (op 707) 


s_add_i32 s3, s8, s18 // (op 707) 


s_ashr_i32 s8, s3, s9 // (op 707) 


s_lshl_b32 s3, s46, 24 // (op 707) 
s_ashr_i32 s9, s3, 31 // (op 707) 

s_xor_b32 s3, s8, s9 // (op 707) 

s_sub_i32 s8, s3, s9 // (op 707) 


s_add_u32 s18, s4, 16 // (op 707) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 707) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 707) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 707) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 707) 

s_mul_i32 s3, s45, s18 // (op 707) 
s_mul_i32 s9, s10, s3 // (op 707) 

s_mul_i32 s3, s8, s9 // (op 707) 

s_sub_u32 s9, s16, s3 // (op 707) 

s_mov_b32 s3, s9 // (op 707) 
s_mul_hi_i32 s20, s3, s44 // (op 707) 
s_add_i32 s21, s20, s3 // (op 707) 


s_ashr_i32 s3, s21, 31 // (op 707) 
s_cmp_eq_i32 s44, 0 // (op 707) 
s_mov_b32 s20, scc // (op 707) 
s_cmp_eq_i32 s20, 0 // (op 707) 
s_cselect_b32 s22, 0, -1 // (op 707) 

s_and_b32 s20, s48, 31 // (op 707) 
s_lshl_b32 s23, 1, s20 // (op 707) 
s_add_i32 s24, s23, s22 // (op 707) 


s_and_b32 s22, s3, s24 // (op 707) 


s_add_i32 s3, s21, s22 // (op 707) 


s_ashr_i32 s21, s3, s20 // (op 707) 


s_lshl_b32 s3, s48, 24 // (op 707) 
s_ashr_i32 s20, s3, 31 // (op 707) 

s_xor_b32 s3, s21, s20 // (op 707) 

s_sub_i32 s21, s3, s20 // (op 707) 


s_mul_i32 s3, s47, s45 // (op 707) 
s_add_u32 s22, s6, 16 // (op 707) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 707) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 707) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 707) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 707) 

s_sub_i32 s20, s22, s3 // (op 707) 

s_mul_i32 s22, s21, s20 // (op 707) 

s_sub_u32 s20, s9, s22 // (op 707) 


s_add_u32 s9, s20, s3 // (op 707) 


s_cmp_eq_i32 s8, 0 // (op 707) 
s_mov_b32 s3, scc // (op 707) 

s_cmp_eq_i32 s3, 0 // (op 707) 
s_cselect_b32 s8, s9, s0 // (op 707) 


s_mul_i32 s0, s2, s18 // (op 707) 


v_lshl_add_u32 v0, s8, 4, v4 // (op 707) 


s_sub_i32 s2, s1, s0 // (op 707) 


s_cmp_eq_i32 s3, 0 // (op 707) 
s_cselect_b32 s0, s21, s2 // (op 707) 



v_lshl_add_u32 v3, s0, 4, v5 // (op 707) 


v_mul_lo_u32 v4, v3, s40 // (op 707) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 707) 


s_mul_i32 s9, s40, s5 // (op 709) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 709) most significant: low of high * low
s_mul_i32 s8, s40, s4 // (op 709) least significant
s_mul_hi_u32 s1, s40, s4 // (op 709) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 709) most significant: sum
s_add_u32 s9, s9, s1 // (op 709) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 709) 

s_mov_b32 s28, s42 // (op 709) reg expression
s_mov_b32 s29, s43 // (op 709) reg expression
s_mov_b32 s30, s0 // (op 709) reg expression
s_mov_b32 s31, 131072 // (op 709) call()





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


s_mul_hi_i32 s0, s16, s12 // (op 955) 
v_lshrrev_b32 v0, 6, v1 // (op 955) 
s_add_i32 s1, s0, s16 // (op 955) 

v_and_b32 v3, 63, v1 // (op 955) 
s_ashr_i32 s0, s1, 31 // (op 955) 
v_lshrrev_b32 v9, 4, v3 // (op 955) 
s_cmp_eq_i32 s12, 0 // (op 955) 
s_mov_b32 s2, scc // (op 955) 
v_lshlrev_b32 v10, 2, v9 // (op 955) 

s_cmp_eq_i32 s2, 0 // (op 955) 
s_cselect_b32 s3, 0, -1 // (op 955) 

v_lshl_add_u32 v9, v0, 4, v10 // (op 955) 


s_and_b32 s2, s13, 31 // (op 955) 
v_lshrrev_b32 v0, 2, v3 // (op 955) 

s_lshl_b32 s8, 1, s2 // (op 955) 
v_and_b32 v3, 3, v0 // (op 955) 

s_add_i32 s9, s8, s3 // (op 955) 


v_and_b32 v0, 3, v1 // (op 955) 
s_and_b32 s3, s0, s9 // (op 955) 


v_lshl_add_u32 v10, v3, 2, v0 // (op 955) 


s_add_i32 s0, s1, s3 // (op 955) 


s_ashr_i32 s1, s0, s2 // (op 955) 


s_lshl_b32 s0, s13, 24 // (op 955) 
s_ashr_i32 s2, s0, 31 // (op 955) 

s_xor_b32 s0, s1, s2 // (op 955) 

s_sub_i32 s1, s0, s2 // (op 955) 


s_mul_hi_i32 s0, s1, s14 // (op 955) 
s_add_i32 s2, s0, s1 // (op 955) 

s_ashr_i32 s0, s2, 31 // (op 955) 
s_cmp_eq_i32 s14, 0 // (op 955) 
s_mov_b32 s3, scc // (op 955) 
s_cmp_eq_i32 s3, 0 // (op 955) 
s_cselect_b32 s8, 0, -1 // (op 955) 

s_and_b32 s3, s15, 31 // (op 955) 
s_lshl_b32 s9, 1, s3 // (op 955) 
s_add_i32 s18, s9, s8 // (op 955) 


s_and_b32 s8, s0, s18 // (op 955) 


s_add_i32 s0, s2, s8 // (op 955) 


s_ashr_i32 s2, s0, s3 // (op 955) 


s_lshl_b32 s0, s15, 24 // (op 955) 
s_ashr_i32 s3, s0, 31 // (op 955) 

s_xor_b32 s0, s2, s3 // (op 955) 

s_sub_i32 s2, s0, s3 // (op 955) 


s_mul_i32 s0, s1, s45 // (op 955) 
s_sub_u32 s3, s16, s0 // (op 955) 

s_mul_i32 s0, s2, s45 // (op 955) 
s_add_u32 s0, s0, s3 // (op 955) 

s_mul_hi_i32 s3, s16, s11 // (op 955) 
s_add_i32 s8, s3, s16 // (op 955) 

s_ashr_i32 s3, s8, 31 // (op 955) 
s_cmp_eq_i32 s11, 0 // (op 955) 
s_mov_b32 s9, scc // (op 955) 
s_cmp_eq_i32 s9, 0 // (op 955) 
s_cselect_b32 s18, 0, -1 // (op 955) 

s_and_b32 s9, s46, 31 // (op 955) 
s_lshl_b32 s19, 1, s9 // (op 955) 
s_add_i32 s20, s19, s18 // (op 955) 


s_and_b32 s18, s3, s20 // (op 955) 


s_add_i32 s3, s8, s18 // (op 955) 


s_ashr_i32 s8, s3, s9 // (op 955) 


s_lshl_b32 s3, s46, 24 // (op 955) 
s_ashr_i32 s9, s3, 31 // (op 955) 

s_xor_b32 s3, s8, s9 // (op 955) 

s_sub_i32 s8, s3, s9 // (op 955) 


s_add_u32 s18, s4, 16 // (op 955) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 955) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 955) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 955) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 955) 

s_mul_i32 s3, s45, s18 // (op 955) 
s_mul_i32 s9, s10, s3 // (op 955) 

s_mul_i32 s3, s8, s9 // (op 955) 

s_sub_u32 s9, s16, s3 // (op 955) 

s_mov_b32 s3, s9 // (op 955) 
s_mul_hi_i32 s20, s3, s44 // (op 955) 
s_add_i32 s21, s20, s3 // (op 955) 


s_ashr_i32 s3, s21, 31 // (op 955) 
s_cmp_eq_i32 s44, 0 // (op 955) 
s_mov_b32 s20, scc // (op 955) 
s_cmp_eq_i32 s20, 0 // (op 955) 
s_cselect_b32 s22, 0, -1 // (op 955) 

s_and_b32 s20, s48, 31 // (op 955) 
s_lshl_b32 s23, 1, s20 // (op 955) 
s_add_i32 s24, s23, s22 // (op 955) 


s_and_b32 s22, s3, s24 // (op 955) 


s_add_i32 s3, s21, s22 // (op 955) 


s_ashr_i32 s21, s3, s20 // (op 955) 


s_lshl_b32 s3, s48, 24 // (op 955) 
s_ashr_i32 s20, s3, 31 // (op 955) 

s_xor_b32 s3, s21, s20 // (op 955) 

s_sub_i32 s21, s3, s20 // (op 955) 


s_mul_i32 s3, s47, s45 // (op 955) 
s_add_u32 s22, s6, 16 // (op 955) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 955) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 955) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 955) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 955) 

s_sub_i32 s20, s22, s3 // (op 955) 

s_mul_i32 s22, s21, s20 // (op 955) 

s_sub_u32 s20, s9, s22 // (op 955) 


s_add_u32 s9, s20, s3 // (op 955) 


s_cmp_eq_i32 s8, 0 // (op 955) 
s_mov_b32 s3, scc // (op 955) 

s_cmp_eq_i32 s3, 0 // (op 955) 
s_cselect_b32 s8, s9, s0 // (op 955) 


s_mul_i32 s0, s2, s18 // (op 955) 


v_lshl_add_u32 v0, s8, 4, v9 // (op 955) 


s_sub_i32 s2, s1, s0 // (op 955) 


s_cmp_eq_i32 s3, 0 // (op 955) 
s_cselect_b32 s0, s21, s2 // (op 955) 



v_lshl_add_u32 v3, s0, 4, v10 // (op 955) 


v_mul_lo_u32 v9, v3, s36 // (op 955) 

v_add_lshl_u32 v3, v0, v9, 2 // (op 955) 


s_mul_i32 s9, s36, s5 // (op 957) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 957) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 957) least significant
s_mul_hi_u32 s1, s36, s4 // (op 957) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 957) most significant: sum
s_add_u32 s9, s9, s1 // (op 957) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 957) 

s_mov_b32 s32, s38 // (op 957) reg expression
s_mov_b32 s33, s39 // (op 957) reg expression
s_mov_b32 s34, s0 // (op 957) reg expression
s_mov_b32 s35, 131072 // (op 957) call()




buffer_store_dwordx4 v[4:7], v3, s[32:35], 0 offen // (op 144) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_133:
















s_endpgm  // End of RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.LRR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_, .LRR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end-RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 76
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 6400 // lds bytes
  .amdhsa_accum_offset 72
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
    DataType: F4
    DataTypeA: F4
    DataTypeB: F8
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
  - .name: RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 6400
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 70
    .vgpr_count: 72
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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

