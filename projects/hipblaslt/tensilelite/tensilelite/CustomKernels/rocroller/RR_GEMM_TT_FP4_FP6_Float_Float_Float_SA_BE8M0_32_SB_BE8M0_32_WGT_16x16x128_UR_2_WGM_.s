// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_,@function
RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 765) 
v_and_b32 v3, 15, v1 // (op 765) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 765) 


v_and_b32 v0, 63, v1 // (op 765) 
v_lshrrev_b32 v3, 4, v0 // (op 765) 

v_lshlrev_b32 v0, 5, v3 // (op 765) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 765) 


v_lshlrev_b32 v0, 2, v3 // (op 765) 

v_lshrrev_b32 v3, 3, v0 // (op 765) 

v_and_b32 v0, 63, v1 // (op 818) 
v_and_b32 v4, 48, v1 // (op 818) 
v_sub_u32 v5, v0, v4 // (op 818) 

v_lshrrev_b32 v4, 4, v0 // (op 818) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 818) 


v_lshlrev_b32 v4, 4, v0 // (op 818) 
v_mul_lo_u32 v5, v4, 6 // (op 818) 

v_lshrrev_b32 v4, 3, v5 // (op 818) 

v_and_b32 v5, 268435455, v0 // (op 818) 

v_lshlrev_b32 v0, 2, v5 // (op 818) 

v_add_u32 v5, v4, v0 // (op 818) 


v_lshrrev_b32 v0, 6, v1 // (op 871) 
v_and_b32 v4, 15, v1 // (op 871) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 871) 


v_and_b32 v0, 63, v1 // (op 871) 
v_lshrrev_b32 v4, 4, v0 // (op 871) 

v_lshl_add_u32 v0, v6, 2, v4 // (op 871) 


v_and_b32 v4, 48, v1 // (op 924) 
v_and_b32 v6, 15, v1 // (op 924) 
v_add_u32 v7, v4, v6 // (op 924) 


v_lshrrev_b32 v4, 6, v1 // (op 982) 
v_and_b32 v6, 15, v1 // (op 982) 
v_lshl_add_u32 v8, v4, 4, v6 // (op 982) 


v_and_b32 v4, 63, v1 // (op 982) 
v_lshrrev_b32 v6, 4, v4 // (op 982) 

v_lshlrev_b32 v4, 5, v6 // (op 982) 

v_lshl_add_u32 v6, v8, 7, v4 // (op 982) 


v_lshlrev_b32 v4, 2, v6 // (op 982) 

v_lshrrev_b32 v6, 3, v4 // (op 982) 

v_and_b32 v4, 63, v1 // (op 1006) 
v_and_b32 v8, 48, v1 // (op 1006) 
v_sub_u32 v9, v4, v8 // (op 1006) 

v_lshrrev_b32 v8, 4, v4 // (op 1006) 

v_lshl_add_u32 v4, v8, 5, v9 // (op 1006) 


v_lshlrev_b32 v8, 4, v4 // (op 1006) 
v_mul_lo_u32 v9, v8, 6 // (op 1006) 

v_lshrrev_b32 v8, 3, v9 // (op 1006) 

v_and_b32 v9, 268435455, v4 // (op 1006) 

v_lshlrev_b32 v4, 2, v9 // (op 1006) 

v_add_u32 v9, v8, v4 // (op 1006) 


v_lshrrev_b32 v4, 6, v1 // (op 1030) 
v_and_b32 v8, 15, v1 // (op 1030) 
v_lshl_add_u32 v10, v4, 4, v8 // (op 1030) 


v_and_b32 v4, 63, v1 // (op 1030) 
v_lshrrev_b32 v8, 4, v4 // (op 1030) 

v_lshl_add_u32 v4, v10, 2, v8 // (op 1030) 


v_and_b32 v8, 48, v1 // (op 1054) 
v_and_b32 v10, 15, v1 // (op 1054) 
v_add_u32 v11, v8, v10 // (op 1054) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v10, s9 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_mul_hi_i32 s0, s16, s11 // (op 475) 
v_lshrrev_b32 v12, 2, v1 // (op 475) 
s_add_i32 s1, s0, s16 // (op 475) 

v_and_b32 v13, -4, v1 // (op 475) 
s_ashr_i32 s0, s1, 31 // (op 475) 
v_sub_u32 v14, v1, v13 // (op 475) 

s_cmp_eq_i32 s11, 0 // (op 475) 
s_mov_b32 s8, scc // (op 475) 
v_lshlrev_b32 v13, 5, v14 // (op 475) 

s_cmp_eq_i32 s8, 0 // (op 475) 
s_cselect_b32 s9, 0, -1 // (op 475) 

v_mov_b32 v15, 0 // (op 475) convert
v_mov_b32 v14, v13 // (op 475) convert

s_and_b32 s8, s12, 31 // (op 475) 
s_lshl_b32 s18, 1, s8 // (op 475) 
s_add_i32 s19, s18, s9 // (op 475) 


s_and_b32 s9, s0, s19 // (op 475) 


s_add_i32 s0, s1, s9 // (op 475) 


s_ashr_i32 s1, s0, s8 // (op 475) 


s_lshl_b32 s0, s12, 24 // (op 475) 
s_ashr_i32 s8, s0, 31 // (op 475) 

s_xor_b32 s0, s1, s8 // (op 475) 

s_sub_i32 s1, s0, s8 // (op 475) 


s_mul_hi_i32 s0, s1, s13 // (op 475) 
s_add_i32 s8, s0, s1 // (op 475) 

s_ashr_i32 s0, s8, 31 // (op 475) 
s_cmp_eq_i32 s13, 0 // (op 475) 
s_mov_b32 s9, scc // (op 475) 
s_cmp_eq_i32 s9, 0 // (op 475) 
s_cselect_b32 s18, 0, -1 // (op 475) 

s_and_b32 s9, s14, 31 // (op 475) 
s_lshl_b32 s19, 1, s9 // (op 475) 
s_add_i32 s51, s19, s18 // (op 475) 


s_and_b32 s18, s0, s51 // (op 475) 


s_add_i32 s0, s8, s18 // (op 475) 


s_ashr_i32 s8, s0, s9 // (op 475) 


s_lshl_b32 s0, s14, 24 // (op 475) 
s_ashr_i32 s9, s0, 31 // (op 475) 

s_xor_b32 s0, s8, s9 // (op 475) 

s_sub_i32 s8, s0, s9 // (op 475) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s46 // (op 475) 

s_sub_u32 s1, s16, s0 // (op 475) 

s_mul_i32 s0, s8, s46 // (op 475) 
s_add_u32 s0, s0, s1 // (op 475) 


s_mul_hi_i32 s1, s16, s10 // (op 475) 
s_add_i32 s8, s1, s16 // (op 475) 

s_ashr_i32 s1, s8, 31 // (op 475) 
s_cmp_eq_i32 s10, 0 // (op 475) 
s_mov_b32 s9, scc // (op 475) 
s_cmp_eq_i32 s9, 0 // (op 475) 
s_cselect_b32 s18, 0, -1 // (op 475) 

s_and_b32 s9, s49, 31 // (op 475) 
s_lshl_b32 s19, 1, s9 // (op 475) 
s_add_i32 s51, s19, s18 // (op 475) 


s_and_b32 s18, s1, s51 // (op 475) 


s_add_i32 s1, s8, s18 // (op 475) 


s_ashr_i32 s8, s1, s9 // (op 475) 


s_lshl_b32 s1, s49, 24 // (op 475) 
s_ashr_i32 s9, s1, 31 // (op 475) 

s_xor_b32 s1, s8, s9 // (op 475) 

s_sub_i32 s8, s1, s9 // (op 475) 


s_add_u32 s18, s2, 16 // (op 475) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 475) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 475) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 475) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 475) 

s_mul_i32 s1, s46, s18 // (op 475) 

s_mul_i32 s9, s50, s1 // (op 475) 

s_mul_i32 s1, s8, s9 // (op 475) 

s_sub_u32 s9, s16, s1 // (op 475) 

s_mov_b32 s1, s9 // (op 475) 
s_mul_hi_i32 s18, s1, s15 // (op 475) 
s_add_i32 s19, s18, s1 // (op 475) 


s_ashr_i32 s1, s19, 31 // (op 475) 
s_cmp_eq_i32 s15, 0 // (op 475) 
s_mov_b32 s18, scc // (op 475) 
s_cmp_eq_i32 s18, 0 // (op 475) 
s_cselect_b32 s51, 0, -1 // (op 475) 

s_and_b32 s18, s48, 31 // (op 475) 
s_lshl_b32 s52, 1, s18 // (op 475) 
s_add_i32 s53, s52, s51 // (op 475) 


s_and_b32 s51, s1, s53 // (op 475) 


s_add_i32 s1, s19, s51 // (op 475) 


s_ashr_i32 s19, s1, s18 // (op 475) 


s_lshl_b32 s1, s48, 24 // (op 475) 
s_ashr_i32 s18, s1, 31 // (op 475) 

s_xor_b32 s1, s19, s18 // (op 475) 

s_sub_i32 s19, s1, s18 // (op 475) 


s_mul_i32 s1, s47, s46 // (op 475) 
s_add_u32 s52, s6, 16 // (op 475) Start of Int64 add, locking SCC
s_addc_u32 s53, s7, 0 // (op 475) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 475) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 475) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 475) 

s_sub_i32 s18, s52, s1 // (op 475) 

s_mul_i32 s51, s19, s18 // (op 475) 


s_sub_u32 s18, s9, s51 // (op 475) 


s_add_u32 s9, s18, s1 // (op 475) 


s_cmp_eq_i32 s8, 0 // (op 475) 
s_mov_b32 s1, scc // (op 475) 

s_cmp_eq_i32 s1, 0 // (op 475) 
s_cselect_b32 s8, s9, s0 // (op 475) 



v_lshl_add_u32 v13, s8, 4, v12 // (op 475) 


v_mul_lo_u32 v17, v13, s5 // (op 475) most significant: low of low * high
v_mul_hi_u32 v12, v13, s4 // (op 475) most significant: high of low * low
v_mul_lo_u32 v16, v13, s4 // (op 475) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 475) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 475) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 475) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 475) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 475) 

v_mov_b32 v12, v14 // (op 475) convert

s_mul_i32 s1, s4, s7 // (op 477) most significant: low of low * high
s_mul_i32 s8, s5, s6 // (op 477) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 477) least significant
s_mul_hi_u32 s9, s4, s6 // (op 477) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 477) most significant: sum
s_add_u32 s1, s1, s9 // (op 477) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 477) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 477) 

s_mov_b32 s52, s20 // (op 477) reg expression
s_mov_b32 s53, s21 // (op 477) reg expression
s_mov_b32 s54, s0 // (op 477) reg expression
s_mov_b32 s55, 131072 // (op 477) call()

s_mul_hi_i32 s51, s16, s11 // (op 494) 
v_lshrrev_b32 v13, 2, v1 // (op 494) 
s_add_i32 s0, s51, s16 // (op 494) 

v_and_b32 v14, -4, v1 // (op 494) 
s_ashr_i32 s1, s0, 31 // (op 494) 
v_sub_u32 v15, v1, v14 // (op 494) 

s_cmp_eq_i32 s11, 0 // (op 494) 
s_mov_b32 s51, scc // (op 494) 
v_lshlrev_b32 v14, 5, v15 // (op 494) 

s_cmp_eq_i32 s51, 0 // (op 494) 
s_cselect_b32 s8, 0, -1 // (op 494) 

v_mov_b32 v17, 0 // (op 494) convert
v_mov_b32 v16, v14 // (op 494) convert

s_and_b32 s9, s12, 31 // (op 494) 
s_lshl_b32 s51, 1, s9 // (op 494) 
s_add_i32 s18, s51, s8 // (op 494) 


s_and_b32 s8, s1, s18 // (op 494) 


s_add_i32 s1, s0, s8 // (op 494) 


s_ashr_i32 s0, s1, s9 // (op 494) 


s_lshl_b32 s1, s12, 24 // (op 494) 
s_ashr_i32 s51, s1, 31 // (op 494) 

s_xor_b32 s1, s0, s51 // (op 494) 

s_sub_i32 s0, s1, s51 // (op 494) 


s_mul_hi_i32 s1, s0, s13 // (op 494) 
s_add_i32 s51, s1, s0 // (op 494) 

s_ashr_i32 s1, s51, 31 // (op 494) 
s_cmp_eq_i32 s13, 0 // (op 494) 
s_mov_b32 s8, scc // (op 494) 
s_cmp_eq_i32 s8, 0 // (op 494) 
s_cselect_b32 s9, 0, -1 // (op 494) 

s_and_b32 s8, s14, 31 // (op 494) 
s_lshl_b32 s18, 1, s8 // (op 494) 
s_add_i32 s19, s18, s9 // (op 494) 


s_and_b32 s9, s1, s19 // (op 494) 


s_add_i32 s1, s51, s9 // (op 494) 


s_ashr_i32 s9, s1, s8 // (op 494) 


s_lshl_b32 s1, s14, 24 // (op 494) 
s_ashr_i32 s8, s1, 31 // (op 494) 

s_xor_b32 s1, s9, s8 // (op 494) 

s_sub_i32 s9, s1, s8 // (op 494) 


s_mul_i32 s1, s0, s46 // (op 494) 

s_sub_u32 s0, s16, s1 // (op 494) 

s_mul_i32 s1, s9, s46 // (op 494) 
s_add_u32 s1, s1, s0 // (op 494) 


s_mul_hi_i32 s0, s16, s10 // (op 494) 
s_add_i32 s51, s0, s16 // (op 494) 

s_ashr_i32 s0, s51, 31 // (op 494) 
s_cmp_eq_i32 s10, 0 // (op 494) 
s_mov_b32 s8, scc // (op 494) 
s_cmp_eq_i32 s8, 0 // (op 494) 
s_cselect_b32 s9, 0, -1 // (op 494) 

s_and_b32 s8, s49, 31 // (op 494) 
s_lshl_b32 s18, 1, s8 // (op 494) 
s_add_i32 s19, s18, s9 // (op 494) 


s_and_b32 s9, s0, s19 // (op 494) 


s_add_i32 s0, s51, s9 // (op 494) 


s_ashr_i32 s9, s0, s8 // (op 494) 


s_lshl_b32 s0, s49, 24 // (op 494) 
s_ashr_i32 s8, s0, 31 // (op 494) 

s_xor_b32 s0, s9, s8 // (op 494) 

s_sub_i32 s9, s0, s8 // (op 494) 


s_add_u32 s18, s2, 16 // (op 494) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 494) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 494) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 494) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 494) 

s_mul_i32 s0, s46, s18 // (op 494) 

s_mul_i32 s8, s50, s0 // (op 494) 

s_mul_i32 s0, s9, s8 // (op 494) 

s_sub_u32 s8, s16, s0 // (op 494) 

s_mov_b32 s0, s8 // (op 494) 
s_mul_hi_i32 s51, s0, s15 // (op 494) 
s_add_i32 s18, s51, s0 // (op 494) 


s_ashr_i32 s0, s18, 31 // (op 494) 
s_cmp_eq_i32 s15, 0 // (op 494) 
s_mov_b32 s19, scc // (op 494) 
s_cmp_eq_i32 s19, 0 // (op 494) 
s_cselect_b32 s51, 0, -1 // (op 494) 

s_and_b32 s19, s48, 31 // (op 494) 
s_lshl_b32 s56, 1, s19 // (op 494) 
s_add_i32 s57, s56, s51 // (op 494) 


s_and_b32 s51, s0, s57 // (op 494) 


s_add_i32 s0, s18, s51 // (op 494) 


s_ashr_i32 s18, s0, s19 // (op 494) 


s_lshl_b32 s0, s48, 24 // (op 494) 
s_ashr_i32 s19, s0, 31 // (op 494) 

s_xor_b32 s0, s18, s19 // (op 494) 

s_sub_i32 s18, s0, s19 // (op 494) 


s_mul_i32 s0, s47, s46 // (op 494) 
s_add_u32 s56, s6, 16 // (op 494) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 494) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 494) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 494) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 494) 

s_sub_i32 s19, s56, s0 // (op 494) 

s_mul_i32 s51, s18, s19 // (op 494) 


s_sub_u32 s18, s8, s51 // (op 494) 


s_add_u32 s8, s18, s0 // (op 494) 


s_cmp_eq_i32 s9, 0 // (op 494) 
s_mov_b32 s0, scc // (op 494) 

s_cmp_eq_i32 s0, 0 // (op 494) 
s_cselect_b32 s9, s8, s1 // (op 494) 



v_lshl_add_u32 v14, s9, 4, v13 // (op 494) 


v_mul_lo_u32 v19, v14, s5 // (op 494) most significant: low of low * high
v_mul_hi_u32 v13, v14, s4 // (op 494) most significant: high of low * low
v_mul_lo_u32 v18, v14, s4 // (op 494) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 494) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 494) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 494) most significant half



v_lshlrev_b64 v[16:17], 2, v[18:19] // (op 494) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 494) 

v_mov_b32 v13, v14 // (op 494) convert

s_mul_i32 s1, s4, s7 // (op 496) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 496) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 496) least significant
s_mul_hi_u32 s8, s4, s6 // (op 496) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 496) most significant: sum
s_add_u32 s1, s1, s8 // (op 496) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 496) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 496) 

s_mov_b32 s52, s20 // (op 496) reg expression
s_mov_b32 s53, s21 // (op 496) reg expression
s_mov_b32 s54, s0 // (op 496) reg expression
s_mov_b32 s55, 131072 // (op 496) call()

s_mul_hi_i32 s51, s16, s11 // (op 539) 
v_lshlrev_b32 v14, 1, v1 // (op 539) 
s_add_i32 s0, s51, s16 // (op 539) 

s_ashr_i32 s1, s0, 31 // (op 539) 
s_cmp_eq_i32 s11, 0 // (op 539) 
s_mov_b32 s51, scc // (op 539) 
s_cmp_eq_i32 s51, 0 // (op 539) 
s_cselect_b32 s8, 0, -1 // (op 539) 

s_and_b32 s9, s12, 31 // (op 539) 
s_lshl_b32 s51, 1, s9 // (op 539) 
s_add_i32 s18, s51, s8 // (op 539) 


s_and_b32 s8, s1, s18 // (op 539) 


s_add_i32 s1, s0, s8 // (op 539) 


s_ashr_i32 s0, s1, s9 // (op 539) 


s_lshl_b32 s1, s12, 24 // (op 539) 
s_ashr_i32 s51, s1, 31 // (op 539) 

s_xor_b32 s1, s0, s51 // (op 539) 

s_sub_i32 s0, s1, s51 // (op 539) 


s_mul_hi_i32 s1, s0, s13 // (op 539) 
s_add_i32 s51, s1, s0 // (op 539) 

s_ashr_i32 s1, s51, 31 // (op 539) 
s_cmp_eq_i32 s13, 0 // (op 539) 
s_mov_b32 s8, scc // (op 539) 
s_cmp_eq_i32 s8, 0 // (op 539) 
s_cselect_b32 s9, 0, -1 // (op 539) 

s_and_b32 s8, s14, 31 // (op 539) 
s_lshl_b32 s18, 1, s8 // (op 539) 
s_add_i32 s19, s18, s9 // (op 539) 


s_and_b32 s9, s1, s19 // (op 539) 


s_add_i32 s1, s51, s9 // (op 539) 


s_ashr_i32 s9, s1, s8 // (op 539) 


s_lshl_b32 s1, s14, 24 // (op 539) 
s_ashr_i32 s8, s1, 31 // (op 539) 

s_xor_b32 s1, s9, s8 // (op 539) 

s_sub_i32 s9, s1, s8 // (op 539) 


s_add_u32 s18, s2, 16 // (op 539) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 539) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 539) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 539) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 539) 

s_mul_i32 s1, s9, s18 // (op 539) 

s_sub_i32 s51, s0, s1 // (op 539) 


s_mul_hi_i32 s0, s16, s10 // (op 539) 
s_add_i32 s1, s0, s16 // (op 539) 

s_ashr_i32 s0, s1, 31 // (op 539) 
s_cmp_eq_i32 s10, 0 // (op 539) 
s_mov_b32 s8, scc // (op 539) 
s_cmp_eq_i32 s8, 0 // (op 539) 
s_cselect_b32 s9, 0, -1 // (op 539) 

s_and_b32 s8, s49, 31 // (op 539) 
s_lshl_b32 s56, 1, s8 // (op 539) 
s_add_i32 s57, s56, s9 // (op 539) 


s_and_b32 s9, s0, s57 // (op 539) 


s_add_i32 s0, s1, s9 // (op 539) 


s_ashr_i32 s1, s0, s8 // (op 539) 


s_lshl_b32 s0, s49, 24 // (op 539) 
s_ashr_i32 s8, s0, 31 // (op 539) 

s_xor_b32 s0, s1, s8 // (op 539) 

s_sub_i32 s1, s0, s8 // (op 539) 


s_mul_i32 s0, s46, s18 // (op 539) 

s_mul_i32 s8, s50, s0 // (op 539) 

s_mul_i32 s0, s1, s8 // (op 539) 

s_sub_u32 s8, s16, s0 // (op 539) 

s_mul_hi_i32 s0, s8, s15 // (op 539) 
s_add_i32 s9, s0, s8 // (op 539) 


s_ashr_i32 s0, s9, 31 // (op 539) 
s_cmp_eq_i32 s15, 0 // (op 539) 
s_mov_b32 s8, scc // (op 539) 
s_cmp_eq_i32 s8, 0 // (op 539) 
s_cselect_b32 s18, 0, -1 // (op 539) 

s_and_b32 s8, s48, 31 // (op 539) 
s_lshl_b32 s19, 1, s8 // (op 539) 
s_add_i32 s56, s19, s18 // (op 539) 


s_and_b32 s18, s0, s56 // (op 539) 


s_add_i32 s0, s9, s18 // (op 539) 


s_ashr_i32 s9, s0, s8 // (op 539) 


s_lshl_b32 s0, s48, 24 // (op 539) 
s_ashr_i32 s8, s0, 31 // (op 539) 

s_xor_b32 s0, s9, s8 // (op 539) 

s_sub_i32 s9, s0, s8 // (op 539) 


s_cmp_eq_i32 s1, 0 // (op 539) 
s_mov_b32 s0, scc // (op 539) 

s_cmp_eq_i32 s0, 0 // (op 539) 
s_cselect_b32 s1, s9, s51 // (op 539) 



s_lshl_b32 s0, s1, 4 // (op 539) 

s_mov_b32 s9, 0 // (op 539) convert
s_mov_b32 s8, s0 // (op 539) convert

v_mul_lo_u32 v17, v14, s23 // (op 539) most significant: low of low * high
v_mul_hi_u32 v15, v14, s22 // (op 539) most significant: high of low * low
v_mul_lo_u32 v16, v14, s22 // (op 539) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 539) most significant: sum

v_mov_b32 v15, s8 // (op 539) 
v_mov_b32 v18, s9 // (op 539) 
v_add_co_u32 v16, s[0:1], v16, v15 // (op 539) least significant half
v_addc_co_u32 v17, s[0:1], v17, v18, s[0:1] // (op 539) most significant half





v_mul_lo_u32 v15, v17, 6 // (op 539) most significant: low of high * low
v_mul_hi_u32 v18, v16, 6 // (op 539) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 539) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 539) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 539) 

v_mov_b32 v14, v16 // (op 539) convert

s_mul_i32 s1, s27, 6 // (op 541) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 541) least significant
s_mul_hi_u32 s51, s26, 6 // (op 541) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 541) most significant: sum

s_mul_i32 s9, s22, s1 // (op 541) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 541) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 541) least significant
s_mul_hi_u32 s18, s22, s0 // (op 541) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 541) most significant: sum
s_add_u32 s9, s9, s18 // (op 541) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 541) 

s_mov_b32 s56, s24 // (op 541) reg expression
s_mov_b32 s57, s25 // (op 541) reg expression
s_mov_b32 s58, s0 // (op 541) reg expression
s_mov_b32 s59, 131072 // (op 541) call()

s_mul_hi_i32 s51, s16, s11 // (op 558) 
v_lshlrev_b32 v15, 1, v1 // (op 558) 
s_add_i32 s0, s51, s16 // (op 558) 

s_ashr_i32 s1, s0, 31 // (op 558) 
s_cmp_eq_i32 s11, 0 // (op 558) 
s_mov_b32 s51, scc // (op 558) 
s_cmp_eq_i32 s51, 0 // (op 558) 
s_cselect_b32 s8, 0, -1 // (op 558) 

s_and_b32 s9, s12, 31 // (op 558) 
s_lshl_b32 s51, 1, s9 // (op 558) 
s_add_i32 s18, s51, s8 // (op 558) 


s_and_b32 s8, s1, s18 // (op 558) 


s_add_i32 s1, s0, s8 // (op 558) 


s_ashr_i32 s0, s1, s9 // (op 558) 


s_lshl_b32 s1, s12, 24 // (op 558) 
s_ashr_i32 s51, s1, 31 // (op 558) 

s_xor_b32 s1, s0, s51 // (op 558) 

s_sub_i32 s0, s1, s51 // (op 558) 


s_mul_hi_i32 s1, s0, s13 // (op 558) 
s_add_i32 s51, s1, s0 // (op 558) 

s_ashr_i32 s1, s51, 31 // (op 558) 
s_cmp_eq_i32 s13, 0 // (op 558) 
s_mov_b32 s8, scc // (op 558) 
s_cmp_eq_i32 s8, 0 // (op 558) 
s_cselect_b32 s9, 0, -1 // (op 558) 

s_and_b32 s8, s14, 31 // (op 558) 
s_lshl_b32 s18, 1, s8 // (op 558) 
s_add_i32 s19, s18, s9 // (op 558) 


s_and_b32 s9, s1, s19 // (op 558) 


s_add_i32 s1, s51, s9 // (op 558) 


s_ashr_i32 s9, s1, s8 // (op 558) 


s_lshl_b32 s1, s14, 24 // (op 558) 
s_ashr_i32 s8, s1, 31 // (op 558) 

s_xor_b32 s1, s9, s8 // (op 558) 

s_sub_i32 s9, s1, s8 // (op 558) 


s_add_u32 s18, s2, 16 // (op 558) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 558) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 558) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 558) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 558) 

s_mul_i32 s1, s9, s18 // (op 558) 

s_sub_i32 s51, s0, s1 // (op 558) 


s_mul_hi_i32 s0, s16, s10 // (op 558) 
s_add_i32 s1, s0, s16 // (op 558) 

s_ashr_i32 s0, s1, 31 // (op 558) 
s_cmp_eq_i32 s10, 0 // (op 558) 
s_mov_b32 s8, scc // (op 558) 
s_cmp_eq_i32 s8, 0 // (op 558) 
s_cselect_b32 s9, 0, -1 // (op 558) 

s_and_b32 s8, s49, 31 // (op 558) 
s_lshl_b32 s60, 1, s8 // (op 558) 
s_add_i32 s61, s60, s9 // (op 558) 


s_and_b32 s9, s0, s61 // (op 558) 


s_add_i32 s0, s1, s9 // (op 558) 


s_ashr_i32 s1, s0, s8 // (op 558) 


s_lshl_b32 s0, s49, 24 // (op 558) 
s_ashr_i32 s8, s0, 31 // (op 558) 

s_xor_b32 s0, s1, s8 // (op 558) 

s_sub_i32 s1, s0, s8 // (op 558) 


s_mul_i32 s0, s46, s18 // (op 558) 

s_mul_i32 s8, s50, s0 // (op 558) 

s_mul_i32 s0, s1, s8 // (op 558) 

s_sub_u32 s8, s16, s0 // (op 558) 

s_mul_hi_i32 s0, s8, s15 // (op 558) 
s_add_i32 s9, s0, s8 // (op 558) 


s_ashr_i32 s0, s9, 31 // (op 558) 
s_cmp_eq_i32 s15, 0 // (op 558) 
s_mov_b32 s8, scc // (op 558) 
s_cmp_eq_i32 s8, 0 // (op 558) 
s_cselect_b32 s18, 0, -1 // (op 558) 

s_and_b32 s8, s48, 31 // (op 558) 
s_lshl_b32 s19, 1, s8 // (op 558) 
s_add_i32 s60, s19, s18 // (op 558) 


s_and_b32 s18, s0, s60 // (op 558) 


s_add_i32 s0, s9, s18 // (op 558) 


s_ashr_i32 s9, s0, s8 // (op 558) 


s_lshl_b32 s0, s48, 24 // (op 558) 
s_ashr_i32 s8, s0, 31 // (op 558) 

s_xor_b32 s0, s9, s8 // (op 558) 

s_sub_i32 s9, s0, s8 // (op 558) 


s_cmp_eq_i32 s1, 0 // (op 558) 
s_mov_b32 s0, scc // (op 558) 

s_cmp_eq_i32 s0, 0 // (op 558) 
s_cselect_b32 s1, s9, s51 // (op 558) 



s_lshl_b32 s0, s1, 4 // (op 558) 

s_mov_b32 s9, 0 // (op 558) convert
s_mov_b32 s8, s0 // (op 558) convert

v_mul_lo_u32 v17, v15, s23 // (op 558) most significant: low of low * high
v_mul_hi_u32 v18, v15, s22 // (op 558) most significant: high of low * low
v_mul_lo_u32 v16, v15, s22 // (op 558) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 558) most significant: sum

v_mov_b32 v18, s8 // (op 558) 
v_mov_b32 v19, s9 // (op 558) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 558) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 558) most significant half





v_mul_lo_u32 v19, v17, 6 // (op 558) most significant: low of high * low
v_mul_hi_u32 v15, v16, 6 // (op 558) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 558) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 558) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 558) 

v_mov_b32 v15, v16 // (op 558) convert

s_mul_i32 s1, s27, 6 // (op 560) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 560) least significant
s_mul_hi_u32 s51, s26, 6 // (op 560) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 560) most significant: sum

s_mul_i32 s9, s22, s1 // (op 560) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 560) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 560) least significant
s_mul_hi_u32 s18, s22, s0 // (op 560) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 560) most significant: sum
s_add_u32 s9, s9, s18 // (op 560) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 560) 

s_mov_b32 s56, s24 // (op 560) reg expression
s_mov_b32 s57, s25 // (op 560) reg expression
s_mov_b32 s58, s0 // (op 560) reg expression
s_mov_b32 s59, 131072 // (op 560) call()

s_mul_hi_i32 s51, s16, s11 // (op 603) 
v_lshrrev_b32 v16, 2, v1 // (op 603) 
s_add_i32 s0, s51, s16 // (op 603) 

v_and_b32 v17, -4, v1 // (op 603) 
s_ashr_i32 s1, s0, 31 // (op 603) 
v_sub_u32 v18, v1, v17 // (op 603) 

s_cmp_eq_i32 s11, 0 // (op 603) 
s_mov_b32 s51, scc // (op 603) 
s_cmp_eq_i32 s51, 0 // (op 603) 
s_cselect_b32 s8, 0, -1 // (op 603) 

s_and_b32 s9, s12, 31 // (op 603) 
s_lshl_b32 s51, 1, s9 // (op 603) 
s_add_i32 s18, s51, s8 // (op 603) 


s_and_b32 s8, s1, s18 // (op 603) 


s_add_i32 s1, s0, s8 // (op 603) 


s_ashr_i32 s0, s1, s9 // (op 603) 


s_lshl_b32 s1, s12, 24 // (op 603) 
s_ashr_i32 s51, s1, 31 // (op 603) 

s_xor_b32 s1, s0, s51 // (op 603) 

s_sub_i32 s0, s1, s51 // (op 603) 


s_mul_hi_i32 s1, s0, s13 // (op 603) 
s_add_i32 s51, s1, s0 // (op 603) 

s_ashr_i32 s1, s51, 31 // (op 603) 
s_cmp_eq_i32 s13, 0 // (op 603) 
s_mov_b32 s8, scc // (op 603) 
s_cmp_eq_i32 s8, 0 // (op 603) 
s_cselect_b32 s9, 0, -1 // (op 603) 

s_and_b32 s8, s14, 31 // (op 603) 
s_lshl_b32 s18, 1, s8 // (op 603) 
s_add_i32 s19, s18, s9 // (op 603) 


s_and_b32 s9, s1, s19 // (op 603) 


s_add_i32 s1, s51, s9 // (op 603) 


s_ashr_i32 s9, s1, s8 // (op 603) 


s_lshl_b32 s1, s14, 24 // (op 603) 
s_ashr_i32 s8, s1, 31 // (op 603) 

s_xor_b32 s1, s9, s8 // (op 603) 

s_sub_i32 s9, s1, s8 // (op 603) 


s_mul_i32 s1, s0, s46 // (op 603) 

s_sub_u32 s0, s16, s1 // (op 603) 

s_mul_i32 s1, s9, s46 // (op 603) 
s_add_u32 s1, s1, s0 // (op 603) 


s_mul_hi_i32 s0, s16, s10 // (op 603) 
s_add_i32 s51, s0, s16 // (op 603) 

s_ashr_i32 s0, s51, 31 // (op 603) 
s_cmp_eq_i32 s10, 0 // (op 603) 
s_mov_b32 s8, scc // (op 603) 
s_cmp_eq_i32 s8, 0 // (op 603) 
s_cselect_b32 s9, 0, -1 // (op 603) 

s_and_b32 s8, s49, 31 // (op 603) 
s_lshl_b32 s18, 1, s8 // (op 603) 
s_add_i32 s19, s18, s9 // (op 603) 


s_and_b32 s9, s0, s19 // (op 603) 


s_add_i32 s0, s51, s9 // (op 603) 


s_ashr_i32 s9, s0, s8 // (op 603) 


s_lshl_b32 s0, s49, 24 // (op 603) 
s_ashr_i32 s8, s0, 31 // (op 603) 

s_xor_b32 s0, s9, s8 // (op 603) 

s_sub_i32 s9, s0, s8 // (op 603) 


s_add_u32 s18, s2, 16 // (op 603) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 603) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 603) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 603) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 603) 

s_mul_i32 s0, s46, s18 // (op 603) 

s_mul_i32 s8, s50, s0 // (op 603) 

s_mul_i32 s0, s9, s8 // (op 603) 

s_sub_u32 s8, s16, s0 // (op 603) 

s_mov_b32 s0, s8 // (op 603) 
s_mul_hi_i32 s51, s0, s15 // (op 603) 
s_add_i32 s18, s51, s0 // (op 603) 


s_ashr_i32 s0, s18, 31 // (op 603) 
s_cmp_eq_i32 s15, 0 // (op 603) 
s_mov_b32 s19, scc // (op 603) 
s_cmp_eq_i32 s19, 0 // (op 603) 
s_cselect_b32 s51, 0, -1 // (op 603) 

s_and_b32 s19, s48, 31 // (op 603) 
s_lshl_b32 s60, 1, s19 // (op 603) 
s_add_i32 s61, s60, s51 // (op 603) 


s_and_b32 s51, s0, s61 // (op 603) 


s_add_i32 s0, s18, s51 // (op 603) 


s_ashr_i32 s18, s0, s19 // (op 603) 


s_lshl_b32 s0, s48, 24 // (op 603) 
s_ashr_i32 s19, s0, 31 // (op 603) 

s_xor_b32 s0, s18, s19 // (op 603) 

s_sub_i32 s18, s0, s19 // (op 603) 


s_mul_i32 s0, s47, s46 // (op 603) 
s_add_u32 s60, s6, 16 // (op 603) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 603) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 603) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 603) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 603) 

s_sub_i32 s19, s60, s0 // (op 603) 

s_mul_i32 s51, s18, s19 // (op 603) 


s_sub_u32 s18, s8, s51 // (op 603) 


s_add_u32 s8, s18, s0 // (op 603) 


s_cmp_eq_i32 s9, 0 // (op 603) 
s_mov_b32 s0, scc // (op 603) 

s_cmp_eq_i32 s0, 0 // (op 603) 
s_cselect_b32 s9, s8, s1 // (op 603) 



v_lshl_add_u32 v17, s9, 4, v16 // (op 603) 


v_mul_lo_u32 v16, v17, s28 // (op 603) 
v_add_u32 v16, v16, v18 // (op 603) 


s_mul_i32 s1, s28, s7 // (op 605) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 605) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 605) least significant
s_mul_hi_u32 s8, s28, s6 // (op 605) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 605) most significant: sum
s_add_u32 s1, s1, s8 // (op 605) most significant: sum


s_mov_b32 s60, s30 // (op 605) reg expression
s_mov_b32 s61, s31 // (op 605) reg expression
s_mov_b32 s62, s0 // (op 605) reg expression
s_mov_b32 s63, 131072 // (op 605) call()

s_mul_hi_i32 s51, s16, s11 // (op 622) 
v_lshrrev_b32 v17, 2, v1 // (op 622) 
s_add_i32 s0, s51, s16 // (op 622) 

v_and_b32 v18, -4, v1 // (op 622) 
s_ashr_i32 s1, s0, 31 // (op 622) 
v_sub_u32 v19, v1, v18 // (op 622) 

s_cmp_eq_i32 s11, 0 // (op 622) 
s_mov_b32 s51, scc // (op 622) 
s_cmp_eq_i32 s51, 0 // (op 622) 
s_cselect_b32 s8, 0, -1 // (op 622) 

s_and_b32 s9, s12, 31 // (op 622) 
s_lshl_b32 s51, 1, s9 // (op 622) 
s_add_i32 s18, s51, s8 // (op 622) 


s_and_b32 s8, s1, s18 // (op 622) 


s_add_i32 s1, s0, s8 // (op 622) 


s_ashr_i32 s0, s1, s9 // (op 622) 


s_lshl_b32 s1, s12, 24 // (op 622) 
s_ashr_i32 s51, s1, 31 // (op 622) 

s_xor_b32 s1, s0, s51 // (op 622) 

s_sub_i32 s0, s1, s51 // (op 622) 


s_mul_hi_i32 s1, s0, s13 // (op 622) 
s_add_i32 s51, s1, s0 // (op 622) 

s_ashr_i32 s1, s51, 31 // (op 622) 
s_cmp_eq_i32 s13, 0 // (op 622) 
s_mov_b32 s8, scc // (op 622) 
s_cmp_eq_i32 s8, 0 // (op 622) 
s_cselect_b32 s9, 0, -1 // (op 622) 

s_and_b32 s8, s14, 31 // (op 622) 
s_lshl_b32 s18, 1, s8 // (op 622) 
s_add_i32 s19, s18, s9 // (op 622) 


s_and_b32 s9, s1, s19 // (op 622) 


s_add_i32 s1, s51, s9 // (op 622) 


s_ashr_i32 s9, s1, s8 // (op 622) 


s_lshl_b32 s1, s14, 24 // (op 622) 
s_ashr_i32 s8, s1, 31 // (op 622) 

s_xor_b32 s1, s9, s8 // (op 622) 

s_sub_i32 s9, s1, s8 // (op 622) 


s_mul_i32 s1, s0, s46 // (op 622) 

s_sub_u32 s0, s16, s1 // (op 622) 

s_mul_i32 s1, s9, s46 // (op 622) 
s_add_u32 s1, s1, s0 // (op 622) 


s_mul_hi_i32 s0, s16, s10 // (op 622) 
s_add_i32 s51, s0, s16 // (op 622) 

s_ashr_i32 s0, s51, 31 // (op 622) 
s_cmp_eq_i32 s10, 0 // (op 622) 
s_mov_b32 s8, scc // (op 622) 
s_cmp_eq_i32 s8, 0 // (op 622) 
s_cselect_b32 s9, 0, -1 // (op 622) 

s_and_b32 s8, s49, 31 // (op 622) 
s_lshl_b32 s18, 1, s8 // (op 622) 
s_add_i32 s19, s18, s9 // (op 622) 


s_and_b32 s9, s0, s19 // (op 622) 


s_add_i32 s0, s51, s9 // (op 622) 


s_ashr_i32 s9, s0, s8 // (op 622) 


s_lshl_b32 s0, s49, 24 // (op 622) 
s_ashr_i32 s8, s0, 31 // (op 622) 

s_xor_b32 s0, s9, s8 // (op 622) 

s_sub_i32 s9, s0, s8 // (op 622) 


s_add_u32 s18, s2, 16 // (op 622) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 622) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 622) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 622) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 622) 

s_mul_i32 s0, s46, s18 // (op 622) 

s_mul_i32 s8, s50, s0 // (op 622) 

s_mul_i32 s0, s9, s8 // (op 622) 

s_sub_u32 s8, s16, s0 // (op 622) 

s_mov_b32 s0, s8 // (op 622) 
s_mul_hi_i32 s51, s0, s15 // (op 622) 
s_add_i32 s18, s51, s0 // (op 622) 


s_ashr_i32 s0, s18, 31 // (op 622) 
s_cmp_eq_i32 s15, 0 // (op 622) 
s_mov_b32 s19, scc // (op 622) 
s_cmp_eq_i32 s19, 0 // (op 622) 
s_cselect_b32 s51, 0, -1 // (op 622) 

s_and_b32 s19, s48, 31 // (op 622) 
s_lshl_b32 s64, 1, s19 // (op 622) 
s_add_i32 s65, s64, s51 // (op 622) 


s_and_b32 s51, s0, s65 // (op 622) 


s_add_i32 s0, s18, s51 // (op 622) 


s_ashr_i32 s18, s0, s19 // (op 622) 


s_lshl_b32 s0, s48, 24 // (op 622) 
s_ashr_i32 s19, s0, 31 // (op 622) 

s_xor_b32 s0, s18, s19 // (op 622) 

s_sub_i32 s18, s0, s19 // (op 622) 


s_mul_i32 s0, s47, s46 // (op 622) 
s_add_u32 s64, s6, 16 // (op 622) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 622) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 622) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 622) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 622) 

s_sub_i32 s19, s64, s0 // (op 622) 

s_mul_i32 s51, s18, s19 // (op 622) 


s_sub_u32 s18, s8, s51 // (op 622) 


s_add_u32 s8, s18, s0 // (op 622) 


s_cmp_eq_i32 s9, 0 // (op 622) 
s_mov_b32 s0, scc // (op 622) 

s_cmp_eq_i32 s0, 0 // (op 622) 
s_cselect_b32 s9, s8, s1 // (op 622) 



v_lshl_add_u32 v18, s9, 4, v17 // (op 622) 


v_mul_lo_u32 v17, v18, s28 // (op 622) 
v_add_u32 v17, v17, v19 // (op 622) 


s_mul_i32 s1, s28, s7 // (op 624) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 624) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 624) least significant
s_mul_hi_u32 s8, s28, s6 // (op 624) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 624) most significant: sum
s_add_u32 s1, s1, s8 // (op 624) most significant: sum


s_mov_b32 s60, s30 // (op 624) reg expression
s_mov_b32 s61, s31 // (op 624) reg expression
s_mov_b32 s62, s0 // (op 624) reg expression
s_mov_b32 s63, 131072 // (op 624) call()

s_mul_hi_i32 s51, s16, s11 // (op 667) 
v_and_b32 v18, -16, v1 // (op 667) 
s_add_i32 s0, s51, s16 // (op 667) 

v_sub_u32 v19, v1, v18 // (op 667) 

s_ashr_i32 s1, s0, 31 // (op 667) 
v_lshrrev_b32 v18, 4, v1 // (op 667) 
s_cmp_eq_i32 s11, 0 // (op 667) 
s_mov_b32 s51, scc // (op 667) 
s_cmp_eq_i32 s51, 0 // (op 667) 
s_cselect_b32 s8, 0, -1 // (op 667) 

s_and_b32 s9, s12, 31 // (op 667) 
s_lshl_b32 s51, 1, s9 // (op 667) 
s_add_i32 s18, s51, s8 // (op 667) 


s_and_b32 s8, s1, s18 // (op 667) 


s_add_i32 s1, s0, s8 // (op 667) 


s_ashr_i32 s0, s1, s9 // (op 667) 


s_lshl_b32 s1, s12, 24 // (op 667) 
s_ashr_i32 s51, s1, 31 // (op 667) 

s_xor_b32 s1, s0, s51 // (op 667) 

s_sub_i32 s0, s1, s51 // (op 667) 


s_mul_hi_i32 s1, s0, s13 // (op 667) 
s_add_i32 s51, s1, s0 // (op 667) 

s_ashr_i32 s1, s51, 31 // (op 667) 
s_cmp_eq_i32 s13, 0 // (op 667) 
s_mov_b32 s8, scc // (op 667) 
s_cmp_eq_i32 s8, 0 // (op 667) 
s_cselect_b32 s9, 0, -1 // (op 667) 

s_and_b32 s8, s14, 31 // (op 667) 
s_lshl_b32 s18, 1, s8 // (op 667) 
s_add_i32 s19, s18, s9 // (op 667) 


s_and_b32 s9, s1, s19 // (op 667) 


s_add_i32 s1, s51, s9 // (op 667) 


s_ashr_i32 s9, s1, s8 // (op 667) 


s_lshl_b32 s1, s14, 24 // (op 667) 
s_ashr_i32 s8, s1, 31 // (op 667) 

s_xor_b32 s1, s9, s8 // (op 667) 

s_sub_i32 s9, s1, s8 // (op 667) 


s_add_u32 s18, s2, 16 // (op 667) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 667) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 667) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 667) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 667) 

s_mul_i32 s1, s9, s18 // (op 667) 

s_sub_i32 s51, s0, s1 // (op 667) 


s_mul_hi_i32 s0, s16, s10 // (op 667) 
s_add_i32 s1, s0, s16 // (op 667) 

s_ashr_i32 s0, s1, 31 // (op 667) 
s_cmp_eq_i32 s10, 0 // (op 667) 
s_mov_b32 s8, scc // (op 667) 
s_cmp_eq_i32 s8, 0 // (op 667) 
s_cselect_b32 s9, 0, -1 // (op 667) 

s_and_b32 s8, s49, 31 // (op 667) 
s_lshl_b32 s64, 1, s8 // (op 667) 
s_add_i32 s65, s64, s9 // (op 667) 


s_and_b32 s9, s0, s65 // (op 667) 


s_add_i32 s0, s1, s9 // (op 667) 


s_ashr_i32 s1, s0, s8 // (op 667) 


s_lshl_b32 s0, s49, 24 // (op 667) 
s_ashr_i32 s8, s0, 31 // (op 667) 

s_xor_b32 s0, s1, s8 // (op 667) 

s_sub_i32 s1, s0, s8 // (op 667) 


s_mul_i32 s0, s46, s18 // (op 667) 

s_mul_i32 s8, s50, s0 // (op 667) 

s_mul_i32 s0, s1, s8 // (op 667) 

s_sub_u32 s8, s16, s0 // (op 667) 

s_mul_hi_i32 s0, s8, s15 // (op 667) 
s_add_i32 s9, s0, s8 // (op 667) 


s_ashr_i32 s0, s9, 31 // (op 667) 
s_cmp_eq_i32 s15, 0 // (op 667) 
s_mov_b32 s8, scc // (op 667) 
s_cmp_eq_i32 s8, 0 // (op 667) 
s_cselect_b32 s18, 0, -1 // (op 667) 

s_and_b32 s8, s48, 31 // (op 667) 
s_lshl_b32 s19, 1, s8 // (op 667) 
s_add_i32 s64, s19, s18 // (op 667) 


s_and_b32 s18, s0, s64 // (op 667) 


s_add_i32 s0, s9, s18 // (op 667) 


s_ashr_i32 s9, s0, s8 // (op 667) 


s_lshl_b32 s0, s48, 24 // (op 667) 
s_ashr_i32 s8, s0, 31 // (op 667) 

s_xor_b32 s0, s9, s8 // (op 667) 

s_sub_i32 s9, s0, s8 // (op 667) 


s_cmp_eq_i32 s1, 0 // (op 667) 
s_mov_b32 s0, scc // (op 667) 

s_cmp_eq_i32 s0, 0 // (op 667) 
s_cselect_b32 s1, s9, s51 // (op 667) 



v_lshl_add_u32 v20, s1, 4, v19 // (op 667) 


v_mul_lo_u32 v19, v18, s32 // (op 667) 
v_add_u32 v19, v19, v20 // (op 667) 


s_mul_i32 s1, s32, s37 // (op 669) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 669) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 669) least significant
s_mul_hi_u32 s8, s32, s36 // (op 669) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 669) most significant: sum
s_add_u32 s1, s1, s8 // (op 669) most significant: sum


s_mov_b32 s64, s34 // (op 669) reg expression
s_mov_b32 s65, s35 // (op 669) reg expression
s_mov_b32 s66, s0 // (op 669) reg expression
s_mov_b32 s67, 131072 // (op 669) call()

s_mul_hi_i32 s51, s16, s11 // (op 686) 
v_and_b32 v18, -16, v1 // (op 686) 
s_add_i32 s0, s51, s16 // (op 686) 

v_sub_u32 v20, v1, v18 // (op 686) 

s_ashr_i32 s1, s0, 31 // (op 686) 
v_lshrrev_b32 v18, 4, v1 // (op 686) 
s_cmp_eq_i32 s11, 0 // (op 686) 
s_mov_b32 s51, scc // (op 686) 
s_cmp_eq_i32 s51, 0 // (op 686) 
s_cselect_b32 s8, 0, -1 // (op 686) 

s_and_b32 s9, s12, 31 // (op 686) 
s_lshl_b32 s51, 1, s9 // (op 686) 
s_add_i32 s18, s51, s8 // (op 686) 


s_and_b32 s8, s1, s18 // (op 686) 


s_add_i32 s1, s0, s8 // (op 686) 


s_ashr_i32 s0, s1, s9 // (op 686) 


s_lshl_b32 s1, s12, 24 // (op 686) 
s_ashr_i32 s51, s1, 31 // (op 686) 

s_xor_b32 s1, s0, s51 // (op 686) 

s_sub_i32 s0, s1, s51 // (op 686) 


s_mul_hi_i32 s1, s0, s13 // (op 686) 
s_add_i32 s51, s1, s0 // (op 686) 

s_ashr_i32 s1, s51, 31 // (op 686) 
s_cmp_eq_i32 s13, 0 // (op 686) 
s_mov_b32 s8, scc // (op 686) 
s_cmp_eq_i32 s8, 0 // (op 686) 
s_cselect_b32 s9, 0, -1 // (op 686) 

s_and_b32 s8, s14, 31 // (op 686) 
s_lshl_b32 s18, 1, s8 // (op 686) 
s_add_i32 s19, s18, s9 // (op 686) 


s_and_b32 s9, s1, s19 // (op 686) 


s_add_i32 s1, s51, s9 // (op 686) 


s_ashr_i32 s9, s1, s8 // (op 686) 


s_lshl_b32 s1, s14, 24 // (op 686) 
s_ashr_i32 s8, s1, 31 // (op 686) 

s_xor_b32 s1, s9, s8 // (op 686) 

s_sub_i32 s9, s1, s8 // (op 686) 


s_add_u32 s18, s2, 16 // (op 686) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 686) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 686) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 686) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 686) 

s_mul_i32 s1, s9, s18 // (op 686) 

s_sub_i32 s51, s0, s1 // (op 686) 


s_mul_hi_i32 s0, s16, s10 // (op 686) 
s_add_i32 s1, s0, s16 // (op 686) 

s_ashr_i32 s0, s1, 31 // (op 686) 
s_cmp_eq_i32 s10, 0 // (op 686) 
s_mov_b32 s8, scc // (op 686) 
s_cmp_eq_i32 s8, 0 // (op 686) 
s_cselect_b32 s9, 0, -1 // (op 686) 

s_and_b32 s8, s49, 31 // (op 686) 
s_lshl_b32 s68, 1, s8 // (op 686) 
s_add_i32 s69, s68, s9 // (op 686) 


s_and_b32 s9, s0, s69 // (op 686) 


s_add_i32 s0, s1, s9 // (op 686) 


s_ashr_i32 s1, s0, s8 // (op 686) 


s_lshl_b32 s0, s49, 24 // (op 686) 
s_ashr_i32 s8, s0, 31 // (op 686) 

s_xor_b32 s0, s1, s8 // (op 686) 

s_sub_i32 s1, s0, s8 // (op 686) 


s_mul_i32 s0, s46, s18 // (op 686) 

s_mul_i32 s8, s50, s0 // (op 686) 

s_mul_i32 s0, s1, s8 // (op 686) 

s_sub_u32 s8, s16, s0 // (op 686) 

s_mul_hi_i32 s0, s8, s15 // (op 686) 
s_add_i32 s9, s0, s8 // (op 686) 


s_ashr_i32 s0, s9, 31 // (op 686) 
s_cmp_eq_i32 s15, 0 // (op 686) 
s_mov_b32 s8, scc // (op 686) 
s_cmp_eq_i32 s8, 0 // (op 686) 
s_cselect_b32 s18, 0, -1 // (op 686) 

s_and_b32 s8, s48, 31 // (op 686) 
s_lshl_b32 s19, 1, s8 // (op 686) 
s_add_i32 s68, s19, s18 // (op 686) 


s_and_b32 s18, s0, s68 // (op 686) 


s_add_i32 s0, s9, s18 // (op 686) 


s_ashr_i32 s9, s0, s8 // (op 686) 


s_lshl_b32 s0, s48, 24 // (op 686) 
s_ashr_i32 s8, s0, 31 // (op 686) 

s_xor_b32 s0, s9, s8 // (op 686) 

s_sub_i32 s9, s0, s8 // (op 686) 


s_cmp_eq_i32 s1, 0 // (op 686) 
s_mov_b32 s0, scc // (op 686) 

s_cmp_eq_i32 s0, 0 // (op 686) 
s_cselect_b32 s1, s9, s51 // (op 686) 



v_lshl_add_u32 v21, s1, 4, v20 // (op 686) 


v_mul_lo_u32 v20, v18, s32 // (op 686) 
v_add_u32 v20, v20, v21 // (op 686) 


s_mul_i32 s1, s32, s37 // (op 688) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 688) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 688) least significant
s_mul_hi_u32 s8, s32, s36 // (op 688) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 688) most significant: sum
s_add_u32 s1, s1, s8 // (op 688) most significant: sum


s_mov_b32 s64, s34 // (op 688) reg expression
s_mov_b32 s65, s35 // (op 688) reg expression
s_mov_b32 s66, s0 // (op 688) reg expression
s_mov_b32 s67, 131072 // (op 688) call()


s_mul_i32 s1, s23, 6 // (op 366) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 366) least significant
s_mul_hi_u32 s51, s22, 6 // (op 366) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 366) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 366) 


v_mov_b32 v18, v14 // (op 366) reg expression
buffer_load_dwordx3 v[24:26], v18, s[56:59], 0 offen // (op 366) Load value
v_add_u32 v18, v18, v22 // (op 366) 
buffer_load_dwordx3 v[28:30], v18, s[56:59], 0 offen // (op 366) Load value




v_mov_b32 v18, v19 // (op 369) reg expression

buffer_load_ubyte v27, v18, s[64:67], 0 offen offset: 0 // (op 369) Load value




buffer_load_dwordx4 v[32:35], v12, s[52:55], 0 offen // (op 372) Load value



buffer_load_ubyte v18, v16, s[60:63], 0 offen offset: 0 // (op 375) Load value

s_mul_i32 s1, s23, 6 // (op 378) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 378) least significant
s_mul_hi_u32 s51, s22, 6 // (op 378) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 378) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 378) 


s_lshl_b64 s[0:1], s[22:23], 7 // (op 378) 
s_mul_i32 s9, s1, 6 // (op 378) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 378) least significant
s_mul_hi_u32 s51, s0, 6 // (op 378) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 378) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 378) 

v_add_u32 v21, v14, s0 // (op 378) 

buffer_load_dwordx3 v[36:38], v21, s[56:59], 0 offen // (op 378) Load value
v_add_u32 v21, v21, v22 // (op 378) 
buffer_load_dwordx3 v[40:42], v21, s[56:59], 0 offen // (op 378) Load value





s_lshl_b32 s51, s32, 2 // (op 381) 
v_add_u32 v14, v19, s51 // (op 381) 


buffer_load_ubyte v31, v14, s[64:67], 0 offen offset: 0 // (op 381) Load value





buffer_load_dwordx4 v[44:47], v12, s[52:55], 0 offen offset:64 // (op 384) Load value




buffer_load_ubyte v12, v16, s[60:63], 0 offen offset:4 // (op 387) Load value

v_lshlrev_b32 v14, 5, v1 // (op 806) 
v_mul_lo_u32 v16, v14, 6 // (op 806) 
v_lshrrev_b32 v19, 3, v16 // (op 806) 

v_lshrrev_b32 v16, 4, v14 // (op 806) 

v_lshlrev_b32 v14, 2, v16 // (op 806) 

v_add_u32 v16, v19, v14 // (op 806) 





s_waitcnt vmcnt(9)
ds_write_b96 v16, v[24:26] // (op 389) Store local data 
s_waitcnt vmcnt(8)
ds_write_b96 v16, v[28:30] offset:16 // (op 389) Store local data 

v_and_b32 v14, -16, v1 // (op 912) 
v_sub_u32 v16, v1, v14 // (op 912) 

v_and_b32 v14, 4294967280, v1 // (op 912) 
v_add_u32 v19, v14, v16 // (op 912) 





s_waitcnt vmcnt(7)
ds_write_b8 v19, v27 offset:2048 // (op 390) Store local data 

v_and_b32 v14, -4, v1 // (op 753) 
v_sub_u32 v16, v1, v14 // (op 753) 

v_lshlrev_b32 v14, 5, v16 // (op 753) 

v_lshrrev_b32 v16, 2, v1 // (op 753) 
v_lshl_add_u32 v19, v16, 7, v14 // (op 753) 


v_lshlrev_b32 v14, 2, v19 // (op 753) 

v_lshrrev_b32 v16, 3, v14 // (op 753) 




s_waitcnt vmcnt(6)
ds_write_b128 v16, v[32:35] offset:2112 // (op 391) Store local data 

v_and_b32 v14, -4, v1 // (op 859) 
v_sub_u32 v16, v1, v14 // (op 859) 

v_and_b32 v14, 4294967292, v1 // (op 859) 
v_add_u32 v19, v14, v16 // (op 859) 





s_waitcnt vmcnt(5)
ds_write_b8 v19, v18 offset:3136 // (op 392) Store local data 

v_and_b32 v14, -4, v1 // (op 724) 
v_sub_u32 v16, v1, v14 // (op 724) 

v_lshlrev_b32 v14, 5, v16 // (op 724) 

v_lshrrev_b32 v16, 2, v1 // (op 724) 
v_lshl_add_u32 v19, v16, 7, v14 // (op 724) 


v_lshlrev_b32 v14, 2, v19 // (op 724) 

v_lshrrev_b32 v16, 3, v14 // (op 724) 

v_lshlrev_b32 v14, 5, v1 // (op 777) 
v_mul_lo_u32 v19, v14, 6 // (op 777) 
v_lshrrev_b32 v39, 3, v19 // (op 777) 

v_lshrrev_b32 v19, 4, v14 // (op 777) 

v_lshlrev_b32 v14, 2, v19 // (op 777) 

v_add_u32 v19, v39, v14 // (op 777) 


v_and_b32 v14, -4, v1 // (op 830) 
v_sub_u32 v39, v1, v14 // (op 830) 

v_and_b32 v14, 4294967292, v1 // (op 830) 
v_add_u32 v43, v14, v39 // (op 830) 


v_and_b32 v14, -16, v1 // (op 883) 
v_sub_u32 v39, v1, v14 // (op 883) 

v_and_b32 v14, 4294967280, v1 // (op 883) 
v_add_u32 v21, v14, v39 // (op 883) 


v_and_b32 v14, -4, v1 // (op 970) 
v_sub_u32 v39, v1, v14 // (op 970) 

v_lshlrev_b32 v14, 5, v39 // (op 970) 

v_lshrrev_b32 v39, 2, v1 // (op 970) 
v_lshl_add_u32 v22, v39, 7, v14 // (op 970) 


v_lshlrev_b32 v14, 2, v22 // (op 970) 

v_lshrrev_b32 v39, 3, v14 // (op 970) 

v_lshlrev_b32 v14, 5, v1 // (op 994) 
v_mul_lo_u32 v22, v14, 6 // (op 994) 
v_lshrrev_b32 v23, 3, v22 // (op 994) 

v_lshrrev_b32 v22, 4, v14 // (op 994) 

v_lshlrev_b32 v14, 2, v22 // (op 994) 

v_add_u32 v22, v23, v14 // (op 994) 


v_and_b32 v14, -4, v1 // (op 1018) 
v_sub_u32 v23, v1, v14 // (op 1018) 

v_and_b32 v14, 4294967292, v1 // (op 1018) 
v_add_u32 v48, v14, v23 // (op 1018) 


v_and_b32 v14, -16, v1 // (op 1042) 
v_sub_u32 v23, v1, v14 // (op 1042) 

v_and_b32 v14, 4294967280, v1 // (op 1042) 
v_add_u32 v49, v14, v23 // (op 1042) 


s_mov_b32 s51, 0 // (op 59) call()
s_ashr_i64 s[0:1], s[26:27], 7 // (op 58) 
s_mov_b32 s8, s0 // (op 58) convert

s_and_b32 s9, s8, 4294967294 // (op 58) 

s_cmp_lt_u32 s51, s9 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_58 if false)
RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_58:

s_add_u32 s51, s51, 2 // (op 60) 
s_mul_i32 s1, s23, 6 // (op 10) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 10) least significant
s_mul_hi_u32 s8, s22, 6 // (op 10) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 10) most significant: sum

v_lshrrev_b64 v[50:51], 3, s[0:1] // (op 10) 


v_mov_b32 v14, v15 // (op 10) 
s_lshl_b64 s[0:1], s[22:23], 7 // (op 10) 
s_mul_i32 s9, s1, 6 // (op 10) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 10) least significant
s_mul_hi_u32 s18, s0, 6 // (op 10) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 10) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 10) 

s_lshl_b32 s8, s0, 1 // (op 10) 

v_add_u32 v23, v14, s8 // (op 10) 

buffer_load_dwordx3 v[24:26], v23, s[56:59], 0 offen // (op 10) Load value
v_add_u32 v23, v23, v50 // (op 10) 
buffer_load_dwordx3 v[28:30], v23, s[56:59], 0 offen // (op 10) Load value


s_waitcnt lgkmcnt(1)
s_barrier  // (op 1091) 





v_mov_b32 v23, v20 // (op 25) 
s_lshl_b32 s0, s32, 3 // (op 25) 
v_add_u32 v50, v23, s0 // (op 25) 


ds_read_b128 v[52:55], v3 offset:2112 // (op 70) Load local data 
buffer_load_ubyte v27, v50, s[64:67], 0 offen offset: 0 // (op 25) Load value



s_barrier  // (op 1104) 
v_mov_b32 v50, v13 // (op 4) 




ds_read_b96_tr_b6 v[56:58], v5 // (op 75) Transpose load local data 
ds_read_b96_tr_b6 v[59:61], v5 offset:256 // (op 75) Transpose load local data 
buffer_load_dwordx4 v[32:35], v50, s[52:55], 0 offen offset:128 // (op 4) Load value


v_mov_b32 v51, v17 // (op 17) 

buffer_load_ubyte v18, v51, s[60:63], 0 offen offset:8 // (op 17) Load value
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1118) 



s_barrier  // (op 1155) 
ds_read_u8 v62, v0 offset:3136 // (op 80) Load local data 



s_barrier  // (op 1132) 
s_waitcnt vmcnt(9)
ds_write_b96 v22, v[36:38] offset:3200 // (op 253) Store local data 


ds_read_u8 v63, v7 offset:2048 // (op 85) Load local data 
s_waitcnt vmcnt(8)
ds_write_b96 v22, v[40:42] offset:3216 // (op 253) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[56:61], a[0:3], v62, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 88) 




s_barrier  // (op 1167) 



s_waitcnt vmcnt(7)
ds_write_b8 v49, v31 offset:5248 // (op 255) Store local data 
s_barrier  // (op 1149) 



s_waitcnt vmcnt(6)
ds_write_b128 v39, v[44:47] offset:5312 // (op 243) Store local data 
s_barrier  // (op 1161) 



s_waitcnt vmcnt(5)
ds_write_b8 v48, v12 offset:6336 // (op 254) Store local data 

s_mul_i32 s1, s23, 6 // (op 256) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 256) least significant
s_mul_hi_u32 s8, s22, 6 // (op 256) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 256) most significant: sum

v_lshrrev_b64 v[52:53], 3, s[0:1] // (op 256) 


s_lshl_b64 s[0:1], s[22:23], 7 // (op 256) 
s_mul_i32 s9, s1, 6 // (op 256) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 256) least significant
s_mul_hi_u32 s18, s0, 6 // (op 256) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 256) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 256) 

s_mul_i32 s8, 3, s0 // (op 256) 

v_add_u32 v54, v14, s8 // (op 256) 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 1146) 
buffer_load_dwordx3 v[36:38], v54, s[56:59], 0 offen // (op 256) Load value



v_add_u32 v54, v54, v52 // (op 256) 
ds_read_b128 v[56:59], v6 offset:5312 // (op 245) Load local data 
buffer_load_dwordx3 v[40:42], v54, s[56:59], 0 offen // (op 256) Load value





s_barrier  // (op 1152) 
s_barrier  // (op 1222) 



s_lshl_b32 s0, s32, 2 // (op 258) 
s_mul_i32 s1, 3, s0 // (op 258) 

v_add_u32 v14, v23, s1 // (op 258) 


buffer_load_ubyte v31, v14, s[64:67], 0 offen offset: 0 // (op 258) Load value

s_lshl_b64 s[0:1], s[22:23], 7 // (op 550) 

s_mul_i32 s9, s1, 6 // (op 550) most significant: low of high * low



s_mul_i32 s8, s0, 6 // (op 550) least significant
s_mul_hi_u32 s18, s0, 6 // (op 550) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 550) most significant: sum


s_lshl_b32 s0, s32, 3 // (op 678) 
v_add_u32 v20, v20, s0 // (op 678) 

s_lshr_b64 s[0:1], s[8:9], 3 // (op 550) 

s_lshl_b32 s8, s0, 1 // (op 550) 

v_add_u32 v15, v15, s8 // (op 550) 

ds_read_b96_tr_b6 v[60:62], v9 offset:3200 // (op 247) Transpose load local data 
ds_read_b96_tr_b6 v[63:65], v9 offset:3456 // (op 247) Transpose load local data 
buffer_load_dwordx4 v[44:47], v50, s[52:55], 0 offen offset:192 // (op 242) Load value




s_barrier  // (op 1228) 
v_add_u32 v13, 128, v13 // (op 486) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1158) 
buffer_load_ubyte v12, v51, s[60:63], 0 offen offset:12 // (op 257) Load value




ds_read_u8 v14, v4 offset:6336 // (op 249) Load local data 
s_barrier  // (op 1107) 



s_barrier  // (op 1164) 
s_barrier  // (op 1234) 


v_add_u32 v17, 8, v17 // (op 614) 
ds_read_u8 v23, v11 offset:5248 // (op 251) Load local data 
s_waitcnt vmcnt(9)
ds_write_b96 v19, v[24:26] // (op 13) Store local data 
s_waitcnt vmcnt(8)
ds_write_b96 v19, v[28:30] offset:16 // (op 13) Store local data 
s_barrier  // (op 1240) 
s_barrier  // (op 1135) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[60:65], a[0:3], v14, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 252) 







s_waitcnt vmcnt(7)
ds_write_b8 v21, v27 offset:2048 // (op 30) Store local data 
s_barrier  // (op 1094) 



s_waitcnt vmcnt(6)
ds_write_b128 v16, v[32:35] offset:2112 // (op 6) Store local data 
s_barrier  // (op 1121) 



s_waitcnt vmcnt(5)
ds_write_b8 v43, v18 offset:3136 // (op 21) Store local data 
s_ashr_i64 s[0:1], s[26:27], 7 // (op 58) 
s_mov_b32 s8, s0 // (op 58) convert

s_and_b32 s9, s8, 4294967294 // (op 58) 

s_cmp_lt_u32 s51, s9 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_58 if true)
RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_58:

















s_mul_hi_i32 s51, s16, s11 // (op 449) 
v_lshrrev_b32 v39, 2, v1 // (op 449) 
s_add_i32 s0, s51, s16 // (op 449) 

v_and_b32 v43, -4, v1 // (op 449) 
s_ashr_i32 s1, s0, 31 // (op 449) 
v_sub_u32 v13, v1, v43 // (op 449) 

s_cmp_eq_i32 s11, 0 // (op 449) 
s_mov_b32 s51, scc // (op 449) 
v_lshlrev_b32 v43, 5, v13 // (op 449) 

s_cmp_eq_i32 s51, 0 // (op 449) 
s_cselect_b32 s8, 0, -1 // (op 449) 

s_and_b32 s9, s12, 31 // (op 449) 
s_lshl_b32 s51, 1, s9 // (op 449) 
s_add_i32 s18, s51, s8 // (op 449) 


s_and_b32 s8, s1, s18 // (op 449) 


s_add_i32 s1, s0, s8 // (op 449) 


s_ashr_i32 s0, s1, s9 // (op 449) 


s_lshl_b32 s1, s12, 24 // (op 449) 
s_ashr_i32 s51, s1, 31 // (op 449) 

s_xor_b32 s1, s0, s51 // (op 449) 

s_sub_i32 s0, s1, s51 // (op 449) 


s_mul_hi_i32 s1, s0, s13 // (op 449) 
s_add_i32 s51, s1, s0 // (op 449) 

s_ashr_i32 s1, s51, 31 // (op 449) 
s_cmp_eq_i32 s13, 0 // (op 449) 
s_mov_b32 s8, scc // (op 449) 
s_cmp_eq_i32 s8, 0 // (op 449) 
s_cselect_b32 s9, 0, -1 // (op 449) 

s_and_b32 s8, s14, 31 // (op 449) 
s_lshl_b32 s18, 1, s8 // (op 449) 
s_add_i32 s19, s18, s9 // (op 449) 


s_and_b32 s9, s1, s19 // (op 449) 


s_add_i32 s1, s51, s9 // (op 449) 


s_ashr_i32 s9, s1, s8 // (op 449) 


s_lshl_b32 s1, s14, 24 // (op 449) 
s_ashr_i32 s8, s1, 31 // (op 449) 

s_xor_b32 s1, s9, s8 // (op 449) 

s_sub_i32 s9, s1, s8 // (op 449) 


s_mul_i32 s1, s0, s46 // (op 449) 

s_sub_u32 s0, s16, s1 // (op 449) 

s_mul_i32 s1, s9, s46 // (op 449) 
s_add_u32 s1, s1, s0 // (op 449) 


s_mul_hi_i32 s0, s16, s10 // (op 449) 
s_add_i32 s51, s0, s16 // (op 449) 

s_ashr_i32 s0, s51, 31 // (op 449) 
s_cmp_eq_i32 s10, 0 // (op 449) 
s_mov_b32 s8, scc // (op 449) 
s_cmp_eq_i32 s8, 0 // (op 449) 
s_cselect_b32 s9, 0, -1 // (op 449) 

s_and_b32 s8, s49, 31 // (op 449) 
s_lshl_b32 s18, 1, s8 // (op 449) 
s_add_i32 s19, s18, s9 // (op 449) 


s_and_b32 s9, s0, s19 // (op 449) 


s_add_i32 s0, s51, s9 // (op 449) 


s_ashr_i32 s9, s0, s8 // (op 449) 


s_lshl_b32 s0, s49, 24 // (op 449) 
s_ashr_i32 s8, s0, 31 // (op 449) 

s_xor_b32 s0, s9, s8 // (op 449) 

s_sub_i32 s9, s0, s8 // (op 449) 


s_add_u32 s18, s2, 16 // (op 449) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 449) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 449) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 449) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 449) 

s_mul_i32 s0, s46, s18 // (op 449) 

s_mul_i32 s8, s50, s0 // (op 449) 

s_mul_i32 s0, s9, s8 // (op 449) 

s_sub_u32 s8, s16, s0 // (op 449) 

s_mov_b32 s0, s8 // (op 449) 
s_mul_hi_i32 s51, s0, s15 // (op 449) 
s_add_i32 s18, s51, s0 // (op 449) 


s_ashr_i32 s0, s18, 31 // (op 449) 
s_cmp_eq_i32 s15, 0 // (op 449) 
s_mov_b32 s19, scc // (op 449) 
s_cmp_eq_i32 s19, 0 // (op 449) 
s_cselect_b32 s51, 0, -1 // (op 449) 

s_and_b32 s19, s48, 31 // (op 449) 
s_lshl_b32 s68, 1, s19 // (op 449) 
s_add_i32 s69, s68, s51 // (op 449) 


s_and_b32 s51, s0, s69 // (op 449) 


s_add_i32 s0, s18, s51 // (op 449) 


s_ashr_i32 s18, s0, s19 // (op 449) 


s_lshl_b32 s0, s48, 24 // (op 449) 
s_ashr_i32 s19, s0, 31 // (op 449) 

s_xor_b32 s0, s18, s19 // (op 449) 

s_sub_i32 s18, s0, s19 // (op 449) 


s_mul_i32 s0, s47, s46 // (op 449) 
s_add_u32 s68, s6, 16 // (op 449) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 449) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 449) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 449) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 449) 

s_sub_i32 s19, s68, s0 // (op 449) 

s_mul_i32 s51, s18, s19 // (op 449) 


s_sub_u32 s18, s8, s51 // (op 449) 


s_add_u32 s8, s18, s0 // (op 449) 


s_cmp_eq_i32 s9, 0 // (op 449) 
s_mov_b32 s0, scc // (op 449) 

s_cmp_eq_i32 s0, 0 // (op 449) 
s_cselect_b32 s9, s8, s1 // (op 449) 



v_lshl_add_u32 v13, s9, 4, v39 // (op 449) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 449) 
s_lshr_b32 s51, s0, 1 // (op 449) 

v_lshl_add_u32 v39, s51, 8, v43 // (op 449) 


v_mov_b32 v15, 0 // (op 449) convert
v_mov_b32 v14, v39 // (op 449) convert

v_mul_lo_u32 v17, v13, s5 // (op 449) most significant: low of low * high
v_mul_hi_u32 v39, v13, s4 // (op 449) most significant: high of low * low
v_mul_lo_u32 v16, v13, s4 // (op 449) least significant: low of low * low
v_add_u32 v17, v17, v39 // (op 449) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 449) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 449) most significant half



v_lshlrev_b64 v[14:15], 2, v[16:17] // (op 449) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 449) 

v_mov_b32 v39, v16 // (op 449) convert

s_mul_i32 s1, s4, s7 // (op 451) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 451) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 451) least significant
s_mul_hi_u32 s8, s4, s6 // (op 451) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 451) most significant: sum
s_add_u32 s1, s1, s8 // (op 451) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 451) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 451) 

s_mov_b32 s52, s20 // (op 451) reg expression
s_mov_b32 s53, s21 // (op 451) reg expression
s_mov_b32 s54, s0 // (op 451) reg expression
s_mov_b32 s55, 131072 // (op 451) call()

s_mul_hi_i32 s51, s16, s11 // (op 516) 
v_lshlrev_b32 v43, 1, v1 // (op 516) 
s_add_i32 s0, s51, s16 // (op 516) 

s_ashr_i32 s1, s0, 31 // (op 516) 
s_cmp_eq_i32 s11, 0 // (op 516) 
s_mov_b32 s51, scc // (op 516) 
s_cmp_eq_i32 s51, 0 // (op 516) 
s_cselect_b32 s8, 0, -1 // (op 516) 

s_and_b32 s9, s12, 31 // (op 516) 
s_lshl_b32 s51, 1, s9 // (op 516) 
s_add_i32 s18, s51, s8 // (op 516) 


s_and_b32 s8, s1, s18 // (op 516) 


s_add_i32 s1, s0, s8 // (op 516) 


s_ashr_i32 s0, s1, s9 // (op 516) 


s_lshl_b32 s1, s12, 24 // (op 516) 
s_ashr_i32 s51, s1, 31 // (op 516) 

s_xor_b32 s1, s0, s51 // (op 516) 

s_sub_i32 s0, s1, s51 // (op 516) 


s_mul_hi_i32 s1, s0, s13 // (op 516) 
s_add_i32 s51, s1, s0 // (op 516) 

s_ashr_i32 s1, s51, 31 // (op 516) 
s_cmp_eq_i32 s13, 0 // (op 516) 
s_mov_b32 s8, scc // (op 516) 
s_cmp_eq_i32 s8, 0 // (op 516) 
s_cselect_b32 s9, 0, -1 // (op 516) 

s_and_b32 s8, s14, 31 // (op 516) 
s_lshl_b32 s18, 1, s8 // (op 516) 
s_add_i32 s19, s18, s9 // (op 516) 


s_and_b32 s9, s1, s19 // (op 516) 


s_add_i32 s1, s51, s9 // (op 516) 


s_ashr_i32 s9, s1, s8 // (op 516) 


s_lshl_b32 s1, s14, 24 // (op 516) 
s_ashr_i32 s8, s1, 31 // (op 516) 

s_xor_b32 s1, s9, s8 // (op 516) 

s_sub_i32 s9, s1, s8 // (op 516) 


s_add_u32 s18, s2, 16 // (op 516) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 516) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 516) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 516) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 516) 

s_mul_i32 s1, s9, s18 // (op 516) 

s_sub_i32 s51, s0, s1 // (op 516) 


s_mul_hi_i32 s0, s16, s10 // (op 516) 
s_add_i32 s1, s0, s16 // (op 516) 

s_ashr_i32 s0, s1, 31 // (op 516) 
s_cmp_eq_i32 s10, 0 // (op 516) 
s_mov_b32 s8, scc // (op 516) 
s_cmp_eq_i32 s8, 0 // (op 516) 
s_cselect_b32 s9, 0, -1 // (op 516) 

s_and_b32 s8, s49, 31 // (op 516) 
s_lshl_b32 s68, 1, s8 // (op 516) 
s_add_i32 s69, s68, s9 // (op 516) 


s_and_b32 s9, s0, s69 // (op 516) 


s_add_i32 s0, s1, s9 // (op 516) 


s_ashr_i32 s1, s0, s8 // (op 516) 


s_lshl_b32 s0, s49, 24 // (op 516) 
s_ashr_i32 s8, s0, 31 // (op 516) 

s_xor_b32 s0, s1, s8 // (op 516) 

s_sub_i32 s1, s0, s8 // (op 516) 


s_mul_i32 s0, s46, s18 // (op 516) 

s_mul_i32 s8, s50, s0 // (op 516) 

s_mul_i32 s0, s1, s8 // (op 516) 

s_sub_u32 s8, s16, s0 // (op 516) 

s_mul_hi_i32 s0, s8, s15 // (op 516) 
s_add_i32 s9, s0, s8 // (op 516) 


s_ashr_i32 s0, s9, 31 // (op 516) 
s_cmp_eq_i32 s15, 0 // (op 516) 
s_mov_b32 s8, scc // (op 516) 
s_cmp_eq_i32 s8, 0 // (op 516) 
s_cselect_b32 s18, 0, -1 // (op 516) 

s_and_b32 s8, s48, 31 // (op 516) 
s_lshl_b32 s19, 1, s8 // (op 516) 
s_add_i32 s68, s19, s18 // (op 516) 


s_and_b32 s18, s0, s68 // (op 516) 


s_add_i32 s0, s9, s18 // (op 516) 


s_ashr_i32 s9, s0, s8 // (op 516) 


s_lshl_b32 s0, s48, 24 // (op 516) 
s_ashr_i32 s8, s0, 31 // (op 516) 

s_xor_b32 s0, s9, s8 // (op 516) 

s_sub_i32 s9, s0, s8 // (op 516) 


s_cmp_eq_i32 s1, 0 // (op 516) 
s_mov_b32 s0, scc // (op 516) 

s_cmp_eq_i32 s0, 0 // (op 516) 
s_cselect_b32 s1, s9, s51 // (op 516) 



s_lshl_b32 s0, s1, 4 // (op 516) 

s_mov_b32 s9, 0 // (op 516) convert
s_mov_b32 s8, s0 // (op 516) convert

s_ashr_i64 s[0:1], s[26:27], 7 // (op 516) 
s_lshr_b32 s51, s0, 1 // (op 516) 

v_lshl_add_u32 v13, s51, 8, v43 // (op 516) 


v_mul_lo_u32 v15, v13, s23 // (op 516) most significant: low of low * high
v_mul_hi_u32 v43, v13, s22 // (op 516) most significant: high of low * low
v_mul_lo_u32 v14, v13, s22 // (op 516) least significant: low of low * low
v_add_u32 v15, v15, v43 // (op 516) most significant: sum

v_mov_b32 v43, s8 // (op 516) 
v_mov_b32 v16, s9 // (op 516) 
v_add_co_u32 v14, s[0:1], v14, v43 // (op 516) least significant half
v_addc_co_u32 v15, s[0:1], v15, v16, s[0:1] // (op 516) most significant half





v_mul_lo_u32 v17, v15, 6 // (op 516) most significant: low of high * low
v_mul_hi_u32 v13, v14, 6 // (op 516) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 516) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 516) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 516) 

v_mov_b32 v13, v14 // (op 516) convert

s_mul_i32 s1, s27, 6 // (op 518) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 518) least significant
s_mul_hi_u32 s51, s26, 6 // (op 518) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 518) most significant: sum

s_mul_i32 s9, s22, s1 // (op 518) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 518) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 518) least significant
s_mul_hi_u32 s18, s22, s0 // (op 518) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 518) most significant: sum
s_add_u32 s9, s9, s18 // (op 518) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 518) 

s_mov_b32 s56, s24 // (op 518) reg expression
s_mov_b32 s57, s25 // (op 518) reg expression
s_mov_b32 s58, s0 // (op 518) reg expression
s_mov_b32 s59, 131072 // (op 518) call()

s_mul_hi_i32 s51, s16, s11 // (op 580) 
v_lshrrev_b32 v43, 2, v1 // (op 580) 
s_add_i32 s0, s51, s16 // (op 580) 

v_and_b32 v14, -4, v1 // (op 580) 
s_ashr_i32 s1, s0, 31 // (op 580) 
v_sub_u32 v15, v1, v14 // (op 580) 

s_cmp_eq_i32 s11, 0 // (op 580) 
s_mov_b32 s51, scc // (op 580) 
s_cmp_eq_i32 s51, 0 // (op 580) 
s_cselect_b32 s8, 0, -1 // (op 580) 

s_and_b32 s9, s12, 31 // (op 580) 
s_lshl_b32 s51, 1, s9 // (op 580) 
s_add_i32 s18, s51, s8 // (op 580) 


s_and_b32 s8, s1, s18 // (op 580) 


s_add_i32 s1, s0, s8 // (op 580) 


s_ashr_i32 s0, s1, s9 // (op 580) 


s_lshl_b32 s1, s12, 24 // (op 580) 
s_ashr_i32 s51, s1, 31 // (op 580) 

s_xor_b32 s1, s0, s51 // (op 580) 

s_sub_i32 s0, s1, s51 // (op 580) 


s_mul_hi_i32 s1, s0, s13 // (op 580) 
s_add_i32 s51, s1, s0 // (op 580) 

s_ashr_i32 s1, s51, 31 // (op 580) 
s_cmp_eq_i32 s13, 0 // (op 580) 
s_mov_b32 s8, scc // (op 580) 
s_cmp_eq_i32 s8, 0 // (op 580) 
s_cselect_b32 s9, 0, -1 // (op 580) 

s_and_b32 s8, s14, 31 // (op 580) 
s_lshl_b32 s18, 1, s8 // (op 580) 
s_add_i32 s19, s18, s9 // (op 580) 


s_and_b32 s9, s1, s19 // (op 580) 


s_add_i32 s1, s51, s9 // (op 580) 


s_ashr_i32 s9, s1, s8 // (op 580) 


s_lshl_b32 s1, s14, 24 // (op 580) 
s_ashr_i32 s8, s1, 31 // (op 580) 

s_xor_b32 s1, s9, s8 // (op 580) 

s_sub_i32 s9, s1, s8 // (op 580) 


s_mul_i32 s1, s0, s46 // (op 580) 

s_sub_u32 s0, s16, s1 // (op 580) 

s_mul_i32 s1, s9, s46 // (op 580) 
s_add_u32 s1, s1, s0 // (op 580) 


s_mul_hi_i32 s0, s16, s10 // (op 580) 
s_add_i32 s51, s0, s16 // (op 580) 

s_ashr_i32 s0, s51, 31 // (op 580) 
s_cmp_eq_i32 s10, 0 // (op 580) 
s_mov_b32 s8, scc // (op 580) 
s_cmp_eq_i32 s8, 0 // (op 580) 
s_cselect_b32 s9, 0, -1 // (op 580) 

s_and_b32 s8, s49, 31 // (op 580) 
s_lshl_b32 s18, 1, s8 // (op 580) 
s_add_i32 s19, s18, s9 // (op 580) 


s_and_b32 s9, s0, s19 // (op 580) 


s_add_i32 s0, s51, s9 // (op 580) 


s_ashr_i32 s9, s0, s8 // (op 580) 


s_lshl_b32 s0, s49, 24 // (op 580) 
s_ashr_i32 s8, s0, 31 // (op 580) 

s_xor_b32 s0, s9, s8 // (op 580) 

s_sub_i32 s9, s0, s8 // (op 580) 


s_add_u32 s18, s2, 16 // (op 580) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 580) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 580) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 580) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 580) 

s_mul_i32 s0, s46, s18 // (op 580) 

s_mul_i32 s8, s50, s0 // (op 580) 

s_mul_i32 s0, s9, s8 // (op 580) 

s_sub_u32 s8, s16, s0 // (op 580) 

s_mov_b32 s0, s8 // (op 580) 
s_mul_hi_i32 s51, s0, s15 // (op 580) 
s_add_i32 s18, s51, s0 // (op 580) 


s_ashr_i32 s0, s18, 31 // (op 580) 
s_cmp_eq_i32 s15, 0 // (op 580) 
s_mov_b32 s19, scc // (op 580) 
s_cmp_eq_i32 s19, 0 // (op 580) 
s_cselect_b32 s51, 0, -1 // (op 580) 

s_and_b32 s19, s48, 31 // (op 580) 
s_lshl_b32 s68, 1, s19 // (op 580) 
s_add_i32 s69, s68, s51 // (op 580) 


s_and_b32 s51, s0, s69 // (op 580) 


s_add_i32 s0, s18, s51 // (op 580) 


s_ashr_i32 s18, s0, s19 // (op 580) 


s_lshl_b32 s0, s48, 24 // (op 580) 
s_ashr_i32 s19, s0, 31 // (op 580) 

s_xor_b32 s0, s18, s19 // (op 580) 

s_sub_i32 s18, s0, s19 // (op 580) 


s_mul_i32 s0, s47, s46 // (op 580) 
s_add_u32 s68, s6, 16 // (op 580) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 580) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 580) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 580) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 580) 

s_sub_i32 s19, s68, s0 // (op 580) 

s_mul_i32 s51, s18, s19 // (op 580) 


s_sub_u32 s18, s8, s51 // (op 580) 


s_add_u32 s8, s18, s0 // (op 580) 


s_cmp_eq_i32 s9, 0 // (op 580) 
s_mov_b32 s0, scc // (op 580) 

s_cmp_eq_i32 s0, 0 // (op 580) 
s_cselect_b32 s9, s8, s1 // (op 580) 



v_lshl_add_u32 v14, s9, 4, v43 // (op 580) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 580) 
s_lshr_b32 s51, s0, 1 // (op 580) 

v_lshl_add_u32 v43, s51, 3, v15 // (op 580) 


v_mul_lo_u32 v15, v14, s28 // (op 580) 
v_add_u32 v15, v15, v43 // (op 580) 


s_mul_i32 s1, s28, s7 // (op 582) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 582) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 582) least significant
s_mul_hi_u32 s8, s28, s6 // (op 582) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 582) most significant: sum
s_add_u32 s1, s1, s8 // (op 582) most significant: sum


s_mov_b32 s60, s30 // (op 582) reg expression
s_mov_b32 s61, s31 // (op 582) reg expression
s_mov_b32 s62, s0 // (op 582) reg expression
s_mov_b32 s63, 131072 // (op 582) call()

s_mul_hi_i32 s51, s16, s11 // (op 644) 
v_and_b32 v14, -16, v1 // (op 644) 
s_add_i32 s0, s51, s16 // (op 644) 

v_sub_u32 v43, v1, v14 // (op 644) 

s_ashr_i32 s1, s0, 31 // (op 644) 
v_lshrrev_b32 v14, 4, v1 // (op 644) 
s_cmp_eq_i32 s11, 0 // (op 644) 
s_mov_b32 s51, scc // (op 644) 
s_cmp_eq_i32 s51, 0 // (op 644) 
s_cselect_b32 s8, 0, -1 // (op 644) 

s_and_b32 s9, s12, 31 // (op 644) 
s_lshl_b32 s51, 1, s9 // (op 644) 
s_add_i32 s18, s51, s8 // (op 644) 


s_and_b32 s8, s1, s18 // (op 644) 


s_add_i32 s1, s0, s8 // (op 644) 


s_ashr_i32 s0, s1, s9 // (op 644) 


s_lshl_b32 s1, s12, 24 // (op 644) 
s_ashr_i32 s51, s1, 31 // (op 644) 

s_xor_b32 s1, s0, s51 // (op 644) 

s_sub_i32 s0, s1, s51 // (op 644) 


s_mul_hi_i32 s1, s0, s13 // (op 644) 
s_add_i32 s51, s1, s0 // (op 644) 

s_ashr_i32 s1, s51, 31 // (op 644) 
s_cmp_eq_i32 s13, 0 // (op 644) 
s_mov_b32 s8, scc // (op 644) 
s_cmp_eq_i32 s8, 0 // (op 644) 
s_cselect_b32 s9, 0, -1 // (op 644) 

s_and_b32 s8, s14, 31 // (op 644) 
s_lshl_b32 s18, 1, s8 // (op 644) 
s_add_i32 s19, s18, s9 // (op 644) 


s_and_b32 s9, s1, s19 // (op 644) 


s_add_i32 s1, s51, s9 // (op 644) 


s_ashr_i32 s9, s1, s8 // (op 644) 


s_lshl_b32 s1, s14, 24 // (op 644) 
s_ashr_i32 s8, s1, 31 // (op 644) 

s_xor_b32 s1, s9, s8 // (op 644) 

s_sub_i32 s9, s1, s8 // (op 644) 


s_add_u32 s18, s2, 16 // (op 644) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 644) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 644) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 644) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 644) 

s_mul_i32 s1, s9, s18 // (op 644) 

s_sub_i32 s51, s0, s1 // (op 644) 


s_mul_hi_i32 s0, s16, s10 // (op 644) 
s_add_i32 s1, s0, s16 // (op 644) 

s_ashr_i32 s0, s1, 31 // (op 644) 
s_cmp_eq_i32 s10, 0 // (op 644) 
s_mov_b32 s8, scc // (op 644) 
s_cmp_eq_i32 s8, 0 // (op 644) 
s_cselect_b32 s9, 0, -1 // (op 644) 

s_and_b32 s8, s49, 31 // (op 644) 
s_lshl_b32 s68, 1, s8 // (op 644) 
s_add_i32 s69, s68, s9 // (op 644) 


s_and_b32 s9, s0, s69 // (op 644) 


s_add_i32 s0, s1, s9 // (op 644) 


s_ashr_i32 s1, s0, s8 // (op 644) 


s_lshl_b32 s0, s49, 24 // (op 644) 
s_ashr_i32 s8, s0, 31 // (op 644) 

s_xor_b32 s0, s1, s8 // (op 644) 

s_sub_i32 s1, s0, s8 // (op 644) 


s_mul_i32 s0, s46, s18 // (op 644) 

s_mul_i32 s8, s50, s0 // (op 644) 

s_mul_i32 s0, s1, s8 // (op 644) 

s_sub_u32 s8, s16, s0 // (op 644) 

s_mul_hi_i32 s0, s8, s15 // (op 644) 
s_add_i32 s9, s0, s8 // (op 644) 


s_ashr_i32 s0, s9, 31 // (op 644) 
s_cmp_eq_i32 s15, 0 // (op 644) 
s_mov_b32 s8, scc // (op 644) 
s_cmp_eq_i32 s8, 0 // (op 644) 
s_cselect_b32 s18, 0, -1 // (op 644) 

s_and_b32 s8, s48, 31 // (op 644) 
s_lshl_b32 s19, 1, s8 // (op 644) 
s_add_i32 s68, s19, s18 // (op 644) 


s_and_b32 s18, s0, s68 // (op 644) 


s_add_i32 s0, s9, s18 // (op 644) 


s_ashr_i32 s9, s0, s8 // (op 644) 


s_lshl_b32 s0, s48, 24 // (op 644) 
s_ashr_i32 s8, s0, 31 // (op 644) 

s_xor_b32 s0, s9, s8 // (op 644) 

s_sub_i32 s9, s0, s8 // (op 644) 


s_cmp_eq_i32 s1, 0 // (op 644) 
s_mov_b32 s0, scc // (op 644) 

s_cmp_eq_i32 s0, 0 // (op 644) 
s_cselect_b32 s1, s9, s51 // (op 644) 



v_lshl_add_u32 v16, s1, 4, v43 // (op 644) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 644) 
s_lshr_b32 s51, s0, 1 // (op 644) 

v_lshl_add_u32 v43, s51, 3, v14 // (op 644) 


v_mul_lo_u32 v14, v43, s32 // (op 644) 
v_add_u32 v14, v14, v16 // (op 644) 


s_mul_i32 s1, s32, s37 // (op 646) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 646) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 646) least significant
s_mul_hi_u32 s8, s32, s36 // (op 646) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 646) most significant: sum
s_add_u32 s1, s1, s8 // (op 646) most significant: sum


s_mov_b32 s64, s34 // (op 646) reg expression
s_mov_b32 s65, s35 // (op 646) reg expression
s_mov_b32 s66, s0 // (op 646) reg expression
s_mov_b32 s67, 131072 // (op 646) call()

v_and_b32 v43, -4, v1 // (op 737) 
v_sub_u32 v16, v1, v43 // (op 737) 

v_lshlrev_b32 v43, 5, v16 // (op 737) 

v_lshrrev_b32 v16, 2, v1 // (op 737) 
v_lshl_add_u32 v17, v16, 7, v43 // (op 737) 


v_lshlrev_b32 v16, 2, v17 // (op 737) 

v_lshrrev_b32 v43, 3, v16 // (op 737) 

v_lshlrev_b32 v16, 5, v1 // (op 790) 
v_mul_lo_u32 v17, v16, 6 // (op 790) 
v_lshrrev_b32 v18, 3, v17 // (op 790) 

v_lshrrev_b32 v17, 4, v16 // (op 790) 

v_lshlrev_b32 v16, 2, v17 // (op 790) 

v_add_u32 v17, v18, v16 // (op 790) 


v_and_b32 v16, -4, v1 // (op 843) 
v_sub_u32 v18, v1, v16 // (op 843) 

v_and_b32 v16, 4294967292, v1 // (op 843) 
v_add_u32 v19, v16, v18 // (op 843) 


v_and_b32 v16, -16, v1 // (op 896) 
v_sub_u32 v18, v1, v16 // (op 896) 

v_and_b32 v16, 4294967280, v1 // (op 896) 
v_add_u32 v20, v16, v18 // (op 896) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 172) 
s_and_b32 s51, s0, 4294967294 // (op 172) 

s_ashr_i64 s[0:1], s[26:27], 7 // (op 171) 
s_mov_b32 s8, s0 // (op 171) convert

s_cmp_lt_u32 s51, s8 // (op 171) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_171 // (op 171) Condition: Top (jump to RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_171 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_171:


s_add_u32 s51, s51, 1 // (op 173) 
v_mov_b32 v12, v39 // (op 176) 

buffer_load_dwordx4 v[22:25], v12, s[52:55], 0 offen // (op 176) Load value

v_add_u32 v39, 64, v39 // (op 439) 

s_barrier  // (op 1097) 
s_mul_i32 s1, s23, 6 // (op 190) most significant: low of high * low



s_mul_i32 s0, s22, 6 // (op 190) least significant
s_mul_hi_u32 s8, s22, 6 // (op 190) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 190) most significant: sum

v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 190) 


v_mov_b32 v12, v13 // (op 190) 
v_mov_b32 v16, v12 // (op 190) reg expression
buffer_load_dwordx3 v[28:30], v16, s[56:59], 0 offen // (op 190) Load value
v_add_u32 v16, v16, v26 // (op 190) 
buffer_load_dwordx3 v[32:34], v16, s[56:59], 0 offen // (op 190) Load value



s_lshl_b64 s[0:1], s[22:23], 7 // (op 508) 
s_mul_i32 s9, s1, 6 // (op 508) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 508) least significant
s_mul_hi_u32 s18, s0, 6 // (op 508) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 508) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 508) 

v_add_u32 v13, v13, s0 // (op 508) 

s_waitcnt vmcnt(2)
ds_write_b128 v43, v[22:25] offset:2112 // (op 177) Store local data 



v_mov_b32 v12, v15 // (op 191) 

s_barrier  // (op 1110) 
buffer_load_ubyte v16, v12, s[60:63], 0 offen offset: 0 // (op 191) Load value




v_add_u32 v15, 4, v15 // (op 572) 
s_waitcnt vmcnt(2)
ds_write_b96 v17, v[28:30] // (op 187) Store local data 


v_mov_b32 v12, v14 // (op 192) 
v_mov_b32 v18, v12 // (op 192) reg expression

buffer_load_ubyte v31, v18, s[64:67], 0 offen offset: 0 // (op 192) Load value


s_lshl_b32 s0, s32, 2 // (op 636) 
v_add_u32 v14, v14, s0 // (op 636) 

s_waitcnt vmcnt(2)
ds_write_b96 v17, v[32:34] offset:16 // (op 187) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1100) 



s_barrier  // (op 1124) 
ds_read_b128 v[32:35], v3 offset:2112 // (op 179) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 1114) 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v16 offset:3136 // (op 188) Store local data 




ds_read_b96_tr_b6 v[22:24], v5 // (op 181) Transpose load local data 
s_barrier  // (op 1138) 
ds_read_b96_tr_b6 v[25:27], v5 offset:256 // (op 181) Transpose load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 1128) 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v31 offset:2048 // (op 189) Store local data 




ds_read_u8 v12, v0 offset:3136 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1142) 


ds_read_u8 v16, v7 offset:2048 // (op 185) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:35], v[22:27], a[0:3], v12, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 186) 




s_ashr_i64 s[0:1], s[26:27], 7 // (op 171) 
s_mov_b32 s8, s0 // (op 171) convert

s_cmp_lt_u32 s51, s8 // (op 171) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_171 // (op 171) Condition: Bottom (jump to RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_171 if true)
RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_171:























s_barrier  // (op 1216) 








v_cmp_eq_f32 vcc, 0.00000, v8 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_133
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v10, v4 // (op 50) 
v_mul_f32 v13, v10, v5 // (op 50) 
v_mul_f32 v14, v10, v6 // (op 50) 
v_mul_f32 v15, v10, v7 // (op 50) 

s_mul_hi_i32 s0, s16, s11 // (op 938) 
v_lshrrev_b32 v0, 6, v1 // (op 938) 
s_add_i32 s1, s0, s16 // (op 938) 

v_and_b32 v9, 63, v1 // (op 938) 
s_ashr_i32 s0, s1, 31 // (op 938) 
v_lshrrev_b32 v11, 4, v9 // (op 938) 
s_cmp_eq_i32 s11, 0 // (op 938) 
s_mov_b32 s4, scc // (op 938) 
v_lshlrev_b32 v3, 2, v11 // (op 938) 

s_cmp_eq_i32 s4, 0 // (op 938) 
s_cselect_b32 s5, 0, -1 // (op 938) 

v_lshl_add_u32 v11, v0, 4, v3 // (op 938) 


s_and_b32 s4, s12, 31 // (op 938) 
v_lshrrev_b32 v0, 2, v9 // (op 938) 

s_lshl_b32 s8, 1, s4 // (op 938) 
v_and_b32 v9, 3, v0 // (op 938) 

s_add_i32 s9, s8, s5 // (op 938) 


v_and_b32 v0, 3, v1 // (op 938) 
s_and_b32 s5, s0, s9 // (op 938) 


v_lshl_add_u32 v3, v9, 2, v0 // (op 938) 


s_add_i32 s0, s1, s5 // (op 938) 


s_ashr_i32 s1, s0, s4 // (op 938) 


s_lshl_b32 s0, s12, 24 // (op 938) 
s_ashr_i32 s4, s0, 31 // (op 938) 

s_xor_b32 s0, s1, s4 // (op 938) 

s_sub_i32 s1, s0, s4 // (op 938) 


s_mul_hi_i32 s0, s1, s13 // (op 938) 
s_add_i32 s4, s0, s1 // (op 938) 

s_ashr_i32 s0, s4, 31 // (op 938) 
s_cmp_eq_i32 s13, 0 // (op 938) 
s_mov_b32 s5, scc // (op 938) 
s_cmp_eq_i32 s5, 0 // (op 938) 
s_cselect_b32 s8, 0, -1 // (op 938) 

s_and_b32 s5, s14, 31 // (op 938) 
s_lshl_b32 s9, 1, s5 // (op 938) 
s_add_i32 s18, s9, s8 // (op 938) 


s_and_b32 s8, s0, s18 // (op 938) 


s_add_i32 s0, s4, s8 // (op 938) 


s_ashr_i32 s4, s0, s5 // (op 938) 


s_lshl_b32 s0, s14, 24 // (op 938) 
s_ashr_i32 s5, s0, 31 // (op 938) 

s_xor_b32 s0, s4, s5 // (op 938) 

s_sub_i32 s4, s0, s5 // (op 938) 


s_mul_i32 s0, s1, s46 // (op 938) 
s_sub_u32 s5, s16, s0 // (op 938) 

s_mul_i32 s0, s4, s46 // (op 938) 
s_add_u32 s0, s0, s5 // (op 938) 

s_mul_hi_i32 s5, s16, s10 // (op 938) 
s_add_i32 s8, s5, s16 // (op 938) 

s_ashr_i32 s5, s8, 31 // (op 938) 
s_cmp_eq_i32 s10, 0 // (op 938) 
s_mov_b32 s9, scc // (op 938) 
s_cmp_eq_i32 s9, 0 // (op 938) 
s_cselect_b32 s18, 0, -1 // (op 938) 

s_and_b32 s9, s49, 31 // (op 938) 
s_lshl_b32 s19, 1, s9 // (op 938) 
s_add_i32 s20, s19, s18 // (op 938) 


s_and_b32 s18, s5, s20 // (op 938) 


s_add_i32 s5, s8, s18 // (op 938) 


s_ashr_i32 s8, s5, s9 // (op 938) 


s_lshl_b32 s5, s49, 24 // (op 938) 
s_ashr_i32 s9, s5, 31 // (op 938) 

s_xor_b32 s5, s8, s9 // (op 938) 

s_sub_i32 s8, s5, s9 // (op 938) 


s_add_u32 s18, s2, 16 // (op 938) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 938) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 938) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 938) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 938) 

s_mul_i32 s5, s46, s18 // (op 938) 
s_mul_i32 s9, s50, s5 // (op 938) 

s_mul_i32 s5, s8, s9 // (op 938) 

s_sub_u32 s9, s16, s5 // (op 938) 

s_mov_b32 s5, s9 // (op 938) 
s_mul_hi_i32 s20, s5, s15 // (op 938) 
s_add_i32 s21, s20, s5 // (op 938) 


s_ashr_i32 s5, s21, 31 // (op 938) 
s_cmp_eq_i32 s15, 0 // (op 938) 
s_mov_b32 s20, scc // (op 938) 
s_cmp_eq_i32 s20, 0 // (op 938) 
s_cselect_b32 s22, 0, -1 // (op 938) 

s_and_b32 s20, s48, 31 // (op 938) 
s_lshl_b32 s23, 1, s20 // (op 938) 
s_add_i32 s24, s23, s22 // (op 938) 


s_and_b32 s22, s5, s24 // (op 938) 


s_add_i32 s5, s21, s22 // (op 938) 


s_ashr_i32 s21, s5, s20 // (op 938) 


s_lshl_b32 s5, s48, 24 // (op 938) 
s_ashr_i32 s20, s5, 31 // (op 938) 

s_xor_b32 s5, s21, s20 // (op 938) 

s_sub_i32 s21, s5, s20 // (op 938) 


s_mul_i32 s5, s47, s46 // (op 938) 
s_add_u32 s22, s6, 16 // (op 938) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 938) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 938) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 938) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 938) 

s_sub_i32 s20, s22, s5 // (op 938) 

s_mul_i32 s22, s21, s20 // (op 938) 

s_sub_u32 s20, s9, s22 // (op 938) 


s_add_u32 s9, s20, s5 // (op 938) 


s_cmp_eq_i32 s8, 0 // (op 938) 
s_mov_b32 s5, scc // (op 938) 

s_cmp_eq_i32 s5, 0 // (op 938) 
s_cselect_b32 s8, s9, s0 // (op 938) 


s_mul_i32 s0, s4, s18 // (op 938) 


v_lshl_add_u32 v0, s8, 4, v11 // (op 938) 


s_sub_i32 s4, s1, s0 // (op 938) 


s_cmp_eq_i32 s5, 0 // (op 938) 
s_cselect_b32 s0, s21, s4 // (op 938) 



v_lshl_add_u32 v9, s0, 4, v3 // (op 938) 


v_mul_lo_u32 v11, v9, s38 // (op 938) 

v_add_lshl_u32 v9, v0, v11, 2 // (op 938) 


s_mul_i32 s1, s38, s3 // (op 940) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 940) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 940) least significant
s_mul_hi_u32 s5, s38, s2 // (op 940) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 940) most significant: sum
s_add_u32 s1, s1, s5 // (op 940) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 940) 

s_mov_b32 s20, s40 // (op 940) reg expression
s_mov_b32 s21, s41 // (op 940) reg expression
s_mov_b32 s22, s4 // (op 940) reg expression
s_mov_b32 s23, 131072 // (op 940) call()




buffer_store_dwordx4 v[12:15], v9, s[20:23], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_133
RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_133:
s_mul_hi_i32 s0, s16, s11 // (op 707) 
v_lshrrev_b32 v0, 6, v1 // (op 707) 
s_add_i32 s1, s0, s16 // (op 707) 

v_and_b32 v9, 63, v1 // (op 707) 
s_ashr_i32 s0, s1, 31 // (op 707) 
v_lshrrev_b32 v3, 4, v9 // (op 707) 
s_cmp_eq_i32 s11, 0 // (op 707) 
s_mov_b32 s4, scc // (op 707) 
v_lshlrev_b32 v4, 2, v3 // (op 707) 

s_cmp_eq_i32 s4, 0 // (op 707) 
s_cselect_b32 s5, 0, -1 // (op 707) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 707) 


s_and_b32 s4, s12, 31 // (op 707) 
v_lshrrev_b32 v0, 2, v9 // (op 707) 

s_lshl_b32 s8, 1, s4 // (op 707) 
v_and_b32 v9, 3, v0 // (op 707) 

s_add_i32 s9, s8, s5 // (op 707) 


v_and_b32 v0, 3, v1 // (op 707) 
s_and_b32 s5, s0, s9 // (op 707) 


v_lshl_add_u32 v4, v9, 2, v0 // (op 707) 


s_add_i32 s0, s1, s5 // (op 707) 


s_ashr_i32 s1, s0, s4 // (op 707) 


s_lshl_b32 s0, s12, 24 // (op 707) 
s_ashr_i32 s4, s0, 31 // (op 707) 

s_xor_b32 s0, s1, s4 // (op 707) 

s_sub_i32 s1, s0, s4 // (op 707) 


s_mul_hi_i32 s0, s1, s13 // (op 707) 
s_add_i32 s4, s0, s1 // (op 707) 

s_ashr_i32 s0, s4, 31 // (op 707) 
s_cmp_eq_i32 s13, 0 // (op 707) 
s_mov_b32 s5, scc // (op 707) 
s_cmp_eq_i32 s5, 0 // (op 707) 
s_cselect_b32 s8, 0, -1 // (op 707) 

s_and_b32 s5, s14, 31 // (op 707) 
s_lshl_b32 s9, 1, s5 // (op 707) 
s_add_i32 s18, s9, s8 // (op 707) 


s_and_b32 s19, s0, s18 // (op 707) 


s_add_i32 s0, s4, s19 // (op 707) 


s_ashr_i32 s4, s0, s5 // (op 707) 


s_lshl_b32 s0, s14, 24 // (op 707) 
s_ashr_i32 s5, s0, 31 // (op 707) 

s_xor_b32 s0, s4, s5 // (op 707) 

s_sub_i32 s4, s0, s5 // (op 707) 


s_mul_i32 s0, s1, s46 // (op 707) 
s_sub_u32 s5, s16, s0 // (op 707) 

s_mul_i32 s0, s4, s46 // (op 707) 
s_add_u32 s0, s0, s5 // (op 707) 

s_mul_hi_i32 s5, s16, s10 // (op 707) 
s_add_i32 s8, s5, s16 // (op 707) 

s_ashr_i32 s5, s8, 31 // (op 707) 
s_cmp_eq_i32 s10, 0 // (op 707) 
s_mov_b32 s9, scc // (op 707) 
s_cmp_eq_i32 s9, 0 // (op 707) 
s_cselect_b32 s18, 0, -1 // (op 707) 

s_and_b32 s9, s49, 31 // (op 707) 
s_lshl_b32 s19, 1, s9 // (op 707) 
s_add_i32 s24, s19, s18 // (op 707) 


s_and_b32 s18, s5, s24 // (op 707) 


s_add_i32 s5, s8, s18 // (op 707) 


s_ashr_i32 s8, s5, s9 // (op 707) 


s_lshl_b32 s5, s49, 24 // (op 707) 
s_ashr_i32 s9, s5, 31 // (op 707) 

s_xor_b32 s5, s8, s9 // (op 707) 

s_sub_i32 s8, s5, s9 // (op 707) 


s_add_u32 s18, s2, 16 // (op 707) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 707) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 707) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 707) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 4 // (op 707) 

s_mul_i32 s5, s46, s18 // (op 707) 
s_mul_i32 s9, s50, s5 // (op 707) 

s_mul_i32 s5, s8, s9 // (op 707) 

s_sub_u32 s9, s16, s5 // (op 707) 

s_mov_b32 s5, s9 // (op 707) 
s_mul_hi_i32 s24, s5, s15 // (op 707) 
s_add_i32 s25, s24, s5 // (op 707) 


s_ashr_i32 s5, s25, 31 // (op 707) 
s_cmp_eq_i32 s15, 0 // (op 707) 
s_mov_b32 s24, scc // (op 707) 
s_cmp_eq_i32 s24, 0 // (op 707) 
s_cselect_b32 s26, 0, -1 // (op 707) 

s_and_b32 s24, s48, 31 // (op 707) 
s_lshl_b32 s27, 1, s24 // (op 707) 
s_add_i32 s28, s27, s26 // (op 707) 


s_and_b32 s26, s5, s28 // (op 707) 


s_add_i32 s5, s25, s26 // (op 707) 


s_ashr_i32 s25, s5, s24 // (op 707) 


s_lshl_b32 s5, s48, 24 // (op 707) 
s_ashr_i32 s24, s5, 31 // (op 707) 

s_xor_b32 s5, s25, s24 // (op 707) 

s_sub_i32 s25, s5, s24 // (op 707) 


s_mul_i32 s5, s47, s46 // (op 707) 
s_add_u32 s26, s6, 16 // (op 707) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 707) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 707) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 707) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 707) 

s_sub_i32 s24, s26, s5 // (op 707) 

s_mul_i32 s26, s25, s24 // (op 707) 

s_sub_u32 s24, s9, s26 // (op 707) 


s_add_u32 s9, s24, s5 // (op 707) 


s_cmp_eq_i32 s8, 0 // (op 707) 
s_mov_b32 s5, scc // (op 707) 

s_cmp_eq_i32 s5, 0 // (op 707) 
s_cselect_b32 s8, s9, s0 // (op 707) 


s_mul_i32 s0, s4, s18 // (op 707) 


v_lshl_add_u32 v0, s8, 4, v3 // (op 707) 


s_sub_i32 s4, s1, s0 // (op 707) 


s_cmp_eq_i32 s5, 0 // (op 707) 
s_cselect_b32 s24, s25, s4 // (op 707) 



v_lshl_add_u32 v3, s24, 4, v4 // (op 707) 


v_mul_lo_u32 v9, v3, s42 // (op 707) 

v_add_lshl_u32 v3, v0, v9, 2 // (op 707) 


s_mul_i32 s1, s42, s3 // (op 709) most significant: low of low * high
s_mul_i32 s4, s43, s2 // (op 709) most significant: low of high * low
s_mul_i32 s0, s42, s2 // (op 709) least significant
s_mul_hi_u32 s5, s42, s2 // (op 709) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 709) most significant: sum
s_add_u32 s1, s1, s5 // (op 709) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 709) 

s_mov_b32 s24, s44 // (op 709) reg expression
s_mov_b32 s25, s45 // (op 709) reg expression
s_mov_b32 s26, s4 // (op 709) reg expression
s_mov_b32 s27, 131072 // (op 709) call()





buffer_load_dwordx4 v[4:7], v3, s[24:27], 0 offen // (op 145) Load value



v_accvgpr_read v12, a0 // (op 142) 
v_accvgpr_read v13, a1 // (op 142) 
v_accvgpr_read v14, a2 // (op 142) 
v_accvgpr_read v15, a3 // (op 142) 
v_mul_f32 v16, v10, v12 // (op 142) 
v_mul_f32 v17, v10, v13 // (op 142) 
v_mul_f32 v18, v10, v14 // (op 142) 
v_mul_f32 v19, v10, v15 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v8, v4 // (op 146) 
v_mul_f32 v13, v8, v5 // (op 146) 
v_mul_f32 v14, v8, v6 // (op 146) 
v_mul_f32 v15, v8, v7 // (op 146) 

v_add_f32 v4, v12, v16 // (op 143) 
v_add_f32 v5, v13, v17 // (op 143) 
v_add_f32 v6, v14, v18 // (op 143) 
v_add_f32 v7, v15, v19 // (op 143) 


s_mul_hi_i32 s0, s16, s11 // (op 955) 
v_lshrrev_b32 v0, 6, v1 // (op 955) 
s_add_i32 s1, s0, s16 // (op 955) 

v_and_b32 v3, 63, v1 // (op 955) 
s_ashr_i32 s0, s1, 31 // (op 955) 
v_lshrrev_b32 v9, 4, v3 // (op 955) 
s_cmp_eq_i32 s11, 0 // (op 955) 
s_mov_b32 s4, scc // (op 955) 
v_lshlrev_b32 v11, 2, v9 // (op 955) 

s_cmp_eq_i32 s4, 0 // (op 955) 
s_cselect_b32 s5, 0, -1 // (op 955) 

v_lshl_add_u32 v9, v0, 4, v11 // (op 955) 


s_and_b32 s4, s12, 31 // (op 955) 
v_lshrrev_b32 v0, 2, v3 // (op 955) 

s_lshl_b32 s8, 1, s4 // (op 955) 
v_and_b32 v3, 3, v0 // (op 955) 

s_add_i32 s9, s8, s5 // (op 955) 


v_and_b32 v0, 3, v1 // (op 955) 
s_and_b32 s5, s0, s9 // (op 955) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 955) 


s_add_i32 s0, s1, s5 // (op 955) 


s_ashr_i32 s1, s0, s4 // (op 955) 


s_lshl_b32 s0, s12, 24 // (op 955) 
s_ashr_i32 s4, s0, 31 // (op 955) 

s_xor_b32 s0, s1, s4 // (op 955) 

s_sub_i32 s1, s0, s4 // (op 955) 


s_mul_hi_i32 s0, s1, s13 // (op 955) 
s_add_i32 s4, s0, s1 // (op 955) 

s_ashr_i32 s0, s4, 31 // (op 955) 
s_cmp_eq_i32 s13, 0 // (op 955) 
s_mov_b32 s5, scc // (op 955) 
s_cmp_eq_i32 s5, 0 // (op 955) 
s_cselect_b32 s8, 0, -1 // (op 955) 

s_and_b32 s5, s14, 31 // (op 955) 
s_lshl_b32 s9, 1, s5 // (op 955) 
s_add_i32 s18, s9, s8 // (op 955) 


s_and_b32 s19, s0, s18 // (op 955) 


s_add_i32 s0, s4, s19 // (op 955) 


s_ashr_i32 s4, s0, s5 // (op 955) 


s_lshl_b32 s0, s14, 24 // (op 955) 
s_ashr_i32 s5, s0, 31 // (op 955) 

s_xor_b32 s0, s4, s5 // (op 955) 

s_sub_i32 s4, s0, s5 // (op 955) 


s_mul_i32 s0, s1, s46 // (op 955) 
s_sub_u32 s5, s16, s0 // (op 955) 

s_mul_i32 s0, s4, s46 // (op 955) 
s_add_u32 s0, s0, s5 // (op 955) 

s_mul_hi_i32 s5, s16, s10 // (op 955) 
s_add_i32 s8, s5, s16 // (op 955) 

s_ashr_i32 s5, s8, 31 // (op 955) 
s_cmp_eq_i32 s10, 0 // (op 955) 
s_mov_b32 s9, scc // (op 955) 
s_cmp_eq_i32 s9, 0 // (op 955) 
s_cselect_b32 s18, 0, -1 // (op 955) 

s_and_b32 s9, s49, 31 // (op 955) 
s_lshl_b32 s19, 1, s9 // (op 955) 
s_add_i32 s24, s19, s18 // (op 955) 


s_and_b32 s18, s5, s24 // (op 955) 


s_add_i32 s5, s8, s18 // (op 955) 


s_ashr_i32 s8, s5, s9 // (op 955) 


s_lshl_b32 s5, s49, 24 // (op 955) 
s_ashr_i32 s9, s5, 31 // (op 955) 

s_xor_b32 s5, s8, s9 // (op 955) 

s_sub_i32 s8, s5, s9 // (op 955) 


s_add_u32 s18, s2, 16 // (op 955) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 955) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 955) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 955) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 4 // (op 955) 

s_mul_i32 s5, s46, s18 // (op 955) 
s_mul_i32 s9, s50, s5 // (op 955) 

s_mul_i32 s5, s8, s9 // (op 955) 

s_sub_u32 s9, s16, s5 // (op 955) 

s_mov_b32 s5, s9 // (op 955) 
s_mul_hi_i32 s24, s5, s15 // (op 955) 
s_add_i32 s25, s24, s5 // (op 955) 


s_ashr_i32 s5, s25, 31 // (op 955) 
s_cmp_eq_i32 s15, 0 // (op 955) 
s_mov_b32 s24, scc // (op 955) 
s_cmp_eq_i32 s24, 0 // (op 955) 
s_cselect_b32 s26, 0, -1 // (op 955) 

s_and_b32 s24, s48, 31 // (op 955) 
s_lshl_b32 s27, 1, s24 // (op 955) 
s_add_i32 s28, s27, s26 // (op 955) 


s_and_b32 s26, s5, s28 // (op 955) 


s_add_i32 s5, s25, s26 // (op 955) 


s_ashr_i32 s25, s5, s24 // (op 955) 


s_lshl_b32 s5, s48, 24 // (op 955) 
s_ashr_i32 s24, s5, 31 // (op 955) 

s_xor_b32 s5, s25, s24 // (op 955) 

s_sub_i32 s25, s5, s24 // (op 955) 


s_mul_i32 s5, s47, s46 // (op 955) 
s_add_u32 s26, s6, 16 // (op 955) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 955) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 955) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 955) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 955) 

s_sub_i32 s24, s26, s5 // (op 955) 

s_mul_i32 s26, s25, s24 // (op 955) 

s_sub_u32 s24, s9, s26 // (op 955) 


s_add_u32 s9, s24, s5 // (op 955) 


s_cmp_eq_i32 s8, 0 // (op 955) 
s_mov_b32 s5, scc // (op 955) 

s_cmp_eq_i32 s5, 0 // (op 955) 
s_cselect_b32 s8, s9, s0 // (op 955) 


s_mul_i32 s0, s4, s18 // (op 955) 


v_lshl_add_u32 v0, s8, 4, v9 // (op 955) 


s_sub_i32 s4, s1, s0 // (op 955) 


s_cmp_eq_i32 s5, 0 // (op 955) 
s_cselect_b32 s24, s25, s4 // (op 955) 



v_lshl_add_u32 v3, s24, 4, v11 // (op 955) 


v_mul_lo_u32 v9, v3, s38 // (op 955) 

v_add_lshl_u32 v3, v0, v9, 2 // (op 955) 


s_mul_i32 s1, s38, s3 // (op 957) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 957) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 957) least significant
s_mul_hi_u32 s5, s38, s2 // (op 957) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 957) most significant: sum
s_add_u32 s1, s1, s5 // (op 957) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 957) 

s_mov_b32 s20, s40 // (op 957) reg expression
s_mov_b32 s21, s41 // (op 957) reg expression
s_mov_b32 s22, s4 // (op 957) reg expression
s_mov_b32 s23, 131072 // (op 957) call()




buffer_store_dwordx4 v[4:7], v3, s[20:23], 0 offen // (op 144) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_133:
















s_endpgm  // End of RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.LRR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end:
.size RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_, .LRR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end-RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 6400 // lds bytes
  .amdhsa_accum_offset 68
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
    DataType: F4
    DataTypeA: F4
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
    macrotile: [16, 16, 128]
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
  - .name: RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
    .symbol: RR_GEMM_TT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 6400
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 66
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

