// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_,@function
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 63, v1 // (op 765) 
v_and_b32 v3, 48, v1 // (op 765) 
v_sub_u32 v4, v0, v3 // (op 765) 

v_lshrrev_b32 v3, 4, v0 // (op 765) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 765) 


v_lshrrev_b32 v3, 6, v1 // (op 765) 
v_lshlrev_b32 v4, 4, v3 // (op 765) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 765) 


v_lshlrev_b32 v0, 2, v3 // (op 765) 

v_lshrrev_b32 v3, 3, v0 // (op 765) 

v_and_b32 v0, 15, v1 // (op 818) 
v_and_b32 v4, 63, v1 // (op 818) 
v_lshrrev_b32 v5, 4, v4 // (op 818) 

v_lshlrev_b32 v4, 5, v5 // (op 818) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 818) 


v_mul_lo_u32 v0, v5, 6 // (op 818) 

v_lshrrev_b32 v4, 3, v0 // (op 818) 

v_lshrrev_b32 v0, 6, v1 // (op 871) 
v_and_b32 v5, 15, v1 // (op 871) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 871) 


v_and_b32 v0, 48, v1 // (op 871) 
v_add_u32 v5, v0, v6 // (op 871) 


v_and_b32 v0, 15, v1 // (op 924) 
v_and_b32 v6, 63, v1 // (op 924) 
v_lshrrev_b32 v7, 4, v6 // (op 924) 

v_lshl_add_u32 v6, v0, 2, v7 // (op 924) 


v_and_b32 v0, 63, v1 // (op 982) 
v_and_b32 v7, 48, v1 // (op 982) 
v_sub_u32 v8, v0, v7 // (op 982) 

v_lshrrev_b32 v7, 4, v0 // (op 982) 

v_lshl_add_u32 v0, v7, 5, v8 // (op 982) 


v_lshrrev_b32 v7, 6, v1 // (op 982) 
v_lshlrev_b32 v8, 4, v7 // (op 982) 

v_lshl_add_u32 v7, v0, 4, v8 // (op 982) 


v_lshlrev_b32 v0, 2, v7 // (op 982) 

v_lshrrev_b32 v7, 3, v0 // (op 982) 

v_and_b32 v0, 15, v1 // (op 1006) 
v_and_b32 v8, 63, v1 // (op 1006) 
v_lshrrev_b32 v9, 4, v8 // (op 1006) 

v_lshlrev_b32 v8, 5, v9 // (op 1006) 

v_lshl_add_u32 v9, v0, 7, v8 // (op 1006) 


v_mul_lo_u32 v0, v9, 6 // (op 1006) 

v_lshrrev_b32 v8, 3, v0 // (op 1006) 

v_lshrrev_b32 v0, 6, v1 // (op 1030) 
v_and_b32 v9, 15, v1 // (op 1030) 
v_lshl_add_u32 v10, v0, 4, v9 // (op 1030) 


v_and_b32 v0, 48, v1 // (op 1030) 
v_add_u32 v9, v0, v10 // (op 1030) 


v_and_b32 v0, 15, v1 // (op 1054) 
v_and_b32 v10, 63, v1 // (op 1054) 
v_lshrrev_b32 v11, 4, v10 // (op 1054) 

v_lshl_add_u32 v10, v0, 2, v11 // (op 1054) 


v_mov_b32 v0, s6 // (op 2) Move value
v_mov_b32 v11, s7 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_mul_hi_i32 s0, s16, s9 // (op 475) 
v_lshlrev_b32 v12, 1, v1 // (op 475) 
s_add_i32 s1, s0, s16 // (op 475) 

s_ashr_i32 s0, s1, 31 // (op 475) 
s_cmp_eq_i32 s9, 0 // (op 475) 
s_mov_b32 s6, scc // (op 475) 
s_cmp_eq_i32 s6, 0 // (op 475) 
s_cselect_b32 s7, 0, -1 // (op 475) 

s_and_b32 s6, s10, 31 // (op 475) 
s_lshl_b32 s18, 1, s6 // (op 475) 
s_add_i32 s19, s18, s7 // (op 475) 


s_and_b32 s7, s0, s19 // (op 475) 


s_add_i32 s0, s1, s7 // (op 475) 


s_ashr_i32 s1, s0, s6 // (op 475) 


s_lshl_b32 s0, s10, 24 // (op 475) 
s_ashr_i32 s6, s0, 31 // (op 475) 

s_xor_b32 s0, s1, s6 // (op 475) 

s_sub_i32 s1, s0, s6 // (op 475) 


s_mul_hi_i32 s0, s1, s11 // (op 475) 
s_add_i32 s6, s0, s1 // (op 475) 

s_ashr_i32 s0, s6, 31 // (op 475) 
s_cmp_eq_i32 s11, 0 // (op 475) 
s_mov_b32 s7, scc // (op 475) 
s_cmp_eq_i32 s7, 0 // (op 475) 
s_cselect_b32 s18, 0, -1 // (op 475) 

s_and_b32 s7, s12, 31 // (op 475) 
s_lshl_b32 s19, 1, s7 // (op 475) 
s_add_i32 s51, s19, s18 // (op 475) 


s_and_b32 s18, s0, s51 // (op 475) 


s_add_i32 s0, s6, s18 // (op 475) 


s_ashr_i32 s6, s0, s7 // (op 475) 


s_lshl_b32 s0, s12, 24 // (op 475) 
s_ashr_i32 s7, s0, 31 // (op 475) 

s_xor_b32 s0, s6, s7 // (op 475) 

s_sub_i32 s6, s0, s7 // (op 475) 


s_mul_i32 s0, s1, s14 // (op 475) 

s_sub_u32 s1, s16, s0 // (op 475) 

s_mul_i32 s0, s6, s14 // (op 475) 
s_add_u32 s0, s0, s1 // (op 475) 


s_mul_hi_i32 s1, s16, s8 // (op 475) 
s_add_i32 s6, s1, s16 // (op 475) 

s_ashr_i32 s1, s6, 31 // (op 475) 
s_cmp_eq_i32 s8, 0 // (op 475) 
s_mov_b32 s7, scc // (op 475) 
s_cmp_eq_i32 s7, 0 // (op 475) 
s_cselect_b32 s18, 0, -1 // (op 475) 

s_and_b32 s7, s15, 31 // (op 475) 
s_lshl_b32 s19, 1, s7 // (op 475) 
s_add_i32 s51, s19, s18 // (op 475) 


s_and_b32 s18, s1, s51 // (op 475) 


s_add_i32 s1, s6, s18 // (op 475) 


s_ashr_i32 s6, s1, s7 // (op 475) 


s_lshl_b32 s1, s15, 24 // (op 475) 
s_ashr_i32 s7, s1, 31 // (op 475) 

s_xor_b32 s1, s6, s7 // (op 475) 

s_sub_i32 s6, s1, s7 // (op 475) 


s_add_u32 s18, s4, 16 // (op 475) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 475) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 475) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 475) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 475) 

s_mul_i32 s1, s14, s18 // (op 475) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 475) 

s_mul_i32 s1, s6, s7 // (op 475) 

s_sub_u32 s7, s16, s1 // (op 475) 

s_mov_b32 s1, s7 // (op 475) 
s_mul_hi_i32 s18, s1, s13 // (op 475) 
s_add_i32 s19, s18, s1 // (op 475) 


s_ashr_i32 s1, s19, 31 // (op 475) 
s_cmp_eq_i32 s13, 0 // (op 475) 
s_mov_b32 s18, scc // (op 475) 
s_cmp_eq_i32 s18, 0 // (op 475) 
s_cselect_b32 s51, 0, -1 // (op 475) 

s_and_b32 s18, s49, 31 // (op 475) 
s_lshl_b32 s52, 1, s18 // (op 475) 
s_add_i32 s53, s52, s51 // (op 475) 


s_and_b32 s51, s1, s53 // (op 475) 


s_add_i32 s1, s19, s51 // (op 475) 


s_ashr_i32 s19, s1, s18 // (op 475) 


s_lshl_b32 s1, s49, 24 // (op 475) 
s_ashr_i32 s18, s1, 31 // (op 475) 

s_xor_b32 s1, s19, s18 // (op 475) 

s_sub_i32 s19, s1, s18 // (op 475) 


s_mul_i32 s1, s48, s14 // (op 475) 
s_add_u32 s52, s20, 16 // (op 475) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 475) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 475) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 475) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 475) 

s_sub_i32 s18, s52, s1 // (op 475) 

s_mul_i32 s51, s19, s18 // (op 475) 


s_sub_u32 s18, s7, s51 // (op 475) 


s_add_u32 s7, s18, s1 // (op 475) 


s_cmp_eq_i32 s6, 0 // (op 475) 
s_mov_b32 s1, scc // (op 475) 

s_cmp_eq_i32 s1, 0 // (op 475) 
s_cselect_b32 s6, s7, s0 // (op 475) 



s_lshl_b32 s7, s6, 4 // (op 475) 

s_mov_b32 s1, 0 // (op 475) convert
s_mov_b32 s0, s7 // (op 475) convert

v_mul_lo_u32 v15, v12, s3 // (op 475) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 475) most significant: high of low * low
v_mul_lo_u32 v14, v12, s2 // (op 475) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 475) most significant: sum


v_mov_b32 v12, s0 // (op 475) 
v_mov_b32 v13, s1 // (op 475) 
v_add_co_u32 v16, s[6:7], v12, v14 // (op 475) least significant half
v_addc_co_u32 v17, s[6:7], v13, v15, s[6:7] // (op 475) most significant half



v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 475) 


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 475) 

v_mov_b32 v14, v12 // (op 475) convert

s_mul_i32 s1, s2, s25 // (op 477) most significant: low of low * high
s_mul_i32 s6, s3, s24 // (op 477) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 477) least significant
s_mul_hi_u32 s7, s2, s24 // (op 477) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 477) most significant: sum
s_add_u32 s1, s1, s7 // (op 477) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 477) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 477) 

s_mov_b32 s52, s22 // (op 477) reg expression
s_mov_b32 s53, s23 // (op 477) reg expression
s_mov_b32 s54, s0 // (op 477) reg expression
s_mov_b32 s55, 131072 // (op 477) call()

s_mul_hi_i32 s51, s16, s9 // (op 494) 
v_lshlrev_b32 v12, 1, v1 // (op 494) 
s_add_i32 s0, s51, s16 // (op 494) 

s_ashr_i32 s1, s0, 31 // (op 494) 
s_cmp_eq_i32 s9, 0 // (op 494) 
s_mov_b32 s51, scc // (op 494) 
s_cmp_eq_i32 s51, 0 // (op 494) 
s_cselect_b32 s6, 0, -1 // (op 494) 

s_and_b32 s7, s10, 31 // (op 494) 
s_lshl_b32 s51, 1, s7 // (op 494) 
s_add_i32 s18, s51, s6 // (op 494) 


s_and_b32 s6, s1, s18 // (op 494) 


s_add_i32 s1, s0, s6 // (op 494) 


s_ashr_i32 s0, s1, s7 // (op 494) 


s_lshl_b32 s1, s10, 24 // (op 494) 
s_ashr_i32 s51, s1, 31 // (op 494) 

s_xor_b32 s1, s0, s51 // (op 494) 

s_sub_i32 s0, s1, s51 // (op 494) 


s_mul_hi_i32 s1, s0, s11 // (op 494) 
s_add_i32 s51, s1, s0 // (op 494) 

s_ashr_i32 s1, s51, 31 // (op 494) 
s_cmp_eq_i32 s11, 0 // (op 494) 
s_mov_b32 s6, scc // (op 494) 
s_cmp_eq_i32 s6, 0 // (op 494) 
s_cselect_b32 s7, 0, -1 // (op 494) 

s_and_b32 s6, s12, 31 // (op 494) 
s_lshl_b32 s18, 1, s6 // (op 494) 
s_add_i32 s19, s18, s7 // (op 494) 


s_and_b32 s7, s1, s19 // (op 494) 


s_add_i32 s1, s51, s7 // (op 494) 


s_ashr_i32 s7, s1, s6 // (op 494) 


s_lshl_b32 s1, s12, 24 // (op 494) 
s_ashr_i32 s6, s1, 31 // (op 494) 

s_xor_b32 s1, s7, s6 // (op 494) 

s_sub_i32 s7, s1, s6 // (op 494) 


s_mul_i32 s1, s0, s14 // (op 494) 

s_sub_u32 s0, s16, s1 // (op 494) 

s_mul_i32 s1, s7, s14 // (op 494) 
s_add_u32 s1, s1, s0 // (op 494) 


s_mul_hi_i32 s0, s16, s8 // (op 494) 
s_add_i32 s51, s0, s16 // (op 494) 

s_ashr_i32 s0, s51, 31 // (op 494) 
s_cmp_eq_i32 s8, 0 // (op 494) 
s_mov_b32 s6, scc // (op 494) 
s_cmp_eq_i32 s6, 0 // (op 494) 
s_cselect_b32 s7, 0, -1 // (op 494) 

s_and_b32 s6, s15, 31 // (op 494) 
s_lshl_b32 s18, 1, s6 // (op 494) 
s_add_i32 s19, s18, s7 // (op 494) 


s_and_b32 s7, s0, s19 // (op 494) 


s_add_i32 s0, s51, s7 // (op 494) 


s_ashr_i32 s7, s0, s6 // (op 494) 


s_lshl_b32 s0, s15, 24 // (op 494) 
s_ashr_i32 s6, s0, 31 // (op 494) 

s_xor_b32 s0, s7, s6 // (op 494) 

s_sub_i32 s7, s0, s6 // (op 494) 


s_add_u32 s18, s4, 16 // (op 494) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 494) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 494) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 494) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 494) 

s_mul_i32 s0, s14, s18 // (op 494) 

s_mul_i32 s6, s50, s0 // (op 494) 

s_mul_i32 s0, s7, s6 // (op 494) 

s_sub_u32 s6, s16, s0 // (op 494) 

s_mov_b32 s0, s6 // (op 494) 
s_mul_hi_i32 s51, s0, s13 // (op 494) 
s_add_i32 s18, s51, s0 // (op 494) 


s_ashr_i32 s0, s18, 31 // (op 494) 
s_cmp_eq_i32 s13, 0 // (op 494) 
s_mov_b32 s19, scc // (op 494) 
s_cmp_eq_i32 s19, 0 // (op 494) 
s_cselect_b32 s51, 0, -1 // (op 494) 

s_and_b32 s19, s49, 31 // (op 494) 
s_lshl_b32 s56, 1, s19 // (op 494) 
s_add_i32 s57, s56, s51 // (op 494) 


s_and_b32 s51, s0, s57 // (op 494) 


s_add_i32 s0, s18, s51 // (op 494) 


s_ashr_i32 s18, s0, s19 // (op 494) 


s_lshl_b32 s0, s49, 24 // (op 494) 
s_ashr_i32 s19, s0, 31 // (op 494) 

s_xor_b32 s0, s18, s19 // (op 494) 

s_sub_i32 s18, s0, s19 // (op 494) 


s_mul_i32 s0, s48, s14 // (op 494) 
s_add_u32 s56, s20, 16 // (op 494) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 494) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 494) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 494) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 494) 

s_sub_i32 s19, s56, s0 // (op 494) 

s_mul_i32 s51, s18, s19 // (op 494) 


s_sub_u32 s18, s6, s51 // (op 494) 


s_add_u32 s6, s18, s0 // (op 494) 


s_cmp_eq_i32 s7, 0 // (op 494) 
s_mov_b32 s0, scc // (op 494) 

s_cmp_eq_i32 s0, 0 // (op 494) 
s_cselect_b32 s7, s6, s1 // (op 494) 



s_lshl_b32 s6, s7, 4 // (op 494) 

s_mov_b32 s1, 0 // (op 494) convert
s_mov_b32 s0, s6 // (op 494) convert

v_mul_lo_u32 v17, v12, s3 // (op 494) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 494) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 494) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 494) most significant: sum


v_mov_b32 v15, s0 // (op 494) 
v_mov_b32 v12, s1 // (op 494) 
v_add_co_u32 v18, s[6:7], v15, v16 // (op 494) least significant half
v_addc_co_u32 v19, s[6:7], v12, v17, s[6:7] // (op 494) most significant half



v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 494) 


v_lshrrev_b64 v[12:13], 3, v[18:19] // (op 494) 

v_mov_b32 v15, v12 // (op 494) convert

s_mul_i32 s1, s2, s25 // (op 496) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 496) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 496) least significant
s_mul_hi_u32 s6, s2, s24 // (op 496) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 496) most significant: sum
s_add_u32 s1, s1, s6 // (op 496) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 496) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 496) 

s_mov_b32 s52, s22 // (op 496) reg expression
s_mov_b32 s53, s23 // (op 496) reg expression
s_mov_b32 s54, s0 // (op 496) reg expression
s_mov_b32 s55, 131072 // (op 496) call()

s_mul_hi_i32 s51, s16, s9 // (op 539) 
v_lshrrev_b32 v12, 2, v1 // (op 539) 
s_add_i32 s0, s51, s16 // (op 539) 

v_and_b32 v13, -4, v1 // (op 539) 
s_ashr_i32 s1, s0, 31 // (op 539) 
v_sub_u32 v16, v1, v13 // (op 539) 

s_cmp_eq_i32 s9, 0 // (op 539) 
s_mov_b32 s51, scc // (op 539) 
v_lshlrev_b32 v13, 5, v16 // (op 539) 

s_cmp_eq_i32 s51, 0 // (op 539) 
s_cselect_b32 s6, 0, -1 // (op 539) 

v_mov_b32 v17, 0 // (op 539) convert
v_mov_b32 v16, v13 // (op 539) convert

s_and_b32 s7, s10, 31 // (op 539) 
s_lshl_b32 s51, 1, s7 // (op 539) 
s_add_i32 s18, s51, s6 // (op 539) 


s_and_b32 s6, s1, s18 // (op 539) 


s_add_i32 s1, s0, s6 // (op 539) 


s_ashr_i32 s0, s1, s7 // (op 539) 


s_lshl_b32 s1, s10, 24 // (op 539) 
s_ashr_i32 s51, s1, 31 // (op 539) 

s_xor_b32 s1, s0, s51 // (op 539) 

s_sub_i32 s0, s1, s51 // (op 539) 


s_mul_hi_i32 s1, s0, s11 // (op 539) 
s_add_i32 s51, s1, s0 // (op 539) 

s_ashr_i32 s1, s51, 31 // (op 539) 
s_cmp_eq_i32 s11, 0 // (op 539) 
s_mov_b32 s6, scc // (op 539) 
s_cmp_eq_i32 s6, 0 // (op 539) 
s_cselect_b32 s7, 0, -1 // (op 539) 

s_and_b32 s6, s12, 31 // (op 539) 
s_lshl_b32 s18, 1, s6 // (op 539) 
s_add_i32 s19, s18, s7 // (op 539) 


s_and_b32 s7, s1, s19 // (op 539) 


s_add_i32 s1, s51, s7 // (op 539) 


s_ashr_i32 s7, s1, s6 // (op 539) 


s_lshl_b32 s1, s12, 24 // (op 539) 
s_ashr_i32 s6, s1, 31 // (op 539) 

s_xor_b32 s1, s7, s6 // (op 539) 

s_sub_i32 s7, s1, s6 // (op 539) 


s_add_u32 s18, s4, 16 // (op 539) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 539) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 539) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 539) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 539) 

s_mul_i32 s1, s7, s18 // (op 539) 

s_sub_i32 s51, s0, s1 // (op 539) 


s_mul_hi_i32 s0, s16, s8 // (op 539) 
s_add_i32 s1, s0, s16 // (op 539) 

s_ashr_i32 s0, s1, 31 // (op 539) 
s_cmp_eq_i32 s8, 0 // (op 539) 
s_mov_b32 s6, scc // (op 539) 
s_cmp_eq_i32 s6, 0 // (op 539) 
s_cselect_b32 s7, 0, -1 // (op 539) 

s_and_b32 s6, s15, 31 // (op 539) 
s_lshl_b32 s56, 1, s6 // (op 539) 
s_add_i32 s57, s56, s7 // (op 539) 


s_and_b32 s7, s0, s57 // (op 539) 


s_add_i32 s0, s1, s7 // (op 539) 


s_ashr_i32 s1, s0, s6 // (op 539) 


s_lshl_b32 s0, s15, 24 // (op 539) 
s_ashr_i32 s6, s0, 31 // (op 539) 

s_xor_b32 s0, s1, s6 // (op 539) 

s_sub_i32 s1, s0, s6 // (op 539) 


s_mul_i32 s0, s14, s18 // (op 539) 

s_mul_i32 s6, s50, s0 // (op 539) 

s_mul_i32 s0, s1, s6 // (op 539) 

s_sub_u32 s6, s16, s0 // (op 539) 

s_mul_hi_i32 s0, s6, s13 // (op 539) 
s_add_i32 s7, s0, s6 // (op 539) 


s_ashr_i32 s0, s7, 31 // (op 539) 
s_cmp_eq_i32 s13, 0 // (op 539) 
s_mov_b32 s6, scc // (op 539) 
s_cmp_eq_i32 s6, 0 // (op 539) 
s_cselect_b32 s18, 0, -1 // (op 539) 

s_and_b32 s6, s49, 31 // (op 539) 
s_lshl_b32 s19, 1, s6 // (op 539) 
s_add_i32 s56, s19, s18 // (op 539) 


s_and_b32 s18, s0, s56 // (op 539) 


s_add_i32 s0, s7, s18 // (op 539) 


s_ashr_i32 s7, s0, s6 // (op 539) 


s_lshl_b32 s0, s49, 24 // (op 539) 
s_ashr_i32 s6, s0, 31 // (op 539) 

s_xor_b32 s0, s7, s6 // (op 539) 

s_sub_i32 s7, s0, s6 // (op 539) 


s_cmp_eq_i32 s1, 0 // (op 539) 
s_mov_b32 s0, scc // (op 539) 

s_cmp_eq_i32 s0, 0 // (op 539) 
s_cselect_b32 s1, s7, s51 // (op 539) 



v_lshl_add_u32 v13, s1, 4, v12 // (op 539) 


v_mul_lo_u32 v19, v13, s27 // (op 539) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 539) most significant: high of low * low
v_mul_lo_u32 v18, v13, s26 // (op 539) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 539) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 539) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 539) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 539) most significant: low of high * low
v_mul_hi_u32 v18, v12, 6 // (op 539) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 539) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 539) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 539) 

v_mov_b32 v16, v12 // (op 539) convert

s_mul_i32 s1, s5, 6 // (op 541) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 541) least significant
s_mul_hi_u32 s51, s4, 6 // (op 541) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 541) most significant: sum

s_mul_i32 s7, s26, s1 // (op 541) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 541) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 541) least significant
s_mul_hi_u32 s18, s26, s0 // (op 541) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 541) most significant: sum
s_add_u32 s7, s7, s18 // (op 541) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 541) 

s_mov_b32 s56, s28 // (op 541) reg expression
s_mov_b32 s57, s29 // (op 541) reg expression
s_mov_b32 s58, s0 // (op 541) reg expression
s_mov_b32 s59, 131072 // (op 541) call()

s_mul_hi_i32 s51, s16, s9 // (op 558) 
v_lshrrev_b32 v12, 2, v1 // (op 558) 
s_add_i32 s0, s51, s16 // (op 558) 

v_and_b32 v13, -4, v1 // (op 558) 
s_ashr_i32 s1, s0, 31 // (op 558) 
v_sub_u32 v17, v1, v13 // (op 558) 

s_cmp_eq_i32 s9, 0 // (op 558) 
s_mov_b32 s51, scc // (op 558) 
v_lshlrev_b32 v13, 5, v17 // (op 558) 

s_cmp_eq_i32 s51, 0 // (op 558) 
s_cselect_b32 s6, 0, -1 // (op 558) 

v_mov_b32 v19, 0 // (op 558) convert
v_mov_b32 v18, v13 // (op 558) convert

s_and_b32 s7, s10, 31 // (op 558) 
s_lshl_b32 s51, 1, s7 // (op 558) 
s_add_i32 s18, s51, s6 // (op 558) 


s_and_b32 s6, s1, s18 // (op 558) 


s_add_i32 s1, s0, s6 // (op 558) 


s_ashr_i32 s0, s1, s7 // (op 558) 


s_lshl_b32 s1, s10, 24 // (op 558) 
s_ashr_i32 s51, s1, 31 // (op 558) 

s_xor_b32 s1, s0, s51 // (op 558) 

s_sub_i32 s0, s1, s51 // (op 558) 


s_mul_hi_i32 s1, s0, s11 // (op 558) 
s_add_i32 s51, s1, s0 // (op 558) 

s_ashr_i32 s1, s51, 31 // (op 558) 
s_cmp_eq_i32 s11, 0 // (op 558) 
s_mov_b32 s6, scc // (op 558) 
s_cmp_eq_i32 s6, 0 // (op 558) 
s_cselect_b32 s7, 0, -1 // (op 558) 

s_and_b32 s6, s12, 31 // (op 558) 
s_lshl_b32 s18, 1, s6 // (op 558) 
s_add_i32 s19, s18, s7 // (op 558) 


s_and_b32 s7, s1, s19 // (op 558) 


s_add_i32 s1, s51, s7 // (op 558) 


s_ashr_i32 s7, s1, s6 // (op 558) 


s_lshl_b32 s1, s12, 24 // (op 558) 
s_ashr_i32 s6, s1, 31 // (op 558) 

s_xor_b32 s1, s7, s6 // (op 558) 

s_sub_i32 s7, s1, s6 // (op 558) 


s_add_u32 s18, s4, 16 // (op 558) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 558) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 558) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 558) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 558) 

s_mul_i32 s1, s7, s18 // (op 558) 

s_sub_i32 s51, s0, s1 // (op 558) 


s_mul_hi_i32 s0, s16, s8 // (op 558) 
s_add_i32 s1, s0, s16 // (op 558) 

s_ashr_i32 s0, s1, 31 // (op 558) 
s_cmp_eq_i32 s8, 0 // (op 558) 
s_mov_b32 s6, scc // (op 558) 
s_cmp_eq_i32 s6, 0 // (op 558) 
s_cselect_b32 s7, 0, -1 // (op 558) 

s_and_b32 s6, s15, 31 // (op 558) 
s_lshl_b32 s60, 1, s6 // (op 558) 
s_add_i32 s61, s60, s7 // (op 558) 


s_and_b32 s7, s0, s61 // (op 558) 


s_add_i32 s0, s1, s7 // (op 558) 


s_ashr_i32 s1, s0, s6 // (op 558) 


s_lshl_b32 s0, s15, 24 // (op 558) 
s_ashr_i32 s6, s0, 31 // (op 558) 

s_xor_b32 s0, s1, s6 // (op 558) 

s_sub_i32 s1, s0, s6 // (op 558) 


s_mul_i32 s0, s14, s18 // (op 558) 

s_mul_i32 s6, s50, s0 // (op 558) 

s_mul_i32 s0, s1, s6 // (op 558) 

s_sub_u32 s6, s16, s0 // (op 558) 

s_mul_hi_i32 s0, s6, s13 // (op 558) 
s_add_i32 s7, s0, s6 // (op 558) 


s_ashr_i32 s0, s7, 31 // (op 558) 
s_cmp_eq_i32 s13, 0 // (op 558) 
s_mov_b32 s6, scc // (op 558) 
s_cmp_eq_i32 s6, 0 // (op 558) 
s_cselect_b32 s18, 0, -1 // (op 558) 

s_and_b32 s6, s49, 31 // (op 558) 
s_lshl_b32 s19, 1, s6 // (op 558) 
s_add_i32 s60, s19, s18 // (op 558) 


s_and_b32 s18, s0, s60 // (op 558) 


s_add_i32 s0, s7, s18 // (op 558) 


s_ashr_i32 s7, s0, s6 // (op 558) 


s_lshl_b32 s0, s49, 24 // (op 558) 
s_ashr_i32 s6, s0, 31 // (op 558) 

s_xor_b32 s0, s7, s6 // (op 558) 

s_sub_i32 s7, s0, s6 // (op 558) 


s_cmp_eq_i32 s1, 0 // (op 558) 
s_mov_b32 s0, scc // (op 558) 

s_cmp_eq_i32 s0, 0 // (op 558) 
s_cselect_b32 s1, s7, s51 // (op 558) 



v_lshl_add_u32 v13, s1, 4, v12 // (op 558) 


v_mul_lo_u32 v21, v13, s27 // (op 558) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 558) most significant: high of low * low
v_mul_lo_u32 v20, v13, s26 // (op 558) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 558) most significant: sum


v_add_co_u32 v12, s[0:1], v18, v20 // (op 558) least significant half
v_addc_co_u32 v13, s[0:1], v19, v21, s[0:1] // (op 558) most significant half



v_mul_lo_u32 v19, v13, 6 // (op 558) most significant: low of high * low
v_mul_hi_u32 v17, v12, 6 // (op 558) most significant: high of low * low
v_mul_lo_u32 v18, v12, 6 // (op 558) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 558) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[18:19] // (op 558) 

v_mov_b32 v17, v12 // (op 558) convert

s_mul_i32 s1, s5, 6 // (op 560) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 560) least significant
s_mul_hi_u32 s51, s4, 6 // (op 560) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 560) most significant: sum

s_mul_i32 s7, s26, s1 // (op 560) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 560) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 560) least significant
s_mul_hi_u32 s18, s26, s0 // (op 560) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 560) most significant: sum
s_add_u32 s7, s7, s18 // (op 560) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 560) 

s_mov_b32 s56, s28 // (op 560) reg expression
s_mov_b32 s57, s29 // (op 560) reg expression
s_mov_b32 s58, s0 // (op 560) reg expression
s_mov_b32 s59, 131072 // (op 560) call()

s_mul_hi_i32 s51, s16, s9 // (op 603) 
v_and_b32 v12, -16, v1 // (op 603) 
s_add_i32 s0, s51, s16 // (op 603) 

v_sub_u32 v13, v1, v12 // (op 603) 

s_ashr_i32 s1, s0, 31 // (op 603) 
v_lshrrev_b32 v12, 4, v1 // (op 603) 
s_cmp_eq_i32 s9, 0 // (op 603) 
s_mov_b32 s51, scc // (op 603) 
s_cmp_eq_i32 s51, 0 // (op 603) 
s_cselect_b32 s6, 0, -1 // (op 603) 

s_and_b32 s7, s10, 31 // (op 603) 
s_lshl_b32 s51, 1, s7 // (op 603) 
s_add_i32 s18, s51, s6 // (op 603) 


s_and_b32 s6, s1, s18 // (op 603) 


s_add_i32 s1, s0, s6 // (op 603) 


s_ashr_i32 s0, s1, s7 // (op 603) 


s_lshl_b32 s1, s10, 24 // (op 603) 
s_ashr_i32 s51, s1, 31 // (op 603) 

s_xor_b32 s1, s0, s51 // (op 603) 

s_sub_i32 s0, s1, s51 // (op 603) 


s_mul_hi_i32 s1, s0, s11 // (op 603) 
s_add_i32 s51, s1, s0 // (op 603) 

s_ashr_i32 s1, s51, 31 // (op 603) 
s_cmp_eq_i32 s11, 0 // (op 603) 
s_mov_b32 s6, scc // (op 603) 
s_cmp_eq_i32 s6, 0 // (op 603) 
s_cselect_b32 s7, 0, -1 // (op 603) 

s_and_b32 s6, s12, 31 // (op 603) 
s_lshl_b32 s18, 1, s6 // (op 603) 
s_add_i32 s19, s18, s7 // (op 603) 


s_and_b32 s7, s1, s19 // (op 603) 


s_add_i32 s1, s51, s7 // (op 603) 


s_ashr_i32 s7, s1, s6 // (op 603) 


s_lshl_b32 s1, s12, 24 // (op 603) 
s_ashr_i32 s6, s1, 31 // (op 603) 

s_xor_b32 s1, s7, s6 // (op 603) 

s_sub_i32 s7, s1, s6 // (op 603) 


s_mul_i32 s1, s0, s14 // (op 603) 

s_sub_u32 s0, s16, s1 // (op 603) 

s_mul_i32 s1, s7, s14 // (op 603) 
s_add_u32 s1, s1, s0 // (op 603) 


s_mul_hi_i32 s0, s16, s8 // (op 603) 
s_add_i32 s51, s0, s16 // (op 603) 

s_ashr_i32 s0, s51, 31 // (op 603) 
s_cmp_eq_i32 s8, 0 // (op 603) 
s_mov_b32 s6, scc // (op 603) 
s_cmp_eq_i32 s6, 0 // (op 603) 
s_cselect_b32 s7, 0, -1 // (op 603) 

s_and_b32 s6, s15, 31 // (op 603) 
s_lshl_b32 s18, 1, s6 // (op 603) 
s_add_i32 s19, s18, s7 // (op 603) 


s_and_b32 s7, s0, s19 // (op 603) 


s_add_i32 s0, s51, s7 // (op 603) 


s_ashr_i32 s7, s0, s6 // (op 603) 


s_lshl_b32 s0, s15, 24 // (op 603) 
s_ashr_i32 s6, s0, 31 // (op 603) 

s_xor_b32 s0, s7, s6 // (op 603) 

s_sub_i32 s7, s0, s6 // (op 603) 


s_add_u32 s18, s4, 16 // (op 603) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 603) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 603) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 603) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 603) 

s_mul_i32 s0, s14, s18 // (op 603) 

s_mul_i32 s6, s50, s0 // (op 603) 

s_mul_i32 s0, s7, s6 // (op 603) 

s_sub_u32 s6, s16, s0 // (op 603) 

s_mov_b32 s0, s6 // (op 603) 
s_mul_hi_i32 s51, s0, s13 // (op 603) 
s_add_i32 s18, s51, s0 // (op 603) 


s_ashr_i32 s0, s18, 31 // (op 603) 
s_cmp_eq_i32 s13, 0 // (op 603) 
s_mov_b32 s19, scc // (op 603) 
s_cmp_eq_i32 s19, 0 // (op 603) 
s_cselect_b32 s51, 0, -1 // (op 603) 

s_and_b32 s19, s49, 31 // (op 603) 
s_lshl_b32 s60, 1, s19 // (op 603) 
s_add_i32 s61, s60, s51 // (op 603) 


s_and_b32 s51, s0, s61 // (op 603) 


s_add_i32 s0, s18, s51 // (op 603) 


s_ashr_i32 s18, s0, s19 // (op 603) 


s_lshl_b32 s0, s49, 24 // (op 603) 
s_ashr_i32 s19, s0, 31 // (op 603) 

s_xor_b32 s0, s18, s19 // (op 603) 

s_sub_i32 s18, s0, s19 // (op 603) 


s_mul_i32 s0, s48, s14 // (op 603) 
s_add_u32 s60, s20, 16 // (op 603) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 603) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 603) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 603) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 603) 

s_sub_i32 s19, s60, s0 // (op 603) 

s_mul_i32 s51, s18, s19 // (op 603) 


s_sub_u32 s18, s6, s51 // (op 603) 


s_add_u32 s6, s18, s0 // (op 603) 


s_cmp_eq_i32 s7, 0 // (op 603) 
s_mov_b32 s0, scc // (op 603) 

s_cmp_eq_i32 s0, 0 // (op 603) 
s_cselect_b32 s7, s6, s1 // (op 603) 



v_lshl_add_u32 v18, s7, 4, v13 // (op 603) 


v_mul_lo_u32 v13, v12, s30 // (op 603) 

v_add_u32 v12, v18, v13 // (op 603) 


s_mul_i32 s1, s30, s35 // (op 605) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 605) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 605) least significant
s_mul_hi_u32 s6, s30, s34 // (op 605) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 605) most significant: sum
s_add_u32 s1, s1, s6 // (op 605) most significant: sum


s_mov_b32 s60, s32 // (op 605) reg expression
s_mov_b32 s61, s33 // (op 605) reg expression
s_mov_b32 s62, s0 // (op 605) reg expression
s_mov_b32 s63, 131072 // (op 605) call()

s_mul_hi_i32 s51, s16, s9 // (op 622) 
v_and_b32 v13, -16, v1 // (op 622) 
s_add_i32 s0, s51, s16 // (op 622) 

v_sub_u32 v18, v1, v13 // (op 622) 

s_ashr_i32 s1, s0, 31 // (op 622) 
v_lshrrev_b32 v13, 4, v1 // (op 622) 
s_cmp_eq_i32 s9, 0 // (op 622) 
s_mov_b32 s51, scc // (op 622) 
s_cmp_eq_i32 s51, 0 // (op 622) 
s_cselect_b32 s6, 0, -1 // (op 622) 

s_and_b32 s7, s10, 31 // (op 622) 
s_lshl_b32 s51, 1, s7 // (op 622) 
s_add_i32 s18, s51, s6 // (op 622) 


s_and_b32 s6, s1, s18 // (op 622) 


s_add_i32 s1, s0, s6 // (op 622) 


s_ashr_i32 s0, s1, s7 // (op 622) 


s_lshl_b32 s1, s10, 24 // (op 622) 
s_ashr_i32 s51, s1, 31 // (op 622) 

s_xor_b32 s1, s0, s51 // (op 622) 

s_sub_i32 s0, s1, s51 // (op 622) 


s_mul_hi_i32 s1, s0, s11 // (op 622) 
s_add_i32 s51, s1, s0 // (op 622) 

s_ashr_i32 s1, s51, 31 // (op 622) 
s_cmp_eq_i32 s11, 0 // (op 622) 
s_mov_b32 s6, scc // (op 622) 
s_cmp_eq_i32 s6, 0 // (op 622) 
s_cselect_b32 s7, 0, -1 // (op 622) 

s_and_b32 s6, s12, 31 // (op 622) 
s_lshl_b32 s18, 1, s6 // (op 622) 
s_add_i32 s19, s18, s7 // (op 622) 


s_and_b32 s7, s1, s19 // (op 622) 


s_add_i32 s1, s51, s7 // (op 622) 


s_ashr_i32 s7, s1, s6 // (op 622) 


s_lshl_b32 s1, s12, 24 // (op 622) 
s_ashr_i32 s6, s1, 31 // (op 622) 

s_xor_b32 s1, s7, s6 // (op 622) 

s_sub_i32 s7, s1, s6 // (op 622) 


s_mul_i32 s1, s0, s14 // (op 622) 

s_sub_u32 s0, s16, s1 // (op 622) 

s_mul_i32 s1, s7, s14 // (op 622) 
s_add_u32 s1, s1, s0 // (op 622) 


s_mul_hi_i32 s0, s16, s8 // (op 622) 
s_add_i32 s51, s0, s16 // (op 622) 

s_ashr_i32 s0, s51, 31 // (op 622) 
s_cmp_eq_i32 s8, 0 // (op 622) 
s_mov_b32 s6, scc // (op 622) 
s_cmp_eq_i32 s6, 0 // (op 622) 
s_cselect_b32 s7, 0, -1 // (op 622) 

s_and_b32 s6, s15, 31 // (op 622) 
s_lshl_b32 s18, 1, s6 // (op 622) 
s_add_i32 s19, s18, s7 // (op 622) 


s_and_b32 s7, s0, s19 // (op 622) 


s_add_i32 s0, s51, s7 // (op 622) 


s_ashr_i32 s7, s0, s6 // (op 622) 


s_lshl_b32 s0, s15, 24 // (op 622) 
s_ashr_i32 s6, s0, 31 // (op 622) 

s_xor_b32 s0, s7, s6 // (op 622) 

s_sub_i32 s7, s0, s6 // (op 622) 


s_add_u32 s18, s4, 16 // (op 622) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 622) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 622) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 622) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 622) 

s_mul_i32 s0, s14, s18 // (op 622) 

s_mul_i32 s6, s50, s0 // (op 622) 

s_mul_i32 s0, s7, s6 // (op 622) 

s_sub_u32 s6, s16, s0 // (op 622) 

s_mov_b32 s0, s6 // (op 622) 
s_mul_hi_i32 s51, s0, s13 // (op 622) 
s_add_i32 s18, s51, s0 // (op 622) 


s_ashr_i32 s0, s18, 31 // (op 622) 
s_cmp_eq_i32 s13, 0 // (op 622) 
s_mov_b32 s19, scc // (op 622) 
s_cmp_eq_i32 s19, 0 // (op 622) 
s_cselect_b32 s51, 0, -1 // (op 622) 

s_and_b32 s19, s49, 31 // (op 622) 
s_lshl_b32 s64, 1, s19 // (op 622) 
s_add_i32 s65, s64, s51 // (op 622) 


s_and_b32 s51, s0, s65 // (op 622) 


s_add_i32 s0, s18, s51 // (op 622) 


s_ashr_i32 s18, s0, s19 // (op 622) 


s_lshl_b32 s0, s49, 24 // (op 622) 
s_ashr_i32 s19, s0, 31 // (op 622) 

s_xor_b32 s0, s18, s19 // (op 622) 

s_sub_i32 s18, s0, s19 // (op 622) 


s_mul_i32 s0, s48, s14 // (op 622) 
s_add_u32 s64, s20, 16 // (op 622) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 622) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 622) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 622) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 622) 

s_sub_i32 s19, s64, s0 // (op 622) 

s_mul_i32 s51, s18, s19 // (op 622) 


s_sub_u32 s18, s6, s51 // (op 622) 


s_add_u32 s6, s18, s0 // (op 622) 


s_cmp_eq_i32 s7, 0 // (op 622) 
s_mov_b32 s0, scc // (op 622) 

s_cmp_eq_i32 s0, 0 // (op 622) 
s_cselect_b32 s7, s6, s1 // (op 622) 



v_lshl_add_u32 v19, s7, 4, v18 // (op 622) 


v_mul_lo_u32 v18, v13, s30 // (op 622) 

v_add_u32 v13, v19, v18 // (op 622) 


s_mul_i32 s1, s30, s35 // (op 624) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 624) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 624) least significant
s_mul_hi_u32 s6, s30, s34 // (op 624) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 624) most significant: sum
s_add_u32 s1, s1, s6 // (op 624) most significant: sum


s_mov_b32 s60, s32 // (op 624) reg expression
s_mov_b32 s61, s33 // (op 624) reg expression
s_mov_b32 s62, s0 // (op 624) reg expression
s_mov_b32 s63, 131072 // (op 624) call()

s_mul_hi_i32 s51, s16, s9 // (op 667) 
v_lshrrev_b32 v18, 2, v1 // (op 667) 
s_add_i32 s0, s51, s16 // (op 667) 

v_and_b32 v19, -4, v1 // (op 667) 
s_ashr_i32 s1, s0, 31 // (op 667) 
v_sub_u32 v20, v1, v19 // (op 667) 

s_cmp_eq_i32 s9, 0 // (op 667) 
s_mov_b32 s51, scc // (op 667) 
s_cmp_eq_i32 s51, 0 // (op 667) 
s_cselect_b32 s6, 0, -1 // (op 667) 

s_and_b32 s7, s10, 31 // (op 667) 
s_lshl_b32 s51, 1, s7 // (op 667) 
s_add_i32 s18, s51, s6 // (op 667) 


s_and_b32 s6, s1, s18 // (op 667) 


s_add_i32 s1, s0, s6 // (op 667) 


s_ashr_i32 s0, s1, s7 // (op 667) 


s_lshl_b32 s1, s10, 24 // (op 667) 
s_ashr_i32 s51, s1, 31 // (op 667) 

s_xor_b32 s1, s0, s51 // (op 667) 

s_sub_i32 s0, s1, s51 // (op 667) 


s_mul_hi_i32 s1, s0, s11 // (op 667) 
s_add_i32 s51, s1, s0 // (op 667) 

s_ashr_i32 s1, s51, 31 // (op 667) 
s_cmp_eq_i32 s11, 0 // (op 667) 
s_mov_b32 s6, scc // (op 667) 
s_cmp_eq_i32 s6, 0 // (op 667) 
s_cselect_b32 s7, 0, -1 // (op 667) 

s_and_b32 s6, s12, 31 // (op 667) 
s_lshl_b32 s18, 1, s6 // (op 667) 
s_add_i32 s19, s18, s7 // (op 667) 


s_and_b32 s7, s1, s19 // (op 667) 


s_add_i32 s1, s51, s7 // (op 667) 


s_ashr_i32 s7, s1, s6 // (op 667) 


s_lshl_b32 s1, s12, 24 // (op 667) 
s_ashr_i32 s6, s1, 31 // (op 667) 

s_xor_b32 s1, s7, s6 // (op 667) 

s_sub_i32 s7, s1, s6 // (op 667) 


s_add_u32 s18, s4, 16 // (op 667) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 667) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 667) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 667) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 667) 

s_mul_i32 s1, s7, s18 // (op 667) 

s_sub_i32 s51, s0, s1 // (op 667) 


s_mul_hi_i32 s0, s16, s8 // (op 667) 
s_add_i32 s1, s0, s16 // (op 667) 

s_ashr_i32 s0, s1, 31 // (op 667) 
s_cmp_eq_i32 s8, 0 // (op 667) 
s_mov_b32 s6, scc // (op 667) 
s_cmp_eq_i32 s6, 0 // (op 667) 
s_cselect_b32 s7, 0, -1 // (op 667) 

s_and_b32 s6, s15, 31 // (op 667) 
s_lshl_b32 s64, 1, s6 // (op 667) 
s_add_i32 s65, s64, s7 // (op 667) 


s_and_b32 s7, s0, s65 // (op 667) 


s_add_i32 s0, s1, s7 // (op 667) 


s_ashr_i32 s1, s0, s6 // (op 667) 


s_lshl_b32 s0, s15, 24 // (op 667) 
s_ashr_i32 s6, s0, 31 // (op 667) 

s_xor_b32 s0, s1, s6 // (op 667) 

s_sub_i32 s1, s0, s6 // (op 667) 


s_mul_i32 s0, s14, s18 // (op 667) 

s_mul_i32 s6, s50, s0 // (op 667) 

s_mul_i32 s0, s1, s6 // (op 667) 

s_sub_u32 s6, s16, s0 // (op 667) 

s_mul_hi_i32 s0, s6, s13 // (op 667) 
s_add_i32 s7, s0, s6 // (op 667) 


s_ashr_i32 s0, s7, 31 // (op 667) 
s_cmp_eq_i32 s13, 0 // (op 667) 
s_mov_b32 s6, scc // (op 667) 
s_cmp_eq_i32 s6, 0 // (op 667) 
s_cselect_b32 s18, 0, -1 // (op 667) 

s_and_b32 s6, s49, 31 // (op 667) 
s_lshl_b32 s19, 1, s6 // (op 667) 
s_add_i32 s64, s19, s18 // (op 667) 


s_and_b32 s18, s0, s64 // (op 667) 


s_add_i32 s0, s7, s18 // (op 667) 


s_ashr_i32 s7, s0, s6 // (op 667) 


s_lshl_b32 s0, s49, 24 // (op 667) 
s_ashr_i32 s6, s0, 31 // (op 667) 

s_xor_b32 s0, s7, s6 // (op 667) 

s_sub_i32 s7, s0, s6 // (op 667) 


s_cmp_eq_i32 s1, 0 // (op 667) 
s_mov_b32 s0, scc // (op 667) 

s_cmp_eq_i32 s0, 0 // (op 667) 
s_cselect_b32 s1, s7, s51 // (op 667) 



v_lshl_add_u32 v19, s1, 4, v18 // (op 667) 


v_mul_lo_u32 v18, v19, s36 // (op 667) 

v_add_u32 v19, v20, v18 // (op 667) 


s_mul_i32 s1, s36, s5 // (op 669) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 669) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 669) least significant
s_mul_hi_u32 s6, s36, s4 // (op 669) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 669) most significant: sum
s_add_u32 s1, s1, s6 // (op 669) most significant: sum


s_mov_b32 s64, s38 // (op 669) reg expression
s_mov_b32 s65, s39 // (op 669) reg expression
s_mov_b32 s66, s0 // (op 669) reg expression
s_mov_b32 s67, 131072 // (op 669) call()

s_mul_hi_i32 s51, s16, s9 // (op 686) 
v_lshrrev_b32 v18, 2, v1 // (op 686) 
s_add_i32 s0, s51, s16 // (op 686) 

v_and_b32 v20, -4, v1 // (op 686) 
s_ashr_i32 s1, s0, 31 // (op 686) 
v_sub_u32 v21, v1, v20 // (op 686) 

s_cmp_eq_i32 s9, 0 // (op 686) 
s_mov_b32 s51, scc // (op 686) 
s_cmp_eq_i32 s51, 0 // (op 686) 
s_cselect_b32 s6, 0, -1 // (op 686) 

s_and_b32 s7, s10, 31 // (op 686) 
s_lshl_b32 s51, 1, s7 // (op 686) 
s_add_i32 s18, s51, s6 // (op 686) 


s_and_b32 s6, s1, s18 // (op 686) 


s_add_i32 s1, s0, s6 // (op 686) 


s_ashr_i32 s0, s1, s7 // (op 686) 


s_lshl_b32 s1, s10, 24 // (op 686) 
s_ashr_i32 s51, s1, 31 // (op 686) 

s_xor_b32 s1, s0, s51 // (op 686) 

s_sub_i32 s0, s1, s51 // (op 686) 


s_mul_hi_i32 s1, s0, s11 // (op 686) 
s_add_i32 s51, s1, s0 // (op 686) 

s_ashr_i32 s1, s51, 31 // (op 686) 
s_cmp_eq_i32 s11, 0 // (op 686) 
s_mov_b32 s6, scc // (op 686) 
s_cmp_eq_i32 s6, 0 // (op 686) 
s_cselect_b32 s7, 0, -1 // (op 686) 

s_and_b32 s6, s12, 31 // (op 686) 
s_lshl_b32 s18, 1, s6 // (op 686) 
s_add_i32 s19, s18, s7 // (op 686) 


s_and_b32 s7, s1, s19 // (op 686) 


s_add_i32 s1, s51, s7 // (op 686) 


s_ashr_i32 s7, s1, s6 // (op 686) 


s_lshl_b32 s1, s12, 24 // (op 686) 
s_ashr_i32 s6, s1, 31 // (op 686) 

s_xor_b32 s1, s7, s6 // (op 686) 

s_sub_i32 s7, s1, s6 // (op 686) 


s_add_u32 s18, s4, 16 // (op 686) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 686) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 686) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 686) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 686) 

s_mul_i32 s1, s7, s18 // (op 686) 

s_sub_i32 s51, s0, s1 // (op 686) 


s_mul_hi_i32 s0, s16, s8 // (op 686) 
s_add_i32 s1, s0, s16 // (op 686) 

s_ashr_i32 s0, s1, 31 // (op 686) 
s_cmp_eq_i32 s8, 0 // (op 686) 
s_mov_b32 s6, scc // (op 686) 
s_cmp_eq_i32 s6, 0 // (op 686) 
s_cselect_b32 s7, 0, -1 // (op 686) 

s_and_b32 s6, s15, 31 // (op 686) 
s_lshl_b32 s68, 1, s6 // (op 686) 
s_add_i32 s69, s68, s7 // (op 686) 


s_and_b32 s7, s0, s69 // (op 686) 


s_add_i32 s0, s1, s7 // (op 686) 


s_ashr_i32 s1, s0, s6 // (op 686) 


s_lshl_b32 s0, s15, 24 // (op 686) 
s_ashr_i32 s6, s0, 31 // (op 686) 

s_xor_b32 s0, s1, s6 // (op 686) 

s_sub_i32 s1, s0, s6 // (op 686) 


s_mul_i32 s0, s14, s18 // (op 686) 

s_mul_i32 s6, s50, s0 // (op 686) 

s_mul_i32 s0, s1, s6 // (op 686) 

s_sub_u32 s6, s16, s0 // (op 686) 

s_mul_hi_i32 s0, s6, s13 // (op 686) 
s_add_i32 s7, s0, s6 // (op 686) 


s_ashr_i32 s0, s7, 31 // (op 686) 
s_cmp_eq_i32 s13, 0 // (op 686) 
s_mov_b32 s6, scc // (op 686) 
s_cmp_eq_i32 s6, 0 // (op 686) 
s_cselect_b32 s18, 0, -1 // (op 686) 

s_and_b32 s6, s49, 31 // (op 686) 
s_lshl_b32 s19, 1, s6 // (op 686) 
s_add_i32 s68, s19, s18 // (op 686) 


s_and_b32 s18, s0, s68 // (op 686) 


s_add_i32 s0, s7, s18 // (op 686) 


s_ashr_i32 s7, s0, s6 // (op 686) 


s_lshl_b32 s0, s49, 24 // (op 686) 
s_ashr_i32 s6, s0, 31 // (op 686) 

s_xor_b32 s0, s7, s6 // (op 686) 

s_sub_i32 s7, s0, s6 // (op 686) 


s_cmp_eq_i32 s1, 0 // (op 686) 
s_mov_b32 s0, scc // (op 686) 

s_cmp_eq_i32 s0, 0 // (op 686) 
s_cselect_b32 s1, s7, s51 // (op 686) 



v_lshl_add_u32 v20, s1, 4, v18 // (op 686) 


v_mul_lo_u32 v18, v20, s36 // (op 686) 

v_add_u32 v20, v21, v18 // (op 686) 


s_mul_i32 s1, s36, s5 // (op 688) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 688) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 688) least significant
s_mul_hi_u32 s6, s36, s4 // (op 688) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 688) most significant: sum
s_add_u32 s1, s1, s6 // (op 688) most significant: sum


s_mov_b32 s64, s38 // (op 688) reg expression
s_mov_b32 s65, s39 // (op 688) reg expression
s_mov_b32 s66, s0 // (op 688) reg expression
s_mov_b32 s67, 131072 // (op 688) call()




buffer_load_dwordx4 v[22:25], v16, s[56:59], 0 offen // (op 366) Load value
buffer_load_dwordx2 v[26:27], v16, s[56:59], 0 offen offset:16 // (op 366) Load value



buffer_load_ubyte v18, v19, s[64:67], 0 offen offset: 0 // (op 369) Load value

s_lshl_b64 s[0:1], s[2:3], 2 // (op 372) 
v_lshrrev_b64 v[28:29], 3, s[0:1] // (op 372) 


v_mov_b32 v21, v14 // (op 372) reg expression
buffer_load_dwordx2 v[30:31], v21, s[52:55], 0 offen // (op 372) Load value
v_add_u32 v21, v21, v28 // (op 372) 
buffer_load_dwordx2 v[32:33], v21, s[52:55], 0 offen // (op 372) Load value




v_mov_b32 v21, v12 // (op 375) reg expression

buffer_load_ubyte v28, v21, s[60:63], 0 offen offset: 0 // (op 375) Load value




buffer_load_dwordx4 v[34:37], v16, s[56:59], 0 offen offset:96 // (op 378) Load value
buffer_load_dwordx2 v[38:39], v16, s[56:59], 0 offen offset:112 // (op 378) Load value




buffer_load_ubyte v16, v19, s[64:67], 0 offen offset:4 // (op 381) Load value


s_lshl_b64 s[0:1], s[2:3], 2 // (op 384) 
v_lshrrev_b64 v[40:41], 3, s[0:1] // (op 384) 


s_lshl_b64 s[0:1], s[2:3], 7 // (op 384) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 384) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 384) 

v_add_u32 v19, v14, s0 // (op 384) 

buffer_load_dwordx2 v[42:43], v19, s[52:55], 0 offen // (op 384) Load value
v_add_u32 v19, v19, v40 // (op 384) 
buffer_load_dwordx2 v[44:45], v19, s[52:55], 0 offen // (op 384) Load value





s_lshl_b32 s51, s30, 2 // (op 387) 
v_add_u32 v14, v12, s51 // (op 387) 


buffer_load_ubyte v19, v14, s[60:63], 0 offen offset: 0 // (op 387) Load value


v_and_b32 v12, -4, v1 // (op 806) 
v_sub_u32 v14, v1, v12 // (op 806) 

v_lshlrev_b32 v12, 5, v14 // (op 806) 

v_lshrrev_b32 v14, 2, v1 // (op 806) 
v_lshl_add_u32 v21, v14, 7, v12 // (op 806) 


v_mul_lo_u32 v12, v21, 6 // (op 806) 

v_lshrrev_b32 v14, 3, v12 // (op 806) 




s_waitcnt vmcnt(11)
ds_write_b128 v14, v[22:25] // (op 389) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v14, v[26:27] offset:16 // (op 389) Store local data 

v_and_b32 v12, -4, v1 // (op 912) 
v_sub_u32 v14, v1, v12 // (op 912) 

v_and_b32 v12, 4294967292, v1 // (op 912) 
v_add_u32 v21, v12, v14 // (op 912) 





s_waitcnt vmcnt(9)
ds_write_b8 v21, v18 offset:1536 // (op 390) Store local data 

v_lshlrev_b32 v12, 7, v1 // (op 753) 
v_lshrrev_b32 v14, 3, v12 // (op 753) 




s_waitcnt vmcnt(8)
ds_write_b64 v14, v[30:31] offset:1600 // (op 391) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v14, v[32:33] offset:1608 // (op 391) Store local data 

v_and_b32 v12, -16, v1 // (op 859) 
v_sub_u32 v14, v1, v12 // (op 859) 

v_and_b32 v12, 4294967280, v1 // (op 859) 
v_add_u32 v21, v12, v14 // (op 859) 





s_waitcnt vmcnt(6)
ds_write_b8 v21, v28 offset:2624 // (op 392) Store local data 

v_lshlrev_b32 v12, 7, v1 // (op 724) 
v_lshrrev_b32 v14, 3, v12 // (op 724) 

v_and_b32 v12, -4, v1 // (op 777) 
v_sub_u32 v21, v1, v12 // (op 777) 

v_lshlrev_b32 v12, 5, v21 // (op 777) 

v_lshrrev_b32 v21, 2, v1 // (op 777) 
v_lshl_add_u32 v29, v21, 7, v12 // (op 777) 


v_mul_lo_u32 v12, v29, 6 // (op 777) 

v_lshrrev_b32 v21, 3, v12 // (op 777) 

v_and_b32 v12, -16, v1 // (op 830) 
v_sub_u32 v29, v1, v12 // (op 830) 

v_and_b32 v12, 4294967280, v1 // (op 830) 
v_add_u32 v40, v12, v29 // (op 830) 


v_and_b32 v12, -4, v1 // (op 883) 
v_sub_u32 v29, v1, v12 // (op 883) 

v_and_b32 v12, 4294967292, v1 // (op 883) 
v_add_u32 v41, v12, v29 // (op 883) 


v_lshlrev_b32 v12, 7, v1 // (op 970) 
v_lshrrev_b32 v29, 3, v12 // (op 970) 

v_and_b32 v12, -4, v1 // (op 994) 
v_sub_u32 v46, v1, v12 // (op 994) 

v_lshlrev_b32 v12, 5, v46 // (op 994) 

v_lshrrev_b32 v46, 2, v1 // (op 994) 
v_lshl_add_u32 v47, v46, 7, v12 // (op 994) 


v_mul_lo_u32 v12, v47, 6 // (op 994) 

v_lshrrev_b32 v46, 3, v12 // (op 994) 

v_and_b32 v12, -16, v1 // (op 1018) 
v_sub_u32 v47, v1, v12 // (op 1018) 

v_and_b32 v12, 4294967280, v1 // (op 1018) 
v_add_u32 v48, v12, v47 // (op 1018) 


v_and_b32 v12, -4, v1 // (op 1042) 
v_sub_u32 v47, v1, v12 // (op 1042) 

v_and_b32 v12, 4294967292, v1 // (op 1042) 
v_add_u32 v49, v12, v47 // (op 1042) 


s_mov_b32 s51, 0 // (op 59) call()
s_ashr_i64 s[0:1], s[24:25], 7 // (op 58) 
s_mov_b32 s6, s0 // (op 58) convert

s_and_b32 s7, s6, 4294967294 // (op 58) 

s_cmp_lt_u32 s51, s7 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_58 if false)
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_58:


s_add_u32 s51, s51, 2 // (op 60) 
v_mov_b32 v12, v17 // (op 10) 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 1091) 
buffer_load_dwordx4 v[22:25], v12, s[56:59], 0 offen offset:192 // (op 10) Load value



ds_read_b64_tr_b4 v[50:51], v3 offset:1600 // (op 70) Transpose load local data 
ds_read_b64_tr_b4 v[52:53], v3 offset:1728 // (op 70) Transpose load local data 
buffer_load_dwordx2 v[26:27], v12, s[56:59], 0 offen offset:208 // (op 10) Load value


s_barrier  // (op 1104) 
v_mov_b32 v47, v20 // (op 25) 




buffer_load_ubyte v18, v47, s[64:67], 0 offen offset:8 // (op 25) Load value
ds_read_b128 v[54:57], v4 // (op 75) Load local data 

s_lshl_b64 s[0:1], s[2:3], 2 // (op 4) 
v_lshrrev_b64 v[60:61], 3, s[0:1] // (op 4) 


v_mov_b32 v62, v15 // (op 4) 
s_lshl_b64 s[0:1], s[2:3], 7 // (op 4) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 4) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 4) 

s_lshl_b32 s6, s0, 1 // (op 4) 

v_add_u32 v63, v62, s6 // (op 4) 

ds_read_b64 v[58:59], v4 offset:16 // (op 75) Load local data 
buffer_load_dwordx2 v[30:31], v63, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v63, v63, v60 // (op 4) 
buffer_load_dwordx2 v[32:33], v63, s[52:55], 0 offen // (op 4) Load value


s_waitcnt lgkmcnt(4)
s_barrier  // (op 1118) 




ds_read_u8 v60, v5 offset:2624 // (op 80) Load local data 
v_mov_b32 v61, v13 // (op 17) 
s_barrier  // (op 1132) 
s_lshl_b32 s0, s30, 3 // (op 17) 



ds_read_u8 v63, v6 offset:1536 // (op 85) Load local data 
v_add_u32 v64, v61, s0 // (op 17) 


buffer_load_ubyte v28, v64, s[60:63], 0 offen offset: 0 // (op 17) Load value

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:53], v[54:59], a[0:3], v60, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 88) 




s_barrier  // (op 1155) 



s_waitcnt vmcnt(11)
ds_write_b128 v46, v[34:37] offset:2688 // (op 253) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v46, v[38:39] offset:2704 // (op 253) Store local data 
s_barrier  // (op 1167) 



s_waitcnt vmcnt(9)
ds_write_b8 v49, v16 offset:4224 // (op 255) Store local data 
s_barrier  // (op 1149) 



s_waitcnt vmcnt(8)
ds_write_b64 v29, v[42:43] offset:4288 // (op 243) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v29, v[44:45] offset:4296 // (op 243) Store local data 
s_barrier  // (op 1161) 



s_waitcnt vmcnt(6)
ds_write_b8 v48, v19 offset:5312 // (op 254) Store local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 1146) 
buffer_load_dwordx4 v[34:37], v12, s[56:59], 0 offen offset:288 // (op 256) Load value



ds_read_b64_tr_b4 v[50:51], v7 offset:4288 // (op 245) Transpose load local data 
ds_read_b64_tr_b4 v[52:53], v7 offset:4416 // (op 245) Transpose load local data 
buffer_load_dwordx2 v[38:39], v12, s[56:59], 0 offen offset:304 // (op 256) Load value




s_barrier  // (op 1152) 
s_barrier  // (op 1222) 



v_add_u32 v17, 192, v17 // (op 550) 
ds_read_b128 v[54:57], v8 offset:2688 // (op 247) Load local data 
buffer_load_ubyte v16, v47, s[64:67], 0 offen offset:12 // (op 258) Load value
ds_read_b64 v[58:59], v8 offset:2704 // (op 247) Load local data 


s_barrier  // (op 1228) 
v_add_u32 v20, 8, v20 // (op 678) 
s_lshl_b64 s[0:1], s[2:3], 2 // (op 242) 
v_lshrrev_b64 v[64:65], 3, s[0:1] // (op 242) 


s_lshl_b64 s[0:1], s[2:3], 7 // (op 242) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 242) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 242) 

s_mul_i32 s6, 3, s0 // (op 242) 

v_add_u32 v12, v62, s6 // (op 242) 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1158) 
buffer_load_dwordx2 v[42:43], v12, s[52:55], 0 offen // (op 242) Load value


v_add_u32 v12, v12, v64 // (op 242) 
ds_read_u8 v47, v9 offset:5312 // (op 249) Load local data 
buffer_load_dwordx2 v[44:45], v12, s[52:55], 0 offen // (op 242) Load value





s_barrier  // (op 1164) 
s_barrier  // (op 1234) 



ds_read_u8 v12, v10 offset:4224 // (op 251) Load local data 
s_lshl_b32 s0, s30, 2 // (op 257) 
s_mul_i32 s1, 3, s0 // (op 257) 

s_barrier  // (op 1240) 
v_add_u32 v60, v61, s1 // (op 257) 


s_lshl_b64 s[0:1], s[2:3], 7 // (op 486) 
buffer_load_ubyte v19, v60, s[60:63], 0 offen offset: 0 // (op 257) Load value

s_lshl_b64 s[6:7], s[0:1], 2 // (op 486) 


s_lshr_b64 s[0:1], s[6:7], 3 // (op 486) 

s_barrier  // (op 1107) 
s_lshl_b32 s6, s0, 1 // (op 486) 




v_add_u32 v15, v15, s6 // (op 486) 

s_lshl_b32 s0, s30, 3 // (op 614) 
v_add_u32 v13, v13, s0 // (op 614) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:53], v[54:59], a[0:3], v47, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 252) 
s_waitcnt vmcnt(11)
ds_write_b128 v21, v[22:25] // (op 13) Store local data 




s_waitcnt vmcnt(10)
ds_write_b64 v21, v[26:27] offset:16 // (op 13) Store local data 
s_barrier  // (op 1135) 



s_waitcnt vmcnt(9)
ds_write_b8 v41, v18 offset:1536 // (op 30) Store local data 
s_barrier  // (op 1094) 



s_waitcnt vmcnt(8)
ds_write_b64 v14, v[30:31] offset:1600 // (op 6) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v14, v[32:33] offset:1608 // (op 6) Store local data 
s_barrier  // (op 1121) 



s_waitcnt vmcnt(6)
ds_write_b8 v40, v28 offset:2624 // (op 21) Store local data 
s_ashr_i64 s[0:1], s[24:25], 7 // (op 58) 
s_mov_b32 s6, s0 // (op 58) convert

s_and_b32 s7, s6, 4294967294 // (op 58) 

s_cmp_lt_u32 s51, s7 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_58 if true)
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_58:

















s_mul_hi_i32 s51, s16, s9 // (op 449) 
v_lshlrev_b32 v12, 1, v1 // (op 449) 
s_add_i32 s0, s51, s16 // (op 449) 

s_ashr_i32 s1, s0, 31 // (op 449) 
s_cmp_eq_i32 s9, 0 // (op 449) 
s_mov_b32 s51, scc // (op 449) 
s_cmp_eq_i32 s51, 0 // (op 449) 
s_cselect_b32 s6, 0, -1 // (op 449) 

s_and_b32 s7, s10, 31 // (op 449) 
s_lshl_b32 s51, 1, s7 // (op 449) 
s_add_i32 s18, s51, s6 // (op 449) 


s_and_b32 s6, s1, s18 // (op 449) 


s_add_i32 s1, s0, s6 // (op 449) 


s_ashr_i32 s0, s1, s7 // (op 449) 


s_lshl_b32 s1, s10, 24 // (op 449) 
s_ashr_i32 s51, s1, 31 // (op 449) 

s_xor_b32 s1, s0, s51 // (op 449) 

s_sub_i32 s0, s1, s51 // (op 449) 


s_mul_hi_i32 s1, s0, s11 // (op 449) 
s_add_i32 s51, s1, s0 // (op 449) 

s_ashr_i32 s1, s51, 31 // (op 449) 
s_cmp_eq_i32 s11, 0 // (op 449) 
s_mov_b32 s6, scc // (op 449) 
s_cmp_eq_i32 s6, 0 // (op 449) 
s_cselect_b32 s7, 0, -1 // (op 449) 

s_and_b32 s6, s12, 31 // (op 449) 
s_lshl_b32 s18, 1, s6 // (op 449) 
s_add_i32 s19, s18, s7 // (op 449) 


s_and_b32 s7, s1, s19 // (op 449) 


s_add_i32 s1, s51, s7 // (op 449) 


s_ashr_i32 s7, s1, s6 // (op 449) 


s_lshl_b32 s1, s12, 24 // (op 449) 
s_ashr_i32 s6, s1, 31 // (op 449) 

s_xor_b32 s1, s7, s6 // (op 449) 

s_sub_i32 s7, s1, s6 // (op 449) 


s_mul_i32 s1, s0, s14 // (op 449) 

s_sub_u32 s0, s16, s1 // (op 449) 

s_mul_i32 s1, s7, s14 // (op 449) 
s_add_u32 s1, s1, s0 // (op 449) 


s_mul_hi_i32 s0, s16, s8 // (op 449) 
s_add_i32 s51, s0, s16 // (op 449) 

s_ashr_i32 s0, s51, 31 // (op 449) 
s_cmp_eq_i32 s8, 0 // (op 449) 
s_mov_b32 s6, scc // (op 449) 
s_cmp_eq_i32 s6, 0 // (op 449) 
s_cselect_b32 s7, 0, -1 // (op 449) 

s_and_b32 s6, s15, 31 // (op 449) 
s_lshl_b32 s18, 1, s6 // (op 449) 
s_add_i32 s19, s18, s7 // (op 449) 


s_and_b32 s7, s0, s19 // (op 449) 


s_add_i32 s0, s51, s7 // (op 449) 


s_ashr_i32 s7, s0, s6 // (op 449) 


s_lshl_b32 s0, s15, 24 // (op 449) 
s_ashr_i32 s6, s0, 31 // (op 449) 

s_xor_b32 s0, s7, s6 // (op 449) 

s_sub_i32 s7, s0, s6 // (op 449) 


s_add_u32 s18, s4, 16 // (op 449) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 449) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 449) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 449) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 449) 

s_mul_i32 s0, s14, s18 // (op 449) 

s_mul_i32 s6, s50, s0 // (op 449) 

s_mul_i32 s0, s7, s6 // (op 449) 

s_sub_u32 s6, s16, s0 // (op 449) 

s_mov_b32 s0, s6 // (op 449) 
s_mul_hi_i32 s51, s0, s13 // (op 449) 
s_add_i32 s18, s51, s0 // (op 449) 


s_ashr_i32 s0, s18, 31 // (op 449) 
s_cmp_eq_i32 s13, 0 // (op 449) 
s_mov_b32 s19, scc // (op 449) 
s_cmp_eq_i32 s19, 0 // (op 449) 
s_cselect_b32 s51, 0, -1 // (op 449) 

s_and_b32 s19, s49, 31 // (op 449) 
s_lshl_b32 s68, 1, s19 // (op 449) 
s_add_i32 s69, s68, s51 // (op 449) 


s_and_b32 s51, s0, s69 // (op 449) 


s_add_i32 s0, s18, s51 // (op 449) 


s_ashr_i32 s18, s0, s19 // (op 449) 


s_lshl_b32 s0, s49, 24 // (op 449) 
s_ashr_i32 s19, s0, 31 // (op 449) 

s_xor_b32 s0, s18, s19 // (op 449) 

s_sub_i32 s18, s0, s19 // (op 449) 


s_mul_i32 s0, s48, s14 // (op 449) 
s_add_u32 s68, s20, 16 // (op 449) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 449) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 449) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 449) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 449) 

s_sub_i32 s19, s68, s0 // (op 449) 

s_mul_i32 s51, s18, s19 // (op 449) 


s_sub_u32 s18, s6, s51 // (op 449) 


s_add_u32 s6, s18, s0 // (op 449) 


s_cmp_eq_i32 s7, 0 // (op 449) 
s_mov_b32 s0, scc // (op 449) 

s_cmp_eq_i32 s0, 0 // (op 449) 
s_cselect_b32 s7, s6, s1 // (op 449) 



s_lshl_b32 s6, s7, 4 // (op 449) 

s_mov_b32 s1, 0 // (op 449) convert
s_mov_b32 s0, s6 // (op 449) convert

s_ashr_i64 s[6:7], s[24:25], 7 // (op 449) 
s_lshr_b32 s51, s6, 1 // (op 449) 

v_lshl_add_u32 v13, s51, 8, v12 // (op 449) 


v_mul_lo_u32 v15, v13, s3 // (op 449) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 449) most significant: high of low * low
v_mul_lo_u32 v14, v13, s2 // (op 449) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 449) most significant: sum


v_mov_b32 v12, s0 // (op 449) 
v_mov_b32 v13, s1 // (op 449) 
v_add_co_u32 v40, s[6:7], v12, v14 // (op 449) least significant half
v_addc_co_u32 v41, s[6:7], v13, v15, s[6:7] // (op 449) most significant half



v_lshlrev_b64 v[40:41], 2, v[40:41] // (op 449) 


v_lshrrev_b64 v[12:13], 3, v[40:41] // (op 449) 

v_mov_b32 v14, v12 // (op 449) convert

s_mul_i32 s1, s2, s25 // (op 451) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 451) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 451) least significant
s_mul_hi_u32 s6, s2, s24 // (op 451) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 451) most significant: sum
s_add_u32 s1, s1, s6 // (op 451) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 451) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 451) 

s_mov_b32 s52, s22 // (op 451) reg expression
s_mov_b32 s53, s23 // (op 451) reg expression
s_mov_b32 s54, s0 // (op 451) reg expression
s_mov_b32 s55, 131072 // (op 451) call()

s_mul_hi_i32 s51, s16, s9 // (op 516) 
v_lshrrev_b32 v15, 2, v1 // (op 516) 
s_add_i32 s0, s51, s16 // (op 516) 

v_and_b32 v12, -4, v1 // (op 516) 
s_ashr_i32 s1, s0, 31 // (op 516) 
v_sub_u32 v13, v1, v12 // (op 516) 

s_cmp_eq_i32 s9, 0 // (op 516) 
s_mov_b32 s51, scc // (op 516) 
v_lshlrev_b32 v12, 5, v13 // (op 516) 

s_cmp_eq_i32 s51, 0 // (op 516) 
s_cselect_b32 s6, 0, -1 // (op 516) 

s_and_b32 s7, s10, 31 // (op 516) 
s_lshl_b32 s51, 1, s7 // (op 516) 
s_add_i32 s18, s51, s6 // (op 516) 


s_and_b32 s6, s1, s18 // (op 516) 


s_add_i32 s1, s0, s6 // (op 516) 


s_ashr_i32 s0, s1, s7 // (op 516) 


s_lshl_b32 s1, s10, 24 // (op 516) 
s_ashr_i32 s51, s1, 31 // (op 516) 

s_xor_b32 s1, s0, s51 // (op 516) 

s_sub_i32 s0, s1, s51 // (op 516) 


s_mul_hi_i32 s1, s0, s11 // (op 516) 
s_add_i32 s51, s1, s0 // (op 516) 

s_ashr_i32 s1, s51, 31 // (op 516) 
s_cmp_eq_i32 s11, 0 // (op 516) 
s_mov_b32 s6, scc // (op 516) 
s_cmp_eq_i32 s6, 0 // (op 516) 
s_cselect_b32 s7, 0, -1 // (op 516) 

s_and_b32 s6, s12, 31 // (op 516) 
s_lshl_b32 s18, 1, s6 // (op 516) 
s_add_i32 s19, s18, s7 // (op 516) 


s_and_b32 s7, s1, s19 // (op 516) 


s_add_i32 s1, s51, s7 // (op 516) 


s_ashr_i32 s7, s1, s6 // (op 516) 


s_lshl_b32 s1, s12, 24 // (op 516) 
s_ashr_i32 s6, s1, 31 // (op 516) 

s_xor_b32 s1, s7, s6 // (op 516) 

s_sub_i32 s7, s1, s6 // (op 516) 


s_add_u32 s18, s4, 16 // (op 516) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 516) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 516) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 516) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 516) 

s_mul_i32 s1, s7, s18 // (op 516) 

s_sub_i32 s51, s0, s1 // (op 516) 


s_mul_hi_i32 s0, s16, s8 // (op 516) 
s_add_i32 s1, s0, s16 // (op 516) 

s_ashr_i32 s0, s1, 31 // (op 516) 
s_cmp_eq_i32 s8, 0 // (op 516) 
s_mov_b32 s6, scc // (op 516) 
s_cmp_eq_i32 s6, 0 // (op 516) 
s_cselect_b32 s7, 0, -1 // (op 516) 

s_and_b32 s6, s15, 31 // (op 516) 
s_lshl_b32 s68, 1, s6 // (op 516) 
s_add_i32 s69, s68, s7 // (op 516) 


s_and_b32 s7, s0, s69 // (op 516) 


s_add_i32 s0, s1, s7 // (op 516) 


s_ashr_i32 s1, s0, s6 // (op 516) 


s_lshl_b32 s0, s15, 24 // (op 516) 
s_ashr_i32 s6, s0, 31 // (op 516) 

s_xor_b32 s0, s1, s6 // (op 516) 

s_sub_i32 s1, s0, s6 // (op 516) 


s_mul_i32 s0, s14, s18 // (op 516) 

s_mul_i32 s6, s50, s0 // (op 516) 

s_mul_i32 s0, s1, s6 // (op 516) 

s_sub_u32 s6, s16, s0 // (op 516) 

s_mul_hi_i32 s0, s6, s13 // (op 516) 
s_add_i32 s7, s0, s6 // (op 516) 


s_ashr_i32 s0, s7, 31 // (op 516) 
s_cmp_eq_i32 s13, 0 // (op 516) 
s_mov_b32 s6, scc // (op 516) 
s_cmp_eq_i32 s6, 0 // (op 516) 
s_cselect_b32 s18, 0, -1 // (op 516) 

s_and_b32 s6, s49, 31 // (op 516) 
s_lshl_b32 s19, 1, s6 // (op 516) 
s_add_i32 s68, s19, s18 // (op 516) 


s_and_b32 s18, s0, s68 // (op 516) 


s_add_i32 s0, s7, s18 // (op 516) 


s_ashr_i32 s7, s0, s6 // (op 516) 


s_lshl_b32 s0, s49, 24 // (op 516) 
s_ashr_i32 s6, s0, 31 // (op 516) 

s_xor_b32 s0, s7, s6 // (op 516) 

s_sub_i32 s7, s0, s6 // (op 516) 


s_cmp_eq_i32 s1, 0 // (op 516) 
s_mov_b32 s0, scc // (op 516) 

s_cmp_eq_i32 s0, 0 // (op 516) 
s_cselect_b32 s1, s7, s51 // (op 516) 



v_lshl_add_u32 v13, s1, 4, v15 // (op 516) 


v_mul_lo_u32 v41, v13, s27 // (op 516) most significant: low of low * high
v_mul_hi_u32 v15, v13, s26 // (op 516) most significant: high of low * low
v_mul_lo_u32 v40, v13, s26 // (op 516) least significant: low of low * low
v_add_u32 v41, v41, v15 // (op 516) most significant: sum


s_ashr_i64 s[0:1], s[24:25], 7 // (op 516) 
s_lshr_b32 s51, s0, 1 // (op 516) 

v_lshl_add_u32 v13, s51, 8, v12 // (op 516) 


v_mov_b32 v21, 0 // (op 516) convert
v_mov_b32 v20, v13 // (op 516) convert

v_add_co_u32 v12, s[0:1], v20, v40 // (op 516) least significant half
v_addc_co_u32 v13, s[0:1], v21, v41, s[0:1] // (op 516) most significant half



v_mul_lo_u32 v41, v13, 6 // (op 516) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 516) most significant: high of low * low
v_mul_lo_u32 v40, v12, 6 // (op 516) least significant: low of low * low
v_add_u32 v41, v41, v15 // (op 516) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[40:41] // (op 516) 

v_mov_b32 v15, v12 // (op 516) convert

s_mul_i32 s1, s5, 6 // (op 518) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 518) least significant
s_mul_hi_u32 s51, s4, 6 // (op 518) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 518) most significant: sum

s_mul_i32 s7, s26, s1 // (op 518) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 518) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 518) least significant
s_mul_hi_u32 s18, s26, s0 // (op 518) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 518) most significant: sum
s_add_u32 s7, s7, s18 // (op 518) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 518) 

s_mov_b32 s56, s28 // (op 518) reg expression
s_mov_b32 s57, s29 // (op 518) reg expression
s_mov_b32 s58, s0 // (op 518) reg expression
s_mov_b32 s59, 131072 // (op 518) call()

s_mul_hi_i32 s51, s16, s9 // (op 580) 
v_and_b32 v12, -16, v1 // (op 580) 
s_add_i32 s0, s51, s16 // (op 580) 

v_sub_u32 v13, v1, v12 // (op 580) 

s_ashr_i32 s1, s0, 31 // (op 580) 
v_lshrrev_b32 v12, 4, v1 // (op 580) 
s_cmp_eq_i32 s9, 0 // (op 580) 
s_mov_b32 s51, scc // (op 580) 
s_cmp_eq_i32 s51, 0 // (op 580) 
s_cselect_b32 s6, 0, -1 // (op 580) 

s_and_b32 s7, s10, 31 // (op 580) 
s_lshl_b32 s51, 1, s7 // (op 580) 
s_add_i32 s18, s51, s6 // (op 580) 


s_and_b32 s6, s1, s18 // (op 580) 


s_add_i32 s1, s0, s6 // (op 580) 


s_ashr_i32 s0, s1, s7 // (op 580) 


s_lshl_b32 s1, s10, 24 // (op 580) 
s_ashr_i32 s51, s1, 31 // (op 580) 

s_xor_b32 s1, s0, s51 // (op 580) 

s_sub_i32 s0, s1, s51 // (op 580) 


s_mul_hi_i32 s1, s0, s11 // (op 580) 
s_add_i32 s51, s1, s0 // (op 580) 

s_ashr_i32 s1, s51, 31 // (op 580) 
s_cmp_eq_i32 s11, 0 // (op 580) 
s_mov_b32 s6, scc // (op 580) 
s_cmp_eq_i32 s6, 0 // (op 580) 
s_cselect_b32 s7, 0, -1 // (op 580) 

s_and_b32 s6, s12, 31 // (op 580) 
s_lshl_b32 s18, 1, s6 // (op 580) 
s_add_i32 s19, s18, s7 // (op 580) 


s_and_b32 s7, s1, s19 // (op 580) 


s_add_i32 s1, s51, s7 // (op 580) 


s_ashr_i32 s7, s1, s6 // (op 580) 


s_lshl_b32 s1, s12, 24 // (op 580) 
s_ashr_i32 s6, s1, 31 // (op 580) 

s_xor_b32 s1, s7, s6 // (op 580) 

s_sub_i32 s7, s1, s6 // (op 580) 


s_mul_i32 s1, s0, s14 // (op 580) 

s_sub_u32 s0, s16, s1 // (op 580) 

s_mul_i32 s1, s7, s14 // (op 580) 
s_add_u32 s1, s1, s0 // (op 580) 


s_mul_hi_i32 s0, s16, s8 // (op 580) 
s_add_i32 s51, s0, s16 // (op 580) 

s_ashr_i32 s0, s51, 31 // (op 580) 
s_cmp_eq_i32 s8, 0 // (op 580) 
s_mov_b32 s6, scc // (op 580) 
s_cmp_eq_i32 s6, 0 // (op 580) 
s_cselect_b32 s7, 0, -1 // (op 580) 

s_and_b32 s6, s15, 31 // (op 580) 
s_lshl_b32 s18, 1, s6 // (op 580) 
s_add_i32 s19, s18, s7 // (op 580) 


s_and_b32 s7, s0, s19 // (op 580) 


s_add_i32 s0, s51, s7 // (op 580) 


s_ashr_i32 s7, s0, s6 // (op 580) 


s_lshl_b32 s0, s15, 24 // (op 580) 
s_ashr_i32 s6, s0, 31 // (op 580) 

s_xor_b32 s0, s7, s6 // (op 580) 

s_sub_i32 s7, s0, s6 // (op 580) 


s_add_u32 s18, s4, 16 // (op 580) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 580) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 580) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 580) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 580) 

s_mul_i32 s0, s14, s18 // (op 580) 

s_mul_i32 s6, s50, s0 // (op 580) 

s_mul_i32 s0, s7, s6 // (op 580) 

s_sub_u32 s6, s16, s0 // (op 580) 

s_mov_b32 s0, s6 // (op 580) 
s_mul_hi_i32 s51, s0, s13 // (op 580) 
s_add_i32 s18, s51, s0 // (op 580) 


s_ashr_i32 s0, s18, 31 // (op 580) 
s_cmp_eq_i32 s13, 0 // (op 580) 
s_mov_b32 s19, scc // (op 580) 
s_cmp_eq_i32 s19, 0 // (op 580) 
s_cselect_b32 s51, 0, -1 // (op 580) 

s_and_b32 s19, s49, 31 // (op 580) 
s_lshl_b32 s68, 1, s19 // (op 580) 
s_add_i32 s69, s68, s51 // (op 580) 


s_and_b32 s51, s0, s69 // (op 580) 


s_add_i32 s0, s18, s51 // (op 580) 


s_ashr_i32 s18, s0, s19 // (op 580) 


s_lshl_b32 s0, s49, 24 // (op 580) 
s_ashr_i32 s19, s0, 31 // (op 580) 

s_xor_b32 s0, s18, s19 // (op 580) 

s_sub_i32 s18, s0, s19 // (op 580) 


s_mul_i32 s0, s48, s14 // (op 580) 
s_add_u32 s68, s20, 16 // (op 580) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 580) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 580) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 580) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 580) 

s_sub_i32 s19, s68, s0 // (op 580) 

s_mul_i32 s51, s18, s19 // (op 580) 


s_sub_u32 s18, s6, s51 // (op 580) 


s_add_u32 s6, s18, s0 // (op 580) 


s_cmp_eq_i32 s7, 0 // (op 580) 
s_mov_b32 s0, scc // (op 580) 

s_cmp_eq_i32 s0, 0 // (op 580) 
s_cselect_b32 s7, s6, s1 // (op 580) 



v_lshl_add_u32 v17, s7, 4, v13 // (op 580) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 580) 
s_lshr_b32 s51, s0, 1 // (op 580) 

v_lshl_add_u32 v13, s51, 3, v12 // (op 580) 


v_mul_lo_u32 v12, v13, s30 // (op 580) 

v_add_u32 v13, v17, v12 // (op 580) 


s_mul_i32 s1, s30, s35 // (op 582) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 582) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 582) least significant
s_mul_hi_u32 s6, s30, s34 // (op 582) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 582) most significant: sum
s_add_u32 s1, s1, s6 // (op 582) most significant: sum


s_mov_b32 s60, s32 // (op 582) reg expression
s_mov_b32 s61, s33 // (op 582) reg expression
s_mov_b32 s62, s0 // (op 582) reg expression
s_mov_b32 s63, 131072 // (op 582) call()

s_mul_hi_i32 s51, s16, s9 // (op 644) 
v_lshrrev_b32 v12, 2, v1 // (op 644) 
s_add_i32 s0, s51, s16 // (op 644) 

v_and_b32 v17, -4, v1 // (op 644) 
s_ashr_i32 s1, s0, 31 // (op 644) 
v_sub_u32 v18, v1, v17 // (op 644) 

s_cmp_eq_i32 s9, 0 // (op 644) 
s_mov_b32 s51, scc // (op 644) 
s_cmp_eq_i32 s51, 0 // (op 644) 
s_cselect_b32 s6, 0, -1 // (op 644) 

s_and_b32 s7, s10, 31 // (op 644) 
s_lshl_b32 s51, 1, s7 // (op 644) 
s_add_i32 s18, s51, s6 // (op 644) 


s_and_b32 s6, s1, s18 // (op 644) 


s_add_i32 s1, s0, s6 // (op 644) 


s_ashr_i32 s0, s1, s7 // (op 644) 


s_lshl_b32 s1, s10, 24 // (op 644) 
s_ashr_i32 s51, s1, 31 // (op 644) 

s_xor_b32 s1, s0, s51 // (op 644) 

s_sub_i32 s0, s1, s51 // (op 644) 


s_mul_hi_i32 s1, s0, s11 // (op 644) 
s_add_i32 s51, s1, s0 // (op 644) 

s_ashr_i32 s1, s51, 31 // (op 644) 
s_cmp_eq_i32 s11, 0 // (op 644) 
s_mov_b32 s6, scc // (op 644) 
s_cmp_eq_i32 s6, 0 // (op 644) 
s_cselect_b32 s7, 0, -1 // (op 644) 

s_and_b32 s6, s12, 31 // (op 644) 
s_lshl_b32 s18, 1, s6 // (op 644) 
s_add_i32 s19, s18, s7 // (op 644) 


s_and_b32 s7, s1, s19 // (op 644) 


s_add_i32 s1, s51, s7 // (op 644) 


s_ashr_i32 s7, s1, s6 // (op 644) 


s_lshl_b32 s1, s12, 24 // (op 644) 
s_ashr_i32 s6, s1, 31 // (op 644) 

s_xor_b32 s1, s7, s6 // (op 644) 

s_sub_i32 s7, s1, s6 // (op 644) 


s_add_u32 s18, s4, 16 // (op 644) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 644) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 644) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 644) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 644) 

s_mul_i32 s1, s7, s18 // (op 644) 

s_sub_i32 s51, s0, s1 // (op 644) 


s_mul_hi_i32 s0, s16, s8 // (op 644) 
s_add_i32 s1, s0, s16 // (op 644) 

s_ashr_i32 s0, s1, 31 // (op 644) 
s_cmp_eq_i32 s8, 0 // (op 644) 
s_mov_b32 s6, scc // (op 644) 
s_cmp_eq_i32 s6, 0 // (op 644) 
s_cselect_b32 s7, 0, -1 // (op 644) 

s_and_b32 s6, s15, 31 // (op 644) 
s_lshl_b32 s68, 1, s6 // (op 644) 
s_add_i32 s69, s68, s7 // (op 644) 


s_and_b32 s7, s0, s69 // (op 644) 


s_add_i32 s0, s1, s7 // (op 644) 


s_ashr_i32 s1, s0, s6 // (op 644) 


s_lshl_b32 s0, s15, 24 // (op 644) 
s_ashr_i32 s6, s0, 31 // (op 644) 

s_xor_b32 s0, s1, s6 // (op 644) 

s_sub_i32 s1, s0, s6 // (op 644) 


s_mul_i32 s0, s14, s18 // (op 644) 

s_mul_i32 s6, s50, s0 // (op 644) 

s_mul_i32 s0, s1, s6 // (op 644) 

s_sub_u32 s6, s16, s0 // (op 644) 

s_mul_hi_i32 s0, s6, s13 // (op 644) 
s_add_i32 s7, s0, s6 // (op 644) 


s_ashr_i32 s0, s7, 31 // (op 644) 
s_cmp_eq_i32 s13, 0 // (op 644) 
s_mov_b32 s6, scc // (op 644) 
s_cmp_eq_i32 s6, 0 // (op 644) 
s_cselect_b32 s18, 0, -1 // (op 644) 

s_and_b32 s6, s49, 31 // (op 644) 
s_lshl_b32 s19, 1, s6 // (op 644) 
s_add_i32 s68, s19, s18 // (op 644) 


s_and_b32 s18, s0, s68 // (op 644) 


s_add_i32 s0, s7, s18 // (op 644) 


s_ashr_i32 s7, s0, s6 // (op 644) 


s_lshl_b32 s0, s49, 24 // (op 644) 
s_ashr_i32 s6, s0, 31 // (op 644) 

s_xor_b32 s0, s7, s6 // (op 644) 

s_sub_i32 s7, s0, s6 // (op 644) 


s_cmp_eq_i32 s1, 0 // (op 644) 
s_mov_b32 s0, scc // (op 644) 

s_cmp_eq_i32 s0, 0 // (op 644) 
s_cselect_b32 s1, s7, s51 // (op 644) 



v_lshl_add_u32 v17, s1, 4, v12 // (op 644) 


v_mul_lo_u32 v12, v17, s36 // (op 644) 

s_ashr_i64 s[0:1], s[24:25], 7 // (op 644) 
s_lshr_b32 s51, s0, 1 // (op 644) 

v_lshl_add_u32 v17, s51, 3, v18 // (op 644) 


v_add_u32 v18, v17, v12 // (op 644) 


s_mul_i32 s1, s36, s5 // (op 646) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 646) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 646) least significant
s_mul_hi_u32 s6, s36, s4 // (op 646) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 646) most significant: sum
s_add_u32 s1, s1, s6 // (op 646) most significant: sum


s_mov_b32 s64, s38 // (op 646) reg expression
s_mov_b32 s65, s39 // (op 646) reg expression
s_mov_b32 s66, s0 // (op 646) reg expression
s_mov_b32 s67, 131072 // (op 646) call()

v_lshlrev_b32 v12, 7, v1 // (op 737) 
v_lshrrev_b32 v17, 3, v12 // (op 737) 

v_and_b32 v12, -4, v1 // (op 790) 
v_sub_u32 v20, v1, v12 // (op 790) 

v_lshlrev_b32 v12, 5, v20 // (op 790) 

v_lshrrev_b32 v20, 2, v1 // (op 790) 
v_lshl_add_u32 v21, v20, 7, v12 // (op 790) 


v_mul_lo_u32 v12, v21, 6 // (op 790) 

v_lshrrev_b32 v20, 3, v12 // (op 790) 

v_and_b32 v12, -16, v1 // (op 843) 
v_sub_u32 v21, v1, v12 // (op 843) 

v_and_b32 v12, 4294967280, v1 // (op 843) 
v_add_u32 v22, v12, v21 // (op 843) 


v_and_b32 v12, -4, v1 // (op 896) 
v_sub_u32 v21, v1, v12 // (op 896) 

v_and_b32 v12, 4294967292, v1 // (op 896) 
v_add_u32 v23, v12, v21 // (op 896) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 172) 
s_and_b32 s51, s0, 4294967294 // (op 172) 

s_ashr_i64 s[0:1], s[24:25], 7 // (op 171) 
s_mov_b32 s6, s0 // (op 171) convert

s_cmp_lt_u32 s51, s6 // (op 171) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_171 // (op 171) Condition: Top (jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_171 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_171:

s_add_u32 s51, s51, 1 // (op 173) 
s_lshl_b64 s[0:1], s[2:3], 2 // (op 176) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 176) 


v_mov_b32 v12, v14 // (op 176) 
v_mov_b32 v16, v12 // (op 176) reg expression
buffer_load_dwordx2 v[26:27], v16, s[52:55], 0 offen // (op 176) Load value
v_add_u32 v16, v16, v24 // (op 176) 
buffer_load_dwordx2 v[28:29], v16, s[52:55], 0 offen // (op 176) Load value



s_lshl_b64 s[0:1], s[2:3], 7 // (op 439) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 439) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 439) 

v_add_u32 v14, v14, s0 // (op 439) 



s_barrier  // (op 1097) 
v_mov_b32 v12, v15 // (op 190) 




buffer_load_dwordx4 v[30:33], v12, s[56:59], 0 offen // (op 190) Load value
s_waitcnt vmcnt(2)
ds_write_b64 v17, v[26:27] offset:1600 // (op 177) Store local data 
buffer_load_dwordx2 v[34:35], v12, s[56:59], 0 offen offset:16 // (op 190) Load value

v_add_u32 v15, 96, v15 // (op 508) 
s_waitcnt vmcnt(2)
ds_write_b64 v17, v[28:29] offset:1608 // (op 177) Store local data 



v_mov_b32 v12, v13 // (op 191) 
s_barrier  // (op 1110) 
v_mov_b32 v16, v12 // (op 191) reg expression




buffer_load_ubyte v19, v16, s[60:63], 0 offen offset: 0 // (op 191) Load value


s_lshl_b32 s0, s30, 2 // (op 572) 
v_add_u32 v13, v13, s0 // (op 572) 

s_waitcnt vmcnt(2)
ds_write_b128 v20, v[30:33] // (op 187) Store local data 


v_mov_b32 v12, v18 // (op 192) 

buffer_load_ubyte v16, v12, s[64:67], 0 offen offset: 0 // (op 192) Load value

v_add_u32 v18, 4, v18 // (op 636) 
s_waitcnt vmcnt(2)
ds_write_b64 v20, v[34:35] offset:16 // (op 187) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1100) 



s_barrier  // (op 1124) 
ds_read_b64_tr_b4 v[24:25], v3 offset:1600 // (op 179) Transpose load local data 



ds_read_b64_tr_b4 v[26:27], v3 offset:1728 // (op 179) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v22, v19 offset:2624 // (op 188) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1114) 



s_barrier  // (op 1138) 
ds_read_b128 v[28:31], v4 // (op 181) Load local data 



ds_read_b64 v[32:33], v4 offset:16 // (op 181) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v16 offset:1536 // (op 189) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1128) 


ds_read_u8 v12, v5 offset:2624 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1142) 



ds_read_u8 v16, v6 offset:1536 // (op 185) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[28:33], a[0:3], v12, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 186) 




s_ashr_i64 s[0:1], s[24:25], 7 // (op 171) 
s_mov_b32 s6, s0 // (op 171) convert

s_cmp_lt_u32 s51, s6 // (op 171) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_171 // (op 171) Condition: Bottom (jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_171 if true)
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_171:























s_barrier  // (op 1216) 








v_cmp_eq_f32 vcc, 0.00000, v0 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_133
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v11, v4 // (op 50) 
v_mul_f32 v13, v11, v5 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 

s_mul_hi_i32 s0, s16, s9 // (op 938) 
v_lshrrev_b32 v3, 6, v1 // (op 938) 
s_add_i32 s1, s0, s16 // (op 938) 

v_and_b32 v4, 63, v1 // (op 938) 
s_ashr_i32 s0, s1, 31 // (op 938) 
v_lshrrev_b32 v5, 4, v4 // (op 938) 
s_cmp_eq_i32 s9, 0 // (op 938) 
s_mov_b32 s2, scc // (op 938) 
v_lshlrev_b32 v6, 2, v5 // (op 938) 

s_cmp_eq_i32 s2, 0 // (op 938) 
s_cselect_b32 s3, 0, -1 // (op 938) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 938) 


s_and_b32 s2, s10, 31 // (op 938) 
v_lshrrev_b32 v3, 2, v4 // (op 938) 

s_lshl_b32 s6, 1, s2 // (op 938) 
v_and_b32 v4, 3, v3 // (op 938) 

s_add_i32 s7, s6, s3 // (op 938) 


v_and_b32 v3, 3, v1 // (op 938) 
s_and_b32 s3, s0, s7 // (op 938) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 938) 


s_add_i32 s0, s1, s3 // (op 938) 


s_ashr_i32 s1, s0, s2 // (op 938) 


s_lshl_b32 s0, s10, 24 // (op 938) 
s_ashr_i32 s2, s0, 31 // (op 938) 

s_xor_b32 s0, s1, s2 // (op 938) 

s_sub_i32 s1, s0, s2 // (op 938) 


s_mul_hi_i32 s0, s1, s11 // (op 938) 
s_add_i32 s2, s0, s1 // (op 938) 

s_ashr_i32 s0, s2, 31 // (op 938) 
s_cmp_eq_i32 s11, 0 // (op 938) 
s_mov_b32 s3, scc // (op 938) 
s_cmp_eq_i32 s3, 0 // (op 938) 
s_cselect_b32 s6, 0, -1 // (op 938) 

s_and_b32 s3, s12, 31 // (op 938) 
s_lshl_b32 s7, 1, s3 // (op 938) 
s_add_i32 s18, s7, s6 // (op 938) 


s_and_b32 s19, s0, s18 // (op 938) 


s_add_i32 s0, s2, s19 // (op 938) 


s_ashr_i32 s2, s0, s3 // (op 938) 


s_lshl_b32 s0, s12, 24 // (op 938) 
s_ashr_i32 s3, s0, 31 // (op 938) 

s_xor_b32 s0, s2, s3 // (op 938) 

s_sub_i32 s2, s0, s3 // (op 938) 


s_mul_i32 s0, s1, s14 // (op 938) 
s_sub_u32 s3, s16, s0 // (op 938) 

s_mul_i32 s0, s2, s14 // (op 938) 
s_add_u32 s0, s0, s3 // (op 938) 

s_mul_hi_i32 s3, s16, s8 // (op 938) 
s_add_i32 s6, s3, s16 // (op 938) 

s_ashr_i32 s3, s6, 31 // (op 938) 
s_cmp_eq_i32 s8, 0 // (op 938) 
s_mov_b32 s7, scc // (op 938) 
s_cmp_eq_i32 s7, 0 // (op 938) 
s_cselect_b32 s18, 0, -1 // (op 938) 

s_and_b32 s7, s15, 31 // (op 938) 
s_lshl_b32 s19, 1, s7 // (op 938) 
s_add_i32 s22, s19, s18 // (op 938) 


s_and_b32 s18, s3, s22 // (op 938) 


s_add_i32 s3, s6, s18 // (op 938) 


s_ashr_i32 s6, s3, s7 // (op 938) 


s_lshl_b32 s3, s15, 24 // (op 938) 
s_ashr_i32 s7, s3, 31 // (op 938) 

s_xor_b32 s3, s6, s7 // (op 938) 

s_sub_i32 s6, s3, s7 // (op 938) 


s_add_u32 s18, s4, 16 // (op 938) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 938) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 938) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 938) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 4 // (op 938) 

s_mul_i32 s3, s14, s18 // (op 938) 
s_mul_i32 s7, s50, s3 // (op 938) 

s_mul_i32 s3, s6, s7 // (op 938) 

s_sub_u32 s7, s16, s3 // (op 938) 

s_mov_b32 s3, s7 // (op 938) 
s_mul_hi_i32 s22, s3, s13 // (op 938) 
s_add_i32 s23, s22, s3 // (op 938) 


s_ashr_i32 s3, s23, 31 // (op 938) 
s_cmp_eq_i32 s13, 0 // (op 938) 
s_mov_b32 s22, scc // (op 938) 
s_cmp_eq_i32 s22, 0 // (op 938) 
s_cselect_b32 s24, 0, -1 // (op 938) 

s_and_b32 s22, s49, 31 // (op 938) 
s_lshl_b32 s25, 1, s22 // (op 938) 
s_add_i32 s26, s25, s24 // (op 938) 


s_and_b32 s24, s3, s26 // (op 938) 


s_add_i32 s3, s23, s24 // (op 938) 


s_ashr_i32 s23, s3, s22 // (op 938) 


s_lshl_b32 s3, s49, 24 // (op 938) 
s_ashr_i32 s22, s3, 31 // (op 938) 

s_xor_b32 s3, s23, s22 // (op 938) 

s_sub_i32 s23, s3, s22 // (op 938) 


s_mul_i32 s3, s48, s14 // (op 938) 
s_add_u32 s24, s20, 16 // (op 938) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 938) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 938) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 938) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 4 // (op 938) 

s_sub_i32 s22, s24, s3 // (op 938) 

s_mul_i32 s24, s23, s22 // (op 938) 

s_sub_u32 s22, s7, s24 // (op 938) 


s_add_u32 s7, s22, s3 // (op 938) 


s_cmp_eq_i32 s6, 0 // (op 938) 
s_mov_b32 s3, scc // (op 938) 

s_cmp_eq_i32 s3, 0 // (op 938) 
s_cselect_b32 s6, s7, s0 // (op 938) 


s_mul_i32 s0, s2, s18 // (op 938) 


v_lshl_add_u32 v3, s6, 4, v5 // (op 938) 


s_sub_i32 s2, s1, s0 // (op 938) 


s_cmp_eq_i32 s3, 0 // (op 938) 
s_cselect_b32 s22, s23, s2 // (op 938) 



v_lshl_add_u32 v4, s22, 4, v6 // (op 938) 


v_mul_lo_u32 v5, v4, s40 // (op 938) 

v_add_lshl_u32 v4, v3, v5, 2 // (op 938) 


s_mul_i32 s7, s40, s5 // (op 940) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 940) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 940) least significant
s_mul_hi_u32 s1, s40, s4 // (op 940) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 940) most significant: sum
s_add_u32 s7, s7, s1 // (op 940) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 940) 

s_mov_b32 s0, s42 // (op 940) reg expression
s_mov_b32 s1, s43 // (op 940) reg expression
s_mov_b32 s2, s18 // (op 940) reg expression
s_mov_b32 s3, 131072 // (op 940) call()




buffer_store_dwordx4 v[12:15], v4, s[0:3], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_133
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_133:
s_mul_hi_i32 s6, s16, s9 // (op 707) 
v_lshrrev_b32 v3, 6, v1 // (op 707) 
s_add_i32 s7, s6, s16 // (op 707) 

v_and_b32 v4, 63, v1 // (op 707) 
s_ashr_i32 s6, s7, 31 // (op 707) 
v_lshrrev_b32 v5, 4, v4 // (op 707) 
s_cmp_eq_i32 s9, 0 // (op 707) 
s_mov_b32 s18, scc // (op 707) 
v_lshlrev_b32 v6, 2, v5 // (op 707) 

s_cmp_eq_i32 s18, 0 // (op 707) 
s_cselect_b32 s19, 0, -1 // (op 707) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 707) 


s_and_b32 s18, s10, 31 // (op 707) 
v_lshrrev_b32 v3, 2, v4 // (op 707) 

s_lshl_b32 s22, 1, s18 // (op 707) 
v_and_b32 v4, 3, v3 // (op 707) 

s_add_i32 s23, s22, s19 // (op 707) 


v_and_b32 v3, 3, v1 // (op 707) 
s_and_b32 s19, s6, s23 // (op 707) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 707) 


s_add_i32 s6, s7, s19 // (op 707) 


s_ashr_i32 s7, s6, s18 // (op 707) 


s_lshl_b32 s6, s10, 24 // (op 707) 
s_ashr_i32 s18, s6, 31 // (op 707) 

s_xor_b32 s6, s7, s18 // (op 707) 

s_sub_i32 s7, s6, s18 // (op 707) 


s_mul_hi_i32 s6, s7, s11 // (op 707) 
s_add_i32 s18, s6, s7 // (op 707) 

s_ashr_i32 s6, s18, 31 // (op 707) 
s_cmp_eq_i32 s11, 0 // (op 707) 
s_mov_b32 s19, scc // (op 707) 
s_cmp_eq_i32 s19, 0 // (op 707) 
s_cselect_b32 s22, 0, -1 // (op 707) 

s_and_b32 s19, s12, 31 // (op 707) 
s_lshl_b32 s23, 1, s19 // (op 707) 
s_add_i32 s24, s23, s22 // (op 707) 


s_and_b32 s22, s6, s24 // (op 707) 


s_add_i32 s6, s18, s22 // (op 707) 


s_ashr_i32 s18, s6, s19 // (op 707) 


s_lshl_b32 s6, s12, 24 // (op 707) 
s_ashr_i32 s19, s6, 31 // (op 707) 

s_xor_b32 s6, s18, s19 // (op 707) 

s_sub_i32 s18, s6, s19 // (op 707) 


s_mul_i32 s6, s7, s14 // (op 707) 
s_sub_u32 s19, s16, s6 // (op 707) 

s_mul_i32 s6, s18, s14 // (op 707) 
s_add_u32 s6, s6, s19 // (op 707) 

s_mul_hi_i32 s19, s16, s8 // (op 707) 
s_add_i32 s22, s19, s16 // (op 707) 

s_ashr_i32 s19, s22, 31 // (op 707) 
s_cmp_eq_i32 s8, 0 // (op 707) 
s_mov_b32 s23, scc // (op 707) 
s_cmp_eq_i32 s23, 0 // (op 707) 
s_cselect_b32 s24, 0, -1 // (op 707) 

s_and_b32 s23, s15, 31 // (op 707) 
s_lshl_b32 s25, 1, s23 // (op 707) 
s_add_i32 s26, s25, s24 // (op 707) 


s_and_b32 s24, s19, s26 // (op 707) 


s_add_i32 s19, s22, s24 // (op 707) 


s_ashr_i32 s22, s19, s23 // (op 707) 


s_lshl_b32 s19, s15, 24 // (op 707) 
s_ashr_i32 s23, s19, 31 // (op 707) 

s_xor_b32 s19, s22, s23 // (op 707) 

s_sub_i32 s22, s19, s23 // (op 707) 


s_add_u32 s38, s4, 16 // (op 707) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 707) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 707) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 707) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 707) 

s_mul_i32 s19, s14, s38 // (op 707) 
s_mul_i32 s23, s50, s19 // (op 707) 

s_mul_i32 s19, s22, s23 // (op 707) 

s_sub_u32 s23, s16, s19 // (op 707) 

s_mov_b32 s19, s23 // (op 707) 
s_mul_hi_i32 s24, s19, s13 // (op 707) 
s_add_i32 s25, s24, s19 // (op 707) 


s_ashr_i32 s19, s25, 31 // (op 707) 
s_cmp_eq_i32 s13, 0 // (op 707) 
s_mov_b32 s24, scc // (op 707) 
s_cmp_eq_i32 s24, 0 // (op 707) 
s_cselect_b32 s26, 0, -1 // (op 707) 

s_and_b32 s24, s49, 31 // (op 707) 
s_lshl_b32 s27, 1, s24 // (op 707) 
s_add_i32 s28, s27, s26 // (op 707) 


s_and_b32 s26, s19, s28 // (op 707) 


s_add_i32 s19, s25, s26 // (op 707) 


s_ashr_i32 s25, s19, s24 // (op 707) 


s_lshl_b32 s19, s49, 24 // (op 707) 
s_ashr_i32 s24, s19, 31 // (op 707) 

s_xor_b32 s19, s25, s24 // (op 707) 

s_sub_i32 s25, s19, s24 // (op 707) 


s_mul_i32 s19, s48, s14 // (op 707) 
s_add_u32 s26, s20, 16 // (op 707) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 707) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 707) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 707) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 707) 

s_sub_i32 s24, s26, s19 // (op 707) 

s_mul_i32 s26, s25, s24 // (op 707) 

s_sub_u32 s24, s23, s26 // (op 707) 


s_add_u32 s23, s24, s19 // (op 707) 


s_cmp_eq_i32 s22, 0 // (op 707) 
s_mov_b32 s19, scc // (op 707) 

s_cmp_eq_i32 s19, 0 // (op 707) 
s_cselect_b32 s22, s23, s6 // (op 707) 


s_mul_i32 s6, s18, s38 // (op 707) 


v_lshl_add_u32 v3, s22, 4, v5 // (op 707) 


s_sub_i32 s18, s7, s6 // (op 707) 


s_cmp_eq_i32 s19, 0 // (op 707) 
s_cselect_b32 s6, s25, s18 // (op 707) 



v_lshl_add_u32 v4, s6, 4, v6 // (op 707) 


v_mul_lo_u32 v5, v4, s44 // (op 707) 

v_add_lshl_u32 v4, v3, v5, 2 // (op 707) 


s_mul_i32 s7, s44, s5 // (op 709) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 709) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 709) least significant
s_mul_hi_u32 s19, s44, s4 // (op 709) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 709) most significant: sum
s_add_u32 s7, s7, s19 // (op 709) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 709) 

s_mov_b32 s36, s46 // (op 709) reg expression
s_mov_b32 s37, s47 // (op 709) reg expression
s_mov_b32 s38, s18 // (op 709) reg expression
s_mov_b32 s39, 131072 // (op 709) call()





buffer_load_dwordx4 v[6:9], v4, s[36:39], 0 offen // (op 145) Load value



v_accvgpr_read v12, a0 // (op 142) 
v_accvgpr_read v13, a1 // (op 142) 
v_accvgpr_read v14, a2 // (op 142) 
v_accvgpr_read v15, a3 // (op 142) 
v_mul_f32 v16, v11, v12 // (op 142) 
v_mul_f32 v17, v11, v13 // (op 142) 
v_mul_f32 v18, v11, v14 // (op 142) 
v_mul_f32 v19, v11, v15 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v0, v6 // (op 146) 
v_mul_f32 v13, v0, v7 // (op 146) 
v_mul_f32 v14, v0, v8 // (op 146) 
v_mul_f32 v15, v0, v9 // (op 146) 

v_add_f32 v4, v12, v16 // (op 143) 
v_add_f32 v5, v13, v17 // (op 143) 
v_add_f32 v6, v14, v18 // (op 143) 
v_add_f32 v7, v15, v19 // (op 143) 


s_mul_hi_i32 s6, s16, s9 // (op 955) 
v_lshrrev_b32 v3, 6, v1 // (op 955) 
s_add_i32 s7, s6, s16 // (op 955) 

v_and_b32 v8, 63, v1 // (op 955) 
s_ashr_i32 s6, s7, 31 // (op 955) 
v_lshrrev_b32 v9, 4, v8 // (op 955) 
s_cmp_eq_i32 s9, 0 // (op 955) 
s_mov_b32 s18, scc // (op 955) 
v_lshlrev_b32 v10, 2, v9 // (op 955) 

s_cmp_eq_i32 s18, 0 // (op 955) 
s_cselect_b32 s19, 0, -1 // (op 955) 

v_lshl_add_u32 v9, v3, 4, v10 // (op 955) 


s_and_b32 s18, s10, 31 // (op 955) 
v_lshrrev_b32 v3, 2, v8 // (op 955) 

s_lshl_b32 s22, 1, s18 // (op 955) 
v_and_b32 v8, 3, v3 // (op 955) 

s_add_i32 s23, s22, s19 // (op 955) 


v_and_b32 v3, 3, v1 // (op 955) 
s_and_b32 s19, s6, s23 // (op 955) 


v_lshl_add_u32 v10, v8, 2, v3 // (op 955) 


s_add_i32 s6, s7, s19 // (op 955) 


s_ashr_i32 s7, s6, s18 // (op 955) 


s_lshl_b32 s6, s10, 24 // (op 955) 
s_ashr_i32 s18, s6, 31 // (op 955) 

s_xor_b32 s6, s7, s18 // (op 955) 

s_sub_i32 s7, s6, s18 // (op 955) 


s_mul_hi_i32 s6, s7, s11 // (op 955) 
s_add_i32 s18, s6, s7 // (op 955) 

s_ashr_i32 s6, s18, 31 // (op 955) 
s_cmp_eq_i32 s11, 0 // (op 955) 
s_mov_b32 s19, scc // (op 955) 
s_cmp_eq_i32 s19, 0 // (op 955) 
s_cselect_b32 s22, 0, -1 // (op 955) 

s_and_b32 s19, s12, 31 // (op 955) 
s_lshl_b32 s23, 1, s19 // (op 955) 
s_add_i32 s24, s23, s22 // (op 955) 


s_and_b32 s22, s6, s24 // (op 955) 


s_add_i32 s6, s18, s22 // (op 955) 


s_ashr_i32 s18, s6, s19 // (op 955) 


s_lshl_b32 s6, s12, 24 // (op 955) 
s_ashr_i32 s19, s6, 31 // (op 955) 

s_xor_b32 s6, s18, s19 // (op 955) 

s_sub_i32 s18, s6, s19 // (op 955) 


s_mul_i32 s6, s7, s14 // (op 955) 
s_sub_u32 s19, s16, s6 // (op 955) 

s_mul_i32 s6, s18, s14 // (op 955) 
s_add_u32 s6, s6, s19 // (op 955) 

s_mul_hi_i32 s19, s16, s8 // (op 955) 
s_add_i32 s22, s19, s16 // (op 955) 

s_ashr_i32 s19, s22, 31 // (op 955) 
s_cmp_eq_i32 s8, 0 // (op 955) 
s_mov_b32 s23, scc // (op 955) 
s_cmp_eq_i32 s23, 0 // (op 955) 
s_cselect_b32 s24, 0, -1 // (op 955) 

s_and_b32 s23, s15, 31 // (op 955) 
s_lshl_b32 s25, 1, s23 // (op 955) 
s_add_i32 s26, s25, s24 // (op 955) 


s_and_b32 s24, s19, s26 // (op 955) 


s_add_i32 s19, s22, s24 // (op 955) 


s_ashr_i32 s22, s19, s23 // (op 955) 


s_lshl_b32 s19, s15, 24 // (op 955) 
s_ashr_i32 s23, s19, 31 // (op 955) 

s_xor_b32 s19, s22, s23 // (op 955) 

s_sub_i32 s22, s19, s23 // (op 955) 


s_add_u32 s38, s4, 16 // (op 955) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 955) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 955) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 955) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 955) 

s_mul_i32 s19, s14, s38 // (op 955) 
s_mul_i32 s23, s50, s19 // (op 955) 

s_mul_i32 s19, s22, s23 // (op 955) 

s_sub_u32 s23, s16, s19 // (op 955) 

s_mov_b32 s19, s23 // (op 955) 
s_mul_hi_i32 s24, s19, s13 // (op 955) 
s_add_i32 s25, s24, s19 // (op 955) 


s_ashr_i32 s19, s25, 31 // (op 955) 
s_cmp_eq_i32 s13, 0 // (op 955) 
s_mov_b32 s24, scc // (op 955) 
s_cmp_eq_i32 s24, 0 // (op 955) 
s_cselect_b32 s26, 0, -1 // (op 955) 

s_and_b32 s24, s49, 31 // (op 955) 
s_lshl_b32 s27, 1, s24 // (op 955) 
s_add_i32 s28, s27, s26 // (op 955) 


s_and_b32 s26, s19, s28 // (op 955) 


s_add_i32 s19, s25, s26 // (op 955) 


s_ashr_i32 s25, s19, s24 // (op 955) 


s_lshl_b32 s19, s49, 24 // (op 955) 
s_ashr_i32 s24, s19, 31 // (op 955) 

s_xor_b32 s19, s25, s24 // (op 955) 

s_sub_i32 s25, s19, s24 // (op 955) 


s_mul_i32 s19, s48, s14 // (op 955) 
s_add_u32 s26, s20, 16 // (op 955) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 955) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 955) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 955) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 955) 

s_sub_i32 s24, s26, s19 // (op 955) 

s_mul_i32 s26, s25, s24 // (op 955) 

s_sub_u32 s24, s23, s26 // (op 955) 


s_add_u32 s23, s24, s19 // (op 955) 


s_cmp_eq_i32 s22, 0 // (op 955) 
s_mov_b32 s19, scc // (op 955) 

s_cmp_eq_i32 s19, 0 // (op 955) 
s_cselect_b32 s22, s23, s6 // (op 955) 


s_mul_i32 s6, s18, s38 // (op 955) 


v_lshl_add_u32 v3, s22, 4, v9 // (op 955) 


s_sub_i32 s18, s7, s6 // (op 955) 


s_cmp_eq_i32 s19, 0 // (op 955) 
s_cselect_b32 s6, s25, s18 // (op 955) 



v_lshl_add_u32 v8, s6, 4, v10 // (op 955) 


v_mul_lo_u32 v9, v8, s40 // (op 955) 

v_add_lshl_u32 v8, v3, v9, 2 // (op 955) 


s_mul_i32 s7, s40, s5 // (op 957) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 957) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 957) least significant
s_mul_hi_u32 s19, s40, s4 // (op 957) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 957) most significant: sum
s_add_u32 s7, s7, s19 // (op 957) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 957) 

s_mov_b32 s0, s42 // (op 957) reg expression
s_mov_b32 s1, s43 // (op 957) reg expression
s_mov_b32 s2, s18 // (op 957) reg expression
s_mov_b32 s3, 131072 // (op 957) call()




buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen // (op 144) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_133:
















s_endpgm  // End of RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.LRR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end:
.size RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_, .LRR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end-RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 5376 // lds bytes
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
    DataTypeB: B6
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
    args: [ { type: int64, semantic: StrideA0 },
            { type: int64, semantic: SizeFree1 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int64, semantic: SizeFree0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
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
  - .name: RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 5376
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 66
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 16
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 20
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 24
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 28
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_0_size_0_15
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 80
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 128
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 136
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 144
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_11
        .size: 8
        .offset: 152
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 160
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Subtract_24
        .size: 4
        .offset: 168
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 172
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 176
        .value_kind: by_value
...

.end_amdgpu_metadata

