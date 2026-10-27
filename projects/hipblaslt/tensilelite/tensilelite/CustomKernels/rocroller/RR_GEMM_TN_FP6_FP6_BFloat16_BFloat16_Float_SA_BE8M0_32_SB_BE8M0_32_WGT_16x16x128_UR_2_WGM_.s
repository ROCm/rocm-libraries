// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_,@function
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 769) 
v_and_b32 v3, 15, v1 // (op 769) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 769) 


v_and_b32 v0, 63, v1 // (op 769) 
v_lshrrev_b32 v3, 4, v0 // (op 769) 

v_lshlrev_b32 v0, 5, v3 // (op 769) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 769) 


v_mul_lo_u32 v0, v3, 6 // (op 769) 

v_lshrrev_b32 v3, 3, v0 // (op 769) 

v_and_b32 v0, 15, v1 // (op 822) 
v_and_b32 v4, 63, v1 // (op 822) 
v_lshrrev_b32 v5, 4, v4 // (op 822) 

v_lshlrev_b32 v4, 5, v5 // (op 822) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 822) 


v_mul_lo_u32 v0, v5, 6 // (op 822) 

v_lshrrev_b32 v4, 3, v0 // (op 822) 

v_lshrrev_b32 v0, 6, v1 // (op 875) 
v_and_b32 v5, 15, v1 // (op 875) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 875) 


v_and_b32 v0, 63, v1 // (op 875) 
v_lshrrev_b32 v5, 4, v0 // (op 875) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 875) 


v_and_b32 v5, 15, v1 // (op 928) 
v_and_b32 v6, 63, v1 // (op 928) 
v_lshrrev_b32 v7, 4, v6 // (op 928) 

v_lshl_add_u32 v6, v5, 2, v7 // (op 928) 


v_lshrrev_b32 v5, 6, v1 // (op 986) 
v_and_b32 v7, 15, v1 // (op 986) 
v_lshl_add_u32 v8, v5, 4, v7 // (op 986) 


v_and_b32 v5, 63, v1 // (op 986) 
v_lshrrev_b32 v7, 4, v5 // (op 986) 

v_lshlrev_b32 v5, 5, v7 // (op 986) 

v_lshl_add_u32 v7, v8, 7, v5 // (op 986) 


v_mul_lo_u32 v5, v7, 6 // (op 986) 

v_lshrrev_b32 v7, 3, v5 // (op 986) 

v_and_b32 v5, 15, v1 // (op 1010) 
v_and_b32 v8, 63, v1 // (op 1010) 
v_lshrrev_b32 v9, 4, v8 // (op 1010) 

v_lshlrev_b32 v8, 5, v9 // (op 1010) 

v_lshl_add_u32 v9, v5, 7, v8 // (op 1010) 


v_mul_lo_u32 v5, v9, 6 // (op 1010) 

v_lshrrev_b32 v8, 3, v5 // (op 1010) 

v_lshrrev_b32 v5, 6, v1 // (op 1034) 
v_and_b32 v9, 15, v1 // (op 1034) 
v_lshl_add_u32 v10, v5, 4, v9 // (op 1034) 


v_and_b32 v5, 63, v1 // (op 1034) 
v_lshrrev_b32 v9, 4, v5 // (op 1034) 

v_lshl_add_u32 v5, v10, 2, v9 // (op 1034) 


v_and_b32 v9, 15, v1 // (op 1058) 
v_and_b32 v10, 63, v1 // (op 1058) 
v_lshrrev_b32 v11, 4, v10 // (op 1058) 

v_lshl_add_u32 v10, v9, 2, v11 // (op 1058) 


v_mov_b32 v9, s8 // (op 2) Move value
v_mov_b32 v11, s9 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
s_mul_hi_i32 s0, s16, s12 // (op 479) 
v_lshrrev_b32 v12, 2, v1 // (op 479) 
s_add_i32 s1, s0, s16 // (op 479) 

v_and_b32 v13, -4, v1 // (op 479) 
s_ashr_i32 s0, s1, 31 // (op 479) 
v_sub_u32 v14, v1, v13 // (op 479) 

s_cmp_eq_i32 s12, 0 // (op 479) 
s_mov_b32 s8, scc // (op 479) 
v_lshlrev_b32 v13, 5, v14 // (op 479) 

s_cmp_eq_i32 s8, 0 // (op 479) 
s_cselect_b32 s9, 0, -1 // (op 479) 

v_mov_b32 v15, 0 // (op 479) convert
v_mov_b32 v14, v13 // (op 479) convert

s_and_b32 s8, s13, 31 // (op 479) 
s_lshl_b32 s18, 1, s8 // (op 479) 
s_add_i32 s19, s18, s9 // (op 479) 


s_and_b32 s9, s0, s19 // (op 479) 


s_add_i32 s0, s1, s9 // (op 479) 


s_ashr_i32 s1, s0, s8 // (op 479) 


s_lshl_b32 s0, s13, 24 // (op 479) 
s_ashr_i32 s8, s0, 31 // (op 479) 

s_xor_b32 s0, s1, s8 // (op 479) 

s_sub_i32 s1, s0, s8 // (op 479) 


s_mul_hi_i32 s0, s1, s14 // (op 479) 
s_add_i32 s8, s0, s1 // (op 479) 

s_ashr_i32 s0, s8, 31 // (op 479) 
s_cmp_eq_i32 s14, 0 // (op 479) 
s_mov_b32 s9, scc // (op 479) 
s_cmp_eq_i32 s9, 0 // (op 479) 
s_cselect_b32 s18, 0, -1 // (op 479) 

s_and_b32 s9, s15, 31 // (op 479) 
s_lshl_b32 s19, 1, s9 // (op 479) 
s_add_i32 s49, s19, s18 // (op 479) 


s_and_b32 s18, s0, s49 // (op 479) 


s_add_i32 s0, s8, s18 // (op 479) 


s_ashr_i32 s8, s0, s9 // (op 479) 


s_lshl_b32 s0, s15, 24 // (op 479) 
s_ashr_i32 s9, s0, 31 // (op 479) 

s_xor_b32 s0, s8, s9 // (op 479) 

s_sub_i32 s8, s0, s9 // (op 479) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s45 // (op 479) 

s_sub_u32 s1, s16, s0 // (op 479) 

s_mul_i32 s0, s8, s45 // (op 479) 
s_add_u32 s0, s0, s1 // (op 479) 


s_mul_hi_i32 s1, s16, s11 // (op 479) 
s_add_i32 s8, s1, s16 // (op 479) 

s_ashr_i32 s1, s8, 31 // (op 479) 
s_cmp_eq_i32 s11, 0 // (op 479) 
s_mov_b32 s9, scc // (op 479) 
s_cmp_eq_i32 s9, 0 // (op 479) 
s_cselect_b32 s18, 0, -1 // (op 479) 

s_and_b32 s9, s46, 31 // (op 479) 
s_lshl_b32 s19, 1, s9 // (op 479) 
s_add_i32 s49, s19, s18 // (op 479) 


s_and_b32 s18, s1, s49 // (op 479) 


s_add_i32 s1, s8, s18 // (op 479) 


s_ashr_i32 s8, s1, s9 // (op 479) 


s_lshl_b32 s1, s46, 24 // (op 479) 
s_ashr_i32 s9, s1, 31 // (op 479) 

s_xor_b32 s1, s8, s9 // (op 479) 

s_sub_i32 s8, s1, s9 // (op 479) 


s_add_u32 s18, s4, 16 // (op 479) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 479) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 479) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 479) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 479) 

s_mul_i32 s1, s45, s18 // (op 479) 

s_mul_i32 s9, s10, s1 // (op 479) 

s_mul_i32 s1, s8, s9 // (op 479) 

s_sub_u32 s9, s16, s1 // (op 479) 

s_mov_b32 s1, s9 // (op 479) 
s_mul_hi_i32 s18, s1, s44 // (op 479) 
s_add_i32 s19, s18, s1 // (op 479) 


s_ashr_i32 s1, s19, 31 // (op 479) 
s_cmp_eq_i32 s44, 0 // (op 479) 
s_mov_b32 s18, scc // (op 479) 
s_cmp_eq_i32 s18, 0 // (op 479) 
s_cselect_b32 s49, 0, -1 // (op 479) 

s_and_b32 s18, s48, 31 // (op 479) 
s_lshl_b32 s50, 1, s18 // (op 479) 
s_add_i32 s51, s50, s49 // (op 479) 


s_and_b32 s49, s1, s51 // (op 479) 


s_add_i32 s1, s19, s49 // (op 479) 


s_ashr_i32 s19, s1, s18 // (op 479) 


s_lshl_b32 s1, s48, 24 // (op 479) 
s_ashr_i32 s18, s1, 31 // (op 479) 

s_xor_b32 s1, s19, s18 // (op 479) 

s_sub_i32 s19, s1, s18 // (op 479) 


s_mul_i32 s1, s47, s45 // (op 479) 
s_add_u32 s50, s6, 16 // (op 479) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 479) End of Int64 add, unlocking SCC
s_sub_u32 s52, s50, 1 // (op 479) least significant half; sets scc
s_subb_u32 s53, s51, 0 // (op 479) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 4 // (op 479) 

s_sub_i32 s18, s50, s1 // (op 479) 

s_mul_i32 s49, s19, s18 // (op 479) 


s_sub_u32 s18, s9, s49 // (op 479) 


s_add_u32 s9, s18, s1 // (op 479) 


s_cmp_eq_i32 s8, 0 // (op 479) 
s_mov_b32 s1, scc // (op 479) 

s_cmp_eq_i32 s1, 0 // (op 479) 
s_cselect_b32 s8, s9, s0 // (op 479) 



v_lshl_add_u32 v13, s8, 4, v12 // (op 479) 


v_mul_lo_u32 v17, v13, s3 // (op 479) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 479) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 479) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 479) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 479) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 479) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 479) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 479) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 479) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 479) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 479) 

v_mov_b32 v12, v14 // (op 479) convert

s_mul_i32 s1, s7, 6 // (op 481) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 481) least significant
s_mul_hi_u32 s8, s6, 6 // (op 481) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 481) most significant: sum

s_mul_i32 s9, s2, s1 // (op 481) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 481) most significant: low of high * low
s_mul_i32 s8, s2, s0 // (op 481) least significant
s_mul_hi_u32 s19, s2, s0 // (op 481) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 481) most significant: sum
s_add_u32 s9, s9, s19 // (op 481) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 481) 

s_mov_b32 s52, s20 // (op 481) reg expression
s_mov_b32 s53, s21 // (op 481) reg expression
s_mov_b32 s54, s0 // (op 481) reg expression
s_mov_b32 s55, 131072 // (op 481) call()

s_mul_hi_i32 s0, s16, s12 // (op 498) 
v_lshrrev_b32 v13, 2, v1 // (op 498) 
s_add_i32 s1, s0, s16 // (op 498) 

v_and_b32 v14, -4, v1 // (op 498) 
s_ashr_i32 s0, s1, 31 // (op 498) 
v_sub_u32 v15, v1, v14 // (op 498) 

s_cmp_eq_i32 s12, 0 // (op 498) 
s_mov_b32 s8, scc // (op 498) 
v_lshlrev_b32 v14, 5, v15 // (op 498) 

s_cmp_eq_i32 s8, 0 // (op 498) 
s_cselect_b32 s9, 0, -1 // (op 498) 

v_mov_b32 v17, 0 // (op 498) convert
v_mov_b32 v16, v14 // (op 498) convert

s_and_b32 s8, s13, 31 // (op 498) 
s_lshl_b32 s18, 1, s8 // (op 498) 
s_add_i32 s19, s18, s9 // (op 498) 


s_and_b32 s9, s0, s19 // (op 498) 


s_add_i32 s0, s1, s9 // (op 498) 


s_ashr_i32 s1, s0, s8 // (op 498) 


s_lshl_b32 s0, s13, 24 // (op 498) 
s_ashr_i32 s8, s0, 31 // (op 498) 

s_xor_b32 s0, s1, s8 // (op 498) 

s_sub_i32 s1, s0, s8 // (op 498) 


s_mul_hi_i32 s0, s1, s14 // (op 498) 
s_add_i32 s8, s0, s1 // (op 498) 

s_ashr_i32 s0, s8, 31 // (op 498) 
s_cmp_eq_i32 s14, 0 // (op 498) 
s_mov_b32 s9, scc // (op 498) 
s_cmp_eq_i32 s9, 0 // (op 498) 
s_cselect_b32 s18, 0, -1 // (op 498) 

s_and_b32 s9, s15, 31 // (op 498) 
s_lshl_b32 s19, 1, s9 // (op 498) 
s_add_i32 s49, s19, s18 // (op 498) 


s_and_b32 s18, s0, s49 // (op 498) 


s_add_i32 s0, s8, s18 // (op 498) 


s_ashr_i32 s8, s0, s9 // (op 498) 


s_lshl_b32 s0, s15, 24 // (op 498) 
s_ashr_i32 s9, s0, 31 // (op 498) 

s_xor_b32 s0, s8, s9 // (op 498) 

s_sub_i32 s8, s0, s9 // (op 498) 


s_mul_i32 s0, s1, s45 // (op 498) 

s_sub_u32 s1, s16, s0 // (op 498) 

s_mul_i32 s0, s8, s45 // (op 498) 
s_add_u32 s0, s0, s1 // (op 498) 


s_mul_hi_i32 s1, s16, s11 // (op 498) 
s_add_i32 s8, s1, s16 // (op 498) 

s_ashr_i32 s1, s8, 31 // (op 498) 
s_cmp_eq_i32 s11, 0 // (op 498) 
s_mov_b32 s9, scc // (op 498) 
s_cmp_eq_i32 s9, 0 // (op 498) 
s_cselect_b32 s18, 0, -1 // (op 498) 

s_and_b32 s9, s46, 31 // (op 498) 
s_lshl_b32 s19, 1, s9 // (op 498) 
s_add_i32 s49, s19, s18 // (op 498) 


s_and_b32 s18, s1, s49 // (op 498) 


s_add_i32 s1, s8, s18 // (op 498) 


s_ashr_i32 s8, s1, s9 // (op 498) 


s_lshl_b32 s1, s46, 24 // (op 498) 
s_ashr_i32 s9, s1, 31 // (op 498) 

s_xor_b32 s1, s8, s9 // (op 498) 

s_sub_i32 s8, s1, s9 // (op 498) 


s_add_u32 s18, s4, 16 // (op 498) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 498) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 498) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 498) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 498) 

s_mul_i32 s1, s45, s18 // (op 498) 

s_mul_i32 s9, s10, s1 // (op 498) 

s_mul_i32 s1, s8, s9 // (op 498) 

s_sub_u32 s9, s16, s1 // (op 498) 

s_mov_b32 s1, s9 // (op 498) 
s_mul_hi_i32 s18, s1, s44 // (op 498) 
s_add_i32 s19, s18, s1 // (op 498) 


s_ashr_i32 s1, s19, 31 // (op 498) 
s_cmp_eq_i32 s44, 0 // (op 498) 
s_mov_b32 s18, scc // (op 498) 
s_cmp_eq_i32 s18, 0 // (op 498) 
s_cselect_b32 s49, 0, -1 // (op 498) 

s_and_b32 s18, s48, 31 // (op 498) 
s_lshl_b32 s50, 1, s18 // (op 498) 
s_add_i32 s51, s50, s49 // (op 498) 


s_and_b32 s49, s1, s51 // (op 498) 


s_add_i32 s1, s19, s49 // (op 498) 


s_ashr_i32 s19, s1, s18 // (op 498) 


s_lshl_b32 s1, s48, 24 // (op 498) 
s_ashr_i32 s18, s1, 31 // (op 498) 

s_xor_b32 s1, s19, s18 // (op 498) 

s_sub_i32 s19, s1, s18 // (op 498) 


s_mul_i32 s1, s47, s45 // (op 498) 
s_add_u32 s50, s6, 16 // (op 498) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 498) End of Int64 add, unlocking SCC
s_sub_u32 s56, s50, 1 // (op 498) least significant half; sets scc
s_subb_u32 s57, s51, 0 // (op 498) most significant half; uses scc

s_ashr_i64 s[50:51], s[56:57], 4 // (op 498) 

s_sub_i32 s18, s50, s1 // (op 498) 

s_mul_i32 s49, s19, s18 // (op 498) 


s_sub_u32 s18, s9, s49 // (op 498) 


s_add_u32 s9, s18, s1 // (op 498) 


s_cmp_eq_i32 s8, 0 // (op 498) 
s_mov_b32 s1, scc // (op 498) 

s_cmp_eq_i32 s1, 0 // (op 498) 
s_cselect_b32 s8, s9, s0 // (op 498) 



v_lshl_add_u32 v14, s8, 4, v13 // (op 498) 


v_mul_lo_u32 v19, v14, s3 // (op 498) most significant: low of low * high
v_mul_hi_u32 v13, v14, s2 // (op 498) most significant: high of low * low
v_mul_lo_u32 v18, v14, s2 // (op 498) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 498) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 498) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 498) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 498) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 498) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 498) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 498) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 498) 

v_mov_b32 v13, v14 // (op 498) convert

s_mul_i32 s1, s7, 6 // (op 500) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 500) least significant
s_mul_hi_u32 s8, s6, 6 // (op 500) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 500) most significant: sum

s_mul_i32 s9, s2, s1 // (op 500) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 500) most significant: low of high * low
s_mul_i32 s8, s2, s0 // (op 500) least significant
s_mul_hi_u32 s19, s2, s0 // (op 500) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 500) most significant: sum
s_add_u32 s9, s9, s19 // (op 500) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 500) 

s_mov_b32 s52, s20 // (op 500) reg expression
s_mov_b32 s53, s21 // (op 500) reg expression
s_mov_b32 s54, s0 // (op 500) reg expression
s_mov_b32 s55, 131072 // (op 500) call()

s_mul_hi_i32 s0, s16, s12 // (op 543) 
v_lshrrev_b32 v14, 2, v1 // (op 543) 
s_add_i32 s1, s0, s16 // (op 543) 

v_and_b32 v15, -4, v1 // (op 543) 
s_ashr_i32 s0, s1, 31 // (op 543) 
v_sub_u32 v16, v1, v15 // (op 543) 

s_cmp_eq_i32 s12, 0 // (op 543) 
s_mov_b32 s8, scc // (op 543) 
v_lshlrev_b32 v15, 5, v16 // (op 543) 

s_cmp_eq_i32 s8, 0 // (op 543) 
s_cselect_b32 s9, 0, -1 // (op 543) 

v_mov_b32 v17, 0 // (op 543) convert
v_mov_b32 v16, v15 // (op 543) convert

s_and_b32 s8, s13, 31 // (op 543) 
s_lshl_b32 s18, 1, s8 // (op 543) 
s_add_i32 s19, s18, s9 // (op 543) 


s_and_b32 s9, s0, s19 // (op 543) 


s_add_i32 s0, s1, s9 // (op 543) 


s_ashr_i32 s1, s0, s8 // (op 543) 


s_lshl_b32 s0, s13, 24 // (op 543) 
s_ashr_i32 s8, s0, 31 // (op 543) 

s_xor_b32 s0, s1, s8 // (op 543) 

s_sub_i32 s1, s0, s8 // (op 543) 


s_mul_hi_i32 s0, s1, s14 // (op 543) 
s_add_i32 s8, s0, s1 // (op 543) 

s_ashr_i32 s0, s8, 31 // (op 543) 
s_cmp_eq_i32 s14, 0 // (op 543) 
s_mov_b32 s9, scc // (op 543) 
s_cmp_eq_i32 s9, 0 // (op 543) 
s_cselect_b32 s18, 0, -1 // (op 543) 

s_and_b32 s9, s15, 31 // (op 543) 
s_lshl_b32 s19, 1, s9 // (op 543) 
s_add_i32 s49, s19, s18 // (op 543) 


s_and_b32 s18, s0, s49 // (op 543) 


s_add_i32 s0, s8, s18 // (op 543) 


s_ashr_i32 s8, s0, s9 // (op 543) 


s_lshl_b32 s0, s15, 24 // (op 543) 
s_ashr_i32 s9, s0, 31 // (op 543) 

s_xor_b32 s0, s8, s9 // (op 543) 

s_sub_i32 s8, s0, s9 // (op 543) 


s_add_u32 s18, s4, 16 // (op 543) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 543) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 543) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 543) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 543) 

s_mul_i32 s0, s8, s18 // (op 543) 

s_sub_i32 s8, s1, s0 // (op 543) 


s_mul_hi_i32 s9, s16, s11 // (op 543) 
s_add_i32 s0, s9, s16 // (op 543) 

s_ashr_i32 s1, s0, 31 // (op 543) 
s_cmp_eq_i32 s11, 0 // (op 543) 
s_mov_b32 s9, scc // (op 543) 
s_cmp_eq_i32 s9, 0 // (op 543) 
s_cselect_b32 s49, 0, -1 // (op 543) 

s_and_b32 s9, s46, 31 // (op 543) 
s_lshl_b32 s50, 1, s9 // (op 543) 
s_add_i32 s51, s50, s49 // (op 543) 


s_and_b32 s49, s1, s51 // (op 543) 


s_add_i32 s1, s0, s49 // (op 543) 


s_ashr_i32 s0, s1, s9 // (op 543) 


s_lshl_b32 s1, s46, 24 // (op 543) 
s_ashr_i32 s9, s1, 31 // (op 543) 

s_xor_b32 s1, s0, s9 // (op 543) 

s_sub_i32 s0, s1, s9 // (op 543) 


s_mul_i32 s1, s45, s18 // (op 543) 

s_mul_i32 s9, s10, s1 // (op 543) 

s_mul_i32 s1, s0, s9 // (op 543) 

s_sub_u32 s9, s16, s1 // (op 543) 

s_mul_hi_i32 s1, s9, s44 // (op 543) 
s_add_i32 s18, s1, s9 // (op 543) 


s_ashr_i32 s1, s18, 31 // (op 543) 
s_cmp_eq_i32 s44, 0 // (op 543) 
s_mov_b32 s9, scc // (op 543) 
s_cmp_eq_i32 s9, 0 // (op 543) 
s_cselect_b32 s19, 0, -1 // (op 543) 

s_and_b32 s9, s48, 31 // (op 543) 
s_lshl_b32 s49, 1, s9 // (op 543) 
s_add_i32 s50, s49, s19 // (op 543) 


s_and_b32 s19, s1, s50 // (op 543) 


s_add_i32 s1, s18, s19 // (op 543) 


s_ashr_i32 s18, s1, s9 // (op 543) 


s_lshl_b32 s1, s48, 24 // (op 543) 
s_ashr_i32 s9, s1, 31 // (op 543) 

s_xor_b32 s1, s18, s9 // (op 543) 

s_sub_i32 s18, s1, s9 // (op 543) 


s_cmp_eq_i32 s0, 0 // (op 543) 
s_mov_b32 s1, scc // (op 543) 

s_cmp_eq_i32 s1, 0 // (op 543) 
s_cselect_b32 s0, s18, s8 // (op 543) 



v_lshl_add_u32 v15, s0, 4, v14 // (op 543) 


v_mul_lo_u32 v19, v15, s23 // (op 543) most significant: low of low * high
v_mul_hi_u32 v14, v15, s22 // (op 543) most significant: high of low * low
v_mul_lo_u32 v18, v15, s22 // (op 543) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 543) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 543) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 543) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 543) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 543) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 543) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 543) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 543) 

v_mov_b32 v16, v14 // (op 543) convert

s_mul_i32 s1, s5, 6 // (op 545) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 545) least significant
s_mul_hi_u32 s8, s4, 6 // (op 545) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 545) most significant: sum

s_mul_i32 s9, s22, s1 // (op 545) most significant: low of low * high
s_mul_i32 s18, s23, s0 // (op 545) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 545) least significant
s_mul_hi_u32 s19, s22, s0 // (op 545) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 545) most significant: sum
s_add_u32 s9, s9, s19 // (op 545) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 545) 

s_mov_b32 s56, s24 // (op 545) reg expression
s_mov_b32 s57, s25 // (op 545) reg expression
s_mov_b32 s58, s0 // (op 545) reg expression
s_mov_b32 s59, 131072 // (op 545) call()

s_mul_hi_i32 s0, s16, s12 // (op 562) 
v_lshrrev_b32 v14, 2, v1 // (op 562) 
s_add_i32 s1, s0, s16 // (op 562) 

v_and_b32 v15, -4, v1 // (op 562) 
s_ashr_i32 s0, s1, 31 // (op 562) 
v_sub_u32 v17, v1, v15 // (op 562) 

s_cmp_eq_i32 s12, 0 // (op 562) 
s_mov_b32 s8, scc // (op 562) 
v_lshlrev_b32 v15, 5, v17 // (op 562) 

s_cmp_eq_i32 s8, 0 // (op 562) 
s_cselect_b32 s9, 0, -1 // (op 562) 

v_mov_b32 v19, 0 // (op 562) convert
v_mov_b32 v18, v15 // (op 562) convert

s_and_b32 s8, s13, 31 // (op 562) 
s_lshl_b32 s18, 1, s8 // (op 562) 
s_add_i32 s19, s18, s9 // (op 562) 


s_and_b32 s9, s0, s19 // (op 562) 


s_add_i32 s0, s1, s9 // (op 562) 


s_ashr_i32 s1, s0, s8 // (op 562) 


s_lshl_b32 s0, s13, 24 // (op 562) 
s_ashr_i32 s8, s0, 31 // (op 562) 

s_xor_b32 s0, s1, s8 // (op 562) 

s_sub_i32 s1, s0, s8 // (op 562) 


s_mul_hi_i32 s0, s1, s14 // (op 562) 
s_add_i32 s8, s0, s1 // (op 562) 

s_ashr_i32 s0, s8, 31 // (op 562) 
s_cmp_eq_i32 s14, 0 // (op 562) 
s_mov_b32 s9, scc // (op 562) 
s_cmp_eq_i32 s9, 0 // (op 562) 
s_cselect_b32 s18, 0, -1 // (op 562) 

s_and_b32 s9, s15, 31 // (op 562) 
s_lshl_b32 s19, 1, s9 // (op 562) 
s_add_i32 s49, s19, s18 // (op 562) 


s_and_b32 s18, s0, s49 // (op 562) 


s_add_i32 s0, s8, s18 // (op 562) 


s_ashr_i32 s8, s0, s9 // (op 562) 


s_lshl_b32 s0, s15, 24 // (op 562) 
s_ashr_i32 s9, s0, 31 // (op 562) 

s_xor_b32 s0, s8, s9 // (op 562) 

s_sub_i32 s8, s0, s9 // (op 562) 


s_add_u32 s18, s4, 16 // (op 562) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 562) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 562) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 562) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 562) 

s_mul_i32 s0, s8, s18 // (op 562) 

s_sub_i32 s8, s1, s0 // (op 562) 


s_mul_hi_i32 s9, s16, s11 // (op 562) 
s_add_i32 s0, s9, s16 // (op 562) 

s_ashr_i32 s1, s0, 31 // (op 562) 
s_cmp_eq_i32 s11, 0 // (op 562) 
s_mov_b32 s9, scc // (op 562) 
s_cmp_eq_i32 s9, 0 // (op 562) 
s_cselect_b32 s49, 0, -1 // (op 562) 

s_and_b32 s9, s46, 31 // (op 562) 
s_lshl_b32 s50, 1, s9 // (op 562) 
s_add_i32 s51, s50, s49 // (op 562) 


s_and_b32 s49, s1, s51 // (op 562) 


s_add_i32 s1, s0, s49 // (op 562) 


s_ashr_i32 s0, s1, s9 // (op 562) 


s_lshl_b32 s1, s46, 24 // (op 562) 
s_ashr_i32 s9, s1, 31 // (op 562) 

s_xor_b32 s1, s0, s9 // (op 562) 

s_sub_i32 s0, s1, s9 // (op 562) 


s_mul_i32 s1, s45, s18 // (op 562) 

s_mul_i32 s9, s10, s1 // (op 562) 

s_mul_i32 s1, s0, s9 // (op 562) 

s_sub_u32 s9, s16, s1 // (op 562) 

s_mul_hi_i32 s1, s9, s44 // (op 562) 
s_add_i32 s18, s1, s9 // (op 562) 


s_ashr_i32 s1, s18, 31 // (op 562) 
s_cmp_eq_i32 s44, 0 // (op 562) 
s_mov_b32 s9, scc // (op 562) 
s_cmp_eq_i32 s9, 0 // (op 562) 
s_cselect_b32 s19, 0, -1 // (op 562) 

s_and_b32 s9, s48, 31 // (op 562) 
s_lshl_b32 s49, 1, s9 // (op 562) 
s_add_i32 s50, s49, s19 // (op 562) 


s_and_b32 s19, s1, s50 // (op 562) 


s_add_i32 s1, s18, s19 // (op 562) 


s_ashr_i32 s18, s1, s9 // (op 562) 


s_lshl_b32 s1, s48, 24 // (op 562) 
s_ashr_i32 s9, s1, 31 // (op 562) 

s_xor_b32 s1, s18, s9 // (op 562) 

s_sub_i32 s18, s1, s9 // (op 562) 


s_cmp_eq_i32 s0, 0 // (op 562) 
s_mov_b32 s1, scc // (op 562) 

s_cmp_eq_i32 s1, 0 // (op 562) 
s_cselect_b32 s0, s18, s8 // (op 562) 



v_lshl_add_u32 v15, s0, 4, v14 // (op 562) 


v_mul_lo_u32 v21, v15, s23 // (op 562) most significant: low of low * high
v_mul_hi_u32 v14, v15, s22 // (op 562) most significant: high of low * low
v_mul_lo_u32 v20, v15, s22 // (op 562) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 562) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 562) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 562) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 562) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 562) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 562) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 562) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 562) 

v_mov_b32 v17, v14 // (op 562) convert

s_mul_i32 s1, s5, 6 // (op 564) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 564) least significant
s_mul_hi_u32 s8, s4, 6 // (op 564) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 564) most significant: sum

s_mul_i32 s9, s22, s1 // (op 564) most significant: low of low * high
s_mul_i32 s18, s23, s0 // (op 564) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 564) least significant
s_mul_hi_u32 s19, s22, s0 // (op 564) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 564) most significant: sum
s_add_u32 s9, s9, s19 // (op 564) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 564) 

s_mov_b32 s56, s24 // (op 564) reg expression
s_mov_b32 s57, s25 // (op 564) reg expression
s_mov_b32 s58, s0 // (op 564) reg expression
s_mov_b32 s59, 131072 // (op 564) call()

s_mul_hi_i32 s0, s16, s12 // (op 607) 
v_lshrrev_b32 v14, 2, v1 // (op 607) 
s_add_i32 s1, s0, s16 // (op 607) 

v_and_b32 v15, -4, v1 // (op 607) 
s_ashr_i32 s0, s1, 31 // (op 607) 
v_sub_u32 v18, v1, v15 // (op 607) 

s_cmp_eq_i32 s12, 0 // (op 607) 
s_mov_b32 s8, scc // (op 607) 
s_cmp_eq_i32 s8, 0 // (op 607) 
s_cselect_b32 s9, 0, -1 // (op 607) 

s_and_b32 s8, s13, 31 // (op 607) 
s_lshl_b32 s18, 1, s8 // (op 607) 
s_add_i32 s19, s18, s9 // (op 607) 


s_and_b32 s9, s0, s19 // (op 607) 


s_add_i32 s0, s1, s9 // (op 607) 


s_ashr_i32 s1, s0, s8 // (op 607) 


s_lshl_b32 s0, s13, 24 // (op 607) 
s_ashr_i32 s8, s0, 31 // (op 607) 

s_xor_b32 s0, s1, s8 // (op 607) 

s_sub_i32 s1, s0, s8 // (op 607) 


s_mul_hi_i32 s0, s1, s14 // (op 607) 
s_add_i32 s8, s0, s1 // (op 607) 

s_ashr_i32 s0, s8, 31 // (op 607) 
s_cmp_eq_i32 s14, 0 // (op 607) 
s_mov_b32 s9, scc // (op 607) 
s_cmp_eq_i32 s9, 0 // (op 607) 
s_cselect_b32 s18, 0, -1 // (op 607) 

s_and_b32 s9, s15, 31 // (op 607) 
s_lshl_b32 s19, 1, s9 // (op 607) 
s_add_i32 s49, s19, s18 // (op 607) 


s_and_b32 s18, s0, s49 // (op 607) 


s_add_i32 s0, s8, s18 // (op 607) 


s_ashr_i32 s8, s0, s9 // (op 607) 


s_lshl_b32 s0, s15, 24 // (op 607) 
s_ashr_i32 s9, s0, 31 // (op 607) 

s_xor_b32 s0, s8, s9 // (op 607) 

s_sub_i32 s8, s0, s9 // (op 607) 


s_mul_i32 s0, s1, s45 // (op 607) 

s_sub_u32 s1, s16, s0 // (op 607) 

s_mul_i32 s0, s8, s45 // (op 607) 
s_add_u32 s0, s0, s1 // (op 607) 


s_mul_hi_i32 s1, s16, s11 // (op 607) 
s_add_i32 s8, s1, s16 // (op 607) 

s_ashr_i32 s1, s8, 31 // (op 607) 
s_cmp_eq_i32 s11, 0 // (op 607) 
s_mov_b32 s9, scc // (op 607) 
s_cmp_eq_i32 s9, 0 // (op 607) 
s_cselect_b32 s18, 0, -1 // (op 607) 

s_and_b32 s9, s46, 31 // (op 607) 
s_lshl_b32 s19, 1, s9 // (op 607) 
s_add_i32 s49, s19, s18 // (op 607) 


s_and_b32 s18, s1, s49 // (op 607) 


s_add_i32 s1, s8, s18 // (op 607) 


s_ashr_i32 s8, s1, s9 // (op 607) 


s_lshl_b32 s1, s46, 24 // (op 607) 
s_ashr_i32 s9, s1, 31 // (op 607) 

s_xor_b32 s1, s8, s9 // (op 607) 

s_sub_i32 s8, s1, s9 // (op 607) 


s_add_u32 s18, s4, 16 // (op 607) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 607) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 607) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 607) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 607) 

s_mul_i32 s1, s45, s18 // (op 607) 

s_mul_i32 s9, s10, s1 // (op 607) 

s_mul_i32 s1, s8, s9 // (op 607) 

s_sub_u32 s9, s16, s1 // (op 607) 

s_mov_b32 s1, s9 // (op 607) 
s_mul_hi_i32 s18, s1, s44 // (op 607) 
s_add_i32 s19, s18, s1 // (op 607) 


s_ashr_i32 s1, s19, 31 // (op 607) 
s_cmp_eq_i32 s44, 0 // (op 607) 
s_mov_b32 s18, scc // (op 607) 
s_cmp_eq_i32 s18, 0 // (op 607) 
s_cselect_b32 s49, 0, -1 // (op 607) 

s_and_b32 s18, s48, 31 // (op 607) 
s_lshl_b32 s50, 1, s18 // (op 607) 
s_add_i32 s51, s50, s49 // (op 607) 


s_and_b32 s49, s1, s51 // (op 607) 


s_add_i32 s1, s19, s49 // (op 607) 


s_ashr_i32 s19, s1, s18 // (op 607) 


s_lshl_b32 s1, s48, 24 // (op 607) 
s_ashr_i32 s18, s1, 31 // (op 607) 

s_xor_b32 s1, s19, s18 // (op 607) 

s_sub_i32 s19, s1, s18 // (op 607) 


s_mul_i32 s1, s47, s45 // (op 607) 
s_add_u32 s50, s6, 16 // (op 607) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 607) End of Int64 add, unlocking SCC
s_sub_u32 s60, s50, 1 // (op 607) least significant half; sets scc
s_subb_u32 s61, s51, 0 // (op 607) most significant half; uses scc

s_ashr_i64 s[50:51], s[60:61], 4 // (op 607) 

s_sub_i32 s18, s50, s1 // (op 607) 

s_mul_i32 s49, s19, s18 // (op 607) 


s_sub_u32 s18, s9, s49 // (op 607) 


s_add_u32 s9, s18, s1 // (op 607) 


s_cmp_eq_i32 s8, 0 // (op 607) 
s_mov_b32 s1, scc // (op 607) 

s_cmp_eq_i32 s1, 0 // (op 607) 
s_cselect_b32 s8, s9, s0 // (op 607) 



v_lshl_add_u32 v15, s8, 4, v14 // (op 607) 


v_mul_lo_u32 v14, v15, s26 // (op 607) 
v_add_u32 v14, v14, v18 // (op 607) 


s_mul_i32 s1, s26, s7 // (op 609) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 609) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 609) least significant
s_mul_hi_u32 s9, s26, s6 // (op 609) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 609) most significant: sum
s_add_u32 s1, s1, s9 // (op 609) most significant: sum


s_mov_b32 s60, s28 // (op 609) reg expression
s_mov_b32 s61, s29 // (op 609) reg expression
s_mov_b32 s62, s0 // (op 609) reg expression
s_mov_b32 s63, 131072 // (op 609) call()

s_mul_hi_i32 s0, s16, s12 // (op 626) 
v_lshrrev_b32 v15, 2, v1 // (op 626) 
s_add_i32 s1, s0, s16 // (op 626) 

v_and_b32 v18, -4, v1 // (op 626) 
s_ashr_i32 s0, s1, 31 // (op 626) 
v_sub_u32 v19, v1, v18 // (op 626) 

s_cmp_eq_i32 s12, 0 // (op 626) 
s_mov_b32 s8, scc // (op 626) 
s_cmp_eq_i32 s8, 0 // (op 626) 
s_cselect_b32 s9, 0, -1 // (op 626) 

s_and_b32 s8, s13, 31 // (op 626) 
s_lshl_b32 s18, 1, s8 // (op 626) 
s_add_i32 s19, s18, s9 // (op 626) 


s_and_b32 s9, s0, s19 // (op 626) 


s_add_i32 s0, s1, s9 // (op 626) 


s_ashr_i32 s1, s0, s8 // (op 626) 


s_lshl_b32 s0, s13, 24 // (op 626) 
s_ashr_i32 s8, s0, 31 // (op 626) 

s_xor_b32 s0, s1, s8 // (op 626) 

s_sub_i32 s1, s0, s8 // (op 626) 


s_mul_hi_i32 s0, s1, s14 // (op 626) 
s_add_i32 s8, s0, s1 // (op 626) 

s_ashr_i32 s0, s8, 31 // (op 626) 
s_cmp_eq_i32 s14, 0 // (op 626) 
s_mov_b32 s9, scc // (op 626) 
s_cmp_eq_i32 s9, 0 // (op 626) 
s_cselect_b32 s18, 0, -1 // (op 626) 

s_and_b32 s9, s15, 31 // (op 626) 
s_lshl_b32 s19, 1, s9 // (op 626) 
s_add_i32 s49, s19, s18 // (op 626) 


s_and_b32 s18, s0, s49 // (op 626) 


s_add_i32 s0, s8, s18 // (op 626) 


s_ashr_i32 s8, s0, s9 // (op 626) 


s_lshl_b32 s0, s15, 24 // (op 626) 
s_ashr_i32 s9, s0, 31 // (op 626) 

s_xor_b32 s0, s8, s9 // (op 626) 

s_sub_i32 s8, s0, s9 // (op 626) 


s_mul_i32 s0, s1, s45 // (op 626) 

s_sub_u32 s1, s16, s0 // (op 626) 

s_mul_i32 s0, s8, s45 // (op 626) 
s_add_u32 s0, s0, s1 // (op 626) 


s_mul_hi_i32 s1, s16, s11 // (op 626) 
s_add_i32 s8, s1, s16 // (op 626) 

s_ashr_i32 s1, s8, 31 // (op 626) 
s_cmp_eq_i32 s11, 0 // (op 626) 
s_mov_b32 s9, scc // (op 626) 
s_cmp_eq_i32 s9, 0 // (op 626) 
s_cselect_b32 s18, 0, -1 // (op 626) 

s_and_b32 s9, s46, 31 // (op 626) 
s_lshl_b32 s19, 1, s9 // (op 626) 
s_add_i32 s49, s19, s18 // (op 626) 


s_and_b32 s18, s1, s49 // (op 626) 


s_add_i32 s1, s8, s18 // (op 626) 


s_ashr_i32 s8, s1, s9 // (op 626) 


s_lshl_b32 s1, s46, 24 // (op 626) 
s_ashr_i32 s9, s1, 31 // (op 626) 

s_xor_b32 s1, s8, s9 // (op 626) 

s_sub_i32 s8, s1, s9 // (op 626) 


s_add_u32 s18, s4, 16 // (op 626) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 626) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 626) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 626) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 626) 

s_mul_i32 s1, s45, s18 // (op 626) 

s_mul_i32 s9, s10, s1 // (op 626) 

s_mul_i32 s1, s8, s9 // (op 626) 

s_sub_u32 s9, s16, s1 // (op 626) 

s_mov_b32 s1, s9 // (op 626) 
s_mul_hi_i32 s18, s1, s44 // (op 626) 
s_add_i32 s19, s18, s1 // (op 626) 


s_ashr_i32 s1, s19, 31 // (op 626) 
s_cmp_eq_i32 s44, 0 // (op 626) 
s_mov_b32 s18, scc // (op 626) 
s_cmp_eq_i32 s18, 0 // (op 626) 
s_cselect_b32 s49, 0, -1 // (op 626) 

s_and_b32 s18, s48, 31 // (op 626) 
s_lshl_b32 s50, 1, s18 // (op 626) 
s_add_i32 s51, s50, s49 // (op 626) 


s_and_b32 s49, s1, s51 // (op 626) 


s_add_i32 s1, s19, s49 // (op 626) 


s_ashr_i32 s19, s1, s18 // (op 626) 


s_lshl_b32 s1, s48, 24 // (op 626) 
s_ashr_i32 s18, s1, 31 // (op 626) 

s_xor_b32 s1, s19, s18 // (op 626) 

s_sub_i32 s19, s1, s18 // (op 626) 


s_mul_i32 s1, s47, s45 // (op 626) 
s_add_u32 s50, s6, 16 // (op 626) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 626) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 626) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 626) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 4 // (op 626) 

s_sub_i32 s18, s50, s1 // (op 626) 

s_mul_i32 s49, s19, s18 // (op 626) 


s_sub_u32 s18, s9, s49 // (op 626) 


s_add_u32 s9, s18, s1 // (op 626) 


s_cmp_eq_i32 s8, 0 // (op 626) 
s_mov_b32 s1, scc // (op 626) 

s_cmp_eq_i32 s1, 0 // (op 626) 
s_cselect_b32 s8, s9, s0 // (op 626) 



v_lshl_add_u32 v18, s8, 4, v15 // (op 626) 


v_mul_lo_u32 v15, v18, s26 // (op 626) 
v_add_u32 v15, v15, v19 // (op 626) 


s_mul_i32 s1, s26, s7 // (op 628) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 628) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 628) least significant
s_mul_hi_u32 s9, s26, s6 // (op 628) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 628) most significant: sum
s_add_u32 s1, s1, s9 // (op 628) most significant: sum


s_mov_b32 s60, s28 // (op 628) reg expression
s_mov_b32 s61, s29 // (op 628) reg expression
s_mov_b32 s62, s0 // (op 628) reg expression
s_mov_b32 s63, 131072 // (op 628) call()

s_mul_hi_i32 s0, s16, s12 // (op 671) 
v_lshrrev_b32 v18, 2, v1 // (op 671) 
s_add_i32 s1, s0, s16 // (op 671) 

v_and_b32 v19, -4, v1 // (op 671) 
s_ashr_i32 s0, s1, 31 // (op 671) 
v_sub_u32 v20, v1, v19 // (op 671) 

s_cmp_eq_i32 s12, 0 // (op 671) 
s_mov_b32 s8, scc // (op 671) 
s_cmp_eq_i32 s8, 0 // (op 671) 
s_cselect_b32 s9, 0, -1 // (op 671) 

s_and_b32 s8, s13, 31 // (op 671) 
s_lshl_b32 s18, 1, s8 // (op 671) 
s_add_i32 s19, s18, s9 // (op 671) 


s_and_b32 s9, s0, s19 // (op 671) 


s_add_i32 s0, s1, s9 // (op 671) 


s_ashr_i32 s1, s0, s8 // (op 671) 


s_lshl_b32 s0, s13, 24 // (op 671) 
s_ashr_i32 s8, s0, 31 // (op 671) 

s_xor_b32 s0, s1, s8 // (op 671) 

s_sub_i32 s1, s0, s8 // (op 671) 


s_mul_hi_i32 s0, s1, s14 // (op 671) 
s_add_i32 s8, s0, s1 // (op 671) 

s_ashr_i32 s0, s8, 31 // (op 671) 
s_cmp_eq_i32 s14, 0 // (op 671) 
s_mov_b32 s9, scc // (op 671) 
s_cmp_eq_i32 s9, 0 // (op 671) 
s_cselect_b32 s18, 0, -1 // (op 671) 

s_and_b32 s9, s15, 31 // (op 671) 
s_lshl_b32 s19, 1, s9 // (op 671) 
s_add_i32 s49, s19, s18 // (op 671) 


s_and_b32 s18, s0, s49 // (op 671) 


s_add_i32 s0, s8, s18 // (op 671) 


s_ashr_i32 s8, s0, s9 // (op 671) 


s_lshl_b32 s0, s15, 24 // (op 671) 
s_ashr_i32 s9, s0, 31 // (op 671) 

s_xor_b32 s0, s8, s9 // (op 671) 

s_sub_i32 s8, s0, s9 // (op 671) 


s_add_u32 s18, s4, 16 // (op 671) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 671) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 671) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 671) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 671) 

s_mul_i32 s0, s8, s18 // (op 671) 

s_sub_i32 s8, s1, s0 // (op 671) 


s_mul_hi_i32 s9, s16, s11 // (op 671) 
s_add_i32 s0, s9, s16 // (op 671) 

s_ashr_i32 s1, s0, 31 // (op 671) 
s_cmp_eq_i32 s11, 0 // (op 671) 
s_mov_b32 s9, scc // (op 671) 
s_cmp_eq_i32 s9, 0 // (op 671) 
s_cselect_b32 s49, 0, -1 // (op 671) 

s_and_b32 s9, s46, 31 // (op 671) 
s_lshl_b32 s50, 1, s9 // (op 671) 
s_add_i32 s51, s50, s49 // (op 671) 


s_and_b32 s49, s1, s51 // (op 671) 


s_add_i32 s1, s0, s49 // (op 671) 


s_ashr_i32 s0, s1, s9 // (op 671) 


s_lshl_b32 s1, s46, 24 // (op 671) 
s_ashr_i32 s9, s1, 31 // (op 671) 

s_xor_b32 s1, s0, s9 // (op 671) 

s_sub_i32 s0, s1, s9 // (op 671) 


s_mul_i32 s1, s45, s18 // (op 671) 

s_mul_i32 s9, s10, s1 // (op 671) 

s_mul_i32 s1, s0, s9 // (op 671) 

s_sub_u32 s9, s16, s1 // (op 671) 

s_mul_hi_i32 s1, s9, s44 // (op 671) 
s_add_i32 s18, s1, s9 // (op 671) 


s_ashr_i32 s1, s18, 31 // (op 671) 
s_cmp_eq_i32 s44, 0 // (op 671) 
s_mov_b32 s9, scc // (op 671) 
s_cmp_eq_i32 s9, 0 // (op 671) 
s_cselect_b32 s19, 0, -1 // (op 671) 

s_and_b32 s9, s48, 31 // (op 671) 
s_lshl_b32 s49, 1, s9 // (op 671) 
s_add_i32 s50, s49, s19 // (op 671) 


s_and_b32 s19, s1, s50 // (op 671) 


s_add_i32 s1, s18, s19 // (op 671) 


s_ashr_i32 s18, s1, s9 // (op 671) 


s_lshl_b32 s1, s48, 24 // (op 671) 
s_ashr_i32 s9, s1, 31 // (op 671) 

s_xor_b32 s1, s18, s9 // (op 671) 

s_sub_i32 s18, s1, s9 // (op 671) 


s_cmp_eq_i32 s0, 0 // (op 671) 
s_mov_b32 s1, scc // (op 671) 

s_cmp_eq_i32 s1, 0 // (op 671) 
s_cselect_b32 s0, s18, s8 // (op 671) 



v_lshl_add_u32 v19, s0, 4, v18 // (op 671) 


v_mul_lo_u32 v18, v19, s30 // (op 671) 

v_add_u32 v19, v20, v18 // (op 671) 


s_mul_i32 s1, s30, s5 // (op 673) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 673) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 673) least significant
s_mul_hi_u32 s9, s30, s4 // (op 673) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 673) most significant: sum
s_add_u32 s1, s1, s9 // (op 673) most significant: sum


s_mov_b32 s64, s32 // (op 673) reg expression
s_mov_b32 s65, s33 // (op 673) reg expression
s_mov_b32 s66, s0 // (op 673) reg expression
s_mov_b32 s67, 131072 // (op 673) call()

s_mul_hi_i32 s0, s16, s12 // (op 690) 
v_lshrrev_b32 v18, 2, v1 // (op 690) 
s_add_i32 s1, s0, s16 // (op 690) 

v_and_b32 v20, -4, v1 // (op 690) 
s_ashr_i32 s0, s1, 31 // (op 690) 
v_sub_u32 v21, v1, v20 // (op 690) 

s_cmp_eq_i32 s12, 0 // (op 690) 
s_mov_b32 s8, scc // (op 690) 
s_cmp_eq_i32 s8, 0 // (op 690) 
s_cselect_b32 s9, 0, -1 // (op 690) 

s_and_b32 s8, s13, 31 // (op 690) 
s_lshl_b32 s18, 1, s8 // (op 690) 
s_add_i32 s19, s18, s9 // (op 690) 


s_and_b32 s9, s0, s19 // (op 690) 


s_add_i32 s0, s1, s9 // (op 690) 


s_ashr_i32 s1, s0, s8 // (op 690) 


s_lshl_b32 s0, s13, 24 // (op 690) 
s_ashr_i32 s8, s0, 31 // (op 690) 

s_xor_b32 s0, s1, s8 // (op 690) 

s_sub_i32 s1, s0, s8 // (op 690) 


s_mul_hi_i32 s0, s1, s14 // (op 690) 
s_add_i32 s8, s0, s1 // (op 690) 

s_ashr_i32 s0, s8, 31 // (op 690) 
s_cmp_eq_i32 s14, 0 // (op 690) 
s_mov_b32 s9, scc // (op 690) 
s_cmp_eq_i32 s9, 0 // (op 690) 
s_cselect_b32 s18, 0, -1 // (op 690) 

s_and_b32 s9, s15, 31 // (op 690) 
s_lshl_b32 s19, 1, s9 // (op 690) 
s_add_i32 s49, s19, s18 // (op 690) 


s_and_b32 s18, s0, s49 // (op 690) 


s_add_i32 s0, s8, s18 // (op 690) 


s_ashr_i32 s8, s0, s9 // (op 690) 


s_lshl_b32 s0, s15, 24 // (op 690) 
s_ashr_i32 s9, s0, 31 // (op 690) 

s_xor_b32 s0, s8, s9 // (op 690) 

s_sub_i32 s8, s0, s9 // (op 690) 


s_add_u32 s18, s4, 16 // (op 690) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 690) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 690) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 690) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 690) 

s_mul_i32 s0, s8, s18 // (op 690) 

s_sub_i32 s8, s1, s0 // (op 690) 


s_mul_hi_i32 s9, s16, s11 // (op 690) 
s_add_i32 s0, s9, s16 // (op 690) 

s_ashr_i32 s1, s0, 31 // (op 690) 
s_cmp_eq_i32 s11, 0 // (op 690) 
s_mov_b32 s9, scc // (op 690) 
s_cmp_eq_i32 s9, 0 // (op 690) 
s_cselect_b32 s49, 0, -1 // (op 690) 

s_and_b32 s9, s46, 31 // (op 690) 
s_lshl_b32 s50, 1, s9 // (op 690) 
s_add_i32 s51, s50, s49 // (op 690) 


s_and_b32 s49, s1, s51 // (op 690) 


s_add_i32 s1, s0, s49 // (op 690) 


s_ashr_i32 s0, s1, s9 // (op 690) 


s_lshl_b32 s1, s46, 24 // (op 690) 
s_ashr_i32 s9, s1, 31 // (op 690) 

s_xor_b32 s1, s0, s9 // (op 690) 

s_sub_i32 s0, s1, s9 // (op 690) 


s_mul_i32 s1, s45, s18 // (op 690) 

s_mul_i32 s9, s10, s1 // (op 690) 

s_mul_i32 s1, s0, s9 // (op 690) 

s_sub_u32 s9, s16, s1 // (op 690) 

s_mul_hi_i32 s1, s9, s44 // (op 690) 
s_add_i32 s18, s1, s9 // (op 690) 


s_ashr_i32 s1, s18, 31 // (op 690) 
s_cmp_eq_i32 s44, 0 // (op 690) 
s_mov_b32 s9, scc // (op 690) 
s_cmp_eq_i32 s9, 0 // (op 690) 
s_cselect_b32 s19, 0, -1 // (op 690) 

s_and_b32 s9, s48, 31 // (op 690) 
s_lshl_b32 s49, 1, s9 // (op 690) 
s_add_i32 s50, s49, s19 // (op 690) 


s_and_b32 s19, s1, s50 // (op 690) 


s_add_i32 s1, s18, s19 // (op 690) 


s_ashr_i32 s18, s1, s9 // (op 690) 


s_lshl_b32 s1, s48, 24 // (op 690) 
s_ashr_i32 s9, s1, 31 // (op 690) 

s_xor_b32 s1, s18, s9 // (op 690) 

s_sub_i32 s18, s1, s9 // (op 690) 


s_cmp_eq_i32 s0, 0 // (op 690) 
s_mov_b32 s1, scc // (op 690) 

s_cmp_eq_i32 s1, 0 // (op 690) 
s_cselect_b32 s0, s18, s8 // (op 690) 



v_lshl_add_u32 v20, s0, 4, v18 // (op 690) 


v_mul_lo_u32 v18, v20, s30 // (op 690) 

v_add_u32 v20, v21, v18 // (op 690) 


s_mul_i32 s1, s30, s5 // (op 692) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 692) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 692) least significant
s_mul_hi_u32 s9, s30, s4 // (op 692) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 692) most significant: sum
s_add_u32 s1, s1, s9 // (op 692) most significant: sum


s_mov_b32 s64, s32 // (op 692) reg expression
s_mov_b32 s65, s33 // (op 692) reg expression
s_mov_b32 s66, s0 // (op 692) reg expression
s_mov_b32 s67, 131072 // (op 692) call()




buffer_load_dwordx4 v[22:25], v16, s[56:59], 0 offen // (op 370) Load value
buffer_load_dwordx2 v[26:27], v16, s[56:59], 0 offen offset:16 // (op 370) Load value



buffer_load_ubyte v18, v19, s[64:67], 0 offen offset: 0 // (op 373) Load value



buffer_load_dwordx4 v[28:31], v12, s[52:55], 0 offen // (op 376) Load value
buffer_load_dwordx2 v[32:33], v12, s[52:55], 0 offen offset:16 // (op 376) Load value



buffer_load_ubyte v21, v14, s[60:63], 0 offen offset: 0 // (op 379) Load value



buffer_load_dwordx4 v[34:37], v16, s[56:59], 0 offen offset:96 // (op 382) Load value
buffer_load_dwordx2 v[38:39], v16, s[56:59], 0 offen offset:112 // (op 382) Load value




buffer_load_ubyte v16, v19, s[64:67], 0 offen offset:4 // (op 385) Load value




buffer_load_dwordx4 v[40:43], v12, s[52:55], 0 offen offset:96 // (op 388) Load value
buffer_load_dwordx2 v[44:45], v12, s[52:55], 0 offen offset:112 // (op 388) Load value




buffer_load_ubyte v12, v14, s[60:63], 0 offen offset:4 // (op 391) Load value

v_and_b32 v14, -4, v1 // (op 810) 
v_sub_u32 v19, v1, v14 // (op 810) 

v_lshlrev_b32 v14, 5, v19 // (op 810) 

v_lshrrev_b32 v19, 2, v1 // (op 810) 
v_lshl_add_u32 v46, v19, 7, v14 // (op 810) 


v_mul_lo_u32 v14, v46, 6 // (op 810) 

v_lshrrev_b32 v19, 3, v14 // (op 810) 




s_waitcnt vmcnt(11)
ds_write_b128 v19, v[22:25] // (op 393) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v19, v[26:27] offset:16 // (op 393) Store local data 

v_and_b32 v14, -4, v1 // (op 916) 
v_sub_u32 v19, v1, v14 // (op 916) 

v_and_b32 v14, 4294967292, v1 // (op 916) 
v_add_u32 v46, v14, v19 // (op 916) 





s_waitcnt vmcnt(9)
ds_write_b8 v46, v18 offset:1536 // (op 394) Store local data 

v_and_b32 v14, -4, v1 // (op 757) 
v_sub_u32 v19, v1, v14 // (op 757) 

v_lshlrev_b32 v14, 5, v19 // (op 757) 

v_lshrrev_b32 v19, 2, v1 // (op 757) 
v_lshl_add_u32 v46, v19, 7, v14 // (op 757) 


v_mul_lo_u32 v14, v46, 6 // (op 757) 

v_lshrrev_b32 v19, 3, v14 // (op 757) 




s_waitcnt vmcnt(8)
ds_write_b128 v19, v[28:31] offset:1600 // (op 395) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v19, v[32:33] offset:1616 // (op 395) Store local data 

v_and_b32 v14, -4, v1 // (op 863) 
v_sub_u32 v19, v1, v14 // (op 863) 

v_and_b32 v14, 4294967292, v1 // (op 863) 
v_add_u32 v46, v14, v19 // (op 863) 





s_waitcnt vmcnt(6)
ds_write_b8 v46, v21 offset:3136 // (op 396) Store local data 

v_and_b32 v14, -4, v1 // (op 728) 
v_sub_u32 v19, v1, v14 // (op 728) 

v_lshlrev_b32 v14, 5, v19 // (op 728) 

v_lshrrev_b32 v19, 2, v1 // (op 728) 
v_lshl_add_u32 v46, v19, 7, v14 // (op 728) 


v_mul_lo_u32 v14, v46, 6 // (op 728) 

v_lshrrev_b32 v19, 3, v14 // (op 728) 

v_and_b32 v14, -4, v1 // (op 781) 
v_sub_u32 v46, v1, v14 // (op 781) 

v_lshlrev_b32 v14, 5, v46 // (op 781) 

v_lshrrev_b32 v46, 2, v1 // (op 781) 
v_lshl_add_u32 v47, v46, 7, v14 // (op 781) 


v_mul_lo_u32 v14, v47, 6 // (op 781) 

v_lshrrev_b32 v46, 3, v14 // (op 781) 

v_and_b32 v14, -4, v1 // (op 834) 
v_sub_u32 v47, v1, v14 // (op 834) 

v_and_b32 v14, 4294967292, v1 // (op 834) 
v_add_u32 v48, v14, v47 // (op 834) 


v_and_b32 v14, -4, v1 // (op 887) 
v_sub_u32 v47, v1, v14 // (op 887) 

v_and_b32 v14, 4294967292, v1 // (op 887) 
v_add_u32 v49, v14, v47 // (op 887) 


v_and_b32 v14, -4, v1 // (op 974) 
v_sub_u32 v47, v1, v14 // (op 974) 

v_lshlrev_b32 v14, 5, v47 // (op 974) 

v_lshrrev_b32 v47, 2, v1 // (op 974) 
v_lshl_add_u32 v50, v47, 7, v14 // (op 974) 


v_mul_lo_u32 v14, v50, 6 // (op 974) 

v_lshrrev_b32 v47, 3, v14 // (op 974) 

v_and_b32 v14, -4, v1 // (op 998) 
v_sub_u32 v50, v1, v14 // (op 998) 

v_lshlrev_b32 v14, 5, v50 // (op 998) 

v_lshrrev_b32 v50, 2, v1 // (op 998) 
v_lshl_add_u32 v51, v50, 7, v14 // (op 998) 


v_mul_lo_u32 v14, v51, 6 // (op 998) 

v_lshrrev_b32 v50, 3, v14 // (op 998) 

v_and_b32 v14, -4, v1 // (op 1022) 
v_sub_u32 v51, v1, v14 // (op 1022) 

v_and_b32 v14, 4294967292, v1 // (op 1022) 
v_add_u32 v52, v14, v51 // (op 1022) 


v_and_b32 v14, -4, v1 // (op 1046) 
v_sub_u32 v51, v1, v14 // (op 1046) 

v_and_b32 v14, 4294967292, v1 // (op 1046) 
v_add_u32 v53, v14, v51 // (op 1046) 


s_mov_b32 s0, 0 // (op 61) call()
s_ashr_i64 s[8:9], s[34:35], 7 // (op 60) 
s_mov_b32 s1, s8 // (op 60) convert

s_and_b32 s8, s1, 4294967294 // (op 60) 

s_cmp_lt_u32 s0, s8 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_60 if false)
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_60:


s_add_u32 s0, s0, 2 // (op 62) 
v_mov_b32 v14, v17 // (op 10) 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 1095) 
buffer_load_dwordx4 v[22:25], v14, s[56:59], 0 offen offset:192 // (op 10) Load value



buffer_load_dwordx2 v[26:27], v14, s[56:59], 0 offen offset:208 // (op 10) Load value
ds_read_b128 v[54:57], v3 offset:1600 // (op 72) Load local data 


ds_read_b64 v[58:59], v3 offset:1616 // (op 72) Load local data 
v_mov_b32 v51, v20 // (op 25) 

s_barrier  // (op 1108) 
buffer_load_ubyte v18, v51, s[64:67], 0 offen offset:8 // (op 25) Load value





v_mov_b32 v60, v13 // (op 4) 

ds_read_b128 v[62:65], v4 // (op 77) Load local data 
ds_read_b64 v[66:67], v4 offset:16 // (op 77) Load local data 
buffer_load_dwordx4 v[28:31], v60, s[52:55], 0 offen offset:192 // (op 4) Load value
buffer_load_dwordx2 v[32:33], v60, s[52:55], 0 offen offset:208 // (op 4) Load value
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1122) 





ds_read_u8 v61, v0 offset:3136 // (op 82) Load local data 
v_mov_b32 v68, v15 // (op 17) 

s_barrier  // (op 1136) 
buffer_load_ubyte v21, v68, s[60:63], 0 offen offset:8 // (op 17) Load value



s_barrier  // (op 1159) 
ds_read_u8 v69, v6 offset:1536 // (op 87) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:59], v[62:67], a[0:3], v61, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 90) 
s_waitcnt vmcnt(11)
ds_write_b128 v50, v[34:37] offset:3200 // (op 257) Store local data 




s_waitcnt vmcnt(10)
ds_write_b64 v50, v[38:39] offset:3216 // (op 257) Store local data 
s_barrier  // (op 1171) 



s_waitcnt vmcnt(9)
ds_write_b8 v53, v16 offset:4736 // (op 259) Store local data 
s_barrier  // (op 1153) 



s_waitcnt vmcnt(8)
ds_write_b128 v47, v[40:43] offset:4800 // (op 247) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v47, v[44:45] offset:4816 // (op 247) Store local data 
s_barrier  // (op 1165) 



s_waitcnt vmcnt(6)
ds_write_b8 v52, v12 offset:6336 // (op 258) Store local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 1150) 
buffer_load_dwordx4 v[34:37], v14, s[56:59], 0 offen offset:288 // (op 260) Load value



buffer_load_dwordx2 v[38:39], v14, s[56:59], 0 offen offset:304 // (op 260) Load value
ds_read_b128 v[54:57], v7 offset:4800 // (op 249) Load local data 

ds_read_b64 v[58:59], v7 offset:4816 // (op 249) Load local data 



v_add_u32 v17, 192, v17 // (op 554) 
s_barrier  // (op 1156) 
s_barrier  // (op 1230) 



buffer_load_ubyte v16, v51, s[64:67], 0 offen offset:12 // (op 262) Load value
ds_read_b128 v[62:65], v8 offset:3200 // (op 251) Load local data 

ds_read_b64 v[66:67], v8 offset:3216 // (op 251) Load local data 



v_add_u32 v20, 8, v20 // (op 682) 
s_barrier  // (op 1236) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1162) 
buffer_load_dwordx4 v[40:43], v60, s[52:55], 0 offen offset:288 // (op 246) Load value



buffer_load_dwordx2 v[44:45], v60, s[52:55], 0 offen offset:304 // (op 246) Load value
ds_read_u8 v14, v5 offset:6336 // (op 253) Load local data 




s_barrier  // (op 1168) 
s_barrier  // (op 1242) 



v_add_u32 v13, 192, v13 // (op 490) 
ds_read_u8 v51, v10 offset:4736 // (op 255) Load local data 
buffer_load_ubyte v12, v68, s[60:63], 0 offen offset:12 // (op 261) Load value

s_barrier  // (op 1248) 
s_barrier  // (op 1111) 
v_add_u32 v15, 8, v15 // (op 618) 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:59], v[62:67], a[0:3], v14, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 256) 
s_waitcnt vmcnt(11)
ds_write_b128 v46, v[22:25] // (op 13) Store local data 




s_waitcnt vmcnt(10)
ds_write_b64 v46, v[26:27] offset:16 // (op 13) Store local data 
s_barrier  // (op 1139) 



s_waitcnt vmcnt(9)
ds_write_b8 v49, v18 offset:1536 // (op 30) Store local data 
s_barrier  // (op 1098) 



s_waitcnt vmcnt(8)
ds_write_b128 v19, v[28:31] offset:1600 // (op 6) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v19, v[32:33] offset:1616 // (op 6) Store local data 
s_barrier  // (op 1125) 



s_waitcnt vmcnt(6)
ds_write_b8 v48, v21 offset:3136 // (op 21) Store local data 
s_ashr_i64 s[8:9], s[34:35], 7 // (op 60) 
s_mov_b32 s1, s8 // (op 60) convert

s_and_b32 s8, s1, 4294967294 // (op 60) 

s_cmp_lt_u32 s0, s8 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_60 if true)
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_60:

















s_mul_hi_i32 s0, s16, s12 // (op 453) 
v_lshrrev_b32 v13, 2, v1 // (op 453) 
s_add_i32 s1, s0, s16 // (op 453) 

v_and_b32 v14, -4, v1 // (op 453) 
s_ashr_i32 s0, s1, 31 // (op 453) 
v_sub_u32 v15, v1, v14 // (op 453) 

s_cmp_eq_i32 s12, 0 // (op 453) 
s_mov_b32 s8, scc // (op 453) 
v_lshlrev_b32 v14, 5, v15 // (op 453) 

s_cmp_eq_i32 s8, 0 // (op 453) 
s_cselect_b32 s9, 0, -1 // (op 453) 

s_and_b32 s8, s13, 31 // (op 453) 
s_lshl_b32 s18, 1, s8 // (op 453) 
s_add_i32 s19, s18, s9 // (op 453) 


s_and_b32 s9, s0, s19 // (op 453) 


s_add_i32 s0, s1, s9 // (op 453) 


s_ashr_i32 s1, s0, s8 // (op 453) 


s_lshl_b32 s0, s13, 24 // (op 453) 
s_ashr_i32 s8, s0, 31 // (op 453) 

s_xor_b32 s0, s1, s8 // (op 453) 

s_sub_i32 s1, s0, s8 // (op 453) 


s_mul_hi_i32 s0, s1, s14 // (op 453) 
s_add_i32 s8, s0, s1 // (op 453) 

s_ashr_i32 s0, s8, 31 // (op 453) 
s_cmp_eq_i32 s14, 0 // (op 453) 
s_mov_b32 s9, scc // (op 453) 
s_cmp_eq_i32 s9, 0 // (op 453) 
s_cselect_b32 s18, 0, -1 // (op 453) 

s_and_b32 s9, s15, 31 // (op 453) 
s_lshl_b32 s19, 1, s9 // (op 453) 
s_add_i32 s49, s19, s18 // (op 453) 


s_and_b32 s18, s0, s49 // (op 453) 


s_add_i32 s0, s8, s18 // (op 453) 


s_ashr_i32 s8, s0, s9 // (op 453) 


s_lshl_b32 s0, s15, 24 // (op 453) 
s_ashr_i32 s9, s0, 31 // (op 453) 

s_xor_b32 s0, s8, s9 // (op 453) 

s_sub_i32 s8, s0, s9 // (op 453) 


s_mul_i32 s0, s1, s45 // (op 453) 

s_sub_u32 s1, s16, s0 // (op 453) 

s_mul_i32 s0, s8, s45 // (op 453) 
s_add_u32 s0, s0, s1 // (op 453) 


s_mul_hi_i32 s1, s16, s11 // (op 453) 
s_add_i32 s8, s1, s16 // (op 453) 

s_ashr_i32 s1, s8, 31 // (op 453) 
s_cmp_eq_i32 s11, 0 // (op 453) 
s_mov_b32 s9, scc // (op 453) 
s_cmp_eq_i32 s9, 0 // (op 453) 
s_cselect_b32 s18, 0, -1 // (op 453) 

s_and_b32 s9, s46, 31 // (op 453) 
s_lshl_b32 s19, 1, s9 // (op 453) 
s_add_i32 s49, s19, s18 // (op 453) 


s_and_b32 s18, s1, s49 // (op 453) 


s_add_i32 s1, s8, s18 // (op 453) 


s_ashr_i32 s8, s1, s9 // (op 453) 


s_lshl_b32 s1, s46, 24 // (op 453) 
s_ashr_i32 s9, s1, 31 // (op 453) 

s_xor_b32 s1, s8, s9 // (op 453) 

s_sub_i32 s8, s1, s9 // (op 453) 


s_add_u32 s18, s4, 16 // (op 453) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 453) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 453) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 453) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 453) 

s_mul_i32 s1, s45, s18 // (op 453) 

s_mul_i32 s9, s10, s1 // (op 453) 

s_mul_i32 s1, s8, s9 // (op 453) 

s_sub_u32 s9, s16, s1 // (op 453) 

s_mov_b32 s1, s9 // (op 453) 
s_mul_hi_i32 s18, s1, s44 // (op 453) 
s_add_i32 s19, s18, s1 // (op 453) 


s_ashr_i32 s1, s19, 31 // (op 453) 
s_cmp_eq_i32 s44, 0 // (op 453) 
s_mov_b32 s18, scc // (op 453) 
s_cmp_eq_i32 s18, 0 // (op 453) 
s_cselect_b32 s49, 0, -1 // (op 453) 

s_and_b32 s18, s48, 31 // (op 453) 
s_lshl_b32 s50, 1, s18 // (op 453) 
s_add_i32 s51, s50, s49 // (op 453) 


s_and_b32 s49, s1, s51 // (op 453) 


s_add_i32 s1, s19, s49 // (op 453) 


s_ashr_i32 s19, s1, s18 // (op 453) 


s_lshl_b32 s1, s48, 24 // (op 453) 
s_ashr_i32 s18, s1, 31 // (op 453) 

s_xor_b32 s1, s19, s18 // (op 453) 

s_sub_i32 s19, s1, s18 // (op 453) 


s_mul_i32 s1, s47, s45 // (op 453) 
s_add_u32 s50, s6, 16 // (op 453) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 453) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 453) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 453) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 453) 

s_sub_i32 s18, s50, s1 // (op 453) 

s_mul_i32 s49, s19, s18 // (op 453) 


s_sub_u32 s18, s9, s49 // (op 453) 


s_add_u32 s9, s18, s1 // (op 453) 


s_cmp_eq_i32 s8, 0 // (op 453) 
s_mov_b32 s1, scc // (op 453) 

s_cmp_eq_i32 s1, 0 // (op 453) 
s_cselect_b32 s8, s9, s0 // (op 453) 



v_lshl_add_u32 v15, s8, 4, v13 // (op 453) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 453) 
s_lshr_b32 s8, s0, 1 // (op 453) 

v_lshl_add_u32 v13, s8, 8, v14 // (op 453) 


v_mov_b32 v33, 0 // (op 453) convert
v_mov_b32 v32, v13 // (op 453) convert

v_mul_lo_u32 v31, v15, s3 // (op 453) most significant: low of low * high
v_mul_hi_u32 v13, v15, s2 // (op 453) most significant: high of low * low
v_mul_lo_u32 v30, v15, s2 // (op 453) least significant: low of low * low
v_add_u32 v31, v31, v13 // (op 453) most significant: sum

v_add_co_u32 v30, s[0:1], v30, v32 // (op 453) least significant half
v_addc_co_u32 v31, s[0:1], v31, v33, s[0:1] // (op 453) most significant half



v_mul_lo_u32 v15, v31, 6 // (op 453) most significant: low of high * low
v_mul_hi_u32 v13, v30, 6 // (op 453) most significant: high of low * low
v_mul_lo_u32 v14, v30, 6 // (op 453) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 453) most significant: sum


v_lshrrev_b64 v[32:33], 3, v[14:15] // (op 453) 

v_mov_b32 v13, v32 // (op 453) convert

s_mul_i32 s1, s7, 6 // (op 455) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 455) least significant
s_mul_hi_u32 s8, s6, 6 // (op 455) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 455) most significant: sum

s_mul_i32 s9, s2, s1 // (op 455) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 455) most significant: low of high * low
s_mul_i32 s8, s2, s0 // (op 455) least significant
s_mul_hi_u32 s19, s2, s0 // (op 455) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 455) most significant: sum
s_add_u32 s9, s9, s19 // (op 455) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 455) 

s_mov_b32 s52, s20 // (op 455) reg expression
s_mov_b32 s53, s21 // (op 455) reg expression
s_mov_b32 s54, s0 // (op 455) reg expression
s_mov_b32 s55, 131072 // (op 455) call()

s_mul_hi_i32 s0, s16, s12 // (op 520) 
v_lshrrev_b32 v14, 2, v1 // (op 520) 
s_add_i32 s1, s0, s16 // (op 520) 

v_and_b32 v15, -4, v1 // (op 520) 
s_ashr_i32 s0, s1, 31 // (op 520) 
v_sub_u32 v17, v1, v15 // (op 520) 

s_cmp_eq_i32 s12, 0 // (op 520) 
s_mov_b32 s8, scc // (op 520) 
v_lshlrev_b32 v15, 5, v17 // (op 520) 

s_cmp_eq_i32 s8, 0 // (op 520) 
s_cselect_b32 s9, 0, -1 // (op 520) 

s_and_b32 s8, s13, 31 // (op 520) 
s_lshl_b32 s18, 1, s8 // (op 520) 
s_add_i32 s19, s18, s9 // (op 520) 


s_and_b32 s9, s0, s19 // (op 520) 


s_add_i32 s0, s1, s9 // (op 520) 


s_ashr_i32 s1, s0, s8 // (op 520) 


s_lshl_b32 s0, s13, 24 // (op 520) 
s_ashr_i32 s8, s0, 31 // (op 520) 

s_xor_b32 s0, s1, s8 // (op 520) 

s_sub_i32 s1, s0, s8 // (op 520) 


s_mul_hi_i32 s0, s1, s14 // (op 520) 
s_add_i32 s8, s0, s1 // (op 520) 

s_ashr_i32 s0, s8, 31 // (op 520) 
s_cmp_eq_i32 s14, 0 // (op 520) 
s_mov_b32 s9, scc // (op 520) 
s_cmp_eq_i32 s9, 0 // (op 520) 
s_cselect_b32 s18, 0, -1 // (op 520) 

s_and_b32 s9, s15, 31 // (op 520) 
s_lshl_b32 s19, 1, s9 // (op 520) 
s_add_i32 s49, s19, s18 // (op 520) 


s_and_b32 s18, s0, s49 // (op 520) 


s_add_i32 s0, s8, s18 // (op 520) 


s_ashr_i32 s8, s0, s9 // (op 520) 


s_lshl_b32 s0, s15, 24 // (op 520) 
s_ashr_i32 s9, s0, 31 // (op 520) 

s_xor_b32 s0, s8, s9 // (op 520) 

s_sub_i32 s8, s0, s9 // (op 520) 


s_add_u32 s18, s4, 16 // (op 520) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 520) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 520) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 520) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 520) 

s_mul_i32 s0, s8, s18 // (op 520) 

s_sub_i32 s8, s1, s0 // (op 520) 


s_mul_hi_i32 s9, s16, s11 // (op 520) 
s_add_i32 s0, s9, s16 // (op 520) 

s_ashr_i32 s1, s0, 31 // (op 520) 
s_cmp_eq_i32 s11, 0 // (op 520) 
s_mov_b32 s9, scc // (op 520) 
s_cmp_eq_i32 s9, 0 // (op 520) 
s_cselect_b32 s49, 0, -1 // (op 520) 

s_and_b32 s9, s46, 31 // (op 520) 
s_lshl_b32 s50, 1, s9 // (op 520) 
s_add_i32 s51, s50, s49 // (op 520) 


s_and_b32 s49, s1, s51 // (op 520) 


s_add_i32 s1, s0, s49 // (op 520) 


s_ashr_i32 s0, s1, s9 // (op 520) 


s_lshl_b32 s1, s46, 24 // (op 520) 
s_ashr_i32 s9, s1, 31 // (op 520) 

s_xor_b32 s1, s0, s9 // (op 520) 

s_sub_i32 s0, s1, s9 // (op 520) 


s_mul_i32 s1, s45, s18 // (op 520) 

s_mul_i32 s9, s10, s1 // (op 520) 

s_mul_i32 s1, s0, s9 // (op 520) 

s_sub_u32 s9, s16, s1 // (op 520) 

s_mul_hi_i32 s1, s9, s44 // (op 520) 
s_add_i32 s18, s1, s9 // (op 520) 


s_ashr_i32 s1, s18, 31 // (op 520) 
s_cmp_eq_i32 s44, 0 // (op 520) 
s_mov_b32 s9, scc // (op 520) 
s_cmp_eq_i32 s9, 0 // (op 520) 
s_cselect_b32 s19, 0, -1 // (op 520) 

s_and_b32 s9, s48, 31 // (op 520) 
s_lshl_b32 s49, 1, s9 // (op 520) 
s_add_i32 s50, s49, s19 // (op 520) 


s_and_b32 s19, s1, s50 // (op 520) 


s_add_i32 s1, s18, s19 // (op 520) 


s_ashr_i32 s18, s1, s9 // (op 520) 


s_lshl_b32 s1, s48, 24 // (op 520) 
s_ashr_i32 s9, s1, 31 // (op 520) 

s_xor_b32 s1, s18, s9 // (op 520) 

s_sub_i32 s18, s1, s9 // (op 520) 


s_cmp_eq_i32 s0, 0 // (op 520) 
s_mov_b32 s1, scc // (op 520) 

s_cmp_eq_i32 s1, 0 // (op 520) 
s_cselect_b32 s0, s18, s8 // (op 520) 



v_lshl_add_u32 v17, s0, 4, v14 // (op 520) 


v_mul_lo_u32 v19, v17, s23 // (op 520) most significant: low of low * high
v_mul_hi_u32 v14, v17, s22 // (op 520) most significant: high of low * low
v_mul_lo_u32 v18, v17, s22 // (op 520) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 520) most significant: sum


s_ashr_i64 s[0:1], s[34:35], 7 // (op 520) 
s_lshr_b32 s8, s0, 1 // (op 520) 

v_lshl_add_u32 v14, s8, 8, v15 // (op 520) 


v_mov_b32 v21, 0 // (op 520) convert
v_mov_b32 v20, v14 // (op 520) convert

v_add_co_u32 v14, s[0:1], v20, v18 // (op 520) least significant half
v_addc_co_u32 v15, s[0:1], v21, v19, s[0:1] // (op 520) most significant half



v_mul_lo_u32 v33, v15, 6 // (op 520) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 520) most significant: high of low * low
v_mul_lo_u32 v32, v14, 6 // (op 520) least significant: low of low * low
v_add_u32 v33, v33, v17 // (op 520) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[32:33] // (op 520) 

v_mov_b32 v17, v14 // (op 520) convert

s_mul_i32 s1, s5, 6 // (op 522) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 522) least significant
s_mul_hi_u32 s8, s4, 6 // (op 522) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 522) most significant: sum

s_mul_i32 s9, s22, s1 // (op 522) most significant: low of low * high
s_mul_i32 s18, s23, s0 // (op 522) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 522) least significant
s_mul_hi_u32 s19, s22, s0 // (op 522) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 522) most significant: sum
s_add_u32 s9, s9, s19 // (op 522) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 522) 

s_mov_b32 s56, s24 // (op 522) reg expression
s_mov_b32 s57, s25 // (op 522) reg expression
s_mov_b32 s58, s0 // (op 522) reg expression
s_mov_b32 s59, 131072 // (op 522) call()

s_mul_hi_i32 s0, s16, s12 // (op 584) 
v_lshrrev_b32 v14, 2, v1 // (op 584) 
s_add_i32 s1, s0, s16 // (op 584) 

v_and_b32 v15, -4, v1 // (op 584) 
s_ashr_i32 s0, s1, 31 // (op 584) 
v_sub_u32 v18, v1, v15 // (op 584) 

s_cmp_eq_i32 s12, 0 // (op 584) 
s_mov_b32 s8, scc // (op 584) 
s_cmp_eq_i32 s8, 0 // (op 584) 
s_cselect_b32 s9, 0, -1 // (op 584) 

s_and_b32 s8, s13, 31 // (op 584) 
s_lshl_b32 s18, 1, s8 // (op 584) 
s_add_i32 s19, s18, s9 // (op 584) 


s_and_b32 s9, s0, s19 // (op 584) 


s_add_i32 s0, s1, s9 // (op 584) 


s_ashr_i32 s1, s0, s8 // (op 584) 


s_lshl_b32 s0, s13, 24 // (op 584) 
s_ashr_i32 s8, s0, 31 // (op 584) 

s_xor_b32 s0, s1, s8 // (op 584) 

s_sub_i32 s1, s0, s8 // (op 584) 


s_mul_hi_i32 s0, s1, s14 // (op 584) 
s_add_i32 s8, s0, s1 // (op 584) 

s_ashr_i32 s0, s8, 31 // (op 584) 
s_cmp_eq_i32 s14, 0 // (op 584) 
s_mov_b32 s9, scc // (op 584) 
s_cmp_eq_i32 s9, 0 // (op 584) 
s_cselect_b32 s18, 0, -1 // (op 584) 

s_and_b32 s9, s15, 31 // (op 584) 
s_lshl_b32 s19, 1, s9 // (op 584) 
s_add_i32 s49, s19, s18 // (op 584) 


s_and_b32 s18, s0, s49 // (op 584) 


s_add_i32 s0, s8, s18 // (op 584) 


s_ashr_i32 s8, s0, s9 // (op 584) 


s_lshl_b32 s0, s15, 24 // (op 584) 
s_ashr_i32 s9, s0, 31 // (op 584) 

s_xor_b32 s0, s8, s9 // (op 584) 

s_sub_i32 s8, s0, s9 // (op 584) 


s_mul_i32 s0, s1, s45 // (op 584) 

s_sub_u32 s1, s16, s0 // (op 584) 

s_mul_i32 s0, s8, s45 // (op 584) 
s_add_u32 s0, s0, s1 // (op 584) 


s_mul_hi_i32 s1, s16, s11 // (op 584) 
s_add_i32 s8, s1, s16 // (op 584) 

s_ashr_i32 s1, s8, 31 // (op 584) 
s_cmp_eq_i32 s11, 0 // (op 584) 
s_mov_b32 s9, scc // (op 584) 
s_cmp_eq_i32 s9, 0 // (op 584) 
s_cselect_b32 s18, 0, -1 // (op 584) 

s_and_b32 s9, s46, 31 // (op 584) 
s_lshl_b32 s19, 1, s9 // (op 584) 
s_add_i32 s49, s19, s18 // (op 584) 


s_and_b32 s18, s1, s49 // (op 584) 


s_add_i32 s1, s8, s18 // (op 584) 


s_ashr_i32 s8, s1, s9 // (op 584) 


s_lshl_b32 s1, s46, 24 // (op 584) 
s_ashr_i32 s9, s1, 31 // (op 584) 

s_xor_b32 s1, s8, s9 // (op 584) 

s_sub_i32 s8, s1, s9 // (op 584) 


s_add_u32 s18, s4, 16 // (op 584) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 584) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 584) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 584) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 584) 

s_mul_i32 s1, s45, s18 // (op 584) 

s_mul_i32 s9, s10, s1 // (op 584) 

s_mul_i32 s1, s8, s9 // (op 584) 

s_sub_u32 s9, s16, s1 // (op 584) 

s_mov_b32 s1, s9 // (op 584) 
s_mul_hi_i32 s18, s1, s44 // (op 584) 
s_add_i32 s19, s18, s1 // (op 584) 


s_ashr_i32 s1, s19, 31 // (op 584) 
s_cmp_eq_i32 s44, 0 // (op 584) 
s_mov_b32 s18, scc // (op 584) 
s_cmp_eq_i32 s18, 0 // (op 584) 
s_cselect_b32 s49, 0, -1 // (op 584) 

s_and_b32 s18, s48, 31 // (op 584) 
s_lshl_b32 s50, 1, s18 // (op 584) 
s_add_i32 s51, s50, s49 // (op 584) 


s_and_b32 s49, s1, s51 // (op 584) 


s_add_i32 s1, s19, s49 // (op 584) 


s_ashr_i32 s19, s1, s18 // (op 584) 


s_lshl_b32 s1, s48, 24 // (op 584) 
s_ashr_i32 s18, s1, 31 // (op 584) 

s_xor_b32 s1, s19, s18 // (op 584) 

s_sub_i32 s19, s1, s18 // (op 584) 


s_mul_i32 s1, s47, s45 // (op 584) 
s_add_u32 s50, s6, 16 // (op 584) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 584) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 584) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 584) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 584) 

s_sub_i32 s18, s50, s1 // (op 584) 

s_mul_i32 s49, s19, s18 // (op 584) 


s_sub_u32 s18, s9, s49 // (op 584) 


s_add_u32 s9, s18, s1 // (op 584) 


s_cmp_eq_i32 s8, 0 // (op 584) 
s_mov_b32 s1, scc // (op 584) 

s_cmp_eq_i32 s1, 0 // (op 584) 
s_cselect_b32 s8, s9, s0 // (op 584) 



v_lshl_add_u32 v15, s8, 4, v14 // (op 584) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 584) 
s_lshr_b32 s8, s0, 1 // (op 584) 

v_lshl_add_u32 v14, s8, 3, v18 // (op 584) 


v_mul_lo_u32 v18, v15, s26 // (op 584) 
v_add_u32 v18, v18, v14 // (op 584) 


s_mul_i32 s1, s26, s7 // (op 586) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 586) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 586) least significant
s_mul_hi_u32 s9, s26, s6 // (op 586) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 586) most significant: sum
s_add_u32 s1, s1, s9 // (op 586) most significant: sum


s_mov_b32 s60, s28 // (op 586) reg expression
s_mov_b32 s61, s29 // (op 586) reg expression
s_mov_b32 s62, s0 // (op 586) reg expression
s_mov_b32 s63, 131072 // (op 586) call()

s_mul_hi_i32 s0, s16, s12 // (op 648) 
v_lshrrev_b32 v14, 2, v1 // (op 648) 
s_add_i32 s1, s0, s16 // (op 648) 

v_and_b32 v15, -4, v1 // (op 648) 
s_ashr_i32 s0, s1, 31 // (op 648) 
v_sub_u32 v19, v1, v15 // (op 648) 

s_cmp_eq_i32 s12, 0 // (op 648) 
s_mov_b32 s8, scc // (op 648) 
s_cmp_eq_i32 s8, 0 // (op 648) 
s_cselect_b32 s9, 0, -1 // (op 648) 

s_and_b32 s8, s13, 31 // (op 648) 
s_lshl_b32 s18, 1, s8 // (op 648) 
s_add_i32 s19, s18, s9 // (op 648) 


s_and_b32 s9, s0, s19 // (op 648) 


s_add_i32 s0, s1, s9 // (op 648) 


s_ashr_i32 s1, s0, s8 // (op 648) 


s_lshl_b32 s0, s13, 24 // (op 648) 
s_ashr_i32 s8, s0, 31 // (op 648) 

s_xor_b32 s0, s1, s8 // (op 648) 

s_sub_i32 s1, s0, s8 // (op 648) 


s_mul_hi_i32 s0, s1, s14 // (op 648) 
s_add_i32 s8, s0, s1 // (op 648) 

s_ashr_i32 s0, s8, 31 // (op 648) 
s_cmp_eq_i32 s14, 0 // (op 648) 
s_mov_b32 s9, scc // (op 648) 
s_cmp_eq_i32 s9, 0 // (op 648) 
s_cselect_b32 s18, 0, -1 // (op 648) 

s_and_b32 s9, s15, 31 // (op 648) 
s_lshl_b32 s19, 1, s9 // (op 648) 
s_add_i32 s49, s19, s18 // (op 648) 


s_and_b32 s18, s0, s49 // (op 648) 


s_add_i32 s0, s8, s18 // (op 648) 


s_ashr_i32 s8, s0, s9 // (op 648) 


s_lshl_b32 s0, s15, 24 // (op 648) 
s_ashr_i32 s9, s0, 31 // (op 648) 

s_xor_b32 s0, s8, s9 // (op 648) 

s_sub_i32 s8, s0, s9 // (op 648) 


s_add_u32 s18, s4, 16 // (op 648) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 648) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 648) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 648) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 648) 

s_mul_i32 s0, s8, s18 // (op 648) 

s_sub_i32 s8, s1, s0 // (op 648) 


s_mul_hi_i32 s9, s16, s11 // (op 648) 
s_add_i32 s0, s9, s16 // (op 648) 

s_ashr_i32 s1, s0, 31 // (op 648) 
s_cmp_eq_i32 s11, 0 // (op 648) 
s_mov_b32 s9, scc // (op 648) 
s_cmp_eq_i32 s9, 0 // (op 648) 
s_cselect_b32 s49, 0, -1 // (op 648) 

s_and_b32 s9, s46, 31 // (op 648) 
s_lshl_b32 s50, 1, s9 // (op 648) 
s_add_i32 s51, s50, s49 // (op 648) 


s_and_b32 s49, s1, s51 // (op 648) 


s_add_i32 s1, s0, s49 // (op 648) 


s_ashr_i32 s0, s1, s9 // (op 648) 


s_lshl_b32 s1, s46, 24 // (op 648) 
s_ashr_i32 s9, s1, 31 // (op 648) 

s_xor_b32 s1, s0, s9 // (op 648) 

s_sub_i32 s0, s1, s9 // (op 648) 


s_mul_i32 s1, s45, s18 // (op 648) 

s_mul_i32 s9, s10, s1 // (op 648) 

s_mul_i32 s1, s0, s9 // (op 648) 

s_sub_u32 s9, s16, s1 // (op 648) 

s_mul_hi_i32 s1, s9, s44 // (op 648) 
s_add_i32 s18, s1, s9 // (op 648) 


s_ashr_i32 s1, s18, 31 // (op 648) 
s_cmp_eq_i32 s44, 0 // (op 648) 
s_mov_b32 s9, scc // (op 648) 
s_cmp_eq_i32 s9, 0 // (op 648) 
s_cselect_b32 s19, 0, -1 // (op 648) 

s_and_b32 s9, s48, 31 // (op 648) 
s_lshl_b32 s49, 1, s9 // (op 648) 
s_add_i32 s50, s49, s19 // (op 648) 


s_and_b32 s19, s1, s50 // (op 648) 


s_add_i32 s1, s18, s19 // (op 648) 


s_ashr_i32 s18, s1, s9 // (op 648) 


s_lshl_b32 s1, s48, 24 // (op 648) 
s_ashr_i32 s9, s1, 31 // (op 648) 

s_xor_b32 s1, s18, s9 // (op 648) 

s_sub_i32 s18, s1, s9 // (op 648) 


s_cmp_eq_i32 s0, 0 // (op 648) 
s_mov_b32 s1, scc // (op 648) 

s_cmp_eq_i32 s1, 0 // (op 648) 
s_cselect_b32 s0, s18, s8 // (op 648) 



v_lshl_add_u32 v15, s0, 4, v14 // (op 648) 


v_mul_lo_u32 v14, v15, s30 // (op 648) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 648) 
s_lshr_b32 s8, s0, 1 // (op 648) 

v_lshl_add_u32 v15, s8, 3, v19 // (op 648) 


v_add_u32 v19, v15, v14 // (op 648) 


s_mul_i32 s1, s30, s5 // (op 650) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 650) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 650) least significant
s_mul_hi_u32 s9, s30, s4 // (op 650) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 650) most significant: sum
s_add_u32 s1, s1, s9 // (op 650) most significant: sum


s_mov_b32 s64, s32 // (op 650) reg expression
s_mov_b32 s65, s33 // (op 650) reg expression
s_mov_b32 s66, s0 // (op 650) reg expression
s_mov_b32 s67, 131072 // (op 650) call()

v_and_b32 v14, -4, v1 // (op 741) 
v_sub_u32 v15, v1, v14 // (op 741) 

v_lshlrev_b32 v14, 5, v15 // (op 741) 

v_lshrrev_b32 v15, 2, v1 // (op 741) 
v_lshl_add_u32 v20, v15, 7, v14 // (op 741) 


v_mul_lo_u32 v14, v20, 6 // (op 741) 

v_lshrrev_b32 v15, 3, v14 // (op 741) 

v_and_b32 v14, -4, v1 // (op 794) 
v_sub_u32 v20, v1, v14 // (op 794) 

v_lshlrev_b32 v14, 5, v20 // (op 794) 

v_lshrrev_b32 v20, 2, v1 // (op 794) 
v_lshl_add_u32 v21, v20, 7, v14 // (op 794) 


v_mul_lo_u32 v14, v21, 6 // (op 794) 

v_lshrrev_b32 v20, 3, v14 // (op 794) 

v_and_b32 v14, -4, v1 // (op 847) 
v_sub_u32 v21, v1, v14 // (op 847) 

v_and_b32 v14, 4294967292, v1 // (op 847) 
v_add_u32 v22, v14, v21 // (op 847) 


v_and_b32 v14, -4, v1 // (op 900) 
v_sub_u32 v21, v1, v14 // (op 900) 

v_and_b32 v14, 4294967292, v1 // (op 900) 
v_add_u32 v23, v14, v21 // (op 900) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 176) 
s_and_b32 s8, s0, 4294967294 // (op 176) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 175) 
s_mov_b32 s9, s0 // (op 175) convert

s_cmp_lt_u32 s8, s9 // (op 175) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_175 // (op 175) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_175 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_175:


s_add_u32 s8, s8, 1 // (op 177) 
v_mov_b32 v12, v13 // (op 180) 

buffer_load_dwordx4 v[24:27], v12, s[52:55], 0 offen // (op 180) Load value
buffer_load_dwordx2 v[28:29], v12, s[52:55], 0 offen offset:16 // (op 180) Load value

v_add_u32 v13, 96, v13 // (op 443) 


s_barrier  // (op 1101) 
v_mov_b32 v12, v17 // (op 194) 




buffer_load_dwordx4 v[30:33], v12, s[56:59], 0 offen // (op 194) Load value
s_waitcnt vmcnt(2)
ds_write_b128 v15, v[24:27] offset:1600 // (op 181) Store local data 
buffer_load_dwordx2 v[34:35], v12, s[56:59], 0 offen offset:16 // (op 194) Load value

v_add_u32 v17, 96, v17 // (op 512) 
s_waitcnt vmcnt(2)
ds_write_b64 v15, v[28:29] offset:1616 // (op 181) Store local data 



v_mov_b32 v12, v18 // (op 195) 

s_barrier  // (op 1114) 
buffer_load_ubyte v14, v12, s[60:63], 0 offen offset: 0 // (op 195) Load value




v_add_u32 v18, 4, v18 // (op 576) 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[30:33] // (op 191) Store local data 


v_mov_b32 v12, v19 // (op 196) 

buffer_load_ubyte v16, v12, s[64:67], 0 offen offset: 0 // (op 196) Load value

v_add_u32 v19, 4, v19 // (op 640) 
s_waitcnt vmcnt(2)
ds_write_b64 v20, v[34:35] offset:16 // (op 191) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1104) 



s_barrier  // (op 1128) 
ds_read_b128 v[24:27], v3 offset:1600 // (op 183) Load local data 



ds_read_b64 v[28:29], v3 offset:1616 // (op 183) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v22, v14 offset:3136 // (op 192) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1118) 



s_barrier  // (op 1142) 
ds_read_b128 v[30:33], v4 // (op 185) Load local data 



ds_read_b64 v[34:35], v4 offset:16 // (op 185) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v16 offset:1536 // (op 193) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1132) 



ds_read_u8 v12, v0 offset:3136 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1146) 



ds_read_u8 v14, v6 offset:1536 // (op 189) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[30:35], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 190) 




s_ashr_i64 s[0:1], s[34:35], 7 // (op 175) 
s_mov_b32 s9, s0 // (op 175) convert

s_cmp_lt_u32 s8, s9 // (op 175) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_175 // (op 175) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_175 if true)
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_175:






















s_barrier  // (op 1224) 








v_cmp_eq_f32 vcc, 0.00000, v9 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_135
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v11, v4 // (op 50) 
v_mul_f32 v13, v11, v5 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 

v_lshrrev_b32 v4, 16, v12 // (op 56) 
v_lshrrev_b32 v5, 16, v13 // (op 56) 
v_lshrrev_b32 v6, 16, v14 // (op 56) 
v_lshrrev_b32 v7, 16, v15 // (op 56) 

s_mul_hi_i32 s0, s16, s12 // (op 942) 
v_lshrrev_b32 v0, 6, v1 // (op 942) 
s_add_i32 s1, s0, s16 // (op 942) 

v_and_b32 v3, 63, v1 // (op 942) 
s_ashr_i32 s0, s1, 31 // (op 942) 
v_lshrrev_b32 v8, 4, v3 // (op 942) 
s_cmp_eq_i32 s12, 0 // (op 942) 
s_mov_b32 s2, scc // (op 942) 
v_lshlrev_b32 v10, 2, v8 // (op 942) 

s_cmp_eq_i32 s2, 0 // (op 942) 
s_cselect_b32 s3, 0, -1 // (op 942) 

v_lshl_add_u32 v8, v0, 4, v10 // (op 942) 


s_and_b32 s2, s13, 31 // (op 942) 
v_lshrrev_b32 v0, 2, v3 // (op 942) 

s_lshl_b32 s8, 1, s2 // (op 942) 
v_and_b32 v3, 3, v0 // (op 942) 

s_add_i32 s9, s8, s3 // (op 942) 


v_and_b32 v0, 3, v1 // (op 942) 
s_and_b32 s3, s0, s9 // (op 942) 


v_lshl_add_u32 v10, v3, 2, v0 // (op 942) 


s_add_i32 s0, s1, s3 // (op 942) 


s_ashr_i32 s1, s0, s2 // (op 942) 


s_lshl_b32 s0, s13, 24 // (op 942) 
s_ashr_i32 s2, s0, 31 // (op 942) 

s_xor_b32 s0, s1, s2 // (op 942) 

s_sub_i32 s1, s0, s2 // (op 942) 


s_mul_hi_i32 s0, s1, s14 // (op 942) 
s_add_i32 s2, s0, s1 // (op 942) 

s_ashr_i32 s0, s2, 31 // (op 942) 
s_cmp_eq_i32 s14, 0 // (op 942) 
s_mov_b32 s3, scc // (op 942) 
s_cmp_eq_i32 s3, 0 // (op 942) 
s_cselect_b32 s8, 0, -1 // (op 942) 

s_and_b32 s3, s15, 31 // (op 942) 
s_lshl_b32 s9, 1, s3 // (op 942) 
s_add_i32 s18, s9, s8 // (op 942) 


s_and_b32 s8, s0, s18 // (op 942) 


s_add_i32 s0, s2, s8 // (op 942) 


s_ashr_i32 s2, s0, s3 // (op 942) 


s_lshl_b32 s0, s15, 24 // (op 942) 
s_ashr_i32 s3, s0, 31 // (op 942) 

s_xor_b32 s0, s2, s3 // (op 942) 

s_sub_i32 s2, s0, s3 // (op 942) 


s_mul_i32 s0, s1, s45 // (op 942) 
s_sub_u32 s3, s16, s0 // (op 942) 

s_mul_i32 s0, s2, s45 // (op 942) 
s_add_u32 s0, s0, s3 // (op 942) 

s_mul_hi_i32 s3, s16, s11 // (op 942) 
s_add_i32 s8, s3, s16 // (op 942) 

s_ashr_i32 s3, s8, 31 // (op 942) 
s_cmp_eq_i32 s11, 0 // (op 942) 
s_mov_b32 s9, scc // (op 942) 
s_cmp_eq_i32 s9, 0 // (op 942) 
s_cselect_b32 s18, 0, -1 // (op 942) 

s_and_b32 s9, s46, 31 // (op 942) 
s_lshl_b32 s19, 1, s9 // (op 942) 
s_add_i32 s20, s19, s18 // (op 942) 


s_and_b32 s18, s3, s20 // (op 942) 


s_add_i32 s3, s8, s18 // (op 942) 


s_ashr_i32 s8, s3, s9 // (op 942) 


s_lshl_b32 s3, s46, 24 // (op 942) 
s_ashr_i32 s9, s3, 31 // (op 942) 

s_xor_b32 s3, s8, s9 // (op 942) 

s_sub_i32 s8, s3, s9 // (op 942) 


s_add_u32 s18, s4, 16 // (op 942) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 942) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 942) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 942) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 942) 

s_mul_i32 s3, s45, s18 // (op 942) 
s_mul_i32 s9, s10, s3 // (op 942) 

s_mul_i32 s3, s8, s9 // (op 942) 

s_sub_u32 s9, s16, s3 // (op 942) 

s_mov_b32 s3, s9 // (op 942) 
s_mul_hi_i32 s20, s3, s44 // (op 942) 
s_add_i32 s21, s20, s3 // (op 942) 


s_ashr_i32 s3, s21, 31 // (op 942) 
s_cmp_eq_i32 s44, 0 // (op 942) 
s_mov_b32 s20, scc // (op 942) 
s_cmp_eq_i32 s20, 0 // (op 942) 
s_cselect_b32 s22, 0, -1 // (op 942) 

s_and_b32 s20, s48, 31 // (op 942) 
s_lshl_b32 s23, 1, s20 // (op 942) 
s_add_i32 s24, s23, s22 // (op 942) 


s_and_b32 s22, s3, s24 // (op 942) 


s_add_i32 s3, s21, s22 // (op 942) 


s_ashr_i32 s21, s3, s20 // (op 942) 


s_lshl_b32 s3, s48, 24 // (op 942) 
s_ashr_i32 s20, s3, 31 // (op 942) 

s_xor_b32 s3, s21, s20 // (op 942) 

s_sub_i32 s21, s3, s20 // (op 942) 


s_mul_i32 s3, s47, s45 // (op 942) 
s_add_u32 s22, s6, 16 // (op 942) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 942) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 942) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 942) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 942) 

s_sub_i32 s20, s22, s3 // (op 942) 

s_mul_i32 s22, s21, s20 // (op 942) 

s_sub_u32 s20, s9, s22 // (op 942) 


s_add_u32 s9, s20, s3 // (op 942) 


s_cmp_eq_i32 s8, 0 // (op 942) 
s_mov_b32 s3, scc // (op 942) 

s_cmp_eq_i32 s3, 0 // (op 942) 
s_cselect_b32 s8, s9, s0 // (op 942) 


s_mul_i32 s0, s2, s18 // (op 942) 


v_lshl_add_u32 v0, s8, 4, v8 // (op 942) 


s_sub_i32 s2, s1, s0 // (op 942) 


s_cmp_eq_i32 s3, 0 // (op 942) 
s_cselect_b32 s0, s21, s2 // (op 942) 



v_lshl_add_u32 v3, s0, 4, v10 // (op 942) 


v_mul_lo_u32 v8, v3, s36 // (op 942) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 942) 


s_mul_i32 s9, s36, s5 // (op 944) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 944) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 944) least significant
s_mul_hi_u32 s1, s36, s4 // (op 944) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 944) most significant: sum
s_add_u32 s9, s9, s1 // (op 944) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 944) 

s_mov_b32 s32, s38 // (op 944) reg expression
s_mov_b32 s33, s39 // (op 944) reg expression
s_mov_b32 s34, s0 // (op 944) reg expression
s_mov_b32 s35, 131072 // (op 944) call()


v_pack_b32_f16 v12, v4, v5 // (op 58) 
v_pack_b32_f16 v13, v6, v7 // (op 58) 


buffer_store_dwordx2 v[12:13], v3, s[32:35], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_135
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_135:
s_mul_hi_i32 s0, s16, s12 // (op 711) 
v_lshrrev_b32 v0, 6, v1 // (op 711) 
s_add_i32 s1, s0, s16 // (op 711) 

v_and_b32 v10, 63, v1 // (op 711) 
s_ashr_i32 s0, s1, 31 // (op 711) 
v_lshrrev_b32 v3, 4, v10 // (op 711) 
s_cmp_eq_i32 s12, 0 // (op 711) 
s_mov_b32 s2, scc // (op 711) 
v_lshlrev_b32 v4, 2, v3 // (op 711) 

s_cmp_eq_i32 s2, 0 // (op 711) 
s_cselect_b32 s3, 0, -1 // (op 711) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 711) 


s_and_b32 s2, s13, 31 // (op 711) 
v_lshrrev_b32 v0, 2, v10 // (op 711) 

s_lshl_b32 s8, 1, s2 // (op 711) 
v_and_b32 v10, 3, v0 // (op 711) 

s_add_i32 s9, s8, s3 // (op 711) 


v_and_b32 v0, 3, v1 // (op 711) 
s_and_b32 s3, s0, s9 // (op 711) 


v_lshl_add_u32 v4, v10, 2, v0 // (op 711) 


s_add_i32 s0, s1, s3 // (op 711) 


s_ashr_i32 s1, s0, s2 // (op 711) 


s_lshl_b32 s0, s13, 24 // (op 711) 
s_ashr_i32 s2, s0, 31 // (op 711) 

s_xor_b32 s0, s1, s2 // (op 711) 

s_sub_i32 s1, s0, s2 // (op 711) 


s_mul_hi_i32 s0, s1, s14 // (op 711) 
s_add_i32 s2, s0, s1 // (op 711) 

s_ashr_i32 s0, s2, 31 // (op 711) 
s_cmp_eq_i32 s14, 0 // (op 711) 
s_mov_b32 s3, scc // (op 711) 
s_cmp_eq_i32 s3, 0 // (op 711) 
s_cselect_b32 s8, 0, -1 // (op 711) 

s_and_b32 s3, s15, 31 // (op 711) 
s_lshl_b32 s9, 1, s3 // (op 711) 
s_add_i32 s18, s9, s8 // (op 711) 


s_and_b32 s8, s0, s18 // (op 711) 


s_add_i32 s0, s2, s8 // (op 711) 


s_ashr_i32 s2, s0, s3 // (op 711) 


s_lshl_b32 s0, s15, 24 // (op 711) 
s_ashr_i32 s3, s0, 31 // (op 711) 

s_xor_b32 s0, s2, s3 // (op 711) 

s_sub_i32 s2, s0, s3 // (op 711) 


s_mul_i32 s0, s1, s45 // (op 711) 
s_sub_u32 s3, s16, s0 // (op 711) 

s_mul_i32 s0, s2, s45 // (op 711) 
s_add_u32 s0, s0, s3 // (op 711) 

s_mul_hi_i32 s3, s16, s11 // (op 711) 
s_add_i32 s8, s3, s16 // (op 711) 

s_ashr_i32 s3, s8, 31 // (op 711) 
s_cmp_eq_i32 s11, 0 // (op 711) 
s_mov_b32 s9, scc // (op 711) 
s_cmp_eq_i32 s9, 0 // (op 711) 
s_cselect_b32 s18, 0, -1 // (op 711) 

s_and_b32 s9, s46, 31 // (op 711) 
s_lshl_b32 s19, 1, s9 // (op 711) 
s_add_i32 s20, s19, s18 // (op 711) 


s_and_b32 s18, s3, s20 // (op 711) 


s_add_i32 s3, s8, s18 // (op 711) 


s_ashr_i32 s8, s3, s9 // (op 711) 


s_lshl_b32 s3, s46, 24 // (op 711) 
s_ashr_i32 s9, s3, 31 // (op 711) 

s_xor_b32 s3, s8, s9 // (op 711) 

s_sub_i32 s8, s3, s9 // (op 711) 


s_add_u32 s18, s4, 16 // (op 711) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 711) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 711) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 711) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 711) 

s_mul_i32 s3, s45, s18 // (op 711) 
s_mul_i32 s9, s10, s3 // (op 711) 

s_mul_i32 s3, s8, s9 // (op 711) 

s_sub_u32 s9, s16, s3 // (op 711) 

s_mov_b32 s3, s9 // (op 711) 
s_mul_hi_i32 s20, s3, s44 // (op 711) 
s_add_i32 s21, s20, s3 // (op 711) 


s_ashr_i32 s3, s21, 31 // (op 711) 
s_cmp_eq_i32 s44, 0 // (op 711) 
s_mov_b32 s20, scc // (op 711) 
s_cmp_eq_i32 s20, 0 // (op 711) 
s_cselect_b32 s22, 0, -1 // (op 711) 

s_and_b32 s20, s48, 31 // (op 711) 
s_lshl_b32 s23, 1, s20 // (op 711) 
s_add_i32 s24, s23, s22 // (op 711) 


s_and_b32 s22, s3, s24 // (op 711) 


s_add_i32 s3, s21, s22 // (op 711) 


s_ashr_i32 s21, s3, s20 // (op 711) 


s_lshl_b32 s3, s48, 24 // (op 711) 
s_ashr_i32 s20, s3, 31 // (op 711) 

s_xor_b32 s3, s21, s20 // (op 711) 

s_sub_i32 s21, s3, s20 // (op 711) 


s_mul_i32 s3, s47, s45 // (op 711) 
s_add_u32 s22, s6, 16 // (op 711) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 711) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 711) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 711) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 711) 

s_sub_i32 s20, s22, s3 // (op 711) 

s_mul_i32 s22, s21, s20 // (op 711) 

s_sub_u32 s20, s9, s22 // (op 711) 


s_add_u32 s9, s20, s3 // (op 711) 


s_cmp_eq_i32 s8, 0 // (op 711) 
s_mov_b32 s3, scc // (op 711) 

s_cmp_eq_i32 s3, 0 // (op 711) 
s_cselect_b32 s8, s9, s0 // (op 711) 


s_mul_i32 s0, s2, s18 // (op 711) 


v_lshl_add_u32 v0, s8, 4, v3 // (op 711) 


s_sub_i32 s2, s1, s0 // (op 711) 


s_cmp_eq_i32 s3, 0 // (op 711) 
s_cselect_b32 s0, s21, s2 // (op 711) 



v_lshl_add_u32 v3, s0, 4, v4 // (op 711) 


v_mul_lo_u32 v10, v3, s40 // (op 711) 

v_add_lshl_u32 v3, v0, v10, 1 // (op 711) 


s_mul_i32 s9, s40, s5 // (op 713) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 713) most significant: low of high * low
s_mul_i32 s8, s40, s4 // (op 713) least significant
s_mul_hi_u32 s1, s40, s4 // (op 713) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 713) most significant: sum
s_add_u32 s9, s9, s1 // (op 713) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 713) 

s_mov_b32 s28, s42 // (op 713) reg expression
s_mov_b32 s29, s43 // (op 713) reg expression
s_mov_b32 s30, s0 // (op 713) reg expression
s_mov_b32 s31, 131072 // (op 713) call()





buffer_load_dwordx2 v[4:5], v3, s[28:31], 0 offen // (op 148) Load value



v_accvgpr_read v12, a0 // (op 144) 
v_accvgpr_read v13, a1 // (op 144) 
v_accvgpr_read v14, a2 // (op 144) 
v_accvgpr_read v15, a3 // (op 144) 
v_mul_f32 v16, v11, v12 // (op 144) 
v_mul_f32 v17, v11, v13 // (op 144) 
v_mul_f32 v18, v11, v14 // (op 144) 
v_mul_f32 v19, v11, v15 // (op 144) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 149) 
v_lshrrev_b32 v7, 16, v4 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v12, v9, v6 // (op 149) 
v_mul_f32 v13, v9, v7 // (op 149) 
v_and_b32 v6, 65535, v5 // (op 149) 
v_lshrrev_b32 v7, 16, v5 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v14, v9, v6 // (op 149) 
v_mul_f32 v15, v9, v7 // (op 149) 


v_add_f32 v4, v12, v16 // (op 145) 
v_add_f32 v5, v13, v17 // (op 145) 
v_add_f32 v6, v14, v18 // (op 145) 
v_add_f32 v7, v15, v19 // (op 145) 


v_lshrrev_b32 v12, 16, v4 // (op 146) 
v_lshrrev_b32 v13, 16, v5 // (op 146) 
v_lshrrev_b32 v14, 16, v6 // (op 146) 
v_lshrrev_b32 v15, 16, v7 // (op 146) 

s_mul_hi_i32 s0, s16, s12 // (op 959) 
v_lshrrev_b32 v0, 6, v1 // (op 959) 
s_add_i32 s1, s0, s16 // (op 959) 

v_and_b32 v10, 63, v1 // (op 959) 
s_ashr_i32 s0, s1, 31 // (op 959) 
v_lshrrev_b32 v3, 4, v10 // (op 959) 
s_cmp_eq_i32 s12, 0 // (op 959) 
s_mov_b32 s2, scc // (op 959) 
v_lshlrev_b32 v4, 2, v3 // (op 959) 

s_cmp_eq_i32 s2, 0 // (op 959) 
s_cselect_b32 s3, 0, -1 // (op 959) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 959) 


s_and_b32 s2, s13, 31 // (op 959) 
v_lshrrev_b32 v0, 2, v10 // (op 959) 

s_lshl_b32 s8, 1, s2 // (op 959) 
v_and_b32 v10, 3, v0 // (op 959) 

s_add_i32 s9, s8, s3 // (op 959) 


v_and_b32 v0, 3, v1 // (op 959) 
s_and_b32 s3, s0, s9 // (op 959) 


v_lshl_add_u32 v4, v10, 2, v0 // (op 959) 


s_add_i32 s0, s1, s3 // (op 959) 


s_ashr_i32 s1, s0, s2 // (op 959) 


s_lshl_b32 s0, s13, 24 // (op 959) 
s_ashr_i32 s2, s0, 31 // (op 959) 

s_xor_b32 s0, s1, s2 // (op 959) 

s_sub_i32 s1, s0, s2 // (op 959) 


s_mul_hi_i32 s0, s1, s14 // (op 959) 
s_add_i32 s2, s0, s1 // (op 959) 

s_ashr_i32 s0, s2, 31 // (op 959) 
s_cmp_eq_i32 s14, 0 // (op 959) 
s_mov_b32 s3, scc // (op 959) 
s_cmp_eq_i32 s3, 0 // (op 959) 
s_cselect_b32 s8, 0, -1 // (op 959) 

s_and_b32 s3, s15, 31 // (op 959) 
s_lshl_b32 s9, 1, s3 // (op 959) 
s_add_i32 s18, s9, s8 // (op 959) 


s_and_b32 s8, s0, s18 // (op 959) 


s_add_i32 s0, s2, s8 // (op 959) 


s_ashr_i32 s2, s0, s3 // (op 959) 


s_lshl_b32 s0, s15, 24 // (op 959) 
s_ashr_i32 s3, s0, 31 // (op 959) 

s_xor_b32 s0, s2, s3 // (op 959) 

s_sub_i32 s2, s0, s3 // (op 959) 


s_mul_i32 s0, s1, s45 // (op 959) 
s_sub_u32 s3, s16, s0 // (op 959) 

s_mul_i32 s0, s2, s45 // (op 959) 
s_add_u32 s0, s0, s3 // (op 959) 

s_mul_hi_i32 s3, s16, s11 // (op 959) 
s_add_i32 s8, s3, s16 // (op 959) 

s_ashr_i32 s3, s8, 31 // (op 959) 
s_cmp_eq_i32 s11, 0 // (op 959) 
s_mov_b32 s9, scc // (op 959) 
s_cmp_eq_i32 s9, 0 // (op 959) 
s_cselect_b32 s18, 0, -1 // (op 959) 

s_and_b32 s9, s46, 31 // (op 959) 
s_lshl_b32 s19, 1, s9 // (op 959) 
s_add_i32 s20, s19, s18 // (op 959) 


s_and_b32 s18, s3, s20 // (op 959) 


s_add_i32 s3, s8, s18 // (op 959) 


s_ashr_i32 s8, s3, s9 // (op 959) 


s_lshl_b32 s3, s46, 24 // (op 959) 
s_ashr_i32 s9, s3, 31 // (op 959) 

s_xor_b32 s3, s8, s9 // (op 959) 

s_sub_i32 s8, s3, s9 // (op 959) 


s_add_u32 s18, s4, 16 // (op 959) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 959) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 959) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 959) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 959) 

s_mul_i32 s3, s45, s18 // (op 959) 
s_mul_i32 s9, s10, s3 // (op 959) 

s_mul_i32 s3, s8, s9 // (op 959) 

s_sub_u32 s9, s16, s3 // (op 959) 

s_mov_b32 s3, s9 // (op 959) 
s_mul_hi_i32 s20, s3, s44 // (op 959) 
s_add_i32 s21, s20, s3 // (op 959) 


s_ashr_i32 s3, s21, 31 // (op 959) 
s_cmp_eq_i32 s44, 0 // (op 959) 
s_mov_b32 s20, scc // (op 959) 
s_cmp_eq_i32 s20, 0 // (op 959) 
s_cselect_b32 s22, 0, -1 // (op 959) 

s_and_b32 s20, s48, 31 // (op 959) 
s_lshl_b32 s23, 1, s20 // (op 959) 
s_add_i32 s24, s23, s22 // (op 959) 


s_and_b32 s22, s3, s24 // (op 959) 


s_add_i32 s3, s21, s22 // (op 959) 


s_ashr_i32 s21, s3, s20 // (op 959) 


s_lshl_b32 s3, s48, 24 // (op 959) 
s_ashr_i32 s20, s3, 31 // (op 959) 

s_xor_b32 s3, s21, s20 // (op 959) 

s_sub_i32 s21, s3, s20 // (op 959) 


s_mul_i32 s3, s47, s45 // (op 959) 
s_add_u32 s22, s6, 16 // (op 959) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 959) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 959) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 959) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 959) 

s_sub_i32 s20, s22, s3 // (op 959) 

s_mul_i32 s22, s21, s20 // (op 959) 

s_sub_u32 s20, s9, s22 // (op 959) 


s_add_u32 s9, s20, s3 // (op 959) 


s_cmp_eq_i32 s8, 0 // (op 959) 
s_mov_b32 s3, scc // (op 959) 

s_cmp_eq_i32 s3, 0 // (op 959) 
s_cselect_b32 s8, s9, s0 // (op 959) 


s_mul_i32 s0, s2, s18 // (op 959) 


v_lshl_add_u32 v0, s8, 4, v3 // (op 959) 


s_sub_i32 s2, s1, s0 // (op 959) 


s_cmp_eq_i32 s3, 0 // (op 959) 
s_cselect_b32 s0, s21, s2 // (op 959) 



v_lshl_add_u32 v3, s0, 4, v4 // (op 959) 


v_mul_lo_u32 v10, v3, s36 // (op 959) 

v_add_lshl_u32 v3, v0, v10, 1 // (op 959) 


s_mul_i32 s9, s36, s5 // (op 961) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 961) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 961) least significant
s_mul_hi_u32 s1, s36, s4 // (op 961) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 961) most significant: sum
s_add_u32 s9, s9, s1 // (op 961) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 961) 

s_mov_b32 s32, s38 // (op 961) reg expression
s_mov_b32 s33, s39 // (op 961) reg expression
s_mov_b32 s34, s0 // (op 961) reg expression
s_mov_b32 s35, 131072 // (op 961) call()


v_pack_b32_f16 v4, v12, v13 // (op 147) 
v_pack_b32_f16 v5, v14, v15 // (op 147) 


buffer_store_dwordx2 v[4:5], v3, s[32:35], 0 offen // (op 147) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_135:
















s_endpgm  // End of RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.LRR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_, .LRR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end-RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: F6
    DestDataType: b
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
  - .name: RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 6400
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 70
    .vgpr_count: 70
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
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

