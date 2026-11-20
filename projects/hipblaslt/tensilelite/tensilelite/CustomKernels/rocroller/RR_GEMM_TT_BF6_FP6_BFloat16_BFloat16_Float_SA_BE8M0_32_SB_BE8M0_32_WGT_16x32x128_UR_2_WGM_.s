// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_,@function
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 747) 
v_and_b32 v3, 15, v1 // (op 747) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 747) 


v_and_b32 v0, 63, v1 // (op 747) 
v_lshrrev_b32 v3, 4, v0 // (op 747) 

v_lshlrev_b32 v0, 5, v3 // (op 747) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 747) 


v_mul_lo_u32 v0, v3, 6 // (op 747) 

v_lshrrev_b32 v3, 3, v0 // (op 747) 

v_and_b32 v0, 63, v1 // (op 800) 
v_and_b32 v4, 48, v1 // (op 800) 
v_sub_u32 v5, v0, v4 // (op 800) 

v_lshrrev_b32 v4, 4, v0 // (op 800) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 800) 


v_lshrrev_b32 v4, 6, v1 // (op 800) 
v_and_b32 v5, 1, v4 // (op 800) 

v_lshlrev_b32 v4, 4, v5 // (op 800) 

v_lshl_add_u32 v5, v0, 5, v4 // (op 800) 


v_mul_lo_u32 v0, v5, 6 // (op 800) 
v_lshrrev_b32 v4, 3, v0 // (op 800) 

v_lshrrev_b32 v0, 4, v5 // (op 800) 

v_lshlrev_b32 v5, 2, v0 // (op 800) 

v_add_u32 v0, v4, v5 // (op 800) 


v_lshrrev_b32 v4, 6, v1 // (op 853) 
v_and_b32 v5, 1, v4 // (op 853) 

v_and_b32 v4, 15, v1 // (op 853) 
v_lshl_add_u32 v6, v5, 4, v4 // (op 853) 


v_and_b32 v4, 63, v1 // (op 853) 
v_lshrrev_b32 v5, 4, v4 // (op 853) 

v_lshl_add_u32 v4, v5, 5, v6 // (op 853) 


v_lshrrev_b32 v5, 7, v1 // (op 911) 
v_and_b32 v6, 15, v1 // (op 911) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 911) 


v_and_b32 v5, 63, v1 // (op 911) 
v_lshrrev_b32 v6, 4, v5 // (op 911) 

v_lshlrev_b32 v5, 5, v6 // (op 911) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 911) 


v_mul_lo_u32 v5, v6, 6 // (op 911) 

v_lshrrev_b32 v6, 3, v5 // (op 911) 

v_and_b32 v5, 63, v1 // (op 935) 
v_and_b32 v7, 48, v1 // (op 935) 
v_sub_u32 v8, v5, v7 // (op 935) 

v_lshrrev_b32 v7, 4, v5 // (op 935) 

v_lshl_add_u32 v5, v7, 5, v8 // (op 935) 


v_lshrrev_b32 v7, 6, v1 // (op 935) 
v_and_b32 v8, 1, v7 // (op 935) 

v_lshlrev_b32 v7, 4, v8 // (op 935) 

v_lshl_add_u32 v8, v5, 5, v7 // (op 935) 


v_mul_lo_u32 v5, v8, 6 // (op 935) 
v_lshrrev_b32 v7, 3, v5 // (op 935) 

v_lshrrev_b32 v5, 4, v8 // (op 935) 

v_lshlrev_b32 v8, 2, v5 // (op 935) 

v_add_u32 v5, v7, v8 // (op 935) 


v_lshrrev_b32 v7, 6, v1 // (op 959) 
v_and_b32 v8, 1, v7 // (op 959) 

v_and_b32 v7, 15, v1 // (op 959) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 959) 


v_and_b32 v7, 63, v1 // (op 959) 
v_lshrrev_b32 v8, 4, v7 // (op 959) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 959) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v9, s9 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
s_mul_hi_i32 s0, s16, s11 // (op 457) 
v_lshrrev_b32 v10, 3, v1 // (op 457) 
s_add_i32 s1, s0, s16 // (op 457) 

v_and_b32 v11, -8, v1 // (op 457) 
s_ashr_i32 s0, s1, 31 // (op 457) 
v_sub_u32 v12, v1, v11 // (op 457) 

s_cmp_eq_i32 s11, 0 // (op 457) 
s_mov_b32 s8, scc // (op 457) 
v_lshlrev_b32 v11, 4, v12 // (op 457) 

s_cmp_eq_i32 s8, 0 // (op 457) 
s_cselect_b32 s9, 0, -1 // (op 457) 

v_mov_b32 v13, 0 // (op 457) convert
v_mov_b32 v12, v11 // (op 457) convert

s_and_b32 s8, s12, 31 // (op 457) 
s_lshl_b32 s18, 1, s8 // (op 457) 
s_add_i32 s19, s18, s9 // (op 457) 


s_and_b32 s9, s0, s19 // (op 457) 


s_add_i32 s0, s1, s9 // (op 457) 


s_ashr_i32 s1, s0, s8 // (op 457) 


s_lshl_b32 s0, s12, 24 // (op 457) 
s_ashr_i32 s8, s0, 31 // (op 457) 

s_xor_b32 s0, s1, s8 // (op 457) 

s_sub_i32 s1, s0, s8 // (op 457) 


s_mul_hi_i32 s0, s1, s13 // (op 457) 
s_add_i32 s8, s0, s1 // (op 457) 

s_ashr_i32 s0, s8, 31 // (op 457) 
s_cmp_eq_i32 s13, 0 // (op 457) 
s_mov_b32 s9, scc // (op 457) 
s_cmp_eq_i32 s9, 0 // (op 457) 
s_cselect_b32 s18, 0, -1 // (op 457) 

s_and_b32 s9, s14, 31 // (op 457) 
s_lshl_b32 s19, 1, s9 // (op 457) 
s_add_i32 s51, s19, s18 // (op 457) 


s_and_b32 s18, s0, s51 // (op 457) 


s_add_i32 s0, s8, s18 // (op 457) 


s_ashr_i32 s8, s0, s9 // (op 457) 


s_lshl_b32 s0, s14, 24 // (op 457) 
s_ashr_i32 s9, s0, 31 // (op 457) 

s_xor_b32 s0, s8, s9 // (op 457) 

s_sub_i32 s8, s0, s9 // (op 457) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s46 // (op 457) 

s_sub_u32 s1, s16, s0 // (op 457) 

s_mul_i32 s0, s8, s46 // (op 457) 
s_add_u32 s0, s0, s1 // (op 457) 


s_mul_hi_i32 s1, s16, s10 // (op 457) 
s_add_i32 s8, s1, s16 // (op 457) 

s_ashr_i32 s1, s8, 31 // (op 457) 
s_cmp_eq_i32 s10, 0 // (op 457) 
s_mov_b32 s9, scc // (op 457) 
s_cmp_eq_i32 s9, 0 // (op 457) 
s_cselect_b32 s18, 0, -1 // (op 457) 

s_and_b32 s9, s49, 31 // (op 457) 
s_lshl_b32 s19, 1, s9 // (op 457) 
s_add_i32 s51, s19, s18 // (op 457) 


s_and_b32 s18, s1, s51 // (op 457) 


s_add_i32 s1, s8, s18 // (op 457) 


s_ashr_i32 s8, s1, s9 // (op 457) 


s_lshl_b32 s1, s49, 24 // (op 457) 
s_ashr_i32 s9, s1, 31 // (op 457) 

s_xor_b32 s1, s8, s9 // (op 457) 

s_sub_i32 s8, s1, s9 // (op 457) 


s_add_u32 s18, s2, 32 // (op 457) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 457) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 457) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 457) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 5 // (op 457) 

s_mul_i32 s1, s46, s18 // (op 457) 

s_mul_i32 s9, s50, s1 // (op 457) 

s_mul_i32 s1, s8, s9 // (op 457) 

s_sub_u32 s9, s16, s1 // (op 457) 

s_mov_b32 s1, s9 // (op 457) 
s_mul_hi_i32 s18, s1, s15 // (op 457) 
s_add_i32 s19, s18, s1 // (op 457) 


s_ashr_i32 s1, s19, 31 // (op 457) 
s_cmp_eq_i32 s15, 0 // (op 457) 
s_mov_b32 s18, scc // (op 457) 
s_cmp_eq_i32 s18, 0 // (op 457) 
s_cselect_b32 s51, 0, -1 // (op 457) 

s_and_b32 s18, s48, 31 // (op 457) 
s_lshl_b32 s52, 1, s18 // (op 457) 
s_add_i32 s53, s52, s51 // (op 457) 


s_and_b32 s51, s1, s53 // (op 457) 


s_add_i32 s1, s19, s51 // (op 457) 


s_ashr_i32 s19, s1, s18 // (op 457) 


s_lshl_b32 s1, s48, 24 // (op 457) 
s_ashr_i32 s18, s1, 31 // (op 457) 

s_xor_b32 s1, s19, s18 // (op 457) 

s_sub_i32 s19, s1, s18 // (op 457) 


s_mul_i32 s1, s47, s46 // (op 457) 
s_add_u32 s52, s6, 16 // (op 457) Start of Int64 add, locking SCC
s_addc_u32 s53, s7, 0 // (op 457) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 457) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 457) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 457) 

s_sub_i32 s18, s52, s1 // (op 457) 

s_mul_i32 s51, s19, s18 // (op 457) 


s_sub_u32 s18, s9, s51 // (op 457) 


s_add_u32 s9, s18, s1 // (op 457) 


s_cmp_eq_i32 s8, 0 // (op 457) 
s_mov_b32 s1, scc // (op 457) 

s_cmp_eq_i32 s1, 0 // (op 457) 
s_cselect_b32 s8, s9, s0 // (op 457) 



v_lshl_add_u32 v11, s8, 4, v10 // (op 457) 


v_mul_lo_u32 v15, v11, s5 // (op 457) most significant: low of low * high
v_mul_hi_u32 v10, v11, s4 // (op 457) most significant: high of low * low
v_mul_lo_u32 v14, v11, s4 // (op 457) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 457) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 457) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 457) most significant half



v_mul_lo_u32 v11, v15, 6 // (op 457) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 457) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 457) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 457) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 457) 

v_mov_b32 v10, v12 // (op 457) convert

s_mul_i32 s1, s7, 6 // (op 459) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 459) least significant
s_mul_hi_u32 s8, s6, 6 // (op 459) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 459) most significant: sum

s_mul_i32 s9, s4, s1 // (op 459) most significant: low of low * high
s_mul_i32 s18, s5, s0 // (op 459) most significant: low of high * low
s_mul_i32 s8, s4, s0 // (op 459) least significant
s_mul_hi_u32 s19, s4, s0 // (op 459) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 459) most significant: sum
s_add_u32 s9, s9, s19 // (op 459) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 459) 

s_mov_b32 s52, s20 // (op 459) reg expression
s_mov_b32 s53, s21 // (op 459) reg expression
s_mov_b32 s54, s0 // (op 459) reg expression
s_mov_b32 s55, 131072 // (op 459) call()

s_mul_hi_i32 s51, s16, s11 // (op 476) 
v_lshrrev_b32 v11, 3, v1 // (op 476) 
s_add_i32 s0, s51, s16 // (op 476) 

v_and_b32 v12, -8, v1 // (op 476) 
s_ashr_i32 s1, s0, 31 // (op 476) 
v_sub_u32 v13, v1, v12 // (op 476) 

s_cmp_eq_i32 s11, 0 // (op 476) 
s_mov_b32 s51, scc // (op 476) 
v_lshlrev_b32 v12, 4, v13 // (op 476) 

s_cmp_eq_i32 s51, 0 // (op 476) 
s_cselect_b32 s8, 0, -1 // (op 476) 

v_mov_b32 v15, 0 // (op 476) convert
v_mov_b32 v14, v12 // (op 476) convert

s_and_b32 s9, s12, 31 // (op 476) 
s_lshl_b32 s51, 1, s9 // (op 476) 
s_add_i32 s18, s51, s8 // (op 476) 


s_and_b32 s8, s1, s18 // (op 476) 


s_add_i32 s1, s0, s8 // (op 476) 


s_ashr_i32 s0, s1, s9 // (op 476) 


s_lshl_b32 s1, s12, 24 // (op 476) 
s_ashr_i32 s51, s1, 31 // (op 476) 

s_xor_b32 s1, s0, s51 // (op 476) 

s_sub_i32 s0, s1, s51 // (op 476) 


s_mul_hi_i32 s1, s0, s13 // (op 476) 
s_add_i32 s51, s1, s0 // (op 476) 

s_ashr_i32 s1, s51, 31 // (op 476) 
s_cmp_eq_i32 s13, 0 // (op 476) 
s_mov_b32 s8, scc // (op 476) 
s_cmp_eq_i32 s8, 0 // (op 476) 
s_cselect_b32 s9, 0, -1 // (op 476) 

s_and_b32 s8, s14, 31 // (op 476) 
s_lshl_b32 s18, 1, s8 // (op 476) 
s_add_i32 s19, s18, s9 // (op 476) 


s_and_b32 s9, s1, s19 // (op 476) 


s_add_i32 s1, s51, s9 // (op 476) 


s_ashr_i32 s9, s1, s8 // (op 476) 


s_lshl_b32 s1, s14, 24 // (op 476) 
s_ashr_i32 s8, s1, 31 // (op 476) 

s_xor_b32 s1, s9, s8 // (op 476) 

s_sub_i32 s9, s1, s8 // (op 476) 


s_mul_i32 s1, s0, s46 // (op 476) 

s_sub_u32 s0, s16, s1 // (op 476) 

s_mul_i32 s1, s9, s46 // (op 476) 
s_add_u32 s1, s1, s0 // (op 476) 


s_mul_hi_i32 s0, s16, s10 // (op 476) 
s_add_i32 s51, s0, s16 // (op 476) 

s_ashr_i32 s0, s51, 31 // (op 476) 
s_cmp_eq_i32 s10, 0 // (op 476) 
s_mov_b32 s8, scc // (op 476) 
s_cmp_eq_i32 s8, 0 // (op 476) 
s_cselect_b32 s9, 0, -1 // (op 476) 

s_and_b32 s8, s49, 31 // (op 476) 
s_lshl_b32 s18, 1, s8 // (op 476) 
s_add_i32 s19, s18, s9 // (op 476) 


s_and_b32 s9, s0, s19 // (op 476) 


s_add_i32 s0, s51, s9 // (op 476) 


s_ashr_i32 s9, s0, s8 // (op 476) 


s_lshl_b32 s0, s49, 24 // (op 476) 
s_ashr_i32 s8, s0, 31 // (op 476) 

s_xor_b32 s0, s9, s8 // (op 476) 

s_sub_i32 s9, s0, s8 // (op 476) 


s_add_u32 s18, s2, 32 // (op 476) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 476) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 476) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 476) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 476) 

s_mul_i32 s0, s46, s18 // (op 476) 

s_mul_i32 s8, s50, s0 // (op 476) 

s_mul_i32 s0, s9, s8 // (op 476) 

s_sub_u32 s8, s16, s0 // (op 476) 

s_mov_b32 s0, s8 // (op 476) 
s_mul_hi_i32 s51, s0, s15 // (op 476) 
s_add_i32 s18, s51, s0 // (op 476) 


s_ashr_i32 s0, s18, 31 // (op 476) 
s_cmp_eq_i32 s15, 0 // (op 476) 
s_mov_b32 s19, scc // (op 476) 
s_cmp_eq_i32 s19, 0 // (op 476) 
s_cselect_b32 s51, 0, -1 // (op 476) 

s_and_b32 s19, s48, 31 // (op 476) 
s_lshl_b32 s56, 1, s19 // (op 476) 
s_add_i32 s57, s56, s51 // (op 476) 


s_and_b32 s51, s0, s57 // (op 476) 


s_add_i32 s0, s18, s51 // (op 476) 


s_ashr_i32 s18, s0, s19 // (op 476) 


s_lshl_b32 s0, s48, 24 // (op 476) 
s_ashr_i32 s19, s0, 31 // (op 476) 

s_xor_b32 s0, s18, s19 // (op 476) 

s_sub_i32 s18, s0, s19 // (op 476) 


s_mul_i32 s0, s47, s46 // (op 476) 
s_add_u32 s56, s6, 16 // (op 476) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 476) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 476) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 476) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 476) 

s_sub_i32 s19, s56, s0 // (op 476) 

s_mul_i32 s51, s18, s19 // (op 476) 


s_sub_u32 s18, s8, s51 // (op 476) 


s_add_u32 s8, s18, s0 // (op 476) 


s_cmp_eq_i32 s9, 0 // (op 476) 
s_mov_b32 s0, scc // (op 476) 

s_cmp_eq_i32 s0, 0 // (op 476) 
s_cselect_b32 s9, s8, s1 // (op 476) 



v_lshl_add_u32 v12, s9, 4, v11 // (op 476) 


v_mul_lo_u32 v17, v12, s5 // (op 476) most significant: low of low * high
v_mul_hi_u32 v11, v12, s4 // (op 476) most significant: high of low * low
v_mul_lo_u32 v16, v12, s4 // (op 476) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 476) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 476) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 476) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 476) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 476) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 476) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 476) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 476) 

v_mov_b32 v11, v12 // (op 476) convert

s_mul_i32 s1, s7, 6 // (op 478) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 478) least significant
s_mul_hi_u32 s51, s6, 6 // (op 478) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 478) most significant: sum

s_mul_i32 s9, s4, s1 // (op 478) most significant: low of low * high
s_mul_i32 s51, s5, s0 // (op 478) most significant: low of high * low
s_mul_i32 s8, s4, s0 // (op 478) least significant
s_mul_hi_u32 s18, s4, s0 // (op 478) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 478) most significant: sum
s_add_u32 s9, s9, s18 // (op 478) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 478) 

s_mov_b32 s52, s20 // (op 478) reg expression
s_mov_b32 s53, s21 // (op 478) reg expression
s_mov_b32 s54, s0 // (op 478) reg expression
s_mov_b32 s55, 131072 // (op 478) call()

s_mul_hi_i32 s51, s16, s11 // (op 521) 
s_add_i32 s0, s51, s16 // (op 521) 

s_ashr_i32 s1, s0, 31 // (op 521) 
s_cmp_eq_i32 s11, 0 // (op 521) 
s_mov_b32 s51, scc // (op 521) 
s_cmp_eq_i32 s51, 0 // (op 521) 
s_cselect_b32 s8, 0, -1 // (op 521) 

s_and_b32 s9, s12, 31 // (op 521) 
s_lshl_b32 s51, 1, s9 // (op 521) 
s_add_i32 s18, s51, s8 // (op 521) 


s_and_b32 s8, s1, s18 // (op 521) 


s_add_i32 s1, s0, s8 // (op 521) 


s_ashr_i32 s0, s1, s9 // (op 521) 


s_lshl_b32 s1, s12, 24 // (op 521) 
s_ashr_i32 s51, s1, 31 // (op 521) 

s_xor_b32 s1, s0, s51 // (op 521) 

s_sub_i32 s0, s1, s51 // (op 521) 


s_mul_hi_i32 s1, s0, s13 // (op 521) 
s_add_i32 s51, s1, s0 // (op 521) 

s_ashr_i32 s1, s51, 31 // (op 521) 
s_cmp_eq_i32 s13, 0 // (op 521) 
s_mov_b32 s8, scc // (op 521) 
s_cmp_eq_i32 s8, 0 // (op 521) 
s_cselect_b32 s9, 0, -1 // (op 521) 

s_and_b32 s8, s14, 31 // (op 521) 
s_lshl_b32 s18, 1, s8 // (op 521) 
s_add_i32 s19, s18, s9 // (op 521) 


s_and_b32 s9, s1, s19 // (op 521) 


s_add_i32 s1, s51, s9 // (op 521) 


s_ashr_i32 s9, s1, s8 // (op 521) 


s_lshl_b32 s1, s14, 24 // (op 521) 
s_ashr_i32 s8, s1, 31 // (op 521) 

s_xor_b32 s1, s9, s8 // (op 521) 

s_sub_i32 s9, s1, s8 // (op 521) 


s_add_u32 s18, s2, 32 // (op 521) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 521) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 521) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 521) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 521) 

s_mul_i32 s1, s9, s18 // (op 521) 

s_sub_i32 s51, s0, s1 // (op 521) 


s_mul_hi_i32 s0, s16, s10 // (op 521) 
s_add_i32 s1, s0, s16 // (op 521) 

s_ashr_i32 s0, s1, 31 // (op 521) 
s_cmp_eq_i32 s10, 0 // (op 521) 
s_mov_b32 s8, scc // (op 521) 
s_cmp_eq_i32 s8, 0 // (op 521) 
s_cselect_b32 s9, 0, -1 // (op 521) 

s_and_b32 s8, s49, 31 // (op 521) 
s_lshl_b32 s56, 1, s8 // (op 521) 
s_add_i32 s57, s56, s9 // (op 521) 


s_and_b32 s9, s0, s57 // (op 521) 


s_add_i32 s0, s1, s9 // (op 521) 


s_ashr_i32 s1, s0, s8 // (op 521) 


s_lshl_b32 s0, s49, 24 // (op 521) 
s_ashr_i32 s8, s0, 31 // (op 521) 

s_xor_b32 s0, s1, s8 // (op 521) 

s_sub_i32 s1, s0, s8 // (op 521) 


s_mul_i32 s0, s46, s18 // (op 521) 

s_mul_i32 s8, s50, s0 // (op 521) 

s_mul_i32 s0, s1, s8 // (op 521) 

s_sub_u32 s8, s16, s0 // (op 521) 

s_mul_hi_i32 s0, s8, s15 // (op 521) 
s_add_i32 s9, s0, s8 // (op 521) 


s_ashr_i32 s0, s9, 31 // (op 521) 
s_cmp_eq_i32 s15, 0 // (op 521) 
s_mov_b32 s8, scc // (op 521) 
s_cmp_eq_i32 s8, 0 // (op 521) 
s_cselect_b32 s18, 0, -1 // (op 521) 

s_and_b32 s8, s48, 31 // (op 521) 
s_lshl_b32 s19, 1, s8 // (op 521) 
s_add_i32 s56, s19, s18 // (op 521) 


s_and_b32 s18, s0, s56 // (op 521) 


s_add_i32 s0, s9, s18 // (op 521) 


s_ashr_i32 s9, s0, s8 // (op 521) 


s_lshl_b32 s0, s48, 24 // (op 521) 
s_ashr_i32 s8, s0, 31 // (op 521) 

s_xor_b32 s0, s9, s8 // (op 521) 

s_sub_i32 s9, s0, s8 // (op 521) 


s_cmp_eq_i32 s1, 0 // (op 521) 
s_mov_b32 s0, scc // (op 521) 

s_cmp_eq_i32 s0, 0 // (op 521) 
s_cselect_b32 s1, s9, s51 // (op 521) 



s_lshl_b32 s0, s1, 5 // (op 521) 

s_mov_b32 s9, 0 // (op 521) convert
s_mov_b32 s8, s0 // (op 521) convert

v_mul_lo_u32 v13, v1, s23 // (op 521) most significant: low of low * high
v_mul_hi_u32 v14, v1, s22 // (op 521) most significant: high of low * low
v_mul_lo_u32 v12, v1, s22 // (op 521) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 521) most significant: sum

v_mov_b32 v14, s8 // (op 521) 
v_mov_b32 v15, s9 // (op 521) 
v_add_co_u32 v12, s[0:1], v12, v14 // (op 521) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 521) most significant half




v_mul_lo_u32 v15, v13, 6 // (op 521) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 521) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 521) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 521) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 521) 

v_mov_b32 v14, v12 // (op 521) convert

s_mul_i32 s1, s27, 6 // (op 523) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 523) least significant
s_mul_hi_u32 s51, s26, 6 // (op 523) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 523) most significant: sum

s_mul_i32 s9, s22, s1 // (op 523) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 523) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 523) least significant
s_mul_hi_u32 s18, s22, s0 // (op 523) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 523) most significant: sum
s_add_u32 s9, s9, s18 // (op 523) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 523) 

s_mov_b32 s56, s24 // (op 523) reg expression
s_mov_b32 s57, s25 // (op 523) reg expression
s_mov_b32 s58, s0 // (op 523) reg expression
s_mov_b32 s59, 131072 // (op 523) call()

s_mul_hi_i32 s51, s16, s11 // (op 540) 
s_add_i32 s0, s51, s16 // (op 540) 

s_ashr_i32 s1, s0, 31 // (op 540) 
s_cmp_eq_i32 s11, 0 // (op 540) 
s_mov_b32 s51, scc // (op 540) 
s_cmp_eq_i32 s51, 0 // (op 540) 
s_cselect_b32 s8, 0, -1 // (op 540) 

s_and_b32 s9, s12, 31 // (op 540) 
s_lshl_b32 s51, 1, s9 // (op 540) 
s_add_i32 s18, s51, s8 // (op 540) 


s_and_b32 s8, s1, s18 // (op 540) 


s_add_i32 s1, s0, s8 // (op 540) 


s_ashr_i32 s0, s1, s9 // (op 540) 


s_lshl_b32 s1, s12, 24 // (op 540) 
s_ashr_i32 s51, s1, 31 // (op 540) 

s_xor_b32 s1, s0, s51 // (op 540) 

s_sub_i32 s0, s1, s51 // (op 540) 


s_mul_hi_i32 s1, s0, s13 // (op 540) 
s_add_i32 s51, s1, s0 // (op 540) 

s_ashr_i32 s1, s51, 31 // (op 540) 
s_cmp_eq_i32 s13, 0 // (op 540) 
s_mov_b32 s8, scc // (op 540) 
s_cmp_eq_i32 s8, 0 // (op 540) 
s_cselect_b32 s9, 0, -1 // (op 540) 

s_and_b32 s8, s14, 31 // (op 540) 
s_lshl_b32 s18, 1, s8 // (op 540) 
s_add_i32 s19, s18, s9 // (op 540) 


s_and_b32 s9, s1, s19 // (op 540) 


s_add_i32 s1, s51, s9 // (op 540) 


s_ashr_i32 s9, s1, s8 // (op 540) 


s_lshl_b32 s1, s14, 24 // (op 540) 
s_ashr_i32 s8, s1, 31 // (op 540) 

s_xor_b32 s1, s9, s8 // (op 540) 

s_sub_i32 s9, s1, s8 // (op 540) 


s_add_u32 s18, s2, 32 // (op 540) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 540) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 540) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 540) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 540) 

s_mul_i32 s1, s9, s18 // (op 540) 

s_sub_i32 s51, s0, s1 // (op 540) 


s_mul_hi_i32 s0, s16, s10 // (op 540) 
s_add_i32 s1, s0, s16 // (op 540) 

s_ashr_i32 s0, s1, 31 // (op 540) 
s_cmp_eq_i32 s10, 0 // (op 540) 
s_mov_b32 s8, scc // (op 540) 
s_cmp_eq_i32 s8, 0 // (op 540) 
s_cselect_b32 s9, 0, -1 // (op 540) 

s_and_b32 s8, s49, 31 // (op 540) 
s_lshl_b32 s60, 1, s8 // (op 540) 
s_add_i32 s61, s60, s9 // (op 540) 


s_and_b32 s9, s0, s61 // (op 540) 


s_add_i32 s0, s1, s9 // (op 540) 


s_ashr_i32 s1, s0, s8 // (op 540) 


s_lshl_b32 s0, s49, 24 // (op 540) 
s_ashr_i32 s8, s0, 31 // (op 540) 

s_xor_b32 s0, s1, s8 // (op 540) 

s_sub_i32 s1, s0, s8 // (op 540) 


s_mul_i32 s0, s46, s18 // (op 540) 

s_mul_i32 s8, s50, s0 // (op 540) 

s_mul_i32 s0, s1, s8 // (op 540) 

s_sub_u32 s8, s16, s0 // (op 540) 

s_mul_hi_i32 s0, s8, s15 // (op 540) 
s_add_i32 s9, s0, s8 // (op 540) 


s_ashr_i32 s0, s9, 31 // (op 540) 
s_cmp_eq_i32 s15, 0 // (op 540) 
s_mov_b32 s8, scc // (op 540) 
s_cmp_eq_i32 s8, 0 // (op 540) 
s_cselect_b32 s18, 0, -1 // (op 540) 

s_and_b32 s8, s48, 31 // (op 540) 
s_lshl_b32 s19, 1, s8 // (op 540) 
s_add_i32 s60, s19, s18 // (op 540) 


s_and_b32 s18, s0, s60 // (op 540) 


s_add_i32 s0, s9, s18 // (op 540) 


s_ashr_i32 s9, s0, s8 // (op 540) 


s_lshl_b32 s0, s48, 24 // (op 540) 
s_ashr_i32 s8, s0, 31 // (op 540) 

s_xor_b32 s0, s9, s8 // (op 540) 

s_sub_i32 s9, s0, s8 // (op 540) 


s_cmp_eq_i32 s1, 0 // (op 540) 
s_mov_b32 s0, scc // (op 540) 

s_cmp_eq_i32 s0, 0 // (op 540) 
s_cselect_b32 s1, s9, s51 // (op 540) 



s_lshl_b32 s0, s1, 5 // (op 540) 

s_mov_b32 s9, 0 // (op 540) convert
s_mov_b32 s8, s0 // (op 540) convert

v_mul_lo_u32 v13, v1, s23 // (op 540) most significant: low of low * high
v_mul_hi_u32 v15, v1, s22 // (op 540) most significant: high of low * low
v_mul_lo_u32 v12, v1, s22 // (op 540) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 540) most significant: sum

v_mov_b32 v15, s8 // (op 540) 
v_mov_b32 v16, s9 // (op 540) 
v_add_co_u32 v12, s[0:1], v12, v15 // (op 540) least significant half
v_addc_co_u32 v13, s[0:1], v13, v16, s[0:1] // (op 540) most significant half




v_mul_lo_u32 v17, v13, 6 // (op 540) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 540) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 540) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 540) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 540) 

v_mov_b32 v15, v12 // (op 540) convert

s_mul_i32 s1, s27, 6 // (op 542) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 542) least significant
s_mul_hi_u32 s51, s26, 6 // (op 542) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 542) most significant: sum

s_mul_i32 s9, s22, s1 // (op 542) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 542) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 542) least significant
s_mul_hi_u32 s18, s22, s0 // (op 542) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 542) most significant: sum
s_add_u32 s9, s9, s18 // (op 542) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 542) 

s_mov_b32 s56, s24 // (op 542) reg expression
s_mov_b32 s57, s25 // (op 542) reg expression
s_mov_b32 s58, s0 // (op 542) reg expression
s_mov_b32 s59, 131072 // (op 542) call()

s_mul_hi_i32 s51, s16, s11 // (op 585) 
v_lshrrev_b32 v12, 7, v1 // (op 585) 
s_add_i32 s0, s51, s16 // (op 585) 

v_and_b32 v13, 15, v1 // (op 585) 
s_ashr_i32 s1, s0, 31 // (op 585) 
v_lshl_add_u32 v16, v12, 4, v13 // (op 585) 


s_cmp_eq_i32 s11, 0 // (op 585) 
s_mov_b32 s51, scc // (op 585) 
v_and_b32 v12, 63, v1 // (op 585) 
s_cmp_eq_i32 s51, 0 // (op 585) 
s_cselect_b32 s8, 0, -1 // (op 585) 

v_lshrrev_b32 v13, 4, v12 // (op 585) 

s_and_b32 s9, s12, 31 // (op 585) 
s_lshl_b32 s51, 1, s9 // (op 585) 
s_add_i32 s18, s51, s8 // (op 585) 


s_and_b32 s8, s1, s18 // (op 585) 


s_add_i32 s1, s0, s8 // (op 585) 


s_ashr_i32 s0, s1, s9 // (op 585) 


s_lshl_b32 s1, s12, 24 // (op 585) 
s_ashr_i32 s51, s1, 31 // (op 585) 

s_xor_b32 s1, s0, s51 // (op 585) 

s_sub_i32 s0, s1, s51 // (op 585) 


s_mul_hi_i32 s1, s0, s13 // (op 585) 
s_add_i32 s51, s1, s0 // (op 585) 

s_ashr_i32 s1, s51, 31 // (op 585) 
s_cmp_eq_i32 s13, 0 // (op 585) 
s_mov_b32 s8, scc // (op 585) 
s_cmp_eq_i32 s8, 0 // (op 585) 
s_cselect_b32 s9, 0, -1 // (op 585) 

s_and_b32 s8, s14, 31 // (op 585) 
s_lshl_b32 s18, 1, s8 // (op 585) 
s_add_i32 s19, s18, s9 // (op 585) 


s_and_b32 s9, s1, s19 // (op 585) 


s_add_i32 s1, s51, s9 // (op 585) 


s_ashr_i32 s9, s1, s8 // (op 585) 


s_lshl_b32 s1, s14, 24 // (op 585) 
s_ashr_i32 s8, s1, 31 // (op 585) 

s_xor_b32 s1, s9, s8 // (op 585) 

s_sub_i32 s9, s1, s8 // (op 585) 


s_mul_i32 s1, s0, s46 // (op 585) 

s_sub_u32 s0, s16, s1 // (op 585) 

s_mul_i32 s1, s9, s46 // (op 585) 
s_add_u32 s1, s1, s0 // (op 585) 


s_mul_hi_i32 s0, s16, s10 // (op 585) 
s_add_i32 s51, s0, s16 // (op 585) 

s_ashr_i32 s0, s51, 31 // (op 585) 
s_cmp_eq_i32 s10, 0 // (op 585) 
s_mov_b32 s8, scc // (op 585) 
s_cmp_eq_i32 s8, 0 // (op 585) 
s_cselect_b32 s9, 0, -1 // (op 585) 

s_and_b32 s8, s49, 31 // (op 585) 
s_lshl_b32 s18, 1, s8 // (op 585) 
s_add_i32 s19, s18, s9 // (op 585) 


s_and_b32 s9, s0, s19 // (op 585) 


s_add_i32 s0, s51, s9 // (op 585) 


s_ashr_i32 s9, s0, s8 // (op 585) 


s_lshl_b32 s0, s49, 24 // (op 585) 
s_ashr_i32 s8, s0, 31 // (op 585) 

s_xor_b32 s0, s9, s8 // (op 585) 

s_sub_i32 s9, s0, s8 // (op 585) 


s_add_u32 s18, s2, 32 // (op 585) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 585) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 585) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 585) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 585) 

s_mul_i32 s0, s46, s18 // (op 585) 

s_mul_i32 s8, s50, s0 // (op 585) 

s_mul_i32 s0, s9, s8 // (op 585) 

s_sub_u32 s8, s16, s0 // (op 585) 

s_mov_b32 s0, s8 // (op 585) 
s_mul_hi_i32 s51, s0, s15 // (op 585) 
s_add_i32 s18, s51, s0 // (op 585) 


s_ashr_i32 s0, s18, 31 // (op 585) 
s_cmp_eq_i32 s15, 0 // (op 585) 
s_mov_b32 s19, scc // (op 585) 
s_cmp_eq_i32 s19, 0 // (op 585) 
s_cselect_b32 s51, 0, -1 // (op 585) 

s_and_b32 s19, s48, 31 // (op 585) 
s_lshl_b32 s60, 1, s19 // (op 585) 
s_add_i32 s61, s60, s51 // (op 585) 


s_and_b32 s51, s0, s61 // (op 585) 


s_add_i32 s0, s18, s51 // (op 585) 


s_ashr_i32 s18, s0, s19 // (op 585) 


s_lshl_b32 s0, s48, 24 // (op 585) 
s_ashr_i32 s19, s0, 31 // (op 585) 

s_xor_b32 s0, s18, s19 // (op 585) 

s_sub_i32 s18, s0, s19 // (op 585) 


s_mul_i32 s0, s47, s46 // (op 585) 
s_add_u32 s60, s6, 16 // (op 585) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 585) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 585) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 585) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 585) 

s_sub_i32 s19, s60, s0 // (op 585) 

s_mul_i32 s51, s18, s19 // (op 585) 


s_sub_u32 s18, s8, s51 // (op 585) 


s_add_u32 s8, s18, s0 // (op 585) 


s_cmp_eq_i32 s9, 0 // (op 585) 
s_mov_b32 s0, scc // (op 585) 

s_cmp_eq_i32 s0, 0 // (op 585) 
s_cselect_b32 s9, s8, s1 // (op 585) 



v_lshl_add_u32 v12, s9, 4, v16 // (op 585) 


v_mul_lo_u32 v16, v12, s28 // (op 585) 
v_add_u32 v16, v16, v13 // (op 585) 


s_mul_i32 s1, s28, s7 // (op 587) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 587) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 587) least significant
s_mul_hi_u32 s8, s28, s6 // (op 587) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 587) most significant: sum
s_add_u32 s1, s1, s8 // (op 587) most significant: sum


s_mov_b32 s60, s30 // (op 587) reg expression
s_mov_b32 s61, s31 // (op 587) reg expression
s_mov_b32 s62, s0 // (op 587) reg expression
s_mov_b32 s63, 131072 // (op 587) call()

s_mul_hi_i32 s51, s16, s11 // (op 604) 
v_lshrrev_b32 v12, 7, v1 // (op 604) 
s_add_i32 s0, s51, s16 // (op 604) 

v_and_b32 v13, 15, v1 // (op 604) 
s_ashr_i32 s1, s0, 31 // (op 604) 
v_lshl_add_u32 v17, v12, 4, v13 // (op 604) 


s_cmp_eq_i32 s11, 0 // (op 604) 
s_mov_b32 s51, scc // (op 604) 
v_and_b32 v12, 63, v1 // (op 604) 
s_cmp_eq_i32 s51, 0 // (op 604) 
s_cselect_b32 s8, 0, -1 // (op 604) 

v_lshrrev_b32 v13, 4, v12 // (op 604) 

s_and_b32 s9, s12, 31 // (op 604) 
s_lshl_b32 s51, 1, s9 // (op 604) 
s_add_i32 s18, s51, s8 // (op 604) 


s_and_b32 s8, s1, s18 // (op 604) 


s_add_i32 s1, s0, s8 // (op 604) 


s_ashr_i32 s0, s1, s9 // (op 604) 


s_lshl_b32 s1, s12, 24 // (op 604) 
s_ashr_i32 s51, s1, 31 // (op 604) 

s_xor_b32 s1, s0, s51 // (op 604) 

s_sub_i32 s0, s1, s51 // (op 604) 


s_mul_hi_i32 s1, s0, s13 // (op 604) 
s_add_i32 s51, s1, s0 // (op 604) 

s_ashr_i32 s1, s51, 31 // (op 604) 
s_cmp_eq_i32 s13, 0 // (op 604) 
s_mov_b32 s8, scc // (op 604) 
s_cmp_eq_i32 s8, 0 // (op 604) 
s_cselect_b32 s9, 0, -1 // (op 604) 

s_and_b32 s8, s14, 31 // (op 604) 
s_lshl_b32 s18, 1, s8 // (op 604) 
s_add_i32 s19, s18, s9 // (op 604) 


s_and_b32 s9, s1, s19 // (op 604) 


s_add_i32 s1, s51, s9 // (op 604) 


s_ashr_i32 s9, s1, s8 // (op 604) 


s_lshl_b32 s1, s14, 24 // (op 604) 
s_ashr_i32 s8, s1, 31 // (op 604) 

s_xor_b32 s1, s9, s8 // (op 604) 

s_sub_i32 s9, s1, s8 // (op 604) 


s_mul_i32 s1, s0, s46 // (op 604) 

s_sub_u32 s0, s16, s1 // (op 604) 

s_mul_i32 s1, s9, s46 // (op 604) 
s_add_u32 s1, s1, s0 // (op 604) 


s_mul_hi_i32 s0, s16, s10 // (op 604) 
s_add_i32 s51, s0, s16 // (op 604) 

s_ashr_i32 s0, s51, 31 // (op 604) 
s_cmp_eq_i32 s10, 0 // (op 604) 
s_mov_b32 s8, scc // (op 604) 
s_cmp_eq_i32 s8, 0 // (op 604) 
s_cselect_b32 s9, 0, -1 // (op 604) 

s_and_b32 s8, s49, 31 // (op 604) 
s_lshl_b32 s18, 1, s8 // (op 604) 
s_add_i32 s19, s18, s9 // (op 604) 


s_and_b32 s9, s0, s19 // (op 604) 


s_add_i32 s0, s51, s9 // (op 604) 


s_ashr_i32 s9, s0, s8 // (op 604) 


s_lshl_b32 s0, s49, 24 // (op 604) 
s_ashr_i32 s8, s0, 31 // (op 604) 

s_xor_b32 s0, s9, s8 // (op 604) 

s_sub_i32 s9, s0, s8 // (op 604) 


s_add_u32 s18, s2, 32 // (op 604) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 604) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 604) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 604) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 604) 

s_mul_i32 s0, s46, s18 // (op 604) 

s_mul_i32 s8, s50, s0 // (op 604) 

s_mul_i32 s0, s9, s8 // (op 604) 

s_sub_u32 s8, s16, s0 // (op 604) 

s_mov_b32 s0, s8 // (op 604) 
s_mul_hi_i32 s51, s0, s15 // (op 604) 
s_add_i32 s18, s51, s0 // (op 604) 


s_ashr_i32 s0, s18, 31 // (op 604) 
s_cmp_eq_i32 s15, 0 // (op 604) 
s_mov_b32 s19, scc // (op 604) 
s_cmp_eq_i32 s19, 0 // (op 604) 
s_cselect_b32 s51, 0, -1 // (op 604) 

s_and_b32 s19, s48, 31 // (op 604) 
s_lshl_b32 s64, 1, s19 // (op 604) 
s_add_i32 s65, s64, s51 // (op 604) 


s_and_b32 s51, s0, s65 // (op 604) 


s_add_i32 s0, s18, s51 // (op 604) 


s_ashr_i32 s18, s0, s19 // (op 604) 


s_lshl_b32 s0, s48, 24 // (op 604) 
s_ashr_i32 s19, s0, 31 // (op 604) 

s_xor_b32 s0, s18, s19 // (op 604) 

s_sub_i32 s18, s0, s19 // (op 604) 


s_mul_i32 s0, s47, s46 // (op 604) 
s_add_u32 s64, s6, 16 // (op 604) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 604) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 604) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 604) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 604) 

s_sub_i32 s19, s64, s0 // (op 604) 

s_mul_i32 s51, s18, s19 // (op 604) 


s_sub_u32 s18, s8, s51 // (op 604) 


s_add_u32 s8, s18, s0 // (op 604) 


s_cmp_eq_i32 s9, 0 // (op 604) 
s_mov_b32 s0, scc // (op 604) 

s_cmp_eq_i32 s0, 0 // (op 604) 
s_cselect_b32 s9, s8, s1 // (op 604) 



v_lshl_add_u32 v12, s9, 4, v17 // (op 604) 


v_mul_lo_u32 v17, v12, s28 // (op 604) 
v_add_u32 v17, v17, v13 // (op 604) 


s_mul_i32 s1, s28, s7 // (op 606) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 606) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 606) least significant
s_mul_hi_u32 s8, s28, s6 // (op 606) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 606) most significant: sum
s_add_u32 s1, s1, s8 // (op 606) most significant: sum


s_mov_b32 s60, s30 // (op 606) reg expression
s_mov_b32 s61, s31 // (op 606) reg expression
s_mov_b32 s62, s0 // (op 606) reg expression
s_mov_b32 s63, 131072 // (op 606) call()

s_mul_hi_i32 s51, s16, s11 // (op 649) 
v_and_b32 v12, -32, v1 // (op 649) 
s_add_i32 s0, s51, s16 // (op 649) 

v_sub_u32 v13, v1, v12 // (op 649) 

s_ashr_i32 s1, s0, 31 // (op 649) 
v_lshrrev_b32 v12, 5, v1 // (op 649) 
s_cmp_eq_i32 s11, 0 // (op 649) 
s_mov_b32 s51, scc // (op 649) 
s_cmp_eq_i32 s51, 0 // (op 649) 
s_cselect_b32 s8, 0, -1 // (op 649) 

s_and_b32 s9, s12, 31 // (op 649) 
s_lshl_b32 s51, 1, s9 // (op 649) 
s_add_i32 s18, s51, s8 // (op 649) 


s_and_b32 s8, s1, s18 // (op 649) 


s_add_i32 s1, s0, s8 // (op 649) 


s_ashr_i32 s0, s1, s9 // (op 649) 


s_lshl_b32 s1, s12, 24 // (op 649) 
s_ashr_i32 s51, s1, 31 // (op 649) 

s_xor_b32 s1, s0, s51 // (op 649) 

s_sub_i32 s0, s1, s51 // (op 649) 


s_mul_hi_i32 s1, s0, s13 // (op 649) 
s_add_i32 s51, s1, s0 // (op 649) 

s_ashr_i32 s1, s51, 31 // (op 649) 
s_cmp_eq_i32 s13, 0 // (op 649) 
s_mov_b32 s8, scc // (op 649) 
s_cmp_eq_i32 s8, 0 // (op 649) 
s_cselect_b32 s9, 0, -1 // (op 649) 

s_and_b32 s8, s14, 31 // (op 649) 
s_lshl_b32 s18, 1, s8 // (op 649) 
s_add_i32 s19, s18, s9 // (op 649) 


s_and_b32 s9, s1, s19 // (op 649) 


s_add_i32 s1, s51, s9 // (op 649) 


s_ashr_i32 s9, s1, s8 // (op 649) 


s_lshl_b32 s1, s14, 24 // (op 649) 
s_ashr_i32 s8, s1, 31 // (op 649) 

s_xor_b32 s1, s9, s8 // (op 649) 

s_sub_i32 s9, s1, s8 // (op 649) 


s_add_u32 s18, s2, 32 // (op 649) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 649) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 649) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 649) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 649) 

s_mul_i32 s1, s9, s18 // (op 649) 

s_sub_i32 s51, s0, s1 // (op 649) 


s_mul_hi_i32 s0, s16, s10 // (op 649) 
s_add_i32 s1, s0, s16 // (op 649) 

s_ashr_i32 s0, s1, 31 // (op 649) 
s_cmp_eq_i32 s10, 0 // (op 649) 
s_mov_b32 s8, scc // (op 649) 
s_cmp_eq_i32 s8, 0 // (op 649) 
s_cselect_b32 s9, 0, -1 // (op 649) 

s_and_b32 s8, s49, 31 // (op 649) 
s_lshl_b32 s64, 1, s8 // (op 649) 
s_add_i32 s65, s64, s9 // (op 649) 


s_and_b32 s9, s0, s65 // (op 649) 


s_add_i32 s0, s1, s9 // (op 649) 


s_ashr_i32 s1, s0, s8 // (op 649) 


s_lshl_b32 s0, s49, 24 // (op 649) 
s_ashr_i32 s8, s0, 31 // (op 649) 

s_xor_b32 s0, s1, s8 // (op 649) 

s_sub_i32 s1, s0, s8 // (op 649) 


s_mul_i32 s0, s46, s18 // (op 649) 

s_mul_i32 s8, s50, s0 // (op 649) 

s_mul_i32 s0, s1, s8 // (op 649) 

s_sub_u32 s8, s16, s0 // (op 649) 

s_mul_hi_i32 s0, s8, s15 // (op 649) 
s_add_i32 s9, s0, s8 // (op 649) 


s_ashr_i32 s0, s9, 31 // (op 649) 
s_cmp_eq_i32 s15, 0 // (op 649) 
s_mov_b32 s8, scc // (op 649) 
s_cmp_eq_i32 s8, 0 // (op 649) 
s_cselect_b32 s18, 0, -1 // (op 649) 

s_and_b32 s8, s48, 31 // (op 649) 
s_lshl_b32 s19, 1, s8 // (op 649) 
s_add_i32 s64, s19, s18 // (op 649) 


s_and_b32 s18, s0, s64 // (op 649) 


s_add_i32 s0, s9, s18 // (op 649) 


s_ashr_i32 s9, s0, s8 // (op 649) 


s_lshl_b32 s0, s48, 24 // (op 649) 
s_ashr_i32 s8, s0, 31 // (op 649) 

s_xor_b32 s0, s9, s8 // (op 649) 

s_sub_i32 s9, s0, s8 // (op 649) 


s_cmp_eq_i32 s1, 0 // (op 649) 
s_mov_b32 s0, scc // (op 649) 

s_cmp_eq_i32 s0, 0 // (op 649) 
s_cselect_b32 s1, s9, s51 // (op 649) 



v_lshl_add_u32 v18, s1, 5, v13 // (op 649) 


v_mul_lo_u32 v13, v12, s32 // (op 649) 
v_add_u32 v13, v13, v18 // (op 649) 


s_mul_i32 s1, s32, s37 // (op 651) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 651) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 651) least significant
s_mul_hi_u32 s8, s32, s36 // (op 651) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 651) most significant: sum
s_add_u32 s1, s1, s8 // (op 651) most significant: sum


s_mov_b32 s64, s34 // (op 651) reg expression
s_mov_b32 s65, s35 // (op 651) reg expression
s_mov_b32 s66, s0 // (op 651) reg expression
s_mov_b32 s67, 131072 // (op 651) call()

s_mul_hi_i32 s51, s16, s11 // (op 668) 
v_and_b32 v12, -32, v1 // (op 668) 
s_add_i32 s0, s51, s16 // (op 668) 

v_sub_u32 v18, v1, v12 // (op 668) 

s_ashr_i32 s1, s0, 31 // (op 668) 
v_lshrrev_b32 v12, 5, v1 // (op 668) 
s_cmp_eq_i32 s11, 0 // (op 668) 
s_mov_b32 s51, scc // (op 668) 
s_cmp_eq_i32 s51, 0 // (op 668) 
s_cselect_b32 s8, 0, -1 // (op 668) 

s_and_b32 s9, s12, 31 // (op 668) 
s_lshl_b32 s51, 1, s9 // (op 668) 
s_add_i32 s18, s51, s8 // (op 668) 


s_and_b32 s8, s1, s18 // (op 668) 


s_add_i32 s1, s0, s8 // (op 668) 


s_ashr_i32 s0, s1, s9 // (op 668) 


s_lshl_b32 s1, s12, 24 // (op 668) 
s_ashr_i32 s51, s1, 31 // (op 668) 

s_xor_b32 s1, s0, s51 // (op 668) 

s_sub_i32 s0, s1, s51 // (op 668) 


s_mul_hi_i32 s1, s0, s13 // (op 668) 
s_add_i32 s51, s1, s0 // (op 668) 

s_ashr_i32 s1, s51, 31 // (op 668) 
s_cmp_eq_i32 s13, 0 // (op 668) 
s_mov_b32 s8, scc // (op 668) 
s_cmp_eq_i32 s8, 0 // (op 668) 
s_cselect_b32 s9, 0, -1 // (op 668) 

s_and_b32 s8, s14, 31 // (op 668) 
s_lshl_b32 s18, 1, s8 // (op 668) 
s_add_i32 s19, s18, s9 // (op 668) 


s_and_b32 s9, s1, s19 // (op 668) 


s_add_i32 s1, s51, s9 // (op 668) 


s_ashr_i32 s9, s1, s8 // (op 668) 


s_lshl_b32 s1, s14, 24 // (op 668) 
s_ashr_i32 s8, s1, 31 // (op 668) 

s_xor_b32 s1, s9, s8 // (op 668) 

s_sub_i32 s9, s1, s8 // (op 668) 


s_add_u32 s18, s2, 32 // (op 668) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 668) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 668) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 668) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 668) 

s_mul_i32 s1, s9, s18 // (op 668) 

s_sub_i32 s51, s0, s1 // (op 668) 


s_mul_hi_i32 s0, s16, s10 // (op 668) 
s_add_i32 s1, s0, s16 // (op 668) 

s_ashr_i32 s0, s1, 31 // (op 668) 
s_cmp_eq_i32 s10, 0 // (op 668) 
s_mov_b32 s8, scc // (op 668) 
s_cmp_eq_i32 s8, 0 // (op 668) 
s_cselect_b32 s9, 0, -1 // (op 668) 

s_and_b32 s8, s49, 31 // (op 668) 
s_lshl_b32 s68, 1, s8 // (op 668) 
s_add_i32 s69, s68, s9 // (op 668) 


s_and_b32 s9, s0, s69 // (op 668) 


s_add_i32 s0, s1, s9 // (op 668) 


s_ashr_i32 s1, s0, s8 // (op 668) 


s_lshl_b32 s0, s49, 24 // (op 668) 
s_ashr_i32 s8, s0, 31 // (op 668) 

s_xor_b32 s0, s1, s8 // (op 668) 

s_sub_i32 s1, s0, s8 // (op 668) 


s_mul_i32 s0, s46, s18 // (op 668) 

s_mul_i32 s8, s50, s0 // (op 668) 

s_mul_i32 s0, s1, s8 // (op 668) 

s_sub_u32 s8, s16, s0 // (op 668) 

s_mul_hi_i32 s0, s8, s15 // (op 668) 
s_add_i32 s9, s0, s8 // (op 668) 


s_ashr_i32 s0, s9, 31 // (op 668) 
s_cmp_eq_i32 s15, 0 // (op 668) 
s_mov_b32 s8, scc // (op 668) 
s_cmp_eq_i32 s8, 0 // (op 668) 
s_cselect_b32 s18, 0, -1 // (op 668) 

s_and_b32 s8, s48, 31 // (op 668) 
s_lshl_b32 s19, 1, s8 // (op 668) 
s_add_i32 s68, s19, s18 // (op 668) 


s_and_b32 s18, s0, s68 // (op 668) 


s_add_i32 s0, s9, s18 // (op 668) 


s_ashr_i32 s9, s0, s8 // (op 668) 


s_lshl_b32 s0, s48, 24 // (op 668) 
s_ashr_i32 s8, s0, 31 // (op 668) 

s_xor_b32 s0, s9, s8 // (op 668) 

s_sub_i32 s9, s0, s8 // (op 668) 


s_cmp_eq_i32 s1, 0 // (op 668) 
s_mov_b32 s0, scc // (op 668) 

s_cmp_eq_i32 s0, 0 // (op 668) 
s_cselect_b32 s1, s9, s51 // (op 668) 



v_lshl_add_u32 v19, s1, 5, v18 // (op 668) 


v_mul_lo_u32 v18, v12, s32 // (op 668) 
v_add_u32 v18, v18, v19 // (op 668) 


s_mul_i32 s1, s32, s37 // (op 670) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 670) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 670) least significant
s_mul_hi_u32 s8, s32, s36 // (op 670) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 670) most significant: sum
s_add_u32 s1, s1, s8 // (op 670) most significant: sum


s_mov_b32 s64, s34 // (op 670) reg expression
s_mov_b32 s65, s35 // (op 670) reg expression
s_mov_b32 s66, s0 // (op 670) reg expression
s_mov_b32 s67, 131072 // (op 670) call()




buffer_load_ubyte v12, v16, s[60:63], 0 offen offset: 0 // (op 341) Load value


v_mov_b32 v19, v14 // (op 345) reg expression

buffer_load_dwordx3 v[20:22], v19, s[56:59], 0 offen // (op 345) Load value
buffer_load_dwordx3 v[24:26], v19, s[56:59], 0 offen offset:12 // (op 345) Load value



v_mov_b32 v19, v13 // (op 348) reg expression

buffer_load_ubyte v23, v19, s[64:67], 0 offen offset: 0 // (op 348) Load value




buffer_load_dwordx3 v[28:30], v10, s[52:55], 0 offen // (op 351) Load value



buffer_load_ubyte v19, v16, s[60:63], 0 offen offset:4 // (op 355) Load value



s_lshl_b64 s[0:1], s[22:23], 7 // (op 359) 
s_mul_i32 s9, s1, 6 // (op 359) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 359) least significant
s_mul_hi_u32 s51, s0, 6 // (op 359) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 359) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 359) 

v_add_u32 v16, v14, s0 // (op 359) 


buffer_load_dwordx3 v[32:34], v16, s[56:59], 0 offen // (op 359) Load value
buffer_load_dwordx3 v[36:38], v16, s[56:59], 0 offen offset:12 // (op 359) Load value




s_lshl_b32 s51, s32, 2 // (op 362) 
v_add_u32 v14, v13, s51 // (op 362) 


buffer_load_ubyte v16, v14, s[64:67], 0 offen offset: 0 // (op 362) Load value





buffer_load_dwordx3 v[40:42], v10, s[52:55], 0 offen offset:96 // (op 365) Load value

v_lshlrev_b32 v10, 5, v1 // (op 788) 
v_mul_lo_u32 v27, v10, 6 // (op 788) 
v_lshrrev_b32 v31, 3, v27 // (op 788) 

v_lshrrev_b32 v27, 4, v10 // (op 788) 

v_lshlrev_b32 v10, 2, v27 // (op 788) 

v_add_u32 v27, v31, v10 // (op 788) 





s_waitcnt vmcnt(8)
ds_write_b96 v27, v[20:22] // (op 367) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v27, v[24:26] offset:16 // (op 367) Store local data 

v_and_b32 v10, -32, v1 // (op 841) 
v_sub_u32 v27, v1, v10 // (op 841) 

v_and_b32 v10, 4294967264, v1 // (op 841) 
v_add_u32 v31, v10, v27 // (op 841) 





s_waitcnt vmcnt(6)
ds_write_b8 v31, v23 offset:4096 // (op 368) Store local data 

v_and_b32 v10, -8, v1 // (op 735) 
v_sub_u32 v27, v1, v10 // (op 735) 

v_lshlrev_b32 v10, 4, v27 // (op 735) 

v_lshrrev_b32 v27, 3, v1 // (op 735) 
v_lshl_add_u32 v31, v27, 7, v10 // (op 735) 


v_mul_lo_u32 v10, v31, 6 // (op 735) 

v_lshrrev_b32 v27, 3, v10 // (op 735) 




s_waitcnt vmcnt(5)
ds_write_b96 v27, v[28:30] offset:4224 // (op 369) Store local data 

v_and_b32 v10, -8, v1 // (op 706) 
v_sub_u32 v27, v1, v10 // (op 706) 

v_lshlrev_b32 v10, 4, v27 // (op 706) 

v_lshrrev_b32 v27, 3, v1 // (op 706) 
v_lshl_add_u32 v31, v27, 7, v10 // (op 706) 


v_mul_lo_u32 v10, v31, 6 // (op 706) 

v_lshrrev_b32 v27, 3, v10 // (op 706) 

v_lshlrev_b32 v10, 5, v1 // (op 759) 
v_mul_lo_u32 v31, v10, 6 // (op 759) 
v_lshrrev_b32 v35, 3, v31 // (op 759) 

v_lshrrev_b32 v31, 4, v10 // (op 759) 

v_lshlrev_b32 v10, 2, v31 // (op 759) 

v_add_u32 v31, v35, v10 // (op 759) 


v_and_b32 v10, -32, v1 // (op 812) 
v_sub_u32 v35, v1, v10 // (op 812) 

v_and_b32 v10, 4294967264, v1 // (op 812) 
v_add_u32 v39, v10, v35 // (op 812) 


v_and_b32 v10, -8, v1 // (op 899) 
v_sub_u32 v35, v1, v10 // (op 899) 

v_lshlrev_b32 v10, 4, v35 // (op 899) 

v_lshrrev_b32 v35, 3, v1 // (op 899) 
v_lshl_add_u32 v13, v35, 7, v10 // (op 899) 


v_mul_lo_u32 v10, v13, 6 // (op 899) 

v_lshrrev_b32 v35, 3, v10 // (op 899) 

v_lshlrev_b32 v10, 5, v1 // (op 923) 
v_mul_lo_u32 v13, v10, 6 // (op 923) 
v_lshrrev_b32 v14, 3, v13 // (op 923) 

v_lshrrev_b32 v13, 4, v10 // (op 923) 

v_lshlrev_b32 v10, 2, v13 // (op 923) 

v_add_u32 v13, v14, v10 // (op 923) 


v_and_b32 v10, -32, v1 // (op 947) 
v_sub_u32 v14, v1, v10 // (op 947) 

v_and_b32 v10, 4294967264, v1 // (op 947) 
v_add_u32 v43, v10, v14 // (op 947) 


s_mov_b32 s51, 0 // (op 57) call()
s_ashr_i64 s[0:1], s[26:27], 7 // (op 56) 
s_mov_b32 s8, s0 // (op 56) convert

s_and_b32 s9, s8, 4294967294 // (op 56) 

s_cmp_lt_u32 s51, s9 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56 if false)
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56:
s_add_u32 s51, s51, 2 // (op 58) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 992) 



ds_read_b128 v[44:47], v3 offset:4224 // (op 67) Load local data 
ds_read_b64 v[48:49], v3 offset:4240 // (op 67) Load local data 
s_barrier  // (op 1005) 



ds_read_b96_tr_b6 v[50:52], v0 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[53:55], v0 offset:512 // (op 72) Transpose load local data 
s_barrier  // (op 1019) 


ds_read_u8 v10, v4 offset:4096 // (op 81) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:49], v[50:55], a[0:3], v12, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 84) 





v_mov_b32 v10, v17 // (op 77) 

buffer_load_ubyte v12, v10, s[60:63], 0 offen offset:8 // (op 77) Load value


v_mov_b32 v14, v15 // (op 10) 
s_lshl_b64 s[0:1], s[22:23], 7 // (op 10) 
s_mul_i32 s9, s1, 6 // (op 10) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 10) least significant
s_mul_hi_u32 s18, s0, 6 // (op 10) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 10) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 10) 

s_lshl_b32 s8, s0, 1 // (op 10) 

v_add_u32 v44, v14, s8 // (op 10) 


buffer_load_dwordx3 v[20:22], v44, s[56:59], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[24:26], v44, s[56:59], 0 offen offset:12 // (op 10) Load value



v_mov_b32 v44, v18 // (op 21) 
s_lshl_b32 s0, s32, 3 // (op 21) 
v_add_u32 v45, v44, s0 // (op 21) 


buffer_load_ubyte v23, v45, s[64:67], 0 offen offset: 0 // (op 21) Load value



v_mov_b32 v45, v11 // (op 4) 

buffer_load_dwordx3 v[28:30], v45, s[52:55], 0 offen offset:192 // (op 4) Load value
s_barrier  // (op 1042) 



s_waitcnt vmcnt(8)
ds_write_b96 v13, v[32:34] offset:5760 // (op 239) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v13, v[36:38] offset:5776 // (op 239) Store local data 
s_barrier  // (op 1048) 



s_waitcnt vmcnt(6)
ds_write_b8 v43, v16 offset:9856 // (op 240) Store local data 
s_barrier  // (op 1036) 



s_waitcnt vmcnt(5)
ds_write_b96 v35, v[40:42] offset:9984 // (op 229) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1033) 



ds_read_b128 v[46:49], v6 offset:9984 // (op 231) Load local data 
ds_read_b64 v[50:51], v6 offset:10000 // (op 231) Load local data 
s_barrier  // (op 1039) 
s_barrier  // (op 1103) 



ds_read_b96_tr_b6 v[52:54], v5 offset:5760 // (op 233) Transpose load local data 
ds_read_b96_tr_b6 v[55:57], v5 offset:6272 // (op 233) Transpose load local data 
s_barrier  // (op 1045) 
s_barrier  // (op 1109) 


ds_read_u8 v58, v7 offset:9856 // (op 237) Load local data 
s_barrier  // (op 1117) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:51], v[52:57], a[0:3], v19, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 238) 






buffer_load_ubyte v19, v10, s[60:63], 0 offen offset:12 // (op 235) Load value



v_add_u32 v17, 8, v17 // (op 596) 
s_lshl_b64 s[0:1], s[22:23], 7 // (op 241) 
s_mul_i32 s9, s1, 6 // (op 241) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 241) least significant
s_mul_hi_u32 s18, s0, 6 // (op 241) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 241) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 241) 

s_mul_i32 s8, 3, s0 // (op 241) 

v_add_u32 v10, v14, s8 // (op 241) 


buffer_load_dwordx3 v[32:34], v10, s[56:59], 0 offen // (op 241) Load value
buffer_load_dwordx3 v[36:38], v10, s[56:59], 0 offen offset:12 // (op 241) Load value




s_lshl_b32 s0, s32, 2 // (op 242) 
s_mul_i32 s1, 3, s0 // (op 242) 

v_add_u32 v10, v44, s1 // (op 242) 


s_lshl_b64 s[0:1], s[22:23], 7 // (op 532) 
s_mul_i32 s9, s1, 6 // (op 532) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 532) least significant
s_mul_hi_u32 s18, s0, 6 // (op 532) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 532) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 532) 

s_lshl_b32 s8, s0, 1 // (op 532) 

v_add_u32 v15, v15, s8 // (op 532) 

buffer_load_ubyte v16, v10, s[64:67], 0 offen offset: 0 // (op 242) Load value





s_lshl_b32 s0, s32, 3 // (op 660) 
v_add_u32 v18, v18, s0 // (op 660) 

buffer_load_dwordx3 v[40:42], v45, s[52:55], 0 offen offset:288 // (op 228) Load value

s_barrier  // (op 1008) 
v_add_u32 v11, 192, v11 // (op 468) 



s_waitcnt vmcnt(8)
ds_write_b96 v31, v[20:22] // (op 13) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v31, v[24:26] offset:16 // (op 13) Store local data 
s_barrier  // (op 1022) 



s_waitcnt vmcnt(6)
ds_write_b8 v39, v23 offset:4096 // (op 26) Store local data 
s_barrier  // (op 995) 



s_waitcnt vmcnt(5)
ds_write_b96 v27, v[28:30] offset:4224 // (op 6) Store local data 
s_ashr_i64 s[0:1], s[26:27], 7 // (op 56) 
s_mov_b32 s8, s0 // (op 56) convert

s_and_b32 s9, s8, 4294967294 // (op 56) 

s_cmp_lt_u32 s51, s9 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56 if true)
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56:















s_mul_hi_i32 s51, s16, s11 // (op 431) 
v_lshrrev_b32 v35, 3, v1 // (op 431) 
s_add_i32 s0, s51, s16 // (op 431) 

v_and_b32 v39, -8, v1 // (op 431) 
s_ashr_i32 s1, s0, 31 // (op 431) 
v_sub_u32 v10, v1, v39 // (op 431) 

s_cmp_eq_i32 s11, 0 // (op 431) 
s_mov_b32 s51, scc // (op 431) 
v_lshlrev_b32 v39, 4, v10 // (op 431) 

s_cmp_eq_i32 s51, 0 // (op 431) 
s_cselect_b32 s8, 0, -1 // (op 431) 

s_and_b32 s9, s12, 31 // (op 431) 
s_lshl_b32 s51, 1, s9 // (op 431) 
s_add_i32 s18, s51, s8 // (op 431) 


s_and_b32 s8, s1, s18 // (op 431) 


s_add_i32 s1, s0, s8 // (op 431) 


s_ashr_i32 s0, s1, s9 // (op 431) 


s_lshl_b32 s1, s12, 24 // (op 431) 
s_ashr_i32 s51, s1, 31 // (op 431) 

s_xor_b32 s1, s0, s51 // (op 431) 

s_sub_i32 s0, s1, s51 // (op 431) 


s_mul_hi_i32 s1, s0, s13 // (op 431) 
s_add_i32 s51, s1, s0 // (op 431) 

s_ashr_i32 s1, s51, 31 // (op 431) 
s_cmp_eq_i32 s13, 0 // (op 431) 
s_mov_b32 s8, scc // (op 431) 
s_cmp_eq_i32 s8, 0 // (op 431) 
s_cselect_b32 s9, 0, -1 // (op 431) 

s_and_b32 s8, s14, 31 // (op 431) 
s_lshl_b32 s18, 1, s8 // (op 431) 
s_add_i32 s19, s18, s9 // (op 431) 


s_and_b32 s9, s1, s19 // (op 431) 


s_add_i32 s1, s51, s9 // (op 431) 


s_ashr_i32 s9, s1, s8 // (op 431) 


s_lshl_b32 s1, s14, 24 // (op 431) 
s_ashr_i32 s8, s1, 31 // (op 431) 

s_xor_b32 s1, s9, s8 // (op 431) 

s_sub_i32 s9, s1, s8 // (op 431) 


s_mul_i32 s1, s0, s46 // (op 431) 

s_sub_u32 s0, s16, s1 // (op 431) 

s_mul_i32 s1, s9, s46 // (op 431) 
s_add_u32 s1, s1, s0 // (op 431) 


s_mul_hi_i32 s0, s16, s10 // (op 431) 
s_add_i32 s51, s0, s16 // (op 431) 

s_ashr_i32 s0, s51, 31 // (op 431) 
s_cmp_eq_i32 s10, 0 // (op 431) 
s_mov_b32 s8, scc // (op 431) 
s_cmp_eq_i32 s8, 0 // (op 431) 
s_cselect_b32 s9, 0, -1 // (op 431) 

s_and_b32 s8, s49, 31 // (op 431) 
s_lshl_b32 s18, 1, s8 // (op 431) 
s_add_i32 s19, s18, s9 // (op 431) 


s_and_b32 s9, s0, s19 // (op 431) 


s_add_i32 s0, s51, s9 // (op 431) 


s_ashr_i32 s9, s0, s8 // (op 431) 


s_lshl_b32 s0, s49, 24 // (op 431) 
s_ashr_i32 s8, s0, 31 // (op 431) 

s_xor_b32 s0, s9, s8 // (op 431) 

s_sub_i32 s9, s0, s8 // (op 431) 


s_add_u32 s18, s2, 32 // (op 431) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 431) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 431) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 431) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 431) 

s_mul_i32 s0, s46, s18 // (op 431) 

s_mul_i32 s8, s50, s0 // (op 431) 

s_mul_i32 s0, s9, s8 // (op 431) 

s_sub_u32 s8, s16, s0 // (op 431) 

s_mov_b32 s0, s8 // (op 431) 
s_mul_hi_i32 s51, s0, s15 // (op 431) 
s_add_i32 s18, s51, s0 // (op 431) 


s_ashr_i32 s0, s18, 31 // (op 431) 
s_cmp_eq_i32 s15, 0 // (op 431) 
s_mov_b32 s19, scc // (op 431) 
s_cmp_eq_i32 s19, 0 // (op 431) 
s_cselect_b32 s51, 0, -1 // (op 431) 

s_and_b32 s19, s48, 31 // (op 431) 
s_lshl_b32 s68, 1, s19 // (op 431) 
s_add_i32 s69, s68, s51 // (op 431) 


s_and_b32 s51, s0, s69 // (op 431) 


s_add_i32 s0, s18, s51 // (op 431) 


s_ashr_i32 s18, s0, s19 // (op 431) 


s_lshl_b32 s0, s48, 24 // (op 431) 
s_ashr_i32 s19, s0, 31 // (op 431) 

s_xor_b32 s0, s18, s19 // (op 431) 

s_sub_i32 s18, s0, s19 // (op 431) 


s_mul_i32 s0, s47, s46 // (op 431) 
s_add_u32 s68, s6, 16 // (op 431) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 431) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 431) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 431) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 431) 

s_sub_i32 s19, s68, s0 // (op 431) 

s_mul_i32 s51, s18, s19 // (op 431) 


s_sub_u32 s18, s8, s51 // (op 431) 


s_add_u32 s8, s18, s0 // (op 431) 


s_cmp_eq_i32 s9, 0 // (op 431) 
s_mov_b32 s0, scc // (op 431) 

s_cmp_eq_i32 s0, 0 // (op 431) 
s_cselect_b32 s9, s8, s1 // (op 431) 



v_lshl_add_u32 v10, s9, 4, v35 // (op 431) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 431) 
s_lshr_b32 s51, s0, 1 // (op 431) 

v_lshl_add_u32 v35, s51, 8, v39 // (op 431) 


v_mov_b32 v15, 0 // (op 431) convert
v_mov_b32 v14, v35 // (op 431) convert

v_mul_lo_u32 v13, v10, s5 // (op 431) most significant: low of low * high
v_mul_hi_u32 v11, v10, s4 // (op 431) most significant: high of low * low
v_mul_lo_u32 v12, v10, s4 // (op 431) least significant: low of low * low
v_add_u32 v13, v13, v11 // (op 431) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v14 // (op 431) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 431) most significant half



v_mul_lo_u32 v11, v13, 6 // (op 431) most significant: low of high * low
v_mul_hi_u32 v35, v12, 6 // (op 431) most significant: high of low * low
v_mul_lo_u32 v10, v12, 6 // (op 431) least significant: low of low * low
v_add_u32 v11, v11, v35 // (op 431) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 431) 

v_mov_b32 v35, v12 // (op 431) convert

s_mul_i32 s1, s7, 6 // (op 433) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 433) least significant
s_mul_hi_u32 s51, s6, 6 // (op 433) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 433) most significant: sum

s_mul_i32 s9, s4, s1 // (op 433) most significant: low of low * high
s_mul_i32 s51, s5, s0 // (op 433) most significant: low of high * low
s_mul_i32 s8, s4, s0 // (op 433) least significant
s_mul_hi_u32 s18, s4, s0 // (op 433) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 433) most significant: sum
s_add_u32 s9, s9, s18 // (op 433) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 433) 

s_mov_b32 s52, s20 // (op 433) reg expression
s_mov_b32 s53, s21 // (op 433) reg expression
s_mov_b32 s54, s0 // (op 433) reg expression
s_mov_b32 s55, 131072 // (op 433) call()

s_mul_hi_i32 s51, s16, s11 // (op 498) 
s_add_i32 s0, s51, s16 // (op 498) 

s_ashr_i32 s1, s0, 31 // (op 498) 
s_cmp_eq_i32 s11, 0 // (op 498) 
s_mov_b32 s51, scc // (op 498) 
s_cmp_eq_i32 s51, 0 // (op 498) 
s_cselect_b32 s8, 0, -1 // (op 498) 

s_and_b32 s9, s12, 31 // (op 498) 
s_lshl_b32 s51, 1, s9 // (op 498) 
s_add_i32 s18, s51, s8 // (op 498) 


s_and_b32 s8, s1, s18 // (op 498) 


s_add_i32 s1, s0, s8 // (op 498) 


s_ashr_i32 s0, s1, s9 // (op 498) 


s_lshl_b32 s1, s12, 24 // (op 498) 
s_ashr_i32 s51, s1, 31 // (op 498) 

s_xor_b32 s1, s0, s51 // (op 498) 

s_sub_i32 s0, s1, s51 // (op 498) 


s_mul_hi_i32 s1, s0, s13 // (op 498) 
s_add_i32 s51, s1, s0 // (op 498) 

s_ashr_i32 s1, s51, 31 // (op 498) 
s_cmp_eq_i32 s13, 0 // (op 498) 
s_mov_b32 s8, scc // (op 498) 
s_cmp_eq_i32 s8, 0 // (op 498) 
s_cselect_b32 s9, 0, -1 // (op 498) 

s_and_b32 s8, s14, 31 // (op 498) 
s_lshl_b32 s18, 1, s8 // (op 498) 
s_add_i32 s19, s18, s9 // (op 498) 


s_and_b32 s9, s1, s19 // (op 498) 


s_add_i32 s1, s51, s9 // (op 498) 


s_ashr_i32 s9, s1, s8 // (op 498) 


s_lshl_b32 s1, s14, 24 // (op 498) 
s_ashr_i32 s8, s1, 31 // (op 498) 

s_xor_b32 s1, s9, s8 // (op 498) 

s_sub_i32 s9, s1, s8 // (op 498) 


s_add_u32 s18, s2, 32 // (op 498) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 498) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 498) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 498) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 498) 

s_mul_i32 s1, s9, s18 // (op 498) 

s_sub_i32 s51, s0, s1 // (op 498) 


s_mul_hi_i32 s0, s16, s10 // (op 498) 
s_add_i32 s1, s0, s16 // (op 498) 

s_ashr_i32 s0, s1, 31 // (op 498) 
s_cmp_eq_i32 s10, 0 // (op 498) 
s_mov_b32 s8, scc // (op 498) 
s_cmp_eq_i32 s8, 0 // (op 498) 
s_cselect_b32 s9, 0, -1 // (op 498) 

s_and_b32 s8, s49, 31 // (op 498) 
s_lshl_b32 s68, 1, s8 // (op 498) 
s_add_i32 s69, s68, s9 // (op 498) 


s_and_b32 s9, s0, s69 // (op 498) 


s_add_i32 s0, s1, s9 // (op 498) 


s_ashr_i32 s1, s0, s8 // (op 498) 


s_lshl_b32 s0, s49, 24 // (op 498) 
s_ashr_i32 s8, s0, 31 // (op 498) 

s_xor_b32 s0, s1, s8 // (op 498) 

s_sub_i32 s1, s0, s8 // (op 498) 


s_mul_i32 s0, s46, s18 // (op 498) 

s_mul_i32 s8, s50, s0 // (op 498) 

s_mul_i32 s0, s1, s8 // (op 498) 

s_sub_u32 s8, s16, s0 // (op 498) 

s_mul_hi_i32 s0, s8, s15 // (op 498) 
s_add_i32 s9, s0, s8 // (op 498) 


s_ashr_i32 s0, s9, 31 // (op 498) 
s_cmp_eq_i32 s15, 0 // (op 498) 
s_mov_b32 s8, scc // (op 498) 
s_cmp_eq_i32 s8, 0 // (op 498) 
s_cselect_b32 s18, 0, -1 // (op 498) 

s_and_b32 s8, s48, 31 // (op 498) 
s_lshl_b32 s19, 1, s8 // (op 498) 
s_add_i32 s68, s19, s18 // (op 498) 


s_and_b32 s18, s0, s68 // (op 498) 


s_add_i32 s0, s9, s18 // (op 498) 


s_ashr_i32 s9, s0, s8 // (op 498) 


s_lshl_b32 s0, s48, 24 // (op 498) 
s_ashr_i32 s8, s0, 31 // (op 498) 

s_xor_b32 s0, s9, s8 // (op 498) 

s_sub_i32 s9, s0, s8 // (op 498) 


s_cmp_eq_i32 s1, 0 // (op 498) 
s_mov_b32 s0, scc // (op 498) 

s_cmp_eq_i32 s0, 0 // (op 498) 
s_cselect_b32 s1, s9, s51 // (op 498) 



s_lshl_b32 s0, s1, 5 // (op 498) 

s_mov_b32 s9, 0 // (op 498) convert
s_mov_b32 s8, s0 // (op 498) convert

s_ashr_i64 s[0:1], s[26:27], 7 // (op 498) 
s_lshr_b32 s51, s0, 1 // (op 498) 

v_lshl_add_u32 v39, s51, 8, v1 // (op 498) 

v_mul_lo_u32 v11, v39, s23 // (op 498) most significant: low of low * high
v_mul_hi_u32 v12, v39, s22 // (op 498) most significant: high of low * low
v_mul_lo_u32 v10, v39, s22 // (op 498) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 498) most significant: sum

v_mov_b32 v12, s8 // (op 498) 
v_mov_b32 v13, s9 // (op 498) 
v_add_co_u32 v10, s[0:1], v10, v12 // (op 498) least significant half
v_addc_co_u32 v11, s[0:1], v11, v13, s[0:1] // (op 498) most significant half





v_mul_lo_u32 v13, v11, 6 // (op 498) most significant: low of high * low
v_mul_hi_u32 v39, v10, 6 // (op 498) most significant: high of low * low
v_mul_lo_u32 v12, v10, 6 // (op 498) least significant: low of low * low
v_add_u32 v13, v13, v39 // (op 498) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 498) 

v_mov_b32 v39, v10 // (op 498) convert

s_mul_i32 s1, s27, 6 // (op 500) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 500) least significant
s_mul_hi_u32 s51, s26, 6 // (op 500) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 500) most significant: sum

s_mul_i32 s9, s22, s1 // (op 500) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 500) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 500) least significant
s_mul_hi_u32 s18, s22, s0 // (op 500) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 500) most significant: sum
s_add_u32 s9, s9, s18 // (op 500) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 500) 

s_mov_b32 s56, s24 // (op 500) reg expression
s_mov_b32 s57, s25 // (op 500) reg expression
s_mov_b32 s58, s0 // (op 500) reg expression
s_mov_b32 s59, 131072 // (op 500) call()

s_mul_hi_i32 s51, s16, s11 // (op 562) 
v_lshrrev_b32 v10, 7, v1 // (op 562) 
s_add_i32 s0, s51, s16 // (op 562) 

v_and_b32 v11, 15, v1 // (op 562) 
s_ashr_i32 s1, s0, 31 // (op 562) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 562) 


s_cmp_eq_i32 s11, 0 // (op 562) 
s_mov_b32 s51, scc // (op 562) 
v_and_b32 v10, 63, v1 // (op 562) 
s_cmp_eq_i32 s51, 0 // (op 562) 
s_cselect_b32 s8, 0, -1 // (op 562) 

v_lshrrev_b32 v11, 4, v10 // (op 562) 

s_and_b32 s9, s12, 31 // (op 562) 
s_lshl_b32 s51, 1, s9 // (op 562) 
s_add_i32 s18, s51, s8 // (op 562) 


s_and_b32 s8, s1, s18 // (op 562) 


s_add_i32 s1, s0, s8 // (op 562) 


s_ashr_i32 s0, s1, s9 // (op 562) 


s_lshl_b32 s1, s12, 24 // (op 562) 
s_ashr_i32 s51, s1, 31 // (op 562) 

s_xor_b32 s1, s0, s51 // (op 562) 

s_sub_i32 s0, s1, s51 // (op 562) 


s_mul_hi_i32 s1, s0, s13 // (op 562) 
s_add_i32 s51, s1, s0 // (op 562) 

s_ashr_i32 s1, s51, 31 // (op 562) 
s_cmp_eq_i32 s13, 0 // (op 562) 
s_mov_b32 s8, scc // (op 562) 
s_cmp_eq_i32 s8, 0 // (op 562) 
s_cselect_b32 s9, 0, -1 // (op 562) 

s_and_b32 s8, s14, 31 // (op 562) 
s_lshl_b32 s18, 1, s8 // (op 562) 
s_add_i32 s19, s18, s9 // (op 562) 


s_and_b32 s9, s1, s19 // (op 562) 


s_add_i32 s1, s51, s9 // (op 562) 


s_ashr_i32 s9, s1, s8 // (op 562) 


s_lshl_b32 s1, s14, 24 // (op 562) 
s_ashr_i32 s8, s1, 31 // (op 562) 

s_xor_b32 s1, s9, s8 // (op 562) 

s_sub_i32 s9, s1, s8 // (op 562) 


s_mul_i32 s1, s0, s46 // (op 562) 

s_sub_u32 s0, s16, s1 // (op 562) 

s_mul_i32 s1, s9, s46 // (op 562) 
s_add_u32 s1, s1, s0 // (op 562) 


s_mul_hi_i32 s0, s16, s10 // (op 562) 
s_add_i32 s51, s0, s16 // (op 562) 

s_ashr_i32 s0, s51, 31 // (op 562) 
s_cmp_eq_i32 s10, 0 // (op 562) 
s_mov_b32 s8, scc // (op 562) 
s_cmp_eq_i32 s8, 0 // (op 562) 
s_cselect_b32 s9, 0, -1 // (op 562) 

s_and_b32 s8, s49, 31 // (op 562) 
s_lshl_b32 s18, 1, s8 // (op 562) 
s_add_i32 s19, s18, s9 // (op 562) 


s_and_b32 s9, s0, s19 // (op 562) 


s_add_i32 s0, s51, s9 // (op 562) 


s_ashr_i32 s9, s0, s8 // (op 562) 


s_lshl_b32 s0, s49, 24 // (op 562) 
s_ashr_i32 s8, s0, 31 // (op 562) 

s_xor_b32 s0, s9, s8 // (op 562) 

s_sub_i32 s9, s0, s8 // (op 562) 


s_add_u32 s18, s2, 32 // (op 562) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 562) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 562) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 562) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 562) 

s_mul_i32 s0, s46, s18 // (op 562) 

s_mul_i32 s8, s50, s0 // (op 562) 

s_mul_i32 s0, s9, s8 // (op 562) 

s_sub_u32 s8, s16, s0 // (op 562) 

s_mov_b32 s0, s8 // (op 562) 
s_mul_hi_i32 s51, s0, s15 // (op 562) 
s_add_i32 s18, s51, s0 // (op 562) 


s_ashr_i32 s0, s18, 31 // (op 562) 
s_cmp_eq_i32 s15, 0 // (op 562) 
s_mov_b32 s19, scc // (op 562) 
s_cmp_eq_i32 s19, 0 // (op 562) 
s_cselect_b32 s51, 0, -1 // (op 562) 

s_and_b32 s19, s48, 31 // (op 562) 
s_lshl_b32 s68, 1, s19 // (op 562) 
s_add_i32 s69, s68, s51 // (op 562) 


s_and_b32 s51, s0, s69 // (op 562) 


s_add_i32 s0, s18, s51 // (op 562) 


s_ashr_i32 s18, s0, s19 // (op 562) 


s_lshl_b32 s0, s48, 24 // (op 562) 
s_ashr_i32 s19, s0, 31 // (op 562) 

s_xor_b32 s0, s18, s19 // (op 562) 

s_sub_i32 s18, s0, s19 // (op 562) 


s_mul_i32 s0, s47, s46 // (op 562) 
s_add_u32 s68, s6, 16 // (op 562) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 562) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 562) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 562) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 562) 

s_sub_i32 s19, s68, s0 // (op 562) 

s_mul_i32 s51, s18, s19 // (op 562) 


s_sub_u32 s18, s8, s51 // (op 562) 


s_add_u32 s8, s18, s0 // (op 562) 


s_cmp_eq_i32 s9, 0 // (op 562) 
s_mov_b32 s0, scc // (op 562) 

s_cmp_eq_i32 s0, 0 // (op 562) 
s_cselect_b32 s9, s8, s1 // (op 562) 



v_lshl_add_u32 v10, s9, 4, v12 // (op 562) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 562) 
s_lshr_b32 s51, s0, 1 // (op 562) 

v_lshl_add_u32 v12, s51, 3, v11 // (op 562) 


v_mul_lo_u32 v11, v10, s28 // (op 562) 
v_add_u32 v11, v11, v12 // (op 562) 


s_mul_i32 s1, s28, s7 // (op 564) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 564) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 564) least significant
s_mul_hi_u32 s8, s28, s6 // (op 564) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 564) most significant: sum
s_add_u32 s1, s1, s8 // (op 564) most significant: sum


s_mov_b32 s60, s30 // (op 564) reg expression
s_mov_b32 s61, s31 // (op 564) reg expression
s_mov_b32 s62, s0 // (op 564) reg expression
s_mov_b32 s63, 131072 // (op 564) call()

s_mul_hi_i32 s51, s16, s11 // (op 626) 
v_and_b32 v10, -32, v1 // (op 626) 
s_add_i32 s0, s51, s16 // (op 626) 

v_sub_u32 v12, v1, v10 // (op 626) 

s_ashr_i32 s1, s0, 31 // (op 626) 
v_lshrrev_b32 v10, 5, v1 // (op 626) 
s_cmp_eq_i32 s11, 0 // (op 626) 
s_mov_b32 s51, scc // (op 626) 
s_cmp_eq_i32 s51, 0 // (op 626) 
s_cselect_b32 s8, 0, -1 // (op 626) 

s_and_b32 s9, s12, 31 // (op 626) 
s_lshl_b32 s51, 1, s9 // (op 626) 
s_add_i32 s18, s51, s8 // (op 626) 


s_and_b32 s8, s1, s18 // (op 626) 


s_add_i32 s1, s0, s8 // (op 626) 


s_ashr_i32 s0, s1, s9 // (op 626) 


s_lshl_b32 s1, s12, 24 // (op 626) 
s_ashr_i32 s51, s1, 31 // (op 626) 

s_xor_b32 s1, s0, s51 // (op 626) 

s_sub_i32 s0, s1, s51 // (op 626) 


s_mul_hi_i32 s1, s0, s13 // (op 626) 
s_add_i32 s51, s1, s0 // (op 626) 

s_ashr_i32 s1, s51, 31 // (op 626) 
s_cmp_eq_i32 s13, 0 // (op 626) 
s_mov_b32 s8, scc // (op 626) 
s_cmp_eq_i32 s8, 0 // (op 626) 
s_cselect_b32 s9, 0, -1 // (op 626) 

s_and_b32 s8, s14, 31 // (op 626) 
s_lshl_b32 s18, 1, s8 // (op 626) 
s_add_i32 s19, s18, s9 // (op 626) 


s_and_b32 s9, s1, s19 // (op 626) 


s_add_i32 s1, s51, s9 // (op 626) 


s_ashr_i32 s9, s1, s8 // (op 626) 


s_lshl_b32 s1, s14, 24 // (op 626) 
s_ashr_i32 s8, s1, 31 // (op 626) 

s_xor_b32 s1, s9, s8 // (op 626) 

s_sub_i32 s9, s1, s8 // (op 626) 


s_add_u32 s18, s2, 32 // (op 626) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 626) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 626) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 626) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 626) 

s_mul_i32 s1, s9, s18 // (op 626) 

s_sub_i32 s51, s0, s1 // (op 626) 


s_mul_hi_i32 s0, s16, s10 // (op 626) 
s_add_i32 s1, s0, s16 // (op 626) 

s_ashr_i32 s0, s1, 31 // (op 626) 
s_cmp_eq_i32 s10, 0 // (op 626) 
s_mov_b32 s8, scc // (op 626) 
s_cmp_eq_i32 s8, 0 // (op 626) 
s_cselect_b32 s9, 0, -1 // (op 626) 

s_and_b32 s8, s49, 31 // (op 626) 
s_lshl_b32 s68, 1, s8 // (op 626) 
s_add_i32 s69, s68, s9 // (op 626) 


s_and_b32 s9, s0, s69 // (op 626) 


s_add_i32 s0, s1, s9 // (op 626) 


s_ashr_i32 s1, s0, s8 // (op 626) 


s_lshl_b32 s0, s49, 24 // (op 626) 
s_ashr_i32 s8, s0, 31 // (op 626) 

s_xor_b32 s0, s1, s8 // (op 626) 

s_sub_i32 s1, s0, s8 // (op 626) 


s_mul_i32 s0, s46, s18 // (op 626) 

s_mul_i32 s8, s50, s0 // (op 626) 

s_mul_i32 s0, s1, s8 // (op 626) 

s_sub_u32 s8, s16, s0 // (op 626) 

s_mul_hi_i32 s0, s8, s15 // (op 626) 
s_add_i32 s9, s0, s8 // (op 626) 


s_ashr_i32 s0, s9, 31 // (op 626) 
s_cmp_eq_i32 s15, 0 // (op 626) 
s_mov_b32 s8, scc // (op 626) 
s_cmp_eq_i32 s8, 0 // (op 626) 
s_cselect_b32 s18, 0, -1 // (op 626) 

s_and_b32 s8, s48, 31 // (op 626) 
s_lshl_b32 s19, 1, s8 // (op 626) 
s_add_i32 s68, s19, s18 // (op 626) 


s_and_b32 s18, s0, s68 // (op 626) 


s_add_i32 s0, s9, s18 // (op 626) 


s_ashr_i32 s9, s0, s8 // (op 626) 


s_lshl_b32 s0, s48, 24 // (op 626) 
s_ashr_i32 s8, s0, 31 // (op 626) 

s_xor_b32 s0, s9, s8 // (op 626) 

s_sub_i32 s9, s0, s8 // (op 626) 


s_cmp_eq_i32 s1, 0 // (op 626) 
s_mov_b32 s0, scc // (op 626) 

s_cmp_eq_i32 s0, 0 // (op 626) 
s_cselect_b32 s1, s9, s51 // (op 626) 



v_lshl_add_u32 v13, s1, 5, v12 // (op 626) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 626) 
s_lshr_b32 s51, s0, 1 // (op 626) 

v_lshl_add_u32 v12, s51, 3, v10 // (op 626) 


v_mul_lo_u32 v10, v12, s32 // (op 626) 
v_add_u32 v10, v10, v13 // (op 626) 


s_mul_i32 s1, s32, s37 // (op 628) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 628) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 628) least significant
s_mul_hi_u32 s8, s32, s36 // (op 628) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 628) most significant: sum
s_add_u32 s1, s1, s8 // (op 628) most significant: sum


s_mov_b32 s64, s34 // (op 628) reg expression
s_mov_b32 s65, s35 // (op 628) reg expression
s_mov_b32 s66, s0 // (op 628) reg expression
s_mov_b32 s67, 131072 // (op 628) call()

v_and_b32 v12, -8, v1 // (op 719) 
v_sub_u32 v13, v1, v12 // (op 719) 

v_lshlrev_b32 v12, 4, v13 // (op 719) 

v_lshrrev_b32 v13, 3, v1 // (op 719) 
v_lshl_add_u32 v14, v13, 7, v12 // (op 719) 


v_mul_lo_u32 v15, v14, 6 // (op 719) 

v_lshrrev_b32 v12, 3, v15 // (op 719) 

v_lshlrev_b32 v13, 5, v1 // (op 772) 
v_mul_lo_u32 v14, v13, 6 // (op 772) 
v_lshrrev_b32 v15, 3, v14 // (op 772) 

v_lshrrev_b32 v14, 4, v13 // (op 772) 

v_lshlrev_b32 v13, 2, v14 // (op 772) 

v_add_u32 v14, v15, v13 // (op 772) 


v_and_b32 v13, -32, v1 // (op 825) 
v_sub_u32 v15, v1, v13 // (op 825) 

v_and_b32 v13, 4294967264, v1 // (op 825) 
v_add_u32 v17, v13, v15 // (op 825) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 166) 
s_and_b32 s51, s0, 4294967294 // (op 166) 

s_ashr_i64 s[0:1], s[26:27], 7 // (op 165) 
s_mov_b32 s8, s0 // (op 165) convert

s_cmp_lt_u32 s51, s8 // (op 165) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165:


s_add_u32 s51, s51, 1 // (op 167) 
v_mov_b32 v13, v35 // (op 170) 

buffer_load_dwordx3 v[18:20], v13, s[52:55], 0 offen // (op 170) Load value

v_add_u32 v35, 96, v35 // (op 421) 


s_barrier  // (op 998) 
v_mov_b32 v13, v39 // (op 183) 



v_mov_b32 v15, v13 // (op 183) reg expression

buffer_load_dwordx3 v[22:24], v15, s[56:59], 0 offen // (op 183) Load value
buffer_load_dwordx3 v[26:28], v15, s[56:59], 0 offen offset:12 // (op 183) Load value


s_lshl_b64 s[0:1], s[22:23], 7 // (op 490) 
s_mul_i32 s9, s1, 6 // (op 490) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 490) least significant
s_mul_hi_u32 s18, s0, 6 // (op 490) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 490) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 490) 

v_add_u32 v39, v39, s0 // (op 490) 

s_waitcnt vmcnt(2)
ds_write_b96 v12, v[18:20] offset:4224 // (op 171) Store local data 



v_mov_b32 v13, v10 // (op 184) 
s_barrier  // (op 1011) 
v_mov_b32 v25, v13 // (op 184) reg expression




buffer_load_ubyte v15, v25, s[64:67], 0 offen offset: 0 // (op 184) Load value


s_lshl_b32 s0, s32, 2 // (op 618) 
v_add_u32 v10, v10, s0 // (op 618) 

s_waitcnt vmcnt(2)
ds_write_b96 v14, v[22:24] // (op 181) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1001) 
s_waitcnt vmcnt(1)
ds_write_b96 v14, v[26:28] offset:16 // (op 181) Store local data 




ds_read_b128 v[18:21], v3 offset:4224 // (op 173) Load local data 
s_barrier  // (op 1025) 
ds_read_b64 v[22:23], v3 offset:4240 // (op 173) Load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 1015) 
s_waitcnt vmcnt(0)
ds_write_b8 v17, v15 offset:4096 // (op 182) Store local data 




ds_read_b96_tr_b6 v[24:26], v0 // (op 175) Transpose load local data 
ds_read_b96_tr_b6 v[27:29], v0 offset:512 // (op 175) Transpose load local data 


v_mov_b32 v13, v11 // (op 177) 

buffer_load_ubyte v15, v13, s[60:63], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v11, 4, v11 // (op 554) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1029) 


ds_read_u8 v13, v4 offset:4096 // (op 179) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:23], v[24:29], a[0:3], v15, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 180) 




s_ashr_i64 s[0:1], s[26:27], 7 // (op 165) 
s_mov_b32 s8, s0 // (op 165) convert

s_cmp_lt_u32 s51, s8 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165 if true)
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165:






















s_barrier  // (op 1097) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

v_lshrrev_b32 v4, 16, v10 // (op 52) 
v_lshrrev_b32 v5, 16, v11 // (op 52) 
v_lshrrev_b32 v6, 16, v12 // (op 52) 
v_lshrrev_b32 v7, 16, v13 // (op 52) 

s_mul_hi_i32 s0, s16, s11 // (op 867) 
v_lshrrev_b32 v0, 7, v1 // (op 867) 
s_add_i32 s1, s0, s16 // (op 867) 

v_and_b32 v3, 63, v1 // (op 867) 
s_ashr_i32 s0, s1, 31 // (op 867) 
v_lshrrev_b32 v10, 4, v3 // (op 867) 
s_cmp_eq_i32 s11, 0 // (op 867) 
s_mov_b32 s4, scc // (op 867) 
v_lshlrev_b32 v11, 2, v10 // (op 867) 

s_cmp_eq_i32 s4, 0 // (op 867) 
s_cselect_b32 s5, 0, -1 // (op 867) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 867) 


s_and_b32 s4, s12, 31 // (op 867) 
v_lshrrev_b32 v0, 2, v3 // (op 867) 

s_lshl_b32 s8, 1, s4 // (op 867) 
v_and_b32 v3, 3, v0 // (op 867) 

s_add_i32 s9, s8, s5 // (op 867) 


v_and_b32 v0, 3, v1 // (op 867) 
s_and_b32 s5, s0, s9 // (op 867) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 867) 


s_add_i32 s0, s1, s5 // (op 867) 


v_lshrrev_b32 v0, 6, v1 // (op 867) 
s_ashr_i32 s1, s0, s4 // (op 867) 


v_and_b32 v3, 1, v0 // (op 867) 

s_lshl_b32 s0, s12, 24 // (op 867) 
v_lshl_add_u32 v0, v3, 4, v11 // (op 867) 


s_ashr_i32 s4, s0, 31 // (op 867) 

s_xor_b32 s0, s1, s4 // (op 867) 

s_sub_i32 s1, s0, s4 // (op 867) 


s_mul_hi_i32 s0, s1, s13 // (op 867) 
s_add_i32 s4, s0, s1 // (op 867) 

s_ashr_i32 s0, s4, 31 // (op 867) 
s_cmp_eq_i32 s13, 0 // (op 867) 
s_mov_b32 s5, scc // (op 867) 
s_cmp_eq_i32 s5, 0 // (op 867) 
s_cselect_b32 s8, 0, -1 // (op 867) 

s_and_b32 s5, s14, 31 // (op 867) 
s_lshl_b32 s9, 1, s5 // (op 867) 
s_add_i32 s18, s9, s8 // (op 867) 


s_and_b32 s8, s0, s18 // (op 867) 


s_add_i32 s0, s4, s8 // (op 867) 


s_ashr_i32 s4, s0, s5 // (op 867) 


s_lshl_b32 s0, s14, 24 // (op 867) 
s_ashr_i32 s5, s0, 31 // (op 867) 

s_xor_b32 s0, s4, s5 // (op 867) 

s_sub_i32 s4, s0, s5 // (op 867) 


s_mul_i32 s0, s1, s46 // (op 867) 
s_sub_u32 s5, s16, s0 // (op 867) 

s_mul_i32 s0, s4, s46 // (op 867) 
s_add_u32 s0, s0, s5 // (op 867) 

s_mul_hi_i32 s5, s16, s10 // (op 867) 
s_add_i32 s8, s5, s16 // (op 867) 

s_ashr_i32 s5, s8, 31 // (op 867) 
s_cmp_eq_i32 s10, 0 // (op 867) 
s_mov_b32 s9, scc // (op 867) 
s_cmp_eq_i32 s9, 0 // (op 867) 
s_cselect_b32 s18, 0, -1 // (op 867) 

s_and_b32 s9, s49, 31 // (op 867) 
s_lshl_b32 s19, 1, s9 // (op 867) 
s_add_i32 s20, s19, s18 // (op 867) 


s_and_b32 s18, s5, s20 // (op 867) 


s_add_i32 s5, s8, s18 // (op 867) 


s_ashr_i32 s8, s5, s9 // (op 867) 


s_lshl_b32 s5, s49, 24 // (op 867) 
s_ashr_i32 s9, s5, 31 // (op 867) 

s_xor_b32 s5, s8, s9 // (op 867) 

s_sub_i32 s8, s5, s9 // (op 867) 


s_add_u32 s18, s2, 32 // (op 867) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 867) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 867) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 867) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 867) 

s_mul_i32 s5, s46, s18 // (op 867) 
s_mul_i32 s9, s50, s5 // (op 867) 

s_mul_i32 s5, s8, s9 // (op 867) 

s_sub_u32 s9, s16, s5 // (op 867) 

s_mov_b32 s5, s9 // (op 867) 
s_mul_hi_i32 s20, s5, s15 // (op 867) 
s_add_i32 s21, s20, s5 // (op 867) 


s_ashr_i32 s5, s21, 31 // (op 867) 
s_cmp_eq_i32 s15, 0 // (op 867) 
s_mov_b32 s20, scc // (op 867) 
s_cmp_eq_i32 s20, 0 // (op 867) 
s_cselect_b32 s22, 0, -1 // (op 867) 

s_and_b32 s20, s48, 31 // (op 867) 
s_lshl_b32 s23, 1, s20 // (op 867) 
s_add_i32 s24, s23, s22 // (op 867) 


s_and_b32 s22, s5, s24 // (op 867) 


s_add_i32 s5, s21, s22 // (op 867) 


s_ashr_i32 s21, s5, s20 // (op 867) 


s_lshl_b32 s5, s48, 24 // (op 867) 
s_ashr_i32 s20, s5, 31 // (op 867) 

s_xor_b32 s5, s21, s20 // (op 867) 

s_sub_i32 s21, s5, s20 // (op 867) 


s_mul_i32 s5, s47, s46 // (op 867) 
s_add_u32 s22, s6, 16 // (op 867) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 867) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 867) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 867) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 867) 

s_sub_i32 s20, s22, s5 // (op 867) 

s_mul_i32 s22, s21, s20 // (op 867) 

s_sub_u32 s20, s9, s22 // (op 867) 


s_add_u32 s9, s20, s5 // (op 867) 


s_cmp_eq_i32 s8, 0 // (op 867) 
s_mov_b32 s5, scc // (op 867) 

s_cmp_eq_i32 s5, 0 // (op 867) 
s_cselect_b32 s8, s9, s0 // (op 867) 


s_mul_i32 s0, s4, s18 // (op 867) 


v_lshl_add_u32 v3, s8, 4, v10 // (op 867) 


s_sub_i32 s4, s1, s0 // (op 867) 


s_cmp_eq_i32 s5, 0 // (op 867) 
s_cselect_b32 s0, s21, s4 // (op 867) 



v_lshl_add_u32 v10, s0, 5, v0 // (op 867) 


v_mul_lo_u32 v0, v10, s38 // (op 867) 

v_add_lshl_u32 v10, v3, v0, 1 // (op 867) 


s_mul_i32 s1, s38, s3 // (op 869) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 869) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 869) least significant
s_mul_hi_u32 s5, s38, s2 // (op 869) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 869) most significant: sum
s_add_u32 s1, s1, s5 // (op 869) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 1 // (op 869) 

s_mov_b32 s20, s40 // (op 869) reg expression
s_mov_b32 s21, s41 // (op 869) reg expression
s_mov_b32 s22, s4 // (op 869) reg expression
s_mov_b32 s23, 131072 // (op 869) call()


v_pack_b32_f16 v12, v4, v5 // (op 54) 
v_pack_b32_f16 v13, v6, v7 // (op 54) 


buffer_store_dwordx2 v[12:13], v10, s[20:23], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:
s_mul_hi_i32 s0, s16, s11 // (op 689) 
v_lshrrev_b32 v0, 7, v1 // (op 689) 
s_add_i32 s1, s0, s16 // (op 689) 

v_and_b32 v3, 63, v1 // (op 689) 
s_ashr_i32 s0, s1, 31 // (op 689) 
v_lshrrev_b32 v4, 4, v3 // (op 689) 
s_cmp_eq_i32 s11, 0 // (op 689) 
s_mov_b32 s4, scc // (op 689) 
v_lshlrev_b32 v5, 2, v4 // (op 689) 

s_cmp_eq_i32 s4, 0 // (op 689) 
s_cselect_b32 s5, 0, -1 // (op 689) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 689) 


s_and_b32 s4, s12, 31 // (op 689) 
v_lshrrev_b32 v0, 2, v3 // (op 689) 

s_lshl_b32 s8, 1, s4 // (op 689) 
v_and_b32 v3, 3, v0 // (op 689) 

s_add_i32 s9, s8, s5 // (op 689) 


v_and_b32 v0, 3, v1 // (op 689) 
s_and_b32 s5, s0, s9 // (op 689) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 689) 


s_add_i32 s0, s1, s5 // (op 689) 


v_lshrrev_b32 v0, 6, v1 // (op 689) 
s_ashr_i32 s1, s0, s4 // (op 689) 


v_and_b32 v3, 1, v0 // (op 689) 

s_lshl_b32 s0, s12, 24 // (op 689) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 689) 


s_ashr_i32 s4, s0, 31 // (op 689) 

s_xor_b32 s0, s1, s4 // (op 689) 

s_sub_i32 s1, s0, s4 // (op 689) 


s_mul_hi_i32 s0, s1, s13 // (op 689) 
s_add_i32 s4, s0, s1 // (op 689) 

s_ashr_i32 s0, s4, 31 // (op 689) 
s_cmp_eq_i32 s13, 0 // (op 689) 
s_mov_b32 s5, scc // (op 689) 
s_cmp_eq_i32 s5, 0 // (op 689) 
s_cselect_b32 s8, 0, -1 // (op 689) 

s_and_b32 s5, s14, 31 // (op 689) 
s_lshl_b32 s9, 1, s5 // (op 689) 
s_add_i32 s18, s9, s8 // (op 689) 


s_and_b32 s19, s0, s18 // (op 689) 


s_add_i32 s0, s4, s19 // (op 689) 


s_ashr_i32 s4, s0, s5 // (op 689) 


s_lshl_b32 s0, s14, 24 // (op 689) 
s_ashr_i32 s5, s0, 31 // (op 689) 

s_xor_b32 s0, s4, s5 // (op 689) 

s_sub_i32 s4, s0, s5 // (op 689) 


s_mul_i32 s0, s1, s46 // (op 689) 
s_sub_u32 s5, s16, s0 // (op 689) 

s_mul_i32 s0, s4, s46 // (op 689) 
s_add_u32 s0, s0, s5 // (op 689) 

s_mul_hi_i32 s5, s16, s10 // (op 689) 
s_add_i32 s8, s5, s16 // (op 689) 

s_ashr_i32 s5, s8, 31 // (op 689) 
s_cmp_eq_i32 s10, 0 // (op 689) 
s_mov_b32 s9, scc // (op 689) 
s_cmp_eq_i32 s9, 0 // (op 689) 
s_cselect_b32 s18, 0, -1 // (op 689) 

s_and_b32 s9, s49, 31 // (op 689) 
s_lshl_b32 s19, 1, s9 // (op 689) 
s_add_i32 s24, s19, s18 // (op 689) 


s_and_b32 s18, s5, s24 // (op 689) 


s_add_i32 s5, s8, s18 // (op 689) 


s_ashr_i32 s8, s5, s9 // (op 689) 


s_lshl_b32 s5, s49, 24 // (op 689) 
s_ashr_i32 s9, s5, 31 // (op 689) 

s_xor_b32 s5, s8, s9 // (op 689) 

s_sub_i32 s8, s5, s9 // (op 689) 


s_add_u32 s18, s2, 32 // (op 689) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 689) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 689) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 689) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 5 // (op 689) 

s_mul_i32 s5, s46, s18 // (op 689) 
s_mul_i32 s9, s50, s5 // (op 689) 

s_mul_i32 s5, s8, s9 // (op 689) 

s_sub_u32 s9, s16, s5 // (op 689) 

s_mov_b32 s5, s9 // (op 689) 
s_mul_hi_i32 s24, s5, s15 // (op 689) 
s_add_i32 s25, s24, s5 // (op 689) 


s_ashr_i32 s5, s25, 31 // (op 689) 
s_cmp_eq_i32 s15, 0 // (op 689) 
s_mov_b32 s24, scc // (op 689) 
s_cmp_eq_i32 s24, 0 // (op 689) 
s_cselect_b32 s26, 0, -1 // (op 689) 

s_and_b32 s24, s48, 31 // (op 689) 
s_lshl_b32 s27, 1, s24 // (op 689) 
s_add_i32 s28, s27, s26 // (op 689) 


s_and_b32 s26, s5, s28 // (op 689) 


s_add_i32 s5, s25, s26 // (op 689) 


s_ashr_i32 s25, s5, s24 // (op 689) 


s_lshl_b32 s5, s48, 24 // (op 689) 
s_ashr_i32 s24, s5, 31 // (op 689) 

s_xor_b32 s5, s25, s24 // (op 689) 

s_sub_i32 s25, s5, s24 // (op 689) 


s_mul_i32 s5, s47, s46 // (op 689) 
s_add_u32 s26, s6, 16 // (op 689) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 689) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 689) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 689) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 689) 

s_sub_i32 s24, s26, s5 // (op 689) 

s_mul_i32 s26, s25, s24 // (op 689) 

s_sub_u32 s24, s9, s26 // (op 689) 


s_add_u32 s9, s24, s5 // (op 689) 


s_cmp_eq_i32 s8, 0 // (op 689) 
s_mov_b32 s5, scc // (op 689) 

s_cmp_eq_i32 s5, 0 // (op 689) 
s_cselect_b32 s8, s9, s0 // (op 689) 


s_mul_i32 s0, s4, s18 // (op 689) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 689) 


s_sub_i32 s4, s1, s0 // (op 689) 


s_cmp_eq_i32 s5, 0 // (op 689) 
s_cselect_b32 s24, s25, s4 // (op 689) 



v_lshl_add_u32 v4, s24, 5, v0 // (op 689) 


v_mul_lo_u32 v0, v4, s42 // (op 689) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 689) 


s_mul_i32 s1, s42, s3 // (op 691) most significant: low of low * high
s_mul_i32 s4, s43, s2 // (op 691) most significant: low of high * low
s_mul_i32 s0, s42, s2 // (op 691) least significant
s_mul_hi_u32 s5, s42, s2 // (op 691) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 691) most significant: sum
s_add_u32 s1, s1, s5 // (op 691) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 1 // (op 691) 

s_mov_b32 s24, s44 // (op 691) reg expression
s_mov_b32 s25, s45 // (op 691) reg expression
s_mov_b32 s26, s4 // (op 691) reg expression
s_mov_b32 s27, 131072 // (op 691) call()





buffer_load_dwordx2 v[6:7], v4, s[24:27], 0 offen // (op 138) Load value



v_accvgpr_read v10, a0 // (op 134) 
v_accvgpr_read v11, a1 // (op 134) 
v_accvgpr_read v12, a2 // (op 134) 
v_accvgpr_read v13, a3 // (op 134) 
v_mul_f32 v14, v9, v10 // (op 134) 
v_mul_f32 v15, v9, v11 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 

s_waitcnt vmcnt(0)
v_and_b32 v4, 65535, v6 // (op 139) 
v_lshrrev_b32 v5, 16, v6 // (op 139) 
v_lshlrev_b32 v4, 16, v4 // (op 139) 
v_lshlrev_b32 v5, 16, v5 // (op 139) 
v_mul_f32 v10, v8, v4 // (op 139) 
v_mul_f32 v11, v8, v5 // (op 139) 
v_and_b32 v4, 65535, v7 // (op 139) 
v_lshrrev_b32 v5, 16, v7 // (op 139) 
v_lshlrev_b32 v4, 16, v4 // (op 139) 
v_lshlrev_b32 v5, 16, v5 // (op 139) 
v_mul_f32 v12, v8, v4 // (op 139) 
v_mul_f32 v13, v8, v5 // (op 139) 


v_add_f32 v4, v10, v14 // (op 135) 
v_add_f32 v5, v11, v15 // (op 135) 
v_add_f32 v6, v12, v16 // (op 135) 
v_add_f32 v7, v13, v17 // (op 135) 


v_lshrrev_b32 v10, 16, v4 // (op 136) 
v_lshrrev_b32 v11, 16, v5 // (op 136) 
v_lshrrev_b32 v12, 16, v6 // (op 136) 
v_lshrrev_b32 v13, 16, v7 // (op 136) 

s_mul_hi_i32 s0, s16, s11 // (op 884) 
v_lshrrev_b32 v0, 7, v1 // (op 884) 
s_add_i32 s1, s0, s16 // (op 884) 

v_and_b32 v3, 63, v1 // (op 884) 
s_ashr_i32 s0, s1, 31 // (op 884) 
v_lshrrev_b32 v4, 4, v3 // (op 884) 
s_cmp_eq_i32 s11, 0 // (op 884) 
s_mov_b32 s4, scc // (op 884) 
v_lshlrev_b32 v5, 2, v4 // (op 884) 

s_cmp_eq_i32 s4, 0 // (op 884) 
s_cselect_b32 s5, 0, -1 // (op 884) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 884) 


s_and_b32 s4, s12, 31 // (op 884) 
v_lshrrev_b32 v0, 2, v3 // (op 884) 

s_lshl_b32 s8, 1, s4 // (op 884) 
v_and_b32 v3, 3, v0 // (op 884) 

s_add_i32 s9, s8, s5 // (op 884) 


v_and_b32 v0, 3, v1 // (op 884) 
s_and_b32 s5, s0, s9 // (op 884) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 884) 


s_add_i32 s0, s1, s5 // (op 884) 


v_lshrrev_b32 v0, 6, v1 // (op 884) 
s_ashr_i32 s1, s0, s4 // (op 884) 


v_and_b32 v3, 1, v0 // (op 884) 

s_lshl_b32 s0, s12, 24 // (op 884) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 884) 


s_ashr_i32 s4, s0, 31 // (op 884) 

s_xor_b32 s0, s1, s4 // (op 884) 

s_sub_i32 s1, s0, s4 // (op 884) 


s_mul_hi_i32 s0, s1, s13 // (op 884) 
s_add_i32 s4, s0, s1 // (op 884) 

s_ashr_i32 s0, s4, 31 // (op 884) 
s_cmp_eq_i32 s13, 0 // (op 884) 
s_mov_b32 s5, scc // (op 884) 
s_cmp_eq_i32 s5, 0 // (op 884) 
s_cselect_b32 s8, 0, -1 // (op 884) 

s_and_b32 s5, s14, 31 // (op 884) 
s_lshl_b32 s9, 1, s5 // (op 884) 
s_add_i32 s18, s9, s8 // (op 884) 


s_and_b32 s19, s0, s18 // (op 884) 


s_add_i32 s0, s4, s19 // (op 884) 


s_ashr_i32 s4, s0, s5 // (op 884) 


s_lshl_b32 s0, s14, 24 // (op 884) 
s_ashr_i32 s5, s0, 31 // (op 884) 

s_xor_b32 s0, s4, s5 // (op 884) 

s_sub_i32 s4, s0, s5 // (op 884) 


s_mul_i32 s0, s1, s46 // (op 884) 
s_sub_u32 s5, s16, s0 // (op 884) 

s_mul_i32 s0, s4, s46 // (op 884) 
s_add_u32 s0, s0, s5 // (op 884) 

s_mul_hi_i32 s5, s16, s10 // (op 884) 
s_add_i32 s8, s5, s16 // (op 884) 

s_ashr_i32 s5, s8, 31 // (op 884) 
s_cmp_eq_i32 s10, 0 // (op 884) 
s_mov_b32 s9, scc // (op 884) 
s_cmp_eq_i32 s9, 0 // (op 884) 
s_cselect_b32 s18, 0, -1 // (op 884) 

s_and_b32 s9, s49, 31 // (op 884) 
s_lshl_b32 s19, 1, s9 // (op 884) 
s_add_i32 s24, s19, s18 // (op 884) 


s_and_b32 s18, s5, s24 // (op 884) 


s_add_i32 s5, s8, s18 // (op 884) 


s_ashr_i32 s8, s5, s9 // (op 884) 


s_lshl_b32 s5, s49, 24 // (op 884) 
s_ashr_i32 s9, s5, 31 // (op 884) 

s_xor_b32 s5, s8, s9 // (op 884) 

s_sub_i32 s8, s5, s9 // (op 884) 


s_add_u32 s18, s2, 32 // (op 884) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 884) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 884) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 884) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 5 // (op 884) 

s_mul_i32 s5, s46, s18 // (op 884) 
s_mul_i32 s9, s50, s5 // (op 884) 

s_mul_i32 s5, s8, s9 // (op 884) 

s_sub_u32 s9, s16, s5 // (op 884) 

s_mov_b32 s5, s9 // (op 884) 
s_mul_hi_i32 s24, s5, s15 // (op 884) 
s_add_i32 s25, s24, s5 // (op 884) 


s_ashr_i32 s5, s25, 31 // (op 884) 
s_cmp_eq_i32 s15, 0 // (op 884) 
s_mov_b32 s24, scc // (op 884) 
s_cmp_eq_i32 s24, 0 // (op 884) 
s_cselect_b32 s26, 0, -1 // (op 884) 

s_and_b32 s24, s48, 31 // (op 884) 
s_lshl_b32 s27, 1, s24 // (op 884) 
s_add_i32 s28, s27, s26 // (op 884) 


s_and_b32 s26, s5, s28 // (op 884) 


s_add_i32 s5, s25, s26 // (op 884) 


s_ashr_i32 s25, s5, s24 // (op 884) 


s_lshl_b32 s5, s48, 24 // (op 884) 
s_ashr_i32 s24, s5, 31 // (op 884) 

s_xor_b32 s5, s25, s24 // (op 884) 

s_sub_i32 s25, s5, s24 // (op 884) 


s_mul_i32 s5, s47, s46 // (op 884) 
s_add_u32 s26, s6, 16 // (op 884) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 884) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 884) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 884) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 884) 

s_sub_i32 s24, s26, s5 // (op 884) 

s_mul_i32 s26, s25, s24 // (op 884) 

s_sub_u32 s24, s9, s26 // (op 884) 


s_add_u32 s9, s24, s5 // (op 884) 


s_cmp_eq_i32 s8, 0 // (op 884) 
s_mov_b32 s5, scc // (op 884) 

s_cmp_eq_i32 s5, 0 // (op 884) 
s_cselect_b32 s8, s9, s0 // (op 884) 


s_mul_i32 s0, s4, s18 // (op 884) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 884) 


s_sub_i32 s4, s1, s0 // (op 884) 


s_cmp_eq_i32 s5, 0 // (op 884) 
s_cselect_b32 s24, s25, s4 // (op 884) 



v_lshl_add_u32 v4, s24, 5, v0 // (op 884) 


v_mul_lo_u32 v0, v4, s38 // (op 884) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 884) 


s_mul_i32 s1, s38, s3 // (op 886) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 886) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 886) least significant
s_mul_hi_u32 s5, s38, s2 // (op 886) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 886) most significant: sum
s_add_u32 s1, s1, s5 // (op 886) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 1 // (op 886) 

s_mov_b32 s20, s40 // (op 886) reg expression
s_mov_b32 s21, s41 // (op 886) reg expression
s_mov_b32 s22, s4 // (op 886) reg expression
s_mov_b32 s23, 131072 // (op 886) call()


v_pack_b32_f16 v6, v10, v11 // (op 137) 
v_pack_b32_f16 v7, v12, v13 // (op 137) 


buffer_store_dwordx2 v[6:7], v4, s[20:23], 0 offen // (op 137) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.LRR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end:
.size RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_, .LRR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end-RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 11520 // lds bytes
  .amdhsa_accum_offset 60
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
    DataTypeB: F6
    DestDataType: b
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
    macrotile: [16, 32, 128]
    threads: [128, 1, 1]
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
  - .name: RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 11520
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 59
    .agpr_count: 4
    .max_flat_workgroup_size: 128
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

