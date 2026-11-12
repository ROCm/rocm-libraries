// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_,@function
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 853) 
v_and_b32 v3, 15, v1 // (op 853) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 853) 


v_and_b32 v0, 63, v1 // (op 853) 
v_lshrrev_b32 v3, 4, v0 // (op 853) 

v_lshlrev_b32 v0, 5, v3 // (op 853) 

v_lshl_add_u32 v3, v4, 8, v0 // (op 853) 


v_lshlrev_b32 v0, 2, v3 // (op 853) 

v_lshrrev_b32 v3, 3, v0 // (op 853) 

v_and_b32 v0, 63, v1 // (op 906) 
v_and_b32 v4, 48, v1 // (op 906) 
v_sub_u32 v5, v0, v4 // (op 906) 

v_lshrrev_b32 v4, 4, v0 // (op 906) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 906) 


v_lshlrev_b32 v4, 4, v0 // (op 906) 
v_mul_lo_u32 v5, v4, 6 // (op 906) 

v_lshrrev_b32 v4, 3, v5 // (op 906) 

v_and_b32 v5, 268435455, v0 // (op 906) 

v_lshlrev_b32 v0, 2, v5 // (op 906) 

v_add_u32 v5, v4, v0 // (op 906) 


v_lshrrev_b32 v0, 6, v1 // (op 959) 
v_and_b32 v4, 15, v1 // (op 959) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 959) 


v_and_b32 v0, 63, v1 // (op 959) 
v_lshrrev_b32 v4, 4, v0 // (op 959) 

v_lshl_add_u32 v0, v6, 3, v4 // (op 959) 


v_and_b32 v4, 48, v1 // (op 1012) 
v_and_b32 v6, 15, v1 // (op 1012) 
v_add_u32 v7, v4, v6 // (op 1012) 


v_lshrrev_b32 v4, 6, v1 // (op 1070) 
v_and_b32 v6, 15, v1 // (op 1070) 
v_lshl_add_u32 v8, v4, 4, v6 // (op 1070) 


v_and_b32 v4, 63, v1 // (op 1070) 
v_lshrrev_b32 v6, 4, v4 // (op 1070) 

v_lshlrev_b32 v4, 5, v6 // (op 1070) 

v_lshl_add_u32 v6, v8, 8, v4 // (op 1070) 


v_lshlrev_b32 v4, 2, v6 // (op 1070) 

v_lshrrev_b32 v6, 3, v4 // (op 1070) 

v_and_b32 v4, 63, v1 // (op 1094) 
v_and_b32 v8, 48, v1 // (op 1094) 
v_sub_u32 v9, v4, v8 // (op 1094) 

v_lshrrev_b32 v8, 4, v4 // (op 1094) 

v_lshl_add_u32 v4, v8, 5, v9 // (op 1094) 


v_lshlrev_b32 v8, 4, v4 // (op 1094) 
v_mul_lo_u32 v9, v8, 6 // (op 1094) 

v_lshrrev_b32 v8, 3, v9 // (op 1094) 

v_and_b32 v9, 268435455, v4 // (op 1094) 

v_lshlrev_b32 v4, 2, v9 // (op 1094) 

v_add_u32 v9, v8, v4 // (op 1094) 


v_lshrrev_b32 v4, 6, v1 // (op 1118) 
v_and_b32 v8, 15, v1 // (op 1118) 
v_lshl_add_u32 v10, v4, 4, v8 // (op 1118) 


v_and_b32 v4, 63, v1 // (op 1118) 
v_lshrrev_b32 v8, 4, v4 // (op 1118) 

v_lshl_add_u32 v4, v10, 3, v8 // (op 1118) 


v_and_b32 v8, 48, v1 // (op 1142) 
v_and_b32 v10, 15, v1 // (op 1142) 
v_add_u32 v11, v8, v10 // (op 1142) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v10, s9 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
s_mul_hi_i32 s0, s16, s11 // (op 563) 
v_lshrrev_b32 v12, 3, v1 // (op 563) 
s_add_i32 s1, s0, s16 // (op 563) 

v_lshlrev_b32 v13, 1, v12 // (op 563) 

s_ashr_i32 s0, s1, 31 // (op 563) 
v_and_b32 v12, -8, v1 // (op 563) 
s_cmp_eq_i32 s11, 0 // (op 563) 
s_mov_b32 s8, scc // (op 563) 
v_sub_u32 v14, v1, v12 // (op 563) 

s_cmp_eq_i32 s8, 0 // (op 563) 
s_cselect_b32 s9, 0, -1 // (op 563) 

v_lshlrev_b32 v12, 5, v14 // (op 563) 

v_mov_b32 v15, 0 // (op 563) convert
v_mov_b32 v14, v12 // (op 563) convert

s_and_b32 s8, s12, 31 // (op 563) 
s_lshl_b32 s18, 1, s8 // (op 563) 
s_add_i32 s19, s18, s9 // (op 563) 


s_and_b32 s9, s0, s19 // (op 563) 


s_add_i32 s0, s1, s9 // (op 563) 


s_ashr_i32 s1, s0, s8 // (op 563) 


s_lshl_b32 s0, s12, 24 // (op 563) 
s_ashr_i32 s8, s0, 31 // (op 563) 

s_xor_b32 s0, s1, s8 // (op 563) 

s_sub_i32 s1, s0, s8 // (op 563) 


s_mul_hi_i32 s0, s1, s13 // (op 563) 
s_add_i32 s8, s0, s1 // (op 563) 

s_ashr_i32 s0, s8, 31 // (op 563) 
s_cmp_eq_i32 s13, 0 // (op 563) 
s_mov_b32 s9, scc // (op 563) 
s_cmp_eq_i32 s9, 0 // (op 563) 
s_cselect_b32 s18, 0, -1 // (op 563) 

s_and_b32 s9, s14, 31 // (op 563) 
s_lshl_b32 s19, 1, s9 // (op 563) 
s_add_i32 s51, s19, s18 // (op 563) 


s_and_b32 s18, s0, s51 // (op 563) 


s_add_i32 s0, s8, s18 // (op 563) 


s_ashr_i32 s8, s0, s9 // (op 563) 


s_lshl_b32 s0, s14, 24 // (op 563) 
s_ashr_i32 s9, s0, 31 // (op 563) 

s_xor_b32 s0, s8, s9 // (op 563) 

s_sub_i32 s8, s0, s9 // (op 563) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s46 // (op 563) 

s_sub_u32 s1, s16, s0 // (op 563) 

s_mul_i32 s0, s8, s46 // (op 563) 
s_add_u32 s0, s0, s1 // (op 563) 


s_mul_hi_i32 s1, s16, s10 // (op 563) 
s_add_i32 s8, s1, s16 // (op 563) 

s_ashr_i32 s1, s8, 31 // (op 563) 
s_cmp_eq_i32 s10, 0 // (op 563) 
s_mov_b32 s9, scc // (op 563) 
s_cmp_eq_i32 s9, 0 // (op 563) 
s_cselect_b32 s18, 0, -1 // (op 563) 

s_and_b32 s9, s49, 31 // (op 563) 
s_lshl_b32 s19, 1, s9 // (op 563) 
s_add_i32 s51, s19, s18 // (op 563) 


s_and_b32 s18, s1, s51 // (op 563) 


s_add_i32 s1, s8, s18 // (op 563) 


s_ashr_i32 s8, s1, s9 // (op 563) 


s_lshl_b32 s1, s49, 24 // (op 563) 
s_ashr_i32 s9, s1, 31 // (op 563) 

s_xor_b32 s1, s8, s9 // (op 563) 

s_sub_i32 s8, s1, s9 // (op 563) 


s_add_u32 s18, s2, 16 // (op 563) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 563) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 563) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 563) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 563) 

s_mul_i32 s1, s46, s18 // (op 563) 

s_mul_i32 s9, s50, s1 // (op 563) 

s_mul_i32 s1, s8, s9 // (op 563) 

s_sub_u32 s9, s16, s1 // (op 563) 

s_mov_b32 s1, s9 // (op 563) 
s_mul_hi_i32 s18, s1, s15 // (op 563) 
s_add_i32 s19, s18, s1 // (op 563) 


s_ashr_i32 s1, s19, 31 // (op 563) 
s_cmp_eq_i32 s15, 0 // (op 563) 
s_mov_b32 s18, scc // (op 563) 
s_cmp_eq_i32 s18, 0 // (op 563) 
s_cselect_b32 s51, 0, -1 // (op 563) 

s_and_b32 s18, s48, 31 // (op 563) 
s_lshl_b32 s52, 1, s18 // (op 563) 
s_add_i32 s53, s52, s51 // (op 563) 


s_and_b32 s51, s1, s53 // (op 563) 


s_add_i32 s1, s19, s51 // (op 563) 


s_ashr_i32 s19, s1, s18 // (op 563) 


s_lshl_b32 s1, s48, 24 // (op 563) 
s_ashr_i32 s18, s1, 31 // (op 563) 

s_xor_b32 s1, s19, s18 // (op 563) 

s_sub_i32 s19, s1, s18 // (op 563) 


s_mul_i32 s1, s47, s46 // (op 563) 
s_add_u32 s52, s6, 16 // (op 563) Start of Int64 add, locking SCC
s_addc_u32 s53, s7, 0 // (op 563) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 563) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 563) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 563) 

s_sub_i32 s18, s52, s1 // (op 563) 

s_mul_i32 s51, s19, s18 // (op 563) 


s_sub_u32 s18, s9, s51 // (op 563) 


s_add_u32 s9, s18, s1 // (op 563) 


s_cmp_eq_i32 s8, 0 // (op 563) 
s_mov_b32 s1, scc // (op 563) 

s_cmp_eq_i32 s1, 0 // (op 563) 
s_cselect_b32 s8, s9, s0 // (op 563) 



v_lshl_add_u32 v12, s8, 4, v13 // (op 563) 


v_mul_lo_u32 v17, v12, s5 // (op 563) most significant: low of low * high
v_mul_hi_u32 v13, v12, s4 // (op 563) most significant: high of low * low
v_mul_lo_u32 v16, v12, s4 // (op 563) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 563) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 563) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 563) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 563) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 563) 

v_mov_b32 v12, v14 // (op 563) convert

s_mul_i32 s1, s4, s7 // (op 565) most significant: low of low * high
s_mul_i32 s8, s5, s6 // (op 565) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 565) least significant
s_mul_hi_u32 s9, s4, s6 // (op 565) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 565) most significant: sum
s_add_u32 s1, s1, s9 // (op 565) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 565) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 565) 

s_mov_b32 s52, s20 // (op 565) reg expression
s_mov_b32 s53, s21 // (op 565) reg expression
s_mov_b32 s54, s0 // (op 565) reg expression
s_mov_b32 s55, 131072 // (op 565) call()

s_mul_hi_i32 s51, s16, s11 // (op 582) 
v_lshrrev_b32 v13, 3, v1 // (op 582) 
s_add_i32 s0, s51, s16 // (op 582) 

v_lshlrev_b32 v14, 1, v13 // (op 582) 

s_ashr_i32 s1, s0, 31 // (op 582) 
v_and_b32 v13, -8, v1 // (op 582) 
s_cmp_eq_i32 s11, 0 // (op 582) 
s_mov_b32 s51, scc // (op 582) 
v_sub_u32 v15, v1, v13 // (op 582) 

s_cmp_eq_i32 s51, 0 // (op 582) 
s_cselect_b32 s8, 0, -1 // (op 582) 

v_lshlrev_b32 v13, 5, v15 // (op 582) 

v_mov_b32 v17, 0 // (op 582) convert
v_mov_b32 v16, v13 // (op 582) convert

s_and_b32 s9, s12, 31 // (op 582) 
s_lshl_b32 s51, 1, s9 // (op 582) 
s_add_i32 s18, s51, s8 // (op 582) 


s_and_b32 s8, s1, s18 // (op 582) 


s_add_i32 s1, s0, s8 // (op 582) 


s_ashr_i32 s0, s1, s9 // (op 582) 


s_lshl_b32 s1, s12, 24 // (op 582) 
s_ashr_i32 s51, s1, 31 // (op 582) 

s_xor_b32 s1, s0, s51 // (op 582) 

s_sub_i32 s0, s1, s51 // (op 582) 


s_mul_hi_i32 s1, s0, s13 // (op 582) 
s_add_i32 s51, s1, s0 // (op 582) 

s_ashr_i32 s1, s51, 31 // (op 582) 
s_cmp_eq_i32 s13, 0 // (op 582) 
s_mov_b32 s8, scc // (op 582) 
s_cmp_eq_i32 s8, 0 // (op 582) 
s_cselect_b32 s9, 0, -1 // (op 582) 

s_and_b32 s8, s14, 31 // (op 582) 
s_lshl_b32 s18, 1, s8 // (op 582) 
s_add_i32 s19, s18, s9 // (op 582) 


s_and_b32 s9, s1, s19 // (op 582) 


s_add_i32 s1, s51, s9 // (op 582) 


s_ashr_i32 s9, s1, s8 // (op 582) 


s_lshl_b32 s1, s14, 24 // (op 582) 
s_ashr_i32 s8, s1, 31 // (op 582) 

s_xor_b32 s1, s9, s8 // (op 582) 

s_sub_i32 s9, s1, s8 // (op 582) 


s_mul_i32 s1, s0, s46 // (op 582) 

s_sub_u32 s0, s16, s1 // (op 582) 

s_mul_i32 s1, s9, s46 // (op 582) 
s_add_u32 s1, s1, s0 // (op 582) 


s_mul_hi_i32 s0, s16, s10 // (op 582) 
s_add_i32 s51, s0, s16 // (op 582) 

s_ashr_i32 s0, s51, 31 // (op 582) 
s_cmp_eq_i32 s10, 0 // (op 582) 
s_mov_b32 s8, scc // (op 582) 
s_cmp_eq_i32 s8, 0 // (op 582) 
s_cselect_b32 s9, 0, -1 // (op 582) 

s_and_b32 s8, s49, 31 // (op 582) 
s_lshl_b32 s18, 1, s8 // (op 582) 
s_add_i32 s19, s18, s9 // (op 582) 


s_and_b32 s9, s0, s19 // (op 582) 


s_add_i32 s0, s51, s9 // (op 582) 


s_ashr_i32 s9, s0, s8 // (op 582) 


s_lshl_b32 s0, s49, 24 // (op 582) 
s_ashr_i32 s8, s0, 31 // (op 582) 

s_xor_b32 s0, s9, s8 // (op 582) 

s_sub_i32 s9, s0, s8 // (op 582) 


s_add_u32 s18, s2, 16 // (op 582) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 582) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 582) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 582) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 582) 

s_mul_i32 s0, s46, s18 // (op 582) 

s_mul_i32 s8, s50, s0 // (op 582) 

s_mul_i32 s0, s9, s8 // (op 582) 

s_sub_u32 s8, s16, s0 // (op 582) 

s_mov_b32 s0, s8 // (op 582) 
s_mul_hi_i32 s51, s0, s15 // (op 582) 
s_add_i32 s18, s51, s0 // (op 582) 


s_ashr_i32 s0, s18, 31 // (op 582) 
s_cmp_eq_i32 s15, 0 // (op 582) 
s_mov_b32 s19, scc // (op 582) 
s_cmp_eq_i32 s19, 0 // (op 582) 
s_cselect_b32 s51, 0, -1 // (op 582) 

s_and_b32 s19, s48, 31 // (op 582) 
s_lshl_b32 s56, 1, s19 // (op 582) 
s_add_i32 s57, s56, s51 // (op 582) 


s_and_b32 s51, s0, s57 // (op 582) 


s_add_i32 s0, s18, s51 // (op 582) 


s_ashr_i32 s18, s0, s19 // (op 582) 


s_lshl_b32 s0, s48, 24 // (op 582) 
s_ashr_i32 s19, s0, 31 // (op 582) 

s_xor_b32 s0, s18, s19 // (op 582) 

s_sub_i32 s18, s0, s19 // (op 582) 


s_mul_i32 s0, s47, s46 // (op 582) 
s_add_u32 s56, s6, 16 // (op 582) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 582) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 582) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 582) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 582) 

s_sub_i32 s19, s56, s0 // (op 582) 

s_mul_i32 s51, s18, s19 // (op 582) 


s_sub_u32 s18, s8, s51 // (op 582) 


s_add_u32 s8, s18, s0 // (op 582) 


s_cmp_eq_i32 s9, 0 // (op 582) 
s_mov_b32 s0, scc // (op 582) 

s_cmp_eq_i32 s0, 0 // (op 582) 
s_cselect_b32 s9, s8, s1 // (op 582) 



v_lshl_add_u32 v13, s9, 4, v14 // (op 582) 


v_mul_lo_u32 v15, v13, s5 // (op 582) most significant: low of low * high
v_mul_hi_u32 v18, v13, s4 // (op 582) most significant: high of low * low
v_mul_lo_u32 v14, v13, s4 // (op 582) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 582) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v16 // (op 582) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 582) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 582) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 582) 

v_mov_b32 v13, v14 // (op 582) convert

s_mul_i32 s1, s4, s7 // (op 584) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 584) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 584) least significant
s_mul_hi_u32 s8, s4, s6 // (op 584) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 584) most significant: sum
s_add_u32 s1, s1, s8 // (op 584) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 584) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 584) 

s_mov_b32 s52, s20 // (op 584) reg expression
s_mov_b32 s53, s21 // (op 584) reg expression
s_mov_b32 s54, s0 // (op 584) reg expression
s_mov_b32 s55, 131072 // (op 584) call()

s_mul_hi_i32 s51, s16, s11 // (op 627) 
v_lshlrev_b32 v14, 2, v1 // (op 627) 
s_add_i32 s0, s51, s16 // (op 627) 

s_ashr_i32 s1, s0, 31 // (op 627) 
s_cmp_eq_i32 s11, 0 // (op 627) 
s_mov_b32 s51, scc // (op 627) 
s_cmp_eq_i32 s51, 0 // (op 627) 
s_cselect_b32 s8, 0, -1 // (op 627) 

s_and_b32 s9, s12, 31 // (op 627) 
s_lshl_b32 s51, 1, s9 // (op 627) 
s_add_i32 s18, s51, s8 // (op 627) 


s_and_b32 s8, s1, s18 // (op 627) 


s_add_i32 s1, s0, s8 // (op 627) 


s_ashr_i32 s0, s1, s9 // (op 627) 


s_lshl_b32 s1, s12, 24 // (op 627) 
s_ashr_i32 s51, s1, 31 // (op 627) 

s_xor_b32 s1, s0, s51 // (op 627) 

s_sub_i32 s0, s1, s51 // (op 627) 


s_mul_hi_i32 s1, s0, s13 // (op 627) 
s_add_i32 s51, s1, s0 // (op 627) 

s_ashr_i32 s1, s51, 31 // (op 627) 
s_cmp_eq_i32 s13, 0 // (op 627) 
s_mov_b32 s8, scc // (op 627) 
s_cmp_eq_i32 s8, 0 // (op 627) 
s_cselect_b32 s9, 0, -1 // (op 627) 

s_and_b32 s8, s14, 31 // (op 627) 
s_lshl_b32 s18, 1, s8 // (op 627) 
s_add_i32 s19, s18, s9 // (op 627) 


s_and_b32 s9, s1, s19 // (op 627) 


s_add_i32 s1, s51, s9 // (op 627) 


s_ashr_i32 s9, s1, s8 // (op 627) 


s_lshl_b32 s1, s14, 24 // (op 627) 
s_ashr_i32 s8, s1, 31 // (op 627) 

s_xor_b32 s1, s9, s8 // (op 627) 

s_sub_i32 s9, s1, s8 // (op 627) 


s_add_u32 s18, s2, 16 // (op 627) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 627) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 627) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 627) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 627) 

s_mul_i32 s1, s9, s18 // (op 627) 

s_sub_i32 s51, s0, s1 // (op 627) 


s_mul_hi_i32 s0, s16, s10 // (op 627) 
s_add_i32 s1, s0, s16 // (op 627) 

s_ashr_i32 s0, s1, 31 // (op 627) 
s_cmp_eq_i32 s10, 0 // (op 627) 
s_mov_b32 s8, scc // (op 627) 
s_cmp_eq_i32 s8, 0 // (op 627) 
s_cselect_b32 s9, 0, -1 // (op 627) 

s_and_b32 s8, s49, 31 // (op 627) 
s_lshl_b32 s56, 1, s8 // (op 627) 
s_add_i32 s57, s56, s9 // (op 627) 


s_and_b32 s9, s0, s57 // (op 627) 


s_add_i32 s0, s1, s9 // (op 627) 


s_ashr_i32 s1, s0, s8 // (op 627) 


s_lshl_b32 s0, s49, 24 // (op 627) 
s_ashr_i32 s8, s0, 31 // (op 627) 

s_xor_b32 s0, s1, s8 // (op 627) 

s_sub_i32 s1, s0, s8 // (op 627) 


s_mul_i32 s0, s46, s18 // (op 627) 

s_mul_i32 s8, s50, s0 // (op 627) 

s_mul_i32 s0, s1, s8 // (op 627) 

s_sub_u32 s8, s16, s0 // (op 627) 

s_mul_hi_i32 s0, s8, s15 // (op 627) 
s_add_i32 s9, s0, s8 // (op 627) 


s_ashr_i32 s0, s9, 31 // (op 627) 
s_cmp_eq_i32 s15, 0 // (op 627) 
s_mov_b32 s8, scc // (op 627) 
s_cmp_eq_i32 s8, 0 // (op 627) 
s_cselect_b32 s18, 0, -1 // (op 627) 

s_and_b32 s8, s48, 31 // (op 627) 
s_lshl_b32 s19, 1, s8 // (op 627) 
s_add_i32 s56, s19, s18 // (op 627) 


s_and_b32 s18, s0, s56 // (op 627) 


s_add_i32 s0, s9, s18 // (op 627) 


s_ashr_i32 s9, s0, s8 // (op 627) 


s_lshl_b32 s0, s48, 24 // (op 627) 
s_ashr_i32 s8, s0, 31 // (op 627) 

s_xor_b32 s0, s9, s8 // (op 627) 

s_sub_i32 s9, s0, s8 // (op 627) 


s_cmp_eq_i32 s1, 0 // (op 627) 
s_mov_b32 s0, scc // (op 627) 

s_cmp_eq_i32 s0, 0 // (op 627) 
s_cselect_b32 s1, s9, s51 // (op 627) 



s_lshl_b32 s0, s1, 4 // (op 627) 

s_mov_b32 s9, 0 // (op 627) convert
s_mov_b32 s8, s0 // (op 627) convert

v_mul_lo_u32 v17, v14, s23 // (op 627) most significant: low of low * high
v_mul_hi_u32 v15, v14, s22 // (op 627) most significant: high of low * low
v_mul_lo_u32 v16, v14, s22 // (op 627) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 627) most significant: sum

v_mov_b32 v15, s8 // (op 627) 
v_mov_b32 v18, s9 // (op 627) 
v_add_co_u32 v16, s[0:1], v16, v15 // (op 627) least significant half
v_addc_co_u32 v17, s[0:1], v17, v18, s[0:1] // (op 627) most significant half





v_mul_lo_u32 v15, v17, 6 // (op 627) most significant: low of high * low
v_mul_hi_u32 v18, v16, 6 // (op 627) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 627) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 627) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 627) 

v_mov_b32 v14, v16 // (op 627) convert

s_mul_i32 s1, s27, 6 // (op 629) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 629) least significant
s_mul_hi_u32 s51, s26, 6 // (op 629) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 629) most significant: sum

s_mul_i32 s9, s22, s1 // (op 629) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 629) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 629) least significant
s_mul_hi_u32 s18, s22, s0 // (op 629) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 629) most significant: sum
s_add_u32 s9, s9, s18 // (op 629) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 629) 

s_mov_b32 s56, s24 // (op 629) reg expression
s_mov_b32 s57, s25 // (op 629) reg expression
s_mov_b32 s58, s0 // (op 629) reg expression
s_mov_b32 s59, 131072 // (op 629) call()

s_mul_hi_i32 s51, s16, s11 // (op 646) 
v_lshlrev_b32 v15, 2, v1 // (op 646) 
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
s_sub_u32 s60, s18, 1 // (op 646) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 646) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 646) 

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
s_lshl_b32 s60, 1, s8 // (op 646) 
s_add_i32 s61, s60, s9 // (op 646) 


s_and_b32 s9, s0, s61 // (op 646) 


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
s_add_i32 s60, s19, s18 // (op 646) 


s_and_b32 s18, s0, s60 // (op 646) 


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

v_mul_lo_u32 v17, v15, s23 // (op 646) most significant: low of low * high
v_mul_hi_u32 v18, v15, s22 // (op 646) most significant: high of low * low
v_mul_lo_u32 v16, v15, s22 // (op 646) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 646) most significant: sum

v_mov_b32 v18, s8 // (op 646) 
v_mov_b32 v19, s9 // (op 646) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 646) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 646) most significant half





v_mul_lo_u32 v19, v17, 6 // (op 646) most significant: low of high * low
v_mul_hi_u32 v15, v16, 6 // (op 646) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 646) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 646) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 646) 

v_mov_b32 v15, v16 // (op 646) convert

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

s_mul_hi_i32 s51, s16, s11 // (op 691) 
v_lshrrev_b32 v16, 3, v1 // (op 691) 
s_add_i32 s0, s51, s16 // (op 691) 

v_lshlrev_b32 v17, 1, v16 // (op 691) 

s_ashr_i32 s1, s0, 31 // (op 691) 
v_and_b32 v16, -8, v1 // (op 691) 
s_cmp_eq_i32 s11, 0 // (op 691) 
s_mov_b32 s51, scc // (op 691) 
v_sub_u32 v18, v1, v16 // (op 691) 

s_cmp_eq_i32 s51, 0 // (op 691) 
s_cselect_b32 s8, 0, -1 // (op 691) 

s_and_b32 s9, s12, 31 // (op 691) 
s_lshl_b32 s51, 1, s9 // (op 691) 
s_add_i32 s18, s51, s8 // (op 691) 


s_and_b32 s8, s1, s18 // (op 691) 


s_add_i32 s1, s0, s8 // (op 691) 


s_ashr_i32 s0, s1, s9 // (op 691) 


s_lshl_b32 s1, s12, 24 // (op 691) 
s_ashr_i32 s51, s1, 31 // (op 691) 

s_xor_b32 s1, s0, s51 // (op 691) 

s_sub_i32 s0, s1, s51 // (op 691) 


s_mul_hi_i32 s1, s0, s13 // (op 691) 
s_add_i32 s51, s1, s0 // (op 691) 

s_ashr_i32 s1, s51, 31 // (op 691) 
s_cmp_eq_i32 s13, 0 // (op 691) 
s_mov_b32 s8, scc // (op 691) 
s_cmp_eq_i32 s8, 0 // (op 691) 
s_cselect_b32 s9, 0, -1 // (op 691) 

s_and_b32 s8, s14, 31 // (op 691) 
s_lshl_b32 s18, 1, s8 // (op 691) 
s_add_i32 s19, s18, s9 // (op 691) 


s_and_b32 s9, s1, s19 // (op 691) 


s_add_i32 s1, s51, s9 // (op 691) 


s_ashr_i32 s9, s1, s8 // (op 691) 


s_lshl_b32 s1, s14, 24 // (op 691) 
s_ashr_i32 s8, s1, 31 // (op 691) 

s_xor_b32 s1, s9, s8 // (op 691) 

s_sub_i32 s9, s1, s8 // (op 691) 


s_mul_i32 s1, s0, s46 // (op 691) 

s_sub_u32 s0, s16, s1 // (op 691) 

s_mul_i32 s1, s9, s46 // (op 691) 
s_add_u32 s1, s1, s0 // (op 691) 


s_mul_hi_i32 s0, s16, s10 // (op 691) 
s_add_i32 s51, s0, s16 // (op 691) 

s_ashr_i32 s0, s51, 31 // (op 691) 
s_cmp_eq_i32 s10, 0 // (op 691) 
s_mov_b32 s8, scc // (op 691) 
s_cmp_eq_i32 s8, 0 // (op 691) 
s_cselect_b32 s9, 0, -1 // (op 691) 

s_and_b32 s8, s49, 31 // (op 691) 
s_lshl_b32 s18, 1, s8 // (op 691) 
s_add_i32 s19, s18, s9 // (op 691) 


s_and_b32 s9, s0, s19 // (op 691) 


s_add_i32 s0, s51, s9 // (op 691) 


s_ashr_i32 s9, s0, s8 // (op 691) 


s_lshl_b32 s0, s49, 24 // (op 691) 
s_ashr_i32 s8, s0, 31 // (op 691) 

s_xor_b32 s0, s9, s8 // (op 691) 

s_sub_i32 s9, s0, s8 // (op 691) 


s_add_u32 s18, s2, 16 // (op 691) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 691) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 691) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 691) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 691) 

s_mul_i32 s0, s46, s18 // (op 691) 

s_mul_i32 s8, s50, s0 // (op 691) 

s_mul_i32 s0, s9, s8 // (op 691) 

s_sub_u32 s8, s16, s0 // (op 691) 

s_mov_b32 s0, s8 // (op 691) 
s_mul_hi_i32 s51, s0, s15 // (op 691) 
s_add_i32 s18, s51, s0 // (op 691) 


s_ashr_i32 s0, s18, 31 // (op 691) 
s_cmp_eq_i32 s15, 0 // (op 691) 
s_mov_b32 s19, scc // (op 691) 
s_cmp_eq_i32 s19, 0 // (op 691) 
s_cselect_b32 s51, 0, -1 // (op 691) 

s_and_b32 s19, s48, 31 // (op 691) 
s_lshl_b32 s60, 1, s19 // (op 691) 
s_add_i32 s61, s60, s51 // (op 691) 


s_and_b32 s51, s0, s61 // (op 691) 


s_add_i32 s0, s18, s51 // (op 691) 


s_ashr_i32 s18, s0, s19 // (op 691) 


s_lshl_b32 s0, s48, 24 // (op 691) 
s_ashr_i32 s19, s0, 31 // (op 691) 

s_xor_b32 s0, s18, s19 // (op 691) 

s_sub_i32 s18, s0, s19 // (op 691) 


s_mul_i32 s0, s47, s46 // (op 691) 
s_add_u32 s60, s6, 16 // (op 691) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 691) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 691) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 691) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 691) 

s_sub_i32 s19, s60, s0 // (op 691) 

s_mul_i32 s51, s18, s19 // (op 691) 


s_sub_u32 s18, s8, s51 // (op 691) 


s_add_u32 s8, s18, s0 // (op 691) 


s_cmp_eq_i32 s9, 0 // (op 691) 
s_mov_b32 s0, scc // (op 691) 

s_cmp_eq_i32 s0, 0 // (op 691) 
s_cselect_b32 s9, s8, s1 // (op 691) 



v_lshl_add_u32 v16, s9, 4, v17 // (op 691) 


v_mul_lo_u32 v17, v16, s28 // (op 691) 
v_add_u32 v17, v17, v18 // (op 691) 


s_mul_i32 s1, s28, s7 // (op 693) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 693) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 693) least significant
s_mul_hi_u32 s8, s28, s6 // (op 693) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 693) most significant: sum
s_add_u32 s1, s1, s8 // (op 693) most significant: sum


s_mov_b32 s60, s30 // (op 693) reg expression
s_mov_b32 s61, s31 // (op 693) reg expression
s_mov_b32 s62, s0 // (op 693) reg expression
s_mov_b32 s63, 131072 // (op 693) call()

s_mul_hi_i32 s51, s16, s11 // (op 710) 
v_lshrrev_b32 v16, 3, v1 // (op 710) 
s_add_i32 s0, s51, s16 // (op 710) 

v_lshlrev_b32 v18, 1, v16 // (op 710) 

s_ashr_i32 s1, s0, 31 // (op 710) 
v_and_b32 v16, -8, v1 // (op 710) 
s_cmp_eq_i32 s11, 0 // (op 710) 
s_mov_b32 s51, scc // (op 710) 
v_sub_u32 v19, v1, v16 // (op 710) 

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
s_sub_u32 s64, s18, 1 // (op 710) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 710) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 710) 

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
s_lshl_b32 s64, 1, s19 // (op 710) 
s_add_i32 s65, s64, s51 // (op 710) 


s_and_b32 s51, s0, s65 // (op 710) 


s_add_i32 s0, s18, s51 // (op 710) 


s_ashr_i32 s18, s0, s19 // (op 710) 


s_lshl_b32 s0, s48, 24 // (op 710) 
s_ashr_i32 s19, s0, 31 // (op 710) 

s_xor_b32 s0, s18, s19 // (op 710) 

s_sub_i32 s18, s0, s19 // (op 710) 


s_mul_i32 s0, s47, s46 // (op 710) 
s_add_u32 s64, s6, 16 // (op 710) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 710) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 710) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 710) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 710) 

s_sub_i32 s19, s64, s0 // (op 710) 

s_mul_i32 s51, s18, s19 // (op 710) 


s_sub_u32 s18, s8, s51 // (op 710) 


s_add_u32 s8, s18, s0 // (op 710) 


s_cmp_eq_i32 s9, 0 // (op 710) 
s_mov_b32 s0, scc // (op 710) 

s_cmp_eq_i32 s0, 0 // (op 710) 
s_cselect_b32 s9, s8, s1 // (op 710) 



v_lshl_add_u32 v16, s9, 4, v18 // (op 710) 


v_mul_lo_u32 v18, v16, s28 // (op 710) 
v_add_u32 v18, v18, v19 // (op 710) 


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

s_mul_hi_i32 s51, s16, s11 // (op 755) 
v_and_b32 v16, -16, v1 // (op 755) 
s_add_i32 s0, s51, s16 // (op 755) 

v_sub_u32 v19, v1, v16 // (op 755) 

s_ashr_i32 s1, s0, 31 // (op 755) 
v_lshrrev_b32 v16, 4, v1 // (op 755) 
s_cmp_eq_i32 s11, 0 // (op 755) 
s_mov_b32 s51, scc // (op 755) 
v_lshlrev_b32 v20, 1, v16 // (op 755) 

s_cmp_eq_i32 s51, 0 // (op 755) 
s_cselect_b32 s8, 0, -1 // (op 755) 

s_and_b32 s9, s12, 31 // (op 755) 
s_lshl_b32 s51, 1, s9 // (op 755) 
s_add_i32 s18, s51, s8 // (op 755) 


s_and_b32 s8, s1, s18 // (op 755) 


s_add_i32 s1, s0, s8 // (op 755) 


s_ashr_i32 s0, s1, s9 // (op 755) 


s_lshl_b32 s1, s12, 24 // (op 755) 
s_ashr_i32 s51, s1, 31 // (op 755) 

s_xor_b32 s1, s0, s51 // (op 755) 

s_sub_i32 s0, s1, s51 // (op 755) 


s_mul_hi_i32 s1, s0, s13 // (op 755) 
s_add_i32 s51, s1, s0 // (op 755) 

s_ashr_i32 s1, s51, 31 // (op 755) 
s_cmp_eq_i32 s13, 0 // (op 755) 
s_mov_b32 s8, scc // (op 755) 
s_cmp_eq_i32 s8, 0 // (op 755) 
s_cselect_b32 s9, 0, -1 // (op 755) 

s_and_b32 s8, s14, 31 // (op 755) 
s_lshl_b32 s18, 1, s8 // (op 755) 
s_add_i32 s19, s18, s9 // (op 755) 


s_and_b32 s9, s1, s19 // (op 755) 


s_add_i32 s1, s51, s9 // (op 755) 


s_ashr_i32 s9, s1, s8 // (op 755) 


s_lshl_b32 s1, s14, 24 // (op 755) 
s_ashr_i32 s8, s1, 31 // (op 755) 

s_xor_b32 s1, s9, s8 // (op 755) 

s_sub_i32 s9, s1, s8 // (op 755) 


s_add_u32 s18, s2, 16 // (op 755) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 755) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 755) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 755) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 755) 

s_mul_i32 s1, s9, s18 // (op 755) 

s_sub_i32 s51, s0, s1 // (op 755) 


s_mul_hi_i32 s0, s16, s10 // (op 755) 
s_add_i32 s1, s0, s16 // (op 755) 

s_ashr_i32 s0, s1, 31 // (op 755) 
s_cmp_eq_i32 s10, 0 // (op 755) 
s_mov_b32 s8, scc // (op 755) 
s_cmp_eq_i32 s8, 0 // (op 755) 
s_cselect_b32 s9, 0, -1 // (op 755) 

s_and_b32 s8, s49, 31 // (op 755) 
s_lshl_b32 s64, 1, s8 // (op 755) 
s_add_i32 s65, s64, s9 // (op 755) 


s_and_b32 s9, s0, s65 // (op 755) 


s_add_i32 s0, s1, s9 // (op 755) 


s_ashr_i32 s1, s0, s8 // (op 755) 


s_lshl_b32 s0, s49, 24 // (op 755) 
s_ashr_i32 s8, s0, 31 // (op 755) 

s_xor_b32 s0, s1, s8 // (op 755) 

s_sub_i32 s1, s0, s8 // (op 755) 


s_mul_i32 s0, s46, s18 // (op 755) 

s_mul_i32 s8, s50, s0 // (op 755) 

s_mul_i32 s0, s1, s8 // (op 755) 

s_sub_u32 s8, s16, s0 // (op 755) 

s_mul_hi_i32 s0, s8, s15 // (op 755) 
s_add_i32 s9, s0, s8 // (op 755) 


s_ashr_i32 s0, s9, 31 // (op 755) 
s_cmp_eq_i32 s15, 0 // (op 755) 
s_mov_b32 s8, scc // (op 755) 
s_cmp_eq_i32 s8, 0 // (op 755) 
s_cselect_b32 s18, 0, -1 // (op 755) 

s_and_b32 s8, s48, 31 // (op 755) 
s_lshl_b32 s19, 1, s8 // (op 755) 
s_add_i32 s64, s19, s18 // (op 755) 


s_and_b32 s18, s0, s64 // (op 755) 


s_add_i32 s0, s9, s18 // (op 755) 


s_ashr_i32 s9, s0, s8 // (op 755) 


s_lshl_b32 s0, s48, 24 // (op 755) 
s_ashr_i32 s8, s0, 31 // (op 755) 

s_xor_b32 s0, s9, s8 // (op 755) 

s_sub_i32 s9, s0, s8 // (op 755) 


s_cmp_eq_i32 s1, 0 // (op 755) 
s_mov_b32 s0, scc // (op 755) 

s_cmp_eq_i32 s0, 0 // (op 755) 
s_cselect_b32 s1, s9, s51 // (op 755) 



v_lshl_add_u32 v16, s1, 4, v19 // (op 755) 


v_mul_lo_u32 v19, v20, s32 // (op 755) 
v_add_u32 v19, v19, v16 // (op 755) 


s_mul_i32 s1, s32, s37 // (op 757) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 757) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 757) least significant
s_mul_hi_u32 s8, s32, s36 // (op 757) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 757) most significant: sum
s_add_u32 s1, s1, s8 // (op 757) most significant: sum


s_mov_b32 s64, s34 // (op 757) reg expression
s_mov_b32 s65, s35 // (op 757) reg expression
s_mov_b32 s66, s0 // (op 757) reg expression
s_mov_b32 s67, 131072 // (op 757) call()

s_mul_hi_i32 s51, s16, s11 // (op 774) 
v_and_b32 v16, -16, v1 // (op 774) 
s_add_i32 s0, s51, s16 // (op 774) 

v_sub_u32 v20, v1, v16 // (op 774) 

s_ashr_i32 s1, s0, 31 // (op 774) 
v_lshrrev_b32 v16, 4, v1 // (op 774) 
s_cmp_eq_i32 s11, 0 // (op 774) 
s_mov_b32 s51, scc // (op 774) 
v_lshlrev_b32 v21, 1, v16 // (op 774) 

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
s_sub_u32 s68, s18, 1 // (op 774) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 774) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 774) 

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
s_lshl_b32 s68, 1, s8 // (op 774) 
s_add_i32 s69, s68, s9 // (op 774) 


s_and_b32 s9, s0, s69 // (op 774) 


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
s_add_i32 s68, s19, s18 // (op 774) 


s_and_b32 s18, s0, s68 // (op 774) 


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



v_lshl_add_u32 v16, s1, 4, v20 // (op 774) 


v_mul_lo_u32 v20, v21, s32 // (op 774) 
v_add_u32 v20, v20, v16 // (op 774) 


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


s_mul_i32 s1, s23, 6 // (op 428) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 428) least significant
s_mul_hi_u32 s51, s22, 6 // (op 428) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 428) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 428) 


v_mov_b32 v16, v14 // (op 428) reg expression
buffer_load_dwordx3 v[24:26], v16, s[56:59], 0 offen // (op 428) Load value
v_add_u32 v16, v16, v22 // (op 428) 
buffer_load_dwordx3 v[28:30], v16, s[56:59], 0 offen // (op 428) Load value
v_add_u32 v16, v16, v22 // (op 428) 
buffer_load_dwordx3 v[32:34], v16, s[56:59], 0 offen // (op 428) Load value
v_add_u32 v16, v16, v22 // (op 428) 
buffer_load_dwordx3 v[36:38], v16, s[56:59], 0 offen // (op 428) Load value



v_mov_b64 v[22:23], s[32:33] // (op 431) convert

v_mov_b32 v16, v19 // (op 431) reg expression
buffer_load_ubyte v40, v16, s[64:67], 0 offen offset: 0 // (op 431) Load value
v_add_u32 v16, v16, v22 // (op 431) 
buffer_load_ubyte v41, v16, s[64:67], 0 offen offset: 0 // (op 431) Load value



s_lshl_b64 s[0:1], s[4:5], 2 // (op 434) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 434) 


v_mov_b32 v16, v12 // (op 434) 
buffer_load_dwordx4 v[42:45], v16, s[52:55], 0 offen // (op 434) Load value
v_add_u32 v16, v16, v22 // (op 434) 
buffer_load_dwordx4 v[46:49], v16, s[52:55], 0 offen // (op 434) Load value



v_mov_b64 v[22:23], s[28:29] // (op 437) convert

v_mov_b32 v16, v17 // (op 437) 
buffer_load_ubyte v50, v16, s[60:63], 0 offen offset: 0 // (op 437) Load value
v_add_u32 v16, v16, v22 // (op 437) 
buffer_load_ubyte v51, v16, s[60:63], 0 offen offset: 0 // (op 437) Load value



s_mul_i32 s1, s23, 6 // (op 440) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 440) least significant
s_mul_hi_u32 s51, s22, 6 // (op 440) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 440) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 440) 


s_lshl_b64 s[0:1], s[22:23], 8 // (op 440) 
s_mul_i32 s9, s1, 6 // (op 440) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 440) least significant
s_mul_hi_u32 s51, s0, 6 // (op 440) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 440) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 440) 

v_add_u32 v16, v14, s0 // (op 440) 

buffer_load_dwordx3 v[52:54], v16, s[56:59], 0 offen // (op 440) Load value
v_add_u32 v16, v16, v22 // (op 440) 
buffer_load_dwordx3 v[56:58], v16, s[56:59], 0 offen // (op 440) Load value
v_add_u32 v16, v16, v22 // (op 440) 
buffer_load_dwordx3 v[60:62], v16, s[56:59], 0 offen // (op 440) Load value
v_add_u32 v16, v16, v22 // (op 440) 
buffer_load_dwordx3 v[64:66], v16, s[56:59], 0 offen // (op 440) Load value




v_mov_b64 v[22:23], s[32:33] // (op 443) convert

s_lshl_b32 s51, s32, 3 // (op 443) 
v_add_u32 v14, v19, s51 // (op 443) 

buffer_load_ubyte v68, v14, s[64:67], 0 offen offset: 0 // (op 443) Load value
v_add_u32 v14, v14, v22 // (op 443) 
buffer_load_ubyte v69, v14, s[64:67], 0 offen offset: 0 // (op 443) Load value




s_lshl_b64 s[0:1], s[4:5], 2 // (op 446) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 446) 


v_mov_b32 v14, v12 // (op 446) 
buffer_load_dwordx4 v[70:73], v14, s[52:55], 0 offen offset:128 // (op 446) Load value
v_add_u32 v14, v14, v22 // (op 446) 
buffer_load_dwordx4 v[74:77], v14, s[52:55], 0 offen offset:128 // (op 446) Load value




v_mov_b64 v[22:23], s[28:29] // (op 449) convert

v_mov_b32 v12, v17 // (op 449) 
buffer_load_ubyte v78, v12, s[60:63], 0 offen offset:8 // (op 449) Load value
v_add_u32 v12, v12, v22 // (op 449) 
buffer_load_ubyte v79, v12, s[60:63], 0 offen offset:8 // (op 449) Load value



v_lshlrev_b32 v12, 6, v1 // (op 894) 
v_mul_lo_u32 v14, v12, 6 // (op 894) 
v_lshrrev_b32 v19, 3, v14 // (op 894) 

v_lshrrev_b32 v14, 4, v12 // (op 894) 

v_lshlrev_b32 v12, 2, v14 // (op 894) 

v_add_u32 v14, v19, v12 // (op 894) 





s_waitcnt vmcnt(19)
ds_write_b96 v14, v[24:26] // (op 451) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v14, v[28:30] offset:16 // (op 451) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v14, v[32:34] offset:32 // (op 451) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v14, v[36:38] offset:48 // (op 451) Store local data 

v_and_b32 v12, -16, v1 // (op 1000) 
v_sub_u32 v14, v1, v12 // (op 1000) 

v_lshrrev_b32 v12, 4, v1 // (op 1000) 
v_lshl_add_u32 v19, v12, 5, v14 // (op 1000) 





s_waitcnt vmcnt(15)
ds_write_b8 v19, v40 offset:4096 // (op 452) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v19, v41 offset:4112 // (op 452) Store local data 

v_and_b32 v12, -8, v1 // (op 841) 
v_sub_u32 v14, v1, v12 // (op 841) 

v_lshlrev_b32 v12, 5, v14 // (op 841) 

v_lshrrev_b32 v14, 3, v1 // (op 841) 
v_lshl_add_u32 v19, v14, 9, v12 // (op 841) 


v_lshlrev_b32 v12, 2, v19 // (op 841) 

v_lshrrev_b32 v14, 3, v12 // (op 841) 




s_waitcnt vmcnt(13)
ds_write_b128 v14, v[42:45] offset:4224 // (op 453) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v14, v[46:49] offset:4352 // (op 453) Store local data 

v_and_b32 v12, -8, v1 // (op 947) 
v_sub_u32 v14, v1, v12 // (op 947) 

v_lshrrev_b32 v12, 3, v1 // (op 947) 
v_lshl_add_u32 v19, v12, 4, v14 // (op 947) 





s_waitcnt vmcnt(11)
ds_write_b8 v19, v50 offset:6272 // (op 454) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v19, v51 offset:6280 // (op 454) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 1198) 



ds_read_b96_tr_b6 v[80:82], v5 // (op 471) Transpose load local data 
ds_read_b96_tr_b6 v[83:85], v5 offset:256 // (op 471) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1239) 


ds_read_u8 v12, v7 offset:4096 // (op 477) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1179) 



ds_read_b128 v[86:89], v3 offset:4224 // (op 468) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1219) 



ds_read_u8 v14, v0 offset:6272 // (op 474) Load local data 
v_and_b32 v19, -8, v1 // (op 812) 
v_sub_u32 v27, v1, v19 // (op 812) 

v_lshlrev_b32 v19, 5, v27 // (op 812) 

v_lshrrev_b32 v27, 3, v1 // (op 812) 
v_lshl_add_u32 v31, v27, 9, v19 // (op 812) 


v_lshlrev_b32 v19, 2, v31 // (op 812) 

v_lshrrev_b32 v27, 3, v19 // (op 812) 

v_lshlrev_b32 v19, 6, v1 // (op 865) 
v_mul_lo_u32 v31, v19, 6 // (op 865) 
v_lshrrev_b32 v35, 3, v31 // (op 865) 

v_lshrrev_b32 v31, 4, v19 // (op 865) 

v_lshlrev_b32 v19, 2, v31 // (op 865) 

v_add_u32 v31, v35, v19 // (op 865) 


v_and_b32 v19, -8, v1 // (op 918) 
v_sub_u32 v35, v1, v19 // (op 918) 

v_lshrrev_b32 v19, 3, v1 // (op 918) 
v_lshl_add_u32 v39, v19, 4, v35 // (op 918) 


v_and_b32 v19, -16, v1 // (op 971) 
v_sub_u32 v35, v1, v19 // (op 971) 

v_lshrrev_b32 v19, 4, v1 // (op 971) 
v_lshl_add_u32 v55, v19, 5, v35 // (op 971) 


v_and_b32 v19, -8, v1 // (op 1058) 
v_sub_u32 v35, v1, v19 // (op 1058) 

v_lshlrev_b32 v19, 5, v35 // (op 1058) 

v_lshrrev_b32 v35, 3, v1 // (op 1058) 
v_lshl_add_u32 v59, v35, 9, v19 // (op 1058) 


v_lshlrev_b32 v19, 2, v59 // (op 1058) 

v_lshrrev_b32 v35, 3, v19 // (op 1058) 

v_lshlrev_b32 v19, 6, v1 // (op 1082) 
v_mul_lo_u32 v59, v19, 6 // (op 1082) 
v_lshrrev_b32 v63, 3, v59 // (op 1082) 

v_lshrrev_b32 v59, 4, v19 // (op 1082) 

v_lshlrev_b32 v19, 2, v59 // (op 1082) 

v_add_u32 v59, v63, v19 // (op 1082) 


v_and_b32 v19, -8, v1 // (op 1106) 
v_sub_u32 v63, v1, v19 // (op 1106) 

v_lshrrev_b32 v19, 3, v1 // (op 1106) 
v_lshl_add_u32 v67, v19, 4, v63 // (op 1106) 


v_and_b32 v19, -16, v1 // (op 1130) 
v_sub_u32 v63, v1, v19 // (op 1130) 

v_lshrrev_b32 v19, 4, v1 // (op 1130) 
v_lshl_add_u32 v16, v19, 5, v63 // (op 1130) 


s_mov_b32 s51, 0 // (op 61) call()
s_ashr_i64 s[0:1], s[26:27], 8 // (op 60) 
s_mov_b32 s8, s0 // (op 60) convert

s_and_b32 s9, s8, 4294967294 // (op 60) 

s_cmp_lt_u32 s51, s9 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_60 if false)
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_60:

s_add_u32 s51, s51, 2 // (op 62) 
s_mul_i32 s1, s23, 6 // (op 10) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 10) least significant
s_mul_hi_u32 s8, s22, 6 // (op 10) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 10) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 10) 


v_mov_b32 v17, v15 // (op 10) 
s_lshl_b64 s[0:1], s[22:23], 8 // (op 10) 
s_mul_i32 s9, s1, 6 // (op 10) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 10) least significant
s_mul_hi_u32 s18, s0, 6 // (op 10) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 10) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 10) 

s_lshl_b32 s8, s0, 1 // (op 10) 

v_add_u32 v19, v17, s8 // (op 10) 

buffer_load_dwordx3 v[24:26], v19, s[56:59], 0 offen // (op 10) Load value
v_add_u32 v19, v19, v22 // (op 10) 
buffer_load_dwordx3 v[28:30], v19, s[56:59], 0 offen // (op 10) Load value
v_add_u32 v19, v19, v22 // (op 10) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[86:89], v[80:85], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 90) 
buffer_load_dwordx3 v[32:34], v19, s[56:59], 0 offen // (op 10) Load value
s_barrier  // (op 1188) 
v_add_u32 v19, v19, v22 // (op 10) 



ds_read_b128 v[86:89], v3 offset:4288 // (op 93) Load local data 
buffer_load_dwordx3 v[36:38], v19, s[56:59], 0 offen // (op 10) Load value



s_barrier  // (op 1208) 
v_mov_b64 v[22:23], s[32:33] // (op 25) convert




ds_read_b96_tr_b6 v[80:82], v5 offset:2048 // (op 97) Transpose load local data 
ds_read_b96_tr_b6 v[83:85], v5 offset:2304 // (op 97) Transpose load local data 
v_mov_b32 v19, v20 // (op 25) 
s_barrier  // (op 1229) 
s_lshl_b32 s0, s32, 4 // (op 25) 



ds_read_u8 v14, v0 offset:6276 // (op 101) Load local data 
v_add_u32 v21, v19, s0 // (op 25) 

s_barrier  // (op 1249) 
buffer_load_ubyte v40, v21, s[64:67], 0 offen offset: 0 // (op 25) Load value


v_add_u32 v21, v21, v22 // (op 25) 
buffer_load_ubyte v41, v21, s[64:67], 0 offen offset: 0 // (op 25) Load value


ds_read_u8 v12, v7 offset:4160 // (op 105) Load local data 

s_lshl_b64 s[0:1], s[4:5], 2 // (op 4) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 4) 


v_mov_b32 v21, v13 // (op 4) 
v_mov_b32 v63, v21 // (op 4) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[86:89], v[80:85], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 107) 
buffer_load_dwordx4 v[42:45], v63, s[52:55], 0 offen offset:256 // (op 4) Load value
v_add_u32 v63, v63, v22 // (op 4) 
buffer_load_dwordx4 v[46:49], v63, s[52:55], 0 offen offset:256 // (op 4) Load value



v_mov_b64 v[22:23], s[28:29] // (op 17) convert

v_mov_b32 v63, v18 // (op 17) 
v_mov_b32 v90, v63 // (op 17) 
buffer_load_ubyte v50, v90, s[60:63], 0 offen offset:16 // (op 17) Load value
v_add_u32 v90, v90, v22 // (op 17) 
buffer_load_ubyte v51, v90, s[60:63], 0 offen offset:16 // (op 17) Load value


s_barrier  // (op 1269) 



s_waitcnt vmcnt(19)
ds_write_b96 v59, v[52:54] offset:6400 // (op 306) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v59, v[56:58] offset:6416 // (op 306) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v59, v[60:62] offset:6432 // (op 306) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v59, v[64:66] offset:6448 // (op 306) Store local data 
s_barrier  // (op 1281) 



s_waitcnt vmcnt(15)
ds_write_b8 v16, v68 offset:10496 // (op 308) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v16, v69 offset:10512 // (op 308) Store local data 
s_barrier  // (op 1262) 



s_waitcnt vmcnt(13)
ds_write_b128 v35, v[70:73] offset:10624 // (op 287) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v35, v[74:77] offset:10752 // (op 287) Store local data 
s_barrier  // (op 1275) 



s_waitcnt vmcnt(11)
ds_write_b8 v67, v78 offset:12672 // (op 307) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v67, v79 offset:12680 // (op 307) Store local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1265) 



ds_read_b96_tr_b6 v[90:92], v9 offset:6400 // (op 291) Transpose load local data 
ds_read_b96_tr_b6 v[93:95], v9 offset:6656 // (op 291) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1278) 


ds_read_u8 v22, v11 offset:10496 // (op 295) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1259) 



ds_read_b128 v[96:99], v6 offset:10624 // (op 289) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1272) 



ds_read_u8 v23, v4 offset:12672 // (op 293) Load local data 

s_mul_i32 s1, s23, 6 // (op 309) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 309) least significant
s_mul_hi_u32 s8, s22, 6 // (op 309) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 309) most significant: sum

v_lshrrev_b64 v[100:101], 3, s[0:1] // (op 309) 


s_lshl_b64 s[0:1], s[22:23], 8 // (op 309) 
s_mul_i32 s9, s1, 6 // (op 309) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 309) least significant
s_mul_hi_u32 s18, s0, 6 // (op 309) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 309) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 309) 

s_mul_i32 s8, 3, s0 // (op 309) 

v_add_u32 v102, v17, s8 // (op 309) 

buffer_load_dwordx3 v[52:54], v102, s[56:59], 0 offen // (op 309) Load value
v_add_u32 v102, v102, v100 // (op 309) 
buffer_load_dwordx3 v[56:58], v102, s[56:59], 0 offen // (op 309) Load value
v_add_u32 v102, v102, v100 // (op 309) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:99], v[90:95], a[0:3], v23, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 296) 
buffer_load_dwordx3 v[60:62], v102, s[56:59], 0 offen // (op 309) Load value




v_add_u32 v102, v102, v100 // (op 309) 



ds_read_b128 v[90:93], v6 offset:10688 // (op 298) Load local data 
buffer_load_dwordx3 v[64:66], v102, s[56:59], 0 offen // (op 309) Load value







s_barrier  // (op 1344) 
v_mov_b64 v[22:23], s[32:33] // (op 311) convert

s_lshl_b32 s0, s32, 3 // (op 311) 
s_mul_i32 s1, 3, s0 // (op 311) 

v_add_u32 v17, v19, s1 // (op 311) 

buffer_load_ubyte v68, v17, s[64:67], 0 offen offset: 0 // (op 311) Load value
v_add_u32 v17, v17, v22 // (op 311) 
s_lshl_b64 s[0:1], s[22:23], 8 // (op 638) 
s_mul_i32 s9, s1, 6 // (op 638) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 638) least significant
s_mul_hi_u32 s18, s0, 6 // (op 638) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 638) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 638) 

s_lshl_b32 s8, s0, 1 // (op 638) 

v_add_u32 v15, v15, s8 // (op 638) 

ds_read_b96_tr_b6 v[94:96], v9 offset:8448 // (op 300) Transpose load local data 
ds_read_b96_tr_b6 v[97:99], v9 offset:8704 // (op 300) Transpose load local data 
buffer_load_ubyte v69, v17, s[64:67], 0 offen offset: 0 // (op 311) Load value







s_barrier  // (op 1348) 
ds_read_u8 v17, v4 offset:12676 // (op 302) Load local data 
s_lshl_b32 s0, s32, 4 // (op 766) 
v_add_u32 v20, v20, s0 // (op 766) 



s_barrier  // (op 1352) 
s_lshl_b64 s[0:1], s[4:5], 2 // (op 286) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 286) 


v_mov_b32 v19, v21 // (op 286) 
ds_read_u8 v100, v11 offset:10560 // (op 304) Load local data 
buffer_load_dwordx4 v[70:73], v19, s[52:55], 0 offen offset:384 // (op 286) Load value
v_add_u32 v19, v19, v22 // (op 286) 
s_barrier  // (op 1356) 
buffer_load_dwordx4 v[74:77], v19, s[52:55], 0 offen offset:384 // (op 286) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:93], v[94:99], a[0:3], v17, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 305) 






v_add_u32 v13, 256, v13 // (op 574) 
v_mov_b64 v[22:23], s[28:29] // (op 310) convert

v_mov_b32 v17, v63 // (op 310) 
buffer_load_ubyte v78, v17, s[60:63], 0 offen offset:24 // (op 310) Load value
v_add_u32 v17, v17, v22 // (op 310) 
buffer_load_ubyte v79, v17, s[60:63], 0 offen offset:24 // (op 310) Load value



s_barrier  // (op 1201) 
v_add_u32 v18, 16, v18 // (op 702) 



s_waitcnt vmcnt(19)
ds_write_b96 v31, v[24:26] // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v31, v[28:30] offset:16 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v31, v[32:34] offset:32 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v31, v[36:38] offset:48 // (op 13) Store local data 
s_barrier  // (op 1242) 



s_waitcnt vmcnt(15)
ds_write_b8 v55, v40 offset:4096 // (op 30) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v55, v41 offset:4112 // (op 30) Store local data 
s_barrier  // (op 1182) 



s_waitcnt vmcnt(13)
ds_write_b128 v27, v[42:45] offset:4224 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v27, v[46:49] offset:4352 // (op 6) Store local data 
s_barrier  // (op 1222) 



s_waitcnt vmcnt(11)
ds_write_b8 v39, v50 offset:6272 // (op 21) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v39, v51 offset:6280 // (op 21) Store local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1211) 



ds_read_b96_tr_b6 v[80:82], v5 // (op 77) Transpose load local data 
ds_read_b96_tr_b6 v[83:85], v5 offset:256 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1252) 


ds_read_u8 v12, v7 offset:4096 // (op 87) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1191) 



ds_read_b128 v[86:89], v3 offset:4224 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1232) 



ds_read_u8 v14, v0 offset:6272 // (op 82) Load local data 
s_ashr_i64 s[0:1], s[26:27], 8 // (op 60) 
s_mov_b32 s8, s0 // (op 60) convert

s_and_b32 s9, s8, 4294967294 // (op 60) 

s_cmp_lt_u32 s51, s9 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_60 if true)
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_60:

















s_mul_hi_i32 s51, s16, s11 // (op 537) 
v_lshrrev_b32 v13, 3, v1 // (op 537) 
s_add_i32 s0, s51, s16 // (op 537) 

v_lshlrev_b32 v55, 1, v13 // (op 537) 

s_ashr_i32 s1, s0, 31 // (op 537) 
v_and_b32 v13, -8, v1 // (op 537) 
s_cmp_eq_i32 s11, 0 // (op 537) 
s_mov_b32 s51, scc // (op 537) 
v_sub_u32 v59, v1, v13 // (op 537) 

s_cmp_eq_i32 s51, 0 // (op 537) 
s_cselect_b32 s8, 0, -1 // (op 537) 

v_lshlrev_b32 v13, 5, v59 // (op 537) 

s_and_b32 s9, s12, 31 // (op 537) 
s_lshl_b32 s51, 1, s9 // (op 537) 
s_add_i32 s18, s51, s8 // (op 537) 


s_and_b32 s8, s1, s18 // (op 537) 


s_add_i32 s1, s0, s8 // (op 537) 


s_ashr_i32 s0, s1, s9 // (op 537) 


s_lshl_b32 s1, s12, 24 // (op 537) 
s_ashr_i32 s51, s1, 31 // (op 537) 

s_xor_b32 s1, s0, s51 // (op 537) 

s_sub_i32 s0, s1, s51 // (op 537) 


s_mul_hi_i32 s1, s0, s13 // (op 537) 
s_add_i32 s51, s1, s0 // (op 537) 

s_ashr_i32 s1, s51, 31 // (op 537) 
s_cmp_eq_i32 s13, 0 // (op 537) 
s_mov_b32 s8, scc // (op 537) 
s_cmp_eq_i32 s8, 0 // (op 537) 
s_cselect_b32 s9, 0, -1 // (op 537) 

s_and_b32 s8, s14, 31 // (op 537) 
s_lshl_b32 s18, 1, s8 // (op 537) 
s_add_i32 s19, s18, s9 // (op 537) 


s_and_b32 s9, s1, s19 // (op 537) 


s_add_i32 s1, s51, s9 // (op 537) 


s_ashr_i32 s9, s1, s8 // (op 537) 


s_lshl_b32 s1, s14, 24 // (op 537) 
s_ashr_i32 s8, s1, 31 // (op 537) 

s_xor_b32 s1, s9, s8 // (op 537) 

s_sub_i32 s9, s1, s8 // (op 537) 


s_mul_i32 s1, s0, s46 // (op 537) 

s_sub_u32 s0, s16, s1 // (op 537) 

s_mul_i32 s1, s9, s46 // (op 537) 
s_add_u32 s1, s1, s0 // (op 537) 


s_mul_hi_i32 s0, s16, s10 // (op 537) 
s_add_i32 s51, s0, s16 // (op 537) 

s_ashr_i32 s0, s51, 31 // (op 537) 
s_cmp_eq_i32 s10, 0 // (op 537) 
s_mov_b32 s8, scc // (op 537) 
s_cmp_eq_i32 s8, 0 // (op 537) 
s_cselect_b32 s9, 0, -1 // (op 537) 

s_and_b32 s8, s49, 31 // (op 537) 
s_lshl_b32 s18, 1, s8 // (op 537) 
s_add_i32 s19, s18, s9 // (op 537) 


s_and_b32 s9, s0, s19 // (op 537) 


s_add_i32 s0, s51, s9 // (op 537) 


s_ashr_i32 s9, s0, s8 // (op 537) 


s_lshl_b32 s0, s49, 24 // (op 537) 
s_ashr_i32 s8, s0, 31 // (op 537) 

s_xor_b32 s0, s9, s8 // (op 537) 

s_sub_i32 s9, s0, s8 // (op 537) 


s_add_u32 s18, s2, 16 // (op 537) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 537) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 537) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 537) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 537) 

s_mul_i32 s0, s46, s18 // (op 537) 

s_mul_i32 s8, s50, s0 // (op 537) 

s_mul_i32 s0, s9, s8 // (op 537) 

s_sub_u32 s8, s16, s0 // (op 537) 

s_mov_b32 s0, s8 // (op 537) 
s_mul_hi_i32 s51, s0, s15 // (op 537) 
s_add_i32 s18, s51, s0 // (op 537) 


s_ashr_i32 s0, s18, 31 // (op 537) 
s_cmp_eq_i32 s15, 0 // (op 537) 
s_mov_b32 s19, scc // (op 537) 
s_cmp_eq_i32 s19, 0 // (op 537) 
s_cselect_b32 s51, 0, -1 // (op 537) 

s_and_b32 s19, s48, 31 // (op 537) 
s_lshl_b32 s68, 1, s19 // (op 537) 
s_add_i32 s69, s68, s51 // (op 537) 


s_and_b32 s51, s0, s69 // (op 537) 


s_add_i32 s0, s18, s51 // (op 537) 


s_ashr_i32 s18, s0, s19 // (op 537) 


s_lshl_b32 s0, s48, 24 // (op 537) 
s_ashr_i32 s19, s0, 31 // (op 537) 

s_xor_b32 s0, s18, s19 // (op 537) 

s_sub_i32 s18, s0, s19 // (op 537) 


s_mul_i32 s0, s47, s46 // (op 537) 
s_add_u32 s68, s6, 16 // (op 537) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 537) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 537) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 537) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 537) 

s_sub_i32 s19, s68, s0 // (op 537) 

s_mul_i32 s51, s18, s19 // (op 537) 


s_sub_u32 s18, s8, s51 // (op 537) 


s_add_u32 s8, s18, s0 // (op 537) 


s_cmp_eq_i32 s9, 0 // (op 537) 
s_mov_b32 s0, scc // (op 537) 

s_cmp_eq_i32 s0, 0 // (op 537) 
s_cselect_b32 s9, s8, s1 // (op 537) 



v_lshl_add_u32 v59, s9, 4, v55 // (op 537) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 537) 
s_lshr_b32 s51, s0, 1 // (op 537) 

v_lshl_add_u32 v55, s51, 9, v13 // (op 537) 


v_mov_b32 v51, 0 // (op 537) convert
v_mov_b32 v50, v55 // (op 537) convert

v_mul_lo_u32 v49, v59, s5 // (op 537) most significant: low of low * high
v_mul_hi_u32 v13, v59, s4 // (op 537) most significant: high of low * low
v_mul_lo_u32 v48, v59, s4 // (op 537) least significant: low of low * low
v_add_u32 v49, v49, v13 // (op 537) most significant: sum

v_add_co_u32 v48, s[0:1], v48, v50 // (op 537) least significant half
v_addc_co_u32 v49, s[0:1], v49, v51, s[0:1] // (op 537) most significant half



v_lshlrev_b64 v[50:51], 2, v[48:49] // (op 537) 

v_lshrrev_b64 v[48:49], 3, v[50:51] // (op 537) 

v_mov_b32 v13, v48 // (op 537) convert

s_mul_i32 s1, s4, s7 // (op 539) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 539) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 539) least significant
s_mul_hi_u32 s8, s4, s6 // (op 539) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 539) most significant: sum
s_add_u32 s1, s1, s8 // (op 539) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 539) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 539) 

s_mov_b32 s52, s20 // (op 539) reg expression
s_mov_b32 s53, s21 // (op 539) reg expression
s_mov_b32 s54, s0 // (op 539) reg expression
s_mov_b32 s55, 131072 // (op 539) call()

s_mul_hi_i32 s51, s16, s11 // (op 604) 
v_lshlrev_b32 v55, 2, v1 // (op 604) 
s_add_i32 s0, s51, s16 // (op 604) 

s_ashr_i32 s1, s0, 31 // (op 604) 
s_cmp_eq_i32 s11, 0 // (op 604) 
s_mov_b32 s51, scc // (op 604) 
s_cmp_eq_i32 s51, 0 // (op 604) 
s_cselect_b32 s8, 0, -1 // (op 604) 

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


s_add_u32 s18, s2, 16 // (op 604) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 604) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 604) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 604) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 604) 

s_mul_i32 s1, s9, s18 // (op 604) 

s_sub_i32 s51, s0, s1 // (op 604) 


s_mul_hi_i32 s0, s16, s10 // (op 604) 
s_add_i32 s1, s0, s16 // (op 604) 

s_ashr_i32 s0, s1, 31 // (op 604) 
s_cmp_eq_i32 s10, 0 // (op 604) 
s_mov_b32 s8, scc // (op 604) 
s_cmp_eq_i32 s8, 0 // (op 604) 
s_cselect_b32 s9, 0, -1 // (op 604) 

s_and_b32 s8, s49, 31 // (op 604) 
s_lshl_b32 s68, 1, s8 // (op 604) 
s_add_i32 s69, s68, s9 // (op 604) 


s_and_b32 s9, s0, s69 // (op 604) 


s_add_i32 s0, s1, s9 // (op 604) 


s_ashr_i32 s1, s0, s8 // (op 604) 


s_lshl_b32 s0, s49, 24 // (op 604) 
s_ashr_i32 s8, s0, 31 // (op 604) 

s_xor_b32 s0, s1, s8 // (op 604) 

s_sub_i32 s1, s0, s8 // (op 604) 


s_mul_i32 s0, s46, s18 // (op 604) 

s_mul_i32 s8, s50, s0 // (op 604) 

s_mul_i32 s0, s1, s8 // (op 604) 

s_sub_u32 s8, s16, s0 // (op 604) 

s_mul_hi_i32 s0, s8, s15 // (op 604) 
s_add_i32 s9, s0, s8 // (op 604) 


s_ashr_i32 s0, s9, 31 // (op 604) 
s_cmp_eq_i32 s15, 0 // (op 604) 
s_mov_b32 s8, scc // (op 604) 
s_cmp_eq_i32 s8, 0 // (op 604) 
s_cselect_b32 s18, 0, -1 // (op 604) 

s_and_b32 s8, s48, 31 // (op 604) 
s_lshl_b32 s19, 1, s8 // (op 604) 
s_add_i32 s68, s19, s18 // (op 604) 


s_and_b32 s18, s0, s68 // (op 604) 


s_add_i32 s0, s9, s18 // (op 604) 


s_ashr_i32 s9, s0, s8 // (op 604) 


s_lshl_b32 s0, s48, 24 // (op 604) 
s_ashr_i32 s8, s0, 31 // (op 604) 

s_xor_b32 s0, s9, s8 // (op 604) 

s_sub_i32 s9, s0, s8 // (op 604) 


s_cmp_eq_i32 s1, 0 // (op 604) 
s_mov_b32 s0, scc // (op 604) 

s_cmp_eq_i32 s0, 0 // (op 604) 
s_cselect_b32 s1, s9, s51 // (op 604) 



s_lshl_b32 s0, s1, 4 // (op 604) 

s_mov_b32 s9, 0 // (op 604) convert
s_mov_b32 s8, s0 // (op 604) convert

s_ashr_i64 s[0:1], s[26:27], 8 // (op 604) 
s_lshr_b32 s51, s0, 1 // (op 604) 

v_lshl_add_u32 v59, s51, 9, v55 // (op 604) 


v_mul_lo_u32 v51, v59, s23 // (op 604) most significant: low of low * high
v_mul_hi_u32 v55, v59, s22 // (op 604) most significant: high of low * low
v_mul_lo_u32 v50, v59, s22 // (op 604) least significant: low of low * low
v_add_u32 v51, v51, v55 // (op 604) most significant: sum

v_mov_b32 v55, s8 // (op 604) 
v_mov_b32 v63, s9 // (op 604) 
v_add_co_u32 v50, s[0:1], v50, v55 // (op 604) least significant half
v_addc_co_u32 v51, s[0:1], v51, v63, s[0:1] // (op 604) most significant half





v_mul_lo_u32 v49, v51, 6 // (op 604) most significant: low of high * low
v_mul_hi_u32 v55, v50, 6 // (op 604) most significant: high of low * low
v_mul_lo_u32 v48, v50, 6 // (op 604) least significant: low of low * low
v_add_u32 v49, v49, v55 // (op 604) most significant: sum


v_lshrrev_b64 v[50:51], 3, v[48:49] // (op 604) 

v_mov_b32 v55, v50 // (op 604) convert

s_mul_i32 s1, s27, 6 // (op 606) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 606) least significant
s_mul_hi_u32 s51, s26, 6 // (op 606) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 606) most significant: sum

s_mul_i32 s9, s22, s1 // (op 606) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 606) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 606) least significant
s_mul_hi_u32 s18, s22, s0 // (op 606) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 606) most significant: sum
s_add_u32 s9, s9, s18 // (op 606) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 606) 

s_mov_b32 s56, s24 // (op 606) reg expression
s_mov_b32 s57, s25 // (op 606) reg expression
s_mov_b32 s58, s0 // (op 606) reg expression
s_mov_b32 s59, 131072 // (op 606) call()

s_mul_hi_i32 s51, s16, s11 // (op 668) 
v_lshrrev_b32 v59, 3, v1 // (op 668) 
s_add_i32 s0, s51, s16 // (op 668) 

v_lshlrev_b32 v63, 1, v59 // (op 668) 

s_ashr_i32 s1, s0, 31 // (op 668) 
v_and_b32 v59, -8, v1 // (op 668) 
s_cmp_eq_i32 s11, 0 // (op 668) 
s_mov_b32 s51, scc // (op 668) 
v_sub_u32 v67, v1, v59 // (op 668) 

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


s_mul_i32 s1, s0, s46 // (op 668) 

s_sub_u32 s0, s16, s1 // (op 668) 

s_mul_i32 s1, s9, s46 // (op 668) 
s_add_u32 s1, s1, s0 // (op 668) 


s_mul_hi_i32 s0, s16, s10 // (op 668) 
s_add_i32 s51, s0, s16 // (op 668) 

s_ashr_i32 s0, s51, 31 // (op 668) 
s_cmp_eq_i32 s10, 0 // (op 668) 
s_mov_b32 s8, scc // (op 668) 
s_cmp_eq_i32 s8, 0 // (op 668) 
s_cselect_b32 s9, 0, -1 // (op 668) 

s_and_b32 s8, s49, 31 // (op 668) 
s_lshl_b32 s18, 1, s8 // (op 668) 
s_add_i32 s19, s18, s9 // (op 668) 


s_and_b32 s9, s0, s19 // (op 668) 


s_add_i32 s0, s51, s9 // (op 668) 


s_ashr_i32 s9, s0, s8 // (op 668) 


s_lshl_b32 s0, s49, 24 // (op 668) 
s_ashr_i32 s8, s0, 31 // (op 668) 

s_xor_b32 s0, s9, s8 // (op 668) 

s_sub_i32 s9, s0, s8 // (op 668) 


s_add_u32 s18, s2, 16 // (op 668) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 668) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 668) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 668) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 668) 

s_mul_i32 s0, s46, s18 // (op 668) 

s_mul_i32 s8, s50, s0 // (op 668) 

s_mul_i32 s0, s9, s8 // (op 668) 

s_sub_u32 s8, s16, s0 // (op 668) 

s_mov_b32 s0, s8 // (op 668) 
s_mul_hi_i32 s51, s0, s15 // (op 668) 
s_add_i32 s18, s51, s0 // (op 668) 


s_ashr_i32 s0, s18, 31 // (op 668) 
s_cmp_eq_i32 s15, 0 // (op 668) 
s_mov_b32 s19, scc // (op 668) 
s_cmp_eq_i32 s19, 0 // (op 668) 
s_cselect_b32 s51, 0, -1 // (op 668) 

s_and_b32 s19, s48, 31 // (op 668) 
s_lshl_b32 s68, 1, s19 // (op 668) 
s_add_i32 s69, s68, s51 // (op 668) 


s_and_b32 s51, s0, s69 // (op 668) 


s_add_i32 s0, s18, s51 // (op 668) 


s_ashr_i32 s18, s0, s19 // (op 668) 


s_lshl_b32 s0, s48, 24 // (op 668) 
s_ashr_i32 s19, s0, 31 // (op 668) 

s_xor_b32 s0, s18, s19 // (op 668) 

s_sub_i32 s18, s0, s19 // (op 668) 


s_mul_i32 s0, s47, s46 // (op 668) 
s_add_u32 s68, s6, 16 // (op 668) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 668) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 668) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 668) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 668) 

s_sub_i32 s19, s68, s0 // (op 668) 

s_mul_i32 s51, s18, s19 // (op 668) 


s_sub_u32 s18, s8, s51 // (op 668) 


s_add_u32 s8, s18, s0 // (op 668) 


s_cmp_eq_i32 s9, 0 // (op 668) 
s_mov_b32 s0, scc // (op 668) 

s_cmp_eq_i32 s0, 0 // (op 668) 
s_cselect_b32 s9, s8, s1 // (op 668) 



v_lshl_add_u32 v59, s9, 4, v63 // (op 668) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 668) 
s_lshr_b32 s51, s0, 1 // (op 668) 

v_lshl_add_u32 v63, s51, 4, v67 // (op 668) 


v_mul_lo_u32 v67, v59, s28 // (op 668) 
v_add_u32 v67, v67, v63 // (op 668) 


s_mul_i32 s1, s28, s7 // (op 670) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 670) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 670) least significant
s_mul_hi_u32 s8, s28, s6 // (op 670) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 670) most significant: sum
s_add_u32 s1, s1, s8 // (op 670) most significant: sum


s_mov_b32 s60, s30 // (op 670) reg expression
s_mov_b32 s61, s31 // (op 670) reg expression
s_mov_b32 s62, s0 // (op 670) reg expression
s_mov_b32 s63, 131072 // (op 670) call()

s_mul_hi_i32 s51, s16, s11 // (op 732) 
v_and_b32 v59, -16, v1 // (op 732) 
s_add_i32 s0, s51, s16 // (op 732) 

v_sub_u32 v63, v1, v59 // (op 732) 

s_ashr_i32 s1, s0, 31 // (op 732) 
v_lshrrev_b32 v59, 4, v1 // (op 732) 
s_cmp_eq_i32 s11, 0 // (op 732) 
s_mov_b32 s51, scc // (op 732) 
v_lshlrev_b32 v15, 1, v59 // (op 732) 

s_cmp_eq_i32 s51, 0 // (op 732) 
s_cselect_b32 s8, 0, -1 // (op 732) 

s_and_b32 s9, s12, 31 // (op 732) 
s_lshl_b32 s51, 1, s9 // (op 732) 
s_add_i32 s18, s51, s8 // (op 732) 


s_and_b32 s8, s1, s18 // (op 732) 


s_add_i32 s1, s0, s8 // (op 732) 


s_ashr_i32 s0, s1, s9 // (op 732) 


s_lshl_b32 s1, s12, 24 // (op 732) 
s_ashr_i32 s51, s1, 31 // (op 732) 

s_xor_b32 s1, s0, s51 // (op 732) 

s_sub_i32 s0, s1, s51 // (op 732) 


s_mul_hi_i32 s1, s0, s13 // (op 732) 
s_add_i32 s51, s1, s0 // (op 732) 

s_ashr_i32 s1, s51, 31 // (op 732) 
s_cmp_eq_i32 s13, 0 // (op 732) 
s_mov_b32 s8, scc // (op 732) 
s_cmp_eq_i32 s8, 0 // (op 732) 
s_cselect_b32 s9, 0, -1 // (op 732) 

s_and_b32 s8, s14, 31 // (op 732) 
s_lshl_b32 s18, 1, s8 // (op 732) 
s_add_i32 s19, s18, s9 // (op 732) 


s_and_b32 s9, s1, s19 // (op 732) 


s_add_i32 s1, s51, s9 // (op 732) 


s_ashr_i32 s9, s1, s8 // (op 732) 


s_lshl_b32 s1, s14, 24 // (op 732) 
s_ashr_i32 s8, s1, 31 // (op 732) 

s_xor_b32 s1, s9, s8 // (op 732) 

s_sub_i32 s9, s1, s8 // (op 732) 


s_add_u32 s18, s2, 16 // (op 732) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 732) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 732) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 732) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 732) 

s_mul_i32 s1, s9, s18 // (op 732) 

s_sub_i32 s51, s0, s1 // (op 732) 


s_mul_hi_i32 s0, s16, s10 // (op 732) 
s_add_i32 s1, s0, s16 // (op 732) 

s_ashr_i32 s0, s1, 31 // (op 732) 
s_cmp_eq_i32 s10, 0 // (op 732) 
s_mov_b32 s8, scc // (op 732) 
s_cmp_eq_i32 s8, 0 // (op 732) 
s_cselect_b32 s9, 0, -1 // (op 732) 

s_and_b32 s8, s49, 31 // (op 732) 
s_lshl_b32 s68, 1, s8 // (op 732) 
s_add_i32 s69, s68, s9 // (op 732) 


s_and_b32 s9, s0, s69 // (op 732) 


s_add_i32 s0, s1, s9 // (op 732) 


s_ashr_i32 s1, s0, s8 // (op 732) 


s_lshl_b32 s0, s49, 24 // (op 732) 
s_ashr_i32 s8, s0, 31 // (op 732) 

s_xor_b32 s0, s1, s8 // (op 732) 

s_sub_i32 s1, s0, s8 // (op 732) 


s_mul_i32 s0, s46, s18 // (op 732) 

s_mul_i32 s8, s50, s0 // (op 732) 

s_mul_i32 s0, s1, s8 // (op 732) 

s_sub_u32 s8, s16, s0 // (op 732) 

s_mul_hi_i32 s0, s8, s15 // (op 732) 
s_add_i32 s9, s0, s8 // (op 732) 


s_ashr_i32 s0, s9, 31 // (op 732) 
s_cmp_eq_i32 s15, 0 // (op 732) 
s_mov_b32 s8, scc // (op 732) 
s_cmp_eq_i32 s8, 0 // (op 732) 
s_cselect_b32 s18, 0, -1 // (op 732) 

s_and_b32 s8, s48, 31 // (op 732) 
s_lshl_b32 s19, 1, s8 // (op 732) 
s_add_i32 s68, s19, s18 // (op 732) 


s_and_b32 s18, s0, s68 // (op 732) 


s_add_i32 s0, s9, s18 // (op 732) 


s_ashr_i32 s9, s0, s8 // (op 732) 


s_lshl_b32 s0, s48, 24 // (op 732) 
s_ashr_i32 s8, s0, 31 // (op 732) 

s_xor_b32 s0, s9, s8 // (op 732) 

s_sub_i32 s9, s0, s8 // (op 732) 


s_cmp_eq_i32 s1, 0 // (op 732) 
s_mov_b32 s0, scc // (op 732) 

s_cmp_eq_i32 s0, 0 // (op 732) 
s_cselect_b32 s1, s9, s51 // (op 732) 



v_lshl_add_u32 v59, s1, 4, v63 // (op 732) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 732) 
s_lshr_b32 s51, s0, 1 // (op 732) 

v_lshl_add_u32 v63, s51, 4, v15 // (op 732) 


v_mul_lo_u32 v15, v63, s32 // (op 732) 
v_add_u32 v15, v15, v59 // (op 732) 


s_mul_i32 s1, s32, s37 // (op 734) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 734) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 734) least significant
s_mul_hi_u32 s8, s32, s36 // (op 734) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 734) most significant: sum
s_add_u32 s1, s1, s8 // (op 734) most significant: sum


s_mov_b32 s64, s34 // (op 734) reg expression
s_mov_b32 s65, s35 // (op 734) reg expression
s_mov_b32 s66, s0 // (op 734) reg expression
s_mov_b32 s67, 131072 // (op 734) call()

v_and_b32 v59, -8, v1 // (op 825) 
v_sub_u32 v63, v1, v59 // (op 825) 

v_lshlrev_b32 v59, 5, v63 // (op 825) 

v_lshrrev_b32 v63, 3, v1 // (op 825) 
v_lshl_add_u32 v16, v63, 9, v59 // (op 825) 


v_lshlrev_b32 v59, 2, v16 // (op 825) 

v_lshrrev_b32 v63, 3, v59 // (op 825) 

v_lshlrev_b32 v59, 6, v1 // (op 878) 
v_mul_lo_u32 v16, v59, 6 // (op 878) 
v_lshrrev_b32 v17, 3, v16 // (op 878) 

v_lshrrev_b32 v16, 4, v59 // (op 878) 

v_lshlrev_b32 v59, 2, v16 // (op 878) 

v_add_u32 v16, v17, v59 // (op 878) 


v_and_b32 v59, -8, v1 // (op 931) 
v_sub_u32 v17, v1, v59 // (op 931) 

v_lshrrev_b32 v59, 3, v1 // (op 931) 
v_lshl_add_u32 v18, v59, 4, v17 // (op 931) 


v_and_b32 v17, -16, v1 // (op 984) 
v_sub_u32 v59, v1, v17 // (op 984) 

v_lshrrev_b32 v17, 4, v1 // (op 984) 
v_lshl_add_u32 v19, v17, 5, v59 // (op 984) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 198) 
s_and_b32 s51, s0, 4294967294 // (op 198) 

s_ashr_i64 s[0:1], s[26:27], 8 // (op 197) 
s_mov_b32 s8, s0 // (op 197) convert

s_cmp_lt_u32 s51, s8 // (op 197) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_197 // (op 197) Condition: Top (jump to RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_197 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_197:

s_add_u32 s51, s51, 1 // (op 199) 
s_lshl_b64 s[0:1], s[4:5], 2 // (op 202) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 202) 


v_mov_b32 v12, v13 // (op 202) 
v_mov_b32 v14, v12 // (op 202) 
buffer_load_dwordx4 v[22:25], v14, s[52:55], 0 offen // (op 202) Load value
v_add_u32 v14, v14, v20 // (op 202) 
buffer_load_dwordx4 v[26:29], v14, s[52:55], 0 offen // (op 202) Load value



v_add_u32 v13, 128, v13 // (op 527) 

s_barrier  // (op 1185) 
s_mul_i32 s1, s23, 6 // (op 225) most significant: low of high * low



s_mul_i32 s0, s22, 6 // (op 225) least significant
s_mul_hi_u32 s8, s22, 6 // (op 225) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 225) most significant: sum

v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 225) 


v_mov_b32 v12, v55 // (op 225) 
v_mov_b32 v14, v12 // (op 225) reg expression
buffer_load_dwordx3 v[30:32], v14, s[56:59], 0 offen // (op 225) Load value
v_add_u32 v14, v14, v20 // (op 225) 
s_waitcnt vmcnt(2)
ds_write_b128 v63, v[22:25] offset:4224 // (op 203) Store local data 
buffer_load_dwordx3 v[34:36], v14, s[56:59], 0 offen // (op 225) Load value
v_add_u32 v14, v14, v20 // (op 225) 
buffer_load_dwordx3 v[38:40], v14, s[56:59], 0 offen // (op 225) Load value
v_add_u32 v14, v14, v20 // (op 225) 
buffer_load_dwordx3 v[42:44], v14, s[56:59], 0 offen // (op 225) Load value



s_lshl_b64 s[0:1], s[22:23], 8 // (op 596) 
s_mul_i32 s9, s1, 6 // (op 596) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 596) least significant
s_mul_hi_u32 s18, s0, 6 // (op 596) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 596) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 596) 

v_add_u32 v55, v55, s0 // (op 596) 

s_waitcnt vmcnt(4)
ds_write_b128 v63, v[26:29] offset:4352 // (op 203) Store local data 


v_mov_b64 v[20:21], s[28:29] // (op 226) convert

s_barrier  // (op 1204) 
v_mov_b32 v12, v67 // (op 226) 



v_mov_b32 v14, v12 // (op 226) 
s_waitcnt vmcnt(3)
ds_write_b96 v16, v[30:32] // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v16, v[34:36] offset:16 // (op 222) Store local data 
buffer_load_ubyte v22, v14, s[60:63], 0 offen offset: 0 // (op 226) Load value
v_add_u32 v14, v14, v20 // (op 226) 
buffer_load_ubyte v23, v14, s[60:63], 0 offen offset: 0 // (op 226) Load value



v_add_u32 v67, 8, v67 // (op 660) 
s_waitcnt vmcnt(3)
ds_write_b96 v16, v[38:40] offset:32 // (op 222) Store local data 

v_mov_b64 v[20:21], s[32:33] // (op 227) convert

v_mov_b32 v12, v15 // (op 227) 
v_mov_b32 v14, v12 // (op 227) reg expression
buffer_load_ubyte v24, v14, s[64:67], 0 offen offset: 0 // (op 227) Load value
v_add_u32 v14, v14, v20 // (op 227) 
s_waitcnt vmcnt(3)
ds_write_b96 v16, v[42:44] offset:48 // (op 222) Store local data 
buffer_load_ubyte v25, v14, s[64:67], 0 offen offset: 0 // (op 227) Load value




s_lshl_b32 s0, s32, 3 // (op 724) 
s_barrier  // (op 1225) 
v_add_u32 v15, v15, s0 // (op 724) 




s_waitcnt lgkmcnt(4)
s_barrier  // (op 1194) 
s_waitcnt vmcnt(3)
ds_write_b8 v18, v22 offset:6272 // (op 223) Store local data 



ds_read_b128 v[26:29], v3 offset:4224 // (op 205) Load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v18, v23 offset:6280 // (op 223) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1215) 



s_barrier  // (op 1245) 
ds_read_b96_tr_b6 v[30:32], v5 // (op 207) Transpose load local data 



ds_read_b96_tr_b6 v[33:35], v5 offset:256 // (op 207) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v24 offset:4096 // (op 224) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1235) 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v25 offset:4112 // (op 224) Store local data 




ds_read_u8 v12, v0 offset:6272 // (op 209) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1255) 


ds_read_u8 v14, v7 offset:4096 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:35], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 212) 







ds_read_b128 v[20:23], v3 offset:4288 // (op 214) Load local data 



ds_read_b96_tr_b6 v[24:26], v5 offset:2048 // (op 216) Transpose load local data 
ds_read_b96_tr_b6 v[27:29], v5 offset:2304 // (op 216) Transpose load local data 



ds_read_u8 v12, v0 offset:6276 // (op 218) Load local data 


ds_read_u8 v14, v7 offset:4160 // (op 220) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:29], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 221) 




s_ashr_i64 s[0:1], s[26:27], 8 // (op 197) 
s_mov_b32 s8, s0 // (op 197) convert

s_cmp_lt_u32 s51, s8 // (op 197) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_197 // (op 197) Condition: Bottom (jump to RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_197 if true)
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_197:























s_barrier  // (op 1336) 








v_cmp_eq_f32 vcc, 0.00000, v8 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_157
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v10, v4 // (op 50) 
v_mul_f32 v13, v10, v5 // (op 50) 
v_mul_f32 v14, v10, v6 // (op 50) 
v_mul_f32 v15, v10, v7 // (op 50) 

v_lshrrev_b32 v4, 16, v12 // (op 56) 
v_lshrrev_b32 v5, 16, v13 // (op 56) 
v_lshrrev_b32 v6, 16, v14 // (op 56) 
v_lshrrev_b32 v7, 16, v15 // (op 56) 

s_mul_hi_i32 s0, s16, s11 // (op 1026) 
v_lshrrev_b32 v0, 6, v1 // (op 1026) 
s_add_i32 s1, s0, s16 // (op 1026) 

v_and_b32 v3, 63, v1 // (op 1026) 
s_ashr_i32 s0, s1, 31 // (op 1026) 
v_lshrrev_b32 v9, 4, v3 // (op 1026) 
s_cmp_eq_i32 s11, 0 // (op 1026) 
s_mov_b32 s4, scc // (op 1026) 
v_lshlrev_b32 v11, 2, v9 // (op 1026) 

s_cmp_eq_i32 s4, 0 // (op 1026) 
s_cselect_b32 s5, 0, -1 // (op 1026) 

v_lshl_add_u32 v9, v0, 4, v11 // (op 1026) 


s_and_b32 s4, s12, 31 // (op 1026) 
v_lshrrev_b32 v0, 2, v3 // (op 1026) 

s_lshl_b32 s8, 1, s4 // (op 1026) 
v_and_b32 v3, 3, v0 // (op 1026) 

s_add_i32 s9, s8, s5 // (op 1026) 


v_and_b32 v0, 3, v1 // (op 1026) 
s_and_b32 s5, s0, s9 // (op 1026) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 1026) 


s_add_i32 s0, s1, s5 // (op 1026) 


s_ashr_i32 s1, s0, s4 // (op 1026) 


s_lshl_b32 s0, s12, 24 // (op 1026) 
s_ashr_i32 s4, s0, 31 // (op 1026) 

s_xor_b32 s0, s1, s4 // (op 1026) 

s_sub_i32 s1, s0, s4 // (op 1026) 


s_mul_hi_i32 s0, s1, s13 // (op 1026) 
s_add_i32 s4, s0, s1 // (op 1026) 

s_ashr_i32 s0, s4, 31 // (op 1026) 
s_cmp_eq_i32 s13, 0 // (op 1026) 
s_mov_b32 s5, scc // (op 1026) 
s_cmp_eq_i32 s5, 0 // (op 1026) 
s_cselect_b32 s8, 0, -1 // (op 1026) 

s_and_b32 s5, s14, 31 // (op 1026) 
s_lshl_b32 s9, 1, s5 // (op 1026) 
s_add_i32 s18, s9, s8 // (op 1026) 


s_and_b32 s8, s0, s18 // (op 1026) 


s_add_i32 s0, s4, s8 // (op 1026) 


s_ashr_i32 s4, s0, s5 // (op 1026) 


s_lshl_b32 s0, s14, 24 // (op 1026) 
s_ashr_i32 s5, s0, 31 // (op 1026) 

s_xor_b32 s0, s4, s5 // (op 1026) 

s_sub_i32 s4, s0, s5 // (op 1026) 


s_mul_i32 s0, s1, s46 // (op 1026) 
s_sub_u32 s5, s16, s0 // (op 1026) 

s_mul_i32 s0, s4, s46 // (op 1026) 
s_add_u32 s0, s0, s5 // (op 1026) 

s_mul_hi_i32 s5, s16, s10 // (op 1026) 
s_add_i32 s8, s5, s16 // (op 1026) 

s_ashr_i32 s5, s8, 31 // (op 1026) 
s_cmp_eq_i32 s10, 0 // (op 1026) 
s_mov_b32 s9, scc // (op 1026) 
s_cmp_eq_i32 s9, 0 // (op 1026) 
s_cselect_b32 s18, 0, -1 // (op 1026) 

s_and_b32 s9, s49, 31 // (op 1026) 
s_lshl_b32 s19, 1, s9 // (op 1026) 
s_add_i32 s20, s19, s18 // (op 1026) 


s_and_b32 s18, s5, s20 // (op 1026) 


s_add_i32 s5, s8, s18 // (op 1026) 


s_ashr_i32 s8, s5, s9 // (op 1026) 


s_lshl_b32 s5, s49, 24 // (op 1026) 
s_ashr_i32 s9, s5, 31 // (op 1026) 

s_xor_b32 s5, s8, s9 // (op 1026) 

s_sub_i32 s8, s5, s9 // (op 1026) 


s_add_u32 s18, s2, 16 // (op 1026) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 1026) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 1026) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 1026) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 1026) 

s_mul_i32 s5, s46, s18 // (op 1026) 
s_mul_i32 s9, s50, s5 // (op 1026) 

s_mul_i32 s5, s8, s9 // (op 1026) 

s_sub_u32 s9, s16, s5 // (op 1026) 

s_mov_b32 s5, s9 // (op 1026) 
s_mul_hi_i32 s20, s5, s15 // (op 1026) 
s_add_i32 s21, s20, s5 // (op 1026) 


s_ashr_i32 s5, s21, 31 // (op 1026) 
s_cmp_eq_i32 s15, 0 // (op 1026) 
s_mov_b32 s20, scc // (op 1026) 
s_cmp_eq_i32 s20, 0 // (op 1026) 
s_cselect_b32 s22, 0, -1 // (op 1026) 

s_and_b32 s20, s48, 31 // (op 1026) 
s_lshl_b32 s23, 1, s20 // (op 1026) 
s_add_i32 s24, s23, s22 // (op 1026) 


s_and_b32 s22, s5, s24 // (op 1026) 


s_add_i32 s5, s21, s22 // (op 1026) 


s_ashr_i32 s21, s5, s20 // (op 1026) 


s_lshl_b32 s5, s48, 24 // (op 1026) 
s_ashr_i32 s20, s5, 31 // (op 1026) 

s_xor_b32 s5, s21, s20 // (op 1026) 

s_sub_i32 s21, s5, s20 // (op 1026) 


s_mul_i32 s5, s47, s46 // (op 1026) 
s_add_u32 s22, s6, 16 // (op 1026) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1026) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1026) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1026) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1026) 

s_sub_i32 s20, s22, s5 // (op 1026) 

s_mul_i32 s22, s21, s20 // (op 1026) 

s_sub_u32 s20, s9, s22 // (op 1026) 


s_add_u32 s9, s20, s5 // (op 1026) 


s_cmp_eq_i32 s8, 0 // (op 1026) 
s_mov_b32 s5, scc // (op 1026) 

s_cmp_eq_i32 s5, 0 // (op 1026) 
s_cselect_b32 s8, s9, s0 // (op 1026) 


s_mul_i32 s0, s4, s18 // (op 1026) 


v_lshl_add_u32 v0, s8, 4, v9 // (op 1026) 


s_sub_i32 s4, s1, s0 // (op 1026) 


s_cmp_eq_i32 s5, 0 // (op 1026) 
s_cselect_b32 s0, s21, s4 // (op 1026) 



v_lshl_add_u32 v3, s0, 4, v11 // (op 1026) 


v_mul_lo_u32 v9, v3, s38 // (op 1026) 

v_add_lshl_u32 v3, v0, v9, 1 // (op 1026) 


s_mul_i32 s1, s38, s3 // (op 1028) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 1028) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 1028) least significant
s_mul_hi_u32 s5, s38, s2 // (op 1028) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 1028) most significant: sum
s_add_u32 s1, s1, s5 // (op 1028) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 1 // (op 1028) 

s_mov_b32 s20, s40 // (op 1028) reg expression
s_mov_b32 s21, s41 // (op 1028) reg expression
s_mov_b32 s22, s4 // (op 1028) reg expression
s_mov_b32 s23, 131072 // (op 1028) call()


v_pack_b32_f16 v12, v4, v5 // (op 58) 
v_pack_b32_f16 v13, v6, v7 // (op 58) 


buffer_store_dwordx2 v[12:13], v3, s[20:23], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_157
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_157:
s_mul_hi_i32 s0, s16, s11 // (op 795) 
v_lshrrev_b32 v0, 6, v1 // (op 795) 
s_add_i32 s1, s0, s16 // (op 795) 

v_and_b32 v9, 63, v1 // (op 795) 
s_ashr_i32 s0, s1, 31 // (op 795) 
v_lshrrev_b32 v3, 4, v9 // (op 795) 
s_cmp_eq_i32 s11, 0 // (op 795) 
s_mov_b32 s4, scc // (op 795) 
v_lshlrev_b32 v4, 2, v3 // (op 795) 

s_cmp_eq_i32 s4, 0 // (op 795) 
s_cselect_b32 s5, 0, -1 // (op 795) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 795) 


s_and_b32 s4, s12, 31 // (op 795) 
v_lshrrev_b32 v0, 2, v9 // (op 795) 

s_lshl_b32 s8, 1, s4 // (op 795) 
v_and_b32 v9, 3, v0 // (op 795) 

s_add_i32 s9, s8, s5 // (op 795) 


v_and_b32 v0, 3, v1 // (op 795) 
s_and_b32 s5, s0, s9 // (op 795) 


v_lshl_add_u32 v4, v9, 2, v0 // (op 795) 


s_add_i32 s0, s1, s5 // (op 795) 


s_ashr_i32 s1, s0, s4 // (op 795) 


s_lshl_b32 s0, s12, 24 // (op 795) 
s_ashr_i32 s4, s0, 31 // (op 795) 

s_xor_b32 s0, s1, s4 // (op 795) 

s_sub_i32 s1, s0, s4 // (op 795) 


s_mul_hi_i32 s0, s1, s13 // (op 795) 
s_add_i32 s4, s0, s1 // (op 795) 

s_ashr_i32 s0, s4, 31 // (op 795) 
s_cmp_eq_i32 s13, 0 // (op 795) 
s_mov_b32 s5, scc // (op 795) 
s_cmp_eq_i32 s5, 0 // (op 795) 
s_cselect_b32 s8, 0, -1 // (op 795) 

s_and_b32 s5, s14, 31 // (op 795) 
s_lshl_b32 s9, 1, s5 // (op 795) 
s_add_i32 s18, s9, s8 // (op 795) 


s_and_b32 s19, s0, s18 // (op 795) 


s_add_i32 s0, s4, s19 // (op 795) 


s_ashr_i32 s4, s0, s5 // (op 795) 


s_lshl_b32 s0, s14, 24 // (op 795) 
s_ashr_i32 s5, s0, 31 // (op 795) 

s_xor_b32 s0, s4, s5 // (op 795) 

s_sub_i32 s4, s0, s5 // (op 795) 


s_mul_i32 s0, s1, s46 // (op 795) 
s_sub_u32 s5, s16, s0 // (op 795) 

s_mul_i32 s0, s4, s46 // (op 795) 
s_add_u32 s0, s0, s5 // (op 795) 

s_mul_hi_i32 s5, s16, s10 // (op 795) 
s_add_i32 s8, s5, s16 // (op 795) 

s_ashr_i32 s5, s8, 31 // (op 795) 
s_cmp_eq_i32 s10, 0 // (op 795) 
s_mov_b32 s9, scc // (op 795) 
s_cmp_eq_i32 s9, 0 // (op 795) 
s_cselect_b32 s18, 0, -1 // (op 795) 

s_and_b32 s9, s49, 31 // (op 795) 
s_lshl_b32 s19, 1, s9 // (op 795) 
s_add_i32 s24, s19, s18 // (op 795) 


s_and_b32 s18, s5, s24 // (op 795) 


s_add_i32 s5, s8, s18 // (op 795) 


s_ashr_i32 s8, s5, s9 // (op 795) 


s_lshl_b32 s5, s49, 24 // (op 795) 
s_ashr_i32 s9, s5, 31 // (op 795) 

s_xor_b32 s5, s8, s9 // (op 795) 

s_sub_i32 s8, s5, s9 // (op 795) 


s_add_u32 s18, s2, 16 // (op 795) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 795) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 795) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 795) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 4 // (op 795) 

s_mul_i32 s5, s46, s18 // (op 795) 
s_mul_i32 s9, s50, s5 // (op 795) 

s_mul_i32 s5, s8, s9 // (op 795) 

s_sub_u32 s9, s16, s5 // (op 795) 

s_mov_b32 s5, s9 // (op 795) 
s_mul_hi_i32 s24, s5, s15 // (op 795) 
s_add_i32 s25, s24, s5 // (op 795) 


s_ashr_i32 s5, s25, 31 // (op 795) 
s_cmp_eq_i32 s15, 0 // (op 795) 
s_mov_b32 s24, scc // (op 795) 
s_cmp_eq_i32 s24, 0 // (op 795) 
s_cselect_b32 s26, 0, -1 // (op 795) 

s_and_b32 s24, s48, 31 // (op 795) 
s_lshl_b32 s27, 1, s24 // (op 795) 
s_add_i32 s28, s27, s26 // (op 795) 


s_and_b32 s26, s5, s28 // (op 795) 


s_add_i32 s5, s25, s26 // (op 795) 


s_ashr_i32 s25, s5, s24 // (op 795) 


s_lshl_b32 s5, s48, 24 // (op 795) 
s_ashr_i32 s24, s5, 31 // (op 795) 

s_xor_b32 s5, s25, s24 // (op 795) 

s_sub_i32 s25, s5, s24 // (op 795) 


s_mul_i32 s5, s47, s46 // (op 795) 
s_add_u32 s26, s6, 16 // (op 795) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 795) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 795) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 795) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 795) 

s_sub_i32 s24, s26, s5 // (op 795) 

s_mul_i32 s26, s25, s24 // (op 795) 

s_sub_u32 s24, s9, s26 // (op 795) 


s_add_u32 s9, s24, s5 // (op 795) 


s_cmp_eq_i32 s8, 0 // (op 795) 
s_mov_b32 s5, scc // (op 795) 

s_cmp_eq_i32 s5, 0 // (op 795) 
s_cselect_b32 s8, s9, s0 // (op 795) 


s_mul_i32 s0, s4, s18 // (op 795) 


v_lshl_add_u32 v0, s8, 4, v3 // (op 795) 


s_sub_i32 s4, s1, s0 // (op 795) 


s_cmp_eq_i32 s5, 0 // (op 795) 
s_cselect_b32 s24, s25, s4 // (op 795) 



v_lshl_add_u32 v3, s24, 4, v4 // (op 795) 


v_mul_lo_u32 v9, v3, s42 // (op 795) 

v_add_lshl_u32 v3, v0, v9, 1 // (op 795) 


s_mul_i32 s1, s42, s3 // (op 797) most significant: low of low * high
s_mul_i32 s4, s43, s2 // (op 797) most significant: low of high * low
s_mul_i32 s0, s42, s2 // (op 797) least significant
s_mul_hi_u32 s5, s42, s2 // (op 797) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 797) most significant: sum
s_add_u32 s1, s1, s5 // (op 797) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 1 // (op 797) 

s_mov_b32 s24, s44 // (op 797) reg expression
s_mov_b32 s25, s45 // (op 797) reg expression
s_mov_b32 s26, s4 // (op 797) reg expression
s_mov_b32 s27, 131072 // (op 797) call()





buffer_load_dwordx2 v[4:5], v3, s[24:27], 0 offen // (op 170) Load value



v_accvgpr_read v12, a0 // (op 166) 
v_accvgpr_read v13, a1 // (op 166) 
v_accvgpr_read v14, a2 // (op 166) 
v_accvgpr_read v15, a3 // (op 166) 
v_mul_f32 v16, v10, v12 // (op 166) 
v_mul_f32 v17, v10, v13 // (op 166) 
v_mul_f32 v18, v10, v14 // (op 166) 
v_mul_f32 v19, v10, v15 // (op 166) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 171) 
v_lshrrev_b32 v7, 16, v4 // (op 171) 
v_lshlrev_b32 v6, 16, v6 // (op 171) 
v_lshlrev_b32 v7, 16, v7 // (op 171) 
v_mul_f32 v12, v8, v6 // (op 171) 
v_mul_f32 v13, v8, v7 // (op 171) 
v_and_b32 v6, 65535, v5 // (op 171) 
v_lshrrev_b32 v7, 16, v5 // (op 171) 
v_lshlrev_b32 v6, 16, v6 // (op 171) 
v_lshlrev_b32 v7, 16, v7 // (op 171) 
v_mul_f32 v14, v8, v6 // (op 171) 
v_mul_f32 v15, v8, v7 // (op 171) 


v_add_f32 v4, v12, v16 // (op 167) 
v_add_f32 v5, v13, v17 // (op 167) 
v_add_f32 v6, v14, v18 // (op 167) 
v_add_f32 v7, v15, v19 // (op 167) 


v_lshrrev_b32 v12, 16, v4 // (op 168) 
v_lshrrev_b32 v13, 16, v5 // (op 168) 
v_lshrrev_b32 v14, 16, v6 // (op 168) 
v_lshrrev_b32 v15, 16, v7 // (op 168) 

s_mul_hi_i32 s0, s16, s11 // (op 1043) 
v_lshrrev_b32 v0, 6, v1 // (op 1043) 
s_add_i32 s1, s0, s16 // (op 1043) 

v_and_b32 v9, 63, v1 // (op 1043) 
s_ashr_i32 s0, s1, 31 // (op 1043) 
v_lshrrev_b32 v11, 4, v9 // (op 1043) 
s_cmp_eq_i32 s11, 0 // (op 1043) 
s_mov_b32 s4, scc // (op 1043) 
v_lshlrev_b32 v3, 2, v11 // (op 1043) 

s_cmp_eq_i32 s4, 0 // (op 1043) 
s_cselect_b32 s5, 0, -1 // (op 1043) 

v_lshl_add_u32 v11, v0, 4, v3 // (op 1043) 


s_and_b32 s4, s12, 31 // (op 1043) 
v_lshrrev_b32 v0, 2, v9 // (op 1043) 

s_lshl_b32 s8, 1, s4 // (op 1043) 
v_and_b32 v9, 3, v0 // (op 1043) 

s_add_i32 s9, s8, s5 // (op 1043) 


v_and_b32 v0, 3, v1 // (op 1043) 
s_and_b32 s5, s0, s9 // (op 1043) 


v_lshl_add_u32 v3, v9, 2, v0 // (op 1043) 


s_add_i32 s0, s1, s5 // (op 1043) 


s_ashr_i32 s1, s0, s4 // (op 1043) 


s_lshl_b32 s0, s12, 24 // (op 1043) 
s_ashr_i32 s4, s0, 31 // (op 1043) 

s_xor_b32 s0, s1, s4 // (op 1043) 

s_sub_i32 s1, s0, s4 // (op 1043) 


s_mul_hi_i32 s0, s1, s13 // (op 1043) 
s_add_i32 s4, s0, s1 // (op 1043) 

s_ashr_i32 s0, s4, 31 // (op 1043) 
s_cmp_eq_i32 s13, 0 // (op 1043) 
s_mov_b32 s5, scc // (op 1043) 
s_cmp_eq_i32 s5, 0 // (op 1043) 
s_cselect_b32 s8, 0, -1 // (op 1043) 

s_and_b32 s5, s14, 31 // (op 1043) 
s_lshl_b32 s9, 1, s5 // (op 1043) 
s_add_i32 s18, s9, s8 // (op 1043) 


s_and_b32 s19, s0, s18 // (op 1043) 


s_add_i32 s0, s4, s19 // (op 1043) 


s_ashr_i32 s4, s0, s5 // (op 1043) 


s_lshl_b32 s0, s14, 24 // (op 1043) 
s_ashr_i32 s5, s0, 31 // (op 1043) 

s_xor_b32 s0, s4, s5 // (op 1043) 

s_sub_i32 s4, s0, s5 // (op 1043) 


s_mul_i32 s0, s1, s46 // (op 1043) 
s_sub_u32 s5, s16, s0 // (op 1043) 

s_mul_i32 s0, s4, s46 // (op 1043) 
s_add_u32 s0, s0, s5 // (op 1043) 

s_mul_hi_i32 s5, s16, s10 // (op 1043) 
s_add_i32 s8, s5, s16 // (op 1043) 

s_ashr_i32 s5, s8, 31 // (op 1043) 
s_cmp_eq_i32 s10, 0 // (op 1043) 
s_mov_b32 s9, scc // (op 1043) 
s_cmp_eq_i32 s9, 0 // (op 1043) 
s_cselect_b32 s18, 0, -1 // (op 1043) 

s_and_b32 s9, s49, 31 // (op 1043) 
s_lshl_b32 s19, 1, s9 // (op 1043) 
s_add_i32 s24, s19, s18 // (op 1043) 


s_and_b32 s18, s5, s24 // (op 1043) 


s_add_i32 s5, s8, s18 // (op 1043) 


s_ashr_i32 s8, s5, s9 // (op 1043) 


s_lshl_b32 s5, s49, 24 // (op 1043) 
s_ashr_i32 s9, s5, 31 // (op 1043) 

s_xor_b32 s5, s8, s9 // (op 1043) 

s_sub_i32 s8, s5, s9 // (op 1043) 


s_add_u32 s18, s2, 16 // (op 1043) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 1043) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 1043) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 1043) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 4 // (op 1043) 

s_mul_i32 s5, s46, s18 // (op 1043) 
s_mul_i32 s9, s50, s5 // (op 1043) 

s_mul_i32 s5, s8, s9 // (op 1043) 

s_sub_u32 s9, s16, s5 // (op 1043) 

s_mov_b32 s5, s9 // (op 1043) 
s_mul_hi_i32 s24, s5, s15 // (op 1043) 
s_add_i32 s25, s24, s5 // (op 1043) 


s_ashr_i32 s5, s25, 31 // (op 1043) 
s_cmp_eq_i32 s15, 0 // (op 1043) 
s_mov_b32 s24, scc // (op 1043) 
s_cmp_eq_i32 s24, 0 // (op 1043) 
s_cselect_b32 s26, 0, -1 // (op 1043) 

s_and_b32 s24, s48, 31 // (op 1043) 
s_lshl_b32 s27, 1, s24 // (op 1043) 
s_add_i32 s28, s27, s26 // (op 1043) 


s_and_b32 s26, s5, s28 // (op 1043) 


s_add_i32 s5, s25, s26 // (op 1043) 


s_ashr_i32 s25, s5, s24 // (op 1043) 


s_lshl_b32 s5, s48, 24 // (op 1043) 
s_ashr_i32 s24, s5, 31 // (op 1043) 

s_xor_b32 s5, s25, s24 // (op 1043) 

s_sub_i32 s25, s5, s24 // (op 1043) 


s_mul_i32 s5, s47, s46 // (op 1043) 
s_add_u32 s26, s6, 16 // (op 1043) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 1043) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 1043) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 1043) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 1043) 

s_sub_i32 s24, s26, s5 // (op 1043) 

s_mul_i32 s26, s25, s24 // (op 1043) 

s_sub_u32 s24, s9, s26 // (op 1043) 


s_add_u32 s9, s24, s5 // (op 1043) 


s_cmp_eq_i32 s8, 0 // (op 1043) 
s_mov_b32 s5, scc // (op 1043) 

s_cmp_eq_i32 s5, 0 // (op 1043) 
s_cselect_b32 s8, s9, s0 // (op 1043) 


s_mul_i32 s0, s4, s18 // (op 1043) 


v_lshl_add_u32 v0, s8, 4, v11 // (op 1043) 


s_sub_i32 s4, s1, s0 // (op 1043) 


s_cmp_eq_i32 s5, 0 // (op 1043) 
s_cselect_b32 s24, s25, s4 // (op 1043) 



v_lshl_add_u32 v9, s24, 4, v3 // (op 1043) 


v_mul_lo_u32 v11, v9, s38 // (op 1043) 

v_add_lshl_u32 v9, v0, v11, 1 // (op 1043) 


s_mul_i32 s1, s38, s3 // (op 1045) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 1045) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 1045) least significant
s_mul_hi_u32 s5, s38, s2 // (op 1045) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 1045) most significant: sum
s_add_u32 s1, s1, s5 // (op 1045) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 1 // (op 1045) 

s_mov_b32 s20, s40 // (op 1045) reg expression
s_mov_b32 s21, s41 // (op 1045) reg expression
s_mov_b32 s22, s4 // (op 1045) reg expression
s_mov_b32 s23, 131072 // (op 1045) call()


v_pack_b32_f16 v6, v12, v13 // (op 169) 
v_pack_b32_f16 v7, v14, v15 // (op 169) 


buffer_store_dwordx2 v[6:7], v9, s[20:23], 0 offen // (op 169) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_157:
















s_endpgm  // End of RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.LRR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end:
.size RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_, .LRR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end-RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 108
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12800 // lds bytes
  .amdhsa_accum_offset 104
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
  - .name: RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
    .symbol: RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 12800
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 103
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

