// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_,@function
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 63, v1 // (op 853) 
v_and_b32 v3, 48, v1 // (op 853) 
v_sub_u32 v4, v0, v3 // (op 853) 

v_lshrrev_b32 v3, 4, v0 // (op 853) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 853) 


v_lshrrev_b32 v3, 6, v1 // (op 853) 
v_lshlrev_b32 v4, 4, v3 // (op 853) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 853) 


v_lshlrev_b32 v0, 2, v3 // (op 853) 

v_lshrrev_b32 v3, 3, v0 // (op 853) 

v_and_b32 v0, 15, v1 // (op 906) 
v_and_b32 v4, 63, v1 // (op 906) 
v_lshrrev_b32 v5, 4, v4 // (op 906) 

v_lshlrev_b32 v4, 5, v5 // (op 906) 

v_lshl_add_u32 v5, v0, 8, v4 // (op 906) 


v_lshlrev_b32 v0, 2, v5 // (op 906) 

v_lshrrev_b32 v4, 3, v0 // (op 906) 

v_lshrrev_b32 v0, 6, v1 // (op 959) 
v_and_b32 v5, 15, v1 // (op 959) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 959) 


v_and_b32 v0, 48, v1 // (op 959) 
v_add_u32 v5, v0, v6 // (op 959) 


v_and_b32 v0, 15, v1 // (op 1012) 
v_and_b32 v6, 63, v1 // (op 1012) 
v_lshrrev_b32 v7, 4, v6 // (op 1012) 

v_lshl_add_u32 v6, v0, 3, v7 // (op 1012) 


v_and_b32 v0, 63, v1 // (op 1070) 
v_and_b32 v7, 48, v1 // (op 1070) 
v_sub_u32 v8, v0, v7 // (op 1070) 

v_lshrrev_b32 v7, 4, v0 // (op 1070) 

v_lshl_add_u32 v0, v7, 5, v8 // (op 1070) 


v_lshrrev_b32 v7, 6, v1 // (op 1070) 
v_lshlrev_b32 v8, 4, v7 // (op 1070) 

v_lshl_add_u32 v7, v0, 4, v8 // (op 1070) 


v_lshlrev_b32 v0, 2, v7 // (op 1070) 

v_lshrrev_b32 v7, 3, v0 // (op 1070) 

v_and_b32 v0, 15, v1 // (op 1094) 
v_and_b32 v8, 63, v1 // (op 1094) 
v_lshrrev_b32 v9, 4, v8 // (op 1094) 

v_lshlrev_b32 v8, 5, v9 // (op 1094) 

v_lshl_add_u32 v9, v0, 8, v8 // (op 1094) 


v_lshlrev_b32 v0, 2, v9 // (op 1094) 

v_lshrrev_b32 v8, 3, v0 // (op 1094) 

v_lshrrev_b32 v0, 6, v1 // (op 1118) 
v_and_b32 v9, 15, v1 // (op 1118) 
v_lshl_add_u32 v10, v0, 4, v9 // (op 1118) 


v_and_b32 v0, 48, v1 // (op 1118) 
v_add_u32 v9, v0, v10 // (op 1118) 


v_and_b32 v0, 15, v1 // (op 1142) 
v_and_b32 v10, 63, v1 // (op 1142) 
v_lshrrev_b32 v11, 4, v10 // (op 1142) 

v_lshl_add_u32 v10, v0, 3, v11 // (op 1142) 


v_mov_b32 v0, s6 // (op 2) Move value
v_mov_b32 v11, s7 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
s_mul_hi_i32 s0, s16, s9 // (op 563) 
v_lshlrev_b32 v12, 2, v1 // (op 563) 
s_add_i32 s1, s0, s16 // (op 563) 

s_ashr_i32 s0, s1, 31 // (op 563) 
s_cmp_eq_i32 s9, 0 // (op 563) 
s_mov_b32 s6, scc // (op 563) 
s_cmp_eq_i32 s6, 0 // (op 563) 
s_cselect_b32 s7, 0, -1 // (op 563) 

s_and_b32 s6, s10, 31 // (op 563) 
s_lshl_b32 s18, 1, s6 // (op 563) 
s_add_i32 s19, s18, s7 // (op 563) 


s_and_b32 s7, s0, s19 // (op 563) 


s_add_i32 s0, s1, s7 // (op 563) 


s_ashr_i32 s1, s0, s6 // (op 563) 


s_lshl_b32 s0, s10, 24 // (op 563) 
s_ashr_i32 s6, s0, 31 // (op 563) 

s_xor_b32 s0, s1, s6 // (op 563) 

s_sub_i32 s1, s0, s6 // (op 563) 


s_mul_hi_i32 s0, s1, s11 // (op 563) 
s_add_i32 s6, s0, s1 // (op 563) 

s_ashr_i32 s0, s6, 31 // (op 563) 
s_cmp_eq_i32 s11, 0 // (op 563) 
s_mov_b32 s7, scc // (op 563) 
s_cmp_eq_i32 s7, 0 // (op 563) 
s_cselect_b32 s18, 0, -1 // (op 563) 

s_and_b32 s7, s12, 31 // (op 563) 
s_lshl_b32 s19, 1, s7 // (op 563) 
s_add_i32 s51, s19, s18 // (op 563) 


s_and_b32 s18, s0, s51 // (op 563) 


s_add_i32 s0, s6, s18 // (op 563) 


s_ashr_i32 s6, s0, s7 // (op 563) 


s_lshl_b32 s0, s12, 24 // (op 563) 
s_ashr_i32 s7, s0, 31 // (op 563) 

s_xor_b32 s0, s6, s7 // (op 563) 

s_sub_i32 s6, s0, s7 // (op 563) 


s_mul_i32 s0, s1, s14 // (op 563) 

s_sub_u32 s1, s16, s0 // (op 563) 

s_mul_i32 s0, s6, s14 // (op 563) 
s_add_u32 s0, s0, s1 // (op 563) 


s_mul_hi_i32 s1, s16, s8 // (op 563) 
s_add_i32 s6, s1, s16 // (op 563) 

s_ashr_i32 s1, s6, 31 // (op 563) 
s_cmp_eq_i32 s8, 0 // (op 563) 
s_mov_b32 s7, scc // (op 563) 
s_cmp_eq_i32 s7, 0 // (op 563) 
s_cselect_b32 s18, 0, -1 // (op 563) 

s_and_b32 s7, s15, 31 // (op 563) 
s_lshl_b32 s19, 1, s7 // (op 563) 
s_add_i32 s51, s19, s18 // (op 563) 


s_and_b32 s18, s1, s51 // (op 563) 


s_add_i32 s1, s6, s18 // (op 563) 


s_ashr_i32 s6, s1, s7 // (op 563) 


s_lshl_b32 s1, s15, 24 // (op 563) 
s_ashr_i32 s7, s1, 31 // (op 563) 

s_xor_b32 s1, s6, s7 // (op 563) 

s_sub_i32 s6, s1, s7 // (op 563) 


s_add_u32 s18, s4, 16 // (op 563) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 563) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 563) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 563) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 563) 

s_mul_i32 s1, s14, s18 // (op 563) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 563) 

s_mul_i32 s1, s6, s7 // (op 563) 

s_sub_u32 s7, s16, s1 // (op 563) 

s_mov_b32 s1, s7 // (op 563) 
s_mul_hi_i32 s18, s1, s13 // (op 563) 
s_add_i32 s19, s18, s1 // (op 563) 


s_ashr_i32 s1, s19, 31 // (op 563) 
s_cmp_eq_i32 s13, 0 // (op 563) 
s_mov_b32 s18, scc // (op 563) 
s_cmp_eq_i32 s18, 0 // (op 563) 
s_cselect_b32 s51, 0, -1 // (op 563) 

s_and_b32 s18, s49, 31 // (op 563) 
s_lshl_b32 s52, 1, s18 // (op 563) 
s_add_i32 s53, s52, s51 // (op 563) 


s_and_b32 s51, s1, s53 // (op 563) 


s_add_i32 s1, s19, s51 // (op 563) 


s_ashr_i32 s19, s1, s18 // (op 563) 


s_lshl_b32 s1, s49, 24 // (op 563) 
s_ashr_i32 s18, s1, 31 // (op 563) 

s_xor_b32 s1, s19, s18 // (op 563) 

s_sub_i32 s19, s1, s18 // (op 563) 


s_mul_i32 s1, s48, s14 // (op 563) 
s_add_u32 s52, s20, 16 // (op 563) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 563) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 563) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 563) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 563) 

s_sub_i32 s18, s52, s1 // (op 563) 

s_mul_i32 s51, s19, s18 // (op 563) 


s_sub_u32 s18, s7, s51 // (op 563) 


s_add_u32 s7, s18, s1 // (op 563) 


s_cmp_eq_i32 s6, 0 // (op 563) 
s_mov_b32 s1, scc // (op 563) 

s_cmp_eq_i32 s1, 0 // (op 563) 
s_cselect_b32 s6, s7, s0 // (op 563) 



s_lshl_b32 s7, s6, 4 // (op 563) 

s_mov_b32 s1, 0 // (op 563) convert
s_mov_b32 s0, s7 // (op 563) convert

v_mul_lo_u32 v15, v12, s3 // (op 563) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 563) most significant: high of low * low
v_mul_lo_u32 v14, v12, s2 // (op 563) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 563) most significant: sum


v_mov_b32 v12, s0 // (op 563) 
v_mov_b32 v13, s1 // (op 563) 
v_add_co_u32 v16, s[6:7], v12, v14 // (op 563) least significant half
v_addc_co_u32 v17, s[6:7], v13, v15, s[6:7] // (op 563) most significant half



v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 563) 


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 563) 

v_mov_b32 v14, v12 // (op 563) convert

s_mul_i32 s1, s2, s25 // (op 565) most significant: low of low * high
s_mul_i32 s6, s3, s24 // (op 565) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 565) least significant
s_mul_hi_u32 s7, s2, s24 // (op 565) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 565) most significant: sum
s_add_u32 s1, s1, s7 // (op 565) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 565) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 565) 

s_mov_b32 s52, s22 // (op 565) reg expression
s_mov_b32 s53, s23 // (op 565) reg expression
s_mov_b32 s54, s0 // (op 565) reg expression
s_mov_b32 s55, 131072 // (op 565) call()

s_mul_hi_i32 s51, s16, s9 // (op 582) 
v_lshlrev_b32 v12, 2, v1 // (op 582) 
s_add_i32 s0, s51, s16 // (op 582) 

s_ashr_i32 s1, s0, 31 // (op 582) 
s_cmp_eq_i32 s9, 0 // (op 582) 
s_mov_b32 s51, scc // (op 582) 
s_cmp_eq_i32 s51, 0 // (op 582) 
s_cselect_b32 s6, 0, -1 // (op 582) 

s_and_b32 s7, s10, 31 // (op 582) 
s_lshl_b32 s51, 1, s7 // (op 582) 
s_add_i32 s18, s51, s6 // (op 582) 


s_and_b32 s6, s1, s18 // (op 582) 


s_add_i32 s1, s0, s6 // (op 582) 


s_ashr_i32 s0, s1, s7 // (op 582) 


s_lshl_b32 s1, s10, 24 // (op 582) 
s_ashr_i32 s51, s1, 31 // (op 582) 

s_xor_b32 s1, s0, s51 // (op 582) 

s_sub_i32 s0, s1, s51 // (op 582) 


s_mul_hi_i32 s1, s0, s11 // (op 582) 
s_add_i32 s51, s1, s0 // (op 582) 

s_ashr_i32 s1, s51, 31 // (op 582) 
s_cmp_eq_i32 s11, 0 // (op 582) 
s_mov_b32 s6, scc // (op 582) 
s_cmp_eq_i32 s6, 0 // (op 582) 
s_cselect_b32 s7, 0, -1 // (op 582) 

s_and_b32 s6, s12, 31 // (op 582) 
s_lshl_b32 s18, 1, s6 // (op 582) 
s_add_i32 s19, s18, s7 // (op 582) 


s_and_b32 s7, s1, s19 // (op 582) 


s_add_i32 s1, s51, s7 // (op 582) 


s_ashr_i32 s7, s1, s6 // (op 582) 


s_lshl_b32 s1, s12, 24 // (op 582) 
s_ashr_i32 s6, s1, 31 // (op 582) 

s_xor_b32 s1, s7, s6 // (op 582) 

s_sub_i32 s7, s1, s6 // (op 582) 


s_mul_i32 s1, s0, s14 // (op 582) 

s_sub_u32 s0, s16, s1 // (op 582) 

s_mul_i32 s1, s7, s14 // (op 582) 
s_add_u32 s1, s1, s0 // (op 582) 


s_mul_hi_i32 s0, s16, s8 // (op 582) 
s_add_i32 s51, s0, s16 // (op 582) 

s_ashr_i32 s0, s51, 31 // (op 582) 
s_cmp_eq_i32 s8, 0 // (op 582) 
s_mov_b32 s6, scc // (op 582) 
s_cmp_eq_i32 s6, 0 // (op 582) 
s_cselect_b32 s7, 0, -1 // (op 582) 

s_and_b32 s6, s15, 31 // (op 582) 
s_lshl_b32 s18, 1, s6 // (op 582) 
s_add_i32 s19, s18, s7 // (op 582) 


s_and_b32 s7, s0, s19 // (op 582) 


s_add_i32 s0, s51, s7 // (op 582) 


s_ashr_i32 s7, s0, s6 // (op 582) 


s_lshl_b32 s0, s15, 24 // (op 582) 
s_ashr_i32 s6, s0, 31 // (op 582) 

s_xor_b32 s0, s7, s6 // (op 582) 

s_sub_i32 s7, s0, s6 // (op 582) 


s_add_u32 s18, s4, 16 // (op 582) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 582) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 582) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 582) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 582) 

s_mul_i32 s0, s14, s18 // (op 582) 

s_mul_i32 s6, s50, s0 // (op 582) 

s_mul_i32 s0, s7, s6 // (op 582) 

s_sub_u32 s6, s16, s0 // (op 582) 

s_mov_b32 s0, s6 // (op 582) 
s_mul_hi_i32 s51, s0, s13 // (op 582) 
s_add_i32 s18, s51, s0 // (op 582) 


s_ashr_i32 s0, s18, 31 // (op 582) 
s_cmp_eq_i32 s13, 0 // (op 582) 
s_mov_b32 s19, scc // (op 582) 
s_cmp_eq_i32 s19, 0 // (op 582) 
s_cselect_b32 s51, 0, -1 // (op 582) 

s_and_b32 s19, s49, 31 // (op 582) 
s_lshl_b32 s56, 1, s19 // (op 582) 
s_add_i32 s57, s56, s51 // (op 582) 


s_and_b32 s51, s0, s57 // (op 582) 


s_add_i32 s0, s18, s51 // (op 582) 


s_ashr_i32 s18, s0, s19 // (op 582) 


s_lshl_b32 s0, s49, 24 // (op 582) 
s_ashr_i32 s19, s0, 31 // (op 582) 

s_xor_b32 s0, s18, s19 // (op 582) 

s_sub_i32 s18, s0, s19 // (op 582) 


s_mul_i32 s0, s48, s14 // (op 582) 
s_add_u32 s56, s20, 16 // (op 582) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 582) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 582) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 582) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 582) 

s_sub_i32 s19, s56, s0 // (op 582) 

s_mul_i32 s51, s18, s19 // (op 582) 


s_sub_u32 s18, s6, s51 // (op 582) 


s_add_u32 s6, s18, s0 // (op 582) 


s_cmp_eq_i32 s7, 0 // (op 582) 
s_mov_b32 s0, scc // (op 582) 

s_cmp_eq_i32 s0, 0 // (op 582) 
s_cselect_b32 s7, s6, s1 // (op 582) 



s_lshl_b32 s6, s7, 4 // (op 582) 

s_mov_b32 s1, 0 // (op 582) convert
s_mov_b32 s0, s6 // (op 582) convert

v_mul_lo_u32 v17, v12, s3 // (op 582) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 582) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 582) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 582) most significant: sum


v_mov_b32 v15, s0 // (op 582) 
v_mov_b32 v12, s1 // (op 582) 
v_add_co_u32 v18, s[6:7], v15, v16 // (op 582) least significant half
v_addc_co_u32 v19, s[6:7], v12, v17, s[6:7] // (op 582) most significant half



v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 582) 


v_lshrrev_b64 v[12:13], 3, v[18:19] // (op 582) 

v_mov_b32 v15, v12 // (op 582) convert

s_mul_i32 s1, s2, s25 // (op 584) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 584) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 584) least significant
s_mul_hi_u32 s6, s2, s24 // (op 584) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 584) most significant: sum
s_add_u32 s1, s1, s6 // (op 584) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 584) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 584) 

s_mov_b32 s52, s22 // (op 584) reg expression
s_mov_b32 s53, s23 // (op 584) reg expression
s_mov_b32 s54, s0 // (op 584) reg expression
s_mov_b32 s55, 131072 // (op 584) call()

s_mul_hi_i32 s51, s16, s9 // (op 627) 
v_lshrrev_b32 v12, 3, v1 // (op 627) 
s_add_i32 s0, s51, s16 // (op 627) 

v_lshlrev_b32 v13, 1, v12 // (op 627) 

s_ashr_i32 s1, s0, 31 // (op 627) 
v_and_b32 v12, -8, v1 // (op 627) 
s_cmp_eq_i32 s9, 0 // (op 627) 
s_mov_b32 s51, scc // (op 627) 
v_sub_u32 v16, v1, v12 // (op 627) 

s_cmp_eq_i32 s51, 0 // (op 627) 
s_cselect_b32 s6, 0, -1 // (op 627) 

v_lshlrev_b32 v12, 5, v16 // (op 627) 

v_mov_b32 v17, 0 // (op 627) convert
v_mov_b32 v16, v12 // (op 627) convert

s_and_b32 s7, s10, 31 // (op 627) 
s_lshl_b32 s51, 1, s7 // (op 627) 
s_add_i32 s18, s51, s6 // (op 627) 


s_and_b32 s6, s1, s18 // (op 627) 


s_add_i32 s1, s0, s6 // (op 627) 


s_ashr_i32 s0, s1, s7 // (op 627) 


s_lshl_b32 s1, s10, 24 // (op 627) 
s_ashr_i32 s51, s1, 31 // (op 627) 

s_xor_b32 s1, s0, s51 // (op 627) 

s_sub_i32 s0, s1, s51 // (op 627) 


s_mul_hi_i32 s1, s0, s11 // (op 627) 
s_add_i32 s51, s1, s0 // (op 627) 

s_ashr_i32 s1, s51, 31 // (op 627) 
s_cmp_eq_i32 s11, 0 // (op 627) 
s_mov_b32 s6, scc // (op 627) 
s_cmp_eq_i32 s6, 0 // (op 627) 
s_cselect_b32 s7, 0, -1 // (op 627) 

s_and_b32 s6, s12, 31 // (op 627) 
s_lshl_b32 s18, 1, s6 // (op 627) 
s_add_i32 s19, s18, s7 // (op 627) 


s_and_b32 s7, s1, s19 // (op 627) 


s_add_i32 s1, s51, s7 // (op 627) 


s_ashr_i32 s7, s1, s6 // (op 627) 


s_lshl_b32 s1, s12, 24 // (op 627) 
s_ashr_i32 s6, s1, 31 // (op 627) 

s_xor_b32 s1, s7, s6 // (op 627) 

s_sub_i32 s7, s1, s6 // (op 627) 


s_add_u32 s18, s4, 16 // (op 627) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 627) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 627) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 627) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 627) 

s_mul_i32 s1, s7, s18 // (op 627) 

s_sub_i32 s51, s0, s1 // (op 627) 


s_mul_hi_i32 s0, s16, s8 // (op 627) 
s_add_i32 s1, s0, s16 // (op 627) 

s_ashr_i32 s0, s1, 31 // (op 627) 
s_cmp_eq_i32 s8, 0 // (op 627) 
s_mov_b32 s6, scc // (op 627) 
s_cmp_eq_i32 s6, 0 // (op 627) 
s_cselect_b32 s7, 0, -1 // (op 627) 

s_and_b32 s6, s15, 31 // (op 627) 
s_lshl_b32 s56, 1, s6 // (op 627) 
s_add_i32 s57, s56, s7 // (op 627) 


s_and_b32 s7, s0, s57 // (op 627) 


s_add_i32 s0, s1, s7 // (op 627) 


s_ashr_i32 s1, s0, s6 // (op 627) 


s_lshl_b32 s0, s15, 24 // (op 627) 
s_ashr_i32 s6, s0, 31 // (op 627) 

s_xor_b32 s0, s1, s6 // (op 627) 

s_sub_i32 s1, s0, s6 // (op 627) 


s_mul_i32 s0, s14, s18 // (op 627) 

s_mul_i32 s6, s50, s0 // (op 627) 

s_mul_i32 s0, s1, s6 // (op 627) 

s_sub_u32 s6, s16, s0 // (op 627) 

s_mul_hi_i32 s0, s6, s13 // (op 627) 
s_add_i32 s7, s0, s6 // (op 627) 


s_ashr_i32 s0, s7, 31 // (op 627) 
s_cmp_eq_i32 s13, 0 // (op 627) 
s_mov_b32 s6, scc // (op 627) 
s_cmp_eq_i32 s6, 0 // (op 627) 
s_cselect_b32 s18, 0, -1 // (op 627) 

s_and_b32 s6, s49, 31 // (op 627) 
s_lshl_b32 s19, 1, s6 // (op 627) 
s_add_i32 s56, s19, s18 // (op 627) 


s_and_b32 s18, s0, s56 // (op 627) 


s_add_i32 s0, s7, s18 // (op 627) 


s_ashr_i32 s7, s0, s6 // (op 627) 


s_lshl_b32 s0, s49, 24 // (op 627) 
s_ashr_i32 s6, s0, 31 // (op 627) 

s_xor_b32 s0, s7, s6 // (op 627) 

s_sub_i32 s7, s0, s6 // (op 627) 


s_cmp_eq_i32 s1, 0 // (op 627) 
s_mov_b32 s0, scc // (op 627) 

s_cmp_eq_i32 s0, 0 // (op 627) 
s_cselect_b32 s1, s7, s51 // (op 627) 



v_lshl_add_u32 v12, s1, 4, v13 // (op 627) 


v_mul_lo_u32 v19, v12, s27 // (op 627) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 627) most significant: high of low * low
v_mul_lo_u32 v18, v12, s26 // (op 627) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 627) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 627) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 627) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 627) 


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 627) 

v_mov_b32 v12, v16 // (op 627) convert

s_mul_i32 s1, s26, s5 // (op 629) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 629) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 629) least significant
s_mul_hi_u32 s6, s26, s4 // (op 629) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 629) most significant: sum
s_add_u32 s1, s1, s6 // (op 629) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 629) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 629) 

s_mov_b32 s56, s28 // (op 629) reg expression
s_mov_b32 s57, s29 // (op 629) reg expression
s_mov_b32 s58, s0 // (op 629) reg expression
s_mov_b32 s59, 131072 // (op 629) call()

s_mul_hi_i32 s51, s16, s9 // (op 646) 
v_lshrrev_b32 v13, 3, v1 // (op 646) 
s_add_i32 s0, s51, s16 // (op 646) 

v_lshlrev_b32 v16, 1, v13 // (op 646) 

s_ashr_i32 s1, s0, 31 // (op 646) 
v_and_b32 v13, -8, v1 // (op 646) 
s_cmp_eq_i32 s9, 0 // (op 646) 
s_mov_b32 s51, scc // (op 646) 
v_sub_u32 v17, v1, v13 // (op 646) 

s_cmp_eq_i32 s51, 0 // (op 646) 
s_cselect_b32 s6, 0, -1 // (op 646) 

v_lshlrev_b32 v13, 5, v17 // (op 646) 

v_mov_b32 v19, 0 // (op 646) convert
v_mov_b32 v18, v13 // (op 646) convert

s_and_b32 s7, s10, 31 // (op 646) 
s_lshl_b32 s51, 1, s7 // (op 646) 
s_add_i32 s18, s51, s6 // (op 646) 


s_and_b32 s6, s1, s18 // (op 646) 


s_add_i32 s1, s0, s6 // (op 646) 


s_ashr_i32 s0, s1, s7 // (op 646) 


s_lshl_b32 s1, s10, 24 // (op 646) 
s_ashr_i32 s51, s1, 31 // (op 646) 

s_xor_b32 s1, s0, s51 // (op 646) 

s_sub_i32 s0, s1, s51 // (op 646) 


s_mul_hi_i32 s1, s0, s11 // (op 646) 
s_add_i32 s51, s1, s0 // (op 646) 

s_ashr_i32 s1, s51, 31 // (op 646) 
s_cmp_eq_i32 s11, 0 // (op 646) 
s_mov_b32 s6, scc // (op 646) 
s_cmp_eq_i32 s6, 0 // (op 646) 
s_cselect_b32 s7, 0, -1 // (op 646) 

s_and_b32 s6, s12, 31 // (op 646) 
s_lshl_b32 s18, 1, s6 // (op 646) 
s_add_i32 s19, s18, s7 // (op 646) 


s_and_b32 s7, s1, s19 // (op 646) 


s_add_i32 s1, s51, s7 // (op 646) 


s_ashr_i32 s7, s1, s6 // (op 646) 


s_lshl_b32 s1, s12, 24 // (op 646) 
s_ashr_i32 s6, s1, 31 // (op 646) 

s_xor_b32 s1, s7, s6 // (op 646) 

s_sub_i32 s7, s1, s6 // (op 646) 


s_add_u32 s18, s4, 16 // (op 646) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 646) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 646) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 646) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 646) 

s_mul_i32 s1, s7, s18 // (op 646) 

s_sub_i32 s51, s0, s1 // (op 646) 


s_mul_hi_i32 s0, s16, s8 // (op 646) 
s_add_i32 s1, s0, s16 // (op 646) 

s_ashr_i32 s0, s1, 31 // (op 646) 
s_cmp_eq_i32 s8, 0 // (op 646) 
s_mov_b32 s6, scc // (op 646) 
s_cmp_eq_i32 s6, 0 // (op 646) 
s_cselect_b32 s7, 0, -1 // (op 646) 

s_and_b32 s6, s15, 31 // (op 646) 
s_lshl_b32 s60, 1, s6 // (op 646) 
s_add_i32 s61, s60, s7 // (op 646) 


s_and_b32 s7, s0, s61 // (op 646) 


s_add_i32 s0, s1, s7 // (op 646) 


s_ashr_i32 s1, s0, s6 // (op 646) 


s_lshl_b32 s0, s15, 24 // (op 646) 
s_ashr_i32 s6, s0, 31 // (op 646) 

s_xor_b32 s0, s1, s6 // (op 646) 

s_sub_i32 s1, s0, s6 // (op 646) 


s_mul_i32 s0, s14, s18 // (op 646) 

s_mul_i32 s6, s50, s0 // (op 646) 

s_mul_i32 s0, s1, s6 // (op 646) 

s_sub_u32 s6, s16, s0 // (op 646) 

s_mul_hi_i32 s0, s6, s13 // (op 646) 
s_add_i32 s7, s0, s6 // (op 646) 


s_ashr_i32 s0, s7, 31 // (op 646) 
s_cmp_eq_i32 s13, 0 // (op 646) 
s_mov_b32 s6, scc // (op 646) 
s_cmp_eq_i32 s6, 0 // (op 646) 
s_cselect_b32 s18, 0, -1 // (op 646) 

s_and_b32 s6, s49, 31 // (op 646) 
s_lshl_b32 s19, 1, s6 // (op 646) 
s_add_i32 s60, s19, s18 // (op 646) 


s_and_b32 s18, s0, s60 // (op 646) 


s_add_i32 s0, s7, s18 // (op 646) 


s_ashr_i32 s7, s0, s6 // (op 646) 


s_lshl_b32 s0, s49, 24 // (op 646) 
s_ashr_i32 s6, s0, 31 // (op 646) 

s_xor_b32 s0, s7, s6 // (op 646) 

s_sub_i32 s7, s0, s6 // (op 646) 


s_cmp_eq_i32 s1, 0 // (op 646) 
s_mov_b32 s0, scc // (op 646) 

s_cmp_eq_i32 s0, 0 // (op 646) 
s_cselect_b32 s1, s7, s51 // (op 646) 



v_lshl_add_u32 v13, s1, 4, v16 // (op 646) 


v_mul_lo_u32 v17, v13, s27 // (op 646) most significant: low of low * high
v_mul_hi_u32 v20, v13, s26 // (op 646) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 646) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 646) most significant: sum


v_add_co_u32 v20, s[0:1], v18, v16 // (op 646) least significant half
v_addc_co_u32 v21, s[0:1], v19, v17, s[0:1] // (op 646) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 646) 


v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 646) 

v_mov_b32 v13, v16 // (op 646) convert

s_mul_i32 s1, s26, s5 // (op 648) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 648) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 648) least significant
s_mul_hi_u32 s6, s26, s4 // (op 648) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 648) most significant: sum
s_add_u32 s1, s1, s6 // (op 648) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 648) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 648) 

s_mov_b32 s56, s28 // (op 648) reg expression
s_mov_b32 s57, s29 // (op 648) reg expression
s_mov_b32 s58, s0 // (op 648) reg expression
s_mov_b32 s59, 131072 // (op 648) call()

s_mul_hi_i32 s51, s16, s9 // (op 691) 
v_and_b32 v16, -16, v1 // (op 691) 
s_add_i32 s0, s51, s16 // (op 691) 

v_sub_u32 v17, v1, v16 // (op 691) 

s_ashr_i32 s1, s0, 31 // (op 691) 
v_lshrrev_b32 v16, 4, v1 // (op 691) 
s_cmp_eq_i32 s9, 0 // (op 691) 
s_mov_b32 s51, scc // (op 691) 
v_lshlrev_b32 v18, 1, v16 // (op 691) 

s_cmp_eq_i32 s51, 0 // (op 691) 
s_cselect_b32 s6, 0, -1 // (op 691) 

s_and_b32 s7, s10, 31 // (op 691) 
s_lshl_b32 s51, 1, s7 // (op 691) 
s_add_i32 s18, s51, s6 // (op 691) 


s_and_b32 s6, s1, s18 // (op 691) 


s_add_i32 s1, s0, s6 // (op 691) 


s_ashr_i32 s0, s1, s7 // (op 691) 


s_lshl_b32 s1, s10, 24 // (op 691) 
s_ashr_i32 s51, s1, 31 // (op 691) 

s_xor_b32 s1, s0, s51 // (op 691) 

s_sub_i32 s0, s1, s51 // (op 691) 


s_mul_hi_i32 s1, s0, s11 // (op 691) 
s_add_i32 s51, s1, s0 // (op 691) 

s_ashr_i32 s1, s51, 31 // (op 691) 
s_cmp_eq_i32 s11, 0 // (op 691) 
s_mov_b32 s6, scc // (op 691) 
s_cmp_eq_i32 s6, 0 // (op 691) 
s_cselect_b32 s7, 0, -1 // (op 691) 

s_and_b32 s6, s12, 31 // (op 691) 
s_lshl_b32 s18, 1, s6 // (op 691) 
s_add_i32 s19, s18, s7 // (op 691) 


s_and_b32 s7, s1, s19 // (op 691) 


s_add_i32 s1, s51, s7 // (op 691) 


s_ashr_i32 s7, s1, s6 // (op 691) 


s_lshl_b32 s1, s12, 24 // (op 691) 
s_ashr_i32 s6, s1, 31 // (op 691) 

s_xor_b32 s1, s7, s6 // (op 691) 

s_sub_i32 s7, s1, s6 // (op 691) 


s_mul_i32 s1, s0, s14 // (op 691) 

s_sub_u32 s0, s16, s1 // (op 691) 

s_mul_i32 s1, s7, s14 // (op 691) 
s_add_u32 s1, s1, s0 // (op 691) 


s_mul_hi_i32 s0, s16, s8 // (op 691) 
s_add_i32 s51, s0, s16 // (op 691) 

s_ashr_i32 s0, s51, 31 // (op 691) 
s_cmp_eq_i32 s8, 0 // (op 691) 
s_mov_b32 s6, scc // (op 691) 
s_cmp_eq_i32 s6, 0 // (op 691) 
s_cselect_b32 s7, 0, -1 // (op 691) 

s_and_b32 s6, s15, 31 // (op 691) 
s_lshl_b32 s18, 1, s6 // (op 691) 
s_add_i32 s19, s18, s7 // (op 691) 


s_and_b32 s7, s0, s19 // (op 691) 


s_add_i32 s0, s51, s7 // (op 691) 


s_ashr_i32 s7, s0, s6 // (op 691) 


s_lshl_b32 s0, s15, 24 // (op 691) 
s_ashr_i32 s6, s0, 31 // (op 691) 

s_xor_b32 s0, s7, s6 // (op 691) 

s_sub_i32 s7, s0, s6 // (op 691) 


s_add_u32 s18, s4, 16 // (op 691) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 691) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 691) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 691) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 691) 

s_mul_i32 s0, s14, s18 // (op 691) 

s_mul_i32 s6, s50, s0 // (op 691) 

s_mul_i32 s0, s7, s6 // (op 691) 

s_sub_u32 s6, s16, s0 // (op 691) 

s_mov_b32 s0, s6 // (op 691) 
s_mul_hi_i32 s51, s0, s13 // (op 691) 
s_add_i32 s18, s51, s0 // (op 691) 


s_ashr_i32 s0, s18, 31 // (op 691) 
s_cmp_eq_i32 s13, 0 // (op 691) 
s_mov_b32 s19, scc // (op 691) 
s_cmp_eq_i32 s19, 0 // (op 691) 
s_cselect_b32 s51, 0, -1 // (op 691) 

s_and_b32 s19, s49, 31 // (op 691) 
s_lshl_b32 s60, 1, s19 // (op 691) 
s_add_i32 s61, s60, s51 // (op 691) 


s_and_b32 s51, s0, s61 // (op 691) 


s_add_i32 s0, s18, s51 // (op 691) 


s_ashr_i32 s18, s0, s19 // (op 691) 


s_lshl_b32 s0, s49, 24 // (op 691) 
s_ashr_i32 s19, s0, 31 // (op 691) 

s_xor_b32 s0, s18, s19 // (op 691) 

s_sub_i32 s18, s0, s19 // (op 691) 


s_mul_i32 s0, s48, s14 // (op 691) 
s_add_u32 s60, s20, 16 // (op 691) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 691) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 691) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 691) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 691) 

s_sub_i32 s19, s60, s0 // (op 691) 

s_mul_i32 s51, s18, s19 // (op 691) 


s_sub_u32 s18, s6, s51 // (op 691) 


s_add_u32 s6, s18, s0 // (op 691) 


s_cmp_eq_i32 s7, 0 // (op 691) 
s_mov_b32 s0, scc // (op 691) 

s_cmp_eq_i32 s0, 0 // (op 691) 
s_cselect_b32 s7, s6, s1 // (op 691) 



v_lshl_add_u32 v16, s7, 4, v17 // (op 691) 


v_mul_lo_u32 v17, v18, s30 // (op 691) 

v_add_u32 v18, v16, v17 // (op 691) 


s_mul_i32 s1, s30, s35 // (op 693) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 693) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 693) least significant
s_mul_hi_u32 s6, s30, s34 // (op 693) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 693) most significant: sum
s_add_u32 s1, s1, s6 // (op 693) most significant: sum


s_mov_b32 s60, s32 // (op 693) reg expression
s_mov_b32 s61, s33 // (op 693) reg expression
s_mov_b32 s62, s0 // (op 693) reg expression
s_mov_b32 s63, 131072 // (op 693) call()

s_mul_hi_i32 s51, s16, s9 // (op 710) 
v_and_b32 v16, -16, v1 // (op 710) 
s_add_i32 s0, s51, s16 // (op 710) 

v_sub_u32 v17, v1, v16 // (op 710) 

s_ashr_i32 s1, s0, 31 // (op 710) 
v_lshrrev_b32 v16, 4, v1 // (op 710) 
s_cmp_eq_i32 s9, 0 // (op 710) 
s_mov_b32 s51, scc // (op 710) 
v_lshlrev_b32 v19, 1, v16 // (op 710) 

s_cmp_eq_i32 s51, 0 // (op 710) 
s_cselect_b32 s6, 0, -1 // (op 710) 

s_and_b32 s7, s10, 31 // (op 710) 
s_lshl_b32 s51, 1, s7 // (op 710) 
s_add_i32 s18, s51, s6 // (op 710) 


s_and_b32 s6, s1, s18 // (op 710) 


s_add_i32 s1, s0, s6 // (op 710) 


s_ashr_i32 s0, s1, s7 // (op 710) 


s_lshl_b32 s1, s10, 24 // (op 710) 
s_ashr_i32 s51, s1, 31 // (op 710) 

s_xor_b32 s1, s0, s51 // (op 710) 

s_sub_i32 s0, s1, s51 // (op 710) 


s_mul_hi_i32 s1, s0, s11 // (op 710) 
s_add_i32 s51, s1, s0 // (op 710) 

s_ashr_i32 s1, s51, 31 // (op 710) 
s_cmp_eq_i32 s11, 0 // (op 710) 
s_mov_b32 s6, scc // (op 710) 
s_cmp_eq_i32 s6, 0 // (op 710) 
s_cselect_b32 s7, 0, -1 // (op 710) 

s_and_b32 s6, s12, 31 // (op 710) 
s_lshl_b32 s18, 1, s6 // (op 710) 
s_add_i32 s19, s18, s7 // (op 710) 


s_and_b32 s7, s1, s19 // (op 710) 


s_add_i32 s1, s51, s7 // (op 710) 


s_ashr_i32 s7, s1, s6 // (op 710) 


s_lshl_b32 s1, s12, 24 // (op 710) 
s_ashr_i32 s6, s1, 31 // (op 710) 

s_xor_b32 s1, s7, s6 // (op 710) 

s_sub_i32 s7, s1, s6 // (op 710) 


s_mul_i32 s1, s0, s14 // (op 710) 

s_sub_u32 s0, s16, s1 // (op 710) 

s_mul_i32 s1, s7, s14 // (op 710) 
s_add_u32 s1, s1, s0 // (op 710) 


s_mul_hi_i32 s0, s16, s8 // (op 710) 
s_add_i32 s51, s0, s16 // (op 710) 

s_ashr_i32 s0, s51, 31 // (op 710) 
s_cmp_eq_i32 s8, 0 // (op 710) 
s_mov_b32 s6, scc // (op 710) 
s_cmp_eq_i32 s6, 0 // (op 710) 
s_cselect_b32 s7, 0, -1 // (op 710) 

s_and_b32 s6, s15, 31 // (op 710) 
s_lshl_b32 s18, 1, s6 // (op 710) 
s_add_i32 s19, s18, s7 // (op 710) 


s_and_b32 s7, s0, s19 // (op 710) 


s_add_i32 s0, s51, s7 // (op 710) 


s_ashr_i32 s7, s0, s6 // (op 710) 


s_lshl_b32 s0, s15, 24 // (op 710) 
s_ashr_i32 s6, s0, 31 // (op 710) 

s_xor_b32 s0, s7, s6 // (op 710) 

s_sub_i32 s7, s0, s6 // (op 710) 


s_add_u32 s18, s4, 16 // (op 710) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 710) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 710) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 710) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 710) 

s_mul_i32 s0, s14, s18 // (op 710) 

s_mul_i32 s6, s50, s0 // (op 710) 

s_mul_i32 s0, s7, s6 // (op 710) 

s_sub_u32 s6, s16, s0 // (op 710) 

s_mov_b32 s0, s6 // (op 710) 
s_mul_hi_i32 s51, s0, s13 // (op 710) 
s_add_i32 s18, s51, s0 // (op 710) 


s_ashr_i32 s0, s18, 31 // (op 710) 
s_cmp_eq_i32 s13, 0 // (op 710) 
s_mov_b32 s19, scc // (op 710) 
s_cmp_eq_i32 s19, 0 // (op 710) 
s_cselect_b32 s51, 0, -1 // (op 710) 

s_and_b32 s19, s49, 31 // (op 710) 
s_lshl_b32 s64, 1, s19 // (op 710) 
s_add_i32 s65, s64, s51 // (op 710) 


s_and_b32 s51, s0, s65 // (op 710) 


s_add_i32 s0, s18, s51 // (op 710) 


s_ashr_i32 s18, s0, s19 // (op 710) 


s_lshl_b32 s0, s49, 24 // (op 710) 
s_ashr_i32 s19, s0, 31 // (op 710) 

s_xor_b32 s0, s18, s19 // (op 710) 

s_sub_i32 s18, s0, s19 // (op 710) 


s_mul_i32 s0, s48, s14 // (op 710) 
s_add_u32 s64, s20, 16 // (op 710) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 710) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 710) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 710) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 710) 

s_sub_i32 s19, s64, s0 // (op 710) 

s_mul_i32 s51, s18, s19 // (op 710) 


s_sub_u32 s18, s6, s51 // (op 710) 


s_add_u32 s6, s18, s0 // (op 710) 


s_cmp_eq_i32 s7, 0 // (op 710) 
s_mov_b32 s0, scc // (op 710) 

s_cmp_eq_i32 s0, 0 // (op 710) 
s_cselect_b32 s7, s6, s1 // (op 710) 



v_lshl_add_u32 v16, s7, 4, v17 // (op 710) 


v_mul_lo_u32 v17, v19, s30 // (op 710) 

v_add_u32 v19, v16, v17 // (op 710) 


s_mul_i32 s1, s30, s35 // (op 712) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 712) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 712) least significant
s_mul_hi_u32 s6, s30, s34 // (op 712) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 712) most significant: sum
s_add_u32 s1, s1, s6 // (op 712) most significant: sum


s_mov_b32 s60, s32 // (op 712) reg expression
s_mov_b32 s61, s33 // (op 712) reg expression
s_mov_b32 s62, s0 // (op 712) reg expression
s_mov_b32 s63, 131072 // (op 712) call()

s_mul_hi_i32 s51, s16, s9 // (op 755) 
v_lshrrev_b32 v16, 3, v1 // (op 755) 
s_add_i32 s0, s51, s16 // (op 755) 

v_lshlrev_b32 v17, 1, v16 // (op 755) 

s_ashr_i32 s1, s0, 31 // (op 755) 
v_and_b32 v16, -8, v1 // (op 755) 
s_cmp_eq_i32 s9, 0 // (op 755) 
s_mov_b32 s51, scc // (op 755) 
v_sub_u32 v20, v1, v16 // (op 755) 

s_cmp_eq_i32 s51, 0 // (op 755) 
s_cselect_b32 s6, 0, -1 // (op 755) 

s_and_b32 s7, s10, 31 // (op 755) 
s_lshl_b32 s51, 1, s7 // (op 755) 
s_add_i32 s18, s51, s6 // (op 755) 


s_and_b32 s6, s1, s18 // (op 755) 


s_add_i32 s1, s0, s6 // (op 755) 


s_ashr_i32 s0, s1, s7 // (op 755) 


s_lshl_b32 s1, s10, 24 // (op 755) 
s_ashr_i32 s51, s1, 31 // (op 755) 

s_xor_b32 s1, s0, s51 // (op 755) 

s_sub_i32 s0, s1, s51 // (op 755) 


s_mul_hi_i32 s1, s0, s11 // (op 755) 
s_add_i32 s51, s1, s0 // (op 755) 

s_ashr_i32 s1, s51, 31 // (op 755) 
s_cmp_eq_i32 s11, 0 // (op 755) 
s_mov_b32 s6, scc // (op 755) 
s_cmp_eq_i32 s6, 0 // (op 755) 
s_cselect_b32 s7, 0, -1 // (op 755) 

s_and_b32 s6, s12, 31 // (op 755) 
s_lshl_b32 s18, 1, s6 // (op 755) 
s_add_i32 s19, s18, s7 // (op 755) 


s_and_b32 s7, s1, s19 // (op 755) 


s_add_i32 s1, s51, s7 // (op 755) 


s_ashr_i32 s7, s1, s6 // (op 755) 


s_lshl_b32 s1, s12, 24 // (op 755) 
s_ashr_i32 s6, s1, 31 // (op 755) 

s_xor_b32 s1, s7, s6 // (op 755) 

s_sub_i32 s7, s1, s6 // (op 755) 


s_add_u32 s18, s4, 16 // (op 755) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 755) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 755) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 755) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 755) 

s_mul_i32 s1, s7, s18 // (op 755) 

s_sub_i32 s51, s0, s1 // (op 755) 


s_mul_hi_i32 s0, s16, s8 // (op 755) 
s_add_i32 s1, s0, s16 // (op 755) 

s_ashr_i32 s0, s1, 31 // (op 755) 
s_cmp_eq_i32 s8, 0 // (op 755) 
s_mov_b32 s6, scc // (op 755) 
s_cmp_eq_i32 s6, 0 // (op 755) 
s_cselect_b32 s7, 0, -1 // (op 755) 

s_and_b32 s6, s15, 31 // (op 755) 
s_lshl_b32 s64, 1, s6 // (op 755) 
s_add_i32 s65, s64, s7 // (op 755) 


s_and_b32 s7, s0, s65 // (op 755) 


s_add_i32 s0, s1, s7 // (op 755) 


s_ashr_i32 s1, s0, s6 // (op 755) 


s_lshl_b32 s0, s15, 24 // (op 755) 
s_ashr_i32 s6, s0, 31 // (op 755) 

s_xor_b32 s0, s1, s6 // (op 755) 

s_sub_i32 s1, s0, s6 // (op 755) 


s_mul_i32 s0, s14, s18 // (op 755) 

s_mul_i32 s6, s50, s0 // (op 755) 

s_mul_i32 s0, s1, s6 // (op 755) 

s_sub_u32 s6, s16, s0 // (op 755) 

s_mul_hi_i32 s0, s6, s13 // (op 755) 
s_add_i32 s7, s0, s6 // (op 755) 


s_ashr_i32 s0, s7, 31 // (op 755) 
s_cmp_eq_i32 s13, 0 // (op 755) 
s_mov_b32 s6, scc // (op 755) 
s_cmp_eq_i32 s6, 0 // (op 755) 
s_cselect_b32 s18, 0, -1 // (op 755) 

s_and_b32 s6, s49, 31 // (op 755) 
s_lshl_b32 s19, 1, s6 // (op 755) 
s_add_i32 s64, s19, s18 // (op 755) 


s_and_b32 s18, s0, s64 // (op 755) 


s_add_i32 s0, s7, s18 // (op 755) 


s_ashr_i32 s7, s0, s6 // (op 755) 


s_lshl_b32 s0, s49, 24 // (op 755) 
s_ashr_i32 s6, s0, 31 // (op 755) 

s_xor_b32 s0, s7, s6 // (op 755) 

s_sub_i32 s7, s0, s6 // (op 755) 


s_cmp_eq_i32 s1, 0 // (op 755) 
s_mov_b32 s0, scc // (op 755) 

s_cmp_eq_i32 s0, 0 // (op 755) 
s_cselect_b32 s1, s7, s51 // (op 755) 



v_lshl_add_u32 v16, s1, 4, v17 // (op 755) 


v_mul_lo_u32 v17, v16, s36 // (op 755) 

v_add_u32 v16, v20, v17 // (op 755) 


s_mul_i32 s1, s36, s5 // (op 757) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 757) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 757) least significant
s_mul_hi_u32 s6, s36, s4 // (op 757) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 757) most significant: sum
s_add_u32 s1, s1, s6 // (op 757) most significant: sum


s_mov_b32 s64, s38 // (op 757) reg expression
s_mov_b32 s65, s39 // (op 757) reg expression
s_mov_b32 s66, s0 // (op 757) reg expression
s_mov_b32 s67, 131072 // (op 757) call()

s_mul_hi_i32 s51, s16, s9 // (op 774) 
v_lshrrev_b32 v17, 3, v1 // (op 774) 
s_add_i32 s0, s51, s16 // (op 774) 

v_lshlrev_b32 v20, 1, v17 // (op 774) 

s_ashr_i32 s1, s0, 31 // (op 774) 
v_and_b32 v17, -8, v1 // (op 774) 
s_cmp_eq_i32 s9, 0 // (op 774) 
s_mov_b32 s51, scc // (op 774) 
v_sub_u32 v21, v1, v17 // (op 774) 

s_cmp_eq_i32 s51, 0 // (op 774) 
s_cselect_b32 s6, 0, -1 // (op 774) 

s_and_b32 s7, s10, 31 // (op 774) 
s_lshl_b32 s51, 1, s7 // (op 774) 
s_add_i32 s18, s51, s6 // (op 774) 


s_and_b32 s6, s1, s18 // (op 774) 


s_add_i32 s1, s0, s6 // (op 774) 


s_ashr_i32 s0, s1, s7 // (op 774) 


s_lshl_b32 s1, s10, 24 // (op 774) 
s_ashr_i32 s51, s1, 31 // (op 774) 

s_xor_b32 s1, s0, s51 // (op 774) 

s_sub_i32 s0, s1, s51 // (op 774) 


s_mul_hi_i32 s1, s0, s11 // (op 774) 
s_add_i32 s51, s1, s0 // (op 774) 

s_ashr_i32 s1, s51, 31 // (op 774) 
s_cmp_eq_i32 s11, 0 // (op 774) 
s_mov_b32 s6, scc // (op 774) 
s_cmp_eq_i32 s6, 0 // (op 774) 
s_cselect_b32 s7, 0, -1 // (op 774) 

s_and_b32 s6, s12, 31 // (op 774) 
s_lshl_b32 s18, 1, s6 // (op 774) 
s_add_i32 s19, s18, s7 // (op 774) 


s_and_b32 s7, s1, s19 // (op 774) 


s_add_i32 s1, s51, s7 // (op 774) 


s_ashr_i32 s7, s1, s6 // (op 774) 


s_lshl_b32 s1, s12, 24 // (op 774) 
s_ashr_i32 s6, s1, 31 // (op 774) 

s_xor_b32 s1, s7, s6 // (op 774) 

s_sub_i32 s7, s1, s6 // (op 774) 


s_add_u32 s18, s4, 16 // (op 774) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 774) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 774) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 774) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 774) 

s_mul_i32 s1, s7, s18 // (op 774) 

s_sub_i32 s51, s0, s1 // (op 774) 


s_mul_hi_i32 s0, s16, s8 // (op 774) 
s_add_i32 s1, s0, s16 // (op 774) 

s_ashr_i32 s0, s1, 31 // (op 774) 
s_cmp_eq_i32 s8, 0 // (op 774) 
s_mov_b32 s6, scc // (op 774) 
s_cmp_eq_i32 s6, 0 // (op 774) 
s_cselect_b32 s7, 0, -1 // (op 774) 

s_and_b32 s6, s15, 31 // (op 774) 
s_lshl_b32 s68, 1, s6 // (op 774) 
s_add_i32 s69, s68, s7 // (op 774) 


s_and_b32 s7, s0, s69 // (op 774) 


s_add_i32 s0, s1, s7 // (op 774) 


s_ashr_i32 s1, s0, s6 // (op 774) 


s_lshl_b32 s0, s15, 24 // (op 774) 
s_ashr_i32 s6, s0, 31 // (op 774) 

s_xor_b32 s0, s1, s6 // (op 774) 

s_sub_i32 s1, s0, s6 // (op 774) 


s_mul_i32 s0, s14, s18 // (op 774) 

s_mul_i32 s6, s50, s0 // (op 774) 

s_mul_i32 s0, s1, s6 // (op 774) 

s_sub_u32 s6, s16, s0 // (op 774) 

s_mul_hi_i32 s0, s6, s13 // (op 774) 
s_add_i32 s7, s0, s6 // (op 774) 


s_ashr_i32 s0, s7, 31 // (op 774) 
s_cmp_eq_i32 s13, 0 // (op 774) 
s_mov_b32 s6, scc // (op 774) 
s_cmp_eq_i32 s6, 0 // (op 774) 
s_cselect_b32 s18, 0, -1 // (op 774) 

s_and_b32 s6, s49, 31 // (op 774) 
s_lshl_b32 s19, 1, s6 // (op 774) 
s_add_i32 s68, s19, s18 // (op 774) 


s_and_b32 s18, s0, s68 // (op 774) 


s_add_i32 s0, s7, s18 // (op 774) 


s_ashr_i32 s7, s0, s6 // (op 774) 


s_lshl_b32 s0, s49, 24 // (op 774) 
s_ashr_i32 s6, s0, 31 // (op 774) 

s_xor_b32 s0, s7, s6 // (op 774) 

s_sub_i32 s7, s0, s6 // (op 774) 


s_cmp_eq_i32 s1, 0 // (op 774) 
s_mov_b32 s0, scc // (op 774) 

s_cmp_eq_i32 s0, 0 // (op 774) 
s_cselect_b32 s1, s7, s51 // (op 774) 



v_lshl_add_u32 v17, s1, 4, v20 // (op 774) 


v_mul_lo_u32 v20, v17, s36 // (op 774) 

v_add_u32 v17, v21, v20 // (op 774) 


s_mul_i32 s1, s36, s5 // (op 776) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 776) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 776) least significant
s_mul_hi_u32 s6, s36, s4 // (op 776) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 776) most significant: sum
s_add_u32 s1, s1, s6 // (op 776) most significant: sum


s_mov_b32 s64, s38 // (op 776) reg expression
s_mov_b32 s65, s39 // (op 776) reg expression
s_mov_b32 s66, s0 // (op 776) reg expression
s_mov_b32 s67, 131072 // (op 776) call()


s_lshl_b64 s[0:1], s[26:27], 2 // (op 428) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 428) 


v_mov_b32 v22, v12 // (op 428) 
buffer_load_dwordx4 v[24:27], v22, s[56:59], 0 offen // (op 428) Load value
v_add_u32 v22, v22, v20 // (op 428) 
buffer_load_dwordx4 v[28:31], v22, s[56:59], 0 offen // (op 428) Load value



v_mov_b64 v[20:21], s[36:37] // (op 431) convert

v_mov_b32 v22, v16 // (op 431) 
buffer_load_ubyte v32, v22, s[64:67], 0 offen offset: 0 // (op 431) Load value
v_add_u32 v22, v22, v20 // (op 431) 
buffer_load_ubyte v33, v22, s[64:67], 0 offen offset: 0 // (op 431) Load value



s_lshl_b64 s[0:1], s[2:3], 2 // (op 434) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 434) 


v_mov_b32 v22, v14 // (op 434) reg expression
buffer_load_dwordx2 v[34:35], v22, s[52:55], 0 offen // (op 434) Load value
v_add_u32 v22, v22, v20 // (op 434) 
buffer_load_dwordx2 v[36:37], v22, s[52:55], 0 offen // (op 434) Load value
v_add_u32 v22, v22, v20 // (op 434) 
buffer_load_dwordx2 v[38:39], v22, s[52:55], 0 offen // (op 434) Load value
v_add_u32 v22, v22, v20 // (op 434) 
buffer_load_dwordx2 v[40:41], v22, s[52:55], 0 offen // (op 434) Load value



v_mov_b64 v[20:21], s[30:31] // (op 437) convert

v_mov_b32 v22, v18 // (op 437) reg expression
buffer_load_ubyte v42, v22, s[60:63], 0 offen offset: 0 // (op 437) Load value
v_add_u32 v22, v22, v20 // (op 437) 
buffer_load_ubyte v43, v22, s[60:63], 0 offen offset: 0 // (op 437) Load value



s_lshl_b64 s[0:1], s[26:27], 2 // (op 440) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 440) 


v_mov_b32 v22, v12 // (op 440) 
buffer_load_dwordx4 v[44:47], v22, s[56:59], 0 offen offset:128 // (op 440) Load value
v_add_u32 v22, v22, v20 // (op 440) 
buffer_load_dwordx4 v[48:51], v22, s[56:59], 0 offen offset:128 // (op 440) Load value




v_mov_b64 v[20:21], s[36:37] // (op 443) convert

v_mov_b32 v12, v16 // (op 443) 
buffer_load_ubyte v22, v12, s[64:67], 0 offen offset:8 // (op 443) Load value
v_add_u32 v12, v12, v20 // (op 443) 
buffer_load_ubyte v23, v12, s[64:67], 0 offen offset:8 // (op 443) Load value




s_lshl_b64 s[0:1], s[2:3], 2 // (op 446) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 446) 


s_lshl_b64 s[0:1], s[2:3], 8 // (op 446) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 446) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 446) 

v_add_u32 v12, v14, s0 // (op 446) 

buffer_load_dwordx2 v[52:53], v12, s[52:55], 0 offen // (op 446) Load value
v_add_u32 v12, v12, v20 // (op 446) 
buffer_load_dwordx2 v[54:55], v12, s[52:55], 0 offen // (op 446) Load value
v_add_u32 v12, v12, v20 // (op 446) 
buffer_load_dwordx2 v[56:57], v12, s[52:55], 0 offen // (op 446) Load value
v_add_u32 v12, v12, v20 // (op 446) 
buffer_load_dwordx2 v[58:59], v12, s[52:55], 0 offen // (op 446) Load value




v_mov_b64 v[20:21], s[30:31] // (op 449) convert

s_lshl_b32 s51, s30, 3 // (op 449) 
v_add_u32 v12, v18, s51 // (op 449) 

buffer_load_ubyte v60, v12, s[60:63], 0 offen offset: 0 // (op 449) Load value
v_add_u32 v12, v12, v20 // (op 449) 
buffer_load_ubyte v61, v12, s[60:63], 0 offen offset: 0 // (op 449) Load value



v_and_b32 v12, -8, v1 // (op 894) 
v_sub_u32 v14, v1, v12 // (op 894) 

v_lshlrev_b32 v12, 5, v14 // (op 894) 

v_lshrrev_b32 v14, 3, v1 // (op 894) 
v_lshl_add_u32 v16, v14, 9, v12 // (op 894) 


v_lshlrev_b32 v12, 2, v16 // (op 894) 

v_lshrrev_b32 v14, 3, v12 // (op 894) 




s_waitcnt vmcnt(19)
ds_write_b128 v14, v[24:27] // (op 451) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v14, v[28:31] offset:128 // (op 451) Store local data 

v_and_b32 v12, -8, v1 // (op 1000) 
v_sub_u32 v14, v1, v12 // (op 1000) 

v_lshrrev_b32 v12, 3, v1 // (op 1000) 
v_lshl_add_u32 v16, v12, 4, v14 // (op 1000) 





s_waitcnt vmcnt(17)
ds_write_b8 v16, v32 offset:2048 // (op 452) Store local data 
s_waitcnt vmcnt(16)
ds_write_b8 v16, v33 offset:2056 // (op 452) Store local data 

v_lshlrev_b32 v12, 8, v1 // (op 841) 
v_lshrrev_b32 v14, 3, v12 // (op 841) 




s_waitcnt vmcnt(15)
ds_write_b64 v14, v[34:35] offset:2176 // (op 453) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v14, v[36:37] offset:2184 // (op 453) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v14, v[38:39] offset:2192 // (op 453) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v14, v[40:41] offset:2200 // (op 453) Store local data 

v_and_b32 v12, -16, v1 // (op 947) 
v_sub_u32 v14, v1, v12 // (op 947) 

v_lshrrev_b32 v12, 4, v1 // (op 947) 
v_lshl_add_u32 v16, v12, 5, v14 // (op 947) 





s_waitcnt vmcnt(11)
ds_write_b8 v16, v42 offset:4224 // (op 454) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v16, v43 offset:4240 // (op 454) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 1198) 



ds_read_b128 v[62:65], v4 // (op 471) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1239) 



ds_read_u8 v12, v6 offset:2048 // (op 477) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1179) 



ds_read_b64_tr_b4 v[66:67], v3 offset:2176 // (op 468) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v3 offset:2304 // (op 468) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1219) 


ds_read_u8 v14, v5 offset:4224 // (op 474) Load local data 
v_lshlrev_b32 v16, 8, v1 // (op 812) 
v_lshrrev_b32 v18, 3, v16 // (op 812) 

v_and_b32 v16, -8, v1 // (op 865) 
v_sub_u32 v20, v1, v16 // (op 865) 

v_lshlrev_b32 v16, 5, v20 // (op 865) 

v_lshrrev_b32 v20, 3, v1 // (op 865) 
v_lshl_add_u32 v21, v20, 9, v16 // (op 865) 


v_lshlrev_b32 v16, 2, v21 // (op 865) 

v_lshrrev_b32 v20, 3, v16 // (op 865) 

v_and_b32 v16, -16, v1 // (op 918) 
v_sub_u32 v21, v1, v16 // (op 918) 

v_lshrrev_b32 v16, 4, v1 // (op 918) 
v_lshl_add_u32 v70, v16, 5, v21 // (op 918) 


v_and_b32 v16, -8, v1 // (op 971) 
v_sub_u32 v21, v1, v16 // (op 971) 

v_lshrrev_b32 v16, 3, v1 // (op 971) 
v_lshl_add_u32 v71, v16, 4, v21 // (op 971) 


v_lshlrev_b32 v16, 8, v1 // (op 1058) 
v_lshrrev_b32 v21, 3, v16 // (op 1058) 

v_and_b32 v16, -8, v1 // (op 1082) 
v_sub_u32 v72, v1, v16 // (op 1082) 

v_lshlrev_b32 v16, 5, v72 // (op 1082) 

v_lshrrev_b32 v72, 3, v1 // (op 1082) 
v_lshl_add_u32 v73, v72, 9, v16 // (op 1082) 


v_lshlrev_b32 v16, 2, v73 // (op 1082) 

v_lshrrev_b32 v72, 3, v16 // (op 1082) 

v_and_b32 v16, -16, v1 // (op 1106) 
v_sub_u32 v73, v1, v16 // (op 1106) 

v_lshrrev_b32 v16, 4, v1 // (op 1106) 
v_lshl_add_u32 v74, v16, 5, v73 // (op 1106) 


v_and_b32 v16, -8, v1 // (op 1130) 
v_sub_u32 v73, v1, v16 // (op 1130) 

v_lshrrev_b32 v16, 3, v1 // (op 1130) 
v_lshl_add_u32 v75, v16, 4, v73 // (op 1130) 


s_mov_b32 s51, 0 // (op 61) call()
s_ashr_i64 s[0:1], s[24:25], 8 // (op 60) 
s_mov_b32 s6, s0 // (op 60) convert

s_and_b32 s7, s6, 4294967294 // (op 60) 

s_cmp_lt_u32 s51, s7 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_60 if false)
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_60:

s_add_u32 s51, s51, 2 // (op 62) 
s_lshl_b64 s[0:1], s[26:27], 2 // (op 10) 
v_lshrrev_b64 v[76:77], 3, s[0:1] // (op 10) 


v_mov_b32 v16, v13 // (op 10) 
v_mov_b32 v73, v16 // (op 10) 
buffer_load_dwordx4 v[24:27], v73, s[56:59], 0 offen offset:256 // (op 10) Load value
v_add_u32 v73, v73, v76 // (op 10) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[62:65], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 90) 
buffer_load_dwordx4 v[28:31], v73, s[56:59], 0 offen offset:256 // (op 10) Load value



s_barrier  // (op 1188) 
v_mov_b64 v[76:77], s[36:37] // (op 25) convert




ds_read_b64_tr_b4 v[66:67], v3 offset:3200 // (op 93) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v3 offset:3328 // (op 93) Transpose load local data 
v_mov_b32 v73, v17 // (op 25) 
s_barrier  // (op 1208) 
v_mov_b32 v78, v73 // (op 25) 



ds_read_b128 v[62:65], v4 offset:64 // (op 97) Load local data 
buffer_load_ubyte v32, v78, s[64:67], 0 offen offset:16 // (op 25) Load value
s_barrier  // (op 1229) 
v_add_u32 v78, v78, v76 // (op 25) 


ds_read_u8 v14, v5 offset:4288 // (op 101) Load local data 
buffer_load_ubyte v33, v78, s[64:67], 0 offen offset:16 // (op 25) Load value



s_barrier  // (op 1249) 
s_lshl_b64 s[0:1], s[2:3], 2 // (op 4) 



ds_read_u8 v12, v6 offset:2052 // (op 105) Load local data 
v_lshrrev_b64 v[76:77], 3, s[0:1] // (op 4) 


v_mov_b32 v78, v15 // (op 4) 
s_lshl_b64 s[0:1], s[2:3], 8 // (op 4) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 4) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 4) 

s_lshl_b32 s6, s0, 1 // (op 4) 

v_add_u32 v79, v78, s6 // (op 4) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[62:65], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 107) 
buffer_load_dwordx2 v[34:35], v79, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v79, v79, v76 // (op 4) 
buffer_load_dwordx2 v[36:37], v79, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v79, v79, v76 // (op 4) 
buffer_load_dwordx2 v[38:39], v79, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v79, v79, v76 // (op 4) 
buffer_load_dwordx2 v[40:41], v79, s[52:55], 0 offen // (op 4) Load value



v_mov_b64 v[76:77], s[30:31] // (op 17) convert

v_mov_b32 v79, v19 // (op 17) 
s_lshl_b32 s0, s30, 4 // (op 17) 
v_add_u32 v80, v79, s0 // (op 17) 

buffer_load_ubyte v42, v80, s[60:63], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v80, v80, v76 // (op 17) 
buffer_load_ubyte v43, v80, s[60:63], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1269) 



s_waitcnt vmcnt(19)
ds_write_b128 v72, v[44:47] offset:4352 // (op 306) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v72, v[48:51] offset:4480 // (op 306) Store local data 
s_barrier  // (op 1281) 



s_waitcnt vmcnt(17)
ds_write_b8 v75, v22 offset:6400 // (op 308) Store local data 
s_waitcnt vmcnt(16)
ds_write_b8 v75, v23 offset:6408 // (op 308) Store local data 
s_barrier  // (op 1262) 



s_waitcnt vmcnt(15)
ds_write_b64 v21, v[52:53] offset:6528 // (op 287) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v21, v[54:55] offset:6536 // (op 287) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v21, v[56:57] offset:6544 // (op 287) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v21, v[58:59] offset:6552 // (op 287) Store local data 
s_barrier  // (op 1275) 



s_waitcnt vmcnt(11)
ds_write_b8 v74, v60 offset:8576 // (op 307) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v74, v61 offset:8592 // (op 307) Store local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1265) 



ds_read_b128 v[80:83], v8 offset:4352 // (op 291) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1278) 



ds_read_u8 v76, v10 offset:6400 // (op 295) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1259) 



ds_read_b64_tr_b4 v[84:85], v7 offset:6528 // (op 289) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v7 offset:6656 // (op 289) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1272) 


ds_read_u8 v77, v9 offset:8576 // (op 293) Load local data 

s_lshl_b64 s[0:1], s[26:27], 2 // (op 309) 
v_lshrrev_b64 v[88:89], 3, s[0:1] // (op 309) 


v_mov_b32 v90, v16 // (op 309) 
buffer_load_dwordx4 v[44:47], v90, s[56:59], 0 offen offset:384 // (op 309) Load value
v_add_u32 v90, v90, v88 // (op 309) 
buffer_load_dwordx4 v[48:51], v90, s[56:59], 0 offen offset:384 // (op 309) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:87], v[80:83], a[0:3], v77, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 296) 









v_add_u32 v13, 256, v13 // (op 638) 
v_mov_b64 v[76:77], s[36:37] // (op 311) convert

v_mov_b32 v16, v73 // (op 311) 
ds_read_b64_tr_b4 v[80:81], v7 offset:7552 // (op 298) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v7 offset:7680 // (op 298) Transpose load local data 
buffer_load_ubyte v22, v16, s[64:67], 0 offen offset:24 // (op 311) Load value
v_add_u32 v16, v16, v76 // (op 311) 



s_barrier  // (op 1344) 
ds_read_b128 v[84:87], v8 offset:4416 // (op 300) Load local data 
buffer_load_ubyte v23, v16, s[64:67], 0 offen offset:24 // (op 311) Load value






s_barrier  // (op 1348) 
v_add_u32 v17, 16, v17 // (op 766) 
ds_read_u8 v16, v9 offset:8640 // (op 302) Load local data 
s_lshl_b64 s[0:1], s[2:3], 2 // (op 286) 
v_lshrrev_b64 v[76:77], 3, s[0:1] // (op 286) 





s_barrier  // (op 1352) 
ds_read_u8 v73, v10 offset:6404 // (op 304) Load local data 
s_lshl_b64 s[0:1], s[2:3], 8 // (op 286) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 286) 

s_barrier  // (op 1356) 
s_lshr_b64 s[0:1], s[6:7], 3 // (op 286) 

s_mul_i32 s6, 3, s0 // (op 286) 

v_add_u32 v88, v78, s6 // (op 286) 

buffer_load_dwordx2 v[52:53], v88, s[52:55], 0 offen // (op 286) Load value
v_add_u32 v88, v88, v76 // (op 286) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:83], v[84:87], a[0:3], v16, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 305) 
buffer_load_dwordx2 v[54:55], v88, s[52:55], 0 offen // (op 286) Load value




v_add_u32 v88, v88, v76 // (op 286) 
buffer_load_dwordx2 v[56:57], v88, s[52:55], 0 offen // (op 286) Load value
v_add_u32 v88, v88, v76 // (op 286) 
buffer_load_dwordx2 v[58:59], v88, s[52:55], 0 offen // (op 286) Load value




v_mov_b64 v[76:77], s[30:31] // (op 310) convert

s_lshl_b32 s0, s30, 3 // (op 310) 
s_mul_i32 s1, 3, s0 // (op 310) 

v_add_u32 v16, v79, s1 // (op 310) 

s_lshl_b64 s[0:1], s[2:3], 8 // (op 574) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 574) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 574) 

s_lshl_b32 s6, s0, 1 // (op 574) 

v_add_u32 v15, v15, s6 // (op 574) 

buffer_load_ubyte v60, v16, s[60:63], 0 offen offset: 0 // (op 310) Load value
v_add_u32 v16, v16, v76 // (op 310) 
buffer_load_ubyte v61, v16, s[60:63], 0 offen offset: 0 // (op 310) Load value



s_barrier  // (op 1201) 
s_lshl_b32 s0, s30, 4 // (op 702) 



v_add_u32 v19, v19, s0 // (op 702) 

s_waitcnt vmcnt(19)
ds_write_b128 v20, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v20, v[28:31] offset:128 // (op 13) Store local data 
s_barrier  // (op 1242) 



s_waitcnt vmcnt(17)
ds_write_b8 v71, v32 offset:2048 // (op 30) Store local data 
s_waitcnt vmcnt(16)
ds_write_b8 v71, v33 offset:2056 // (op 30) Store local data 
s_barrier  // (op 1182) 



s_waitcnt vmcnt(15)
ds_write_b64 v18, v[34:35] offset:2176 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v18, v[36:37] offset:2184 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v18, v[38:39] offset:2192 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v18, v[40:41] offset:2200 // (op 6) Store local data 
s_barrier  // (op 1222) 



s_waitcnt vmcnt(11)
ds_write_b8 v70, v42 offset:4224 // (op 21) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v70, v43 offset:4240 // (op 21) Store local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1211) 



ds_read_b128 v[62:65], v4 // (op 77) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1252) 



ds_read_u8 v12, v6 offset:2048 // (op 87) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1191) 



ds_read_b64_tr_b4 v[66:67], v3 offset:2176 // (op 72) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v3 offset:2304 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1232) 


ds_read_u8 v14, v5 offset:4224 // (op 82) Load local data 
s_ashr_i64 s[0:1], s[24:25], 8 // (op 60) 
s_mov_b32 s6, s0 // (op 60) convert

s_and_b32 s7, s6, 4294967294 // (op 60) 

s_cmp_lt_u32 s51, s7 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_60 if true)
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_60:

















s_mul_hi_i32 s51, s16, s9 // (op 537) 
v_lshlrev_b32 v13, 2, v1 // (op 537) 
s_add_i32 s0, s51, s16 // (op 537) 

s_ashr_i32 s1, s0, 31 // (op 537) 
s_cmp_eq_i32 s9, 0 // (op 537) 
s_mov_b32 s51, scc // (op 537) 
s_cmp_eq_i32 s51, 0 // (op 537) 
s_cselect_b32 s6, 0, -1 // (op 537) 

s_and_b32 s7, s10, 31 // (op 537) 
s_lshl_b32 s51, 1, s7 // (op 537) 
s_add_i32 s18, s51, s6 // (op 537) 


s_and_b32 s6, s1, s18 // (op 537) 


s_add_i32 s1, s0, s6 // (op 537) 


s_ashr_i32 s0, s1, s7 // (op 537) 


s_lshl_b32 s1, s10, 24 // (op 537) 
s_ashr_i32 s51, s1, 31 // (op 537) 

s_xor_b32 s1, s0, s51 // (op 537) 

s_sub_i32 s0, s1, s51 // (op 537) 


s_mul_hi_i32 s1, s0, s11 // (op 537) 
s_add_i32 s51, s1, s0 // (op 537) 

s_ashr_i32 s1, s51, 31 // (op 537) 
s_cmp_eq_i32 s11, 0 // (op 537) 
s_mov_b32 s6, scc // (op 537) 
s_cmp_eq_i32 s6, 0 // (op 537) 
s_cselect_b32 s7, 0, -1 // (op 537) 

s_and_b32 s6, s12, 31 // (op 537) 
s_lshl_b32 s18, 1, s6 // (op 537) 
s_add_i32 s19, s18, s7 // (op 537) 


s_and_b32 s7, s1, s19 // (op 537) 


s_add_i32 s1, s51, s7 // (op 537) 


s_ashr_i32 s7, s1, s6 // (op 537) 


s_lshl_b32 s1, s12, 24 // (op 537) 
s_ashr_i32 s6, s1, 31 // (op 537) 

s_xor_b32 s1, s7, s6 // (op 537) 

s_sub_i32 s7, s1, s6 // (op 537) 


s_mul_i32 s1, s0, s14 // (op 537) 

s_sub_u32 s0, s16, s1 // (op 537) 

s_mul_i32 s1, s7, s14 // (op 537) 
s_add_u32 s1, s1, s0 // (op 537) 


s_mul_hi_i32 s0, s16, s8 // (op 537) 
s_add_i32 s51, s0, s16 // (op 537) 

s_ashr_i32 s0, s51, 31 // (op 537) 
s_cmp_eq_i32 s8, 0 // (op 537) 
s_mov_b32 s6, scc // (op 537) 
s_cmp_eq_i32 s6, 0 // (op 537) 
s_cselect_b32 s7, 0, -1 // (op 537) 

s_and_b32 s6, s15, 31 // (op 537) 
s_lshl_b32 s18, 1, s6 // (op 537) 
s_add_i32 s19, s18, s7 // (op 537) 


s_and_b32 s7, s0, s19 // (op 537) 


s_add_i32 s0, s51, s7 // (op 537) 


s_ashr_i32 s7, s0, s6 // (op 537) 


s_lshl_b32 s0, s15, 24 // (op 537) 
s_ashr_i32 s6, s0, 31 // (op 537) 

s_xor_b32 s0, s7, s6 // (op 537) 

s_sub_i32 s7, s0, s6 // (op 537) 


s_add_u32 s18, s4, 16 // (op 537) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 537) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 537) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 537) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 537) 

s_mul_i32 s0, s14, s18 // (op 537) 

s_mul_i32 s6, s50, s0 // (op 537) 

s_mul_i32 s0, s7, s6 // (op 537) 

s_sub_u32 s6, s16, s0 // (op 537) 

s_mov_b32 s0, s6 // (op 537) 
s_mul_hi_i32 s51, s0, s13 // (op 537) 
s_add_i32 s18, s51, s0 // (op 537) 


s_ashr_i32 s0, s18, 31 // (op 537) 
s_cmp_eq_i32 s13, 0 // (op 537) 
s_mov_b32 s19, scc // (op 537) 
s_cmp_eq_i32 s19, 0 // (op 537) 
s_cselect_b32 s51, 0, -1 // (op 537) 

s_and_b32 s19, s49, 31 // (op 537) 
s_lshl_b32 s68, 1, s19 // (op 537) 
s_add_i32 s69, s68, s51 // (op 537) 


s_and_b32 s51, s0, s69 // (op 537) 


s_add_i32 s0, s18, s51 // (op 537) 


s_ashr_i32 s18, s0, s19 // (op 537) 


s_lshl_b32 s0, s49, 24 // (op 537) 
s_ashr_i32 s19, s0, 31 // (op 537) 

s_xor_b32 s0, s18, s19 // (op 537) 

s_sub_i32 s18, s0, s19 // (op 537) 


s_mul_i32 s0, s48, s14 // (op 537) 
s_add_u32 s68, s20, 16 // (op 537) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 537) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 537) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 537) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 537) 

s_sub_i32 s19, s68, s0 // (op 537) 

s_mul_i32 s51, s18, s19 // (op 537) 


s_sub_u32 s18, s6, s51 // (op 537) 


s_add_u32 s6, s18, s0 // (op 537) 


s_cmp_eq_i32 s7, 0 // (op 537) 
s_mov_b32 s0, scc // (op 537) 

s_cmp_eq_i32 s0, 0 // (op 537) 
s_cselect_b32 s7, s6, s1 // (op 537) 



s_lshl_b32 s6, s7, 4 // (op 537) 

s_mov_b32 s1, 0 // (op 537) convert
s_mov_b32 s0, s6 // (op 537) convert

s_ashr_i64 s[6:7], s[24:25], 8 // (op 537) 
s_lshr_b32 s51, s6, 1 // (op 537) 

v_lshl_add_u32 v15, s51, 9, v13 // (op 537) 


v_mul_lo_u32 v17, v15, s3 // (op 537) most significant: low of low * high
v_mul_hi_u32 v13, v15, s2 // (op 537) most significant: high of low * low
v_mul_lo_u32 v16, v15, s2 // (op 537) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 537) most significant: sum


v_mov_b32 v13, s0 // (op 537) 
v_mov_b32 v15, s1 // (op 537) 
v_add_co_u32 v18, s[6:7], v13, v16 // (op 537) least significant half
v_addc_co_u32 v19, s[6:7], v15, v17, s[6:7] // (op 537) most significant half



v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 537) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 537) 

v_mov_b32 v13, v16 // (op 537) convert

s_mul_i32 s1, s2, s25 // (op 539) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 539) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 539) least significant
s_mul_hi_u32 s6, s2, s24 // (op 539) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 539) most significant: sum
s_add_u32 s1, s1, s6 // (op 539) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 539) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 539) 

s_mov_b32 s52, s22 // (op 539) reg expression
s_mov_b32 s53, s23 // (op 539) reg expression
s_mov_b32 s54, s0 // (op 539) reg expression
s_mov_b32 s55, 131072 // (op 539) call()

s_mul_hi_i32 s51, s16, s9 // (op 604) 
v_lshrrev_b32 v15, 3, v1 // (op 604) 
s_add_i32 s0, s51, s16 // (op 604) 

v_lshlrev_b32 v16, 1, v15 // (op 604) 

s_ashr_i32 s1, s0, 31 // (op 604) 
v_and_b32 v15, -8, v1 // (op 604) 
s_cmp_eq_i32 s9, 0 // (op 604) 
s_mov_b32 s51, scc // (op 604) 
v_sub_u32 v17, v1, v15 // (op 604) 

s_cmp_eq_i32 s51, 0 // (op 604) 
s_cselect_b32 s6, 0, -1 // (op 604) 

v_lshlrev_b32 v15, 5, v17 // (op 604) 

s_and_b32 s7, s10, 31 // (op 604) 
s_lshl_b32 s51, 1, s7 // (op 604) 
s_add_i32 s18, s51, s6 // (op 604) 


s_and_b32 s6, s1, s18 // (op 604) 


s_add_i32 s1, s0, s6 // (op 604) 


s_ashr_i32 s0, s1, s7 // (op 604) 


s_lshl_b32 s1, s10, 24 // (op 604) 
s_ashr_i32 s51, s1, 31 // (op 604) 

s_xor_b32 s1, s0, s51 // (op 604) 

s_sub_i32 s0, s1, s51 // (op 604) 


s_mul_hi_i32 s1, s0, s11 // (op 604) 
s_add_i32 s51, s1, s0 // (op 604) 

s_ashr_i32 s1, s51, 31 // (op 604) 
s_cmp_eq_i32 s11, 0 // (op 604) 
s_mov_b32 s6, scc // (op 604) 
s_cmp_eq_i32 s6, 0 // (op 604) 
s_cselect_b32 s7, 0, -1 // (op 604) 

s_and_b32 s6, s12, 31 // (op 604) 
s_lshl_b32 s18, 1, s6 // (op 604) 
s_add_i32 s19, s18, s7 // (op 604) 


s_and_b32 s7, s1, s19 // (op 604) 


s_add_i32 s1, s51, s7 // (op 604) 


s_ashr_i32 s7, s1, s6 // (op 604) 


s_lshl_b32 s1, s12, 24 // (op 604) 
s_ashr_i32 s6, s1, 31 // (op 604) 

s_xor_b32 s1, s7, s6 // (op 604) 

s_sub_i32 s7, s1, s6 // (op 604) 


s_add_u32 s18, s4, 16 // (op 604) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 604) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 604) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 604) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 604) 

s_mul_i32 s1, s7, s18 // (op 604) 

s_sub_i32 s51, s0, s1 // (op 604) 


s_mul_hi_i32 s0, s16, s8 // (op 604) 
s_add_i32 s1, s0, s16 // (op 604) 

s_ashr_i32 s0, s1, 31 // (op 604) 
s_cmp_eq_i32 s8, 0 // (op 604) 
s_mov_b32 s6, scc // (op 604) 
s_cmp_eq_i32 s6, 0 // (op 604) 
s_cselect_b32 s7, 0, -1 // (op 604) 

s_and_b32 s6, s15, 31 // (op 604) 
s_lshl_b32 s68, 1, s6 // (op 604) 
s_add_i32 s69, s68, s7 // (op 604) 


s_and_b32 s7, s0, s69 // (op 604) 


s_add_i32 s0, s1, s7 // (op 604) 


s_ashr_i32 s1, s0, s6 // (op 604) 


s_lshl_b32 s0, s15, 24 // (op 604) 
s_ashr_i32 s6, s0, 31 // (op 604) 

s_xor_b32 s0, s1, s6 // (op 604) 

s_sub_i32 s1, s0, s6 // (op 604) 


s_mul_i32 s0, s14, s18 // (op 604) 

s_mul_i32 s6, s50, s0 // (op 604) 

s_mul_i32 s0, s1, s6 // (op 604) 

s_sub_u32 s6, s16, s0 // (op 604) 

s_mul_hi_i32 s0, s6, s13 // (op 604) 
s_add_i32 s7, s0, s6 // (op 604) 


s_ashr_i32 s0, s7, 31 // (op 604) 
s_cmp_eq_i32 s13, 0 // (op 604) 
s_mov_b32 s6, scc // (op 604) 
s_cmp_eq_i32 s6, 0 // (op 604) 
s_cselect_b32 s18, 0, -1 // (op 604) 

s_and_b32 s6, s49, 31 // (op 604) 
s_lshl_b32 s19, 1, s6 // (op 604) 
s_add_i32 s68, s19, s18 // (op 604) 


s_and_b32 s18, s0, s68 // (op 604) 


s_add_i32 s0, s7, s18 // (op 604) 


s_ashr_i32 s7, s0, s6 // (op 604) 


s_lshl_b32 s0, s49, 24 // (op 604) 
s_ashr_i32 s6, s0, 31 // (op 604) 

s_xor_b32 s0, s7, s6 // (op 604) 

s_sub_i32 s7, s0, s6 // (op 604) 


s_cmp_eq_i32 s1, 0 // (op 604) 
s_mov_b32 s0, scc // (op 604) 

s_cmp_eq_i32 s0, 0 // (op 604) 
s_cselect_b32 s1, s7, s51 // (op 604) 



v_lshl_add_u32 v17, s1, 4, v16 // (op 604) 


v_mul_lo_u32 v19, v17, s27 // (op 604) most significant: low of low * high
v_mul_hi_u32 v16, v17, s26 // (op 604) most significant: high of low * low
v_mul_lo_u32 v18, v17, s26 // (op 604) least significant: low of low * low
v_add_u32 v19, v19, v16 // (op 604) most significant: sum


s_ashr_i64 s[0:1], s[24:25], 8 // (op 604) 
s_lshr_b32 s51, s0, 1 // (op 604) 

v_lshl_add_u32 v16, s51, 9, v15 // (op 604) 


v_mov_b32 v21, 0 // (op 604) convert
v_mov_b32 v20, v16 // (op 604) convert

v_add_co_u32 v16, s[0:1], v20, v18 // (op 604) least significant half
v_addc_co_u32 v17, s[0:1], v21, v19, s[0:1] // (op 604) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 604) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 604) 

v_mov_b32 v15, v18 // (op 604) convert

s_mul_i32 s1, s26, s5 // (op 606) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 606) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 606) least significant
s_mul_hi_u32 s6, s26, s4 // (op 606) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 606) most significant: sum
s_add_u32 s1, s1, s6 // (op 606) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 606) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 606) 

s_mov_b32 s56, s28 // (op 606) reg expression
s_mov_b32 s57, s29 // (op 606) reg expression
s_mov_b32 s58, s0 // (op 606) reg expression
s_mov_b32 s59, 131072 // (op 606) call()

s_mul_hi_i32 s51, s16, s9 // (op 668) 
v_and_b32 v16, -16, v1 // (op 668) 
s_add_i32 s0, s51, s16 // (op 668) 

v_sub_u32 v17, v1, v16 // (op 668) 

s_ashr_i32 s1, s0, 31 // (op 668) 
v_lshrrev_b32 v16, 4, v1 // (op 668) 
s_cmp_eq_i32 s9, 0 // (op 668) 
s_mov_b32 s51, scc // (op 668) 
v_lshlrev_b32 v18, 1, v16 // (op 668) 

s_cmp_eq_i32 s51, 0 // (op 668) 
s_cselect_b32 s6, 0, -1 // (op 668) 

s_and_b32 s7, s10, 31 // (op 668) 
s_lshl_b32 s51, 1, s7 // (op 668) 
s_add_i32 s18, s51, s6 // (op 668) 


s_and_b32 s6, s1, s18 // (op 668) 


s_add_i32 s1, s0, s6 // (op 668) 


s_ashr_i32 s0, s1, s7 // (op 668) 


s_lshl_b32 s1, s10, 24 // (op 668) 
s_ashr_i32 s51, s1, 31 // (op 668) 

s_xor_b32 s1, s0, s51 // (op 668) 

s_sub_i32 s0, s1, s51 // (op 668) 


s_mul_hi_i32 s1, s0, s11 // (op 668) 
s_add_i32 s51, s1, s0 // (op 668) 

s_ashr_i32 s1, s51, 31 // (op 668) 
s_cmp_eq_i32 s11, 0 // (op 668) 
s_mov_b32 s6, scc // (op 668) 
s_cmp_eq_i32 s6, 0 // (op 668) 
s_cselect_b32 s7, 0, -1 // (op 668) 

s_and_b32 s6, s12, 31 // (op 668) 
s_lshl_b32 s18, 1, s6 // (op 668) 
s_add_i32 s19, s18, s7 // (op 668) 


s_and_b32 s7, s1, s19 // (op 668) 


s_add_i32 s1, s51, s7 // (op 668) 


s_ashr_i32 s7, s1, s6 // (op 668) 


s_lshl_b32 s1, s12, 24 // (op 668) 
s_ashr_i32 s6, s1, 31 // (op 668) 

s_xor_b32 s1, s7, s6 // (op 668) 

s_sub_i32 s7, s1, s6 // (op 668) 


s_mul_i32 s1, s0, s14 // (op 668) 

s_sub_u32 s0, s16, s1 // (op 668) 

s_mul_i32 s1, s7, s14 // (op 668) 
s_add_u32 s1, s1, s0 // (op 668) 


s_mul_hi_i32 s0, s16, s8 // (op 668) 
s_add_i32 s51, s0, s16 // (op 668) 

s_ashr_i32 s0, s51, 31 // (op 668) 
s_cmp_eq_i32 s8, 0 // (op 668) 
s_mov_b32 s6, scc // (op 668) 
s_cmp_eq_i32 s6, 0 // (op 668) 
s_cselect_b32 s7, 0, -1 // (op 668) 

s_and_b32 s6, s15, 31 // (op 668) 
s_lshl_b32 s18, 1, s6 // (op 668) 
s_add_i32 s19, s18, s7 // (op 668) 


s_and_b32 s7, s0, s19 // (op 668) 


s_add_i32 s0, s51, s7 // (op 668) 


s_ashr_i32 s7, s0, s6 // (op 668) 


s_lshl_b32 s0, s15, 24 // (op 668) 
s_ashr_i32 s6, s0, 31 // (op 668) 

s_xor_b32 s0, s7, s6 // (op 668) 

s_sub_i32 s7, s0, s6 // (op 668) 


s_add_u32 s18, s4, 16 // (op 668) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 668) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 668) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 668) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 668) 

s_mul_i32 s0, s14, s18 // (op 668) 

s_mul_i32 s6, s50, s0 // (op 668) 

s_mul_i32 s0, s7, s6 // (op 668) 

s_sub_u32 s6, s16, s0 // (op 668) 

s_mov_b32 s0, s6 // (op 668) 
s_mul_hi_i32 s51, s0, s13 // (op 668) 
s_add_i32 s18, s51, s0 // (op 668) 


s_ashr_i32 s0, s18, 31 // (op 668) 
s_cmp_eq_i32 s13, 0 // (op 668) 
s_mov_b32 s19, scc // (op 668) 
s_cmp_eq_i32 s19, 0 // (op 668) 
s_cselect_b32 s51, 0, -1 // (op 668) 

s_and_b32 s19, s49, 31 // (op 668) 
s_lshl_b32 s68, 1, s19 // (op 668) 
s_add_i32 s69, s68, s51 // (op 668) 


s_and_b32 s51, s0, s69 // (op 668) 


s_add_i32 s0, s18, s51 // (op 668) 


s_ashr_i32 s18, s0, s19 // (op 668) 


s_lshl_b32 s0, s49, 24 // (op 668) 
s_ashr_i32 s19, s0, 31 // (op 668) 

s_xor_b32 s0, s18, s19 // (op 668) 

s_sub_i32 s18, s0, s19 // (op 668) 


s_mul_i32 s0, s48, s14 // (op 668) 
s_add_u32 s68, s20, 16 // (op 668) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 668) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 668) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 668) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 668) 

s_sub_i32 s19, s68, s0 // (op 668) 

s_mul_i32 s51, s18, s19 // (op 668) 


s_sub_u32 s18, s6, s51 // (op 668) 


s_add_u32 s6, s18, s0 // (op 668) 


s_cmp_eq_i32 s7, 0 // (op 668) 
s_mov_b32 s0, scc // (op 668) 

s_cmp_eq_i32 s0, 0 // (op 668) 
s_cselect_b32 s7, s6, s1 // (op 668) 



v_lshl_add_u32 v16, s7, 4, v17 // (op 668) 


s_ashr_i64 s[0:1], s[24:25], 8 // (op 668) 
s_lshr_b32 s51, s0, 1 // (op 668) 

v_lshl_add_u32 v17, s51, 4, v18 // (op 668) 


v_mul_lo_u32 v18, v17, s30 // (op 668) 

v_add_u32 v17, v16, v18 // (op 668) 


s_mul_i32 s1, s30, s35 // (op 670) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 670) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 670) least significant
s_mul_hi_u32 s6, s30, s34 // (op 670) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 670) most significant: sum
s_add_u32 s1, s1, s6 // (op 670) most significant: sum


s_mov_b32 s60, s32 // (op 670) reg expression
s_mov_b32 s61, s33 // (op 670) reg expression
s_mov_b32 s62, s0 // (op 670) reg expression
s_mov_b32 s63, 131072 // (op 670) call()

s_mul_hi_i32 s51, s16, s9 // (op 732) 
v_lshrrev_b32 v16, 3, v1 // (op 732) 
s_add_i32 s0, s51, s16 // (op 732) 

v_lshlrev_b32 v18, 1, v16 // (op 732) 

s_ashr_i32 s1, s0, 31 // (op 732) 
v_and_b32 v16, -8, v1 // (op 732) 
s_cmp_eq_i32 s9, 0 // (op 732) 
s_mov_b32 s51, scc // (op 732) 
v_sub_u32 v19, v1, v16 // (op 732) 

s_cmp_eq_i32 s51, 0 // (op 732) 
s_cselect_b32 s6, 0, -1 // (op 732) 

s_and_b32 s7, s10, 31 // (op 732) 
s_lshl_b32 s51, 1, s7 // (op 732) 
s_add_i32 s18, s51, s6 // (op 732) 


s_and_b32 s6, s1, s18 // (op 732) 


s_add_i32 s1, s0, s6 // (op 732) 


s_ashr_i32 s0, s1, s7 // (op 732) 


s_lshl_b32 s1, s10, 24 // (op 732) 
s_ashr_i32 s51, s1, 31 // (op 732) 

s_xor_b32 s1, s0, s51 // (op 732) 

s_sub_i32 s0, s1, s51 // (op 732) 


s_mul_hi_i32 s1, s0, s11 // (op 732) 
s_add_i32 s51, s1, s0 // (op 732) 

s_ashr_i32 s1, s51, 31 // (op 732) 
s_cmp_eq_i32 s11, 0 // (op 732) 
s_mov_b32 s6, scc // (op 732) 
s_cmp_eq_i32 s6, 0 // (op 732) 
s_cselect_b32 s7, 0, -1 // (op 732) 

s_and_b32 s6, s12, 31 // (op 732) 
s_lshl_b32 s18, 1, s6 // (op 732) 
s_add_i32 s19, s18, s7 // (op 732) 


s_and_b32 s7, s1, s19 // (op 732) 


s_add_i32 s1, s51, s7 // (op 732) 


s_ashr_i32 s7, s1, s6 // (op 732) 


s_lshl_b32 s1, s12, 24 // (op 732) 
s_ashr_i32 s6, s1, 31 // (op 732) 

s_xor_b32 s1, s7, s6 // (op 732) 

s_sub_i32 s7, s1, s6 // (op 732) 


s_add_u32 s18, s4, 16 // (op 732) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 732) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 732) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 732) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 732) 

s_mul_i32 s1, s7, s18 // (op 732) 

s_sub_i32 s51, s0, s1 // (op 732) 


s_mul_hi_i32 s0, s16, s8 // (op 732) 
s_add_i32 s1, s0, s16 // (op 732) 

s_ashr_i32 s0, s1, 31 // (op 732) 
s_cmp_eq_i32 s8, 0 // (op 732) 
s_mov_b32 s6, scc // (op 732) 
s_cmp_eq_i32 s6, 0 // (op 732) 
s_cselect_b32 s7, 0, -1 // (op 732) 

s_and_b32 s6, s15, 31 // (op 732) 
s_lshl_b32 s68, 1, s6 // (op 732) 
s_add_i32 s69, s68, s7 // (op 732) 


s_and_b32 s7, s0, s69 // (op 732) 


s_add_i32 s0, s1, s7 // (op 732) 


s_ashr_i32 s1, s0, s6 // (op 732) 


s_lshl_b32 s0, s15, 24 // (op 732) 
s_ashr_i32 s6, s0, 31 // (op 732) 

s_xor_b32 s0, s1, s6 // (op 732) 

s_sub_i32 s1, s0, s6 // (op 732) 


s_mul_i32 s0, s14, s18 // (op 732) 

s_mul_i32 s6, s50, s0 // (op 732) 

s_mul_i32 s0, s1, s6 // (op 732) 

s_sub_u32 s6, s16, s0 // (op 732) 

s_mul_hi_i32 s0, s6, s13 // (op 732) 
s_add_i32 s7, s0, s6 // (op 732) 


s_ashr_i32 s0, s7, 31 // (op 732) 
s_cmp_eq_i32 s13, 0 // (op 732) 
s_mov_b32 s6, scc // (op 732) 
s_cmp_eq_i32 s6, 0 // (op 732) 
s_cselect_b32 s18, 0, -1 // (op 732) 

s_and_b32 s6, s49, 31 // (op 732) 
s_lshl_b32 s19, 1, s6 // (op 732) 
s_add_i32 s68, s19, s18 // (op 732) 


s_and_b32 s18, s0, s68 // (op 732) 


s_add_i32 s0, s7, s18 // (op 732) 


s_ashr_i32 s7, s0, s6 // (op 732) 


s_lshl_b32 s0, s49, 24 // (op 732) 
s_ashr_i32 s6, s0, 31 // (op 732) 

s_xor_b32 s0, s7, s6 // (op 732) 

s_sub_i32 s7, s0, s6 // (op 732) 


s_cmp_eq_i32 s1, 0 // (op 732) 
s_mov_b32 s0, scc // (op 732) 

s_cmp_eq_i32 s0, 0 // (op 732) 
s_cselect_b32 s1, s7, s51 // (op 732) 



v_lshl_add_u32 v16, s1, 4, v18 // (op 732) 


v_mul_lo_u32 v18, v16, s36 // (op 732) 

s_ashr_i64 s[0:1], s[24:25], 8 // (op 732) 
s_lshr_b32 s51, s0, 1 // (op 732) 

v_lshl_add_u32 v16, s51, 4, v19 // (op 732) 


v_add_u32 v19, v16, v18 // (op 732) 


s_mul_i32 s1, s36, s5 // (op 734) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 734) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 734) least significant
s_mul_hi_u32 s6, s36, s4 // (op 734) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 734) most significant: sum
s_add_u32 s1, s1, s6 // (op 734) most significant: sum


s_mov_b32 s64, s38 // (op 734) reg expression
s_mov_b32 s65, s39 // (op 734) reg expression
s_mov_b32 s66, s0 // (op 734) reg expression
s_mov_b32 s67, 131072 // (op 734) call()

v_lshlrev_b32 v16, 8, v1 // (op 825) 
v_lshrrev_b32 v18, 3, v16 // (op 825) 

v_and_b32 v16, -8, v1 // (op 878) 
v_sub_u32 v20, v1, v16 // (op 878) 

v_lshlrev_b32 v16, 5, v20 // (op 878) 

v_lshrrev_b32 v20, 3, v1 // (op 878) 
v_lshl_add_u32 v21, v20, 9, v16 // (op 878) 


v_lshlrev_b32 v16, 2, v21 // (op 878) 

v_lshrrev_b32 v20, 3, v16 // (op 878) 

v_and_b32 v16, -16, v1 // (op 931) 
v_sub_u32 v21, v1, v16 // (op 931) 

v_lshrrev_b32 v16, 4, v1 // (op 931) 
v_lshl_add_u32 v24, v16, 5, v21 // (op 931) 


v_and_b32 v16, -8, v1 // (op 984) 
v_sub_u32 v21, v1, v16 // (op 984) 

v_lshrrev_b32 v16, 3, v1 // (op 984) 
v_lshl_add_u32 v25, v16, 4, v21 // (op 984) 


s_ashr_i64 s[0:1], s[24:25], 8 // (op 198) 
s_and_b32 s51, s0, 4294967294 // (op 198) 

s_ashr_i64 s[0:1], s[24:25], 8 // (op 197) 
s_mov_b32 s6, s0 // (op 197) convert

s_cmp_lt_u32 s51, s6 // (op 197) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_197 // (op 197) Condition: Top (jump to RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_197 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_197:

s_add_u32 s51, s51, 1 // (op 199) 
s_lshl_b64 s[0:1], s[2:3], 2 // (op 202) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 202) 


v_mov_b32 v12, v13 // (op 202) 
v_mov_b32 v14, v12 // (op 202) reg expression
buffer_load_dwordx2 v[26:27], v14, s[52:55], 0 offen // (op 202) Load value
v_add_u32 v14, v14, v22 // (op 202) 
buffer_load_dwordx2 v[28:29], v14, s[52:55], 0 offen // (op 202) Load value
v_add_u32 v14, v14, v22 // (op 202) 
buffer_load_dwordx2 v[30:31], v14, s[52:55], 0 offen // (op 202) Load value
v_add_u32 v14, v14, v22 // (op 202) 
buffer_load_dwordx2 v[32:33], v14, s[52:55], 0 offen // (op 202) Load value



s_lshl_b64 s[0:1], s[2:3], 8 // (op 527) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 527) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 527) 

v_add_u32 v13, v13, s0 // (op 527) 


s_barrier  // (op 1185) 
s_lshl_b64 s[0:1], s[26:27], 2 // (op 225) 



v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 225) 


v_mov_b32 v12, v15 // (op 225) 
v_mov_b32 v14, v12 // (op 225) 
buffer_load_dwordx4 v[34:37], v14, s[56:59], 0 offen // (op 225) Load value
v_add_u32 v14, v14, v22 // (op 225) 
buffer_load_dwordx4 v[38:41], v14, s[56:59], 0 offen // (op 225) Load value



v_add_u32 v15, 128, v15 // (op 596) 
s_waitcnt vmcnt(5)
ds_write_b64 v18, v[26:27] offset:2176 // (op 203) Store local data 

v_mov_b64 v[22:23], s[30:31] // (op 226) convert

v_mov_b32 v12, v17 // (op 226) 
v_mov_b32 v14, v12 // (op 226) reg expression
s_waitcnt vmcnt(4)
ds_write_b64 v18, v[28:29] offset:2184 // (op 203) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v18, v[30:31] offset:2192 // (op 203) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v18, v[32:33] offset:2200 // (op 203) Store local data 
buffer_load_ubyte v42, v14, s[60:63], 0 offen offset: 0 // (op 226) Load value

v_add_u32 v14, v14, v22 // (op 226) 
s_barrier  // (op 1204) 
buffer_load_ubyte v43, v14, s[60:63], 0 offen offset: 0 // (op 226) Load value






s_lshl_b32 s0, s30, 3 // (op 660) 
v_add_u32 v17, v17, s0 // (op 660) 

s_waitcnt vmcnt(3)
ds_write_b128 v20, v[34:37] // (op 222) Store local data 

v_mov_b64 v[22:23], s[36:37] // (op 227) convert

v_mov_b32 v12, v19 // (op 227) 
v_mov_b32 v14, v12 // (op 227) 
buffer_load_ubyte v26, v14, s[64:67], 0 offen offset: 0 // (op 227) Load value
v_add_u32 v14, v14, v22 // (op 227) 
s_waitcnt vmcnt(3)
ds_write_b128 v20, v[38:41] offset:128 // (op 222) Store local data 
buffer_load_ubyte v27, v14, s[64:67], 0 offen offset: 0 // (op 227) Load value




v_add_u32 v19, 8, v19 // (op 724) 
s_barrier  // (op 1225) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1194) 






ds_read_b64_tr_b4 v[28:29], v3 offset:2176 // (op 205) Transpose load local data 
ds_read_b64_tr_b4 v[30:31], v3 offset:2304 // (op 205) Transpose load local data 
s_waitcnt vmcnt(3)
ds_write_b8 v24, v42 offset:4224 // (op 223) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1215) 
s_waitcnt vmcnt(2)
ds_write_b8 v24, v43 offset:4240 // (op 223) Store local data 




ds_read_b128 v[32:35], v4 // (op 207) Load local data 
s_barrier  // (op 1245) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1235) 





ds_read_u8 v12, v5 offset:4224 // (op 209) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v25, v26 offset:2048 // (op 224) Store local data 
s_waitcnt vmcnt(0)
ds_write_b8 v25, v27 offset:2056 // (op 224) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1255) 



ds_read_u8 v14, v6 offset:2048 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:31], v[32:35], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 212) 







ds_read_b64_tr_b4 v[26:27], v3 offset:3200 // (op 214) Transpose load local data 
ds_read_b64_tr_b4 v[28:29], v3 offset:3328 // (op 214) Transpose load local data 



ds_read_b128 v[30:33], v4 offset:64 // (op 216) Load local data 


ds_read_u8 v12, v5 offset:4288 // (op 218) Load local data 



ds_read_u8 v14, v6 offset:2052 // (op 220) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:33], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 221) 




s_ashr_i64 s[0:1], s[24:25], 8 // (op 197) 
s_mov_b32 s6, s0 // (op 197) convert

s_cmp_lt_u32 s51, s6 // (op 197) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_197 // (op 197) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_197 if true)
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_197:























s_barrier  // (op 1336) 








v_cmp_eq_f32 vcc, 0.00000, v0 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_157
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v11, v4 // (op 50) 
v_mul_f32 v13, v11, v5 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 

v_cvt_f16_f32 v4, v12 // (op 56) 
v_cvt_f16_f32 v5, v13 // (op 56) 
v_cvt_f16_f32 v6, v14 // (op 56) 
v_cvt_f16_f32 v7, v15 // (op 56) 

s_mul_hi_i32 s0, s16, s9 // (op 1026) 
v_lshrrev_b32 v3, 6, v1 // (op 1026) 
s_add_i32 s1, s0, s16 // (op 1026) 

v_and_b32 v8, 63, v1 // (op 1026) 
s_ashr_i32 s0, s1, 31 // (op 1026) 
v_lshrrev_b32 v9, 4, v8 // (op 1026) 
s_cmp_eq_i32 s9, 0 // (op 1026) 
s_mov_b32 s2, scc // (op 1026) 
v_lshlrev_b32 v10, 2, v9 // (op 1026) 

s_cmp_eq_i32 s2, 0 // (op 1026) 
s_cselect_b32 s3, 0, -1 // (op 1026) 

v_lshl_add_u32 v9, v3, 4, v10 // (op 1026) 


s_and_b32 s2, s10, 31 // (op 1026) 
v_lshrrev_b32 v3, 2, v8 // (op 1026) 

s_lshl_b32 s6, 1, s2 // (op 1026) 
v_and_b32 v8, 3, v3 // (op 1026) 

s_add_i32 s7, s6, s3 // (op 1026) 


v_and_b32 v3, 3, v1 // (op 1026) 
s_and_b32 s3, s0, s7 // (op 1026) 


v_lshl_add_u32 v10, v8, 2, v3 // (op 1026) 


s_add_i32 s0, s1, s3 // (op 1026) 


s_ashr_i32 s1, s0, s2 // (op 1026) 


s_lshl_b32 s0, s10, 24 // (op 1026) 
s_ashr_i32 s2, s0, 31 // (op 1026) 

s_xor_b32 s0, s1, s2 // (op 1026) 

s_sub_i32 s1, s0, s2 // (op 1026) 


s_mul_hi_i32 s0, s1, s11 // (op 1026) 
s_add_i32 s2, s0, s1 // (op 1026) 

s_ashr_i32 s0, s2, 31 // (op 1026) 
s_cmp_eq_i32 s11, 0 // (op 1026) 
s_mov_b32 s3, scc // (op 1026) 
s_cmp_eq_i32 s3, 0 // (op 1026) 
s_cselect_b32 s6, 0, -1 // (op 1026) 

s_and_b32 s3, s12, 31 // (op 1026) 
s_lshl_b32 s7, 1, s3 // (op 1026) 
s_add_i32 s18, s7, s6 // (op 1026) 


s_and_b32 s19, s0, s18 // (op 1026) 


s_add_i32 s0, s2, s19 // (op 1026) 


s_ashr_i32 s2, s0, s3 // (op 1026) 


s_lshl_b32 s0, s12, 24 // (op 1026) 
s_ashr_i32 s3, s0, 31 // (op 1026) 

s_xor_b32 s0, s2, s3 // (op 1026) 

s_sub_i32 s2, s0, s3 // (op 1026) 


s_mul_i32 s0, s1, s14 // (op 1026) 
s_sub_u32 s3, s16, s0 // (op 1026) 

s_mul_i32 s0, s2, s14 // (op 1026) 
s_add_u32 s0, s0, s3 // (op 1026) 

s_mul_hi_i32 s3, s16, s8 // (op 1026) 
s_add_i32 s6, s3, s16 // (op 1026) 

s_ashr_i32 s3, s6, 31 // (op 1026) 
s_cmp_eq_i32 s8, 0 // (op 1026) 
s_mov_b32 s7, scc // (op 1026) 
s_cmp_eq_i32 s7, 0 // (op 1026) 
s_cselect_b32 s18, 0, -1 // (op 1026) 

s_and_b32 s7, s15, 31 // (op 1026) 
s_lshl_b32 s19, 1, s7 // (op 1026) 
s_add_i32 s22, s19, s18 // (op 1026) 


s_and_b32 s18, s3, s22 // (op 1026) 


s_add_i32 s3, s6, s18 // (op 1026) 


s_ashr_i32 s6, s3, s7 // (op 1026) 


s_lshl_b32 s3, s15, 24 // (op 1026) 
s_ashr_i32 s7, s3, 31 // (op 1026) 

s_xor_b32 s3, s6, s7 // (op 1026) 

s_sub_i32 s6, s3, s7 // (op 1026) 


s_add_u32 s18, s4, 16 // (op 1026) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 1026) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 1026) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 1026) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 4 // (op 1026) 

s_mul_i32 s3, s14, s18 // (op 1026) 
s_mul_i32 s7, s50, s3 // (op 1026) 

s_mul_i32 s3, s6, s7 // (op 1026) 

s_sub_u32 s7, s16, s3 // (op 1026) 

s_mov_b32 s3, s7 // (op 1026) 
s_mul_hi_i32 s22, s3, s13 // (op 1026) 
s_add_i32 s23, s22, s3 // (op 1026) 


s_ashr_i32 s3, s23, 31 // (op 1026) 
s_cmp_eq_i32 s13, 0 // (op 1026) 
s_mov_b32 s22, scc // (op 1026) 
s_cmp_eq_i32 s22, 0 // (op 1026) 
s_cselect_b32 s24, 0, -1 // (op 1026) 

s_and_b32 s22, s49, 31 // (op 1026) 
s_lshl_b32 s25, 1, s22 // (op 1026) 
s_add_i32 s26, s25, s24 // (op 1026) 


s_and_b32 s24, s3, s26 // (op 1026) 


s_add_i32 s3, s23, s24 // (op 1026) 


s_ashr_i32 s23, s3, s22 // (op 1026) 


s_lshl_b32 s3, s49, 24 // (op 1026) 
s_ashr_i32 s22, s3, 31 // (op 1026) 

s_xor_b32 s3, s23, s22 // (op 1026) 

s_sub_i32 s23, s3, s22 // (op 1026) 


s_mul_i32 s3, s48, s14 // (op 1026) 
s_add_u32 s24, s20, 16 // (op 1026) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1026) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 1026) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 1026) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 4 // (op 1026) 

s_sub_i32 s22, s24, s3 // (op 1026) 

s_mul_i32 s24, s23, s22 // (op 1026) 

s_sub_u32 s22, s7, s24 // (op 1026) 


s_add_u32 s7, s22, s3 // (op 1026) 


s_cmp_eq_i32 s6, 0 // (op 1026) 
s_mov_b32 s3, scc // (op 1026) 

s_cmp_eq_i32 s3, 0 // (op 1026) 
s_cselect_b32 s6, s7, s0 // (op 1026) 


s_mul_i32 s0, s2, s18 // (op 1026) 


v_lshl_add_u32 v3, s6, 4, v9 // (op 1026) 


s_sub_i32 s2, s1, s0 // (op 1026) 


s_cmp_eq_i32 s3, 0 // (op 1026) 
s_cselect_b32 s22, s23, s2 // (op 1026) 



v_lshl_add_u32 v8, s22, 4, v10 // (op 1026) 


v_mul_lo_u32 v9, v8, s40 // (op 1026) 

v_add_lshl_u32 v8, v3, v9, 1 // (op 1026) 


s_mul_i32 s7, s40, s5 // (op 1028) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 1028) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 1028) least significant
s_mul_hi_u32 s1, s40, s4 // (op 1028) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 1028) most significant: sum
s_add_u32 s7, s7, s1 // (op 1028) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 1028) 

s_mov_b32 s0, s42 // (op 1028) reg expression
s_mov_b32 s1, s43 // (op 1028) reg expression
s_mov_b32 s2, s18 // (op 1028) reg expression
s_mov_b32 s3, 131072 // (op 1028) call()


v_pack_b32_f16 v12, v4, v5 // (op 58) 
v_pack_b32_f16 v13, v6, v7 // (op 58) 


buffer_store_dwordx2 v[12:13], v8, s[0:3], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_157
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_157:
s_mul_hi_i32 s6, s16, s9 // (op 795) 
v_lshrrev_b32 v3, 6, v1 // (op 795) 
s_add_i32 s7, s6, s16 // (op 795) 

v_and_b32 v4, 63, v1 // (op 795) 
s_ashr_i32 s6, s7, 31 // (op 795) 
v_lshrrev_b32 v5, 4, v4 // (op 795) 
s_cmp_eq_i32 s9, 0 // (op 795) 
s_mov_b32 s18, scc // (op 795) 
v_lshlrev_b32 v6, 2, v5 // (op 795) 

s_cmp_eq_i32 s18, 0 // (op 795) 
s_cselect_b32 s19, 0, -1 // (op 795) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 795) 


s_and_b32 s18, s10, 31 // (op 795) 
v_lshrrev_b32 v3, 2, v4 // (op 795) 

s_lshl_b32 s22, 1, s18 // (op 795) 
v_and_b32 v4, 3, v3 // (op 795) 

s_add_i32 s23, s22, s19 // (op 795) 


v_and_b32 v3, 3, v1 // (op 795) 
s_and_b32 s19, s6, s23 // (op 795) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 795) 


s_add_i32 s6, s7, s19 // (op 795) 


s_ashr_i32 s7, s6, s18 // (op 795) 


s_lshl_b32 s6, s10, 24 // (op 795) 
s_ashr_i32 s18, s6, 31 // (op 795) 

s_xor_b32 s6, s7, s18 // (op 795) 

s_sub_i32 s7, s6, s18 // (op 795) 


s_mul_hi_i32 s6, s7, s11 // (op 795) 
s_add_i32 s18, s6, s7 // (op 795) 

s_ashr_i32 s6, s18, 31 // (op 795) 
s_cmp_eq_i32 s11, 0 // (op 795) 
s_mov_b32 s19, scc // (op 795) 
s_cmp_eq_i32 s19, 0 // (op 795) 
s_cselect_b32 s22, 0, -1 // (op 795) 

s_and_b32 s19, s12, 31 // (op 795) 
s_lshl_b32 s23, 1, s19 // (op 795) 
s_add_i32 s24, s23, s22 // (op 795) 


s_and_b32 s22, s6, s24 // (op 795) 


s_add_i32 s6, s18, s22 // (op 795) 


s_ashr_i32 s18, s6, s19 // (op 795) 


s_lshl_b32 s6, s12, 24 // (op 795) 
s_ashr_i32 s19, s6, 31 // (op 795) 

s_xor_b32 s6, s18, s19 // (op 795) 

s_sub_i32 s18, s6, s19 // (op 795) 


s_mul_i32 s6, s7, s14 // (op 795) 
s_sub_u32 s19, s16, s6 // (op 795) 

s_mul_i32 s6, s18, s14 // (op 795) 
s_add_u32 s6, s6, s19 // (op 795) 

s_mul_hi_i32 s19, s16, s8 // (op 795) 
s_add_i32 s22, s19, s16 // (op 795) 

s_ashr_i32 s19, s22, 31 // (op 795) 
s_cmp_eq_i32 s8, 0 // (op 795) 
s_mov_b32 s23, scc // (op 795) 
s_cmp_eq_i32 s23, 0 // (op 795) 
s_cselect_b32 s24, 0, -1 // (op 795) 

s_and_b32 s23, s15, 31 // (op 795) 
s_lshl_b32 s25, 1, s23 // (op 795) 
s_add_i32 s26, s25, s24 // (op 795) 


s_and_b32 s24, s19, s26 // (op 795) 


s_add_i32 s19, s22, s24 // (op 795) 


s_ashr_i32 s22, s19, s23 // (op 795) 


s_lshl_b32 s19, s15, 24 // (op 795) 
s_ashr_i32 s23, s19, 31 // (op 795) 

s_xor_b32 s19, s22, s23 // (op 795) 

s_sub_i32 s22, s19, s23 // (op 795) 


s_add_u32 s38, s4, 16 // (op 795) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 795) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 795) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 795) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 795) 

s_mul_i32 s19, s14, s38 // (op 795) 
s_mul_i32 s23, s50, s19 // (op 795) 

s_mul_i32 s19, s22, s23 // (op 795) 

s_sub_u32 s23, s16, s19 // (op 795) 

s_mov_b32 s19, s23 // (op 795) 
s_mul_hi_i32 s24, s19, s13 // (op 795) 
s_add_i32 s25, s24, s19 // (op 795) 


s_ashr_i32 s19, s25, 31 // (op 795) 
s_cmp_eq_i32 s13, 0 // (op 795) 
s_mov_b32 s24, scc // (op 795) 
s_cmp_eq_i32 s24, 0 // (op 795) 
s_cselect_b32 s26, 0, -1 // (op 795) 

s_and_b32 s24, s49, 31 // (op 795) 
s_lshl_b32 s27, 1, s24 // (op 795) 
s_add_i32 s28, s27, s26 // (op 795) 


s_and_b32 s26, s19, s28 // (op 795) 


s_add_i32 s19, s25, s26 // (op 795) 


s_ashr_i32 s25, s19, s24 // (op 795) 


s_lshl_b32 s19, s49, 24 // (op 795) 
s_ashr_i32 s24, s19, 31 // (op 795) 

s_xor_b32 s19, s25, s24 // (op 795) 

s_sub_i32 s25, s19, s24 // (op 795) 


s_mul_i32 s19, s48, s14 // (op 795) 
s_add_u32 s26, s20, 16 // (op 795) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 795) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 795) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 795) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 795) 

s_sub_i32 s24, s26, s19 // (op 795) 

s_mul_i32 s26, s25, s24 // (op 795) 

s_sub_u32 s24, s23, s26 // (op 795) 


s_add_u32 s23, s24, s19 // (op 795) 


s_cmp_eq_i32 s22, 0 // (op 795) 
s_mov_b32 s19, scc // (op 795) 

s_cmp_eq_i32 s19, 0 // (op 795) 
s_cselect_b32 s22, s23, s6 // (op 795) 


s_mul_i32 s6, s18, s38 // (op 795) 


v_lshl_add_u32 v3, s22, 4, v5 // (op 795) 


s_sub_i32 s18, s7, s6 // (op 795) 


s_cmp_eq_i32 s19, 0 // (op 795) 
s_cselect_b32 s6, s25, s18 // (op 795) 



v_lshl_add_u32 v4, s6, 4, v6 // (op 795) 


v_mul_lo_u32 v5, v4, s44 // (op 795) 

v_add_lshl_u32 v4, v3, v5, 1 // (op 795) 


s_mul_i32 s7, s44, s5 // (op 797) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 797) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 797) least significant
s_mul_hi_u32 s19, s44, s4 // (op 797) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 797) most significant: sum
s_add_u32 s7, s7, s19 // (op 797) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 797) 

s_mov_b32 s36, s46 // (op 797) reg expression
s_mov_b32 s37, s47 // (op 797) reg expression
s_mov_b32 s38, s18 // (op 797) reg expression
s_mov_b32 s39, 131072 // (op 797) call()





buffer_load_dwordx2 v[6:7], v4, s[36:39], 0 offen // (op 170) Load value



v_accvgpr_read v12, a0 // (op 166) 
v_accvgpr_read v13, a1 // (op 166) 
v_accvgpr_read v14, a2 // (op 166) 
v_accvgpr_read v15, a3 // (op 166) 
v_mul_f32 v16, v11, v12 // (op 166) 
v_mul_f32 v17, v11, v13 // (op 166) 
v_mul_f32 v18, v11, v14 // (op 166) 
v_mul_f32 v19, v11, v15 // (op 166) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v4, v6 // (op 171) 
v_lshrrev_b32 v5, 16, v6 // (op 171) 
v_cvt_f32_f16 v5, v5 // (op 171) 
v_mul_f32 v12, v0, v4 // (op 171) 
v_mul_f32 v13, v0, v5 // (op 171) 
v_cvt_f32_f16 v4, v7 // (op 171) 
v_lshrrev_b32 v5, 16, v7 // (op 171) 
v_cvt_f32_f16 v5, v5 // (op 171) 
v_mul_f32 v14, v0, v4 // (op 171) 
v_mul_f32 v15, v0, v5 // (op 171) 


v_add_f32 v4, v12, v16 // (op 167) 
v_add_f32 v5, v13, v17 // (op 167) 
v_add_f32 v6, v14, v18 // (op 167) 
v_add_f32 v7, v15, v19 // (op 167) 


v_cvt_f16_f32 v12, v4 // (op 168) 
v_cvt_f16_f32 v13, v5 // (op 168) 
v_cvt_f16_f32 v14, v6 // (op 168) 
v_cvt_f16_f32 v15, v7 // (op 168) 

s_mul_hi_i32 s6, s16, s9 // (op 1043) 
v_lshrrev_b32 v3, 6, v1 // (op 1043) 
s_add_i32 s7, s6, s16 // (op 1043) 

v_and_b32 v4, 63, v1 // (op 1043) 
s_ashr_i32 s6, s7, 31 // (op 1043) 
v_lshrrev_b32 v5, 4, v4 // (op 1043) 
s_cmp_eq_i32 s9, 0 // (op 1043) 
s_mov_b32 s18, scc // (op 1043) 
v_lshlrev_b32 v6, 2, v5 // (op 1043) 

s_cmp_eq_i32 s18, 0 // (op 1043) 
s_cselect_b32 s19, 0, -1 // (op 1043) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 1043) 


s_and_b32 s18, s10, 31 // (op 1043) 
v_lshrrev_b32 v3, 2, v4 // (op 1043) 

s_lshl_b32 s22, 1, s18 // (op 1043) 
v_and_b32 v4, 3, v3 // (op 1043) 

s_add_i32 s23, s22, s19 // (op 1043) 


v_and_b32 v3, 3, v1 // (op 1043) 
s_and_b32 s19, s6, s23 // (op 1043) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 1043) 


s_add_i32 s6, s7, s19 // (op 1043) 


s_ashr_i32 s7, s6, s18 // (op 1043) 


s_lshl_b32 s6, s10, 24 // (op 1043) 
s_ashr_i32 s18, s6, 31 // (op 1043) 

s_xor_b32 s6, s7, s18 // (op 1043) 

s_sub_i32 s7, s6, s18 // (op 1043) 


s_mul_hi_i32 s6, s7, s11 // (op 1043) 
s_add_i32 s18, s6, s7 // (op 1043) 

s_ashr_i32 s6, s18, 31 // (op 1043) 
s_cmp_eq_i32 s11, 0 // (op 1043) 
s_mov_b32 s19, scc // (op 1043) 
s_cmp_eq_i32 s19, 0 // (op 1043) 
s_cselect_b32 s22, 0, -1 // (op 1043) 

s_and_b32 s19, s12, 31 // (op 1043) 
s_lshl_b32 s23, 1, s19 // (op 1043) 
s_add_i32 s24, s23, s22 // (op 1043) 


s_and_b32 s22, s6, s24 // (op 1043) 


s_add_i32 s6, s18, s22 // (op 1043) 


s_ashr_i32 s18, s6, s19 // (op 1043) 


s_lshl_b32 s6, s12, 24 // (op 1043) 
s_ashr_i32 s19, s6, 31 // (op 1043) 

s_xor_b32 s6, s18, s19 // (op 1043) 

s_sub_i32 s18, s6, s19 // (op 1043) 


s_mul_i32 s6, s7, s14 // (op 1043) 
s_sub_u32 s19, s16, s6 // (op 1043) 

s_mul_i32 s6, s18, s14 // (op 1043) 
s_add_u32 s6, s6, s19 // (op 1043) 

s_mul_hi_i32 s19, s16, s8 // (op 1043) 
s_add_i32 s22, s19, s16 // (op 1043) 

s_ashr_i32 s19, s22, 31 // (op 1043) 
s_cmp_eq_i32 s8, 0 // (op 1043) 
s_mov_b32 s23, scc // (op 1043) 
s_cmp_eq_i32 s23, 0 // (op 1043) 
s_cselect_b32 s24, 0, -1 // (op 1043) 

s_and_b32 s23, s15, 31 // (op 1043) 
s_lshl_b32 s25, 1, s23 // (op 1043) 
s_add_i32 s26, s25, s24 // (op 1043) 


s_and_b32 s24, s19, s26 // (op 1043) 


s_add_i32 s19, s22, s24 // (op 1043) 


s_ashr_i32 s22, s19, s23 // (op 1043) 


s_lshl_b32 s19, s15, 24 // (op 1043) 
s_ashr_i32 s23, s19, 31 // (op 1043) 

s_xor_b32 s19, s22, s23 // (op 1043) 

s_sub_i32 s22, s19, s23 // (op 1043) 


s_add_u32 s38, s4, 16 // (op 1043) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 1043) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 1043) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 1043) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 1043) 

s_mul_i32 s19, s14, s38 // (op 1043) 
s_mul_i32 s23, s50, s19 // (op 1043) 

s_mul_i32 s19, s22, s23 // (op 1043) 

s_sub_u32 s23, s16, s19 // (op 1043) 

s_mov_b32 s19, s23 // (op 1043) 
s_mul_hi_i32 s24, s19, s13 // (op 1043) 
s_add_i32 s25, s24, s19 // (op 1043) 


s_ashr_i32 s19, s25, 31 // (op 1043) 
s_cmp_eq_i32 s13, 0 // (op 1043) 
s_mov_b32 s24, scc // (op 1043) 
s_cmp_eq_i32 s24, 0 // (op 1043) 
s_cselect_b32 s26, 0, -1 // (op 1043) 

s_and_b32 s24, s49, 31 // (op 1043) 
s_lshl_b32 s27, 1, s24 // (op 1043) 
s_add_i32 s28, s27, s26 // (op 1043) 


s_and_b32 s26, s19, s28 // (op 1043) 


s_add_i32 s19, s25, s26 // (op 1043) 


s_ashr_i32 s25, s19, s24 // (op 1043) 


s_lshl_b32 s19, s49, 24 // (op 1043) 
s_ashr_i32 s24, s19, 31 // (op 1043) 

s_xor_b32 s19, s25, s24 // (op 1043) 

s_sub_i32 s25, s19, s24 // (op 1043) 


s_mul_i32 s19, s48, s14 // (op 1043) 
s_add_u32 s26, s20, 16 // (op 1043) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 1043) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 1043) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 1043) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 1043) 

s_sub_i32 s24, s26, s19 // (op 1043) 

s_mul_i32 s26, s25, s24 // (op 1043) 

s_sub_u32 s24, s23, s26 // (op 1043) 


s_add_u32 s23, s24, s19 // (op 1043) 


s_cmp_eq_i32 s22, 0 // (op 1043) 
s_mov_b32 s19, scc // (op 1043) 

s_cmp_eq_i32 s19, 0 // (op 1043) 
s_cselect_b32 s22, s23, s6 // (op 1043) 


s_mul_i32 s6, s18, s38 // (op 1043) 


v_lshl_add_u32 v3, s22, 4, v5 // (op 1043) 


s_sub_i32 s18, s7, s6 // (op 1043) 


s_cmp_eq_i32 s19, 0 // (op 1043) 
s_cselect_b32 s6, s25, s18 // (op 1043) 



v_lshl_add_u32 v4, s6, 4, v6 // (op 1043) 


v_mul_lo_u32 v5, v4, s40 // (op 1043) 

v_add_lshl_u32 v4, v3, v5, 1 // (op 1043) 


s_mul_i32 s7, s40, s5 // (op 1045) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 1045) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 1045) least significant
s_mul_hi_u32 s19, s40, s4 // (op 1045) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 1045) most significant: sum
s_add_u32 s7, s7, s19 // (op 1045) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 1045) 

s_mov_b32 s0, s42 // (op 1045) reg expression
s_mov_b32 s1, s43 // (op 1045) reg expression
s_mov_b32 s2, s18 // (op 1045) reg expression
s_mov_b32 s3, 131072 // (op 1045) call()


v_pack_b32_f16 v6, v12, v13 // (op 169) 
v_pack_b32_f16 v7, v14, v15 // (op 169) 


buffer_store_dwordx2 v[6:7], v4, s[0:3], 0 offen // (op 169) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_157:
















s_endpgm  // End of RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.LRR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end:
.size RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_, .LRR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end-RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 8704 // lds bytes
  .amdhsa_accum_offset 92
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
    DataTypeB: F4
    DestDataType: h
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
  - .name: RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
    .symbol: RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 8704
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 91
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

