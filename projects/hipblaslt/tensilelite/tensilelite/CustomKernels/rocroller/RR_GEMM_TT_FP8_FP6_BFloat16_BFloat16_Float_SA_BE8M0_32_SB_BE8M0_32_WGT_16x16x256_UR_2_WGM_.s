// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_,@function
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 875) 
v_and_b32 v3, 15, v1 // (op 875) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 875) 


v_and_b32 v0, 48, v1 // (op 875) 
v_lshl_add_u32 v3, v4, 8, v0 // (op 875) 


v_and_b32 v0, 63, v1 // (op 928) 
v_and_b32 v4, 48, v1 // (op 928) 
v_sub_u32 v5, v0, v4 // (op 928) 

v_lshrrev_b32 v4, 4, v0 // (op 928) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 928) 


v_lshlrev_b32 v4, 4, v0 // (op 928) 
v_mul_lo_u32 v5, v4, 6 // (op 928) 

v_lshrrev_b32 v4, 3, v5 // (op 928) 

v_and_b32 v5, 268435455, v0 // (op 928) 

v_lshlrev_b32 v0, 2, v5 // (op 928) 

v_add_u32 v5, v4, v0 // (op 928) 


v_lshrrev_b32 v0, 6, v1 // (op 981) 
v_and_b32 v4, 15, v1 // (op 981) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 981) 


v_and_b32 v0, 63, v1 // (op 981) 
v_lshrrev_b32 v4, 4, v0 // (op 981) 

v_lshl_add_u32 v0, v6, 3, v4 // (op 981) 


v_and_b32 v4, 48, v1 // (op 1034) 
v_and_b32 v6, 15, v1 // (op 1034) 
v_add_u32 v7, v4, v6 // (op 1034) 


v_lshrrev_b32 v4, 6, v1 // (op 1107) 
v_and_b32 v6, 15, v1 // (op 1107) 
v_lshl_add_u32 v8, v4, 4, v6 // (op 1107) 


v_and_b32 v4, 48, v1 // (op 1107) 
v_lshl_add_u32 v6, v8, 8, v4 // (op 1107) 


v_and_b32 v4, 63, v1 // (op 1131) 
v_and_b32 v8, 48, v1 // (op 1131) 
v_sub_u32 v9, v4, v8 // (op 1131) 

v_lshrrev_b32 v8, 4, v4 // (op 1131) 

v_lshl_add_u32 v4, v8, 5, v9 // (op 1131) 


v_lshlrev_b32 v8, 4, v4 // (op 1131) 
v_mul_lo_u32 v9, v8, 6 // (op 1131) 

v_lshrrev_b32 v8, 3, v9 // (op 1131) 

v_and_b32 v9, 268435455, v4 // (op 1131) 

v_lshlrev_b32 v4, 2, v9 // (op 1131) 

v_add_u32 v9, v8, v4 // (op 1131) 


v_lshrrev_b32 v4, 6, v1 // (op 1155) 
v_and_b32 v8, 15, v1 // (op 1155) 
v_lshl_add_u32 v10, v4, 4, v8 // (op 1155) 


v_and_b32 v4, 63, v1 // (op 1155) 
v_lshrrev_b32 v8, 4, v4 // (op 1155) 

v_lshl_add_u32 v4, v10, 3, v8 // (op 1155) 


v_and_b32 v8, 48, v1 // (op 1179) 
v_and_b32 v10, 15, v1 // (op 1179) 
v_add_u32 v11, v8, v10 // (op 1179) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v10, s9 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
s_mul_hi_i32 s0, s16, s11 // (op 586) 
v_lshrrev_b32 v12, 4, v1 // (op 586) 
s_add_i32 s1, s0, s16 // (op 586) 

v_and_b32 v13, -16, v1 // (op 586) 
s_ashr_i32 s0, s1, 31 // (op 586) 
v_sub_u32 v14, v1, v13 // (op 586) 

s_cmp_eq_i32 s11, 0 // (op 586) 
s_mov_b32 s8, scc // (op 586) 
v_lshlrev_b32 v13, 4, v14 // (op 586) 

s_cmp_eq_i32 s8, 0 // (op 586) 
s_cselect_b32 s9, 0, -1 // (op 586) 

v_mov_b32 v15, 0 // (op 586) convert
v_mov_b32 v14, v13 // (op 586) convert

s_and_b32 s8, s12, 31 // (op 586) 
s_lshl_b32 s18, 1, s8 // (op 586) 
s_add_i32 s19, s18, s9 // (op 586) 


s_and_b32 s9, s0, s19 // (op 586) 


s_add_i32 s0, s1, s9 // (op 586) 


s_ashr_i32 s1, s0, s8 // (op 586) 


s_lshl_b32 s0, s12, 24 // (op 586) 
s_ashr_i32 s8, s0, 31 // (op 586) 

s_xor_b32 s0, s1, s8 // (op 586) 

s_sub_i32 s1, s0, s8 // (op 586) 


s_mul_hi_i32 s0, s1, s13 // (op 586) 
s_add_i32 s8, s0, s1 // (op 586) 

s_ashr_i32 s0, s8, 31 // (op 586) 
s_cmp_eq_i32 s13, 0 // (op 586) 
s_mov_b32 s9, scc // (op 586) 
s_cmp_eq_i32 s9, 0 // (op 586) 
s_cselect_b32 s18, 0, -1 // (op 586) 

s_and_b32 s9, s14, 31 // (op 586) 
s_lshl_b32 s19, 1, s9 // (op 586) 
s_add_i32 s51, s19, s18 // (op 586) 


s_and_b32 s18, s0, s51 // (op 586) 


s_add_i32 s0, s8, s18 // (op 586) 


s_ashr_i32 s8, s0, s9 // (op 586) 


s_lshl_b32 s0, s14, 24 // (op 586) 
s_ashr_i32 s9, s0, 31 // (op 586) 

s_xor_b32 s0, s8, s9 // (op 586) 

s_sub_i32 s8, s0, s9 // (op 586) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s46 // (op 586) 

s_sub_u32 s1, s16, s0 // (op 586) 

s_mul_i32 s0, s8, s46 // (op 586) 
s_add_u32 s0, s0, s1 // (op 586) 


s_mul_hi_i32 s1, s16, s10 // (op 586) 
s_add_i32 s8, s1, s16 // (op 586) 

s_ashr_i32 s1, s8, 31 // (op 586) 
s_cmp_eq_i32 s10, 0 // (op 586) 
s_mov_b32 s9, scc // (op 586) 
s_cmp_eq_i32 s9, 0 // (op 586) 
s_cselect_b32 s18, 0, -1 // (op 586) 

s_and_b32 s9, s49, 31 // (op 586) 
s_lshl_b32 s19, 1, s9 // (op 586) 
s_add_i32 s51, s19, s18 // (op 586) 


s_and_b32 s18, s1, s51 // (op 586) 


s_add_i32 s1, s8, s18 // (op 586) 


s_ashr_i32 s8, s1, s9 // (op 586) 


s_lshl_b32 s1, s49, 24 // (op 586) 
s_ashr_i32 s9, s1, 31 // (op 586) 

s_xor_b32 s1, s8, s9 // (op 586) 

s_sub_i32 s8, s1, s9 // (op 586) 


s_add_u32 s18, s2, 16 // (op 586) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 586) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 586) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 586) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 586) 

s_mul_i32 s1, s46, s18 // (op 586) 

s_mul_i32 s9, s50, s1 // (op 586) 

s_mul_i32 s1, s8, s9 // (op 586) 

s_sub_u32 s9, s16, s1 // (op 586) 

s_mov_b32 s1, s9 // (op 586) 
s_mul_hi_i32 s18, s1, s15 // (op 586) 
s_add_i32 s19, s18, s1 // (op 586) 


s_ashr_i32 s1, s19, 31 // (op 586) 
s_cmp_eq_i32 s15, 0 // (op 586) 
s_mov_b32 s18, scc // (op 586) 
s_cmp_eq_i32 s18, 0 // (op 586) 
s_cselect_b32 s51, 0, -1 // (op 586) 

s_and_b32 s18, s48, 31 // (op 586) 
s_lshl_b32 s52, 1, s18 // (op 586) 
s_add_i32 s53, s52, s51 // (op 586) 


s_and_b32 s51, s1, s53 // (op 586) 


s_add_i32 s1, s19, s51 // (op 586) 


s_ashr_i32 s19, s1, s18 // (op 586) 


s_lshl_b32 s1, s48, 24 // (op 586) 
s_ashr_i32 s18, s1, 31 // (op 586) 

s_xor_b32 s1, s19, s18 // (op 586) 

s_sub_i32 s19, s1, s18 // (op 586) 


s_mul_i32 s1, s47, s46 // (op 586) 
s_add_u32 s52, s6, 16 // (op 586) Start of Int64 add, locking SCC
s_addc_u32 s53, s7, 0 // (op 586) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 586) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 586) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 586) 

s_sub_i32 s18, s52, s1 // (op 586) 

s_mul_i32 s51, s19, s18 // (op 586) 


s_sub_u32 s18, s9, s51 // (op 586) 


s_add_u32 s9, s18, s1 // (op 586) 


s_cmp_eq_i32 s8, 0 // (op 586) 
s_mov_b32 s1, scc // (op 586) 

s_cmp_eq_i32 s1, 0 // (op 586) 
s_cselect_b32 s8, s9, s0 // (op 586) 



v_lshl_add_u32 v13, s8, 4, v12 // (op 586) 


v_mul_lo_u32 v17, v13, s5 // (op 586) most significant: low of low * high
v_mul_hi_u32 v12, v13, s4 // (op 586) most significant: high of low * low
v_mul_lo_u32 v16, v13, s4 // (op 586) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 586) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 586) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 586) most significant half



s_mul_i32 s1, s4, s7 // (op 588) most significant: low of low * high
s_mul_i32 s8, s5, s6 // (op 588) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 588) least significant
s_mul_hi_u32 s9, s4, s6 // (op 588) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 588) most significant: sum
s_add_u32 s1, s1, s9 // (op 588) most significant: sum


s_mov_b32 s52, s20 // (op 588) reg expression
s_mov_b32 s53, s21 // (op 588) reg expression
s_mov_b32 s54, s0 // (op 588) reg expression
s_mov_b32 s55, 131072 // (op 588) call()

s_mul_hi_i32 s51, s16, s11 // (op 605) 
v_lshrrev_b32 v12, 4, v1 // (op 605) 
s_add_i32 s0, s51, s16 // (op 605) 

v_and_b32 v13, -16, v1 // (op 605) 
s_ashr_i32 s1, s0, 31 // (op 605) 
v_sub_u32 v14, v1, v13 // (op 605) 

s_cmp_eq_i32 s11, 0 // (op 605) 
s_mov_b32 s51, scc // (op 605) 
v_lshlrev_b32 v13, 4, v14 // (op 605) 

s_cmp_eq_i32 s51, 0 // (op 605) 
s_cselect_b32 s8, 0, -1 // (op 605) 

v_mov_b32 v15, 0 // (op 605) convert
v_mov_b32 v14, v13 // (op 605) convert

s_and_b32 s9, s12, 31 // (op 605) 
s_lshl_b32 s51, 1, s9 // (op 605) 
s_add_i32 s18, s51, s8 // (op 605) 


s_and_b32 s8, s1, s18 // (op 605) 


s_add_i32 s1, s0, s8 // (op 605) 


s_ashr_i32 s0, s1, s9 // (op 605) 


s_lshl_b32 s1, s12, 24 // (op 605) 
s_ashr_i32 s51, s1, 31 // (op 605) 

s_xor_b32 s1, s0, s51 // (op 605) 

s_sub_i32 s0, s1, s51 // (op 605) 


s_mul_hi_i32 s1, s0, s13 // (op 605) 
s_add_i32 s51, s1, s0 // (op 605) 

s_ashr_i32 s1, s51, 31 // (op 605) 
s_cmp_eq_i32 s13, 0 // (op 605) 
s_mov_b32 s8, scc // (op 605) 
s_cmp_eq_i32 s8, 0 // (op 605) 
s_cselect_b32 s9, 0, -1 // (op 605) 

s_and_b32 s8, s14, 31 // (op 605) 
s_lshl_b32 s18, 1, s8 // (op 605) 
s_add_i32 s19, s18, s9 // (op 605) 


s_and_b32 s9, s1, s19 // (op 605) 


s_add_i32 s1, s51, s9 // (op 605) 


s_ashr_i32 s9, s1, s8 // (op 605) 


s_lshl_b32 s1, s14, 24 // (op 605) 
s_ashr_i32 s8, s1, 31 // (op 605) 

s_xor_b32 s1, s9, s8 // (op 605) 

s_sub_i32 s9, s1, s8 // (op 605) 


s_mul_i32 s1, s0, s46 // (op 605) 

s_sub_u32 s0, s16, s1 // (op 605) 

s_mul_i32 s1, s9, s46 // (op 605) 
s_add_u32 s1, s1, s0 // (op 605) 


s_mul_hi_i32 s0, s16, s10 // (op 605) 
s_add_i32 s51, s0, s16 // (op 605) 

s_ashr_i32 s0, s51, 31 // (op 605) 
s_cmp_eq_i32 s10, 0 // (op 605) 
s_mov_b32 s8, scc // (op 605) 
s_cmp_eq_i32 s8, 0 // (op 605) 
s_cselect_b32 s9, 0, -1 // (op 605) 

s_and_b32 s8, s49, 31 // (op 605) 
s_lshl_b32 s18, 1, s8 // (op 605) 
s_add_i32 s19, s18, s9 // (op 605) 


s_and_b32 s9, s0, s19 // (op 605) 


s_add_i32 s0, s51, s9 // (op 605) 


s_ashr_i32 s9, s0, s8 // (op 605) 


s_lshl_b32 s0, s49, 24 // (op 605) 
s_ashr_i32 s8, s0, 31 // (op 605) 

s_xor_b32 s0, s9, s8 // (op 605) 

s_sub_i32 s9, s0, s8 // (op 605) 


s_add_u32 s18, s2, 16 // (op 605) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 605) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 605) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 605) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 605) 

s_mul_i32 s0, s46, s18 // (op 605) 

s_mul_i32 s8, s50, s0 // (op 605) 

s_mul_i32 s0, s9, s8 // (op 605) 

s_sub_u32 s8, s16, s0 // (op 605) 

s_mov_b32 s0, s8 // (op 605) 
s_mul_hi_i32 s51, s0, s15 // (op 605) 
s_add_i32 s18, s51, s0 // (op 605) 


s_ashr_i32 s0, s18, 31 // (op 605) 
s_cmp_eq_i32 s15, 0 // (op 605) 
s_mov_b32 s19, scc // (op 605) 
s_cmp_eq_i32 s19, 0 // (op 605) 
s_cselect_b32 s51, 0, -1 // (op 605) 

s_and_b32 s19, s48, 31 // (op 605) 
s_lshl_b32 s56, 1, s19 // (op 605) 
s_add_i32 s57, s56, s51 // (op 605) 


s_and_b32 s51, s0, s57 // (op 605) 


s_add_i32 s0, s18, s51 // (op 605) 


s_ashr_i32 s18, s0, s19 // (op 605) 


s_lshl_b32 s0, s48, 24 // (op 605) 
s_ashr_i32 s19, s0, 31 // (op 605) 

s_xor_b32 s0, s18, s19 // (op 605) 

s_sub_i32 s18, s0, s19 // (op 605) 


s_mul_i32 s0, s47, s46 // (op 605) 
s_add_u32 s56, s6, 16 // (op 605) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 605) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 605) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 605) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 605) 

s_sub_i32 s19, s56, s0 // (op 605) 

s_mul_i32 s51, s18, s19 // (op 605) 


s_sub_u32 s18, s8, s51 // (op 605) 


s_add_u32 s8, s18, s0 // (op 605) 


s_cmp_eq_i32 s9, 0 // (op 605) 
s_mov_b32 s0, scc // (op 605) 

s_cmp_eq_i32 s0, 0 // (op 605) 
s_cselect_b32 s9, s8, s1 // (op 605) 



v_lshl_add_u32 v13, s9, 4, v12 // (op 605) 


v_mul_lo_u32 v19, v13, s5 // (op 605) most significant: low of low * high
v_mul_hi_u32 v12, v13, s4 // (op 605) most significant: high of low * low
v_mul_lo_u32 v18, v13, s4 // (op 605) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 605) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 605) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 605) most significant half



s_mul_i32 s1, s4, s7 // (op 607) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 607) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 607) least significant
s_mul_hi_u32 s8, s4, s6 // (op 607) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 607) most significant: sum
s_add_u32 s1, s1, s8 // (op 607) most significant: sum


s_mov_b32 s52, s20 // (op 607) reg expression
s_mov_b32 s53, s21 // (op 607) reg expression
s_mov_b32 s54, s0 // (op 607) reg expression
s_mov_b32 s55, 131072 // (op 607) call()

s_mul_hi_i32 s51, s16, s11 // (op 650) 
v_lshlrev_b32 v12, 2, v1 // (op 650) 
s_add_i32 s0, s51, s16 // (op 650) 

s_ashr_i32 s1, s0, 31 // (op 650) 
s_cmp_eq_i32 s11, 0 // (op 650) 
s_mov_b32 s51, scc // (op 650) 
s_cmp_eq_i32 s51, 0 // (op 650) 
s_cselect_b32 s8, 0, -1 // (op 650) 

s_and_b32 s9, s12, 31 // (op 650) 
s_lshl_b32 s51, 1, s9 // (op 650) 
s_add_i32 s18, s51, s8 // (op 650) 


s_and_b32 s8, s1, s18 // (op 650) 


s_add_i32 s1, s0, s8 // (op 650) 


s_ashr_i32 s0, s1, s9 // (op 650) 


s_lshl_b32 s1, s12, 24 // (op 650) 
s_ashr_i32 s51, s1, 31 // (op 650) 

s_xor_b32 s1, s0, s51 // (op 650) 

s_sub_i32 s0, s1, s51 // (op 650) 


s_mul_hi_i32 s1, s0, s13 // (op 650) 
s_add_i32 s51, s1, s0 // (op 650) 

s_ashr_i32 s1, s51, 31 // (op 650) 
s_cmp_eq_i32 s13, 0 // (op 650) 
s_mov_b32 s8, scc // (op 650) 
s_cmp_eq_i32 s8, 0 // (op 650) 
s_cselect_b32 s9, 0, -1 // (op 650) 

s_and_b32 s8, s14, 31 // (op 650) 
s_lshl_b32 s18, 1, s8 // (op 650) 
s_add_i32 s19, s18, s9 // (op 650) 


s_and_b32 s9, s1, s19 // (op 650) 


s_add_i32 s1, s51, s9 // (op 650) 


s_ashr_i32 s9, s1, s8 // (op 650) 


s_lshl_b32 s1, s14, 24 // (op 650) 
s_ashr_i32 s8, s1, 31 // (op 650) 

s_xor_b32 s1, s9, s8 // (op 650) 

s_sub_i32 s9, s1, s8 // (op 650) 


s_add_u32 s18, s2, 16 // (op 650) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 650) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 650) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 650) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 650) 

s_mul_i32 s1, s9, s18 // (op 650) 

s_sub_i32 s51, s0, s1 // (op 650) 


s_mul_hi_i32 s0, s16, s10 // (op 650) 
s_add_i32 s1, s0, s16 // (op 650) 

s_ashr_i32 s0, s1, 31 // (op 650) 
s_cmp_eq_i32 s10, 0 // (op 650) 
s_mov_b32 s8, scc // (op 650) 
s_cmp_eq_i32 s8, 0 // (op 650) 
s_cselect_b32 s9, 0, -1 // (op 650) 

s_and_b32 s8, s49, 31 // (op 650) 
s_lshl_b32 s56, 1, s8 // (op 650) 
s_add_i32 s57, s56, s9 // (op 650) 


s_and_b32 s9, s0, s57 // (op 650) 


s_add_i32 s0, s1, s9 // (op 650) 


s_ashr_i32 s1, s0, s8 // (op 650) 


s_lshl_b32 s0, s49, 24 // (op 650) 
s_ashr_i32 s8, s0, 31 // (op 650) 

s_xor_b32 s0, s1, s8 // (op 650) 

s_sub_i32 s1, s0, s8 // (op 650) 


s_mul_i32 s0, s46, s18 // (op 650) 

s_mul_i32 s8, s50, s0 // (op 650) 

s_mul_i32 s0, s1, s8 // (op 650) 

s_sub_u32 s8, s16, s0 // (op 650) 

s_mul_hi_i32 s0, s8, s15 // (op 650) 
s_add_i32 s9, s0, s8 // (op 650) 


s_ashr_i32 s0, s9, 31 // (op 650) 
s_cmp_eq_i32 s15, 0 // (op 650) 
s_mov_b32 s8, scc // (op 650) 
s_cmp_eq_i32 s8, 0 // (op 650) 
s_cselect_b32 s18, 0, -1 // (op 650) 

s_and_b32 s8, s48, 31 // (op 650) 
s_lshl_b32 s19, 1, s8 // (op 650) 
s_add_i32 s56, s19, s18 // (op 650) 


s_and_b32 s18, s0, s56 // (op 650) 


s_add_i32 s0, s9, s18 // (op 650) 


s_ashr_i32 s9, s0, s8 // (op 650) 


s_lshl_b32 s0, s48, 24 // (op 650) 
s_ashr_i32 s8, s0, 31 // (op 650) 

s_xor_b32 s0, s9, s8 // (op 650) 

s_sub_i32 s9, s0, s8 // (op 650) 


s_cmp_eq_i32 s1, 0 // (op 650) 
s_mov_b32 s0, scc // (op 650) 

s_cmp_eq_i32 s0, 0 // (op 650) 
s_cselect_b32 s1, s9, s51 // (op 650) 



s_lshl_b32 s0, s1, 4 // (op 650) 

s_mov_b32 s9, 0 // (op 650) convert
s_mov_b32 s8, s0 // (op 650) convert

v_mul_lo_u32 v15, v12, s23 // (op 650) most significant: low of low * high
v_mul_hi_u32 v13, v12, s22 // (op 650) most significant: high of low * low
v_mul_lo_u32 v14, v12, s22 // (op 650) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 650) most significant: sum

v_mov_b32 v13, s8 // (op 650) 
v_mov_b32 v20, s9 // (op 650) 
v_add_co_u32 v14, s[0:1], v14, v13 // (op 650) least significant half
v_addc_co_u32 v15, s[0:1], v15, v20, s[0:1] // (op 650) most significant half





v_mul_lo_u32 v13, v15, 6 // (op 650) most significant: low of high * low
v_mul_hi_u32 v20, v14, 6 // (op 650) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 650) least significant: low of low * low
v_add_u32 v13, v13, v20 // (op 650) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 650) 

v_mov_b32 v12, v14 // (op 650) convert

s_mul_i32 s1, s27, 6 // (op 652) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 652) least significant
s_mul_hi_u32 s51, s26, 6 // (op 652) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 652) most significant: sum

s_mul_i32 s9, s22, s1 // (op 652) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 652) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 652) least significant
s_mul_hi_u32 s18, s22, s0 // (op 652) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 652) most significant: sum
s_add_u32 s9, s9, s18 // (op 652) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 652) 

s_mov_b32 s56, s24 // (op 652) reg expression
s_mov_b32 s57, s25 // (op 652) reg expression
s_mov_b32 s58, s0 // (op 652) reg expression
s_mov_b32 s59, 131072 // (op 652) call()

s_mul_hi_i32 s51, s16, s11 // (op 669) 
v_lshlrev_b32 v13, 2, v1 // (op 669) 
s_add_i32 s0, s51, s16 // (op 669) 

s_ashr_i32 s1, s0, 31 // (op 669) 
s_cmp_eq_i32 s11, 0 // (op 669) 
s_mov_b32 s51, scc // (op 669) 
s_cmp_eq_i32 s51, 0 // (op 669) 
s_cselect_b32 s8, 0, -1 // (op 669) 

s_and_b32 s9, s12, 31 // (op 669) 
s_lshl_b32 s51, 1, s9 // (op 669) 
s_add_i32 s18, s51, s8 // (op 669) 


s_and_b32 s8, s1, s18 // (op 669) 


s_add_i32 s1, s0, s8 // (op 669) 


s_ashr_i32 s0, s1, s9 // (op 669) 


s_lshl_b32 s1, s12, 24 // (op 669) 
s_ashr_i32 s51, s1, 31 // (op 669) 

s_xor_b32 s1, s0, s51 // (op 669) 

s_sub_i32 s0, s1, s51 // (op 669) 


s_mul_hi_i32 s1, s0, s13 // (op 669) 
s_add_i32 s51, s1, s0 // (op 669) 

s_ashr_i32 s1, s51, 31 // (op 669) 
s_cmp_eq_i32 s13, 0 // (op 669) 
s_mov_b32 s8, scc // (op 669) 
s_cmp_eq_i32 s8, 0 // (op 669) 
s_cselect_b32 s9, 0, -1 // (op 669) 

s_and_b32 s8, s14, 31 // (op 669) 
s_lshl_b32 s18, 1, s8 // (op 669) 
s_add_i32 s19, s18, s9 // (op 669) 


s_and_b32 s9, s1, s19 // (op 669) 


s_add_i32 s1, s51, s9 // (op 669) 


s_ashr_i32 s9, s1, s8 // (op 669) 


s_lshl_b32 s1, s14, 24 // (op 669) 
s_ashr_i32 s8, s1, 31 // (op 669) 

s_xor_b32 s1, s9, s8 // (op 669) 

s_sub_i32 s9, s1, s8 // (op 669) 


s_add_u32 s18, s2, 16 // (op 669) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 669) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 669) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 669) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 669) 

s_mul_i32 s1, s9, s18 // (op 669) 

s_sub_i32 s51, s0, s1 // (op 669) 


s_mul_hi_i32 s0, s16, s10 // (op 669) 
s_add_i32 s1, s0, s16 // (op 669) 

s_ashr_i32 s0, s1, 31 // (op 669) 
s_cmp_eq_i32 s10, 0 // (op 669) 
s_mov_b32 s8, scc // (op 669) 
s_cmp_eq_i32 s8, 0 // (op 669) 
s_cselect_b32 s9, 0, -1 // (op 669) 

s_and_b32 s8, s49, 31 // (op 669) 
s_lshl_b32 s60, 1, s8 // (op 669) 
s_add_i32 s61, s60, s9 // (op 669) 


s_and_b32 s9, s0, s61 // (op 669) 


s_add_i32 s0, s1, s9 // (op 669) 


s_ashr_i32 s1, s0, s8 // (op 669) 


s_lshl_b32 s0, s49, 24 // (op 669) 
s_ashr_i32 s8, s0, 31 // (op 669) 

s_xor_b32 s0, s1, s8 // (op 669) 

s_sub_i32 s1, s0, s8 // (op 669) 


s_mul_i32 s0, s46, s18 // (op 669) 

s_mul_i32 s8, s50, s0 // (op 669) 

s_mul_i32 s0, s1, s8 // (op 669) 

s_sub_u32 s8, s16, s0 // (op 669) 

s_mul_hi_i32 s0, s8, s15 // (op 669) 
s_add_i32 s9, s0, s8 // (op 669) 


s_ashr_i32 s0, s9, 31 // (op 669) 
s_cmp_eq_i32 s15, 0 // (op 669) 
s_mov_b32 s8, scc // (op 669) 
s_cmp_eq_i32 s8, 0 // (op 669) 
s_cselect_b32 s18, 0, -1 // (op 669) 

s_and_b32 s8, s48, 31 // (op 669) 
s_lshl_b32 s19, 1, s8 // (op 669) 
s_add_i32 s60, s19, s18 // (op 669) 


s_and_b32 s18, s0, s60 // (op 669) 


s_add_i32 s0, s9, s18 // (op 669) 


s_ashr_i32 s9, s0, s8 // (op 669) 


s_lshl_b32 s0, s48, 24 // (op 669) 
s_ashr_i32 s8, s0, 31 // (op 669) 

s_xor_b32 s0, s9, s8 // (op 669) 

s_sub_i32 s9, s0, s8 // (op 669) 


s_cmp_eq_i32 s1, 0 // (op 669) 
s_mov_b32 s0, scc // (op 669) 

s_cmp_eq_i32 s0, 0 // (op 669) 
s_cselect_b32 s1, s9, s51 // (op 669) 



s_lshl_b32 s0, s1, 4 // (op 669) 

s_mov_b32 s9, 0 // (op 669) convert
s_mov_b32 s8, s0 // (op 669) convert

v_mul_lo_u32 v15, v13, s23 // (op 669) most significant: low of low * high
v_mul_hi_u32 v20, v13, s22 // (op 669) most significant: high of low * low
v_mul_lo_u32 v14, v13, s22 // (op 669) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 669) most significant: sum

v_mov_b32 v20, s8 // (op 669) 
v_mov_b32 v21, s9 // (op 669) 
v_add_co_u32 v14, s[0:1], v14, v20 // (op 669) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 669) most significant half





v_mul_lo_u32 v21, v15, 6 // (op 669) most significant: low of high * low
v_mul_hi_u32 v13, v14, 6 // (op 669) most significant: high of low * low
v_mul_lo_u32 v20, v14, 6 // (op 669) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 669) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 669) 

v_mov_b32 v13, v14 // (op 669) convert

s_mul_i32 s1, s27, 6 // (op 671) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 671) least significant
s_mul_hi_u32 s51, s26, 6 // (op 671) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 671) most significant: sum

s_mul_i32 s9, s22, s1 // (op 671) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 671) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 671) least significant
s_mul_hi_u32 s18, s22, s0 // (op 671) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 671) most significant: sum
s_add_u32 s9, s9, s18 // (op 671) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 671) 

s_mov_b32 s56, s24 // (op 671) reg expression
s_mov_b32 s57, s25 // (op 671) reg expression
s_mov_b32 s58, s0 // (op 671) reg expression
s_mov_b32 s59, 131072 // (op 671) call()

s_mul_hi_i32 s51, s16, s11 // (op 714) 
v_lshrrev_b32 v14, 3, v1 // (op 714) 
s_add_i32 s0, s51, s16 // (op 714) 

v_lshlrev_b32 v15, 1, v14 // (op 714) 

s_ashr_i32 s1, s0, 31 // (op 714) 
v_and_b32 v14, -8, v1 // (op 714) 
s_cmp_eq_i32 s11, 0 // (op 714) 
s_mov_b32 s51, scc // (op 714) 
v_sub_u32 v20, v1, v14 // (op 714) 

s_cmp_eq_i32 s51, 0 // (op 714) 
s_cselect_b32 s8, 0, -1 // (op 714) 

s_and_b32 s9, s12, 31 // (op 714) 
s_lshl_b32 s51, 1, s9 // (op 714) 
s_add_i32 s18, s51, s8 // (op 714) 


s_and_b32 s8, s1, s18 // (op 714) 


s_add_i32 s1, s0, s8 // (op 714) 


s_ashr_i32 s0, s1, s9 // (op 714) 


s_lshl_b32 s1, s12, 24 // (op 714) 
s_ashr_i32 s51, s1, 31 // (op 714) 

s_xor_b32 s1, s0, s51 // (op 714) 

s_sub_i32 s0, s1, s51 // (op 714) 


s_mul_hi_i32 s1, s0, s13 // (op 714) 
s_add_i32 s51, s1, s0 // (op 714) 

s_ashr_i32 s1, s51, 31 // (op 714) 
s_cmp_eq_i32 s13, 0 // (op 714) 
s_mov_b32 s8, scc // (op 714) 
s_cmp_eq_i32 s8, 0 // (op 714) 
s_cselect_b32 s9, 0, -1 // (op 714) 

s_and_b32 s8, s14, 31 // (op 714) 
s_lshl_b32 s18, 1, s8 // (op 714) 
s_add_i32 s19, s18, s9 // (op 714) 


s_and_b32 s9, s1, s19 // (op 714) 


s_add_i32 s1, s51, s9 // (op 714) 


s_ashr_i32 s9, s1, s8 // (op 714) 


s_lshl_b32 s1, s14, 24 // (op 714) 
s_ashr_i32 s8, s1, 31 // (op 714) 

s_xor_b32 s1, s9, s8 // (op 714) 

s_sub_i32 s9, s1, s8 // (op 714) 


s_mul_i32 s1, s0, s46 // (op 714) 

s_sub_u32 s0, s16, s1 // (op 714) 

s_mul_i32 s1, s9, s46 // (op 714) 
s_add_u32 s1, s1, s0 // (op 714) 


s_mul_hi_i32 s0, s16, s10 // (op 714) 
s_add_i32 s51, s0, s16 // (op 714) 

s_ashr_i32 s0, s51, 31 // (op 714) 
s_cmp_eq_i32 s10, 0 // (op 714) 
s_mov_b32 s8, scc // (op 714) 
s_cmp_eq_i32 s8, 0 // (op 714) 
s_cselect_b32 s9, 0, -1 // (op 714) 

s_and_b32 s8, s49, 31 // (op 714) 
s_lshl_b32 s18, 1, s8 // (op 714) 
s_add_i32 s19, s18, s9 // (op 714) 


s_and_b32 s9, s0, s19 // (op 714) 


s_add_i32 s0, s51, s9 // (op 714) 


s_ashr_i32 s9, s0, s8 // (op 714) 


s_lshl_b32 s0, s49, 24 // (op 714) 
s_ashr_i32 s8, s0, 31 // (op 714) 

s_xor_b32 s0, s9, s8 // (op 714) 

s_sub_i32 s9, s0, s8 // (op 714) 


s_add_u32 s18, s2, 16 // (op 714) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 714) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 714) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 714) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 714) 

s_mul_i32 s0, s46, s18 // (op 714) 

s_mul_i32 s8, s50, s0 // (op 714) 

s_mul_i32 s0, s9, s8 // (op 714) 

s_sub_u32 s8, s16, s0 // (op 714) 

s_mov_b32 s0, s8 // (op 714) 
s_mul_hi_i32 s51, s0, s15 // (op 714) 
s_add_i32 s18, s51, s0 // (op 714) 


s_ashr_i32 s0, s18, 31 // (op 714) 
s_cmp_eq_i32 s15, 0 // (op 714) 
s_mov_b32 s19, scc // (op 714) 
s_cmp_eq_i32 s19, 0 // (op 714) 
s_cselect_b32 s51, 0, -1 // (op 714) 

s_and_b32 s19, s48, 31 // (op 714) 
s_lshl_b32 s60, 1, s19 // (op 714) 
s_add_i32 s61, s60, s51 // (op 714) 


s_and_b32 s51, s0, s61 // (op 714) 


s_add_i32 s0, s18, s51 // (op 714) 


s_ashr_i32 s18, s0, s19 // (op 714) 


s_lshl_b32 s0, s48, 24 // (op 714) 
s_ashr_i32 s19, s0, 31 // (op 714) 

s_xor_b32 s0, s18, s19 // (op 714) 

s_sub_i32 s18, s0, s19 // (op 714) 


s_mul_i32 s0, s47, s46 // (op 714) 
s_add_u32 s60, s6, 16 // (op 714) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 714) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 714) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 714) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 714) 

s_sub_i32 s19, s60, s0 // (op 714) 

s_mul_i32 s51, s18, s19 // (op 714) 


s_sub_u32 s18, s8, s51 // (op 714) 


s_add_u32 s8, s18, s0 // (op 714) 


s_cmp_eq_i32 s9, 0 // (op 714) 
s_mov_b32 s0, scc // (op 714) 

s_cmp_eq_i32 s0, 0 // (op 714) 
s_cselect_b32 s9, s8, s1 // (op 714) 



v_lshl_add_u32 v14, s9, 4, v15 // (op 714) 


v_mul_lo_u32 v15, v14, s28 // (op 714) 
v_add_u32 v15, v15, v20 // (op 714) 


s_mul_i32 s1, s28, s7 // (op 716) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 716) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 716) least significant
s_mul_hi_u32 s8, s28, s6 // (op 716) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 716) most significant: sum
s_add_u32 s1, s1, s8 // (op 716) most significant: sum


s_mov_b32 s60, s30 // (op 716) reg expression
s_mov_b32 s61, s31 // (op 716) reg expression
s_mov_b32 s62, s0 // (op 716) reg expression
s_mov_b32 s63, 131072 // (op 716) call()

s_mul_hi_i32 s51, s16, s11 // (op 733) 
v_lshrrev_b32 v14, 3, v1 // (op 733) 
s_add_i32 s0, s51, s16 // (op 733) 

v_lshlrev_b32 v20, 1, v14 // (op 733) 

s_ashr_i32 s1, s0, 31 // (op 733) 
v_and_b32 v14, -8, v1 // (op 733) 
s_cmp_eq_i32 s11, 0 // (op 733) 
s_mov_b32 s51, scc // (op 733) 
v_sub_u32 v21, v1, v14 // (op 733) 

s_cmp_eq_i32 s51, 0 // (op 733) 
s_cselect_b32 s8, 0, -1 // (op 733) 

s_and_b32 s9, s12, 31 // (op 733) 
s_lshl_b32 s51, 1, s9 // (op 733) 
s_add_i32 s18, s51, s8 // (op 733) 


s_and_b32 s8, s1, s18 // (op 733) 


s_add_i32 s1, s0, s8 // (op 733) 


s_ashr_i32 s0, s1, s9 // (op 733) 


s_lshl_b32 s1, s12, 24 // (op 733) 
s_ashr_i32 s51, s1, 31 // (op 733) 

s_xor_b32 s1, s0, s51 // (op 733) 

s_sub_i32 s0, s1, s51 // (op 733) 


s_mul_hi_i32 s1, s0, s13 // (op 733) 
s_add_i32 s51, s1, s0 // (op 733) 

s_ashr_i32 s1, s51, 31 // (op 733) 
s_cmp_eq_i32 s13, 0 // (op 733) 
s_mov_b32 s8, scc // (op 733) 
s_cmp_eq_i32 s8, 0 // (op 733) 
s_cselect_b32 s9, 0, -1 // (op 733) 

s_and_b32 s8, s14, 31 // (op 733) 
s_lshl_b32 s18, 1, s8 // (op 733) 
s_add_i32 s19, s18, s9 // (op 733) 


s_and_b32 s9, s1, s19 // (op 733) 


s_add_i32 s1, s51, s9 // (op 733) 


s_ashr_i32 s9, s1, s8 // (op 733) 


s_lshl_b32 s1, s14, 24 // (op 733) 
s_ashr_i32 s8, s1, 31 // (op 733) 

s_xor_b32 s1, s9, s8 // (op 733) 

s_sub_i32 s9, s1, s8 // (op 733) 


s_mul_i32 s1, s0, s46 // (op 733) 

s_sub_u32 s0, s16, s1 // (op 733) 

s_mul_i32 s1, s9, s46 // (op 733) 
s_add_u32 s1, s1, s0 // (op 733) 


s_mul_hi_i32 s0, s16, s10 // (op 733) 
s_add_i32 s51, s0, s16 // (op 733) 

s_ashr_i32 s0, s51, 31 // (op 733) 
s_cmp_eq_i32 s10, 0 // (op 733) 
s_mov_b32 s8, scc // (op 733) 
s_cmp_eq_i32 s8, 0 // (op 733) 
s_cselect_b32 s9, 0, -1 // (op 733) 

s_and_b32 s8, s49, 31 // (op 733) 
s_lshl_b32 s18, 1, s8 // (op 733) 
s_add_i32 s19, s18, s9 // (op 733) 


s_and_b32 s9, s0, s19 // (op 733) 


s_add_i32 s0, s51, s9 // (op 733) 


s_ashr_i32 s9, s0, s8 // (op 733) 


s_lshl_b32 s0, s49, 24 // (op 733) 
s_ashr_i32 s8, s0, 31 // (op 733) 

s_xor_b32 s0, s9, s8 // (op 733) 

s_sub_i32 s9, s0, s8 // (op 733) 


s_add_u32 s18, s2, 16 // (op 733) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 733) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 733) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 733) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 733) 

s_mul_i32 s0, s46, s18 // (op 733) 

s_mul_i32 s8, s50, s0 // (op 733) 

s_mul_i32 s0, s9, s8 // (op 733) 

s_sub_u32 s8, s16, s0 // (op 733) 

s_mov_b32 s0, s8 // (op 733) 
s_mul_hi_i32 s51, s0, s15 // (op 733) 
s_add_i32 s18, s51, s0 // (op 733) 


s_ashr_i32 s0, s18, 31 // (op 733) 
s_cmp_eq_i32 s15, 0 // (op 733) 
s_mov_b32 s19, scc // (op 733) 
s_cmp_eq_i32 s19, 0 // (op 733) 
s_cselect_b32 s51, 0, -1 // (op 733) 

s_and_b32 s19, s48, 31 // (op 733) 
s_lshl_b32 s64, 1, s19 // (op 733) 
s_add_i32 s65, s64, s51 // (op 733) 


s_and_b32 s51, s0, s65 // (op 733) 


s_add_i32 s0, s18, s51 // (op 733) 


s_ashr_i32 s18, s0, s19 // (op 733) 


s_lshl_b32 s0, s48, 24 // (op 733) 
s_ashr_i32 s19, s0, 31 // (op 733) 

s_xor_b32 s0, s18, s19 // (op 733) 

s_sub_i32 s18, s0, s19 // (op 733) 


s_mul_i32 s0, s47, s46 // (op 733) 
s_add_u32 s64, s6, 16 // (op 733) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 733) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 733) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 733) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 733) 

s_sub_i32 s19, s64, s0 // (op 733) 

s_mul_i32 s51, s18, s19 // (op 733) 


s_sub_u32 s18, s8, s51 // (op 733) 


s_add_u32 s8, s18, s0 // (op 733) 


s_cmp_eq_i32 s9, 0 // (op 733) 
s_mov_b32 s0, scc // (op 733) 

s_cmp_eq_i32 s0, 0 // (op 733) 
s_cselect_b32 s9, s8, s1 // (op 733) 



v_lshl_add_u32 v14, s9, 4, v20 // (op 733) 


v_mul_lo_u32 v20, v14, s28 // (op 733) 
v_add_u32 v20, v20, v21 // (op 733) 


s_mul_i32 s1, s28, s7 // (op 735) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 735) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 735) least significant
s_mul_hi_u32 s8, s28, s6 // (op 735) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 735) most significant: sum
s_add_u32 s1, s1, s8 // (op 735) most significant: sum


s_mov_b32 s60, s30 // (op 735) reg expression
s_mov_b32 s61, s31 // (op 735) reg expression
s_mov_b32 s62, s0 // (op 735) reg expression
s_mov_b32 s63, 131072 // (op 735) call()

s_mul_hi_i32 s51, s16, s11 // (op 778) 
v_and_b32 v14, -16, v1 // (op 778) 
s_add_i32 s0, s51, s16 // (op 778) 

v_sub_u32 v21, v1, v14 // (op 778) 

s_ashr_i32 s1, s0, 31 // (op 778) 
v_lshrrev_b32 v14, 4, v1 // (op 778) 
s_cmp_eq_i32 s11, 0 // (op 778) 
s_mov_b32 s51, scc // (op 778) 
v_lshlrev_b32 v22, 1, v14 // (op 778) 

s_cmp_eq_i32 s51, 0 // (op 778) 
s_cselect_b32 s8, 0, -1 // (op 778) 

s_and_b32 s9, s12, 31 // (op 778) 
s_lshl_b32 s51, 1, s9 // (op 778) 
s_add_i32 s18, s51, s8 // (op 778) 


s_and_b32 s8, s1, s18 // (op 778) 


s_add_i32 s1, s0, s8 // (op 778) 


s_ashr_i32 s0, s1, s9 // (op 778) 


s_lshl_b32 s1, s12, 24 // (op 778) 
s_ashr_i32 s51, s1, 31 // (op 778) 

s_xor_b32 s1, s0, s51 // (op 778) 

s_sub_i32 s0, s1, s51 // (op 778) 


s_mul_hi_i32 s1, s0, s13 // (op 778) 
s_add_i32 s51, s1, s0 // (op 778) 

s_ashr_i32 s1, s51, 31 // (op 778) 
s_cmp_eq_i32 s13, 0 // (op 778) 
s_mov_b32 s8, scc // (op 778) 
s_cmp_eq_i32 s8, 0 // (op 778) 
s_cselect_b32 s9, 0, -1 // (op 778) 

s_and_b32 s8, s14, 31 // (op 778) 
s_lshl_b32 s18, 1, s8 // (op 778) 
s_add_i32 s19, s18, s9 // (op 778) 


s_and_b32 s9, s1, s19 // (op 778) 


s_add_i32 s1, s51, s9 // (op 778) 


s_ashr_i32 s9, s1, s8 // (op 778) 


s_lshl_b32 s1, s14, 24 // (op 778) 
s_ashr_i32 s8, s1, 31 // (op 778) 

s_xor_b32 s1, s9, s8 // (op 778) 

s_sub_i32 s9, s1, s8 // (op 778) 


s_add_u32 s18, s2, 16 // (op 778) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 778) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 778) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 778) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 778) 

s_mul_i32 s1, s9, s18 // (op 778) 

s_sub_i32 s51, s0, s1 // (op 778) 


s_mul_hi_i32 s0, s16, s10 // (op 778) 
s_add_i32 s1, s0, s16 // (op 778) 

s_ashr_i32 s0, s1, 31 // (op 778) 
s_cmp_eq_i32 s10, 0 // (op 778) 
s_mov_b32 s8, scc // (op 778) 
s_cmp_eq_i32 s8, 0 // (op 778) 
s_cselect_b32 s9, 0, -1 // (op 778) 

s_and_b32 s8, s49, 31 // (op 778) 
s_lshl_b32 s64, 1, s8 // (op 778) 
s_add_i32 s65, s64, s9 // (op 778) 


s_and_b32 s9, s0, s65 // (op 778) 


s_add_i32 s0, s1, s9 // (op 778) 


s_ashr_i32 s1, s0, s8 // (op 778) 


s_lshl_b32 s0, s49, 24 // (op 778) 
s_ashr_i32 s8, s0, 31 // (op 778) 

s_xor_b32 s0, s1, s8 // (op 778) 

s_sub_i32 s1, s0, s8 // (op 778) 


s_mul_i32 s0, s46, s18 // (op 778) 

s_mul_i32 s8, s50, s0 // (op 778) 

s_mul_i32 s0, s1, s8 // (op 778) 

s_sub_u32 s8, s16, s0 // (op 778) 

s_mul_hi_i32 s0, s8, s15 // (op 778) 
s_add_i32 s9, s0, s8 // (op 778) 


s_ashr_i32 s0, s9, 31 // (op 778) 
s_cmp_eq_i32 s15, 0 // (op 778) 
s_mov_b32 s8, scc // (op 778) 
s_cmp_eq_i32 s8, 0 // (op 778) 
s_cselect_b32 s18, 0, -1 // (op 778) 

s_and_b32 s8, s48, 31 // (op 778) 
s_lshl_b32 s19, 1, s8 // (op 778) 
s_add_i32 s64, s19, s18 // (op 778) 


s_and_b32 s18, s0, s64 // (op 778) 


s_add_i32 s0, s9, s18 // (op 778) 


s_ashr_i32 s9, s0, s8 // (op 778) 


s_lshl_b32 s0, s48, 24 // (op 778) 
s_ashr_i32 s8, s0, 31 // (op 778) 

s_xor_b32 s0, s9, s8 // (op 778) 

s_sub_i32 s9, s0, s8 // (op 778) 


s_cmp_eq_i32 s1, 0 // (op 778) 
s_mov_b32 s0, scc // (op 778) 

s_cmp_eq_i32 s0, 0 // (op 778) 
s_cselect_b32 s1, s9, s51 // (op 778) 



v_lshl_add_u32 v14, s1, 4, v21 // (op 778) 


v_mul_lo_u32 v21, v22, s32 // (op 778) 
v_add_u32 v21, v21, v14 // (op 778) 


s_mul_i32 s1, s32, s37 // (op 780) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 780) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 780) least significant
s_mul_hi_u32 s8, s32, s36 // (op 780) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 780) most significant: sum
s_add_u32 s1, s1, s8 // (op 780) most significant: sum


s_mov_b32 s64, s34 // (op 780) reg expression
s_mov_b32 s65, s35 // (op 780) reg expression
s_mov_b32 s66, s0 // (op 780) reg expression
s_mov_b32 s67, 131072 // (op 780) call()

s_mul_hi_i32 s51, s16, s11 // (op 797) 
v_and_b32 v14, -16, v1 // (op 797) 
s_add_i32 s0, s51, s16 // (op 797) 

v_sub_u32 v22, v1, v14 // (op 797) 

s_ashr_i32 s1, s0, 31 // (op 797) 
v_lshrrev_b32 v14, 4, v1 // (op 797) 
s_cmp_eq_i32 s11, 0 // (op 797) 
s_mov_b32 s51, scc // (op 797) 
v_lshlrev_b32 v23, 1, v14 // (op 797) 

s_cmp_eq_i32 s51, 0 // (op 797) 
s_cselect_b32 s8, 0, -1 // (op 797) 

s_and_b32 s9, s12, 31 // (op 797) 
s_lshl_b32 s51, 1, s9 // (op 797) 
s_add_i32 s18, s51, s8 // (op 797) 


s_and_b32 s8, s1, s18 // (op 797) 


s_add_i32 s1, s0, s8 // (op 797) 


s_ashr_i32 s0, s1, s9 // (op 797) 


s_lshl_b32 s1, s12, 24 // (op 797) 
s_ashr_i32 s51, s1, 31 // (op 797) 

s_xor_b32 s1, s0, s51 // (op 797) 

s_sub_i32 s0, s1, s51 // (op 797) 


s_mul_hi_i32 s1, s0, s13 // (op 797) 
s_add_i32 s51, s1, s0 // (op 797) 

s_ashr_i32 s1, s51, 31 // (op 797) 
s_cmp_eq_i32 s13, 0 // (op 797) 
s_mov_b32 s8, scc // (op 797) 
s_cmp_eq_i32 s8, 0 // (op 797) 
s_cselect_b32 s9, 0, -1 // (op 797) 

s_and_b32 s8, s14, 31 // (op 797) 
s_lshl_b32 s18, 1, s8 // (op 797) 
s_add_i32 s19, s18, s9 // (op 797) 


s_and_b32 s9, s1, s19 // (op 797) 


s_add_i32 s1, s51, s9 // (op 797) 


s_ashr_i32 s9, s1, s8 // (op 797) 


s_lshl_b32 s1, s14, 24 // (op 797) 
s_ashr_i32 s8, s1, 31 // (op 797) 

s_xor_b32 s1, s9, s8 // (op 797) 

s_sub_i32 s9, s1, s8 // (op 797) 


s_add_u32 s18, s2, 16 // (op 797) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 797) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 797) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 797) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 797) 

s_mul_i32 s1, s9, s18 // (op 797) 

s_sub_i32 s51, s0, s1 // (op 797) 


s_mul_hi_i32 s0, s16, s10 // (op 797) 
s_add_i32 s1, s0, s16 // (op 797) 

s_ashr_i32 s0, s1, 31 // (op 797) 
s_cmp_eq_i32 s10, 0 // (op 797) 
s_mov_b32 s8, scc // (op 797) 
s_cmp_eq_i32 s8, 0 // (op 797) 
s_cselect_b32 s9, 0, -1 // (op 797) 

s_and_b32 s8, s49, 31 // (op 797) 
s_lshl_b32 s68, 1, s8 // (op 797) 
s_add_i32 s69, s68, s9 // (op 797) 


s_and_b32 s9, s0, s69 // (op 797) 


s_add_i32 s0, s1, s9 // (op 797) 


s_ashr_i32 s1, s0, s8 // (op 797) 


s_lshl_b32 s0, s49, 24 // (op 797) 
s_ashr_i32 s8, s0, 31 // (op 797) 

s_xor_b32 s0, s1, s8 // (op 797) 

s_sub_i32 s1, s0, s8 // (op 797) 


s_mul_i32 s0, s46, s18 // (op 797) 

s_mul_i32 s8, s50, s0 // (op 797) 

s_mul_i32 s0, s1, s8 // (op 797) 

s_sub_u32 s8, s16, s0 // (op 797) 

s_mul_hi_i32 s0, s8, s15 // (op 797) 
s_add_i32 s9, s0, s8 // (op 797) 


s_ashr_i32 s0, s9, 31 // (op 797) 
s_cmp_eq_i32 s15, 0 // (op 797) 
s_mov_b32 s8, scc // (op 797) 
s_cmp_eq_i32 s8, 0 // (op 797) 
s_cselect_b32 s18, 0, -1 // (op 797) 

s_and_b32 s8, s48, 31 // (op 797) 
s_lshl_b32 s19, 1, s8 // (op 797) 
s_add_i32 s68, s19, s18 // (op 797) 


s_and_b32 s18, s0, s68 // (op 797) 


s_add_i32 s0, s9, s18 // (op 797) 


s_ashr_i32 s9, s0, s8 // (op 797) 


s_lshl_b32 s0, s48, 24 // (op 797) 
s_ashr_i32 s8, s0, 31 // (op 797) 

s_xor_b32 s0, s9, s8 // (op 797) 

s_sub_i32 s9, s0, s8 // (op 797) 


s_cmp_eq_i32 s1, 0 // (op 797) 
s_mov_b32 s0, scc // (op 797) 

s_cmp_eq_i32 s0, 0 // (op 797) 
s_cselect_b32 s1, s9, s51 // (op 797) 



v_lshl_add_u32 v14, s1, 4, v22 // (op 797) 


v_mul_lo_u32 v22, v23, s32 // (op 797) 
v_add_u32 v22, v22, v14 // (op 797) 


s_mul_i32 s1, s32, s37 // (op 799) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 799) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 799) least significant
s_mul_hi_u32 s8, s32, s36 // (op 799) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 799) most significant: sum
s_add_u32 s1, s1, s8 // (op 799) most significant: sum


s_mov_b32 s64, s34 // (op 799) reg expression
s_mov_b32 s65, s35 // (op 799) reg expression
s_mov_b32 s66, s0 // (op 799) reg expression
s_mov_b32 s67, 131072 // (op 799) call()


s_mul_i32 s1, s23, 6 // (op 428) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 428) least significant
s_mul_hi_u32 s51, s22, 6 // (op 428) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 428) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 428) 


v_mov_b32 v14, v12 // (op 428) reg expression
buffer_load_dwordx3 v[26:28], v14, s[56:59], 0 offen // (op 428) Load value
v_add_u32 v14, v14, v24 // (op 428) 
buffer_load_dwordx3 v[30:32], v14, s[56:59], 0 offen // (op 428) Load value
v_add_u32 v14, v14, v24 // (op 428) 
buffer_load_dwordx3 v[34:36], v14, s[56:59], 0 offen // (op 428) Load value
v_add_u32 v14, v14, v24 // (op 428) 
buffer_load_dwordx3 v[38:40], v14, s[56:59], 0 offen // (op 428) Load value



v_mov_b64 v[24:25], s[32:33] // (op 431) convert

v_mov_b32 v14, v21 // (op 431) reg expression
buffer_load_ubyte v42, v14, s[64:67], 0 offen offset: 0 // (op 431) Load value
v_add_u32 v14, v14, v24 // (op 431) 
buffer_load_ubyte v43, v14, s[64:67], 0 offen offset: 0 // (op 431) Load value


v_and_b32 v14, -16, v1 // (op 863) 
v_sub_u32 v29, v1, v14 // (op 863) 

v_lshlrev_b32 v14, 4, v29 // (op 863) 

v_lshrrev_b32 v29, 4, v1 // (op 863) 
v_lshl_add_u32 v33, v29, 8, v14 // (op 863) 


s_nop 0
v_readfirstlane_b32 s51, v33 // (op 863) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[4:5], 2 // (op 542) 
v_mov_b64 v[24:25], s[0:1] // (op 542) convert


v_mov_b64 v[44:45], v[16:17] // (op 542) 
s_mov_b32 m0, s51 // (op 542) reg expression
buffer_load_dwordx4 v44, s[52:55], 0 offen offset: 0 lds // (op 542) Load value direct to lds
v_add_co_u32 v44, s[0:1], v44, v24 // (op 542) least significant half
v_addc_co_u32 v45, s[0:1], 0, v45, s[0:1] // (op 542) most significant half

s_add_u32 m0, m0, 1024 // (op 542) 
buffer_load_dwordx4 v44, s[52:55], 0 offen offset: 0 lds // (op 542) Load value direct to lds
v_add_co_u32 v44, s[0:1], v44, v24 // (op 542) least significant half
v_addc_co_u32 v45, s[0:1], 0, v45, s[0:1] // (op 542) most significant half

s_add_u32 m0, m0, 1024 // (op 542) 
buffer_load_dwordx4 v44, s[52:55], 0 offen offset: 0 lds // (op 542) Load value direct to lds
v_add_co_u32 v44, s[0:1], v44, v24 // (op 542) least significant half
v_addc_co_u32 v45, s[0:1], 0, v45, s[0:1] // (op 542) most significant half

s_add_u32 m0, m0, 1024 // (op 542) 
buffer_load_dwordx4 v44, s[52:55], 0 offen offset: 0 lds // (op 542) Load value direct to lds




v_mov_b64 v[24:25], s[28:29] // (op 437) convert

v_mov_b32 v14, v15 // (op 437) 
buffer_load_ubyte v44, v14, s[60:63], 0 offen offset: 0 // (op 437) Load value
v_add_u32 v14, v14, v24 // (op 437) 
buffer_load_ubyte v45, v14, s[60:63], 0 offen offset: 0 // (op 437) Load value



s_mul_i32 s1, s23, 6 // (op 440) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 440) least significant
s_mul_hi_u32 s51, s22, 6 // (op 440) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 440) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 440) 


s_lshl_b64 s[0:1], s[22:23], 8 // (op 440) 
s_mul_i32 s9, s1, 6 // (op 440) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 440) least significant
s_mul_hi_u32 s51, s0, 6 // (op 440) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 440) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 440) 

v_add_u32 v14, v12, s0 // (op 440) 

buffer_load_dwordx3 v[46:48], v14, s[56:59], 0 offen // (op 440) Load value
v_add_u32 v14, v14, v24 // (op 440) 
buffer_load_dwordx3 v[50:52], v14, s[56:59], 0 offen // (op 440) Load value
v_add_u32 v14, v14, v24 // (op 440) 
buffer_load_dwordx3 v[54:56], v14, s[56:59], 0 offen // (op 440) Load value
v_add_u32 v14, v14, v24 // (op 440) 
buffer_load_dwordx3 v[58:60], v14, s[56:59], 0 offen // (op 440) Load value




v_mov_b64 v[24:25], s[32:33] // (op 443) convert

s_lshl_b32 s51, s32, 3 // (op 443) 
v_add_u32 v12, v21, s51 // (op 443) 

buffer_load_ubyte v62, v12, s[64:67], 0 offen offset: 0 // (op 443) Load value
v_add_u32 v12, v12, v24 // (op 443) 
buffer_load_ubyte v63, v12, s[64:67], 0 offen offset: 0 // (op 443) Load value



v_and_b32 v12, -16, v1 // (op 1095) 
v_sub_u32 v14, v1, v12 // (op 1095) 

v_lshlrev_b32 v12, 4, v14 // (op 1095) 

v_lshrrev_b32 v14, 4, v1 // (op 1095) 
v_lshl_add_u32 v21, v14, 8, v12 // (op 1095) 


s_nop 0
v_readfirstlane_b32 s51, v21 // (op 1095) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[4:5], 2 // (op 547) 
v_mov_b64 v[24:25], s[0:1] // (op 547) convert


v_mov_b64 v[64:65], v[16:17] // (op 547) 
s_add_u32 s0, 4096, s51 // (op 547) 
s_mov_b32 m0, s0 // (op 547) reg expression
s_mov_b32 s1, 256 // (op 547) call()
buffer_load_dwordx4 v64, s[52:55], s1 offen offset: 0 lds // (op 547) Load value direct to lds
v_add_co_u32 v64, s[8:9], v64, v24 // (op 547) least significant half
v_addc_co_u32 v65, s[8:9], 0, v65, s[8:9] // (op 547) most significant half

s_add_u32 m0, m0, 1024 // (op 547) 
buffer_load_dwordx4 v64, s[52:55], s1 offen offset: 0 lds // (op 547) Load value direct to lds
v_add_co_u32 v64, s[8:9], v64, v24 // (op 547) least significant half
v_addc_co_u32 v65, s[8:9], 0, v65, s[8:9] // (op 547) most significant half

s_add_u32 m0, m0, 1024 // (op 547) 
buffer_load_dwordx4 v64, s[52:55], s1 offen offset: 0 lds // (op 547) Load value direct to lds
v_add_co_u32 v64, s[8:9], v64, v24 // (op 547) least significant half
v_addc_co_u32 v65, s[8:9], 0, v65, s[8:9] // (op 547) most significant half

s_add_u32 m0, m0, 1024 // (op 547) 
buffer_load_dwordx4 v64, s[52:55], s1 offen offset: 0 lds // (op 547) Load value direct to lds







v_mov_b64 v[16:17], s[28:29] // (op 449) convert

v_mov_b32 v12, v15 // (op 449) 
buffer_load_ubyte v24, v12, s[60:63], 0 offen offset:8 // (op 449) Load value
v_add_u32 v12, v12, v16 // (op 449) 
buffer_load_ubyte v25, v12, s[60:63], 0 offen offset:8 // (op 449) Load value



v_lshlrev_b32 v12, 6, v1 // (op 916) 
v_mul_lo_u32 v21, v12, 6 // (op 916) 
v_lshrrev_b32 v23, 3, v21 // (op 916) 

v_lshrrev_b32 v21, 4, v12 // (op 916) 

v_lshlrev_b32 v12, 2, v21 // (op 916) 

v_add_u32 v21, v23, v12 // (op 916) 





s_waitcnt vmcnt(23)
ds_write_b96 v21, v[26:28] offset:8192 // (op 451) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v21, v[30:32] offset:8208 // (op 451) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v21, v[34:36] offset:8224 // (op 451) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v21, v[38:40] offset:8240 // (op 451) Store local data 

v_and_b32 v12, -16, v1 // (op 1022) 
v_sub_u32 v21, v1, v12 // (op 1022) 

v_lshrrev_b32 v12, 4, v1 // (op 1022) 
v_lshl_add_u32 v23, v12, 5, v21 // (op 1022) 





s_waitcnt vmcnt(19)
ds_write_b8 v23, v42 offset:12288 // (op 452) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v23, v43 offset:12304 // (op 452) Store local data 

v_and_b32 v12, -8, v1 // (op 969) 
v_sub_u32 v21, v1, v12 // (op 969) 

v_lshrrev_b32 v12, 3, v1 // (op 969) 
v_lshl_add_u32 v23, v12, 4, v21 // (op 969) 





s_waitcnt vmcnt(13)
ds_write_b8 v23, v44 offset:12416 // (op 454) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v23, v45 offset:12424 // (op 454) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1235) 



ds_read_b96_tr_b6 v[64:66], v5 offset:8192 // (op 471) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v5 offset:8448 // (op 471) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1276) 


ds_read_u8 v12, v7 offset:12288 // (op 477) Load local data 
s_barrier  // (op 1216) 



ds_read_b128 v[70:73], v3 // (op 468) Load local data 
ds_read_b128 v[74:77], v3 offset:64 // (op 468) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1256) 



ds_read_u8 v21, v0 offset:12416 // (op 474) Load local data 
v_and_b32 v23, -16, v1 // (op 835) 
v_sub_u32 v29, v1, v23 // (op 835) 

v_lshlrev_b32 v23, 4, v29 // (op 835) 

v_lshrrev_b32 v29, 4, v1 // (op 835) 
v_lshl_add_u32 v33, v29, 8, v23 // (op 835) 


s_nop 0
v_readfirstlane_b32 s51, v33 // (op 835) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v23, 6, v1 // (op 887) 
v_mul_lo_u32 v29, v23, 6 // (op 887) 
v_lshrrev_b32 v33, 3, v29 // (op 887) 

v_lshrrev_b32 v29, 4, v23 // (op 887) 

v_lshlrev_b32 v23, 2, v29 // (op 887) 

v_add_u32 v29, v33, v23 // (op 887) 


v_and_b32 v23, -8, v1 // (op 940) 
v_sub_u32 v33, v1, v23 // (op 940) 

v_lshrrev_b32 v23, 3, v1 // (op 940) 
v_lshl_add_u32 v37, v23, 4, v33 // (op 940) 


v_and_b32 v23, -16, v1 // (op 993) 
v_sub_u32 v33, v1, v23 // (op 993) 

v_lshrrev_b32 v23, 4, v1 // (op 993) 
v_lshl_add_u32 v41, v23, 5, v33 // (op 993) 


v_and_b32 v23, -16, v1 // (op 1080) 
v_sub_u32 v33, v1, v23 // (op 1080) 

v_lshlrev_b32 v23, 4, v33 // (op 1080) 

v_lshrrev_b32 v33, 4, v1 // (op 1080) 
v_lshl_add_u32 v49, v33, 8, v23 // (op 1080) 


s_nop 0
v_readfirstlane_b32 s0, v49 // (op 1080) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v23, 6, v1 // (op 1119) 
v_mul_lo_u32 v33, v23, 6 // (op 1119) 
v_lshrrev_b32 v49, 3, v33 // (op 1119) 

v_lshrrev_b32 v33, 4, v23 // (op 1119) 

v_lshlrev_b32 v23, 2, v33 // (op 1119) 

v_add_u32 v33, v49, v23 // (op 1119) 


v_and_b32 v23, -8, v1 // (op 1143) 
v_sub_u32 v49, v1, v23 // (op 1143) 

v_lshrrev_b32 v23, 3, v1 // (op 1143) 
v_lshl_add_u32 v53, v23, 4, v49 // (op 1143) 


v_and_b32 v23, -16, v1 // (op 1167) 
v_sub_u32 v49, v1, v23 // (op 1167) 

v_lshrrev_b32 v23, 4, v1 // (op 1167) 
v_lshl_add_u32 v57, v23, 5, v49 // (op 1167) 


s_mov_b32 s1, 0 // (op 61) call()
s_ashr_i64 s[8:9], s[26:27], 8 // (op 60) 
s_mov_b32 s18, s8 // (op 60) convert

s_and_b32 s19, s18, 4294967294 // (op 60) 

s_cmp_lt_u32 s1, s19 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_60 if false)
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_60:

s_add_u32 s1, s1, 2 // (op 62) 
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
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[64:69], a[0:3], v21, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 90) 
buffer_load_dwordx3 v[26:28], v49, s[56:59], 0 offen // (op 10) Load value
s_barrier  // (op 1225) 
v_add_u32 v49, v49, v14 // (op 10) 



ds_read_b128 v[70:73], v3 offset:128 // (op 93) Load local data 
ds_read_b128 v[74:77], v3 offset:192 // (op 93) Load local data 
buffer_load_dwordx3 v[30:32], v49, s[56:59], 0 offen // (op 10) Load value
s_barrier  // (op 1245) 
v_add_u32 v49, v49, v14 // (op 10) 



ds_read_b96_tr_b6 v[64:66], v5 offset:10240 // (op 97) Transpose load local data 
buffer_load_dwordx3 v[34:36], v49, s[56:59], 0 offen // (op 10) Load value
v_add_u32 v49, v49, v14 // (op 10) 
ds_read_b96_tr_b6 v[67:69], v5 offset:10496 // (op 97) Transpose load local data 
buffer_load_dwordx3 v[38:40], v49, s[56:59], 0 offen // (op 10) Load value



s_barrier  // (op 1266) 
v_mov_b64 v[14:15], s[32:33] // (op 25) convert




ds_read_u8 v21, v0 offset:12420 // (op 101) Load local data 
v_mov_b32 v49, v22 // (op 25) 
s_barrier  // (op 1286) 
s_lshl_b32 s8, s32, 4 // (op 25) 


ds_read_u8 v12, v7 offset:12352 // (op 105) Load local data 
v_add_u32 v61, v49, s8 // (op 25) 

buffer_load_ubyte v42, v61, s[64:67], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v61, v61, v14 // (op 25) 
buffer_load_ubyte v43, v61, s[64:67], 0 offen offset: 0 // (op 25) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[64:69], a[0:3], v21, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 107) 
s_barrier  // (op 1219) 

s_lshl_b64 s[8:9], s[4:5], 2 // (op 526) 
v_mov_b64 v[14:15], s[8:9] // (op 526) convert


v_mov_b64 v[16:17], v[18:19] // (op 526) 
v_mov_b64 v[78:79], v[16:17] // (op 526) 
s_mov_b32 m0, s51 // (op 526) reg expression
s_mov_b32 s8, 512 // (op 526) call()
buffer_load_dwordx4 v78, s[52:55], s8 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 526) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
buffer_load_dwordx4 v78, s[52:55], s8 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 526) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
buffer_load_dwordx4 v78, s[52:55], s8 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 526) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
buffer_load_dwordx4 v78, s[52:55], s8 offen offset: 0 lds // (op 526) Load value direct to lds




v_mov_b64 v[14:15], s[28:29] // (op 17) convert

v_mov_b32 v61, v20 // (op 17) 
v_mov_b32 v78, v61 // (op 17) 
buffer_load_ubyte v44, v78, s[60:63], 0 offen offset:16 // (op 17) Load value
v_add_u32 v78, v78, v14 // (op 17) 
buffer_load_ubyte v45, v78, s[60:63], 0 offen offset:16 // (op 17) Load value


s_barrier  // (op 1306) 



s_waitcnt vmcnt(23)
ds_write_b96 v33, v[46:48] offset:12544 // (op 306) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v33, v[50:52] offset:12560 // (op 306) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v33, v[54:56] offset:12576 // (op 306) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v33, v[58:60] offset:12592 // (op 306) Store local data 
s_barrier  // (op 1318) 



s_waitcnt vmcnt(19)
ds_write_b8 v57, v62 offset:16640 // (op 308) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v57, v63 offset:16656 // (op 308) Store local data 
s_barrier  // (op 1312) 



s_waitcnt vmcnt(13)
ds_write_b8 v53, v24 offset:16768 // (op 307) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v53, v25 offset:16776 // (op 307) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1302) 



ds_read_b96_tr_b6 v[78:80], v9 offset:12544 // (op 291) Transpose load local data 
ds_read_b96_tr_b6 v[81:83], v9 offset:12800 // (op 291) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1315) 


ds_read_u8 v14, v11 offset:16640 // (op 295) Load local data 
s_barrier  // (op 1296) 



ds_read_b128 v[84:87], v6 offset:4096 // (op 289) Load local data 
ds_read_b128 v[88:91], v6 offset:4160 // (op 289) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1309) 



ds_read_u8 v15, v4 offset:16768 // (op 293) Load local data 

s_mul_i32 s9, s23, 6 // (op 309) most significant: low of high * low
s_mul_i32 s8, s22, 6 // (op 309) least significant
s_mul_hi_u32 s18, s22, 6 // (op 309) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 309) most significant: sum

v_lshrrev_b64 v[92:93], 3, s[8:9] // (op 309) 


s_lshl_b64 s[8:9], s[22:23], 8 // (op 309) 
s_mul_i32 s19, s9, 6 // (op 309) most significant: low of high * low
s_mul_i32 s18, s8, 6 // (op 309) least significant
s_mul_hi_u32 s68, s8, 6 // (op 309) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 309) most significant: sum


s_lshr_b64 s[8:9], s[18:19], 3 // (op 309) 

s_mul_i32 s18, 3, s8 // (op 309) 

v_add_u32 v94, v23, s18 // (op 309) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:91], v[78:83], a[0:3], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 296) 
buffer_load_dwordx3 v[46:48], v94, s[56:59], 0 offen // (op 309) Load value




v_add_u32 v94, v94, v92 // (op 309) 



ds_read_b128 v[78:81], v6 offset:4224 // (op 298) Load local data 
ds_read_b128 v[82:85], v6 offset:4288 // (op 298) Load local data 
buffer_load_dwordx3 v[50:52], v94, s[56:59], 0 offen // (op 309) Load value
v_add_u32 v94, v94, v92 // (op 309) 



ds_read_b96_tr_b6 v[86:88], v9 offset:14592 // (op 300) Transpose load local data 
buffer_load_dwordx3 v[54:56], v94, s[56:59], 0 offen // (op 309) Load value
v_add_u32 v94, v94, v92 // (op 309) 
ds_read_b96_tr_b6 v[89:91], v9 offset:14848 // (op 300) Transpose load local data 
buffer_load_dwordx3 v[58:60], v94, s[56:59], 0 offen // (op 309) Load value







s_barrier  // (op 1377) 
ds_read_u8 v23, v4 offset:16772 // (op 302) Load local data 
v_mov_b64 v[14:15], s[32:33] // (op 311) convert

s_lshl_b32 s8, s32, 3 // (op 311) 


s_barrier  // (op 1381) 
ds_read_u8 v92, v11 offset:16704 // (op 304) Load local data 
s_mul_i32 s9, 3, s8 // (op 311) 

v_add_u32 v93, v49, s9 // (op 311) 

s_barrier  // (op 1385) 
buffer_load_ubyte v62, v93, s[64:67], 0 offen offset: 0 // (op 311) Load value
v_add_u32 v93, v93, v14 // (op 311) 
s_lshl_b64 s[8:9], s[22:23], 8 // (op 661) 
buffer_load_ubyte v63, v93, s[64:67], 0 offen offset: 0 // (op 311) Load value


s_mul_i32 s19, s9, 6 // (op 661) most significant: low of high * low

s_mul_i32 s18, s8, 6 // (op 661) least significant
s_mul_hi_u32 s68, s8, 6 // (op 661) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 661) most significant: sum


s_lshl_b32 s8, s32, 4 // (op 789) 
v_add_u32 v22, v22, s8 // (op 789) 

s_lshr_b64 s[8:9], s[18:19], 3 // (op 661) 

s_lshl_b32 s18, s8, 1 // (op 661) 

v_add_u32 v13, v13, s18 // (op 661) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:85], v[86:91], a[0:3], v23, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 305) 
s_barrier  // (op 1299) 





s_lshl_b64 s[8:9], s[4:5], 2 // (op 537) 
v_mov_b64 v[14:15], s[8:9] // (op 537) convert


v_mov_b64 v[78:79], v[16:17] // (op 537) 
s_add_u32 s8, 4096, s0 // (op 537) 
s_mov_b32 m0, s8 // (op 537) reg expression
s_mov_b32 s9, 768 // (op 537) call()
buffer_load_dwordx4 v78, s[52:55], s9 offen offset: 0 lds // (op 537) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 537) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 537) most significant half

s_add_u32 m0, m0, 1024 // (op 537) 
buffer_load_dwordx4 v78, s[52:55], s9 offen offset: 0 lds // (op 537) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 537) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 537) most significant half

s_add_u32 m0, m0, 1024 // (op 537) 
buffer_load_dwordx4 v78, s[52:55], s9 offen offset: 0 lds // (op 537) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 537) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 537) most significant half

s_add_u32 m0, m0, 1024 // (op 537) 
buffer_load_dwordx4 v78, s[52:55], s9 offen offset: 0 lds // (op 537) Load value direct to lds






v_add_co_u32 v18, vcc, 512, v18 // (op 597) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 597) most significant half
v_mov_b64 v[14:15], s[28:29] // (op 310) convert

v_mov_b32 v23, v61 // (op 310) 
buffer_load_ubyte v24, v23, s[60:63], 0 offen offset:24 // (op 310) Load value
v_add_u32 v23, v23, v14 // (op 310) 
buffer_load_ubyte v25, v23, s[60:63], 0 offen offset:24 // (op 310) Load value



s_barrier  // (op 1238) 
v_add_u32 v20, 16, v20 // (op 725) 



s_waitcnt vmcnt(23)
ds_write_b96 v29, v[26:28] offset:8192 // (op 13) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v29, v[30:32] offset:8208 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v29, v[34:36] offset:8224 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v29, v[38:40] offset:8240 // (op 13) Store local data 
s_barrier  // (op 1279) 



s_waitcnt vmcnt(19)
ds_write_b8 v41, v42 offset:12288 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v41, v43 offset:12304 // (op 30) Store local data 
s_barrier  // (op 1259) 



s_waitcnt vmcnt(13)
ds_write_b8 v37, v44 offset:12416 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v37, v45 offset:12424 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1248) 



ds_read_b96_tr_b6 v[64:66], v5 offset:8192 // (op 77) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v5 offset:8448 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1289) 


ds_read_u8 v12, v7 offset:12288 // (op 87) Load local data 
s_barrier  // (op 1228) 



ds_read_b128 v[70:73], v3 // (op 72) Load local data 
ds_read_b128 v[74:77], v3 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1269) 



ds_read_u8 v21, v0 offset:12416 // (op 82) Load local data 
s_ashr_i64 s[8:9], s[26:27], 8 // (op 60) 
s_mov_b32 s18, s8 // (op 60) convert

s_and_b32 s19, s18, 4294967294 // (op 60) 

s_cmp_lt_u32 s1, s19 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_60 if true)
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_60:
















s_barrier  // (op 1333) 
s_mul_hi_i32 s51, s16, s11 // (op 560) 
v_lshrrev_b32 v49, 4, v1 // (op 560) 
s_add_i32 s0, s51, s16 // (op 560) 

v_and_b32 v53, -16, v1 // (op 560) 
s_ashr_i32 s1, s0, 31 // (op 560) 
v_sub_u32 v57, v1, v53 // (op 560) 

s_cmp_eq_i32 s11, 0 // (op 560) 
s_mov_b32 s51, scc // (op 560) 
v_lshlrev_b32 v53, 4, v57 // (op 560) 

s_cmp_eq_i32 s51, 0 // (op 560) 
s_cselect_b32 s8, 0, -1 // (op 560) 

s_and_b32 s9, s12, 31 // (op 560) 
s_lshl_b32 s51, 1, s9 // (op 560) 
s_add_i32 s18, s51, s8 // (op 560) 


s_and_b32 s8, s1, s18 // (op 560) 


s_add_i32 s1, s0, s8 // (op 560) 


s_ashr_i32 s0, s1, s9 // (op 560) 


s_lshl_b32 s1, s12, 24 // (op 560) 
s_ashr_i32 s51, s1, 31 // (op 560) 

s_xor_b32 s1, s0, s51 // (op 560) 

s_sub_i32 s0, s1, s51 // (op 560) 


s_mul_hi_i32 s1, s0, s13 // (op 560) 
s_add_i32 s51, s1, s0 // (op 560) 

s_ashr_i32 s1, s51, 31 // (op 560) 
s_cmp_eq_i32 s13, 0 // (op 560) 
s_mov_b32 s8, scc // (op 560) 
s_cmp_eq_i32 s8, 0 // (op 560) 
s_cselect_b32 s9, 0, -1 // (op 560) 

s_and_b32 s8, s14, 31 // (op 560) 
s_lshl_b32 s18, 1, s8 // (op 560) 
s_add_i32 s19, s18, s9 // (op 560) 


s_and_b32 s9, s1, s19 // (op 560) 


s_add_i32 s1, s51, s9 // (op 560) 


s_ashr_i32 s9, s1, s8 // (op 560) 


s_lshl_b32 s1, s14, 24 // (op 560) 
s_ashr_i32 s8, s1, 31 // (op 560) 

s_xor_b32 s1, s9, s8 // (op 560) 

s_sub_i32 s9, s1, s8 // (op 560) 


s_mul_i32 s1, s0, s46 // (op 560) 

s_sub_u32 s0, s16, s1 // (op 560) 

s_mul_i32 s1, s9, s46 // (op 560) 
s_add_u32 s1, s1, s0 // (op 560) 


s_mul_hi_i32 s0, s16, s10 // (op 560) 
s_add_i32 s51, s0, s16 // (op 560) 

s_ashr_i32 s0, s51, 31 // (op 560) 
s_cmp_eq_i32 s10, 0 // (op 560) 
s_mov_b32 s8, scc // (op 560) 
s_cmp_eq_i32 s8, 0 // (op 560) 
s_cselect_b32 s9, 0, -1 // (op 560) 

s_and_b32 s8, s49, 31 // (op 560) 
s_lshl_b32 s18, 1, s8 // (op 560) 
s_add_i32 s19, s18, s9 // (op 560) 


s_and_b32 s9, s0, s19 // (op 560) 


s_add_i32 s0, s51, s9 // (op 560) 


s_ashr_i32 s9, s0, s8 // (op 560) 


s_lshl_b32 s0, s49, 24 // (op 560) 
s_ashr_i32 s8, s0, 31 // (op 560) 

s_xor_b32 s0, s9, s8 // (op 560) 

s_sub_i32 s9, s0, s8 // (op 560) 


s_add_u32 s18, s2, 16 // (op 560) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 560) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 560) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 560) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 560) 

s_mul_i32 s0, s46, s18 // (op 560) 

s_mul_i32 s8, s50, s0 // (op 560) 

s_mul_i32 s0, s9, s8 // (op 560) 

s_sub_u32 s8, s16, s0 // (op 560) 

s_mov_b32 s0, s8 // (op 560) 
s_mul_hi_i32 s51, s0, s15 // (op 560) 
s_add_i32 s18, s51, s0 // (op 560) 


s_ashr_i32 s0, s18, 31 // (op 560) 
s_cmp_eq_i32 s15, 0 // (op 560) 
s_mov_b32 s19, scc // (op 560) 
s_cmp_eq_i32 s19, 0 // (op 560) 
s_cselect_b32 s51, 0, -1 // (op 560) 

s_and_b32 s19, s48, 31 // (op 560) 
s_lshl_b32 s68, 1, s19 // (op 560) 
s_add_i32 s69, s68, s51 // (op 560) 


s_and_b32 s51, s0, s69 // (op 560) 


s_add_i32 s0, s18, s51 // (op 560) 


s_ashr_i32 s18, s0, s19 // (op 560) 


s_lshl_b32 s0, s48, 24 // (op 560) 
s_ashr_i32 s19, s0, 31 // (op 560) 

s_xor_b32 s0, s18, s19 // (op 560) 

s_sub_i32 s18, s0, s19 // (op 560) 


s_mul_i32 s0, s47, s46 // (op 560) 
s_add_u32 s68, s6, 16 // (op 560) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 560) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 560) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 560) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 560) 

s_sub_i32 s19, s68, s0 // (op 560) 

s_mul_i32 s51, s18, s19 // (op 560) 


s_sub_u32 s18, s8, s51 // (op 560) 


s_add_u32 s8, s18, s0 // (op 560) 


s_cmp_eq_i32 s9, 0 // (op 560) 
s_mov_b32 s0, scc // (op 560) 

s_cmp_eq_i32 s0, 0 // (op 560) 
s_cselect_b32 s9, s8, s1 // (op 560) 



v_lshl_add_u32 v57, s9, 4, v49 // (op 560) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 560) 
s_lshr_b32 s51, s0, 1 // (op 560) 

v_lshl_add_u32 v49, s51, 9, v53 // (op 560) 


v_mov_b32 v23, 0 // (op 560) convert
v_mov_b32 v22, v49 // (op 560) convert

v_mul_lo_u32 v27, v57, s5 // (op 560) most significant: low of low * high
v_mul_hi_u32 v49, v57, s4 // (op 560) most significant: high of low * low
v_mul_lo_u32 v26, v57, s4 // (op 560) least significant: low of low * low
v_add_u32 v27, v27, v49 // (op 560) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v22 // (op 560) least significant half
v_addc_co_u32 v27, s[0:1], v27, v23, s[0:1] // (op 560) most significant half



s_mul_i32 s1, s4, s7 // (op 562) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 562) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 562) least significant
s_mul_hi_u32 s8, s4, s6 // (op 562) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 562) most significant: sum
s_add_u32 s1, s1, s8 // (op 562) most significant: sum


s_mov_b32 s52, s20 // (op 562) reg expression
s_mov_b32 s53, s21 // (op 562) reg expression
s_mov_b32 s54, s0 // (op 562) reg expression
s_mov_b32 s55, 131072 // (op 562) call()

s_mul_hi_i32 s51, s16, s11 // (op 627) 
v_lshlrev_b32 v49, 2, v1 // (op 627) 
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
s_sub_u32 s68, s18, 1 // (op 627) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 627) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 627) 

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
s_lshl_b32 s68, 1, s8 // (op 627) 
s_add_i32 s69, s68, s9 // (op 627) 


s_and_b32 s9, s0, s69 // (op 627) 


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
s_add_i32 s68, s19, s18 // (op 627) 


s_and_b32 s18, s0, s68 // (op 627) 


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

s_ashr_i64 s[0:1], s[26:27], 8 // (op 627) 
s_lshr_b32 s51, s0, 1 // (op 627) 

v_lshl_add_u32 v53, s51, 9, v49 // (op 627) 


v_mul_lo_u32 v23, v53, s23 // (op 627) most significant: low of low * high
v_mul_hi_u32 v49, v53, s22 // (op 627) most significant: high of low * low
v_mul_lo_u32 v22, v53, s22 // (op 627) least significant: low of low * low
v_add_u32 v23, v23, v49 // (op 627) most significant: sum

v_mov_b32 v49, s8 // (op 627) 
v_mov_b32 v57, s9 // (op 627) 
v_add_co_u32 v22, s[0:1], v22, v49 // (op 627) least significant half
v_addc_co_u32 v23, s[0:1], v23, v57, s[0:1] // (op 627) most significant half





v_mul_lo_u32 v29, v23, 6 // (op 627) most significant: low of high * low
v_mul_hi_u32 v49, v22, 6 // (op 627) most significant: high of low * low
v_mul_lo_u32 v28, v22, 6 // (op 627) least significant: low of low * low
v_add_u32 v29, v29, v49 // (op 627) most significant: sum


v_lshrrev_b64 v[22:23], 3, v[28:29] // (op 627) 

v_mov_b32 v49, v22 // (op 627) convert

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

s_mul_hi_i32 s51, s16, s11 // (op 691) 
v_lshrrev_b32 v53, 3, v1 // (op 691) 
s_add_i32 s0, s51, s16 // (op 691) 

v_lshlrev_b32 v57, 1, v53 // (op 691) 

s_ashr_i32 s1, s0, 31 // (op 691) 
v_and_b32 v53, -8, v1 // (op 691) 
s_cmp_eq_i32 s11, 0 // (op 691) 
s_mov_b32 s51, scc // (op 691) 
v_sub_u32 v61, v1, v53 // (op 691) 

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
s_sub_u32 s68, s18, 1 // (op 691) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 691) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 691) 

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
s_lshl_b32 s68, 1, s19 // (op 691) 
s_add_i32 s69, s68, s51 // (op 691) 


s_and_b32 s51, s0, s69 // (op 691) 


s_add_i32 s0, s18, s51 // (op 691) 


s_ashr_i32 s18, s0, s19 // (op 691) 


s_lshl_b32 s0, s48, 24 // (op 691) 
s_ashr_i32 s19, s0, 31 // (op 691) 

s_xor_b32 s0, s18, s19 // (op 691) 

s_sub_i32 s18, s0, s19 // (op 691) 


s_mul_i32 s0, s47, s46 // (op 691) 
s_add_u32 s68, s6, 16 // (op 691) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 691) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 691) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 691) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 691) 

s_sub_i32 s19, s68, s0 // (op 691) 

s_mul_i32 s51, s18, s19 // (op 691) 


s_sub_u32 s18, s8, s51 // (op 691) 


s_add_u32 s8, s18, s0 // (op 691) 


s_cmp_eq_i32 s9, 0 // (op 691) 
s_mov_b32 s0, scc // (op 691) 

s_cmp_eq_i32 s0, 0 // (op 691) 
s_cselect_b32 s9, s8, s1 // (op 691) 



v_lshl_add_u32 v53, s9, 4, v57 // (op 691) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 691) 
s_lshr_b32 s51, s0, 1 // (op 691) 

v_lshl_add_u32 v57, s51, 4, v61 // (op 691) 


v_mul_lo_u32 v61, v53, s28 // (op 691) 
v_add_u32 v61, v61, v57 // (op 691) 


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

s_mul_hi_i32 s51, s16, s11 // (op 755) 
v_and_b32 v53, -16, v1 // (op 755) 
s_add_i32 s0, s51, s16 // (op 755) 

v_sub_u32 v57, v1, v53 // (op 755) 

s_ashr_i32 s1, s0, 31 // (op 755) 
v_lshrrev_b32 v53, 4, v1 // (op 755) 
s_cmp_eq_i32 s11, 0 // (op 755) 
s_mov_b32 s51, scc // (op 755) 
v_lshlrev_b32 v13, 1, v53 // (op 755) 

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
s_sub_u32 s68, s18, 1 // (op 755) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 755) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 755) 

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
s_lshl_b32 s68, 1, s8 // (op 755) 
s_add_i32 s69, s68, s9 // (op 755) 


s_and_b32 s9, s0, s69 // (op 755) 


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
s_add_i32 s68, s19, s18 // (op 755) 


s_and_b32 s18, s0, s68 // (op 755) 


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



v_lshl_add_u32 v53, s1, 4, v57 // (op 755) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 755) 
s_lshr_b32 s51, s0, 1 // (op 755) 

v_lshl_add_u32 v57, s51, 4, v13 // (op 755) 


v_mul_lo_u32 v13, v57, s32 // (op 755) 
v_add_u32 v13, v13, v53 // (op 755) 


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

v_and_b32 v53, -16, v1 // (op 848) 
v_sub_u32 v57, v1, v53 // (op 848) 

v_lshlrev_b32 v53, 4, v57 // (op 848) 

v_lshrrev_b32 v57, 4, v1 // (op 848) 
v_lshl_add_u32 v14, v57, 8, v53 // (op 848) 


s_nop 0
v_readfirstlane_b32 s51, v14 // (op 848) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v53, 6, v1 // (op 900) 
v_mul_lo_u32 v57, v53, 6 // (op 900) 
v_lshrrev_b32 v14, 3, v57 // (op 900) 

v_lshrrev_b32 v57, 4, v53 // (op 900) 

v_lshlrev_b32 v53, 2, v57 // (op 900) 

v_add_u32 v57, v14, v53 // (op 900) 


v_and_b32 v53, -8, v1 // (op 953) 
v_sub_u32 v14, v1, v53 // (op 953) 

v_lshrrev_b32 v53, 3, v1 // (op 953) 
v_lshl_add_u32 v15, v53, 4, v14 // (op 953) 


v_and_b32 v14, -16, v1 // (op 1006) 
v_sub_u32 v53, v1, v14 // (op 1006) 

v_lshrrev_b32 v14, 4, v1 // (op 1006) 
v_lshl_add_u32 v16, v14, 5, v53 // (op 1006) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 198) 
s_and_b32 s8, s0, 4294967294 // (op 198) 

s_ashr_i64 s[0:1], s[26:27], 8 // (op 197) 
s_mov_b32 s9, s0 // (op 197) convert

s_cmp_lt_u32 s8, s9 // (op 197) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_197 // (op 197) Condition: Top (jump to RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_197 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_197:
s_barrier  // (op 1222) 
s_add_u32 s8, s8, 1 // (op 199) 

s_lshl_b64 s[0:1], s[4:5], 2 // (op 531) 
v_mov_b64 v[24:25], s[0:1] // (op 531) convert


v_mov_b64 v[22:23], v[26:27] // (op 531) 
v_mov_b64 v[20:21], v[22:23] // (op 531) 
s_mov_b32 m0, s51 // (op 531) reg expression
buffer_load_dwordx4 v20, s[52:55], 0 offen offset: 0 lds // (op 531) Load value direct to lds
v_add_co_u32 v20, s[0:1], v20, v24 // (op 531) least significant half
v_addc_co_u32 v21, s[0:1], 0, v21, s[0:1] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
buffer_load_dwordx4 v20, s[52:55], 0 offen offset: 0 lds // (op 531) Load value direct to lds
v_add_co_u32 v20, s[0:1], v20, v24 // (op 531) least significant half
v_addc_co_u32 v21, s[0:1], 0, v21, s[0:1] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
buffer_load_dwordx4 v20, s[52:55], 0 offen offset: 0 lds // (op 531) Load value direct to lds
v_add_co_u32 v20, s[0:1], v20, v24 // (op 531) least significant half
v_addc_co_u32 v21, s[0:1], 0, v21, s[0:1] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
buffer_load_dwordx4 v20, s[52:55], 0 offen offset: 0 lds // (op 531) Load value direct to lds



v_add_co_u32 v26, vcc, 256, v26 // (op 550) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 550) most significant half

s_mul_i32 s1, s23, 6 // (op 225) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 225) least significant
s_mul_hi_u32 s9, s22, 6 // (op 225) most significant: high of low * low
s_add_u32 s1, s1, s9 // (op 225) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 225) 


v_mov_b32 v12, v49 // (op 225) 
v_mov_b32 v14, v12 // (op 225) reg expression
buffer_load_dwordx3 v[18:20], v14, s[56:59], 0 offen // (op 225) Load value
v_add_u32 v14, v14, v24 // (op 225) 
buffer_load_dwordx3 v[28:30], v14, s[56:59], 0 offen // (op 225) Load value
v_add_u32 v14, v14, v24 // (op 225) 
buffer_load_dwordx3 v[32:34], v14, s[56:59], 0 offen // (op 225) Load value
v_add_u32 v14, v14, v24 // (op 225) 
buffer_load_dwordx3 v[36:38], v14, s[56:59], 0 offen // (op 225) Load value



s_lshl_b64 s[0:1], s[22:23], 8 // (op 619) 
s_mul_i32 s19, s1, 6 // (op 619) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 619) least significant
s_mul_hi_u32 s9, s0, 6 // (op 619) most significant: high of low * low
s_add_u32 s19, s19, s9 // (op 619) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 619) 

v_add_u32 v49, v49, s0 // (op 619) 


s_barrier  // (op 1241) 
v_mov_b64 v[24:25], s[28:29] // (op 226) convert




v_mov_b32 v12, v61 // (op 226) 
v_mov_b32 v14, v12 // (op 226) 
buffer_load_ubyte v22, v14, s[60:63], 0 offen offset: 0 // (op 226) Load value
v_add_u32 v14, v14, v24 // (op 226) 
s_waitcnt vmcnt(4)
ds_write_b96 v57, v[18:20] offset:8192 // (op 222) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v57, v[28:30] offset:8208 // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v57, v[32:34] offset:8224 // (op 222) Store local data 
buffer_load_ubyte v23, v14, s[60:63], 0 offen offset: 0 // (op 226) Load value



v_add_u32 v61, 8, v61 // (op 683) 
s_waitcnt vmcnt(2)
ds_write_b96 v57, v[36:38] offset:8240 // (op 222) Store local data 


v_mov_b64 v[24:25], s[32:33] // (op 227) convert

s_barrier  // (op 1262) 
v_mov_b32 v12, v13 // (op 227) 



v_mov_b32 v14, v12 // (op 227) reg expression
buffer_load_ubyte v20, v14, s[64:67], 0 offen offset: 0 // (op 227) Load value
v_add_u32 v14, v14, v24 // (op 227) 
buffer_load_ubyte v21, v14, s[64:67], 0 offen offset: 0 // (op 227) Load value



s_lshl_b32 s9, s32, 3 // (op 747) 
v_add_u32 v13, v13, s9 // (op 747) 

s_waitcnt vmcnt(3)
ds_write_b8 v15, v22 offset:12416 // (op 223) Store local data 
s_barrier  // (op 1231) 
s_waitcnt vmcnt(2)
ds_write_b8 v15, v23 offset:12424 // (op 223) Store local data 




ds_read_b128 v[28:31], v3 // (op 205) Load local data 
s_barrier  // (op 1282) 
ds_read_b128 v[32:35], v3 offset:64 // (op 205) Load local data 



s_waitcnt lgkmcnt(4)
s_barrier  // (op 1252) 
s_waitcnt vmcnt(1)
ds_write_b8 v16, v20 offset:12288 // (op 224) Store local data 



ds_read_b96_tr_b6 v[36:38], v5 offset:8192 // (op 207) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v5 offset:8448 // (op 207) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v16, v21 offset:12304 // (op 224) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 1272) 



ds_read_u8 v12, v0 offset:12416 // (op 209) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1292) 


ds_read_u8 v14, v7 offset:12288 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:41], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 212) 







ds_read_b128 v[18:21], v3 offset:128 // (op 214) Load local data 
ds_read_b128 v[22:25], v3 offset:192 // (op 214) Load local data 



ds_read_b96_tr_b6 v[28:30], v5 offset:10240 // (op 216) Transpose load local data 
ds_read_b96_tr_b6 v[31:33], v5 offset:10496 // (op 216) Transpose load local data 



ds_read_u8 v12, v0 offset:12420 // (op 218) Load local data 


ds_read_u8 v14, v7 offset:12352 // (op 220) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[28:33], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 221) 




s_ashr_i64 s[0:1], s[26:27], 8 // (op 197) 
s_mov_b32 s9, s0 // (op 197) convert

s_cmp_lt_u32 s8, s9 // (op 197) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_197 // (op 197) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_197 if true)
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_197:























s_barrier  // (op 1369) 








v_cmp_eq_f32 vcc, 0.00000, v8 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_157
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

s_mul_hi_i32 s0, s16, s11 // (op 1048) 
v_lshrrev_b32 v0, 6, v1 // (op 1048) 
s_add_i32 s1, s0, s16 // (op 1048) 

v_and_b32 v3, 63, v1 // (op 1048) 
s_ashr_i32 s0, s1, 31 // (op 1048) 
v_lshrrev_b32 v9, 4, v3 // (op 1048) 
s_cmp_eq_i32 s11, 0 // (op 1048) 
s_mov_b32 s4, scc // (op 1048) 
v_lshlrev_b32 v11, 2, v9 // (op 1048) 

s_cmp_eq_i32 s4, 0 // (op 1048) 
s_cselect_b32 s5, 0, -1 // (op 1048) 

v_lshl_add_u32 v9, v0, 4, v11 // (op 1048) 


s_and_b32 s4, s12, 31 // (op 1048) 
v_lshrrev_b32 v0, 2, v3 // (op 1048) 

s_lshl_b32 s8, 1, s4 // (op 1048) 
v_and_b32 v3, 3, v0 // (op 1048) 

s_add_i32 s9, s8, s5 // (op 1048) 


v_and_b32 v0, 3, v1 // (op 1048) 
s_and_b32 s5, s0, s9 // (op 1048) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 1048) 


s_add_i32 s0, s1, s5 // (op 1048) 


s_ashr_i32 s1, s0, s4 // (op 1048) 


s_lshl_b32 s0, s12, 24 // (op 1048) 
s_ashr_i32 s4, s0, 31 // (op 1048) 

s_xor_b32 s0, s1, s4 // (op 1048) 

s_sub_i32 s1, s0, s4 // (op 1048) 


s_mul_hi_i32 s0, s1, s13 // (op 1048) 
s_add_i32 s4, s0, s1 // (op 1048) 

s_ashr_i32 s0, s4, 31 // (op 1048) 
s_cmp_eq_i32 s13, 0 // (op 1048) 
s_mov_b32 s5, scc // (op 1048) 
s_cmp_eq_i32 s5, 0 // (op 1048) 
s_cselect_b32 s8, 0, -1 // (op 1048) 

s_and_b32 s5, s14, 31 // (op 1048) 
s_lshl_b32 s9, 1, s5 // (op 1048) 
s_add_i32 s18, s9, s8 // (op 1048) 


s_and_b32 s8, s0, s18 // (op 1048) 


s_add_i32 s0, s4, s8 // (op 1048) 


s_ashr_i32 s4, s0, s5 // (op 1048) 


s_lshl_b32 s0, s14, 24 // (op 1048) 
s_ashr_i32 s5, s0, 31 // (op 1048) 

s_xor_b32 s0, s4, s5 // (op 1048) 

s_sub_i32 s4, s0, s5 // (op 1048) 


s_mul_i32 s0, s1, s46 // (op 1048) 
s_sub_u32 s5, s16, s0 // (op 1048) 

s_mul_i32 s0, s4, s46 // (op 1048) 
s_add_u32 s0, s0, s5 // (op 1048) 

s_mul_hi_i32 s5, s16, s10 // (op 1048) 
s_add_i32 s8, s5, s16 // (op 1048) 

s_ashr_i32 s5, s8, 31 // (op 1048) 
s_cmp_eq_i32 s10, 0 // (op 1048) 
s_mov_b32 s9, scc // (op 1048) 
s_cmp_eq_i32 s9, 0 // (op 1048) 
s_cselect_b32 s18, 0, -1 // (op 1048) 

s_and_b32 s9, s49, 31 // (op 1048) 
s_lshl_b32 s19, 1, s9 // (op 1048) 
s_add_i32 s20, s19, s18 // (op 1048) 


s_and_b32 s18, s5, s20 // (op 1048) 


s_add_i32 s5, s8, s18 // (op 1048) 


s_ashr_i32 s8, s5, s9 // (op 1048) 


s_lshl_b32 s5, s49, 24 // (op 1048) 
s_ashr_i32 s9, s5, 31 // (op 1048) 

s_xor_b32 s5, s8, s9 // (op 1048) 

s_sub_i32 s8, s5, s9 // (op 1048) 


s_add_u32 s18, s2, 16 // (op 1048) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 1048) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 1048) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 1048) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 1048) 

s_mul_i32 s5, s46, s18 // (op 1048) 
s_mul_i32 s9, s50, s5 // (op 1048) 

s_mul_i32 s5, s8, s9 // (op 1048) 

s_sub_u32 s9, s16, s5 // (op 1048) 

s_mov_b32 s5, s9 // (op 1048) 
s_mul_hi_i32 s20, s5, s15 // (op 1048) 
s_add_i32 s21, s20, s5 // (op 1048) 


s_ashr_i32 s5, s21, 31 // (op 1048) 
s_cmp_eq_i32 s15, 0 // (op 1048) 
s_mov_b32 s20, scc // (op 1048) 
s_cmp_eq_i32 s20, 0 // (op 1048) 
s_cselect_b32 s22, 0, -1 // (op 1048) 

s_and_b32 s20, s48, 31 // (op 1048) 
s_lshl_b32 s23, 1, s20 // (op 1048) 
s_add_i32 s24, s23, s22 // (op 1048) 


s_and_b32 s22, s5, s24 // (op 1048) 


s_add_i32 s5, s21, s22 // (op 1048) 


s_ashr_i32 s21, s5, s20 // (op 1048) 


s_lshl_b32 s5, s48, 24 // (op 1048) 
s_ashr_i32 s20, s5, 31 // (op 1048) 

s_xor_b32 s5, s21, s20 // (op 1048) 

s_sub_i32 s21, s5, s20 // (op 1048) 


s_mul_i32 s5, s47, s46 // (op 1048) 
s_add_u32 s22, s6, 16 // (op 1048) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1048) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1048) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1048) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1048) 

s_sub_i32 s20, s22, s5 // (op 1048) 

s_mul_i32 s22, s21, s20 // (op 1048) 

s_sub_u32 s20, s9, s22 // (op 1048) 


s_add_u32 s9, s20, s5 // (op 1048) 


s_cmp_eq_i32 s8, 0 // (op 1048) 
s_mov_b32 s5, scc // (op 1048) 

s_cmp_eq_i32 s5, 0 // (op 1048) 
s_cselect_b32 s8, s9, s0 // (op 1048) 


s_mul_i32 s0, s4, s18 // (op 1048) 


v_lshl_add_u32 v0, s8, 4, v9 // (op 1048) 


s_sub_i32 s4, s1, s0 // (op 1048) 


s_cmp_eq_i32 s5, 0 // (op 1048) 
s_cselect_b32 s0, s21, s4 // (op 1048) 



v_lshl_add_u32 v3, s0, 4, v11 // (op 1048) 


v_mul_lo_u32 v9, v3, s38 // (op 1048) 

v_add_lshl_u32 v3, v0, v9, 1 // (op 1048) 


s_mul_i32 s1, s38, s3 // (op 1050) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 1050) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 1050) least significant
s_mul_hi_u32 s5, s38, s2 // (op 1050) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 1050) most significant: sum
s_add_u32 s1, s1, s5 // (op 1050) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 1 // (op 1050) 

s_mov_b32 s20, s40 // (op 1050) reg expression
s_mov_b32 s21, s41 // (op 1050) reg expression
s_mov_b32 s22, s4 // (op 1050) reg expression
s_mov_b32 s23, 131072 // (op 1050) call()


v_pack_b32_f16 v12, v4, v5 // (op 58) 
v_pack_b32_f16 v13, v6, v7 // (op 58) 


buffer_store_dwordx2 v[12:13], v3, s[20:23], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_157
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_157:
s_mul_hi_i32 s0, s16, s11 // (op 818) 
v_lshrrev_b32 v0, 6, v1 // (op 818) 
s_add_i32 s1, s0, s16 // (op 818) 

v_and_b32 v9, 63, v1 // (op 818) 
s_ashr_i32 s0, s1, 31 // (op 818) 
v_lshrrev_b32 v3, 4, v9 // (op 818) 
s_cmp_eq_i32 s11, 0 // (op 818) 
s_mov_b32 s4, scc // (op 818) 
v_lshlrev_b32 v4, 2, v3 // (op 818) 

s_cmp_eq_i32 s4, 0 // (op 818) 
s_cselect_b32 s5, 0, -1 // (op 818) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 818) 


s_and_b32 s4, s12, 31 // (op 818) 
v_lshrrev_b32 v0, 2, v9 // (op 818) 

s_lshl_b32 s8, 1, s4 // (op 818) 
v_and_b32 v9, 3, v0 // (op 818) 

s_add_i32 s9, s8, s5 // (op 818) 


v_and_b32 v0, 3, v1 // (op 818) 
s_and_b32 s5, s0, s9 // (op 818) 


v_lshl_add_u32 v4, v9, 2, v0 // (op 818) 


s_add_i32 s0, s1, s5 // (op 818) 


s_ashr_i32 s1, s0, s4 // (op 818) 


s_lshl_b32 s0, s12, 24 // (op 818) 
s_ashr_i32 s4, s0, 31 // (op 818) 

s_xor_b32 s0, s1, s4 // (op 818) 

s_sub_i32 s1, s0, s4 // (op 818) 


s_mul_hi_i32 s0, s1, s13 // (op 818) 
s_add_i32 s4, s0, s1 // (op 818) 

s_ashr_i32 s0, s4, 31 // (op 818) 
s_cmp_eq_i32 s13, 0 // (op 818) 
s_mov_b32 s5, scc // (op 818) 
s_cmp_eq_i32 s5, 0 // (op 818) 
s_cselect_b32 s8, 0, -1 // (op 818) 

s_and_b32 s5, s14, 31 // (op 818) 
s_lshl_b32 s9, 1, s5 // (op 818) 
s_add_i32 s18, s9, s8 // (op 818) 


s_and_b32 s19, s0, s18 // (op 818) 


s_add_i32 s0, s4, s19 // (op 818) 


s_ashr_i32 s4, s0, s5 // (op 818) 


s_lshl_b32 s0, s14, 24 // (op 818) 
s_ashr_i32 s5, s0, 31 // (op 818) 

s_xor_b32 s0, s4, s5 // (op 818) 

s_sub_i32 s4, s0, s5 // (op 818) 


s_mul_i32 s0, s1, s46 // (op 818) 
s_sub_u32 s5, s16, s0 // (op 818) 

s_mul_i32 s0, s4, s46 // (op 818) 
s_add_u32 s0, s0, s5 // (op 818) 

s_mul_hi_i32 s5, s16, s10 // (op 818) 
s_add_i32 s8, s5, s16 // (op 818) 

s_ashr_i32 s5, s8, 31 // (op 818) 
s_cmp_eq_i32 s10, 0 // (op 818) 
s_mov_b32 s9, scc // (op 818) 
s_cmp_eq_i32 s9, 0 // (op 818) 
s_cselect_b32 s18, 0, -1 // (op 818) 

s_and_b32 s9, s49, 31 // (op 818) 
s_lshl_b32 s19, 1, s9 // (op 818) 
s_add_i32 s24, s19, s18 // (op 818) 


s_and_b32 s18, s5, s24 // (op 818) 


s_add_i32 s5, s8, s18 // (op 818) 


s_ashr_i32 s8, s5, s9 // (op 818) 


s_lshl_b32 s5, s49, 24 // (op 818) 
s_ashr_i32 s9, s5, 31 // (op 818) 

s_xor_b32 s5, s8, s9 // (op 818) 

s_sub_i32 s8, s5, s9 // (op 818) 


s_add_u32 s18, s2, 16 // (op 818) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 818) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 818) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 818) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 4 // (op 818) 

s_mul_i32 s5, s46, s18 // (op 818) 
s_mul_i32 s9, s50, s5 // (op 818) 

s_mul_i32 s5, s8, s9 // (op 818) 

s_sub_u32 s9, s16, s5 // (op 818) 

s_mov_b32 s5, s9 // (op 818) 
s_mul_hi_i32 s24, s5, s15 // (op 818) 
s_add_i32 s25, s24, s5 // (op 818) 


s_ashr_i32 s5, s25, 31 // (op 818) 
s_cmp_eq_i32 s15, 0 // (op 818) 
s_mov_b32 s24, scc // (op 818) 
s_cmp_eq_i32 s24, 0 // (op 818) 
s_cselect_b32 s26, 0, -1 // (op 818) 

s_and_b32 s24, s48, 31 // (op 818) 
s_lshl_b32 s27, 1, s24 // (op 818) 
s_add_i32 s28, s27, s26 // (op 818) 


s_and_b32 s26, s5, s28 // (op 818) 


s_add_i32 s5, s25, s26 // (op 818) 


s_ashr_i32 s25, s5, s24 // (op 818) 


s_lshl_b32 s5, s48, 24 // (op 818) 
s_ashr_i32 s24, s5, 31 // (op 818) 

s_xor_b32 s5, s25, s24 // (op 818) 

s_sub_i32 s25, s5, s24 // (op 818) 


s_mul_i32 s5, s47, s46 // (op 818) 
s_add_u32 s26, s6, 16 // (op 818) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 818) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 818) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 818) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 818) 

s_sub_i32 s24, s26, s5 // (op 818) 

s_mul_i32 s26, s25, s24 // (op 818) 

s_sub_u32 s24, s9, s26 // (op 818) 


s_add_u32 s9, s24, s5 // (op 818) 


s_cmp_eq_i32 s8, 0 // (op 818) 
s_mov_b32 s5, scc // (op 818) 

s_cmp_eq_i32 s5, 0 // (op 818) 
s_cselect_b32 s8, s9, s0 // (op 818) 


s_mul_i32 s0, s4, s18 // (op 818) 


v_lshl_add_u32 v0, s8, 4, v3 // (op 818) 


s_sub_i32 s4, s1, s0 // (op 818) 


s_cmp_eq_i32 s5, 0 // (op 818) 
s_cselect_b32 s24, s25, s4 // (op 818) 



v_lshl_add_u32 v3, s24, 4, v4 // (op 818) 


v_mul_lo_u32 v9, v3, s42 // (op 818) 

v_add_lshl_u32 v3, v0, v9, 1 // (op 818) 


s_mul_i32 s1, s42, s3 // (op 820) most significant: low of low * high
s_mul_i32 s4, s43, s2 // (op 820) most significant: low of high * low
s_mul_i32 s0, s42, s2 // (op 820) least significant
s_mul_hi_u32 s5, s42, s2 // (op 820) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 820) most significant: sum
s_add_u32 s1, s1, s5 // (op 820) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 1 // (op 820) 

s_mov_b32 s24, s44 // (op 820) reg expression
s_mov_b32 s25, s45 // (op 820) reg expression
s_mov_b32 s26, s4 // (op 820) reg expression
s_mov_b32 s27, 131072 // (op 820) call()





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

s_mul_hi_i32 s0, s16, s11 // (op 1065) 
v_lshrrev_b32 v0, 6, v1 // (op 1065) 
s_add_i32 s1, s0, s16 // (op 1065) 

v_and_b32 v9, 63, v1 // (op 1065) 
s_ashr_i32 s0, s1, 31 // (op 1065) 
v_lshrrev_b32 v11, 4, v9 // (op 1065) 
s_cmp_eq_i32 s11, 0 // (op 1065) 
s_mov_b32 s4, scc // (op 1065) 
v_lshlrev_b32 v3, 2, v11 // (op 1065) 

s_cmp_eq_i32 s4, 0 // (op 1065) 
s_cselect_b32 s5, 0, -1 // (op 1065) 

v_lshl_add_u32 v11, v0, 4, v3 // (op 1065) 


s_and_b32 s4, s12, 31 // (op 1065) 
v_lshrrev_b32 v0, 2, v9 // (op 1065) 

s_lshl_b32 s8, 1, s4 // (op 1065) 
v_and_b32 v9, 3, v0 // (op 1065) 

s_add_i32 s9, s8, s5 // (op 1065) 


v_and_b32 v0, 3, v1 // (op 1065) 
s_and_b32 s5, s0, s9 // (op 1065) 


v_lshl_add_u32 v3, v9, 2, v0 // (op 1065) 


s_add_i32 s0, s1, s5 // (op 1065) 


s_ashr_i32 s1, s0, s4 // (op 1065) 


s_lshl_b32 s0, s12, 24 // (op 1065) 
s_ashr_i32 s4, s0, 31 // (op 1065) 

s_xor_b32 s0, s1, s4 // (op 1065) 

s_sub_i32 s1, s0, s4 // (op 1065) 


s_mul_hi_i32 s0, s1, s13 // (op 1065) 
s_add_i32 s4, s0, s1 // (op 1065) 

s_ashr_i32 s0, s4, 31 // (op 1065) 
s_cmp_eq_i32 s13, 0 // (op 1065) 
s_mov_b32 s5, scc // (op 1065) 
s_cmp_eq_i32 s5, 0 // (op 1065) 
s_cselect_b32 s8, 0, -1 // (op 1065) 

s_and_b32 s5, s14, 31 // (op 1065) 
s_lshl_b32 s9, 1, s5 // (op 1065) 
s_add_i32 s18, s9, s8 // (op 1065) 


s_and_b32 s19, s0, s18 // (op 1065) 


s_add_i32 s0, s4, s19 // (op 1065) 


s_ashr_i32 s4, s0, s5 // (op 1065) 


s_lshl_b32 s0, s14, 24 // (op 1065) 
s_ashr_i32 s5, s0, 31 // (op 1065) 

s_xor_b32 s0, s4, s5 // (op 1065) 

s_sub_i32 s4, s0, s5 // (op 1065) 


s_mul_i32 s0, s1, s46 // (op 1065) 
s_sub_u32 s5, s16, s0 // (op 1065) 

s_mul_i32 s0, s4, s46 // (op 1065) 
s_add_u32 s0, s0, s5 // (op 1065) 

s_mul_hi_i32 s5, s16, s10 // (op 1065) 
s_add_i32 s8, s5, s16 // (op 1065) 

s_ashr_i32 s5, s8, 31 // (op 1065) 
s_cmp_eq_i32 s10, 0 // (op 1065) 
s_mov_b32 s9, scc // (op 1065) 
s_cmp_eq_i32 s9, 0 // (op 1065) 
s_cselect_b32 s18, 0, -1 // (op 1065) 

s_and_b32 s9, s49, 31 // (op 1065) 
s_lshl_b32 s19, 1, s9 // (op 1065) 
s_add_i32 s24, s19, s18 // (op 1065) 


s_and_b32 s18, s5, s24 // (op 1065) 


s_add_i32 s5, s8, s18 // (op 1065) 


s_ashr_i32 s8, s5, s9 // (op 1065) 


s_lshl_b32 s5, s49, 24 // (op 1065) 
s_ashr_i32 s9, s5, 31 // (op 1065) 

s_xor_b32 s5, s8, s9 // (op 1065) 

s_sub_i32 s8, s5, s9 // (op 1065) 


s_add_u32 s18, s2, 16 // (op 1065) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 1065) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 1065) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 1065) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 4 // (op 1065) 

s_mul_i32 s5, s46, s18 // (op 1065) 
s_mul_i32 s9, s50, s5 // (op 1065) 

s_mul_i32 s5, s8, s9 // (op 1065) 

s_sub_u32 s9, s16, s5 // (op 1065) 

s_mov_b32 s5, s9 // (op 1065) 
s_mul_hi_i32 s24, s5, s15 // (op 1065) 
s_add_i32 s25, s24, s5 // (op 1065) 


s_ashr_i32 s5, s25, 31 // (op 1065) 
s_cmp_eq_i32 s15, 0 // (op 1065) 
s_mov_b32 s24, scc // (op 1065) 
s_cmp_eq_i32 s24, 0 // (op 1065) 
s_cselect_b32 s26, 0, -1 // (op 1065) 

s_and_b32 s24, s48, 31 // (op 1065) 
s_lshl_b32 s27, 1, s24 // (op 1065) 
s_add_i32 s28, s27, s26 // (op 1065) 


s_and_b32 s26, s5, s28 // (op 1065) 


s_add_i32 s5, s25, s26 // (op 1065) 


s_ashr_i32 s25, s5, s24 // (op 1065) 


s_lshl_b32 s5, s48, 24 // (op 1065) 
s_ashr_i32 s24, s5, 31 // (op 1065) 

s_xor_b32 s5, s25, s24 // (op 1065) 

s_sub_i32 s25, s5, s24 // (op 1065) 


s_mul_i32 s5, s47, s46 // (op 1065) 
s_add_u32 s26, s6, 16 // (op 1065) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 1065) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 1065) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 1065) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 1065) 

s_sub_i32 s24, s26, s5 // (op 1065) 

s_mul_i32 s26, s25, s24 // (op 1065) 

s_sub_u32 s24, s9, s26 // (op 1065) 


s_add_u32 s9, s24, s5 // (op 1065) 


s_cmp_eq_i32 s8, 0 // (op 1065) 
s_mov_b32 s5, scc // (op 1065) 

s_cmp_eq_i32 s5, 0 // (op 1065) 
s_cselect_b32 s8, s9, s0 // (op 1065) 


s_mul_i32 s0, s4, s18 // (op 1065) 


v_lshl_add_u32 v0, s8, 4, v11 // (op 1065) 


s_sub_i32 s4, s1, s0 // (op 1065) 


s_cmp_eq_i32 s5, 0 // (op 1065) 
s_cselect_b32 s24, s25, s4 // (op 1065) 



v_lshl_add_u32 v9, s24, 4, v3 // (op 1065) 


v_mul_lo_u32 v11, v9, s38 // (op 1065) 

v_add_lshl_u32 v9, v0, v11, 1 // (op 1065) 


s_mul_i32 s1, s38, s3 // (op 1067) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 1067) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 1067) least significant
s_mul_hi_u32 s5, s38, s2 // (op 1067) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 1067) most significant: sum
s_add_u32 s1, s1, s5 // (op 1067) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 1 // (op 1067) 

s_mov_b32 s20, s40 // (op 1067) reg expression
s_mov_b32 s21, s41 // (op 1067) reg expression
s_mov_b32 s22, s4 // (op 1067) reg expression
s_mov_b32 s23, 131072 // (op 1067) call()


v_pack_b32_f16 v6, v12, v13 // (op 169) 
v_pack_b32_f16 v7, v14, v15 // (op 169) 


buffer_store_dwordx2 v[6:7], v9, s[20:23], 0 offen // (op 169) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_157:
















s_endpgm  // End of RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.LRR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end:
.size RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_, .LRR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end-RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
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
    DataType: F8
    DataTypeA: F8
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
  - .name: RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
    .symbol: RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_.kd
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

