// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_,@function
RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 917) 
v_and_b32 v3, 15, v1 // (op 917) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 917) 


v_and_b32 v0, 48, v1 // (op 917) 
v_lshl_add_u32 v3, v4, 8, v0 // (op 917) 


v_and_b32 v0, 48, v1 // (op 969) 
v_and_b32 v4, 63, v1 // (op 969) 
v_sub_u32 v5, v4, v0 // (op 969) 

v_lshrrev_b32 v4, 1, v5 // (op 969) 
v_add_lshl_u32 v6, v0, v4, 4 // (op 969) 


v_and_b32 v0, 1, v5 // (op 969) 

v_lshlrev_b32 v4, 3, v0 // (op 969) 

v_add_u32 v0, v6, v4 // (op 969) 


v_lshrrev_b32 v4, 6, v1 // (op 1022) 
v_and_b32 v5, 15, v1 // (op 1022) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 1022) 


v_and_b32 v4, 63, v1 // (op 1022) 
v_lshrrev_b32 v5, 4, v4 // (op 1022) 

v_lshl_add_u32 v4, v6, 3, v5 // (op 1022) 


v_and_b32 v5, 48, v1 // (op 1075) 
v_and_b32 v6, 15, v1 // (op 1075) 
v_add_u32 v7, v5, v6 // (op 1075) 


v_lshrrev_b32 v5, 6, v1 // (op 1148) 
v_and_b32 v6, 15, v1 // (op 1148) 
v_lshl_add_u32 v8, v5, 4, v6 // (op 1148) 


v_and_b32 v5, 48, v1 // (op 1148) 
v_lshl_add_u32 v6, v8, 8, v5 // (op 1148) 


v_and_b32 v5, 48, v1 // (op 1187) 
v_and_b32 v8, 63, v1 // (op 1187) 
v_sub_u32 v9, v8, v5 // (op 1187) 

v_lshrrev_b32 v8, 1, v9 // (op 1187) 
v_add_lshl_u32 v10, v5, v8, 4 // (op 1187) 


v_and_b32 v5, 1, v9 // (op 1187) 

v_lshlrev_b32 v8, 3, v5 // (op 1187) 

v_add_u32 v5, v10, v8 // (op 1187) 


v_lshrrev_b32 v8, 6, v1 // (op 1211) 
v_and_b32 v9, 15, v1 // (op 1211) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 1211) 


v_and_b32 v8, 63, v1 // (op 1211) 
v_lshrrev_b32 v9, 4, v8 // (op 1211) 

v_lshl_add_u32 v8, v10, 3, v9 // (op 1211) 


v_and_b32 v9, 48, v1 // (op 1235) 
v_and_b32 v10, 15, v1 // (op 1235) 
v_add_u32 v11, v9, v10 // (op 1235) 


v_mov_b32 v9, s8 // (op 2) Move value
v_mov_b32 v10, s9 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
s_mul_hi_i32 s0, s16, s11 // (op 628) 
v_lshrrev_b32 v12, 4, v1 // (op 628) 
s_add_i32 s1, s0, s16 // (op 628) 

v_and_b32 v13, -16, v1 // (op 628) 
s_ashr_i32 s0, s1, 31 // (op 628) 
v_sub_u32 v14, v1, v13 // (op 628) 

s_cmp_eq_i32 s11, 0 // (op 628) 
s_mov_b32 s8, scc // (op 628) 
v_lshlrev_b32 v13, 4, v14 // (op 628) 

s_cmp_eq_i32 s8, 0 // (op 628) 
s_cselect_b32 s9, 0, -1 // (op 628) 

v_mov_b32 v15, 0 // (op 628) convert
v_mov_b32 v14, v13 // (op 628) convert

s_and_b32 s8, s12, 31 // (op 628) 
s_lshl_b32 s18, 1, s8 // (op 628) 
s_add_i32 s19, s18, s9 // (op 628) 


s_and_b32 s9, s0, s19 // (op 628) 


s_add_i32 s0, s1, s9 // (op 628) 


s_ashr_i32 s1, s0, s8 // (op 628) 


s_lshl_b32 s0, s12, 24 // (op 628) 
s_ashr_i32 s8, s0, 31 // (op 628) 

s_xor_b32 s0, s1, s8 // (op 628) 

s_sub_i32 s1, s0, s8 // (op 628) 


s_mul_hi_i32 s0, s1, s13 // (op 628) 
s_add_i32 s8, s0, s1 // (op 628) 

s_ashr_i32 s0, s8, 31 // (op 628) 
s_cmp_eq_i32 s13, 0 // (op 628) 
s_mov_b32 s9, scc // (op 628) 
s_cmp_eq_i32 s9, 0 // (op 628) 
s_cselect_b32 s18, 0, -1 // (op 628) 

s_and_b32 s9, s14, 31 // (op 628) 
s_lshl_b32 s19, 1, s9 // (op 628) 
s_add_i32 s51, s19, s18 // (op 628) 


s_and_b32 s18, s0, s51 // (op 628) 


s_add_i32 s0, s8, s18 // (op 628) 


s_ashr_i32 s8, s0, s9 // (op 628) 


s_lshl_b32 s0, s14, 24 // (op 628) 
s_ashr_i32 s9, s0, 31 // (op 628) 

s_xor_b32 s0, s8, s9 // (op 628) 

s_sub_i32 s8, s0, s9 // (op 628) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s46 // (op 628) 

s_sub_u32 s1, s16, s0 // (op 628) 

s_mul_i32 s0, s8, s46 // (op 628) 
s_add_u32 s0, s0, s1 // (op 628) 


s_mul_hi_i32 s1, s16, s10 // (op 628) 
s_add_i32 s8, s1, s16 // (op 628) 

s_ashr_i32 s1, s8, 31 // (op 628) 
s_cmp_eq_i32 s10, 0 // (op 628) 
s_mov_b32 s9, scc // (op 628) 
s_cmp_eq_i32 s9, 0 // (op 628) 
s_cselect_b32 s18, 0, -1 // (op 628) 

s_and_b32 s9, s49, 31 // (op 628) 
s_lshl_b32 s19, 1, s9 // (op 628) 
s_add_i32 s51, s19, s18 // (op 628) 


s_and_b32 s18, s1, s51 // (op 628) 


s_add_i32 s1, s8, s18 // (op 628) 


s_ashr_i32 s8, s1, s9 // (op 628) 


s_lshl_b32 s1, s49, 24 // (op 628) 
s_ashr_i32 s9, s1, 31 // (op 628) 

s_xor_b32 s1, s8, s9 // (op 628) 

s_sub_i32 s8, s1, s9 // (op 628) 


s_add_u32 s18, s2, 16 // (op 628) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 628) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 628) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 628) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 628) 

s_mul_i32 s1, s46, s18 // (op 628) 

s_mul_i32 s9, s50, s1 // (op 628) 

s_mul_i32 s1, s8, s9 // (op 628) 

s_sub_u32 s9, s16, s1 // (op 628) 

s_mov_b32 s1, s9 // (op 628) 
s_mul_hi_i32 s18, s1, s15 // (op 628) 
s_add_i32 s19, s18, s1 // (op 628) 


s_ashr_i32 s1, s19, 31 // (op 628) 
s_cmp_eq_i32 s15, 0 // (op 628) 
s_mov_b32 s18, scc // (op 628) 
s_cmp_eq_i32 s18, 0 // (op 628) 
s_cselect_b32 s51, 0, -1 // (op 628) 

s_and_b32 s18, s48, 31 // (op 628) 
s_lshl_b32 s52, 1, s18 // (op 628) 
s_add_i32 s53, s52, s51 // (op 628) 


s_and_b32 s51, s1, s53 // (op 628) 


s_add_i32 s1, s19, s51 // (op 628) 


s_ashr_i32 s19, s1, s18 // (op 628) 


s_lshl_b32 s1, s48, 24 // (op 628) 
s_ashr_i32 s18, s1, 31 // (op 628) 

s_xor_b32 s1, s19, s18 // (op 628) 

s_sub_i32 s19, s1, s18 // (op 628) 


s_mul_i32 s1, s47, s46 // (op 628) 
s_add_u32 s52, s6, 16 // (op 628) Start of Int64 add, locking SCC
s_addc_u32 s53, s7, 0 // (op 628) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 628) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 628) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 628) 

s_sub_i32 s18, s52, s1 // (op 628) 

s_mul_i32 s51, s19, s18 // (op 628) 


s_sub_u32 s18, s9, s51 // (op 628) 


s_add_u32 s9, s18, s1 // (op 628) 


s_cmp_eq_i32 s8, 0 // (op 628) 
s_mov_b32 s1, scc // (op 628) 

s_cmp_eq_i32 s1, 0 // (op 628) 
s_cselect_b32 s8, s9, s0 // (op 628) 



v_lshl_add_u32 v13, s8, 4, v12 // (op 628) 


v_mul_lo_u32 v17, v13, s5 // (op 628) most significant: low of low * high
v_mul_hi_u32 v12, v13, s4 // (op 628) most significant: high of low * low
v_mul_lo_u32 v16, v13, s4 // (op 628) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 628) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 628) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 628) most significant half



s_mul_i32 s1, s4, s7 // (op 630) most significant: low of low * high
s_mul_i32 s8, s5, s6 // (op 630) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 630) least significant
s_mul_hi_u32 s9, s4, s6 // (op 630) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 630) most significant: sum
s_add_u32 s1, s1, s9 // (op 630) most significant: sum


s_mov_b32 s52, s20 // (op 630) reg expression
s_mov_b32 s53, s21 // (op 630) reg expression
s_mov_b32 s54, s0 // (op 630) reg expression
s_mov_b32 s55, 131072 // (op 630) call()

s_mul_hi_i32 s51, s16, s11 // (op 647) 
v_lshrrev_b32 v12, 4, v1 // (op 647) 
s_add_i32 s0, s51, s16 // (op 647) 

v_and_b32 v13, -16, v1 // (op 647) 
s_ashr_i32 s1, s0, 31 // (op 647) 
v_sub_u32 v14, v1, v13 // (op 647) 

s_cmp_eq_i32 s11, 0 // (op 647) 
s_mov_b32 s51, scc // (op 647) 
v_lshlrev_b32 v13, 4, v14 // (op 647) 

s_cmp_eq_i32 s51, 0 // (op 647) 
s_cselect_b32 s8, 0, -1 // (op 647) 

v_mov_b32 v15, 0 // (op 647) convert
v_mov_b32 v14, v13 // (op 647) convert

s_and_b32 s9, s12, 31 // (op 647) 
s_lshl_b32 s51, 1, s9 // (op 647) 
s_add_i32 s18, s51, s8 // (op 647) 


s_and_b32 s8, s1, s18 // (op 647) 


s_add_i32 s1, s0, s8 // (op 647) 


s_ashr_i32 s0, s1, s9 // (op 647) 


s_lshl_b32 s1, s12, 24 // (op 647) 
s_ashr_i32 s51, s1, 31 // (op 647) 

s_xor_b32 s1, s0, s51 // (op 647) 

s_sub_i32 s0, s1, s51 // (op 647) 


s_mul_hi_i32 s1, s0, s13 // (op 647) 
s_add_i32 s51, s1, s0 // (op 647) 

s_ashr_i32 s1, s51, 31 // (op 647) 
s_cmp_eq_i32 s13, 0 // (op 647) 
s_mov_b32 s8, scc // (op 647) 
s_cmp_eq_i32 s8, 0 // (op 647) 
s_cselect_b32 s9, 0, -1 // (op 647) 

s_and_b32 s8, s14, 31 // (op 647) 
s_lshl_b32 s18, 1, s8 // (op 647) 
s_add_i32 s19, s18, s9 // (op 647) 


s_and_b32 s9, s1, s19 // (op 647) 


s_add_i32 s1, s51, s9 // (op 647) 


s_ashr_i32 s9, s1, s8 // (op 647) 


s_lshl_b32 s1, s14, 24 // (op 647) 
s_ashr_i32 s8, s1, 31 // (op 647) 

s_xor_b32 s1, s9, s8 // (op 647) 

s_sub_i32 s9, s1, s8 // (op 647) 


s_mul_i32 s1, s0, s46 // (op 647) 

s_sub_u32 s0, s16, s1 // (op 647) 

s_mul_i32 s1, s9, s46 // (op 647) 
s_add_u32 s1, s1, s0 // (op 647) 


s_mul_hi_i32 s0, s16, s10 // (op 647) 
s_add_i32 s51, s0, s16 // (op 647) 

s_ashr_i32 s0, s51, 31 // (op 647) 
s_cmp_eq_i32 s10, 0 // (op 647) 
s_mov_b32 s8, scc // (op 647) 
s_cmp_eq_i32 s8, 0 // (op 647) 
s_cselect_b32 s9, 0, -1 // (op 647) 

s_and_b32 s8, s49, 31 // (op 647) 
s_lshl_b32 s18, 1, s8 // (op 647) 
s_add_i32 s19, s18, s9 // (op 647) 


s_and_b32 s9, s0, s19 // (op 647) 


s_add_i32 s0, s51, s9 // (op 647) 


s_ashr_i32 s9, s0, s8 // (op 647) 


s_lshl_b32 s0, s49, 24 // (op 647) 
s_ashr_i32 s8, s0, 31 // (op 647) 

s_xor_b32 s0, s9, s8 // (op 647) 

s_sub_i32 s9, s0, s8 // (op 647) 


s_add_u32 s18, s2, 16 // (op 647) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 647) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 647) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 647) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 647) 

s_mul_i32 s0, s46, s18 // (op 647) 

s_mul_i32 s8, s50, s0 // (op 647) 

s_mul_i32 s0, s9, s8 // (op 647) 

s_sub_u32 s8, s16, s0 // (op 647) 

s_mov_b32 s0, s8 // (op 647) 
s_mul_hi_i32 s51, s0, s15 // (op 647) 
s_add_i32 s18, s51, s0 // (op 647) 


s_ashr_i32 s0, s18, 31 // (op 647) 
s_cmp_eq_i32 s15, 0 // (op 647) 
s_mov_b32 s19, scc // (op 647) 
s_cmp_eq_i32 s19, 0 // (op 647) 
s_cselect_b32 s51, 0, -1 // (op 647) 

s_and_b32 s19, s48, 31 // (op 647) 
s_lshl_b32 s56, 1, s19 // (op 647) 
s_add_i32 s57, s56, s51 // (op 647) 


s_and_b32 s51, s0, s57 // (op 647) 


s_add_i32 s0, s18, s51 // (op 647) 


s_ashr_i32 s18, s0, s19 // (op 647) 


s_lshl_b32 s0, s48, 24 // (op 647) 
s_ashr_i32 s19, s0, 31 // (op 647) 

s_xor_b32 s0, s18, s19 // (op 647) 

s_sub_i32 s18, s0, s19 // (op 647) 


s_mul_i32 s0, s47, s46 // (op 647) 
s_add_u32 s56, s6, 16 // (op 647) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 647) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 647) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 647) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 647) 

s_sub_i32 s19, s56, s0 // (op 647) 

s_mul_i32 s51, s18, s19 // (op 647) 


s_sub_u32 s18, s8, s51 // (op 647) 


s_add_u32 s8, s18, s0 // (op 647) 


s_cmp_eq_i32 s9, 0 // (op 647) 
s_mov_b32 s0, scc // (op 647) 

s_cmp_eq_i32 s0, 0 // (op 647) 
s_cselect_b32 s9, s8, s1 // (op 647) 



v_lshl_add_u32 v13, s9, 4, v12 // (op 647) 


v_mul_lo_u32 v19, v13, s5 // (op 647) most significant: low of low * high
v_mul_hi_u32 v12, v13, s4 // (op 647) most significant: high of low * low
v_mul_lo_u32 v18, v13, s4 // (op 647) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 647) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 647) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 647) most significant half



s_mul_i32 s1, s4, s7 // (op 649) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 649) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 649) least significant
s_mul_hi_u32 s8, s4, s6 // (op 649) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 649) most significant: sum
s_add_u32 s1, s1, s8 // (op 649) most significant: sum


s_mov_b32 s52, s20 // (op 649) reg expression
s_mov_b32 s53, s21 // (op 649) reg expression
s_mov_b32 s54, s0 // (op 649) reg expression
s_mov_b32 s55, 131072 // (op 649) call()

s_mul_hi_i32 s51, s16, s11 // (op 692) 
s_add_i32 s0, s51, s16 // (op 692) 

s_ashr_i32 s1, s0, 31 // (op 692) 
s_cmp_eq_i32 s11, 0 // (op 692) 
s_mov_b32 s51, scc // (op 692) 
s_cmp_eq_i32 s51, 0 // (op 692) 
s_cselect_b32 s8, 0, -1 // (op 692) 

s_and_b32 s9, s12, 31 // (op 692) 
s_lshl_b32 s51, 1, s9 // (op 692) 
s_add_i32 s18, s51, s8 // (op 692) 


s_and_b32 s8, s1, s18 // (op 692) 


s_add_i32 s1, s0, s8 // (op 692) 


s_ashr_i32 s0, s1, s9 // (op 692) 


s_lshl_b32 s1, s12, 24 // (op 692) 
s_ashr_i32 s51, s1, 31 // (op 692) 

s_xor_b32 s1, s0, s51 // (op 692) 

s_sub_i32 s0, s1, s51 // (op 692) 


s_mul_hi_i32 s1, s0, s13 // (op 692) 
s_add_i32 s51, s1, s0 // (op 692) 

s_ashr_i32 s1, s51, 31 // (op 692) 
s_cmp_eq_i32 s13, 0 // (op 692) 
s_mov_b32 s8, scc // (op 692) 
s_cmp_eq_i32 s8, 0 // (op 692) 
s_cselect_b32 s9, 0, -1 // (op 692) 

s_and_b32 s8, s14, 31 // (op 692) 
s_lshl_b32 s18, 1, s8 // (op 692) 
s_add_i32 s19, s18, s9 // (op 692) 


s_and_b32 s9, s1, s19 // (op 692) 


s_add_i32 s1, s51, s9 // (op 692) 


s_ashr_i32 s9, s1, s8 // (op 692) 


s_lshl_b32 s1, s14, 24 // (op 692) 
s_ashr_i32 s8, s1, 31 // (op 692) 

s_xor_b32 s1, s9, s8 // (op 692) 

s_sub_i32 s9, s1, s8 // (op 692) 


s_add_u32 s18, s2, 16 // (op 692) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 692) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 692) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 692) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 692) 

s_mul_i32 s1, s9, s18 // (op 692) 

s_sub_i32 s51, s0, s1 // (op 692) 


s_mul_hi_i32 s0, s16, s10 // (op 692) 
s_add_i32 s1, s0, s16 // (op 692) 

s_ashr_i32 s0, s1, 31 // (op 692) 
s_cmp_eq_i32 s10, 0 // (op 692) 
s_mov_b32 s8, scc // (op 692) 
s_cmp_eq_i32 s8, 0 // (op 692) 
s_cselect_b32 s9, 0, -1 // (op 692) 

s_and_b32 s8, s49, 31 // (op 692) 
s_lshl_b32 s56, 1, s8 // (op 692) 
s_add_i32 s57, s56, s9 // (op 692) 


s_and_b32 s9, s0, s57 // (op 692) 


s_add_i32 s0, s1, s9 // (op 692) 


s_ashr_i32 s1, s0, s8 // (op 692) 


s_lshl_b32 s0, s49, 24 // (op 692) 
s_ashr_i32 s8, s0, 31 // (op 692) 

s_xor_b32 s0, s1, s8 // (op 692) 

s_sub_i32 s1, s0, s8 // (op 692) 


s_mul_i32 s0, s46, s18 // (op 692) 

s_mul_i32 s8, s50, s0 // (op 692) 

s_mul_i32 s0, s1, s8 // (op 692) 

s_sub_u32 s8, s16, s0 // (op 692) 

s_mul_hi_i32 s0, s8, s15 // (op 692) 
s_add_i32 s9, s0, s8 // (op 692) 


s_ashr_i32 s0, s9, 31 // (op 692) 
s_cmp_eq_i32 s15, 0 // (op 692) 
s_mov_b32 s8, scc // (op 692) 
s_cmp_eq_i32 s8, 0 // (op 692) 
s_cselect_b32 s18, 0, -1 // (op 692) 

s_and_b32 s8, s48, 31 // (op 692) 
s_lshl_b32 s19, 1, s8 // (op 692) 
s_add_i32 s56, s19, s18 // (op 692) 


s_and_b32 s18, s0, s56 // (op 692) 


s_add_i32 s0, s9, s18 // (op 692) 


s_ashr_i32 s9, s0, s8 // (op 692) 


s_lshl_b32 s0, s48, 24 // (op 692) 
s_ashr_i32 s8, s0, 31 // (op 692) 

s_xor_b32 s0, s9, s8 // (op 692) 

s_sub_i32 s9, s0, s8 // (op 692) 


s_cmp_eq_i32 s1, 0 // (op 692) 
s_mov_b32 s0, scc // (op 692) 

s_cmp_eq_i32 s0, 0 // (op 692) 
s_cselect_b32 s1, s9, s51 // (op 692) 



s_lshl_b32 s0, s1, 4 // (op 692) 

s_mov_b32 s9, 0 // (op 692) convert
s_mov_b32 s8, s0 // (op 692) convert

v_mul_lo_u32 v13, v1, s23 // (op 692) most significant: low of low * high
v_mul_hi_u32 v14, v1, s22 // (op 692) most significant: high of low * low
v_mul_lo_u32 v12, v1, s22 // (op 692) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 692) most significant: sum

v_mov_b32 v14, s8 // (op 692) 
v_mov_b32 v15, s9 // (op 692) 
v_add_co_u32 v12, s[0:1], v12, v14 // (op 692) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 692) most significant half




s_mul_i32 s1, s22, s27 // (op 694) most significant: low of low * high
s_mul_i32 s51, s23, s26 // (op 694) most significant: low of high * low
s_mul_i32 s0, s22, s26 // (op 694) least significant
s_mul_hi_u32 s8, s22, s26 // (op 694) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 694) most significant: sum
s_add_u32 s1, s1, s8 // (op 694) most significant: sum


s_mov_b32 s56, s24 // (op 694) reg expression
s_mov_b32 s57, s25 // (op 694) reg expression
s_mov_b32 s58, s0 // (op 694) reg expression
s_mov_b32 s59, 131072 // (op 694) call()

s_mul_hi_i32 s51, s16, s11 // (op 711) 
s_add_i32 s0, s51, s16 // (op 711) 

s_ashr_i32 s1, s0, 31 // (op 711) 
s_cmp_eq_i32 s11, 0 // (op 711) 
s_mov_b32 s51, scc // (op 711) 
s_cmp_eq_i32 s51, 0 // (op 711) 
s_cselect_b32 s8, 0, -1 // (op 711) 

s_and_b32 s9, s12, 31 // (op 711) 
s_lshl_b32 s51, 1, s9 // (op 711) 
s_add_i32 s18, s51, s8 // (op 711) 


s_and_b32 s8, s1, s18 // (op 711) 


s_add_i32 s1, s0, s8 // (op 711) 


s_ashr_i32 s0, s1, s9 // (op 711) 


s_lshl_b32 s1, s12, 24 // (op 711) 
s_ashr_i32 s51, s1, 31 // (op 711) 

s_xor_b32 s1, s0, s51 // (op 711) 

s_sub_i32 s0, s1, s51 // (op 711) 


s_mul_hi_i32 s1, s0, s13 // (op 711) 
s_add_i32 s51, s1, s0 // (op 711) 

s_ashr_i32 s1, s51, 31 // (op 711) 
s_cmp_eq_i32 s13, 0 // (op 711) 
s_mov_b32 s8, scc // (op 711) 
s_cmp_eq_i32 s8, 0 // (op 711) 
s_cselect_b32 s9, 0, -1 // (op 711) 

s_and_b32 s8, s14, 31 // (op 711) 
s_lshl_b32 s18, 1, s8 // (op 711) 
s_add_i32 s19, s18, s9 // (op 711) 


s_and_b32 s9, s1, s19 // (op 711) 


s_add_i32 s1, s51, s9 // (op 711) 


s_ashr_i32 s9, s1, s8 // (op 711) 


s_lshl_b32 s1, s14, 24 // (op 711) 
s_ashr_i32 s8, s1, 31 // (op 711) 

s_xor_b32 s1, s9, s8 // (op 711) 

s_sub_i32 s9, s1, s8 // (op 711) 


s_add_u32 s18, s2, 16 // (op 711) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 711) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 711) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 711) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 711) 

s_mul_i32 s1, s9, s18 // (op 711) 

s_sub_i32 s51, s0, s1 // (op 711) 


s_mul_hi_i32 s0, s16, s10 // (op 711) 
s_add_i32 s1, s0, s16 // (op 711) 

s_ashr_i32 s0, s1, 31 // (op 711) 
s_cmp_eq_i32 s10, 0 // (op 711) 
s_mov_b32 s8, scc // (op 711) 
s_cmp_eq_i32 s8, 0 // (op 711) 
s_cselect_b32 s9, 0, -1 // (op 711) 

s_and_b32 s8, s49, 31 // (op 711) 
s_lshl_b32 s60, 1, s8 // (op 711) 
s_add_i32 s61, s60, s9 // (op 711) 


s_and_b32 s9, s0, s61 // (op 711) 


s_add_i32 s0, s1, s9 // (op 711) 


s_ashr_i32 s1, s0, s8 // (op 711) 


s_lshl_b32 s0, s49, 24 // (op 711) 
s_ashr_i32 s8, s0, 31 // (op 711) 

s_xor_b32 s0, s1, s8 // (op 711) 

s_sub_i32 s1, s0, s8 // (op 711) 


s_mul_i32 s0, s46, s18 // (op 711) 

s_mul_i32 s8, s50, s0 // (op 711) 

s_mul_i32 s0, s1, s8 // (op 711) 

s_sub_u32 s8, s16, s0 // (op 711) 

s_mul_hi_i32 s0, s8, s15 // (op 711) 
s_add_i32 s9, s0, s8 // (op 711) 


s_ashr_i32 s0, s9, 31 // (op 711) 
s_cmp_eq_i32 s15, 0 // (op 711) 
s_mov_b32 s8, scc // (op 711) 
s_cmp_eq_i32 s8, 0 // (op 711) 
s_cselect_b32 s18, 0, -1 // (op 711) 

s_and_b32 s8, s48, 31 // (op 711) 
s_lshl_b32 s19, 1, s8 // (op 711) 
s_add_i32 s60, s19, s18 // (op 711) 


s_and_b32 s18, s0, s60 // (op 711) 


s_add_i32 s0, s9, s18 // (op 711) 


s_ashr_i32 s9, s0, s8 // (op 711) 


s_lshl_b32 s0, s48, 24 // (op 711) 
s_ashr_i32 s8, s0, 31 // (op 711) 

s_xor_b32 s0, s9, s8 // (op 711) 

s_sub_i32 s9, s0, s8 // (op 711) 


s_cmp_eq_i32 s1, 0 // (op 711) 
s_mov_b32 s0, scc // (op 711) 

s_cmp_eq_i32 s0, 0 // (op 711) 
s_cselect_b32 s1, s9, s51 // (op 711) 



s_lshl_b32 s0, s1, 4 // (op 711) 

s_mov_b32 s9, 0 // (op 711) convert
s_mov_b32 s8, s0 // (op 711) convert

v_mul_lo_u32 v15, v1, s23 // (op 711) most significant: low of low * high
v_mul_hi_u32 v20, v1, s22 // (op 711) most significant: high of low * low
v_mul_lo_u32 v14, v1, s22 // (op 711) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 711) most significant: sum

v_mov_b32 v20, s8 // (op 711) 
v_mov_b32 v21, s9 // (op 711) 
v_add_co_u32 v14, s[0:1], v14, v20 // (op 711) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 711) most significant half




s_mul_i32 s1, s22, s27 // (op 713) most significant: low of low * high
s_mul_i32 s51, s23, s26 // (op 713) most significant: low of high * low
s_mul_i32 s0, s22, s26 // (op 713) least significant
s_mul_hi_u32 s8, s22, s26 // (op 713) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 713) most significant: sum
s_add_u32 s1, s1, s8 // (op 713) most significant: sum


s_mov_b32 s56, s24 // (op 713) reg expression
s_mov_b32 s57, s25 // (op 713) reg expression
s_mov_b32 s58, s0 // (op 713) reg expression
s_mov_b32 s59, 131072 // (op 713) call()

s_mul_hi_i32 s51, s16, s11 // (op 756) 
v_lshrrev_b32 v20, 3, v1 // (op 756) 
s_add_i32 s0, s51, s16 // (op 756) 

v_lshlrev_b32 v21, 1, v20 // (op 756) 

s_ashr_i32 s1, s0, 31 // (op 756) 
v_and_b32 v20, -8, v1 // (op 756) 
s_cmp_eq_i32 s11, 0 // (op 756) 
s_mov_b32 s51, scc // (op 756) 
v_sub_u32 v22, v1, v20 // (op 756) 

s_cmp_eq_i32 s51, 0 // (op 756) 
s_cselect_b32 s8, 0, -1 // (op 756) 

s_and_b32 s9, s12, 31 // (op 756) 
s_lshl_b32 s51, 1, s9 // (op 756) 
s_add_i32 s18, s51, s8 // (op 756) 


s_and_b32 s8, s1, s18 // (op 756) 


s_add_i32 s1, s0, s8 // (op 756) 


s_ashr_i32 s0, s1, s9 // (op 756) 


s_lshl_b32 s1, s12, 24 // (op 756) 
s_ashr_i32 s51, s1, 31 // (op 756) 

s_xor_b32 s1, s0, s51 // (op 756) 

s_sub_i32 s0, s1, s51 // (op 756) 


s_mul_hi_i32 s1, s0, s13 // (op 756) 
s_add_i32 s51, s1, s0 // (op 756) 

s_ashr_i32 s1, s51, 31 // (op 756) 
s_cmp_eq_i32 s13, 0 // (op 756) 
s_mov_b32 s8, scc // (op 756) 
s_cmp_eq_i32 s8, 0 // (op 756) 
s_cselect_b32 s9, 0, -1 // (op 756) 

s_and_b32 s8, s14, 31 // (op 756) 
s_lshl_b32 s18, 1, s8 // (op 756) 
s_add_i32 s19, s18, s9 // (op 756) 


s_and_b32 s9, s1, s19 // (op 756) 


s_add_i32 s1, s51, s9 // (op 756) 


s_ashr_i32 s9, s1, s8 // (op 756) 


s_lshl_b32 s1, s14, 24 // (op 756) 
s_ashr_i32 s8, s1, 31 // (op 756) 

s_xor_b32 s1, s9, s8 // (op 756) 

s_sub_i32 s9, s1, s8 // (op 756) 


s_mul_i32 s1, s0, s46 // (op 756) 

s_sub_u32 s0, s16, s1 // (op 756) 

s_mul_i32 s1, s9, s46 // (op 756) 
s_add_u32 s1, s1, s0 // (op 756) 


s_mul_hi_i32 s0, s16, s10 // (op 756) 
s_add_i32 s51, s0, s16 // (op 756) 

s_ashr_i32 s0, s51, 31 // (op 756) 
s_cmp_eq_i32 s10, 0 // (op 756) 
s_mov_b32 s8, scc // (op 756) 
s_cmp_eq_i32 s8, 0 // (op 756) 
s_cselect_b32 s9, 0, -1 // (op 756) 

s_and_b32 s8, s49, 31 // (op 756) 
s_lshl_b32 s18, 1, s8 // (op 756) 
s_add_i32 s19, s18, s9 // (op 756) 


s_and_b32 s9, s0, s19 // (op 756) 


s_add_i32 s0, s51, s9 // (op 756) 


s_ashr_i32 s9, s0, s8 // (op 756) 


s_lshl_b32 s0, s49, 24 // (op 756) 
s_ashr_i32 s8, s0, 31 // (op 756) 

s_xor_b32 s0, s9, s8 // (op 756) 

s_sub_i32 s9, s0, s8 // (op 756) 


s_add_u32 s18, s2, 16 // (op 756) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 756) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 756) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 756) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 756) 

s_mul_i32 s0, s46, s18 // (op 756) 

s_mul_i32 s8, s50, s0 // (op 756) 

s_mul_i32 s0, s9, s8 // (op 756) 

s_sub_u32 s8, s16, s0 // (op 756) 

s_mov_b32 s0, s8 // (op 756) 
s_mul_hi_i32 s51, s0, s15 // (op 756) 
s_add_i32 s18, s51, s0 // (op 756) 


s_ashr_i32 s0, s18, 31 // (op 756) 
s_cmp_eq_i32 s15, 0 // (op 756) 
s_mov_b32 s19, scc // (op 756) 
s_cmp_eq_i32 s19, 0 // (op 756) 
s_cselect_b32 s51, 0, -1 // (op 756) 

s_and_b32 s19, s48, 31 // (op 756) 
s_lshl_b32 s60, 1, s19 // (op 756) 
s_add_i32 s61, s60, s51 // (op 756) 


s_and_b32 s51, s0, s61 // (op 756) 


s_add_i32 s0, s18, s51 // (op 756) 


s_ashr_i32 s18, s0, s19 // (op 756) 


s_lshl_b32 s0, s48, 24 // (op 756) 
s_ashr_i32 s19, s0, 31 // (op 756) 

s_xor_b32 s0, s18, s19 // (op 756) 

s_sub_i32 s18, s0, s19 // (op 756) 


s_mul_i32 s0, s47, s46 // (op 756) 
s_add_u32 s60, s6, 16 // (op 756) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 756) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 756) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 756) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 756) 

s_sub_i32 s19, s60, s0 // (op 756) 

s_mul_i32 s51, s18, s19 // (op 756) 


s_sub_u32 s18, s8, s51 // (op 756) 


s_add_u32 s8, s18, s0 // (op 756) 


s_cmp_eq_i32 s9, 0 // (op 756) 
s_mov_b32 s0, scc // (op 756) 

s_cmp_eq_i32 s0, 0 // (op 756) 
s_cselect_b32 s9, s8, s1 // (op 756) 



v_lshl_add_u32 v20, s9, 4, v21 // (op 756) 


v_mul_lo_u32 v21, v20, s28 // (op 756) 
v_add_u32 v21, v21, v22 // (op 756) 


s_mul_i32 s1, s28, s7 // (op 758) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 758) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 758) least significant
s_mul_hi_u32 s8, s28, s6 // (op 758) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 758) most significant: sum
s_add_u32 s1, s1, s8 // (op 758) most significant: sum


s_mov_b32 s60, s30 // (op 758) reg expression
s_mov_b32 s61, s31 // (op 758) reg expression
s_mov_b32 s62, s0 // (op 758) reg expression
s_mov_b32 s63, 131072 // (op 758) call()

s_mul_hi_i32 s51, s16, s11 // (op 775) 
v_lshrrev_b32 v20, 3, v1 // (op 775) 
s_add_i32 s0, s51, s16 // (op 775) 

v_lshlrev_b32 v22, 1, v20 // (op 775) 

s_ashr_i32 s1, s0, 31 // (op 775) 
v_and_b32 v20, -8, v1 // (op 775) 
s_cmp_eq_i32 s11, 0 // (op 775) 
s_mov_b32 s51, scc // (op 775) 
v_sub_u32 v23, v1, v20 // (op 775) 

s_cmp_eq_i32 s51, 0 // (op 775) 
s_cselect_b32 s8, 0, -1 // (op 775) 

s_and_b32 s9, s12, 31 // (op 775) 
s_lshl_b32 s51, 1, s9 // (op 775) 
s_add_i32 s18, s51, s8 // (op 775) 


s_and_b32 s8, s1, s18 // (op 775) 


s_add_i32 s1, s0, s8 // (op 775) 


s_ashr_i32 s0, s1, s9 // (op 775) 


s_lshl_b32 s1, s12, 24 // (op 775) 
s_ashr_i32 s51, s1, 31 // (op 775) 

s_xor_b32 s1, s0, s51 // (op 775) 

s_sub_i32 s0, s1, s51 // (op 775) 


s_mul_hi_i32 s1, s0, s13 // (op 775) 
s_add_i32 s51, s1, s0 // (op 775) 

s_ashr_i32 s1, s51, 31 // (op 775) 
s_cmp_eq_i32 s13, 0 // (op 775) 
s_mov_b32 s8, scc // (op 775) 
s_cmp_eq_i32 s8, 0 // (op 775) 
s_cselect_b32 s9, 0, -1 // (op 775) 

s_and_b32 s8, s14, 31 // (op 775) 
s_lshl_b32 s18, 1, s8 // (op 775) 
s_add_i32 s19, s18, s9 // (op 775) 


s_and_b32 s9, s1, s19 // (op 775) 


s_add_i32 s1, s51, s9 // (op 775) 


s_ashr_i32 s9, s1, s8 // (op 775) 


s_lshl_b32 s1, s14, 24 // (op 775) 
s_ashr_i32 s8, s1, 31 // (op 775) 

s_xor_b32 s1, s9, s8 // (op 775) 

s_sub_i32 s9, s1, s8 // (op 775) 


s_mul_i32 s1, s0, s46 // (op 775) 

s_sub_u32 s0, s16, s1 // (op 775) 

s_mul_i32 s1, s9, s46 // (op 775) 
s_add_u32 s1, s1, s0 // (op 775) 


s_mul_hi_i32 s0, s16, s10 // (op 775) 
s_add_i32 s51, s0, s16 // (op 775) 

s_ashr_i32 s0, s51, 31 // (op 775) 
s_cmp_eq_i32 s10, 0 // (op 775) 
s_mov_b32 s8, scc // (op 775) 
s_cmp_eq_i32 s8, 0 // (op 775) 
s_cselect_b32 s9, 0, -1 // (op 775) 

s_and_b32 s8, s49, 31 // (op 775) 
s_lshl_b32 s18, 1, s8 // (op 775) 
s_add_i32 s19, s18, s9 // (op 775) 


s_and_b32 s9, s0, s19 // (op 775) 


s_add_i32 s0, s51, s9 // (op 775) 


s_ashr_i32 s9, s0, s8 // (op 775) 


s_lshl_b32 s0, s49, 24 // (op 775) 
s_ashr_i32 s8, s0, 31 // (op 775) 

s_xor_b32 s0, s9, s8 // (op 775) 

s_sub_i32 s9, s0, s8 // (op 775) 


s_add_u32 s18, s2, 16 // (op 775) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 775) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 775) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 775) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 775) 

s_mul_i32 s0, s46, s18 // (op 775) 

s_mul_i32 s8, s50, s0 // (op 775) 

s_mul_i32 s0, s9, s8 // (op 775) 

s_sub_u32 s8, s16, s0 // (op 775) 

s_mov_b32 s0, s8 // (op 775) 
s_mul_hi_i32 s51, s0, s15 // (op 775) 
s_add_i32 s18, s51, s0 // (op 775) 


s_ashr_i32 s0, s18, 31 // (op 775) 
s_cmp_eq_i32 s15, 0 // (op 775) 
s_mov_b32 s19, scc // (op 775) 
s_cmp_eq_i32 s19, 0 // (op 775) 
s_cselect_b32 s51, 0, -1 // (op 775) 

s_and_b32 s19, s48, 31 // (op 775) 
s_lshl_b32 s64, 1, s19 // (op 775) 
s_add_i32 s65, s64, s51 // (op 775) 


s_and_b32 s51, s0, s65 // (op 775) 


s_add_i32 s0, s18, s51 // (op 775) 


s_ashr_i32 s18, s0, s19 // (op 775) 


s_lshl_b32 s0, s48, 24 // (op 775) 
s_ashr_i32 s19, s0, 31 // (op 775) 

s_xor_b32 s0, s18, s19 // (op 775) 

s_sub_i32 s18, s0, s19 // (op 775) 


s_mul_i32 s0, s47, s46 // (op 775) 
s_add_u32 s64, s6, 16 // (op 775) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 775) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 775) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 775) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 775) 

s_sub_i32 s19, s64, s0 // (op 775) 

s_mul_i32 s51, s18, s19 // (op 775) 


s_sub_u32 s18, s8, s51 // (op 775) 


s_add_u32 s8, s18, s0 // (op 775) 


s_cmp_eq_i32 s9, 0 // (op 775) 
s_mov_b32 s0, scc // (op 775) 

s_cmp_eq_i32 s0, 0 // (op 775) 
s_cselect_b32 s9, s8, s1 // (op 775) 



v_lshl_add_u32 v20, s9, 4, v22 // (op 775) 


v_mul_lo_u32 v22, v20, s28 // (op 775) 
v_add_u32 v22, v22, v23 // (op 775) 


s_mul_i32 s1, s28, s7 // (op 777) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 777) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 777) least significant
s_mul_hi_u32 s8, s28, s6 // (op 777) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 777) most significant: sum
s_add_u32 s1, s1, s8 // (op 777) most significant: sum


s_mov_b32 s60, s30 // (op 777) reg expression
s_mov_b32 s61, s31 // (op 777) reg expression
s_mov_b32 s62, s0 // (op 777) reg expression
s_mov_b32 s63, 131072 // (op 777) call()

s_mul_hi_i32 s51, s16, s11 // (op 820) 
v_and_b32 v20, -16, v1 // (op 820) 
s_add_i32 s0, s51, s16 // (op 820) 

v_sub_u32 v23, v1, v20 // (op 820) 

s_ashr_i32 s1, s0, 31 // (op 820) 
v_lshrrev_b32 v20, 4, v1 // (op 820) 
s_cmp_eq_i32 s11, 0 // (op 820) 
s_mov_b32 s51, scc // (op 820) 
v_lshlrev_b32 v24, 1, v20 // (op 820) 

s_cmp_eq_i32 s51, 0 // (op 820) 
s_cselect_b32 s8, 0, -1 // (op 820) 

s_and_b32 s9, s12, 31 // (op 820) 
s_lshl_b32 s51, 1, s9 // (op 820) 
s_add_i32 s18, s51, s8 // (op 820) 


s_and_b32 s8, s1, s18 // (op 820) 


s_add_i32 s1, s0, s8 // (op 820) 


s_ashr_i32 s0, s1, s9 // (op 820) 


s_lshl_b32 s1, s12, 24 // (op 820) 
s_ashr_i32 s51, s1, 31 // (op 820) 

s_xor_b32 s1, s0, s51 // (op 820) 

s_sub_i32 s0, s1, s51 // (op 820) 


s_mul_hi_i32 s1, s0, s13 // (op 820) 
s_add_i32 s51, s1, s0 // (op 820) 

s_ashr_i32 s1, s51, 31 // (op 820) 
s_cmp_eq_i32 s13, 0 // (op 820) 
s_mov_b32 s8, scc // (op 820) 
s_cmp_eq_i32 s8, 0 // (op 820) 
s_cselect_b32 s9, 0, -1 // (op 820) 

s_and_b32 s8, s14, 31 // (op 820) 
s_lshl_b32 s18, 1, s8 // (op 820) 
s_add_i32 s19, s18, s9 // (op 820) 


s_and_b32 s9, s1, s19 // (op 820) 


s_add_i32 s1, s51, s9 // (op 820) 


s_ashr_i32 s9, s1, s8 // (op 820) 


s_lshl_b32 s1, s14, 24 // (op 820) 
s_ashr_i32 s8, s1, 31 // (op 820) 

s_xor_b32 s1, s9, s8 // (op 820) 

s_sub_i32 s9, s1, s8 // (op 820) 


s_add_u32 s18, s2, 16 // (op 820) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 820) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 820) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 820) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 820) 

s_mul_i32 s1, s9, s18 // (op 820) 

s_sub_i32 s51, s0, s1 // (op 820) 


s_mul_hi_i32 s0, s16, s10 // (op 820) 
s_add_i32 s1, s0, s16 // (op 820) 

s_ashr_i32 s0, s1, 31 // (op 820) 
s_cmp_eq_i32 s10, 0 // (op 820) 
s_mov_b32 s8, scc // (op 820) 
s_cmp_eq_i32 s8, 0 // (op 820) 
s_cselect_b32 s9, 0, -1 // (op 820) 

s_and_b32 s8, s49, 31 // (op 820) 
s_lshl_b32 s64, 1, s8 // (op 820) 
s_add_i32 s65, s64, s9 // (op 820) 


s_and_b32 s9, s0, s65 // (op 820) 


s_add_i32 s0, s1, s9 // (op 820) 


s_ashr_i32 s1, s0, s8 // (op 820) 


s_lshl_b32 s0, s49, 24 // (op 820) 
s_ashr_i32 s8, s0, 31 // (op 820) 

s_xor_b32 s0, s1, s8 // (op 820) 

s_sub_i32 s1, s0, s8 // (op 820) 


s_mul_i32 s0, s46, s18 // (op 820) 

s_mul_i32 s8, s50, s0 // (op 820) 

s_mul_i32 s0, s1, s8 // (op 820) 

s_sub_u32 s8, s16, s0 // (op 820) 

s_mul_hi_i32 s0, s8, s15 // (op 820) 
s_add_i32 s9, s0, s8 // (op 820) 


s_ashr_i32 s0, s9, 31 // (op 820) 
s_cmp_eq_i32 s15, 0 // (op 820) 
s_mov_b32 s8, scc // (op 820) 
s_cmp_eq_i32 s8, 0 // (op 820) 
s_cselect_b32 s18, 0, -1 // (op 820) 

s_and_b32 s8, s48, 31 // (op 820) 
s_lshl_b32 s19, 1, s8 // (op 820) 
s_add_i32 s64, s19, s18 // (op 820) 


s_and_b32 s18, s0, s64 // (op 820) 


s_add_i32 s0, s9, s18 // (op 820) 


s_ashr_i32 s9, s0, s8 // (op 820) 


s_lshl_b32 s0, s48, 24 // (op 820) 
s_ashr_i32 s8, s0, 31 // (op 820) 

s_xor_b32 s0, s9, s8 // (op 820) 

s_sub_i32 s9, s0, s8 // (op 820) 


s_cmp_eq_i32 s1, 0 // (op 820) 
s_mov_b32 s0, scc // (op 820) 

s_cmp_eq_i32 s0, 0 // (op 820) 
s_cselect_b32 s1, s9, s51 // (op 820) 



v_lshl_add_u32 v20, s1, 4, v23 // (op 820) 


v_mul_lo_u32 v23, v24, s32 // (op 820) 
v_add_u32 v23, v23, v20 // (op 820) 


s_mul_i32 s1, s32, s37 // (op 822) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 822) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 822) least significant
s_mul_hi_u32 s8, s32, s36 // (op 822) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 822) most significant: sum
s_add_u32 s1, s1, s8 // (op 822) most significant: sum


s_mov_b32 s64, s34 // (op 822) reg expression
s_mov_b32 s65, s35 // (op 822) reg expression
s_mov_b32 s66, s0 // (op 822) reg expression
s_mov_b32 s67, 131072 // (op 822) call()

s_mul_hi_i32 s51, s16, s11 // (op 839) 
v_and_b32 v20, -16, v1 // (op 839) 
s_add_i32 s0, s51, s16 // (op 839) 

v_sub_u32 v24, v1, v20 // (op 839) 

s_ashr_i32 s1, s0, 31 // (op 839) 
v_lshrrev_b32 v20, 4, v1 // (op 839) 
s_cmp_eq_i32 s11, 0 // (op 839) 
s_mov_b32 s51, scc // (op 839) 
v_lshlrev_b32 v25, 1, v20 // (op 839) 

s_cmp_eq_i32 s51, 0 // (op 839) 
s_cselect_b32 s8, 0, -1 // (op 839) 

s_and_b32 s9, s12, 31 // (op 839) 
s_lshl_b32 s51, 1, s9 // (op 839) 
s_add_i32 s18, s51, s8 // (op 839) 


s_and_b32 s8, s1, s18 // (op 839) 


s_add_i32 s1, s0, s8 // (op 839) 


s_ashr_i32 s0, s1, s9 // (op 839) 


s_lshl_b32 s1, s12, 24 // (op 839) 
s_ashr_i32 s51, s1, 31 // (op 839) 

s_xor_b32 s1, s0, s51 // (op 839) 

s_sub_i32 s0, s1, s51 // (op 839) 


s_mul_hi_i32 s1, s0, s13 // (op 839) 
s_add_i32 s51, s1, s0 // (op 839) 

s_ashr_i32 s1, s51, 31 // (op 839) 
s_cmp_eq_i32 s13, 0 // (op 839) 
s_mov_b32 s8, scc // (op 839) 
s_cmp_eq_i32 s8, 0 // (op 839) 
s_cselect_b32 s9, 0, -1 // (op 839) 

s_and_b32 s8, s14, 31 // (op 839) 
s_lshl_b32 s18, 1, s8 // (op 839) 
s_add_i32 s19, s18, s9 // (op 839) 


s_and_b32 s9, s1, s19 // (op 839) 


s_add_i32 s1, s51, s9 // (op 839) 


s_ashr_i32 s9, s1, s8 // (op 839) 


s_lshl_b32 s1, s14, 24 // (op 839) 
s_ashr_i32 s8, s1, 31 // (op 839) 

s_xor_b32 s1, s9, s8 // (op 839) 

s_sub_i32 s9, s1, s8 // (op 839) 


s_add_u32 s18, s2, 16 // (op 839) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 839) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 839) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 839) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 839) 

s_mul_i32 s1, s9, s18 // (op 839) 

s_sub_i32 s51, s0, s1 // (op 839) 


s_mul_hi_i32 s0, s16, s10 // (op 839) 
s_add_i32 s1, s0, s16 // (op 839) 

s_ashr_i32 s0, s1, 31 // (op 839) 
s_cmp_eq_i32 s10, 0 // (op 839) 
s_mov_b32 s8, scc // (op 839) 
s_cmp_eq_i32 s8, 0 // (op 839) 
s_cselect_b32 s9, 0, -1 // (op 839) 

s_and_b32 s8, s49, 31 // (op 839) 
s_lshl_b32 s68, 1, s8 // (op 839) 
s_add_i32 s69, s68, s9 // (op 839) 


s_and_b32 s9, s0, s69 // (op 839) 


s_add_i32 s0, s1, s9 // (op 839) 


s_ashr_i32 s1, s0, s8 // (op 839) 


s_lshl_b32 s0, s49, 24 // (op 839) 
s_ashr_i32 s8, s0, 31 // (op 839) 

s_xor_b32 s0, s1, s8 // (op 839) 

s_sub_i32 s1, s0, s8 // (op 839) 


s_mul_i32 s0, s46, s18 // (op 839) 

s_mul_i32 s8, s50, s0 // (op 839) 

s_mul_i32 s0, s1, s8 // (op 839) 

s_sub_u32 s8, s16, s0 // (op 839) 

s_mul_hi_i32 s0, s8, s15 // (op 839) 
s_add_i32 s9, s0, s8 // (op 839) 


s_ashr_i32 s0, s9, 31 // (op 839) 
s_cmp_eq_i32 s15, 0 // (op 839) 
s_mov_b32 s8, scc // (op 839) 
s_cmp_eq_i32 s8, 0 // (op 839) 
s_cselect_b32 s18, 0, -1 // (op 839) 

s_and_b32 s8, s48, 31 // (op 839) 
s_lshl_b32 s19, 1, s8 // (op 839) 
s_add_i32 s68, s19, s18 // (op 839) 


s_and_b32 s18, s0, s68 // (op 839) 


s_add_i32 s0, s9, s18 // (op 839) 


s_ashr_i32 s9, s0, s8 // (op 839) 


s_lshl_b32 s0, s48, 24 // (op 839) 
s_ashr_i32 s8, s0, 31 // (op 839) 

s_xor_b32 s0, s9, s8 // (op 839) 

s_sub_i32 s9, s0, s8 // (op 839) 


s_cmp_eq_i32 s1, 0 // (op 839) 
s_mov_b32 s0, scc // (op 839) 

s_cmp_eq_i32 s0, 0 // (op 839) 
s_cselect_b32 s1, s9, s51 // (op 839) 



v_lshl_add_u32 v20, s1, 4, v24 // (op 839) 


v_mul_lo_u32 v24, v25, s32 // (op 839) 
v_add_u32 v24, v24, v20 // (op 839) 


s_mul_i32 s1, s32, s37 // (op 841) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 841) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 841) least significant
s_mul_hi_u32 s8, s32, s36 // (op 841) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 841) most significant: sum
s_add_u32 s1, s1, s8 // (op 841) most significant: sum


s_mov_b32 s64, s34 // (op 841) reg expression
s_mov_b32 s65, s35 // (op 841) reg expression
s_mov_b32 s66, s0 // (op 841) reg expression
s_mov_b32 s67, 131072 // (op 841) call()

v_lshlrev_b32 v20, 4, v1 // (op 957) 
s_nop 0
v_readfirstlane_b32 s51, v20 // (op 957) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[22:23], 6 // (op 577) 
v_mov_b64 v[26:27], s[0:1] // (op 577) convert


v_mov_b64 v[28:29], v[12:13] // (op 577) reg expression
s_mov_b32 m0, s51 // (op 577) reg expression
buffer_load_dwordx4 v28, s[56:59], 0 offen offset: 0 lds // (op 577) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 577) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 577) most significant half

s_add_u32 m0, m0, 1024 // (op 577) 
buffer_load_dwordx4 v28, s[56:59], 0 offen offset: 0 lds // (op 577) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 577) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 577) most significant half

s_add_u32 m0, m0, 1024 // (op 577) 
buffer_load_dwordx4 v28, s[56:59], 0 offen offset: 0 lds // (op 577) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 577) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 577) most significant half

s_add_u32 m0, m0, 1024 // (op 577) 
buffer_load_dwordx4 v28, s[56:59], 0 offen offset: 0 lds // (op 577) Load value direct to lds




v_mov_b64 v[26:27], s[32:33] // (op 431) convert

v_mov_b32 v20, v23 // (op 431) reg expression
buffer_load_ubyte v28, v20, s[64:67], 0 offen offset: 0 // (op 431) Load value
v_add_u32 v20, v20, v26 // (op 431) 
buffer_load_ubyte v29, v20, s[64:67], 0 offen offset: 0 // (op 431) Load value


v_and_b32 v20, -16, v1 // (op 905) 
v_sub_u32 v25, v1, v20 // (op 905) 

v_lshlrev_b32 v20, 4, v25 // (op 905) 

v_lshrrev_b32 v25, 4, v1 // (op 905) 
v_lshl_add_u32 v26, v25, 8, v20 // (op 905) 


s_nop 0
v_readfirstlane_b32 s51, v26 // (op 905) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[4:5], 2 // (op 582) 
v_mov_b64 v[26:27], s[0:1] // (op 582) convert


v_mov_b64 v[30:31], v[16:17] // (op 582) 
s_add_u32 s0, 4096, s51 // (op 582) 
s_mov_b32 m0, s0 // (op 582) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 582) Load value direct to lds
v_add_co_u32 v30, s[8:9], v30, v26 // (op 582) least significant half
v_addc_co_u32 v31, s[8:9], 0, v31, s[8:9] // (op 582) most significant half

s_add_u32 m0, m0, 1024 // (op 582) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 582) Load value direct to lds
v_add_co_u32 v30, s[8:9], v30, v26 // (op 582) least significant half
v_addc_co_u32 v31, s[8:9], 0, v31, s[8:9] // (op 582) most significant half

s_add_u32 m0, m0, 1024 // (op 582) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 582) Load value direct to lds
v_add_co_u32 v30, s[8:9], v30, v26 // (op 582) least significant half
v_addc_co_u32 v31, s[8:9], 0, v31, s[8:9] // (op 582) most significant half

s_add_u32 m0, m0, 1024 // (op 582) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 582) Load value direct to lds





v_mov_b64 v[26:27], s[28:29] // (op 437) convert

v_mov_b32 v20, v21 // (op 437) 
buffer_load_ubyte v30, v20, s[60:63], 0 offen offset: 0 // (op 437) Load value
v_add_u32 v20, v20, v26 // (op 437) 
buffer_load_ubyte v31, v20, s[60:63], 0 offen offset: 0 // (op 437) Load value


v_lshlrev_b32 v20, 4, v1 // (op 1175) 
s_nop 0
v_readfirstlane_b32 s51, v20 // (op 1175) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[22:23], 6 // (op 587) 
v_mov_b64 v[26:27], s[0:1] // (op 587) convert


s_lshl_b64 s[0:1], s[22:23], 8 // (op 587) 
v_mov_b32 v20, s0 // (op 587) 
v_mov_b32 v25, s1 // (op 587) 
v_add_co_u32 v32, s[8:9], v12, v20 // (op 587) least significant half
v_addc_co_u32 v33, s[8:9], v13, v25, s[8:9] // (op 587) most significant half




s_add_u32 s0, 8192, s51 // (op 587) 
s_mov_b32 m0, s0 // (op 587) reg expression
buffer_load_dwordx4 v32, s[56:59], 0 offen offset: 0 lds // (op 587) Load value direct to lds
v_add_co_u32 v32, s[8:9], v32, v26 // (op 587) least significant half
v_addc_co_u32 v33, s[8:9], 0, v33, s[8:9] // (op 587) most significant half

s_add_u32 m0, m0, 1024 // (op 587) 
buffer_load_dwordx4 v32, s[56:59], 0 offen offset: 0 lds // (op 587) Load value direct to lds
v_add_co_u32 v32, s[8:9], v32, v26 // (op 587) least significant half
v_addc_co_u32 v33, s[8:9], 0, v33, s[8:9] // (op 587) most significant half

s_add_u32 m0, m0, 1024 // (op 587) 
buffer_load_dwordx4 v32, s[56:59], 0 offen offset: 0 lds // (op 587) Load value direct to lds
v_add_co_u32 v32, s[8:9], v32, v26 // (op 587) least significant half
v_addc_co_u32 v33, s[8:9], 0, v33, s[8:9] // (op 587) most significant half

s_add_u32 m0, m0, 1024 // (op 587) 
buffer_load_dwordx4 v32, s[56:59], 0 offen offset: 0 lds // (op 587) Load value direct to lds






v_mov_b64 v[12:13], s[32:33] // (op 443) convert

s_lshl_b32 s51, s32, 3 // (op 443) 
v_add_u32 v20, v23, s51 // (op 443) 

buffer_load_ubyte v26, v20, s[64:67], 0 offen offset: 0 // (op 443) Load value
v_add_u32 v20, v20, v12 // (op 443) 
buffer_load_ubyte v27, v20, s[64:67], 0 offen offset: 0 // (op 443) Load value



v_and_b32 v20, -16, v1 // (op 1136) 
v_sub_u32 v23, v1, v20 // (op 1136) 

v_lshlrev_b32 v20, 4, v23 // (op 1136) 

v_lshrrev_b32 v23, 4, v1 // (op 1136) 
v_lshl_add_u32 v25, v23, 8, v20 // (op 1136) 


s_nop 0
v_readfirstlane_b32 s51, v25 // (op 1136) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[4:5], 2 // (op 589) 
v_mov_b64 v[12:13], s[0:1] // (op 589) convert


v_mov_b64 v[32:33], v[16:17] // (op 589) 
s_add_u32 s0, 12288, s51 // (op 589) 
s_mov_b32 m0, s0 // (op 589) reg expression
s_mov_b32 s1, 256 // (op 589) call()
buffer_load_dwordx4 v32, s[52:55], s1 offen offset: 0 lds // (op 589) Load value direct to lds
v_add_co_u32 v32, s[8:9], v32, v12 // (op 589) least significant half
v_addc_co_u32 v33, s[8:9], 0, v33, s[8:9] // (op 589) most significant half

s_add_u32 m0, m0, 1024 // (op 589) 
buffer_load_dwordx4 v32, s[52:55], s1 offen offset: 0 lds // (op 589) Load value direct to lds
v_add_co_u32 v32, s[8:9], v32, v12 // (op 589) least significant half
v_addc_co_u32 v33, s[8:9], 0, v33, s[8:9] // (op 589) most significant half

s_add_u32 m0, m0, 1024 // (op 589) 
buffer_load_dwordx4 v32, s[52:55], s1 offen offset: 0 lds // (op 589) Load value direct to lds
v_add_co_u32 v32, s[8:9], v32, v12 // (op 589) least significant half
v_addc_co_u32 v33, s[8:9], 0, v33, s[8:9] // (op 589) most significant half

s_add_u32 m0, m0, 1024 // (op 589) 
buffer_load_dwordx4 v32, s[52:55], s1 offen offset: 0 lds // (op 589) Load value direct to lds







v_mov_b64 v[12:13], s[28:29] // (op 449) convert

v_mov_b32 v20, v21 // (op 449) 
buffer_load_ubyte v16, v20, s[60:63], 0 offen offset:8 // (op 449) Load value
v_add_u32 v20, v20, v12 // (op 449) 
buffer_load_ubyte v17, v20, s[60:63], 0 offen offset:8 // (op 449) Load value



v_and_b32 v23, -16, v1 // (op 1063) 
v_sub_u32 v25, v1, v23 // (op 1063) 

v_lshrrev_b32 v23, 4, v1 // (op 1063) 
v_lshl_add_u32 v12, v23, 5, v25 // (op 1063) 





s_waitcnt vmcnt(19)
ds_write_b8 v12, v28 offset:16384 // (op 452) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v12, v29 offset:16400 // (op 452) Store local data 

v_and_b32 v23, -8, v1 // (op 1010) 
v_sub_u32 v25, v1, v23 // (op 1010) 

v_lshrrev_b32 v23, 3, v1 // (op 1010) 
v_lshl_add_u32 v12, v23, 4, v25 // (op 1010) 





s_waitcnt vmcnt(13)
ds_write_b8 v12, v30 offset:16512 // (op 454) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v12, v31 offset:16520 // (op 454) Store local data 

s_barrier  // (op 1302) 



ds_read_b64_tr_b8 v[32:33], v0 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:128 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:1024 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:1152 // (op 474) Transpose load local data 



ds_read_b64_tr_b8 v[40:41], v0 offset:2048 // (op 477) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:2176 // (op 477) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:3072 // (op 477) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:3200 // (op 477) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1336) 


ds_read_u8 v23, v7 offset:16384 // (op 486) Load local data 


ds_read_u8 v25, v7 offset:16448 // (op 489) Load local data 
s_barrier  // (op 1286) 



ds_read_b128 v[48:51], v3 offset:4096 // (op 468) Load local data 
ds_read_b128 v[52:55], v3 offset:4160 // (op 468) Load local data 



ds_read_b128 v[56:59], v3 offset:4224 // (op 471) Load local data 
ds_read_b128 v[60:63], v3 offset:4288 // (op 471) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1319) 



ds_read_u8 v12, v4 offset:16512 // (op 480) Load local data 



ds_read_u8 v13, v4 offset:16516 // (op 483) Load local data 
v_and_b32 v20, -16, v1 // (op 877) 
v_sub_u32 v21, v1, v20 // (op 877) 

v_lshlrev_b32 v20, 4, v21 // (op 877) 

v_lshrrev_b32 v21, 4, v1 // (op 877) 
v_lshl_add_u32 v64, v21, 8, v20 // (op 877) 


s_nop 0
v_readfirstlane_b32 s51, v64 // (op 877) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v20, 4, v1 // (op 929) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 929) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 981) 
v_sub_u32 v21, v1, v20 // (op 981) 

v_lshrrev_b32 v20, 3, v1 // (op 981) 
v_lshl_add_u32 v64, v20, 4, v21 // (op 981) 


v_and_b32 v20, -16, v1 // (op 1034) 
v_sub_u32 v21, v1, v20 // (op 1034) 

v_lshrrev_b32 v20, 4, v1 // (op 1034) 
v_lshl_add_u32 v65, v20, 5, v21 // (op 1034) 


v_and_b32 v20, -16, v1 // (op 1121) 
v_sub_u32 v21, v1, v20 // (op 1121) 

v_lshlrev_b32 v20, 4, v21 // (op 1121) 

v_lshrrev_b32 v21, 4, v1 // (op 1121) 
v_lshl_add_u32 v66, v21, 8, v20 // (op 1121) 


s_nop 0
v_readfirstlane_b32 s1, v66 // (op 1121) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v20, 4, v1 // (op 1160) 
s_nop 0
v_readfirstlane_b32 s8, v20 // (op 1160) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 1199) 
v_sub_u32 v21, v1, v20 // (op 1199) 

v_lshrrev_b32 v20, 3, v1 // (op 1199) 
v_lshl_add_u32 v66, v20, 4, v21 // (op 1199) 


v_and_b32 v20, -16, v1 // (op 1223) 
v_sub_u32 v21, v1, v20 // (op 1223) 

v_lshrrev_b32 v20, 4, v1 // (op 1223) 
v_lshl_add_u32 v67, v20, 5, v21 // (op 1223) 


s_mov_b32 s9, 0 // (op 61) call()
s_ashr_i64 s[18:19], s[26:27], 8 // (op 60) 
s_mov_b32 s68, s18 // (op 60) convert

s_and_b32 s18, s68, 4294967294 // (op 60) 

s_cmp_lt_u32 s9, s18 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_60 if false)
RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_60:
s_add_u32 s9, s9, 2 // (op 62) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v12, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 90) 
s_barrier  // (op 1305) 

s_lshl_b64 s[18:19], s[22:23], 6 // (op 549) 
v_mov_b64 v[20:21], s[18:19] // (op 549) convert


v_mov_b64 v[68:69], v[14:15] // (op 549) 
s_lshl_b64 s[18:19], s[22:23], 8 // (op 549) 
s_lshl_b64 s[68:69], s[18:19], 1 // (op 549) 

v_mov_b32 v70, s68 // (op 549) 
v_mov_b32 v71, s69 // (op 549) 
v_add_co_u32 v72, s[18:19], v68, v70 // (op 549) least significant half
v_addc_co_u32 v73, s[18:19], v69, v71, s[18:19] // (op 549) most significant half




s_mov_b32 m0, s0 // (op 549) reg expression
buffer_load_dwordx4 v72, s[56:59], 0 offen offset: 0 lds // (op 549) Load value direct to lds
v_add_co_u32 v72, s[18:19], v72, v20 // (op 549) least significant half
v_addc_co_u32 v73, s[18:19], 0, v73, s[18:19] // (op 549) most significant half

s_add_u32 m0, m0, 1024 // (op 549) 
buffer_load_dwordx4 v72, s[56:59], 0 offen offset: 0 lds // (op 549) Load value direct to lds
v_add_co_u32 v72, s[18:19], v72, v20 // (op 549) least significant half
v_addc_co_u32 v73, s[18:19], 0, v73, s[18:19] // (op 549) most significant half

s_add_u32 m0, m0, 1024 // (op 549) 
buffer_load_dwordx4 v72, s[56:59], 0 offen offset: 0 lds // (op 549) Load value direct to lds
v_add_co_u32 v72, s[18:19], v72, v20 // (op 549) least significant half
v_addc_co_u32 v73, s[18:19], 0, v73, s[18:19] // (op 549) most significant half

s_add_u32 m0, m0, 1024 // (op 549) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[40:47], a[0:3], v13, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 107) 
buffer_load_dwordx4 v72, s[56:59], 0 offen offset: 0 lds // (op 549) Load value direct to lds



v_mov_b64 v[20:21], s[32:33] // (op 25) convert

v_mov_b32 v70, v24 // (op 25) 
s_lshl_b32 s18, s32, 4 // (op 25) 
v_add_u32 v71, v70, s18 // (op 25) 

buffer_load_ubyte v28, v71, s[64:67], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v71, v71, v20 // (op 25) 
buffer_load_ubyte v29, v71, s[64:67], 0 offen offset: 0 // (op 25) Load value


s_barrier  // (op 1289) 

s_lshl_b64 s[18:19], s[4:5], 2 // (op 544) 
v_mov_b64 v[20:21], s[18:19] // (op 544) convert


v_mov_b64 v[72:73], v[18:19] // (op 544) 
v_mov_b64 v[74:75], v[72:73] // (op 544) 
s_add_u32 s18, 4096, s51 // (op 544) 
s_mov_b32 m0, s18 // (op 544) reg expression
s_mov_b32 s19, 512 // (op 544) call()
buffer_load_dwordx4 v74, s[52:55], s19 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v74, s[68:69], v74, v20 // (op 544) least significant half
v_addc_co_u32 v75, s[68:69], 0, v75, s[68:69] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v74, s[52:55], s19 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v74, s[68:69], v74, v20 // (op 544) least significant half
v_addc_co_u32 v75, s[68:69], 0, v75, s[68:69] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v74, s[52:55], s19 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v74, s[68:69], v74, v20 // (op 544) least significant half
v_addc_co_u32 v75, s[68:69], 0, v75, s[68:69] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v74, s[52:55], s19 offen offset: 0 lds // (op 544) Load value direct to lds





v_mov_b64 v[20:21], s[28:29] // (op 17) convert

v_mov_b32 v71, v22 // (op 17) 
v_mov_b32 v74, v71 // (op 17) 
buffer_load_ubyte v30, v74, s[60:63], 0 offen offset:16 // (op 17) Load value
v_add_u32 v74, v74, v20 // (op 17) 
buffer_load_ubyte v31, v74, s[60:63], 0 offen offset:16 // (op 17) Load value


s_barrier  // (op 1375) 



s_waitcnt vmcnt(19)
ds_write_b8 v67, v26 offset:16640 // (op 308) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v67, v27 offset:16656 // (op 308) Store local data 
s_barrier  // (op 1369) 



s_waitcnt vmcnt(13)
ds_write_b8 v66, v16 offset:16768 // (op 307) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v66, v17 offset:16776 // (op 307) Store local data 
s_barrier  // (op 1359) 



ds_read_b64_tr_b8 v[32:33], v5 offset:8192 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v5 offset:8320 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v5 offset:9216 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v5 offset:9344 // (op 291) Transpose load local data 



ds_read_b64_tr_b8 v[40:41], v5 offset:10240 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v5 offset:10368 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v5 offset:11264 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v5 offset:11392 // (op 300) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1372) 


ds_read_u8 v23, v11 offset:16640 // (op 295) Load local data 


ds_read_u8 v25, v11 offset:16704 // (op 304) Load local data 
s_barrier  // (op 1353) 
s_barrier  // (op 1432) 



ds_read_b128 v[48:51], v6 offset:12288 // (op 289) Load local data 
ds_read_b128 v[52:55], v6 offset:12352 // (op 289) Load local data 



ds_read_b128 v[56:59], v6 offset:12416 // (op 298) Load local data 
ds_read_b128 v[60:63], v6 offset:12480 // (op 298) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1366) 



ds_read_u8 v12, v8 offset:16768 // (op 293) Load local data 



ds_read_u8 v13, v8 offset:16772 // (op 302) Load local data 
s_barrier  // (op 1428) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v12, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 296) 
s_barrier  // (op 1363) 

s_lshl_b64 s[18:19], s[22:23], 6 // (op 572) 
v_mov_b64 v[20:21], s[18:19] // (op 572) convert


s_lshl_b64 s[18:19], s[22:23], 8 // (op 572) 
s_mul_i32 s69, 3, s19 // (op 572) most significant: low of low * high
s_mul_i32 s68, 3, s18 // (op 572) least significant
s_mul_hi_u32 s70, 3, s18 // (op 572) most significant: high of low * low
s_add_u32 s69, s69, s70 // (op 572) most significant: sum


v_mov_b32 v74, s68 // (op 572) 
v_mov_b32 v75, s69 // (op 572) 
v_add_co_u32 v76, s[18:19], v68, v74 // (op 572) least significant half
v_addc_co_u32 v77, s[18:19], v69, v75, s[18:19] // (op 572) most significant half




s_add_u32 s18, 8192, s8 // (op 572) 
s_nop 0
s_mov_b32 m0, s18 // (op 572) reg expression
buffer_load_dwordx4 v76, s[56:59], 0 offen offset: 0 lds // (op 572) Load value direct to lds
v_add_co_u32 v76, s[68:69], v76, v20 // (op 572) least significant half
v_addc_co_u32 v77, s[68:69], 0, v77, s[68:69] // (op 572) most significant half

s_add_u32 m0, m0, 1024 // (op 572) 
buffer_load_dwordx4 v76, s[56:59], 0 offen offset: 0 lds // (op 572) Load value direct to lds
v_add_co_u32 v76, s[68:69], v76, v20 // (op 572) least significant half
v_addc_co_u32 v77, s[68:69], 0, v77, s[68:69] // (op 572) most significant half

s_add_u32 m0, m0, 1024 // (op 572) 
buffer_load_dwordx4 v76, s[56:59], 0 offen offset: 0 lds // (op 572) Load value direct to lds
v_add_co_u32 v76, s[68:69], v76, v20 // (op 572) least significant half
v_addc_co_u32 v77, s[68:69], 0, v77, s[68:69] // (op 572) most significant half

s_add_u32 m0, m0, 1024 // (op 572) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[40:47], a[0:3], v13, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 305) 
buffer_load_dwordx4 v76, s[56:59], 0 offen offset: 0 lds // (op 572) Load value direct to lds





v_mov_b64 v[20:21], s[32:33] // (op 311) convert

s_lshl_b32 s18, s32, 3 // (op 311) 
s_mul_i32 s19, 3, s18 // (op 311) 

v_add_u32 v68, v70, s19 // (op 311) 

buffer_load_ubyte v26, v68, s[64:67], 0 offen offset: 0 // (op 311) Load value
v_add_u32 v68, v68, v20 // (op 311) 
s_lshl_b64 s[18:19], s[22:23], 8 // (op 703) 
buffer_load_ubyte v27, v68, s[64:67], 0 offen offset: 0 // (op 311) Load value


s_lshl_b64 s[68:69], s[18:19], 1 // (op 703) 


v_mov_b32 v20, s68 // (op 703) 
s_barrier  // (op 1356) 
s_lshl_b32 s18, s32, 4 // (op 831) 

v_add_u32 v24, v24, s18 // (op 831) 

v_mov_b32 v21, s69 // (op 703) 
s_lshl_b64 s[18:19], s[4:5], 2 // (op 567) 
v_mov_b64 v[68:69], s[18:19] // (op 567) convert


v_mov_b64 v[74:75], v[72:73] // (op 567) 
s_add_u32 s18, 12288, s1 // (op 567) 
s_mov_b32 m0, s18 // (op 567) reg expression
s_mov_b32 s19, 768 // (op 567) call()
v_add_co_u32 v14, s[70:71], v14, v20 // (op 703) least significant half
v_addc_co_u32 v15, s[70:71], v15, v21, s[70:71] // (op 703) most significant half




buffer_load_dwordx4 v74, s[52:55], s19 offen offset: 0 lds // (op 567) Load value direct to lds
v_add_co_u32 v74, s[68:69], v74, v68 // (op 567) least significant half
v_addc_co_u32 v75, s[68:69], 0, v75, s[68:69] // (op 567) most significant half

s_add_u32 m0, m0, 1024 // (op 567) 
buffer_load_dwordx4 v74, s[52:55], s19 offen offset: 0 lds // (op 567) Load value direct to lds
v_add_co_u32 v74, s[68:69], v74, v68 // (op 567) least significant half
v_addc_co_u32 v75, s[68:69], 0, v75, s[68:69] // (op 567) most significant half

s_add_u32 m0, m0, 1024 // (op 567) 
buffer_load_dwordx4 v74, s[52:55], s19 offen offset: 0 lds // (op 567) Load value direct to lds
v_add_co_u32 v74, s[68:69], v74, v68 // (op 567) least significant half
v_addc_co_u32 v75, s[68:69], 0, v75, s[68:69] // (op 567) most significant half

s_add_u32 m0, m0, 1024 // (op 567) 
buffer_load_dwordx4 v74, s[52:55], s19 offen offset: 0 lds // (op 567) Load value direct to lds






v_add_co_u32 v18, vcc, 512, v18 // (op 639) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 639) most significant half
v_mov_b64 v[20:21], s[28:29] // (op 310) convert

v_mov_b32 v68, v71 // (op 310) 
buffer_load_ubyte v16, v68, s[60:63], 0 offen offset:24 // (op 310) Load value
v_add_u32 v68, v68, v20 // (op 310) 
buffer_load_ubyte v17, v68, s[60:63], 0 offen offset:24 // (op 310) Load value



s_barrier  // (op 1339) 
v_add_u32 v22, 16, v22 // (op 767) 



s_waitcnt vmcnt(19)
ds_write_b8 v65, v28 offset:16384 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v65, v29 offset:16400 // (op 30) Store local data 
s_barrier  // (op 1322) 



s_waitcnt vmcnt(13)
ds_write_b8 v64, v30 offset:16512 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v64, v31 offset:16520 // (op 21) Store local data 
s_barrier  // (op 1311) 



ds_read_b64_tr_b8 v[32:33], v0 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:128 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:1024 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:1152 // (op 77) Transpose load local data 



ds_read_b64_tr_b8 v[40:41], v0 offset:2048 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:2176 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:3072 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:3200 // (op 97) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1346) 


ds_read_u8 v23, v7 offset:16384 // (op 87) Load local data 


ds_read_u8 v25, v7 offset:16448 // (op 105) Load local data 
s_barrier  // (op 1295) 



ds_read_b128 v[48:51], v3 offset:4096 // (op 72) Load local data 
ds_read_b128 v[52:55], v3 offset:4160 // (op 72) Load local data 



ds_read_b128 v[56:59], v3 offset:4224 // (op 93) Load local data 
ds_read_b128 v[60:63], v3 offset:4288 // (op 93) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1329) 



ds_read_u8 v12, v4 offset:16512 // (op 82) Load local data 



ds_read_u8 v13, v4 offset:16516 // (op 101) Load local data 
s_ashr_i64 s[18:19], s[26:27], 8 // (op 60) 
s_mov_b32 s68, s18 // (op 60) convert

s_and_b32 s18, s68, 4294967294 // (op 60) 

s_cmp_lt_u32 s9, s18 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_60 if true)
RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_60:















s_barrier  // (op 1388) 
s_mul_hi_i32 s51, s16, s11 // (op 602) 
v_lshrrev_b32 v24, 4, v1 // (op 602) 
s_add_i32 s0, s51, s16 // (op 602) 

v_and_b32 v14, -16, v1 // (op 602) 
s_ashr_i32 s1, s0, 31 // (op 602) 
v_sub_u32 v15, v1, v14 // (op 602) 

s_cmp_eq_i32 s11, 0 // (op 602) 
s_mov_b32 s51, scc // (op 602) 
v_lshlrev_b32 v14, 4, v15 // (op 602) 

s_cmp_eq_i32 s51, 0 // (op 602) 
s_cselect_b32 s8, 0, -1 // (op 602) 

s_and_b32 s9, s12, 31 // (op 602) 
s_lshl_b32 s51, 1, s9 // (op 602) 
s_add_i32 s18, s51, s8 // (op 602) 


s_and_b32 s8, s1, s18 // (op 602) 


s_add_i32 s1, s0, s8 // (op 602) 


s_ashr_i32 s0, s1, s9 // (op 602) 


s_lshl_b32 s1, s12, 24 // (op 602) 
s_ashr_i32 s51, s1, 31 // (op 602) 

s_xor_b32 s1, s0, s51 // (op 602) 

s_sub_i32 s0, s1, s51 // (op 602) 


s_mul_hi_i32 s1, s0, s13 // (op 602) 
s_add_i32 s51, s1, s0 // (op 602) 

s_ashr_i32 s1, s51, 31 // (op 602) 
s_cmp_eq_i32 s13, 0 // (op 602) 
s_mov_b32 s8, scc // (op 602) 
s_cmp_eq_i32 s8, 0 // (op 602) 
s_cselect_b32 s9, 0, -1 // (op 602) 

s_and_b32 s8, s14, 31 // (op 602) 
s_lshl_b32 s18, 1, s8 // (op 602) 
s_add_i32 s19, s18, s9 // (op 602) 


s_and_b32 s9, s1, s19 // (op 602) 


s_add_i32 s1, s51, s9 // (op 602) 


s_ashr_i32 s9, s1, s8 // (op 602) 


s_lshl_b32 s1, s14, 24 // (op 602) 
s_ashr_i32 s8, s1, 31 // (op 602) 

s_xor_b32 s1, s9, s8 // (op 602) 

s_sub_i32 s9, s1, s8 // (op 602) 


s_mul_i32 s1, s0, s46 // (op 602) 

s_sub_u32 s0, s16, s1 // (op 602) 

s_mul_i32 s1, s9, s46 // (op 602) 
s_add_u32 s1, s1, s0 // (op 602) 


s_mul_hi_i32 s0, s16, s10 // (op 602) 
s_add_i32 s51, s0, s16 // (op 602) 

s_ashr_i32 s0, s51, 31 // (op 602) 
s_cmp_eq_i32 s10, 0 // (op 602) 
s_mov_b32 s8, scc // (op 602) 
s_cmp_eq_i32 s8, 0 // (op 602) 
s_cselect_b32 s9, 0, -1 // (op 602) 

s_and_b32 s8, s49, 31 // (op 602) 
s_lshl_b32 s18, 1, s8 // (op 602) 
s_add_i32 s19, s18, s9 // (op 602) 


s_and_b32 s9, s0, s19 // (op 602) 


s_add_i32 s0, s51, s9 // (op 602) 


s_ashr_i32 s9, s0, s8 // (op 602) 


s_lshl_b32 s0, s49, 24 // (op 602) 
s_ashr_i32 s8, s0, 31 // (op 602) 

s_xor_b32 s0, s9, s8 // (op 602) 

s_sub_i32 s9, s0, s8 // (op 602) 


s_add_u32 s18, s2, 16 // (op 602) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 602) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 602) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 602) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 602) 

s_mul_i32 s0, s46, s18 // (op 602) 

s_mul_i32 s8, s50, s0 // (op 602) 

s_mul_i32 s0, s9, s8 // (op 602) 

s_sub_u32 s8, s16, s0 // (op 602) 

s_mov_b32 s0, s8 // (op 602) 
s_mul_hi_i32 s51, s0, s15 // (op 602) 
s_add_i32 s18, s51, s0 // (op 602) 


s_ashr_i32 s0, s18, 31 // (op 602) 
s_cmp_eq_i32 s15, 0 // (op 602) 
s_mov_b32 s19, scc // (op 602) 
s_cmp_eq_i32 s19, 0 // (op 602) 
s_cselect_b32 s51, 0, -1 // (op 602) 

s_and_b32 s19, s48, 31 // (op 602) 
s_lshl_b32 s68, 1, s19 // (op 602) 
s_add_i32 s69, s68, s51 // (op 602) 


s_and_b32 s51, s0, s69 // (op 602) 


s_add_i32 s0, s18, s51 // (op 602) 


s_ashr_i32 s18, s0, s19 // (op 602) 


s_lshl_b32 s0, s48, 24 // (op 602) 
s_ashr_i32 s19, s0, 31 // (op 602) 

s_xor_b32 s0, s18, s19 // (op 602) 

s_sub_i32 s18, s0, s19 // (op 602) 


s_mul_i32 s0, s47, s46 // (op 602) 
s_add_u32 s68, s6, 16 // (op 602) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 602) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 602) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 602) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 602) 

s_sub_i32 s19, s68, s0 // (op 602) 

s_mul_i32 s51, s18, s19 // (op 602) 


s_sub_u32 s18, s8, s51 // (op 602) 


s_add_u32 s8, s18, s0 // (op 602) 


s_cmp_eq_i32 s9, 0 // (op 602) 
s_mov_b32 s0, scc // (op 602) 

s_cmp_eq_i32 s0, 0 // (op 602) 
s_cselect_b32 s9, s8, s1 // (op 602) 



v_lshl_add_u32 v15, s9, 4, v24 // (op 602) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 602) 
s_lshr_b32 s51, s0, 1 // (op 602) 

v_lshl_add_u32 v24, s51, 9, v14 // (op 602) 


v_mov_b32 v19, 0 // (op 602) convert
v_mov_b32 v18, v24 // (op 602) convert

v_mul_lo_u32 v21, v15, s5 // (op 602) most significant: low of low * high
v_mul_hi_u32 v14, v15, s4 // (op 602) most significant: high of low * low
v_mul_lo_u32 v20, v15, s4 // (op 602) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 602) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 602) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 602) most significant half



s_mul_i32 s1, s4, s7 // (op 604) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 604) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 604) least significant
s_mul_hi_u32 s8, s4, s6 // (op 604) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 604) most significant: sum
s_add_u32 s1, s1, s8 // (op 604) most significant: sum


s_mov_b32 s52, s20 // (op 604) reg expression
s_mov_b32 s53, s21 // (op 604) reg expression
s_mov_b32 s54, s0 // (op 604) reg expression
s_mov_b32 s55, 131072 // (op 604) call()

s_mul_hi_i32 s51, s16, s11 // (op 669) 
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
s_sub_u32 s68, s18, 1 // (op 669) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 669) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 669) 

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
s_lshl_b32 s68, 1, s8 // (op 669) 
s_add_i32 s69, s68, s9 // (op 669) 


s_and_b32 s9, s0, s69 // (op 669) 


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
s_add_i32 s68, s19, s18 // (op 669) 


s_and_b32 s18, s0, s68 // (op 669) 


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

s_ashr_i64 s[0:1], s[26:27], 8 // (op 669) 
s_lshr_b32 s51, s0, 1 // (op 669) 

v_lshl_add_u32 v22, s51, 9, v1 // (op 669) 

v_mul_lo_u32 v15, v22, s23 // (op 669) most significant: low of low * high
v_mul_hi_u32 v24, v22, s22 // (op 669) most significant: high of low * low
v_mul_lo_u32 v14, v22, s22 // (op 669) least significant: low of low * low
v_add_u32 v15, v15, v24 // (op 669) most significant: sum

v_mov_b32 v24, s8 // (op 669) 
v_mov_b32 v18, s9 // (op 669) 
v_add_co_u32 v14, s[0:1], v14, v24 // (op 669) least significant half
v_addc_co_u32 v15, s[0:1], v15, v18, s[0:1] // (op 669) most significant half





s_mul_i32 s1, s22, s27 // (op 671) most significant: low of low * high
s_mul_i32 s51, s23, s26 // (op 671) most significant: low of high * low
s_mul_i32 s0, s22, s26 // (op 671) least significant
s_mul_hi_u32 s8, s22, s26 // (op 671) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 671) most significant: sum
s_add_u32 s1, s1, s8 // (op 671) most significant: sum


s_mov_b32 s56, s24 // (op 671) reg expression
s_mov_b32 s57, s25 // (op 671) reg expression
s_mov_b32 s58, s0 // (op 671) reg expression
s_mov_b32 s59, 131072 // (op 671) call()

s_mul_hi_i32 s51, s16, s11 // (op 733) 
v_lshrrev_b32 v22, 3, v1 // (op 733) 
s_add_i32 s0, s51, s16 // (op 733) 

v_lshlrev_b32 v24, 1, v22 // (op 733) 

s_ashr_i32 s1, s0, 31 // (op 733) 
v_and_b32 v22, -8, v1 // (op 733) 
s_cmp_eq_i32 s11, 0 // (op 733) 
s_mov_b32 s51, scc // (op 733) 
v_sub_u32 v18, v1, v22 // (op 733) 

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
s_sub_u32 s68, s18, 1 // (op 733) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 733) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 733) 

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
s_lshl_b32 s68, 1, s19 // (op 733) 
s_add_i32 s69, s68, s51 // (op 733) 


s_and_b32 s51, s0, s69 // (op 733) 


s_add_i32 s0, s18, s51 // (op 733) 


s_ashr_i32 s18, s0, s19 // (op 733) 


s_lshl_b32 s0, s48, 24 // (op 733) 
s_ashr_i32 s19, s0, 31 // (op 733) 

s_xor_b32 s0, s18, s19 // (op 733) 

s_sub_i32 s18, s0, s19 // (op 733) 


s_mul_i32 s0, s47, s46 // (op 733) 
s_add_u32 s68, s6, 16 // (op 733) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 733) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 733) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 733) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 733) 

s_sub_i32 s19, s68, s0 // (op 733) 

s_mul_i32 s51, s18, s19 // (op 733) 


s_sub_u32 s18, s8, s51 // (op 733) 


s_add_u32 s8, s18, s0 // (op 733) 


s_cmp_eq_i32 s9, 0 // (op 733) 
s_mov_b32 s0, scc // (op 733) 

s_cmp_eq_i32 s0, 0 // (op 733) 
s_cselect_b32 s9, s8, s1 // (op 733) 



v_lshl_add_u32 v19, s9, 4, v24 // (op 733) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 733) 
s_lshr_b32 s51, s0, 1 // (op 733) 

v_lshl_add_u32 v22, s51, 4, v18 // (op 733) 


v_mul_lo_u32 v18, v19, s28 // (op 733) 
v_add_u32 v18, v18, v22 // (op 733) 


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

s_mul_hi_i32 s51, s16, s11 // (op 797) 
v_and_b32 v19, -16, v1 // (op 797) 
s_add_i32 s0, s51, s16 // (op 797) 

v_sub_u32 v22, v1, v19 // (op 797) 

s_ashr_i32 s1, s0, 31 // (op 797) 
v_lshrrev_b32 v19, 4, v1 // (op 797) 
s_cmp_eq_i32 s11, 0 // (op 797) 
s_mov_b32 s51, scc // (op 797) 
v_lshlrev_b32 v24, 1, v19 // (op 797) 

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



v_lshl_add_u32 v19, s1, 4, v22 // (op 797) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 797) 
s_lshr_b32 s51, s0, 1 // (op 797) 

v_lshl_add_u32 v22, s51, 4, v24 // (op 797) 


v_mul_lo_u32 v24, v22, s32 // (op 797) 
v_add_u32 v24, v24, v19 // (op 797) 


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

v_and_b32 v19, -16, v1 // (op 890) 
v_sub_u32 v22, v1, v19 // (op 890) 

v_lshlrev_b32 v19, 4, v22 // (op 890) 

v_lshrrev_b32 v22, 4, v1 // (op 890) 
v_lshl_add_u32 v28, v22, 8, v19 // (op 890) 


s_nop 0
v_readfirstlane_b32 s51, v28 // (op 890) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v19, 4, v1 // (op 942) 
s_nop 0
v_readfirstlane_b32 s0, v19 // (op 942) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -8, v1 // (op 994) 
v_sub_u32 v22, v1, v19 // (op 994) 

v_lshrrev_b32 v19, 3, v1 // (op 994) 
v_lshl_add_u32 v28, v19, 4, v22 // (op 994) 


v_and_b32 v19, -16, v1 // (op 1047) 
v_sub_u32 v22, v1, v19 // (op 1047) 

v_lshrrev_b32 v19, 4, v1 // (op 1047) 
v_lshl_add_u32 v29, v19, 5, v22 // (op 1047) 


s_ashr_i64 s[8:9], s[26:27], 8 // (op 198) 
s_and_b32 s1, s8, 4294967294 // (op 198) 

s_ashr_i64 s[8:9], s[26:27], 8 // (op 197) 
s_mov_b32 s18, s8 // (op 197) convert

s_cmp_lt_u32 s1, s18 // (op 197) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_197 // (op 197) Condition: Top (jump to RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_197 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_197:
s_barrier  // (op 1292) 
s_add_u32 s1, s1, 1 // (op 199) 

s_lshl_b64 s[8:9], s[4:5], 2 // (op 554) 
v_mov_b64 v[12:13], s[8:9] // (op 554) convert


v_mov_b64 v[16:17], v[20:21] // (op 554) 
v_mov_b64 v[22:23], v[16:17] // (op 554) 
s_add_u32 s8, 4096, s51 // (op 554) 
s_mov_b32 m0, s8 // (op 554) reg expression
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 554) Load value direct to lds
v_add_co_u32 v22, s[18:19], v22, v12 // (op 554) least significant half
v_addc_co_u32 v23, s[18:19], 0, v23, s[18:19] // (op 554) most significant half

s_add_u32 m0, m0, 1024 // (op 554) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 554) Load value direct to lds
v_add_co_u32 v22, s[18:19], v22, v12 // (op 554) least significant half
v_addc_co_u32 v23, s[18:19], 0, v23, s[18:19] // (op 554) most significant half

s_add_u32 m0, m0, 1024 // (op 554) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 554) Load value direct to lds
v_add_co_u32 v22, s[18:19], v22, v12 // (op 554) least significant half
v_addc_co_u32 v23, s[18:19], 0, v23, s[18:19] // (op 554) most significant half

s_add_u32 m0, m0, 1024 // (op 554) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 554) Load value direct to lds




v_add_co_u32 v20, vcc, 256, v20 // (op 592) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 592) most significant half
s_barrier  // (op 1308) 

s_lshl_b64 s[8:9], s[22:23], 6 // (op 560) 
v_mov_b64 v[12:13], s[8:9] // (op 560) convert


v_mov_b64 v[16:17], v[14:15] // (op 560) 
v_mov_b64 v[22:23], v[16:17] // (op 560) reg expression
s_mov_b32 m0, s0 // (op 560) reg expression
buffer_load_dwordx4 v22, s[56:59], 0 offen offset: 0 lds // (op 560) Load value direct to lds
v_add_co_u32 v22, s[8:9], v22, v12 // (op 560) least significant half
v_addc_co_u32 v23, s[8:9], 0, v23, s[8:9] // (op 560) most significant half

s_add_u32 m0, m0, 1024 // (op 560) 
buffer_load_dwordx4 v22, s[56:59], 0 offen offset: 0 lds // (op 560) Load value direct to lds
v_add_co_u32 v22, s[8:9], v22, v12 // (op 560) least significant half
v_addc_co_u32 v23, s[8:9], 0, v23, s[8:9] // (op 560) most significant half

s_add_u32 m0, m0, 1024 // (op 560) 
buffer_load_dwordx4 v22, s[56:59], 0 offen offset: 0 lds // (op 560) Load value direct to lds
v_add_co_u32 v22, s[8:9], v22, v12 // (op 560) least significant half
v_addc_co_u32 v23, s[8:9], 0, v23, s[8:9] // (op 560) most significant half

s_add_u32 m0, m0, 1024 // (op 560) 
buffer_load_dwordx4 v22, s[56:59], 0 offen offset: 0 lds // (op 560) Load value direct to lds



s_lshl_b64 s[8:9], s[22:23], 8 // (op 661) 
v_mov_b32 v19, s8 // (op 661) 
v_mov_b32 v12, s9 // (op 661) 
v_add_co_u32 v14, s[18:19], v14, v19 // (op 661) least significant half
v_addc_co_u32 v15, s[18:19], v15, v12, s[18:19] // (op 661) most significant half





v_mov_b64 v[12:13], s[28:29] // (op 226) convert

v_mov_b32 v19, v18 // (op 226) 
v_mov_b32 v16, v19 // (op 226) 
buffer_load_ubyte v22, v16, s[60:63], 0 offen offset: 0 // (op 226) Load value
v_add_u32 v16, v16, v12 // (op 226) 
buffer_load_ubyte v23, v16, s[60:63], 0 offen offset: 0 // (op 226) Load value



v_add_u32 v18, 8, v18 // (op 725) 

s_barrier  // (op 1325) 
v_mov_b64 v[12:13], s[32:33] // (op 227) convert




v_mov_b32 v19, v24 // (op 227) 
v_mov_b32 v16, v19 // (op 227) reg expression
buffer_load_ubyte v26, v16, s[64:67], 0 offen offset: 0 // (op 227) Load value
v_add_u32 v16, v16, v12 // (op 227) 
buffer_load_ubyte v27, v16, s[64:67], 0 offen offset: 0 // (op 227) Load value



s_lshl_b32 s8, s32, 3 // (op 789) 
v_add_u32 v24, v24, s8 // (op 789) 

s_waitcnt vmcnt(3)
ds_write_b8 v28, v22 offset:16512 // (op 223) Store local data 
s_barrier  // (op 1298) 
s_waitcnt vmcnt(2)
ds_write_b8 v28, v23 offset:16520 // (op 223) Store local data 




ds_read_b128 v[30:33], v3 offset:4096 // (op 205) Load local data 
s_barrier  // (op 1342) 
ds_read_b128 v[34:37], v3 offset:4160 // (op 205) Load local data 



s_barrier  // (op 1315) 
s_waitcnt vmcnt(1)
ds_write_b8 v29, v26 offset:16384 // (op 224) Store local data 



ds_read_b64_tr_b8 v[38:39], v0 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:128 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:1024 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:1152 // (op 207) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v29, v27 offset:16400 // (op 224) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 1332) 



ds_read_u8 v19, v4 offset:16512 // (op 209) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1349) 


ds_read_u8 v12, v7 offset:16384 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v19, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 212) 







ds_read_b128 v[30:33], v3 offset:4224 // (op 214) Load local data 
ds_read_b128 v[34:37], v3 offset:4288 // (op 214) Load local data 



ds_read_b64_tr_b8 v[38:39], v0 offset:2048 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:2176 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:3072 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:3200 // (op 216) Transpose load local data 



ds_read_u8 v19, v4 offset:16516 // (op 218) Load local data 


ds_read_u8 v12, v7 offset:16448 // (op 220) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v19, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 221) 




s_ashr_i64 s[8:9], s[26:27], 8 // (op 197) 
s_mov_b32 s18, s8 // (op 197) convert

s_cmp_lt_u32 s1, s18 // (op 197) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_197 // (op 197) Condition: Bottom (jump to RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_197 if true)
RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_197:























s_barrier  // (op 1418) 








v_cmp_eq_f32 vcc, 0.00000, v9 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_157
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

s_mul_hi_i32 s0, s16, s11 // (op 1089) 
v_lshrrev_b32 v0, 6, v1 // (op 1089) 
s_add_i32 s1, s0, s16 // (op 1089) 

v_and_b32 v3, 63, v1 // (op 1089) 
s_ashr_i32 s0, s1, 31 // (op 1089) 
v_lshrrev_b32 v8, 4, v3 // (op 1089) 
s_cmp_eq_i32 s11, 0 // (op 1089) 
s_mov_b32 s4, scc // (op 1089) 
v_lshlrev_b32 v11, 2, v8 // (op 1089) 

s_cmp_eq_i32 s4, 0 // (op 1089) 
s_cselect_b32 s5, 0, -1 // (op 1089) 

v_lshl_add_u32 v8, v0, 4, v11 // (op 1089) 


s_and_b32 s4, s12, 31 // (op 1089) 
v_lshrrev_b32 v0, 2, v3 // (op 1089) 

s_lshl_b32 s8, 1, s4 // (op 1089) 
v_and_b32 v3, 3, v0 // (op 1089) 

s_add_i32 s9, s8, s5 // (op 1089) 


v_and_b32 v0, 3, v1 // (op 1089) 
s_and_b32 s5, s0, s9 // (op 1089) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 1089) 


s_add_i32 s0, s1, s5 // (op 1089) 


s_ashr_i32 s1, s0, s4 // (op 1089) 


s_lshl_b32 s0, s12, 24 // (op 1089) 
s_ashr_i32 s4, s0, 31 // (op 1089) 

s_xor_b32 s0, s1, s4 // (op 1089) 

s_sub_i32 s1, s0, s4 // (op 1089) 


s_mul_hi_i32 s0, s1, s13 // (op 1089) 
s_add_i32 s4, s0, s1 // (op 1089) 

s_ashr_i32 s0, s4, 31 // (op 1089) 
s_cmp_eq_i32 s13, 0 // (op 1089) 
s_mov_b32 s5, scc // (op 1089) 
s_cmp_eq_i32 s5, 0 // (op 1089) 
s_cselect_b32 s8, 0, -1 // (op 1089) 

s_and_b32 s5, s14, 31 // (op 1089) 
s_lshl_b32 s9, 1, s5 // (op 1089) 
s_add_i32 s18, s9, s8 // (op 1089) 


s_and_b32 s8, s0, s18 // (op 1089) 


s_add_i32 s0, s4, s8 // (op 1089) 


s_ashr_i32 s4, s0, s5 // (op 1089) 


s_lshl_b32 s0, s14, 24 // (op 1089) 
s_ashr_i32 s5, s0, 31 // (op 1089) 

s_xor_b32 s0, s4, s5 // (op 1089) 

s_sub_i32 s4, s0, s5 // (op 1089) 


s_mul_i32 s0, s1, s46 // (op 1089) 
s_sub_u32 s5, s16, s0 // (op 1089) 

s_mul_i32 s0, s4, s46 // (op 1089) 
s_add_u32 s0, s0, s5 // (op 1089) 

s_mul_hi_i32 s5, s16, s10 // (op 1089) 
s_add_i32 s8, s5, s16 // (op 1089) 

s_ashr_i32 s5, s8, 31 // (op 1089) 
s_cmp_eq_i32 s10, 0 // (op 1089) 
s_mov_b32 s9, scc // (op 1089) 
s_cmp_eq_i32 s9, 0 // (op 1089) 
s_cselect_b32 s18, 0, -1 // (op 1089) 

s_and_b32 s9, s49, 31 // (op 1089) 
s_lshl_b32 s19, 1, s9 // (op 1089) 
s_add_i32 s20, s19, s18 // (op 1089) 


s_and_b32 s18, s5, s20 // (op 1089) 


s_add_i32 s5, s8, s18 // (op 1089) 


s_ashr_i32 s8, s5, s9 // (op 1089) 


s_lshl_b32 s5, s49, 24 // (op 1089) 
s_ashr_i32 s9, s5, 31 // (op 1089) 

s_xor_b32 s5, s8, s9 // (op 1089) 

s_sub_i32 s8, s5, s9 // (op 1089) 


s_add_u32 s18, s2, 16 // (op 1089) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 1089) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 1089) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 1089) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 1089) 

s_mul_i32 s5, s46, s18 // (op 1089) 
s_mul_i32 s9, s50, s5 // (op 1089) 

s_mul_i32 s5, s8, s9 // (op 1089) 

s_sub_u32 s9, s16, s5 // (op 1089) 

s_mov_b32 s5, s9 // (op 1089) 
s_mul_hi_i32 s20, s5, s15 // (op 1089) 
s_add_i32 s21, s20, s5 // (op 1089) 


s_ashr_i32 s5, s21, 31 // (op 1089) 
s_cmp_eq_i32 s15, 0 // (op 1089) 
s_mov_b32 s20, scc // (op 1089) 
s_cmp_eq_i32 s20, 0 // (op 1089) 
s_cselect_b32 s22, 0, -1 // (op 1089) 

s_and_b32 s20, s48, 31 // (op 1089) 
s_lshl_b32 s23, 1, s20 // (op 1089) 
s_add_i32 s24, s23, s22 // (op 1089) 


s_and_b32 s22, s5, s24 // (op 1089) 


s_add_i32 s5, s21, s22 // (op 1089) 


s_ashr_i32 s21, s5, s20 // (op 1089) 


s_lshl_b32 s5, s48, 24 // (op 1089) 
s_ashr_i32 s20, s5, 31 // (op 1089) 

s_xor_b32 s5, s21, s20 // (op 1089) 

s_sub_i32 s21, s5, s20 // (op 1089) 


s_mul_i32 s5, s47, s46 // (op 1089) 
s_add_u32 s22, s6, 16 // (op 1089) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1089) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1089) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1089) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1089) 

s_sub_i32 s20, s22, s5 // (op 1089) 

s_mul_i32 s22, s21, s20 // (op 1089) 

s_sub_u32 s20, s9, s22 // (op 1089) 


s_add_u32 s9, s20, s5 // (op 1089) 


s_cmp_eq_i32 s8, 0 // (op 1089) 
s_mov_b32 s5, scc // (op 1089) 

s_cmp_eq_i32 s5, 0 // (op 1089) 
s_cselect_b32 s8, s9, s0 // (op 1089) 


s_mul_i32 s0, s4, s18 // (op 1089) 


v_lshl_add_u32 v0, s8, 4, v8 // (op 1089) 


s_sub_i32 s4, s1, s0 // (op 1089) 


s_cmp_eq_i32 s5, 0 // (op 1089) 
s_cselect_b32 s0, s21, s4 // (op 1089) 



v_lshl_add_u32 v3, s0, 4, v11 // (op 1089) 


v_mul_lo_u32 v8, v3, s38 // (op 1089) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 1089) 


s_mul_i32 s1, s38, s3 // (op 1091) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 1091) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 1091) least significant
s_mul_hi_u32 s5, s38, s2 // (op 1091) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 1091) most significant: sum
s_add_u32 s1, s1, s5 // (op 1091) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 1 // (op 1091) 

s_mov_b32 s20, s40 // (op 1091) reg expression
s_mov_b32 s21, s41 // (op 1091) reg expression
s_mov_b32 s22, s4 // (op 1091) reg expression
s_mov_b32 s23, 131072 // (op 1091) call()


v_pack_b32_f16 v12, v4, v5 // (op 58) 
v_pack_b32_f16 v13, v6, v7 // (op 58) 


buffer_store_dwordx2 v[12:13], v3, s[20:23], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_157
RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_157:
s_mul_hi_i32 s0, s16, s11 // (op 860) 
v_lshrrev_b32 v0, 6, v1 // (op 860) 
s_add_i32 s1, s0, s16 // (op 860) 

v_and_b32 v3, 63, v1 // (op 860) 
s_ashr_i32 s0, s1, 31 // (op 860) 
v_lshrrev_b32 v4, 4, v3 // (op 860) 
s_cmp_eq_i32 s11, 0 // (op 860) 
s_mov_b32 s4, scc // (op 860) 
v_lshlrev_b32 v5, 2, v4 // (op 860) 

s_cmp_eq_i32 s4, 0 // (op 860) 
s_cselect_b32 s5, 0, -1 // (op 860) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 860) 


s_and_b32 s4, s12, 31 // (op 860) 
v_lshrrev_b32 v0, 2, v3 // (op 860) 

s_lshl_b32 s8, 1, s4 // (op 860) 
v_and_b32 v3, 3, v0 // (op 860) 

s_add_i32 s9, s8, s5 // (op 860) 


v_and_b32 v0, 3, v1 // (op 860) 
s_and_b32 s5, s0, s9 // (op 860) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 860) 


s_add_i32 s0, s1, s5 // (op 860) 


s_ashr_i32 s1, s0, s4 // (op 860) 


s_lshl_b32 s0, s12, 24 // (op 860) 
s_ashr_i32 s4, s0, 31 // (op 860) 

s_xor_b32 s0, s1, s4 // (op 860) 

s_sub_i32 s1, s0, s4 // (op 860) 


s_mul_hi_i32 s0, s1, s13 // (op 860) 
s_add_i32 s4, s0, s1 // (op 860) 

s_ashr_i32 s0, s4, 31 // (op 860) 
s_cmp_eq_i32 s13, 0 // (op 860) 
s_mov_b32 s5, scc // (op 860) 
s_cmp_eq_i32 s5, 0 // (op 860) 
s_cselect_b32 s8, 0, -1 // (op 860) 

s_and_b32 s5, s14, 31 // (op 860) 
s_lshl_b32 s9, 1, s5 // (op 860) 
s_add_i32 s18, s9, s8 // (op 860) 


s_and_b32 s19, s0, s18 // (op 860) 


s_add_i32 s0, s4, s19 // (op 860) 


s_ashr_i32 s4, s0, s5 // (op 860) 


s_lshl_b32 s0, s14, 24 // (op 860) 
s_ashr_i32 s5, s0, 31 // (op 860) 

s_xor_b32 s0, s4, s5 // (op 860) 

s_sub_i32 s4, s0, s5 // (op 860) 


s_mul_i32 s0, s1, s46 // (op 860) 
s_sub_u32 s5, s16, s0 // (op 860) 

s_mul_i32 s0, s4, s46 // (op 860) 
s_add_u32 s0, s0, s5 // (op 860) 

s_mul_hi_i32 s5, s16, s10 // (op 860) 
s_add_i32 s8, s5, s16 // (op 860) 

s_ashr_i32 s5, s8, 31 // (op 860) 
s_cmp_eq_i32 s10, 0 // (op 860) 
s_mov_b32 s9, scc // (op 860) 
s_cmp_eq_i32 s9, 0 // (op 860) 
s_cselect_b32 s18, 0, -1 // (op 860) 

s_and_b32 s9, s49, 31 // (op 860) 
s_lshl_b32 s19, 1, s9 // (op 860) 
s_add_i32 s24, s19, s18 // (op 860) 


s_and_b32 s18, s5, s24 // (op 860) 


s_add_i32 s5, s8, s18 // (op 860) 


s_ashr_i32 s8, s5, s9 // (op 860) 


s_lshl_b32 s5, s49, 24 // (op 860) 
s_ashr_i32 s9, s5, 31 // (op 860) 

s_xor_b32 s5, s8, s9 // (op 860) 

s_sub_i32 s8, s5, s9 // (op 860) 


s_add_u32 s18, s2, 16 // (op 860) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 860) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 860) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 860) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 4 // (op 860) 

s_mul_i32 s5, s46, s18 // (op 860) 
s_mul_i32 s9, s50, s5 // (op 860) 

s_mul_i32 s5, s8, s9 // (op 860) 

s_sub_u32 s9, s16, s5 // (op 860) 

s_mov_b32 s5, s9 // (op 860) 
s_mul_hi_i32 s24, s5, s15 // (op 860) 
s_add_i32 s25, s24, s5 // (op 860) 


s_ashr_i32 s5, s25, 31 // (op 860) 
s_cmp_eq_i32 s15, 0 // (op 860) 
s_mov_b32 s24, scc // (op 860) 
s_cmp_eq_i32 s24, 0 // (op 860) 
s_cselect_b32 s26, 0, -1 // (op 860) 

s_and_b32 s24, s48, 31 // (op 860) 
s_lshl_b32 s27, 1, s24 // (op 860) 
s_add_i32 s28, s27, s26 // (op 860) 


s_and_b32 s26, s5, s28 // (op 860) 


s_add_i32 s5, s25, s26 // (op 860) 


s_ashr_i32 s25, s5, s24 // (op 860) 


s_lshl_b32 s5, s48, 24 // (op 860) 
s_ashr_i32 s24, s5, 31 // (op 860) 

s_xor_b32 s5, s25, s24 // (op 860) 

s_sub_i32 s25, s5, s24 // (op 860) 


s_mul_i32 s5, s47, s46 // (op 860) 
s_add_u32 s26, s6, 16 // (op 860) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 860) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 860) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 860) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 860) 

s_sub_i32 s24, s26, s5 // (op 860) 

s_mul_i32 s26, s25, s24 // (op 860) 

s_sub_u32 s24, s9, s26 // (op 860) 


s_add_u32 s9, s24, s5 // (op 860) 


s_cmp_eq_i32 s8, 0 // (op 860) 
s_mov_b32 s5, scc // (op 860) 

s_cmp_eq_i32 s5, 0 // (op 860) 
s_cselect_b32 s8, s9, s0 // (op 860) 


s_mul_i32 s0, s4, s18 // (op 860) 


v_lshl_add_u32 v0, s8, 4, v4 // (op 860) 


s_sub_i32 s4, s1, s0 // (op 860) 


s_cmp_eq_i32 s5, 0 // (op 860) 
s_cselect_b32 s24, s25, s4 // (op 860) 



v_lshl_add_u32 v3, s24, 4, v5 // (op 860) 


v_mul_lo_u32 v4, v3, s42 // (op 860) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 860) 


s_mul_i32 s1, s42, s3 // (op 862) most significant: low of low * high
s_mul_i32 s4, s43, s2 // (op 862) most significant: low of high * low
s_mul_i32 s0, s42, s2 // (op 862) least significant
s_mul_hi_u32 s5, s42, s2 // (op 862) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 862) most significant: sum
s_add_u32 s1, s1, s5 // (op 862) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 1 // (op 862) 

s_mov_b32 s24, s44 // (op 862) reg expression
s_mov_b32 s25, s45 // (op 862) reg expression
s_mov_b32 s26, s4 // (op 862) reg expression
s_mov_b32 s27, 131072 // (op 862) call()





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
v_mul_f32 v12, v9, v6 // (op 171) 
v_mul_f32 v13, v9, v7 // (op 171) 
v_and_b32 v6, 65535, v5 // (op 171) 
v_lshrrev_b32 v7, 16, v5 // (op 171) 
v_lshlrev_b32 v6, 16, v6 // (op 171) 
v_lshlrev_b32 v7, 16, v7 // (op 171) 
v_mul_f32 v14, v9, v6 // (op 171) 
v_mul_f32 v15, v9, v7 // (op 171) 


v_add_f32 v4, v12, v16 // (op 167) 
v_add_f32 v5, v13, v17 // (op 167) 
v_add_f32 v6, v14, v18 // (op 167) 
v_add_f32 v7, v15, v19 // (op 167) 


v_lshrrev_b32 v12, 16, v4 // (op 168) 
v_lshrrev_b32 v13, 16, v5 // (op 168) 
v_lshrrev_b32 v14, 16, v6 // (op 168) 
v_lshrrev_b32 v15, 16, v7 // (op 168) 

s_mul_hi_i32 s0, s16, s11 // (op 1106) 
v_lshrrev_b32 v0, 6, v1 // (op 1106) 
s_add_i32 s1, s0, s16 // (op 1106) 

v_and_b32 v11, 63, v1 // (op 1106) 
s_ashr_i32 s0, s1, 31 // (op 1106) 
v_lshrrev_b32 v3, 4, v11 // (op 1106) 
s_cmp_eq_i32 s11, 0 // (op 1106) 
s_mov_b32 s4, scc // (op 1106) 
v_lshlrev_b32 v4, 2, v3 // (op 1106) 

s_cmp_eq_i32 s4, 0 // (op 1106) 
s_cselect_b32 s5, 0, -1 // (op 1106) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1106) 


s_and_b32 s4, s12, 31 // (op 1106) 
v_lshrrev_b32 v0, 2, v11 // (op 1106) 

s_lshl_b32 s8, 1, s4 // (op 1106) 
v_and_b32 v11, 3, v0 // (op 1106) 

s_add_i32 s9, s8, s5 // (op 1106) 


v_and_b32 v0, 3, v1 // (op 1106) 
s_and_b32 s5, s0, s9 // (op 1106) 


v_lshl_add_u32 v4, v11, 2, v0 // (op 1106) 


s_add_i32 s0, s1, s5 // (op 1106) 


s_ashr_i32 s1, s0, s4 // (op 1106) 


s_lshl_b32 s0, s12, 24 // (op 1106) 
s_ashr_i32 s4, s0, 31 // (op 1106) 

s_xor_b32 s0, s1, s4 // (op 1106) 

s_sub_i32 s1, s0, s4 // (op 1106) 


s_mul_hi_i32 s0, s1, s13 // (op 1106) 
s_add_i32 s4, s0, s1 // (op 1106) 

s_ashr_i32 s0, s4, 31 // (op 1106) 
s_cmp_eq_i32 s13, 0 // (op 1106) 
s_mov_b32 s5, scc // (op 1106) 
s_cmp_eq_i32 s5, 0 // (op 1106) 
s_cselect_b32 s8, 0, -1 // (op 1106) 

s_and_b32 s5, s14, 31 // (op 1106) 
s_lshl_b32 s9, 1, s5 // (op 1106) 
s_add_i32 s18, s9, s8 // (op 1106) 


s_and_b32 s19, s0, s18 // (op 1106) 


s_add_i32 s0, s4, s19 // (op 1106) 


s_ashr_i32 s4, s0, s5 // (op 1106) 


s_lshl_b32 s0, s14, 24 // (op 1106) 
s_ashr_i32 s5, s0, 31 // (op 1106) 

s_xor_b32 s0, s4, s5 // (op 1106) 

s_sub_i32 s4, s0, s5 // (op 1106) 


s_mul_i32 s0, s1, s46 // (op 1106) 
s_sub_u32 s5, s16, s0 // (op 1106) 

s_mul_i32 s0, s4, s46 // (op 1106) 
s_add_u32 s0, s0, s5 // (op 1106) 

s_mul_hi_i32 s5, s16, s10 // (op 1106) 
s_add_i32 s8, s5, s16 // (op 1106) 

s_ashr_i32 s5, s8, 31 // (op 1106) 
s_cmp_eq_i32 s10, 0 // (op 1106) 
s_mov_b32 s9, scc // (op 1106) 
s_cmp_eq_i32 s9, 0 // (op 1106) 
s_cselect_b32 s18, 0, -1 // (op 1106) 

s_and_b32 s9, s49, 31 // (op 1106) 
s_lshl_b32 s19, 1, s9 // (op 1106) 
s_add_i32 s24, s19, s18 // (op 1106) 


s_and_b32 s18, s5, s24 // (op 1106) 


s_add_i32 s5, s8, s18 // (op 1106) 


s_ashr_i32 s8, s5, s9 // (op 1106) 


s_lshl_b32 s5, s49, 24 // (op 1106) 
s_ashr_i32 s9, s5, 31 // (op 1106) 

s_xor_b32 s5, s8, s9 // (op 1106) 

s_sub_i32 s8, s5, s9 // (op 1106) 


s_add_u32 s18, s2, 16 // (op 1106) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 1106) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 1106) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 1106) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 4 // (op 1106) 

s_mul_i32 s5, s46, s18 // (op 1106) 
s_mul_i32 s9, s50, s5 // (op 1106) 

s_mul_i32 s5, s8, s9 // (op 1106) 

s_sub_u32 s9, s16, s5 // (op 1106) 

s_mov_b32 s5, s9 // (op 1106) 
s_mul_hi_i32 s24, s5, s15 // (op 1106) 
s_add_i32 s25, s24, s5 // (op 1106) 


s_ashr_i32 s5, s25, 31 // (op 1106) 
s_cmp_eq_i32 s15, 0 // (op 1106) 
s_mov_b32 s24, scc // (op 1106) 
s_cmp_eq_i32 s24, 0 // (op 1106) 
s_cselect_b32 s26, 0, -1 // (op 1106) 

s_and_b32 s24, s48, 31 // (op 1106) 
s_lshl_b32 s27, 1, s24 // (op 1106) 
s_add_i32 s28, s27, s26 // (op 1106) 


s_and_b32 s26, s5, s28 // (op 1106) 


s_add_i32 s5, s25, s26 // (op 1106) 


s_ashr_i32 s25, s5, s24 // (op 1106) 


s_lshl_b32 s5, s48, 24 // (op 1106) 
s_ashr_i32 s24, s5, 31 // (op 1106) 

s_xor_b32 s5, s25, s24 // (op 1106) 

s_sub_i32 s25, s5, s24 // (op 1106) 


s_mul_i32 s5, s47, s46 // (op 1106) 
s_add_u32 s26, s6, 16 // (op 1106) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 1106) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 1106) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 1106) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 1106) 

s_sub_i32 s24, s26, s5 // (op 1106) 

s_mul_i32 s26, s25, s24 // (op 1106) 

s_sub_u32 s24, s9, s26 // (op 1106) 


s_add_u32 s9, s24, s5 // (op 1106) 


s_cmp_eq_i32 s8, 0 // (op 1106) 
s_mov_b32 s5, scc // (op 1106) 

s_cmp_eq_i32 s5, 0 // (op 1106) 
s_cselect_b32 s8, s9, s0 // (op 1106) 


s_mul_i32 s0, s4, s18 // (op 1106) 


v_lshl_add_u32 v0, s8, 4, v3 // (op 1106) 


s_sub_i32 s4, s1, s0 // (op 1106) 


s_cmp_eq_i32 s5, 0 // (op 1106) 
s_cselect_b32 s24, s25, s4 // (op 1106) 



v_lshl_add_u32 v3, s24, 4, v4 // (op 1106) 


v_mul_lo_u32 v11, v3, s38 // (op 1106) 

v_add_lshl_u32 v3, v0, v11, 1 // (op 1106) 


s_mul_i32 s1, s38, s3 // (op 1108) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 1108) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 1108) least significant
s_mul_hi_u32 s5, s38, s2 // (op 1108) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 1108) most significant: sum
s_add_u32 s1, s1, s5 // (op 1108) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 1 // (op 1108) 

s_mov_b32 s20, s40 // (op 1108) reg expression
s_mov_b32 s21, s41 // (op 1108) reg expression
s_mov_b32 s22, s4 // (op 1108) reg expression
s_mov_b32 s23, 131072 // (op 1108) call()


v_pack_b32_f16 v4, v12, v13 // (op 169) 
v_pack_b32_f16 v5, v14, v15 // (op 169) 


buffer_store_dwordx2 v[4:5], v3, s[20:23], 0 offen // (op 169) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_157:
















s_endpgm  // End of RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.LRR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end:
.size RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_, .LRR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end-RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 84
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16896 // lds bytes
  .amdhsa_accum_offset 80
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
    DataTypeB: F8
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
  - .name: RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 78
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

