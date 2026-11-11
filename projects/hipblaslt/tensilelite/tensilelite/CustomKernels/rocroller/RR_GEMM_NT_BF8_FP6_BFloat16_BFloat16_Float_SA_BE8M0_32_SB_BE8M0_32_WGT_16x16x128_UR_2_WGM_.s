// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_,@function
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 48, v1 // (op 791) 
v_and_b32 v3, 63, v1 // (op 791) 
v_sub_u32 v4, v3, v0 // (op 791) 

v_lshrrev_b32 v3, 1, v4 // (op 791) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 791) 


v_and_b32 v0, 1, v4 // (op 791) 

v_lshlrev_b32 v3, 3, v0 // (op 791) 

v_lshrrev_b32 v0, 6, v1 // (op 791) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 791) 


v_add_u32 v0, v5, v4 // (op 791) 


v_and_b32 v3, 63, v1 // (op 844) 
v_and_b32 v4, 48, v1 // (op 844) 
v_sub_u32 v5, v3, v4 // (op 844) 

v_lshrrev_b32 v4, 4, v3 // (op 844) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 844) 


v_lshlrev_b32 v4, 4, v3 // (op 844) 
v_mul_lo_u32 v5, v4, 6 // (op 844) 

v_lshrrev_b32 v4, 3, v5 // (op 844) 

v_and_b32 v5, 268435455, v3 // (op 844) 

v_lshlrev_b32 v3, 2, v5 // (op 844) 

v_add_u32 v5, v4, v3 // (op 844) 


v_lshrrev_b32 v3, 6, v1 // (op 897) 
v_and_b32 v4, 15, v1 // (op 897) 
v_lshl_add_u32 v6, v3, 4, v4 // (op 897) 


v_and_b32 v3, 48, v1 // (op 897) 
v_add_u32 v4, v3, v6 // (op 897) 


v_and_b32 v3, 48, v1 // (op 950) 
v_and_b32 v6, 15, v1 // (op 950) 
v_add_u32 v7, v3, v6 // (op 950) 


v_and_b32 v3, 48, v1 // (op 1023) 
v_and_b32 v6, 63, v1 // (op 1023) 
v_sub_u32 v8, v6, v3 // (op 1023) 

v_lshrrev_b32 v6, 1, v8 // (op 1023) 
v_add_lshl_u32 v9, v3, v6, 4 // (op 1023) 


v_and_b32 v3, 1, v8 // (op 1023) 

v_lshlrev_b32 v6, 3, v3 // (op 1023) 

v_lshrrev_b32 v3, 6, v1 // (op 1023) 
v_lshl_add_u32 v8, v3, 4, v6 // (op 1023) 


v_add_u32 v3, v9, v8 // (op 1023) 


v_and_b32 v6, 63, v1 // (op 1047) 
v_and_b32 v8, 48, v1 // (op 1047) 
v_sub_u32 v9, v6, v8 // (op 1047) 

v_lshrrev_b32 v8, 4, v6 // (op 1047) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 1047) 


v_lshlrev_b32 v8, 4, v6 // (op 1047) 
v_mul_lo_u32 v9, v8, 6 // (op 1047) 

v_lshrrev_b32 v8, 3, v9 // (op 1047) 

v_and_b32 v9, 268435455, v6 // (op 1047) 

v_lshlrev_b32 v6, 2, v9 // (op 1047) 

v_add_u32 v9, v8, v6 // (op 1047) 


v_lshrrev_b32 v6, 6, v1 // (op 1071) 
v_and_b32 v8, 15, v1 // (op 1071) 
v_lshl_add_u32 v10, v6, 4, v8 // (op 1071) 


v_and_b32 v6, 48, v1 // (op 1071) 
v_add_u32 v8, v6, v10 // (op 1071) 


v_and_b32 v6, 48, v1 // (op 1095) 
v_and_b32 v10, 15, v1 // (op 1095) 
v_add_u32 v11, v6, v10 // (op 1095) 


v_mov_b32 v6, s6 // (op 2) Move value
v_mov_b32 v10, s7 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
s_mul_hi_i32 s0, s16, s9 // (op 502) 
s_add_i32 s1, s0, s16 // (op 502) 

s_ashr_i32 s0, s1, 31 // (op 502) 
s_cmp_eq_i32 s9, 0 // (op 502) 
s_mov_b32 s6, scc // (op 502) 
s_cmp_eq_i32 s6, 0 // (op 502) 
s_cselect_b32 s7, 0, -1 // (op 502) 

s_and_b32 s6, s10, 31 // (op 502) 
s_lshl_b32 s18, 1, s6 // (op 502) 
s_add_i32 s19, s18, s7 // (op 502) 


s_and_b32 s7, s0, s19 // (op 502) 


s_add_i32 s0, s1, s7 // (op 502) 


s_ashr_i32 s1, s0, s6 // (op 502) 


s_lshl_b32 s0, s10, 24 // (op 502) 
s_ashr_i32 s6, s0, 31 // (op 502) 

s_xor_b32 s0, s1, s6 // (op 502) 

s_sub_i32 s1, s0, s6 // (op 502) 


s_mul_hi_i32 s0, s1, s11 // (op 502) 
s_add_i32 s6, s0, s1 // (op 502) 

s_ashr_i32 s0, s6, 31 // (op 502) 
s_cmp_eq_i32 s11, 0 // (op 502) 
s_mov_b32 s7, scc // (op 502) 
s_cmp_eq_i32 s7, 0 // (op 502) 
s_cselect_b32 s18, 0, -1 // (op 502) 

s_and_b32 s7, s12, 31 // (op 502) 
s_lshl_b32 s19, 1, s7 // (op 502) 
s_add_i32 s51, s19, s18 // (op 502) 


s_and_b32 s18, s0, s51 // (op 502) 


s_add_i32 s0, s6, s18 // (op 502) 


s_ashr_i32 s6, s0, s7 // (op 502) 


s_lshl_b32 s0, s12, 24 // (op 502) 
s_ashr_i32 s7, s0, 31 // (op 502) 

s_xor_b32 s0, s6, s7 // (op 502) 

s_sub_i32 s6, s0, s7 // (op 502) 


s_mul_i32 s0, s1, s14 // (op 502) 

s_sub_u32 s1, s16, s0 // (op 502) 

s_mul_i32 s0, s6, s14 // (op 502) 
s_add_u32 s0, s0, s1 // (op 502) 


s_mul_hi_i32 s1, s16, s8 // (op 502) 
s_add_i32 s6, s1, s16 // (op 502) 

s_ashr_i32 s1, s6, 31 // (op 502) 
s_cmp_eq_i32 s8, 0 // (op 502) 
s_mov_b32 s7, scc // (op 502) 
s_cmp_eq_i32 s7, 0 // (op 502) 
s_cselect_b32 s18, 0, -1 // (op 502) 

s_and_b32 s7, s15, 31 // (op 502) 
s_lshl_b32 s19, 1, s7 // (op 502) 
s_add_i32 s51, s19, s18 // (op 502) 


s_and_b32 s18, s1, s51 // (op 502) 


s_add_i32 s1, s6, s18 // (op 502) 


s_ashr_i32 s6, s1, s7 // (op 502) 


s_lshl_b32 s1, s15, 24 // (op 502) 
s_ashr_i32 s7, s1, 31 // (op 502) 

s_xor_b32 s1, s6, s7 // (op 502) 

s_sub_i32 s6, s1, s7 // (op 502) 


s_add_u32 s18, s4, 16 // (op 502) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 502) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 502) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 502) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 502) 

s_mul_i32 s1, s14, s18 // (op 502) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 502) 

s_mul_i32 s1, s6, s7 // (op 502) 

s_sub_u32 s7, s16, s1 // (op 502) 

s_mov_b32 s1, s7 // (op 502) 
s_mul_hi_i32 s18, s1, s13 // (op 502) 
s_add_i32 s19, s18, s1 // (op 502) 


s_ashr_i32 s1, s19, 31 // (op 502) 
s_cmp_eq_i32 s13, 0 // (op 502) 
s_mov_b32 s18, scc // (op 502) 
s_cmp_eq_i32 s18, 0 // (op 502) 
s_cselect_b32 s51, 0, -1 // (op 502) 

s_and_b32 s18, s49, 31 // (op 502) 
s_lshl_b32 s52, 1, s18 // (op 502) 
s_add_i32 s53, s52, s51 // (op 502) 


s_and_b32 s51, s1, s53 // (op 502) 


s_add_i32 s1, s19, s51 // (op 502) 


s_ashr_i32 s19, s1, s18 // (op 502) 


s_lshl_b32 s1, s49, 24 // (op 502) 
s_ashr_i32 s18, s1, 31 // (op 502) 

s_xor_b32 s1, s19, s18 // (op 502) 

s_sub_i32 s19, s1, s18 // (op 502) 


s_mul_i32 s1, s48, s14 // (op 502) 
s_add_u32 s52, s20, 16 // (op 502) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 502) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 502) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 502) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 502) 

s_sub_i32 s18, s52, s1 // (op 502) 

s_mul_i32 s51, s19, s18 // (op 502) 


s_sub_u32 s18, s7, s51 // (op 502) 


s_add_u32 s7, s18, s1 // (op 502) 


s_cmp_eq_i32 s6, 0 // (op 502) 
s_mov_b32 s1, scc // (op 502) 

s_cmp_eq_i32 s1, 0 // (op 502) 
s_cselect_b32 s6, s7, s0 // (op 502) 



s_lshl_b32 s7, s6, 4 // (op 502) 

s_mov_b32 s1, 0 // (op 502) convert
s_mov_b32 s0, s7 // (op 502) convert

v_mul_lo_u32 v13, v1, s3 // (op 502) most significant: low of low * high
v_mul_hi_u32 v14, v1, s2 // (op 502) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 502) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 502) most significant: sum

v_mov_b32 v14, s0 // (op 502) 
v_mov_b32 v15, s1 // (op 502) 
v_add_co_u32 v16, s[6:7], v14, v12 // (op 502) least significant half
v_addc_co_u32 v17, s[6:7], v15, v13, s[6:7] // (op 502) most significant half





s_mul_i32 s1, s2, s25 // (op 504) most significant: low of low * high
s_mul_i32 s6, s3, s24 // (op 504) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 504) least significant
s_mul_hi_u32 s7, s2, s24 // (op 504) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 504) most significant: sum
s_add_u32 s1, s1, s7 // (op 504) most significant: sum


s_mov_b32 s52, s22 // (op 504) reg expression
s_mov_b32 s53, s23 // (op 504) reg expression
s_mov_b32 s54, s0 // (op 504) reg expression
s_mov_b32 s55, 131072 // (op 504) call()

s_mul_hi_i32 s51, s16, s9 // (op 521) 
s_add_i32 s0, s51, s16 // (op 521) 

s_ashr_i32 s1, s0, 31 // (op 521) 
s_cmp_eq_i32 s9, 0 // (op 521) 
s_mov_b32 s51, scc // (op 521) 
s_cmp_eq_i32 s51, 0 // (op 521) 
s_cselect_b32 s6, 0, -1 // (op 521) 

s_and_b32 s7, s10, 31 // (op 521) 
s_lshl_b32 s51, 1, s7 // (op 521) 
s_add_i32 s18, s51, s6 // (op 521) 


s_and_b32 s6, s1, s18 // (op 521) 


s_add_i32 s1, s0, s6 // (op 521) 


s_ashr_i32 s0, s1, s7 // (op 521) 


s_lshl_b32 s1, s10, 24 // (op 521) 
s_ashr_i32 s51, s1, 31 // (op 521) 

s_xor_b32 s1, s0, s51 // (op 521) 

s_sub_i32 s0, s1, s51 // (op 521) 


s_mul_hi_i32 s1, s0, s11 // (op 521) 
s_add_i32 s51, s1, s0 // (op 521) 

s_ashr_i32 s1, s51, 31 // (op 521) 
s_cmp_eq_i32 s11, 0 // (op 521) 
s_mov_b32 s6, scc // (op 521) 
s_cmp_eq_i32 s6, 0 // (op 521) 
s_cselect_b32 s7, 0, -1 // (op 521) 

s_and_b32 s6, s12, 31 // (op 521) 
s_lshl_b32 s18, 1, s6 // (op 521) 
s_add_i32 s19, s18, s7 // (op 521) 


s_and_b32 s7, s1, s19 // (op 521) 


s_add_i32 s1, s51, s7 // (op 521) 


s_ashr_i32 s7, s1, s6 // (op 521) 


s_lshl_b32 s1, s12, 24 // (op 521) 
s_ashr_i32 s6, s1, 31 // (op 521) 

s_xor_b32 s1, s7, s6 // (op 521) 

s_sub_i32 s7, s1, s6 // (op 521) 


s_mul_i32 s1, s0, s14 // (op 521) 

s_sub_u32 s0, s16, s1 // (op 521) 

s_mul_i32 s1, s7, s14 // (op 521) 
s_add_u32 s1, s1, s0 // (op 521) 


s_mul_hi_i32 s0, s16, s8 // (op 521) 
s_add_i32 s51, s0, s16 // (op 521) 

s_ashr_i32 s0, s51, 31 // (op 521) 
s_cmp_eq_i32 s8, 0 // (op 521) 
s_mov_b32 s6, scc // (op 521) 
s_cmp_eq_i32 s6, 0 // (op 521) 
s_cselect_b32 s7, 0, -1 // (op 521) 

s_and_b32 s6, s15, 31 // (op 521) 
s_lshl_b32 s18, 1, s6 // (op 521) 
s_add_i32 s19, s18, s7 // (op 521) 


s_and_b32 s7, s0, s19 // (op 521) 


s_add_i32 s0, s51, s7 // (op 521) 


s_ashr_i32 s7, s0, s6 // (op 521) 


s_lshl_b32 s0, s15, 24 // (op 521) 
s_ashr_i32 s6, s0, 31 // (op 521) 

s_xor_b32 s0, s7, s6 // (op 521) 

s_sub_i32 s7, s0, s6 // (op 521) 


s_add_u32 s18, s4, 16 // (op 521) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 521) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 521) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 521) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 521) 

s_mul_i32 s0, s14, s18 // (op 521) 

s_mul_i32 s6, s50, s0 // (op 521) 

s_mul_i32 s0, s7, s6 // (op 521) 

s_sub_u32 s6, s16, s0 // (op 521) 

s_mov_b32 s0, s6 // (op 521) 
s_mul_hi_i32 s51, s0, s13 // (op 521) 
s_add_i32 s18, s51, s0 // (op 521) 


s_ashr_i32 s0, s18, 31 // (op 521) 
s_cmp_eq_i32 s13, 0 // (op 521) 
s_mov_b32 s19, scc // (op 521) 
s_cmp_eq_i32 s19, 0 // (op 521) 
s_cselect_b32 s51, 0, -1 // (op 521) 

s_and_b32 s19, s49, 31 // (op 521) 
s_lshl_b32 s56, 1, s19 // (op 521) 
s_add_i32 s57, s56, s51 // (op 521) 


s_and_b32 s51, s0, s57 // (op 521) 


s_add_i32 s0, s18, s51 // (op 521) 


s_ashr_i32 s18, s0, s19 // (op 521) 


s_lshl_b32 s0, s49, 24 // (op 521) 
s_ashr_i32 s19, s0, 31 // (op 521) 

s_xor_b32 s0, s18, s19 // (op 521) 

s_sub_i32 s18, s0, s19 // (op 521) 


s_mul_i32 s0, s48, s14 // (op 521) 
s_add_u32 s56, s20, 16 // (op 521) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 521) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 521) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 521) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 521) 

s_sub_i32 s19, s56, s0 // (op 521) 

s_mul_i32 s51, s18, s19 // (op 521) 


s_sub_u32 s18, s6, s51 // (op 521) 


s_add_u32 s6, s18, s0 // (op 521) 


s_cmp_eq_i32 s7, 0 // (op 521) 
s_mov_b32 s0, scc // (op 521) 

s_cmp_eq_i32 s0, 0 // (op 521) 
s_cselect_b32 s7, s6, s1 // (op 521) 



s_lshl_b32 s6, s7, 4 // (op 521) 

s_mov_b32 s1, 0 // (op 521) convert
s_mov_b32 s0, s6 // (op 521) convert

v_mul_lo_u32 v13, v1, s3 // (op 521) most significant: low of low * high
v_mul_hi_u32 v14, v1, s2 // (op 521) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 521) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 521) most significant: sum

v_mov_b32 v14, s0 // (op 521) 
v_mov_b32 v15, s1 // (op 521) 
v_add_co_u32 v18, s[6:7], v14, v12 // (op 521) least significant half
v_addc_co_u32 v19, s[6:7], v15, v13, s[6:7] // (op 521) most significant half





s_mul_i32 s1, s2, s25 // (op 523) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 523) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 523) least significant
s_mul_hi_u32 s6, s2, s24 // (op 523) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 523) most significant: sum
s_add_u32 s1, s1, s6 // (op 523) most significant: sum


s_mov_b32 s52, s22 // (op 523) reg expression
s_mov_b32 s53, s23 // (op 523) reg expression
s_mov_b32 s54, s0 // (op 523) reg expression
s_mov_b32 s55, 131072 // (op 523) call()

s_mul_hi_i32 s51, s16, s9 // (op 566) 
v_lshlrev_b32 v12, 1, v1 // (op 566) 
s_add_i32 s0, s51, s16 // (op 566) 

s_ashr_i32 s1, s0, 31 // (op 566) 
s_cmp_eq_i32 s9, 0 // (op 566) 
s_mov_b32 s51, scc // (op 566) 
s_cmp_eq_i32 s51, 0 // (op 566) 
s_cselect_b32 s6, 0, -1 // (op 566) 

s_and_b32 s7, s10, 31 // (op 566) 
s_lshl_b32 s51, 1, s7 // (op 566) 
s_add_i32 s18, s51, s6 // (op 566) 


s_and_b32 s6, s1, s18 // (op 566) 


s_add_i32 s1, s0, s6 // (op 566) 


s_ashr_i32 s0, s1, s7 // (op 566) 


s_lshl_b32 s1, s10, 24 // (op 566) 
s_ashr_i32 s51, s1, 31 // (op 566) 

s_xor_b32 s1, s0, s51 // (op 566) 

s_sub_i32 s0, s1, s51 // (op 566) 


s_mul_hi_i32 s1, s0, s11 // (op 566) 
s_add_i32 s51, s1, s0 // (op 566) 

s_ashr_i32 s1, s51, 31 // (op 566) 
s_cmp_eq_i32 s11, 0 // (op 566) 
s_mov_b32 s6, scc // (op 566) 
s_cmp_eq_i32 s6, 0 // (op 566) 
s_cselect_b32 s7, 0, -1 // (op 566) 

s_and_b32 s6, s12, 31 // (op 566) 
s_lshl_b32 s18, 1, s6 // (op 566) 
s_add_i32 s19, s18, s7 // (op 566) 


s_and_b32 s7, s1, s19 // (op 566) 


s_add_i32 s1, s51, s7 // (op 566) 


s_ashr_i32 s7, s1, s6 // (op 566) 


s_lshl_b32 s1, s12, 24 // (op 566) 
s_ashr_i32 s6, s1, 31 // (op 566) 

s_xor_b32 s1, s7, s6 // (op 566) 

s_sub_i32 s7, s1, s6 // (op 566) 


s_add_u32 s18, s4, 16 // (op 566) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 566) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 566) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 566) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 566) 

s_mul_i32 s1, s7, s18 // (op 566) 

s_sub_i32 s51, s0, s1 // (op 566) 


s_mul_hi_i32 s0, s16, s8 // (op 566) 
s_add_i32 s1, s0, s16 // (op 566) 

s_ashr_i32 s0, s1, 31 // (op 566) 
s_cmp_eq_i32 s8, 0 // (op 566) 
s_mov_b32 s6, scc // (op 566) 
s_cmp_eq_i32 s6, 0 // (op 566) 
s_cselect_b32 s7, 0, -1 // (op 566) 

s_and_b32 s6, s15, 31 // (op 566) 
s_lshl_b32 s56, 1, s6 // (op 566) 
s_add_i32 s57, s56, s7 // (op 566) 


s_and_b32 s7, s0, s57 // (op 566) 


s_add_i32 s0, s1, s7 // (op 566) 


s_ashr_i32 s1, s0, s6 // (op 566) 


s_lshl_b32 s0, s15, 24 // (op 566) 
s_ashr_i32 s6, s0, 31 // (op 566) 

s_xor_b32 s0, s1, s6 // (op 566) 

s_sub_i32 s1, s0, s6 // (op 566) 


s_mul_i32 s0, s14, s18 // (op 566) 

s_mul_i32 s6, s50, s0 // (op 566) 

s_mul_i32 s0, s1, s6 // (op 566) 

s_sub_u32 s6, s16, s0 // (op 566) 

s_mul_hi_i32 s0, s6, s13 // (op 566) 
s_add_i32 s7, s0, s6 // (op 566) 


s_ashr_i32 s0, s7, 31 // (op 566) 
s_cmp_eq_i32 s13, 0 // (op 566) 
s_mov_b32 s6, scc // (op 566) 
s_cmp_eq_i32 s6, 0 // (op 566) 
s_cselect_b32 s18, 0, -1 // (op 566) 

s_and_b32 s6, s49, 31 // (op 566) 
s_lshl_b32 s19, 1, s6 // (op 566) 
s_add_i32 s56, s19, s18 // (op 566) 


s_and_b32 s18, s0, s56 // (op 566) 


s_add_i32 s0, s7, s18 // (op 566) 


s_ashr_i32 s7, s0, s6 // (op 566) 


s_lshl_b32 s0, s49, 24 // (op 566) 
s_ashr_i32 s6, s0, 31 // (op 566) 

s_xor_b32 s0, s7, s6 // (op 566) 

s_sub_i32 s7, s0, s6 // (op 566) 


s_cmp_eq_i32 s1, 0 // (op 566) 
s_mov_b32 s0, scc // (op 566) 

s_cmp_eq_i32 s0, 0 // (op 566) 
s_cselect_b32 s1, s7, s51 // (op 566) 



s_lshl_b32 s0, s1, 4 // (op 566) 

s_mov_b32 s7, 0 // (op 566) convert
s_mov_b32 s6, s0 // (op 566) convert

v_mul_lo_u32 v15, v12, s27 // (op 566) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 566) most significant: high of low * low
v_mul_lo_u32 v14, v12, s26 // (op 566) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 566) most significant: sum

v_mov_b32 v13, s6 // (op 566) 
v_mov_b32 v20, s7 // (op 566) 
v_add_co_u32 v14, s[0:1], v14, v13 // (op 566) least significant half
v_addc_co_u32 v15, s[0:1], v15, v20, s[0:1] // (op 566) most significant half





v_mul_lo_u32 v13, v15, 6 // (op 566) most significant: low of high * low
v_mul_hi_u32 v20, v14, 6 // (op 566) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 566) least significant: low of low * low
v_add_u32 v13, v13, v20 // (op 566) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 566) 

v_mov_b32 v12, v14 // (op 566) convert

s_mul_i32 s1, s25, 6 // (op 568) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 568) least significant
s_mul_hi_u32 s51, s24, 6 // (op 568) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 568) most significant: sum

s_mul_i32 s7, s26, s1 // (op 568) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 568) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 568) least significant
s_mul_hi_u32 s18, s26, s0 // (op 568) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 568) most significant: sum
s_add_u32 s7, s7, s18 // (op 568) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 568) 

s_mov_b32 s56, s28 // (op 568) reg expression
s_mov_b32 s57, s29 // (op 568) reg expression
s_mov_b32 s58, s0 // (op 568) reg expression
s_mov_b32 s59, 131072 // (op 568) call()

s_mul_hi_i32 s51, s16, s9 // (op 585) 
v_lshlrev_b32 v13, 1, v1 // (op 585) 
s_add_i32 s0, s51, s16 // (op 585) 

s_ashr_i32 s1, s0, 31 // (op 585) 
s_cmp_eq_i32 s9, 0 // (op 585) 
s_mov_b32 s51, scc // (op 585) 
s_cmp_eq_i32 s51, 0 // (op 585) 
s_cselect_b32 s6, 0, -1 // (op 585) 

s_and_b32 s7, s10, 31 // (op 585) 
s_lshl_b32 s51, 1, s7 // (op 585) 
s_add_i32 s18, s51, s6 // (op 585) 


s_and_b32 s6, s1, s18 // (op 585) 


s_add_i32 s1, s0, s6 // (op 585) 


s_ashr_i32 s0, s1, s7 // (op 585) 


s_lshl_b32 s1, s10, 24 // (op 585) 
s_ashr_i32 s51, s1, 31 // (op 585) 

s_xor_b32 s1, s0, s51 // (op 585) 

s_sub_i32 s0, s1, s51 // (op 585) 


s_mul_hi_i32 s1, s0, s11 // (op 585) 
s_add_i32 s51, s1, s0 // (op 585) 

s_ashr_i32 s1, s51, 31 // (op 585) 
s_cmp_eq_i32 s11, 0 // (op 585) 
s_mov_b32 s6, scc // (op 585) 
s_cmp_eq_i32 s6, 0 // (op 585) 
s_cselect_b32 s7, 0, -1 // (op 585) 

s_and_b32 s6, s12, 31 // (op 585) 
s_lshl_b32 s18, 1, s6 // (op 585) 
s_add_i32 s19, s18, s7 // (op 585) 


s_and_b32 s7, s1, s19 // (op 585) 


s_add_i32 s1, s51, s7 // (op 585) 


s_ashr_i32 s7, s1, s6 // (op 585) 


s_lshl_b32 s1, s12, 24 // (op 585) 
s_ashr_i32 s6, s1, 31 // (op 585) 

s_xor_b32 s1, s7, s6 // (op 585) 

s_sub_i32 s7, s1, s6 // (op 585) 


s_add_u32 s18, s4, 16 // (op 585) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 585) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 585) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 585) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 585) 

s_mul_i32 s1, s7, s18 // (op 585) 

s_sub_i32 s51, s0, s1 // (op 585) 


s_mul_hi_i32 s0, s16, s8 // (op 585) 
s_add_i32 s1, s0, s16 // (op 585) 

s_ashr_i32 s0, s1, 31 // (op 585) 
s_cmp_eq_i32 s8, 0 // (op 585) 
s_mov_b32 s6, scc // (op 585) 
s_cmp_eq_i32 s6, 0 // (op 585) 
s_cselect_b32 s7, 0, -1 // (op 585) 

s_and_b32 s6, s15, 31 // (op 585) 
s_lshl_b32 s60, 1, s6 // (op 585) 
s_add_i32 s61, s60, s7 // (op 585) 


s_and_b32 s7, s0, s61 // (op 585) 


s_add_i32 s0, s1, s7 // (op 585) 


s_ashr_i32 s1, s0, s6 // (op 585) 


s_lshl_b32 s0, s15, 24 // (op 585) 
s_ashr_i32 s6, s0, 31 // (op 585) 

s_xor_b32 s0, s1, s6 // (op 585) 

s_sub_i32 s1, s0, s6 // (op 585) 


s_mul_i32 s0, s14, s18 // (op 585) 

s_mul_i32 s6, s50, s0 // (op 585) 

s_mul_i32 s0, s1, s6 // (op 585) 

s_sub_u32 s6, s16, s0 // (op 585) 

s_mul_hi_i32 s0, s6, s13 // (op 585) 
s_add_i32 s7, s0, s6 // (op 585) 


s_ashr_i32 s0, s7, 31 // (op 585) 
s_cmp_eq_i32 s13, 0 // (op 585) 
s_mov_b32 s6, scc // (op 585) 
s_cmp_eq_i32 s6, 0 // (op 585) 
s_cselect_b32 s18, 0, -1 // (op 585) 

s_and_b32 s6, s49, 31 // (op 585) 
s_lshl_b32 s19, 1, s6 // (op 585) 
s_add_i32 s60, s19, s18 // (op 585) 


s_and_b32 s18, s0, s60 // (op 585) 


s_add_i32 s0, s7, s18 // (op 585) 


s_ashr_i32 s7, s0, s6 // (op 585) 


s_lshl_b32 s0, s49, 24 // (op 585) 
s_ashr_i32 s6, s0, 31 // (op 585) 

s_xor_b32 s0, s7, s6 // (op 585) 

s_sub_i32 s7, s0, s6 // (op 585) 


s_cmp_eq_i32 s1, 0 // (op 585) 
s_mov_b32 s0, scc // (op 585) 

s_cmp_eq_i32 s0, 0 // (op 585) 
s_cselect_b32 s1, s7, s51 // (op 585) 



s_lshl_b32 s0, s1, 4 // (op 585) 

s_mov_b32 s7, 0 // (op 585) convert
s_mov_b32 s6, s0 // (op 585) convert

v_mul_lo_u32 v15, v13, s27 // (op 585) most significant: low of low * high
v_mul_hi_u32 v20, v13, s26 // (op 585) most significant: high of low * low
v_mul_lo_u32 v14, v13, s26 // (op 585) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 585) most significant: sum

v_mov_b32 v20, s6 // (op 585) 
v_mov_b32 v21, s7 // (op 585) 
v_add_co_u32 v14, s[0:1], v14, v20 // (op 585) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 585) most significant half





v_mul_lo_u32 v21, v15, 6 // (op 585) most significant: low of high * low
v_mul_hi_u32 v13, v14, 6 // (op 585) most significant: high of low * low
v_mul_lo_u32 v20, v14, 6 // (op 585) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 585) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 585) 

v_mov_b32 v13, v14 // (op 585) convert

s_mul_i32 s1, s25, 6 // (op 587) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 587) least significant
s_mul_hi_u32 s51, s24, 6 // (op 587) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 587) most significant: sum

s_mul_i32 s7, s26, s1 // (op 587) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 587) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 587) least significant
s_mul_hi_u32 s18, s26, s0 // (op 587) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 587) most significant: sum
s_add_u32 s7, s7, s18 // (op 587) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 587) 

s_mov_b32 s56, s28 // (op 587) reg expression
s_mov_b32 s57, s29 // (op 587) reg expression
s_mov_b32 s58, s0 // (op 587) reg expression
s_mov_b32 s59, 131072 // (op 587) call()

s_mul_hi_i32 s51, s16, s9 // (op 630) 
v_and_b32 v14, -16, v1 // (op 630) 
s_add_i32 s0, s51, s16 // (op 630) 

v_sub_u32 v15, v1, v14 // (op 630) 

s_ashr_i32 s1, s0, 31 // (op 630) 
v_lshrrev_b32 v14, 4, v1 // (op 630) 
s_cmp_eq_i32 s9, 0 // (op 630) 
s_mov_b32 s51, scc // (op 630) 
s_cmp_eq_i32 s51, 0 // (op 630) 
s_cselect_b32 s6, 0, -1 // (op 630) 

s_and_b32 s7, s10, 31 // (op 630) 
s_lshl_b32 s51, 1, s7 // (op 630) 
s_add_i32 s18, s51, s6 // (op 630) 


s_and_b32 s6, s1, s18 // (op 630) 


s_add_i32 s1, s0, s6 // (op 630) 


s_ashr_i32 s0, s1, s7 // (op 630) 


s_lshl_b32 s1, s10, 24 // (op 630) 
s_ashr_i32 s51, s1, 31 // (op 630) 

s_xor_b32 s1, s0, s51 // (op 630) 

s_sub_i32 s0, s1, s51 // (op 630) 


s_mul_hi_i32 s1, s0, s11 // (op 630) 
s_add_i32 s51, s1, s0 // (op 630) 

s_ashr_i32 s1, s51, 31 // (op 630) 
s_cmp_eq_i32 s11, 0 // (op 630) 
s_mov_b32 s6, scc // (op 630) 
s_cmp_eq_i32 s6, 0 // (op 630) 
s_cselect_b32 s7, 0, -1 // (op 630) 

s_and_b32 s6, s12, 31 // (op 630) 
s_lshl_b32 s18, 1, s6 // (op 630) 
s_add_i32 s19, s18, s7 // (op 630) 


s_and_b32 s7, s1, s19 // (op 630) 


s_add_i32 s1, s51, s7 // (op 630) 


s_ashr_i32 s7, s1, s6 // (op 630) 


s_lshl_b32 s1, s12, 24 // (op 630) 
s_ashr_i32 s6, s1, 31 // (op 630) 

s_xor_b32 s1, s7, s6 // (op 630) 

s_sub_i32 s7, s1, s6 // (op 630) 


s_mul_i32 s1, s0, s14 // (op 630) 

s_sub_u32 s0, s16, s1 // (op 630) 

s_mul_i32 s1, s7, s14 // (op 630) 
s_add_u32 s1, s1, s0 // (op 630) 


s_mul_hi_i32 s0, s16, s8 // (op 630) 
s_add_i32 s51, s0, s16 // (op 630) 

s_ashr_i32 s0, s51, 31 // (op 630) 
s_cmp_eq_i32 s8, 0 // (op 630) 
s_mov_b32 s6, scc // (op 630) 
s_cmp_eq_i32 s6, 0 // (op 630) 
s_cselect_b32 s7, 0, -1 // (op 630) 

s_and_b32 s6, s15, 31 // (op 630) 
s_lshl_b32 s18, 1, s6 // (op 630) 
s_add_i32 s19, s18, s7 // (op 630) 


s_and_b32 s7, s0, s19 // (op 630) 


s_add_i32 s0, s51, s7 // (op 630) 


s_ashr_i32 s7, s0, s6 // (op 630) 


s_lshl_b32 s0, s15, 24 // (op 630) 
s_ashr_i32 s6, s0, 31 // (op 630) 

s_xor_b32 s0, s7, s6 // (op 630) 

s_sub_i32 s7, s0, s6 // (op 630) 


s_add_u32 s18, s4, 16 // (op 630) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 630) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 630) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 630) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 630) 

s_mul_i32 s0, s14, s18 // (op 630) 

s_mul_i32 s6, s50, s0 // (op 630) 

s_mul_i32 s0, s7, s6 // (op 630) 

s_sub_u32 s6, s16, s0 // (op 630) 

s_mov_b32 s0, s6 // (op 630) 
s_mul_hi_i32 s51, s0, s13 // (op 630) 
s_add_i32 s18, s51, s0 // (op 630) 


s_ashr_i32 s0, s18, 31 // (op 630) 
s_cmp_eq_i32 s13, 0 // (op 630) 
s_mov_b32 s19, scc // (op 630) 
s_cmp_eq_i32 s19, 0 // (op 630) 
s_cselect_b32 s51, 0, -1 // (op 630) 

s_and_b32 s19, s49, 31 // (op 630) 
s_lshl_b32 s60, 1, s19 // (op 630) 
s_add_i32 s61, s60, s51 // (op 630) 


s_and_b32 s51, s0, s61 // (op 630) 


s_add_i32 s0, s18, s51 // (op 630) 


s_ashr_i32 s18, s0, s19 // (op 630) 


s_lshl_b32 s0, s49, 24 // (op 630) 
s_ashr_i32 s19, s0, 31 // (op 630) 

s_xor_b32 s0, s18, s19 // (op 630) 

s_sub_i32 s18, s0, s19 // (op 630) 


s_mul_i32 s0, s48, s14 // (op 630) 
s_add_u32 s60, s20, 16 // (op 630) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 630) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 630) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 630) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 630) 

s_sub_i32 s19, s60, s0 // (op 630) 

s_mul_i32 s51, s18, s19 // (op 630) 


s_sub_u32 s18, s6, s51 // (op 630) 


s_add_u32 s6, s18, s0 // (op 630) 


s_cmp_eq_i32 s7, 0 // (op 630) 
s_mov_b32 s0, scc // (op 630) 

s_cmp_eq_i32 s0, 0 // (op 630) 
s_cselect_b32 s7, s6, s1 // (op 630) 



v_lshl_add_u32 v20, s7, 4, v15 // (op 630) 


v_mul_lo_u32 v15, v14, s30 // (op 630) 

v_add_u32 v14, v20, v15 // (op 630) 


s_mul_i32 s1, s30, s35 // (op 632) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 632) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 632) least significant
s_mul_hi_u32 s6, s30, s34 // (op 632) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 632) most significant: sum
s_add_u32 s1, s1, s6 // (op 632) most significant: sum


s_mov_b32 s60, s32 // (op 632) reg expression
s_mov_b32 s61, s33 // (op 632) reg expression
s_mov_b32 s62, s0 // (op 632) reg expression
s_mov_b32 s63, 131072 // (op 632) call()

s_mul_hi_i32 s51, s16, s9 // (op 649) 
v_and_b32 v15, -16, v1 // (op 649) 
s_add_i32 s0, s51, s16 // (op 649) 

v_sub_u32 v20, v1, v15 // (op 649) 

s_ashr_i32 s1, s0, 31 // (op 649) 
v_lshrrev_b32 v15, 4, v1 // (op 649) 
s_cmp_eq_i32 s9, 0 // (op 649) 
s_mov_b32 s51, scc // (op 649) 
s_cmp_eq_i32 s51, 0 // (op 649) 
s_cselect_b32 s6, 0, -1 // (op 649) 

s_and_b32 s7, s10, 31 // (op 649) 
s_lshl_b32 s51, 1, s7 // (op 649) 
s_add_i32 s18, s51, s6 // (op 649) 


s_and_b32 s6, s1, s18 // (op 649) 


s_add_i32 s1, s0, s6 // (op 649) 


s_ashr_i32 s0, s1, s7 // (op 649) 


s_lshl_b32 s1, s10, 24 // (op 649) 
s_ashr_i32 s51, s1, 31 // (op 649) 

s_xor_b32 s1, s0, s51 // (op 649) 

s_sub_i32 s0, s1, s51 // (op 649) 


s_mul_hi_i32 s1, s0, s11 // (op 649) 
s_add_i32 s51, s1, s0 // (op 649) 

s_ashr_i32 s1, s51, 31 // (op 649) 
s_cmp_eq_i32 s11, 0 // (op 649) 
s_mov_b32 s6, scc // (op 649) 
s_cmp_eq_i32 s6, 0 // (op 649) 
s_cselect_b32 s7, 0, -1 // (op 649) 

s_and_b32 s6, s12, 31 // (op 649) 
s_lshl_b32 s18, 1, s6 // (op 649) 
s_add_i32 s19, s18, s7 // (op 649) 


s_and_b32 s7, s1, s19 // (op 649) 


s_add_i32 s1, s51, s7 // (op 649) 


s_ashr_i32 s7, s1, s6 // (op 649) 


s_lshl_b32 s1, s12, 24 // (op 649) 
s_ashr_i32 s6, s1, 31 // (op 649) 

s_xor_b32 s1, s7, s6 // (op 649) 

s_sub_i32 s7, s1, s6 // (op 649) 


s_mul_i32 s1, s0, s14 // (op 649) 

s_sub_u32 s0, s16, s1 // (op 649) 

s_mul_i32 s1, s7, s14 // (op 649) 
s_add_u32 s1, s1, s0 // (op 649) 


s_mul_hi_i32 s0, s16, s8 // (op 649) 
s_add_i32 s51, s0, s16 // (op 649) 

s_ashr_i32 s0, s51, 31 // (op 649) 
s_cmp_eq_i32 s8, 0 // (op 649) 
s_mov_b32 s6, scc // (op 649) 
s_cmp_eq_i32 s6, 0 // (op 649) 
s_cselect_b32 s7, 0, -1 // (op 649) 

s_and_b32 s6, s15, 31 // (op 649) 
s_lshl_b32 s18, 1, s6 // (op 649) 
s_add_i32 s19, s18, s7 // (op 649) 


s_and_b32 s7, s0, s19 // (op 649) 


s_add_i32 s0, s51, s7 // (op 649) 


s_ashr_i32 s7, s0, s6 // (op 649) 


s_lshl_b32 s0, s15, 24 // (op 649) 
s_ashr_i32 s6, s0, 31 // (op 649) 

s_xor_b32 s0, s7, s6 // (op 649) 

s_sub_i32 s7, s0, s6 // (op 649) 


s_add_u32 s18, s4, 16 // (op 649) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 649) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 649) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 649) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 649) 

s_mul_i32 s0, s14, s18 // (op 649) 

s_mul_i32 s6, s50, s0 // (op 649) 

s_mul_i32 s0, s7, s6 // (op 649) 

s_sub_u32 s6, s16, s0 // (op 649) 

s_mov_b32 s0, s6 // (op 649) 
s_mul_hi_i32 s51, s0, s13 // (op 649) 
s_add_i32 s18, s51, s0 // (op 649) 


s_ashr_i32 s0, s18, 31 // (op 649) 
s_cmp_eq_i32 s13, 0 // (op 649) 
s_mov_b32 s19, scc // (op 649) 
s_cmp_eq_i32 s19, 0 // (op 649) 
s_cselect_b32 s51, 0, -1 // (op 649) 

s_and_b32 s19, s49, 31 // (op 649) 
s_lshl_b32 s64, 1, s19 // (op 649) 
s_add_i32 s65, s64, s51 // (op 649) 


s_and_b32 s51, s0, s65 // (op 649) 


s_add_i32 s0, s18, s51 // (op 649) 


s_ashr_i32 s18, s0, s19 // (op 649) 


s_lshl_b32 s0, s49, 24 // (op 649) 
s_ashr_i32 s19, s0, 31 // (op 649) 

s_xor_b32 s0, s18, s19 // (op 649) 

s_sub_i32 s18, s0, s19 // (op 649) 


s_mul_i32 s0, s48, s14 // (op 649) 
s_add_u32 s64, s20, 16 // (op 649) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 649) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 649) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 649) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 649) 

s_sub_i32 s19, s64, s0 // (op 649) 

s_mul_i32 s51, s18, s19 // (op 649) 


s_sub_u32 s18, s6, s51 // (op 649) 


s_add_u32 s6, s18, s0 // (op 649) 


s_cmp_eq_i32 s7, 0 // (op 649) 
s_mov_b32 s0, scc // (op 649) 

s_cmp_eq_i32 s0, 0 // (op 649) 
s_cselect_b32 s7, s6, s1 // (op 649) 



v_lshl_add_u32 v21, s7, 4, v20 // (op 649) 


v_mul_lo_u32 v20, v15, s30 // (op 649) 

v_add_u32 v15, v21, v20 // (op 649) 


s_mul_i32 s1, s30, s35 // (op 651) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 651) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 651) least significant
s_mul_hi_u32 s6, s30, s34 // (op 651) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 651) most significant: sum
s_add_u32 s1, s1, s6 // (op 651) most significant: sum


s_mov_b32 s60, s32 // (op 651) reg expression
s_mov_b32 s61, s33 // (op 651) reg expression
s_mov_b32 s62, s0 // (op 651) reg expression
s_mov_b32 s63, 131072 // (op 651) call()

s_mul_hi_i32 s51, s16, s9 // (op 694) 
v_and_b32 v20, -16, v1 // (op 694) 
s_add_i32 s0, s51, s16 // (op 694) 

v_sub_u32 v21, v1, v20 // (op 694) 

s_ashr_i32 s1, s0, 31 // (op 694) 
v_lshrrev_b32 v20, 4, v1 // (op 694) 
s_cmp_eq_i32 s9, 0 // (op 694) 
s_mov_b32 s51, scc // (op 694) 
s_cmp_eq_i32 s51, 0 // (op 694) 
s_cselect_b32 s6, 0, -1 // (op 694) 

s_and_b32 s7, s10, 31 // (op 694) 
s_lshl_b32 s51, 1, s7 // (op 694) 
s_add_i32 s18, s51, s6 // (op 694) 


s_and_b32 s6, s1, s18 // (op 694) 


s_add_i32 s1, s0, s6 // (op 694) 


s_ashr_i32 s0, s1, s7 // (op 694) 


s_lshl_b32 s1, s10, 24 // (op 694) 
s_ashr_i32 s51, s1, 31 // (op 694) 

s_xor_b32 s1, s0, s51 // (op 694) 

s_sub_i32 s0, s1, s51 // (op 694) 


s_mul_hi_i32 s1, s0, s11 // (op 694) 
s_add_i32 s51, s1, s0 // (op 694) 

s_ashr_i32 s1, s51, 31 // (op 694) 
s_cmp_eq_i32 s11, 0 // (op 694) 
s_mov_b32 s6, scc // (op 694) 
s_cmp_eq_i32 s6, 0 // (op 694) 
s_cselect_b32 s7, 0, -1 // (op 694) 

s_and_b32 s6, s12, 31 // (op 694) 
s_lshl_b32 s18, 1, s6 // (op 694) 
s_add_i32 s19, s18, s7 // (op 694) 


s_and_b32 s7, s1, s19 // (op 694) 


s_add_i32 s1, s51, s7 // (op 694) 


s_ashr_i32 s7, s1, s6 // (op 694) 


s_lshl_b32 s1, s12, 24 // (op 694) 
s_ashr_i32 s6, s1, 31 // (op 694) 

s_xor_b32 s1, s7, s6 // (op 694) 

s_sub_i32 s7, s1, s6 // (op 694) 


s_add_u32 s18, s4, 16 // (op 694) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 694) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 694) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 694) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 694) 

s_mul_i32 s1, s7, s18 // (op 694) 

s_sub_i32 s51, s0, s1 // (op 694) 


s_mul_hi_i32 s0, s16, s8 // (op 694) 
s_add_i32 s1, s0, s16 // (op 694) 

s_ashr_i32 s0, s1, 31 // (op 694) 
s_cmp_eq_i32 s8, 0 // (op 694) 
s_mov_b32 s6, scc // (op 694) 
s_cmp_eq_i32 s6, 0 // (op 694) 
s_cselect_b32 s7, 0, -1 // (op 694) 

s_and_b32 s6, s15, 31 // (op 694) 
s_lshl_b32 s64, 1, s6 // (op 694) 
s_add_i32 s65, s64, s7 // (op 694) 


s_and_b32 s7, s0, s65 // (op 694) 


s_add_i32 s0, s1, s7 // (op 694) 


s_ashr_i32 s1, s0, s6 // (op 694) 


s_lshl_b32 s0, s15, 24 // (op 694) 
s_ashr_i32 s6, s0, 31 // (op 694) 

s_xor_b32 s0, s1, s6 // (op 694) 

s_sub_i32 s1, s0, s6 // (op 694) 


s_mul_i32 s0, s14, s18 // (op 694) 

s_mul_i32 s6, s50, s0 // (op 694) 

s_mul_i32 s0, s1, s6 // (op 694) 

s_sub_u32 s6, s16, s0 // (op 694) 

s_mul_hi_i32 s0, s6, s13 // (op 694) 
s_add_i32 s7, s0, s6 // (op 694) 


s_ashr_i32 s0, s7, 31 // (op 694) 
s_cmp_eq_i32 s13, 0 // (op 694) 
s_mov_b32 s6, scc // (op 694) 
s_cmp_eq_i32 s6, 0 // (op 694) 
s_cselect_b32 s18, 0, -1 // (op 694) 

s_and_b32 s6, s49, 31 // (op 694) 
s_lshl_b32 s19, 1, s6 // (op 694) 
s_add_i32 s64, s19, s18 // (op 694) 


s_and_b32 s18, s0, s64 // (op 694) 


s_add_i32 s0, s7, s18 // (op 694) 


s_ashr_i32 s7, s0, s6 // (op 694) 


s_lshl_b32 s0, s49, 24 // (op 694) 
s_ashr_i32 s6, s0, 31 // (op 694) 

s_xor_b32 s0, s7, s6 // (op 694) 

s_sub_i32 s7, s0, s6 // (op 694) 


s_cmp_eq_i32 s1, 0 // (op 694) 
s_mov_b32 s0, scc // (op 694) 

s_cmp_eq_i32 s0, 0 // (op 694) 
s_cselect_b32 s1, s7, s51 // (op 694) 



v_lshl_add_u32 v22, s1, 4, v21 // (op 694) 


v_mul_lo_u32 v21, v20, s36 // (op 694) 
v_add_u32 v21, v21, v22 // (op 694) 


s_mul_i32 s1, s36, s35 // (op 696) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 696) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 696) least significant
s_mul_hi_u32 s6, s36, s34 // (op 696) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 696) most significant: sum
s_add_u32 s1, s1, s6 // (op 696) most significant: sum


s_mov_b32 s64, s38 // (op 696) reg expression
s_mov_b32 s65, s39 // (op 696) reg expression
s_mov_b32 s66, s0 // (op 696) reg expression
s_mov_b32 s67, 131072 // (op 696) call()

s_mul_hi_i32 s51, s16, s9 // (op 713) 
v_and_b32 v20, -16, v1 // (op 713) 
s_add_i32 s0, s51, s16 // (op 713) 

v_sub_u32 v22, v1, v20 // (op 713) 

s_ashr_i32 s1, s0, 31 // (op 713) 
v_lshrrev_b32 v20, 4, v1 // (op 713) 
s_cmp_eq_i32 s9, 0 // (op 713) 
s_mov_b32 s51, scc // (op 713) 
s_cmp_eq_i32 s51, 0 // (op 713) 
s_cselect_b32 s6, 0, -1 // (op 713) 

s_and_b32 s7, s10, 31 // (op 713) 
s_lshl_b32 s51, 1, s7 // (op 713) 
s_add_i32 s18, s51, s6 // (op 713) 


s_and_b32 s6, s1, s18 // (op 713) 


s_add_i32 s1, s0, s6 // (op 713) 


s_ashr_i32 s0, s1, s7 // (op 713) 


s_lshl_b32 s1, s10, 24 // (op 713) 
s_ashr_i32 s51, s1, 31 // (op 713) 

s_xor_b32 s1, s0, s51 // (op 713) 

s_sub_i32 s0, s1, s51 // (op 713) 


s_mul_hi_i32 s1, s0, s11 // (op 713) 
s_add_i32 s51, s1, s0 // (op 713) 

s_ashr_i32 s1, s51, 31 // (op 713) 
s_cmp_eq_i32 s11, 0 // (op 713) 
s_mov_b32 s6, scc // (op 713) 
s_cmp_eq_i32 s6, 0 // (op 713) 
s_cselect_b32 s7, 0, -1 // (op 713) 

s_and_b32 s6, s12, 31 // (op 713) 
s_lshl_b32 s18, 1, s6 // (op 713) 
s_add_i32 s19, s18, s7 // (op 713) 


s_and_b32 s7, s1, s19 // (op 713) 


s_add_i32 s1, s51, s7 // (op 713) 


s_ashr_i32 s7, s1, s6 // (op 713) 


s_lshl_b32 s1, s12, 24 // (op 713) 
s_ashr_i32 s6, s1, 31 // (op 713) 

s_xor_b32 s1, s7, s6 // (op 713) 

s_sub_i32 s7, s1, s6 // (op 713) 


s_add_u32 s18, s4, 16 // (op 713) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 713) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 713) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 713) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 713) 

s_mul_i32 s1, s7, s18 // (op 713) 

s_sub_i32 s51, s0, s1 // (op 713) 


s_mul_hi_i32 s0, s16, s8 // (op 713) 
s_add_i32 s1, s0, s16 // (op 713) 

s_ashr_i32 s0, s1, 31 // (op 713) 
s_cmp_eq_i32 s8, 0 // (op 713) 
s_mov_b32 s6, scc // (op 713) 
s_cmp_eq_i32 s6, 0 // (op 713) 
s_cselect_b32 s7, 0, -1 // (op 713) 

s_and_b32 s6, s15, 31 // (op 713) 
s_lshl_b32 s68, 1, s6 // (op 713) 
s_add_i32 s69, s68, s7 // (op 713) 


s_and_b32 s7, s0, s69 // (op 713) 


s_add_i32 s0, s1, s7 // (op 713) 


s_ashr_i32 s1, s0, s6 // (op 713) 


s_lshl_b32 s0, s15, 24 // (op 713) 
s_ashr_i32 s6, s0, 31 // (op 713) 

s_xor_b32 s0, s1, s6 // (op 713) 

s_sub_i32 s1, s0, s6 // (op 713) 


s_mul_i32 s0, s14, s18 // (op 713) 

s_mul_i32 s6, s50, s0 // (op 713) 

s_mul_i32 s0, s1, s6 // (op 713) 

s_sub_u32 s6, s16, s0 // (op 713) 

s_mul_hi_i32 s0, s6, s13 // (op 713) 
s_add_i32 s7, s0, s6 // (op 713) 


s_ashr_i32 s0, s7, 31 // (op 713) 
s_cmp_eq_i32 s13, 0 // (op 713) 
s_mov_b32 s6, scc // (op 713) 
s_cmp_eq_i32 s6, 0 // (op 713) 
s_cselect_b32 s18, 0, -1 // (op 713) 

s_and_b32 s6, s49, 31 // (op 713) 
s_lshl_b32 s19, 1, s6 // (op 713) 
s_add_i32 s68, s19, s18 // (op 713) 


s_and_b32 s18, s0, s68 // (op 713) 


s_add_i32 s0, s7, s18 // (op 713) 


s_ashr_i32 s7, s0, s6 // (op 713) 


s_lshl_b32 s0, s49, 24 // (op 713) 
s_ashr_i32 s6, s0, 31 // (op 713) 

s_xor_b32 s0, s7, s6 // (op 713) 

s_sub_i32 s7, s0, s6 // (op 713) 


s_cmp_eq_i32 s1, 0 // (op 713) 
s_mov_b32 s0, scc // (op 713) 

s_cmp_eq_i32 s0, 0 // (op 713) 
s_cselect_b32 s1, s7, s51 // (op 713) 



v_lshl_add_u32 v23, s1, 4, v22 // (op 713) 


v_mul_lo_u32 v22, v20, s36 // (op 713) 
v_add_u32 v22, v22, v23 // (op 713) 


s_mul_i32 s1, s36, s35 // (op 715) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 715) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 715) least significant
s_mul_hi_u32 s6, s36, s34 // (op 715) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 715) most significant: sum
s_add_u32 s1, s1, s6 // (op 715) most significant: sum


s_mov_b32 s64, s38 // (op 715) reg expression
s_mov_b32 s65, s39 // (op 715) reg expression
s_mov_b32 s66, s0 // (op 715) reg expression
s_mov_b32 s67, 131072 // (op 715) call()


s_mul_i32 s1, s27, 6 // (op 370) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 370) least significant
s_mul_hi_u32 s51, s26, 6 // (op 370) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 370) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 370) 


v_mov_b32 v20, v12 // (op 370) reg expression
buffer_load_dwordx3 v[26:28], v20, s[56:59], 0 offen // (op 370) Load value
v_add_u32 v20, v20, v24 // (op 370) 
buffer_load_dwordx3 v[30:32], v20, s[56:59], 0 offen // (op 370) Load value




v_mov_b32 v20, v21 // (op 373) reg expression

buffer_load_ubyte v29, v20, s[64:67], 0 offen offset: 0 // (op 373) Load value

v_lshlrev_b32 v20, 4, v1 // (op 779) 
s_nop 0
v_readfirstlane_b32 s51, v20 // (op 779) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[2:3], 6 // (op 458) 
v_mov_b64 v[24:25], s[0:1] // (op 458) convert


v_mov_b64 v[34:35], v[16:17] // (op 458) reg expression
s_mov_b32 m0, s51 // (op 458) reg expression
buffer_load_dwordx4 v34, s[52:55], 0 offen offset: 0 lds // (op 458) Load value direct to lds
v_add_co_u32 v34, s[0:1], v34, v24 // (op 458) least significant half
v_addc_co_u32 v35, s[0:1], 0, v35, s[0:1] // (op 458) most significant half

s_add_u32 m0, m0, 1024 // (op 458) 
buffer_load_dwordx4 v34, s[52:55], 0 offen offset: 0 lds // (op 458) Load value direct to lds





v_mov_b32 v20, v14 // (op 379) reg expression

buffer_load_ubyte v23, v20, s[60:63], 0 offen offset: 0 // (op 379) Load value


s_mul_i32 s1, s27, 6 // (op 382) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 382) least significant
s_mul_hi_u32 s51, s26, 6 // (op 382) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 382) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 382) 


s_lshl_b64 s[0:1], s[26:27], 7 // (op 382) 
s_mul_i32 s7, s1, 6 // (op 382) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 382) least significant
s_mul_hi_u32 s51, s0, 6 // (op 382) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 382) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 382) 

v_add_u32 v20, v12, s0 // (op 382) 

buffer_load_dwordx3 v[34:36], v20, s[56:59], 0 offen // (op 382) Load value
v_add_u32 v20, v20, v24 // (op 382) 
buffer_load_dwordx3 v[38:40], v20, s[56:59], 0 offen // (op 382) Load value





s_lshl_b32 s51, s36, 2 // (op 385) 
v_add_u32 v12, v21, s51 // (op 385) 


buffer_load_ubyte v20, v12, s[64:67], 0 offen offset: 0 // (op 385) Load value


v_lshlrev_b32 v12, 4, v1 // (op 1011) 
s_nop 0
v_readfirstlane_b32 s51, v12 // (op 1011) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[2:3], 6 // (op 463) 
v_mov_b64 v[24:25], s[0:1] // (op 463) convert


s_lshl_b64 s[0:1], s[2:3], 7 // (op 463) 
v_mov_b32 v12, s0 // (op 463) 
v_mov_b32 v21, s1 // (op 463) 
v_add_co_u32 v42, s[6:7], v16, v12 // (op 463) least significant half
v_addc_co_u32 v43, s[6:7], v17, v21, s[6:7] // (op 463) most significant half




s_add_u32 s0, 2048, s51 // (op 463) 
s_mov_b32 m0, s0 // (op 463) reg expression
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 463) Load value direct to lds
v_add_co_u32 v42, s[6:7], v42, v24 // (op 463) least significant half
v_addc_co_u32 v43, s[6:7], 0, v43, s[6:7] // (op 463) most significant half

s_add_u32 m0, m0, 1024 // (op 463) 
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 463) Load value direct to lds







s_lshl_b32 s51, s30, 2 // (op 391) 
v_add_u32 v12, v14, s51 // (op 391) 


buffer_load_ubyte v21, v12, s[60:63], 0 offen offset: 0 // (op 391) Load value


v_lshlrev_b32 v12, 5, v1 // (op 832) 
v_mul_lo_u32 v14, v12, 6 // (op 832) 
v_lshrrev_b32 v33, 3, v14 // (op 832) 

v_lshrrev_b32 v14, 4, v12 // (op 832) 

v_lshlrev_b32 v12, 2, v14 // (op 832) 

v_add_u32 v14, v33, v12 // (op 832) 





s_waitcnt vmcnt(11)
ds_write_b96 v14, v[26:28] offset:4096 // (op 393) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v14, v[30:32] offset:4112 // (op 393) Store local data 

v_and_b32 v12, -16, v1 // (op 938) 
v_sub_u32 v14, v1, v12 // (op 938) 

v_and_b32 v12, 4294967280, v1 // (op 938) 
v_add_u32 v33, v12, v14 // (op 938) 





s_waitcnt vmcnt(9)
ds_write_b8 v33, v29 offset:6144 // (op 394) Store local data 

v_and_b32 v12, -16, v1 // (op 885) 
v_sub_u32 v14, v1, v12 // (op 885) 

v_and_b32 v12, 4294967280, v1 // (op 885) 
v_add_u32 v33, v12, v14 // (op 885) 





s_waitcnt vmcnt(6)
ds_write_b8 v33, v23 offset:6208 // (op 396) Store local data 

v_lshlrev_b32 v12, 4, v1 // (op 751) 
s_nop 0
v_readfirstlane_b32 s51, v12 // (op 751) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v12, 5, v1 // (op 803) 
v_mul_lo_u32 v14, v12, 6 // (op 803) 
v_lshrrev_b32 v33, 3, v14 // (op 803) 

v_lshrrev_b32 v14, 4, v12 // (op 803) 

v_lshlrev_b32 v12, 2, v14 // (op 803) 

v_add_u32 v14, v33, v12 // (op 803) 


v_and_b32 v12, -16, v1 // (op 856) 
v_sub_u32 v33, v1, v12 // (op 856) 

v_and_b32 v12, 4294967280, v1 // (op 856) 
v_add_u32 v37, v12, v33 // (op 856) 


v_and_b32 v12, -16, v1 // (op 909) 
v_sub_u32 v33, v1, v12 // (op 909) 

v_and_b32 v12, 4294967280, v1 // (op 909) 
v_add_u32 v16, v12, v33 // (op 909) 


v_lshlrev_b32 v12, 4, v1 // (op 996) 
s_nop 0
v_readfirstlane_b32 s0, v12 // (op 996) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v12, 5, v1 // (op 1035) 
v_mul_lo_u32 v17, v12, 6 // (op 1035) 
v_lshrrev_b32 v33, 3, v17 // (op 1035) 

v_lshrrev_b32 v17, 4, v12 // (op 1035) 

v_lshlrev_b32 v12, 2, v17 // (op 1035) 

v_add_u32 v17, v33, v12 // (op 1035) 


v_and_b32 v12, -16, v1 // (op 1059) 
v_sub_u32 v33, v1, v12 // (op 1059) 

v_and_b32 v12, 4294967280, v1 // (op 1059) 
v_add_u32 v24, v12, v33 // (op 1059) 


v_and_b32 v12, -16, v1 // (op 1083) 
v_sub_u32 v25, v1, v12 // (op 1083) 

v_and_b32 v12, 4294967280, v1 // (op 1083) 
v_add_u32 v33, v12, v25 // (op 1083) 


s_mov_b32 s1, 0 // (op 61) call()
s_ashr_i64 s[6:7], s[24:25], 7 // (op 60) 
s_mov_b32 s18, s6 // (op 60) convert

s_and_b32 s19, s18, 4294967294 // (op 60) 

s_cmp_lt_u32 s1, s19 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_60 if false)
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_60:

s_barrier  // (op 1132) 
s_add_u32 s1, s1, 2 // (op 62) 



s_mul_i32 s7, s27, 6 // (op 10) most significant: low of high * low
s_mul_i32 s6, s26, 6 // (op 10) least significant
s_mul_hi_u32 s18, s26, 6 // (op 10) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 10) most significant: sum

v_lshrrev_b64 v[42:43], 3, s[6:7] // (op 10) 


v_mov_b32 v12, v13 // (op 10) 
s_lshl_b64 s[6:7], s[26:27], 7 // (op 10) 
s_mul_i32 s19, s7, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s6, 6 // (op 10) least significant
s_mul_hi_u32 s68, s6, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 10) most significant: sum


s_lshr_b64 s[6:7], s[18:19], 3 // (op 10) 

s_lshl_b32 s18, s6, 1 // (op 10) 

v_add_u32 v25, v12, s18 // (op 10) 

ds_read_b64_tr_b8 v[44:45], v0 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:128 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:1024 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:1152 // (op 72) Transpose load local data 
buffer_load_dwordx3 v[26:28], v25, s[56:59], 0 offen // (op 10) Load value
v_add_u32 v25, v25, v42 // (op 10) 
buffer_load_dwordx3 v[30:32], v25, s[56:59], 0 offen // (op 10) Load value


s_waitcnt lgkmcnt(6)
s_barrier  // (op 1145) 





v_mov_b32 v25, v22 // (op 25) 
s_lshl_b32 s6, s36, 3 // (op 25) 
v_add_u32 v41, v25, s6 // (op 25) 


buffer_load_ubyte v29, v41, s[64:67], 0 offen offset: 0 // (op 25) Load value

ds_read_b96_tr_b6 v[52:54], v5 offset:4096 // (op 77) Transpose load local data 
ds_read_b96_tr_b6 v[55:57], v5 offset:4352 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1135) 

s_barrier  // (op 1159) 
s_lshl_b64 s[6:7], s[2:3], 6 // (op 442) 


ds_read_u8 v41, v4 offset:6208 // (op 82) Load local data 
v_mov_b64 v[42:43], s[6:7] // (op 442) convert


s_barrier  // (op 1173) 
v_mov_b64 v[58:59], v[18:19] // (op 442) 


s_lshl_b64 s[6:7], s[2:3], 7 // (op 442) 
s_lshl_b64 s[18:19], s[6:7], 1 // (op 442) 

v_mov_b32 v60, s18 // (op 442) 
v_mov_b32 v61, s19 // (op 442) 
v_add_co_u32 v62, s[6:7], v58, v60 // (op 442) least significant half
v_addc_co_u32 v63, s[6:7], v59, v61, s[6:7] // (op 442) most significant half




s_mov_b32 m0, s51 // (op 442) reg expression
ds_read_u8 v60, v7 offset:6144 // (op 87) Load local data 
buffer_load_dwordx4 v62, s[52:55], 0 offen offset: 0 lds // (op 442) Load value direct to lds
v_add_co_u32 v62, s[6:7], v62, v42 // (op 442) least significant half
v_addc_co_u32 v63, s[6:7], 0, v63, s[6:7] // (op 442) most significant half

s_add_u32 m0, m0, 1024 // (op 442) 
buffer_load_dwordx4 v62, s[52:55], 0 offen offset: 0 lds // (op 442) Load value direct to lds


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[52:57], a[0:3], v41, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 90) 






v_mov_b32 v41, v15 // (op 17) 
s_lshl_b32 s6, s30, 3 // (op 17) 
v_add_u32 v42, v41, s6 // (op 17) 


buffer_load_ubyte v23, v42, s[60:63], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1196) 



s_waitcnt vmcnt(11)
ds_write_b96 v17, v[34:36] offset:6272 // (op 257) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v17, v[38:40] offset:6288 // (op 257) Store local data 
s_barrier  // (op 1208) 



s_waitcnt vmcnt(9)
ds_write_b8 v33, v20 offset:8320 // (op 259) Store local data 
s_barrier  // (op 1202) 



s_waitcnt vmcnt(6)
ds_write_b8 v24, v21 offset:8384 // (op 258) Store local data 

s_barrier  // (op 1187) 
s_mul_i32 s7, s27, 6 // (op 260) most significant: low of high * low



s_mul_i32 s6, s26, 6 // (op 260) least significant
s_mul_hi_u32 s18, s26, 6 // (op 260) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 260) most significant: sum

v_lshrrev_b64 v[42:43], 3, s[6:7] // (op 260) 


s_lshl_b64 s[6:7], s[26:27], 7 // (op 260) 
s_mul_i32 s19, s7, 6 // (op 260) most significant: low of high * low
s_mul_i32 s18, s6, 6 // (op 260) least significant
s_mul_hi_u32 s68, s6, 6 // (op 260) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 260) most significant: sum


s_lshr_b64 s[6:7], s[18:19], 3 // (op 260) 

s_mul_i32 s18, 3, s6 // (op 260) 

v_add_u32 v44, v12, s18 // (op 260) 

ds_read_b64_tr_b8 v[50:51], v3 offset:2048 // (op 249) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v3 offset:2176 // (op 249) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v3 offset:3072 // (op 249) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v3 offset:3200 // (op 249) Transpose load local data 
buffer_load_dwordx3 v[34:36], v44, s[56:59], 0 offen // (op 260) Load value
v_add_u32 v44, v44, v42 // (op 260) 
buffer_load_dwordx3 v[38:40], v44, s[56:59], 0 offen // (op 260) Load value


s_waitcnt lgkmcnt(6)
s_barrier  // (op 1193) 






s_lshl_b32 s6, s36, 2 // (op 262) 
s_mul_i32 s7, 3, s6 // (op 262) 

v_add_u32 v12, v25, s7 // (op 262) 


buffer_load_ubyte v20, v12, s[64:67], 0 offen offset: 0 // (op 262) Load value

s_lshl_b64 s[6:7], s[26:27], 7 // (op 577) 

s_mul_i32 s19, s7, 6 // (op 577) most significant: low of high * low
s_mul_i32 s18, s6, 6 // (op 577) least significant
s_mul_hi_u32 s68, s6, 6 // (op 577) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 577) most significant: sum


s_lshl_b32 s6, s36, 3 // (op 705) 
v_add_u32 v22, v22, s6 // (op 705) 

s_lshr_b64 s[6:7], s[18:19], 3 // (op 577) 

s_lshl_b32 s18, s6, 1 // (op 577) 

v_add_u32 v13, v13, s18 // (op 577) 

ds_read_b96_tr_b6 v[42:44], v9 offset:6272 // (op 251) Transpose load local data 
ds_read_b96_tr_b6 v[45:47], v9 offset:6528 // (op 251) Transpose load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1190) 

s_barrier  // (op 1199) 
s_barrier  // (op 1263) 


ds_read_u8 v12, v8 offset:8384 // (op 253) Load local data 
s_lshl_b64 s[6:7], s[2:3], 6 // (op 453) 
v_mov_b64 v[48:49], s[6:7] // (op 453) convert


s_barrier  // (op 1205) 
s_barrier  // (op 1269) 


s_lshl_b64 s[6:7], s[2:3], 7 // (op 453) 
s_mul_i32 s19, 3, s7 // (op 453) most significant: low of low * high
s_mul_i32 s18, 3, s6 // (op 453) least significant
s_mul_hi_u32 s68, 3, s6 // (op 453) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 453) most significant: sum


v_mov_b32 v25, s18 // (op 453) 
v_mov_b32 v60, s19 // (op 453) 
v_add_co_u32 v62, s[6:7], v58, v25 // (op 453) least significant half
v_addc_co_u32 v63, s[6:7], v59, v60, s[6:7] // (op 453) most significant half




s_add_u32 s6, 2048, s0 // (op 453) 
s_nop 0
s_mov_b32 m0, s6 // (op 453) reg expression
ds_read_u8 v25, v11 offset:8320 // (op 255) Load local data 
buffer_load_dwordx4 v62, s[52:55], 0 offen offset: 0 lds // (op 453) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v48 // (op 453) least significant half
s_barrier  // (op 1275) 
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 453) most significant half

s_add_u32 m0, m0, 1024 // (op 453) 
buffer_load_dwordx4 v62, s[52:55], 0 offen offset: 0 lds // (op 453) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:57], v[42:47], a[0:3], v12, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 256) 







s_lshl_b32 s6, s30, 2 // (op 261) 
s_mul_i32 s7, 3, s6 // (op 261) 

v_add_u32 v12, v41, s7 // (op 261) 


buffer_load_ubyte v21, v12, s[60:63], 0 offen offset: 0 // (op 261) Load value

s_lshl_b64 s[6:7], s[2:3], 7 // (op 513) 

s_lshl_b64 s[18:19], s[6:7], 1 // (op 513) 

s_barrier  // (op 1148) 
s_lshl_b32 s6, s30, 3 // (op 641) 



v_add_u32 v15, v15, s6 // (op 641) 

v_mov_b32 v12, s18 // (op 513) 
v_mov_b32 v25, s19 // (op 513) 
v_add_co_u32 v18, s[6:7], v18, v12 // (op 513) least significant half
v_addc_co_u32 v19, s[6:7], v19, v25, s[6:7] // (op 513) most significant half




s_waitcnt vmcnt(11)
ds_write_b96 v14, v[26:28] offset:4096 // (op 13) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v14, v[30:32] offset:4112 // (op 13) Store local data 
s_barrier  // (op 1176) 



s_waitcnt vmcnt(9)
ds_write_b8 v16, v29 offset:6144 // (op 30) Store local data 
s_barrier  // (op 1162) 



s_waitcnt vmcnt(6)
ds_write_b8 v37, v23 offset:6208 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[24:25], 7 // (op 60) 
s_mov_b32 s18, s6 // (op 60) convert

s_and_b32 s19, s18, 4294967294 // (op 60) 

s_cmp_lt_u32 s1, s19 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_60 if true)
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_60:
















s_barrier  // (op 1223) 
s_mul_hi_i32 s51, s16, s9 // (op 476) 
s_add_i32 s0, s51, s16 // (op 476) 

s_ashr_i32 s1, s0, 31 // (op 476) 
s_cmp_eq_i32 s9, 0 // (op 476) 
s_mov_b32 s51, scc // (op 476) 
s_cmp_eq_i32 s51, 0 // (op 476) 
s_cselect_b32 s6, 0, -1 // (op 476) 

s_and_b32 s7, s10, 31 // (op 476) 
s_lshl_b32 s51, 1, s7 // (op 476) 
s_add_i32 s18, s51, s6 // (op 476) 


s_and_b32 s6, s1, s18 // (op 476) 


s_add_i32 s1, s0, s6 // (op 476) 


s_ashr_i32 s0, s1, s7 // (op 476) 


s_lshl_b32 s1, s10, 24 // (op 476) 
s_ashr_i32 s51, s1, 31 // (op 476) 

s_xor_b32 s1, s0, s51 // (op 476) 

s_sub_i32 s0, s1, s51 // (op 476) 


s_mul_hi_i32 s1, s0, s11 // (op 476) 
s_add_i32 s51, s1, s0 // (op 476) 

s_ashr_i32 s1, s51, 31 // (op 476) 
s_cmp_eq_i32 s11, 0 // (op 476) 
s_mov_b32 s6, scc // (op 476) 
s_cmp_eq_i32 s6, 0 // (op 476) 
s_cselect_b32 s7, 0, -1 // (op 476) 

s_and_b32 s6, s12, 31 // (op 476) 
s_lshl_b32 s18, 1, s6 // (op 476) 
s_add_i32 s19, s18, s7 // (op 476) 


s_and_b32 s7, s1, s19 // (op 476) 


s_add_i32 s1, s51, s7 // (op 476) 


s_ashr_i32 s7, s1, s6 // (op 476) 


s_lshl_b32 s1, s12, 24 // (op 476) 
s_ashr_i32 s6, s1, 31 // (op 476) 

s_xor_b32 s1, s7, s6 // (op 476) 

s_sub_i32 s7, s1, s6 // (op 476) 


s_mul_i32 s1, s0, s14 // (op 476) 

s_sub_u32 s0, s16, s1 // (op 476) 

s_mul_i32 s1, s7, s14 // (op 476) 
s_add_u32 s1, s1, s0 // (op 476) 


s_mul_hi_i32 s0, s16, s8 // (op 476) 
s_add_i32 s51, s0, s16 // (op 476) 

s_ashr_i32 s0, s51, 31 // (op 476) 
s_cmp_eq_i32 s8, 0 // (op 476) 
s_mov_b32 s6, scc // (op 476) 
s_cmp_eq_i32 s6, 0 // (op 476) 
s_cselect_b32 s7, 0, -1 // (op 476) 

s_and_b32 s6, s15, 31 // (op 476) 
s_lshl_b32 s18, 1, s6 // (op 476) 
s_add_i32 s19, s18, s7 // (op 476) 


s_and_b32 s7, s0, s19 // (op 476) 


s_add_i32 s0, s51, s7 // (op 476) 


s_ashr_i32 s7, s0, s6 // (op 476) 


s_lshl_b32 s0, s15, 24 // (op 476) 
s_ashr_i32 s6, s0, 31 // (op 476) 

s_xor_b32 s0, s7, s6 // (op 476) 

s_sub_i32 s7, s0, s6 // (op 476) 


s_add_u32 s18, s4, 16 // (op 476) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 476) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 476) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 476) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 476) 

s_mul_i32 s0, s14, s18 // (op 476) 

s_mul_i32 s6, s50, s0 // (op 476) 

s_mul_i32 s0, s7, s6 // (op 476) 

s_sub_u32 s6, s16, s0 // (op 476) 

s_mov_b32 s0, s6 // (op 476) 
s_mul_hi_i32 s51, s0, s13 // (op 476) 
s_add_i32 s18, s51, s0 // (op 476) 


s_ashr_i32 s0, s18, 31 // (op 476) 
s_cmp_eq_i32 s13, 0 // (op 476) 
s_mov_b32 s19, scc // (op 476) 
s_cmp_eq_i32 s19, 0 // (op 476) 
s_cselect_b32 s51, 0, -1 // (op 476) 

s_and_b32 s19, s49, 31 // (op 476) 
s_lshl_b32 s68, 1, s19 // (op 476) 
s_add_i32 s69, s68, s51 // (op 476) 


s_and_b32 s51, s0, s69 // (op 476) 


s_add_i32 s0, s18, s51 // (op 476) 


s_ashr_i32 s18, s0, s19 // (op 476) 


s_lshl_b32 s0, s49, 24 // (op 476) 
s_ashr_i32 s19, s0, 31 // (op 476) 

s_xor_b32 s0, s18, s19 // (op 476) 

s_sub_i32 s18, s0, s19 // (op 476) 


s_mul_i32 s0, s48, s14 // (op 476) 
s_add_u32 s68, s20, 16 // (op 476) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 476) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 476) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 476) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 476) 

s_sub_i32 s19, s68, s0 // (op 476) 

s_mul_i32 s51, s18, s19 // (op 476) 


s_sub_u32 s18, s6, s51 // (op 476) 


s_add_u32 s6, s18, s0 // (op 476) 


s_cmp_eq_i32 s7, 0 // (op 476) 
s_mov_b32 s0, scc // (op 476) 

s_cmp_eq_i32 s0, 0 // (op 476) 
s_cselect_b32 s7, s6, s1 // (op 476) 



s_lshl_b32 s6, s7, 4 // (op 476) 

s_mov_b32 s1, 0 // (op 476) convert
s_mov_b32 s0, s6 // (op 476) convert

s_ashr_i64 s[6:7], s[24:25], 7 // (op 476) 
s_lshr_b32 s51, s6, 1 // (op 476) 

v_lshl_add_u32 v37, s51, 8, v1 // (op 476) 

v_mul_lo_u32 v13, v37, s3 // (op 476) most significant: low of low * high
v_mul_hi_u32 v14, v37, s2 // (op 476) most significant: high of low * low
v_mul_lo_u32 v12, v37, s2 // (op 476) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 476) most significant: sum


v_mov_b32 v37, s0 // (op 476) 
v_mov_b32 v14, s1 // (op 476) 
v_add_co_u32 v18, s[6:7], v37, v12 // (op 476) least significant half
v_addc_co_u32 v19, s[6:7], v14, v13, s[6:7] // (op 476) most significant half





s_mul_i32 s1, s2, s25 // (op 478) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 478) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 478) least significant
s_mul_hi_u32 s6, s2, s24 // (op 478) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 478) most significant: sum
s_add_u32 s1, s1, s6 // (op 478) most significant: sum


s_mov_b32 s52, s22 // (op 478) reg expression
s_mov_b32 s53, s23 // (op 478) reg expression
s_mov_b32 s54, s0 // (op 478) reg expression
s_mov_b32 s55, 131072 // (op 478) call()

s_mul_hi_i32 s51, s16, s9 // (op 543) 
v_lshlrev_b32 v37, 1, v1 // (op 543) 
s_add_i32 s0, s51, s16 // (op 543) 

s_ashr_i32 s1, s0, 31 // (op 543) 
s_cmp_eq_i32 s9, 0 // (op 543) 
s_mov_b32 s51, scc // (op 543) 
s_cmp_eq_i32 s51, 0 // (op 543) 
s_cselect_b32 s6, 0, -1 // (op 543) 

s_and_b32 s7, s10, 31 // (op 543) 
s_lshl_b32 s51, 1, s7 // (op 543) 
s_add_i32 s18, s51, s6 // (op 543) 


s_and_b32 s6, s1, s18 // (op 543) 


s_add_i32 s1, s0, s6 // (op 543) 


s_ashr_i32 s0, s1, s7 // (op 543) 


s_lshl_b32 s1, s10, 24 // (op 543) 
s_ashr_i32 s51, s1, 31 // (op 543) 

s_xor_b32 s1, s0, s51 // (op 543) 

s_sub_i32 s0, s1, s51 // (op 543) 


s_mul_hi_i32 s1, s0, s11 // (op 543) 
s_add_i32 s51, s1, s0 // (op 543) 

s_ashr_i32 s1, s51, 31 // (op 543) 
s_cmp_eq_i32 s11, 0 // (op 543) 
s_mov_b32 s6, scc // (op 543) 
s_cmp_eq_i32 s6, 0 // (op 543) 
s_cselect_b32 s7, 0, -1 // (op 543) 

s_and_b32 s6, s12, 31 // (op 543) 
s_lshl_b32 s18, 1, s6 // (op 543) 
s_add_i32 s19, s18, s7 // (op 543) 


s_and_b32 s7, s1, s19 // (op 543) 


s_add_i32 s1, s51, s7 // (op 543) 


s_ashr_i32 s7, s1, s6 // (op 543) 


s_lshl_b32 s1, s12, 24 // (op 543) 
s_ashr_i32 s6, s1, 31 // (op 543) 

s_xor_b32 s1, s7, s6 // (op 543) 

s_sub_i32 s7, s1, s6 // (op 543) 


s_add_u32 s18, s4, 16 // (op 543) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 543) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 543) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 543) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 543) 

s_mul_i32 s1, s7, s18 // (op 543) 

s_sub_i32 s51, s0, s1 // (op 543) 


s_mul_hi_i32 s0, s16, s8 // (op 543) 
s_add_i32 s1, s0, s16 // (op 543) 

s_ashr_i32 s0, s1, 31 // (op 543) 
s_cmp_eq_i32 s8, 0 // (op 543) 
s_mov_b32 s6, scc // (op 543) 
s_cmp_eq_i32 s6, 0 // (op 543) 
s_cselect_b32 s7, 0, -1 // (op 543) 

s_and_b32 s6, s15, 31 // (op 543) 
s_lshl_b32 s68, 1, s6 // (op 543) 
s_add_i32 s69, s68, s7 // (op 543) 


s_and_b32 s7, s0, s69 // (op 543) 


s_add_i32 s0, s1, s7 // (op 543) 


s_ashr_i32 s1, s0, s6 // (op 543) 


s_lshl_b32 s0, s15, 24 // (op 543) 
s_ashr_i32 s6, s0, 31 // (op 543) 

s_xor_b32 s0, s1, s6 // (op 543) 

s_sub_i32 s1, s0, s6 // (op 543) 


s_mul_i32 s0, s14, s18 // (op 543) 

s_mul_i32 s6, s50, s0 // (op 543) 

s_mul_i32 s0, s1, s6 // (op 543) 

s_sub_u32 s6, s16, s0 // (op 543) 

s_mul_hi_i32 s0, s6, s13 // (op 543) 
s_add_i32 s7, s0, s6 // (op 543) 


s_ashr_i32 s0, s7, 31 // (op 543) 
s_cmp_eq_i32 s13, 0 // (op 543) 
s_mov_b32 s6, scc // (op 543) 
s_cmp_eq_i32 s6, 0 // (op 543) 
s_cselect_b32 s18, 0, -1 // (op 543) 

s_and_b32 s6, s49, 31 // (op 543) 
s_lshl_b32 s19, 1, s6 // (op 543) 
s_add_i32 s68, s19, s18 // (op 543) 


s_and_b32 s18, s0, s68 // (op 543) 


s_add_i32 s0, s7, s18 // (op 543) 


s_ashr_i32 s7, s0, s6 // (op 543) 


s_lshl_b32 s0, s49, 24 // (op 543) 
s_ashr_i32 s6, s0, 31 // (op 543) 

s_xor_b32 s0, s7, s6 // (op 543) 

s_sub_i32 s7, s0, s6 // (op 543) 


s_cmp_eq_i32 s1, 0 // (op 543) 
s_mov_b32 s0, scc // (op 543) 

s_cmp_eq_i32 s0, 0 // (op 543) 
s_cselect_b32 s1, s7, s51 // (op 543) 



s_lshl_b32 s0, s1, 4 // (op 543) 

s_mov_b32 s7, 0 // (op 543) convert
s_mov_b32 s6, s0 // (op 543) convert

s_ashr_i64 s[0:1], s[24:25], 7 // (op 543) 
s_lshr_b32 s51, s0, 1 // (op 543) 

v_lshl_add_u32 v12, s51, 8, v37 // (op 543) 


v_mul_lo_u32 v17, v12, s27 // (op 543) most significant: low of low * high
v_mul_hi_u32 v37, v12, s26 // (op 543) most significant: high of low * low
v_mul_lo_u32 v16, v12, s26 // (op 543) least significant: low of low * low
v_add_u32 v17, v17, v37 // (op 543) most significant: sum

v_mov_b32 v37, s6 // (op 543) 
v_mov_b32 v13, s7 // (op 543) 
v_add_co_u32 v16, s[0:1], v16, v37 // (op 543) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 543) most significant half





v_mul_lo_u32 v13, v17, 6 // (op 543) most significant: low of high * low
v_mul_hi_u32 v37, v16, 6 // (op 543) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 543) least significant: low of low * low
v_add_u32 v13, v13, v37 // (op 543) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 543) 

v_mov_b32 v37, v14 // (op 543) convert

s_mul_i32 s1, s25, 6 // (op 545) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 545) least significant
s_mul_hi_u32 s51, s24, 6 // (op 545) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 545) most significant: sum

s_mul_i32 s7, s26, s1 // (op 545) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 545) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 545) least significant
s_mul_hi_u32 s18, s26, s0 // (op 545) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 545) most significant: sum
s_add_u32 s7, s7, s18 // (op 545) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 545) 

s_mov_b32 s56, s28 // (op 545) reg expression
s_mov_b32 s57, s29 // (op 545) reg expression
s_mov_b32 s58, s0 // (op 545) reg expression
s_mov_b32 s59, 131072 // (op 545) call()

s_mul_hi_i32 s51, s16, s9 // (op 607) 
v_and_b32 v12, -16, v1 // (op 607) 
s_add_i32 s0, s51, s16 // (op 607) 

v_sub_u32 v13, v1, v12 // (op 607) 

s_ashr_i32 s1, s0, 31 // (op 607) 
v_lshrrev_b32 v12, 4, v1 // (op 607) 
s_cmp_eq_i32 s9, 0 // (op 607) 
s_mov_b32 s51, scc // (op 607) 
s_cmp_eq_i32 s51, 0 // (op 607) 
s_cselect_b32 s6, 0, -1 // (op 607) 

s_and_b32 s7, s10, 31 // (op 607) 
s_lshl_b32 s51, 1, s7 // (op 607) 
s_add_i32 s18, s51, s6 // (op 607) 


s_and_b32 s6, s1, s18 // (op 607) 


s_add_i32 s1, s0, s6 // (op 607) 


s_ashr_i32 s0, s1, s7 // (op 607) 


s_lshl_b32 s1, s10, 24 // (op 607) 
s_ashr_i32 s51, s1, 31 // (op 607) 

s_xor_b32 s1, s0, s51 // (op 607) 

s_sub_i32 s0, s1, s51 // (op 607) 


s_mul_hi_i32 s1, s0, s11 // (op 607) 
s_add_i32 s51, s1, s0 // (op 607) 

s_ashr_i32 s1, s51, 31 // (op 607) 
s_cmp_eq_i32 s11, 0 // (op 607) 
s_mov_b32 s6, scc // (op 607) 
s_cmp_eq_i32 s6, 0 // (op 607) 
s_cselect_b32 s7, 0, -1 // (op 607) 

s_and_b32 s6, s12, 31 // (op 607) 
s_lshl_b32 s18, 1, s6 // (op 607) 
s_add_i32 s19, s18, s7 // (op 607) 


s_and_b32 s7, s1, s19 // (op 607) 


s_add_i32 s1, s51, s7 // (op 607) 


s_ashr_i32 s7, s1, s6 // (op 607) 


s_lshl_b32 s1, s12, 24 // (op 607) 
s_ashr_i32 s6, s1, 31 // (op 607) 

s_xor_b32 s1, s7, s6 // (op 607) 

s_sub_i32 s7, s1, s6 // (op 607) 


s_mul_i32 s1, s0, s14 // (op 607) 

s_sub_u32 s0, s16, s1 // (op 607) 

s_mul_i32 s1, s7, s14 // (op 607) 
s_add_u32 s1, s1, s0 // (op 607) 


s_mul_hi_i32 s0, s16, s8 // (op 607) 
s_add_i32 s51, s0, s16 // (op 607) 

s_ashr_i32 s0, s51, 31 // (op 607) 
s_cmp_eq_i32 s8, 0 // (op 607) 
s_mov_b32 s6, scc // (op 607) 
s_cmp_eq_i32 s6, 0 // (op 607) 
s_cselect_b32 s7, 0, -1 // (op 607) 

s_and_b32 s6, s15, 31 // (op 607) 
s_lshl_b32 s18, 1, s6 // (op 607) 
s_add_i32 s19, s18, s7 // (op 607) 


s_and_b32 s7, s0, s19 // (op 607) 


s_add_i32 s0, s51, s7 // (op 607) 


s_ashr_i32 s7, s0, s6 // (op 607) 


s_lshl_b32 s0, s15, 24 // (op 607) 
s_ashr_i32 s6, s0, 31 // (op 607) 

s_xor_b32 s0, s7, s6 // (op 607) 

s_sub_i32 s7, s0, s6 // (op 607) 


s_add_u32 s18, s4, 16 // (op 607) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 607) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 607) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 607) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 607) 

s_mul_i32 s0, s14, s18 // (op 607) 

s_mul_i32 s6, s50, s0 // (op 607) 

s_mul_i32 s0, s7, s6 // (op 607) 

s_sub_u32 s6, s16, s0 // (op 607) 

s_mov_b32 s0, s6 // (op 607) 
s_mul_hi_i32 s51, s0, s13 // (op 607) 
s_add_i32 s18, s51, s0 // (op 607) 


s_ashr_i32 s0, s18, 31 // (op 607) 
s_cmp_eq_i32 s13, 0 // (op 607) 
s_mov_b32 s19, scc // (op 607) 
s_cmp_eq_i32 s19, 0 // (op 607) 
s_cselect_b32 s51, 0, -1 // (op 607) 

s_and_b32 s19, s49, 31 // (op 607) 
s_lshl_b32 s68, 1, s19 // (op 607) 
s_add_i32 s69, s68, s51 // (op 607) 


s_and_b32 s51, s0, s69 // (op 607) 


s_add_i32 s0, s18, s51 // (op 607) 


s_ashr_i32 s18, s0, s19 // (op 607) 


s_lshl_b32 s0, s49, 24 // (op 607) 
s_ashr_i32 s19, s0, 31 // (op 607) 

s_xor_b32 s0, s18, s19 // (op 607) 

s_sub_i32 s18, s0, s19 // (op 607) 


s_mul_i32 s0, s48, s14 // (op 607) 
s_add_u32 s68, s20, 16 // (op 607) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 607) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 607) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 607) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 607) 

s_sub_i32 s19, s68, s0 // (op 607) 

s_mul_i32 s51, s18, s19 // (op 607) 


s_sub_u32 s18, s6, s51 // (op 607) 


s_add_u32 s6, s18, s0 // (op 607) 


s_cmp_eq_i32 s7, 0 // (op 607) 
s_mov_b32 s0, scc // (op 607) 

s_cmp_eq_i32 s0, 0 // (op 607) 
s_cselect_b32 s7, s6, s1 // (op 607) 



v_lshl_add_u32 v14, s7, 4, v13 // (op 607) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 607) 
s_lshr_b32 s51, s0, 1 // (op 607) 

v_lshl_add_u32 v13, s51, 3, v12 // (op 607) 


v_mul_lo_u32 v12, v13, s30 // (op 607) 

v_add_u32 v13, v14, v12 // (op 607) 


s_mul_i32 s1, s30, s35 // (op 609) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 609) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 609) least significant
s_mul_hi_u32 s6, s30, s34 // (op 609) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 609) most significant: sum
s_add_u32 s1, s1, s6 // (op 609) most significant: sum


s_mov_b32 s60, s32 // (op 609) reg expression
s_mov_b32 s61, s33 // (op 609) reg expression
s_mov_b32 s62, s0 // (op 609) reg expression
s_mov_b32 s63, 131072 // (op 609) call()

s_mul_hi_i32 s51, s16, s9 // (op 671) 
v_and_b32 v12, -16, v1 // (op 671) 
s_add_i32 s0, s51, s16 // (op 671) 

v_sub_u32 v14, v1, v12 // (op 671) 

s_ashr_i32 s1, s0, 31 // (op 671) 
v_lshrrev_b32 v12, 4, v1 // (op 671) 
s_cmp_eq_i32 s9, 0 // (op 671) 
s_mov_b32 s51, scc // (op 671) 
s_cmp_eq_i32 s51, 0 // (op 671) 
s_cselect_b32 s6, 0, -1 // (op 671) 

s_and_b32 s7, s10, 31 // (op 671) 
s_lshl_b32 s51, 1, s7 // (op 671) 
s_add_i32 s18, s51, s6 // (op 671) 


s_and_b32 s6, s1, s18 // (op 671) 


s_add_i32 s1, s0, s6 // (op 671) 


s_ashr_i32 s0, s1, s7 // (op 671) 


s_lshl_b32 s1, s10, 24 // (op 671) 
s_ashr_i32 s51, s1, 31 // (op 671) 

s_xor_b32 s1, s0, s51 // (op 671) 

s_sub_i32 s0, s1, s51 // (op 671) 


s_mul_hi_i32 s1, s0, s11 // (op 671) 
s_add_i32 s51, s1, s0 // (op 671) 

s_ashr_i32 s1, s51, 31 // (op 671) 
s_cmp_eq_i32 s11, 0 // (op 671) 
s_mov_b32 s6, scc // (op 671) 
s_cmp_eq_i32 s6, 0 // (op 671) 
s_cselect_b32 s7, 0, -1 // (op 671) 

s_and_b32 s6, s12, 31 // (op 671) 
s_lshl_b32 s18, 1, s6 // (op 671) 
s_add_i32 s19, s18, s7 // (op 671) 


s_and_b32 s7, s1, s19 // (op 671) 


s_add_i32 s1, s51, s7 // (op 671) 


s_ashr_i32 s7, s1, s6 // (op 671) 


s_lshl_b32 s1, s12, 24 // (op 671) 
s_ashr_i32 s6, s1, 31 // (op 671) 

s_xor_b32 s1, s7, s6 // (op 671) 

s_sub_i32 s7, s1, s6 // (op 671) 


s_add_u32 s18, s4, 16 // (op 671) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 671) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 671) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 671) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 671) 

s_mul_i32 s1, s7, s18 // (op 671) 

s_sub_i32 s51, s0, s1 // (op 671) 


s_mul_hi_i32 s0, s16, s8 // (op 671) 
s_add_i32 s1, s0, s16 // (op 671) 

s_ashr_i32 s0, s1, 31 // (op 671) 
s_cmp_eq_i32 s8, 0 // (op 671) 
s_mov_b32 s6, scc // (op 671) 
s_cmp_eq_i32 s6, 0 // (op 671) 
s_cselect_b32 s7, 0, -1 // (op 671) 

s_and_b32 s6, s15, 31 // (op 671) 
s_lshl_b32 s68, 1, s6 // (op 671) 
s_add_i32 s69, s68, s7 // (op 671) 


s_and_b32 s7, s0, s69 // (op 671) 


s_add_i32 s0, s1, s7 // (op 671) 


s_ashr_i32 s1, s0, s6 // (op 671) 


s_lshl_b32 s0, s15, 24 // (op 671) 
s_ashr_i32 s6, s0, 31 // (op 671) 

s_xor_b32 s0, s1, s6 // (op 671) 

s_sub_i32 s1, s0, s6 // (op 671) 


s_mul_i32 s0, s14, s18 // (op 671) 

s_mul_i32 s6, s50, s0 // (op 671) 

s_mul_i32 s0, s1, s6 // (op 671) 

s_sub_u32 s6, s16, s0 // (op 671) 

s_mul_hi_i32 s0, s6, s13 // (op 671) 
s_add_i32 s7, s0, s6 // (op 671) 


s_ashr_i32 s0, s7, 31 // (op 671) 
s_cmp_eq_i32 s13, 0 // (op 671) 
s_mov_b32 s6, scc // (op 671) 
s_cmp_eq_i32 s6, 0 // (op 671) 
s_cselect_b32 s18, 0, -1 // (op 671) 

s_and_b32 s6, s49, 31 // (op 671) 
s_lshl_b32 s19, 1, s6 // (op 671) 
s_add_i32 s68, s19, s18 // (op 671) 


s_and_b32 s18, s0, s68 // (op 671) 


s_add_i32 s0, s7, s18 // (op 671) 


s_ashr_i32 s7, s0, s6 // (op 671) 


s_lshl_b32 s0, s49, 24 // (op 671) 
s_ashr_i32 s6, s0, 31 // (op 671) 

s_xor_b32 s0, s7, s6 // (op 671) 

s_sub_i32 s7, s0, s6 // (op 671) 


s_cmp_eq_i32 s1, 0 // (op 671) 
s_mov_b32 s0, scc // (op 671) 

s_cmp_eq_i32 s0, 0 // (op 671) 
s_cselect_b32 s1, s7, s51 // (op 671) 



v_lshl_add_u32 v15, s1, 4, v14 // (op 671) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 671) 
s_lshr_b32 s51, s0, 1 // (op 671) 

v_lshl_add_u32 v14, s51, 3, v12 // (op 671) 


v_mul_lo_u32 v12, v14, s36 // (op 671) 
v_add_u32 v12, v12, v15 // (op 671) 


s_mul_i32 s1, s36, s35 // (op 673) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 673) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 673) least significant
s_mul_hi_u32 s6, s36, s34 // (op 673) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 673) most significant: sum
s_add_u32 s1, s1, s6 // (op 673) most significant: sum


s_mov_b32 s64, s38 // (op 673) reg expression
s_mov_b32 s65, s39 // (op 673) reg expression
s_mov_b32 s66, s0 // (op 673) reg expression
s_mov_b32 s67, 131072 // (op 673) call()

v_lshlrev_b32 v14, 4, v1 // (op 764) 
s_nop 0
v_readfirstlane_b32 s51, v14 // (op 764) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v14, 5, v1 // (op 816) 
v_mul_lo_u32 v15, v14, 6 // (op 816) 
v_lshrrev_b32 v16, 3, v15 // (op 816) 

v_lshrrev_b32 v15, 4, v14 // (op 816) 

v_lshlrev_b32 v14, 2, v15 // (op 816) 

v_add_u32 v15, v16, v14 // (op 816) 


v_and_b32 v14, -16, v1 // (op 869) 
v_sub_u32 v16, v1, v14 // (op 869) 

v_and_b32 v14, 4294967280, v1 // (op 869) 
v_add_u32 v17, v14, v16 // (op 869) 


v_and_b32 v14, -16, v1 // (op 922) 
v_sub_u32 v16, v1, v14 // (op 922) 

v_and_b32 v14, 4294967280, v1 // (op 922) 
v_add_u32 v22, v14, v16 // (op 922) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 176) 
s_and_b32 s6, s0, 4294967294 // (op 176) 

s_ashr_i64 s[0:1], s[24:25], 7 // (op 175) 
s_mov_b32 s7, s0 // (op 175) convert

s_cmp_lt_u32 s6, s7 // (op 175) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_175 // (op 175) Condition: Top (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_175 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_175:
s_barrier  // (op 1138) 
s_add_u32 s6, s6, 1 // (op 177) 

s_lshl_b64 s[0:1], s[2:3], 6 // (op 447) 
v_mov_b64 v[20:21], s[0:1] // (op 447) convert


v_mov_b64 v[24:25], v[18:19] // (op 447) 
v_mov_b64 v[26:27], v[24:25] // (op 447) reg expression
s_mov_b32 m0, s51 // (op 447) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 447) Load value direct to lds
v_add_co_u32 v26, s[0:1], v26, v20 // (op 447) least significant half
v_addc_co_u32 v27, s[0:1], 0, v27, s[0:1] // (op 447) most significant half

s_add_u32 m0, m0, 1024 // (op 447) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 447) Load value direct to lds



s_lshl_b64 s[0:1], s[2:3], 7 // (op 466) 
v_mov_b32 v14, s0 // (op 466) 
v_mov_b32 v16, s1 // (op 466) 
v_add_co_u32 v18, s[18:19], v18, v14 // (op 466) least significant half
v_addc_co_u32 v19, s[18:19], v19, v16, s[18:19] // (op 466) most significant half





s_mul_i32 s1, s27, 6 // (op 194) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 194) least significant
s_mul_hi_u32 s7, s26, 6 // (op 194) most significant: high of low * low
s_add_u32 s1, s1, s7 // (op 194) most significant: sum

v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 194) 


v_mov_b32 v14, v37 // (op 194) 
v_mov_b32 v16, v14 // (op 194) reg expression
buffer_load_dwordx3 v[24:26], v16, s[56:59], 0 offen // (op 194) Load value
v_add_u32 v16, v16, v20 // (op 194) 
buffer_load_dwordx3 v[28:30], v16, s[56:59], 0 offen // (op 194) Load value



s_lshl_b64 s[0:1], s[26:27], 7 // (op 535) 
s_mul_i32 s19, s1, 6 // (op 535) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 535) least significant
s_mul_hi_u32 s7, s0, 6 // (op 535) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 535) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 535) 

v_add_u32 v37, v37, s0 // (op 535) 



s_barrier  // (op 1151) 
v_mov_b32 v14, v13 // (op 195) 



v_mov_b32 v16, v14 // (op 195) reg expression

buffer_load_ubyte v23, v16, s[60:63], 0 offen offset: 0 // (op 195) Load value


s_lshl_b32 s7, s30, 2 // (op 599) 
v_add_u32 v13, v13, s7 // (op 599) 

s_waitcnt vmcnt(2)
ds_write_b96 v15, v[24:26] offset:4096 // (op 191) Store local data 


v_mov_b32 v14, v12 // (op 196) 
v_mov_b32 v16, v14 // (op 196) reg expression

buffer_load_ubyte v27, v16, s[64:67], 0 offen offset: 0 // (op 196) Load value


s_lshl_b32 s7, s36, 2 // (op 663) 
v_add_u32 v12, v12, s7 // (op 663) 

s_waitcnt vmcnt(2)
ds_write_b96 v15, v[28:30] offset:4112 // (op 191) Store local data 

s_barrier  // (op 1141) 



s_barrier  // (op 1165) 
ds_read_b64_tr_b8 v[28:29], v0 // (op 183) Transpose load local data 



ds_read_b64_tr_b8 v[30:31], v0 offset:128 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:1024 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:1152 // (op 183) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v17, v23 offset:6208 // (op 192) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 1155) 



s_barrier  // (op 1179) 
ds_read_b96_tr_b6 v[38:40], v5 offset:4096 // (op 185) Transpose load local data 



ds_read_b96_tr_b6 v[41:43], v5 offset:4352 // (op 185) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v22, v27 offset:6144 // (op 193) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1169) 


ds_read_u8 v14, v4 offset:6208 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1183) 


ds_read_u8 v16, v7 offset:6144 // (op 189) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[38:43], a[0:3], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 190) 




s_ashr_i64 s[0:1], s[24:25], 7 // (op 175) 
s_mov_b32 s7, s0 // (op 175) convert

s_cmp_lt_u32 s6, s7 // (op 175) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_175 // (op 175) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_175 if true)
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_175:























s_barrier  // (op 1257) 








v_cmp_eq_f32 vcc, 0.00000, v6 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_135
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v10, v12 // (op 50) 
v_mul_f32 v17, v10, v13 // (op 50) 
v_mul_f32 v18, v10, v14 // (op 50) 
v_mul_f32 v19, v10, v15 // (op 50) 

v_lshrrev_b32 v12, 16, v16 // (op 56) 
v_lshrrev_b32 v13, 16, v17 // (op 56) 
v_lshrrev_b32 v14, 16, v18 // (op 56) 
v_lshrrev_b32 v15, 16, v19 // (op 56) 

s_mul_hi_i32 s0, s16, s9 // (op 964) 
v_lshrrev_b32 v0, 6, v1 // (op 964) 
s_add_i32 s1, s0, s16 // (op 964) 

v_and_b32 v11, 63, v1 // (op 964) 
s_ashr_i32 s0, s1, 31 // (op 964) 
v_lshrrev_b32 v3, 4, v11 // (op 964) 
s_cmp_eq_i32 s9, 0 // (op 964) 
s_mov_b32 s2, scc // (op 964) 
v_lshlrev_b32 v4, 2, v3 // (op 964) 

s_cmp_eq_i32 s2, 0 // (op 964) 
s_cselect_b32 s3, 0, -1 // (op 964) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 964) 


s_and_b32 s2, s10, 31 // (op 964) 
v_lshrrev_b32 v0, 2, v11 // (op 964) 

s_lshl_b32 s6, 1, s2 // (op 964) 
v_and_b32 v11, 3, v0 // (op 964) 

s_add_i32 s7, s6, s3 // (op 964) 


v_and_b32 v0, 3, v1 // (op 964) 
s_and_b32 s3, s0, s7 // (op 964) 


v_lshl_add_u32 v4, v11, 2, v0 // (op 964) 


s_add_i32 s0, s1, s3 // (op 964) 


s_ashr_i32 s1, s0, s2 // (op 964) 


s_lshl_b32 s0, s10, 24 // (op 964) 
s_ashr_i32 s2, s0, 31 // (op 964) 

s_xor_b32 s0, s1, s2 // (op 964) 

s_sub_i32 s1, s0, s2 // (op 964) 


s_mul_hi_i32 s0, s1, s11 // (op 964) 
s_add_i32 s2, s0, s1 // (op 964) 

s_ashr_i32 s0, s2, 31 // (op 964) 
s_cmp_eq_i32 s11, 0 // (op 964) 
s_mov_b32 s3, scc // (op 964) 
s_cmp_eq_i32 s3, 0 // (op 964) 
s_cselect_b32 s6, 0, -1 // (op 964) 

s_and_b32 s3, s12, 31 // (op 964) 
s_lshl_b32 s7, 1, s3 // (op 964) 
s_add_i32 s18, s7, s6 // (op 964) 


s_and_b32 s19, s0, s18 // (op 964) 


s_add_i32 s0, s2, s19 // (op 964) 


s_ashr_i32 s2, s0, s3 // (op 964) 


s_lshl_b32 s0, s12, 24 // (op 964) 
s_ashr_i32 s3, s0, 31 // (op 964) 

s_xor_b32 s0, s2, s3 // (op 964) 

s_sub_i32 s2, s0, s3 // (op 964) 


s_mul_i32 s0, s1, s14 // (op 964) 
s_sub_u32 s3, s16, s0 // (op 964) 

s_mul_i32 s0, s2, s14 // (op 964) 
s_add_u32 s0, s0, s3 // (op 964) 

s_mul_hi_i32 s3, s16, s8 // (op 964) 
s_add_i32 s6, s3, s16 // (op 964) 

s_ashr_i32 s3, s6, 31 // (op 964) 
s_cmp_eq_i32 s8, 0 // (op 964) 
s_mov_b32 s7, scc // (op 964) 
s_cmp_eq_i32 s7, 0 // (op 964) 
s_cselect_b32 s18, 0, -1 // (op 964) 

s_and_b32 s7, s15, 31 // (op 964) 
s_lshl_b32 s19, 1, s7 // (op 964) 
s_add_i32 s22, s19, s18 // (op 964) 


s_and_b32 s18, s3, s22 // (op 964) 


s_add_i32 s3, s6, s18 // (op 964) 


s_ashr_i32 s6, s3, s7 // (op 964) 


s_lshl_b32 s3, s15, 24 // (op 964) 
s_ashr_i32 s7, s3, 31 // (op 964) 

s_xor_b32 s3, s6, s7 // (op 964) 

s_sub_i32 s6, s3, s7 // (op 964) 


s_add_u32 s18, s4, 16 // (op 964) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 964) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 964) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 964) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 4 // (op 964) 

s_mul_i32 s3, s14, s18 // (op 964) 
s_mul_i32 s7, s50, s3 // (op 964) 

s_mul_i32 s3, s6, s7 // (op 964) 

s_sub_u32 s7, s16, s3 // (op 964) 

s_mov_b32 s3, s7 // (op 964) 
s_mul_hi_i32 s22, s3, s13 // (op 964) 
s_add_i32 s23, s22, s3 // (op 964) 


s_ashr_i32 s3, s23, 31 // (op 964) 
s_cmp_eq_i32 s13, 0 // (op 964) 
s_mov_b32 s22, scc // (op 964) 
s_cmp_eq_i32 s22, 0 // (op 964) 
s_cselect_b32 s24, 0, -1 // (op 964) 

s_and_b32 s22, s49, 31 // (op 964) 
s_lshl_b32 s25, 1, s22 // (op 964) 
s_add_i32 s26, s25, s24 // (op 964) 


s_and_b32 s24, s3, s26 // (op 964) 


s_add_i32 s3, s23, s24 // (op 964) 


s_ashr_i32 s23, s3, s22 // (op 964) 


s_lshl_b32 s3, s49, 24 // (op 964) 
s_ashr_i32 s22, s3, 31 // (op 964) 

s_xor_b32 s3, s23, s22 // (op 964) 

s_sub_i32 s23, s3, s22 // (op 964) 


s_mul_i32 s3, s48, s14 // (op 964) 
s_add_u32 s24, s20, 16 // (op 964) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 964) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 964) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 964) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 4 // (op 964) 

s_sub_i32 s22, s24, s3 // (op 964) 

s_mul_i32 s24, s23, s22 // (op 964) 

s_sub_u32 s22, s7, s24 // (op 964) 


s_add_u32 s7, s22, s3 // (op 964) 


s_cmp_eq_i32 s6, 0 // (op 964) 
s_mov_b32 s3, scc // (op 964) 

s_cmp_eq_i32 s3, 0 // (op 964) 
s_cselect_b32 s6, s7, s0 // (op 964) 


s_mul_i32 s0, s2, s18 // (op 964) 


v_lshl_add_u32 v0, s6, 4, v3 // (op 964) 


s_sub_i32 s2, s1, s0 // (op 964) 


s_cmp_eq_i32 s3, 0 // (op 964) 
s_cselect_b32 s22, s23, s2 // (op 964) 



v_lshl_add_u32 v3, s22, 4, v4 // (op 964) 


v_mul_lo_u32 v11, v3, s40 // (op 964) 

v_add_lshl_u32 v3, v0, v11, 1 // (op 964) 


s_mul_i32 s7, s40, s5 // (op 966) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 966) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 966) least significant
s_mul_hi_u32 s1, s40, s4 // (op 966) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 966) most significant: sum
s_add_u32 s7, s7, s1 // (op 966) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 966) 

s_mov_b32 s0, s42 // (op 966) reg expression
s_mov_b32 s1, s43 // (op 966) reg expression
s_mov_b32 s2, s18 // (op 966) reg expression
s_mov_b32 s3, 131072 // (op 966) call()


v_pack_b32_f16 v4, v12, v13 // (op 58) 
v_pack_b32_f16 v5, v14, v15 // (op 58) 


buffer_store_dwordx2 v[4:5], v3, s[0:3], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_135
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_135:
s_mul_hi_i32 s6, s16, s9 // (op 734) 
v_lshrrev_b32 v0, 6, v1 // (op 734) 
s_add_i32 s7, s6, s16 // (op 734) 

v_and_b32 v3, 63, v1 // (op 734) 
s_ashr_i32 s6, s7, 31 // (op 734) 
v_lshrrev_b32 v4, 4, v3 // (op 734) 
s_cmp_eq_i32 s9, 0 // (op 734) 
s_mov_b32 s18, scc // (op 734) 
v_lshlrev_b32 v5, 2, v4 // (op 734) 

s_cmp_eq_i32 s18, 0 // (op 734) 
s_cselect_b32 s19, 0, -1 // (op 734) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 734) 


s_and_b32 s18, s10, 31 // (op 734) 
v_lshrrev_b32 v0, 2, v3 // (op 734) 

s_lshl_b32 s22, 1, s18 // (op 734) 
v_and_b32 v3, 3, v0 // (op 734) 

s_add_i32 s23, s22, s19 // (op 734) 


v_and_b32 v0, 3, v1 // (op 734) 
s_and_b32 s19, s6, s23 // (op 734) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 734) 


s_add_i32 s6, s7, s19 // (op 734) 


s_ashr_i32 s7, s6, s18 // (op 734) 


s_lshl_b32 s6, s10, 24 // (op 734) 
s_ashr_i32 s18, s6, 31 // (op 734) 

s_xor_b32 s6, s7, s18 // (op 734) 

s_sub_i32 s7, s6, s18 // (op 734) 


s_mul_hi_i32 s6, s7, s11 // (op 734) 
s_add_i32 s18, s6, s7 // (op 734) 

s_ashr_i32 s6, s18, 31 // (op 734) 
s_cmp_eq_i32 s11, 0 // (op 734) 
s_mov_b32 s19, scc // (op 734) 
s_cmp_eq_i32 s19, 0 // (op 734) 
s_cselect_b32 s22, 0, -1 // (op 734) 

s_and_b32 s19, s12, 31 // (op 734) 
s_lshl_b32 s23, 1, s19 // (op 734) 
s_add_i32 s24, s23, s22 // (op 734) 


s_and_b32 s22, s6, s24 // (op 734) 


s_add_i32 s6, s18, s22 // (op 734) 


s_ashr_i32 s18, s6, s19 // (op 734) 


s_lshl_b32 s6, s12, 24 // (op 734) 
s_ashr_i32 s19, s6, 31 // (op 734) 

s_xor_b32 s6, s18, s19 // (op 734) 

s_sub_i32 s18, s6, s19 // (op 734) 


s_mul_i32 s6, s7, s14 // (op 734) 
s_sub_u32 s19, s16, s6 // (op 734) 

s_mul_i32 s6, s18, s14 // (op 734) 
s_add_u32 s6, s6, s19 // (op 734) 

s_mul_hi_i32 s19, s16, s8 // (op 734) 
s_add_i32 s22, s19, s16 // (op 734) 

s_ashr_i32 s19, s22, 31 // (op 734) 
s_cmp_eq_i32 s8, 0 // (op 734) 
s_mov_b32 s23, scc // (op 734) 
s_cmp_eq_i32 s23, 0 // (op 734) 
s_cselect_b32 s24, 0, -1 // (op 734) 

s_and_b32 s23, s15, 31 // (op 734) 
s_lshl_b32 s25, 1, s23 // (op 734) 
s_add_i32 s26, s25, s24 // (op 734) 


s_and_b32 s24, s19, s26 // (op 734) 


s_add_i32 s19, s22, s24 // (op 734) 


s_ashr_i32 s22, s19, s23 // (op 734) 


s_lshl_b32 s19, s15, 24 // (op 734) 
s_ashr_i32 s23, s19, 31 // (op 734) 

s_xor_b32 s19, s22, s23 // (op 734) 

s_sub_i32 s22, s19, s23 // (op 734) 


s_add_u32 s38, s4, 16 // (op 734) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 734) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 734) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 734) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 734) 

s_mul_i32 s19, s14, s38 // (op 734) 
s_mul_i32 s23, s50, s19 // (op 734) 

s_mul_i32 s19, s22, s23 // (op 734) 

s_sub_u32 s23, s16, s19 // (op 734) 

s_mov_b32 s19, s23 // (op 734) 
s_mul_hi_i32 s24, s19, s13 // (op 734) 
s_add_i32 s25, s24, s19 // (op 734) 


s_ashr_i32 s19, s25, 31 // (op 734) 
s_cmp_eq_i32 s13, 0 // (op 734) 
s_mov_b32 s24, scc // (op 734) 
s_cmp_eq_i32 s24, 0 // (op 734) 
s_cselect_b32 s26, 0, -1 // (op 734) 

s_and_b32 s24, s49, 31 // (op 734) 
s_lshl_b32 s27, 1, s24 // (op 734) 
s_add_i32 s28, s27, s26 // (op 734) 


s_and_b32 s26, s19, s28 // (op 734) 


s_add_i32 s19, s25, s26 // (op 734) 


s_ashr_i32 s25, s19, s24 // (op 734) 


s_lshl_b32 s19, s49, 24 // (op 734) 
s_ashr_i32 s24, s19, 31 // (op 734) 

s_xor_b32 s19, s25, s24 // (op 734) 

s_sub_i32 s25, s19, s24 // (op 734) 


s_mul_i32 s19, s48, s14 // (op 734) 
s_add_u32 s26, s20, 16 // (op 734) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 734) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 734) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 734) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 734) 

s_sub_i32 s24, s26, s19 // (op 734) 

s_mul_i32 s26, s25, s24 // (op 734) 

s_sub_u32 s24, s23, s26 // (op 734) 


s_add_u32 s23, s24, s19 // (op 734) 


s_cmp_eq_i32 s22, 0 // (op 734) 
s_mov_b32 s19, scc // (op 734) 

s_cmp_eq_i32 s19, 0 // (op 734) 
s_cselect_b32 s22, s23, s6 // (op 734) 


s_mul_i32 s6, s18, s38 // (op 734) 


v_lshl_add_u32 v0, s22, 4, v4 // (op 734) 


s_sub_i32 s18, s7, s6 // (op 734) 


s_cmp_eq_i32 s19, 0 // (op 734) 
s_cselect_b32 s6, s25, s18 // (op 734) 



v_lshl_add_u32 v3, s6, 4, v5 // (op 734) 


v_mul_lo_u32 v4, v3, s44 // (op 734) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 734) 


s_mul_i32 s7, s44, s5 // (op 736) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 736) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 736) least significant
s_mul_hi_u32 s19, s44, s4 // (op 736) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 736) most significant: sum
s_add_u32 s7, s7, s19 // (op 736) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 736) 

s_mov_b32 s36, s46 // (op 736) reg expression
s_mov_b32 s37, s47 // (op 736) reg expression
s_mov_b32 s38, s18 // (op 736) reg expression
s_mov_b32 s39, 131072 // (op 736) call()





buffer_load_dwordx2 v[4:5], v3, s[36:39], 0 offen // (op 148) Load value



v_accvgpr_read v12, a0 // (op 144) 
v_accvgpr_read v13, a1 // (op 144) 
v_accvgpr_read v14, a2 // (op 144) 
v_accvgpr_read v15, a3 // (op 144) 
v_mul_f32 v16, v10, v12 // (op 144) 
v_mul_f32 v17, v10, v13 // (op 144) 
v_mul_f32 v18, v10, v14 // (op 144) 
v_mul_f32 v19, v10, v15 // (op 144) 

s_waitcnt vmcnt(0)
v_and_b32 v8, 65535, v4 // (op 149) 
v_lshrrev_b32 v9, 16, v4 // (op 149) 
v_lshlrev_b32 v8, 16, v8 // (op 149) 
v_lshlrev_b32 v9, 16, v9 // (op 149) 
v_mul_f32 v12, v6, v8 // (op 149) 
v_mul_f32 v13, v6, v9 // (op 149) 
v_and_b32 v8, 65535, v5 // (op 149) 
v_lshrrev_b32 v9, 16, v5 // (op 149) 
v_lshlrev_b32 v8, 16, v8 // (op 149) 
v_lshlrev_b32 v9, 16, v9 // (op 149) 
v_mul_f32 v14, v6, v8 // (op 149) 
v_mul_f32 v15, v6, v9 // (op 149) 


v_add_f32 v20, v12, v16 // (op 145) 
v_add_f32 v21, v13, v17 // (op 145) 
v_add_f32 v22, v14, v18 // (op 145) 
v_add_f32 v23, v15, v19 // (op 145) 


v_lshrrev_b32 v16, 16, v20 // (op 146) 
v_lshrrev_b32 v17, 16, v21 // (op 146) 
v_lshrrev_b32 v18, 16, v22 // (op 146) 
v_lshrrev_b32 v19, 16, v23 // (op 146) 

s_mul_hi_i32 s6, s16, s9 // (op 981) 
v_lshrrev_b32 v0, 6, v1 // (op 981) 
s_add_i32 s7, s6, s16 // (op 981) 

v_and_b32 v3, 63, v1 // (op 981) 
s_ashr_i32 s6, s7, 31 // (op 981) 
v_lshrrev_b32 v4, 4, v3 // (op 981) 
s_cmp_eq_i32 s9, 0 // (op 981) 
s_mov_b32 s18, scc // (op 981) 
v_lshlrev_b32 v5, 2, v4 // (op 981) 

s_cmp_eq_i32 s18, 0 // (op 981) 
s_cselect_b32 s19, 0, -1 // (op 981) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 981) 


s_and_b32 s18, s10, 31 // (op 981) 
v_lshrrev_b32 v0, 2, v3 // (op 981) 

s_lshl_b32 s22, 1, s18 // (op 981) 
v_and_b32 v3, 3, v0 // (op 981) 

s_add_i32 s23, s22, s19 // (op 981) 


v_and_b32 v0, 3, v1 // (op 981) 
s_and_b32 s19, s6, s23 // (op 981) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 981) 


s_add_i32 s6, s7, s19 // (op 981) 


s_ashr_i32 s7, s6, s18 // (op 981) 


s_lshl_b32 s6, s10, 24 // (op 981) 
s_ashr_i32 s18, s6, 31 // (op 981) 

s_xor_b32 s6, s7, s18 // (op 981) 

s_sub_i32 s7, s6, s18 // (op 981) 


s_mul_hi_i32 s6, s7, s11 // (op 981) 
s_add_i32 s18, s6, s7 // (op 981) 

s_ashr_i32 s6, s18, 31 // (op 981) 
s_cmp_eq_i32 s11, 0 // (op 981) 
s_mov_b32 s19, scc // (op 981) 
s_cmp_eq_i32 s19, 0 // (op 981) 
s_cselect_b32 s22, 0, -1 // (op 981) 

s_and_b32 s19, s12, 31 // (op 981) 
s_lshl_b32 s23, 1, s19 // (op 981) 
s_add_i32 s24, s23, s22 // (op 981) 


s_and_b32 s22, s6, s24 // (op 981) 


s_add_i32 s6, s18, s22 // (op 981) 


s_ashr_i32 s18, s6, s19 // (op 981) 


s_lshl_b32 s6, s12, 24 // (op 981) 
s_ashr_i32 s19, s6, 31 // (op 981) 

s_xor_b32 s6, s18, s19 // (op 981) 

s_sub_i32 s18, s6, s19 // (op 981) 


s_mul_i32 s6, s7, s14 // (op 981) 
s_sub_u32 s19, s16, s6 // (op 981) 

s_mul_i32 s6, s18, s14 // (op 981) 
s_add_u32 s6, s6, s19 // (op 981) 

s_mul_hi_i32 s19, s16, s8 // (op 981) 
s_add_i32 s22, s19, s16 // (op 981) 

s_ashr_i32 s19, s22, 31 // (op 981) 
s_cmp_eq_i32 s8, 0 // (op 981) 
s_mov_b32 s23, scc // (op 981) 
s_cmp_eq_i32 s23, 0 // (op 981) 
s_cselect_b32 s24, 0, -1 // (op 981) 

s_and_b32 s23, s15, 31 // (op 981) 
s_lshl_b32 s25, 1, s23 // (op 981) 
s_add_i32 s26, s25, s24 // (op 981) 


s_and_b32 s24, s19, s26 // (op 981) 


s_add_i32 s19, s22, s24 // (op 981) 


s_ashr_i32 s22, s19, s23 // (op 981) 


s_lshl_b32 s19, s15, 24 // (op 981) 
s_ashr_i32 s23, s19, 31 // (op 981) 

s_xor_b32 s19, s22, s23 // (op 981) 

s_sub_i32 s22, s19, s23 // (op 981) 


s_add_u32 s38, s4, 16 // (op 981) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 981) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 981) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 981) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 981) 

s_mul_i32 s19, s14, s38 // (op 981) 
s_mul_i32 s23, s50, s19 // (op 981) 

s_mul_i32 s19, s22, s23 // (op 981) 

s_sub_u32 s23, s16, s19 // (op 981) 

s_mov_b32 s19, s23 // (op 981) 
s_mul_hi_i32 s24, s19, s13 // (op 981) 
s_add_i32 s25, s24, s19 // (op 981) 


s_ashr_i32 s19, s25, 31 // (op 981) 
s_cmp_eq_i32 s13, 0 // (op 981) 
s_mov_b32 s24, scc // (op 981) 
s_cmp_eq_i32 s24, 0 // (op 981) 
s_cselect_b32 s26, 0, -1 // (op 981) 

s_and_b32 s24, s49, 31 // (op 981) 
s_lshl_b32 s27, 1, s24 // (op 981) 
s_add_i32 s28, s27, s26 // (op 981) 


s_and_b32 s26, s19, s28 // (op 981) 


s_add_i32 s19, s25, s26 // (op 981) 


s_ashr_i32 s25, s19, s24 // (op 981) 


s_lshl_b32 s19, s49, 24 // (op 981) 
s_ashr_i32 s24, s19, 31 // (op 981) 

s_xor_b32 s19, s25, s24 // (op 981) 

s_sub_i32 s25, s19, s24 // (op 981) 


s_mul_i32 s19, s48, s14 // (op 981) 
s_add_u32 s26, s20, 16 // (op 981) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 981) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 981) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 981) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 981) 

s_sub_i32 s24, s26, s19 // (op 981) 

s_mul_i32 s26, s25, s24 // (op 981) 

s_sub_u32 s24, s23, s26 // (op 981) 


s_add_u32 s23, s24, s19 // (op 981) 


s_cmp_eq_i32 s22, 0 // (op 981) 
s_mov_b32 s19, scc // (op 981) 

s_cmp_eq_i32 s19, 0 // (op 981) 
s_cselect_b32 s22, s23, s6 // (op 981) 


s_mul_i32 s6, s18, s38 // (op 981) 


v_lshl_add_u32 v0, s22, 4, v4 // (op 981) 


s_sub_i32 s18, s7, s6 // (op 981) 


s_cmp_eq_i32 s19, 0 // (op 981) 
s_cselect_b32 s6, s25, s18 // (op 981) 



v_lshl_add_u32 v3, s6, 4, v5 // (op 981) 


v_mul_lo_u32 v4, v3, s40 // (op 981) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 981) 


s_mul_i32 s7, s40, s5 // (op 983) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 983) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 983) least significant
s_mul_hi_u32 s19, s40, s4 // (op 983) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 983) most significant: sum
s_add_u32 s7, s7, s19 // (op 983) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 983) 

s_mov_b32 s0, s42 // (op 983) reg expression
s_mov_b32 s1, s43 // (op 983) reg expression
s_mov_b32 s2, s18 // (op 983) reg expression
s_mov_b32 s3, 131072 // (op 983) call()


v_pack_b32_f16 v4, v16, v17 // (op 147) 
v_pack_b32_f16 v5, v18, v19 // (op 147) 


buffer_store_dwordx2 v[4:5], v3, s[0:3], 0 offen // (op 147) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_135:
















s_endpgm  // End of RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.LRR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end:
.size RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_, .LRR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end-RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 68
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 8448 // lds bytes
  .amdhsa_accum_offset 64
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
    DataTypeB: F6
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
    TransposeB: True
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
  - .name: RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 8448
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 64
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_2_size_1_11
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
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 80
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_5
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_7_stride_0_8
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
      - .name: Tensor_10_stride_1_10
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

