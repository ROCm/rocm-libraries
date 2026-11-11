// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 769) 
v_and_b32 v3, 15, v1 // (op 769) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 769) 


v_and_b32 v0, 48, v1 // (op 769) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 769) 


v_lshrrev_b32 v0, 6, v1 // (op 822) 
v_and_b32 v4, 1, v0 // (op 822) 

v_and_b32 v0, 15, v1 // (op 822) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 822) 


v_and_b32 v0, 63, v1 // (op 822) 
v_lshrrev_b32 v4, 4, v0 // (op 822) 

v_lshlrev_b32 v0, 5, v4 // (op 822) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 822) 


v_mul_lo_u32 v0, v4, 6 // (op 822) 

v_lshrrev_b32 v4, 3, v0 // (op 822) 

v_lshrrev_b32 v0, 6, v1 // (op 875) 
v_and_b32 v5, 1, v0 // (op 875) 

v_and_b32 v0, 15, v1 // (op 875) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 875) 


v_and_b32 v0, 63, v1 // (op 875) 
v_lshrrev_b32 v5, 4, v0 // (op 875) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 875) 


v_lshrrev_b32 v5, 7, v1 // (op 948) 
v_and_b32 v6, 15, v1 // (op 948) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 948) 


v_and_b32 v5, 48, v1 // (op 948) 
v_lshl_add_u32 v6, v7, 7, v5 // (op 948) 


v_lshrrev_b32 v5, 6, v1 // (op 972) 
v_and_b32 v7, 1, v5 // (op 972) 

v_and_b32 v5, 15, v1 // (op 972) 
v_lshl_add_u32 v8, v7, 4, v5 // (op 972) 


v_and_b32 v5, 63, v1 // (op 972) 
v_lshrrev_b32 v7, 4, v5 // (op 972) 

v_lshlrev_b32 v5, 5, v7 // (op 972) 

v_lshl_add_u32 v7, v8, 7, v5 // (op 972) 


v_mul_lo_u32 v5, v7, 6 // (op 972) 

v_lshrrev_b32 v7, 3, v5 // (op 972) 

v_lshrrev_b32 v5, 6, v1 // (op 996) 
v_and_b32 v8, 1, v5 // (op 996) 

v_and_b32 v5, 15, v1 // (op 996) 
v_lshl_add_u32 v9, v8, 4, v5 // (op 996) 


v_and_b32 v5, 63, v1 // (op 996) 
v_lshrrev_b32 v8, 4, v5 // (op 996) 

v_lshl_add_u32 v5, v9, 2, v8 // (op 996) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v9, s9 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
s_mul_hi_i32 s0, s16, s12 // (op 480) 
v_lshrrev_b32 v10, 3, v1 // (op 480) 
s_add_i32 s1, s0, s16 // (op 480) 

v_and_b32 v11, -8, v1 // (op 480) 
s_ashr_i32 s0, s1, 31 // (op 480) 
v_sub_u32 v12, v1, v11 // (op 480) 

s_cmp_eq_i32 s12, 0 // (op 480) 
s_mov_b32 s8, scc // (op 480) 
v_lshlrev_b32 v11, 4, v12 // (op 480) 

s_cmp_eq_i32 s8, 0 // (op 480) 
s_cselect_b32 s9, 0, -1 // (op 480) 

v_mov_b32 v13, 0 // (op 480) convert
v_mov_b32 v12, v11 // (op 480) convert

s_and_b32 s8, s13, 31 // (op 480) 
s_lshl_b32 s18, 1, s8 // (op 480) 
s_add_i32 s19, s18, s9 // (op 480) 


s_and_b32 s9, s0, s19 // (op 480) 


s_add_i32 s0, s1, s9 // (op 480) 


s_ashr_i32 s1, s0, s8 // (op 480) 


s_lshl_b32 s0, s13, 24 // (op 480) 
s_ashr_i32 s8, s0, 31 // (op 480) 

s_xor_b32 s0, s1, s8 // (op 480) 

s_sub_i32 s1, s0, s8 // (op 480) 


s_mul_hi_i32 s0, s1, s14 // (op 480) 
s_add_i32 s8, s0, s1 // (op 480) 

s_ashr_i32 s0, s8, 31 // (op 480) 
s_cmp_eq_i32 s14, 0 // (op 480) 
s_mov_b32 s9, scc // (op 480) 
s_cmp_eq_i32 s9, 0 // (op 480) 
s_cselect_b32 s18, 0, -1 // (op 480) 

s_and_b32 s9, s15, 31 // (op 480) 
s_lshl_b32 s19, 1, s9 // (op 480) 
s_add_i32 s49, s19, s18 // (op 480) 


s_and_b32 s18, s0, s49 // (op 480) 


s_add_i32 s0, s8, s18 // (op 480) 


s_ashr_i32 s8, s0, s9 // (op 480) 


s_lshl_b32 s0, s15, 24 // (op 480) 
s_ashr_i32 s9, s0, 31 // (op 480) 

s_xor_b32 s0, s8, s9 // (op 480) 

s_sub_i32 s8, s0, s9 // (op 480) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s45 // (op 480) 

s_sub_u32 s1, s16, s0 // (op 480) 

s_mul_i32 s0, s8, s45 // (op 480) 
s_add_u32 s0, s0, s1 // (op 480) 


s_mul_hi_i32 s1, s16, s11 // (op 480) 
s_add_i32 s8, s1, s16 // (op 480) 

s_ashr_i32 s1, s8, 31 // (op 480) 
s_cmp_eq_i32 s11, 0 // (op 480) 
s_mov_b32 s9, scc // (op 480) 
s_cmp_eq_i32 s9, 0 // (op 480) 
s_cselect_b32 s18, 0, -1 // (op 480) 

s_and_b32 s9, s46, 31 // (op 480) 
s_lshl_b32 s19, 1, s9 // (op 480) 
s_add_i32 s49, s19, s18 // (op 480) 


s_and_b32 s18, s1, s49 // (op 480) 


s_add_i32 s1, s8, s18 // (op 480) 


s_ashr_i32 s8, s1, s9 // (op 480) 


s_lshl_b32 s1, s46, 24 // (op 480) 
s_ashr_i32 s9, s1, 31 // (op 480) 

s_xor_b32 s1, s8, s9 // (op 480) 

s_sub_i32 s8, s1, s9 // (op 480) 


s_add_u32 s18, s4, 32 // (op 480) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 480) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 480) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 480) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 480) 

s_mul_i32 s1, s45, s18 // (op 480) 

s_mul_i32 s9, s10, s1 // (op 480) 

s_mul_i32 s1, s8, s9 // (op 480) 

s_sub_u32 s9, s16, s1 // (op 480) 

s_mov_b32 s1, s9 // (op 480) 
s_mul_hi_i32 s18, s1, s44 // (op 480) 
s_add_i32 s19, s18, s1 // (op 480) 


s_ashr_i32 s1, s19, 31 // (op 480) 
s_cmp_eq_i32 s44, 0 // (op 480) 
s_mov_b32 s18, scc // (op 480) 
s_cmp_eq_i32 s18, 0 // (op 480) 
s_cselect_b32 s49, 0, -1 // (op 480) 

s_and_b32 s18, s48, 31 // (op 480) 
s_lshl_b32 s50, 1, s18 // (op 480) 
s_add_i32 s51, s50, s49 // (op 480) 


s_and_b32 s49, s1, s51 // (op 480) 


s_add_i32 s1, s19, s49 // (op 480) 


s_ashr_i32 s19, s1, s18 // (op 480) 


s_lshl_b32 s1, s48, 24 // (op 480) 
s_ashr_i32 s18, s1, 31 // (op 480) 

s_xor_b32 s1, s19, s18 // (op 480) 

s_sub_i32 s19, s1, s18 // (op 480) 


s_mul_i32 s1, s47, s45 // (op 480) 
s_add_u32 s50, s6, 16 // (op 480) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 480) End of Int64 add, unlocking SCC
s_sub_u32 s52, s50, 1 // (op 480) least significant half; sets scc
s_subb_u32 s53, s51, 0 // (op 480) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 4 // (op 480) 

s_sub_i32 s18, s50, s1 // (op 480) 

s_mul_i32 s49, s19, s18 // (op 480) 


s_sub_u32 s18, s9, s49 // (op 480) 


s_add_u32 s9, s18, s1 // (op 480) 


s_cmp_eq_i32 s8, 0 // (op 480) 
s_mov_b32 s1, scc // (op 480) 

s_cmp_eq_i32 s1, 0 // (op 480) 
s_cselect_b32 s8, s9, s0 // (op 480) 



v_lshl_add_u32 v11, s8, 4, v10 // (op 480) 


v_mul_lo_u32 v15, v11, s3 // (op 480) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 480) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 480) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 480) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 480) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 480) most significant half



s_mul_i32 s1, s2, s7 // (op 482) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 482) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 482) least significant
s_mul_hi_u32 s9, s2, s6 // (op 482) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 482) most significant: sum
s_add_u32 s1, s1, s9 // (op 482) most significant: sum


s_mov_b32 s52, s20 // (op 482) reg expression
s_mov_b32 s53, s21 // (op 482) reg expression
s_mov_b32 s54, s0 // (op 482) reg expression
s_mov_b32 s55, 131072 // (op 482) call()

s_mul_hi_i32 s0, s16, s12 // (op 499) 
v_lshrrev_b32 v10, 3, v1 // (op 499) 
s_add_i32 s1, s0, s16 // (op 499) 

v_and_b32 v11, -8, v1 // (op 499) 
s_ashr_i32 s0, s1, 31 // (op 499) 
v_sub_u32 v12, v1, v11 // (op 499) 

s_cmp_eq_i32 s12, 0 // (op 499) 
s_mov_b32 s8, scc // (op 499) 
v_lshlrev_b32 v11, 4, v12 // (op 499) 

s_cmp_eq_i32 s8, 0 // (op 499) 
s_cselect_b32 s9, 0, -1 // (op 499) 

v_mov_b32 v13, 0 // (op 499) convert
v_mov_b32 v12, v11 // (op 499) convert

s_and_b32 s8, s13, 31 // (op 499) 
s_lshl_b32 s18, 1, s8 // (op 499) 
s_add_i32 s19, s18, s9 // (op 499) 


s_and_b32 s9, s0, s19 // (op 499) 


s_add_i32 s0, s1, s9 // (op 499) 


s_ashr_i32 s1, s0, s8 // (op 499) 


s_lshl_b32 s0, s13, 24 // (op 499) 
s_ashr_i32 s8, s0, 31 // (op 499) 

s_xor_b32 s0, s1, s8 // (op 499) 

s_sub_i32 s1, s0, s8 // (op 499) 


s_mul_hi_i32 s0, s1, s14 // (op 499) 
s_add_i32 s8, s0, s1 // (op 499) 

s_ashr_i32 s0, s8, 31 // (op 499) 
s_cmp_eq_i32 s14, 0 // (op 499) 
s_mov_b32 s9, scc // (op 499) 
s_cmp_eq_i32 s9, 0 // (op 499) 
s_cselect_b32 s18, 0, -1 // (op 499) 

s_and_b32 s9, s15, 31 // (op 499) 
s_lshl_b32 s19, 1, s9 // (op 499) 
s_add_i32 s49, s19, s18 // (op 499) 


s_and_b32 s18, s0, s49 // (op 499) 


s_add_i32 s0, s8, s18 // (op 499) 


s_ashr_i32 s8, s0, s9 // (op 499) 


s_lshl_b32 s0, s15, 24 // (op 499) 
s_ashr_i32 s9, s0, 31 // (op 499) 

s_xor_b32 s0, s8, s9 // (op 499) 

s_sub_i32 s8, s0, s9 // (op 499) 


s_mul_i32 s0, s1, s45 // (op 499) 

s_sub_u32 s1, s16, s0 // (op 499) 

s_mul_i32 s0, s8, s45 // (op 499) 
s_add_u32 s0, s0, s1 // (op 499) 


s_mul_hi_i32 s1, s16, s11 // (op 499) 
s_add_i32 s8, s1, s16 // (op 499) 

s_ashr_i32 s1, s8, 31 // (op 499) 
s_cmp_eq_i32 s11, 0 // (op 499) 
s_mov_b32 s9, scc // (op 499) 
s_cmp_eq_i32 s9, 0 // (op 499) 
s_cselect_b32 s18, 0, -1 // (op 499) 

s_and_b32 s9, s46, 31 // (op 499) 
s_lshl_b32 s19, 1, s9 // (op 499) 
s_add_i32 s49, s19, s18 // (op 499) 


s_and_b32 s18, s1, s49 // (op 499) 


s_add_i32 s1, s8, s18 // (op 499) 


s_ashr_i32 s8, s1, s9 // (op 499) 


s_lshl_b32 s1, s46, 24 // (op 499) 
s_ashr_i32 s9, s1, 31 // (op 499) 

s_xor_b32 s1, s8, s9 // (op 499) 

s_sub_i32 s8, s1, s9 // (op 499) 


s_add_u32 s18, s4, 32 // (op 499) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 499) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 499) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 499) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 499) 

s_mul_i32 s1, s45, s18 // (op 499) 

s_mul_i32 s9, s10, s1 // (op 499) 

s_mul_i32 s1, s8, s9 // (op 499) 

s_sub_u32 s9, s16, s1 // (op 499) 

s_mov_b32 s1, s9 // (op 499) 
s_mul_hi_i32 s18, s1, s44 // (op 499) 
s_add_i32 s19, s18, s1 // (op 499) 


s_ashr_i32 s1, s19, 31 // (op 499) 
s_cmp_eq_i32 s44, 0 // (op 499) 
s_mov_b32 s18, scc // (op 499) 
s_cmp_eq_i32 s18, 0 // (op 499) 
s_cselect_b32 s49, 0, -1 // (op 499) 

s_and_b32 s18, s48, 31 // (op 499) 
s_lshl_b32 s50, 1, s18 // (op 499) 
s_add_i32 s51, s50, s49 // (op 499) 


s_and_b32 s49, s1, s51 // (op 499) 


s_add_i32 s1, s19, s49 // (op 499) 


s_ashr_i32 s19, s1, s18 // (op 499) 


s_lshl_b32 s1, s48, 24 // (op 499) 
s_ashr_i32 s18, s1, 31 // (op 499) 

s_xor_b32 s1, s19, s18 // (op 499) 

s_sub_i32 s19, s1, s18 // (op 499) 


s_mul_i32 s1, s47, s45 // (op 499) 
s_add_u32 s50, s6, 16 // (op 499) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 499) End of Int64 add, unlocking SCC
s_sub_u32 s56, s50, 1 // (op 499) least significant half; sets scc
s_subb_u32 s57, s51, 0 // (op 499) most significant half; uses scc

s_ashr_i64 s[50:51], s[56:57], 4 // (op 499) 

s_sub_i32 s18, s50, s1 // (op 499) 

s_mul_i32 s49, s19, s18 // (op 499) 


s_sub_u32 s18, s9, s49 // (op 499) 


s_add_u32 s9, s18, s1 // (op 499) 


s_cmp_eq_i32 s8, 0 // (op 499) 
s_mov_b32 s1, scc // (op 499) 

s_cmp_eq_i32 s1, 0 // (op 499) 
s_cselect_b32 s8, s9, s0 // (op 499) 



v_lshl_add_u32 v11, s8, 4, v10 // (op 499) 


v_mul_lo_u32 v17, v11, s3 // (op 499) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 499) most significant: high of low * low
v_mul_lo_u32 v16, v11, s2 // (op 499) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 499) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 499) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 499) most significant half



s_mul_i32 s1, s2, s7 // (op 501) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 501) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 501) least significant
s_mul_hi_u32 s9, s2, s6 // (op 501) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 501) most significant: sum
s_add_u32 s1, s1, s9 // (op 501) most significant: sum


s_mov_b32 s52, s20 // (op 501) reg expression
s_mov_b32 s53, s21 // (op 501) reg expression
s_mov_b32 s54, s0 // (op 501) reg expression
s_mov_b32 s55, 131072 // (op 501) call()

s_mul_hi_i32 s0, s16, s12 // (op 544) 
v_lshrrev_b32 v10, 2, v1 // (op 544) 
s_add_i32 s1, s0, s16 // (op 544) 

v_and_b32 v11, -4, v1 // (op 544) 
s_ashr_i32 s0, s1, 31 // (op 544) 
v_sub_u32 v12, v1, v11 // (op 544) 

s_cmp_eq_i32 s12, 0 // (op 544) 
s_mov_b32 s8, scc // (op 544) 
v_lshlrev_b32 v11, 5, v12 // (op 544) 

s_cmp_eq_i32 s8, 0 // (op 544) 
s_cselect_b32 s9, 0, -1 // (op 544) 

v_mov_b32 v13, 0 // (op 544) convert
v_mov_b32 v12, v11 // (op 544) convert

s_and_b32 s8, s13, 31 // (op 544) 
s_lshl_b32 s18, 1, s8 // (op 544) 
s_add_i32 s19, s18, s9 // (op 544) 


s_and_b32 s9, s0, s19 // (op 544) 


s_add_i32 s0, s1, s9 // (op 544) 


s_ashr_i32 s1, s0, s8 // (op 544) 


s_lshl_b32 s0, s13, 24 // (op 544) 
s_ashr_i32 s8, s0, 31 // (op 544) 

s_xor_b32 s0, s1, s8 // (op 544) 

s_sub_i32 s1, s0, s8 // (op 544) 


s_mul_hi_i32 s0, s1, s14 // (op 544) 
s_add_i32 s8, s0, s1 // (op 544) 

s_ashr_i32 s0, s8, 31 // (op 544) 
s_cmp_eq_i32 s14, 0 // (op 544) 
s_mov_b32 s9, scc // (op 544) 
s_cmp_eq_i32 s9, 0 // (op 544) 
s_cselect_b32 s18, 0, -1 // (op 544) 

s_and_b32 s9, s15, 31 // (op 544) 
s_lshl_b32 s19, 1, s9 // (op 544) 
s_add_i32 s49, s19, s18 // (op 544) 


s_and_b32 s18, s0, s49 // (op 544) 


s_add_i32 s0, s8, s18 // (op 544) 


s_ashr_i32 s8, s0, s9 // (op 544) 


s_lshl_b32 s0, s15, 24 // (op 544) 
s_ashr_i32 s9, s0, 31 // (op 544) 

s_xor_b32 s0, s8, s9 // (op 544) 

s_sub_i32 s8, s0, s9 // (op 544) 


s_add_u32 s18, s4, 32 // (op 544) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 544) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 544) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 544) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 544) 

s_mul_i32 s0, s8, s18 // (op 544) 

s_sub_i32 s8, s1, s0 // (op 544) 


s_mul_hi_i32 s9, s16, s11 // (op 544) 
s_add_i32 s0, s9, s16 // (op 544) 

s_ashr_i32 s1, s0, 31 // (op 544) 
s_cmp_eq_i32 s11, 0 // (op 544) 
s_mov_b32 s9, scc // (op 544) 
s_cmp_eq_i32 s9, 0 // (op 544) 
s_cselect_b32 s49, 0, -1 // (op 544) 

s_and_b32 s9, s46, 31 // (op 544) 
s_lshl_b32 s50, 1, s9 // (op 544) 
s_add_i32 s51, s50, s49 // (op 544) 


s_and_b32 s49, s1, s51 // (op 544) 


s_add_i32 s1, s0, s49 // (op 544) 


s_ashr_i32 s0, s1, s9 // (op 544) 


s_lshl_b32 s1, s46, 24 // (op 544) 
s_ashr_i32 s9, s1, 31 // (op 544) 

s_xor_b32 s1, s0, s9 // (op 544) 

s_sub_i32 s0, s1, s9 // (op 544) 


s_mul_i32 s1, s45, s18 // (op 544) 

s_mul_i32 s9, s10, s1 // (op 544) 

s_mul_i32 s1, s0, s9 // (op 544) 

s_sub_u32 s9, s16, s1 // (op 544) 

s_mul_hi_i32 s1, s9, s44 // (op 544) 
s_add_i32 s18, s1, s9 // (op 544) 


s_ashr_i32 s1, s18, 31 // (op 544) 
s_cmp_eq_i32 s44, 0 // (op 544) 
s_mov_b32 s9, scc // (op 544) 
s_cmp_eq_i32 s9, 0 // (op 544) 
s_cselect_b32 s19, 0, -1 // (op 544) 

s_and_b32 s9, s48, 31 // (op 544) 
s_lshl_b32 s49, 1, s9 // (op 544) 
s_add_i32 s50, s49, s19 // (op 544) 


s_and_b32 s19, s1, s50 // (op 544) 


s_add_i32 s1, s18, s19 // (op 544) 


s_ashr_i32 s18, s1, s9 // (op 544) 


s_lshl_b32 s1, s48, 24 // (op 544) 
s_ashr_i32 s9, s1, 31 // (op 544) 

s_xor_b32 s1, s18, s9 // (op 544) 

s_sub_i32 s18, s1, s9 // (op 544) 


s_cmp_eq_i32 s0, 0 // (op 544) 
s_mov_b32 s1, scc // (op 544) 

s_cmp_eq_i32 s1, 0 // (op 544) 
s_cselect_b32 s0, s18, s8 // (op 544) 



v_lshl_add_u32 v11, s0, 5, v10 // (op 544) 


v_mul_lo_u32 v19, v11, s23 // (op 544) most significant: low of low * high
v_mul_hi_u32 v10, v11, s22 // (op 544) most significant: high of low * low
v_mul_lo_u32 v18, v11, s22 // (op 544) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 544) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v18 // (op 544) least significant half
v_addc_co_u32 v11, s[0:1], v13, v19, s[0:1] // (op 544) most significant half



v_mul_lo_u32 v13, v11, 6 // (op 544) most significant: low of high * low
v_mul_hi_u32 v18, v10, 6 // (op 544) most significant: high of low * low
v_mul_lo_u32 v12, v10, 6 // (op 544) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 544) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 544) 

v_mov_b32 v12, v10 // (op 544) convert

s_mul_i32 s1, s5, 6 // (op 546) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 546) least significant
s_mul_hi_u32 s8, s4, 6 // (op 546) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 546) most significant: sum

s_mul_i32 s9, s22, s1 // (op 546) most significant: low of low * high
s_mul_i32 s18, s23, s0 // (op 546) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 546) least significant
s_mul_hi_u32 s19, s22, s0 // (op 546) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 546) most significant: sum
s_add_u32 s9, s9, s19 // (op 546) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 546) 

s_mov_b32 s56, s24 // (op 546) reg expression
s_mov_b32 s57, s25 // (op 546) reg expression
s_mov_b32 s58, s0 // (op 546) reg expression
s_mov_b32 s59, 131072 // (op 546) call()

s_mul_hi_i32 s0, s16, s12 // (op 563) 
v_lshrrev_b32 v13, 2, v1 // (op 563) 
s_add_i32 s1, s0, s16 // (op 563) 

v_and_b32 v10, -4, v1 // (op 563) 
s_ashr_i32 s0, s1, 31 // (op 563) 
v_sub_u32 v11, v1, v10 // (op 563) 

s_cmp_eq_i32 s12, 0 // (op 563) 
s_mov_b32 s8, scc // (op 563) 
v_lshlrev_b32 v10, 5, v11 // (op 563) 

s_cmp_eq_i32 s8, 0 // (op 563) 
s_cselect_b32 s9, 0, -1 // (op 563) 

v_mov_b32 v19, 0 // (op 563) convert
v_mov_b32 v18, v10 // (op 563) convert

s_and_b32 s8, s13, 31 // (op 563) 
s_lshl_b32 s18, 1, s8 // (op 563) 
s_add_i32 s19, s18, s9 // (op 563) 


s_and_b32 s9, s0, s19 // (op 563) 


s_add_i32 s0, s1, s9 // (op 563) 


s_ashr_i32 s1, s0, s8 // (op 563) 


s_lshl_b32 s0, s13, 24 // (op 563) 
s_ashr_i32 s8, s0, 31 // (op 563) 

s_xor_b32 s0, s1, s8 // (op 563) 

s_sub_i32 s1, s0, s8 // (op 563) 


s_mul_hi_i32 s0, s1, s14 // (op 563) 
s_add_i32 s8, s0, s1 // (op 563) 

s_ashr_i32 s0, s8, 31 // (op 563) 
s_cmp_eq_i32 s14, 0 // (op 563) 
s_mov_b32 s9, scc // (op 563) 
s_cmp_eq_i32 s9, 0 // (op 563) 
s_cselect_b32 s18, 0, -1 // (op 563) 

s_and_b32 s9, s15, 31 // (op 563) 
s_lshl_b32 s19, 1, s9 // (op 563) 
s_add_i32 s49, s19, s18 // (op 563) 


s_and_b32 s18, s0, s49 // (op 563) 


s_add_i32 s0, s8, s18 // (op 563) 


s_ashr_i32 s8, s0, s9 // (op 563) 


s_lshl_b32 s0, s15, 24 // (op 563) 
s_ashr_i32 s9, s0, 31 // (op 563) 

s_xor_b32 s0, s8, s9 // (op 563) 

s_sub_i32 s8, s0, s9 // (op 563) 


s_add_u32 s18, s4, 32 // (op 563) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 563) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 563) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 563) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 563) 

s_mul_i32 s0, s8, s18 // (op 563) 

s_sub_i32 s8, s1, s0 // (op 563) 


s_mul_hi_i32 s9, s16, s11 // (op 563) 
s_add_i32 s0, s9, s16 // (op 563) 

s_ashr_i32 s1, s0, 31 // (op 563) 
s_cmp_eq_i32 s11, 0 // (op 563) 
s_mov_b32 s9, scc // (op 563) 
s_cmp_eq_i32 s9, 0 // (op 563) 
s_cselect_b32 s49, 0, -1 // (op 563) 

s_and_b32 s9, s46, 31 // (op 563) 
s_lshl_b32 s50, 1, s9 // (op 563) 
s_add_i32 s51, s50, s49 // (op 563) 


s_and_b32 s49, s1, s51 // (op 563) 


s_add_i32 s1, s0, s49 // (op 563) 


s_ashr_i32 s0, s1, s9 // (op 563) 


s_lshl_b32 s1, s46, 24 // (op 563) 
s_ashr_i32 s9, s1, 31 // (op 563) 

s_xor_b32 s1, s0, s9 // (op 563) 

s_sub_i32 s0, s1, s9 // (op 563) 


s_mul_i32 s1, s45, s18 // (op 563) 

s_mul_i32 s9, s10, s1 // (op 563) 

s_mul_i32 s1, s0, s9 // (op 563) 

s_sub_u32 s9, s16, s1 // (op 563) 

s_mul_hi_i32 s1, s9, s44 // (op 563) 
s_add_i32 s18, s1, s9 // (op 563) 


s_ashr_i32 s1, s18, 31 // (op 563) 
s_cmp_eq_i32 s44, 0 // (op 563) 
s_mov_b32 s9, scc // (op 563) 
s_cmp_eq_i32 s9, 0 // (op 563) 
s_cselect_b32 s19, 0, -1 // (op 563) 

s_and_b32 s9, s48, 31 // (op 563) 
s_lshl_b32 s49, 1, s9 // (op 563) 
s_add_i32 s50, s49, s19 // (op 563) 


s_and_b32 s19, s1, s50 // (op 563) 


s_add_i32 s1, s18, s19 // (op 563) 


s_ashr_i32 s18, s1, s9 // (op 563) 


s_lshl_b32 s1, s48, 24 // (op 563) 
s_ashr_i32 s9, s1, 31 // (op 563) 

s_xor_b32 s1, s18, s9 // (op 563) 

s_sub_i32 s18, s1, s9 // (op 563) 


s_cmp_eq_i32 s0, 0 // (op 563) 
s_mov_b32 s1, scc // (op 563) 

s_cmp_eq_i32 s1, 0 // (op 563) 
s_cselect_b32 s0, s18, s8 // (op 563) 



v_lshl_add_u32 v10, s0, 5, v13 // (op 563) 


v_mul_lo_u32 v21, v10, s23 // (op 563) most significant: low of low * high
v_mul_hi_u32 v11, v10, s22 // (op 563) most significant: high of low * low
v_mul_lo_u32 v20, v10, s22 // (op 563) least significant: low of low * low
v_add_u32 v21, v21, v11 // (op 563) most significant: sum


v_add_co_u32 v10, s[0:1], v18, v20 // (op 563) least significant half
v_addc_co_u32 v11, s[0:1], v19, v21, s[0:1] // (op 563) most significant half



v_mul_lo_u32 v19, v11, 6 // (op 563) most significant: low of high * low
v_mul_hi_u32 v13, v10, 6 // (op 563) most significant: high of low * low
v_mul_lo_u32 v18, v10, 6 // (op 563) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 563) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[18:19] // (op 563) 

v_mov_b32 v13, v10 // (op 563) convert

s_mul_i32 s1, s5, 6 // (op 565) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 565) least significant
s_mul_hi_u32 s8, s4, 6 // (op 565) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 565) most significant: sum

s_mul_i32 s9, s22, s1 // (op 565) most significant: low of low * high
s_mul_i32 s18, s23, s0 // (op 565) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 565) least significant
s_mul_hi_u32 s19, s22, s0 // (op 565) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 565) most significant: sum
s_add_u32 s9, s9, s19 // (op 565) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 565) 

s_mov_b32 s56, s24 // (op 565) reg expression
s_mov_b32 s57, s25 // (op 565) reg expression
s_mov_b32 s58, s0 // (op 565) reg expression
s_mov_b32 s59, 131072 // (op 565) call()

s_mul_hi_i32 s0, s16, s12 // (op 608) 
v_lshrrev_b32 v10, 7, v1 // (op 608) 
s_add_i32 s1, s0, s16 // (op 608) 

v_and_b32 v11, 15, v1 // (op 608) 
s_ashr_i32 s0, s1, 31 // (op 608) 
v_lshl_add_u32 v18, v10, 4, v11 // (op 608) 


s_cmp_eq_i32 s12, 0 // (op 608) 
s_mov_b32 s8, scc // (op 608) 
v_and_b32 v10, 63, v1 // (op 608) 
s_cmp_eq_i32 s8, 0 // (op 608) 
s_cselect_b32 s9, 0, -1 // (op 608) 

v_lshrrev_b32 v11, 4, v10 // (op 608) 

s_and_b32 s8, s13, 31 // (op 608) 
s_lshl_b32 s18, 1, s8 // (op 608) 
s_add_i32 s19, s18, s9 // (op 608) 


s_and_b32 s9, s0, s19 // (op 608) 


s_add_i32 s0, s1, s9 // (op 608) 


s_ashr_i32 s1, s0, s8 // (op 608) 


s_lshl_b32 s0, s13, 24 // (op 608) 
s_ashr_i32 s8, s0, 31 // (op 608) 

s_xor_b32 s0, s1, s8 // (op 608) 

s_sub_i32 s1, s0, s8 // (op 608) 


s_mul_hi_i32 s0, s1, s14 // (op 608) 
s_add_i32 s8, s0, s1 // (op 608) 

s_ashr_i32 s0, s8, 31 // (op 608) 
s_cmp_eq_i32 s14, 0 // (op 608) 
s_mov_b32 s9, scc // (op 608) 
s_cmp_eq_i32 s9, 0 // (op 608) 
s_cselect_b32 s18, 0, -1 // (op 608) 

s_and_b32 s9, s15, 31 // (op 608) 
s_lshl_b32 s19, 1, s9 // (op 608) 
s_add_i32 s49, s19, s18 // (op 608) 


s_and_b32 s18, s0, s49 // (op 608) 


s_add_i32 s0, s8, s18 // (op 608) 


s_ashr_i32 s8, s0, s9 // (op 608) 


s_lshl_b32 s0, s15, 24 // (op 608) 
s_ashr_i32 s9, s0, 31 // (op 608) 

s_xor_b32 s0, s8, s9 // (op 608) 

s_sub_i32 s8, s0, s9 // (op 608) 


s_mul_i32 s0, s1, s45 // (op 608) 

s_sub_u32 s1, s16, s0 // (op 608) 

s_mul_i32 s0, s8, s45 // (op 608) 
s_add_u32 s0, s0, s1 // (op 608) 


s_mul_hi_i32 s1, s16, s11 // (op 608) 
s_add_i32 s8, s1, s16 // (op 608) 

s_ashr_i32 s1, s8, 31 // (op 608) 
s_cmp_eq_i32 s11, 0 // (op 608) 
s_mov_b32 s9, scc // (op 608) 
s_cmp_eq_i32 s9, 0 // (op 608) 
s_cselect_b32 s18, 0, -1 // (op 608) 

s_and_b32 s9, s46, 31 // (op 608) 
s_lshl_b32 s19, 1, s9 // (op 608) 
s_add_i32 s49, s19, s18 // (op 608) 


s_and_b32 s18, s1, s49 // (op 608) 


s_add_i32 s1, s8, s18 // (op 608) 


s_ashr_i32 s8, s1, s9 // (op 608) 


s_lshl_b32 s1, s46, 24 // (op 608) 
s_ashr_i32 s9, s1, 31 // (op 608) 

s_xor_b32 s1, s8, s9 // (op 608) 

s_sub_i32 s8, s1, s9 // (op 608) 


s_add_u32 s18, s4, 32 // (op 608) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 608) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 608) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 608) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 608) 

s_mul_i32 s1, s45, s18 // (op 608) 

s_mul_i32 s9, s10, s1 // (op 608) 

s_mul_i32 s1, s8, s9 // (op 608) 

s_sub_u32 s9, s16, s1 // (op 608) 

s_mov_b32 s1, s9 // (op 608) 
s_mul_hi_i32 s18, s1, s44 // (op 608) 
s_add_i32 s19, s18, s1 // (op 608) 


s_ashr_i32 s1, s19, 31 // (op 608) 
s_cmp_eq_i32 s44, 0 // (op 608) 
s_mov_b32 s18, scc // (op 608) 
s_cmp_eq_i32 s18, 0 // (op 608) 
s_cselect_b32 s49, 0, -1 // (op 608) 

s_and_b32 s18, s48, 31 // (op 608) 
s_lshl_b32 s50, 1, s18 // (op 608) 
s_add_i32 s51, s50, s49 // (op 608) 


s_and_b32 s49, s1, s51 // (op 608) 


s_add_i32 s1, s19, s49 // (op 608) 


s_ashr_i32 s19, s1, s18 // (op 608) 


s_lshl_b32 s1, s48, 24 // (op 608) 
s_ashr_i32 s18, s1, 31 // (op 608) 

s_xor_b32 s1, s19, s18 // (op 608) 

s_sub_i32 s19, s1, s18 // (op 608) 


s_mul_i32 s1, s47, s45 // (op 608) 
s_add_u32 s50, s6, 16 // (op 608) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 608) End of Int64 add, unlocking SCC
s_sub_u32 s60, s50, 1 // (op 608) least significant half; sets scc
s_subb_u32 s61, s51, 0 // (op 608) most significant half; uses scc

s_ashr_i64 s[50:51], s[60:61], 4 // (op 608) 

s_sub_i32 s18, s50, s1 // (op 608) 

s_mul_i32 s49, s19, s18 // (op 608) 


s_sub_u32 s18, s9, s49 // (op 608) 


s_add_u32 s9, s18, s1 // (op 608) 


s_cmp_eq_i32 s8, 0 // (op 608) 
s_mov_b32 s1, scc // (op 608) 

s_cmp_eq_i32 s1, 0 // (op 608) 
s_cselect_b32 s8, s9, s0 // (op 608) 



v_lshl_add_u32 v10, s8, 4, v18 // (op 608) 


v_mul_lo_u32 v18, v10, s26 // (op 608) 
v_add_u32 v18, v18, v11 // (op 608) 


s_mul_i32 s1, s26, s7 // (op 610) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 610) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 610) least significant
s_mul_hi_u32 s9, s26, s6 // (op 610) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 610) most significant: sum
s_add_u32 s1, s1, s9 // (op 610) most significant: sum


s_mov_b32 s60, s28 // (op 610) reg expression
s_mov_b32 s61, s29 // (op 610) reg expression
s_mov_b32 s62, s0 // (op 610) reg expression
s_mov_b32 s63, 131072 // (op 610) call()

s_mul_hi_i32 s0, s16, s12 // (op 627) 
v_lshrrev_b32 v10, 7, v1 // (op 627) 
s_add_i32 s1, s0, s16 // (op 627) 

v_and_b32 v11, 15, v1 // (op 627) 
s_ashr_i32 s0, s1, 31 // (op 627) 
v_lshl_add_u32 v19, v10, 4, v11 // (op 627) 


s_cmp_eq_i32 s12, 0 // (op 627) 
s_mov_b32 s8, scc // (op 627) 
v_and_b32 v10, 63, v1 // (op 627) 
s_cmp_eq_i32 s8, 0 // (op 627) 
s_cselect_b32 s9, 0, -1 // (op 627) 

v_lshrrev_b32 v11, 4, v10 // (op 627) 

s_and_b32 s8, s13, 31 // (op 627) 
s_lshl_b32 s18, 1, s8 // (op 627) 
s_add_i32 s19, s18, s9 // (op 627) 


s_and_b32 s9, s0, s19 // (op 627) 


s_add_i32 s0, s1, s9 // (op 627) 


s_ashr_i32 s1, s0, s8 // (op 627) 


s_lshl_b32 s0, s13, 24 // (op 627) 
s_ashr_i32 s8, s0, 31 // (op 627) 

s_xor_b32 s0, s1, s8 // (op 627) 

s_sub_i32 s1, s0, s8 // (op 627) 


s_mul_hi_i32 s0, s1, s14 // (op 627) 
s_add_i32 s8, s0, s1 // (op 627) 

s_ashr_i32 s0, s8, 31 // (op 627) 
s_cmp_eq_i32 s14, 0 // (op 627) 
s_mov_b32 s9, scc // (op 627) 
s_cmp_eq_i32 s9, 0 // (op 627) 
s_cselect_b32 s18, 0, -1 // (op 627) 

s_and_b32 s9, s15, 31 // (op 627) 
s_lshl_b32 s19, 1, s9 // (op 627) 
s_add_i32 s49, s19, s18 // (op 627) 


s_and_b32 s18, s0, s49 // (op 627) 


s_add_i32 s0, s8, s18 // (op 627) 


s_ashr_i32 s8, s0, s9 // (op 627) 


s_lshl_b32 s0, s15, 24 // (op 627) 
s_ashr_i32 s9, s0, 31 // (op 627) 

s_xor_b32 s0, s8, s9 // (op 627) 

s_sub_i32 s8, s0, s9 // (op 627) 


s_mul_i32 s0, s1, s45 // (op 627) 

s_sub_u32 s1, s16, s0 // (op 627) 

s_mul_i32 s0, s8, s45 // (op 627) 
s_add_u32 s0, s0, s1 // (op 627) 


s_mul_hi_i32 s1, s16, s11 // (op 627) 
s_add_i32 s8, s1, s16 // (op 627) 

s_ashr_i32 s1, s8, 31 // (op 627) 
s_cmp_eq_i32 s11, 0 // (op 627) 
s_mov_b32 s9, scc // (op 627) 
s_cmp_eq_i32 s9, 0 // (op 627) 
s_cselect_b32 s18, 0, -1 // (op 627) 

s_and_b32 s9, s46, 31 // (op 627) 
s_lshl_b32 s19, 1, s9 // (op 627) 
s_add_i32 s49, s19, s18 // (op 627) 


s_and_b32 s18, s1, s49 // (op 627) 


s_add_i32 s1, s8, s18 // (op 627) 


s_ashr_i32 s8, s1, s9 // (op 627) 


s_lshl_b32 s1, s46, 24 // (op 627) 
s_ashr_i32 s9, s1, 31 // (op 627) 

s_xor_b32 s1, s8, s9 // (op 627) 

s_sub_i32 s8, s1, s9 // (op 627) 


s_add_u32 s18, s4, 32 // (op 627) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 627) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 627) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 627) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 627) 

s_mul_i32 s1, s45, s18 // (op 627) 

s_mul_i32 s9, s10, s1 // (op 627) 

s_mul_i32 s1, s8, s9 // (op 627) 

s_sub_u32 s9, s16, s1 // (op 627) 

s_mov_b32 s1, s9 // (op 627) 
s_mul_hi_i32 s18, s1, s44 // (op 627) 
s_add_i32 s19, s18, s1 // (op 627) 


s_ashr_i32 s1, s19, 31 // (op 627) 
s_cmp_eq_i32 s44, 0 // (op 627) 
s_mov_b32 s18, scc // (op 627) 
s_cmp_eq_i32 s18, 0 // (op 627) 
s_cselect_b32 s49, 0, -1 // (op 627) 

s_and_b32 s18, s48, 31 // (op 627) 
s_lshl_b32 s50, 1, s18 // (op 627) 
s_add_i32 s51, s50, s49 // (op 627) 


s_and_b32 s49, s1, s51 // (op 627) 


s_add_i32 s1, s19, s49 // (op 627) 


s_ashr_i32 s19, s1, s18 // (op 627) 


s_lshl_b32 s1, s48, 24 // (op 627) 
s_ashr_i32 s18, s1, 31 // (op 627) 

s_xor_b32 s1, s19, s18 // (op 627) 

s_sub_i32 s19, s1, s18 // (op 627) 


s_mul_i32 s1, s47, s45 // (op 627) 
s_add_u32 s50, s6, 16 // (op 627) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 627) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 627) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 627) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 4 // (op 627) 

s_sub_i32 s18, s50, s1 // (op 627) 

s_mul_i32 s49, s19, s18 // (op 627) 


s_sub_u32 s18, s9, s49 // (op 627) 


s_add_u32 s9, s18, s1 // (op 627) 


s_cmp_eq_i32 s8, 0 // (op 627) 
s_mov_b32 s1, scc // (op 627) 

s_cmp_eq_i32 s1, 0 // (op 627) 
s_cselect_b32 s8, s9, s0 // (op 627) 



v_lshl_add_u32 v10, s8, 4, v19 // (op 627) 


v_mul_lo_u32 v19, v10, s26 // (op 627) 
v_add_u32 v19, v19, v11 // (op 627) 


s_mul_i32 s1, s26, s7 // (op 629) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 629) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 629) least significant
s_mul_hi_u32 s9, s26, s6 // (op 629) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 629) most significant: sum
s_add_u32 s1, s1, s9 // (op 629) most significant: sum


s_mov_b32 s60, s28 // (op 629) reg expression
s_mov_b32 s61, s29 // (op 629) reg expression
s_mov_b32 s62, s0 // (op 629) reg expression
s_mov_b32 s63, 131072 // (op 629) call()

s_mul_hi_i32 s0, s16, s12 // (op 672) 
v_lshrrev_b32 v10, 2, v1 // (op 672) 
s_add_i32 s1, s0, s16 // (op 672) 

v_and_b32 v11, -4, v1 // (op 672) 
s_ashr_i32 s0, s1, 31 // (op 672) 
v_sub_u32 v20, v1, v11 // (op 672) 

s_cmp_eq_i32 s12, 0 // (op 672) 
s_mov_b32 s8, scc // (op 672) 
s_cmp_eq_i32 s8, 0 // (op 672) 
s_cselect_b32 s9, 0, -1 // (op 672) 

s_and_b32 s8, s13, 31 // (op 672) 
s_lshl_b32 s18, 1, s8 // (op 672) 
s_add_i32 s19, s18, s9 // (op 672) 


s_and_b32 s9, s0, s19 // (op 672) 


s_add_i32 s0, s1, s9 // (op 672) 


s_ashr_i32 s1, s0, s8 // (op 672) 


s_lshl_b32 s0, s13, 24 // (op 672) 
s_ashr_i32 s8, s0, 31 // (op 672) 

s_xor_b32 s0, s1, s8 // (op 672) 

s_sub_i32 s1, s0, s8 // (op 672) 


s_mul_hi_i32 s0, s1, s14 // (op 672) 
s_add_i32 s8, s0, s1 // (op 672) 

s_ashr_i32 s0, s8, 31 // (op 672) 
s_cmp_eq_i32 s14, 0 // (op 672) 
s_mov_b32 s9, scc // (op 672) 
s_cmp_eq_i32 s9, 0 // (op 672) 
s_cselect_b32 s18, 0, -1 // (op 672) 

s_and_b32 s9, s15, 31 // (op 672) 
s_lshl_b32 s19, 1, s9 // (op 672) 
s_add_i32 s49, s19, s18 // (op 672) 


s_and_b32 s18, s0, s49 // (op 672) 


s_add_i32 s0, s8, s18 // (op 672) 


s_ashr_i32 s8, s0, s9 // (op 672) 


s_lshl_b32 s0, s15, 24 // (op 672) 
s_ashr_i32 s9, s0, 31 // (op 672) 

s_xor_b32 s0, s8, s9 // (op 672) 

s_sub_i32 s8, s0, s9 // (op 672) 


s_add_u32 s18, s4, 32 // (op 672) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 672) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 672) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 672) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 672) 

s_mul_i32 s0, s8, s18 // (op 672) 

s_sub_i32 s8, s1, s0 // (op 672) 


s_mul_hi_i32 s9, s16, s11 // (op 672) 
s_add_i32 s0, s9, s16 // (op 672) 

s_ashr_i32 s1, s0, 31 // (op 672) 
s_cmp_eq_i32 s11, 0 // (op 672) 
s_mov_b32 s9, scc // (op 672) 
s_cmp_eq_i32 s9, 0 // (op 672) 
s_cselect_b32 s49, 0, -1 // (op 672) 

s_and_b32 s9, s46, 31 // (op 672) 
s_lshl_b32 s50, 1, s9 // (op 672) 
s_add_i32 s51, s50, s49 // (op 672) 


s_and_b32 s49, s1, s51 // (op 672) 


s_add_i32 s1, s0, s49 // (op 672) 


s_ashr_i32 s0, s1, s9 // (op 672) 


s_lshl_b32 s1, s46, 24 // (op 672) 
s_ashr_i32 s9, s1, 31 // (op 672) 

s_xor_b32 s1, s0, s9 // (op 672) 

s_sub_i32 s0, s1, s9 // (op 672) 


s_mul_i32 s1, s45, s18 // (op 672) 

s_mul_i32 s9, s10, s1 // (op 672) 

s_mul_i32 s1, s0, s9 // (op 672) 

s_sub_u32 s9, s16, s1 // (op 672) 

s_mul_hi_i32 s1, s9, s44 // (op 672) 
s_add_i32 s18, s1, s9 // (op 672) 


s_ashr_i32 s1, s18, 31 // (op 672) 
s_cmp_eq_i32 s44, 0 // (op 672) 
s_mov_b32 s9, scc // (op 672) 
s_cmp_eq_i32 s9, 0 // (op 672) 
s_cselect_b32 s19, 0, -1 // (op 672) 

s_and_b32 s9, s48, 31 // (op 672) 
s_lshl_b32 s49, 1, s9 // (op 672) 
s_add_i32 s50, s49, s19 // (op 672) 


s_and_b32 s19, s1, s50 // (op 672) 


s_add_i32 s1, s18, s19 // (op 672) 


s_ashr_i32 s18, s1, s9 // (op 672) 


s_lshl_b32 s1, s48, 24 // (op 672) 
s_ashr_i32 s9, s1, 31 // (op 672) 

s_xor_b32 s1, s18, s9 // (op 672) 

s_sub_i32 s18, s1, s9 // (op 672) 


s_cmp_eq_i32 s0, 0 // (op 672) 
s_mov_b32 s1, scc // (op 672) 

s_cmp_eq_i32 s1, 0 // (op 672) 
s_cselect_b32 s0, s18, s8 // (op 672) 



v_lshl_add_u32 v11, s0, 5, v10 // (op 672) 


v_mul_lo_u32 v10, v11, s30 // (op 672) 

v_add_u32 v11, v20, v10 // (op 672) 


s_mul_i32 s1, s30, s5 // (op 674) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 674) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 674) least significant
s_mul_hi_u32 s9, s30, s4 // (op 674) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 674) most significant: sum
s_add_u32 s1, s1, s9 // (op 674) most significant: sum


s_mov_b32 s64, s32 // (op 674) reg expression
s_mov_b32 s65, s33 // (op 674) reg expression
s_mov_b32 s66, s0 // (op 674) reg expression
s_mov_b32 s67, 131072 // (op 674) call()

s_mul_hi_i32 s0, s16, s12 // (op 691) 
v_lshrrev_b32 v10, 2, v1 // (op 691) 
s_add_i32 s1, s0, s16 // (op 691) 

v_and_b32 v20, -4, v1 // (op 691) 
s_ashr_i32 s0, s1, 31 // (op 691) 
v_sub_u32 v21, v1, v20 // (op 691) 

s_cmp_eq_i32 s12, 0 // (op 691) 
s_mov_b32 s8, scc // (op 691) 
s_cmp_eq_i32 s8, 0 // (op 691) 
s_cselect_b32 s9, 0, -1 // (op 691) 

s_and_b32 s8, s13, 31 // (op 691) 
s_lshl_b32 s18, 1, s8 // (op 691) 
s_add_i32 s19, s18, s9 // (op 691) 


s_and_b32 s9, s0, s19 // (op 691) 


s_add_i32 s0, s1, s9 // (op 691) 


s_ashr_i32 s1, s0, s8 // (op 691) 


s_lshl_b32 s0, s13, 24 // (op 691) 
s_ashr_i32 s8, s0, 31 // (op 691) 

s_xor_b32 s0, s1, s8 // (op 691) 

s_sub_i32 s1, s0, s8 // (op 691) 


s_mul_hi_i32 s0, s1, s14 // (op 691) 
s_add_i32 s8, s0, s1 // (op 691) 

s_ashr_i32 s0, s8, 31 // (op 691) 
s_cmp_eq_i32 s14, 0 // (op 691) 
s_mov_b32 s9, scc // (op 691) 
s_cmp_eq_i32 s9, 0 // (op 691) 
s_cselect_b32 s18, 0, -1 // (op 691) 

s_and_b32 s9, s15, 31 // (op 691) 
s_lshl_b32 s19, 1, s9 // (op 691) 
s_add_i32 s49, s19, s18 // (op 691) 


s_and_b32 s18, s0, s49 // (op 691) 


s_add_i32 s0, s8, s18 // (op 691) 


s_ashr_i32 s8, s0, s9 // (op 691) 


s_lshl_b32 s0, s15, 24 // (op 691) 
s_ashr_i32 s9, s0, 31 // (op 691) 

s_xor_b32 s0, s8, s9 // (op 691) 

s_sub_i32 s8, s0, s9 // (op 691) 


s_add_u32 s18, s4, 32 // (op 691) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 691) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 691) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 691) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 691) 

s_mul_i32 s0, s8, s18 // (op 691) 

s_sub_i32 s8, s1, s0 // (op 691) 


s_mul_hi_i32 s9, s16, s11 // (op 691) 
s_add_i32 s0, s9, s16 // (op 691) 

s_ashr_i32 s1, s0, 31 // (op 691) 
s_cmp_eq_i32 s11, 0 // (op 691) 
s_mov_b32 s9, scc // (op 691) 
s_cmp_eq_i32 s9, 0 // (op 691) 
s_cselect_b32 s49, 0, -1 // (op 691) 

s_and_b32 s9, s46, 31 // (op 691) 
s_lshl_b32 s50, 1, s9 // (op 691) 
s_add_i32 s51, s50, s49 // (op 691) 


s_and_b32 s49, s1, s51 // (op 691) 


s_add_i32 s1, s0, s49 // (op 691) 


s_ashr_i32 s0, s1, s9 // (op 691) 


s_lshl_b32 s1, s46, 24 // (op 691) 
s_ashr_i32 s9, s1, 31 // (op 691) 

s_xor_b32 s1, s0, s9 // (op 691) 

s_sub_i32 s0, s1, s9 // (op 691) 


s_mul_i32 s1, s45, s18 // (op 691) 

s_mul_i32 s9, s10, s1 // (op 691) 

s_mul_i32 s1, s0, s9 // (op 691) 

s_sub_u32 s9, s16, s1 // (op 691) 

s_mul_hi_i32 s1, s9, s44 // (op 691) 
s_add_i32 s18, s1, s9 // (op 691) 


s_ashr_i32 s1, s18, 31 // (op 691) 
s_cmp_eq_i32 s44, 0 // (op 691) 
s_mov_b32 s9, scc // (op 691) 
s_cmp_eq_i32 s9, 0 // (op 691) 
s_cselect_b32 s19, 0, -1 // (op 691) 

s_and_b32 s9, s48, 31 // (op 691) 
s_lshl_b32 s49, 1, s9 // (op 691) 
s_add_i32 s50, s49, s19 // (op 691) 


s_and_b32 s19, s1, s50 // (op 691) 


s_add_i32 s1, s18, s19 // (op 691) 


s_ashr_i32 s18, s1, s9 // (op 691) 


s_lshl_b32 s1, s48, 24 // (op 691) 
s_ashr_i32 s9, s1, 31 // (op 691) 

s_xor_b32 s1, s18, s9 // (op 691) 

s_sub_i32 s18, s1, s9 // (op 691) 


s_cmp_eq_i32 s0, 0 // (op 691) 
s_mov_b32 s1, scc // (op 691) 

s_cmp_eq_i32 s1, 0 // (op 691) 
s_cselect_b32 s0, s18, s8 // (op 691) 



v_lshl_add_u32 v20, s0, 5, v10 // (op 691) 


v_mul_lo_u32 v10, v20, s30 // (op 691) 

v_add_u32 v20, v21, v10 // (op 691) 


s_mul_i32 s1, s30, s5 // (op 693) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 693) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 693) least significant
s_mul_hi_u32 s9, s30, s4 // (op 693) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 693) most significant: sum
s_add_u32 s1, s1, s9 // (op 693) most significant: sum


s_mov_b32 s64, s32 // (op 693) reg expression
s_mov_b32 s65, s33 // (op 693) reg expression
s_mov_b32 s66, s0 // (op 693) reg expression
s_mov_b32 s67, 131072 // (op 693) call()




buffer_load_ubyte v10, v18, s[60:63], 0 offen offset: 0 // (op 341) Load value



buffer_load_dwordx4 v[22:25], v12, s[56:59], 0 offen // (op 345) Load value
buffer_load_dwordx2 v[26:27], v12, s[56:59], 0 offen offset:16 // (op 345) Load value



buffer_load_ubyte v21, v11, s[64:67], 0 offen offset: 0 // (op 348) Load value
v_and_b32 v28, -8, v1 // (op 757) 
v_sub_u32 v29, v1, v28 // (op 757) 

v_lshlrev_b32 v28, 4, v29 // (op 757) 

v_lshrrev_b32 v29, 3, v1 // (op 757) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 757) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 757) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 436) reg expression
buffer_load_dwordx4 v14, s[52:55], 0 offen offset: 0 lds // (op 436) Load value direct to lds




buffer_load_ubyte v28, v18, s[60:63], 0 offen offset:4 // (op 355) Load value




buffer_load_dwordx4 v[30:33], v12, s[56:59], 0 offen offset:96 // (op 359) Load value
buffer_load_dwordx2 v[34:35], v12, s[56:59], 0 offen offset:112 // (op 359) Load value




buffer_load_ubyte v12, v11, s[64:67], 0 offen offset:4 // (op 362) Load value

v_and_b32 v11, -8, v1 // (op 936) 
v_sub_u32 v18, v1, v11 // (op 936) 

v_lshlrev_b32 v11, 4, v18 // (op 936) 

v_lshrrev_b32 v18, 3, v1 // (op 936) 
v_lshl_add_u32 v29, v18, 7, v11 // (op 936) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 936) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 2048, s0 // (op 441) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 441) reg expression
s_mov_b32 s8, 128 // (op 441) call()
buffer_load_dwordx4 v14, s[52:55], s8 offen offset: 0 lds // (op 441) Load value direct to lds




v_and_b32 v11, -4, v1 // (op 810) 
v_sub_u32 v18, v1, v11 // (op 810) 

v_lshlrev_b32 v11, 5, v18 // (op 810) 

v_lshrrev_b32 v18, 2, v1 // (op 810) 
v_lshl_add_u32 v29, v18, 7, v11 // (op 810) 


v_mul_lo_u32 v11, v29, 6 // (op 810) 

v_lshrrev_b32 v18, 3, v11 // (op 810) 




s_waitcnt vmcnt(8)
ds_write_b128 v18, v[22:25] offset:4096 // (op 367) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v18, v[26:27] offset:4112 // (op 367) Store local data 

v_and_b32 v11, -4, v1 // (op 863) 
v_sub_u32 v18, v1, v11 // (op 863) 

v_and_b32 v11, 4294967292, v1 // (op 863) 
v_add_u32 v29, v11, v18 // (op 863) 





s_waitcnt vmcnt(6)
ds_write_b8 v29, v21 offset:7168 // (op 368) Store local data 

v_and_b32 v11, -8, v1 // (op 729) 
v_sub_u32 v18, v1, v11 // (op 729) 

v_lshlrev_b32 v11, 4, v18 // (op 729) 

v_lshrrev_b32 v18, 3, v1 // (op 729) 
v_lshl_add_u32 v29, v18, 7, v11 // (op 729) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 729) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v11, -4, v1 // (op 781) 
v_sub_u32 v18, v1, v11 // (op 781) 

v_lshlrev_b32 v11, 5, v18 // (op 781) 

v_lshrrev_b32 v18, 2, v1 // (op 781) 
v_lshl_add_u32 v29, v18, 7, v11 // (op 781) 


v_mul_lo_u32 v11, v29, 6 // (op 781) 

v_lshrrev_b32 v18, 3, v11 // (op 781) 

v_and_b32 v11, -4, v1 // (op 834) 
v_sub_u32 v29, v1, v11 // (op 834) 

v_and_b32 v11, 4294967292, v1 // (op 834) 
v_add_u32 v14, v11, v29 // (op 834) 


v_and_b32 v11, -8, v1 // (op 921) 
v_sub_u32 v15, v1, v11 // (op 921) 

v_lshlrev_b32 v11, 4, v15 // (op 921) 

v_lshrrev_b32 v15, 3, v1 // (op 921) 
v_lshl_add_u32 v29, v15, 7, v11 // (op 921) 


s_nop 0
v_readfirstlane_b32 s1, v29 // (op 921) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v11, -4, v1 // (op 960) 
v_sub_u32 v15, v1, v11 // (op 960) 

v_lshlrev_b32 v11, 5, v15 // (op 960) 

v_lshrrev_b32 v15, 2, v1 // (op 960) 
v_lshl_add_u32 v29, v15, 7, v11 // (op 960) 


v_mul_lo_u32 v11, v29, 6 // (op 960) 

v_lshrrev_b32 v15, 3, v11 // (op 960) 

v_and_b32 v11, -4, v1 // (op 984) 
v_sub_u32 v29, v1, v11 // (op 984) 

v_and_b32 v11, 4294967292, v1 // (op 984) 
v_add_u32 v36, v11, v29 // (op 984) 


s_mov_b32 s8, 0 // (op 57) call()
s_ashr_i64 s[18:19], s[34:35], 7 // (op 56) 
s_mov_b32 s9, s18 // (op 56) convert

s_and_b32 s18, s9, 4294967294 // (op 56) 

s_cmp_lt_u32 s8, s18 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56 if false)
RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56:
s_add_u32 s8, s8, 2 // (op 58) 
s_waitcnt vmcnt(5)
s_barrier  // (op 1029) 



ds_read_b128 v[38:41], v3 // (op 67) Load local data 
ds_read_b128 v[42:45], v3 offset:64 // (op 67) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1042) 



ds_read_b128 v[46:49], v4 offset:4096 // (op 72) Load local data 
ds_read_b64 v[50:51], v4 offset:4112 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1056) 



ds_read_u8 v11, v0 offset:7168 // (op 81) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[46:51], a[0:3], v10, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 84) 





v_mov_b32 v11, v19 // (op 77) 

buffer_load_ubyte v10, v11, s[60:63], 0 offen offset:8 // (op 77) Load value


v_mov_b32 v29, v13 // (op 10) 

buffer_load_dwordx4 v[22:25], v29, s[56:59], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[26:27], v29, s[56:59], 0 offen offset:208 // (op 10) Load value


v_mov_b32 v37, v20 // (op 21) 

buffer_load_ubyte v21, v37, s[64:67], 0 offen offset:8 // (op 21) Load value
s_barrier  // (op 1032) 


v_mov_b64 v[38:39], v[16:17] // (op 420) 

s_mov_b32 m0, s0 // (op 420) reg expression
s_mov_b32 s9, 256 // (op 420) call()
buffer_load_dwordx4 v38, s[52:55], s9 offen offset: 0 lds // (op 420) Load value direct to lds

s_barrier  // (op 1079) 



s_waitcnt vmcnt(8)
ds_write_b128 v15, v[30:33] offset:7296 // (op 239) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v15, v[34:35] offset:7312 // (op 239) Store local data 
s_barrier  // (op 1085) 



s_waitcnt vmcnt(6)
ds_write_b8 v36, v12 offset:10368 // (op 240) Store local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1070) 



ds_read_b128 v[40:43], v6 offset:2048 // (op 231) Load local data 
ds_read_b128 v[44:47], v6 offset:2112 // (op 231) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1076) 



ds_read_b128 v[48:51], v7 offset:7296 // (op 233) Load local data 
ds_read_b64 v[52:53], v7 offset:7312 // (op 233) Load local data 
s_barrier  // (op 1136) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1082) 



ds_read_u8 v54, v5 offset:10368 // (op 237) Load local data 
s_barrier  // (op 1144) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:53], a[0:3], v28, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 238) 






buffer_load_ubyte v28, v11, s[60:63], 0 offen offset:12 // (op 235) Load value




v_add_u32 v19, 8, v19 // (op 619) 
buffer_load_dwordx4 v[30:33], v29, s[56:59], 0 offen offset:288 // (op 241) Load value
buffer_load_dwordx2 v[34:35], v29, s[56:59], 0 offen offset:304 // (op 241) Load value




v_add_u32 v13, 192, v13 // (op 555) 
buffer_load_ubyte v12, v37, s[64:67], 0 offen offset:12 // (op 242) Load value

s_barrier  // (op 1073) 
v_add_u32 v20, 8, v20 // (op 683) 


s_add_u32 s9, 2048, s1 // (op 431) 

s_mov_b32 m0, s9 // (op 431) reg expression
s_mov_b32 s18, 384 // (op 431) call()
buffer_load_dwordx4 v38, s[52:55], s18 offen offset: 0 lds // (op 431) Load value direct to lds



s_barrier  // (op 1045) 
v_add_co_u32 v16, vcc, 256, v16 // (op 491) least significant half



v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 491) most significant half
s_waitcnt vmcnt(8)
ds_write_b128 v18, v[22:25] offset:4096 // (op 13) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v18, v[26:27] offset:4112 // (op 13) Store local data 
s_barrier  // (op 1059) 



s_waitcnt vmcnt(6)
ds_write_b8 v14, v21 offset:7168 // (op 26) Store local data 
s_ashr_i64 s[18:19], s[34:35], 7 // (op 56) 
s_mov_b32 s9, s18 // (op 56) convert

s_and_b32 s18, s9, 4294967294 // (op 56) 

s_cmp_lt_u32 s8, s18 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56 if true)
RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56:














s_barrier  // (op 1098) 
s_mul_hi_i32 s0, s16, s12 // (op 454) 
v_lshrrev_b32 v29, 3, v1 // (op 454) 
s_add_i32 s1, s0, s16 // (op 454) 

v_and_b32 v10, -8, v1 // (op 454) 
s_ashr_i32 s0, s1, 31 // (op 454) 
v_sub_u32 v11, v1, v10 // (op 454) 

s_cmp_eq_i32 s12, 0 // (op 454) 
s_mov_b32 s8, scc // (op 454) 
v_lshlrev_b32 v10, 4, v11 // (op 454) 

s_cmp_eq_i32 s8, 0 // (op 454) 
s_cselect_b32 s9, 0, -1 // (op 454) 

s_and_b32 s8, s13, 31 // (op 454) 
s_lshl_b32 s18, 1, s8 // (op 454) 
s_add_i32 s19, s18, s9 // (op 454) 


s_and_b32 s9, s0, s19 // (op 454) 


s_add_i32 s0, s1, s9 // (op 454) 


s_ashr_i32 s1, s0, s8 // (op 454) 


s_lshl_b32 s0, s13, 24 // (op 454) 
s_ashr_i32 s8, s0, 31 // (op 454) 

s_xor_b32 s0, s1, s8 // (op 454) 

s_sub_i32 s1, s0, s8 // (op 454) 


s_mul_hi_i32 s0, s1, s14 // (op 454) 
s_add_i32 s8, s0, s1 // (op 454) 

s_ashr_i32 s0, s8, 31 // (op 454) 
s_cmp_eq_i32 s14, 0 // (op 454) 
s_mov_b32 s9, scc // (op 454) 
s_cmp_eq_i32 s9, 0 // (op 454) 
s_cselect_b32 s18, 0, -1 // (op 454) 

s_and_b32 s9, s15, 31 // (op 454) 
s_lshl_b32 s19, 1, s9 // (op 454) 
s_add_i32 s49, s19, s18 // (op 454) 


s_and_b32 s18, s0, s49 // (op 454) 


s_add_i32 s0, s8, s18 // (op 454) 


s_ashr_i32 s8, s0, s9 // (op 454) 


s_lshl_b32 s0, s15, 24 // (op 454) 
s_ashr_i32 s9, s0, 31 // (op 454) 

s_xor_b32 s0, s8, s9 // (op 454) 

s_sub_i32 s8, s0, s9 // (op 454) 


s_mul_i32 s0, s1, s45 // (op 454) 

s_sub_u32 s1, s16, s0 // (op 454) 

s_mul_i32 s0, s8, s45 // (op 454) 
s_add_u32 s0, s0, s1 // (op 454) 


s_mul_hi_i32 s1, s16, s11 // (op 454) 
s_add_i32 s8, s1, s16 // (op 454) 

s_ashr_i32 s1, s8, 31 // (op 454) 
s_cmp_eq_i32 s11, 0 // (op 454) 
s_mov_b32 s9, scc // (op 454) 
s_cmp_eq_i32 s9, 0 // (op 454) 
s_cselect_b32 s18, 0, -1 // (op 454) 

s_and_b32 s9, s46, 31 // (op 454) 
s_lshl_b32 s19, 1, s9 // (op 454) 
s_add_i32 s49, s19, s18 // (op 454) 


s_and_b32 s18, s1, s49 // (op 454) 


s_add_i32 s1, s8, s18 // (op 454) 


s_ashr_i32 s8, s1, s9 // (op 454) 


s_lshl_b32 s1, s46, 24 // (op 454) 
s_ashr_i32 s9, s1, 31 // (op 454) 

s_xor_b32 s1, s8, s9 // (op 454) 

s_sub_i32 s8, s1, s9 // (op 454) 


s_add_u32 s18, s4, 32 // (op 454) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 454) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 454) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 454) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 454) 

s_mul_i32 s1, s45, s18 // (op 454) 

s_mul_i32 s9, s10, s1 // (op 454) 

s_mul_i32 s1, s8, s9 // (op 454) 

s_sub_u32 s9, s16, s1 // (op 454) 

s_mov_b32 s1, s9 // (op 454) 
s_mul_hi_i32 s18, s1, s44 // (op 454) 
s_add_i32 s19, s18, s1 // (op 454) 


s_ashr_i32 s1, s19, 31 // (op 454) 
s_cmp_eq_i32 s44, 0 // (op 454) 
s_mov_b32 s18, scc // (op 454) 
s_cmp_eq_i32 s18, 0 // (op 454) 
s_cselect_b32 s49, 0, -1 // (op 454) 

s_and_b32 s18, s48, 31 // (op 454) 
s_lshl_b32 s50, 1, s18 // (op 454) 
s_add_i32 s51, s50, s49 // (op 454) 


s_and_b32 s49, s1, s51 // (op 454) 


s_add_i32 s1, s19, s49 // (op 454) 


s_ashr_i32 s19, s1, s18 // (op 454) 


s_lshl_b32 s1, s48, 24 // (op 454) 
s_ashr_i32 s18, s1, 31 // (op 454) 

s_xor_b32 s1, s19, s18 // (op 454) 

s_sub_i32 s19, s1, s18 // (op 454) 


s_mul_i32 s1, s47, s45 // (op 454) 
s_add_u32 s50, s6, 16 // (op 454) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 454) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 454) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 454) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 454) 

s_sub_i32 s18, s50, s1 // (op 454) 

s_mul_i32 s49, s19, s18 // (op 454) 


s_sub_u32 s18, s9, s49 // (op 454) 


s_add_u32 s9, s18, s1 // (op 454) 


s_cmp_eq_i32 s8, 0 // (op 454) 
s_mov_b32 s1, scc // (op 454) 

s_cmp_eq_i32 s1, 0 // (op 454) 
s_cselect_b32 s8, s9, s0 // (op 454) 



v_lshl_add_u32 v11, s8, 4, v29 // (op 454) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 454) 
s_lshr_b32 s8, s0, 1 // (op 454) 

v_lshl_add_u32 v29, s8, 8, v10 // (op 454) 


v_mov_b32 v27, 0 // (op 454) convert
v_mov_b32 v26, v29 // (op 454) convert

v_mul_lo_u32 v25, v11, s3 // (op 454) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 454) most significant: high of low * low
v_mul_lo_u32 v24, v11, s2 // (op 454) least significant: low of low * low
v_add_u32 v25, v25, v10 // (op 454) most significant: sum

v_add_co_u32 v24, s[0:1], v24, v26 // (op 454) least significant half
v_addc_co_u32 v25, s[0:1], v25, v27, s[0:1] // (op 454) most significant half



s_mul_i32 s1, s2, s7 // (op 456) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 456) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 456) least significant
s_mul_hi_u32 s9, s2, s6 // (op 456) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 456) most significant: sum
s_add_u32 s1, s1, s9 // (op 456) most significant: sum


s_mov_b32 s52, s20 // (op 456) reg expression
s_mov_b32 s53, s21 // (op 456) reg expression
s_mov_b32 s54, s0 // (op 456) reg expression
s_mov_b32 s55, 131072 // (op 456) call()

s_mul_hi_i32 s0, s16, s12 // (op 521) 
v_lshrrev_b32 v29, 2, v1 // (op 521) 
s_add_i32 s1, s0, s16 // (op 521) 

v_and_b32 v10, -4, v1 // (op 521) 
s_ashr_i32 s0, s1, 31 // (op 521) 
v_sub_u32 v11, v1, v10 // (op 521) 

s_cmp_eq_i32 s12, 0 // (op 521) 
s_mov_b32 s8, scc // (op 521) 
v_lshlrev_b32 v10, 5, v11 // (op 521) 

s_cmp_eq_i32 s8, 0 // (op 521) 
s_cselect_b32 s9, 0, -1 // (op 521) 

s_and_b32 s8, s13, 31 // (op 521) 
s_lshl_b32 s18, 1, s8 // (op 521) 
s_add_i32 s19, s18, s9 // (op 521) 


s_and_b32 s9, s0, s19 // (op 521) 


s_add_i32 s0, s1, s9 // (op 521) 


s_ashr_i32 s1, s0, s8 // (op 521) 


s_lshl_b32 s0, s13, 24 // (op 521) 
s_ashr_i32 s8, s0, 31 // (op 521) 

s_xor_b32 s0, s1, s8 // (op 521) 

s_sub_i32 s1, s0, s8 // (op 521) 


s_mul_hi_i32 s0, s1, s14 // (op 521) 
s_add_i32 s8, s0, s1 // (op 521) 

s_ashr_i32 s0, s8, 31 // (op 521) 
s_cmp_eq_i32 s14, 0 // (op 521) 
s_mov_b32 s9, scc // (op 521) 
s_cmp_eq_i32 s9, 0 // (op 521) 
s_cselect_b32 s18, 0, -1 // (op 521) 

s_and_b32 s9, s15, 31 // (op 521) 
s_lshl_b32 s19, 1, s9 // (op 521) 
s_add_i32 s49, s19, s18 // (op 521) 


s_and_b32 s18, s0, s49 // (op 521) 


s_add_i32 s0, s8, s18 // (op 521) 


s_ashr_i32 s8, s0, s9 // (op 521) 


s_lshl_b32 s0, s15, 24 // (op 521) 
s_ashr_i32 s9, s0, 31 // (op 521) 

s_xor_b32 s0, s8, s9 // (op 521) 

s_sub_i32 s8, s0, s9 // (op 521) 


s_add_u32 s18, s4, 32 // (op 521) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 521) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 521) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 521) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 521) 

s_mul_i32 s0, s8, s18 // (op 521) 

s_sub_i32 s8, s1, s0 // (op 521) 


s_mul_hi_i32 s9, s16, s11 // (op 521) 
s_add_i32 s0, s9, s16 // (op 521) 

s_ashr_i32 s1, s0, 31 // (op 521) 
s_cmp_eq_i32 s11, 0 // (op 521) 
s_mov_b32 s9, scc // (op 521) 
s_cmp_eq_i32 s9, 0 // (op 521) 
s_cselect_b32 s49, 0, -1 // (op 521) 

s_and_b32 s9, s46, 31 // (op 521) 
s_lshl_b32 s50, 1, s9 // (op 521) 
s_add_i32 s51, s50, s49 // (op 521) 


s_and_b32 s49, s1, s51 // (op 521) 


s_add_i32 s1, s0, s49 // (op 521) 


s_ashr_i32 s0, s1, s9 // (op 521) 


s_lshl_b32 s1, s46, 24 // (op 521) 
s_ashr_i32 s9, s1, 31 // (op 521) 

s_xor_b32 s1, s0, s9 // (op 521) 

s_sub_i32 s0, s1, s9 // (op 521) 


s_mul_i32 s1, s45, s18 // (op 521) 

s_mul_i32 s9, s10, s1 // (op 521) 

s_mul_i32 s1, s0, s9 // (op 521) 

s_sub_u32 s9, s16, s1 // (op 521) 

s_mul_hi_i32 s1, s9, s44 // (op 521) 
s_add_i32 s18, s1, s9 // (op 521) 


s_ashr_i32 s1, s18, 31 // (op 521) 
s_cmp_eq_i32 s44, 0 // (op 521) 
s_mov_b32 s9, scc // (op 521) 
s_cmp_eq_i32 s9, 0 // (op 521) 
s_cselect_b32 s19, 0, -1 // (op 521) 

s_and_b32 s9, s48, 31 // (op 521) 
s_lshl_b32 s49, 1, s9 // (op 521) 
s_add_i32 s50, s49, s19 // (op 521) 


s_and_b32 s19, s1, s50 // (op 521) 


s_add_i32 s1, s18, s19 // (op 521) 


s_ashr_i32 s18, s1, s9 // (op 521) 


s_lshl_b32 s1, s48, 24 // (op 521) 
s_ashr_i32 s9, s1, 31 // (op 521) 

s_xor_b32 s1, s18, s9 // (op 521) 

s_sub_i32 s18, s1, s9 // (op 521) 


s_cmp_eq_i32 s0, 0 // (op 521) 
s_mov_b32 s1, scc // (op 521) 

s_cmp_eq_i32 s1, 0 // (op 521) 
s_cselect_b32 s0, s18, s8 // (op 521) 



v_lshl_add_u32 v11, s0, 5, v29 // (op 521) 


v_mul_lo_u32 v27, v11, s23 // (op 521) most significant: low of low * high
v_mul_hi_u32 v29, v11, s22 // (op 521) most significant: high of low * low
v_mul_lo_u32 v26, v11, s22 // (op 521) least significant: low of low * low
v_add_u32 v27, v27, v29 // (op 521) most significant: sum


s_ashr_i64 s[0:1], s[34:35], 7 // (op 521) 
s_lshr_b32 s8, s0, 1 // (op 521) 

v_lshl_add_u32 v11, s8, 8, v10 // (op 521) 


v_mov_b32 v23, 0 // (op 521) convert
v_mov_b32 v22, v11 // (op 521) convert

v_add_co_u32 v10, s[0:1], v22, v26 // (op 521) least significant half
v_addc_co_u32 v11, s[0:1], v23, v27, s[0:1] // (op 521) most significant half



v_mul_lo_u32 v27, v11, 6 // (op 521) most significant: low of high * low
v_mul_hi_u32 v29, v10, 6 // (op 521) most significant: high of low * low
v_mul_lo_u32 v26, v10, 6 // (op 521) least significant: low of low * low
v_add_u32 v27, v27, v29 // (op 521) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[26:27] // (op 521) 

v_mov_b32 v29, v10 // (op 521) convert

s_mul_i32 s1, s5, 6 // (op 523) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 523) least significant
s_mul_hi_u32 s8, s4, 6 // (op 523) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 523) most significant: sum

s_mul_i32 s9, s22, s1 // (op 523) most significant: low of low * high
s_mul_i32 s18, s23, s0 // (op 523) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 523) least significant
s_mul_hi_u32 s19, s22, s0 // (op 523) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 523) most significant: sum
s_add_u32 s9, s9, s19 // (op 523) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 523) 

s_mov_b32 s56, s24 // (op 523) reg expression
s_mov_b32 s57, s25 // (op 523) reg expression
s_mov_b32 s58, s0 // (op 523) reg expression
s_mov_b32 s59, 131072 // (op 523) call()

s_mul_hi_i32 s0, s16, s12 // (op 585) 
v_lshrrev_b32 v10, 7, v1 // (op 585) 
s_add_i32 s1, s0, s16 // (op 585) 

v_and_b32 v11, 15, v1 // (op 585) 
s_ashr_i32 s0, s1, 31 // (op 585) 
v_lshl_add_u32 v13, v10, 4, v11 // (op 585) 


s_cmp_eq_i32 s12, 0 // (op 585) 
s_mov_b32 s8, scc // (op 585) 
v_and_b32 v10, 63, v1 // (op 585) 
s_cmp_eq_i32 s8, 0 // (op 585) 
s_cselect_b32 s9, 0, -1 // (op 585) 

v_lshrrev_b32 v11, 4, v10 // (op 585) 

s_and_b32 s8, s13, 31 // (op 585) 
s_lshl_b32 s18, 1, s8 // (op 585) 
s_add_i32 s19, s18, s9 // (op 585) 


s_and_b32 s9, s0, s19 // (op 585) 


s_add_i32 s0, s1, s9 // (op 585) 


s_ashr_i32 s1, s0, s8 // (op 585) 


s_lshl_b32 s0, s13, 24 // (op 585) 
s_ashr_i32 s8, s0, 31 // (op 585) 

s_xor_b32 s0, s1, s8 // (op 585) 

s_sub_i32 s1, s0, s8 // (op 585) 


s_mul_hi_i32 s0, s1, s14 // (op 585) 
s_add_i32 s8, s0, s1 // (op 585) 

s_ashr_i32 s0, s8, 31 // (op 585) 
s_cmp_eq_i32 s14, 0 // (op 585) 
s_mov_b32 s9, scc // (op 585) 
s_cmp_eq_i32 s9, 0 // (op 585) 
s_cselect_b32 s18, 0, -1 // (op 585) 

s_and_b32 s9, s15, 31 // (op 585) 
s_lshl_b32 s19, 1, s9 // (op 585) 
s_add_i32 s49, s19, s18 // (op 585) 


s_and_b32 s18, s0, s49 // (op 585) 


s_add_i32 s0, s8, s18 // (op 585) 


s_ashr_i32 s8, s0, s9 // (op 585) 


s_lshl_b32 s0, s15, 24 // (op 585) 
s_ashr_i32 s9, s0, 31 // (op 585) 

s_xor_b32 s0, s8, s9 // (op 585) 

s_sub_i32 s8, s0, s9 // (op 585) 


s_mul_i32 s0, s1, s45 // (op 585) 

s_sub_u32 s1, s16, s0 // (op 585) 

s_mul_i32 s0, s8, s45 // (op 585) 
s_add_u32 s0, s0, s1 // (op 585) 


s_mul_hi_i32 s1, s16, s11 // (op 585) 
s_add_i32 s8, s1, s16 // (op 585) 

s_ashr_i32 s1, s8, 31 // (op 585) 
s_cmp_eq_i32 s11, 0 // (op 585) 
s_mov_b32 s9, scc // (op 585) 
s_cmp_eq_i32 s9, 0 // (op 585) 
s_cselect_b32 s18, 0, -1 // (op 585) 

s_and_b32 s9, s46, 31 // (op 585) 
s_lshl_b32 s19, 1, s9 // (op 585) 
s_add_i32 s49, s19, s18 // (op 585) 


s_and_b32 s18, s1, s49 // (op 585) 


s_add_i32 s1, s8, s18 // (op 585) 


s_ashr_i32 s8, s1, s9 // (op 585) 


s_lshl_b32 s1, s46, 24 // (op 585) 
s_ashr_i32 s9, s1, 31 // (op 585) 

s_xor_b32 s1, s8, s9 // (op 585) 

s_sub_i32 s8, s1, s9 // (op 585) 


s_add_u32 s18, s4, 32 // (op 585) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 585) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 585) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 585) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 585) 

s_mul_i32 s1, s45, s18 // (op 585) 

s_mul_i32 s9, s10, s1 // (op 585) 

s_mul_i32 s1, s8, s9 // (op 585) 

s_sub_u32 s9, s16, s1 // (op 585) 

s_mov_b32 s1, s9 // (op 585) 
s_mul_hi_i32 s18, s1, s44 // (op 585) 
s_add_i32 s19, s18, s1 // (op 585) 


s_ashr_i32 s1, s19, 31 // (op 585) 
s_cmp_eq_i32 s44, 0 // (op 585) 
s_mov_b32 s18, scc // (op 585) 
s_cmp_eq_i32 s18, 0 // (op 585) 
s_cselect_b32 s49, 0, -1 // (op 585) 

s_and_b32 s18, s48, 31 // (op 585) 
s_lshl_b32 s50, 1, s18 // (op 585) 
s_add_i32 s51, s50, s49 // (op 585) 


s_and_b32 s49, s1, s51 // (op 585) 


s_add_i32 s1, s19, s49 // (op 585) 


s_ashr_i32 s19, s1, s18 // (op 585) 


s_lshl_b32 s1, s48, 24 // (op 585) 
s_ashr_i32 s18, s1, 31 // (op 585) 

s_xor_b32 s1, s19, s18 // (op 585) 

s_sub_i32 s19, s1, s18 // (op 585) 


s_mul_i32 s1, s47, s45 // (op 585) 
s_add_u32 s50, s6, 16 // (op 585) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 585) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 585) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 585) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 585) 

s_sub_i32 s18, s50, s1 // (op 585) 

s_mul_i32 s49, s19, s18 // (op 585) 


s_sub_u32 s18, s9, s49 // (op 585) 


s_add_u32 s9, s18, s1 // (op 585) 


s_cmp_eq_i32 s8, 0 // (op 585) 
s_mov_b32 s1, scc // (op 585) 

s_cmp_eq_i32 s1, 0 // (op 585) 
s_cselect_b32 s8, s9, s0 // (op 585) 



v_lshl_add_u32 v10, s8, 4, v13 // (op 585) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 585) 
s_lshr_b32 s8, s0, 1 // (op 585) 

v_lshl_add_u32 v13, s8, 3, v11 // (op 585) 


v_mul_lo_u32 v11, v10, s26 // (op 585) 
v_add_u32 v11, v11, v13 // (op 585) 


s_mul_i32 s1, s26, s7 // (op 587) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 587) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 587) least significant
s_mul_hi_u32 s9, s26, s6 // (op 587) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 587) most significant: sum
s_add_u32 s1, s1, s9 // (op 587) most significant: sum


s_mov_b32 s60, s28 // (op 587) reg expression
s_mov_b32 s61, s29 // (op 587) reg expression
s_mov_b32 s62, s0 // (op 587) reg expression
s_mov_b32 s63, 131072 // (op 587) call()

s_mul_hi_i32 s0, s16, s12 // (op 649) 
v_lshrrev_b32 v10, 2, v1 // (op 649) 
s_add_i32 s1, s0, s16 // (op 649) 

v_and_b32 v13, -4, v1 // (op 649) 
s_ashr_i32 s0, s1, 31 // (op 649) 
v_sub_u32 v14, v1, v13 // (op 649) 

s_cmp_eq_i32 s12, 0 // (op 649) 
s_mov_b32 s8, scc // (op 649) 
s_cmp_eq_i32 s8, 0 // (op 649) 
s_cselect_b32 s9, 0, -1 // (op 649) 

s_and_b32 s8, s13, 31 // (op 649) 
s_lshl_b32 s18, 1, s8 // (op 649) 
s_add_i32 s19, s18, s9 // (op 649) 


s_and_b32 s9, s0, s19 // (op 649) 


s_add_i32 s0, s1, s9 // (op 649) 


s_ashr_i32 s1, s0, s8 // (op 649) 


s_lshl_b32 s0, s13, 24 // (op 649) 
s_ashr_i32 s8, s0, 31 // (op 649) 

s_xor_b32 s0, s1, s8 // (op 649) 

s_sub_i32 s1, s0, s8 // (op 649) 


s_mul_hi_i32 s0, s1, s14 // (op 649) 
s_add_i32 s8, s0, s1 // (op 649) 

s_ashr_i32 s0, s8, 31 // (op 649) 
s_cmp_eq_i32 s14, 0 // (op 649) 
s_mov_b32 s9, scc // (op 649) 
s_cmp_eq_i32 s9, 0 // (op 649) 
s_cselect_b32 s18, 0, -1 // (op 649) 

s_and_b32 s9, s15, 31 // (op 649) 
s_lshl_b32 s19, 1, s9 // (op 649) 
s_add_i32 s49, s19, s18 // (op 649) 


s_and_b32 s18, s0, s49 // (op 649) 


s_add_i32 s0, s8, s18 // (op 649) 


s_ashr_i32 s8, s0, s9 // (op 649) 


s_lshl_b32 s0, s15, 24 // (op 649) 
s_ashr_i32 s9, s0, 31 // (op 649) 

s_xor_b32 s0, s8, s9 // (op 649) 

s_sub_i32 s8, s0, s9 // (op 649) 


s_add_u32 s18, s4, 32 // (op 649) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 649) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 649) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 649) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 649) 

s_mul_i32 s0, s8, s18 // (op 649) 

s_sub_i32 s8, s1, s0 // (op 649) 


s_mul_hi_i32 s9, s16, s11 // (op 649) 
s_add_i32 s0, s9, s16 // (op 649) 

s_ashr_i32 s1, s0, 31 // (op 649) 
s_cmp_eq_i32 s11, 0 // (op 649) 
s_mov_b32 s9, scc // (op 649) 
s_cmp_eq_i32 s9, 0 // (op 649) 
s_cselect_b32 s49, 0, -1 // (op 649) 

s_and_b32 s9, s46, 31 // (op 649) 
s_lshl_b32 s50, 1, s9 // (op 649) 
s_add_i32 s51, s50, s49 // (op 649) 


s_and_b32 s49, s1, s51 // (op 649) 


s_add_i32 s1, s0, s49 // (op 649) 


s_ashr_i32 s0, s1, s9 // (op 649) 


s_lshl_b32 s1, s46, 24 // (op 649) 
s_ashr_i32 s9, s1, 31 // (op 649) 

s_xor_b32 s1, s0, s9 // (op 649) 

s_sub_i32 s0, s1, s9 // (op 649) 


s_mul_i32 s1, s45, s18 // (op 649) 

s_mul_i32 s9, s10, s1 // (op 649) 

s_mul_i32 s1, s0, s9 // (op 649) 

s_sub_u32 s9, s16, s1 // (op 649) 

s_mul_hi_i32 s1, s9, s44 // (op 649) 
s_add_i32 s18, s1, s9 // (op 649) 


s_ashr_i32 s1, s18, 31 // (op 649) 
s_cmp_eq_i32 s44, 0 // (op 649) 
s_mov_b32 s9, scc // (op 649) 
s_cmp_eq_i32 s9, 0 // (op 649) 
s_cselect_b32 s19, 0, -1 // (op 649) 

s_and_b32 s9, s48, 31 // (op 649) 
s_lshl_b32 s49, 1, s9 // (op 649) 
s_add_i32 s50, s49, s19 // (op 649) 


s_and_b32 s19, s1, s50 // (op 649) 


s_add_i32 s1, s18, s19 // (op 649) 


s_ashr_i32 s18, s1, s9 // (op 649) 


s_lshl_b32 s1, s48, 24 // (op 649) 
s_ashr_i32 s9, s1, 31 // (op 649) 

s_xor_b32 s1, s18, s9 // (op 649) 

s_sub_i32 s18, s1, s9 // (op 649) 


s_cmp_eq_i32 s0, 0 // (op 649) 
s_mov_b32 s1, scc // (op 649) 

s_cmp_eq_i32 s1, 0 // (op 649) 
s_cselect_b32 s0, s18, s8 // (op 649) 



v_lshl_add_u32 v13, s0, 5, v10 // (op 649) 


v_mul_lo_u32 v10, v13, s30 // (op 649) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 649) 
s_lshr_b32 s8, s0, 1 // (op 649) 

v_lshl_add_u32 v13, s8, 3, v14 // (op 649) 


v_add_u32 v14, v13, v10 // (op 649) 


s_mul_i32 s1, s30, s5 // (op 651) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 651) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 651) least significant
s_mul_hi_u32 s9, s30, s4 // (op 651) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 651) most significant: sum
s_add_u32 s1, s1, s9 // (op 651) most significant: sum


s_mov_b32 s64, s32 // (op 651) reg expression
s_mov_b32 s65, s33 // (op 651) reg expression
s_mov_b32 s66, s0 // (op 651) reg expression
s_mov_b32 s67, 131072 // (op 651) call()

v_and_b32 v10, -8, v1 // (op 742) 
v_sub_u32 v13, v1, v10 // (op 742) 

v_lshlrev_b32 v10, 4, v13 // (op 742) 

v_lshrrev_b32 v13, 3, v1 // (op 742) 
v_lshl_add_u32 v15, v13, 7, v10 // (op 742) 


s_nop 0
v_readfirstlane_b32 s0, v15 // (op 742) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 794) 
v_sub_u32 v13, v1, v10 // (op 794) 

v_lshlrev_b32 v10, 5, v13 // (op 794) 

v_lshrrev_b32 v13, 2, v1 // (op 794) 
v_lshl_add_u32 v15, v13, 7, v10 // (op 794) 


v_mul_lo_u32 v10, v15, 6 // (op 794) 

v_lshrrev_b32 v13, 3, v10 // (op 794) 

v_and_b32 v10, -4, v1 // (op 847) 
v_sub_u32 v15, v1, v10 // (op 847) 

v_and_b32 v10, 4294967292, v1 // (op 847) 
v_add_u32 v16, v10, v15 // (op 847) 


s_ashr_i64 s[8:9], s[34:35], 7 // (op 166) 
s_and_b32 s1, s8, 4294967294 // (op 166) 

s_ashr_i64 s[8:9], s[34:35], 7 // (op 165) 
s_mov_b32 s18, s8 // (op 165) convert

s_cmp_lt_u32 s1, s18 // (op 165) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165:
s_barrier  // (op 1035) 
s_add_u32 s1, s1, 1 // (op 167) 


v_mov_b64 v[22:23], v[24:25] // (op 425) 

s_mov_b32 m0, s0 // (op 425) reg expression
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 425) Load value direct to lds

v_add_co_u32 v24, vcc, 128, v24 // (op 444) least significant half
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 444) most significant half


v_mov_b32 v10, v29 // (op 183) 

buffer_load_dwordx4 v[18:21], v10, s[56:59], 0 offen // (op 183) Load value
buffer_load_dwordx2 v[22:23], v10, s[56:59], 0 offen offset:16 // (op 183) Load value

v_add_u32 v29, 96, v29 // (op 513) 


s_barrier  // (op 1048) 
v_mov_b32 v10, v14 // (op 184) 




s_waitcnt vmcnt(1)
ds_write_b128 v13, v[18:21] offset:4096 // (op 181) Store local data 
buffer_load_ubyte v12, v10, s[64:67], 0 offen offset: 0 // (op 184) Load value

v_add_u32 v14, 4, v14 // (op 641) 
s_waitcnt vmcnt(1)
ds_write_b64 v13, v[22:23] offset:4112 // (op 181) Store local data 

s_barrier  // (op 1038) 



s_barrier  // (op 1062) 
ds_read_b128 v[30:33], v3 // (op 173) Load local data 



ds_read_b128 v[34:37], v3 offset:64 // (op 173) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v16, v12 offset:7168 // (op 182) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1052) 



ds_read_b128 v[18:21], v4 offset:4096 // (op 175) Load local data 
ds_read_b64 v[22:23], v4 offset:4112 // (op 175) Load local data 


v_mov_b32 v10, v11 // (op 177) 

buffer_load_ubyte v12, v10, s[60:63], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v11, 4, v11 // (op 577) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1066) 



ds_read_u8 v10, v0 offset:7168 // (op 179) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[18:23], a[0:3], v12, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 180) 




s_ashr_i64 s[8:9], s[34:35], 7 // (op 165) 
s_mov_b32 s18, s8 // (op 165) convert

s_cmp_lt_u32 s1, s18 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165 if true)
RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165:





















s_barrier  // (op 1130) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
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

s_mul_hi_i32 s0, s16, s12 // (op 889) 
v_lshrrev_b32 v0, 7, v1 // (op 889) 
s_add_i32 s1, s0, s16 // (op 889) 

v_and_b32 v3, 63, v1 // (op 889) 
s_ashr_i32 s0, s1, 31 // (op 889) 
v_lshrrev_b32 v10, 4, v3 // (op 889) 
s_cmp_eq_i32 s12, 0 // (op 889) 
s_mov_b32 s2, scc // (op 889) 
v_lshlrev_b32 v11, 2, v10 // (op 889) 

s_cmp_eq_i32 s2, 0 // (op 889) 
s_cselect_b32 s3, 0, -1 // (op 889) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 889) 


s_and_b32 s2, s13, 31 // (op 889) 
v_lshrrev_b32 v0, 2, v3 // (op 889) 

s_lshl_b32 s8, 1, s2 // (op 889) 
v_and_b32 v3, 3, v0 // (op 889) 

s_add_i32 s9, s8, s3 // (op 889) 


v_and_b32 v0, 3, v1 // (op 889) 
s_and_b32 s3, s0, s9 // (op 889) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 889) 


s_add_i32 s0, s1, s3 // (op 889) 


v_lshrrev_b32 v0, 6, v1 // (op 889) 
s_ashr_i32 s1, s0, s2 // (op 889) 


v_and_b32 v3, 1, v0 // (op 889) 

s_lshl_b32 s0, s13, 24 // (op 889) 
v_lshl_add_u32 v0, v3, 4, v11 // (op 889) 


s_ashr_i32 s2, s0, 31 // (op 889) 

s_xor_b32 s0, s1, s2 // (op 889) 

s_sub_i32 s1, s0, s2 // (op 889) 


s_mul_hi_i32 s0, s1, s14 // (op 889) 
s_add_i32 s2, s0, s1 // (op 889) 

s_ashr_i32 s0, s2, 31 // (op 889) 
s_cmp_eq_i32 s14, 0 // (op 889) 
s_mov_b32 s3, scc // (op 889) 
s_cmp_eq_i32 s3, 0 // (op 889) 
s_cselect_b32 s8, 0, -1 // (op 889) 

s_and_b32 s3, s15, 31 // (op 889) 
s_lshl_b32 s9, 1, s3 // (op 889) 
s_add_i32 s18, s9, s8 // (op 889) 


s_and_b32 s8, s0, s18 // (op 889) 


s_add_i32 s0, s2, s8 // (op 889) 


s_ashr_i32 s2, s0, s3 // (op 889) 


s_lshl_b32 s0, s15, 24 // (op 889) 
s_ashr_i32 s3, s0, 31 // (op 889) 

s_xor_b32 s0, s2, s3 // (op 889) 

s_sub_i32 s2, s0, s3 // (op 889) 


s_mul_i32 s0, s1, s45 // (op 889) 
s_sub_u32 s3, s16, s0 // (op 889) 

s_mul_i32 s0, s2, s45 // (op 889) 
s_add_u32 s0, s0, s3 // (op 889) 

s_mul_hi_i32 s3, s16, s11 // (op 889) 
s_add_i32 s8, s3, s16 // (op 889) 

s_ashr_i32 s3, s8, 31 // (op 889) 
s_cmp_eq_i32 s11, 0 // (op 889) 
s_mov_b32 s9, scc // (op 889) 
s_cmp_eq_i32 s9, 0 // (op 889) 
s_cselect_b32 s18, 0, -1 // (op 889) 

s_and_b32 s9, s46, 31 // (op 889) 
s_lshl_b32 s19, 1, s9 // (op 889) 
s_add_i32 s20, s19, s18 // (op 889) 


s_and_b32 s18, s3, s20 // (op 889) 


s_add_i32 s3, s8, s18 // (op 889) 


s_ashr_i32 s8, s3, s9 // (op 889) 


s_lshl_b32 s3, s46, 24 // (op 889) 
s_ashr_i32 s9, s3, 31 // (op 889) 

s_xor_b32 s3, s8, s9 // (op 889) 

s_sub_i32 s8, s3, s9 // (op 889) 


s_add_u32 s18, s4, 32 // (op 889) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 889) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 889) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 889) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 889) 

s_mul_i32 s3, s45, s18 // (op 889) 
s_mul_i32 s9, s10, s3 // (op 889) 

s_mul_i32 s3, s8, s9 // (op 889) 

s_sub_u32 s9, s16, s3 // (op 889) 

s_mov_b32 s3, s9 // (op 889) 
s_mul_hi_i32 s20, s3, s44 // (op 889) 
s_add_i32 s21, s20, s3 // (op 889) 


s_ashr_i32 s3, s21, 31 // (op 889) 
s_cmp_eq_i32 s44, 0 // (op 889) 
s_mov_b32 s20, scc // (op 889) 
s_cmp_eq_i32 s20, 0 // (op 889) 
s_cselect_b32 s22, 0, -1 // (op 889) 

s_and_b32 s20, s48, 31 // (op 889) 
s_lshl_b32 s23, 1, s20 // (op 889) 
s_add_i32 s24, s23, s22 // (op 889) 


s_and_b32 s22, s3, s24 // (op 889) 


s_add_i32 s3, s21, s22 // (op 889) 


s_ashr_i32 s21, s3, s20 // (op 889) 


s_lshl_b32 s3, s48, 24 // (op 889) 
s_ashr_i32 s20, s3, 31 // (op 889) 

s_xor_b32 s3, s21, s20 // (op 889) 

s_sub_i32 s21, s3, s20 // (op 889) 


s_mul_i32 s3, s47, s45 // (op 889) 
s_add_u32 s22, s6, 16 // (op 889) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 889) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 889) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 889) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 889) 

s_sub_i32 s20, s22, s3 // (op 889) 

s_mul_i32 s22, s21, s20 // (op 889) 

s_sub_u32 s20, s9, s22 // (op 889) 


s_add_u32 s9, s20, s3 // (op 889) 


s_cmp_eq_i32 s8, 0 // (op 889) 
s_mov_b32 s3, scc // (op 889) 

s_cmp_eq_i32 s3, 0 // (op 889) 
s_cselect_b32 s8, s9, s0 // (op 889) 


s_mul_i32 s0, s2, s18 // (op 889) 


v_lshl_add_u32 v3, s8, 4, v10 // (op 889) 


s_sub_i32 s2, s1, s0 // (op 889) 


s_cmp_eq_i32 s3, 0 // (op 889) 
s_cselect_b32 s0, s21, s2 // (op 889) 



v_lshl_add_u32 v10, s0, 5, v0 // (op 889) 


v_mul_lo_u32 v0, v10, s36 // (op 889) 

v_add_lshl_u32 v10, v3, v0, 1 // (op 889) 


s_mul_i32 s9, s36, s5 // (op 891) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 891) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 891) least significant
s_mul_hi_u32 s1, s36, s4 // (op 891) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 891) most significant: sum
s_add_u32 s9, s9, s1 // (op 891) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 891) 

s_mov_b32 s32, s38 // (op 891) reg expression
s_mov_b32 s33, s39 // (op 891) reg expression
s_mov_b32 s34, s0 // (op 891) reg expression
s_mov_b32 s35, 131072 // (op 891) call()


v_pack_b32_f16 v12, v4, v5 // (op 54) 
v_pack_b32_f16 v13, v6, v7 // (op 54) 


buffer_store_dwordx2 v[12:13], v10, s[32:35], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:
s_mul_hi_i32 s0, s16, s12 // (op 712) 
v_lshrrev_b32 v0, 7, v1 // (op 712) 
s_add_i32 s1, s0, s16 // (op 712) 

v_and_b32 v3, 63, v1 // (op 712) 
s_ashr_i32 s0, s1, 31 // (op 712) 
v_lshrrev_b32 v4, 4, v3 // (op 712) 
s_cmp_eq_i32 s12, 0 // (op 712) 
s_mov_b32 s2, scc // (op 712) 
v_lshlrev_b32 v5, 2, v4 // (op 712) 

s_cmp_eq_i32 s2, 0 // (op 712) 
s_cselect_b32 s3, 0, -1 // (op 712) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 712) 


s_and_b32 s2, s13, 31 // (op 712) 
v_lshrrev_b32 v0, 2, v3 // (op 712) 

s_lshl_b32 s8, 1, s2 // (op 712) 
v_and_b32 v3, 3, v0 // (op 712) 

s_add_i32 s9, s8, s3 // (op 712) 


v_and_b32 v0, 3, v1 // (op 712) 
s_and_b32 s3, s0, s9 // (op 712) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 712) 


s_add_i32 s0, s1, s3 // (op 712) 


v_lshrrev_b32 v0, 6, v1 // (op 712) 
s_ashr_i32 s1, s0, s2 // (op 712) 


v_and_b32 v3, 1, v0 // (op 712) 

s_lshl_b32 s0, s13, 24 // (op 712) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 712) 


s_ashr_i32 s2, s0, 31 // (op 712) 

s_xor_b32 s0, s1, s2 // (op 712) 

s_sub_i32 s1, s0, s2 // (op 712) 


s_mul_hi_i32 s0, s1, s14 // (op 712) 
s_add_i32 s2, s0, s1 // (op 712) 

s_ashr_i32 s0, s2, 31 // (op 712) 
s_cmp_eq_i32 s14, 0 // (op 712) 
s_mov_b32 s3, scc // (op 712) 
s_cmp_eq_i32 s3, 0 // (op 712) 
s_cselect_b32 s8, 0, -1 // (op 712) 

s_and_b32 s3, s15, 31 // (op 712) 
s_lshl_b32 s9, 1, s3 // (op 712) 
s_add_i32 s18, s9, s8 // (op 712) 


s_and_b32 s8, s0, s18 // (op 712) 


s_add_i32 s0, s2, s8 // (op 712) 


s_ashr_i32 s2, s0, s3 // (op 712) 


s_lshl_b32 s0, s15, 24 // (op 712) 
s_ashr_i32 s3, s0, 31 // (op 712) 

s_xor_b32 s0, s2, s3 // (op 712) 

s_sub_i32 s2, s0, s3 // (op 712) 


s_mul_i32 s0, s1, s45 // (op 712) 
s_sub_u32 s3, s16, s0 // (op 712) 

s_mul_i32 s0, s2, s45 // (op 712) 
s_add_u32 s0, s0, s3 // (op 712) 

s_mul_hi_i32 s3, s16, s11 // (op 712) 
s_add_i32 s8, s3, s16 // (op 712) 

s_ashr_i32 s3, s8, 31 // (op 712) 
s_cmp_eq_i32 s11, 0 // (op 712) 
s_mov_b32 s9, scc // (op 712) 
s_cmp_eq_i32 s9, 0 // (op 712) 
s_cselect_b32 s18, 0, -1 // (op 712) 

s_and_b32 s9, s46, 31 // (op 712) 
s_lshl_b32 s19, 1, s9 // (op 712) 
s_add_i32 s20, s19, s18 // (op 712) 


s_and_b32 s18, s3, s20 // (op 712) 


s_add_i32 s3, s8, s18 // (op 712) 


s_ashr_i32 s8, s3, s9 // (op 712) 


s_lshl_b32 s3, s46, 24 // (op 712) 
s_ashr_i32 s9, s3, 31 // (op 712) 

s_xor_b32 s3, s8, s9 // (op 712) 

s_sub_i32 s8, s3, s9 // (op 712) 


s_add_u32 s18, s4, 32 // (op 712) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 712) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 712) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 712) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 712) 

s_mul_i32 s3, s45, s18 // (op 712) 
s_mul_i32 s9, s10, s3 // (op 712) 

s_mul_i32 s3, s8, s9 // (op 712) 

s_sub_u32 s9, s16, s3 // (op 712) 

s_mov_b32 s3, s9 // (op 712) 
s_mul_hi_i32 s20, s3, s44 // (op 712) 
s_add_i32 s21, s20, s3 // (op 712) 


s_ashr_i32 s3, s21, 31 // (op 712) 
s_cmp_eq_i32 s44, 0 // (op 712) 
s_mov_b32 s20, scc // (op 712) 
s_cmp_eq_i32 s20, 0 // (op 712) 
s_cselect_b32 s22, 0, -1 // (op 712) 

s_and_b32 s20, s48, 31 // (op 712) 
s_lshl_b32 s23, 1, s20 // (op 712) 
s_add_i32 s24, s23, s22 // (op 712) 


s_and_b32 s22, s3, s24 // (op 712) 


s_add_i32 s3, s21, s22 // (op 712) 


s_ashr_i32 s21, s3, s20 // (op 712) 


s_lshl_b32 s3, s48, 24 // (op 712) 
s_ashr_i32 s20, s3, 31 // (op 712) 

s_xor_b32 s3, s21, s20 // (op 712) 

s_sub_i32 s21, s3, s20 // (op 712) 


s_mul_i32 s3, s47, s45 // (op 712) 
s_add_u32 s22, s6, 16 // (op 712) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 712) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 712) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 712) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 712) 

s_sub_i32 s20, s22, s3 // (op 712) 

s_mul_i32 s22, s21, s20 // (op 712) 

s_sub_u32 s20, s9, s22 // (op 712) 


s_add_u32 s9, s20, s3 // (op 712) 


s_cmp_eq_i32 s8, 0 // (op 712) 
s_mov_b32 s3, scc // (op 712) 

s_cmp_eq_i32 s3, 0 // (op 712) 
s_cselect_b32 s8, s9, s0 // (op 712) 


s_mul_i32 s0, s2, s18 // (op 712) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 712) 


s_sub_i32 s2, s1, s0 // (op 712) 


s_cmp_eq_i32 s3, 0 // (op 712) 
s_cselect_b32 s0, s21, s2 // (op 712) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 712) 


v_mul_lo_u32 v0, v4, s40 // (op 712) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 712) 


s_mul_i32 s9, s40, s5 // (op 714) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 714) most significant: low of high * low
s_mul_i32 s8, s40, s4 // (op 714) least significant
s_mul_hi_u32 s1, s40, s4 // (op 714) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 714) most significant: sum
s_add_u32 s9, s9, s1 // (op 714) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 714) 

s_mov_b32 s28, s42 // (op 714) reg expression
s_mov_b32 s29, s43 // (op 714) reg expression
s_mov_b32 s30, s0 // (op 714) reg expression
s_mov_b32 s31, 131072 // (op 714) call()





buffer_load_dwordx2 v[6:7], v4, s[28:31], 0 offen // (op 138) Load value



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

s_mul_hi_i32 s0, s16, s12 // (op 906) 
v_lshrrev_b32 v0, 7, v1 // (op 906) 
s_add_i32 s1, s0, s16 // (op 906) 

v_and_b32 v3, 63, v1 // (op 906) 
s_ashr_i32 s0, s1, 31 // (op 906) 
v_lshrrev_b32 v4, 4, v3 // (op 906) 
s_cmp_eq_i32 s12, 0 // (op 906) 
s_mov_b32 s2, scc // (op 906) 
v_lshlrev_b32 v5, 2, v4 // (op 906) 

s_cmp_eq_i32 s2, 0 // (op 906) 
s_cselect_b32 s3, 0, -1 // (op 906) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 906) 


s_and_b32 s2, s13, 31 // (op 906) 
v_lshrrev_b32 v0, 2, v3 // (op 906) 

s_lshl_b32 s8, 1, s2 // (op 906) 
v_and_b32 v3, 3, v0 // (op 906) 

s_add_i32 s9, s8, s3 // (op 906) 


v_and_b32 v0, 3, v1 // (op 906) 
s_and_b32 s3, s0, s9 // (op 906) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 906) 


s_add_i32 s0, s1, s3 // (op 906) 


v_lshrrev_b32 v0, 6, v1 // (op 906) 
s_ashr_i32 s1, s0, s2 // (op 906) 


v_and_b32 v3, 1, v0 // (op 906) 

s_lshl_b32 s0, s13, 24 // (op 906) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 906) 


s_ashr_i32 s2, s0, 31 // (op 906) 

s_xor_b32 s0, s1, s2 // (op 906) 

s_sub_i32 s1, s0, s2 // (op 906) 


s_mul_hi_i32 s0, s1, s14 // (op 906) 
s_add_i32 s2, s0, s1 // (op 906) 

s_ashr_i32 s0, s2, 31 // (op 906) 
s_cmp_eq_i32 s14, 0 // (op 906) 
s_mov_b32 s3, scc // (op 906) 
s_cmp_eq_i32 s3, 0 // (op 906) 
s_cselect_b32 s8, 0, -1 // (op 906) 

s_and_b32 s3, s15, 31 // (op 906) 
s_lshl_b32 s9, 1, s3 // (op 906) 
s_add_i32 s18, s9, s8 // (op 906) 


s_and_b32 s8, s0, s18 // (op 906) 


s_add_i32 s0, s2, s8 // (op 906) 


s_ashr_i32 s2, s0, s3 // (op 906) 


s_lshl_b32 s0, s15, 24 // (op 906) 
s_ashr_i32 s3, s0, 31 // (op 906) 

s_xor_b32 s0, s2, s3 // (op 906) 

s_sub_i32 s2, s0, s3 // (op 906) 


s_mul_i32 s0, s1, s45 // (op 906) 
s_sub_u32 s3, s16, s0 // (op 906) 

s_mul_i32 s0, s2, s45 // (op 906) 
s_add_u32 s0, s0, s3 // (op 906) 

s_mul_hi_i32 s3, s16, s11 // (op 906) 
s_add_i32 s8, s3, s16 // (op 906) 

s_ashr_i32 s3, s8, 31 // (op 906) 
s_cmp_eq_i32 s11, 0 // (op 906) 
s_mov_b32 s9, scc // (op 906) 
s_cmp_eq_i32 s9, 0 // (op 906) 
s_cselect_b32 s18, 0, -1 // (op 906) 

s_and_b32 s9, s46, 31 // (op 906) 
s_lshl_b32 s19, 1, s9 // (op 906) 
s_add_i32 s20, s19, s18 // (op 906) 


s_and_b32 s18, s3, s20 // (op 906) 


s_add_i32 s3, s8, s18 // (op 906) 


s_ashr_i32 s8, s3, s9 // (op 906) 


s_lshl_b32 s3, s46, 24 // (op 906) 
s_ashr_i32 s9, s3, 31 // (op 906) 

s_xor_b32 s3, s8, s9 // (op 906) 

s_sub_i32 s8, s3, s9 // (op 906) 


s_add_u32 s18, s4, 32 // (op 906) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 906) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 906) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 906) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 906) 

s_mul_i32 s3, s45, s18 // (op 906) 
s_mul_i32 s9, s10, s3 // (op 906) 

s_mul_i32 s3, s8, s9 // (op 906) 

s_sub_u32 s9, s16, s3 // (op 906) 

s_mov_b32 s3, s9 // (op 906) 
s_mul_hi_i32 s20, s3, s44 // (op 906) 
s_add_i32 s21, s20, s3 // (op 906) 


s_ashr_i32 s3, s21, 31 // (op 906) 
s_cmp_eq_i32 s44, 0 // (op 906) 
s_mov_b32 s20, scc // (op 906) 
s_cmp_eq_i32 s20, 0 // (op 906) 
s_cselect_b32 s22, 0, -1 // (op 906) 

s_and_b32 s20, s48, 31 // (op 906) 
s_lshl_b32 s23, 1, s20 // (op 906) 
s_add_i32 s24, s23, s22 // (op 906) 


s_and_b32 s22, s3, s24 // (op 906) 


s_add_i32 s3, s21, s22 // (op 906) 


s_ashr_i32 s21, s3, s20 // (op 906) 


s_lshl_b32 s3, s48, 24 // (op 906) 
s_ashr_i32 s20, s3, 31 // (op 906) 

s_xor_b32 s3, s21, s20 // (op 906) 

s_sub_i32 s21, s3, s20 // (op 906) 


s_mul_i32 s3, s47, s45 // (op 906) 
s_add_u32 s22, s6, 16 // (op 906) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 906) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 906) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 906) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 906) 

s_sub_i32 s20, s22, s3 // (op 906) 

s_mul_i32 s22, s21, s20 // (op 906) 

s_sub_u32 s20, s9, s22 // (op 906) 


s_add_u32 s9, s20, s3 // (op 906) 


s_cmp_eq_i32 s8, 0 // (op 906) 
s_mov_b32 s3, scc // (op 906) 

s_cmp_eq_i32 s3, 0 // (op 906) 
s_cselect_b32 s8, s9, s0 // (op 906) 


s_mul_i32 s0, s2, s18 // (op 906) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 906) 


s_sub_i32 s2, s1, s0 // (op 906) 


s_cmp_eq_i32 s3, 0 // (op 906) 
s_cselect_b32 s0, s21, s2 // (op 906) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 906) 


v_mul_lo_u32 v0, v4, s36 // (op 906) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 906) 


s_mul_i32 s9, s36, s5 // (op 908) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 908) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 908) least significant
s_mul_hi_u32 s1, s36, s4 // (op 908) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 908) most significant: sum
s_add_u32 s9, s9, s1 // (op 908) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 908) 

s_mov_b32 s32, s38 // (op 908) reg expression
s_mov_b32 s33, s39 // (op 908) reg expression
s_mov_b32 s34, s0 // (op 908) reg expression
s_mov_b32 s35, 131072 // (op 908) call()


v_pack_b32_f16 v6, v10, v11 // (op 137) 
v_pack_b32_f16 v7, v12, v13 // (op 137) 


buffer_store_dwordx2 v[6:7], v4, s[32:35], 0 offen // (op 137) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end-RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 60
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 10496 // lds bytes
  .amdhsa_accum_offset 56
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
    DataTypeB: B6
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
  - .name: RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 10496
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 70
    .vgpr_count: 55
    .agpr_count: 4
    .max_flat_workgroup_size: 128
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
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
      - .name: Subtract_21
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 72
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 80
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 128
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 136
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 144
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: MagicMultiple_30
        .size: 4
        .offset: 152
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 156
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
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
...

.end_amdgpu_metadata

