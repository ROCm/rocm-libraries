// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_,@function
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 743) 
v_and_b32 v3, 15, v1 // (op 743) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 743) 


v_and_b32 v0, 63, v1 // (op 743) 
v_lshrrev_b32 v3, 4, v0 // (op 743) 

v_lshlrev_b32 v0, 5, v3 // (op 743) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 743) 


v_mul_lo_u32 v0, v3, 6 // (op 743) 

v_lshrrev_b32 v3, 3, v0 // (op 743) 

v_lshrrev_b32 v0, 6, v1 // (op 796) 
v_and_b32 v4, 1, v0 // (op 796) 

v_and_b32 v0, 15, v1 // (op 796) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 796) 


v_and_b32 v0, 63, v1 // (op 796) 
v_lshrrev_b32 v4, 4, v0 // (op 796) 

v_lshlrev_b32 v0, 5, v4 // (op 796) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 796) 


v_mul_lo_u32 v0, v4, 6 // (op 796) 

v_lshrrev_b32 v4, 3, v0 // (op 796) 

v_lshrrev_b32 v0, 6, v1 // (op 849) 
v_and_b32 v5, 1, v0 // (op 849) 

v_and_b32 v0, 15, v1 // (op 849) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 849) 


v_and_b32 v0, 63, v1 // (op 849) 
v_lshrrev_b32 v5, 4, v0 // (op 849) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 849) 


v_lshrrev_b32 v5, 7, v1 // (op 907) 
v_and_b32 v6, 15, v1 // (op 907) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 907) 


v_and_b32 v5, 63, v1 // (op 907) 
v_lshrrev_b32 v6, 4, v5 // (op 907) 

v_lshlrev_b32 v5, 5, v6 // (op 907) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 907) 


v_mul_lo_u32 v5, v6, 6 // (op 907) 

v_lshrrev_b32 v6, 3, v5 // (op 907) 

v_lshrrev_b32 v5, 6, v1 // (op 931) 
v_and_b32 v7, 1, v5 // (op 931) 

v_and_b32 v5, 15, v1 // (op 931) 
v_lshl_add_u32 v8, v7, 4, v5 // (op 931) 


v_and_b32 v5, 63, v1 // (op 931) 
v_lshrrev_b32 v7, 4, v5 // (op 931) 

v_lshlrev_b32 v5, 5, v7 // (op 931) 

v_lshl_add_u32 v7, v8, 7, v5 // (op 931) 


v_mul_lo_u32 v5, v7, 6 // (op 931) 

v_lshrrev_b32 v7, 3, v5 // (op 931) 

v_lshrrev_b32 v5, 6, v1 // (op 955) 
v_and_b32 v8, 1, v5 // (op 955) 

v_and_b32 v5, 15, v1 // (op 955) 
v_lshl_add_u32 v9, v8, 4, v5 // (op 955) 


v_and_b32 v5, 63, v1 // (op 955) 
v_lshrrev_b32 v8, 4, v5 // (op 955) 

v_lshl_add_u32 v5, v9, 2, v8 // (op 955) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v9, s9 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
s_mul_hi_i32 s0, s16, s12 // (op 453) 
v_lshrrev_b32 v10, 3, v1 // (op 453) 
s_add_i32 s1, s0, s16 // (op 453) 

v_and_b32 v11, -8, v1 // (op 453) 
s_ashr_i32 s0, s1, 31 // (op 453) 
v_sub_u32 v12, v1, v11 // (op 453) 

s_cmp_eq_i32 s12, 0 // (op 453) 
s_mov_b32 s8, scc // (op 453) 
v_lshlrev_b32 v11, 4, v12 // (op 453) 

s_cmp_eq_i32 s8, 0 // (op 453) 
s_cselect_b32 s9, 0, -1 // (op 453) 

v_mov_b32 v13, 0 // (op 453) convert
v_mov_b32 v12, v11 // (op 453) convert

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



s_waitcnt lgkmcnt(0)
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


s_add_u32 s18, s4, 32 // (op 453) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 453) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 453) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 453) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 453) 

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
s_sub_u32 s52, s50, 1 // (op 453) least significant half; sets scc
s_subb_u32 s53, s51, 0 // (op 453) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 4 // (op 453) 

s_sub_i32 s18, s50, s1 // (op 453) 

s_mul_i32 s49, s19, s18 // (op 453) 


s_sub_u32 s18, s9, s49 // (op 453) 


s_add_u32 s9, s18, s1 // (op 453) 


s_cmp_eq_i32 s8, 0 // (op 453) 
s_mov_b32 s1, scc // (op 453) 

s_cmp_eq_i32 s1, 0 // (op 453) 
s_cselect_b32 s8, s9, s0 // (op 453) 



v_lshl_add_u32 v11, s8, 4, v10 // (op 453) 


v_mul_lo_u32 v15, v11, s3 // (op 453) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 453) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 453) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 453) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 453) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 453) most significant half



v_mul_lo_u32 v11, v15, 6 // (op 453) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 453) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 453) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 453) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 453) 

v_mov_b32 v10, v12 // (op 453) convert

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

s_mul_hi_i32 s0, s16, s12 // (op 472) 
v_lshrrev_b32 v11, 3, v1 // (op 472) 
s_add_i32 s1, s0, s16 // (op 472) 

v_and_b32 v12, -8, v1 // (op 472) 
s_ashr_i32 s0, s1, 31 // (op 472) 
v_sub_u32 v13, v1, v12 // (op 472) 

s_cmp_eq_i32 s12, 0 // (op 472) 
s_mov_b32 s8, scc // (op 472) 
v_lshlrev_b32 v12, 4, v13 // (op 472) 

s_cmp_eq_i32 s8, 0 // (op 472) 
s_cselect_b32 s9, 0, -1 // (op 472) 

v_mov_b32 v15, 0 // (op 472) convert
v_mov_b32 v14, v12 // (op 472) convert

s_and_b32 s8, s13, 31 // (op 472) 
s_lshl_b32 s18, 1, s8 // (op 472) 
s_add_i32 s19, s18, s9 // (op 472) 


s_and_b32 s9, s0, s19 // (op 472) 


s_add_i32 s0, s1, s9 // (op 472) 


s_ashr_i32 s1, s0, s8 // (op 472) 


s_lshl_b32 s0, s13, 24 // (op 472) 
s_ashr_i32 s8, s0, 31 // (op 472) 

s_xor_b32 s0, s1, s8 // (op 472) 

s_sub_i32 s1, s0, s8 // (op 472) 


s_mul_hi_i32 s0, s1, s14 // (op 472) 
s_add_i32 s8, s0, s1 // (op 472) 

s_ashr_i32 s0, s8, 31 // (op 472) 
s_cmp_eq_i32 s14, 0 // (op 472) 
s_mov_b32 s9, scc // (op 472) 
s_cmp_eq_i32 s9, 0 // (op 472) 
s_cselect_b32 s18, 0, -1 // (op 472) 

s_and_b32 s9, s15, 31 // (op 472) 
s_lshl_b32 s19, 1, s9 // (op 472) 
s_add_i32 s49, s19, s18 // (op 472) 


s_and_b32 s18, s0, s49 // (op 472) 


s_add_i32 s0, s8, s18 // (op 472) 


s_ashr_i32 s8, s0, s9 // (op 472) 


s_lshl_b32 s0, s15, 24 // (op 472) 
s_ashr_i32 s9, s0, 31 // (op 472) 

s_xor_b32 s0, s8, s9 // (op 472) 

s_sub_i32 s8, s0, s9 // (op 472) 


s_mul_i32 s0, s1, s45 // (op 472) 

s_sub_u32 s1, s16, s0 // (op 472) 

s_mul_i32 s0, s8, s45 // (op 472) 
s_add_u32 s0, s0, s1 // (op 472) 


s_mul_hi_i32 s1, s16, s11 // (op 472) 
s_add_i32 s8, s1, s16 // (op 472) 

s_ashr_i32 s1, s8, 31 // (op 472) 
s_cmp_eq_i32 s11, 0 // (op 472) 
s_mov_b32 s9, scc // (op 472) 
s_cmp_eq_i32 s9, 0 // (op 472) 
s_cselect_b32 s18, 0, -1 // (op 472) 

s_and_b32 s9, s46, 31 // (op 472) 
s_lshl_b32 s19, 1, s9 // (op 472) 
s_add_i32 s49, s19, s18 // (op 472) 


s_and_b32 s18, s1, s49 // (op 472) 


s_add_i32 s1, s8, s18 // (op 472) 


s_ashr_i32 s8, s1, s9 // (op 472) 


s_lshl_b32 s1, s46, 24 // (op 472) 
s_ashr_i32 s9, s1, 31 // (op 472) 

s_xor_b32 s1, s8, s9 // (op 472) 

s_sub_i32 s8, s1, s9 // (op 472) 


s_add_u32 s18, s4, 32 // (op 472) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 472) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 472) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 472) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 472) 

s_mul_i32 s1, s45, s18 // (op 472) 

s_mul_i32 s9, s10, s1 // (op 472) 

s_mul_i32 s1, s8, s9 // (op 472) 

s_sub_u32 s9, s16, s1 // (op 472) 

s_mov_b32 s1, s9 // (op 472) 
s_mul_hi_i32 s18, s1, s44 // (op 472) 
s_add_i32 s19, s18, s1 // (op 472) 


s_ashr_i32 s1, s19, 31 // (op 472) 
s_cmp_eq_i32 s44, 0 // (op 472) 
s_mov_b32 s18, scc // (op 472) 
s_cmp_eq_i32 s18, 0 // (op 472) 
s_cselect_b32 s49, 0, -1 // (op 472) 

s_and_b32 s18, s48, 31 // (op 472) 
s_lshl_b32 s50, 1, s18 // (op 472) 
s_add_i32 s51, s50, s49 // (op 472) 


s_and_b32 s49, s1, s51 // (op 472) 


s_add_i32 s1, s19, s49 // (op 472) 


s_ashr_i32 s19, s1, s18 // (op 472) 


s_lshl_b32 s1, s48, 24 // (op 472) 
s_ashr_i32 s18, s1, 31 // (op 472) 

s_xor_b32 s1, s19, s18 // (op 472) 

s_sub_i32 s19, s1, s18 // (op 472) 


s_mul_i32 s1, s47, s45 // (op 472) 
s_add_u32 s50, s6, 16 // (op 472) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 472) End of Int64 add, unlocking SCC
s_sub_u32 s56, s50, 1 // (op 472) least significant half; sets scc
s_subb_u32 s57, s51, 0 // (op 472) most significant half; uses scc

s_ashr_i64 s[50:51], s[56:57], 4 // (op 472) 

s_sub_i32 s18, s50, s1 // (op 472) 

s_mul_i32 s49, s19, s18 // (op 472) 


s_sub_u32 s18, s9, s49 // (op 472) 


s_add_u32 s9, s18, s1 // (op 472) 


s_cmp_eq_i32 s8, 0 // (op 472) 
s_mov_b32 s1, scc // (op 472) 

s_cmp_eq_i32 s1, 0 // (op 472) 
s_cselect_b32 s8, s9, s0 // (op 472) 



v_lshl_add_u32 v12, s8, 4, v11 // (op 472) 


v_mul_lo_u32 v17, v12, s3 // (op 472) most significant: low of low * high
v_mul_hi_u32 v11, v12, s2 // (op 472) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 472) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 472) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 472) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 472) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 472) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 472) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 472) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 472) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 472) 

v_mov_b32 v11, v12 // (op 472) convert

s_mul_i32 s1, s7, 6 // (op 474) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 474) least significant
s_mul_hi_u32 s8, s6, 6 // (op 474) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 474) most significant: sum

s_mul_i32 s9, s2, s1 // (op 474) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 474) most significant: low of high * low
s_mul_i32 s8, s2, s0 // (op 474) least significant
s_mul_hi_u32 s19, s2, s0 // (op 474) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 474) most significant: sum
s_add_u32 s9, s9, s19 // (op 474) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 474) 

s_mov_b32 s52, s20 // (op 474) reg expression
s_mov_b32 s53, s21 // (op 474) reg expression
s_mov_b32 s54, s0 // (op 474) reg expression
s_mov_b32 s55, 131072 // (op 474) call()

s_mul_hi_i32 s0, s16, s12 // (op 517) 
v_lshrrev_b32 v12, 2, v1 // (op 517) 
s_add_i32 s1, s0, s16 // (op 517) 

v_and_b32 v13, -4, v1 // (op 517) 
s_ashr_i32 s0, s1, 31 // (op 517) 
v_sub_u32 v14, v1, v13 // (op 517) 

s_cmp_eq_i32 s12, 0 // (op 517) 
s_mov_b32 s8, scc // (op 517) 
v_lshlrev_b32 v13, 5, v14 // (op 517) 

s_cmp_eq_i32 s8, 0 // (op 517) 
s_cselect_b32 s9, 0, -1 // (op 517) 

v_mov_b32 v15, 0 // (op 517) convert
v_mov_b32 v14, v13 // (op 517) convert

s_and_b32 s8, s13, 31 // (op 517) 
s_lshl_b32 s18, 1, s8 // (op 517) 
s_add_i32 s19, s18, s9 // (op 517) 


s_and_b32 s9, s0, s19 // (op 517) 


s_add_i32 s0, s1, s9 // (op 517) 


s_ashr_i32 s1, s0, s8 // (op 517) 


s_lshl_b32 s0, s13, 24 // (op 517) 
s_ashr_i32 s8, s0, 31 // (op 517) 

s_xor_b32 s0, s1, s8 // (op 517) 

s_sub_i32 s1, s0, s8 // (op 517) 


s_mul_hi_i32 s0, s1, s14 // (op 517) 
s_add_i32 s8, s0, s1 // (op 517) 

s_ashr_i32 s0, s8, 31 // (op 517) 
s_cmp_eq_i32 s14, 0 // (op 517) 
s_mov_b32 s9, scc // (op 517) 
s_cmp_eq_i32 s9, 0 // (op 517) 
s_cselect_b32 s18, 0, -1 // (op 517) 

s_and_b32 s9, s15, 31 // (op 517) 
s_lshl_b32 s19, 1, s9 // (op 517) 
s_add_i32 s49, s19, s18 // (op 517) 


s_and_b32 s18, s0, s49 // (op 517) 


s_add_i32 s0, s8, s18 // (op 517) 


s_ashr_i32 s8, s0, s9 // (op 517) 


s_lshl_b32 s0, s15, 24 // (op 517) 
s_ashr_i32 s9, s0, 31 // (op 517) 

s_xor_b32 s0, s8, s9 // (op 517) 

s_sub_i32 s8, s0, s9 // (op 517) 


s_add_u32 s18, s4, 32 // (op 517) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 517) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 517) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 517) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 517) 

s_mul_i32 s0, s8, s18 // (op 517) 

s_sub_i32 s8, s1, s0 // (op 517) 


s_mul_hi_i32 s9, s16, s11 // (op 517) 
s_add_i32 s0, s9, s16 // (op 517) 

s_ashr_i32 s1, s0, 31 // (op 517) 
s_cmp_eq_i32 s11, 0 // (op 517) 
s_mov_b32 s9, scc // (op 517) 
s_cmp_eq_i32 s9, 0 // (op 517) 
s_cselect_b32 s49, 0, -1 // (op 517) 

s_and_b32 s9, s46, 31 // (op 517) 
s_lshl_b32 s50, 1, s9 // (op 517) 
s_add_i32 s51, s50, s49 // (op 517) 


s_and_b32 s49, s1, s51 // (op 517) 


s_add_i32 s1, s0, s49 // (op 517) 


s_ashr_i32 s0, s1, s9 // (op 517) 


s_lshl_b32 s1, s46, 24 // (op 517) 
s_ashr_i32 s9, s1, 31 // (op 517) 

s_xor_b32 s1, s0, s9 // (op 517) 

s_sub_i32 s0, s1, s9 // (op 517) 


s_mul_i32 s1, s45, s18 // (op 517) 

s_mul_i32 s9, s10, s1 // (op 517) 

s_mul_i32 s1, s0, s9 // (op 517) 

s_sub_u32 s9, s16, s1 // (op 517) 

s_mul_hi_i32 s1, s9, s44 // (op 517) 
s_add_i32 s18, s1, s9 // (op 517) 


s_ashr_i32 s1, s18, 31 // (op 517) 
s_cmp_eq_i32 s44, 0 // (op 517) 
s_mov_b32 s9, scc // (op 517) 
s_cmp_eq_i32 s9, 0 // (op 517) 
s_cselect_b32 s19, 0, -1 // (op 517) 

s_and_b32 s9, s48, 31 // (op 517) 
s_lshl_b32 s49, 1, s9 // (op 517) 
s_add_i32 s50, s49, s19 // (op 517) 


s_and_b32 s19, s1, s50 // (op 517) 


s_add_i32 s1, s18, s19 // (op 517) 


s_ashr_i32 s18, s1, s9 // (op 517) 


s_lshl_b32 s1, s48, 24 // (op 517) 
s_ashr_i32 s9, s1, 31 // (op 517) 

s_xor_b32 s1, s18, s9 // (op 517) 

s_sub_i32 s18, s1, s9 // (op 517) 


s_cmp_eq_i32 s0, 0 // (op 517) 
s_mov_b32 s1, scc // (op 517) 

s_cmp_eq_i32 s1, 0 // (op 517) 
s_cselect_b32 s0, s18, s8 // (op 517) 



v_lshl_add_u32 v13, s0, 5, v12 // (op 517) 


v_mul_lo_u32 v17, v13, s23 // (op 517) most significant: low of low * high
v_mul_hi_u32 v12, v13, s22 // (op 517) most significant: high of low * low
v_mul_lo_u32 v16, v13, s22 // (op 517) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 517) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 517) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 517) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 517) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 517) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 517) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 517) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 517) 

v_mov_b32 v14, v12 // (op 517) convert

s_mul_i32 s1, s5, 6 // (op 519) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 519) least significant
s_mul_hi_u32 s8, s4, 6 // (op 519) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 519) most significant: sum

s_mul_i32 s9, s22, s1 // (op 519) most significant: low of low * high
s_mul_i32 s18, s23, s0 // (op 519) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 519) least significant
s_mul_hi_u32 s19, s22, s0 // (op 519) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 519) most significant: sum
s_add_u32 s9, s9, s19 // (op 519) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 519) 

s_mov_b32 s56, s24 // (op 519) reg expression
s_mov_b32 s57, s25 // (op 519) reg expression
s_mov_b32 s58, s0 // (op 519) reg expression
s_mov_b32 s59, 131072 // (op 519) call()

s_mul_hi_i32 s0, s16, s12 // (op 536) 
v_lshrrev_b32 v12, 2, v1 // (op 536) 
s_add_i32 s1, s0, s16 // (op 536) 

v_and_b32 v13, -4, v1 // (op 536) 
s_ashr_i32 s0, s1, 31 // (op 536) 
v_sub_u32 v15, v1, v13 // (op 536) 

s_cmp_eq_i32 s12, 0 // (op 536) 
s_mov_b32 s8, scc // (op 536) 
v_lshlrev_b32 v13, 5, v15 // (op 536) 

s_cmp_eq_i32 s8, 0 // (op 536) 
s_cselect_b32 s9, 0, -1 // (op 536) 

v_mov_b32 v17, 0 // (op 536) convert
v_mov_b32 v16, v13 // (op 536) convert

s_and_b32 s8, s13, 31 // (op 536) 
s_lshl_b32 s18, 1, s8 // (op 536) 
s_add_i32 s19, s18, s9 // (op 536) 


s_and_b32 s9, s0, s19 // (op 536) 


s_add_i32 s0, s1, s9 // (op 536) 


s_ashr_i32 s1, s0, s8 // (op 536) 


s_lshl_b32 s0, s13, 24 // (op 536) 
s_ashr_i32 s8, s0, 31 // (op 536) 

s_xor_b32 s0, s1, s8 // (op 536) 

s_sub_i32 s1, s0, s8 // (op 536) 


s_mul_hi_i32 s0, s1, s14 // (op 536) 
s_add_i32 s8, s0, s1 // (op 536) 

s_ashr_i32 s0, s8, 31 // (op 536) 
s_cmp_eq_i32 s14, 0 // (op 536) 
s_mov_b32 s9, scc // (op 536) 
s_cmp_eq_i32 s9, 0 // (op 536) 
s_cselect_b32 s18, 0, -1 // (op 536) 

s_and_b32 s9, s15, 31 // (op 536) 
s_lshl_b32 s19, 1, s9 // (op 536) 
s_add_i32 s49, s19, s18 // (op 536) 


s_and_b32 s18, s0, s49 // (op 536) 


s_add_i32 s0, s8, s18 // (op 536) 


s_ashr_i32 s8, s0, s9 // (op 536) 


s_lshl_b32 s0, s15, 24 // (op 536) 
s_ashr_i32 s9, s0, 31 // (op 536) 

s_xor_b32 s0, s8, s9 // (op 536) 

s_sub_i32 s8, s0, s9 // (op 536) 


s_add_u32 s18, s4, 32 // (op 536) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 536) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 536) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 536) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 536) 

s_mul_i32 s0, s8, s18 // (op 536) 

s_sub_i32 s8, s1, s0 // (op 536) 


s_mul_hi_i32 s9, s16, s11 // (op 536) 
s_add_i32 s0, s9, s16 // (op 536) 

s_ashr_i32 s1, s0, 31 // (op 536) 
s_cmp_eq_i32 s11, 0 // (op 536) 
s_mov_b32 s9, scc // (op 536) 
s_cmp_eq_i32 s9, 0 // (op 536) 
s_cselect_b32 s49, 0, -1 // (op 536) 

s_and_b32 s9, s46, 31 // (op 536) 
s_lshl_b32 s50, 1, s9 // (op 536) 
s_add_i32 s51, s50, s49 // (op 536) 


s_and_b32 s49, s1, s51 // (op 536) 


s_add_i32 s1, s0, s49 // (op 536) 


s_ashr_i32 s0, s1, s9 // (op 536) 


s_lshl_b32 s1, s46, 24 // (op 536) 
s_ashr_i32 s9, s1, 31 // (op 536) 

s_xor_b32 s1, s0, s9 // (op 536) 

s_sub_i32 s0, s1, s9 // (op 536) 


s_mul_i32 s1, s45, s18 // (op 536) 

s_mul_i32 s9, s10, s1 // (op 536) 

s_mul_i32 s1, s0, s9 // (op 536) 

s_sub_u32 s9, s16, s1 // (op 536) 

s_mul_hi_i32 s1, s9, s44 // (op 536) 
s_add_i32 s18, s1, s9 // (op 536) 


s_ashr_i32 s1, s18, 31 // (op 536) 
s_cmp_eq_i32 s44, 0 // (op 536) 
s_mov_b32 s9, scc // (op 536) 
s_cmp_eq_i32 s9, 0 // (op 536) 
s_cselect_b32 s19, 0, -1 // (op 536) 

s_and_b32 s9, s48, 31 // (op 536) 
s_lshl_b32 s49, 1, s9 // (op 536) 
s_add_i32 s50, s49, s19 // (op 536) 


s_and_b32 s19, s1, s50 // (op 536) 


s_add_i32 s1, s18, s19 // (op 536) 


s_ashr_i32 s18, s1, s9 // (op 536) 


s_lshl_b32 s1, s48, 24 // (op 536) 
s_ashr_i32 s9, s1, 31 // (op 536) 

s_xor_b32 s1, s18, s9 // (op 536) 

s_sub_i32 s18, s1, s9 // (op 536) 


s_cmp_eq_i32 s0, 0 // (op 536) 
s_mov_b32 s1, scc // (op 536) 

s_cmp_eq_i32 s1, 0 // (op 536) 
s_cselect_b32 s0, s18, s8 // (op 536) 



v_lshl_add_u32 v13, s0, 5, v12 // (op 536) 


v_mul_lo_u32 v19, v13, s23 // (op 536) most significant: low of low * high
v_mul_hi_u32 v12, v13, s22 // (op 536) most significant: high of low * low
v_mul_lo_u32 v18, v13, s22 // (op 536) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 536) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 536) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 536) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 536) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 536) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 536) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 536) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 536) 

v_mov_b32 v15, v12 // (op 536) convert

s_mul_i32 s1, s5, 6 // (op 538) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 538) least significant
s_mul_hi_u32 s8, s4, 6 // (op 538) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 538) most significant: sum

s_mul_i32 s9, s22, s1 // (op 538) most significant: low of low * high
s_mul_i32 s18, s23, s0 // (op 538) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 538) least significant
s_mul_hi_u32 s19, s22, s0 // (op 538) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 538) most significant: sum
s_add_u32 s9, s9, s19 // (op 538) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 538) 

s_mov_b32 s56, s24 // (op 538) reg expression
s_mov_b32 s57, s25 // (op 538) reg expression
s_mov_b32 s58, s0 // (op 538) reg expression
s_mov_b32 s59, 131072 // (op 538) call()

s_mul_hi_i32 s0, s16, s12 // (op 581) 
v_lshrrev_b32 v12, 7, v1 // (op 581) 
s_add_i32 s1, s0, s16 // (op 581) 

v_and_b32 v13, 15, v1 // (op 581) 
s_ashr_i32 s0, s1, 31 // (op 581) 
v_lshl_add_u32 v16, v12, 4, v13 // (op 581) 


s_cmp_eq_i32 s12, 0 // (op 581) 
s_mov_b32 s8, scc // (op 581) 
v_and_b32 v12, 63, v1 // (op 581) 
s_cmp_eq_i32 s8, 0 // (op 581) 
s_cselect_b32 s9, 0, -1 // (op 581) 

v_lshrrev_b32 v13, 4, v12 // (op 581) 

s_and_b32 s8, s13, 31 // (op 581) 
s_lshl_b32 s18, 1, s8 // (op 581) 
s_add_i32 s19, s18, s9 // (op 581) 


s_and_b32 s9, s0, s19 // (op 581) 


s_add_i32 s0, s1, s9 // (op 581) 


s_ashr_i32 s1, s0, s8 // (op 581) 


s_lshl_b32 s0, s13, 24 // (op 581) 
s_ashr_i32 s8, s0, 31 // (op 581) 

s_xor_b32 s0, s1, s8 // (op 581) 

s_sub_i32 s1, s0, s8 // (op 581) 


s_mul_hi_i32 s0, s1, s14 // (op 581) 
s_add_i32 s8, s0, s1 // (op 581) 

s_ashr_i32 s0, s8, 31 // (op 581) 
s_cmp_eq_i32 s14, 0 // (op 581) 
s_mov_b32 s9, scc // (op 581) 
s_cmp_eq_i32 s9, 0 // (op 581) 
s_cselect_b32 s18, 0, -1 // (op 581) 

s_and_b32 s9, s15, 31 // (op 581) 
s_lshl_b32 s19, 1, s9 // (op 581) 
s_add_i32 s49, s19, s18 // (op 581) 


s_and_b32 s18, s0, s49 // (op 581) 


s_add_i32 s0, s8, s18 // (op 581) 


s_ashr_i32 s8, s0, s9 // (op 581) 


s_lshl_b32 s0, s15, 24 // (op 581) 
s_ashr_i32 s9, s0, 31 // (op 581) 

s_xor_b32 s0, s8, s9 // (op 581) 

s_sub_i32 s8, s0, s9 // (op 581) 


s_mul_i32 s0, s1, s45 // (op 581) 

s_sub_u32 s1, s16, s0 // (op 581) 

s_mul_i32 s0, s8, s45 // (op 581) 
s_add_u32 s0, s0, s1 // (op 581) 


s_mul_hi_i32 s1, s16, s11 // (op 581) 
s_add_i32 s8, s1, s16 // (op 581) 

s_ashr_i32 s1, s8, 31 // (op 581) 
s_cmp_eq_i32 s11, 0 // (op 581) 
s_mov_b32 s9, scc // (op 581) 
s_cmp_eq_i32 s9, 0 // (op 581) 
s_cselect_b32 s18, 0, -1 // (op 581) 

s_and_b32 s9, s46, 31 // (op 581) 
s_lshl_b32 s19, 1, s9 // (op 581) 
s_add_i32 s49, s19, s18 // (op 581) 


s_and_b32 s18, s1, s49 // (op 581) 


s_add_i32 s1, s8, s18 // (op 581) 


s_ashr_i32 s8, s1, s9 // (op 581) 


s_lshl_b32 s1, s46, 24 // (op 581) 
s_ashr_i32 s9, s1, 31 // (op 581) 

s_xor_b32 s1, s8, s9 // (op 581) 

s_sub_i32 s8, s1, s9 // (op 581) 


s_add_u32 s18, s4, 32 // (op 581) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 581) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 581) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 581) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 581) 

s_mul_i32 s1, s45, s18 // (op 581) 

s_mul_i32 s9, s10, s1 // (op 581) 

s_mul_i32 s1, s8, s9 // (op 581) 

s_sub_u32 s9, s16, s1 // (op 581) 

s_mov_b32 s1, s9 // (op 581) 
s_mul_hi_i32 s18, s1, s44 // (op 581) 
s_add_i32 s19, s18, s1 // (op 581) 


s_ashr_i32 s1, s19, 31 // (op 581) 
s_cmp_eq_i32 s44, 0 // (op 581) 
s_mov_b32 s18, scc // (op 581) 
s_cmp_eq_i32 s18, 0 // (op 581) 
s_cselect_b32 s49, 0, -1 // (op 581) 

s_and_b32 s18, s48, 31 // (op 581) 
s_lshl_b32 s50, 1, s18 // (op 581) 
s_add_i32 s51, s50, s49 // (op 581) 


s_and_b32 s49, s1, s51 // (op 581) 


s_add_i32 s1, s19, s49 // (op 581) 


s_ashr_i32 s19, s1, s18 // (op 581) 


s_lshl_b32 s1, s48, 24 // (op 581) 
s_ashr_i32 s18, s1, 31 // (op 581) 

s_xor_b32 s1, s19, s18 // (op 581) 

s_sub_i32 s19, s1, s18 // (op 581) 


s_mul_i32 s1, s47, s45 // (op 581) 
s_add_u32 s50, s6, 16 // (op 581) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 581) End of Int64 add, unlocking SCC
s_sub_u32 s60, s50, 1 // (op 581) least significant half; sets scc
s_subb_u32 s61, s51, 0 // (op 581) most significant half; uses scc

s_ashr_i64 s[50:51], s[60:61], 4 // (op 581) 

s_sub_i32 s18, s50, s1 // (op 581) 

s_mul_i32 s49, s19, s18 // (op 581) 


s_sub_u32 s18, s9, s49 // (op 581) 


s_add_u32 s9, s18, s1 // (op 581) 


s_cmp_eq_i32 s8, 0 // (op 581) 
s_mov_b32 s1, scc // (op 581) 

s_cmp_eq_i32 s1, 0 // (op 581) 
s_cselect_b32 s8, s9, s0 // (op 581) 



v_lshl_add_u32 v12, s8, 4, v16 // (op 581) 


v_mul_lo_u32 v16, v12, s26 // (op 581) 
v_add_u32 v16, v16, v13 // (op 581) 


s_mul_i32 s1, s26, s7 // (op 583) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 583) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 583) least significant
s_mul_hi_u32 s9, s26, s6 // (op 583) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 583) most significant: sum
s_add_u32 s1, s1, s9 // (op 583) most significant: sum


s_mov_b32 s60, s28 // (op 583) reg expression
s_mov_b32 s61, s29 // (op 583) reg expression
s_mov_b32 s62, s0 // (op 583) reg expression
s_mov_b32 s63, 131072 // (op 583) call()

s_mul_hi_i32 s0, s16, s12 // (op 600) 
v_lshrrev_b32 v12, 7, v1 // (op 600) 
s_add_i32 s1, s0, s16 // (op 600) 

v_and_b32 v13, 15, v1 // (op 600) 
s_ashr_i32 s0, s1, 31 // (op 600) 
v_lshl_add_u32 v17, v12, 4, v13 // (op 600) 


s_cmp_eq_i32 s12, 0 // (op 600) 
s_mov_b32 s8, scc // (op 600) 
v_and_b32 v12, 63, v1 // (op 600) 
s_cmp_eq_i32 s8, 0 // (op 600) 
s_cselect_b32 s9, 0, -1 // (op 600) 

v_lshrrev_b32 v13, 4, v12 // (op 600) 

s_and_b32 s8, s13, 31 // (op 600) 
s_lshl_b32 s18, 1, s8 // (op 600) 
s_add_i32 s19, s18, s9 // (op 600) 


s_and_b32 s9, s0, s19 // (op 600) 


s_add_i32 s0, s1, s9 // (op 600) 


s_ashr_i32 s1, s0, s8 // (op 600) 


s_lshl_b32 s0, s13, 24 // (op 600) 
s_ashr_i32 s8, s0, 31 // (op 600) 

s_xor_b32 s0, s1, s8 // (op 600) 

s_sub_i32 s1, s0, s8 // (op 600) 


s_mul_hi_i32 s0, s1, s14 // (op 600) 
s_add_i32 s8, s0, s1 // (op 600) 

s_ashr_i32 s0, s8, 31 // (op 600) 
s_cmp_eq_i32 s14, 0 // (op 600) 
s_mov_b32 s9, scc // (op 600) 
s_cmp_eq_i32 s9, 0 // (op 600) 
s_cselect_b32 s18, 0, -1 // (op 600) 

s_and_b32 s9, s15, 31 // (op 600) 
s_lshl_b32 s19, 1, s9 // (op 600) 
s_add_i32 s49, s19, s18 // (op 600) 


s_and_b32 s18, s0, s49 // (op 600) 


s_add_i32 s0, s8, s18 // (op 600) 


s_ashr_i32 s8, s0, s9 // (op 600) 


s_lshl_b32 s0, s15, 24 // (op 600) 
s_ashr_i32 s9, s0, 31 // (op 600) 

s_xor_b32 s0, s8, s9 // (op 600) 

s_sub_i32 s8, s0, s9 // (op 600) 


s_mul_i32 s0, s1, s45 // (op 600) 

s_sub_u32 s1, s16, s0 // (op 600) 

s_mul_i32 s0, s8, s45 // (op 600) 
s_add_u32 s0, s0, s1 // (op 600) 


s_mul_hi_i32 s1, s16, s11 // (op 600) 
s_add_i32 s8, s1, s16 // (op 600) 

s_ashr_i32 s1, s8, 31 // (op 600) 
s_cmp_eq_i32 s11, 0 // (op 600) 
s_mov_b32 s9, scc // (op 600) 
s_cmp_eq_i32 s9, 0 // (op 600) 
s_cselect_b32 s18, 0, -1 // (op 600) 

s_and_b32 s9, s46, 31 // (op 600) 
s_lshl_b32 s19, 1, s9 // (op 600) 
s_add_i32 s49, s19, s18 // (op 600) 


s_and_b32 s18, s1, s49 // (op 600) 


s_add_i32 s1, s8, s18 // (op 600) 


s_ashr_i32 s8, s1, s9 // (op 600) 


s_lshl_b32 s1, s46, 24 // (op 600) 
s_ashr_i32 s9, s1, 31 // (op 600) 

s_xor_b32 s1, s8, s9 // (op 600) 

s_sub_i32 s8, s1, s9 // (op 600) 


s_add_u32 s18, s4, 32 // (op 600) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 600) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 600) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 600) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 600) 

s_mul_i32 s1, s45, s18 // (op 600) 

s_mul_i32 s9, s10, s1 // (op 600) 

s_mul_i32 s1, s8, s9 // (op 600) 

s_sub_u32 s9, s16, s1 // (op 600) 

s_mov_b32 s1, s9 // (op 600) 
s_mul_hi_i32 s18, s1, s44 // (op 600) 
s_add_i32 s19, s18, s1 // (op 600) 


s_ashr_i32 s1, s19, 31 // (op 600) 
s_cmp_eq_i32 s44, 0 // (op 600) 
s_mov_b32 s18, scc // (op 600) 
s_cmp_eq_i32 s18, 0 // (op 600) 
s_cselect_b32 s49, 0, -1 // (op 600) 

s_and_b32 s18, s48, 31 // (op 600) 
s_lshl_b32 s50, 1, s18 // (op 600) 
s_add_i32 s51, s50, s49 // (op 600) 


s_and_b32 s49, s1, s51 // (op 600) 


s_add_i32 s1, s19, s49 // (op 600) 


s_ashr_i32 s19, s1, s18 // (op 600) 


s_lshl_b32 s1, s48, 24 // (op 600) 
s_ashr_i32 s18, s1, 31 // (op 600) 

s_xor_b32 s1, s19, s18 // (op 600) 

s_sub_i32 s19, s1, s18 // (op 600) 


s_mul_i32 s1, s47, s45 // (op 600) 
s_add_u32 s50, s6, 16 // (op 600) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 600) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 600) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 600) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 4 // (op 600) 

s_sub_i32 s18, s50, s1 // (op 600) 

s_mul_i32 s49, s19, s18 // (op 600) 


s_sub_u32 s18, s9, s49 // (op 600) 


s_add_u32 s9, s18, s1 // (op 600) 


s_cmp_eq_i32 s8, 0 // (op 600) 
s_mov_b32 s1, scc // (op 600) 

s_cmp_eq_i32 s1, 0 // (op 600) 
s_cselect_b32 s8, s9, s0 // (op 600) 



v_lshl_add_u32 v12, s8, 4, v17 // (op 600) 


v_mul_lo_u32 v17, v12, s26 // (op 600) 
v_add_u32 v17, v17, v13 // (op 600) 


s_mul_i32 s1, s26, s7 // (op 602) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 602) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 602) least significant
s_mul_hi_u32 s9, s26, s6 // (op 602) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 602) most significant: sum
s_add_u32 s1, s1, s9 // (op 602) most significant: sum


s_mov_b32 s60, s28 // (op 602) reg expression
s_mov_b32 s61, s29 // (op 602) reg expression
s_mov_b32 s62, s0 // (op 602) reg expression
s_mov_b32 s63, 131072 // (op 602) call()

s_mul_hi_i32 s0, s16, s12 // (op 645) 
v_lshrrev_b32 v12, 2, v1 // (op 645) 
s_add_i32 s1, s0, s16 // (op 645) 

v_and_b32 v13, -4, v1 // (op 645) 
s_ashr_i32 s0, s1, 31 // (op 645) 
v_sub_u32 v18, v1, v13 // (op 645) 

s_cmp_eq_i32 s12, 0 // (op 645) 
s_mov_b32 s8, scc // (op 645) 
s_cmp_eq_i32 s8, 0 // (op 645) 
s_cselect_b32 s9, 0, -1 // (op 645) 

s_and_b32 s8, s13, 31 // (op 645) 
s_lshl_b32 s18, 1, s8 // (op 645) 
s_add_i32 s19, s18, s9 // (op 645) 


s_and_b32 s9, s0, s19 // (op 645) 


s_add_i32 s0, s1, s9 // (op 645) 


s_ashr_i32 s1, s0, s8 // (op 645) 


s_lshl_b32 s0, s13, 24 // (op 645) 
s_ashr_i32 s8, s0, 31 // (op 645) 

s_xor_b32 s0, s1, s8 // (op 645) 

s_sub_i32 s1, s0, s8 // (op 645) 


s_mul_hi_i32 s0, s1, s14 // (op 645) 
s_add_i32 s8, s0, s1 // (op 645) 

s_ashr_i32 s0, s8, 31 // (op 645) 
s_cmp_eq_i32 s14, 0 // (op 645) 
s_mov_b32 s9, scc // (op 645) 
s_cmp_eq_i32 s9, 0 // (op 645) 
s_cselect_b32 s18, 0, -1 // (op 645) 

s_and_b32 s9, s15, 31 // (op 645) 
s_lshl_b32 s19, 1, s9 // (op 645) 
s_add_i32 s49, s19, s18 // (op 645) 


s_and_b32 s18, s0, s49 // (op 645) 


s_add_i32 s0, s8, s18 // (op 645) 


s_ashr_i32 s8, s0, s9 // (op 645) 


s_lshl_b32 s0, s15, 24 // (op 645) 
s_ashr_i32 s9, s0, 31 // (op 645) 

s_xor_b32 s0, s8, s9 // (op 645) 

s_sub_i32 s8, s0, s9 // (op 645) 


s_add_u32 s18, s4, 32 // (op 645) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 645) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 645) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 645) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 645) 

s_mul_i32 s0, s8, s18 // (op 645) 

s_sub_i32 s8, s1, s0 // (op 645) 


s_mul_hi_i32 s9, s16, s11 // (op 645) 
s_add_i32 s0, s9, s16 // (op 645) 

s_ashr_i32 s1, s0, 31 // (op 645) 
s_cmp_eq_i32 s11, 0 // (op 645) 
s_mov_b32 s9, scc // (op 645) 
s_cmp_eq_i32 s9, 0 // (op 645) 
s_cselect_b32 s49, 0, -1 // (op 645) 

s_and_b32 s9, s46, 31 // (op 645) 
s_lshl_b32 s50, 1, s9 // (op 645) 
s_add_i32 s51, s50, s49 // (op 645) 


s_and_b32 s49, s1, s51 // (op 645) 


s_add_i32 s1, s0, s49 // (op 645) 


s_ashr_i32 s0, s1, s9 // (op 645) 


s_lshl_b32 s1, s46, 24 // (op 645) 
s_ashr_i32 s9, s1, 31 // (op 645) 

s_xor_b32 s1, s0, s9 // (op 645) 

s_sub_i32 s0, s1, s9 // (op 645) 


s_mul_i32 s1, s45, s18 // (op 645) 

s_mul_i32 s9, s10, s1 // (op 645) 

s_mul_i32 s1, s0, s9 // (op 645) 

s_sub_u32 s9, s16, s1 // (op 645) 

s_mul_hi_i32 s1, s9, s44 // (op 645) 
s_add_i32 s18, s1, s9 // (op 645) 


s_ashr_i32 s1, s18, 31 // (op 645) 
s_cmp_eq_i32 s44, 0 // (op 645) 
s_mov_b32 s9, scc // (op 645) 
s_cmp_eq_i32 s9, 0 // (op 645) 
s_cselect_b32 s19, 0, -1 // (op 645) 

s_and_b32 s9, s48, 31 // (op 645) 
s_lshl_b32 s49, 1, s9 // (op 645) 
s_add_i32 s50, s49, s19 // (op 645) 


s_and_b32 s19, s1, s50 // (op 645) 


s_add_i32 s1, s18, s19 // (op 645) 


s_ashr_i32 s18, s1, s9 // (op 645) 


s_lshl_b32 s1, s48, 24 // (op 645) 
s_ashr_i32 s9, s1, 31 // (op 645) 

s_xor_b32 s1, s18, s9 // (op 645) 

s_sub_i32 s18, s1, s9 // (op 645) 


s_cmp_eq_i32 s0, 0 // (op 645) 
s_mov_b32 s1, scc // (op 645) 

s_cmp_eq_i32 s1, 0 // (op 645) 
s_cselect_b32 s0, s18, s8 // (op 645) 



v_lshl_add_u32 v13, s0, 5, v12 // (op 645) 


v_mul_lo_u32 v12, v13, s30 // (op 645) 

v_add_u32 v13, v18, v12 // (op 645) 


s_mul_i32 s1, s30, s5 // (op 647) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 647) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 647) least significant
s_mul_hi_u32 s9, s30, s4 // (op 647) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 647) most significant: sum
s_add_u32 s1, s1, s9 // (op 647) most significant: sum


s_mov_b32 s64, s32 // (op 647) reg expression
s_mov_b32 s65, s33 // (op 647) reg expression
s_mov_b32 s66, s0 // (op 647) reg expression
s_mov_b32 s67, 131072 // (op 647) call()

s_mul_hi_i32 s0, s16, s12 // (op 664) 
v_lshrrev_b32 v12, 2, v1 // (op 664) 
s_add_i32 s1, s0, s16 // (op 664) 

v_and_b32 v18, -4, v1 // (op 664) 
s_ashr_i32 s0, s1, 31 // (op 664) 
v_sub_u32 v19, v1, v18 // (op 664) 

s_cmp_eq_i32 s12, 0 // (op 664) 
s_mov_b32 s8, scc // (op 664) 
s_cmp_eq_i32 s8, 0 // (op 664) 
s_cselect_b32 s9, 0, -1 // (op 664) 

s_and_b32 s8, s13, 31 // (op 664) 
s_lshl_b32 s18, 1, s8 // (op 664) 
s_add_i32 s19, s18, s9 // (op 664) 


s_and_b32 s9, s0, s19 // (op 664) 


s_add_i32 s0, s1, s9 // (op 664) 


s_ashr_i32 s1, s0, s8 // (op 664) 


s_lshl_b32 s0, s13, 24 // (op 664) 
s_ashr_i32 s8, s0, 31 // (op 664) 

s_xor_b32 s0, s1, s8 // (op 664) 

s_sub_i32 s1, s0, s8 // (op 664) 


s_mul_hi_i32 s0, s1, s14 // (op 664) 
s_add_i32 s8, s0, s1 // (op 664) 

s_ashr_i32 s0, s8, 31 // (op 664) 
s_cmp_eq_i32 s14, 0 // (op 664) 
s_mov_b32 s9, scc // (op 664) 
s_cmp_eq_i32 s9, 0 // (op 664) 
s_cselect_b32 s18, 0, -1 // (op 664) 

s_and_b32 s9, s15, 31 // (op 664) 
s_lshl_b32 s19, 1, s9 // (op 664) 
s_add_i32 s49, s19, s18 // (op 664) 


s_and_b32 s18, s0, s49 // (op 664) 


s_add_i32 s0, s8, s18 // (op 664) 


s_ashr_i32 s8, s0, s9 // (op 664) 


s_lshl_b32 s0, s15, 24 // (op 664) 
s_ashr_i32 s9, s0, 31 // (op 664) 

s_xor_b32 s0, s8, s9 // (op 664) 

s_sub_i32 s8, s0, s9 // (op 664) 


s_add_u32 s18, s4, 32 // (op 664) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 664) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 664) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 664) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 664) 

s_mul_i32 s0, s8, s18 // (op 664) 

s_sub_i32 s8, s1, s0 // (op 664) 


s_mul_hi_i32 s9, s16, s11 // (op 664) 
s_add_i32 s0, s9, s16 // (op 664) 

s_ashr_i32 s1, s0, 31 // (op 664) 
s_cmp_eq_i32 s11, 0 // (op 664) 
s_mov_b32 s9, scc // (op 664) 
s_cmp_eq_i32 s9, 0 // (op 664) 
s_cselect_b32 s49, 0, -1 // (op 664) 

s_and_b32 s9, s46, 31 // (op 664) 
s_lshl_b32 s50, 1, s9 // (op 664) 
s_add_i32 s51, s50, s49 // (op 664) 


s_and_b32 s49, s1, s51 // (op 664) 


s_add_i32 s1, s0, s49 // (op 664) 


s_ashr_i32 s0, s1, s9 // (op 664) 


s_lshl_b32 s1, s46, 24 // (op 664) 
s_ashr_i32 s9, s1, 31 // (op 664) 

s_xor_b32 s1, s0, s9 // (op 664) 

s_sub_i32 s0, s1, s9 // (op 664) 


s_mul_i32 s1, s45, s18 // (op 664) 

s_mul_i32 s9, s10, s1 // (op 664) 

s_mul_i32 s1, s0, s9 // (op 664) 

s_sub_u32 s9, s16, s1 // (op 664) 

s_mul_hi_i32 s1, s9, s44 // (op 664) 
s_add_i32 s18, s1, s9 // (op 664) 


s_ashr_i32 s1, s18, 31 // (op 664) 
s_cmp_eq_i32 s44, 0 // (op 664) 
s_mov_b32 s9, scc // (op 664) 
s_cmp_eq_i32 s9, 0 // (op 664) 
s_cselect_b32 s19, 0, -1 // (op 664) 

s_and_b32 s9, s48, 31 // (op 664) 
s_lshl_b32 s49, 1, s9 // (op 664) 
s_add_i32 s50, s49, s19 // (op 664) 


s_and_b32 s19, s1, s50 // (op 664) 


s_add_i32 s1, s18, s19 // (op 664) 


s_ashr_i32 s18, s1, s9 // (op 664) 


s_lshl_b32 s1, s48, 24 // (op 664) 
s_ashr_i32 s9, s1, 31 // (op 664) 

s_xor_b32 s1, s18, s9 // (op 664) 

s_sub_i32 s18, s1, s9 // (op 664) 


s_cmp_eq_i32 s0, 0 // (op 664) 
s_mov_b32 s1, scc // (op 664) 

s_cmp_eq_i32 s1, 0 // (op 664) 
s_cselect_b32 s0, s18, s8 // (op 664) 



v_lshl_add_u32 v18, s0, 5, v12 // (op 664) 


v_mul_lo_u32 v12, v18, s30 // (op 664) 

v_add_u32 v18, v19, v12 // (op 664) 


s_mul_i32 s1, s30, s5 // (op 666) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 666) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 666) least significant
s_mul_hi_u32 s9, s30, s4 // (op 666) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 666) most significant: sum
s_add_u32 s1, s1, s9 // (op 666) most significant: sum


s_mov_b32 s64, s32 // (op 666) reg expression
s_mov_b32 s65, s33 // (op 666) reg expression
s_mov_b32 s66, s0 // (op 666) reg expression
s_mov_b32 s67, 131072 // (op 666) call()




buffer_load_ubyte v12, v16, s[60:63], 0 offen offset: 0 // (op 337) Load value



buffer_load_dwordx4 v[20:23], v14, s[56:59], 0 offen // (op 341) Load value
buffer_load_dwordx2 v[24:25], v14, s[56:59], 0 offen offset:16 // (op 341) Load value



buffer_load_ubyte v19, v13, s[64:67], 0 offen offset: 0 // (op 344) Load value



buffer_load_dwordx3 v[26:28], v10, s[52:55], 0 offen // (op 347) Load value



buffer_load_ubyte v29, v16, s[60:63], 0 offen offset:4 // (op 351) Load value




buffer_load_dwordx4 v[30:33], v14, s[56:59], 0 offen offset:96 // (op 355) Load value
buffer_load_dwordx2 v[34:35], v14, s[56:59], 0 offen offset:112 // (op 355) Load value




buffer_load_ubyte v14, v13, s[64:67], 0 offen offset:4 // (op 358) Load value




buffer_load_dwordx3 v[36:38], v10, s[52:55], 0 offen offset:96 // (op 361) Load value

v_and_b32 v10, -4, v1 // (op 784) 
v_sub_u32 v13, v1, v10 // (op 784) 

v_lshlrev_b32 v10, 5, v13 // (op 784) 

v_lshrrev_b32 v13, 2, v1 // (op 784) 
v_lshl_add_u32 v16, v13, 7, v10 // (op 784) 


v_mul_lo_u32 v10, v16, 6 // (op 784) 

v_lshrrev_b32 v13, 3, v10 // (op 784) 




s_waitcnt vmcnt(8)
ds_write_b128 v13, v[20:23] // (op 363) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v13, v[24:25] offset:16 // (op 363) Store local data 

v_and_b32 v10, -4, v1 // (op 837) 
v_sub_u32 v13, v1, v10 // (op 837) 

v_and_b32 v10, 4294967292, v1 // (op 837) 
v_add_u32 v16, v10, v13 // (op 837) 





s_waitcnt vmcnt(6)
ds_write_b8 v16, v19 offset:3072 // (op 364) Store local data 

v_and_b32 v10, -8, v1 // (op 731) 
v_sub_u32 v13, v1, v10 // (op 731) 

v_lshlrev_b32 v10, 4, v13 // (op 731) 

v_lshrrev_b32 v13, 3, v1 // (op 731) 
v_lshl_add_u32 v16, v13, 7, v10 // (op 731) 


v_mul_lo_u32 v10, v16, 6 // (op 731) 

v_lshrrev_b32 v13, 3, v10 // (op 731) 




s_waitcnt vmcnt(5)
ds_write_b96 v13, v[26:28] offset:3200 // (op 365) Store local data 

v_and_b32 v10, -8, v1 // (op 702) 
v_sub_u32 v13, v1, v10 // (op 702) 

v_lshlrev_b32 v10, 4, v13 // (op 702) 

v_lshrrev_b32 v13, 3, v1 // (op 702) 
v_lshl_add_u32 v16, v13, 7, v10 // (op 702) 


v_mul_lo_u32 v10, v16, 6 // (op 702) 

v_lshrrev_b32 v13, 3, v10 // (op 702) 

v_and_b32 v10, -4, v1 // (op 755) 
v_sub_u32 v16, v1, v10 // (op 755) 

v_lshlrev_b32 v10, 5, v16 // (op 755) 

v_lshrrev_b32 v16, 2, v1 // (op 755) 
v_lshl_add_u32 v39, v16, 7, v10 // (op 755) 


v_mul_lo_u32 v10, v39, 6 // (op 755) 

v_lshrrev_b32 v16, 3, v10 // (op 755) 

v_and_b32 v10, -4, v1 // (op 808) 
v_sub_u32 v39, v1, v10 // (op 808) 

v_and_b32 v10, 4294967292, v1 // (op 808) 
v_add_u32 v40, v10, v39 // (op 808) 


v_and_b32 v10, -8, v1 // (op 895) 
v_sub_u32 v39, v1, v10 // (op 895) 

v_lshlrev_b32 v10, 4, v39 // (op 895) 

v_lshrrev_b32 v39, 3, v1 // (op 895) 
v_lshl_add_u32 v41, v39, 7, v10 // (op 895) 


v_mul_lo_u32 v10, v41, 6 // (op 895) 

v_lshrrev_b32 v39, 3, v10 // (op 895) 

v_and_b32 v10, -4, v1 // (op 919) 
v_sub_u32 v41, v1, v10 // (op 919) 

v_lshlrev_b32 v10, 5, v41 // (op 919) 

v_lshrrev_b32 v41, 2, v1 // (op 919) 
v_lshl_add_u32 v42, v41, 7, v10 // (op 919) 


v_mul_lo_u32 v10, v42, 6 // (op 919) 

v_lshrrev_b32 v41, 3, v10 // (op 919) 

v_and_b32 v10, -4, v1 // (op 943) 
v_sub_u32 v42, v1, v10 // (op 943) 

v_and_b32 v10, 4294967292, v1 // (op 943) 
v_add_u32 v43, v10, v42 // (op 943) 


s_mov_b32 s0, 0 // (op 55) call()
s_ashr_i64 s[8:9], s[34:35], 7 // (op 54) 
s_mov_b32 s1, s8 // (op 54) convert

s_and_b32 s8, s1, 4294967294 // (op 54) 

s_cmp_lt_u32 s0, s8 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54 if false)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54:
s_add_u32 s0, s0, 2 // (op 56) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 988) 



ds_read_b128 v[44:47], v3 offset:3200 // (op 65) Load local data 
ds_read_b64 v[48:49], v3 offset:3216 // (op 65) Load local data 
s_barrier  // (op 1001) 



ds_read_b128 v[50:53], v4 // (op 70) Load local data 
ds_read_b64 v[54:55], v4 offset:16 // (op 70) Load local data 
s_barrier  // (op 1015) 



ds_read_u8 v10, v0 offset:3072 // (op 79) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:49], v[50:55], a[0:3], v12, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 82) 





v_mov_b32 v10, v17 // (op 75) 

buffer_load_ubyte v12, v10, s[60:63], 0 offen offset:8 // (op 75) Load value


v_mov_b32 v42, v15 // (op 10) 

buffer_load_dwordx4 v[20:23], v42, s[56:59], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[24:25], v42, s[56:59], 0 offen offset:208 // (op 10) Load value


v_mov_b32 v44, v18 // (op 21) 

buffer_load_ubyte v19, v44, s[64:67], 0 offen offset:8 // (op 21) Load value


v_mov_b32 v45, v11 // (op 4) 

buffer_load_dwordx3 v[26:28], v45, s[52:55], 0 offen offset:192 // (op 4) Load value
s_barrier  // (op 1038) 



s_waitcnt vmcnt(8)
ds_write_b128 v41, v[30:33] offset:4736 // (op 235) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v41, v[34:35] offset:4752 // (op 235) Store local data 
s_barrier  // (op 1044) 



s_waitcnt vmcnt(6)
ds_write_b8 v43, v14 offset:7808 // (op 236) Store local data 
s_barrier  // (op 1032) 



s_waitcnt vmcnt(5)
ds_write_b96 v39, v[36:38] offset:7936 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1029) 



ds_read_b128 v[46:49], v6 offset:7936 // (op 227) Load local data 
ds_read_b64 v[50:51], v6 offset:7952 // (op 227) Load local data 
s_barrier  // (op 1035) 
s_barrier  // (op 1095) 



ds_read_b128 v[52:55], v7 offset:4736 // (op 229) Load local data 
ds_read_b64 v[56:57], v7 offset:4752 // (op 229) Load local data 
s_barrier  // (op 1041) 
s_barrier  // (op 1101) 



ds_read_u8 v58, v5 offset:7808 // (op 233) Load local data 
s_barrier  // (op 1109) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:51], v[52:57], a[0:3], v29, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 234) 






buffer_load_ubyte v29, v10, s[60:63], 0 offen offset:12 // (op 231) Load value




v_add_u32 v17, 8, v17 // (op 592) 
buffer_load_dwordx4 v[30:33], v42, s[56:59], 0 offen offset:288 // (op 237) Load value
buffer_load_dwordx2 v[34:35], v42, s[56:59], 0 offen offset:304 // (op 237) Load value




v_add_u32 v15, 192, v15 // (op 528) 
buffer_load_ubyte v14, v44, s[64:67], 0 offen offset:12 // (op 238) Load value




v_add_u32 v18, 8, v18 // (op 656) 
buffer_load_dwordx3 v[36:38], v45, s[52:55], 0 offen offset:288 // (op 224) Load value

s_barrier  // (op 1004) 
v_add_u32 v11, 192, v11 // (op 464) 



s_waitcnt vmcnt(8)
ds_write_b128 v16, v[20:23] // (op 13) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v16, v[24:25] offset:16 // (op 13) Store local data 
s_barrier  // (op 1018) 



s_waitcnt vmcnt(6)
ds_write_b8 v40, v19 offset:3072 // (op 26) Store local data 
s_barrier  // (op 991) 



s_waitcnt vmcnt(5)
ds_write_b96 v13, v[26:28] offset:3200 // (op 6) Store local data 
s_ashr_i64 s[8:9], s[34:35], 7 // (op 54) 
s_mov_b32 s1, s8 // (op 54) convert

s_and_b32 s8, s1, 4294967294 // (op 54) 

s_cmp_lt_u32 s0, s8 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54 if true)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54:















s_mul_hi_i32 s0, s16, s12 // (op 427) 
v_lshrrev_b32 v10, 3, v1 // (op 427) 
s_add_i32 s1, s0, s16 // (op 427) 

v_and_b32 v11, -8, v1 // (op 427) 
s_ashr_i32 s0, s1, 31 // (op 427) 
v_sub_u32 v12, v1, v11 // (op 427) 

s_cmp_eq_i32 s12, 0 // (op 427) 
s_mov_b32 s8, scc // (op 427) 
v_lshlrev_b32 v11, 4, v12 // (op 427) 

s_cmp_eq_i32 s8, 0 // (op 427) 
s_cselect_b32 s9, 0, -1 // (op 427) 

s_and_b32 s8, s13, 31 // (op 427) 
s_lshl_b32 s18, 1, s8 // (op 427) 
s_add_i32 s19, s18, s9 // (op 427) 


s_and_b32 s9, s0, s19 // (op 427) 


s_add_i32 s0, s1, s9 // (op 427) 


s_ashr_i32 s1, s0, s8 // (op 427) 


s_lshl_b32 s0, s13, 24 // (op 427) 
s_ashr_i32 s8, s0, 31 // (op 427) 

s_xor_b32 s0, s1, s8 // (op 427) 

s_sub_i32 s1, s0, s8 // (op 427) 


s_mul_hi_i32 s0, s1, s14 // (op 427) 
s_add_i32 s8, s0, s1 // (op 427) 

s_ashr_i32 s0, s8, 31 // (op 427) 
s_cmp_eq_i32 s14, 0 // (op 427) 
s_mov_b32 s9, scc // (op 427) 
s_cmp_eq_i32 s9, 0 // (op 427) 
s_cselect_b32 s18, 0, -1 // (op 427) 

s_and_b32 s9, s15, 31 // (op 427) 
s_lshl_b32 s19, 1, s9 // (op 427) 
s_add_i32 s49, s19, s18 // (op 427) 


s_and_b32 s18, s0, s49 // (op 427) 


s_add_i32 s0, s8, s18 // (op 427) 


s_ashr_i32 s8, s0, s9 // (op 427) 


s_lshl_b32 s0, s15, 24 // (op 427) 
s_ashr_i32 s9, s0, 31 // (op 427) 

s_xor_b32 s0, s8, s9 // (op 427) 

s_sub_i32 s8, s0, s9 // (op 427) 


s_mul_i32 s0, s1, s45 // (op 427) 

s_sub_u32 s1, s16, s0 // (op 427) 

s_mul_i32 s0, s8, s45 // (op 427) 
s_add_u32 s0, s0, s1 // (op 427) 


s_mul_hi_i32 s1, s16, s11 // (op 427) 
s_add_i32 s8, s1, s16 // (op 427) 

s_ashr_i32 s1, s8, 31 // (op 427) 
s_cmp_eq_i32 s11, 0 // (op 427) 
s_mov_b32 s9, scc // (op 427) 
s_cmp_eq_i32 s9, 0 // (op 427) 
s_cselect_b32 s18, 0, -1 // (op 427) 

s_and_b32 s9, s46, 31 // (op 427) 
s_lshl_b32 s19, 1, s9 // (op 427) 
s_add_i32 s49, s19, s18 // (op 427) 


s_and_b32 s18, s1, s49 // (op 427) 


s_add_i32 s1, s8, s18 // (op 427) 


s_ashr_i32 s8, s1, s9 // (op 427) 


s_lshl_b32 s1, s46, 24 // (op 427) 
s_ashr_i32 s9, s1, 31 // (op 427) 

s_xor_b32 s1, s8, s9 // (op 427) 

s_sub_i32 s8, s1, s9 // (op 427) 


s_add_u32 s18, s4, 32 // (op 427) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 427) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 427) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 427) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 427) 

s_mul_i32 s1, s45, s18 // (op 427) 

s_mul_i32 s9, s10, s1 // (op 427) 

s_mul_i32 s1, s8, s9 // (op 427) 

s_sub_u32 s9, s16, s1 // (op 427) 

s_mov_b32 s1, s9 // (op 427) 
s_mul_hi_i32 s18, s1, s44 // (op 427) 
s_add_i32 s19, s18, s1 // (op 427) 


s_ashr_i32 s1, s19, 31 // (op 427) 
s_cmp_eq_i32 s44, 0 // (op 427) 
s_mov_b32 s18, scc // (op 427) 
s_cmp_eq_i32 s18, 0 // (op 427) 
s_cselect_b32 s49, 0, -1 // (op 427) 

s_and_b32 s18, s48, 31 // (op 427) 
s_lshl_b32 s50, 1, s18 // (op 427) 
s_add_i32 s51, s50, s49 // (op 427) 


s_and_b32 s49, s1, s51 // (op 427) 


s_add_i32 s1, s19, s49 // (op 427) 


s_ashr_i32 s19, s1, s18 // (op 427) 


s_lshl_b32 s1, s48, 24 // (op 427) 
s_ashr_i32 s18, s1, 31 // (op 427) 

s_xor_b32 s1, s19, s18 // (op 427) 

s_sub_i32 s19, s1, s18 // (op 427) 


s_mul_i32 s1, s47, s45 // (op 427) 
s_add_u32 s50, s6, 16 // (op 427) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 427) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 427) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 427) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 427) 

s_sub_i32 s18, s50, s1 // (op 427) 

s_mul_i32 s49, s19, s18 // (op 427) 


s_sub_u32 s18, s9, s49 // (op 427) 


s_add_u32 s9, s18, s1 // (op 427) 


s_cmp_eq_i32 s8, 0 // (op 427) 
s_mov_b32 s1, scc // (op 427) 

s_cmp_eq_i32 s1, 0 // (op 427) 
s_cselect_b32 s8, s9, s0 // (op 427) 



v_lshl_add_u32 v12, s8, 4, v10 // (op 427) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 427) 
s_lshr_b32 s8, s0, 1 // (op 427) 

v_lshl_add_u32 v10, s8, 8, v11 // (op 427) 


v_mov_b32 v17, 0 // (op 427) convert
v_mov_b32 v16, v10 // (op 427) convert

v_mul_lo_u32 v11, v12, s3 // (op 427) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 427) most significant: high of low * low
v_mul_lo_u32 v10, v12, s2 // (op 427) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 427) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v16 // (op 427) least significant half
v_addc_co_u32 v11, s[0:1], v11, v17, s[0:1] // (op 427) most significant half



v_mul_lo_u32 v13, v11, 6 // (op 427) most significant: low of high * low
v_mul_hi_u32 v15, v10, 6 // (op 427) most significant: high of low * low
v_mul_lo_u32 v12, v10, 6 // (op 427) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 427) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 427) 

v_mov_b32 v12, v10 // (op 427) convert

s_mul_i32 s1, s7, 6 // (op 429) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 429) least significant
s_mul_hi_u32 s8, s6, 6 // (op 429) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 429) most significant: sum

s_mul_i32 s9, s2, s1 // (op 429) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 429) most significant: low of high * low
s_mul_i32 s8, s2, s0 // (op 429) least significant
s_mul_hi_u32 s19, s2, s0 // (op 429) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 429) most significant: sum
s_add_u32 s9, s9, s19 // (op 429) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 429) 

s_mov_b32 s52, s20 // (op 429) reg expression
s_mov_b32 s53, s21 // (op 429) reg expression
s_mov_b32 s54, s0 // (op 429) reg expression
s_mov_b32 s55, 131072 // (op 429) call()

s_mul_hi_i32 s0, s16, s12 // (op 494) 
v_lshrrev_b32 v13, 2, v1 // (op 494) 
s_add_i32 s1, s0, s16 // (op 494) 

v_and_b32 v10, -4, v1 // (op 494) 
s_ashr_i32 s0, s1, 31 // (op 494) 
v_sub_u32 v11, v1, v10 // (op 494) 

s_cmp_eq_i32 s12, 0 // (op 494) 
s_mov_b32 s8, scc // (op 494) 
v_lshlrev_b32 v10, 5, v11 // (op 494) 

s_cmp_eq_i32 s8, 0 // (op 494) 
s_cselect_b32 s9, 0, -1 // (op 494) 

s_and_b32 s8, s13, 31 // (op 494) 
s_lshl_b32 s18, 1, s8 // (op 494) 
s_add_i32 s19, s18, s9 // (op 494) 


s_and_b32 s9, s0, s19 // (op 494) 


s_add_i32 s0, s1, s9 // (op 494) 


s_ashr_i32 s1, s0, s8 // (op 494) 


s_lshl_b32 s0, s13, 24 // (op 494) 
s_ashr_i32 s8, s0, 31 // (op 494) 

s_xor_b32 s0, s1, s8 // (op 494) 

s_sub_i32 s1, s0, s8 // (op 494) 


s_mul_hi_i32 s0, s1, s14 // (op 494) 
s_add_i32 s8, s0, s1 // (op 494) 

s_ashr_i32 s0, s8, 31 // (op 494) 
s_cmp_eq_i32 s14, 0 // (op 494) 
s_mov_b32 s9, scc // (op 494) 
s_cmp_eq_i32 s9, 0 // (op 494) 
s_cselect_b32 s18, 0, -1 // (op 494) 

s_and_b32 s9, s15, 31 // (op 494) 
s_lshl_b32 s19, 1, s9 // (op 494) 
s_add_i32 s49, s19, s18 // (op 494) 


s_and_b32 s18, s0, s49 // (op 494) 


s_add_i32 s0, s8, s18 // (op 494) 


s_ashr_i32 s8, s0, s9 // (op 494) 


s_lshl_b32 s0, s15, 24 // (op 494) 
s_ashr_i32 s9, s0, 31 // (op 494) 

s_xor_b32 s0, s8, s9 // (op 494) 

s_sub_i32 s8, s0, s9 // (op 494) 


s_add_u32 s18, s4, 32 // (op 494) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 494) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 494) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 494) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 494) 

s_mul_i32 s0, s8, s18 // (op 494) 

s_sub_i32 s8, s1, s0 // (op 494) 


s_mul_hi_i32 s9, s16, s11 // (op 494) 
s_add_i32 s0, s9, s16 // (op 494) 

s_ashr_i32 s1, s0, 31 // (op 494) 
s_cmp_eq_i32 s11, 0 // (op 494) 
s_mov_b32 s9, scc // (op 494) 
s_cmp_eq_i32 s9, 0 // (op 494) 
s_cselect_b32 s49, 0, -1 // (op 494) 

s_and_b32 s9, s46, 31 // (op 494) 
s_lshl_b32 s50, 1, s9 // (op 494) 
s_add_i32 s51, s50, s49 // (op 494) 


s_and_b32 s49, s1, s51 // (op 494) 


s_add_i32 s1, s0, s49 // (op 494) 


s_ashr_i32 s0, s1, s9 // (op 494) 


s_lshl_b32 s1, s46, 24 // (op 494) 
s_ashr_i32 s9, s1, 31 // (op 494) 

s_xor_b32 s1, s0, s9 // (op 494) 

s_sub_i32 s0, s1, s9 // (op 494) 


s_mul_i32 s1, s45, s18 // (op 494) 

s_mul_i32 s9, s10, s1 // (op 494) 

s_mul_i32 s1, s0, s9 // (op 494) 

s_sub_u32 s9, s16, s1 // (op 494) 

s_mul_hi_i32 s1, s9, s44 // (op 494) 
s_add_i32 s18, s1, s9 // (op 494) 


s_ashr_i32 s1, s18, 31 // (op 494) 
s_cmp_eq_i32 s44, 0 // (op 494) 
s_mov_b32 s9, scc // (op 494) 
s_cmp_eq_i32 s9, 0 // (op 494) 
s_cselect_b32 s19, 0, -1 // (op 494) 

s_and_b32 s9, s48, 31 // (op 494) 
s_lshl_b32 s49, 1, s9 // (op 494) 
s_add_i32 s50, s49, s19 // (op 494) 


s_and_b32 s19, s1, s50 // (op 494) 


s_add_i32 s1, s18, s19 // (op 494) 


s_ashr_i32 s18, s1, s9 // (op 494) 


s_lshl_b32 s1, s48, 24 // (op 494) 
s_ashr_i32 s9, s1, 31 // (op 494) 

s_xor_b32 s1, s18, s9 // (op 494) 

s_sub_i32 s18, s1, s9 // (op 494) 


s_cmp_eq_i32 s0, 0 // (op 494) 
s_mov_b32 s1, scc // (op 494) 

s_cmp_eq_i32 s1, 0 // (op 494) 
s_cselect_b32 s0, s18, s8 // (op 494) 



v_lshl_add_u32 v11, s0, 5, v13 // (op 494) 


v_mul_lo_u32 v17, v11, s23 // (op 494) most significant: low of low * high
v_mul_hi_u32 v13, v11, s22 // (op 494) most significant: high of low * low
v_mul_lo_u32 v16, v11, s22 // (op 494) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 494) most significant: sum


s_ashr_i64 s[0:1], s[34:35], 7 // (op 494) 
s_lshr_b32 s8, s0, 1 // (op 494) 

v_lshl_add_u32 v11, s8, 8, v10 // (op 494) 


v_mov_b32 v19, 0 // (op 494) convert
v_mov_b32 v18, v11 // (op 494) convert

v_add_co_u32 v10, s[0:1], v18, v16 // (op 494) least significant half
v_addc_co_u32 v11, s[0:1], v19, v17, s[0:1] // (op 494) most significant half



v_mul_lo_u32 v17, v11, 6 // (op 494) most significant: low of high * low
v_mul_hi_u32 v13, v10, 6 // (op 494) most significant: high of low * low
v_mul_lo_u32 v16, v10, 6 // (op 494) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 494) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[16:17] // (op 494) 

v_mov_b32 v13, v10 // (op 494) convert

s_mul_i32 s1, s5, 6 // (op 496) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 496) least significant
s_mul_hi_u32 s8, s4, 6 // (op 496) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 496) most significant: sum

s_mul_i32 s9, s22, s1 // (op 496) most significant: low of low * high
s_mul_i32 s18, s23, s0 // (op 496) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 496) least significant
s_mul_hi_u32 s19, s22, s0 // (op 496) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 496) most significant: sum
s_add_u32 s9, s9, s19 // (op 496) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 496) 

s_mov_b32 s56, s24 // (op 496) reg expression
s_mov_b32 s57, s25 // (op 496) reg expression
s_mov_b32 s58, s0 // (op 496) reg expression
s_mov_b32 s59, 131072 // (op 496) call()

s_mul_hi_i32 s0, s16, s12 // (op 558) 
v_lshrrev_b32 v10, 7, v1 // (op 558) 
s_add_i32 s1, s0, s16 // (op 558) 

v_and_b32 v11, 15, v1 // (op 558) 
s_ashr_i32 s0, s1, 31 // (op 558) 
v_lshl_add_u32 v15, v10, 4, v11 // (op 558) 


s_cmp_eq_i32 s12, 0 // (op 558) 
s_mov_b32 s8, scc // (op 558) 
v_and_b32 v10, 63, v1 // (op 558) 
s_cmp_eq_i32 s8, 0 // (op 558) 
s_cselect_b32 s9, 0, -1 // (op 558) 

v_lshrrev_b32 v11, 4, v10 // (op 558) 

s_and_b32 s8, s13, 31 // (op 558) 
s_lshl_b32 s18, 1, s8 // (op 558) 
s_add_i32 s19, s18, s9 // (op 558) 


s_and_b32 s9, s0, s19 // (op 558) 


s_add_i32 s0, s1, s9 // (op 558) 


s_ashr_i32 s1, s0, s8 // (op 558) 


s_lshl_b32 s0, s13, 24 // (op 558) 
s_ashr_i32 s8, s0, 31 // (op 558) 

s_xor_b32 s0, s1, s8 // (op 558) 

s_sub_i32 s1, s0, s8 // (op 558) 


s_mul_hi_i32 s0, s1, s14 // (op 558) 
s_add_i32 s8, s0, s1 // (op 558) 

s_ashr_i32 s0, s8, 31 // (op 558) 
s_cmp_eq_i32 s14, 0 // (op 558) 
s_mov_b32 s9, scc // (op 558) 
s_cmp_eq_i32 s9, 0 // (op 558) 
s_cselect_b32 s18, 0, -1 // (op 558) 

s_and_b32 s9, s15, 31 // (op 558) 
s_lshl_b32 s19, 1, s9 // (op 558) 
s_add_i32 s49, s19, s18 // (op 558) 


s_and_b32 s18, s0, s49 // (op 558) 


s_add_i32 s0, s8, s18 // (op 558) 


s_ashr_i32 s8, s0, s9 // (op 558) 


s_lshl_b32 s0, s15, 24 // (op 558) 
s_ashr_i32 s9, s0, 31 // (op 558) 

s_xor_b32 s0, s8, s9 // (op 558) 

s_sub_i32 s8, s0, s9 // (op 558) 


s_mul_i32 s0, s1, s45 // (op 558) 

s_sub_u32 s1, s16, s0 // (op 558) 

s_mul_i32 s0, s8, s45 // (op 558) 
s_add_u32 s0, s0, s1 // (op 558) 


s_mul_hi_i32 s1, s16, s11 // (op 558) 
s_add_i32 s8, s1, s16 // (op 558) 

s_ashr_i32 s1, s8, 31 // (op 558) 
s_cmp_eq_i32 s11, 0 // (op 558) 
s_mov_b32 s9, scc // (op 558) 
s_cmp_eq_i32 s9, 0 // (op 558) 
s_cselect_b32 s18, 0, -1 // (op 558) 

s_and_b32 s9, s46, 31 // (op 558) 
s_lshl_b32 s19, 1, s9 // (op 558) 
s_add_i32 s49, s19, s18 // (op 558) 


s_and_b32 s18, s1, s49 // (op 558) 


s_add_i32 s1, s8, s18 // (op 558) 


s_ashr_i32 s8, s1, s9 // (op 558) 


s_lshl_b32 s1, s46, 24 // (op 558) 
s_ashr_i32 s9, s1, 31 // (op 558) 

s_xor_b32 s1, s8, s9 // (op 558) 

s_sub_i32 s8, s1, s9 // (op 558) 


s_add_u32 s18, s4, 32 // (op 558) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 558) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 558) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 558) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 558) 

s_mul_i32 s1, s45, s18 // (op 558) 

s_mul_i32 s9, s10, s1 // (op 558) 

s_mul_i32 s1, s8, s9 // (op 558) 

s_sub_u32 s9, s16, s1 // (op 558) 

s_mov_b32 s1, s9 // (op 558) 
s_mul_hi_i32 s18, s1, s44 // (op 558) 
s_add_i32 s19, s18, s1 // (op 558) 


s_ashr_i32 s1, s19, 31 // (op 558) 
s_cmp_eq_i32 s44, 0 // (op 558) 
s_mov_b32 s18, scc // (op 558) 
s_cmp_eq_i32 s18, 0 // (op 558) 
s_cselect_b32 s49, 0, -1 // (op 558) 

s_and_b32 s18, s48, 31 // (op 558) 
s_lshl_b32 s50, 1, s18 // (op 558) 
s_add_i32 s51, s50, s49 // (op 558) 


s_and_b32 s49, s1, s51 // (op 558) 


s_add_i32 s1, s19, s49 // (op 558) 


s_ashr_i32 s19, s1, s18 // (op 558) 


s_lshl_b32 s1, s48, 24 // (op 558) 
s_ashr_i32 s18, s1, 31 // (op 558) 

s_xor_b32 s1, s19, s18 // (op 558) 

s_sub_i32 s19, s1, s18 // (op 558) 


s_mul_i32 s1, s47, s45 // (op 558) 
s_add_u32 s50, s6, 16 // (op 558) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 558) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 558) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 558) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 558) 

s_sub_i32 s18, s50, s1 // (op 558) 

s_mul_i32 s49, s19, s18 // (op 558) 


s_sub_u32 s18, s9, s49 // (op 558) 


s_add_u32 s9, s18, s1 // (op 558) 


s_cmp_eq_i32 s8, 0 // (op 558) 
s_mov_b32 s1, scc // (op 558) 

s_cmp_eq_i32 s1, 0 // (op 558) 
s_cselect_b32 s8, s9, s0 // (op 558) 



v_lshl_add_u32 v10, s8, 4, v15 // (op 558) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 558) 
s_lshr_b32 s8, s0, 1 // (op 558) 

v_lshl_add_u32 v15, s8, 3, v11 // (op 558) 


v_mul_lo_u32 v11, v10, s26 // (op 558) 
v_add_u32 v11, v11, v15 // (op 558) 


s_mul_i32 s1, s26, s7 // (op 560) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 560) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 560) least significant
s_mul_hi_u32 s9, s26, s6 // (op 560) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 560) most significant: sum
s_add_u32 s1, s1, s9 // (op 560) most significant: sum


s_mov_b32 s60, s28 // (op 560) reg expression
s_mov_b32 s61, s29 // (op 560) reg expression
s_mov_b32 s62, s0 // (op 560) reg expression
s_mov_b32 s63, 131072 // (op 560) call()

s_mul_hi_i32 s0, s16, s12 // (op 622) 
v_lshrrev_b32 v10, 2, v1 // (op 622) 
s_add_i32 s1, s0, s16 // (op 622) 

v_and_b32 v15, -4, v1 // (op 622) 
s_ashr_i32 s0, s1, 31 // (op 622) 
v_sub_u32 v16, v1, v15 // (op 622) 

s_cmp_eq_i32 s12, 0 // (op 622) 
s_mov_b32 s8, scc // (op 622) 
s_cmp_eq_i32 s8, 0 // (op 622) 
s_cselect_b32 s9, 0, -1 // (op 622) 

s_and_b32 s8, s13, 31 // (op 622) 
s_lshl_b32 s18, 1, s8 // (op 622) 
s_add_i32 s19, s18, s9 // (op 622) 


s_and_b32 s9, s0, s19 // (op 622) 


s_add_i32 s0, s1, s9 // (op 622) 


s_ashr_i32 s1, s0, s8 // (op 622) 


s_lshl_b32 s0, s13, 24 // (op 622) 
s_ashr_i32 s8, s0, 31 // (op 622) 

s_xor_b32 s0, s1, s8 // (op 622) 

s_sub_i32 s1, s0, s8 // (op 622) 


s_mul_hi_i32 s0, s1, s14 // (op 622) 
s_add_i32 s8, s0, s1 // (op 622) 

s_ashr_i32 s0, s8, 31 // (op 622) 
s_cmp_eq_i32 s14, 0 // (op 622) 
s_mov_b32 s9, scc // (op 622) 
s_cmp_eq_i32 s9, 0 // (op 622) 
s_cselect_b32 s18, 0, -1 // (op 622) 

s_and_b32 s9, s15, 31 // (op 622) 
s_lshl_b32 s19, 1, s9 // (op 622) 
s_add_i32 s49, s19, s18 // (op 622) 


s_and_b32 s18, s0, s49 // (op 622) 


s_add_i32 s0, s8, s18 // (op 622) 


s_ashr_i32 s8, s0, s9 // (op 622) 


s_lshl_b32 s0, s15, 24 // (op 622) 
s_ashr_i32 s9, s0, 31 // (op 622) 

s_xor_b32 s0, s8, s9 // (op 622) 

s_sub_i32 s8, s0, s9 // (op 622) 


s_add_u32 s18, s4, 32 // (op 622) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 622) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 622) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 622) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 622) 

s_mul_i32 s0, s8, s18 // (op 622) 

s_sub_i32 s8, s1, s0 // (op 622) 


s_mul_hi_i32 s9, s16, s11 // (op 622) 
s_add_i32 s0, s9, s16 // (op 622) 

s_ashr_i32 s1, s0, 31 // (op 622) 
s_cmp_eq_i32 s11, 0 // (op 622) 
s_mov_b32 s9, scc // (op 622) 
s_cmp_eq_i32 s9, 0 // (op 622) 
s_cselect_b32 s49, 0, -1 // (op 622) 

s_and_b32 s9, s46, 31 // (op 622) 
s_lshl_b32 s50, 1, s9 // (op 622) 
s_add_i32 s51, s50, s49 // (op 622) 


s_and_b32 s49, s1, s51 // (op 622) 


s_add_i32 s1, s0, s49 // (op 622) 


s_ashr_i32 s0, s1, s9 // (op 622) 


s_lshl_b32 s1, s46, 24 // (op 622) 
s_ashr_i32 s9, s1, 31 // (op 622) 

s_xor_b32 s1, s0, s9 // (op 622) 

s_sub_i32 s0, s1, s9 // (op 622) 


s_mul_i32 s1, s45, s18 // (op 622) 

s_mul_i32 s9, s10, s1 // (op 622) 

s_mul_i32 s1, s0, s9 // (op 622) 

s_sub_u32 s9, s16, s1 // (op 622) 

s_mul_hi_i32 s1, s9, s44 // (op 622) 
s_add_i32 s18, s1, s9 // (op 622) 


s_ashr_i32 s1, s18, 31 // (op 622) 
s_cmp_eq_i32 s44, 0 // (op 622) 
s_mov_b32 s9, scc // (op 622) 
s_cmp_eq_i32 s9, 0 // (op 622) 
s_cselect_b32 s19, 0, -1 // (op 622) 

s_and_b32 s9, s48, 31 // (op 622) 
s_lshl_b32 s49, 1, s9 // (op 622) 
s_add_i32 s50, s49, s19 // (op 622) 


s_and_b32 s19, s1, s50 // (op 622) 


s_add_i32 s1, s18, s19 // (op 622) 


s_ashr_i32 s18, s1, s9 // (op 622) 


s_lshl_b32 s1, s48, 24 // (op 622) 
s_ashr_i32 s9, s1, 31 // (op 622) 

s_xor_b32 s1, s18, s9 // (op 622) 

s_sub_i32 s18, s1, s9 // (op 622) 


s_cmp_eq_i32 s0, 0 // (op 622) 
s_mov_b32 s1, scc // (op 622) 

s_cmp_eq_i32 s1, 0 // (op 622) 
s_cselect_b32 s0, s18, s8 // (op 622) 



v_lshl_add_u32 v15, s0, 5, v10 // (op 622) 


v_mul_lo_u32 v10, v15, s30 // (op 622) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 622) 
s_lshr_b32 s8, s0, 1 // (op 622) 

v_lshl_add_u32 v15, s8, 3, v16 // (op 622) 


v_add_u32 v16, v15, v10 // (op 622) 


s_mul_i32 s1, s30, s5 // (op 624) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 624) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 624) least significant
s_mul_hi_u32 s9, s30, s4 // (op 624) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 624) most significant: sum
s_add_u32 s1, s1, s9 // (op 624) most significant: sum


s_mov_b32 s64, s32 // (op 624) reg expression
s_mov_b32 s65, s33 // (op 624) reg expression
s_mov_b32 s66, s0 // (op 624) reg expression
s_mov_b32 s67, 131072 // (op 624) call()

v_and_b32 v10, -8, v1 // (op 715) 
v_sub_u32 v15, v1, v10 // (op 715) 

v_lshlrev_b32 v10, 4, v15 // (op 715) 

v_lshrrev_b32 v15, 3, v1 // (op 715) 
v_lshl_add_u32 v17, v15, 7, v10 // (op 715) 


v_mul_lo_u32 v10, v17, 6 // (op 715) 

v_lshrrev_b32 v15, 3, v10 // (op 715) 

v_and_b32 v10, -4, v1 // (op 768) 
v_sub_u32 v17, v1, v10 // (op 768) 

v_lshlrev_b32 v10, 5, v17 // (op 768) 

v_lshrrev_b32 v17, 2, v1 // (op 768) 
v_lshl_add_u32 v18, v17, 7, v10 // (op 768) 


v_mul_lo_u32 v10, v18, 6 // (op 768) 

v_lshrrev_b32 v17, 3, v10 // (op 768) 

v_and_b32 v10, -4, v1 // (op 821) 
v_sub_u32 v18, v1, v10 // (op 821) 

v_and_b32 v10, 4294967292, v1 // (op 821) 
v_add_u32 v19, v10, v18 // (op 821) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 162) 
s_and_b32 s8, s0, 4294967294 // (op 162) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 161) 
s_mov_b32 s9, s0 // (op 161) convert

s_cmp_lt_u32 s8, s9 // (op 161) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161:


s_add_u32 s8, s8, 1 // (op 163) 
v_mov_b32 v10, v12 // (op 166) 

buffer_load_dwordx3 v[20:22], v10, s[52:55], 0 offen // (op 166) Load value

v_add_u32 v12, 96, v12 // (op 417) 


s_barrier  // (op 994) 
v_mov_b32 v10, v13 // (op 179) 




buffer_load_dwordx4 v[24:27], v10, s[56:59], 0 offen // (op 179) Load value
buffer_load_dwordx2 v[28:29], v10, s[56:59], 0 offen offset:16 // (op 179) Load value

v_add_u32 v13, 96, v13 // (op 486) 
s_waitcnt vmcnt(2)
ds_write_b96 v15, v[20:22] offset:3200 // (op 167) Store local data 



v_mov_b32 v10, v16 // (op 180) 

s_barrier  // (op 1007) 
buffer_load_ubyte v14, v10, s[64:67], 0 offen offset: 0 // (op 180) Load value




v_add_u32 v16, 4, v16 // (op 614) 
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[24:27] // (op 177) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 997) 
s_waitcnt vmcnt(1)
ds_write_b64 v17, v[28:29] offset:16 // (op 177) Store local data 




ds_read_b128 v[20:23], v3 offset:3200 // (op 169) Load local data 
s_barrier  // (op 1021) 
ds_read_b64 v[24:25], v3 offset:3216 // (op 169) Load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 1011) 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v14 offset:3072 // (op 178) Store local data 




ds_read_b128 v[26:29], v4 // (op 171) Load local data 
ds_read_b64 v[30:31], v4 offset:16 // (op 171) Load local data 


v_mov_b32 v10, v11 // (op 173) 

buffer_load_ubyte v14, v10, s[60:63], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v11, 4, v11 // (op 550) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1025) 



ds_read_u8 v10, v0 offset:3072 // (op 175) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v14, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 176) 




s_ashr_i64 s[0:1], s[34:35], 7 // (op 161) 
s_mov_b32 s9, s0 // (op 161) convert

s_cmp_lt_u32 s8, s9 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161 if true)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161:





















s_barrier  // (op 1089) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

s_mul_hi_i32 s0, s16, s12 // (op 863) 
v_lshrrev_b32 v0, 7, v1 // (op 863) 
s_add_i32 s1, s0, s16 // (op 863) 

v_and_b32 v3, 63, v1 // (op 863) 
s_ashr_i32 s0, s1, 31 // (op 863) 
v_lshrrev_b32 v4, 4, v3 // (op 863) 
s_cmp_eq_i32 s12, 0 // (op 863) 
s_mov_b32 s2, scc // (op 863) 
v_lshlrev_b32 v5, 2, v4 // (op 863) 

s_cmp_eq_i32 s2, 0 // (op 863) 
s_cselect_b32 s3, 0, -1 // (op 863) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 863) 


s_and_b32 s2, s13, 31 // (op 863) 
v_lshrrev_b32 v0, 2, v3 // (op 863) 

s_lshl_b32 s8, 1, s2 // (op 863) 
v_and_b32 v3, 3, v0 // (op 863) 

s_add_i32 s9, s8, s3 // (op 863) 


v_and_b32 v0, 3, v1 // (op 863) 
s_and_b32 s3, s0, s9 // (op 863) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 863) 


s_add_i32 s0, s1, s3 // (op 863) 


v_lshrrev_b32 v0, 6, v1 // (op 863) 
s_ashr_i32 s1, s0, s2 // (op 863) 


v_and_b32 v3, 1, v0 // (op 863) 

s_lshl_b32 s0, s13, 24 // (op 863) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 863) 


s_ashr_i32 s2, s0, 31 // (op 863) 

s_xor_b32 s0, s1, s2 // (op 863) 

s_sub_i32 s1, s0, s2 // (op 863) 


s_mul_hi_i32 s0, s1, s14 // (op 863) 
s_add_i32 s2, s0, s1 // (op 863) 

s_ashr_i32 s0, s2, 31 // (op 863) 
s_cmp_eq_i32 s14, 0 // (op 863) 
s_mov_b32 s3, scc // (op 863) 
s_cmp_eq_i32 s3, 0 // (op 863) 
s_cselect_b32 s8, 0, -1 // (op 863) 

s_and_b32 s3, s15, 31 // (op 863) 
s_lshl_b32 s9, 1, s3 // (op 863) 
s_add_i32 s18, s9, s8 // (op 863) 


s_and_b32 s8, s0, s18 // (op 863) 


s_add_i32 s0, s2, s8 // (op 863) 


s_ashr_i32 s2, s0, s3 // (op 863) 


s_lshl_b32 s0, s15, 24 // (op 863) 
s_ashr_i32 s3, s0, 31 // (op 863) 

s_xor_b32 s0, s2, s3 // (op 863) 

s_sub_i32 s2, s0, s3 // (op 863) 


s_mul_i32 s0, s1, s45 // (op 863) 
s_sub_u32 s3, s16, s0 // (op 863) 

s_mul_i32 s0, s2, s45 // (op 863) 
s_add_u32 s0, s0, s3 // (op 863) 

s_mul_hi_i32 s3, s16, s11 // (op 863) 
s_add_i32 s8, s3, s16 // (op 863) 

s_ashr_i32 s3, s8, 31 // (op 863) 
s_cmp_eq_i32 s11, 0 // (op 863) 
s_mov_b32 s9, scc // (op 863) 
s_cmp_eq_i32 s9, 0 // (op 863) 
s_cselect_b32 s18, 0, -1 // (op 863) 

s_and_b32 s9, s46, 31 // (op 863) 
s_lshl_b32 s19, 1, s9 // (op 863) 
s_add_i32 s20, s19, s18 // (op 863) 


s_and_b32 s18, s3, s20 // (op 863) 


s_add_i32 s3, s8, s18 // (op 863) 


s_ashr_i32 s8, s3, s9 // (op 863) 


s_lshl_b32 s3, s46, 24 // (op 863) 
s_ashr_i32 s9, s3, 31 // (op 863) 

s_xor_b32 s3, s8, s9 // (op 863) 

s_sub_i32 s8, s3, s9 // (op 863) 


s_add_u32 s18, s4, 32 // (op 863) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 863) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 863) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 863) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 863) 

s_mul_i32 s3, s45, s18 // (op 863) 
s_mul_i32 s9, s10, s3 // (op 863) 

s_mul_i32 s3, s8, s9 // (op 863) 

s_sub_u32 s9, s16, s3 // (op 863) 

s_mov_b32 s3, s9 // (op 863) 
s_mul_hi_i32 s20, s3, s44 // (op 863) 
s_add_i32 s21, s20, s3 // (op 863) 


s_ashr_i32 s3, s21, 31 // (op 863) 
s_cmp_eq_i32 s44, 0 // (op 863) 
s_mov_b32 s20, scc // (op 863) 
s_cmp_eq_i32 s20, 0 // (op 863) 
s_cselect_b32 s22, 0, -1 // (op 863) 

s_and_b32 s20, s48, 31 // (op 863) 
s_lshl_b32 s23, 1, s20 // (op 863) 
s_add_i32 s24, s23, s22 // (op 863) 


s_and_b32 s22, s3, s24 // (op 863) 


s_add_i32 s3, s21, s22 // (op 863) 


s_ashr_i32 s21, s3, s20 // (op 863) 


s_lshl_b32 s3, s48, 24 // (op 863) 
s_ashr_i32 s20, s3, 31 // (op 863) 

s_xor_b32 s3, s21, s20 // (op 863) 

s_sub_i32 s21, s3, s20 // (op 863) 


s_mul_i32 s3, s47, s45 // (op 863) 
s_add_u32 s22, s6, 16 // (op 863) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 863) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 863) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 863) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 863) 

s_sub_i32 s20, s22, s3 // (op 863) 

s_mul_i32 s22, s21, s20 // (op 863) 

s_sub_u32 s20, s9, s22 // (op 863) 


s_add_u32 s9, s20, s3 // (op 863) 


s_cmp_eq_i32 s8, 0 // (op 863) 
s_mov_b32 s3, scc // (op 863) 

s_cmp_eq_i32 s3, 0 // (op 863) 
s_cselect_b32 s8, s9, s0 // (op 863) 


s_mul_i32 s0, s2, s18 // (op 863) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 863) 


s_sub_i32 s2, s1, s0 // (op 863) 


s_cmp_eq_i32 s3, 0 // (op 863) 
s_cselect_b32 s0, s21, s2 // (op 863) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 863) 


v_mul_lo_u32 v0, v4, s36 // (op 863) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 863) 


s_mul_i32 s9, s36, s5 // (op 865) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 865) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 865) least significant
s_mul_hi_u32 s1, s36, s4 // (op 865) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 865) most significant: sum
s_add_u32 s9, s9, s1 // (op 865) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 865) 

s_mov_b32 s32, s38 // (op 865) reg expression
s_mov_b32 s33, s39 // (op 865) reg expression
s_mov_b32 s34, s0 // (op 865) reg expression
s_mov_b32 s35, 131072 // (op 865) call()




buffer_store_dwordx4 v[10:13], v4, s[32:35], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123:
s_mul_hi_i32 s0, s16, s12 // (op 685) 
v_lshrrev_b32 v0, 7, v1 // (op 685) 
s_add_i32 s1, s0, s16 // (op 685) 

v_and_b32 v3, 63, v1 // (op 685) 
s_ashr_i32 s0, s1, 31 // (op 685) 
v_lshrrev_b32 v4, 4, v3 // (op 685) 
s_cmp_eq_i32 s12, 0 // (op 685) 
s_mov_b32 s2, scc // (op 685) 
v_lshlrev_b32 v5, 2, v4 // (op 685) 

s_cmp_eq_i32 s2, 0 // (op 685) 
s_cselect_b32 s3, 0, -1 // (op 685) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 685) 


s_and_b32 s2, s13, 31 // (op 685) 
v_lshrrev_b32 v0, 2, v3 // (op 685) 

s_lshl_b32 s8, 1, s2 // (op 685) 
v_and_b32 v3, 3, v0 // (op 685) 

s_add_i32 s9, s8, s3 // (op 685) 


v_and_b32 v0, 3, v1 // (op 685) 
s_and_b32 s3, s0, s9 // (op 685) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 685) 


s_add_i32 s0, s1, s3 // (op 685) 


v_lshrrev_b32 v0, 6, v1 // (op 685) 
s_ashr_i32 s1, s0, s2 // (op 685) 


v_and_b32 v3, 1, v0 // (op 685) 

s_lshl_b32 s0, s13, 24 // (op 685) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 685) 


s_ashr_i32 s2, s0, 31 // (op 685) 

s_xor_b32 s0, s1, s2 // (op 685) 

s_sub_i32 s1, s0, s2 // (op 685) 


s_mul_hi_i32 s0, s1, s14 // (op 685) 
s_add_i32 s2, s0, s1 // (op 685) 

s_ashr_i32 s0, s2, 31 // (op 685) 
s_cmp_eq_i32 s14, 0 // (op 685) 
s_mov_b32 s3, scc // (op 685) 
s_cmp_eq_i32 s3, 0 // (op 685) 
s_cselect_b32 s8, 0, -1 // (op 685) 

s_and_b32 s3, s15, 31 // (op 685) 
s_lshl_b32 s9, 1, s3 // (op 685) 
s_add_i32 s18, s9, s8 // (op 685) 


s_and_b32 s8, s0, s18 // (op 685) 


s_add_i32 s0, s2, s8 // (op 685) 


s_ashr_i32 s2, s0, s3 // (op 685) 


s_lshl_b32 s0, s15, 24 // (op 685) 
s_ashr_i32 s3, s0, 31 // (op 685) 

s_xor_b32 s0, s2, s3 // (op 685) 

s_sub_i32 s2, s0, s3 // (op 685) 


s_mul_i32 s0, s1, s45 // (op 685) 
s_sub_u32 s3, s16, s0 // (op 685) 

s_mul_i32 s0, s2, s45 // (op 685) 
s_add_u32 s0, s0, s3 // (op 685) 

s_mul_hi_i32 s3, s16, s11 // (op 685) 
s_add_i32 s8, s3, s16 // (op 685) 

s_ashr_i32 s3, s8, 31 // (op 685) 
s_cmp_eq_i32 s11, 0 // (op 685) 
s_mov_b32 s9, scc // (op 685) 
s_cmp_eq_i32 s9, 0 // (op 685) 
s_cselect_b32 s18, 0, -1 // (op 685) 

s_and_b32 s9, s46, 31 // (op 685) 
s_lshl_b32 s19, 1, s9 // (op 685) 
s_add_i32 s20, s19, s18 // (op 685) 


s_and_b32 s18, s3, s20 // (op 685) 


s_add_i32 s3, s8, s18 // (op 685) 


s_ashr_i32 s8, s3, s9 // (op 685) 


s_lshl_b32 s3, s46, 24 // (op 685) 
s_ashr_i32 s9, s3, 31 // (op 685) 

s_xor_b32 s3, s8, s9 // (op 685) 

s_sub_i32 s8, s3, s9 // (op 685) 


s_add_u32 s18, s4, 32 // (op 685) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 685) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 685) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 685) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 685) 

s_mul_i32 s3, s45, s18 // (op 685) 
s_mul_i32 s9, s10, s3 // (op 685) 

s_mul_i32 s3, s8, s9 // (op 685) 

s_sub_u32 s9, s16, s3 // (op 685) 

s_mov_b32 s3, s9 // (op 685) 
s_mul_hi_i32 s20, s3, s44 // (op 685) 
s_add_i32 s21, s20, s3 // (op 685) 


s_ashr_i32 s3, s21, 31 // (op 685) 
s_cmp_eq_i32 s44, 0 // (op 685) 
s_mov_b32 s20, scc // (op 685) 
s_cmp_eq_i32 s20, 0 // (op 685) 
s_cselect_b32 s22, 0, -1 // (op 685) 

s_and_b32 s20, s48, 31 // (op 685) 
s_lshl_b32 s23, 1, s20 // (op 685) 
s_add_i32 s24, s23, s22 // (op 685) 


s_and_b32 s22, s3, s24 // (op 685) 


s_add_i32 s3, s21, s22 // (op 685) 


s_ashr_i32 s21, s3, s20 // (op 685) 


s_lshl_b32 s3, s48, 24 // (op 685) 
s_ashr_i32 s20, s3, 31 // (op 685) 

s_xor_b32 s3, s21, s20 // (op 685) 

s_sub_i32 s21, s3, s20 // (op 685) 


s_mul_i32 s3, s47, s45 // (op 685) 
s_add_u32 s22, s6, 16 // (op 685) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 685) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 685) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 685) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 685) 

s_sub_i32 s20, s22, s3 // (op 685) 

s_mul_i32 s22, s21, s20 // (op 685) 

s_sub_u32 s20, s9, s22 // (op 685) 


s_add_u32 s9, s20, s3 // (op 685) 


s_cmp_eq_i32 s8, 0 // (op 685) 
s_mov_b32 s3, scc // (op 685) 

s_cmp_eq_i32 s3, 0 // (op 685) 
s_cselect_b32 s8, s9, s0 // (op 685) 


s_mul_i32 s0, s2, s18 // (op 685) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 685) 


s_sub_i32 s2, s1, s0 // (op 685) 


s_cmp_eq_i32 s3, 0 // (op 685) 
s_cselect_b32 s0, s21, s2 // (op 685) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 685) 


v_mul_lo_u32 v0, v4, s40 // (op 685) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 685) 


s_mul_i32 s9, s40, s5 // (op 687) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 687) most significant: low of high * low
s_mul_i32 s8, s40, s4 // (op 687) least significant
s_mul_hi_u32 s1, s40, s4 // (op 687) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 687) most significant: sum
s_add_u32 s9, s9, s1 // (op 687) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 687) 

s_mov_b32 s28, s42 // (op 687) reg expression
s_mov_b32 s29, s43 // (op 687) reg expression
s_mov_b32 s30, s0 // (op 687) reg expression
s_mov_b32 s31, 131072 // (op 687) call()





buffer_load_dwordx4 v[10:13], v4, s[28:31], 0 offen // (op 135) Load value



v_accvgpr_read v4, a0 // (op 132) 
v_accvgpr_read v5, a1 // (op 132) 
v_accvgpr_read v6, a2 // (op 132) 
v_accvgpr_read v7, a3 // (op 132) 
v_mul_f32 v14, v9, v4 // (op 132) 
v_mul_f32 v15, v9, v5 // (op 132) 
v_mul_f32 v16, v9, v6 // (op 132) 
v_mul_f32 v17, v9, v7 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v4, v8, v10 // (op 136) 
v_mul_f32 v5, v8, v11 // (op 136) 
v_mul_f32 v6, v8, v12 // (op 136) 
v_mul_f32 v7, v8, v13 // (op 136) 

v_add_f32 v10, v4, v14 // (op 133) 
v_add_f32 v11, v5, v15 // (op 133) 
v_add_f32 v12, v6, v16 // (op 133) 
v_add_f32 v13, v7, v17 // (op 133) 


s_mul_hi_i32 s0, s16, s12 // (op 880) 
v_lshrrev_b32 v0, 7, v1 // (op 880) 
s_add_i32 s1, s0, s16 // (op 880) 

v_and_b32 v3, 63, v1 // (op 880) 
s_ashr_i32 s0, s1, 31 // (op 880) 
v_lshrrev_b32 v4, 4, v3 // (op 880) 
s_cmp_eq_i32 s12, 0 // (op 880) 
s_mov_b32 s2, scc // (op 880) 
v_lshlrev_b32 v5, 2, v4 // (op 880) 

s_cmp_eq_i32 s2, 0 // (op 880) 
s_cselect_b32 s3, 0, -1 // (op 880) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 880) 


s_and_b32 s2, s13, 31 // (op 880) 
v_lshrrev_b32 v0, 2, v3 // (op 880) 

s_lshl_b32 s8, 1, s2 // (op 880) 
v_and_b32 v3, 3, v0 // (op 880) 

s_add_i32 s9, s8, s3 // (op 880) 


v_and_b32 v0, 3, v1 // (op 880) 
s_and_b32 s3, s0, s9 // (op 880) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 880) 


s_add_i32 s0, s1, s3 // (op 880) 


v_lshrrev_b32 v0, 6, v1 // (op 880) 
s_ashr_i32 s1, s0, s2 // (op 880) 


v_and_b32 v3, 1, v0 // (op 880) 

s_lshl_b32 s0, s13, 24 // (op 880) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 880) 


s_ashr_i32 s2, s0, 31 // (op 880) 

s_xor_b32 s0, s1, s2 // (op 880) 

s_sub_i32 s1, s0, s2 // (op 880) 


s_mul_hi_i32 s0, s1, s14 // (op 880) 
s_add_i32 s2, s0, s1 // (op 880) 

s_ashr_i32 s0, s2, 31 // (op 880) 
s_cmp_eq_i32 s14, 0 // (op 880) 
s_mov_b32 s3, scc // (op 880) 
s_cmp_eq_i32 s3, 0 // (op 880) 
s_cselect_b32 s8, 0, -1 // (op 880) 

s_and_b32 s3, s15, 31 // (op 880) 
s_lshl_b32 s9, 1, s3 // (op 880) 
s_add_i32 s18, s9, s8 // (op 880) 


s_and_b32 s8, s0, s18 // (op 880) 


s_add_i32 s0, s2, s8 // (op 880) 


s_ashr_i32 s2, s0, s3 // (op 880) 


s_lshl_b32 s0, s15, 24 // (op 880) 
s_ashr_i32 s3, s0, 31 // (op 880) 

s_xor_b32 s0, s2, s3 // (op 880) 

s_sub_i32 s2, s0, s3 // (op 880) 


s_mul_i32 s0, s1, s45 // (op 880) 
s_sub_u32 s3, s16, s0 // (op 880) 

s_mul_i32 s0, s2, s45 // (op 880) 
s_add_u32 s0, s0, s3 // (op 880) 

s_mul_hi_i32 s3, s16, s11 // (op 880) 
s_add_i32 s8, s3, s16 // (op 880) 

s_ashr_i32 s3, s8, 31 // (op 880) 
s_cmp_eq_i32 s11, 0 // (op 880) 
s_mov_b32 s9, scc // (op 880) 
s_cmp_eq_i32 s9, 0 // (op 880) 
s_cselect_b32 s18, 0, -1 // (op 880) 

s_and_b32 s9, s46, 31 // (op 880) 
s_lshl_b32 s19, 1, s9 // (op 880) 
s_add_i32 s20, s19, s18 // (op 880) 


s_and_b32 s18, s3, s20 // (op 880) 


s_add_i32 s3, s8, s18 // (op 880) 


s_ashr_i32 s8, s3, s9 // (op 880) 


s_lshl_b32 s3, s46, 24 // (op 880) 
s_ashr_i32 s9, s3, 31 // (op 880) 

s_xor_b32 s3, s8, s9 // (op 880) 

s_sub_i32 s8, s3, s9 // (op 880) 


s_add_u32 s18, s4, 32 // (op 880) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 880) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 880) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 880) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 880) 

s_mul_i32 s3, s45, s18 // (op 880) 
s_mul_i32 s9, s10, s3 // (op 880) 

s_mul_i32 s3, s8, s9 // (op 880) 

s_sub_u32 s9, s16, s3 // (op 880) 

s_mov_b32 s3, s9 // (op 880) 
s_mul_hi_i32 s20, s3, s44 // (op 880) 
s_add_i32 s21, s20, s3 // (op 880) 


s_ashr_i32 s3, s21, 31 // (op 880) 
s_cmp_eq_i32 s44, 0 // (op 880) 
s_mov_b32 s20, scc // (op 880) 
s_cmp_eq_i32 s20, 0 // (op 880) 
s_cselect_b32 s22, 0, -1 // (op 880) 

s_and_b32 s20, s48, 31 // (op 880) 
s_lshl_b32 s23, 1, s20 // (op 880) 
s_add_i32 s24, s23, s22 // (op 880) 


s_and_b32 s22, s3, s24 // (op 880) 


s_add_i32 s3, s21, s22 // (op 880) 


s_ashr_i32 s21, s3, s20 // (op 880) 


s_lshl_b32 s3, s48, 24 // (op 880) 
s_ashr_i32 s20, s3, 31 // (op 880) 

s_xor_b32 s3, s21, s20 // (op 880) 

s_sub_i32 s21, s3, s20 // (op 880) 


s_mul_i32 s3, s47, s45 // (op 880) 
s_add_u32 s22, s6, 16 // (op 880) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 880) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 880) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 880) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 880) 

s_sub_i32 s20, s22, s3 // (op 880) 

s_mul_i32 s22, s21, s20 // (op 880) 

s_sub_u32 s20, s9, s22 // (op 880) 


s_add_u32 s9, s20, s3 // (op 880) 


s_cmp_eq_i32 s8, 0 // (op 880) 
s_mov_b32 s3, scc // (op 880) 

s_cmp_eq_i32 s3, 0 // (op 880) 
s_cselect_b32 s8, s9, s0 // (op 880) 


s_mul_i32 s0, s2, s18 // (op 880) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 880) 


s_sub_i32 s2, s1, s0 // (op 880) 


s_cmp_eq_i32 s3, 0 // (op 880) 
s_cselect_b32 s0, s21, s2 // (op 880) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 880) 


v_mul_lo_u32 v0, v4, s36 // (op 880) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 880) 


s_mul_i32 s9, s36, s5 // (op 882) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 882) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 882) least significant
s_mul_hi_u32 s1, s36, s4 // (op 882) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 882) most significant: sum
s_add_u32 s9, s9, s1 // (op 882) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 882) 

s_mov_b32 s32, s38 // (op 882) reg expression
s_mov_b32 s33, s39 // (op 882) reg expression
s_mov_b32 s34, s0 // (op 882) reg expression
s_mov_b32 s35, 131072 // (op 882) call()




buffer_store_dwordx4 v[10:13], v4, s[32:35], 0 offen // (op 134) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.LRR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end:
.size RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_, .LRR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end-RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 9472 // lds bytes
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
  ProblemType:
    OperationType: GEMM
    DataType: B6
    DataTypeA: B6
    DataTypeB: B6
    DestDataType: s
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
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 9472
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 70
    .vgpr_count: 59
    .agpr_count: 4
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 36
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}}
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

