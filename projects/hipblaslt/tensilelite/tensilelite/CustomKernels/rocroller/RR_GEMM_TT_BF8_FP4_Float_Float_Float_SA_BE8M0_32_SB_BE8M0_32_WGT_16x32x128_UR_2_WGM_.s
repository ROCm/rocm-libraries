// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_,@function
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 743) 
v_and_b32 v3, 15, v1 // (op 743) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 743) 


v_and_b32 v0, 48, v1 // (op 743) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 743) 


v_and_b32 v0, 63, v1 // (op 796) 
v_and_b32 v4, 48, v1 // (op 796) 
v_sub_u32 v5, v0, v4 // (op 796) 

v_lshrrev_b32 v4, 4, v0 // (op 796) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 796) 


v_lshrrev_b32 v4, 6, v1 // (op 796) 
v_and_b32 v5, 1, v4 // (op 796) 

v_lshlrev_b32 v4, 4, v5 // (op 796) 

v_lshl_add_u32 v5, v0, 5, v4 // (op 796) 


v_lshlrev_b32 v0, 2, v5 // (op 796) 

v_lshrrev_b32 v4, 3, v0 // (op 796) 

v_lshrrev_b32 v0, 6, v1 // (op 849) 
v_and_b32 v5, 1, v0 // (op 849) 

v_and_b32 v0, 15, v1 // (op 849) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 849) 


v_and_b32 v0, 63, v1 // (op 849) 
v_lshrrev_b32 v5, 4, v0 // (op 849) 

v_lshl_add_u32 v0, v5, 5, v6 // (op 849) 


v_lshrrev_b32 v5, 7, v1 // (op 907) 
v_and_b32 v6, 15, v1 // (op 907) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 907) 


v_and_b32 v5, 48, v1 // (op 907) 
v_lshl_add_u32 v6, v7, 7, v5 // (op 907) 


v_and_b32 v5, 63, v1 // (op 931) 
v_and_b32 v7, 48, v1 // (op 931) 
v_sub_u32 v8, v5, v7 // (op 931) 

v_lshrrev_b32 v7, 4, v5 // (op 931) 

v_lshl_add_u32 v5, v7, 5, v8 // (op 931) 


v_lshrrev_b32 v7, 6, v1 // (op 931) 
v_and_b32 v8, 1, v7 // (op 931) 

v_lshlrev_b32 v7, 4, v8 // (op 931) 

v_lshl_add_u32 v8, v5, 5, v7 // (op 931) 


v_lshlrev_b32 v5, 2, v8 // (op 931) 

v_lshrrev_b32 v7, 3, v5 // (op 931) 

v_lshrrev_b32 v5, 6, v1 // (op 955) 
v_and_b32 v8, 1, v5 // (op 955) 

v_and_b32 v5, 15, v1 // (op 955) 
v_lshl_add_u32 v9, v8, 4, v5 // (op 955) 


v_and_b32 v5, 63, v1 // (op 955) 
v_lshrrev_b32 v8, 4, v5 // (op 955) 

v_lshl_add_u32 v5, v8, 5, v9 // (op 955) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v9, s9 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
s_mul_hi_i32 s0, s16, s11 // (op 453) 
v_lshrrev_b32 v10, 3, v1 // (op 453) 
s_add_i32 s1, s0, s16 // (op 453) 

v_and_b32 v11, -8, v1 // (op 453) 
s_ashr_i32 s0, s1, 31 // (op 453) 
v_sub_u32 v12, v1, v11 // (op 453) 

s_cmp_eq_i32 s11, 0 // (op 453) 
s_mov_b32 s8, scc // (op 453) 
v_lshlrev_b32 v11, 4, v12 // (op 453) 

s_cmp_eq_i32 s8, 0 // (op 453) 
s_cselect_b32 s9, 0, -1 // (op 453) 

s_and_b32 s8, s12, 31 // (op 453) 
s_lshl_b32 s18, 1, s8 // (op 453) 
s_add_i32 s19, s18, s9 // (op 453) 


s_and_b32 s9, s0, s19 // (op 453) 


s_add_i32 s0, s1, s9 // (op 453) 


s_ashr_i32 s1, s0, s8 // (op 453) 


s_lshl_b32 s0, s12, 24 // (op 453) 
s_ashr_i32 s8, s0, 31 // (op 453) 

s_xor_b32 s0, s1, s8 // (op 453) 

s_sub_i32 s1, s0, s8 // (op 453) 


s_mul_hi_i32 s0, s1, s13 // (op 453) 
s_add_i32 s8, s0, s1 // (op 453) 

s_ashr_i32 s0, s8, 31 // (op 453) 
s_cmp_eq_i32 s13, 0 // (op 453) 
s_mov_b32 s9, scc // (op 453) 
s_cmp_eq_i32 s9, 0 // (op 453) 
s_cselect_b32 s18, 0, -1 // (op 453) 

s_and_b32 s9, s14, 31 // (op 453) 
s_lshl_b32 s19, 1, s9 // (op 453) 
s_add_i32 s51, s19, s18 // (op 453) 


s_and_b32 s18, s0, s51 // (op 453) 


s_add_i32 s0, s8, s18 // (op 453) 


s_ashr_i32 s8, s0, s9 // (op 453) 


s_lshl_b32 s0, s14, 24 // (op 453) 
s_ashr_i32 s9, s0, 31 // (op 453) 

s_xor_b32 s0, s8, s9 // (op 453) 

s_sub_i32 s8, s0, s9 // (op 453) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s46 // (op 453) 

s_sub_u32 s1, s16, s0 // (op 453) 

s_mul_i32 s0, s8, s46 // (op 453) 
s_add_u32 s0, s0, s1 // (op 453) 


s_mul_hi_i32 s1, s16, s10 // (op 453) 
s_add_i32 s8, s1, s16 // (op 453) 

s_ashr_i32 s1, s8, 31 // (op 453) 
s_cmp_eq_i32 s10, 0 // (op 453) 
s_mov_b32 s9, scc // (op 453) 
s_cmp_eq_i32 s9, 0 // (op 453) 
s_cselect_b32 s18, 0, -1 // (op 453) 

s_and_b32 s9, s49, 31 // (op 453) 
s_lshl_b32 s19, 1, s9 // (op 453) 
s_add_i32 s51, s19, s18 // (op 453) 


s_and_b32 s18, s1, s51 // (op 453) 


s_add_i32 s1, s8, s18 // (op 453) 


s_ashr_i32 s8, s1, s9 // (op 453) 


s_lshl_b32 s1, s49, 24 // (op 453) 
s_ashr_i32 s9, s1, 31 // (op 453) 

s_xor_b32 s1, s8, s9 // (op 453) 

s_sub_i32 s8, s1, s9 // (op 453) 


s_add_u32 s18, s2, 32 // (op 453) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 453) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 453) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 453) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 5 // (op 453) 

s_mul_i32 s1, s46, s18 // (op 453) 

s_mul_i32 s9, s50, s1 // (op 453) 

s_mul_i32 s1, s8, s9 // (op 453) 

s_sub_u32 s9, s16, s1 // (op 453) 

s_mov_b32 s1, s9 // (op 453) 
s_mul_hi_i32 s18, s1, s15 // (op 453) 
s_add_i32 s19, s18, s1 // (op 453) 


s_ashr_i32 s1, s19, 31 // (op 453) 
s_cmp_eq_i32 s15, 0 // (op 453) 
s_mov_b32 s18, scc // (op 453) 
s_cmp_eq_i32 s18, 0 // (op 453) 
s_cselect_b32 s51, 0, -1 // (op 453) 

s_and_b32 s18, s48, 31 // (op 453) 
s_lshl_b32 s52, 1, s18 // (op 453) 
s_add_i32 s53, s52, s51 // (op 453) 


s_and_b32 s51, s1, s53 // (op 453) 


s_add_i32 s1, s19, s51 // (op 453) 


s_ashr_i32 s19, s1, s18 // (op 453) 


s_lshl_b32 s1, s48, 24 // (op 453) 
s_ashr_i32 s18, s1, 31 // (op 453) 

s_xor_b32 s1, s19, s18 // (op 453) 

s_sub_i32 s19, s1, s18 // (op 453) 


s_mul_i32 s1, s47, s46 // (op 453) 
s_add_u32 s52, s6, 16 // (op 453) Start of Int64 add, locking SCC
s_addc_u32 s53, s7, 0 // (op 453) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 453) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 453) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 453) 

s_sub_i32 s18, s52, s1 // (op 453) 

s_mul_i32 s51, s19, s18 // (op 453) 


s_sub_u32 s18, s9, s51 // (op 453) 


s_add_u32 s9, s18, s1 // (op 453) 


s_cmp_eq_i32 s8, 0 // (op 453) 
s_mov_b32 s1, scc // (op 453) 

s_cmp_eq_i32 s1, 0 // (op 453) 
s_cselect_b32 s8, s9, s0 // (op 453) 



v_lshl_add_u32 v12, s8, 4, v10 // (op 453) 


v_mul_lo_u32 v10, v12, s4 // (op 453) 
v_add_u32 v10, v10, v11 // (op 453) 


s_mul_i32 s1, s4, s7 // (op 455) most significant: low of low * high
s_mul_i32 s8, s5, s6 // (op 455) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 455) least significant
s_mul_hi_u32 s9, s4, s6 // (op 455) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 455) most significant: sum
s_add_u32 s1, s1, s9 // (op 455) most significant: sum


s_mov_b32 s52, s20 // (op 455) reg expression
s_mov_b32 s53, s21 // (op 455) reg expression
s_mov_b32 s54, s0 // (op 455) reg expression
s_mov_b32 s55, 131072 // (op 455) call()

s_mul_hi_i32 s51, s16, s11 // (op 472) 
v_lshrrev_b32 v11, 3, v1 // (op 472) 
s_add_i32 s0, s51, s16 // (op 472) 

v_and_b32 v12, -8, v1 // (op 472) 
s_ashr_i32 s1, s0, 31 // (op 472) 
v_sub_u32 v13, v1, v12 // (op 472) 

s_cmp_eq_i32 s11, 0 // (op 472) 
s_mov_b32 s51, scc // (op 472) 
v_lshlrev_b32 v12, 4, v13 // (op 472) 

s_cmp_eq_i32 s51, 0 // (op 472) 
s_cselect_b32 s8, 0, -1 // (op 472) 

s_and_b32 s9, s12, 31 // (op 472) 
s_lshl_b32 s51, 1, s9 // (op 472) 
s_add_i32 s18, s51, s8 // (op 472) 


s_and_b32 s8, s1, s18 // (op 472) 


s_add_i32 s1, s0, s8 // (op 472) 


s_ashr_i32 s0, s1, s9 // (op 472) 


s_lshl_b32 s1, s12, 24 // (op 472) 
s_ashr_i32 s51, s1, 31 // (op 472) 

s_xor_b32 s1, s0, s51 // (op 472) 

s_sub_i32 s0, s1, s51 // (op 472) 


s_mul_hi_i32 s1, s0, s13 // (op 472) 
s_add_i32 s51, s1, s0 // (op 472) 

s_ashr_i32 s1, s51, 31 // (op 472) 
s_cmp_eq_i32 s13, 0 // (op 472) 
s_mov_b32 s8, scc // (op 472) 
s_cmp_eq_i32 s8, 0 // (op 472) 
s_cselect_b32 s9, 0, -1 // (op 472) 

s_and_b32 s8, s14, 31 // (op 472) 
s_lshl_b32 s18, 1, s8 // (op 472) 
s_add_i32 s19, s18, s9 // (op 472) 


s_and_b32 s9, s1, s19 // (op 472) 


s_add_i32 s1, s51, s9 // (op 472) 


s_ashr_i32 s9, s1, s8 // (op 472) 


s_lshl_b32 s1, s14, 24 // (op 472) 
s_ashr_i32 s8, s1, 31 // (op 472) 

s_xor_b32 s1, s9, s8 // (op 472) 

s_sub_i32 s9, s1, s8 // (op 472) 


s_mul_i32 s1, s0, s46 // (op 472) 

s_sub_u32 s0, s16, s1 // (op 472) 

s_mul_i32 s1, s9, s46 // (op 472) 
s_add_u32 s1, s1, s0 // (op 472) 


s_mul_hi_i32 s0, s16, s10 // (op 472) 
s_add_i32 s51, s0, s16 // (op 472) 

s_ashr_i32 s0, s51, 31 // (op 472) 
s_cmp_eq_i32 s10, 0 // (op 472) 
s_mov_b32 s8, scc // (op 472) 
s_cmp_eq_i32 s8, 0 // (op 472) 
s_cselect_b32 s9, 0, -1 // (op 472) 

s_and_b32 s8, s49, 31 // (op 472) 
s_lshl_b32 s18, 1, s8 // (op 472) 
s_add_i32 s19, s18, s9 // (op 472) 


s_and_b32 s9, s0, s19 // (op 472) 


s_add_i32 s0, s51, s9 // (op 472) 


s_ashr_i32 s9, s0, s8 // (op 472) 


s_lshl_b32 s0, s49, 24 // (op 472) 
s_ashr_i32 s8, s0, 31 // (op 472) 

s_xor_b32 s0, s9, s8 // (op 472) 

s_sub_i32 s9, s0, s8 // (op 472) 


s_add_u32 s18, s2, 32 // (op 472) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 472) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 472) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 472) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 472) 

s_mul_i32 s0, s46, s18 // (op 472) 

s_mul_i32 s8, s50, s0 // (op 472) 

s_mul_i32 s0, s9, s8 // (op 472) 

s_sub_u32 s8, s16, s0 // (op 472) 

s_mov_b32 s0, s8 // (op 472) 
s_mul_hi_i32 s51, s0, s15 // (op 472) 
s_add_i32 s18, s51, s0 // (op 472) 


s_ashr_i32 s0, s18, 31 // (op 472) 
s_cmp_eq_i32 s15, 0 // (op 472) 
s_mov_b32 s19, scc // (op 472) 
s_cmp_eq_i32 s19, 0 // (op 472) 
s_cselect_b32 s51, 0, -1 // (op 472) 

s_and_b32 s19, s48, 31 // (op 472) 
s_lshl_b32 s56, 1, s19 // (op 472) 
s_add_i32 s57, s56, s51 // (op 472) 


s_and_b32 s51, s0, s57 // (op 472) 


s_add_i32 s0, s18, s51 // (op 472) 


s_ashr_i32 s18, s0, s19 // (op 472) 


s_lshl_b32 s0, s48, 24 // (op 472) 
s_ashr_i32 s19, s0, 31 // (op 472) 

s_xor_b32 s0, s18, s19 // (op 472) 

s_sub_i32 s18, s0, s19 // (op 472) 


s_mul_i32 s0, s47, s46 // (op 472) 
s_add_u32 s56, s6, 16 // (op 472) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 472) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 472) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 472) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 472) 

s_sub_i32 s19, s56, s0 // (op 472) 

s_mul_i32 s51, s18, s19 // (op 472) 


s_sub_u32 s18, s8, s51 // (op 472) 


s_add_u32 s8, s18, s0 // (op 472) 


s_cmp_eq_i32 s9, 0 // (op 472) 
s_mov_b32 s0, scc // (op 472) 

s_cmp_eq_i32 s0, 0 // (op 472) 
s_cselect_b32 s9, s8, s1 // (op 472) 



v_lshl_add_u32 v13, s9, 4, v11 // (op 472) 


v_mul_lo_u32 v11, v13, s4 // (op 472) 
v_add_u32 v11, v11, v12 // (op 472) 


s_mul_i32 s1, s4, s7 // (op 474) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 474) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 474) least significant
s_mul_hi_u32 s8, s4, s6 // (op 474) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 474) most significant: sum
s_add_u32 s1, s1, s8 // (op 474) most significant: sum


s_mov_b32 s52, s20 // (op 474) reg expression
s_mov_b32 s53, s21 // (op 474) reg expression
s_mov_b32 s54, s0 // (op 474) reg expression
s_mov_b32 s55, 131072 // (op 474) call()

s_mul_hi_i32 s51, s16, s11 // (op 517) 
s_add_i32 s0, s51, s16 // (op 517) 

s_ashr_i32 s1, s0, 31 // (op 517) 
s_cmp_eq_i32 s11, 0 // (op 517) 
s_mov_b32 s51, scc // (op 517) 
s_cmp_eq_i32 s51, 0 // (op 517) 
s_cselect_b32 s8, 0, -1 // (op 517) 

s_and_b32 s9, s12, 31 // (op 517) 
s_lshl_b32 s51, 1, s9 // (op 517) 
s_add_i32 s18, s51, s8 // (op 517) 


s_and_b32 s8, s1, s18 // (op 517) 


s_add_i32 s1, s0, s8 // (op 517) 


s_ashr_i32 s0, s1, s9 // (op 517) 


s_lshl_b32 s1, s12, 24 // (op 517) 
s_ashr_i32 s51, s1, 31 // (op 517) 

s_xor_b32 s1, s0, s51 // (op 517) 

s_sub_i32 s0, s1, s51 // (op 517) 


s_mul_hi_i32 s1, s0, s13 // (op 517) 
s_add_i32 s51, s1, s0 // (op 517) 

s_ashr_i32 s1, s51, 31 // (op 517) 
s_cmp_eq_i32 s13, 0 // (op 517) 
s_mov_b32 s8, scc // (op 517) 
s_cmp_eq_i32 s8, 0 // (op 517) 
s_cselect_b32 s9, 0, -1 // (op 517) 

s_and_b32 s8, s14, 31 // (op 517) 
s_lshl_b32 s18, 1, s8 // (op 517) 
s_add_i32 s19, s18, s9 // (op 517) 


s_and_b32 s9, s1, s19 // (op 517) 


s_add_i32 s1, s51, s9 // (op 517) 


s_ashr_i32 s9, s1, s8 // (op 517) 


s_lshl_b32 s1, s14, 24 // (op 517) 
s_ashr_i32 s8, s1, 31 // (op 517) 

s_xor_b32 s1, s9, s8 // (op 517) 

s_sub_i32 s9, s1, s8 // (op 517) 


s_add_u32 s18, s2, 32 // (op 517) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 517) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 517) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 517) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 517) 

s_mul_i32 s1, s9, s18 // (op 517) 

s_sub_i32 s51, s0, s1 // (op 517) 


s_mul_hi_i32 s0, s16, s10 // (op 517) 
s_add_i32 s1, s0, s16 // (op 517) 

s_ashr_i32 s0, s1, 31 // (op 517) 
s_cmp_eq_i32 s10, 0 // (op 517) 
s_mov_b32 s8, scc // (op 517) 
s_cmp_eq_i32 s8, 0 // (op 517) 
s_cselect_b32 s9, 0, -1 // (op 517) 

s_and_b32 s8, s49, 31 // (op 517) 
s_lshl_b32 s56, 1, s8 // (op 517) 
s_add_i32 s57, s56, s9 // (op 517) 


s_and_b32 s9, s0, s57 // (op 517) 


s_add_i32 s0, s1, s9 // (op 517) 


s_ashr_i32 s1, s0, s8 // (op 517) 


s_lshl_b32 s0, s49, 24 // (op 517) 
s_ashr_i32 s8, s0, 31 // (op 517) 

s_xor_b32 s0, s1, s8 // (op 517) 

s_sub_i32 s1, s0, s8 // (op 517) 


s_mul_i32 s0, s46, s18 // (op 517) 

s_mul_i32 s8, s50, s0 // (op 517) 

s_mul_i32 s0, s1, s8 // (op 517) 

s_sub_u32 s8, s16, s0 // (op 517) 

s_mul_hi_i32 s0, s8, s15 // (op 517) 
s_add_i32 s9, s0, s8 // (op 517) 


s_ashr_i32 s0, s9, 31 // (op 517) 
s_cmp_eq_i32 s15, 0 // (op 517) 
s_mov_b32 s8, scc // (op 517) 
s_cmp_eq_i32 s8, 0 // (op 517) 
s_cselect_b32 s18, 0, -1 // (op 517) 

s_and_b32 s8, s48, 31 // (op 517) 
s_lshl_b32 s19, 1, s8 // (op 517) 
s_add_i32 s56, s19, s18 // (op 517) 


s_and_b32 s18, s0, s56 // (op 517) 


s_add_i32 s0, s9, s18 // (op 517) 


s_ashr_i32 s9, s0, s8 // (op 517) 


s_lshl_b32 s0, s48, 24 // (op 517) 
s_ashr_i32 s8, s0, 31 // (op 517) 

s_xor_b32 s0, s9, s8 // (op 517) 

s_sub_i32 s9, s0, s8 // (op 517) 


s_cmp_eq_i32 s1, 0 // (op 517) 
s_mov_b32 s0, scc // (op 517) 

s_cmp_eq_i32 s0, 0 // (op 517) 
s_cselect_b32 s1, s9, s51 // (op 517) 



s_lshl_b32 s0, s1, 5 // (op 517) 

s_mov_b32 s9, 0 // (op 517) convert
s_mov_b32 s8, s0 // (op 517) convert

v_mul_lo_u32 v13, v1, s23 // (op 517) most significant: low of low * high
v_mul_hi_u32 v14, v1, s22 // (op 517) most significant: high of low * low
v_mul_lo_u32 v12, v1, s22 // (op 517) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 517) most significant: sum

v_mov_b32 v14, s8 // (op 517) 
v_mov_b32 v15, s9 // (op 517) 
v_add_co_u32 v12, s[0:1], v12, v14 // (op 517) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 517) most significant half




v_lshlrev_b64 v[14:15], 2, v[12:13] // (op 517) 

v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 517) 

v_mov_b32 v14, v12 // (op 517) convert

s_mul_i32 s1, s22, s27 // (op 519) most significant: low of low * high
s_mul_i32 s51, s23, s26 // (op 519) most significant: low of high * low
s_mul_i32 s0, s22, s26 // (op 519) least significant
s_mul_hi_u32 s8, s22, s26 // (op 519) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 519) most significant: sum
s_add_u32 s1, s1, s8 // (op 519) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 519) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 519) 

s_mov_b32 s56, s24 // (op 519) reg expression
s_mov_b32 s57, s25 // (op 519) reg expression
s_mov_b32 s58, s0 // (op 519) reg expression
s_mov_b32 s59, 131072 // (op 519) call()

s_mul_hi_i32 s51, s16, s11 // (op 536) 
s_add_i32 s0, s51, s16 // (op 536) 

s_ashr_i32 s1, s0, 31 // (op 536) 
s_cmp_eq_i32 s11, 0 // (op 536) 
s_mov_b32 s51, scc // (op 536) 
s_cmp_eq_i32 s51, 0 // (op 536) 
s_cselect_b32 s8, 0, -1 // (op 536) 

s_and_b32 s9, s12, 31 // (op 536) 
s_lshl_b32 s51, 1, s9 // (op 536) 
s_add_i32 s18, s51, s8 // (op 536) 


s_and_b32 s8, s1, s18 // (op 536) 


s_add_i32 s1, s0, s8 // (op 536) 


s_ashr_i32 s0, s1, s9 // (op 536) 


s_lshl_b32 s1, s12, 24 // (op 536) 
s_ashr_i32 s51, s1, 31 // (op 536) 

s_xor_b32 s1, s0, s51 // (op 536) 

s_sub_i32 s0, s1, s51 // (op 536) 


s_mul_hi_i32 s1, s0, s13 // (op 536) 
s_add_i32 s51, s1, s0 // (op 536) 

s_ashr_i32 s1, s51, 31 // (op 536) 
s_cmp_eq_i32 s13, 0 // (op 536) 
s_mov_b32 s8, scc // (op 536) 
s_cmp_eq_i32 s8, 0 // (op 536) 
s_cselect_b32 s9, 0, -1 // (op 536) 

s_and_b32 s8, s14, 31 // (op 536) 
s_lshl_b32 s18, 1, s8 // (op 536) 
s_add_i32 s19, s18, s9 // (op 536) 


s_and_b32 s9, s1, s19 // (op 536) 


s_add_i32 s1, s51, s9 // (op 536) 


s_ashr_i32 s9, s1, s8 // (op 536) 


s_lshl_b32 s1, s14, 24 // (op 536) 
s_ashr_i32 s8, s1, 31 // (op 536) 

s_xor_b32 s1, s9, s8 // (op 536) 

s_sub_i32 s9, s1, s8 // (op 536) 


s_add_u32 s18, s2, 32 // (op 536) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 536) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 536) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 536) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 536) 

s_mul_i32 s1, s9, s18 // (op 536) 

s_sub_i32 s51, s0, s1 // (op 536) 


s_mul_hi_i32 s0, s16, s10 // (op 536) 
s_add_i32 s1, s0, s16 // (op 536) 

s_ashr_i32 s0, s1, 31 // (op 536) 
s_cmp_eq_i32 s10, 0 // (op 536) 
s_mov_b32 s8, scc // (op 536) 
s_cmp_eq_i32 s8, 0 // (op 536) 
s_cselect_b32 s9, 0, -1 // (op 536) 

s_and_b32 s8, s49, 31 // (op 536) 
s_lshl_b32 s60, 1, s8 // (op 536) 
s_add_i32 s61, s60, s9 // (op 536) 


s_and_b32 s9, s0, s61 // (op 536) 


s_add_i32 s0, s1, s9 // (op 536) 


s_ashr_i32 s1, s0, s8 // (op 536) 


s_lshl_b32 s0, s49, 24 // (op 536) 
s_ashr_i32 s8, s0, 31 // (op 536) 

s_xor_b32 s0, s1, s8 // (op 536) 

s_sub_i32 s1, s0, s8 // (op 536) 


s_mul_i32 s0, s46, s18 // (op 536) 

s_mul_i32 s8, s50, s0 // (op 536) 

s_mul_i32 s0, s1, s8 // (op 536) 

s_sub_u32 s8, s16, s0 // (op 536) 

s_mul_hi_i32 s0, s8, s15 // (op 536) 
s_add_i32 s9, s0, s8 // (op 536) 


s_ashr_i32 s0, s9, 31 // (op 536) 
s_cmp_eq_i32 s15, 0 // (op 536) 
s_mov_b32 s8, scc // (op 536) 
s_cmp_eq_i32 s8, 0 // (op 536) 
s_cselect_b32 s18, 0, -1 // (op 536) 

s_and_b32 s8, s48, 31 // (op 536) 
s_lshl_b32 s19, 1, s8 // (op 536) 
s_add_i32 s60, s19, s18 // (op 536) 


s_and_b32 s18, s0, s60 // (op 536) 


s_add_i32 s0, s9, s18 // (op 536) 


s_ashr_i32 s9, s0, s8 // (op 536) 


s_lshl_b32 s0, s48, 24 // (op 536) 
s_ashr_i32 s8, s0, 31 // (op 536) 

s_xor_b32 s0, s9, s8 // (op 536) 

s_sub_i32 s9, s0, s8 // (op 536) 


s_cmp_eq_i32 s1, 0 // (op 536) 
s_mov_b32 s0, scc // (op 536) 

s_cmp_eq_i32 s0, 0 // (op 536) 
s_cselect_b32 s1, s9, s51 // (op 536) 



s_lshl_b32 s0, s1, 5 // (op 536) 

s_mov_b32 s9, 0 // (op 536) convert
s_mov_b32 s8, s0 // (op 536) convert

v_mul_lo_u32 v13, v1, s23 // (op 536) most significant: low of low * high
v_mul_hi_u32 v15, v1, s22 // (op 536) most significant: high of low * low
v_mul_lo_u32 v12, v1, s22 // (op 536) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 536) most significant: sum

v_mov_b32 v15, s8 // (op 536) 
v_mov_b32 v16, s9 // (op 536) 
v_add_co_u32 v12, s[0:1], v12, v15 // (op 536) least significant half
v_addc_co_u32 v13, s[0:1], v13, v16, s[0:1] // (op 536) most significant half




v_lshlrev_b64 v[16:17], 2, v[12:13] // (op 536) 

v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 536) 

v_mov_b32 v15, v12 // (op 536) convert

s_mul_i32 s1, s22, s27 // (op 538) most significant: low of low * high
s_mul_i32 s51, s23, s26 // (op 538) most significant: low of high * low
s_mul_i32 s0, s22, s26 // (op 538) least significant
s_mul_hi_u32 s8, s22, s26 // (op 538) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 538) most significant: sum
s_add_u32 s1, s1, s8 // (op 538) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 538) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 538) 

s_mov_b32 s56, s24 // (op 538) reg expression
s_mov_b32 s57, s25 // (op 538) reg expression
s_mov_b32 s58, s0 // (op 538) reg expression
s_mov_b32 s59, 131072 // (op 538) call()

s_mul_hi_i32 s51, s16, s11 // (op 581) 
v_lshrrev_b32 v12, 7, v1 // (op 581) 
s_add_i32 s0, s51, s16 // (op 581) 

v_and_b32 v13, 15, v1 // (op 581) 
s_ashr_i32 s1, s0, 31 // (op 581) 
v_lshl_add_u32 v16, v12, 4, v13 // (op 581) 


s_cmp_eq_i32 s11, 0 // (op 581) 
s_mov_b32 s51, scc // (op 581) 
v_and_b32 v12, 63, v1 // (op 581) 
s_cmp_eq_i32 s51, 0 // (op 581) 
s_cselect_b32 s8, 0, -1 // (op 581) 

v_lshrrev_b32 v13, 4, v12 // (op 581) 

s_and_b32 s9, s12, 31 // (op 581) 
s_lshl_b32 s51, 1, s9 // (op 581) 
s_add_i32 s18, s51, s8 // (op 581) 


s_and_b32 s8, s1, s18 // (op 581) 


s_add_i32 s1, s0, s8 // (op 581) 


s_ashr_i32 s0, s1, s9 // (op 581) 


s_lshl_b32 s1, s12, 24 // (op 581) 
s_ashr_i32 s51, s1, 31 // (op 581) 

s_xor_b32 s1, s0, s51 // (op 581) 

s_sub_i32 s0, s1, s51 // (op 581) 


s_mul_hi_i32 s1, s0, s13 // (op 581) 
s_add_i32 s51, s1, s0 // (op 581) 

s_ashr_i32 s1, s51, 31 // (op 581) 
s_cmp_eq_i32 s13, 0 // (op 581) 
s_mov_b32 s8, scc // (op 581) 
s_cmp_eq_i32 s8, 0 // (op 581) 
s_cselect_b32 s9, 0, -1 // (op 581) 

s_and_b32 s8, s14, 31 // (op 581) 
s_lshl_b32 s18, 1, s8 // (op 581) 
s_add_i32 s19, s18, s9 // (op 581) 


s_and_b32 s9, s1, s19 // (op 581) 


s_add_i32 s1, s51, s9 // (op 581) 


s_ashr_i32 s9, s1, s8 // (op 581) 


s_lshl_b32 s1, s14, 24 // (op 581) 
s_ashr_i32 s8, s1, 31 // (op 581) 

s_xor_b32 s1, s9, s8 // (op 581) 

s_sub_i32 s9, s1, s8 // (op 581) 


s_mul_i32 s1, s0, s46 // (op 581) 

s_sub_u32 s0, s16, s1 // (op 581) 

s_mul_i32 s1, s9, s46 // (op 581) 
s_add_u32 s1, s1, s0 // (op 581) 


s_mul_hi_i32 s0, s16, s10 // (op 581) 
s_add_i32 s51, s0, s16 // (op 581) 

s_ashr_i32 s0, s51, 31 // (op 581) 
s_cmp_eq_i32 s10, 0 // (op 581) 
s_mov_b32 s8, scc // (op 581) 
s_cmp_eq_i32 s8, 0 // (op 581) 
s_cselect_b32 s9, 0, -1 // (op 581) 

s_and_b32 s8, s49, 31 // (op 581) 
s_lshl_b32 s18, 1, s8 // (op 581) 
s_add_i32 s19, s18, s9 // (op 581) 


s_and_b32 s9, s0, s19 // (op 581) 


s_add_i32 s0, s51, s9 // (op 581) 


s_ashr_i32 s9, s0, s8 // (op 581) 


s_lshl_b32 s0, s49, 24 // (op 581) 
s_ashr_i32 s8, s0, 31 // (op 581) 

s_xor_b32 s0, s9, s8 // (op 581) 

s_sub_i32 s9, s0, s8 // (op 581) 


s_add_u32 s18, s2, 32 // (op 581) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 581) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 581) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 581) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 581) 

s_mul_i32 s0, s46, s18 // (op 581) 

s_mul_i32 s8, s50, s0 // (op 581) 

s_mul_i32 s0, s9, s8 // (op 581) 

s_sub_u32 s8, s16, s0 // (op 581) 

s_mov_b32 s0, s8 // (op 581) 
s_mul_hi_i32 s51, s0, s15 // (op 581) 
s_add_i32 s18, s51, s0 // (op 581) 


s_ashr_i32 s0, s18, 31 // (op 581) 
s_cmp_eq_i32 s15, 0 // (op 581) 
s_mov_b32 s19, scc // (op 581) 
s_cmp_eq_i32 s19, 0 // (op 581) 
s_cselect_b32 s51, 0, -1 // (op 581) 

s_and_b32 s19, s48, 31 // (op 581) 
s_lshl_b32 s60, 1, s19 // (op 581) 
s_add_i32 s61, s60, s51 // (op 581) 


s_and_b32 s51, s0, s61 // (op 581) 


s_add_i32 s0, s18, s51 // (op 581) 


s_ashr_i32 s18, s0, s19 // (op 581) 


s_lshl_b32 s0, s48, 24 // (op 581) 
s_ashr_i32 s19, s0, 31 // (op 581) 

s_xor_b32 s0, s18, s19 // (op 581) 

s_sub_i32 s18, s0, s19 // (op 581) 


s_mul_i32 s0, s47, s46 // (op 581) 
s_add_u32 s60, s6, 16 // (op 581) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 581) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 581) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 581) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 581) 

s_sub_i32 s19, s60, s0 // (op 581) 

s_mul_i32 s51, s18, s19 // (op 581) 


s_sub_u32 s18, s8, s51 // (op 581) 


s_add_u32 s8, s18, s0 // (op 581) 


s_cmp_eq_i32 s9, 0 // (op 581) 
s_mov_b32 s0, scc // (op 581) 

s_cmp_eq_i32 s0, 0 // (op 581) 
s_cselect_b32 s9, s8, s1 // (op 581) 



v_lshl_add_u32 v12, s9, 4, v16 // (op 581) 


v_mul_lo_u32 v16, v12, s28 // (op 581) 
v_add_u32 v16, v16, v13 // (op 581) 


s_mul_i32 s1, s28, s7 // (op 583) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 583) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 583) least significant
s_mul_hi_u32 s8, s28, s6 // (op 583) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 583) most significant: sum
s_add_u32 s1, s1, s8 // (op 583) most significant: sum


s_mov_b32 s60, s30 // (op 583) reg expression
s_mov_b32 s61, s31 // (op 583) reg expression
s_mov_b32 s62, s0 // (op 583) reg expression
s_mov_b32 s63, 131072 // (op 583) call()

s_mul_hi_i32 s51, s16, s11 // (op 600) 
v_lshrrev_b32 v12, 7, v1 // (op 600) 
s_add_i32 s0, s51, s16 // (op 600) 

v_and_b32 v13, 15, v1 // (op 600) 
s_ashr_i32 s1, s0, 31 // (op 600) 
v_lshl_add_u32 v17, v12, 4, v13 // (op 600) 


s_cmp_eq_i32 s11, 0 // (op 600) 
s_mov_b32 s51, scc // (op 600) 
v_and_b32 v12, 63, v1 // (op 600) 
s_cmp_eq_i32 s51, 0 // (op 600) 
s_cselect_b32 s8, 0, -1 // (op 600) 

v_lshrrev_b32 v13, 4, v12 // (op 600) 

s_and_b32 s9, s12, 31 // (op 600) 
s_lshl_b32 s51, 1, s9 // (op 600) 
s_add_i32 s18, s51, s8 // (op 600) 


s_and_b32 s8, s1, s18 // (op 600) 


s_add_i32 s1, s0, s8 // (op 600) 


s_ashr_i32 s0, s1, s9 // (op 600) 


s_lshl_b32 s1, s12, 24 // (op 600) 
s_ashr_i32 s51, s1, 31 // (op 600) 

s_xor_b32 s1, s0, s51 // (op 600) 

s_sub_i32 s0, s1, s51 // (op 600) 


s_mul_hi_i32 s1, s0, s13 // (op 600) 
s_add_i32 s51, s1, s0 // (op 600) 

s_ashr_i32 s1, s51, 31 // (op 600) 
s_cmp_eq_i32 s13, 0 // (op 600) 
s_mov_b32 s8, scc // (op 600) 
s_cmp_eq_i32 s8, 0 // (op 600) 
s_cselect_b32 s9, 0, -1 // (op 600) 

s_and_b32 s8, s14, 31 // (op 600) 
s_lshl_b32 s18, 1, s8 // (op 600) 
s_add_i32 s19, s18, s9 // (op 600) 


s_and_b32 s9, s1, s19 // (op 600) 


s_add_i32 s1, s51, s9 // (op 600) 


s_ashr_i32 s9, s1, s8 // (op 600) 


s_lshl_b32 s1, s14, 24 // (op 600) 
s_ashr_i32 s8, s1, 31 // (op 600) 

s_xor_b32 s1, s9, s8 // (op 600) 

s_sub_i32 s9, s1, s8 // (op 600) 


s_mul_i32 s1, s0, s46 // (op 600) 

s_sub_u32 s0, s16, s1 // (op 600) 

s_mul_i32 s1, s9, s46 // (op 600) 
s_add_u32 s1, s1, s0 // (op 600) 


s_mul_hi_i32 s0, s16, s10 // (op 600) 
s_add_i32 s51, s0, s16 // (op 600) 

s_ashr_i32 s0, s51, 31 // (op 600) 
s_cmp_eq_i32 s10, 0 // (op 600) 
s_mov_b32 s8, scc // (op 600) 
s_cmp_eq_i32 s8, 0 // (op 600) 
s_cselect_b32 s9, 0, -1 // (op 600) 

s_and_b32 s8, s49, 31 // (op 600) 
s_lshl_b32 s18, 1, s8 // (op 600) 
s_add_i32 s19, s18, s9 // (op 600) 


s_and_b32 s9, s0, s19 // (op 600) 


s_add_i32 s0, s51, s9 // (op 600) 


s_ashr_i32 s9, s0, s8 // (op 600) 


s_lshl_b32 s0, s49, 24 // (op 600) 
s_ashr_i32 s8, s0, 31 // (op 600) 

s_xor_b32 s0, s9, s8 // (op 600) 

s_sub_i32 s9, s0, s8 // (op 600) 


s_add_u32 s18, s2, 32 // (op 600) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 600) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 600) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 600) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 600) 

s_mul_i32 s0, s46, s18 // (op 600) 

s_mul_i32 s8, s50, s0 // (op 600) 

s_mul_i32 s0, s9, s8 // (op 600) 

s_sub_u32 s8, s16, s0 // (op 600) 

s_mov_b32 s0, s8 // (op 600) 
s_mul_hi_i32 s51, s0, s15 // (op 600) 
s_add_i32 s18, s51, s0 // (op 600) 


s_ashr_i32 s0, s18, 31 // (op 600) 
s_cmp_eq_i32 s15, 0 // (op 600) 
s_mov_b32 s19, scc // (op 600) 
s_cmp_eq_i32 s19, 0 // (op 600) 
s_cselect_b32 s51, 0, -1 // (op 600) 

s_and_b32 s19, s48, 31 // (op 600) 
s_lshl_b32 s64, 1, s19 // (op 600) 
s_add_i32 s65, s64, s51 // (op 600) 


s_and_b32 s51, s0, s65 // (op 600) 


s_add_i32 s0, s18, s51 // (op 600) 


s_ashr_i32 s18, s0, s19 // (op 600) 


s_lshl_b32 s0, s48, 24 // (op 600) 
s_ashr_i32 s19, s0, 31 // (op 600) 

s_xor_b32 s0, s18, s19 // (op 600) 

s_sub_i32 s18, s0, s19 // (op 600) 


s_mul_i32 s0, s47, s46 // (op 600) 
s_add_u32 s64, s6, 16 // (op 600) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 600) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 600) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 600) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 600) 

s_sub_i32 s19, s64, s0 // (op 600) 

s_mul_i32 s51, s18, s19 // (op 600) 


s_sub_u32 s18, s8, s51 // (op 600) 


s_add_u32 s8, s18, s0 // (op 600) 


s_cmp_eq_i32 s9, 0 // (op 600) 
s_mov_b32 s0, scc // (op 600) 

s_cmp_eq_i32 s0, 0 // (op 600) 
s_cselect_b32 s9, s8, s1 // (op 600) 



v_lshl_add_u32 v12, s9, 4, v17 // (op 600) 


v_mul_lo_u32 v17, v12, s28 // (op 600) 
v_add_u32 v17, v17, v13 // (op 600) 


s_mul_i32 s1, s28, s7 // (op 602) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 602) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 602) least significant
s_mul_hi_u32 s8, s28, s6 // (op 602) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 602) most significant: sum
s_add_u32 s1, s1, s8 // (op 602) most significant: sum


s_mov_b32 s60, s30 // (op 602) reg expression
s_mov_b32 s61, s31 // (op 602) reg expression
s_mov_b32 s62, s0 // (op 602) reg expression
s_mov_b32 s63, 131072 // (op 602) call()

s_mul_hi_i32 s51, s16, s11 // (op 645) 
v_and_b32 v12, -32, v1 // (op 645) 
s_add_i32 s0, s51, s16 // (op 645) 

v_sub_u32 v13, v1, v12 // (op 645) 

s_ashr_i32 s1, s0, 31 // (op 645) 
v_lshrrev_b32 v12, 5, v1 // (op 645) 
s_cmp_eq_i32 s11, 0 // (op 645) 
s_mov_b32 s51, scc // (op 645) 
s_cmp_eq_i32 s51, 0 // (op 645) 
s_cselect_b32 s8, 0, -1 // (op 645) 

s_and_b32 s9, s12, 31 // (op 645) 
s_lshl_b32 s51, 1, s9 // (op 645) 
s_add_i32 s18, s51, s8 // (op 645) 


s_and_b32 s8, s1, s18 // (op 645) 


s_add_i32 s1, s0, s8 // (op 645) 


s_ashr_i32 s0, s1, s9 // (op 645) 


s_lshl_b32 s1, s12, 24 // (op 645) 
s_ashr_i32 s51, s1, 31 // (op 645) 

s_xor_b32 s1, s0, s51 // (op 645) 

s_sub_i32 s0, s1, s51 // (op 645) 


s_mul_hi_i32 s1, s0, s13 // (op 645) 
s_add_i32 s51, s1, s0 // (op 645) 

s_ashr_i32 s1, s51, 31 // (op 645) 
s_cmp_eq_i32 s13, 0 // (op 645) 
s_mov_b32 s8, scc // (op 645) 
s_cmp_eq_i32 s8, 0 // (op 645) 
s_cselect_b32 s9, 0, -1 // (op 645) 

s_and_b32 s8, s14, 31 // (op 645) 
s_lshl_b32 s18, 1, s8 // (op 645) 
s_add_i32 s19, s18, s9 // (op 645) 


s_and_b32 s9, s1, s19 // (op 645) 


s_add_i32 s1, s51, s9 // (op 645) 


s_ashr_i32 s9, s1, s8 // (op 645) 


s_lshl_b32 s1, s14, 24 // (op 645) 
s_ashr_i32 s8, s1, 31 // (op 645) 

s_xor_b32 s1, s9, s8 // (op 645) 

s_sub_i32 s9, s1, s8 // (op 645) 


s_add_u32 s18, s2, 32 // (op 645) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 645) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 645) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 645) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 645) 

s_mul_i32 s1, s9, s18 // (op 645) 

s_sub_i32 s51, s0, s1 // (op 645) 


s_mul_hi_i32 s0, s16, s10 // (op 645) 
s_add_i32 s1, s0, s16 // (op 645) 

s_ashr_i32 s0, s1, 31 // (op 645) 
s_cmp_eq_i32 s10, 0 // (op 645) 
s_mov_b32 s8, scc // (op 645) 
s_cmp_eq_i32 s8, 0 // (op 645) 
s_cselect_b32 s9, 0, -1 // (op 645) 

s_and_b32 s8, s49, 31 // (op 645) 
s_lshl_b32 s64, 1, s8 // (op 645) 
s_add_i32 s65, s64, s9 // (op 645) 


s_and_b32 s9, s0, s65 // (op 645) 


s_add_i32 s0, s1, s9 // (op 645) 


s_ashr_i32 s1, s0, s8 // (op 645) 


s_lshl_b32 s0, s49, 24 // (op 645) 
s_ashr_i32 s8, s0, 31 // (op 645) 

s_xor_b32 s0, s1, s8 // (op 645) 

s_sub_i32 s1, s0, s8 // (op 645) 


s_mul_i32 s0, s46, s18 // (op 645) 

s_mul_i32 s8, s50, s0 // (op 645) 

s_mul_i32 s0, s1, s8 // (op 645) 

s_sub_u32 s8, s16, s0 // (op 645) 

s_mul_hi_i32 s0, s8, s15 // (op 645) 
s_add_i32 s9, s0, s8 // (op 645) 


s_ashr_i32 s0, s9, 31 // (op 645) 
s_cmp_eq_i32 s15, 0 // (op 645) 
s_mov_b32 s8, scc // (op 645) 
s_cmp_eq_i32 s8, 0 // (op 645) 
s_cselect_b32 s18, 0, -1 // (op 645) 

s_and_b32 s8, s48, 31 // (op 645) 
s_lshl_b32 s19, 1, s8 // (op 645) 
s_add_i32 s64, s19, s18 // (op 645) 


s_and_b32 s18, s0, s64 // (op 645) 


s_add_i32 s0, s9, s18 // (op 645) 


s_ashr_i32 s9, s0, s8 // (op 645) 


s_lshl_b32 s0, s48, 24 // (op 645) 
s_ashr_i32 s8, s0, 31 // (op 645) 

s_xor_b32 s0, s9, s8 // (op 645) 

s_sub_i32 s9, s0, s8 // (op 645) 


s_cmp_eq_i32 s1, 0 // (op 645) 
s_mov_b32 s0, scc // (op 645) 

s_cmp_eq_i32 s0, 0 // (op 645) 
s_cselect_b32 s1, s9, s51 // (op 645) 



v_lshl_add_u32 v18, s1, 5, v13 // (op 645) 


v_mul_lo_u32 v13, v12, s32 // (op 645) 
v_add_u32 v13, v13, v18 // (op 645) 


s_mul_i32 s1, s32, s37 // (op 647) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 647) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 647) least significant
s_mul_hi_u32 s8, s32, s36 // (op 647) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 647) most significant: sum
s_add_u32 s1, s1, s8 // (op 647) most significant: sum


s_mov_b32 s64, s34 // (op 647) reg expression
s_mov_b32 s65, s35 // (op 647) reg expression
s_mov_b32 s66, s0 // (op 647) reg expression
s_mov_b32 s67, 131072 // (op 647) call()

s_mul_hi_i32 s51, s16, s11 // (op 664) 
v_and_b32 v12, -32, v1 // (op 664) 
s_add_i32 s0, s51, s16 // (op 664) 

v_sub_u32 v18, v1, v12 // (op 664) 

s_ashr_i32 s1, s0, 31 // (op 664) 
v_lshrrev_b32 v12, 5, v1 // (op 664) 
s_cmp_eq_i32 s11, 0 // (op 664) 
s_mov_b32 s51, scc // (op 664) 
s_cmp_eq_i32 s51, 0 // (op 664) 
s_cselect_b32 s8, 0, -1 // (op 664) 

s_and_b32 s9, s12, 31 // (op 664) 
s_lshl_b32 s51, 1, s9 // (op 664) 
s_add_i32 s18, s51, s8 // (op 664) 


s_and_b32 s8, s1, s18 // (op 664) 


s_add_i32 s1, s0, s8 // (op 664) 


s_ashr_i32 s0, s1, s9 // (op 664) 


s_lshl_b32 s1, s12, 24 // (op 664) 
s_ashr_i32 s51, s1, 31 // (op 664) 

s_xor_b32 s1, s0, s51 // (op 664) 

s_sub_i32 s0, s1, s51 // (op 664) 


s_mul_hi_i32 s1, s0, s13 // (op 664) 
s_add_i32 s51, s1, s0 // (op 664) 

s_ashr_i32 s1, s51, 31 // (op 664) 
s_cmp_eq_i32 s13, 0 // (op 664) 
s_mov_b32 s8, scc // (op 664) 
s_cmp_eq_i32 s8, 0 // (op 664) 
s_cselect_b32 s9, 0, -1 // (op 664) 

s_and_b32 s8, s14, 31 // (op 664) 
s_lshl_b32 s18, 1, s8 // (op 664) 
s_add_i32 s19, s18, s9 // (op 664) 


s_and_b32 s9, s1, s19 // (op 664) 


s_add_i32 s1, s51, s9 // (op 664) 


s_ashr_i32 s9, s1, s8 // (op 664) 


s_lshl_b32 s1, s14, 24 // (op 664) 
s_ashr_i32 s8, s1, 31 // (op 664) 

s_xor_b32 s1, s9, s8 // (op 664) 

s_sub_i32 s9, s1, s8 // (op 664) 


s_add_u32 s18, s2, 32 // (op 664) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 664) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 664) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 664) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 664) 

s_mul_i32 s1, s9, s18 // (op 664) 

s_sub_i32 s51, s0, s1 // (op 664) 


s_mul_hi_i32 s0, s16, s10 // (op 664) 
s_add_i32 s1, s0, s16 // (op 664) 

s_ashr_i32 s0, s1, 31 // (op 664) 
s_cmp_eq_i32 s10, 0 // (op 664) 
s_mov_b32 s8, scc // (op 664) 
s_cmp_eq_i32 s8, 0 // (op 664) 
s_cselect_b32 s9, 0, -1 // (op 664) 

s_and_b32 s8, s49, 31 // (op 664) 
s_lshl_b32 s68, 1, s8 // (op 664) 
s_add_i32 s69, s68, s9 // (op 664) 


s_and_b32 s9, s0, s69 // (op 664) 


s_add_i32 s0, s1, s9 // (op 664) 


s_ashr_i32 s1, s0, s8 // (op 664) 


s_lshl_b32 s0, s49, 24 // (op 664) 
s_ashr_i32 s8, s0, 31 // (op 664) 

s_xor_b32 s0, s1, s8 // (op 664) 

s_sub_i32 s1, s0, s8 // (op 664) 


s_mul_i32 s0, s46, s18 // (op 664) 

s_mul_i32 s8, s50, s0 // (op 664) 

s_mul_i32 s0, s1, s8 // (op 664) 

s_sub_u32 s8, s16, s0 // (op 664) 

s_mul_hi_i32 s0, s8, s15 // (op 664) 
s_add_i32 s9, s0, s8 // (op 664) 


s_ashr_i32 s0, s9, 31 // (op 664) 
s_cmp_eq_i32 s15, 0 // (op 664) 
s_mov_b32 s8, scc // (op 664) 
s_cmp_eq_i32 s8, 0 // (op 664) 
s_cselect_b32 s18, 0, -1 // (op 664) 

s_and_b32 s8, s48, 31 // (op 664) 
s_lshl_b32 s19, 1, s8 // (op 664) 
s_add_i32 s68, s19, s18 // (op 664) 


s_and_b32 s18, s0, s68 // (op 664) 


s_add_i32 s0, s9, s18 // (op 664) 


s_ashr_i32 s9, s0, s8 // (op 664) 


s_lshl_b32 s0, s48, 24 // (op 664) 
s_ashr_i32 s8, s0, 31 // (op 664) 

s_xor_b32 s0, s9, s8 // (op 664) 

s_sub_i32 s9, s0, s8 // (op 664) 


s_cmp_eq_i32 s1, 0 // (op 664) 
s_mov_b32 s0, scc // (op 664) 

s_cmp_eq_i32 s0, 0 // (op 664) 
s_cselect_b32 s1, s9, s51 // (op 664) 



v_lshl_add_u32 v19, s1, 5, v18 // (op 664) 


v_mul_lo_u32 v18, v12, s32 // (op 664) 
v_add_u32 v18, v18, v19 // (op 664) 


s_mul_i32 s1, s32, s37 // (op 666) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 666) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 666) least significant
s_mul_hi_u32 s8, s32, s36 // (op 666) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 666) most significant: sum
s_add_u32 s1, s1, s8 // (op 666) most significant: sum


s_mov_b32 s64, s34 // (op 666) reg expression
s_mov_b32 s65, s35 // (op 666) reg expression
s_mov_b32 s66, s0 // (op 666) reg expression
s_mov_b32 s67, 131072 // (op 666) call()




buffer_load_ubyte v12, v16, s[60:63], 0 offen offset: 0 // (op 337) Load value


v_mov_b32 v19, v14 // (op 341) reg expression

buffer_load_dwordx4 v[20:23], v19, s[56:59], 0 offen // (op 341) Load value



v_mov_b32 v19, v13 // (op 344) reg expression

buffer_load_ubyte v24, v19, s[64:67], 0 offen offset: 0 // (op 344) Load value




buffer_load_dwordx4 v[26:29], v10, s[52:55], 0 offen // (op 347) Load value



buffer_load_ubyte v19, v16, s[60:63], 0 offen offset:4 // (op 351) Load value



s_lshl_b64 s[0:1], s[22:23], 7 // (op 355) 
s_lshl_b64 s[8:9], s[0:1], 2 // (op 355) 

s_lshr_b64 s[0:1], s[8:9], 3 // (op 355) 

v_add_u32 v16, v14, s0 // (op 355) 


buffer_load_dwordx4 v[30:33], v16, s[56:59], 0 offen // (op 355) Load value




s_lshl_b32 s51, s32, 2 // (op 358) 
v_add_u32 v14, v13, s51 // (op 358) 


buffer_load_ubyte v16, v14, s[64:67], 0 offen offset: 0 // (op 358) Load value





buffer_load_dwordx4 v[34:37], v10, s[52:55], 0 offen offset:128 // (op 361) Load value

v_lshlrev_b32 v10, 7, v1 // (op 784) 
v_lshrrev_b32 v25, 3, v10 // (op 784) 




s_waitcnt vmcnt(6)
ds_write_b128 v25, v[20:23] // (op 363) Store local data 

v_and_b32 v10, -32, v1 // (op 837) 
v_sub_u32 v25, v1, v10 // (op 837) 

v_and_b32 v10, 4294967264, v1 // (op 837) 
v_add_u32 v13, v10, v25 // (op 837) 





s_waitcnt vmcnt(5)
ds_write_b8 v13, v24 offset:2048 // (op 364) Store local data 

v_and_b32 v10, -8, v1 // (op 731) 
v_sub_u32 v25, v1, v10 // (op 731) 

v_lshlrev_b32 v10, 4, v25 // (op 731) 

v_lshrrev_b32 v25, 3, v1 // (op 731) 
v_lshl_add_u32 v13, v25, 7, v10 // (op 731) 





s_waitcnt vmcnt(4)
ds_write_b128 v13, v[26:29] offset:2176 // (op 365) Store local data 

v_and_b32 v10, -8, v1 // (op 702) 
v_sub_u32 v25, v1, v10 // (op 702) 

v_lshlrev_b32 v10, 4, v25 // (op 702) 

v_lshrrev_b32 v25, 3, v1 // (op 702) 
v_lshl_add_u32 v13, v25, 7, v10 // (op 702) 


v_lshlrev_b32 v10, 7, v1 // (op 755) 
v_lshrrev_b32 v14, 3, v10 // (op 755) 

v_and_b32 v10, -32, v1 // (op 808) 
v_sub_u32 v25, v1, v10 // (op 808) 

v_and_b32 v10, 4294967264, v1 // (op 808) 
v_add_u32 v38, v10, v25 // (op 808) 


v_and_b32 v10, -8, v1 // (op 895) 
v_sub_u32 v25, v1, v10 // (op 895) 

v_lshlrev_b32 v10, 4, v25 // (op 895) 

v_lshrrev_b32 v25, 3, v1 // (op 895) 
v_lshl_add_u32 v39, v25, 7, v10 // (op 895) 


v_lshlrev_b32 v10, 7, v1 // (op 919) 
v_lshrrev_b32 v25, 3, v10 // (op 919) 

v_and_b32 v10, -32, v1 // (op 943) 
v_sub_u32 v40, v1, v10 // (op 943) 

v_and_b32 v10, 4294967264, v1 // (op 943) 
v_add_u32 v41, v10, v40 // (op 943) 


s_mov_b32 s51, 0 // (op 55) call()
s_ashr_i64 s[0:1], s[26:27], 7 // (op 54) 
s_mov_b32 s8, s0 // (op 54) convert

s_and_b32 s9, s8, 4294967294 // (op 54) 

s_cmp_lt_u32 s51, s9 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54 if false)
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54:
s_add_u32 s51, s51, 2 // (op 56) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 988) 



ds_read_b128 v[42:45], v3 offset:2176 // (op 65) Load local data 
ds_read_b128 v[46:49], v3 offset:2240 // (op 65) Load local data 
s_barrier  // (op 1001) 



ds_read_b64_tr_b4 v[50:51], v4 // (op 70) Transpose load local data 
ds_read_b64_tr_b4 v[52:53], v4 offset:256 // (op 70) Transpose load local data 
s_barrier  // (op 1015) 


ds_read_u8 v10, v0 offset:2048 // (op 79) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:53], a[0:3], v12, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 82) 





v_mov_b32 v10, v17 // (op 75) 

buffer_load_ubyte v12, v10, s[60:63], 0 offen offset:8 // (op 75) Load value


v_mov_b32 v40, v15 // (op 10) 
s_lshl_b64 s[0:1], s[22:23], 7 // (op 10) 
s_lshl_b64 s[8:9], s[0:1], 2 // (op 10) 

s_lshr_b64 s[0:1], s[8:9], 3 // (op 10) 

s_lshl_b32 s8, s0, 1 // (op 10) 

v_add_u32 v42, v40, s8 // (op 10) 


buffer_load_dwordx4 v[20:23], v42, s[56:59], 0 offen // (op 10) Load value



v_mov_b32 v42, v18 // (op 21) 
s_lshl_b32 s0, s32, 3 // (op 21) 
v_add_u32 v43, v42, s0 // (op 21) 


buffer_load_ubyte v24, v43, s[64:67], 0 offen offset: 0 // (op 21) Load value



v_mov_b32 v43, v11 // (op 4) 

buffer_load_dwordx4 v[26:29], v43, s[52:55], 0 offen offset:256 // (op 4) Load value
s_barrier  // (op 1038) 



s_waitcnt vmcnt(6)
ds_write_b128 v25, v[30:33] offset:4224 // (op 235) Store local data 
s_barrier  // (op 1044) 



s_waitcnt vmcnt(5)
ds_write_b8 v41, v16 offset:6272 // (op 236) Store local data 
s_barrier  // (op 1032) 



s_waitcnt vmcnt(4)
ds_write_b128 v39, v[34:37] offset:6400 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1029) 



ds_read_b128 v[44:47], v6 offset:6400 // (op 227) Load local data 
ds_read_b128 v[48:51], v6 offset:6464 // (op 227) Load local data 
s_barrier  // (op 1035) 
s_barrier  // (op 1095) 



ds_read_b64_tr_b4 v[52:53], v7 offset:4224 // (op 229) Transpose load local data 
ds_read_b64_tr_b4 v[54:55], v7 offset:4480 // (op 229) Transpose load local data 
s_barrier  // (op 1041) 
s_barrier  // (op 1101) 


ds_read_u8 v56, v5 offset:6272 // (op 233) Load local data 
s_barrier  // (op 1109) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[52:55], a[0:3], v19, v56 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 234) 






buffer_load_ubyte v19, v10, s[60:63], 0 offen offset:12 // (op 231) Load value



v_add_u32 v17, 8, v17 // (op 592) 
s_lshl_b64 s[0:1], s[22:23], 7 // (op 237) 
s_lshl_b64 s[8:9], s[0:1], 2 // (op 237) 

s_lshr_b64 s[0:1], s[8:9], 3 // (op 237) 

s_mul_i32 s8, 3, s0 // (op 237) 

v_add_u32 v10, v40, s8 // (op 237) 


buffer_load_dwordx4 v[30:33], v10, s[56:59], 0 offen // (op 237) Load value




s_lshl_b32 s0, s32, 2 // (op 238) 
s_mul_i32 s1, 3, s0 // (op 238) 

v_add_u32 v10, v42, s1 // (op 238) 


s_lshl_b64 s[0:1], s[22:23], 7 // (op 528) 
s_lshl_b64 s[8:9], s[0:1], 2 // (op 528) 

s_lshr_b64 s[0:1], s[8:9], 3 // (op 528) 

s_lshl_b32 s8, s0, 1 // (op 528) 

v_add_u32 v15, v15, s8 // (op 528) 

buffer_load_ubyte v16, v10, s[64:67], 0 offen offset: 0 // (op 238) Load value





s_lshl_b32 s0, s32, 3 // (op 656) 
v_add_u32 v18, v18, s0 // (op 656) 

buffer_load_dwordx4 v[34:37], v43, s[52:55], 0 offen offset:384 // (op 224) Load value

s_barrier  // (op 1004) 
v_add_u32 v11, 256, v11 // (op 464) 



s_waitcnt vmcnt(6)
ds_write_b128 v14, v[20:23] // (op 13) Store local data 
s_barrier  // (op 1018) 



s_waitcnt vmcnt(5)
ds_write_b8 v38, v24 offset:2048 // (op 26) Store local data 
s_barrier  // (op 991) 



s_waitcnt vmcnt(4)
ds_write_b128 v13, v[26:29] offset:2176 // (op 6) Store local data 
s_ashr_i64 s[0:1], s[26:27], 7 // (op 54) 
s_mov_b32 s8, s0 // (op 54) convert

s_and_b32 s9, s8, 4294967294 // (op 54) 

s_cmp_lt_u32 s51, s9 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54 if true)
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54:















s_mul_hi_i32 s51, s16, s11 // (op 427) 
v_lshrrev_b32 v10, 3, v1 // (op 427) 
s_add_i32 s0, s51, s16 // (op 427) 

v_and_b32 v11, -8, v1 // (op 427) 
s_ashr_i32 s1, s0, 31 // (op 427) 
v_sub_u32 v12, v1, v11 // (op 427) 

s_cmp_eq_i32 s11, 0 // (op 427) 
s_mov_b32 s51, scc // (op 427) 
v_lshlrev_b32 v11, 4, v12 // (op 427) 

s_cmp_eq_i32 s51, 0 // (op 427) 
s_cselect_b32 s8, 0, -1 // (op 427) 

s_and_b32 s9, s12, 31 // (op 427) 
s_lshl_b32 s51, 1, s9 // (op 427) 
s_add_i32 s18, s51, s8 // (op 427) 


s_and_b32 s8, s1, s18 // (op 427) 


s_add_i32 s1, s0, s8 // (op 427) 


s_ashr_i32 s0, s1, s9 // (op 427) 


s_lshl_b32 s1, s12, 24 // (op 427) 
s_ashr_i32 s51, s1, 31 // (op 427) 

s_xor_b32 s1, s0, s51 // (op 427) 

s_sub_i32 s0, s1, s51 // (op 427) 


s_mul_hi_i32 s1, s0, s13 // (op 427) 
s_add_i32 s51, s1, s0 // (op 427) 

s_ashr_i32 s1, s51, 31 // (op 427) 
s_cmp_eq_i32 s13, 0 // (op 427) 
s_mov_b32 s8, scc // (op 427) 
s_cmp_eq_i32 s8, 0 // (op 427) 
s_cselect_b32 s9, 0, -1 // (op 427) 

s_and_b32 s8, s14, 31 // (op 427) 
s_lshl_b32 s18, 1, s8 // (op 427) 
s_add_i32 s19, s18, s9 // (op 427) 


s_and_b32 s9, s1, s19 // (op 427) 


s_add_i32 s1, s51, s9 // (op 427) 


s_ashr_i32 s9, s1, s8 // (op 427) 


s_lshl_b32 s1, s14, 24 // (op 427) 
s_ashr_i32 s8, s1, 31 // (op 427) 

s_xor_b32 s1, s9, s8 // (op 427) 

s_sub_i32 s9, s1, s8 // (op 427) 


s_mul_i32 s1, s0, s46 // (op 427) 

s_sub_u32 s0, s16, s1 // (op 427) 

s_mul_i32 s1, s9, s46 // (op 427) 
s_add_u32 s1, s1, s0 // (op 427) 


s_mul_hi_i32 s0, s16, s10 // (op 427) 
s_add_i32 s51, s0, s16 // (op 427) 

s_ashr_i32 s0, s51, 31 // (op 427) 
s_cmp_eq_i32 s10, 0 // (op 427) 
s_mov_b32 s8, scc // (op 427) 
s_cmp_eq_i32 s8, 0 // (op 427) 
s_cselect_b32 s9, 0, -1 // (op 427) 

s_and_b32 s8, s49, 31 // (op 427) 
s_lshl_b32 s18, 1, s8 // (op 427) 
s_add_i32 s19, s18, s9 // (op 427) 


s_and_b32 s9, s0, s19 // (op 427) 


s_add_i32 s0, s51, s9 // (op 427) 


s_ashr_i32 s9, s0, s8 // (op 427) 


s_lshl_b32 s0, s49, 24 // (op 427) 
s_ashr_i32 s8, s0, 31 // (op 427) 

s_xor_b32 s0, s9, s8 // (op 427) 

s_sub_i32 s9, s0, s8 // (op 427) 


s_add_u32 s18, s2, 32 // (op 427) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 427) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 427) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 427) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 427) 

s_mul_i32 s0, s46, s18 // (op 427) 

s_mul_i32 s8, s50, s0 // (op 427) 

s_mul_i32 s0, s9, s8 // (op 427) 

s_sub_u32 s8, s16, s0 // (op 427) 

s_mov_b32 s0, s8 // (op 427) 
s_mul_hi_i32 s51, s0, s15 // (op 427) 
s_add_i32 s18, s51, s0 // (op 427) 


s_ashr_i32 s0, s18, 31 // (op 427) 
s_cmp_eq_i32 s15, 0 // (op 427) 
s_mov_b32 s19, scc // (op 427) 
s_cmp_eq_i32 s19, 0 // (op 427) 
s_cselect_b32 s51, 0, -1 // (op 427) 

s_and_b32 s19, s48, 31 // (op 427) 
s_lshl_b32 s68, 1, s19 // (op 427) 
s_add_i32 s69, s68, s51 // (op 427) 


s_and_b32 s51, s0, s69 // (op 427) 


s_add_i32 s0, s18, s51 // (op 427) 


s_ashr_i32 s18, s0, s19 // (op 427) 


s_lshl_b32 s0, s48, 24 // (op 427) 
s_ashr_i32 s19, s0, 31 // (op 427) 

s_xor_b32 s0, s18, s19 // (op 427) 

s_sub_i32 s18, s0, s19 // (op 427) 


s_mul_i32 s0, s47, s46 // (op 427) 
s_add_u32 s68, s6, 16 // (op 427) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 427) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 427) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 427) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 427) 

s_sub_i32 s19, s68, s0 // (op 427) 

s_mul_i32 s51, s18, s19 // (op 427) 


s_sub_u32 s18, s8, s51 // (op 427) 


s_add_u32 s8, s18, s0 // (op 427) 


s_cmp_eq_i32 s9, 0 // (op 427) 
s_mov_b32 s0, scc // (op 427) 

s_cmp_eq_i32 s0, 0 // (op 427) 
s_cselect_b32 s9, s8, s1 // (op 427) 



v_lshl_add_u32 v12, s9, 4, v10 // (op 427) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 427) 
s_lshr_b32 s51, s0, 1 // (op 427) 

v_lshl_add_u32 v10, s51, 8, v11 // (op 427) 


v_mul_lo_u32 v11, v12, s4 // (op 427) 
v_add_u32 v11, v11, v10 // (op 427) 


s_mul_i32 s1, s4, s7 // (op 429) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 429) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 429) least significant
s_mul_hi_u32 s8, s4, s6 // (op 429) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 429) most significant: sum
s_add_u32 s1, s1, s8 // (op 429) most significant: sum


s_mov_b32 s52, s20 // (op 429) reg expression
s_mov_b32 s53, s21 // (op 429) reg expression
s_mov_b32 s54, s0 // (op 429) reg expression
s_mov_b32 s55, 131072 // (op 429) call()

s_mul_hi_i32 s51, s16, s11 // (op 494) 
s_add_i32 s0, s51, s16 // (op 494) 

s_ashr_i32 s1, s0, 31 // (op 494) 
s_cmp_eq_i32 s11, 0 // (op 494) 
s_mov_b32 s51, scc // (op 494) 
s_cmp_eq_i32 s51, 0 // (op 494) 
s_cselect_b32 s8, 0, -1 // (op 494) 

s_and_b32 s9, s12, 31 // (op 494) 
s_lshl_b32 s51, 1, s9 // (op 494) 
s_add_i32 s18, s51, s8 // (op 494) 


s_and_b32 s8, s1, s18 // (op 494) 


s_add_i32 s1, s0, s8 // (op 494) 


s_ashr_i32 s0, s1, s9 // (op 494) 


s_lshl_b32 s1, s12, 24 // (op 494) 
s_ashr_i32 s51, s1, 31 // (op 494) 

s_xor_b32 s1, s0, s51 // (op 494) 

s_sub_i32 s0, s1, s51 // (op 494) 


s_mul_hi_i32 s1, s0, s13 // (op 494) 
s_add_i32 s51, s1, s0 // (op 494) 

s_ashr_i32 s1, s51, 31 // (op 494) 
s_cmp_eq_i32 s13, 0 // (op 494) 
s_mov_b32 s8, scc // (op 494) 
s_cmp_eq_i32 s8, 0 // (op 494) 
s_cselect_b32 s9, 0, -1 // (op 494) 

s_and_b32 s8, s14, 31 // (op 494) 
s_lshl_b32 s18, 1, s8 // (op 494) 
s_add_i32 s19, s18, s9 // (op 494) 


s_and_b32 s9, s1, s19 // (op 494) 


s_add_i32 s1, s51, s9 // (op 494) 


s_ashr_i32 s9, s1, s8 // (op 494) 


s_lshl_b32 s1, s14, 24 // (op 494) 
s_ashr_i32 s8, s1, 31 // (op 494) 

s_xor_b32 s1, s9, s8 // (op 494) 

s_sub_i32 s9, s1, s8 // (op 494) 


s_add_u32 s18, s2, 32 // (op 494) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 494) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 494) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 494) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 494) 

s_mul_i32 s1, s9, s18 // (op 494) 

s_sub_i32 s51, s0, s1 // (op 494) 


s_mul_hi_i32 s0, s16, s10 // (op 494) 
s_add_i32 s1, s0, s16 // (op 494) 

s_ashr_i32 s0, s1, 31 // (op 494) 
s_cmp_eq_i32 s10, 0 // (op 494) 
s_mov_b32 s8, scc // (op 494) 
s_cmp_eq_i32 s8, 0 // (op 494) 
s_cselect_b32 s9, 0, -1 // (op 494) 

s_and_b32 s8, s49, 31 // (op 494) 
s_lshl_b32 s68, 1, s8 // (op 494) 
s_add_i32 s69, s68, s9 // (op 494) 


s_and_b32 s9, s0, s69 // (op 494) 


s_add_i32 s0, s1, s9 // (op 494) 


s_ashr_i32 s1, s0, s8 // (op 494) 


s_lshl_b32 s0, s49, 24 // (op 494) 
s_ashr_i32 s8, s0, 31 // (op 494) 

s_xor_b32 s0, s1, s8 // (op 494) 

s_sub_i32 s1, s0, s8 // (op 494) 


s_mul_i32 s0, s46, s18 // (op 494) 

s_mul_i32 s8, s50, s0 // (op 494) 

s_mul_i32 s0, s1, s8 // (op 494) 

s_sub_u32 s8, s16, s0 // (op 494) 

s_mul_hi_i32 s0, s8, s15 // (op 494) 
s_add_i32 s9, s0, s8 // (op 494) 


s_ashr_i32 s0, s9, 31 // (op 494) 
s_cmp_eq_i32 s15, 0 // (op 494) 
s_mov_b32 s8, scc // (op 494) 
s_cmp_eq_i32 s8, 0 // (op 494) 
s_cselect_b32 s18, 0, -1 // (op 494) 

s_and_b32 s8, s48, 31 // (op 494) 
s_lshl_b32 s19, 1, s8 // (op 494) 
s_add_i32 s68, s19, s18 // (op 494) 


s_and_b32 s18, s0, s68 // (op 494) 


s_add_i32 s0, s9, s18 // (op 494) 


s_ashr_i32 s9, s0, s8 // (op 494) 


s_lshl_b32 s0, s48, 24 // (op 494) 
s_ashr_i32 s8, s0, 31 // (op 494) 

s_xor_b32 s0, s9, s8 // (op 494) 

s_sub_i32 s9, s0, s8 // (op 494) 


s_cmp_eq_i32 s1, 0 // (op 494) 
s_mov_b32 s0, scc // (op 494) 

s_cmp_eq_i32 s0, 0 // (op 494) 
s_cselect_b32 s1, s9, s51 // (op 494) 



s_lshl_b32 s0, s1, 5 // (op 494) 

s_mov_b32 s9, 0 // (op 494) convert
s_mov_b32 s8, s0 // (op 494) convert

s_ashr_i64 s[0:1], s[26:27], 7 // (op 494) 
s_lshr_b32 s51, s0, 1 // (op 494) 

v_lshl_add_u32 v10, s51, 8, v1 // (op 494) 

v_mul_lo_u32 v13, v10, s23 // (op 494) most significant: low of low * high
v_mul_hi_u32 v14, v10, s22 // (op 494) most significant: high of low * low
v_mul_lo_u32 v12, v10, s22 // (op 494) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 494) most significant: sum

v_mov_b32 v14, s8 // (op 494) 
v_mov_b32 v15, s9 // (op 494) 
v_add_co_u32 v12, s[0:1], v12, v14 // (op 494) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 494) most significant half





v_lshlrev_b64 v[14:15], 2, v[12:13] // (op 494) 

v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 494) 

v_mov_b32 v10, v12 // (op 494) convert

s_mul_i32 s1, s22, s27 // (op 496) most significant: low of low * high
s_mul_i32 s51, s23, s26 // (op 496) most significant: low of high * low
s_mul_i32 s0, s22, s26 // (op 496) least significant
s_mul_hi_u32 s8, s22, s26 // (op 496) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 496) most significant: sum
s_add_u32 s1, s1, s8 // (op 496) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 496) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 496) 

s_mov_b32 s56, s24 // (op 496) reg expression
s_mov_b32 s57, s25 // (op 496) reg expression
s_mov_b32 s58, s0 // (op 496) reg expression
s_mov_b32 s59, 131072 // (op 496) call()

s_mul_hi_i32 s51, s16, s11 // (op 558) 
v_lshrrev_b32 v12, 7, v1 // (op 558) 
s_add_i32 s0, s51, s16 // (op 558) 

v_and_b32 v13, 15, v1 // (op 558) 
s_ashr_i32 s1, s0, 31 // (op 558) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 558) 


s_cmp_eq_i32 s11, 0 // (op 558) 
s_mov_b32 s51, scc // (op 558) 
v_and_b32 v15, 63, v1 // (op 558) 
s_cmp_eq_i32 s51, 0 // (op 558) 
s_cselect_b32 s8, 0, -1 // (op 558) 

v_lshrrev_b32 v12, 4, v15 // (op 558) 

s_and_b32 s9, s12, 31 // (op 558) 
s_lshl_b32 s51, 1, s9 // (op 558) 
s_add_i32 s18, s51, s8 // (op 558) 


s_and_b32 s8, s1, s18 // (op 558) 


s_add_i32 s1, s0, s8 // (op 558) 


s_ashr_i32 s0, s1, s9 // (op 558) 


s_lshl_b32 s1, s12, 24 // (op 558) 
s_ashr_i32 s51, s1, 31 // (op 558) 

s_xor_b32 s1, s0, s51 // (op 558) 

s_sub_i32 s0, s1, s51 // (op 558) 


s_mul_hi_i32 s1, s0, s13 // (op 558) 
s_add_i32 s51, s1, s0 // (op 558) 

s_ashr_i32 s1, s51, 31 // (op 558) 
s_cmp_eq_i32 s13, 0 // (op 558) 
s_mov_b32 s8, scc // (op 558) 
s_cmp_eq_i32 s8, 0 // (op 558) 
s_cselect_b32 s9, 0, -1 // (op 558) 

s_and_b32 s8, s14, 31 // (op 558) 
s_lshl_b32 s18, 1, s8 // (op 558) 
s_add_i32 s19, s18, s9 // (op 558) 


s_and_b32 s9, s1, s19 // (op 558) 


s_add_i32 s1, s51, s9 // (op 558) 


s_ashr_i32 s9, s1, s8 // (op 558) 


s_lshl_b32 s1, s14, 24 // (op 558) 
s_ashr_i32 s8, s1, 31 // (op 558) 

s_xor_b32 s1, s9, s8 // (op 558) 

s_sub_i32 s9, s1, s8 // (op 558) 


s_mul_i32 s1, s0, s46 // (op 558) 

s_sub_u32 s0, s16, s1 // (op 558) 

s_mul_i32 s1, s9, s46 // (op 558) 
s_add_u32 s1, s1, s0 // (op 558) 


s_mul_hi_i32 s0, s16, s10 // (op 558) 
s_add_i32 s51, s0, s16 // (op 558) 

s_ashr_i32 s0, s51, 31 // (op 558) 
s_cmp_eq_i32 s10, 0 // (op 558) 
s_mov_b32 s8, scc // (op 558) 
s_cmp_eq_i32 s8, 0 // (op 558) 
s_cselect_b32 s9, 0, -1 // (op 558) 

s_and_b32 s8, s49, 31 // (op 558) 
s_lshl_b32 s18, 1, s8 // (op 558) 
s_add_i32 s19, s18, s9 // (op 558) 


s_and_b32 s9, s0, s19 // (op 558) 


s_add_i32 s0, s51, s9 // (op 558) 


s_ashr_i32 s9, s0, s8 // (op 558) 


s_lshl_b32 s0, s49, 24 // (op 558) 
s_ashr_i32 s8, s0, 31 // (op 558) 

s_xor_b32 s0, s9, s8 // (op 558) 

s_sub_i32 s9, s0, s8 // (op 558) 


s_add_u32 s18, s2, 32 // (op 558) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 558) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 558) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 558) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 558) 

s_mul_i32 s0, s46, s18 // (op 558) 

s_mul_i32 s8, s50, s0 // (op 558) 

s_mul_i32 s0, s9, s8 // (op 558) 

s_sub_u32 s8, s16, s0 // (op 558) 

s_mov_b32 s0, s8 // (op 558) 
s_mul_hi_i32 s51, s0, s15 // (op 558) 
s_add_i32 s18, s51, s0 // (op 558) 


s_ashr_i32 s0, s18, 31 // (op 558) 
s_cmp_eq_i32 s15, 0 // (op 558) 
s_mov_b32 s19, scc // (op 558) 
s_cmp_eq_i32 s19, 0 // (op 558) 
s_cselect_b32 s51, 0, -1 // (op 558) 

s_and_b32 s19, s48, 31 // (op 558) 
s_lshl_b32 s68, 1, s19 // (op 558) 
s_add_i32 s69, s68, s51 // (op 558) 


s_and_b32 s51, s0, s69 // (op 558) 


s_add_i32 s0, s18, s51 // (op 558) 


s_ashr_i32 s18, s0, s19 // (op 558) 


s_lshl_b32 s0, s48, 24 // (op 558) 
s_ashr_i32 s19, s0, 31 // (op 558) 

s_xor_b32 s0, s18, s19 // (op 558) 

s_sub_i32 s18, s0, s19 // (op 558) 


s_mul_i32 s0, s47, s46 // (op 558) 
s_add_u32 s68, s6, 16 // (op 558) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 558) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 558) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 558) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 558) 

s_sub_i32 s19, s68, s0 // (op 558) 

s_mul_i32 s51, s18, s19 // (op 558) 


s_sub_u32 s18, s8, s51 // (op 558) 


s_add_u32 s8, s18, s0 // (op 558) 


s_cmp_eq_i32 s9, 0 // (op 558) 
s_mov_b32 s0, scc // (op 558) 

s_cmp_eq_i32 s0, 0 // (op 558) 
s_cselect_b32 s9, s8, s1 // (op 558) 



v_lshl_add_u32 v13, s9, 4, v14 // (op 558) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 558) 
s_lshr_b32 s51, s0, 1 // (op 558) 

v_lshl_add_u32 v14, s51, 3, v12 // (op 558) 


v_mul_lo_u32 v12, v13, s28 // (op 558) 
v_add_u32 v12, v12, v14 // (op 558) 


s_mul_i32 s1, s28, s7 // (op 560) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 560) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 560) least significant
s_mul_hi_u32 s8, s28, s6 // (op 560) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 560) most significant: sum
s_add_u32 s1, s1, s8 // (op 560) most significant: sum


s_mov_b32 s60, s30 // (op 560) reg expression
s_mov_b32 s61, s31 // (op 560) reg expression
s_mov_b32 s62, s0 // (op 560) reg expression
s_mov_b32 s63, 131072 // (op 560) call()

s_mul_hi_i32 s51, s16, s11 // (op 622) 
v_and_b32 v13, -32, v1 // (op 622) 
s_add_i32 s0, s51, s16 // (op 622) 

v_sub_u32 v14, v1, v13 // (op 622) 

s_ashr_i32 s1, s0, 31 // (op 622) 
v_lshrrev_b32 v13, 5, v1 // (op 622) 
s_cmp_eq_i32 s11, 0 // (op 622) 
s_mov_b32 s51, scc // (op 622) 
s_cmp_eq_i32 s51, 0 // (op 622) 
s_cselect_b32 s8, 0, -1 // (op 622) 

s_and_b32 s9, s12, 31 // (op 622) 
s_lshl_b32 s51, 1, s9 // (op 622) 
s_add_i32 s18, s51, s8 // (op 622) 


s_and_b32 s8, s1, s18 // (op 622) 


s_add_i32 s1, s0, s8 // (op 622) 


s_ashr_i32 s0, s1, s9 // (op 622) 


s_lshl_b32 s1, s12, 24 // (op 622) 
s_ashr_i32 s51, s1, 31 // (op 622) 

s_xor_b32 s1, s0, s51 // (op 622) 

s_sub_i32 s0, s1, s51 // (op 622) 


s_mul_hi_i32 s1, s0, s13 // (op 622) 
s_add_i32 s51, s1, s0 // (op 622) 

s_ashr_i32 s1, s51, 31 // (op 622) 
s_cmp_eq_i32 s13, 0 // (op 622) 
s_mov_b32 s8, scc // (op 622) 
s_cmp_eq_i32 s8, 0 // (op 622) 
s_cselect_b32 s9, 0, -1 // (op 622) 

s_and_b32 s8, s14, 31 // (op 622) 
s_lshl_b32 s18, 1, s8 // (op 622) 
s_add_i32 s19, s18, s9 // (op 622) 


s_and_b32 s9, s1, s19 // (op 622) 


s_add_i32 s1, s51, s9 // (op 622) 


s_ashr_i32 s9, s1, s8 // (op 622) 


s_lshl_b32 s1, s14, 24 // (op 622) 
s_ashr_i32 s8, s1, 31 // (op 622) 

s_xor_b32 s1, s9, s8 // (op 622) 

s_sub_i32 s9, s1, s8 // (op 622) 


s_add_u32 s18, s2, 32 // (op 622) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 622) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 622) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 622) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 622) 

s_mul_i32 s1, s9, s18 // (op 622) 

s_sub_i32 s51, s0, s1 // (op 622) 


s_mul_hi_i32 s0, s16, s10 // (op 622) 
s_add_i32 s1, s0, s16 // (op 622) 

s_ashr_i32 s0, s1, 31 // (op 622) 
s_cmp_eq_i32 s10, 0 // (op 622) 
s_mov_b32 s8, scc // (op 622) 
s_cmp_eq_i32 s8, 0 // (op 622) 
s_cselect_b32 s9, 0, -1 // (op 622) 

s_and_b32 s8, s49, 31 // (op 622) 
s_lshl_b32 s68, 1, s8 // (op 622) 
s_add_i32 s69, s68, s9 // (op 622) 


s_and_b32 s9, s0, s69 // (op 622) 


s_add_i32 s0, s1, s9 // (op 622) 


s_ashr_i32 s1, s0, s8 // (op 622) 


s_lshl_b32 s0, s49, 24 // (op 622) 
s_ashr_i32 s8, s0, 31 // (op 622) 

s_xor_b32 s0, s1, s8 // (op 622) 

s_sub_i32 s1, s0, s8 // (op 622) 


s_mul_i32 s0, s46, s18 // (op 622) 

s_mul_i32 s8, s50, s0 // (op 622) 

s_mul_i32 s0, s1, s8 // (op 622) 

s_sub_u32 s8, s16, s0 // (op 622) 

s_mul_hi_i32 s0, s8, s15 // (op 622) 
s_add_i32 s9, s0, s8 // (op 622) 


s_ashr_i32 s0, s9, 31 // (op 622) 
s_cmp_eq_i32 s15, 0 // (op 622) 
s_mov_b32 s8, scc // (op 622) 
s_cmp_eq_i32 s8, 0 // (op 622) 
s_cselect_b32 s18, 0, -1 // (op 622) 

s_and_b32 s8, s48, 31 // (op 622) 
s_lshl_b32 s19, 1, s8 // (op 622) 
s_add_i32 s68, s19, s18 // (op 622) 


s_and_b32 s18, s0, s68 // (op 622) 


s_add_i32 s0, s9, s18 // (op 622) 


s_ashr_i32 s9, s0, s8 // (op 622) 


s_lshl_b32 s0, s48, 24 // (op 622) 
s_ashr_i32 s8, s0, 31 // (op 622) 

s_xor_b32 s0, s9, s8 // (op 622) 

s_sub_i32 s9, s0, s8 // (op 622) 


s_cmp_eq_i32 s1, 0 // (op 622) 
s_mov_b32 s0, scc // (op 622) 

s_cmp_eq_i32 s0, 0 // (op 622) 
s_cselect_b32 s1, s9, s51 // (op 622) 



v_lshl_add_u32 v15, s1, 5, v14 // (op 622) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 622) 
s_lshr_b32 s51, s0, 1 // (op 622) 

v_lshl_add_u32 v14, s51, 3, v13 // (op 622) 


v_mul_lo_u32 v13, v14, s32 // (op 622) 
v_add_u32 v13, v13, v15 // (op 622) 


s_mul_i32 s1, s32, s37 // (op 624) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 624) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 624) least significant
s_mul_hi_u32 s8, s32, s36 // (op 624) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 624) most significant: sum
s_add_u32 s1, s1, s8 // (op 624) most significant: sum


s_mov_b32 s64, s34 // (op 624) reg expression
s_mov_b32 s65, s35 // (op 624) reg expression
s_mov_b32 s66, s0 // (op 624) reg expression
s_mov_b32 s67, 131072 // (op 624) call()

v_and_b32 v14, -8, v1 // (op 715) 
v_sub_u32 v15, v1, v14 // (op 715) 

v_lshlrev_b32 v14, 4, v15 // (op 715) 

v_lshrrev_b32 v15, 3, v1 // (op 715) 
v_lshl_add_u32 v17, v15, 7, v14 // (op 715) 


v_lshlrev_b32 v18, 7, v1 // (op 768) 
v_lshrrev_b32 v14, 3, v18 // (op 768) 

v_and_b32 v15, -32, v1 // (op 821) 
v_sub_u32 v18, v1, v15 // (op 821) 

v_and_b32 v15, 4294967264, v1 // (op 821) 
v_add_u32 v20, v15, v18 // (op 821) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 162) 
s_and_b32 s51, s0, 4294967294 // (op 162) 

s_ashr_i64 s[0:1], s[26:27], 7 // (op 161) 
s_mov_b32 s8, s0 // (op 161) convert

s_cmp_lt_u32 s51, s8 // (op 161) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161:


s_add_u32 s51, s51, 1 // (op 163) 
v_mov_b32 v15, v11 // (op 166) 

buffer_load_dwordx4 v[22:25], v15, s[52:55], 0 offen // (op 166) Load value

v_add_u32 v11, 128, v11 // (op 417) 


s_barrier  // (op 994) 
v_mov_b32 v21, v10 // (op 179) 



v_mov_b32 v15, v21 // (op 179) reg expression

buffer_load_dwordx4 v[26:29], v15, s[56:59], 0 offen // (op 179) Load value


s_lshl_b64 s[0:1], s[22:23], 7 // (op 486) 
s_lshl_b64 s[8:9], s[0:1], 2 // (op 486) 

s_lshr_b64 s[0:1], s[8:9], 3 // (op 486) 

v_add_u32 v10, v10, s0 // (op 486) 

s_waitcnt vmcnt(1)
ds_write_b128 v17, v[22:25] offset:2176 // (op 167) Store local data 



v_mov_b32 v15, v13 // (op 180) 
s_barrier  // (op 1007) 
v_mov_b32 v16, v15 // (op 180) reg expression




buffer_load_ubyte v18, v16, s[64:67], 0 offen offset: 0 // (op 180) Load value


s_lshl_b32 s0, s32, 2 // (op 614) 
v_add_u32 v13, v13, s0 // (op 614) 

s_waitcnt vmcnt(1)
ds_write_b128 v14, v[26:29] // (op 177) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 997) 



s_barrier  // (op 1021) 
ds_read_b128 v[22:25], v3 offset:2176 // (op 169) Load local data 



ds_read_b128 v[26:29], v3 offset:2240 // (op 169) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v18 offset:2048 // (op 178) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1011) 



ds_read_b64_tr_b4 v[30:31], v4 // (op 171) Transpose load local data 
ds_read_b64_tr_b4 v[32:33], v4 offset:256 // (op 171) Transpose load local data 


v_mov_b32 v21, v12 // (op 173) 

buffer_load_ubyte v15, v21, s[60:63], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v12, 4, v12 // (op 550) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1025) 


ds_read_u8 v16, v0 offset:2048 // (op 175) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:33], a[0:3], v15, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 176) 




s_ashr_i64 s[0:1], s[26:27], 7 // (op 161) 
s_mov_b32 s8, s0 // (op 161) convert

s_cmp_lt_u32 s51, s8 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161 if true)
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161:






















s_barrier  // (op 1089) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

s_mul_hi_i32 s0, s16, s11 // (op 863) 
v_lshrrev_b32 v0, 7, v1 // (op 863) 
s_add_i32 s1, s0, s16 // (op 863) 

v_and_b32 v3, 63, v1 // (op 863) 
s_ashr_i32 s0, s1, 31 // (op 863) 
v_lshrrev_b32 v4, 4, v3 // (op 863) 
s_cmp_eq_i32 s11, 0 // (op 863) 
s_mov_b32 s4, scc // (op 863) 
v_lshlrev_b32 v5, 2, v4 // (op 863) 

s_cmp_eq_i32 s4, 0 // (op 863) 
s_cselect_b32 s5, 0, -1 // (op 863) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 863) 


s_and_b32 s4, s12, 31 // (op 863) 
v_lshrrev_b32 v0, 2, v3 // (op 863) 

s_lshl_b32 s8, 1, s4 // (op 863) 
v_and_b32 v3, 3, v0 // (op 863) 

s_add_i32 s9, s8, s5 // (op 863) 


v_and_b32 v0, 3, v1 // (op 863) 
s_and_b32 s5, s0, s9 // (op 863) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 863) 


s_add_i32 s0, s1, s5 // (op 863) 


v_lshrrev_b32 v0, 6, v1 // (op 863) 
s_ashr_i32 s1, s0, s4 // (op 863) 


v_and_b32 v3, 1, v0 // (op 863) 

s_lshl_b32 s0, s12, 24 // (op 863) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 863) 


s_ashr_i32 s4, s0, 31 // (op 863) 

s_xor_b32 s0, s1, s4 // (op 863) 

s_sub_i32 s1, s0, s4 // (op 863) 


s_mul_hi_i32 s0, s1, s13 // (op 863) 
s_add_i32 s4, s0, s1 // (op 863) 

s_ashr_i32 s0, s4, 31 // (op 863) 
s_cmp_eq_i32 s13, 0 // (op 863) 
s_mov_b32 s5, scc // (op 863) 
s_cmp_eq_i32 s5, 0 // (op 863) 
s_cselect_b32 s8, 0, -1 // (op 863) 

s_and_b32 s5, s14, 31 // (op 863) 
s_lshl_b32 s9, 1, s5 // (op 863) 
s_add_i32 s18, s9, s8 // (op 863) 


s_and_b32 s8, s0, s18 // (op 863) 


s_add_i32 s0, s4, s8 // (op 863) 


s_ashr_i32 s4, s0, s5 // (op 863) 


s_lshl_b32 s0, s14, 24 // (op 863) 
s_ashr_i32 s5, s0, 31 // (op 863) 

s_xor_b32 s0, s4, s5 // (op 863) 

s_sub_i32 s4, s0, s5 // (op 863) 


s_mul_i32 s0, s1, s46 // (op 863) 
s_sub_u32 s5, s16, s0 // (op 863) 

s_mul_i32 s0, s4, s46 // (op 863) 
s_add_u32 s0, s0, s5 // (op 863) 

s_mul_hi_i32 s5, s16, s10 // (op 863) 
s_add_i32 s8, s5, s16 // (op 863) 

s_ashr_i32 s5, s8, 31 // (op 863) 
s_cmp_eq_i32 s10, 0 // (op 863) 
s_mov_b32 s9, scc // (op 863) 
s_cmp_eq_i32 s9, 0 // (op 863) 
s_cselect_b32 s18, 0, -1 // (op 863) 

s_and_b32 s9, s49, 31 // (op 863) 
s_lshl_b32 s19, 1, s9 // (op 863) 
s_add_i32 s20, s19, s18 // (op 863) 


s_and_b32 s18, s5, s20 // (op 863) 


s_add_i32 s5, s8, s18 // (op 863) 


s_ashr_i32 s8, s5, s9 // (op 863) 


s_lshl_b32 s5, s49, 24 // (op 863) 
s_ashr_i32 s9, s5, 31 // (op 863) 

s_xor_b32 s5, s8, s9 // (op 863) 

s_sub_i32 s8, s5, s9 // (op 863) 


s_add_u32 s18, s2, 32 // (op 863) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 863) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 863) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 863) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 863) 

s_mul_i32 s5, s46, s18 // (op 863) 
s_mul_i32 s9, s50, s5 // (op 863) 

s_mul_i32 s5, s8, s9 // (op 863) 

s_sub_u32 s9, s16, s5 // (op 863) 

s_mov_b32 s5, s9 // (op 863) 
s_mul_hi_i32 s20, s5, s15 // (op 863) 
s_add_i32 s21, s20, s5 // (op 863) 


s_ashr_i32 s5, s21, 31 // (op 863) 
s_cmp_eq_i32 s15, 0 // (op 863) 
s_mov_b32 s20, scc // (op 863) 
s_cmp_eq_i32 s20, 0 // (op 863) 
s_cselect_b32 s22, 0, -1 // (op 863) 

s_and_b32 s20, s48, 31 // (op 863) 
s_lshl_b32 s23, 1, s20 // (op 863) 
s_add_i32 s24, s23, s22 // (op 863) 


s_and_b32 s22, s5, s24 // (op 863) 


s_add_i32 s5, s21, s22 // (op 863) 


s_ashr_i32 s21, s5, s20 // (op 863) 


s_lshl_b32 s5, s48, 24 // (op 863) 
s_ashr_i32 s20, s5, 31 // (op 863) 

s_xor_b32 s5, s21, s20 // (op 863) 

s_sub_i32 s21, s5, s20 // (op 863) 


s_mul_i32 s5, s47, s46 // (op 863) 
s_add_u32 s22, s6, 16 // (op 863) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 863) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 863) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 863) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 863) 

s_sub_i32 s20, s22, s5 // (op 863) 

s_mul_i32 s22, s21, s20 // (op 863) 

s_sub_u32 s20, s9, s22 // (op 863) 


s_add_u32 s9, s20, s5 // (op 863) 


s_cmp_eq_i32 s8, 0 // (op 863) 
s_mov_b32 s5, scc // (op 863) 

s_cmp_eq_i32 s5, 0 // (op 863) 
s_cselect_b32 s8, s9, s0 // (op 863) 


s_mul_i32 s0, s4, s18 // (op 863) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 863) 


s_sub_i32 s4, s1, s0 // (op 863) 


s_cmp_eq_i32 s5, 0 // (op 863) 
s_cselect_b32 s0, s21, s4 // (op 863) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 863) 


v_mul_lo_u32 v0, v4, s38 // (op 863) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 863) 


s_mul_i32 s1, s38, s3 // (op 865) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 865) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 865) least significant
s_mul_hi_u32 s5, s38, s2 // (op 865) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 865) most significant: sum
s_add_u32 s1, s1, s5 // (op 865) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 865) 

s_mov_b32 s20, s40 // (op 865) reg expression
s_mov_b32 s21, s41 // (op 865) reg expression
s_mov_b32 s22, s4 // (op 865) reg expression
s_mov_b32 s23, 131072 // (op 865) call()




buffer_store_dwordx4 v[10:13], v4, s[20:23], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123:
s_mul_hi_i32 s0, s16, s11 // (op 685) 
v_lshrrev_b32 v0, 7, v1 // (op 685) 
s_add_i32 s1, s0, s16 // (op 685) 

v_and_b32 v3, 63, v1 // (op 685) 
s_ashr_i32 s0, s1, 31 // (op 685) 
v_lshrrev_b32 v4, 4, v3 // (op 685) 
s_cmp_eq_i32 s11, 0 // (op 685) 
s_mov_b32 s4, scc // (op 685) 
v_lshlrev_b32 v5, 2, v4 // (op 685) 

s_cmp_eq_i32 s4, 0 // (op 685) 
s_cselect_b32 s5, 0, -1 // (op 685) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 685) 


s_and_b32 s4, s12, 31 // (op 685) 
v_lshrrev_b32 v0, 2, v3 // (op 685) 

s_lshl_b32 s8, 1, s4 // (op 685) 
v_and_b32 v3, 3, v0 // (op 685) 

s_add_i32 s9, s8, s5 // (op 685) 


v_and_b32 v0, 3, v1 // (op 685) 
s_and_b32 s5, s0, s9 // (op 685) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 685) 


s_add_i32 s0, s1, s5 // (op 685) 


v_lshrrev_b32 v0, 6, v1 // (op 685) 
s_ashr_i32 s1, s0, s4 // (op 685) 


v_and_b32 v3, 1, v0 // (op 685) 

s_lshl_b32 s0, s12, 24 // (op 685) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 685) 


s_ashr_i32 s4, s0, 31 // (op 685) 

s_xor_b32 s0, s1, s4 // (op 685) 

s_sub_i32 s1, s0, s4 // (op 685) 


s_mul_hi_i32 s0, s1, s13 // (op 685) 
s_add_i32 s4, s0, s1 // (op 685) 

s_ashr_i32 s0, s4, 31 // (op 685) 
s_cmp_eq_i32 s13, 0 // (op 685) 
s_mov_b32 s5, scc // (op 685) 
s_cmp_eq_i32 s5, 0 // (op 685) 
s_cselect_b32 s8, 0, -1 // (op 685) 

s_and_b32 s5, s14, 31 // (op 685) 
s_lshl_b32 s9, 1, s5 // (op 685) 
s_add_i32 s18, s9, s8 // (op 685) 


s_and_b32 s19, s0, s18 // (op 685) 


s_add_i32 s0, s4, s19 // (op 685) 


s_ashr_i32 s4, s0, s5 // (op 685) 


s_lshl_b32 s0, s14, 24 // (op 685) 
s_ashr_i32 s5, s0, 31 // (op 685) 

s_xor_b32 s0, s4, s5 // (op 685) 

s_sub_i32 s4, s0, s5 // (op 685) 


s_mul_i32 s0, s1, s46 // (op 685) 
s_sub_u32 s5, s16, s0 // (op 685) 

s_mul_i32 s0, s4, s46 // (op 685) 
s_add_u32 s0, s0, s5 // (op 685) 

s_mul_hi_i32 s5, s16, s10 // (op 685) 
s_add_i32 s8, s5, s16 // (op 685) 

s_ashr_i32 s5, s8, 31 // (op 685) 
s_cmp_eq_i32 s10, 0 // (op 685) 
s_mov_b32 s9, scc // (op 685) 
s_cmp_eq_i32 s9, 0 // (op 685) 
s_cselect_b32 s18, 0, -1 // (op 685) 

s_and_b32 s9, s49, 31 // (op 685) 
s_lshl_b32 s19, 1, s9 // (op 685) 
s_add_i32 s24, s19, s18 // (op 685) 


s_and_b32 s18, s5, s24 // (op 685) 


s_add_i32 s5, s8, s18 // (op 685) 


s_ashr_i32 s8, s5, s9 // (op 685) 


s_lshl_b32 s5, s49, 24 // (op 685) 
s_ashr_i32 s9, s5, 31 // (op 685) 

s_xor_b32 s5, s8, s9 // (op 685) 

s_sub_i32 s8, s5, s9 // (op 685) 


s_add_u32 s18, s2, 32 // (op 685) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 685) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 685) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 685) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 5 // (op 685) 

s_mul_i32 s5, s46, s18 // (op 685) 
s_mul_i32 s9, s50, s5 // (op 685) 

s_mul_i32 s5, s8, s9 // (op 685) 

s_sub_u32 s9, s16, s5 // (op 685) 

s_mov_b32 s5, s9 // (op 685) 
s_mul_hi_i32 s24, s5, s15 // (op 685) 
s_add_i32 s25, s24, s5 // (op 685) 


s_ashr_i32 s5, s25, 31 // (op 685) 
s_cmp_eq_i32 s15, 0 // (op 685) 
s_mov_b32 s24, scc // (op 685) 
s_cmp_eq_i32 s24, 0 // (op 685) 
s_cselect_b32 s26, 0, -1 // (op 685) 

s_and_b32 s24, s48, 31 // (op 685) 
s_lshl_b32 s27, 1, s24 // (op 685) 
s_add_i32 s28, s27, s26 // (op 685) 


s_and_b32 s26, s5, s28 // (op 685) 


s_add_i32 s5, s25, s26 // (op 685) 


s_ashr_i32 s25, s5, s24 // (op 685) 


s_lshl_b32 s5, s48, 24 // (op 685) 
s_ashr_i32 s24, s5, 31 // (op 685) 

s_xor_b32 s5, s25, s24 // (op 685) 

s_sub_i32 s25, s5, s24 // (op 685) 


s_mul_i32 s5, s47, s46 // (op 685) 
s_add_u32 s26, s6, 16 // (op 685) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 685) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 685) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 685) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 685) 

s_sub_i32 s24, s26, s5 // (op 685) 

s_mul_i32 s26, s25, s24 // (op 685) 

s_sub_u32 s24, s9, s26 // (op 685) 


s_add_u32 s9, s24, s5 // (op 685) 


s_cmp_eq_i32 s8, 0 // (op 685) 
s_mov_b32 s5, scc // (op 685) 

s_cmp_eq_i32 s5, 0 // (op 685) 
s_cselect_b32 s8, s9, s0 // (op 685) 


s_mul_i32 s0, s4, s18 // (op 685) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 685) 


s_sub_i32 s4, s1, s0 // (op 685) 


s_cmp_eq_i32 s5, 0 // (op 685) 
s_cselect_b32 s24, s25, s4 // (op 685) 



v_lshl_add_u32 v4, s24, 5, v0 // (op 685) 


v_mul_lo_u32 v0, v4, s42 // (op 685) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 685) 


s_mul_i32 s1, s42, s3 // (op 687) most significant: low of low * high
s_mul_i32 s4, s43, s2 // (op 687) most significant: low of high * low
s_mul_i32 s0, s42, s2 // (op 687) least significant
s_mul_hi_u32 s5, s42, s2 // (op 687) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 687) most significant: sum
s_add_u32 s1, s1, s5 // (op 687) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 687) 

s_mov_b32 s24, s44 // (op 687) reg expression
s_mov_b32 s25, s45 // (op 687) reg expression
s_mov_b32 s26, s4 // (op 687) reg expression
s_mov_b32 s27, 131072 // (op 687) call()





buffer_load_dwordx4 v[10:13], v4, s[24:27], 0 offen // (op 135) Load value



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


s_mul_hi_i32 s0, s16, s11 // (op 880) 
v_lshrrev_b32 v0, 7, v1 // (op 880) 
s_add_i32 s1, s0, s16 // (op 880) 

v_and_b32 v3, 63, v1 // (op 880) 
s_ashr_i32 s0, s1, 31 // (op 880) 
v_lshrrev_b32 v4, 4, v3 // (op 880) 
s_cmp_eq_i32 s11, 0 // (op 880) 
s_mov_b32 s4, scc // (op 880) 
v_lshlrev_b32 v5, 2, v4 // (op 880) 

s_cmp_eq_i32 s4, 0 // (op 880) 
s_cselect_b32 s5, 0, -1 // (op 880) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 880) 


s_and_b32 s4, s12, 31 // (op 880) 
v_lshrrev_b32 v0, 2, v3 // (op 880) 

s_lshl_b32 s8, 1, s4 // (op 880) 
v_and_b32 v3, 3, v0 // (op 880) 

s_add_i32 s9, s8, s5 // (op 880) 


v_and_b32 v0, 3, v1 // (op 880) 
s_and_b32 s5, s0, s9 // (op 880) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 880) 


s_add_i32 s0, s1, s5 // (op 880) 


v_lshrrev_b32 v0, 6, v1 // (op 880) 
s_ashr_i32 s1, s0, s4 // (op 880) 


v_and_b32 v3, 1, v0 // (op 880) 

s_lshl_b32 s0, s12, 24 // (op 880) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 880) 


s_ashr_i32 s4, s0, 31 // (op 880) 

s_xor_b32 s0, s1, s4 // (op 880) 

s_sub_i32 s1, s0, s4 // (op 880) 


s_mul_hi_i32 s0, s1, s13 // (op 880) 
s_add_i32 s4, s0, s1 // (op 880) 

s_ashr_i32 s0, s4, 31 // (op 880) 
s_cmp_eq_i32 s13, 0 // (op 880) 
s_mov_b32 s5, scc // (op 880) 
s_cmp_eq_i32 s5, 0 // (op 880) 
s_cselect_b32 s8, 0, -1 // (op 880) 

s_and_b32 s5, s14, 31 // (op 880) 
s_lshl_b32 s9, 1, s5 // (op 880) 
s_add_i32 s18, s9, s8 // (op 880) 


s_and_b32 s19, s0, s18 // (op 880) 


s_add_i32 s0, s4, s19 // (op 880) 


s_ashr_i32 s4, s0, s5 // (op 880) 


s_lshl_b32 s0, s14, 24 // (op 880) 
s_ashr_i32 s5, s0, 31 // (op 880) 

s_xor_b32 s0, s4, s5 // (op 880) 

s_sub_i32 s4, s0, s5 // (op 880) 


s_mul_i32 s0, s1, s46 // (op 880) 
s_sub_u32 s5, s16, s0 // (op 880) 

s_mul_i32 s0, s4, s46 // (op 880) 
s_add_u32 s0, s0, s5 // (op 880) 

s_mul_hi_i32 s5, s16, s10 // (op 880) 
s_add_i32 s8, s5, s16 // (op 880) 

s_ashr_i32 s5, s8, 31 // (op 880) 
s_cmp_eq_i32 s10, 0 // (op 880) 
s_mov_b32 s9, scc // (op 880) 
s_cmp_eq_i32 s9, 0 // (op 880) 
s_cselect_b32 s18, 0, -1 // (op 880) 

s_and_b32 s9, s49, 31 // (op 880) 
s_lshl_b32 s19, 1, s9 // (op 880) 
s_add_i32 s24, s19, s18 // (op 880) 


s_and_b32 s18, s5, s24 // (op 880) 


s_add_i32 s5, s8, s18 // (op 880) 


s_ashr_i32 s8, s5, s9 // (op 880) 


s_lshl_b32 s5, s49, 24 // (op 880) 
s_ashr_i32 s9, s5, 31 // (op 880) 

s_xor_b32 s5, s8, s9 // (op 880) 

s_sub_i32 s8, s5, s9 // (op 880) 


s_add_u32 s18, s2, 32 // (op 880) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 880) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 880) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 880) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 5 // (op 880) 

s_mul_i32 s5, s46, s18 // (op 880) 
s_mul_i32 s9, s50, s5 // (op 880) 

s_mul_i32 s5, s8, s9 // (op 880) 

s_sub_u32 s9, s16, s5 // (op 880) 

s_mov_b32 s5, s9 // (op 880) 
s_mul_hi_i32 s24, s5, s15 // (op 880) 
s_add_i32 s25, s24, s5 // (op 880) 


s_ashr_i32 s5, s25, 31 // (op 880) 
s_cmp_eq_i32 s15, 0 // (op 880) 
s_mov_b32 s24, scc // (op 880) 
s_cmp_eq_i32 s24, 0 // (op 880) 
s_cselect_b32 s26, 0, -1 // (op 880) 

s_and_b32 s24, s48, 31 // (op 880) 
s_lshl_b32 s27, 1, s24 // (op 880) 
s_add_i32 s28, s27, s26 // (op 880) 


s_and_b32 s26, s5, s28 // (op 880) 


s_add_i32 s5, s25, s26 // (op 880) 


s_ashr_i32 s25, s5, s24 // (op 880) 


s_lshl_b32 s5, s48, 24 // (op 880) 
s_ashr_i32 s24, s5, 31 // (op 880) 

s_xor_b32 s5, s25, s24 // (op 880) 

s_sub_i32 s25, s5, s24 // (op 880) 


s_mul_i32 s5, s47, s46 // (op 880) 
s_add_u32 s26, s6, 16 // (op 880) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 880) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 880) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 880) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 880) 

s_sub_i32 s24, s26, s5 // (op 880) 

s_mul_i32 s26, s25, s24 // (op 880) 

s_sub_u32 s24, s9, s26 // (op 880) 


s_add_u32 s9, s24, s5 // (op 880) 


s_cmp_eq_i32 s8, 0 // (op 880) 
s_mov_b32 s5, scc // (op 880) 

s_cmp_eq_i32 s5, 0 // (op 880) 
s_cselect_b32 s8, s9, s0 // (op 880) 


s_mul_i32 s0, s4, s18 // (op 880) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 880) 


s_sub_i32 s4, s1, s0 // (op 880) 


s_cmp_eq_i32 s5, 0 // (op 880) 
s_cselect_b32 s24, s25, s4 // (op 880) 



v_lshl_add_u32 v4, s24, 5, v0 // (op 880) 


v_mul_lo_u32 v0, v4, s38 // (op 880) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 880) 


s_mul_i32 s1, s38, s3 // (op 882) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 882) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 882) least significant
s_mul_hi_u32 s5, s38, s2 // (op 882) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 882) most significant: sum
s_add_u32 s1, s1, s5 // (op 882) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 882) 

s_mov_b32 s20, s40 // (op 882) reg expression
s_mov_b32 s21, s41 // (op 882) reg expression
s_mov_b32 s22, s4 // (op 882) reg expression
s_mov_b32 s23, 131072 // (op 882) call()




buffer_store_dwordx4 v[10:13], v4, s[20:23], 0 offen // (op 134) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.LRR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end:
.size RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_, .LRR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end-RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 8448 // lds bytes
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: F4
    DestDataType: s
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
  - .name: RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 8448
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 57
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

