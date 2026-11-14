// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_,@function
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 48, v1 // (op 739) 
v_and_b32 v3, 63, v1 // (op 739) 
v_sub_u32 v4, v3, v0 // (op 739) 

v_lshrrev_b32 v3, 1, v4 // (op 739) 
v_add_lshl_u32 v5, v0, v3, 5 // (op 739) 


v_and_b32 v0, 1, v4 // (op 739) 

v_lshlrev_b32 v3, 3, v0 // (op 739) 

v_lshrrev_b32 v0, 6, v1 // (op 739) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 739) 


v_add_u32 v0, v5, v4 // (op 739) 


v_and_b32 v3, 15, v1 // (op 792) 
v_and_b32 v4, 63, v1 // (op 792) 
v_lshrrev_b32 v5, 4, v4 // (op 792) 

v_lshlrev_b32 v4, 5, v5 // (op 792) 

v_lshl_add_u32 v5, v3, 7, v4 // (op 792) 


v_lshlrev_b32 v3, 2, v5 // (op 792) 

v_lshrrev_b32 v4, 3, v3 // (op 792) 

v_lshrrev_b32 v3, 6, v1 // (op 845) 
v_and_b32 v5, 15, v1 // (op 845) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 845) 


v_and_b32 v3, 63, v1 // (op 845) 
v_lshrrev_b32 v5, 4, v3 // (op 845) 

v_lshl_add_u32 v3, v5, 5, v6 // (op 845) 


v_and_b32 v5, 48, v1 // (op 903) 
v_and_b32 v6, 63, v1 // (op 903) 
v_sub_u32 v7, v6, v5 // (op 903) 

v_lshrrev_b32 v6, 1, v7 // (op 903) 
v_add_lshl_u32 v8, v5, v6, 5 // (op 903) 


v_and_b32 v5, 1, v7 // (op 903) 

v_lshlrev_b32 v6, 3, v5 // (op 903) 

v_lshrrev_b32 v5, 6, v1 // (op 903) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 903) 


v_add_u32 v5, v8, v7 // (op 903) 


v_and_b32 v6, 15, v1 // (op 927) 
v_and_b32 v7, 63, v1 // (op 927) 
v_lshrrev_b32 v8, 4, v7 // (op 927) 

v_lshlrev_b32 v7, 5, v8 // (op 927) 

v_lshl_add_u32 v8, v6, 7, v7 // (op 927) 


v_lshlrev_b32 v6, 2, v8 // (op 927) 

v_lshrrev_b32 v7, 3, v6 // (op 927) 

v_lshrrev_b32 v6, 6, v1 // (op 951) 
v_and_b32 v8, 15, v1 // (op 951) 
v_lshl_add_u32 v9, v6, 4, v8 // (op 951) 


v_and_b32 v6, 63, v1 // (op 951) 
v_lshrrev_b32 v8, 4, v6 // (op 951) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 951) 


v_mov_b32 v8, s6 // (op 2) Move value
v_mov_b32 v9, s7 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
s_mul_hi_i32 s0, s16, s9 // (op 449) 
v_and_b32 v10, -2, v1 // (op 449) 
s_add_i32 s1, s0, s16 // (op 449) 

v_sub_u32 v11, v1, v10 // (op 449) 

s_ashr_i32 s0, s1, 31 // (op 449) 
v_lshlrev_b32 v10, 4, v11 // (op 449) 

s_cmp_eq_i32 s9, 0 // (op 449) 
s_mov_b32 s6, scc // (op 449) 
v_and_b32 v11, 4294967294, v1 // (op 449) 
s_cmp_eq_i32 s6, 0 // (op 449) 
s_cselect_b32 s7, 0, -1 // (op 449) 

s_and_b32 s6, s10, 31 // (op 449) 
s_lshl_b32 s18, 1, s6 // (op 449) 
s_add_i32 s19, s18, s7 // (op 449) 


s_and_b32 s7, s0, s19 // (op 449) 


s_add_i32 s0, s1, s7 // (op 449) 


s_ashr_i32 s1, s0, s6 // (op 449) 


s_lshl_b32 s0, s10, 24 // (op 449) 
s_ashr_i32 s6, s0, 31 // (op 449) 

s_xor_b32 s0, s1, s6 // (op 449) 

s_sub_i32 s1, s0, s6 // (op 449) 


s_mul_hi_i32 s0, s1, s11 // (op 449) 
s_add_i32 s6, s0, s1 // (op 449) 

s_ashr_i32 s0, s6, 31 // (op 449) 
s_cmp_eq_i32 s11, 0 // (op 449) 
s_mov_b32 s7, scc // (op 449) 
s_cmp_eq_i32 s7, 0 // (op 449) 
s_cselect_b32 s18, 0, -1 // (op 449) 

s_and_b32 s7, s12, 31 // (op 449) 
s_lshl_b32 s19, 1, s7 // (op 449) 
s_add_i32 s51, s19, s18 // (op 449) 


s_and_b32 s18, s0, s51 // (op 449) 


s_add_i32 s0, s6, s18 // (op 449) 


s_ashr_i32 s6, s0, s7 // (op 449) 


s_lshl_b32 s0, s12, 24 // (op 449) 
s_ashr_i32 s7, s0, 31 // (op 449) 

s_xor_b32 s0, s6, s7 // (op 449) 

s_sub_i32 s6, s0, s7 // (op 449) 


s_mul_i32 s0, s1, s14 // (op 449) 

s_sub_u32 s1, s16, s0 // (op 449) 

s_mul_i32 s0, s6, s14 // (op 449) 
s_add_u32 s0, s0, s1 // (op 449) 


s_mul_hi_i32 s1, s16, s8 // (op 449) 
s_add_i32 s6, s1, s16 // (op 449) 

s_ashr_i32 s1, s6, 31 // (op 449) 
s_cmp_eq_i32 s8, 0 // (op 449) 
s_mov_b32 s7, scc // (op 449) 
s_cmp_eq_i32 s7, 0 // (op 449) 
s_cselect_b32 s18, 0, -1 // (op 449) 

s_and_b32 s7, s15, 31 // (op 449) 
s_lshl_b32 s19, 1, s7 // (op 449) 
s_add_i32 s51, s19, s18 // (op 449) 


s_and_b32 s18, s1, s51 // (op 449) 


s_add_i32 s1, s6, s18 // (op 449) 


s_ashr_i32 s6, s1, s7 // (op 449) 


s_lshl_b32 s1, s15, 24 // (op 449) 
s_ashr_i32 s7, s1, 31 // (op 449) 

s_xor_b32 s1, s6, s7 // (op 449) 

s_sub_i32 s6, s1, s7 // (op 449) 


s_add_u32 s18, s4, 16 // (op 449) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 449) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 449) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 449) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 449) 

s_mul_i32 s1, s14, s18 // (op 449) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 449) 

s_mul_i32 s1, s6, s7 // (op 449) 

s_sub_u32 s7, s16, s1 // (op 449) 

s_mov_b32 s1, s7 // (op 449) 
s_mul_hi_i32 s18, s1, s13 // (op 449) 
s_add_i32 s19, s18, s1 // (op 449) 


s_ashr_i32 s1, s19, 31 // (op 449) 
s_cmp_eq_i32 s13, 0 // (op 449) 
s_mov_b32 s18, scc // (op 449) 
s_cmp_eq_i32 s18, 0 // (op 449) 
s_cselect_b32 s51, 0, -1 // (op 449) 

s_and_b32 s18, s49, 31 // (op 449) 
s_lshl_b32 s52, 1, s18 // (op 449) 
s_add_i32 s53, s52, s51 // (op 449) 


s_and_b32 s51, s1, s53 // (op 449) 


s_add_i32 s1, s19, s51 // (op 449) 


s_ashr_i32 s19, s1, s18 // (op 449) 


s_lshl_b32 s1, s49, 24 // (op 449) 
s_ashr_i32 s18, s1, 31 // (op 449) 

s_xor_b32 s1, s19, s18 // (op 449) 

s_sub_i32 s19, s1, s18 // (op 449) 


s_mul_i32 s1, s48, s14 // (op 449) 
s_add_u32 s52, s20, 32 // (op 449) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 449) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 449) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 449) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 5 // (op 449) 

s_sub_i32 s18, s52, s1 // (op 449) 

s_mul_i32 s51, s19, s18 // (op 449) 


s_sub_u32 s18, s7, s51 // (op 449) 


s_add_u32 s7, s18, s1 // (op 449) 


s_cmp_eq_i32 s6, 0 // (op 449) 
s_mov_b32 s1, scc // (op 449) 

s_cmp_eq_i32 s1, 0 // (op 449) 
s_cselect_b32 s6, s7, s0 // (op 449) 



v_lshl_add_u32 v12, s6, 5, v10 // (op 449) 


v_mul_lo_u32 v10, v11, s2 // (op 449) 

v_add_u32 v11, v12, v10 // (op 449) 


s_mul_i32 s1, s2, s25 // (op 451) most significant: low of low * high
s_mul_i32 s6, s3, s24 // (op 451) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 451) least significant
s_mul_hi_u32 s7, s2, s24 // (op 451) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 451) most significant: sum
s_add_u32 s1, s1, s7 // (op 451) most significant: sum


s_mov_b32 s52, s22 // (op 451) reg expression
s_mov_b32 s53, s23 // (op 451) reg expression
s_mov_b32 s54, s0 // (op 451) reg expression
s_mov_b32 s55, 131072 // (op 451) call()

s_mul_hi_i32 s51, s16, s9 // (op 468) 
v_and_b32 v10, -2, v1 // (op 468) 
s_add_i32 s0, s51, s16 // (op 468) 

v_sub_u32 v12, v1, v10 // (op 468) 

s_ashr_i32 s1, s0, 31 // (op 468) 
v_lshlrev_b32 v10, 4, v12 // (op 468) 

s_cmp_eq_i32 s9, 0 // (op 468) 
s_mov_b32 s51, scc // (op 468) 
v_and_b32 v12, 4294967294, v1 // (op 468) 
s_cmp_eq_i32 s51, 0 // (op 468) 
s_cselect_b32 s6, 0, -1 // (op 468) 

s_and_b32 s7, s10, 31 // (op 468) 
s_lshl_b32 s51, 1, s7 // (op 468) 
s_add_i32 s18, s51, s6 // (op 468) 


s_and_b32 s6, s1, s18 // (op 468) 


s_add_i32 s1, s0, s6 // (op 468) 


s_ashr_i32 s0, s1, s7 // (op 468) 


s_lshl_b32 s1, s10, 24 // (op 468) 
s_ashr_i32 s51, s1, 31 // (op 468) 

s_xor_b32 s1, s0, s51 // (op 468) 

s_sub_i32 s0, s1, s51 // (op 468) 


s_mul_hi_i32 s1, s0, s11 // (op 468) 
s_add_i32 s51, s1, s0 // (op 468) 

s_ashr_i32 s1, s51, 31 // (op 468) 
s_cmp_eq_i32 s11, 0 // (op 468) 
s_mov_b32 s6, scc // (op 468) 
s_cmp_eq_i32 s6, 0 // (op 468) 
s_cselect_b32 s7, 0, -1 // (op 468) 

s_and_b32 s6, s12, 31 // (op 468) 
s_lshl_b32 s18, 1, s6 // (op 468) 
s_add_i32 s19, s18, s7 // (op 468) 


s_and_b32 s7, s1, s19 // (op 468) 


s_add_i32 s1, s51, s7 // (op 468) 


s_ashr_i32 s7, s1, s6 // (op 468) 


s_lshl_b32 s1, s12, 24 // (op 468) 
s_ashr_i32 s6, s1, 31 // (op 468) 

s_xor_b32 s1, s7, s6 // (op 468) 

s_sub_i32 s7, s1, s6 // (op 468) 


s_mul_i32 s1, s0, s14 // (op 468) 

s_sub_u32 s0, s16, s1 // (op 468) 

s_mul_i32 s1, s7, s14 // (op 468) 
s_add_u32 s1, s1, s0 // (op 468) 


s_mul_hi_i32 s0, s16, s8 // (op 468) 
s_add_i32 s51, s0, s16 // (op 468) 

s_ashr_i32 s0, s51, 31 // (op 468) 
s_cmp_eq_i32 s8, 0 // (op 468) 
s_mov_b32 s6, scc // (op 468) 
s_cmp_eq_i32 s6, 0 // (op 468) 
s_cselect_b32 s7, 0, -1 // (op 468) 

s_and_b32 s6, s15, 31 // (op 468) 
s_lshl_b32 s18, 1, s6 // (op 468) 
s_add_i32 s19, s18, s7 // (op 468) 


s_and_b32 s7, s0, s19 // (op 468) 


s_add_i32 s0, s51, s7 // (op 468) 


s_ashr_i32 s7, s0, s6 // (op 468) 


s_lshl_b32 s0, s15, 24 // (op 468) 
s_ashr_i32 s6, s0, 31 // (op 468) 

s_xor_b32 s0, s7, s6 // (op 468) 

s_sub_i32 s7, s0, s6 // (op 468) 


s_add_u32 s18, s4, 16 // (op 468) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 468) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 468) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 468) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 468) 

s_mul_i32 s0, s14, s18 // (op 468) 

s_mul_i32 s6, s50, s0 // (op 468) 

s_mul_i32 s0, s7, s6 // (op 468) 

s_sub_u32 s6, s16, s0 // (op 468) 

s_mov_b32 s0, s6 // (op 468) 
s_mul_hi_i32 s51, s0, s13 // (op 468) 
s_add_i32 s18, s51, s0 // (op 468) 


s_ashr_i32 s0, s18, 31 // (op 468) 
s_cmp_eq_i32 s13, 0 // (op 468) 
s_mov_b32 s19, scc // (op 468) 
s_cmp_eq_i32 s19, 0 // (op 468) 
s_cselect_b32 s51, 0, -1 // (op 468) 

s_and_b32 s19, s49, 31 // (op 468) 
s_lshl_b32 s56, 1, s19 // (op 468) 
s_add_i32 s57, s56, s51 // (op 468) 


s_and_b32 s51, s0, s57 // (op 468) 


s_add_i32 s0, s18, s51 // (op 468) 


s_ashr_i32 s18, s0, s19 // (op 468) 


s_lshl_b32 s0, s49, 24 // (op 468) 
s_ashr_i32 s19, s0, 31 // (op 468) 

s_xor_b32 s0, s18, s19 // (op 468) 

s_sub_i32 s18, s0, s19 // (op 468) 


s_mul_i32 s0, s48, s14 // (op 468) 
s_add_u32 s56, s20, 32 // (op 468) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 468) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 468) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 468) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 468) 

s_sub_i32 s19, s56, s0 // (op 468) 

s_mul_i32 s51, s18, s19 // (op 468) 


s_sub_u32 s18, s6, s51 // (op 468) 


s_add_u32 s6, s18, s0 // (op 468) 


s_cmp_eq_i32 s7, 0 // (op 468) 
s_mov_b32 s0, scc // (op 468) 

s_cmp_eq_i32 s0, 0 // (op 468) 
s_cselect_b32 s7, s6, s1 // (op 468) 



v_lshl_add_u32 v13, s7, 5, v10 // (op 468) 


v_mul_lo_u32 v10, v12, s2 // (op 468) 

v_add_u32 v12, v13, v10 // (op 468) 


s_mul_i32 s1, s2, s25 // (op 470) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 470) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 470) least significant
s_mul_hi_u32 s6, s2, s24 // (op 470) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 470) most significant: sum
s_add_u32 s1, s1, s6 // (op 470) most significant: sum


s_mov_b32 s52, s22 // (op 470) reg expression
s_mov_b32 s53, s23 // (op 470) reg expression
s_mov_b32 s54, s0 // (op 470) reg expression
s_mov_b32 s55, 131072 // (op 470) call()

s_mul_hi_i32 s51, s16, s9 // (op 513) 
v_lshrrev_b32 v10, 3, v1 // (op 513) 
s_add_i32 s0, s51, s16 // (op 513) 

v_and_b32 v13, -8, v1 // (op 513) 
s_ashr_i32 s1, s0, 31 // (op 513) 
v_sub_u32 v14, v1, v13 // (op 513) 

s_cmp_eq_i32 s9, 0 // (op 513) 
s_mov_b32 s51, scc // (op 513) 
v_lshlrev_b32 v13, 4, v14 // (op 513) 

s_cmp_eq_i32 s51, 0 // (op 513) 
s_cselect_b32 s6, 0, -1 // (op 513) 

v_mov_b32 v15, 0 // (op 513) convert
v_mov_b32 v14, v13 // (op 513) convert

s_and_b32 s7, s10, 31 // (op 513) 
s_lshl_b32 s51, 1, s7 // (op 513) 
s_add_i32 s18, s51, s6 // (op 513) 


s_and_b32 s6, s1, s18 // (op 513) 


s_add_i32 s1, s0, s6 // (op 513) 


s_ashr_i32 s0, s1, s7 // (op 513) 


s_lshl_b32 s1, s10, 24 // (op 513) 
s_ashr_i32 s51, s1, 31 // (op 513) 

s_xor_b32 s1, s0, s51 // (op 513) 

s_sub_i32 s0, s1, s51 // (op 513) 


s_mul_hi_i32 s1, s0, s11 // (op 513) 
s_add_i32 s51, s1, s0 // (op 513) 

s_ashr_i32 s1, s51, 31 // (op 513) 
s_cmp_eq_i32 s11, 0 // (op 513) 
s_mov_b32 s6, scc // (op 513) 
s_cmp_eq_i32 s6, 0 // (op 513) 
s_cselect_b32 s7, 0, -1 // (op 513) 

s_and_b32 s6, s12, 31 // (op 513) 
s_lshl_b32 s18, 1, s6 // (op 513) 
s_add_i32 s19, s18, s7 // (op 513) 


s_and_b32 s7, s1, s19 // (op 513) 


s_add_i32 s1, s51, s7 // (op 513) 


s_ashr_i32 s7, s1, s6 // (op 513) 


s_lshl_b32 s1, s12, 24 // (op 513) 
s_ashr_i32 s6, s1, 31 // (op 513) 

s_xor_b32 s1, s7, s6 // (op 513) 

s_sub_i32 s7, s1, s6 // (op 513) 


s_add_u32 s18, s4, 16 // (op 513) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 513) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 513) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 513) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 513) 

s_mul_i32 s1, s7, s18 // (op 513) 

s_sub_i32 s51, s0, s1 // (op 513) 


s_mul_hi_i32 s0, s16, s8 // (op 513) 
s_add_i32 s1, s0, s16 // (op 513) 

s_ashr_i32 s0, s1, 31 // (op 513) 
s_cmp_eq_i32 s8, 0 // (op 513) 
s_mov_b32 s6, scc // (op 513) 
s_cmp_eq_i32 s6, 0 // (op 513) 
s_cselect_b32 s7, 0, -1 // (op 513) 

s_and_b32 s6, s15, 31 // (op 513) 
s_lshl_b32 s56, 1, s6 // (op 513) 
s_add_i32 s57, s56, s7 // (op 513) 


s_and_b32 s7, s0, s57 // (op 513) 


s_add_i32 s0, s1, s7 // (op 513) 


s_ashr_i32 s1, s0, s6 // (op 513) 


s_lshl_b32 s0, s15, 24 // (op 513) 
s_ashr_i32 s6, s0, 31 // (op 513) 

s_xor_b32 s0, s1, s6 // (op 513) 

s_sub_i32 s1, s0, s6 // (op 513) 


s_mul_i32 s0, s14, s18 // (op 513) 

s_mul_i32 s6, s50, s0 // (op 513) 

s_mul_i32 s0, s1, s6 // (op 513) 

s_sub_u32 s6, s16, s0 // (op 513) 

s_mul_hi_i32 s0, s6, s13 // (op 513) 
s_add_i32 s7, s0, s6 // (op 513) 


s_ashr_i32 s0, s7, 31 // (op 513) 
s_cmp_eq_i32 s13, 0 // (op 513) 
s_mov_b32 s6, scc // (op 513) 
s_cmp_eq_i32 s6, 0 // (op 513) 
s_cselect_b32 s18, 0, -1 // (op 513) 

s_and_b32 s6, s49, 31 // (op 513) 
s_lshl_b32 s19, 1, s6 // (op 513) 
s_add_i32 s56, s19, s18 // (op 513) 


s_and_b32 s18, s0, s56 // (op 513) 


s_add_i32 s0, s7, s18 // (op 513) 


s_ashr_i32 s7, s0, s6 // (op 513) 


s_lshl_b32 s0, s49, 24 // (op 513) 
s_ashr_i32 s6, s0, 31 // (op 513) 

s_xor_b32 s0, s7, s6 // (op 513) 

s_sub_i32 s7, s0, s6 // (op 513) 


s_cmp_eq_i32 s1, 0 // (op 513) 
s_mov_b32 s0, scc // (op 513) 

s_cmp_eq_i32 s0, 0 // (op 513) 
s_cselect_b32 s1, s7, s51 // (op 513) 



v_lshl_add_u32 v13, s1, 4, v10 // (op 513) 


v_mul_lo_u32 v17, v13, s27 // (op 513) most significant: low of low * high
v_mul_hi_u32 v10, v13, s26 // (op 513) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 513) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 513) most significant: sum


v_add_co_u32 v18, s[0:1], v14, v16 // (op 513) least significant half
v_addc_co_u32 v19, s[0:1], v15, v17, s[0:1] // (op 513) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 513) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 513) 

v_mov_b32 v10, v16 // (op 513) convert

s_mul_i32 s1, s26, s5 // (op 515) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 515) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 515) least significant
s_mul_hi_u32 s6, s26, s4 // (op 515) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 515) most significant: sum
s_add_u32 s1, s1, s6 // (op 515) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 515) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 515) 

s_mov_b32 s56, s28 // (op 515) reg expression
s_mov_b32 s57, s29 // (op 515) reg expression
s_mov_b32 s58, s0 // (op 515) reg expression
s_mov_b32 s59, 131072 // (op 515) call()

s_mul_hi_i32 s51, s16, s9 // (op 532) 
v_lshrrev_b32 v13, 3, v1 // (op 532) 
s_add_i32 s0, s51, s16 // (op 532) 

v_and_b32 v14, -8, v1 // (op 532) 
s_ashr_i32 s1, s0, 31 // (op 532) 
v_sub_u32 v15, v1, v14 // (op 532) 

s_cmp_eq_i32 s9, 0 // (op 532) 
s_mov_b32 s51, scc // (op 532) 
v_lshlrev_b32 v14, 4, v15 // (op 532) 

s_cmp_eq_i32 s51, 0 // (op 532) 
s_cselect_b32 s6, 0, -1 // (op 532) 

v_mov_b32 v17, 0 // (op 532) convert
v_mov_b32 v16, v14 // (op 532) convert

s_and_b32 s7, s10, 31 // (op 532) 
s_lshl_b32 s51, 1, s7 // (op 532) 
s_add_i32 s18, s51, s6 // (op 532) 


s_and_b32 s6, s1, s18 // (op 532) 


s_add_i32 s1, s0, s6 // (op 532) 


s_ashr_i32 s0, s1, s7 // (op 532) 


s_lshl_b32 s1, s10, 24 // (op 532) 
s_ashr_i32 s51, s1, 31 // (op 532) 

s_xor_b32 s1, s0, s51 // (op 532) 

s_sub_i32 s0, s1, s51 // (op 532) 


s_mul_hi_i32 s1, s0, s11 // (op 532) 
s_add_i32 s51, s1, s0 // (op 532) 

s_ashr_i32 s1, s51, 31 // (op 532) 
s_cmp_eq_i32 s11, 0 // (op 532) 
s_mov_b32 s6, scc // (op 532) 
s_cmp_eq_i32 s6, 0 // (op 532) 
s_cselect_b32 s7, 0, -1 // (op 532) 

s_and_b32 s6, s12, 31 // (op 532) 
s_lshl_b32 s18, 1, s6 // (op 532) 
s_add_i32 s19, s18, s7 // (op 532) 


s_and_b32 s7, s1, s19 // (op 532) 


s_add_i32 s1, s51, s7 // (op 532) 


s_ashr_i32 s7, s1, s6 // (op 532) 


s_lshl_b32 s1, s12, 24 // (op 532) 
s_ashr_i32 s6, s1, 31 // (op 532) 

s_xor_b32 s1, s7, s6 // (op 532) 

s_sub_i32 s7, s1, s6 // (op 532) 


s_add_u32 s18, s4, 16 // (op 532) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 532) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 532) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 532) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 532) 

s_mul_i32 s1, s7, s18 // (op 532) 

s_sub_i32 s51, s0, s1 // (op 532) 


s_mul_hi_i32 s0, s16, s8 // (op 532) 
s_add_i32 s1, s0, s16 // (op 532) 

s_ashr_i32 s0, s1, 31 // (op 532) 
s_cmp_eq_i32 s8, 0 // (op 532) 
s_mov_b32 s6, scc // (op 532) 
s_cmp_eq_i32 s6, 0 // (op 532) 
s_cselect_b32 s7, 0, -1 // (op 532) 

s_and_b32 s6, s15, 31 // (op 532) 
s_lshl_b32 s60, 1, s6 // (op 532) 
s_add_i32 s61, s60, s7 // (op 532) 


s_and_b32 s7, s0, s61 // (op 532) 


s_add_i32 s0, s1, s7 // (op 532) 


s_ashr_i32 s1, s0, s6 // (op 532) 


s_lshl_b32 s0, s15, 24 // (op 532) 
s_ashr_i32 s6, s0, 31 // (op 532) 

s_xor_b32 s0, s1, s6 // (op 532) 

s_sub_i32 s1, s0, s6 // (op 532) 


s_mul_i32 s0, s14, s18 // (op 532) 

s_mul_i32 s6, s50, s0 // (op 532) 

s_mul_i32 s0, s1, s6 // (op 532) 

s_sub_u32 s6, s16, s0 // (op 532) 

s_mul_hi_i32 s0, s6, s13 // (op 532) 
s_add_i32 s7, s0, s6 // (op 532) 


s_ashr_i32 s0, s7, 31 // (op 532) 
s_cmp_eq_i32 s13, 0 // (op 532) 
s_mov_b32 s6, scc // (op 532) 
s_cmp_eq_i32 s6, 0 // (op 532) 
s_cselect_b32 s18, 0, -1 // (op 532) 

s_and_b32 s6, s49, 31 // (op 532) 
s_lshl_b32 s19, 1, s6 // (op 532) 
s_add_i32 s60, s19, s18 // (op 532) 


s_and_b32 s18, s0, s60 // (op 532) 


s_add_i32 s0, s7, s18 // (op 532) 


s_ashr_i32 s7, s0, s6 // (op 532) 


s_lshl_b32 s0, s49, 24 // (op 532) 
s_ashr_i32 s6, s0, 31 // (op 532) 

s_xor_b32 s0, s7, s6 // (op 532) 

s_sub_i32 s7, s0, s6 // (op 532) 


s_cmp_eq_i32 s1, 0 // (op 532) 
s_mov_b32 s0, scc // (op 532) 

s_cmp_eq_i32 s0, 0 // (op 532) 
s_cselect_b32 s1, s7, s51 // (op 532) 



v_lshl_add_u32 v14, s1, 4, v13 // (op 532) 


v_mul_lo_u32 v19, v14, s27 // (op 532) most significant: low of low * high
v_mul_hi_u32 v13, v14, s26 // (op 532) most significant: high of low * low
v_mul_lo_u32 v18, v14, s26 // (op 532) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 532) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 532) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 532) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 532) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 532) 

v_mov_b32 v13, v16 // (op 532) convert

s_mul_i32 s1, s26, s5 // (op 534) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 534) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 534) least significant
s_mul_hi_u32 s6, s26, s4 // (op 534) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 534) most significant: sum
s_add_u32 s1, s1, s6 // (op 534) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 534) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 534) 

s_mov_b32 s56, s28 // (op 534) reg expression
s_mov_b32 s57, s29 // (op 534) reg expression
s_mov_b32 s58, s0 // (op 534) reg expression
s_mov_b32 s59, 131072 // (op 534) call()

s_mul_hi_i32 s51, s16, s9 // (op 577) 
v_and_b32 v14, -32, v1 // (op 577) 
s_add_i32 s0, s51, s16 // (op 577) 

v_sub_u32 v15, v1, v14 // (op 577) 

s_ashr_i32 s1, s0, 31 // (op 577) 
v_lshrrev_b32 v14, 5, v1 // (op 577) 
s_cmp_eq_i32 s9, 0 // (op 577) 
s_mov_b32 s51, scc // (op 577) 
s_cmp_eq_i32 s51, 0 // (op 577) 
s_cselect_b32 s6, 0, -1 // (op 577) 

s_and_b32 s7, s10, 31 // (op 577) 
s_lshl_b32 s51, 1, s7 // (op 577) 
s_add_i32 s18, s51, s6 // (op 577) 


s_and_b32 s6, s1, s18 // (op 577) 


s_add_i32 s1, s0, s6 // (op 577) 


s_ashr_i32 s0, s1, s7 // (op 577) 


s_lshl_b32 s1, s10, 24 // (op 577) 
s_ashr_i32 s51, s1, 31 // (op 577) 

s_xor_b32 s1, s0, s51 // (op 577) 

s_sub_i32 s0, s1, s51 // (op 577) 


s_mul_hi_i32 s1, s0, s11 // (op 577) 
s_add_i32 s51, s1, s0 // (op 577) 

s_ashr_i32 s1, s51, 31 // (op 577) 
s_cmp_eq_i32 s11, 0 // (op 577) 
s_mov_b32 s6, scc // (op 577) 
s_cmp_eq_i32 s6, 0 // (op 577) 
s_cselect_b32 s7, 0, -1 // (op 577) 

s_and_b32 s6, s12, 31 // (op 577) 
s_lshl_b32 s18, 1, s6 // (op 577) 
s_add_i32 s19, s18, s7 // (op 577) 


s_and_b32 s7, s1, s19 // (op 577) 


s_add_i32 s1, s51, s7 // (op 577) 


s_ashr_i32 s7, s1, s6 // (op 577) 


s_lshl_b32 s1, s12, 24 // (op 577) 
s_ashr_i32 s6, s1, 31 // (op 577) 

s_xor_b32 s1, s7, s6 // (op 577) 

s_sub_i32 s7, s1, s6 // (op 577) 


s_mul_i32 s1, s0, s14 // (op 577) 

s_sub_u32 s0, s16, s1 // (op 577) 

s_mul_i32 s1, s7, s14 // (op 577) 
s_add_u32 s1, s1, s0 // (op 577) 


s_mul_hi_i32 s0, s16, s8 // (op 577) 
s_add_i32 s51, s0, s16 // (op 577) 

s_ashr_i32 s0, s51, 31 // (op 577) 
s_cmp_eq_i32 s8, 0 // (op 577) 
s_mov_b32 s6, scc // (op 577) 
s_cmp_eq_i32 s6, 0 // (op 577) 
s_cselect_b32 s7, 0, -1 // (op 577) 

s_and_b32 s6, s15, 31 // (op 577) 
s_lshl_b32 s18, 1, s6 // (op 577) 
s_add_i32 s19, s18, s7 // (op 577) 


s_and_b32 s7, s0, s19 // (op 577) 


s_add_i32 s0, s51, s7 // (op 577) 


s_ashr_i32 s7, s0, s6 // (op 577) 


s_lshl_b32 s0, s15, 24 // (op 577) 
s_ashr_i32 s6, s0, 31 // (op 577) 

s_xor_b32 s0, s7, s6 // (op 577) 

s_sub_i32 s7, s0, s6 // (op 577) 


s_add_u32 s18, s4, 16 // (op 577) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 577) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 577) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 577) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 577) 

s_mul_i32 s0, s14, s18 // (op 577) 

s_mul_i32 s6, s50, s0 // (op 577) 

s_mul_i32 s0, s7, s6 // (op 577) 

s_sub_u32 s6, s16, s0 // (op 577) 

s_mov_b32 s0, s6 // (op 577) 
s_mul_hi_i32 s51, s0, s13 // (op 577) 
s_add_i32 s18, s51, s0 // (op 577) 


s_ashr_i32 s0, s18, 31 // (op 577) 
s_cmp_eq_i32 s13, 0 // (op 577) 
s_mov_b32 s19, scc // (op 577) 
s_cmp_eq_i32 s19, 0 // (op 577) 
s_cselect_b32 s51, 0, -1 // (op 577) 

s_and_b32 s19, s49, 31 // (op 577) 
s_lshl_b32 s60, 1, s19 // (op 577) 
s_add_i32 s61, s60, s51 // (op 577) 


s_and_b32 s51, s0, s61 // (op 577) 


s_add_i32 s0, s18, s51 // (op 577) 


s_ashr_i32 s18, s0, s19 // (op 577) 


s_lshl_b32 s0, s49, 24 // (op 577) 
s_ashr_i32 s19, s0, 31 // (op 577) 

s_xor_b32 s0, s18, s19 // (op 577) 

s_sub_i32 s18, s0, s19 // (op 577) 


s_mul_i32 s0, s48, s14 // (op 577) 
s_add_u32 s60, s20, 32 // (op 577) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 577) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 577) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 577) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 5 // (op 577) 

s_sub_i32 s19, s60, s0 // (op 577) 

s_mul_i32 s51, s18, s19 // (op 577) 


s_sub_u32 s18, s6, s51 // (op 577) 


s_add_u32 s6, s18, s0 // (op 577) 


s_cmp_eq_i32 s7, 0 // (op 577) 
s_mov_b32 s0, scc // (op 577) 

s_cmp_eq_i32 s0, 0 // (op 577) 
s_cselect_b32 s7, s6, s1 // (op 577) 



v_lshl_add_u32 v16, s7, 5, v15 // (op 577) 


v_mul_lo_u32 v15, v14, s30 // (op 577) 

v_add_u32 v14, v16, v15 // (op 577) 


s_mul_i32 s1, s30, s35 // (op 579) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 579) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 579) least significant
s_mul_hi_u32 s6, s30, s34 // (op 579) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 579) most significant: sum
s_add_u32 s1, s1, s6 // (op 579) most significant: sum


s_mov_b32 s60, s32 // (op 579) reg expression
s_mov_b32 s61, s33 // (op 579) reg expression
s_mov_b32 s62, s0 // (op 579) reg expression
s_mov_b32 s63, 131072 // (op 579) call()

s_mul_hi_i32 s51, s16, s9 // (op 596) 
v_and_b32 v15, -32, v1 // (op 596) 
s_add_i32 s0, s51, s16 // (op 596) 

v_sub_u32 v16, v1, v15 // (op 596) 

s_ashr_i32 s1, s0, 31 // (op 596) 
v_lshrrev_b32 v15, 5, v1 // (op 596) 
s_cmp_eq_i32 s9, 0 // (op 596) 
s_mov_b32 s51, scc // (op 596) 
s_cmp_eq_i32 s51, 0 // (op 596) 
s_cselect_b32 s6, 0, -1 // (op 596) 

s_and_b32 s7, s10, 31 // (op 596) 
s_lshl_b32 s51, 1, s7 // (op 596) 
s_add_i32 s18, s51, s6 // (op 596) 


s_and_b32 s6, s1, s18 // (op 596) 


s_add_i32 s1, s0, s6 // (op 596) 


s_ashr_i32 s0, s1, s7 // (op 596) 


s_lshl_b32 s1, s10, 24 // (op 596) 
s_ashr_i32 s51, s1, 31 // (op 596) 

s_xor_b32 s1, s0, s51 // (op 596) 

s_sub_i32 s0, s1, s51 // (op 596) 


s_mul_hi_i32 s1, s0, s11 // (op 596) 
s_add_i32 s51, s1, s0 // (op 596) 

s_ashr_i32 s1, s51, 31 // (op 596) 
s_cmp_eq_i32 s11, 0 // (op 596) 
s_mov_b32 s6, scc // (op 596) 
s_cmp_eq_i32 s6, 0 // (op 596) 
s_cselect_b32 s7, 0, -1 // (op 596) 

s_and_b32 s6, s12, 31 // (op 596) 
s_lshl_b32 s18, 1, s6 // (op 596) 
s_add_i32 s19, s18, s7 // (op 596) 


s_and_b32 s7, s1, s19 // (op 596) 


s_add_i32 s1, s51, s7 // (op 596) 


s_ashr_i32 s7, s1, s6 // (op 596) 


s_lshl_b32 s1, s12, 24 // (op 596) 
s_ashr_i32 s6, s1, 31 // (op 596) 

s_xor_b32 s1, s7, s6 // (op 596) 

s_sub_i32 s7, s1, s6 // (op 596) 


s_mul_i32 s1, s0, s14 // (op 596) 

s_sub_u32 s0, s16, s1 // (op 596) 

s_mul_i32 s1, s7, s14 // (op 596) 
s_add_u32 s1, s1, s0 // (op 596) 


s_mul_hi_i32 s0, s16, s8 // (op 596) 
s_add_i32 s51, s0, s16 // (op 596) 

s_ashr_i32 s0, s51, 31 // (op 596) 
s_cmp_eq_i32 s8, 0 // (op 596) 
s_mov_b32 s6, scc // (op 596) 
s_cmp_eq_i32 s6, 0 // (op 596) 
s_cselect_b32 s7, 0, -1 // (op 596) 

s_and_b32 s6, s15, 31 // (op 596) 
s_lshl_b32 s18, 1, s6 // (op 596) 
s_add_i32 s19, s18, s7 // (op 596) 


s_and_b32 s7, s0, s19 // (op 596) 


s_add_i32 s0, s51, s7 // (op 596) 


s_ashr_i32 s7, s0, s6 // (op 596) 


s_lshl_b32 s0, s15, 24 // (op 596) 
s_ashr_i32 s6, s0, 31 // (op 596) 

s_xor_b32 s0, s7, s6 // (op 596) 

s_sub_i32 s7, s0, s6 // (op 596) 


s_add_u32 s18, s4, 16 // (op 596) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 596) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 596) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 596) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 596) 

s_mul_i32 s0, s14, s18 // (op 596) 

s_mul_i32 s6, s50, s0 // (op 596) 

s_mul_i32 s0, s7, s6 // (op 596) 

s_sub_u32 s6, s16, s0 // (op 596) 

s_mov_b32 s0, s6 // (op 596) 
s_mul_hi_i32 s51, s0, s13 // (op 596) 
s_add_i32 s18, s51, s0 // (op 596) 


s_ashr_i32 s0, s18, 31 // (op 596) 
s_cmp_eq_i32 s13, 0 // (op 596) 
s_mov_b32 s19, scc // (op 596) 
s_cmp_eq_i32 s19, 0 // (op 596) 
s_cselect_b32 s51, 0, -1 // (op 596) 

s_and_b32 s19, s49, 31 // (op 596) 
s_lshl_b32 s64, 1, s19 // (op 596) 
s_add_i32 s65, s64, s51 // (op 596) 


s_and_b32 s51, s0, s65 // (op 596) 


s_add_i32 s0, s18, s51 // (op 596) 


s_ashr_i32 s18, s0, s19 // (op 596) 


s_lshl_b32 s0, s49, 24 // (op 596) 
s_ashr_i32 s19, s0, 31 // (op 596) 

s_xor_b32 s0, s18, s19 // (op 596) 

s_sub_i32 s18, s0, s19 // (op 596) 


s_mul_i32 s0, s48, s14 // (op 596) 
s_add_u32 s64, s20, 32 // (op 596) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 596) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 596) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 596) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 5 // (op 596) 

s_sub_i32 s19, s64, s0 // (op 596) 

s_mul_i32 s51, s18, s19 // (op 596) 


s_sub_u32 s18, s6, s51 // (op 596) 


s_add_u32 s6, s18, s0 // (op 596) 


s_cmp_eq_i32 s7, 0 // (op 596) 
s_mov_b32 s0, scc // (op 596) 

s_cmp_eq_i32 s0, 0 // (op 596) 
s_cselect_b32 s7, s6, s1 // (op 596) 



v_lshl_add_u32 v17, s7, 5, v16 // (op 596) 


v_mul_lo_u32 v16, v15, s30 // (op 596) 

v_add_u32 v15, v17, v16 // (op 596) 


s_mul_i32 s1, s30, s35 // (op 598) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 598) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 598) least significant
s_mul_hi_u32 s6, s30, s34 // (op 598) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 598) most significant: sum
s_add_u32 s1, s1, s6 // (op 598) most significant: sum


s_mov_b32 s60, s32 // (op 598) reg expression
s_mov_b32 s61, s33 // (op 598) reg expression
s_mov_b32 s62, s0 // (op 598) reg expression
s_mov_b32 s63, 131072 // (op 598) call()

s_mul_hi_i32 s51, s16, s9 // (op 641) 
v_and_b32 v16, 15, v1 // (op 641) 
s_add_i32 s0, s51, s16 // (op 641) 

v_and_b32 v17, 63, v1 // (op 641) 
s_ashr_i32 s1, s0, 31 // (op 641) 
v_lshrrev_b32 v18, 4, v17 // (op 641) 

s_cmp_eq_i32 s9, 0 // (op 641) 
s_mov_b32 s51, scc // (op 641) 
s_cmp_eq_i32 s51, 0 // (op 641) 
s_cselect_b32 s6, 0, -1 // (op 641) 

s_and_b32 s7, s10, 31 // (op 641) 
s_lshl_b32 s51, 1, s7 // (op 641) 
s_add_i32 s18, s51, s6 // (op 641) 


s_and_b32 s6, s1, s18 // (op 641) 


s_add_i32 s1, s0, s6 // (op 641) 


s_ashr_i32 s0, s1, s7 // (op 641) 


s_lshl_b32 s1, s10, 24 // (op 641) 
s_ashr_i32 s51, s1, 31 // (op 641) 

s_xor_b32 s1, s0, s51 // (op 641) 

s_sub_i32 s0, s1, s51 // (op 641) 


s_mul_hi_i32 s1, s0, s11 // (op 641) 
s_add_i32 s51, s1, s0 // (op 641) 

s_ashr_i32 s1, s51, 31 // (op 641) 
s_cmp_eq_i32 s11, 0 // (op 641) 
s_mov_b32 s6, scc // (op 641) 
s_cmp_eq_i32 s6, 0 // (op 641) 
s_cselect_b32 s7, 0, -1 // (op 641) 

s_and_b32 s6, s12, 31 // (op 641) 
s_lshl_b32 s18, 1, s6 // (op 641) 
s_add_i32 s19, s18, s7 // (op 641) 


s_and_b32 s7, s1, s19 // (op 641) 


s_add_i32 s1, s51, s7 // (op 641) 


s_ashr_i32 s7, s1, s6 // (op 641) 


s_lshl_b32 s1, s12, 24 // (op 641) 
s_ashr_i32 s6, s1, 31 // (op 641) 

s_xor_b32 s1, s7, s6 // (op 641) 

s_sub_i32 s7, s1, s6 // (op 641) 


s_add_u32 s18, s4, 16 // (op 641) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 641) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 641) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 641) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 641) 

s_mul_i32 s1, s7, s18 // (op 641) 

s_sub_i32 s51, s0, s1 // (op 641) 


s_mul_hi_i32 s0, s16, s8 // (op 641) 
s_add_i32 s1, s0, s16 // (op 641) 

s_ashr_i32 s0, s1, 31 // (op 641) 
s_cmp_eq_i32 s8, 0 // (op 641) 
s_mov_b32 s6, scc // (op 641) 
s_cmp_eq_i32 s6, 0 // (op 641) 
s_cselect_b32 s7, 0, -1 // (op 641) 

s_and_b32 s6, s15, 31 // (op 641) 
s_lshl_b32 s64, 1, s6 // (op 641) 
s_add_i32 s65, s64, s7 // (op 641) 


s_and_b32 s7, s0, s65 // (op 641) 


s_add_i32 s0, s1, s7 // (op 641) 


s_ashr_i32 s1, s0, s6 // (op 641) 


s_lshl_b32 s0, s15, 24 // (op 641) 
s_ashr_i32 s6, s0, 31 // (op 641) 

s_xor_b32 s0, s1, s6 // (op 641) 

s_sub_i32 s1, s0, s6 // (op 641) 


s_mul_i32 s0, s14, s18 // (op 641) 

s_mul_i32 s6, s50, s0 // (op 641) 

s_mul_i32 s0, s1, s6 // (op 641) 

s_sub_u32 s6, s16, s0 // (op 641) 

s_mul_hi_i32 s0, s6, s13 // (op 641) 
s_add_i32 s7, s0, s6 // (op 641) 


s_ashr_i32 s0, s7, 31 // (op 641) 
s_cmp_eq_i32 s13, 0 // (op 641) 
s_mov_b32 s6, scc // (op 641) 
s_cmp_eq_i32 s6, 0 // (op 641) 
s_cselect_b32 s18, 0, -1 // (op 641) 

s_and_b32 s6, s49, 31 // (op 641) 
s_lshl_b32 s19, 1, s6 // (op 641) 
s_add_i32 s64, s19, s18 // (op 641) 


s_and_b32 s18, s0, s64 // (op 641) 


s_add_i32 s0, s7, s18 // (op 641) 


s_ashr_i32 s7, s0, s6 // (op 641) 


s_lshl_b32 s0, s49, 24 // (op 641) 
s_ashr_i32 s6, s0, 31 // (op 641) 

s_xor_b32 s0, s7, s6 // (op 641) 

s_sub_i32 s7, s0, s6 // (op 641) 


s_cmp_eq_i32 s1, 0 // (op 641) 
s_mov_b32 s0, scc // (op 641) 

s_cmp_eq_i32 s0, 0 // (op 641) 
s_cselect_b32 s1, s7, s51 // (op 641) 



v_lshl_add_u32 v17, s1, 4, v16 // (op 641) 


v_mul_lo_u32 v16, v17, s36 // (op 641) 

v_add_u32 v17, v18, v16 // (op 641) 


s_mul_i32 s1, s36, s5 // (op 643) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 643) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 643) least significant
s_mul_hi_u32 s6, s36, s4 // (op 643) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 643) most significant: sum
s_add_u32 s1, s1, s6 // (op 643) most significant: sum


s_mov_b32 s64, s38 // (op 643) reg expression
s_mov_b32 s65, s39 // (op 643) reg expression
s_mov_b32 s66, s0 // (op 643) reg expression
s_mov_b32 s67, 131072 // (op 643) call()

s_mul_hi_i32 s51, s16, s9 // (op 660) 
v_and_b32 v16, 15, v1 // (op 660) 
s_add_i32 s0, s51, s16 // (op 660) 

v_and_b32 v18, 63, v1 // (op 660) 
s_ashr_i32 s1, s0, 31 // (op 660) 
v_lshrrev_b32 v19, 4, v18 // (op 660) 

s_cmp_eq_i32 s9, 0 // (op 660) 
s_mov_b32 s51, scc // (op 660) 
s_cmp_eq_i32 s51, 0 // (op 660) 
s_cselect_b32 s6, 0, -1 // (op 660) 

s_and_b32 s7, s10, 31 // (op 660) 
s_lshl_b32 s51, 1, s7 // (op 660) 
s_add_i32 s18, s51, s6 // (op 660) 


s_and_b32 s6, s1, s18 // (op 660) 


s_add_i32 s1, s0, s6 // (op 660) 


s_ashr_i32 s0, s1, s7 // (op 660) 


s_lshl_b32 s1, s10, 24 // (op 660) 
s_ashr_i32 s51, s1, 31 // (op 660) 

s_xor_b32 s1, s0, s51 // (op 660) 

s_sub_i32 s0, s1, s51 // (op 660) 


s_mul_hi_i32 s1, s0, s11 // (op 660) 
s_add_i32 s51, s1, s0 // (op 660) 

s_ashr_i32 s1, s51, 31 // (op 660) 
s_cmp_eq_i32 s11, 0 // (op 660) 
s_mov_b32 s6, scc // (op 660) 
s_cmp_eq_i32 s6, 0 // (op 660) 
s_cselect_b32 s7, 0, -1 // (op 660) 

s_and_b32 s6, s12, 31 // (op 660) 
s_lshl_b32 s18, 1, s6 // (op 660) 
s_add_i32 s19, s18, s7 // (op 660) 


s_and_b32 s7, s1, s19 // (op 660) 


s_add_i32 s1, s51, s7 // (op 660) 


s_ashr_i32 s7, s1, s6 // (op 660) 


s_lshl_b32 s1, s12, 24 // (op 660) 
s_ashr_i32 s6, s1, 31 // (op 660) 

s_xor_b32 s1, s7, s6 // (op 660) 

s_sub_i32 s7, s1, s6 // (op 660) 


s_add_u32 s18, s4, 16 // (op 660) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 660) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 660) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 660) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 660) 

s_mul_i32 s1, s7, s18 // (op 660) 

s_sub_i32 s51, s0, s1 // (op 660) 


s_mul_hi_i32 s0, s16, s8 // (op 660) 
s_add_i32 s1, s0, s16 // (op 660) 

s_ashr_i32 s0, s1, 31 // (op 660) 
s_cmp_eq_i32 s8, 0 // (op 660) 
s_mov_b32 s6, scc // (op 660) 
s_cmp_eq_i32 s6, 0 // (op 660) 
s_cselect_b32 s7, 0, -1 // (op 660) 

s_and_b32 s6, s15, 31 // (op 660) 
s_lshl_b32 s68, 1, s6 // (op 660) 
s_add_i32 s69, s68, s7 // (op 660) 


s_and_b32 s7, s0, s69 // (op 660) 


s_add_i32 s0, s1, s7 // (op 660) 


s_ashr_i32 s1, s0, s6 // (op 660) 


s_lshl_b32 s0, s15, 24 // (op 660) 
s_ashr_i32 s6, s0, 31 // (op 660) 

s_xor_b32 s0, s1, s6 // (op 660) 

s_sub_i32 s1, s0, s6 // (op 660) 


s_mul_i32 s0, s14, s18 // (op 660) 

s_mul_i32 s6, s50, s0 // (op 660) 

s_mul_i32 s0, s1, s6 // (op 660) 

s_sub_u32 s6, s16, s0 // (op 660) 

s_mul_hi_i32 s0, s6, s13 // (op 660) 
s_add_i32 s7, s0, s6 // (op 660) 


s_ashr_i32 s0, s7, 31 // (op 660) 
s_cmp_eq_i32 s13, 0 // (op 660) 
s_mov_b32 s6, scc // (op 660) 
s_cmp_eq_i32 s6, 0 // (op 660) 
s_cselect_b32 s18, 0, -1 // (op 660) 

s_and_b32 s6, s49, 31 // (op 660) 
s_lshl_b32 s19, 1, s6 // (op 660) 
s_add_i32 s68, s19, s18 // (op 660) 


s_and_b32 s18, s0, s68 // (op 660) 


s_add_i32 s0, s7, s18 // (op 660) 


s_ashr_i32 s7, s0, s6 // (op 660) 


s_lshl_b32 s0, s49, 24 // (op 660) 
s_ashr_i32 s6, s0, 31 // (op 660) 

s_xor_b32 s0, s7, s6 // (op 660) 

s_sub_i32 s7, s0, s6 // (op 660) 


s_cmp_eq_i32 s1, 0 // (op 660) 
s_mov_b32 s0, scc // (op 660) 

s_cmp_eq_i32 s0, 0 // (op 660) 
s_cselect_b32 s1, s7, s51 // (op 660) 



v_lshl_add_u32 v18, s1, 4, v16 // (op 660) 


v_mul_lo_u32 v16, v18, s36 // (op 660) 

v_add_u32 v18, v19, v16 // (op 660) 


s_mul_i32 s1, s36, s5 // (op 662) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 662) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 662) least significant
s_mul_hi_u32 s6, s36, s4 // (op 662) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 662) most significant: sum
s_add_u32 s1, s1, s6 // (op 662) most significant: sum


s_mov_b32 s64, s38 // (op 662) reg expression
s_mov_b32 s65, s39 // (op 662) reg expression
s_mov_b32 s66, s0 // (op 662) reg expression
s_mov_b32 s67, 131072 // (op 662) call()




buffer_load_ubyte v16, v17, s[64:67], 0 offen offset: 0 // (op 337) Load value



buffer_load_dwordx2 v[20:21], v10, s[56:59], 0 offen // (op 341) Load value

v_mov_b64 v[22:23], s[2:3] // (op 344) convert

v_mov_b32 v19, v11 // (op 344) reg expression
buffer_load_dwordx4 v[24:27], v19, s[52:55], 0 offen // (op 344) Load value
v_add_u32 v19, v19, v22 // (op 344) 
buffer_load_dwordx4 v[28:31], v19, s[52:55], 0 offen // (op 344) Load value




v_mov_b32 v19, v14 // (op 347) reg expression

buffer_load_ubyte v22, v19, s[60:63], 0 offen offset: 0 // (op 347) Load value




buffer_load_ubyte v19, v17, s[64:67], 0 offen offset:4 // (op 351) Load value




buffer_load_dwordx2 v[32:33], v10, s[56:59], 0 offen offset:64 // (op 355) Load value


v_mov_b64 v[34:35], s[2:3] // (op 358) convert

s_lshl_b32 s51, s2, 7 // (op 358) 
v_add_u32 v10, v11, s51 // (op 358) 

buffer_load_dwordx4 v[36:39], v10, s[52:55], 0 offen // (op 358) Load value
v_add_u32 v10, v10, v34 // (op 358) 
buffer_load_dwordx4 v[40:43], v10, s[52:55], 0 offen // (op 358) Load value





s_lshl_b32 s51, s30, 2 // (op 361) 
v_add_u32 v17, v14, s51 // (op 361) 


buffer_load_ubyte v23, v17, s[60:63], 0 offen offset: 0 // (op 361) Load value


v_and_b32 v14, -8, v1 // (op 780) 
v_sub_u32 v17, v1, v14 // (op 780) 

v_lshlrev_b32 v14, 4, v17 // (op 780) 

v_lshrrev_b32 v17, 3, v1 // (op 780) 
v_lshl_add_u32 v10, v17, 7, v14 // (op 780) 


v_lshlrev_b32 v11, 2, v10 // (op 780) 

v_lshrrev_b32 v10, 3, v11 // (op 780) 




s_waitcnt vmcnt(8)
ds_write_b64 v10, v[20:21] // (op 363) Store local data 

v_and_b32 v14, -2, v1 // (op 727) 
v_sub_u32 v17, v1, v14 // (op 727) 

v_lshlrev_b32 v14, 4, v17 // (op 727) 

v_lshrrev_b32 v17, 1, v1 // (op 727) 
v_lshl_add_u32 v10, v17, 6, v14 // (op 727) 





s_waitcnt vmcnt(7)
ds_write_b128 v10, v[24:27] offset:1024 // (op 364) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v10, v[28:31] offset:1056 // (op 364) Store local data 

v_and_b32 v14, -32, v1 // (op 833) 
v_sub_u32 v17, v1, v14 // (op 833) 

v_and_b32 v14, 4294967264, v1 // (op 833) 
v_add_u32 v10, v14, v17 // (op 833) 





s_waitcnt vmcnt(5)
ds_write_b8 v10, v22 offset:5120 // (op 365) Store local data 

v_and_b32 v14, -2, v1 // (op 698) 
v_sub_u32 v17, v1, v14 // (op 698) 

v_lshlrev_b32 v14, 4, v17 // (op 698) 

v_lshrrev_b32 v17, 1, v1 // (op 698) 
v_lshl_add_u32 v10, v17, 6, v14 // (op 698) 


v_and_b32 v11, -8, v1 // (op 751) 
v_sub_u32 v14, v1, v11 // (op 751) 

v_lshlrev_b32 v11, 4, v14 // (op 751) 

v_lshrrev_b32 v14, 3, v1 // (op 751) 
v_lshl_add_u32 v17, v14, 7, v11 // (op 751) 


v_lshlrev_b32 v11, 2, v17 // (op 751) 

v_lshrrev_b32 v14, 3, v11 // (op 751) 

v_and_b32 v11, -32, v1 // (op 804) 
v_sub_u32 v17, v1, v11 // (op 804) 

v_and_b32 v11, 4294967264, v1 // (op 804) 
v_add_u32 v34, v11, v17 // (op 804) 


v_and_b32 v11, -2, v1 // (op 891) 
v_sub_u32 v17, v1, v11 // (op 891) 

v_lshlrev_b32 v11, 4, v17 // (op 891) 

v_lshrrev_b32 v17, 1, v1 // (op 891) 
v_lshl_add_u32 v35, v17, 6, v11 // (op 891) 


v_and_b32 v11, -8, v1 // (op 915) 
v_sub_u32 v17, v1, v11 // (op 915) 

v_lshlrev_b32 v11, 4, v17 // (op 915) 

v_lshrrev_b32 v17, 3, v1 // (op 915) 
v_lshl_add_u32 v44, v17, 7, v11 // (op 915) 


v_lshlrev_b32 v11, 2, v44 // (op 915) 

v_lshrrev_b32 v17, 3, v11 // (op 915) 

v_and_b32 v11, -32, v1 // (op 939) 
v_sub_u32 v44, v1, v11 // (op 939) 

v_and_b32 v11, 4294967264, v1 // (op 939) 
v_add_u32 v45, v11, v44 // (op 939) 


s_mov_b32 s51, 0 // (op 55) call()
s_ashr_i64 s[0:1], s[24:25], 7 // (op 54) 
s_mov_b32 s6, s0 // (op 54) convert

s_and_b32 s7, s6, 4294967294 // (op 54) 

s_cmp_lt_u32 s51, s7 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_54 if false)
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_54:
s_add_u32 s51, s51, 2 // (op 56) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 984) 



ds_read_b64_tr_b8 v[46:47], v0 offset:1024 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:1280 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:3072 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:3328 // (op 65) Transpose load local data 
s_barrier  // (op 997) 



ds_read_b128 v[54:57], v4 // (op 70) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1011) 


ds_read_u8 v11, v3 offset:5120 // (op 75) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:53], v[54:57], a[0:3], v11, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 82) 





v_mov_b32 v11, v18 // (op 80) 

buffer_load_ubyte v16, v11, s[64:67], 0 offen offset:8 // (op 80) Load value


v_mov_b32 v44, v13 // (op 10) 

buffer_load_dwordx2 v[20:21], v44, s[56:59], 0 offen offset:128 // (op 10) Load value

v_mov_b64 v[46:47], s[2:3] // (op 4) convert

v_mov_b32 v48, v12 // (op 4) 
s_lshl_b32 s0, s2, 8 // (op 4) 
v_add_u32 v49, v48, s0 // (op 4) 

buffer_load_dwordx4 v[24:27], v49, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v49, v49, v46 // (op 4) 
buffer_load_dwordx4 v[28:31], v49, s[52:55], 0 offen // (op 4) Load value




v_mov_b32 v46, v15 // (op 17) 
s_lshl_b32 s0, s30, 3 // (op 17) 
v_add_u32 v47, v46, s0 // (op 17) 


buffer_load_ubyte v22, v47, s[60:63], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1034) 



s_waitcnt vmcnt(8)
ds_write_b64 v17, v[32:33] offset:5248 // (op 235) Store local data 
s_barrier  // (op 1028) 



s_waitcnt vmcnt(7)
ds_write_b128 v35, v[36:39] offset:6272 // (op 225) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v35, v[40:43] offset:6304 // (op 225) Store local data 
s_barrier  // (op 1040) 



s_waitcnt vmcnt(5)
ds_write_b8 v45, v23 offset:10368 // (op 236) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1025) 



ds_read_b64_tr_b8 v[50:51], v5 offset:6272 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v5 offset:6528 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v5 offset:8320 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v5 offset:8576 // (op 227) Transpose load local data 
s_barrier  // (op 1031) 
s_barrier  // (op 1091) 



ds_read_b128 v[58:61], v7 offset:5248 // (op 229) Load local data 
s_barrier  // (op 1097) 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1037) 


ds_read_u8 v47, v6 offset:10368 // (op 231) Load local data 
s_barrier  // (op 1103) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:57], v[58:61], a[0:3], v47, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 234) 






buffer_load_ubyte v19, v11, s[64:67], 0 offen offset:12 // (op 233) Load value




v_add_u32 v18, 8, v18 // (op 652) 
buffer_load_dwordx2 v[32:33], v44, s[56:59], 0 offen offset:192 // (op 237) Load value


v_add_u32 v13, 128, v13 // (op 524) 
v_mov_b64 v[50:51], s[2:3] // (op 224) convert

s_lshl_b32 s0, s2, 7 // (op 224) 
s_mul_i32 s1, 3, s0 // (op 224) 

v_add_u32 v11, v48, s1 // (op 224) 

buffer_load_dwordx4 v[36:39], v11, s[52:55], 0 offen // (op 224) Load value
v_add_u32 v11, v11, v50 // (op 224) 
buffer_load_dwordx4 v[40:43], v11, s[52:55], 0 offen // (op 224) Load value





s_lshl_b32 s0, s30, 2 // (op 238) 
s_mul_i32 s1, 3, s0 // (op 238) 

s_lshl_b32 s0, s2, 8 // (op 460) 
v_add_u32 v12, v12, s0 // (op 460) 

v_add_u32 v11, v46, s1 // (op 238) 


buffer_load_ubyte v23, v11, s[60:63], 0 offen offset: 0 // (op 238) Load value


s_barrier  // (op 1000) 
s_lshl_b32 s0, s30, 3 // (op 588) 



v_add_u32 v15, v15, s0 // (op 588) 

s_waitcnt vmcnt(8)
ds_write_b64 v14, v[20:21] // (op 13) Store local data 
s_barrier  // (op 987) 



s_waitcnt vmcnt(7)
ds_write_b128 v10, v[24:27] offset:1024 // (op 6) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v10, v[28:31] offset:1056 // (op 6) Store local data 
s_barrier  // (op 1014) 



s_waitcnt vmcnt(5)
ds_write_b8 v34, v22 offset:5120 // (op 21) Store local data 
s_ashr_i64 s[0:1], s[24:25], 7 // (op 54) 
s_mov_b32 s6, s0 // (op 54) convert

s_and_b32 s7, s6, 4294967294 // (op 54) 

s_cmp_lt_u32 s51, s7 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_54 if true)
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_54:















s_mul_hi_i32 s51, s16, s9 // (op 423) 
v_and_b32 v10, -2, v1 // (op 423) 
s_add_i32 s0, s51, s16 // (op 423) 

v_sub_u32 v11, v1, v10 // (op 423) 

s_ashr_i32 s1, s0, 31 // (op 423) 
v_lshlrev_b32 v10, 4, v11 // (op 423) 

s_cmp_eq_i32 s9, 0 // (op 423) 
s_mov_b32 s51, scc // (op 423) 
v_and_b32 v11, 4294967294, v1 // (op 423) 
s_cmp_eq_i32 s51, 0 // (op 423) 
s_cselect_b32 s6, 0, -1 // (op 423) 

s_and_b32 s7, s10, 31 // (op 423) 
s_lshl_b32 s51, 1, s7 // (op 423) 
s_add_i32 s18, s51, s6 // (op 423) 


s_and_b32 s6, s1, s18 // (op 423) 


s_add_i32 s1, s0, s6 // (op 423) 


s_ashr_i32 s0, s1, s7 // (op 423) 


s_lshl_b32 s1, s10, 24 // (op 423) 
s_ashr_i32 s51, s1, 31 // (op 423) 

s_xor_b32 s1, s0, s51 // (op 423) 

s_sub_i32 s0, s1, s51 // (op 423) 


s_mul_hi_i32 s1, s0, s11 // (op 423) 
s_add_i32 s51, s1, s0 // (op 423) 

s_ashr_i32 s1, s51, 31 // (op 423) 
s_cmp_eq_i32 s11, 0 // (op 423) 
s_mov_b32 s6, scc // (op 423) 
s_cmp_eq_i32 s6, 0 // (op 423) 
s_cselect_b32 s7, 0, -1 // (op 423) 

s_and_b32 s6, s12, 31 // (op 423) 
s_lshl_b32 s18, 1, s6 // (op 423) 
s_add_i32 s19, s18, s7 // (op 423) 


s_and_b32 s7, s1, s19 // (op 423) 


s_add_i32 s1, s51, s7 // (op 423) 


s_ashr_i32 s7, s1, s6 // (op 423) 


s_lshl_b32 s1, s12, 24 // (op 423) 
s_ashr_i32 s6, s1, 31 // (op 423) 

s_xor_b32 s1, s7, s6 // (op 423) 

s_sub_i32 s7, s1, s6 // (op 423) 


s_mul_i32 s1, s0, s14 // (op 423) 

s_sub_u32 s0, s16, s1 // (op 423) 

s_mul_i32 s1, s7, s14 // (op 423) 
s_add_u32 s1, s1, s0 // (op 423) 


s_mul_hi_i32 s0, s16, s8 // (op 423) 
s_add_i32 s51, s0, s16 // (op 423) 

s_ashr_i32 s0, s51, 31 // (op 423) 
s_cmp_eq_i32 s8, 0 // (op 423) 
s_mov_b32 s6, scc // (op 423) 
s_cmp_eq_i32 s6, 0 // (op 423) 
s_cselect_b32 s7, 0, -1 // (op 423) 

s_and_b32 s6, s15, 31 // (op 423) 
s_lshl_b32 s18, 1, s6 // (op 423) 
s_add_i32 s19, s18, s7 // (op 423) 


s_and_b32 s7, s0, s19 // (op 423) 


s_add_i32 s0, s51, s7 // (op 423) 


s_ashr_i32 s7, s0, s6 // (op 423) 


s_lshl_b32 s0, s15, 24 // (op 423) 
s_ashr_i32 s6, s0, 31 // (op 423) 

s_xor_b32 s0, s7, s6 // (op 423) 

s_sub_i32 s7, s0, s6 // (op 423) 


s_add_u32 s18, s4, 16 // (op 423) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 423) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 423) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 423) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 423) 

s_mul_i32 s0, s14, s18 // (op 423) 

s_mul_i32 s6, s50, s0 // (op 423) 

s_mul_i32 s0, s7, s6 // (op 423) 

s_sub_u32 s6, s16, s0 // (op 423) 

s_mov_b32 s0, s6 // (op 423) 
s_mul_hi_i32 s51, s0, s13 // (op 423) 
s_add_i32 s18, s51, s0 // (op 423) 


s_ashr_i32 s0, s18, 31 // (op 423) 
s_cmp_eq_i32 s13, 0 // (op 423) 
s_mov_b32 s19, scc // (op 423) 
s_cmp_eq_i32 s19, 0 // (op 423) 
s_cselect_b32 s51, 0, -1 // (op 423) 

s_and_b32 s19, s49, 31 // (op 423) 
s_lshl_b32 s68, 1, s19 // (op 423) 
s_add_i32 s69, s68, s51 // (op 423) 


s_and_b32 s51, s0, s69 // (op 423) 


s_add_i32 s0, s18, s51 // (op 423) 


s_ashr_i32 s18, s0, s19 // (op 423) 


s_lshl_b32 s0, s49, 24 // (op 423) 
s_ashr_i32 s19, s0, 31 // (op 423) 

s_xor_b32 s0, s18, s19 // (op 423) 

s_sub_i32 s18, s0, s19 // (op 423) 


s_mul_i32 s0, s48, s14 // (op 423) 
s_add_u32 s68, s20, 32 // (op 423) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 423) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 423) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 423) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 423) 

s_sub_i32 s19, s68, s0 // (op 423) 

s_mul_i32 s51, s18, s19 // (op 423) 


s_sub_u32 s18, s6, s51 // (op 423) 


s_add_u32 s6, s18, s0 // (op 423) 


s_cmp_eq_i32 s7, 0 // (op 423) 
s_mov_b32 s0, scc // (op 423) 

s_cmp_eq_i32 s0, 0 // (op 423) 
s_cselect_b32 s7, s6, s1 // (op 423) 



v_lshl_add_u32 v12, s7, 5, v10 // (op 423) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 423) 
s_lshr_b32 s51, s0, 1 // (op 423) 

v_lshl_add_u32 v10, s51, 8, v11 // (op 423) 


v_mul_lo_u32 v11, v10, s2 // (op 423) 

v_add_u32 v10, v12, v11 // (op 423) 


s_mul_i32 s1, s2, s25 // (op 425) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 425) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 425) least significant
s_mul_hi_u32 s6, s2, s24 // (op 425) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 425) most significant: sum
s_add_u32 s1, s1, s6 // (op 425) most significant: sum


s_mov_b32 s52, s22 // (op 425) reg expression
s_mov_b32 s53, s23 // (op 425) reg expression
s_mov_b32 s54, s0 // (op 425) reg expression
s_mov_b32 s55, 131072 // (op 425) call()

s_mul_hi_i32 s51, s16, s9 // (op 490) 
v_lshrrev_b32 v11, 3, v1 // (op 490) 
s_add_i32 s0, s51, s16 // (op 490) 

v_and_b32 v12, -8, v1 // (op 490) 
s_ashr_i32 s1, s0, 31 // (op 490) 
v_sub_u32 v13, v1, v12 // (op 490) 

s_cmp_eq_i32 s9, 0 // (op 490) 
s_mov_b32 s51, scc // (op 490) 
v_lshlrev_b32 v12, 4, v13 // (op 490) 

s_cmp_eq_i32 s51, 0 // (op 490) 
s_cselect_b32 s6, 0, -1 // (op 490) 

s_and_b32 s7, s10, 31 // (op 490) 
s_lshl_b32 s51, 1, s7 // (op 490) 
s_add_i32 s18, s51, s6 // (op 490) 


s_and_b32 s6, s1, s18 // (op 490) 


s_add_i32 s1, s0, s6 // (op 490) 


s_ashr_i32 s0, s1, s7 // (op 490) 


s_lshl_b32 s1, s10, 24 // (op 490) 
s_ashr_i32 s51, s1, 31 // (op 490) 

s_xor_b32 s1, s0, s51 // (op 490) 

s_sub_i32 s0, s1, s51 // (op 490) 


s_mul_hi_i32 s1, s0, s11 // (op 490) 
s_add_i32 s51, s1, s0 // (op 490) 

s_ashr_i32 s1, s51, 31 // (op 490) 
s_cmp_eq_i32 s11, 0 // (op 490) 
s_mov_b32 s6, scc // (op 490) 
s_cmp_eq_i32 s6, 0 // (op 490) 
s_cselect_b32 s7, 0, -1 // (op 490) 

s_and_b32 s6, s12, 31 // (op 490) 
s_lshl_b32 s18, 1, s6 // (op 490) 
s_add_i32 s19, s18, s7 // (op 490) 


s_and_b32 s7, s1, s19 // (op 490) 


s_add_i32 s1, s51, s7 // (op 490) 


s_ashr_i32 s7, s1, s6 // (op 490) 


s_lshl_b32 s1, s12, 24 // (op 490) 
s_ashr_i32 s6, s1, 31 // (op 490) 

s_xor_b32 s1, s7, s6 // (op 490) 

s_sub_i32 s7, s1, s6 // (op 490) 


s_add_u32 s18, s4, 16 // (op 490) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 490) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 490) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 490) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 490) 

s_mul_i32 s1, s7, s18 // (op 490) 

s_sub_i32 s51, s0, s1 // (op 490) 


s_mul_hi_i32 s0, s16, s8 // (op 490) 
s_add_i32 s1, s0, s16 // (op 490) 

s_ashr_i32 s0, s1, 31 // (op 490) 
s_cmp_eq_i32 s8, 0 // (op 490) 
s_mov_b32 s6, scc // (op 490) 
s_cmp_eq_i32 s6, 0 // (op 490) 
s_cselect_b32 s7, 0, -1 // (op 490) 

s_and_b32 s6, s15, 31 // (op 490) 
s_lshl_b32 s68, 1, s6 // (op 490) 
s_add_i32 s69, s68, s7 // (op 490) 


s_and_b32 s7, s0, s69 // (op 490) 


s_add_i32 s0, s1, s7 // (op 490) 


s_ashr_i32 s1, s0, s6 // (op 490) 


s_lshl_b32 s0, s15, 24 // (op 490) 
s_ashr_i32 s6, s0, 31 // (op 490) 

s_xor_b32 s0, s1, s6 // (op 490) 

s_sub_i32 s1, s0, s6 // (op 490) 


s_mul_i32 s0, s14, s18 // (op 490) 

s_mul_i32 s6, s50, s0 // (op 490) 

s_mul_i32 s0, s1, s6 // (op 490) 

s_sub_u32 s6, s16, s0 // (op 490) 

s_mul_hi_i32 s0, s6, s13 // (op 490) 
s_add_i32 s7, s0, s6 // (op 490) 


s_ashr_i32 s0, s7, 31 // (op 490) 
s_cmp_eq_i32 s13, 0 // (op 490) 
s_mov_b32 s6, scc // (op 490) 
s_cmp_eq_i32 s6, 0 // (op 490) 
s_cselect_b32 s18, 0, -1 // (op 490) 

s_and_b32 s6, s49, 31 // (op 490) 
s_lshl_b32 s19, 1, s6 // (op 490) 
s_add_i32 s68, s19, s18 // (op 490) 


s_and_b32 s18, s0, s68 // (op 490) 


s_add_i32 s0, s7, s18 // (op 490) 


s_ashr_i32 s7, s0, s6 // (op 490) 


s_lshl_b32 s0, s49, 24 // (op 490) 
s_ashr_i32 s6, s0, 31 // (op 490) 

s_xor_b32 s0, s7, s6 // (op 490) 

s_sub_i32 s7, s0, s6 // (op 490) 


s_cmp_eq_i32 s1, 0 // (op 490) 
s_mov_b32 s0, scc // (op 490) 

s_cmp_eq_i32 s0, 0 // (op 490) 
s_cselect_b32 s1, s7, s51 // (op 490) 



v_lshl_add_u32 v13, s1, 4, v11 // (op 490) 


v_mul_lo_u32 v35, v13, s27 // (op 490) most significant: low of low * high
v_mul_hi_u32 v11, v13, s26 // (op 490) most significant: high of low * low
v_mul_lo_u32 v34, v13, s26 // (op 490) least significant: low of low * low
v_add_u32 v35, v35, v11 // (op 490) most significant: sum


s_ashr_i64 s[0:1], s[24:25], 7 // (op 490) 
s_lshr_b32 s51, s0, 1 // (op 490) 

v_lshl_add_u32 v11, s51, 8, v12 // (op 490) 


v_mov_b32 v13, 0 // (op 490) convert
v_mov_b32 v12, v11 // (op 490) convert

v_add_co_u32 v14, s[0:1], v12, v34 // (op 490) least significant half
v_addc_co_u32 v15, s[0:1], v13, v35, s[0:1] // (op 490) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 490) 


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 490) 

v_mov_b32 v11, v12 // (op 490) convert

s_mul_i32 s1, s26, s5 // (op 492) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 492) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 492) least significant
s_mul_hi_u32 s6, s26, s4 // (op 492) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 492) most significant: sum
s_add_u32 s1, s1, s6 // (op 492) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 492) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 492) 

s_mov_b32 s56, s28 // (op 492) reg expression
s_mov_b32 s57, s29 // (op 492) reg expression
s_mov_b32 s58, s0 // (op 492) reg expression
s_mov_b32 s59, 131072 // (op 492) call()

s_mul_hi_i32 s51, s16, s9 // (op 554) 
v_and_b32 v12, -32, v1 // (op 554) 
s_add_i32 s0, s51, s16 // (op 554) 

v_sub_u32 v13, v1, v12 // (op 554) 

s_ashr_i32 s1, s0, 31 // (op 554) 
v_lshrrev_b32 v12, 5, v1 // (op 554) 
s_cmp_eq_i32 s9, 0 // (op 554) 
s_mov_b32 s51, scc // (op 554) 
s_cmp_eq_i32 s51, 0 // (op 554) 
s_cselect_b32 s6, 0, -1 // (op 554) 

s_and_b32 s7, s10, 31 // (op 554) 
s_lshl_b32 s51, 1, s7 // (op 554) 
s_add_i32 s18, s51, s6 // (op 554) 


s_and_b32 s6, s1, s18 // (op 554) 


s_add_i32 s1, s0, s6 // (op 554) 


s_ashr_i32 s0, s1, s7 // (op 554) 


s_lshl_b32 s1, s10, 24 // (op 554) 
s_ashr_i32 s51, s1, 31 // (op 554) 

s_xor_b32 s1, s0, s51 // (op 554) 

s_sub_i32 s0, s1, s51 // (op 554) 


s_mul_hi_i32 s1, s0, s11 // (op 554) 
s_add_i32 s51, s1, s0 // (op 554) 

s_ashr_i32 s1, s51, 31 // (op 554) 
s_cmp_eq_i32 s11, 0 // (op 554) 
s_mov_b32 s6, scc // (op 554) 
s_cmp_eq_i32 s6, 0 // (op 554) 
s_cselect_b32 s7, 0, -1 // (op 554) 

s_and_b32 s6, s12, 31 // (op 554) 
s_lshl_b32 s18, 1, s6 // (op 554) 
s_add_i32 s19, s18, s7 // (op 554) 


s_and_b32 s7, s1, s19 // (op 554) 


s_add_i32 s1, s51, s7 // (op 554) 


s_ashr_i32 s7, s1, s6 // (op 554) 


s_lshl_b32 s1, s12, 24 // (op 554) 
s_ashr_i32 s6, s1, 31 // (op 554) 

s_xor_b32 s1, s7, s6 // (op 554) 

s_sub_i32 s7, s1, s6 // (op 554) 


s_mul_i32 s1, s0, s14 // (op 554) 

s_sub_u32 s0, s16, s1 // (op 554) 

s_mul_i32 s1, s7, s14 // (op 554) 
s_add_u32 s1, s1, s0 // (op 554) 


s_mul_hi_i32 s0, s16, s8 // (op 554) 
s_add_i32 s51, s0, s16 // (op 554) 

s_ashr_i32 s0, s51, 31 // (op 554) 
s_cmp_eq_i32 s8, 0 // (op 554) 
s_mov_b32 s6, scc // (op 554) 
s_cmp_eq_i32 s6, 0 // (op 554) 
s_cselect_b32 s7, 0, -1 // (op 554) 

s_and_b32 s6, s15, 31 // (op 554) 
s_lshl_b32 s18, 1, s6 // (op 554) 
s_add_i32 s19, s18, s7 // (op 554) 


s_and_b32 s7, s0, s19 // (op 554) 


s_add_i32 s0, s51, s7 // (op 554) 


s_ashr_i32 s7, s0, s6 // (op 554) 


s_lshl_b32 s0, s15, 24 // (op 554) 
s_ashr_i32 s6, s0, 31 // (op 554) 

s_xor_b32 s0, s7, s6 // (op 554) 

s_sub_i32 s7, s0, s6 // (op 554) 


s_add_u32 s18, s4, 16 // (op 554) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 554) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 554) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 554) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 554) 

s_mul_i32 s0, s14, s18 // (op 554) 

s_mul_i32 s6, s50, s0 // (op 554) 

s_mul_i32 s0, s7, s6 // (op 554) 

s_sub_u32 s6, s16, s0 // (op 554) 

s_mov_b32 s0, s6 // (op 554) 
s_mul_hi_i32 s51, s0, s13 // (op 554) 
s_add_i32 s18, s51, s0 // (op 554) 


s_ashr_i32 s0, s18, 31 // (op 554) 
s_cmp_eq_i32 s13, 0 // (op 554) 
s_mov_b32 s19, scc // (op 554) 
s_cmp_eq_i32 s19, 0 // (op 554) 
s_cselect_b32 s51, 0, -1 // (op 554) 

s_and_b32 s19, s49, 31 // (op 554) 
s_lshl_b32 s68, 1, s19 // (op 554) 
s_add_i32 s69, s68, s51 // (op 554) 


s_and_b32 s51, s0, s69 // (op 554) 


s_add_i32 s0, s18, s51 // (op 554) 


s_ashr_i32 s18, s0, s19 // (op 554) 


s_lshl_b32 s0, s49, 24 // (op 554) 
s_ashr_i32 s19, s0, 31 // (op 554) 

s_xor_b32 s0, s18, s19 // (op 554) 

s_sub_i32 s18, s0, s19 // (op 554) 


s_mul_i32 s0, s48, s14 // (op 554) 
s_add_u32 s68, s20, 32 // (op 554) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 554) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 554) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 554) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 554) 

s_sub_i32 s19, s68, s0 // (op 554) 

s_mul_i32 s51, s18, s19 // (op 554) 


s_sub_u32 s18, s6, s51 // (op 554) 


s_add_u32 s6, s18, s0 // (op 554) 


s_cmp_eq_i32 s7, 0 // (op 554) 
s_mov_b32 s0, scc // (op 554) 

s_cmp_eq_i32 s0, 0 // (op 554) 
s_cselect_b32 s7, s6, s1 // (op 554) 



v_lshl_add_u32 v14, s7, 5, v13 // (op 554) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 554) 
s_lshr_b32 s51, s0, 1 // (op 554) 

v_lshl_add_u32 v13, s51, 3, v12 // (op 554) 


v_mul_lo_u32 v12, v13, s30 // (op 554) 

v_add_u32 v13, v14, v12 // (op 554) 


s_mul_i32 s1, s30, s35 // (op 556) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 556) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 556) least significant
s_mul_hi_u32 s6, s30, s34 // (op 556) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 556) most significant: sum
s_add_u32 s1, s1, s6 // (op 556) most significant: sum


s_mov_b32 s60, s32 // (op 556) reg expression
s_mov_b32 s61, s33 // (op 556) reg expression
s_mov_b32 s62, s0 // (op 556) reg expression
s_mov_b32 s63, 131072 // (op 556) call()

s_mul_hi_i32 s51, s16, s9 // (op 618) 
v_and_b32 v12, 15, v1 // (op 618) 
s_add_i32 s0, s51, s16 // (op 618) 

v_and_b32 v14, 63, v1 // (op 618) 
s_ashr_i32 s1, s0, 31 // (op 618) 
v_lshrrev_b32 v15, 4, v14 // (op 618) 

s_cmp_eq_i32 s9, 0 // (op 618) 
s_mov_b32 s51, scc // (op 618) 
s_cmp_eq_i32 s51, 0 // (op 618) 
s_cselect_b32 s6, 0, -1 // (op 618) 

s_and_b32 s7, s10, 31 // (op 618) 
s_lshl_b32 s51, 1, s7 // (op 618) 
s_add_i32 s18, s51, s6 // (op 618) 


s_and_b32 s6, s1, s18 // (op 618) 


s_add_i32 s1, s0, s6 // (op 618) 


s_ashr_i32 s0, s1, s7 // (op 618) 


s_lshl_b32 s1, s10, 24 // (op 618) 
s_ashr_i32 s51, s1, 31 // (op 618) 

s_xor_b32 s1, s0, s51 // (op 618) 

s_sub_i32 s0, s1, s51 // (op 618) 


s_mul_hi_i32 s1, s0, s11 // (op 618) 
s_add_i32 s51, s1, s0 // (op 618) 

s_ashr_i32 s1, s51, 31 // (op 618) 
s_cmp_eq_i32 s11, 0 // (op 618) 
s_mov_b32 s6, scc // (op 618) 
s_cmp_eq_i32 s6, 0 // (op 618) 
s_cselect_b32 s7, 0, -1 // (op 618) 

s_and_b32 s6, s12, 31 // (op 618) 
s_lshl_b32 s18, 1, s6 // (op 618) 
s_add_i32 s19, s18, s7 // (op 618) 


s_and_b32 s7, s1, s19 // (op 618) 


s_add_i32 s1, s51, s7 // (op 618) 


s_ashr_i32 s7, s1, s6 // (op 618) 


s_lshl_b32 s1, s12, 24 // (op 618) 
s_ashr_i32 s6, s1, 31 // (op 618) 

s_xor_b32 s1, s7, s6 // (op 618) 

s_sub_i32 s7, s1, s6 // (op 618) 


s_add_u32 s18, s4, 16 // (op 618) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 618) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 618) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 618) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 618) 

s_mul_i32 s1, s7, s18 // (op 618) 

s_sub_i32 s51, s0, s1 // (op 618) 


s_mul_hi_i32 s0, s16, s8 // (op 618) 
s_add_i32 s1, s0, s16 // (op 618) 

s_ashr_i32 s0, s1, 31 // (op 618) 
s_cmp_eq_i32 s8, 0 // (op 618) 
s_mov_b32 s6, scc // (op 618) 
s_cmp_eq_i32 s6, 0 // (op 618) 
s_cselect_b32 s7, 0, -1 // (op 618) 

s_and_b32 s6, s15, 31 // (op 618) 
s_lshl_b32 s68, 1, s6 // (op 618) 
s_add_i32 s69, s68, s7 // (op 618) 


s_and_b32 s7, s0, s69 // (op 618) 


s_add_i32 s0, s1, s7 // (op 618) 


s_ashr_i32 s1, s0, s6 // (op 618) 


s_lshl_b32 s0, s15, 24 // (op 618) 
s_ashr_i32 s6, s0, 31 // (op 618) 

s_xor_b32 s0, s1, s6 // (op 618) 

s_sub_i32 s1, s0, s6 // (op 618) 


s_mul_i32 s0, s14, s18 // (op 618) 

s_mul_i32 s6, s50, s0 // (op 618) 

s_mul_i32 s0, s1, s6 // (op 618) 

s_sub_u32 s6, s16, s0 // (op 618) 

s_mul_hi_i32 s0, s6, s13 // (op 618) 
s_add_i32 s7, s0, s6 // (op 618) 


s_ashr_i32 s0, s7, 31 // (op 618) 
s_cmp_eq_i32 s13, 0 // (op 618) 
s_mov_b32 s6, scc // (op 618) 
s_cmp_eq_i32 s6, 0 // (op 618) 
s_cselect_b32 s18, 0, -1 // (op 618) 

s_and_b32 s6, s49, 31 // (op 618) 
s_lshl_b32 s19, 1, s6 // (op 618) 
s_add_i32 s68, s19, s18 // (op 618) 


s_and_b32 s18, s0, s68 // (op 618) 


s_add_i32 s0, s7, s18 // (op 618) 


s_ashr_i32 s7, s0, s6 // (op 618) 


s_lshl_b32 s0, s49, 24 // (op 618) 
s_ashr_i32 s6, s0, 31 // (op 618) 

s_xor_b32 s0, s7, s6 // (op 618) 

s_sub_i32 s7, s0, s6 // (op 618) 


s_cmp_eq_i32 s1, 0 // (op 618) 
s_mov_b32 s0, scc // (op 618) 

s_cmp_eq_i32 s0, 0 // (op 618) 
s_cselect_b32 s1, s7, s51 // (op 618) 



v_lshl_add_u32 v14, s1, 4, v12 // (op 618) 


v_mul_lo_u32 v12, v14, s36 // (op 618) 

s_ashr_i64 s[0:1], s[24:25], 7 // (op 618) 
s_lshr_b32 s51, s0, 1 // (op 618) 

v_lshl_add_u32 v14, s51, 3, v15 // (op 618) 


v_add_u32 v15, v14, v12 // (op 618) 


s_mul_i32 s1, s36, s5 // (op 620) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 620) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 620) least significant
s_mul_hi_u32 s6, s36, s4 // (op 620) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 620) most significant: sum
s_add_u32 s1, s1, s6 // (op 620) most significant: sum


s_mov_b32 s64, s38 // (op 620) reg expression
s_mov_b32 s65, s39 // (op 620) reg expression
s_mov_b32 s66, s0 // (op 620) reg expression
s_mov_b32 s67, 131072 // (op 620) call()

v_and_b32 v12, -2, v1 // (op 711) 
v_sub_u32 v14, v1, v12 // (op 711) 

v_lshlrev_b32 v12, 4, v14 // (op 711) 

v_lshrrev_b32 v14, 1, v1 // (op 711) 
v_lshl_add_u32 v16, v14, 6, v12 // (op 711) 


v_and_b32 v12, -8, v1 // (op 764) 
v_sub_u32 v14, v1, v12 // (op 764) 

v_lshlrev_b32 v12, 4, v14 // (op 764) 

v_lshrrev_b32 v14, 3, v1 // (op 764) 
v_lshl_add_u32 v17, v14, 7, v12 // (op 764) 


v_lshlrev_b32 v12, 2, v17 // (op 764) 

v_lshrrev_b32 v14, 3, v12 // (op 764) 

v_and_b32 v12, -32, v1 // (op 817) 
v_sub_u32 v17, v1, v12 // (op 817) 

v_and_b32 v12, 4294967264, v1 // (op 817) 
v_add_u32 v18, v12, v17 // (op 817) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 162) 
s_and_b32 s51, s0, 4294967294 // (op 162) 

s_ashr_i64 s[0:1], s[24:25], 7 // (op 161) 
s_mov_b32 s6, s0 // (op 161) convert

s_cmp_lt_u32 s51, s6 // (op 161) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_161:

s_add_u32 s51, s51, 1 // (op 163) 
v_mov_b64 v[20:21], s[2:3] // (op 166) convert

v_mov_b32 v12, v10 // (op 166) 
v_mov_b32 v17, v12 // (op 166) reg expression
buffer_load_dwordx4 v[22:25], v17, s[52:55], 0 offen // (op 166) Load value
v_add_u32 v17, v17, v20 // (op 166) 
buffer_load_dwordx4 v[26:29], v17, s[52:55], 0 offen // (op 166) Load value



s_lshl_b32 s0, s2, 7 // (op 413) 
v_add_u32 v10, v10, s0 // (op 413) 



s_barrier  // (op 990) 
v_mov_b32 v12, v11 // (op 179) 




buffer_load_dwordx2 v[20:21], v12, s[56:59], 0 offen // (op 179) Load value

v_add_u32 v11, 64, v11 // (op 482) 
s_waitcnt vmcnt(2)
ds_write_b128 v16, v[22:25] offset:1024 // (op 167) Store local data 


v_mov_b32 v12, v13 // (op 180) 
v_mov_b32 v17, v12 // (op 180) reg expression

buffer_load_ubyte v19, v17, s[60:63], 0 offen offset: 0 // (op 180) Load value


s_lshl_b32 s0, s30, 2 // (op 546) 
v_add_u32 v13, v13, s0 // (op 546) 

s_waitcnt vmcnt(2)
ds_write_b128 v16, v[26:29] offset:1056 // (op 167) Store local data 

s_barrier  // (op 1003) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 993) 






ds_read_b64_tr_b8 v[22:23], v0 offset:1024 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:1280 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:3072 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:3328 // (op 169) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b64 v14, v[20:21] // (op 177) Store local data 

s_barrier  // (op 1017) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1007) 






ds_read_b128 v[30:33], v4 // (op 171) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v19 offset:5120 // (op 178) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1021) 


ds_read_u8 v12, v3 offset:5120 // (op 173) Load local data 


v_mov_b32 v17, v15 // (op 175) 

buffer_load_ubyte v19, v17, s[64:67], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v15, 4, v15 // (op 610) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:33], a[0:3], v12, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 176) 




s_ashr_i64 s[0:1], s[24:25], 7 // (op 161) 
s_mov_b32 s6, s0 // (op 161) convert

s_cmp_lt_u32 s51, s6 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_161 if true)
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_161:






















s_barrier  // (op 1085) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

s_mul_hi_i32 s0, s16, s9 // (op 859) 
v_lshrrev_b32 v0, 6, v1 // (op 859) 
s_add_i32 s1, s0, s16 // (op 859) 

v_and_b32 v3, 63, v1 // (op 859) 
s_ashr_i32 s0, s1, 31 // (op 859) 
v_lshrrev_b32 v4, 4, v3 // (op 859) 
s_cmp_eq_i32 s9, 0 // (op 859) 
s_mov_b32 s2, scc // (op 859) 
v_lshlrev_b32 v5, 2, v4 // (op 859) 

s_cmp_eq_i32 s2, 0 // (op 859) 
s_cselect_b32 s3, 0, -1 // (op 859) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 859) 


s_and_b32 s2, s10, 31 // (op 859) 
v_lshrrev_b32 v0, 2, v3 // (op 859) 

s_lshl_b32 s6, 1, s2 // (op 859) 
v_and_b32 v3, 3, v0 // (op 859) 

s_add_i32 s7, s6, s3 // (op 859) 


v_and_b32 v0, 3, v1 // (op 859) 
s_and_b32 s3, s0, s7 // (op 859) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 859) 


s_add_i32 s0, s1, s3 // (op 859) 


s_ashr_i32 s1, s0, s2 // (op 859) 


s_lshl_b32 s0, s10, 24 // (op 859) 
s_ashr_i32 s2, s0, 31 // (op 859) 

s_xor_b32 s0, s1, s2 // (op 859) 

s_sub_i32 s1, s0, s2 // (op 859) 


s_mul_hi_i32 s0, s1, s11 // (op 859) 
s_add_i32 s2, s0, s1 // (op 859) 

s_ashr_i32 s0, s2, 31 // (op 859) 
s_cmp_eq_i32 s11, 0 // (op 859) 
s_mov_b32 s3, scc // (op 859) 
s_cmp_eq_i32 s3, 0 // (op 859) 
s_cselect_b32 s6, 0, -1 // (op 859) 

s_and_b32 s3, s12, 31 // (op 859) 
s_lshl_b32 s7, 1, s3 // (op 859) 
s_add_i32 s18, s7, s6 // (op 859) 


s_and_b32 s19, s0, s18 // (op 859) 


s_add_i32 s0, s2, s19 // (op 859) 


s_ashr_i32 s2, s0, s3 // (op 859) 


s_lshl_b32 s0, s12, 24 // (op 859) 
s_ashr_i32 s3, s0, 31 // (op 859) 

s_xor_b32 s0, s2, s3 // (op 859) 

s_sub_i32 s2, s0, s3 // (op 859) 


s_mul_i32 s0, s1, s14 // (op 859) 
s_sub_u32 s3, s16, s0 // (op 859) 

s_mul_i32 s0, s2, s14 // (op 859) 
s_add_u32 s0, s0, s3 // (op 859) 

s_mul_hi_i32 s3, s16, s8 // (op 859) 
s_add_i32 s6, s3, s16 // (op 859) 

s_ashr_i32 s3, s6, 31 // (op 859) 
s_cmp_eq_i32 s8, 0 // (op 859) 
s_mov_b32 s7, scc // (op 859) 
s_cmp_eq_i32 s7, 0 // (op 859) 
s_cselect_b32 s18, 0, -1 // (op 859) 

s_and_b32 s7, s15, 31 // (op 859) 
s_lshl_b32 s19, 1, s7 // (op 859) 
s_add_i32 s22, s19, s18 // (op 859) 


s_and_b32 s18, s3, s22 // (op 859) 


s_add_i32 s3, s6, s18 // (op 859) 


s_ashr_i32 s6, s3, s7 // (op 859) 


s_lshl_b32 s3, s15, 24 // (op 859) 
s_ashr_i32 s7, s3, 31 // (op 859) 

s_xor_b32 s3, s6, s7 // (op 859) 

s_sub_i32 s6, s3, s7 // (op 859) 


s_add_u32 s18, s4, 16 // (op 859) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 859) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 859) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 859) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 4 // (op 859) 

s_mul_i32 s3, s14, s18 // (op 859) 
s_mul_i32 s7, s50, s3 // (op 859) 

s_mul_i32 s3, s6, s7 // (op 859) 

s_sub_u32 s7, s16, s3 // (op 859) 

s_mov_b32 s3, s7 // (op 859) 
s_mul_hi_i32 s22, s3, s13 // (op 859) 
s_add_i32 s23, s22, s3 // (op 859) 


s_ashr_i32 s3, s23, 31 // (op 859) 
s_cmp_eq_i32 s13, 0 // (op 859) 
s_mov_b32 s22, scc // (op 859) 
s_cmp_eq_i32 s22, 0 // (op 859) 
s_cselect_b32 s24, 0, -1 // (op 859) 

s_and_b32 s22, s49, 31 // (op 859) 
s_lshl_b32 s25, 1, s22 // (op 859) 
s_add_i32 s26, s25, s24 // (op 859) 


s_and_b32 s24, s3, s26 // (op 859) 


s_add_i32 s3, s23, s24 // (op 859) 


s_ashr_i32 s23, s3, s22 // (op 859) 


s_lshl_b32 s3, s49, 24 // (op 859) 
s_ashr_i32 s22, s3, 31 // (op 859) 

s_xor_b32 s3, s23, s22 // (op 859) 

s_sub_i32 s23, s3, s22 // (op 859) 


s_mul_i32 s3, s48, s14 // (op 859) 
s_add_u32 s24, s20, 32 // (op 859) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 859) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 859) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 859) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 5 // (op 859) 

s_sub_i32 s22, s24, s3 // (op 859) 

s_mul_i32 s24, s23, s22 // (op 859) 

s_sub_u32 s22, s7, s24 // (op 859) 


s_add_u32 s7, s22, s3 // (op 859) 


s_cmp_eq_i32 s6, 0 // (op 859) 
s_mov_b32 s3, scc // (op 859) 

s_cmp_eq_i32 s3, 0 // (op 859) 
s_cselect_b32 s6, s7, s0 // (op 859) 


s_mul_i32 s0, s2, s18 // (op 859) 


v_lshl_add_u32 v0, s6, 5, v4 // (op 859) 


s_sub_i32 s2, s1, s0 // (op 859) 


s_cmp_eq_i32 s3, 0 // (op 859) 
s_cselect_b32 s22, s23, s2 // (op 859) 



v_lshl_add_u32 v3, s22, 4, v5 // (op 859) 


v_mul_lo_u32 v4, v3, s40 // (op 859) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 859) 


s_mul_i32 s7, s40, s5 // (op 861) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 861) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 861) least significant
s_mul_hi_u32 s1, s40, s4 // (op 861) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 861) most significant: sum
s_add_u32 s7, s7, s1 // (op 861) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 861) 

s_mov_b32 s0, s42 // (op 861) reg expression
s_mov_b32 s1, s43 // (op 861) reg expression
s_mov_b32 s2, s18 // (op 861) reg expression
s_mov_b32 s3, 131072 // (op 861) call()




buffer_store_dwordx4 v[10:13], v3, s[0:3], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_123:
s_mul_hi_i32 s6, s16, s9 // (op 681) 
v_lshrrev_b32 v0, 6, v1 // (op 681) 
s_add_i32 s7, s6, s16 // (op 681) 

v_and_b32 v3, 63, v1 // (op 681) 
s_ashr_i32 s6, s7, 31 // (op 681) 
v_lshrrev_b32 v4, 4, v3 // (op 681) 
s_cmp_eq_i32 s9, 0 // (op 681) 
s_mov_b32 s18, scc // (op 681) 
v_lshlrev_b32 v5, 2, v4 // (op 681) 

s_cmp_eq_i32 s18, 0 // (op 681) 
s_cselect_b32 s19, 0, -1 // (op 681) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 681) 


s_and_b32 s18, s10, 31 // (op 681) 
v_lshrrev_b32 v0, 2, v3 // (op 681) 

s_lshl_b32 s22, 1, s18 // (op 681) 
v_and_b32 v3, 3, v0 // (op 681) 

s_add_i32 s23, s22, s19 // (op 681) 


v_and_b32 v0, 3, v1 // (op 681) 
s_and_b32 s19, s6, s23 // (op 681) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 681) 


s_add_i32 s6, s7, s19 // (op 681) 


s_ashr_i32 s7, s6, s18 // (op 681) 


s_lshl_b32 s6, s10, 24 // (op 681) 
s_ashr_i32 s18, s6, 31 // (op 681) 

s_xor_b32 s6, s7, s18 // (op 681) 

s_sub_i32 s7, s6, s18 // (op 681) 


s_mul_hi_i32 s6, s7, s11 // (op 681) 
s_add_i32 s18, s6, s7 // (op 681) 

s_ashr_i32 s6, s18, 31 // (op 681) 
s_cmp_eq_i32 s11, 0 // (op 681) 
s_mov_b32 s19, scc // (op 681) 
s_cmp_eq_i32 s19, 0 // (op 681) 
s_cselect_b32 s22, 0, -1 // (op 681) 

s_and_b32 s19, s12, 31 // (op 681) 
s_lshl_b32 s23, 1, s19 // (op 681) 
s_add_i32 s24, s23, s22 // (op 681) 


s_and_b32 s22, s6, s24 // (op 681) 


s_add_i32 s6, s18, s22 // (op 681) 


s_ashr_i32 s18, s6, s19 // (op 681) 


s_lshl_b32 s6, s12, 24 // (op 681) 
s_ashr_i32 s19, s6, 31 // (op 681) 

s_xor_b32 s6, s18, s19 // (op 681) 

s_sub_i32 s18, s6, s19 // (op 681) 


s_mul_i32 s6, s7, s14 // (op 681) 
s_sub_u32 s19, s16, s6 // (op 681) 

s_mul_i32 s6, s18, s14 // (op 681) 
s_add_u32 s6, s6, s19 // (op 681) 

s_mul_hi_i32 s19, s16, s8 // (op 681) 
s_add_i32 s22, s19, s16 // (op 681) 

s_ashr_i32 s19, s22, 31 // (op 681) 
s_cmp_eq_i32 s8, 0 // (op 681) 
s_mov_b32 s23, scc // (op 681) 
s_cmp_eq_i32 s23, 0 // (op 681) 
s_cselect_b32 s24, 0, -1 // (op 681) 

s_and_b32 s23, s15, 31 // (op 681) 
s_lshl_b32 s25, 1, s23 // (op 681) 
s_add_i32 s26, s25, s24 // (op 681) 


s_and_b32 s24, s19, s26 // (op 681) 


s_add_i32 s19, s22, s24 // (op 681) 


s_ashr_i32 s22, s19, s23 // (op 681) 


s_lshl_b32 s19, s15, 24 // (op 681) 
s_ashr_i32 s23, s19, 31 // (op 681) 

s_xor_b32 s19, s22, s23 // (op 681) 

s_sub_i32 s22, s19, s23 // (op 681) 


s_add_u32 s38, s4, 16 // (op 681) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 681) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 681) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 681) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 681) 

s_mul_i32 s19, s14, s38 // (op 681) 
s_mul_i32 s23, s50, s19 // (op 681) 

s_mul_i32 s19, s22, s23 // (op 681) 

s_sub_u32 s23, s16, s19 // (op 681) 

s_mov_b32 s19, s23 // (op 681) 
s_mul_hi_i32 s24, s19, s13 // (op 681) 
s_add_i32 s25, s24, s19 // (op 681) 


s_ashr_i32 s19, s25, 31 // (op 681) 
s_cmp_eq_i32 s13, 0 // (op 681) 
s_mov_b32 s24, scc // (op 681) 
s_cmp_eq_i32 s24, 0 // (op 681) 
s_cselect_b32 s26, 0, -1 // (op 681) 

s_and_b32 s24, s49, 31 // (op 681) 
s_lshl_b32 s27, 1, s24 // (op 681) 
s_add_i32 s28, s27, s26 // (op 681) 


s_and_b32 s26, s19, s28 // (op 681) 


s_add_i32 s19, s25, s26 // (op 681) 


s_ashr_i32 s25, s19, s24 // (op 681) 


s_lshl_b32 s19, s49, 24 // (op 681) 
s_ashr_i32 s24, s19, 31 // (op 681) 

s_xor_b32 s19, s25, s24 // (op 681) 

s_sub_i32 s25, s19, s24 // (op 681) 


s_mul_i32 s19, s48, s14 // (op 681) 
s_add_u32 s26, s20, 32 // (op 681) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 681) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 681) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 681) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 5 // (op 681) 

s_sub_i32 s24, s26, s19 // (op 681) 

s_mul_i32 s26, s25, s24 // (op 681) 

s_sub_u32 s24, s23, s26 // (op 681) 


s_add_u32 s23, s24, s19 // (op 681) 


s_cmp_eq_i32 s22, 0 // (op 681) 
s_mov_b32 s19, scc // (op 681) 

s_cmp_eq_i32 s19, 0 // (op 681) 
s_cselect_b32 s22, s23, s6 // (op 681) 


s_mul_i32 s6, s18, s38 // (op 681) 


v_lshl_add_u32 v0, s22, 5, v4 // (op 681) 


s_sub_i32 s18, s7, s6 // (op 681) 


s_cmp_eq_i32 s19, 0 // (op 681) 
s_cselect_b32 s6, s25, s18 // (op 681) 



v_lshl_add_u32 v3, s6, 4, v5 // (op 681) 


v_mul_lo_u32 v4, v3, s44 // (op 681) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 681) 


s_mul_i32 s7, s44, s5 // (op 683) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 683) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 683) least significant
s_mul_hi_u32 s19, s44, s4 // (op 683) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 683) most significant: sum
s_add_u32 s7, s7, s19 // (op 683) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 683) 

s_mov_b32 s36, s46 // (op 683) reg expression
s_mov_b32 s37, s47 // (op 683) reg expression
s_mov_b32 s38, s18 // (op 683) reg expression
s_mov_b32 s39, 131072 // (op 683) call()





buffer_load_dwordx4 v[4:7], v3, s[36:39], 0 offen // (op 135) Load value



v_accvgpr_read v10, a0 // (op 132) 
v_accvgpr_read v11, a1 // (op 132) 
v_accvgpr_read v12, a2 // (op 132) 
v_accvgpr_read v13, a3 // (op 132) 
v_mul_f32 v14, v9, v10 // (op 132) 
v_mul_f32 v15, v9, v11 // (op 132) 
v_mul_f32 v16, v9, v12 // (op 132) 
v_mul_f32 v17, v9, v13 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v8, v4 // (op 136) 
v_mul_f32 v11, v8, v5 // (op 136) 
v_mul_f32 v12, v8, v6 // (op 136) 
v_mul_f32 v13, v8, v7 // (op 136) 

v_add_f32 v4, v10, v14 // (op 133) 
v_add_f32 v5, v11, v15 // (op 133) 
v_add_f32 v6, v12, v16 // (op 133) 
v_add_f32 v7, v13, v17 // (op 133) 


s_mul_hi_i32 s6, s16, s9 // (op 876) 
v_lshrrev_b32 v0, 6, v1 // (op 876) 
s_add_i32 s7, s6, s16 // (op 876) 

v_and_b32 v3, 63, v1 // (op 876) 
s_ashr_i32 s6, s7, 31 // (op 876) 
v_lshrrev_b32 v10, 4, v3 // (op 876) 
s_cmp_eq_i32 s9, 0 // (op 876) 
s_mov_b32 s18, scc // (op 876) 
v_lshlrev_b32 v11, 2, v10 // (op 876) 

s_cmp_eq_i32 s18, 0 // (op 876) 
s_cselect_b32 s19, 0, -1 // (op 876) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 876) 


s_and_b32 s18, s10, 31 // (op 876) 
v_lshrrev_b32 v0, 2, v3 // (op 876) 

s_lshl_b32 s22, 1, s18 // (op 876) 
v_and_b32 v3, 3, v0 // (op 876) 

s_add_i32 s23, s22, s19 // (op 876) 


v_and_b32 v0, 3, v1 // (op 876) 
s_and_b32 s19, s6, s23 // (op 876) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 876) 


s_add_i32 s6, s7, s19 // (op 876) 


s_ashr_i32 s7, s6, s18 // (op 876) 


s_lshl_b32 s6, s10, 24 // (op 876) 
s_ashr_i32 s18, s6, 31 // (op 876) 

s_xor_b32 s6, s7, s18 // (op 876) 

s_sub_i32 s7, s6, s18 // (op 876) 


s_mul_hi_i32 s6, s7, s11 // (op 876) 
s_add_i32 s18, s6, s7 // (op 876) 

s_ashr_i32 s6, s18, 31 // (op 876) 
s_cmp_eq_i32 s11, 0 // (op 876) 
s_mov_b32 s19, scc // (op 876) 
s_cmp_eq_i32 s19, 0 // (op 876) 
s_cselect_b32 s22, 0, -1 // (op 876) 

s_and_b32 s19, s12, 31 // (op 876) 
s_lshl_b32 s23, 1, s19 // (op 876) 
s_add_i32 s24, s23, s22 // (op 876) 


s_and_b32 s22, s6, s24 // (op 876) 


s_add_i32 s6, s18, s22 // (op 876) 


s_ashr_i32 s18, s6, s19 // (op 876) 


s_lshl_b32 s6, s12, 24 // (op 876) 
s_ashr_i32 s19, s6, 31 // (op 876) 

s_xor_b32 s6, s18, s19 // (op 876) 

s_sub_i32 s18, s6, s19 // (op 876) 


s_mul_i32 s6, s7, s14 // (op 876) 
s_sub_u32 s19, s16, s6 // (op 876) 

s_mul_i32 s6, s18, s14 // (op 876) 
s_add_u32 s6, s6, s19 // (op 876) 

s_mul_hi_i32 s19, s16, s8 // (op 876) 
s_add_i32 s22, s19, s16 // (op 876) 

s_ashr_i32 s19, s22, 31 // (op 876) 
s_cmp_eq_i32 s8, 0 // (op 876) 
s_mov_b32 s23, scc // (op 876) 
s_cmp_eq_i32 s23, 0 // (op 876) 
s_cselect_b32 s24, 0, -1 // (op 876) 

s_and_b32 s23, s15, 31 // (op 876) 
s_lshl_b32 s25, 1, s23 // (op 876) 
s_add_i32 s26, s25, s24 // (op 876) 


s_and_b32 s24, s19, s26 // (op 876) 


s_add_i32 s19, s22, s24 // (op 876) 


s_ashr_i32 s22, s19, s23 // (op 876) 


s_lshl_b32 s19, s15, 24 // (op 876) 
s_ashr_i32 s23, s19, 31 // (op 876) 

s_xor_b32 s19, s22, s23 // (op 876) 

s_sub_i32 s22, s19, s23 // (op 876) 


s_add_u32 s38, s4, 16 // (op 876) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 876) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 876) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 876) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 876) 

s_mul_i32 s19, s14, s38 // (op 876) 
s_mul_i32 s23, s50, s19 // (op 876) 

s_mul_i32 s19, s22, s23 // (op 876) 

s_sub_u32 s23, s16, s19 // (op 876) 

s_mov_b32 s19, s23 // (op 876) 
s_mul_hi_i32 s24, s19, s13 // (op 876) 
s_add_i32 s25, s24, s19 // (op 876) 


s_ashr_i32 s19, s25, 31 // (op 876) 
s_cmp_eq_i32 s13, 0 // (op 876) 
s_mov_b32 s24, scc // (op 876) 
s_cmp_eq_i32 s24, 0 // (op 876) 
s_cselect_b32 s26, 0, -1 // (op 876) 

s_and_b32 s24, s49, 31 // (op 876) 
s_lshl_b32 s27, 1, s24 // (op 876) 
s_add_i32 s28, s27, s26 // (op 876) 


s_and_b32 s26, s19, s28 // (op 876) 


s_add_i32 s19, s25, s26 // (op 876) 


s_ashr_i32 s25, s19, s24 // (op 876) 


s_lshl_b32 s19, s49, 24 // (op 876) 
s_ashr_i32 s24, s19, 31 // (op 876) 

s_xor_b32 s19, s25, s24 // (op 876) 

s_sub_i32 s25, s19, s24 // (op 876) 


s_mul_i32 s19, s48, s14 // (op 876) 
s_add_u32 s26, s20, 32 // (op 876) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 876) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 876) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 876) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 5 // (op 876) 

s_sub_i32 s24, s26, s19 // (op 876) 

s_mul_i32 s26, s25, s24 // (op 876) 

s_sub_u32 s24, s23, s26 // (op 876) 


s_add_u32 s23, s24, s19 // (op 876) 


s_cmp_eq_i32 s22, 0 // (op 876) 
s_mov_b32 s19, scc // (op 876) 

s_cmp_eq_i32 s19, 0 // (op 876) 
s_cselect_b32 s22, s23, s6 // (op 876) 


s_mul_i32 s6, s18, s38 // (op 876) 


v_lshl_add_u32 v0, s22, 5, v10 // (op 876) 


s_sub_i32 s18, s7, s6 // (op 876) 


s_cmp_eq_i32 s19, 0 // (op 876) 
s_cselect_b32 s6, s25, s18 // (op 876) 



v_lshl_add_u32 v3, s6, 4, v11 // (op 876) 


v_mul_lo_u32 v10, v3, s40 // (op 876) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 876) 


s_mul_i32 s7, s40, s5 // (op 878) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 878) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 878) least significant
s_mul_hi_u32 s19, s40, s4 // (op 878) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 878) most significant: sum
s_add_u32 s7, s7, s19 // (op 878) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 878) 

s_mov_b32 s0, s42 // (op 878) reg expression
s_mov_b32 s1, s43 // (op 878) reg expression
s_mov_b32 s2, s18 // (op 878) reg expression
s_mov_b32 s3, 131072 // (op 878) call()




buffer_store_dwordx4 v[4:7], v3, s[0:3], 0 offen // (op 134) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.LRR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__end:
.size RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_, .LRR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__end-RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 68
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 10496 // lds bytes
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: F4
    DestDataType: s
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
    macrotile: [32, 16, 128]
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
  - .name: RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 10496
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 62
    .agpr_count: 4
    .max_flat_workgroup_size: 128
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

