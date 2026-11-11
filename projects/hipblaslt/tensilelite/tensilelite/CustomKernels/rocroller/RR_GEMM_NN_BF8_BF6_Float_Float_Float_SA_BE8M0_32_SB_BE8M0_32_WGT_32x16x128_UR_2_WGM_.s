// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_,@function
RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 48, v1 // (op 761) 
v_and_b32 v3, 63, v1 // (op 761) 
v_sub_u32 v4, v3, v0 // (op 761) 

v_lshrrev_b32 v3, 1, v4 // (op 761) 
v_add_lshl_u32 v5, v0, v3, 5 // (op 761) 


v_and_b32 v0, 1, v4 // (op 761) 

v_lshlrev_b32 v3, 3, v0 // (op 761) 

v_lshrrev_b32 v0, 6, v1 // (op 761) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 761) 


v_add_u32 v0, v5, v4 // (op 761) 


v_and_b32 v3, 15, v1 // (op 814) 
v_and_b32 v4, 63, v1 // (op 814) 
v_lshrrev_b32 v5, 4, v4 // (op 814) 

v_lshlrev_b32 v4, 5, v5 // (op 814) 

v_lshl_add_u32 v5, v3, 7, v4 // (op 814) 


v_mul_lo_u32 v3, v5, 6 // (op 814) 

v_lshrrev_b32 v4, 3, v3 // (op 814) 

v_lshrrev_b32 v3, 6, v1 // (op 867) 
v_and_b32 v5, 15, v1 // (op 867) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 867) 


v_and_b32 v3, 63, v1 // (op 867) 
v_lshrrev_b32 v5, 4, v3 // (op 867) 

v_lshl_add_u32 v3, v5, 5, v6 // (op 867) 


v_and_b32 v5, 48, v1 // (op 940) 
v_and_b32 v6, 63, v1 // (op 940) 
v_sub_u32 v7, v6, v5 // (op 940) 

v_lshrrev_b32 v6, 1, v7 // (op 940) 
v_add_lshl_u32 v8, v5, v6, 5 // (op 940) 


v_and_b32 v5, 1, v7 // (op 940) 

v_lshlrev_b32 v6, 3, v5 // (op 940) 

v_lshrrev_b32 v5, 6, v1 // (op 940) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 940) 


v_add_u32 v5, v8, v7 // (op 940) 


v_and_b32 v6, 15, v1 // (op 964) 
v_and_b32 v7, 63, v1 // (op 964) 
v_lshrrev_b32 v8, 4, v7 // (op 964) 

v_lshlrev_b32 v7, 5, v8 // (op 964) 

v_lshl_add_u32 v8, v6, 7, v7 // (op 964) 


v_mul_lo_u32 v6, v8, 6 // (op 964) 

v_lshrrev_b32 v7, 3, v6 // (op 964) 

v_lshrrev_b32 v6, 6, v1 // (op 988) 
v_and_b32 v8, 15, v1 // (op 988) 
v_lshl_add_u32 v9, v6, 4, v8 // (op 988) 


v_and_b32 v6, 63, v1 // (op 988) 
v_lshrrev_b32 v8, 4, v6 // (op 988) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 988) 


v_mov_b32 v8, s6 // (op 2) Move value
v_mov_b32 v9, s7 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
s_mul_hi_i32 s0, s16, s9 // (op 472) 
v_and_b32 v10, -2, v1 // (op 472) 
s_add_i32 s1, s0, s16 // (op 472) 

v_sub_u32 v11, v1, v10 // (op 472) 

s_ashr_i32 s0, s1, 31 // (op 472) 
v_lshlrev_b32 v10, 4, v11 // (op 472) 

s_cmp_eq_i32 s9, 0 // (op 472) 
s_mov_b32 s6, scc // (op 472) 
v_lshrrev_b32 v11, 1, v1 // (op 472) 
s_cmp_eq_i32 s6, 0 // (op 472) 
s_cselect_b32 s7, 0, -1 // (op 472) 

s_and_b32 s6, s10, 31 // (op 472) 
s_lshl_b32 s18, 1, s6 // (op 472) 
s_add_i32 s19, s18, s7 // (op 472) 


s_and_b32 s7, s0, s19 // (op 472) 


s_add_i32 s0, s1, s7 // (op 472) 


s_ashr_i32 s1, s0, s6 // (op 472) 


s_lshl_b32 s0, s10, 24 // (op 472) 
s_ashr_i32 s6, s0, 31 // (op 472) 

s_xor_b32 s0, s1, s6 // (op 472) 

s_sub_i32 s1, s0, s6 // (op 472) 


s_mul_hi_i32 s0, s1, s11 // (op 472) 
s_add_i32 s6, s0, s1 // (op 472) 

s_ashr_i32 s0, s6, 31 // (op 472) 
s_cmp_eq_i32 s11, 0 // (op 472) 
s_mov_b32 s7, scc // (op 472) 
s_cmp_eq_i32 s7, 0 // (op 472) 
s_cselect_b32 s18, 0, -1 // (op 472) 

s_and_b32 s7, s12, 31 // (op 472) 
s_lshl_b32 s19, 1, s7 // (op 472) 
s_add_i32 s51, s19, s18 // (op 472) 


s_and_b32 s18, s0, s51 // (op 472) 


s_add_i32 s0, s6, s18 // (op 472) 


s_ashr_i32 s6, s0, s7 // (op 472) 


s_lshl_b32 s0, s12, 24 // (op 472) 
s_ashr_i32 s7, s0, 31 // (op 472) 

s_xor_b32 s0, s6, s7 // (op 472) 

s_sub_i32 s6, s0, s7 // (op 472) 


s_mul_i32 s0, s1, s14 // (op 472) 

s_sub_u32 s1, s16, s0 // (op 472) 

s_mul_i32 s0, s6, s14 // (op 472) 
s_add_u32 s0, s0, s1 // (op 472) 


s_mul_hi_i32 s1, s16, s8 // (op 472) 
s_add_i32 s6, s1, s16 // (op 472) 

s_ashr_i32 s1, s6, 31 // (op 472) 
s_cmp_eq_i32 s8, 0 // (op 472) 
s_mov_b32 s7, scc // (op 472) 
s_cmp_eq_i32 s7, 0 // (op 472) 
s_cselect_b32 s18, 0, -1 // (op 472) 

s_and_b32 s7, s15, 31 // (op 472) 
s_lshl_b32 s19, 1, s7 // (op 472) 
s_add_i32 s51, s19, s18 // (op 472) 


s_and_b32 s18, s1, s51 // (op 472) 


s_add_i32 s1, s6, s18 // (op 472) 


s_ashr_i32 s6, s1, s7 // (op 472) 


s_lshl_b32 s1, s15, 24 // (op 472) 
s_ashr_i32 s7, s1, 31 // (op 472) 

s_xor_b32 s1, s6, s7 // (op 472) 

s_sub_i32 s6, s1, s7 // (op 472) 


s_add_u32 s18, s4, 16 // (op 472) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 472) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 472) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 472) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 472) 

s_mul_i32 s1, s14, s18 // (op 472) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 472) 

s_mul_i32 s1, s6, s7 // (op 472) 

s_sub_u32 s7, s16, s1 // (op 472) 

s_mov_b32 s1, s7 // (op 472) 
s_mul_hi_i32 s18, s1, s13 // (op 472) 
s_add_i32 s19, s18, s1 // (op 472) 


s_ashr_i32 s1, s19, 31 // (op 472) 
s_cmp_eq_i32 s13, 0 // (op 472) 
s_mov_b32 s18, scc // (op 472) 
s_cmp_eq_i32 s18, 0 // (op 472) 
s_cselect_b32 s51, 0, -1 // (op 472) 

s_and_b32 s18, s49, 31 // (op 472) 
s_lshl_b32 s52, 1, s18 // (op 472) 
s_add_i32 s53, s52, s51 // (op 472) 


s_and_b32 s51, s1, s53 // (op 472) 


s_add_i32 s1, s19, s51 // (op 472) 


s_ashr_i32 s19, s1, s18 // (op 472) 


s_lshl_b32 s1, s49, 24 // (op 472) 
s_ashr_i32 s18, s1, 31 // (op 472) 

s_xor_b32 s1, s19, s18 // (op 472) 

s_sub_i32 s19, s1, s18 // (op 472) 


s_mul_i32 s1, s48, s14 // (op 472) 
s_add_u32 s52, s20, 32 // (op 472) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 472) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 472) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 472) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 5 // (op 472) 

s_sub_i32 s18, s52, s1 // (op 472) 

s_mul_i32 s51, s19, s18 // (op 472) 


s_sub_u32 s18, s7, s51 // (op 472) 


s_add_u32 s7, s18, s1 // (op 472) 


s_cmp_eq_i32 s6, 0 // (op 472) 
s_mov_b32 s1, scc // (op 472) 

s_cmp_eq_i32 s1, 0 // (op 472) 
s_cselect_b32 s6, s7, s0 // (op 472) 



v_lshl_add_u32 v12, s6, 5, v10 // (op 472) 


v_mov_b32 v15, 0 // (op 472) convert
v_mov_b32 v14, v12 // (op 472) convert

v_mul_lo_u32 v13, v11, s3 // (op 472) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 472) most significant: high of low * low
v_mul_lo_u32 v12, v11, s2 // (op 472) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 472) most significant: sum


v_add_co_u32 v10, s[0:1], v14, v12 // (op 472) least significant half
v_addc_co_u32 v11, s[0:1], v15, v13, s[0:1] // (op 472) most significant half



s_mul_i32 s1, s2, s25 // (op 474) most significant: low of low * high
s_mul_i32 s6, s3, s24 // (op 474) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 474) least significant
s_mul_hi_u32 s7, s2, s24 // (op 474) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 474) most significant: sum
s_add_u32 s1, s1, s7 // (op 474) most significant: sum


s_mov_b32 s52, s22 // (op 474) reg expression
s_mov_b32 s53, s23 // (op 474) reg expression
s_mov_b32 s54, s0 // (op 474) reg expression
s_mov_b32 s55, 131072 // (op 474) call()

s_mul_hi_i32 s51, s16, s9 // (op 491) 
v_and_b32 v12, -2, v1 // (op 491) 
s_add_i32 s0, s51, s16 // (op 491) 

v_sub_u32 v13, v1, v12 // (op 491) 

s_ashr_i32 s1, s0, 31 // (op 491) 
v_lshlrev_b32 v12, 4, v13 // (op 491) 

s_cmp_eq_i32 s9, 0 // (op 491) 
s_mov_b32 s51, scc // (op 491) 
v_lshrrev_b32 v13, 1, v1 // (op 491) 
s_cmp_eq_i32 s51, 0 // (op 491) 
s_cselect_b32 s6, 0, -1 // (op 491) 

s_and_b32 s7, s10, 31 // (op 491) 
s_lshl_b32 s51, 1, s7 // (op 491) 
s_add_i32 s18, s51, s6 // (op 491) 


s_and_b32 s6, s1, s18 // (op 491) 


s_add_i32 s1, s0, s6 // (op 491) 


s_ashr_i32 s0, s1, s7 // (op 491) 


s_lshl_b32 s1, s10, 24 // (op 491) 
s_ashr_i32 s51, s1, 31 // (op 491) 

s_xor_b32 s1, s0, s51 // (op 491) 

s_sub_i32 s0, s1, s51 // (op 491) 


s_mul_hi_i32 s1, s0, s11 // (op 491) 
s_add_i32 s51, s1, s0 // (op 491) 

s_ashr_i32 s1, s51, 31 // (op 491) 
s_cmp_eq_i32 s11, 0 // (op 491) 
s_mov_b32 s6, scc // (op 491) 
s_cmp_eq_i32 s6, 0 // (op 491) 
s_cselect_b32 s7, 0, -1 // (op 491) 

s_and_b32 s6, s12, 31 // (op 491) 
s_lshl_b32 s18, 1, s6 // (op 491) 
s_add_i32 s19, s18, s7 // (op 491) 


s_and_b32 s7, s1, s19 // (op 491) 


s_add_i32 s1, s51, s7 // (op 491) 


s_ashr_i32 s7, s1, s6 // (op 491) 


s_lshl_b32 s1, s12, 24 // (op 491) 
s_ashr_i32 s6, s1, 31 // (op 491) 

s_xor_b32 s1, s7, s6 // (op 491) 

s_sub_i32 s7, s1, s6 // (op 491) 


s_mul_i32 s1, s0, s14 // (op 491) 

s_sub_u32 s0, s16, s1 // (op 491) 

s_mul_i32 s1, s7, s14 // (op 491) 
s_add_u32 s1, s1, s0 // (op 491) 


s_mul_hi_i32 s0, s16, s8 // (op 491) 
s_add_i32 s51, s0, s16 // (op 491) 

s_ashr_i32 s0, s51, 31 // (op 491) 
s_cmp_eq_i32 s8, 0 // (op 491) 
s_mov_b32 s6, scc // (op 491) 
s_cmp_eq_i32 s6, 0 // (op 491) 
s_cselect_b32 s7, 0, -1 // (op 491) 

s_and_b32 s6, s15, 31 // (op 491) 
s_lshl_b32 s18, 1, s6 // (op 491) 
s_add_i32 s19, s18, s7 // (op 491) 


s_and_b32 s7, s0, s19 // (op 491) 


s_add_i32 s0, s51, s7 // (op 491) 


s_ashr_i32 s7, s0, s6 // (op 491) 


s_lshl_b32 s0, s15, 24 // (op 491) 
s_ashr_i32 s6, s0, 31 // (op 491) 

s_xor_b32 s0, s7, s6 // (op 491) 

s_sub_i32 s7, s0, s6 // (op 491) 


s_add_u32 s18, s4, 16 // (op 491) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 491) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 491) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 491) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 491) 

s_mul_i32 s0, s14, s18 // (op 491) 

s_mul_i32 s6, s50, s0 // (op 491) 

s_mul_i32 s0, s7, s6 // (op 491) 

s_sub_u32 s6, s16, s0 // (op 491) 

s_mov_b32 s0, s6 // (op 491) 
s_mul_hi_i32 s51, s0, s13 // (op 491) 
s_add_i32 s18, s51, s0 // (op 491) 


s_ashr_i32 s0, s18, 31 // (op 491) 
s_cmp_eq_i32 s13, 0 // (op 491) 
s_mov_b32 s19, scc // (op 491) 
s_cmp_eq_i32 s19, 0 // (op 491) 
s_cselect_b32 s51, 0, -1 // (op 491) 

s_and_b32 s19, s49, 31 // (op 491) 
s_lshl_b32 s56, 1, s19 // (op 491) 
s_add_i32 s57, s56, s51 // (op 491) 


s_and_b32 s51, s0, s57 // (op 491) 


s_add_i32 s0, s18, s51 // (op 491) 


s_ashr_i32 s18, s0, s19 // (op 491) 


s_lshl_b32 s0, s49, 24 // (op 491) 
s_ashr_i32 s19, s0, 31 // (op 491) 

s_xor_b32 s0, s18, s19 // (op 491) 

s_sub_i32 s18, s0, s19 // (op 491) 


s_mul_i32 s0, s48, s14 // (op 491) 
s_add_u32 s56, s20, 32 // (op 491) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 491) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 491) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 491) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 491) 

s_sub_i32 s19, s56, s0 // (op 491) 

s_mul_i32 s51, s18, s19 // (op 491) 


s_sub_u32 s18, s6, s51 // (op 491) 


s_add_u32 s6, s18, s0 // (op 491) 


s_cmp_eq_i32 s7, 0 // (op 491) 
s_mov_b32 s0, scc // (op 491) 

s_cmp_eq_i32 s0, 0 // (op 491) 
s_cselect_b32 s7, s6, s1 // (op 491) 



v_lshl_add_u32 v14, s7, 5, v12 // (op 491) 


v_mov_b32 v17, 0 // (op 491) convert
v_mov_b32 v16, v14 // (op 491) convert

v_mul_lo_u32 v15, v13, s3 // (op 491) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 491) most significant: high of low * low
v_mul_lo_u32 v14, v13, s2 // (op 491) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 491) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v14 // (op 491) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 491) most significant half



s_mul_i32 s1, s2, s25 // (op 493) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 493) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 493) least significant
s_mul_hi_u32 s6, s2, s24 // (op 493) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 493) most significant: sum
s_add_u32 s1, s1, s6 // (op 493) most significant: sum


s_mov_b32 s52, s22 // (op 493) reg expression
s_mov_b32 s53, s23 // (op 493) reg expression
s_mov_b32 s54, s0 // (op 493) reg expression
s_mov_b32 s55, 131072 // (op 493) call()

s_mul_hi_i32 s51, s16, s9 // (op 536) 
v_lshrrev_b32 v14, 3, v1 // (op 536) 
s_add_i32 s0, s51, s16 // (op 536) 

v_and_b32 v15, -8, v1 // (op 536) 
s_ashr_i32 s1, s0, 31 // (op 536) 
v_sub_u32 v16, v1, v15 // (op 536) 

s_cmp_eq_i32 s9, 0 // (op 536) 
s_mov_b32 s51, scc // (op 536) 
v_lshlrev_b32 v15, 4, v16 // (op 536) 

s_cmp_eq_i32 s51, 0 // (op 536) 
s_cselect_b32 s6, 0, -1 // (op 536) 

v_mov_b32 v17, 0 // (op 536) convert
v_mov_b32 v16, v15 // (op 536) convert

s_and_b32 s7, s10, 31 // (op 536) 
s_lshl_b32 s51, 1, s7 // (op 536) 
s_add_i32 s18, s51, s6 // (op 536) 


s_and_b32 s6, s1, s18 // (op 536) 


s_add_i32 s1, s0, s6 // (op 536) 


s_ashr_i32 s0, s1, s7 // (op 536) 


s_lshl_b32 s1, s10, 24 // (op 536) 
s_ashr_i32 s51, s1, 31 // (op 536) 

s_xor_b32 s1, s0, s51 // (op 536) 

s_sub_i32 s0, s1, s51 // (op 536) 


s_mul_hi_i32 s1, s0, s11 // (op 536) 
s_add_i32 s51, s1, s0 // (op 536) 

s_ashr_i32 s1, s51, 31 // (op 536) 
s_cmp_eq_i32 s11, 0 // (op 536) 
s_mov_b32 s6, scc // (op 536) 
s_cmp_eq_i32 s6, 0 // (op 536) 
s_cselect_b32 s7, 0, -1 // (op 536) 

s_and_b32 s6, s12, 31 // (op 536) 
s_lshl_b32 s18, 1, s6 // (op 536) 
s_add_i32 s19, s18, s7 // (op 536) 


s_and_b32 s7, s1, s19 // (op 536) 


s_add_i32 s1, s51, s7 // (op 536) 


s_ashr_i32 s7, s1, s6 // (op 536) 


s_lshl_b32 s1, s12, 24 // (op 536) 
s_ashr_i32 s6, s1, 31 // (op 536) 

s_xor_b32 s1, s7, s6 // (op 536) 

s_sub_i32 s7, s1, s6 // (op 536) 


s_add_u32 s18, s4, 16 // (op 536) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 536) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 536) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 536) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 536) 

s_mul_i32 s1, s7, s18 // (op 536) 

s_sub_i32 s51, s0, s1 // (op 536) 


s_mul_hi_i32 s0, s16, s8 // (op 536) 
s_add_i32 s1, s0, s16 // (op 536) 

s_ashr_i32 s0, s1, 31 // (op 536) 
s_cmp_eq_i32 s8, 0 // (op 536) 
s_mov_b32 s6, scc // (op 536) 
s_cmp_eq_i32 s6, 0 // (op 536) 
s_cselect_b32 s7, 0, -1 // (op 536) 

s_and_b32 s6, s15, 31 // (op 536) 
s_lshl_b32 s56, 1, s6 // (op 536) 
s_add_i32 s57, s56, s7 // (op 536) 


s_and_b32 s7, s0, s57 // (op 536) 


s_add_i32 s0, s1, s7 // (op 536) 


s_ashr_i32 s1, s0, s6 // (op 536) 


s_lshl_b32 s0, s15, 24 // (op 536) 
s_ashr_i32 s6, s0, 31 // (op 536) 

s_xor_b32 s0, s1, s6 // (op 536) 

s_sub_i32 s1, s0, s6 // (op 536) 


s_mul_i32 s0, s14, s18 // (op 536) 

s_mul_i32 s6, s50, s0 // (op 536) 

s_mul_i32 s0, s1, s6 // (op 536) 

s_sub_u32 s6, s16, s0 // (op 536) 

s_mul_hi_i32 s0, s6, s13 // (op 536) 
s_add_i32 s7, s0, s6 // (op 536) 


s_ashr_i32 s0, s7, 31 // (op 536) 
s_cmp_eq_i32 s13, 0 // (op 536) 
s_mov_b32 s6, scc // (op 536) 
s_cmp_eq_i32 s6, 0 // (op 536) 
s_cselect_b32 s18, 0, -1 // (op 536) 

s_and_b32 s6, s49, 31 // (op 536) 
s_lshl_b32 s19, 1, s6 // (op 536) 
s_add_i32 s56, s19, s18 // (op 536) 


s_and_b32 s18, s0, s56 // (op 536) 


s_add_i32 s0, s7, s18 // (op 536) 


s_ashr_i32 s7, s0, s6 // (op 536) 


s_lshl_b32 s0, s49, 24 // (op 536) 
s_ashr_i32 s6, s0, 31 // (op 536) 

s_xor_b32 s0, s7, s6 // (op 536) 

s_sub_i32 s7, s0, s6 // (op 536) 


s_cmp_eq_i32 s1, 0 // (op 536) 
s_mov_b32 s0, scc // (op 536) 

s_cmp_eq_i32 s0, 0 // (op 536) 
s_cselect_b32 s1, s7, s51 // (op 536) 



v_lshl_add_u32 v15, s1, 4, v14 // (op 536) 


v_mul_lo_u32 v19, v15, s27 // (op 536) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 536) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 536) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 536) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 536) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 536) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 536) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 536) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 536) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 536) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 536) 

v_mov_b32 v16, v14 // (op 536) convert

s_mul_i32 s1, s5, 6 // (op 538) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 538) least significant
s_mul_hi_u32 s51, s4, 6 // (op 538) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 538) most significant: sum

s_mul_i32 s7, s26, s1 // (op 538) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 538) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 538) least significant
s_mul_hi_u32 s18, s26, s0 // (op 538) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 538) most significant: sum
s_add_u32 s7, s7, s18 // (op 538) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 538) 

s_mov_b32 s56, s28 // (op 538) reg expression
s_mov_b32 s57, s29 // (op 538) reg expression
s_mov_b32 s58, s0 // (op 538) reg expression
s_mov_b32 s59, 131072 // (op 538) call()

s_mul_hi_i32 s51, s16, s9 // (op 555) 
v_lshrrev_b32 v14, 3, v1 // (op 555) 
s_add_i32 s0, s51, s16 // (op 555) 

v_and_b32 v15, -8, v1 // (op 555) 
s_ashr_i32 s1, s0, 31 // (op 555) 
v_sub_u32 v17, v1, v15 // (op 555) 

s_cmp_eq_i32 s9, 0 // (op 555) 
s_mov_b32 s51, scc // (op 555) 
v_lshlrev_b32 v15, 4, v17 // (op 555) 

s_cmp_eq_i32 s51, 0 // (op 555) 
s_cselect_b32 s6, 0, -1 // (op 555) 

v_mov_b32 v19, 0 // (op 555) convert
v_mov_b32 v18, v15 // (op 555) convert

s_and_b32 s7, s10, 31 // (op 555) 
s_lshl_b32 s51, 1, s7 // (op 555) 
s_add_i32 s18, s51, s6 // (op 555) 


s_and_b32 s6, s1, s18 // (op 555) 


s_add_i32 s1, s0, s6 // (op 555) 


s_ashr_i32 s0, s1, s7 // (op 555) 


s_lshl_b32 s1, s10, 24 // (op 555) 
s_ashr_i32 s51, s1, 31 // (op 555) 

s_xor_b32 s1, s0, s51 // (op 555) 

s_sub_i32 s0, s1, s51 // (op 555) 


s_mul_hi_i32 s1, s0, s11 // (op 555) 
s_add_i32 s51, s1, s0 // (op 555) 

s_ashr_i32 s1, s51, 31 // (op 555) 
s_cmp_eq_i32 s11, 0 // (op 555) 
s_mov_b32 s6, scc // (op 555) 
s_cmp_eq_i32 s6, 0 // (op 555) 
s_cselect_b32 s7, 0, -1 // (op 555) 

s_and_b32 s6, s12, 31 // (op 555) 
s_lshl_b32 s18, 1, s6 // (op 555) 
s_add_i32 s19, s18, s7 // (op 555) 


s_and_b32 s7, s1, s19 // (op 555) 


s_add_i32 s1, s51, s7 // (op 555) 


s_ashr_i32 s7, s1, s6 // (op 555) 


s_lshl_b32 s1, s12, 24 // (op 555) 
s_ashr_i32 s6, s1, 31 // (op 555) 

s_xor_b32 s1, s7, s6 // (op 555) 

s_sub_i32 s7, s1, s6 // (op 555) 


s_add_u32 s18, s4, 16 // (op 555) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 555) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 555) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 555) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 555) 

s_mul_i32 s1, s7, s18 // (op 555) 

s_sub_i32 s51, s0, s1 // (op 555) 


s_mul_hi_i32 s0, s16, s8 // (op 555) 
s_add_i32 s1, s0, s16 // (op 555) 

s_ashr_i32 s0, s1, 31 // (op 555) 
s_cmp_eq_i32 s8, 0 // (op 555) 
s_mov_b32 s6, scc // (op 555) 
s_cmp_eq_i32 s6, 0 // (op 555) 
s_cselect_b32 s7, 0, -1 // (op 555) 

s_and_b32 s6, s15, 31 // (op 555) 
s_lshl_b32 s60, 1, s6 // (op 555) 
s_add_i32 s61, s60, s7 // (op 555) 


s_and_b32 s7, s0, s61 // (op 555) 


s_add_i32 s0, s1, s7 // (op 555) 


s_ashr_i32 s1, s0, s6 // (op 555) 


s_lshl_b32 s0, s15, 24 // (op 555) 
s_ashr_i32 s6, s0, 31 // (op 555) 

s_xor_b32 s0, s1, s6 // (op 555) 

s_sub_i32 s1, s0, s6 // (op 555) 


s_mul_i32 s0, s14, s18 // (op 555) 

s_mul_i32 s6, s50, s0 // (op 555) 

s_mul_i32 s0, s1, s6 // (op 555) 

s_sub_u32 s6, s16, s0 // (op 555) 

s_mul_hi_i32 s0, s6, s13 // (op 555) 
s_add_i32 s7, s0, s6 // (op 555) 


s_ashr_i32 s0, s7, 31 // (op 555) 
s_cmp_eq_i32 s13, 0 // (op 555) 
s_mov_b32 s6, scc // (op 555) 
s_cmp_eq_i32 s6, 0 // (op 555) 
s_cselect_b32 s18, 0, -1 // (op 555) 

s_and_b32 s6, s49, 31 // (op 555) 
s_lshl_b32 s19, 1, s6 // (op 555) 
s_add_i32 s60, s19, s18 // (op 555) 


s_and_b32 s18, s0, s60 // (op 555) 


s_add_i32 s0, s7, s18 // (op 555) 


s_ashr_i32 s7, s0, s6 // (op 555) 


s_lshl_b32 s0, s49, 24 // (op 555) 
s_ashr_i32 s6, s0, 31 // (op 555) 

s_xor_b32 s0, s7, s6 // (op 555) 

s_sub_i32 s7, s0, s6 // (op 555) 


s_cmp_eq_i32 s1, 0 // (op 555) 
s_mov_b32 s0, scc // (op 555) 

s_cmp_eq_i32 s0, 0 // (op 555) 
s_cselect_b32 s1, s7, s51 // (op 555) 



v_lshl_add_u32 v15, s1, 4, v14 // (op 555) 


v_mul_lo_u32 v21, v15, s27 // (op 555) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 555) most significant: high of low * low
v_mul_lo_u32 v20, v15, s26 // (op 555) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 555) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 555) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 555) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 555) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 555) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 555) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 555) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 555) 

v_mov_b32 v17, v14 // (op 555) convert

s_mul_i32 s1, s5, 6 // (op 557) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 557) least significant
s_mul_hi_u32 s51, s4, 6 // (op 557) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 557) most significant: sum

s_mul_i32 s7, s26, s1 // (op 557) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 557) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 557) least significant
s_mul_hi_u32 s18, s26, s0 // (op 557) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 557) most significant: sum
s_add_u32 s7, s7, s18 // (op 557) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 557) 

s_mov_b32 s56, s28 // (op 557) reg expression
s_mov_b32 s57, s29 // (op 557) reg expression
s_mov_b32 s58, s0 // (op 557) reg expression
s_mov_b32 s59, 131072 // (op 557) call()

s_mul_hi_i32 s51, s16, s9 // (op 600) 
v_and_b32 v14, -32, v1 // (op 600) 
s_add_i32 s0, s51, s16 // (op 600) 

v_sub_u32 v15, v1, v14 // (op 600) 

s_ashr_i32 s1, s0, 31 // (op 600) 
v_lshrrev_b32 v14, 5, v1 // (op 600) 
s_cmp_eq_i32 s9, 0 // (op 600) 
s_mov_b32 s51, scc // (op 600) 
s_cmp_eq_i32 s51, 0 // (op 600) 
s_cselect_b32 s6, 0, -1 // (op 600) 

s_and_b32 s7, s10, 31 // (op 600) 
s_lshl_b32 s51, 1, s7 // (op 600) 
s_add_i32 s18, s51, s6 // (op 600) 


s_and_b32 s6, s1, s18 // (op 600) 


s_add_i32 s1, s0, s6 // (op 600) 


s_ashr_i32 s0, s1, s7 // (op 600) 


s_lshl_b32 s1, s10, 24 // (op 600) 
s_ashr_i32 s51, s1, 31 // (op 600) 

s_xor_b32 s1, s0, s51 // (op 600) 

s_sub_i32 s0, s1, s51 // (op 600) 


s_mul_hi_i32 s1, s0, s11 // (op 600) 
s_add_i32 s51, s1, s0 // (op 600) 

s_ashr_i32 s1, s51, 31 // (op 600) 
s_cmp_eq_i32 s11, 0 // (op 600) 
s_mov_b32 s6, scc // (op 600) 
s_cmp_eq_i32 s6, 0 // (op 600) 
s_cselect_b32 s7, 0, -1 // (op 600) 

s_and_b32 s6, s12, 31 // (op 600) 
s_lshl_b32 s18, 1, s6 // (op 600) 
s_add_i32 s19, s18, s7 // (op 600) 


s_and_b32 s7, s1, s19 // (op 600) 


s_add_i32 s1, s51, s7 // (op 600) 


s_ashr_i32 s7, s1, s6 // (op 600) 


s_lshl_b32 s1, s12, 24 // (op 600) 
s_ashr_i32 s6, s1, 31 // (op 600) 

s_xor_b32 s1, s7, s6 // (op 600) 

s_sub_i32 s7, s1, s6 // (op 600) 


s_mul_i32 s1, s0, s14 // (op 600) 

s_sub_u32 s0, s16, s1 // (op 600) 

s_mul_i32 s1, s7, s14 // (op 600) 
s_add_u32 s1, s1, s0 // (op 600) 


s_mul_hi_i32 s0, s16, s8 // (op 600) 
s_add_i32 s51, s0, s16 // (op 600) 

s_ashr_i32 s0, s51, 31 // (op 600) 
s_cmp_eq_i32 s8, 0 // (op 600) 
s_mov_b32 s6, scc // (op 600) 
s_cmp_eq_i32 s6, 0 // (op 600) 
s_cselect_b32 s7, 0, -1 // (op 600) 

s_and_b32 s6, s15, 31 // (op 600) 
s_lshl_b32 s18, 1, s6 // (op 600) 
s_add_i32 s19, s18, s7 // (op 600) 


s_and_b32 s7, s0, s19 // (op 600) 


s_add_i32 s0, s51, s7 // (op 600) 


s_ashr_i32 s7, s0, s6 // (op 600) 


s_lshl_b32 s0, s15, 24 // (op 600) 
s_ashr_i32 s6, s0, 31 // (op 600) 

s_xor_b32 s0, s7, s6 // (op 600) 

s_sub_i32 s7, s0, s6 // (op 600) 


s_add_u32 s18, s4, 16 // (op 600) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 600) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 600) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 600) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 600) 

s_mul_i32 s0, s14, s18 // (op 600) 

s_mul_i32 s6, s50, s0 // (op 600) 

s_mul_i32 s0, s7, s6 // (op 600) 

s_sub_u32 s6, s16, s0 // (op 600) 

s_mov_b32 s0, s6 // (op 600) 
s_mul_hi_i32 s51, s0, s13 // (op 600) 
s_add_i32 s18, s51, s0 // (op 600) 


s_ashr_i32 s0, s18, 31 // (op 600) 
s_cmp_eq_i32 s13, 0 // (op 600) 
s_mov_b32 s19, scc // (op 600) 
s_cmp_eq_i32 s19, 0 // (op 600) 
s_cselect_b32 s51, 0, -1 // (op 600) 

s_and_b32 s19, s49, 31 // (op 600) 
s_lshl_b32 s60, 1, s19 // (op 600) 
s_add_i32 s61, s60, s51 // (op 600) 


s_and_b32 s51, s0, s61 // (op 600) 


s_add_i32 s0, s18, s51 // (op 600) 


s_ashr_i32 s18, s0, s19 // (op 600) 


s_lshl_b32 s0, s49, 24 // (op 600) 
s_ashr_i32 s19, s0, 31 // (op 600) 

s_xor_b32 s0, s18, s19 // (op 600) 

s_sub_i32 s18, s0, s19 // (op 600) 


s_mul_i32 s0, s48, s14 // (op 600) 
s_add_u32 s60, s20, 32 // (op 600) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 600) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 600) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 600) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 5 // (op 600) 

s_sub_i32 s19, s60, s0 // (op 600) 

s_mul_i32 s51, s18, s19 // (op 600) 


s_sub_u32 s18, s6, s51 // (op 600) 


s_add_u32 s6, s18, s0 // (op 600) 


s_cmp_eq_i32 s7, 0 // (op 600) 
s_mov_b32 s0, scc // (op 600) 

s_cmp_eq_i32 s0, 0 // (op 600) 
s_cselect_b32 s7, s6, s1 // (op 600) 



v_lshl_add_u32 v18, s7, 5, v15 // (op 600) 


v_mul_lo_u32 v15, v14, s30 // (op 600) 

v_add_u32 v14, v18, v15 // (op 600) 


s_mul_i32 s1, s30, s35 // (op 602) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 602) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 602) least significant
s_mul_hi_u32 s6, s30, s34 // (op 602) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 602) most significant: sum
s_add_u32 s1, s1, s6 // (op 602) most significant: sum


s_mov_b32 s60, s32 // (op 602) reg expression
s_mov_b32 s61, s33 // (op 602) reg expression
s_mov_b32 s62, s0 // (op 602) reg expression
s_mov_b32 s63, 131072 // (op 602) call()

s_mul_hi_i32 s51, s16, s9 // (op 619) 
v_and_b32 v15, -32, v1 // (op 619) 
s_add_i32 s0, s51, s16 // (op 619) 

v_sub_u32 v18, v1, v15 // (op 619) 

s_ashr_i32 s1, s0, 31 // (op 619) 
v_lshrrev_b32 v15, 5, v1 // (op 619) 
s_cmp_eq_i32 s9, 0 // (op 619) 
s_mov_b32 s51, scc // (op 619) 
s_cmp_eq_i32 s51, 0 // (op 619) 
s_cselect_b32 s6, 0, -1 // (op 619) 

s_and_b32 s7, s10, 31 // (op 619) 
s_lshl_b32 s51, 1, s7 // (op 619) 
s_add_i32 s18, s51, s6 // (op 619) 


s_and_b32 s6, s1, s18 // (op 619) 


s_add_i32 s1, s0, s6 // (op 619) 


s_ashr_i32 s0, s1, s7 // (op 619) 


s_lshl_b32 s1, s10, 24 // (op 619) 
s_ashr_i32 s51, s1, 31 // (op 619) 

s_xor_b32 s1, s0, s51 // (op 619) 

s_sub_i32 s0, s1, s51 // (op 619) 


s_mul_hi_i32 s1, s0, s11 // (op 619) 
s_add_i32 s51, s1, s0 // (op 619) 

s_ashr_i32 s1, s51, 31 // (op 619) 
s_cmp_eq_i32 s11, 0 // (op 619) 
s_mov_b32 s6, scc // (op 619) 
s_cmp_eq_i32 s6, 0 // (op 619) 
s_cselect_b32 s7, 0, -1 // (op 619) 

s_and_b32 s6, s12, 31 // (op 619) 
s_lshl_b32 s18, 1, s6 // (op 619) 
s_add_i32 s19, s18, s7 // (op 619) 


s_and_b32 s7, s1, s19 // (op 619) 


s_add_i32 s1, s51, s7 // (op 619) 


s_ashr_i32 s7, s1, s6 // (op 619) 


s_lshl_b32 s1, s12, 24 // (op 619) 
s_ashr_i32 s6, s1, 31 // (op 619) 

s_xor_b32 s1, s7, s6 // (op 619) 

s_sub_i32 s7, s1, s6 // (op 619) 


s_mul_i32 s1, s0, s14 // (op 619) 

s_sub_u32 s0, s16, s1 // (op 619) 

s_mul_i32 s1, s7, s14 // (op 619) 
s_add_u32 s1, s1, s0 // (op 619) 


s_mul_hi_i32 s0, s16, s8 // (op 619) 
s_add_i32 s51, s0, s16 // (op 619) 

s_ashr_i32 s0, s51, 31 // (op 619) 
s_cmp_eq_i32 s8, 0 // (op 619) 
s_mov_b32 s6, scc // (op 619) 
s_cmp_eq_i32 s6, 0 // (op 619) 
s_cselect_b32 s7, 0, -1 // (op 619) 

s_and_b32 s6, s15, 31 // (op 619) 
s_lshl_b32 s18, 1, s6 // (op 619) 
s_add_i32 s19, s18, s7 // (op 619) 


s_and_b32 s7, s0, s19 // (op 619) 


s_add_i32 s0, s51, s7 // (op 619) 


s_ashr_i32 s7, s0, s6 // (op 619) 


s_lshl_b32 s0, s15, 24 // (op 619) 
s_ashr_i32 s6, s0, 31 // (op 619) 

s_xor_b32 s0, s7, s6 // (op 619) 

s_sub_i32 s7, s0, s6 // (op 619) 


s_add_u32 s18, s4, 16 // (op 619) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 619) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 619) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 619) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 619) 

s_mul_i32 s0, s14, s18 // (op 619) 

s_mul_i32 s6, s50, s0 // (op 619) 

s_mul_i32 s0, s7, s6 // (op 619) 

s_sub_u32 s6, s16, s0 // (op 619) 

s_mov_b32 s0, s6 // (op 619) 
s_mul_hi_i32 s51, s0, s13 // (op 619) 
s_add_i32 s18, s51, s0 // (op 619) 


s_ashr_i32 s0, s18, 31 // (op 619) 
s_cmp_eq_i32 s13, 0 // (op 619) 
s_mov_b32 s19, scc // (op 619) 
s_cmp_eq_i32 s19, 0 // (op 619) 
s_cselect_b32 s51, 0, -1 // (op 619) 

s_and_b32 s19, s49, 31 // (op 619) 
s_lshl_b32 s64, 1, s19 // (op 619) 
s_add_i32 s65, s64, s51 // (op 619) 


s_and_b32 s51, s0, s65 // (op 619) 


s_add_i32 s0, s18, s51 // (op 619) 


s_ashr_i32 s18, s0, s19 // (op 619) 


s_lshl_b32 s0, s49, 24 // (op 619) 
s_ashr_i32 s19, s0, 31 // (op 619) 

s_xor_b32 s0, s18, s19 // (op 619) 

s_sub_i32 s18, s0, s19 // (op 619) 


s_mul_i32 s0, s48, s14 // (op 619) 
s_add_u32 s64, s20, 32 // (op 619) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 619) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 619) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 619) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 5 // (op 619) 

s_sub_i32 s19, s64, s0 // (op 619) 

s_mul_i32 s51, s18, s19 // (op 619) 


s_sub_u32 s18, s6, s51 // (op 619) 


s_add_u32 s6, s18, s0 // (op 619) 


s_cmp_eq_i32 s7, 0 // (op 619) 
s_mov_b32 s0, scc // (op 619) 

s_cmp_eq_i32 s0, 0 // (op 619) 
s_cselect_b32 s7, s6, s1 // (op 619) 



v_lshl_add_u32 v19, s7, 5, v18 // (op 619) 


v_mul_lo_u32 v18, v15, s30 // (op 619) 

v_add_u32 v15, v19, v18 // (op 619) 


s_mul_i32 s1, s30, s35 // (op 621) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 621) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 621) least significant
s_mul_hi_u32 s6, s30, s34 // (op 621) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 621) most significant: sum
s_add_u32 s1, s1, s6 // (op 621) most significant: sum


s_mov_b32 s60, s32 // (op 621) reg expression
s_mov_b32 s61, s33 // (op 621) reg expression
s_mov_b32 s62, s0 // (op 621) reg expression
s_mov_b32 s63, 131072 // (op 621) call()

s_mul_hi_i32 s51, s16, s9 // (op 664) 
v_and_b32 v18, 15, v1 // (op 664) 
s_add_i32 s0, s51, s16 // (op 664) 

v_and_b32 v19, 63, v1 // (op 664) 
s_ashr_i32 s1, s0, 31 // (op 664) 
v_lshrrev_b32 v20, 4, v19 // (op 664) 

s_cmp_eq_i32 s9, 0 // (op 664) 
s_mov_b32 s51, scc // (op 664) 
s_cmp_eq_i32 s51, 0 // (op 664) 
s_cselect_b32 s6, 0, -1 // (op 664) 

s_and_b32 s7, s10, 31 // (op 664) 
s_lshl_b32 s51, 1, s7 // (op 664) 
s_add_i32 s18, s51, s6 // (op 664) 


s_and_b32 s6, s1, s18 // (op 664) 


s_add_i32 s1, s0, s6 // (op 664) 


s_ashr_i32 s0, s1, s7 // (op 664) 


s_lshl_b32 s1, s10, 24 // (op 664) 
s_ashr_i32 s51, s1, 31 // (op 664) 

s_xor_b32 s1, s0, s51 // (op 664) 

s_sub_i32 s0, s1, s51 // (op 664) 


s_mul_hi_i32 s1, s0, s11 // (op 664) 
s_add_i32 s51, s1, s0 // (op 664) 

s_ashr_i32 s1, s51, 31 // (op 664) 
s_cmp_eq_i32 s11, 0 // (op 664) 
s_mov_b32 s6, scc // (op 664) 
s_cmp_eq_i32 s6, 0 // (op 664) 
s_cselect_b32 s7, 0, -1 // (op 664) 

s_and_b32 s6, s12, 31 // (op 664) 
s_lshl_b32 s18, 1, s6 // (op 664) 
s_add_i32 s19, s18, s7 // (op 664) 


s_and_b32 s7, s1, s19 // (op 664) 


s_add_i32 s1, s51, s7 // (op 664) 


s_ashr_i32 s7, s1, s6 // (op 664) 


s_lshl_b32 s1, s12, 24 // (op 664) 
s_ashr_i32 s6, s1, 31 // (op 664) 

s_xor_b32 s1, s7, s6 // (op 664) 

s_sub_i32 s7, s1, s6 // (op 664) 


s_add_u32 s18, s4, 16 // (op 664) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 664) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 664) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 664) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 664) 

s_mul_i32 s1, s7, s18 // (op 664) 

s_sub_i32 s51, s0, s1 // (op 664) 


s_mul_hi_i32 s0, s16, s8 // (op 664) 
s_add_i32 s1, s0, s16 // (op 664) 

s_ashr_i32 s0, s1, 31 // (op 664) 
s_cmp_eq_i32 s8, 0 // (op 664) 
s_mov_b32 s6, scc // (op 664) 
s_cmp_eq_i32 s6, 0 // (op 664) 
s_cselect_b32 s7, 0, -1 // (op 664) 

s_and_b32 s6, s15, 31 // (op 664) 
s_lshl_b32 s64, 1, s6 // (op 664) 
s_add_i32 s65, s64, s7 // (op 664) 


s_and_b32 s7, s0, s65 // (op 664) 


s_add_i32 s0, s1, s7 // (op 664) 


s_ashr_i32 s1, s0, s6 // (op 664) 


s_lshl_b32 s0, s15, 24 // (op 664) 
s_ashr_i32 s6, s0, 31 // (op 664) 

s_xor_b32 s0, s1, s6 // (op 664) 

s_sub_i32 s1, s0, s6 // (op 664) 


s_mul_i32 s0, s14, s18 // (op 664) 

s_mul_i32 s6, s50, s0 // (op 664) 

s_mul_i32 s0, s1, s6 // (op 664) 

s_sub_u32 s6, s16, s0 // (op 664) 

s_mul_hi_i32 s0, s6, s13 // (op 664) 
s_add_i32 s7, s0, s6 // (op 664) 


s_ashr_i32 s0, s7, 31 // (op 664) 
s_cmp_eq_i32 s13, 0 // (op 664) 
s_mov_b32 s6, scc // (op 664) 
s_cmp_eq_i32 s6, 0 // (op 664) 
s_cselect_b32 s18, 0, -1 // (op 664) 

s_and_b32 s6, s49, 31 // (op 664) 
s_lshl_b32 s19, 1, s6 // (op 664) 
s_add_i32 s64, s19, s18 // (op 664) 


s_and_b32 s18, s0, s64 // (op 664) 


s_add_i32 s0, s7, s18 // (op 664) 


s_ashr_i32 s7, s0, s6 // (op 664) 


s_lshl_b32 s0, s49, 24 // (op 664) 
s_ashr_i32 s6, s0, 31 // (op 664) 

s_xor_b32 s0, s7, s6 // (op 664) 

s_sub_i32 s7, s0, s6 // (op 664) 


s_cmp_eq_i32 s1, 0 // (op 664) 
s_mov_b32 s0, scc // (op 664) 

s_cmp_eq_i32 s0, 0 // (op 664) 
s_cselect_b32 s1, s7, s51 // (op 664) 



v_lshl_add_u32 v19, s1, 4, v18 // (op 664) 


v_mul_lo_u32 v18, v19, s36 // (op 664) 

v_add_u32 v19, v20, v18 // (op 664) 


s_mul_i32 s1, s36, s5 // (op 666) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 666) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 666) least significant
s_mul_hi_u32 s6, s36, s4 // (op 666) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 666) most significant: sum
s_add_u32 s1, s1, s6 // (op 666) most significant: sum


s_mov_b32 s64, s38 // (op 666) reg expression
s_mov_b32 s65, s39 // (op 666) reg expression
s_mov_b32 s66, s0 // (op 666) reg expression
s_mov_b32 s67, 131072 // (op 666) call()

s_mul_hi_i32 s51, s16, s9 // (op 683) 
v_and_b32 v18, 15, v1 // (op 683) 
s_add_i32 s0, s51, s16 // (op 683) 

v_and_b32 v20, 63, v1 // (op 683) 
s_ashr_i32 s1, s0, 31 // (op 683) 
v_lshrrev_b32 v21, 4, v20 // (op 683) 

s_cmp_eq_i32 s9, 0 // (op 683) 
s_mov_b32 s51, scc // (op 683) 
s_cmp_eq_i32 s51, 0 // (op 683) 
s_cselect_b32 s6, 0, -1 // (op 683) 

s_and_b32 s7, s10, 31 // (op 683) 
s_lshl_b32 s51, 1, s7 // (op 683) 
s_add_i32 s18, s51, s6 // (op 683) 


s_and_b32 s6, s1, s18 // (op 683) 


s_add_i32 s1, s0, s6 // (op 683) 


s_ashr_i32 s0, s1, s7 // (op 683) 


s_lshl_b32 s1, s10, 24 // (op 683) 
s_ashr_i32 s51, s1, 31 // (op 683) 

s_xor_b32 s1, s0, s51 // (op 683) 

s_sub_i32 s0, s1, s51 // (op 683) 


s_mul_hi_i32 s1, s0, s11 // (op 683) 
s_add_i32 s51, s1, s0 // (op 683) 

s_ashr_i32 s1, s51, 31 // (op 683) 
s_cmp_eq_i32 s11, 0 // (op 683) 
s_mov_b32 s6, scc // (op 683) 
s_cmp_eq_i32 s6, 0 // (op 683) 
s_cselect_b32 s7, 0, -1 // (op 683) 

s_and_b32 s6, s12, 31 // (op 683) 
s_lshl_b32 s18, 1, s6 // (op 683) 
s_add_i32 s19, s18, s7 // (op 683) 


s_and_b32 s7, s1, s19 // (op 683) 


s_add_i32 s1, s51, s7 // (op 683) 


s_ashr_i32 s7, s1, s6 // (op 683) 


s_lshl_b32 s1, s12, 24 // (op 683) 
s_ashr_i32 s6, s1, 31 // (op 683) 

s_xor_b32 s1, s7, s6 // (op 683) 

s_sub_i32 s7, s1, s6 // (op 683) 


s_add_u32 s18, s4, 16 // (op 683) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 683) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 683) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 683) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 683) 

s_mul_i32 s1, s7, s18 // (op 683) 

s_sub_i32 s51, s0, s1 // (op 683) 


s_mul_hi_i32 s0, s16, s8 // (op 683) 
s_add_i32 s1, s0, s16 // (op 683) 

s_ashr_i32 s0, s1, 31 // (op 683) 
s_cmp_eq_i32 s8, 0 // (op 683) 
s_mov_b32 s6, scc // (op 683) 
s_cmp_eq_i32 s6, 0 // (op 683) 
s_cselect_b32 s7, 0, -1 // (op 683) 

s_and_b32 s6, s15, 31 // (op 683) 
s_lshl_b32 s68, 1, s6 // (op 683) 
s_add_i32 s69, s68, s7 // (op 683) 


s_and_b32 s7, s0, s69 // (op 683) 


s_add_i32 s0, s1, s7 // (op 683) 


s_ashr_i32 s1, s0, s6 // (op 683) 


s_lshl_b32 s0, s15, 24 // (op 683) 
s_ashr_i32 s6, s0, 31 // (op 683) 

s_xor_b32 s0, s1, s6 // (op 683) 

s_sub_i32 s1, s0, s6 // (op 683) 


s_mul_i32 s0, s14, s18 // (op 683) 

s_mul_i32 s6, s50, s0 // (op 683) 

s_mul_i32 s0, s1, s6 // (op 683) 

s_sub_u32 s6, s16, s0 // (op 683) 

s_mul_hi_i32 s0, s6, s13 // (op 683) 
s_add_i32 s7, s0, s6 // (op 683) 


s_ashr_i32 s0, s7, 31 // (op 683) 
s_cmp_eq_i32 s13, 0 // (op 683) 
s_mov_b32 s6, scc // (op 683) 
s_cmp_eq_i32 s6, 0 // (op 683) 
s_cselect_b32 s18, 0, -1 // (op 683) 

s_and_b32 s6, s49, 31 // (op 683) 
s_lshl_b32 s19, 1, s6 // (op 683) 
s_add_i32 s68, s19, s18 // (op 683) 


s_and_b32 s18, s0, s68 // (op 683) 


s_add_i32 s0, s7, s18 // (op 683) 


s_ashr_i32 s7, s0, s6 // (op 683) 


s_lshl_b32 s0, s49, 24 // (op 683) 
s_ashr_i32 s6, s0, 31 // (op 683) 

s_xor_b32 s0, s7, s6 // (op 683) 

s_sub_i32 s7, s0, s6 // (op 683) 


s_cmp_eq_i32 s1, 0 // (op 683) 
s_mov_b32 s0, scc // (op 683) 

s_cmp_eq_i32 s0, 0 // (op 683) 
s_cselect_b32 s1, s7, s51 // (op 683) 



v_lshl_add_u32 v20, s1, 4, v18 // (op 683) 


v_mul_lo_u32 v18, v20, s36 // (op 683) 

v_add_u32 v20, v21, v18 // (op 683) 


s_mul_i32 s1, s36, s5 // (op 685) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 685) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 685) least significant
s_mul_hi_u32 s6, s36, s4 // (op 685) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 685) most significant: sum
s_add_u32 s1, s1, s6 // (op 685) most significant: sum


s_mov_b32 s64, s38 // (op 685) reg expression
s_mov_b32 s65, s39 // (op 685) reg expression
s_mov_b32 s66, s0 // (op 685) reg expression
s_mov_b32 s67, 131072 // (op 685) call()




buffer_load_ubyte v18, v19, s[64:67], 0 offen offset: 0 // (op 337) Load value



buffer_load_dwordx3 v[22:24], v16, s[56:59], 0 offen // (op 341) Load value
v_and_b32 v21, -2, v1 // (op 749) 
v_sub_u32 v25, v1, v21 // (op 749) 

v_lshlrev_b32 v21, 4, v25 // (op 749) 

v_lshrrev_b32 v25, 1, v1 // (op 749) 
v_lshl_add_u32 v26, v25, 5, v21 // (op 749) 


s_nop 0
v_readfirstlane_b32 s51, v26 // (op 749) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[2:3], 6 // (op 428) 
v_mov_b64 v[26:27], s[0:1] // (op 428) convert


v_mov_b64 v[28:29], v[10:11] // (op 428) reg expression
s_mov_b32 m0, s51 // (op 428) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 428) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 428) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 428) most significant half

s_add_u32 m0, m0, 2048 // (op 428) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 428) Load value direct to lds





v_mov_b32 v21, v14 // (op 347) reg expression

buffer_load_ubyte v25, v21, s[60:63], 0 offen offset: 0 // (op 347) Load value




buffer_load_ubyte v21, v19, s[64:67], 0 offen offset:4 // (op 351) Load value




buffer_load_dwordx3 v[26:28], v16, s[56:59], 0 offen offset:96 // (op 355) Load value

v_and_b32 v16, -2, v1 // (op 928) 
v_sub_u32 v19, v1, v16 // (op 928) 

v_lshlrev_b32 v16, 4, v19 // (op 928) 

v_lshrrev_b32 v19, 1, v1 // (op 928) 
v_lshl_add_u32 v29, v19, 5, v16 // (op 928) 


s_nop 0
v_readfirstlane_b32 s51, v29 // (op 928) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[2:3], 6 // (op 433) 
v_mov_b64 v[30:31], s[0:1] // (op 433) convert


s_lshl_b64 s[0:1], s[2:3], 7 // (op 433) 
v_mov_b32 v16, s0 // (op 433) 
v_mov_b32 v19, s1 // (op 433) 
v_add_co_u32 v32, s[6:7], v10, v16 // (op 433) least significant half
v_addc_co_u32 v33, s[6:7], v11, v19, s[6:7] // (op 433) most significant half




s_add_u32 s0, 4096, s51 // (op 433) 
s_mov_b32 m0, s0 // (op 433) reg expression
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 433) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v30 // (op 433) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 433) most significant half

s_add_u32 m0, m0, 2048 // (op 433) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 433) Load value direct to lds







s_lshl_b32 s51, s30, 2 // (op 361) 
v_add_u32 v16, v14, s51 // (op 361) 


buffer_load_ubyte v19, v16, s[60:63], 0 offen offset: 0 // (op 361) Load value


v_and_b32 v14, -8, v1 // (op 802) 
v_sub_u32 v16, v1, v14 // (op 802) 

v_lshlrev_b32 v14, 4, v16 // (op 802) 

v_lshrrev_b32 v16, 3, v1 // (op 802) 
v_lshl_add_u32 v10, v16, 7, v14 // (op 802) 


v_mul_lo_u32 v11, v10, 6 // (op 802) 

v_lshrrev_b32 v10, 3, v11 // (op 802) 




s_waitcnt vmcnt(8)
ds_write_b96 v10, v[22:24] offset:8192 // (op 363) Store local data 

v_and_b32 v14, -32, v1 // (op 855) 
v_sub_u32 v16, v1, v14 // (op 855) 

v_and_b32 v14, 4294967264, v1 // (op 855) 
v_add_u32 v10, v14, v16 // (op 855) 





s_waitcnt vmcnt(5)
ds_write_b8 v10, v25 offset:9728 // (op 365) Store local data 

v_and_b32 v14, -2, v1 // (op 721) 
v_sub_u32 v16, v1, v14 // (op 721) 

v_lshlrev_b32 v14, 4, v16 // (op 721) 

v_lshrrev_b32 v16, 1, v1 // (op 721) 
v_lshl_add_u32 v10, v16, 5, v14 // (op 721) 


s_nop 0
v_readfirstlane_b32 s51, v10 // (op 721) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 773) 
v_sub_u32 v16, v1, v14 // (op 773) 

v_lshlrev_b32 v14, 4, v16 // (op 773) 

v_lshrrev_b32 v16, 3, v1 // (op 773) 
v_lshl_add_u32 v10, v16, 7, v14 // (op 773) 


v_mul_lo_u32 v11, v10, 6 // (op 773) 

v_lshrrev_b32 v10, 3, v11 // (op 773) 

v_and_b32 v11, -32, v1 // (op 826) 
v_sub_u32 v14, v1, v11 // (op 826) 

v_and_b32 v11, 4294967264, v1 // (op 826) 
v_add_u32 v16, v11, v14 // (op 826) 


v_and_b32 v11, -2, v1 // (op 913) 
v_sub_u32 v14, v1, v11 // (op 913) 

v_lshlrev_b32 v11, 4, v14 // (op 913) 

v_lshrrev_b32 v14, 1, v1 // (op 913) 
v_lshl_add_u32 v29, v14, 5, v11 // (op 913) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 913) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v11, -8, v1 // (op 952) 
v_sub_u32 v14, v1, v11 // (op 952) 

v_lshlrev_b32 v11, 4, v14 // (op 952) 

v_lshrrev_b32 v14, 3, v1 // (op 952) 
v_lshl_add_u32 v29, v14, 7, v11 // (op 952) 


v_mul_lo_u32 v11, v29, 6 // (op 952) 

v_lshrrev_b32 v14, 3, v11 // (op 952) 

v_and_b32 v11, -32, v1 // (op 976) 
v_sub_u32 v29, v1, v11 // (op 976) 

v_and_b32 v11, 4294967264, v1 // (op 976) 
v_add_u32 v30, v11, v29 // (op 976) 


s_mov_b32 s1, 0 // (op 55) call()
s_ashr_i64 s[6:7], s[24:25], 7 // (op 54) 
s_mov_b32 s18, s6 // (op 54) convert

s_and_b32 s19, s18, 4294967294 // (op 54) 

s_cmp_lt_u32 s1, s19 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_54 if false)
RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_54:
s_barrier  // (op 1021) 
s_add_u32 s1, s1, 2 // (op 56) 



ds_read_b64_tr_b8 v[32:33], v0 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:256 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:2048 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:2304 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1034) 



ds_read_b128 v[40:43], v4 offset:8192 // (op 70) Load local data 
ds_read_b64 v[44:45], v4 offset:8208 // (op 70) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1048) 


ds_read_u8 v11, v3 offset:9728 // (op 75) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:45], a[0:3], v11, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 82) 





v_mov_b32 v11, v20 // (op 80) 

buffer_load_ubyte v18, v11, s[64:67], 0 offen offset:8 // (op 80) Load value


v_mov_b32 v29, v17 // (op 10) 

buffer_load_dwordx3 v[22:24], v29, s[56:59], 0 offen offset:192 // (op 10) Load value
s_barrier  // (op 1024) 

s_lshl_b64 s[6:7], s[2:3], 6 // (op 412) 
v_mov_b64 v[32:33], s[6:7] // (op 412) convert


v_mov_b64 v[34:35], v[12:13] // (op 412) 
s_lshl_b64 s[6:7], s[2:3], 7 // (op 412) 
s_lshl_b64 s[18:19], s[6:7], 1 // (op 412) 

v_mov_b32 v31, s18 // (op 412) 
v_mov_b32 v36, s19 // (op 412) 
v_add_co_u32 v38, s[6:7], v34, v31 // (op 412) least significant half
v_addc_co_u32 v39, s[6:7], v35, v36, s[6:7] // (op 412) most significant half




s_mov_b32 m0, s51 // (op 412) reg expression
buffer_load_dwordx4 v38, s[52:55], 0 offen offset: 0 lds // (op 412) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v32 // (op 412) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 412) most significant half

s_add_u32 m0, m0, 2048 // (op 412) 
buffer_load_dwordx4 v38, s[52:55], 0 offen offset: 0 lds // (op 412) Load value direct to lds




v_mov_b32 v31, v15 // (op 17) 
s_lshl_b32 s6, s30, 3 // (op 17) 
v_add_u32 v32, v31, s6 // (op 17) 


buffer_load_ubyte v25, v32, s[60:63], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1071) 



s_waitcnt vmcnt(8)
ds_write_b96 v14, v[26:28] offset:9856 // (op 235) Store local data 
s_barrier  // (op 1077) 



s_waitcnt vmcnt(5)
ds_write_b8 v30, v19 offset:11392 // (op 236) Store local data 
s_barrier  // (op 1062) 



ds_read_b64_tr_b8 v[36:37], v5 offset:4096 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v5 offset:4352 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v5 offset:6144 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v5 offset:6400 // (op 227) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1068) 



ds_read_b128 v[44:47], v7 offset:9856 // (op 229) Load local data 
ds_read_b64 v[48:49], v7 offset:9872 // (op 229) Load local data 
s_barrier  // (op 1124) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1074) 


ds_read_u8 v32, v6 offset:11392 // (op 231) Load local data 
s_barrier  // (op 1130) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:49], a[0:3], v32, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 234) 






buffer_load_ubyte v21, v11, s[64:67], 0 offen offset:12 // (op 233) Load value




v_add_u32 v20, 8, v20 // (op 675) 
buffer_load_dwordx3 v[26:28], v29, s[56:59], 0 offen offset:288 // (op 237) Load value

s_barrier  // (op 1065) 
v_add_u32 v17, 192, v17 // (op 547) 

s_lshl_b64 s[6:7], s[2:3], 6 // (op 423) 
v_mov_b64 v[32:33], s[6:7] // (op 423) convert


s_lshl_b64 s[6:7], s[2:3], 7 // (op 423) 
s_mul_i32 s19, 3, s7 // (op 423) most significant: low of low * high
s_mul_i32 s18, 3, s6 // (op 423) least significant
s_mul_hi_u32 s68, 3, s6 // (op 423) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 423) most significant: sum


v_mov_b32 v11, s18 // (op 423) 
v_mov_b32 v29, s19 // (op 423) 
v_add_co_u32 v36, s[6:7], v34, v11 // (op 423) least significant half
v_addc_co_u32 v37, s[6:7], v35, v29, s[6:7] // (op 423) most significant half




s_add_u32 s6, 4096, s0 // (op 423) 
s_nop 0
s_mov_b32 m0, s6 // (op 423) reg expression
buffer_load_dwordx4 v36, s[52:55], 0 offen offset: 0 lds // (op 423) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 423) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 423) most significant half

s_add_u32 m0, m0, 2048 // (op 423) 
buffer_load_dwordx4 v36, s[52:55], 0 offen offset: 0 lds // (op 423) Load value direct to lds






s_lshl_b32 s6, s30, 2 // (op 238) 
s_mul_i32 s7, 3, s6 // (op 238) 

v_add_u32 v11, v31, s7 // (op 238) 


s_lshl_b64 s[6:7], s[2:3], 7 // (op 483) 
s_lshl_b64 s[18:19], s[6:7], 1 // (op 483) 

v_mov_b32 v29, s18 // (op 483) 
v_mov_b32 v32, s19 // (op 483) 
v_add_co_u32 v12, s[6:7], v12, v29 // (op 483) least significant half
v_addc_co_u32 v13, s[6:7], v13, v32, s[6:7] // (op 483) most significant half




buffer_load_ubyte v19, v11, s[60:63], 0 offen offset: 0 // (op 238) Load value


s_barrier  // (op 1037) 
s_lshl_b32 s6, s30, 3 // (op 611) 



v_add_u32 v15, v15, s6 // (op 611) 

s_waitcnt vmcnt(8)
ds_write_b96 v10, v[22:24] offset:8192 // (op 13) Store local data 
s_barrier  // (op 1051) 



s_waitcnt vmcnt(5)
ds_write_b8 v16, v25 offset:9728 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[24:25], 7 // (op 54) 
s_mov_b32 s18, s6 // (op 54) convert

s_and_b32 s19, s18, 4294967294 // (op 54) 

s_cmp_lt_u32 s1, s19 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_54 if true)
RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_54:














s_barrier  // (op 1088) 
s_mul_hi_i32 s51, s16, s9 // (op 446) 
v_and_b32 v20, -2, v1 // (op 446) 
s_add_i32 s0, s51, s16 // (op 446) 

v_sub_u32 v10, v1, v20 // (op 446) 

s_ashr_i32 s1, s0, 31 // (op 446) 
v_lshlrev_b32 v20, 4, v10 // (op 446) 

s_cmp_eq_i32 s9, 0 // (op 446) 
s_mov_b32 s51, scc // (op 446) 
v_lshrrev_b32 v10, 1, v1 // (op 446) 
s_cmp_eq_i32 s51, 0 // (op 446) 
s_cselect_b32 s6, 0, -1 // (op 446) 

s_and_b32 s7, s10, 31 // (op 446) 
s_lshl_b32 s51, 1, s7 // (op 446) 
s_add_i32 s18, s51, s6 // (op 446) 


s_and_b32 s6, s1, s18 // (op 446) 


s_add_i32 s1, s0, s6 // (op 446) 


s_ashr_i32 s0, s1, s7 // (op 446) 


s_lshl_b32 s1, s10, 24 // (op 446) 
s_ashr_i32 s51, s1, 31 // (op 446) 

s_xor_b32 s1, s0, s51 // (op 446) 

s_sub_i32 s0, s1, s51 // (op 446) 


s_mul_hi_i32 s1, s0, s11 // (op 446) 
s_add_i32 s51, s1, s0 // (op 446) 

s_ashr_i32 s1, s51, 31 // (op 446) 
s_cmp_eq_i32 s11, 0 // (op 446) 
s_mov_b32 s6, scc // (op 446) 
s_cmp_eq_i32 s6, 0 // (op 446) 
s_cselect_b32 s7, 0, -1 // (op 446) 

s_and_b32 s6, s12, 31 // (op 446) 
s_lshl_b32 s18, 1, s6 // (op 446) 
s_add_i32 s19, s18, s7 // (op 446) 


s_and_b32 s7, s1, s19 // (op 446) 


s_add_i32 s1, s51, s7 // (op 446) 


s_ashr_i32 s7, s1, s6 // (op 446) 


s_lshl_b32 s1, s12, 24 // (op 446) 
s_ashr_i32 s6, s1, 31 // (op 446) 

s_xor_b32 s1, s7, s6 // (op 446) 

s_sub_i32 s7, s1, s6 // (op 446) 


s_mul_i32 s1, s0, s14 // (op 446) 

s_sub_u32 s0, s16, s1 // (op 446) 

s_mul_i32 s1, s7, s14 // (op 446) 
s_add_u32 s1, s1, s0 // (op 446) 


s_mul_hi_i32 s0, s16, s8 // (op 446) 
s_add_i32 s51, s0, s16 // (op 446) 

s_ashr_i32 s0, s51, 31 // (op 446) 
s_cmp_eq_i32 s8, 0 // (op 446) 
s_mov_b32 s6, scc // (op 446) 
s_cmp_eq_i32 s6, 0 // (op 446) 
s_cselect_b32 s7, 0, -1 // (op 446) 

s_and_b32 s6, s15, 31 // (op 446) 
s_lshl_b32 s18, 1, s6 // (op 446) 
s_add_i32 s19, s18, s7 // (op 446) 


s_and_b32 s7, s0, s19 // (op 446) 


s_add_i32 s0, s51, s7 // (op 446) 


s_ashr_i32 s7, s0, s6 // (op 446) 


s_lshl_b32 s0, s15, 24 // (op 446) 
s_ashr_i32 s6, s0, 31 // (op 446) 

s_xor_b32 s0, s7, s6 // (op 446) 

s_sub_i32 s7, s0, s6 // (op 446) 


s_add_u32 s18, s4, 16 // (op 446) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 446) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 446) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 446) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 446) 

s_mul_i32 s0, s14, s18 // (op 446) 

s_mul_i32 s6, s50, s0 // (op 446) 

s_mul_i32 s0, s7, s6 // (op 446) 

s_sub_u32 s6, s16, s0 // (op 446) 

s_mov_b32 s0, s6 // (op 446) 
s_mul_hi_i32 s51, s0, s13 // (op 446) 
s_add_i32 s18, s51, s0 // (op 446) 


s_ashr_i32 s0, s18, 31 // (op 446) 
s_cmp_eq_i32 s13, 0 // (op 446) 
s_mov_b32 s19, scc // (op 446) 
s_cmp_eq_i32 s19, 0 // (op 446) 
s_cselect_b32 s51, 0, -1 // (op 446) 

s_and_b32 s19, s49, 31 // (op 446) 
s_lshl_b32 s68, 1, s19 // (op 446) 
s_add_i32 s69, s68, s51 // (op 446) 


s_and_b32 s51, s0, s69 // (op 446) 


s_add_i32 s0, s18, s51 // (op 446) 


s_ashr_i32 s18, s0, s19 // (op 446) 


s_lshl_b32 s0, s49, 24 // (op 446) 
s_ashr_i32 s19, s0, 31 // (op 446) 

s_xor_b32 s0, s18, s19 // (op 446) 

s_sub_i32 s18, s0, s19 // (op 446) 


s_mul_i32 s0, s48, s14 // (op 446) 
s_add_u32 s68, s20, 32 // (op 446) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 446) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 446) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 446) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 446) 

s_sub_i32 s19, s68, s0 // (op 446) 

s_mul_i32 s51, s18, s19 // (op 446) 


s_sub_u32 s18, s6, s51 // (op 446) 


s_add_u32 s6, s18, s0 // (op 446) 


s_cmp_eq_i32 s7, 0 // (op 446) 
s_mov_b32 s0, scc // (op 446) 

s_cmp_eq_i32 s0, 0 // (op 446) 
s_cselect_b32 s7, s6, s1 // (op 446) 



v_lshl_add_u32 v11, s7, 5, v20 // (op 446) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 446) 
v_mov_b32 v13, 0 // (op 446) convert
v_mov_b32 v12, v11 // (op 446) convert

s_lshr_b32 s51, s0, 1 // (op 446) 

v_lshl_add_u32 v11, s51, 8, v10 // (op 446) 


v_mul_lo_u32 v15, v11, s3 // (op 446) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 446) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 446) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 446) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 446) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 446) most significant half



s_mul_i32 s1, s2, s25 // (op 448) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 448) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 448) least significant
s_mul_hi_u32 s6, s2, s24 // (op 448) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 448) most significant: sum
s_add_u32 s1, s1, s6 // (op 448) most significant: sum


s_mov_b32 s52, s22 // (op 448) reg expression
s_mov_b32 s53, s23 // (op 448) reg expression
s_mov_b32 s54, s0 // (op 448) reg expression
s_mov_b32 s55, 131072 // (op 448) call()

s_mul_hi_i32 s51, s16, s9 // (op 513) 
v_lshrrev_b32 v20, 3, v1 // (op 513) 
s_add_i32 s0, s51, s16 // (op 513) 

v_and_b32 v12, -8, v1 // (op 513) 
s_ashr_i32 s1, s0, 31 // (op 513) 
v_sub_u32 v13, v1, v12 // (op 513) 

s_cmp_eq_i32 s9, 0 // (op 513) 
s_mov_b32 s51, scc // (op 513) 
v_lshlrev_b32 v12, 4, v13 // (op 513) 

s_cmp_eq_i32 s51, 0 // (op 513) 
s_cselect_b32 s6, 0, -1 // (op 513) 

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
s_sub_u32 s68, s18, 1 // (op 513) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 513) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 513) 

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
s_lshl_b32 s68, 1, s6 // (op 513) 
s_add_i32 s69, s68, s7 // (op 513) 


s_and_b32 s7, s0, s69 // (op 513) 


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
s_add_i32 s68, s19, s18 // (op 513) 


s_and_b32 s18, s0, s68 // (op 513) 


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



v_lshl_add_u32 v13, s1, 4, v20 // (op 513) 


v_mul_lo_u32 v15, v13, s27 // (op 513) most significant: low of low * high
v_mul_hi_u32 v20, v13, s26 // (op 513) most significant: high of low * low
v_mul_lo_u32 v14, v13, s26 // (op 513) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 513) most significant: sum


s_ashr_i64 s[0:1], s[24:25], 7 // (op 513) 
s_lshr_b32 s51, s0, 1 // (op 513) 

v_lshl_add_u32 v13, s51, 8, v12 // (op 513) 


v_mov_b32 v17, 0 // (op 513) convert
v_mov_b32 v16, v13 // (op 513) convert

v_add_co_u32 v12, s[0:1], v16, v14 // (op 513) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 513) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 513) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 513) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 513) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 513) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 513) 

v_mov_b32 v20, v12 // (op 513) convert

s_mul_i32 s1, s5, 6 // (op 515) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 515) least significant
s_mul_hi_u32 s51, s4, 6 // (op 515) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 515) most significant: sum

s_mul_i32 s7, s26, s1 // (op 515) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 515) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 515) least significant
s_mul_hi_u32 s18, s26, s0 // (op 515) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 515) most significant: sum
s_add_u32 s7, s7, s18 // (op 515) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 515) 

s_mov_b32 s56, s28 // (op 515) reg expression
s_mov_b32 s57, s29 // (op 515) reg expression
s_mov_b32 s58, s0 // (op 515) reg expression
s_mov_b32 s59, 131072 // (op 515) call()

s_mul_hi_i32 s51, s16, s9 // (op 577) 
v_and_b32 v12, -32, v1 // (op 577) 
s_add_i32 s0, s51, s16 // (op 577) 

v_sub_u32 v13, v1, v12 // (op 577) 

s_ashr_i32 s1, s0, 31 // (op 577) 
v_lshrrev_b32 v12, 5, v1 // (op 577) 
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
s_sub_u32 s68, s18, 1 // (op 577) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 577) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 577) 

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
s_lshl_b32 s68, 1, s19 // (op 577) 
s_add_i32 s69, s68, s51 // (op 577) 


s_and_b32 s51, s0, s69 // (op 577) 


s_add_i32 s0, s18, s51 // (op 577) 


s_ashr_i32 s18, s0, s19 // (op 577) 


s_lshl_b32 s0, s49, 24 // (op 577) 
s_ashr_i32 s19, s0, 31 // (op 577) 

s_xor_b32 s0, s18, s19 // (op 577) 

s_sub_i32 s18, s0, s19 // (op 577) 


s_mul_i32 s0, s48, s14 // (op 577) 
s_add_u32 s68, s20, 32 // (op 577) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 577) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 577) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 577) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 577) 

s_sub_i32 s19, s68, s0 // (op 577) 

s_mul_i32 s51, s18, s19 // (op 577) 


s_sub_u32 s18, s6, s51 // (op 577) 


s_add_u32 s6, s18, s0 // (op 577) 


s_cmp_eq_i32 s7, 0 // (op 577) 
s_mov_b32 s0, scc // (op 577) 

s_cmp_eq_i32 s0, 0 // (op 577) 
s_cselect_b32 s7, s6, s1 // (op 577) 



v_lshl_add_u32 v14, s7, 5, v13 // (op 577) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 577) 
s_lshr_b32 s51, s0, 1 // (op 577) 

v_lshl_add_u32 v13, s51, 3, v12 // (op 577) 


v_mul_lo_u32 v12, v13, s30 // (op 577) 

v_add_u32 v13, v14, v12 // (op 577) 


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

s_mul_hi_i32 s51, s16, s9 // (op 641) 
v_and_b32 v12, 15, v1 // (op 641) 
s_add_i32 s0, s51, s16 // (op 641) 

v_and_b32 v14, 63, v1 // (op 641) 
s_ashr_i32 s1, s0, 31 // (op 641) 
v_lshrrev_b32 v15, 4, v14 // (op 641) 

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
s_sub_u32 s68, s18, 1 // (op 641) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 641) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 641) 

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
s_lshl_b32 s68, 1, s6 // (op 641) 
s_add_i32 s69, s68, s7 // (op 641) 


s_and_b32 s7, s0, s69 // (op 641) 


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
s_add_i32 s68, s19, s18 // (op 641) 


s_and_b32 s18, s0, s68 // (op 641) 


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



v_lshl_add_u32 v14, s1, 4, v12 // (op 641) 


v_mul_lo_u32 v12, v14, s36 // (op 641) 

s_ashr_i64 s[0:1], s[24:25], 7 // (op 641) 
s_lshr_b32 s51, s0, 1 // (op 641) 

v_lshl_add_u32 v14, s51, 3, v15 // (op 641) 


v_add_u32 v15, v14, v12 // (op 641) 


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

v_and_b32 v12, -2, v1 // (op 734) 
v_sub_u32 v14, v1, v12 // (op 734) 

v_lshlrev_b32 v12, 4, v14 // (op 734) 

v_lshrrev_b32 v14, 1, v1 // (op 734) 
v_lshl_add_u32 v16, v14, 5, v12 // (op 734) 


s_nop 0
v_readfirstlane_b32 s51, v16 // (op 734) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 786) 
v_sub_u32 v14, v1, v12 // (op 786) 

v_lshlrev_b32 v12, 4, v14 // (op 786) 

v_lshrrev_b32 v14, 3, v1 // (op 786) 
v_lshl_add_u32 v16, v14, 7, v12 // (op 786) 


v_mul_lo_u32 v12, v16, 6 // (op 786) 

v_lshrrev_b32 v14, 3, v12 // (op 786) 

v_and_b32 v12, -32, v1 // (op 839) 
v_sub_u32 v16, v1, v12 // (op 839) 

v_and_b32 v12, 4294967264, v1 // (op 839) 
v_add_u32 v17, v12, v16 // (op 839) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 162) 
s_and_b32 s6, s0, 4294967294 // (op 162) 

s_ashr_i64 s[0:1], s[24:25], 7 // (op 161) 
s_mov_b32 s7, s0 // (op 161) convert

s_cmp_lt_u32 s6, s7 // (op 161) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_161:
s_barrier  // (op 1027) 
s_add_u32 s6, s6, 1 // (op 163) 

s_lshl_b64 s[0:1], s[2:3], 6 // (op 417) 
v_mov_b64 v[18:19], s[0:1] // (op 417) convert


v_mov_b64 v[22:23], v[10:11] // (op 417) 
v_mov_b64 v[24:25], v[22:23] // (op 417) reg expression
s_mov_b32 m0, s51 // (op 417) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 417) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v18 // (op 417) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 417) most significant half

s_add_u32 m0, m0, 2048 // (op 417) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 417) Load value direct to lds



s_lshl_b64 s[0:1], s[2:3], 7 // (op 436) 
v_mov_b32 v12, s0 // (op 436) 
v_mov_b32 v16, s1 // (op 436) 
v_add_co_u32 v10, s[18:19], v10, v12 // (op 436) least significant half
v_addc_co_u32 v11, s[18:19], v11, v16, s[18:19] // (op 436) most significant half






v_mov_b32 v12, v20 // (op 179) 

buffer_load_dwordx3 v[22:24], v12, s[56:59], 0 offen // (op 179) Load value

v_add_u32 v20, 96, v20 // (op 505) 


s_barrier  // (op 1040) 
v_mov_b32 v12, v13 // (op 180) 



v_mov_b32 v16, v12 // (op 180) reg expression

s_waitcnt vmcnt(0)
ds_write_b96 v14, v[22:24] offset:8192 // (op 177) Store local data 
buffer_load_ubyte v21, v16, s[60:63], 0 offen offset: 0 // (op 180) Load value



s_lshl_b32 s7, s30, 2 // (op 569) 
v_add_u32 v13, v13, s7 // (op 569) 

s_barrier  // (op 1030) 
s_barrier  // (op 1054) 






ds_read_b64_tr_b8 v[22:23], v0 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:256 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:2048 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:2304 // (op 169) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v17, v21 offset:9728 // (op 178) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 1044) 



ds_read_b128 v[30:33], v4 offset:8192 // (op 171) Load local data 
ds_read_b64 v[34:35], v4 offset:8208 // (op 171) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1058) 


ds_read_u8 v12, v3 offset:9728 // (op 173) Load local data 


v_mov_b32 v16, v15 // (op 175) 

buffer_load_ubyte v21, v16, s[64:67], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v15, 4, v15 // (op 633) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:35], a[0:3], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 176) 




s_ashr_i64 s[0:1], s[24:25], 7 // (op 161) 
s_mov_b32 s7, s0 // (op 161) convert

s_cmp_lt_u32 s6, s7 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_161 if true)
RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_161:






















s_barrier  // (op 1118) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

s_mul_hi_i32 s0, s16, s9 // (op 881) 
v_lshrrev_b32 v0, 6, v1 // (op 881) 
s_add_i32 s1, s0, s16 // (op 881) 

v_and_b32 v3, 63, v1 // (op 881) 
s_ashr_i32 s0, s1, 31 // (op 881) 
v_lshrrev_b32 v4, 4, v3 // (op 881) 
s_cmp_eq_i32 s9, 0 // (op 881) 
s_mov_b32 s2, scc // (op 881) 
v_lshlrev_b32 v5, 2, v4 // (op 881) 

s_cmp_eq_i32 s2, 0 // (op 881) 
s_cselect_b32 s3, 0, -1 // (op 881) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 881) 


s_and_b32 s2, s10, 31 // (op 881) 
v_lshrrev_b32 v0, 2, v3 // (op 881) 

s_lshl_b32 s6, 1, s2 // (op 881) 
v_and_b32 v3, 3, v0 // (op 881) 

s_add_i32 s7, s6, s3 // (op 881) 


v_and_b32 v0, 3, v1 // (op 881) 
s_and_b32 s3, s0, s7 // (op 881) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 881) 


s_add_i32 s0, s1, s3 // (op 881) 


s_ashr_i32 s1, s0, s2 // (op 881) 


s_lshl_b32 s0, s10, 24 // (op 881) 
s_ashr_i32 s2, s0, 31 // (op 881) 

s_xor_b32 s0, s1, s2 // (op 881) 

s_sub_i32 s1, s0, s2 // (op 881) 


s_mul_hi_i32 s0, s1, s11 // (op 881) 
s_add_i32 s2, s0, s1 // (op 881) 

s_ashr_i32 s0, s2, 31 // (op 881) 
s_cmp_eq_i32 s11, 0 // (op 881) 
s_mov_b32 s3, scc // (op 881) 
s_cmp_eq_i32 s3, 0 // (op 881) 
s_cselect_b32 s6, 0, -1 // (op 881) 

s_and_b32 s3, s12, 31 // (op 881) 
s_lshl_b32 s7, 1, s3 // (op 881) 
s_add_i32 s18, s7, s6 // (op 881) 


s_and_b32 s19, s0, s18 // (op 881) 


s_add_i32 s0, s2, s19 // (op 881) 


s_ashr_i32 s2, s0, s3 // (op 881) 


s_lshl_b32 s0, s12, 24 // (op 881) 
s_ashr_i32 s3, s0, 31 // (op 881) 

s_xor_b32 s0, s2, s3 // (op 881) 

s_sub_i32 s2, s0, s3 // (op 881) 


s_mul_i32 s0, s1, s14 // (op 881) 
s_sub_u32 s3, s16, s0 // (op 881) 

s_mul_i32 s0, s2, s14 // (op 881) 
s_add_u32 s0, s0, s3 // (op 881) 

s_mul_hi_i32 s3, s16, s8 // (op 881) 
s_add_i32 s6, s3, s16 // (op 881) 

s_ashr_i32 s3, s6, 31 // (op 881) 
s_cmp_eq_i32 s8, 0 // (op 881) 
s_mov_b32 s7, scc // (op 881) 
s_cmp_eq_i32 s7, 0 // (op 881) 
s_cselect_b32 s18, 0, -1 // (op 881) 

s_and_b32 s7, s15, 31 // (op 881) 
s_lshl_b32 s19, 1, s7 // (op 881) 
s_add_i32 s22, s19, s18 // (op 881) 


s_and_b32 s18, s3, s22 // (op 881) 


s_add_i32 s3, s6, s18 // (op 881) 


s_ashr_i32 s6, s3, s7 // (op 881) 


s_lshl_b32 s3, s15, 24 // (op 881) 
s_ashr_i32 s7, s3, 31 // (op 881) 

s_xor_b32 s3, s6, s7 // (op 881) 

s_sub_i32 s6, s3, s7 // (op 881) 


s_add_u32 s18, s4, 16 // (op 881) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 881) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 881) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 881) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 4 // (op 881) 

s_mul_i32 s3, s14, s18 // (op 881) 
s_mul_i32 s7, s50, s3 // (op 881) 

s_mul_i32 s3, s6, s7 // (op 881) 

s_sub_u32 s7, s16, s3 // (op 881) 

s_mov_b32 s3, s7 // (op 881) 
s_mul_hi_i32 s22, s3, s13 // (op 881) 
s_add_i32 s23, s22, s3 // (op 881) 


s_ashr_i32 s3, s23, 31 // (op 881) 
s_cmp_eq_i32 s13, 0 // (op 881) 
s_mov_b32 s22, scc // (op 881) 
s_cmp_eq_i32 s22, 0 // (op 881) 
s_cselect_b32 s24, 0, -1 // (op 881) 

s_and_b32 s22, s49, 31 // (op 881) 
s_lshl_b32 s25, 1, s22 // (op 881) 
s_add_i32 s26, s25, s24 // (op 881) 


s_and_b32 s24, s3, s26 // (op 881) 


s_add_i32 s3, s23, s24 // (op 881) 


s_ashr_i32 s23, s3, s22 // (op 881) 


s_lshl_b32 s3, s49, 24 // (op 881) 
s_ashr_i32 s22, s3, 31 // (op 881) 

s_xor_b32 s3, s23, s22 // (op 881) 

s_sub_i32 s23, s3, s22 // (op 881) 


s_mul_i32 s3, s48, s14 // (op 881) 
s_add_u32 s24, s20, 32 // (op 881) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 881) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 881) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 881) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 5 // (op 881) 

s_sub_i32 s22, s24, s3 // (op 881) 

s_mul_i32 s24, s23, s22 // (op 881) 

s_sub_u32 s22, s7, s24 // (op 881) 


s_add_u32 s7, s22, s3 // (op 881) 


s_cmp_eq_i32 s6, 0 // (op 881) 
s_mov_b32 s3, scc // (op 881) 

s_cmp_eq_i32 s3, 0 // (op 881) 
s_cselect_b32 s6, s7, s0 // (op 881) 


s_mul_i32 s0, s2, s18 // (op 881) 


v_lshl_add_u32 v0, s6, 5, v4 // (op 881) 


s_sub_i32 s2, s1, s0 // (op 881) 


s_cmp_eq_i32 s3, 0 // (op 881) 
s_cselect_b32 s22, s23, s2 // (op 881) 



v_lshl_add_u32 v3, s22, 4, v5 // (op 881) 


v_mul_lo_u32 v4, v3, s40 // (op 881) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 881) 


s_mul_i32 s7, s40, s5 // (op 883) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 883) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 883) least significant
s_mul_hi_u32 s1, s40, s4 // (op 883) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 883) most significant: sum
s_add_u32 s7, s7, s1 // (op 883) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 883) 

s_mov_b32 s0, s42 // (op 883) reg expression
s_mov_b32 s1, s43 // (op 883) reg expression
s_mov_b32 s2, s18 // (op 883) reg expression
s_mov_b32 s3, 131072 // (op 883) call()




buffer_store_dwordx4 v[10:13], v3, s[0:3], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_123:
s_mul_hi_i32 s6, s16, s9 // (op 704) 
v_lshrrev_b32 v0, 6, v1 // (op 704) 
s_add_i32 s7, s6, s16 // (op 704) 

v_and_b32 v3, 63, v1 // (op 704) 
s_ashr_i32 s6, s7, 31 // (op 704) 
v_lshrrev_b32 v4, 4, v3 // (op 704) 
s_cmp_eq_i32 s9, 0 // (op 704) 
s_mov_b32 s18, scc // (op 704) 
v_lshlrev_b32 v5, 2, v4 // (op 704) 

s_cmp_eq_i32 s18, 0 // (op 704) 
s_cselect_b32 s19, 0, -1 // (op 704) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 704) 


s_and_b32 s18, s10, 31 // (op 704) 
v_lshrrev_b32 v0, 2, v3 // (op 704) 

s_lshl_b32 s22, 1, s18 // (op 704) 
v_and_b32 v3, 3, v0 // (op 704) 

s_add_i32 s23, s22, s19 // (op 704) 


v_and_b32 v0, 3, v1 // (op 704) 
s_and_b32 s19, s6, s23 // (op 704) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 704) 


s_add_i32 s6, s7, s19 // (op 704) 


s_ashr_i32 s7, s6, s18 // (op 704) 


s_lshl_b32 s6, s10, 24 // (op 704) 
s_ashr_i32 s18, s6, 31 // (op 704) 

s_xor_b32 s6, s7, s18 // (op 704) 

s_sub_i32 s7, s6, s18 // (op 704) 


s_mul_hi_i32 s6, s7, s11 // (op 704) 
s_add_i32 s18, s6, s7 // (op 704) 

s_ashr_i32 s6, s18, 31 // (op 704) 
s_cmp_eq_i32 s11, 0 // (op 704) 
s_mov_b32 s19, scc // (op 704) 
s_cmp_eq_i32 s19, 0 // (op 704) 
s_cselect_b32 s22, 0, -1 // (op 704) 

s_and_b32 s19, s12, 31 // (op 704) 
s_lshl_b32 s23, 1, s19 // (op 704) 
s_add_i32 s24, s23, s22 // (op 704) 


s_and_b32 s22, s6, s24 // (op 704) 


s_add_i32 s6, s18, s22 // (op 704) 


s_ashr_i32 s18, s6, s19 // (op 704) 


s_lshl_b32 s6, s12, 24 // (op 704) 
s_ashr_i32 s19, s6, 31 // (op 704) 

s_xor_b32 s6, s18, s19 // (op 704) 

s_sub_i32 s18, s6, s19 // (op 704) 


s_mul_i32 s6, s7, s14 // (op 704) 
s_sub_u32 s19, s16, s6 // (op 704) 

s_mul_i32 s6, s18, s14 // (op 704) 
s_add_u32 s6, s6, s19 // (op 704) 

s_mul_hi_i32 s19, s16, s8 // (op 704) 
s_add_i32 s22, s19, s16 // (op 704) 

s_ashr_i32 s19, s22, 31 // (op 704) 
s_cmp_eq_i32 s8, 0 // (op 704) 
s_mov_b32 s23, scc // (op 704) 
s_cmp_eq_i32 s23, 0 // (op 704) 
s_cselect_b32 s24, 0, -1 // (op 704) 

s_and_b32 s23, s15, 31 // (op 704) 
s_lshl_b32 s25, 1, s23 // (op 704) 
s_add_i32 s26, s25, s24 // (op 704) 


s_and_b32 s24, s19, s26 // (op 704) 


s_add_i32 s19, s22, s24 // (op 704) 


s_ashr_i32 s22, s19, s23 // (op 704) 


s_lshl_b32 s19, s15, 24 // (op 704) 
s_ashr_i32 s23, s19, 31 // (op 704) 

s_xor_b32 s19, s22, s23 // (op 704) 

s_sub_i32 s22, s19, s23 // (op 704) 


s_add_u32 s38, s4, 16 // (op 704) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 704) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 704) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 704) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 704) 

s_mul_i32 s19, s14, s38 // (op 704) 
s_mul_i32 s23, s50, s19 // (op 704) 

s_mul_i32 s19, s22, s23 // (op 704) 

s_sub_u32 s23, s16, s19 // (op 704) 

s_mov_b32 s19, s23 // (op 704) 
s_mul_hi_i32 s24, s19, s13 // (op 704) 
s_add_i32 s25, s24, s19 // (op 704) 


s_ashr_i32 s19, s25, 31 // (op 704) 
s_cmp_eq_i32 s13, 0 // (op 704) 
s_mov_b32 s24, scc // (op 704) 
s_cmp_eq_i32 s24, 0 // (op 704) 
s_cselect_b32 s26, 0, -1 // (op 704) 

s_and_b32 s24, s49, 31 // (op 704) 
s_lshl_b32 s27, 1, s24 // (op 704) 
s_add_i32 s28, s27, s26 // (op 704) 


s_and_b32 s26, s19, s28 // (op 704) 


s_add_i32 s19, s25, s26 // (op 704) 


s_ashr_i32 s25, s19, s24 // (op 704) 


s_lshl_b32 s19, s49, 24 // (op 704) 
s_ashr_i32 s24, s19, 31 // (op 704) 

s_xor_b32 s19, s25, s24 // (op 704) 

s_sub_i32 s25, s19, s24 // (op 704) 


s_mul_i32 s19, s48, s14 // (op 704) 
s_add_u32 s26, s20, 32 // (op 704) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 704) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 704) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 704) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 5 // (op 704) 

s_sub_i32 s24, s26, s19 // (op 704) 

s_mul_i32 s26, s25, s24 // (op 704) 

s_sub_u32 s24, s23, s26 // (op 704) 


s_add_u32 s23, s24, s19 // (op 704) 


s_cmp_eq_i32 s22, 0 // (op 704) 
s_mov_b32 s19, scc // (op 704) 

s_cmp_eq_i32 s19, 0 // (op 704) 
s_cselect_b32 s22, s23, s6 // (op 704) 


s_mul_i32 s6, s18, s38 // (op 704) 


v_lshl_add_u32 v0, s22, 5, v4 // (op 704) 


s_sub_i32 s18, s7, s6 // (op 704) 


s_cmp_eq_i32 s19, 0 // (op 704) 
s_cselect_b32 s6, s25, s18 // (op 704) 



v_lshl_add_u32 v3, s6, 4, v5 // (op 704) 


v_mul_lo_u32 v4, v3, s44 // (op 704) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 704) 


s_mul_i32 s7, s44, s5 // (op 706) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 706) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 706) least significant
s_mul_hi_u32 s19, s44, s4 // (op 706) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 706) most significant: sum
s_add_u32 s7, s7, s19 // (op 706) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 706) 

s_mov_b32 s36, s46 // (op 706) reg expression
s_mov_b32 s37, s47 // (op 706) reg expression
s_mov_b32 s38, s18 // (op 706) reg expression
s_mov_b32 s39, 131072 // (op 706) call()





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


s_mul_hi_i32 s6, s16, s9 // (op 898) 
v_lshrrev_b32 v0, 6, v1 // (op 898) 
s_add_i32 s7, s6, s16 // (op 898) 

v_and_b32 v3, 63, v1 // (op 898) 
s_ashr_i32 s6, s7, 31 // (op 898) 
v_lshrrev_b32 v10, 4, v3 // (op 898) 
s_cmp_eq_i32 s9, 0 // (op 898) 
s_mov_b32 s18, scc // (op 898) 
v_lshlrev_b32 v11, 2, v10 // (op 898) 

s_cmp_eq_i32 s18, 0 // (op 898) 
s_cselect_b32 s19, 0, -1 // (op 898) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 898) 


s_and_b32 s18, s10, 31 // (op 898) 
v_lshrrev_b32 v0, 2, v3 // (op 898) 

s_lshl_b32 s22, 1, s18 // (op 898) 
v_and_b32 v3, 3, v0 // (op 898) 

s_add_i32 s23, s22, s19 // (op 898) 


v_and_b32 v0, 3, v1 // (op 898) 
s_and_b32 s19, s6, s23 // (op 898) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 898) 


s_add_i32 s6, s7, s19 // (op 898) 


s_ashr_i32 s7, s6, s18 // (op 898) 


s_lshl_b32 s6, s10, 24 // (op 898) 
s_ashr_i32 s18, s6, 31 // (op 898) 

s_xor_b32 s6, s7, s18 // (op 898) 

s_sub_i32 s7, s6, s18 // (op 898) 


s_mul_hi_i32 s6, s7, s11 // (op 898) 
s_add_i32 s18, s6, s7 // (op 898) 

s_ashr_i32 s6, s18, 31 // (op 898) 
s_cmp_eq_i32 s11, 0 // (op 898) 
s_mov_b32 s19, scc // (op 898) 
s_cmp_eq_i32 s19, 0 // (op 898) 
s_cselect_b32 s22, 0, -1 // (op 898) 

s_and_b32 s19, s12, 31 // (op 898) 
s_lshl_b32 s23, 1, s19 // (op 898) 
s_add_i32 s24, s23, s22 // (op 898) 


s_and_b32 s22, s6, s24 // (op 898) 


s_add_i32 s6, s18, s22 // (op 898) 


s_ashr_i32 s18, s6, s19 // (op 898) 


s_lshl_b32 s6, s12, 24 // (op 898) 
s_ashr_i32 s19, s6, 31 // (op 898) 

s_xor_b32 s6, s18, s19 // (op 898) 

s_sub_i32 s18, s6, s19 // (op 898) 


s_mul_i32 s6, s7, s14 // (op 898) 
s_sub_u32 s19, s16, s6 // (op 898) 

s_mul_i32 s6, s18, s14 // (op 898) 
s_add_u32 s6, s6, s19 // (op 898) 

s_mul_hi_i32 s19, s16, s8 // (op 898) 
s_add_i32 s22, s19, s16 // (op 898) 

s_ashr_i32 s19, s22, 31 // (op 898) 
s_cmp_eq_i32 s8, 0 // (op 898) 
s_mov_b32 s23, scc // (op 898) 
s_cmp_eq_i32 s23, 0 // (op 898) 
s_cselect_b32 s24, 0, -1 // (op 898) 

s_and_b32 s23, s15, 31 // (op 898) 
s_lshl_b32 s25, 1, s23 // (op 898) 
s_add_i32 s26, s25, s24 // (op 898) 


s_and_b32 s24, s19, s26 // (op 898) 


s_add_i32 s19, s22, s24 // (op 898) 


s_ashr_i32 s22, s19, s23 // (op 898) 


s_lshl_b32 s19, s15, 24 // (op 898) 
s_ashr_i32 s23, s19, 31 // (op 898) 

s_xor_b32 s19, s22, s23 // (op 898) 

s_sub_i32 s22, s19, s23 // (op 898) 


s_add_u32 s38, s4, 16 // (op 898) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 898) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 898) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 898) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 898) 

s_mul_i32 s19, s14, s38 // (op 898) 
s_mul_i32 s23, s50, s19 // (op 898) 

s_mul_i32 s19, s22, s23 // (op 898) 

s_sub_u32 s23, s16, s19 // (op 898) 

s_mov_b32 s19, s23 // (op 898) 
s_mul_hi_i32 s24, s19, s13 // (op 898) 
s_add_i32 s25, s24, s19 // (op 898) 


s_ashr_i32 s19, s25, 31 // (op 898) 
s_cmp_eq_i32 s13, 0 // (op 898) 
s_mov_b32 s24, scc // (op 898) 
s_cmp_eq_i32 s24, 0 // (op 898) 
s_cselect_b32 s26, 0, -1 // (op 898) 

s_and_b32 s24, s49, 31 // (op 898) 
s_lshl_b32 s27, 1, s24 // (op 898) 
s_add_i32 s28, s27, s26 // (op 898) 


s_and_b32 s26, s19, s28 // (op 898) 


s_add_i32 s19, s25, s26 // (op 898) 


s_ashr_i32 s25, s19, s24 // (op 898) 


s_lshl_b32 s19, s49, 24 // (op 898) 
s_ashr_i32 s24, s19, 31 // (op 898) 

s_xor_b32 s19, s25, s24 // (op 898) 

s_sub_i32 s25, s19, s24 // (op 898) 


s_mul_i32 s19, s48, s14 // (op 898) 
s_add_u32 s26, s20, 32 // (op 898) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 898) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 898) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 898) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 5 // (op 898) 

s_sub_i32 s24, s26, s19 // (op 898) 

s_mul_i32 s26, s25, s24 // (op 898) 

s_sub_u32 s24, s23, s26 // (op 898) 


s_add_u32 s23, s24, s19 // (op 898) 


s_cmp_eq_i32 s22, 0 // (op 898) 
s_mov_b32 s19, scc // (op 898) 

s_cmp_eq_i32 s19, 0 // (op 898) 
s_cselect_b32 s22, s23, s6 // (op 898) 


s_mul_i32 s6, s18, s38 // (op 898) 


v_lshl_add_u32 v0, s22, 5, v10 // (op 898) 


s_sub_i32 s18, s7, s6 // (op 898) 


s_cmp_eq_i32 s19, 0 // (op 898) 
s_cselect_b32 s6, s25, s18 // (op 898) 



v_lshl_add_u32 v3, s6, 4, v11 // (op 898) 


v_mul_lo_u32 v10, v3, s40 // (op 898) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 898) 


s_mul_i32 s7, s40, s5 // (op 900) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 900) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 900) least significant
s_mul_hi_u32 s19, s40, s4 // (op 900) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 900) most significant: sum
s_add_u32 s7, s7, s19 // (op 900) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 900) 

s_mov_b32 s0, s42 // (op 900) reg expression
s_mov_b32 s1, s43 // (op 900) reg expression
s_mov_b32 s2, s18 // (op 900) reg expression
s_mov_b32 s3, 131072 // (op 900) call()




buffer_store_dwordx4 v[4:7], v3, s[0:3], 0 offen // (op 134) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.LRR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__end:
.size RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_, .LRR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__end-RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 56
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 11520 // lds bytes
  .amdhsa_accum_offset 52
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
    DataTypeB: B6
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
  - .name: RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 11520
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 50
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

