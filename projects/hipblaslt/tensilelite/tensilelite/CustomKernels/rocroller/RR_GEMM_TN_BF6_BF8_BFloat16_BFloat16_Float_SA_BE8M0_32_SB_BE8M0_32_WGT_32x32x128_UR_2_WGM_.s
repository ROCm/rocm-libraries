// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_,@function
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 735) 
v_and_b32 v3, 15, v1 // (op 735) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 735) 


v_and_b32 v0, 63, v1 // (op 735) 
v_lshrrev_b32 v3, 4, v0 // (op 735) 

v_lshlrev_b32 v0, 5, v3 // (op 735) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 735) 


v_mul_lo_u32 v0, v3, 6 // (op 735) 

v_lshrrev_b32 v3, 3, v0 // (op 735) 

v_lshrrev_b32 v0, 6, v1 // (op 787) 
v_and_b32 v4, 1, v0 // (op 787) 

v_and_b32 v0, 15, v1 // (op 787) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 787) 


v_and_b32 v0, 48, v1 // (op 787) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 787) 


v_lshrrev_b32 v0, 7, v1 // (op 845) 
v_and_b32 v5, 15, v1 // (op 845) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 845) 


v_and_b32 v0, 63, v1 // (op 845) 
v_lshrrev_b32 v5, 4, v0 // (op 845) 

v_lshlrev_b32 v0, 5, v5 // (op 845) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 845) 


v_mul_lo_u32 v0, v5, 6 // (op 845) 

v_lshrrev_b32 v5, 3, v0 // (op 845) 

v_lshrrev_b32 v0, 6, v1 // (op 884) 
v_and_b32 v6, 1, v0 // (op 884) 

v_and_b32 v0, 15, v1 // (op 884) 
v_lshl_add_u32 v7, v6, 4, v0 // (op 884) 


v_and_b32 v0, 48, v1 // (op 884) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 884) 


v_mov_b32 v0, s8 // (op 2) Move value
v_mov_b32 v7, s9 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
s_mul_hi_i32 s0, s16, s12 // (op 445) 
v_lshrrev_b32 v8, 3, v1 // (op 445) 
s_add_i32 s1, s0, s16 // (op 445) 

v_and_b32 v9, -8, v1 // (op 445) 
s_ashr_i32 s0, s1, 31 // (op 445) 
v_sub_u32 v10, v1, v9 // (op 445) 

s_cmp_eq_i32 s12, 0 // (op 445) 
s_mov_b32 s8, scc // (op 445) 
v_lshlrev_b32 v9, 4, v10 // (op 445) 

s_cmp_eq_i32 s8, 0 // (op 445) 
s_cselect_b32 s9, 0, -1 // (op 445) 

v_mov_b32 v11, 0 // (op 445) convert
v_mov_b32 v10, v9 // (op 445) convert

s_and_b32 s8, s13, 31 // (op 445) 
s_lshl_b32 s18, 1, s8 // (op 445) 
s_add_i32 s19, s18, s9 // (op 445) 


s_and_b32 s9, s0, s19 // (op 445) 


s_add_i32 s0, s1, s9 // (op 445) 


s_ashr_i32 s1, s0, s8 // (op 445) 


s_lshl_b32 s0, s13, 24 // (op 445) 
s_ashr_i32 s8, s0, 31 // (op 445) 

s_xor_b32 s0, s1, s8 // (op 445) 

s_sub_i32 s1, s0, s8 // (op 445) 


s_mul_hi_i32 s0, s1, s14 // (op 445) 
s_add_i32 s8, s0, s1 // (op 445) 

s_ashr_i32 s0, s8, 31 // (op 445) 
s_cmp_eq_i32 s14, 0 // (op 445) 
s_mov_b32 s9, scc // (op 445) 
s_cmp_eq_i32 s9, 0 // (op 445) 
s_cselect_b32 s18, 0, -1 // (op 445) 

s_and_b32 s9, s15, 31 // (op 445) 
s_lshl_b32 s19, 1, s9 // (op 445) 
s_add_i32 s49, s19, s18 // (op 445) 


s_and_b32 s18, s0, s49 // (op 445) 


s_add_i32 s0, s8, s18 // (op 445) 


s_ashr_i32 s8, s0, s9 // (op 445) 


s_lshl_b32 s0, s15, 24 // (op 445) 
s_ashr_i32 s9, s0, 31 // (op 445) 

s_xor_b32 s0, s8, s9 // (op 445) 

s_sub_i32 s8, s0, s9 // (op 445) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s45 // (op 445) 

s_sub_u32 s1, s16, s0 // (op 445) 

s_mul_i32 s0, s8, s45 // (op 445) 
s_add_u32 s0, s0, s1 // (op 445) 


s_mul_hi_i32 s1, s16, s11 // (op 445) 
s_add_i32 s8, s1, s16 // (op 445) 

s_ashr_i32 s1, s8, 31 // (op 445) 
s_cmp_eq_i32 s11, 0 // (op 445) 
s_mov_b32 s9, scc // (op 445) 
s_cmp_eq_i32 s9, 0 // (op 445) 
s_cselect_b32 s18, 0, -1 // (op 445) 

s_and_b32 s9, s46, 31 // (op 445) 
s_lshl_b32 s19, 1, s9 // (op 445) 
s_add_i32 s49, s19, s18 // (op 445) 


s_and_b32 s18, s1, s49 // (op 445) 


s_add_i32 s1, s8, s18 // (op 445) 


s_ashr_i32 s8, s1, s9 // (op 445) 


s_lshl_b32 s1, s46, 24 // (op 445) 
s_ashr_i32 s9, s1, 31 // (op 445) 

s_xor_b32 s1, s8, s9 // (op 445) 

s_sub_i32 s8, s1, s9 // (op 445) 


s_add_u32 s18, s4, 32 // (op 445) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 445) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 445) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 445) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 445) 

s_mul_i32 s1, s45, s18 // (op 445) 

s_mul_i32 s9, s10, s1 // (op 445) 

s_mul_i32 s1, s8, s9 // (op 445) 

s_sub_u32 s9, s16, s1 // (op 445) 

s_mov_b32 s1, s9 // (op 445) 
s_mul_hi_i32 s18, s1, s44 // (op 445) 
s_add_i32 s19, s18, s1 // (op 445) 


s_ashr_i32 s1, s19, 31 // (op 445) 
s_cmp_eq_i32 s44, 0 // (op 445) 
s_mov_b32 s18, scc // (op 445) 
s_cmp_eq_i32 s18, 0 // (op 445) 
s_cselect_b32 s49, 0, -1 // (op 445) 

s_and_b32 s18, s48, 31 // (op 445) 
s_lshl_b32 s50, 1, s18 // (op 445) 
s_add_i32 s51, s50, s49 // (op 445) 


s_and_b32 s49, s1, s51 // (op 445) 


s_add_i32 s1, s19, s49 // (op 445) 


s_ashr_i32 s19, s1, s18 // (op 445) 


s_lshl_b32 s1, s48, 24 // (op 445) 
s_ashr_i32 s18, s1, 31 // (op 445) 

s_xor_b32 s1, s19, s18 // (op 445) 

s_sub_i32 s19, s1, s18 // (op 445) 


s_mul_i32 s1, s47, s45 // (op 445) 
s_add_u32 s50, s6, 32 // (op 445) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 445) End of Int64 add, unlocking SCC
s_sub_u32 s52, s50, 1 // (op 445) least significant half; sets scc
s_subb_u32 s53, s51, 0 // (op 445) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 445) 

s_sub_i32 s18, s50, s1 // (op 445) 

s_mul_i32 s49, s19, s18 // (op 445) 


s_sub_u32 s18, s9, s49 // (op 445) 


s_add_u32 s9, s18, s1 // (op 445) 


s_cmp_eq_i32 s8, 0 // (op 445) 
s_mov_b32 s1, scc // (op 445) 

s_cmp_eq_i32 s1, 0 // (op 445) 
s_cselect_b32 s8, s9, s0 // (op 445) 



v_lshl_add_u32 v9, s8, 5, v8 // (op 445) 


v_mul_lo_u32 v13, v9, s3 // (op 445) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 445) most significant: high of low * low
v_mul_lo_u32 v12, v9, s2 // (op 445) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 445) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v10 // (op 445) least significant half
v_addc_co_u32 v13, s[0:1], v13, v11, s[0:1] // (op 445) most significant half



v_mul_lo_u32 v9, v13, 6 // (op 445) most significant: low of high * low
v_mul_hi_u32 v10, v12, 6 // (op 445) most significant: high of low * low
v_mul_lo_u32 v8, v12, 6 // (op 445) least significant: low of low * low
v_add_u32 v9, v9, v10 // (op 445) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[8:9] // (op 445) 

v_mov_b32 v8, v10 // (op 445) convert

s_mul_i32 s1, s7, 6 // (op 447) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 447) least significant
s_mul_hi_u32 s8, s6, 6 // (op 447) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 447) most significant: sum

s_mul_i32 s9, s2, s1 // (op 447) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 447) most significant: low of high * low
s_mul_i32 s8, s2, s0 // (op 447) least significant
s_mul_hi_u32 s19, s2, s0 // (op 447) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 447) most significant: sum
s_add_u32 s9, s9, s19 // (op 447) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 447) 

s_mov_b32 s52, s20 // (op 447) reg expression
s_mov_b32 s53, s21 // (op 447) reg expression
s_mov_b32 s54, s0 // (op 447) reg expression
s_mov_b32 s55, 131072 // (op 447) call()

s_mul_hi_i32 s0, s16, s12 // (op 464) 
v_lshrrev_b32 v9, 3, v1 // (op 464) 
s_add_i32 s1, s0, s16 // (op 464) 

v_and_b32 v10, -8, v1 // (op 464) 
s_ashr_i32 s0, s1, 31 // (op 464) 
v_sub_u32 v11, v1, v10 // (op 464) 

s_cmp_eq_i32 s12, 0 // (op 464) 
s_mov_b32 s8, scc // (op 464) 
v_lshlrev_b32 v10, 4, v11 // (op 464) 

s_cmp_eq_i32 s8, 0 // (op 464) 
s_cselect_b32 s9, 0, -1 // (op 464) 

v_mov_b32 v13, 0 // (op 464) convert
v_mov_b32 v12, v10 // (op 464) convert

s_and_b32 s8, s13, 31 // (op 464) 
s_lshl_b32 s18, 1, s8 // (op 464) 
s_add_i32 s19, s18, s9 // (op 464) 


s_and_b32 s9, s0, s19 // (op 464) 


s_add_i32 s0, s1, s9 // (op 464) 


s_ashr_i32 s1, s0, s8 // (op 464) 


s_lshl_b32 s0, s13, 24 // (op 464) 
s_ashr_i32 s8, s0, 31 // (op 464) 

s_xor_b32 s0, s1, s8 // (op 464) 

s_sub_i32 s1, s0, s8 // (op 464) 


s_mul_hi_i32 s0, s1, s14 // (op 464) 
s_add_i32 s8, s0, s1 // (op 464) 

s_ashr_i32 s0, s8, 31 // (op 464) 
s_cmp_eq_i32 s14, 0 // (op 464) 
s_mov_b32 s9, scc // (op 464) 
s_cmp_eq_i32 s9, 0 // (op 464) 
s_cselect_b32 s18, 0, -1 // (op 464) 

s_and_b32 s9, s15, 31 // (op 464) 
s_lshl_b32 s19, 1, s9 // (op 464) 
s_add_i32 s49, s19, s18 // (op 464) 


s_and_b32 s18, s0, s49 // (op 464) 


s_add_i32 s0, s8, s18 // (op 464) 


s_ashr_i32 s8, s0, s9 // (op 464) 


s_lshl_b32 s0, s15, 24 // (op 464) 
s_ashr_i32 s9, s0, 31 // (op 464) 

s_xor_b32 s0, s8, s9 // (op 464) 

s_sub_i32 s8, s0, s9 // (op 464) 


s_mul_i32 s0, s1, s45 // (op 464) 

s_sub_u32 s1, s16, s0 // (op 464) 

s_mul_i32 s0, s8, s45 // (op 464) 
s_add_u32 s0, s0, s1 // (op 464) 


s_mul_hi_i32 s1, s16, s11 // (op 464) 
s_add_i32 s8, s1, s16 // (op 464) 

s_ashr_i32 s1, s8, 31 // (op 464) 
s_cmp_eq_i32 s11, 0 // (op 464) 
s_mov_b32 s9, scc // (op 464) 
s_cmp_eq_i32 s9, 0 // (op 464) 
s_cselect_b32 s18, 0, -1 // (op 464) 

s_and_b32 s9, s46, 31 // (op 464) 
s_lshl_b32 s19, 1, s9 // (op 464) 
s_add_i32 s49, s19, s18 // (op 464) 


s_and_b32 s18, s1, s49 // (op 464) 


s_add_i32 s1, s8, s18 // (op 464) 


s_ashr_i32 s8, s1, s9 // (op 464) 


s_lshl_b32 s1, s46, 24 // (op 464) 
s_ashr_i32 s9, s1, 31 // (op 464) 

s_xor_b32 s1, s8, s9 // (op 464) 

s_sub_i32 s8, s1, s9 // (op 464) 


s_add_u32 s18, s4, 32 // (op 464) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 464) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 464) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 464) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 464) 

s_mul_i32 s1, s45, s18 // (op 464) 

s_mul_i32 s9, s10, s1 // (op 464) 

s_mul_i32 s1, s8, s9 // (op 464) 

s_sub_u32 s9, s16, s1 // (op 464) 

s_mov_b32 s1, s9 // (op 464) 
s_mul_hi_i32 s18, s1, s44 // (op 464) 
s_add_i32 s19, s18, s1 // (op 464) 


s_ashr_i32 s1, s19, 31 // (op 464) 
s_cmp_eq_i32 s44, 0 // (op 464) 
s_mov_b32 s18, scc // (op 464) 
s_cmp_eq_i32 s18, 0 // (op 464) 
s_cselect_b32 s49, 0, -1 // (op 464) 

s_and_b32 s18, s48, 31 // (op 464) 
s_lshl_b32 s50, 1, s18 // (op 464) 
s_add_i32 s51, s50, s49 // (op 464) 


s_and_b32 s49, s1, s51 // (op 464) 


s_add_i32 s1, s19, s49 // (op 464) 


s_ashr_i32 s19, s1, s18 // (op 464) 


s_lshl_b32 s1, s48, 24 // (op 464) 
s_ashr_i32 s18, s1, 31 // (op 464) 

s_xor_b32 s1, s19, s18 // (op 464) 

s_sub_i32 s19, s1, s18 // (op 464) 


s_mul_i32 s1, s47, s45 // (op 464) 
s_add_u32 s50, s6, 32 // (op 464) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 464) End of Int64 add, unlocking SCC
s_sub_u32 s56, s50, 1 // (op 464) least significant half; sets scc
s_subb_u32 s57, s51, 0 // (op 464) most significant half; uses scc

s_ashr_i64 s[50:51], s[56:57], 5 // (op 464) 

s_sub_i32 s18, s50, s1 // (op 464) 

s_mul_i32 s49, s19, s18 // (op 464) 


s_sub_u32 s18, s9, s49 // (op 464) 


s_add_u32 s9, s18, s1 // (op 464) 


s_cmp_eq_i32 s8, 0 // (op 464) 
s_mov_b32 s1, scc // (op 464) 

s_cmp_eq_i32 s1, 0 // (op 464) 
s_cselect_b32 s8, s9, s0 // (op 464) 



v_lshl_add_u32 v10, s8, 5, v9 // (op 464) 


v_mul_lo_u32 v15, v10, s3 // (op 464) most significant: low of low * high
v_mul_hi_u32 v9, v10, s2 // (op 464) most significant: high of low * low
v_mul_lo_u32 v14, v10, s2 // (op 464) least significant: low of low * low
v_add_u32 v15, v15, v9 // (op 464) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 464) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 464) most significant half



v_mul_lo_u32 v13, v15, 6 // (op 464) most significant: low of high * low
v_mul_hi_u32 v9, v14, 6 // (op 464) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 464) least significant: low of low * low
v_add_u32 v13, v13, v9 // (op 464) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 464) 

v_mov_b32 v9, v10 // (op 464) convert

s_mul_i32 s1, s7, 6 // (op 466) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 466) least significant
s_mul_hi_u32 s8, s6, 6 // (op 466) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 466) most significant: sum

s_mul_i32 s9, s2, s1 // (op 466) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 466) most significant: low of high * low
s_mul_i32 s8, s2, s0 // (op 466) least significant
s_mul_hi_u32 s19, s2, s0 // (op 466) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 466) most significant: sum
s_add_u32 s9, s9, s19 // (op 466) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 466) 

s_mov_b32 s52, s20 // (op 466) reg expression
s_mov_b32 s53, s21 // (op 466) reg expression
s_mov_b32 s54, s0 // (op 466) reg expression
s_mov_b32 s55, 131072 // (op 466) call()

s_mul_hi_i32 s0, s16, s12 // (op 509) 
v_lshrrev_b32 v10, 3, v1 // (op 509) 
s_add_i32 s1, s0, s16 // (op 509) 

v_and_b32 v11, -8, v1 // (op 509) 
s_ashr_i32 s0, s1, 31 // (op 509) 
v_sub_u32 v12, v1, v11 // (op 509) 

s_cmp_eq_i32 s12, 0 // (op 509) 
s_mov_b32 s8, scc // (op 509) 
v_lshlrev_b32 v11, 4, v12 // (op 509) 

s_cmp_eq_i32 s8, 0 // (op 509) 
s_cselect_b32 s9, 0, -1 // (op 509) 

v_mov_b32 v13, 0 // (op 509) convert
v_mov_b32 v12, v11 // (op 509) convert

s_and_b32 s8, s13, 31 // (op 509) 
s_lshl_b32 s18, 1, s8 // (op 509) 
s_add_i32 s19, s18, s9 // (op 509) 


s_and_b32 s9, s0, s19 // (op 509) 


s_add_i32 s0, s1, s9 // (op 509) 


s_ashr_i32 s1, s0, s8 // (op 509) 


s_lshl_b32 s0, s13, 24 // (op 509) 
s_ashr_i32 s8, s0, 31 // (op 509) 

s_xor_b32 s0, s1, s8 // (op 509) 

s_sub_i32 s1, s0, s8 // (op 509) 


s_mul_hi_i32 s0, s1, s14 // (op 509) 
s_add_i32 s8, s0, s1 // (op 509) 

s_ashr_i32 s0, s8, 31 // (op 509) 
s_cmp_eq_i32 s14, 0 // (op 509) 
s_mov_b32 s9, scc // (op 509) 
s_cmp_eq_i32 s9, 0 // (op 509) 
s_cselect_b32 s18, 0, -1 // (op 509) 

s_and_b32 s9, s15, 31 // (op 509) 
s_lshl_b32 s19, 1, s9 // (op 509) 
s_add_i32 s49, s19, s18 // (op 509) 


s_and_b32 s18, s0, s49 // (op 509) 


s_add_i32 s0, s8, s18 // (op 509) 


s_ashr_i32 s8, s0, s9 // (op 509) 


s_lshl_b32 s0, s15, 24 // (op 509) 
s_ashr_i32 s9, s0, 31 // (op 509) 

s_xor_b32 s0, s8, s9 // (op 509) 

s_sub_i32 s8, s0, s9 // (op 509) 


s_add_u32 s18, s4, 32 // (op 509) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 509) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 509) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 509) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 509) 

s_mul_i32 s0, s8, s18 // (op 509) 

s_sub_i32 s8, s1, s0 // (op 509) 


s_mul_hi_i32 s9, s16, s11 // (op 509) 
s_add_i32 s0, s9, s16 // (op 509) 

s_ashr_i32 s1, s0, 31 // (op 509) 
s_cmp_eq_i32 s11, 0 // (op 509) 
s_mov_b32 s9, scc // (op 509) 
s_cmp_eq_i32 s9, 0 // (op 509) 
s_cselect_b32 s49, 0, -1 // (op 509) 

s_and_b32 s9, s46, 31 // (op 509) 
s_lshl_b32 s50, 1, s9 // (op 509) 
s_add_i32 s51, s50, s49 // (op 509) 


s_and_b32 s49, s1, s51 // (op 509) 


s_add_i32 s1, s0, s49 // (op 509) 


s_ashr_i32 s0, s1, s9 // (op 509) 


s_lshl_b32 s1, s46, 24 // (op 509) 
s_ashr_i32 s9, s1, 31 // (op 509) 

s_xor_b32 s1, s0, s9 // (op 509) 

s_sub_i32 s0, s1, s9 // (op 509) 


s_mul_i32 s1, s45, s18 // (op 509) 

s_mul_i32 s9, s10, s1 // (op 509) 

s_mul_i32 s1, s0, s9 // (op 509) 

s_sub_u32 s9, s16, s1 // (op 509) 

s_mul_hi_i32 s1, s9, s44 // (op 509) 
s_add_i32 s18, s1, s9 // (op 509) 


s_ashr_i32 s1, s18, 31 // (op 509) 
s_cmp_eq_i32 s44, 0 // (op 509) 
s_mov_b32 s9, scc // (op 509) 
s_cmp_eq_i32 s9, 0 // (op 509) 
s_cselect_b32 s19, 0, -1 // (op 509) 

s_and_b32 s9, s48, 31 // (op 509) 
s_lshl_b32 s49, 1, s9 // (op 509) 
s_add_i32 s50, s49, s19 // (op 509) 


s_and_b32 s19, s1, s50 // (op 509) 


s_add_i32 s1, s18, s19 // (op 509) 


s_ashr_i32 s18, s1, s9 // (op 509) 


s_lshl_b32 s1, s48, 24 // (op 509) 
s_ashr_i32 s9, s1, 31 // (op 509) 

s_xor_b32 s1, s18, s9 // (op 509) 

s_sub_i32 s18, s1, s9 // (op 509) 


s_cmp_eq_i32 s0, 0 // (op 509) 
s_mov_b32 s1, scc // (op 509) 

s_cmp_eq_i32 s1, 0 // (op 509) 
s_cselect_b32 s0, s18, s8 // (op 509) 



v_lshl_add_u32 v11, s0, 5, v10 // (op 509) 


v_mul_lo_u32 v15, v11, s23 // (op 509) most significant: low of low * high
v_mul_hi_u32 v10, v11, s22 // (op 509) most significant: high of low * low
v_mul_lo_u32 v14, v11, s22 // (op 509) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 509) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 509) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 509) most significant half



s_mul_i32 s1, s22, s5 // (op 511) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 511) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 511) least significant
s_mul_hi_u32 s9, s22, s4 // (op 511) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 511) most significant: sum
s_add_u32 s1, s1, s9 // (op 511) most significant: sum


s_mov_b32 s56, s24 // (op 511) reg expression
s_mov_b32 s57, s25 // (op 511) reg expression
s_mov_b32 s58, s0 // (op 511) reg expression
s_mov_b32 s59, 131072 // (op 511) call()

s_mul_hi_i32 s0, s16, s12 // (op 528) 
v_lshrrev_b32 v12, 3, v1 // (op 528) 
s_add_i32 s1, s0, s16 // (op 528) 

v_and_b32 v13, -8, v1 // (op 528) 
s_ashr_i32 s0, s1, 31 // (op 528) 
v_sub_u32 v14, v1, v13 // (op 528) 

s_cmp_eq_i32 s12, 0 // (op 528) 
s_mov_b32 s8, scc // (op 528) 
v_lshlrev_b32 v13, 4, v14 // (op 528) 

s_cmp_eq_i32 s8, 0 // (op 528) 
s_cselect_b32 s9, 0, -1 // (op 528) 

v_mov_b32 v15, 0 // (op 528) convert
v_mov_b32 v14, v13 // (op 528) convert

s_and_b32 s8, s13, 31 // (op 528) 
s_lshl_b32 s18, 1, s8 // (op 528) 
s_add_i32 s19, s18, s9 // (op 528) 


s_and_b32 s9, s0, s19 // (op 528) 


s_add_i32 s0, s1, s9 // (op 528) 


s_ashr_i32 s1, s0, s8 // (op 528) 


s_lshl_b32 s0, s13, 24 // (op 528) 
s_ashr_i32 s8, s0, 31 // (op 528) 

s_xor_b32 s0, s1, s8 // (op 528) 

s_sub_i32 s1, s0, s8 // (op 528) 


s_mul_hi_i32 s0, s1, s14 // (op 528) 
s_add_i32 s8, s0, s1 // (op 528) 

s_ashr_i32 s0, s8, 31 // (op 528) 
s_cmp_eq_i32 s14, 0 // (op 528) 
s_mov_b32 s9, scc // (op 528) 
s_cmp_eq_i32 s9, 0 // (op 528) 
s_cselect_b32 s18, 0, -1 // (op 528) 

s_and_b32 s9, s15, 31 // (op 528) 
s_lshl_b32 s19, 1, s9 // (op 528) 
s_add_i32 s49, s19, s18 // (op 528) 


s_and_b32 s18, s0, s49 // (op 528) 


s_add_i32 s0, s8, s18 // (op 528) 


s_ashr_i32 s8, s0, s9 // (op 528) 


s_lshl_b32 s0, s15, 24 // (op 528) 
s_ashr_i32 s9, s0, 31 // (op 528) 

s_xor_b32 s0, s8, s9 // (op 528) 

s_sub_i32 s8, s0, s9 // (op 528) 


s_add_u32 s18, s4, 32 // (op 528) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 528) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 528) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 528) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 528) 

s_mul_i32 s0, s8, s18 // (op 528) 

s_sub_i32 s8, s1, s0 // (op 528) 


s_mul_hi_i32 s9, s16, s11 // (op 528) 
s_add_i32 s0, s9, s16 // (op 528) 

s_ashr_i32 s1, s0, 31 // (op 528) 
s_cmp_eq_i32 s11, 0 // (op 528) 
s_mov_b32 s9, scc // (op 528) 
s_cmp_eq_i32 s9, 0 // (op 528) 
s_cselect_b32 s49, 0, -1 // (op 528) 

s_and_b32 s9, s46, 31 // (op 528) 
s_lshl_b32 s50, 1, s9 // (op 528) 
s_add_i32 s51, s50, s49 // (op 528) 


s_and_b32 s49, s1, s51 // (op 528) 


s_add_i32 s1, s0, s49 // (op 528) 


s_ashr_i32 s0, s1, s9 // (op 528) 


s_lshl_b32 s1, s46, 24 // (op 528) 
s_ashr_i32 s9, s1, 31 // (op 528) 

s_xor_b32 s1, s0, s9 // (op 528) 

s_sub_i32 s0, s1, s9 // (op 528) 


s_mul_i32 s1, s45, s18 // (op 528) 

s_mul_i32 s9, s10, s1 // (op 528) 

s_mul_i32 s1, s0, s9 // (op 528) 

s_sub_u32 s9, s16, s1 // (op 528) 

s_mul_hi_i32 s1, s9, s44 // (op 528) 
s_add_i32 s18, s1, s9 // (op 528) 


s_ashr_i32 s1, s18, 31 // (op 528) 
s_cmp_eq_i32 s44, 0 // (op 528) 
s_mov_b32 s9, scc // (op 528) 
s_cmp_eq_i32 s9, 0 // (op 528) 
s_cselect_b32 s19, 0, -1 // (op 528) 

s_and_b32 s9, s48, 31 // (op 528) 
s_lshl_b32 s49, 1, s9 // (op 528) 
s_add_i32 s50, s49, s19 // (op 528) 


s_and_b32 s19, s1, s50 // (op 528) 


s_add_i32 s1, s18, s19 // (op 528) 


s_ashr_i32 s18, s1, s9 // (op 528) 


s_lshl_b32 s1, s48, 24 // (op 528) 
s_ashr_i32 s9, s1, 31 // (op 528) 

s_xor_b32 s1, s18, s9 // (op 528) 

s_sub_i32 s18, s1, s9 // (op 528) 


s_cmp_eq_i32 s0, 0 // (op 528) 
s_mov_b32 s1, scc // (op 528) 

s_cmp_eq_i32 s1, 0 // (op 528) 
s_cselect_b32 s0, s18, s8 // (op 528) 



v_lshl_add_u32 v13, s0, 5, v12 // (op 528) 


v_mul_lo_u32 v17, v13, s23 // (op 528) most significant: low of low * high
v_mul_hi_u32 v12, v13, s22 // (op 528) most significant: high of low * low
v_mul_lo_u32 v16, v13, s22 // (op 528) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 528) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 528) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 528) most significant half



s_mul_i32 s1, s22, s5 // (op 530) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 530) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 530) least significant
s_mul_hi_u32 s9, s22, s4 // (op 530) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 530) most significant: sum
s_add_u32 s1, s1, s9 // (op 530) most significant: sum


s_mov_b32 s56, s24 // (op 530) reg expression
s_mov_b32 s57, s25 // (op 530) reg expression
s_mov_b32 s58, s0 // (op 530) reg expression
s_mov_b32 s59, 131072 // (op 530) call()

s_mul_hi_i32 s0, s16, s12 // (op 573) 
v_lshrrev_b32 v14, 7, v1 // (op 573) 
s_add_i32 s1, s0, s16 // (op 573) 

v_and_b32 v15, 15, v1 // (op 573) 
s_ashr_i32 s0, s1, 31 // (op 573) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 573) 


s_cmp_eq_i32 s12, 0 // (op 573) 
s_mov_b32 s8, scc // (op 573) 
v_and_b32 v14, 63, v1 // (op 573) 
s_cmp_eq_i32 s8, 0 // (op 573) 
s_cselect_b32 s9, 0, -1 // (op 573) 

v_lshrrev_b32 v15, 4, v14 // (op 573) 

s_and_b32 s8, s13, 31 // (op 573) 
s_lshl_b32 s18, 1, s8 // (op 573) 
s_add_i32 s19, s18, s9 // (op 573) 


s_and_b32 s9, s0, s19 // (op 573) 


s_add_i32 s0, s1, s9 // (op 573) 


s_ashr_i32 s1, s0, s8 // (op 573) 


s_lshl_b32 s0, s13, 24 // (op 573) 
s_ashr_i32 s8, s0, 31 // (op 573) 

s_xor_b32 s0, s1, s8 // (op 573) 

s_sub_i32 s1, s0, s8 // (op 573) 


s_mul_hi_i32 s0, s1, s14 // (op 573) 
s_add_i32 s8, s0, s1 // (op 573) 

s_ashr_i32 s0, s8, 31 // (op 573) 
s_cmp_eq_i32 s14, 0 // (op 573) 
s_mov_b32 s9, scc // (op 573) 
s_cmp_eq_i32 s9, 0 // (op 573) 
s_cselect_b32 s18, 0, -1 // (op 573) 

s_and_b32 s9, s15, 31 // (op 573) 
s_lshl_b32 s19, 1, s9 // (op 573) 
s_add_i32 s49, s19, s18 // (op 573) 


s_and_b32 s18, s0, s49 // (op 573) 


s_add_i32 s0, s8, s18 // (op 573) 


s_ashr_i32 s8, s0, s9 // (op 573) 


s_lshl_b32 s0, s15, 24 // (op 573) 
s_ashr_i32 s9, s0, 31 // (op 573) 

s_xor_b32 s0, s8, s9 // (op 573) 

s_sub_i32 s8, s0, s9 // (op 573) 


s_mul_i32 s0, s1, s45 // (op 573) 

s_sub_u32 s1, s16, s0 // (op 573) 

s_mul_i32 s0, s8, s45 // (op 573) 
s_add_u32 s0, s0, s1 // (op 573) 


s_mul_hi_i32 s1, s16, s11 // (op 573) 
s_add_i32 s8, s1, s16 // (op 573) 

s_ashr_i32 s1, s8, 31 // (op 573) 
s_cmp_eq_i32 s11, 0 // (op 573) 
s_mov_b32 s9, scc // (op 573) 
s_cmp_eq_i32 s9, 0 // (op 573) 
s_cselect_b32 s18, 0, -1 // (op 573) 

s_and_b32 s9, s46, 31 // (op 573) 
s_lshl_b32 s19, 1, s9 // (op 573) 
s_add_i32 s49, s19, s18 // (op 573) 


s_and_b32 s18, s1, s49 // (op 573) 


s_add_i32 s1, s8, s18 // (op 573) 


s_ashr_i32 s8, s1, s9 // (op 573) 


s_lshl_b32 s1, s46, 24 // (op 573) 
s_ashr_i32 s9, s1, 31 // (op 573) 

s_xor_b32 s1, s8, s9 // (op 573) 

s_sub_i32 s8, s1, s9 // (op 573) 


s_add_u32 s18, s4, 32 // (op 573) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 573) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 573) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 573) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 573) 

s_mul_i32 s1, s45, s18 // (op 573) 

s_mul_i32 s9, s10, s1 // (op 573) 

s_mul_i32 s1, s8, s9 // (op 573) 

s_sub_u32 s9, s16, s1 // (op 573) 

s_mov_b32 s1, s9 // (op 573) 
s_mul_hi_i32 s18, s1, s44 // (op 573) 
s_add_i32 s19, s18, s1 // (op 573) 


s_ashr_i32 s1, s19, 31 // (op 573) 
s_cmp_eq_i32 s44, 0 // (op 573) 
s_mov_b32 s18, scc // (op 573) 
s_cmp_eq_i32 s18, 0 // (op 573) 
s_cselect_b32 s49, 0, -1 // (op 573) 

s_and_b32 s18, s48, 31 // (op 573) 
s_lshl_b32 s50, 1, s18 // (op 573) 
s_add_i32 s51, s50, s49 // (op 573) 


s_and_b32 s49, s1, s51 // (op 573) 


s_add_i32 s1, s19, s49 // (op 573) 


s_ashr_i32 s19, s1, s18 // (op 573) 


s_lshl_b32 s1, s48, 24 // (op 573) 
s_ashr_i32 s18, s1, 31 // (op 573) 

s_xor_b32 s1, s19, s18 // (op 573) 

s_sub_i32 s19, s1, s18 // (op 573) 


s_mul_i32 s1, s47, s45 // (op 573) 
s_add_u32 s50, s6, 32 // (op 573) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 573) End of Int64 add, unlocking SCC
s_sub_u32 s60, s50, 1 // (op 573) least significant half; sets scc
s_subb_u32 s61, s51, 0 // (op 573) most significant half; uses scc

s_ashr_i64 s[50:51], s[60:61], 5 // (op 573) 

s_sub_i32 s18, s50, s1 // (op 573) 

s_mul_i32 s49, s19, s18 // (op 573) 


s_sub_u32 s18, s9, s49 // (op 573) 


s_add_u32 s9, s18, s1 // (op 573) 


s_cmp_eq_i32 s8, 0 // (op 573) 
s_mov_b32 s1, scc // (op 573) 

s_cmp_eq_i32 s1, 0 // (op 573) 
s_cselect_b32 s8, s9, s0 // (op 573) 



v_lshl_add_u32 v14, s8, 5, v16 // (op 573) 


v_mul_lo_u32 v16, v14, s26 // (op 573) 
v_add_u32 v16, v16, v15 // (op 573) 


s_mul_i32 s1, s26, s7 // (op 575) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 575) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 575) least significant
s_mul_hi_u32 s9, s26, s6 // (op 575) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 575) most significant: sum
s_add_u32 s1, s1, s9 // (op 575) most significant: sum


s_mov_b32 s60, s28 // (op 575) reg expression
s_mov_b32 s61, s29 // (op 575) reg expression
s_mov_b32 s62, s0 // (op 575) reg expression
s_mov_b32 s63, 131072 // (op 575) call()

s_mul_hi_i32 s0, s16, s12 // (op 592) 
v_lshrrev_b32 v14, 7, v1 // (op 592) 
s_add_i32 s1, s0, s16 // (op 592) 

v_and_b32 v15, 15, v1 // (op 592) 
s_ashr_i32 s0, s1, 31 // (op 592) 
v_lshl_add_u32 v17, v14, 4, v15 // (op 592) 


s_cmp_eq_i32 s12, 0 // (op 592) 
s_mov_b32 s8, scc // (op 592) 
v_and_b32 v14, 63, v1 // (op 592) 
s_cmp_eq_i32 s8, 0 // (op 592) 
s_cselect_b32 s9, 0, -1 // (op 592) 

v_lshrrev_b32 v15, 4, v14 // (op 592) 

s_and_b32 s8, s13, 31 // (op 592) 
s_lshl_b32 s18, 1, s8 // (op 592) 
s_add_i32 s19, s18, s9 // (op 592) 


s_and_b32 s9, s0, s19 // (op 592) 


s_add_i32 s0, s1, s9 // (op 592) 


s_ashr_i32 s1, s0, s8 // (op 592) 


s_lshl_b32 s0, s13, 24 // (op 592) 
s_ashr_i32 s8, s0, 31 // (op 592) 

s_xor_b32 s0, s1, s8 // (op 592) 

s_sub_i32 s1, s0, s8 // (op 592) 


s_mul_hi_i32 s0, s1, s14 // (op 592) 
s_add_i32 s8, s0, s1 // (op 592) 

s_ashr_i32 s0, s8, 31 // (op 592) 
s_cmp_eq_i32 s14, 0 // (op 592) 
s_mov_b32 s9, scc // (op 592) 
s_cmp_eq_i32 s9, 0 // (op 592) 
s_cselect_b32 s18, 0, -1 // (op 592) 

s_and_b32 s9, s15, 31 // (op 592) 
s_lshl_b32 s19, 1, s9 // (op 592) 
s_add_i32 s49, s19, s18 // (op 592) 


s_and_b32 s18, s0, s49 // (op 592) 


s_add_i32 s0, s8, s18 // (op 592) 


s_ashr_i32 s8, s0, s9 // (op 592) 


s_lshl_b32 s0, s15, 24 // (op 592) 
s_ashr_i32 s9, s0, 31 // (op 592) 

s_xor_b32 s0, s8, s9 // (op 592) 

s_sub_i32 s8, s0, s9 // (op 592) 


s_mul_i32 s0, s1, s45 // (op 592) 

s_sub_u32 s1, s16, s0 // (op 592) 

s_mul_i32 s0, s8, s45 // (op 592) 
s_add_u32 s0, s0, s1 // (op 592) 


s_mul_hi_i32 s1, s16, s11 // (op 592) 
s_add_i32 s8, s1, s16 // (op 592) 

s_ashr_i32 s1, s8, 31 // (op 592) 
s_cmp_eq_i32 s11, 0 // (op 592) 
s_mov_b32 s9, scc // (op 592) 
s_cmp_eq_i32 s9, 0 // (op 592) 
s_cselect_b32 s18, 0, -1 // (op 592) 

s_and_b32 s9, s46, 31 // (op 592) 
s_lshl_b32 s19, 1, s9 // (op 592) 
s_add_i32 s49, s19, s18 // (op 592) 


s_and_b32 s18, s1, s49 // (op 592) 


s_add_i32 s1, s8, s18 // (op 592) 


s_ashr_i32 s8, s1, s9 // (op 592) 


s_lshl_b32 s1, s46, 24 // (op 592) 
s_ashr_i32 s9, s1, 31 // (op 592) 

s_xor_b32 s1, s8, s9 // (op 592) 

s_sub_i32 s8, s1, s9 // (op 592) 


s_add_u32 s18, s4, 32 // (op 592) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 592) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 592) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 592) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 592) 

s_mul_i32 s1, s45, s18 // (op 592) 

s_mul_i32 s9, s10, s1 // (op 592) 

s_mul_i32 s1, s8, s9 // (op 592) 

s_sub_u32 s9, s16, s1 // (op 592) 

s_mov_b32 s1, s9 // (op 592) 
s_mul_hi_i32 s18, s1, s44 // (op 592) 
s_add_i32 s19, s18, s1 // (op 592) 


s_ashr_i32 s1, s19, 31 // (op 592) 
s_cmp_eq_i32 s44, 0 // (op 592) 
s_mov_b32 s18, scc // (op 592) 
s_cmp_eq_i32 s18, 0 // (op 592) 
s_cselect_b32 s49, 0, -1 // (op 592) 

s_and_b32 s18, s48, 31 // (op 592) 
s_lshl_b32 s50, 1, s18 // (op 592) 
s_add_i32 s51, s50, s49 // (op 592) 


s_and_b32 s49, s1, s51 // (op 592) 


s_add_i32 s1, s19, s49 // (op 592) 


s_ashr_i32 s19, s1, s18 // (op 592) 


s_lshl_b32 s1, s48, 24 // (op 592) 
s_ashr_i32 s18, s1, 31 // (op 592) 

s_xor_b32 s1, s19, s18 // (op 592) 

s_sub_i32 s19, s1, s18 // (op 592) 


s_mul_i32 s1, s47, s45 // (op 592) 
s_add_u32 s50, s6, 32 // (op 592) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 592) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 592) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 592) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 592) 

s_sub_i32 s18, s50, s1 // (op 592) 

s_mul_i32 s49, s19, s18 // (op 592) 


s_sub_u32 s18, s9, s49 // (op 592) 


s_add_u32 s9, s18, s1 // (op 592) 


s_cmp_eq_i32 s8, 0 // (op 592) 
s_mov_b32 s1, scc // (op 592) 

s_cmp_eq_i32 s1, 0 // (op 592) 
s_cselect_b32 s8, s9, s0 // (op 592) 



v_lshl_add_u32 v14, s8, 5, v17 // (op 592) 


v_mul_lo_u32 v17, v14, s26 // (op 592) 
v_add_u32 v17, v17, v15 // (op 592) 


s_mul_i32 s1, s26, s7 // (op 594) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 594) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 594) least significant
s_mul_hi_u32 s9, s26, s6 // (op 594) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 594) most significant: sum
s_add_u32 s1, s1, s9 // (op 594) most significant: sum


s_mov_b32 s60, s28 // (op 594) reg expression
s_mov_b32 s61, s29 // (op 594) reg expression
s_mov_b32 s62, s0 // (op 594) reg expression
s_mov_b32 s63, 131072 // (op 594) call()

s_mul_hi_i32 s0, s16, s12 // (op 637) 
v_lshrrev_b32 v14, 6, v1 // (op 637) 
s_add_i32 s1, s0, s16 // (op 637) 

v_and_b32 v15, 1, v14 // (op 637) 

s_ashr_i32 s0, s1, 31 // (op 637) 
v_and_b32 v14, 15, v1 // (op 637) 
s_cmp_eq_i32 s12, 0 // (op 637) 
s_mov_b32 s8, scc // (op 637) 
v_lshl_add_u32 v18, v15, 4, v14 // (op 637) 


s_cmp_eq_i32 s8, 0 // (op 637) 
s_cselect_b32 s9, 0, -1 // (op 637) 

v_and_b32 v14, 63, v1 // (op 637) 
s_and_b32 s8, s13, 31 // (op 637) 
v_lshrrev_b32 v15, 4, v14 // (op 637) 

s_lshl_b32 s18, 1, s8 // (op 637) 
s_add_i32 s19, s18, s9 // (op 637) 


s_and_b32 s9, s0, s19 // (op 637) 


s_add_i32 s0, s1, s9 // (op 637) 


s_ashr_i32 s1, s0, s8 // (op 637) 


s_lshl_b32 s0, s13, 24 // (op 637) 
s_ashr_i32 s8, s0, 31 // (op 637) 

s_xor_b32 s0, s1, s8 // (op 637) 

s_sub_i32 s1, s0, s8 // (op 637) 


s_mul_hi_i32 s0, s1, s14 // (op 637) 
s_add_i32 s8, s0, s1 // (op 637) 

s_ashr_i32 s0, s8, 31 // (op 637) 
s_cmp_eq_i32 s14, 0 // (op 637) 
s_mov_b32 s9, scc // (op 637) 
s_cmp_eq_i32 s9, 0 // (op 637) 
s_cselect_b32 s18, 0, -1 // (op 637) 

s_and_b32 s9, s15, 31 // (op 637) 
s_lshl_b32 s19, 1, s9 // (op 637) 
s_add_i32 s49, s19, s18 // (op 637) 


s_and_b32 s18, s0, s49 // (op 637) 


s_add_i32 s0, s8, s18 // (op 637) 


s_ashr_i32 s8, s0, s9 // (op 637) 


s_lshl_b32 s0, s15, 24 // (op 637) 
s_ashr_i32 s9, s0, 31 // (op 637) 

s_xor_b32 s0, s8, s9 // (op 637) 

s_sub_i32 s8, s0, s9 // (op 637) 


s_add_u32 s18, s4, 32 // (op 637) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 637) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 637) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 637) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 637) 

s_mul_i32 s0, s8, s18 // (op 637) 

s_sub_i32 s8, s1, s0 // (op 637) 


s_mul_hi_i32 s9, s16, s11 // (op 637) 
s_add_i32 s0, s9, s16 // (op 637) 

s_ashr_i32 s1, s0, 31 // (op 637) 
s_cmp_eq_i32 s11, 0 // (op 637) 
s_mov_b32 s9, scc // (op 637) 
s_cmp_eq_i32 s9, 0 // (op 637) 
s_cselect_b32 s49, 0, -1 // (op 637) 

s_and_b32 s9, s46, 31 // (op 637) 
s_lshl_b32 s50, 1, s9 // (op 637) 
s_add_i32 s51, s50, s49 // (op 637) 


s_and_b32 s49, s1, s51 // (op 637) 


s_add_i32 s1, s0, s49 // (op 637) 


s_ashr_i32 s0, s1, s9 // (op 637) 


s_lshl_b32 s1, s46, 24 // (op 637) 
s_ashr_i32 s9, s1, 31 // (op 637) 

s_xor_b32 s1, s0, s9 // (op 637) 

s_sub_i32 s0, s1, s9 // (op 637) 


s_mul_i32 s1, s45, s18 // (op 637) 

s_mul_i32 s9, s10, s1 // (op 637) 

s_mul_i32 s1, s0, s9 // (op 637) 

s_sub_u32 s9, s16, s1 // (op 637) 

s_mul_hi_i32 s1, s9, s44 // (op 637) 
s_add_i32 s18, s1, s9 // (op 637) 


s_ashr_i32 s1, s18, 31 // (op 637) 
s_cmp_eq_i32 s44, 0 // (op 637) 
s_mov_b32 s9, scc // (op 637) 
s_cmp_eq_i32 s9, 0 // (op 637) 
s_cselect_b32 s19, 0, -1 // (op 637) 

s_and_b32 s9, s48, 31 // (op 637) 
s_lshl_b32 s49, 1, s9 // (op 637) 
s_add_i32 s50, s49, s19 // (op 637) 


s_and_b32 s19, s1, s50 // (op 637) 


s_add_i32 s1, s18, s19 // (op 637) 


s_ashr_i32 s18, s1, s9 // (op 637) 


s_lshl_b32 s1, s48, 24 // (op 637) 
s_ashr_i32 s9, s1, 31 // (op 637) 

s_xor_b32 s1, s18, s9 // (op 637) 

s_sub_i32 s18, s1, s9 // (op 637) 


s_cmp_eq_i32 s0, 0 // (op 637) 
s_mov_b32 s1, scc // (op 637) 

s_cmp_eq_i32 s1, 0 // (op 637) 
s_cselect_b32 s0, s18, s8 // (op 637) 



v_lshl_add_u32 v14, s0, 5, v18 // (op 637) 


v_mul_lo_u32 v18, v14, s30 // (op 637) 

v_add_u32 v14, v15, v18 // (op 637) 


s_mul_i32 s1, s30, s5 // (op 639) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 639) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 639) least significant
s_mul_hi_u32 s9, s30, s4 // (op 639) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 639) most significant: sum
s_add_u32 s1, s1, s9 // (op 639) most significant: sum


s_mov_b32 s64, s32 // (op 639) reg expression
s_mov_b32 s65, s33 // (op 639) reg expression
s_mov_b32 s66, s0 // (op 639) reg expression
s_mov_b32 s67, 131072 // (op 639) call()

s_mul_hi_i32 s0, s16, s12 // (op 656) 
v_lshrrev_b32 v15, 6, v1 // (op 656) 
s_add_i32 s1, s0, s16 // (op 656) 

v_and_b32 v18, 1, v15 // (op 656) 

s_ashr_i32 s0, s1, 31 // (op 656) 
v_and_b32 v15, 15, v1 // (op 656) 
s_cmp_eq_i32 s12, 0 // (op 656) 
s_mov_b32 s8, scc // (op 656) 
v_lshl_add_u32 v19, v18, 4, v15 // (op 656) 


s_cmp_eq_i32 s8, 0 // (op 656) 
s_cselect_b32 s9, 0, -1 // (op 656) 

v_and_b32 v15, 63, v1 // (op 656) 
s_and_b32 s8, s13, 31 // (op 656) 
v_lshrrev_b32 v18, 4, v15 // (op 656) 

s_lshl_b32 s18, 1, s8 // (op 656) 
s_add_i32 s19, s18, s9 // (op 656) 


s_and_b32 s9, s0, s19 // (op 656) 


s_add_i32 s0, s1, s9 // (op 656) 


s_ashr_i32 s1, s0, s8 // (op 656) 


s_lshl_b32 s0, s13, 24 // (op 656) 
s_ashr_i32 s8, s0, 31 // (op 656) 

s_xor_b32 s0, s1, s8 // (op 656) 

s_sub_i32 s1, s0, s8 // (op 656) 


s_mul_hi_i32 s0, s1, s14 // (op 656) 
s_add_i32 s8, s0, s1 // (op 656) 

s_ashr_i32 s0, s8, 31 // (op 656) 
s_cmp_eq_i32 s14, 0 // (op 656) 
s_mov_b32 s9, scc // (op 656) 
s_cmp_eq_i32 s9, 0 // (op 656) 
s_cselect_b32 s18, 0, -1 // (op 656) 

s_and_b32 s9, s15, 31 // (op 656) 
s_lshl_b32 s19, 1, s9 // (op 656) 
s_add_i32 s49, s19, s18 // (op 656) 


s_and_b32 s18, s0, s49 // (op 656) 


s_add_i32 s0, s8, s18 // (op 656) 


s_ashr_i32 s8, s0, s9 // (op 656) 


s_lshl_b32 s0, s15, 24 // (op 656) 
s_ashr_i32 s9, s0, 31 // (op 656) 

s_xor_b32 s0, s8, s9 // (op 656) 

s_sub_i32 s8, s0, s9 // (op 656) 


s_add_u32 s18, s4, 32 // (op 656) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 656) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 656) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 656) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 656) 

s_mul_i32 s0, s8, s18 // (op 656) 

s_sub_i32 s8, s1, s0 // (op 656) 


s_mul_hi_i32 s9, s16, s11 // (op 656) 
s_add_i32 s0, s9, s16 // (op 656) 

s_ashr_i32 s1, s0, 31 // (op 656) 
s_cmp_eq_i32 s11, 0 // (op 656) 
s_mov_b32 s9, scc // (op 656) 
s_cmp_eq_i32 s9, 0 // (op 656) 
s_cselect_b32 s49, 0, -1 // (op 656) 

s_and_b32 s9, s46, 31 // (op 656) 
s_lshl_b32 s50, 1, s9 // (op 656) 
s_add_i32 s51, s50, s49 // (op 656) 


s_and_b32 s49, s1, s51 // (op 656) 


s_add_i32 s1, s0, s49 // (op 656) 


s_ashr_i32 s0, s1, s9 // (op 656) 


s_lshl_b32 s1, s46, 24 // (op 656) 
s_ashr_i32 s9, s1, 31 // (op 656) 

s_xor_b32 s1, s0, s9 // (op 656) 

s_sub_i32 s0, s1, s9 // (op 656) 


s_mul_i32 s1, s45, s18 // (op 656) 

s_mul_i32 s9, s10, s1 // (op 656) 

s_mul_i32 s1, s0, s9 // (op 656) 

s_sub_u32 s9, s16, s1 // (op 656) 

s_mul_hi_i32 s1, s9, s44 // (op 656) 
s_add_i32 s18, s1, s9 // (op 656) 


s_ashr_i32 s1, s18, 31 // (op 656) 
s_cmp_eq_i32 s44, 0 // (op 656) 
s_mov_b32 s9, scc // (op 656) 
s_cmp_eq_i32 s9, 0 // (op 656) 
s_cselect_b32 s19, 0, -1 // (op 656) 

s_and_b32 s9, s48, 31 // (op 656) 
s_lshl_b32 s49, 1, s9 // (op 656) 
s_add_i32 s50, s49, s19 // (op 656) 


s_and_b32 s19, s1, s50 // (op 656) 


s_add_i32 s1, s18, s19 // (op 656) 


s_ashr_i32 s18, s1, s9 // (op 656) 


s_lshl_b32 s1, s48, 24 // (op 656) 
s_ashr_i32 s9, s1, 31 // (op 656) 

s_xor_b32 s1, s18, s9 // (op 656) 

s_sub_i32 s18, s1, s9 // (op 656) 


s_cmp_eq_i32 s0, 0 // (op 656) 
s_mov_b32 s1, scc // (op 656) 

s_cmp_eq_i32 s1, 0 // (op 656) 
s_cselect_b32 s0, s18, s8 // (op 656) 



v_lshl_add_u32 v15, s0, 5, v19 // (op 656) 


v_mul_lo_u32 v19, v15, s30 // (op 656) 

v_add_u32 v15, v18, v19 // (op 656) 


s_mul_i32 s1, s30, s5 // (op 658) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 658) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 658) least significant
s_mul_hi_u32 s9, s30, s4 // (op 658) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 658) most significant: sum
s_add_u32 s1, s1, s9 // (op 658) most significant: sum


s_mov_b32 s64, s32 // (op 658) reg expression
s_mov_b32 s65, s33 // (op 658) reg expression
s_mov_b32 s66, s0 // (op 658) reg expression
s_mov_b32 s67, 131072 // (op 658) call()




buffer_load_ubyte v18, v16, s[60:63], 0 offen offset: 0 // (op 312) Load value



buffer_load_ubyte v19, v14, s[64:67], 0 offen offset: 0 // (op 317) Load value
v_and_b32 v20, -8, v1 // (op 775) 
v_sub_u32 v21, v1, v20 // (op 775) 

v_lshlrev_b32 v20, 4, v21 // (op 775) 

v_lshrrev_b32 v21, 3, v1 // (op 775) 
v_lshl_add_u32 v22, v21, 7, v20 // (op 775) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 775) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 401) reg expression
buffer_load_dwordx4 v10, s[56:59], 0 offen offset: 0 lds // (op 401) Load value direct to lds




buffer_load_dwordx3 v[20:22], v8, s[52:55], 0 offen // (op 324) Load value



buffer_load_ubyte v23, v16, s[60:63], 0 offen offset:4 // (op 328) Load value




buffer_load_ubyte v16, v14, s[64:67], 0 offen offset:4 // (op 333) Load value

v_and_b32 v14, -8, v1 // (op 872) 
v_sub_u32 v24, v1, v14 // (op 872) 

v_lshlrev_b32 v14, 4, v24 // (op 872) 

v_lshrrev_b32 v24, 3, v1 // (op 872) 
v_lshl_add_u32 v25, v24, 7, v14 // (op 872) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 872) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 406) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 406) reg expression
s_mov_b32 s8, 128 // (op 406) call()
buffer_load_dwordx4 v10, s[56:59], s8 offen offset: 0 lds // (op 406) Load value direct to lds







buffer_load_dwordx3 v[24:26], v8, s[52:55], 0 offen offset:96 // (op 340) Load value

v_and_b32 v8, -8, v1 // (op 723) 
v_sub_u32 v14, v1, v8 // (op 723) 

v_lshlrev_b32 v8, 4, v14 // (op 723) 

v_lshrrev_b32 v14, 3, v1 // (op 723) 
v_lshl_add_u32 v10, v14, 7, v8 // (op 723) 


v_mul_lo_u32 v8, v10, 6 // (op 723) 

v_lshrrev_b32 v14, 3, v8 // (op 723) 




s_waitcnt vmcnt(4)
ds_write_b96 v14, v[20:22] offset:8192 // (op 343) Store local data 

v_and_b32 v8, -8, v1 // (op 694) 
v_sub_u32 v14, v1, v8 // (op 694) 

v_lshlrev_b32 v8, 4, v14 // (op 694) 

v_lshrrev_b32 v14, 3, v1 // (op 694) 
v_lshl_add_u32 v10, v14, 7, v8 // (op 694) 


v_mul_lo_u32 v8, v10, 6 // (op 694) 

v_lshrrev_b32 v14, 3, v8 // (op 694) 

v_and_b32 v8, -8, v1 // (op 747) 
v_sub_u32 v10, v1, v8 // (op 747) 

v_lshlrev_b32 v8, 4, v10 // (op 747) 

v_lshrrev_b32 v10, 3, v1 // (op 747) 
v_lshl_add_u32 v11, v10, 7, v8 // (op 747) 


s_nop 0
v_readfirstlane_b32 s0, v11 // (op 747) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v8, -8, v1 // (op 833) 
v_sub_u32 v10, v1, v8 // (op 833) 

v_lshlrev_b32 v8, 4, v10 // (op 833) 

v_lshrrev_b32 v10, 3, v1 // (op 833) 
v_lshl_add_u32 v11, v10, 7, v8 // (op 833) 


v_mul_lo_u32 v8, v11, 6 // (op 833) 

v_lshrrev_b32 v10, 3, v8 // (op 833) 

v_and_b32 v8, -8, v1 // (op 857) 
v_sub_u32 v11, v1, v8 // (op 857) 

v_lshlrev_b32 v8, 4, v11 // (op 857) 

v_lshrrev_b32 v11, 3, v1 // (op 857) 
v_lshl_add_u32 v27, v11, 7, v8 // (op 857) 


s_nop 0
v_readfirstlane_b32 s1, v27 // (op 857) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s8, 0 // (op 53) call()
s_ashr_i64 s[18:19], s[34:35], 7 // (op 52) 
s_mov_b32 s9, s18 // (op 52) convert

s_and_b32 s18, s9, 4294967294 // (op 52) 

s_cmp_lt_u32 s8, s18 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_52 // (op 52) Condition: Top (jump to RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_52 if false)
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_52:
s_add_u32 s8, s8, 2 // (op 54) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 913) 



ds_read_b128 v[28:31], v3 offset:8192 // (op 62) Load local data 
ds_read_b64 v[32:33], v3 offset:8208 // (op 62) Load local data 
s_barrier  // (op 926) 



ds_read_b128 v[34:37], v4 // (op 67) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 67) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:33], v[34:41], a[0:3], v18, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 78) 




v_mov_b32 v8, v17 // (op 72) 

buffer_load_ubyte v18, v8, s[60:63], 0 offen offset:8 // (op 72) Load value


v_mov_b32 v11, v15 // (op 76) 

buffer_load_ubyte v19, v11, s[64:67], 0 offen offset:8 // (op 76) Load value
s_barrier  // (op 929) 


v_mov_b64 v[28:29], v[12:13] // (op 385) 

s_mov_b32 m0, s0 // (op 385) reg expression
s_mov_b32 s9, 256 // (op 385) call()
buffer_load_dwordx4 v28, s[56:59], s9 offen offset: 0 lds // (op 385) Load value direct to lds



v_mov_b32 v27, v9 // (op 4) 

buffer_load_dwordx3 v[20:22], v27, s[52:55], 0 offen offset:192 // (op 4) Load value
s_barrier  // (op 942) 



s_waitcnt vmcnt(4)
ds_write_b96 v10, v[24:26] offset:11264 // (op 212) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 939) 



ds_read_b128 v[30:33], v5 offset:11264 // (op 214) Load local data 
ds_read_b64 v[34:35], v5 offset:11280 // (op 214) Load local data 
s_barrier  // (op 945) 
s_barrier  // (op 995) 



ds_read_b128 v[36:39], v6 offset:4096 // (op 216) Load local data 
ds_read_b128 v[40:43], v6 offset:4160 // (op 216) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:35], v[36:43], a[0:3], v23, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 221) 





buffer_load_ubyte v23, v8, s[60:63], 0 offen offset:12 // (op 218) Load value




v_add_u32 v17, 8, v17 // (op 584) 
buffer_load_ubyte v16, v11, s[64:67], 0 offen offset:12 // (op 220) Load value

s_barrier  // (op 948) 
v_add_u32 v15, 8, v15 // (op 648) 


s_add_u32 s9, 4096, s1 // (op 396) 

s_mov_b32 m0, s9 // (op 396) reg expression
s_mov_b32 s18, 384 // (op 396) call()
buffer_load_dwordx4 v28, s[56:59], s18 offen offset: 0 lds // (op 396) Load value direct to lds






v_add_co_u32 v12, vcc, 256, v12 // (op 520) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 520) most significant half
buffer_load_dwordx3 v[24:26], v27, s[52:55], 0 offen offset:288 // (op 211) Load value

s_barrier  // (op 916) 
v_add_u32 v9, 192, v9 // (op 456) 



s_waitcnt vmcnt(4)
ds_write_b96 v14, v[20:22] offset:8192 // (op 6) Store local data 
s_ashr_i64 s[18:19], s[34:35], 7 // (op 52) 
s_mov_b32 s9, s18 // (op 52) convert

s_and_b32 s18, s9, 4294967294 // (op 52) 

s_cmp_lt_u32 s8, s18 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_52 // (op 52) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_52 if true)
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_52:












s_barrier  // (op 959) 
s_mul_hi_i32 s0, s16, s12 // (op 419) 
v_lshrrev_b32 v8, 3, v1 // (op 419) 
s_add_i32 s1, s0, s16 // (op 419) 

v_and_b32 v9, -8, v1 // (op 419) 
s_ashr_i32 s0, s1, 31 // (op 419) 
v_sub_u32 v10, v1, v9 // (op 419) 

s_cmp_eq_i32 s12, 0 // (op 419) 
s_mov_b32 s8, scc // (op 419) 
v_lshlrev_b32 v9, 4, v10 // (op 419) 

s_cmp_eq_i32 s8, 0 // (op 419) 
s_cselect_b32 s9, 0, -1 // (op 419) 

s_and_b32 s8, s13, 31 // (op 419) 
s_lshl_b32 s18, 1, s8 // (op 419) 
s_add_i32 s19, s18, s9 // (op 419) 


s_and_b32 s9, s0, s19 // (op 419) 


s_add_i32 s0, s1, s9 // (op 419) 


s_ashr_i32 s1, s0, s8 // (op 419) 


s_lshl_b32 s0, s13, 24 // (op 419) 
s_ashr_i32 s8, s0, 31 // (op 419) 

s_xor_b32 s0, s1, s8 // (op 419) 

s_sub_i32 s1, s0, s8 // (op 419) 


s_mul_hi_i32 s0, s1, s14 // (op 419) 
s_add_i32 s8, s0, s1 // (op 419) 

s_ashr_i32 s0, s8, 31 // (op 419) 
s_cmp_eq_i32 s14, 0 // (op 419) 
s_mov_b32 s9, scc // (op 419) 
s_cmp_eq_i32 s9, 0 // (op 419) 
s_cselect_b32 s18, 0, -1 // (op 419) 

s_and_b32 s9, s15, 31 // (op 419) 
s_lshl_b32 s19, 1, s9 // (op 419) 
s_add_i32 s49, s19, s18 // (op 419) 


s_and_b32 s18, s0, s49 // (op 419) 


s_add_i32 s0, s8, s18 // (op 419) 


s_ashr_i32 s8, s0, s9 // (op 419) 


s_lshl_b32 s0, s15, 24 // (op 419) 
s_ashr_i32 s9, s0, 31 // (op 419) 

s_xor_b32 s0, s8, s9 // (op 419) 

s_sub_i32 s8, s0, s9 // (op 419) 


s_mul_i32 s0, s1, s45 // (op 419) 

s_sub_u32 s1, s16, s0 // (op 419) 

s_mul_i32 s0, s8, s45 // (op 419) 
s_add_u32 s0, s0, s1 // (op 419) 


s_mul_hi_i32 s1, s16, s11 // (op 419) 
s_add_i32 s8, s1, s16 // (op 419) 

s_ashr_i32 s1, s8, 31 // (op 419) 
s_cmp_eq_i32 s11, 0 // (op 419) 
s_mov_b32 s9, scc // (op 419) 
s_cmp_eq_i32 s9, 0 // (op 419) 
s_cselect_b32 s18, 0, -1 // (op 419) 

s_and_b32 s9, s46, 31 // (op 419) 
s_lshl_b32 s19, 1, s9 // (op 419) 
s_add_i32 s49, s19, s18 // (op 419) 


s_and_b32 s18, s1, s49 // (op 419) 


s_add_i32 s1, s8, s18 // (op 419) 


s_ashr_i32 s8, s1, s9 // (op 419) 


s_lshl_b32 s1, s46, 24 // (op 419) 
s_ashr_i32 s9, s1, 31 // (op 419) 

s_xor_b32 s1, s8, s9 // (op 419) 

s_sub_i32 s8, s1, s9 // (op 419) 


s_add_u32 s18, s4, 32 // (op 419) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 419) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 419) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 419) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 419) 

s_mul_i32 s1, s45, s18 // (op 419) 

s_mul_i32 s9, s10, s1 // (op 419) 

s_mul_i32 s1, s8, s9 // (op 419) 

s_sub_u32 s9, s16, s1 // (op 419) 

s_mov_b32 s1, s9 // (op 419) 
s_mul_hi_i32 s18, s1, s44 // (op 419) 
s_add_i32 s19, s18, s1 // (op 419) 


s_ashr_i32 s1, s19, 31 // (op 419) 
s_cmp_eq_i32 s44, 0 // (op 419) 
s_mov_b32 s18, scc // (op 419) 
s_cmp_eq_i32 s18, 0 // (op 419) 
s_cselect_b32 s49, 0, -1 // (op 419) 

s_and_b32 s18, s48, 31 // (op 419) 
s_lshl_b32 s50, 1, s18 // (op 419) 
s_add_i32 s51, s50, s49 // (op 419) 


s_and_b32 s49, s1, s51 // (op 419) 


s_add_i32 s1, s19, s49 // (op 419) 


s_ashr_i32 s19, s1, s18 // (op 419) 


s_lshl_b32 s1, s48, 24 // (op 419) 
s_ashr_i32 s18, s1, 31 // (op 419) 

s_xor_b32 s1, s19, s18 // (op 419) 

s_sub_i32 s19, s1, s18 // (op 419) 


s_mul_i32 s1, s47, s45 // (op 419) 
s_add_u32 s50, s6, 32 // (op 419) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 419) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 419) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 419) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 419) 

s_sub_i32 s18, s50, s1 // (op 419) 

s_mul_i32 s49, s19, s18 // (op 419) 


s_sub_u32 s18, s9, s49 // (op 419) 


s_add_u32 s9, s18, s1 // (op 419) 


s_cmp_eq_i32 s8, 0 // (op 419) 
s_mov_b32 s1, scc // (op 419) 

s_cmp_eq_i32 s1, 0 // (op 419) 
s_cselect_b32 s8, s9, s0 // (op 419) 



v_lshl_add_u32 v10, s8, 5, v8 // (op 419) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 419) 
s_lshr_b32 s8, s0, 1 // (op 419) 

v_lshl_add_u32 v8, s8, 8, v9 // (op 419) 


v_mov_b32 v15, 0 // (op 419) convert
v_mov_b32 v14, v8 // (op 419) convert

v_mul_lo_u32 v9, v10, s3 // (op 419) most significant: low of low * high
v_mul_hi_u32 v11, v10, s2 // (op 419) most significant: high of low * low
v_mul_lo_u32 v8, v10, s2 // (op 419) least significant: low of low * low
v_add_u32 v9, v9, v11 // (op 419) most significant: sum

v_add_co_u32 v8, s[0:1], v8, v14 // (op 419) least significant half
v_addc_co_u32 v9, s[0:1], v9, v15, s[0:1] // (op 419) most significant half



v_mul_lo_u32 v11, v9, 6 // (op 419) most significant: low of high * low
v_mul_hi_u32 v12, v8, 6 // (op 419) most significant: high of low * low
v_mul_lo_u32 v10, v8, 6 // (op 419) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 419) most significant: sum


v_lshrrev_b64 v[8:9], 3, v[10:11] // (op 419) 

v_mov_b32 v10, v8 // (op 419) convert

s_mul_i32 s1, s7, 6 // (op 421) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 421) least significant
s_mul_hi_u32 s8, s6, 6 // (op 421) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 421) most significant: sum

s_mul_i32 s9, s2, s1 // (op 421) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 421) most significant: low of high * low
s_mul_i32 s8, s2, s0 // (op 421) least significant
s_mul_hi_u32 s19, s2, s0 // (op 421) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 421) most significant: sum
s_add_u32 s9, s9, s19 // (op 421) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 421) 

s_mov_b32 s52, s20 // (op 421) reg expression
s_mov_b32 s53, s21 // (op 421) reg expression
s_mov_b32 s54, s0 // (op 421) reg expression
s_mov_b32 s55, 131072 // (op 421) call()

s_mul_hi_i32 s0, s16, s12 // (op 486) 
v_lshrrev_b32 v8, 3, v1 // (op 486) 
s_add_i32 s1, s0, s16 // (op 486) 

v_and_b32 v9, -8, v1 // (op 486) 
s_ashr_i32 s0, s1, 31 // (op 486) 
v_sub_u32 v11, v1, v9 // (op 486) 

s_cmp_eq_i32 s12, 0 // (op 486) 
s_mov_b32 s8, scc // (op 486) 
v_lshlrev_b32 v9, 4, v11 // (op 486) 

s_cmp_eq_i32 s8, 0 // (op 486) 
s_cselect_b32 s9, 0, -1 // (op 486) 

s_and_b32 s8, s13, 31 // (op 486) 
s_lshl_b32 s18, 1, s8 // (op 486) 
s_add_i32 s19, s18, s9 // (op 486) 


s_and_b32 s9, s0, s19 // (op 486) 


s_add_i32 s0, s1, s9 // (op 486) 


s_ashr_i32 s1, s0, s8 // (op 486) 


s_lshl_b32 s0, s13, 24 // (op 486) 
s_ashr_i32 s8, s0, 31 // (op 486) 

s_xor_b32 s0, s1, s8 // (op 486) 

s_sub_i32 s1, s0, s8 // (op 486) 


s_mul_hi_i32 s0, s1, s14 // (op 486) 
s_add_i32 s8, s0, s1 // (op 486) 

s_ashr_i32 s0, s8, 31 // (op 486) 
s_cmp_eq_i32 s14, 0 // (op 486) 
s_mov_b32 s9, scc // (op 486) 
s_cmp_eq_i32 s9, 0 // (op 486) 
s_cselect_b32 s18, 0, -1 // (op 486) 

s_and_b32 s9, s15, 31 // (op 486) 
s_lshl_b32 s19, 1, s9 // (op 486) 
s_add_i32 s49, s19, s18 // (op 486) 


s_and_b32 s18, s0, s49 // (op 486) 


s_add_i32 s0, s8, s18 // (op 486) 


s_ashr_i32 s8, s0, s9 // (op 486) 


s_lshl_b32 s0, s15, 24 // (op 486) 
s_ashr_i32 s9, s0, 31 // (op 486) 

s_xor_b32 s0, s8, s9 // (op 486) 

s_sub_i32 s8, s0, s9 // (op 486) 


s_add_u32 s18, s4, 32 // (op 486) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 486) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 486) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 486) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 486) 

s_mul_i32 s0, s8, s18 // (op 486) 

s_sub_i32 s8, s1, s0 // (op 486) 


s_mul_hi_i32 s9, s16, s11 // (op 486) 
s_add_i32 s0, s9, s16 // (op 486) 

s_ashr_i32 s1, s0, 31 // (op 486) 
s_cmp_eq_i32 s11, 0 // (op 486) 
s_mov_b32 s9, scc // (op 486) 
s_cmp_eq_i32 s9, 0 // (op 486) 
s_cselect_b32 s49, 0, -1 // (op 486) 

s_and_b32 s9, s46, 31 // (op 486) 
s_lshl_b32 s50, 1, s9 // (op 486) 
s_add_i32 s51, s50, s49 // (op 486) 


s_and_b32 s49, s1, s51 // (op 486) 


s_add_i32 s1, s0, s49 // (op 486) 


s_ashr_i32 s0, s1, s9 // (op 486) 


s_lshl_b32 s1, s46, 24 // (op 486) 
s_ashr_i32 s9, s1, 31 // (op 486) 

s_xor_b32 s1, s0, s9 // (op 486) 

s_sub_i32 s0, s1, s9 // (op 486) 


s_mul_i32 s1, s45, s18 // (op 486) 

s_mul_i32 s9, s10, s1 // (op 486) 

s_mul_i32 s1, s0, s9 // (op 486) 

s_sub_u32 s9, s16, s1 // (op 486) 

s_mul_hi_i32 s1, s9, s44 // (op 486) 
s_add_i32 s18, s1, s9 // (op 486) 


s_ashr_i32 s1, s18, 31 // (op 486) 
s_cmp_eq_i32 s44, 0 // (op 486) 
s_mov_b32 s9, scc // (op 486) 
s_cmp_eq_i32 s9, 0 // (op 486) 
s_cselect_b32 s19, 0, -1 // (op 486) 

s_and_b32 s9, s48, 31 // (op 486) 
s_lshl_b32 s49, 1, s9 // (op 486) 
s_add_i32 s50, s49, s19 // (op 486) 


s_and_b32 s19, s1, s50 // (op 486) 


s_add_i32 s1, s18, s19 // (op 486) 


s_ashr_i32 s18, s1, s9 // (op 486) 


s_lshl_b32 s1, s48, 24 // (op 486) 
s_ashr_i32 s9, s1, 31 // (op 486) 

s_xor_b32 s1, s18, s9 // (op 486) 

s_sub_i32 s18, s1, s9 // (op 486) 


s_cmp_eq_i32 s0, 0 // (op 486) 
s_mov_b32 s1, scc // (op 486) 

s_cmp_eq_i32 s1, 0 // (op 486) 
s_cselect_b32 s0, s18, s8 // (op 486) 



v_lshl_add_u32 v11, s0, 5, v8 // (op 486) 


v_mul_lo_u32 v13, v11, s23 // (op 486) most significant: low of low * high
v_mul_hi_u32 v8, v11, s22 // (op 486) most significant: high of low * low
v_mul_lo_u32 v12, v11, s22 // (op 486) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 486) most significant: sum


s_ashr_i64 s[0:1], s[34:35], 7 // (op 486) 
s_lshr_b32 s8, s0, 1 // (op 486) 

v_lshl_add_u32 v8, s8, 8, v9 // (op 486) 


v_mov_b32 v15, 0 // (op 486) convert
v_mov_b32 v14, v8 // (op 486) convert

v_add_co_u32 v8, s[0:1], v14, v12 // (op 486) least significant half
v_addc_co_u32 v9, s[0:1], v15, v13, s[0:1] // (op 486) most significant half



s_mul_i32 s1, s22, s5 // (op 488) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 488) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 488) least significant
s_mul_hi_u32 s9, s22, s4 // (op 488) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 488) most significant: sum
s_add_u32 s1, s1, s9 // (op 488) most significant: sum


s_mov_b32 s56, s24 // (op 488) reg expression
s_mov_b32 s57, s25 // (op 488) reg expression
s_mov_b32 s58, s0 // (op 488) reg expression
s_mov_b32 s59, 131072 // (op 488) call()

s_mul_hi_i32 s0, s16, s12 // (op 550) 
v_lshrrev_b32 v11, 7, v1 // (op 550) 
s_add_i32 s1, s0, s16 // (op 550) 

v_and_b32 v12, 15, v1 // (op 550) 
s_ashr_i32 s0, s1, 31 // (op 550) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 550) 


s_cmp_eq_i32 s12, 0 // (op 550) 
s_mov_b32 s8, scc // (op 550) 
v_and_b32 v11, 63, v1 // (op 550) 
s_cmp_eq_i32 s8, 0 // (op 550) 
s_cselect_b32 s9, 0, -1 // (op 550) 

v_lshrrev_b32 v12, 4, v11 // (op 550) 

s_and_b32 s8, s13, 31 // (op 550) 
s_lshl_b32 s18, 1, s8 // (op 550) 
s_add_i32 s19, s18, s9 // (op 550) 


s_and_b32 s9, s0, s19 // (op 550) 


s_add_i32 s0, s1, s9 // (op 550) 


s_ashr_i32 s1, s0, s8 // (op 550) 


s_lshl_b32 s0, s13, 24 // (op 550) 
s_ashr_i32 s8, s0, 31 // (op 550) 

s_xor_b32 s0, s1, s8 // (op 550) 

s_sub_i32 s1, s0, s8 // (op 550) 


s_mul_hi_i32 s0, s1, s14 // (op 550) 
s_add_i32 s8, s0, s1 // (op 550) 

s_ashr_i32 s0, s8, 31 // (op 550) 
s_cmp_eq_i32 s14, 0 // (op 550) 
s_mov_b32 s9, scc // (op 550) 
s_cmp_eq_i32 s9, 0 // (op 550) 
s_cselect_b32 s18, 0, -1 // (op 550) 

s_and_b32 s9, s15, 31 // (op 550) 
s_lshl_b32 s19, 1, s9 // (op 550) 
s_add_i32 s49, s19, s18 // (op 550) 


s_and_b32 s18, s0, s49 // (op 550) 


s_add_i32 s0, s8, s18 // (op 550) 


s_ashr_i32 s8, s0, s9 // (op 550) 


s_lshl_b32 s0, s15, 24 // (op 550) 
s_ashr_i32 s9, s0, 31 // (op 550) 

s_xor_b32 s0, s8, s9 // (op 550) 

s_sub_i32 s8, s0, s9 // (op 550) 


s_mul_i32 s0, s1, s45 // (op 550) 

s_sub_u32 s1, s16, s0 // (op 550) 

s_mul_i32 s0, s8, s45 // (op 550) 
s_add_u32 s0, s0, s1 // (op 550) 


s_mul_hi_i32 s1, s16, s11 // (op 550) 
s_add_i32 s8, s1, s16 // (op 550) 

s_ashr_i32 s1, s8, 31 // (op 550) 
s_cmp_eq_i32 s11, 0 // (op 550) 
s_mov_b32 s9, scc // (op 550) 
s_cmp_eq_i32 s9, 0 // (op 550) 
s_cselect_b32 s18, 0, -1 // (op 550) 

s_and_b32 s9, s46, 31 // (op 550) 
s_lshl_b32 s19, 1, s9 // (op 550) 
s_add_i32 s49, s19, s18 // (op 550) 


s_and_b32 s18, s1, s49 // (op 550) 


s_add_i32 s1, s8, s18 // (op 550) 


s_ashr_i32 s8, s1, s9 // (op 550) 


s_lshl_b32 s1, s46, 24 // (op 550) 
s_ashr_i32 s9, s1, 31 // (op 550) 

s_xor_b32 s1, s8, s9 // (op 550) 

s_sub_i32 s8, s1, s9 // (op 550) 


s_add_u32 s18, s4, 32 // (op 550) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 550) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 550) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 550) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 550) 

s_mul_i32 s1, s45, s18 // (op 550) 

s_mul_i32 s9, s10, s1 // (op 550) 

s_mul_i32 s1, s8, s9 // (op 550) 

s_sub_u32 s9, s16, s1 // (op 550) 

s_mov_b32 s1, s9 // (op 550) 
s_mul_hi_i32 s18, s1, s44 // (op 550) 
s_add_i32 s19, s18, s1 // (op 550) 


s_ashr_i32 s1, s19, 31 // (op 550) 
s_cmp_eq_i32 s44, 0 // (op 550) 
s_mov_b32 s18, scc // (op 550) 
s_cmp_eq_i32 s18, 0 // (op 550) 
s_cselect_b32 s49, 0, -1 // (op 550) 

s_and_b32 s18, s48, 31 // (op 550) 
s_lshl_b32 s50, 1, s18 // (op 550) 
s_add_i32 s51, s50, s49 // (op 550) 


s_and_b32 s49, s1, s51 // (op 550) 


s_add_i32 s1, s19, s49 // (op 550) 


s_ashr_i32 s19, s1, s18 // (op 550) 


s_lshl_b32 s1, s48, 24 // (op 550) 
s_ashr_i32 s18, s1, 31 // (op 550) 

s_xor_b32 s1, s19, s18 // (op 550) 

s_sub_i32 s19, s1, s18 // (op 550) 


s_mul_i32 s1, s47, s45 // (op 550) 
s_add_u32 s50, s6, 32 // (op 550) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 550) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 550) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 550) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 550) 

s_sub_i32 s18, s50, s1 // (op 550) 

s_mul_i32 s49, s19, s18 // (op 550) 


s_sub_u32 s18, s9, s49 // (op 550) 


s_add_u32 s9, s18, s1 // (op 550) 


s_cmp_eq_i32 s8, 0 // (op 550) 
s_mov_b32 s1, scc // (op 550) 

s_cmp_eq_i32 s1, 0 // (op 550) 
s_cselect_b32 s8, s9, s0 // (op 550) 



v_lshl_add_u32 v11, s8, 5, v13 // (op 550) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 550) 
s_lshr_b32 s8, s0, 1 // (op 550) 

v_lshl_add_u32 v13, s8, 3, v12 // (op 550) 


v_mul_lo_u32 v12, v11, s26 // (op 550) 
v_add_u32 v12, v12, v13 // (op 550) 


s_mul_i32 s1, s26, s7 // (op 552) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 552) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 552) least significant
s_mul_hi_u32 s9, s26, s6 // (op 552) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 552) most significant: sum
s_add_u32 s1, s1, s9 // (op 552) most significant: sum


s_mov_b32 s60, s28 // (op 552) reg expression
s_mov_b32 s61, s29 // (op 552) reg expression
s_mov_b32 s62, s0 // (op 552) reg expression
s_mov_b32 s63, 131072 // (op 552) call()

s_mul_hi_i32 s0, s16, s12 // (op 614) 
v_lshrrev_b32 v11, 6, v1 // (op 614) 
s_add_i32 s1, s0, s16 // (op 614) 

v_and_b32 v13, 1, v11 // (op 614) 

s_ashr_i32 s0, s1, 31 // (op 614) 
v_and_b32 v11, 15, v1 // (op 614) 
s_cmp_eq_i32 s12, 0 // (op 614) 
s_mov_b32 s8, scc // (op 614) 
v_lshl_add_u32 v14, v13, 4, v11 // (op 614) 


s_cmp_eq_i32 s8, 0 // (op 614) 
s_cselect_b32 s9, 0, -1 // (op 614) 

v_and_b32 v11, 63, v1 // (op 614) 
s_and_b32 s8, s13, 31 // (op 614) 
v_lshrrev_b32 v13, 4, v11 // (op 614) 

s_lshl_b32 s18, 1, s8 // (op 614) 
s_add_i32 s19, s18, s9 // (op 614) 


s_and_b32 s9, s0, s19 // (op 614) 


s_add_i32 s0, s1, s9 // (op 614) 


s_ashr_i32 s1, s0, s8 // (op 614) 


s_lshl_b32 s0, s13, 24 // (op 614) 
s_ashr_i32 s8, s0, 31 // (op 614) 

s_xor_b32 s0, s1, s8 // (op 614) 

s_sub_i32 s1, s0, s8 // (op 614) 


s_mul_hi_i32 s0, s1, s14 // (op 614) 
s_add_i32 s8, s0, s1 // (op 614) 

s_ashr_i32 s0, s8, 31 // (op 614) 
s_cmp_eq_i32 s14, 0 // (op 614) 
s_mov_b32 s9, scc // (op 614) 
s_cmp_eq_i32 s9, 0 // (op 614) 
s_cselect_b32 s18, 0, -1 // (op 614) 

s_and_b32 s9, s15, 31 // (op 614) 
s_lshl_b32 s19, 1, s9 // (op 614) 
s_add_i32 s49, s19, s18 // (op 614) 


s_and_b32 s18, s0, s49 // (op 614) 


s_add_i32 s0, s8, s18 // (op 614) 


s_ashr_i32 s8, s0, s9 // (op 614) 


s_lshl_b32 s0, s15, 24 // (op 614) 
s_ashr_i32 s9, s0, 31 // (op 614) 

s_xor_b32 s0, s8, s9 // (op 614) 

s_sub_i32 s8, s0, s9 // (op 614) 


s_add_u32 s18, s4, 32 // (op 614) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 614) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 614) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 614) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 614) 

s_mul_i32 s0, s8, s18 // (op 614) 

s_sub_i32 s8, s1, s0 // (op 614) 


s_mul_hi_i32 s9, s16, s11 // (op 614) 
s_add_i32 s0, s9, s16 // (op 614) 

s_ashr_i32 s1, s0, 31 // (op 614) 
s_cmp_eq_i32 s11, 0 // (op 614) 
s_mov_b32 s9, scc // (op 614) 
s_cmp_eq_i32 s9, 0 // (op 614) 
s_cselect_b32 s49, 0, -1 // (op 614) 

s_and_b32 s9, s46, 31 // (op 614) 
s_lshl_b32 s50, 1, s9 // (op 614) 
s_add_i32 s51, s50, s49 // (op 614) 


s_and_b32 s49, s1, s51 // (op 614) 


s_add_i32 s1, s0, s49 // (op 614) 


s_ashr_i32 s0, s1, s9 // (op 614) 


s_lshl_b32 s1, s46, 24 // (op 614) 
s_ashr_i32 s9, s1, 31 // (op 614) 

s_xor_b32 s1, s0, s9 // (op 614) 

s_sub_i32 s0, s1, s9 // (op 614) 


s_mul_i32 s1, s45, s18 // (op 614) 

s_mul_i32 s9, s10, s1 // (op 614) 

s_mul_i32 s1, s0, s9 // (op 614) 

s_sub_u32 s9, s16, s1 // (op 614) 

s_mul_hi_i32 s1, s9, s44 // (op 614) 
s_add_i32 s18, s1, s9 // (op 614) 


s_ashr_i32 s1, s18, 31 // (op 614) 
s_cmp_eq_i32 s44, 0 // (op 614) 
s_mov_b32 s9, scc // (op 614) 
s_cmp_eq_i32 s9, 0 // (op 614) 
s_cselect_b32 s19, 0, -1 // (op 614) 

s_and_b32 s9, s48, 31 // (op 614) 
s_lshl_b32 s49, 1, s9 // (op 614) 
s_add_i32 s50, s49, s19 // (op 614) 


s_and_b32 s19, s1, s50 // (op 614) 


s_add_i32 s1, s18, s19 // (op 614) 


s_ashr_i32 s18, s1, s9 // (op 614) 


s_lshl_b32 s1, s48, 24 // (op 614) 
s_ashr_i32 s9, s1, 31 // (op 614) 

s_xor_b32 s1, s18, s9 // (op 614) 

s_sub_i32 s18, s1, s9 // (op 614) 


s_cmp_eq_i32 s0, 0 // (op 614) 
s_mov_b32 s1, scc // (op 614) 

s_cmp_eq_i32 s1, 0 // (op 614) 
s_cselect_b32 s0, s18, s8 // (op 614) 



v_lshl_add_u32 v11, s0, 5, v14 // (op 614) 


v_mul_lo_u32 v14, v11, s30 // (op 614) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 614) 
s_lshr_b32 s8, s0, 1 // (op 614) 

v_lshl_add_u32 v11, s8, 3, v13 // (op 614) 


v_add_u32 v13, v11, v14 // (op 614) 


s_mul_i32 s1, s30, s5 // (op 616) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 616) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 616) least significant
s_mul_hi_u32 s9, s30, s4 // (op 616) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 616) most significant: sum
s_add_u32 s1, s1, s9 // (op 616) most significant: sum


s_mov_b32 s64, s32 // (op 616) reg expression
s_mov_b32 s65, s33 // (op 616) reg expression
s_mov_b32 s66, s0 // (op 616) reg expression
s_mov_b32 s67, 131072 // (op 616) call()

v_and_b32 v11, -8, v1 // (op 707) 
v_sub_u32 v14, v1, v11 // (op 707) 

v_lshlrev_b32 v11, 4, v14 // (op 707) 

v_lshrrev_b32 v14, 3, v1 // (op 707) 
v_lshl_add_u32 v15, v14, 7, v11 // (op 707) 


v_mul_lo_u32 v11, v15, 6 // (op 707) 

v_lshrrev_b32 v14, 3, v11 // (op 707) 

v_and_b32 v11, -8, v1 // (op 760) 
v_sub_u32 v15, v1, v11 // (op 760) 

v_lshlrev_b32 v11, 4, v15 // (op 760) 

v_lshrrev_b32 v15, 3, v1 // (op 760) 
v_lshl_add_u32 v17, v15, 7, v11 // (op 760) 


s_nop 0
v_readfirstlane_b32 s0, v17 // (op 760) Wait state hazard: VALU Write Readlane Hazard

s_ashr_i64 s[8:9], s[34:35], 7 // (op 157) 
s_and_b32 s1, s8, 4294967294 // (op 157) 

s_ashr_i64 s[8:9], s[34:35], 7 // (op 156) 
s_mov_b32 s18, s8 // (op 156) convert

s_cmp_lt_u32 s1, s18 // (op 156) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_156 // (op 156) Condition: Top (jump to RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_156 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_156:


s_add_u32 s1, s1, 1 // (op 158) 
v_mov_b32 v11, v10 // (op 161) 

buffer_load_dwordx3 v[16:18], v11, s[52:55], 0 offen // (op 161) Load value

v_add_u32 v10, 96, v10 // (op 409) 
s_barrier  // (op 919) 
s_barrier  // (op 932) 





v_mov_b64 v[20:21], v[8:9] // (op 390) 

s_mov_b32 m0, s0 // (op 390) reg expression
buffer_load_dwordx4 v20, s[56:59], 0 offen offset: 0 lds // (op 390) Load value direct to lds

v_add_co_u32 v8, vcc, 128, v8 // (op 478) least significant half
v_addc_co_u32 v9, vcc, 0, v9, vcc // (op 478) most significant half
s_waitcnt vmcnt(1)
ds_write_b96 v14, v[16:18] offset:8192 // (op 162) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 922) 



ds_read_b128 v[16:19], v3 offset:8192 // (op 164) Load local data 
ds_read_b64 v[20:21], v3 offset:8208 // (op 164) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 935) 



ds_read_b128 v[22:25], v4 // (op 166) Load local data 
ds_read_b128 v[26:29], v4 offset:64 // (op 166) Load local data 


v_mov_b32 v11, v12 // (op 168) 

buffer_load_ubyte v15, v11, s[60:63], 0 offen offset: 0 // (op 168) Load value

v_add_u32 v12, 4, v12 // (op 542) 


v_mov_b32 v11, v13 // (op 170) 

buffer_load_ubyte v30, v11, s[64:67], 0 offen offset: 0 // (op 170) Load value

v_add_u32 v13, 4, v13 // (op 606) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:21], v[22:29], a[0:3], v15, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 171) 




s_ashr_i64 s[8:9], s[34:35], 7 // (op 156) 
s_mov_b32 s18, s8 // (op 156) convert

s_cmp_lt_u32 s1, s18 // (op 156) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_156 // (op 156) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_156 if true)
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_156:




















s_barrier  // (op 989) 




v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_116
s_nop 0
v_accvgpr_read v8, a0 // (op 42) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 42) 
v_accvgpr_read v10, a2 // (op 42) 
v_accvgpr_read v11, a3 // (op 42) 
v_mul_f32 v12, v7, v8 // (op 42) 
v_mul_f32 v13, v7, v9 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 

v_lshrrev_b32 v8, 16, v12 // (op 48) 
v_lshrrev_b32 v9, 16, v13 // (op 48) 
v_lshrrev_b32 v10, 16, v14 // (op 48) 
v_lshrrev_b32 v11, 16, v15 // (op 48) 

s_mul_hi_i32 s0, s16, s12 // (op 801) 
v_lshrrev_b32 v3, 7, v1 // (op 801) 
s_add_i32 s1, s0, s16 // (op 801) 

v_and_b32 v4, 63, v1 // (op 801) 
s_ashr_i32 s0, s1, 31 // (op 801) 
v_lshrrev_b32 v5, 4, v4 // (op 801) 
s_cmp_eq_i32 s12, 0 // (op 801) 
s_mov_b32 s2, scc // (op 801) 
v_lshlrev_b32 v6, 2, v5 // (op 801) 

s_cmp_eq_i32 s2, 0 // (op 801) 
s_cselect_b32 s3, 0, -1 // (op 801) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 801) 


s_and_b32 s2, s13, 31 // (op 801) 
v_lshrrev_b32 v3, 2, v4 // (op 801) 

s_lshl_b32 s8, 1, s2 // (op 801) 
v_and_b32 v4, 3, v3 // (op 801) 

s_add_i32 s9, s8, s3 // (op 801) 


v_and_b32 v3, 3, v1 // (op 801) 
s_and_b32 s3, s0, s9 // (op 801) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 801) 


s_add_i32 s0, s1, s3 // (op 801) 


v_lshrrev_b32 v3, 6, v1 // (op 801) 
s_ashr_i32 s1, s0, s2 // (op 801) 


v_and_b32 v4, 1, v3 // (op 801) 

s_lshl_b32 s0, s13, 24 // (op 801) 
v_lshl_add_u32 v3, v4, 4, v6 // (op 801) 


s_ashr_i32 s2, s0, 31 // (op 801) 

s_xor_b32 s0, s1, s2 // (op 801) 

s_sub_i32 s1, s0, s2 // (op 801) 


s_mul_hi_i32 s0, s1, s14 // (op 801) 
s_add_i32 s2, s0, s1 // (op 801) 

s_ashr_i32 s0, s2, 31 // (op 801) 
s_cmp_eq_i32 s14, 0 // (op 801) 
s_mov_b32 s3, scc // (op 801) 
s_cmp_eq_i32 s3, 0 // (op 801) 
s_cselect_b32 s8, 0, -1 // (op 801) 

s_and_b32 s3, s15, 31 // (op 801) 
s_lshl_b32 s9, 1, s3 // (op 801) 
s_add_i32 s18, s9, s8 // (op 801) 


s_and_b32 s8, s0, s18 // (op 801) 


s_add_i32 s0, s2, s8 // (op 801) 


s_ashr_i32 s2, s0, s3 // (op 801) 


s_lshl_b32 s0, s15, 24 // (op 801) 
s_ashr_i32 s3, s0, 31 // (op 801) 

s_xor_b32 s0, s2, s3 // (op 801) 

s_sub_i32 s2, s0, s3 // (op 801) 


s_mul_i32 s0, s1, s45 // (op 801) 
s_sub_u32 s3, s16, s0 // (op 801) 

s_mul_i32 s0, s2, s45 // (op 801) 
s_add_u32 s0, s0, s3 // (op 801) 

s_mul_hi_i32 s3, s16, s11 // (op 801) 
s_add_i32 s8, s3, s16 // (op 801) 

s_ashr_i32 s3, s8, 31 // (op 801) 
s_cmp_eq_i32 s11, 0 // (op 801) 
s_mov_b32 s9, scc // (op 801) 
s_cmp_eq_i32 s9, 0 // (op 801) 
s_cselect_b32 s18, 0, -1 // (op 801) 

s_and_b32 s9, s46, 31 // (op 801) 
s_lshl_b32 s19, 1, s9 // (op 801) 
s_add_i32 s20, s19, s18 // (op 801) 


s_and_b32 s18, s3, s20 // (op 801) 


s_add_i32 s3, s8, s18 // (op 801) 


s_ashr_i32 s8, s3, s9 // (op 801) 


s_lshl_b32 s3, s46, 24 // (op 801) 
s_ashr_i32 s9, s3, 31 // (op 801) 

s_xor_b32 s3, s8, s9 // (op 801) 

s_sub_i32 s8, s3, s9 // (op 801) 


s_add_u32 s18, s4, 32 // (op 801) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 801) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 801) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 801) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 801) 

s_mul_i32 s3, s45, s18 // (op 801) 
s_mul_i32 s9, s10, s3 // (op 801) 

s_mul_i32 s3, s8, s9 // (op 801) 

s_sub_u32 s9, s16, s3 // (op 801) 

s_mov_b32 s3, s9 // (op 801) 
s_mul_hi_i32 s20, s3, s44 // (op 801) 
s_add_i32 s21, s20, s3 // (op 801) 


s_ashr_i32 s3, s21, 31 // (op 801) 
s_cmp_eq_i32 s44, 0 // (op 801) 
s_mov_b32 s20, scc // (op 801) 
s_cmp_eq_i32 s20, 0 // (op 801) 
s_cselect_b32 s22, 0, -1 // (op 801) 

s_and_b32 s20, s48, 31 // (op 801) 
s_lshl_b32 s23, 1, s20 // (op 801) 
s_add_i32 s24, s23, s22 // (op 801) 


s_and_b32 s22, s3, s24 // (op 801) 


s_add_i32 s3, s21, s22 // (op 801) 


s_ashr_i32 s21, s3, s20 // (op 801) 


s_lshl_b32 s3, s48, 24 // (op 801) 
s_ashr_i32 s20, s3, 31 // (op 801) 

s_xor_b32 s3, s21, s20 // (op 801) 

s_sub_i32 s21, s3, s20 // (op 801) 


s_mul_i32 s3, s47, s45 // (op 801) 
s_add_u32 s22, s6, 32 // (op 801) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 801) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 801) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 801) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 801) 

s_sub_i32 s20, s22, s3 // (op 801) 

s_mul_i32 s22, s21, s20 // (op 801) 

s_sub_u32 s20, s9, s22 // (op 801) 


s_add_u32 s9, s20, s3 // (op 801) 


s_cmp_eq_i32 s8, 0 // (op 801) 
s_mov_b32 s3, scc // (op 801) 

s_cmp_eq_i32 s3, 0 // (op 801) 
s_cselect_b32 s8, s9, s0 // (op 801) 


s_mul_i32 s0, s2, s18 // (op 801) 


v_lshl_add_u32 v4, s8, 5, v5 // (op 801) 


s_sub_i32 s2, s1, s0 // (op 801) 


s_cmp_eq_i32 s3, 0 // (op 801) 
s_cselect_b32 s0, s21, s2 // (op 801) 



v_lshl_add_u32 v5, s0, 5, v3 // (op 801) 


v_mul_lo_u32 v3, v5, s36 // (op 801) 

v_add_lshl_u32 v5, v4, v3, 1 // (op 801) 


s_mul_i32 s9, s36, s5 // (op 803) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 803) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 803) least significant
s_mul_hi_u32 s1, s36, s4 // (op 803) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 803) most significant: sum
s_add_u32 s9, s9, s1 // (op 803) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 803) 

s_mov_b32 s32, s38 // (op 803) reg expression
s_mov_b32 s33, s39 // (op 803) reg expression
s_mov_b32 s34, s0 // (op 803) reg expression
s_mov_b32 s35, 131072 // (op 803) call()


v_pack_b32_f16 v12, v8, v9 // (op 50) 
v_pack_b32_f16 v13, v10, v11 // (op 50) 


buffer_store_dwordx2 v[12:13], v5, s[32:35], 0 offen // (op 50) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_116
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_116:
s_mul_hi_i32 s0, s16, s12 // (op 677) 
v_lshrrev_b32 v3, 7, v1 // (op 677) 
s_add_i32 s1, s0, s16 // (op 677) 

v_and_b32 v4, 63, v1 // (op 677) 
s_ashr_i32 s0, s1, 31 // (op 677) 
v_lshrrev_b32 v5, 4, v4 // (op 677) 
s_cmp_eq_i32 s12, 0 // (op 677) 
s_mov_b32 s2, scc // (op 677) 
v_lshlrev_b32 v6, 2, v5 // (op 677) 

s_cmp_eq_i32 s2, 0 // (op 677) 
s_cselect_b32 s3, 0, -1 // (op 677) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 677) 


s_and_b32 s2, s13, 31 // (op 677) 
v_lshrrev_b32 v3, 2, v4 // (op 677) 

s_lshl_b32 s8, 1, s2 // (op 677) 
v_and_b32 v4, 3, v3 // (op 677) 

s_add_i32 s9, s8, s3 // (op 677) 


v_and_b32 v3, 3, v1 // (op 677) 
s_and_b32 s3, s0, s9 // (op 677) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 677) 


s_add_i32 s0, s1, s3 // (op 677) 


v_lshrrev_b32 v3, 6, v1 // (op 677) 
s_ashr_i32 s1, s0, s2 // (op 677) 


v_and_b32 v4, 1, v3 // (op 677) 

s_lshl_b32 s0, s13, 24 // (op 677) 
v_lshl_add_u32 v3, v4, 4, v6 // (op 677) 


s_ashr_i32 s2, s0, 31 // (op 677) 

s_xor_b32 s0, s1, s2 // (op 677) 

s_sub_i32 s1, s0, s2 // (op 677) 


s_mul_hi_i32 s0, s1, s14 // (op 677) 
s_add_i32 s2, s0, s1 // (op 677) 

s_ashr_i32 s0, s2, 31 // (op 677) 
s_cmp_eq_i32 s14, 0 // (op 677) 
s_mov_b32 s3, scc // (op 677) 
s_cmp_eq_i32 s3, 0 // (op 677) 
s_cselect_b32 s8, 0, -1 // (op 677) 

s_and_b32 s3, s15, 31 // (op 677) 
s_lshl_b32 s9, 1, s3 // (op 677) 
s_add_i32 s18, s9, s8 // (op 677) 


s_and_b32 s8, s0, s18 // (op 677) 


s_add_i32 s0, s2, s8 // (op 677) 


s_ashr_i32 s2, s0, s3 // (op 677) 


s_lshl_b32 s0, s15, 24 // (op 677) 
s_ashr_i32 s3, s0, 31 // (op 677) 

s_xor_b32 s0, s2, s3 // (op 677) 

s_sub_i32 s2, s0, s3 // (op 677) 


s_mul_i32 s0, s1, s45 // (op 677) 
s_sub_u32 s3, s16, s0 // (op 677) 

s_mul_i32 s0, s2, s45 // (op 677) 
s_add_u32 s0, s0, s3 // (op 677) 

s_mul_hi_i32 s3, s16, s11 // (op 677) 
s_add_i32 s8, s3, s16 // (op 677) 

s_ashr_i32 s3, s8, 31 // (op 677) 
s_cmp_eq_i32 s11, 0 // (op 677) 
s_mov_b32 s9, scc // (op 677) 
s_cmp_eq_i32 s9, 0 // (op 677) 
s_cselect_b32 s18, 0, -1 // (op 677) 

s_and_b32 s9, s46, 31 // (op 677) 
s_lshl_b32 s19, 1, s9 // (op 677) 
s_add_i32 s20, s19, s18 // (op 677) 


s_and_b32 s18, s3, s20 // (op 677) 


s_add_i32 s3, s8, s18 // (op 677) 


s_ashr_i32 s8, s3, s9 // (op 677) 


s_lshl_b32 s3, s46, 24 // (op 677) 
s_ashr_i32 s9, s3, 31 // (op 677) 

s_xor_b32 s3, s8, s9 // (op 677) 

s_sub_i32 s8, s3, s9 // (op 677) 


s_add_u32 s18, s4, 32 // (op 677) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 677) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 677) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 677) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 677) 

s_mul_i32 s3, s45, s18 // (op 677) 
s_mul_i32 s9, s10, s3 // (op 677) 

s_mul_i32 s3, s8, s9 // (op 677) 

s_sub_u32 s9, s16, s3 // (op 677) 

s_mov_b32 s3, s9 // (op 677) 
s_mul_hi_i32 s20, s3, s44 // (op 677) 
s_add_i32 s21, s20, s3 // (op 677) 


s_ashr_i32 s3, s21, 31 // (op 677) 
s_cmp_eq_i32 s44, 0 // (op 677) 
s_mov_b32 s20, scc // (op 677) 
s_cmp_eq_i32 s20, 0 // (op 677) 
s_cselect_b32 s22, 0, -1 // (op 677) 

s_and_b32 s20, s48, 31 // (op 677) 
s_lshl_b32 s23, 1, s20 // (op 677) 
s_add_i32 s24, s23, s22 // (op 677) 


s_and_b32 s22, s3, s24 // (op 677) 


s_add_i32 s3, s21, s22 // (op 677) 


s_ashr_i32 s21, s3, s20 // (op 677) 


s_lshl_b32 s3, s48, 24 // (op 677) 
s_ashr_i32 s20, s3, 31 // (op 677) 

s_xor_b32 s3, s21, s20 // (op 677) 

s_sub_i32 s21, s3, s20 // (op 677) 


s_mul_i32 s3, s47, s45 // (op 677) 
s_add_u32 s22, s6, 32 // (op 677) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 677) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 677) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 677) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 677) 

s_sub_i32 s20, s22, s3 // (op 677) 

s_mul_i32 s22, s21, s20 // (op 677) 

s_sub_u32 s20, s9, s22 // (op 677) 


s_add_u32 s9, s20, s3 // (op 677) 


s_cmp_eq_i32 s8, 0 // (op 677) 
s_mov_b32 s3, scc // (op 677) 

s_cmp_eq_i32 s3, 0 // (op 677) 
s_cselect_b32 s8, s9, s0 // (op 677) 


s_mul_i32 s0, s2, s18 // (op 677) 


v_lshl_add_u32 v4, s8, 5, v5 // (op 677) 


s_sub_i32 s2, s1, s0 // (op 677) 


s_cmp_eq_i32 s3, 0 // (op 677) 
s_cselect_b32 s0, s21, s2 // (op 677) 



v_lshl_add_u32 v5, s0, 5, v3 // (op 677) 


v_mul_lo_u32 v3, v5, s40 // (op 677) 

v_add_lshl_u32 v5, v4, v3, 1 // (op 677) 


s_mul_i32 s9, s40, s5 // (op 679) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 679) most significant: low of high * low
s_mul_i32 s8, s40, s4 // (op 679) least significant
s_mul_hi_u32 s1, s40, s4 // (op 679) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 679) most significant: sum
s_add_u32 s9, s9, s1 // (op 679) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 679) 

s_mov_b32 s28, s42 // (op 679) reg expression
s_mov_b32 s29, s43 // (op 679) reg expression
s_mov_b32 s30, s0 // (op 679) reg expression
s_mov_b32 s31, 131072 // (op 679) call()





buffer_load_dwordx2 v[8:9], v5, s[28:31], 0 offen // (op 129) Load value



v_accvgpr_read v10, a0 // (op 125) 
v_accvgpr_read v11, a1 // (op 125) 
v_accvgpr_read v12, a2 // (op 125) 
v_accvgpr_read v13, a3 // (op 125) 
v_mul_f32 v14, v7, v10 // (op 125) 
v_mul_f32 v15, v7, v11 // (op 125) 
v_mul_f32 v16, v7, v12 // (op 125) 
v_mul_f32 v17, v7, v13 // (op 125) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v8 // (op 130) 
v_lshrrev_b32 v11, 16, v8 // (op 130) 
v_lshlrev_b32 v10, 16, v10 // (op 130) 
v_lshlrev_b32 v11, 16, v11 // (op 130) 
v_mul_f32 v18, v0, v10 // (op 130) 
v_mul_f32 v19, v0, v11 // (op 130) 
v_and_b32 v10, 65535, v9 // (op 130) 
v_lshrrev_b32 v11, 16, v9 // (op 130) 
v_lshlrev_b32 v10, 16, v10 // (op 130) 
v_lshlrev_b32 v11, 16, v11 // (op 130) 
v_mul_f32 v20, v0, v10 // (op 130) 
v_mul_f32 v21, v0, v11 // (op 130) 


v_add_f32 v8, v18, v14 // (op 126) 
v_add_f32 v9, v19, v15 // (op 126) 
v_add_f32 v10, v20, v16 // (op 126) 
v_add_f32 v11, v21, v17 // (op 126) 


v_lshrrev_b32 v12, 16, v8 // (op 127) 
v_lshrrev_b32 v13, 16, v9 // (op 127) 
v_lshrrev_b32 v14, 16, v10 // (op 127) 
v_lshrrev_b32 v15, 16, v11 // (op 127) 

s_mul_hi_i32 s0, s16, s12 // (op 818) 
v_lshrrev_b32 v3, 7, v1 // (op 818) 
s_add_i32 s1, s0, s16 // (op 818) 

v_and_b32 v4, 63, v1 // (op 818) 
s_ashr_i32 s0, s1, 31 // (op 818) 
v_lshrrev_b32 v5, 4, v4 // (op 818) 
s_cmp_eq_i32 s12, 0 // (op 818) 
s_mov_b32 s2, scc // (op 818) 
v_lshlrev_b32 v6, 2, v5 // (op 818) 

s_cmp_eq_i32 s2, 0 // (op 818) 
s_cselect_b32 s3, 0, -1 // (op 818) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 818) 


s_and_b32 s2, s13, 31 // (op 818) 
v_lshrrev_b32 v3, 2, v4 // (op 818) 

s_lshl_b32 s8, 1, s2 // (op 818) 
v_and_b32 v4, 3, v3 // (op 818) 

s_add_i32 s9, s8, s3 // (op 818) 


v_and_b32 v3, 3, v1 // (op 818) 
s_and_b32 s3, s0, s9 // (op 818) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 818) 


s_add_i32 s0, s1, s3 // (op 818) 


v_lshrrev_b32 v3, 6, v1 // (op 818) 
s_ashr_i32 s1, s0, s2 // (op 818) 


v_and_b32 v4, 1, v3 // (op 818) 

s_lshl_b32 s0, s13, 24 // (op 818) 
v_lshl_add_u32 v3, v4, 4, v6 // (op 818) 


s_ashr_i32 s2, s0, 31 // (op 818) 

s_xor_b32 s0, s1, s2 // (op 818) 

s_sub_i32 s1, s0, s2 // (op 818) 


s_mul_hi_i32 s0, s1, s14 // (op 818) 
s_add_i32 s2, s0, s1 // (op 818) 

s_ashr_i32 s0, s2, 31 // (op 818) 
s_cmp_eq_i32 s14, 0 // (op 818) 
s_mov_b32 s3, scc // (op 818) 
s_cmp_eq_i32 s3, 0 // (op 818) 
s_cselect_b32 s8, 0, -1 // (op 818) 

s_and_b32 s3, s15, 31 // (op 818) 
s_lshl_b32 s9, 1, s3 // (op 818) 
s_add_i32 s18, s9, s8 // (op 818) 


s_and_b32 s8, s0, s18 // (op 818) 


s_add_i32 s0, s2, s8 // (op 818) 


s_ashr_i32 s2, s0, s3 // (op 818) 


s_lshl_b32 s0, s15, 24 // (op 818) 
s_ashr_i32 s3, s0, 31 // (op 818) 

s_xor_b32 s0, s2, s3 // (op 818) 

s_sub_i32 s2, s0, s3 // (op 818) 


s_mul_i32 s0, s1, s45 // (op 818) 
s_sub_u32 s3, s16, s0 // (op 818) 

s_mul_i32 s0, s2, s45 // (op 818) 
s_add_u32 s0, s0, s3 // (op 818) 

s_mul_hi_i32 s3, s16, s11 // (op 818) 
s_add_i32 s8, s3, s16 // (op 818) 

s_ashr_i32 s3, s8, 31 // (op 818) 
s_cmp_eq_i32 s11, 0 // (op 818) 
s_mov_b32 s9, scc // (op 818) 
s_cmp_eq_i32 s9, 0 // (op 818) 
s_cselect_b32 s18, 0, -1 // (op 818) 

s_and_b32 s9, s46, 31 // (op 818) 
s_lshl_b32 s19, 1, s9 // (op 818) 
s_add_i32 s20, s19, s18 // (op 818) 


s_and_b32 s18, s3, s20 // (op 818) 


s_add_i32 s3, s8, s18 // (op 818) 


s_ashr_i32 s8, s3, s9 // (op 818) 


s_lshl_b32 s3, s46, 24 // (op 818) 
s_ashr_i32 s9, s3, 31 // (op 818) 

s_xor_b32 s3, s8, s9 // (op 818) 

s_sub_i32 s8, s3, s9 // (op 818) 


s_add_u32 s18, s4, 32 // (op 818) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 818) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 818) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 818) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 818) 

s_mul_i32 s3, s45, s18 // (op 818) 
s_mul_i32 s9, s10, s3 // (op 818) 

s_mul_i32 s3, s8, s9 // (op 818) 

s_sub_u32 s9, s16, s3 // (op 818) 

s_mov_b32 s3, s9 // (op 818) 
s_mul_hi_i32 s20, s3, s44 // (op 818) 
s_add_i32 s21, s20, s3 // (op 818) 


s_ashr_i32 s3, s21, 31 // (op 818) 
s_cmp_eq_i32 s44, 0 // (op 818) 
s_mov_b32 s20, scc // (op 818) 
s_cmp_eq_i32 s20, 0 // (op 818) 
s_cselect_b32 s22, 0, -1 // (op 818) 

s_and_b32 s20, s48, 31 // (op 818) 
s_lshl_b32 s23, 1, s20 // (op 818) 
s_add_i32 s24, s23, s22 // (op 818) 


s_and_b32 s22, s3, s24 // (op 818) 


s_add_i32 s3, s21, s22 // (op 818) 


s_ashr_i32 s21, s3, s20 // (op 818) 


s_lshl_b32 s3, s48, 24 // (op 818) 
s_ashr_i32 s20, s3, 31 // (op 818) 

s_xor_b32 s3, s21, s20 // (op 818) 

s_sub_i32 s21, s3, s20 // (op 818) 


s_mul_i32 s3, s47, s45 // (op 818) 
s_add_u32 s22, s6, 32 // (op 818) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 818) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 818) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 818) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 818) 

s_sub_i32 s20, s22, s3 // (op 818) 

s_mul_i32 s22, s21, s20 // (op 818) 

s_sub_u32 s20, s9, s22 // (op 818) 


s_add_u32 s9, s20, s3 // (op 818) 


s_cmp_eq_i32 s8, 0 // (op 818) 
s_mov_b32 s3, scc // (op 818) 

s_cmp_eq_i32 s3, 0 // (op 818) 
s_cselect_b32 s8, s9, s0 // (op 818) 


s_mul_i32 s0, s2, s18 // (op 818) 


v_lshl_add_u32 v4, s8, 5, v5 // (op 818) 


s_sub_i32 s2, s1, s0 // (op 818) 


s_cmp_eq_i32 s3, 0 // (op 818) 
s_cselect_b32 s0, s21, s2 // (op 818) 



v_lshl_add_u32 v5, s0, 5, v3 // (op 818) 


v_mul_lo_u32 v3, v5, s36 // (op 818) 

v_add_lshl_u32 v5, v4, v3, 1 // (op 818) 


s_mul_i32 s9, s36, s5 // (op 820) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 820) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 820) least significant
s_mul_hi_u32 s1, s36, s4 // (op 820) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 820) most significant: sum
s_add_u32 s9, s9, s1 // (op 820) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 820) 

s_mov_b32 s32, s38 // (op 820) reg expression
s_mov_b32 s33, s39 // (op 820) reg expression
s_mov_b32 s34, s0 // (op 820) reg expression
s_mov_b32 s35, 131072 // (op 820) call()


v_pack_b32_f16 v8, v12, v13 // (op 128) 
v_pack_b32_f16 v9, v14, v15 // (op 128) 


buffer_store_dwordx2 v[8:9], v5, s[32:35], 0 offen // (op 128) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_116:
















s_endpgm  // End of RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.LRR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__end:
.size RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_, .LRR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__end-RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 48
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 14336 // lds bytes
  .amdhsa_accum_offset 44
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: B8
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
    macrotile: [32, 32, 128]
    threads: [256, 1, 1]
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
  - .name: RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 14336
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 70
    .vgpr_count: 44
    .agpr_count: 4
    .max_flat_workgroup_size: 256
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

