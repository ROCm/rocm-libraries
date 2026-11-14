// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_,@function
RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 63, v1 // (op 735) 
v_and_b32 v3, 48, v1 // (op 735) 
v_sub_u32 v4, v0, v3 // (op 735) 

v_lshrrev_b32 v3, 4, v0 // (op 735) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 735) 


v_lshrrev_b32 v3, 7, v1 // (op 735) 
v_lshlrev_b32 v4, 4, v3 // (op 735) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 735) 


v_mul_lo_u32 v0, v3, 6 // (op 735) 
v_lshrrev_b32 v4, 3, v0 // (op 735) 

v_lshrrev_b32 v0, 4, v3 // (op 735) 

v_lshlrev_b32 v3, 2, v0 // (op 735) 

v_add_u32 v0, v4, v3 // (op 735) 


v_and_b32 v3, 48, v1 // (op 787) 
v_and_b32 v4, 63, v1 // (op 787) 
v_sub_u32 v5, v4, v3 // (op 787) 

v_lshrrev_b32 v4, 1, v5 // (op 787) 
v_add_lshl_u32 v6, v3, v4, 5 // (op 787) 


v_and_b32 v3, 1, v5 // (op 787) 

v_lshlrev_b32 v4, 3, v3 // (op 787) 

v_lshrrev_b32 v3, 6, v1 // (op 787) 
v_and_b32 v5, 1, v3 // (op 787) 

v_lshl_add_u32 v3, v5, 4, v4 // (op 787) 


v_add_u32 v4, v6, v3 // (op 787) 


v_and_b32 v3, 63, v1 // (op 845) 
v_and_b32 v5, 48, v1 // (op 845) 
v_sub_u32 v6, v3, v5 // (op 845) 

v_lshrrev_b32 v5, 4, v3 // (op 845) 

v_lshl_add_u32 v3, v5, 5, v6 // (op 845) 


v_lshrrev_b32 v5, 7, v1 // (op 845) 
v_lshlrev_b32 v6, 4, v5 // (op 845) 

v_lshl_add_u32 v5, v3, 5, v6 // (op 845) 


v_mul_lo_u32 v3, v5, 6 // (op 845) 
v_lshrrev_b32 v6, 3, v3 // (op 845) 

v_lshrrev_b32 v3, 4, v5 // (op 845) 

v_lshlrev_b32 v5, 2, v3 // (op 845) 

v_add_u32 v3, v6, v5 // (op 845) 


v_and_b32 v5, 48, v1 // (op 884) 
v_and_b32 v6, 63, v1 // (op 884) 
v_sub_u32 v7, v6, v5 // (op 884) 

v_lshrrev_b32 v6, 1, v7 // (op 884) 
v_add_lshl_u32 v8, v5, v6, 5 // (op 884) 


v_and_b32 v5, 1, v7 // (op 884) 

v_lshlrev_b32 v6, 3, v5 // (op 884) 

v_lshrrev_b32 v5, 6, v1 // (op 884) 
v_and_b32 v7, 1, v5 // (op 884) 

v_lshl_add_u32 v5, v7, 4, v6 // (op 884) 


v_add_u32 v6, v8, v5 // (op 884) 


v_mov_b32 v5, s6 // (op 2) Move value
v_mov_b32 v7, s7 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
s_mul_hi_i32 s0, s16, s9 // (op 445) 
v_and_b32 v8, -2, v1 // (op 445) 
s_add_i32 s1, s0, s16 // (op 445) 

v_sub_u32 v9, v1, v8 // (op 445) 

s_ashr_i32 s0, s1, 31 // (op 445) 
v_lshlrev_b32 v8, 4, v9 // (op 445) 

s_cmp_eq_i32 s9, 0 // (op 445) 
s_mov_b32 s6, scc // (op 445) 
v_lshrrev_b32 v9, 1, v1 // (op 445) 
s_cmp_eq_i32 s6, 0 // (op 445) 
s_cselect_b32 s7, 0, -1 // (op 445) 

s_and_b32 s6, s10, 31 // (op 445) 
s_lshl_b32 s18, 1, s6 // (op 445) 
s_add_i32 s19, s18, s7 // (op 445) 


s_and_b32 s7, s0, s19 // (op 445) 


s_add_i32 s0, s1, s7 // (op 445) 


s_ashr_i32 s1, s0, s6 // (op 445) 


s_lshl_b32 s0, s10, 24 // (op 445) 
s_ashr_i32 s6, s0, 31 // (op 445) 

s_xor_b32 s0, s1, s6 // (op 445) 

s_sub_i32 s1, s0, s6 // (op 445) 


s_mul_hi_i32 s0, s1, s11 // (op 445) 
s_add_i32 s6, s0, s1 // (op 445) 

s_ashr_i32 s0, s6, 31 // (op 445) 
s_cmp_eq_i32 s11, 0 // (op 445) 
s_mov_b32 s7, scc // (op 445) 
s_cmp_eq_i32 s7, 0 // (op 445) 
s_cselect_b32 s18, 0, -1 // (op 445) 

s_and_b32 s7, s12, 31 // (op 445) 
s_lshl_b32 s19, 1, s7 // (op 445) 
s_add_i32 s51, s19, s18 // (op 445) 


s_and_b32 s18, s0, s51 // (op 445) 


s_add_i32 s0, s6, s18 // (op 445) 


s_ashr_i32 s6, s0, s7 // (op 445) 


s_lshl_b32 s0, s12, 24 // (op 445) 
s_ashr_i32 s7, s0, 31 // (op 445) 

s_xor_b32 s0, s6, s7 // (op 445) 

s_sub_i32 s6, s0, s7 // (op 445) 


s_mul_i32 s0, s1, s14 // (op 445) 

s_sub_u32 s1, s16, s0 // (op 445) 

s_mul_i32 s0, s6, s14 // (op 445) 
s_add_u32 s0, s0, s1 // (op 445) 


s_mul_hi_i32 s1, s16, s8 // (op 445) 
s_add_i32 s6, s1, s16 // (op 445) 

s_ashr_i32 s1, s6, 31 // (op 445) 
s_cmp_eq_i32 s8, 0 // (op 445) 
s_mov_b32 s7, scc // (op 445) 
s_cmp_eq_i32 s7, 0 // (op 445) 
s_cselect_b32 s18, 0, -1 // (op 445) 

s_and_b32 s7, s15, 31 // (op 445) 
s_lshl_b32 s19, 1, s7 // (op 445) 
s_add_i32 s51, s19, s18 // (op 445) 


s_and_b32 s18, s1, s51 // (op 445) 


s_add_i32 s1, s6, s18 // (op 445) 


s_ashr_i32 s6, s1, s7 // (op 445) 


s_lshl_b32 s1, s15, 24 // (op 445) 
s_ashr_i32 s7, s1, 31 // (op 445) 

s_xor_b32 s1, s6, s7 // (op 445) 

s_sub_i32 s6, s1, s7 // (op 445) 


s_add_u32 s18, s4, 32 // (op 445) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 445) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 445) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 445) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 5 // (op 445) 

s_mul_i32 s1, s14, s18 // (op 445) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 445) 

s_mul_i32 s1, s6, s7 // (op 445) 

s_sub_u32 s7, s16, s1 // (op 445) 

s_mov_b32 s1, s7 // (op 445) 
s_mul_hi_i32 s18, s1, s13 // (op 445) 
s_add_i32 s19, s18, s1 // (op 445) 


s_ashr_i32 s1, s19, 31 // (op 445) 
s_cmp_eq_i32 s13, 0 // (op 445) 
s_mov_b32 s18, scc // (op 445) 
s_cmp_eq_i32 s18, 0 // (op 445) 
s_cselect_b32 s51, 0, -1 // (op 445) 

s_and_b32 s18, s49, 31 // (op 445) 
s_lshl_b32 s52, 1, s18 // (op 445) 
s_add_i32 s53, s52, s51 // (op 445) 


s_and_b32 s51, s1, s53 // (op 445) 


s_add_i32 s1, s19, s51 // (op 445) 


s_ashr_i32 s19, s1, s18 // (op 445) 


s_lshl_b32 s1, s49, 24 // (op 445) 
s_ashr_i32 s18, s1, 31 // (op 445) 

s_xor_b32 s1, s19, s18 // (op 445) 

s_sub_i32 s19, s1, s18 // (op 445) 


s_mul_i32 s1, s48, s14 // (op 445) 
s_add_u32 s52, s20, 32 // (op 445) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 445) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 445) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 445) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 5 // (op 445) 

s_sub_i32 s18, s52, s1 // (op 445) 

s_mul_i32 s51, s19, s18 // (op 445) 


s_sub_u32 s18, s7, s51 // (op 445) 


s_add_u32 s7, s18, s1 // (op 445) 


s_cmp_eq_i32 s6, 0 // (op 445) 
s_mov_b32 s1, scc // (op 445) 

s_cmp_eq_i32 s1, 0 // (op 445) 
s_cselect_b32 s6, s7, s0 // (op 445) 



v_lshl_add_u32 v10, s6, 5, v8 // (op 445) 


v_mov_b32 v13, 0 // (op 445) convert
v_mov_b32 v12, v10 // (op 445) convert

v_mul_lo_u32 v11, v9, s3 // (op 445) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 445) most significant: high of low * low
v_mul_lo_u32 v10, v9, s2 // (op 445) least significant: low of low * low
v_add_u32 v11, v11, v8 // (op 445) most significant: sum


v_add_co_u32 v8, s[0:1], v12, v10 // (op 445) least significant half
v_addc_co_u32 v9, s[0:1], v13, v11, s[0:1] // (op 445) most significant half



v_mul_lo_u32 v11, v9, 6 // (op 445) most significant: low of high * low
v_mul_hi_u32 v12, v8, 6 // (op 445) most significant: high of low * low
v_mul_lo_u32 v10, v8, 6 // (op 445) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 445) most significant: sum


v_lshrrev_b64 v[8:9], 3, v[10:11] // (op 445) 

v_mov_b32 v10, v8 // (op 445) convert

s_mul_i32 s1, s25, 6 // (op 447) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 447) least significant
s_mul_hi_u32 s6, s24, 6 // (op 447) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 447) most significant: sum

s_mul_i32 s7, s2, s1 // (op 447) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 447) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 447) least significant
s_mul_hi_u32 s19, s2, s0 // (op 447) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 447) most significant: sum
s_add_u32 s7, s7, s19 // (op 447) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 447) 

s_mov_b32 s52, s22 // (op 447) reg expression
s_mov_b32 s53, s23 // (op 447) reg expression
s_mov_b32 s54, s0 // (op 447) reg expression
s_mov_b32 s55, 131072 // (op 447) call()

s_mul_hi_i32 s51, s16, s9 // (op 464) 
v_and_b32 v8, -2, v1 // (op 464) 
s_add_i32 s0, s51, s16 // (op 464) 

v_sub_u32 v9, v1, v8 // (op 464) 

s_ashr_i32 s1, s0, 31 // (op 464) 
v_lshlrev_b32 v8, 4, v9 // (op 464) 

s_cmp_eq_i32 s9, 0 // (op 464) 
s_mov_b32 s51, scc // (op 464) 
v_lshrrev_b32 v9, 1, v1 // (op 464) 
s_cmp_eq_i32 s51, 0 // (op 464) 
s_cselect_b32 s6, 0, -1 // (op 464) 

s_and_b32 s7, s10, 31 // (op 464) 
s_lshl_b32 s51, 1, s7 // (op 464) 
s_add_i32 s18, s51, s6 // (op 464) 


s_and_b32 s6, s1, s18 // (op 464) 


s_add_i32 s1, s0, s6 // (op 464) 


s_ashr_i32 s0, s1, s7 // (op 464) 


s_lshl_b32 s1, s10, 24 // (op 464) 
s_ashr_i32 s51, s1, 31 // (op 464) 

s_xor_b32 s1, s0, s51 // (op 464) 

s_sub_i32 s0, s1, s51 // (op 464) 


s_mul_hi_i32 s1, s0, s11 // (op 464) 
s_add_i32 s51, s1, s0 // (op 464) 

s_ashr_i32 s1, s51, 31 // (op 464) 
s_cmp_eq_i32 s11, 0 // (op 464) 
s_mov_b32 s6, scc // (op 464) 
s_cmp_eq_i32 s6, 0 // (op 464) 
s_cselect_b32 s7, 0, -1 // (op 464) 

s_and_b32 s6, s12, 31 // (op 464) 
s_lshl_b32 s18, 1, s6 // (op 464) 
s_add_i32 s19, s18, s7 // (op 464) 


s_and_b32 s7, s1, s19 // (op 464) 


s_add_i32 s1, s51, s7 // (op 464) 


s_ashr_i32 s7, s1, s6 // (op 464) 


s_lshl_b32 s1, s12, 24 // (op 464) 
s_ashr_i32 s6, s1, 31 // (op 464) 

s_xor_b32 s1, s7, s6 // (op 464) 

s_sub_i32 s7, s1, s6 // (op 464) 


s_mul_i32 s1, s0, s14 // (op 464) 

s_sub_u32 s0, s16, s1 // (op 464) 

s_mul_i32 s1, s7, s14 // (op 464) 
s_add_u32 s1, s1, s0 // (op 464) 


s_mul_hi_i32 s0, s16, s8 // (op 464) 
s_add_i32 s51, s0, s16 // (op 464) 

s_ashr_i32 s0, s51, 31 // (op 464) 
s_cmp_eq_i32 s8, 0 // (op 464) 
s_mov_b32 s6, scc // (op 464) 
s_cmp_eq_i32 s6, 0 // (op 464) 
s_cselect_b32 s7, 0, -1 // (op 464) 

s_and_b32 s6, s15, 31 // (op 464) 
s_lshl_b32 s18, 1, s6 // (op 464) 
s_add_i32 s19, s18, s7 // (op 464) 


s_and_b32 s7, s0, s19 // (op 464) 


s_add_i32 s0, s51, s7 // (op 464) 


s_ashr_i32 s7, s0, s6 // (op 464) 


s_lshl_b32 s0, s15, 24 // (op 464) 
s_ashr_i32 s6, s0, 31 // (op 464) 

s_xor_b32 s0, s7, s6 // (op 464) 

s_sub_i32 s7, s0, s6 // (op 464) 


s_add_u32 s18, s4, 32 // (op 464) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 464) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 464) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 464) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 464) 

s_mul_i32 s0, s14, s18 // (op 464) 

s_mul_i32 s6, s50, s0 // (op 464) 

s_mul_i32 s0, s7, s6 // (op 464) 

s_sub_u32 s6, s16, s0 // (op 464) 

s_mov_b32 s0, s6 // (op 464) 
s_mul_hi_i32 s51, s0, s13 // (op 464) 
s_add_i32 s18, s51, s0 // (op 464) 


s_ashr_i32 s0, s18, 31 // (op 464) 
s_cmp_eq_i32 s13, 0 // (op 464) 
s_mov_b32 s19, scc // (op 464) 
s_cmp_eq_i32 s19, 0 // (op 464) 
s_cselect_b32 s51, 0, -1 // (op 464) 

s_and_b32 s19, s49, 31 // (op 464) 
s_lshl_b32 s56, 1, s19 // (op 464) 
s_add_i32 s57, s56, s51 // (op 464) 


s_and_b32 s51, s0, s57 // (op 464) 


s_add_i32 s0, s18, s51 // (op 464) 


s_ashr_i32 s18, s0, s19 // (op 464) 


s_lshl_b32 s0, s49, 24 // (op 464) 
s_ashr_i32 s19, s0, 31 // (op 464) 

s_xor_b32 s0, s18, s19 // (op 464) 

s_sub_i32 s18, s0, s19 // (op 464) 


s_mul_i32 s0, s48, s14 // (op 464) 
s_add_u32 s56, s20, 32 // (op 464) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 464) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 464) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 464) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 464) 

s_sub_i32 s19, s56, s0 // (op 464) 

s_mul_i32 s51, s18, s19 // (op 464) 


s_sub_u32 s18, s6, s51 // (op 464) 


s_add_u32 s6, s18, s0 // (op 464) 


s_cmp_eq_i32 s7, 0 // (op 464) 
s_mov_b32 s0, scc // (op 464) 

s_cmp_eq_i32 s0, 0 // (op 464) 
s_cselect_b32 s7, s6, s1 // (op 464) 



v_lshl_add_u32 v11, s7, 5, v8 // (op 464) 


v_mov_b32 v13, 0 // (op 464) convert
v_mov_b32 v12, v11 // (op 464) convert

v_mul_lo_u32 v15, v9, s3 // (op 464) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 464) most significant: high of low * low
v_mul_lo_u32 v14, v9, s2 // (op 464) least significant: low of low * low
v_add_u32 v15, v15, v8 // (op 464) most significant: sum


v_add_co_u32 v8, s[0:1], v12, v14 // (op 464) least significant half
v_addc_co_u32 v9, s[0:1], v13, v15, s[0:1] // (op 464) most significant half



v_mul_lo_u32 v13, v9, 6 // (op 464) most significant: low of high * low
v_mul_hi_u32 v11, v8, 6 // (op 464) most significant: high of low * low
v_mul_lo_u32 v12, v8, 6 // (op 464) least significant: low of low * low
v_add_u32 v13, v13, v11 // (op 464) most significant: sum


v_lshrrev_b64 v[8:9], 3, v[12:13] // (op 464) 

v_mov_b32 v11, v8 // (op 464) convert

s_mul_i32 s1, s25, 6 // (op 466) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 466) least significant
s_mul_hi_u32 s51, s24, 6 // (op 466) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 466) most significant: sum

s_mul_i32 s7, s2, s1 // (op 466) most significant: low of low * high
s_mul_i32 s51, s3, s0 // (op 466) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 466) least significant
s_mul_hi_u32 s18, s2, s0 // (op 466) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 466) most significant: sum
s_add_u32 s7, s7, s18 // (op 466) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 466) 

s_mov_b32 s52, s22 // (op 466) reg expression
s_mov_b32 s53, s23 // (op 466) reg expression
s_mov_b32 s54, s0 // (op 466) reg expression
s_mov_b32 s55, 131072 // (op 466) call()

s_mul_hi_i32 s51, s16, s9 // (op 509) 
v_and_b32 v8, -2, v1 // (op 509) 
s_add_i32 s0, s51, s16 // (op 509) 

v_sub_u32 v9, v1, v8 // (op 509) 

s_ashr_i32 s1, s0, 31 // (op 509) 
v_lshlrev_b32 v8, 4, v9 // (op 509) 

s_cmp_eq_i32 s9, 0 // (op 509) 
s_mov_b32 s51, scc // (op 509) 
v_lshrrev_b32 v9, 1, v1 // (op 509) 
s_cmp_eq_i32 s51, 0 // (op 509) 
s_cselect_b32 s6, 0, -1 // (op 509) 

s_and_b32 s7, s10, 31 // (op 509) 
s_lshl_b32 s51, 1, s7 // (op 509) 
s_add_i32 s18, s51, s6 // (op 509) 


s_and_b32 s6, s1, s18 // (op 509) 


s_add_i32 s1, s0, s6 // (op 509) 


s_ashr_i32 s0, s1, s7 // (op 509) 


s_lshl_b32 s1, s10, 24 // (op 509) 
s_ashr_i32 s51, s1, 31 // (op 509) 

s_xor_b32 s1, s0, s51 // (op 509) 

s_sub_i32 s0, s1, s51 // (op 509) 


s_mul_hi_i32 s1, s0, s11 // (op 509) 
s_add_i32 s51, s1, s0 // (op 509) 

s_ashr_i32 s1, s51, 31 // (op 509) 
s_cmp_eq_i32 s11, 0 // (op 509) 
s_mov_b32 s6, scc // (op 509) 
s_cmp_eq_i32 s6, 0 // (op 509) 
s_cselect_b32 s7, 0, -1 // (op 509) 

s_and_b32 s6, s12, 31 // (op 509) 
s_lshl_b32 s18, 1, s6 // (op 509) 
s_add_i32 s19, s18, s7 // (op 509) 


s_and_b32 s7, s1, s19 // (op 509) 


s_add_i32 s1, s51, s7 // (op 509) 


s_ashr_i32 s7, s1, s6 // (op 509) 


s_lshl_b32 s1, s12, 24 // (op 509) 
s_ashr_i32 s6, s1, 31 // (op 509) 

s_xor_b32 s1, s7, s6 // (op 509) 

s_sub_i32 s7, s1, s6 // (op 509) 


s_add_u32 s18, s4, 32 // (op 509) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 509) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 509) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 509) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 509) 

s_mul_i32 s1, s7, s18 // (op 509) 

s_sub_i32 s51, s0, s1 // (op 509) 


s_mul_hi_i32 s0, s16, s8 // (op 509) 
s_add_i32 s1, s0, s16 // (op 509) 

s_ashr_i32 s0, s1, 31 // (op 509) 
s_cmp_eq_i32 s8, 0 // (op 509) 
s_mov_b32 s6, scc // (op 509) 
s_cmp_eq_i32 s6, 0 // (op 509) 
s_cselect_b32 s7, 0, -1 // (op 509) 

s_and_b32 s6, s15, 31 // (op 509) 
s_lshl_b32 s56, 1, s6 // (op 509) 
s_add_i32 s57, s56, s7 // (op 509) 


s_and_b32 s7, s0, s57 // (op 509) 


s_add_i32 s0, s1, s7 // (op 509) 


s_ashr_i32 s1, s0, s6 // (op 509) 


s_lshl_b32 s0, s15, 24 // (op 509) 
s_ashr_i32 s6, s0, 31 // (op 509) 

s_xor_b32 s0, s1, s6 // (op 509) 

s_sub_i32 s1, s0, s6 // (op 509) 


s_mul_i32 s0, s14, s18 // (op 509) 

s_mul_i32 s6, s50, s0 // (op 509) 

s_mul_i32 s0, s1, s6 // (op 509) 

s_sub_u32 s6, s16, s0 // (op 509) 

s_mul_hi_i32 s0, s6, s13 // (op 509) 
s_add_i32 s7, s0, s6 // (op 509) 


s_ashr_i32 s0, s7, 31 // (op 509) 
s_cmp_eq_i32 s13, 0 // (op 509) 
s_mov_b32 s6, scc // (op 509) 
s_cmp_eq_i32 s6, 0 // (op 509) 
s_cselect_b32 s18, 0, -1 // (op 509) 

s_and_b32 s6, s49, 31 // (op 509) 
s_lshl_b32 s19, 1, s6 // (op 509) 
s_add_i32 s56, s19, s18 // (op 509) 


s_and_b32 s18, s0, s56 // (op 509) 


s_add_i32 s0, s7, s18 // (op 509) 


s_ashr_i32 s7, s0, s6 // (op 509) 


s_lshl_b32 s0, s49, 24 // (op 509) 
s_ashr_i32 s6, s0, 31 // (op 509) 

s_xor_b32 s0, s7, s6 // (op 509) 

s_sub_i32 s7, s0, s6 // (op 509) 


s_cmp_eq_i32 s1, 0 // (op 509) 
s_mov_b32 s0, scc // (op 509) 

s_cmp_eq_i32 s0, 0 // (op 509) 
s_cselect_b32 s1, s7, s51 // (op 509) 



v_lshl_add_u32 v12, s1, 5, v8 // (op 509) 


v_mov_b32 v15, 0 // (op 509) convert
v_mov_b32 v14, v12 // (op 509) convert

v_mul_lo_u32 v13, v9, s27 // (op 509) most significant: low of low * high
v_mul_hi_u32 v8, v9, s26 // (op 509) most significant: high of low * low
v_mul_lo_u32 v12, v9, s26 // (op 509) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 509) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v14 // (op 509) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 509) most significant half



s_mul_i32 s1, s26, s25 // (op 511) most significant: low of low * high
s_mul_i32 s51, s27, s24 // (op 511) most significant: low of high * low
s_mul_i32 s0, s26, s24 // (op 511) least significant
s_mul_hi_u32 s6, s26, s24 // (op 511) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 511) most significant: sum
s_add_u32 s1, s1, s6 // (op 511) most significant: sum


s_mov_b32 s56, s28 // (op 511) reg expression
s_mov_b32 s57, s29 // (op 511) reg expression
s_mov_b32 s58, s0 // (op 511) reg expression
s_mov_b32 s59, 131072 // (op 511) call()

s_mul_hi_i32 s51, s16, s9 // (op 528) 
v_and_b32 v8, -2, v1 // (op 528) 
s_add_i32 s0, s51, s16 // (op 528) 

v_sub_u32 v9, v1, v8 // (op 528) 

s_ashr_i32 s1, s0, 31 // (op 528) 
v_lshlrev_b32 v8, 4, v9 // (op 528) 

s_cmp_eq_i32 s9, 0 // (op 528) 
s_mov_b32 s51, scc // (op 528) 
v_lshrrev_b32 v9, 1, v1 // (op 528) 
s_cmp_eq_i32 s51, 0 // (op 528) 
s_cselect_b32 s6, 0, -1 // (op 528) 

s_and_b32 s7, s10, 31 // (op 528) 
s_lshl_b32 s51, 1, s7 // (op 528) 
s_add_i32 s18, s51, s6 // (op 528) 


s_and_b32 s6, s1, s18 // (op 528) 


s_add_i32 s1, s0, s6 // (op 528) 


s_ashr_i32 s0, s1, s7 // (op 528) 


s_lshl_b32 s1, s10, 24 // (op 528) 
s_ashr_i32 s51, s1, 31 // (op 528) 

s_xor_b32 s1, s0, s51 // (op 528) 

s_sub_i32 s0, s1, s51 // (op 528) 


s_mul_hi_i32 s1, s0, s11 // (op 528) 
s_add_i32 s51, s1, s0 // (op 528) 

s_ashr_i32 s1, s51, 31 // (op 528) 
s_cmp_eq_i32 s11, 0 // (op 528) 
s_mov_b32 s6, scc // (op 528) 
s_cmp_eq_i32 s6, 0 // (op 528) 
s_cselect_b32 s7, 0, -1 // (op 528) 

s_and_b32 s6, s12, 31 // (op 528) 
s_lshl_b32 s18, 1, s6 // (op 528) 
s_add_i32 s19, s18, s7 // (op 528) 


s_and_b32 s7, s1, s19 // (op 528) 


s_add_i32 s1, s51, s7 // (op 528) 


s_ashr_i32 s7, s1, s6 // (op 528) 


s_lshl_b32 s1, s12, 24 // (op 528) 
s_ashr_i32 s6, s1, 31 // (op 528) 

s_xor_b32 s1, s7, s6 // (op 528) 

s_sub_i32 s7, s1, s6 // (op 528) 


s_add_u32 s18, s4, 32 // (op 528) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 528) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 528) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 528) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 528) 

s_mul_i32 s1, s7, s18 // (op 528) 

s_sub_i32 s51, s0, s1 // (op 528) 


s_mul_hi_i32 s0, s16, s8 // (op 528) 
s_add_i32 s1, s0, s16 // (op 528) 

s_ashr_i32 s0, s1, 31 // (op 528) 
s_cmp_eq_i32 s8, 0 // (op 528) 
s_mov_b32 s6, scc // (op 528) 
s_cmp_eq_i32 s6, 0 // (op 528) 
s_cselect_b32 s7, 0, -1 // (op 528) 

s_and_b32 s6, s15, 31 // (op 528) 
s_lshl_b32 s60, 1, s6 // (op 528) 
s_add_i32 s61, s60, s7 // (op 528) 


s_and_b32 s7, s0, s61 // (op 528) 


s_add_i32 s0, s1, s7 // (op 528) 


s_ashr_i32 s1, s0, s6 // (op 528) 


s_lshl_b32 s0, s15, 24 // (op 528) 
s_ashr_i32 s6, s0, 31 // (op 528) 

s_xor_b32 s0, s1, s6 // (op 528) 

s_sub_i32 s1, s0, s6 // (op 528) 


s_mul_i32 s0, s14, s18 // (op 528) 

s_mul_i32 s6, s50, s0 // (op 528) 

s_mul_i32 s0, s1, s6 // (op 528) 

s_sub_u32 s6, s16, s0 // (op 528) 

s_mul_hi_i32 s0, s6, s13 // (op 528) 
s_add_i32 s7, s0, s6 // (op 528) 


s_ashr_i32 s0, s7, 31 // (op 528) 
s_cmp_eq_i32 s13, 0 // (op 528) 
s_mov_b32 s6, scc // (op 528) 
s_cmp_eq_i32 s6, 0 // (op 528) 
s_cselect_b32 s18, 0, -1 // (op 528) 

s_and_b32 s6, s49, 31 // (op 528) 
s_lshl_b32 s19, 1, s6 // (op 528) 
s_add_i32 s60, s19, s18 // (op 528) 


s_and_b32 s18, s0, s60 // (op 528) 


s_add_i32 s0, s7, s18 // (op 528) 


s_ashr_i32 s7, s0, s6 // (op 528) 


s_lshl_b32 s0, s49, 24 // (op 528) 
s_ashr_i32 s6, s0, 31 // (op 528) 

s_xor_b32 s0, s7, s6 // (op 528) 

s_sub_i32 s7, s0, s6 // (op 528) 


s_cmp_eq_i32 s1, 0 // (op 528) 
s_mov_b32 s0, scc // (op 528) 

s_cmp_eq_i32 s0, 0 // (op 528) 
s_cselect_b32 s1, s7, s51 // (op 528) 



v_lshl_add_u32 v14, s1, 5, v8 // (op 528) 


v_mov_b32 v17, 0 // (op 528) convert
v_mov_b32 v16, v14 // (op 528) convert

v_mul_lo_u32 v15, v9, s27 // (op 528) most significant: low of low * high
v_mul_hi_u32 v8, v9, s26 // (op 528) most significant: high of low * low
v_mul_lo_u32 v14, v9, s26 // (op 528) least significant: low of low * low
v_add_u32 v15, v15, v8 // (op 528) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v16 // (op 528) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 528) most significant half



s_mul_i32 s1, s26, s25 // (op 530) most significant: low of low * high
s_mul_i32 s51, s27, s24 // (op 530) most significant: low of high * low
s_mul_i32 s0, s26, s24 // (op 530) least significant
s_mul_hi_u32 s6, s26, s24 // (op 530) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 530) most significant: sum
s_add_u32 s1, s1, s6 // (op 530) most significant: sum


s_mov_b32 s56, s28 // (op 530) reg expression
s_mov_b32 s57, s29 // (op 530) reg expression
s_mov_b32 s58, s0 // (op 530) reg expression
s_mov_b32 s59, 131072 // (op 530) call()

s_mul_hi_i32 s51, s16, s9 // (op 573) 
v_lshrrev_b32 v8, 7, v1 // (op 573) 
s_add_i32 s0, s51, s16 // (op 573) 

v_and_b32 v9, 15, v1 // (op 573) 
s_ashr_i32 s1, s0, 31 // (op 573) 
v_lshl_add_u32 v16, v8, 4, v9 // (op 573) 


s_cmp_eq_i32 s9, 0 // (op 573) 
s_mov_b32 s51, scc // (op 573) 
v_and_b32 v8, 63, v1 // (op 573) 
s_cmp_eq_i32 s51, 0 // (op 573) 
s_cselect_b32 s6, 0, -1 // (op 573) 

v_lshrrev_b32 v9, 4, v8 // (op 573) 

s_and_b32 s7, s10, 31 // (op 573) 
s_lshl_b32 s51, 1, s7 // (op 573) 
s_add_i32 s18, s51, s6 // (op 573) 


s_and_b32 s6, s1, s18 // (op 573) 


s_add_i32 s1, s0, s6 // (op 573) 


s_ashr_i32 s0, s1, s7 // (op 573) 


s_lshl_b32 s1, s10, 24 // (op 573) 
s_ashr_i32 s51, s1, 31 // (op 573) 

s_xor_b32 s1, s0, s51 // (op 573) 

s_sub_i32 s0, s1, s51 // (op 573) 


s_mul_hi_i32 s1, s0, s11 // (op 573) 
s_add_i32 s51, s1, s0 // (op 573) 

s_ashr_i32 s1, s51, 31 // (op 573) 
s_cmp_eq_i32 s11, 0 // (op 573) 
s_mov_b32 s6, scc // (op 573) 
s_cmp_eq_i32 s6, 0 // (op 573) 
s_cselect_b32 s7, 0, -1 // (op 573) 

s_and_b32 s6, s12, 31 // (op 573) 
s_lshl_b32 s18, 1, s6 // (op 573) 
s_add_i32 s19, s18, s7 // (op 573) 


s_and_b32 s7, s1, s19 // (op 573) 


s_add_i32 s1, s51, s7 // (op 573) 


s_ashr_i32 s7, s1, s6 // (op 573) 


s_lshl_b32 s1, s12, 24 // (op 573) 
s_ashr_i32 s6, s1, 31 // (op 573) 

s_xor_b32 s1, s7, s6 // (op 573) 

s_sub_i32 s7, s1, s6 // (op 573) 


s_mul_i32 s1, s0, s14 // (op 573) 

s_sub_u32 s0, s16, s1 // (op 573) 

s_mul_i32 s1, s7, s14 // (op 573) 
s_add_u32 s1, s1, s0 // (op 573) 


s_mul_hi_i32 s0, s16, s8 // (op 573) 
s_add_i32 s51, s0, s16 // (op 573) 

s_ashr_i32 s0, s51, 31 // (op 573) 
s_cmp_eq_i32 s8, 0 // (op 573) 
s_mov_b32 s6, scc // (op 573) 
s_cmp_eq_i32 s6, 0 // (op 573) 
s_cselect_b32 s7, 0, -1 // (op 573) 

s_and_b32 s6, s15, 31 // (op 573) 
s_lshl_b32 s18, 1, s6 // (op 573) 
s_add_i32 s19, s18, s7 // (op 573) 


s_and_b32 s7, s0, s19 // (op 573) 


s_add_i32 s0, s51, s7 // (op 573) 


s_ashr_i32 s7, s0, s6 // (op 573) 


s_lshl_b32 s0, s15, 24 // (op 573) 
s_ashr_i32 s6, s0, 31 // (op 573) 

s_xor_b32 s0, s7, s6 // (op 573) 

s_sub_i32 s7, s0, s6 // (op 573) 


s_add_u32 s18, s4, 32 // (op 573) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 573) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 573) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 573) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 573) 

s_mul_i32 s0, s14, s18 // (op 573) 

s_mul_i32 s6, s50, s0 // (op 573) 

s_mul_i32 s0, s7, s6 // (op 573) 

s_sub_u32 s6, s16, s0 // (op 573) 

s_mov_b32 s0, s6 // (op 573) 
s_mul_hi_i32 s51, s0, s13 // (op 573) 
s_add_i32 s18, s51, s0 // (op 573) 


s_ashr_i32 s0, s18, 31 // (op 573) 
s_cmp_eq_i32 s13, 0 // (op 573) 
s_mov_b32 s19, scc // (op 573) 
s_cmp_eq_i32 s19, 0 // (op 573) 
s_cselect_b32 s51, 0, -1 // (op 573) 

s_and_b32 s19, s49, 31 // (op 573) 
s_lshl_b32 s60, 1, s19 // (op 573) 
s_add_i32 s61, s60, s51 // (op 573) 


s_and_b32 s51, s0, s61 // (op 573) 


s_add_i32 s0, s18, s51 // (op 573) 


s_ashr_i32 s18, s0, s19 // (op 573) 


s_lshl_b32 s0, s49, 24 // (op 573) 
s_ashr_i32 s19, s0, 31 // (op 573) 

s_xor_b32 s0, s18, s19 // (op 573) 

s_sub_i32 s18, s0, s19 // (op 573) 


s_mul_i32 s0, s48, s14 // (op 573) 
s_add_u32 s60, s20, 32 // (op 573) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 573) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 573) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 573) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 5 // (op 573) 

s_sub_i32 s19, s60, s0 // (op 573) 

s_mul_i32 s51, s18, s19 // (op 573) 


s_sub_u32 s18, s6, s51 // (op 573) 


s_add_u32 s6, s18, s0 // (op 573) 


s_cmp_eq_i32 s7, 0 // (op 573) 
s_mov_b32 s0, scc // (op 573) 

s_cmp_eq_i32 s0, 0 // (op 573) 
s_cselect_b32 s7, s6, s1 // (op 573) 



v_lshl_add_u32 v8, s7, 5, v16 // (op 573) 


v_mul_lo_u32 v16, v9, s30 // (op 573) 

v_add_u32 v9, v8, v16 // (op 573) 


s_mul_i32 s1, s30, s35 // (op 575) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 575) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 575) least significant
s_mul_hi_u32 s6, s30, s34 // (op 575) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 575) most significant: sum
s_add_u32 s1, s1, s6 // (op 575) most significant: sum


s_mov_b32 s60, s32 // (op 575) reg expression
s_mov_b32 s61, s33 // (op 575) reg expression
s_mov_b32 s62, s0 // (op 575) reg expression
s_mov_b32 s63, 131072 // (op 575) call()

s_mul_hi_i32 s51, s16, s9 // (op 592) 
v_lshrrev_b32 v8, 7, v1 // (op 592) 
s_add_i32 s0, s51, s16 // (op 592) 

v_and_b32 v16, 15, v1 // (op 592) 
s_ashr_i32 s1, s0, 31 // (op 592) 
v_lshl_add_u32 v17, v8, 4, v16 // (op 592) 


s_cmp_eq_i32 s9, 0 // (op 592) 
s_mov_b32 s51, scc // (op 592) 
v_and_b32 v8, 63, v1 // (op 592) 
s_cmp_eq_i32 s51, 0 // (op 592) 
s_cselect_b32 s6, 0, -1 // (op 592) 

v_lshrrev_b32 v16, 4, v8 // (op 592) 

s_and_b32 s7, s10, 31 // (op 592) 
s_lshl_b32 s51, 1, s7 // (op 592) 
s_add_i32 s18, s51, s6 // (op 592) 


s_and_b32 s6, s1, s18 // (op 592) 


s_add_i32 s1, s0, s6 // (op 592) 


s_ashr_i32 s0, s1, s7 // (op 592) 


s_lshl_b32 s1, s10, 24 // (op 592) 
s_ashr_i32 s51, s1, 31 // (op 592) 

s_xor_b32 s1, s0, s51 // (op 592) 

s_sub_i32 s0, s1, s51 // (op 592) 


s_mul_hi_i32 s1, s0, s11 // (op 592) 
s_add_i32 s51, s1, s0 // (op 592) 

s_ashr_i32 s1, s51, 31 // (op 592) 
s_cmp_eq_i32 s11, 0 // (op 592) 
s_mov_b32 s6, scc // (op 592) 
s_cmp_eq_i32 s6, 0 // (op 592) 
s_cselect_b32 s7, 0, -1 // (op 592) 

s_and_b32 s6, s12, 31 // (op 592) 
s_lshl_b32 s18, 1, s6 // (op 592) 
s_add_i32 s19, s18, s7 // (op 592) 


s_and_b32 s7, s1, s19 // (op 592) 


s_add_i32 s1, s51, s7 // (op 592) 


s_ashr_i32 s7, s1, s6 // (op 592) 


s_lshl_b32 s1, s12, 24 // (op 592) 
s_ashr_i32 s6, s1, 31 // (op 592) 

s_xor_b32 s1, s7, s6 // (op 592) 

s_sub_i32 s7, s1, s6 // (op 592) 


s_mul_i32 s1, s0, s14 // (op 592) 

s_sub_u32 s0, s16, s1 // (op 592) 

s_mul_i32 s1, s7, s14 // (op 592) 
s_add_u32 s1, s1, s0 // (op 592) 


s_mul_hi_i32 s0, s16, s8 // (op 592) 
s_add_i32 s51, s0, s16 // (op 592) 

s_ashr_i32 s0, s51, 31 // (op 592) 
s_cmp_eq_i32 s8, 0 // (op 592) 
s_mov_b32 s6, scc // (op 592) 
s_cmp_eq_i32 s6, 0 // (op 592) 
s_cselect_b32 s7, 0, -1 // (op 592) 

s_and_b32 s6, s15, 31 // (op 592) 
s_lshl_b32 s18, 1, s6 // (op 592) 
s_add_i32 s19, s18, s7 // (op 592) 


s_and_b32 s7, s0, s19 // (op 592) 


s_add_i32 s0, s51, s7 // (op 592) 


s_ashr_i32 s7, s0, s6 // (op 592) 


s_lshl_b32 s0, s15, 24 // (op 592) 
s_ashr_i32 s6, s0, 31 // (op 592) 

s_xor_b32 s0, s7, s6 // (op 592) 

s_sub_i32 s7, s0, s6 // (op 592) 


s_add_u32 s18, s4, 32 // (op 592) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 592) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 592) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 592) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 592) 

s_mul_i32 s0, s14, s18 // (op 592) 

s_mul_i32 s6, s50, s0 // (op 592) 

s_mul_i32 s0, s7, s6 // (op 592) 

s_sub_u32 s6, s16, s0 // (op 592) 

s_mov_b32 s0, s6 // (op 592) 
s_mul_hi_i32 s51, s0, s13 // (op 592) 
s_add_i32 s18, s51, s0 // (op 592) 


s_ashr_i32 s0, s18, 31 // (op 592) 
s_cmp_eq_i32 s13, 0 // (op 592) 
s_mov_b32 s19, scc // (op 592) 
s_cmp_eq_i32 s19, 0 // (op 592) 
s_cselect_b32 s51, 0, -1 // (op 592) 

s_and_b32 s19, s49, 31 // (op 592) 
s_lshl_b32 s64, 1, s19 // (op 592) 
s_add_i32 s65, s64, s51 // (op 592) 


s_and_b32 s51, s0, s65 // (op 592) 


s_add_i32 s0, s18, s51 // (op 592) 


s_ashr_i32 s18, s0, s19 // (op 592) 


s_lshl_b32 s0, s49, 24 // (op 592) 
s_ashr_i32 s19, s0, 31 // (op 592) 

s_xor_b32 s0, s18, s19 // (op 592) 

s_sub_i32 s18, s0, s19 // (op 592) 


s_mul_i32 s0, s48, s14 // (op 592) 
s_add_u32 s64, s20, 32 // (op 592) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 592) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 592) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 592) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 5 // (op 592) 

s_sub_i32 s19, s64, s0 // (op 592) 

s_mul_i32 s51, s18, s19 // (op 592) 


s_sub_u32 s18, s6, s51 // (op 592) 


s_add_u32 s6, s18, s0 // (op 592) 


s_cmp_eq_i32 s7, 0 // (op 592) 
s_mov_b32 s0, scc // (op 592) 

s_cmp_eq_i32 s0, 0 // (op 592) 
s_cselect_b32 s7, s6, s1 // (op 592) 



v_lshl_add_u32 v8, s7, 5, v17 // (op 592) 


v_mul_lo_u32 v17, v16, s30 // (op 592) 

v_add_u32 v16, v8, v17 // (op 592) 


s_mul_i32 s1, s30, s35 // (op 594) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 594) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 594) least significant
s_mul_hi_u32 s6, s30, s34 // (op 594) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 594) most significant: sum
s_add_u32 s1, s1, s6 // (op 594) most significant: sum


s_mov_b32 s60, s32 // (op 594) reg expression
s_mov_b32 s61, s33 // (op 594) reg expression
s_mov_b32 s62, s0 // (op 594) reg expression
s_mov_b32 s63, 131072 // (op 594) call()

s_mul_hi_i32 s51, s16, s9 // (op 637) 
v_lshrrev_b32 v8, 6, v1 // (op 637) 
s_add_i32 s0, s51, s16 // (op 637) 

v_and_b32 v17, 1, v8 // (op 637) 

s_ashr_i32 s1, s0, 31 // (op 637) 
v_and_b32 v8, 15, v1 // (op 637) 
s_cmp_eq_i32 s9, 0 // (op 637) 
s_mov_b32 s51, scc // (op 637) 
v_lshl_add_u32 v18, v17, 4, v8 // (op 637) 


s_cmp_eq_i32 s51, 0 // (op 637) 
s_cselect_b32 s6, 0, -1 // (op 637) 

v_and_b32 v8, 63, v1 // (op 637) 
s_and_b32 s7, s10, 31 // (op 637) 
v_lshrrev_b32 v17, 4, v8 // (op 637) 

s_lshl_b32 s51, 1, s7 // (op 637) 
s_add_i32 s18, s51, s6 // (op 637) 


s_and_b32 s6, s1, s18 // (op 637) 


s_add_i32 s1, s0, s6 // (op 637) 


s_ashr_i32 s0, s1, s7 // (op 637) 


s_lshl_b32 s1, s10, 24 // (op 637) 
s_ashr_i32 s51, s1, 31 // (op 637) 

s_xor_b32 s1, s0, s51 // (op 637) 

s_sub_i32 s0, s1, s51 // (op 637) 


s_mul_hi_i32 s1, s0, s11 // (op 637) 
s_add_i32 s51, s1, s0 // (op 637) 

s_ashr_i32 s1, s51, 31 // (op 637) 
s_cmp_eq_i32 s11, 0 // (op 637) 
s_mov_b32 s6, scc // (op 637) 
s_cmp_eq_i32 s6, 0 // (op 637) 
s_cselect_b32 s7, 0, -1 // (op 637) 

s_and_b32 s6, s12, 31 // (op 637) 
s_lshl_b32 s18, 1, s6 // (op 637) 
s_add_i32 s19, s18, s7 // (op 637) 


s_and_b32 s7, s1, s19 // (op 637) 


s_add_i32 s1, s51, s7 // (op 637) 


s_ashr_i32 s7, s1, s6 // (op 637) 


s_lshl_b32 s1, s12, 24 // (op 637) 
s_ashr_i32 s6, s1, 31 // (op 637) 

s_xor_b32 s1, s7, s6 // (op 637) 

s_sub_i32 s7, s1, s6 // (op 637) 


s_add_u32 s18, s4, 32 // (op 637) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 637) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 637) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 637) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 637) 

s_mul_i32 s1, s7, s18 // (op 637) 

s_sub_i32 s51, s0, s1 // (op 637) 


s_mul_hi_i32 s0, s16, s8 // (op 637) 
s_add_i32 s1, s0, s16 // (op 637) 

s_ashr_i32 s0, s1, 31 // (op 637) 
s_cmp_eq_i32 s8, 0 // (op 637) 
s_mov_b32 s6, scc // (op 637) 
s_cmp_eq_i32 s6, 0 // (op 637) 
s_cselect_b32 s7, 0, -1 // (op 637) 

s_and_b32 s6, s15, 31 // (op 637) 
s_lshl_b32 s64, 1, s6 // (op 637) 
s_add_i32 s65, s64, s7 // (op 637) 


s_and_b32 s7, s0, s65 // (op 637) 


s_add_i32 s0, s1, s7 // (op 637) 


s_ashr_i32 s1, s0, s6 // (op 637) 


s_lshl_b32 s0, s15, 24 // (op 637) 
s_ashr_i32 s6, s0, 31 // (op 637) 

s_xor_b32 s0, s1, s6 // (op 637) 

s_sub_i32 s1, s0, s6 // (op 637) 


s_mul_i32 s0, s14, s18 // (op 637) 

s_mul_i32 s6, s50, s0 // (op 637) 

s_mul_i32 s0, s1, s6 // (op 637) 

s_sub_u32 s6, s16, s0 // (op 637) 

s_mul_hi_i32 s0, s6, s13 // (op 637) 
s_add_i32 s7, s0, s6 // (op 637) 


s_ashr_i32 s0, s7, 31 // (op 637) 
s_cmp_eq_i32 s13, 0 // (op 637) 
s_mov_b32 s6, scc // (op 637) 
s_cmp_eq_i32 s6, 0 // (op 637) 
s_cselect_b32 s18, 0, -1 // (op 637) 

s_and_b32 s6, s49, 31 // (op 637) 
s_lshl_b32 s19, 1, s6 // (op 637) 
s_add_i32 s64, s19, s18 // (op 637) 


s_and_b32 s18, s0, s64 // (op 637) 


s_add_i32 s0, s7, s18 // (op 637) 


s_ashr_i32 s7, s0, s6 // (op 637) 


s_lshl_b32 s0, s49, 24 // (op 637) 
s_ashr_i32 s6, s0, 31 // (op 637) 

s_xor_b32 s0, s7, s6 // (op 637) 

s_sub_i32 s7, s0, s6 // (op 637) 


s_cmp_eq_i32 s1, 0 // (op 637) 
s_mov_b32 s0, scc // (op 637) 

s_cmp_eq_i32 s0, 0 // (op 637) 
s_cselect_b32 s1, s7, s51 // (op 637) 



v_lshl_add_u32 v8, s1, 5, v18 // (op 637) 


v_mul_lo_u32 v18, v17, s36 // (op 637) 
v_add_u32 v18, v18, v8 // (op 637) 


s_mul_i32 s1, s36, s35 // (op 639) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 639) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 639) least significant
s_mul_hi_u32 s6, s36, s34 // (op 639) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 639) most significant: sum
s_add_u32 s1, s1, s6 // (op 639) most significant: sum


s_mov_b32 s64, s38 // (op 639) reg expression
s_mov_b32 s65, s39 // (op 639) reg expression
s_mov_b32 s66, s0 // (op 639) reg expression
s_mov_b32 s67, 131072 // (op 639) call()

s_mul_hi_i32 s51, s16, s9 // (op 656) 
v_lshrrev_b32 v8, 6, v1 // (op 656) 
s_add_i32 s0, s51, s16 // (op 656) 

v_and_b32 v17, 1, v8 // (op 656) 

s_ashr_i32 s1, s0, 31 // (op 656) 
v_and_b32 v8, 15, v1 // (op 656) 
s_cmp_eq_i32 s9, 0 // (op 656) 
s_mov_b32 s51, scc // (op 656) 
v_lshl_add_u32 v19, v17, 4, v8 // (op 656) 


s_cmp_eq_i32 s51, 0 // (op 656) 
s_cselect_b32 s6, 0, -1 // (op 656) 

v_and_b32 v8, 63, v1 // (op 656) 
s_and_b32 s7, s10, 31 // (op 656) 
v_lshrrev_b32 v17, 4, v8 // (op 656) 

s_lshl_b32 s51, 1, s7 // (op 656) 
s_add_i32 s18, s51, s6 // (op 656) 


s_and_b32 s6, s1, s18 // (op 656) 


s_add_i32 s1, s0, s6 // (op 656) 


s_ashr_i32 s0, s1, s7 // (op 656) 


s_lshl_b32 s1, s10, 24 // (op 656) 
s_ashr_i32 s51, s1, 31 // (op 656) 

s_xor_b32 s1, s0, s51 // (op 656) 

s_sub_i32 s0, s1, s51 // (op 656) 


s_mul_hi_i32 s1, s0, s11 // (op 656) 
s_add_i32 s51, s1, s0 // (op 656) 

s_ashr_i32 s1, s51, 31 // (op 656) 
s_cmp_eq_i32 s11, 0 // (op 656) 
s_mov_b32 s6, scc // (op 656) 
s_cmp_eq_i32 s6, 0 // (op 656) 
s_cselect_b32 s7, 0, -1 // (op 656) 

s_and_b32 s6, s12, 31 // (op 656) 
s_lshl_b32 s18, 1, s6 // (op 656) 
s_add_i32 s19, s18, s7 // (op 656) 


s_and_b32 s7, s1, s19 // (op 656) 


s_add_i32 s1, s51, s7 // (op 656) 


s_ashr_i32 s7, s1, s6 // (op 656) 


s_lshl_b32 s1, s12, 24 // (op 656) 
s_ashr_i32 s6, s1, 31 // (op 656) 

s_xor_b32 s1, s7, s6 // (op 656) 

s_sub_i32 s7, s1, s6 // (op 656) 


s_add_u32 s18, s4, 32 // (op 656) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 656) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 656) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 656) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 656) 

s_mul_i32 s1, s7, s18 // (op 656) 

s_sub_i32 s51, s0, s1 // (op 656) 


s_mul_hi_i32 s0, s16, s8 // (op 656) 
s_add_i32 s1, s0, s16 // (op 656) 

s_ashr_i32 s0, s1, 31 // (op 656) 
s_cmp_eq_i32 s8, 0 // (op 656) 
s_mov_b32 s6, scc // (op 656) 
s_cmp_eq_i32 s6, 0 // (op 656) 
s_cselect_b32 s7, 0, -1 // (op 656) 

s_and_b32 s6, s15, 31 // (op 656) 
s_lshl_b32 s68, 1, s6 // (op 656) 
s_add_i32 s69, s68, s7 // (op 656) 


s_and_b32 s7, s0, s69 // (op 656) 


s_add_i32 s0, s1, s7 // (op 656) 


s_ashr_i32 s1, s0, s6 // (op 656) 


s_lshl_b32 s0, s15, 24 // (op 656) 
s_ashr_i32 s6, s0, 31 // (op 656) 

s_xor_b32 s0, s1, s6 // (op 656) 

s_sub_i32 s1, s0, s6 // (op 656) 


s_mul_i32 s0, s14, s18 // (op 656) 

s_mul_i32 s6, s50, s0 // (op 656) 

s_mul_i32 s0, s1, s6 // (op 656) 

s_sub_u32 s6, s16, s0 // (op 656) 

s_mul_hi_i32 s0, s6, s13 // (op 656) 
s_add_i32 s7, s0, s6 // (op 656) 


s_ashr_i32 s0, s7, 31 // (op 656) 
s_cmp_eq_i32 s13, 0 // (op 656) 
s_mov_b32 s6, scc // (op 656) 
s_cmp_eq_i32 s6, 0 // (op 656) 
s_cselect_b32 s18, 0, -1 // (op 656) 

s_and_b32 s6, s49, 31 // (op 656) 
s_lshl_b32 s19, 1, s6 // (op 656) 
s_add_i32 s68, s19, s18 // (op 656) 


s_and_b32 s18, s0, s68 // (op 656) 


s_add_i32 s0, s7, s18 // (op 656) 


s_ashr_i32 s7, s0, s6 // (op 656) 


s_lshl_b32 s0, s49, 24 // (op 656) 
s_ashr_i32 s6, s0, 31 // (op 656) 

s_xor_b32 s0, s7, s6 // (op 656) 

s_sub_i32 s7, s0, s6 // (op 656) 


s_cmp_eq_i32 s1, 0 // (op 656) 
s_mov_b32 s0, scc // (op 656) 

s_cmp_eq_i32 s0, 0 // (op 656) 
s_cselect_b32 s1, s7, s51 // (op 656) 



v_lshl_add_u32 v8, s1, 5, v19 // (op 656) 


v_mul_lo_u32 v19, v17, s36 // (op 656) 
v_add_u32 v19, v19, v8 // (op 656) 


s_mul_i32 s1, s36, s35 // (op 658) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 658) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 658) least significant
s_mul_hi_u32 s6, s36, s34 // (op 658) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 658) most significant: sum
s_add_u32 s1, s1, s6 // (op 658) most significant: sum


s_mov_b32 s64, s38 // (op 658) reg expression
s_mov_b32 s65, s39 // (op 658) reg expression
s_mov_b32 s66, s0 // (op 658) reg expression
s_mov_b32 s67, 131072 // (op 658) call()


v_mov_b64 v[20:21], s[30:31] // (op 312) convert

v_mov_b32 v8, v9 // (op 312) reg expression
v_mov_b32 v17, v8 // (op 312) 
buffer_load_ubyte v22, v17, s[60:63], 0 offen offset: 0 // (op 312) Load value




v_mov_b64 v[20:21], s[36:37] // (op 317) convert

v_mov_b32 v8, v18 // (op 317) reg expression
v_mov_b32 v17, v8 // (op 317) 
buffer_load_ubyte v23, v17, s[64:67], 0 offen offset: 0 // (op 317) Load value



v_and_b32 v8, -2, v1 // (op 775) 
v_sub_u32 v17, v1, v8 // (op 775) 

v_lshlrev_b32 v8, 4, v17 // (op 775) 

v_lshrrev_b32 v17, 1, v1 // (op 775) 
v_lshl_add_u32 v20, v17, 5, v8 // (op 775) 


s_nop 0
v_readfirstlane_b32 s51, v20 // (op 775) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[20:21], v[12:13] // (op 401) reg expression

s_nop 0
s_mov_b32 m0, s51 // (op 401) reg expression
buffer_load_dwordx4 v20, s[56:59], 0 offen offset: 0 lds // (op 401) Load value direct to lds




v_mov_b32 v8, v10 // (op 324) reg expression

buffer_load_dwordx3 v[24:26], v8, s[52:55], 0 offen // (op 324) Load value


v_mov_b64 v[20:21], s[30:31] // (op 328) convert

s_lshl_b32 s51, s30, 2 // (op 328) 
v_add_u32 v8, v9, s51 // (op 328) 

v_mov_b32 v17, v8 // (op 328) 
buffer_load_ubyte v27, v17, s[60:63], 0 offen offset: 0 // (op 328) Load value





v_mov_b64 v[8:9], s[36:37] // (op 333) convert

s_lshl_b32 s51, s36, 2 // (op 333) 
v_add_u32 v17, v18, s51 // (op 333) 

v_mov_b32 v20, v17 // (op 333) 
buffer_load_ubyte v21, v20, s[64:67], 0 offen offset: 0 // (op 333) Load value




v_and_b32 v20, -2, v1 // (op 872) 
v_sub_u32 v8, v1, v20 // (op 872) 

v_lshlrev_b32 v9, 4, v8 // (op 872) 

v_lshrrev_b32 v8, 1, v1 // (op 872) 
v_lshl_add_u32 v20, v8, 5, v9 // (op 872) 


s_nop 0
v_readfirstlane_b32 s51, v20 // (op 872) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[0:1], s[26:27], 7 // (op 406) 
v_mov_b32 v20, s0 // (op 406) 
v_mov_b32 v8, s1 // (op 406) 
v_add_co_u32 v28, s[6:7], v12, v20 // (op 406) least significant half
v_addc_co_u32 v29, s[6:7], v13, v8, s[6:7] // (op 406) most significant half




s_add_u32 s0, 4096, s51 // (op 406) 

s_mov_b32 m0, s0 // (op 406) reg expression
buffer_load_dwordx4 v28, s[56:59], 0 offen offset: 0 lds // (op 406) Load value direct to lds






s_lshl_b64 s[0:1], s[2:3], 7 // (op 340) 
s_mul_i32 s7, s1, 6 // (op 340) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 340) least significant
s_mul_hi_u32 s51, s0, 6 // (op 340) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 340) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 340) 

v_add_u32 v20, v10, s0 // (op 340) 


buffer_load_dwordx3 v[28:30], v20, s[52:55], 0 offen // (op 340) Load value


v_and_b32 v20, -2, v1 // (op 723) 
v_sub_u32 v8, v1, v20 // (op 723) 

v_lshlrev_b32 v20, 4, v8 // (op 723) 

v_lshrrev_b32 v8, 1, v1 // (op 723) 
v_lshl_add_u32 v9, v8, 5, v20 // (op 723) 


v_mul_lo_u32 v8, v9, 6 // (op 723) 
v_lshrrev_b32 v10, 3, v8 // (op 723) 

v_lshrrev_b32 v8, 4, v9 // (op 723) 

v_lshlrev_b32 v9, 2, v8 // (op 723) 

v_add_u32 v8, v10, v9 // (op 723) 





s_waitcnt vmcnt(4)
ds_write_b96 v8, v[24:26] offset:8192 // (op 343) Store local data 

v_and_b32 v20, -2, v1 // (op 694) 
v_sub_u32 v8, v1, v20 // (op 694) 

v_lshlrev_b32 v20, 4, v8 // (op 694) 

v_lshrrev_b32 v8, 1, v1 // (op 694) 
v_lshl_add_u32 v9, v8, 5, v20 // (op 694) 


v_mul_lo_u32 v8, v9, 6 // (op 694) 
v_lshrrev_b32 v10, 3, v8 // (op 694) 

v_lshrrev_b32 v8, 4, v9 // (op 694) 

v_lshlrev_b32 v9, 2, v8 // (op 694) 

v_add_u32 v8, v10, v9 // (op 694) 


v_and_b32 v20, -2, v1 // (op 747) 
v_sub_u32 v9, v1, v20 // (op 747) 

v_lshlrev_b32 v10, 4, v9 // (op 747) 

v_lshrrev_b32 v9, 1, v1 // (op 747) 
v_lshl_add_u32 v20, v9, 5, v10 // (op 747) 


s_nop 0
v_readfirstlane_b32 s51, v20 // (op 747) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -2, v1 // (op 833) 
v_sub_u32 v9, v1, v20 // (op 833) 

v_lshlrev_b32 v10, 4, v9 // (op 833) 

v_lshrrev_b32 v9, 1, v1 // (op 833) 
v_lshl_add_u32 v20, v9, 5, v10 // (op 833) 


v_mul_lo_u32 v9, v20, 6 // (op 833) 
v_lshrrev_b32 v10, 3, v9 // (op 833) 

v_lshrrev_b32 v9, 4, v20 // (op 833) 

v_lshlrev_b32 v20, 2, v9 // (op 833) 

v_add_u32 v9, v10, v20 // (op 833) 


v_and_b32 v10, -2, v1 // (op 857) 
v_sub_u32 v20, v1, v10 // (op 857) 

v_lshlrev_b32 v10, 4, v20 // (op 857) 

v_lshrrev_b32 v20, 1, v1 // (op 857) 
v_lshl_add_u32 v12, v20, 5, v10 // (op 857) 


s_nop 0
v_readfirstlane_b32 s0, v12 // (op 857) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s1, 0 // (op 53) call()
s_ashr_i64 s[6:7], s[24:25], 7 // (op 52) 
s_mov_b32 s18, s6 // (op 52) convert

s_and_b32 s19, s18, 4294967294 // (op 52) 

s_cmp_lt_u32 s1, s19 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_52 // (op 52) Condition: Top (jump to RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_52 if false)
RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_52:
s_add_u32 s1, s1, 2 // (op 54) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 913) 



ds_read_b96_tr_b6 v[32:34], v0 offset:8192 // (op 62) Transpose load local data 
ds_read_b96_tr_b6 v[35:37], v0 offset:8704 // (op 62) Transpose load local data 
s_barrier  // (op 926) 



ds_read_b64_tr_b8 v[38:39], v4 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:256 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:2048 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:2304 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[38:45], a[0:3], v22, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 78) 



v_mov_b64 v[12:13], s[30:31] // (op 72) convert

v_mov_b32 v10, v16 // (op 72) 
s_lshl_b32 s6, s30, 3 // (op 72) 
v_add_u32 v20, v10, s6 // (op 72) 

v_mov_b32 v17, v20 // (op 72) 
buffer_load_ubyte v22, v17, s[60:63], 0 offen offset: 0 // (op 72) Load value




v_mov_b64 v[12:13], s[36:37] // (op 76) convert

v_mov_b32 v20, v19 // (op 76) 
s_lshl_b32 s6, s36, 3 // (op 76) 
v_add_u32 v17, v20, s6 // (op 76) 

v_mov_b32 v18, v17 // (op 76) 
buffer_load_ubyte v23, v18, s[64:67], 0 offen offset: 0 // (op 76) Load value



s_barrier  // (op 929) 


v_mov_b64 v[12:13], v[14:15] // (op 385) 
s_lshl_b64 s[6:7], s[26:27], 7 // (op 385) 
s_lshl_b64 s[18:19], s[6:7], 1 // (op 385) 

v_mov_b32 v17, s18 // (op 385) 
v_mov_b32 v18, s19 // (op 385) 
v_add_co_u32 v32, s[6:7], v12, v17 // (op 385) least significant half
v_addc_co_u32 v33, s[6:7], v13, v18, s[6:7] // (op 385) most significant half





s_mov_b32 m0, s51 // (op 385) reg expression
buffer_load_dwordx4 v32, s[56:59], 0 offen offset: 0 lds // (op 385) Load value direct to lds



v_mov_b32 v17, v11 // (op 4) 
s_lshl_b64 s[6:7], s[2:3], 7 // (op 4) 
s_mul_i32 s19, s7, 6 // (op 4) most significant: low of high * low
s_mul_i32 s18, s6, 6 // (op 4) least significant
s_mul_hi_u32 s68, s6, 6 // (op 4) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 4) most significant: sum


s_lshr_b64 s[6:7], s[18:19], 3 // (op 4) 

s_lshl_b32 s18, s6, 1 // (op 4) 

v_add_u32 v18, v17, s18 // (op 4) 


buffer_load_dwordx3 v[24:26], v18, s[52:55], 0 offen // (op 4) Load value

s_barrier  // (op 942) 



s_waitcnt vmcnt(4)
ds_write_b96 v9, v[28:30] offset:12288 // (op 212) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 939) 



ds_read_b96_tr_b6 v[32:34], v3 offset:12288 // (op 214) Transpose load local data 
ds_read_b96_tr_b6 v[35:37], v3 offset:12800 // (op 214) Transpose load local data 
s_barrier  // (op 945) 
s_barrier  // (op 995) 



ds_read_b64_tr_b8 v[38:39], v6 offset:4096 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v6 offset:4352 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v6 offset:6144 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v6 offset:6400 // (op 216) Transpose load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[38:45], a[0:3], v27, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 221) 



v_mov_b64 v[32:33], s[30:31] // (op 218) convert

s_lshl_b32 s6, s30, 2 // (op 218) 
s_mul_i32 s7, 3, s6 // (op 218) 

v_add_u32 v18, v10, s7 // (op 218) 

v_mov_b32 v31, v18 // (op 218) 
buffer_load_ubyte v27, v31, s[60:63], 0 offen offset: 0 // (op 218) Load value





s_lshl_b32 s6, s30, 3 // (op 584) 
v_add_u32 v16, v16, s6 // (op 584) 

v_mov_b64 v[32:33], s[36:37] // (op 220) convert

s_lshl_b32 s6, s36, 2 // (op 220) 
s_mul_i32 s7, 3, s6 // (op 220) 

v_add_u32 v10, v20, s7 // (op 220) 

v_mov_b32 v18, v10 // (op 220) 
buffer_load_ubyte v21, v18, s[64:67], 0 offen offset: 0 // (op 220) Load value




s_barrier  // (op 948) 
s_lshl_b32 s6, s36, 3 // (op 648) 


v_add_u32 v19, v19, s6 // (op 648) 

s_lshl_b64 s[6:7], s[26:27], 7 // (op 396) 
s_mul_i32 s19, 3, s7 // (op 396) most significant: low of low * high
s_mul_i32 s18, 3, s6 // (op 396) least significant
s_mul_hi_u32 s68, 3, s6 // (op 396) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 396) most significant: sum


v_mov_b32 v10, s18 // (op 396) 
v_mov_b32 v18, s19 // (op 396) 
v_add_co_u32 v32, s[6:7], v12, v10 // (op 396) least significant half
v_addc_co_u32 v33, s[6:7], v13, v18, s[6:7] // (op 396) most significant half




s_add_u32 s6, 4096, s0 // (op 396) 

s_nop 0
s_mov_b32 m0, s6 // (op 396) reg expression
buffer_load_dwordx4 v32, s[56:59], 0 offen offset: 0 lds // (op 396) Load value direct to lds





s_lshl_b64 s[6:7], s[2:3], 7 // (op 211) 
s_mul_i32 s19, s7, 6 // (op 211) most significant: low of high * low
s_mul_i32 s18, s6, 6 // (op 211) least significant
s_mul_hi_u32 s68, s6, 6 // (op 211) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 211) most significant: sum


s_lshr_b64 s[6:7], s[18:19], 3 // (op 211) 

s_mul_i32 s18, 3, s6 // (op 211) 

v_add_u32 v10, v17, s18 // (op 211) 


s_lshl_b64 s[6:7], s[26:27], 7 // (op 520) 
s_lshl_b64 s[18:19], s[6:7], 1 // (op 520) 

v_mov_b32 v18, s18 // (op 520) 
v_mov_b32 v20, s19 // (op 520) 
v_add_co_u32 v14, s[6:7], v14, v18 // (op 520) least significant half
v_addc_co_u32 v15, s[6:7], v15, v20, s[6:7] // (op 520) most significant half




buffer_load_dwordx3 v[28:30], v10, s[52:55], 0 offen // (op 211) Load value


s_barrier  // (op 916) 
s_lshl_b64 s[6:7], s[2:3], 7 // (op 456) 



s_mul_i32 s19, s7, 6 // (op 456) most significant: low of high * low
s_mul_i32 s18, s6, 6 // (op 456) least significant
s_mul_hi_u32 s68, s6, 6 // (op 456) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 456) most significant: sum


s_lshr_b64 s[6:7], s[18:19], 3 // (op 456) 

s_lshl_b32 s18, s6, 1 // (op 456) 

v_add_u32 v11, v11, s18 // (op 456) 

s_waitcnt vmcnt(4)
ds_write_b96 v8, v[24:26] offset:8192 // (op 6) Store local data 
s_ashr_i64 s[6:7], s[24:25], 7 // (op 52) 
s_mov_b32 s18, s6 // (op 52) convert

s_and_b32 s19, s18, 4294967294 // (op 52) 

s_cmp_lt_u32 s1, s19 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_52 // (op 52) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_52 if true)
RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_52:












s_barrier  // (op 959) 
s_mul_hi_i32 s51, s16, s9 // (op 419) 
v_and_b32 v8, -2, v1 // (op 419) 
s_add_i32 s0, s51, s16 // (op 419) 

v_sub_u32 v9, v1, v8 // (op 419) 

s_ashr_i32 s1, s0, 31 // (op 419) 
v_lshlrev_b32 v8, 4, v9 // (op 419) 

s_cmp_eq_i32 s9, 0 // (op 419) 
s_mov_b32 s51, scc // (op 419) 
v_lshrrev_b32 v9, 1, v1 // (op 419) 
s_cmp_eq_i32 s51, 0 // (op 419) 
s_cselect_b32 s6, 0, -1 // (op 419) 

s_and_b32 s7, s10, 31 // (op 419) 
s_lshl_b32 s51, 1, s7 // (op 419) 
s_add_i32 s18, s51, s6 // (op 419) 


s_and_b32 s6, s1, s18 // (op 419) 


s_add_i32 s1, s0, s6 // (op 419) 


s_ashr_i32 s0, s1, s7 // (op 419) 


s_lshl_b32 s1, s10, 24 // (op 419) 
s_ashr_i32 s51, s1, 31 // (op 419) 

s_xor_b32 s1, s0, s51 // (op 419) 

s_sub_i32 s0, s1, s51 // (op 419) 


s_mul_hi_i32 s1, s0, s11 // (op 419) 
s_add_i32 s51, s1, s0 // (op 419) 

s_ashr_i32 s1, s51, 31 // (op 419) 
s_cmp_eq_i32 s11, 0 // (op 419) 
s_mov_b32 s6, scc // (op 419) 
s_cmp_eq_i32 s6, 0 // (op 419) 
s_cselect_b32 s7, 0, -1 // (op 419) 

s_and_b32 s6, s12, 31 // (op 419) 
s_lshl_b32 s18, 1, s6 // (op 419) 
s_add_i32 s19, s18, s7 // (op 419) 


s_and_b32 s7, s1, s19 // (op 419) 


s_add_i32 s1, s51, s7 // (op 419) 


s_ashr_i32 s7, s1, s6 // (op 419) 


s_lshl_b32 s1, s12, 24 // (op 419) 
s_ashr_i32 s6, s1, 31 // (op 419) 

s_xor_b32 s1, s7, s6 // (op 419) 

s_sub_i32 s7, s1, s6 // (op 419) 


s_mul_i32 s1, s0, s14 // (op 419) 

s_sub_u32 s0, s16, s1 // (op 419) 

s_mul_i32 s1, s7, s14 // (op 419) 
s_add_u32 s1, s1, s0 // (op 419) 


s_mul_hi_i32 s0, s16, s8 // (op 419) 
s_add_i32 s51, s0, s16 // (op 419) 

s_ashr_i32 s0, s51, 31 // (op 419) 
s_cmp_eq_i32 s8, 0 // (op 419) 
s_mov_b32 s6, scc // (op 419) 
s_cmp_eq_i32 s6, 0 // (op 419) 
s_cselect_b32 s7, 0, -1 // (op 419) 

s_and_b32 s6, s15, 31 // (op 419) 
s_lshl_b32 s18, 1, s6 // (op 419) 
s_add_i32 s19, s18, s7 // (op 419) 


s_and_b32 s7, s0, s19 // (op 419) 


s_add_i32 s0, s51, s7 // (op 419) 


s_ashr_i32 s7, s0, s6 // (op 419) 


s_lshl_b32 s0, s15, 24 // (op 419) 
s_ashr_i32 s6, s0, 31 // (op 419) 

s_xor_b32 s0, s7, s6 // (op 419) 

s_sub_i32 s7, s0, s6 // (op 419) 


s_add_u32 s18, s4, 32 // (op 419) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 419) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 419) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 419) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 419) 

s_mul_i32 s0, s14, s18 // (op 419) 

s_mul_i32 s6, s50, s0 // (op 419) 

s_mul_i32 s0, s7, s6 // (op 419) 

s_sub_u32 s6, s16, s0 // (op 419) 

s_mov_b32 s0, s6 // (op 419) 
s_mul_hi_i32 s51, s0, s13 // (op 419) 
s_add_i32 s18, s51, s0 // (op 419) 


s_ashr_i32 s0, s18, 31 // (op 419) 
s_cmp_eq_i32 s13, 0 // (op 419) 
s_mov_b32 s19, scc // (op 419) 
s_cmp_eq_i32 s19, 0 // (op 419) 
s_cselect_b32 s51, 0, -1 // (op 419) 

s_and_b32 s19, s49, 31 // (op 419) 
s_lshl_b32 s68, 1, s19 // (op 419) 
s_add_i32 s69, s68, s51 // (op 419) 


s_and_b32 s51, s0, s69 // (op 419) 


s_add_i32 s0, s18, s51 // (op 419) 


s_ashr_i32 s18, s0, s19 // (op 419) 


s_lshl_b32 s0, s49, 24 // (op 419) 
s_ashr_i32 s19, s0, 31 // (op 419) 

s_xor_b32 s0, s18, s19 // (op 419) 

s_sub_i32 s18, s0, s19 // (op 419) 


s_mul_i32 s0, s48, s14 // (op 419) 
s_add_u32 s68, s20, 32 // (op 419) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 419) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 419) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 419) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 419) 

s_sub_i32 s19, s68, s0 // (op 419) 

s_mul_i32 s51, s18, s19 // (op 419) 


s_sub_u32 s18, s6, s51 // (op 419) 


s_add_u32 s6, s18, s0 // (op 419) 


s_cmp_eq_i32 s7, 0 // (op 419) 
s_mov_b32 s0, scc // (op 419) 

s_cmp_eq_i32 s0, 0 // (op 419) 
s_cselect_b32 s7, s6, s1 // (op 419) 



v_lshl_add_u32 v10, s7, 5, v8 // (op 419) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 419) 
v_mov_b32 v23, 0 // (op 419) convert
v_mov_b32 v22, v10 // (op 419) convert

s_lshr_b32 s51, s0, 1 // (op 419) 

v_lshl_add_u32 v8, s51, 8, v9 // (op 419) 


v_mul_lo_u32 v25, v8, s3 // (op 419) most significant: low of low * high
v_mul_hi_u32 v26, v8, s2 // (op 419) most significant: high of low * low
v_mul_lo_u32 v24, v8, s2 // (op 419) least significant: low of low * low
v_add_u32 v25, v25, v26 // (op 419) most significant: sum


v_add_co_u32 v8, s[0:1], v22, v24 // (op 419) least significant half
v_addc_co_u32 v9, s[0:1], v23, v25, s[0:1] // (op 419) most significant half



v_mul_lo_u32 v11, v9, 6 // (op 419) most significant: low of high * low
v_mul_hi_u32 v12, v8, 6 // (op 419) most significant: high of low * low
v_mul_lo_u32 v10, v8, 6 // (op 419) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 419) most significant: sum


v_lshrrev_b64 v[8:9], 3, v[10:11] // (op 419) 

v_mov_b32 v10, v8 // (op 419) convert

s_mul_i32 s1, s25, 6 // (op 421) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 421) least significant
s_mul_hi_u32 s51, s24, 6 // (op 421) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 421) most significant: sum

s_mul_i32 s7, s2, s1 // (op 421) most significant: low of low * high
s_mul_i32 s51, s3, s0 // (op 421) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 421) least significant
s_mul_hi_u32 s18, s2, s0 // (op 421) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 421) most significant: sum
s_add_u32 s7, s7, s18 // (op 421) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 421) 

s_mov_b32 s52, s22 // (op 421) reg expression
s_mov_b32 s53, s23 // (op 421) reg expression
s_mov_b32 s54, s0 // (op 421) reg expression
s_mov_b32 s55, 131072 // (op 421) call()

s_mul_hi_i32 s51, s16, s9 // (op 486) 
v_and_b32 v8, -2, v1 // (op 486) 
s_add_i32 s0, s51, s16 // (op 486) 

v_sub_u32 v9, v1, v8 // (op 486) 

s_ashr_i32 s1, s0, 31 // (op 486) 
v_lshlrev_b32 v8, 4, v9 // (op 486) 

s_cmp_eq_i32 s9, 0 // (op 486) 
s_mov_b32 s51, scc // (op 486) 
v_lshrrev_b32 v9, 1, v1 // (op 486) 
s_cmp_eq_i32 s51, 0 // (op 486) 
s_cselect_b32 s6, 0, -1 // (op 486) 

s_and_b32 s7, s10, 31 // (op 486) 
s_lshl_b32 s51, 1, s7 // (op 486) 
s_add_i32 s18, s51, s6 // (op 486) 


s_and_b32 s6, s1, s18 // (op 486) 


s_add_i32 s1, s0, s6 // (op 486) 


s_ashr_i32 s0, s1, s7 // (op 486) 


s_lshl_b32 s1, s10, 24 // (op 486) 
s_ashr_i32 s51, s1, 31 // (op 486) 

s_xor_b32 s1, s0, s51 // (op 486) 

s_sub_i32 s0, s1, s51 // (op 486) 


s_mul_hi_i32 s1, s0, s11 // (op 486) 
s_add_i32 s51, s1, s0 // (op 486) 

s_ashr_i32 s1, s51, 31 // (op 486) 
s_cmp_eq_i32 s11, 0 // (op 486) 
s_mov_b32 s6, scc // (op 486) 
s_cmp_eq_i32 s6, 0 // (op 486) 
s_cselect_b32 s7, 0, -1 // (op 486) 

s_and_b32 s6, s12, 31 // (op 486) 
s_lshl_b32 s18, 1, s6 // (op 486) 
s_add_i32 s19, s18, s7 // (op 486) 


s_and_b32 s7, s1, s19 // (op 486) 


s_add_i32 s1, s51, s7 // (op 486) 


s_ashr_i32 s7, s1, s6 // (op 486) 


s_lshl_b32 s1, s12, 24 // (op 486) 
s_ashr_i32 s6, s1, 31 // (op 486) 

s_xor_b32 s1, s7, s6 // (op 486) 

s_sub_i32 s7, s1, s6 // (op 486) 


s_add_u32 s18, s4, 32 // (op 486) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 486) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 486) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 486) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 486) 

s_mul_i32 s1, s7, s18 // (op 486) 

s_sub_i32 s51, s0, s1 // (op 486) 


s_mul_hi_i32 s0, s16, s8 // (op 486) 
s_add_i32 s1, s0, s16 // (op 486) 

s_ashr_i32 s0, s1, 31 // (op 486) 
s_cmp_eq_i32 s8, 0 // (op 486) 
s_mov_b32 s6, scc // (op 486) 
s_cmp_eq_i32 s6, 0 // (op 486) 
s_cselect_b32 s7, 0, -1 // (op 486) 

s_and_b32 s6, s15, 31 // (op 486) 
s_lshl_b32 s68, 1, s6 // (op 486) 
s_add_i32 s69, s68, s7 // (op 486) 


s_and_b32 s7, s0, s69 // (op 486) 


s_add_i32 s0, s1, s7 // (op 486) 


s_ashr_i32 s1, s0, s6 // (op 486) 


s_lshl_b32 s0, s15, 24 // (op 486) 
s_ashr_i32 s6, s0, 31 // (op 486) 

s_xor_b32 s0, s1, s6 // (op 486) 

s_sub_i32 s1, s0, s6 // (op 486) 


s_mul_i32 s0, s14, s18 // (op 486) 

s_mul_i32 s6, s50, s0 // (op 486) 

s_mul_i32 s0, s1, s6 // (op 486) 

s_sub_u32 s6, s16, s0 // (op 486) 

s_mul_hi_i32 s0, s6, s13 // (op 486) 
s_add_i32 s7, s0, s6 // (op 486) 


s_ashr_i32 s0, s7, 31 // (op 486) 
s_cmp_eq_i32 s13, 0 // (op 486) 
s_mov_b32 s6, scc // (op 486) 
s_cmp_eq_i32 s6, 0 // (op 486) 
s_cselect_b32 s18, 0, -1 // (op 486) 

s_and_b32 s6, s49, 31 // (op 486) 
s_lshl_b32 s19, 1, s6 // (op 486) 
s_add_i32 s68, s19, s18 // (op 486) 


s_and_b32 s18, s0, s68 // (op 486) 


s_add_i32 s0, s7, s18 // (op 486) 


s_ashr_i32 s7, s0, s6 // (op 486) 


s_lshl_b32 s0, s49, 24 // (op 486) 
s_ashr_i32 s6, s0, 31 // (op 486) 

s_xor_b32 s0, s7, s6 // (op 486) 

s_sub_i32 s7, s0, s6 // (op 486) 


s_cmp_eq_i32 s1, 0 // (op 486) 
s_mov_b32 s0, scc // (op 486) 

s_cmp_eq_i32 s0, 0 // (op 486) 
s_cselect_b32 s1, s7, s51 // (op 486) 



v_lshl_add_u32 v11, s1, 5, v8 // (op 486) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 486) 
v_mov_b32 v13, 0 // (op 486) convert
v_mov_b32 v12, v11 // (op 486) convert

s_lshr_b32 s51, s0, 1 // (op 486) 

v_lshl_add_u32 v8, s51, 8, v9 // (op 486) 


v_mul_lo_u32 v15, v8, s27 // (op 486) most significant: low of low * high
v_mul_hi_u32 v9, v8, s26 // (op 486) most significant: high of low * low
v_mul_lo_u32 v14, v8, s26 // (op 486) least significant: low of low * low
v_add_u32 v15, v15, v9 // (op 486) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 486) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 486) most significant half



s_mul_i32 s1, s26, s25 // (op 488) most significant: low of low * high
s_mul_i32 s51, s27, s24 // (op 488) most significant: low of high * low
s_mul_i32 s0, s26, s24 // (op 488) least significant
s_mul_hi_u32 s6, s26, s24 // (op 488) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 488) most significant: sum
s_add_u32 s1, s1, s6 // (op 488) most significant: sum


s_mov_b32 s56, s28 // (op 488) reg expression
s_mov_b32 s57, s29 // (op 488) reg expression
s_mov_b32 s58, s0 // (op 488) reg expression
s_mov_b32 s59, 131072 // (op 488) call()

s_mul_hi_i32 s51, s16, s9 // (op 550) 
v_lshrrev_b32 v8, 7, v1 // (op 550) 
s_add_i32 s0, s51, s16 // (op 550) 

v_and_b32 v9, 15, v1 // (op 550) 
s_ashr_i32 s1, s0, 31 // (op 550) 
v_lshl_add_u32 v11, v8, 4, v9 // (op 550) 


s_cmp_eq_i32 s9, 0 // (op 550) 
s_mov_b32 s51, scc // (op 550) 
v_and_b32 v8, 63, v1 // (op 550) 
s_cmp_eq_i32 s51, 0 // (op 550) 
s_cselect_b32 s6, 0, -1 // (op 550) 

v_lshrrev_b32 v9, 4, v8 // (op 550) 

s_and_b32 s7, s10, 31 // (op 550) 
s_lshl_b32 s51, 1, s7 // (op 550) 
s_add_i32 s18, s51, s6 // (op 550) 


s_and_b32 s6, s1, s18 // (op 550) 


s_add_i32 s1, s0, s6 // (op 550) 


s_ashr_i32 s0, s1, s7 // (op 550) 


s_lshl_b32 s1, s10, 24 // (op 550) 
s_ashr_i32 s51, s1, 31 // (op 550) 

s_xor_b32 s1, s0, s51 // (op 550) 

s_sub_i32 s0, s1, s51 // (op 550) 


s_mul_hi_i32 s1, s0, s11 // (op 550) 
s_add_i32 s51, s1, s0 // (op 550) 

s_ashr_i32 s1, s51, 31 // (op 550) 
s_cmp_eq_i32 s11, 0 // (op 550) 
s_mov_b32 s6, scc // (op 550) 
s_cmp_eq_i32 s6, 0 // (op 550) 
s_cselect_b32 s7, 0, -1 // (op 550) 

s_and_b32 s6, s12, 31 // (op 550) 
s_lshl_b32 s18, 1, s6 // (op 550) 
s_add_i32 s19, s18, s7 // (op 550) 


s_and_b32 s7, s1, s19 // (op 550) 


s_add_i32 s1, s51, s7 // (op 550) 


s_ashr_i32 s7, s1, s6 // (op 550) 


s_lshl_b32 s1, s12, 24 // (op 550) 
s_ashr_i32 s6, s1, 31 // (op 550) 

s_xor_b32 s1, s7, s6 // (op 550) 

s_sub_i32 s7, s1, s6 // (op 550) 


s_mul_i32 s1, s0, s14 // (op 550) 

s_sub_u32 s0, s16, s1 // (op 550) 

s_mul_i32 s1, s7, s14 // (op 550) 
s_add_u32 s1, s1, s0 // (op 550) 


s_mul_hi_i32 s0, s16, s8 // (op 550) 
s_add_i32 s51, s0, s16 // (op 550) 

s_ashr_i32 s0, s51, 31 // (op 550) 
s_cmp_eq_i32 s8, 0 // (op 550) 
s_mov_b32 s6, scc // (op 550) 
s_cmp_eq_i32 s6, 0 // (op 550) 
s_cselect_b32 s7, 0, -1 // (op 550) 

s_and_b32 s6, s15, 31 // (op 550) 
s_lshl_b32 s18, 1, s6 // (op 550) 
s_add_i32 s19, s18, s7 // (op 550) 


s_and_b32 s7, s0, s19 // (op 550) 


s_add_i32 s0, s51, s7 // (op 550) 


s_ashr_i32 s7, s0, s6 // (op 550) 


s_lshl_b32 s0, s15, 24 // (op 550) 
s_ashr_i32 s6, s0, 31 // (op 550) 

s_xor_b32 s0, s7, s6 // (op 550) 

s_sub_i32 s7, s0, s6 // (op 550) 


s_add_u32 s18, s4, 32 // (op 550) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 550) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 550) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 550) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 550) 

s_mul_i32 s0, s14, s18 // (op 550) 

s_mul_i32 s6, s50, s0 // (op 550) 

s_mul_i32 s0, s7, s6 // (op 550) 

s_sub_u32 s6, s16, s0 // (op 550) 

s_mov_b32 s0, s6 // (op 550) 
s_mul_hi_i32 s51, s0, s13 // (op 550) 
s_add_i32 s18, s51, s0 // (op 550) 


s_ashr_i32 s0, s18, 31 // (op 550) 
s_cmp_eq_i32 s13, 0 // (op 550) 
s_mov_b32 s19, scc // (op 550) 
s_cmp_eq_i32 s19, 0 // (op 550) 
s_cselect_b32 s51, 0, -1 // (op 550) 

s_and_b32 s19, s49, 31 // (op 550) 
s_lshl_b32 s68, 1, s19 // (op 550) 
s_add_i32 s69, s68, s51 // (op 550) 


s_and_b32 s51, s0, s69 // (op 550) 


s_add_i32 s0, s18, s51 // (op 550) 


s_ashr_i32 s18, s0, s19 // (op 550) 


s_lshl_b32 s0, s49, 24 // (op 550) 
s_ashr_i32 s19, s0, 31 // (op 550) 

s_xor_b32 s0, s18, s19 // (op 550) 

s_sub_i32 s18, s0, s19 // (op 550) 


s_mul_i32 s0, s48, s14 // (op 550) 
s_add_u32 s68, s20, 32 // (op 550) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 550) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 550) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 550) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 550) 

s_sub_i32 s19, s68, s0 // (op 550) 

s_mul_i32 s51, s18, s19 // (op 550) 


s_sub_u32 s18, s6, s51 // (op 550) 


s_add_u32 s6, s18, s0 // (op 550) 


s_cmp_eq_i32 s7, 0 // (op 550) 
s_mov_b32 s0, scc // (op 550) 

s_cmp_eq_i32 s0, 0 // (op 550) 
s_cselect_b32 s7, s6, s1 // (op 550) 



v_lshl_add_u32 v8, s7, 5, v11 // (op 550) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 550) 
s_lshr_b32 s51, s0, 1 // (op 550) 

v_lshl_add_u32 v11, s51, 3, v9 // (op 550) 


v_mul_lo_u32 v9, v11, s30 // (op 550) 

v_add_u32 v11, v8, v9 // (op 550) 


s_mul_i32 s1, s30, s35 // (op 552) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 552) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 552) least significant
s_mul_hi_u32 s6, s30, s34 // (op 552) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 552) most significant: sum
s_add_u32 s1, s1, s6 // (op 552) most significant: sum


s_mov_b32 s60, s32 // (op 552) reg expression
s_mov_b32 s61, s33 // (op 552) reg expression
s_mov_b32 s62, s0 // (op 552) reg expression
s_mov_b32 s63, 131072 // (op 552) call()

s_mul_hi_i32 s51, s16, s9 // (op 614) 
v_lshrrev_b32 v8, 6, v1 // (op 614) 
s_add_i32 s0, s51, s16 // (op 614) 

v_and_b32 v9, 1, v8 // (op 614) 

s_ashr_i32 s1, s0, 31 // (op 614) 
v_and_b32 v8, 15, v1 // (op 614) 
s_cmp_eq_i32 s9, 0 // (op 614) 
s_mov_b32 s51, scc // (op 614) 
v_lshl_add_u32 v12, v9, 4, v8 // (op 614) 


s_cmp_eq_i32 s51, 0 // (op 614) 
s_cselect_b32 s6, 0, -1 // (op 614) 

v_and_b32 v13, 63, v1 // (op 614) 
s_and_b32 s7, s10, 31 // (op 614) 
v_lshrrev_b32 v8, 4, v13 // (op 614) 

s_lshl_b32 s51, 1, s7 // (op 614) 
s_add_i32 s18, s51, s6 // (op 614) 


s_and_b32 s6, s1, s18 // (op 614) 


s_add_i32 s1, s0, s6 // (op 614) 


s_ashr_i32 s0, s1, s7 // (op 614) 


s_lshl_b32 s1, s10, 24 // (op 614) 
s_ashr_i32 s51, s1, 31 // (op 614) 

s_xor_b32 s1, s0, s51 // (op 614) 

s_sub_i32 s0, s1, s51 // (op 614) 


s_mul_hi_i32 s1, s0, s11 // (op 614) 
s_add_i32 s51, s1, s0 // (op 614) 

s_ashr_i32 s1, s51, 31 // (op 614) 
s_cmp_eq_i32 s11, 0 // (op 614) 
s_mov_b32 s6, scc // (op 614) 
s_cmp_eq_i32 s6, 0 // (op 614) 
s_cselect_b32 s7, 0, -1 // (op 614) 

s_and_b32 s6, s12, 31 // (op 614) 
s_lshl_b32 s18, 1, s6 // (op 614) 
s_add_i32 s19, s18, s7 // (op 614) 


s_and_b32 s7, s1, s19 // (op 614) 


s_add_i32 s1, s51, s7 // (op 614) 


s_ashr_i32 s7, s1, s6 // (op 614) 


s_lshl_b32 s1, s12, 24 // (op 614) 
s_ashr_i32 s6, s1, 31 // (op 614) 

s_xor_b32 s1, s7, s6 // (op 614) 

s_sub_i32 s7, s1, s6 // (op 614) 


s_add_u32 s18, s4, 32 // (op 614) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 614) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 614) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 614) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 614) 

s_mul_i32 s1, s7, s18 // (op 614) 

s_sub_i32 s51, s0, s1 // (op 614) 


s_mul_hi_i32 s0, s16, s8 // (op 614) 
s_add_i32 s1, s0, s16 // (op 614) 

s_ashr_i32 s0, s1, 31 // (op 614) 
s_cmp_eq_i32 s8, 0 // (op 614) 
s_mov_b32 s6, scc // (op 614) 
s_cmp_eq_i32 s6, 0 // (op 614) 
s_cselect_b32 s7, 0, -1 // (op 614) 

s_and_b32 s6, s15, 31 // (op 614) 
s_lshl_b32 s68, 1, s6 // (op 614) 
s_add_i32 s69, s68, s7 // (op 614) 


s_and_b32 s7, s0, s69 // (op 614) 


s_add_i32 s0, s1, s7 // (op 614) 


s_ashr_i32 s1, s0, s6 // (op 614) 


s_lshl_b32 s0, s15, 24 // (op 614) 
s_ashr_i32 s6, s0, 31 // (op 614) 

s_xor_b32 s0, s1, s6 // (op 614) 

s_sub_i32 s1, s0, s6 // (op 614) 


s_mul_i32 s0, s14, s18 // (op 614) 

s_mul_i32 s6, s50, s0 // (op 614) 

s_mul_i32 s0, s1, s6 // (op 614) 

s_sub_u32 s6, s16, s0 // (op 614) 

s_mul_hi_i32 s0, s6, s13 // (op 614) 
s_add_i32 s7, s0, s6 // (op 614) 


s_ashr_i32 s0, s7, 31 // (op 614) 
s_cmp_eq_i32 s13, 0 // (op 614) 
s_mov_b32 s6, scc // (op 614) 
s_cmp_eq_i32 s6, 0 // (op 614) 
s_cselect_b32 s18, 0, -1 // (op 614) 

s_and_b32 s6, s49, 31 // (op 614) 
s_lshl_b32 s19, 1, s6 // (op 614) 
s_add_i32 s68, s19, s18 // (op 614) 


s_and_b32 s18, s0, s68 // (op 614) 


s_add_i32 s0, s7, s18 // (op 614) 


s_ashr_i32 s7, s0, s6 // (op 614) 


s_lshl_b32 s0, s49, 24 // (op 614) 
s_ashr_i32 s6, s0, 31 // (op 614) 

s_xor_b32 s0, s7, s6 // (op 614) 

s_sub_i32 s7, s0, s6 // (op 614) 


s_cmp_eq_i32 s1, 0 // (op 614) 
s_mov_b32 s0, scc // (op 614) 

s_cmp_eq_i32 s0, 0 // (op 614) 
s_cselect_b32 s1, s7, s51 // (op 614) 



v_lshl_add_u32 v9, s1, 5, v12 // (op 614) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 614) 
s_lshr_b32 s51, s0, 1 // (op 614) 

v_lshl_add_u32 v12, s51, 3, v8 // (op 614) 


v_mul_lo_u32 v8, v12, s36 // (op 614) 
v_add_u32 v8, v8, v9 // (op 614) 


s_mul_i32 s1, s36, s35 // (op 616) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 616) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 616) least significant
s_mul_hi_u32 s6, s36, s34 // (op 616) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 616) most significant: sum
s_add_u32 s1, s1, s6 // (op 616) most significant: sum


s_mov_b32 s64, s38 // (op 616) reg expression
s_mov_b32 s65, s39 // (op 616) reg expression
s_mov_b32 s66, s0 // (op 616) reg expression
s_mov_b32 s67, 131072 // (op 616) call()

v_and_b32 v9, -2, v1 // (op 707) 
v_sub_u32 v12, v1, v9 // (op 707) 

v_lshlrev_b32 v9, 4, v12 // (op 707) 

v_lshrrev_b32 v12, 1, v1 // (op 707) 
v_lshl_add_u32 v13, v12, 5, v9 // (op 707) 


v_mul_lo_u32 v9, v13, 6 // (op 707) 
v_lshrrev_b32 v12, 3, v9 // (op 707) 

v_lshrrev_b32 v9, 4, v13 // (op 707) 

v_lshlrev_b32 v13, 2, v9 // (op 707) 

v_add_u32 v9, v12, v13 // (op 707) 


v_and_b32 v12, -2, v1 // (op 760) 
v_sub_u32 v13, v1, v12 // (op 760) 

v_lshlrev_b32 v12, 4, v13 // (op 760) 

v_lshrrev_b32 v13, 1, v1 // (op 760) 
v_lshl_add_u32 v16, v13, 5, v12 // (op 760) 


s_nop 0
v_readfirstlane_b32 s51, v16 // (op 760) Wait state hazard: VALU Write Readlane Hazard

s_ashr_i64 s[0:1], s[24:25], 7 // (op 157) 
s_and_b32 s6, s0, 4294967294 // (op 157) 

s_ashr_i64 s[0:1], s[24:25], 7 // (op 156) 
s_mov_b32 s7, s0 // (op 156) convert

s_cmp_lt_u32 s6, s7 // (op 156) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_156 // (op 156) Condition: Top (jump to RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_156 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_156:


s_add_u32 s6, s6, 1 // (op 158) 
v_mov_b32 v12, v10 // (op 161) 
v_mov_b32 v13, v12 // (op 161) reg expression

buffer_load_dwordx3 v[16:18], v13, s[52:55], 0 offen // (op 161) Load value


s_lshl_b64 s[0:1], s[2:3], 7 // (op 409) 
s_mul_i32 s19, s1, 6 // (op 409) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 409) least significant
s_mul_hi_u32 s7, s0, 6 // (op 409) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 409) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 409) 

v_add_u32 v10, v10, s0 // (op 409) 

s_barrier  // (op 919) 
s_barrier  // (op 932) 





v_mov_b64 v[12:13], v[14:15] // (op 390) 
v_mov_b64 v[20:21], v[12:13] // (op 390) reg expression

s_mov_b32 m0, s51 // (op 390) reg expression
buffer_load_dwordx4 v20, s[56:59], 0 offen offset: 0 lds // (op 390) Load value direct to lds


s_lshl_b64 s[0:1], s[26:27], 7 // (op 478) 
v_mov_b32 v12, s0 // (op 478) 
v_mov_b32 v13, s1 // (op 478) 
v_add_co_u32 v14, s[18:19], v14, v12 // (op 478) least significant half
v_addc_co_u32 v15, s[18:19], v15, v13, s[18:19] // (op 478) most significant half




s_waitcnt vmcnt(1)
ds_write_b96 v9, v[16:18] offset:8192 // (op 162) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 922) 



ds_read_b96_tr_b6 v[16:18], v0 offset:8192 // (op 164) Transpose load local data 
ds_read_b96_tr_b6 v[19:21], v0 offset:8704 // (op 164) Transpose load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 935) 



ds_read_b64_tr_b8 v[22:23], v4 // (op 166) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v4 offset:256 // (op 166) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v4 offset:2048 // (op 166) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v4 offset:2304 // (op 166) Transpose load local data 

v_mov_b64 v[12:13], s[30:31] // (op 168) convert

v_mov_b32 v30, v11 // (op 168) 
v_mov_b32 v31, v30 // (op 168) reg expression
v_mov_b32 v32, v31 // (op 168) 
buffer_load_ubyte v33, v32, s[60:63], 0 offen offset: 0 // (op 168) Load value




s_lshl_b32 s7, s30, 2 // (op 542) 
v_add_u32 v11, v11, s7 // (op 542) 


v_mov_b64 v[12:13], s[36:37] // (op 170) convert

v_mov_b32 v30, v8 // (op 170) 
v_mov_b32 v31, v30 // (op 170) reg expression
v_mov_b32 v32, v31 // (op 170) 
buffer_load_ubyte v34, v32, s[64:67], 0 offen offset: 0 // (op 170) Load value




s_lshl_b32 s7, s36, 2 // (op 606) 
v_add_u32 v8, v8, s7 // (op 606) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:21], v[22:29], a[0:3], v33, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 171) 




s_ashr_i64 s[0:1], s[24:25], 7 // (op 156) 
s_mov_b32 s7, s0 // (op 156) convert

s_cmp_lt_u32 s6, s7 // (op 156) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_156 // (op 156) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_156 if true)
RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_156:





















s_barrier  // (op 989) 




v_cmp_eq_f32 vcc, 0.00000, v5 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_116
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

s_mul_hi_i32 s0, s16, s9 // (op 801) 
v_lshrrev_b32 v0, 7, v1 // (op 801) 
s_add_i32 s1, s0, s16 // (op 801) 

v_and_b32 v6, 63, v1 // (op 801) 
s_ashr_i32 s0, s1, 31 // (op 801) 
v_lshrrev_b32 v3, 4, v6 // (op 801) 
s_cmp_eq_i32 s9, 0 // (op 801) 
s_mov_b32 s2, scc // (op 801) 
v_lshlrev_b32 v4, 2, v3 // (op 801) 

s_cmp_eq_i32 s2, 0 // (op 801) 
s_cselect_b32 s3, 0, -1 // (op 801) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 801) 


s_and_b32 s2, s10, 31 // (op 801) 
v_lshrrev_b32 v0, 2, v6 // (op 801) 

s_lshl_b32 s6, 1, s2 // (op 801) 
v_and_b32 v4, 3, v0 // (op 801) 

s_add_i32 s7, s6, s3 // (op 801) 


v_and_b32 v0, 3, v1 // (op 801) 
s_and_b32 s3, s0, s7 // (op 801) 


v_lshl_add_u32 v6, v4, 2, v0 // (op 801) 


s_add_i32 s0, s1, s3 // (op 801) 


v_lshrrev_b32 v0, 6, v1 // (op 801) 
s_ashr_i32 s1, s0, s2 // (op 801) 


v_and_b32 v4, 1, v0 // (op 801) 

s_lshl_b32 s0, s10, 24 // (op 801) 
v_lshl_add_u32 v0, v4, 4, v6 // (op 801) 


s_ashr_i32 s2, s0, 31 // (op 801) 

s_xor_b32 s0, s1, s2 // (op 801) 

s_sub_i32 s1, s0, s2 // (op 801) 


s_mul_hi_i32 s0, s1, s11 // (op 801) 
s_add_i32 s2, s0, s1 // (op 801) 

s_ashr_i32 s0, s2, 31 // (op 801) 
s_cmp_eq_i32 s11, 0 // (op 801) 
s_mov_b32 s3, scc // (op 801) 
s_cmp_eq_i32 s3, 0 // (op 801) 
s_cselect_b32 s6, 0, -1 // (op 801) 

s_and_b32 s3, s12, 31 // (op 801) 
s_lshl_b32 s7, 1, s3 // (op 801) 
s_add_i32 s18, s7, s6 // (op 801) 


s_and_b32 s19, s0, s18 // (op 801) 


s_add_i32 s0, s2, s19 // (op 801) 


s_ashr_i32 s2, s0, s3 // (op 801) 


s_lshl_b32 s0, s12, 24 // (op 801) 
s_ashr_i32 s3, s0, 31 // (op 801) 

s_xor_b32 s0, s2, s3 // (op 801) 

s_sub_i32 s2, s0, s3 // (op 801) 


s_mul_i32 s0, s1, s14 // (op 801) 
s_sub_u32 s3, s16, s0 // (op 801) 

s_mul_i32 s0, s2, s14 // (op 801) 
s_add_u32 s0, s0, s3 // (op 801) 

s_mul_hi_i32 s3, s16, s8 // (op 801) 
s_add_i32 s6, s3, s16 // (op 801) 

s_ashr_i32 s3, s6, 31 // (op 801) 
s_cmp_eq_i32 s8, 0 // (op 801) 
s_mov_b32 s7, scc // (op 801) 
s_cmp_eq_i32 s7, 0 // (op 801) 
s_cselect_b32 s18, 0, -1 // (op 801) 

s_and_b32 s7, s15, 31 // (op 801) 
s_lshl_b32 s19, 1, s7 // (op 801) 
s_add_i32 s22, s19, s18 // (op 801) 


s_and_b32 s18, s3, s22 // (op 801) 


s_add_i32 s3, s6, s18 // (op 801) 


s_ashr_i32 s6, s3, s7 // (op 801) 


s_lshl_b32 s3, s15, 24 // (op 801) 
s_ashr_i32 s7, s3, 31 // (op 801) 

s_xor_b32 s3, s6, s7 // (op 801) 

s_sub_i32 s6, s3, s7 // (op 801) 


s_add_u32 s18, s4, 32 // (op 801) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 801) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 801) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 801) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 5 // (op 801) 

s_mul_i32 s3, s14, s18 // (op 801) 
s_mul_i32 s7, s50, s3 // (op 801) 

s_mul_i32 s3, s6, s7 // (op 801) 

s_sub_u32 s7, s16, s3 // (op 801) 

s_mov_b32 s3, s7 // (op 801) 
s_mul_hi_i32 s22, s3, s13 // (op 801) 
s_add_i32 s23, s22, s3 // (op 801) 


s_ashr_i32 s3, s23, 31 // (op 801) 
s_cmp_eq_i32 s13, 0 // (op 801) 
s_mov_b32 s22, scc // (op 801) 
s_cmp_eq_i32 s22, 0 // (op 801) 
s_cselect_b32 s24, 0, -1 // (op 801) 

s_and_b32 s22, s49, 31 // (op 801) 
s_lshl_b32 s25, 1, s22 // (op 801) 
s_add_i32 s26, s25, s24 // (op 801) 


s_and_b32 s24, s3, s26 // (op 801) 


s_add_i32 s3, s23, s24 // (op 801) 


s_ashr_i32 s23, s3, s22 // (op 801) 


s_lshl_b32 s3, s49, 24 // (op 801) 
s_ashr_i32 s22, s3, 31 // (op 801) 

s_xor_b32 s3, s23, s22 // (op 801) 

s_sub_i32 s23, s3, s22 // (op 801) 


s_mul_i32 s3, s48, s14 // (op 801) 
s_add_u32 s24, s20, 32 // (op 801) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 801) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 801) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 801) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 5 // (op 801) 

s_sub_i32 s22, s24, s3 // (op 801) 

s_mul_i32 s24, s23, s22 // (op 801) 

s_sub_u32 s22, s7, s24 // (op 801) 


s_add_u32 s7, s22, s3 // (op 801) 


s_cmp_eq_i32 s6, 0 // (op 801) 
s_mov_b32 s3, scc // (op 801) 

s_cmp_eq_i32 s3, 0 // (op 801) 
s_cselect_b32 s6, s7, s0 // (op 801) 


s_mul_i32 s0, s2, s18 // (op 801) 


v_lshl_add_u32 v4, s6, 5, v3 // (op 801) 


s_sub_i32 s2, s1, s0 // (op 801) 


s_cmp_eq_i32 s3, 0 // (op 801) 
s_cselect_b32 s22, s23, s2 // (op 801) 



v_lshl_add_u32 v3, s22, 5, v0 // (op 801) 


v_mul_lo_u32 v0, v3, s40 // (op 801) 

v_add_lshl_u32 v3, v4, v0, 1 // (op 801) 


s_mul_i32 s7, s40, s5 // (op 803) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 803) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 803) least significant
s_mul_hi_u32 s1, s40, s4 // (op 803) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 803) most significant: sum
s_add_u32 s7, s7, s1 // (op 803) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 803) 

s_mov_b32 s0, s42 // (op 803) reg expression
s_mov_b32 s1, s43 // (op 803) reg expression
s_mov_b32 s2, s18 // (op 803) reg expression
s_mov_b32 s3, 131072 // (op 803) call()


v_pack_b32_f16 v12, v8, v9 // (op 50) 
v_pack_b32_f16 v13, v10, v11 // (op 50) 


buffer_store_dwordx2 v[12:13], v3, s[0:3], 0 offen // (op 50) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_116
RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_116:
s_mul_hi_i32 s6, s16, s9 // (op 677) 
v_lshrrev_b32 v0, 7, v1 // (op 677) 
s_add_i32 s7, s6, s16 // (op 677) 

v_and_b32 v6, 63, v1 // (op 677) 
s_ashr_i32 s6, s7, 31 // (op 677) 
v_lshrrev_b32 v3, 4, v6 // (op 677) 
s_cmp_eq_i32 s9, 0 // (op 677) 
s_mov_b32 s18, scc // (op 677) 
v_lshlrev_b32 v4, 2, v3 // (op 677) 

s_cmp_eq_i32 s18, 0 // (op 677) 
s_cselect_b32 s19, 0, -1 // (op 677) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 677) 


s_and_b32 s18, s10, 31 // (op 677) 
v_lshrrev_b32 v0, 2, v6 // (op 677) 

s_lshl_b32 s22, 1, s18 // (op 677) 
v_and_b32 v4, 3, v0 // (op 677) 

s_add_i32 s23, s22, s19 // (op 677) 


v_and_b32 v0, 3, v1 // (op 677) 
s_and_b32 s19, s6, s23 // (op 677) 


v_lshl_add_u32 v6, v4, 2, v0 // (op 677) 


s_add_i32 s6, s7, s19 // (op 677) 


v_lshrrev_b32 v0, 6, v1 // (op 677) 
s_ashr_i32 s7, s6, s18 // (op 677) 


v_and_b32 v4, 1, v0 // (op 677) 

s_lshl_b32 s6, s10, 24 // (op 677) 
v_lshl_add_u32 v0, v4, 4, v6 // (op 677) 


s_ashr_i32 s18, s6, 31 // (op 677) 

s_xor_b32 s6, s7, s18 // (op 677) 

s_sub_i32 s7, s6, s18 // (op 677) 


s_mul_hi_i32 s6, s7, s11 // (op 677) 
s_add_i32 s18, s6, s7 // (op 677) 

s_ashr_i32 s6, s18, 31 // (op 677) 
s_cmp_eq_i32 s11, 0 // (op 677) 
s_mov_b32 s19, scc // (op 677) 
s_cmp_eq_i32 s19, 0 // (op 677) 
s_cselect_b32 s22, 0, -1 // (op 677) 

s_and_b32 s19, s12, 31 // (op 677) 
s_lshl_b32 s23, 1, s19 // (op 677) 
s_add_i32 s24, s23, s22 // (op 677) 


s_and_b32 s22, s6, s24 // (op 677) 


s_add_i32 s6, s18, s22 // (op 677) 


s_ashr_i32 s18, s6, s19 // (op 677) 


s_lshl_b32 s6, s12, 24 // (op 677) 
s_ashr_i32 s19, s6, 31 // (op 677) 

s_xor_b32 s6, s18, s19 // (op 677) 

s_sub_i32 s18, s6, s19 // (op 677) 


s_mul_i32 s6, s7, s14 // (op 677) 
s_sub_u32 s19, s16, s6 // (op 677) 

s_mul_i32 s6, s18, s14 // (op 677) 
s_add_u32 s6, s6, s19 // (op 677) 

s_mul_hi_i32 s19, s16, s8 // (op 677) 
s_add_i32 s22, s19, s16 // (op 677) 

s_ashr_i32 s19, s22, 31 // (op 677) 
s_cmp_eq_i32 s8, 0 // (op 677) 
s_mov_b32 s23, scc // (op 677) 
s_cmp_eq_i32 s23, 0 // (op 677) 
s_cselect_b32 s24, 0, -1 // (op 677) 

s_and_b32 s23, s15, 31 // (op 677) 
s_lshl_b32 s25, 1, s23 // (op 677) 
s_add_i32 s26, s25, s24 // (op 677) 


s_and_b32 s24, s19, s26 // (op 677) 


s_add_i32 s19, s22, s24 // (op 677) 


s_ashr_i32 s22, s19, s23 // (op 677) 


s_lshl_b32 s19, s15, 24 // (op 677) 
s_ashr_i32 s23, s19, 31 // (op 677) 

s_xor_b32 s19, s22, s23 // (op 677) 

s_sub_i32 s22, s19, s23 // (op 677) 


s_add_u32 s38, s4, 32 // (op 677) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 677) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 677) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 677) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 5 // (op 677) 

s_mul_i32 s19, s14, s38 // (op 677) 
s_mul_i32 s23, s50, s19 // (op 677) 

s_mul_i32 s19, s22, s23 // (op 677) 

s_sub_u32 s23, s16, s19 // (op 677) 

s_mov_b32 s19, s23 // (op 677) 
s_mul_hi_i32 s24, s19, s13 // (op 677) 
s_add_i32 s25, s24, s19 // (op 677) 


s_ashr_i32 s19, s25, 31 // (op 677) 
s_cmp_eq_i32 s13, 0 // (op 677) 
s_mov_b32 s24, scc // (op 677) 
s_cmp_eq_i32 s24, 0 // (op 677) 
s_cselect_b32 s26, 0, -1 // (op 677) 

s_and_b32 s24, s49, 31 // (op 677) 
s_lshl_b32 s27, 1, s24 // (op 677) 
s_add_i32 s28, s27, s26 // (op 677) 


s_and_b32 s26, s19, s28 // (op 677) 


s_add_i32 s19, s25, s26 // (op 677) 


s_ashr_i32 s25, s19, s24 // (op 677) 


s_lshl_b32 s19, s49, 24 // (op 677) 
s_ashr_i32 s24, s19, 31 // (op 677) 

s_xor_b32 s19, s25, s24 // (op 677) 

s_sub_i32 s25, s19, s24 // (op 677) 


s_mul_i32 s19, s48, s14 // (op 677) 
s_add_u32 s26, s20, 32 // (op 677) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 677) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 677) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 677) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 5 // (op 677) 

s_sub_i32 s24, s26, s19 // (op 677) 

s_mul_i32 s26, s25, s24 // (op 677) 

s_sub_u32 s24, s23, s26 // (op 677) 


s_add_u32 s23, s24, s19 // (op 677) 


s_cmp_eq_i32 s22, 0 // (op 677) 
s_mov_b32 s19, scc // (op 677) 

s_cmp_eq_i32 s19, 0 // (op 677) 
s_cselect_b32 s22, s23, s6 // (op 677) 


s_mul_i32 s6, s18, s38 // (op 677) 


v_lshl_add_u32 v4, s22, 5, v3 // (op 677) 


s_sub_i32 s18, s7, s6 // (op 677) 


s_cmp_eq_i32 s19, 0 // (op 677) 
s_cselect_b32 s6, s25, s18 // (op 677) 



v_lshl_add_u32 v3, s6, 5, v0 // (op 677) 


v_mul_lo_u32 v0, v3, s44 // (op 677) 

v_add_lshl_u32 v3, v4, v0, 1 // (op 677) 


s_mul_i32 s7, s44, s5 // (op 679) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 679) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 679) least significant
s_mul_hi_u32 s19, s44, s4 // (op 679) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 679) most significant: sum
s_add_u32 s7, s7, s19 // (op 679) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 679) 

s_mov_b32 s36, s46 // (op 679) reg expression
s_mov_b32 s37, s47 // (op 679) reg expression
s_mov_b32 s38, s18 // (op 679) reg expression
s_mov_b32 s39, 131072 // (op 679) call()





buffer_load_dwordx2 v[8:9], v3, s[36:39], 0 offen // (op 129) Load value



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
v_mul_f32 v18, v5, v10 // (op 130) 
v_mul_f32 v19, v5, v11 // (op 130) 
v_and_b32 v10, 65535, v9 // (op 130) 
v_lshrrev_b32 v11, 16, v9 // (op 130) 
v_lshlrev_b32 v10, 16, v10 // (op 130) 
v_lshlrev_b32 v11, 16, v11 // (op 130) 
v_mul_f32 v20, v5, v10 // (op 130) 
v_mul_f32 v21, v5, v11 // (op 130) 


v_add_f32 v8, v18, v14 // (op 126) 
v_add_f32 v9, v19, v15 // (op 126) 
v_add_f32 v10, v20, v16 // (op 126) 
v_add_f32 v11, v21, v17 // (op 126) 


v_lshrrev_b32 v12, 16, v8 // (op 127) 
v_lshrrev_b32 v13, 16, v9 // (op 127) 
v_lshrrev_b32 v14, 16, v10 // (op 127) 
v_lshrrev_b32 v15, 16, v11 // (op 127) 

s_mul_hi_i32 s6, s16, s9 // (op 818) 
v_lshrrev_b32 v0, 7, v1 // (op 818) 
s_add_i32 s7, s6, s16 // (op 818) 

v_and_b32 v6, 63, v1 // (op 818) 
s_ashr_i32 s6, s7, 31 // (op 818) 
v_lshrrev_b32 v3, 4, v6 // (op 818) 
s_cmp_eq_i32 s9, 0 // (op 818) 
s_mov_b32 s18, scc // (op 818) 
v_lshlrev_b32 v4, 2, v3 // (op 818) 

s_cmp_eq_i32 s18, 0 // (op 818) 
s_cselect_b32 s19, 0, -1 // (op 818) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 818) 


s_and_b32 s18, s10, 31 // (op 818) 
v_lshrrev_b32 v0, 2, v6 // (op 818) 

s_lshl_b32 s22, 1, s18 // (op 818) 
v_and_b32 v4, 3, v0 // (op 818) 

s_add_i32 s23, s22, s19 // (op 818) 


v_and_b32 v0, 3, v1 // (op 818) 
s_and_b32 s19, s6, s23 // (op 818) 


v_lshl_add_u32 v6, v4, 2, v0 // (op 818) 


s_add_i32 s6, s7, s19 // (op 818) 


v_lshrrev_b32 v0, 6, v1 // (op 818) 
s_ashr_i32 s7, s6, s18 // (op 818) 


v_and_b32 v4, 1, v0 // (op 818) 

s_lshl_b32 s6, s10, 24 // (op 818) 
v_lshl_add_u32 v0, v4, 4, v6 // (op 818) 


s_ashr_i32 s18, s6, 31 // (op 818) 

s_xor_b32 s6, s7, s18 // (op 818) 

s_sub_i32 s7, s6, s18 // (op 818) 


s_mul_hi_i32 s6, s7, s11 // (op 818) 
s_add_i32 s18, s6, s7 // (op 818) 

s_ashr_i32 s6, s18, 31 // (op 818) 
s_cmp_eq_i32 s11, 0 // (op 818) 
s_mov_b32 s19, scc // (op 818) 
s_cmp_eq_i32 s19, 0 // (op 818) 
s_cselect_b32 s22, 0, -1 // (op 818) 

s_and_b32 s19, s12, 31 // (op 818) 
s_lshl_b32 s23, 1, s19 // (op 818) 
s_add_i32 s24, s23, s22 // (op 818) 


s_and_b32 s22, s6, s24 // (op 818) 


s_add_i32 s6, s18, s22 // (op 818) 


s_ashr_i32 s18, s6, s19 // (op 818) 


s_lshl_b32 s6, s12, 24 // (op 818) 
s_ashr_i32 s19, s6, 31 // (op 818) 

s_xor_b32 s6, s18, s19 // (op 818) 

s_sub_i32 s18, s6, s19 // (op 818) 


s_mul_i32 s6, s7, s14 // (op 818) 
s_sub_u32 s19, s16, s6 // (op 818) 

s_mul_i32 s6, s18, s14 // (op 818) 
s_add_u32 s6, s6, s19 // (op 818) 

s_mul_hi_i32 s19, s16, s8 // (op 818) 
s_add_i32 s22, s19, s16 // (op 818) 

s_ashr_i32 s19, s22, 31 // (op 818) 
s_cmp_eq_i32 s8, 0 // (op 818) 
s_mov_b32 s23, scc // (op 818) 
s_cmp_eq_i32 s23, 0 // (op 818) 
s_cselect_b32 s24, 0, -1 // (op 818) 

s_and_b32 s23, s15, 31 // (op 818) 
s_lshl_b32 s25, 1, s23 // (op 818) 
s_add_i32 s26, s25, s24 // (op 818) 


s_and_b32 s24, s19, s26 // (op 818) 


s_add_i32 s19, s22, s24 // (op 818) 


s_ashr_i32 s22, s19, s23 // (op 818) 


s_lshl_b32 s19, s15, 24 // (op 818) 
s_ashr_i32 s23, s19, 31 // (op 818) 

s_xor_b32 s19, s22, s23 // (op 818) 

s_sub_i32 s22, s19, s23 // (op 818) 


s_add_u32 s38, s4, 32 // (op 818) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 818) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 818) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 818) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 5 // (op 818) 

s_mul_i32 s19, s14, s38 // (op 818) 
s_mul_i32 s23, s50, s19 // (op 818) 

s_mul_i32 s19, s22, s23 // (op 818) 

s_sub_u32 s23, s16, s19 // (op 818) 

s_mov_b32 s19, s23 // (op 818) 
s_mul_hi_i32 s24, s19, s13 // (op 818) 
s_add_i32 s25, s24, s19 // (op 818) 


s_ashr_i32 s19, s25, 31 // (op 818) 
s_cmp_eq_i32 s13, 0 // (op 818) 
s_mov_b32 s24, scc // (op 818) 
s_cmp_eq_i32 s24, 0 // (op 818) 
s_cselect_b32 s26, 0, -1 // (op 818) 

s_and_b32 s24, s49, 31 // (op 818) 
s_lshl_b32 s27, 1, s24 // (op 818) 
s_add_i32 s28, s27, s26 // (op 818) 


s_and_b32 s26, s19, s28 // (op 818) 


s_add_i32 s19, s25, s26 // (op 818) 


s_ashr_i32 s25, s19, s24 // (op 818) 


s_lshl_b32 s19, s49, 24 // (op 818) 
s_ashr_i32 s24, s19, 31 // (op 818) 

s_xor_b32 s19, s25, s24 // (op 818) 

s_sub_i32 s25, s19, s24 // (op 818) 


s_mul_i32 s19, s48, s14 // (op 818) 
s_add_u32 s26, s20, 32 // (op 818) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 818) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 818) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 818) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 5 // (op 818) 

s_sub_i32 s24, s26, s19 // (op 818) 

s_mul_i32 s26, s25, s24 // (op 818) 

s_sub_u32 s24, s23, s26 // (op 818) 


s_add_u32 s23, s24, s19 // (op 818) 


s_cmp_eq_i32 s22, 0 // (op 818) 
s_mov_b32 s19, scc // (op 818) 

s_cmp_eq_i32 s19, 0 // (op 818) 
s_cselect_b32 s22, s23, s6 // (op 818) 


s_mul_i32 s6, s18, s38 // (op 818) 


v_lshl_add_u32 v4, s22, 5, v3 // (op 818) 


s_sub_i32 s18, s7, s6 // (op 818) 


s_cmp_eq_i32 s19, 0 // (op 818) 
s_cselect_b32 s6, s25, s18 // (op 818) 



v_lshl_add_u32 v3, s6, 5, v0 // (op 818) 


v_mul_lo_u32 v0, v3, s40 // (op 818) 

v_add_lshl_u32 v3, v4, v0, 1 // (op 818) 


s_mul_i32 s7, s40, s5 // (op 820) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 820) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 820) least significant
s_mul_hi_u32 s19, s40, s4 // (op 820) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 820) most significant: sum
s_add_u32 s7, s7, s19 // (op 820) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 820) 

s_mov_b32 s0, s42 // (op 820) reg expression
s_mov_b32 s1, s43 // (op 820) reg expression
s_mov_b32 s2, s18 // (op 820) reg expression
s_mov_b32 s3, 131072 // (op 820) call()


v_pack_b32_f16 v8, v12, v13 // (op 128) 
v_pack_b32_f16 v9, v14, v15 // (op 128) 


buffer_store_dwordx2 v[8:9], v3, s[0:3], 0 offen // (op 128) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_116:
















s_endpgm  // End of RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.LRR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__end:
.size RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_, .LRR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__end-RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16384 // lds bytes
  .amdhsa_accum_offset 48
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
    DataTypeB: F8
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
  - .name: RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 46
    .agpr_count: 4
    .max_flat_workgroup_size: 256
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

