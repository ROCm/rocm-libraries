// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_,@function
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 771) 
v_and_b32 v3, 15, v1 // (op 771) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 771) 


v_and_b32 v0, 63, v1 // (op 771) 
v_lshrrev_b32 v3, 4, v0 // (op 771) 

v_lshlrev_b32 v0, 5, v3 // (op 771) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 771) 


v_mul_lo_u32 v0, v3, 6 // (op 771) 

v_lshrrev_b32 v3, 3, v0 // (op 771) 

v_lshrrev_b32 v0, 6, v1 // (op 823) 
v_and_b32 v4, 1, v0 // (op 823) 

v_and_b32 v0, 15, v1 // (op 823) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 823) 


v_and_b32 v0, 48, v1 // (op 823) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 823) 


v_lshrrev_b32 v0, 6, v1 // (op 876) 
v_and_b32 v5, 1, v0 // (op 876) 

v_and_b32 v0, 15, v1 // (op 876) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 876) 


v_and_b32 v0, 63, v1 // (op 876) 
v_lshrrev_b32 v5, 4, v0 // (op 876) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 876) 


v_lshrrev_b32 v5, 7, v1 // (op 934) 
v_and_b32 v6, 15, v1 // (op 934) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 934) 


v_and_b32 v5, 63, v1 // (op 934) 
v_lshrrev_b32 v6, 4, v5 // (op 934) 

v_lshlrev_b32 v5, 5, v6 // (op 934) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 934) 


v_mul_lo_u32 v5, v6, 6 // (op 934) 

v_lshrrev_b32 v6, 3, v5 // (op 934) 

v_lshrrev_b32 v5, 6, v1 // (op 973) 
v_and_b32 v7, 1, v5 // (op 973) 

v_and_b32 v5, 15, v1 // (op 973) 
v_lshl_add_u32 v8, v7, 4, v5 // (op 973) 


v_and_b32 v5, 48, v1 // (op 973) 
v_lshl_add_u32 v7, v8, 7, v5 // (op 973) 


v_lshrrev_b32 v5, 6, v1 // (op 997) 
v_and_b32 v8, 1, v5 // (op 997) 

v_and_b32 v5, 15, v1 // (op 997) 
v_lshl_add_u32 v9, v8, 4, v5 // (op 997) 


v_and_b32 v5, 63, v1 // (op 997) 
v_lshrrev_b32 v8, 4, v5 // (op 997) 

v_lshl_add_u32 v5, v9, 2, v8 // (op 997) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v9, s9 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
s_mul_hi_i32 s0, s16, s12 // (op 481) 
v_lshrrev_b32 v10, 3, v1 // (op 481) 
s_add_i32 s1, s0, s16 // (op 481) 

v_and_b32 v11, -8, v1 // (op 481) 
s_ashr_i32 s0, s1, 31 // (op 481) 
v_sub_u32 v12, v1, v11 // (op 481) 

s_cmp_eq_i32 s12, 0 // (op 481) 
s_mov_b32 s8, scc // (op 481) 
v_lshlrev_b32 v11, 4, v12 // (op 481) 

s_cmp_eq_i32 s8, 0 // (op 481) 
s_cselect_b32 s9, 0, -1 // (op 481) 

v_mov_b32 v13, 0 // (op 481) convert
v_mov_b32 v12, v11 // (op 481) convert

s_and_b32 s8, s13, 31 // (op 481) 
s_lshl_b32 s18, 1, s8 // (op 481) 
s_add_i32 s19, s18, s9 // (op 481) 


s_and_b32 s9, s0, s19 // (op 481) 


s_add_i32 s0, s1, s9 // (op 481) 


s_ashr_i32 s1, s0, s8 // (op 481) 


s_lshl_b32 s0, s13, 24 // (op 481) 
s_ashr_i32 s8, s0, 31 // (op 481) 

s_xor_b32 s0, s1, s8 // (op 481) 

s_sub_i32 s1, s0, s8 // (op 481) 


s_mul_hi_i32 s0, s1, s14 // (op 481) 
s_add_i32 s8, s0, s1 // (op 481) 

s_ashr_i32 s0, s8, 31 // (op 481) 
s_cmp_eq_i32 s14, 0 // (op 481) 
s_mov_b32 s9, scc // (op 481) 
s_cmp_eq_i32 s9, 0 // (op 481) 
s_cselect_b32 s18, 0, -1 // (op 481) 

s_and_b32 s9, s15, 31 // (op 481) 
s_lshl_b32 s19, 1, s9 // (op 481) 
s_add_i32 s49, s19, s18 // (op 481) 


s_and_b32 s18, s0, s49 // (op 481) 


s_add_i32 s0, s8, s18 // (op 481) 


s_ashr_i32 s8, s0, s9 // (op 481) 


s_lshl_b32 s0, s15, 24 // (op 481) 
s_ashr_i32 s9, s0, 31 // (op 481) 

s_xor_b32 s0, s8, s9 // (op 481) 

s_sub_i32 s8, s0, s9 // (op 481) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s45 // (op 481) 

s_sub_u32 s1, s16, s0 // (op 481) 

s_mul_i32 s0, s8, s45 // (op 481) 
s_add_u32 s0, s0, s1 // (op 481) 


s_mul_hi_i32 s1, s16, s11 // (op 481) 
s_add_i32 s8, s1, s16 // (op 481) 

s_ashr_i32 s1, s8, 31 // (op 481) 
s_cmp_eq_i32 s11, 0 // (op 481) 
s_mov_b32 s9, scc // (op 481) 
s_cmp_eq_i32 s9, 0 // (op 481) 
s_cselect_b32 s18, 0, -1 // (op 481) 

s_and_b32 s9, s46, 31 // (op 481) 
s_lshl_b32 s19, 1, s9 // (op 481) 
s_add_i32 s49, s19, s18 // (op 481) 


s_and_b32 s18, s1, s49 // (op 481) 


s_add_i32 s1, s8, s18 // (op 481) 


s_ashr_i32 s8, s1, s9 // (op 481) 


s_lshl_b32 s1, s46, 24 // (op 481) 
s_ashr_i32 s9, s1, 31 // (op 481) 

s_xor_b32 s1, s8, s9 // (op 481) 

s_sub_i32 s8, s1, s9 // (op 481) 


s_add_u32 s18, s4, 32 // (op 481) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 481) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 481) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 481) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 481) 

s_mul_i32 s1, s45, s18 // (op 481) 

s_mul_i32 s9, s10, s1 // (op 481) 

s_mul_i32 s1, s8, s9 // (op 481) 

s_sub_u32 s9, s16, s1 // (op 481) 

s_mov_b32 s1, s9 // (op 481) 
s_mul_hi_i32 s18, s1, s44 // (op 481) 
s_add_i32 s19, s18, s1 // (op 481) 


s_ashr_i32 s1, s19, 31 // (op 481) 
s_cmp_eq_i32 s44, 0 // (op 481) 
s_mov_b32 s18, scc // (op 481) 
s_cmp_eq_i32 s18, 0 // (op 481) 
s_cselect_b32 s49, 0, -1 // (op 481) 

s_and_b32 s18, s48, 31 // (op 481) 
s_lshl_b32 s50, 1, s18 // (op 481) 
s_add_i32 s51, s50, s49 // (op 481) 


s_and_b32 s49, s1, s51 // (op 481) 


s_add_i32 s1, s19, s49 // (op 481) 


s_ashr_i32 s19, s1, s18 // (op 481) 


s_lshl_b32 s1, s48, 24 // (op 481) 
s_ashr_i32 s18, s1, 31 // (op 481) 

s_xor_b32 s1, s19, s18 // (op 481) 

s_sub_i32 s19, s1, s18 // (op 481) 


s_mul_i32 s1, s47, s45 // (op 481) 
s_add_u32 s50, s6, 16 // (op 481) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 481) End of Int64 add, unlocking SCC
s_sub_u32 s52, s50, 1 // (op 481) least significant half; sets scc
s_subb_u32 s53, s51, 0 // (op 481) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 4 // (op 481) 

s_sub_i32 s18, s50, s1 // (op 481) 

s_mul_i32 s49, s19, s18 // (op 481) 


s_sub_u32 s18, s9, s49 // (op 481) 


s_add_u32 s9, s18, s1 // (op 481) 


s_cmp_eq_i32 s8, 0 // (op 481) 
s_mov_b32 s1, scc // (op 481) 

s_cmp_eq_i32 s1, 0 // (op 481) 
s_cselect_b32 s8, s9, s0 // (op 481) 



v_lshl_add_u32 v11, s8, 4, v10 // (op 481) 


v_mul_lo_u32 v15, v11, s3 // (op 481) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 481) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 481) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 481) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 481) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 481) most significant half



v_mul_lo_u32 v11, v15, 6 // (op 481) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 481) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 481) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 481) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 481) 

v_mov_b32 v10, v12 // (op 481) convert

s_mul_i32 s1, s7, 6 // (op 483) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 483) least significant
s_mul_hi_u32 s8, s6, 6 // (op 483) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 483) most significant: sum

s_mul_i32 s9, s2, s1 // (op 483) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 483) most significant: low of high * low
s_mul_i32 s8, s2, s0 // (op 483) least significant
s_mul_hi_u32 s19, s2, s0 // (op 483) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 483) most significant: sum
s_add_u32 s9, s9, s19 // (op 483) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 483) 

s_mov_b32 s52, s20 // (op 483) reg expression
s_mov_b32 s53, s21 // (op 483) reg expression
s_mov_b32 s54, s0 // (op 483) reg expression
s_mov_b32 s55, 131072 // (op 483) call()

s_mul_hi_i32 s0, s16, s12 // (op 500) 
v_lshrrev_b32 v11, 3, v1 // (op 500) 
s_add_i32 s1, s0, s16 // (op 500) 

v_and_b32 v12, -8, v1 // (op 500) 
s_ashr_i32 s0, s1, 31 // (op 500) 
v_sub_u32 v13, v1, v12 // (op 500) 

s_cmp_eq_i32 s12, 0 // (op 500) 
s_mov_b32 s8, scc // (op 500) 
v_lshlrev_b32 v12, 4, v13 // (op 500) 

s_cmp_eq_i32 s8, 0 // (op 500) 
s_cselect_b32 s9, 0, -1 // (op 500) 

v_mov_b32 v15, 0 // (op 500) convert
v_mov_b32 v14, v12 // (op 500) convert

s_and_b32 s8, s13, 31 // (op 500) 
s_lshl_b32 s18, 1, s8 // (op 500) 
s_add_i32 s19, s18, s9 // (op 500) 


s_and_b32 s9, s0, s19 // (op 500) 


s_add_i32 s0, s1, s9 // (op 500) 


s_ashr_i32 s1, s0, s8 // (op 500) 


s_lshl_b32 s0, s13, 24 // (op 500) 
s_ashr_i32 s8, s0, 31 // (op 500) 

s_xor_b32 s0, s1, s8 // (op 500) 

s_sub_i32 s1, s0, s8 // (op 500) 


s_mul_hi_i32 s0, s1, s14 // (op 500) 
s_add_i32 s8, s0, s1 // (op 500) 

s_ashr_i32 s0, s8, 31 // (op 500) 
s_cmp_eq_i32 s14, 0 // (op 500) 
s_mov_b32 s9, scc // (op 500) 
s_cmp_eq_i32 s9, 0 // (op 500) 
s_cselect_b32 s18, 0, -1 // (op 500) 

s_and_b32 s9, s15, 31 // (op 500) 
s_lshl_b32 s19, 1, s9 // (op 500) 
s_add_i32 s49, s19, s18 // (op 500) 


s_and_b32 s18, s0, s49 // (op 500) 


s_add_i32 s0, s8, s18 // (op 500) 


s_ashr_i32 s8, s0, s9 // (op 500) 


s_lshl_b32 s0, s15, 24 // (op 500) 
s_ashr_i32 s9, s0, 31 // (op 500) 

s_xor_b32 s0, s8, s9 // (op 500) 

s_sub_i32 s8, s0, s9 // (op 500) 


s_mul_i32 s0, s1, s45 // (op 500) 

s_sub_u32 s1, s16, s0 // (op 500) 

s_mul_i32 s0, s8, s45 // (op 500) 
s_add_u32 s0, s0, s1 // (op 500) 


s_mul_hi_i32 s1, s16, s11 // (op 500) 
s_add_i32 s8, s1, s16 // (op 500) 

s_ashr_i32 s1, s8, 31 // (op 500) 
s_cmp_eq_i32 s11, 0 // (op 500) 
s_mov_b32 s9, scc // (op 500) 
s_cmp_eq_i32 s9, 0 // (op 500) 
s_cselect_b32 s18, 0, -1 // (op 500) 

s_and_b32 s9, s46, 31 // (op 500) 
s_lshl_b32 s19, 1, s9 // (op 500) 
s_add_i32 s49, s19, s18 // (op 500) 


s_and_b32 s18, s1, s49 // (op 500) 


s_add_i32 s1, s8, s18 // (op 500) 


s_ashr_i32 s8, s1, s9 // (op 500) 


s_lshl_b32 s1, s46, 24 // (op 500) 
s_ashr_i32 s9, s1, 31 // (op 500) 

s_xor_b32 s1, s8, s9 // (op 500) 

s_sub_i32 s8, s1, s9 // (op 500) 


s_add_u32 s18, s4, 32 // (op 500) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 500) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 500) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 500) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 500) 

s_mul_i32 s1, s45, s18 // (op 500) 

s_mul_i32 s9, s10, s1 // (op 500) 

s_mul_i32 s1, s8, s9 // (op 500) 

s_sub_u32 s9, s16, s1 // (op 500) 

s_mov_b32 s1, s9 // (op 500) 
s_mul_hi_i32 s18, s1, s44 // (op 500) 
s_add_i32 s19, s18, s1 // (op 500) 


s_ashr_i32 s1, s19, 31 // (op 500) 
s_cmp_eq_i32 s44, 0 // (op 500) 
s_mov_b32 s18, scc // (op 500) 
s_cmp_eq_i32 s18, 0 // (op 500) 
s_cselect_b32 s49, 0, -1 // (op 500) 

s_and_b32 s18, s48, 31 // (op 500) 
s_lshl_b32 s50, 1, s18 // (op 500) 
s_add_i32 s51, s50, s49 // (op 500) 


s_and_b32 s49, s1, s51 // (op 500) 


s_add_i32 s1, s19, s49 // (op 500) 


s_ashr_i32 s19, s1, s18 // (op 500) 


s_lshl_b32 s1, s48, 24 // (op 500) 
s_ashr_i32 s18, s1, 31 // (op 500) 

s_xor_b32 s1, s19, s18 // (op 500) 

s_sub_i32 s19, s1, s18 // (op 500) 


s_mul_i32 s1, s47, s45 // (op 500) 
s_add_u32 s50, s6, 16 // (op 500) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 500) End of Int64 add, unlocking SCC
s_sub_u32 s56, s50, 1 // (op 500) least significant half; sets scc
s_subb_u32 s57, s51, 0 // (op 500) most significant half; uses scc

s_ashr_i64 s[50:51], s[56:57], 4 // (op 500) 

s_sub_i32 s18, s50, s1 // (op 500) 

s_mul_i32 s49, s19, s18 // (op 500) 


s_sub_u32 s18, s9, s49 // (op 500) 


s_add_u32 s9, s18, s1 // (op 500) 


s_cmp_eq_i32 s8, 0 // (op 500) 
s_mov_b32 s1, scc // (op 500) 

s_cmp_eq_i32 s1, 0 // (op 500) 
s_cselect_b32 s8, s9, s0 // (op 500) 



v_lshl_add_u32 v12, s8, 4, v11 // (op 500) 


v_mul_lo_u32 v17, v12, s3 // (op 500) most significant: low of low * high
v_mul_hi_u32 v11, v12, s2 // (op 500) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 500) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 500) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 500) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 500) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 500) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 500) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 500) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 500) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 500) 

v_mov_b32 v11, v12 // (op 500) convert

s_mul_i32 s1, s7, 6 // (op 502) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 502) least significant
s_mul_hi_u32 s8, s6, 6 // (op 502) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 502) most significant: sum

s_mul_i32 s9, s2, s1 // (op 502) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 502) most significant: low of high * low
s_mul_i32 s8, s2, s0 // (op 502) least significant
s_mul_hi_u32 s19, s2, s0 // (op 502) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 502) most significant: sum
s_add_u32 s9, s9, s19 // (op 502) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 502) 

s_mov_b32 s52, s20 // (op 502) reg expression
s_mov_b32 s53, s21 // (op 502) reg expression
s_mov_b32 s54, s0 // (op 502) reg expression
s_mov_b32 s55, 131072 // (op 502) call()

s_mul_hi_i32 s0, s16, s12 // (op 545) 
v_lshrrev_b32 v12, 3, v1 // (op 545) 
s_add_i32 s1, s0, s16 // (op 545) 

v_and_b32 v13, -8, v1 // (op 545) 
s_ashr_i32 s0, s1, 31 // (op 545) 
v_sub_u32 v14, v1, v13 // (op 545) 

s_cmp_eq_i32 s12, 0 // (op 545) 
s_mov_b32 s8, scc // (op 545) 
v_lshlrev_b32 v13, 4, v14 // (op 545) 

s_cmp_eq_i32 s8, 0 // (op 545) 
s_cselect_b32 s9, 0, -1 // (op 545) 

v_mov_b32 v15, 0 // (op 545) convert
v_mov_b32 v14, v13 // (op 545) convert

s_and_b32 s8, s13, 31 // (op 545) 
s_lshl_b32 s18, 1, s8 // (op 545) 
s_add_i32 s19, s18, s9 // (op 545) 


s_and_b32 s9, s0, s19 // (op 545) 


s_add_i32 s0, s1, s9 // (op 545) 


s_ashr_i32 s1, s0, s8 // (op 545) 


s_lshl_b32 s0, s13, 24 // (op 545) 
s_ashr_i32 s8, s0, 31 // (op 545) 

s_xor_b32 s0, s1, s8 // (op 545) 

s_sub_i32 s1, s0, s8 // (op 545) 


s_mul_hi_i32 s0, s1, s14 // (op 545) 
s_add_i32 s8, s0, s1 // (op 545) 

s_ashr_i32 s0, s8, 31 // (op 545) 
s_cmp_eq_i32 s14, 0 // (op 545) 
s_mov_b32 s9, scc // (op 545) 
s_cmp_eq_i32 s9, 0 // (op 545) 
s_cselect_b32 s18, 0, -1 // (op 545) 

s_and_b32 s9, s15, 31 // (op 545) 
s_lshl_b32 s19, 1, s9 // (op 545) 
s_add_i32 s49, s19, s18 // (op 545) 


s_and_b32 s18, s0, s49 // (op 545) 


s_add_i32 s0, s8, s18 // (op 545) 


s_ashr_i32 s8, s0, s9 // (op 545) 


s_lshl_b32 s0, s15, 24 // (op 545) 
s_ashr_i32 s9, s0, 31 // (op 545) 

s_xor_b32 s0, s8, s9 // (op 545) 

s_sub_i32 s8, s0, s9 // (op 545) 


s_add_u32 s18, s4, 32 // (op 545) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 545) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 545) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 545) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 545) 

s_mul_i32 s0, s8, s18 // (op 545) 

s_sub_i32 s8, s1, s0 // (op 545) 


s_mul_hi_i32 s9, s16, s11 // (op 545) 
s_add_i32 s0, s9, s16 // (op 545) 

s_ashr_i32 s1, s0, 31 // (op 545) 
s_cmp_eq_i32 s11, 0 // (op 545) 
s_mov_b32 s9, scc // (op 545) 
s_cmp_eq_i32 s9, 0 // (op 545) 
s_cselect_b32 s49, 0, -1 // (op 545) 

s_and_b32 s9, s46, 31 // (op 545) 
s_lshl_b32 s50, 1, s9 // (op 545) 
s_add_i32 s51, s50, s49 // (op 545) 


s_and_b32 s49, s1, s51 // (op 545) 


s_add_i32 s1, s0, s49 // (op 545) 


s_ashr_i32 s0, s1, s9 // (op 545) 


s_lshl_b32 s1, s46, 24 // (op 545) 
s_ashr_i32 s9, s1, 31 // (op 545) 

s_xor_b32 s1, s0, s9 // (op 545) 

s_sub_i32 s0, s1, s9 // (op 545) 


s_mul_i32 s1, s45, s18 // (op 545) 

s_mul_i32 s9, s10, s1 // (op 545) 

s_mul_i32 s1, s0, s9 // (op 545) 

s_sub_u32 s9, s16, s1 // (op 545) 

s_mul_hi_i32 s1, s9, s44 // (op 545) 
s_add_i32 s18, s1, s9 // (op 545) 


s_ashr_i32 s1, s18, 31 // (op 545) 
s_cmp_eq_i32 s44, 0 // (op 545) 
s_mov_b32 s9, scc // (op 545) 
s_cmp_eq_i32 s9, 0 // (op 545) 
s_cselect_b32 s19, 0, -1 // (op 545) 

s_and_b32 s9, s48, 31 // (op 545) 
s_lshl_b32 s49, 1, s9 // (op 545) 
s_add_i32 s50, s49, s19 // (op 545) 


s_and_b32 s19, s1, s50 // (op 545) 


s_add_i32 s1, s18, s19 // (op 545) 


s_ashr_i32 s18, s1, s9 // (op 545) 


s_lshl_b32 s1, s48, 24 // (op 545) 
s_ashr_i32 s9, s1, 31 // (op 545) 

s_xor_b32 s1, s18, s9 // (op 545) 

s_sub_i32 s18, s1, s9 // (op 545) 


s_cmp_eq_i32 s0, 0 // (op 545) 
s_mov_b32 s1, scc // (op 545) 

s_cmp_eq_i32 s1, 0 // (op 545) 
s_cselect_b32 s0, s18, s8 // (op 545) 



v_lshl_add_u32 v13, s0, 5, v12 // (op 545) 


v_mul_lo_u32 v17, v13, s23 // (op 545) most significant: low of low * high
v_mul_hi_u32 v12, v13, s22 // (op 545) most significant: high of low * low
v_mul_lo_u32 v16, v13, s22 // (op 545) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 545) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 545) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 545) most significant half



s_mul_i32 s1, s22, s5 // (op 547) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 547) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 547) least significant
s_mul_hi_u32 s9, s22, s4 // (op 547) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 547) most significant: sum
s_add_u32 s1, s1, s9 // (op 547) most significant: sum


s_mov_b32 s56, s24 // (op 547) reg expression
s_mov_b32 s57, s25 // (op 547) reg expression
s_mov_b32 s58, s0 // (op 547) reg expression
s_mov_b32 s59, 131072 // (op 547) call()

s_mul_hi_i32 s0, s16, s12 // (op 564) 
v_lshrrev_b32 v14, 3, v1 // (op 564) 
s_add_i32 s1, s0, s16 // (op 564) 

v_and_b32 v15, -8, v1 // (op 564) 
s_ashr_i32 s0, s1, 31 // (op 564) 
v_sub_u32 v16, v1, v15 // (op 564) 

s_cmp_eq_i32 s12, 0 // (op 564) 
s_mov_b32 s8, scc // (op 564) 
v_lshlrev_b32 v15, 4, v16 // (op 564) 

s_cmp_eq_i32 s8, 0 // (op 564) 
s_cselect_b32 s9, 0, -1 // (op 564) 

v_mov_b32 v17, 0 // (op 564) convert
v_mov_b32 v16, v15 // (op 564) convert

s_and_b32 s8, s13, 31 // (op 564) 
s_lshl_b32 s18, 1, s8 // (op 564) 
s_add_i32 s19, s18, s9 // (op 564) 


s_and_b32 s9, s0, s19 // (op 564) 


s_add_i32 s0, s1, s9 // (op 564) 


s_ashr_i32 s1, s0, s8 // (op 564) 


s_lshl_b32 s0, s13, 24 // (op 564) 
s_ashr_i32 s8, s0, 31 // (op 564) 

s_xor_b32 s0, s1, s8 // (op 564) 

s_sub_i32 s1, s0, s8 // (op 564) 


s_mul_hi_i32 s0, s1, s14 // (op 564) 
s_add_i32 s8, s0, s1 // (op 564) 

s_ashr_i32 s0, s8, 31 // (op 564) 
s_cmp_eq_i32 s14, 0 // (op 564) 
s_mov_b32 s9, scc // (op 564) 
s_cmp_eq_i32 s9, 0 // (op 564) 
s_cselect_b32 s18, 0, -1 // (op 564) 

s_and_b32 s9, s15, 31 // (op 564) 
s_lshl_b32 s19, 1, s9 // (op 564) 
s_add_i32 s49, s19, s18 // (op 564) 


s_and_b32 s18, s0, s49 // (op 564) 


s_add_i32 s0, s8, s18 // (op 564) 


s_ashr_i32 s8, s0, s9 // (op 564) 


s_lshl_b32 s0, s15, 24 // (op 564) 
s_ashr_i32 s9, s0, 31 // (op 564) 

s_xor_b32 s0, s8, s9 // (op 564) 

s_sub_i32 s8, s0, s9 // (op 564) 


s_add_u32 s18, s4, 32 // (op 564) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 564) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 564) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 564) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 564) 

s_mul_i32 s0, s8, s18 // (op 564) 

s_sub_i32 s8, s1, s0 // (op 564) 


s_mul_hi_i32 s9, s16, s11 // (op 564) 
s_add_i32 s0, s9, s16 // (op 564) 

s_ashr_i32 s1, s0, 31 // (op 564) 
s_cmp_eq_i32 s11, 0 // (op 564) 
s_mov_b32 s9, scc // (op 564) 
s_cmp_eq_i32 s9, 0 // (op 564) 
s_cselect_b32 s49, 0, -1 // (op 564) 

s_and_b32 s9, s46, 31 // (op 564) 
s_lshl_b32 s50, 1, s9 // (op 564) 
s_add_i32 s51, s50, s49 // (op 564) 


s_and_b32 s49, s1, s51 // (op 564) 


s_add_i32 s1, s0, s49 // (op 564) 


s_ashr_i32 s0, s1, s9 // (op 564) 


s_lshl_b32 s1, s46, 24 // (op 564) 
s_ashr_i32 s9, s1, 31 // (op 564) 

s_xor_b32 s1, s0, s9 // (op 564) 

s_sub_i32 s0, s1, s9 // (op 564) 


s_mul_i32 s1, s45, s18 // (op 564) 

s_mul_i32 s9, s10, s1 // (op 564) 

s_mul_i32 s1, s0, s9 // (op 564) 

s_sub_u32 s9, s16, s1 // (op 564) 

s_mul_hi_i32 s1, s9, s44 // (op 564) 
s_add_i32 s18, s1, s9 // (op 564) 


s_ashr_i32 s1, s18, 31 // (op 564) 
s_cmp_eq_i32 s44, 0 // (op 564) 
s_mov_b32 s9, scc // (op 564) 
s_cmp_eq_i32 s9, 0 // (op 564) 
s_cselect_b32 s19, 0, -1 // (op 564) 

s_and_b32 s9, s48, 31 // (op 564) 
s_lshl_b32 s49, 1, s9 // (op 564) 
s_add_i32 s50, s49, s19 // (op 564) 


s_and_b32 s19, s1, s50 // (op 564) 


s_add_i32 s1, s18, s19 // (op 564) 


s_ashr_i32 s18, s1, s9 // (op 564) 


s_lshl_b32 s1, s48, 24 // (op 564) 
s_ashr_i32 s9, s1, 31 // (op 564) 

s_xor_b32 s1, s18, s9 // (op 564) 

s_sub_i32 s18, s1, s9 // (op 564) 


s_cmp_eq_i32 s0, 0 // (op 564) 
s_mov_b32 s1, scc // (op 564) 

s_cmp_eq_i32 s1, 0 // (op 564) 
s_cselect_b32 s0, s18, s8 // (op 564) 



v_lshl_add_u32 v15, s0, 5, v14 // (op 564) 


v_mul_lo_u32 v19, v15, s23 // (op 564) most significant: low of low * high
v_mul_hi_u32 v14, v15, s22 // (op 564) most significant: high of low * low
v_mul_lo_u32 v18, v15, s22 // (op 564) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 564) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 564) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 564) most significant half



s_mul_i32 s1, s22, s5 // (op 566) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 566) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 566) least significant
s_mul_hi_u32 s9, s22, s4 // (op 566) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 566) most significant: sum
s_add_u32 s1, s1, s9 // (op 566) most significant: sum


s_mov_b32 s56, s24 // (op 566) reg expression
s_mov_b32 s57, s25 // (op 566) reg expression
s_mov_b32 s58, s0 // (op 566) reg expression
s_mov_b32 s59, 131072 // (op 566) call()

s_mul_hi_i32 s0, s16, s12 // (op 609) 
v_lshrrev_b32 v16, 7, v1 // (op 609) 
s_add_i32 s1, s0, s16 // (op 609) 

v_and_b32 v17, 15, v1 // (op 609) 
s_ashr_i32 s0, s1, 31 // (op 609) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 609) 


s_cmp_eq_i32 s12, 0 // (op 609) 
s_mov_b32 s8, scc // (op 609) 
v_and_b32 v16, 63, v1 // (op 609) 
s_cmp_eq_i32 s8, 0 // (op 609) 
s_cselect_b32 s9, 0, -1 // (op 609) 

v_lshrrev_b32 v17, 4, v16 // (op 609) 

s_and_b32 s8, s13, 31 // (op 609) 
s_lshl_b32 s18, 1, s8 // (op 609) 
s_add_i32 s19, s18, s9 // (op 609) 


s_and_b32 s9, s0, s19 // (op 609) 


s_add_i32 s0, s1, s9 // (op 609) 


s_ashr_i32 s1, s0, s8 // (op 609) 


s_lshl_b32 s0, s13, 24 // (op 609) 
s_ashr_i32 s8, s0, 31 // (op 609) 

s_xor_b32 s0, s1, s8 // (op 609) 

s_sub_i32 s1, s0, s8 // (op 609) 


s_mul_hi_i32 s0, s1, s14 // (op 609) 
s_add_i32 s8, s0, s1 // (op 609) 

s_ashr_i32 s0, s8, 31 // (op 609) 
s_cmp_eq_i32 s14, 0 // (op 609) 
s_mov_b32 s9, scc // (op 609) 
s_cmp_eq_i32 s9, 0 // (op 609) 
s_cselect_b32 s18, 0, -1 // (op 609) 

s_and_b32 s9, s15, 31 // (op 609) 
s_lshl_b32 s19, 1, s9 // (op 609) 
s_add_i32 s49, s19, s18 // (op 609) 


s_and_b32 s18, s0, s49 // (op 609) 


s_add_i32 s0, s8, s18 // (op 609) 


s_ashr_i32 s8, s0, s9 // (op 609) 


s_lshl_b32 s0, s15, 24 // (op 609) 
s_ashr_i32 s9, s0, 31 // (op 609) 

s_xor_b32 s0, s8, s9 // (op 609) 

s_sub_i32 s8, s0, s9 // (op 609) 


s_mul_i32 s0, s1, s45 // (op 609) 

s_sub_u32 s1, s16, s0 // (op 609) 

s_mul_i32 s0, s8, s45 // (op 609) 
s_add_u32 s0, s0, s1 // (op 609) 


s_mul_hi_i32 s1, s16, s11 // (op 609) 
s_add_i32 s8, s1, s16 // (op 609) 

s_ashr_i32 s1, s8, 31 // (op 609) 
s_cmp_eq_i32 s11, 0 // (op 609) 
s_mov_b32 s9, scc // (op 609) 
s_cmp_eq_i32 s9, 0 // (op 609) 
s_cselect_b32 s18, 0, -1 // (op 609) 

s_and_b32 s9, s46, 31 // (op 609) 
s_lshl_b32 s19, 1, s9 // (op 609) 
s_add_i32 s49, s19, s18 // (op 609) 


s_and_b32 s18, s1, s49 // (op 609) 


s_add_i32 s1, s8, s18 // (op 609) 


s_ashr_i32 s8, s1, s9 // (op 609) 


s_lshl_b32 s1, s46, 24 // (op 609) 
s_ashr_i32 s9, s1, 31 // (op 609) 

s_xor_b32 s1, s8, s9 // (op 609) 

s_sub_i32 s8, s1, s9 // (op 609) 


s_add_u32 s18, s4, 32 // (op 609) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 609) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 609) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 609) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 609) 

s_mul_i32 s1, s45, s18 // (op 609) 

s_mul_i32 s9, s10, s1 // (op 609) 

s_mul_i32 s1, s8, s9 // (op 609) 

s_sub_u32 s9, s16, s1 // (op 609) 

s_mov_b32 s1, s9 // (op 609) 
s_mul_hi_i32 s18, s1, s44 // (op 609) 
s_add_i32 s19, s18, s1 // (op 609) 


s_ashr_i32 s1, s19, 31 // (op 609) 
s_cmp_eq_i32 s44, 0 // (op 609) 
s_mov_b32 s18, scc // (op 609) 
s_cmp_eq_i32 s18, 0 // (op 609) 
s_cselect_b32 s49, 0, -1 // (op 609) 

s_and_b32 s18, s48, 31 // (op 609) 
s_lshl_b32 s50, 1, s18 // (op 609) 
s_add_i32 s51, s50, s49 // (op 609) 


s_and_b32 s49, s1, s51 // (op 609) 


s_add_i32 s1, s19, s49 // (op 609) 


s_ashr_i32 s19, s1, s18 // (op 609) 


s_lshl_b32 s1, s48, 24 // (op 609) 
s_ashr_i32 s18, s1, 31 // (op 609) 

s_xor_b32 s1, s19, s18 // (op 609) 

s_sub_i32 s19, s1, s18 // (op 609) 


s_mul_i32 s1, s47, s45 // (op 609) 
s_add_u32 s50, s6, 16 // (op 609) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 609) End of Int64 add, unlocking SCC
s_sub_u32 s60, s50, 1 // (op 609) least significant half; sets scc
s_subb_u32 s61, s51, 0 // (op 609) most significant half; uses scc

s_ashr_i64 s[50:51], s[60:61], 4 // (op 609) 

s_sub_i32 s18, s50, s1 // (op 609) 

s_mul_i32 s49, s19, s18 // (op 609) 


s_sub_u32 s18, s9, s49 // (op 609) 


s_add_u32 s9, s18, s1 // (op 609) 


s_cmp_eq_i32 s8, 0 // (op 609) 
s_mov_b32 s1, scc // (op 609) 

s_cmp_eq_i32 s1, 0 // (op 609) 
s_cselect_b32 s8, s9, s0 // (op 609) 



v_lshl_add_u32 v16, s8, 4, v18 // (op 609) 


v_mul_lo_u32 v18, v16, s26 // (op 609) 
v_add_u32 v18, v18, v17 // (op 609) 


s_mul_i32 s1, s26, s7 // (op 611) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 611) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 611) least significant
s_mul_hi_u32 s9, s26, s6 // (op 611) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 611) most significant: sum
s_add_u32 s1, s1, s9 // (op 611) most significant: sum


s_mov_b32 s60, s28 // (op 611) reg expression
s_mov_b32 s61, s29 // (op 611) reg expression
s_mov_b32 s62, s0 // (op 611) reg expression
s_mov_b32 s63, 131072 // (op 611) call()

s_mul_hi_i32 s0, s16, s12 // (op 628) 
v_lshrrev_b32 v16, 7, v1 // (op 628) 
s_add_i32 s1, s0, s16 // (op 628) 

v_and_b32 v17, 15, v1 // (op 628) 
s_ashr_i32 s0, s1, 31 // (op 628) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 628) 


s_cmp_eq_i32 s12, 0 // (op 628) 
s_mov_b32 s8, scc // (op 628) 
v_and_b32 v16, 63, v1 // (op 628) 
s_cmp_eq_i32 s8, 0 // (op 628) 
s_cselect_b32 s9, 0, -1 // (op 628) 

v_lshrrev_b32 v17, 4, v16 // (op 628) 

s_and_b32 s8, s13, 31 // (op 628) 
s_lshl_b32 s18, 1, s8 // (op 628) 
s_add_i32 s19, s18, s9 // (op 628) 


s_and_b32 s9, s0, s19 // (op 628) 


s_add_i32 s0, s1, s9 // (op 628) 


s_ashr_i32 s1, s0, s8 // (op 628) 


s_lshl_b32 s0, s13, 24 // (op 628) 
s_ashr_i32 s8, s0, 31 // (op 628) 

s_xor_b32 s0, s1, s8 // (op 628) 

s_sub_i32 s1, s0, s8 // (op 628) 


s_mul_hi_i32 s0, s1, s14 // (op 628) 
s_add_i32 s8, s0, s1 // (op 628) 

s_ashr_i32 s0, s8, 31 // (op 628) 
s_cmp_eq_i32 s14, 0 // (op 628) 
s_mov_b32 s9, scc // (op 628) 
s_cmp_eq_i32 s9, 0 // (op 628) 
s_cselect_b32 s18, 0, -1 // (op 628) 

s_and_b32 s9, s15, 31 // (op 628) 
s_lshl_b32 s19, 1, s9 // (op 628) 
s_add_i32 s49, s19, s18 // (op 628) 


s_and_b32 s18, s0, s49 // (op 628) 


s_add_i32 s0, s8, s18 // (op 628) 


s_ashr_i32 s8, s0, s9 // (op 628) 


s_lshl_b32 s0, s15, 24 // (op 628) 
s_ashr_i32 s9, s0, 31 // (op 628) 

s_xor_b32 s0, s8, s9 // (op 628) 

s_sub_i32 s8, s0, s9 // (op 628) 


s_mul_i32 s0, s1, s45 // (op 628) 

s_sub_u32 s1, s16, s0 // (op 628) 

s_mul_i32 s0, s8, s45 // (op 628) 
s_add_u32 s0, s0, s1 // (op 628) 


s_mul_hi_i32 s1, s16, s11 // (op 628) 
s_add_i32 s8, s1, s16 // (op 628) 

s_ashr_i32 s1, s8, 31 // (op 628) 
s_cmp_eq_i32 s11, 0 // (op 628) 
s_mov_b32 s9, scc // (op 628) 
s_cmp_eq_i32 s9, 0 // (op 628) 
s_cselect_b32 s18, 0, -1 // (op 628) 

s_and_b32 s9, s46, 31 // (op 628) 
s_lshl_b32 s19, 1, s9 // (op 628) 
s_add_i32 s49, s19, s18 // (op 628) 


s_and_b32 s18, s1, s49 // (op 628) 


s_add_i32 s1, s8, s18 // (op 628) 


s_ashr_i32 s8, s1, s9 // (op 628) 


s_lshl_b32 s1, s46, 24 // (op 628) 
s_ashr_i32 s9, s1, 31 // (op 628) 

s_xor_b32 s1, s8, s9 // (op 628) 

s_sub_i32 s8, s1, s9 // (op 628) 


s_add_u32 s18, s4, 32 // (op 628) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 628) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 628) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 628) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 628) 

s_mul_i32 s1, s45, s18 // (op 628) 

s_mul_i32 s9, s10, s1 // (op 628) 

s_mul_i32 s1, s8, s9 // (op 628) 

s_sub_u32 s9, s16, s1 // (op 628) 

s_mov_b32 s1, s9 // (op 628) 
s_mul_hi_i32 s18, s1, s44 // (op 628) 
s_add_i32 s19, s18, s1 // (op 628) 


s_ashr_i32 s1, s19, 31 // (op 628) 
s_cmp_eq_i32 s44, 0 // (op 628) 
s_mov_b32 s18, scc // (op 628) 
s_cmp_eq_i32 s18, 0 // (op 628) 
s_cselect_b32 s49, 0, -1 // (op 628) 

s_and_b32 s18, s48, 31 // (op 628) 
s_lshl_b32 s50, 1, s18 // (op 628) 
s_add_i32 s51, s50, s49 // (op 628) 


s_and_b32 s49, s1, s51 // (op 628) 


s_add_i32 s1, s19, s49 // (op 628) 


s_ashr_i32 s19, s1, s18 // (op 628) 


s_lshl_b32 s1, s48, 24 // (op 628) 
s_ashr_i32 s18, s1, 31 // (op 628) 

s_xor_b32 s1, s19, s18 // (op 628) 

s_sub_i32 s19, s1, s18 // (op 628) 


s_mul_i32 s1, s47, s45 // (op 628) 
s_add_u32 s50, s6, 16 // (op 628) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 628) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 628) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 628) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 4 // (op 628) 

s_sub_i32 s18, s50, s1 // (op 628) 

s_mul_i32 s49, s19, s18 // (op 628) 


s_sub_u32 s18, s9, s49 // (op 628) 


s_add_u32 s9, s18, s1 // (op 628) 


s_cmp_eq_i32 s8, 0 // (op 628) 
s_mov_b32 s1, scc // (op 628) 

s_cmp_eq_i32 s1, 0 // (op 628) 
s_cselect_b32 s8, s9, s0 // (op 628) 



v_lshl_add_u32 v16, s8, 4, v19 // (op 628) 


v_mul_lo_u32 v19, v16, s26 // (op 628) 
v_add_u32 v19, v19, v17 // (op 628) 


s_mul_i32 s1, s26, s7 // (op 630) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 630) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 630) least significant
s_mul_hi_u32 s9, s26, s6 // (op 630) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 630) most significant: sum
s_add_u32 s1, s1, s9 // (op 630) most significant: sum


s_mov_b32 s60, s28 // (op 630) reg expression
s_mov_b32 s61, s29 // (op 630) reg expression
s_mov_b32 s62, s0 // (op 630) reg expression
s_mov_b32 s63, 131072 // (op 630) call()

s_mul_hi_i32 s0, s16, s12 // (op 673) 
v_lshrrev_b32 v16, 2, v1 // (op 673) 
s_add_i32 s1, s0, s16 // (op 673) 

v_and_b32 v17, -4, v1 // (op 673) 
s_ashr_i32 s0, s1, 31 // (op 673) 
v_sub_u32 v20, v1, v17 // (op 673) 

s_cmp_eq_i32 s12, 0 // (op 673) 
s_mov_b32 s8, scc // (op 673) 
s_cmp_eq_i32 s8, 0 // (op 673) 
s_cselect_b32 s9, 0, -1 // (op 673) 

s_and_b32 s8, s13, 31 // (op 673) 
s_lshl_b32 s18, 1, s8 // (op 673) 
s_add_i32 s19, s18, s9 // (op 673) 


s_and_b32 s9, s0, s19 // (op 673) 


s_add_i32 s0, s1, s9 // (op 673) 


s_ashr_i32 s1, s0, s8 // (op 673) 


s_lshl_b32 s0, s13, 24 // (op 673) 
s_ashr_i32 s8, s0, 31 // (op 673) 

s_xor_b32 s0, s1, s8 // (op 673) 

s_sub_i32 s1, s0, s8 // (op 673) 


s_mul_hi_i32 s0, s1, s14 // (op 673) 
s_add_i32 s8, s0, s1 // (op 673) 

s_ashr_i32 s0, s8, 31 // (op 673) 
s_cmp_eq_i32 s14, 0 // (op 673) 
s_mov_b32 s9, scc // (op 673) 
s_cmp_eq_i32 s9, 0 // (op 673) 
s_cselect_b32 s18, 0, -1 // (op 673) 

s_and_b32 s9, s15, 31 // (op 673) 
s_lshl_b32 s19, 1, s9 // (op 673) 
s_add_i32 s49, s19, s18 // (op 673) 


s_and_b32 s18, s0, s49 // (op 673) 


s_add_i32 s0, s8, s18 // (op 673) 


s_ashr_i32 s8, s0, s9 // (op 673) 


s_lshl_b32 s0, s15, 24 // (op 673) 
s_ashr_i32 s9, s0, 31 // (op 673) 

s_xor_b32 s0, s8, s9 // (op 673) 

s_sub_i32 s8, s0, s9 // (op 673) 


s_add_u32 s18, s4, 32 // (op 673) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 673) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 673) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 673) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 673) 

s_mul_i32 s0, s8, s18 // (op 673) 

s_sub_i32 s8, s1, s0 // (op 673) 


s_mul_hi_i32 s9, s16, s11 // (op 673) 
s_add_i32 s0, s9, s16 // (op 673) 

s_ashr_i32 s1, s0, 31 // (op 673) 
s_cmp_eq_i32 s11, 0 // (op 673) 
s_mov_b32 s9, scc // (op 673) 
s_cmp_eq_i32 s9, 0 // (op 673) 
s_cselect_b32 s49, 0, -1 // (op 673) 

s_and_b32 s9, s46, 31 // (op 673) 
s_lshl_b32 s50, 1, s9 // (op 673) 
s_add_i32 s51, s50, s49 // (op 673) 


s_and_b32 s49, s1, s51 // (op 673) 


s_add_i32 s1, s0, s49 // (op 673) 


s_ashr_i32 s0, s1, s9 // (op 673) 


s_lshl_b32 s1, s46, 24 // (op 673) 
s_ashr_i32 s9, s1, 31 // (op 673) 

s_xor_b32 s1, s0, s9 // (op 673) 

s_sub_i32 s0, s1, s9 // (op 673) 


s_mul_i32 s1, s45, s18 // (op 673) 

s_mul_i32 s9, s10, s1 // (op 673) 

s_mul_i32 s1, s0, s9 // (op 673) 

s_sub_u32 s9, s16, s1 // (op 673) 

s_mul_hi_i32 s1, s9, s44 // (op 673) 
s_add_i32 s18, s1, s9 // (op 673) 


s_ashr_i32 s1, s18, 31 // (op 673) 
s_cmp_eq_i32 s44, 0 // (op 673) 
s_mov_b32 s9, scc // (op 673) 
s_cmp_eq_i32 s9, 0 // (op 673) 
s_cselect_b32 s19, 0, -1 // (op 673) 

s_and_b32 s9, s48, 31 // (op 673) 
s_lshl_b32 s49, 1, s9 // (op 673) 
s_add_i32 s50, s49, s19 // (op 673) 


s_and_b32 s19, s1, s50 // (op 673) 


s_add_i32 s1, s18, s19 // (op 673) 


s_ashr_i32 s18, s1, s9 // (op 673) 


s_lshl_b32 s1, s48, 24 // (op 673) 
s_ashr_i32 s9, s1, 31 // (op 673) 

s_xor_b32 s1, s18, s9 // (op 673) 

s_sub_i32 s18, s1, s9 // (op 673) 


s_cmp_eq_i32 s0, 0 // (op 673) 
s_mov_b32 s1, scc // (op 673) 

s_cmp_eq_i32 s1, 0 // (op 673) 
s_cselect_b32 s0, s18, s8 // (op 673) 



v_lshl_add_u32 v17, s0, 5, v16 // (op 673) 


v_mul_lo_u32 v16, v17, s30 // (op 673) 

v_add_u32 v17, v20, v16 // (op 673) 


s_mul_i32 s1, s30, s5 // (op 675) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 675) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 675) least significant
s_mul_hi_u32 s9, s30, s4 // (op 675) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 675) most significant: sum
s_add_u32 s1, s1, s9 // (op 675) most significant: sum


s_mov_b32 s64, s32 // (op 675) reg expression
s_mov_b32 s65, s33 // (op 675) reg expression
s_mov_b32 s66, s0 // (op 675) reg expression
s_mov_b32 s67, 131072 // (op 675) call()

s_mul_hi_i32 s0, s16, s12 // (op 692) 
v_lshrrev_b32 v16, 2, v1 // (op 692) 
s_add_i32 s1, s0, s16 // (op 692) 

v_and_b32 v20, -4, v1 // (op 692) 
s_ashr_i32 s0, s1, 31 // (op 692) 
v_sub_u32 v21, v1, v20 // (op 692) 

s_cmp_eq_i32 s12, 0 // (op 692) 
s_mov_b32 s8, scc // (op 692) 
s_cmp_eq_i32 s8, 0 // (op 692) 
s_cselect_b32 s9, 0, -1 // (op 692) 

s_and_b32 s8, s13, 31 // (op 692) 
s_lshl_b32 s18, 1, s8 // (op 692) 
s_add_i32 s19, s18, s9 // (op 692) 


s_and_b32 s9, s0, s19 // (op 692) 


s_add_i32 s0, s1, s9 // (op 692) 


s_ashr_i32 s1, s0, s8 // (op 692) 


s_lshl_b32 s0, s13, 24 // (op 692) 
s_ashr_i32 s8, s0, 31 // (op 692) 

s_xor_b32 s0, s1, s8 // (op 692) 

s_sub_i32 s1, s0, s8 // (op 692) 


s_mul_hi_i32 s0, s1, s14 // (op 692) 
s_add_i32 s8, s0, s1 // (op 692) 

s_ashr_i32 s0, s8, 31 // (op 692) 
s_cmp_eq_i32 s14, 0 // (op 692) 
s_mov_b32 s9, scc // (op 692) 
s_cmp_eq_i32 s9, 0 // (op 692) 
s_cselect_b32 s18, 0, -1 // (op 692) 

s_and_b32 s9, s15, 31 // (op 692) 
s_lshl_b32 s19, 1, s9 // (op 692) 
s_add_i32 s49, s19, s18 // (op 692) 


s_and_b32 s18, s0, s49 // (op 692) 


s_add_i32 s0, s8, s18 // (op 692) 


s_ashr_i32 s8, s0, s9 // (op 692) 


s_lshl_b32 s0, s15, 24 // (op 692) 
s_ashr_i32 s9, s0, 31 // (op 692) 

s_xor_b32 s0, s8, s9 // (op 692) 

s_sub_i32 s8, s0, s9 // (op 692) 


s_add_u32 s18, s4, 32 // (op 692) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 692) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 692) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 692) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 692) 

s_mul_i32 s0, s8, s18 // (op 692) 

s_sub_i32 s8, s1, s0 // (op 692) 


s_mul_hi_i32 s9, s16, s11 // (op 692) 
s_add_i32 s0, s9, s16 // (op 692) 

s_ashr_i32 s1, s0, 31 // (op 692) 
s_cmp_eq_i32 s11, 0 // (op 692) 
s_mov_b32 s9, scc // (op 692) 
s_cmp_eq_i32 s9, 0 // (op 692) 
s_cselect_b32 s49, 0, -1 // (op 692) 

s_and_b32 s9, s46, 31 // (op 692) 
s_lshl_b32 s50, 1, s9 // (op 692) 
s_add_i32 s51, s50, s49 // (op 692) 


s_and_b32 s49, s1, s51 // (op 692) 


s_add_i32 s1, s0, s49 // (op 692) 


s_ashr_i32 s0, s1, s9 // (op 692) 


s_lshl_b32 s1, s46, 24 // (op 692) 
s_ashr_i32 s9, s1, 31 // (op 692) 

s_xor_b32 s1, s0, s9 // (op 692) 

s_sub_i32 s0, s1, s9 // (op 692) 


s_mul_i32 s1, s45, s18 // (op 692) 

s_mul_i32 s9, s10, s1 // (op 692) 

s_mul_i32 s1, s0, s9 // (op 692) 

s_sub_u32 s9, s16, s1 // (op 692) 

s_mul_hi_i32 s1, s9, s44 // (op 692) 
s_add_i32 s18, s1, s9 // (op 692) 


s_ashr_i32 s1, s18, 31 // (op 692) 
s_cmp_eq_i32 s44, 0 // (op 692) 
s_mov_b32 s9, scc // (op 692) 
s_cmp_eq_i32 s9, 0 // (op 692) 
s_cselect_b32 s19, 0, -1 // (op 692) 

s_and_b32 s9, s48, 31 // (op 692) 
s_lshl_b32 s49, 1, s9 // (op 692) 
s_add_i32 s50, s49, s19 // (op 692) 


s_and_b32 s19, s1, s50 // (op 692) 


s_add_i32 s1, s18, s19 // (op 692) 


s_ashr_i32 s18, s1, s9 // (op 692) 


s_lshl_b32 s1, s48, 24 // (op 692) 
s_ashr_i32 s9, s1, 31 // (op 692) 

s_xor_b32 s1, s18, s9 // (op 692) 

s_sub_i32 s18, s1, s9 // (op 692) 


s_cmp_eq_i32 s0, 0 // (op 692) 
s_mov_b32 s1, scc // (op 692) 

s_cmp_eq_i32 s1, 0 // (op 692) 
s_cselect_b32 s0, s18, s8 // (op 692) 



v_lshl_add_u32 v20, s0, 5, v16 // (op 692) 


v_mul_lo_u32 v16, v20, s30 // (op 692) 

v_add_u32 v20, v21, v16 // (op 692) 


s_mul_i32 s1, s30, s5 // (op 694) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 694) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 694) least significant
s_mul_hi_u32 s9, s30, s4 // (op 694) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 694) most significant: sum
s_add_u32 s1, s1, s9 // (op 694) most significant: sum


s_mov_b32 s64, s32 // (op 694) reg expression
s_mov_b32 s65, s33 // (op 694) reg expression
s_mov_b32 s66, s0 // (op 694) reg expression
s_mov_b32 s67, 131072 // (op 694) call()




buffer_load_ubyte v16, v18, s[60:63], 0 offen offset: 0 // (op 341) Load value
v_and_b32 v21, -8, v1 // (op 811) 
v_sub_u32 v22, v1, v21 // (op 811) 

v_lshlrev_b32 v21, 4, v22 // (op 811) 

v_lshrrev_b32 v22, 3, v1 // (op 811) 
v_lshl_add_u32 v23, v22, 7, v21 // (op 811) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 811) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[8:9], s[22:23], 4 // (op 437) 
v_mov_b64 v[22:23], s[8:9] // (op 437) convert


v_mov_b64 v[24:25], v[12:13] // (op 437) 
s_mov_b32 m0, s0 // (op 437) reg expression
buffer_load_dwordx4 v24, s[56:59], 0 offen offset: 0 lds // (op 437) Load value direct to lds
v_add_co_u32 v24, s[8:9], v24, v22 // (op 437) least significant half
v_addc_co_u32 v25, s[8:9], 0, v25, s[8:9] // (op 437) most significant half

s_add_u32 m0, m0, 2048 // (op 437) 
buffer_load_dwordx4 v24, s[56:59], 0 offen offset: 0 lds // (op 437) Load value direct to lds






buffer_load_ubyte v21, v17, s[64:67], 0 offen offset: 0 // (op 348) Load value



buffer_load_dwordx3 v[22:24], v10, s[52:55], 0 offen // (op 351) Load value



buffer_load_ubyte v25, v18, s[60:63], 0 offen offset:4 // (op 355) Load value

v_and_b32 v18, -8, v1 // (op 961) 
v_sub_u32 v26, v1, v18 // (op 961) 

v_lshlrev_b32 v18, 4, v26 // (op 961) 

v_lshrrev_b32 v26, 3, v1 // (op 961) 
v_lshl_add_u32 v27, v26, 7, v18 // (op 961) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 961) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[8:9], s[22:23], 4 // (op 442) 
v_mov_b64 v[26:27], s[8:9] // (op 442) convert


v_mov_b64 v[28:29], v[12:13] // (op 442) 
s_add_u32 s1, 4096, s0 // (op 442) 
s_mov_b32 m0, s1 // (op 442) reg expression
s_mov_b32 s8, 128 // (op 442) call()
buffer_load_dwordx4 v28, s[56:59], s8 offen offset: 0 lds // (op 442) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 442) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 442) most significant half

s_add_u32 m0, m0, 2048 // (op 442) 
buffer_load_dwordx4 v28, s[56:59], s8 offen offset: 0 lds // (op 442) Load value direct to lds









buffer_load_ubyte v18, v17, s[64:67], 0 offen offset:4 // (op 362) Load value




buffer_load_dwordx3 v[26:28], v10, s[52:55], 0 offen offset:96 // (op 365) Load value

v_and_b32 v10, -4, v1 // (op 864) 
v_sub_u32 v17, v1, v10 // (op 864) 

v_and_b32 v10, 4294967292, v1 // (op 864) 
v_add_u32 v12, v10, v17 // (op 864) 





s_waitcnt vmcnt(6)
ds_write_b8 v12, v21 offset:8192 // (op 368) Store local data 

v_and_b32 v10, -8, v1 // (op 759) 
v_sub_u32 v17, v1, v10 // (op 759) 

v_lshlrev_b32 v10, 4, v17 // (op 759) 

v_lshrrev_b32 v17, 3, v1 // (op 759) 
v_lshl_add_u32 v12, v17, 7, v10 // (op 759) 


v_mul_lo_u32 v10, v12, 6 // (op 759) 

v_lshrrev_b32 v17, 3, v10 // (op 759) 




s_waitcnt vmcnt(5)
ds_write_b96 v17, v[22:24] offset:8320 // (op 369) Store local data 

v_and_b32 v10, -8, v1 // (op 730) 
v_sub_u32 v17, v1, v10 // (op 730) 

v_lshlrev_b32 v10, 4, v17 // (op 730) 

v_lshrrev_b32 v17, 3, v1 // (op 730) 
v_lshl_add_u32 v12, v17, 7, v10 // (op 730) 


v_mul_lo_u32 v10, v12, 6 // (op 730) 

v_lshrrev_b32 v17, 3, v10 // (op 730) 

v_and_b32 v10, -8, v1 // (op 783) 
v_sub_u32 v12, v1, v10 // (op 783) 

v_lshlrev_b32 v10, 4, v12 // (op 783) 

v_lshrrev_b32 v12, 3, v1 // (op 783) 
v_lshl_add_u32 v13, v12, 7, v10 // (op 783) 


s_nop 0
v_readfirstlane_b32 s0, v13 // (op 783) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 835) 
v_sub_u32 v12, v1, v10 // (op 835) 

v_and_b32 v10, 4294967292, v1 // (op 835) 
v_add_u32 v13, v10, v12 // (op 835) 


v_and_b32 v10, -8, v1 // (op 922) 
v_sub_u32 v12, v1, v10 // (op 922) 

v_lshlrev_b32 v10, 4, v12 // (op 922) 

v_lshrrev_b32 v12, 3, v1 // (op 922) 
v_lshl_add_u32 v29, v12, 7, v10 // (op 922) 


v_mul_lo_u32 v10, v29, 6 // (op 922) 

v_lshrrev_b32 v12, 3, v10 // (op 922) 

v_and_b32 v10, -8, v1 // (op 946) 
v_sub_u32 v29, v1, v10 // (op 946) 

v_lshlrev_b32 v10, 4, v29 // (op 946) 

v_lshrrev_b32 v29, 3, v1 // (op 946) 
v_lshl_add_u32 v30, v29, 7, v10 // (op 946) 


s_nop 0
v_readfirstlane_b32 s1, v30 // (op 946) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 985) 
v_sub_u32 v29, v1, v10 // (op 985) 

v_and_b32 v10, 4294967292, v1 // (op 985) 
v_add_u32 v30, v10, v29 // (op 985) 


s_mov_b32 s8, 0 // (op 57) call()
s_ashr_i64 s[18:19], s[34:35], 7 // (op 56) 
s_mov_b32 s9, s18 // (op 56) convert

s_and_b32 s18, s9, 4294967294 // (op 56) 

s_cmp_lt_u32 s8, s18 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56 if false)
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56:
s_add_u32 s8, s8, 2 // (op 58) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1030) 



ds_read_b128 v[32:35], v3 offset:8320 // (op 67) Load local data 
ds_read_b64 v[36:37], v3 offset:8336 // (op 67) Load local data 
s_barrier  // (op 1043) 



ds_read_b128 v[38:41], v4 // (op 72) Load local data 
ds_read_b128 v[42:45], v4 offset:64 // (op 72) Load local data 
s_barrier  // (op 1056) 



ds_read_u8 v10, v0 offset:8192 // (op 81) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[38:45], a[0:3], v16, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 84) 





v_mov_b32 v10, v19 // (op 77) 

buffer_load_ubyte v16, v10, s[60:63], 0 offen offset:8 // (op 77) Load value
s_barrier  // (op 1046) 

s_lshl_b64 s[18:19], s[22:23], 4 // (op 420) 
v_mov_b64 v[32:33], s[18:19] // (op 420) convert


v_mov_b64 v[34:35], v[14:15] // (op 420) 
v_mov_b64 v[36:37], v[34:35] // (op 420) 
s_mov_b32 m0, s0 // (op 420) reg expression
s_mov_b32 s9, 256 // (op 420) call()
buffer_load_dwordx4 v36, s[56:59], s9 offen offset: 0 lds // (op 420) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 420) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 420) most significant half

s_add_u32 m0, m0, 2048 // (op 420) 
buffer_load_dwordx4 v36, s[56:59], s9 offen offset: 0 lds // (op 420) Load value direct to lds





v_mov_b32 v29, v20 // (op 21) 

buffer_load_ubyte v21, v29, s[64:67], 0 offen offset:8 // (op 21) Load value


v_mov_b32 v31, v11 // (op 4) 

buffer_load_dwordx3 v[22:24], v31, s[52:55], 0 offen offset:192 // (op 4) Load value
s_barrier  // (op 1085) 



s_waitcnt vmcnt(6)
ds_write_b8 v30, v18 offset:9856 // (op 240) Store local data 
s_barrier  // (op 1073) 



s_waitcnt vmcnt(5)
ds_write_b96 v12, v[26:28] offset:9984 // (op 229) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1070) 



ds_read_b128 v[36:39], v6 offset:9984 // (op 231) Load local data 
ds_read_b64 v[40:41], v6 offset:10000 // (op 231) Load local data 
s_barrier  // (op 1076) 
s_barrier  // (op 1136) 



ds_read_b128 v[42:45], v7 offset:4096 // (op 233) Load local data 
ds_read_b128 v[46:49], v7 offset:4160 // (op 233) Load local data 
s_barrier  // (op 1082) 



ds_read_u8 v32, v5 offset:9856 // (op 237) Load local data 
s_barrier  // (op 1146) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:49], a[0:3], v25, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 238) 






buffer_load_ubyte v25, v10, s[60:63], 0 offen offset:12 // (op 235) Load value

s_barrier  // (op 1079) 
v_add_u32 v19, 8, v19 // (op 620) 

s_lshl_b64 s[18:19], s[22:23], 4 // (op 432) 
v_mov_b64 v[32:33], s[18:19] // (op 432) convert


v_mov_b64 v[36:37], v[34:35] // (op 432) 
s_add_u32 s9, 4096, s1 // (op 432) 
s_mov_b32 m0, s9 // (op 432) reg expression
s_mov_b32 s18, 384 // (op 432) call()
buffer_load_dwordx4 v36, s[56:59], s18 offen offset: 0 lds // (op 432) Load value direct to lds
v_add_co_u32 v36, s[50:51], v36, v32 // (op 432) least significant half
v_addc_co_u32 v37, s[50:51], 0, v37, s[50:51] // (op 432) most significant half

s_add_u32 m0, m0, 2048 // (op 432) 
buffer_load_dwordx4 v36, s[56:59], s18 offen offset: 0 lds // (op 432) Load value direct to lds








v_add_co_u32 v14, vcc, 256, v14 // (op 556) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 556) most significant half
buffer_load_ubyte v18, v29, s[64:67], 0 offen offset:12 // (op 242) Load value




v_add_u32 v20, 8, v20 // (op 684) 
buffer_load_dwordx3 v[26:28], v31, s[52:55], 0 offen offset:288 // (op 228) Load value

s_barrier  // (op 1059) 
v_add_u32 v11, 192, v11 // (op 492) 



s_waitcnt vmcnt(6)
ds_write_b8 v13, v21 offset:8192 // (op 26) Store local data 
s_barrier  // (op 1033) 



s_waitcnt vmcnt(5)
ds_write_b96 v17, v[22:24] offset:8320 // (op 6) Store local data 
s_ashr_i64 s[18:19], s[34:35], 7 // (op 56) 
s_mov_b32 s9, s18 // (op 56) convert

s_and_b32 s18, s9, 4294967294 // (op 56) 

s_cmp_lt_u32 s8, s18 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56 if true)
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56:














s_barrier  // (op 1098) 
s_mul_hi_i32 s0, s16, s12 // (op 455) 
v_lshrrev_b32 v10, 3, v1 // (op 455) 
s_add_i32 s1, s0, s16 // (op 455) 

v_and_b32 v11, -8, v1 // (op 455) 
s_ashr_i32 s0, s1, 31 // (op 455) 
v_sub_u32 v12, v1, v11 // (op 455) 

s_cmp_eq_i32 s12, 0 // (op 455) 
s_mov_b32 s8, scc // (op 455) 
v_lshlrev_b32 v11, 4, v12 // (op 455) 

s_cmp_eq_i32 s8, 0 // (op 455) 
s_cselect_b32 s9, 0, -1 // (op 455) 

s_and_b32 s8, s13, 31 // (op 455) 
s_lshl_b32 s18, 1, s8 // (op 455) 
s_add_i32 s19, s18, s9 // (op 455) 


s_and_b32 s9, s0, s19 // (op 455) 


s_add_i32 s0, s1, s9 // (op 455) 


s_ashr_i32 s1, s0, s8 // (op 455) 


s_lshl_b32 s0, s13, 24 // (op 455) 
s_ashr_i32 s8, s0, 31 // (op 455) 

s_xor_b32 s0, s1, s8 // (op 455) 

s_sub_i32 s1, s0, s8 // (op 455) 


s_mul_hi_i32 s0, s1, s14 // (op 455) 
s_add_i32 s8, s0, s1 // (op 455) 

s_ashr_i32 s0, s8, 31 // (op 455) 
s_cmp_eq_i32 s14, 0 // (op 455) 
s_mov_b32 s9, scc // (op 455) 
s_cmp_eq_i32 s9, 0 // (op 455) 
s_cselect_b32 s18, 0, -1 // (op 455) 

s_and_b32 s9, s15, 31 // (op 455) 
s_lshl_b32 s19, 1, s9 // (op 455) 
s_add_i32 s49, s19, s18 // (op 455) 


s_and_b32 s18, s0, s49 // (op 455) 


s_add_i32 s0, s8, s18 // (op 455) 


s_ashr_i32 s8, s0, s9 // (op 455) 


s_lshl_b32 s0, s15, 24 // (op 455) 
s_ashr_i32 s9, s0, 31 // (op 455) 

s_xor_b32 s0, s8, s9 // (op 455) 

s_sub_i32 s8, s0, s9 // (op 455) 


s_mul_i32 s0, s1, s45 // (op 455) 

s_sub_u32 s1, s16, s0 // (op 455) 

s_mul_i32 s0, s8, s45 // (op 455) 
s_add_u32 s0, s0, s1 // (op 455) 


s_mul_hi_i32 s1, s16, s11 // (op 455) 
s_add_i32 s8, s1, s16 // (op 455) 

s_ashr_i32 s1, s8, 31 // (op 455) 
s_cmp_eq_i32 s11, 0 // (op 455) 
s_mov_b32 s9, scc // (op 455) 
s_cmp_eq_i32 s9, 0 // (op 455) 
s_cselect_b32 s18, 0, -1 // (op 455) 

s_and_b32 s9, s46, 31 // (op 455) 
s_lshl_b32 s19, 1, s9 // (op 455) 
s_add_i32 s49, s19, s18 // (op 455) 


s_and_b32 s18, s1, s49 // (op 455) 


s_add_i32 s1, s8, s18 // (op 455) 


s_ashr_i32 s8, s1, s9 // (op 455) 


s_lshl_b32 s1, s46, 24 // (op 455) 
s_ashr_i32 s9, s1, 31 // (op 455) 

s_xor_b32 s1, s8, s9 // (op 455) 

s_sub_i32 s8, s1, s9 // (op 455) 


s_add_u32 s18, s4, 32 // (op 455) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 455) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 455) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 455) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 455) 

s_mul_i32 s1, s45, s18 // (op 455) 

s_mul_i32 s9, s10, s1 // (op 455) 

s_mul_i32 s1, s8, s9 // (op 455) 

s_sub_u32 s9, s16, s1 // (op 455) 

s_mov_b32 s1, s9 // (op 455) 
s_mul_hi_i32 s18, s1, s44 // (op 455) 
s_add_i32 s19, s18, s1 // (op 455) 


s_ashr_i32 s1, s19, 31 // (op 455) 
s_cmp_eq_i32 s44, 0 // (op 455) 
s_mov_b32 s18, scc // (op 455) 
s_cmp_eq_i32 s18, 0 // (op 455) 
s_cselect_b32 s49, 0, -1 // (op 455) 

s_and_b32 s18, s48, 31 // (op 455) 
s_lshl_b32 s50, 1, s18 // (op 455) 
s_add_i32 s51, s50, s49 // (op 455) 


s_and_b32 s49, s1, s51 // (op 455) 


s_add_i32 s1, s19, s49 // (op 455) 


s_ashr_i32 s19, s1, s18 // (op 455) 


s_lshl_b32 s1, s48, 24 // (op 455) 
s_ashr_i32 s18, s1, 31 // (op 455) 

s_xor_b32 s1, s19, s18 // (op 455) 

s_sub_i32 s19, s1, s18 // (op 455) 


s_mul_i32 s1, s47, s45 // (op 455) 
s_add_u32 s50, s6, 16 // (op 455) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 455) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 455) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 455) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 455) 

s_sub_i32 s18, s50, s1 // (op 455) 

s_mul_i32 s49, s19, s18 // (op 455) 


s_sub_u32 s18, s9, s49 // (op 455) 


s_add_u32 s9, s18, s1 // (op 455) 


s_cmp_eq_i32 s8, 0 // (op 455) 
s_mov_b32 s1, scc // (op 455) 

s_cmp_eq_i32 s1, 0 // (op 455) 
s_cselect_b32 s8, s9, s0 // (op 455) 



v_lshl_add_u32 v12, s8, 4, v10 // (op 455) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 455) 
s_lshr_b32 s8, s0, 1 // (op 455) 

v_lshl_add_u32 v10, s8, 8, v11 // (op 455) 


v_mov_b32 v17, 0 // (op 455) convert
v_mov_b32 v16, v10 // (op 455) convert

v_mul_lo_u32 v11, v12, s3 // (op 455) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 455) most significant: high of low * low
v_mul_lo_u32 v10, v12, s2 // (op 455) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 455) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v16 // (op 455) least significant half
v_addc_co_u32 v11, s[0:1], v11, v17, s[0:1] // (op 455) most significant half



v_mul_lo_u32 v13, v11, 6 // (op 455) most significant: low of high * low
v_mul_hi_u32 v14, v10, 6 // (op 455) most significant: high of low * low
v_mul_lo_u32 v12, v10, 6 // (op 455) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 455) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 455) 

v_mov_b32 v12, v10 // (op 455) convert

s_mul_i32 s1, s7, 6 // (op 457) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 457) least significant
s_mul_hi_u32 s8, s6, 6 // (op 457) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 457) most significant: sum

s_mul_i32 s9, s2, s1 // (op 457) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 457) most significant: low of high * low
s_mul_i32 s8, s2, s0 // (op 457) least significant
s_mul_hi_u32 s19, s2, s0 // (op 457) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 457) most significant: sum
s_add_u32 s9, s9, s19 // (op 457) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 457) 

s_mov_b32 s52, s20 // (op 457) reg expression
s_mov_b32 s53, s21 // (op 457) reg expression
s_mov_b32 s54, s0 // (op 457) reg expression
s_mov_b32 s55, 131072 // (op 457) call()

s_mul_hi_i32 s0, s16, s12 // (op 522) 
v_lshrrev_b32 v10, 3, v1 // (op 522) 
s_add_i32 s1, s0, s16 // (op 522) 

v_and_b32 v11, -8, v1 // (op 522) 
s_ashr_i32 s0, s1, 31 // (op 522) 
v_sub_u32 v13, v1, v11 // (op 522) 

s_cmp_eq_i32 s12, 0 // (op 522) 
s_mov_b32 s8, scc // (op 522) 
v_lshlrev_b32 v11, 4, v13 // (op 522) 

s_cmp_eq_i32 s8, 0 // (op 522) 
s_cselect_b32 s9, 0, -1 // (op 522) 

s_and_b32 s8, s13, 31 // (op 522) 
s_lshl_b32 s18, 1, s8 // (op 522) 
s_add_i32 s19, s18, s9 // (op 522) 


s_and_b32 s9, s0, s19 // (op 522) 


s_add_i32 s0, s1, s9 // (op 522) 


s_ashr_i32 s1, s0, s8 // (op 522) 


s_lshl_b32 s0, s13, 24 // (op 522) 
s_ashr_i32 s8, s0, 31 // (op 522) 

s_xor_b32 s0, s1, s8 // (op 522) 

s_sub_i32 s1, s0, s8 // (op 522) 


s_mul_hi_i32 s0, s1, s14 // (op 522) 
s_add_i32 s8, s0, s1 // (op 522) 

s_ashr_i32 s0, s8, 31 // (op 522) 
s_cmp_eq_i32 s14, 0 // (op 522) 
s_mov_b32 s9, scc // (op 522) 
s_cmp_eq_i32 s9, 0 // (op 522) 
s_cselect_b32 s18, 0, -1 // (op 522) 

s_and_b32 s9, s15, 31 // (op 522) 
s_lshl_b32 s19, 1, s9 // (op 522) 
s_add_i32 s49, s19, s18 // (op 522) 


s_and_b32 s18, s0, s49 // (op 522) 


s_add_i32 s0, s8, s18 // (op 522) 


s_ashr_i32 s8, s0, s9 // (op 522) 


s_lshl_b32 s0, s15, 24 // (op 522) 
s_ashr_i32 s9, s0, 31 // (op 522) 

s_xor_b32 s0, s8, s9 // (op 522) 

s_sub_i32 s8, s0, s9 // (op 522) 


s_add_u32 s18, s4, 32 // (op 522) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 522) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 522) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 522) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 522) 

s_mul_i32 s0, s8, s18 // (op 522) 

s_sub_i32 s8, s1, s0 // (op 522) 


s_mul_hi_i32 s9, s16, s11 // (op 522) 
s_add_i32 s0, s9, s16 // (op 522) 

s_ashr_i32 s1, s0, 31 // (op 522) 
s_cmp_eq_i32 s11, 0 // (op 522) 
s_mov_b32 s9, scc // (op 522) 
s_cmp_eq_i32 s9, 0 // (op 522) 
s_cselect_b32 s49, 0, -1 // (op 522) 

s_and_b32 s9, s46, 31 // (op 522) 
s_lshl_b32 s50, 1, s9 // (op 522) 
s_add_i32 s51, s50, s49 // (op 522) 


s_and_b32 s49, s1, s51 // (op 522) 


s_add_i32 s1, s0, s49 // (op 522) 


s_ashr_i32 s0, s1, s9 // (op 522) 


s_lshl_b32 s1, s46, 24 // (op 522) 
s_ashr_i32 s9, s1, 31 // (op 522) 

s_xor_b32 s1, s0, s9 // (op 522) 

s_sub_i32 s0, s1, s9 // (op 522) 


s_mul_i32 s1, s45, s18 // (op 522) 

s_mul_i32 s9, s10, s1 // (op 522) 

s_mul_i32 s1, s0, s9 // (op 522) 

s_sub_u32 s9, s16, s1 // (op 522) 

s_mul_hi_i32 s1, s9, s44 // (op 522) 
s_add_i32 s18, s1, s9 // (op 522) 


s_ashr_i32 s1, s18, 31 // (op 522) 
s_cmp_eq_i32 s44, 0 // (op 522) 
s_mov_b32 s9, scc // (op 522) 
s_cmp_eq_i32 s9, 0 // (op 522) 
s_cselect_b32 s19, 0, -1 // (op 522) 

s_and_b32 s9, s48, 31 // (op 522) 
s_lshl_b32 s49, 1, s9 // (op 522) 
s_add_i32 s50, s49, s19 // (op 522) 


s_and_b32 s19, s1, s50 // (op 522) 


s_add_i32 s1, s18, s19 // (op 522) 


s_ashr_i32 s18, s1, s9 // (op 522) 


s_lshl_b32 s1, s48, 24 // (op 522) 
s_ashr_i32 s9, s1, 31 // (op 522) 

s_xor_b32 s1, s18, s9 // (op 522) 

s_sub_i32 s18, s1, s9 // (op 522) 


s_cmp_eq_i32 s0, 0 // (op 522) 
s_mov_b32 s1, scc // (op 522) 

s_cmp_eq_i32 s1, 0 // (op 522) 
s_cselect_b32 s0, s18, s8 // (op 522) 



v_lshl_add_u32 v13, s0, 5, v10 // (op 522) 


v_mul_lo_u32 v15, v13, s23 // (op 522) most significant: low of low * high
v_mul_hi_u32 v10, v13, s22 // (op 522) most significant: high of low * low
v_mul_lo_u32 v14, v13, s22 // (op 522) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 522) most significant: sum


s_ashr_i64 s[0:1], s[34:35], 7 // (op 522) 
s_lshr_b32 s8, s0, 1 // (op 522) 

v_lshl_add_u32 v10, s8, 8, v11 // (op 522) 


v_mov_b32 v17, 0 // (op 522) convert
v_mov_b32 v16, v10 // (op 522) convert

v_add_co_u32 v10, s[0:1], v16, v14 // (op 522) least significant half
v_addc_co_u32 v11, s[0:1], v17, v15, s[0:1] // (op 522) most significant half



s_mul_i32 s1, s22, s5 // (op 524) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 524) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 524) least significant
s_mul_hi_u32 s9, s22, s4 // (op 524) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 524) most significant: sum
s_add_u32 s1, s1, s9 // (op 524) most significant: sum


s_mov_b32 s56, s24 // (op 524) reg expression
s_mov_b32 s57, s25 // (op 524) reg expression
s_mov_b32 s58, s0 // (op 524) reg expression
s_mov_b32 s59, 131072 // (op 524) call()

s_mul_hi_i32 s0, s16, s12 // (op 586) 
v_lshrrev_b32 v13, 7, v1 // (op 586) 
s_add_i32 s1, s0, s16 // (op 586) 

v_and_b32 v14, 15, v1 // (op 586) 
s_ashr_i32 s0, s1, 31 // (op 586) 
v_lshl_add_u32 v15, v13, 4, v14 // (op 586) 


s_cmp_eq_i32 s12, 0 // (op 586) 
s_mov_b32 s8, scc // (op 586) 
v_and_b32 v13, 63, v1 // (op 586) 
s_cmp_eq_i32 s8, 0 // (op 586) 
s_cselect_b32 s9, 0, -1 // (op 586) 

v_lshrrev_b32 v14, 4, v13 // (op 586) 

s_and_b32 s8, s13, 31 // (op 586) 
s_lshl_b32 s18, 1, s8 // (op 586) 
s_add_i32 s19, s18, s9 // (op 586) 


s_and_b32 s9, s0, s19 // (op 586) 


s_add_i32 s0, s1, s9 // (op 586) 


s_ashr_i32 s1, s0, s8 // (op 586) 


s_lshl_b32 s0, s13, 24 // (op 586) 
s_ashr_i32 s8, s0, 31 // (op 586) 

s_xor_b32 s0, s1, s8 // (op 586) 

s_sub_i32 s1, s0, s8 // (op 586) 


s_mul_hi_i32 s0, s1, s14 // (op 586) 
s_add_i32 s8, s0, s1 // (op 586) 

s_ashr_i32 s0, s8, 31 // (op 586) 
s_cmp_eq_i32 s14, 0 // (op 586) 
s_mov_b32 s9, scc // (op 586) 
s_cmp_eq_i32 s9, 0 // (op 586) 
s_cselect_b32 s18, 0, -1 // (op 586) 

s_and_b32 s9, s15, 31 // (op 586) 
s_lshl_b32 s19, 1, s9 // (op 586) 
s_add_i32 s49, s19, s18 // (op 586) 


s_and_b32 s18, s0, s49 // (op 586) 


s_add_i32 s0, s8, s18 // (op 586) 


s_ashr_i32 s8, s0, s9 // (op 586) 


s_lshl_b32 s0, s15, 24 // (op 586) 
s_ashr_i32 s9, s0, 31 // (op 586) 

s_xor_b32 s0, s8, s9 // (op 586) 

s_sub_i32 s8, s0, s9 // (op 586) 


s_mul_i32 s0, s1, s45 // (op 586) 

s_sub_u32 s1, s16, s0 // (op 586) 

s_mul_i32 s0, s8, s45 // (op 586) 
s_add_u32 s0, s0, s1 // (op 586) 


s_mul_hi_i32 s1, s16, s11 // (op 586) 
s_add_i32 s8, s1, s16 // (op 586) 

s_ashr_i32 s1, s8, 31 // (op 586) 
s_cmp_eq_i32 s11, 0 // (op 586) 
s_mov_b32 s9, scc // (op 586) 
s_cmp_eq_i32 s9, 0 // (op 586) 
s_cselect_b32 s18, 0, -1 // (op 586) 

s_and_b32 s9, s46, 31 // (op 586) 
s_lshl_b32 s19, 1, s9 // (op 586) 
s_add_i32 s49, s19, s18 // (op 586) 


s_and_b32 s18, s1, s49 // (op 586) 


s_add_i32 s1, s8, s18 // (op 586) 


s_ashr_i32 s8, s1, s9 // (op 586) 


s_lshl_b32 s1, s46, 24 // (op 586) 
s_ashr_i32 s9, s1, 31 // (op 586) 

s_xor_b32 s1, s8, s9 // (op 586) 

s_sub_i32 s8, s1, s9 // (op 586) 


s_add_u32 s18, s4, 32 // (op 586) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 586) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 586) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 586) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 586) 

s_mul_i32 s1, s45, s18 // (op 586) 

s_mul_i32 s9, s10, s1 // (op 586) 

s_mul_i32 s1, s8, s9 // (op 586) 

s_sub_u32 s9, s16, s1 // (op 586) 

s_mov_b32 s1, s9 // (op 586) 
s_mul_hi_i32 s18, s1, s44 // (op 586) 
s_add_i32 s19, s18, s1 // (op 586) 


s_ashr_i32 s1, s19, 31 // (op 586) 
s_cmp_eq_i32 s44, 0 // (op 586) 
s_mov_b32 s18, scc // (op 586) 
s_cmp_eq_i32 s18, 0 // (op 586) 
s_cselect_b32 s49, 0, -1 // (op 586) 

s_and_b32 s18, s48, 31 // (op 586) 
s_lshl_b32 s50, 1, s18 // (op 586) 
s_add_i32 s51, s50, s49 // (op 586) 


s_and_b32 s49, s1, s51 // (op 586) 


s_add_i32 s1, s19, s49 // (op 586) 


s_ashr_i32 s19, s1, s18 // (op 586) 


s_lshl_b32 s1, s48, 24 // (op 586) 
s_ashr_i32 s18, s1, 31 // (op 586) 

s_xor_b32 s1, s19, s18 // (op 586) 

s_sub_i32 s19, s1, s18 // (op 586) 


s_mul_i32 s1, s47, s45 // (op 586) 
s_add_u32 s50, s6, 16 // (op 586) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 586) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 586) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 586) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 586) 

s_sub_i32 s18, s50, s1 // (op 586) 

s_mul_i32 s49, s19, s18 // (op 586) 


s_sub_u32 s18, s9, s49 // (op 586) 


s_add_u32 s9, s18, s1 // (op 586) 


s_cmp_eq_i32 s8, 0 // (op 586) 
s_mov_b32 s1, scc // (op 586) 

s_cmp_eq_i32 s1, 0 // (op 586) 
s_cselect_b32 s8, s9, s0 // (op 586) 



v_lshl_add_u32 v13, s8, 4, v15 // (op 586) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 586) 
s_lshr_b32 s8, s0, 1 // (op 586) 

v_lshl_add_u32 v15, s8, 3, v14 // (op 586) 


v_mul_lo_u32 v14, v13, s26 // (op 586) 
v_add_u32 v14, v14, v15 // (op 586) 


s_mul_i32 s1, s26, s7 // (op 588) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 588) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 588) least significant
s_mul_hi_u32 s9, s26, s6 // (op 588) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 588) most significant: sum
s_add_u32 s1, s1, s9 // (op 588) most significant: sum


s_mov_b32 s60, s28 // (op 588) reg expression
s_mov_b32 s61, s29 // (op 588) reg expression
s_mov_b32 s62, s0 // (op 588) reg expression
s_mov_b32 s63, 131072 // (op 588) call()

s_mul_hi_i32 s0, s16, s12 // (op 650) 
v_lshrrev_b32 v13, 2, v1 // (op 650) 
s_add_i32 s1, s0, s16 // (op 650) 

v_and_b32 v15, -4, v1 // (op 650) 
s_ashr_i32 s0, s1, 31 // (op 650) 
v_sub_u32 v16, v1, v15 // (op 650) 

s_cmp_eq_i32 s12, 0 // (op 650) 
s_mov_b32 s8, scc // (op 650) 
s_cmp_eq_i32 s8, 0 // (op 650) 
s_cselect_b32 s9, 0, -1 // (op 650) 

s_and_b32 s8, s13, 31 // (op 650) 
s_lshl_b32 s18, 1, s8 // (op 650) 
s_add_i32 s19, s18, s9 // (op 650) 


s_and_b32 s9, s0, s19 // (op 650) 


s_add_i32 s0, s1, s9 // (op 650) 


s_ashr_i32 s1, s0, s8 // (op 650) 


s_lshl_b32 s0, s13, 24 // (op 650) 
s_ashr_i32 s8, s0, 31 // (op 650) 

s_xor_b32 s0, s1, s8 // (op 650) 

s_sub_i32 s1, s0, s8 // (op 650) 


s_mul_hi_i32 s0, s1, s14 // (op 650) 
s_add_i32 s8, s0, s1 // (op 650) 

s_ashr_i32 s0, s8, 31 // (op 650) 
s_cmp_eq_i32 s14, 0 // (op 650) 
s_mov_b32 s9, scc // (op 650) 
s_cmp_eq_i32 s9, 0 // (op 650) 
s_cselect_b32 s18, 0, -1 // (op 650) 

s_and_b32 s9, s15, 31 // (op 650) 
s_lshl_b32 s19, 1, s9 // (op 650) 
s_add_i32 s49, s19, s18 // (op 650) 


s_and_b32 s18, s0, s49 // (op 650) 


s_add_i32 s0, s8, s18 // (op 650) 


s_ashr_i32 s8, s0, s9 // (op 650) 


s_lshl_b32 s0, s15, 24 // (op 650) 
s_ashr_i32 s9, s0, 31 // (op 650) 

s_xor_b32 s0, s8, s9 // (op 650) 

s_sub_i32 s8, s0, s9 // (op 650) 


s_add_u32 s18, s4, 32 // (op 650) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 650) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 650) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 650) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 650) 

s_mul_i32 s0, s8, s18 // (op 650) 

s_sub_i32 s8, s1, s0 // (op 650) 


s_mul_hi_i32 s9, s16, s11 // (op 650) 
s_add_i32 s0, s9, s16 // (op 650) 

s_ashr_i32 s1, s0, 31 // (op 650) 
s_cmp_eq_i32 s11, 0 // (op 650) 
s_mov_b32 s9, scc // (op 650) 
s_cmp_eq_i32 s9, 0 // (op 650) 
s_cselect_b32 s49, 0, -1 // (op 650) 

s_and_b32 s9, s46, 31 // (op 650) 
s_lshl_b32 s50, 1, s9 // (op 650) 
s_add_i32 s51, s50, s49 // (op 650) 


s_and_b32 s49, s1, s51 // (op 650) 


s_add_i32 s1, s0, s49 // (op 650) 


s_ashr_i32 s0, s1, s9 // (op 650) 


s_lshl_b32 s1, s46, 24 // (op 650) 
s_ashr_i32 s9, s1, 31 // (op 650) 

s_xor_b32 s1, s0, s9 // (op 650) 

s_sub_i32 s0, s1, s9 // (op 650) 


s_mul_i32 s1, s45, s18 // (op 650) 

s_mul_i32 s9, s10, s1 // (op 650) 

s_mul_i32 s1, s0, s9 // (op 650) 

s_sub_u32 s9, s16, s1 // (op 650) 

s_mul_hi_i32 s1, s9, s44 // (op 650) 
s_add_i32 s18, s1, s9 // (op 650) 


s_ashr_i32 s1, s18, 31 // (op 650) 
s_cmp_eq_i32 s44, 0 // (op 650) 
s_mov_b32 s9, scc // (op 650) 
s_cmp_eq_i32 s9, 0 // (op 650) 
s_cselect_b32 s19, 0, -1 // (op 650) 

s_and_b32 s9, s48, 31 // (op 650) 
s_lshl_b32 s49, 1, s9 // (op 650) 
s_add_i32 s50, s49, s19 // (op 650) 


s_and_b32 s19, s1, s50 // (op 650) 


s_add_i32 s1, s18, s19 // (op 650) 


s_ashr_i32 s18, s1, s9 // (op 650) 


s_lshl_b32 s1, s48, 24 // (op 650) 
s_ashr_i32 s9, s1, 31 // (op 650) 

s_xor_b32 s1, s18, s9 // (op 650) 

s_sub_i32 s18, s1, s9 // (op 650) 


s_cmp_eq_i32 s0, 0 // (op 650) 
s_mov_b32 s1, scc // (op 650) 

s_cmp_eq_i32 s1, 0 // (op 650) 
s_cselect_b32 s0, s18, s8 // (op 650) 



v_lshl_add_u32 v15, s0, 5, v13 // (op 650) 


v_mul_lo_u32 v13, v15, s30 // (op 650) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 650) 
s_lshr_b32 s8, s0, 1 // (op 650) 

v_lshl_add_u32 v15, s8, 3, v16 // (op 650) 


v_add_u32 v16, v15, v13 // (op 650) 


s_mul_i32 s1, s30, s5 // (op 652) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 652) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 652) least significant
s_mul_hi_u32 s9, s30, s4 // (op 652) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 652) most significant: sum
s_add_u32 s1, s1, s9 // (op 652) most significant: sum


s_mov_b32 s64, s32 // (op 652) reg expression
s_mov_b32 s65, s33 // (op 652) reg expression
s_mov_b32 s66, s0 // (op 652) reg expression
s_mov_b32 s67, 131072 // (op 652) call()

v_and_b32 v13, -8, v1 // (op 743) 
v_sub_u32 v15, v1, v13 // (op 743) 

v_lshlrev_b32 v13, 4, v15 // (op 743) 

v_lshrrev_b32 v15, 3, v1 // (op 743) 
v_lshl_add_u32 v17, v15, 7, v13 // (op 743) 


v_mul_lo_u32 v13, v17, 6 // (op 743) 

v_lshrrev_b32 v15, 3, v13 // (op 743) 

v_and_b32 v13, -8, v1 // (op 796) 
v_sub_u32 v17, v1, v13 // (op 796) 

v_lshlrev_b32 v13, 4, v17 // (op 796) 

v_lshrrev_b32 v17, 3, v1 // (op 796) 
v_lshl_add_u32 v19, v17, 7, v13 // (op 796) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 796) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -4, v1 // (op 848) 
v_sub_u32 v17, v1, v13 // (op 848) 

v_and_b32 v13, 4294967292, v1 // (op 848) 
v_add_u32 v19, v13, v17 // (op 848) 


s_ashr_i64 s[8:9], s[34:35], 7 // (op 166) 
s_and_b32 s1, s8, 4294967294 // (op 166) 

s_ashr_i64 s[8:9], s[34:35], 7 // (op 165) 
s_mov_b32 s18, s8 // (op 165) convert

s_cmp_lt_u32 s1, s18 // (op 165) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165:


s_add_u32 s1, s1, 1 // (op 167) 
v_mov_b32 v13, v12 // (op 170) 

buffer_load_dwordx3 v[20:22], v13, s[52:55], 0 offen // (op 170) Load value

v_add_u32 v12, 96, v12 // (op 445) 
s_barrier  // (op 1036) 
s_barrier  // (op 1049) 




s_lshl_b64 s[8:9], s[22:23], 4 // (op 425) 
v_mov_b64 v[24:25], s[8:9] // (op 425) convert


v_mov_b64 v[26:27], v[10:11] // (op 425) 
v_mov_b64 v[28:29], v[26:27] // (op 425) 
s_mov_b32 m0, s0 // (op 425) reg expression
buffer_load_dwordx4 v28, s[56:59], 0 offen offset: 0 lds // (op 425) Load value direct to lds
v_add_co_u32 v28, s[8:9], v28, v24 // (op 425) least significant half
v_addc_co_u32 v29, s[8:9], 0, v29, s[8:9] // (op 425) most significant half

s_add_u32 m0, m0, 2048 // (op 425) 
buffer_load_dwordx4 v28, s[56:59], 0 offen offset: 0 lds // (op 425) Load value direct to lds



v_add_co_u32 v10, vcc, 128, v10 // (op 514) least significant half
v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 514) most significant half
s_waitcnt vmcnt(2)
ds_write_b96 v15, v[20:22] offset:8320 // (op 171) Store local data 



v_mov_b32 v13, v16 // (op 184) 

buffer_load_ubyte v17, v13, s[64:67], 0 offen offset: 0 // (op 184) Load value

v_add_u32 v16, 4, v16 // (op 642) 
s_barrier  // (op 1062) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1039) 






ds_read_b128 v[20:23], v3 offset:8320 // (op 173) Load local data 
ds_read_b64 v[24:25], v3 offset:8336 // (op 173) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v17 offset:8192 // (op 182) Store local data 

s_barrier  // (op 1052) 



ds_read_b128 v[26:29], v4 // (op 175) Load local data 
ds_read_b128 v[30:33], v4 offset:64 // (op 175) Load local data 


v_mov_b32 v13, v14 // (op 177) 

buffer_load_ubyte v17, v13, s[60:63], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v14, 4, v14 // (op 578) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1066) 



ds_read_u8 v13, v0 offset:8192 // (op 179) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:33], a[0:3], v17, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 180) 




s_ashr_i64 s[8:9], s[34:35], 7 // (op 165) 
s_mov_b32 s18, s8 // (op 165) convert

s_cmp_lt_u32 s1, s18 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165 if true)
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165:





















s_barrier  // (op 1130) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

v_cvt_f16_f32 v4, v10 // (op 52) 
v_cvt_f16_f32 v5, v11 // (op 52) 
v_cvt_f16_f32 v6, v12 // (op 52) 
v_cvt_f16_f32 v7, v13 // (op 52) 

s_mul_hi_i32 s0, s16, s12 // (op 890) 
v_lshrrev_b32 v0, 7, v1 // (op 890) 
s_add_i32 s1, s0, s16 // (op 890) 

v_and_b32 v3, 63, v1 // (op 890) 
s_ashr_i32 s0, s1, 31 // (op 890) 
v_lshrrev_b32 v10, 4, v3 // (op 890) 
s_cmp_eq_i32 s12, 0 // (op 890) 
s_mov_b32 s2, scc // (op 890) 
v_lshlrev_b32 v11, 2, v10 // (op 890) 

s_cmp_eq_i32 s2, 0 // (op 890) 
s_cselect_b32 s3, 0, -1 // (op 890) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 890) 


s_and_b32 s2, s13, 31 // (op 890) 
v_lshrrev_b32 v0, 2, v3 // (op 890) 

s_lshl_b32 s8, 1, s2 // (op 890) 
v_and_b32 v3, 3, v0 // (op 890) 

s_add_i32 s9, s8, s3 // (op 890) 


v_and_b32 v0, 3, v1 // (op 890) 
s_and_b32 s3, s0, s9 // (op 890) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 890) 


s_add_i32 s0, s1, s3 // (op 890) 


v_lshrrev_b32 v0, 6, v1 // (op 890) 
s_ashr_i32 s1, s0, s2 // (op 890) 


v_and_b32 v3, 1, v0 // (op 890) 

s_lshl_b32 s0, s13, 24 // (op 890) 
v_lshl_add_u32 v0, v3, 4, v11 // (op 890) 


s_ashr_i32 s2, s0, 31 // (op 890) 

s_xor_b32 s0, s1, s2 // (op 890) 

s_sub_i32 s1, s0, s2 // (op 890) 


s_mul_hi_i32 s0, s1, s14 // (op 890) 
s_add_i32 s2, s0, s1 // (op 890) 

s_ashr_i32 s0, s2, 31 // (op 890) 
s_cmp_eq_i32 s14, 0 // (op 890) 
s_mov_b32 s3, scc // (op 890) 
s_cmp_eq_i32 s3, 0 // (op 890) 
s_cselect_b32 s8, 0, -1 // (op 890) 

s_and_b32 s3, s15, 31 // (op 890) 
s_lshl_b32 s9, 1, s3 // (op 890) 
s_add_i32 s18, s9, s8 // (op 890) 


s_and_b32 s8, s0, s18 // (op 890) 


s_add_i32 s0, s2, s8 // (op 890) 


s_ashr_i32 s2, s0, s3 // (op 890) 


s_lshl_b32 s0, s15, 24 // (op 890) 
s_ashr_i32 s3, s0, 31 // (op 890) 

s_xor_b32 s0, s2, s3 // (op 890) 

s_sub_i32 s2, s0, s3 // (op 890) 


s_mul_i32 s0, s1, s45 // (op 890) 
s_sub_u32 s3, s16, s0 // (op 890) 

s_mul_i32 s0, s2, s45 // (op 890) 
s_add_u32 s0, s0, s3 // (op 890) 

s_mul_hi_i32 s3, s16, s11 // (op 890) 
s_add_i32 s8, s3, s16 // (op 890) 

s_ashr_i32 s3, s8, 31 // (op 890) 
s_cmp_eq_i32 s11, 0 // (op 890) 
s_mov_b32 s9, scc // (op 890) 
s_cmp_eq_i32 s9, 0 // (op 890) 
s_cselect_b32 s18, 0, -1 // (op 890) 

s_and_b32 s9, s46, 31 // (op 890) 
s_lshl_b32 s19, 1, s9 // (op 890) 
s_add_i32 s20, s19, s18 // (op 890) 


s_and_b32 s18, s3, s20 // (op 890) 


s_add_i32 s3, s8, s18 // (op 890) 


s_ashr_i32 s8, s3, s9 // (op 890) 


s_lshl_b32 s3, s46, 24 // (op 890) 
s_ashr_i32 s9, s3, 31 // (op 890) 

s_xor_b32 s3, s8, s9 // (op 890) 

s_sub_i32 s8, s3, s9 // (op 890) 


s_add_u32 s18, s4, 32 // (op 890) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 890) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 890) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 890) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 890) 

s_mul_i32 s3, s45, s18 // (op 890) 
s_mul_i32 s9, s10, s3 // (op 890) 

s_mul_i32 s3, s8, s9 // (op 890) 

s_sub_u32 s9, s16, s3 // (op 890) 

s_mov_b32 s3, s9 // (op 890) 
s_mul_hi_i32 s20, s3, s44 // (op 890) 
s_add_i32 s21, s20, s3 // (op 890) 


s_ashr_i32 s3, s21, 31 // (op 890) 
s_cmp_eq_i32 s44, 0 // (op 890) 
s_mov_b32 s20, scc // (op 890) 
s_cmp_eq_i32 s20, 0 // (op 890) 
s_cselect_b32 s22, 0, -1 // (op 890) 

s_and_b32 s20, s48, 31 // (op 890) 
s_lshl_b32 s23, 1, s20 // (op 890) 
s_add_i32 s24, s23, s22 // (op 890) 


s_and_b32 s22, s3, s24 // (op 890) 


s_add_i32 s3, s21, s22 // (op 890) 


s_ashr_i32 s21, s3, s20 // (op 890) 


s_lshl_b32 s3, s48, 24 // (op 890) 
s_ashr_i32 s20, s3, 31 // (op 890) 

s_xor_b32 s3, s21, s20 // (op 890) 

s_sub_i32 s21, s3, s20 // (op 890) 


s_mul_i32 s3, s47, s45 // (op 890) 
s_add_u32 s22, s6, 16 // (op 890) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 890) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 890) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 890) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 890) 

s_sub_i32 s20, s22, s3 // (op 890) 

s_mul_i32 s22, s21, s20 // (op 890) 

s_sub_u32 s20, s9, s22 // (op 890) 


s_add_u32 s9, s20, s3 // (op 890) 


s_cmp_eq_i32 s8, 0 // (op 890) 
s_mov_b32 s3, scc // (op 890) 

s_cmp_eq_i32 s3, 0 // (op 890) 
s_cselect_b32 s8, s9, s0 // (op 890) 


s_mul_i32 s0, s2, s18 // (op 890) 


v_lshl_add_u32 v3, s8, 4, v10 // (op 890) 


s_sub_i32 s2, s1, s0 // (op 890) 


s_cmp_eq_i32 s3, 0 // (op 890) 
s_cselect_b32 s0, s21, s2 // (op 890) 



v_lshl_add_u32 v10, s0, 5, v0 // (op 890) 


v_mul_lo_u32 v0, v10, s36 // (op 890) 

v_add_lshl_u32 v10, v3, v0, 1 // (op 890) 


s_mul_i32 s9, s36, s5 // (op 892) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 892) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 892) least significant
s_mul_hi_u32 s1, s36, s4 // (op 892) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 892) most significant: sum
s_add_u32 s9, s9, s1 // (op 892) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 892) 

s_mov_b32 s32, s38 // (op 892) reg expression
s_mov_b32 s33, s39 // (op 892) reg expression
s_mov_b32 s34, s0 // (op 892) reg expression
s_mov_b32 s35, 131072 // (op 892) call()


v_pack_b32_f16 v12, v4, v5 // (op 54) 
v_pack_b32_f16 v13, v6, v7 // (op 54) 


buffer_store_dwordx2 v[12:13], v10, s[32:35], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:
s_mul_hi_i32 s0, s16, s12 // (op 713) 
v_lshrrev_b32 v0, 7, v1 // (op 713) 
s_add_i32 s1, s0, s16 // (op 713) 

v_and_b32 v3, 63, v1 // (op 713) 
s_ashr_i32 s0, s1, 31 // (op 713) 
v_lshrrev_b32 v4, 4, v3 // (op 713) 
s_cmp_eq_i32 s12, 0 // (op 713) 
s_mov_b32 s2, scc // (op 713) 
v_lshlrev_b32 v5, 2, v4 // (op 713) 

s_cmp_eq_i32 s2, 0 // (op 713) 
s_cselect_b32 s3, 0, -1 // (op 713) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 713) 


s_and_b32 s2, s13, 31 // (op 713) 
v_lshrrev_b32 v0, 2, v3 // (op 713) 

s_lshl_b32 s8, 1, s2 // (op 713) 
v_and_b32 v3, 3, v0 // (op 713) 

s_add_i32 s9, s8, s3 // (op 713) 


v_and_b32 v0, 3, v1 // (op 713) 
s_and_b32 s3, s0, s9 // (op 713) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 713) 


s_add_i32 s0, s1, s3 // (op 713) 


v_lshrrev_b32 v0, 6, v1 // (op 713) 
s_ashr_i32 s1, s0, s2 // (op 713) 


v_and_b32 v3, 1, v0 // (op 713) 

s_lshl_b32 s0, s13, 24 // (op 713) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 713) 


s_ashr_i32 s2, s0, 31 // (op 713) 

s_xor_b32 s0, s1, s2 // (op 713) 

s_sub_i32 s1, s0, s2 // (op 713) 


s_mul_hi_i32 s0, s1, s14 // (op 713) 
s_add_i32 s2, s0, s1 // (op 713) 

s_ashr_i32 s0, s2, 31 // (op 713) 
s_cmp_eq_i32 s14, 0 // (op 713) 
s_mov_b32 s3, scc // (op 713) 
s_cmp_eq_i32 s3, 0 // (op 713) 
s_cselect_b32 s8, 0, -1 // (op 713) 

s_and_b32 s3, s15, 31 // (op 713) 
s_lshl_b32 s9, 1, s3 // (op 713) 
s_add_i32 s18, s9, s8 // (op 713) 


s_and_b32 s8, s0, s18 // (op 713) 


s_add_i32 s0, s2, s8 // (op 713) 


s_ashr_i32 s2, s0, s3 // (op 713) 


s_lshl_b32 s0, s15, 24 // (op 713) 
s_ashr_i32 s3, s0, 31 // (op 713) 

s_xor_b32 s0, s2, s3 // (op 713) 

s_sub_i32 s2, s0, s3 // (op 713) 


s_mul_i32 s0, s1, s45 // (op 713) 
s_sub_u32 s3, s16, s0 // (op 713) 

s_mul_i32 s0, s2, s45 // (op 713) 
s_add_u32 s0, s0, s3 // (op 713) 

s_mul_hi_i32 s3, s16, s11 // (op 713) 
s_add_i32 s8, s3, s16 // (op 713) 

s_ashr_i32 s3, s8, 31 // (op 713) 
s_cmp_eq_i32 s11, 0 // (op 713) 
s_mov_b32 s9, scc // (op 713) 
s_cmp_eq_i32 s9, 0 // (op 713) 
s_cselect_b32 s18, 0, -1 // (op 713) 

s_and_b32 s9, s46, 31 // (op 713) 
s_lshl_b32 s19, 1, s9 // (op 713) 
s_add_i32 s20, s19, s18 // (op 713) 


s_and_b32 s18, s3, s20 // (op 713) 


s_add_i32 s3, s8, s18 // (op 713) 


s_ashr_i32 s8, s3, s9 // (op 713) 


s_lshl_b32 s3, s46, 24 // (op 713) 
s_ashr_i32 s9, s3, 31 // (op 713) 

s_xor_b32 s3, s8, s9 // (op 713) 

s_sub_i32 s8, s3, s9 // (op 713) 


s_add_u32 s18, s4, 32 // (op 713) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 713) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 713) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 713) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 713) 

s_mul_i32 s3, s45, s18 // (op 713) 
s_mul_i32 s9, s10, s3 // (op 713) 

s_mul_i32 s3, s8, s9 // (op 713) 

s_sub_u32 s9, s16, s3 // (op 713) 

s_mov_b32 s3, s9 // (op 713) 
s_mul_hi_i32 s20, s3, s44 // (op 713) 
s_add_i32 s21, s20, s3 // (op 713) 


s_ashr_i32 s3, s21, 31 // (op 713) 
s_cmp_eq_i32 s44, 0 // (op 713) 
s_mov_b32 s20, scc // (op 713) 
s_cmp_eq_i32 s20, 0 // (op 713) 
s_cselect_b32 s22, 0, -1 // (op 713) 

s_and_b32 s20, s48, 31 // (op 713) 
s_lshl_b32 s23, 1, s20 // (op 713) 
s_add_i32 s24, s23, s22 // (op 713) 


s_and_b32 s22, s3, s24 // (op 713) 


s_add_i32 s3, s21, s22 // (op 713) 


s_ashr_i32 s21, s3, s20 // (op 713) 


s_lshl_b32 s3, s48, 24 // (op 713) 
s_ashr_i32 s20, s3, 31 // (op 713) 

s_xor_b32 s3, s21, s20 // (op 713) 

s_sub_i32 s21, s3, s20 // (op 713) 


s_mul_i32 s3, s47, s45 // (op 713) 
s_add_u32 s22, s6, 16 // (op 713) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 713) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 713) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 713) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 713) 

s_sub_i32 s20, s22, s3 // (op 713) 

s_mul_i32 s22, s21, s20 // (op 713) 

s_sub_u32 s20, s9, s22 // (op 713) 


s_add_u32 s9, s20, s3 // (op 713) 


s_cmp_eq_i32 s8, 0 // (op 713) 
s_mov_b32 s3, scc // (op 713) 

s_cmp_eq_i32 s3, 0 // (op 713) 
s_cselect_b32 s8, s9, s0 // (op 713) 


s_mul_i32 s0, s2, s18 // (op 713) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 713) 


s_sub_i32 s2, s1, s0 // (op 713) 


s_cmp_eq_i32 s3, 0 // (op 713) 
s_cselect_b32 s0, s21, s2 // (op 713) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 713) 


v_mul_lo_u32 v0, v4, s40 // (op 713) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 713) 


s_mul_i32 s9, s40, s5 // (op 715) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 715) most significant: low of high * low
s_mul_i32 s8, s40, s4 // (op 715) least significant
s_mul_hi_u32 s1, s40, s4 // (op 715) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 715) most significant: sum
s_add_u32 s9, s9, s1 // (op 715) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 715) 

s_mov_b32 s28, s42 // (op 715) reg expression
s_mov_b32 s29, s43 // (op 715) reg expression
s_mov_b32 s30, s0 // (op 715) reg expression
s_mov_b32 s31, 131072 // (op 715) call()





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
v_cvt_f32_f16 v4, v6 // (op 139) 
v_lshrrev_b32 v5, 16, v6 // (op 139) 
v_cvt_f32_f16 v5, v5 // (op 139) 
v_mul_f32 v10, v8, v4 // (op 139) 
v_mul_f32 v11, v8, v5 // (op 139) 
v_cvt_f32_f16 v4, v7 // (op 139) 
v_lshrrev_b32 v5, 16, v7 // (op 139) 
v_cvt_f32_f16 v5, v5 // (op 139) 
v_mul_f32 v12, v8, v4 // (op 139) 
v_mul_f32 v13, v8, v5 // (op 139) 


v_add_f32 v4, v10, v14 // (op 135) 
v_add_f32 v5, v11, v15 // (op 135) 
v_add_f32 v6, v12, v16 // (op 135) 
v_add_f32 v7, v13, v17 // (op 135) 


v_cvt_f16_f32 v10, v4 // (op 136) 
v_cvt_f16_f32 v11, v5 // (op 136) 
v_cvt_f16_f32 v12, v6 // (op 136) 
v_cvt_f16_f32 v13, v7 // (op 136) 

s_mul_hi_i32 s0, s16, s12 // (op 907) 
v_lshrrev_b32 v0, 7, v1 // (op 907) 
s_add_i32 s1, s0, s16 // (op 907) 

v_and_b32 v3, 63, v1 // (op 907) 
s_ashr_i32 s0, s1, 31 // (op 907) 
v_lshrrev_b32 v4, 4, v3 // (op 907) 
s_cmp_eq_i32 s12, 0 // (op 907) 
s_mov_b32 s2, scc // (op 907) 
v_lshlrev_b32 v5, 2, v4 // (op 907) 

s_cmp_eq_i32 s2, 0 // (op 907) 
s_cselect_b32 s3, 0, -1 // (op 907) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 907) 


s_and_b32 s2, s13, 31 // (op 907) 
v_lshrrev_b32 v0, 2, v3 // (op 907) 

s_lshl_b32 s8, 1, s2 // (op 907) 
v_and_b32 v3, 3, v0 // (op 907) 

s_add_i32 s9, s8, s3 // (op 907) 


v_and_b32 v0, 3, v1 // (op 907) 
s_and_b32 s3, s0, s9 // (op 907) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 907) 


s_add_i32 s0, s1, s3 // (op 907) 


v_lshrrev_b32 v0, 6, v1 // (op 907) 
s_ashr_i32 s1, s0, s2 // (op 907) 


v_and_b32 v3, 1, v0 // (op 907) 

s_lshl_b32 s0, s13, 24 // (op 907) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 907) 


s_ashr_i32 s2, s0, 31 // (op 907) 

s_xor_b32 s0, s1, s2 // (op 907) 

s_sub_i32 s1, s0, s2 // (op 907) 


s_mul_hi_i32 s0, s1, s14 // (op 907) 
s_add_i32 s2, s0, s1 // (op 907) 

s_ashr_i32 s0, s2, 31 // (op 907) 
s_cmp_eq_i32 s14, 0 // (op 907) 
s_mov_b32 s3, scc // (op 907) 
s_cmp_eq_i32 s3, 0 // (op 907) 
s_cselect_b32 s8, 0, -1 // (op 907) 

s_and_b32 s3, s15, 31 // (op 907) 
s_lshl_b32 s9, 1, s3 // (op 907) 
s_add_i32 s18, s9, s8 // (op 907) 


s_and_b32 s8, s0, s18 // (op 907) 


s_add_i32 s0, s2, s8 // (op 907) 


s_ashr_i32 s2, s0, s3 // (op 907) 


s_lshl_b32 s0, s15, 24 // (op 907) 
s_ashr_i32 s3, s0, 31 // (op 907) 

s_xor_b32 s0, s2, s3 // (op 907) 

s_sub_i32 s2, s0, s3 // (op 907) 


s_mul_i32 s0, s1, s45 // (op 907) 
s_sub_u32 s3, s16, s0 // (op 907) 

s_mul_i32 s0, s2, s45 // (op 907) 
s_add_u32 s0, s0, s3 // (op 907) 

s_mul_hi_i32 s3, s16, s11 // (op 907) 
s_add_i32 s8, s3, s16 // (op 907) 

s_ashr_i32 s3, s8, 31 // (op 907) 
s_cmp_eq_i32 s11, 0 // (op 907) 
s_mov_b32 s9, scc // (op 907) 
s_cmp_eq_i32 s9, 0 // (op 907) 
s_cselect_b32 s18, 0, -1 // (op 907) 

s_and_b32 s9, s46, 31 // (op 907) 
s_lshl_b32 s19, 1, s9 // (op 907) 
s_add_i32 s20, s19, s18 // (op 907) 


s_and_b32 s18, s3, s20 // (op 907) 


s_add_i32 s3, s8, s18 // (op 907) 


s_ashr_i32 s8, s3, s9 // (op 907) 


s_lshl_b32 s3, s46, 24 // (op 907) 
s_ashr_i32 s9, s3, 31 // (op 907) 

s_xor_b32 s3, s8, s9 // (op 907) 

s_sub_i32 s8, s3, s9 // (op 907) 


s_add_u32 s18, s4, 32 // (op 907) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 907) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 907) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 907) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 907) 

s_mul_i32 s3, s45, s18 // (op 907) 
s_mul_i32 s9, s10, s3 // (op 907) 

s_mul_i32 s3, s8, s9 // (op 907) 

s_sub_u32 s9, s16, s3 // (op 907) 

s_mov_b32 s3, s9 // (op 907) 
s_mul_hi_i32 s20, s3, s44 // (op 907) 
s_add_i32 s21, s20, s3 // (op 907) 


s_ashr_i32 s3, s21, 31 // (op 907) 
s_cmp_eq_i32 s44, 0 // (op 907) 
s_mov_b32 s20, scc // (op 907) 
s_cmp_eq_i32 s20, 0 // (op 907) 
s_cselect_b32 s22, 0, -1 // (op 907) 

s_and_b32 s20, s48, 31 // (op 907) 
s_lshl_b32 s23, 1, s20 // (op 907) 
s_add_i32 s24, s23, s22 // (op 907) 


s_and_b32 s22, s3, s24 // (op 907) 


s_add_i32 s3, s21, s22 // (op 907) 


s_ashr_i32 s21, s3, s20 // (op 907) 


s_lshl_b32 s3, s48, 24 // (op 907) 
s_ashr_i32 s20, s3, 31 // (op 907) 

s_xor_b32 s3, s21, s20 // (op 907) 

s_sub_i32 s21, s3, s20 // (op 907) 


s_mul_i32 s3, s47, s45 // (op 907) 
s_add_u32 s22, s6, 16 // (op 907) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 907) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 907) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 907) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 907) 

s_sub_i32 s20, s22, s3 // (op 907) 

s_mul_i32 s22, s21, s20 // (op 907) 

s_sub_u32 s20, s9, s22 // (op 907) 


s_add_u32 s9, s20, s3 // (op 907) 


s_cmp_eq_i32 s8, 0 // (op 907) 
s_mov_b32 s3, scc // (op 907) 

s_cmp_eq_i32 s3, 0 // (op 907) 
s_cselect_b32 s8, s9, s0 // (op 907) 


s_mul_i32 s0, s2, s18 // (op 907) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 907) 


s_sub_i32 s2, s1, s0 // (op 907) 


s_cmp_eq_i32 s3, 0 // (op 907) 
s_cselect_b32 s0, s21, s2 // (op 907) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 907) 


v_mul_lo_u32 v0, v4, s36 // (op 907) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 907) 


s_mul_i32 s9, s36, s5 // (op 909) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 909) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 909) least significant
s_mul_hi_u32 s1, s36, s4 // (op 909) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 909) most significant: sum
s_add_u32 s9, s9, s1 // (op 909) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 909) 

s_mov_b32 s32, s38 // (op 909) reg expression
s_mov_b32 s33, s39 // (op 909) reg expression
s_mov_b32 s34, s0 // (op 909) reg expression
s_mov_b32 s35, 131072 // (op 909) call()


v_pack_b32_f16 v6, v10, v11 // (op 137) 
v_pack_b32_f16 v7, v12, v13 // (op 137) 


buffer_store_dwordx2 v[6:7], v4, s[32:35], 0 offen // (op 137) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.LRR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_, .LRR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end-RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: B8
    DestDataType: h
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
  - .name: RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 11520
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 70
    .vgpr_count: 50
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

