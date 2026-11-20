// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_,@function
RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 63, v1 // (op 743) 
v_and_b32 v3, 48, v1 // (op 743) 
v_sub_u32 v4, v0, v3 // (op 743) 

v_lshrrev_b32 v3, 4, v0 // (op 743) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 743) 


v_lshrrev_b32 v3, 6, v1 // (op 743) 
v_lshlrev_b32 v4, 4, v3 // (op 743) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 743) 


v_mul_lo_u32 v0, v3, 6 // (op 743) 
v_lshrrev_b32 v4, 3, v0 // (op 743) 

v_lshrrev_b32 v0, 4, v3 // (op 743) 

v_lshlrev_b32 v3, 2, v0 // (op 743) 

v_add_u32 v0, v4, v3 // (op 743) 


v_and_b32 v3, 63, v1 // (op 796) 
v_and_b32 v4, 48, v1 // (op 796) 
v_sub_u32 v5, v3, v4 // (op 796) 

v_lshrrev_b32 v4, 4, v3 // (op 796) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 796) 


v_lshlrev_b32 v4, 4, v3 // (op 796) 
v_mul_lo_u32 v5, v4, 6 // (op 796) 

v_lshrrev_b32 v4, 3, v5 // (op 796) 

v_and_b32 v5, 268435455, v3 // (op 796) 

v_lshlrev_b32 v3, 2, v5 // (op 796) 

v_add_u32 v5, v4, v3 // (op 796) 


v_lshrrev_b32 v3, 6, v1 // (op 849) 
v_and_b32 v4, 15, v1 // (op 849) 
v_lshl_add_u32 v6, v3, 4, v4 // (op 849) 


v_and_b32 v3, 63, v1 // (op 849) 
v_lshrrev_b32 v4, 4, v3 // (op 849) 

v_lshl_add_u32 v3, v4, 5, v6 // (op 849) 


v_and_b32 v4, 63, v1 // (op 907) 
v_and_b32 v6, 48, v1 // (op 907) 
v_sub_u32 v7, v4, v6 // (op 907) 

v_lshrrev_b32 v6, 4, v4 // (op 907) 

v_lshl_add_u32 v4, v6, 5, v7 // (op 907) 


v_lshrrev_b32 v6, 6, v1 // (op 907) 
v_lshlrev_b32 v7, 4, v6 // (op 907) 

v_lshl_add_u32 v6, v4, 5, v7 // (op 907) 


v_mul_lo_u32 v4, v6, 6 // (op 907) 
v_lshrrev_b32 v7, 3, v4 // (op 907) 

v_lshrrev_b32 v4, 4, v6 // (op 907) 

v_lshlrev_b32 v6, 2, v4 // (op 907) 

v_add_u32 v4, v7, v6 // (op 907) 


v_and_b32 v6, 63, v1 // (op 931) 
v_and_b32 v7, 48, v1 // (op 931) 
v_sub_u32 v8, v6, v7 // (op 931) 

v_lshrrev_b32 v7, 4, v6 // (op 931) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 931) 


v_lshlrev_b32 v7, 4, v6 // (op 931) 
v_mul_lo_u32 v8, v7, 6 // (op 931) 

v_lshrrev_b32 v7, 3, v8 // (op 931) 

v_and_b32 v8, 268435455, v6 // (op 931) 

v_lshlrev_b32 v6, 2, v8 // (op 931) 

v_add_u32 v8, v7, v6 // (op 931) 


v_lshrrev_b32 v6, 6, v1 // (op 955) 
v_and_b32 v7, 15, v1 // (op 955) 
v_lshl_add_u32 v9, v6, 4, v7 // (op 955) 


v_and_b32 v6, 63, v1 // (op 955) 
v_lshrrev_b32 v7, 4, v6 // (op 955) 

v_lshl_add_u32 v6, v7, 5, v9 // (op 955) 


v_mov_b32 v7, s6 // (op 2) Move value
v_mov_b32 v9, s7 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
s_mul_hi_i32 s0, s16, s9 // (op 453) 
s_add_i32 s1, s0, s16 // (op 453) 

s_ashr_i32 s0, s1, 31 // (op 453) 
s_cmp_eq_i32 s9, 0 // (op 453) 
s_mov_b32 s6, scc // (op 453) 
s_cmp_eq_i32 s6, 0 // (op 453) 
s_cselect_b32 s7, 0, -1 // (op 453) 

s_and_b32 s6, s10, 31 // (op 453) 
s_lshl_b32 s18, 1, s6 // (op 453) 
s_add_i32 s19, s18, s7 // (op 453) 


s_and_b32 s7, s0, s19 // (op 453) 


s_add_i32 s0, s1, s7 // (op 453) 


s_ashr_i32 s1, s0, s6 // (op 453) 


s_lshl_b32 s0, s10, 24 // (op 453) 
s_ashr_i32 s6, s0, 31 // (op 453) 

s_xor_b32 s0, s1, s6 // (op 453) 

s_sub_i32 s1, s0, s6 // (op 453) 


s_mul_hi_i32 s0, s1, s11 // (op 453) 
s_add_i32 s6, s0, s1 // (op 453) 

s_ashr_i32 s0, s6, 31 // (op 453) 
s_cmp_eq_i32 s11, 0 // (op 453) 
s_mov_b32 s7, scc // (op 453) 
s_cmp_eq_i32 s7, 0 // (op 453) 
s_cselect_b32 s18, 0, -1 // (op 453) 

s_and_b32 s7, s12, 31 // (op 453) 
s_lshl_b32 s19, 1, s7 // (op 453) 
s_add_i32 s51, s19, s18 // (op 453) 


s_and_b32 s18, s0, s51 // (op 453) 


s_add_i32 s0, s6, s18 // (op 453) 


s_ashr_i32 s6, s0, s7 // (op 453) 


s_lshl_b32 s0, s12, 24 // (op 453) 
s_ashr_i32 s7, s0, 31 // (op 453) 

s_xor_b32 s0, s6, s7 // (op 453) 

s_sub_i32 s6, s0, s7 // (op 453) 


s_mul_i32 s0, s1, s14 // (op 453) 

s_sub_u32 s1, s16, s0 // (op 453) 

s_mul_i32 s0, s6, s14 // (op 453) 
s_add_u32 s0, s0, s1 // (op 453) 


s_mul_hi_i32 s1, s16, s8 // (op 453) 
s_add_i32 s6, s1, s16 // (op 453) 

s_ashr_i32 s1, s6, 31 // (op 453) 
s_cmp_eq_i32 s8, 0 // (op 453) 
s_mov_b32 s7, scc // (op 453) 
s_cmp_eq_i32 s7, 0 // (op 453) 
s_cselect_b32 s18, 0, -1 // (op 453) 

s_and_b32 s7, s15, 31 // (op 453) 
s_lshl_b32 s19, 1, s7 // (op 453) 
s_add_i32 s51, s19, s18 // (op 453) 


s_and_b32 s18, s1, s51 // (op 453) 


s_add_i32 s1, s6, s18 // (op 453) 


s_ashr_i32 s6, s1, s7 // (op 453) 


s_lshl_b32 s1, s15, 24 // (op 453) 
s_ashr_i32 s7, s1, 31 // (op 453) 

s_xor_b32 s1, s6, s7 // (op 453) 

s_sub_i32 s6, s1, s7 // (op 453) 


s_add_u32 s18, s4, 16 // (op 453) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 453) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 453) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 453) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 453) 

s_mul_i32 s1, s14, s18 // (op 453) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 453) 

s_mul_i32 s1, s6, s7 // (op 453) 

s_sub_u32 s7, s16, s1 // (op 453) 

s_mov_b32 s1, s7 // (op 453) 
s_mul_hi_i32 s18, s1, s13 // (op 453) 
s_add_i32 s19, s18, s1 // (op 453) 


s_ashr_i32 s1, s19, 31 // (op 453) 
s_cmp_eq_i32 s13, 0 // (op 453) 
s_mov_b32 s18, scc // (op 453) 
s_cmp_eq_i32 s18, 0 // (op 453) 
s_cselect_b32 s51, 0, -1 // (op 453) 

s_and_b32 s18, s49, 31 // (op 453) 
s_lshl_b32 s52, 1, s18 // (op 453) 
s_add_i32 s53, s52, s51 // (op 453) 


s_and_b32 s51, s1, s53 // (op 453) 


s_add_i32 s1, s19, s51 // (op 453) 


s_ashr_i32 s19, s1, s18 // (op 453) 


s_lshl_b32 s1, s49, 24 // (op 453) 
s_ashr_i32 s18, s1, 31 // (op 453) 

s_xor_b32 s1, s19, s18 // (op 453) 

s_sub_i32 s19, s1, s18 // (op 453) 


s_mul_i32 s1, s48, s14 // (op 453) 
s_add_u32 s52, s20, 32 // (op 453) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 453) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 453) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 453) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 5 // (op 453) 

s_sub_i32 s18, s52, s1 // (op 453) 

s_mul_i32 s51, s19, s18 // (op 453) 


s_sub_u32 s18, s7, s51 // (op 453) 


s_add_u32 s7, s18, s1 // (op 453) 


s_cmp_eq_i32 s6, 0 // (op 453) 
s_mov_b32 s1, scc // (op 453) 

s_cmp_eq_i32 s1, 0 // (op 453) 
s_cselect_b32 s6, s7, s0 // (op 453) 



s_lshl_b32 s7, s6, 5 // (op 453) 

s_mov_b32 s1, 0 // (op 453) convert
s_mov_b32 s0, s7 // (op 453) convert

v_mul_lo_u32 v11, v1, s3 // (op 453) most significant: low of low * high
v_mul_hi_u32 v12, v1, s2 // (op 453) most significant: high of low * low
v_mul_lo_u32 v10, v1, s2 // (op 453) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 453) most significant: sum

v_mov_b32 v12, s0 // (op 453) 
v_mov_b32 v13, s1 // (op 453) 
v_add_co_u32 v14, s[6:7], v12, v10 // (op 453) least significant half
v_addc_co_u32 v15, s[6:7], v13, v11, s[6:7] // (op 453) most significant half





v_mul_lo_u32 v11, v15, 6 // (op 453) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 453) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 453) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 453) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 453) 

v_mov_b32 v10, v12 // (op 453) convert

s_mul_i32 s1, s25, 6 // (op 455) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 455) least significant
s_mul_hi_u32 s6, s24, 6 // (op 455) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 455) most significant: sum

s_mul_i32 s7, s2, s1 // (op 455) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 455) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 455) least significant
s_mul_hi_u32 s19, s2, s0 // (op 455) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 455) most significant: sum
s_add_u32 s7, s7, s19 // (op 455) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 455) 

s_mov_b32 s52, s22 // (op 455) reg expression
s_mov_b32 s53, s23 // (op 455) reg expression
s_mov_b32 s54, s0 // (op 455) reg expression
s_mov_b32 s55, 131072 // (op 455) call()

s_mul_hi_i32 s51, s16, s9 // (op 472) 
s_add_i32 s0, s51, s16 // (op 472) 

s_ashr_i32 s1, s0, 31 // (op 472) 
s_cmp_eq_i32 s9, 0 // (op 472) 
s_mov_b32 s51, scc // (op 472) 
s_cmp_eq_i32 s51, 0 // (op 472) 
s_cselect_b32 s6, 0, -1 // (op 472) 

s_and_b32 s7, s10, 31 // (op 472) 
s_lshl_b32 s51, 1, s7 // (op 472) 
s_add_i32 s18, s51, s6 // (op 472) 


s_and_b32 s6, s1, s18 // (op 472) 


s_add_i32 s1, s0, s6 // (op 472) 


s_ashr_i32 s0, s1, s7 // (op 472) 


s_lshl_b32 s1, s10, 24 // (op 472) 
s_ashr_i32 s51, s1, 31 // (op 472) 

s_xor_b32 s1, s0, s51 // (op 472) 

s_sub_i32 s0, s1, s51 // (op 472) 


s_mul_hi_i32 s1, s0, s11 // (op 472) 
s_add_i32 s51, s1, s0 // (op 472) 

s_ashr_i32 s1, s51, 31 // (op 472) 
s_cmp_eq_i32 s11, 0 // (op 472) 
s_mov_b32 s6, scc // (op 472) 
s_cmp_eq_i32 s6, 0 // (op 472) 
s_cselect_b32 s7, 0, -1 // (op 472) 

s_and_b32 s6, s12, 31 // (op 472) 
s_lshl_b32 s18, 1, s6 // (op 472) 
s_add_i32 s19, s18, s7 // (op 472) 


s_and_b32 s7, s1, s19 // (op 472) 


s_add_i32 s1, s51, s7 // (op 472) 


s_ashr_i32 s7, s1, s6 // (op 472) 


s_lshl_b32 s1, s12, 24 // (op 472) 
s_ashr_i32 s6, s1, 31 // (op 472) 

s_xor_b32 s1, s7, s6 // (op 472) 

s_sub_i32 s7, s1, s6 // (op 472) 


s_mul_i32 s1, s0, s14 // (op 472) 

s_sub_u32 s0, s16, s1 // (op 472) 

s_mul_i32 s1, s7, s14 // (op 472) 
s_add_u32 s1, s1, s0 // (op 472) 


s_mul_hi_i32 s0, s16, s8 // (op 472) 
s_add_i32 s51, s0, s16 // (op 472) 

s_ashr_i32 s0, s51, 31 // (op 472) 
s_cmp_eq_i32 s8, 0 // (op 472) 
s_mov_b32 s6, scc // (op 472) 
s_cmp_eq_i32 s6, 0 // (op 472) 
s_cselect_b32 s7, 0, -1 // (op 472) 

s_and_b32 s6, s15, 31 // (op 472) 
s_lshl_b32 s18, 1, s6 // (op 472) 
s_add_i32 s19, s18, s7 // (op 472) 


s_and_b32 s7, s0, s19 // (op 472) 


s_add_i32 s0, s51, s7 // (op 472) 


s_ashr_i32 s7, s0, s6 // (op 472) 


s_lshl_b32 s0, s15, 24 // (op 472) 
s_ashr_i32 s6, s0, 31 // (op 472) 

s_xor_b32 s0, s7, s6 // (op 472) 

s_sub_i32 s7, s0, s6 // (op 472) 


s_add_u32 s18, s4, 16 // (op 472) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 472) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 472) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 472) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 472) 

s_mul_i32 s0, s14, s18 // (op 472) 

s_mul_i32 s6, s50, s0 // (op 472) 

s_mul_i32 s0, s7, s6 // (op 472) 

s_sub_u32 s6, s16, s0 // (op 472) 

s_mov_b32 s0, s6 // (op 472) 
s_mul_hi_i32 s51, s0, s13 // (op 472) 
s_add_i32 s18, s51, s0 // (op 472) 


s_ashr_i32 s0, s18, 31 // (op 472) 
s_cmp_eq_i32 s13, 0 // (op 472) 
s_mov_b32 s19, scc // (op 472) 
s_cmp_eq_i32 s19, 0 // (op 472) 
s_cselect_b32 s51, 0, -1 // (op 472) 

s_and_b32 s19, s49, 31 // (op 472) 
s_lshl_b32 s56, 1, s19 // (op 472) 
s_add_i32 s57, s56, s51 // (op 472) 


s_and_b32 s51, s0, s57 // (op 472) 


s_add_i32 s0, s18, s51 // (op 472) 


s_ashr_i32 s18, s0, s19 // (op 472) 


s_lshl_b32 s0, s49, 24 // (op 472) 
s_ashr_i32 s19, s0, 31 // (op 472) 

s_xor_b32 s0, s18, s19 // (op 472) 

s_sub_i32 s18, s0, s19 // (op 472) 


s_mul_i32 s0, s48, s14 // (op 472) 
s_add_u32 s56, s20, 32 // (op 472) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 472) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 472) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 472) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 472) 

s_sub_i32 s19, s56, s0 // (op 472) 

s_mul_i32 s51, s18, s19 // (op 472) 


s_sub_u32 s18, s6, s51 // (op 472) 


s_add_u32 s6, s18, s0 // (op 472) 


s_cmp_eq_i32 s7, 0 // (op 472) 
s_mov_b32 s0, scc // (op 472) 

s_cmp_eq_i32 s0, 0 // (op 472) 
s_cselect_b32 s7, s6, s1 // (op 472) 



s_lshl_b32 s6, s7, 5 // (op 472) 

s_mov_b32 s1, 0 // (op 472) convert
s_mov_b32 s0, s6 // (op 472) convert

v_mul_lo_u32 v13, v1, s3 // (op 472) most significant: low of low * high
v_mul_hi_u32 v11, v1, s2 // (op 472) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 472) least significant: low of low * low
v_add_u32 v13, v13, v11 // (op 472) most significant: sum

v_mov_b32 v11, s0 // (op 472) 
v_mov_b32 v14, s1 // (op 472) 
v_add_co_u32 v16, s[6:7], v11, v12 // (op 472) least significant half
v_addc_co_u32 v17, s[6:7], v14, v13, s[6:7] // (op 472) most significant half





v_mul_lo_u32 v15, v17, 6 // (op 472) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 472) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 472) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 472) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 472) 

v_mov_b32 v11, v12 // (op 472) convert

s_mul_i32 s1, s25, 6 // (op 474) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 474) least significant
s_mul_hi_u32 s51, s24, 6 // (op 474) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 474) most significant: sum

s_mul_i32 s7, s2, s1 // (op 474) most significant: low of low * high
s_mul_i32 s51, s3, s0 // (op 474) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 474) least significant
s_mul_hi_u32 s18, s2, s0 // (op 474) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 474) most significant: sum
s_add_u32 s7, s7, s18 // (op 474) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 474) 

s_mov_b32 s52, s22 // (op 474) reg expression
s_mov_b32 s53, s23 // (op 474) reg expression
s_mov_b32 s54, s0 // (op 474) reg expression
s_mov_b32 s55, 131072 // (op 474) call()

s_mul_hi_i32 s51, s16, s9 // (op 517) 
s_add_i32 s0, s51, s16 // (op 517) 

s_ashr_i32 s1, s0, 31 // (op 517) 
s_cmp_eq_i32 s9, 0 // (op 517) 
s_mov_b32 s51, scc // (op 517) 
s_cmp_eq_i32 s51, 0 // (op 517) 
s_cselect_b32 s6, 0, -1 // (op 517) 

s_and_b32 s7, s10, 31 // (op 517) 
s_lshl_b32 s51, 1, s7 // (op 517) 
s_add_i32 s18, s51, s6 // (op 517) 


s_and_b32 s6, s1, s18 // (op 517) 


s_add_i32 s1, s0, s6 // (op 517) 


s_ashr_i32 s0, s1, s7 // (op 517) 


s_lshl_b32 s1, s10, 24 // (op 517) 
s_ashr_i32 s51, s1, 31 // (op 517) 

s_xor_b32 s1, s0, s51 // (op 517) 

s_sub_i32 s0, s1, s51 // (op 517) 


s_mul_hi_i32 s1, s0, s11 // (op 517) 
s_add_i32 s51, s1, s0 // (op 517) 

s_ashr_i32 s1, s51, 31 // (op 517) 
s_cmp_eq_i32 s11, 0 // (op 517) 
s_mov_b32 s6, scc // (op 517) 
s_cmp_eq_i32 s6, 0 // (op 517) 
s_cselect_b32 s7, 0, -1 // (op 517) 

s_and_b32 s6, s12, 31 // (op 517) 
s_lshl_b32 s18, 1, s6 // (op 517) 
s_add_i32 s19, s18, s7 // (op 517) 


s_and_b32 s7, s1, s19 // (op 517) 


s_add_i32 s1, s51, s7 // (op 517) 


s_ashr_i32 s7, s1, s6 // (op 517) 


s_lshl_b32 s1, s12, 24 // (op 517) 
s_ashr_i32 s6, s1, 31 // (op 517) 

s_xor_b32 s1, s7, s6 // (op 517) 

s_sub_i32 s7, s1, s6 // (op 517) 


s_add_u32 s18, s4, 16 // (op 517) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 517) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 517) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 517) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 517) 

s_mul_i32 s1, s7, s18 // (op 517) 

s_sub_i32 s51, s0, s1 // (op 517) 


s_mul_hi_i32 s0, s16, s8 // (op 517) 
s_add_i32 s1, s0, s16 // (op 517) 

s_ashr_i32 s0, s1, 31 // (op 517) 
s_cmp_eq_i32 s8, 0 // (op 517) 
s_mov_b32 s6, scc // (op 517) 
s_cmp_eq_i32 s6, 0 // (op 517) 
s_cselect_b32 s7, 0, -1 // (op 517) 

s_and_b32 s6, s15, 31 // (op 517) 
s_lshl_b32 s56, 1, s6 // (op 517) 
s_add_i32 s57, s56, s7 // (op 517) 


s_and_b32 s7, s0, s57 // (op 517) 


s_add_i32 s0, s1, s7 // (op 517) 


s_ashr_i32 s1, s0, s6 // (op 517) 


s_lshl_b32 s0, s15, 24 // (op 517) 
s_ashr_i32 s6, s0, 31 // (op 517) 

s_xor_b32 s0, s1, s6 // (op 517) 

s_sub_i32 s1, s0, s6 // (op 517) 


s_mul_i32 s0, s14, s18 // (op 517) 

s_mul_i32 s6, s50, s0 // (op 517) 

s_mul_i32 s0, s1, s6 // (op 517) 

s_sub_u32 s6, s16, s0 // (op 517) 

s_mul_hi_i32 s0, s6, s13 // (op 517) 
s_add_i32 s7, s0, s6 // (op 517) 


s_ashr_i32 s0, s7, 31 // (op 517) 
s_cmp_eq_i32 s13, 0 // (op 517) 
s_mov_b32 s6, scc // (op 517) 
s_cmp_eq_i32 s6, 0 // (op 517) 
s_cselect_b32 s18, 0, -1 // (op 517) 

s_and_b32 s6, s49, 31 // (op 517) 
s_lshl_b32 s19, 1, s6 // (op 517) 
s_add_i32 s56, s19, s18 // (op 517) 


s_and_b32 s18, s0, s56 // (op 517) 


s_add_i32 s0, s7, s18 // (op 517) 


s_ashr_i32 s7, s0, s6 // (op 517) 


s_lshl_b32 s0, s49, 24 // (op 517) 
s_ashr_i32 s6, s0, 31 // (op 517) 

s_xor_b32 s0, s7, s6 // (op 517) 

s_sub_i32 s7, s0, s6 // (op 517) 


s_cmp_eq_i32 s1, 0 // (op 517) 
s_mov_b32 s0, scc // (op 517) 

s_cmp_eq_i32 s0, 0 // (op 517) 
s_cselect_b32 s1, s7, s51 // (op 517) 



s_lshl_b32 s0, s1, 4 // (op 517) 

s_mov_b32 s7, 0 // (op 517) convert
s_mov_b32 s6, s0 // (op 517) convert

v_mul_lo_u32 v13, v1, s27 // (op 517) most significant: low of low * high
v_mul_hi_u32 v14, v1, s26 // (op 517) most significant: high of low * low
v_mul_lo_u32 v12, v1, s26 // (op 517) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 517) most significant: sum

v_mov_b32 v14, s6 // (op 517) 
v_mov_b32 v15, s7 // (op 517) 
v_add_co_u32 v12, s[0:1], v12, v14 // (op 517) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 517) most significant half




v_mul_lo_u32 v15, v13, 6 // (op 517) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 517) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 517) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 517) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 517) 

v_mov_b32 v14, v12 // (op 517) convert

s_mul_i32 s1, s25, 6 // (op 519) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 519) least significant
s_mul_hi_u32 s51, s24, 6 // (op 519) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 519) most significant: sum

s_mul_i32 s7, s26, s1 // (op 519) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 519) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 519) least significant
s_mul_hi_u32 s18, s26, s0 // (op 519) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 519) most significant: sum
s_add_u32 s7, s7, s18 // (op 519) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 519) 

s_mov_b32 s56, s28 // (op 519) reg expression
s_mov_b32 s57, s29 // (op 519) reg expression
s_mov_b32 s58, s0 // (op 519) reg expression
s_mov_b32 s59, 131072 // (op 519) call()

s_mul_hi_i32 s51, s16, s9 // (op 536) 
s_add_i32 s0, s51, s16 // (op 536) 

s_ashr_i32 s1, s0, 31 // (op 536) 
s_cmp_eq_i32 s9, 0 // (op 536) 
s_mov_b32 s51, scc // (op 536) 
s_cmp_eq_i32 s51, 0 // (op 536) 
s_cselect_b32 s6, 0, -1 // (op 536) 

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
s_sub_u32 s60, s18, 1 // (op 536) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 536) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 536) 

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
s_lshl_b32 s60, 1, s6 // (op 536) 
s_add_i32 s61, s60, s7 // (op 536) 


s_and_b32 s7, s0, s61 // (op 536) 


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
s_add_i32 s60, s19, s18 // (op 536) 


s_and_b32 s18, s0, s60 // (op 536) 


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



s_lshl_b32 s0, s1, 4 // (op 536) 

s_mov_b32 s7, 0 // (op 536) convert
s_mov_b32 s6, s0 // (op 536) convert

v_mul_lo_u32 v13, v1, s27 // (op 536) most significant: low of low * high
v_mul_hi_u32 v15, v1, s26 // (op 536) most significant: high of low * low
v_mul_lo_u32 v12, v1, s26 // (op 536) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 536) most significant: sum

v_mov_b32 v15, s6 // (op 536) 
v_mov_b32 v16, s7 // (op 536) 
v_add_co_u32 v12, s[0:1], v12, v15 // (op 536) least significant half
v_addc_co_u32 v13, s[0:1], v13, v16, s[0:1] // (op 536) most significant half




v_mul_lo_u32 v17, v13, 6 // (op 536) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 536) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 536) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 536) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 536) 

v_mov_b32 v15, v12 // (op 536) convert

s_mul_i32 s1, s25, 6 // (op 538) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 538) least significant
s_mul_hi_u32 s51, s24, 6 // (op 538) most significant: high of low * low
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

s_mul_hi_i32 s51, s16, s9 // (op 581) 
v_and_b32 v12, -32, v1 // (op 581) 
s_add_i32 s0, s51, s16 // (op 581) 

v_sub_u32 v13, v1, v12 // (op 581) 

s_ashr_i32 s1, s0, 31 // (op 581) 
v_lshrrev_b32 v12, 5, v1 // (op 581) 
s_cmp_eq_i32 s9, 0 // (op 581) 
s_mov_b32 s51, scc // (op 581) 
s_cmp_eq_i32 s51, 0 // (op 581) 
s_cselect_b32 s6, 0, -1 // (op 581) 

s_and_b32 s7, s10, 31 // (op 581) 
s_lshl_b32 s51, 1, s7 // (op 581) 
s_add_i32 s18, s51, s6 // (op 581) 


s_and_b32 s6, s1, s18 // (op 581) 


s_add_i32 s1, s0, s6 // (op 581) 


s_ashr_i32 s0, s1, s7 // (op 581) 


s_lshl_b32 s1, s10, 24 // (op 581) 
s_ashr_i32 s51, s1, 31 // (op 581) 

s_xor_b32 s1, s0, s51 // (op 581) 

s_sub_i32 s0, s1, s51 // (op 581) 


s_mul_hi_i32 s1, s0, s11 // (op 581) 
s_add_i32 s51, s1, s0 // (op 581) 

s_ashr_i32 s1, s51, 31 // (op 581) 
s_cmp_eq_i32 s11, 0 // (op 581) 
s_mov_b32 s6, scc // (op 581) 
s_cmp_eq_i32 s6, 0 // (op 581) 
s_cselect_b32 s7, 0, -1 // (op 581) 

s_and_b32 s6, s12, 31 // (op 581) 
s_lshl_b32 s18, 1, s6 // (op 581) 
s_add_i32 s19, s18, s7 // (op 581) 


s_and_b32 s7, s1, s19 // (op 581) 


s_add_i32 s1, s51, s7 // (op 581) 


s_ashr_i32 s7, s1, s6 // (op 581) 


s_lshl_b32 s1, s12, 24 // (op 581) 
s_ashr_i32 s6, s1, 31 // (op 581) 

s_xor_b32 s1, s7, s6 // (op 581) 

s_sub_i32 s7, s1, s6 // (op 581) 


s_mul_i32 s1, s0, s14 // (op 581) 

s_sub_u32 s0, s16, s1 // (op 581) 

s_mul_i32 s1, s7, s14 // (op 581) 
s_add_u32 s1, s1, s0 // (op 581) 


s_mul_hi_i32 s0, s16, s8 // (op 581) 
s_add_i32 s51, s0, s16 // (op 581) 

s_ashr_i32 s0, s51, 31 // (op 581) 
s_cmp_eq_i32 s8, 0 // (op 581) 
s_mov_b32 s6, scc // (op 581) 
s_cmp_eq_i32 s6, 0 // (op 581) 
s_cselect_b32 s7, 0, -1 // (op 581) 

s_and_b32 s6, s15, 31 // (op 581) 
s_lshl_b32 s18, 1, s6 // (op 581) 
s_add_i32 s19, s18, s7 // (op 581) 


s_and_b32 s7, s0, s19 // (op 581) 


s_add_i32 s0, s51, s7 // (op 581) 


s_ashr_i32 s7, s0, s6 // (op 581) 


s_lshl_b32 s0, s15, 24 // (op 581) 
s_ashr_i32 s6, s0, 31 // (op 581) 

s_xor_b32 s0, s7, s6 // (op 581) 

s_sub_i32 s7, s0, s6 // (op 581) 


s_add_u32 s18, s4, 16 // (op 581) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 581) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 581) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 581) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 581) 

s_mul_i32 s0, s14, s18 // (op 581) 

s_mul_i32 s6, s50, s0 // (op 581) 

s_mul_i32 s0, s7, s6 // (op 581) 

s_sub_u32 s6, s16, s0 // (op 581) 

s_mov_b32 s0, s6 // (op 581) 
s_mul_hi_i32 s51, s0, s13 // (op 581) 
s_add_i32 s18, s51, s0 // (op 581) 


s_ashr_i32 s0, s18, 31 // (op 581) 
s_cmp_eq_i32 s13, 0 // (op 581) 
s_mov_b32 s19, scc // (op 581) 
s_cmp_eq_i32 s19, 0 // (op 581) 
s_cselect_b32 s51, 0, -1 // (op 581) 

s_and_b32 s19, s49, 31 // (op 581) 
s_lshl_b32 s60, 1, s19 // (op 581) 
s_add_i32 s61, s60, s51 // (op 581) 


s_and_b32 s51, s0, s61 // (op 581) 


s_add_i32 s0, s18, s51 // (op 581) 


s_ashr_i32 s18, s0, s19 // (op 581) 


s_lshl_b32 s0, s49, 24 // (op 581) 
s_ashr_i32 s19, s0, 31 // (op 581) 

s_xor_b32 s0, s18, s19 // (op 581) 

s_sub_i32 s18, s0, s19 // (op 581) 


s_mul_i32 s0, s48, s14 // (op 581) 
s_add_u32 s60, s20, 32 // (op 581) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 581) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 581) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 581) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 5 // (op 581) 

s_sub_i32 s19, s60, s0 // (op 581) 

s_mul_i32 s51, s18, s19 // (op 581) 


s_sub_u32 s18, s6, s51 // (op 581) 


s_add_u32 s6, s18, s0 // (op 581) 


s_cmp_eq_i32 s7, 0 // (op 581) 
s_mov_b32 s0, scc // (op 581) 

s_cmp_eq_i32 s0, 0 // (op 581) 
s_cselect_b32 s7, s6, s1 // (op 581) 



v_lshl_add_u32 v16, s7, 5, v13 // (op 581) 


v_mul_lo_u32 v13, v12, s30 // (op 581) 

v_add_u32 v12, v16, v13 // (op 581) 


s_mul_i32 s1, s30, s35 // (op 583) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 583) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 583) least significant
s_mul_hi_u32 s6, s30, s34 // (op 583) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 583) most significant: sum
s_add_u32 s1, s1, s6 // (op 583) most significant: sum


s_mov_b32 s60, s32 // (op 583) reg expression
s_mov_b32 s61, s33 // (op 583) reg expression
s_mov_b32 s62, s0 // (op 583) reg expression
s_mov_b32 s63, 131072 // (op 583) call()

s_mul_hi_i32 s51, s16, s9 // (op 600) 
v_and_b32 v13, -32, v1 // (op 600) 
s_add_i32 s0, s51, s16 // (op 600) 

v_sub_u32 v16, v1, v13 // (op 600) 

s_ashr_i32 s1, s0, 31 // (op 600) 
v_lshrrev_b32 v13, 5, v1 // (op 600) 
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
s_sub_u32 s64, s18, 1 // (op 600) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 600) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 600) 

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
s_lshl_b32 s64, 1, s19 // (op 600) 
s_add_i32 s65, s64, s51 // (op 600) 


s_and_b32 s51, s0, s65 // (op 600) 


s_add_i32 s0, s18, s51 // (op 600) 


s_ashr_i32 s18, s0, s19 // (op 600) 


s_lshl_b32 s0, s49, 24 // (op 600) 
s_ashr_i32 s19, s0, 31 // (op 600) 

s_xor_b32 s0, s18, s19 // (op 600) 

s_sub_i32 s18, s0, s19 // (op 600) 


s_mul_i32 s0, s48, s14 // (op 600) 
s_add_u32 s64, s20, 32 // (op 600) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 600) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 600) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 600) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 5 // (op 600) 

s_sub_i32 s19, s64, s0 // (op 600) 

s_mul_i32 s51, s18, s19 // (op 600) 


s_sub_u32 s18, s6, s51 // (op 600) 


s_add_u32 s6, s18, s0 // (op 600) 


s_cmp_eq_i32 s7, 0 // (op 600) 
s_mov_b32 s0, scc // (op 600) 

s_cmp_eq_i32 s0, 0 // (op 600) 
s_cselect_b32 s7, s6, s1 // (op 600) 



v_lshl_add_u32 v17, s7, 5, v16 // (op 600) 


v_mul_lo_u32 v16, v13, s30 // (op 600) 

v_add_u32 v13, v17, v16 // (op 600) 


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

s_mul_hi_i32 s51, s16, s9 // (op 645) 
v_and_b32 v16, 15, v1 // (op 645) 
s_add_i32 s0, s51, s16 // (op 645) 

v_and_b32 v17, 63, v1 // (op 645) 
s_ashr_i32 s1, s0, 31 // (op 645) 
v_lshrrev_b32 v18, 4, v17 // (op 645) 

s_cmp_eq_i32 s9, 0 // (op 645) 
s_mov_b32 s51, scc // (op 645) 
s_cmp_eq_i32 s51, 0 // (op 645) 
s_cselect_b32 s6, 0, -1 // (op 645) 

s_and_b32 s7, s10, 31 // (op 645) 
s_lshl_b32 s51, 1, s7 // (op 645) 
s_add_i32 s18, s51, s6 // (op 645) 


s_and_b32 s6, s1, s18 // (op 645) 


s_add_i32 s1, s0, s6 // (op 645) 


s_ashr_i32 s0, s1, s7 // (op 645) 


s_lshl_b32 s1, s10, 24 // (op 645) 
s_ashr_i32 s51, s1, 31 // (op 645) 

s_xor_b32 s1, s0, s51 // (op 645) 

s_sub_i32 s0, s1, s51 // (op 645) 


s_mul_hi_i32 s1, s0, s11 // (op 645) 
s_add_i32 s51, s1, s0 // (op 645) 

s_ashr_i32 s1, s51, 31 // (op 645) 
s_cmp_eq_i32 s11, 0 // (op 645) 
s_mov_b32 s6, scc // (op 645) 
s_cmp_eq_i32 s6, 0 // (op 645) 
s_cselect_b32 s7, 0, -1 // (op 645) 

s_and_b32 s6, s12, 31 // (op 645) 
s_lshl_b32 s18, 1, s6 // (op 645) 
s_add_i32 s19, s18, s7 // (op 645) 


s_and_b32 s7, s1, s19 // (op 645) 


s_add_i32 s1, s51, s7 // (op 645) 


s_ashr_i32 s7, s1, s6 // (op 645) 


s_lshl_b32 s1, s12, 24 // (op 645) 
s_ashr_i32 s6, s1, 31 // (op 645) 

s_xor_b32 s1, s7, s6 // (op 645) 

s_sub_i32 s7, s1, s6 // (op 645) 


s_add_u32 s18, s4, 16 // (op 645) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 645) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 645) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 645) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 645) 

s_mul_i32 s1, s7, s18 // (op 645) 

s_sub_i32 s51, s0, s1 // (op 645) 


s_mul_hi_i32 s0, s16, s8 // (op 645) 
s_add_i32 s1, s0, s16 // (op 645) 

s_ashr_i32 s0, s1, 31 // (op 645) 
s_cmp_eq_i32 s8, 0 // (op 645) 
s_mov_b32 s6, scc // (op 645) 
s_cmp_eq_i32 s6, 0 // (op 645) 
s_cselect_b32 s7, 0, -1 // (op 645) 

s_and_b32 s6, s15, 31 // (op 645) 
s_lshl_b32 s64, 1, s6 // (op 645) 
s_add_i32 s65, s64, s7 // (op 645) 


s_and_b32 s7, s0, s65 // (op 645) 


s_add_i32 s0, s1, s7 // (op 645) 


s_ashr_i32 s1, s0, s6 // (op 645) 


s_lshl_b32 s0, s15, 24 // (op 645) 
s_ashr_i32 s6, s0, 31 // (op 645) 

s_xor_b32 s0, s1, s6 // (op 645) 

s_sub_i32 s1, s0, s6 // (op 645) 


s_mul_i32 s0, s14, s18 // (op 645) 

s_mul_i32 s6, s50, s0 // (op 645) 

s_mul_i32 s0, s1, s6 // (op 645) 

s_sub_u32 s6, s16, s0 // (op 645) 

s_mul_hi_i32 s0, s6, s13 // (op 645) 
s_add_i32 s7, s0, s6 // (op 645) 


s_ashr_i32 s0, s7, 31 // (op 645) 
s_cmp_eq_i32 s13, 0 // (op 645) 
s_mov_b32 s6, scc // (op 645) 
s_cmp_eq_i32 s6, 0 // (op 645) 
s_cselect_b32 s18, 0, -1 // (op 645) 

s_and_b32 s6, s49, 31 // (op 645) 
s_lshl_b32 s19, 1, s6 // (op 645) 
s_add_i32 s64, s19, s18 // (op 645) 


s_and_b32 s18, s0, s64 // (op 645) 


s_add_i32 s0, s7, s18 // (op 645) 


s_ashr_i32 s7, s0, s6 // (op 645) 


s_lshl_b32 s0, s49, 24 // (op 645) 
s_ashr_i32 s6, s0, 31 // (op 645) 

s_xor_b32 s0, s7, s6 // (op 645) 

s_sub_i32 s7, s0, s6 // (op 645) 


s_cmp_eq_i32 s1, 0 // (op 645) 
s_mov_b32 s0, scc // (op 645) 

s_cmp_eq_i32 s0, 0 // (op 645) 
s_cselect_b32 s1, s7, s51 // (op 645) 



v_lshl_add_u32 v17, s1, 4, v16 // (op 645) 


v_mul_lo_u32 v16, v18, s36 // (op 645) 
v_add_u32 v16, v16, v17 // (op 645) 


s_mul_i32 s1, s36, s35 // (op 647) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 647) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 647) least significant
s_mul_hi_u32 s6, s36, s34 // (op 647) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 647) most significant: sum
s_add_u32 s1, s1, s6 // (op 647) most significant: sum


s_mov_b32 s64, s38 // (op 647) reg expression
s_mov_b32 s65, s39 // (op 647) reg expression
s_mov_b32 s66, s0 // (op 647) reg expression
s_mov_b32 s67, 131072 // (op 647) call()

s_mul_hi_i32 s51, s16, s9 // (op 664) 
v_and_b32 v17, 15, v1 // (op 664) 
s_add_i32 s0, s51, s16 // (op 664) 

v_and_b32 v18, 63, v1 // (op 664) 
s_ashr_i32 s1, s0, 31 // (op 664) 
v_lshrrev_b32 v19, 4, v18 // (op 664) 

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
s_sub_u32 s68, s18, 1 // (op 664) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 664) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 664) 

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
s_lshl_b32 s68, 1, s6 // (op 664) 
s_add_i32 s69, s68, s7 // (op 664) 


s_and_b32 s7, s0, s69 // (op 664) 


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
s_add_i32 s68, s19, s18 // (op 664) 


s_and_b32 s18, s0, s68 // (op 664) 


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



v_lshl_add_u32 v18, s1, 4, v17 // (op 664) 


v_mul_lo_u32 v17, v19, s36 // (op 664) 
v_add_u32 v17, v17, v18 // (op 664) 


s_mul_i32 s1, s36, s35 // (op 666) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 666) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 666) least significant
s_mul_hi_u32 s6, s36, s34 // (op 666) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 666) most significant: sum
s_add_u32 s1, s1, s6 // (op 666) most significant: sum


s_mov_b32 s64, s38 // (op 666) reg expression
s_mov_b32 s65, s39 // (op 666) reg expression
s_mov_b32 s66, s0 // (op 666) reg expression
s_mov_b32 s67, 131072 // (op 666) call()


v_mov_b64 v[18:19], s[36:37] // (op 341) convert

v_mov_b32 v20, v16 // (op 341) reg expression
v_mov_b32 v21, v20 // (op 341) 
buffer_load_ubyte v22, v21, s[64:67], 0 offen offset: 0 // (op 341) Load value





v_mov_b32 v18, v14 // (op 345) reg expression

buffer_load_dwordx3 v[24:26], v18, s[56:59], 0 offen // (op 345) Load value



v_mov_b32 v23, v10 // (op 348) reg expression

buffer_load_dwordx3 v[18:20], v23, s[52:55], 0 offen // (op 348) Load value
buffer_load_dwordx3 v[28:30], v23, s[52:55], 0 offen offset:12 // (op 348) Load value



v_mov_b32 v21, v12 // (op 351) reg expression

buffer_load_ubyte v23, v21, s[60:63], 0 offen offset: 0 // (op 351) Load value


v_mov_b64 v[32:33], s[36:37] // (op 355) convert

s_lshl_b32 s51, s36, 2 // (op 355) 
v_add_u32 v21, v16, s51 // (op 355) 

v_mov_b32 v27, v21 // (op 355) 
buffer_load_ubyte v31, v27, s[64:67], 0 offen offset: 0 // (op 355) Load value






s_lshl_b64 s[0:1], s[26:27], 7 // (op 359) 
s_mul_i32 s7, s1, 6 // (op 359) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 359) least significant
s_mul_hi_u32 s51, s0, 6 // (op 359) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 359) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 359) 

v_add_u32 v16, v14, s0 // (op 359) 


buffer_load_dwordx3 v[32:34], v16, s[56:59], 0 offen // (op 359) Load value




s_lshl_b64 s[0:1], s[2:3], 7 // (op 362) 
s_mul_i32 s7, s1, 6 // (op 362) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 362) least significant
s_mul_hi_u32 s51, s0, 6 // (op 362) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 362) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 362) 

v_add_u32 v14, v10, s0 // (op 362) 


buffer_load_dwordx3 v[36:38], v14, s[52:55], 0 offen // (op 362) Load value
buffer_load_dwordx3 v[40:42], v14, s[52:55], 0 offen offset:12 // (op 362) Load value




s_lshl_b32 s51, s30, 2 // (op 365) 
v_add_u32 v10, v12, s51 // (op 365) 


buffer_load_ubyte v14, v10, s[60:63], 0 offen offset: 0 // (op 365) Load value


v_lshlrev_b32 v10, 4, v1 // (op 784) 
v_mul_lo_u32 v12, v10, 6 // (op 784) 

v_lshrrev_b32 v10, 3, v12 // (op 784) 

v_and_b32 v12, 268435455, v1 // (op 784) 
v_lshlrev_b32 v16, 2, v12 // (op 784) 

v_add_u32 v12, v10, v16 // (op 784) 





s_waitcnt vmcnt(8)
ds_write_b96 v12, v[24:26] // (op 367) Store local data 

v_lshlrev_b32 v10, 5, v1 // (op 731) 
v_mul_lo_u32 v12, v10, 6 // (op 731) 
v_lshrrev_b32 v16, 3, v12 // (op 731) 

v_lshrrev_b32 v12, 4, v10 // (op 731) 

v_lshlrev_b32 v10, 2, v12 // (op 731) 

v_add_u32 v12, v16, v10 // (op 731) 





s_waitcnt vmcnt(7)
ds_write_b96 v12, v[18:20] offset:2048 // (op 368) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v12, v[28:30] offset:2064 // (op 368) Store local data 

v_and_b32 v10, -32, v1 // (op 837) 
v_sub_u32 v12, v1, v10 // (op 837) 

v_and_b32 v10, 4294967264, v1 // (op 837) 
v_add_u32 v16, v10, v12 // (op 837) 





s_waitcnt vmcnt(5)
ds_write_b8 v16, v23 offset:6144 // (op 369) Store local data 

v_lshlrev_b32 v10, 5, v1 // (op 702) 
v_mul_lo_u32 v12, v10, 6 // (op 702) 
v_lshrrev_b32 v16, 3, v12 // (op 702) 

v_lshrrev_b32 v12, 4, v10 // (op 702) 

v_lshlrev_b32 v10, 2, v12 // (op 702) 

v_add_u32 v12, v16, v10 // (op 702) 


v_lshlrev_b32 v10, 4, v1 // (op 755) 
v_mul_lo_u32 v16, v10, 6 // (op 755) 

v_lshrrev_b32 v10, 3, v16 // (op 755) 

v_and_b32 v16, 268435455, v1 // (op 755) 
v_lshlrev_b32 v21, 2, v16 // (op 755) 

v_add_u32 v16, v10, v21 // (op 755) 


v_and_b32 v10, -32, v1 // (op 808) 
v_sub_u32 v21, v1, v10 // (op 808) 

v_and_b32 v10, 4294967264, v1 // (op 808) 
v_add_u32 v27, v10, v21 // (op 808) 


v_lshlrev_b32 v10, 5, v1 // (op 895) 
v_mul_lo_u32 v21, v10, 6 // (op 895) 
v_lshrrev_b32 v35, 3, v21 // (op 895) 

v_lshrrev_b32 v21, 4, v10 // (op 895) 

v_lshlrev_b32 v10, 2, v21 // (op 895) 

v_add_u32 v21, v35, v10 // (op 895) 


v_lshlrev_b32 v10, 4, v1 // (op 919) 
v_mul_lo_u32 v35, v10, 6 // (op 919) 

v_lshrrev_b32 v10, 3, v35 // (op 919) 

v_and_b32 v35, 268435455, v1 // (op 919) 
v_lshlrev_b32 v39, 2, v35 // (op 919) 

v_add_u32 v35, v10, v39 // (op 919) 


v_and_b32 v10, -32, v1 // (op 943) 
v_sub_u32 v39, v1, v10 // (op 943) 

v_and_b32 v10, 4294967264, v1 // (op 943) 
v_add_u32 v43, v10, v39 // (op 943) 


s_mov_b32 s51, 0 // (op 57) call()
s_ashr_i64 s[0:1], s[24:25], 7 // (op 56) 
s_mov_b32 s6, s0 // (op 56) convert

s_and_b32 s7, s6, 4294967294 // (op 56) 

s_cmp_lt_u32 s51, s7 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_56 if false)
RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_56:
s_add_u32 s51, s51, 2 // (op 58) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 988) 



ds_read_b96_tr_b6 v[44:46], v0 offset:2048 // (op 67) Transpose load local data 
ds_read_b96_tr_b6 v[47:49], v0 offset:2560 // (op 67) Transpose load local data 
s_barrier  // (op 1001) 



ds_read_b96_tr_b6 v[50:52], v5 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[53:55], v5 offset:256 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1015) 


ds_read_u8 v10, v3 offset:6144 // (op 77) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:49], v[50:55], a[0:3], v10, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 84) 




v_mov_b64 v[44:45], s[36:37] // (op 82) convert

v_mov_b32 v10, v17 // (op 82) 
s_lshl_b32 s0, s36, 3 // (op 82) 
v_add_u32 v39, v10, s0 // (op 82) 

v_mov_b32 v46, v39 // (op 82) 
buffer_load_ubyte v22, v46, s[64:67], 0 offen offset: 0 // (op 82) Load value





v_mov_b32 v39, v15 // (op 10) 
s_lshl_b64 s[0:1], s[26:27], 7 // (op 10) 
s_mul_i32 s7, s1, 6 // (op 10) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 10) least significant
s_mul_hi_u32 s18, s0, 6 // (op 10) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 10) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 10) 

s_lshl_b32 s6, s0, 1 // (op 10) 

v_add_u32 v44, v39, s6 // (op 10) 


buffer_load_dwordx3 v[24:26], v44, s[56:59], 0 offen // (op 10) Load value



v_mov_b32 v44, v11 // (op 4) 
s_lshl_b64 s[0:1], s[2:3], 7 // (op 4) 
s_mul_i32 s7, s1, 6 // (op 4) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 4) least significant
s_mul_hi_u32 s18, s0, 6 // (op 4) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 4) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 4) 

s_lshl_b32 s6, s0, 1 // (op 4) 

v_add_u32 v45, v44, s6 // (op 4) 


buffer_load_dwordx3 v[18:20], v45, s[52:55], 0 offen // (op 4) Load value
buffer_load_dwordx3 v[28:30], v45, s[52:55], 0 offen offset:12 // (op 4) Load value



v_mov_b32 v45, v13 // (op 17) 
s_lshl_b32 s0, s30, 3 // (op 17) 
v_add_u32 v46, v45, s0 // (op 17) 


buffer_load_ubyte v23, v46, s[60:63], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1038) 



s_waitcnt vmcnt(8)
ds_write_b96 v35, v[32:34] offset:6272 // (op 239) Store local data 
s_barrier  // (op 1032) 



s_waitcnt vmcnt(7)
ds_write_b96 v21, v[36:38] offset:8320 // (op 229) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v21, v[40:42] offset:8336 // (op 229) Store local data 
s_barrier  // (op 1044) 



s_waitcnt vmcnt(5)
ds_write_b8 v43, v14 offset:12416 // (op 240) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1029) 



ds_read_b96_tr_b6 v[46:48], v4 offset:8320 // (op 231) Transpose load local data 
ds_read_b96_tr_b6 v[49:51], v4 offset:8832 // (op 231) Transpose load local data 
s_barrier  // (op 1035) 
s_barrier  // (op 1099) 



ds_read_b96_tr_b6 v[52:54], v8 offset:6272 // (op 233) Transpose load local data 
ds_read_b96_tr_b6 v[55:57], v8 offset:6528 // (op 233) Transpose load local data 
s_barrier  // (op 1105) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1041) 


ds_read_u8 v58, v6 offset:12416 // (op 235) Load local data 
s_barrier  // (op 1111) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:51], v[52:57], a[0:3], v58, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 238) 




v_mov_b64 v[46:47], s[36:37] // (op 237) convert

s_lshl_b32 s0, s36, 2 // (op 237) 
s_mul_i32 s1, 3, s0 // (op 237) 

v_add_u32 v48, v10, s1 // (op 237) 

v_mov_b32 v49, v48 // (op 237) 
buffer_load_ubyte v31, v49, s[64:67], 0 offen offset: 0 // (op 237) Load value






s_lshl_b32 s0, s36, 3 // (op 656) 
v_add_u32 v17, v17, s0 // (op 656) 

s_lshl_b64 s[0:1], s[26:27], 7 // (op 241) 
s_mul_i32 s7, s1, 6 // (op 241) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 241) least significant
s_mul_hi_u32 s18, s0, 6 // (op 241) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 241) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 241) 

s_mul_i32 s6, 3, s0 // (op 241) 

v_add_u32 v10, v39, s6 // (op 241) 


buffer_load_dwordx3 v[32:34], v10, s[56:59], 0 offen // (op 241) Load value




s_lshl_b64 s[0:1], s[2:3], 7 // (op 228) 
s_mul_i32 s7, s1, 6 // (op 228) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 228) least significant
s_mul_hi_u32 s18, s0, 6 // (op 228) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 228) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 228) 

s_mul_i32 s6, 3, s0 // (op 228) 

v_add_u32 v10, v44, s6 // (op 228) 


s_lshl_b64 s[0:1], s[26:27], 7 // (op 528) 
buffer_load_dwordx3 v[36:38], v10, s[52:55], 0 offen // (op 228) Load value
s_mul_i32 s7, s1, 6 // (op 528) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 528) least significant
s_mul_hi_u32 s18, s0, 6 // (op 528) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 528) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 528) 

s_lshl_b32 s6, s0, 1 // (op 528) 

v_add_u32 v15, v15, s6 // (op 528) 

buffer_load_dwordx3 v[40:42], v10, s[52:55], 0 offen offset:12 // (op 228) Load value




s_lshl_b32 s0, s30, 2 // (op 242) 
s_mul_i32 s1, 3, s0 // (op 242) 

v_add_u32 v10, v45, s1 // (op 242) 


s_lshl_b64 s[0:1], s[2:3], 7 // (op 464) 
s_mul_i32 s7, s1, 6 // (op 464) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 464) least significant
s_mul_hi_u32 s18, s0, 6 // (op 464) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 464) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 464) 

s_lshl_b32 s6, s0, 1 // (op 464) 

v_add_u32 v11, v11, s6 // (op 464) 

buffer_load_ubyte v14, v10, s[60:63], 0 offen offset: 0 // (op 242) Load value


s_barrier  // (op 1004) 
s_lshl_b32 s0, s30, 3 // (op 592) 



v_add_u32 v13, v13, s0 // (op 592) 

s_waitcnt vmcnt(8)
ds_write_b96 v16, v[24:26] // (op 13) Store local data 
s_barrier  // (op 991) 



s_waitcnt vmcnt(7)
ds_write_b96 v12, v[18:20] offset:2048 // (op 6) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v12, v[28:30] offset:2064 // (op 6) Store local data 
s_barrier  // (op 1018) 



s_waitcnt vmcnt(5)
ds_write_b8 v27, v23 offset:6144 // (op 21) Store local data 
s_ashr_i64 s[0:1], s[24:25], 7 // (op 56) 
s_mov_b32 s6, s0 // (op 56) convert

s_and_b32 s7, s6, 4294967294 // (op 56) 

s_cmp_lt_u32 s51, s7 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_56 if true)
RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_56:















s_mul_hi_i32 s51, s16, s9 // (op 427) 
s_add_i32 s0, s51, s16 // (op 427) 

s_ashr_i32 s1, s0, 31 // (op 427) 
s_cmp_eq_i32 s9, 0 // (op 427) 
s_mov_b32 s51, scc // (op 427) 
s_cmp_eq_i32 s51, 0 // (op 427) 
s_cselect_b32 s6, 0, -1 // (op 427) 

s_and_b32 s7, s10, 31 // (op 427) 
s_lshl_b32 s51, 1, s7 // (op 427) 
s_add_i32 s18, s51, s6 // (op 427) 


s_and_b32 s6, s1, s18 // (op 427) 


s_add_i32 s1, s0, s6 // (op 427) 


s_ashr_i32 s0, s1, s7 // (op 427) 


s_lshl_b32 s1, s10, 24 // (op 427) 
s_ashr_i32 s51, s1, 31 // (op 427) 

s_xor_b32 s1, s0, s51 // (op 427) 

s_sub_i32 s0, s1, s51 // (op 427) 


s_mul_hi_i32 s1, s0, s11 // (op 427) 
s_add_i32 s51, s1, s0 // (op 427) 

s_ashr_i32 s1, s51, 31 // (op 427) 
s_cmp_eq_i32 s11, 0 // (op 427) 
s_mov_b32 s6, scc // (op 427) 
s_cmp_eq_i32 s6, 0 // (op 427) 
s_cselect_b32 s7, 0, -1 // (op 427) 

s_and_b32 s6, s12, 31 // (op 427) 
s_lshl_b32 s18, 1, s6 // (op 427) 
s_add_i32 s19, s18, s7 // (op 427) 


s_and_b32 s7, s1, s19 // (op 427) 


s_add_i32 s1, s51, s7 // (op 427) 


s_ashr_i32 s7, s1, s6 // (op 427) 


s_lshl_b32 s1, s12, 24 // (op 427) 
s_ashr_i32 s6, s1, 31 // (op 427) 

s_xor_b32 s1, s7, s6 // (op 427) 

s_sub_i32 s7, s1, s6 // (op 427) 


s_mul_i32 s1, s0, s14 // (op 427) 

s_sub_u32 s0, s16, s1 // (op 427) 

s_mul_i32 s1, s7, s14 // (op 427) 
s_add_u32 s1, s1, s0 // (op 427) 


s_mul_hi_i32 s0, s16, s8 // (op 427) 
s_add_i32 s51, s0, s16 // (op 427) 

s_ashr_i32 s0, s51, 31 // (op 427) 
s_cmp_eq_i32 s8, 0 // (op 427) 
s_mov_b32 s6, scc // (op 427) 
s_cmp_eq_i32 s6, 0 // (op 427) 
s_cselect_b32 s7, 0, -1 // (op 427) 

s_and_b32 s6, s15, 31 // (op 427) 
s_lshl_b32 s18, 1, s6 // (op 427) 
s_add_i32 s19, s18, s7 // (op 427) 


s_and_b32 s7, s0, s19 // (op 427) 


s_add_i32 s0, s51, s7 // (op 427) 


s_ashr_i32 s7, s0, s6 // (op 427) 


s_lshl_b32 s0, s15, 24 // (op 427) 
s_ashr_i32 s6, s0, 31 // (op 427) 

s_xor_b32 s0, s7, s6 // (op 427) 

s_sub_i32 s7, s0, s6 // (op 427) 


s_add_u32 s18, s4, 16 // (op 427) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 427) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 427) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 427) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 427) 

s_mul_i32 s0, s14, s18 // (op 427) 

s_mul_i32 s6, s50, s0 // (op 427) 

s_mul_i32 s0, s7, s6 // (op 427) 

s_sub_u32 s6, s16, s0 // (op 427) 

s_mov_b32 s0, s6 // (op 427) 
s_mul_hi_i32 s51, s0, s13 // (op 427) 
s_add_i32 s18, s51, s0 // (op 427) 


s_ashr_i32 s0, s18, 31 // (op 427) 
s_cmp_eq_i32 s13, 0 // (op 427) 
s_mov_b32 s19, scc // (op 427) 
s_cmp_eq_i32 s19, 0 // (op 427) 
s_cselect_b32 s51, 0, -1 // (op 427) 

s_and_b32 s19, s49, 31 // (op 427) 
s_lshl_b32 s68, 1, s19 // (op 427) 
s_add_i32 s69, s68, s51 // (op 427) 


s_and_b32 s51, s0, s69 // (op 427) 


s_add_i32 s0, s18, s51 // (op 427) 


s_ashr_i32 s18, s0, s19 // (op 427) 


s_lshl_b32 s0, s49, 24 // (op 427) 
s_ashr_i32 s19, s0, 31 // (op 427) 

s_xor_b32 s0, s18, s19 // (op 427) 

s_sub_i32 s18, s0, s19 // (op 427) 


s_mul_i32 s0, s48, s14 // (op 427) 
s_add_u32 s68, s20, 32 // (op 427) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 427) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 427) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 427) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 427) 

s_sub_i32 s19, s68, s0 // (op 427) 

s_mul_i32 s51, s18, s19 // (op 427) 


s_sub_u32 s18, s6, s51 // (op 427) 


s_add_u32 s6, s18, s0 // (op 427) 


s_cmp_eq_i32 s7, 0 // (op 427) 
s_mov_b32 s0, scc // (op 427) 

s_cmp_eq_i32 s0, 0 // (op 427) 
s_cselect_b32 s7, s6, s1 // (op 427) 



s_lshl_b32 s6, s7, 5 // (op 427) 

s_mov_b32 s1, 0 // (op 427) convert
s_mov_b32 s0, s6 // (op 427) convert

s_ashr_i64 s[6:7], s[24:25], 7 // (op 427) 
s_lshr_b32 s51, s6, 1 // (op 427) 

v_lshl_add_u32 v35, s51, 8, v1 // (op 427) 

v_mul_lo_u32 v11, v35, s3 // (op 427) most significant: low of low * high
v_mul_hi_u32 v39, v35, s2 // (op 427) most significant: high of low * low
v_mul_lo_u32 v10, v35, s2 // (op 427) least significant: low of low * low
v_add_u32 v11, v11, v39 // (op 427) most significant: sum


v_mov_b32 v35, s0 // (op 427) 
v_mov_b32 v39, s1 // (op 427) 
v_add_co_u32 v12, s[6:7], v35, v10 // (op 427) least significant half
v_addc_co_u32 v13, s[6:7], v39, v11, s[6:7] // (op 427) most significant half





v_mul_lo_u32 v11, v13, 6 // (op 427) most significant: low of high * low
v_mul_hi_u32 v35, v12, 6 // (op 427) most significant: high of low * low
v_mul_lo_u32 v10, v12, 6 // (op 427) least significant: low of low * low
v_add_u32 v11, v11, v35 // (op 427) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 427) 

v_mov_b32 v35, v12 // (op 427) convert

s_mul_i32 s1, s25, 6 // (op 429) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 429) least significant
s_mul_hi_u32 s51, s24, 6 // (op 429) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 429) most significant: sum

s_mul_i32 s7, s2, s1 // (op 429) most significant: low of low * high
s_mul_i32 s51, s3, s0 // (op 429) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 429) least significant
s_mul_hi_u32 s18, s2, s0 // (op 429) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 429) most significant: sum
s_add_u32 s7, s7, s18 // (op 429) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 429) 

s_mov_b32 s52, s22 // (op 429) reg expression
s_mov_b32 s53, s23 // (op 429) reg expression
s_mov_b32 s54, s0 // (op 429) reg expression
s_mov_b32 s55, 131072 // (op 429) call()

s_mul_hi_i32 s51, s16, s9 // (op 494) 
s_add_i32 s0, s51, s16 // (op 494) 

s_ashr_i32 s1, s0, 31 // (op 494) 
s_cmp_eq_i32 s9, 0 // (op 494) 
s_mov_b32 s51, scc // (op 494) 
s_cmp_eq_i32 s51, 0 // (op 494) 
s_cselect_b32 s6, 0, -1 // (op 494) 

s_and_b32 s7, s10, 31 // (op 494) 
s_lshl_b32 s51, 1, s7 // (op 494) 
s_add_i32 s18, s51, s6 // (op 494) 


s_and_b32 s6, s1, s18 // (op 494) 


s_add_i32 s1, s0, s6 // (op 494) 


s_ashr_i32 s0, s1, s7 // (op 494) 


s_lshl_b32 s1, s10, 24 // (op 494) 
s_ashr_i32 s51, s1, 31 // (op 494) 

s_xor_b32 s1, s0, s51 // (op 494) 

s_sub_i32 s0, s1, s51 // (op 494) 


s_mul_hi_i32 s1, s0, s11 // (op 494) 
s_add_i32 s51, s1, s0 // (op 494) 

s_ashr_i32 s1, s51, 31 // (op 494) 
s_cmp_eq_i32 s11, 0 // (op 494) 
s_mov_b32 s6, scc // (op 494) 
s_cmp_eq_i32 s6, 0 // (op 494) 
s_cselect_b32 s7, 0, -1 // (op 494) 

s_and_b32 s6, s12, 31 // (op 494) 
s_lshl_b32 s18, 1, s6 // (op 494) 
s_add_i32 s19, s18, s7 // (op 494) 


s_and_b32 s7, s1, s19 // (op 494) 


s_add_i32 s1, s51, s7 // (op 494) 


s_ashr_i32 s7, s1, s6 // (op 494) 


s_lshl_b32 s1, s12, 24 // (op 494) 
s_ashr_i32 s6, s1, 31 // (op 494) 

s_xor_b32 s1, s7, s6 // (op 494) 

s_sub_i32 s7, s1, s6 // (op 494) 


s_add_u32 s18, s4, 16 // (op 494) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 494) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 494) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 494) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 494) 

s_mul_i32 s1, s7, s18 // (op 494) 

s_sub_i32 s51, s0, s1 // (op 494) 


s_mul_hi_i32 s0, s16, s8 // (op 494) 
s_add_i32 s1, s0, s16 // (op 494) 

s_ashr_i32 s0, s1, 31 // (op 494) 
s_cmp_eq_i32 s8, 0 // (op 494) 
s_mov_b32 s6, scc // (op 494) 
s_cmp_eq_i32 s6, 0 // (op 494) 
s_cselect_b32 s7, 0, -1 // (op 494) 

s_and_b32 s6, s15, 31 // (op 494) 
s_lshl_b32 s68, 1, s6 // (op 494) 
s_add_i32 s69, s68, s7 // (op 494) 


s_and_b32 s7, s0, s69 // (op 494) 


s_add_i32 s0, s1, s7 // (op 494) 


s_ashr_i32 s1, s0, s6 // (op 494) 


s_lshl_b32 s0, s15, 24 // (op 494) 
s_ashr_i32 s6, s0, 31 // (op 494) 

s_xor_b32 s0, s1, s6 // (op 494) 

s_sub_i32 s1, s0, s6 // (op 494) 


s_mul_i32 s0, s14, s18 // (op 494) 

s_mul_i32 s6, s50, s0 // (op 494) 

s_mul_i32 s0, s1, s6 // (op 494) 

s_sub_u32 s6, s16, s0 // (op 494) 

s_mul_hi_i32 s0, s6, s13 // (op 494) 
s_add_i32 s7, s0, s6 // (op 494) 


s_ashr_i32 s0, s7, 31 // (op 494) 
s_cmp_eq_i32 s13, 0 // (op 494) 
s_mov_b32 s6, scc // (op 494) 
s_cmp_eq_i32 s6, 0 // (op 494) 
s_cselect_b32 s18, 0, -1 // (op 494) 

s_and_b32 s6, s49, 31 // (op 494) 
s_lshl_b32 s19, 1, s6 // (op 494) 
s_add_i32 s68, s19, s18 // (op 494) 


s_and_b32 s18, s0, s68 // (op 494) 


s_add_i32 s0, s7, s18 // (op 494) 


s_ashr_i32 s7, s0, s6 // (op 494) 


s_lshl_b32 s0, s49, 24 // (op 494) 
s_ashr_i32 s6, s0, 31 // (op 494) 

s_xor_b32 s0, s7, s6 // (op 494) 

s_sub_i32 s7, s0, s6 // (op 494) 


s_cmp_eq_i32 s1, 0 // (op 494) 
s_mov_b32 s0, scc // (op 494) 

s_cmp_eq_i32 s0, 0 // (op 494) 
s_cselect_b32 s1, s7, s51 // (op 494) 



s_lshl_b32 s0, s1, 4 // (op 494) 

s_mov_b32 s7, 0 // (op 494) convert
s_mov_b32 s6, s0 // (op 494) convert

s_ashr_i64 s[0:1], s[24:25], 7 // (op 494) 
s_lshr_b32 s51, s0, 1 // (op 494) 

v_lshl_add_u32 v39, s51, 8, v1 // (op 494) 

v_mul_lo_u32 v11, v39, s27 // (op 494) most significant: low of low * high
v_mul_hi_u32 v12, v39, s26 // (op 494) most significant: high of low * low
v_mul_lo_u32 v10, v39, s26 // (op 494) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 494) most significant: sum

v_mov_b32 v12, s6 // (op 494) 
v_mov_b32 v13, s7 // (op 494) 
v_add_co_u32 v10, s[0:1], v10, v12 // (op 494) least significant half
v_addc_co_u32 v11, s[0:1], v11, v13, s[0:1] // (op 494) most significant half





v_mul_lo_u32 v13, v11, 6 // (op 494) most significant: low of high * low
v_mul_hi_u32 v39, v10, 6 // (op 494) most significant: high of low * low
v_mul_lo_u32 v12, v10, 6 // (op 494) least significant: low of low * low
v_add_u32 v13, v13, v39 // (op 494) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 494) 

v_mov_b32 v39, v10 // (op 494) convert

s_mul_i32 s1, s25, 6 // (op 496) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 496) least significant
s_mul_hi_u32 s51, s24, 6 // (op 496) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 496) most significant: sum

s_mul_i32 s7, s26, s1 // (op 496) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 496) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 496) least significant
s_mul_hi_u32 s18, s26, s0 // (op 496) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 496) most significant: sum
s_add_u32 s7, s7, s18 // (op 496) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 496) 

s_mov_b32 s56, s28 // (op 496) reg expression
s_mov_b32 s57, s29 // (op 496) reg expression
s_mov_b32 s58, s0 // (op 496) reg expression
s_mov_b32 s59, 131072 // (op 496) call()

s_mul_hi_i32 s51, s16, s9 // (op 558) 
v_and_b32 v10, -32, v1 // (op 558) 
s_add_i32 s0, s51, s16 // (op 558) 

v_sub_u32 v11, v1, v10 // (op 558) 

s_ashr_i32 s1, s0, 31 // (op 558) 
v_lshrrev_b32 v10, 5, v1 // (op 558) 
s_cmp_eq_i32 s9, 0 // (op 558) 
s_mov_b32 s51, scc // (op 558) 
s_cmp_eq_i32 s51, 0 // (op 558) 
s_cselect_b32 s6, 0, -1 // (op 558) 

s_and_b32 s7, s10, 31 // (op 558) 
s_lshl_b32 s51, 1, s7 // (op 558) 
s_add_i32 s18, s51, s6 // (op 558) 


s_and_b32 s6, s1, s18 // (op 558) 


s_add_i32 s1, s0, s6 // (op 558) 


s_ashr_i32 s0, s1, s7 // (op 558) 


s_lshl_b32 s1, s10, 24 // (op 558) 
s_ashr_i32 s51, s1, 31 // (op 558) 

s_xor_b32 s1, s0, s51 // (op 558) 

s_sub_i32 s0, s1, s51 // (op 558) 


s_mul_hi_i32 s1, s0, s11 // (op 558) 
s_add_i32 s51, s1, s0 // (op 558) 

s_ashr_i32 s1, s51, 31 // (op 558) 
s_cmp_eq_i32 s11, 0 // (op 558) 
s_mov_b32 s6, scc // (op 558) 
s_cmp_eq_i32 s6, 0 // (op 558) 
s_cselect_b32 s7, 0, -1 // (op 558) 

s_and_b32 s6, s12, 31 // (op 558) 
s_lshl_b32 s18, 1, s6 // (op 558) 
s_add_i32 s19, s18, s7 // (op 558) 


s_and_b32 s7, s1, s19 // (op 558) 


s_add_i32 s1, s51, s7 // (op 558) 


s_ashr_i32 s7, s1, s6 // (op 558) 


s_lshl_b32 s1, s12, 24 // (op 558) 
s_ashr_i32 s6, s1, 31 // (op 558) 

s_xor_b32 s1, s7, s6 // (op 558) 

s_sub_i32 s7, s1, s6 // (op 558) 


s_mul_i32 s1, s0, s14 // (op 558) 

s_sub_u32 s0, s16, s1 // (op 558) 

s_mul_i32 s1, s7, s14 // (op 558) 
s_add_u32 s1, s1, s0 // (op 558) 


s_mul_hi_i32 s0, s16, s8 // (op 558) 
s_add_i32 s51, s0, s16 // (op 558) 

s_ashr_i32 s0, s51, 31 // (op 558) 
s_cmp_eq_i32 s8, 0 // (op 558) 
s_mov_b32 s6, scc // (op 558) 
s_cmp_eq_i32 s6, 0 // (op 558) 
s_cselect_b32 s7, 0, -1 // (op 558) 

s_and_b32 s6, s15, 31 // (op 558) 
s_lshl_b32 s18, 1, s6 // (op 558) 
s_add_i32 s19, s18, s7 // (op 558) 


s_and_b32 s7, s0, s19 // (op 558) 


s_add_i32 s0, s51, s7 // (op 558) 


s_ashr_i32 s7, s0, s6 // (op 558) 


s_lshl_b32 s0, s15, 24 // (op 558) 
s_ashr_i32 s6, s0, 31 // (op 558) 

s_xor_b32 s0, s7, s6 // (op 558) 

s_sub_i32 s7, s0, s6 // (op 558) 


s_add_u32 s18, s4, 16 // (op 558) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 558) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 558) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 558) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 558) 

s_mul_i32 s0, s14, s18 // (op 558) 

s_mul_i32 s6, s50, s0 // (op 558) 

s_mul_i32 s0, s7, s6 // (op 558) 

s_sub_u32 s6, s16, s0 // (op 558) 

s_mov_b32 s0, s6 // (op 558) 
s_mul_hi_i32 s51, s0, s13 // (op 558) 
s_add_i32 s18, s51, s0 // (op 558) 


s_ashr_i32 s0, s18, 31 // (op 558) 
s_cmp_eq_i32 s13, 0 // (op 558) 
s_mov_b32 s19, scc // (op 558) 
s_cmp_eq_i32 s19, 0 // (op 558) 
s_cselect_b32 s51, 0, -1 // (op 558) 

s_and_b32 s19, s49, 31 // (op 558) 
s_lshl_b32 s68, 1, s19 // (op 558) 
s_add_i32 s69, s68, s51 // (op 558) 


s_and_b32 s51, s0, s69 // (op 558) 


s_add_i32 s0, s18, s51 // (op 558) 


s_ashr_i32 s18, s0, s19 // (op 558) 


s_lshl_b32 s0, s49, 24 // (op 558) 
s_ashr_i32 s19, s0, 31 // (op 558) 

s_xor_b32 s0, s18, s19 // (op 558) 

s_sub_i32 s18, s0, s19 // (op 558) 


s_mul_i32 s0, s48, s14 // (op 558) 
s_add_u32 s68, s20, 32 // (op 558) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 558) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 558) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 558) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 558) 

s_sub_i32 s19, s68, s0 // (op 558) 

s_mul_i32 s51, s18, s19 // (op 558) 


s_sub_u32 s18, s6, s51 // (op 558) 


s_add_u32 s6, s18, s0 // (op 558) 


s_cmp_eq_i32 s7, 0 // (op 558) 
s_mov_b32 s0, scc // (op 558) 

s_cmp_eq_i32 s0, 0 // (op 558) 
s_cselect_b32 s7, s6, s1 // (op 558) 



v_lshl_add_u32 v12, s7, 5, v11 // (op 558) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 558) 
s_lshr_b32 s51, s0, 1 // (op 558) 

v_lshl_add_u32 v11, s51, 3, v10 // (op 558) 


v_mul_lo_u32 v10, v11, s30 // (op 558) 

v_add_u32 v11, v12, v10 // (op 558) 


s_mul_i32 s1, s30, s35 // (op 560) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 560) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 560) least significant
s_mul_hi_u32 s6, s30, s34 // (op 560) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 560) most significant: sum
s_add_u32 s1, s1, s6 // (op 560) most significant: sum


s_mov_b32 s60, s32 // (op 560) reg expression
s_mov_b32 s61, s33 // (op 560) reg expression
s_mov_b32 s62, s0 // (op 560) reg expression
s_mov_b32 s63, 131072 // (op 560) call()

s_mul_hi_i32 s51, s16, s9 // (op 622) 
v_and_b32 v10, 15, v1 // (op 622) 
s_add_i32 s0, s51, s16 // (op 622) 

v_and_b32 v12, 63, v1 // (op 622) 
s_ashr_i32 s1, s0, 31 // (op 622) 
v_lshrrev_b32 v13, 4, v12 // (op 622) 

s_cmp_eq_i32 s9, 0 // (op 622) 
s_mov_b32 s51, scc // (op 622) 
s_cmp_eq_i32 s51, 0 // (op 622) 
s_cselect_b32 s6, 0, -1 // (op 622) 

s_and_b32 s7, s10, 31 // (op 622) 
s_lshl_b32 s51, 1, s7 // (op 622) 
s_add_i32 s18, s51, s6 // (op 622) 


s_and_b32 s6, s1, s18 // (op 622) 


s_add_i32 s1, s0, s6 // (op 622) 


s_ashr_i32 s0, s1, s7 // (op 622) 


s_lshl_b32 s1, s10, 24 // (op 622) 
s_ashr_i32 s51, s1, 31 // (op 622) 

s_xor_b32 s1, s0, s51 // (op 622) 

s_sub_i32 s0, s1, s51 // (op 622) 


s_mul_hi_i32 s1, s0, s11 // (op 622) 
s_add_i32 s51, s1, s0 // (op 622) 

s_ashr_i32 s1, s51, 31 // (op 622) 
s_cmp_eq_i32 s11, 0 // (op 622) 
s_mov_b32 s6, scc // (op 622) 
s_cmp_eq_i32 s6, 0 // (op 622) 
s_cselect_b32 s7, 0, -1 // (op 622) 

s_and_b32 s6, s12, 31 // (op 622) 
s_lshl_b32 s18, 1, s6 // (op 622) 
s_add_i32 s19, s18, s7 // (op 622) 


s_and_b32 s7, s1, s19 // (op 622) 


s_add_i32 s1, s51, s7 // (op 622) 


s_ashr_i32 s7, s1, s6 // (op 622) 


s_lshl_b32 s1, s12, 24 // (op 622) 
s_ashr_i32 s6, s1, 31 // (op 622) 

s_xor_b32 s1, s7, s6 // (op 622) 

s_sub_i32 s7, s1, s6 // (op 622) 


s_add_u32 s18, s4, 16 // (op 622) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 622) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 622) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 622) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 622) 

s_mul_i32 s1, s7, s18 // (op 622) 

s_sub_i32 s51, s0, s1 // (op 622) 


s_mul_hi_i32 s0, s16, s8 // (op 622) 
s_add_i32 s1, s0, s16 // (op 622) 

s_ashr_i32 s0, s1, 31 // (op 622) 
s_cmp_eq_i32 s8, 0 // (op 622) 
s_mov_b32 s6, scc // (op 622) 
s_cmp_eq_i32 s6, 0 // (op 622) 
s_cselect_b32 s7, 0, -1 // (op 622) 

s_and_b32 s6, s15, 31 // (op 622) 
s_lshl_b32 s68, 1, s6 // (op 622) 
s_add_i32 s69, s68, s7 // (op 622) 


s_and_b32 s7, s0, s69 // (op 622) 


s_add_i32 s0, s1, s7 // (op 622) 


s_ashr_i32 s1, s0, s6 // (op 622) 


s_lshl_b32 s0, s15, 24 // (op 622) 
s_ashr_i32 s6, s0, 31 // (op 622) 

s_xor_b32 s0, s1, s6 // (op 622) 

s_sub_i32 s1, s0, s6 // (op 622) 


s_mul_i32 s0, s14, s18 // (op 622) 

s_mul_i32 s6, s50, s0 // (op 622) 

s_mul_i32 s0, s1, s6 // (op 622) 

s_sub_u32 s6, s16, s0 // (op 622) 

s_mul_hi_i32 s0, s6, s13 // (op 622) 
s_add_i32 s7, s0, s6 // (op 622) 


s_ashr_i32 s0, s7, 31 // (op 622) 
s_cmp_eq_i32 s13, 0 // (op 622) 
s_mov_b32 s6, scc // (op 622) 
s_cmp_eq_i32 s6, 0 // (op 622) 
s_cselect_b32 s18, 0, -1 // (op 622) 

s_and_b32 s6, s49, 31 // (op 622) 
s_lshl_b32 s19, 1, s6 // (op 622) 
s_add_i32 s68, s19, s18 // (op 622) 


s_and_b32 s18, s0, s68 // (op 622) 


s_add_i32 s0, s7, s18 // (op 622) 


s_ashr_i32 s7, s0, s6 // (op 622) 


s_lshl_b32 s0, s49, 24 // (op 622) 
s_ashr_i32 s6, s0, 31 // (op 622) 

s_xor_b32 s0, s7, s6 // (op 622) 

s_sub_i32 s7, s0, s6 // (op 622) 


s_cmp_eq_i32 s1, 0 // (op 622) 
s_mov_b32 s0, scc // (op 622) 

s_cmp_eq_i32 s0, 0 // (op 622) 
s_cselect_b32 s1, s7, s51 // (op 622) 



v_lshl_add_u32 v12, s1, 4, v10 // (op 622) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 622) 
s_lshr_b32 s51, s0, 1 // (op 622) 

v_lshl_add_u32 v10, s51, 3, v13 // (op 622) 


v_mul_lo_u32 v13, v10, s36 // (op 622) 
v_add_u32 v13, v13, v12 // (op 622) 


s_mul_i32 s1, s36, s35 // (op 624) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 624) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 624) least significant
s_mul_hi_u32 s6, s36, s34 // (op 624) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 624) most significant: sum
s_add_u32 s1, s1, s6 // (op 624) most significant: sum


s_mov_b32 s64, s38 // (op 624) reg expression
s_mov_b32 s65, s39 // (op 624) reg expression
s_mov_b32 s66, s0 // (op 624) reg expression
s_mov_b32 s67, 131072 // (op 624) call()

v_lshlrev_b32 v10, 5, v1 // (op 715) 
v_mul_lo_u32 v12, v10, 6 // (op 715) 
v_lshrrev_b32 v15, 3, v12 // (op 715) 

v_lshrrev_b32 v12, 4, v10 // (op 715) 

v_lshlrev_b32 v10, 2, v12 // (op 715) 

v_add_u32 v12, v15, v10 // (op 715) 


v_lshlrev_b32 v10, 4, v1 // (op 768) 
v_mul_lo_u32 v15, v10, 6 // (op 768) 

v_lshrrev_b32 v10, 3, v15 // (op 768) 

v_and_b32 v15, 268435455, v1 // (op 768) 
v_lshlrev_b32 v16, 2, v15 // (op 768) 

v_add_u32 v15, v10, v16 // (op 768) 


v_and_b32 v10, -32, v1 // (op 821) 
v_sub_u32 v16, v1, v10 // (op 821) 

v_and_b32 v10, 4294967264, v1 // (op 821) 
v_add_u32 v17, v10, v16 // (op 821) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 166) 
s_and_b32 s51, s0, 4294967294 // (op 166) 

s_ashr_i64 s[0:1], s[24:25], 7 // (op 165) 
s_mov_b32 s6, s0 // (op 165) convert

s_cmp_lt_u32 s51, s6 // (op 165) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_165:


s_add_u32 s51, s51, 1 // (op 167) 
v_mov_b32 v10, v35 // (op 170) 
v_mov_b32 v14, v10 // (op 170) reg expression

buffer_load_dwordx3 v[18:20], v14, s[52:55], 0 offen // (op 170) Load value
buffer_load_dwordx3 v[22:24], v14, s[52:55], 0 offen offset:12 // (op 170) Load value


s_lshl_b64 s[0:1], s[2:3], 7 // (op 417) 
s_mul_i32 s7, s1, 6 // (op 417) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 417) least significant
s_mul_hi_u32 s18, s0, 6 // (op 417) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 417) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 417) 

v_add_u32 v35, v35, s0 // (op 417) 



s_barrier  // (op 994) 
v_mov_b32 v10, v39 // (op 183) 



v_mov_b32 v14, v10 // (op 183) reg expression

buffer_load_dwordx3 v[26:28], v14, s[56:59], 0 offen // (op 183) Load value


s_lshl_b64 s[0:1], s[26:27], 7 // (op 486) 
s_mul_i32 s7, s1, 6 // (op 486) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 486) least significant
s_mul_hi_u32 s18, s0, 6 // (op 486) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 486) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 486) 

v_add_u32 v39, v39, s0 // (op 486) 

s_waitcnt vmcnt(2)
ds_write_b96 v12, v[18:20] offset:2048 // (op 171) Store local data 


v_mov_b32 v10, v11 // (op 184) 
v_mov_b32 v14, v10 // (op 184) reg expression

buffer_load_ubyte v16, v14, s[60:63], 0 offen offset: 0 // (op 184) Load value


s_lshl_b32 s0, s30, 2 // (op 550) 
v_add_u32 v11, v11, s0 // (op 550) 

s_waitcnt vmcnt(2)
ds_write_b96 v12, v[22:24] offset:2064 // (op 171) Store local data 

s_barrier  // (op 1007) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 997) 






ds_read_b96_tr_b6 v[18:20], v0 offset:2048 // (op 173) Transpose load local data 
ds_read_b96_tr_b6 v[21:23], v0 offset:2560 // (op 173) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b96 v15, v[26:28] // (op 181) Store local data 

s_barrier  // (op 1021) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1011) 






ds_read_b96_tr_b6 v[24:26], v5 // (op 175) Transpose load local data 
ds_read_b96_tr_b6 v[27:29], v5 offset:256 // (op 175) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v17, v16 offset:6144 // (op 182) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1025) 


ds_read_u8 v10, v3 offset:6144 // (op 177) Load local data 

v_mov_b64 v[30:31], s[36:37] // (op 179) convert

v_mov_b32 v14, v13 // (op 179) 
v_mov_b32 v16, v14 // (op 179) reg expression
v_mov_b32 v32, v16 // (op 179) 
buffer_load_ubyte v33, v32, s[64:67], 0 offen offset: 0 // (op 179) Load value




s_lshl_b32 s0, s36, 2 // (op 614) 
v_add_u32 v13, v13, s0 // (op 614) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:23], v[24:29], a[0:3], v10, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 180) 




s_ashr_i64 s[0:1], s[24:25], 7 // (op 165) 
s_mov_b32 s6, s0 // (op 165) convert

s_cmp_lt_u32 s51, s6 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_165 if true)
RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_165:






















s_barrier  // (op 1093) 






v_cmp_eq_f32 vcc, 0.00000, v7 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_125
s_nop 0
v_accvgpr_read v10, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 

v_lshrrev_b32 v10, 16, v14 // (op 52) 
v_lshrrev_b32 v11, 16, v15 // (op 52) 
v_lshrrev_b32 v12, 16, v16 // (op 52) 
v_lshrrev_b32 v13, 16, v17 // (op 52) 

s_mul_hi_i32 s0, s16, s9 // (op 863) 
v_lshrrev_b32 v0, 6, v1 // (op 863) 
s_add_i32 s1, s0, s16 // (op 863) 

v_and_b32 v8, 63, v1 // (op 863) 
s_ashr_i32 s0, s1, 31 // (op 863) 
v_lshrrev_b32 v3, 4, v8 // (op 863) 
s_cmp_eq_i32 s9, 0 // (op 863) 
s_mov_b32 s2, scc // (op 863) 
v_lshlrev_b32 v4, 2, v3 // (op 863) 

s_cmp_eq_i32 s2, 0 // (op 863) 
s_cselect_b32 s3, 0, -1 // (op 863) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 863) 


s_and_b32 s2, s10, 31 // (op 863) 
v_lshrrev_b32 v0, 2, v8 // (op 863) 

s_lshl_b32 s6, 1, s2 // (op 863) 
v_and_b32 v8, 3, v0 // (op 863) 

s_add_i32 s7, s6, s3 // (op 863) 


v_and_b32 v0, 3, v1 // (op 863) 
s_and_b32 s3, s0, s7 // (op 863) 


v_lshl_add_u32 v4, v8, 2, v0 // (op 863) 


s_add_i32 s0, s1, s3 // (op 863) 


s_ashr_i32 s1, s0, s2 // (op 863) 


s_lshl_b32 s0, s10, 24 // (op 863) 
s_ashr_i32 s2, s0, 31 // (op 863) 

s_xor_b32 s0, s1, s2 // (op 863) 

s_sub_i32 s1, s0, s2 // (op 863) 


s_mul_hi_i32 s0, s1, s11 // (op 863) 
s_add_i32 s2, s0, s1 // (op 863) 

s_ashr_i32 s0, s2, 31 // (op 863) 
s_cmp_eq_i32 s11, 0 // (op 863) 
s_mov_b32 s3, scc // (op 863) 
s_cmp_eq_i32 s3, 0 // (op 863) 
s_cselect_b32 s6, 0, -1 // (op 863) 

s_and_b32 s3, s12, 31 // (op 863) 
s_lshl_b32 s7, 1, s3 // (op 863) 
s_add_i32 s18, s7, s6 // (op 863) 


s_and_b32 s19, s0, s18 // (op 863) 


s_add_i32 s0, s2, s19 // (op 863) 


s_ashr_i32 s2, s0, s3 // (op 863) 


s_lshl_b32 s0, s12, 24 // (op 863) 
s_ashr_i32 s3, s0, 31 // (op 863) 

s_xor_b32 s0, s2, s3 // (op 863) 

s_sub_i32 s2, s0, s3 // (op 863) 


s_mul_i32 s0, s1, s14 // (op 863) 
s_sub_u32 s3, s16, s0 // (op 863) 

s_mul_i32 s0, s2, s14 // (op 863) 
s_add_u32 s0, s0, s3 // (op 863) 

s_mul_hi_i32 s3, s16, s8 // (op 863) 
s_add_i32 s6, s3, s16 // (op 863) 

s_ashr_i32 s3, s6, 31 // (op 863) 
s_cmp_eq_i32 s8, 0 // (op 863) 
s_mov_b32 s7, scc // (op 863) 
s_cmp_eq_i32 s7, 0 // (op 863) 
s_cselect_b32 s18, 0, -1 // (op 863) 

s_and_b32 s7, s15, 31 // (op 863) 
s_lshl_b32 s19, 1, s7 // (op 863) 
s_add_i32 s22, s19, s18 // (op 863) 


s_and_b32 s18, s3, s22 // (op 863) 


s_add_i32 s3, s6, s18 // (op 863) 


s_ashr_i32 s6, s3, s7 // (op 863) 


s_lshl_b32 s3, s15, 24 // (op 863) 
s_ashr_i32 s7, s3, 31 // (op 863) 

s_xor_b32 s3, s6, s7 // (op 863) 

s_sub_i32 s6, s3, s7 // (op 863) 


s_add_u32 s18, s4, 16 // (op 863) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 863) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 863) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 863) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 4 // (op 863) 

s_mul_i32 s3, s14, s18 // (op 863) 
s_mul_i32 s7, s50, s3 // (op 863) 

s_mul_i32 s3, s6, s7 // (op 863) 

s_sub_u32 s7, s16, s3 // (op 863) 

s_mov_b32 s3, s7 // (op 863) 
s_mul_hi_i32 s22, s3, s13 // (op 863) 
s_add_i32 s23, s22, s3 // (op 863) 


s_ashr_i32 s3, s23, 31 // (op 863) 
s_cmp_eq_i32 s13, 0 // (op 863) 
s_mov_b32 s22, scc // (op 863) 
s_cmp_eq_i32 s22, 0 // (op 863) 
s_cselect_b32 s24, 0, -1 // (op 863) 

s_and_b32 s22, s49, 31 // (op 863) 
s_lshl_b32 s25, 1, s22 // (op 863) 
s_add_i32 s26, s25, s24 // (op 863) 


s_and_b32 s24, s3, s26 // (op 863) 


s_add_i32 s3, s23, s24 // (op 863) 


s_ashr_i32 s23, s3, s22 // (op 863) 


s_lshl_b32 s3, s49, 24 // (op 863) 
s_ashr_i32 s22, s3, 31 // (op 863) 

s_xor_b32 s3, s23, s22 // (op 863) 

s_sub_i32 s23, s3, s22 // (op 863) 


s_mul_i32 s3, s48, s14 // (op 863) 
s_add_u32 s24, s20, 32 // (op 863) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 863) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 863) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 863) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 5 // (op 863) 

s_sub_i32 s22, s24, s3 // (op 863) 

s_mul_i32 s24, s23, s22 // (op 863) 

s_sub_u32 s22, s7, s24 // (op 863) 


s_add_u32 s7, s22, s3 // (op 863) 


s_cmp_eq_i32 s6, 0 // (op 863) 
s_mov_b32 s3, scc // (op 863) 

s_cmp_eq_i32 s3, 0 // (op 863) 
s_cselect_b32 s6, s7, s0 // (op 863) 


s_mul_i32 s0, s2, s18 // (op 863) 


v_lshl_add_u32 v0, s6, 5, v3 // (op 863) 


s_sub_i32 s2, s1, s0 // (op 863) 


s_cmp_eq_i32 s3, 0 // (op 863) 
s_cselect_b32 s22, s23, s2 // (op 863) 



v_lshl_add_u32 v3, s22, 4, v4 // (op 863) 


v_mul_lo_u32 v8, v3, s40 // (op 863) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 863) 


s_mul_i32 s7, s40, s5 // (op 865) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 865) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 865) least significant
s_mul_hi_u32 s1, s40, s4 // (op 865) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 865) most significant: sum
s_add_u32 s7, s7, s1 // (op 865) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 865) 

s_mov_b32 s0, s42 // (op 865) reg expression
s_mov_b32 s1, s43 // (op 865) reg expression
s_mov_b32 s2, s18 // (op 865) reg expression
s_mov_b32 s3, 131072 // (op 865) call()


v_pack_b32_f16 v4, v10, v11 // (op 54) 
v_pack_b32_f16 v5, v12, v13 // (op 54) 


buffer_store_dwordx2 v[4:5], v3, s[0:3], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_125:
s_mul_hi_i32 s6, s16, s9 // (op 685) 
v_lshrrev_b32 v0, 6, v1 // (op 685) 
s_add_i32 s7, s6, s16 // (op 685) 

v_and_b32 v8, 63, v1 // (op 685) 
s_ashr_i32 s6, s7, 31 // (op 685) 
v_lshrrev_b32 v3, 4, v8 // (op 685) 
s_cmp_eq_i32 s9, 0 // (op 685) 
s_mov_b32 s18, scc // (op 685) 
v_lshlrev_b32 v4, 2, v3 // (op 685) 

s_cmp_eq_i32 s18, 0 // (op 685) 
s_cselect_b32 s19, 0, -1 // (op 685) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 685) 


s_and_b32 s18, s10, 31 // (op 685) 
v_lshrrev_b32 v0, 2, v8 // (op 685) 

s_lshl_b32 s22, 1, s18 // (op 685) 
v_and_b32 v8, 3, v0 // (op 685) 

s_add_i32 s23, s22, s19 // (op 685) 


v_and_b32 v0, 3, v1 // (op 685) 
s_and_b32 s19, s6, s23 // (op 685) 


v_lshl_add_u32 v4, v8, 2, v0 // (op 685) 


s_add_i32 s6, s7, s19 // (op 685) 


s_ashr_i32 s7, s6, s18 // (op 685) 


s_lshl_b32 s6, s10, 24 // (op 685) 
s_ashr_i32 s18, s6, 31 // (op 685) 

s_xor_b32 s6, s7, s18 // (op 685) 

s_sub_i32 s7, s6, s18 // (op 685) 


s_mul_hi_i32 s6, s7, s11 // (op 685) 
s_add_i32 s18, s6, s7 // (op 685) 

s_ashr_i32 s6, s18, 31 // (op 685) 
s_cmp_eq_i32 s11, 0 // (op 685) 
s_mov_b32 s19, scc // (op 685) 
s_cmp_eq_i32 s19, 0 // (op 685) 
s_cselect_b32 s22, 0, -1 // (op 685) 

s_and_b32 s19, s12, 31 // (op 685) 
s_lshl_b32 s23, 1, s19 // (op 685) 
s_add_i32 s24, s23, s22 // (op 685) 


s_and_b32 s22, s6, s24 // (op 685) 


s_add_i32 s6, s18, s22 // (op 685) 


s_ashr_i32 s18, s6, s19 // (op 685) 


s_lshl_b32 s6, s12, 24 // (op 685) 
s_ashr_i32 s19, s6, 31 // (op 685) 

s_xor_b32 s6, s18, s19 // (op 685) 

s_sub_i32 s18, s6, s19 // (op 685) 


s_mul_i32 s6, s7, s14 // (op 685) 
s_sub_u32 s19, s16, s6 // (op 685) 

s_mul_i32 s6, s18, s14 // (op 685) 
s_add_u32 s6, s6, s19 // (op 685) 

s_mul_hi_i32 s19, s16, s8 // (op 685) 
s_add_i32 s22, s19, s16 // (op 685) 

s_ashr_i32 s19, s22, 31 // (op 685) 
s_cmp_eq_i32 s8, 0 // (op 685) 
s_mov_b32 s23, scc // (op 685) 
s_cmp_eq_i32 s23, 0 // (op 685) 
s_cselect_b32 s24, 0, -1 // (op 685) 

s_and_b32 s23, s15, 31 // (op 685) 
s_lshl_b32 s25, 1, s23 // (op 685) 
s_add_i32 s26, s25, s24 // (op 685) 


s_and_b32 s24, s19, s26 // (op 685) 


s_add_i32 s19, s22, s24 // (op 685) 


s_ashr_i32 s22, s19, s23 // (op 685) 


s_lshl_b32 s19, s15, 24 // (op 685) 
s_ashr_i32 s23, s19, 31 // (op 685) 

s_xor_b32 s19, s22, s23 // (op 685) 

s_sub_i32 s22, s19, s23 // (op 685) 


s_add_u32 s38, s4, 16 // (op 685) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 685) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 685) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 685) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 685) 

s_mul_i32 s19, s14, s38 // (op 685) 
s_mul_i32 s23, s50, s19 // (op 685) 

s_mul_i32 s19, s22, s23 // (op 685) 

s_sub_u32 s23, s16, s19 // (op 685) 

s_mov_b32 s19, s23 // (op 685) 
s_mul_hi_i32 s24, s19, s13 // (op 685) 
s_add_i32 s25, s24, s19 // (op 685) 


s_ashr_i32 s19, s25, 31 // (op 685) 
s_cmp_eq_i32 s13, 0 // (op 685) 
s_mov_b32 s24, scc // (op 685) 
s_cmp_eq_i32 s24, 0 // (op 685) 
s_cselect_b32 s26, 0, -1 // (op 685) 

s_and_b32 s24, s49, 31 // (op 685) 
s_lshl_b32 s27, 1, s24 // (op 685) 
s_add_i32 s28, s27, s26 // (op 685) 


s_and_b32 s26, s19, s28 // (op 685) 


s_add_i32 s19, s25, s26 // (op 685) 


s_ashr_i32 s25, s19, s24 // (op 685) 


s_lshl_b32 s19, s49, 24 // (op 685) 
s_ashr_i32 s24, s19, 31 // (op 685) 

s_xor_b32 s19, s25, s24 // (op 685) 

s_sub_i32 s25, s19, s24 // (op 685) 


s_mul_i32 s19, s48, s14 // (op 685) 
s_add_u32 s26, s20, 32 // (op 685) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 685) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 685) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 685) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 5 // (op 685) 

s_sub_i32 s24, s26, s19 // (op 685) 

s_mul_i32 s26, s25, s24 // (op 685) 

s_sub_u32 s24, s23, s26 // (op 685) 


s_add_u32 s23, s24, s19 // (op 685) 


s_cmp_eq_i32 s22, 0 // (op 685) 
s_mov_b32 s19, scc // (op 685) 

s_cmp_eq_i32 s19, 0 // (op 685) 
s_cselect_b32 s22, s23, s6 // (op 685) 


s_mul_i32 s6, s18, s38 // (op 685) 


v_lshl_add_u32 v0, s22, 5, v3 // (op 685) 


s_sub_i32 s18, s7, s6 // (op 685) 


s_cmp_eq_i32 s19, 0 // (op 685) 
s_cselect_b32 s6, s25, s18 // (op 685) 



v_lshl_add_u32 v3, s6, 4, v4 // (op 685) 


v_mul_lo_u32 v8, v3, s44 // (op 685) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 685) 


s_mul_i32 s7, s44, s5 // (op 687) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 687) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 687) least significant
s_mul_hi_u32 s19, s44, s4 // (op 687) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 687) most significant: sum
s_add_u32 s7, s7, s19 // (op 687) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 687) 

s_mov_b32 s36, s46 // (op 687) reg expression
s_mov_b32 s37, s47 // (op 687) reg expression
s_mov_b32 s38, s18 // (op 687) reg expression
s_mov_b32 s39, 131072 // (op 687) call()





buffer_load_dwordx2 v[4:5], v3, s[36:39], 0 offen // (op 138) Load value



v_accvgpr_read v10, a0 // (op 134) 
v_accvgpr_read v11, a1 // (op 134) 
v_accvgpr_read v12, a2 // (op 134) 
v_accvgpr_read v13, a3 // (op 134) 
v_mul_f32 v14, v9, v10 // (op 134) 
v_mul_f32 v15, v9, v11 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 139) 
v_lshrrev_b32 v11, 16, v4 // (op 139) 
v_lshlrev_b32 v10, 16, v10 // (op 139) 
v_lshlrev_b32 v11, 16, v11 // (op 139) 
v_mul_f32 v18, v7, v10 // (op 139) 
v_mul_f32 v19, v7, v11 // (op 139) 
v_and_b32 v10, 65535, v5 // (op 139) 
v_lshrrev_b32 v11, 16, v5 // (op 139) 
v_lshlrev_b32 v10, 16, v10 // (op 139) 
v_lshlrev_b32 v11, 16, v11 // (op 139) 
v_mul_f32 v20, v7, v10 // (op 139) 
v_mul_f32 v21, v7, v11 // (op 139) 


v_add_f32 v10, v18, v14 // (op 135) 
v_add_f32 v11, v19, v15 // (op 135) 
v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 


v_lshrrev_b32 v14, 16, v10 // (op 136) 
v_lshrrev_b32 v15, 16, v11 // (op 136) 
v_lshrrev_b32 v16, 16, v12 // (op 136) 
v_lshrrev_b32 v17, 16, v13 // (op 136) 

s_mul_hi_i32 s6, s16, s9 // (op 880) 
v_lshrrev_b32 v0, 6, v1 // (op 880) 
s_add_i32 s7, s6, s16 // (op 880) 

v_and_b32 v8, 63, v1 // (op 880) 
s_ashr_i32 s6, s7, 31 // (op 880) 
v_lshrrev_b32 v3, 4, v8 // (op 880) 
s_cmp_eq_i32 s9, 0 // (op 880) 
s_mov_b32 s18, scc // (op 880) 
v_lshlrev_b32 v4, 2, v3 // (op 880) 

s_cmp_eq_i32 s18, 0 // (op 880) 
s_cselect_b32 s19, 0, -1 // (op 880) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 880) 


s_and_b32 s18, s10, 31 // (op 880) 
v_lshrrev_b32 v0, 2, v8 // (op 880) 

s_lshl_b32 s22, 1, s18 // (op 880) 
v_and_b32 v8, 3, v0 // (op 880) 

s_add_i32 s23, s22, s19 // (op 880) 


v_and_b32 v0, 3, v1 // (op 880) 
s_and_b32 s19, s6, s23 // (op 880) 


v_lshl_add_u32 v4, v8, 2, v0 // (op 880) 


s_add_i32 s6, s7, s19 // (op 880) 


s_ashr_i32 s7, s6, s18 // (op 880) 


s_lshl_b32 s6, s10, 24 // (op 880) 
s_ashr_i32 s18, s6, 31 // (op 880) 

s_xor_b32 s6, s7, s18 // (op 880) 

s_sub_i32 s7, s6, s18 // (op 880) 


s_mul_hi_i32 s6, s7, s11 // (op 880) 
s_add_i32 s18, s6, s7 // (op 880) 

s_ashr_i32 s6, s18, 31 // (op 880) 
s_cmp_eq_i32 s11, 0 // (op 880) 
s_mov_b32 s19, scc // (op 880) 
s_cmp_eq_i32 s19, 0 // (op 880) 
s_cselect_b32 s22, 0, -1 // (op 880) 

s_and_b32 s19, s12, 31 // (op 880) 
s_lshl_b32 s23, 1, s19 // (op 880) 
s_add_i32 s24, s23, s22 // (op 880) 


s_and_b32 s22, s6, s24 // (op 880) 


s_add_i32 s6, s18, s22 // (op 880) 


s_ashr_i32 s18, s6, s19 // (op 880) 


s_lshl_b32 s6, s12, 24 // (op 880) 
s_ashr_i32 s19, s6, 31 // (op 880) 

s_xor_b32 s6, s18, s19 // (op 880) 

s_sub_i32 s18, s6, s19 // (op 880) 


s_mul_i32 s6, s7, s14 // (op 880) 
s_sub_u32 s19, s16, s6 // (op 880) 

s_mul_i32 s6, s18, s14 // (op 880) 
s_add_u32 s6, s6, s19 // (op 880) 

s_mul_hi_i32 s19, s16, s8 // (op 880) 
s_add_i32 s22, s19, s16 // (op 880) 

s_ashr_i32 s19, s22, 31 // (op 880) 
s_cmp_eq_i32 s8, 0 // (op 880) 
s_mov_b32 s23, scc // (op 880) 
s_cmp_eq_i32 s23, 0 // (op 880) 
s_cselect_b32 s24, 0, -1 // (op 880) 

s_and_b32 s23, s15, 31 // (op 880) 
s_lshl_b32 s25, 1, s23 // (op 880) 
s_add_i32 s26, s25, s24 // (op 880) 


s_and_b32 s24, s19, s26 // (op 880) 


s_add_i32 s19, s22, s24 // (op 880) 


s_ashr_i32 s22, s19, s23 // (op 880) 


s_lshl_b32 s19, s15, 24 // (op 880) 
s_ashr_i32 s23, s19, 31 // (op 880) 

s_xor_b32 s19, s22, s23 // (op 880) 

s_sub_i32 s22, s19, s23 // (op 880) 


s_add_u32 s38, s4, 16 // (op 880) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 880) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 880) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 880) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 880) 

s_mul_i32 s19, s14, s38 // (op 880) 
s_mul_i32 s23, s50, s19 // (op 880) 

s_mul_i32 s19, s22, s23 // (op 880) 

s_sub_u32 s23, s16, s19 // (op 880) 

s_mov_b32 s19, s23 // (op 880) 
s_mul_hi_i32 s24, s19, s13 // (op 880) 
s_add_i32 s25, s24, s19 // (op 880) 


s_ashr_i32 s19, s25, 31 // (op 880) 
s_cmp_eq_i32 s13, 0 // (op 880) 
s_mov_b32 s24, scc // (op 880) 
s_cmp_eq_i32 s24, 0 // (op 880) 
s_cselect_b32 s26, 0, -1 // (op 880) 

s_and_b32 s24, s49, 31 // (op 880) 
s_lshl_b32 s27, 1, s24 // (op 880) 
s_add_i32 s28, s27, s26 // (op 880) 


s_and_b32 s26, s19, s28 // (op 880) 


s_add_i32 s19, s25, s26 // (op 880) 


s_ashr_i32 s25, s19, s24 // (op 880) 


s_lshl_b32 s19, s49, 24 // (op 880) 
s_ashr_i32 s24, s19, 31 // (op 880) 

s_xor_b32 s19, s25, s24 // (op 880) 

s_sub_i32 s25, s19, s24 // (op 880) 


s_mul_i32 s19, s48, s14 // (op 880) 
s_add_u32 s26, s20, 32 // (op 880) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 880) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 880) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 880) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 5 // (op 880) 

s_sub_i32 s24, s26, s19 // (op 880) 

s_mul_i32 s26, s25, s24 // (op 880) 

s_sub_u32 s24, s23, s26 // (op 880) 


s_add_u32 s23, s24, s19 // (op 880) 


s_cmp_eq_i32 s22, 0 // (op 880) 
s_mov_b32 s19, scc // (op 880) 

s_cmp_eq_i32 s19, 0 // (op 880) 
s_cselect_b32 s22, s23, s6 // (op 880) 


s_mul_i32 s6, s18, s38 // (op 880) 


v_lshl_add_u32 v0, s22, 5, v3 // (op 880) 


s_sub_i32 s18, s7, s6 // (op 880) 


s_cmp_eq_i32 s19, 0 // (op 880) 
s_cselect_b32 s6, s25, s18 // (op 880) 



v_lshl_add_u32 v3, s6, 4, v4 // (op 880) 


v_mul_lo_u32 v8, v3, s40 // (op 880) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 880) 


s_mul_i32 s7, s40, s5 // (op 882) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 882) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 882) least significant
s_mul_hi_u32 s19, s40, s4 // (op 882) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 882) most significant: sum
s_add_u32 s7, s7, s19 // (op 882) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 882) 

s_mov_b32 s0, s42 // (op 882) reg expression
s_mov_b32 s1, s43 // (op 882) reg expression
s_mov_b32 s2, s18 // (op 882) reg expression
s_mov_b32 s3, 131072 // (op 882) call()


v_pack_b32_f16 v4, v14, v15 // (op 137) 
v_pack_b32_f16 v5, v16, v17 // (op 137) 


buffer_store_dwordx2 v[4:5], v3, s[0:3], 0 offen // (op 137) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.LRR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__end:
.size RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_, .LRR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__end-RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12544 // lds bytes
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
    DataType: B6
    DataTypeA: B6
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
  - .name: RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 12544
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 59
    .agpr_count: 4
    .max_flat_workgroup_size: 128
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

