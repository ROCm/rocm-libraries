// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_,@function
RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 63, v1 // (op 769) 
v_and_b32 v3, 48, v1 // (op 769) 
v_sub_u32 v4, v0, v3 // (op 769) 

v_lshrrev_b32 v3, 4, v0 // (op 769) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 769) 


v_lshrrev_b32 v3, 6, v1 // (op 769) 
v_lshlrev_b32 v4, 4, v3 // (op 769) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 769) 


v_lshlrev_b32 v0, 2, v3 // (op 769) 

v_lshrrev_b32 v3, 3, v0 // (op 769) 

v_and_b32 v0, 63, v1 // (op 822) 
v_and_b32 v4, 48, v1 // (op 822) 
v_sub_u32 v5, v0, v4 // (op 822) 

v_lshrrev_b32 v4, 4, v0 // (op 822) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 822) 


v_lshlrev_b32 v4, 4, v0 // (op 822) 
v_mul_lo_u32 v5, v4, 6 // (op 822) 

v_lshrrev_b32 v4, 3, v5 // (op 822) 

v_and_b32 v5, 268435455, v0 // (op 822) 

v_lshlrev_b32 v0, 2, v5 // (op 822) 

v_add_u32 v5, v4, v0 // (op 822) 


v_lshrrev_b32 v0, 6, v1 // (op 875) 
v_and_b32 v4, 15, v1 // (op 875) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 875) 


v_and_b32 v0, 48, v1 // (op 875) 
v_add_u32 v4, v0, v6 // (op 875) 


v_and_b32 v0, 48, v1 // (op 928) 
v_and_b32 v6, 15, v1 // (op 928) 
v_add_u32 v7, v0, v6 // (op 928) 


v_and_b32 v0, 63, v1 // (op 986) 
v_and_b32 v6, 48, v1 // (op 986) 
v_sub_u32 v8, v0, v6 // (op 986) 

v_lshrrev_b32 v6, 4, v0 // (op 986) 

v_lshl_add_u32 v0, v6, 5, v8 // (op 986) 


v_lshrrev_b32 v6, 6, v1 // (op 986) 
v_lshlrev_b32 v8, 4, v6 // (op 986) 

v_lshl_add_u32 v6, v0, 4, v8 // (op 986) 


v_lshlrev_b32 v0, 2, v6 // (op 986) 

v_lshrrev_b32 v6, 3, v0 // (op 986) 

v_and_b32 v0, 63, v1 // (op 1010) 
v_and_b32 v8, 48, v1 // (op 1010) 
v_sub_u32 v9, v0, v8 // (op 1010) 

v_lshrrev_b32 v8, 4, v0 // (op 1010) 

v_lshl_add_u32 v0, v8, 5, v9 // (op 1010) 


v_lshlrev_b32 v8, 4, v0 // (op 1010) 
v_mul_lo_u32 v9, v8, 6 // (op 1010) 

v_lshrrev_b32 v8, 3, v9 // (op 1010) 

v_and_b32 v9, 268435455, v0 // (op 1010) 

v_lshlrev_b32 v0, 2, v9 // (op 1010) 

v_add_u32 v9, v8, v0 // (op 1010) 


v_lshrrev_b32 v0, 6, v1 // (op 1034) 
v_and_b32 v8, 15, v1 // (op 1034) 
v_lshl_add_u32 v10, v0, 4, v8 // (op 1034) 


v_and_b32 v0, 48, v1 // (op 1034) 
v_add_u32 v8, v0, v10 // (op 1034) 


v_and_b32 v0, 48, v1 // (op 1058) 
v_and_b32 v10, 15, v1 // (op 1058) 
v_add_u32 v11, v0, v10 // (op 1058) 


v_mov_b32 v0, s6 // (op 2) Move value
v_mov_b32 v10, s7 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
s_mul_hi_i32 s0, s16, s9 // (op 479) 
v_lshlrev_b32 v12, 1, v1 // (op 479) 
s_add_i32 s1, s0, s16 // (op 479) 

s_ashr_i32 s0, s1, 31 // (op 479) 
s_cmp_eq_i32 s9, 0 // (op 479) 
s_mov_b32 s6, scc // (op 479) 
s_cmp_eq_i32 s6, 0 // (op 479) 
s_cselect_b32 s7, 0, -1 // (op 479) 

s_and_b32 s6, s10, 31 // (op 479) 
s_lshl_b32 s18, 1, s6 // (op 479) 
s_add_i32 s19, s18, s7 // (op 479) 


s_and_b32 s7, s0, s19 // (op 479) 


s_add_i32 s0, s1, s7 // (op 479) 


s_ashr_i32 s1, s0, s6 // (op 479) 


s_lshl_b32 s0, s10, 24 // (op 479) 
s_ashr_i32 s6, s0, 31 // (op 479) 

s_xor_b32 s0, s1, s6 // (op 479) 

s_sub_i32 s1, s0, s6 // (op 479) 


s_mul_hi_i32 s0, s1, s11 // (op 479) 
s_add_i32 s6, s0, s1 // (op 479) 

s_ashr_i32 s0, s6, 31 // (op 479) 
s_cmp_eq_i32 s11, 0 // (op 479) 
s_mov_b32 s7, scc // (op 479) 
s_cmp_eq_i32 s7, 0 // (op 479) 
s_cselect_b32 s18, 0, -1 // (op 479) 

s_and_b32 s7, s12, 31 // (op 479) 
s_lshl_b32 s19, 1, s7 // (op 479) 
s_add_i32 s51, s19, s18 // (op 479) 


s_and_b32 s18, s0, s51 // (op 479) 


s_add_i32 s0, s6, s18 // (op 479) 


s_ashr_i32 s6, s0, s7 // (op 479) 


s_lshl_b32 s0, s12, 24 // (op 479) 
s_ashr_i32 s7, s0, 31 // (op 479) 

s_xor_b32 s0, s6, s7 // (op 479) 

s_sub_i32 s6, s0, s7 // (op 479) 


s_mul_i32 s0, s1, s14 // (op 479) 

s_sub_u32 s1, s16, s0 // (op 479) 

s_mul_i32 s0, s6, s14 // (op 479) 
s_add_u32 s0, s0, s1 // (op 479) 


s_mul_hi_i32 s1, s16, s8 // (op 479) 
s_add_i32 s6, s1, s16 // (op 479) 

s_ashr_i32 s1, s6, 31 // (op 479) 
s_cmp_eq_i32 s8, 0 // (op 479) 
s_mov_b32 s7, scc // (op 479) 
s_cmp_eq_i32 s7, 0 // (op 479) 
s_cselect_b32 s18, 0, -1 // (op 479) 

s_and_b32 s7, s15, 31 // (op 479) 
s_lshl_b32 s19, 1, s7 // (op 479) 
s_add_i32 s51, s19, s18 // (op 479) 


s_and_b32 s18, s1, s51 // (op 479) 


s_add_i32 s1, s6, s18 // (op 479) 


s_ashr_i32 s6, s1, s7 // (op 479) 


s_lshl_b32 s1, s15, 24 // (op 479) 
s_ashr_i32 s7, s1, 31 // (op 479) 

s_xor_b32 s1, s6, s7 // (op 479) 

s_sub_i32 s6, s1, s7 // (op 479) 


s_add_u32 s18, s4, 16 // (op 479) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 479) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 479) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 479) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 479) 

s_mul_i32 s1, s14, s18 // (op 479) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 479) 

s_mul_i32 s1, s6, s7 // (op 479) 

s_sub_u32 s7, s16, s1 // (op 479) 

s_mov_b32 s1, s7 // (op 479) 
s_mul_hi_i32 s18, s1, s13 // (op 479) 
s_add_i32 s19, s18, s1 // (op 479) 


s_ashr_i32 s1, s19, 31 // (op 479) 
s_cmp_eq_i32 s13, 0 // (op 479) 
s_mov_b32 s18, scc // (op 479) 
s_cmp_eq_i32 s18, 0 // (op 479) 
s_cselect_b32 s51, 0, -1 // (op 479) 

s_and_b32 s18, s49, 31 // (op 479) 
s_lshl_b32 s52, 1, s18 // (op 479) 
s_add_i32 s53, s52, s51 // (op 479) 


s_and_b32 s51, s1, s53 // (op 479) 


s_add_i32 s1, s19, s51 // (op 479) 


s_ashr_i32 s19, s1, s18 // (op 479) 


s_lshl_b32 s1, s49, 24 // (op 479) 
s_ashr_i32 s18, s1, 31 // (op 479) 

s_xor_b32 s1, s19, s18 // (op 479) 

s_sub_i32 s19, s1, s18 // (op 479) 


s_mul_i32 s1, s48, s14 // (op 479) 
s_add_u32 s52, s20, 16 // (op 479) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 479) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 479) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 479) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 479) 

s_sub_i32 s18, s52, s1 // (op 479) 

s_mul_i32 s51, s19, s18 // (op 479) 


s_sub_u32 s18, s7, s51 // (op 479) 


s_add_u32 s7, s18, s1 // (op 479) 


s_cmp_eq_i32 s6, 0 // (op 479) 
s_mov_b32 s1, scc // (op 479) 

s_cmp_eq_i32 s1, 0 // (op 479) 
s_cselect_b32 s6, s7, s0 // (op 479) 



s_lshl_b32 s7, s6, 4 // (op 479) 

s_mov_b32 s1, 0 // (op 479) convert
s_mov_b32 s0, s7 // (op 479) convert

v_mul_lo_u32 v15, v12, s3 // (op 479) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 479) most significant: high of low * low
v_mul_lo_u32 v14, v12, s2 // (op 479) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 479) most significant: sum


v_mov_b32 v12, s0 // (op 479) 
v_mov_b32 v13, s1 // (op 479) 
v_add_co_u32 v16, s[6:7], v12, v14 // (op 479) least significant half
v_addc_co_u32 v17, s[6:7], v13, v15, s[6:7] // (op 479) most significant half



v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 479) 


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 479) 

v_mov_b32 v14, v12 // (op 479) convert

s_mul_i32 s1, s2, s25 // (op 481) most significant: low of low * high
s_mul_i32 s6, s3, s24 // (op 481) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 481) least significant
s_mul_hi_u32 s7, s2, s24 // (op 481) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 481) most significant: sum
s_add_u32 s1, s1, s7 // (op 481) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 481) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 481) 

s_mov_b32 s52, s22 // (op 481) reg expression
s_mov_b32 s53, s23 // (op 481) reg expression
s_mov_b32 s54, s0 // (op 481) reg expression
s_mov_b32 s55, 131072 // (op 481) call()

s_mul_hi_i32 s51, s16, s9 // (op 498) 
v_lshlrev_b32 v12, 1, v1 // (op 498) 
s_add_i32 s0, s51, s16 // (op 498) 

s_ashr_i32 s1, s0, 31 // (op 498) 
s_cmp_eq_i32 s9, 0 // (op 498) 
s_mov_b32 s51, scc // (op 498) 
s_cmp_eq_i32 s51, 0 // (op 498) 
s_cselect_b32 s6, 0, -1 // (op 498) 

s_and_b32 s7, s10, 31 // (op 498) 
s_lshl_b32 s51, 1, s7 // (op 498) 
s_add_i32 s18, s51, s6 // (op 498) 


s_and_b32 s6, s1, s18 // (op 498) 


s_add_i32 s1, s0, s6 // (op 498) 


s_ashr_i32 s0, s1, s7 // (op 498) 


s_lshl_b32 s1, s10, 24 // (op 498) 
s_ashr_i32 s51, s1, 31 // (op 498) 

s_xor_b32 s1, s0, s51 // (op 498) 

s_sub_i32 s0, s1, s51 // (op 498) 


s_mul_hi_i32 s1, s0, s11 // (op 498) 
s_add_i32 s51, s1, s0 // (op 498) 

s_ashr_i32 s1, s51, 31 // (op 498) 
s_cmp_eq_i32 s11, 0 // (op 498) 
s_mov_b32 s6, scc // (op 498) 
s_cmp_eq_i32 s6, 0 // (op 498) 
s_cselect_b32 s7, 0, -1 // (op 498) 

s_and_b32 s6, s12, 31 // (op 498) 
s_lshl_b32 s18, 1, s6 // (op 498) 
s_add_i32 s19, s18, s7 // (op 498) 


s_and_b32 s7, s1, s19 // (op 498) 


s_add_i32 s1, s51, s7 // (op 498) 


s_ashr_i32 s7, s1, s6 // (op 498) 


s_lshl_b32 s1, s12, 24 // (op 498) 
s_ashr_i32 s6, s1, 31 // (op 498) 

s_xor_b32 s1, s7, s6 // (op 498) 

s_sub_i32 s7, s1, s6 // (op 498) 


s_mul_i32 s1, s0, s14 // (op 498) 

s_sub_u32 s0, s16, s1 // (op 498) 

s_mul_i32 s1, s7, s14 // (op 498) 
s_add_u32 s1, s1, s0 // (op 498) 


s_mul_hi_i32 s0, s16, s8 // (op 498) 
s_add_i32 s51, s0, s16 // (op 498) 

s_ashr_i32 s0, s51, 31 // (op 498) 
s_cmp_eq_i32 s8, 0 // (op 498) 
s_mov_b32 s6, scc // (op 498) 
s_cmp_eq_i32 s6, 0 // (op 498) 
s_cselect_b32 s7, 0, -1 // (op 498) 

s_and_b32 s6, s15, 31 // (op 498) 
s_lshl_b32 s18, 1, s6 // (op 498) 
s_add_i32 s19, s18, s7 // (op 498) 


s_and_b32 s7, s0, s19 // (op 498) 


s_add_i32 s0, s51, s7 // (op 498) 


s_ashr_i32 s7, s0, s6 // (op 498) 


s_lshl_b32 s0, s15, 24 // (op 498) 
s_ashr_i32 s6, s0, 31 // (op 498) 

s_xor_b32 s0, s7, s6 // (op 498) 

s_sub_i32 s7, s0, s6 // (op 498) 


s_add_u32 s18, s4, 16 // (op 498) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 498) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 498) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 498) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 498) 

s_mul_i32 s0, s14, s18 // (op 498) 

s_mul_i32 s6, s50, s0 // (op 498) 

s_mul_i32 s0, s7, s6 // (op 498) 

s_sub_u32 s6, s16, s0 // (op 498) 

s_mov_b32 s0, s6 // (op 498) 
s_mul_hi_i32 s51, s0, s13 // (op 498) 
s_add_i32 s18, s51, s0 // (op 498) 


s_ashr_i32 s0, s18, 31 // (op 498) 
s_cmp_eq_i32 s13, 0 // (op 498) 
s_mov_b32 s19, scc // (op 498) 
s_cmp_eq_i32 s19, 0 // (op 498) 
s_cselect_b32 s51, 0, -1 // (op 498) 

s_and_b32 s19, s49, 31 // (op 498) 
s_lshl_b32 s56, 1, s19 // (op 498) 
s_add_i32 s57, s56, s51 // (op 498) 


s_and_b32 s51, s0, s57 // (op 498) 


s_add_i32 s0, s18, s51 // (op 498) 


s_ashr_i32 s18, s0, s19 // (op 498) 


s_lshl_b32 s0, s49, 24 // (op 498) 
s_ashr_i32 s19, s0, 31 // (op 498) 

s_xor_b32 s0, s18, s19 // (op 498) 

s_sub_i32 s18, s0, s19 // (op 498) 


s_mul_i32 s0, s48, s14 // (op 498) 
s_add_u32 s56, s20, 16 // (op 498) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 498) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 498) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 498) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 498) 

s_sub_i32 s19, s56, s0 // (op 498) 

s_mul_i32 s51, s18, s19 // (op 498) 


s_sub_u32 s18, s6, s51 // (op 498) 


s_add_u32 s6, s18, s0 // (op 498) 


s_cmp_eq_i32 s7, 0 // (op 498) 
s_mov_b32 s0, scc // (op 498) 

s_cmp_eq_i32 s0, 0 // (op 498) 
s_cselect_b32 s7, s6, s1 // (op 498) 



s_lshl_b32 s6, s7, 4 // (op 498) 

s_mov_b32 s1, 0 // (op 498) convert
s_mov_b32 s0, s6 // (op 498) convert

v_mul_lo_u32 v17, v12, s3 // (op 498) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 498) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 498) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 498) most significant: sum


v_mov_b32 v15, s0 // (op 498) 
v_mov_b32 v12, s1 // (op 498) 
v_add_co_u32 v18, s[6:7], v15, v16 // (op 498) least significant half
v_addc_co_u32 v19, s[6:7], v12, v17, s[6:7] // (op 498) most significant half



v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 498) 


v_lshrrev_b64 v[12:13], 3, v[18:19] // (op 498) 

v_mov_b32 v15, v12 // (op 498) convert

s_mul_i32 s1, s2, s25 // (op 500) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 500) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 500) least significant
s_mul_hi_u32 s6, s2, s24 // (op 500) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 500) most significant: sum
s_add_u32 s1, s1, s6 // (op 500) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 500) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 500) 

s_mov_b32 s52, s22 // (op 500) reg expression
s_mov_b32 s53, s23 // (op 500) reg expression
s_mov_b32 s54, s0 // (op 500) reg expression
s_mov_b32 s55, 131072 // (op 500) call()

s_mul_hi_i32 s51, s16, s9 // (op 543) 
v_lshlrev_b32 v12, 1, v1 // (op 543) 
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
s_sub_u32 s56, s18, 1 // (op 543) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 543) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 543) 

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
s_lshl_b32 s56, 1, s6 // (op 543) 
s_add_i32 s57, s56, s7 // (op 543) 


s_and_b32 s7, s0, s57 // (op 543) 


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
s_add_i32 s56, s19, s18 // (op 543) 


s_and_b32 s18, s0, s56 // (op 543) 


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

v_mul_lo_u32 v17, v12, s27 // (op 543) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 543) most significant: high of low * low
v_mul_lo_u32 v16, v12, s26 // (op 543) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 543) most significant: sum

v_mov_b32 v13, s6 // (op 543) 
v_mov_b32 v18, s7 // (op 543) 
v_add_co_u32 v16, s[0:1], v16, v13 // (op 543) least significant half
v_addc_co_u32 v17, s[0:1], v17, v18, s[0:1] // (op 543) most significant half





v_mul_lo_u32 v13, v17, 6 // (op 543) most significant: low of high * low
v_mul_hi_u32 v18, v16, 6 // (op 543) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 543) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 543) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 543) 

v_mov_b32 v12, v16 // (op 543) convert

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

s_mul_hi_i32 s51, s16, s9 // (op 562) 
v_lshlrev_b32 v13, 1, v1 // (op 562) 
s_add_i32 s0, s51, s16 // (op 562) 

s_ashr_i32 s1, s0, 31 // (op 562) 
s_cmp_eq_i32 s9, 0 // (op 562) 
s_mov_b32 s51, scc // (op 562) 
s_cmp_eq_i32 s51, 0 // (op 562) 
s_cselect_b32 s6, 0, -1 // (op 562) 

s_and_b32 s7, s10, 31 // (op 562) 
s_lshl_b32 s51, 1, s7 // (op 562) 
s_add_i32 s18, s51, s6 // (op 562) 


s_and_b32 s6, s1, s18 // (op 562) 


s_add_i32 s1, s0, s6 // (op 562) 


s_ashr_i32 s0, s1, s7 // (op 562) 


s_lshl_b32 s1, s10, 24 // (op 562) 
s_ashr_i32 s51, s1, 31 // (op 562) 

s_xor_b32 s1, s0, s51 // (op 562) 

s_sub_i32 s0, s1, s51 // (op 562) 


s_mul_hi_i32 s1, s0, s11 // (op 562) 
s_add_i32 s51, s1, s0 // (op 562) 

s_ashr_i32 s1, s51, 31 // (op 562) 
s_cmp_eq_i32 s11, 0 // (op 562) 
s_mov_b32 s6, scc // (op 562) 
s_cmp_eq_i32 s6, 0 // (op 562) 
s_cselect_b32 s7, 0, -1 // (op 562) 

s_and_b32 s6, s12, 31 // (op 562) 
s_lshl_b32 s18, 1, s6 // (op 562) 
s_add_i32 s19, s18, s7 // (op 562) 


s_and_b32 s7, s1, s19 // (op 562) 


s_add_i32 s1, s51, s7 // (op 562) 


s_ashr_i32 s7, s1, s6 // (op 562) 


s_lshl_b32 s1, s12, 24 // (op 562) 
s_ashr_i32 s6, s1, 31 // (op 562) 

s_xor_b32 s1, s7, s6 // (op 562) 

s_sub_i32 s7, s1, s6 // (op 562) 


s_add_u32 s18, s4, 16 // (op 562) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 562) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 562) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 562) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 562) 

s_mul_i32 s1, s7, s18 // (op 562) 

s_sub_i32 s51, s0, s1 // (op 562) 


s_mul_hi_i32 s0, s16, s8 // (op 562) 
s_add_i32 s1, s0, s16 // (op 562) 

s_ashr_i32 s0, s1, 31 // (op 562) 
s_cmp_eq_i32 s8, 0 // (op 562) 
s_mov_b32 s6, scc // (op 562) 
s_cmp_eq_i32 s6, 0 // (op 562) 
s_cselect_b32 s7, 0, -1 // (op 562) 

s_and_b32 s6, s15, 31 // (op 562) 
s_lshl_b32 s60, 1, s6 // (op 562) 
s_add_i32 s61, s60, s7 // (op 562) 


s_and_b32 s7, s0, s61 // (op 562) 


s_add_i32 s0, s1, s7 // (op 562) 


s_ashr_i32 s1, s0, s6 // (op 562) 


s_lshl_b32 s0, s15, 24 // (op 562) 
s_ashr_i32 s6, s0, 31 // (op 562) 

s_xor_b32 s0, s1, s6 // (op 562) 

s_sub_i32 s1, s0, s6 // (op 562) 


s_mul_i32 s0, s14, s18 // (op 562) 

s_mul_i32 s6, s50, s0 // (op 562) 

s_mul_i32 s0, s1, s6 // (op 562) 

s_sub_u32 s6, s16, s0 // (op 562) 

s_mul_hi_i32 s0, s6, s13 // (op 562) 
s_add_i32 s7, s0, s6 // (op 562) 


s_ashr_i32 s0, s7, 31 // (op 562) 
s_cmp_eq_i32 s13, 0 // (op 562) 
s_mov_b32 s6, scc // (op 562) 
s_cmp_eq_i32 s6, 0 // (op 562) 
s_cselect_b32 s18, 0, -1 // (op 562) 

s_and_b32 s6, s49, 31 // (op 562) 
s_lshl_b32 s19, 1, s6 // (op 562) 
s_add_i32 s60, s19, s18 // (op 562) 


s_and_b32 s18, s0, s60 // (op 562) 


s_add_i32 s0, s7, s18 // (op 562) 


s_ashr_i32 s7, s0, s6 // (op 562) 


s_lshl_b32 s0, s49, 24 // (op 562) 
s_ashr_i32 s6, s0, 31 // (op 562) 

s_xor_b32 s0, s7, s6 // (op 562) 

s_sub_i32 s7, s0, s6 // (op 562) 


s_cmp_eq_i32 s1, 0 // (op 562) 
s_mov_b32 s0, scc // (op 562) 

s_cmp_eq_i32 s0, 0 // (op 562) 
s_cselect_b32 s1, s7, s51 // (op 562) 



s_lshl_b32 s0, s1, 4 // (op 562) 

s_mov_b32 s7, 0 // (op 562) convert
s_mov_b32 s6, s0 // (op 562) convert

v_mul_lo_u32 v17, v13, s27 // (op 562) most significant: low of low * high
v_mul_hi_u32 v18, v13, s26 // (op 562) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 562) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 562) most significant: sum

v_mov_b32 v18, s6 // (op 562) 
v_mov_b32 v19, s7 // (op 562) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 562) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 562) most significant half





v_mul_lo_u32 v19, v17, 6 // (op 562) most significant: low of high * low
v_mul_hi_u32 v13, v16, 6 // (op 562) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 562) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 562) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 562) 

v_mov_b32 v13, v16 // (op 562) convert

s_mul_i32 s1, s25, 6 // (op 564) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 564) least significant
s_mul_hi_u32 s51, s24, 6 // (op 564) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 564) most significant: sum

s_mul_i32 s7, s26, s1 // (op 564) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 564) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 564) least significant
s_mul_hi_u32 s18, s26, s0 // (op 564) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 564) most significant: sum
s_add_u32 s7, s7, s18 // (op 564) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 564) 

s_mov_b32 s56, s28 // (op 564) reg expression
s_mov_b32 s57, s29 // (op 564) reg expression
s_mov_b32 s58, s0 // (op 564) reg expression
s_mov_b32 s59, 131072 // (op 564) call()

s_mul_hi_i32 s51, s16, s9 // (op 607) 
v_and_b32 v16, -16, v1 // (op 607) 
s_add_i32 s0, s51, s16 // (op 607) 

v_sub_u32 v17, v1, v16 // (op 607) 

s_ashr_i32 s1, s0, 31 // (op 607) 
v_lshrrev_b32 v16, 4, v1 // (op 607) 
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
s_sub_u32 s60, s18, 1 // (op 607) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 607) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 607) 

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
s_lshl_b32 s60, 1, s19 // (op 607) 
s_add_i32 s61, s60, s51 // (op 607) 


s_and_b32 s51, s0, s61 // (op 607) 


s_add_i32 s0, s18, s51 // (op 607) 


s_ashr_i32 s18, s0, s19 // (op 607) 


s_lshl_b32 s0, s49, 24 // (op 607) 
s_ashr_i32 s19, s0, 31 // (op 607) 

s_xor_b32 s0, s18, s19 // (op 607) 

s_sub_i32 s18, s0, s19 // (op 607) 


s_mul_i32 s0, s48, s14 // (op 607) 
s_add_u32 s60, s20, 16 // (op 607) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 607) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 607) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 607) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 607) 

s_sub_i32 s19, s60, s0 // (op 607) 

s_mul_i32 s51, s18, s19 // (op 607) 


s_sub_u32 s18, s6, s51 // (op 607) 


s_add_u32 s6, s18, s0 // (op 607) 


s_cmp_eq_i32 s7, 0 // (op 607) 
s_mov_b32 s0, scc // (op 607) 

s_cmp_eq_i32 s0, 0 // (op 607) 
s_cselect_b32 s7, s6, s1 // (op 607) 



v_lshl_add_u32 v18, s7, 4, v17 // (op 607) 


v_mul_lo_u32 v17, v16, s30 // (op 607) 

v_add_u32 v16, v18, v17 // (op 607) 


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

s_mul_hi_i32 s51, s16, s9 // (op 626) 
v_and_b32 v17, -16, v1 // (op 626) 
s_add_i32 s0, s51, s16 // (op 626) 

v_sub_u32 v18, v1, v17 // (op 626) 

s_ashr_i32 s1, s0, 31 // (op 626) 
v_lshrrev_b32 v17, 4, v1 // (op 626) 
s_cmp_eq_i32 s9, 0 // (op 626) 
s_mov_b32 s51, scc // (op 626) 
s_cmp_eq_i32 s51, 0 // (op 626) 
s_cselect_b32 s6, 0, -1 // (op 626) 

s_and_b32 s7, s10, 31 // (op 626) 
s_lshl_b32 s51, 1, s7 // (op 626) 
s_add_i32 s18, s51, s6 // (op 626) 


s_and_b32 s6, s1, s18 // (op 626) 


s_add_i32 s1, s0, s6 // (op 626) 


s_ashr_i32 s0, s1, s7 // (op 626) 


s_lshl_b32 s1, s10, 24 // (op 626) 
s_ashr_i32 s51, s1, 31 // (op 626) 

s_xor_b32 s1, s0, s51 // (op 626) 

s_sub_i32 s0, s1, s51 // (op 626) 


s_mul_hi_i32 s1, s0, s11 // (op 626) 
s_add_i32 s51, s1, s0 // (op 626) 

s_ashr_i32 s1, s51, 31 // (op 626) 
s_cmp_eq_i32 s11, 0 // (op 626) 
s_mov_b32 s6, scc // (op 626) 
s_cmp_eq_i32 s6, 0 // (op 626) 
s_cselect_b32 s7, 0, -1 // (op 626) 

s_and_b32 s6, s12, 31 // (op 626) 
s_lshl_b32 s18, 1, s6 // (op 626) 
s_add_i32 s19, s18, s7 // (op 626) 


s_and_b32 s7, s1, s19 // (op 626) 


s_add_i32 s1, s51, s7 // (op 626) 


s_ashr_i32 s7, s1, s6 // (op 626) 


s_lshl_b32 s1, s12, 24 // (op 626) 
s_ashr_i32 s6, s1, 31 // (op 626) 

s_xor_b32 s1, s7, s6 // (op 626) 

s_sub_i32 s7, s1, s6 // (op 626) 


s_mul_i32 s1, s0, s14 // (op 626) 

s_sub_u32 s0, s16, s1 // (op 626) 

s_mul_i32 s1, s7, s14 // (op 626) 
s_add_u32 s1, s1, s0 // (op 626) 


s_mul_hi_i32 s0, s16, s8 // (op 626) 
s_add_i32 s51, s0, s16 // (op 626) 

s_ashr_i32 s0, s51, 31 // (op 626) 
s_cmp_eq_i32 s8, 0 // (op 626) 
s_mov_b32 s6, scc // (op 626) 
s_cmp_eq_i32 s6, 0 // (op 626) 
s_cselect_b32 s7, 0, -1 // (op 626) 

s_and_b32 s6, s15, 31 // (op 626) 
s_lshl_b32 s18, 1, s6 // (op 626) 
s_add_i32 s19, s18, s7 // (op 626) 


s_and_b32 s7, s0, s19 // (op 626) 


s_add_i32 s0, s51, s7 // (op 626) 


s_ashr_i32 s7, s0, s6 // (op 626) 


s_lshl_b32 s0, s15, 24 // (op 626) 
s_ashr_i32 s6, s0, 31 // (op 626) 

s_xor_b32 s0, s7, s6 // (op 626) 

s_sub_i32 s7, s0, s6 // (op 626) 


s_add_u32 s18, s4, 16 // (op 626) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 626) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 626) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 626) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 626) 

s_mul_i32 s0, s14, s18 // (op 626) 

s_mul_i32 s6, s50, s0 // (op 626) 

s_mul_i32 s0, s7, s6 // (op 626) 

s_sub_u32 s6, s16, s0 // (op 626) 

s_mov_b32 s0, s6 // (op 626) 
s_mul_hi_i32 s51, s0, s13 // (op 626) 
s_add_i32 s18, s51, s0 // (op 626) 


s_ashr_i32 s0, s18, 31 // (op 626) 
s_cmp_eq_i32 s13, 0 // (op 626) 
s_mov_b32 s19, scc // (op 626) 
s_cmp_eq_i32 s19, 0 // (op 626) 
s_cselect_b32 s51, 0, -1 // (op 626) 

s_and_b32 s19, s49, 31 // (op 626) 
s_lshl_b32 s64, 1, s19 // (op 626) 
s_add_i32 s65, s64, s51 // (op 626) 


s_and_b32 s51, s0, s65 // (op 626) 


s_add_i32 s0, s18, s51 // (op 626) 


s_ashr_i32 s18, s0, s19 // (op 626) 


s_lshl_b32 s0, s49, 24 // (op 626) 
s_ashr_i32 s19, s0, 31 // (op 626) 

s_xor_b32 s0, s18, s19 // (op 626) 

s_sub_i32 s18, s0, s19 // (op 626) 


s_mul_i32 s0, s48, s14 // (op 626) 
s_add_u32 s64, s20, 16 // (op 626) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 626) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 626) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 626) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 626) 

s_sub_i32 s19, s64, s0 // (op 626) 

s_mul_i32 s51, s18, s19 // (op 626) 


s_sub_u32 s18, s6, s51 // (op 626) 


s_add_u32 s6, s18, s0 // (op 626) 


s_cmp_eq_i32 s7, 0 // (op 626) 
s_mov_b32 s0, scc // (op 626) 

s_cmp_eq_i32 s0, 0 // (op 626) 
s_cselect_b32 s7, s6, s1 // (op 626) 



v_lshl_add_u32 v19, s7, 4, v18 // (op 626) 


v_mul_lo_u32 v18, v17, s30 // (op 626) 

v_add_u32 v17, v19, v18 // (op 626) 


s_mul_i32 s1, s30, s35 // (op 628) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 628) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 628) least significant
s_mul_hi_u32 s6, s30, s34 // (op 628) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 628) most significant: sum
s_add_u32 s1, s1, s6 // (op 628) most significant: sum


s_mov_b32 s60, s32 // (op 628) reg expression
s_mov_b32 s61, s33 // (op 628) reg expression
s_mov_b32 s62, s0 // (op 628) reg expression
s_mov_b32 s63, 131072 // (op 628) call()

s_mul_hi_i32 s51, s16, s9 // (op 671) 
v_and_b32 v18, -16, v1 // (op 671) 
s_add_i32 s0, s51, s16 // (op 671) 

v_sub_u32 v19, v1, v18 // (op 671) 

s_ashr_i32 s1, s0, 31 // (op 671) 
v_lshrrev_b32 v18, 4, v1 // (op 671) 
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
s_sub_u32 s64, s18, 1 // (op 671) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 671) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 671) 

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
s_lshl_b32 s64, 1, s6 // (op 671) 
s_add_i32 s65, s64, s7 // (op 671) 


s_and_b32 s7, s0, s65 // (op 671) 


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
s_add_i32 s64, s19, s18 // (op 671) 


s_and_b32 s18, s0, s64 // (op 671) 


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



v_lshl_add_u32 v20, s1, 4, v19 // (op 671) 


v_mul_lo_u32 v19, v18, s36 // (op 671) 
v_add_u32 v19, v19, v20 // (op 671) 


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

s_mul_hi_i32 s51, s16, s9 // (op 690) 
v_and_b32 v18, -16, v1 // (op 690) 
s_add_i32 s0, s51, s16 // (op 690) 

v_sub_u32 v20, v1, v18 // (op 690) 

s_ashr_i32 s1, s0, 31 // (op 690) 
v_lshrrev_b32 v18, 4, v1 // (op 690) 
s_cmp_eq_i32 s9, 0 // (op 690) 
s_mov_b32 s51, scc // (op 690) 
s_cmp_eq_i32 s51, 0 // (op 690) 
s_cselect_b32 s6, 0, -1 // (op 690) 

s_and_b32 s7, s10, 31 // (op 690) 
s_lshl_b32 s51, 1, s7 // (op 690) 
s_add_i32 s18, s51, s6 // (op 690) 


s_and_b32 s6, s1, s18 // (op 690) 


s_add_i32 s1, s0, s6 // (op 690) 


s_ashr_i32 s0, s1, s7 // (op 690) 


s_lshl_b32 s1, s10, 24 // (op 690) 
s_ashr_i32 s51, s1, 31 // (op 690) 

s_xor_b32 s1, s0, s51 // (op 690) 

s_sub_i32 s0, s1, s51 // (op 690) 


s_mul_hi_i32 s1, s0, s11 // (op 690) 
s_add_i32 s51, s1, s0 // (op 690) 

s_ashr_i32 s1, s51, 31 // (op 690) 
s_cmp_eq_i32 s11, 0 // (op 690) 
s_mov_b32 s6, scc // (op 690) 
s_cmp_eq_i32 s6, 0 // (op 690) 
s_cselect_b32 s7, 0, -1 // (op 690) 

s_and_b32 s6, s12, 31 // (op 690) 
s_lshl_b32 s18, 1, s6 // (op 690) 
s_add_i32 s19, s18, s7 // (op 690) 


s_and_b32 s7, s1, s19 // (op 690) 


s_add_i32 s1, s51, s7 // (op 690) 


s_ashr_i32 s7, s1, s6 // (op 690) 


s_lshl_b32 s1, s12, 24 // (op 690) 
s_ashr_i32 s6, s1, 31 // (op 690) 

s_xor_b32 s1, s7, s6 // (op 690) 

s_sub_i32 s7, s1, s6 // (op 690) 


s_add_u32 s18, s4, 16 // (op 690) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 690) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 690) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 690) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 690) 

s_mul_i32 s1, s7, s18 // (op 690) 

s_sub_i32 s51, s0, s1 // (op 690) 


s_mul_hi_i32 s0, s16, s8 // (op 690) 
s_add_i32 s1, s0, s16 // (op 690) 

s_ashr_i32 s0, s1, 31 // (op 690) 
s_cmp_eq_i32 s8, 0 // (op 690) 
s_mov_b32 s6, scc // (op 690) 
s_cmp_eq_i32 s6, 0 // (op 690) 
s_cselect_b32 s7, 0, -1 // (op 690) 

s_and_b32 s6, s15, 31 // (op 690) 
s_lshl_b32 s68, 1, s6 // (op 690) 
s_add_i32 s69, s68, s7 // (op 690) 


s_and_b32 s7, s0, s69 // (op 690) 


s_add_i32 s0, s1, s7 // (op 690) 


s_ashr_i32 s1, s0, s6 // (op 690) 


s_lshl_b32 s0, s15, 24 // (op 690) 
s_ashr_i32 s6, s0, 31 // (op 690) 

s_xor_b32 s0, s1, s6 // (op 690) 

s_sub_i32 s1, s0, s6 // (op 690) 


s_mul_i32 s0, s14, s18 // (op 690) 

s_mul_i32 s6, s50, s0 // (op 690) 

s_mul_i32 s0, s1, s6 // (op 690) 

s_sub_u32 s6, s16, s0 // (op 690) 

s_mul_hi_i32 s0, s6, s13 // (op 690) 
s_add_i32 s7, s0, s6 // (op 690) 


s_ashr_i32 s0, s7, 31 // (op 690) 
s_cmp_eq_i32 s13, 0 // (op 690) 
s_mov_b32 s6, scc // (op 690) 
s_cmp_eq_i32 s6, 0 // (op 690) 
s_cselect_b32 s18, 0, -1 // (op 690) 

s_and_b32 s6, s49, 31 // (op 690) 
s_lshl_b32 s19, 1, s6 // (op 690) 
s_add_i32 s68, s19, s18 // (op 690) 


s_and_b32 s18, s0, s68 // (op 690) 


s_add_i32 s0, s7, s18 // (op 690) 


s_ashr_i32 s7, s0, s6 // (op 690) 


s_lshl_b32 s0, s49, 24 // (op 690) 
s_ashr_i32 s6, s0, 31 // (op 690) 

s_xor_b32 s0, s7, s6 // (op 690) 

s_sub_i32 s7, s0, s6 // (op 690) 


s_cmp_eq_i32 s1, 0 // (op 690) 
s_mov_b32 s0, scc // (op 690) 

s_cmp_eq_i32 s0, 0 // (op 690) 
s_cselect_b32 s1, s7, s51 // (op 690) 



v_lshl_add_u32 v21, s1, 4, v20 // (op 690) 


v_mul_lo_u32 v20, v18, s36 // (op 690) 
v_add_u32 v20, v20, v21 // (op 690) 


s_mul_i32 s1, s36, s35 // (op 692) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 692) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 692) least significant
s_mul_hi_u32 s6, s36, s34 // (op 692) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 692) most significant: sum
s_add_u32 s1, s1, s6 // (op 692) most significant: sum


s_mov_b32 s64, s38 // (op 692) reg expression
s_mov_b32 s65, s39 // (op 692) reg expression
s_mov_b32 s66, s0 // (op 692) reg expression
s_mov_b32 s67, 131072 // (op 692) call()


s_mul_i32 s1, s27, 6 // (op 370) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 370) least significant
s_mul_hi_u32 s51, s26, 6 // (op 370) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 370) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 370) 


v_mov_b32 v18, v12 // (op 370) reg expression
buffer_load_dwordx3 v[24:26], v18, s[56:59], 0 offen // (op 370) Load value
v_add_u32 v18, v18, v22 // (op 370) 
buffer_load_dwordx3 v[28:30], v18, s[56:59], 0 offen // (op 370) Load value




v_mov_b32 v18, v19 // (op 373) reg expression

buffer_load_ubyte v27, v18, s[64:67], 0 offen offset: 0 // (op 373) Load value


s_lshl_b64 s[0:1], s[2:3], 2 // (op 376) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 376) 


v_mov_b32 v18, v14 // (op 376) reg expression
buffer_load_dwordx2 v[32:33], v18, s[52:55], 0 offen // (op 376) Load value
v_add_u32 v18, v18, v22 // (op 376) 
buffer_load_dwordx2 v[34:35], v18, s[52:55], 0 offen // (op 376) Load value




v_mov_b32 v18, v16 // (op 379) reg expression

buffer_load_ubyte v31, v18, s[60:63], 0 offen offset: 0 // (op 379) Load value


s_mul_i32 s1, s27, 6 // (op 382) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 382) least significant
s_mul_hi_u32 s51, s26, 6 // (op 382) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 382) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 382) 


s_lshl_b64 s[0:1], s[26:27], 7 // (op 382) 
s_mul_i32 s7, s1, 6 // (op 382) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 382) least significant
s_mul_hi_u32 s51, s0, 6 // (op 382) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 382) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 382) 

v_add_u32 v18, v12, s0 // (op 382) 

buffer_load_dwordx3 v[36:38], v18, s[56:59], 0 offen // (op 382) Load value
v_add_u32 v18, v18, v22 // (op 382) 
buffer_load_dwordx3 v[40:42], v18, s[56:59], 0 offen // (op 382) Load value





s_lshl_b32 s51, s36, 2 // (op 385) 
v_add_u32 v12, v19, s51 // (op 385) 


buffer_load_ubyte v18, v12, s[64:67], 0 offen offset: 0 // (op 385) Load value



s_lshl_b64 s[0:1], s[2:3], 2 // (op 388) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 388) 


s_lshl_b64 s[0:1], s[2:3], 7 // (op 388) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 388) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 388) 

v_add_u32 v12, v14, s0 // (op 388) 

buffer_load_dwordx2 v[44:45], v12, s[52:55], 0 offen // (op 388) Load value
v_add_u32 v12, v12, v22 // (op 388) 
buffer_load_dwordx2 v[46:47], v12, s[52:55], 0 offen // (op 388) Load value





s_lshl_b32 s51, s30, 2 // (op 391) 
v_add_u32 v12, v16, s51 // (op 391) 


buffer_load_ubyte v14, v12, s[60:63], 0 offen offset: 0 // (op 391) Load value


v_lshlrev_b32 v12, 5, v1 // (op 810) 
v_mul_lo_u32 v16, v12, 6 // (op 810) 
v_lshrrev_b32 v19, 3, v16 // (op 810) 

v_lshrrev_b32 v16, 4, v12 // (op 810) 

v_lshlrev_b32 v12, 2, v16 // (op 810) 

v_add_u32 v16, v19, v12 // (op 810) 





s_waitcnt vmcnt(11)
ds_write_b96 v16, v[24:26] // (op 393) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v16, v[28:30] offset:16 // (op 393) Store local data 

v_and_b32 v12, -16, v1 // (op 916) 
v_sub_u32 v16, v1, v12 // (op 916) 

v_and_b32 v12, 4294967280, v1 // (op 916) 
v_add_u32 v19, v12, v16 // (op 916) 





s_waitcnt vmcnt(9)
ds_write_b8 v19, v27 offset:2048 // (op 394) Store local data 

v_lshlrev_b32 v12, 7, v1 // (op 757) 
v_lshrrev_b32 v16, 3, v12 // (op 757) 




s_waitcnt vmcnt(8)
ds_write_b64 v16, v[32:33] offset:2112 // (op 395) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v16, v[34:35] offset:2120 // (op 395) Store local data 

v_and_b32 v12, -16, v1 // (op 863) 
v_sub_u32 v16, v1, v12 // (op 863) 

v_and_b32 v12, 4294967280, v1 // (op 863) 
v_add_u32 v19, v12, v16 // (op 863) 





s_waitcnt vmcnt(6)
ds_write_b8 v19, v31 offset:3136 // (op 396) Store local data 

v_lshlrev_b32 v12, 7, v1 // (op 728) 
v_lshrrev_b32 v16, 3, v12 // (op 728) 

v_lshlrev_b32 v12, 5, v1 // (op 781) 
v_mul_lo_u32 v19, v12, 6 // (op 781) 
v_lshrrev_b32 v39, 3, v19 // (op 781) 

v_lshrrev_b32 v19, 4, v12 // (op 781) 

v_lshlrev_b32 v12, 2, v19 // (op 781) 

v_add_u32 v19, v39, v12 // (op 781) 


v_and_b32 v12, -16, v1 // (op 834) 
v_sub_u32 v39, v1, v12 // (op 834) 

v_and_b32 v12, 4294967280, v1 // (op 834) 
v_add_u32 v43, v12, v39 // (op 834) 


v_and_b32 v12, -16, v1 // (op 887) 
v_sub_u32 v39, v1, v12 // (op 887) 

v_and_b32 v12, 4294967280, v1 // (op 887) 
v_add_u32 v21, v12, v39 // (op 887) 


v_lshlrev_b32 v12, 7, v1 // (op 974) 
v_lshrrev_b32 v39, 3, v12 // (op 974) 

v_lshlrev_b32 v12, 5, v1 // (op 998) 
v_mul_lo_u32 v22, v12, 6 // (op 998) 
v_lshrrev_b32 v23, 3, v22 // (op 998) 

v_lshrrev_b32 v22, 4, v12 // (op 998) 

v_lshlrev_b32 v12, 2, v22 // (op 998) 

v_add_u32 v22, v23, v12 // (op 998) 


v_and_b32 v12, -16, v1 // (op 1022) 
v_sub_u32 v23, v1, v12 // (op 1022) 

v_and_b32 v12, 4294967280, v1 // (op 1022) 
v_add_u32 v48, v12, v23 // (op 1022) 


v_and_b32 v12, -16, v1 // (op 1046) 
v_sub_u32 v23, v1, v12 // (op 1046) 

v_and_b32 v12, 4294967280, v1 // (op 1046) 
v_add_u32 v49, v12, v23 // (op 1046) 


s_mov_b32 s51, 0 // (op 61) call()
s_ashr_i64 s[0:1], s[24:25], 7 // (op 60) 
s_mov_b32 s6, s0 // (op 60) convert

s_and_b32 s7, s6, 4294967294 // (op 60) 

s_cmp_lt_u32 s51, s7 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_60 if false)
RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_60:

s_add_u32 s51, s51, 2 // (op 62) 
s_mul_i32 s1, s27, 6 // (op 10) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 10) least significant
s_mul_hi_u32 s6, s26, 6 // (op 10) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 10) most significant: sum

v_lshrrev_b64 v[50:51], 3, s[0:1] // (op 10) 


v_mov_b32 v12, v13 // (op 10) 
s_lshl_b64 s[0:1], s[26:27], 7 // (op 10) 
s_mul_i32 s7, s1, 6 // (op 10) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 10) least significant
s_mul_hi_u32 s18, s0, 6 // (op 10) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 10) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 10) 

s_lshl_b32 s6, s0, 1 // (op 10) 

v_add_u32 v23, v12, s6 // (op 10) 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 1095) 
buffer_load_dwordx3 v[24:26], v23, s[56:59], 0 offen // (op 10) Load value



v_add_u32 v23, v23, v50 // (op 10) 
buffer_load_dwordx3 v[28:30], v23, s[56:59], 0 offen // (op 10) Load value


ds_read_b64_tr_b4 v[50:51], v3 offset:2112 // (op 72) Transpose load local data 


ds_read_b64_tr_b4 v[52:53], v3 offset:2240 // (op 72) Transpose load local data 
v_mov_b32 v23, v20 // (op 25) 
s_barrier  // (op 1108) 
s_lshl_b32 s0, s36, 3 // (op 25) 



v_add_u32 v54, v23, s0 // (op 25) 


buffer_load_ubyte v27, v54, s[64:67], 0 offen offset: 0 // (op 25) Load value

ds_read_b96_tr_b6 v[54:56], v5 // (op 77) Transpose load local data 

s_lshl_b64 s[0:1], s[2:3], 2 // (op 4) 
v_lshrrev_b64 v[60:61], 3, s[0:1] // (op 4) 


v_mov_b32 v62, v15 // (op 4) 
s_lshl_b64 s[0:1], s[2:3], 7 // (op 4) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 4) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 4) 

s_lshl_b32 s6, s0, 1 // (op 4) 

v_add_u32 v63, v62, s6 // (op 4) 

ds_read_b96_tr_b6 v[57:59], v5 offset:256 // (op 77) Transpose load local data 
buffer_load_dwordx2 v[32:33], v63, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v63, v63, v60 // (op 4) 
buffer_load_dwordx2 v[34:35], v63, s[52:55], 0 offen // (op 4) Load value


s_waitcnt lgkmcnt(4)
s_barrier  // (op 1122) 




ds_read_u8 v60, v4 offset:3136 // (op 82) Load local data 
v_mov_b32 v61, v17 // (op 17) 
s_barrier  // (op 1136) 
s_lshl_b32 s0, s30, 3 // (op 17) 


ds_read_u8 v63, v7 offset:2048 // (op 87) Load local data 
v_add_u32 v64, v61, s0 // (op 17) 


buffer_load_ubyte v31, v64, s[60:63], 0 offen offset: 0 // (op 17) Load value

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:53], v[54:59], a[0:3], v60, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 90) 




s_barrier  // (op 1159) 



s_waitcnt vmcnt(11)
ds_write_b96 v22, v[36:38] offset:3200 // (op 257) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v22, v[40:42] offset:3216 // (op 257) Store local data 
s_barrier  // (op 1171) 



s_waitcnt vmcnt(9)
ds_write_b8 v49, v18 offset:5248 // (op 259) Store local data 
s_barrier  // (op 1153) 



s_waitcnt vmcnt(8)
ds_write_b64 v39, v[44:45] offset:5312 // (op 247) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v39, v[46:47] offset:5320 // (op 247) Store local data 
s_barrier  // (op 1165) 



s_waitcnt vmcnt(6)
ds_write_b8 v48, v14 offset:6336 // (op 258) Store local data 

s_mul_i32 s1, s27, 6 // (op 260) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 260) least significant
s_mul_hi_u32 s6, s26, 6 // (op 260) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 260) most significant: sum

v_lshrrev_b64 v[50:51], 3, s[0:1] // (op 260) 


s_lshl_b64 s[0:1], s[26:27], 7 // (op 260) 
s_mul_i32 s7, s1, 6 // (op 260) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 260) least significant
s_mul_hi_u32 s18, s0, 6 // (op 260) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 260) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 260) 

s_mul_i32 s6, 3, s0 // (op 260) 

v_add_u32 v52, v12, s6 // (op 260) 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 1150) 
buffer_load_dwordx3 v[36:38], v52, s[56:59], 0 offen // (op 260) Load value



v_add_u32 v52, v52, v50 // (op 260) 
buffer_load_dwordx3 v[40:42], v52, s[56:59], 0 offen // (op 260) Load value


ds_read_b64_tr_b4 v[50:51], v6 offset:5312 // (op 249) Transpose load local data 

ds_read_b64_tr_b4 v[52:53], v6 offset:5440 // (op 249) Transpose load local data 


s_lshl_b32 s0, s36, 2 // (op 262) 
s_barrier  // (op 1156) 
s_barrier  // (op 1230) 



s_mul_i32 s1, 3, s0 // (op 262) 

v_add_u32 v12, v23, s1 // (op 262) 


s_lshl_b64 s[0:1], s[26:27], 7 // (op 554) 
buffer_load_ubyte v18, v12, s[64:67], 0 offen offset: 0 // (op 262) Load value

s_mul_i32 s7, s1, 6 // (op 554) most significant: low of high * low

s_mul_i32 s6, s0, 6 // (op 554) least significant

s_mul_hi_u32 s18, s0, 6 // (op 554) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 554) most significant: sum


s_lshl_b32 s0, s36, 3 // (op 682) 
v_add_u32 v20, v20, s0 // (op 682) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 554) 

s_lshl_b32 s6, s0, 1 // (op 554) 

v_add_u32 v13, v13, s6 // (op 554) 

s_lshl_b64 s[0:1], s[2:3], 2 // (op 246) 
v_lshrrev_b64 v[54:55], 3, s[0:1] // (op 246) 


s_lshl_b64 s[0:1], s[2:3], 7 // (op 246) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 246) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 246) 

s_mul_i32 s6, 3, s0 // (op 246) 

v_add_u32 v12, v62, s6 // (op 246) 

ds_read_b96_tr_b6 v[64:66], v9 offset:3200 // (op 251) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v9 offset:3456 // (op 251) Transpose load local data 
buffer_load_dwordx2 v[44:45], v12, s[52:55], 0 offen // (op 246) Load value
v_add_u32 v12, v12, v54 // (op 246) 
s_barrier  // (op 1236) 
buffer_load_dwordx2 v[46:47], v12, s[52:55], 0 offen // (op 246) Load value


s_waitcnt lgkmcnt(4)
s_barrier  // (op 1162) 





ds_read_u8 v12, v8 offset:6336 // (op 253) Load local data 
s_lshl_b32 s0, s30, 2 // (op 261) 
s_mul_i32 s1, 3, s0 // (op 261) 

s_barrier  // (op 1168) 
s_barrier  // (op 1242) 


ds_read_u8 v23, v11 offset:5248 // (op 255) Load local data 
v_add_u32 v54, v61, s1 // (op 261) 


buffer_load_ubyte v14, v54, s[60:63], 0 offen offset: 0 // (op 261) Load value


s_barrier  // (op 1248) 
s_barrier  // (op 1111) 
s_lshl_b32 s0, s30, 3 // (op 618) 



v_add_u32 v17, v17, s0 // (op 618) 

s_lshl_b64 s[0:1], s[2:3], 7 // (op 490) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 490) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 490) 

s_lshl_b32 s6, s0, 1 // (op 490) 

v_add_u32 v15, v15, s6 // (op 490) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:53], v[64:69], a[0:3], v12, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 256) 
s_waitcnt vmcnt(11)
ds_write_b96 v19, v[24:26] // (op 13) Store local data 




s_waitcnt vmcnt(10)
ds_write_b96 v19, v[28:30] offset:16 // (op 13) Store local data 
s_barrier  // (op 1139) 



s_waitcnt vmcnt(9)
ds_write_b8 v21, v27 offset:2048 // (op 30) Store local data 
s_barrier  // (op 1098) 



s_waitcnt vmcnt(8)
ds_write_b64 v16, v[32:33] offset:2112 // (op 6) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v16, v[34:35] offset:2120 // (op 6) Store local data 
s_barrier  // (op 1125) 



s_waitcnt vmcnt(6)
ds_write_b8 v43, v31 offset:3136 // (op 21) Store local data 
s_ashr_i64 s[0:1], s[24:25], 7 // (op 60) 
s_mov_b32 s6, s0 // (op 60) convert

s_and_b32 s7, s6, 4294967294 // (op 60) 

s_cmp_lt_u32 s51, s7 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_60 if true)
RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_60:

















s_mul_hi_i32 s51, s16, s9 // (op 453) 
v_lshlrev_b32 v39, 1, v1 // (op 453) 
s_add_i32 s0, s51, s16 // (op 453) 

s_ashr_i32 s1, s0, 31 // (op 453) 
s_cmp_eq_i32 s9, 0 // (op 453) 
s_mov_b32 s51, scc // (op 453) 
s_cmp_eq_i32 s51, 0 // (op 453) 
s_cselect_b32 s6, 0, -1 // (op 453) 

s_and_b32 s7, s10, 31 // (op 453) 
s_lshl_b32 s51, 1, s7 // (op 453) 
s_add_i32 s18, s51, s6 // (op 453) 


s_and_b32 s6, s1, s18 // (op 453) 


s_add_i32 s1, s0, s6 // (op 453) 


s_ashr_i32 s0, s1, s7 // (op 453) 


s_lshl_b32 s1, s10, 24 // (op 453) 
s_ashr_i32 s51, s1, 31 // (op 453) 

s_xor_b32 s1, s0, s51 // (op 453) 

s_sub_i32 s0, s1, s51 // (op 453) 


s_mul_hi_i32 s1, s0, s11 // (op 453) 
s_add_i32 s51, s1, s0 // (op 453) 

s_ashr_i32 s1, s51, 31 // (op 453) 
s_cmp_eq_i32 s11, 0 // (op 453) 
s_mov_b32 s6, scc // (op 453) 
s_cmp_eq_i32 s6, 0 // (op 453) 
s_cselect_b32 s7, 0, -1 // (op 453) 

s_and_b32 s6, s12, 31 // (op 453) 
s_lshl_b32 s18, 1, s6 // (op 453) 
s_add_i32 s19, s18, s7 // (op 453) 


s_and_b32 s7, s1, s19 // (op 453) 


s_add_i32 s1, s51, s7 // (op 453) 


s_ashr_i32 s7, s1, s6 // (op 453) 


s_lshl_b32 s1, s12, 24 // (op 453) 
s_ashr_i32 s6, s1, 31 // (op 453) 

s_xor_b32 s1, s7, s6 // (op 453) 

s_sub_i32 s7, s1, s6 // (op 453) 


s_mul_i32 s1, s0, s14 // (op 453) 

s_sub_u32 s0, s16, s1 // (op 453) 

s_mul_i32 s1, s7, s14 // (op 453) 
s_add_u32 s1, s1, s0 // (op 453) 


s_mul_hi_i32 s0, s16, s8 // (op 453) 
s_add_i32 s51, s0, s16 // (op 453) 

s_ashr_i32 s0, s51, 31 // (op 453) 
s_cmp_eq_i32 s8, 0 // (op 453) 
s_mov_b32 s6, scc // (op 453) 
s_cmp_eq_i32 s6, 0 // (op 453) 
s_cselect_b32 s7, 0, -1 // (op 453) 

s_and_b32 s6, s15, 31 // (op 453) 
s_lshl_b32 s18, 1, s6 // (op 453) 
s_add_i32 s19, s18, s7 // (op 453) 


s_and_b32 s7, s0, s19 // (op 453) 


s_add_i32 s0, s51, s7 // (op 453) 


s_ashr_i32 s7, s0, s6 // (op 453) 


s_lshl_b32 s0, s15, 24 // (op 453) 
s_ashr_i32 s6, s0, 31 // (op 453) 

s_xor_b32 s0, s7, s6 // (op 453) 

s_sub_i32 s7, s0, s6 // (op 453) 


s_add_u32 s18, s4, 16 // (op 453) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 453) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 453) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 453) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 453) 

s_mul_i32 s0, s14, s18 // (op 453) 

s_mul_i32 s6, s50, s0 // (op 453) 

s_mul_i32 s0, s7, s6 // (op 453) 

s_sub_u32 s6, s16, s0 // (op 453) 

s_mov_b32 s0, s6 // (op 453) 
s_mul_hi_i32 s51, s0, s13 // (op 453) 
s_add_i32 s18, s51, s0 // (op 453) 


s_ashr_i32 s0, s18, 31 // (op 453) 
s_cmp_eq_i32 s13, 0 // (op 453) 
s_mov_b32 s19, scc // (op 453) 
s_cmp_eq_i32 s19, 0 // (op 453) 
s_cselect_b32 s51, 0, -1 // (op 453) 

s_and_b32 s19, s49, 31 // (op 453) 
s_lshl_b32 s68, 1, s19 // (op 453) 
s_add_i32 s69, s68, s51 // (op 453) 


s_and_b32 s51, s0, s69 // (op 453) 


s_add_i32 s0, s18, s51 // (op 453) 


s_ashr_i32 s18, s0, s19 // (op 453) 


s_lshl_b32 s0, s49, 24 // (op 453) 
s_ashr_i32 s19, s0, 31 // (op 453) 

s_xor_b32 s0, s18, s19 // (op 453) 

s_sub_i32 s18, s0, s19 // (op 453) 


s_mul_i32 s0, s48, s14 // (op 453) 
s_add_u32 s68, s20, 16 // (op 453) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 453) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 453) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 453) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 453) 

s_sub_i32 s19, s68, s0 // (op 453) 

s_mul_i32 s51, s18, s19 // (op 453) 


s_sub_u32 s18, s6, s51 // (op 453) 


s_add_u32 s6, s18, s0 // (op 453) 


s_cmp_eq_i32 s7, 0 // (op 453) 
s_mov_b32 s0, scc // (op 453) 

s_cmp_eq_i32 s0, 0 // (op 453) 
s_cselect_b32 s7, s6, s1 // (op 453) 



s_lshl_b32 s6, s7, 4 // (op 453) 

s_mov_b32 s1, 0 // (op 453) convert
s_mov_b32 s0, s6 // (op 453) convert

s_ashr_i64 s[6:7], s[24:25], 7 // (op 453) 
s_lshr_b32 s51, s6, 1 // (op 453) 

v_lshl_add_u32 v43, s51, 8, v39 // (op 453) 


v_mul_lo_u32 v13, v43, s3 // (op 453) most significant: low of low * high
v_mul_hi_u32 v39, v43, s2 // (op 453) most significant: high of low * low
v_mul_lo_u32 v12, v43, s2 // (op 453) least significant: low of low * low
v_add_u32 v13, v13, v39 // (op 453) most significant: sum


v_mov_b32 v39, s0 // (op 453) 
v_mov_b32 v43, s1 // (op 453) 
v_add_co_u32 v16, s[6:7], v39, v12 // (op 453) least significant half
v_addc_co_u32 v17, s[6:7], v43, v13, s[6:7] // (op 453) most significant half



v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 453) 


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 453) 

v_mov_b32 v39, v12 // (op 453) convert

s_mul_i32 s1, s2, s25 // (op 455) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 455) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 455) least significant
s_mul_hi_u32 s6, s2, s24 // (op 455) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 455) most significant: sum
s_add_u32 s1, s1, s6 // (op 455) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 455) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 455) 

s_mov_b32 s52, s22 // (op 455) reg expression
s_mov_b32 s53, s23 // (op 455) reg expression
s_mov_b32 s54, s0 // (op 455) reg expression
s_mov_b32 s55, 131072 // (op 455) call()

s_mul_hi_i32 s51, s16, s9 // (op 520) 
v_lshlrev_b32 v43, 1, v1 // (op 520) 
s_add_i32 s0, s51, s16 // (op 520) 

s_ashr_i32 s1, s0, 31 // (op 520) 
s_cmp_eq_i32 s9, 0 // (op 520) 
s_mov_b32 s51, scc // (op 520) 
s_cmp_eq_i32 s51, 0 // (op 520) 
s_cselect_b32 s6, 0, -1 // (op 520) 

s_and_b32 s7, s10, 31 // (op 520) 
s_lshl_b32 s51, 1, s7 // (op 520) 
s_add_i32 s18, s51, s6 // (op 520) 


s_and_b32 s6, s1, s18 // (op 520) 


s_add_i32 s1, s0, s6 // (op 520) 


s_ashr_i32 s0, s1, s7 // (op 520) 


s_lshl_b32 s1, s10, 24 // (op 520) 
s_ashr_i32 s51, s1, 31 // (op 520) 

s_xor_b32 s1, s0, s51 // (op 520) 

s_sub_i32 s0, s1, s51 // (op 520) 


s_mul_hi_i32 s1, s0, s11 // (op 520) 
s_add_i32 s51, s1, s0 // (op 520) 

s_ashr_i32 s1, s51, 31 // (op 520) 
s_cmp_eq_i32 s11, 0 // (op 520) 
s_mov_b32 s6, scc // (op 520) 
s_cmp_eq_i32 s6, 0 // (op 520) 
s_cselect_b32 s7, 0, -1 // (op 520) 

s_and_b32 s6, s12, 31 // (op 520) 
s_lshl_b32 s18, 1, s6 // (op 520) 
s_add_i32 s19, s18, s7 // (op 520) 


s_and_b32 s7, s1, s19 // (op 520) 


s_add_i32 s1, s51, s7 // (op 520) 


s_ashr_i32 s7, s1, s6 // (op 520) 


s_lshl_b32 s1, s12, 24 // (op 520) 
s_ashr_i32 s6, s1, 31 // (op 520) 

s_xor_b32 s1, s7, s6 // (op 520) 

s_sub_i32 s7, s1, s6 // (op 520) 


s_add_u32 s18, s4, 16 // (op 520) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 520) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 520) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 520) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 520) 

s_mul_i32 s1, s7, s18 // (op 520) 

s_sub_i32 s51, s0, s1 // (op 520) 


s_mul_hi_i32 s0, s16, s8 // (op 520) 
s_add_i32 s1, s0, s16 // (op 520) 

s_ashr_i32 s0, s1, 31 // (op 520) 
s_cmp_eq_i32 s8, 0 // (op 520) 
s_mov_b32 s6, scc // (op 520) 
s_cmp_eq_i32 s6, 0 // (op 520) 
s_cselect_b32 s7, 0, -1 // (op 520) 

s_and_b32 s6, s15, 31 // (op 520) 
s_lshl_b32 s68, 1, s6 // (op 520) 
s_add_i32 s69, s68, s7 // (op 520) 


s_and_b32 s7, s0, s69 // (op 520) 


s_add_i32 s0, s1, s7 // (op 520) 


s_ashr_i32 s1, s0, s6 // (op 520) 


s_lshl_b32 s0, s15, 24 // (op 520) 
s_ashr_i32 s6, s0, 31 // (op 520) 

s_xor_b32 s0, s1, s6 // (op 520) 

s_sub_i32 s1, s0, s6 // (op 520) 


s_mul_i32 s0, s14, s18 // (op 520) 

s_mul_i32 s6, s50, s0 // (op 520) 

s_mul_i32 s0, s1, s6 // (op 520) 

s_sub_u32 s6, s16, s0 // (op 520) 

s_mul_hi_i32 s0, s6, s13 // (op 520) 
s_add_i32 s7, s0, s6 // (op 520) 


s_ashr_i32 s0, s7, 31 // (op 520) 
s_cmp_eq_i32 s13, 0 // (op 520) 
s_mov_b32 s6, scc // (op 520) 
s_cmp_eq_i32 s6, 0 // (op 520) 
s_cselect_b32 s18, 0, -1 // (op 520) 

s_and_b32 s6, s49, 31 // (op 520) 
s_lshl_b32 s19, 1, s6 // (op 520) 
s_add_i32 s68, s19, s18 // (op 520) 


s_and_b32 s18, s0, s68 // (op 520) 


s_add_i32 s0, s7, s18 // (op 520) 


s_ashr_i32 s7, s0, s6 // (op 520) 


s_lshl_b32 s0, s49, 24 // (op 520) 
s_ashr_i32 s6, s0, 31 // (op 520) 

s_xor_b32 s0, s7, s6 // (op 520) 

s_sub_i32 s7, s0, s6 // (op 520) 


s_cmp_eq_i32 s1, 0 // (op 520) 
s_mov_b32 s0, scc // (op 520) 

s_cmp_eq_i32 s0, 0 // (op 520) 
s_cselect_b32 s1, s7, s51 // (op 520) 



s_lshl_b32 s0, s1, 4 // (op 520) 

s_mov_b32 s7, 0 // (op 520) convert
s_mov_b32 s6, s0 // (op 520) convert

s_ashr_i64 s[0:1], s[24:25], 7 // (op 520) 
s_lshr_b32 s51, s0, 1 // (op 520) 

v_lshl_add_u32 v12, s51, 8, v43 // (op 520) 


v_mul_lo_u32 v17, v12, s27 // (op 520) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 520) most significant: high of low * low
v_mul_lo_u32 v16, v12, s26 // (op 520) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 520) most significant: sum

v_mov_b32 v13, s6 // (op 520) 
v_mov_b32 v15, s7 // (op 520) 
v_add_co_u32 v16, s[0:1], v16, v13 // (op 520) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 520) most significant half





v_mul_lo_u32 v13, v17, 6 // (op 520) most significant: low of high * low
v_mul_hi_u32 v15, v16, 6 // (op 520) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 520) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 520) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 520) 

v_mov_b32 v15, v16 // (op 520) convert

s_mul_i32 s1, s25, 6 // (op 522) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 522) least significant
s_mul_hi_u32 s51, s24, 6 // (op 522) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 522) most significant: sum

s_mul_i32 s7, s26, s1 // (op 522) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 522) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 522) least significant
s_mul_hi_u32 s18, s26, s0 // (op 522) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 522) most significant: sum
s_add_u32 s7, s7, s18 // (op 522) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 522) 

s_mov_b32 s56, s28 // (op 522) reg expression
s_mov_b32 s57, s29 // (op 522) reg expression
s_mov_b32 s58, s0 // (op 522) reg expression
s_mov_b32 s59, 131072 // (op 522) call()

s_mul_hi_i32 s51, s16, s9 // (op 584) 
v_and_b32 v43, -16, v1 // (op 584) 
s_add_i32 s0, s51, s16 // (op 584) 

v_sub_u32 v12, v1, v43 // (op 584) 

s_ashr_i32 s1, s0, 31 // (op 584) 
v_lshrrev_b32 v13, 4, v1 // (op 584) 
s_cmp_eq_i32 s9, 0 // (op 584) 
s_mov_b32 s51, scc // (op 584) 
s_cmp_eq_i32 s51, 0 // (op 584) 
s_cselect_b32 s6, 0, -1 // (op 584) 

s_and_b32 s7, s10, 31 // (op 584) 
s_lshl_b32 s51, 1, s7 // (op 584) 
s_add_i32 s18, s51, s6 // (op 584) 


s_and_b32 s6, s1, s18 // (op 584) 


s_add_i32 s1, s0, s6 // (op 584) 


s_ashr_i32 s0, s1, s7 // (op 584) 


s_lshl_b32 s1, s10, 24 // (op 584) 
s_ashr_i32 s51, s1, 31 // (op 584) 

s_xor_b32 s1, s0, s51 // (op 584) 

s_sub_i32 s0, s1, s51 // (op 584) 


s_mul_hi_i32 s1, s0, s11 // (op 584) 
s_add_i32 s51, s1, s0 // (op 584) 

s_ashr_i32 s1, s51, 31 // (op 584) 
s_cmp_eq_i32 s11, 0 // (op 584) 
s_mov_b32 s6, scc // (op 584) 
s_cmp_eq_i32 s6, 0 // (op 584) 
s_cselect_b32 s7, 0, -1 // (op 584) 

s_and_b32 s6, s12, 31 // (op 584) 
s_lshl_b32 s18, 1, s6 // (op 584) 
s_add_i32 s19, s18, s7 // (op 584) 


s_and_b32 s7, s1, s19 // (op 584) 


s_add_i32 s1, s51, s7 // (op 584) 


s_ashr_i32 s7, s1, s6 // (op 584) 


s_lshl_b32 s1, s12, 24 // (op 584) 
s_ashr_i32 s6, s1, 31 // (op 584) 

s_xor_b32 s1, s7, s6 // (op 584) 

s_sub_i32 s7, s1, s6 // (op 584) 


s_mul_i32 s1, s0, s14 // (op 584) 

s_sub_u32 s0, s16, s1 // (op 584) 

s_mul_i32 s1, s7, s14 // (op 584) 
s_add_u32 s1, s1, s0 // (op 584) 


s_mul_hi_i32 s0, s16, s8 // (op 584) 
s_add_i32 s51, s0, s16 // (op 584) 

s_ashr_i32 s0, s51, 31 // (op 584) 
s_cmp_eq_i32 s8, 0 // (op 584) 
s_mov_b32 s6, scc // (op 584) 
s_cmp_eq_i32 s6, 0 // (op 584) 
s_cselect_b32 s7, 0, -1 // (op 584) 

s_and_b32 s6, s15, 31 // (op 584) 
s_lshl_b32 s18, 1, s6 // (op 584) 
s_add_i32 s19, s18, s7 // (op 584) 


s_and_b32 s7, s0, s19 // (op 584) 


s_add_i32 s0, s51, s7 // (op 584) 


s_ashr_i32 s7, s0, s6 // (op 584) 


s_lshl_b32 s0, s15, 24 // (op 584) 
s_ashr_i32 s6, s0, 31 // (op 584) 

s_xor_b32 s0, s7, s6 // (op 584) 

s_sub_i32 s7, s0, s6 // (op 584) 


s_add_u32 s18, s4, 16 // (op 584) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 584) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 584) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 584) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 584) 

s_mul_i32 s0, s14, s18 // (op 584) 

s_mul_i32 s6, s50, s0 // (op 584) 

s_mul_i32 s0, s7, s6 // (op 584) 

s_sub_u32 s6, s16, s0 // (op 584) 

s_mov_b32 s0, s6 // (op 584) 
s_mul_hi_i32 s51, s0, s13 // (op 584) 
s_add_i32 s18, s51, s0 // (op 584) 


s_ashr_i32 s0, s18, 31 // (op 584) 
s_cmp_eq_i32 s13, 0 // (op 584) 
s_mov_b32 s19, scc // (op 584) 
s_cmp_eq_i32 s19, 0 // (op 584) 
s_cselect_b32 s51, 0, -1 // (op 584) 

s_and_b32 s19, s49, 31 // (op 584) 
s_lshl_b32 s68, 1, s19 // (op 584) 
s_add_i32 s69, s68, s51 // (op 584) 


s_and_b32 s51, s0, s69 // (op 584) 


s_add_i32 s0, s18, s51 // (op 584) 


s_ashr_i32 s18, s0, s19 // (op 584) 


s_lshl_b32 s0, s49, 24 // (op 584) 
s_ashr_i32 s19, s0, 31 // (op 584) 

s_xor_b32 s0, s18, s19 // (op 584) 

s_sub_i32 s18, s0, s19 // (op 584) 


s_mul_i32 s0, s48, s14 // (op 584) 
s_add_u32 s68, s20, 16 // (op 584) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 584) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 584) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 584) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 584) 

s_sub_i32 s19, s68, s0 // (op 584) 

s_mul_i32 s51, s18, s19 // (op 584) 


s_sub_u32 s18, s6, s51 // (op 584) 


s_add_u32 s6, s18, s0 // (op 584) 


s_cmp_eq_i32 s7, 0 // (op 584) 
s_mov_b32 s0, scc // (op 584) 

s_cmp_eq_i32 s0, 0 // (op 584) 
s_cselect_b32 s7, s6, s1 // (op 584) 



v_lshl_add_u32 v43, s7, 4, v12 // (op 584) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 584) 
s_lshr_b32 s51, s0, 1 // (op 584) 

v_lshl_add_u32 v12, s51, 3, v13 // (op 584) 


v_mul_lo_u32 v13, v12, s30 // (op 584) 

v_add_u32 v12, v43, v13 // (op 584) 


s_mul_i32 s1, s30, s35 // (op 586) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 586) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 586) least significant
s_mul_hi_u32 s6, s30, s34 // (op 586) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 586) most significant: sum
s_add_u32 s1, s1, s6 // (op 586) most significant: sum


s_mov_b32 s60, s32 // (op 586) reg expression
s_mov_b32 s61, s33 // (op 586) reg expression
s_mov_b32 s62, s0 // (op 586) reg expression
s_mov_b32 s63, 131072 // (op 586) call()

s_mul_hi_i32 s51, s16, s9 // (op 648) 
v_and_b32 v13, -16, v1 // (op 648) 
s_add_i32 s0, s51, s16 // (op 648) 

v_sub_u32 v43, v1, v13 // (op 648) 

s_ashr_i32 s1, s0, 31 // (op 648) 
v_lshrrev_b32 v13, 4, v1 // (op 648) 
s_cmp_eq_i32 s9, 0 // (op 648) 
s_mov_b32 s51, scc // (op 648) 
s_cmp_eq_i32 s51, 0 // (op 648) 
s_cselect_b32 s6, 0, -1 // (op 648) 

s_and_b32 s7, s10, 31 // (op 648) 
s_lshl_b32 s51, 1, s7 // (op 648) 
s_add_i32 s18, s51, s6 // (op 648) 


s_and_b32 s6, s1, s18 // (op 648) 


s_add_i32 s1, s0, s6 // (op 648) 


s_ashr_i32 s0, s1, s7 // (op 648) 


s_lshl_b32 s1, s10, 24 // (op 648) 
s_ashr_i32 s51, s1, 31 // (op 648) 

s_xor_b32 s1, s0, s51 // (op 648) 

s_sub_i32 s0, s1, s51 // (op 648) 


s_mul_hi_i32 s1, s0, s11 // (op 648) 
s_add_i32 s51, s1, s0 // (op 648) 

s_ashr_i32 s1, s51, 31 // (op 648) 
s_cmp_eq_i32 s11, 0 // (op 648) 
s_mov_b32 s6, scc // (op 648) 
s_cmp_eq_i32 s6, 0 // (op 648) 
s_cselect_b32 s7, 0, -1 // (op 648) 

s_and_b32 s6, s12, 31 // (op 648) 
s_lshl_b32 s18, 1, s6 // (op 648) 
s_add_i32 s19, s18, s7 // (op 648) 


s_and_b32 s7, s1, s19 // (op 648) 


s_add_i32 s1, s51, s7 // (op 648) 


s_ashr_i32 s7, s1, s6 // (op 648) 


s_lshl_b32 s1, s12, 24 // (op 648) 
s_ashr_i32 s6, s1, 31 // (op 648) 

s_xor_b32 s1, s7, s6 // (op 648) 

s_sub_i32 s7, s1, s6 // (op 648) 


s_add_u32 s18, s4, 16 // (op 648) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 648) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 648) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 648) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 648) 

s_mul_i32 s1, s7, s18 // (op 648) 

s_sub_i32 s51, s0, s1 // (op 648) 


s_mul_hi_i32 s0, s16, s8 // (op 648) 
s_add_i32 s1, s0, s16 // (op 648) 

s_ashr_i32 s0, s1, 31 // (op 648) 
s_cmp_eq_i32 s8, 0 // (op 648) 
s_mov_b32 s6, scc // (op 648) 
s_cmp_eq_i32 s6, 0 // (op 648) 
s_cselect_b32 s7, 0, -1 // (op 648) 

s_and_b32 s6, s15, 31 // (op 648) 
s_lshl_b32 s68, 1, s6 // (op 648) 
s_add_i32 s69, s68, s7 // (op 648) 


s_and_b32 s7, s0, s69 // (op 648) 


s_add_i32 s0, s1, s7 // (op 648) 


s_ashr_i32 s1, s0, s6 // (op 648) 


s_lshl_b32 s0, s15, 24 // (op 648) 
s_ashr_i32 s6, s0, 31 // (op 648) 

s_xor_b32 s0, s1, s6 // (op 648) 

s_sub_i32 s1, s0, s6 // (op 648) 


s_mul_i32 s0, s14, s18 // (op 648) 

s_mul_i32 s6, s50, s0 // (op 648) 

s_mul_i32 s0, s1, s6 // (op 648) 

s_sub_u32 s6, s16, s0 // (op 648) 

s_mul_hi_i32 s0, s6, s13 // (op 648) 
s_add_i32 s7, s0, s6 // (op 648) 


s_ashr_i32 s0, s7, 31 // (op 648) 
s_cmp_eq_i32 s13, 0 // (op 648) 
s_mov_b32 s6, scc // (op 648) 
s_cmp_eq_i32 s6, 0 // (op 648) 
s_cselect_b32 s18, 0, -1 // (op 648) 

s_and_b32 s6, s49, 31 // (op 648) 
s_lshl_b32 s19, 1, s6 // (op 648) 
s_add_i32 s68, s19, s18 // (op 648) 


s_and_b32 s18, s0, s68 // (op 648) 


s_add_i32 s0, s7, s18 // (op 648) 


s_ashr_i32 s7, s0, s6 // (op 648) 


s_lshl_b32 s0, s49, 24 // (op 648) 
s_ashr_i32 s6, s0, 31 // (op 648) 

s_xor_b32 s0, s7, s6 // (op 648) 

s_sub_i32 s7, s0, s6 // (op 648) 


s_cmp_eq_i32 s1, 0 // (op 648) 
s_mov_b32 s0, scc // (op 648) 

s_cmp_eq_i32 s0, 0 // (op 648) 
s_cselect_b32 s1, s7, s51 // (op 648) 



v_lshl_add_u32 v16, s1, 4, v43 // (op 648) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 648) 
s_lshr_b32 s51, s0, 1 // (op 648) 

v_lshl_add_u32 v17, s51, 3, v13 // (op 648) 


v_mul_lo_u32 v13, v17, s36 // (op 648) 
v_add_u32 v13, v13, v16 // (op 648) 


s_mul_i32 s1, s36, s35 // (op 650) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 650) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 650) least significant
s_mul_hi_u32 s6, s36, s34 // (op 650) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 650) most significant: sum
s_add_u32 s1, s1, s6 // (op 650) most significant: sum


s_mov_b32 s64, s38 // (op 650) reg expression
s_mov_b32 s65, s39 // (op 650) reg expression
s_mov_b32 s66, s0 // (op 650) reg expression
s_mov_b32 s67, 131072 // (op 650) call()

v_lshlrev_b32 v43, 7, v1 // (op 741) 
v_lshrrev_b32 v16, 3, v43 // (op 741) 

v_lshlrev_b32 v17, 5, v1 // (op 794) 
v_mul_lo_u32 v43, v17, 6 // (op 794) 
v_lshrrev_b32 v19, 3, v43 // (op 794) 

v_lshrrev_b32 v43, 4, v17 // (op 794) 

v_lshlrev_b32 v17, 2, v43 // (op 794) 

v_add_u32 v43, v19, v17 // (op 794) 


v_and_b32 v17, -16, v1 // (op 847) 
v_sub_u32 v19, v1, v17 // (op 847) 

v_and_b32 v17, 4294967280, v1 // (op 847) 
v_add_u32 v20, v17, v19 // (op 847) 


v_and_b32 v17, -16, v1 // (op 900) 
v_sub_u32 v19, v1, v17 // (op 900) 

v_and_b32 v17, 4294967280, v1 // (op 900) 
v_add_u32 v21, v17, v19 // (op 900) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 176) 
s_and_b32 s51, s0, 4294967294 // (op 176) 

s_ashr_i64 s[0:1], s[24:25], 7 // (op 175) 
s_mov_b32 s6, s0 // (op 175) convert

s_cmp_lt_u32 s51, s6 // (op 175) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_175 // (op 175) Condition: Top (jump to RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_175 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_175:

s_add_u32 s51, s51, 1 // (op 177) 
s_lshl_b64 s[0:1], s[2:3], 2 // (op 180) 
v_lshrrev_b64 v[18:19], 3, s[0:1] // (op 180) 


v_mov_b32 v14, v39 // (op 180) 
v_mov_b32 v17, v14 // (op 180) reg expression
buffer_load_dwordx2 v[22:23], v17, s[52:55], 0 offen // (op 180) Load value
v_add_u32 v17, v17, v18 // (op 180) 
buffer_load_dwordx2 v[24:25], v17, s[52:55], 0 offen // (op 180) Load value



s_lshl_b64 s[0:1], s[2:3], 7 // (op 443) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 443) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 443) 

v_add_u32 v39, v39, s0 // (op 443) 


s_barrier  // (op 1101) 
s_mul_i32 s1, s27, 6 // (op 194) most significant: low of high * low



s_mul_i32 s0, s26, 6 // (op 194) least significant
s_mul_hi_u32 s6, s26, 6 // (op 194) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 194) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[0:1] // (op 194) 


v_mov_b32 v14, v15 // (op 194) 
v_mov_b32 v17, v14 // (op 194) reg expression
buffer_load_dwordx3 v[26:28], v17, s[56:59], 0 offen // (op 194) Load value
v_add_u32 v17, v17, v18 // (op 194) 
s_waitcnt vmcnt(2)
ds_write_b64 v16, v[22:23] offset:2112 // (op 181) Store local data 
buffer_load_dwordx3 v[30:32], v17, s[56:59], 0 offen // (op 194) Load value



s_lshl_b64 s[0:1], s[26:27], 7 // (op 512) 
s_mul_i32 s7, s1, 6 // (op 512) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 512) least significant
s_mul_hi_u32 s18, s0, 6 // (op 512) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 512) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 512) 

v_add_u32 v15, v15, s0 // (op 512) 

s_waitcnt vmcnt(2)
ds_write_b64 v16, v[24:25] offset:2120 // (op 181) Store local data 



v_mov_b32 v14, v12 // (op 195) 
s_barrier  // (op 1114) 
v_mov_b32 v29, v14 // (op 195) reg expression




buffer_load_ubyte v17, v29, s[60:63], 0 offen offset: 0 // (op 195) Load value


s_lshl_b32 s0, s30, 2 // (op 576) 
v_add_u32 v12, v12, s0 // (op 576) 

s_waitcnt vmcnt(2)
ds_write_b96 v43, v[26:28] // (op 191) Store local data 


v_mov_b32 v14, v13 // (op 196) 
v_mov_b32 v29, v14 // (op 196) reg expression

buffer_load_ubyte v18, v29, s[64:67], 0 offen offset: 0 // (op 196) Load value


s_lshl_b32 s0, s36, 2 // (op 640) 
v_add_u32 v13, v13, s0 // (op 640) 

s_waitcnt vmcnt(2)
ds_write_b96 v43, v[30:32] offset:16 // (op 191) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1104) 



s_barrier  // (op 1128) 
ds_read_b64_tr_b4 v[22:23], v3 offset:2112 // (op 183) Transpose load local data 



ds_read_b64_tr_b4 v[24:25], v3 offset:2240 // (op 183) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v17 offset:3136 // (op 192) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1118) 



s_barrier  // (op 1142) 
ds_read_b96_tr_b6 v[26:28], v5 // (op 185) Transpose load local data 



ds_read_b96_tr_b6 v[29:31], v5 offset:256 // (op 185) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v21, v18 offset:2048 // (op 193) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1132) 


ds_read_u8 v14, v4 offset:3136 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1146) 


ds_read_u8 v17, v7 offset:2048 // (op 189) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:31], a[0:3], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 190) 




s_ashr_i64 s[0:1], s[24:25], 7 // (op 175) 
s_mov_b32 s6, s0 // (op 175) convert

s_cmp_lt_u32 s51, s6 // (op 175) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_175 // (op 175) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_175 if true)
RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_175:























s_barrier  // (op 1224) 








v_cmp_eq_f32 vcc, 0.00000, v0 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_135
s_nop 0
v_accvgpr_read v6, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v12, v10, v6 // (op 50) 
v_mul_f32 v13, v10, v7 // (op 50) 
v_mul_f32 v14, v10, v8 // (op 50) 
v_mul_f32 v15, v10, v9 // (op 50) 

v_lshrrev_b32 v6, 16, v12 // (op 56) 
v_lshrrev_b32 v7, 16, v13 // (op 56) 
v_lshrrev_b32 v8, 16, v14 // (op 56) 
v_lshrrev_b32 v9, 16, v15 // (op 56) 

s_mul_hi_i32 s0, s16, s9 // (op 942) 
v_lshrrev_b32 v3, 6, v1 // (op 942) 
s_add_i32 s1, s0, s16 // (op 942) 

v_and_b32 v4, 63, v1 // (op 942) 
s_ashr_i32 s0, s1, 31 // (op 942) 
v_lshrrev_b32 v5, 4, v4 // (op 942) 
s_cmp_eq_i32 s9, 0 // (op 942) 
s_mov_b32 s2, scc // (op 942) 
v_lshlrev_b32 v11, 2, v5 // (op 942) 

s_cmp_eq_i32 s2, 0 // (op 942) 
s_cselect_b32 s3, 0, -1 // (op 942) 

v_lshl_add_u32 v5, v3, 4, v11 // (op 942) 


s_and_b32 s2, s10, 31 // (op 942) 
v_lshrrev_b32 v3, 2, v4 // (op 942) 

s_lshl_b32 s6, 1, s2 // (op 942) 
v_and_b32 v4, 3, v3 // (op 942) 

s_add_i32 s7, s6, s3 // (op 942) 


v_and_b32 v3, 3, v1 // (op 942) 
s_and_b32 s3, s0, s7 // (op 942) 


v_lshl_add_u32 v11, v4, 2, v3 // (op 942) 


s_add_i32 s0, s1, s3 // (op 942) 


s_ashr_i32 s1, s0, s2 // (op 942) 


s_lshl_b32 s0, s10, 24 // (op 942) 
s_ashr_i32 s2, s0, 31 // (op 942) 

s_xor_b32 s0, s1, s2 // (op 942) 

s_sub_i32 s1, s0, s2 // (op 942) 


s_mul_hi_i32 s0, s1, s11 // (op 942) 
s_add_i32 s2, s0, s1 // (op 942) 

s_ashr_i32 s0, s2, 31 // (op 942) 
s_cmp_eq_i32 s11, 0 // (op 942) 
s_mov_b32 s3, scc // (op 942) 
s_cmp_eq_i32 s3, 0 // (op 942) 
s_cselect_b32 s6, 0, -1 // (op 942) 

s_and_b32 s3, s12, 31 // (op 942) 
s_lshl_b32 s7, 1, s3 // (op 942) 
s_add_i32 s18, s7, s6 // (op 942) 


s_and_b32 s19, s0, s18 // (op 942) 


s_add_i32 s0, s2, s19 // (op 942) 


s_ashr_i32 s2, s0, s3 // (op 942) 


s_lshl_b32 s0, s12, 24 // (op 942) 
s_ashr_i32 s3, s0, 31 // (op 942) 

s_xor_b32 s0, s2, s3 // (op 942) 

s_sub_i32 s2, s0, s3 // (op 942) 


s_mul_i32 s0, s1, s14 // (op 942) 
s_sub_u32 s3, s16, s0 // (op 942) 

s_mul_i32 s0, s2, s14 // (op 942) 
s_add_u32 s0, s0, s3 // (op 942) 

s_mul_hi_i32 s3, s16, s8 // (op 942) 
s_add_i32 s6, s3, s16 // (op 942) 

s_ashr_i32 s3, s6, 31 // (op 942) 
s_cmp_eq_i32 s8, 0 // (op 942) 
s_mov_b32 s7, scc // (op 942) 
s_cmp_eq_i32 s7, 0 // (op 942) 
s_cselect_b32 s18, 0, -1 // (op 942) 

s_and_b32 s7, s15, 31 // (op 942) 
s_lshl_b32 s19, 1, s7 // (op 942) 
s_add_i32 s22, s19, s18 // (op 942) 


s_and_b32 s18, s3, s22 // (op 942) 


s_add_i32 s3, s6, s18 // (op 942) 


s_ashr_i32 s6, s3, s7 // (op 942) 


s_lshl_b32 s3, s15, 24 // (op 942) 
s_ashr_i32 s7, s3, 31 // (op 942) 

s_xor_b32 s3, s6, s7 // (op 942) 

s_sub_i32 s6, s3, s7 // (op 942) 


s_add_u32 s18, s4, 16 // (op 942) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 942) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 942) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 942) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 4 // (op 942) 

s_mul_i32 s3, s14, s18 // (op 942) 
s_mul_i32 s7, s50, s3 // (op 942) 

s_mul_i32 s3, s6, s7 // (op 942) 

s_sub_u32 s7, s16, s3 // (op 942) 

s_mov_b32 s3, s7 // (op 942) 
s_mul_hi_i32 s22, s3, s13 // (op 942) 
s_add_i32 s23, s22, s3 // (op 942) 


s_ashr_i32 s3, s23, 31 // (op 942) 
s_cmp_eq_i32 s13, 0 // (op 942) 
s_mov_b32 s22, scc // (op 942) 
s_cmp_eq_i32 s22, 0 // (op 942) 
s_cselect_b32 s24, 0, -1 // (op 942) 

s_and_b32 s22, s49, 31 // (op 942) 
s_lshl_b32 s25, 1, s22 // (op 942) 
s_add_i32 s26, s25, s24 // (op 942) 


s_and_b32 s24, s3, s26 // (op 942) 


s_add_i32 s3, s23, s24 // (op 942) 


s_ashr_i32 s23, s3, s22 // (op 942) 


s_lshl_b32 s3, s49, 24 // (op 942) 
s_ashr_i32 s22, s3, 31 // (op 942) 

s_xor_b32 s3, s23, s22 // (op 942) 

s_sub_i32 s23, s3, s22 // (op 942) 


s_mul_i32 s3, s48, s14 // (op 942) 
s_add_u32 s24, s20, 16 // (op 942) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 942) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 942) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 942) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 4 // (op 942) 

s_sub_i32 s22, s24, s3 // (op 942) 

s_mul_i32 s24, s23, s22 // (op 942) 

s_sub_u32 s22, s7, s24 // (op 942) 


s_add_u32 s7, s22, s3 // (op 942) 


s_cmp_eq_i32 s6, 0 // (op 942) 
s_mov_b32 s3, scc // (op 942) 

s_cmp_eq_i32 s3, 0 // (op 942) 
s_cselect_b32 s6, s7, s0 // (op 942) 


s_mul_i32 s0, s2, s18 // (op 942) 


v_lshl_add_u32 v3, s6, 4, v5 // (op 942) 


s_sub_i32 s2, s1, s0 // (op 942) 


s_cmp_eq_i32 s3, 0 // (op 942) 
s_cselect_b32 s22, s23, s2 // (op 942) 



v_lshl_add_u32 v4, s22, 4, v11 // (op 942) 


v_mul_lo_u32 v5, v4, s40 // (op 942) 

v_add_lshl_u32 v4, v3, v5, 1 // (op 942) 


s_mul_i32 s7, s40, s5 // (op 944) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 944) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 944) least significant
s_mul_hi_u32 s1, s40, s4 // (op 944) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 944) most significant: sum
s_add_u32 s7, s7, s1 // (op 944) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 944) 

s_mov_b32 s0, s42 // (op 944) reg expression
s_mov_b32 s1, s43 // (op 944) reg expression
s_mov_b32 s2, s18 // (op 944) reg expression
s_mov_b32 s3, 131072 // (op 944) call()


v_pack_b32_f16 v12, v6, v7 // (op 58) 
v_pack_b32_f16 v13, v8, v9 // (op 58) 


buffer_store_dwordx2 v[12:13], v4, s[0:3], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_135
RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_135:
s_mul_hi_i32 s6, s16, s9 // (op 711) 
v_lshrrev_b32 v3, 6, v1 // (op 711) 
s_add_i32 s7, s6, s16 // (op 711) 

v_and_b32 v4, 63, v1 // (op 711) 
s_ashr_i32 s6, s7, 31 // (op 711) 
v_lshrrev_b32 v5, 4, v4 // (op 711) 
s_cmp_eq_i32 s9, 0 // (op 711) 
s_mov_b32 s18, scc // (op 711) 
v_lshlrev_b32 v6, 2, v5 // (op 711) 

s_cmp_eq_i32 s18, 0 // (op 711) 
s_cselect_b32 s19, 0, -1 // (op 711) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 711) 


s_and_b32 s18, s10, 31 // (op 711) 
v_lshrrev_b32 v3, 2, v4 // (op 711) 

s_lshl_b32 s22, 1, s18 // (op 711) 
v_and_b32 v4, 3, v3 // (op 711) 

s_add_i32 s23, s22, s19 // (op 711) 


v_and_b32 v3, 3, v1 // (op 711) 
s_and_b32 s19, s6, s23 // (op 711) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 711) 


s_add_i32 s6, s7, s19 // (op 711) 


s_ashr_i32 s7, s6, s18 // (op 711) 


s_lshl_b32 s6, s10, 24 // (op 711) 
s_ashr_i32 s18, s6, 31 // (op 711) 

s_xor_b32 s6, s7, s18 // (op 711) 

s_sub_i32 s7, s6, s18 // (op 711) 


s_mul_hi_i32 s6, s7, s11 // (op 711) 
s_add_i32 s18, s6, s7 // (op 711) 

s_ashr_i32 s6, s18, 31 // (op 711) 
s_cmp_eq_i32 s11, 0 // (op 711) 
s_mov_b32 s19, scc // (op 711) 
s_cmp_eq_i32 s19, 0 // (op 711) 
s_cselect_b32 s22, 0, -1 // (op 711) 

s_and_b32 s19, s12, 31 // (op 711) 
s_lshl_b32 s23, 1, s19 // (op 711) 
s_add_i32 s24, s23, s22 // (op 711) 


s_and_b32 s22, s6, s24 // (op 711) 


s_add_i32 s6, s18, s22 // (op 711) 


s_ashr_i32 s18, s6, s19 // (op 711) 


s_lshl_b32 s6, s12, 24 // (op 711) 
s_ashr_i32 s19, s6, 31 // (op 711) 

s_xor_b32 s6, s18, s19 // (op 711) 

s_sub_i32 s18, s6, s19 // (op 711) 


s_mul_i32 s6, s7, s14 // (op 711) 
s_sub_u32 s19, s16, s6 // (op 711) 

s_mul_i32 s6, s18, s14 // (op 711) 
s_add_u32 s6, s6, s19 // (op 711) 

s_mul_hi_i32 s19, s16, s8 // (op 711) 
s_add_i32 s22, s19, s16 // (op 711) 

s_ashr_i32 s19, s22, 31 // (op 711) 
s_cmp_eq_i32 s8, 0 // (op 711) 
s_mov_b32 s23, scc // (op 711) 
s_cmp_eq_i32 s23, 0 // (op 711) 
s_cselect_b32 s24, 0, -1 // (op 711) 

s_and_b32 s23, s15, 31 // (op 711) 
s_lshl_b32 s25, 1, s23 // (op 711) 
s_add_i32 s26, s25, s24 // (op 711) 


s_and_b32 s24, s19, s26 // (op 711) 


s_add_i32 s19, s22, s24 // (op 711) 


s_ashr_i32 s22, s19, s23 // (op 711) 


s_lshl_b32 s19, s15, 24 // (op 711) 
s_ashr_i32 s23, s19, 31 // (op 711) 

s_xor_b32 s19, s22, s23 // (op 711) 

s_sub_i32 s22, s19, s23 // (op 711) 


s_add_u32 s38, s4, 16 // (op 711) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 711) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 711) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 711) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 711) 

s_mul_i32 s19, s14, s38 // (op 711) 
s_mul_i32 s23, s50, s19 // (op 711) 

s_mul_i32 s19, s22, s23 // (op 711) 

s_sub_u32 s23, s16, s19 // (op 711) 

s_mov_b32 s19, s23 // (op 711) 
s_mul_hi_i32 s24, s19, s13 // (op 711) 
s_add_i32 s25, s24, s19 // (op 711) 


s_ashr_i32 s19, s25, 31 // (op 711) 
s_cmp_eq_i32 s13, 0 // (op 711) 
s_mov_b32 s24, scc // (op 711) 
s_cmp_eq_i32 s24, 0 // (op 711) 
s_cselect_b32 s26, 0, -1 // (op 711) 

s_and_b32 s24, s49, 31 // (op 711) 
s_lshl_b32 s27, 1, s24 // (op 711) 
s_add_i32 s28, s27, s26 // (op 711) 


s_and_b32 s26, s19, s28 // (op 711) 


s_add_i32 s19, s25, s26 // (op 711) 


s_ashr_i32 s25, s19, s24 // (op 711) 


s_lshl_b32 s19, s49, 24 // (op 711) 
s_ashr_i32 s24, s19, 31 // (op 711) 

s_xor_b32 s19, s25, s24 // (op 711) 

s_sub_i32 s25, s19, s24 // (op 711) 


s_mul_i32 s19, s48, s14 // (op 711) 
s_add_u32 s26, s20, 16 // (op 711) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 711) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 711) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 711) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 711) 

s_sub_i32 s24, s26, s19 // (op 711) 

s_mul_i32 s26, s25, s24 // (op 711) 

s_sub_u32 s24, s23, s26 // (op 711) 


s_add_u32 s23, s24, s19 // (op 711) 


s_cmp_eq_i32 s22, 0 // (op 711) 
s_mov_b32 s19, scc // (op 711) 

s_cmp_eq_i32 s19, 0 // (op 711) 
s_cselect_b32 s22, s23, s6 // (op 711) 


s_mul_i32 s6, s18, s38 // (op 711) 


v_lshl_add_u32 v3, s22, 4, v5 // (op 711) 


s_sub_i32 s18, s7, s6 // (op 711) 


s_cmp_eq_i32 s19, 0 // (op 711) 
s_cselect_b32 s6, s25, s18 // (op 711) 



v_lshl_add_u32 v4, s6, 4, v6 // (op 711) 


v_mul_lo_u32 v5, v4, s44 // (op 711) 

v_add_lshl_u32 v4, v3, v5, 1 // (op 711) 


s_mul_i32 s7, s44, s5 // (op 713) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 713) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 713) least significant
s_mul_hi_u32 s19, s44, s4 // (op 713) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 713) most significant: sum
s_add_u32 s7, s7, s19 // (op 713) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 713) 

s_mov_b32 s36, s46 // (op 713) reg expression
s_mov_b32 s37, s47 // (op 713) reg expression
s_mov_b32 s38, s18 // (op 713) reg expression
s_mov_b32 s39, 131072 // (op 713) call()





buffer_load_dwordx2 v[8:9], v4, s[36:39], 0 offen // (op 148) Load value



v_accvgpr_read v4, a0 // (op 144) 
v_accvgpr_read v5, a1 // (op 144) 
v_accvgpr_read v6, a2 // (op 144) 
v_accvgpr_read v7, a3 // (op 144) 
v_mul_f32 v12, v10, v4 // (op 144) 
v_mul_f32 v13, v10, v5 // (op 144) 
v_mul_f32 v14, v10, v6 // (op 144) 
v_mul_f32 v15, v10, v7 // (op 144) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 149) 
v_lshrrev_b32 v7, 16, v8 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v16, v0, v6 // (op 149) 
v_mul_f32 v17, v0, v7 // (op 149) 
v_and_b32 v6, 65535, v9 // (op 149) 
v_lshrrev_b32 v7, 16, v9 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v18, v0, v6 // (op 149) 
v_mul_f32 v19, v0, v7 // (op 149) 


v_add_f32 v6, v16, v12 // (op 145) 
v_add_f32 v7, v17, v13 // (op 145) 
v_add_f32 v8, v18, v14 // (op 145) 
v_add_f32 v9, v19, v15 // (op 145) 


v_lshrrev_b32 v12, 16, v6 // (op 146) 
v_lshrrev_b32 v13, 16, v7 // (op 146) 
v_lshrrev_b32 v14, 16, v8 // (op 146) 
v_lshrrev_b32 v15, 16, v9 // (op 146) 

s_mul_hi_i32 s6, s16, s9 // (op 959) 
v_lshrrev_b32 v11, 6, v1 // (op 959) 
s_add_i32 s7, s6, s16 // (op 959) 

v_and_b32 v3, 63, v1 // (op 959) 
s_ashr_i32 s6, s7, 31 // (op 959) 
v_lshrrev_b32 v4, 4, v3 // (op 959) 
s_cmp_eq_i32 s9, 0 // (op 959) 
s_mov_b32 s18, scc // (op 959) 
v_lshlrev_b32 v5, 2, v4 // (op 959) 

s_cmp_eq_i32 s18, 0 // (op 959) 
s_cselect_b32 s19, 0, -1 // (op 959) 

v_lshl_add_u32 v4, v11, 4, v5 // (op 959) 


s_and_b32 s18, s10, 31 // (op 959) 
v_lshrrev_b32 v11, 2, v3 // (op 959) 

s_lshl_b32 s22, 1, s18 // (op 959) 
v_and_b32 v3, 3, v11 // (op 959) 

s_add_i32 s23, s22, s19 // (op 959) 


v_and_b32 v11, 3, v1 // (op 959) 
s_and_b32 s19, s6, s23 // (op 959) 


v_lshl_add_u32 v5, v3, 2, v11 // (op 959) 


s_add_i32 s6, s7, s19 // (op 959) 


s_ashr_i32 s7, s6, s18 // (op 959) 


s_lshl_b32 s6, s10, 24 // (op 959) 
s_ashr_i32 s18, s6, 31 // (op 959) 

s_xor_b32 s6, s7, s18 // (op 959) 

s_sub_i32 s7, s6, s18 // (op 959) 


s_mul_hi_i32 s6, s7, s11 // (op 959) 
s_add_i32 s18, s6, s7 // (op 959) 

s_ashr_i32 s6, s18, 31 // (op 959) 
s_cmp_eq_i32 s11, 0 // (op 959) 
s_mov_b32 s19, scc // (op 959) 
s_cmp_eq_i32 s19, 0 // (op 959) 
s_cselect_b32 s22, 0, -1 // (op 959) 

s_and_b32 s19, s12, 31 // (op 959) 
s_lshl_b32 s23, 1, s19 // (op 959) 
s_add_i32 s24, s23, s22 // (op 959) 


s_and_b32 s22, s6, s24 // (op 959) 


s_add_i32 s6, s18, s22 // (op 959) 


s_ashr_i32 s18, s6, s19 // (op 959) 


s_lshl_b32 s6, s12, 24 // (op 959) 
s_ashr_i32 s19, s6, 31 // (op 959) 

s_xor_b32 s6, s18, s19 // (op 959) 

s_sub_i32 s18, s6, s19 // (op 959) 


s_mul_i32 s6, s7, s14 // (op 959) 
s_sub_u32 s19, s16, s6 // (op 959) 

s_mul_i32 s6, s18, s14 // (op 959) 
s_add_u32 s6, s6, s19 // (op 959) 

s_mul_hi_i32 s19, s16, s8 // (op 959) 
s_add_i32 s22, s19, s16 // (op 959) 

s_ashr_i32 s19, s22, 31 // (op 959) 
s_cmp_eq_i32 s8, 0 // (op 959) 
s_mov_b32 s23, scc // (op 959) 
s_cmp_eq_i32 s23, 0 // (op 959) 
s_cselect_b32 s24, 0, -1 // (op 959) 

s_and_b32 s23, s15, 31 // (op 959) 
s_lshl_b32 s25, 1, s23 // (op 959) 
s_add_i32 s26, s25, s24 // (op 959) 


s_and_b32 s24, s19, s26 // (op 959) 


s_add_i32 s19, s22, s24 // (op 959) 


s_ashr_i32 s22, s19, s23 // (op 959) 


s_lshl_b32 s19, s15, 24 // (op 959) 
s_ashr_i32 s23, s19, 31 // (op 959) 

s_xor_b32 s19, s22, s23 // (op 959) 

s_sub_i32 s22, s19, s23 // (op 959) 


s_add_u32 s38, s4, 16 // (op 959) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 959) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 959) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 959) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 959) 

s_mul_i32 s19, s14, s38 // (op 959) 
s_mul_i32 s23, s50, s19 // (op 959) 

s_mul_i32 s19, s22, s23 // (op 959) 

s_sub_u32 s23, s16, s19 // (op 959) 

s_mov_b32 s19, s23 // (op 959) 
s_mul_hi_i32 s24, s19, s13 // (op 959) 
s_add_i32 s25, s24, s19 // (op 959) 


s_ashr_i32 s19, s25, 31 // (op 959) 
s_cmp_eq_i32 s13, 0 // (op 959) 
s_mov_b32 s24, scc // (op 959) 
s_cmp_eq_i32 s24, 0 // (op 959) 
s_cselect_b32 s26, 0, -1 // (op 959) 

s_and_b32 s24, s49, 31 // (op 959) 
s_lshl_b32 s27, 1, s24 // (op 959) 
s_add_i32 s28, s27, s26 // (op 959) 


s_and_b32 s26, s19, s28 // (op 959) 


s_add_i32 s19, s25, s26 // (op 959) 


s_ashr_i32 s25, s19, s24 // (op 959) 


s_lshl_b32 s19, s49, 24 // (op 959) 
s_ashr_i32 s24, s19, 31 // (op 959) 

s_xor_b32 s19, s25, s24 // (op 959) 

s_sub_i32 s25, s19, s24 // (op 959) 


s_mul_i32 s19, s48, s14 // (op 959) 
s_add_u32 s26, s20, 16 // (op 959) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 959) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 959) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 959) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 959) 

s_sub_i32 s24, s26, s19 // (op 959) 

s_mul_i32 s26, s25, s24 // (op 959) 

s_sub_u32 s24, s23, s26 // (op 959) 


s_add_u32 s23, s24, s19 // (op 959) 


s_cmp_eq_i32 s22, 0 // (op 959) 
s_mov_b32 s19, scc // (op 959) 

s_cmp_eq_i32 s19, 0 // (op 959) 
s_cselect_b32 s22, s23, s6 // (op 959) 


s_mul_i32 s6, s18, s38 // (op 959) 


v_lshl_add_u32 v3, s22, 4, v4 // (op 959) 


s_sub_i32 s18, s7, s6 // (op 959) 


s_cmp_eq_i32 s19, 0 // (op 959) 
s_cselect_b32 s6, s25, s18 // (op 959) 



v_lshl_add_u32 v4, s6, 4, v5 // (op 959) 


v_mul_lo_u32 v11, v4, s40 // (op 959) 

v_add_lshl_u32 v4, v3, v11, 1 // (op 959) 


s_mul_i32 s7, s40, s5 // (op 961) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 961) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 961) least significant
s_mul_hi_u32 s19, s40, s4 // (op 961) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 961) most significant: sum
s_add_u32 s7, s7, s19 // (op 961) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 961) 

s_mov_b32 s0, s42 // (op 961) reg expression
s_mov_b32 s1, s43 // (op 961) reg expression
s_mov_b32 s2, s18 // (op 961) reg expression
s_mov_b32 s3, 131072 // (op 961) call()


v_pack_b32_f16 v8, v12, v13 // (op 147) 
v_pack_b32_f16 v9, v14, v15 // (op 147) 


buffer_store_dwordx2 v[8:9], v4, s[0:3], 0 offen // (op 147) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_135:
















s_endpgm  // End of RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.LRR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end:
.size RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_, .LRR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end-RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 76
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 6400 // lds bytes
  .amdhsa_accum_offset 72
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
    DataType: F4
    DataTypeA: F4
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
  - .name: RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 6400
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 70
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

