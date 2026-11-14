// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_,@function
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 767) 
v_and_b32 v3, 15, v1 // (op 767) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 767) 


v_and_b32 v0, 63, v1 // (op 767) 
v_lshrrev_b32 v3, 4, v0 // (op 767) 

v_lshlrev_b32 v0, 5, v3 // (op 767) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 767) 


v_mul_lo_u32 v0, v3, 6 // (op 767) 

v_lshrrev_b32 v3, 3, v0 // (op 767) 

v_and_b32 v0, 15, v1 // (op 819) 
v_and_b32 v4, 48, v1 // (op 819) 
v_lshl_add_u32 v5, v0, 7, v4 // (op 819) 


v_lshrrev_b32 v0, 6, v1 // (op 872) 
v_and_b32 v4, 15, v1 // (op 872) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 872) 


v_and_b32 v0, 63, v1 // (op 872) 
v_lshrrev_b32 v4, 4, v0 // (op 872) 

v_lshl_add_u32 v0, v6, 2, v4 // (op 872) 


v_lshrrev_b32 v4, 6, v1 // (op 930) 
v_and_b32 v6, 15, v1 // (op 930) 
v_lshl_add_u32 v7, v4, 4, v6 // (op 930) 


v_and_b32 v4, 63, v1 // (op 930) 
v_lshrrev_b32 v6, 4, v4 // (op 930) 

v_lshlrev_b32 v4, 5, v6 // (op 930) 

v_lshl_add_u32 v6, v7, 7, v4 // (op 930) 


v_mul_lo_u32 v4, v6, 6 // (op 930) 

v_lshrrev_b32 v6, 3, v4 // (op 930) 

v_and_b32 v4, 15, v1 // (op 969) 
v_and_b32 v7, 48, v1 // (op 969) 
v_lshl_add_u32 v8, v4, 7, v7 // (op 969) 


v_lshrrev_b32 v4, 6, v1 // (op 993) 
v_and_b32 v7, 15, v1 // (op 993) 
v_lshl_add_u32 v9, v4, 4, v7 // (op 993) 


v_and_b32 v4, 63, v1 // (op 993) 
v_lshrrev_b32 v7, 4, v4 // (op 993) 

v_lshl_add_u32 v4, v9, 2, v7 // (op 993) 


v_mov_b32 v7, s8 // (op 2) Move value
v_mov_b32 v9, s9 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
s_mul_hi_i32 s0, s16, s12 // (op 477) 
v_lshrrev_b32 v10, 2, v1 // (op 477) 
s_add_i32 s1, s0, s16 // (op 477) 

v_and_b32 v11, -4, v1 // (op 477) 
s_ashr_i32 s0, s1, 31 // (op 477) 
v_sub_u32 v12, v1, v11 // (op 477) 

s_cmp_eq_i32 s12, 0 // (op 477) 
s_mov_b32 s8, scc // (op 477) 
v_lshlrev_b32 v11, 5, v12 // (op 477) 

s_cmp_eq_i32 s8, 0 // (op 477) 
s_cselect_b32 s9, 0, -1 // (op 477) 

v_mov_b32 v13, 0 // (op 477) convert
v_mov_b32 v12, v11 // (op 477) convert

s_and_b32 s8, s13, 31 // (op 477) 
s_lshl_b32 s18, 1, s8 // (op 477) 
s_add_i32 s19, s18, s9 // (op 477) 


s_and_b32 s9, s0, s19 // (op 477) 


s_add_i32 s0, s1, s9 // (op 477) 


s_ashr_i32 s1, s0, s8 // (op 477) 


s_lshl_b32 s0, s13, 24 // (op 477) 
s_ashr_i32 s8, s0, 31 // (op 477) 

s_xor_b32 s0, s1, s8 // (op 477) 

s_sub_i32 s1, s0, s8 // (op 477) 


s_mul_hi_i32 s0, s1, s14 // (op 477) 
s_add_i32 s8, s0, s1 // (op 477) 

s_ashr_i32 s0, s8, 31 // (op 477) 
s_cmp_eq_i32 s14, 0 // (op 477) 
s_mov_b32 s9, scc // (op 477) 
s_cmp_eq_i32 s9, 0 // (op 477) 
s_cselect_b32 s18, 0, -1 // (op 477) 

s_and_b32 s9, s15, 31 // (op 477) 
s_lshl_b32 s19, 1, s9 // (op 477) 
s_add_i32 s49, s19, s18 // (op 477) 


s_and_b32 s18, s0, s49 // (op 477) 


s_add_i32 s0, s8, s18 // (op 477) 


s_ashr_i32 s8, s0, s9 // (op 477) 


s_lshl_b32 s0, s15, 24 // (op 477) 
s_ashr_i32 s9, s0, 31 // (op 477) 

s_xor_b32 s0, s8, s9 // (op 477) 

s_sub_i32 s8, s0, s9 // (op 477) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s45 // (op 477) 

s_sub_u32 s1, s16, s0 // (op 477) 

s_mul_i32 s0, s8, s45 // (op 477) 
s_add_u32 s0, s0, s1 // (op 477) 


s_mul_hi_i32 s1, s16, s11 // (op 477) 
s_add_i32 s8, s1, s16 // (op 477) 

s_ashr_i32 s1, s8, 31 // (op 477) 
s_cmp_eq_i32 s11, 0 // (op 477) 
s_mov_b32 s9, scc // (op 477) 
s_cmp_eq_i32 s9, 0 // (op 477) 
s_cselect_b32 s18, 0, -1 // (op 477) 

s_and_b32 s9, s46, 31 // (op 477) 
s_lshl_b32 s19, 1, s9 // (op 477) 
s_add_i32 s49, s19, s18 // (op 477) 


s_and_b32 s18, s1, s49 // (op 477) 


s_add_i32 s1, s8, s18 // (op 477) 


s_ashr_i32 s8, s1, s9 // (op 477) 


s_lshl_b32 s1, s46, 24 // (op 477) 
s_ashr_i32 s9, s1, 31 // (op 477) 

s_xor_b32 s1, s8, s9 // (op 477) 

s_sub_i32 s8, s1, s9 // (op 477) 


s_add_u32 s18, s4, 16 // (op 477) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 477) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 477) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 477) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 477) 

s_mul_i32 s1, s45, s18 // (op 477) 

s_mul_i32 s9, s10, s1 // (op 477) 

s_mul_i32 s1, s8, s9 // (op 477) 

s_sub_u32 s9, s16, s1 // (op 477) 

s_mov_b32 s1, s9 // (op 477) 
s_mul_hi_i32 s18, s1, s44 // (op 477) 
s_add_i32 s19, s18, s1 // (op 477) 


s_ashr_i32 s1, s19, 31 // (op 477) 
s_cmp_eq_i32 s44, 0 // (op 477) 
s_mov_b32 s18, scc // (op 477) 
s_cmp_eq_i32 s18, 0 // (op 477) 
s_cselect_b32 s49, 0, -1 // (op 477) 

s_and_b32 s18, s48, 31 // (op 477) 
s_lshl_b32 s50, 1, s18 // (op 477) 
s_add_i32 s51, s50, s49 // (op 477) 


s_and_b32 s49, s1, s51 // (op 477) 


s_add_i32 s1, s19, s49 // (op 477) 


s_ashr_i32 s19, s1, s18 // (op 477) 


s_lshl_b32 s1, s48, 24 // (op 477) 
s_ashr_i32 s18, s1, 31 // (op 477) 

s_xor_b32 s1, s19, s18 // (op 477) 

s_sub_i32 s19, s1, s18 // (op 477) 


s_mul_i32 s1, s47, s45 // (op 477) 
s_add_u32 s50, s6, 32 // (op 477) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 477) End of Int64 add, unlocking SCC
s_sub_u32 s52, s50, 1 // (op 477) least significant half; sets scc
s_subb_u32 s53, s51, 0 // (op 477) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 477) 

s_sub_i32 s18, s50, s1 // (op 477) 

s_mul_i32 s49, s19, s18 // (op 477) 


s_sub_u32 s18, s9, s49 // (op 477) 


s_add_u32 s9, s18, s1 // (op 477) 


s_cmp_eq_i32 s8, 0 // (op 477) 
s_mov_b32 s1, scc // (op 477) 

s_cmp_eq_i32 s1, 0 // (op 477) 
s_cselect_b32 s8, s9, s0 // (op 477) 



v_lshl_add_u32 v11, s8, 5, v10 // (op 477) 


v_mul_lo_u32 v15, v11, s3 // (op 477) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 477) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 477) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 477) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 477) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 477) most significant half



v_mul_lo_u32 v11, v15, 6 // (op 477) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 477) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 477) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 477) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 477) 

v_mov_b32 v10, v12 // (op 477) convert

s_mul_i32 s1, s7, 6 // (op 479) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 479) least significant
s_mul_hi_u32 s8, s6, 6 // (op 479) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 479) most significant: sum

s_mul_i32 s9, s2, s1 // (op 479) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 479) most significant: low of high * low
s_mul_i32 s8, s2, s0 // (op 479) least significant
s_mul_hi_u32 s19, s2, s0 // (op 479) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 479) most significant: sum
s_add_u32 s9, s9, s19 // (op 479) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 479) 

s_mov_b32 s52, s20 // (op 479) reg expression
s_mov_b32 s53, s21 // (op 479) reg expression
s_mov_b32 s54, s0 // (op 479) reg expression
s_mov_b32 s55, 131072 // (op 479) call()

s_mul_hi_i32 s0, s16, s12 // (op 496) 
v_lshrrev_b32 v11, 2, v1 // (op 496) 
s_add_i32 s1, s0, s16 // (op 496) 

v_and_b32 v12, -4, v1 // (op 496) 
s_ashr_i32 s0, s1, 31 // (op 496) 
v_sub_u32 v13, v1, v12 // (op 496) 

s_cmp_eq_i32 s12, 0 // (op 496) 
s_mov_b32 s8, scc // (op 496) 
v_lshlrev_b32 v12, 5, v13 // (op 496) 

s_cmp_eq_i32 s8, 0 // (op 496) 
s_cselect_b32 s9, 0, -1 // (op 496) 

v_mov_b32 v15, 0 // (op 496) convert
v_mov_b32 v14, v12 // (op 496) convert

s_and_b32 s8, s13, 31 // (op 496) 
s_lshl_b32 s18, 1, s8 // (op 496) 
s_add_i32 s19, s18, s9 // (op 496) 


s_and_b32 s9, s0, s19 // (op 496) 


s_add_i32 s0, s1, s9 // (op 496) 


s_ashr_i32 s1, s0, s8 // (op 496) 


s_lshl_b32 s0, s13, 24 // (op 496) 
s_ashr_i32 s8, s0, 31 // (op 496) 

s_xor_b32 s0, s1, s8 // (op 496) 

s_sub_i32 s1, s0, s8 // (op 496) 


s_mul_hi_i32 s0, s1, s14 // (op 496) 
s_add_i32 s8, s0, s1 // (op 496) 

s_ashr_i32 s0, s8, 31 // (op 496) 
s_cmp_eq_i32 s14, 0 // (op 496) 
s_mov_b32 s9, scc // (op 496) 
s_cmp_eq_i32 s9, 0 // (op 496) 
s_cselect_b32 s18, 0, -1 // (op 496) 

s_and_b32 s9, s15, 31 // (op 496) 
s_lshl_b32 s19, 1, s9 // (op 496) 
s_add_i32 s49, s19, s18 // (op 496) 


s_and_b32 s18, s0, s49 // (op 496) 


s_add_i32 s0, s8, s18 // (op 496) 


s_ashr_i32 s8, s0, s9 // (op 496) 


s_lshl_b32 s0, s15, 24 // (op 496) 
s_ashr_i32 s9, s0, 31 // (op 496) 

s_xor_b32 s0, s8, s9 // (op 496) 

s_sub_i32 s8, s0, s9 // (op 496) 


s_mul_i32 s0, s1, s45 // (op 496) 

s_sub_u32 s1, s16, s0 // (op 496) 

s_mul_i32 s0, s8, s45 // (op 496) 
s_add_u32 s0, s0, s1 // (op 496) 


s_mul_hi_i32 s1, s16, s11 // (op 496) 
s_add_i32 s8, s1, s16 // (op 496) 

s_ashr_i32 s1, s8, 31 // (op 496) 
s_cmp_eq_i32 s11, 0 // (op 496) 
s_mov_b32 s9, scc // (op 496) 
s_cmp_eq_i32 s9, 0 // (op 496) 
s_cselect_b32 s18, 0, -1 // (op 496) 

s_and_b32 s9, s46, 31 // (op 496) 
s_lshl_b32 s19, 1, s9 // (op 496) 
s_add_i32 s49, s19, s18 // (op 496) 


s_and_b32 s18, s1, s49 // (op 496) 


s_add_i32 s1, s8, s18 // (op 496) 


s_ashr_i32 s8, s1, s9 // (op 496) 


s_lshl_b32 s1, s46, 24 // (op 496) 
s_ashr_i32 s9, s1, 31 // (op 496) 

s_xor_b32 s1, s8, s9 // (op 496) 

s_sub_i32 s8, s1, s9 // (op 496) 


s_add_u32 s18, s4, 16 // (op 496) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 496) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 496) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 496) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 496) 

s_mul_i32 s1, s45, s18 // (op 496) 

s_mul_i32 s9, s10, s1 // (op 496) 

s_mul_i32 s1, s8, s9 // (op 496) 

s_sub_u32 s9, s16, s1 // (op 496) 

s_mov_b32 s1, s9 // (op 496) 
s_mul_hi_i32 s18, s1, s44 // (op 496) 
s_add_i32 s19, s18, s1 // (op 496) 


s_ashr_i32 s1, s19, 31 // (op 496) 
s_cmp_eq_i32 s44, 0 // (op 496) 
s_mov_b32 s18, scc // (op 496) 
s_cmp_eq_i32 s18, 0 // (op 496) 
s_cselect_b32 s49, 0, -1 // (op 496) 

s_and_b32 s18, s48, 31 // (op 496) 
s_lshl_b32 s50, 1, s18 // (op 496) 
s_add_i32 s51, s50, s49 // (op 496) 


s_and_b32 s49, s1, s51 // (op 496) 


s_add_i32 s1, s19, s49 // (op 496) 


s_ashr_i32 s19, s1, s18 // (op 496) 


s_lshl_b32 s1, s48, 24 // (op 496) 
s_ashr_i32 s18, s1, 31 // (op 496) 

s_xor_b32 s1, s19, s18 // (op 496) 

s_sub_i32 s19, s1, s18 // (op 496) 


s_mul_i32 s1, s47, s45 // (op 496) 
s_add_u32 s50, s6, 32 // (op 496) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 496) End of Int64 add, unlocking SCC
s_sub_u32 s56, s50, 1 // (op 496) least significant half; sets scc
s_subb_u32 s57, s51, 0 // (op 496) most significant half; uses scc

s_ashr_i64 s[50:51], s[56:57], 5 // (op 496) 

s_sub_i32 s18, s50, s1 // (op 496) 

s_mul_i32 s49, s19, s18 // (op 496) 


s_sub_u32 s18, s9, s49 // (op 496) 


s_add_u32 s9, s18, s1 // (op 496) 


s_cmp_eq_i32 s8, 0 // (op 496) 
s_mov_b32 s1, scc // (op 496) 

s_cmp_eq_i32 s1, 0 // (op 496) 
s_cselect_b32 s8, s9, s0 // (op 496) 



v_lshl_add_u32 v12, s8, 5, v11 // (op 496) 


v_mul_lo_u32 v17, v12, s3 // (op 496) most significant: low of low * high
v_mul_hi_u32 v11, v12, s2 // (op 496) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 496) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 496) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 496) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 496) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 496) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 496) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 496) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 496) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 496) 

v_mov_b32 v11, v12 // (op 496) convert

s_mul_i32 s1, s7, 6 // (op 498) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 498) least significant
s_mul_hi_u32 s8, s6, 6 // (op 498) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 498) most significant: sum

s_mul_i32 s9, s2, s1 // (op 498) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 498) most significant: low of high * low
s_mul_i32 s8, s2, s0 // (op 498) least significant
s_mul_hi_u32 s19, s2, s0 // (op 498) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 498) most significant: sum
s_add_u32 s9, s9, s19 // (op 498) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 498) 

s_mov_b32 s52, s20 // (op 498) reg expression
s_mov_b32 s53, s21 // (op 498) reg expression
s_mov_b32 s54, s0 // (op 498) reg expression
s_mov_b32 s55, 131072 // (op 498) call()

s_mul_hi_i32 s0, s16, s12 // (op 541) 
v_lshrrev_b32 v12, 3, v1 // (op 541) 
s_add_i32 s1, s0, s16 // (op 541) 

v_and_b32 v13, -8, v1 // (op 541) 
s_ashr_i32 s0, s1, 31 // (op 541) 
v_sub_u32 v14, v1, v13 // (op 541) 

s_cmp_eq_i32 s12, 0 // (op 541) 
s_mov_b32 s8, scc // (op 541) 
v_lshlrev_b32 v13, 4, v14 // (op 541) 

s_cmp_eq_i32 s8, 0 // (op 541) 
s_cselect_b32 s9, 0, -1 // (op 541) 

v_mov_b32 v15, 0 // (op 541) convert
v_mov_b32 v14, v13 // (op 541) convert

s_and_b32 s8, s13, 31 // (op 541) 
s_lshl_b32 s18, 1, s8 // (op 541) 
s_add_i32 s19, s18, s9 // (op 541) 


s_and_b32 s9, s0, s19 // (op 541) 


s_add_i32 s0, s1, s9 // (op 541) 


s_ashr_i32 s1, s0, s8 // (op 541) 


s_lshl_b32 s0, s13, 24 // (op 541) 
s_ashr_i32 s8, s0, 31 // (op 541) 

s_xor_b32 s0, s1, s8 // (op 541) 

s_sub_i32 s1, s0, s8 // (op 541) 


s_mul_hi_i32 s0, s1, s14 // (op 541) 
s_add_i32 s8, s0, s1 // (op 541) 

s_ashr_i32 s0, s8, 31 // (op 541) 
s_cmp_eq_i32 s14, 0 // (op 541) 
s_mov_b32 s9, scc // (op 541) 
s_cmp_eq_i32 s9, 0 // (op 541) 
s_cselect_b32 s18, 0, -1 // (op 541) 

s_and_b32 s9, s15, 31 // (op 541) 
s_lshl_b32 s19, 1, s9 // (op 541) 
s_add_i32 s49, s19, s18 // (op 541) 


s_and_b32 s18, s0, s49 // (op 541) 


s_add_i32 s0, s8, s18 // (op 541) 


s_ashr_i32 s8, s0, s9 // (op 541) 


s_lshl_b32 s0, s15, 24 // (op 541) 
s_ashr_i32 s9, s0, 31 // (op 541) 

s_xor_b32 s0, s8, s9 // (op 541) 

s_sub_i32 s8, s0, s9 // (op 541) 


s_add_u32 s18, s4, 16 // (op 541) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 541) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 541) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 541) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 541) 

s_mul_i32 s0, s8, s18 // (op 541) 

s_sub_i32 s8, s1, s0 // (op 541) 


s_mul_hi_i32 s9, s16, s11 // (op 541) 
s_add_i32 s0, s9, s16 // (op 541) 

s_ashr_i32 s1, s0, 31 // (op 541) 
s_cmp_eq_i32 s11, 0 // (op 541) 
s_mov_b32 s9, scc // (op 541) 
s_cmp_eq_i32 s9, 0 // (op 541) 
s_cselect_b32 s49, 0, -1 // (op 541) 

s_and_b32 s9, s46, 31 // (op 541) 
s_lshl_b32 s50, 1, s9 // (op 541) 
s_add_i32 s51, s50, s49 // (op 541) 


s_and_b32 s49, s1, s51 // (op 541) 


s_add_i32 s1, s0, s49 // (op 541) 


s_ashr_i32 s0, s1, s9 // (op 541) 


s_lshl_b32 s1, s46, 24 // (op 541) 
s_ashr_i32 s9, s1, 31 // (op 541) 

s_xor_b32 s1, s0, s9 // (op 541) 

s_sub_i32 s0, s1, s9 // (op 541) 


s_mul_i32 s1, s45, s18 // (op 541) 

s_mul_i32 s9, s10, s1 // (op 541) 

s_mul_i32 s1, s0, s9 // (op 541) 

s_sub_u32 s9, s16, s1 // (op 541) 

s_mul_hi_i32 s1, s9, s44 // (op 541) 
s_add_i32 s18, s1, s9 // (op 541) 


s_ashr_i32 s1, s18, 31 // (op 541) 
s_cmp_eq_i32 s44, 0 // (op 541) 
s_mov_b32 s9, scc // (op 541) 
s_cmp_eq_i32 s9, 0 // (op 541) 
s_cselect_b32 s19, 0, -1 // (op 541) 

s_and_b32 s9, s48, 31 // (op 541) 
s_lshl_b32 s49, 1, s9 // (op 541) 
s_add_i32 s50, s49, s19 // (op 541) 


s_and_b32 s19, s1, s50 // (op 541) 


s_add_i32 s1, s18, s19 // (op 541) 


s_ashr_i32 s18, s1, s9 // (op 541) 


s_lshl_b32 s1, s48, 24 // (op 541) 
s_ashr_i32 s9, s1, 31 // (op 541) 

s_xor_b32 s1, s18, s9 // (op 541) 

s_sub_i32 s18, s1, s9 // (op 541) 


s_cmp_eq_i32 s0, 0 // (op 541) 
s_mov_b32 s1, scc // (op 541) 

s_cmp_eq_i32 s1, 0 // (op 541) 
s_cselect_b32 s0, s18, s8 // (op 541) 



v_lshl_add_u32 v13, s0, 4, v12 // (op 541) 


v_mul_lo_u32 v17, v13, s23 // (op 541) most significant: low of low * high
v_mul_hi_u32 v12, v13, s22 // (op 541) most significant: high of low * low
v_mul_lo_u32 v16, v13, s22 // (op 541) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 541) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 541) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 541) most significant half



s_mul_i32 s1, s22, s5 // (op 543) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 543) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 543) least significant
s_mul_hi_u32 s9, s22, s4 // (op 543) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 543) most significant: sum
s_add_u32 s1, s1, s9 // (op 543) most significant: sum


s_mov_b32 s56, s24 // (op 543) reg expression
s_mov_b32 s57, s25 // (op 543) reg expression
s_mov_b32 s58, s0 // (op 543) reg expression
s_mov_b32 s59, 131072 // (op 543) call()

s_mul_hi_i32 s0, s16, s12 // (op 560) 
v_lshrrev_b32 v14, 3, v1 // (op 560) 
s_add_i32 s1, s0, s16 // (op 560) 

v_and_b32 v15, -8, v1 // (op 560) 
s_ashr_i32 s0, s1, 31 // (op 560) 
v_sub_u32 v16, v1, v15 // (op 560) 

s_cmp_eq_i32 s12, 0 // (op 560) 
s_mov_b32 s8, scc // (op 560) 
v_lshlrev_b32 v15, 4, v16 // (op 560) 

s_cmp_eq_i32 s8, 0 // (op 560) 
s_cselect_b32 s9, 0, -1 // (op 560) 

v_mov_b32 v17, 0 // (op 560) convert
v_mov_b32 v16, v15 // (op 560) convert

s_and_b32 s8, s13, 31 // (op 560) 
s_lshl_b32 s18, 1, s8 // (op 560) 
s_add_i32 s19, s18, s9 // (op 560) 


s_and_b32 s9, s0, s19 // (op 560) 


s_add_i32 s0, s1, s9 // (op 560) 


s_ashr_i32 s1, s0, s8 // (op 560) 


s_lshl_b32 s0, s13, 24 // (op 560) 
s_ashr_i32 s8, s0, 31 // (op 560) 

s_xor_b32 s0, s1, s8 // (op 560) 

s_sub_i32 s1, s0, s8 // (op 560) 


s_mul_hi_i32 s0, s1, s14 // (op 560) 
s_add_i32 s8, s0, s1 // (op 560) 

s_ashr_i32 s0, s8, 31 // (op 560) 
s_cmp_eq_i32 s14, 0 // (op 560) 
s_mov_b32 s9, scc // (op 560) 
s_cmp_eq_i32 s9, 0 // (op 560) 
s_cselect_b32 s18, 0, -1 // (op 560) 

s_and_b32 s9, s15, 31 // (op 560) 
s_lshl_b32 s19, 1, s9 // (op 560) 
s_add_i32 s49, s19, s18 // (op 560) 


s_and_b32 s18, s0, s49 // (op 560) 


s_add_i32 s0, s8, s18 // (op 560) 


s_ashr_i32 s8, s0, s9 // (op 560) 


s_lshl_b32 s0, s15, 24 // (op 560) 
s_ashr_i32 s9, s0, 31 // (op 560) 

s_xor_b32 s0, s8, s9 // (op 560) 

s_sub_i32 s8, s0, s9 // (op 560) 


s_add_u32 s18, s4, 16 // (op 560) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 560) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 560) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 560) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 560) 

s_mul_i32 s0, s8, s18 // (op 560) 

s_sub_i32 s8, s1, s0 // (op 560) 


s_mul_hi_i32 s9, s16, s11 // (op 560) 
s_add_i32 s0, s9, s16 // (op 560) 

s_ashr_i32 s1, s0, 31 // (op 560) 
s_cmp_eq_i32 s11, 0 // (op 560) 
s_mov_b32 s9, scc // (op 560) 
s_cmp_eq_i32 s9, 0 // (op 560) 
s_cselect_b32 s49, 0, -1 // (op 560) 

s_and_b32 s9, s46, 31 // (op 560) 
s_lshl_b32 s50, 1, s9 // (op 560) 
s_add_i32 s51, s50, s49 // (op 560) 


s_and_b32 s49, s1, s51 // (op 560) 


s_add_i32 s1, s0, s49 // (op 560) 


s_ashr_i32 s0, s1, s9 // (op 560) 


s_lshl_b32 s1, s46, 24 // (op 560) 
s_ashr_i32 s9, s1, 31 // (op 560) 

s_xor_b32 s1, s0, s9 // (op 560) 

s_sub_i32 s0, s1, s9 // (op 560) 


s_mul_i32 s1, s45, s18 // (op 560) 

s_mul_i32 s9, s10, s1 // (op 560) 

s_mul_i32 s1, s0, s9 // (op 560) 

s_sub_u32 s9, s16, s1 // (op 560) 

s_mul_hi_i32 s1, s9, s44 // (op 560) 
s_add_i32 s18, s1, s9 // (op 560) 


s_ashr_i32 s1, s18, 31 // (op 560) 
s_cmp_eq_i32 s44, 0 // (op 560) 
s_mov_b32 s9, scc // (op 560) 
s_cmp_eq_i32 s9, 0 // (op 560) 
s_cselect_b32 s19, 0, -1 // (op 560) 

s_and_b32 s9, s48, 31 // (op 560) 
s_lshl_b32 s49, 1, s9 // (op 560) 
s_add_i32 s50, s49, s19 // (op 560) 


s_and_b32 s19, s1, s50 // (op 560) 


s_add_i32 s1, s18, s19 // (op 560) 


s_ashr_i32 s18, s1, s9 // (op 560) 


s_lshl_b32 s1, s48, 24 // (op 560) 
s_ashr_i32 s9, s1, 31 // (op 560) 

s_xor_b32 s1, s18, s9 // (op 560) 

s_sub_i32 s18, s1, s9 // (op 560) 


s_cmp_eq_i32 s0, 0 // (op 560) 
s_mov_b32 s1, scc // (op 560) 

s_cmp_eq_i32 s1, 0 // (op 560) 
s_cselect_b32 s0, s18, s8 // (op 560) 



v_lshl_add_u32 v15, s0, 4, v14 // (op 560) 


v_mul_lo_u32 v19, v15, s23 // (op 560) most significant: low of low * high
v_mul_hi_u32 v14, v15, s22 // (op 560) most significant: high of low * low
v_mul_lo_u32 v18, v15, s22 // (op 560) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 560) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 560) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 560) most significant half



s_mul_i32 s1, s22, s5 // (op 562) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 562) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 562) least significant
s_mul_hi_u32 s9, s22, s4 // (op 562) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 562) most significant: sum
s_add_u32 s1, s1, s9 // (op 562) most significant: sum


s_mov_b32 s56, s24 // (op 562) reg expression
s_mov_b32 s57, s25 // (op 562) reg expression
s_mov_b32 s58, s0 // (op 562) reg expression
s_mov_b32 s59, 131072 // (op 562) call()

s_mul_hi_i32 s0, s16, s12 // (op 605) 
v_lshrrev_b32 v16, 2, v1 // (op 605) 
s_add_i32 s1, s0, s16 // (op 605) 

v_and_b32 v17, -4, v1 // (op 605) 
s_ashr_i32 s0, s1, 31 // (op 605) 
v_sub_u32 v18, v1, v17 // (op 605) 

s_cmp_eq_i32 s12, 0 // (op 605) 
s_mov_b32 s8, scc // (op 605) 
s_cmp_eq_i32 s8, 0 // (op 605) 
s_cselect_b32 s9, 0, -1 // (op 605) 

s_and_b32 s8, s13, 31 // (op 605) 
s_lshl_b32 s18, 1, s8 // (op 605) 
s_add_i32 s19, s18, s9 // (op 605) 


s_and_b32 s9, s0, s19 // (op 605) 


s_add_i32 s0, s1, s9 // (op 605) 


s_ashr_i32 s1, s0, s8 // (op 605) 


s_lshl_b32 s0, s13, 24 // (op 605) 
s_ashr_i32 s8, s0, 31 // (op 605) 

s_xor_b32 s0, s1, s8 // (op 605) 

s_sub_i32 s1, s0, s8 // (op 605) 


s_mul_hi_i32 s0, s1, s14 // (op 605) 
s_add_i32 s8, s0, s1 // (op 605) 

s_ashr_i32 s0, s8, 31 // (op 605) 
s_cmp_eq_i32 s14, 0 // (op 605) 
s_mov_b32 s9, scc // (op 605) 
s_cmp_eq_i32 s9, 0 // (op 605) 
s_cselect_b32 s18, 0, -1 // (op 605) 

s_and_b32 s9, s15, 31 // (op 605) 
s_lshl_b32 s19, 1, s9 // (op 605) 
s_add_i32 s49, s19, s18 // (op 605) 


s_and_b32 s18, s0, s49 // (op 605) 


s_add_i32 s0, s8, s18 // (op 605) 


s_ashr_i32 s8, s0, s9 // (op 605) 


s_lshl_b32 s0, s15, 24 // (op 605) 
s_ashr_i32 s9, s0, 31 // (op 605) 

s_xor_b32 s0, s8, s9 // (op 605) 

s_sub_i32 s8, s0, s9 // (op 605) 


s_mul_i32 s0, s1, s45 // (op 605) 

s_sub_u32 s1, s16, s0 // (op 605) 

s_mul_i32 s0, s8, s45 // (op 605) 
s_add_u32 s0, s0, s1 // (op 605) 


s_mul_hi_i32 s1, s16, s11 // (op 605) 
s_add_i32 s8, s1, s16 // (op 605) 

s_ashr_i32 s1, s8, 31 // (op 605) 
s_cmp_eq_i32 s11, 0 // (op 605) 
s_mov_b32 s9, scc // (op 605) 
s_cmp_eq_i32 s9, 0 // (op 605) 
s_cselect_b32 s18, 0, -1 // (op 605) 

s_and_b32 s9, s46, 31 // (op 605) 
s_lshl_b32 s19, 1, s9 // (op 605) 
s_add_i32 s49, s19, s18 // (op 605) 


s_and_b32 s18, s1, s49 // (op 605) 


s_add_i32 s1, s8, s18 // (op 605) 


s_ashr_i32 s8, s1, s9 // (op 605) 


s_lshl_b32 s1, s46, 24 // (op 605) 
s_ashr_i32 s9, s1, 31 // (op 605) 

s_xor_b32 s1, s8, s9 // (op 605) 

s_sub_i32 s8, s1, s9 // (op 605) 


s_add_u32 s18, s4, 16 // (op 605) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 605) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 605) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 605) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 605) 

s_mul_i32 s1, s45, s18 // (op 605) 

s_mul_i32 s9, s10, s1 // (op 605) 

s_mul_i32 s1, s8, s9 // (op 605) 

s_sub_u32 s9, s16, s1 // (op 605) 

s_mov_b32 s1, s9 // (op 605) 
s_mul_hi_i32 s18, s1, s44 // (op 605) 
s_add_i32 s19, s18, s1 // (op 605) 


s_ashr_i32 s1, s19, 31 // (op 605) 
s_cmp_eq_i32 s44, 0 // (op 605) 
s_mov_b32 s18, scc // (op 605) 
s_cmp_eq_i32 s18, 0 // (op 605) 
s_cselect_b32 s49, 0, -1 // (op 605) 

s_and_b32 s18, s48, 31 // (op 605) 
s_lshl_b32 s50, 1, s18 // (op 605) 
s_add_i32 s51, s50, s49 // (op 605) 


s_and_b32 s49, s1, s51 // (op 605) 


s_add_i32 s1, s19, s49 // (op 605) 


s_ashr_i32 s19, s1, s18 // (op 605) 


s_lshl_b32 s1, s48, 24 // (op 605) 
s_ashr_i32 s18, s1, 31 // (op 605) 

s_xor_b32 s1, s19, s18 // (op 605) 

s_sub_i32 s19, s1, s18 // (op 605) 


s_mul_i32 s1, s47, s45 // (op 605) 
s_add_u32 s50, s6, 32 // (op 605) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 605) End of Int64 add, unlocking SCC
s_sub_u32 s60, s50, 1 // (op 605) least significant half; sets scc
s_subb_u32 s61, s51, 0 // (op 605) most significant half; uses scc

s_ashr_i64 s[50:51], s[60:61], 5 // (op 605) 

s_sub_i32 s18, s50, s1 // (op 605) 

s_mul_i32 s49, s19, s18 // (op 605) 


s_sub_u32 s18, s9, s49 // (op 605) 


s_add_u32 s9, s18, s1 // (op 605) 


s_cmp_eq_i32 s8, 0 // (op 605) 
s_mov_b32 s1, scc // (op 605) 

s_cmp_eq_i32 s1, 0 // (op 605) 
s_cselect_b32 s8, s9, s0 // (op 605) 



v_lshl_add_u32 v17, s8, 5, v16 // (op 605) 


v_mul_lo_u32 v16, v17, s26 // (op 605) 
v_add_u32 v16, v16, v18 // (op 605) 


s_mul_i32 s1, s26, s7 // (op 607) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 607) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 607) least significant
s_mul_hi_u32 s9, s26, s6 // (op 607) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 607) most significant: sum
s_add_u32 s1, s1, s9 // (op 607) most significant: sum


s_mov_b32 s60, s28 // (op 607) reg expression
s_mov_b32 s61, s29 // (op 607) reg expression
s_mov_b32 s62, s0 // (op 607) reg expression
s_mov_b32 s63, 131072 // (op 607) call()

s_mul_hi_i32 s0, s16, s12 // (op 624) 
v_lshrrev_b32 v17, 2, v1 // (op 624) 
s_add_i32 s1, s0, s16 // (op 624) 

v_and_b32 v18, -4, v1 // (op 624) 
s_ashr_i32 s0, s1, 31 // (op 624) 
v_sub_u32 v19, v1, v18 // (op 624) 

s_cmp_eq_i32 s12, 0 // (op 624) 
s_mov_b32 s8, scc // (op 624) 
s_cmp_eq_i32 s8, 0 // (op 624) 
s_cselect_b32 s9, 0, -1 // (op 624) 

s_and_b32 s8, s13, 31 // (op 624) 
s_lshl_b32 s18, 1, s8 // (op 624) 
s_add_i32 s19, s18, s9 // (op 624) 


s_and_b32 s9, s0, s19 // (op 624) 


s_add_i32 s0, s1, s9 // (op 624) 


s_ashr_i32 s1, s0, s8 // (op 624) 


s_lshl_b32 s0, s13, 24 // (op 624) 
s_ashr_i32 s8, s0, 31 // (op 624) 

s_xor_b32 s0, s1, s8 // (op 624) 

s_sub_i32 s1, s0, s8 // (op 624) 


s_mul_hi_i32 s0, s1, s14 // (op 624) 
s_add_i32 s8, s0, s1 // (op 624) 

s_ashr_i32 s0, s8, 31 // (op 624) 
s_cmp_eq_i32 s14, 0 // (op 624) 
s_mov_b32 s9, scc // (op 624) 
s_cmp_eq_i32 s9, 0 // (op 624) 
s_cselect_b32 s18, 0, -1 // (op 624) 

s_and_b32 s9, s15, 31 // (op 624) 
s_lshl_b32 s19, 1, s9 // (op 624) 
s_add_i32 s49, s19, s18 // (op 624) 


s_and_b32 s18, s0, s49 // (op 624) 


s_add_i32 s0, s8, s18 // (op 624) 


s_ashr_i32 s8, s0, s9 // (op 624) 


s_lshl_b32 s0, s15, 24 // (op 624) 
s_ashr_i32 s9, s0, 31 // (op 624) 

s_xor_b32 s0, s8, s9 // (op 624) 

s_sub_i32 s8, s0, s9 // (op 624) 


s_mul_i32 s0, s1, s45 // (op 624) 

s_sub_u32 s1, s16, s0 // (op 624) 

s_mul_i32 s0, s8, s45 // (op 624) 
s_add_u32 s0, s0, s1 // (op 624) 


s_mul_hi_i32 s1, s16, s11 // (op 624) 
s_add_i32 s8, s1, s16 // (op 624) 

s_ashr_i32 s1, s8, 31 // (op 624) 
s_cmp_eq_i32 s11, 0 // (op 624) 
s_mov_b32 s9, scc // (op 624) 
s_cmp_eq_i32 s9, 0 // (op 624) 
s_cselect_b32 s18, 0, -1 // (op 624) 

s_and_b32 s9, s46, 31 // (op 624) 
s_lshl_b32 s19, 1, s9 // (op 624) 
s_add_i32 s49, s19, s18 // (op 624) 


s_and_b32 s18, s1, s49 // (op 624) 


s_add_i32 s1, s8, s18 // (op 624) 


s_ashr_i32 s8, s1, s9 // (op 624) 


s_lshl_b32 s1, s46, 24 // (op 624) 
s_ashr_i32 s9, s1, 31 // (op 624) 

s_xor_b32 s1, s8, s9 // (op 624) 

s_sub_i32 s8, s1, s9 // (op 624) 


s_add_u32 s18, s4, 16 // (op 624) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 624) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 624) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 624) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 624) 

s_mul_i32 s1, s45, s18 // (op 624) 

s_mul_i32 s9, s10, s1 // (op 624) 

s_mul_i32 s1, s8, s9 // (op 624) 

s_sub_u32 s9, s16, s1 // (op 624) 

s_mov_b32 s1, s9 // (op 624) 
s_mul_hi_i32 s18, s1, s44 // (op 624) 
s_add_i32 s19, s18, s1 // (op 624) 


s_ashr_i32 s1, s19, 31 // (op 624) 
s_cmp_eq_i32 s44, 0 // (op 624) 
s_mov_b32 s18, scc // (op 624) 
s_cmp_eq_i32 s18, 0 // (op 624) 
s_cselect_b32 s49, 0, -1 // (op 624) 

s_and_b32 s18, s48, 31 // (op 624) 
s_lshl_b32 s50, 1, s18 // (op 624) 
s_add_i32 s51, s50, s49 // (op 624) 


s_and_b32 s49, s1, s51 // (op 624) 


s_add_i32 s1, s19, s49 // (op 624) 


s_ashr_i32 s19, s1, s18 // (op 624) 


s_lshl_b32 s1, s48, 24 // (op 624) 
s_ashr_i32 s18, s1, 31 // (op 624) 

s_xor_b32 s1, s19, s18 // (op 624) 

s_sub_i32 s19, s1, s18 // (op 624) 


s_mul_i32 s1, s47, s45 // (op 624) 
s_add_u32 s50, s6, 32 // (op 624) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 624) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 624) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 624) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 624) 

s_sub_i32 s18, s50, s1 // (op 624) 

s_mul_i32 s49, s19, s18 // (op 624) 


s_sub_u32 s18, s9, s49 // (op 624) 


s_add_u32 s9, s18, s1 // (op 624) 


s_cmp_eq_i32 s8, 0 // (op 624) 
s_mov_b32 s1, scc // (op 624) 

s_cmp_eq_i32 s1, 0 // (op 624) 
s_cselect_b32 s8, s9, s0 // (op 624) 



v_lshl_add_u32 v18, s8, 5, v17 // (op 624) 


v_mul_lo_u32 v17, v18, s26 // (op 624) 
v_add_u32 v17, v17, v19 // (op 624) 


s_mul_i32 s1, s26, s7 // (op 626) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 626) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 626) least significant
s_mul_hi_u32 s9, s26, s6 // (op 626) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 626) most significant: sum
s_add_u32 s1, s1, s9 // (op 626) most significant: sum


s_mov_b32 s60, s28 // (op 626) reg expression
s_mov_b32 s61, s29 // (op 626) reg expression
s_mov_b32 s62, s0 // (op 626) reg expression
s_mov_b32 s63, 131072 // (op 626) call()

s_mul_hi_i32 s0, s16, s12 // (op 669) 
v_and_b32 v18, 15, v1 // (op 669) 
s_add_i32 s1, s0, s16 // (op 669) 

v_and_b32 v19, 63, v1 // (op 669) 
s_ashr_i32 s0, s1, 31 // (op 669) 
v_lshrrev_b32 v20, 4, v19 // (op 669) 

s_cmp_eq_i32 s12, 0 // (op 669) 
s_mov_b32 s8, scc // (op 669) 
s_cmp_eq_i32 s8, 0 // (op 669) 
s_cselect_b32 s9, 0, -1 // (op 669) 

s_and_b32 s8, s13, 31 // (op 669) 
s_lshl_b32 s18, 1, s8 // (op 669) 
s_add_i32 s19, s18, s9 // (op 669) 


s_and_b32 s9, s0, s19 // (op 669) 


s_add_i32 s0, s1, s9 // (op 669) 


s_ashr_i32 s1, s0, s8 // (op 669) 


s_lshl_b32 s0, s13, 24 // (op 669) 
s_ashr_i32 s8, s0, 31 // (op 669) 

s_xor_b32 s0, s1, s8 // (op 669) 

s_sub_i32 s1, s0, s8 // (op 669) 


s_mul_hi_i32 s0, s1, s14 // (op 669) 
s_add_i32 s8, s0, s1 // (op 669) 

s_ashr_i32 s0, s8, 31 // (op 669) 
s_cmp_eq_i32 s14, 0 // (op 669) 
s_mov_b32 s9, scc // (op 669) 
s_cmp_eq_i32 s9, 0 // (op 669) 
s_cselect_b32 s18, 0, -1 // (op 669) 

s_and_b32 s9, s15, 31 // (op 669) 
s_lshl_b32 s19, 1, s9 // (op 669) 
s_add_i32 s49, s19, s18 // (op 669) 


s_and_b32 s18, s0, s49 // (op 669) 


s_add_i32 s0, s8, s18 // (op 669) 


s_ashr_i32 s8, s0, s9 // (op 669) 


s_lshl_b32 s0, s15, 24 // (op 669) 
s_ashr_i32 s9, s0, 31 // (op 669) 

s_xor_b32 s0, s8, s9 // (op 669) 

s_sub_i32 s8, s0, s9 // (op 669) 


s_add_u32 s18, s4, 16 // (op 669) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 669) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 669) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 669) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 669) 

s_mul_i32 s0, s8, s18 // (op 669) 

s_sub_i32 s8, s1, s0 // (op 669) 


s_mul_hi_i32 s9, s16, s11 // (op 669) 
s_add_i32 s0, s9, s16 // (op 669) 

s_ashr_i32 s1, s0, 31 // (op 669) 
s_cmp_eq_i32 s11, 0 // (op 669) 
s_mov_b32 s9, scc // (op 669) 
s_cmp_eq_i32 s9, 0 // (op 669) 
s_cselect_b32 s49, 0, -1 // (op 669) 

s_and_b32 s9, s46, 31 // (op 669) 
s_lshl_b32 s50, 1, s9 // (op 669) 
s_add_i32 s51, s50, s49 // (op 669) 


s_and_b32 s49, s1, s51 // (op 669) 


s_add_i32 s1, s0, s49 // (op 669) 


s_ashr_i32 s0, s1, s9 // (op 669) 


s_lshl_b32 s1, s46, 24 // (op 669) 
s_ashr_i32 s9, s1, 31 // (op 669) 

s_xor_b32 s1, s0, s9 // (op 669) 

s_sub_i32 s0, s1, s9 // (op 669) 


s_mul_i32 s1, s45, s18 // (op 669) 

s_mul_i32 s9, s10, s1 // (op 669) 

s_mul_i32 s1, s0, s9 // (op 669) 

s_sub_u32 s9, s16, s1 // (op 669) 

s_mul_hi_i32 s1, s9, s44 // (op 669) 
s_add_i32 s18, s1, s9 // (op 669) 


s_ashr_i32 s1, s18, 31 // (op 669) 
s_cmp_eq_i32 s44, 0 // (op 669) 
s_mov_b32 s9, scc // (op 669) 
s_cmp_eq_i32 s9, 0 // (op 669) 
s_cselect_b32 s19, 0, -1 // (op 669) 

s_and_b32 s9, s48, 31 // (op 669) 
s_lshl_b32 s49, 1, s9 // (op 669) 
s_add_i32 s50, s49, s19 // (op 669) 


s_and_b32 s19, s1, s50 // (op 669) 


s_add_i32 s1, s18, s19 // (op 669) 


s_ashr_i32 s18, s1, s9 // (op 669) 


s_lshl_b32 s1, s48, 24 // (op 669) 
s_ashr_i32 s9, s1, 31 // (op 669) 

s_xor_b32 s1, s18, s9 // (op 669) 

s_sub_i32 s18, s1, s9 // (op 669) 


s_cmp_eq_i32 s0, 0 // (op 669) 
s_mov_b32 s1, scc // (op 669) 

s_cmp_eq_i32 s1, 0 // (op 669) 
s_cselect_b32 s0, s18, s8 // (op 669) 



v_lshl_add_u32 v19, s0, 4, v18 // (op 669) 


v_mul_lo_u32 v18, v19, s30 // (op 669) 

v_add_u32 v19, v20, v18 // (op 669) 


s_mul_i32 s1, s30, s5 // (op 671) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 671) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 671) least significant
s_mul_hi_u32 s9, s30, s4 // (op 671) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 671) most significant: sum
s_add_u32 s1, s1, s9 // (op 671) most significant: sum


s_mov_b32 s64, s32 // (op 671) reg expression
s_mov_b32 s65, s33 // (op 671) reg expression
s_mov_b32 s66, s0 // (op 671) reg expression
s_mov_b32 s67, 131072 // (op 671) call()

s_mul_hi_i32 s0, s16, s12 // (op 688) 
v_and_b32 v18, 15, v1 // (op 688) 
s_add_i32 s1, s0, s16 // (op 688) 

v_and_b32 v20, 63, v1 // (op 688) 
s_ashr_i32 s0, s1, 31 // (op 688) 
v_lshrrev_b32 v21, 4, v20 // (op 688) 

s_cmp_eq_i32 s12, 0 // (op 688) 
s_mov_b32 s8, scc // (op 688) 
s_cmp_eq_i32 s8, 0 // (op 688) 
s_cselect_b32 s9, 0, -1 // (op 688) 

s_and_b32 s8, s13, 31 // (op 688) 
s_lshl_b32 s18, 1, s8 // (op 688) 
s_add_i32 s19, s18, s9 // (op 688) 


s_and_b32 s9, s0, s19 // (op 688) 


s_add_i32 s0, s1, s9 // (op 688) 


s_ashr_i32 s1, s0, s8 // (op 688) 


s_lshl_b32 s0, s13, 24 // (op 688) 
s_ashr_i32 s8, s0, 31 // (op 688) 

s_xor_b32 s0, s1, s8 // (op 688) 

s_sub_i32 s1, s0, s8 // (op 688) 


s_mul_hi_i32 s0, s1, s14 // (op 688) 
s_add_i32 s8, s0, s1 // (op 688) 

s_ashr_i32 s0, s8, 31 // (op 688) 
s_cmp_eq_i32 s14, 0 // (op 688) 
s_mov_b32 s9, scc // (op 688) 
s_cmp_eq_i32 s9, 0 // (op 688) 
s_cselect_b32 s18, 0, -1 // (op 688) 

s_and_b32 s9, s15, 31 // (op 688) 
s_lshl_b32 s19, 1, s9 // (op 688) 
s_add_i32 s49, s19, s18 // (op 688) 


s_and_b32 s18, s0, s49 // (op 688) 


s_add_i32 s0, s8, s18 // (op 688) 


s_ashr_i32 s8, s0, s9 // (op 688) 


s_lshl_b32 s0, s15, 24 // (op 688) 
s_ashr_i32 s9, s0, 31 // (op 688) 

s_xor_b32 s0, s8, s9 // (op 688) 

s_sub_i32 s8, s0, s9 // (op 688) 


s_add_u32 s18, s4, 16 // (op 688) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 688) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 688) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 688) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 688) 

s_mul_i32 s0, s8, s18 // (op 688) 

s_sub_i32 s8, s1, s0 // (op 688) 


s_mul_hi_i32 s9, s16, s11 // (op 688) 
s_add_i32 s0, s9, s16 // (op 688) 

s_ashr_i32 s1, s0, 31 // (op 688) 
s_cmp_eq_i32 s11, 0 // (op 688) 
s_mov_b32 s9, scc // (op 688) 
s_cmp_eq_i32 s9, 0 // (op 688) 
s_cselect_b32 s49, 0, -1 // (op 688) 

s_and_b32 s9, s46, 31 // (op 688) 
s_lshl_b32 s50, 1, s9 // (op 688) 
s_add_i32 s51, s50, s49 // (op 688) 


s_and_b32 s49, s1, s51 // (op 688) 


s_add_i32 s1, s0, s49 // (op 688) 


s_ashr_i32 s0, s1, s9 // (op 688) 


s_lshl_b32 s1, s46, 24 // (op 688) 
s_ashr_i32 s9, s1, 31 // (op 688) 

s_xor_b32 s1, s0, s9 // (op 688) 

s_sub_i32 s0, s1, s9 // (op 688) 


s_mul_i32 s1, s45, s18 // (op 688) 

s_mul_i32 s9, s10, s1 // (op 688) 

s_mul_i32 s1, s0, s9 // (op 688) 

s_sub_u32 s9, s16, s1 // (op 688) 

s_mul_hi_i32 s1, s9, s44 // (op 688) 
s_add_i32 s18, s1, s9 // (op 688) 


s_ashr_i32 s1, s18, 31 // (op 688) 
s_cmp_eq_i32 s44, 0 // (op 688) 
s_mov_b32 s9, scc // (op 688) 
s_cmp_eq_i32 s9, 0 // (op 688) 
s_cselect_b32 s19, 0, -1 // (op 688) 

s_and_b32 s9, s48, 31 // (op 688) 
s_lshl_b32 s49, 1, s9 // (op 688) 
s_add_i32 s50, s49, s19 // (op 688) 


s_and_b32 s19, s1, s50 // (op 688) 


s_add_i32 s1, s18, s19 // (op 688) 


s_ashr_i32 s18, s1, s9 // (op 688) 


s_lshl_b32 s1, s48, 24 // (op 688) 
s_ashr_i32 s9, s1, 31 // (op 688) 

s_xor_b32 s1, s18, s9 // (op 688) 

s_sub_i32 s18, s1, s9 // (op 688) 


s_cmp_eq_i32 s0, 0 // (op 688) 
s_mov_b32 s1, scc // (op 688) 

s_cmp_eq_i32 s1, 0 // (op 688) 
s_cselect_b32 s0, s18, s8 // (op 688) 



v_lshl_add_u32 v20, s0, 4, v18 // (op 688) 


v_mul_lo_u32 v18, v20, s30 // (op 688) 

v_add_u32 v20, v21, v18 // (op 688) 


s_mul_i32 s1, s30, s5 // (op 690) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 690) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 690) least significant
s_mul_hi_u32 s9, s30, s4 // (op 690) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 690) most significant: sum
s_add_u32 s1, s1, s9 // (op 690) most significant: sum


s_mov_b32 s64, s32 // (op 690) reg expression
s_mov_b32 s65, s33 // (op 690) reg expression
s_mov_b32 s66, s0 // (op 690) reg expression
s_mov_b32 s67, 131072 // (op 690) call()




buffer_load_ubyte v18, v19, s[64:67], 0 offen offset: 0 // (op 341) Load value
v_and_b32 v21, -8, v1 // (op 807) 
v_sub_u32 v22, v1, v21 // (op 807) 

v_lshlrev_b32 v21, 4, v22 // (op 807) 

v_lshrrev_b32 v22, 3, v1 // (op 807) 
v_lshl_add_u32 v23, v22, 7, v21 // (op 807) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 807) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 433) reg expression
buffer_load_dwordx4 v12, s[56:59], 0 offen offset: 0 lds // (op 433) Load value direct to lds




buffer_load_dwordx4 v[22:25], v10, s[52:55], 0 offen // (op 348) Load value
buffer_load_dwordx2 v[26:27], v10, s[52:55], 0 offen offset:16 // (op 348) Load value



buffer_load_ubyte v21, v16, s[60:63], 0 offen offset: 0 // (op 351) Load value



buffer_load_ubyte v28, v19, s[64:67], 0 offen offset:4 // (op 355) Load value

v_and_b32 v19, -8, v1 // (op 957) 
v_sub_u32 v29, v1, v19 // (op 957) 

v_lshlrev_b32 v19, 4, v29 // (op 957) 

v_lshrrev_b32 v29, 3, v1 // (op 957) 
v_lshl_add_u32 v30, v29, 7, v19 // (op 957) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 957) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 2048, s0 // (op 438) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 438) reg expression
s_mov_b32 s8, 128 // (op 438) call()
buffer_load_dwordx4 v12, s[56:59], s8 offen offset: 0 lds // (op 438) Load value direct to lds







buffer_load_dwordx4 v[30:33], v10, s[52:55], 0 offen offset:96 // (op 362) Load value
buffer_load_dwordx2 v[34:35], v10, s[52:55], 0 offen offset:112 // (op 362) Load value




buffer_load_ubyte v10, v16, s[60:63], 0 offen offset:4 // (op 365) Load value

v_and_b32 v16, -4, v1 // (op 755) 
v_sub_u32 v19, v1, v16 // (op 755) 

v_lshlrev_b32 v16, 5, v19 // (op 755) 

v_lshrrev_b32 v19, 2, v1 // (op 755) 
v_lshl_add_u32 v29, v19, 7, v16 // (op 755) 


v_mul_lo_u32 v16, v29, 6 // (op 755) 

v_lshrrev_b32 v19, 3, v16 // (op 755) 




s_waitcnt vmcnt(7)
ds_write_b128 v19, v[22:25] offset:4096 // (op 368) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v19, v[26:27] offset:4112 // (op 368) Store local data 

v_and_b32 v16, -4, v1 // (op 860) 
v_sub_u32 v19, v1, v16 // (op 860) 

v_and_b32 v16, 4294967292, v1 // (op 860) 
v_add_u32 v29, v16, v19 // (op 860) 





s_waitcnt vmcnt(5)
ds_write_b8 v29, v21 offset:7168 // (op 369) Store local data 

v_and_b32 v16, -4, v1 // (op 726) 
v_sub_u32 v19, v1, v16 // (op 726) 

v_lshlrev_b32 v16, 5, v19 // (op 726) 

v_lshrrev_b32 v19, 2, v1 // (op 726) 
v_lshl_add_u32 v29, v19, 7, v16 // (op 726) 


v_mul_lo_u32 v16, v29, 6 // (op 726) 

v_lshrrev_b32 v19, 3, v16 // (op 726) 

v_and_b32 v16, -8, v1 // (op 779) 
v_sub_u32 v29, v1, v16 // (op 779) 

v_lshlrev_b32 v16, 4, v29 // (op 779) 

v_lshrrev_b32 v29, 3, v1 // (op 779) 
v_lshl_add_u32 v12, v29, 7, v16 // (op 779) 


s_nop 0
v_readfirstlane_b32 s0, v12 // (op 779) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 831) 
v_sub_u32 v29, v1, v16 // (op 831) 

v_and_b32 v16, 4294967292, v1 // (op 831) 
v_add_u32 v12, v16, v29 // (op 831) 


v_and_b32 v13, -4, v1 // (op 918) 
v_sub_u32 v16, v1, v13 // (op 918) 

v_lshlrev_b32 v13, 5, v16 // (op 918) 

v_lshrrev_b32 v16, 2, v1 // (op 918) 
v_lshl_add_u32 v29, v16, 7, v13 // (op 918) 


v_mul_lo_u32 v13, v29, 6 // (op 918) 

v_lshrrev_b32 v16, 3, v13 // (op 918) 

v_and_b32 v13, -8, v1 // (op 942) 
v_sub_u32 v29, v1, v13 // (op 942) 

v_lshlrev_b32 v13, 4, v29 // (op 942) 

v_lshrrev_b32 v29, 3, v1 // (op 942) 
v_lshl_add_u32 v36, v29, 7, v13 // (op 942) 


s_nop 0
v_readfirstlane_b32 s1, v36 // (op 942) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -4, v1 // (op 981) 
v_sub_u32 v29, v1, v13 // (op 981) 

v_and_b32 v13, 4294967292, v1 // (op 981) 
v_add_u32 v36, v13, v29 // (op 981) 


s_mov_b32 s8, 0 // (op 57) call()
s_ashr_i64 s[18:19], s[34:35], 7 // (op 56) 
s_mov_b32 s9, s18 // (op 56) convert

s_and_b32 s18, s9, 4294967294 // (op 56) 

s_cmp_lt_u32 s8, s18 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_56 if false)
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_56:
s_add_u32 s8, s8, 2 // (op 58) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1026) 



ds_read_b128 v[38:41], v3 offset:4096 // (op 67) Load local data 
ds_read_b64 v[42:43], v3 offset:4112 // (op 67) Load local data 
s_barrier  // (op 1039) 



ds_read_b128 v[44:47], v5 // (op 72) Load local data 
ds_read_b128 v[48:51], v5 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1052) 



ds_read_u8 v13, v0 offset:7168 // (op 77) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:43], v[44:51], a[0:3], v13, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 84) 





v_mov_b32 v13, v20 // (op 82) 

buffer_load_ubyte v18, v13, s[64:67], 0 offen offset:8 // (op 82) Load value
s_barrier  // (op 1042) 


v_mov_b64 v[38:39], v[14:15] // (op 416) 

s_mov_b32 m0, s0 // (op 416) reg expression
s_mov_b32 s9, 256 // (op 416) call()
buffer_load_dwordx4 v38, s[56:59], s9 offen offset: 0 lds // (op 416) Load value direct to lds



v_mov_b32 v29, v11 // (op 4) 

buffer_load_dwordx4 v[22:25], v29, s[52:55], 0 offen offset:192 // (op 4) Load value
buffer_load_dwordx2 v[26:27], v29, s[52:55], 0 offen offset:208 // (op 4) Load value


v_mov_b32 v37, v17 // (op 17) 

buffer_load_ubyte v21, v37, s[60:63], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 1069) 



s_waitcnt vmcnt(7)
ds_write_b128 v16, v[30:33] offset:7296 // (op 229) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v16, v[34:35] offset:7312 // (op 229) Store local data 
s_barrier  // (op 1081) 



s_waitcnt vmcnt(5)
ds_write_b8 v36, v10 offset:10368 // (op 240) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1066) 



ds_read_b128 v[40:43], v6 offset:7296 // (op 231) Load local data 
ds_read_b64 v[44:45], v6 offset:7312 // (op 231) Load local data 
s_barrier  // (op 1072) 
s_barrier  // (op 1132) 



ds_read_b128 v[46:49], v8 offset:2048 // (op 233) Load local data 
ds_read_b128 v[50:53], v8 offset:2112 // (op 233) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1078) 



ds_read_u8 v54, v4 offset:10368 // (op 235) Load local data 
s_barrier  // (op 1140) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:45], v[46:53], a[0:3], v54, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 238) 






buffer_load_ubyte v28, v13, s[64:67], 0 offen offset:12 // (op 237) Load value

s_barrier  // (op 1075) 
v_add_u32 v20, 8, v20 // (op 680) 


s_add_u32 s9, 2048, s1 // (op 428) 

s_mov_b32 m0, s9 // (op 428) reg expression
s_mov_b32 s18, 384 // (op 428) call()
buffer_load_dwordx4 v38, s[56:59], s18 offen offset: 0 lds // (op 428) Load value direct to lds






v_add_co_u32 v14, vcc, 256, v14 // (op 552) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 552) most significant half
buffer_load_dwordx4 v[30:33], v29, s[52:55], 0 offen offset:288 // (op 228) Load value
buffer_load_dwordx2 v[34:35], v29, s[52:55], 0 offen offset:304 // (op 228) Load value




v_add_u32 v11, 192, v11 // (op 488) 
buffer_load_ubyte v10, v37, s[60:63], 0 offen offset:12 // (op 242) Load value

s_barrier  // (op 1029) 
v_add_u32 v17, 8, v17 // (op 616) 



s_waitcnt vmcnt(7)
ds_write_b128 v19, v[22:25] offset:4096 // (op 6) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v19, v[26:27] offset:4112 // (op 6) Store local data 
s_barrier  // (op 1055) 



s_waitcnt vmcnt(5)
ds_write_b8 v12, v21 offset:7168 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[34:35], 7 // (op 56) 
s_mov_b32 s9, s18 // (op 56) convert

s_and_b32 s18, s9, 4294967294 // (op 56) 

s_cmp_lt_u32 s8, s18 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_56 if true)
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_56:














s_barrier  // (op 1094) 
s_mul_hi_i32 s0, s16, s12 // (op 451) 
v_lshrrev_b32 v29, 2, v1 // (op 451) 
s_add_i32 s1, s0, s16 // (op 451) 

v_and_b32 v11, -4, v1 // (op 451) 
s_ashr_i32 s0, s1, 31 // (op 451) 
v_sub_u32 v12, v1, v11 // (op 451) 

s_cmp_eq_i32 s12, 0 // (op 451) 
s_mov_b32 s8, scc // (op 451) 
v_lshlrev_b32 v11, 5, v12 // (op 451) 

s_cmp_eq_i32 s8, 0 // (op 451) 
s_cselect_b32 s9, 0, -1 // (op 451) 

s_and_b32 s8, s13, 31 // (op 451) 
s_lshl_b32 s18, 1, s8 // (op 451) 
s_add_i32 s19, s18, s9 // (op 451) 


s_and_b32 s9, s0, s19 // (op 451) 


s_add_i32 s0, s1, s9 // (op 451) 


s_ashr_i32 s1, s0, s8 // (op 451) 


s_lshl_b32 s0, s13, 24 // (op 451) 
s_ashr_i32 s8, s0, 31 // (op 451) 

s_xor_b32 s0, s1, s8 // (op 451) 

s_sub_i32 s1, s0, s8 // (op 451) 


s_mul_hi_i32 s0, s1, s14 // (op 451) 
s_add_i32 s8, s0, s1 // (op 451) 

s_ashr_i32 s0, s8, 31 // (op 451) 
s_cmp_eq_i32 s14, 0 // (op 451) 
s_mov_b32 s9, scc // (op 451) 
s_cmp_eq_i32 s9, 0 // (op 451) 
s_cselect_b32 s18, 0, -1 // (op 451) 

s_and_b32 s9, s15, 31 // (op 451) 
s_lshl_b32 s19, 1, s9 // (op 451) 
s_add_i32 s49, s19, s18 // (op 451) 


s_and_b32 s18, s0, s49 // (op 451) 


s_add_i32 s0, s8, s18 // (op 451) 


s_ashr_i32 s8, s0, s9 // (op 451) 


s_lshl_b32 s0, s15, 24 // (op 451) 
s_ashr_i32 s9, s0, 31 // (op 451) 

s_xor_b32 s0, s8, s9 // (op 451) 

s_sub_i32 s8, s0, s9 // (op 451) 


s_mul_i32 s0, s1, s45 // (op 451) 

s_sub_u32 s1, s16, s0 // (op 451) 

s_mul_i32 s0, s8, s45 // (op 451) 
s_add_u32 s0, s0, s1 // (op 451) 


s_mul_hi_i32 s1, s16, s11 // (op 451) 
s_add_i32 s8, s1, s16 // (op 451) 

s_ashr_i32 s1, s8, 31 // (op 451) 
s_cmp_eq_i32 s11, 0 // (op 451) 
s_mov_b32 s9, scc // (op 451) 
s_cmp_eq_i32 s9, 0 // (op 451) 
s_cselect_b32 s18, 0, -1 // (op 451) 

s_and_b32 s9, s46, 31 // (op 451) 
s_lshl_b32 s19, 1, s9 // (op 451) 
s_add_i32 s49, s19, s18 // (op 451) 


s_and_b32 s18, s1, s49 // (op 451) 


s_add_i32 s1, s8, s18 // (op 451) 


s_ashr_i32 s8, s1, s9 // (op 451) 


s_lshl_b32 s1, s46, 24 // (op 451) 
s_ashr_i32 s9, s1, 31 // (op 451) 

s_xor_b32 s1, s8, s9 // (op 451) 

s_sub_i32 s8, s1, s9 // (op 451) 


s_add_u32 s18, s4, 16 // (op 451) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 451) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 451) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 451) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 451) 

s_mul_i32 s1, s45, s18 // (op 451) 

s_mul_i32 s9, s10, s1 // (op 451) 

s_mul_i32 s1, s8, s9 // (op 451) 

s_sub_u32 s9, s16, s1 // (op 451) 

s_mov_b32 s1, s9 // (op 451) 
s_mul_hi_i32 s18, s1, s44 // (op 451) 
s_add_i32 s19, s18, s1 // (op 451) 


s_ashr_i32 s1, s19, 31 // (op 451) 
s_cmp_eq_i32 s44, 0 // (op 451) 
s_mov_b32 s18, scc // (op 451) 
s_cmp_eq_i32 s18, 0 // (op 451) 
s_cselect_b32 s49, 0, -1 // (op 451) 

s_and_b32 s18, s48, 31 // (op 451) 
s_lshl_b32 s50, 1, s18 // (op 451) 
s_add_i32 s51, s50, s49 // (op 451) 


s_and_b32 s49, s1, s51 // (op 451) 


s_add_i32 s1, s19, s49 // (op 451) 


s_ashr_i32 s19, s1, s18 // (op 451) 


s_lshl_b32 s1, s48, 24 // (op 451) 
s_ashr_i32 s18, s1, 31 // (op 451) 

s_xor_b32 s1, s19, s18 // (op 451) 

s_sub_i32 s19, s1, s18 // (op 451) 


s_mul_i32 s1, s47, s45 // (op 451) 
s_add_u32 s50, s6, 32 // (op 451) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 451) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 451) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 451) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 451) 

s_sub_i32 s18, s50, s1 // (op 451) 

s_mul_i32 s49, s19, s18 // (op 451) 


s_sub_u32 s18, s9, s49 // (op 451) 


s_add_u32 s9, s18, s1 // (op 451) 


s_cmp_eq_i32 s8, 0 // (op 451) 
s_mov_b32 s1, scc // (op 451) 

s_cmp_eq_i32 s1, 0 // (op 451) 
s_cselect_b32 s8, s9, s0 // (op 451) 



v_lshl_add_u32 v12, s8, 5, v29 // (op 451) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 451) 
s_lshr_b32 s8, s0, 1 // (op 451) 

v_lshl_add_u32 v29, s8, 8, v11 // (op 451) 


v_mov_b32 v27, 0 // (op 451) convert
v_mov_b32 v26, v29 // (op 451) convert

v_mul_lo_u32 v25, v12, s3 // (op 451) most significant: low of low * high
v_mul_hi_u32 v11, v12, s2 // (op 451) most significant: high of low * low
v_mul_lo_u32 v24, v12, s2 // (op 451) least significant: low of low * low
v_add_u32 v25, v25, v11 // (op 451) most significant: sum

v_add_co_u32 v24, s[0:1], v24, v26 // (op 451) least significant half
v_addc_co_u32 v25, s[0:1], v25, v27, s[0:1] // (op 451) most significant half



v_mul_lo_u32 v27, v25, 6 // (op 451) most significant: low of high * low
v_mul_hi_u32 v29, v24, 6 // (op 451) most significant: high of low * low
v_mul_lo_u32 v26, v24, 6 // (op 451) least significant: low of low * low
v_add_u32 v27, v27, v29 // (op 451) most significant: sum


v_lshrrev_b64 v[24:25], 3, v[26:27] // (op 451) 

v_mov_b32 v29, v24 // (op 451) convert

s_mul_i32 s1, s7, 6 // (op 453) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 453) least significant
s_mul_hi_u32 s8, s6, 6 // (op 453) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 453) most significant: sum

s_mul_i32 s9, s2, s1 // (op 453) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 453) most significant: low of high * low
s_mul_i32 s8, s2, s0 // (op 453) least significant
s_mul_hi_u32 s19, s2, s0 // (op 453) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 453) most significant: sum
s_add_u32 s9, s9, s19 // (op 453) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 453) 

s_mov_b32 s52, s20 // (op 453) reg expression
s_mov_b32 s53, s21 // (op 453) reg expression
s_mov_b32 s54, s0 // (op 453) reg expression
s_mov_b32 s55, 131072 // (op 453) call()

s_mul_hi_i32 s0, s16, s12 // (op 518) 
v_lshrrev_b32 v11, 3, v1 // (op 518) 
s_add_i32 s1, s0, s16 // (op 518) 

v_and_b32 v12, -8, v1 // (op 518) 
s_ashr_i32 s0, s1, 31 // (op 518) 
v_sub_u32 v13, v1, v12 // (op 518) 

s_cmp_eq_i32 s12, 0 // (op 518) 
s_mov_b32 s8, scc // (op 518) 
v_lshlrev_b32 v12, 4, v13 // (op 518) 

s_cmp_eq_i32 s8, 0 // (op 518) 
s_cselect_b32 s9, 0, -1 // (op 518) 

s_and_b32 s8, s13, 31 // (op 518) 
s_lshl_b32 s18, 1, s8 // (op 518) 
s_add_i32 s19, s18, s9 // (op 518) 


s_and_b32 s9, s0, s19 // (op 518) 


s_add_i32 s0, s1, s9 // (op 518) 


s_ashr_i32 s1, s0, s8 // (op 518) 


s_lshl_b32 s0, s13, 24 // (op 518) 
s_ashr_i32 s8, s0, 31 // (op 518) 

s_xor_b32 s0, s1, s8 // (op 518) 

s_sub_i32 s1, s0, s8 // (op 518) 


s_mul_hi_i32 s0, s1, s14 // (op 518) 
s_add_i32 s8, s0, s1 // (op 518) 

s_ashr_i32 s0, s8, 31 // (op 518) 
s_cmp_eq_i32 s14, 0 // (op 518) 
s_mov_b32 s9, scc // (op 518) 
s_cmp_eq_i32 s9, 0 // (op 518) 
s_cselect_b32 s18, 0, -1 // (op 518) 

s_and_b32 s9, s15, 31 // (op 518) 
s_lshl_b32 s19, 1, s9 // (op 518) 
s_add_i32 s49, s19, s18 // (op 518) 


s_and_b32 s18, s0, s49 // (op 518) 


s_add_i32 s0, s8, s18 // (op 518) 


s_ashr_i32 s8, s0, s9 // (op 518) 


s_lshl_b32 s0, s15, 24 // (op 518) 
s_ashr_i32 s9, s0, 31 // (op 518) 

s_xor_b32 s0, s8, s9 // (op 518) 

s_sub_i32 s8, s0, s9 // (op 518) 


s_add_u32 s18, s4, 16 // (op 518) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 518) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 518) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 518) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 518) 

s_mul_i32 s0, s8, s18 // (op 518) 

s_sub_i32 s8, s1, s0 // (op 518) 


s_mul_hi_i32 s9, s16, s11 // (op 518) 
s_add_i32 s0, s9, s16 // (op 518) 

s_ashr_i32 s1, s0, 31 // (op 518) 
s_cmp_eq_i32 s11, 0 // (op 518) 
s_mov_b32 s9, scc // (op 518) 
s_cmp_eq_i32 s9, 0 // (op 518) 
s_cselect_b32 s49, 0, -1 // (op 518) 

s_and_b32 s9, s46, 31 // (op 518) 
s_lshl_b32 s50, 1, s9 // (op 518) 
s_add_i32 s51, s50, s49 // (op 518) 


s_and_b32 s49, s1, s51 // (op 518) 


s_add_i32 s1, s0, s49 // (op 518) 


s_ashr_i32 s0, s1, s9 // (op 518) 


s_lshl_b32 s1, s46, 24 // (op 518) 
s_ashr_i32 s9, s1, 31 // (op 518) 

s_xor_b32 s1, s0, s9 // (op 518) 

s_sub_i32 s0, s1, s9 // (op 518) 


s_mul_i32 s1, s45, s18 // (op 518) 

s_mul_i32 s9, s10, s1 // (op 518) 

s_mul_i32 s1, s0, s9 // (op 518) 

s_sub_u32 s9, s16, s1 // (op 518) 

s_mul_hi_i32 s1, s9, s44 // (op 518) 
s_add_i32 s18, s1, s9 // (op 518) 


s_ashr_i32 s1, s18, 31 // (op 518) 
s_cmp_eq_i32 s44, 0 // (op 518) 
s_mov_b32 s9, scc // (op 518) 
s_cmp_eq_i32 s9, 0 // (op 518) 
s_cselect_b32 s19, 0, -1 // (op 518) 

s_and_b32 s9, s48, 31 // (op 518) 
s_lshl_b32 s49, 1, s9 // (op 518) 
s_add_i32 s50, s49, s19 // (op 518) 


s_and_b32 s19, s1, s50 // (op 518) 


s_add_i32 s1, s18, s19 // (op 518) 


s_ashr_i32 s18, s1, s9 // (op 518) 


s_lshl_b32 s1, s48, 24 // (op 518) 
s_ashr_i32 s9, s1, 31 // (op 518) 

s_xor_b32 s1, s18, s9 // (op 518) 

s_sub_i32 s18, s1, s9 // (op 518) 


s_cmp_eq_i32 s0, 0 // (op 518) 
s_mov_b32 s1, scc // (op 518) 

s_cmp_eq_i32 s1, 0 // (op 518) 
s_cselect_b32 s0, s18, s8 // (op 518) 



v_lshl_add_u32 v13, s0, 4, v11 // (op 518) 


v_mul_lo_u32 v15, v13, s23 // (op 518) most significant: low of low * high
v_mul_hi_u32 v11, v13, s22 // (op 518) most significant: high of low * low
v_mul_lo_u32 v14, v13, s22 // (op 518) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 518) most significant: sum


s_ashr_i64 s[0:1], s[34:35], 7 // (op 518) 
s_lshr_b32 s8, s0, 1 // (op 518) 

v_lshl_add_u32 v11, s8, 8, v12 // (op 518) 


v_mov_b32 v13, 0 // (op 518) convert
v_mov_b32 v12, v11 // (op 518) convert

v_add_co_u32 v16, s[0:1], v12, v14 // (op 518) least significant half
v_addc_co_u32 v17, s[0:1], v13, v15, s[0:1] // (op 518) most significant half



s_mul_i32 s1, s22, s5 // (op 520) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 520) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 520) least significant
s_mul_hi_u32 s9, s22, s4 // (op 520) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 520) most significant: sum
s_add_u32 s1, s1, s9 // (op 520) most significant: sum


s_mov_b32 s56, s24 // (op 520) reg expression
s_mov_b32 s57, s25 // (op 520) reg expression
s_mov_b32 s58, s0 // (op 520) reg expression
s_mov_b32 s59, 131072 // (op 520) call()

s_mul_hi_i32 s0, s16, s12 // (op 582) 
v_lshrrev_b32 v11, 2, v1 // (op 582) 
s_add_i32 s1, s0, s16 // (op 582) 

v_and_b32 v12, -4, v1 // (op 582) 
s_ashr_i32 s0, s1, 31 // (op 582) 
v_sub_u32 v13, v1, v12 // (op 582) 

s_cmp_eq_i32 s12, 0 // (op 582) 
s_mov_b32 s8, scc // (op 582) 
s_cmp_eq_i32 s8, 0 // (op 582) 
s_cselect_b32 s9, 0, -1 // (op 582) 

s_and_b32 s8, s13, 31 // (op 582) 
s_lshl_b32 s18, 1, s8 // (op 582) 
s_add_i32 s19, s18, s9 // (op 582) 


s_and_b32 s9, s0, s19 // (op 582) 


s_add_i32 s0, s1, s9 // (op 582) 


s_ashr_i32 s1, s0, s8 // (op 582) 


s_lshl_b32 s0, s13, 24 // (op 582) 
s_ashr_i32 s8, s0, 31 // (op 582) 

s_xor_b32 s0, s1, s8 // (op 582) 

s_sub_i32 s1, s0, s8 // (op 582) 


s_mul_hi_i32 s0, s1, s14 // (op 582) 
s_add_i32 s8, s0, s1 // (op 582) 

s_ashr_i32 s0, s8, 31 // (op 582) 
s_cmp_eq_i32 s14, 0 // (op 582) 
s_mov_b32 s9, scc // (op 582) 
s_cmp_eq_i32 s9, 0 // (op 582) 
s_cselect_b32 s18, 0, -1 // (op 582) 

s_and_b32 s9, s15, 31 // (op 582) 
s_lshl_b32 s19, 1, s9 // (op 582) 
s_add_i32 s49, s19, s18 // (op 582) 


s_and_b32 s18, s0, s49 // (op 582) 


s_add_i32 s0, s8, s18 // (op 582) 


s_ashr_i32 s8, s0, s9 // (op 582) 


s_lshl_b32 s0, s15, 24 // (op 582) 
s_ashr_i32 s9, s0, 31 // (op 582) 

s_xor_b32 s0, s8, s9 // (op 582) 

s_sub_i32 s8, s0, s9 // (op 582) 


s_mul_i32 s0, s1, s45 // (op 582) 

s_sub_u32 s1, s16, s0 // (op 582) 

s_mul_i32 s0, s8, s45 // (op 582) 
s_add_u32 s0, s0, s1 // (op 582) 


s_mul_hi_i32 s1, s16, s11 // (op 582) 
s_add_i32 s8, s1, s16 // (op 582) 

s_ashr_i32 s1, s8, 31 // (op 582) 
s_cmp_eq_i32 s11, 0 // (op 582) 
s_mov_b32 s9, scc // (op 582) 
s_cmp_eq_i32 s9, 0 // (op 582) 
s_cselect_b32 s18, 0, -1 // (op 582) 

s_and_b32 s9, s46, 31 // (op 582) 
s_lshl_b32 s19, 1, s9 // (op 582) 
s_add_i32 s49, s19, s18 // (op 582) 


s_and_b32 s18, s1, s49 // (op 582) 


s_add_i32 s1, s8, s18 // (op 582) 


s_ashr_i32 s8, s1, s9 // (op 582) 


s_lshl_b32 s1, s46, 24 // (op 582) 
s_ashr_i32 s9, s1, 31 // (op 582) 

s_xor_b32 s1, s8, s9 // (op 582) 

s_sub_i32 s8, s1, s9 // (op 582) 


s_add_u32 s18, s4, 16 // (op 582) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 582) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 582) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 582) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 582) 

s_mul_i32 s1, s45, s18 // (op 582) 

s_mul_i32 s9, s10, s1 // (op 582) 

s_mul_i32 s1, s8, s9 // (op 582) 

s_sub_u32 s9, s16, s1 // (op 582) 

s_mov_b32 s1, s9 // (op 582) 
s_mul_hi_i32 s18, s1, s44 // (op 582) 
s_add_i32 s19, s18, s1 // (op 582) 


s_ashr_i32 s1, s19, 31 // (op 582) 
s_cmp_eq_i32 s44, 0 // (op 582) 
s_mov_b32 s18, scc // (op 582) 
s_cmp_eq_i32 s18, 0 // (op 582) 
s_cselect_b32 s49, 0, -1 // (op 582) 

s_and_b32 s18, s48, 31 // (op 582) 
s_lshl_b32 s50, 1, s18 // (op 582) 
s_add_i32 s51, s50, s49 // (op 582) 


s_and_b32 s49, s1, s51 // (op 582) 


s_add_i32 s1, s19, s49 // (op 582) 


s_ashr_i32 s19, s1, s18 // (op 582) 


s_lshl_b32 s1, s48, 24 // (op 582) 
s_ashr_i32 s18, s1, 31 // (op 582) 

s_xor_b32 s1, s19, s18 // (op 582) 

s_sub_i32 s19, s1, s18 // (op 582) 


s_mul_i32 s1, s47, s45 // (op 582) 
s_add_u32 s50, s6, 32 // (op 582) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 582) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 582) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 582) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 582) 

s_sub_i32 s18, s50, s1 // (op 582) 

s_mul_i32 s49, s19, s18 // (op 582) 


s_sub_u32 s18, s9, s49 // (op 582) 


s_add_u32 s9, s18, s1 // (op 582) 


s_cmp_eq_i32 s8, 0 // (op 582) 
s_mov_b32 s1, scc // (op 582) 

s_cmp_eq_i32 s1, 0 // (op 582) 
s_cselect_b32 s8, s9, s0 // (op 582) 



v_lshl_add_u32 v12, s8, 5, v11 // (op 582) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 582) 
s_lshr_b32 s8, s0, 1 // (op 582) 

v_lshl_add_u32 v11, s8, 3, v13 // (op 582) 


v_mul_lo_u32 v13, v12, s26 // (op 582) 
v_add_u32 v13, v13, v11 // (op 582) 


s_mul_i32 s1, s26, s7 // (op 584) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 584) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 584) least significant
s_mul_hi_u32 s9, s26, s6 // (op 584) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 584) most significant: sum
s_add_u32 s1, s1, s9 // (op 584) most significant: sum


s_mov_b32 s60, s28 // (op 584) reg expression
s_mov_b32 s61, s29 // (op 584) reg expression
s_mov_b32 s62, s0 // (op 584) reg expression
s_mov_b32 s63, 131072 // (op 584) call()

s_mul_hi_i32 s0, s16, s12 // (op 646) 
v_and_b32 v11, 15, v1 // (op 646) 
s_add_i32 s1, s0, s16 // (op 646) 

v_and_b32 v12, 63, v1 // (op 646) 
s_ashr_i32 s0, s1, 31 // (op 646) 
v_lshrrev_b32 v14, 4, v12 // (op 646) 

s_cmp_eq_i32 s12, 0 // (op 646) 
s_mov_b32 s8, scc // (op 646) 
s_cmp_eq_i32 s8, 0 // (op 646) 
s_cselect_b32 s9, 0, -1 // (op 646) 

s_and_b32 s8, s13, 31 // (op 646) 
s_lshl_b32 s18, 1, s8 // (op 646) 
s_add_i32 s19, s18, s9 // (op 646) 


s_and_b32 s9, s0, s19 // (op 646) 


s_add_i32 s0, s1, s9 // (op 646) 


s_ashr_i32 s1, s0, s8 // (op 646) 


s_lshl_b32 s0, s13, 24 // (op 646) 
s_ashr_i32 s8, s0, 31 // (op 646) 

s_xor_b32 s0, s1, s8 // (op 646) 

s_sub_i32 s1, s0, s8 // (op 646) 


s_mul_hi_i32 s0, s1, s14 // (op 646) 
s_add_i32 s8, s0, s1 // (op 646) 

s_ashr_i32 s0, s8, 31 // (op 646) 
s_cmp_eq_i32 s14, 0 // (op 646) 
s_mov_b32 s9, scc // (op 646) 
s_cmp_eq_i32 s9, 0 // (op 646) 
s_cselect_b32 s18, 0, -1 // (op 646) 

s_and_b32 s9, s15, 31 // (op 646) 
s_lshl_b32 s19, 1, s9 // (op 646) 
s_add_i32 s49, s19, s18 // (op 646) 


s_and_b32 s18, s0, s49 // (op 646) 


s_add_i32 s0, s8, s18 // (op 646) 


s_ashr_i32 s8, s0, s9 // (op 646) 


s_lshl_b32 s0, s15, 24 // (op 646) 
s_ashr_i32 s9, s0, 31 // (op 646) 

s_xor_b32 s0, s8, s9 // (op 646) 

s_sub_i32 s8, s0, s9 // (op 646) 


s_add_u32 s18, s4, 16 // (op 646) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 646) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 646) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 646) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 646) 

s_mul_i32 s0, s8, s18 // (op 646) 

s_sub_i32 s8, s1, s0 // (op 646) 


s_mul_hi_i32 s9, s16, s11 // (op 646) 
s_add_i32 s0, s9, s16 // (op 646) 

s_ashr_i32 s1, s0, 31 // (op 646) 
s_cmp_eq_i32 s11, 0 // (op 646) 
s_mov_b32 s9, scc // (op 646) 
s_cmp_eq_i32 s9, 0 // (op 646) 
s_cselect_b32 s49, 0, -1 // (op 646) 

s_and_b32 s9, s46, 31 // (op 646) 
s_lshl_b32 s50, 1, s9 // (op 646) 
s_add_i32 s51, s50, s49 // (op 646) 


s_and_b32 s49, s1, s51 // (op 646) 


s_add_i32 s1, s0, s49 // (op 646) 


s_ashr_i32 s0, s1, s9 // (op 646) 


s_lshl_b32 s1, s46, 24 // (op 646) 
s_ashr_i32 s9, s1, 31 // (op 646) 

s_xor_b32 s1, s0, s9 // (op 646) 

s_sub_i32 s0, s1, s9 // (op 646) 


s_mul_i32 s1, s45, s18 // (op 646) 

s_mul_i32 s9, s10, s1 // (op 646) 

s_mul_i32 s1, s0, s9 // (op 646) 

s_sub_u32 s9, s16, s1 // (op 646) 

s_mul_hi_i32 s1, s9, s44 // (op 646) 
s_add_i32 s18, s1, s9 // (op 646) 


s_ashr_i32 s1, s18, 31 // (op 646) 
s_cmp_eq_i32 s44, 0 // (op 646) 
s_mov_b32 s9, scc // (op 646) 
s_cmp_eq_i32 s9, 0 // (op 646) 
s_cselect_b32 s19, 0, -1 // (op 646) 

s_and_b32 s9, s48, 31 // (op 646) 
s_lshl_b32 s49, 1, s9 // (op 646) 
s_add_i32 s50, s49, s19 // (op 646) 


s_and_b32 s19, s1, s50 // (op 646) 


s_add_i32 s1, s18, s19 // (op 646) 


s_ashr_i32 s18, s1, s9 // (op 646) 


s_lshl_b32 s1, s48, 24 // (op 646) 
s_ashr_i32 s9, s1, 31 // (op 646) 

s_xor_b32 s1, s18, s9 // (op 646) 

s_sub_i32 s18, s1, s9 // (op 646) 


s_cmp_eq_i32 s0, 0 // (op 646) 
s_mov_b32 s1, scc // (op 646) 

s_cmp_eq_i32 s1, 0 // (op 646) 
s_cselect_b32 s0, s18, s8 // (op 646) 



v_lshl_add_u32 v12, s0, 4, v11 // (op 646) 


v_mul_lo_u32 v11, v12, s30 // (op 646) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 646) 
s_lshr_b32 s8, s0, 1 // (op 646) 

v_lshl_add_u32 v12, s8, 3, v14 // (op 646) 


v_add_u32 v14, v12, v11 // (op 646) 


s_mul_i32 s1, s30, s5 // (op 648) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 648) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 648) least significant
s_mul_hi_u32 s9, s30, s4 // (op 648) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 648) most significant: sum
s_add_u32 s1, s1, s9 // (op 648) most significant: sum


s_mov_b32 s64, s32 // (op 648) reg expression
s_mov_b32 s65, s33 // (op 648) reg expression
s_mov_b32 s66, s0 // (op 648) reg expression
s_mov_b32 s67, 131072 // (op 648) call()

v_and_b32 v15, -4, v1 // (op 739) 
v_sub_u32 v11, v1, v15 // (op 739) 

v_lshlrev_b32 v12, 5, v11 // (op 739) 

v_lshrrev_b32 v11, 2, v1 // (op 739) 
v_lshl_add_u32 v15, v11, 7, v12 // (op 739) 


v_mul_lo_u32 v11, v15, 6 // (op 739) 

v_lshrrev_b32 v12, 3, v11 // (op 739) 

v_and_b32 v11, -8, v1 // (op 792) 
v_sub_u32 v15, v1, v11 // (op 792) 

v_lshlrev_b32 v11, 4, v15 // (op 792) 

v_lshrrev_b32 v15, 3, v1 // (op 792) 
v_lshl_add_u32 v18, v15, 7, v11 // (op 792) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 792) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v11, -4, v1 // (op 844) 
v_sub_u32 v15, v1, v11 // (op 844) 

v_and_b32 v11, 4294967292, v1 // (op 844) 
v_add_u32 v18, v11, v15 // (op 844) 


s_ashr_i64 s[8:9], s[34:35], 7 // (op 166) 
s_and_b32 s1, s8, 4294967294 // (op 166) 

s_ashr_i64 s[8:9], s[34:35], 7 // (op 165) 
s_mov_b32 s18, s8 // (op 165) convert

s_cmp_lt_u32 s1, s18 // (op 165) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_165:


s_add_u32 s1, s1, 1 // (op 167) 
v_mov_b32 v15, v29 // (op 170) 

buffer_load_dwordx4 v[20:23], v15, s[52:55], 0 offen // (op 170) Load value
buffer_load_dwordx2 v[24:25], v15, s[52:55], 0 offen offset:16 // (op 170) Load value

v_add_u32 v29, 96, v29 // (op 441) 
s_barrier  // (op 1032) 
s_barrier  // (op 1045) 





v_mov_b64 v[10:11], v[16:17] // (op 421) 

s_mov_b32 m0, s0 // (op 421) reg expression
buffer_load_dwordx4 v10, s[56:59], 0 offen offset: 0 lds // (op 421) Load value direct to lds

v_add_co_u32 v16, vcc, 128, v16 // (op 510) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 510) most significant half
s_waitcnt vmcnt(2)
ds_write_b128 v12, v[20:23] offset:4096 // (op 171) Store local data 


v_mov_b32 v15, v13 // (op 184) 

buffer_load_ubyte v19, v15, s[60:63], 0 offen offset: 0 // (op 184) Load value

v_add_u32 v13, 4, v13 // (op 574) 
s_waitcnt vmcnt(2)
ds_write_b64 v12, v[24:25] offset:4112 // (op 171) Store local data 

s_barrier  // (op 1058) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1035) 






ds_read_b128 v[20:23], v3 offset:4096 // (op 173) Load local data 
ds_read_b64 v[24:25], v3 offset:4112 // (op 173) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v19 offset:7168 // (op 182) Store local data 

s_barrier  // (op 1048) 



ds_read_b128 v[30:33], v5 // (op 175) Load local data 
ds_read_b128 v[34:37], v5 offset:64 // (op 175) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1062) 



ds_read_u8 v15, v0 offset:7168 // (op 177) Load local data 


v_mov_b32 v19, v14 // (op 179) 

buffer_load_ubyte v10, v19, s[64:67], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v14, 4, v14 // (op 638) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[30:37], a[0:3], v15, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 180) 




s_ashr_i64 s[8:9], s[34:35], 7 // (op 165) 
s_mov_b32 s18, s8 // (op 165) convert

s_cmp_lt_u32 s1, s18 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_165 if true)
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_165:





















s_barrier  // (op 1126) 






v_cmp_eq_f32 vcc, 0.00000, v7 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_125
s_nop 0
v_accvgpr_read v10, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 

v_cvt_f16_f32 v10, v14 // (op 52) 
v_cvt_f16_f32 v11, v15 // (op 52) 
v_cvt_f16_f32 v12, v16 // (op 52) 
v_cvt_f16_f32 v13, v17 // (op 52) 

s_mul_hi_i32 s0, s16, s12 // (op 886) 
v_lshrrev_b32 v0, 6, v1 // (op 886) 
s_add_i32 s1, s0, s16 // (op 886) 

v_and_b32 v8, 63, v1 // (op 886) 
s_ashr_i32 s0, s1, 31 // (op 886) 
v_lshrrev_b32 v3, 4, v8 // (op 886) 
s_cmp_eq_i32 s12, 0 // (op 886) 
s_mov_b32 s2, scc // (op 886) 
v_lshlrev_b32 v4, 2, v3 // (op 886) 

s_cmp_eq_i32 s2, 0 // (op 886) 
s_cselect_b32 s3, 0, -1 // (op 886) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 886) 


s_and_b32 s2, s13, 31 // (op 886) 
v_lshrrev_b32 v0, 2, v8 // (op 886) 

s_lshl_b32 s8, 1, s2 // (op 886) 
v_and_b32 v8, 3, v0 // (op 886) 

s_add_i32 s9, s8, s3 // (op 886) 


v_and_b32 v0, 3, v1 // (op 886) 
s_and_b32 s3, s0, s9 // (op 886) 


v_lshl_add_u32 v4, v8, 2, v0 // (op 886) 


s_add_i32 s0, s1, s3 // (op 886) 


s_ashr_i32 s1, s0, s2 // (op 886) 


s_lshl_b32 s0, s13, 24 // (op 886) 
s_ashr_i32 s2, s0, 31 // (op 886) 

s_xor_b32 s0, s1, s2 // (op 886) 

s_sub_i32 s1, s0, s2 // (op 886) 


s_mul_hi_i32 s0, s1, s14 // (op 886) 
s_add_i32 s2, s0, s1 // (op 886) 

s_ashr_i32 s0, s2, 31 // (op 886) 
s_cmp_eq_i32 s14, 0 // (op 886) 
s_mov_b32 s3, scc // (op 886) 
s_cmp_eq_i32 s3, 0 // (op 886) 
s_cselect_b32 s8, 0, -1 // (op 886) 

s_and_b32 s3, s15, 31 // (op 886) 
s_lshl_b32 s9, 1, s3 // (op 886) 
s_add_i32 s18, s9, s8 // (op 886) 


s_and_b32 s8, s0, s18 // (op 886) 


s_add_i32 s0, s2, s8 // (op 886) 


s_ashr_i32 s2, s0, s3 // (op 886) 


s_lshl_b32 s0, s15, 24 // (op 886) 
s_ashr_i32 s3, s0, 31 // (op 886) 

s_xor_b32 s0, s2, s3 // (op 886) 

s_sub_i32 s2, s0, s3 // (op 886) 


s_mul_i32 s0, s1, s45 // (op 886) 
s_sub_u32 s3, s16, s0 // (op 886) 

s_mul_i32 s0, s2, s45 // (op 886) 
s_add_u32 s0, s0, s3 // (op 886) 

s_mul_hi_i32 s3, s16, s11 // (op 886) 
s_add_i32 s8, s3, s16 // (op 886) 

s_ashr_i32 s3, s8, 31 // (op 886) 
s_cmp_eq_i32 s11, 0 // (op 886) 
s_mov_b32 s9, scc // (op 886) 
s_cmp_eq_i32 s9, 0 // (op 886) 
s_cselect_b32 s18, 0, -1 // (op 886) 

s_and_b32 s9, s46, 31 // (op 886) 
s_lshl_b32 s19, 1, s9 // (op 886) 
s_add_i32 s20, s19, s18 // (op 886) 


s_and_b32 s18, s3, s20 // (op 886) 


s_add_i32 s3, s8, s18 // (op 886) 


s_ashr_i32 s8, s3, s9 // (op 886) 


s_lshl_b32 s3, s46, 24 // (op 886) 
s_ashr_i32 s9, s3, 31 // (op 886) 

s_xor_b32 s3, s8, s9 // (op 886) 

s_sub_i32 s8, s3, s9 // (op 886) 


s_add_u32 s18, s4, 16 // (op 886) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 886) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 886) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 886) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 886) 

s_mul_i32 s3, s45, s18 // (op 886) 
s_mul_i32 s9, s10, s3 // (op 886) 

s_mul_i32 s3, s8, s9 // (op 886) 

s_sub_u32 s9, s16, s3 // (op 886) 

s_mov_b32 s3, s9 // (op 886) 
s_mul_hi_i32 s20, s3, s44 // (op 886) 
s_add_i32 s21, s20, s3 // (op 886) 


s_ashr_i32 s3, s21, 31 // (op 886) 
s_cmp_eq_i32 s44, 0 // (op 886) 
s_mov_b32 s20, scc // (op 886) 
s_cmp_eq_i32 s20, 0 // (op 886) 
s_cselect_b32 s22, 0, -1 // (op 886) 

s_and_b32 s20, s48, 31 // (op 886) 
s_lshl_b32 s23, 1, s20 // (op 886) 
s_add_i32 s24, s23, s22 // (op 886) 


s_and_b32 s22, s3, s24 // (op 886) 


s_add_i32 s3, s21, s22 // (op 886) 


s_ashr_i32 s21, s3, s20 // (op 886) 


s_lshl_b32 s3, s48, 24 // (op 886) 
s_ashr_i32 s20, s3, 31 // (op 886) 

s_xor_b32 s3, s21, s20 // (op 886) 

s_sub_i32 s21, s3, s20 // (op 886) 


s_mul_i32 s3, s47, s45 // (op 886) 
s_add_u32 s22, s6, 32 // (op 886) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 886) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 886) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 886) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 886) 

s_sub_i32 s20, s22, s3 // (op 886) 

s_mul_i32 s22, s21, s20 // (op 886) 

s_sub_u32 s20, s9, s22 // (op 886) 


s_add_u32 s9, s20, s3 // (op 886) 


s_cmp_eq_i32 s8, 0 // (op 886) 
s_mov_b32 s3, scc // (op 886) 

s_cmp_eq_i32 s3, 0 // (op 886) 
s_cselect_b32 s8, s9, s0 // (op 886) 


s_mul_i32 s0, s2, s18 // (op 886) 


v_lshl_add_u32 v0, s8, 5, v3 // (op 886) 


s_sub_i32 s2, s1, s0 // (op 886) 


s_cmp_eq_i32 s3, 0 // (op 886) 
s_cselect_b32 s0, s21, s2 // (op 886) 



v_lshl_add_u32 v3, s0, 4, v4 // (op 886) 


v_mul_lo_u32 v8, v3, s36 // (op 886) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 886) 


s_mul_i32 s9, s36, s5 // (op 888) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 888) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 888) least significant
s_mul_hi_u32 s1, s36, s4 // (op 888) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 888) most significant: sum
s_add_u32 s9, s9, s1 // (op 888) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 888) 

s_mov_b32 s32, s38 // (op 888) reg expression
s_mov_b32 s33, s39 // (op 888) reg expression
s_mov_b32 s34, s0 // (op 888) reg expression
s_mov_b32 s35, 131072 // (op 888) call()


v_pack_b32_f16 v4, v10, v11 // (op 54) 
v_pack_b32_f16 v5, v12, v13 // (op 54) 


buffer_store_dwordx2 v[4:5], v3, s[32:35], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_125:
s_mul_hi_i32 s0, s16, s12 // (op 709) 
v_lshrrev_b32 v0, 6, v1 // (op 709) 
s_add_i32 s1, s0, s16 // (op 709) 

v_and_b32 v8, 63, v1 // (op 709) 
s_ashr_i32 s0, s1, 31 // (op 709) 
v_lshrrev_b32 v3, 4, v8 // (op 709) 
s_cmp_eq_i32 s12, 0 // (op 709) 
s_mov_b32 s2, scc // (op 709) 
v_lshlrev_b32 v4, 2, v3 // (op 709) 

s_cmp_eq_i32 s2, 0 // (op 709) 
s_cselect_b32 s3, 0, -1 // (op 709) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 709) 


s_and_b32 s2, s13, 31 // (op 709) 
v_lshrrev_b32 v0, 2, v8 // (op 709) 

s_lshl_b32 s8, 1, s2 // (op 709) 
v_and_b32 v8, 3, v0 // (op 709) 

s_add_i32 s9, s8, s3 // (op 709) 


v_and_b32 v0, 3, v1 // (op 709) 
s_and_b32 s3, s0, s9 // (op 709) 


v_lshl_add_u32 v4, v8, 2, v0 // (op 709) 


s_add_i32 s0, s1, s3 // (op 709) 


s_ashr_i32 s1, s0, s2 // (op 709) 


s_lshl_b32 s0, s13, 24 // (op 709) 
s_ashr_i32 s2, s0, 31 // (op 709) 

s_xor_b32 s0, s1, s2 // (op 709) 

s_sub_i32 s1, s0, s2 // (op 709) 


s_mul_hi_i32 s0, s1, s14 // (op 709) 
s_add_i32 s2, s0, s1 // (op 709) 

s_ashr_i32 s0, s2, 31 // (op 709) 
s_cmp_eq_i32 s14, 0 // (op 709) 
s_mov_b32 s3, scc // (op 709) 
s_cmp_eq_i32 s3, 0 // (op 709) 
s_cselect_b32 s8, 0, -1 // (op 709) 

s_and_b32 s3, s15, 31 // (op 709) 
s_lshl_b32 s9, 1, s3 // (op 709) 
s_add_i32 s18, s9, s8 // (op 709) 


s_and_b32 s8, s0, s18 // (op 709) 


s_add_i32 s0, s2, s8 // (op 709) 


s_ashr_i32 s2, s0, s3 // (op 709) 


s_lshl_b32 s0, s15, 24 // (op 709) 
s_ashr_i32 s3, s0, 31 // (op 709) 

s_xor_b32 s0, s2, s3 // (op 709) 

s_sub_i32 s2, s0, s3 // (op 709) 


s_mul_i32 s0, s1, s45 // (op 709) 
s_sub_u32 s3, s16, s0 // (op 709) 

s_mul_i32 s0, s2, s45 // (op 709) 
s_add_u32 s0, s0, s3 // (op 709) 

s_mul_hi_i32 s3, s16, s11 // (op 709) 
s_add_i32 s8, s3, s16 // (op 709) 

s_ashr_i32 s3, s8, 31 // (op 709) 
s_cmp_eq_i32 s11, 0 // (op 709) 
s_mov_b32 s9, scc // (op 709) 
s_cmp_eq_i32 s9, 0 // (op 709) 
s_cselect_b32 s18, 0, -1 // (op 709) 

s_and_b32 s9, s46, 31 // (op 709) 
s_lshl_b32 s19, 1, s9 // (op 709) 
s_add_i32 s20, s19, s18 // (op 709) 


s_and_b32 s18, s3, s20 // (op 709) 


s_add_i32 s3, s8, s18 // (op 709) 


s_ashr_i32 s8, s3, s9 // (op 709) 


s_lshl_b32 s3, s46, 24 // (op 709) 
s_ashr_i32 s9, s3, 31 // (op 709) 

s_xor_b32 s3, s8, s9 // (op 709) 

s_sub_i32 s8, s3, s9 // (op 709) 


s_add_u32 s18, s4, 16 // (op 709) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 709) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 709) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 709) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 709) 

s_mul_i32 s3, s45, s18 // (op 709) 
s_mul_i32 s9, s10, s3 // (op 709) 

s_mul_i32 s3, s8, s9 // (op 709) 

s_sub_u32 s9, s16, s3 // (op 709) 

s_mov_b32 s3, s9 // (op 709) 
s_mul_hi_i32 s20, s3, s44 // (op 709) 
s_add_i32 s21, s20, s3 // (op 709) 


s_ashr_i32 s3, s21, 31 // (op 709) 
s_cmp_eq_i32 s44, 0 // (op 709) 
s_mov_b32 s20, scc // (op 709) 
s_cmp_eq_i32 s20, 0 // (op 709) 
s_cselect_b32 s22, 0, -1 // (op 709) 

s_and_b32 s20, s48, 31 // (op 709) 
s_lshl_b32 s23, 1, s20 // (op 709) 
s_add_i32 s24, s23, s22 // (op 709) 


s_and_b32 s22, s3, s24 // (op 709) 


s_add_i32 s3, s21, s22 // (op 709) 


s_ashr_i32 s21, s3, s20 // (op 709) 


s_lshl_b32 s3, s48, 24 // (op 709) 
s_ashr_i32 s20, s3, 31 // (op 709) 

s_xor_b32 s3, s21, s20 // (op 709) 

s_sub_i32 s21, s3, s20 // (op 709) 


s_mul_i32 s3, s47, s45 // (op 709) 
s_add_u32 s22, s6, 32 // (op 709) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 709) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 709) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 709) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 709) 

s_sub_i32 s20, s22, s3 // (op 709) 

s_mul_i32 s22, s21, s20 // (op 709) 

s_sub_u32 s20, s9, s22 // (op 709) 


s_add_u32 s9, s20, s3 // (op 709) 


s_cmp_eq_i32 s8, 0 // (op 709) 
s_mov_b32 s3, scc // (op 709) 

s_cmp_eq_i32 s3, 0 // (op 709) 
s_cselect_b32 s8, s9, s0 // (op 709) 


s_mul_i32 s0, s2, s18 // (op 709) 


v_lshl_add_u32 v0, s8, 5, v3 // (op 709) 


s_sub_i32 s2, s1, s0 // (op 709) 


s_cmp_eq_i32 s3, 0 // (op 709) 
s_cselect_b32 s0, s21, s2 // (op 709) 



v_lshl_add_u32 v3, s0, 4, v4 // (op 709) 


v_mul_lo_u32 v8, v3, s40 // (op 709) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 709) 


s_mul_i32 s9, s40, s5 // (op 711) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 711) most significant: low of high * low
s_mul_i32 s8, s40, s4 // (op 711) least significant
s_mul_hi_u32 s1, s40, s4 // (op 711) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 711) most significant: sum
s_add_u32 s9, s9, s1 // (op 711) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 711) 

s_mov_b32 s28, s42 // (op 711) reg expression
s_mov_b32 s29, s43 // (op 711) reg expression
s_mov_b32 s30, s0 // (op 711) reg expression
s_mov_b32 s31, 131072 // (op 711) call()





buffer_load_dwordx2 v[4:5], v3, s[28:31], 0 offen // (op 138) Load value



v_accvgpr_read v10, a0 // (op 134) 
v_accvgpr_read v11, a1 // (op 134) 
v_accvgpr_read v12, a2 // (op 134) 
v_accvgpr_read v13, a3 // (op 134) 
v_mul_f32 v14, v9, v10 // (op 134) 
v_mul_f32 v15, v9, v11 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 139) 
v_lshrrev_b32 v11, 16, v4 // (op 139) 
v_cvt_f32_f16 v11, v11 // (op 139) 
v_mul_f32 v18, v7, v10 // (op 139) 
v_mul_f32 v19, v7, v11 // (op 139) 
v_cvt_f32_f16 v10, v5 // (op 139) 
v_lshrrev_b32 v11, 16, v5 // (op 139) 
v_cvt_f32_f16 v11, v11 // (op 139) 
v_mul_f32 v20, v7, v10 // (op 139) 
v_mul_f32 v21, v7, v11 // (op 139) 


v_add_f32 v10, v18, v14 // (op 135) 
v_add_f32 v11, v19, v15 // (op 135) 
v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 


v_cvt_f16_f32 v14, v10 // (op 136) 
v_cvt_f16_f32 v15, v11 // (op 136) 
v_cvt_f16_f32 v16, v12 // (op 136) 
v_cvt_f16_f32 v17, v13 // (op 136) 

s_mul_hi_i32 s0, s16, s12 // (op 903) 
v_lshrrev_b32 v0, 6, v1 // (op 903) 
s_add_i32 s1, s0, s16 // (op 903) 

v_and_b32 v8, 63, v1 // (op 903) 
s_ashr_i32 s0, s1, 31 // (op 903) 
v_lshrrev_b32 v3, 4, v8 // (op 903) 
s_cmp_eq_i32 s12, 0 // (op 903) 
s_mov_b32 s2, scc // (op 903) 
v_lshlrev_b32 v4, 2, v3 // (op 903) 

s_cmp_eq_i32 s2, 0 // (op 903) 
s_cselect_b32 s3, 0, -1 // (op 903) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 903) 


s_and_b32 s2, s13, 31 // (op 903) 
v_lshrrev_b32 v0, 2, v8 // (op 903) 

s_lshl_b32 s8, 1, s2 // (op 903) 
v_and_b32 v8, 3, v0 // (op 903) 

s_add_i32 s9, s8, s3 // (op 903) 


v_and_b32 v0, 3, v1 // (op 903) 
s_and_b32 s3, s0, s9 // (op 903) 


v_lshl_add_u32 v4, v8, 2, v0 // (op 903) 


s_add_i32 s0, s1, s3 // (op 903) 


s_ashr_i32 s1, s0, s2 // (op 903) 


s_lshl_b32 s0, s13, 24 // (op 903) 
s_ashr_i32 s2, s0, 31 // (op 903) 

s_xor_b32 s0, s1, s2 // (op 903) 

s_sub_i32 s1, s0, s2 // (op 903) 


s_mul_hi_i32 s0, s1, s14 // (op 903) 
s_add_i32 s2, s0, s1 // (op 903) 

s_ashr_i32 s0, s2, 31 // (op 903) 
s_cmp_eq_i32 s14, 0 // (op 903) 
s_mov_b32 s3, scc // (op 903) 
s_cmp_eq_i32 s3, 0 // (op 903) 
s_cselect_b32 s8, 0, -1 // (op 903) 

s_and_b32 s3, s15, 31 // (op 903) 
s_lshl_b32 s9, 1, s3 // (op 903) 
s_add_i32 s18, s9, s8 // (op 903) 


s_and_b32 s8, s0, s18 // (op 903) 


s_add_i32 s0, s2, s8 // (op 903) 


s_ashr_i32 s2, s0, s3 // (op 903) 


s_lshl_b32 s0, s15, 24 // (op 903) 
s_ashr_i32 s3, s0, 31 // (op 903) 

s_xor_b32 s0, s2, s3 // (op 903) 

s_sub_i32 s2, s0, s3 // (op 903) 


s_mul_i32 s0, s1, s45 // (op 903) 
s_sub_u32 s3, s16, s0 // (op 903) 

s_mul_i32 s0, s2, s45 // (op 903) 
s_add_u32 s0, s0, s3 // (op 903) 

s_mul_hi_i32 s3, s16, s11 // (op 903) 
s_add_i32 s8, s3, s16 // (op 903) 

s_ashr_i32 s3, s8, 31 // (op 903) 
s_cmp_eq_i32 s11, 0 // (op 903) 
s_mov_b32 s9, scc // (op 903) 
s_cmp_eq_i32 s9, 0 // (op 903) 
s_cselect_b32 s18, 0, -1 // (op 903) 

s_and_b32 s9, s46, 31 // (op 903) 
s_lshl_b32 s19, 1, s9 // (op 903) 
s_add_i32 s20, s19, s18 // (op 903) 


s_and_b32 s18, s3, s20 // (op 903) 


s_add_i32 s3, s8, s18 // (op 903) 


s_ashr_i32 s8, s3, s9 // (op 903) 


s_lshl_b32 s3, s46, 24 // (op 903) 
s_ashr_i32 s9, s3, 31 // (op 903) 

s_xor_b32 s3, s8, s9 // (op 903) 

s_sub_i32 s8, s3, s9 // (op 903) 


s_add_u32 s18, s4, 16 // (op 903) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 903) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 903) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 903) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 903) 

s_mul_i32 s3, s45, s18 // (op 903) 
s_mul_i32 s9, s10, s3 // (op 903) 

s_mul_i32 s3, s8, s9 // (op 903) 

s_sub_u32 s9, s16, s3 // (op 903) 

s_mov_b32 s3, s9 // (op 903) 
s_mul_hi_i32 s20, s3, s44 // (op 903) 
s_add_i32 s21, s20, s3 // (op 903) 


s_ashr_i32 s3, s21, 31 // (op 903) 
s_cmp_eq_i32 s44, 0 // (op 903) 
s_mov_b32 s20, scc // (op 903) 
s_cmp_eq_i32 s20, 0 // (op 903) 
s_cselect_b32 s22, 0, -1 // (op 903) 

s_and_b32 s20, s48, 31 // (op 903) 
s_lshl_b32 s23, 1, s20 // (op 903) 
s_add_i32 s24, s23, s22 // (op 903) 


s_and_b32 s22, s3, s24 // (op 903) 


s_add_i32 s3, s21, s22 // (op 903) 


s_ashr_i32 s21, s3, s20 // (op 903) 


s_lshl_b32 s3, s48, 24 // (op 903) 
s_ashr_i32 s20, s3, 31 // (op 903) 

s_xor_b32 s3, s21, s20 // (op 903) 

s_sub_i32 s21, s3, s20 // (op 903) 


s_mul_i32 s3, s47, s45 // (op 903) 
s_add_u32 s22, s6, 32 // (op 903) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 903) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 903) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 903) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 903) 

s_sub_i32 s20, s22, s3 // (op 903) 

s_mul_i32 s22, s21, s20 // (op 903) 

s_sub_u32 s20, s9, s22 // (op 903) 


s_add_u32 s9, s20, s3 // (op 903) 


s_cmp_eq_i32 s8, 0 // (op 903) 
s_mov_b32 s3, scc // (op 903) 

s_cmp_eq_i32 s3, 0 // (op 903) 
s_cselect_b32 s8, s9, s0 // (op 903) 


s_mul_i32 s0, s2, s18 // (op 903) 


v_lshl_add_u32 v0, s8, 5, v3 // (op 903) 


s_sub_i32 s2, s1, s0 // (op 903) 


s_cmp_eq_i32 s3, 0 // (op 903) 
s_cselect_b32 s0, s21, s2 // (op 903) 



v_lshl_add_u32 v3, s0, 4, v4 // (op 903) 


v_mul_lo_u32 v8, v3, s36 // (op 903) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 903) 


s_mul_i32 s9, s36, s5 // (op 905) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 905) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 905) least significant
s_mul_hi_u32 s1, s36, s4 // (op 905) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 905) most significant: sum
s_add_u32 s9, s9, s1 // (op 905) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 905) 

s_mov_b32 s32, s38 // (op 905) reg expression
s_mov_b32 s33, s39 // (op 905) reg expression
s_mov_b32 s34, s0 // (op 905) reg expression
s_mov_b32 s35, 131072 // (op 905) call()


v_pack_b32_f16 v4, v14, v15 // (op 137) 
v_pack_b32_f16 v5, v16, v17 // (op 137) 


buffer_store_dwordx2 v[4:5], v3, s[32:35], 0 offen // (op 137) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.LRR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_, .LRR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__end-RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
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
  - .name: RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_.kd
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

