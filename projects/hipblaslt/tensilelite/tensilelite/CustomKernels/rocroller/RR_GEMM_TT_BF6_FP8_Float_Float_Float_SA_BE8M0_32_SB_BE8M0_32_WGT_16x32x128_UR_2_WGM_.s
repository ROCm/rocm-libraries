// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_,@function
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 767) 
v_and_b32 v3, 15, v1 // (op 767) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 767) 


v_and_b32 v0, 63, v1 // (op 767) 
v_lshrrev_b32 v3, 4, v0 // (op 767) 

v_lshlrev_b32 v0, 5, v3 // (op 767) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 767) 


v_mul_lo_u32 v0, v3, 6 // (op 767) 

v_lshrrev_b32 v3, 3, v0 // (op 767) 

v_and_b32 v0, 48, v1 // (op 819) 
v_and_b32 v4, 63, v1 // (op 819) 
v_sub_u32 v5, v4, v0 // (op 819) 

v_lshrrev_b32 v4, 1, v5 // (op 819) 
v_add_lshl_u32 v6, v0, v4, 5 // (op 819) 


v_and_b32 v0, 1, v5 // (op 819) 

v_lshlrev_b32 v4, 3, v0 // (op 819) 

v_lshrrev_b32 v0, 6, v1 // (op 819) 
v_and_b32 v5, 1, v0 // (op 819) 

v_lshl_add_u32 v0, v5, 4, v4 // (op 819) 


v_add_u32 v4, v6, v0 // (op 819) 


v_lshrrev_b32 v0, 6, v1 // (op 872) 
v_and_b32 v5, 1, v0 // (op 872) 

v_and_b32 v0, 15, v1 // (op 872) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 872) 


v_and_b32 v0, 63, v1 // (op 872) 
v_lshrrev_b32 v5, 4, v0 // (op 872) 

v_lshl_add_u32 v0, v5, 5, v6 // (op 872) 


v_lshrrev_b32 v5, 7, v1 // (op 930) 
v_and_b32 v6, 15, v1 // (op 930) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 930) 


v_and_b32 v5, 63, v1 // (op 930) 
v_lshrrev_b32 v6, 4, v5 // (op 930) 

v_lshlrev_b32 v5, 5, v6 // (op 930) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 930) 


v_mul_lo_u32 v5, v6, 6 // (op 930) 

v_lshrrev_b32 v6, 3, v5 // (op 930) 

v_and_b32 v5, 48, v1 // (op 969) 
v_and_b32 v7, 63, v1 // (op 969) 
v_sub_u32 v8, v7, v5 // (op 969) 

v_lshrrev_b32 v7, 1, v8 // (op 969) 
v_add_lshl_u32 v9, v5, v7, 5 // (op 969) 


v_and_b32 v5, 1, v8 // (op 969) 

v_lshlrev_b32 v7, 3, v5 // (op 969) 

v_lshrrev_b32 v5, 6, v1 // (op 969) 
v_and_b32 v8, 1, v5 // (op 969) 

v_lshl_add_u32 v5, v8, 4, v7 // (op 969) 


v_add_u32 v7, v9, v5 // (op 969) 


v_lshrrev_b32 v5, 6, v1 // (op 993) 
v_and_b32 v8, 1, v5 // (op 993) 

v_and_b32 v5, 15, v1 // (op 993) 
v_lshl_add_u32 v9, v8, 4, v5 // (op 993) 


v_and_b32 v5, 63, v1 // (op 993) 
v_lshrrev_b32 v8, 4, v5 // (op 993) 

v_lshl_add_u32 v5, v8, 5, v9 // (op 993) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v9, s9 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
s_mul_hi_i32 s0, s16, s11 // (op 477) 
v_lshrrev_b32 v10, 3, v1 // (op 477) 
s_add_i32 s1, s0, s16 // (op 477) 

v_and_b32 v11, -8, v1 // (op 477) 
s_ashr_i32 s0, s1, 31 // (op 477) 
v_sub_u32 v12, v1, v11 // (op 477) 

s_cmp_eq_i32 s11, 0 // (op 477) 
s_mov_b32 s8, scc // (op 477) 
v_lshlrev_b32 v11, 4, v12 // (op 477) 

s_cmp_eq_i32 s8, 0 // (op 477) 
s_cselect_b32 s9, 0, -1 // (op 477) 

v_mov_b32 v13, 0 // (op 477) convert
v_mov_b32 v12, v11 // (op 477) convert

s_and_b32 s8, s12, 31 // (op 477) 
s_lshl_b32 s18, 1, s8 // (op 477) 
s_add_i32 s19, s18, s9 // (op 477) 


s_and_b32 s9, s0, s19 // (op 477) 


s_add_i32 s0, s1, s9 // (op 477) 


s_ashr_i32 s1, s0, s8 // (op 477) 


s_lshl_b32 s0, s12, 24 // (op 477) 
s_ashr_i32 s8, s0, 31 // (op 477) 

s_xor_b32 s0, s1, s8 // (op 477) 

s_sub_i32 s1, s0, s8 // (op 477) 


s_mul_hi_i32 s0, s1, s13 // (op 477) 
s_add_i32 s8, s0, s1 // (op 477) 

s_ashr_i32 s0, s8, 31 // (op 477) 
s_cmp_eq_i32 s13, 0 // (op 477) 
s_mov_b32 s9, scc // (op 477) 
s_cmp_eq_i32 s9, 0 // (op 477) 
s_cselect_b32 s18, 0, -1 // (op 477) 

s_and_b32 s9, s14, 31 // (op 477) 
s_lshl_b32 s19, 1, s9 // (op 477) 
s_add_i32 s51, s19, s18 // (op 477) 


s_and_b32 s18, s0, s51 // (op 477) 


s_add_i32 s0, s8, s18 // (op 477) 


s_ashr_i32 s8, s0, s9 // (op 477) 


s_lshl_b32 s0, s14, 24 // (op 477) 
s_ashr_i32 s9, s0, 31 // (op 477) 

s_xor_b32 s0, s8, s9 // (op 477) 

s_sub_i32 s8, s0, s9 // (op 477) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s46 // (op 477) 

s_sub_u32 s1, s16, s0 // (op 477) 

s_mul_i32 s0, s8, s46 // (op 477) 
s_add_u32 s0, s0, s1 // (op 477) 


s_mul_hi_i32 s1, s16, s10 // (op 477) 
s_add_i32 s8, s1, s16 // (op 477) 

s_ashr_i32 s1, s8, 31 // (op 477) 
s_cmp_eq_i32 s10, 0 // (op 477) 
s_mov_b32 s9, scc // (op 477) 
s_cmp_eq_i32 s9, 0 // (op 477) 
s_cselect_b32 s18, 0, -1 // (op 477) 

s_and_b32 s9, s49, 31 // (op 477) 
s_lshl_b32 s19, 1, s9 // (op 477) 
s_add_i32 s51, s19, s18 // (op 477) 


s_and_b32 s18, s1, s51 // (op 477) 


s_add_i32 s1, s8, s18 // (op 477) 


s_ashr_i32 s8, s1, s9 // (op 477) 


s_lshl_b32 s1, s49, 24 // (op 477) 
s_ashr_i32 s9, s1, 31 // (op 477) 

s_xor_b32 s1, s8, s9 // (op 477) 

s_sub_i32 s8, s1, s9 // (op 477) 


s_add_u32 s18, s2, 32 // (op 477) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 477) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 477) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 477) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 5 // (op 477) 

s_mul_i32 s1, s46, s18 // (op 477) 

s_mul_i32 s9, s50, s1 // (op 477) 

s_mul_i32 s1, s8, s9 // (op 477) 

s_sub_u32 s9, s16, s1 // (op 477) 

s_mov_b32 s1, s9 // (op 477) 
s_mul_hi_i32 s18, s1, s15 // (op 477) 
s_add_i32 s19, s18, s1 // (op 477) 


s_ashr_i32 s1, s19, 31 // (op 477) 
s_cmp_eq_i32 s15, 0 // (op 477) 
s_mov_b32 s18, scc // (op 477) 
s_cmp_eq_i32 s18, 0 // (op 477) 
s_cselect_b32 s51, 0, -1 // (op 477) 

s_and_b32 s18, s48, 31 // (op 477) 
s_lshl_b32 s52, 1, s18 // (op 477) 
s_add_i32 s53, s52, s51 // (op 477) 


s_and_b32 s51, s1, s53 // (op 477) 


s_add_i32 s1, s19, s51 // (op 477) 


s_ashr_i32 s19, s1, s18 // (op 477) 


s_lshl_b32 s1, s48, 24 // (op 477) 
s_ashr_i32 s18, s1, 31 // (op 477) 

s_xor_b32 s1, s19, s18 // (op 477) 

s_sub_i32 s19, s1, s18 // (op 477) 


s_mul_i32 s1, s47, s46 // (op 477) 
s_add_u32 s52, s6, 16 // (op 477) Start of Int64 add, locking SCC
s_addc_u32 s53, s7, 0 // (op 477) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 477) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 477) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 477) 

s_sub_i32 s18, s52, s1 // (op 477) 

s_mul_i32 s51, s19, s18 // (op 477) 


s_sub_u32 s18, s9, s51 // (op 477) 


s_add_u32 s9, s18, s1 // (op 477) 


s_cmp_eq_i32 s8, 0 // (op 477) 
s_mov_b32 s1, scc // (op 477) 

s_cmp_eq_i32 s1, 0 // (op 477) 
s_cselect_b32 s8, s9, s0 // (op 477) 



v_lshl_add_u32 v11, s8, 4, v10 // (op 477) 


v_mul_lo_u32 v15, v11, s5 // (op 477) most significant: low of low * high
v_mul_hi_u32 v10, v11, s4 // (op 477) most significant: high of low * low
v_mul_lo_u32 v14, v11, s4 // (op 477) least significant: low of low * low
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

s_mul_i32 s9, s4, s1 // (op 479) most significant: low of low * high
s_mul_i32 s18, s5, s0 // (op 479) most significant: low of high * low
s_mul_i32 s8, s4, s0 // (op 479) least significant
s_mul_hi_u32 s19, s4, s0 // (op 479) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 479) most significant: sum
s_add_u32 s9, s9, s19 // (op 479) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 479) 

s_mov_b32 s52, s20 // (op 479) reg expression
s_mov_b32 s53, s21 // (op 479) reg expression
s_mov_b32 s54, s0 // (op 479) reg expression
s_mov_b32 s55, 131072 // (op 479) call()

s_mul_hi_i32 s51, s16, s11 // (op 496) 
v_lshrrev_b32 v11, 3, v1 // (op 496) 
s_add_i32 s0, s51, s16 // (op 496) 

v_and_b32 v12, -8, v1 // (op 496) 
s_ashr_i32 s1, s0, 31 // (op 496) 
v_sub_u32 v13, v1, v12 // (op 496) 

s_cmp_eq_i32 s11, 0 // (op 496) 
s_mov_b32 s51, scc // (op 496) 
v_lshlrev_b32 v12, 4, v13 // (op 496) 

s_cmp_eq_i32 s51, 0 // (op 496) 
s_cselect_b32 s8, 0, -1 // (op 496) 

v_mov_b32 v15, 0 // (op 496) convert
v_mov_b32 v14, v12 // (op 496) convert

s_and_b32 s9, s12, 31 // (op 496) 
s_lshl_b32 s51, 1, s9 // (op 496) 
s_add_i32 s18, s51, s8 // (op 496) 


s_and_b32 s8, s1, s18 // (op 496) 


s_add_i32 s1, s0, s8 // (op 496) 


s_ashr_i32 s0, s1, s9 // (op 496) 


s_lshl_b32 s1, s12, 24 // (op 496) 
s_ashr_i32 s51, s1, 31 // (op 496) 

s_xor_b32 s1, s0, s51 // (op 496) 

s_sub_i32 s0, s1, s51 // (op 496) 


s_mul_hi_i32 s1, s0, s13 // (op 496) 
s_add_i32 s51, s1, s0 // (op 496) 

s_ashr_i32 s1, s51, 31 // (op 496) 
s_cmp_eq_i32 s13, 0 // (op 496) 
s_mov_b32 s8, scc // (op 496) 
s_cmp_eq_i32 s8, 0 // (op 496) 
s_cselect_b32 s9, 0, -1 // (op 496) 

s_and_b32 s8, s14, 31 // (op 496) 
s_lshl_b32 s18, 1, s8 // (op 496) 
s_add_i32 s19, s18, s9 // (op 496) 


s_and_b32 s9, s1, s19 // (op 496) 


s_add_i32 s1, s51, s9 // (op 496) 


s_ashr_i32 s9, s1, s8 // (op 496) 


s_lshl_b32 s1, s14, 24 // (op 496) 
s_ashr_i32 s8, s1, 31 // (op 496) 

s_xor_b32 s1, s9, s8 // (op 496) 

s_sub_i32 s9, s1, s8 // (op 496) 


s_mul_i32 s1, s0, s46 // (op 496) 

s_sub_u32 s0, s16, s1 // (op 496) 

s_mul_i32 s1, s9, s46 // (op 496) 
s_add_u32 s1, s1, s0 // (op 496) 


s_mul_hi_i32 s0, s16, s10 // (op 496) 
s_add_i32 s51, s0, s16 // (op 496) 

s_ashr_i32 s0, s51, 31 // (op 496) 
s_cmp_eq_i32 s10, 0 // (op 496) 
s_mov_b32 s8, scc // (op 496) 
s_cmp_eq_i32 s8, 0 // (op 496) 
s_cselect_b32 s9, 0, -1 // (op 496) 

s_and_b32 s8, s49, 31 // (op 496) 
s_lshl_b32 s18, 1, s8 // (op 496) 
s_add_i32 s19, s18, s9 // (op 496) 


s_and_b32 s9, s0, s19 // (op 496) 


s_add_i32 s0, s51, s9 // (op 496) 


s_ashr_i32 s9, s0, s8 // (op 496) 


s_lshl_b32 s0, s49, 24 // (op 496) 
s_ashr_i32 s8, s0, 31 // (op 496) 

s_xor_b32 s0, s9, s8 // (op 496) 

s_sub_i32 s9, s0, s8 // (op 496) 


s_add_u32 s18, s2, 32 // (op 496) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 496) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 496) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 496) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 496) 

s_mul_i32 s0, s46, s18 // (op 496) 

s_mul_i32 s8, s50, s0 // (op 496) 

s_mul_i32 s0, s9, s8 // (op 496) 

s_sub_u32 s8, s16, s0 // (op 496) 

s_mov_b32 s0, s8 // (op 496) 
s_mul_hi_i32 s51, s0, s15 // (op 496) 
s_add_i32 s18, s51, s0 // (op 496) 


s_ashr_i32 s0, s18, 31 // (op 496) 
s_cmp_eq_i32 s15, 0 // (op 496) 
s_mov_b32 s19, scc // (op 496) 
s_cmp_eq_i32 s19, 0 // (op 496) 
s_cselect_b32 s51, 0, -1 // (op 496) 

s_and_b32 s19, s48, 31 // (op 496) 
s_lshl_b32 s56, 1, s19 // (op 496) 
s_add_i32 s57, s56, s51 // (op 496) 


s_and_b32 s51, s0, s57 // (op 496) 


s_add_i32 s0, s18, s51 // (op 496) 


s_ashr_i32 s18, s0, s19 // (op 496) 


s_lshl_b32 s0, s48, 24 // (op 496) 
s_ashr_i32 s19, s0, 31 // (op 496) 

s_xor_b32 s0, s18, s19 // (op 496) 

s_sub_i32 s18, s0, s19 // (op 496) 


s_mul_i32 s0, s47, s46 // (op 496) 
s_add_u32 s56, s6, 16 // (op 496) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 496) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 496) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 496) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 496) 

s_sub_i32 s19, s56, s0 // (op 496) 

s_mul_i32 s51, s18, s19 // (op 496) 


s_sub_u32 s18, s8, s51 // (op 496) 


s_add_u32 s8, s18, s0 // (op 496) 


s_cmp_eq_i32 s9, 0 // (op 496) 
s_mov_b32 s0, scc // (op 496) 

s_cmp_eq_i32 s0, 0 // (op 496) 
s_cselect_b32 s9, s8, s1 // (op 496) 



v_lshl_add_u32 v12, s9, 4, v11 // (op 496) 


v_mul_lo_u32 v17, v12, s5 // (op 496) most significant: low of low * high
v_mul_hi_u32 v11, v12, s4 // (op 496) most significant: high of low * low
v_mul_lo_u32 v16, v12, s4 // (op 496) least significant: low of low * low
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
s_mul_hi_u32 s51, s6, 6 // (op 498) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 498) most significant: sum

s_mul_i32 s9, s4, s1 // (op 498) most significant: low of low * high
s_mul_i32 s51, s5, s0 // (op 498) most significant: low of high * low
s_mul_i32 s8, s4, s0 // (op 498) least significant
s_mul_hi_u32 s18, s4, s0 // (op 498) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 498) most significant: sum
s_add_u32 s9, s9, s18 // (op 498) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 498) 

s_mov_b32 s52, s20 // (op 498) reg expression
s_mov_b32 s53, s21 // (op 498) reg expression
s_mov_b32 s54, s0 // (op 498) reg expression
s_mov_b32 s55, 131072 // (op 498) call()

s_mul_hi_i32 s51, s16, s11 // (op 541) 
v_and_b32 v12, -2, v1 // (op 541) 
s_add_i32 s0, s51, s16 // (op 541) 

v_sub_u32 v13, v1, v12 // (op 541) 

s_ashr_i32 s1, s0, 31 // (op 541) 
v_lshlrev_b32 v12, 4, v13 // (op 541) 

s_cmp_eq_i32 s11, 0 // (op 541) 
s_mov_b32 s51, scc // (op 541) 
v_lshrrev_b32 v13, 1, v1 // (op 541) 
s_cmp_eq_i32 s51, 0 // (op 541) 
s_cselect_b32 s8, 0, -1 // (op 541) 

s_and_b32 s9, s12, 31 // (op 541) 
s_lshl_b32 s51, 1, s9 // (op 541) 
s_add_i32 s18, s51, s8 // (op 541) 


s_and_b32 s8, s1, s18 // (op 541) 


s_add_i32 s1, s0, s8 // (op 541) 


s_ashr_i32 s0, s1, s9 // (op 541) 


s_lshl_b32 s1, s12, 24 // (op 541) 
s_ashr_i32 s51, s1, 31 // (op 541) 

s_xor_b32 s1, s0, s51 // (op 541) 

s_sub_i32 s0, s1, s51 // (op 541) 


s_mul_hi_i32 s1, s0, s13 // (op 541) 
s_add_i32 s51, s1, s0 // (op 541) 

s_ashr_i32 s1, s51, 31 // (op 541) 
s_cmp_eq_i32 s13, 0 // (op 541) 
s_mov_b32 s8, scc // (op 541) 
s_cmp_eq_i32 s8, 0 // (op 541) 
s_cselect_b32 s9, 0, -1 // (op 541) 

s_and_b32 s8, s14, 31 // (op 541) 
s_lshl_b32 s18, 1, s8 // (op 541) 
s_add_i32 s19, s18, s9 // (op 541) 


s_and_b32 s9, s1, s19 // (op 541) 


s_add_i32 s1, s51, s9 // (op 541) 


s_ashr_i32 s9, s1, s8 // (op 541) 


s_lshl_b32 s1, s14, 24 // (op 541) 
s_ashr_i32 s8, s1, 31 // (op 541) 

s_xor_b32 s1, s9, s8 // (op 541) 

s_sub_i32 s9, s1, s8 // (op 541) 


s_add_u32 s18, s2, 32 // (op 541) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 541) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 541) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 541) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 541) 

s_mul_i32 s1, s9, s18 // (op 541) 

s_sub_i32 s51, s0, s1 // (op 541) 


s_mul_hi_i32 s0, s16, s10 // (op 541) 
s_add_i32 s1, s0, s16 // (op 541) 

s_ashr_i32 s0, s1, 31 // (op 541) 
s_cmp_eq_i32 s10, 0 // (op 541) 
s_mov_b32 s8, scc // (op 541) 
s_cmp_eq_i32 s8, 0 // (op 541) 
s_cselect_b32 s9, 0, -1 // (op 541) 

s_and_b32 s8, s49, 31 // (op 541) 
s_lshl_b32 s56, 1, s8 // (op 541) 
s_add_i32 s57, s56, s9 // (op 541) 


s_and_b32 s9, s0, s57 // (op 541) 


s_add_i32 s0, s1, s9 // (op 541) 


s_ashr_i32 s1, s0, s8 // (op 541) 


s_lshl_b32 s0, s49, 24 // (op 541) 
s_ashr_i32 s8, s0, 31 // (op 541) 

s_xor_b32 s0, s1, s8 // (op 541) 

s_sub_i32 s1, s0, s8 // (op 541) 


s_mul_i32 s0, s46, s18 // (op 541) 

s_mul_i32 s8, s50, s0 // (op 541) 

s_mul_i32 s0, s1, s8 // (op 541) 

s_sub_u32 s8, s16, s0 // (op 541) 

s_mul_hi_i32 s0, s8, s15 // (op 541) 
s_add_i32 s9, s0, s8 // (op 541) 


s_ashr_i32 s0, s9, 31 // (op 541) 
s_cmp_eq_i32 s15, 0 // (op 541) 
s_mov_b32 s8, scc // (op 541) 
s_cmp_eq_i32 s8, 0 // (op 541) 
s_cselect_b32 s18, 0, -1 // (op 541) 

s_and_b32 s8, s48, 31 // (op 541) 
s_lshl_b32 s19, 1, s8 // (op 541) 
s_add_i32 s56, s19, s18 // (op 541) 


s_and_b32 s18, s0, s56 // (op 541) 


s_add_i32 s0, s9, s18 // (op 541) 


s_ashr_i32 s9, s0, s8 // (op 541) 


s_lshl_b32 s0, s48, 24 // (op 541) 
s_ashr_i32 s8, s0, 31 // (op 541) 

s_xor_b32 s0, s9, s8 // (op 541) 

s_sub_i32 s9, s0, s8 // (op 541) 


s_cmp_eq_i32 s1, 0 // (op 541) 
s_mov_b32 s0, scc // (op 541) 

s_cmp_eq_i32 s0, 0 // (op 541) 
s_cselect_b32 s1, s9, s51 // (op 541) 



v_lshl_add_u32 v14, s1, 5, v12 // (op 541) 


v_mov_b32 v17, 0 // (op 541) convert
v_mov_b32 v16, v14 // (op 541) convert

v_mul_lo_u32 v15, v13, s23 // (op 541) most significant: low of low * high
v_mul_hi_u32 v12, v13, s22 // (op 541) most significant: high of low * low
v_mul_lo_u32 v14, v13, s22 // (op 541) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 541) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v16 // (op 541) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 541) most significant half



s_mul_i32 s1, s22, s27 // (op 543) most significant: low of low * high
s_mul_i32 s51, s23, s26 // (op 543) most significant: low of high * low
s_mul_i32 s0, s22, s26 // (op 543) least significant
s_mul_hi_u32 s8, s22, s26 // (op 543) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 543) most significant: sum
s_add_u32 s1, s1, s8 // (op 543) most significant: sum


s_mov_b32 s56, s24 // (op 543) reg expression
s_mov_b32 s57, s25 // (op 543) reg expression
s_mov_b32 s58, s0 // (op 543) reg expression
s_mov_b32 s59, 131072 // (op 543) call()

s_mul_hi_i32 s51, s16, s11 // (op 560) 
v_and_b32 v12, -2, v1 // (op 560) 
s_add_i32 s0, s51, s16 // (op 560) 

v_sub_u32 v13, v1, v12 // (op 560) 

s_ashr_i32 s1, s0, 31 // (op 560) 
v_lshlrev_b32 v12, 4, v13 // (op 560) 

s_cmp_eq_i32 s11, 0 // (op 560) 
s_mov_b32 s51, scc // (op 560) 
v_lshrrev_b32 v13, 1, v1 // (op 560) 
s_cmp_eq_i32 s51, 0 // (op 560) 
s_cselect_b32 s8, 0, -1 // (op 560) 

s_and_b32 s9, s12, 31 // (op 560) 
s_lshl_b32 s51, 1, s9 // (op 560) 
s_add_i32 s18, s51, s8 // (op 560) 


s_and_b32 s8, s1, s18 // (op 560) 


s_add_i32 s1, s0, s8 // (op 560) 


s_ashr_i32 s0, s1, s9 // (op 560) 


s_lshl_b32 s1, s12, 24 // (op 560) 
s_ashr_i32 s51, s1, 31 // (op 560) 

s_xor_b32 s1, s0, s51 // (op 560) 

s_sub_i32 s0, s1, s51 // (op 560) 


s_mul_hi_i32 s1, s0, s13 // (op 560) 
s_add_i32 s51, s1, s0 // (op 560) 

s_ashr_i32 s1, s51, 31 // (op 560) 
s_cmp_eq_i32 s13, 0 // (op 560) 
s_mov_b32 s8, scc // (op 560) 
s_cmp_eq_i32 s8, 0 // (op 560) 
s_cselect_b32 s9, 0, -1 // (op 560) 

s_and_b32 s8, s14, 31 // (op 560) 
s_lshl_b32 s18, 1, s8 // (op 560) 
s_add_i32 s19, s18, s9 // (op 560) 


s_and_b32 s9, s1, s19 // (op 560) 


s_add_i32 s1, s51, s9 // (op 560) 


s_ashr_i32 s9, s1, s8 // (op 560) 


s_lshl_b32 s1, s14, 24 // (op 560) 
s_ashr_i32 s8, s1, 31 // (op 560) 

s_xor_b32 s1, s9, s8 // (op 560) 

s_sub_i32 s9, s1, s8 // (op 560) 


s_add_u32 s18, s2, 32 // (op 560) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 560) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 560) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 560) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 560) 

s_mul_i32 s1, s9, s18 // (op 560) 

s_sub_i32 s51, s0, s1 // (op 560) 


s_mul_hi_i32 s0, s16, s10 // (op 560) 
s_add_i32 s1, s0, s16 // (op 560) 

s_ashr_i32 s0, s1, 31 // (op 560) 
s_cmp_eq_i32 s10, 0 // (op 560) 
s_mov_b32 s8, scc // (op 560) 
s_cmp_eq_i32 s8, 0 // (op 560) 
s_cselect_b32 s9, 0, -1 // (op 560) 

s_and_b32 s8, s49, 31 // (op 560) 
s_lshl_b32 s60, 1, s8 // (op 560) 
s_add_i32 s61, s60, s9 // (op 560) 


s_and_b32 s9, s0, s61 // (op 560) 


s_add_i32 s0, s1, s9 // (op 560) 


s_ashr_i32 s1, s0, s8 // (op 560) 


s_lshl_b32 s0, s49, 24 // (op 560) 
s_ashr_i32 s8, s0, 31 // (op 560) 

s_xor_b32 s0, s1, s8 // (op 560) 

s_sub_i32 s1, s0, s8 // (op 560) 


s_mul_i32 s0, s46, s18 // (op 560) 

s_mul_i32 s8, s50, s0 // (op 560) 

s_mul_i32 s0, s1, s8 // (op 560) 

s_sub_u32 s8, s16, s0 // (op 560) 

s_mul_hi_i32 s0, s8, s15 // (op 560) 
s_add_i32 s9, s0, s8 // (op 560) 


s_ashr_i32 s0, s9, 31 // (op 560) 
s_cmp_eq_i32 s15, 0 // (op 560) 
s_mov_b32 s8, scc // (op 560) 
s_cmp_eq_i32 s8, 0 // (op 560) 
s_cselect_b32 s18, 0, -1 // (op 560) 

s_and_b32 s8, s48, 31 // (op 560) 
s_lshl_b32 s19, 1, s8 // (op 560) 
s_add_i32 s60, s19, s18 // (op 560) 


s_and_b32 s18, s0, s60 // (op 560) 


s_add_i32 s0, s9, s18 // (op 560) 


s_ashr_i32 s9, s0, s8 // (op 560) 


s_lshl_b32 s0, s48, 24 // (op 560) 
s_ashr_i32 s8, s0, 31 // (op 560) 

s_xor_b32 s0, s9, s8 // (op 560) 

s_sub_i32 s9, s0, s8 // (op 560) 


s_cmp_eq_i32 s1, 0 // (op 560) 
s_mov_b32 s0, scc // (op 560) 

s_cmp_eq_i32 s0, 0 // (op 560) 
s_cselect_b32 s1, s9, s51 // (op 560) 



v_lshl_add_u32 v16, s1, 5, v12 // (op 560) 


v_mov_b32 v19, 0 // (op 560) convert
v_mov_b32 v18, v16 // (op 560) convert

v_mul_lo_u32 v17, v13, s23 // (op 560) most significant: low of low * high
v_mul_hi_u32 v12, v13, s22 // (op 560) most significant: high of low * low
v_mul_lo_u32 v16, v13, s22 // (op 560) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 560) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v18 // (op 560) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 560) most significant half



s_mul_i32 s1, s22, s27 // (op 562) most significant: low of low * high
s_mul_i32 s51, s23, s26 // (op 562) most significant: low of high * low
s_mul_i32 s0, s22, s26 // (op 562) least significant
s_mul_hi_u32 s8, s22, s26 // (op 562) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 562) most significant: sum
s_add_u32 s1, s1, s8 // (op 562) most significant: sum


s_mov_b32 s56, s24 // (op 562) reg expression
s_mov_b32 s57, s25 // (op 562) reg expression
s_mov_b32 s58, s0 // (op 562) reg expression
s_mov_b32 s59, 131072 // (op 562) call()

s_mul_hi_i32 s51, s16, s11 // (op 605) 
v_lshrrev_b32 v12, 7, v1 // (op 605) 
s_add_i32 s0, s51, s16 // (op 605) 

v_and_b32 v13, 15, v1 // (op 605) 
s_ashr_i32 s1, s0, 31 // (op 605) 
v_lshl_add_u32 v18, v12, 4, v13 // (op 605) 


s_cmp_eq_i32 s11, 0 // (op 605) 
s_mov_b32 s51, scc // (op 605) 
v_and_b32 v12, 63, v1 // (op 605) 
s_cmp_eq_i32 s51, 0 // (op 605) 
s_cselect_b32 s8, 0, -1 // (op 605) 

v_lshrrev_b32 v13, 4, v12 // (op 605) 

s_and_b32 s9, s12, 31 // (op 605) 
s_lshl_b32 s51, 1, s9 // (op 605) 
s_add_i32 s18, s51, s8 // (op 605) 


s_and_b32 s8, s1, s18 // (op 605) 


s_add_i32 s1, s0, s8 // (op 605) 


s_ashr_i32 s0, s1, s9 // (op 605) 


s_lshl_b32 s1, s12, 24 // (op 605) 
s_ashr_i32 s51, s1, 31 // (op 605) 

s_xor_b32 s1, s0, s51 // (op 605) 

s_sub_i32 s0, s1, s51 // (op 605) 


s_mul_hi_i32 s1, s0, s13 // (op 605) 
s_add_i32 s51, s1, s0 // (op 605) 

s_ashr_i32 s1, s51, 31 // (op 605) 
s_cmp_eq_i32 s13, 0 // (op 605) 
s_mov_b32 s8, scc // (op 605) 
s_cmp_eq_i32 s8, 0 // (op 605) 
s_cselect_b32 s9, 0, -1 // (op 605) 

s_and_b32 s8, s14, 31 // (op 605) 
s_lshl_b32 s18, 1, s8 // (op 605) 
s_add_i32 s19, s18, s9 // (op 605) 


s_and_b32 s9, s1, s19 // (op 605) 


s_add_i32 s1, s51, s9 // (op 605) 


s_ashr_i32 s9, s1, s8 // (op 605) 


s_lshl_b32 s1, s14, 24 // (op 605) 
s_ashr_i32 s8, s1, 31 // (op 605) 

s_xor_b32 s1, s9, s8 // (op 605) 

s_sub_i32 s9, s1, s8 // (op 605) 


s_mul_i32 s1, s0, s46 // (op 605) 

s_sub_u32 s0, s16, s1 // (op 605) 

s_mul_i32 s1, s9, s46 // (op 605) 
s_add_u32 s1, s1, s0 // (op 605) 


s_mul_hi_i32 s0, s16, s10 // (op 605) 
s_add_i32 s51, s0, s16 // (op 605) 

s_ashr_i32 s0, s51, 31 // (op 605) 
s_cmp_eq_i32 s10, 0 // (op 605) 
s_mov_b32 s8, scc // (op 605) 
s_cmp_eq_i32 s8, 0 // (op 605) 
s_cselect_b32 s9, 0, -1 // (op 605) 

s_and_b32 s8, s49, 31 // (op 605) 
s_lshl_b32 s18, 1, s8 // (op 605) 
s_add_i32 s19, s18, s9 // (op 605) 


s_and_b32 s9, s0, s19 // (op 605) 


s_add_i32 s0, s51, s9 // (op 605) 


s_ashr_i32 s9, s0, s8 // (op 605) 


s_lshl_b32 s0, s49, 24 // (op 605) 
s_ashr_i32 s8, s0, 31 // (op 605) 

s_xor_b32 s0, s9, s8 // (op 605) 

s_sub_i32 s9, s0, s8 // (op 605) 


s_add_u32 s18, s2, 32 // (op 605) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 605) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 605) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 605) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 605) 

s_mul_i32 s0, s46, s18 // (op 605) 

s_mul_i32 s8, s50, s0 // (op 605) 

s_mul_i32 s0, s9, s8 // (op 605) 

s_sub_u32 s8, s16, s0 // (op 605) 

s_mov_b32 s0, s8 // (op 605) 
s_mul_hi_i32 s51, s0, s15 // (op 605) 
s_add_i32 s18, s51, s0 // (op 605) 


s_ashr_i32 s0, s18, 31 // (op 605) 
s_cmp_eq_i32 s15, 0 // (op 605) 
s_mov_b32 s19, scc // (op 605) 
s_cmp_eq_i32 s19, 0 // (op 605) 
s_cselect_b32 s51, 0, -1 // (op 605) 

s_and_b32 s19, s48, 31 // (op 605) 
s_lshl_b32 s60, 1, s19 // (op 605) 
s_add_i32 s61, s60, s51 // (op 605) 


s_and_b32 s51, s0, s61 // (op 605) 


s_add_i32 s0, s18, s51 // (op 605) 


s_ashr_i32 s18, s0, s19 // (op 605) 


s_lshl_b32 s0, s48, 24 // (op 605) 
s_ashr_i32 s19, s0, 31 // (op 605) 

s_xor_b32 s0, s18, s19 // (op 605) 

s_sub_i32 s18, s0, s19 // (op 605) 


s_mul_i32 s0, s47, s46 // (op 605) 
s_add_u32 s60, s6, 16 // (op 605) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 605) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 605) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 605) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 605) 

s_sub_i32 s19, s60, s0 // (op 605) 

s_mul_i32 s51, s18, s19 // (op 605) 


s_sub_u32 s18, s8, s51 // (op 605) 


s_add_u32 s8, s18, s0 // (op 605) 


s_cmp_eq_i32 s9, 0 // (op 605) 
s_mov_b32 s0, scc // (op 605) 

s_cmp_eq_i32 s0, 0 // (op 605) 
s_cselect_b32 s9, s8, s1 // (op 605) 



v_lshl_add_u32 v12, s9, 4, v18 // (op 605) 


v_mul_lo_u32 v18, v12, s28 // (op 605) 
v_add_u32 v18, v18, v13 // (op 605) 


s_mul_i32 s1, s28, s7 // (op 607) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 607) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 607) least significant
s_mul_hi_u32 s8, s28, s6 // (op 607) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 607) most significant: sum
s_add_u32 s1, s1, s8 // (op 607) most significant: sum


s_mov_b32 s60, s30 // (op 607) reg expression
s_mov_b32 s61, s31 // (op 607) reg expression
s_mov_b32 s62, s0 // (op 607) reg expression
s_mov_b32 s63, 131072 // (op 607) call()

s_mul_hi_i32 s51, s16, s11 // (op 624) 
v_lshrrev_b32 v12, 7, v1 // (op 624) 
s_add_i32 s0, s51, s16 // (op 624) 

v_and_b32 v13, 15, v1 // (op 624) 
s_ashr_i32 s1, s0, 31 // (op 624) 
v_lshl_add_u32 v19, v12, 4, v13 // (op 624) 


s_cmp_eq_i32 s11, 0 // (op 624) 
s_mov_b32 s51, scc // (op 624) 
v_and_b32 v12, 63, v1 // (op 624) 
s_cmp_eq_i32 s51, 0 // (op 624) 
s_cselect_b32 s8, 0, -1 // (op 624) 

v_lshrrev_b32 v13, 4, v12 // (op 624) 

s_and_b32 s9, s12, 31 // (op 624) 
s_lshl_b32 s51, 1, s9 // (op 624) 
s_add_i32 s18, s51, s8 // (op 624) 


s_and_b32 s8, s1, s18 // (op 624) 


s_add_i32 s1, s0, s8 // (op 624) 


s_ashr_i32 s0, s1, s9 // (op 624) 


s_lshl_b32 s1, s12, 24 // (op 624) 
s_ashr_i32 s51, s1, 31 // (op 624) 

s_xor_b32 s1, s0, s51 // (op 624) 

s_sub_i32 s0, s1, s51 // (op 624) 


s_mul_hi_i32 s1, s0, s13 // (op 624) 
s_add_i32 s51, s1, s0 // (op 624) 

s_ashr_i32 s1, s51, 31 // (op 624) 
s_cmp_eq_i32 s13, 0 // (op 624) 
s_mov_b32 s8, scc // (op 624) 
s_cmp_eq_i32 s8, 0 // (op 624) 
s_cselect_b32 s9, 0, -1 // (op 624) 

s_and_b32 s8, s14, 31 // (op 624) 
s_lshl_b32 s18, 1, s8 // (op 624) 
s_add_i32 s19, s18, s9 // (op 624) 


s_and_b32 s9, s1, s19 // (op 624) 


s_add_i32 s1, s51, s9 // (op 624) 


s_ashr_i32 s9, s1, s8 // (op 624) 


s_lshl_b32 s1, s14, 24 // (op 624) 
s_ashr_i32 s8, s1, 31 // (op 624) 

s_xor_b32 s1, s9, s8 // (op 624) 

s_sub_i32 s9, s1, s8 // (op 624) 


s_mul_i32 s1, s0, s46 // (op 624) 

s_sub_u32 s0, s16, s1 // (op 624) 

s_mul_i32 s1, s9, s46 // (op 624) 
s_add_u32 s1, s1, s0 // (op 624) 


s_mul_hi_i32 s0, s16, s10 // (op 624) 
s_add_i32 s51, s0, s16 // (op 624) 

s_ashr_i32 s0, s51, 31 // (op 624) 
s_cmp_eq_i32 s10, 0 // (op 624) 
s_mov_b32 s8, scc // (op 624) 
s_cmp_eq_i32 s8, 0 // (op 624) 
s_cselect_b32 s9, 0, -1 // (op 624) 

s_and_b32 s8, s49, 31 // (op 624) 
s_lshl_b32 s18, 1, s8 // (op 624) 
s_add_i32 s19, s18, s9 // (op 624) 


s_and_b32 s9, s0, s19 // (op 624) 


s_add_i32 s0, s51, s9 // (op 624) 


s_ashr_i32 s9, s0, s8 // (op 624) 


s_lshl_b32 s0, s49, 24 // (op 624) 
s_ashr_i32 s8, s0, 31 // (op 624) 

s_xor_b32 s0, s9, s8 // (op 624) 

s_sub_i32 s9, s0, s8 // (op 624) 


s_add_u32 s18, s2, 32 // (op 624) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 624) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 624) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 624) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 624) 

s_mul_i32 s0, s46, s18 // (op 624) 

s_mul_i32 s8, s50, s0 // (op 624) 

s_mul_i32 s0, s9, s8 // (op 624) 

s_sub_u32 s8, s16, s0 // (op 624) 

s_mov_b32 s0, s8 // (op 624) 
s_mul_hi_i32 s51, s0, s15 // (op 624) 
s_add_i32 s18, s51, s0 // (op 624) 


s_ashr_i32 s0, s18, 31 // (op 624) 
s_cmp_eq_i32 s15, 0 // (op 624) 
s_mov_b32 s19, scc // (op 624) 
s_cmp_eq_i32 s19, 0 // (op 624) 
s_cselect_b32 s51, 0, -1 // (op 624) 

s_and_b32 s19, s48, 31 // (op 624) 
s_lshl_b32 s64, 1, s19 // (op 624) 
s_add_i32 s65, s64, s51 // (op 624) 


s_and_b32 s51, s0, s65 // (op 624) 


s_add_i32 s0, s18, s51 // (op 624) 


s_ashr_i32 s18, s0, s19 // (op 624) 


s_lshl_b32 s0, s48, 24 // (op 624) 
s_ashr_i32 s19, s0, 31 // (op 624) 

s_xor_b32 s0, s18, s19 // (op 624) 

s_sub_i32 s18, s0, s19 // (op 624) 


s_mul_i32 s0, s47, s46 // (op 624) 
s_add_u32 s64, s6, 16 // (op 624) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 624) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 624) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 624) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 624) 

s_sub_i32 s19, s64, s0 // (op 624) 

s_mul_i32 s51, s18, s19 // (op 624) 


s_sub_u32 s18, s8, s51 // (op 624) 


s_add_u32 s8, s18, s0 // (op 624) 


s_cmp_eq_i32 s9, 0 // (op 624) 
s_mov_b32 s0, scc // (op 624) 

s_cmp_eq_i32 s0, 0 // (op 624) 
s_cselect_b32 s9, s8, s1 // (op 624) 



v_lshl_add_u32 v12, s9, 4, v19 // (op 624) 


v_mul_lo_u32 v19, v12, s28 // (op 624) 
v_add_u32 v19, v19, v13 // (op 624) 


s_mul_i32 s1, s28, s7 // (op 626) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 626) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 626) least significant
s_mul_hi_u32 s8, s28, s6 // (op 626) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 626) most significant: sum
s_add_u32 s1, s1, s8 // (op 626) most significant: sum


s_mov_b32 s60, s30 // (op 626) reg expression
s_mov_b32 s61, s31 // (op 626) reg expression
s_mov_b32 s62, s0 // (op 626) reg expression
s_mov_b32 s63, 131072 // (op 626) call()

s_mul_hi_i32 s51, s16, s11 // (op 669) 
v_and_b32 v12, -32, v1 // (op 669) 
s_add_i32 s0, s51, s16 // (op 669) 

v_sub_u32 v13, v1, v12 // (op 669) 

s_ashr_i32 s1, s0, 31 // (op 669) 
v_lshrrev_b32 v12, 5, v1 // (op 669) 
s_cmp_eq_i32 s11, 0 // (op 669) 
s_mov_b32 s51, scc // (op 669) 
s_cmp_eq_i32 s51, 0 // (op 669) 
s_cselect_b32 s8, 0, -1 // (op 669) 

s_and_b32 s9, s12, 31 // (op 669) 
s_lshl_b32 s51, 1, s9 // (op 669) 
s_add_i32 s18, s51, s8 // (op 669) 


s_and_b32 s8, s1, s18 // (op 669) 


s_add_i32 s1, s0, s8 // (op 669) 


s_ashr_i32 s0, s1, s9 // (op 669) 


s_lshl_b32 s1, s12, 24 // (op 669) 
s_ashr_i32 s51, s1, 31 // (op 669) 

s_xor_b32 s1, s0, s51 // (op 669) 

s_sub_i32 s0, s1, s51 // (op 669) 


s_mul_hi_i32 s1, s0, s13 // (op 669) 
s_add_i32 s51, s1, s0 // (op 669) 

s_ashr_i32 s1, s51, 31 // (op 669) 
s_cmp_eq_i32 s13, 0 // (op 669) 
s_mov_b32 s8, scc // (op 669) 
s_cmp_eq_i32 s8, 0 // (op 669) 
s_cselect_b32 s9, 0, -1 // (op 669) 

s_and_b32 s8, s14, 31 // (op 669) 
s_lshl_b32 s18, 1, s8 // (op 669) 
s_add_i32 s19, s18, s9 // (op 669) 


s_and_b32 s9, s1, s19 // (op 669) 


s_add_i32 s1, s51, s9 // (op 669) 


s_ashr_i32 s9, s1, s8 // (op 669) 


s_lshl_b32 s1, s14, 24 // (op 669) 
s_ashr_i32 s8, s1, 31 // (op 669) 

s_xor_b32 s1, s9, s8 // (op 669) 

s_sub_i32 s9, s1, s8 // (op 669) 


s_add_u32 s18, s2, 32 // (op 669) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 669) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 669) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 669) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 669) 

s_mul_i32 s1, s9, s18 // (op 669) 

s_sub_i32 s51, s0, s1 // (op 669) 


s_mul_hi_i32 s0, s16, s10 // (op 669) 
s_add_i32 s1, s0, s16 // (op 669) 

s_ashr_i32 s0, s1, 31 // (op 669) 
s_cmp_eq_i32 s10, 0 // (op 669) 
s_mov_b32 s8, scc // (op 669) 
s_cmp_eq_i32 s8, 0 // (op 669) 
s_cselect_b32 s9, 0, -1 // (op 669) 

s_and_b32 s8, s49, 31 // (op 669) 
s_lshl_b32 s64, 1, s8 // (op 669) 
s_add_i32 s65, s64, s9 // (op 669) 


s_and_b32 s9, s0, s65 // (op 669) 


s_add_i32 s0, s1, s9 // (op 669) 


s_ashr_i32 s1, s0, s8 // (op 669) 


s_lshl_b32 s0, s49, 24 // (op 669) 
s_ashr_i32 s8, s0, 31 // (op 669) 

s_xor_b32 s0, s1, s8 // (op 669) 

s_sub_i32 s1, s0, s8 // (op 669) 


s_mul_i32 s0, s46, s18 // (op 669) 

s_mul_i32 s8, s50, s0 // (op 669) 

s_mul_i32 s0, s1, s8 // (op 669) 

s_sub_u32 s8, s16, s0 // (op 669) 

s_mul_hi_i32 s0, s8, s15 // (op 669) 
s_add_i32 s9, s0, s8 // (op 669) 


s_ashr_i32 s0, s9, 31 // (op 669) 
s_cmp_eq_i32 s15, 0 // (op 669) 
s_mov_b32 s8, scc // (op 669) 
s_cmp_eq_i32 s8, 0 // (op 669) 
s_cselect_b32 s18, 0, -1 // (op 669) 

s_and_b32 s8, s48, 31 // (op 669) 
s_lshl_b32 s19, 1, s8 // (op 669) 
s_add_i32 s64, s19, s18 // (op 669) 


s_and_b32 s18, s0, s64 // (op 669) 


s_add_i32 s0, s9, s18 // (op 669) 


s_ashr_i32 s9, s0, s8 // (op 669) 


s_lshl_b32 s0, s48, 24 // (op 669) 
s_ashr_i32 s8, s0, 31 // (op 669) 

s_xor_b32 s0, s9, s8 // (op 669) 

s_sub_i32 s9, s0, s8 // (op 669) 


s_cmp_eq_i32 s1, 0 // (op 669) 
s_mov_b32 s0, scc // (op 669) 

s_cmp_eq_i32 s0, 0 // (op 669) 
s_cselect_b32 s1, s9, s51 // (op 669) 



v_lshl_add_u32 v20, s1, 5, v13 // (op 669) 


v_mul_lo_u32 v13, v12, s32 // (op 669) 
v_add_u32 v13, v13, v20 // (op 669) 


s_mul_i32 s1, s32, s37 // (op 671) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 671) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 671) least significant
s_mul_hi_u32 s8, s32, s36 // (op 671) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 671) most significant: sum
s_add_u32 s1, s1, s8 // (op 671) most significant: sum


s_mov_b32 s64, s34 // (op 671) reg expression
s_mov_b32 s65, s35 // (op 671) reg expression
s_mov_b32 s66, s0 // (op 671) reg expression
s_mov_b32 s67, 131072 // (op 671) call()

s_mul_hi_i32 s51, s16, s11 // (op 688) 
v_and_b32 v12, -32, v1 // (op 688) 
s_add_i32 s0, s51, s16 // (op 688) 

v_sub_u32 v20, v1, v12 // (op 688) 

s_ashr_i32 s1, s0, 31 // (op 688) 
v_lshrrev_b32 v12, 5, v1 // (op 688) 
s_cmp_eq_i32 s11, 0 // (op 688) 
s_mov_b32 s51, scc // (op 688) 
s_cmp_eq_i32 s51, 0 // (op 688) 
s_cselect_b32 s8, 0, -1 // (op 688) 

s_and_b32 s9, s12, 31 // (op 688) 
s_lshl_b32 s51, 1, s9 // (op 688) 
s_add_i32 s18, s51, s8 // (op 688) 


s_and_b32 s8, s1, s18 // (op 688) 


s_add_i32 s1, s0, s8 // (op 688) 


s_ashr_i32 s0, s1, s9 // (op 688) 


s_lshl_b32 s1, s12, 24 // (op 688) 
s_ashr_i32 s51, s1, 31 // (op 688) 

s_xor_b32 s1, s0, s51 // (op 688) 

s_sub_i32 s0, s1, s51 // (op 688) 


s_mul_hi_i32 s1, s0, s13 // (op 688) 
s_add_i32 s51, s1, s0 // (op 688) 

s_ashr_i32 s1, s51, 31 // (op 688) 
s_cmp_eq_i32 s13, 0 // (op 688) 
s_mov_b32 s8, scc // (op 688) 
s_cmp_eq_i32 s8, 0 // (op 688) 
s_cselect_b32 s9, 0, -1 // (op 688) 

s_and_b32 s8, s14, 31 // (op 688) 
s_lshl_b32 s18, 1, s8 // (op 688) 
s_add_i32 s19, s18, s9 // (op 688) 


s_and_b32 s9, s1, s19 // (op 688) 


s_add_i32 s1, s51, s9 // (op 688) 


s_ashr_i32 s9, s1, s8 // (op 688) 


s_lshl_b32 s1, s14, 24 // (op 688) 
s_ashr_i32 s8, s1, 31 // (op 688) 

s_xor_b32 s1, s9, s8 // (op 688) 

s_sub_i32 s9, s1, s8 // (op 688) 


s_add_u32 s18, s2, 32 // (op 688) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 688) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 688) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 688) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 688) 

s_mul_i32 s1, s9, s18 // (op 688) 

s_sub_i32 s51, s0, s1 // (op 688) 


s_mul_hi_i32 s0, s16, s10 // (op 688) 
s_add_i32 s1, s0, s16 // (op 688) 

s_ashr_i32 s0, s1, 31 // (op 688) 
s_cmp_eq_i32 s10, 0 // (op 688) 
s_mov_b32 s8, scc // (op 688) 
s_cmp_eq_i32 s8, 0 // (op 688) 
s_cselect_b32 s9, 0, -1 // (op 688) 

s_and_b32 s8, s49, 31 // (op 688) 
s_lshl_b32 s68, 1, s8 // (op 688) 
s_add_i32 s69, s68, s9 // (op 688) 


s_and_b32 s9, s0, s69 // (op 688) 


s_add_i32 s0, s1, s9 // (op 688) 


s_ashr_i32 s1, s0, s8 // (op 688) 


s_lshl_b32 s0, s49, 24 // (op 688) 
s_ashr_i32 s8, s0, 31 // (op 688) 

s_xor_b32 s0, s1, s8 // (op 688) 

s_sub_i32 s1, s0, s8 // (op 688) 


s_mul_i32 s0, s46, s18 // (op 688) 

s_mul_i32 s8, s50, s0 // (op 688) 

s_mul_i32 s0, s1, s8 // (op 688) 

s_sub_u32 s8, s16, s0 // (op 688) 

s_mul_hi_i32 s0, s8, s15 // (op 688) 
s_add_i32 s9, s0, s8 // (op 688) 


s_ashr_i32 s0, s9, 31 // (op 688) 
s_cmp_eq_i32 s15, 0 // (op 688) 
s_mov_b32 s8, scc // (op 688) 
s_cmp_eq_i32 s8, 0 // (op 688) 
s_cselect_b32 s18, 0, -1 // (op 688) 

s_and_b32 s8, s48, 31 // (op 688) 
s_lshl_b32 s19, 1, s8 // (op 688) 
s_add_i32 s68, s19, s18 // (op 688) 


s_and_b32 s18, s0, s68 // (op 688) 


s_add_i32 s0, s9, s18 // (op 688) 


s_ashr_i32 s9, s0, s8 // (op 688) 


s_lshl_b32 s0, s48, 24 // (op 688) 
s_ashr_i32 s8, s0, 31 // (op 688) 

s_xor_b32 s0, s9, s8 // (op 688) 

s_sub_i32 s9, s0, s8 // (op 688) 


s_cmp_eq_i32 s1, 0 // (op 688) 
s_mov_b32 s0, scc // (op 688) 

s_cmp_eq_i32 s0, 0 // (op 688) 
s_cselect_b32 s1, s9, s51 // (op 688) 



v_lshl_add_u32 v21, s1, 5, v20 // (op 688) 


v_mul_lo_u32 v20, v12, s32 // (op 688) 
v_add_u32 v20, v20, v21 // (op 688) 


s_mul_i32 s1, s32, s37 // (op 690) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 690) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 690) least significant
s_mul_hi_u32 s8, s32, s36 // (op 690) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 690) most significant: sum
s_add_u32 s1, s1, s8 // (op 690) most significant: sum


s_mov_b32 s64, s34 // (op 690) reg expression
s_mov_b32 s65, s35 // (op 690) reg expression
s_mov_b32 s66, s0 // (op 690) reg expression
s_mov_b32 s67, 131072 // (op 690) call()




buffer_load_ubyte v12, v18, s[60:63], 0 offen offset: 0 // (op 337) Load value
v_and_b32 v21, -2, v1 // (op 807) 
v_sub_u32 v22, v1, v21 // (op 807) 

v_lshlrev_b32 v21, 4, v22 // (op 807) 

v_lshrrev_b32 v22, 1, v1 // (op 807) 
v_lshl_add_u32 v23, v22, 5, v21 // (op 807) 


s_nop 0
v_readfirstlane_b32 s51, v23 // (op 807) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[22:23], 6 // (op 433) 
v_mov_b64 v[22:23], s[0:1] // (op 433) convert


v_mov_b64 v[24:25], v[14:15] // (op 433) reg expression
s_mov_b32 m0, s51 // (op 433) reg expression
buffer_load_dwordx4 v24, s[56:59], 0 offen offset: 0 lds // (op 433) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 433) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 433) most significant half

s_add_u32 m0, m0, 2048 // (op 433) 
buffer_load_dwordx4 v24, s[56:59], 0 offen offset: 0 lds // (op 433) Load value direct to lds





v_mov_b32 v21, v13 // (op 344) reg expression

buffer_load_ubyte v22, v21, s[64:67], 0 offen offset: 0 // (op 344) Load value




buffer_load_dwordx3 v[24:26], v10, s[52:55], 0 offen // (op 347) Load value



buffer_load_ubyte v21, v18, s[60:63], 0 offen offset:4 // (op 351) Load value

v_and_b32 v18, -2, v1 // (op 957) 
v_sub_u32 v23, v1, v18 // (op 957) 

v_lshlrev_b32 v18, 4, v23 // (op 957) 

v_lshrrev_b32 v23, 1, v1 // (op 957) 
v_lshl_add_u32 v27, v23, 5, v18 // (op 957) 


s_nop 0
v_readfirstlane_b32 s51, v27 // (op 957) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[22:23], 6 // (op 438) 
v_mov_b64 v[28:29], s[0:1] // (op 438) convert


s_lshl_b64 s[0:1], s[22:23], 7 // (op 438) 
v_mov_b32 v18, s0 // (op 438) 
v_mov_b32 v23, s1 // (op 438) 
v_add_co_u32 v30, s[8:9], v14, v18 // (op 438) least significant half
v_addc_co_u32 v31, s[8:9], v15, v23, s[8:9] // (op 438) most significant half




s_add_u32 s0, 4096, s51 // (op 438) 
s_mov_b32 m0, s0 // (op 438) reg expression
buffer_load_dwordx4 v30, s[56:59], 0 offen offset: 0 lds // (op 438) Load value direct to lds
v_add_co_u32 v30, s[8:9], v30, v28 // (op 438) least significant half
v_addc_co_u32 v31, s[8:9], 0, v31, s[8:9] // (op 438) most significant half

s_add_u32 m0, m0, 2048 // (op 438) 
buffer_load_dwordx4 v30, s[56:59], 0 offen offset: 0 lds // (op 438) Load value direct to lds







s_lshl_b32 s51, s32, 2 // (op 358) 
v_add_u32 v18, v13, s51 // (op 358) 


buffer_load_ubyte v23, v18, s[64:67], 0 offen offset: 0 // (op 358) Load value





buffer_load_dwordx3 v[28:30], v10, s[52:55], 0 offen offset:96 // (op 361) Load value

v_and_b32 v10, -32, v1 // (op 860) 
v_sub_u32 v18, v1, v10 // (op 860) 

v_and_b32 v10, 4294967264, v1 // (op 860) 
v_add_u32 v27, v10, v18 // (op 860) 





s_waitcnt vmcnt(6)
ds_write_b8 v27, v22 offset:8192 // (op 364) Store local data 

v_and_b32 v10, -8, v1 // (op 755) 
v_sub_u32 v18, v1, v10 // (op 755) 

v_lshlrev_b32 v10, 4, v18 // (op 755) 

v_lshrrev_b32 v18, 3, v1 // (op 755) 
v_lshl_add_u32 v27, v18, 7, v10 // (op 755) 


v_mul_lo_u32 v10, v27, 6 // (op 755) 

v_lshrrev_b32 v18, 3, v10 // (op 755) 




s_waitcnt vmcnt(5)
ds_write_b96 v18, v[24:26] offset:8320 // (op 365) Store local data 

v_and_b32 v10, -8, v1 // (op 726) 
v_sub_u32 v18, v1, v10 // (op 726) 

v_lshlrev_b32 v10, 4, v18 // (op 726) 

v_lshrrev_b32 v18, 3, v1 // (op 726) 
v_lshl_add_u32 v27, v18, 7, v10 // (op 726) 


v_mul_lo_u32 v10, v27, 6 // (op 726) 

v_lshrrev_b32 v18, 3, v10 // (op 726) 

v_and_b32 v10, -2, v1 // (op 779) 
v_sub_u32 v27, v1, v10 // (op 779) 

v_lshlrev_b32 v10, 4, v27 // (op 779) 

v_lshrrev_b32 v27, 1, v1 // (op 779) 
v_lshl_add_u32 v13, v27, 5, v10 // (op 779) 


s_nop 0
v_readfirstlane_b32 s51, v13 // (op 779) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -32, v1 // (op 831) 
v_sub_u32 v27, v1, v10 // (op 831) 

v_and_b32 v10, 4294967264, v1 // (op 831) 
v_add_u32 v13, v10, v27 // (op 831) 


v_and_b32 v10, -8, v1 // (op 918) 
v_sub_u32 v27, v1, v10 // (op 918) 

v_lshlrev_b32 v10, 4, v27 // (op 918) 

v_lshrrev_b32 v27, 3, v1 // (op 918) 
v_lshl_add_u32 v14, v27, 7, v10 // (op 918) 


v_mul_lo_u32 v10, v14, 6 // (op 918) 

v_lshrrev_b32 v27, 3, v10 // (op 918) 

v_and_b32 v10, -2, v1 // (op 942) 
v_sub_u32 v14, v1, v10 // (op 942) 

v_lshlrev_b32 v10, 4, v14 // (op 942) 

v_lshrrev_b32 v14, 1, v1 // (op 942) 
v_lshl_add_u32 v15, v14, 5, v10 // (op 942) 


s_nop 0
v_readfirstlane_b32 s0, v15 // (op 942) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -32, v1 // (op 981) 
v_sub_u32 v14, v1, v10 // (op 981) 

v_and_b32 v10, 4294967264, v1 // (op 981) 
v_add_u32 v15, v10, v14 // (op 981) 


s_mov_b32 s1, 0 // (op 55) call()
s_ashr_i64 s[8:9], s[26:27], 7 // (op 54) 
s_mov_b32 s18, s8 // (op 54) convert

s_and_b32 s19, s18, 4294967294 // (op 54) 

s_cmp_lt_u32 s1, s19 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54 if false)
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54:
s_add_u32 s1, s1, 2 // (op 56) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1026) 



ds_read_b128 v[32:35], v3 offset:8320 // (op 65) Load local data 
ds_read_b64 v[36:37], v3 offset:8336 // (op 65) Load local data 
s_barrier  // (op 1039) 



ds_read_b64_tr_b8 v[38:39], v4 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:256 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:2048 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:2304 // (op 70) Transpose load local data 
s_barrier  // (op 1052) 


ds_read_u8 v10, v0 offset:8192 // (op 79) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[38:45], a[0:3], v12, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 82) 





v_mov_b32 v10, v19 // (op 75) 

buffer_load_ubyte v12, v10, s[60:63], 0 offen offset:8 // (op 75) Load value
s_barrier  // (op 1042) 

s_lshl_b64 s[8:9], s[22:23], 6 // (op 416) 
v_mov_b64 v[32:33], s[8:9] // (op 416) convert


v_mov_b64 v[34:35], v[16:17] // (op 416) 
s_lshl_b64 s[8:9], s[22:23], 7 // (op 416) 
s_lshl_b64 s[18:19], s[8:9], 1 // (op 416) 

v_mov_b32 v14, s18 // (op 416) 
v_mov_b32 v31, s19 // (op 416) 
v_add_co_u32 v36, s[8:9], v34, v14 // (op 416) least significant half
v_addc_co_u32 v37, s[8:9], v35, v31, s[8:9] // (op 416) most significant half




s_mov_b32 m0, s51 // (op 416) reg expression
buffer_load_dwordx4 v36, s[56:59], 0 offen offset: 0 lds // (op 416) Load value direct to lds
v_add_co_u32 v36, s[8:9], v36, v32 // (op 416) least significant half
v_addc_co_u32 v37, s[8:9], 0, v37, s[8:9] // (op 416) most significant half

s_add_u32 m0, m0, 2048 // (op 416) 
buffer_load_dwordx4 v36, s[56:59], 0 offen offset: 0 lds // (op 416) Load value direct to lds




v_mov_b32 v14, v20 // (op 21) 
s_lshl_b32 s8, s32, 3 // (op 21) 
v_add_u32 v31, v14, s8 // (op 21) 


buffer_load_ubyte v22, v31, s[64:67], 0 offen offset: 0 // (op 21) Load value



v_mov_b32 v31, v11 // (op 4) 

buffer_load_dwordx3 v[24:26], v31, s[52:55], 0 offen offset:192 // (op 4) Load value
s_barrier  // (op 1081) 



s_waitcnt vmcnt(6)
ds_write_b8 v15, v23 offset:9856 // (op 236) Store local data 
s_barrier  // (op 1069) 



s_waitcnt vmcnt(5)
ds_write_b96 v27, v[28:30] offset:9984 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1066) 



ds_read_b128 v[36:39], v6 offset:9984 // (op 227) Load local data 
ds_read_b64 v[40:41], v6 offset:10000 // (op 227) Load local data 
s_barrier  // (op 1072) 
s_barrier  // (op 1128) 



ds_read_b64_tr_b8 v[42:43], v7 offset:4096 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:4352 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:6144 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:6400 // (op 229) Transpose load local data 
s_barrier  // (op 1078) 


ds_read_u8 v32, v5 offset:9856 // (op 233) Load local data 
s_barrier  // (op 1138) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:49], a[0:3], v21, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 234) 






buffer_load_ubyte v21, v10, s[60:63], 0 offen offset:12 // (op 231) Load value

s_barrier  // (op 1075) 
v_add_u32 v19, 8, v19 // (op 616) 

s_lshl_b64 s[8:9], s[22:23], 6 // (op 428) 
v_mov_b64 v[32:33], s[8:9] // (op 428) convert


s_lshl_b64 s[8:9], s[22:23], 7 // (op 428) 
s_mul_i32 s19, 3, s9 // (op 428) most significant: low of low * high
s_mul_i32 s18, 3, s8 // (op 428) least significant
s_mul_hi_u32 s68, 3, s8 // (op 428) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 428) most significant: sum


v_mov_b32 v10, s18 // (op 428) 
v_mov_b32 v36, s19 // (op 428) 
v_add_co_u32 v38, s[8:9], v34, v10 // (op 428) least significant half
v_addc_co_u32 v39, s[8:9], v35, v36, s[8:9] // (op 428) most significant half




s_add_u32 s8, 4096, s0 // (op 428) 
s_nop 0
s_mov_b32 m0, s8 // (op 428) reg expression
buffer_load_dwordx4 v38, s[56:59], 0 offen offset: 0 lds // (op 428) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v32 // (op 428) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 428) most significant half

s_add_u32 m0, m0, 2048 // (op 428) 
buffer_load_dwordx4 v38, s[56:59], 0 offen offset: 0 lds // (op 428) Load value direct to lds






s_lshl_b32 s8, s32, 2 // (op 238) 
s_mul_i32 s9, 3, s8 // (op 238) 

v_add_u32 v10, v14, s9 // (op 238) 


s_lshl_b64 s[8:9], s[22:23], 7 // (op 552) 
s_lshl_b64 s[18:19], s[8:9], 1 // (op 552) 

v_mov_b32 v32, s18 // (op 552) 
v_mov_b32 v33, s19 // (op 552) 
v_add_co_u32 v16, s[8:9], v16, v32 // (op 552) least significant half
v_addc_co_u32 v17, s[8:9], v17, v33, s[8:9] // (op 552) most significant half




buffer_load_ubyte v23, v10, s[64:67], 0 offen offset: 0 // (op 238) Load value





s_lshl_b32 s8, s32, 3 // (op 680) 
v_add_u32 v20, v20, s8 // (op 680) 

buffer_load_dwordx3 v[28:30], v31, s[52:55], 0 offen offset:288 // (op 224) Load value

s_barrier  // (op 1055) 
v_add_u32 v11, 192, v11 // (op 488) 



s_waitcnt vmcnt(6)
ds_write_b8 v13, v22 offset:8192 // (op 26) Store local data 
s_barrier  // (op 1029) 



s_waitcnt vmcnt(5)
ds_write_b96 v18, v[24:26] offset:8320 // (op 6) Store local data 
s_ashr_i64 s[8:9], s[26:27], 7 // (op 54) 
s_mov_b32 s18, s8 // (op 54) convert

s_and_b32 s19, s18, 4294967294 // (op 54) 

s_cmp_lt_u32 s1, s19 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54 if true)
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54:














s_barrier  // (op 1092) 
s_mul_hi_i32 s51, s16, s11 // (op 451) 
v_lshrrev_b32 v22, 3, v1 // (op 451) 
s_add_i32 s0, s51, s16 // (op 451) 

v_and_b32 v10, -8, v1 // (op 451) 
s_ashr_i32 s1, s0, 31 // (op 451) 
v_sub_u32 v11, v1, v10 // (op 451) 

s_cmp_eq_i32 s11, 0 // (op 451) 
s_mov_b32 s51, scc // (op 451) 
v_lshlrev_b32 v10, 4, v11 // (op 451) 

s_cmp_eq_i32 s51, 0 // (op 451) 
s_cselect_b32 s8, 0, -1 // (op 451) 

s_and_b32 s9, s12, 31 // (op 451) 
s_lshl_b32 s51, 1, s9 // (op 451) 
s_add_i32 s18, s51, s8 // (op 451) 


s_and_b32 s8, s1, s18 // (op 451) 


s_add_i32 s1, s0, s8 // (op 451) 


s_ashr_i32 s0, s1, s9 // (op 451) 


s_lshl_b32 s1, s12, 24 // (op 451) 
s_ashr_i32 s51, s1, 31 // (op 451) 

s_xor_b32 s1, s0, s51 // (op 451) 

s_sub_i32 s0, s1, s51 // (op 451) 


s_mul_hi_i32 s1, s0, s13 // (op 451) 
s_add_i32 s51, s1, s0 // (op 451) 

s_ashr_i32 s1, s51, 31 // (op 451) 
s_cmp_eq_i32 s13, 0 // (op 451) 
s_mov_b32 s8, scc // (op 451) 
s_cmp_eq_i32 s8, 0 // (op 451) 
s_cselect_b32 s9, 0, -1 // (op 451) 

s_and_b32 s8, s14, 31 // (op 451) 
s_lshl_b32 s18, 1, s8 // (op 451) 
s_add_i32 s19, s18, s9 // (op 451) 


s_and_b32 s9, s1, s19 // (op 451) 


s_add_i32 s1, s51, s9 // (op 451) 


s_ashr_i32 s9, s1, s8 // (op 451) 


s_lshl_b32 s1, s14, 24 // (op 451) 
s_ashr_i32 s8, s1, 31 // (op 451) 

s_xor_b32 s1, s9, s8 // (op 451) 

s_sub_i32 s9, s1, s8 // (op 451) 


s_mul_i32 s1, s0, s46 // (op 451) 

s_sub_u32 s0, s16, s1 // (op 451) 

s_mul_i32 s1, s9, s46 // (op 451) 
s_add_u32 s1, s1, s0 // (op 451) 


s_mul_hi_i32 s0, s16, s10 // (op 451) 
s_add_i32 s51, s0, s16 // (op 451) 

s_ashr_i32 s0, s51, 31 // (op 451) 
s_cmp_eq_i32 s10, 0 // (op 451) 
s_mov_b32 s8, scc // (op 451) 
s_cmp_eq_i32 s8, 0 // (op 451) 
s_cselect_b32 s9, 0, -1 // (op 451) 

s_and_b32 s8, s49, 31 // (op 451) 
s_lshl_b32 s18, 1, s8 // (op 451) 
s_add_i32 s19, s18, s9 // (op 451) 


s_and_b32 s9, s0, s19 // (op 451) 


s_add_i32 s0, s51, s9 // (op 451) 


s_ashr_i32 s9, s0, s8 // (op 451) 


s_lshl_b32 s0, s49, 24 // (op 451) 
s_ashr_i32 s8, s0, 31 // (op 451) 

s_xor_b32 s0, s9, s8 // (op 451) 

s_sub_i32 s9, s0, s8 // (op 451) 


s_add_u32 s18, s2, 32 // (op 451) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 451) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 451) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 451) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 451) 

s_mul_i32 s0, s46, s18 // (op 451) 

s_mul_i32 s8, s50, s0 // (op 451) 

s_mul_i32 s0, s9, s8 // (op 451) 

s_sub_u32 s8, s16, s0 // (op 451) 

s_mov_b32 s0, s8 // (op 451) 
s_mul_hi_i32 s51, s0, s15 // (op 451) 
s_add_i32 s18, s51, s0 // (op 451) 


s_ashr_i32 s0, s18, 31 // (op 451) 
s_cmp_eq_i32 s15, 0 // (op 451) 
s_mov_b32 s19, scc // (op 451) 
s_cmp_eq_i32 s19, 0 // (op 451) 
s_cselect_b32 s51, 0, -1 // (op 451) 

s_and_b32 s19, s48, 31 // (op 451) 
s_lshl_b32 s68, 1, s19 // (op 451) 
s_add_i32 s69, s68, s51 // (op 451) 


s_and_b32 s51, s0, s69 // (op 451) 


s_add_i32 s0, s18, s51 // (op 451) 


s_ashr_i32 s18, s0, s19 // (op 451) 


s_lshl_b32 s0, s48, 24 // (op 451) 
s_ashr_i32 s19, s0, 31 // (op 451) 

s_xor_b32 s0, s18, s19 // (op 451) 

s_sub_i32 s18, s0, s19 // (op 451) 


s_mul_i32 s0, s47, s46 // (op 451) 
s_add_u32 s68, s6, 16 // (op 451) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 451) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 451) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 451) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 451) 

s_sub_i32 s19, s68, s0 // (op 451) 

s_mul_i32 s51, s18, s19 // (op 451) 


s_sub_u32 s18, s8, s51 // (op 451) 


s_add_u32 s8, s18, s0 // (op 451) 


s_cmp_eq_i32 s9, 0 // (op 451) 
s_mov_b32 s0, scc // (op 451) 

s_cmp_eq_i32 s0, 0 // (op 451) 
s_cselect_b32 s9, s8, s1 // (op 451) 



v_lshl_add_u32 v11, s9, 4, v22 // (op 451) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 451) 
s_lshr_b32 s51, s0, 1 // (op 451) 

v_lshl_add_u32 v22, s51, 8, v10 // (op 451) 


v_mov_b32 v13, 0 // (op 451) convert
v_mov_b32 v12, v22 // (op 451) convert

v_mul_lo_u32 v15, v11, s5 // (op 451) most significant: low of low * high
v_mul_hi_u32 v10, v11, s4 // (op 451) most significant: high of low * low
v_mul_lo_u32 v14, v11, s4 // (op 451) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 451) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 451) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 451) most significant half



v_mul_lo_u32 v11, v15, 6 // (op 451) most significant: low of high * low
v_mul_hi_u32 v22, v14, 6 // (op 451) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 451) least significant: low of low * low
v_add_u32 v11, v11, v22 // (op 451) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 451) 

v_mov_b32 v22, v12 // (op 451) convert

s_mul_i32 s1, s7, 6 // (op 453) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 453) least significant
s_mul_hi_u32 s51, s6, 6 // (op 453) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 453) most significant: sum

s_mul_i32 s9, s4, s1 // (op 453) most significant: low of low * high
s_mul_i32 s51, s5, s0 // (op 453) most significant: low of high * low
s_mul_i32 s8, s4, s0 // (op 453) least significant
s_mul_hi_u32 s18, s4, s0 // (op 453) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 453) most significant: sum
s_add_u32 s9, s9, s18 // (op 453) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 453) 

s_mov_b32 s52, s20 // (op 453) reg expression
s_mov_b32 s53, s21 // (op 453) reg expression
s_mov_b32 s54, s0 // (op 453) reg expression
s_mov_b32 s55, 131072 // (op 453) call()

s_mul_hi_i32 s51, s16, s11 // (op 518) 
v_and_b32 v10, -2, v1 // (op 518) 
s_add_i32 s0, s51, s16 // (op 518) 

v_sub_u32 v11, v1, v10 // (op 518) 

s_ashr_i32 s1, s0, 31 // (op 518) 
v_lshlrev_b32 v10, 4, v11 // (op 518) 

s_cmp_eq_i32 s11, 0 // (op 518) 
s_mov_b32 s51, scc // (op 518) 
v_lshrrev_b32 v11, 1, v1 // (op 518) 
s_cmp_eq_i32 s51, 0 // (op 518) 
s_cselect_b32 s8, 0, -1 // (op 518) 

s_and_b32 s9, s12, 31 // (op 518) 
s_lshl_b32 s51, 1, s9 // (op 518) 
s_add_i32 s18, s51, s8 // (op 518) 


s_and_b32 s8, s1, s18 // (op 518) 


s_add_i32 s1, s0, s8 // (op 518) 


s_ashr_i32 s0, s1, s9 // (op 518) 


s_lshl_b32 s1, s12, 24 // (op 518) 
s_ashr_i32 s51, s1, 31 // (op 518) 

s_xor_b32 s1, s0, s51 // (op 518) 

s_sub_i32 s0, s1, s51 // (op 518) 


s_mul_hi_i32 s1, s0, s13 // (op 518) 
s_add_i32 s51, s1, s0 // (op 518) 

s_ashr_i32 s1, s51, 31 // (op 518) 
s_cmp_eq_i32 s13, 0 // (op 518) 
s_mov_b32 s8, scc // (op 518) 
s_cmp_eq_i32 s8, 0 // (op 518) 
s_cselect_b32 s9, 0, -1 // (op 518) 

s_and_b32 s8, s14, 31 // (op 518) 
s_lshl_b32 s18, 1, s8 // (op 518) 
s_add_i32 s19, s18, s9 // (op 518) 


s_and_b32 s9, s1, s19 // (op 518) 


s_add_i32 s1, s51, s9 // (op 518) 


s_ashr_i32 s9, s1, s8 // (op 518) 


s_lshl_b32 s1, s14, 24 // (op 518) 
s_ashr_i32 s8, s1, 31 // (op 518) 

s_xor_b32 s1, s9, s8 // (op 518) 

s_sub_i32 s9, s1, s8 // (op 518) 


s_add_u32 s18, s2, 32 // (op 518) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 518) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 518) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 518) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 518) 

s_mul_i32 s1, s9, s18 // (op 518) 

s_sub_i32 s51, s0, s1 // (op 518) 


s_mul_hi_i32 s0, s16, s10 // (op 518) 
s_add_i32 s1, s0, s16 // (op 518) 

s_ashr_i32 s0, s1, 31 // (op 518) 
s_cmp_eq_i32 s10, 0 // (op 518) 
s_mov_b32 s8, scc // (op 518) 
s_cmp_eq_i32 s8, 0 // (op 518) 
s_cselect_b32 s9, 0, -1 // (op 518) 

s_and_b32 s8, s49, 31 // (op 518) 
s_lshl_b32 s68, 1, s8 // (op 518) 
s_add_i32 s69, s68, s9 // (op 518) 


s_and_b32 s9, s0, s69 // (op 518) 


s_add_i32 s0, s1, s9 // (op 518) 


s_ashr_i32 s1, s0, s8 // (op 518) 


s_lshl_b32 s0, s49, 24 // (op 518) 
s_ashr_i32 s8, s0, 31 // (op 518) 

s_xor_b32 s0, s1, s8 // (op 518) 

s_sub_i32 s1, s0, s8 // (op 518) 


s_mul_i32 s0, s46, s18 // (op 518) 

s_mul_i32 s8, s50, s0 // (op 518) 

s_mul_i32 s0, s1, s8 // (op 518) 

s_sub_u32 s8, s16, s0 // (op 518) 

s_mul_hi_i32 s0, s8, s15 // (op 518) 
s_add_i32 s9, s0, s8 // (op 518) 


s_ashr_i32 s0, s9, 31 // (op 518) 
s_cmp_eq_i32 s15, 0 // (op 518) 
s_mov_b32 s8, scc // (op 518) 
s_cmp_eq_i32 s8, 0 // (op 518) 
s_cselect_b32 s18, 0, -1 // (op 518) 

s_and_b32 s8, s48, 31 // (op 518) 
s_lshl_b32 s19, 1, s8 // (op 518) 
s_add_i32 s68, s19, s18 // (op 518) 


s_and_b32 s18, s0, s68 // (op 518) 


s_add_i32 s0, s9, s18 // (op 518) 


s_ashr_i32 s9, s0, s8 // (op 518) 


s_lshl_b32 s0, s48, 24 // (op 518) 
s_ashr_i32 s8, s0, 31 // (op 518) 

s_xor_b32 s0, s9, s8 // (op 518) 

s_sub_i32 s9, s0, s8 // (op 518) 


s_cmp_eq_i32 s1, 0 // (op 518) 
s_mov_b32 s0, scc // (op 518) 

s_cmp_eq_i32 s0, 0 // (op 518) 
s_cselect_b32 s1, s9, s51 // (op 518) 



v_lshl_add_u32 v12, s1, 5, v10 // (op 518) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 518) 
v_mov_b32 v25, 0 // (op 518) convert
v_mov_b32 v24, v12 // (op 518) convert

s_lshr_b32 s51, s0, 1 // (op 518) 

v_lshl_add_u32 v10, s51, 8, v11 // (op 518) 


v_mul_lo_u32 v27, v10, s23 // (op 518) most significant: low of low * high
v_mul_hi_u32 v11, v10, s22 // (op 518) most significant: high of low * low
v_mul_lo_u32 v26, v10, s22 // (op 518) least significant: low of low * low
v_add_u32 v27, v27, v11 // (op 518) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v24 // (op 518) least significant half
v_addc_co_u32 v27, s[0:1], v27, v25, s[0:1] // (op 518) most significant half



s_mul_i32 s1, s22, s27 // (op 520) most significant: low of low * high
s_mul_i32 s51, s23, s26 // (op 520) most significant: low of high * low
s_mul_i32 s0, s22, s26 // (op 520) least significant
s_mul_hi_u32 s8, s22, s26 // (op 520) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 520) most significant: sum
s_add_u32 s1, s1, s8 // (op 520) most significant: sum


s_mov_b32 s56, s24 // (op 520) reg expression
s_mov_b32 s57, s25 // (op 520) reg expression
s_mov_b32 s58, s0 // (op 520) reg expression
s_mov_b32 s59, 131072 // (op 520) call()

s_mul_hi_i32 s51, s16, s11 // (op 582) 
v_lshrrev_b32 v10, 7, v1 // (op 582) 
s_add_i32 s0, s51, s16 // (op 582) 

v_and_b32 v11, 15, v1 // (op 582) 
s_ashr_i32 s1, s0, 31 // (op 582) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 582) 


s_cmp_eq_i32 s11, 0 // (op 582) 
s_mov_b32 s51, scc // (op 582) 
v_and_b32 v10, 63, v1 // (op 582) 
s_cmp_eq_i32 s51, 0 // (op 582) 
s_cselect_b32 s8, 0, -1 // (op 582) 

v_lshrrev_b32 v11, 4, v10 // (op 582) 

s_and_b32 s9, s12, 31 // (op 582) 
s_lshl_b32 s51, 1, s9 // (op 582) 
s_add_i32 s18, s51, s8 // (op 582) 


s_and_b32 s8, s1, s18 // (op 582) 


s_add_i32 s1, s0, s8 // (op 582) 


s_ashr_i32 s0, s1, s9 // (op 582) 


s_lshl_b32 s1, s12, 24 // (op 582) 
s_ashr_i32 s51, s1, 31 // (op 582) 

s_xor_b32 s1, s0, s51 // (op 582) 

s_sub_i32 s0, s1, s51 // (op 582) 


s_mul_hi_i32 s1, s0, s13 // (op 582) 
s_add_i32 s51, s1, s0 // (op 582) 

s_ashr_i32 s1, s51, 31 // (op 582) 
s_cmp_eq_i32 s13, 0 // (op 582) 
s_mov_b32 s8, scc // (op 582) 
s_cmp_eq_i32 s8, 0 // (op 582) 
s_cselect_b32 s9, 0, -1 // (op 582) 

s_and_b32 s8, s14, 31 // (op 582) 
s_lshl_b32 s18, 1, s8 // (op 582) 
s_add_i32 s19, s18, s9 // (op 582) 


s_and_b32 s9, s1, s19 // (op 582) 


s_add_i32 s1, s51, s9 // (op 582) 


s_ashr_i32 s9, s1, s8 // (op 582) 


s_lshl_b32 s1, s14, 24 // (op 582) 
s_ashr_i32 s8, s1, 31 // (op 582) 

s_xor_b32 s1, s9, s8 // (op 582) 

s_sub_i32 s9, s1, s8 // (op 582) 


s_mul_i32 s1, s0, s46 // (op 582) 

s_sub_u32 s0, s16, s1 // (op 582) 

s_mul_i32 s1, s9, s46 // (op 582) 
s_add_u32 s1, s1, s0 // (op 582) 


s_mul_hi_i32 s0, s16, s10 // (op 582) 
s_add_i32 s51, s0, s16 // (op 582) 

s_ashr_i32 s0, s51, 31 // (op 582) 
s_cmp_eq_i32 s10, 0 // (op 582) 
s_mov_b32 s8, scc // (op 582) 
s_cmp_eq_i32 s8, 0 // (op 582) 
s_cselect_b32 s9, 0, -1 // (op 582) 

s_and_b32 s8, s49, 31 // (op 582) 
s_lshl_b32 s18, 1, s8 // (op 582) 
s_add_i32 s19, s18, s9 // (op 582) 


s_and_b32 s9, s0, s19 // (op 582) 


s_add_i32 s0, s51, s9 // (op 582) 


s_ashr_i32 s9, s0, s8 // (op 582) 


s_lshl_b32 s0, s49, 24 // (op 582) 
s_ashr_i32 s8, s0, 31 // (op 582) 

s_xor_b32 s0, s9, s8 // (op 582) 

s_sub_i32 s9, s0, s8 // (op 582) 


s_add_u32 s18, s2, 32 // (op 582) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 582) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 582) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 582) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 582) 

s_mul_i32 s0, s46, s18 // (op 582) 

s_mul_i32 s8, s50, s0 // (op 582) 

s_mul_i32 s0, s9, s8 // (op 582) 

s_sub_u32 s8, s16, s0 // (op 582) 

s_mov_b32 s0, s8 // (op 582) 
s_mul_hi_i32 s51, s0, s15 // (op 582) 
s_add_i32 s18, s51, s0 // (op 582) 


s_ashr_i32 s0, s18, 31 // (op 582) 
s_cmp_eq_i32 s15, 0 // (op 582) 
s_mov_b32 s19, scc // (op 582) 
s_cmp_eq_i32 s19, 0 // (op 582) 
s_cselect_b32 s51, 0, -1 // (op 582) 

s_and_b32 s19, s48, 31 // (op 582) 
s_lshl_b32 s68, 1, s19 // (op 582) 
s_add_i32 s69, s68, s51 // (op 582) 


s_and_b32 s51, s0, s69 // (op 582) 


s_add_i32 s0, s18, s51 // (op 582) 


s_ashr_i32 s18, s0, s19 // (op 582) 


s_lshl_b32 s0, s48, 24 // (op 582) 
s_ashr_i32 s19, s0, 31 // (op 582) 

s_xor_b32 s0, s18, s19 // (op 582) 

s_sub_i32 s18, s0, s19 // (op 582) 


s_mul_i32 s0, s47, s46 // (op 582) 
s_add_u32 s68, s6, 16 // (op 582) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 582) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 582) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 582) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 582) 

s_sub_i32 s19, s68, s0 // (op 582) 

s_mul_i32 s51, s18, s19 // (op 582) 


s_sub_u32 s18, s8, s51 // (op 582) 


s_add_u32 s8, s18, s0 // (op 582) 


s_cmp_eq_i32 s9, 0 // (op 582) 
s_mov_b32 s0, scc // (op 582) 

s_cmp_eq_i32 s0, 0 // (op 582) 
s_cselect_b32 s9, s8, s1 // (op 582) 



v_lshl_add_u32 v10, s9, 4, v12 // (op 582) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 582) 
s_lshr_b32 s51, s0, 1 // (op 582) 

v_lshl_add_u32 v12, s51, 3, v11 // (op 582) 


v_mul_lo_u32 v11, v10, s28 // (op 582) 
v_add_u32 v11, v11, v12 // (op 582) 


s_mul_i32 s1, s28, s7 // (op 584) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 584) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 584) least significant
s_mul_hi_u32 s8, s28, s6 // (op 584) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 584) most significant: sum
s_add_u32 s1, s1, s8 // (op 584) most significant: sum


s_mov_b32 s60, s30 // (op 584) reg expression
s_mov_b32 s61, s31 // (op 584) reg expression
s_mov_b32 s62, s0 // (op 584) reg expression
s_mov_b32 s63, 131072 // (op 584) call()

s_mul_hi_i32 s51, s16, s11 // (op 646) 
v_and_b32 v10, -32, v1 // (op 646) 
s_add_i32 s0, s51, s16 // (op 646) 

v_sub_u32 v12, v1, v10 // (op 646) 

s_ashr_i32 s1, s0, 31 // (op 646) 
v_lshrrev_b32 v10, 5, v1 // (op 646) 
s_cmp_eq_i32 s11, 0 // (op 646) 
s_mov_b32 s51, scc // (op 646) 
s_cmp_eq_i32 s51, 0 // (op 646) 
s_cselect_b32 s8, 0, -1 // (op 646) 

s_and_b32 s9, s12, 31 // (op 646) 
s_lshl_b32 s51, 1, s9 // (op 646) 
s_add_i32 s18, s51, s8 // (op 646) 


s_and_b32 s8, s1, s18 // (op 646) 


s_add_i32 s1, s0, s8 // (op 646) 


s_ashr_i32 s0, s1, s9 // (op 646) 


s_lshl_b32 s1, s12, 24 // (op 646) 
s_ashr_i32 s51, s1, 31 // (op 646) 

s_xor_b32 s1, s0, s51 // (op 646) 

s_sub_i32 s0, s1, s51 // (op 646) 


s_mul_hi_i32 s1, s0, s13 // (op 646) 
s_add_i32 s51, s1, s0 // (op 646) 

s_ashr_i32 s1, s51, 31 // (op 646) 
s_cmp_eq_i32 s13, 0 // (op 646) 
s_mov_b32 s8, scc // (op 646) 
s_cmp_eq_i32 s8, 0 // (op 646) 
s_cselect_b32 s9, 0, -1 // (op 646) 

s_and_b32 s8, s14, 31 // (op 646) 
s_lshl_b32 s18, 1, s8 // (op 646) 
s_add_i32 s19, s18, s9 // (op 646) 


s_and_b32 s9, s1, s19 // (op 646) 


s_add_i32 s1, s51, s9 // (op 646) 


s_ashr_i32 s9, s1, s8 // (op 646) 


s_lshl_b32 s1, s14, 24 // (op 646) 
s_ashr_i32 s8, s1, 31 // (op 646) 

s_xor_b32 s1, s9, s8 // (op 646) 

s_sub_i32 s9, s1, s8 // (op 646) 


s_add_u32 s18, s2, 32 // (op 646) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 646) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 646) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 646) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 646) 

s_mul_i32 s1, s9, s18 // (op 646) 

s_sub_i32 s51, s0, s1 // (op 646) 


s_mul_hi_i32 s0, s16, s10 // (op 646) 
s_add_i32 s1, s0, s16 // (op 646) 

s_ashr_i32 s0, s1, 31 // (op 646) 
s_cmp_eq_i32 s10, 0 // (op 646) 
s_mov_b32 s8, scc // (op 646) 
s_cmp_eq_i32 s8, 0 // (op 646) 
s_cselect_b32 s9, 0, -1 // (op 646) 

s_and_b32 s8, s49, 31 // (op 646) 
s_lshl_b32 s68, 1, s8 // (op 646) 
s_add_i32 s69, s68, s9 // (op 646) 


s_and_b32 s9, s0, s69 // (op 646) 


s_add_i32 s0, s1, s9 // (op 646) 


s_ashr_i32 s1, s0, s8 // (op 646) 


s_lshl_b32 s0, s49, 24 // (op 646) 
s_ashr_i32 s8, s0, 31 // (op 646) 

s_xor_b32 s0, s1, s8 // (op 646) 

s_sub_i32 s1, s0, s8 // (op 646) 


s_mul_i32 s0, s46, s18 // (op 646) 

s_mul_i32 s8, s50, s0 // (op 646) 

s_mul_i32 s0, s1, s8 // (op 646) 

s_sub_u32 s8, s16, s0 // (op 646) 

s_mul_hi_i32 s0, s8, s15 // (op 646) 
s_add_i32 s9, s0, s8 // (op 646) 


s_ashr_i32 s0, s9, 31 // (op 646) 
s_cmp_eq_i32 s15, 0 // (op 646) 
s_mov_b32 s8, scc // (op 646) 
s_cmp_eq_i32 s8, 0 // (op 646) 
s_cselect_b32 s18, 0, -1 // (op 646) 

s_and_b32 s8, s48, 31 // (op 646) 
s_lshl_b32 s19, 1, s8 // (op 646) 
s_add_i32 s68, s19, s18 // (op 646) 


s_and_b32 s18, s0, s68 // (op 646) 


s_add_i32 s0, s9, s18 // (op 646) 


s_ashr_i32 s9, s0, s8 // (op 646) 


s_lshl_b32 s0, s48, 24 // (op 646) 
s_ashr_i32 s8, s0, 31 // (op 646) 

s_xor_b32 s0, s9, s8 // (op 646) 

s_sub_i32 s9, s0, s8 // (op 646) 


s_cmp_eq_i32 s1, 0 // (op 646) 
s_mov_b32 s0, scc // (op 646) 

s_cmp_eq_i32 s0, 0 // (op 646) 
s_cselect_b32 s1, s9, s51 // (op 646) 



v_lshl_add_u32 v13, s1, 5, v12 // (op 646) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 646) 
s_lshr_b32 s51, s0, 1 // (op 646) 

v_lshl_add_u32 v12, s51, 3, v10 // (op 646) 


v_mul_lo_u32 v10, v12, s32 // (op 646) 
v_add_u32 v10, v10, v13 // (op 646) 


s_mul_i32 s1, s32, s37 // (op 648) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 648) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 648) least significant
s_mul_hi_u32 s8, s32, s36 // (op 648) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 648) most significant: sum
s_add_u32 s1, s1, s8 // (op 648) most significant: sum


s_mov_b32 s64, s34 // (op 648) reg expression
s_mov_b32 s65, s35 // (op 648) reg expression
s_mov_b32 s66, s0 // (op 648) reg expression
s_mov_b32 s67, 131072 // (op 648) call()

v_and_b32 v12, -8, v1 // (op 739) 
v_sub_u32 v13, v1, v12 // (op 739) 

v_lshlrev_b32 v12, 4, v13 // (op 739) 

v_lshrrev_b32 v13, 3, v1 // (op 739) 
v_lshl_add_u32 v14, v13, 7, v12 // (op 739) 


v_mul_lo_u32 v12, v14, 6 // (op 739) 

v_lshrrev_b32 v13, 3, v12 // (op 739) 

v_and_b32 v12, -2, v1 // (op 792) 
v_sub_u32 v14, v1, v12 // (op 792) 

v_lshlrev_b32 v12, 4, v14 // (op 792) 

v_lshrrev_b32 v14, 1, v1 // (op 792) 
v_lshl_add_u32 v15, v14, 5, v12 // (op 792) 


s_nop 0
v_readfirstlane_b32 s51, v15 // (op 792) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -32, v1 // (op 844) 
v_sub_u32 v14, v1, v12 // (op 844) 

v_and_b32 v12, 4294967264, v1 // (op 844) 
v_add_u32 v15, v12, v14 // (op 844) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 162) 
s_and_b32 s8, s0, 4294967294 // (op 162) 

s_ashr_i64 s[0:1], s[26:27], 7 // (op 161) 
s_mov_b32 s9, s0 // (op 161) convert

s_cmp_lt_u32 s8, s9 // (op 161) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161:


s_add_u32 s8, s8, 1 // (op 163) 
v_mov_b32 v12, v22 // (op 166) 

buffer_load_dwordx3 v[16:18], v12, s[52:55], 0 offen // (op 166) Load value

v_add_u32 v22, 96, v22 // (op 441) 
s_barrier  // (op 1032) 
s_barrier  // (op 1045) 




s_lshl_b64 s[0:1], s[22:23], 6 // (op 421) 
v_mov_b64 v[20:21], s[0:1] // (op 421) convert


v_mov_b64 v[24:25], v[26:27] // (op 421) 
v_mov_b64 v[28:29], v[24:25] // (op 421) reg expression
s_mov_b32 m0, s51 // (op 421) reg expression
buffer_load_dwordx4 v28, s[56:59], 0 offen offset: 0 lds // (op 421) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v20 // (op 421) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 421) most significant half

s_add_u32 m0, m0, 2048 // (op 421) 
buffer_load_dwordx4 v28, s[56:59], 0 offen offset: 0 lds // (op 421) Load value direct to lds



s_lshl_b64 s[0:1], s[22:23], 7 // (op 510) 
v_mov_b32 v12, s0 // (op 510) 
v_mov_b32 v14, s1 // (op 510) 
v_add_co_u32 v26, s[18:19], v26, v12 // (op 510) least significant half
v_addc_co_u32 v27, s[18:19], v27, v14, s[18:19] // (op 510) most significant half




s_waitcnt vmcnt(2)
ds_write_b96 v13, v[16:18] offset:8320 // (op 167) Store local data 



v_mov_b32 v12, v10 // (op 180) 
v_mov_b32 v14, v12 // (op 180) reg expression

buffer_load_ubyte v16, v14, s[64:67], 0 offen offset: 0 // (op 180) Load value


s_lshl_b32 s9, s32, 2 // (op 638) 
v_add_u32 v10, v10, s9 // (op 638) 

s_barrier  // (op 1058) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1035) 






ds_read_b128 v[28:31], v3 offset:8320 // (op 169) Load local data 
ds_read_b64 v[32:33], v3 offset:8336 // (op 169) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v15, v16 offset:8192 // (op 178) Store local data 

s_barrier  // (op 1048) 



ds_read_b64_tr_b8 v[34:35], v4 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:256 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:2048 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:2304 // (op 171) Transpose load local data 


v_mov_b32 v12, v11 // (op 173) 

buffer_load_ubyte v14, v12, s[60:63], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v11, 4, v11 // (op 574) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1062) 


ds_read_u8 v12, v0 offset:8192 // (op 175) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:33], v[34:41], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 176) 




s_ashr_i64 s[0:1], s[26:27], 7 // (op 161) 
s_mov_b32 s9, s0 // (op 161) convert

s_cmp_lt_u32 s8, s9 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161 if true)
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161:






















s_barrier  // (op 1122) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

s_mul_hi_i32 s0, s16, s11 // (op 886) 
v_lshrrev_b32 v0, 7, v1 // (op 886) 
s_add_i32 s1, s0, s16 // (op 886) 

v_and_b32 v3, 63, v1 // (op 886) 
s_ashr_i32 s0, s1, 31 // (op 886) 
v_lshrrev_b32 v4, 4, v3 // (op 886) 
s_cmp_eq_i32 s11, 0 // (op 886) 
s_mov_b32 s4, scc // (op 886) 
v_lshlrev_b32 v5, 2, v4 // (op 886) 

s_cmp_eq_i32 s4, 0 // (op 886) 
s_cselect_b32 s5, 0, -1 // (op 886) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 886) 


s_and_b32 s4, s12, 31 // (op 886) 
v_lshrrev_b32 v0, 2, v3 // (op 886) 

s_lshl_b32 s8, 1, s4 // (op 886) 
v_and_b32 v3, 3, v0 // (op 886) 

s_add_i32 s9, s8, s5 // (op 886) 


v_and_b32 v0, 3, v1 // (op 886) 
s_and_b32 s5, s0, s9 // (op 886) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 886) 


s_add_i32 s0, s1, s5 // (op 886) 


v_lshrrev_b32 v0, 6, v1 // (op 886) 
s_ashr_i32 s1, s0, s4 // (op 886) 


v_and_b32 v3, 1, v0 // (op 886) 

s_lshl_b32 s0, s12, 24 // (op 886) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 886) 


s_ashr_i32 s4, s0, 31 // (op 886) 

s_xor_b32 s0, s1, s4 // (op 886) 

s_sub_i32 s1, s0, s4 // (op 886) 


s_mul_hi_i32 s0, s1, s13 // (op 886) 
s_add_i32 s4, s0, s1 // (op 886) 

s_ashr_i32 s0, s4, 31 // (op 886) 
s_cmp_eq_i32 s13, 0 // (op 886) 
s_mov_b32 s5, scc // (op 886) 
s_cmp_eq_i32 s5, 0 // (op 886) 
s_cselect_b32 s8, 0, -1 // (op 886) 

s_and_b32 s5, s14, 31 // (op 886) 
s_lshl_b32 s9, 1, s5 // (op 886) 
s_add_i32 s18, s9, s8 // (op 886) 


s_and_b32 s8, s0, s18 // (op 886) 


s_add_i32 s0, s4, s8 // (op 886) 


s_ashr_i32 s4, s0, s5 // (op 886) 


s_lshl_b32 s0, s14, 24 // (op 886) 
s_ashr_i32 s5, s0, 31 // (op 886) 

s_xor_b32 s0, s4, s5 // (op 886) 

s_sub_i32 s4, s0, s5 // (op 886) 


s_mul_i32 s0, s1, s46 // (op 886) 
s_sub_u32 s5, s16, s0 // (op 886) 

s_mul_i32 s0, s4, s46 // (op 886) 
s_add_u32 s0, s0, s5 // (op 886) 

s_mul_hi_i32 s5, s16, s10 // (op 886) 
s_add_i32 s8, s5, s16 // (op 886) 

s_ashr_i32 s5, s8, 31 // (op 886) 
s_cmp_eq_i32 s10, 0 // (op 886) 
s_mov_b32 s9, scc // (op 886) 
s_cmp_eq_i32 s9, 0 // (op 886) 
s_cselect_b32 s18, 0, -1 // (op 886) 

s_and_b32 s9, s49, 31 // (op 886) 
s_lshl_b32 s19, 1, s9 // (op 886) 
s_add_i32 s20, s19, s18 // (op 886) 


s_and_b32 s18, s5, s20 // (op 886) 


s_add_i32 s5, s8, s18 // (op 886) 


s_ashr_i32 s8, s5, s9 // (op 886) 


s_lshl_b32 s5, s49, 24 // (op 886) 
s_ashr_i32 s9, s5, 31 // (op 886) 

s_xor_b32 s5, s8, s9 // (op 886) 

s_sub_i32 s8, s5, s9 // (op 886) 


s_add_u32 s18, s2, 32 // (op 886) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 886) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 886) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 886) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 886) 

s_mul_i32 s5, s46, s18 // (op 886) 
s_mul_i32 s9, s50, s5 // (op 886) 

s_mul_i32 s5, s8, s9 // (op 886) 

s_sub_u32 s9, s16, s5 // (op 886) 

s_mov_b32 s5, s9 // (op 886) 
s_mul_hi_i32 s20, s5, s15 // (op 886) 
s_add_i32 s21, s20, s5 // (op 886) 


s_ashr_i32 s5, s21, 31 // (op 886) 
s_cmp_eq_i32 s15, 0 // (op 886) 
s_mov_b32 s20, scc // (op 886) 
s_cmp_eq_i32 s20, 0 // (op 886) 
s_cselect_b32 s22, 0, -1 // (op 886) 

s_and_b32 s20, s48, 31 // (op 886) 
s_lshl_b32 s23, 1, s20 // (op 886) 
s_add_i32 s24, s23, s22 // (op 886) 


s_and_b32 s22, s5, s24 // (op 886) 


s_add_i32 s5, s21, s22 // (op 886) 


s_ashr_i32 s21, s5, s20 // (op 886) 


s_lshl_b32 s5, s48, 24 // (op 886) 
s_ashr_i32 s20, s5, 31 // (op 886) 

s_xor_b32 s5, s21, s20 // (op 886) 

s_sub_i32 s21, s5, s20 // (op 886) 


s_mul_i32 s5, s47, s46 // (op 886) 
s_add_u32 s22, s6, 16 // (op 886) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 886) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 886) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 886) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 886) 

s_sub_i32 s20, s22, s5 // (op 886) 

s_mul_i32 s22, s21, s20 // (op 886) 

s_sub_u32 s20, s9, s22 // (op 886) 


s_add_u32 s9, s20, s5 // (op 886) 


s_cmp_eq_i32 s8, 0 // (op 886) 
s_mov_b32 s5, scc // (op 886) 

s_cmp_eq_i32 s5, 0 // (op 886) 
s_cselect_b32 s8, s9, s0 // (op 886) 


s_mul_i32 s0, s4, s18 // (op 886) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 886) 


s_sub_i32 s4, s1, s0 // (op 886) 


s_cmp_eq_i32 s5, 0 // (op 886) 
s_cselect_b32 s0, s21, s4 // (op 886) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 886) 


v_mul_lo_u32 v0, v4, s38 // (op 886) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 886) 


s_mul_i32 s1, s38, s3 // (op 888) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 888) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 888) least significant
s_mul_hi_u32 s5, s38, s2 // (op 888) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 888) most significant: sum
s_add_u32 s1, s1, s5 // (op 888) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 888) 

s_mov_b32 s20, s40 // (op 888) reg expression
s_mov_b32 s21, s41 // (op 888) reg expression
s_mov_b32 s22, s4 // (op 888) reg expression
s_mov_b32 s23, 131072 // (op 888) call()




buffer_store_dwordx4 v[10:13], v4, s[20:23], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123:
s_mul_hi_i32 s0, s16, s11 // (op 709) 
v_lshrrev_b32 v0, 7, v1 // (op 709) 
s_add_i32 s1, s0, s16 // (op 709) 

v_and_b32 v3, 63, v1 // (op 709) 
s_ashr_i32 s0, s1, 31 // (op 709) 
v_lshrrev_b32 v4, 4, v3 // (op 709) 
s_cmp_eq_i32 s11, 0 // (op 709) 
s_mov_b32 s4, scc // (op 709) 
v_lshlrev_b32 v5, 2, v4 // (op 709) 

s_cmp_eq_i32 s4, 0 // (op 709) 
s_cselect_b32 s5, 0, -1 // (op 709) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 709) 


s_and_b32 s4, s12, 31 // (op 709) 
v_lshrrev_b32 v0, 2, v3 // (op 709) 

s_lshl_b32 s8, 1, s4 // (op 709) 
v_and_b32 v3, 3, v0 // (op 709) 

s_add_i32 s9, s8, s5 // (op 709) 


v_and_b32 v0, 3, v1 // (op 709) 
s_and_b32 s5, s0, s9 // (op 709) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 709) 


s_add_i32 s0, s1, s5 // (op 709) 


v_lshrrev_b32 v0, 6, v1 // (op 709) 
s_ashr_i32 s1, s0, s4 // (op 709) 


v_and_b32 v3, 1, v0 // (op 709) 

s_lshl_b32 s0, s12, 24 // (op 709) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 709) 


s_ashr_i32 s4, s0, 31 // (op 709) 

s_xor_b32 s0, s1, s4 // (op 709) 

s_sub_i32 s1, s0, s4 // (op 709) 


s_mul_hi_i32 s0, s1, s13 // (op 709) 
s_add_i32 s4, s0, s1 // (op 709) 

s_ashr_i32 s0, s4, 31 // (op 709) 
s_cmp_eq_i32 s13, 0 // (op 709) 
s_mov_b32 s5, scc // (op 709) 
s_cmp_eq_i32 s5, 0 // (op 709) 
s_cselect_b32 s8, 0, -1 // (op 709) 

s_and_b32 s5, s14, 31 // (op 709) 
s_lshl_b32 s9, 1, s5 // (op 709) 
s_add_i32 s18, s9, s8 // (op 709) 


s_and_b32 s19, s0, s18 // (op 709) 


s_add_i32 s0, s4, s19 // (op 709) 


s_ashr_i32 s4, s0, s5 // (op 709) 


s_lshl_b32 s0, s14, 24 // (op 709) 
s_ashr_i32 s5, s0, 31 // (op 709) 

s_xor_b32 s0, s4, s5 // (op 709) 

s_sub_i32 s4, s0, s5 // (op 709) 


s_mul_i32 s0, s1, s46 // (op 709) 
s_sub_u32 s5, s16, s0 // (op 709) 

s_mul_i32 s0, s4, s46 // (op 709) 
s_add_u32 s0, s0, s5 // (op 709) 

s_mul_hi_i32 s5, s16, s10 // (op 709) 
s_add_i32 s8, s5, s16 // (op 709) 

s_ashr_i32 s5, s8, 31 // (op 709) 
s_cmp_eq_i32 s10, 0 // (op 709) 
s_mov_b32 s9, scc // (op 709) 
s_cmp_eq_i32 s9, 0 // (op 709) 
s_cselect_b32 s18, 0, -1 // (op 709) 

s_and_b32 s9, s49, 31 // (op 709) 
s_lshl_b32 s19, 1, s9 // (op 709) 
s_add_i32 s24, s19, s18 // (op 709) 


s_and_b32 s18, s5, s24 // (op 709) 


s_add_i32 s5, s8, s18 // (op 709) 


s_ashr_i32 s8, s5, s9 // (op 709) 


s_lshl_b32 s5, s49, 24 // (op 709) 
s_ashr_i32 s9, s5, 31 // (op 709) 

s_xor_b32 s5, s8, s9 // (op 709) 

s_sub_i32 s8, s5, s9 // (op 709) 


s_add_u32 s18, s2, 32 // (op 709) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 709) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 709) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 709) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 5 // (op 709) 

s_mul_i32 s5, s46, s18 // (op 709) 
s_mul_i32 s9, s50, s5 // (op 709) 

s_mul_i32 s5, s8, s9 // (op 709) 

s_sub_u32 s9, s16, s5 // (op 709) 

s_mov_b32 s5, s9 // (op 709) 
s_mul_hi_i32 s24, s5, s15 // (op 709) 
s_add_i32 s25, s24, s5 // (op 709) 


s_ashr_i32 s5, s25, 31 // (op 709) 
s_cmp_eq_i32 s15, 0 // (op 709) 
s_mov_b32 s24, scc // (op 709) 
s_cmp_eq_i32 s24, 0 // (op 709) 
s_cselect_b32 s26, 0, -1 // (op 709) 

s_and_b32 s24, s48, 31 // (op 709) 
s_lshl_b32 s27, 1, s24 // (op 709) 
s_add_i32 s28, s27, s26 // (op 709) 


s_and_b32 s26, s5, s28 // (op 709) 


s_add_i32 s5, s25, s26 // (op 709) 


s_ashr_i32 s25, s5, s24 // (op 709) 


s_lshl_b32 s5, s48, 24 // (op 709) 
s_ashr_i32 s24, s5, 31 // (op 709) 

s_xor_b32 s5, s25, s24 // (op 709) 

s_sub_i32 s25, s5, s24 // (op 709) 


s_mul_i32 s5, s47, s46 // (op 709) 
s_add_u32 s26, s6, 16 // (op 709) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 709) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 709) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 709) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 709) 

s_sub_i32 s24, s26, s5 // (op 709) 

s_mul_i32 s26, s25, s24 // (op 709) 

s_sub_u32 s24, s9, s26 // (op 709) 


s_add_u32 s9, s24, s5 // (op 709) 


s_cmp_eq_i32 s8, 0 // (op 709) 
s_mov_b32 s5, scc // (op 709) 

s_cmp_eq_i32 s5, 0 // (op 709) 
s_cselect_b32 s8, s9, s0 // (op 709) 


s_mul_i32 s0, s4, s18 // (op 709) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 709) 


s_sub_i32 s4, s1, s0 // (op 709) 


s_cmp_eq_i32 s5, 0 // (op 709) 
s_cselect_b32 s24, s25, s4 // (op 709) 



v_lshl_add_u32 v4, s24, 5, v0 // (op 709) 


v_mul_lo_u32 v0, v4, s42 // (op 709) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 709) 


s_mul_i32 s1, s42, s3 // (op 711) most significant: low of low * high
s_mul_i32 s4, s43, s2 // (op 711) most significant: low of high * low
s_mul_i32 s0, s42, s2 // (op 711) least significant
s_mul_hi_u32 s5, s42, s2 // (op 711) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 711) most significant: sum
s_add_u32 s1, s1, s5 // (op 711) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 711) 

s_mov_b32 s24, s44 // (op 711) reg expression
s_mov_b32 s25, s45 // (op 711) reg expression
s_mov_b32 s26, s4 // (op 711) reg expression
s_mov_b32 s27, 131072 // (op 711) call()





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


s_mul_hi_i32 s0, s16, s11 // (op 903) 
v_lshrrev_b32 v0, 7, v1 // (op 903) 
s_add_i32 s1, s0, s16 // (op 903) 

v_and_b32 v3, 63, v1 // (op 903) 
s_ashr_i32 s0, s1, 31 // (op 903) 
v_lshrrev_b32 v4, 4, v3 // (op 903) 
s_cmp_eq_i32 s11, 0 // (op 903) 
s_mov_b32 s4, scc // (op 903) 
v_lshlrev_b32 v5, 2, v4 // (op 903) 

s_cmp_eq_i32 s4, 0 // (op 903) 
s_cselect_b32 s5, 0, -1 // (op 903) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 903) 


s_and_b32 s4, s12, 31 // (op 903) 
v_lshrrev_b32 v0, 2, v3 // (op 903) 

s_lshl_b32 s8, 1, s4 // (op 903) 
v_and_b32 v3, 3, v0 // (op 903) 

s_add_i32 s9, s8, s5 // (op 903) 


v_and_b32 v0, 3, v1 // (op 903) 
s_and_b32 s5, s0, s9 // (op 903) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 903) 


s_add_i32 s0, s1, s5 // (op 903) 


v_lshrrev_b32 v0, 6, v1 // (op 903) 
s_ashr_i32 s1, s0, s4 // (op 903) 


v_and_b32 v3, 1, v0 // (op 903) 

s_lshl_b32 s0, s12, 24 // (op 903) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 903) 


s_ashr_i32 s4, s0, 31 // (op 903) 

s_xor_b32 s0, s1, s4 // (op 903) 

s_sub_i32 s1, s0, s4 // (op 903) 


s_mul_hi_i32 s0, s1, s13 // (op 903) 
s_add_i32 s4, s0, s1 // (op 903) 

s_ashr_i32 s0, s4, 31 // (op 903) 
s_cmp_eq_i32 s13, 0 // (op 903) 
s_mov_b32 s5, scc // (op 903) 
s_cmp_eq_i32 s5, 0 // (op 903) 
s_cselect_b32 s8, 0, -1 // (op 903) 

s_and_b32 s5, s14, 31 // (op 903) 
s_lshl_b32 s9, 1, s5 // (op 903) 
s_add_i32 s18, s9, s8 // (op 903) 


s_and_b32 s19, s0, s18 // (op 903) 


s_add_i32 s0, s4, s19 // (op 903) 


s_ashr_i32 s4, s0, s5 // (op 903) 


s_lshl_b32 s0, s14, 24 // (op 903) 
s_ashr_i32 s5, s0, 31 // (op 903) 

s_xor_b32 s0, s4, s5 // (op 903) 

s_sub_i32 s4, s0, s5 // (op 903) 


s_mul_i32 s0, s1, s46 // (op 903) 
s_sub_u32 s5, s16, s0 // (op 903) 

s_mul_i32 s0, s4, s46 // (op 903) 
s_add_u32 s0, s0, s5 // (op 903) 

s_mul_hi_i32 s5, s16, s10 // (op 903) 
s_add_i32 s8, s5, s16 // (op 903) 

s_ashr_i32 s5, s8, 31 // (op 903) 
s_cmp_eq_i32 s10, 0 // (op 903) 
s_mov_b32 s9, scc // (op 903) 
s_cmp_eq_i32 s9, 0 // (op 903) 
s_cselect_b32 s18, 0, -1 // (op 903) 

s_and_b32 s9, s49, 31 // (op 903) 
s_lshl_b32 s19, 1, s9 // (op 903) 
s_add_i32 s24, s19, s18 // (op 903) 


s_and_b32 s18, s5, s24 // (op 903) 


s_add_i32 s5, s8, s18 // (op 903) 


s_ashr_i32 s8, s5, s9 // (op 903) 


s_lshl_b32 s5, s49, 24 // (op 903) 
s_ashr_i32 s9, s5, 31 // (op 903) 

s_xor_b32 s5, s8, s9 // (op 903) 

s_sub_i32 s8, s5, s9 // (op 903) 


s_add_u32 s18, s2, 32 // (op 903) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 903) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 903) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 903) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 5 // (op 903) 

s_mul_i32 s5, s46, s18 // (op 903) 
s_mul_i32 s9, s50, s5 // (op 903) 

s_mul_i32 s5, s8, s9 // (op 903) 

s_sub_u32 s9, s16, s5 // (op 903) 

s_mov_b32 s5, s9 // (op 903) 
s_mul_hi_i32 s24, s5, s15 // (op 903) 
s_add_i32 s25, s24, s5 // (op 903) 


s_ashr_i32 s5, s25, 31 // (op 903) 
s_cmp_eq_i32 s15, 0 // (op 903) 
s_mov_b32 s24, scc // (op 903) 
s_cmp_eq_i32 s24, 0 // (op 903) 
s_cselect_b32 s26, 0, -1 // (op 903) 

s_and_b32 s24, s48, 31 // (op 903) 
s_lshl_b32 s27, 1, s24 // (op 903) 
s_add_i32 s28, s27, s26 // (op 903) 


s_and_b32 s26, s5, s28 // (op 903) 


s_add_i32 s5, s25, s26 // (op 903) 


s_ashr_i32 s25, s5, s24 // (op 903) 


s_lshl_b32 s5, s48, 24 // (op 903) 
s_ashr_i32 s24, s5, 31 // (op 903) 

s_xor_b32 s5, s25, s24 // (op 903) 

s_sub_i32 s25, s5, s24 // (op 903) 


s_mul_i32 s5, s47, s46 // (op 903) 
s_add_u32 s26, s6, 16 // (op 903) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 903) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 903) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 903) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 903) 

s_sub_i32 s24, s26, s5 // (op 903) 

s_mul_i32 s26, s25, s24 // (op 903) 

s_sub_u32 s24, s9, s26 // (op 903) 


s_add_u32 s9, s24, s5 // (op 903) 


s_cmp_eq_i32 s8, 0 // (op 903) 
s_mov_b32 s5, scc // (op 903) 

s_cmp_eq_i32 s5, 0 // (op 903) 
s_cselect_b32 s8, s9, s0 // (op 903) 


s_mul_i32 s0, s4, s18 // (op 903) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 903) 


s_sub_i32 s4, s1, s0 // (op 903) 


s_cmp_eq_i32 s5, 0 // (op 903) 
s_cselect_b32 s24, s25, s4 // (op 903) 



v_lshl_add_u32 v4, s24, 5, v0 // (op 903) 


v_mul_lo_u32 v0, v4, s38 // (op 903) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 903) 


s_mul_i32 s1, s38, s3 // (op 905) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 905) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 905) least significant
s_mul_hi_u32 s5, s38, s2 // (op 905) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 905) most significant: sum
s_add_u32 s1, s1, s5 // (op 905) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 905) 

s_mov_b32 s20, s40 // (op 905) reg expression
s_mov_b32 s21, s41 // (op 905) reg expression
s_mov_b32 s22, s4 // (op 905) reg expression
s_mov_b32 s23, 131072 // (op 905) call()




buffer_store_dwordx4 v[10:13], v4, s[20:23], 0 offen // (op 134) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.LRR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end:
.size RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_, .LRR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end-RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: F8
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
  - .name: RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_.kd
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

