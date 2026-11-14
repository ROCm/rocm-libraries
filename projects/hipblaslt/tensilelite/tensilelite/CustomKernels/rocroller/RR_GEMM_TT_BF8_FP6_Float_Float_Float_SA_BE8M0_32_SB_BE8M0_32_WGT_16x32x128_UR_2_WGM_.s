// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_,@function
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 765) 
v_and_b32 v3, 15, v1 // (op 765) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 765) 


v_and_b32 v0, 48, v1 // (op 765) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 765) 


v_and_b32 v0, 63, v1 // (op 818) 
v_and_b32 v4, 48, v1 // (op 818) 
v_sub_u32 v5, v0, v4 // (op 818) 

v_lshrrev_b32 v4, 4, v0 // (op 818) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 818) 


v_lshrrev_b32 v4, 6, v1 // (op 818) 
v_and_b32 v5, 1, v4 // (op 818) 

v_lshlrev_b32 v4, 4, v5 // (op 818) 

v_lshl_add_u32 v5, v0, 5, v4 // (op 818) 


v_mul_lo_u32 v0, v5, 6 // (op 818) 
v_lshrrev_b32 v4, 3, v0 // (op 818) 

v_lshrrev_b32 v0, 4, v5 // (op 818) 

v_lshlrev_b32 v5, 2, v0 // (op 818) 

v_add_u32 v0, v4, v5 // (op 818) 


v_lshrrev_b32 v4, 6, v1 // (op 871) 
v_and_b32 v5, 1, v4 // (op 871) 

v_and_b32 v4, 15, v1 // (op 871) 
v_lshl_add_u32 v6, v5, 4, v4 // (op 871) 


v_and_b32 v4, 63, v1 // (op 871) 
v_lshrrev_b32 v5, 4, v4 // (op 871) 

v_lshl_add_u32 v4, v5, 5, v6 // (op 871) 


v_lshrrev_b32 v5, 7, v1 // (op 944) 
v_and_b32 v6, 15, v1 // (op 944) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 944) 


v_and_b32 v5, 48, v1 // (op 944) 
v_lshl_add_u32 v6, v7, 7, v5 // (op 944) 


v_and_b32 v5, 63, v1 // (op 968) 
v_and_b32 v7, 48, v1 // (op 968) 
v_sub_u32 v8, v5, v7 // (op 968) 

v_lshrrev_b32 v7, 4, v5 // (op 968) 

v_lshl_add_u32 v5, v7, 5, v8 // (op 968) 


v_lshrrev_b32 v7, 6, v1 // (op 968) 
v_and_b32 v8, 1, v7 // (op 968) 

v_lshlrev_b32 v7, 4, v8 // (op 968) 

v_lshl_add_u32 v8, v5, 5, v7 // (op 968) 


v_mul_lo_u32 v5, v8, 6 // (op 968) 
v_lshrrev_b32 v7, 3, v5 // (op 968) 

v_lshrrev_b32 v5, 4, v8 // (op 968) 

v_lshlrev_b32 v8, 2, v5 // (op 968) 

v_add_u32 v5, v7, v8 // (op 968) 


v_lshrrev_b32 v7, 6, v1 // (op 992) 
v_and_b32 v8, 1, v7 // (op 992) 

v_and_b32 v7, 15, v1 // (op 992) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 992) 


v_and_b32 v7, 63, v1 // (op 992) 
v_lshrrev_b32 v8, 4, v7 // (op 992) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 992) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v9, s9 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
s_mul_hi_i32 s0, s16, s11 // (op 476) 
v_lshrrev_b32 v10, 3, v1 // (op 476) 
s_add_i32 s1, s0, s16 // (op 476) 

v_and_b32 v11, -8, v1 // (op 476) 
s_ashr_i32 s0, s1, 31 // (op 476) 
v_sub_u32 v12, v1, v11 // (op 476) 

s_cmp_eq_i32 s11, 0 // (op 476) 
s_mov_b32 s8, scc // (op 476) 
v_lshlrev_b32 v11, 4, v12 // (op 476) 

s_cmp_eq_i32 s8, 0 // (op 476) 
s_cselect_b32 s9, 0, -1 // (op 476) 

v_mov_b32 v13, 0 // (op 476) convert
v_mov_b32 v12, v11 // (op 476) convert

s_and_b32 s8, s12, 31 // (op 476) 
s_lshl_b32 s18, 1, s8 // (op 476) 
s_add_i32 s19, s18, s9 // (op 476) 


s_and_b32 s9, s0, s19 // (op 476) 


s_add_i32 s0, s1, s9 // (op 476) 


s_ashr_i32 s1, s0, s8 // (op 476) 


s_lshl_b32 s0, s12, 24 // (op 476) 
s_ashr_i32 s8, s0, 31 // (op 476) 

s_xor_b32 s0, s1, s8 // (op 476) 

s_sub_i32 s1, s0, s8 // (op 476) 


s_mul_hi_i32 s0, s1, s13 // (op 476) 
s_add_i32 s8, s0, s1 // (op 476) 

s_ashr_i32 s0, s8, 31 // (op 476) 
s_cmp_eq_i32 s13, 0 // (op 476) 
s_mov_b32 s9, scc // (op 476) 
s_cmp_eq_i32 s9, 0 // (op 476) 
s_cselect_b32 s18, 0, -1 // (op 476) 

s_and_b32 s9, s14, 31 // (op 476) 
s_lshl_b32 s19, 1, s9 // (op 476) 
s_add_i32 s51, s19, s18 // (op 476) 


s_and_b32 s18, s0, s51 // (op 476) 


s_add_i32 s0, s8, s18 // (op 476) 


s_ashr_i32 s8, s0, s9 // (op 476) 


s_lshl_b32 s0, s14, 24 // (op 476) 
s_ashr_i32 s9, s0, 31 // (op 476) 

s_xor_b32 s0, s8, s9 // (op 476) 

s_sub_i32 s8, s0, s9 // (op 476) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s46 // (op 476) 

s_sub_u32 s1, s16, s0 // (op 476) 

s_mul_i32 s0, s8, s46 // (op 476) 
s_add_u32 s0, s0, s1 // (op 476) 


s_mul_hi_i32 s1, s16, s10 // (op 476) 
s_add_i32 s8, s1, s16 // (op 476) 

s_ashr_i32 s1, s8, 31 // (op 476) 
s_cmp_eq_i32 s10, 0 // (op 476) 
s_mov_b32 s9, scc // (op 476) 
s_cmp_eq_i32 s9, 0 // (op 476) 
s_cselect_b32 s18, 0, -1 // (op 476) 

s_and_b32 s9, s49, 31 // (op 476) 
s_lshl_b32 s19, 1, s9 // (op 476) 
s_add_i32 s51, s19, s18 // (op 476) 


s_and_b32 s18, s1, s51 // (op 476) 


s_add_i32 s1, s8, s18 // (op 476) 


s_ashr_i32 s8, s1, s9 // (op 476) 


s_lshl_b32 s1, s49, 24 // (op 476) 
s_ashr_i32 s9, s1, 31 // (op 476) 

s_xor_b32 s1, s8, s9 // (op 476) 

s_sub_i32 s8, s1, s9 // (op 476) 


s_add_u32 s18, s2, 32 // (op 476) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 476) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 476) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 476) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 5 // (op 476) 

s_mul_i32 s1, s46, s18 // (op 476) 

s_mul_i32 s9, s50, s1 // (op 476) 

s_mul_i32 s1, s8, s9 // (op 476) 

s_sub_u32 s9, s16, s1 // (op 476) 

s_mov_b32 s1, s9 // (op 476) 
s_mul_hi_i32 s18, s1, s15 // (op 476) 
s_add_i32 s19, s18, s1 // (op 476) 


s_ashr_i32 s1, s19, 31 // (op 476) 
s_cmp_eq_i32 s15, 0 // (op 476) 
s_mov_b32 s18, scc // (op 476) 
s_cmp_eq_i32 s18, 0 // (op 476) 
s_cselect_b32 s51, 0, -1 // (op 476) 

s_and_b32 s18, s48, 31 // (op 476) 
s_lshl_b32 s52, 1, s18 // (op 476) 
s_add_i32 s53, s52, s51 // (op 476) 


s_and_b32 s51, s1, s53 // (op 476) 


s_add_i32 s1, s19, s51 // (op 476) 


s_ashr_i32 s19, s1, s18 // (op 476) 


s_lshl_b32 s1, s48, 24 // (op 476) 
s_ashr_i32 s18, s1, 31 // (op 476) 

s_xor_b32 s1, s19, s18 // (op 476) 

s_sub_i32 s19, s1, s18 // (op 476) 


s_mul_i32 s1, s47, s46 // (op 476) 
s_add_u32 s52, s6, 16 // (op 476) Start of Int64 add, locking SCC
s_addc_u32 s53, s7, 0 // (op 476) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 476) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 476) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 476) 

s_sub_i32 s18, s52, s1 // (op 476) 

s_mul_i32 s51, s19, s18 // (op 476) 


s_sub_u32 s18, s9, s51 // (op 476) 


s_add_u32 s9, s18, s1 // (op 476) 


s_cmp_eq_i32 s8, 0 // (op 476) 
s_mov_b32 s1, scc // (op 476) 

s_cmp_eq_i32 s1, 0 // (op 476) 
s_cselect_b32 s8, s9, s0 // (op 476) 



v_lshl_add_u32 v11, s8, 4, v10 // (op 476) 


v_mul_lo_u32 v15, v11, s5 // (op 476) most significant: low of low * high
v_mul_hi_u32 v10, v11, s4 // (op 476) most significant: high of low * low
v_mul_lo_u32 v14, v11, s4 // (op 476) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 476) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 476) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 476) most significant half



s_mul_i32 s1, s4, s7 // (op 478) most significant: low of low * high
s_mul_i32 s8, s5, s6 // (op 478) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 478) least significant
s_mul_hi_u32 s9, s4, s6 // (op 478) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 478) most significant: sum
s_add_u32 s1, s1, s9 // (op 478) most significant: sum


s_mov_b32 s52, s20 // (op 478) reg expression
s_mov_b32 s53, s21 // (op 478) reg expression
s_mov_b32 s54, s0 // (op 478) reg expression
s_mov_b32 s55, 131072 // (op 478) call()

s_mul_hi_i32 s51, s16, s11 // (op 495) 
v_lshrrev_b32 v10, 3, v1 // (op 495) 
s_add_i32 s0, s51, s16 // (op 495) 

v_and_b32 v11, -8, v1 // (op 495) 
s_ashr_i32 s1, s0, 31 // (op 495) 
v_sub_u32 v12, v1, v11 // (op 495) 

s_cmp_eq_i32 s11, 0 // (op 495) 
s_mov_b32 s51, scc // (op 495) 
v_lshlrev_b32 v11, 4, v12 // (op 495) 

s_cmp_eq_i32 s51, 0 // (op 495) 
s_cselect_b32 s8, 0, -1 // (op 495) 

v_mov_b32 v13, 0 // (op 495) convert
v_mov_b32 v12, v11 // (op 495) convert

s_and_b32 s9, s12, 31 // (op 495) 
s_lshl_b32 s51, 1, s9 // (op 495) 
s_add_i32 s18, s51, s8 // (op 495) 


s_and_b32 s8, s1, s18 // (op 495) 


s_add_i32 s1, s0, s8 // (op 495) 


s_ashr_i32 s0, s1, s9 // (op 495) 


s_lshl_b32 s1, s12, 24 // (op 495) 
s_ashr_i32 s51, s1, 31 // (op 495) 

s_xor_b32 s1, s0, s51 // (op 495) 

s_sub_i32 s0, s1, s51 // (op 495) 


s_mul_hi_i32 s1, s0, s13 // (op 495) 
s_add_i32 s51, s1, s0 // (op 495) 

s_ashr_i32 s1, s51, 31 // (op 495) 
s_cmp_eq_i32 s13, 0 // (op 495) 
s_mov_b32 s8, scc // (op 495) 
s_cmp_eq_i32 s8, 0 // (op 495) 
s_cselect_b32 s9, 0, -1 // (op 495) 

s_and_b32 s8, s14, 31 // (op 495) 
s_lshl_b32 s18, 1, s8 // (op 495) 
s_add_i32 s19, s18, s9 // (op 495) 


s_and_b32 s9, s1, s19 // (op 495) 


s_add_i32 s1, s51, s9 // (op 495) 


s_ashr_i32 s9, s1, s8 // (op 495) 


s_lshl_b32 s1, s14, 24 // (op 495) 
s_ashr_i32 s8, s1, 31 // (op 495) 

s_xor_b32 s1, s9, s8 // (op 495) 

s_sub_i32 s9, s1, s8 // (op 495) 


s_mul_i32 s1, s0, s46 // (op 495) 

s_sub_u32 s0, s16, s1 // (op 495) 

s_mul_i32 s1, s9, s46 // (op 495) 
s_add_u32 s1, s1, s0 // (op 495) 


s_mul_hi_i32 s0, s16, s10 // (op 495) 
s_add_i32 s51, s0, s16 // (op 495) 

s_ashr_i32 s0, s51, 31 // (op 495) 
s_cmp_eq_i32 s10, 0 // (op 495) 
s_mov_b32 s8, scc // (op 495) 
s_cmp_eq_i32 s8, 0 // (op 495) 
s_cselect_b32 s9, 0, -1 // (op 495) 

s_and_b32 s8, s49, 31 // (op 495) 
s_lshl_b32 s18, 1, s8 // (op 495) 
s_add_i32 s19, s18, s9 // (op 495) 


s_and_b32 s9, s0, s19 // (op 495) 


s_add_i32 s0, s51, s9 // (op 495) 


s_ashr_i32 s9, s0, s8 // (op 495) 


s_lshl_b32 s0, s49, 24 // (op 495) 
s_ashr_i32 s8, s0, 31 // (op 495) 

s_xor_b32 s0, s9, s8 // (op 495) 

s_sub_i32 s9, s0, s8 // (op 495) 


s_add_u32 s18, s2, 32 // (op 495) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 495) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 495) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 495) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 495) 

s_mul_i32 s0, s46, s18 // (op 495) 

s_mul_i32 s8, s50, s0 // (op 495) 

s_mul_i32 s0, s9, s8 // (op 495) 

s_sub_u32 s8, s16, s0 // (op 495) 

s_mov_b32 s0, s8 // (op 495) 
s_mul_hi_i32 s51, s0, s15 // (op 495) 
s_add_i32 s18, s51, s0 // (op 495) 


s_ashr_i32 s0, s18, 31 // (op 495) 
s_cmp_eq_i32 s15, 0 // (op 495) 
s_mov_b32 s19, scc // (op 495) 
s_cmp_eq_i32 s19, 0 // (op 495) 
s_cselect_b32 s51, 0, -1 // (op 495) 

s_and_b32 s19, s48, 31 // (op 495) 
s_lshl_b32 s56, 1, s19 // (op 495) 
s_add_i32 s57, s56, s51 // (op 495) 


s_and_b32 s51, s0, s57 // (op 495) 


s_add_i32 s0, s18, s51 // (op 495) 


s_ashr_i32 s18, s0, s19 // (op 495) 


s_lshl_b32 s0, s48, 24 // (op 495) 
s_ashr_i32 s19, s0, 31 // (op 495) 

s_xor_b32 s0, s18, s19 // (op 495) 

s_sub_i32 s18, s0, s19 // (op 495) 


s_mul_i32 s0, s47, s46 // (op 495) 
s_add_u32 s56, s6, 16 // (op 495) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 495) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 495) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 495) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 495) 

s_sub_i32 s19, s56, s0 // (op 495) 

s_mul_i32 s51, s18, s19 // (op 495) 


s_sub_u32 s18, s8, s51 // (op 495) 


s_add_u32 s8, s18, s0 // (op 495) 


s_cmp_eq_i32 s9, 0 // (op 495) 
s_mov_b32 s0, scc // (op 495) 

s_cmp_eq_i32 s0, 0 // (op 495) 
s_cselect_b32 s9, s8, s1 // (op 495) 



v_lshl_add_u32 v11, s9, 4, v10 // (op 495) 


v_mul_lo_u32 v17, v11, s5 // (op 495) most significant: low of low * high
v_mul_hi_u32 v10, v11, s4 // (op 495) most significant: high of low * low
v_mul_lo_u32 v16, v11, s4 // (op 495) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 495) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 495) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 495) most significant half



s_mul_i32 s1, s4, s7 // (op 497) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 497) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 497) least significant
s_mul_hi_u32 s8, s4, s6 // (op 497) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 497) most significant: sum
s_add_u32 s1, s1, s8 // (op 497) most significant: sum


s_mov_b32 s52, s20 // (op 497) reg expression
s_mov_b32 s53, s21 // (op 497) reg expression
s_mov_b32 s54, s0 // (op 497) reg expression
s_mov_b32 s55, 131072 // (op 497) call()

s_mul_hi_i32 s51, s16, s11 // (op 540) 
s_add_i32 s0, s51, s16 // (op 540) 

s_ashr_i32 s1, s0, 31 // (op 540) 
s_cmp_eq_i32 s11, 0 // (op 540) 
s_mov_b32 s51, scc // (op 540) 
s_cmp_eq_i32 s51, 0 // (op 540) 
s_cselect_b32 s8, 0, -1 // (op 540) 

s_and_b32 s9, s12, 31 // (op 540) 
s_lshl_b32 s51, 1, s9 // (op 540) 
s_add_i32 s18, s51, s8 // (op 540) 


s_and_b32 s8, s1, s18 // (op 540) 


s_add_i32 s1, s0, s8 // (op 540) 


s_ashr_i32 s0, s1, s9 // (op 540) 


s_lshl_b32 s1, s12, 24 // (op 540) 
s_ashr_i32 s51, s1, 31 // (op 540) 

s_xor_b32 s1, s0, s51 // (op 540) 

s_sub_i32 s0, s1, s51 // (op 540) 


s_mul_hi_i32 s1, s0, s13 // (op 540) 
s_add_i32 s51, s1, s0 // (op 540) 

s_ashr_i32 s1, s51, 31 // (op 540) 
s_cmp_eq_i32 s13, 0 // (op 540) 
s_mov_b32 s8, scc // (op 540) 
s_cmp_eq_i32 s8, 0 // (op 540) 
s_cselect_b32 s9, 0, -1 // (op 540) 

s_and_b32 s8, s14, 31 // (op 540) 
s_lshl_b32 s18, 1, s8 // (op 540) 
s_add_i32 s19, s18, s9 // (op 540) 


s_and_b32 s9, s1, s19 // (op 540) 


s_add_i32 s1, s51, s9 // (op 540) 


s_ashr_i32 s9, s1, s8 // (op 540) 


s_lshl_b32 s1, s14, 24 // (op 540) 
s_ashr_i32 s8, s1, 31 // (op 540) 

s_xor_b32 s1, s9, s8 // (op 540) 

s_sub_i32 s9, s1, s8 // (op 540) 


s_add_u32 s18, s2, 32 // (op 540) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 540) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 540) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 540) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 540) 

s_mul_i32 s1, s9, s18 // (op 540) 

s_sub_i32 s51, s0, s1 // (op 540) 


s_mul_hi_i32 s0, s16, s10 // (op 540) 
s_add_i32 s1, s0, s16 // (op 540) 

s_ashr_i32 s0, s1, 31 // (op 540) 
s_cmp_eq_i32 s10, 0 // (op 540) 
s_mov_b32 s8, scc // (op 540) 
s_cmp_eq_i32 s8, 0 // (op 540) 
s_cselect_b32 s9, 0, -1 // (op 540) 

s_and_b32 s8, s49, 31 // (op 540) 
s_lshl_b32 s56, 1, s8 // (op 540) 
s_add_i32 s57, s56, s9 // (op 540) 


s_and_b32 s9, s0, s57 // (op 540) 


s_add_i32 s0, s1, s9 // (op 540) 


s_ashr_i32 s1, s0, s8 // (op 540) 


s_lshl_b32 s0, s49, 24 // (op 540) 
s_ashr_i32 s8, s0, 31 // (op 540) 

s_xor_b32 s0, s1, s8 // (op 540) 

s_sub_i32 s1, s0, s8 // (op 540) 


s_mul_i32 s0, s46, s18 // (op 540) 

s_mul_i32 s8, s50, s0 // (op 540) 

s_mul_i32 s0, s1, s8 // (op 540) 

s_sub_u32 s8, s16, s0 // (op 540) 

s_mul_hi_i32 s0, s8, s15 // (op 540) 
s_add_i32 s9, s0, s8 // (op 540) 


s_ashr_i32 s0, s9, 31 // (op 540) 
s_cmp_eq_i32 s15, 0 // (op 540) 
s_mov_b32 s8, scc // (op 540) 
s_cmp_eq_i32 s8, 0 // (op 540) 
s_cselect_b32 s18, 0, -1 // (op 540) 

s_and_b32 s8, s48, 31 // (op 540) 
s_lshl_b32 s19, 1, s8 // (op 540) 
s_add_i32 s56, s19, s18 // (op 540) 


s_and_b32 s18, s0, s56 // (op 540) 


s_add_i32 s0, s9, s18 // (op 540) 


s_ashr_i32 s9, s0, s8 // (op 540) 


s_lshl_b32 s0, s48, 24 // (op 540) 
s_ashr_i32 s8, s0, 31 // (op 540) 

s_xor_b32 s0, s9, s8 // (op 540) 

s_sub_i32 s9, s0, s8 // (op 540) 


s_cmp_eq_i32 s1, 0 // (op 540) 
s_mov_b32 s0, scc // (op 540) 

s_cmp_eq_i32 s0, 0 // (op 540) 
s_cselect_b32 s1, s9, s51 // (op 540) 



s_lshl_b32 s0, s1, 5 // (op 540) 

s_mov_b32 s9, 0 // (op 540) convert
s_mov_b32 s8, s0 // (op 540) convert

v_mul_lo_u32 v11, v1, s23 // (op 540) most significant: low of low * high
v_mul_hi_u32 v12, v1, s22 // (op 540) most significant: high of low * low
v_mul_lo_u32 v10, v1, s22 // (op 540) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 540) most significant: sum

v_mov_b32 v12, s8 // (op 540) 
v_mov_b32 v13, s9 // (op 540) 
v_add_co_u32 v10, s[0:1], v10, v12 // (op 540) least significant half
v_addc_co_u32 v11, s[0:1], v11, v13, s[0:1] // (op 540) most significant half




v_mul_lo_u32 v13, v11, 6 // (op 540) most significant: low of high * low
v_mul_hi_u32 v18, v10, 6 // (op 540) most significant: high of low * low
v_mul_lo_u32 v12, v10, 6 // (op 540) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 540) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 540) 

v_mov_b32 v12, v10 // (op 540) convert

s_mul_i32 s1, s27, 6 // (op 542) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 542) least significant
s_mul_hi_u32 s51, s26, 6 // (op 542) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 542) most significant: sum

s_mul_i32 s9, s22, s1 // (op 542) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 542) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 542) least significant
s_mul_hi_u32 s18, s22, s0 // (op 542) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 542) most significant: sum
s_add_u32 s9, s9, s18 // (op 542) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 542) 

s_mov_b32 s56, s24 // (op 542) reg expression
s_mov_b32 s57, s25 // (op 542) reg expression
s_mov_b32 s58, s0 // (op 542) reg expression
s_mov_b32 s59, 131072 // (op 542) call()

s_mul_hi_i32 s51, s16, s11 // (op 559) 
s_add_i32 s0, s51, s16 // (op 559) 

s_ashr_i32 s1, s0, 31 // (op 559) 
s_cmp_eq_i32 s11, 0 // (op 559) 
s_mov_b32 s51, scc // (op 559) 
s_cmp_eq_i32 s51, 0 // (op 559) 
s_cselect_b32 s8, 0, -1 // (op 559) 

s_and_b32 s9, s12, 31 // (op 559) 
s_lshl_b32 s51, 1, s9 // (op 559) 
s_add_i32 s18, s51, s8 // (op 559) 


s_and_b32 s8, s1, s18 // (op 559) 


s_add_i32 s1, s0, s8 // (op 559) 


s_ashr_i32 s0, s1, s9 // (op 559) 


s_lshl_b32 s1, s12, 24 // (op 559) 
s_ashr_i32 s51, s1, 31 // (op 559) 

s_xor_b32 s1, s0, s51 // (op 559) 

s_sub_i32 s0, s1, s51 // (op 559) 


s_mul_hi_i32 s1, s0, s13 // (op 559) 
s_add_i32 s51, s1, s0 // (op 559) 

s_ashr_i32 s1, s51, 31 // (op 559) 
s_cmp_eq_i32 s13, 0 // (op 559) 
s_mov_b32 s8, scc // (op 559) 
s_cmp_eq_i32 s8, 0 // (op 559) 
s_cselect_b32 s9, 0, -1 // (op 559) 

s_and_b32 s8, s14, 31 // (op 559) 
s_lshl_b32 s18, 1, s8 // (op 559) 
s_add_i32 s19, s18, s9 // (op 559) 


s_and_b32 s9, s1, s19 // (op 559) 


s_add_i32 s1, s51, s9 // (op 559) 


s_ashr_i32 s9, s1, s8 // (op 559) 


s_lshl_b32 s1, s14, 24 // (op 559) 
s_ashr_i32 s8, s1, 31 // (op 559) 

s_xor_b32 s1, s9, s8 // (op 559) 

s_sub_i32 s9, s1, s8 // (op 559) 


s_add_u32 s18, s2, 32 // (op 559) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 559) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 559) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 559) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 559) 

s_mul_i32 s1, s9, s18 // (op 559) 

s_sub_i32 s51, s0, s1 // (op 559) 


s_mul_hi_i32 s0, s16, s10 // (op 559) 
s_add_i32 s1, s0, s16 // (op 559) 

s_ashr_i32 s0, s1, 31 // (op 559) 
s_cmp_eq_i32 s10, 0 // (op 559) 
s_mov_b32 s8, scc // (op 559) 
s_cmp_eq_i32 s8, 0 // (op 559) 
s_cselect_b32 s9, 0, -1 // (op 559) 

s_and_b32 s8, s49, 31 // (op 559) 
s_lshl_b32 s60, 1, s8 // (op 559) 
s_add_i32 s61, s60, s9 // (op 559) 


s_and_b32 s9, s0, s61 // (op 559) 


s_add_i32 s0, s1, s9 // (op 559) 


s_ashr_i32 s1, s0, s8 // (op 559) 


s_lshl_b32 s0, s49, 24 // (op 559) 
s_ashr_i32 s8, s0, 31 // (op 559) 

s_xor_b32 s0, s1, s8 // (op 559) 

s_sub_i32 s1, s0, s8 // (op 559) 


s_mul_i32 s0, s46, s18 // (op 559) 

s_mul_i32 s8, s50, s0 // (op 559) 

s_mul_i32 s0, s1, s8 // (op 559) 

s_sub_u32 s8, s16, s0 // (op 559) 

s_mul_hi_i32 s0, s8, s15 // (op 559) 
s_add_i32 s9, s0, s8 // (op 559) 


s_ashr_i32 s0, s9, 31 // (op 559) 
s_cmp_eq_i32 s15, 0 // (op 559) 
s_mov_b32 s8, scc // (op 559) 
s_cmp_eq_i32 s8, 0 // (op 559) 
s_cselect_b32 s18, 0, -1 // (op 559) 

s_and_b32 s8, s48, 31 // (op 559) 
s_lshl_b32 s19, 1, s8 // (op 559) 
s_add_i32 s60, s19, s18 // (op 559) 


s_and_b32 s18, s0, s60 // (op 559) 


s_add_i32 s0, s9, s18 // (op 559) 


s_ashr_i32 s9, s0, s8 // (op 559) 


s_lshl_b32 s0, s48, 24 // (op 559) 
s_ashr_i32 s8, s0, 31 // (op 559) 

s_xor_b32 s0, s9, s8 // (op 559) 

s_sub_i32 s9, s0, s8 // (op 559) 


s_cmp_eq_i32 s1, 0 // (op 559) 
s_mov_b32 s0, scc // (op 559) 

s_cmp_eq_i32 s0, 0 // (op 559) 
s_cselect_b32 s1, s9, s51 // (op 559) 



s_lshl_b32 s0, s1, 5 // (op 559) 

s_mov_b32 s9, 0 // (op 559) convert
s_mov_b32 s8, s0 // (op 559) convert

v_mul_lo_u32 v11, v1, s23 // (op 559) most significant: low of low * high
v_mul_hi_u32 v13, v1, s22 // (op 559) most significant: high of low * low
v_mul_lo_u32 v10, v1, s22 // (op 559) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 559) most significant: sum

v_mov_b32 v13, s8 // (op 559) 
v_mov_b32 v18, s9 // (op 559) 
v_add_co_u32 v10, s[0:1], v10, v13 // (op 559) least significant half
v_addc_co_u32 v11, s[0:1], v11, v18, s[0:1] // (op 559) most significant half




v_mul_lo_u32 v19, v11, 6 // (op 559) most significant: low of high * low
v_mul_hi_u32 v13, v10, 6 // (op 559) most significant: high of low * low
v_mul_lo_u32 v18, v10, 6 // (op 559) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 559) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[18:19] // (op 559) 

v_mov_b32 v13, v10 // (op 559) convert

s_mul_i32 s1, s27, 6 // (op 561) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 561) least significant
s_mul_hi_u32 s51, s26, 6 // (op 561) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 561) most significant: sum

s_mul_i32 s9, s22, s1 // (op 561) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 561) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 561) least significant
s_mul_hi_u32 s18, s22, s0 // (op 561) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 561) most significant: sum
s_add_u32 s9, s9, s18 // (op 561) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 561) 

s_mov_b32 s56, s24 // (op 561) reg expression
s_mov_b32 s57, s25 // (op 561) reg expression
s_mov_b32 s58, s0 // (op 561) reg expression
s_mov_b32 s59, 131072 // (op 561) call()

s_mul_hi_i32 s51, s16, s11 // (op 604) 
v_lshrrev_b32 v10, 7, v1 // (op 604) 
s_add_i32 s0, s51, s16 // (op 604) 

v_and_b32 v11, 15, v1 // (op 604) 
s_ashr_i32 s1, s0, 31 // (op 604) 
v_lshl_add_u32 v18, v10, 4, v11 // (op 604) 


s_cmp_eq_i32 s11, 0 // (op 604) 
s_mov_b32 s51, scc // (op 604) 
v_and_b32 v10, 63, v1 // (op 604) 
s_cmp_eq_i32 s51, 0 // (op 604) 
s_cselect_b32 s8, 0, -1 // (op 604) 

v_lshrrev_b32 v11, 4, v10 // (op 604) 

s_and_b32 s9, s12, 31 // (op 604) 
s_lshl_b32 s51, 1, s9 // (op 604) 
s_add_i32 s18, s51, s8 // (op 604) 


s_and_b32 s8, s1, s18 // (op 604) 


s_add_i32 s1, s0, s8 // (op 604) 


s_ashr_i32 s0, s1, s9 // (op 604) 


s_lshl_b32 s1, s12, 24 // (op 604) 
s_ashr_i32 s51, s1, 31 // (op 604) 

s_xor_b32 s1, s0, s51 // (op 604) 

s_sub_i32 s0, s1, s51 // (op 604) 


s_mul_hi_i32 s1, s0, s13 // (op 604) 
s_add_i32 s51, s1, s0 // (op 604) 

s_ashr_i32 s1, s51, 31 // (op 604) 
s_cmp_eq_i32 s13, 0 // (op 604) 
s_mov_b32 s8, scc // (op 604) 
s_cmp_eq_i32 s8, 0 // (op 604) 
s_cselect_b32 s9, 0, -1 // (op 604) 

s_and_b32 s8, s14, 31 // (op 604) 
s_lshl_b32 s18, 1, s8 // (op 604) 
s_add_i32 s19, s18, s9 // (op 604) 


s_and_b32 s9, s1, s19 // (op 604) 


s_add_i32 s1, s51, s9 // (op 604) 


s_ashr_i32 s9, s1, s8 // (op 604) 


s_lshl_b32 s1, s14, 24 // (op 604) 
s_ashr_i32 s8, s1, 31 // (op 604) 

s_xor_b32 s1, s9, s8 // (op 604) 

s_sub_i32 s9, s1, s8 // (op 604) 


s_mul_i32 s1, s0, s46 // (op 604) 

s_sub_u32 s0, s16, s1 // (op 604) 

s_mul_i32 s1, s9, s46 // (op 604) 
s_add_u32 s1, s1, s0 // (op 604) 


s_mul_hi_i32 s0, s16, s10 // (op 604) 
s_add_i32 s51, s0, s16 // (op 604) 

s_ashr_i32 s0, s51, 31 // (op 604) 
s_cmp_eq_i32 s10, 0 // (op 604) 
s_mov_b32 s8, scc // (op 604) 
s_cmp_eq_i32 s8, 0 // (op 604) 
s_cselect_b32 s9, 0, -1 // (op 604) 

s_and_b32 s8, s49, 31 // (op 604) 
s_lshl_b32 s18, 1, s8 // (op 604) 
s_add_i32 s19, s18, s9 // (op 604) 


s_and_b32 s9, s0, s19 // (op 604) 


s_add_i32 s0, s51, s9 // (op 604) 


s_ashr_i32 s9, s0, s8 // (op 604) 


s_lshl_b32 s0, s49, 24 // (op 604) 
s_ashr_i32 s8, s0, 31 // (op 604) 

s_xor_b32 s0, s9, s8 // (op 604) 

s_sub_i32 s9, s0, s8 // (op 604) 


s_add_u32 s18, s2, 32 // (op 604) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 604) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 604) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 604) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 604) 

s_mul_i32 s0, s46, s18 // (op 604) 

s_mul_i32 s8, s50, s0 // (op 604) 

s_mul_i32 s0, s9, s8 // (op 604) 

s_sub_u32 s8, s16, s0 // (op 604) 

s_mov_b32 s0, s8 // (op 604) 
s_mul_hi_i32 s51, s0, s15 // (op 604) 
s_add_i32 s18, s51, s0 // (op 604) 


s_ashr_i32 s0, s18, 31 // (op 604) 
s_cmp_eq_i32 s15, 0 // (op 604) 
s_mov_b32 s19, scc // (op 604) 
s_cmp_eq_i32 s19, 0 // (op 604) 
s_cselect_b32 s51, 0, -1 // (op 604) 

s_and_b32 s19, s48, 31 // (op 604) 
s_lshl_b32 s60, 1, s19 // (op 604) 
s_add_i32 s61, s60, s51 // (op 604) 


s_and_b32 s51, s0, s61 // (op 604) 


s_add_i32 s0, s18, s51 // (op 604) 


s_ashr_i32 s18, s0, s19 // (op 604) 


s_lshl_b32 s0, s48, 24 // (op 604) 
s_ashr_i32 s19, s0, 31 // (op 604) 

s_xor_b32 s0, s18, s19 // (op 604) 

s_sub_i32 s18, s0, s19 // (op 604) 


s_mul_i32 s0, s47, s46 // (op 604) 
s_add_u32 s60, s6, 16 // (op 604) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 604) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 604) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 604) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 604) 

s_sub_i32 s19, s60, s0 // (op 604) 

s_mul_i32 s51, s18, s19 // (op 604) 


s_sub_u32 s18, s8, s51 // (op 604) 


s_add_u32 s8, s18, s0 // (op 604) 


s_cmp_eq_i32 s9, 0 // (op 604) 
s_mov_b32 s0, scc // (op 604) 

s_cmp_eq_i32 s0, 0 // (op 604) 
s_cselect_b32 s9, s8, s1 // (op 604) 



v_lshl_add_u32 v10, s9, 4, v18 // (op 604) 


v_mul_lo_u32 v18, v10, s28 // (op 604) 
v_add_u32 v18, v18, v11 // (op 604) 


s_mul_i32 s1, s28, s7 // (op 606) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 606) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 606) least significant
s_mul_hi_u32 s8, s28, s6 // (op 606) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 606) most significant: sum
s_add_u32 s1, s1, s8 // (op 606) most significant: sum


s_mov_b32 s60, s30 // (op 606) reg expression
s_mov_b32 s61, s31 // (op 606) reg expression
s_mov_b32 s62, s0 // (op 606) reg expression
s_mov_b32 s63, 131072 // (op 606) call()

s_mul_hi_i32 s51, s16, s11 // (op 623) 
v_lshrrev_b32 v10, 7, v1 // (op 623) 
s_add_i32 s0, s51, s16 // (op 623) 

v_and_b32 v11, 15, v1 // (op 623) 
s_ashr_i32 s1, s0, 31 // (op 623) 
v_lshl_add_u32 v19, v10, 4, v11 // (op 623) 


s_cmp_eq_i32 s11, 0 // (op 623) 
s_mov_b32 s51, scc // (op 623) 
v_and_b32 v10, 63, v1 // (op 623) 
s_cmp_eq_i32 s51, 0 // (op 623) 
s_cselect_b32 s8, 0, -1 // (op 623) 

v_lshrrev_b32 v11, 4, v10 // (op 623) 

s_and_b32 s9, s12, 31 // (op 623) 
s_lshl_b32 s51, 1, s9 // (op 623) 
s_add_i32 s18, s51, s8 // (op 623) 


s_and_b32 s8, s1, s18 // (op 623) 


s_add_i32 s1, s0, s8 // (op 623) 


s_ashr_i32 s0, s1, s9 // (op 623) 


s_lshl_b32 s1, s12, 24 // (op 623) 
s_ashr_i32 s51, s1, 31 // (op 623) 

s_xor_b32 s1, s0, s51 // (op 623) 

s_sub_i32 s0, s1, s51 // (op 623) 


s_mul_hi_i32 s1, s0, s13 // (op 623) 
s_add_i32 s51, s1, s0 // (op 623) 

s_ashr_i32 s1, s51, 31 // (op 623) 
s_cmp_eq_i32 s13, 0 // (op 623) 
s_mov_b32 s8, scc // (op 623) 
s_cmp_eq_i32 s8, 0 // (op 623) 
s_cselect_b32 s9, 0, -1 // (op 623) 

s_and_b32 s8, s14, 31 // (op 623) 
s_lshl_b32 s18, 1, s8 // (op 623) 
s_add_i32 s19, s18, s9 // (op 623) 


s_and_b32 s9, s1, s19 // (op 623) 


s_add_i32 s1, s51, s9 // (op 623) 


s_ashr_i32 s9, s1, s8 // (op 623) 


s_lshl_b32 s1, s14, 24 // (op 623) 
s_ashr_i32 s8, s1, 31 // (op 623) 

s_xor_b32 s1, s9, s8 // (op 623) 

s_sub_i32 s9, s1, s8 // (op 623) 


s_mul_i32 s1, s0, s46 // (op 623) 

s_sub_u32 s0, s16, s1 // (op 623) 

s_mul_i32 s1, s9, s46 // (op 623) 
s_add_u32 s1, s1, s0 // (op 623) 


s_mul_hi_i32 s0, s16, s10 // (op 623) 
s_add_i32 s51, s0, s16 // (op 623) 

s_ashr_i32 s0, s51, 31 // (op 623) 
s_cmp_eq_i32 s10, 0 // (op 623) 
s_mov_b32 s8, scc // (op 623) 
s_cmp_eq_i32 s8, 0 // (op 623) 
s_cselect_b32 s9, 0, -1 // (op 623) 

s_and_b32 s8, s49, 31 // (op 623) 
s_lshl_b32 s18, 1, s8 // (op 623) 
s_add_i32 s19, s18, s9 // (op 623) 


s_and_b32 s9, s0, s19 // (op 623) 


s_add_i32 s0, s51, s9 // (op 623) 


s_ashr_i32 s9, s0, s8 // (op 623) 


s_lshl_b32 s0, s49, 24 // (op 623) 
s_ashr_i32 s8, s0, 31 // (op 623) 

s_xor_b32 s0, s9, s8 // (op 623) 

s_sub_i32 s9, s0, s8 // (op 623) 


s_add_u32 s18, s2, 32 // (op 623) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 623) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 623) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 623) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 623) 

s_mul_i32 s0, s46, s18 // (op 623) 

s_mul_i32 s8, s50, s0 // (op 623) 

s_mul_i32 s0, s9, s8 // (op 623) 

s_sub_u32 s8, s16, s0 // (op 623) 

s_mov_b32 s0, s8 // (op 623) 
s_mul_hi_i32 s51, s0, s15 // (op 623) 
s_add_i32 s18, s51, s0 // (op 623) 


s_ashr_i32 s0, s18, 31 // (op 623) 
s_cmp_eq_i32 s15, 0 // (op 623) 
s_mov_b32 s19, scc // (op 623) 
s_cmp_eq_i32 s19, 0 // (op 623) 
s_cselect_b32 s51, 0, -1 // (op 623) 

s_and_b32 s19, s48, 31 // (op 623) 
s_lshl_b32 s64, 1, s19 // (op 623) 
s_add_i32 s65, s64, s51 // (op 623) 


s_and_b32 s51, s0, s65 // (op 623) 


s_add_i32 s0, s18, s51 // (op 623) 


s_ashr_i32 s18, s0, s19 // (op 623) 


s_lshl_b32 s0, s48, 24 // (op 623) 
s_ashr_i32 s19, s0, 31 // (op 623) 

s_xor_b32 s0, s18, s19 // (op 623) 

s_sub_i32 s18, s0, s19 // (op 623) 


s_mul_i32 s0, s47, s46 // (op 623) 
s_add_u32 s64, s6, 16 // (op 623) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 623) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 623) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 623) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 623) 

s_sub_i32 s19, s64, s0 // (op 623) 

s_mul_i32 s51, s18, s19 // (op 623) 


s_sub_u32 s18, s8, s51 // (op 623) 


s_add_u32 s8, s18, s0 // (op 623) 


s_cmp_eq_i32 s9, 0 // (op 623) 
s_mov_b32 s0, scc // (op 623) 

s_cmp_eq_i32 s0, 0 // (op 623) 
s_cselect_b32 s9, s8, s1 // (op 623) 



v_lshl_add_u32 v10, s9, 4, v19 // (op 623) 


v_mul_lo_u32 v19, v10, s28 // (op 623) 
v_add_u32 v19, v19, v11 // (op 623) 


s_mul_i32 s1, s28, s7 // (op 625) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 625) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 625) least significant
s_mul_hi_u32 s8, s28, s6 // (op 625) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 625) most significant: sum
s_add_u32 s1, s1, s8 // (op 625) most significant: sum


s_mov_b32 s60, s30 // (op 625) reg expression
s_mov_b32 s61, s31 // (op 625) reg expression
s_mov_b32 s62, s0 // (op 625) reg expression
s_mov_b32 s63, 131072 // (op 625) call()

s_mul_hi_i32 s51, s16, s11 // (op 668) 
v_and_b32 v10, -32, v1 // (op 668) 
s_add_i32 s0, s51, s16 // (op 668) 

v_sub_u32 v11, v1, v10 // (op 668) 

s_ashr_i32 s1, s0, 31 // (op 668) 
v_lshrrev_b32 v10, 5, v1 // (op 668) 
s_cmp_eq_i32 s11, 0 // (op 668) 
s_mov_b32 s51, scc // (op 668) 
s_cmp_eq_i32 s51, 0 // (op 668) 
s_cselect_b32 s8, 0, -1 // (op 668) 

s_and_b32 s9, s12, 31 // (op 668) 
s_lshl_b32 s51, 1, s9 // (op 668) 
s_add_i32 s18, s51, s8 // (op 668) 


s_and_b32 s8, s1, s18 // (op 668) 


s_add_i32 s1, s0, s8 // (op 668) 


s_ashr_i32 s0, s1, s9 // (op 668) 


s_lshl_b32 s1, s12, 24 // (op 668) 
s_ashr_i32 s51, s1, 31 // (op 668) 

s_xor_b32 s1, s0, s51 // (op 668) 

s_sub_i32 s0, s1, s51 // (op 668) 


s_mul_hi_i32 s1, s0, s13 // (op 668) 
s_add_i32 s51, s1, s0 // (op 668) 

s_ashr_i32 s1, s51, 31 // (op 668) 
s_cmp_eq_i32 s13, 0 // (op 668) 
s_mov_b32 s8, scc // (op 668) 
s_cmp_eq_i32 s8, 0 // (op 668) 
s_cselect_b32 s9, 0, -1 // (op 668) 

s_and_b32 s8, s14, 31 // (op 668) 
s_lshl_b32 s18, 1, s8 // (op 668) 
s_add_i32 s19, s18, s9 // (op 668) 


s_and_b32 s9, s1, s19 // (op 668) 


s_add_i32 s1, s51, s9 // (op 668) 


s_ashr_i32 s9, s1, s8 // (op 668) 


s_lshl_b32 s1, s14, 24 // (op 668) 
s_ashr_i32 s8, s1, 31 // (op 668) 

s_xor_b32 s1, s9, s8 // (op 668) 

s_sub_i32 s9, s1, s8 // (op 668) 


s_add_u32 s18, s2, 32 // (op 668) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 668) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 668) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 668) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 668) 

s_mul_i32 s1, s9, s18 // (op 668) 

s_sub_i32 s51, s0, s1 // (op 668) 


s_mul_hi_i32 s0, s16, s10 // (op 668) 
s_add_i32 s1, s0, s16 // (op 668) 

s_ashr_i32 s0, s1, 31 // (op 668) 
s_cmp_eq_i32 s10, 0 // (op 668) 
s_mov_b32 s8, scc // (op 668) 
s_cmp_eq_i32 s8, 0 // (op 668) 
s_cselect_b32 s9, 0, -1 // (op 668) 

s_and_b32 s8, s49, 31 // (op 668) 
s_lshl_b32 s64, 1, s8 // (op 668) 
s_add_i32 s65, s64, s9 // (op 668) 


s_and_b32 s9, s0, s65 // (op 668) 


s_add_i32 s0, s1, s9 // (op 668) 


s_ashr_i32 s1, s0, s8 // (op 668) 


s_lshl_b32 s0, s49, 24 // (op 668) 
s_ashr_i32 s8, s0, 31 // (op 668) 

s_xor_b32 s0, s1, s8 // (op 668) 

s_sub_i32 s1, s0, s8 // (op 668) 


s_mul_i32 s0, s46, s18 // (op 668) 

s_mul_i32 s8, s50, s0 // (op 668) 

s_mul_i32 s0, s1, s8 // (op 668) 

s_sub_u32 s8, s16, s0 // (op 668) 

s_mul_hi_i32 s0, s8, s15 // (op 668) 
s_add_i32 s9, s0, s8 // (op 668) 


s_ashr_i32 s0, s9, 31 // (op 668) 
s_cmp_eq_i32 s15, 0 // (op 668) 
s_mov_b32 s8, scc // (op 668) 
s_cmp_eq_i32 s8, 0 // (op 668) 
s_cselect_b32 s18, 0, -1 // (op 668) 

s_and_b32 s8, s48, 31 // (op 668) 
s_lshl_b32 s19, 1, s8 // (op 668) 
s_add_i32 s64, s19, s18 // (op 668) 


s_and_b32 s18, s0, s64 // (op 668) 


s_add_i32 s0, s9, s18 // (op 668) 


s_ashr_i32 s9, s0, s8 // (op 668) 


s_lshl_b32 s0, s48, 24 // (op 668) 
s_ashr_i32 s8, s0, 31 // (op 668) 

s_xor_b32 s0, s9, s8 // (op 668) 

s_sub_i32 s9, s0, s8 // (op 668) 


s_cmp_eq_i32 s1, 0 // (op 668) 
s_mov_b32 s0, scc // (op 668) 

s_cmp_eq_i32 s0, 0 // (op 668) 
s_cselect_b32 s1, s9, s51 // (op 668) 



v_lshl_add_u32 v20, s1, 5, v11 // (op 668) 


v_mul_lo_u32 v11, v10, s32 // (op 668) 
v_add_u32 v11, v11, v20 // (op 668) 


s_mul_i32 s1, s32, s37 // (op 670) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 670) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 670) least significant
s_mul_hi_u32 s8, s32, s36 // (op 670) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 670) most significant: sum
s_add_u32 s1, s1, s8 // (op 670) most significant: sum


s_mov_b32 s64, s34 // (op 670) reg expression
s_mov_b32 s65, s35 // (op 670) reg expression
s_mov_b32 s66, s0 // (op 670) reg expression
s_mov_b32 s67, 131072 // (op 670) call()

s_mul_hi_i32 s51, s16, s11 // (op 687) 
v_and_b32 v10, -32, v1 // (op 687) 
s_add_i32 s0, s51, s16 // (op 687) 

v_sub_u32 v20, v1, v10 // (op 687) 

s_ashr_i32 s1, s0, 31 // (op 687) 
v_lshrrev_b32 v10, 5, v1 // (op 687) 
s_cmp_eq_i32 s11, 0 // (op 687) 
s_mov_b32 s51, scc // (op 687) 
s_cmp_eq_i32 s51, 0 // (op 687) 
s_cselect_b32 s8, 0, -1 // (op 687) 

s_and_b32 s9, s12, 31 // (op 687) 
s_lshl_b32 s51, 1, s9 // (op 687) 
s_add_i32 s18, s51, s8 // (op 687) 


s_and_b32 s8, s1, s18 // (op 687) 


s_add_i32 s1, s0, s8 // (op 687) 


s_ashr_i32 s0, s1, s9 // (op 687) 


s_lshl_b32 s1, s12, 24 // (op 687) 
s_ashr_i32 s51, s1, 31 // (op 687) 

s_xor_b32 s1, s0, s51 // (op 687) 

s_sub_i32 s0, s1, s51 // (op 687) 


s_mul_hi_i32 s1, s0, s13 // (op 687) 
s_add_i32 s51, s1, s0 // (op 687) 

s_ashr_i32 s1, s51, 31 // (op 687) 
s_cmp_eq_i32 s13, 0 // (op 687) 
s_mov_b32 s8, scc // (op 687) 
s_cmp_eq_i32 s8, 0 // (op 687) 
s_cselect_b32 s9, 0, -1 // (op 687) 

s_and_b32 s8, s14, 31 // (op 687) 
s_lshl_b32 s18, 1, s8 // (op 687) 
s_add_i32 s19, s18, s9 // (op 687) 


s_and_b32 s9, s1, s19 // (op 687) 


s_add_i32 s1, s51, s9 // (op 687) 


s_ashr_i32 s9, s1, s8 // (op 687) 


s_lshl_b32 s1, s14, 24 // (op 687) 
s_ashr_i32 s8, s1, 31 // (op 687) 

s_xor_b32 s1, s9, s8 // (op 687) 

s_sub_i32 s9, s1, s8 // (op 687) 


s_add_u32 s18, s2, 32 // (op 687) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 687) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 687) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 687) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 687) 

s_mul_i32 s1, s9, s18 // (op 687) 

s_sub_i32 s51, s0, s1 // (op 687) 


s_mul_hi_i32 s0, s16, s10 // (op 687) 
s_add_i32 s1, s0, s16 // (op 687) 

s_ashr_i32 s0, s1, 31 // (op 687) 
s_cmp_eq_i32 s10, 0 // (op 687) 
s_mov_b32 s8, scc // (op 687) 
s_cmp_eq_i32 s8, 0 // (op 687) 
s_cselect_b32 s9, 0, -1 // (op 687) 

s_and_b32 s8, s49, 31 // (op 687) 
s_lshl_b32 s68, 1, s8 // (op 687) 
s_add_i32 s69, s68, s9 // (op 687) 


s_and_b32 s9, s0, s69 // (op 687) 


s_add_i32 s0, s1, s9 // (op 687) 


s_ashr_i32 s1, s0, s8 // (op 687) 


s_lshl_b32 s0, s49, 24 // (op 687) 
s_ashr_i32 s8, s0, 31 // (op 687) 

s_xor_b32 s0, s1, s8 // (op 687) 

s_sub_i32 s1, s0, s8 // (op 687) 


s_mul_i32 s0, s46, s18 // (op 687) 

s_mul_i32 s8, s50, s0 // (op 687) 

s_mul_i32 s0, s1, s8 // (op 687) 

s_sub_u32 s8, s16, s0 // (op 687) 

s_mul_hi_i32 s0, s8, s15 // (op 687) 
s_add_i32 s9, s0, s8 // (op 687) 


s_ashr_i32 s0, s9, 31 // (op 687) 
s_cmp_eq_i32 s15, 0 // (op 687) 
s_mov_b32 s8, scc // (op 687) 
s_cmp_eq_i32 s8, 0 // (op 687) 
s_cselect_b32 s18, 0, -1 // (op 687) 

s_and_b32 s8, s48, 31 // (op 687) 
s_lshl_b32 s19, 1, s8 // (op 687) 
s_add_i32 s68, s19, s18 // (op 687) 


s_and_b32 s18, s0, s68 // (op 687) 


s_add_i32 s0, s9, s18 // (op 687) 


s_ashr_i32 s9, s0, s8 // (op 687) 


s_lshl_b32 s0, s48, 24 // (op 687) 
s_ashr_i32 s8, s0, 31 // (op 687) 

s_xor_b32 s0, s9, s8 // (op 687) 

s_sub_i32 s9, s0, s8 // (op 687) 


s_cmp_eq_i32 s1, 0 // (op 687) 
s_mov_b32 s0, scc // (op 687) 

s_cmp_eq_i32 s0, 0 // (op 687) 
s_cselect_b32 s1, s9, s51 // (op 687) 



v_lshl_add_u32 v21, s1, 5, v20 // (op 687) 


v_mul_lo_u32 v20, v10, s32 // (op 687) 
v_add_u32 v20, v20, v21 // (op 687) 


s_mul_i32 s1, s32, s37 // (op 689) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 689) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 689) least significant
s_mul_hi_u32 s8, s32, s36 // (op 689) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 689) most significant: sum
s_add_u32 s1, s1, s8 // (op 689) most significant: sum


s_mov_b32 s64, s34 // (op 689) reg expression
s_mov_b32 s65, s35 // (op 689) reg expression
s_mov_b32 s66, s0 // (op 689) reg expression
s_mov_b32 s67, 131072 // (op 689) call()




buffer_load_ubyte v10, v18, s[60:63], 0 offen offset: 0 // (op 337) Load value


v_mov_b32 v21, v12 // (op 341) reg expression

buffer_load_dwordx3 v[22:24], v21, s[56:59], 0 offen // (op 341) Load value
buffer_load_dwordx3 v[26:28], v21, s[56:59], 0 offen offset:12 // (op 341) Load value



v_mov_b32 v21, v11 // (op 344) reg expression

buffer_load_ubyte v25, v21, s[64:67], 0 offen offset: 0 // (op 344) Load value

v_and_b32 v21, -8, v1 // (op 753) 
v_sub_u32 v29, v1, v21 // (op 753) 

v_lshlrev_b32 v21, 4, v29 // (op 753) 

v_lshrrev_b32 v29, 3, v1 // (op 753) 
v_lshl_add_u32 v30, v29, 7, v21 // (op 753) 


s_nop 0
v_readfirstlane_b32 s51, v30 // (op 753) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s51 // (op 432) reg expression
buffer_load_dwordx4 v14, s[52:55], 0 offen offset: 0 lds // (op 432) Load value direct to lds




buffer_load_ubyte v21, v18, s[60:63], 0 offen offset:4 // (op 351) Load value



s_lshl_b64 s[0:1], s[22:23], 7 // (op 355) 
s_mul_i32 s9, s1, 6 // (op 355) most significant: low of high * low
s_mul_i32 s8, s0, 6 // (op 355) least significant
s_mul_hi_u32 s51, s0, 6 // (op 355) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 355) most significant: sum


s_lshr_b64 s[0:1], s[8:9], 3 // (op 355) 

v_add_u32 v18, v12, s0 // (op 355) 


buffer_load_dwordx3 v[30:32], v18, s[56:59], 0 offen // (op 355) Load value
buffer_load_dwordx3 v[34:36], v18, s[56:59], 0 offen offset:12 // (op 355) Load value




s_lshl_b32 s51, s32, 2 // (op 358) 
v_add_u32 v12, v11, s51 // (op 358) 


buffer_load_ubyte v18, v12, s[64:67], 0 offen offset: 0 // (op 358) Load value


v_and_b32 v29, -8, v1 // (op 932) 
v_sub_u32 v33, v1, v29 // (op 932) 

v_lshlrev_b32 v29, 4, v33 // (op 932) 

v_lshrrev_b32 v33, 3, v1 // (op 932) 
v_lshl_add_u32 v11, v33, 7, v29 // (op 932) 


s_nop 0
v_readfirstlane_b32 s51, v11 // (op 932) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s0, 2048, s51 // (op 437) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s0 // (op 437) reg expression
s_mov_b32 s1, 128 // (op 437) call()
buffer_load_dwordx4 v14, s[52:55], s1 offen offset: 0 lds // (op 437) Load value direct to lds




v_lshlrev_b32 v29, 5, v1 // (op 806) 
v_mul_lo_u32 v33, v29, 6 // (op 806) 
v_lshrrev_b32 v11, 3, v33 // (op 806) 

v_lshrrev_b32 v12, 4, v29 // (op 806) 

v_lshlrev_b32 v29, 2, v12 // (op 806) 

v_add_u32 v12, v11, v29 // (op 806) 





s_waitcnt vmcnt(8)
ds_write_b96 v12, v[22:24] offset:4096 // (op 363) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v12, v[26:28] offset:4112 // (op 363) Store local data 

v_and_b32 v29, -32, v1 // (op 859) 
v_sub_u32 v33, v1, v29 // (op 859) 

v_and_b32 v29, 4294967264, v1 // (op 859) 
v_add_u32 v11, v29, v33 // (op 859) 





s_waitcnt vmcnt(6)
ds_write_b8 v11, v25 offset:8192 // (op 364) Store local data 

v_and_b32 v29, -8, v1 // (op 725) 
v_sub_u32 v33, v1, v29 // (op 725) 

v_lshlrev_b32 v29, 4, v33 // (op 725) 

v_lshrrev_b32 v33, 3, v1 // (op 725) 
v_lshl_add_u32 v11, v33, 7, v29 // (op 725) 


s_nop 0
v_readfirstlane_b32 s51, v11 // (op 725) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 5, v1 // (op 777) 
v_mul_lo_u32 v33, v29, 6 // (op 777) 
v_lshrrev_b32 v11, 3, v33 // (op 777) 

v_lshrrev_b32 v12, 4, v29 // (op 777) 

v_lshlrev_b32 v29, 2, v12 // (op 777) 

v_add_u32 v12, v11, v29 // (op 777) 


v_and_b32 v11, -32, v1 // (op 830) 
v_sub_u32 v29, v1, v11 // (op 830) 

v_and_b32 v11, 4294967264, v1 // (op 830) 
v_add_u32 v33, v11, v29 // (op 830) 


v_and_b32 v11, -8, v1 // (op 917) 
v_sub_u32 v29, v1, v11 // (op 917) 

v_lshlrev_b32 v11, 4, v29 // (op 917) 

v_lshrrev_b32 v29, 3, v1 // (op 917) 
v_lshl_add_u32 v14, v29, 7, v11 // (op 917) 


s_nop 0
v_readfirstlane_b32 s0, v14 // (op 917) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v11, 5, v1 // (op 956) 
v_mul_lo_u32 v29, v11, 6 // (op 956) 
v_lshrrev_b32 v14, 3, v29 // (op 956) 

v_lshrrev_b32 v15, 4, v11 // (op 956) 

v_lshlrev_b32 v11, 2, v15 // (op 956) 

v_add_u32 v15, v14, v11 // (op 956) 


v_and_b32 v11, -32, v1 // (op 980) 
v_sub_u32 v14, v1, v11 // (op 980) 

v_and_b32 v11, 4294967264, v1 // (op 980) 
v_add_u32 v29, v11, v14 // (op 980) 


s_mov_b32 s1, 0 // (op 55) call()
s_ashr_i64 s[8:9], s[26:27], 7 // (op 54) 
s_mov_b32 s18, s8 // (op 54) convert

s_and_b32 s19, s18, 4294967294 // (op 54) 

s_cmp_lt_u32 s1, s19 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54 if false)
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54:
s_add_u32 s1, s1, 2 // (op 56) 
s_waitcnt vmcnt(5)
s_barrier  // (op 1025) 



ds_read_b128 v[38:41], v3 // (op 65) Load local data 
ds_read_b128 v[42:45], v3 offset:64 // (op 65) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1038) 



ds_read_b96_tr_b6 v[46:48], v0 offset:4096 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[49:51], v0 offset:4608 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1052) 


ds_read_u8 v11, v4 offset:8192 // (op 79) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[46:51], a[0:3], v10, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 82) 





v_mov_b32 v11, v19 // (op 75) 

buffer_load_ubyte v10, v11, s[60:63], 0 offen offset:8 // (op 75) Load value


v_mov_b32 v14, v13 // (op 10) 
s_lshl_b64 s[8:9], s[22:23], 7 // (op 10) 
s_mul_i32 s19, s9, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s8, 6 // (op 10) least significant
s_mul_hi_u32 s68, s8, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 10) most significant: sum


s_lshr_b64 s[8:9], s[18:19], 3 // (op 10) 

s_lshl_b32 s18, s8, 1 // (op 10) 

v_add_u32 v37, v14, s18 // (op 10) 


buffer_load_dwordx3 v[22:24], v37, s[56:59], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[26:28], v37, s[56:59], 0 offen offset:12 // (op 10) Load value



v_mov_b32 v37, v20 // (op 21) 
s_lshl_b32 s8, s32, 3 // (op 21) 
v_add_u32 v38, v37, s8 // (op 21) 


buffer_load_ubyte v25, v38, s[64:67], 0 offen offset: 0 // (op 21) Load value

s_barrier  // (op 1028) 


v_mov_b64 v[38:39], v[16:17] // (op 416) 

s_mov_b32 m0, s51 // (op 416) reg expression
s_mov_b32 s8, 256 // (op 416) call()
buffer_load_dwordx4 v38, s[52:55], s8 offen offset: 0 lds // (op 416) Load value direct to lds

s_barrier  // (op 1075) 



s_waitcnt vmcnt(8)
ds_write_b96 v15, v[30:32] offset:8320 // (op 235) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v15, v[34:36] offset:8336 // (op 235) Store local data 
s_barrier  // (op 1081) 



s_waitcnt vmcnt(6)
ds_write_b8 v29, v18 offset:12416 // (op 236) Store local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1066) 



ds_read_b128 v[40:43], v6 offset:2048 // (op 227) Load local data 
ds_read_b128 v[44:47], v6 offset:2112 // (op 227) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1072) 



ds_read_b96_tr_b6 v[48:50], v5 offset:8320 // (op 229) Transpose load local data 
ds_read_b96_tr_b6 v[51:53], v5 offset:8832 // (op 229) Transpose load local data 
s_barrier  // (op 1128) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1078) 


ds_read_u8 v54, v7 offset:12416 // (op 233) Load local data 
s_barrier  // (op 1136) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:53], a[0:3], v21, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 234) 






buffer_load_ubyte v21, v11, s[60:63], 0 offen offset:12 // (op 231) Load value



v_add_u32 v19, 8, v19 // (op 615) 
s_lshl_b64 s[8:9], s[22:23], 7 // (op 237) 
s_mul_i32 s19, s9, 6 // (op 237) most significant: low of high * low
s_mul_i32 s18, s8, 6 // (op 237) least significant
s_mul_hi_u32 s68, s8, 6 // (op 237) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 237) most significant: sum


s_lshr_b64 s[8:9], s[18:19], 3 // (op 237) 

s_mul_i32 s18, 3, s8 // (op 237) 

v_add_u32 v11, v14, s18 // (op 237) 


buffer_load_dwordx3 v[30:32], v11, s[56:59], 0 offen // (op 237) Load value
buffer_load_dwordx3 v[34:36], v11, s[56:59], 0 offen offset:12 // (op 237) Load value




s_lshl_b32 s8, s32, 2 // (op 238) 
s_mul_i32 s9, 3, s8 // (op 238) 

v_add_u32 v11, v37, s9 // (op 238) 


s_lshl_b64 s[8:9], s[22:23], 7 // (op 551) 
s_mul_i32 s19, s9, 6 // (op 551) most significant: low of high * low
s_mul_i32 s18, s8, 6 // (op 551) least significant
s_mul_hi_u32 s68, s8, 6 // (op 551) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 551) most significant: sum


s_lshr_b64 s[8:9], s[18:19], 3 // (op 551) 

s_lshl_b32 s18, s8, 1 // (op 551) 

v_add_u32 v13, v13, s18 // (op 551) 

buffer_load_ubyte v18, v11, s[64:67], 0 offen offset: 0 // (op 238) Load value


s_barrier  // (op 1069) 
s_lshl_b32 s8, s32, 3 // (op 679) 


v_add_u32 v20, v20, s8 // (op 679) 

s_add_u32 s8, 2048, s0 // (op 427) 

s_mov_b32 m0, s8 // (op 427) reg expression
s_mov_b32 s9, 384 // (op 427) call()
buffer_load_dwordx4 v38, s[52:55], s9 offen offset: 0 lds // (op 427) Load value direct to lds



s_barrier  // (op 1041) 
v_add_co_u32 v16, vcc, 256, v16 // (op 487) least significant half



v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 487) most significant half
s_waitcnt vmcnt(8)
ds_write_b96 v12, v[22:24] offset:4096 // (op 13) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v12, v[26:28] offset:4112 // (op 13) Store local data 
s_barrier  // (op 1055) 



s_waitcnt vmcnt(6)
ds_write_b8 v33, v25 offset:8192 // (op 26) Store local data 
s_ashr_i64 s[8:9], s[26:27], 7 // (op 54) 
s_mov_b32 s18, s8 // (op 54) convert

s_and_b32 s19, s18, 4294967294 // (op 54) 

s_cmp_lt_u32 s1, s19 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54 if true)
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54:














s_barrier  // (op 1092) 
s_mul_hi_i32 s51, s16, s11 // (op 450) 
v_lshrrev_b32 v33, 3, v1 // (op 450) 
s_add_i32 s0, s51, s16 // (op 450) 

v_and_b32 v10, -8, v1 // (op 450) 
s_ashr_i32 s1, s0, 31 // (op 450) 
v_sub_u32 v11, v1, v10 // (op 450) 

s_cmp_eq_i32 s11, 0 // (op 450) 
s_mov_b32 s51, scc // (op 450) 
v_lshlrev_b32 v10, 4, v11 // (op 450) 

s_cmp_eq_i32 s51, 0 // (op 450) 
s_cselect_b32 s8, 0, -1 // (op 450) 

s_and_b32 s9, s12, 31 // (op 450) 
s_lshl_b32 s51, 1, s9 // (op 450) 
s_add_i32 s18, s51, s8 // (op 450) 


s_and_b32 s8, s1, s18 // (op 450) 


s_add_i32 s1, s0, s8 // (op 450) 


s_ashr_i32 s0, s1, s9 // (op 450) 


s_lshl_b32 s1, s12, 24 // (op 450) 
s_ashr_i32 s51, s1, 31 // (op 450) 

s_xor_b32 s1, s0, s51 // (op 450) 

s_sub_i32 s0, s1, s51 // (op 450) 


s_mul_hi_i32 s1, s0, s13 // (op 450) 
s_add_i32 s51, s1, s0 // (op 450) 

s_ashr_i32 s1, s51, 31 // (op 450) 
s_cmp_eq_i32 s13, 0 // (op 450) 
s_mov_b32 s8, scc // (op 450) 
s_cmp_eq_i32 s8, 0 // (op 450) 
s_cselect_b32 s9, 0, -1 // (op 450) 

s_and_b32 s8, s14, 31 // (op 450) 
s_lshl_b32 s18, 1, s8 // (op 450) 
s_add_i32 s19, s18, s9 // (op 450) 


s_and_b32 s9, s1, s19 // (op 450) 


s_add_i32 s1, s51, s9 // (op 450) 


s_ashr_i32 s9, s1, s8 // (op 450) 


s_lshl_b32 s1, s14, 24 // (op 450) 
s_ashr_i32 s8, s1, 31 // (op 450) 

s_xor_b32 s1, s9, s8 // (op 450) 

s_sub_i32 s9, s1, s8 // (op 450) 


s_mul_i32 s1, s0, s46 // (op 450) 

s_sub_u32 s0, s16, s1 // (op 450) 

s_mul_i32 s1, s9, s46 // (op 450) 
s_add_u32 s1, s1, s0 // (op 450) 


s_mul_hi_i32 s0, s16, s10 // (op 450) 
s_add_i32 s51, s0, s16 // (op 450) 

s_ashr_i32 s0, s51, 31 // (op 450) 
s_cmp_eq_i32 s10, 0 // (op 450) 
s_mov_b32 s8, scc // (op 450) 
s_cmp_eq_i32 s8, 0 // (op 450) 
s_cselect_b32 s9, 0, -1 // (op 450) 

s_and_b32 s8, s49, 31 // (op 450) 
s_lshl_b32 s18, 1, s8 // (op 450) 
s_add_i32 s19, s18, s9 // (op 450) 


s_and_b32 s9, s0, s19 // (op 450) 


s_add_i32 s0, s51, s9 // (op 450) 


s_ashr_i32 s9, s0, s8 // (op 450) 


s_lshl_b32 s0, s49, 24 // (op 450) 
s_ashr_i32 s8, s0, 31 // (op 450) 

s_xor_b32 s0, s9, s8 // (op 450) 

s_sub_i32 s9, s0, s8 // (op 450) 


s_add_u32 s18, s2, 32 // (op 450) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 450) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 450) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 450) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 450) 

s_mul_i32 s0, s46, s18 // (op 450) 

s_mul_i32 s8, s50, s0 // (op 450) 

s_mul_i32 s0, s9, s8 // (op 450) 

s_sub_u32 s8, s16, s0 // (op 450) 

s_mov_b32 s0, s8 // (op 450) 
s_mul_hi_i32 s51, s0, s15 // (op 450) 
s_add_i32 s18, s51, s0 // (op 450) 


s_ashr_i32 s0, s18, 31 // (op 450) 
s_cmp_eq_i32 s15, 0 // (op 450) 
s_mov_b32 s19, scc // (op 450) 
s_cmp_eq_i32 s19, 0 // (op 450) 
s_cselect_b32 s51, 0, -1 // (op 450) 

s_and_b32 s19, s48, 31 // (op 450) 
s_lshl_b32 s68, 1, s19 // (op 450) 
s_add_i32 s69, s68, s51 // (op 450) 


s_and_b32 s51, s0, s69 // (op 450) 


s_add_i32 s0, s18, s51 // (op 450) 


s_ashr_i32 s18, s0, s19 // (op 450) 


s_lshl_b32 s0, s48, 24 // (op 450) 
s_ashr_i32 s19, s0, 31 // (op 450) 

s_xor_b32 s0, s18, s19 // (op 450) 

s_sub_i32 s18, s0, s19 // (op 450) 


s_mul_i32 s0, s47, s46 // (op 450) 
s_add_u32 s68, s6, 16 // (op 450) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 450) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 450) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 450) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 450) 

s_sub_i32 s19, s68, s0 // (op 450) 

s_mul_i32 s51, s18, s19 // (op 450) 


s_sub_u32 s18, s8, s51 // (op 450) 


s_add_u32 s8, s18, s0 // (op 450) 


s_cmp_eq_i32 s9, 0 // (op 450) 
s_mov_b32 s0, scc // (op 450) 

s_cmp_eq_i32 s0, 0 // (op 450) 
s_cselect_b32 s9, s8, s1 // (op 450) 



v_lshl_add_u32 v11, s9, 4, v33 // (op 450) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 450) 
s_lshr_b32 s51, s0, 1 // (op 450) 

v_lshl_add_u32 v33, s51, 8, v10 // (op 450) 


v_mov_b32 v13, 0 // (op 450) convert
v_mov_b32 v12, v33 // (op 450) convert

v_mul_lo_u32 v15, v11, s5 // (op 450) most significant: low of low * high
v_mul_hi_u32 v10, v11, s4 // (op 450) most significant: high of low * low
v_mul_lo_u32 v14, v11, s4 // (op 450) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 450) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 450) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 450) most significant half



s_mul_i32 s1, s4, s7 // (op 452) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 452) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 452) least significant
s_mul_hi_u32 s8, s4, s6 // (op 452) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 452) most significant: sum
s_add_u32 s1, s1, s8 // (op 452) most significant: sum


s_mov_b32 s52, s20 // (op 452) reg expression
s_mov_b32 s53, s21 // (op 452) reg expression
s_mov_b32 s54, s0 // (op 452) reg expression
s_mov_b32 s55, 131072 // (op 452) call()

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
s_sub_u32 s68, s18, 1 // (op 517) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 517) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 517) 

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
s_lshl_b32 s68, 1, s8 // (op 517) 
s_add_i32 s69, s68, s9 // (op 517) 


s_and_b32 s9, s0, s69 // (op 517) 


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
s_add_i32 s68, s19, s18 // (op 517) 


s_and_b32 s18, s0, s68 // (op 517) 


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

s_ashr_i64 s[0:1], s[26:27], 7 // (op 517) 
s_lshr_b32 s51, s0, 1 // (op 517) 

v_lshl_add_u32 v33, s51, 8, v1 // (op 517) 

v_mul_lo_u32 v17, v33, s23 // (op 517) most significant: low of low * high
v_mul_hi_u32 v10, v33, s22 // (op 517) most significant: high of low * low
v_mul_lo_u32 v16, v33, s22 // (op 517) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 517) most significant: sum

v_mov_b32 v10, s8 // (op 517) 
v_mov_b32 v11, s9 // (op 517) 
v_add_co_u32 v16, s[0:1], v16, v10 // (op 517) least significant half
v_addc_co_u32 v17, s[0:1], v17, v11, s[0:1] // (op 517) most significant half





v_mul_lo_u32 v11, v17, 6 // (op 517) most significant: low of high * low
v_mul_hi_u32 v33, v16, 6 // (op 517) most significant: high of low * low
v_mul_lo_u32 v10, v16, 6 // (op 517) least significant: low of low * low
v_add_u32 v11, v11, v33 // (op 517) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 517) 

v_mov_b32 v33, v12 // (op 517) convert

s_mul_i32 s1, s27, 6 // (op 519) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 519) least significant
s_mul_hi_u32 s51, s26, 6 // (op 519) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 519) most significant: sum

s_mul_i32 s9, s22, s1 // (op 519) most significant: low of low * high
s_mul_i32 s51, s23, s0 // (op 519) most significant: low of high * low
s_mul_i32 s8, s22, s0 // (op 519) least significant
s_mul_hi_u32 s18, s22, s0 // (op 519) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 519) most significant: sum
s_add_u32 s9, s9, s18 // (op 519) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 519) 

s_mov_b32 s56, s24 // (op 519) reg expression
s_mov_b32 s57, s25 // (op 519) reg expression
s_mov_b32 s58, s0 // (op 519) reg expression
s_mov_b32 s59, 131072 // (op 519) call()

s_mul_hi_i32 s51, s16, s11 // (op 581) 
v_lshrrev_b32 v10, 7, v1 // (op 581) 
s_add_i32 s0, s51, s16 // (op 581) 

v_and_b32 v11, 15, v1 // (op 581) 
s_ashr_i32 s1, s0, 31 // (op 581) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 581) 


s_cmp_eq_i32 s11, 0 // (op 581) 
s_mov_b32 s51, scc // (op 581) 
v_and_b32 v13, 63, v1 // (op 581) 
s_cmp_eq_i32 s51, 0 // (op 581) 
s_cselect_b32 s8, 0, -1 // (op 581) 

v_lshrrev_b32 v10, 4, v13 // (op 581) 

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
s_sub_u32 s68, s18, 1 // (op 581) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 581) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 581) 

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
s_lshl_b32 s68, 1, s19 // (op 581) 
s_add_i32 s69, s68, s51 // (op 581) 


s_and_b32 s51, s0, s69 // (op 581) 


s_add_i32 s0, s18, s51 // (op 581) 


s_ashr_i32 s18, s0, s19 // (op 581) 


s_lshl_b32 s0, s48, 24 // (op 581) 
s_ashr_i32 s19, s0, 31 // (op 581) 

s_xor_b32 s0, s18, s19 // (op 581) 

s_sub_i32 s18, s0, s19 // (op 581) 


s_mul_i32 s0, s47, s46 // (op 581) 
s_add_u32 s68, s6, 16 // (op 581) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 581) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 581) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 581) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 581) 

s_sub_i32 s19, s68, s0 // (op 581) 

s_mul_i32 s51, s18, s19 // (op 581) 


s_sub_u32 s18, s8, s51 // (op 581) 


s_add_u32 s8, s18, s0 // (op 581) 


s_cmp_eq_i32 s9, 0 // (op 581) 
s_mov_b32 s0, scc // (op 581) 

s_cmp_eq_i32 s0, 0 // (op 581) 
s_cselect_b32 s9, s8, s1 // (op 581) 



v_lshl_add_u32 v11, s9, 4, v12 // (op 581) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 581) 
s_lshr_b32 s51, s0, 1 // (op 581) 

v_lshl_add_u32 v12, s51, 3, v10 // (op 581) 


v_mul_lo_u32 v10, v11, s28 // (op 581) 
v_add_u32 v10, v10, v12 // (op 581) 


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

s_mul_hi_i32 s51, s16, s11 // (op 645) 
v_and_b32 v11, -32, v1 // (op 645) 
s_add_i32 s0, s51, s16 // (op 645) 

v_sub_u32 v12, v1, v11 // (op 645) 

s_ashr_i32 s1, s0, 31 // (op 645) 
v_lshrrev_b32 v11, 5, v1 // (op 645) 
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
s_sub_u32 s68, s18, 1 // (op 645) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 645) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 645) 

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
s_lshl_b32 s68, 1, s8 // (op 645) 
s_add_i32 s69, s68, s9 // (op 645) 


s_and_b32 s9, s0, s69 // (op 645) 


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
s_add_i32 s68, s19, s18 // (op 645) 


s_and_b32 s18, s0, s68 // (op 645) 


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



v_lshl_add_u32 v13, s1, 5, v12 // (op 645) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 645) 
s_lshr_b32 s51, s0, 1 // (op 645) 

v_lshl_add_u32 v12, s51, 3, v11 // (op 645) 


v_mul_lo_u32 v11, v12, s32 // (op 645) 
v_add_u32 v11, v11, v13 // (op 645) 


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

v_and_b32 v12, -8, v1 // (op 738) 
v_sub_u32 v13, v1, v12 // (op 738) 

v_lshlrev_b32 v12, 4, v13 // (op 738) 

v_lshrrev_b32 v13, 3, v1 // (op 738) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 738) 


s_nop 0
v_readfirstlane_b32 s51, v16 // (op 738) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v12, 5, v1 // (op 790) 
v_mul_lo_u32 v13, v12, 6 // (op 790) 
v_lshrrev_b32 v16, 3, v13 // (op 790) 

v_lshrrev_b32 v13, 4, v12 // (op 790) 

v_lshlrev_b32 v12, 2, v13 // (op 790) 

v_add_u32 v13, v16, v12 // (op 790) 


v_and_b32 v12, -32, v1 // (op 843) 
v_sub_u32 v16, v1, v12 // (op 843) 

v_and_b32 v12, 4294967264, v1 // (op 843) 
v_add_u32 v17, v12, v16 // (op 843) 


s_ashr_i64 s[0:1], s[26:27], 7 // (op 162) 
s_and_b32 s8, s0, 4294967294 // (op 162) 

s_ashr_i64 s[0:1], s[26:27], 7 // (op 161) 
s_mov_b32 s9, s0 // (op 161) convert

s_cmp_lt_u32 s8, s9 // (op 161) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161:
s_barrier  // (op 1031) 
s_add_u32 s8, s8, 1 // (op 163) 


v_mov_b64 v[18:19], v[14:15] // (op 421) 

s_mov_b32 m0, s51 // (op 421) reg expression
buffer_load_dwordx4 v18, s[52:55], 0 offen offset: 0 lds // (op 421) Load value direct to lds

v_add_co_u32 v14, vcc, 128, v14 // (op 440) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 440) most significant half


v_mov_b32 v12, v33 // (op 179) 
v_mov_b32 v16, v12 // (op 179) reg expression

buffer_load_dwordx3 v[18:20], v16, s[56:59], 0 offen // (op 179) Load value
buffer_load_dwordx3 v[22:24], v16, s[56:59], 0 offen offset:12 // (op 179) Load value


s_lshl_b64 s[0:1], s[22:23], 7 // (op 509) 
s_mul_i32 s19, s1, 6 // (op 509) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 509) least significant
s_mul_hi_u32 s9, s0, 6 // (op 509) most significant: high of low * low
s_add_u32 s19, s19, s9 // (op 509) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 509) 

v_add_u32 v33, v33, s0 // (op 509) 



s_barrier  // (op 1044) 
v_mov_b32 v12, v11 // (op 180) 



v_mov_b32 v16, v12 // (op 180) reg expression

s_waitcnt vmcnt(1)
ds_write_b96 v13, v[18:20] offset:4096 // (op 177) Store local data 
buffer_load_ubyte v21, v16, s[64:67], 0 offen offset: 0 // (op 180) Load value


s_lshl_b32 s9, s32, 2 // (op 637) 
v_add_u32 v11, v11, s9 // (op 637) 

s_waitcnt vmcnt(1)
ds_write_b96 v13, v[22:24] offset:4112 // (op 177) Store local data 

s_barrier  // (op 1034) 



s_barrier  // (op 1058) 
ds_read_b128 v[22:25], v3 // (op 169) Load local data 



ds_read_b128 v[26:29], v3 offset:64 // (op 169) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v17, v21 offset:8192 // (op 178) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1048) 



ds_read_b96_tr_b6 v[34:36], v0 offset:4096 // (op 171) Transpose load local data 
ds_read_b96_tr_b6 v[37:39], v0 offset:4608 // (op 171) Transpose load local data 


v_mov_b32 v12, v10 // (op 173) 

buffer_load_ubyte v16, v12, s[60:63], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v10, 4, v10 // (op 573) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1062) 


ds_read_u8 v12, v4 offset:8192 // (op 175) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[34:39], a[0:3], v16, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 176) 




s_ashr_i64 s[0:1], s[26:27], 7 // (op 161) 
s_mov_b32 s9, s0 // (op 161) convert

s_cmp_lt_u32 s8, s9 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161 if true)
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161:






















s_barrier  // (op 1122) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

s_mul_hi_i32 s0, s16, s11 // (op 885) 
v_lshrrev_b32 v0, 7, v1 // (op 885) 
s_add_i32 s1, s0, s16 // (op 885) 

v_and_b32 v3, 63, v1 // (op 885) 
s_ashr_i32 s0, s1, 31 // (op 885) 
v_lshrrev_b32 v4, 4, v3 // (op 885) 
s_cmp_eq_i32 s11, 0 // (op 885) 
s_mov_b32 s4, scc // (op 885) 
v_lshlrev_b32 v5, 2, v4 // (op 885) 

s_cmp_eq_i32 s4, 0 // (op 885) 
s_cselect_b32 s5, 0, -1 // (op 885) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 885) 


s_and_b32 s4, s12, 31 // (op 885) 
v_lshrrev_b32 v0, 2, v3 // (op 885) 

s_lshl_b32 s8, 1, s4 // (op 885) 
v_and_b32 v3, 3, v0 // (op 885) 

s_add_i32 s9, s8, s5 // (op 885) 


v_and_b32 v0, 3, v1 // (op 885) 
s_and_b32 s5, s0, s9 // (op 885) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 885) 


s_add_i32 s0, s1, s5 // (op 885) 


v_lshrrev_b32 v0, 6, v1 // (op 885) 
s_ashr_i32 s1, s0, s4 // (op 885) 


v_and_b32 v3, 1, v0 // (op 885) 

s_lshl_b32 s0, s12, 24 // (op 885) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 885) 


s_ashr_i32 s4, s0, 31 // (op 885) 

s_xor_b32 s0, s1, s4 // (op 885) 

s_sub_i32 s1, s0, s4 // (op 885) 


s_mul_hi_i32 s0, s1, s13 // (op 885) 
s_add_i32 s4, s0, s1 // (op 885) 

s_ashr_i32 s0, s4, 31 // (op 885) 
s_cmp_eq_i32 s13, 0 // (op 885) 
s_mov_b32 s5, scc // (op 885) 
s_cmp_eq_i32 s5, 0 // (op 885) 
s_cselect_b32 s8, 0, -1 // (op 885) 

s_and_b32 s5, s14, 31 // (op 885) 
s_lshl_b32 s9, 1, s5 // (op 885) 
s_add_i32 s18, s9, s8 // (op 885) 


s_and_b32 s8, s0, s18 // (op 885) 


s_add_i32 s0, s4, s8 // (op 885) 


s_ashr_i32 s4, s0, s5 // (op 885) 


s_lshl_b32 s0, s14, 24 // (op 885) 
s_ashr_i32 s5, s0, 31 // (op 885) 

s_xor_b32 s0, s4, s5 // (op 885) 

s_sub_i32 s4, s0, s5 // (op 885) 


s_mul_i32 s0, s1, s46 // (op 885) 
s_sub_u32 s5, s16, s0 // (op 885) 

s_mul_i32 s0, s4, s46 // (op 885) 
s_add_u32 s0, s0, s5 // (op 885) 

s_mul_hi_i32 s5, s16, s10 // (op 885) 
s_add_i32 s8, s5, s16 // (op 885) 

s_ashr_i32 s5, s8, 31 // (op 885) 
s_cmp_eq_i32 s10, 0 // (op 885) 
s_mov_b32 s9, scc // (op 885) 
s_cmp_eq_i32 s9, 0 // (op 885) 
s_cselect_b32 s18, 0, -1 // (op 885) 

s_and_b32 s9, s49, 31 // (op 885) 
s_lshl_b32 s19, 1, s9 // (op 885) 
s_add_i32 s20, s19, s18 // (op 885) 


s_and_b32 s18, s5, s20 // (op 885) 


s_add_i32 s5, s8, s18 // (op 885) 


s_ashr_i32 s8, s5, s9 // (op 885) 


s_lshl_b32 s5, s49, 24 // (op 885) 
s_ashr_i32 s9, s5, 31 // (op 885) 

s_xor_b32 s5, s8, s9 // (op 885) 

s_sub_i32 s8, s5, s9 // (op 885) 


s_add_u32 s18, s2, 32 // (op 885) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 885) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 885) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 885) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 885) 

s_mul_i32 s5, s46, s18 // (op 885) 
s_mul_i32 s9, s50, s5 // (op 885) 

s_mul_i32 s5, s8, s9 // (op 885) 

s_sub_u32 s9, s16, s5 // (op 885) 

s_mov_b32 s5, s9 // (op 885) 
s_mul_hi_i32 s20, s5, s15 // (op 885) 
s_add_i32 s21, s20, s5 // (op 885) 


s_ashr_i32 s5, s21, 31 // (op 885) 
s_cmp_eq_i32 s15, 0 // (op 885) 
s_mov_b32 s20, scc // (op 885) 
s_cmp_eq_i32 s20, 0 // (op 885) 
s_cselect_b32 s22, 0, -1 // (op 885) 

s_and_b32 s20, s48, 31 // (op 885) 
s_lshl_b32 s23, 1, s20 // (op 885) 
s_add_i32 s24, s23, s22 // (op 885) 


s_and_b32 s22, s5, s24 // (op 885) 


s_add_i32 s5, s21, s22 // (op 885) 


s_ashr_i32 s21, s5, s20 // (op 885) 


s_lshl_b32 s5, s48, 24 // (op 885) 
s_ashr_i32 s20, s5, 31 // (op 885) 

s_xor_b32 s5, s21, s20 // (op 885) 

s_sub_i32 s21, s5, s20 // (op 885) 


s_mul_i32 s5, s47, s46 // (op 885) 
s_add_u32 s22, s6, 16 // (op 885) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 885) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 885) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 885) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 885) 

s_sub_i32 s20, s22, s5 // (op 885) 

s_mul_i32 s22, s21, s20 // (op 885) 

s_sub_u32 s20, s9, s22 // (op 885) 


s_add_u32 s9, s20, s5 // (op 885) 


s_cmp_eq_i32 s8, 0 // (op 885) 
s_mov_b32 s5, scc // (op 885) 

s_cmp_eq_i32 s5, 0 // (op 885) 
s_cselect_b32 s8, s9, s0 // (op 885) 


s_mul_i32 s0, s4, s18 // (op 885) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 885) 


s_sub_i32 s4, s1, s0 // (op 885) 


s_cmp_eq_i32 s5, 0 // (op 885) 
s_cselect_b32 s0, s21, s4 // (op 885) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 885) 


v_mul_lo_u32 v0, v4, s38 // (op 885) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 885) 


s_mul_i32 s1, s38, s3 // (op 887) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 887) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 887) least significant
s_mul_hi_u32 s5, s38, s2 // (op 887) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 887) most significant: sum
s_add_u32 s1, s1, s5 // (op 887) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 887) 

s_mov_b32 s20, s40 // (op 887) reg expression
s_mov_b32 s21, s41 // (op 887) reg expression
s_mov_b32 s22, s4 // (op 887) reg expression
s_mov_b32 s23, 131072 // (op 887) call()




buffer_store_dwordx4 v[10:13], v4, s[20:23], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123:
s_mul_hi_i32 s0, s16, s11 // (op 708) 
v_lshrrev_b32 v0, 7, v1 // (op 708) 
s_add_i32 s1, s0, s16 // (op 708) 

v_and_b32 v3, 63, v1 // (op 708) 
s_ashr_i32 s0, s1, 31 // (op 708) 
v_lshrrev_b32 v4, 4, v3 // (op 708) 
s_cmp_eq_i32 s11, 0 // (op 708) 
s_mov_b32 s4, scc // (op 708) 
v_lshlrev_b32 v5, 2, v4 // (op 708) 

s_cmp_eq_i32 s4, 0 // (op 708) 
s_cselect_b32 s5, 0, -1 // (op 708) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 708) 


s_and_b32 s4, s12, 31 // (op 708) 
v_lshrrev_b32 v0, 2, v3 // (op 708) 

s_lshl_b32 s8, 1, s4 // (op 708) 
v_and_b32 v3, 3, v0 // (op 708) 

s_add_i32 s9, s8, s5 // (op 708) 


v_and_b32 v0, 3, v1 // (op 708) 
s_and_b32 s5, s0, s9 // (op 708) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 708) 


s_add_i32 s0, s1, s5 // (op 708) 


v_lshrrev_b32 v0, 6, v1 // (op 708) 
s_ashr_i32 s1, s0, s4 // (op 708) 


v_and_b32 v3, 1, v0 // (op 708) 

s_lshl_b32 s0, s12, 24 // (op 708) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 708) 


s_ashr_i32 s4, s0, 31 // (op 708) 

s_xor_b32 s0, s1, s4 // (op 708) 

s_sub_i32 s1, s0, s4 // (op 708) 


s_mul_hi_i32 s0, s1, s13 // (op 708) 
s_add_i32 s4, s0, s1 // (op 708) 

s_ashr_i32 s0, s4, 31 // (op 708) 
s_cmp_eq_i32 s13, 0 // (op 708) 
s_mov_b32 s5, scc // (op 708) 
s_cmp_eq_i32 s5, 0 // (op 708) 
s_cselect_b32 s8, 0, -1 // (op 708) 

s_and_b32 s5, s14, 31 // (op 708) 
s_lshl_b32 s9, 1, s5 // (op 708) 
s_add_i32 s18, s9, s8 // (op 708) 


s_and_b32 s19, s0, s18 // (op 708) 


s_add_i32 s0, s4, s19 // (op 708) 


s_ashr_i32 s4, s0, s5 // (op 708) 


s_lshl_b32 s0, s14, 24 // (op 708) 
s_ashr_i32 s5, s0, 31 // (op 708) 

s_xor_b32 s0, s4, s5 // (op 708) 

s_sub_i32 s4, s0, s5 // (op 708) 


s_mul_i32 s0, s1, s46 // (op 708) 
s_sub_u32 s5, s16, s0 // (op 708) 

s_mul_i32 s0, s4, s46 // (op 708) 
s_add_u32 s0, s0, s5 // (op 708) 

s_mul_hi_i32 s5, s16, s10 // (op 708) 
s_add_i32 s8, s5, s16 // (op 708) 

s_ashr_i32 s5, s8, 31 // (op 708) 
s_cmp_eq_i32 s10, 0 // (op 708) 
s_mov_b32 s9, scc // (op 708) 
s_cmp_eq_i32 s9, 0 // (op 708) 
s_cselect_b32 s18, 0, -1 // (op 708) 

s_and_b32 s9, s49, 31 // (op 708) 
s_lshl_b32 s19, 1, s9 // (op 708) 
s_add_i32 s24, s19, s18 // (op 708) 


s_and_b32 s18, s5, s24 // (op 708) 


s_add_i32 s5, s8, s18 // (op 708) 


s_ashr_i32 s8, s5, s9 // (op 708) 


s_lshl_b32 s5, s49, 24 // (op 708) 
s_ashr_i32 s9, s5, 31 // (op 708) 

s_xor_b32 s5, s8, s9 // (op 708) 

s_sub_i32 s8, s5, s9 // (op 708) 


s_add_u32 s18, s2, 32 // (op 708) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 708) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 708) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 708) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 5 // (op 708) 

s_mul_i32 s5, s46, s18 // (op 708) 
s_mul_i32 s9, s50, s5 // (op 708) 

s_mul_i32 s5, s8, s9 // (op 708) 

s_sub_u32 s9, s16, s5 // (op 708) 

s_mov_b32 s5, s9 // (op 708) 
s_mul_hi_i32 s24, s5, s15 // (op 708) 
s_add_i32 s25, s24, s5 // (op 708) 


s_ashr_i32 s5, s25, 31 // (op 708) 
s_cmp_eq_i32 s15, 0 // (op 708) 
s_mov_b32 s24, scc // (op 708) 
s_cmp_eq_i32 s24, 0 // (op 708) 
s_cselect_b32 s26, 0, -1 // (op 708) 

s_and_b32 s24, s48, 31 // (op 708) 
s_lshl_b32 s27, 1, s24 // (op 708) 
s_add_i32 s28, s27, s26 // (op 708) 


s_and_b32 s26, s5, s28 // (op 708) 


s_add_i32 s5, s25, s26 // (op 708) 


s_ashr_i32 s25, s5, s24 // (op 708) 


s_lshl_b32 s5, s48, 24 // (op 708) 
s_ashr_i32 s24, s5, 31 // (op 708) 

s_xor_b32 s5, s25, s24 // (op 708) 

s_sub_i32 s25, s5, s24 // (op 708) 


s_mul_i32 s5, s47, s46 // (op 708) 
s_add_u32 s26, s6, 16 // (op 708) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 708) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 708) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 708) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 708) 

s_sub_i32 s24, s26, s5 // (op 708) 

s_mul_i32 s26, s25, s24 // (op 708) 

s_sub_u32 s24, s9, s26 // (op 708) 


s_add_u32 s9, s24, s5 // (op 708) 


s_cmp_eq_i32 s8, 0 // (op 708) 
s_mov_b32 s5, scc // (op 708) 

s_cmp_eq_i32 s5, 0 // (op 708) 
s_cselect_b32 s8, s9, s0 // (op 708) 


s_mul_i32 s0, s4, s18 // (op 708) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 708) 


s_sub_i32 s4, s1, s0 // (op 708) 


s_cmp_eq_i32 s5, 0 // (op 708) 
s_cselect_b32 s24, s25, s4 // (op 708) 



v_lshl_add_u32 v4, s24, 5, v0 // (op 708) 


v_mul_lo_u32 v0, v4, s42 // (op 708) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 708) 


s_mul_i32 s1, s42, s3 // (op 710) most significant: low of low * high
s_mul_i32 s4, s43, s2 // (op 710) most significant: low of high * low
s_mul_i32 s0, s42, s2 // (op 710) least significant
s_mul_hi_u32 s5, s42, s2 // (op 710) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 710) most significant: sum
s_add_u32 s1, s1, s5 // (op 710) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 710) 

s_mov_b32 s24, s44 // (op 710) reg expression
s_mov_b32 s25, s45 // (op 710) reg expression
s_mov_b32 s26, s4 // (op 710) reg expression
s_mov_b32 s27, 131072 // (op 710) call()





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


s_mul_hi_i32 s0, s16, s11 // (op 902) 
v_lshrrev_b32 v0, 7, v1 // (op 902) 
s_add_i32 s1, s0, s16 // (op 902) 

v_and_b32 v3, 63, v1 // (op 902) 
s_ashr_i32 s0, s1, 31 // (op 902) 
v_lshrrev_b32 v4, 4, v3 // (op 902) 
s_cmp_eq_i32 s11, 0 // (op 902) 
s_mov_b32 s4, scc // (op 902) 
v_lshlrev_b32 v5, 2, v4 // (op 902) 

s_cmp_eq_i32 s4, 0 // (op 902) 
s_cselect_b32 s5, 0, -1 // (op 902) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 902) 


s_and_b32 s4, s12, 31 // (op 902) 
v_lshrrev_b32 v0, 2, v3 // (op 902) 

s_lshl_b32 s8, 1, s4 // (op 902) 
v_and_b32 v3, 3, v0 // (op 902) 

s_add_i32 s9, s8, s5 // (op 902) 


v_and_b32 v0, 3, v1 // (op 902) 
s_and_b32 s5, s0, s9 // (op 902) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 902) 


s_add_i32 s0, s1, s5 // (op 902) 


v_lshrrev_b32 v0, 6, v1 // (op 902) 
s_ashr_i32 s1, s0, s4 // (op 902) 


v_and_b32 v3, 1, v0 // (op 902) 

s_lshl_b32 s0, s12, 24 // (op 902) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 902) 


s_ashr_i32 s4, s0, 31 // (op 902) 

s_xor_b32 s0, s1, s4 // (op 902) 

s_sub_i32 s1, s0, s4 // (op 902) 


s_mul_hi_i32 s0, s1, s13 // (op 902) 
s_add_i32 s4, s0, s1 // (op 902) 

s_ashr_i32 s0, s4, 31 // (op 902) 
s_cmp_eq_i32 s13, 0 // (op 902) 
s_mov_b32 s5, scc // (op 902) 
s_cmp_eq_i32 s5, 0 // (op 902) 
s_cselect_b32 s8, 0, -1 // (op 902) 

s_and_b32 s5, s14, 31 // (op 902) 
s_lshl_b32 s9, 1, s5 // (op 902) 
s_add_i32 s18, s9, s8 // (op 902) 


s_and_b32 s19, s0, s18 // (op 902) 


s_add_i32 s0, s4, s19 // (op 902) 


s_ashr_i32 s4, s0, s5 // (op 902) 


s_lshl_b32 s0, s14, 24 // (op 902) 
s_ashr_i32 s5, s0, 31 // (op 902) 

s_xor_b32 s0, s4, s5 // (op 902) 

s_sub_i32 s4, s0, s5 // (op 902) 


s_mul_i32 s0, s1, s46 // (op 902) 
s_sub_u32 s5, s16, s0 // (op 902) 

s_mul_i32 s0, s4, s46 // (op 902) 
s_add_u32 s0, s0, s5 // (op 902) 

s_mul_hi_i32 s5, s16, s10 // (op 902) 
s_add_i32 s8, s5, s16 // (op 902) 

s_ashr_i32 s5, s8, 31 // (op 902) 
s_cmp_eq_i32 s10, 0 // (op 902) 
s_mov_b32 s9, scc // (op 902) 
s_cmp_eq_i32 s9, 0 // (op 902) 
s_cselect_b32 s18, 0, -1 // (op 902) 

s_and_b32 s9, s49, 31 // (op 902) 
s_lshl_b32 s19, 1, s9 // (op 902) 
s_add_i32 s24, s19, s18 // (op 902) 


s_and_b32 s18, s5, s24 // (op 902) 


s_add_i32 s5, s8, s18 // (op 902) 


s_ashr_i32 s8, s5, s9 // (op 902) 


s_lshl_b32 s5, s49, 24 // (op 902) 
s_ashr_i32 s9, s5, 31 // (op 902) 

s_xor_b32 s5, s8, s9 // (op 902) 

s_sub_i32 s8, s5, s9 // (op 902) 


s_add_u32 s18, s2, 32 // (op 902) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 902) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 902) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 902) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 5 // (op 902) 

s_mul_i32 s5, s46, s18 // (op 902) 
s_mul_i32 s9, s50, s5 // (op 902) 

s_mul_i32 s5, s8, s9 // (op 902) 

s_sub_u32 s9, s16, s5 // (op 902) 

s_mov_b32 s5, s9 // (op 902) 
s_mul_hi_i32 s24, s5, s15 // (op 902) 
s_add_i32 s25, s24, s5 // (op 902) 


s_ashr_i32 s5, s25, 31 // (op 902) 
s_cmp_eq_i32 s15, 0 // (op 902) 
s_mov_b32 s24, scc // (op 902) 
s_cmp_eq_i32 s24, 0 // (op 902) 
s_cselect_b32 s26, 0, -1 // (op 902) 

s_and_b32 s24, s48, 31 // (op 902) 
s_lshl_b32 s27, 1, s24 // (op 902) 
s_add_i32 s28, s27, s26 // (op 902) 


s_and_b32 s26, s5, s28 // (op 902) 


s_add_i32 s5, s25, s26 // (op 902) 


s_ashr_i32 s25, s5, s24 // (op 902) 


s_lshl_b32 s5, s48, 24 // (op 902) 
s_ashr_i32 s24, s5, 31 // (op 902) 

s_xor_b32 s5, s25, s24 // (op 902) 

s_sub_i32 s25, s5, s24 // (op 902) 


s_mul_i32 s5, s47, s46 // (op 902) 
s_add_u32 s26, s6, 16 // (op 902) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 902) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 902) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 902) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 902) 

s_sub_i32 s24, s26, s5 // (op 902) 

s_mul_i32 s26, s25, s24 // (op 902) 

s_sub_u32 s24, s9, s26 // (op 902) 


s_add_u32 s9, s24, s5 // (op 902) 


s_cmp_eq_i32 s8, 0 // (op 902) 
s_mov_b32 s5, scc // (op 902) 

s_cmp_eq_i32 s5, 0 // (op 902) 
s_cselect_b32 s8, s9, s0 // (op 902) 


s_mul_i32 s0, s4, s18 // (op 902) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 902) 


s_sub_i32 s4, s1, s0 // (op 902) 


s_cmp_eq_i32 s5, 0 // (op 902) 
s_cselect_b32 s24, s25, s4 // (op 902) 



v_lshl_add_u32 v4, s24, 5, v0 // (op 902) 


v_mul_lo_u32 v0, v4, s38 // (op 902) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 902) 


s_mul_i32 s1, s38, s3 // (op 904) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 904) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 904) least significant
s_mul_hi_u32 s5, s38, s2 // (op 904) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 904) most significant: sum
s_add_u32 s1, s1, s5 // (op 904) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 904) 

s_mov_b32 s20, s40 // (op 904) reg expression
s_mov_b32 s21, s41 // (op 904) reg expression
s_mov_b32 s22, s4 // (op 904) reg expression
s_mov_b32 s23, 131072 // (op 904) call()




buffer_store_dwordx4 v[10:13], v4, s[20:23], 0 offen // (op 134) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.LRR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end:
.size RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_, .LRR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end-RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 60
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12544 // lds bytes
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: F6
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
  - .name: RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 12544
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 55
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

