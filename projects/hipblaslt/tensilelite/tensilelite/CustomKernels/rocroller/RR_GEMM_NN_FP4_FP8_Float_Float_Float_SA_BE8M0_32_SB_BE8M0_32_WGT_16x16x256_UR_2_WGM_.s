// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_,@function
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 63, v1 // (op 849) 
v_and_b32 v3, 48, v1 // (op 849) 
v_sub_u32 v4, v0, v3 // (op 849) 

v_lshrrev_b32 v3, 4, v0 // (op 849) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 849) 


v_lshrrev_b32 v3, 6, v1 // (op 849) 
v_lshlrev_b32 v4, 4, v3 // (op 849) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 849) 


v_lshlrev_b32 v0, 2, v3 // (op 849) 

v_lshrrev_b32 v3, 3, v0 // (op 849) 

v_and_b32 v0, 15, v1 // (op 902) 
v_and_b32 v4, 48, v1 // (op 902) 
v_lshl_add_u32 v5, v0, 8, v4 // (op 902) 


v_lshrrev_b32 v0, 6, v1 // (op 955) 
v_and_b32 v4, 15, v1 // (op 955) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 955) 


v_and_b32 v0, 48, v1 // (op 955) 
v_add_u32 v4, v0, v6 // (op 955) 


v_and_b32 v0, 15, v1 // (op 1008) 
v_and_b32 v6, 63, v1 // (op 1008) 
v_lshrrev_b32 v7, 4, v6 // (op 1008) 

v_lshl_add_u32 v6, v0, 3, v7 // (op 1008) 


v_and_b32 v0, 63, v1 // (op 1066) 
v_and_b32 v7, 48, v1 // (op 1066) 
v_sub_u32 v8, v0, v7 // (op 1066) 

v_lshrrev_b32 v7, 4, v0 // (op 1066) 

v_lshl_add_u32 v0, v7, 5, v8 // (op 1066) 


v_lshrrev_b32 v7, 6, v1 // (op 1066) 
v_lshlrev_b32 v8, 4, v7 // (op 1066) 

v_lshl_add_u32 v7, v0, 4, v8 // (op 1066) 


v_lshlrev_b32 v0, 2, v7 // (op 1066) 

v_lshrrev_b32 v7, 3, v0 // (op 1066) 

v_and_b32 v0, 15, v1 // (op 1090) 
v_and_b32 v8, 48, v1 // (op 1090) 
v_lshl_add_u32 v9, v0, 8, v8 // (op 1090) 


v_lshrrev_b32 v0, 6, v1 // (op 1114) 
v_and_b32 v8, 15, v1 // (op 1114) 
v_lshl_add_u32 v10, v0, 4, v8 // (op 1114) 


v_and_b32 v0, 48, v1 // (op 1114) 
v_add_u32 v8, v0, v10 // (op 1114) 


v_and_b32 v0, 15, v1 // (op 1138) 
v_and_b32 v10, 63, v1 // (op 1138) 
v_lshrrev_b32 v11, 4, v10 // (op 1138) 

v_lshl_add_u32 v10, v0, 3, v11 // (op 1138) 


v_mov_b32 v0, s6 // (op 2) Move value
v_mov_b32 v11, s7 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_mul_hi_i32 s0, s16, s9 // (op 559) 
v_lshlrev_b32 v12, 2, v1 // (op 559) 
s_add_i32 s1, s0, s16 // (op 559) 

s_ashr_i32 s0, s1, 31 // (op 559) 
s_cmp_eq_i32 s9, 0 // (op 559) 
s_mov_b32 s6, scc // (op 559) 
s_cmp_eq_i32 s6, 0 // (op 559) 
s_cselect_b32 s7, 0, -1 // (op 559) 

s_and_b32 s6, s10, 31 // (op 559) 
s_lshl_b32 s18, 1, s6 // (op 559) 
s_add_i32 s19, s18, s7 // (op 559) 


s_and_b32 s7, s0, s19 // (op 559) 


s_add_i32 s0, s1, s7 // (op 559) 


s_ashr_i32 s1, s0, s6 // (op 559) 


s_lshl_b32 s0, s10, 24 // (op 559) 
s_ashr_i32 s6, s0, 31 // (op 559) 

s_xor_b32 s0, s1, s6 // (op 559) 

s_sub_i32 s1, s0, s6 // (op 559) 


s_mul_hi_i32 s0, s1, s11 // (op 559) 
s_add_i32 s6, s0, s1 // (op 559) 

s_ashr_i32 s0, s6, 31 // (op 559) 
s_cmp_eq_i32 s11, 0 // (op 559) 
s_mov_b32 s7, scc // (op 559) 
s_cmp_eq_i32 s7, 0 // (op 559) 
s_cselect_b32 s18, 0, -1 // (op 559) 

s_and_b32 s7, s12, 31 // (op 559) 
s_lshl_b32 s19, 1, s7 // (op 559) 
s_add_i32 s51, s19, s18 // (op 559) 


s_and_b32 s18, s0, s51 // (op 559) 


s_add_i32 s0, s6, s18 // (op 559) 


s_ashr_i32 s6, s0, s7 // (op 559) 


s_lshl_b32 s0, s12, 24 // (op 559) 
s_ashr_i32 s7, s0, 31 // (op 559) 

s_xor_b32 s0, s6, s7 // (op 559) 

s_sub_i32 s6, s0, s7 // (op 559) 


s_mul_i32 s0, s1, s14 // (op 559) 

s_sub_u32 s1, s16, s0 // (op 559) 

s_mul_i32 s0, s6, s14 // (op 559) 
s_add_u32 s0, s0, s1 // (op 559) 


s_mul_hi_i32 s1, s16, s8 // (op 559) 
s_add_i32 s6, s1, s16 // (op 559) 

s_ashr_i32 s1, s6, 31 // (op 559) 
s_cmp_eq_i32 s8, 0 // (op 559) 
s_mov_b32 s7, scc // (op 559) 
s_cmp_eq_i32 s7, 0 // (op 559) 
s_cselect_b32 s18, 0, -1 // (op 559) 

s_and_b32 s7, s15, 31 // (op 559) 
s_lshl_b32 s19, 1, s7 // (op 559) 
s_add_i32 s51, s19, s18 // (op 559) 


s_and_b32 s18, s1, s51 // (op 559) 


s_add_i32 s1, s6, s18 // (op 559) 


s_ashr_i32 s6, s1, s7 // (op 559) 


s_lshl_b32 s1, s15, 24 // (op 559) 
s_ashr_i32 s7, s1, 31 // (op 559) 

s_xor_b32 s1, s6, s7 // (op 559) 

s_sub_i32 s6, s1, s7 // (op 559) 


s_add_u32 s18, s4, 16 // (op 559) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 559) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 559) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 559) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 559) 

s_mul_i32 s1, s14, s18 // (op 559) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 559) 

s_mul_i32 s1, s6, s7 // (op 559) 

s_sub_u32 s7, s16, s1 // (op 559) 

s_mov_b32 s1, s7 // (op 559) 
s_mul_hi_i32 s18, s1, s13 // (op 559) 
s_add_i32 s19, s18, s1 // (op 559) 


s_ashr_i32 s1, s19, 31 // (op 559) 
s_cmp_eq_i32 s13, 0 // (op 559) 
s_mov_b32 s18, scc // (op 559) 
s_cmp_eq_i32 s18, 0 // (op 559) 
s_cselect_b32 s51, 0, -1 // (op 559) 

s_and_b32 s18, s49, 31 // (op 559) 
s_lshl_b32 s52, 1, s18 // (op 559) 
s_add_i32 s53, s52, s51 // (op 559) 


s_and_b32 s51, s1, s53 // (op 559) 


s_add_i32 s1, s19, s51 // (op 559) 


s_ashr_i32 s19, s1, s18 // (op 559) 


s_lshl_b32 s1, s49, 24 // (op 559) 
s_ashr_i32 s18, s1, 31 // (op 559) 

s_xor_b32 s1, s19, s18 // (op 559) 

s_sub_i32 s19, s1, s18 // (op 559) 


s_mul_i32 s1, s48, s14 // (op 559) 
s_add_u32 s52, s20, 16 // (op 559) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 559) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 559) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 559) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 559) 

s_sub_i32 s18, s52, s1 // (op 559) 

s_mul_i32 s51, s19, s18 // (op 559) 


s_sub_u32 s18, s7, s51 // (op 559) 


s_add_u32 s7, s18, s1 // (op 559) 


s_cmp_eq_i32 s6, 0 // (op 559) 
s_mov_b32 s1, scc // (op 559) 

s_cmp_eq_i32 s1, 0 // (op 559) 
s_cselect_b32 s6, s7, s0 // (op 559) 



s_lshl_b32 s7, s6, 4 // (op 559) 

s_mov_b32 s1, 0 // (op 559) convert
s_mov_b32 s0, s7 // (op 559) convert

v_mul_lo_u32 v15, v12, s3 // (op 559) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 559) most significant: high of low * low
v_mul_lo_u32 v14, v12, s2 // (op 559) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 559) most significant: sum


v_mov_b32 v12, s0 // (op 559) 
v_mov_b32 v13, s1 // (op 559) 
v_add_co_u32 v16, s[6:7], v12, v14 // (op 559) least significant half
v_addc_co_u32 v17, s[6:7], v13, v15, s[6:7] // (op 559) most significant half



v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 559) 


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 559) 

v_mov_b32 v14, v12 // (op 559) convert

s_mul_i32 s1, s2, s25 // (op 561) most significant: low of low * high
s_mul_i32 s6, s3, s24 // (op 561) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 561) least significant
s_mul_hi_u32 s7, s2, s24 // (op 561) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 561) most significant: sum
s_add_u32 s1, s1, s7 // (op 561) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 561) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 561) 

s_mov_b32 s52, s22 // (op 561) reg expression
s_mov_b32 s53, s23 // (op 561) reg expression
s_mov_b32 s54, s0 // (op 561) reg expression
s_mov_b32 s55, 131072 // (op 561) call()

s_mul_hi_i32 s51, s16, s9 // (op 578) 
v_lshlrev_b32 v12, 2, v1 // (op 578) 
s_add_i32 s0, s51, s16 // (op 578) 

s_ashr_i32 s1, s0, 31 // (op 578) 
s_cmp_eq_i32 s9, 0 // (op 578) 
s_mov_b32 s51, scc // (op 578) 
s_cmp_eq_i32 s51, 0 // (op 578) 
s_cselect_b32 s6, 0, -1 // (op 578) 

s_and_b32 s7, s10, 31 // (op 578) 
s_lshl_b32 s51, 1, s7 // (op 578) 
s_add_i32 s18, s51, s6 // (op 578) 


s_and_b32 s6, s1, s18 // (op 578) 


s_add_i32 s1, s0, s6 // (op 578) 


s_ashr_i32 s0, s1, s7 // (op 578) 


s_lshl_b32 s1, s10, 24 // (op 578) 
s_ashr_i32 s51, s1, 31 // (op 578) 

s_xor_b32 s1, s0, s51 // (op 578) 

s_sub_i32 s0, s1, s51 // (op 578) 


s_mul_hi_i32 s1, s0, s11 // (op 578) 
s_add_i32 s51, s1, s0 // (op 578) 

s_ashr_i32 s1, s51, 31 // (op 578) 
s_cmp_eq_i32 s11, 0 // (op 578) 
s_mov_b32 s6, scc // (op 578) 
s_cmp_eq_i32 s6, 0 // (op 578) 
s_cselect_b32 s7, 0, -1 // (op 578) 

s_and_b32 s6, s12, 31 // (op 578) 
s_lshl_b32 s18, 1, s6 // (op 578) 
s_add_i32 s19, s18, s7 // (op 578) 


s_and_b32 s7, s1, s19 // (op 578) 


s_add_i32 s1, s51, s7 // (op 578) 


s_ashr_i32 s7, s1, s6 // (op 578) 


s_lshl_b32 s1, s12, 24 // (op 578) 
s_ashr_i32 s6, s1, 31 // (op 578) 

s_xor_b32 s1, s7, s6 // (op 578) 

s_sub_i32 s7, s1, s6 // (op 578) 


s_mul_i32 s1, s0, s14 // (op 578) 

s_sub_u32 s0, s16, s1 // (op 578) 

s_mul_i32 s1, s7, s14 // (op 578) 
s_add_u32 s1, s1, s0 // (op 578) 


s_mul_hi_i32 s0, s16, s8 // (op 578) 
s_add_i32 s51, s0, s16 // (op 578) 

s_ashr_i32 s0, s51, 31 // (op 578) 
s_cmp_eq_i32 s8, 0 // (op 578) 
s_mov_b32 s6, scc // (op 578) 
s_cmp_eq_i32 s6, 0 // (op 578) 
s_cselect_b32 s7, 0, -1 // (op 578) 

s_and_b32 s6, s15, 31 // (op 578) 
s_lshl_b32 s18, 1, s6 // (op 578) 
s_add_i32 s19, s18, s7 // (op 578) 


s_and_b32 s7, s0, s19 // (op 578) 


s_add_i32 s0, s51, s7 // (op 578) 


s_ashr_i32 s7, s0, s6 // (op 578) 


s_lshl_b32 s0, s15, 24 // (op 578) 
s_ashr_i32 s6, s0, 31 // (op 578) 

s_xor_b32 s0, s7, s6 // (op 578) 

s_sub_i32 s7, s0, s6 // (op 578) 


s_add_u32 s18, s4, 16 // (op 578) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 578) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 578) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 578) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 578) 

s_mul_i32 s0, s14, s18 // (op 578) 

s_mul_i32 s6, s50, s0 // (op 578) 

s_mul_i32 s0, s7, s6 // (op 578) 

s_sub_u32 s6, s16, s0 // (op 578) 

s_mov_b32 s0, s6 // (op 578) 
s_mul_hi_i32 s51, s0, s13 // (op 578) 
s_add_i32 s18, s51, s0 // (op 578) 


s_ashr_i32 s0, s18, 31 // (op 578) 
s_cmp_eq_i32 s13, 0 // (op 578) 
s_mov_b32 s19, scc // (op 578) 
s_cmp_eq_i32 s19, 0 // (op 578) 
s_cselect_b32 s51, 0, -1 // (op 578) 

s_and_b32 s19, s49, 31 // (op 578) 
s_lshl_b32 s56, 1, s19 // (op 578) 
s_add_i32 s57, s56, s51 // (op 578) 


s_and_b32 s51, s0, s57 // (op 578) 


s_add_i32 s0, s18, s51 // (op 578) 


s_ashr_i32 s18, s0, s19 // (op 578) 


s_lshl_b32 s0, s49, 24 // (op 578) 
s_ashr_i32 s19, s0, 31 // (op 578) 

s_xor_b32 s0, s18, s19 // (op 578) 

s_sub_i32 s18, s0, s19 // (op 578) 


s_mul_i32 s0, s48, s14 // (op 578) 
s_add_u32 s56, s20, 16 // (op 578) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 578) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 578) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 578) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 578) 

s_sub_i32 s19, s56, s0 // (op 578) 

s_mul_i32 s51, s18, s19 // (op 578) 


s_sub_u32 s18, s6, s51 // (op 578) 


s_add_u32 s6, s18, s0 // (op 578) 


s_cmp_eq_i32 s7, 0 // (op 578) 
s_mov_b32 s0, scc // (op 578) 

s_cmp_eq_i32 s0, 0 // (op 578) 
s_cselect_b32 s7, s6, s1 // (op 578) 



s_lshl_b32 s6, s7, 4 // (op 578) 

s_mov_b32 s1, 0 // (op 578) convert
s_mov_b32 s0, s6 // (op 578) convert

v_mul_lo_u32 v17, v12, s3 // (op 578) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 578) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 578) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 578) most significant: sum


v_mov_b32 v15, s0 // (op 578) 
v_mov_b32 v12, s1 // (op 578) 
v_add_co_u32 v18, s[6:7], v15, v16 // (op 578) least significant half
v_addc_co_u32 v19, s[6:7], v12, v17, s[6:7] // (op 578) most significant half



v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 578) 


v_lshrrev_b64 v[12:13], 3, v[18:19] // (op 578) 

v_mov_b32 v15, v12 // (op 578) convert

s_mul_i32 s1, s2, s25 // (op 580) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 580) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 580) least significant
s_mul_hi_u32 s6, s2, s24 // (op 580) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 580) most significant: sum
s_add_u32 s1, s1, s6 // (op 580) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 580) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 580) 

s_mov_b32 s52, s22 // (op 580) reg expression
s_mov_b32 s53, s23 // (op 580) reg expression
s_mov_b32 s54, s0 // (op 580) reg expression
s_mov_b32 s55, 131072 // (op 580) call()

s_mul_hi_i32 s51, s16, s9 // (op 623) 
v_lshrrev_b32 v12, 4, v1 // (op 623) 
s_add_i32 s0, s51, s16 // (op 623) 

v_lshlrev_b32 v13, 2, v12 // (op 623) 

s_ashr_i32 s1, s0, 31 // (op 623) 
v_and_b32 v12, -16, v1 // (op 623) 
s_cmp_eq_i32 s9, 0 // (op 623) 
s_mov_b32 s51, scc // (op 623) 
v_sub_u32 v16, v1, v12 // (op 623) 

s_cmp_eq_i32 s51, 0 // (op 623) 
s_cselect_b32 s6, 0, -1 // (op 623) 

s_and_b32 s7, s10, 31 // (op 623) 
s_lshl_b32 s51, 1, s7 // (op 623) 
s_add_i32 s18, s51, s6 // (op 623) 


s_and_b32 s6, s1, s18 // (op 623) 


s_add_i32 s1, s0, s6 // (op 623) 


s_ashr_i32 s0, s1, s7 // (op 623) 


s_lshl_b32 s1, s10, 24 // (op 623) 
s_ashr_i32 s51, s1, 31 // (op 623) 

s_xor_b32 s1, s0, s51 // (op 623) 

s_sub_i32 s0, s1, s51 // (op 623) 


s_mul_hi_i32 s1, s0, s11 // (op 623) 
s_add_i32 s51, s1, s0 // (op 623) 

s_ashr_i32 s1, s51, 31 // (op 623) 
s_cmp_eq_i32 s11, 0 // (op 623) 
s_mov_b32 s6, scc // (op 623) 
s_cmp_eq_i32 s6, 0 // (op 623) 
s_cselect_b32 s7, 0, -1 // (op 623) 

s_and_b32 s6, s12, 31 // (op 623) 
s_lshl_b32 s18, 1, s6 // (op 623) 
s_add_i32 s19, s18, s7 // (op 623) 


s_and_b32 s7, s1, s19 // (op 623) 


s_add_i32 s1, s51, s7 // (op 623) 


s_ashr_i32 s7, s1, s6 // (op 623) 


s_lshl_b32 s1, s12, 24 // (op 623) 
s_ashr_i32 s6, s1, 31 // (op 623) 

s_xor_b32 s1, s7, s6 // (op 623) 

s_sub_i32 s7, s1, s6 // (op 623) 


s_add_u32 s18, s4, 16 // (op 623) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 623) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 623) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 623) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 623) 

s_mul_i32 s1, s7, s18 // (op 623) 

s_sub_i32 s51, s0, s1 // (op 623) 


s_mul_hi_i32 s0, s16, s8 // (op 623) 
s_add_i32 s1, s0, s16 // (op 623) 

s_ashr_i32 s0, s1, 31 // (op 623) 
s_cmp_eq_i32 s8, 0 // (op 623) 
s_mov_b32 s6, scc // (op 623) 
s_cmp_eq_i32 s6, 0 // (op 623) 
s_cselect_b32 s7, 0, -1 // (op 623) 

s_and_b32 s6, s15, 31 // (op 623) 
s_lshl_b32 s56, 1, s6 // (op 623) 
s_add_i32 s57, s56, s7 // (op 623) 


s_and_b32 s7, s0, s57 // (op 623) 


s_add_i32 s0, s1, s7 // (op 623) 


s_ashr_i32 s1, s0, s6 // (op 623) 


s_lshl_b32 s0, s15, 24 // (op 623) 
s_ashr_i32 s6, s0, 31 // (op 623) 

s_xor_b32 s0, s1, s6 // (op 623) 

s_sub_i32 s1, s0, s6 // (op 623) 


s_mul_i32 s0, s14, s18 // (op 623) 

s_mul_i32 s6, s50, s0 // (op 623) 

s_mul_i32 s0, s1, s6 // (op 623) 

s_sub_u32 s6, s16, s0 // (op 623) 

s_mul_hi_i32 s0, s6, s13 // (op 623) 
s_add_i32 s7, s0, s6 // (op 623) 


s_ashr_i32 s0, s7, 31 // (op 623) 
s_cmp_eq_i32 s13, 0 // (op 623) 
s_mov_b32 s6, scc // (op 623) 
s_cmp_eq_i32 s6, 0 // (op 623) 
s_cselect_b32 s18, 0, -1 // (op 623) 

s_and_b32 s6, s49, 31 // (op 623) 
s_lshl_b32 s19, 1, s6 // (op 623) 
s_add_i32 s56, s19, s18 // (op 623) 


s_and_b32 s18, s0, s56 // (op 623) 


s_add_i32 s0, s7, s18 // (op 623) 


s_ashr_i32 s7, s0, s6 // (op 623) 


s_lshl_b32 s0, s49, 24 // (op 623) 
s_ashr_i32 s6, s0, 31 // (op 623) 

s_xor_b32 s0, s7, s6 // (op 623) 

s_sub_i32 s7, s0, s6 // (op 623) 


s_cmp_eq_i32 s1, 0 // (op 623) 
s_mov_b32 s0, scc // (op 623) 

s_cmp_eq_i32 s0, 0 // (op 623) 
s_cselect_b32 s1, s7, s51 // (op 623) 



v_lshl_add_u32 v12, s1, 4, v13 // (op 623) 


v_mul_lo_u32 v13, v12, s26 // (op 623) 

v_lshl_add_u32 v12, v16, 4, v13 // (op 623) 


s_mul_i32 s1, s26, s5 // (op 625) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 625) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 625) least significant
s_mul_hi_u32 s6, s26, s4 // (op 625) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 625) most significant: sum
s_add_u32 s1, s1, s6 // (op 625) most significant: sum


s_mov_b32 s56, s28 // (op 625) reg expression
s_mov_b32 s57, s29 // (op 625) reg expression
s_mov_b32 s58, s0 // (op 625) reg expression
s_mov_b32 s59, 131072 // (op 625) call()

s_mul_hi_i32 s51, s16, s9 // (op 642) 
v_lshrrev_b32 v13, 4, v1 // (op 642) 
s_add_i32 s0, s51, s16 // (op 642) 

v_lshlrev_b32 v16, 2, v13 // (op 642) 

s_ashr_i32 s1, s0, 31 // (op 642) 
v_and_b32 v13, -16, v1 // (op 642) 
s_cmp_eq_i32 s9, 0 // (op 642) 
s_mov_b32 s51, scc // (op 642) 
v_sub_u32 v17, v1, v13 // (op 642) 

s_cmp_eq_i32 s51, 0 // (op 642) 
s_cselect_b32 s6, 0, -1 // (op 642) 

s_and_b32 s7, s10, 31 // (op 642) 
s_lshl_b32 s51, 1, s7 // (op 642) 
s_add_i32 s18, s51, s6 // (op 642) 


s_and_b32 s6, s1, s18 // (op 642) 


s_add_i32 s1, s0, s6 // (op 642) 


s_ashr_i32 s0, s1, s7 // (op 642) 


s_lshl_b32 s1, s10, 24 // (op 642) 
s_ashr_i32 s51, s1, 31 // (op 642) 

s_xor_b32 s1, s0, s51 // (op 642) 

s_sub_i32 s0, s1, s51 // (op 642) 


s_mul_hi_i32 s1, s0, s11 // (op 642) 
s_add_i32 s51, s1, s0 // (op 642) 

s_ashr_i32 s1, s51, 31 // (op 642) 
s_cmp_eq_i32 s11, 0 // (op 642) 
s_mov_b32 s6, scc // (op 642) 
s_cmp_eq_i32 s6, 0 // (op 642) 
s_cselect_b32 s7, 0, -1 // (op 642) 

s_and_b32 s6, s12, 31 // (op 642) 
s_lshl_b32 s18, 1, s6 // (op 642) 
s_add_i32 s19, s18, s7 // (op 642) 


s_and_b32 s7, s1, s19 // (op 642) 


s_add_i32 s1, s51, s7 // (op 642) 


s_ashr_i32 s7, s1, s6 // (op 642) 


s_lshl_b32 s1, s12, 24 // (op 642) 
s_ashr_i32 s6, s1, 31 // (op 642) 

s_xor_b32 s1, s7, s6 // (op 642) 

s_sub_i32 s7, s1, s6 // (op 642) 


s_add_u32 s18, s4, 16 // (op 642) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 642) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 642) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 642) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 642) 

s_mul_i32 s1, s7, s18 // (op 642) 

s_sub_i32 s51, s0, s1 // (op 642) 


s_mul_hi_i32 s0, s16, s8 // (op 642) 
s_add_i32 s1, s0, s16 // (op 642) 

s_ashr_i32 s0, s1, 31 // (op 642) 
s_cmp_eq_i32 s8, 0 // (op 642) 
s_mov_b32 s6, scc // (op 642) 
s_cmp_eq_i32 s6, 0 // (op 642) 
s_cselect_b32 s7, 0, -1 // (op 642) 

s_and_b32 s6, s15, 31 // (op 642) 
s_lshl_b32 s60, 1, s6 // (op 642) 
s_add_i32 s61, s60, s7 // (op 642) 


s_and_b32 s7, s0, s61 // (op 642) 


s_add_i32 s0, s1, s7 // (op 642) 


s_ashr_i32 s1, s0, s6 // (op 642) 


s_lshl_b32 s0, s15, 24 // (op 642) 
s_ashr_i32 s6, s0, 31 // (op 642) 

s_xor_b32 s0, s1, s6 // (op 642) 

s_sub_i32 s1, s0, s6 // (op 642) 


s_mul_i32 s0, s14, s18 // (op 642) 

s_mul_i32 s6, s50, s0 // (op 642) 

s_mul_i32 s0, s1, s6 // (op 642) 

s_sub_u32 s6, s16, s0 // (op 642) 

s_mul_hi_i32 s0, s6, s13 // (op 642) 
s_add_i32 s7, s0, s6 // (op 642) 


s_ashr_i32 s0, s7, 31 // (op 642) 
s_cmp_eq_i32 s13, 0 // (op 642) 
s_mov_b32 s6, scc // (op 642) 
s_cmp_eq_i32 s6, 0 // (op 642) 
s_cselect_b32 s18, 0, -1 // (op 642) 

s_and_b32 s6, s49, 31 // (op 642) 
s_lshl_b32 s19, 1, s6 // (op 642) 
s_add_i32 s60, s19, s18 // (op 642) 


s_and_b32 s18, s0, s60 // (op 642) 


s_add_i32 s0, s7, s18 // (op 642) 


s_ashr_i32 s7, s0, s6 // (op 642) 


s_lshl_b32 s0, s49, 24 // (op 642) 
s_ashr_i32 s6, s0, 31 // (op 642) 

s_xor_b32 s0, s7, s6 // (op 642) 

s_sub_i32 s7, s0, s6 // (op 642) 


s_cmp_eq_i32 s1, 0 // (op 642) 
s_mov_b32 s0, scc // (op 642) 

s_cmp_eq_i32 s0, 0 // (op 642) 
s_cselect_b32 s1, s7, s51 // (op 642) 



v_lshl_add_u32 v13, s1, 4, v16 // (op 642) 


v_mul_lo_u32 v16, v13, s26 // (op 642) 

v_lshl_add_u32 v13, v17, 4, v16 // (op 642) 


s_mul_i32 s1, s26, s5 // (op 644) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 644) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 644) least significant
s_mul_hi_u32 s6, s26, s4 // (op 644) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 644) most significant: sum
s_add_u32 s1, s1, s6 // (op 644) most significant: sum


s_mov_b32 s56, s28 // (op 644) reg expression
s_mov_b32 s57, s29 // (op 644) reg expression
s_mov_b32 s58, s0 // (op 644) reg expression
s_mov_b32 s59, 131072 // (op 644) call()

s_mul_hi_i32 s51, s16, s9 // (op 687) 
v_and_b32 v16, -16, v1 // (op 687) 
s_add_i32 s0, s51, s16 // (op 687) 

v_sub_u32 v17, v1, v16 // (op 687) 

s_ashr_i32 s1, s0, 31 // (op 687) 
v_lshrrev_b32 v16, 4, v1 // (op 687) 
s_cmp_eq_i32 s9, 0 // (op 687) 
s_mov_b32 s51, scc // (op 687) 
v_lshlrev_b32 v18, 1, v16 // (op 687) 

s_cmp_eq_i32 s51, 0 // (op 687) 
s_cselect_b32 s6, 0, -1 // (op 687) 

s_and_b32 s7, s10, 31 // (op 687) 
s_lshl_b32 s51, 1, s7 // (op 687) 
s_add_i32 s18, s51, s6 // (op 687) 


s_and_b32 s6, s1, s18 // (op 687) 


s_add_i32 s1, s0, s6 // (op 687) 


s_ashr_i32 s0, s1, s7 // (op 687) 


s_lshl_b32 s1, s10, 24 // (op 687) 
s_ashr_i32 s51, s1, 31 // (op 687) 

s_xor_b32 s1, s0, s51 // (op 687) 

s_sub_i32 s0, s1, s51 // (op 687) 


s_mul_hi_i32 s1, s0, s11 // (op 687) 
s_add_i32 s51, s1, s0 // (op 687) 

s_ashr_i32 s1, s51, 31 // (op 687) 
s_cmp_eq_i32 s11, 0 // (op 687) 
s_mov_b32 s6, scc // (op 687) 
s_cmp_eq_i32 s6, 0 // (op 687) 
s_cselect_b32 s7, 0, -1 // (op 687) 

s_and_b32 s6, s12, 31 // (op 687) 
s_lshl_b32 s18, 1, s6 // (op 687) 
s_add_i32 s19, s18, s7 // (op 687) 


s_and_b32 s7, s1, s19 // (op 687) 


s_add_i32 s1, s51, s7 // (op 687) 


s_ashr_i32 s7, s1, s6 // (op 687) 


s_lshl_b32 s1, s12, 24 // (op 687) 
s_ashr_i32 s6, s1, 31 // (op 687) 

s_xor_b32 s1, s7, s6 // (op 687) 

s_sub_i32 s7, s1, s6 // (op 687) 


s_mul_i32 s1, s0, s14 // (op 687) 

s_sub_u32 s0, s16, s1 // (op 687) 

s_mul_i32 s1, s7, s14 // (op 687) 
s_add_u32 s1, s1, s0 // (op 687) 


s_mul_hi_i32 s0, s16, s8 // (op 687) 
s_add_i32 s51, s0, s16 // (op 687) 

s_ashr_i32 s0, s51, 31 // (op 687) 
s_cmp_eq_i32 s8, 0 // (op 687) 
s_mov_b32 s6, scc // (op 687) 
s_cmp_eq_i32 s6, 0 // (op 687) 
s_cselect_b32 s7, 0, -1 // (op 687) 

s_and_b32 s6, s15, 31 // (op 687) 
s_lshl_b32 s18, 1, s6 // (op 687) 
s_add_i32 s19, s18, s7 // (op 687) 


s_and_b32 s7, s0, s19 // (op 687) 


s_add_i32 s0, s51, s7 // (op 687) 


s_ashr_i32 s7, s0, s6 // (op 687) 


s_lshl_b32 s0, s15, 24 // (op 687) 
s_ashr_i32 s6, s0, 31 // (op 687) 

s_xor_b32 s0, s7, s6 // (op 687) 

s_sub_i32 s7, s0, s6 // (op 687) 


s_add_u32 s18, s4, 16 // (op 687) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 687) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 687) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 687) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 687) 

s_mul_i32 s0, s14, s18 // (op 687) 

s_mul_i32 s6, s50, s0 // (op 687) 

s_mul_i32 s0, s7, s6 // (op 687) 

s_sub_u32 s6, s16, s0 // (op 687) 

s_mov_b32 s0, s6 // (op 687) 
s_mul_hi_i32 s51, s0, s13 // (op 687) 
s_add_i32 s18, s51, s0 // (op 687) 


s_ashr_i32 s0, s18, 31 // (op 687) 
s_cmp_eq_i32 s13, 0 // (op 687) 
s_mov_b32 s19, scc // (op 687) 
s_cmp_eq_i32 s19, 0 // (op 687) 
s_cselect_b32 s51, 0, -1 // (op 687) 

s_and_b32 s19, s49, 31 // (op 687) 
s_lshl_b32 s60, 1, s19 // (op 687) 
s_add_i32 s61, s60, s51 // (op 687) 


s_and_b32 s51, s0, s61 // (op 687) 


s_add_i32 s0, s18, s51 // (op 687) 


s_ashr_i32 s18, s0, s19 // (op 687) 


s_lshl_b32 s0, s49, 24 // (op 687) 
s_ashr_i32 s19, s0, 31 // (op 687) 

s_xor_b32 s0, s18, s19 // (op 687) 

s_sub_i32 s18, s0, s19 // (op 687) 


s_mul_i32 s0, s48, s14 // (op 687) 
s_add_u32 s60, s20, 16 // (op 687) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 687) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 687) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 687) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 687) 

s_sub_i32 s19, s60, s0 // (op 687) 

s_mul_i32 s51, s18, s19 // (op 687) 


s_sub_u32 s18, s6, s51 // (op 687) 


s_add_u32 s6, s18, s0 // (op 687) 


s_cmp_eq_i32 s7, 0 // (op 687) 
s_mov_b32 s0, scc // (op 687) 

s_cmp_eq_i32 s0, 0 // (op 687) 
s_cselect_b32 s7, s6, s1 // (op 687) 



v_lshl_add_u32 v16, s7, 4, v17 // (op 687) 


v_mul_lo_u32 v17, v18, s30 // (op 687) 

v_add_u32 v18, v16, v17 // (op 687) 


s_mul_i32 s1, s30, s35 // (op 689) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 689) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 689) least significant
s_mul_hi_u32 s6, s30, s34 // (op 689) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 689) most significant: sum
s_add_u32 s1, s1, s6 // (op 689) most significant: sum


s_mov_b32 s60, s32 // (op 689) reg expression
s_mov_b32 s61, s33 // (op 689) reg expression
s_mov_b32 s62, s0 // (op 689) reg expression
s_mov_b32 s63, 131072 // (op 689) call()

s_mul_hi_i32 s51, s16, s9 // (op 706) 
v_and_b32 v16, -16, v1 // (op 706) 
s_add_i32 s0, s51, s16 // (op 706) 

v_sub_u32 v17, v1, v16 // (op 706) 

s_ashr_i32 s1, s0, 31 // (op 706) 
v_lshrrev_b32 v16, 4, v1 // (op 706) 
s_cmp_eq_i32 s9, 0 // (op 706) 
s_mov_b32 s51, scc // (op 706) 
v_lshlrev_b32 v19, 1, v16 // (op 706) 

s_cmp_eq_i32 s51, 0 // (op 706) 
s_cselect_b32 s6, 0, -1 // (op 706) 

s_and_b32 s7, s10, 31 // (op 706) 
s_lshl_b32 s51, 1, s7 // (op 706) 
s_add_i32 s18, s51, s6 // (op 706) 


s_and_b32 s6, s1, s18 // (op 706) 


s_add_i32 s1, s0, s6 // (op 706) 


s_ashr_i32 s0, s1, s7 // (op 706) 


s_lshl_b32 s1, s10, 24 // (op 706) 
s_ashr_i32 s51, s1, 31 // (op 706) 

s_xor_b32 s1, s0, s51 // (op 706) 

s_sub_i32 s0, s1, s51 // (op 706) 


s_mul_hi_i32 s1, s0, s11 // (op 706) 
s_add_i32 s51, s1, s0 // (op 706) 

s_ashr_i32 s1, s51, 31 // (op 706) 
s_cmp_eq_i32 s11, 0 // (op 706) 
s_mov_b32 s6, scc // (op 706) 
s_cmp_eq_i32 s6, 0 // (op 706) 
s_cselect_b32 s7, 0, -1 // (op 706) 

s_and_b32 s6, s12, 31 // (op 706) 
s_lshl_b32 s18, 1, s6 // (op 706) 
s_add_i32 s19, s18, s7 // (op 706) 


s_and_b32 s7, s1, s19 // (op 706) 


s_add_i32 s1, s51, s7 // (op 706) 


s_ashr_i32 s7, s1, s6 // (op 706) 


s_lshl_b32 s1, s12, 24 // (op 706) 
s_ashr_i32 s6, s1, 31 // (op 706) 

s_xor_b32 s1, s7, s6 // (op 706) 

s_sub_i32 s7, s1, s6 // (op 706) 


s_mul_i32 s1, s0, s14 // (op 706) 

s_sub_u32 s0, s16, s1 // (op 706) 

s_mul_i32 s1, s7, s14 // (op 706) 
s_add_u32 s1, s1, s0 // (op 706) 


s_mul_hi_i32 s0, s16, s8 // (op 706) 
s_add_i32 s51, s0, s16 // (op 706) 

s_ashr_i32 s0, s51, 31 // (op 706) 
s_cmp_eq_i32 s8, 0 // (op 706) 
s_mov_b32 s6, scc // (op 706) 
s_cmp_eq_i32 s6, 0 // (op 706) 
s_cselect_b32 s7, 0, -1 // (op 706) 

s_and_b32 s6, s15, 31 // (op 706) 
s_lshl_b32 s18, 1, s6 // (op 706) 
s_add_i32 s19, s18, s7 // (op 706) 


s_and_b32 s7, s0, s19 // (op 706) 


s_add_i32 s0, s51, s7 // (op 706) 


s_ashr_i32 s7, s0, s6 // (op 706) 


s_lshl_b32 s0, s15, 24 // (op 706) 
s_ashr_i32 s6, s0, 31 // (op 706) 

s_xor_b32 s0, s7, s6 // (op 706) 

s_sub_i32 s7, s0, s6 // (op 706) 


s_add_u32 s18, s4, 16 // (op 706) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 706) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 706) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 706) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 706) 

s_mul_i32 s0, s14, s18 // (op 706) 

s_mul_i32 s6, s50, s0 // (op 706) 

s_mul_i32 s0, s7, s6 // (op 706) 

s_sub_u32 s6, s16, s0 // (op 706) 

s_mov_b32 s0, s6 // (op 706) 
s_mul_hi_i32 s51, s0, s13 // (op 706) 
s_add_i32 s18, s51, s0 // (op 706) 


s_ashr_i32 s0, s18, 31 // (op 706) 
s_cmp_eq_i32 s13, 0 // (op 706) 
s_mov_b32 s19, scc // (op 706) 
s_cmp_eq_i32 s19, 0 // (op 706) 
s_cselect_b32 s51, 0, -1 // (op 706) 

s_and_b32 s19, s49, 31 // (op 706) 
s_lshl_b32 s64, 1, s19 // (op 706) 
s_add_i32 s65, s64, s51 // (op 706) 


s_and_b32 s51, s0, s65 // (op 706) 


s_add_i32 s0, s18, s51 // (op 706) 


s_ashr_i32 s18, s0, s19 // (op 706) 


s_lshl_b32 s0, s49, 24 // (op 706) 
s_ashr_i32 s19, s0, 31 // (op 706) 

s_xor_b32 s0, s18, s19 // (op 706) 

s_sub_i32 s18, s0, s19 // (op 706) 


s_mul_i32 s0, s48, s14 // (op 706) 
s_add_u32 s64, s20, 16 // (op 706) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 706) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 706) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 706) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 706) 

s_sub_i32 s19, s64, s0 // (op 706) 

s_mul_i32 s51, s18, s19 // (op 706) 


s_sub_u32 s18, s6, s51 // (op 706) 


s_add_u32 s6, s18, s0 // (op 706) 


s_cmp_eq_i32 s7, 0 // (op 706) 
s_mov_b32 s0, scc // (op 706) 

s_cmp_eq_i32 s0, 0 // (op 706) 
s_cselect_b32 s7, s6, s1 // (op 706) 



v_lshl_add_u32 v16, s7, 4, v17 // (op 706) 


v_mul_lo_u32 v17, v19, s30 // (op 706) 

v_add_u32 v19, v16, v17 // (op 706) 


s_mul_i32 s1, s30, s35 // (op 708) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 708) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 708) least significant
s_mul_hi_u32 s6, s30, s34 // (op 708) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 708) most significant: sum
s_add_u32 s1, s1, s6 // (op 708) most significant: sum


s_mov_b32 s60, s32 // (op 708) reg expression
s_mov_b32 s61, s33 // (op 708) reg expression
s_mov_b32 s62, s0 // (op 708) reg expression
s_mov_b32 s63, 131072 // (op 708) call()

s_mul_hi_i32 s51, s16, s9 // (op 751) 
v_lshrrev_b32 v16, 3, v1 // (op 751) 
s_add_i32 s0, s51, s16 // (op 751) 

v_lshlrev_b32 v17, 1, v16 // (op 751) 

s_ashr_i32 s1, s0, 31 // (op 751) 
v_and_b32 v16, -8, v1 // (op 751) 
s_cmp_eq_i32 s9, 0 // (op 751) 
s_mov_b32 s51, scc // (op 751) 
v_sub_u32 v20, v1, v16 // (op 751) 

s_cmp_eq_i32 s51, 0 // (op 751) 
s_cselect_b32 s6, 0, -1 // (op 751) 

s_and_b32 s7, s10, 31 // (op 751) 
s_lshl_b32 s51, 1, s7 // (op 751) 
s_add_i32 s18, s51, s6 // (op 751) 


s_and_b32 s6, s1, s18 // (op 751) 


s_add_i32 s1, s0, s6 // (op 751) 


s_ashr_i32 s0, s1, s7 // (op 751) 


s_lshl_b32 s1, s10, 24 // (op 751) 
s_ashr_i32 s51, s1, 31 // (op 751) 

s_xor_b32 s1, s0, s51 // (op 751) 

s_sub_i32 s0, s1, s51 // (op 751) 


s_mul_hi_i32 s1, s0, s11 // (op 751) 
s_add_i32 s51, s1, s0 // (op 751) 

s_ashr_i32 s1, s51, 31 // (op 751) 
s_cmp_eq_i32 s11, 0 // (op 751) 
s_mov_b32 s6, scc // (op 751) 
s_cmp_eq_i32 s6, 0 // (op 751) 
s_cselect_b32 s7, 0, -1 // (op 751) 

s_and_b32 s6, s12, 31 // (op 751) 
s_lshl_b32 s18, 1, s6 // (op 751) 
s_add_i32 s19, s18, s7 // (op 751) 


s_and_b32 s7, s1, s19 // (op 751) 


s_add_i32 s1, s51, s7 // (op 751) 


s_ashr_i32 s7, s1, s6 // (op 751) 


s_lshl_b32 s1, s12, 24 // (op 751) 
s_ashr_i32 s6, s1, 31 // (op 751) 

s_xor_b32 s1, s7, s6 // (op 751) 

s_sub_i32 s7, s1, s6 // (op 751) 


s_add_u32 s18, s4, 16 // (op 751) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 751) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 751) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 751) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 751) 

s_mul_i32 s1, s7, s18 // (op 751) 

s_sub_i32 s51, s0, s1 // (op 751) 


s_mul_hi_i32 s0, s16, s8 // (op 751) 
s_add_i32 s1, s0, s16 // (op 751) 

s_ashr_i32 s0, s1, 31 // (op 751) 
s_cmp_eq_i32 s8, 0 // (op 751) 
s_mov_b32 s6, scc // (op 751) 
s_cmp_eq_i32 s6, 0 // (op 751) 
s_cselect_b32 s7, 0, -1 // (op 751) 

s_and_b32 s6, s15, 31 // (op 751) 
s_lshl_b32 s64, 1, s6 // (op 751) 
s_add_i32 s65, s64, s7 // (op 751) 


s_and_b32 s7, s0, s65 // (op 751) 


s_add_i32 s0, s1, s7 // (op 751) 


s_ashr_i32 s1, s0, s6 // (op 751) 


s_lshl_b32 s0, s15, 24 // (op 751) 
s_ashr_i32 s6, s0, 31 // (op 751) 

s_xor_b32 s0, s1, s6 // (op 751) 

s_sub_i32 s1, s0, s6 // (op 751) 


s_mul_i32 s0, s14, s18 // (op 751) 

s_mul_i32 s6, s50, s0 // (op 751) 

s_mul_i32 s0, s1, s6 // (op 751) 

s_sub_u32 s6, s16, s0 // (op 751) 

s_mul_hi_i32 s0, s6, s13 // (op 751) 
s_add_i32 s7, s0, s6 // (op 751) 


s_ashr_i32 s0, s7, 31 // (op 751) 
s_cmp_eq_i32 s13, 0 // (op 751) 
s_mov_b32 s6, scc // (op 751) 
s_cmp_eq_i32 s6, 0 // (op 751) 
s_cselect_b32 s18, 0, -1 // (op 751) 

s_and_b32 s6, s49, 31 // (op 751) 
s_lshl_b32 s19, 1, s6 // (op 751) 
s_add_i32 s64, s19, s18 // (op 751) 


s_and_b32 s18, s0, s64 // (op 751) 


s_add_i32 s0, s7, s18 // (op 751) 


s_ashr_i32 s7, s0, s6 // (op 751) 


s_lshl_b32 s0, s49, 24 // (op 751) 
s_ashr_i32 s6, s0, 31 // (op 751) 

s_xor_b32 s0, s7, s6 // (op 751) 

s_sub_i32 s7, s0, s6 // (op 751) 


s_cmp_eq_i32 s1, 0 // (op 751) 
s_mov_b32 s0, scc // (op 751) 

s_cmp_eq_i32 s0, 0 // (op 751) 
s_cselect_b32 s1, s7, s51 // (op 751) 



v_lshl_add_u32 v16, s1, 4, v17 // (op 751) 


v_mul_lo_u32 v17, v16, s36 // (op 751) 

v_add_u32 v16, v20, v17 // (op 751) 


s_mul_i32 s1, s36, s5 // (op 753) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 753) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 753) least significant
s_mul_hi_u32 s6, s36, s4 // (op 753) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 753) most significant: sum
s_add_u32 s1, s1, s6 // (op 753) most significant: sum


s_mov_b32 s64, s38 // (op 753) reg expression
s_mov_b32 s65, s39 // (op 753) reg expression
s_mov_b32 s66, s0 // (op 753) reg expression
s_mov_b32 s67, 131072 // (op 753) call()

s_mul_hi_i32 s51, s16, s9 // (op 770) 
v_lshrrev_b32 v17, 3, v1 // (op 770) 
s_add_i32 s0, s51, s16 // (op 770) 

v_lshlrev_b32 v20, 1, v17 // (op 770) 

s_ashr_i32 s1, s0, 31 // (op 770) 
v_and_b32 v17, -8, v1 // (op 770) 
s_cmp_eq_i32 s9, 0 // (op 770) 
s_mov_b32 s51, scc // (op 770) 
v_sub_u32 v21, v1, v17 // (op 770) 

s_cmp_eq_i32 s51, 0 // (op 770) 
s_cselect_b32 s6, 0, -1 // (op 770) 

s_and_b32 s7, s10, 31 // (op 770) 
s_lshl_b32 s51, 1, s7 // (op 770) 
s_add_i32 s18, s51, s6 // (op 770) 


s_and_b32 s6, s1, s18 // (op 770) 


s_add_i32 s1, s0, s6 // (op 770) 


s_ashr_i32 s0, s1, s7 // (op 770) 


s_lshl_b32 s1, s10, 24 // (op 770) 
s_ashr_i32 s51, s1, 31 // (op 770) 

s_xor_b32 s1, s0, s51 // (op 770) 

s_sub_i32 s0, s1, s51 // (op 770) 


s_mul_hi_i32 s1, s0, s11 // (op 770) 
s_add_i32 s51, s1, s0 // (op 770) 

s_ashr_i32 s1, s51, 31 // (op 770) 
s_cmp_eq_i32 s11, 0 // (op 770) 
s_mov_b32 s6, scc // (op 770) 
s_cmp_eq_i32 s6, 0 // (op 770) 
s_cselect_b32 s7, 0, -1 // (op 770) 

s_and_b32 s6, s12, 31 // (op 770) 
s_lshl_b32 s18, 1, s6 // (op 770) 
s_add_i32 s19, s18, s7 // (op 770) 


s_and_b32 s7, s1, s19 // (op 770) 


s_add_i32 s1, s51, s7 // (op 770) 


s_ashr_i32 s7, s1, s6 // (op 770) 


s_lshl_b32 s1, s12, 24 // (op 770) 
s_ashr_i32 s6, s1, 31 // (op 770) 

s_xor_b32 s1, s7, s6 // (op 770) 

s_sub_i32 s7, s1, s6 // (op 770) 


s_add_u32 s18, s4, 16 // (op 770) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 770) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 770) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 770) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 770) 

s_mul_i32 s1, s7, s18 // (op 770) 

s_sub_i32 s51, s0, s1 // (op 770) 


s_mul_hi_i32 s0, s16, s8 // (op 770) 
s_add_i32 s1, s0, s16 // (op 770) 

s_ashr_i32 s0, s1, 31 // (op 770) 
s_cmp_eq_i32 s8, 0 // (op 770) 
s_mov_b32 s6, scc // (op 770) 
s_cmp_eq_i32 s6, 0 // (op 770) 
s_cselect_b32 s7, 0, -1 // (op 770) 

s_and_b32 s6, s15, 31 // (op 770) 
s_lshl_b32 s68, 1, s6 // (op 770) 
s_add_i32 s69, s68, s7 // (op 770) 


s_and_b32 s7, s0, s69 // (op 770) 


s_add_i32 s0, s1, s7 // (op 770) 


s_ashr_i32 s1, s0, s6 // (op 770) 


s_lshl_b32 s0, s15, 24 // (op 770) 
s_ashr_i32 s6, s0, 31 // (op 770) 

s_xor_b32 s0, s1, s6 // (op 770) 

s_sub_i32 s1, s0, s6 // (op 770) 


s_mul_i32 s0, s14, s18 // (op 770) 

s_mul_i32 s6, s50, s0 // (op 770) 

s_mul_i32 s0, s1, s6 // (op 770) 

s_sub_u32 s6, s16, s0 // (op 770) 

s_mul_hi_i32 s0, s6, s13 // (op 770) 
s_add_i32 s7, s0, s6 // (op 770) 


s_ashr_i32 s0, s7, 31 // (op 770) 
s_cmp_eq_i32 s13, 0 // (op 770) 
s_mov_b32 s6, scc // (op 770) 
s_cmp_eq_i32 s6, 0 // (op 770) 
s_cselect_b32 s18, 0, -1 // (op 770) 

s_and_b32 s6, s49, 31 // (op 770) 
s_lshl_b32 s19, 1, s6 // (op 770) 
s_add_i32 s68, s19, s18 // (op 770) 


s_and_b32 s18, s0, s68 // (op 770) 


s_add_i32 s0, s7, s18 // (op 770) 


s_ashr_i32 s7, s0, s6 // (op 770) 


s_lshl_b32 s0, s49, 24 // (op 770) 
s_ashr_i32 s6, s0, 31 // (op 770) 

s_xor_b32 s0, s7, s6 // (op 770) 

s_sub_i32 s7, s0, s6 // (op 770) 


s_cmp_eq_i32 s1, 0 // (op 770) 
s_mov_b32 s0, scc // (op 770) 

s_cmp_eq_i32 s0, 0 // (op 770) 
s_cselect_b32 s1, s7, s51 // (op 770) 



v_lshl_add_u32 v17, s1, 4, v20 // (op 770) 


v_mul_lo_u32 v20, v17, s36 // (op 770) 

v_add_u32 v17, v21, v20 // (op 770) 


s_mul_i32 s1, s36, s5 // (op 772) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 772) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 772) least significant
s_mul_hi_u32 s6, s36, s4 // (op 772) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 772) most significant: sum
s_add_u32 s1, s1, s6 // (op 772) most significant: sum


s_mov_b32 s64, s38 // (op 772) reg expression
s_mov_b32 s65, s39 // (op 772) reg expression
s_mov_b32 s66, s0 // (op 772) reg expression
s_mov_b32 s67, 131072 // (op 772) call()


v_mov_b64 v[20:21], s[26:27] // (op 424) convert

v_mov_b32 v22, v12 // (op 424) 
buffer_load_dwordx4 v[24:27], v22, s[56:59], 0 offen // (op 424) Load value
v_add_u32 v22, v22, v20 // (op 424) 
buffer_load_dwordx4 v[28:31], v22, s[56:59], 0 offen // (op 424) Load value
v_add_u32 v22, v22, v20 // (op 424) 
buffer_load_dwordx4 v[32:35], v22, s[56:59], 0 offen // (op 424) Load value
v_add_u32 v22, v22, v20 // (op 424) 
buffer_load_dwordx4 v[36:39], v22, s[56:59], 0 offen // (op 424) Load value



v_mov_b64 v[20:21], s[36:37] // (op 427) convert

v_mov_b32 v22, v16 // (op 427) 
buffer_load_ubyte v40, v22, s[64:67], 0 offen offset: 0 // (op 427) Load value
v_add_u32 v22, v22, v20 // (op 427) 
buffer_load_ubyte v41, v22, s[64:67], 0 offen offset: 0 // (op 427) Load value



s_lshl_b64 s[0:1], s[2:3], 2 // (op 430) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 430) 


v_mov_b32 v22, v14 // (op 430) reg expression
buffer_load_dwordx2 v[42:43], v22, s[52:55], 0 offen // (op 430) Load value
v_add_u32 v22, v22, v20 // (op 430) 
buffer_load_dwordx2 v[44:45], v22, s[52:55], 0 offen // (op 430) Load value
v_add_u32 v22, v22, v20 // (op 430) 
buffer_load_dwordx2 v[46:47], v22, s[52:55], 0 offen // (op 430) Load value
v_add_u32 v22, v22, v20 // (op 430) 
buffer_load_dwordx2 v[48:49], v22, s[52:55], 0 offen // (op 430) Load value



v_mov_b64 v[20:21], s[30:31] // (op 433) convert

v_mov_b32 v22, v18 // (op 433) reg expression
buffer_load_ubyte v50, v22, s[60:63], 0 offen offset: 0 // (op 433) Load value
v_add_u32 v22, v22, v20 // (op 433) 
buffer_load_ubyte v51, v22, s[60:63], 0 offen offset: 0 // (op 433) Load value



v_mov_b64 v[20:21], s[26:27] // (op 436) convert

v_mov_b32 v22, v12 // (op 436) 
buffer_load_dwordx4 v[52:55], v22, s[56:59], 0 offen offset:256 // (op 436) Load value
v_add_u32 v22, v22, v20 // (op 436) 
buffer_load_dwordx4 v[56:59], v22, s[56:59], 0 offen offset:256 // (op 436) Load value
v_add_u32 v22, v22, v20 // (op 436) 
buffer_load_dwordx4 v[60:63], v22, s[56:59], 0 offen offset:256 // (op 436) Load value
v_add_u32 v22, v22, v20 // (op 436) 
buffer_load_dwordx4 v[64:67], v22, s[56:59], 0 offen offset:256 // (op 436) Load value




v_mov_b64 v[20:21], s[36:37] // (op 439) convert

v_mov_b32 v12, v16 // (op 439) 
buffer_load_ubyte v22, v12, s[64:67], 0 offen offset:8 // (op 439) Load value
v_add_u32 v12, v12, v20 // (op 439) 
buffer_load_ubyte v23, v12, s[64:67], 0 offen offset:8 // (op 439) Load value




s_lshl_b64 s[0:1], s[2:3], 2 // (op 442) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 442) 


s_lshl_b64 s[0:1], s[2:3], 8 // (op 442) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 442) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 442) 

v_add_u32 v12, v14, s0 // (op 442) 

buffer_load_dwordx2 v[68:69], v12, s[52:55], 0 offen // (op 442) Load value
v_add_u32 v12, v12, v20 // (op 442) 
buffer_load_dwordx2 v[70:71], v12, s[52:55], 0 offen // (op 442) Load value
v_add_u32 v12, v12, v20 // (op 442) 
buffer_load_dwordx2 v[72:73], v12, s[52:55], 0 offen // (op 442) Load value
v_add_u32 v12, v12, v20 // (op 442) 
buffer_load_dwordx2 v[74:75], v12, s[52:55], 0 offen // (op 442) Load value




v_mov_b64 v[20:21], s[30:31] // (op 445) convert

s_lshl_b32 s51, s30, 3 // (op 445) 
v_add_u32 v12, v18, s51 // (op 445) 

buffer_load_ubyte v76, v12, s[60:63], 0 offen offset: 0 // (op 445) Load value
v_add_u32 v12, v12, v20 // (op 445) 
buffer_load_ubyte v77, v12, s[60:63], 0 offen offset: 0 // (op 445) Load value



v_and_b32 v12, -16, v1 // (op 890) 
v_sub_u32 v14, v1, v12 // (op 890) 

v_lshlrev_b32 v12, 4, v14 // (op 890) 

v_lshrrev_b32 v14, 4, v1 // (op 890) 
v_lshl_add_u32 v16, v14, 10, v12 // (op 890) 





s_waitcnt vmcnt(23)
ds_write_b128 v16, v[24:27] // (op 447) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v16, v[28:31] offset:256 // (op 447) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v16, v[32:35] offset:512 // (op 447) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v16, v[36:39] offset:768 // (op 447) Store local data 

v_and_b32 v12, -8, v1 // (op 996) 
v_sub_u32 v14, v1, v12 // (op 996) 

v_lshrrev_b32 v12, 3, v1 // (op 996) 
v_lshl_add_u32 v16, v12, 4, v14 // (op 996) 





s_waitcnt vmcnt(19)
ds_write_b8 v16, v40 offset:4096 // (op 448) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v16, v41 offset:4104 // (op 448) Store local data 

v_lshlrev_b32 v12, 8, v1 // (op 837) 
v_lshrrev_b32 v14, 3, v12 // (op 837) 




s_waitcnt vmcnt(17)
ds_write_b64 v14, v[42:43] offset:4224 // (op 449) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v14, v[44:45] offset:4232 // (op 449) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v14, v[46:47] offset:4240 // (op 449) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v14, v[48:49] offset:4248 // (op 449) Store local data 

v_and_b32 v12, -16, v1 // (op 943) 
v_sub_u32 v14, v1, v12 // (op 943) 

v_lshrrev_b32 v12, 4, v1 // (op 943) 
v_lshl_add_u32 v16, v12, 5, v14 // (op 943) 





s_waitcnt vmcnt(13)
ds_write_b8 v16, v50 offset:6272 // (op 450) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v16, v51 offset:6288 // (op 450) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 1194) 



ds_read_b128 v[78:81], v5 // (op 467) Load local data 
ds_read_b128 v[82:85], v5 offset:64 // (op 467) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1235) 



ds_read_u8 v12, v6 offset:4096 // (op 473) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1175) 



ds_read_b64_tr_b4 v[86:87], v3 offset:4224 // (op 464) Transpose load local data 
ds_read_b64_tr_b4 v[88:89], v3 offset:4352 // (op 464) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1215) 


ds_read_u8 v14, v4 offset:6272 // (op 470) Load local data 
v_lshlrev_b32 v16, 8, v1 // (op 808) 
v_lshrrev_b32 v18, 3, v16 // (op 808) 

v_and_b32 v16, -16, v1 // (op 861) 
v_sub_u32 v20, v1, v16 // (op 861) 

v_lshlrev_b32 v16, 4, v20 // (op 861) 

v_lshrrev_b32 v20, 4, v1 // (op 861) 
v_lshl_add_u32 v21, v20, 10, v16 // (op 861) 


v_and_b32 v16, -16, v1 // (op 914) 
v_sub_u32 v20, v1, v16 // (op 914) 

v_lshrrev_b32 v16, 4, v1 // (op 914) 
v_lshl_add_u32 v90, v16, 5, v20 // (op 914) 


v_and_b32 v16, -8, v1 // (op 967) 
v_sub_u32 v20, v1, v16 // (op 967) 

v_lshrrev_b32 v16, 3, v1 // (op 967) 
v_lshl_add_u32 v91, v16, 4, v20 // (op 967) 


v_lshlrev_b32 v16, 8, v1 // (op 1054) 
v_lshrrev_b32 v20, 3, v16 // (op 1054) 

v_and_b32 v16, -16, v1 // (op 1078) 
v_sub_u32 v92, v1, v16 // (op 1078) 

v_lshlrev_b32 v16, 4, v92 // (op 1078) 

v_lshrrev_b32 v92, 4, v1 // (op 1078) 
v_lshl_add_u32 v93, v92, 10, v16 // (op 1078) 


v_and_b32 v16, -16, v1 // (op 1102) 
v_sub_u32 v92, v1, v16 // (op 1102) 

v_lshrrev_b32 v16, 4, v1 // (op 1102) 
v_lshl_add_u32 v94, v16, 5, v92 // (op 1102) 


v_and_b32 v16, -8, v1 // (op 1126) 
v_sub_u32 v92, v1, v16 // (op 1126) 

v_lshrrev_b32 v16, 3, v1 // (op 1126) 
v_lshl_add_u32 v95, v16, 4, v92 // (op 1126) 


s_mov_b32 s51, 0 // (op 59) call()
s_ashr_i64 s[0:1], s[24:25], 8 // (op 58) 
s_mov_b32 s6, s0 // (op 58) convert

s_and_b32 s7, s6, 4294967294 // (op 58) 

s_cmp_lt_u32 s51, s7 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58 if false)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58:

s_add_u32 s51, s51, 2 // (op 60) 
v_mov_b64 v[96:97], s[26:27] // (op 10) convert

v_mov_b32 v16, v13 // (op 10) 
v_mov_b32 v92, v16 // (op 10) 
buffer_load_dwordx4 v[24:27], v92, s[56:59], 0 offen offset:512 // (op 10) Load value
v_add_u32 v92, v92, v96 // (op 10) 
buffer_load_dwordx4 v[28:31], v92, s[56:59], 0 offen offset:512 // (op 10) Load value
v_add_u32 v92, v92, v96 // (op 10) 
buffer_load_dwordx4 v[32:35], v92, s[56:59], 0 offen offset:512 // (op 10) Load value
v_add_u32 v92, v92, v96 // (op 10) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[86:89], v[78:85], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 88) 
buffer_load_dwordx4 v[36:39], v92, s[56:59], 0 offen offset:512 // (op 10) Load value



s_barrier  // (op 1184) 
v_mov_b64 v[96:97], s[36:37] // (op 25) convert




ds_read_b64_tr_b4 v[86:87], v3 offset:5248 // (op 91) Transpose load local data 
ds_read_b64_tr_b4 v[88:89], v3 offset:5376 // (op 91) Transpose load local data 
v_mov_b32 v92, v17 // (op 25) 
s_barrier  // (op 1204) 
v_mov_b32 v98, v92 // (op 25) 



ds_read_b128 v[78:81], v5 offset:128 // (op 95) Load local data 
buffer_load_ubyte v40, v98, s[64:67], 0 offen offset:16 // (op 25) Load value
v_add_u32 v98, v98, v96 // (op 25) 
buffer_load_ubyte v41, v98, s[64:67], 0 offen offset:16 // (op 25) Load value


ds_read_b128 v[82:85], v5 offset:192 // (op 95) Load local data 

s_barrier  // (op 1225) 
s_lshl_b64 s[0:1], s[2:3], 2 // (op 4) 


ds_read_u8 v14, v4 offset:6336 // (op 99) Load local data 
v_lshrrev_b64 v[96:97], 3, s[0:1] // (op 4) 


s_barrier  // (op 1245) 
v_mov_b32 v98, v15 // (op 4) 



ds_read_u8 v12, v6 offset:4100 // (op 103) Load local data 
s_lshl_b64 s[0:1], s[2:3], 8 // (op 4) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 4) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 4) 

s_lshl_b32 s6, s0, 1 // (op 4) 

v_add_u32 v99, v98, s6 // (op 4) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[86:89], v[78:85], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 105) 
buffer_load_dwordx2 v[42:43], v99, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v99, v99, v96 // (op 4) 
buffer_load_dwordx2 v[44:45], v99, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v99, v99, v96 // (op 4) 
buffer_load_dwordx2 v[46:47], v99, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v99, v99, v96 // (op 4) 
buffer_load_dwordx2 v[48:49], v99, s[52:55], 0 offen // (op 4) Load value



v_mov_b64 v[96:97], s[30:31] // (op 17) convert

v_mov_b32 v99, v19 // (op 17) 
s_lshl_b32 s0, s30, 4 // (op 17) 
v_add_u32 v100, v99, s0 // (op 17) 

buffer_load_ubyte v50, v100, s[60:63], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v100, v100, v96 // (op 17) 
buffer_load_ubyte v51, v100, s[60:63], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1265) 



s_waitcnt vmcnt(23)
ds_write_b128 v93, v[52:55] offset:6400 // (op 302) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v93, v[56:59] offset:6656 // (op 302) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v93, v[60:63] offset:6912 // (op 302) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v93, v[64:67] offset:7168 // (op 302) Store local data 
s_barrier  // (op 1277) 



s_waitcnt vmcnt(19)
ds_write_b8 v95, v22 offset:10496 // (op 304) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v95, v23 offset:10504 // (op 304) Store local data 
s_barrier  // (op 1258) 



s_waitcnt vmcnt(17)
ds_write_b64 v20, v[68:69] offset:10624 // (op 283) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v20, v[70:71] offset:10632 // (op 283) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v20, v[72:73] offset:10640 // (op 283) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v20, v[74:75] offset:10648 // (op 283) Store local data 
s_barrier  // (op 1271) 



s_waitcnt vmcnt(13)
ds_write_b8 v94, v76 offset:12672 // (op 303) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v94, v77 offset:12688 // (op 303) Store local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1261) 



ds_read_b128 v[100:103], v9 offset:6400 // (op 287) Load local data 
ds_read_b128 v[104:107], v9 offset:6464 // (op 287) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1274) 



ds_read_u8 v96, v10 offset:10496 // (op 291) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1255) 



ds_read_b64_tr_b4 v[108:109], v7 offset:10624 // (op 285) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v7 offset:10752 // (op 285) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1268) 


ds_read_u8 v97, v8 offset:12672 // (op 289) Load local data 

v_mov_b64 v[112:113], s[26:27] // (op 305) convert

v_mov_b32 v114, v16 // (op 305) 
buffer_load_dwordx4 v[52:55], v114, s[56:59], 0 offen offset:768 // (op 305) Load value
v_add_u32 v114, v114, v112 // (op 305) 
buffer_load_dwordx4 v[56:59], v114, s[56:59], 0 offen offset:768 // (op 305) Load value
v_add_u32 v114, v114, v112 // (op 305) 
buffer_load_dwordx4 v[60:63], v114, s[56:59], 0 offen offset:768 // (op 305) Load value
v_add_u32 v114, v114, v112 // (op 305) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[108:111], v[100:107], a[0:3], v97, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 292) 
buffer_load_dwordx4 v[64:67], v114, s[56:59], 0 offen offset:768 // (op 305) Load value











v_add_u32 v13, 512, v13 // (op 634) 
v_mov_b64 v[96:97], s[36:37] // (op 307) convert

v_mov_b32 v16, v92 // (op 307) 
buffer_load_ubyte v22, v16, s[64:67], 0 offen offset:24 // (op 307) Load value
v_add_u32 v16, v16, v96 // (op 307) 
buffer_load_ubyte v23, v16, s[64:67], 0 offen offset:24 // (op 307) Load value


ds_read_b64_tr_b4 v[100:101], v7 offset:11648 // (op 294) Transpose load local data 

ds_read_b64_tr_b4 v[102:103], v7 offset:11776 // (op 294) Transpose load local data 

v_add_u32 v17, 16, v17 // (op 762) 



s_barrier  // (op 1336) 
s_lshl_b64 s[0:1], s[2:3], 2 // (op 282) 
v_lshrrev_b64 v[96:97], 3, s[0:1] // (op 282) 


s_lshl_b64 s[0:1], s[2:3], 8 // (op 282) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 282) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 282) 

s_mul_i32 s6, 3, s0 // (op 282) 

v_add_u32 v16, v98, s6 // (op 282) 

ds_read_b128 v[104:107], v9 offset:6528 // (op 296) Load local data 
ds_read_b128 v[108:111], v9 offset:6592 // (op 296) Load local data 
buffer_load_dwordx2 v[68:69], v16, s[52:55], 0 offen // (op 282) Load value
v_add_u32 v16, v16, v96 // (op 282) 


s_barrier  // (op 1340) 
ds_read_u8 v92, v8 offset:12736 // (op 298) Load local data 
buffer_load_dwordx2 v[70:71], v16, s[52:55], 0 offen // (op 282) Load value
v_add_u32 v16, v16, v96 // (op 282) 



s_barrier  // (op 1344) 
buffer_load_dwordx2 v[72:73], v16, s[52:55], 0 offen // (op 282) Load value
v_add_u32 v16, v16, v96 // (op 282) 
ds_read_u8 v112, v10 offset:10500 // (op 300) Load local data 
buffer_load_dwordx2 v[74:75], v16, s[52:55], 0 offen // (op 282) Load value




s_barrier  // (op 1348) 
v_mov_b64 v[96:97], s[30:31] // (op 306) convert

s_lshl_b32 s0, s30, 3 // (op 306) 
s_mul_i32 s1, 3, s0 // (op 306) 

v_add_u32 v16, v99, s1 // (op 306) 

buffer_load_ubyte v76, v16, s[60:63], 0 offen offset: 0 // (op 306) Load value
v_add_u32 v16, v16, v96 // (op 306) 
buffer_load_ubyte v77, v16, s[60:63], 0 offen offset: 0 // (op 306) Load value


s_lshl_b64 s[0:1], s[2:3], 8 // (op 570) 

s_lshl_b64 s[6:7], s[0:1], 2 // (op 570) 

s_barrier  // (op 1197) 
s_lshl_b32 s0, s30, 4 // (op 698) 



v_add_u32 v19, v19, s0 // (op 698) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 570) 

s_lshl_b32 s6, s0, 1 // (op 570) 

v_add_u32 v15, v15, s6 // (op 570) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[100:103], v[104:111], a[0:3], v92, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 301) 
s_waitcnt vmcnt(23)
ds_write_b128 v21, v[24:27] // (op 13) Store local data 




s_waitcnt vmcnt(22)
ds_write_b128 v21, v[28:31] offset:256 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v21, v[32:35] offset:512 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v21, v[36:39] offset:768 // (op 13) Store local data 
s_barrier  // (op 1238) 



s_waitcnt vmcnt(19)
ds_write_b8 v91, v40 offset:4096 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v91, v41 offset:4104 // (op 30) Store local data 
s_barrier  // (op 1178) 



s_waitcnt vmcnt(17)
ds_write_b64 v18, v[42:43] offset:4224 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v18, v[44:45] offset:4232 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v18, v[46:47] offset:4240 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v18, v[48:49] offset:4248 // (op 6) Store local data 
s_barrier  // (op 1218) 



s_waitcnt vmcnt(13)
ds_write_b8 v90, v50 offset:6272 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v90, v51 offset:6288 // (op 21) Store local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1207) 



ds_read_b128 v[78:81], v5 // (op 75) Load local data 
ds_read_b128 v[82:85], v5 offset:64 // (op 75) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1248) 



ds_read_u8 v12, v6 offset:4096 // (op 85) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1187) 



ds_read_b64_tr_b4 v[86:87], v3 offset:4224 // (op 70) Transpose load local data 
ds_read_b64_tr_b4 v[88:89], v3 offset:4352 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1228) 


ds_read_u8 v14, v4 offset:6272 // (op 80) Load local data 
s_ashr_i64 s[0:1], s[24:25], 8 // (op 58) 
s_mov_b32 s6, s0 // (op 58) convert

s_and_b32 s7, s6, 4294967294 // (op 58) 

s_cmp_lt_u32 s51, s7 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58 if true)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58:

















s_mul_hi_i32 s51, s16, s9 // (op 533) 
v_lshlrev_b32 v13, 2, v1 // (op 533) 
s_add_i32 s0, s51, s16 // (op 533) 

s_ashr_i32 s1, s0, 31 // (op 533) 
s_cmp_eq_i32 s9, 0 // (op 533) 
s_mov_b32 s51, scc // (op 533) 
s_cmp_eq_i32 s51, 0 // (op 533) 
s_cselect_b32 s6, 0, -1 // (op 533) 

s_and_b32 s7, s10, 31 // (op 533) 
s_lshl_b32 s51, 1, s7 // (op 533) 
s_add_i32 s18, s51, s6 // (op 533) 


s_and_b32 s6, s1, s18 // (op 533) 


s_add_i32 s1, s0, s6 // (op 533) 


s_ashr_i32 s0, s1, s7 // (op 533) 


s_lshl_b32 s1, s10, 24 // (op 533) 
s_ashr_i32 s51, s1, 31 // (op 533) 

s_xor_b32 s1, s0, s51 // (op 533) 

s_sub_i32 s0, s1, s51 // (op 533) 


s_mul_hi_i32 s1, s0, s11 // (op 533) 
s_add_i32 s51, s1, s0 // (op 533) 

s_ashr_i32 s1, s51, 31 // (op 533) 
s_cmp_eq_i32 s11, 0 // (op 533) 
s_mov_b32 s6, scc // (op 533) 
s_cmp_eq_i32 s6, 0 // (op 533) 
s_cselect_b32 s7, 0, -1 // (op 533) 

s_and_b32 s6, s12, 31 // (op 533) 
s_lshl_b32 s18, 1, s6 // (op 533) 
s_add_i32 s19, s18, s7 // (op 533) 


s_and_b32 s7, s1, s19 // (op 533) 


s_add_i32 s1, s51, s7 // (op 533) 


s_ashr_i32 s7, s1, s6 // (op 533) 


s_lshl_b32 s1, s12, 24 // (op 533) 
s_ashr_i32 s6, s1, 31 // (op 533) 

s_xor_b32 s1, s7, s6 // (op 533) 

s_sub_i32 s7, s1, s6 // (op 533) 


s_mul_i32 s1, s0, s14 // (op 533) 

s_sub_u32 s0, s16, s1 // (op 533) 

s_mul_i32 s1, s7, s14 // (op 533) 
s_add_u32 s1, s1, s0 // (op 533) 


s_mul_hi_i32 s0, s16, s8 // (op 533) 
s_add_i32 s51, s0, s16 // (op 533) 

s_ashr_i32 s0, s51, 31 // (op 533) 
s_cmp_eq_i32 s8, 0 // (op 533) 
s_mov_b32 s6, scc // (op 533) 
s_cmp_eq_i32 s6, 0 // (op 533) 
s_cselect_b32 s7, 0, -1 // (op 533) 

s_and_b32 s6, s15, 31 // (op 533) 
s_lshl_b32 s18, 1, s6 // (op 533) 
s_add_i32 s19, s18, s7 // (op 533) 


s_and_b32 s7, s0, s19 // (op 533) 


s_add_i32 s0, s51, s7 // (op 533) 


s_ashr_i32 s7, s0, s6 // (op 533) 


s_lshl_b32 s0, s15, 24 // (op 533) 
s_ashr_i32 s6, s0, 31 // (op 533) 

s_xor_b32 s0, s7, s6 // (op 533) 

s_sub_i32 s7, s0, s6 // (op 533) 


s_add_u32 s18, s4, 16 // (op 533) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 533) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 533) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 533) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 533) 

s_mul_i32 s0, s14, s18 // (op 533) 

s_mul_i32 s6, s50, s0 // (op 533) 

s_mul_i32 s0, s7, s6 // (op 533) 

s_sub_u32 s6, s16, s0 // (op 533) 

s_mov_b32 s0, s6 // (op 533) 
s_mul_hi_i32 s51, s0, s13 // (op 533) 
s_add_i32 s18, s51, s0 // (op 533) 


s_ashr_i32 s0, s18, 31 // (op 533) 
s_cmp_eq_i32 s13, 0 // (op 533) 
s_mov_b32 s19, scc // (op 533) 
s_cmp_eq_i32 s19, 0 // (op 533) 
s_cselect_b32 s51, 0, -1 // (op 533) 

s_and_b32 s19, s49, 31 // (op 533) 
s_lshl_b32 s68, 1, s19 // (op 533) 
s_add_i32 s69, s68, s51 // (op 533) 


s_and_b32 s51, s0, s69 // (op 533) 


s_add_i32 s0, s18, s51 // (op 533) 


s_ashr_i32 s18, s0, s19 // (op 533) 


s_lshl_b32 s0, s49, 24 // (op 533) 
s_ashr_i32 s19, s0, 31 // (op 533) 

s_xor_b32 s0, s18, s19 // (op 533) 

s_sub_i32 s18, s0, s19 // (op 533) 


s_mul_i32 s0, s48, s14 // (op 533) 
s_add_u32 s68, s20, 16 // (op 533) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 533) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 533) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 533) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 533) 

s_sub_i32 s19, s68, s0 // (op 533) 

s_mul_i32 s51, s18, s19 // (op 533) 


s_sub_u32 s18, s6, s51 // (op 533) 


s_add_u32 s6, s18, s0 // (op 533) 


s_cmp_eq_i32 s7, 0 // (op 533) 
s_mov_b32 s0, scc // (op 533) 

s_cmp_eq_i32 s0, 0 // (op 533) 
s_cselect_b32 s7, s6, s1 // (op 533) 



s_lshl_b32 s6, s7, 4 // (op 533) 

s_mov_b32 s1, 0 // (op 533) convert
s_mov_b32 s0, s6 // (op 533) convert

s_ashr_i64 s[6:7], s[24:25], 8 // (op 533) 
s_lshr_b32 s51, s6, 1 // (op 533) 

v_lshl_add_u32 v15, s51, 9, v13 // (op 533) 


v_mul_lo_u32 v17, v15, s3 // (op 533) most significant: low of low * high
v_mul_hi_u32 v13, v15, s2 // (op 533) most significant: high of low * low
v_mul_lo_u32 v16, v15, s2 // (op 533) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 533) most significant: sum


v_mov_b32 v13, s0 // (op 533) 
v_mov_b32 v15, s1 // (op 533) 
v_add_co_u32 v18, s[6:7], v13, v16 // (op 533) least significant half
v_addc_co_u32 v19, s[6:7], v15, v17, s[6:7] // (op 533) most significant half



v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 533) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 533) 

v_mov_b32 v13, v16 // (op 533) convert

s_mul_i32 s1, s2, s25 // (op 535) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 535) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 535) least significant
s_mul_hi_u32 s6, s2, s24 // (op 535) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 535) most significant: sum
s_add_u32 s1, s1, s6 // (op 535) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 535) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 535) 

s_mov_b32 s52, s22 // (op 535) reg expression
s_mov_b32 s53, s23 // (op 535) reg expression
s_mov_b32 s54, s0 // (op 535) reg expression
s_mov_b32 s55, 131072 // (op 535) call()

s_mul_hi_i32 s51, s16, s9 // (op 600) 
v_lshrrev_b32 v15, 4, v1 // (op 600) 
s_add_i32 s0, s51, s16 // (op 600) 

v_lshlrev_b32 v16, 2, v15 // (op 600) 

s_ashr_i32 s1, s0, 31 // (op 600) 
v_and_b32 v15, -16, v1 // (op 600) 
s_cmp_eq_i32 s9, 0 // (op 600) 
s_mov_b32 s51, scc // (op 600) 
v_sub_u32 v17, v1, v15 // (op 600) 

s_cmp_eq_i32 s51, 0 // (op 600) 
s_cselect_b32 s6, 0, -1 // (op 600) 

v_lshlrev_b32 v15, 4, v17 // (op 600) 

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


s_add_u32 s18, s4, 16 // (op 600) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 600) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 600) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 600) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 600) 

s_mul_i32 s1, s7, s18 // (op 600) 

s_sub_i32 s51, s0, s1 // (op 600) 


s_mul_hi_i32 s0, s16, s8 // (op 600) 
s_add_i32 s1, s0, s16 // (op 600) 

s_ashr_i32 s0, s1, 31 // (op 600) 
s_cmp_eq_i32 s8, 0 // (op 600) 
s_mov_b32 s6, scc // (op 600) 
s_cmp_eq_i32 s6, 0 // (op 600) 
s_cselect_b32 s7, 0, -1 // (op 600) 

s_and_b32 s6, s15, 31 // (op 600) 
s_lshl_b32 s68, 1, s6 // (op 600) 
s_add_i32 s69, s68, s7 // (op 600) 


s_and_b32 s7, s0, s69 // (op 600) 


s_add_i32 s0, s1, s7 // (op 600) 


s_ashr_i32 s1, s0, s6 // (op 600) 


s_lshl_b32 s0, s15, 24 // (op 600) 
s_ashr_i32 s6, s0, 31 // (op 600) 

s_xor_b32 s0, s1, s6 // (op 600) 

s_sub_i32 s1, s0, s6 // (op 600) 


s_mul_i32 s0, s14, s18 // (op 600) 

s_mul_i32 s6, s50, s0 // (op 600) 

s_mul_i32 s0, s1, s6 // (op 600) 

s_sub_u32 s6, s16, s0 // (op 600) 

s_mul_hi_i32 s0, s6, s13 // (op 600) 
s_add_i32 s7, s0, s6 // (op 600) 


s_ashr_i32 s0, s7, 31 // (op 600) 
s_cmp_eq_i32 s13, 0 // (op 600) 
s_mov_b32 s6, scc // (op 600) 
s_cmp_eq_i32 s6, 0 // (op 600) 
s_cselect_b32 s18, 0, -1 // (op 600) 

s_and_b32 s6, s49, 31 // (op 600) 
s_lshl_b32 s19, 1, s6 // (op 600) 
s_add_i32 s68, s19, s18 // (op 600) 


s_and_b32 s18, s0, s68 // (op 600) 


s_add_i32 s0, s7, s18 // (op 600) 


s_ashr_i32 s7, s0, s6 // (op 600) 


s_lshl_b32 s0, s49, 24 // (op 600) 
s_ashr_i32 s6, s0, 31 // (op 600) 

s_xor_b32 s0, s7, s6 // (op 600) 

s_sub_i32 s7, s0, s6 // (op 600) 


s_cmp_eq_i32 s1, 0 // (op 600) 
s_mov_b32 s0, scc // (op 600) 

s_cmp_eq_i32 s0, 0 // (op 600) 
s_cselect_b32 s1, s7, s51 // (op 600) 



v_lshl_add_u32 v17, s1, 4, v16 // (op 600) 


v_mul_lo_u32 v16, v17, s26 // (op 600) 

s_ashr_i64 s[0:1], s[24:25], 8 // (op 600) 
s_lshr_b32 s51, s0, 1 // (op 600) 

v_lshl_add_u32 v17, s51, 9, v15 // (op 600) 


v_add_u32 v15, v17, v16 // (op 600) 


s_mul_i32 s1, s26, s5 // (op 602) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 602) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 602) least significant
s_mul_hi_u32 s6, s26, s4 // (op 602) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 602) most significant: sum
s_add_u32 s1, s1, s6 // (op 602) most significant: sum


s_mov_b32 s56, s28 // (op 602) reg expression
s_mov_b32 s57, s29 // (op 602) reg expression
s_mov_b32 s58, s0 // (op 602) reg expression
s_mov_b32 s59, 131072 // (op 602) call()

s_mul_hi_i32 s51, s16, s9 // (op 664) 
v_and_b32 v16, -16, v1 // (op 664) 
s_add_i32 s0, s51, s16 // (op 664) 

v_sub_u32 v17, v1, v16 // (op 664) 

s_ashr_i32 s1, s0, 31 // (op 664) 
v_lshrrev_b32 v16, 4, v1 // (op 664) 
s_cmp_eq_i32 s9, 0 // (op 664) 
s_mov_b32 s51, scc // (op 664) 
v_lshlrev_b32 v18, 1, v16 // (op 664) 

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


s_mul_i32 s1, s0, s14 // (op 664) 

s_sub_u32 s0, s16, s1 // (op 664) 

s_mul_i32 s1, s7, s14 // (op 664) 
s_add_u32 s1, s1, s0 // (op 664) 


s_mul_hi_i32 s0, s16, s8 // (op 664) 
s_add_i32 s51, s0, s16 // (op 664) 

s_ashr_i32 s0, s51, 31 // (op 664) 
s_cmp_eq_i32 s8, 0 // (op 664) 
s_mov_b32 s6, scc // (op 664) 
s_cmp_eq_i32 s6, 0 // (op 664) 
s_cselect_b32 s7, 0, -1 // (op 664) 

s_and_b32 s6, s15, 31 // (op 664) 
s_lshl_b32 s18, 1, s6 // (op 664) 
s_add_i32 s19, s18, s7 // (op 664) 


s_and_b32 s7, s0, s19 // (op 664) 


s_add_i32 s0, s51, s7 // (op 664) 


s_ashr_i32 s7, s0, s6 // (op 664) 


s_lshl_b32 s0, s15, 24 // (op 664) 
s_ashr_i32 s6, s0, 31 // (op 664) 

s_xor_b32 s0, s7, s6 // (op 664) 

s_sub_i32 s7, s0, s6 // (op 664) 


s_add_u32 s18, s4, 16 // (op 664) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 664) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 664) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 664) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 664) 

s_mul_i32 s0, s14, s18 // (op 664) 

s_mul_i32 s6, s50, s0 // (op 664) 

s_mul_i32 s0, s7, s6 // (op 664) 

s_sub_u32 s6, s16, s0 // (op 664) 

s_mov_b32 s0, s6 // (op 664) 
s_mul_hi_i32 s51, s0, s13 // (op 664) 
s_add_i32 s18, s51, s0 // (op 664) 


s_ashr_i32 s0, s18, 31 // (op 664) 
s_cmp_eq_i32 s13, 0 // (op 664) 
s_mov_b32 s19, scc // (op 664) 
s_cmp_eq_i32 s19, 0 // (op 664) 
s_cselect_b32 s51, 0, -1 // (op 664) 

s_and_b32 s19, s49, 31 // (op 664) 
s_lshl_b32 s68, 1, s19 // (op 664) 
s_add_i32 s69, s68, s51 // (op 664) 


s_and_b32 s51, s0, s69 // (op 664) 


s_add_i32 s0, s18, s51 // (op 664) 


s_ashr_i32 s18, s0, s19 // (op 664) 


s_lshl_b32 s0, s49, 24 // (op 664) 
s_ashr_i32 s19, s0, 31 // (op 664) 

s_xor_b32 s0, s18, s19 // (op 664) 

s_sub_i32 s18, s0, s19 // (op 664) 


s_mul_i32 s0, s48, s14 // (op 664) 
s_add_u32 s68, s20, 16 // (op 664) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 664) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 664) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 664) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 664) 

s_sub_i32 s19, s68, s0 // (op 664) 

s_mul_i32 s51, s18, s19 // (op 664) 


s_sub_u32 s18, s6, s51 // (op 664) 


s_add_u32 s6, s18, s0 // (op 664) 


s_cmp_eq_i32 s7, 0 // (op 664) 
s_mov_b32 s0, scc // (op 664) 

s_cmp_eq_i32 s0, 0 // (op 664) 
s_cselect_b32 s7, s6, s1 // (op 664) 



v_lshl_add_u32 v16, s7, 4, v17 // (op 664) 


s_ashr_i64 s[0:1], s[24:25], 8 // (op 664) 
s_lshr_b32 s51, s0, 1 // (op 664) 

v_lshl_add_u32 v17, s51, 4, v18 // (op 664) 


v_mul_lo_u32 v18, v17, s30 // (op 664) 

v_add_u32 v17, v16, v18 // (op 664) 


s_mul_i32 s1, s30, s35 // (op 666) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 666) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 666) least significant
s_mul_hi_u32 s6, s30, s34 // (op 666) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 666) most significant: sum
s_add_u32 s1, s1, s6 // (op 666) most significant: sum


s_mov_b32 s60, s32 // (op 666) reg expression
s_mov_b32 s61, s33 // (op 666) reg expression
s_mov_b32 s62, s0 // (op 666) reg expression
s_mov_b32 s63, 131072 // (op 666) call()

s_mul_hi_i32 s51, s16, s9 // (op 728) 
v_lshrrev_b32 v16, 3, v1 // (op 728) 
s_add_i32 s0, s51, s16 // (op 728) 

v_lshlrev_b32 v18, 1, v16 // (op 728) 

s_ashr_i32 s1, s0, 31 // (op 728) 
v_and_b32 v16, -8, v1 // (op 728) 
s_cmp_eq_i32 s9, 0 // (op 728) 
s_mov_b32 s51, scc // (op 728) 
v_sub_u32 v19, v1, v16 // (op 728) 

s_cmp_eq_i32 s51, 0 // (op 728) 
s_cselect_b32 s6, 0, -1 // (op 728) 

s_and_b32 s7, s10, 31 // (op 728) 
s_lshl_b32 s51, 1, s7 // (op 728) 
s_add_i32 s18, s51, s6 // (op 728) 


s_and_b32 s6, s1, s18 // (op 728) 


s_add_i32 s1, s0, s6 // (op 728) 


s_ashr_i32 s0, s1, s7 // (op 728) 


s_lshl_b32 s1, s10, 24 // (op 728) 
s_ashr_i32 s51, s1, 31 // (op 728) 

s_xor_b32 s1, s0, s51 // (op 728) 

s_sub_i32 s0, s1, s51 // (op 728) 


s_mul_hi_i32 s1, s0, s11 // (op 728) 
s_add_i32 s51, s1, s0 // (op 728) 

s_ashr_i32 s1, s51, 31 // (op 728) 
s_cmp_eq_i32 s11, 0 // (op 728) 
s_mov_b32 s6, scc // (op 728) 
s_cmp_eq_i32 s6, 0 // (op 728) 
s_cselect_b32 s7, 0, -1 // (op 728) 

s_and_b32 s6, s12, 31 // (op 728) 
s_lshl_b32 s18, 1, s6 // (op 728) 
s_add_i32 s19, s18, s7 // (op 728) 


s_and_b32 s7, s1, s19 // (op 728) 


s_add_i32 s1, s51, s7 // (op 728) 


s_ashr_i32 s7, s1, s6 // (op 728) 


s_lshl_b32 s1, s12, 24 // (op 728) 
s_ashr_i32 s6, s1, 31 // (op 728) 

s_xor_b32 s1, s7, s6 // (op 728) 

s_sub_i32 s7, s1, s6 // (op 728) 


s_add_u32 s18, s4, 16 // (op 728) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 728) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 728) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 728) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 728) 

s_mul_i32 s1, s7, s18 // (op 728) 

s_sub_i32 s51, s0, s1 // (op 728) 


s_mul_hi_i32 s0, s16, s8 // (op 728) 
s_add_i32 s1, s0, s16 // (op 728) 

s_ashr_i32 s0, s1, 31 // (op 728) 
s_cmp_eq_i32 s8, 0 // (op 728) 
s_mov_b32 s6, scc // (op 728) 
s_cmp_eq_i32 s6, 0 // (op 728) 
s_cselect_b32 s7, 0, -1 // (op 728) 

s_and_b32 s6, s15, 31 // (op 728) 
s_lshl_b32 s68, 1, s6 // (op 728) 
s_add_i32 s69, s68, s7 // (op 728) 


s_and_b32 s7, s0, s69 // (op 728) 


s_add_i32 s0, s1, s7 // (op 728) 


s_ashr_i32 s1, s0, s6 // (op 728) 


s_lshl_b32 s0, s15, 24 // (op 728) 
s_ashr_i32 s6, s0, 31 // (op 728) 

s_xor_b32 s0, s1, s6 // (op 728) 

s_sub_i32 s1, s0, s6 // (op 728) 


s_mul_i32 s0, s14, s18 // (op 728) 

s_mul_i32 s6, s50, s0 // (op 728) 

s_mul_i32 s0, s1, s6 // (op 728) 

s_sub_u32 s6, s16, s0 // (op 728) 

s_mul_hi_i32 s0, s6, s13 // (op 728) 
s_add_i32 s7, s0, s6 // (op 728) 


s_ashr_i32 s0, s7, 31 // (op 728) 
s_cmp_eq_i32 s13, 0 // (op 728) 
s_mov_b32 s6, scc // (op 728) 
s_cmp_eq_i32 s6, 0 // (op 728) 
s_cselect_b32 s18, 0, -1 // (op 728) 

s_and_b32 s6, s49, 31 // (op 728) 
s_lshl_b32 s19, 1, s6 // (op 728) 
s_add_i32 s68, s19, s18 // (op 728) 


s_and_b32 s18, s0, s68 // (op 728) 


s_add_i32 s0, s7, s18 // (op 728) 


s_ashr_i32 s7, s0, s6 // (op 728) 


s_lshl_b32 s0, s49, 24 // (op 728) 
s_ashr_i32 s6, s0, 31 // (op 728) 

s_xor_b32 s0, s7, s6 // (op 728) 

s_sub_i32 s7, s0, s6 // (op 728) 


s_cmp_eq_i32 s1, 0 // (op 728) 
s_mov_b32 s0, scc // (op 728) 

s_cmp_eq_i32 s0, 0 // (op 728) 
s_cselect_b32 s1, s7, s51 // (op 728) 



v_lshl_add_u32 v16, s1, 4, v18 // (op 728) 


v_mul_lo_u32 v18, v16, s36 // (op 728) 

s_ashr_i64 s[0:1], s[24:25], 8 // (op 728) 
s_lshr_b32 s51, s0, 1 // (op 728) 

v_lshl_add_u32 v16, s51, 4, v19 // (op 728) 


v_add_u32 v19, v16, v18 // (op 728) 


s_mul_i32 s1, s36, s5 // (op 730) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 730) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 730) least significant
s_mul_hi_u32 s6, s36, s4 // (op 730) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 730) most significant: sum
s_add_u32 s1, s1, s6 // (op 730) most significant: sum


s_mov_b32 s64, s38 // (op 730) reg expression
s_mov_b32 s65, s39 // (op 730) reg expression
s_mov_b32 s66, s0 // (op 730) reg expression
s_mov_b32 s67, 131072 // (op 730) call()

v_lshlrev_b32 v16, 8, v1 // (op 821) 
v_lshrrev_b32 v18, 3, v16 // (op 821) 

v_and_b32 v16, -16, v1 // (op 874) 
v_sub_u32 v20, v1, v16 // (op 874) 

v_lshlrev_b32 v16, 4, v20 // (op 874) 

v_lshrrev_b32 v20, 4, v1 // (op 874) 
v_lshl_add_u32 v21, v20, 10, v16 // (op 874) 


v_and_b32 v16, -16, v1 // (op 927) 
v_sub_u32 v20, v1, v16 // (op 927) 

v_lshrrev_b32 v16, 4, v1 // (op 927) 
v_lshl_add_u32 v24, v16, 5, v20 // (op 927) 


v_and_b32 v16, -8, v1 // (op 980) 
v_sub_u32 v20, v1, v16 // (op 980) 

v_lshrrev_b32 v16, 3, v1 // (op 980) 
v_lshl_add_u32 v25, v16, 4, v20 // (op 980) 


s_ashr_i64 s[0:1], s[24:25], 8 // (op 194) 
s_and_b32 s51, s0, 4294967294 // (op 194) 

s_ashr_i64 s[0:1], s[24:25], 8 // (op 193) 
s_mov_b32 s6, s0 // (op 193) convert

s_cmp_lt_u32 s51, s6 // (op 193) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193 // (op 193) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193:

s_add_u32 s51, s51, 1 // (op 195) 
s_lshl_b64 s[0:1], s[2:3], 2 // (op 198) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 198) 


v_mov_b32 v12, v13 // (op 198) 
v_mov_b32 v14, v12 // (op 198) reg expression
buffer_load_dwordx2 v[26:27], v14, s[52:55], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v22 // (op 198) 
buffer_load_dwordx2 v[28:29], v14, s[52:55], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v22 // (op 198) 
buffer_load_dwordx2 v[30:31], v14, s[52:55], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v22 // (op 198) 
buffer_load_dwordx2 v[32:33], v14, s[52:55], 0 offen // (op 198) Load value



s_lshl_b64 s[0:1], s[2:3], 8 // (op 523) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 523) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 523) 

v_add_u32 v13, v13, s0 // (op 523) 


s_barrier  // (op 1181) 
v_mov_b64 v[22:23], s[26:27] // (op 221) convert




v_mov_b32 v12, v15 // (op 221) 
v_mov_b32 v14, v12 // (op 221) 
buffer_load_dwordx4 v[34:37], v14, s[56:59], 0 offen // (op 221) Load value
v_add_u32 v14, v14, v22 // (op 221) 
buffer_load_dwordx4 v[38:41], v14, s[56:59], 0 offen // (op 221) Load value
v_add_u32 v14, v14, v22 // (op 221) 
s_waitcnt vmcnt(5)
ds_write_b64 v18, v[26:27] offset:4224 // (op 199) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v18, v[28:29] offset:4232 // (op 199) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v18, v[30:31] offset:4240 // (op 199) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v18, v[32:33] offset:4248 // (op 199) Store local data 
buffer_load_dwordx4 v[42:45], v14, s[56:59], 0 offen // (op 221) Load value

v_add_u32 v14, v14, v22 // (op 221) 
buffer_load_dwordx4 v[46:49], v14, s[56:59], 0 offen // (op 221) Load value



v_add_u32 v15, 256, v15 // (op 592) 

s_barrier  // (op 1200) 
v_mov_b64 v[22:23], s[30:31] // (op 222) convert




v_mov_b32 v12, v17 // (op 222) 
v_mov_b32 v14, v12 // (op 222) reg expression
buffer_load_ubyte v26, v14, s[60:63], 0 offen offset: 0 // (op 222) Load value
v_add_u32 v14, v14, v22 // (op 222) 
s_waitcnt vmcnt(4)
ds_write_b128 v21, v[34:37] // (op 218) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v21, v[38:41] offset:256 // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v21, v[42:45] offset:512 // (op 218) Store local data 
buffer_load_ubyte v27, v14, s[60:63], 0 offen offset: 0 // (op 222) Load value



s_lshl_b32 s0, s30, 3 // (op 656) 
v_add_u32 v17, v17, s0 // (op 656) 

s_waitcnt vmcnt(2)
ds_write_b128 v21, v[46:49] offset:768 // (op 218) Store local data 


v_mov_b64 v[22:23], s[36:37] // (op 223) convert

s_barrier  // (op 1221) 
v_mov_b32 v12, v19 // (op 223) 



v_mov_b32 v14, v12 // (op 223) 
buffer_load_ubyte v28, v14, s[64:67], 0 offen offset: 0 // (op 223) Load value
v_add_u32 v14, v14, v22 // (op 223) 
buffer_load_ubyte v29, v14, s[64:67], 0 offen offset: 0 // (op 223) Load value



v_add_u32 v19, 8, v19 // (op 720) 
s_waitcnt vmcnt(3)
ds_write_b8 v24, v26 offset:6272 // (op 219) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1190) 
s_waitcnt vmcnt(2)
ds_write_b8 v24, v27 offset:6288 // (op 219) Store local data 




ds_read_b64_tr_b4 v[30:31], v3 offset:4224 // (op 201) Transpose load local data 
s_barrier  // (op 1241) 
ds_read_b64_tr_b4 v[32:33], v3 offset:4352 // (op 201) Transpose load local data 



s_waitcnt lgkmcnt(4)
s_barrier  // (op 1211) 
s_waitcnt vmcnt(1)
ds_write_b8 v25, v28 offset:4096 // (op 220) Store local data 



ds_read_b128 v[34:37], v5 // (op 203) Load local data 
ds_read_b128 v[38:41], v5 offset:64 // (op 203) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v25, v29 offset:4104 // (op 220) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 1231) 


ds_read_u8 v12, v4 offset:6272 // (op 205) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1251) 



ds_read_u8 v14, v6 offset:4096 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:33], v[34:41], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 208) 







ds_read_b64_tr_b4 v[26:27], v3 offset:5248 // (op 210) Transpose load local data 
ds_read_b64_tr_b4 v[28:29], v3 offset:5376 // (op 210) Transpose load local data 



ds_read_b128 v[30:33], v5 offset:128 // (op 212) Load local data 
ds_read_b128 v[34:37], v5 offset:192 // (op 212) Load local data 


ds_read_u8 v12, v4 offset:6336 // (op 214) Load local data 



ds_read_u8 v14, v6 offset:4100 // (op 216) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:37], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 217) 




s_ashr_i64 s[0:1], s[24:25], 8 // (op 193) 
s_mov_b32 s6, s0 // (op 193) convert

s_cmp_lt_u32 s51, s6 // (op 193) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193 // (op 193) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193 if true)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193:























s_barrier  // (op 1328) 








v_cmp_eq_f32 vcc, 0.00000, v0 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v11, v4 // (op 50) 
v_mul_f32 v13, v11, v5 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 

s_mul_hi_i32 s0, s16, s9 // (op 1022) 
v_lshrrev_b32 v3, 6, v1 // (op 1022) 
s_add_i32 s1, s0, s16 // (op 1022) 

v_and_b32 v4, 63, v1 // (op 1022) 
s_ashr_i32 s0, s1, 31 // (op 1022) 
v_lshrrev_b32 v5, 4, v4 // (op 1022) 
s_cmp_eq_i32 s9, 0 // (op 1022) 
s_mov_b32 s2, scc // (op 1022) 
v_lshlrev_b32 v6, 2, v5 // (op 1022) 

s_cmp_eq_i32 s2, 0 // (op 1022) 
s_cselect_b32 s3, 0, -1 // (op 1022) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 1022) 


s_and_b32 s2, s10, 31 // (op 1022) 
v_lshrrev_b32 v3, 2, v4 // (op 1022) 

s_lshl_b32 s6, 1, s2 // (op 1022) 
v_and_b32 v4, 3, v3 // (op 1022) 

s_add_i32 s7, s6, s3 // (op 1022) 


v_and_b32 v3, 3, v1 // (op 1022) 
s_and_b32 s3, s0, s7 // (op 1022) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 1022) 


s_add_i32 s0, s1, s3 // (op 1022) 


s_ashr_i32 s1, s0, s2 // (op 1022) 


s_lshl_b32 s0, s10, 24 // (op 1022) 
s_ashr_i32 s2, s0, 31 // (op 1022) 

s_xor_b32 s0, s1, s2 // (op 1022) 

s_sub_i32 s1, s0, s2 // (op 1022) 


s_mul_hi_i32 s0, s1, s11 // (op 1022) 
s_add_i32 s2, s0, s1 // (op 1022) 

s_ashr_i32 s0, s2, 31 // (op 1022) 
s_cmp_eq_i32 s11, 0 // (op 1022) 
s_mov_b32 s3, scc // (op 1022) 
s_cmp_eq_i32 s3, 0 // (op 1022) 
s_cselect_b32 s6, 0, -1 // (op 1022) 

s_and_b32 s3, s12, 31 // (op 1022) 
s_lshl_b32 s7, 1, s3 // (op 1022) 
s_add_i32 s18, s7, s6 // (op 1022) 


s_and_b32 s19, s0, s18 // (op 1022) 


s_add_i32 s0, s2, s19 // (op 1022) 


s_ashr_i32 s2, s0, s3 // (op 1022) 


s_lshl_b32 s0, s12, 24 // (op 1022) 
s_ashr_i32 s3, s0, 31 // (op 1022) 

s_xor_b32 s0, s2, s3 // (op 1022) 

s_sub_i32 s2, s0, s3 // (op 1022) 


s_mul_i32 s0, s1, s14 // (op 1022) 
s_sub_u32 s3, s16, s0 // (op 1022) 

s_mul_i32 s0, s2, s14 // (op 1022) 
s_add_u32 s0, s0, s3 // (op 1022) 

s_mul_hi_i32 s3, s16, s8 // (op 1022) 
s_add_i32 s6, s3, s16 // (op 1022) 

s_ashr_i32 s3, s6, 31 // (op 1022) 
s_cmp_eq_i32 s8, 0 // (op 1022) 
s_mov_b32 s7, scc // (op 1022) 
s_cmp_eq_i32 s7, 0 // (op 1022) 
s_cselect_b32 s18, 0, -1 // (op 1022) 

s_and_b32 s7, s15, 31 // (op 1022) 
s_lshl_b32 s19, 1, s7 // (op 1022) 
s_add_i32 s22, s19, s18 // (op 1022) 


s_and_b32 s18, s3, s22 // (op 1022) 


s_add_i32 s3, s6, s18 // (op 1022) 


s_ashr_i32 s6, s3, s7 // (op 1022) 


s_lshl_b32 s3, s15, 24 // (op 1022) 
s_ashr_i32 s7, s3, 31 // (op 1022) 

s_xor_b32 s3, s6, s7 // (op 1022) 

s_sub_i32 s6, s3, s7 // (op 1022) 


s_add_u32 s18, s4, 16 // (op 1022) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 1022) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 1022) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 1022) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 4 // (op 1022) 

s_mul_i32 s3, s14, s18 // (op 1022) 
s_mul_i32 s7, s50, s3 // (op 1022) 

s_mul_i32 s3, s6, s7 // (op 1022) 

s_sub_u32 s7, s16, s3 // (op 1022) 

s_mov_b32 s3, s7 // (op 1022) 
s_mul_hi_i32 s22, s3, s13 // (op 1022) 
s_add_i32 s23, s22, s3 // (op 1022) 


s_ashr_i32 s3, s23, 31 // (op 1022) 
s_cmp_eq_i32 s13, 0 // (op 1022) 
s_mov_b32 s22, scc // (op 1022) 
s_cmp_eq_i32 s22, 0 // (op 1022) 
s_cselect_b32 s24, 0, -1 // (op 1022) 

s_and_b32 s22, s49, 31 // (op 1022) 
s_lshl_b32 s25, 1, s22 // (op 1022) 
s_add_i32 s26, s25, s24 // (op 1022) 


s_and_b32 s24, s3, s26 // (op 1022) 


s_add_i32 s3, s23, s24 // (op 1022) 


s_ashr_i32 s23, s3, s22 // (op 1022) 


s_lshl_b32 s3, s49, 24 // (op 1022) 
s_ashr_i32 s22, s3, 31 // (op 1022) 

s_xor_b32 s3, s23, s22 // (op 1022) 

s_sub_i32 s23, s3, s22 // (op 1022) 


s_mul_i32 s3, s48, s14 // (op 1022) 
s_add_u32 s24, s20, 16 // (op 1022) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1022) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 1022) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 1022) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 4 // (op 1022) 

s_sub_i32 s22, s24, s3 // (op 1022) 

s_mul_i32 s24, s23, s22 // (op 1022) 

s_sub_u32 s22, s7, s24 // (op 1022) 


s_add_u32 s7, s22, s3 // (op 1022) 


s_cmp_eq_i32 s6, 0 // (op 1022) 
s_mov_b32 s3, scc // (op 1022) 

s_cmp_eq_i32 s3, 0 // (op 1022) 
s_cselect_b32 s6, s7, s0 // (op 1022) 


s_mul_i32 s0, s2, s18 // (op 1022) 


v_lshl_add_u32 v3, s6, 4, v5 // (op 1022) 


s_sub_i32 s2, s1, s0 // (op 1022) 


s_cmp_eq_i32 s3, 0 // (op 1022) 
s_cselect_b32 s22, s23, s2 // (op 1022) 



v_lshl_add_u32 v4, s22, 4, v6 // (op 1022) 


v_mul_lo_u32 v5, v4, s40 // (op 1022) 

v_add_lshl_u32 v4, v3, v5, 2 // (op 1022) 


s_mul_i32 s7, s40, s5 // (op 1024) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 1024) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 1024) least significant
s_mul_hi_u32 s1, s40, s4 // (op 1024) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 1024) most significant: sum
s_add_u32 s7, s7, s1 // (op 1024) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 1024) 

s_mov_b32 s0, s42 // (op 1024) reg expression
s_mov_b32 s1, s43 // (op 1024) reg expression
s_mov_b32 s2, s18 // (op 1024) reg expression
s_mov_b32 s3, 131072 // (op 1024) call()




buffer_store_dwordx4 v[12:15], v4, s[0:3], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155:
s_mul_hi_i32 s6, s16, s9 // (op 791) 
v_lshrrev_b32 v3, 6, v1 // (op 791) 
s_add_i32 s7, s6, s16 // (op 791) 

v_and_b32 v4, 63, v1 // (op 791) 
s_ashr_i32 s6, s7, 31 // (op 791) 
v_lshrrev_b32 v5, 4, v4 // (op 791) 
s_cmp_eq_i32 s9, 0 // (op 791) 
s_mov_b32 s18, scc // (op 791) 
v_lshlrev_b32 v6, 2, v5 // (op 791) 

s_cmp_eq_i32 s18, 0 // (op 791) 
s_cselect_b32 s19, 0, -1 // (op 791) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 791) 


s_and_b32 s18, s10, 31 // (op 791) 
v_lshrrev_b32 v3, 2, v4 // (op 791) 

s_lshl_b32 s22, 1, s18 // (op 791) 
v_and_b32 v4, 3, v3 // (op 791) 

s_add_i32 s23, s22, s19 // (op 791) 


v_and_b32 v3, 3, v1 // (op 791) 
s_and_b32 s19, s6, s23 // (op 791) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 791) 


s_add_i32 s6, s7, s19 // (op 791) 


s_ashr_i32 s7, s6, s18 // (op 791) 


s_lshl_b32 s6, s10, 24 // (op 791) 
s_ashr_i32 s18, s6, 31 // (op 791) 

s_xor_b32 s6, s7, s18 // (op 791) 

s_sub_i32 s7, s6, s18 // (op 791) 


s_mul_hi_i32 s6, s7, s11 // (op 791) 
s_add_i32 s18, s6, s7 // (op 791) 

s_ashr_i32 s6, s18, 31 // (op 791) 
s_cmp_eq_i32 s11, 0 // (op 791) 
s_mov_b32 s19, scc // (op 791) 
s_cmp_eq_i32 s19, 0 // (op 791) 
s_cselect_b32 s22, 0, -1 // (op 791) 

s_and_b32 s19, s12, 31 // (op 791) 
s_lshl_b32 s23, 1, s19 // (op 791) 
s_add_i32 s24, s23, s22 // (op 791) 


s_and_b32 s22, s6, s24 // (op 791) 


s_add_i32 s6, s18, s22 // (op 791) 


s_ashr_i32 s18, s6, s19 // (op 791) 


s_lshl_b32 s6, s12, 24 // (op 791) 
s_ashr_i32 s19, s6, 31 // (op 791) 

s_xor_b32 s6, s18, s19 // (op 791) 

s_sub_i32 s18, s6, s19 // (op 791) 


s_mul_i32 s6, s7, s14 // (op 791) 
s_sub_u32 s19, s16, s6 // (op 791) 

s_mul_i32 s6, s18, s14 // (op 791) 
s_add_u32 s6, s6, s19 // (op 791) 

s_mul_hi_i32 s19, s16, s8 // (op 791) 
s_add_i32 s22, s19, s16 // (op 791) 

s_ashr_i32 s19, s22, 31 // (op 791) 
s_cmp_eq_i32 s8, 0 // (op 791) 
s_mov_b32 s23, scc // (op 791) 
s_cmp_eq_i32 s23, 0 // (op 791) 
s_cselect_b32 s24, 0, -1 // (op 791) 

s_and_b32 s23, s15, 31 // (op 791) 
s_lshl_b32 s25, 1, s23 // (op 791) 
s_add_i32 s26, s25, s24 // (op 791) 


s_and_b32 s24, s19, s26 // (op 791) 


s_add_i32 s19, s22, s24 // (op 791) 


s_ashr_i32 s22, s19, s23 // (op 791) 


s_lshl_b32 s19, s15, 24 // (op 791) 
s_ashr_i32 s23, s19, 31 // (op 791) 

s_xor_b32 s19, s22, s23 // (op 791) 

s_sub_i32 s22, s19, s23 // (op 791) 


s_add_u32 s38, s4, 16 // (op 791) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 791) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 791) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 791) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 791) 

s_mul_i32 s19, s14, s38 // (op 791) 
s_mul_i32 s23, s50, s19 // (op 791) 

s_mul_i32 s19, s22, s23 // (op 791) 

s_sub_u32 s23, s16, s19 // (op 791) 

s_mov_b32 s19, s23 // (op 791) 
s_mul_hi_i32 s24, s19, s13 // (op 791) 
s_add_i32 s25, s24, s19 // (op 791) 


s_ashr_i32 s19, s25, 31 // (op 791) 
s_cmp_eq_i32 s13, 0 // (op 791) 
s_mov_b32 s24, scc // (op 791) 
s_cmp_eq_i32 s24, 0 // (op 791) 
s_cselect_b32 s26, 0, -1 // (op 791) 

s_and_b32 s24, s49, 31 // (op 791) 
s_lshl_b32 s27, 1, s24 // (op 791) 
s_add_i32 s28, s27, s26 // (op 791) 


s_and_b32 s26, s19, s28 // (op 791) 


s_add_i32 s19, s25, s26 // (op 791) 


s_ashr_i32 s25, s19, s24 // (op 791) 


s_lshl_b32 s19, s49, 24 // (op 791) 
s_ashr_i32 s24, s19, 31 // (op 791) 

s_xor_b32 s19, s25, s24 // (op 791) 

s_sub_i32 s25, s19, s24 // (op 791) 


s_mul_i32 s19, s48, s14 // (op 791) 
s_add_u32 s26, s20, 16 // (op 791) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 791) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 791) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 791) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 791) 

s_sub_i32 s24, s26, s19 // (op 791) 

s_mul_i32 s26, s25, s24 // (op 791) 

s_sub_u32 s24, s23, s26 // (op 791) 


s_add_u32 s23, s24, s19 // (op 791) 


s_cmp_eq_i32 s22, 0 // (op 791) 
s_mov_b32 s19, scc // (op 791) 

s_cmp_eq_i32 s19, 0 // (op 791) 
s_cselect_b32 s22, s23, s6 // (op 791) 


s_mul_i32 s6, s18, s38 // (op 791) 


v_lshl_add_u32 v3, s22, 4, v5 // (op 791) 


s_sub_i32 s18, s7, s6 // (op 791) 


s_cmp_eq_i32 s19, 0 // (op 791) 
s_cselect_b32 s6, s25, s18 // (op 791) 



v_lshl_add_u32 v4, s6, 4, v6 // (op 791) 


v_mul_lo_u32 v5, v4, s44 // (op 791) 

v_add_lshl_u32 v4, v3, v5, 2 // (op 791) 


s_mul_i32 s7, s44, s5 // (op 793) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 793) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 793) least significant
s_mul_hi_u32 s19, s44, s4 // (op 793) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 793) most significant: sum
s_add_u32 s7, s7, s19 // (op 793) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 793) 

s_mov_b32 s36, s46 // (op 793) reg expression
s_mov_b32 s37, s47 // (op 793) reg expression
s_mov_b32 s38, s18 // (op 793) reg expression
s_mov_b32 s39, 131072 // (op 793) call()





buffer_load_dwordx4 v[6:9], v4, s[36:39], 0 offen // (op 167) Load value



v_accvgpr_read v12, a0 // (op 164) 
v_accvgpr_read v13, a1 // (op 164) 
v_accvgpr_read v14, a2 // (op 164) 
v_accvgpr_read v15, a3 // (op 164) 
v_mul_f32 v16, v11, v12 // (op 164) 
v_mul_f32 v17, v11, v13 // (op 164) 
v_mul_f32 v18, v11, v14 // (op 164) 
v_mul_f32 v19, v11, v15 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v0, v6 // (op 168) 
v_mul_f32 v13, v0, v7 // (op 168) 
v_mul_f32 v14, v0, v8 // (op 168) 
v_mul_f32 v15, v0, v9 // (op 168) 

v_add_f32 v4, v12, v16 // (op 165) 
v_add_f32 v5, v13, v17 // (op 165) 
v_add_f32 v6, v14, v18 // (op 165) 
v_add_f32 v7, v15, v19 // (op 165) 


s_mul_hi_i32 s6, s16, s9 // (op 1039) 
v_lshrrev_b32 v3, 6, v1 // (op 1039) 
s_add_i32 s7, s6, s16 // (op 1039) 

v_and_b32 v8, 63, v1 // (op 1039) 
s_ashr_i32 s6, s7, 31 // (op 1039) 
v_lshrrev_b32 v9, 4, v8 // (op 1039) 
s_cmp_eq_i32 s9, 0 // (op 1039) 
s_mov_b32 s18, scc // (op 1039) 
v_lshlrev_b32 v10, 2, v9 // (op 1039) 

s_cmp_eq_i32 s18, 0 // (op 1039) 
s_cselect_b32 s19, 0, -1 // (op 1039) 

v_lshl_add_u32 v9, v3, 4, v10 // (op 1039) 


s_and_b32 s18, s10, 31 // (op 1039) 
v_lshrrev_b32 v3, 2, v8 // (op 1039) 

s_lshl_b32 s22, 1, s18 // (op 1039) 
v_and_b32 v8, 3, v3 // (op 1039) 

s_add_i32 s23, s22, s19 // (op 1039) 


v_and_b32 v3, 3, v1 // (op 1039) 
s_and_b32 s19, s6, s23 // (op 1039) 


v_lshl_add_u32 v10, v8, 2, v3 // (op 1039) 


s_add_i32 s6, s7, s19 // (op 1039) 


s_ashr_i32 s7, s6, s18 // (op 1039) 


s_lshl_b32 s6, s10, 24 // (op 1039) 
s_ashr_i32 s18, s6, 31 // (op 1039) 

s_xor_b32 s6, s7, s18 // (op 1039) 

s_sub_i32 s7, s6, s18 // (op 1039) 


s_mul_hi_i32 s6, s7, s11 // (op 1039) 
s_add_i32 s18, s6, s7 // (op 1039) 

s_ashr_i32 s6, s18, 31 // (op 1039) 
s_cmp_eq_i32 s11, 0 // (op 1039) 
s_mov_b32 s19, scc // (op 1039) 
s_cmp_eq_i32 s19, 0 // (op 1039) 
s_cselect_b32 s22, 0, -1 // (op 1039) 

s_and_b32 s19, s12, 31 // (op 1039) 
s_lshl_b32 s23, 1, s19 // (op 1039) 
s_add_i32 s24, s23, s22 // (op 1039) 


s_and_b32 s22, s6, s24 // (op 1039) 


s_add_i32 s6, s18, s22 // (op 1039) 


s_ashr_i32 s18, s6, s19 // (op 1039) 


s_lshl_b32 s6, s12, 24 // (op 1039) 
s_ashr_i32 s19, s6, 31 // (op 1039) 

s_xor_b32 s6, s18, s19 // (op 1039) 

s_sub_i32 s18, s6, s19 // (op 1039) 


s_mul_i32 s6, s7, s14 // (op 1039) 
s_sub_u32 s19, s16, s6 // (op 1039) 

s_mul_i32 s6, s18, s14 // (op 1039) 
s_add_u32 s6, s6, s19 // (op 1039) 

s_mul_hi_i32 s19, s16, s8 // (op 1039) 
s_add_i32 s22, s19, s16 // (op 1039) 

s_ashr_i32 s19, s22, 31 // (op 1039) 
s_cmp_eq_i32 s8, 0 // (op 1039) 
s_mov_b32 s23, scc // (op 1039) 
s_cmp_eq_i32 s23, 0 // (op 1039) 
s_cselect_b32 s24, 0, -1 // (op 1039) 

s_and_b32 s23, s15, 31 // (op 1039) 
s_lshl_b32 s25, 1, s23 // (op 1039) 
s_add_i32 s26, s25, s24 // (op 1039) 


s_and_b32 s24, s19, s26 // (op 1039) 


s_add_i32 s19, s22, s24 // (op 1039) 


s_ashr_i32 s22, s19, s23 // (op 1039) 


s_lshl_b32 s19, s15, 24 // (op 1039) 
s_ashr_i32 s23, s19, 31 // (op 1039) 

s_xor_b32 s19, s22, s23 // (op 1039) 

s_sub_i32 s22, s19, s23 // (op 1039) 


s_add_u32 s38, s4, 16 // (op 1039) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 1039) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 1039) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 1039) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 1039) 

s_mul_i32 s19, s14, s38 // (op 1039) 
s_mul_i32 s23, s50, s19 // (op 1039) 

s_mul_i32 s19, s22, s23 // (op 1039) 

s_sub_u32 s23, s16, s19 // (op 1039) 

s_mov_b32 s19, s23 // (op 1039) 
s_mul_hi_i32 s24, s19, s13 // (op 1039) 
s_add_i32 s25, s24, s19 // (op 1039) 


s_ashr_i32 s19, s25, 31 // (op 1039) 
s_cmp_eq_i32 s13, 0 // (op 1039) 
s_mov_b32 s24, scc // (op 1039) 
s_cmp_eq_i32 s24, 0 // (op 1039) 
s_cselect_b32 s26, 0, -1 // (op 1039) 

s_and_b32 s24, s49, 31 // (op 1039) 
s_lshl_b32 s27, 1, s24 // (op 1039) 
s_add_i32 s28, s27, s26 // (op 1039) 


s_and_b32 s26, s19, s28 // (op 1039) 


s_add_i32 s19, s25, s26 // (op 1039) 


s_ashr_i32 s25, s19, s24 // (op 1039) 


s_lshl_b32 s19, s49, 24 // (op 1039) 
s_ashr_i32 s24, s19, 31 // (op 1039) 

s_xor_b32 s19, s25, s24 // (op 1039) 

s_sub_i32 s25, s19, s24 // (op 1039) 


s_mul_i32 s19, s48, s14 // (op 1039) 
s_add_u32 s26, s20, 16 // (op 1039) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 1039) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 1039) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 1039) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 1039) 

s_sub_i32 s24, s26, s19 // (op 1039) 

s_mul_i32 s26, s25, s24 // (op 1039) 

s_sub_u32 s24, s23, s26 // (op 1039) 


s_add_u32 s23, s24, s19 // (op 1039) 


s_cmp_eq_i32 s22, 0 // (op 1039) 
s_mov_b32 s19, scc // (op 1039) 

s_cmp_eq_i32 s19, 0 // (op 1039) 
s_cselect_b32 s22, s23, s6 // (op 1039) 


s_mul_i32 s6, s18, s38 // (op 1039) 


v_lshl_add_u32 v3, s22, 4, v9 // (op 1039) 


s_sub_i32 s18, s7, s6 // (op 1039) 


s_cmp_eq_i32 s19, 0 // (op 1039) 
s_cselect_b32 s6, s25, s18 // (op 1039) 



v_lshl_add_u32 v8, s6, 4, v10 // (op 1039) 


v_mul_lo_u32 v9, v8, s40 // (op 1039) 

v_add_lshl_u32 v8, v3, v9, 2 // (op 1039) 


s_mul_i32 s7, s40, s5 // (op 1041) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 1041) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 1041) least significant
s_mul_hi_u32 s19, s40, s4 // (op 1041) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 1041) most significant: sum
s_add_u32 s7, s7, s19 // (op 1041) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 1041) 

s_mov_b32 s0, s42 // (op 1041) reg expression
s_mov_b32 s1, s43 // (op 1041) reg expression
s_mov_b32 s2, s18 // (op 1041) reg expression
s_mov_b32 s3, 131072 // (op 1041) call()




buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen // (op 166) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155:
















s_endpgm  // End of RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.LRR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end:
.size RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_, .LRR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end-RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 120
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12800 // lds bytes
  .amdhsa_accum_offset 116
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
    DataTypeB: F8
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
    macrotile: [16, 16, 256]
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
  - .name: RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
    .symbol: RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 12800
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 115
    .agpr_count: 4
    .max_flat_workgroup_size: 64
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

