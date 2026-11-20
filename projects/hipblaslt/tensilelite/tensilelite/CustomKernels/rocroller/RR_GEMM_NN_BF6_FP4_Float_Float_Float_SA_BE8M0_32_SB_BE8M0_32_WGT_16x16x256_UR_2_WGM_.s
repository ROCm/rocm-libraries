// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_,@function
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__exec_begin:


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


v_mul_lo_u32 v0, v3, 6 // (op 849) 
v_lshrrev_b32 v4, 3, v0 // (op 849) 

v_lshrrev_b32 v0, 4, v3 // (op 849) 

v_lshlrev_b32 v3, 2, v0 // (op 849) 

v_add_u32 v0, v4, v3 // (op 849) 


v_and_b32 v3, 15, v1 // (op 902) 
v_and_b32 v4, 63, v1 // (op 902) 
v_lshrrev_b32 v5, 4, v4 // (op 902) 

v_lshlrev_b32 v4, 5, v5 // (op 902) 

v_lshl_add_u32 v5, v3, 8, v4 // (op 902) 


v_lshlrev_b32 v3, 2, v5 // (op 902) 

v_lshrrev_b32 v4, 3, v3 // (op 902) 

v_lshrrev_b32 v3, 6, v1 // (op 955) 
v_and_b32 v5, 15, v1 // (op 955) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 955) 


v_and_b32 v3, 48, v1 // (op 955) 
v_add_u32 v5, v3, v6 // (op 955) 


v_and_b32 v3, 15, v1 // (op 1008) 
v_and_b32 v6, 63, v1 // (op 1008) 
v_lshrrev_b32 v7, 4, v6 // (op 1008) 

v_lshl_add_u32 v6, v3, 3, v7 // (op 1008) 


v_and_b32 v3, 63, v1 // (op 1066) 
v_and_b32 v7, 48, v1 // (op 1066) 
v_sub_u32 v8, v3, v7 // (op 1066) 

v_lshrrev_b32 v7, 4, v3 // (op 1066) 

v_lshl_add_u32 v3, v7, 5, v8 // (op 1066) 


v_lshrrev_b32 v7, 6, v1 // (op 1066) 
v_lshlrev_b32 v8, 4, v7 // (op 1066) 

v_lshl_add_u32 v7, v3, 4, v8 // (op 1066) 


v_mul_lo_u32 v3, v7, 6 // (op 1066) 
v_lshrrev_b32 v8, 3, v3 // (op 1066) 

v_lshrrev_b32 v3, 4, v7 // (op 1066) 

v_lshlrev_b32 v7, 2, v3 // (op 1066) 

v_add_u32 v3, v8, v7 // (op 1066) 


v_and_b32 v7, 15, v1 // (op 1090) 
v_and_b32 v8, 63, v1 // (op 1090) 
v_lshrrev_b32 v9, 4, v8 // (op 1090) 

v_lshlrev_b32 v8, 5, v9 // (op 1090) 

v_lshl_add_u32 v9, v7, 8, v8 // (op 1090) 


v_lshlrev_b32 v7, 2, v9 // (op 1090) 

v_lshrrev_b32 v8, 3, v7 // (op 1090) 

v_lshrrev_b32 v7, 6, v1 // (op 1114) 
v_and_b32 v9, 15, v1 // (op 1114) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 1114) 


v_and_b32 v7, 48, v1 // (op 1114) 
v_add_u32 v9, v7, v10 // (op 1114) 


v_and_b32 v7, 15, v1 // (op 1138) 
v_and_b32 v10, 63, v1 // (op 1138) 
v_lshrrev_b32 v11, 4, v10 // (op 1138) 

v_lshl_add_u32 v10, v7, 3, v11 // (op 1138) 


v_mov_b32 v7, s6 // (op 2) Move value
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





v_mul_lo_u32 v13, v17, 6 // (op 559) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 559) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 559) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 559) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 559) 

v_mov_b32 v12, v14 // (op 559) convert

s_mul_i32 s1, s25, 6 // (op 561) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 561) least significant
s_mul_hi_u32 s6, s24, 6 // (op 561) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 561) most significant: sum

s_mul_i32 s7, s2, s1 // (op 561) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 561) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 561) least significant
s_mul_hi_u32 s19, s2, s0 // (op 561) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 561) most significant: sum
s_add_u32 s7, s7, s19 // (op 561) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 561) 

s_mov_b32 s52, s22 // (op 561) reg expression
s_mov_b32 s53, s23 // (op 561) reg expression
s_mov_b32 s54, s0 // (op 561) reg expression
s_mov_b32 s55, 131072 // (op 561) call()

s_mul_hi_i32 s51, s16, s9 // (op 578) 
v_lshlrev_b32 v13, 2, v1 // (op 578) 
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

v_mul_lo_u32 v15, v13, s3 // (op 578) most significant: low of low * high
v_mul_hi_u32 v16, v13, s2 // (op 578) most significant: high of low * low
v_mul_lo_u32 v14, v13, s2 // (op 578) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 578) most significant: sum


v_mov_b32 v13, s0 // (op 578) 
v_mov_b32 v16, s1 // (op 578) 
v_add_co_u32 v18, s[6:7], v13, v14 // (op 578) least significant half
v_addc_co_u32 v19, s[6:7], v16, v15, s[6:7] // (op 578) most significant half





v_mul_lo_u32 v17, v19, 6 // (op 578) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 578) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 578) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 578) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 578) 

v_mov_b32 v13, v14 // (op 578) convert

s_mul_i32 s1, s25, 6 // (op 580) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 580) least significant
s_mul_hi_u32 s51, s24, 6 // (op 580) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 580) most significant: sum

s_mul_i32 s7, s2, s1 // (op 580) most significant: low of low * high
s_mul_i32 s51, s3, s0 // (op 580) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 580) least significant
s_mul_hi_u32 s18, s2, s0 // (op 580) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 580) most significant: sum
s_add_u32 s7, s7, s18 // (op 580) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 580) 

s_mov_b32 s52, s22 // (op 580) reg expression
s_mov_b32 s53, s23 // (op 580) reg expression
s_mov_b32 s54, s0 // (op 580) reg expression
s_mov_b32 s55, 131072 // (op 580) call()

s_mul_hi_i32 s51, s16, s9 // (op 623) 
v_lshrrev_b32 v14, 3, v1 // (op 623) 
s_add_i32 s0, s51, s16 // (op 623) 

v_lshlrev_b32 v15, 1, v14 // (op 623) 

s_ashr_i32 s1, s0, 31 // (op 623) 
v_and_b32 v14, -8, v1 // (op 623) 
s_cmp_eq_i32 s9, 0 // (op 623) 
s_mov_b32 s51, scc // (op 623) 
v_sub_u32 v16, v1, v14 // (op 623) 

s_cmp_eq_i32 s51, 0 // (op 623) 
s_cselect_b32 s6, 0, -1 // (op 623) 

v_lshlrev_b32 v14, 5, v16 // (op 623) 

v_mov_b32 v17, 0 // (op 623) convert
v_mov_b32 v16, v14 // (op 623) convert

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



v_lshl_add_u32 v14, s1, 4, v15 // (op 623) 


v_mul_lo_u32 v19, v14, s27 // (op 623) most significant: low of low * high
v_mul_hi_u32 v15, v14, s26 // (op 623) most significant: high of low * low
v_mul_lo_u32 v18, v14, s26 // (op 623) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 623) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 623) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 623) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 623) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 623) 

v_mov_b32 v14, v16 // (op 623) convert

s_mul_i32 s1, s26, s5 // (op 625) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 625) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 625) least significant
s_mul_hi_u32 s6, s26, s4 // (op 625) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 625) most significant: sum
s_add_u32 s1, s1, s6 // (op 625) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 625) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 625) 

s_mov_b32 s56, s28 // (op 625) reg expression
s_mov_b32 s57, s29 // (op 625) reg expression
s_mov_b32 s58, s0 // (op 625) reg expression
s_mov_b32 s59, 131072 // (op 625) call()

s_mul_hi_i32 s51, s16, s9 // (op 642) 
v_lshrrev_b32 v15, 3, v1 // (op 642) 
s_add_i32 s0, s51, s16 // (op 642) 

v_lshlrev_b32 v16, 1, v15 // (op 642) 

s_ashr_i32 s1, s0, 31 // (op 642) 
v_and_b32 v15, -8, v1 // (op 642) 
s_cmp_eq_i32 s9, 0 // (op 642) 
s_mov_b32 s51, scc // (op 642) 
v_sub_u32 v17, v1, v15 // (op 642) 

s_cmp_eq_i32 s51, 0 // (op 642) 
s_cselect_b32 s6, 0, -1 // (op 642) 

v_lshlrev_b32 v15, 5, v17 // (op 642) 

v_mov_b32 v19, 0 // (op 642) convert
v_mov_b32 v18, v15 // (op 642) convert

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



v_lshl_add_u32 v15, s1, 4, v16 // (op 642) 


v_mul_lo_u32 v17, v15, s27 // (op 642) most significant: low of low * high
v_mul_hi_u32 v20, v15, s26 // (op 642) most significant: high of low * low
v_mul_lo_u32 v16, v15, s26 // (op 642) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 642) most significant: sum


v_add_co_u32 v20, s[0:1], v18, v16 // (op 642) least significant half
v_addc_co_u32 v21, s[0:1], v19, v17, s[0:1] // (op 642) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 642) 


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 642) 

v_mov_b32 v15, v18 // (op 642) convert

s_mul_i32 s1, s26, s5 // (op 644) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 644) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 644) least significant
s_mul_hi_u32 s6, s26, s4 // (op 644) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 644) most significant: sum
s_add_u32 s1, s1, s6 // (op 644) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 644) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 644) 

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


s_lshl_b64 s[0:1], s[26:27], 2 // (op 424) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 424) 


v_mov_b32 v22, v14 // (op 424) 
buffer_load_dwordx4 v[24:27], v22, s[56:59], 0 offen // (op 424) Load value
v_add_u32 v22, v22, v20 // (op 424) 
buffer_load_dwordx4 v[28:31], v22, s[56:59], 0 offen // (op 424) Load value



v_mov_b64 v[20:21], s[36:37] // (op 427) convert

v_mov_b32 v22, v16 // (op 427) 
buffer_load_ubyte v32, v22, s[64:67], 0 offen offset: 0 // (op 427) Load value
v_add_u32 v22, v22, v20 // (op 427) 
buffer_load_ubyte v33, v22, s[64:67], 0 offen offset: 0 // (op 427) Load value



s_mul_i32 s1, s3, 6 // (op 430) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 430) least significant
s_mul_hi_u32 s51, s2, 6 // (op 430) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 430) most significant: sum

v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 430) 


v_mov_b32 v22, v12 // (op 430) reg expression
buffer_load_dwordx3 v[34:36], v22, s[52:55], 0 offen // (op 430) Load value
v_add_u32 v22, v22, v20 // (op 430) 
buffer_load_dwordx3 v[38:40], v22, s[52:55], 0 offen // (op 430) Load value
v_add_u32 v22, v22, v20 // (op 430) 
buffer_load_dwordx3 v[42:44], v22, s[52:55], 0 offen // (op 430) Load value
v_add_u32 v22, v22, v20 // (op 430) 
buffer_load_dwordx3 v[46:48], v22, s[52:55], 0 offen // (op 430) Load value



v_mov_b64 v[20:21], s[30:31] // (op 433) convert

v_mov_b32 v37, v18 // (op 433) reg expression
buffer_load_ubyte v22, v37, s[60:63], 0 offen offset: 0 // (op 433) Load value
v_add_u32 v37, v37, v20 // (op 433) 
buffer_load_ubyte v23, v37, s[60:63], 0 offen offset: 0 // (op 433) Load value



s_lshl_b64 s[0:1], s[26:27], 2 // (op 436) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 436) 


v_mov_b32 v37, v14 // (op 436) 
buffer_load_dwordx4 v[50:53], v37, s[56:59], 0 offen offset:128 // (op 436) Load value
v_add_u32 v37, v37, v20 // (op 436) 
buffer_load_dwordx4 v[54:57], v37, s[56:59], 0 offen offset:128 // (op 436) Load value




v_mov_b64 v[20:21], s[36:37] // (op 439) convert

v_mov_b32 v14, v16 // (op 439) 
buffer_load_ubyte v58, v14, s[64:67], 0 offen offset:8 // (op 439) Load value
v_add_u32 v14, v14, v20 // (op 439) 
buffer_load_ubyte v59, v14, s[64:67], 0 offen offset:8 // (op 439) Load value




s_mul_i32 s1, s3, 6 // (op 442) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 442) least significant
s_mul_hi_u32 s51, s2, 6 // (op 442) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 442) most significant: sum

v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 442) 


s_lshl_b64 s[0:1], s[2:3], 8 // (op 442) 
s_mul_i32 s7, s1, 6 // (op 442) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 442) least significant
s_mul_hi_u32 s51, s0, 6 // (op 442) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 442) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 442) 

v_add_u32 v14, v12, s0 // (op 442) 

buffer_load_dwordx3 v[60:62], v14, s[52:55], 0 offen // (op 442) Load value
v_add_u32 v14, v14, v20 // (op 442) 
buffer_load_dwordx3 v[64:66], v14, s[52:55], 0 offen // (op 442) Load value
v_add_u32 v14, v14, v20 // (op 442) 
buffer_load_dwordx3 v[68:70], v14, s[52:55], 0 offen // (op 442) Load value
v_add_u32 v14, v14, v20 // (op 442) 
buffer_load_dwordx3 v[72:74], v14, s[52:55], 0 offen // (op 442) Load value




v_mov_b64 v[20:21], s[30:31] // (op 445) convert

s_lshl_b32 s51, s30, 3 // (op 445) 
v_add_u32 v12, v18, s51 // (op 445) 

buffer_load_ubyte v76, v12, s[60:63], 0 offen offset: 0 // (op 445) Load value
v_add_u32 v12, v12, v20 // (op 445) 
buffer_load_ubyte v77, v12, s[60:63], 0 offen offset: 0 // (op 445) Load value



v_and_b32 v12, -8, v1 // (op 890) 
v_sub_u32 v14, v1, v12 // (op 890) 

v_lshlrev_b32 v12, 5, v14 // (op 890) 

v_lshrrev_b32 v14, 3, v1 // (op 890) 
v_lshl_add_u32 v16, v14, 9, v12 // (op 890) 


v_lshlrev_b32 v12, 2, v16 // (op 890) 

v_lshrrev_b32 v14, 3, v12 // (op 890) 




s_waitcnt vmcnt(19)
ds_write_b128 v14, v[24:27] // (op 447) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v14, v[28:31] offset:128 // (op 447) Store local data 

v_and_b32 v12, -8, v1 // (op 996) 
v_sub_u32 v14, v1, v12 // (op 996) 

v_lshrrev_b32 v12, 3, v1 // (op 996) 
v_lshl_add_u32 v16, v12, 4, v14 // (op 996) 





s_waitcnt vmcnt(17)
ds_write_b8 v16, v32 offset:2048 // (op 448) Store local data 
s_waitcnt vmcnt(16)
ds_write_b8 v16, v33 offset:2056 // (op 448) Store local data 

v_lshlrev_b32 v12, 6, v1 // (op 837) 
v_mul_lo_u32 v14, v12, 6 // (op 837) 
v_lshrrev_b32 v16, 3, v14 // (op 837) 

v_lshrrev_b32 v14, 4, v12 // (op 837) 

v_lshlrev_b32 v12, 2, v14 // (op 837) 

v_add_u32 v14, v16, v12 // (op 837) 





s_waitcnt vmcnt(15)
ds_write_b96 v14, v[34:36] offset:2176 // (op 449) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v14, v[38:40] offset:2192 // (op 449) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v14, v[42:44] offset:2208 // (op 449) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v14, v[46:48] offset:2224 // (op 449) Store local data 

v_and_b32 v12, -16, v1 // (op 943) 
v_sub_u32 v14, v1, v12 // (op 943) 

v_lshrrev_b32 v12, 4, v1 // (op 943) 
v_lshl_add_u32 v16, v12, 5, v14 // (op 943) 





s_waitcnt vmcnt(11)
ds_write_b8 v16, v22 offset:6272 // (op 450) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v16, v23 offset:6288 // (op 450) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 1194) 



ds_read_b128 v[78:81], v4 // (op 467) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1235) 



ds_read_u8 v12, v6 offset:2048 // (op 473) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1175) 



ds_read_b96_tr_b6 v[82:84], v0 offset:2176 // (op 464) Transpose load local data 
ds_read_b96_tr_b6 v[85:87], v0 offset:2432 // (op 464) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1215) 


ds_read_u8 v14, v5 offset:6272 // (op 470) Load local data 
v_lshlrev_b32 v16, 6, v1 // (op 808) 
v_mul_lo_u32 v18, v16, 6 // (op 808) 
v_lshrrev_b32 v37, 3, v18 // (op 808) 

v_lshrrev_b32 v18, 4, v16 // (op 808) 

v_lshlrev_b32 v16, 2, v18 // (op 808) 

v_add_u32 v18, v37, v16 // (op 808) 


v_and_b32 v16, -8, v1 // (op 861) 
v_sub_u32 v37, v1, v16 // (op 861) 

v_lshlrev_b32 v16, 5, v37 // (op 861) 

v_lshrrev_b32 v37, 3, v1 // (op 861) 
v_lshl_add_u32 v41, v37, 9, v16 // (op 861) 


v_lshlrev_b32 v16, 2, v41 // (op 861) 

v_lshrrev_b32 v37, 3, v16 // (op 861) 

v_and_b32 v16, -16, v1 // (op 914) 
v_sub_u32 v41, v1, v16 // (op 914) 

v_lshrrev_b32 v16, 4, v1 // (op 914) 
v_lshl_add_u32 v45, v16, 5, v41 // (op 914) 


v_and_b32 v16, -8, v1 // (op 967) 
v_sub_u32 v41, v1, v16 // (op 967) 

v_lshrrev_b32 v16, 3, v1 // (op 967) 
v_lshl_add_u32 v49, v16, 4, v41 // (op 967) 


v_lshlrev_b32 v16, 6, v1 // (op 1054) 
v_mul_lo_u32 v41, v16, 6 // (op 1054) 
v_lshrrev_b32 v63, 3, v41 // (op 1054) 

v_lshrrev_b32 v41, 4, v16 // (op 1054) 

v_lshlrev_b32 v16, 2, v41 // (op 1054) 

v_add_u32 v41, v63, v16 // (op 1054) 


v_and_b32 v16, -8, v1 // (op 1078) 
v_sub_u32 v63, v1, v16 // (op 1078) 

v_lshlrev_b32 v16, 5, v63 // (op 1078) 

v_lshrrev_b32 v63, 3, v1 // (op 1078) 
v_lshl_add_u32 v67, v63, 9, v16 // (op 1078) 


v_lshlrev_b32 v16, 2, v67 // (op 1078) 

v_lshrrev_b32 v63, 3, v16 // (op 1078) 

v_and_b32 v16, -16, v1 // (op 1102) 
v_sub_u32 v67, v1, v16 // (op 1102) 

v_lshrrev_b32 v16, 4, v1 // (op 1102) 
v_lshl_add_u32 v71, v16, 5, v67 // (op 1102) 


v_and_b32 v16, -8, v1 // (op 1126) 
v_sub_u32 v67, v1, v16 // (op 1126) 

v_lshrrev_b32 v16, 3, v1 // (op 1126) 
v_lshl_add_u32 v75, v16, 4, v67 // (op 1126) 


s_mov_b32 s51, 0 // (op 59) call()
s_ashr_i64 s[0:1], s[24:25], 8 // (op 58) 
s_mov_b32 s6, s0 // (op 58) convert

s_and_b32 s7, s6, 4294967294 // (op 58) 

s_cmp_lt_u32 s51, s7 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58 if false)
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58:

s_add_u32 s51, s51, 2 // (op 60) 
s_lshl_b64 s[0:1], s[26:27], 2 // (op 10) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 10) 


v_mov_b32 v16, v15 // (op 10) 
v_mov_b32 v67, v16 // (op 10) 
buffer_load_dwordx4 v[24:27], v67, s[56:59], 0 offen offset:256 // (op 10) Load value
v_add_u32 v67, v67, v20 // (op 10) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:87], v[78:81], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 88) 
buffer_load_dwordx4 v[28:31], v67, s[56:59], 0 offen offset:256 // (op 10) Load value



s_barrier  // (op 1184) 
v_mov_b64 v[20:21], s[36:37] // (op 25) convert




ds_read_b96_tr_b6 v[82:84], v0 offset:4224 // (op 91) Transpose load local data 
ds_read_b96_tr_b6 v[85:87], v0 offset:4480 // (op 91) Transpose load local data 
v_mov_b32 v67, v17 // (op 25) 
s_barrier  // (op 1204) 
v_mov_b32 v88, v67 // (op 25) 



ds_read_b128 v[78:81], v4 offset:64 // (op 95) Load local data 
buffer_load_ubyte v32, v88, s[64:67], 0 offen offset:16 // (op 25) Load value
s_barrier  // (op 1225) 
v_add_u32 v88, v88, v20 // (op 25) 


ds_read_u8 v14, v5 offset:6336 // (op 99) Load local data 
buffer_load_ubyte v33, v88, s[64:67], 0 offen offset:16 // (op 25) Load value



s_barrier  // (op 1245) 
s_mul_i32 s1, s3, 6 // (op 4) most significant: low of high * low



s_mul_i32 s0, s2, 6 // (op 4) least significant
ds_read_u8 v12, v6 offset:2052 // (op 103) Load local data 
s_mul_hi_u32 s6, s2, 6 // (op 4) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4) most significant: sum

v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 4) 


v_mov_b32 v88, v13 // (op 4) 
s_lshl_b64 s[0:1], s[2:3], 8 // (op 4) 
s_mul_i32 s7, s1, 6 // (op 4) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 4) least significant
s_mul_hi_u32 s18, s0, 6 // (op 4) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 4) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 4) 

s_lshl_b32 s6, s0, 1 // (op 4) 

v_add_u32 v89, v88, s6 // (op 4) 

buffer_load_dwordx3 v[34:36], v89, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v89, v89, v20 // (op 4) 
buffer_load_dwordx3 v[38:40], v89, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v89, v89, v20 // (op 4) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:87], v[78:81], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 105) 
buffer_load_dwordx3 v[42:44], v89, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v89, v89, v20 // (op 4) 
buffer_load_dwordx3 v[46:48], v89, s[52:55], 0 offen // (op 4) Load value



v_mov_b64 v[20:21], s[30:31] // (op 17) convert

v_mov_b32 v89, v19 // (op 17) 
s_lshl_b32 s0, s30, 4 // (op 17) 
v_add_u32 v90, v89, s0 // (op 17) 

buffer_load_ubyte v22, v90, s[60:63], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v90, v90, v20 // (op 17) 
buffer_load_ubyte v23, v90, s[60:63], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1265) 



s_waitcnt vmcnt(19)
ds_write_b128 v63, v[50:53] offset:6400 // (op 302) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v63, v[54:57] offset:6528 // (op 302) Store local data 
s_barrier  // (op 1277) 



s_waitcnt vmcnt(17)
ds_write_b8 v75, v58 offset:8448 // (op 304) Store local data 
s_waitcnt vmcnt(16)
ds_write_b8 v75, v59 offset:8456 // (op 304) Store local data 
s_barrier  // (op 1258) 



s_waitcnt vmcnt(15)
ds_write_b96 v41, v[60:62] offset:8576 // (op 283) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v41, v[64:66] offset:8592 // (op 283) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v41, v[68:70] offset:8608 // (op 283) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v41, v[72:74] offset:8624 // (op 283) Store local data 
s_barrier  // (op 1271) 



s_waitcnt vmcnt(11)
ds_write_b8 v71, v76 offset:12672 // (op 303) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v71, v77 offset:12688 // (op 303) Store local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1261) 



ds_read_b128 v[90:93], v8 offset:6400 // (op 287) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1274) 



ds_read_u8 v20, v10 offset:8448 // (op 291) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1255) 



ds_read_b96_tr_b6 v[94:96], v3 offset:8576 // (op 285) Transpose load local data 
ds_read_b96_tr_b6 v[97:99], v3 offset:8832 // (op 285) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1268) 


ds_read_u8 v21, v9 offset:12672 // (op 289) Load local data 

s_lshl_b64 s[0:1], s[26:27], 2 // (op 305) 
v_lshrrev_b64 v[100:101], 3, s[0:1] // (op 305) 


v_mov_b32 v102, v16 // (op 305) 
buffer_load_dwordx4 v[50:53], v102, s[56:59], 0 offen offset:384 // (op 305) Load value
v_add_u32 v102, v102, v100 // (op 305) 
buffer_load_dwordx4 v[54:57], v102, s[56:59], 0 offen offset:384 // (op 305) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[94:99], v[90:93], a[0:3], v21, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 292) 









v_add_u32 v15, 256, v15 // (op 634) 
v_mov_b64 v[20:21], s[36:37] // (op 307) convert

v_mov_b32 v16, v67 // (op 307) 
ds_read_b96_tr_b6 v[90:92], v3 offset:10624 // (op 294) Transpose load local data 
ds_read_b96_tr_b6 v[93:95], v3 offset:10880 // (op 294) Transpose load local data 
buffer_load_ubyte v58, v16, s[64:67], 0 offen offset:24 // (op 307) Load value
v_add_u32 v16, v16, v20 // (op 307) 



s_barrier  // (op 1336) 
ds_read_b128 v[96:99], v8 offset:6464 // (op 296) Load local data 
buffer_load_ubyte v59, v16, s[64:67], 0 offen offset:24 // (op 307) Load value






s_barrier  // (op 1340) 
v_add_u32 v17, 16, v17 // (op 762) 
ds_read_u8 v16, v9 offset:12736 // (op 298) Load local data 
s_mul_i32 s1, s3, 6 // (op 282) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 282) least significant



s_barrier  // (op 1344) 
ds_read_u8 v67, v10 offset:8452 // (op 300) Load local data 
s_mul_hi_u32 s6, s2, 6 // (op 282) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 282) most significant: sum

s_barrier  // (op 1348) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 282) 


s_lshl_b64 s[0:1], s[2:3], 8 // (op 282) 
s_mul_i32 s7, s1, 6 // (op 282) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 282) least significant
s_mul_hi_u32 s18, s0, 6 // (op 282) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 282) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 282) 

s_mul_i32 s6, 3, s0 // (op 282) 

v_add_u32 v100, v88, s6 // (op 282) 

buffer_load_dwordx3 v[60:62], v100, s[52:55], 0 offen // (op 282) Load value
v_add_u32 v100, v100, v20 // (op 282) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:95], v[96:99], a[0:3], v16, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 301) 
buffer_load_dwordx3 v[64:66], v100, s[52:55], 0 offen // (op 282) Load value




v_add_u32 v100, v100, v20 // (op 282) 
buffer_load_dwordx3 v[68:70], v100, s[52:55], 0 offen // (op 282) Load value
v_add_u32 v100, v100, v20 // (op 282) 
buffer_load_dwordx3 v[72:74], v100, s[52:55], 0 offen // (op 282) Load value




v_mov_b64 v[20:21], s[30:31] // (op 306) convert

s_lshl_b32 s0, s30, 3 // (op 306) 
s_mul_i32 s1, 3, s0 // (op 306) 

v_add_u32 v16, v89, s1 // (op 306) 

s_lshl_b64 s[0:1], s[2:3], 8 // (op 570) 
s_mul_i32 s7, s1, 6 // (op 570) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 570) least significant
s_mul_hi_u32 s18, s0, 6 // (op 570) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 570) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 570) 

s_lshl_b32 s6, s0, 1 // (op 570) 

v_add_u32 v13, v13, s6 // (op 570) 

buffer_load_ubyte v76, v16, s[60:63], 0 offen offset: 0 // (op 306) Load value
v_add_u32 v16, v16, v20 // (op 306) 
buffer_load_ubyte v77, v16, s[60:63], 0 offen offset: 0 // (op 306) Load value



s_barrier  // (op 1197) 
s_lshl_b32 s0, s30, 4 // (op 698) 



v_add_u32 v19, v19, s0 // (op 698) 

s_waitcnt vmcnt(19)
ds_write_b128 v37, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v37, v[28:31] offset:128 // (op 13) Store local data 
s_barrier  // (op 1238) 



s_waitcnt vmcnt(17)
ds_write_b8 v49, v32 offset:2048 // (op 30) Store local data 
s_waitcnt vmcnt(16)
ds_write_b8 v49, v33 offset:2056 // (op 30) Store local data 
s_barrier  // (op 1178) 



s_waitcnt vmcnt(15)
ds_write_b96 v18, v[34:36] offset:2176 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v18, v[38:40] offset:2192 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v18, v[42:44] offset:2208 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v18, v[46:48] offset:2224 // (op 6) Store local data 
s_barrier  // (op 1218) 



s_waitcnt vmcnt(11)
ds_write_b8 v45, v22 offset:6272 // (op 21) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v45, v23 offset:6288 // (op 21) Store local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1207) 



ds_read_b128 v[78:81], v4 // (op 75) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1248) 



ds_read_u8 v12, v6 offset:2048 // (op 85) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1187) 



ds_read_b96_tr_b6 v[82:84], v0 offset:2176 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[85:87], v0 offset:2432 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1228) 


ds_read_u8 v14, v5 offset:6272 // (op 80) Load local data 
s_ashr_i64 s[0:1], s[24:25], 8 // (op 58) 
s_mov_b32 s6, s0 // (op 58) convert

s_and_b32 s7, s6, 4294967294 // (op 58) 

s_cmp_lt_u32 s51, s7 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58 if true)
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58:

















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

v_lshl_add_u32 v63, s51, 9, v13 // (op 533) 


v_mul_lo_u32 v49, v63, s3 // (op 533) most significant: low of low * high
v_mul_hi_u32 v13, v63, s2 // (op 533) most significant: high of low * low
v_mul_lo_u32 v48, v63, s2 // (op 533) least significant: low of low * low
v_add_u32 v49, v49, v13 // (op 533) most significant: sum


v_mov_b32 v13, s0 // (op 533) 
v_mov_b32 v63, s1 // (op 533) 
v_add_co_u32 v46, s[6:7], v13, v48 // (op 533) least significant half
v_addc_co_u32 v47, s[6:7], v63, v49, s[6:7] // (op 533) most significant half





v_mul_lo_u32 v49, v47, 6 // (op 533) most significant: low of high * low
v_mul_hi_u32 v13, v46, 6 // (op 533) most significant: high of low * low
v_mul_lo_u32 v48, v46, 6 // (op 533) least significant: low of low * low
v_add_u32 v49, v49, v13 // (op 533) most significant: sum


v_lshrrev_b64 v[46:47], 3, v[48:49] // (op 533) 

v_mov_b32 v13, v46 // (op 533) convert

s_mul_i32 s1, s25, 6 // (op 535) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 535) least significant
s_mul_hi_u32 s51, s24, 6 // (op 535) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 535) most significant: sum

s_mul_i32 s7, s2, s1 // (op 535) most significant: low of low * high
s_mul_i32 s51, s3, s0 // (op 535) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 535) least significant
s_mul_hi_u32 s18, s2, s0 // (op 535) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 535) most significant: sum
s_add_u32 s7, s7, s18 // (op 535) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 535) 

s_mov_b32 s52, s22 // (op 535) reg expression
s_mov_b32 s53, s23 // (op 535) reg expression
s_mov_b32 s54, s0 // (op 535) reg expression
s_mov_b32 s55, 131072 // (op 535) call()

s_mul_hi_i32 s51, s16, s9 // (op 600) 
v_lshrrev_b32 v63, 3, v1 // (op 600) 
s_add_i32 s0, s51, s16 // (op 600) 

v_lshlrev_b32 v67, 1, v63 // (op 600) 

s_ashr_i32 s1, s0, 31 // (op 600) 
v_and_b32 v63, -8, v1 // (op 600) 
s_cmp_eq_i32 s9, 0 // (op 600) 
s_mov_b32 s51, scc // (op 600) 
v_sub_u32 v71, v1, v63 // (op 600) 

s_cmp_eq_i32 s51, 0 // (op 600) 
s_cselect_b32 s6, 0, -1 // (op 600) 

v_lshlrev_b32 v63, 5, v71 // (op 600) 

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



v_lshl_add_u32 v71, s1, 4, v67 // (op 600) 


v_mul_lo_u32 v49, v71, s27 // (op 600) most significant: low of low * high
v_mul_hi_u32 v67, v71, s26 // (op 600) most significant: high of low * low
v_mul_lo_u32 v48, v71, s26 // (op 600) least significant: low of low * low
v_add_u32 v49, v49, v67 // (op 600) most significant: sum


s_ashr_i64 s[0:1], s[24:25], 8 // (op 600) 
s_lshr_b32 s51, s0, 1 // (op 600) 

v_lshl_add_u32 v67, s51, 9, v63 // (op 600) 


v_mov_b32 v47, 0 // (op 600) convert
v_mov_b32 v46, v67 // (op 600) convert

v_add_co_u32 v44, s[0:1], v46, v48 // (op 600) least significant half
v_addc_co_u32 v45, s[0:1], v47, v49, s[0:1] // (op 600) most significant half

v_lshlrev_b64 v[44:45], 2, v[44:45] // (op 600) 


v_lshrrev_b64 v[42:43], 3, v[44:45] // (op 600) 

v_mov_b32 v63, v42 // (op 600) convert

s_mul_i32 s1, s26, s5 // (op 602) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 602) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 602) least significant
s_mul_hi_u32 s6, s26, s4 // (op 602) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 602) most significant: sum
s_add_u32 s1, s1, s6 // (op 602) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 602) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 602) 

s_mov_b32 s56, s28 // (op 602) reg expression
s_mov_b32 s57, s29 // (op 602) reg expression
s_mov_b32 s58, s0 // (op 602) reg expression
s_mov_b32 s59, 131072 // (op 602) call()

s_mul_hi_i32 s51, s16, s9 // (op 664) 
v_and_b32 v67, -16, v1 // (op 664) 
s_add_i32 s0, s51, s16 // (op 664) 

v_sub_u32 v71, v1, v67 // (op 664) 

s_ashr_i32 s1, s0, 31 // (op 664) 
v_lshrrev_b32 v67, 4, v1 // (op 664) 
s_cmp_eq_i32 s9, 0 // (op 664) 
s_mov_b32 s51, scc // (op 664) 
v_lshlrev_b32 v75, 1, v67 // (op 664) 

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



v_lshl_add_u32 v67, s7, 4, v71 // (op 664) 


s_ashr_i64 s[0:1], s[24:25], 8 // (op 664) 
s_lshr_b32 s51, s0, 1 // (op 664) 

v_lshl_add_u32 v71, s51, 4, v75 // (op 664) 


v_mul_lo_u32 v75, v71, s30 // (op 664) 

v_add_u32 v71, v67, v75 // (op 664) 


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
v_lshrrev_b32 v67, 3, v1 // (op 728) 
s_add_i32 s0, s51, s16 // (op 728) 

v_lshlrev_b32 v75, 1, v67 // (op 728) 

s_ashr_i32 s1, s0, 31 // (op 728) 
v_and_b32 v67, -8, v1 // (op 728) 
s_cmp_eq_i32 s9, 0 // (op 728) 
s_mov_b32 s51, scc // (op 728) 
v_sub_u32 v15, v1, v67 // (op 728) 

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



v_lshl_add_u32 v67, s1, 4, v75 // (op 728) 


v_mul_lo_u32 v75, v67, s36 // (op 728) 

s_ashr_i64 s[0:1], s[24:25], 8 // (op 728) 
s_lshr_b32 s51, s0, 1 // (op 728) 

v_lshl_add_u32 v67, s51, 4, v15 // (op 728) 


v_add_u32 v15, v67, v75 // (op 728) 


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

v_lshlrev_b32 v67, 6, v1 // (op 821) 
v_mul_lo_u32 v75, v67, 6 // (op 821) 
v_lshrrev_b32 v16, 3, v75 // (op 821) 

v_lshrrev_b32 v75, 4, v67 // (op 821) 

v_lshlrev_b32 v67, 2, v75 // (op 821) 

v_add_u32 v75, v16, v67 // (op 821) 


v_and_b32 v67, -8, v1 // (op 874) 
v_sub_u32 v16, v1, v67 // (op 874) 

v_lshlrev_b32 v67, 5, v16 // (op 874) 

v_lshrrev_b32 v16, 3, v1 // (op 874) 
v_lshl_add_u32 v17, v16, 9, v67 // (op 874) 


v_lshlrev_b32 v16, 2, v17 // (op 874) 

v_lshrrev_b32 v67, 3, v16 // (op 874) 

v_and_b32 v16, -16, v1 // (op 927) 
v_sub_u32 v17, v1, v16 // (op 927) 

v_lshrrev_b32 v16, 4, v1 // (op 927) 
v_lshl_add_u32 v18, v16, 5, v17 // (op 927) 


v_and_b32 v16, -8, v1 // (op 980) 
v_sub_u32 v17, v1, v16 // (op 980) 

v_lshrrev_b32 v16, 3, v1 // (op 980) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 980) 


s_ashr_i64 s[0:1], s[24:25], 8 // (op 194) 
s_and_b32 s51, s0, 4294967294 // (op 194) 

s_ashr_i64 s[0:1], s[24:25], 8 // (op 193) 
s_mov_b32 s6, s0 // (op 193) convert

s_cmp_lt_u32 s51, s6 // (op 193) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193 // (op 193) Condition: Top (jump to RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193:

s_add_u32 s51, s51, 1 // (op 195) 
s_mul_i32 s1, s3, 6 // (op 198) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 198) least significant
s_mul_hi_u32 s6, s2, 6 // (op 198) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 198) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[0:1] // (op 198) 


v_mov_b32 v12, v13 // (op 198) 
v_mov_b32 v14, v12 // (op 198) reg expression
buffer_load_dwordx3 v[20:22], v14, s[52:55], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v16 // (op 198) 
buffer_load_dwordx3 v[24:26], v14, s[52:55], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v16 // (op 198) 
buffer_load_dwordx3 v[28:30], v14, s[52:55], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v16 // (op 198) 
buffer_load_dwordx3 v[32:34], v14, s[52:55], 0 offen // (op 198) Load value



s_lshl_b64 s[0:1], s[2:3], 8 // (op 523) 
s_mul_i32 s7, s1, 6 // (op 523) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 523) least significant
s_mul_hi_u32 s18, s0, 6 // (op 523) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 523) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 523) 

v_add_u32 v13, v13, s0 // (op 523) 


s_barrier  // (op 1181) 
s_lshl_b64 s[0:1], s[26:27], 2 // (op 221) 



v_lshrrev_b64 v[16:17], 3, s[0:1] // (op 221) 


v_mov_b32 v12, v63 // (op 221) 
v_mov_b32 v14, v12 // (op 221) 
buffer_load_dwordx4 v[36:39], v14, s[56:59], 0 offen // (op 221) Load value
v_add_u32 v14, v14, v16 // (op 221) 
buffer_load_dwordx4 v[40:43], v14, s[56:59], 0 offen // (op 221) Load value



v_add_u32 v63, 128, v63 // (op 592) 
s_waitcnt vmcnt(5)
ds_write_b96 v75, v[20:22] offset:2176 // (op 199) Store local data 

v_mov_b64 v[16:17], s[30:31] // (op 222) convert

v_mov_b32 v12, v71 // (op 222) 
v_mov_b32 v14, v12 // (op 222) reg expression
s_waitcnt vmcnt(4)
ds_write_b96 v75, v[24:26] offset:2192 // (op 199) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v75, v[28:30] offset:2208 // (op 199) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v75, v[32:34] offset:2224 // (op 199) Store local data 
buffer_load_ubyte v44, v14, s[60:63], 0 offen offset: 0 // (op 222) Load value

v_add_u32 v14, v14, v16 // (op 222) 
s_barrier  // (op 1200) 
buffer_load_ubyte v45, v14, s[60:63], 0 offen offset: 0 // (op 222) Load value






s_lshl_b32 s0, s30, 3 // (op 656) 
v_add_u32 v71, v71, s0 // (op 656) 

s_waitcnt vmcnt(3)
ds_write_b128 v67, v[36:39] // (op 218) Store local data 

v_mov_b64 v[16:17], s[36:37] // (op 223) convert

v_mov_b32 v12, v15 // (op 223) 
v_mov_b32 v14, v12 // (op 223) 
buffer_load_ubyte v20, v14, s[64:67], 0 offen offset: 0 // (op 223) Load value
v_add_u32 v14, v14, v16 // (op 223) 
s_waitcnt vmcnt(3)
ds_write_b128 v67, v[40:43] offset:128 // (op 218) Store local data 
buffer_load_ubyte v21, v14, s[64:67], 0 offen offset: 0 // (op 223) Load value




v_add_u32 v15, 8, v15 // (op 720) 
s_barrier  // (op 1221) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1190) 






ds_read_b96_tr_b6 v[22:24], v0 offset:2176 // (op 201) Transpose load local data 
ds_read_b96_tr_b6 v[25:27], v0 offset:2432 // (op 201) Transpose load local data 
s_waitcnt vmcnt(3)
ds_write_b8 v18, v44 offset:6272 // (op 219) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1211) 
s_waitcnt vmcnt(2)
ds_write_b8 v18, v45 offset:6288 // (op 219) Store local data 




ds_read_b128 v[28:31], v4 // (op 203) Load local data 
s_barrier  // (op 1241) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1231) 





ds_read_u8 v12, v5 offset:6272 // (op 205) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v20 offset:2048 // (op 220) Store local data 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v21 offset:2056 // (op 220) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1251) 



ds_read_u8 v14, v6 offset:2048 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:27], v[28:31], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 208) 







ds_read_b96_tr_b6 v[20:22], v0 offset:4224 // (op 210) Transpose load local data 
ds_read_b96_tr_b6 v[23:25], v0 offset:4480 // (op 210) Transpose load local data 



ds_read_b128 v[26:29], v4 offset:64 // (op 212) Load local data 


ds_read_u8 v12, v5 offset:6336 // (op 214) Load local data 



ds_read_u8 v14, v6 offset:2052 // (op 216) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:29], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 217) 




s_ashr_i64 s[0:1], s[24:25], 8 // (op 193) 
s_mov_b32 s6, s0 // (op 193) convert

s_cmp_lt_u32 s51, s6 // (op 193) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193 // (op 193) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193 if true)
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193:























s_barrier  // (op 1328) 








v_cmp_eq_f32 vcc, 0.00000, v7 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v11, v12 // (op 50) 
v_mul_f32 v17, v11, v13 // (op 50) 
v_mul_f32 v18, v11, v14 // (op 50) 
v_mul_f32 v19, v11, v15 // (op 50) 

s_mul_hi_i32 s0, s16, s9 // (op 1022) 
v_lshrrev_b32 v0, 6, v1 // (op 1022) 
s_add_i32 s1, s0, s16 // (op 1022) 

v_and_b32 v3, 63, v1 // (op 1022) 
s_ashr_i32 s0, s1, 31 // (op 1022) 
v_lshrrev_b32 v4, 4, v3 // (op 1022) 
s_cmp_eq_i32 s9, 0 // (op 1022) 
s_mov_b32 s2, scc // (op 1022) 
v_lshlrev_b32 v5, 2, v4 // (op 1022) 

s_cmp_eq_i32 s2, 0 // (op 1022) 
s_cselect_b32 s3, 0, -1 // (op 1022) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1022) 


s_and_b32 s2, s10, 31 // (op 1022) 
v_lshrrev_b32 v0, 2, v3 // (op 1022) 

s_lshl_b32 s6, 1, s2 // (op 1022) 
v_and_b32 v3, 3, v0 // (op 1022) 

s_add_i32 s7, s6, s3 // (op 1022) 


v_and_b32 v0, 3, v1 // (op 1022) 
s_and_b32 s3, s0, s7 // (op 1022) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 1022) 


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


v_lshl_add_u32 v0, s6, 4, v4 // (op 1022) 


s_sub_i32 s2, s1, s0 // (op 1022) 


s_cmp_eq_i32 s3, 0 // (op 1022) 
s_cselect_b32 s22, s23, s2 // (op 1022) 



v_lshl_add_u32 v6, s22, 4, v5 // (op 1022) 


v_mul_lo_u32 v3, v6, s40 // (op 1022) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 1022) 


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




buffer_store_dwordx4 v[16:19], v4, s[0:3], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155:
s_mul_hi_i32 s6, s16, s9 // (op 791) 
v_lshrrev_b32 v0, 6, v1 // (op 791) 
s_add_i32 s7, s6, s16 // (op 791) 

v_and_b32 v3, 63, v1 // (op 791) 
s_ashr_i32 s6, s7, 31 // (op 791) 
v_lshrrev_b32 v4, 4, v3 // (op 791) 
s_cmp_eq_i32 s9, 0 // (op 791) 
s_mov_b32 s18, scc // (op 791) 
v_lshlrev_b32 v5, 2, v4 // (op 791) 

s_cmp_eq_i32 s18, 0 // (op 791) 
s_cselect_b32 s19, 0, -1 // (op 791) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 791) 


s_and_b32 s18, s10, 31 // (op 791) 
v_lshrrev_b32 v0, 2, v3 // (op 791) 

s_lshl_b32 s22, 1, s18 // (op 791) 
v_and_b32 v3, 3, v0 // (op 791) 

s_add_i32 s23, s22, s19 // (op 791) 


v_and_b32 v0, 3, v1 // (op 791) 
s_and_b32 s19, s6, s23 // (op 791) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 791) 


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


v_lshl_add_u32 v0, s22, 4, v4 // (op 791) 


s_sub_i32 s18, s7, s6 // (op 791) 


s_cmp_eq_i32 s19, 0 // (op 791) 
s_cselect_b32 s6, s25, s18 // (op 791) 



v_lshl_add_u32 v6, s6, 4, v5 // (op 791) 


v_mul_lo_u32 v3, v6, s44 // (op 791) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 791) 


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





buffer_load_dwordx4 v[12:15], v4, s[36:39], 0 offen // (op 167) Load value



v_accvgpr_read v16, a0 // (op 164) 
v_accvgpr_read v17, a1 // (op 164) 
v_accvgpr_read v18, a2 // (op 164) 
v_accvgpr_read v19, a3 // (op 164) 
v_mul_f32 v20, v11, v16 // (op 164) 
v_mul_f32 v21, v11, v17 // (op 164) 
v_mul_f32 v22, v11, v18 // (op 164) 
v_mul_f32 v23, v11, v19 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v7, v12 // (op 168) 
v_mul_f32 v17, v7, v13 // (op 168) 
v_mul_f32 v18, v7, v14 // (op 168) 
v_mul_f32 v19, v7, v15 // (op 168) 

v_add_f32 v12, v16, v20 // (op 165) 
v_add_f32 v13, v17, v21 // (op 165) 
v_add_f32 v14, v18, v22 // (op 165) 
v_add_f32 v15, v19, v23 // (op 165) 


s_mul_hi_i32 s6, s16, s9 // (op 1039) 
v_lshrrev_b32 v0, 6, v1 // (op 1039) 
s_add_i32 s7, s6, s16 // (op 1039) 

v_and_b32 v3, 63, v1 // (op 1039) 
s_ashr_i32 s6, s7, 31 // (op 1039) 
v_lshrrev_b32 v4, 4, v3 // (op 1039) 
s_cmp_eq_i32 s9, 0 // (op 1039) 
s_mov_b32 s18, scc // (op 1039) 
v_lshlrev_b32 v5, 2, v4 // (op 1039) 

s_cmp_eq_i32 s18, 0 // (op 1039) 
s_cselect_b32 s19, 0, -1 // (op 1039) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1039) 


s_and_b32 s18, s10, 31 // (op 1039) 
v_lshrrev_b32 v0, 2, v3 // (op 1039) 

s_lshl_b32 s22, 1, s18 // (op 1039) 
v_and_b32 v3, 3, v0 // (op 1039) 

s_add_i32 s23, s22, s19 // (op 1039) 


v_and_b32 v0, 3, v1 // (op 1039) 
s_and_b32 s19, s6, s23 // (op 1039) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 1039) 


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


v_lshl_add_u32 v0, s22, 4, v4 // (op 1039) 


s_sub_i32 s18, s7, s6 // (op 1039) 


s_cmp_eq_i32 s19, 0 // (op 1039) 
s_cselect_b32 s6, s25, s18 // (op 1039) 



v_lshl_add_u32 v6, s6, 4, v5 // (op 1039) 


v_mul_lo_u32 v3, v6, s40 // (op 1039) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 1039) 


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




buffer_store_dwordx4 v[12:15], v4, s[0:3], 0 offen // (op 166) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155:
















s_endpgm  // End of RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.LRR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end:
.size RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_, .LRR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end-RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 108
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12800 // lds bytes
  .amdhsa_accum_offset 104
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
    DataTypeB: F4
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
  - .name: RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
    .symbol: RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 12800
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 103
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

