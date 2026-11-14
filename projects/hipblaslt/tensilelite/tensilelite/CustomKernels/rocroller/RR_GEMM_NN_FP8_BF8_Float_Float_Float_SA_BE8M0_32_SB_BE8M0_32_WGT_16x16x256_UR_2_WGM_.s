// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_,@function
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 48, v1 // (op 913) 
v_and_b32 v3, 63, v1 // (op 913) 
v_sub_u32 v4, v3, v0 // (op 913) 

v_lshrrev_b32 v3, 1, v4 // (op 913) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 913) 


v_and_b32 v0, 1, v4 // (op 913) 

v_lshlrev_b32 v3, 3, v0 // (op 913) 

v_lshrrev_b32 v0, 6, v1 // (op 913) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 913) 


v_add_u32 v0, v5, v4 // (op 913) 


v_and_b32 v3, 15, v1 // (op 965) 
v_and_b32 v4, 48, v1 // (op 965) 
v_lshl_add_u32 v5, v3, 8, v4 // (op 965) 


v_lshrrev_b32 v3, 6, v1 // (op 1018) 
v_and_b32 v4, 15, v1 // (op 1018) 
v_lshl_add_u32 v6, v3, 4, v4 // (op 1018) 


v_and_b32 v3, 48, v1 // (op 1018) 
v_add_u32 v4, v3, v6 // (op 1018) 


v_and_b32 v3, 15, v1 // (op 1071) 
v_and_b32 v6, 63, v1 // (op 1071) 
v_lshrrev_b32 v7, 4, v6 // (op 1071) 

v_lshl_add_u32 v6, v3, 3, v7 // (op 1071) 


v_and_b32 v3, 48, v1 // (op 1144) 
v_and_b32 v7, 63, v1 // (op 1144) 
v_sub_u32 v8, v7, v3 // (op 1144) 

v_lshrrev_b32 v7, 1, v8 // (op 1144) 
v_add_lshl_u32 v9, v3, v7, 4 // (op 1144) 


v_and_b32 v3, 1, v8 // (op 1144) 

v_lshlrev_b32 v7, 3, v3 // (op 1144) 

v_lshrrev_b32 v3, 6, v1 // (op 1144) 
v_lshl_add_u32 v8, v3, 4, v7 // (op 1144) 


v_add_u32 v3, v9, v8 // (op 1144) 


v_and_b32 v7, 15, v1 // (op 1183) 
v_and_b32 v8, 48, v1 // (op 1183) 
v_lshl_add_u32 v9, v7, 8, v8 // (op 1183) 


v_lshrrev_b32 v7, 6, v1 // (op 1207) 
v_and_b32 v8, 15, v1 // (op 1207) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 1207) 


v_and_b32 v7, 48, v1 // (op 1207) 
v_add_u32 v8, v7, v10 // (op 1207) 


v_and_b32 v7, 15, v1 // (op 1231) 
v_and_b32 v10, 63, v1 // (op 1231) 
v_lshrrev_b32 v11, 4, v10 // (op 1231) 

v_lshl_add_u32 v10, v7, 3, v11 // (op 1231) 


v_mov_b32 v7, s6 // (op 2) Move value
v_mov_b32 v11, s7 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_mul_hi_i32 s0, s16, s9 // (op 624) 
s_add_i32 s1, s0, s16 // (op 624) 

s_ashr_i32 s0, s1, 31 // (op 624) 
s_cmp_eq_i32 s9, 0 // (op 624) 
s_mov_b32 s6, scc // (op 624) 
s_cmp_eq_i32 s6, 0 // (op 624) 
s_cselect_b32 s7, 0, -1 // (op 624) 

s_and_b32 s6, s10, 31 // (op 624) 
s_lshl_b32 s18, 1, s6 // (op 624) 
s_add_i32 s19, s18, s7 // (op 624) 


s_and_b32 s7, s0, s19 // (op 624) 


s_add_i32 s0, s1, s7 // (op 624) 


s_ashr_i32 s1, s0, s6 // (op 624) 


s_lshl_b32 s0, s10, 24 // (op 624) 
s_ashr_i32 s6, s0, 31 // (op 624) 

s_xor_b32 s0, s1, s6 // (op 624) 

s_sub_i32 s1, s0, s6 // (op 624) 


s_mul_hi_i32 s0, s1, s11 // (op 624) 
s_add_i32 s6, s0, s1 // (op 624) 

s_ashr_i32 s0, s6, 31 // (op 624) 
s_cmp_eq_i32 s11, 0 // (op 624) 
s_mov_b32 s7, scc // (op 624) 
s_cmp_eq_i32 s7, 0 // (op 624) 
s_cselect_b32 s18, 0, -1 // (op 624) 

s_and_b32 s7, s12, 31 // (op 624) 
s_lshl_b32 s19, 1, s7 // (op 624) 
s_add_i32 s51, s19, s18 // (op 624) 


s_and_b32 s18, s0, s51 // (op 624) 


s_add_i32 s0, s6, s18 // (op 624) 


s_ashr_i32 s6, s0, s7 // (op 624) 


s_lshl_b32 s0, s12, 24 // (op 624) 
s_ashr_i32 s7, s0, 31 // (op 624) 

s_xor_b32 s0, s6, s7 // (op 624) 

s_sub_i32 s6, s0, s7 // (op 624) 


s_mul_i32 s0, s1, s14 // (op 624) 

s_sub_u32 s1, s16, s0 // (op 624) 

s_mul_i32 s0, s6, s14 // (op 624) 
s_add_u32 s0, s0, s1 // (op 624) 


s_mul_hi_i32 s1, s16, s8 // (op 624) 
s_add_i32 s6, s1, s16 // (op 624) 

s_ashr_i32 s1, s6, 31 // (op 624) 
s_cmp_eq_i32 s8, 0 // (op 624) 
s_mov_b32 s7, scc // (op 624) 
s_cmp_eq_i32 s7, 0 // (op 624) 
s_cselect_b32 s18, 0, -1 // (op 624) 

s_and_b32 s7, s15, 31 // (op 624) 
s_lshl_b32 s19, 1, s7 // (op 624) 
s_add_i32 s51, s19, s18 // (op 624) 


s_and_b32 s18, s1, s51 // (op 624) 


s_add_i32 s1, s6, s18 // (op 624) 


s_ashr_i32 s6, s1, s7 // (op 624) 


s_lshl_b32 s1, s15, 24 // (op 624) 
s_ashr_i32 s7, s1, 31 // (op 624) 

s_xor_b32 s1, s6, s7 // (op 624) 

s_sub_i32 s6, s1, s7 // (op 624) 


s_add_u32 s18, s4, 16 // (op 624) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 624) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 624) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 624) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 624) 

s_mul_i32 s1, s14, s18 // (op 624) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 624) 

s_mul_i32 s1, s6, s7 // (op 624) 

s_sub_u32 s7, s16, s1 // (op 624) 

s_mov_b32 s1, s7 // (op 624) 
s_mul_hi_i32 s18, s1, s13 // (op 624) 
s_add_i32 s19, s18, s1 // (op 624) 


s_ashr_i32 s1, s19, 31 // (op 624) 
s_cmp_eq_i32 s13, 0 // (op 624) 
s_mov_b32 s18, scc // (op 624) 
s_cmp_eq_i32 s18, 0 // (op 624) 
s_cselect_b32 s51, 0, -1 // (op 624) 

s_and_b32 s18, s49, 31 // (op 624) 
s_lshl_b32 s52, 1, s18 // (op 624) 
s_add_i32 s53, s52, s51 // (op 624) 


s_and_b32 s51, s1, s53 // (op 624) 


s_add_i32 s1, s19, s51 // (op 624) 


s_ashr_i32 s19, s1, s18 // (op 624) 


s_lshl_b32 s1, s49, 24 // (op 624) 
s_ashr_i32 s18, s1, 31 // (op 624) 

s_xor_b32 s1, s19, s18 // (op 624) 

s_sub_i32 s19, s1, s18 // (op 624) 


s_mul_i32 s1, s48, s14 // (op 624) 
s_add_u32 s52, s20, 16 // (op 624) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 624) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 624) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 624) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 624) 

s_sub_i32 s18, s52, s1 // (op 624) 

s_mul_i32 s51, s19, s18 // (op 624) 


s_sub_u32 s18, s7, s51 // (op 624) 


s_add_u32 s7, s18, s1 // (op 624) 


s_cmp_eq_i32 s6, 0 // (op 624) 
s_mov_b32 s1, scc // (op 624) 

s_cmp_eq_i32 s1, 0 // (op 624) 
s_cselect_b32 s6, s7, s0 // (op 624) 



s_lshl_b32 s7, s6, 4 // (op 624) 

s_mov_b32 s1, 0 // (op 624) convert
s_mov_b32 s0, s7 // (op 624) convert

v_mul_lo_u32 v13, v1, s3 // (op 624) most significant: low of low * high
v_mul_hi_u32 v14, v1, s2 // (op 624) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 624) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 624) most significant: sum

v_mov_b32 v14, s0 // (op 624) 
v_mov_b32 v15, s1 // (op 624) 
v_add_co_u32 v16, s[6:7], v14, v12 // (op 624) least significant half
v_addc_co_u32 v17, s[6:7], v15, v13, s[6:7] // (op 624) most significant half





s_mul_i32 s1, s2, s25 // (op 626) most significant: low of low * high
s_mul_i32 s6, s3, s24 // (op 626) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 626) least significant
s_mul_hi_u32 s7, s2, s24 // (op 626) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 626) most significant: sum
s_add_u32 s1, s1, s7 // (op 626) most significant: sum


s_mov_b32 s52, s22 // (op 626) reg expression
s_mov_b32 s53, s23 // (op 626) reg expression
s_mov_b32 s54, s0 // (op 626) reg expression
s_mov_b32 s55, 131072 // (op 626) call()

s_mul_hi_i32 s51, s16, s9 // (op 643) 
s_add_i32 s0, s51, s16 // (op 643) 

s_ashr_i32 s1, s0, 31 // (op 643) 
s_cmp_eq_i32 s9, 0 // (op 643) 
s_mov_b32 s51, scc // (op 643) 
s_cmp_eq_i32 s51, 0 // (op 643) 
s_cselect_b32 s6, 0, -1 // (op 643) 

s_and_b32 s7, s10, 31 // (op 643) 
s_lshl_b32 s51, 1, s7 // (op 643) 
s_add_i32 s18, s51, s6 // (op 643) 


s_and_b32 s6, s1, s18 // (op 643) 


s_add_i32 s1, s0, s6 // (op 643) 


s_ashr_i32 s0, s1, s7 // (op 643) 


s_lshl_b32 s1, s10, 24 // (op 643) 
s_ashr_i32 s51, s1, 31 // (op 643) 

s_xor_b32 s1, s0, s51 // (op 643) 

s_sub_i32 s0, s1, s51 // (op 643) 


s_mul_hi_i32 s1, s0, s11 // (op 643) 
s_add_i32 s51, s1, s0 // (op 643) 

s_ashr_i32 s1, s51, 31 // (op 643) 
s_cmp_eq_i32 s11, 0 // (op 643) 
s_mov_b32 s6, scc // (op 643) 
s_cmp_eq_i32 s6, 0 // (op 643) 
s_cselect_b32 s7, 0, -1 // (op 643) 

s_and_b32 s6, s12, 31 // (op 643) 
s_lshl_b32 s18, 1, s6 // (op 643) 
s_add_i32 s19, s18, s7 // (op 643) 


s_and_b32 s7, s1, s19 // (op 643) 


s_add_i32 s1, s51, s7 // (op 643) 


s_ashr_i32 s7, s1, s6 // (op 643) 


s_lshl_b32 s1, s12, 24 // (op 643) 
s_ashr_i32 s6, s1, 31 // (op 643) 

s_xor_b32 s1, s7, s6 // (op 643) 

s_sub_i32 s7, s1, s6 // (op 643) 


s_mul_i32 s1, s0, s14 // (op 643) 

s_sub_u32 s0, s16, s1 // (op 643) 

s_mul_i32 s1, s7, s14 // (op 643) 
s_add_u32 s1, s1, s0 // (op 643) 


s_mul_hi_i32 s0, s16, s8 // (op 643) 
s_add_i32 s51, s0, s16 // (op 643) 

s_ashr_i32 s0, s51, 31 // (op 643) 
s_cmp_eq_i32 s8, 0 // (op 643) 
s_mov_b32 s6, scc // (op 643) 
s_cmp_eq_i32 s6, 0 // (op 643) 
s_cselect_b32 s7, 0, -1 // (op 643) 

s_and_b32 s6, s15, 31 // (op 643) 
s_lshl_b32 s18, 1, s6 // (op 643) 
s_add_i32 s19, s18, s7 // (op 643) 


s_and_b32 s7, s0, s19 // (op 643) 


s_add_i32 s0, s51, s7 // (op 643) 


s_ashr_i32 s7, s0, s6 // (op 643) 


s_lshl_b32 s0, s15, 24 // (op 643) 
s_ashr_i32 s6, s0, 31 // (op 643) 

s_xor_b32 s0, s7, s6 // (op 643) 

s_sub_i32 s7, s0, s6 // (op 643) 


s_add_u32 s18, s4, 16 // (op 643) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 643) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 643) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 643) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 643) 

s_mul_i32 s0, s14, s18 // (op 643) 

s_mul_i32 s6, s50, s0 // (op 643) 

s_mul_i32 s0, s7, s6 // (op 643) 

s_sub_u32 s6, s16, s0 // (op 643) 

s_mov_b32 s0, s6 // (op 643) 
s_mul_hi_i32 s51, s0, s13 // (op 643) 
s_add_i32 s18, s51, s0 // (op 643) 


s_ashr_i32 s0, s18, 31 // (op 643) 
s_cmp_eq_i32 s13, 0 // (op 643) 
s_mov_b32 s19, scc // (op 643) 
s_cmp_eq_i32 s19, 0 // (op 643) 
s_cselect_b32 s51, 0, -1 // (op 643) 

s_and_b32 s19, s49, 31 // (op 643) 
s_lshl_b32 s56, 1, s19 // (op 643) 
s_add_i32 s57, s56, s51 // (op 643) 


s_and_b32 s51, s0, s57 // (op 643) 


s_add_i32 s0, s18, s51 // (op 643) 


s_ashr_i32 s18, s0, s19 // (op 643) 


s_lshl_b32 s0, s49, 24 // (op 643) 
s_ashr_i32 s19, s0, 31 // (op 643) 

s_xor_b32 s0, s18, s19 // (op 643) 

s_sub_i32 s18, s0, s19 // (op 643) 


s_mul_i32 s0, s48, s14 // (op 643) 
s_add_u32 s56, s20, 16 // (op 643) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 643) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 643) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 643) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 643) 

s_sub_i32 s19, s56, s0 // (op 643) 

s_mul_i32 s51, s18, s19 // (op 643) 


s_sub_u32 s18, s6, s51 // (op 643) 


s_add_u32 s6, s18, s0 // (op 643) 


s_cmp_eq_i32 s7, 0 // (op 643) 
s_mov_b32 s0, scc // (op 643) 

s_cmp_eq_i32 s0, 0 // (op 643) 
s_cselect_b32 s7, s6, s1 // (op 643) 



s_lshl_b32 s6, s7, 4 // (op 643) 

s_mov_b32 s1, 0 // (op 643) convert
s_mov_b32 s0, s6 // (op 643) convert

v_mul_lo_u32 v13, v1, s3 // (op 643) most significant: low of low * high
v_mul_hi_u32 v14, v1, s2 // (op 643) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 643) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 643) most significant: sum

v_mov_b32 v14, s0 // (op 643) 
v_mov_b32 v15, s1 // (op 643) 
v_add_co_u32 v18, s[6:7], v14, v12 // (op 643) least significant half
v_addc_co_u32 v19, s[6:7], v15, v13, s[6:7] // (op 643) most significant half





s_mul_i32 s1, s2, s25 // (op 645) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 645) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 645) least significant
s_mul_hi_u32 s6, s2, s24 // (op 645) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 645) most significant: sum
s_add_u32 s1, s1, s6 // (op 645) most significant: sum


s_mov_b32 s52, s22 // (op 645) reg expression
s_mov_b32 s53, s23 // (op 645) reg expression
s_mov_b32 s54, s0 // (op 645) reg expression
s_mov_b32 s55, 131072 // (op 645) call()

s_mul_hi_i32 s51, s16, s9 // (op 688) 
v_lshrrev_b32 v12, 4, v1 // (op 688) 
s_add_i32 s0, s51, s16 // (op 688) 

v_and_b32 v13, -16, v1 // (op 688) 
s_ashr_i32 s1, s0, 31 // (op 688) 
v_sub_u32 v14, v1, v13 // (op 688) 

s_cmp_eq_i32 s9, 0 // (op 688) 
s_mov_b32 s51, scc // (op 688) 
v_lshlrev_b32 v13, 4, v14 // (op 688) 

s_cmp_eq_i32 s51, 0 // (op 688) 
s_cselect_b32 s6, 0, -1 // (op 688) 

v_mov_b32 v15, 0 // (op 688) convert
v_mov_b32 v14, v13 // (op 688) convert

s_and_b32 s7, s10, 31 // (op 688) 
s_lshl_b32 s51, 1, s7 // (op 688) 
s_add_i32 s18, s51, s6 // (op 688) 


s_and_b32 s6, s1, s18 // (op 688) 


s_add_i32 s1, s0, s6 // (op 688) 


s_ashr_i32 s0, s1, s7 // (op 688) 


s_lshl_b32 s1, s10, 24 // (op 688) 
s_ashr_i32 s51, s1, 31 // (op 688) 

s_xor_b32 s1, s0, s51 // (op 688) 

s_sub_i32 s0, s1, s51 // (op 688) 


s_mul_hi_i32 s1, s0, s11 // (op 688) 
s_add_i32 s51, s1, s0 // (op 688) 

s_ashr_i32 s1, s51, 31 // (op 688) 
s_cmp_eq_i32 s11, 0 // (op 688) 
s_mov_b32 s6, scc // (op 688) 
s_cmp_eq_i32 s6, 0 // (op 688) 
s_cselect_b32 s7, 0, -1 // (op 688) 

s_and_b32 s6, s12, 31 // (op 688) 
s_lshl_b32 s18, 1, s6 // (op 688) 
s_add_i32 s19, s18, s7 // (op 688) 


s_and_b32 s7, s1, s19 // (op 688) 


s_add_i32 s1, s51, s7 // (op 688) 


s_ashr_i32 s7, s1, s6 // (op 688) 


s_lshl_b32 s1, s12, 24 // (op 688) 
s_ashr_i32 s6, s1, 31 // (op 688) 

s_xor_b32 s1, s7, s6 // (op 688) 

s_sub_i32 s7, s1, s6 // (op 688) 


s_add_u32 s18, s4, 16 // (op 688) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 688) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 688) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 688) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 688) 

s_mul_i32 s1, s7, s18 // (op 688) 

s_sub_i32 s51, s0, s1 // (op 688) 


s_mul_hi_i32 s0, s16, s8 // (op 688) 
s_add_i32 s1, s0, s16 // (op 688) 

s_ashr_i32 s0, s1, 31 // (op 688) 
s_cmp_eq_i32 s8, 0 // (op 688) 
s_mov_b32 s6, scc // (op 688) 
s_cmp_eq_i32 s6, 0 // (op 688) 
s_cselect_b32 s7, 0, -1 // (op 688) 

s_and_b32 s6, s15, 31 // (op 688) 
s_lshl_b32 s56, 1, s6 // (op 688) 
s_add_i32 s57, s56, s7 // (op 688) 


s_and_b32 s7, s0, s57 // (op 688) 


s_add_i32 s0, s1, s7 // (op 688) 


s_ashr_i32 s1, s0, s6 // (op 688) 


s_lshl_b32 s0, s15, 24 // (op 688) 
s_ashr_i32 s6, s0, 31 // (op 688) 

s_xor_b32 s0, s1, s6 // (op 688) 

s_sub_i32 s1, s0, s6 // (op 688) 


s_mul_i32 s0, s14, s18 // (op 688) 

s_mul_i32 s6, s50, s0 // (op 688) 

s_mul_i32 s0, s1, s6 // (op 688) 

s_sub_u32 s6, s16, s0 // (op 688) 

s_mul_hi_i32 s0, s6, s13 // (op 688) 
s_add_i32 s7, s0, s6 // (op 688) 


s_ashr_i32 s0, s7, 31 // (op 688) 
s_cmp_eq_i32 s13, 0 // (op 688) 
s_mov_b32 s6, scc // (op 688) 
s_cmp_eq_i32 s6, 0 // (op 688) 
s_cselect_b32 s18, 0, -1 // (op 688) 

s_and_b32 s6, s49, 31 // (op 688) 
s_lshl_b32 s19, 1, s6 // (op 688) 
s_add_i32 s56, s19, s18 // (op 688) 


s_and_b32 s18, s0, s56 // (op 688) 


s_add_i32 s0, s7, s18 // (op 688) 


s_ashr_i32 s7, s0, s6 // (op 688) 


s_lshl_b32 s0, s49, 24 // (op 688) 
s_ashr_i32 s6, s0, 31 // (op 688) 

s_xor_b32 s0, s7, s6 // (op 688) 

s_sub_i32 s7, s0, s6 // (op 688) 


s_cmp_eq_i32 s1, 0 // (op 688) 
s_mov_b32 s0, scc // (op 688) 

s_cmp_eq_i32 s0, 0 // (op 688) 
s_cselect_b32 s1, s7, s51 // (op 688) 



v_lshl_add_u32 v13, s1, 4, v12 // (op 688) 


v_mul_lo_u32 v21, v13, s27 // (op 688) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 688) most significant: high of low * low
v_mul_lo_u32 v20, v13, s26 // (op 688) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 688) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 688) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 688) most significant half



s_mul_i32 s1, s26, s5 // (op 690) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 690) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 690) least significant
s_mul_hi_u32 s6, s26, s4 // (op 690) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 690) most significant: sum
s_add_u32 s1, s1, s6 // (op 690) most significant: sum


s_mov_b32 s56, s28 // (op 690) reg expression
s_mov_b32 s57, s29 // (op 690) reg expression
s_mov_b32 s58, s0 // (op 690) reg expression
s_mov_b32 s59, 131072 // (op 690) call()

s_mul_hi_i32 s51, s16, s9 // (op 707) 
v_lshrrev_b32 v14, 4, v1 // (op 707) 
s_add_i32 s0, s51, s16 // (op 707) 

v_and_b32 v15, -16, v1 // (op 707) 
s_ashr_i32 s1, s0, 31 // (op 707) 
v_sub_u32 v20, v1, v15 // (op 707) 

s_cmp_eq_i32 s9, 0 // (op 707) 
s_mov_b32 s51, scc // (op 707) 
v_lshlrev_b32 v15, 4, v20 // (op 707) 

s_cmp_eq_i32 s51, 0 // (op 707) 
s_cselect_b32 s6, 0, -1 // (op 707) 

v_mov_b32 v21, 0 // (op 707) convert
v_mov_b32 v20, v15 // (op 707) convert

s_and_b32 s7, s10, 31 // (op 707) 
s_lshl_b32 s51, 1, s7 // (op 707) 
s_add_i32 s18, s51, s6 // (op 707) 


s_and_b32 s6, s1, s18 // (op 707) 


s_add_i32 s1, s0, s6 // (op 707) 


s_ashr_i32 s0, s1, s7 // (op 707) 


s_lshl_b32 s1, s10, 24 // (op 707) 
s_ashr_i32 s51, s1, 31 // (op 707) 

s_xor_b32 s1, s0, s51 // (op 707) 

s_sub_i32 s0, s1, s51 // (op 707) 


s_mul_hi_i32 s1, s0, s11 // (op 707) 
s_add_i32 s51, s1, s0 // (op 707) 

s_ashr_i32 s1, s51, 31 // (op 707) 
s_cmp_eq_i32 s11, 0 // (op 707) 
s_mov_b32 s6, scc // (op 707) 
s_cmp_eq_i32 s6, 0 // (op 707) 
s_cselect_b32 s7, 0, -1 // (op 707) 

s_and_b32 s6, s12, 31 // (op 707) 
s_lshl_b32 s18, 1, s6 // (op 707) 
s_add_i32 s19, s18, s7 // (op 707) 


s_and_b32 s7, s1, s19 // (op 707) 


s_add_i32 s1, s51, s7 // (op 707) 


s_ashr_i32 s7, s1, s6 // (op 707) 


s_lshl_b32 s1, s12, 24 // (op 707) 
s_ashr_i32 s6, s1, 31 // (op 707) 

s_xor_b32 s1, s7, s6 // (op 707) 

s_sub_i32 s7, s1, s6 // (op 707) 


s_add_u32 s18, s4, 16 // (op 707) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 707) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 707) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 707) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 707) 

s_mul_i32 s1, s7, s18 // (op 707) 

s_sub_i32 s51, s0, s1 // (op 707) 


s_mul_hi_i32 s0, s16, s8 // (op 707) 
s_add_i32 s1, s0, s16 // (op 707) 

s_ashr_i32 s0, s1, 31 // (op 707) 
s_cmp_eq_i32 s8, 0 // (op 707) 
s_mov_b32 s6, scc // (op 707) 
s_cmp_eq_i32 s6, 0 // (op 707) 
s_cselect_b32 s7, 0, -1 // (op 707) 

s_and_b32 s6, s15, 31 // (op 707) 
s_lshl_b32 s60, 1, s6 // (op 707) 
s_add_i32 s61, s60, s7 // (op 707) 


s_and_b32 s7, s0, s61 // (op 707) 


s_add_i32 s0, s1, s7 // (op 707) 


s_ashr_i32 s1, s0, s6 // (op 707) 


s_lshl_b32 s0, s15, 24 // (op 707) 
s_ashr_i32 s6, s0, 31 // (op 707) 

s_xor_b32 s0, s1, s6 // (op 707) 

s_sub_i32 s1, s0, s6 // (op 707) 


s_mul_i32 s0, s14, s18 // (op 707) 

s_mul_i32 s6, s50, s0 // (op 707) 

s_mul_i32 s0, s1, s6 // (op 707) 

s_sub_u32 s6, s16, s0 // (op 707) 

s_mul_hi_i32 s0, s6, s13 // (op 707) 
s_add_i32 s7, s0, s6 // (op 707) 


s_ashr_i32 s0, s7, 31 // (op 707) 
s_cmp_eq_i32 s13, 0 // (op 707) 
s_mov_b32 s6, scc // (op 707) 
s_cmp_eq_i32 s6, 0 // (op 707) 
s_cselect_b32 s18, 0, -1 // (op 707) 

s_and_b32 s6, s49, 31 // (op 707) 
s_lshl_b32 s19, 1, s6 // (op 707) 
s_add_i32 s60, s19, s18 // (op 707) 


s_and_b32 s18, s0, s60 // (op 707) 


s_add_i32 s0, s7, s18 // (op 707) 


s_ashr_i32 s7, s0, s6 // (op 707) 


s_lshl_b32 s0, s49, 24 // (op 707) 
s_ashr_i32 s6, s0, 31 // (op 707) 

s_xor_b32 s0, s7, s6 // (op 707) 

s_sub_i32 s7, s0, s6 // (op 707) 


s_cmp_eq_i32 s1, 0 // (op 707) 
s_mov_b32 s0, scc // (op 707) 

s_cmp_eq_i32 s0, 0 // (op 707) 
s_cselect_b32 s1, s7, s51 // (op 707) 



v_lshl_add_u32 v15, s1, 4, v14 // (op 707) 


v_mul_lo_u32 v23, v15, s27 // (op 707) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 707) most significant: high of low * low
v_mul_lo_u32 v22, v15, s26 // (op 707) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 707) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 707) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 707) most significant half



s_mul_i32 s1, s26, s5 // (op 709) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 709) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 709) least significant
s_mul_hi_u32 s6, s26, s4 // (op 709) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 709) most significant: sum
s_add_u32 s1, s1, s6 // (op 709) most significant: sum


s_mov_b32 s56, s28 // (op 709) reg expression
s_mov_b32 s57, s29 // (op 709) reg expression
s_mov_b32 s58, s0 // (op 709) reg expression
s_mov_b32 s59, 131072 // (op 709) call()

s_mul_hi_i32 s51, s16, s9 // (op 752) 
v_and_b32 v20, -16, v1 // (op 752) 
s_add_i32 s0, s51, s16 // (op 752) 

v_sub_u32 v21, v1, v20 // (op 752) 

s_ashr_i32 s1, s0, 31 // (op 752) 
v_lshrrev_b32 v20, 4, v1 // (op 752) 
s_cmp_eq_i32 s9, 0 // (op 752) 
s_mov_b32 s51, scc // (op 752) 
v_lshlrev_b32 v22, 1, v20 // (op 752) 

s_cmp_eq_i32 s51, 0 // (op 752) 
s_cselect_b32 s6, 0, -1 // (op 752) 

s_and_b32 s7, s10, 31 // (op 752) 
s_lshl_b32 s51, 1, s7 // (op 752) 
s_add_i32 s18, s51, s6 // (op 752) 


s_and_b32 s6, s1, s18 // (op 752) 


s_add_i32 s1, s0, s6 // (op 752) 


s_ashr_i32 s0, s1, s7 // (op 752) 


s_lshl_b32 s1, s10, 24 // (op 752) 
s_ashr_i32 s51, s1, 31 // (op 752) 

s_xor_b32 s1, s0, s51 // (op 752) 

s_sub_i32 s0, s1, s51 // (op 752) 


s_mul_hi_i32 s1, s0, s11 // (op 752) 
s_add_i32 s51, s1, s0 // (op 752) 

s_ashr_i32 s1, s51, 31 // (op 752) 
s_cmp_eq_i32 s11, 0 // (op 752) 
s_mov_b32 s6, scc // (op 752) 
s_cmp_eq_i32 s6, 0 // (op 752) 
s_cselect_b32 s7, 0, -1 // (op 752) 

s_and_b32 s6, s12, 31 // (op 752) 
s_lshl_b32 s18, 1, s6 // (op 752) 
s_add_i32 s19, s18, s7 // (op 752) 


s_and_b32 s7, s1, s19 // (op 752) 


s_add_i32 s1, s51, s7 // (op 752) 


s_ashr_i32 s7, s1, s6 // (op 752) 


s_lshl_b32 s1, s12, 24 // (op 752) 
s_ashr_i32 s6, s1, 31 // (op 752) 

s_xor_b32 s1, s7, s6 // (op 752) 

s_sub_i32 s7, s1, s6 // (op 752) 


s_mul_i32 s1, s0, s14 // (op 752) 

s_sub_u32 s0, s16, s1 // (op 752) 

s_mul_i32 s1, s7, s14 // (op 752) 
s_add_u32 s1, s1, s0 // (op 752) 


s_mul_hi_i32 s0, s16, s8 // (op 752) 
s_add_i32 s51, s0, s16 // (op 752) 

s_ashr_i32 s0, s51, 31 // (op 752) 
s_cmp_eq_i32 s8, 0 // (op 752) 
s_mov_b32 s6, scc // (op 752) 
s_cmp_eq_i32 s6, 0 // (op 752) 
s_cselect_b32 s7, 0, -1 // (op 752) 

s_and_b32 s6, s15, 31 // (op 752) 
s_lshl_b32 s18, 1, s6 // (op 752) 
s_add_i32 s19, s18, s7 // (op 752) 


s_and_b32 s7, s0, s19 // (op 752) 


s_add_i32 s0, s51, s7 // (op 752) 


s_ashr_i32 s7, s0, s6 // (op 752) 


s_lshl_b32 s0, s15, 24 // (op 752) 
s_ashr_i32 s6, s0, 31 // (op 752) 

s_xor_b32 s0, s7, s6 // (op 752) 

s_sub_i32 s7, s0, s6 // (op 752) 


s_add_u32 s18, s4, 16 // (op 752) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 752) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 752) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 752) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 752) 

s_mul_i32 s0, s14, s18 // (op 752) 

s_mul_i32 s6, s50, s0 // (op 752) 

s_mul_i32 s0, s7, s6 // (op 752) 

s_sub_u32 s6, s16, s0 // (op 752) 

s_mov_b32 s0, s6 // (op 752) 
s_mul_hi_i32 s51, s0, s13 // (op 752) 
s_add_i32 s18, s51, s0 // (op 752) 


s_ashr_i32 s0, s18, 31 // (op 752) 
s_cmp_eq_i32 s13, 0 // (op 752) 
s_mov_b32 s19, scc // (op 752) 
s_cmp_eq_i32 s19, 0 // (op 752) 
s_cselect_b32 s51, 0, -1 // (op 752) 

s_and_b32 s19, s49, 31 // (op 752) 
s_lshl_b32 s60, 1, s19 // (op 752) 
s_add_i32 s61, s60, s51 // (op 752) 


s_and_b32 s51, s0, s61 // (op 752) 


s_add_i32 s0, s18, s51 // (op 752) 


s_ashr_i32 s18, s0, s19 // (op 752) 


s_lshl_b32 s0, s49, 24 // (op 752) 
s_ashr_i32 s19, s0, 31 // (op 752) 

s_xor_b32 s0, s18, s19 // (op 752) 

s_sub_i32 s18, s0, s19 // (op 752) 


s_mul_i32 s0, s48, s14 // (op 752) 
s_add_u32 s60, s20, 16 // (op 752) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 752) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 752) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 752) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 752) 

s_sub_i32 s19, s60, s0 // (op 752) 

s_mul_i32 s51, s18, s19 // (op 752) 


s_sub_u32 s18, s6, s51 // (op 752) 


s_add_u32 s6, s18, s0 // (op 752) 


s_cmp_eq_i32 s7, 0 // (op 752) 
s_mov_b32 s0, scc // (op 752) 

s_cmp_eq_i32 s0, 0 // (op 752) 
s_cselect_b32 s7, s6, s1 // (op 752) 



v_lshl_add_u32 v20, s7, 4, v21 // (op 752) 


v_mul_lo_u32 v21, v22, s30 // (op 752) 

v_add_u32 v22, v20, v21 // (op 752) 


s_mul_i32 s1, s30, s35 // (op 754) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 754) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 754) least significant
s_mul_hi_u32 s6, s30, s34 // (op 754) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 754) most significant: sum
s_add_u32 s1, s1, s6 // (op 754) most significant: sum


s_mov_b32 s60, s32 // (op 754) reg expression
s_mov_b32 s61, s33 // (op 754) reg expression
s_mov_b32 s62, s0 // (op 754) reg expression
s_mov_b32 s63, 131072 // (op 754) call()

s_mul_hi_i32 s51, s16, s9 // (op 771) 
v_and_b32 v20, -16, v1 // (op 771) 
s_add_i32 s0, s51, s16 // (op 771) 

v_sub_u32 v21, v1, v20 // (op 771) 

s_ashr_i32 s1, s0, 31 // (op 771) 
v_lshrrev_b32 v20, 4, v1 // (op 771) 
s_cmp_eq_i32 s9, 0 // (op 771) 
s_mov_b32 s51, scc // (op 771) 
v_lshlrev_b32 v23, 1, v20 // (op 771) 

s_cmp_eq_i32 s51, 0 // (op 771) 
s_cselect_b32 s6, 0, -1 // (op 771) 

s_and_b32 s7, s10, 31 // (op 771) 
s_lshl_b32 s51, 1, s7 // (op 771) 
s_add_i32 s18, s51, s6 // (op 771) 


s_and_b32 s6, s1, s18 // (op 771) 


s_add_i32 s1, s0, s6 // (op 771) 


s_ashr_i32 s0, s1, s7 // (op 771) 


s_lshl_b32 s1, s10, 24 // (op 771) 
s_ashr_i32 s51, s1, 31 // (op 771) 

s_xor_b32 s1, s0, s51 // (op 771) 

s_sub_i32 s0, s1, s51 // (op 771) 


s_mul_hi_i32 s1, s0, s11 // (op 771) 
s_add_i32 s51, s1, s0 // (op 771) 

s_ashr_i32 s1, s51, 31 // (op 771) 
s_cmp_eq_i32 s11, 0 // (op 771) 
s_mov_b32 s6, scc // (op 771) 
s_cmp_eq_i32 s6, 0 // (op 771) 
s_cselect_b32 s7, 0, -1 // (op 771) 

s_and_b32 s6, s12, 31 // (op 771) 
s_lshl_b32 s18, 1, s6 // (op 771) 
s_add_i32 s19, s18, s7 // (op 771) 


s_and_b32 s7, s1, s19 // (op 771) 


s_add_i32 s1, s51, s7 // (op 771) 


s_ashr_i32 s7, s1, s6 // (op 771) 


s_lshl_b32 s1, s12, 24 // (op 771) 
s_ashr_i32 s6, s1, 31 // (op 771) 

s_xor_b32 s1, s7, s6 // (op 771) 

s_sub_i32 s7, s1, s6 // (op 771) 


s_mul_i32 s1, s0, s14 // (op 771) 

s_sub_u32 s0, s16, s1 // (op 771) 

s_mul_i32 s1, s7, s14 // (op 771) 
s_add_u32 s1, s1, s0 // (op 771) 


s_mul_hi_i32 s0, s16, s8 // (op 771) 
s_add_i32 s51, s0, s16 // (op 771) 

s_ashr_i32 s0, s51, 31 // (op 771) 
s_cmp_eq_i32 s8, 0 // (op 771) 
s_mov_b32 s6, scc // (op 771) 
s_cmp_eq_i32 s6, 0 // (op 771) 
s_cselect_b32 s7, 0, -1 // (op 771) 

s_and_b32 s6, s15, 31 // (op 771) 
s_lshl_b32 s18, 1, s6 // (op 771) 
s_add_i32 s19, s18, s7 // (op 771) 


s_and_b32 s7, s0, s19 // (op 771) 


s_add_i32 s0, s51, s7 // (op 771) 


s_ashr_i32 s7, s0, s6 // (op 771) 


s_lshl_b32 s0, s15, 24 // (op 771) 
s_ashr_i32 s6, s0, 31 // (op 771) 

s_xor_b32 s0, s7, s6 // (op 771) 

s_sub_i32 s7, s0, s6 // (op 771) 


s_add_u32 s18, s4, 16 // (op 771) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 771) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 771) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 771) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 771) 

s_mul_i32 s0, s14, s18 // (op 771) 

s_mul_i32 s6, s50, s0 // (op 771) 

s_mul_i32 s0, s7, s6 // (op 771) 

s_sub_u32 s6, s16, s0 // (op 771) 

s_mov_b32 s0, s6 // (op 771) 
s_mul_hi_i32 s51, s0, s13 // (op 771) 
s_add_i32 s18, s51, s0 // (op 771) 


s_ashr_i32 s0, s18, 31 // (op 771) 
s_cmp_eq_i32 s13, 0 // (op 771) 
s_mov_b32 s19, scc // (op 771) 
s_cmp_eq_i32 s19, 0 // (op 771) 
s_cselect_b32 s51, 0, -1 // (op 771) 

s_and_b32 s19, s49, 31 // (op 771) 
s_lshl_b32 s64, 1, s19 // (op 771) 
s_add_i32 s65, s64, s51 // (op 771) 


s_and_b32 s51, s0, s65 // (op 771) 


s_add_i32 s0, s18, s51 // (op 771) 


s_ashr_i32 s18, s0, s19 // (op 771) 


s_lshl_b32 s0, s49, 24 // (op 771) 
s_ashr_i32 s19, s0, 31 // (op 771) 

s_xor_b32 s0, s18, s19 // (op 771) 

s_sub_i32 s18, s0, s19 // (op 771) 


s_mul_i32 s0, s48, s14 // (op 771) 
s_add_u32 s64, s20, 16 // (op 771) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 771) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 771) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 771) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 771) 

s_sub_i32 s19, s64, s0 // (op 771) 

s_mul_i32 s51, s18, s19 // (op 771) 


s_sub_u32 s18, s6, s51 // (op 771) 


s_add_u32 s6, s18, s0 // (op 771) 


s_cmp_eq_i32 s7, 0 // (op 771) 
s_mov_b32 s0, scc // (op 771) 

s_cmp_eq_i32 s0, 0 // (op 771) 
s_cselect_b32 s7, s6, s1 // (op 771) 



v_lshl_add_u32 v20, s7, 4, v21 // (op 771) 


v_mul_lo_u32 v21, v23, s30 // (op 771) 

v_add_u32 v23, v20, v21 // (op 771) 


s_mul_i32 s1, s30, s35 // (op 773) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 773) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 773) least significant
s_mul_hi_u32 s6, s30, s34 // (op 773) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 773) most significant: sum
s_add_u32 s1, s1, s6 // (op 773) most significant: sum


s_mov_b32 s60, s32 // (op 773) reg expression
s_mov_b32 s61, s33 // (op 773) reg expression
s_mov_b32 s62, s0 // (op 773) reg expression
s_mov_b32 s63, 131072 // (op 773) call()

s_mul_hi_i32 s51, s16, s9 // (op 816) 
v_lshrrev_b32 v20, 3, v1 // (op 816) 
s_add_i32 s0, s51, s16 // (op 816) 

v_lshlrev_b32 v21, 1, v20 // (op 816) 

s_ashr_i32 s1, s0, 31 // (op 816) 
v_and_b32 v20, -8, v1 // (op 816) 
s_cmp_eq_i32 s9, 0 // (op 816) 
s_mov_b32 s51, scc // (op 816) 
v_sub_u32 v24, v1, v20 // (op 816) 

s_cmp_eq_i32 s51, 0 // (op 816) 
s_cselect_b32 s6, 0, -1 // (op 816) 

s_and_b32 s7, s10, 31 // (op 816) 
s_lshl_b32 s51, 1, s7 // (op 816) 
s_add_i32 s18, s51, s6 // (op 816) 


s_and_b32 s6, s1, s18 // (op 816) 


s_add_i32 s1, s0, s6 // (op 816) 


s_ashr_i32 s0, s1, s7 // (op 816) 


s_lshl_b32 s1, s10, 24 // (op 816) 
s_ashr_i32 s51, s1, 31 // (op 816) 

s_xor_b32 s1, s0, s51 // (op 816) 

s_sub_i32 s0, s1, s51 // (op 816) 


s_mul_hi_i32 s1, s0, s11 // (op 816) 
s_add_i32 s51, s1, s0 // (op 816) 

s_ashr_i32 s1, s51, 31 // (op 816) 
s_cmp_eq_i32 s11, 0 // (op 816) 
s_mov_b32 s6, scc // (op 816) 
s_cmp_eq_i32 s6, 0 // (op 816) 
s_cselect_b32 s7, 0, -1 // (op 816) 

s_and_b32 s6, s12, 31 // (op 816) 
s_lshl_b32 s18, 1, s6 // (op 816) 
s_add_i32 s19, s18, s7 // (op 816) 


s_and_b32 s7, s1, s19 // (op 816) 


s_add_i32 s1, s51, s7 // (op 816) 


s_ashr_i32 s7, s1, s6 // (op 816) 


s_lshl_b32 s1, s12, 24 // (op 816) 
s_ashr_i32 s6, s1, 31 // (op 816) 

s_xor_b32 s1, s7, s6 // (op 816) 

s_sub_i32 s7, s1, s6 // (op 816) 


s_add_u32 s18, s4, 16 // (op 816) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 816) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 816) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 816) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 816) 

s_mul_i32 s1, s7, s18 // (op 816) 

s_sub_i32 s51, s0, s1 // (op 816) 


s_mul_hi_i32 s0, s16, s8 // (op 816) 
s_add_i32 s1, s0, s16 // (op 816) 

s_ashr_i32 s0, s1, 31 // (op 816) 
s_cmp_eq_i32 s8, 0 // (op 816) 
s_mov_b32 s6, scc // (op 816) 
s_cmp_eq_i32 s6, 0 // (op 816) 
s_cselect_b32 s7, 0, -1 // (op 816) 

s_and_b32 s6, s15, 31 // (op 816) 
s_lshl_b32 s64, 1, s6 // (op 816) 
s_add_i32 s65, s64, s7 // (op 816) 


s_and_b32 s7, s0, s65 // (op 816) 


s_add_i32 s0, s1, s7 // (op 816) 


s_ashr_i32 s1, s0, s6 // (op 816) 


s_lshl_b32 s0, s15, 24 // (op 816) 
s_ashr_i32 s6, s0, 31 // (op 816) 

s_xor_b32 s0, s1, s6 // (op 816) 

s_sub_i32 s1, s0, s6 // (op 816) 


s_mul_i32 s0, s14, s18 // (op 816) 

s_mul_i32 s6, s50, s0 // (op 816) 

s_mul_i32 s0, s1, s6 // (op 816) 

s_sub_u32 s6, s16, s0 // (op 816) 

s_mul_hi_i32 s0, s6, s13 // (op 816) 
s_add_i32 s7, s0, s6 // (op 816) 


s_ashr_i32 s0, s7, 31 // (op 816) 
s_cmp_eq_i32 s13, 0 // (op 816) 
s_mov_b32 s6, scc // (op 816) 
s_cmp_eq_i32 s6, 0 // (op 816) 
s_cselect_b32 s18, 0, -1 // (op 816) 

s_and_b32 s6, s49, 31 // (op 816) 
s_lshl_b32 s19, 1, s6 // (op 816) 
s_add_i32 s64, s19, s18 // (op 816) 


s_and_b32 s18, s0, s64 // (op 816) 


s_add_i32 s0, s7, s18 // (op 816) 


s_ashr_i32 s7, s0, s6 // (op 816) 


s_lshl_b32 s0, s49, 24 // (op 816) 
s_ashr_i32 s6, s0, 31 // (op 816) 

s_xor_b32 s0, s7, s6 // (op 816) 

s_sub_i32 s7, s0, s6 // (op 816) 


s_cmp_eq_i32 s1, 0 // (op 816) 
s_mov_b32 s0, scc // (op 816) 

s_cmp_eq_i32 s0, 0 // (op 816) 
s_cselect_b32 s1, s7, s51 // (op 816) 



v_lshl_add_u32 v20, s1, 4, v21 // (op 816) 


v_mul_lo_u32 v21, v20, s36 // (op 816) 

v_add_u32 v20, v24, v21 // (op 816) 


s_mul_i32 s1, s36, s5 // (op 818) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 818) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 818) least significant
s_mul_hi_u32 s6, s36, s4 // (op 818) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 818) most significant: sum
s_add_u32 s1, s1, s6 // (op 818) most significant: sum


s_mov_b32 s64, s38 // (op 818) reg expression
s_mov_b32 s65, s39 // (op 818) reg expression
s_mov_b32 s66, s0 // (op 818) reg expression
s_mov_b32 s67, 131072 // (op 818) call()

s_mul_hi_i32 s51, s16, s9 // (op 835) 
v_lshrrev_b32 v21, 3, v1 // (op 835) 
s_add_i32 s0, s51, s16 // (op 835) 

v_lshlrev_b32 v24, 1, v21 // (op 835) 

s_ashr_i32 s1, s0, 31 // (op 835) 
v_and_b32 v21, -8, v1 // (op 835) 
s_cmp_eq_i32 s9, 0 // (op 835) 
s_mov_b32 s51, scc // (op 835) 
v_sub_u32 v25, v1, v21 // (op 835) 

s_cmp_eq_i32 s51, 0 // (op 835) 
s_cselect_b32 s6, 0, -1 // (op 835) 

s_and_b32 s7, s10, 31 // (op 835) 
s_lshl_b32 s51, 1, s7 // (op 835) 
s_add_i32 s18, s51, s6 // (op 835) 


s_and_b32 s6, s1, s18 // (op 835) 


s_add_i32 s1, s0, s6 // (op 835) 


s_ashr_i32 s0, s1, s7 // (op 835) 


s_lshl_b32 s1, s10, 24 // (op 835) 
s_ashr_i32 s51, s1, 31 // (op 835) 

s_xor_b32 s1, s0, s51 // (op 835) 

s_sub_i32 s0, s1, s51 // (op 835) 


s_mul_hi_i32 s1, s0, s11 // (op 835) 
s_add_i32 s51, s1, s0 // (op 835) 

s_ashr_i32 s1, s51, 31 // (op 835) 
s_cmp_eq_i32 s11, 0 // (op 835) 
s_mov_b32 s6, scc // (op 835) 
s_cmp_eq_i32 s6, 0 // (op 835) 
s_cselect_b32 s7, 0, -1 // (op 835) 

s_and_b32 s6, s12, 31 // (op 835) 
s_lshl_b32 s18, 1, s6 // (op 835) 
s_add_i32 s19, s18, s7 // (op 835) 


s_and_b32 s7, s1, s19 // (op 835) 


s_add_i32 s1, s51, s7 // (op 835) 


s_ashr_i32 s7, s1, s6 // (op 835) 


s_lshl_b32 s1, s12, 24 // (op 835) 
s_ashr_i32 s6, s1, 31 // (op 835) 

s_xor_b32 s1, s7, s6 // (op 835) 

s_sub_i32 s7, s1, s6 // (op 835) 


s_add_u32 s18, s4, 16 // (op 835) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 835) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 835) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 835) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 835) 

s_mul_i32 s1, s7, s18 // (op 835) 

s_sub_i32 s51, s0, s1 // (op 835) 


s_mul_hi_i32 s0, s16, s8 // (op 835) 
s_add_i32 s1, s0, s16 // (op 835) 

s_ashr_i32 s0, s1, 31 // (op 835) 
s_cmp_eq_i32 s8, 0 // (op 835) 
s_mov_b32 s6, scc // (op 835) 
s_cmp_eq_i32 s6, 0 // (op 835) 
s_cselect_b32 s7, 0, -1 // (op 835) 

s_and_b32 s6, s15, 31 // (op 835) 
s_lshl_b32 s68, 1, s6 // (op 835) 
s_add_i32 s69, s68, s7 // (op 835) 


s_and_b32 s7, s0, s69 // (op 835) 


s_add_i32 s0, s1, s7 // (op 835) 


s_ashr_i32 s1, s0, s6 // (op 835) 


s_lshl_b32 s0, s15, 24 // (op 835) 
s_ashr_i32 s6, s0, 31 // (op 835) 

s_xor_b32 s0, s1, s6 // (op 835) 

s_sub_i32 s1, s0, s6 // (op 835) 


s_mul_i32 s0, s14, s18 // (op 835) 

s_mul_i32 s6, s50, s0 // (op 835) 

s_mul_i32 s0, s1, s6 // (op 835) 

s_sub_u32 s6, s16, s0 // (op 835) 

s_mul_hi_i32 s0, s6, s13 // (op 835) 
s_add_i32 s7, s0, s6 // (op 835) 


s_ashr_i32 s0, s7, 31 // (op 835) 
s_cmp_eq_i32 s13, 0 // (op 835) 
s_mov_b32 s6, scc // (op 835) 
s_cmp_eq_i32 s6, 0 // (op 835) 
s_cselect_b32 s18, 0, -1 // (op 835) 

s_and_b32 s6, s49, 31 // (op 835) 
s_lshl_b32 s19, 1, s6 // (op 835) 
s_add_i32 s68, s19, s18 // (op 835) 


s_and_b32 s18, s0, s68 // (op 835) 


s_add_i32 s0, s7, s18 // (op 835) 


s_ashr_i32 s7, s0, s6 // (op 835) 


s_lshl_b32 s0, s49, 24 // (op 835) 
s_ashr_i32 s6, s0, 31 // (op 835) 

s_xor_b32 s0, s7, s6 // (op 835) 

s_sub_i32 s7, s0, s6 // (op 835) 


s_cmp_eq_i32 s1, 0 // (op 835) 
s_mov_b32 s0, scc // (op 835) 

s_cmp_eq_i32 s0, 0 // (op 835) 
s_cselect_b32 s1, s7, s51 // (op 835) 



v_lshl_add_u32 v21, s1, 4, v24 // (op 835) 


v_mul_lo_u32 v24, v21, s36 // (op 835) 

v_add_u32 v21, v25, v24 // (op 835) 


s_mul_i32 s1, s36, s5 // (op 837) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 837) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 837) least significant
s_mul_hi_u32 s6, s36, s4 // (op 837) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 837) most significant: sum
s_add_u32 s1, s1, s6 // (op 837) most significant: sum


s_mov_b32 s64, s38 // (op 837) reg expression
s_mov_b32 s65, s39 // (op 837) reg expression
s_mov_b32 s66, s0 // (op 837) reg expression
s_mov_b32 s67, 131072 // (op 837) call()

v_and_b32 v24, -16, v1 // (op 953) 
v_sub_u32 v25, v1, v24 // (op 953) 

v_lshlrev_b32 v24, 4, v25 // (op 953) 

v_lshrrev_b32 v25, 4, v1 // (op 953) 
v_lshl_add_u32 v26, v25, 8, v24 // (op 953) 


s_nop 0
v_readfirstlane_b32 s51, v26 // (op 953) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 2 // (op 573) 
v_mov_b64 v[24:25], s[0:1] // (op 573) convert


v_mov_b64 v[26:27], v[12:13] // (op 573) 
s_mov_b32 m0, s51 // (op 573) reg expression
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 573) Load value direct to lds
v_add_co_u32 v26, s[0:1], v26, v24 // (op 573) least significant half
v_addc_co_u32 v27, s[0:1], 0, v27, s[0:1] // (op 573) most significant half

s_add_u32 m0, m0, 1024 // (op 573) 
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 573) Load value direct to lds
v_add_co_u32 v26, s[0:1], v26, v24 // (op 573) least significant half
v_addc_co_u32 v27, s[0:1], 0, v27, s[0:1] // (op 573) most significant half

s_add_u32 m0, m0, 1024 // (op 573) 
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 573) Load value direct to lds
v_add_co_u32 v26, s[0:1], v26, v24 // (op 573) least significant half
v_addc_co_u32 v27, s[0:1], 0, v27, s[0:1] // (op 573) most significant half

s_add_u32 m0, m0, 1024 // (op 573) 
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 573) Load value direct to lds




v_mov_b64 v[24:25], s[36:37] // (op 427) convert

v_mov_b32 v26, v20 // (op 427) 
buffer_load_ubyte v28, v26, s[64:67], 0 offen offset: 0 // (op 427) Load value
v_add_u32 v26, v26, v24 // (op 427) 
buffer_load_ubyte v29, v26, s[64:67], 0 offen offset: 0 // (op 427) Load value


v_lshlrev_b32 v24, 4, v1 // (op 901) 
s_nop 0
v_readfirstlane_b32 s51, v24 // (op 901) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[2:3], 6 // (op 578) 
v_mov_b64 v[24:25], s[0:1] // (op 578) convert


v_mov_b64 v[26:27], v[16:17] // (op 578) reg expression
s_add_u32 s0, 4096, s51 // (op 578) 
s_mov_b32 m0, s0 // (op 578) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 578) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 578) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 578) most significant half

s_add_u32 m0, m0, 1024 // (op 578) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 578) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 578) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 578) most significant half

s_add_u32 m0, m0, 1024 // (op 578) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 578) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 578) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 578) most significant half

s_add_u32 m0, m0, 1024 // (op 578) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 578) Load value direct to lds





v_mov_b64 v[24:25], s[30:31] // (op 433) convert

v_mov_b32 v26, v22 // (op 433) reg expression
buffer_load_ubyte v30, v26, s[60:63], 0 offen offset: 0 // (op 433) Load value
v_add_u32 v26, v26, v24 // (op 433) 
buffer_load_ubyte v31, v26, s[60:63], 0 offen offset: 0 // (op 433) Load value


v_and_b32 v24, -16, v1 // (op 1171) 
v_sub_u32 v25, v1, v24 // (op 1171) 

v_lshlrev_b32 v24, 4, v25 // (op 1171) 

v_lshrrev_b32 v25, 4, v1 // (op 1171) 
v_lshl_add_u32 v26, v25, 8, v24 // (op 1171) 


s_nop 0
v_readfirstlane_b32 s51, v26 // (op 1171) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 2 // (op 583) 
v_mov_b64 v[24:25], s[0:1] // (op 583) convert


v_mov_b64 v[26:27], v[12:13] // (op 583) 
s_add_u32 s0, 8192, s51 // (op 583) 
s_mov_b32 m0, s0 // (op 583) reg expression
s_mov_b32 s1, 256 // (op 583) call()
buffer_load_dwordx4 v26, s[56:59], s1 offen offset: 0 lds // (op 583) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 583) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 583) most significant half

s_add_u32 m0, m0, 1024 // (op 583) 
buffer_load_dwordx4 v26, s[56:59], s1 offen offset: 0 lds // (op 583) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 583) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 583) most significant half

s_add_u32 m0, m0, 1024 // (op 583) 
buffer_load_dwordx4 v26, s[56:59], s1 offen offset: 0 lds // (op 583) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 583) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 583) most significant half

s_add_u32 m0, m0, 1024 // (op 583) 
buffer_load_dwordx4 v26, s[56:59], s1 offen offset: 0 lds // (op 583) Load value direct to lds







v_mov_b64 v[12:13], s[36:37] // (op 439) convert

v_mov_b32 v24, v20 // (op 439) 
buffer_load_ubyte v26, v24, s[64:67], 0 offen offset:8 // (op 439) Load value
v_add_u32 v24, v24, v12 // (op 439) 
buffer_load_ubyte v27, v24, s[64:67], 0 offen offset:8 // (op 439) Load value



v_lshlrev_b32 v20, 4, v1 // (op 1132) 
s_nop 0
v_readfirstlane_b32 s51, v20 // (op 1132) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[2:3], 6 // (op 585) 
v_mov_b64 v[12:13], s[0:1] // (op 585) convert


s_lshl_b64 s[0:1], s[2:3], 8 // (op 585) 
v_mov_b32 v20, s0 // (op 585) 
v_mov_b32 v24, s1 // (op 585) 
v_add_co_u32 v32, s[6:7], v16, v20 // (op 585) least significant half
v_addc_co_u32 v33, s[6:7], v17, v24, s[6:7] // (op 585) most significant half




s_add_u32 s0, 12288, s51 // (op 585) 
s_mov_b32 m0, s0 // (op 585) reg expression
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 585) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v12 // (op 585) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 585) most significant half

s_add_u32 m0, m0, 1024 // (op 585) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 585) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v12 // (op 585) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 585) most significant half

s_add_u32 m0, m0, 1024 // (op 585) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 585) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v12 // (op 585) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 585) most significant half

s_add_u32 m0, m0, 1024 // (op 585) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 585) Load value direct to lds






v_mov_b64 v[12:13], s[30:31] // (op 445) convert

s_lshl_b32 s51, s30, 3 // (op 445) 
v_add_u32 v20, v22, s51 // (op 445) 

buffer_load_ubyte v16, v20, s[60:63], 0 offen offset: 0 // (op 445) Load value
v_add_u32 v20, v20, v12 // (op 445) 
buffer_load_ubyte v17, v20, s[60:63], 0 offen offset: 0 // (op 445) Load value



v_and_b32 v20, -8, v1 // (op 1059) 
v_sub_u32 v22, v1, v20 // (op 1059) 

v_lshrrev_b32 v20, 3, v1 // (op 1059) 
v_lshl_add_u32 v12, v20, 4, v22 // (op 1059) 





s_waitcnt vmcnt(19)
ds_write_b8 v12, v28 offset:16384 // (op 448) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v12, v29 offset:16392 // (op 448) Store local data 

v_and_b32 v20, -16, v1 // (op 1006) 
v_sub_u32 v22, v1, v20 // (op 1006) 

v_lshrrev_b32 v20, 4, v1 // (op 1006) 
v_lshl_add_u32 v12, v20, 5, v22 // (op 1006) 





s_waitcnt vmcnt(13)
ds_write_b8 v12, v30 offset:16512 // (op 450) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v12, v31 offset:16528 // (op 450) Store local data 

s_barrier  // (op 1298) 



ds_read_b128 v[32:35], v5 // (op 470) Load local data 
ds_read_b128 v[36:39], v5 offset:64 // (op 470) Load local data 



ds_read_b128 v[40:43], v5 offset:128 // (op 473) Load local data 
ds_read_b128 v[44:47], v5 offset:192 // (op 473) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1332) 



ds_read_u8 v20, v6 offset:16384 // (op 482) Load local data 



ds_read_u8 v22, v6 offset:16388 // (op 485) Load local data 
s_barrier  // (op 1282) 



ds_read_b64_tr_b8 v[48:49], v0 offset:4096 // (op 464) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:4224 // (op 464) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:5120 // (op 464) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:5248 // (op 464) Transpose load local data 



ds_read_b64_tr_b8 v[56:57], v0 offset:6144 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:6272 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:7168 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:7296 // (op 467) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1315) 


ds_read_u8 v12, v4 offset:16512 // (op 476) Load local data 


ds_read_u8 v13, v4 offset:16576 // (op 479) Load local data 
v_lshlrev_b32 v24, 4, v1 // (op 873) 
s_nop 0
v_readfirstlane_b32 s51, v24 // (op 873) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 925) 
v_sub_u32 v25, v1, v24 // (op 925) 

v_lshlrev_b32 v24, 4, v25 // (op 925) 

v_lshrrev_b32 v25, 4, v1 // (op 925) 
v_lshl_add_u32 v64, v25, 8, v24 // (op 925) 


s_nop 0
v_readfirstlane_b32 s0, v64 // (op 925) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 977) 
v_sub_u32 v25, v1, v24 // (op 977) 

v_lshrrev_b32 v24, 4, v1 // (op 977) 
v_lshl_add_u32 v64, v24, 5, v25 // (op 977) 


v_and_b32 v24, -8, v1 // (op 1030) 
v_sub_u32 v25, v1, v24 // (op 1030) 

v_lshrrev_b32 v24, 3, v1 // (op 1030) 
v_lshl_add_u32 v65, v24, 4, v25 // (op 1030) 


v_lshlrev_b32 v24, 4, v1 // (op 1117) 
s_nop 0
v_readfirstlane_b32 s1, v24 // (op 1117) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 1156) 
v_sub_u32 v25, v1, v24 // (op 1156) 

v_lshlrev_b32 v24, 4, v25 // (op 1156) 

v_lshrrev_b32 v25, 4, v1 // (op 1156) 
v_lshl_add_u32 v66, v25, 8, v24 // (op 1156) 


s_nop 0
v_readfirstlane_b32 s6, v66 // (op 1156) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 1195) 
v_sub_u32 v25, v1, v24 // (op 1195) 

v_lshrrev_b32 v24, 4, v1 // (op 1195) 
v_lshl_add_u32 v66, v24, 5, v25 // (op 1195) 


v_and_b32 v24, -8, v1 // (op 1219) 
v_sub_u32 v25, v1, v24 // (op 1219) 

v_lshrrev_b32 v24, 3, v1 // (op 1219) 
v_lshl_add_u32 v67, v24, 4, v25 // (op 1219) 


s_mov_b32 s7, 0 // (op 59) call()
s_ashr_i64 s[18:19], s[24:25], 8 // (op 58) 
s_mov_b32 s68, s18 // (op 58) convert

s_and_b32 s18, s68, 4294967294 // (op 58) 

s_cmp_lt_u32 s7, s18 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58 if false)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58:
s_add_u32 s7, s7, 2 // (op 60) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 88) 
s_barrier  // (op 1301) 

s_lshl_b64 s[18:19], s[26:27], 2 // (op 545) 
v_mov_b64 v[24:25], s[18:19] // (op 545) convert


v_mov_b64 v[68:69], v[14:15] // (op 545) 
v_mov_b64 v[70:71], v[68:69] // (op 545) 
s_mov_b32 m0, s0 // (op 545) reg expression
s_mov_b32 s18, 512 // (op 545) call()
buffer_load_dwordx4 v70, s[56:59], s18 offen offset: 0 lds // (op 545) Load value direct to lds
v_add_co_u32 v70, s[68:69], v70, v24 // (op 545) least significant half
v_addc_co_u32 v71, s[68:69], 0, v71, s[68:69] // (op 545) most significant half

s_add_u32 m0, m0, 1024 // (op 545) 
buffer_load_dwordx4 v70, s[56:59], s18 offen offset: 0 lds // (op 545) Load value direct to lds
v_add_co_u32 v70, s[68:69], v70, v24 // (op 545) least significant half
v_addc_co_u32 v71, s[68:69], 0, v71, s[68:69] // (op 545) most significant half

s_add_u32 m0, m0, 1024 // (op 545) 
buffer_load_dwordx4 v70, s[56:59], s18 offen offset: 0 lds // (op 545) Load value direct to lds
v_add_co_u32 v70, s[68:69], v70, v24 // (op 545) least significant half
v_addc_co_u32 v71, s[68:69], 0, v71, s[68:69] // (op 545) most significant half

s_add_u32 m0, m0, 1024 // (op 545) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[40:47], a[0:3], v13, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 105) 
buffer_load_dwordx4 v70, s[56:59], s18 offen offset: 0 lds // (op 545) Load value direct to lds




v_mov_b64 v[24:25], s[36:37] // (op 25) convert

v_mov_b32 v70, v21 // (op 25) 
v_mov_b32 v71, v70 // (op 25) 
buffer_load_ubyte v28, v71, s[64:67], 0 offen offset:16 // (op 25) Load value
v_add_u32 v71, v71, v24 // (op 25) 
buffer_load_ubyte v29, v71, s[64:67], 0 offen offset:16 // (op 25) Load value


s_barrier  // (op 1285) 

s_lshl_b64 s[18:19], s[2:3], 6 // (op 540) 
v_mov_b64 v[24:25], s[18:19] // (op 540) convert


v_mov_b64 v[72:73], v[18:19] // (op 540) 
s_lshl_b64 s[18:19], s[2:3], 8 // (op 540) 
s_lshl_b64 s[68:69], s[18:19], 1 // (op 540) 

v_mov_b32 v71, s68 // (op 540) 
v_mov_b32 v74, s69 // (op 540) 
v_add_co_u32 v76, s[18:19], v72, v71 // (op 540) least significant half
v_addc_co_u32 v77, s[18:19], v73, v74, s[18:19] // (op 540) most significant half




s_add_u32 s18, 4096, s51 // (op 540) 
s_nop 0
s_mov_b32 m0, s18 // (op 540) reg expression
buffer_load_dwordx4 v76, s[52:55], 0 offen offset: 0 lds // (op 540) Load value direct to lds
v_add_co_u32 v76, s[68:69], v76, v24 // (op 540) least significant half
v_addc_co_u32 v77, s[68:69], 0, v77, s[68:69] // (op 540) most significant half

s_add_u32 m0, m0, 1024 // (op 540) 
buffer_load_dwordx4 v76, s[52:55], 0 offen offset: 0 lds // (op 540) Load value direct to lds
v_add_co_u32 v76, s[68:69], v76, v24 // (op 540) least significant half
v_addc_co_u32 v77, s[68:69], 0, v77, s[68:69] // (op 540) most significant half

s_add_u32 m0, m0, 1024 // (op 540) 
buffer_load_dwordx4 v76, s[52:55], 0 offen offset: 0 lds // (op 540) Load value direct to lds
v_add_co_u32 v76, s[68:69], v76, v24 // (op 540) least significant half
v_addc_co_u32 v77, s[68:69], 0, v77, s[68:69] // (op 540) most significant half

s_add_u32 m0, m0, 1024 // (op 540) 
buffer_load_dwordx4 v76, s[52:55], 0 offen offset: 0 lds // (op 540) Load value direct to lds




v_mov_b64 v[24:25], s[30:31] // (op 17) convert

v_mov_b32 v71, v23 // (op 17) 
s_lshl_b32 s18, s30, 4 // (op 17) 
v_add_u32 v74, v71, s18 // (op 17) 

buffer_load_ubyte v30, v74, s[60:63], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v74, v74, v24 // (op 17) 
buffer_load_ubyte v31, v74, s[60:63], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1371) 



s_waitcnt vmcnt(19)
ds_write_b8 v67, v26 offset:16640 // (op 304) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v67, v27 offset:16648 // (op 304) Store local data 
s_barrier  // (op 1365) 



s_waitcnt vmcnt(13)
ds_write_b8 v66, v16 offset:16768 // (op 303) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v66, v17 offset:16784 // (op 303) Store local data 
s_barrier  // (op 1355) 



ds_read_b128 v[32:35], v9 offset:8192 // (op 287) Load local data 
ds_read_b128 v[36:39], v9 offset:8256 // (op 287) Load local data 



ds_read_b128 v[40:43], v9 offset:8320 // (op 296) Load local data 
ds_read_b128 v[44:47], v9 offset:8384 // (op 296) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1368) 



ds_read_u8 v20, v10 offset:16640 // (op 291) Load local data 



ds_read_u8 v22, v10 offset:16644 // (op 300) Load local data 
s_barrier  // (op 1349) 
s_barrier  // (op 1424) 



ds_read_b64_tr_b8 v[48:49], v3 offset:12288 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v3 offset:12416 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v3 offset:13312 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v3 offset:13440 // (op 285) Transpose load local data 



ds_read_b64_tr_b8 v[56:57], v3 offset:14336 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v3 offset:14464 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v3 offset:15360 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v3 offset:15488 // (op 294) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1362) 


ds_read_u8 v12, v8 offset:16768 // (op 289) Load local data 


ds_read_u8 v13, v8 offset:16832 // (op 298) Load local data 
s_barrier  // (op 1420) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 292) 
s_barrier  // (op 1359) 

s_lshl_b64 s[18:19], s[26:27], 2 // (op 568) 
v_mov_b64 v[24:25], s[18:19] // (op 568) convert


v_mov_b64 v[74:75], v[68:69] // (op 568) 
s_add_u32 s18, 8192, s6 // (op 568) 
s_mov_b32 m0, s18 // (op 568) reg expression
s_mov_b32 s19, 768 // (op 568) call()
buffer_load_dwordx4 v74, s[56:59], s19 offen offset: 0 lds // (op 568) Load value direct to lds
v_add_co_u32 v74, s[68:69], v74, v24 // (op 568) least significant half
v_addc_co_u32 v75, s[68:69], 0, v75, s[68:69] // (op 568) most significant half

s_add_u32 m0, m0, 1024 // (op 568) 
buffer_load_dwordx4 v74, s[56:59], s19 offen offset: 0 lds // (op 568) Load value direct to lds
v_add_co_u32 v74, s[68:69], v74, v24 // (op 568) least significant half
v_addc_co_u32 v75, s[68:69], 0, v75, s[68:69] // (op 568) most significant half

s_add_u32 m0, m0, 1024 // (op 568) 
buffer_load_dwordx4 v74, s[56:59], s19 offen offset: 0 lds // (op 568) Load value direct to lds
v_add_co_u32 v74, s[68:69], v74, v24 // (op 568) least significant half
v_addc_co_u32 v75, s[68:69], 0, v75, s[68:69] // (op 568) most significant half

s_add_u32 m0, m0, 1024 // (op 568) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[40:47], a[0:3], v13, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 301) 
buffer_load_dwordx4 v74, s[56:59], s19 offen offset: 0 lds // (op 568) Load value direct to lds






v_add_co_u32 v14, vcc, 512, v14 // (op 699) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 699) most significant half
v_mov_b64 v[24:25], s[36:37] // (op 307) convert

v_mov_b32 v68, v70 // (op 307) 
buffer_load_ubyte v26, v68, s[64:67], 0 offen offset:24 // (op 307) Load value
v_add_u32 v68, v68, v24 // (op 307) 
buffer_load_ubyte v27, v68, s[64:67], 0 offen offset:24 // (op 307) Load value



s_barrier  // (op 1352) 
v_add_u32 v21, 16, v21 // (op 827) 

s_lshl_b64 s[18:19], s[2:3], 6 // (op 563) 
v_mov_b64 v[24:25], s[18:19] // (op 563) convert


s_lshl_b64 s[18:19], s[2:3], 8 // (op 563) 
s_mul_i32 s69, 3, s19 // (op 563) most significant: low of low * high
s_mul_i32 s68, 3, s18 // (op 563) least significant
s_mul_hi_u32 s70, 3, s18 // (op 563) most significant: high of low * low
s_add_u32 s69, s69, s70 // (op 563) most significant: sum


v_mov_b32 v68, s68 // (op 563) 
v_mov_b32 v69, s69 // (op 563) 
v_add_co_u32 v74, s[18:19], v72, v68 // (op 563) least significant half
v_addc_co_u32 v75, s[18:19], v73, v69, s[18:19] // (op 563) most significant half




s_add_u32 s18, 12288, s1 // (op 563) 
s_nop 0
s_mov_b32 m0, s18 // (op 563) reg expression
buffer_load_dwordx4 v74, s[52:55], 0 offen offset: 0 lds // (op 563) Load value direct to lds
v_add_co_u32 v74, s[68:69], v74, v24 // (op 563) least significant half
v_addc_co_u32 v75, s[68:69], 0, v75, s[68:69] // (op 563) most significant half

s_add_u32 m0, m0, 1024 // (op 563) 
buffer_load_dwordx4 v74, s[52:55], 0 offen offset: 0 lds // (op 563) Load value direct to lds
v_add_co_u32 v74, s[68:69], v74, v24 // (op 563) least significant half
v_addc_co_u32 v75, s[68:69], 0, v75, s[68:69] // (op 563) most significant half

s_add_u32 m0, m0, 1024 // (op 563) 
buffer_load_dwordx4 v74, s[52:55], 0 offen offset: 0 lds // (op 563) Load value direct to lds
v_add_co_u32 v74, s[68:69], v74, v24 // (op 563) least significant half
v_addc_co_u32 v75, s[68:69], 0, v75, s[68:69] // (op 563) most significant half

s_add_u32 m0, m0, 1024 // (op 563) 
buffer_load_dwordx4 v74, s[52:55], 0 offen offset: 0 lds // (op 563) Load value direct to lds





v_mov_b64 v[24:25], s[30:31] // (op 306) convert

s_lshl_b32 s18, s30, 3 // (op 306) 
s_mul_i32 s19, 3, s18 // (op 306) 

v_add_u32 v68, v71, s19 // (op 306) 

buffer_load_ubyte v16, v68, s[60:63], 0 offen offset: 0 // (op 306) Load value
v_add_u32 v68, v68, v24 // (op 306) 
s_lshl_b64 s[18:19], s[2:3], 8 // (op 635) 
s_lshl_b64 s[68:69], s[18:19], 1 // (op 635) 

buffer_load_ubyte v17, v68, s[60:63], 0 offen offset: 0 // (op 306) Load value


v_mov_b32 v24, s68 // (op 635) 

v_mov_b32 v25, s69 // (op 635) 
s_barrier  // (op 1335) 
s_lshl_b32 s18, s30, 4 // (op 763) 



v_add_u32 v23, v23, s18 // (op 763) 

v_add_co_u32 v18, s[18:19], v18, v24 // (op 635) least significant half
v_addc_co_u32 v19, s[18:19], v19, v25, s[18:19] // (op 635) most significant half




s_waitcnt vmcnt(19)
ds_write_b8 v65, v28 offset:16384 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v65, v29 offset:16392 // (op 30) Store local data 
s_barrier  // (op 1318) 



s_waitcnt vmcnt(13)
ds_write_b8 v64, v30 offset:16512 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v64, v31 offset:16528 // (op 21) Store local data 
s_barrier  // (op 1307) 



ds_read_b128 v[32:35], v5 // (op 75) Load local data 
ds_read_b128 v[36:39], v5 offset:64 // (op 75) Load local data 



ds_read_b128 v[40:43], v5 offset:128 // (op 95) Load local data 
ds_read_b128 v[44:47], v5 offset:192 // (op 95) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1342) 



ds_read_u8 v20, v6 offset:16384 // (op 85) Load local data 



ds_read_u8 v22, v6 offset:16388 // (op 103) Load local data 
s_barrier  // (op 1291) 



ds_read_b64_tr_b8 v[48:49], v0 offset:4096 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:4224 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:5120 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:5248 // (op 70) Transpose load local data 



ds_read_b64_tr_b8 v[56:57], v0 offset:6144 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:6272 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:7168 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:7296 // (op 91) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1325) 


ds_read_u8 v12, v4 offset:16512 // (op 80) Load local data 


ds_read_u8 v13, v4 offset:16576 // (op 99) Load local data 
s_ashr_i64 s[18:19], s[24:25], 8 // (op 58) 
s_mov_b32 s68, s18 // (op 58) convert

s_and_b32 s18, s68, 4294967294 // (op 58) 

s_cmp_lt_u32 s7, s18 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58 if true)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58:















s_barrier  // (op 1382) 
s_mul_hi_i32 s51, s16, s9 // (op 598) 
s_add_i32 s0, s51, s16 // (op 598) 

s_ashr_i32 s1, s0, 31 // (op 598) 
s_cmp_eq_i32 s9, 0 // (op 598) 
s_mov_b32 s51, scc // (op 598) 
s_cmp_eq_i32 s51, 0 // (op 598) 
s_cselect_b32 s6, 0, -1 // (op 598) 

s_and_b32 s7, s10, 31 // (op 598) 
s_lshl_b32 s51, 1, s7 // (op 598) 
s_add_i32 s18, s51, s6 // (op 598) 


s_and_b32 s6, s1, s18 // (op 598) 


s_add_i32 s1, s0, s6 // (op 598) 


s_ashr_i32 s0, s1, s7 // (op 598) 


s_lshl_b32 s1, s10, 24 // (op 598) 
s_ashr_i32 s51, s1, 31 // (op 598) 

s_xor_b32 s1, s0, s51 // (op 598) 

s_sub_i32 s0, s1, s51 // (op 598) 


s_mul_hi_i32 s1, s0, s11 // (op 598) 
s_add_i32 s51, s1, s0 // (op 598) 

s_ashr_i32 s1, s51, 31 // (op 598) 
s_cmp_eq_i32 s11, 0 // (op 598) 
s_mov_b32 s6, scc // (op 598) 
s_cmp_eq_i32 s6, 0 // (op 598) 
s_cselect_b32 s7, 0, -1 // (op 598) 

s_and_b32 s6, s12, 31 // (op 598) 
s_lshl_b32 s18, 1, s6 // (op 598) 
s_add_i32 s19, s18, s7 // (op 598) 


s_and_b32 s7, s1, s19 // (op 598) 


s_add_i32 s1, s51, s7 // (op 598) 


s_ashr_i32 s7, s1, s6 // (op 598) 


s_lshl_b32 s1, s12, 24 // (op 598) 
s_ashr_i32 s6, s1, 31 // (op 598) 

s_xor_b32 s1, s7, s6 // (op 598) 

s_sub_i32 s7, s1, s6 // (op 598) 


s_mul_i32 s1, s0, s14 // (op 598) 

s_sub_u32 s0, s16, s1 // (op 598) 

s_mul_i32 s1, s7, s14 // (op 598) 
s_add_u32 s1, s1, s0 // (op 598) 


s_mul_hi_i32 s0, s16, s8 // (op 598) 
s_add_i32 s51, s0, s16 // (op 598) 

s_ashr_i32 s0, s51, 31 // (op 598) 
s_cmp_eq_i32 s8, 0 // (op 598) 
s_mov_b32 s6, scc // (op 598) 
s_cmp_eq_i32 s6, 0 // (op 598) 
s_cselect_b32 s7, 0, -1 // (op 598) 

s_and_b32 s6, s15, 31 // (op 598) 
s_lshl_b32 s18, 1, s6 // (op 598) 
s_add_i32 s19, s18, s7 // (op 598) 


s_and_b32 s7, s0, s19 // (op 598) 


s_add_i32 s0, s51, s7 // (op 598) 


s_ashr_i32 s7, s0, s6 // (op 598) 


s_lshl_b32 s0, s15, 24 // (op 598) 
s_ashr_i32 s6, s0, 31 // (op 598) 

s_xor_b32 s0, s7, s6 // (op 598) 

s_sub_i32 s7, s0, s6 // (op 598) 


s_add_u32 s18, s4, 16 // (op 598) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 598) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 598) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 598) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 598) 

s_mul_i32 s0, s14, s18 // (op 598) 

s_mul_i32 s6, s50, s0 // (op 598) 

s_mul_i32 s0, s7, s6 // (op 598) 

s_sub_u32 s6, s16, s0 // (op 598) 

s_mov_b32 s0, s6 // (op 598) 
s_mul_hi_i32 s51, s0, s13 // (op 598) 
s_add_i32 s18, s51, s0 // (op 598) 


s_ashr_i32 s0, s18, 31 // (op 598) 
s_cmp_eq_i32 s13, 0 // (op 598) 
s_mov_b32 s19, scc // (op 598) 
s_cmp_eq_i32 s19, 0 // (op 598) 
s_cselect_b32 s51, 0, -1 // (op 598) 

s_and_b32 s19, s49, 31 // (op 598) 
s_lshl_b32 s68, 1, s19 // (op 598) 
s_add_i32 s69, s68, s51 // (op 598) 


s_and_b32 s51, s0, s69 // (op 598) 


s_add_i32 s0, s18, s51 // (op 598) 


s_ashr_i32 s18, s0, s19 // (op 598) 


s_lshl_b32 s0, s49, 24 // (op 598) 
s_ashr_i32 s19, s0, 31 // (op 598) 

s_xor_b32 s0, s18, s19 // (op 598) 

s_sub_i32 s18, s0, s19 // (op 598) 


s_mul_i32 s0, s48, s14 // (op 598) 
s_add_u32 s68, s20, 16 // (op 598) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 598) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 598) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 598) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 598) 

s_sub_i32 s19, s68, s0 // (op 598) 

s_mul_i32 s51, s18, s19 // (op 598) 


s_sub_u32 s18, s6, s51 // (op 598) 


s_add_u32 s6, s18, s0 // (op 598) 


s_cmp_eq_i32 s7, 0 // (op 598) 
s_mov_b32 s0, scc // (op 598) 

s_cmp_eq_i32 s0, 0 // (op 598) 
s_cselect_b32 s7, s6, s1 // (op 598) 



s_lshl_b32 s6, s7, 4 // (op 598) 

s_mov_b32 s1, 0 // (op 598) convert
s_mov_b32 s0, s6 // (op 598) convert

s_ashr_i64 s[6:7], s[24:25], 8 // (op 598) 
s_lshr_b32 s51, s6, 1 // (op 598) 

v_lshl_add_u32 v21, s51, 9, v1 // (op 598) 

v_mul_lo_u32 v15, v21, s3 // (op 598) most significant: low of low * high
v_mul_hi_u32 v18, v21, s2 // (op 598) most significant: high of low * low
v_mul_lo_u32 v14, v21, s2 // (op 598) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 598) most significant: sum


v_mov_b32 v21, s0 // (op 598) 
v_mov_b32 v18, s1 // (op 598) 
v_add_co_u32 v24, s[6:7], v21, v14 // (op 598) least significant half
v_addc_co_u32 v25, s[6:7], v18, v15, s[6:7] // (op 598) most significant half





s_mul_i32 s1, s2, s25 // (op 600) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 600) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 600) least significant
s_mul_hi_u32 s6, s2, s24 // (op 600) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 600) most significant: sum
s_add_u32 s1, s1, s6 // (op 600) most significant: sum


s_mov_b32 s52, s22 // (op 600) reg expression
s_mov_b32 s53, s23 // (op 600) reg expression
s_mov_b32 s54, s0 // (op 600) reg expression
s_mov_b32 s55, 131072 // (op 600) call()

s_mul_hi_i32 s51, s16, s9 // (op 665) 
v_lshrrev_b32 v21, 4, v1 // (op 665) 
s_add_i32 s0, s51, s16 // (op 665) 

v_and_b32 v23, -16, v1 // (op 665) 
s_ashr_i32 s1, s0, 31 // (op 665) 
v_sub_u32 v14, v1, v23 // (op 665) 

s_cmp_eq_i32 s9, 0 // (op 665) 
s_mov_b32 s51, scc // (op 665) 
v_lshlrev_b32 v15, 4, v14 // (op 665) 

s_cmp_eq_i32 s51, 0 // (op 665) 
s_cselect_b32 s6, 0, -1 // (op 665) 

s_and_b32 s7, s10, 31 // (op 665) 
s_lshl_b32 s51, 1, s7 // (op 665) 
s_add_i32 s18, s51, s6 // (op 665) 


s_and_b32 s6, s1, s18 // (op 665) 


s_add_i32 s1, s0, s6 // (op 665) 


s_ashr_i32 s0, s1, s7 // (op 665) 


s_lshl_b32 s1, s10, 24 // (op 665) 
s_ashr_i32 s51, s1, 31 // (op 665) 

s_xor_b32 s1, s0, s51 // (op 665) 

s_sub_i32 s0, s1, s51 // (op 665) 


s_mul_hi_i32 s1, s0, s11 // (op 665) 
s_add_i32 s51, s1, s0 // (op 665) 

s_ashr_i32 s1, s51, 31 // (op 665) 
s_cmp_eq_i32 s11, 0 // (op 665) 
s_mov_b32 s6, scc // (op 665) 
s_cmp_eq_i32 s6, 0 // (op 665) 
s_cselect_b32 s7, 0, -1 // (op 665) 

s_and_b32 s6, s12, 31 // (op 665) 
s_lshl_b32 s18, 1, s6 // (op 665) 
s_add_i32 s19, s18, s7 // (op 665) 


s_and_b32 s7, s1, s19 // (op 665) 


s_add_i32 s1, s51, s7 // (op 665) 


s_ashr_i32 s7, s1, s6 // (op 665) 


s_lshl_b32 s1, s12, 24 // (op 665) 
s_ashr_i32 s6, s1, 31 // (op 665) 

s_xor_b32 s1, s7, s6 // (op 665) 

s_sub_i32 s7, s1, s6 // (op 665) 


s_add_u32 s18, s4, 16 // (op 665) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 665) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 665) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 665) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 665) 

s_mul_i32 s1, s7, s18 // (op 665) 

s_sub_i32 s51, s0, s1 // (op 665) 


s_mul_hi_i32 s0, s16, s8 // (op 665) 
s_add_i32 s1, s0, s16 // (op 665) 

s_ashr_i32 s0, s1, 31 // (op 665) 
s_cmp_eq_i32 s8, 0 // (op 665) 
s_mov_b32 s6, scc // (op 665) 
s_cmp_eq_i32 s6, 0 // (op 665) 
s_cselect_b32 s7, 0, -1 // (op 665) 

s_and_b32 s6, s15, 31 // (op 665) 
s_lshl_b32 s68, 1, s6 // (op 665) 
s_add_i32 s69, s68, s7 // (op 665) 


s_and_b32 s7, s0, s69 // (op 665) 


s_add_i32 s0, s1, s7 // (op 665) 


s_ashr_i32 s1, s0, s6 // (op 665) 


s_lshl_b32 s0, s15, 24 // (op 665) 
s_ashr_i32 s6, s0, 31 // (op 665) 

s_xor_b32 s0, s1, s6 // (op 665) 

s_sub_i32 s1, s0, s6 // (op 665) 


s_mul_i32 s0, s14, s18 // (op 665) 

s_mul_i32 s6, s50, s0 // (op 665) 

s_mul_i32 s0, s1, s6 // (op 665) 

s_sub_u32 s6, s16, s0 // (op 665) 

s_mul_hi_i32 s0, s6, s13 // (op 665) 
s_add_i32 s7, s0, s6 // (op 665) 


s_ashr_i32 s0, s7, 31 // (op 665) 
s_cmp_eq_i32 s13, 0 // (op 665) 
s_mov_b32 s6, scc // (op 665) 
s_cmp_eq_i32 s6, 0 // (op 665) 
s_cselect_b32 s18, 0, -1 // (op 665) 

s_and_b32 s6, s49, 31 // (op 665) 
s_lshl_b32 s19, 1, s6 // (op 665) 
s_add_i32 s68, s19, s18 // (op 665) 


s_and_b32 s18, s0, s68 // (op 665) 


s_add_i32 s0, s7, s18 // (op 665) 


s_ashr_i32 s7, s0, s6 // (op 665) 


s_lshl_b32 s0, s49, 24 // (op 665) 
s_ashr_i32 s6, s0, 31 // (op 665) 

s_xor_b32 s0, s7, s6 // (op 665) 

s_sub_i32 s7, s0, s6 // (op 665) 


s_cmp_eq_i32 s1, 0 // (op 665) 
s_mov_b32 s0, scc // (op 665) 

s_cmp_eq_i32 s0, 0 // (op 665) 
s_cselect_b32 s1, s7, s51 // (op 665) 



v_lshl_add_u32 v14, s1, 4, v21 // (op 665) 


v_mul_lo_u32 v19, v14, s27 // (op 665) most significant: low of low * high
v_mul_hi_u32 v21, v14, s26 // (op 665) most significant: high of low * low
v_mul_lo_u32 v18, v14, s26 // (op 665) least significant: low of low * low
v_add_u32 v19, v19, v21 // (op 665) most significant: sum


s_ashr_i64 s[0:1], s[24:25], 8 // (op 665) 
s_lshr_b32 s51, s0, 1 // (op 665) 

v_lshl_add_u32 v14, s51, 9, v15 // (op 665) 


v_mov_b32 v29, 0 // (op 665) convert
v_mov_b32 v28, v14 // (op 665) convert

v_add_co_u32 v14, s[0:1], v28, v18 // (op 665) least significant half
v_addc_co_u32 v15, s[0:1], v29, v19, s[0:1] // (op 665) most significant half



s_mul_i32 s1, s26, s5 // (op 667) most significant: low of low * high
s_mul_i32 s51, s27, s4 // (op 667) most significant: low of high * low
s_mul_i32 s0, s26, s4 // (op 667) least significant
s_mul_hi_u32 s6, s26, s4 // (op 667) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 667) most significant: sum
s_add_u32 s1, s1, s6 // (op 667) most significant: sum


s_mov_b32 s56, s28 // (op 667) reg expression
s_mov_b32 s57, s29 // (op 667) reg expression
s_mov_b32 s58, s0 // (op 667) reg expression
s_mov_b32 s59, 131072 // (op 667) call()

s_mul_hi_i32 s51, s16, s9 // (op 729) 
v_and_b32 v21, -16, v1 // (op 729) 
s_add_i32 s0, s51, s16 // (op 729) 

v_sub_u32 v23, v1, v21 // (op 729) 

s_ashr_i32 s1, s0, 31 // (op 729) 
v_lshrrev_b32 v21, 4, v1 // (op 729) 
s_cmp_eq_i32 s9, 0 // (op 729) 
s_mov_b32 s51, scc // (op 729) 
v_lshlrev_b32 v18, 1, v21 // (op 729) 

s_cmp_eq_i32 s51, 0 // (op 729) 
s_cselect_b32 s6, 0, -1 // (op 729) 

s_and_b32 s7, s10, 31 // (op 729) 
s_lshl_b32 s51, 1, s7 // (op 729) 
s_add_i32 s18, s51, s6 // (op 729) 


s_and_b32 s6, s1, s18 // (op 729) 


s_add_i32 s1, s0, s6 // (op 729) 


s_ashr_i32 s0, s1, s7 // (op 729) 


s_lshl_b32 s1, s10, 24 // (op 729) 
s_ashr_i32 s51, s1, 31 // (op 729) 

s_xor_b32 s1, s0, s51 // (op 729) 

s_sub_i32 s0, s1, s51 // (op 729) 


s_mul_hi_i32 s1, s0, s11 // (op 729) 
s_add_i32 s51, s1, s0 // (op 729) 

s_ashr_i32 s1, s51, 31 // (op 729) 
s_cmp_eq_i32 s11, 0 // (op 729) 
s_mov_b32 s6, scc // (op 729) 
s_cmp_eq_i32 s6, 0 // (op 729) 
s_cselect_b32 s7, 0, -1 // (op 729) 

s_and_b32 s6, s12, 31 // (op 729) 
s_lshl_b32 s18, 1, s6 // (op 729) 
s_add_i32 s19, s18, s7 // (op 729) 


s_and_b32 s7, s1, s19 // (op 729) 


s_add_i32 s1, s51, s7 // (op 729) 


s_ashr_i32 s7, s1, s6 // (op 729) 


s_lshl_b32 s1, s12, 24 // (op 729) 
s_ashr_i32 s6, s1, 31 // (op 729) 

s_xor_b32 s1, s7, s6 // (op 729) 

s_sub_i32 s7, s1, s6 // (op 729) 


s_mul_i32 s1, s0, s14 // (op 729) 

s_sub_u32 s0, s16, s1 // (op 729) 

s_mul_i32 s1, s7, s14 // (op 729) 
s_add_u32 s1, s1, s0 // (op 729) 


s_mul_hi_i32 s0, s16, s8 // (op 729) 
s_add_i32 s51, s0, s16 // (op 729) 

s_ashr_i32 s0, s51, 31 // (op 729) 
s_cmp_eq_i32 s8, 0 // (op 729) 
s_mov_b32 s6, scc // (op 729) 
s_cmp_eq_i32 s6, 0 // (op 729) 
s_cselect_b32 s7, 0, -1 // (op 729) 

s_and_b32 s6, s15, 31 // (op 729) 
s_lshl_b32 s18, 1, s6 // (op 729) 
s_add_i32 s19, s18, s7 // (op 729) 


s_and_b32 s7, s0, s19 // (op 729) 


s_add_i32 s0, s51, s7 // (op 729) 


s_ashr_i32 s7, s0, s6 // (op 729) 


s_lshl_b32 s0, s15, 24 // (op 729) 
s_ashr_i32 s6, s0, 31 // (op 729) 

s_xor_b32 s0, s7, s6 // (op 729) 

s_sub_i32 s7, s0, s6 // (op 729) 


s_add_u32 s18, s4, 16 // (op 729) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 729) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 729) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 729) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 729) 

s_mul_i32 s0, s14, s18 // (op 729) 

s_mul_i32 s6, s50, s0 // (op 729) 

s_mul_i32 s0, s7, s6 // (op 729) 

s_sub_u32 s6, s16, s0 // (op 729) 

s_mov_b32 s0, s6 // (op 729) 
s_mul_hi_i32 s51, s0, s13 // (op 729) 
s_add_i32 s18, s51, s0 // (op 729) 


s_ashr_i32 s0, s18, 31 // (op 729) 
s_cmp_eq_i32 s13, 0 // (op 729) 
s_mov_b32 s19, scc // (op 729) 
s_cmp_eq_i32 s19, 0 // (op 729) 
s_cselect_b32 s51, 0, -1 // (op 729) 

s_and_b32 s19, s49, 31 // (op 729) 
s_lshl_b32 s68, 1, s19 // (op 729) 
s_add_i32 s69, s68, s51 // (op 729) 


s_and_b32 s51, s0, s69 // (op 729) 


s_add_i32 s0, s18, s51 // (op 729) 


s_ashr_i32 s18, s0, s19 // (op 729) 


s_lshl_b32 s0, s49, 24 // (op 729) 
s_ashr_i32 s19, s0, 31 // (op 729) 

s_xor_b32 s0, s18, s19 // (op 729) 

s_sub_i32 s18, s0, s19 // (op 729) 


s_mul_i32 s0, s48, s14 // (op 729) 
s_add_u32 s68, s20, 16 // (op 729) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 729) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 729) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 729) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 729) 

s_sub_i32 s19, s68, s0 // (op 729) 

s_mul_i32 s51, s18, s19 // (op 729) 


s_sub_u32 s18, s6, s51 // (op 729) 


s_add_u32 s6, s18, s0 // (op 729) 


s_cmp_eq_i32 s7, 0 // (op 729) 
s_mov_b32 s0, scc // (op 729) 

s_cmp_eq_i32 s0, 0 // (op 729) 
s_cselect_b32 s7, s6, s1 // (op 729) 



v_lshl_add_u32 v19, s7, 4, v23 // (op 729) 


s_ashr_i64 s[0:1], s[24:25], 8 // (op 729) 
s_lshr_b32 s51, s0, 1 // (op 729) 

v_lshl_add_u32 v21, s51, 4, v18 // (op 729) 


v_mul_lo_u32 v18, v21, s30 // (op 729) 

v_add_u32 v21, v19, v18 // (op 729) 


s_mul_i32 s1, s30, s35 // (op 731) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 731) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 731) least significant
s_mul_hi_u32 s6, s30, s34 // (op 731) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 731) most significant: sum
s_add_u32 s1, s1, s6 // (op 731) most significant: sum


s_mov_b32 s60, s32 // (op 731) reg expression
s_mov_b32 s61, s33 // (op 731) reg expression
s_mov_b32 s62, s0 // (op 731) reg expression
s_mov_b32 s63, 131072 // (op 731) call()

s_mul_hi_i32 s51, s16, s9 // (op 793) 
v_lshrrev_b32 v23, 3, v1 // (op 793) 
s_add_i32 s0, s51, s16 // (op 793) 

v_lshlrev_b32 v18, 1, v23 // (op 793) 

s_ashr_i32 s1, s0, 31 // (op 793) 
v_and_b32 v19, -8, v1 // (op 793) 
s_cmp_eq_i32 s9, 0 // (op 793) 
s_mov_b32 s51, scc // (op 793) 
v_sub_u32 v23, v1, v19 // (op 793) 

s_cmp_eq_i32 s51, 0 // (op 793) 
s_cselect_b32 s6, 0, -1 // (op 793) 

s_and_b32 s7, s10, 31 // (op 793) 
s_lshl_b32 s51, 1, s7 // (op 793) 
s_add_i32 s18, s51, s6 // (op 793) 


s_and_b32 s6, s1, s18 // (op 793) 


s_add_i32 s1, s0, s6 // (op 793) 


s_ashr_i32 s0, s1, s7 // (op 793) 


s_lshl_b32 s1, s10, 24 // (op 793) 
s_ashr_i32 s51, s1, 31 // (op 793) 

s_xor_b32 s1, s0, s51 // (op 793) 

s_sub_i32 s0, s1, s51 // (op 793) 


s_mul_hi_i32 s1, s0, s11 // (op 793) 
s_add_i32 s51, s1, s0 // (op 793) 

s_ashr_i32 s1, s51, 31 // (op 793) 
s_cmp_eq_i32 s11, 0 // (op 793) 
s_mov_b32 s6, scc // (op 793) 
s_cmp_eq_i32 s6, 0 // (op 793) 
s_cselect_b32 s7, 0, -1 // (op 793) 

s_and_b32 s6, s12, 31 // (op 793) 
s_lshl_b32 s18, 1, s6 // (op 793) 
s_add_i32 s19, s18, s7 // (op 793) 


s_and_b32 s7, s1, s19 // (op 793) 


s_add_i32 s1, s51, s7 // (op 793) 


s_ashr_i32 s7, s1, s6 // (op 793) 


s_lshl_b32 s1, s12, 24 // (op 793) 
s_ashr_i32 s6, s1, 31 // (op 793) 

s_xor_b32 s1, s7, s6 // (op 793) 

s_sub_i32 s7, s1, s6 // (op 793) 


s_add_u32 s18, s4, 16 // (op 793) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 793) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 793) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 793) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 793) 

s_mul_i32 s1, s7, s18 // (op 793) 

s_sub_i32 s51, s0, s1 // (op 793) 


s_mul_hi_i32 s0, s16, s8 // (op 793) 
s_add_i32 s1, s0, s16 // (op 793) 

s_ashr_i32 s0, s1, 31 // (op 793) 
s_cmp_eq_i32 s8, 0 // (op 793) 
s_mov_b32 s6, scc // (op 793) 
s_cmp_eq_i32 s6, 0 // (op 793) 
s_cselect_b32 s7, 0, -1 // (op 793) 

s_and_b32 s6, s15, 31 // (op 793) 
s_lshl_b32 s68, 1, s6 // (op 793) 
s_add_i32 s69, s68, s7 // (op 793) 


s_and_b32 s7, s0, s69 // (op 793) 


s_add_i32 s0, s1, s7 // (op 793) 


s_ashr_i32 s1, s0, s6 // (op 793) 


s_lshl_b32 s0, s15, 24 // (op 793) 
s_ashr_i32 s6, s0, 31 // (op 793) 

s_xor_b32 s0, s1, s6 // (op 793) 

s_sub_i32 s1, s0, s6 // (op 793) 


s_mul_i32 s0, s14, s18 // (op 793) 

s_mul_i32 s6, s50, s0 // (op 793) 

s_mul_i32 s0, s1, s6 // (op 793) 

s_sub_u32 s6, s16, s0 // (op 793) 

s_mul_hi_i32 s0, s6, s13 // (op 793) 
s_add_i32 s7, s0, s6 // (op 793) 


s_ashr_i32 s0, s7, 31 // (op 793) 
s_cmp_eq_i32 s13, 0 // (op 793) 
s_mov_b32 s6, scc // (op 793) 
s_cmp_eq_i32 s6, 0 // (op 793) 
s_cselect_b32 s18, 0, -1 // (op 793) 

s_and_b32 s6, s49, 31 // (op 793) 
s_lshl_b32 s19, 1, s6 // (op 793) 
s_add_i32 s68, s19, s18 // (op 793) 


s_and_b32 s18, s0, s68 // (op 793) 


s_add_i32 s0, s7, s18 // (op 793) 


s_ashr_i32 s7, s0, s6 // (op 793) 


s_lshl_b32 s0, s49, 24 // (op 793) 
s_ashr_i32 s6, s0, 31 // (op 793) 

s_xor_b32 s0, s7, s6 // (op 793) 

s_sub_i32 s7, s0, s6 // (op 793) 


s_cmp_eq_i32 s1, 0 // (op 793) 
s_mov_b32 s0, scc // (op 793) 

s_cmp_eq_i32 s0, 0 // (op 793) 
s_cselect_b32 s1, s7, s51 // (op 793) 



v_lshl_add_u32 v19, s1, 4, v18 // (op 793) 


v_mul_lo_u32 v18, v19, s36 // (op 793) 

s_ashr_i64 s[0:1], s[24:25], 8 // (op 793) 
s_lshr_b32 s51, s0, 1 // (op 793) 

v_lshl_add_u32 v19, s51, 4, v23 // (op 793) 


v_add_u32 v23, v19, v18 // (op 793) 


s_mul_i32 s1, s36, s5 // (op 795) most significant: low of low * high
s_mul_i32 s51, s37, s4 // (op 795) most significant: low of high * low
s_mul_i32 s0, s36, s4 // (op 795) least significant
s_mul_hi_u32 s6, s36, s4 // (op 795) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 795) most significant: sum
s_add_u32 s1, s1, s6 // (op 795) most significant: sum


s_mov_b32 s64, s38 // (op 795) reg expression
s_mov_b32 s65, s39 // (op 795) reg expression
s_mov_b32 s66, s0 // (op 795) reg expression
s_mov_b32 s67, 131072 // (op 795) call()

v_lshlrev_b32 v18, 4, v1 // (op 886) 
s_nop 0
v_readfirstlane_b32 s51, v18 // (op 886) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -16, v1 // (op 938) 
v_sub_u32 v19, v1, v18 // (op 938) 

v_lshlrev_b32 v18, 4, v19 // (op 938) 

v_lshrrev_b32 v19, 4, v1 // (op 938) 
v_lshl_add_u32 v28, v19, 8, v18 // (op 938) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 938) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -16, v1 // (op 990) 
v_sub_u32 v19, v1, v18 // (op 990) 

v_lshrrev_b32 v18, 4, v1 // (op 990) 
v_lshl_add_u32 v28, v18, 5, v19 // (op 990) 


v_and_b32 v18, -8, v1 // (op 1043) 
v_sub_u32 v19, v1, v18 // (op 1043) 

v_lshrrev_b32 v18, 3, v1 // (op 1043) 
v_lshl_add_u32 v29, v18, 4, v19 // (op 1043) 


s_ashr_i64 s[6:7], s[24:25], 8 // (op 194) 
s_and_b32 s1, s6, 4294967294 // (op 194) 

s_ashr_i64 s[6:7], s[24:25], 8 // (op 193) 
s_mov_b32 s18, s6 // (op 193) convert

s_cmp_lt_u32 s1, s18 // (op 193) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193 // (op 193) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193:
s_barrier  // (op 1288) 
s_add_u32 s1, s1, 1 // (op 195) 

s_lshl_b64 s[6:7], s[2:3], 6 // (op 550) 
v_mov_b64 v[12:13], s[6:7] // (op 550) convert


v_mov_b64 v[26:27], v[24:25] // (op 550) 
v_mov_b64 v[16:17], v[26:27] // (op 550) reg expression
s_add_u32 s6, 4096, s51 // (op 550) 
s_mov_b32 m0, s6 // (op 550) reg expression
buffer_load_dwordx4 v16, s[52:55], 0 offen offset: 0 lds // (op 550) Load value direct to lds
v_add_co_u32 v16, s[18:19], v16, v12 // (op 550) least significant half
v_addc_co_u32 v17, s[18:19], 0, v17, s[18:19] // (op 550) most significant half

s_add_u32 m0, m0, 1024 // (op 550) 
buffer_load_dwordx4 v16, s[52:55], 0 offen offset: 0 lds // (op 550) Load value direct to lds
v_add_co_u32 v16, s[18:19], v16, v12 // (op 550) least significant half
v_addc_co_u32 v17, s[18:19], 0, v17, s[18:19] // (op 550) most significant half

s_add_u32 m0, m0, 1024 // (op 550) 
buffer_load_dwordx4 v16, s[52:55], 0 offen offset: 0 lds // (op 550) Load value direct to lds
v_add_co_u32 v16, s[18:19], v16, v12 // (op 550) least significant half
v_addc_co_u32 v17, s[18:19], 0, v17, s[18:19] // (op 550) most significant half

s_add_u32 m0, m0, 1024 // (op 550) 
buffer_load_dwordx4 v16, s[52:55], 0 offen offset: 0 lds // (op 550) Load value direct to lds




s_lshl_b64 s[6:7], s[2:3], 8 // (op 588) 
v_mov_b32 v22, s6 // (op 588) 
v_mov_b32 v12, s7 // (op 588) 
v_add_co_u32 v24, s[18:19], v24, v22 // (op 588) least significant half
v_addc_co_u32 v25, s[18:19], v25, v12, s[18:19] // (op 588) most significant half




s_barrier  // (op 1304) 

s_lshl_b64 s[6:7], s[26:27], 2 // (op 556) 
v_mov_b64 v[12:13], s[6:7] // (op 556) convert


v_mov_b64 v[26:27], v[14:15] // (op 556) 
v_mov_b64 v[16:17], v[26:27] // (op 556) 
s_mov_b32 m0, s0 // (op 556) reg expression
buffer_load_dwordx4 v16, s[56:59], 0 offen offset: 0 lds // (op 556) Load value direct to lds
v_add_co_u32 v16, s[6:7], v16, v12 // (op 556) least significant half
v_addc_co_u32 v17, s[6:7], 0, v17, s[6:7] // (op 556) most significant half

s_add_u32 m0, m0, 1024 // (op 556) 
buffer_load_dwordx4 v16, s[56:59], 0 offen offset: 0 lds // (op 556) Load value direct to lds
v_add_co_u32 v16, s[6:7], v16, v12 // (op 556) least significant half
v_addc_co_u32 v17, s[6:7], 0, v17, s[6:7] // (op 556) most significant half

s_add_u32 m0, m0, 1024 // (op 556) 
buffer_load_dwordx4 v16, s[56:59], 0 offen offset: 0 lds // (op 556) Load value direct to lds
v_add_co_u32 v16, s[6:7], v16, v12 // (op 556) least significant half
v_addc_co_u32 v17, s[6:7], 0, v17, s[6:7] // (op 556) most significant half

s_add_u32 m0, m0, 1024 // (op 556) 
buffer_load_dwordx4 v16, s[56:59], 0 offen offset: 0 lds // (op 556) Load value direct to lds



v_add_co_u32 v14, vcc, 256, v14 // (op 657) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 657) most significant half

v_mov_b64 v[12:13], s[30:31] // (op 222) convert

v_mov_b32 v22, v21 // (op 222) 
v_mov_b32 v16, v22 // (op 222) reg expression
buffer_load_ubyte v26, v16, s[60:63], 0 offen offset: 0 // (op 222) Load value
v_add_u32 v16, v16, v12 // (op 222) 
buffer_load_ubyte v27, v16, s[60:63], 0 offen offset: 0 // (op 222) Load value



s_lshl_b32 s6, s30, 3 // (op 721) 
v_add_u32 v21, v21, s6 // (op 721) 


s_barrier  // (op 1321) 
v_mov_b64 v[12:13], s[36:37] // (op 223) convert




v_mov_b32 v22, v23 // (op 223) 
v_mov_b32 v16, v22 // (op 223) 
buffer_load_ubyte v18, v16, s[64:67], 0 offen offset: 0 // (op 223) Load value
v_add_u32 v16, v16, v12 // (op 223) 
buffer_load_ubyte v19, v16, s[64:67], 0 offen offset: 0 // (op 223) Load value



v_add_u32 v23, 8, v23 // (op 785) 
s_waitcnt vmcnt(3)
ds_write_b8 v28, v26 offset:16512 // (op 219) Store local data 
s_barrier  // (op 1294) 
s_waitcnt vmcnt(2)
ds_write_b8 v28, v27 offset:16528 // (op 219) Store local data 




ds_read_b64_tr_b8 v[30:31], v0 offset:4096 // (op 201) Transpose load local data 
s_barrier  // (op 1338) 
ds_read_b64_tr_b8 v[32:33], v0 offset:4224 // (op 201) Transpose load local data 



ds_read_b64_tr_b8 v[34:35], v0 offset:5120 // (op 201) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:5248 // (op 201) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v29, v18 offset:16384 // (op 220) Store local data 
s_barrier  // (op 1311) 
s_waitcnt vmcnt(0)
ds_write_b8 v29, v19 offset:16392 // (op 220) Store local data 




ds_read_b128 v[38:41], v5 // (op 203) Load local data 
ds_read_b128 v[42:45], v5 offset:64 // (op 203) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1328) 


ds_read_u8 v22, v4 offset:16512 // (op 205) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1345) 



ds_read_u8 v12, v6 offset:16384 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 208) 







ds_read_b64_tr_b8 v[30:31], v0 offset:6144 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:6272 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:7168 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:7296 // (op 210) Transpose load local data 



ds_read_b128 v[38:41], v5 offset:128 // (op 212) Load local data 
ds_read_b128 v[42:45], v5 offset:192 // (op 212) Load local data 


ds_read_u8 v22, v4 offset:16576 // (op 214) Load local data 



ds_read_u8 v12, v6 offset:16388 // (op 216) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 217) 




s_ashr_i64 s[6:7], s[24:25], 8 // (op 193) 
s_mov_b32 s18, s6 // (op 193) convert

s_cmp_lt_u32 s1, s18 // (op 193) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193 // (op 193) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193 if true)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193:























s_barrier  // (op 1410) 








v_cmp_eq_f32 vcc, 0.00000, v7 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v11, v12 // (op 50) 
v_mul_f32 v17, v11, v13 // (op 50) 
v_mul_f32 v18, v11, v14 // (op 50) 
v_mul_f32 v19, v11, v15 // (op 50) 

s_mul_hi_i32 s0, s16, s9 // (op 1085) 
v_lshrrev_b32 v0, 6, v1 // (op 1085) 
s_add_i32 s1, s0, s16 // (op 1085) 

v_and_b32 v3, 63, v1 // (op 1085) 
s_ashr_i32 s0, s1, 31 // (op 1085) 
v_lshrrev_b32 v4, 4, v3 // (op 1085) 
s_cmp_eq_i32 s9, 0 // (op 1085) 
s_mov_b32 s2, scc // (op 1085) 
v_lshlrev_b32 v5, 2, v4 // (op 1085) 

s_cmp_eq_i32 s2, 0 // (op 1085) 
s_cselect_b32 s3, 0, -1 // (op 1085) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1085) 


s_and_b32 s2, s10, 31 // (op 1085) 
v_lshrrev_b32 v0, 2, v3 // (op 1085) 

s_lshl_b32 s6, 1, s2 // (op 1085) 
v_and_b32 v3, 3, v0 // (op 1085) 

s_add_i32 s7, s6, s3 // (op 1085) 


v_and_b32 v0, 3, v1 // (op 1085) 
s_and_b32 s3, s0, s7 // (op 1085) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 1085) 


s_add_i32 s0, s1, s3 // (op 1085) 


s_ashr_i32 s1, s0, s2 // (op 1085) 


s_lshl_b32 s0, s10, 24 // (op 1085) 
s_ashr_i32 s2, s0, 31 // (op 1085) 

s_xor_b32 s0, s1, s2 // (op 1085) 

s_sub_i32 s1, s0, s2 // (op 1085) 


s_mul_hi_i32 s0, s1, s11 // (op 1085) 
s_add_i32 s2, s0, s1 // (op 1085) 

s_ashr_i32 s0, s2, 31 // (op 1085) 
s_cmp_eq_i32 s11, 0 // (op 1085) 
s_mov_b32 s3, scc // (op 1085) 
s_cmp_eq_i32 s3, 0 // (op 1085) 
s_cselect_b32 s6, 0, -1 // (op 1085) 

s_and_b32 s3, s12, 31 // (op 1085) 
s_lshl_b32 s7, 1, s3 // (op 1085) 
s_add_i32 s18, s7, s6 // (op 1085) 


s_and_b32 s19, s0, s18 // (op 1085) 


s_add_i32 s0, s2, s19 // (op 1085) 


s_ashr_i32 s2, s0, s3 // (op 1085) 


s_lshl_b32 s0, s12, 24 // (op 1085) 
s_ashr_i32 s3, s0, 31 // (op 1085) 

s_xor_b32 s0, s2, s3 // (op 1085) 

s_sub_i32 s2, s0, s3 // (op 1085) 


s_mul_i32 s0, s1, s14 // (op 1085) 
s_sub_u32 s3, s16, s0 // (op 1085) 

s_mul_i32 s0, s2, s14 // (op 1085) 
s_add_u32 s0, s0, s3 // (op 1085) 

s_mul_hi_i32 s3, s16, s8 // (op 1085) 
s_add_i32 s6, s3, s16 // (op 1085) 

s_ashr_i32 s3, s6, 31 // (op 1085) 
s_cmp_eq_i32 s8, 0 // (op 1085) 
s_mov_b32 s7, scc // (op 1085) 
s_cmp_eq_i32 s7, 0 // (op 1085) 
s_cselect_b32 s18, 0, -1 // (op 1085) 

s_and_b32 s7, s15, 31 // (op 1085) 
s_lshl_b32 s19, 1, s7 // (op 1085) 
s_add_i32 s22, s19, s18 // (op 1085) 


s_and_b32 s18, s3, s22 // (op 1085) 


s_add_i32 s3, s6, s18 // (op 1085) 


s_ashr_i32 s6, s3, s7 // (op 1085) 


s_lshl_b32 s3, s15, 24 // (op 1085) 
s_ashr_i32 s7, s3, 31 // (op 1085) 

s_xor_b32 s3, s6, s7 // (op 1085) 

s_sub_i32 s6, s3, s7 // (op 1085) 


s_add_u32 s18, s4, 16 // (op 1085) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 1085) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 1085) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 1085) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 4 // (op 1085) 

s_mul_i32 s3, s14, s18 // (op 1085) 
s_mul_i32 s7, s50, s3 // (op 1085) 

s_mul_i32 s3, s6, s7 // (op 1085) 

s_sub_u32 s7, s16, s3 // (op 1085) 

s_mov_b32 s3, s7 // (op 1085) 
s_mul_hi_i32 s22, s3, s13 // (op 1085) 
s_add_i32 s23, s22, s3 // (op 1085) 


s_ashr_i32 s3, s23, 31 // (op 1085) 
s_cmp_eq_i32 s13, 0 // (op 1085) 
s_mov_b32 s22, scc // (op 1085) 
s_cmp_eq_i32 s22, 0 // (op 1085) 
s_cselect_b32 s24, 0, -1 // (op 1085) 

s_and_b32 s22, s49, 31 // (op 1085) 
s_lshl_b32 s25, 1, s22 // (op 1085) 
s_add_i32 s26, s25, s24 // (op 1085) 


s_and_b32 s24, s3, s26 // (op 1085) 


s_add_i32 s3, s23, s24 // (op 1085) 


s_ashr_i32 s23, s3, s22 // (op 1085) 


s_lshl_b32 s3, s49, 24 // (op 1085) 
s_ashr_i32 s22, s3, 31 // (op 1085) 

s_xor_b32 s3, s23, s22 // (op 1085) 

s_sub_i32 s23, s3, s22 // (op 1085) 


s_mul_i32 s3, s48, s14 // (op 1085) 
s_add_u32 s24, s20, 16 // (op 1085) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1085) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 1085) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 1085) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 4 // (op 1085) 

s_sub_i32 s22, s24, s3 // (op 1085) 

s_mul_i32 s24, s23, s22 // (op 1085) 

s_sub_u32 s22, s7, s24 // (op 1085) 


s_add_u32 s7, s22, s3 // (op 1085) 


s_cmp_eq_i32 s6, 0 // (op 1085) 
s_mov_b32 s3, scc // (op 1085) 

s_cmp_eq_i32 s3, 0 // (op 1085) 
s_cselect_b32 s6, s7, s0 // (op 1085) 


s_mul_i32 s0, s2, s18 // (op 1085) 


v_lshl_add_u32 v0, s6, 4, v4 // (op 1085) 


s_sub_i32 s2, s1, s0 // (op 1085) 


s_cmp_eq_i32 s3, 0 // (op 1085) 
s_cselect_b32 s22, s23, s2 // (op 1085) 



v_lshl_add_u32 v6, s22, 4, v5 // (op 1085) 


v_mul_lo_u32 v3, v6, s40 // (op 1085) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 1085) 


s_mul_i32 s7, s40, s5 // (op 1087) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 1087) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 1087) least significant
s_mul_hi_u32 s1, s40, s4 // (op 1087) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 1087) most significant: sum
s_add_u32 s7, s7, s1 // (op 1087) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 1087) 

s_mov_b32 s0, s42 // (op 1087) reg expression
s_mov_b32 s1, s43 // (op 1087) reg expression
s_mov_b32 s2, s18 // (op 1087) reg expression
s_mov_b32 s3, 131072 // (op 1087) call()




buffer_store_dwordx4 v[16:19], v4, s[0:3], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155:
s_mul_hi_i32 s6, s16, s9 // (op 856) 
v_lshrrev_b32 v0, 6, v1 // (op 856) 
s_add_i32 s7, s6, s16 // (op 856) 

v_and_b32 v3, 63, v1 // (op 856) 
s_ashr_i32 s6, s7, 31 // (op 856) 
v_lshrrev_b32 v4, 4, v3 // (op 856) 
s_cmp_eq_i32 s9, 0 // (op 856) 
s_mov_b32 s18, scc // (op 856) 
v_lshlrev_b32 v5, 2, v4 // (op 856) 

s_cmp_eq_i32 s18, 0 // (op 856) 
s_cselect_b32 s19, 0, -1 // (op 856) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 856) 


s_and_b32 s18, s10, 31 // (op 856) 
v_lshrrev_b32 v0, 2, v3 // (op 856) 

s_lshl_b32 s22, 1, s18 // (op 856) 
v_and_b32 v3, 3, v0 // (op 856) 

s_add_i32 s23, s22, s19 // (op 856) 


v_and_b32 v0, 3, v1 // (op 856) 
s_and_b32 s19, s6, s23 // (op 856) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 856) 


s_add_i32 s6, s7, s19 // (op 856) 


s_ashr_i32 s7, s6, s18 // (op 856) 


s_lshl_b32 s6, s10, 24 // (op 856) 
s_ashr_i32 s18, s6, 31 // (op 856) 

s_xor_b32 s6, s7, s18 // (op 856) 

s_sub_i32 s7, s6, s18 // (op 856) 


s_mul_hi_i32 s6, s7, s11 // (op 856) 
s_add_i32 s18, s6, s7 // (op 856) 

s_ashr_i32 s6, s18, 31 // (op 856) 
s_cmp_eq_i32 s11, 0 // (op 856) 
s_mov_b32 s19, scc // (op 856) 
s_cmp_eq_i32 s19, 0 // (op 856) 
s_cselect_b32 s22, 0, -1 // (op 856) 

s_and_b32 s19, s12, 31 // (op 856) 
s_lshl_b32 s23, 1, s19 // (op 856) 
s_add_i32 s24, s23, s22 // (op 856) 


s_and_b32 s22, s6, s24 // (op 856) 


s_add_i32 s6, s18, s22 // (op 856) 


s_ashr_i32 s18, s6, s19 // (op 856) 


s_lshl_b32 s6, s12, 24 // (op 856) 
s_ashr_i32 s19, s6, 31 // (op 856) 

s_xor_b32 s6, s18, s19 // (op 856) 

s_sub_i32 s18, s6, s19 // (op 856) 


s_mul_i32 s6, s7, s14 // (op 856) 
s_sub_u32 s19, s16, s6 // (op 856) 

s_mul_i32 s6, s18, s14 // (op 856) 
s_add_u32 s6, s6, s19 // (op 856) 

s_mul_hi_i32 s19, s16, s8 // (op 856) 
s_add_i32 s22, s19, s16 // (op 856) 

s_ashr_i32 s19, s22, 31 // (op 856) 
s_cmp_eq_i32 s8, 0 // (op 856) 
s_mov_b32 s23, scc // (op 856) 
s_cmp_eq_i32 s23, 0 // (op 856) 
s_cselect_b32 s24, 0, -1 // (op 856) 

s_and_b32 s23, s15, 31 // (op 856) 
s_lshl_b32 s25, 1, s23 // (op 856) 
s_add_i32 s26, s25, s24 // (op 856) 


s_and_b32 s24, s19, s26 // (op 856) 


s_add_i32 s19, s22, s24 // (op 856) 


s_ashr_i32 s22, s19, s23 // (op 856) 


s_lshl_b32 s19, s15, 24 // (op 856) 
s_ashr_i32 s23, s19, 31 // (op 856) 

s_xor_b32 s19, s22, s23 // (op 856) 

s_sub_i32 s22, s19, s23 // (op 856) 


s_add_u32 s38, s4, 16 // (op 856) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 856) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 856) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 856) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 856) 

s_mul_i32 s19, s14, s38 // (op 856) 
s_mul_i32 s23, s50, s19 // (op 856) 

s_mul_i32 s19, s22, s23 // (op 856) 

s_sub_u32 s23, s16, s19 // (op 856) 

s_mov_b32 s19, s23 // (op 856) 
s_mul_hi_i32 s24, s19, s13 // (op 856) 
s_add_i32 s25, s24, s19 // (op 856) 


s_ashr_i32 s19, s25, 31 // (op 856) 
s_cmp_eq_i32 s13, 0 // (op 856) 
s_mov_b32 s24, scc // (op 856) 
s_cmp_eq_i32 s24, 0 // (op 856) 
s_cselect_b32 s26, 0, -1 // (op 856) 

s_and_b32 s24, s49, 31 // (op 856) 
s_lshl_b32 s27, 1, s24 // (op 856) 
s_add_i32 s28, s27, s26 // (op 856) 


s_and_b32 s26, s19, s28 // (op 856) 


s_add_i32 s19, s25, s26 // (op 856) 


s_ashr_i32 s25, s19, s24 // (op 856) 


s_lshl_b32 s19, s49, 24 // (op 856) 
s_ashr_i32 s24, s19, 31 // (op 856) 

s_xor_b32 s19, s25, s24 // (op 856) 

s_sub_i32 s25, s19, s24 // (op 856) 


s_mul_i32 s19, s48, s14 // (op 856) 
s_add_u32 s26, s20, 16 // (op 856) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 856) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 856) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 856) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 856) 

s_sub_i32 s24, s26, s19 // (op 856) 

s_mul_i32 s26, s25, s24 // (op 856) 

s_sub_u32 s24, s23, s26 // (op 856) 


s_add_u32 s23, s24, s19 // (op 856) 


s_cmp_eq_i32 s22, 0 // (op 856) 
s_mov_b32 s19, scc // (op 856) 

s_cmp_eq_i32 s19, 0 // (op 856) 
s_cselect_b32 s22, s23, s6 // (op 856) 


s_mul_i32 s6, s18, s38 // (op 856) 


v_lshl_add_u32 v0, s22, 4, v4 // (op 856) 


s_sub_i32 s18, s7, s6 // (op 856) 


s_cmp_eq_i32 s19, 0 // (op 856) 
s_cselect_b32 s6, s25, s18 // (op 856) 



v_lshl_add_u32 v6, s6, 4, v5 // (op 856) 


v_mul_lo_u32 v3, v6, s44 // (op 856) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 856) 


s_mul_i32 s7, s44, s5 // (op 858) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 858) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 858) least significant
s_mul_hi_u32 s19, s44, s4 // (op 858) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 858) most significant: sum
s_add_u32 s7, s7, s19 // (op 858) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 858) 

s_mov_b32 s36, s46 // (op 858) reg expression
s_mov_b32 s37, s47 // (op 858) reg expression
s_mov_b32 s38, s18 // (op 858) reg expression
s_mov_b32 s39, 131072 // (op 858) call()





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


s_mul_hi_i32 s6, s16, s9 // (op 1102) 
v_lshrrev_b32 v0, 6, v1 // (op 1102) 
s_add_i32 s7, s6, s16 // (op 1102) 

v_and_b32 v3, 63, v1 // (op 1102) 
s_ashr_i32 s6, s7, 31 // (op 1102) 
v_lshrrev_b32 v4, 4, v3 // (op 1102) 
s_cmp_eq_i32 s9, 0 // (op 1102) 
s_mov_b32 s18, scc // (op 1102) 
v_lshlrev_b32 v5, 2, v4 // (op 1102) 

s_cmp_eq_i32 s18, 0 // (op 1102) 
s_cselect_b32 s19, 0, -1 // (op 1102) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1102) 


s_and_b32 s18, s10, 31 // (op 1102) 
v_lshrrev_b32 v0, 2, v3 // (op 1102) 

s_lshl_b32 s22, 1, s18 // (op 1102) 
v_and_b32 v3, 3, v0 // (op 1102) 

s_add_i32 s23, s22, s19 // (op 1102) 


v_and_b32 v0, 3, v1 // (op 1102) 
s_and_b32 s19, s6, s23 // (op 1102) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 1102) 


s_add_i32 s6, s7, s19 // (op 1102) 


s_ashr_i32 s7, s6, s18 // (op 1102) 


s_lshl_b32 s6, s10, 24 // (op 1102) 
s_ashr_i32 s18, s6, 31 // (op 1102) 

s_xor_b32 s6, s7, s18 // (op 1102) 

s_sub_i32 s7, s6, s18 // (op 1102) 


s_mul_hi_i32 s6, s7, s11 // (op 1102) 
s_add_i32 s18, s6, s7 // (op 1102) 

s_ashr_i32 s6, s18, 31 // (op 1102) 
s_cmp_eq_i32 s11, 0 // (op 1102) 
s_mov_b32 s19, scc // (op 1102) 
s_cmp_eq_i32 s19, 0 // (op 1102) 
s_cselect_b32 s22, 0, -1 // (op 1102) 

s_and_b32 s19, s12, 31 // (op 1102) 
s_lshl_b32 s23, 1, s19 // (op 1102) 
s_add_i32 s24, s23, s22 // (op 1102) 


s_and_b32 s22, s6, s24 // (op 1102) 


s_add_i32 s6, s18, s22 // (op 1102) 


s_ashr_i32 s18, s6, s19 // (op 1102) 


s_lshl_b32 s6, s12, 24 // (op 1102) 
s_ashr_i32 s19, s6, 31 // (op 1102) 

s_xor_b32 s6, s18, s19 // (op 1102) 

s_sub_i32 s18, s6, s19 // (op 1102) 


s_mul_i32 s6, s7, s14 // (op 1102) 
s_sub_u32 s19, s16, s6 // (op 1102) 

s_mul_i32 s6, s18, s14 // (op 1102) 
s_add_u32 s6, s6, s19 // (op 1102) 

s_mul_hi_i32 s19, s16, s8 // (op 1102) 
s_add_i32 s22, s19, s16 // (op 1102) 

s_ashr_i32 s19, s22, 31 // (op 1102) 
s_cmp_eq_i32 s8, 0 // (op 1102) 
s_mov_b32 s23, scc // (op 1102) 
s_cmp_eq_i32 s23, 0 // (op 1102) 
s_cselect_b32 s24, 0, -1 // (op 1102) 

s_and_b32 s23, s15, 31 // (op 1102) 
s_lshl_b32 s25, 1, s23 // (op 1102) 
s_add_i32 s26, s25, s24 // (op 1102) 


s_and_b32 s24, s19, s26 // (op 1102) 


s_add_i32 s19, s22, s24 // (op 1102) 


s_ashr_i32 s22, s19, s23 // (op 1102) 


s_lshl_b32 s19, s15, 24 // (op 1102) 
s_ashr_i32 s23, s19, 31 // (op 1102) 

s_xor_b32 s19, s22, s23 // (op 1102) 

s_sub_i32 s22, s19, s23 // (op 1102) 


s_add_u32 s38, s4, 16 // (op 1102) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 1102) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 1102) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 1102) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 1102) 

s_mul_i32 s19, s14, s38 // (op 1102) 
s_mul_i32 s23, s50, s19 // (op 1102) 

s_mul_i32 s19, s22, s23 // (op 1102) 

s_sub_u32 s23, s16, s19 // (op 1102) 

s_mov_b32 s19, s23 // (op 1102) 
s_mul_hi_i32 s24, s19, s13 // (op 1102) 
s_add_i32 s25, s24, s19 // (op 1102) 


s_ashr_i32 s19, s25, 31 // (op 1102) 
s_cmp_eq_i32 s13, 0 // (op 1102) 
s_mov_b32 s24, scc // (op 1102) 
s_cmp_eq_i32 s24, 0 // (op 1102) 
s_cselect_b32 s26, 0, -1 // (op 1102) 

s_and_b32 s24, s49, 31 // (op 1102) 
s_lshl_b32 s27, 1, s24 // (op 1102) 
s_add_i32 s28, s27, s26 // (op 1102) 


s_and_b32 s26, s19, s28 // (op 1102) 


s_add_i32 s19, s25, s26 // (op 1102) 


s_ashr_i32 s25, s19, s24 // (op 1102) 


s_lshl_b32 s19, s49, 24 // (op 1102) 
s_ashr_i32 s24, s19, 31 // (op 1102) 

s_xor_b32 s19, s25, s24 // (op 1102) 

s_sub_i32 s25, s19, s24 // (op 1102) 


s_mul_i32 s19, s48, s14 // (op 1102) 
s_add_u32 s26, s20, 16 // (op 1102) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 1102) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 1102) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 1102) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 1102) 

s_sub_i32 s24, s26, s19 // (op 1102) 

s_mul_i32 s26, s25, s24 // (op 1102) 

s_sub_u32 s24, s23, s26 // (op 1102) 


s_add_u32 s23, s24, s19 // (op 1102) 


s_cmp_eq_i32 s22, 0 // (op 1102) 
s_mov_b32 s19, scc // (op 1102) 

s_cmp_eq_i32 s19, 0 // (op 1102) 
s_cselect_b32 s22, s23, s6 // (op 1102) 


s_mul_i32 s6, s18, s38 // (op 1102) 


v_lshl_add_u32 v0, s22, 4, v4 // (op 1102) 


s_sub_i32 s18, s7, s6 // (op 1102) 


s_cmp_eq_i32 s19, 0 // (op 1102) 
s_cselect_b32 s6, s25, s18 // (op 1102) 



v_lshl_add_u32 v6, s6, 4, v5 // (op 1102) 


v_mul_lo_u32 v3, v6, s40 // (op 1102) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 1102) 


s_mul_i32 s7, s40, s5 // (op 1104) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 1104) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 1104) least significant
s_mul_hi_u32 s19, s40, s4 // (op 1104) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 1104) most significant: sum
s_add_u32 s7, s7, s19 // (op 1104) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 1104) 

s_mov_b32 s0, s42 // (op 1104) reg expression
s_mov_b32 s1, s43 // (op 1104) reg expression
s_mov_b32 s2, s18 // (op 1104) reg expression
s_mov_b32 s3, 131072 // (op 1104) call()




buffer_store_dwordx4 v[12:15], v4, s[0:3], 0 offen // (op 166) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155:
















s_endpgm  // End of RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.LRR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end:
.size RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_, .LRR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end-RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 84
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16896 // lds bytes
  .amdhsa_accum_offset 80
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: B8
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
  - .name: RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
    .symbol: RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 78
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

