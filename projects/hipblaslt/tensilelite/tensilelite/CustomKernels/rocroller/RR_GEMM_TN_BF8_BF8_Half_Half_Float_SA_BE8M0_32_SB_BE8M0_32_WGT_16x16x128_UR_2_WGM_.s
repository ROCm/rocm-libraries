// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_,@function
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 831) 
v_and_b32 v3, 15, v1 // (op 831) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 831) 


v_and_b32 v0, 48, v1 // (op 831) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 831) 


v_and_b32 v0, 15, v1 // (op 883) 
v_and_b32 v4, 48, v1 // (op 883) 
v_lshl_add_u32 v5, v0, 7, v4 // (op 883) 


v_lshrrev_b32 v0, 6, v1 // (op 936) 
v_and_b32 v4, 15, v1 // (op 936) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 936) 


v_and_b32 v0, 63, v1 // (op 936) 
v_lshrrev_b32 v4, 4, v0 // (op 936) 

v_lshl_add_u32 v0, v6, 2, v4 // (op 936) 


v_and_b32 v4, 15, v1 // (op 989) 
v_and_b32 v6, 63, v1 // (op 989) 
v_lshrrev_b32 v7, 4, v6 // (op 989) 

v_lshl_add_u32 v6, v4, 2, v7 // (op 989) 


v_lshrrev_b32 v4, 6, v1 // (op 1062) 
v_and_b32 v7, 15, v1 // (op 1062) 
v_lshl_add_u32 v8, v4, 4, v7 // (op 1062) 


v_and_b32 v4, 48, v1 // (op 1062) 
v_lshl_add_u32 v7, v8, 7, v4 // (op 1062) 


v_and_b32 v4, 15, v1 // (op 1101) 
v_and_b32 v8, 48, v1 // (op 1101) 
v_lshl_add_u32 v9, v4, 7, v8 // (op 1101) 


v_lshrrev_b32 v4, 6, v1 // (op 1125) 
v_and_b32 v8, 15, v1 // (op 1125) 
v_lshl_add_u32 v10, v4, 4, v8 // (op 1125) 


v_and_b32 v4, 63, v1 // (op 1125) 
v_lshrrev_b32 v8, 4, v4 // (op 1125) 

v_lshl_add_u32 v4, v10, 2, v8 // (op 1125) 


v_and_b32 v8, 15, v1 // (op 1149) 
v_and_b32 v10, 63, v1 // (op 1149) 
v_lshrrev_b32 v11, 4, v10 // (op 1149) 

v_lshl_add_u32 v10, v8, 2, v11 // (op 1149) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v11, s9 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
s_mul_hi_i32 s0, s16, s12 // (op 542) 
v_lshrrev_b32 v12, 3, v1 // (op 542) 
s_add_i32 s1, s0, s16 // (op 542) 

v_and_b32 v13, -8, v1 // (op 542) 
s_ashr_i32 s0, s1, 31 // (op 542) 
v_sub_u32 v14, v1, v13 // (op 542) 

s_cmp_eq_i32 s12, 0 // (op 542) 
s_mov_b32 s8, scc // (op 542) 
v_lshlrev_b32 v13, 4, v14 // (op 542) 

s_cmp_eq_i32 s8, 0 // (op 542) 
s_cselect_b32 s9, 0, -1 // (op 542) 

v_mov_b32 v15, 0 // (op 542) convert
v_mov_b32 v14, v13 // (op 542) convert

s_and_b32 s8, s13, 31 // (op 542) 
s_lshl_b32 s18, 1, s8 // (op 542) 
s_add_i32 s19, s18, s9 // (op 542) 


s_and_b32 s9, s0, s19 // (op 542) 


s_add_i32 s0, s1, s9 // (op 542) 


s_ashr_i32 s1, s0, s8 // (op 542) 


s_lshl_b32 s0, s13, 24 // (op 542) 
s_ashr_i32 s8, s0, 31 // (op 542) 

s_xor_b32 s0, s1, s8 // (op 542) 

s_sub_i32 s1, s0, s8 // (op 542) 


s_mul_hi_i32 s0, s1, s14 // (op 542) 
s_add_i32 s8, s0, s1 // (op 542) 

s_ashr_i32 s0, s8, 31 // (op 542) 
s_cmp_eq_i32 s14, 0 // (op 542) 
s_mov_b32 s9, scc // (op 542) 
s_cmp_eq_i32 s9, 0 // (op 542) 
s_cselect_b32 s18, 0, -1 // (op 542) 

s_and_b32 s9, s15, 31 // (op 542) 
s_lshl_b32 s19, 1, s9 // (op 542) 
s_add_i32 s49, s19, s18 // (op 542) 


s_and_b32 s18, s0, s49 // (op 542) 


s_add_i32 s0, s8, s18 // (op 542) 


s_ashr_i32 s8, s0, s9 // (op 542) 


s_lshl_b32 s0, s15, 24 // (op 542) 
s_ashr_i32 s9, s0, 31 // (op 542) 

s_xor_b32 s0, s8, s9 // (op 542) 

s_sub_i32 s8, s0, s9 // (op 542) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s45 // (op 542) 

s_sub_u32 s1, s16, s0 // (op 542) 

s_mul_i32 s0, s8, s45 // (op 542) 
s_add_u32 s0, s0, s1 // (op 542) 


s_mul_hi_i32 s1, s16, s11 // (op 542) 
s_add_i32 s8, s1, s16 // (op 542) 

s_ashr_i32 s1, s8, 31 // (op 542) 
s_cmp_eq_i32 s11, 0 // (op 542) 
s_mov_b32 s9, scc // (op 542) 
s_cmp_eq_i32 s9, 0 // (op 542) 
s_cselect_b32 s18, 0, -1 // (op 542) 

s_and_b32 s9, s46, 31 // (op 542) 
s_lshl_b32 s19, 1, s9 // (op 542) 
s_add_i32 s49, s19, s18 // (op 542) 


s_and_b32 s18, s1, s49 // (op 542) 


s_add_i32 s1, s8, s18 // (op 542) 


s_ashr_i32 s8, s1, s9 // (op 542) 


s_lshl_b32 s1, s46, 24 // (op 542) 
s_ashr_i32 s9, s1, 31 // (op 542) 

s_xor_b32 s1, s8, s9 // (op 542) 

s_sub_i32 s8, s1, s9 // (op 542) 


s_add_u32 s18, s4, 16 // (op 542) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 542) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 542) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 542) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 542) 

s_mul_i32 s1, s45, s18 // (op 542) 

s_mul_i32 s9, s10, s1 // (op 542) 

s_mul_i32 s1, s8, s9 // (op 542) 

s_sub_u32 s9, s16, s1 // (op 542) 

s_mov_b32 s1, s9 // (op 542) 
s_mul_hi_i32 s18, s1, s44 // (op 542) 
s_add_i32 s19, s18, s1 // (op 542) 


s_ashr_i32 s1, s19, 31 // (op 542) 
s_cmp_eq_i32 s44, 0 // (op 542) 
s_mov_b32 s18, scc // (op 542) 
s_cmp_eq_i32 s18, 0 // (op 542) 
s_cselect_b32 s49, 0, -1 // (op 542) 

s_and_b32 s18, s48, 31 // (op 542) 
s_lshl_b32 s50, 1, s18 // (op 542) 
s_add_i32 s51, s50, s49 // (op 542) 


s_and_b32 s49, s1, s51 // (op 542) 


s_add_i32 s1, s19, s49 // (op 542) 


s_ashr_i32 s19, s1, s18 // (op 542) 


s_lshl_b32 s1, s48, 24 // (op 542) 
s_ashr_i32 s18, s1, 31 // (op 542) 

s_xor_b32 s1, s19, s18 // (op 542) 

s_sub_i32 s19, s1, s18 // (op 542) 


s_mul_i32 s1, s47, s45 // (op 542) 
s_add_u32 s50, s6, 16 // (op 542) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 542) End of Int64 add, unlocking SCC
s_sub_u32 s52, s50, 1 // (op 542) least significant half; sets scc
s_subb_u32 s53, s51, 0 // (op 542) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 4 // (op 542) 

s_sub_i32 s18, s50, s1 // (op 542) 

s_mul_i32 s49, s19, s18 // (op 542) 


s_sub_u32 s18, s9, s49 // (op 542) 


s_add_u32 s9, s18, s1 // (op 542) 


s_cmp_eq_i32 s8, 0 // (op 542) 
s_mov_b32 s1, scc // (op 542) 

s_cmp_eq_i32 s1, 0 // (op 542) 
s_cselect_b32 s8, s9, s0 // (op 542) 



v_lshl_add_u32 v13, s8, 4, v12 // (op 542) 


v_mul_lo_u32 v17, v13, s3 // (op 542) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 542) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 542) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 542) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 542) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 542) most significant half



s_mul_i32 s1, s2, s7 // (op 544) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 544) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 544) least significant
s_mul_hi_u32 s9, s2, s6 // (op 544) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 544) most significant: sum
s_add_u32 s1, s1, s9 // (op 544) most significant: sum


s_mov_b32 s52, s20 // (op 544) reg expression
s_mov_b32 s53, s21 // (op 544) reg expression
s_mov_b32 s54, s0 // (op 544) reg expression
s_mov_b32 s55, 131072 // (op 544) call()

s_mul_hi_i32 s0, s16, s12 // (op 561) 
v_lshrrev_b32 v12, 3, v1 // (op 561) 
s_add_i32 s1, s0, s16 // (op 561) 

v_and_b32 v13, -8, v1 // (op 561) 
s_ashr_i32 s0, s1, 31 // (op 561) 
v_sub_u32 v14, v1, v13 // (op 561) 

s_cmp_eq_i32 s12, 0 // (op 561) 
s_mov_b32 s8, scc // (op 561) 
v_lshlrev_b32 v13, 4, v14 // (op 561) 

s_cmp_eq_i32 s8, 0 // (op 561) 
s_cselect_b32 s9, 0, -1 // (op 561) 

v_mov_b32 v15, 0 // (op 561) convert
v_mov_b32 v14, v13 // (op 561) convert

s_and_b32 s8, s13, 31 // (op 561) 
s_lshl_b32 s18, 1, s8 // (op 561) 
s_add_i32 s19, s18, s9 // (op 561) 


s_and_b32 s9, s0, s19 // (op 561) 


s_add_i32 s0, s1, s9 // (op 561) 


s_ashr_i32 s1, s0, s8 // (op 561) 


s_lshl_b32 s0, s13, 24 // (op 561) 
s_ashr_i32 s8, s0, 31 // (op 561) 

s_xor_b32 s0, s1, s8 // (op 561) 

s_sub_i32 s1, s0, s8 // (op 561) 


s_mul_hi_i32 s0, s1, s14 // (op 561) 
s_add_i32 s8, s0, s1 // (op 561) 

s_ashr_i32 s0, s8, 31 // (op 561) 
s_cmp_eq_i32 s14, 0 // (op 561) 
s_mov_b32 s9, scc // (op 561) 
s_cmp_eq_i32 s9, 0 // (op 561) 
s_cselect_b32 s18, 0, -1 // (op 561) 

s_and_b32 s9, s15, 31 // (op 561) 
s_lshl_b32 s19, 1, s9 // (op 561) 
s_add_i32 s49, s19, s18 // (op 561) 


s_and_b32 s18, s0, s49 // (op 561) 


s_add_i32 s0, s8, s18 // (op 561) 


s_ashr_i32 s8, s0, s9 // (op 561) 


s_lshl_b32 s0, s15, 24 // (op 561) 
s_ashr_i32 s9, s0, 31 // (op 561) 

s_xor_b32 s0, s8, s9 // (op 561) 

s_sub_i32 s8, s0, s9 // (op 561) 


s_mul_i32 s0, s1, s45 // (op 561) 

s_sub_u32 s1, s16, s0 // (op 561) 

s_mul_i32 s0, s8, s45 // (op 561) 
s_add_u32 s0, s0, s1 // (op 561) 


s_mul_hi_i32 s1, s16, s11 // (op 561) 
s_add_i32 s8, s1, s16 // (op 561) 

s_ashr_i32 s1, s8, 31 // (op 561) 
s_cmp_eq_i32 s11, 0 // (op 561) 
s_mov_b32 s9, scc // (op 561) 
s_cmp_eq_i32 s9, 0 // (op 561) 
s_cselect_b32 s18, 0, -1 // (op 561) 

s_and_b32 s9, s46, 31 // (op 561) 
s_lshl_b32 s19, 1, s9 // (op 561) 
s_add_i32 s49, s19, s18 // (op 561) 


s_and_b32 s18, s1, s49 // (op 561) 


s_add_i32 s1, s8, s18 // (op 561) 


s_ashr_i32 s8, s1, s9 // (op 561) 


s_lshl_b32 s1, s46, 24 // (op 561) 
s_ashr_i32 s9, s1, 31 // (op 561) 

s_xor_b32 s1, s8, s9 // (op 561) 

s_sub_i32 s8, s1, s9 // (op 561) 


s_add_u32 s18, s4, 16 // (op 561) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 561) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 561) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 561) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 561) 

s_mul_i32 s1, s45, s18 // (op 561) 

s_mul_i32 s9, s10, s1 // (op 561) 

s_mul_i32 s1, s8, s9 // (op 561) 

s_sub_u32 s9, s16, s1 // (op 561) 

s_mov_b32 s1, s9 // (op 561) 
s_mul_hi_i32 s18, s1, s44 // (op 561) 
s_add_i32 s19, s18, s1 // (op 561) 


s_ashr_i32 s1, s19, 31 // (op 561) 
s_cmp_eq_i32 s44, 0 // (op 561) 
s_mov_b32 s18, scc // (op 561) 
s_cmp_eq_i32 s18, 0 // (op 561) 
s_cselect_b32 s49, 0, -1 // (op 561) 

s_and_b32 s18, s48, 31 // (op 561) 
s_lshl_b32 s50, 1, s18 // (op 561) 
s_add_i32 s51, s50, s49 // (op 561) 


s_and_b32 s49, s1, s51 // (op 561) 


s_add_i32 s1, s19, s49 // (op 561) 


s_ashr_i32 s19, s1, s18 // (op 561) 


s_lshl_b32 s1, s48, 24 // (op 561) 
s_ashr_i32 s18, s1, 31 // (op 561) 

s_xor_b32 s1, s19, s18 // (op 561) 

s_sub_i32 s19, s1, s18 // (op 561) 


s_mul_i32 s1, s47, s45 // (op 561) 
s_add_u32 s50, s6, 16 // (op 561) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 561) End of Int64 add, unlocking SCC
s_sub_u32 s56, s50, 1 // (op 561) least significant half; sets scc
s_subb_u32 s57, s51, 0 // (op 561) most significant half; uses scc

s_ashr_i64 s[50:51], s[56:57], 4 // (op 561) 

s_sub_i32 s18, s50, s1 // (op 561) 

s_mul_i32 s49, s19, s18 // (op 561) 


s_sub_u32 s18, s9, s49 // (op 561) 


s_add_u32 s9, s18, s1 // (op 561) 


s_cmp_eq_i32 s8, 0 // (op 561) 
s_mov_b32 s1, scc // (op 561) 

s_cmp_eq_i32 s1, 0 // (op 561) 
s_cselect_b32 s8, s9, s0 // (op 561) 



v_lshl_add_u32 v13, s8, 4, v12 // (op 561) 


v_mul_lo_u32 v19, v13, s3 // (op 561) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 561) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 561) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 561) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 561) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 561) most significant half



s_mul_i32 s1, s2, s7 // (op 563) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 563) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 563) least significant
s_mul_hi_u32 s9, s2, s6 // (op 563) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 563) most significant: sum
s_add_u32 s1, s1, s9 // (op 563) most significant: sum


s_mov_b32 s52, s20 // (op 563) reg expression
s_mov_b32 s53, s21 // (op 563) reg expression
s_mov_b32 s54, s0 // (op 563) reg expression
s_mov_b32 s55, 131072 // (op 563) call()

s_mul_hi_i32 s0, s16, s12 // (op 606) 
v_lshrrev_b32 v12, 3, v1 // (op 606) 
s_add_i32 s1, s0, s16 // (op 606) 

v_and_b32 v13, -8, v1 // (op 606) 
s_ashr_i32 s0, s1, 31 // (op 606) 
v_sub_u32 v14, v1, v13 // (op 606) 

s_cmp_eq_i32 s12, 0 // (op 606) 
s_mov_b32 s8, scc // (op 606) 
v_lshlrev_b32 v13, 4, v14 // (op 606) 

s_cmp_eq_i32 s8, 0 // (op 606) 
s_cselect_b32 s9, 0, -1 // (op 606) 

v_mov_b32 v15, 0 // (op 606) convert
v_mov_b32 v14, v13 // (op 606) convert

s_and_b32 s8, s13, 31 // (op 606) 
s_lshl_b32 s18, 1, s8 // (op 606) 
s_add_i32 s19, s18, s9 // (op 606) 


s_and_b32 s9, s0, s19 // (op 606) 


s_add_i32 s0, s1, s9 // (op 606) 


s_ashr_i32 s1, s0, s8 // (op 606) 


s_lshl_b32 s0, s13, 24 // (op 606) 
s_ashr_i32 s8, s0, 31 // (op 606) 

s_xor_b32 s0, s1, s8 // (op 606) 

s_sub_i32 s1, s0, s8 // (op 606) 


s_mul_hi_i32 s0, s1, s14 // (op 606) 
s_add_i32 s8, s0, s1 // (op 606) 

s_ashr_i32 s0, s8, 31 // (op 606) 
s_cmp_eq_i32 s14, 0 // (op 606) 
s_mov_b32 s9, scc // (op 606) 
s_cmp_eq_i32 s9, 0 // (op 606) 
s_cselect_b32 s18, 0, -1 // (op 606) 

s_and_b32 s9, s15, 31 // (op 606) 
s_lshl_b32 s19, 1, s9 // (op 606) 
s_add_i32 s49, s19, s18 // (op 606) 


s_and_b32 s18, s0, s49 // (op 606) 


s_add_i32 s0, s8, s18 // (op 606) 


s_ashr_i32 s8, s0, s9 // (op 606) 


s_lshl_b32 s0, s15, 24 // (op 606) 
s_ashr_i32 s9, s0, 31 // (op 606) 

s_xor_b32 s0, s8, s9 // (op 606) 

s_sub_i32 s8, s0, s9 // (op 606) 


s_add_u32 s18, s4, 16 // (op 606) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 606) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 606) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 606) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 606) 

s_mul_i32 s0, s8, s18 // (op 606) 

s_sub_i32 s8, s1, s0 // (op 606) 


s_mul_hi_i32 s9, s16, s11 // (op 606) 
s_add_i32 s0, s9, s16 // (op 606) 

s_ashr_i32 s1, s0, 31 // (op 606) 
s_cmp_eq_i32 s11, 0 // (op 606) 
s_mov_b32 s9, scc // (op 606) 
s_cmp_eq_i32 s9, 0 // (op 606) 
s_cselect_b32 s49, 0, -1 // (op 606) 

s_and_b32 s9, s46, 31 // (op 606) 
s_lshl_b32 s50, 1, s9 // (op 606) 
s_add_i32 s51, s50, s49 // (op 606) 


s_and_b32 s49, s1, s51 // (op 606) 


s_add_i32 s1, s0, s49 // (op 606) 


s_ashr_i32 s0, s1, s9 // (op 606) 


s_lshl_b32 s1, s46, 24 // (op 606) 
s_ashr_i32 s9, s1, 31 // (op 606) 

s_xor_b32 s1, s0, s9 // (op 606) 

s_sub_i32 s0, s1, s9 // (op 606) 


s_mul_i32 s1, s45, s18 // (op 606) 

s_mul_i32 s9, s10, s1 // (op 606) 

s_mul_i32 s1, s0, s9 // (op 606) 

s_sub_u32 s9, s16, s1 // (op 606) 

s_mul_hi_i32 s1, s9, s44 // (op 606) 
s_add_i32 s18, s1, s9 // (op 606) 


s_ashr_i32 s1, s18, 31 // (op 606) 
s_cmp_eq_i32 s44, 0 // (op 606) 
s_mov_b32 s9, scc // (op 606) 
s_cmp_eq_i32 s9, 0 // (op 606) 
s_cselect_b32 s19, 0, -1 // (op 606) 

s_and_b32 s9, s48, 31 // (op 606) 
s_lshl_b32 s49, 1, s9 // (op 606) 
s_add_i32 s50, s49, s19 // (op 606) 


s_and_b32 s19, s1, s50 // (op 606) 


s_add_i32 s1, s18, s19 // (op 606) 


s_ashr_i32 s18, s1, s9 // (op 606) 


s_lshl_b32 s1, s48, 24 // (op 606) 
s_ashr_i32 s9, s1, 31 // (op 606) 

s_xor_b32 s1, s18, s9 // (op 606) 

s_sub_i32 s18, s1, s9 // (op 606) 


s_cmp_eq_i32 s0, 0 // (op 606) 
s_mov_b32 s1, scc // (op 606) 

s_cmp_eq_i32 s1, 0 // (op 606) 
s_cselect_b32 s0, s18, s8 // (op 606) 



v_lshl_add_u32 v13, s0, 4, v12 // (op 606) 


v_mul_lo_u32 v21, v13, s23 // (op 606) most significant: low of low * high
v_mul_hi_u32 v12, v13, s22 // (op 606) most significant: high of low * low
v_mul_lo_u32 v20, v13, s22 // (op 606) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 606) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 606) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 606) most significant half



s_mul_i32 s1, s22, s5 // (op 608) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 608) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 608) least significant
s_mul_hi_u32 s9, s22, s4 // (op 608) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 608) most significant: sum
s_add_u32 s1, s1, s9 // (op 608) most significant: sum


s_mov_b32 s56, s24 // (op 608) reg expression
s_mov_b32 s57, s25 // (op 608) reg expression
s_mov_b32 s58, s0 // (op 608) reg expression
s_mov_b32 s59, 131072 // (op 608) call()

s_mul_hi_i32 s0, s16, s12 // (op 625) 
v_lshrrev_b32 v14, 3, v1 // (op 625) 
s_add_i32 s1, s0, s16 // (op 625) 

v_and_b32 v15, -8, v1 // (op 625) 
s_ashr_i32 s0, s1, 31 // (op 625) 
v_sub_u32 v20, v1, v15 // (op 625) 

s_cmp_eq_i32 s12, 0 // (op 625) 
s_mov_b32 s8, scc // (op 625) 
v_lshlrev_b32 v15, 4, v20 // (op 625) 

s_cmp_eq_i32 s8, 0 // (op 625) 
s_cselect_b32 s9, 0, -1 // (op 625) 

v_mov_b32 v21, 0 // (op 625) convert
v_mov_b32 v20, v15 // (op 625) convert

s_and_b32 s8, s13, 31 // (op 625) 
s_lshl_b32 s18, 1, s8 // (op 625) 
s_add_i32 s19, s18, s9 // (op 625) 


s_and_b32 s9, s0, s19 // (op 625) 


s_add_i32 s0, s1, s9 // (op 625) 


s_ashr_i32 s1, s0, s8 // (op 625) 


s_lshl_b32 s0, s13, 24 // (op 625) 
s_ashr_i32 s8, s0, 31 // (op 625) 

s_xor_b32 s0, s1, s8 // (op 625) 

s_sub_i32 s1, s0, s8 // (op 625) 


s_mul_hi_i32 s0, s1, s14 // (op 625) 
s_add_i32 s8, s0, s1 // (op 625) 

s_ashr_i32 s0, s8, 31 // (op 625) 
s_cmp_eq_i32 s14, 0 // (op 625) 
s_mov_b32 s9, scc // (op 625) 
s_cmp_eq_i32 s9, 0 // (op 625) 
s_cselect_b32 s18, 0, -1 // (op 625) 

s_and_b32 s9, s15, 31 // (op 625) 
s_lshl_b32 s19, 1, s9 // (op 625) 
s_add_i32 s49, s19, s18 // (op 625) 


s_and_b32 s18, s0, s49 // (op 625) 


s_add_i32 s0, s8, s18 // (op 625) 


s_ashr_i32 s8, s0, s9 // (op 625) 


s_lshl_b32 s0, s15, 24 // (op 625) 
s_ashr_i32 s9, s0, 31 // (op 625) 

s_xor_b32 s0, s8, s9 // (op 625) 

s_sub_i32 s8, s0, s9 // (op 625) 


s_add_u32 s18, s4, 16 // (op 625) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 625) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 625) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 625) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 625) 

s_mul_i32 s0, s8, s18 // (op 625) 

s_sub_i32 s8, s1, s0 // (op 625) 


s_mul_hi_i32 s9, s16, s11 // (op 625) 
s_add_i32 s0, s9, s16 // (op 625) 

s_ashr_i32 s1, s0, 31 // (op 625) 
s_cmp_eq_i32 s11, 0 // (op 625) 
s_mov_b32 s9, scc // (op 625) 
s_cmp_eq_i32 s9, 0 // (op 625) 
s_cselect_b32 s49, 0, -1 // (op 625) 

s_and_b32 s9, s46, 31 // (op 625) 
s_lshl_b32 s50, 1, s9 // (op 625) 
s_add_i32 s51, s50, s49 // (op 625) 


s_and_b32 s49, s1, s51 // (op 625) 


s_add_i32 s1, s0, s49 // (op 625) 


s_ashr_i32 s0, s1, s9 // (op 625) 


s_lshl_b32 s1, s46, 24 // (op 625) 
s_ashr_i32 s9, s1, 31 // (op 625) 

s_xor_b32 s1, s0, s9 // (op 625) 

s_sub_i32 s0, s1, s9 // (op 625) 


s_mul_i32 s1, s45, s18 // (op 625) 

s_mul_i32 s9, s10, s1 // (op 625) 

s_mul_i32 s1, s0, s9 // (op 625) 

s_sub_u32 s9, s16, s1 // (op 625) 

s_mul_hi_i32 s1, s9, s44 // (op 625) 
s_add_i32 s18, s1, s9 // (op 625) 


s_ashr_i32 s1, s18, 31 // (op 625) 
s_cmp_eq_i32 s44, 0 // (op 625) 
s_mov_b32 s9, scc // (op 625) 
s_cmp_eq_i32 s9, 0 // (op 625) 
s_cselect_b32 s19, 0, -1 // (op 625) 

s_and_b32 s9, s48, 31 // (op 625) 
s_lshl_b32 s49, 1, s9 // (op 625) 
s_add_i32 s50, s49, s19 // (op 625) 


s_and_b32 s19, s1, s50 // (op 625) 


s_add_i32 s1, s18, s19 // (op 625) 


s_ashr_i32 s18, s1, s9 // (op 625) 


s_lshl_b32 s1, s48, 24 // (op 625) 
s_ashr_i32 s9, s1, 31 // (op 625) 

s_xor_b32 s1, s18, s9 // (op 625) 

s_sub_i32 s18, s1, s9 // (op 625) 


s_cmp_eq_i32 s0, 0 // (op 625) 
s_mov_b32 s1, scc // (op 625) 

s_cmp_eq_i32 s1, 0 // (op 625) 
s_cselect_b32 s0, s18, s8 // (op 625) 



v_lshl_add_u32 v15, s0, 4, v14 // (op 625) 


v_mul_lo_u32 v23, v15, s23 // (op 625) most significant: low of low * high
v_mul_hi_u32 v14, v15, s22 // (op 625) most significant: high of low * low
v_mul_lo_u32 v22, v15, s22 // (op 625) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 625) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 625) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 625) most significant half



s_mul_i32 s1, s22, s5 // (op 627) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 627) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 627) least significant
s_mul_hi_u32 s9, s22, s4 // (op 627) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 627) most significant: sum
s_add_u32 s1, s1, s9 // (op 627) most significant: sum


s_mov_b32 s56, s24 // (op 627) reg expression
s_mov_b32 s57, s25 // (op 627) reg expression
s_mov_b32 s58, s0 // (op 627) reg expression
s_mov_b32 s59, 131072 // (op 627) call()

s_mul_hi_i32 s0, s16, s12 // (op 670) 
v_lshrrev_b32 v20, 2, v1 // (op 670) 
s_add_i32 s1, s0, s16 // (op 670) 

v_and_b32 v21, -4, v1 // (op 670) 
s_ashr_i32 s0, s1, 31 // (op 670) 
v_sub_u32 v22, v1, v21 // (op 670) 

s_cmp_eq_i32 s12, 0 // (op 670) 
s_mov_b32 s8, scc // (op 670) 
s_cmp_eq_i32 s8, 0 // (op 670) 
s_cselect_b32 s9, 0, -1 // (op 670) 

s_and_b32 s8, s13, 31 // (op 670) 
s_lshl_b32 s18, 1, s8 // (op 670) 
s_add_i32 s19, s18, s9 // (op 670) 


s_and_b32 s9, s0, s19 // (op 670) 


s_add_i32 s0, s1, s9 // (op 670) 


s_ashr_i32 s1, s0, s8 // (op 670) 


s_lshl_b32 s0, s13, 24 // (op 670) 
s_ashr_i32 s8, s0, 31 // (op 670) 

s_xor_b32 s0, s1, s8 // (op 670) 

s_sub_i32 s1, s0, s8 // (op 670) 


s_mul_hi_i32 s0, s1, s14 // (op 670) 
s_add_i32 s8, s0, s1 // (op 670) 

s_ashr_i32 s0, s8, 31 // (op 670) 
s_cmp_eq_i32 s14, 0 // (op 670) 
s_mov_b32 s9, scc // (op 670) 
s_cmp_eq_i32 s9, 0 // (op 670) 
s_cselect_b32 s18, 0, -1 // (op 670) 

s_and_b32 s9, s15, 31 // (op 670) 
s_lshl_b32 s19, 1, s9 // (op 670) 
s_add_i32 s49, s19, s18 // (op 670) 


s_and_b32 s18, s0, s49 // (op 670) 


s_add_i32 s0, s8, s18 // (op 670) 


s_ashr_i32 s8, s0, s9 // (op 670) 


s_lshl_b32 s0, s15, 24 // (op 670) 
s_ashr_i32 s9, s0, 31 // (op 670) 

s_xor_b32 s0, s8, s9 // (op 670) 

s_sub_i32 s8, s0, s9 // (op 670) 


s_mul_i32 s0, s1, s45 // (op 670) 

s_sub_u32 s1, s16, s0 // (op 670) 

s_mul_i32 s0, s8, s45 // (op 670) 
s_add_u32 s0, s0, s1 // (op 670) 


s_mul_hi_i32 s1, s16, s11 // (op 670) 
s_add_i32 s8, s1, s16 // (op 670) 

s_ashr_i32 s1, s8, 31 // (op 670) 
s_cmp_eq_i32 s11, 0 // (op 670) 
s_mov_b32 s9, scc // (op 670) 
s_cmp_eq_i32 s9, 0 // (op 670) 
s_cselect_b32 s18, 0, -1 // (op 670) 

s_and_b32 s9, s46, 31 // (op 670) 
s_lshl_b32 s19, 1, s9 // (op 670) 
s_add_i32 s49, s19, s18 // (op 670) 


s_and_b32 s18, s1, s49 // (op 670) 


s_add_i32 s1, s8, s18 // (op 670) 


s_ashr_i32 s8, s1, s9 // (op 670) 


s_lshl_b32 s1, s46, 24 // (op 670) 
s_ashr_i32 s9, s1, 31 // (op 670) 

s_xor_b32 s1, s8, s9 // (op 670) 

s_sub_i32 s8, s1, s9 // (op 670) 


s_add_u32 s18, s4, 16 // (op 670) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 670) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 670) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 670) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 670) 

s_mul_i32 s1, s45, s18 // (op 670) 

s_mul_i32 s9, s10, s1 // (op 670) 

s_mul_i32 s1, s8, s9 // (op 670) 

s_sub_u32 s9, s16, s1 // (op 670) 

s_mov_b32 s1, s9 // (op 670) 
s_mul_hi_i32 s18, s1, s44 // (op 670) 
s_add_i32 s19, s18, s1 // (op 670) 


s_ashr_i32 s1, s19, 31 // (op 670) 
s_cmp_eq_i32 s44, 0 // (op 670) 
s_mov_b32 s18, scc // (op 670) 
s_cmp_eq_i32 s18, 0 // (op 670) 
s_cselect_b32 s49, 0, -1 // (op 670) 

s_and_b32 s18, s48, 31 // (op 670) 
s_lshl_b32 s50, 1, s18 // (op 670) 
s_add_i32 s51, s50, s49 // (op 670) 


s_and_b32 s49, s1, s51 // (op 670) 


s_add_i32 s1, s19, s49 // (op 670) 


s_ashr_i32 s19, s1, s18 // (op 670) 


s_lshl_b32 s1, s48, 24 // (op 670) 
s_ashr_i32 s18, s1, 31 // (op 670) 

s_xor_b32 s1, s19, s18 // (op 670) 

s_sub_i32 s19, s1, s18 // (op 670) 


s_mul_i32 s1, s47, s45 // (op 670) 
s_add_u32 s50, s6, 16 // (op 670) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 670) End of Int64 add, unlocking SCC
s_sub_u32 s60, s50, 1 // (op 670) least significant half; sets scc
s_subb_u32 s61, s51, 0 // (op 670) most significant half; uses scc

s_ashr_i64 s[50:51], s[60:61], 4 // (op 670) 

s_sub_i32 s18, s50, s1 // (op 670) 

s_mul_i32 s49, s19, s18 // (op 670) 


s_sub_u32 s18, s9, s49 // (op 670) 


s_add_u32 s9, s18, s1 // (op 670) 


s_cmp_eq_i32 s8, 0 // (op 670) 
s_mov_b32 s1, scc // (op 670) 

s_cmp_eq_i32 s1, 0 // (op 670) 
s_cselect_b32 s8, s9, s0 // (op 670) 



v_lshl_add_u32 v21, s8, 4, v20 // (op 670) 


v_mul_lo_u32 v20, v21, s26 // (op 670) 
v_add_u32 v20, v20, v22 // (op 670) 


s_mul_i32 s1, s26, s7 // (op 672) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 672) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 672) least significant
s_mul_hi_u32 s9, s26, s6 // (op 672) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 672) most significant: sum
s_add_u32 s1, s1, s9 // (op 672) most significant: sum


s_mov_b32 s60, s28 // (op 672) reg expression
s_mov_b32 s61, s29 // (op 672) reg expression
s_mov_b32 s62, s0 // (op 672) reg expression
s_mov_b32 s63, 131072 // (op 672) call()

s_mul_hi_i32 s0, s16, s12 // (op 689) 
v_lshrrev_b32 v21, 2, v1 // (op 689) 
s_add_i32 s1, s0, s16 // (op 689) 

v_and_b32 v22, -4, v1 // (op 689) 
s_ashr_i32 s0, s1, 31 // (op 689) 
v_sub_u32 v23, v1, v22 // (op 689) 

s_cmp_eq_i32 s12, 0 // (op 689) 
s_mov_b32 s8, scc // (op 689) 
s_cmp_eq_i32 s8, 0 // (op 689) 
s_cselect_b32 s9, 0, -1 // (op 689) 

s_and_b32 s8, s13, 31 // (op 689) 
s_lshl_b32 s18, 1, s8 // (op 689) 
s_add_i32 s19, s18, s9 // (op 689) 


s_and_b32 s9, s0, s19 // (op 689) 


s_add_i32 s0, s1, s9 // (op 689) 


s_ashr_i32 s1, s0, s8 // (op 689) 


s_lshl_b32 s0, s13, 24 // (op 689) 
s_ashr_i32 s8, s0, 31 // (op 689) 

s_xor_b32 s0, s1, s8 // (op 689) 

s_sub_i32 s1, s0, s8 // (op 689) 


s_mul_hi_i32 s0, s1, s14 // (op 689) 
s_add_i32 s8, s0, s1 // (op 689) 

s_ashr_i32 s0, s8, 31 // (op 689) 
s_cmp_eq_i32 s14, 0 // (op 689) 
s_mov_b32 s9, scc // (op 689) 
s_cmp_eq_i32 s9, 0 // (op 689) 
s_cselect_b32 s18, 0, -1 // (op 689) 

s_and_b32 s9, s15, 31 // (op 689) 
s_lshl_b32 s19, 1, s9 // (op 689) 
s_add_i32 s49, s19, s18 // (op 689) 


s_and_b32 s18, s0, s49 // (op 689) 


s_add_i32 s0, s8, s18 // (op 689) 


s_ashr_i32 s8, s0, s9 // (op 689) 


s_lshl_b32 s0, s15, 24 // (op 689) 
s_ashr_i32 s9, s0, 31 // (op 689) 

s_xor_b32 s0, s8, s9 // (op 689) 

s_sub_i32 s8, s0, s9 // (op 689) 


s_mul_i32 s0, s1, s45 // (op 689) 

s_sub_u32 s1, s16, s0 // (op 689) 

s_mul_i32 s0, s8, s45 // (op 689) 
s_add_u32 s0, s0, s1 // (op 689) 


s_mul_hi_i32 s1, s16, s11 // (op 689) 
s_add_i32 s8, s1, s16 // (op 689) 

s_ashr_i32 s1, s8, 31 // (op 689) 
s_cmp_eq_i32 s11, 0 // (op 689) 
s_mov_b32 s9, scc // (op 689) 
s_cmp_eq_i32 s9, 0 // (op 689) 
s_cselect_b32 s18, 0, -1 // (op 689) 

s_and_b32 s9, s46, 31 // (op 689) 
s_lshl_b32 s19, 1, s9 // (op 689) 
s_add_i32 s49, s19, s18 // (op 689) 


s_and_b32 s18, s1, s49 // (op 689) 


s_add_i32 s1, s8, s18 // (op 689) 


s_ashr_i32 s8, s1, s9 // (op 689) 


s_lshl_b32 s1, s46, 24 // (op 689) 
s_ashr_i32 s9, s1, 31 // (op 689) 

s_xor_b32 s1, s8, s9 // (op 689) 

s_sub_i32 s8, s1, s9 // (op 689) 


s_add_u32 s18, s4, 16 // (op 689) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 689) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 689) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 689) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 689) 

s_mul_i32 s1, s45, s18 // (op 689) 

s_mul_i32 s9, s10, s1 // (op 689) 

s_mul_i32 s1, s8, s9 // (op 689) 

s_sub_u32 s9, s16, s1 // (op 689) 

s_mov_b32 s1, s9 // (op 689) 
s_mul_hi_i32 s18, s1, s44 // (op 689) 
s_add_i32 s19, s18, s1 // (op 689) 


s_ashr_i32 s1, s19, 31 // (op 689) 
s_cmp_eq_i32 s44, 0 // (op 689) 
s_mov_b32 s18, scc // (op 689) 
s_cmp_eq_i32 s18, 0 // (op 689) 
s_cselect_b32 s49, 0, -1 // (op 689) 

s_and_b32 s18, s48, 31 // (op 689) 
s_lshl_b32 s50, 1, s18 // (op 689) 
s_add_i32 s51, s50, s49 // (op 689) 


s_and_b32 s49, s1, s51 // (op 689) 


s_add_i32 s1, s19, s49 // (op 689) 


s_ashr_i32 s19, s1, s18 // (op 689) 


s_lshl_b32 s1, s48, 24 // (op 689) 
s_ashr_i32 s18, s1, 31 // (op 689) 

s_xor_b32 s1, s19, s18 // (op 689) 

s_sub_i32 s19, s1, s18 // (op 689) 


s_mul_i32 s1, s47, s45 // (op 689) 
s_add_u32 s50, s6, 16 // (op 689) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 689) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 689) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 689) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 4 // (op 689) 

s_sub_i32 s18, s50, s1 // (op 689) 

s_mul_i32 s49, s19, s18 // (op 689) 


s_sub_u32 s18, s9, s49 // (op 689) 


s_add_u32 s9, s18, s1 // (op 689) 


s_cmp_eq_i32 s8, 0 // (op 689) 
s_mov_b32 s1, scc // (op 689) 

s_cmp_eq_i32 s1, 0 // (op 689) 
s_cselect_b32 s8, s9, s0 // (op 689) 



v_lshl_add_u32 v22, s8, 4, v21 // (op 689) 


v_mul_lo_u32 v21, v22, s26 // (op 689) 
v_add_u32 v21, v21, v23 // (op 689) 


s_mul_i32 s1, s26, s7 // (op 691) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 691) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 691) least significant
s_mul_hi_u32 s9, s26, s6 // (op 691) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 691) most significant: sum
s_add_u32 s1, s1, s9 // (op 691) most significant: sum


s_mov_b32 s60, s28 // (op 691) reg expression
s_mov_b32 s61, s29 // (op 691) reg expression
s_mov_b32 s62, s0 // (op 691) reg expression
s_mov_b32 s63, 131072 // (op 691) call()

s_mul_hi_i32 s0, s16, s12 // (op 734) 
v_lshrrev_b32 v22, 2, v1 // (op 734) 
s_add_i32 s1, s0, s16 // (op 734) 

v_and_b32 v23, -4, v1 // (op 734) 
s_ashr_i32 s0, s1, 31 // (op 734) 
v_sub_u32 v24, v1, v23 // (op 734) 

s_cmp_eq_i32 s12, 0 // (op 734) 
s_mov_b32 s8, scc // (op 734) 
s_cmp_eq_i32 s8, 0 // (op 734) 
s_cselect_b32 s9, 0, -1 // (op 734) 

s_and_b32 s8, s13, 31 // (op 734) 
s_lshl_b32 s18, 1, s8 // (op 734) 
s_add_i32 s19, s18, s9 // (op 734) 


s_and_b32 s9, s0, s19 // (op 734) 


s_add_i32 s0, s1, s9 // (op 734) 


s_ashr_i32 s1, s0, s8 // (op 734) 


s_lshl_b32 s0, s13, 24 // (op 734) 
s_ashr_i32 s8, s0, 31 // (op 734) 

s_xor_b32 s0, s1, s8 // (op 734) 

s_sub_i32 s1, s0, s8 // (op 734) 


s_mul_hi_i32 s0, s1, s14 // (op 734) 
s_add_i32 s8, s0, s1 // (op 734) 

s_ashr_i32 s0, s8, 31 // (op 734) 
s_cmp_eq_i32 s14, 0 // (op 734) 
s_mov_b32 s9, scc // (op 734) 
s_cmp_eq_i32 s9, 0 // (op 734) 
s_cselect_b32 s18, 0, -1 // (op 734) 

s_and_b32 s9, s15, 31 // (op 734) 
s_lshl_b32 s19, 1, s9 // (op 734) 
s_add_i32 s49, s19, s18 // (op 734) 


s_and_b32 s18, s0, s49 // (op 734) 


s_add_i32 s0, s8, s18 // (op 734) 


s_ashr_i32 s8, s0, s9 // (op 734) 


s_lshl_b32 s0, s15, 24 // (op 734) 
s_ashr_i32 s9, s0, 31 // (op 734) 

s_xor_b32 s0, s8, s9 // (op 734) 

s_sub_i32 s8, s0, s9 // (op 734) 


s_add_u32 s18, s4, 16 // (op 734) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 734) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 734) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 734) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 734) 

s_mul_i32 s0, s8, s18 // (op 734) 

s_sub_i32 s8, s1, s0 // (op 734) 


s_mul_hi_i32 s9, s16, s11 // (op 734) 
s_add_i32 s0, s9, s16 // (op 734) 

s_ashr_i32 s1, s0, 31 // (op 734) 
s_cmp_eq_i32 s11, 0 // (op 734) 
s_mov_b32 s9, scc // (op 734) 
s_cmp_eq_i32 s9, 0 // (op 734) 
s_cselect_b32 s49, 0, -1 // (op 734) 

s_and_b32 s9, s46, 31 // (op 734) 
s_lshl_b32 s50, 1, s9 // (op 734) 
s_add_i32 s51, s50, s49 // (op 734) 


s_and_b32 s49, s1, s51 // (op 734) 


s_add_i32 s1, s0, s49 // (op 734) 


s_ashr_i32 s0, s1, s9 // (op 734) 


s_lshl_b32 s1, s46, 24 // (op 734) 
s_ashr_i32 s9, s1, 31 // (op 734) 

s_xor_b32 s1, s0, s9 // (op 734) 

s_sub_i32 s0, s1, s9 // (op 734) 


s_mul_i32 s1, s45, s18 // (op 734) 

s_mul_i32 s9, s10, s1 // (op 734) 

s_mul_i32 s1, s0, s9 // (op 734) 

s_sub_u32 s9, s16, s1 // (op 734) 

s_mul_hi_i32 s1, s9, s44 // (op 734) 
s_add_i32 s18, s1, s9 // (op 734) 


s_ashr_i32 s1, s18, 31 // (op 734) 
s_cmp_eq_i32 s44, 0 // (op 734) 
s_mov_b32 s9, scc // (op 734) 
s_cmp_eq_i32 s9, 0 // (op 734) 
s_cselect_b32 s19, 0, -1 // (op 734) 

s_and_b32 s9, s48, 31 // (op 734) 
s_lshl_b32 s49, 1, s9 // (op 734) 
s_add_i32 s50, s49, s19 // (op 734) 


s_and_b32 s19, s1, s50 // (op 734) 


s_add_i32 s1, s18, s19 // (op 734) 


s_ashr_i32 s18, s1, s9 // (op 734) 


s_lshl_b32 s1, s48, 24 // (op 734) 
s_ashr_i32 s9, s1, 31 // (op 734) 

s_xor_b32 s1, s18, s9 // (op 734) 

s_sub_i32 s18, s1, s9 // (op 734) 


s_cmp_eq_i32 s0, 0 // (op 734) 
s_mov_b32 s1, scc // (op 734) 

s_cmp_eq_i32 s1, 0 // (op 734) 
s_cselect_b32 s0, s18, s8 // (op 734) 



v_lshl_add_u32 v23, s0, 4, v22 // (op 734) 


v_mul_lo_u32 v22, v23, s30 // (op 734) 

v_add_u32 v23, v24, v22 // (op 734) 


s_mul_i32 s1, s30, s5 // (op 736) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 736) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 736) least significant
s_mul_hi_u32 s9, s30, s4 // (op 736) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 736) most significant: sum
s_add_u32 s1, s1, s9 // (op 736) most significant: sum


s_mov_b32 s64, s32 // (op 736) reg expression
s_mov_b32 s65, s33 // (op 736) reg expression
s_mov_b32 s66, s0 // (op 736) reg expression
s_mov_b32 s67, 131072 // (op 736) call()

s_mul_hi_i32 s0, s16, s12 // (op 753) 
v_lshrrev_b32 v22, 2, v1 // (op 753) 
s_add_i32 s1, s0, s16 // (op 753) 

v_and_b32 v24, -4, v1 // (op 753) 
s_ashr_i32 s0, s1, 31 // (op 753) 
v_sub_u32 v25, v1, v24 // (op 753) 

s_cmp_eq_i32 s12, 0 // (op 753) 
s_mov_b32 s8, scc // (op 753) 
s_cmp_eq_i32 s8, 0 // (op 753) 
s_cselect_b32 s9, 0, -1 // (op 753) 

s_and_b32 s8, s13, 31 // (op 753) 
s_lshl_b32 s18, 1, s8 // (op 753) 
s_add_i32 s19, s18, s9 // (op 753) 


s_and_b32 s9, s0, s19 // (op 753) 


s_add_i32 s0, s1, s9 // (op 753) 


s_ashr_i32 s1, s0, s8 // (op 753) 


s_lshl_b32 s0, s13, 24 // (op 753) 
s_ashr_i32 s8, s0, 31 // (op 753) 

s_xor_b32 s0, s1, s8 // (op 753) 

s_sub_i32 s1, s0, s8 // (op 753) 


s_mul_hi_i32 s0, s1, s14 // (op 753) 
s_add_i32 s8, s0, s1 // (op 753) 

s_ashr_i32 s0, s8, 31 // (op 753) 
s_cmp_eq_i32 s14, 0 // (op 753) 
s_mov_b32 s9, scc // (op 753) 
s_cmp_eq_i32 s9, 0 // (op 753) 
s_cselect_b32 s18, 0, -1 // (op 753) 

s_and_b32 s9, s15, 31 // (op 753) 
s_lshl_b32 s19, 1, s9 // (op 753) 
s_add_i32 s49, s19, s18 // (op 753) 


s_and_b32 s18, s0, s49 // (op 753) 


s_add_i32 s0, s8, s18 // (op 753) 


s_ashr_i32 s8, s0, s9 // (op 753) 


s_lshl_b32 s0, s15, 24 // (op 753) 
s_ashr_i32 s9, s0, 31 // (op 753) 

s_xor_b32 s0, s8, s9 // (op 753) 

s_sub_i32 s8, s0, s9 // (op 753) 


s_add_u32 s18, s4, 16 // (op 753) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 753) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 753) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 753) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 753) 

s_mul_i32 s0, s8, s18 // (op 753) 

s_sub_i32 s8, s1, s0 // (op 753) 


s_mul_hi_i32 s9, s16, s11 // (op 753) 
s_add_i32 s0, s9, s16 // (op 753) 

s_ashr_i32 s1, s0, 31 // (op 753) 
s_cmp_eq_i32 s11, 0 // (op 753) 
s_mov_b32 s9, scc // (op 753) 
s_cmp_eq_i32 s9, 0 // (op 753) 
s_cselect_b32 s49, 0, -1 // (op 753) 

s_and_b32 s9, s46, 31 // (op 753) 
s_lshl_b32 s50, 1, s9 // (op 753) 
s_add_i32 s51, s50, s49 // (op 753) 


s_and_b32 s49, s1, s51 // (op 753) 


s_add_i32 s1, s0, s49 // (op 753) 


s_ashr_i32 s0, s1, s9 // (op 753) 


s_lshl_b32 s1, s46, 24 // (op 753) 
s_ashr_i32 s9, s1, 31 // (op 753) 

s_xor_b32 s1, s0, s9 // (op 753) 

s_sub_i32 s0, s1, s9 // (op 753) 


s_mul_i32 s1, s45, s18 // (op 753) 

s_mul_i32 s9, s10, s1 // (op 753) 

s_mul_i32 s1, s0, s9 // (op 753) 

s_sub_u32 s9, s16, s1 // (op 753) 

s_mul_hi_i32 s1, s9, s44 // (op 753) 
s_add_i32 s18, s1, s9 // (op 753) 


s_ashr_i32 s1, s18, 31 // (op 753) 
s_cmp_eq_i32 s44, 0 // (op 753) 
s_mov_b32 s9, scc // (op 753) 
s_cmp_eq_i32 s9, 0 // (op 753) 
s_cselect_b32 s19, 0, -1 // (op 753) 

s_and_b32 s9, s48, 31 // (op 753) 
s_lshl_b32 s49, 1, s9 // (op 753) 
s_add_i32 s50, s49, s19 // (op 753) 


s_and_b32 s19, s1, s50 // (op 753) 


s_add_i32 s1, s18, s19 // (op 753) 


s_ashr_i32 s18, s1, s9 // (op 753) 


s_lshl_b32 s1, s48, 24 // (op 753) 
s_ashr_i32 s9, s1, 31 // (op 753) 

s_xor_b32 s1, s18, s9 // (op 753) 

s_sub_i32 s18, s1, s9 // (op 753) 


s_cmp_eq_i32 s0, 0 // (op 753) 
s_mov_b32 s1, scc // (op 753) 

s_cmp_eq_i32 s1, 0 // (op 753) 
s_cselect_b32 s0, s18, s8 // (op 753) 



v_lshl_add_u32 v24, s0, 4, v22 // (op 753) 


v_mul_lo_u32 v22, v24, s30 // (op 753) 

v_add_u32 v24, v25, v22 // (op 753) 


s_mul_i32 s1, s30, s5 // (op 755) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 755) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 755) least significant
s_mul_hi_u32 s9, s30, s4 // (op 755) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 755) most significant: sum
s_add_u32 s1, s1, s9 // (op 755) most significant: sum


s_mov_b32 s64, s32 // (op 755) reg expression
s_mov_b32 s65, s33 // (op 755) reg expression
s_mov_b32 s66, s0 // (op 755) reg expression
s_mov_b32 s67, 131072 // (op 755) call()

v_and_b32 v22, -8, v1 // (op 871) 
v_sub_u32 v25, v1, v22 // (op 871) 

v_lshlrev_b32 v22, 4, v25 // (op 871) 

v_lshrrev_b32 v25, 3, v1 // (op 871) 
v_lshl_add_u32 v26, v25, 7, v22 // (op 871) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 871) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[8:9], s[22:23], 3 // (op 491) 
v_mov_b64 v[26:27], s[8:9] // (op 491) convert


v_mov_b64 v[28:29], v[12:13] // (op 491) 
s_mov_b32 m0, s0 // (op 491) reg expression
buffer_load_dwordx4 v28, s[56:59], 0 offen offset: 0 lds // (op 491) Load value direct to lds
v_add_co_u32 v28, s[8:9], v28, v26 // (op 491) least significant half
v_addc_co_u32 v29, s[8:9], 0, v29, s[8:9] // (op 491) most significant half

s_add_u32 m0, m0, 1024 // (op 491) 
buffer_load_dwordx4 v28, s[56:59], 0 offen offset: 0 lds // (op 491) Load value direct to lds






buffer_load_ubyte v22, v23, s[64:67], 0 offen offset: 0 // (op 373) Load value
v_and_b32 v25, -8, v1 // (op 819) 
v_sub_u32 v26, v1, v25 // (op 819) 

v_lshlrev_b32 v25, 4, v26 // (op 819) 

v_lshrrev_b32 v26, 3, v1 // (op 819) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 819) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 819) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[8:9], s[2:3], 3 // (op 496) 
v_mov_b64 v[26:27], s[8:9] // (op 496) convert


v_mov_b64 v[28:29], v[16:17] // (op 496) 
s_add_u32 s1, 2048, s0 // (op 496) 
s_mov_b32 m0, s1 // (op 496) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 496) Load value direct to lds
v_add_co_u32 v28, s[8:9], v28, v26 // (op 496) least significant half
v_addc_co_u32 v29, s[8:9], 0, v29, s[8:9] // (op 496) most significant half

s_add_u32 m0, m0, 1024 // (op 496) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 496) Load value direct to lds







buffer_load_ubyte v25, v20, s[60:63], 0 offen offset: 0 // (op 379) Load value
v_and_b32 v26, -8, v1 // (op 1089) 
v_sub_u32 v27, v1, v26 // (op 1089) 

v_lshlrev_b32 v26, 4, v27 // (op 1089) 

v_lshrrev_b32 v27, 3, v1 // (op 1089) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 1089) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1089) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[8:9], s[22:23], 3 // (op 501) 
v_mov_b64 v[26:27], s[8:9] // (op 501) convert


v_mov_b64 v[28:29], v[12:13] // (op 501) 
s_add_u32 s1, 4096, s0 // (op 501) 
s_mov_b32 m0, s1 // (op 501) reg expression
s_mov_b32 s8, 128 // (op 501) call()
buffer_load_dwordx4 v28, s[56:59], s8 offen offset: 0 lds // (op 501) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 501) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 501) most significant half

s_add_u32 m0, m0, 1024 // (op 501) 
buffer_load_dwordx4 v28, s[56:59], s8 offen offset: 0 lds // (op 501) Load value direct to lds









buffer_load_ubyte v12, v23, s[64:67], 0 offen offset:4 // (op 385) Load value

v_and_b32 v13, -8, v1 // (op 1050) 
v_sub_u32 v23, v1, v13 // (op 1050) 

v_lshlrev_b32 v13, 4, v23 // (op 1050) 

v_lshrrev_b32 v23, 3, v1 // (op 1050) 
v_lshl_add_u32 v26, v23, 7, v13 // (op 1050) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1050) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[8:9], s[2:3], 3 // (op 503) 
v_mov_b64 v[26:27], s[8:9] // (op 503) convert


v_mov_b64 v[28:29], v[16:17] // (op 503) 
s_add_u32 s1, 6144, s0 // (op 503) 
s_mov_b32 m0, s1 // (op 503) reg expression
s_mov_b32 s8, 128 // (op 503) call()
buffer_load_dwordx4 v28, s[52:55], s8 offen offset: 0 lds // (op 503) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 503) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 503) most significant half

s_add_u32 m0, m0, 1024 // (op 503) 
buffer_load_dwordx4 v28, s[52:55], s8 offen offset: 0 lds // (op 503) Load value direct to lds









buffer_load_ubyte v13, v20, s[60:63], 0 offen offset:4 // (op 391) Load value

v_and_b32 v20, -4, v1 // (op 977) 
v_sub_u32 v23, v1, v20 // (op 977) 

v_and_b32 v20, 4294967292, v1 // (op 977) 
v_add_u32 v16, v20, v23 // (op 977) 





s_waitcnt vmcnt(9)
ds_write_b8 v16, v22 offset:8192 // (op 394) Store local data 

v_and_b32 v20, -4, v1 // (op 924) 
v_sub_u32 v23, v1, v20 // (op 924) 

v_and_b32 v20, 4294967292, v1 // (op 924) 
v_add_u32 v16, v20, v23 // (op 924) 





s_waitcnt vmcnt(6)
ds_write_b8 v16, v25 offset:8256 // (op 396) Store local data 

s_barrier  // (op 1207) 



ds_read_b128 v[26:29], v5 // (op 413) Load local data 
ds_read_b128 v[30:33], v5 offset:64 // (op 413) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1241) 



ds_read_u8 v20, v6 offset:8192 // (op 419) Load local data 
s_barrier  // (op 1191) 



ds_read_b128 v[34:37], v3 offset:2048 // (op 410) Load local data 
ds_read_b128 v[38:41], v3 offset:2112 // (op 410) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1224) 



ds_read_u8 v23, v0 offset:8256 // (op 416) Load local data 
v_and_b32 v16, -8, v1 // (op 791) 
v_sub_u32 v17, v1, v16 // (op 791) 

v_lshlrev_b32 v16, 4, v17 // (op 791) 

v_lshrrev_b32 v17, 3, v1 // (op 791) 
v_lshl_add_u32 v42, v17, 7, v16 // (op 791) 


s_nop 0
v_readfirstlane_b32 s0, v42 // (op 791) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 843) 
v_sub_u32 v17, v1, v16 // (op 843) 

v_lshlrev_b32 v16, 4, v17 // (op 843) 

v_lshrrev_b32 v17, 3, v1 // (op 843) 
v_lshl_add_u32 v42, v17, 7, v16 // (op 843) 


s_nop 0
v_readfirstlane_b32 s1, v42 // (op 843) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 895) 
v_sub_u32 v17, v1, v16 // (op 895) 

v_and_b32 v16, 4294967292, v1 // (op 895) 
v_add_u32 v42, v16, v17 // (op 895) 


v_and_b32 v16, -4, v1 // (op 948) 
v_sub_u32 v17, v1, v16 // (op 948) 

v_and_b32 v16, 4294967292, v1 // (op 948) 
v_add_u32 v43, v16, v17 // (op 948) 


v_and_b32 v16, -8, v1 // (op 1035) 
v_sub_u32 v17, v1, v16 // (op 1035) 

v_lshlrev_b32 v16, 4, v17 // (op 1035) 

v_lshrrev_b32 v17, 3, v1 // (op 1035) 
v_lshl_add_u32 v44, v17, 7, v16 // (op 1035) 


s_nop 0
v_readfirstlane_b32 s8, v44 // (op 1035) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 1074) 
v_sub_u32 v17, v1, v16 // (op 1074) 

v_lshlrev_b32 v16, 4, v17 // (op 1074) 

v_lshrrev_b32 v17, 3, v1 // (op 1074) 
v_lshl_add_u32 v44, v17, 7, v16 // (op 1074) 


s_nop 0
v_readfirstlane_b32 s9, v44 // (op 1074) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 1113) 
v_sub_u32 v17, v1, v16 // (op 1113) 

v_and_b32 v16, 4294967292, v1 // (op 1113) 
v_add_u32 v44, v16, v17 // (op 1113) 


v_and_b32 v16, -4, v1 // (op 1137) 
v_sub_u32 v17, v1, v16 // (op 1137) 

v_and_b32 v16, 4294967292, v1 // (op 1137) 
v_add_u32 v45, v16, v17 // (op 1137) 


s_mov_b32 s18, 0 // (op 61) call()
s_ashr_i64 s[50:51], s[34:35], 7 // (op 60) 
s_mov_b32 s19, s50 // (op 60) convert

s_and_b32 s49, s19, 4294967294 // (op 60) 

s_cmp_lt_u32 s18, s49 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_60 if false)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_60:
s_add_u32 s18, s18, 2 // (op 62) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v23, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 90) 
s_barrier  // (op 1210) 

s_lshl_b64 s[50:51], s[22:23], 3 // (op 463) 
v_mov_b64 v[16:17], s[50:51] // (op 463) convert


v_mov_b64 v[46:47], v[14:15] // (op 463) 
v_mov_b64 v[48:49], v[46:47] // (op 463) 
s_mov_b32 m0, s1 // (op 463) reg expression
s_mov_b32 s19, 256 // (op 463) call()
buffer_load_dwordx4 v48, s[56:59], s19 offen offset: 0 lds // (op 463) Load value direct to lds
v_add_co_u32 v48, s[50:51], v48, v16 // (op 463) least significant half
v_addc_co_u32 v49, s[50:51], 0, v49, s[50:51] // (op 463) most significant half

s_add_u32 m0, m0, 1024 // (op 463) 
buffer_load_dwordx4 v48, s[56:59], s19 offen offset: 0 lds // (op 463) Load value direct to lds





v_mov_b32 v16, v24 // (op 25) 

buffer_load_ubyte v22, v16, s[64:67], 0 offen offset:8 // (op 25) Load value
s_barrier  // (op 1194) 

s_lshl_b64 s[50:51], s[2:3], 3 // (op 458) 
v_mov_b64 v[48:49], s[50:51] // (op 458) convert


v_mov_b64 v[50:51], v[18:19] // (op 458) 
v_mov_b64 v[52:53], v[50:51] // (op 458) 
s_add_u32 s19, 2048, s0 // (op 458) 
s_mov_b32 m0, s19 // (op 458) reg expression
s_mov_b32 s49, 256 // (op 458) call()
buffer_load_dwordx4 v52, s[52:55], s49 offen offset: 0 lds // (op 458) Load value direct to lds
v_add_co_u32 v52, s[50:51], v52, v48 // (op 458) least significant half
v_addc_co_u32 v53, s[50:51], 0, v53, s[50:51] // (op 458) most significant half

s_add_u32 m0, m0, 1024 // (op 458) 
buffer_load_dwordx4 v52, s[52:55], s49 offen offset: 0 lds // (op 458) Load value direct to lds






v_mov_b32 v17, v21 // (op 17) 

buffer_load_ubyte v25, v17, s[60:63], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 1280) 



s_waitcnt vmcnt(9)
ds_write_b8 v45, v12 offset:8320 // (op 259) Store local data 
s_barrier  // (op 1274) 



s_waitcnt vmcnt(6)
ds_write_b8 v44, v13 offset:8384 // (op 258) Store local data 
s_barrier  // (op 1264) 



ds_read_b128 v[26:29], v9 offset:4096 // (op 251) Load local data 
ds_read_b128 v[30:33], v9 offset:4160 // (op 251) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1277) 



ds_read_u8 v20, v10 offset:8320 // (op 255) Load local data 
s_barrier  // (op 1258) 
s_barrier  // (op 1335) 



ds_read_b128 v[34:37], v7 offset:6144 // (op 249) Load local data 
ds_read_b128 v[38:41], v7 offset:6208 // (op 249) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1271) 



ds_read_u8 v23, v4 offset:8384 // (op 253) Load local data 
s_barrier  // (op 1329) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v23, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 256) 
s_barrier  // (op 1268) 

s_lshl_b64 s[50:51], s[22:23], 3 // (op 486) 
v_mov_b64 v[48:49], s[50:51] // (op 486) convert


v_mov_b64 v[52:53], v[46:47] // (op 486) 
s_add_u32 s19, 4096, s9 // (op 486) 
s_mov_b32 m0, s19 // (op 486) reg expression
s_mov_b32 s49, 384 // (op 486) call()
buffer_load_dwordx4 v52, s[56:59], s49 offen offset: 0 lds // (op 486) Load value direct to lds
v_add_co_u32 v52, s[50:51], v52, v48 // (op 486) least significant half
v_addc_co_u32 v53, s[50:51], 0, v53, s[50:51] // (op 486) most significant half

s_add_u32 m0, m0, 1024 // (op 486) 
buffer_load_dwordx4 v52, s[56:59], s49 offen offset: 0 lds // (op 486) Load value direct to lds








v_add_co_u32 v14, vcc, 256, v14 // (op 617) least significant half
buffer_load_ubyte v12, v16, s[64:67], 0 offen offset:12 // (op 262) Load value
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 617) most significant half

s_barrier  // (op 1261) 
v_add_u32 v24, 8, v24 // (op 745) 

s_lshl_b64 s[50:51], s[2:3], 3 // (op 481) 
v_mov_b64 v[46:47], s[50:51] // (op 481) convert


v_mov_b64 v[48:49], v[50:51] // (op 481) 
s_add_u32 s19, 6144, s8 // (op 481) 
s_mov_b32 m0, s19 // (op 481) reg expression
s_mov_b32 s49, 384 // (op 481) call()
buffer_load_dwordx4 v48, s[52:55], s49 offen offset: 0 lds // (op 481) Load value direct to lds
v_add_co_u32 v48, s[50:51], v48, v46 // (op 481) least significant half
v_addc_co_u32 v49, s[50:51], 0, v49, s[50:51] // (op 481) most significant half

s_add_u32 m0, m0, 1024 // (op 481) 
buffer_load_dwordx4 v48, s[52:55], s49 offen offset: 0 lds // (op 481) Load value direct to lds








v_add_co_u32 v18, vcc, 256, v18 // (op 553) least significant half
buffer_load_ubyte v13, v17, s[60:63], 0 offen offset:12 // (op 261) Load value
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 553) most significant half

s_barrier  // (op 1244) 
v_add_u32 v21, 8, v21 // (op 681) 



s_waitcnt vmcnt(9)
ds_write_b8 v43, v22 offset:8192 // (op 30) Store local data 
s_barrier  // (op 1227) 



s_waitcnt vmcnt(6)
ds_write_b8 v42, v25 offset:8256 // (op 21) Store local data 
s_barrier  // (op 1216) 



ds_read_b128 v[26:29], v5 // (op 77) Load local data 
ds_read_b128 v[30:33], v5 offset:64 // (op 77) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1251) 



ds_read_u8 v20, v6 offset:8192 // (op 87) Load local data 
s_barrier  // (op 1200) 



ds_read_b128 v[34:37], v3 offset:2048 // (op 72) Load local data 
ds_read_b128 v[38:41], v3 offset:2112 // (op 72) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1234) 



ds_read_u8 v23, v0 offset:8256 // (op 82) Load local data 
s_ashr_i64 s[50:51], s[34:35], 7 // (op 60) 
s_mov_b32 s19, s50 // (op 60) convert

s_and_b32 s49, s19, 4294967294 // (op 60) 

s_cmp_lt_u32 s18, s49 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoop_60 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoop_60:















s_barrier  // (op 1293) 
s_mul_hi_i32 s0, s16, s12 // (op 516) 
v_lshrrev_b32 v14, 3, v1 // (op 516) 
s_add_i32 s1, s0, s16 // (op 516) 

v_and_b32 v15, -8, v1 // (op 516) 
s_ashr_i32 s0, s1, 31 // (op 516) 
v_sub_u32 v16, v1, v15 // (op 516) 

s_cmp_eq_i32 s12, 0 // (op 516) 
s_mov_b32 s8, scc // (op 516) 
v_lshlrev_b32 v15, 4, v16 // (op 516) 

s_cmp_eq_i32 s8, 0 // (op 516) 
s_cselect_b32 s9, 0, -1 // (op 516) 

s_and_b32 s8, s13, 31 // (op 516) 
s_lshl_b32 s18, 1, s8 // (op 516) 
s_add_i32 s19, s18, s9 // (op 516) 


s_and_b32 s9, s0, s19 // (op 516) 


s_add_i32 s0, s1, s9 // (op 516) 


s_ashr_i32 s1, s0, s8 // (op 516) 


s_lshl_b32 s0, s13, 24 // (op 516) 
s_ashr_i32 s8, s0, 31 // (op 516) 

s_xor_b32 s0, s1, s8 // (op 516) 

s_sub_i32 s1, s0, s8 // (op 516) 


s_mul_hi_i32 s0, s1, s14 // (op 516) 
s_add_i32 s8, s0, s1 // (op 516) 

s_ashr_i32 s0, s8, 31 // (op 516) 
s_cmp_eq_i32 s14, 0 // (op 516) 
s_mov_b32 s9, scc // (op 516) 
s_cmp_eq_i32 s9, 0 // (op 516) 
s_cselect_b32 s18, 0, -1 // (op 516) 

s_and_b32 s9, s15, 31 // (op 516) 
s_lshl_b32 s19, 1, s9 // (op 516) 
s_add_i32 s49, s19, s18 // (op 516) 


s_and_b32 s18, s0, s49 // (op 516) 


s_add_i32 s0, s8, s18 // (op 516) 


s_ashr_i32 s8, s0, s9 // (op 516) 


s_lshl_b32 s0, s15, 24 // (op 516) 
s_ashr_i32 s9, s0, 31 // (op 516) 

s_xor_b32 s0, s8, s9 // (op 516) 

s_sub_i32 s8, s0, s9 // (op 516) 


s_mul_i32 s0, s1, s45 // (op 516) 

s_sub_u32 s1, s16, s0 // (op 516) 

s_mul_i32 s0, s8, s45 // (op 516) 
s_add_u32 s0, s0, s1 // (op 516) 


s_mul_hi_i32 s1, s16, s11 // (op 516) 
s_add_i32 s8, s1, s16 // (op 516) 

s_ashr_i32 s1, s8, 31 // (op 516) 
s_cmp_eq_i32 s11, 0 // (op 516) 
s_mov_b32 s9, scc // (op 516) 
s_cmp_eq_i32 s9, 0 // (op 516) 
s_cselect_b32 s18, 0, -1 // (op 516) 

s_and_b32 s9, s46, 31 // (op 516) 
s_lshl_b32 s19, 1, s9 // (op 516) 
s_add_i32 s49, s19, s18 // (op 516) 


s_and_b32 s18, s1, s49 // (op 516) 


s_add_i32 s1, s8, s18 // (op 516) 


s_ashr_i32 s8, s1, s9 // (op 516) 


s_lshl_b32 s1, s46, 24 // (op 516) 
s_ashr_i32 s9, s1, 31 // (op 516) 

s_xor_b32 s1, s8, s9 // (op 516) 

s_sub_i32 s8, s1, s9 // (op 516) 


s_add_u32 s18, s4, 16 // (op 516) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 516) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 516) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 516) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 516) 

s_mul_i32 s1, s45, s18 // (op 516) 

s_mul_i32 s9, s10, s1 // (op 516) 

s_mul_i32 s1, s8, s9 // (op 516) 

s_sub_u32 s9, s16, s1 // (op 516) 

s_mov_b32 s1, s9 // (op 516) 
s_mul_hi_i32 s18, s1, s44 // (op 516) 
s_add_i32 s19, s18, s1 // (op 516) 


s_ashr_i32 s1, s19, 31 // (op 516) 
s_cmp_eq_i32 s44, 0 // (op 516) 
s_mov_b32 s18, scc // (op 516) 
s_cmp_eq_i32 s18, 0 // (op 516) 
s_cselect_b32 s49, 0, -1 // (op 516) 

s_and_b32 s18, s48, 31 // (op 516) 
s_lshl_b32 s50, 1, s18 // (op 516) 
s_add_i32 s51, s50, s49 // (op 516) 


s_and_b32 s49, s1, s51 // (op 516) 


s_add_i32 s1, s19, s49 // (op 516) 


s_ashr_i32 s19, s1, s18 // (op 516) 


s_lshl_b32 s1, s48, 24 // (op 516) 
s_ashr_i32 s18, s1, 31 // (op 516) 

s_xor_b32 s1, s19, s18 // (op 516) 

s_sub_i32 s19, s1, s18 // (op 516) 


s_mul_i32 s1, s47, s45 // (op 516) 
s_add_u32 s50, s6, 16 // (op 516) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 516) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 516) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 516) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 516) 

s_sub_i32 s18, s50, s1 // (op 516) 

s_mul_i32 s49, s19, s18 // (op 516) 


s_sub_u32 s18, s9, s49 // (op 516) 


s_add_u32 s9, s18, s1 // (op 516) 


s_cmp_eq_i32 s8, 0 // (op 516) 
s_mov_b32 s1, scc // (op 516) 

s_cmp_eq_i32 s1, 0 // (op 516) 
s_cselect_b32 s8, s9, s0 // (op 516) 



v_lshl_add_u32 v16, s8, 4, v14 // (op 516) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 516) 
s_lshr_b32 s8, s0, 1 // (op 516) 

v_lshl_add_u32 v14, s8, 8, v15 // (op 516) 


v_mov_b32 v19, 0 // (op 516) convert
v_mov_b32 v18, v14 // (op 516) convert

v_mul_lo_u32 v15, v16, s3 // (op 516) most significant: low of low * high
v_mul_hi_u32 v17, v16, s2 // (op 516) most significant: high of low * low
v_mul_lo_u32 v14, v16, s2 // (op 516) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 516) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v18 // (op 516) least significant half
v_addc_co_u32 v15, s[0:1], v15, v19, s[0:1] // (op 516) most significant half



s_mul_i32 s1, s2, s7 // (op 518) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 518) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 518) least significant
s_mul_hi_u32 s9, s2, s6 // (op 518) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 518) most significant: sum
s_add_u32 s1, s1, s9 // (op 518) most significant: sum


s_mov_b32 s52, s20 // (op 518) reg expression
s_mov_b32 s53, s21 // (op 518) reg expression
s_mov_b32 s54, s0 // (op 518) reg expression
s_mov_b32 s55, 131072 // (op 518) call()

s_mul_hi_i32 s0, s16, s12 // (op 583) 
v_lshrrev_b32 v16, 3, v1 // (op 583) 
s_add_i32 s1, s0, s16 // (op 583) 

v_and_b32 v17, -8, v1 // (op 583) 
s_ashr_i32 s0, s1, 31 // (op 583) 
v_sub_u32 v18, v1, v17 // (op 583) 

s_cmp_eq_i32 s12, 0 // (op 583) 
s_mov_b32 s8, scc // (op 583) 
v_lshlrev_b32 v17, 4, v18 // (op 583) 

s_cmp_eq_i32 s8, 0 // (op 583) 
s_cselect_b32 s9, 0, -1 // (op 583) 

s_and_b32 s8, s13, 31 // (op 583) 
s_lshl_b32 s18, 1, s8 // (op 583) 
s_add_i32 s19, s18, s9 // (op 583) 


s_and_b32 s9, s0, s19 // (op 583) 


s_add_i32 s0, s1, s9 // (op 583) 


s_ashr_i32 s1, s0, s8 // (op 583) 


s_lshl_b32 s0, s13, 24 // (op 583) 
s_ashr_i32 s8, s0, 31 // (op 583) 

s_xor_b32 s0, s1, s8 // (op 583) 

s_sub_i32 s1, s0, s8 // (op 583) 


s_mul_hi_i32 s0, s1, s14 // (op 583) 
s_add_i32 s8, s0, s1 // (op 583) 

s_ashr_i32 s0, s8, 31 // (op 583) 
s_cmp_eq_i32 s14, 0 // (op 583) 
s_mov_b32 s9, scc // (op 583) 
s_cmp_eq_i32 s9, 0 // (op 583) 
s_cselect_b32 s18, 0, -1 // (op 583) 

s_and_b32 s9, s15, 31 // (op 583) 
s_lshl_b32 s19, 1, s9 // (op 583) 
s_add_i32 s49, s19, s18 // (op 583) 


s_and_b32 s18, s0, s49 // (op 583) 


s_add_i32 s0, s8, s18 // (op 583) 


s_ashr_i32 s8, s0, s9 // (op 583) 


s_lshl_b32 s0, s15, 24 // (op 583) 
s_ashr_i32 s9, s0, 31 // (op 583) 

s_xor_b32 s0, s8, s9 // (op 583) 

s_sub_i32 s8, s0, s9 // (op 583) 


s_add_u32 s18, s4, 16 // (op 583) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 583) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 583) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 583) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 583) 

s_mul_i32 s0, s8, s18 // (op 583) 

s_sub_i32 s8, s1, s0 // (op 583) 


s_mul_hi_i32 s9, s16, s11 // (op 583) 
s_add_i32 s0, s9, s16 // (op 583) 

s_ashr_i32 s1, s0, 31 // (op 583) 
s_cmp_eq_i32 s11, 0 // (op 583) 
s_mov_b32 s9, scc // (op 583) 
s_cmp_eq_i32 s9, 0 // (op 583) 
s_cselect_b32 s49, 0, -1 // (op 583) 

s_and_b32 s9, s46, 31 // (op 583) 
s_lshl_b32 s50, 1, s9 // (op 583) 
s_add_i32 s51, s50, s49 // (op 583) 


s_and_b32 s49, s1, s51 // (op 583) 


s_add_i32 s1, s0, s49 // (op 583) 


s_ashr_i32 s0, s1, s9 // (op 583) 


s_lshl_b32 s1, s46, 24 // (op 583) 
s_ashr_i32 s9, s1, 31 // (op 583) 

s_xor_b32 s1, s0, s9 // (op 583) 

s_sub_i32 s0, s1, s9 // (op 583) 


s_mul_i32 s1, s45, s18 // (op 583) 

s_mul_i32 s9, s10, s1 // (op 583) 

s_mul_i32 s1, s0, s9 // (op 583) 

s_sub_u32 s9, s16, s1 // (op 583) 

s_mul_hi_i32 s1, s9, s44 // (op 583) 
s_add_i32 s18, s1, s9 // (op 583) 


s_ashr_i32 s1, s18, 31 // (op 583) 
s_cmp_eq_i32 s44, 0 // (op 583) 
s_mov_b32 s9, scc // (op 583) 
s_cmp_eq_i32 s9, 0 // (op 583) 
s_cselect_b32 s19, 0, -1 // (op 583) 

s_and_b32 s9, s48, 31 // (op 583) 
s_lshl_b32 s49, 1, s9 // (op 583) 
s_add_i32 s50, s49, s19 // (op 583) 


s_and_b32 s19, s1, s50 // (op 583) 


s_add_i32 s1, s18, s19 // (op 583) 


s_ashr_i32 s18, s1, s9 // (op 583) 


s_lshl_b32 s1, s48, 24 // (op 583) 
s_ashr_i32 s9, s1, 31 // (op 583) 

s_xor_b32 s1, s18, s9 // (op 583) 

s_sub_i32 s18, s1, s9 // (op 583) 


s_cmp_eq_i32 s0, 0 // (op 583) 
s_mov_b32 s1, scc // (op 583) 

s_cmp_eq_i32 s1, 0 // (op 583) 
s_cselect_b32 s0, s18, s8 // (op 583) 



v_lshl_add_u32 v18, s0, 4, v16 // (op 583) 


v_mul_lo_u32 v25, v18, s23 // (op 583) most significant: low of low * high
v_mul_hi_u32 v16, v18, s22 // (op 583) most significant: high of low * low
v_mul_lo_u32 v24, v18, s22 // (op 583) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 583) most significant: sum


s_ashr_i64 s[0:1], s[34:35], 7 // (op 583) 
s_lshr_b32 s8, s0, 1 // (op 583) 

v_lshl_add_u32 v16, s8, 8, v17 // (op 583) 


v_mov_b32 v19, 0 // (op 583) convert
v_mov_b32 v18, v16 // (op 583) convert

v_add_co_u32 v16, s[0:1], v18, v24 // (op 583) least significant half
v_addc_co_u32 v17, s[0:1], v19, v25, s[0:1] // (op 583) most significant half



s_mul_i32 s1, s22, s5 // (op 585) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 585) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 585) least significant
s_mul_hi_u32 s9, s22, s4 // (op 585) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 585) most significant: sum
s_add_u32 s1, s1, s9 // (op 585) most significant: sum


s_mov_b32 s56, s24 // (op 585) reg expression
s_mov_b32 s57, s25 // (op 585) reg expression
s_mov_b32 s58, s0 // (op 585) reg expression
s_mov_b32 s59, 131072 // (op 585) call()

s_mul_hi_i32 s0, s16, s12 // (op 647) 
v_lshrrev_b32 v18, 2, v1 // (op 647) 
s_add_i32 s1, s0, s16 // (op 647) 

v_and_b32 v19, -4, v1 // (op 647) 
s_ashr_i32 s0, s1, 31 // (op 647) 
v_sub_u32 v21, v1, v19 // (op 647) 

s_cmp_eq_i32 s12, 0 // (op 647) 
s_mov_b32 s8, scc // (op 647) 
s_cmp_eq_i32 s8, 0 // (op 647) 
s_cselect_b32 s9, 0, -1 // (op 647) 

s_and_b32 s8, s13, 31 // (op 647) 
s_lshl_b32 s18, 1, s8 // (op 647) 
s_add_i32 s19, s18, s9 // (op 647) 


s_and_b32 s9, s0, s19 // (op 647) 


s_add_i32 s0, s1, s9 // (op 647) 


s_ashr_i32 s1, s0, s8 // (op 647) 


s_lshl_b32 s0, s13, 24 // (op 647) 
s_ashr_i32 s8, s0, 31 // (op 647) 

s_xor_b32 s0, s1, s8 // (op 647) 

s_sub_i32 s1, s0, s8 // (op 647) 


s_mul_hi_i32 s0, s1, s14 // (op 647) 
s_add_i32 s8, s0, s1 // (op 647) 

s_ashr_i32 s0, s8, 31 // (op 647) 
s_cmp_eq_i32 s14, 0 // (op 647) 
s_mov_b32 s9, scc // (op 647) 
s_cmp_eq_i32 s9, 0 // (op 647) 
s_cselect_b32 s18, 0, -1 // (op 647) 

s_and_b32 s9, s15, 31 // (op 647) 
s_lshl_b32 s19, 1, s9 // (op 647) 
s_add_i32 s49, s19, s18 // (op 647) 


s_and_b32 s18, s0, s49 // (op 647) 


s_add_i32 s0, s8, s18 // (op 647) 


s_ashr_i32 s8, s0, s9 // (op 647) 


s_lshl_b32 s0, s15, 24 // (op 647) 
s_ashr_i32 s9, s0, 31 // (op 647) 

s_xor_b32 s0, s8, s9 // (op 647) 

s_sub_i32 s8, s0, s9 // (op 647) 


s_mul_i32 s0, s1, s45 // (op 647) 

s_sub_u32 s1, s16, s0 // (op 647) 

s_mul_i32 s0, s8, s45 // (op 647) 
s_add_u32 s0, s0, s1 // (op 647) 


s_mul_hi_i32 s1, s16, s11 // (op 647) 
s_add_i32 s8, s1, s16 // (op 647) 

s_ashr_i32 s1, s8, 31 // (op 647) 
s_cmp_eq_i32 s11, 0 // (op 647) 
s_mov_b32 s9, scc // (op 647) 
s_cmp_eq_i32 s9, 0 // (op 647) 
s_cselect_b32 s18, 0, -1 // (op 647) 

s_and_b32 s9, s46, 31 // (op 647) 
s_lshl_b32 s19, 1, s9 // (op 647) 
s_add_i32 s49, s19, s18 // (op 647) 


s_and_b32 s18, s1, s49 // (op 647) 


s_add_i32 s1, s8, s18 // (op 647) 


s_ashr_i32 s8, s1, s9 // (op 647) 


s_lshl_b32 s1, s46, 24 // (op 647) 
s_ashr_i32 s9, s1, 31 // (op 647) 

s_xor_b32 s1, s8, s9 // (op 647) 

s_sub_i32 s8, s1, s9 // (op 647) 


s_add_u32 s18, s4, 16 // (op 647) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 647) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 647) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 647) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 647) 

s_mul_i32 s1, s45, s18 // (op 647) 

s_mul_i32 s9, s10, s1 // (op 647) 

s_mul_i32 s1, s8, s9 // (op 647) 

s_sub_u32 s9, s16, s1 // (op 647) 

s_mov_b32 s1, s9 // (op 647) 
s_mul_hi_i32 s18, s1, s44 // (op 647) 
s_add_i32 s19, s18, s1 // (op 647) 


s_ashr_i32 s1, s19, 31 // (op 647) 
s_cmp_eq_i32 s44, 0 // (op 647) 
s_mov_b32 s18, scc // (op 647) 
s_cmp_eq_i32 s18, 0 // (op 647) 
s_cselect_b32 s49, 0, -1 // (op 647) 

s_and_b32 s18, s48, 31 // (op 647) 
s_lshl_b32 s50, 1, s18 // (op 647) 
s_add_i32 s51, s50, s49 // (op 647) 


s_and_b32 s49, s1, s51 // (op 647) 


s_add_i32 s1, s19, s49 // (op 647) 


s_ashr_i32 s19, s1, s18 // (op 647) 


s_lshl_b32 s1, s48, 24 // (op 647) 
s_ashr_i32 s18, s1, 31 // (op 647) 

s_xor_b32 s1, s19, s18 // (op 647) 

s_sub_i32 s19, s1, s18 // (op 647) 


s_mul_i32 s1, s47, s45 // (op 647) 
s_add_u32 s50, s6, 16 // (op 647) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 647) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 647) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 647) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 647) 

s_sub_i32 s18, s50, s1 // (op 647) 

s_mul_i32 s49, s19, s18 // (op 647) 


s_sub_u32 s18, s9, s49 // (op 647) 


s_add_u32 s9, s18, s1 // (op 647) 


s_cmp_eq_i32 s8, 0 // (op 647) 
s_mov_b32 s1, scc // (op 647) 

s_cmp_eq_i32 s1, 0 // (op 647) 
s_cselect_b32 s8, s9, s0 // (op 647) 



v_lshl_add_u32 v19, s8, 4, v18 // (op 647) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 647) 
s_lshr_b32 s8, s0, 1 // (op 647) 

v_lshl_add_u32 v18, s8, 3, v21 // (op 647) 


v_mul_lo_u32 v21, v19, s26 // (op 647) 
v_add_u32 v21, v21, v18 // (op 647) 


s_mul_i32 s1, s26, s7 // (op 649) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 649) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 649) least significant
s_mul_hi_u32 s9, s26, s6 // (op 649) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 649) most significant: sum
s_add_u32 s1, s1, s9 // (op 649) most significant: sum


s_mov_b32 s60, s28 // (op 649) reg expression
s_mov_b32 s61, s29 // (op 649) reg expression
s_mov_b32 s62, s0 // (op 649) reg expression
s_mov_b32 s63, 131072 // (op 649) call()

s_mul_hi_i32 s0, s16, s12 // (op 711) 
v_lshrrev_b32 v22, 2, v1 // (op 711) 
s_add_i32 s1, s0, s16 // (op 711) 

v_and_b32 v18, -4, v1 // (op 711) 
s_ashr_i32 s0, s1, 31 // (op 711) 
v_sub_u32 v19, v1, v18 // (op 711) 

s_cmp_eq_i32 s12, 0 // (op 711) 
s_mov_b32 s8, scc // (op 711) 
s_cmp_eq_i32 s8, 0 // (op 711) 
s_cselect_b32 s9, 0, -1 // (op 711) 

s_and_b32 s8, s13, 31 // (op 711) 
s_lshl_b32 s18, 1, s8 // (op 711) 
s_add_i32 s19, s18, s9 // (op 711) 


s_and_b32 s9, s0, s19 // (op 711) 


s_add_i32 s0, s1, s9 // (op 711) 


s_ashr_i32 s1, s0, s8 // (op 711) 


s_lshl_b32 s0, s13, 24 // (op 711) 
s_ashr_i32 s8, s0, 31 // (op 711) 

s_xor_b32 s0, s1, s8 // (op 711) 

s_sub_i32 s1, s0, s8 // (op 711) 


s_mul_hi_i32 s0, s1, s14 // (op 711) 
s_add_i32 s8, s0, s1 // (op 711) 

s_ashr_i32 s0, s8, 31 // (op 711) 
s_cmp_eq_i32 s14, 0 // (op 711) 
s_mov_b32 s9, scc // (op 711) 
s_cmp_eq_i32 s9, 0 // (op 711) 
s_cselect_b32 s18, 0, -1 // (op 711) 

s_and_b32 s9, s15, 31 // (op 711) 
s_lshl_b32 s19, 1, s9 // (op 711) 
s_add_i32 s49, s19, s18 // (op 711) 


s_and_b32 s18, s0, s49 // (op 711) 


s_add_i32 s0, s8, s18 // (op 711) 


s_ashr_i32 s8, s0, s9 // (op 711) 


s_lshl_b32 s0, s15, 24 // (op 711) 
s_ashr_i32 s9, s0, 31 // (op 711) 

s_xor_b32 s0, s8, s9 // (op 711) 

s_sub_i32 s8, s0, s9 // (op 711) 


s_add_u32 s18, s4, 16 // (op 711) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 711) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 711) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 711) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 711) 

s_mul_i32 s0, s8, s18 // (op 711) 

s_sub_i32 s8, s1, s0 // (op 711) 


s_mul_hi_i32 s9, s16, s11 // (op 711) 
s_add_i32 s0, s9, s16 // (op 711) 

s_ashr_i32 s1, s0, 31 // (op 711) 
s_cmp_eq_i32 s11, 0 // (op 711) 
s_mov_b32 s9, scc // (op 711) 
s_cmp_eq_i32 s9, 0 // (op 711) 
s_cselect_b32 s49, 0, -1 // (op 711) 

s_and_b32 s9, s46, 31 // (op 711) 
s_lshl_b32 s50, 1, s9 // (op 711) 
s_add_i32 s51, s50, s49 // (op 711) 


s_and_b32 s49, s1, s51 // (op 711) 


s_add_i32 s1, s0, s49 // (op 711) 


s_ashr_i32 s0, s1, s9 // (op 711) 


s_lshl_b32 s1, s46, 24 // (op 711) 
s_ashr_i32 s9, s1, 31 // (op 711) 

s_xor_b32 s1, s0, s9 // (op 711) 

s_sub_i32 s0, s1, s9 // (op 711) 


s_mul_i32 s1, s45, s18 // (op 711) 

s_mul_i32 s9, s10, s1 // (op 711) 

s_mul_i32 s1, s0, s9 // (op 711) 

s_sub_u32 s9, s16, s1 // (op 711) 

s_mul_hi_i32 s1, s9, s44 // (op 711) 
s_add_i32 s18, s1, s9 // (op 711) 


s_ashr_i32 s1, s18, 31 // (op 711) 
s_cmp_eq_i32 s44, 0 // (op 711) 
s_mov_b32 s9, scc // (op 711) 
s_cmp_eq_i32 s9, 0 // (op 711) 
s_cselect_b32 s19, 0, -1 // (op 711) 

s_and_b32 s9, s48, 31 // (op 711) 
s_lshl_b32 s49, 1, s9 // (op 711) 
s_add_i32 s50, s49, s19 // (op 711) 


s_and_b32 s19, s1, s50 // (op 711) 


s_add_i32 s1, s18, s19 // (op 711) 


s_ashr_i32 s18, s1, s9 // (op 711) 


s_lshl_b32 s1, s48, 24 // (op 711) 
s_ashr_i32 s9, s1, 31 // (op 711) 

s_xor_b32 s1, s18, s9 // (op 711) 

s_sub_i32 s18, s1, s9 // (op 711) 


s_cmp_eq_i32 s0, 0 // (op 711) 
s_mov_b32 s1, scc // (op 711) 

s_cmp_eq_i32 s1, 0 // (op 711) 
s_cselect_b32 s0, s18, s8 // (op 711) 



v_lshl_add_u32 v18, s0, 4, v22 // (op 711) 


v_mul_lo_u32 v22, v18, s30 // (op 711) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 711) 
s_lshr_b32 s8, s0, 1 // (op 711) 

v_lshl_add_u32 v18, s8, 3, v19 // (op 711) 


v_add_u32 v19, v18, v22 // (op 711) 


s_mul_i32 s1, s30, s5 // (op 713) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 713) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 713) least significant
s_mul_hi_u32 s9, s30, s4 // (op 713) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 713) most significant: sum
s_add_u32 s1, s1, s9 // (op 713) most significant: sum


s_mov_b32 s64, s32 // (op 713) reg expression
s_mov_b32 s65, s33 // (op 713) reg expression
s_mov_b32 s66, s0 // (op 713) reg expression
s_mov_b32 s67, 131072 // (op 713) call()

v_and_b32 v18, -8, v1 // (op 804) 
v_sub_u32 v22, v1, v18 // (op 804) 

v_lshlrev_b32 v18, 4, v22 // (op 804) 

v_lshrrev_b32 v22, 3, v1 // (op 804) 
v_lshl_add_u32 v24, v22, 7, v18 // (op 804) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 804) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 856) 
v_sub_u32 v22, v1, v18 // (op 856) 

v_lshlrev_b32 v18, 4, v22 // (op 856) 

v_lshrrev_b32 v22, 3, v1 // (op 856) 
v_lshl_add_u32 v24, v22, 7, v18 // (op 856) 


s_nop 0
v_readfirstlane_b32 s1, v24 // (op 856) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -4, v1 // (op 908) 
v_sub_u32 v22, v1, v18 // (op 908) 

v_and_b32 v18, 4294967292, v1 // (op 908) 
v_add_u32 v24, v18, v22 // (op 908) 


v_and_b32 v18, -4, v1 // (op 961) 
v_sub_u32 v22, v1, v18 // (op 961) 

v_and_b32 v18, 4294967292, v1 // (op 961) 
v_add_u32 v25, v18, v22 // (op 961) 


s_ashr_i64 s[8:9], s[34:35], 7 // (op 176) 
s_and_b32 s18, s8, 4294967294 // (op 176) 

s_ashr_i64 s[8:9], s[34:35], 7 // (op 175) 
s_mov_b32 s19, s8 // (op 175) convert

s_cmp_lt_u32 s18, s19 // (op 175) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_175 // (op 175) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_175 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_175:
s_barrier  // (op 1197) 
s_add_u32 s18, s18, 1 // (op 177) 

s_lshl_b64 s[8:9], s[2:3], 3 // (op 468) 
v_mov_b64 v[12:13], s[8:9] // (op 468) convert


v_mov_b64 v[22:23], v[14:15] // (op 468) 
v_mov_b64 v[26:27], v[22:23] // (op 468) 
s_add_u32 s19, 2048, s0 // (op 468) 
s_mov_b32 m0, s19 // (op 468) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 468) Load value direct to lds
v_add_co_u32 v26, s[8:9], v26, v12 // (op 468) least significant half
v_addc_co_u32 v27, s[8:9], 0, v27, s[8:9] // (op 468) most significant half

s_add_u32 m0, m0, 1024 // (op 468) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 468) Load value direct to lds




v_add_co_u32 v14, vcc, 128, v14 // (op 506) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 506) most significant half
s_barrier  // (op 1213) 

s_lshl_b64 s[8:9], s[22:23], 3 // (op 474) 
v_mov_b64 v[12:13], s[8:9] // (op 474) convert


v_mov_b64 v[22:23], v[16:17] // (op 474) 
v_mov_b64 v[26:27], v[22:23] // (op 474) 
s_mov_b32 m0, s1 // (op 474) reg expression
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 474) Load value direct to lds
v_add_co_u32 v26, s[8:9], v26, v12 // (op 474) least significant half
v_addc_co_u32 v27, s[8:9], 0, v27, s[8:9] // (op 474) most significant half

s_add_u32 m0, m0, 1024 // (op 474) 
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 474) Load value direct to lds



v_add_co_u32 v16, vcc, 128, v16 // (op 575) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 575) most significant half


v_mov_b32 v18, v21 // (op 195) 

buffer_load_ubyte v20, v18, s[60:63], 0 offen offset: 0 // (op 195) Load value

v_add_u32 v21, 4, v21 // (op 639) 


s_barrier  // (op 1230) 
v_mov_b32 v18, v19 // (op 196) 




buffer_load_ubyte v12, v18, s[64:67], 0 offen offset: 0 // (op 196) Load value

v_add_u32 v19, 4, v19 // (op 703) 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v20 offset:8256 // (op 192) Store local data 

s_barrier  // (op 1203) 



s_barrier  // (op 1247) 
ds_read_b128 v[26:29], v3 offset:2048 // (op 183) Load local data 



ds_read_b128 v[30:33], v3 offset:2112 // (op 183) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v25, v12 offset:8192 // (op 193) Store local data 

s_barrier  // (op 1220) 



ds_read_b128 v[34:37], v5 // (op 185) Load local data 
ds_read_b128 v[38:41], v5 offset:64 // (op 185) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1237) 



ds_read_u8 v18, v0 offset:8256 // (op 187) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1254) 



ds_read_u8 v20, v6 offset:8192 // (op 189) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v18, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 190) 




s_ashr_i64 s[8:9], s[34:35], 7 // (op 175) 
s_mov_b32 s19, s8 // (op 175) convert

s_cmp_lt_u32 s18, s19 // (op 175) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_175 // (op 175) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopTop_KLoopTail_175 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_175:






















s_barrier  // (op 1321) 








v_cmp_eq_f32 vcc, 0.00000, v8 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_135
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v11, v4 // (op 50) 
v_mul_f32 v13, v11, v5 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 

v_cvt_f16_f32 v4, v12 // (op 56) 
v_cvt_f16_f32 v5, v13 // (op 56) 
v_cvt_f16_f32 v6, v14 // (op 56) 
v_cvt_f16_f32 v7, v15 // (op 56) 

s_mul_hi_i32 s0, s16, s12 // (op 1003) 
v_lshrrev_b32 v0, 6, v1 // (op 1003) 
s_add_i32 s1, s0, s16 // (op 1003) 

v_and_b32 v3, 63, v1 // (op 1003) 
s_ashr_i32 s0, s1, 31 // (op 1003) 
v_lshrrev_b32 v9, 4, v3 // (op 1003) 
s_cmp_eq_i32 s12, 0 // (op 1003) 
s_mov_b32 s2, scc // (op 1003) 
v_lshlrev_b32 v10, 2, v9 // (op 1003) 

s_cmp_eq_i32 s2, 0 // (op 1003) 
s_cselect_b32 s3, 0, -1 // (op 1003) 

v_lshl_add_u32 v9, v0, 4, v10 // (op 1003) 


s_and_b32 s2, s13, 31 // (op 1003) 
v_lshrrev_b32 v0, 2, v3 // (op 1003) 

s_lshl_b32 s8, 1, s2 // (op 1003) 
v_and_b32 v3, 3, v0 // (op 1003) 

s_add_i32 s9, s8, s3 // (op 1003) 


v_and_b32 v0, 3, v1 // (op 1003) 
s_and_b32 s3, s0, s9 // (op 1003) 


v_lshl_add_u32 v10, v3, 2, v0 // (op 1003) 


s_add_i32 s0, s1, s3 // (op 1003) 


s_ashr_i32 s1, s0, s2 // (op 1003) 


s_lshl_b32 s0, s13, 24 // (op 1003) 
s_ashr_i32 s2, s0, 31 // (op 1003) 

s_xor_b32 s0, s1, s2 // (op 1003) 

s_sub_i32 s1, s0, s2 // (op 1003) 


s_mul_hi_i32 s0, s1, s14 // (op 1003) 
s_add_i32 s2, s0, s1 // (op 1003) 

s_ashr_i32 s0, s2, 31 // (op 1003) 
s_cmp_eq_i32 s14, 0 // (op 1003) 
s_mov_b32 s3, scc // (op 1003) 
s_cmp_eq_i32 s3, 0 // (op 1003) 
s_cselect_b32 s8, 0, -1 // (op 1003) 

s_and_b32 s3, s15, 31 // (op 1003) 
s_lshl_b32 s9, 1, s3 // (op 1003) 
s_add_i32 s18, s9, s8 // (op 1003) 


s_and_b32 s8, s0, s18 // (op 1003) 


s_add_i32 s0, s2, s8 // (op 1003) 


s_ashr_i32 s2, s0, s3 // (op 1003) 


s_lshl_b32 s0, s15, 24 // (op 1003) 
s_ashr_i32 s3, s0, 31 // (op 1003) 

s_xor_b32 s0, s2, s3 // (op 1003) 

s_sub_i32 s2, s0, s3 // (op 1003) 


s_mul_i32 s0, s1, s45 // (op 1003) 
s_sub_u32 s3, s16, s0 // (op 1003) 

s_mul_i32 s0, s2, s45 // (op 1003) 
s_add_u32 s0, s0, s3 // (op 1003) 

s_mul_hi_i32 s3, s16, s11 // (op 1003) 
s_add_i32 s8, s3, s16 // (op 1003) 

s_ashr_i32 s3, s8, 31 // (op 1003) 
s_cmp_eq_i32 s11, 0 // (op 1003) 
s_mov_b32 s9, scc // (op 1003) 
s_cmp_eq_i32 s9, 0 // (op 1003) 
s_cselect_b32 s18, 0, -1 // (op 1003) 

s_and_b32 s9, s46, 31 // (op 1003) 
s_lshl_b32 s19, 1, s9 // (op 1003) 
s_add_i32 s20, s19, s18 // (op 1003) 


s_and_b32 s18, s3, s20 // (op 1003) 


s_add_i32 s3, s8, s18 // (op 1003) 


s_ashr_i32 s8, s3, s9 // (op 1003) 


s_lshl_b32 s3, s46, 24 // (op 1003) 
s_ashr_i32 s9, s3, 31 // (op 1003) 

s_xor_b32 s3, s8, s9 // (op 1003) 

s_sub_i32 s8, s3, s9 // (op 1003) 


s_add_u32 s18, s4, 16 // (op 1003) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 1003) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 1003) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 1003) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 1003) 

s_mul_i32 s3, s45, s18 // (op 1003) 
s_mul_i32 s9, s10, s3 // (op 1003) 

s_mul_i32 s3, s8, s9 // (op 1003) 

s_sub_u32 s9, s16, s3 // (op 1003) 

s_mov_b32 s3, s9 // (op 1003) 
s_mul_hi_i32 s20, s3, s44 // (op 1003) 
s_add_i32 s21, s20, s3 // (op 1003) 


s_ashr_i32 s3, s21, 31 // (op 1003) 
s_cmp_eq_i32 s44, 0 // (op 1003) 
s_mov_b32 s20, scc // (op 1003) 
s_cmp_eq_i32 s20, 0 // (op 1003) 
s_cselect_b32 s22, 0, -1 // (op 1003) 

s_and_b32 s20, s48, 31 // (op 1003) 
s_lshl_b32 s23, 1, s20 // (op 1003) 
s_add_i32 s24, s23, s22 // (op 1003) 


s_and_b32 s22, s3, s24 // (op 1003) 


s_add_i32 s3, s21, s22 // (op 1003) 


s_ashr_i32 s21, s3, s20 // (op 1003) 


s_lshl_b32 s3, s48, 24 // (op 1003) 
s_ashr_i32 s20, s3, 31 // (op 1003) 

s_xor_b32 s3, s21, s20 // (op 1003) 

s_sub_i32 s21, s3, s20 // (op 1003) 


s_mul_i32 s3, s47, s45 // (op 1003) 
s_add_u32 s22, s6, 16 // (op 1003) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1003) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1003) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1003) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1003) 

s_sub_i32 s20, s22, s3 // (op 1003) 

s_mul_i32 s22, s21, s20 // (op 1003) 

s_sub_u32 s20, s9, s22 // (op 1003) 


s_add_u32 s9, s20, s3 // (op 1003) 


s_cmp_eq_i32 s8, 0 // (op 1003) 
s_mov_b32 s3, scc // (op 1003) 

s_cmp_eq_i32 s3, 0 // (op 1003) 
s_cselect_b32 s8, s9, s0 // (op 1003) 


s_mul_i32 s0, s2, s18 // (op 1003) 


v_lshl_add_u32 v0, s8, 4, v9 // (op 1003) 


s_sub_i32 s2, s1, s0 // (op 1003) 


s_cmp_eq_i32 s3, 0 // (op 1003) 
s_cselect_b32 s0, s21, s2 // (op 1003) 



v_lshl_add_u32 v3, s0, 4, v10 // (op 1003) 


v_mul_lo_u32 v9, v3, s36 // (op 1003) 

v_add_lshl_u32 v3, v0, v9, 1 // (op 1003) 


s_mul_i32 s9, s36, s5 // (op 1005) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 1005) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 1005) least significant
s_mul_hi_u32 s1, s36, s4 // (op 1005) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 1005) most significant: sum
s_add_u32 s9, s9, s1 // (op 1005) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 1005) 

s_mov_b32 s32, s38 // (op 1005) reg expression
s_mov_b32 s33, s39 // (op 1005) reg expression
s_mov_b32 s34, s0 // (op 1005) reg expression
s_mov_b32 s35, 131072 // (op 1005) call()


v_pack_b32_f16 v12, v4, v5 // (op 58) 
v_pack_b32_f16 v13, v6, v7 // (op 58) 


buffer_store_dwordx2 v[12:13], v3, s[32:35], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_135
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalFalse_0_beta_135:
s_mul_hi_i32 s0, s16, s12 // (op 774) 
v_lshrrev_b32 v0, 6, v1 // (op 774) 
s_add_i32 s1, s0, s16 // (op 774) 

v_and_b32 v3, 63, v1 // (op 774) 
s_ashr_i32 s0, s1, 31 // (op 774) 
v_lshrrev_b32 v4, 4, v3 // (op 774) 
s_cmp_eq_i32 s12, 0 // (op 774) 
s_mov_b32 s2, scc // (op 774) 
v_lshlrev_b32 v5, 2, v4 // (op 774) 

s_cmp_eq_i32 s2, 0 // (op 774) 
s_cselect_b32 s3, 0, -1 // (op 774) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 774) 


s_and_b32 s2, s13, 31 // (op 774) 
v_lshrrev_b32 v0, 2, v3 // (op 774) 

s_lshl_b32 s8, 1, s2 // (op 774) 
v_and_b32 v3, 3, v0 // (op 774) 

s_add_i32 s9, s8, s3 // (op 774) 


v_and_b32 v0, 3, v1 // (op 774) 
s_and_b32 s3, s0, s9 // (op 774) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 774) 


s_add_i32 s0, s1, s3 // (op 774) 


s_ashr_i32 s1, s0, s2 // (op 774) 


s_lshl_b32 s0, s13, 24 // (op 774) 
s_ashr_i32 s2, s0, 31 // (op 774) 

s_xor_b32 s0, s1, s2 // (op 774) 

s_sub_i32 s1, s0, s2 // (op 774) 


s_mul_hi_i32 s0, s1, s14 // (op 774) 
s_add_i32 s2, s0, s1 // (op 774) 

s_ashr_i32 s0, s2, 31 // (op 774) 
s_cmp_eq_i32 s14, 0 // (op 774) 
s_mov_b32 s3, scc // (op 774) 
s_cmp_eq_i32 s3, 0 // (op 774) 
s_cselect_b32 s8, 0, -1 // (op 774) 

s_and_b32 s3, s15, 31 // (op 774) 
s_lshl_b32 s9, 1, s3 // (op 774) 
s_add_i32 s18, s9, s8 // (op 774) 


s_and_b32 s8, s0, s18 // (op 774) 


s_add_i32 s0, s2, s8 // (op 774) 


s_ashr_i32 s2, s0, s3 // (op 774) 


s_lshl_b32 s0, s15, 24 // (op 774) 
s_ashr_i32 s3, s0, 31 // (op 774) 

s_xor_b32 s0, s2, s3 // (op 774) 

s_sub_i32 s2, s0, s3 // (op 774) 


s_mul_i32 s0, s1, s45 // (op 774) 
s_sub_u32 s3, s16, s0 // (op 774) 

s_mul_i32 s0, s2, s45 // (op 774) 
s_add_u32 s0, s0, s3 // (op 774) 

s_mul_hi_i32 s3, s16, s11 // (op 774) 
s_add_i32 s8, s3, s16 // (op 774) 

s_ashr_i32 s3, s8, 31 // (op 774) 
s_cmp_eq_i32 s11, 0 // (op 774) 
s_mov_b32 s9, scc // (op 774) 
s_cmp_eq_i32 s9, 0 // (op 774) 
s_cselect_b32 s18, 0, -1 // (op 774) 

s_and_b32 s9, s46, 31 // (op 774) 
s_lshl_b32 s19, 1, s9 // (op 774) 
s_add_i32 s20, s19, s18 // (op 774) 


s_and_b32 s18, s3, s20 // (op 774) 


s_add_i32 s3, s8, s18 // (op 774) 


s_ashr_i32 s8, s3, s9 // (op 774) 


s_lshl_b32 s3, s46, 24 // (op 774) 
s_ashr_i32 s9, s3, 31 // (op 774) 

s_xor_b32 s3, s8, s9 // (op 774) 

s_sub_i32 s8, s3, s9 // (op 774) 


s_add_u32 s18, s4, 16 // (op 774) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 774) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 774) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 774) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 774) 

s_mul_i32 s3, s45, s18 // (op 774) 
s_mul_i32 s9, s10, s3 // (op 774) 

s_mul_i32 s3, s8, s9 // (op 774) 

s_sub_u32 s9, s16, s3 // (op 774) 

s_mov_b32 s3, s9 // (op 774) 
s_mul_hi_i32 s20, s3, s44 // (op 774) 
s_add_i32 s21, s20, s3 // (op 774) 


s_ashr_i32 s3, s21, 31 // (op 774) 
s_cmp_eq_i32 s44, 0 // (op 774) 
s_mov_b32 s20, scc // (op 774) 
s_cmp_eq_i32 s20, 0 // (op 774) 
s_cselect_b32 s22, 0, -1 // (op 774) 

s_and_b32 s20, s48, 31 // (op 774) 
s_lshl_b32 s23, 1, s20 // (op 774) 
s_add_i32 s24, s23, s22 // (op 774) 


s_and_b32 s22, s3, s24 // (op 774) 


s_add_i32 s3, s21, s22 // (op 774) 


s_ashr_i32 s21, s3, s20 // (op 774) 


s_lshl_b32 s3, s48, 24 // (op 774) 
s_ashr_i32 s20, s3, 31 // (op 774) 

s_xor_b32 s3, s21, s20 // (op 774) 

s_sub_i32 s21, s3, s20 // (op 774) 


s_mul_i32 s3, s47, s45 // (op 774) 
s_add_u32 s22, s6, 16 // (op 774) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 774) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 774) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 774) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 774) 

s_sub_i32 s20, s22, s3 // (op 774) 

s_mul_i32 s22, s21, s20 // (op 774) 

s_sub_u32 s20, s9, s22 // (op 774) 


s_add_u32 s9, s20, s3 // (op 774) 


s_cmp_eq_i32 s8, 0 // (op 774) 
s_mov_b32 s3, scc // (op 774) 

s_cmp_eq_i32 s3, 0 // (op 774) 
s_cselect_b32 s8, s9, s0 // (op 774) 


s_mul_i32 s0, s2, s18 // (op 774) 


v_lshl_add_u32 v0, s8, 4, v4 // (op 774) 


s_sub_i32 s2, s1, s0 // (op 774) 


s_cmp_eq_i32 s3, 0 // (op 774) 
s_cselect_b32 s0, s21, s2 // (op 774) 



v_lshl_add_u32 v3, s0, 4, v5 // (op 774) 


v_mul_lo_u32 v4, v3, s40 // (op 774) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 774) 


s_mul_i32 s9, s40, s5 // (op 776) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 776) most significant: low of high * low
s_mul_i32 s8, s40, s4 // (op 776) least significant
s_mul_hi_u32 s1, s40, s4 // (op 776) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 776) most significant: sum
s_add_u32 s9, s9, s1 // (op 776) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 776) 

s_mov_b32 s28, s42 // (op 776) reg expression
s_mov_b32 s29, s43 // (op 776) reg expression
s_mov_b32 s30, s0 // (op 776) reg expression
s_mov_b32 s31, 131072 // (op 776) call()





buffer_load_dwordx2 v[4:5], v3, s[28:31], 0 offen // (op 148) Load value



v_accvgpr_read v12, a0 // (op 144) 
v_accvgpr_read v13, a1 // (op 144) 
v_accvgpr_read v14, a2 // (op 144) 
v_accvgpr_read v15, a3 // (op 144) 
v_mul_f32 v16, v11, v12 // (op 144) 
v_mul_f32 v17, v11, v13 // (op 144) 
v_mul_f32 v18, v11, v14 // (op 144) 
v_mul_f32 v19, v11, v15 // (op 144) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 149) 
v_lshrrev_b32 v7, 16, v4 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v12, v8, v6 // (op 149) 
v_mul_f32 v13, v8, v7 // (op 149) 
v_cvt_f32_f16 v6, v5 // (op 149) 
v_lshrrev_b32 v7, 16, v5 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v14, v8, v6 // (op 149) 
v_mul_f32 v15, v8, v7 // (op 149) 


v_add_f32 v4, v12, v16 // (op 145) 
v_add_f32 v5, v13, v17 // (op 145) 
v_add_f32 v6, v14, v18 // (op 145) 
v_add_f32 v7, v15, v19 // (op 145) 


v_cvt_f16_f32 v12, v4 // (op 146) 
v_cvt_f16_f32 v13, v5 // (op 146) 
v_cvt_f16_f32 v14, v6 // (op 146) 
v_cvt_f16_f32 v15, v7 // (op 146) 

s_mul_hi_i32 s0, s16, s12 // (op 1020) 
v_lshrrev_b32 v0, 6, v1 // (op 1020) 
s_add_i32 s1, s0, s16 // (op 1020) 

v_and_b32 v3, 63, v1 // (op 1020) 
s_ashr_i32 s0, s1, 31 // (op 1020) 
v_lshrrev_b32 v4, 4, v3 // (op 1020) 
s_cmp_eq_i32 s12, 0 // (op 1020) 
s_mov_b32 s2, scc // (op 1020) 
v_lshlrev_b32 v5, 2, v4 // (op 1020) 

s_cmp_eq_i32 s2, 0 // (op 1020) 
s_cselect_b32 s3, 0, -1 // (op 1020) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1020) 


s_and_b32 s2, s13, 31 // (op 1020) 
v_lshrrev_b32 v0, 2, v3 // (op 1020) 

s_lshl_b32 s8, 1, s2 // (op 1020) 
v_and_b32 v3, 3, v0 // (op 1020) 

s_add_i32 s9, s8, s3 // (op 1020) 


v_and_b32 v0, 3, v1 // (op 1020) 
s_and_b32 s3, s0, s9 // (op 1020) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 1020) 


s_add_i32 s0, s1, s3 // (op 1020) 


s_ashr_i32 s1, s0, s2 // (op 1020) 


s_lshl_b32 s0, s13, 24 // (op 1020) 
s_ashr_i32 s2, s0, 31 // (op 1020) 

s_xor_b32 s0, s1, s2 // (op 1020) 

s_sub_i32 s1, s0, s2 // (op 1020) 


s_mul_hi_i32 s0, s1, s14 // (op 1020) 
s_add_i32 s2, s0, s1 // (op 1020) 

s_ashr_i32 s0, s2, 31 // (op 1020) 
s_cmp_eq_i32 s14, 0 // (op 1020) 
s_mov_b32 s3, scc // (op 1020) 
s_cmp_eq_i32 s3, 0 // (op 1020) 
s_cselect_b32 s8, 0, -1 // (op 1020) 

s_and_b32 s3, s15, 31 // (op 1020) 
s_lshl_b32 s9, 1, s3 // (op 1020) 
s_add_i32 s18, s9, s8 // (op 1020) 


s_and_b32 s8, s0, s18 // (op 1020) 


s_add_i32 s0, s2, s8 // (op 1020) 


s_ashr_i32 s2, s0, s3 // (op 1020) 


s_lshl_b32 s0, s15, 24 // (op 1020) 
s_ashr_i32 s3, s0, 31 // (op 1020) 

s_xor_b32 s0, s2, s3 // (op 1020) 

s_sub_i32 s2, s0, s3 // (op 1020) 


s_mul_i32 s0, s1, s45 // (op 1020) 
s_sub_u32 s3, s16, s0 // (op 1020) 

s_mul_i32 s0, s2, s45 // (op 1020) 
s_add_u32 s0, s0, s3 // (op 1020) 

s_mul_hi_i32 s3, s16, s11 // (op 1020) 
s_add_i32 s8, s3, s16 // (op 1020) 

s_ashr_i32 s3, s8, 31 // (op 1020) 
s_cmp_eq_i32 s11, 0 // (op 1020) 
s_mov_b32 s9, scc // (op 1020) 
s_cmp_eq_i32 s9, 0 // (op 1020) 
s_cselect_b32 s18, 0, -1 // (op 1020) 

s_and_b32 s9, s46, 31 // (op 1020) 
s_lshl_b32 s19, 1, s9 // (op 1020) 
s_add_i32 s20, s19, s18 // (op 1020) 


s_and_b32 s18, s3, s20 // (op 1020) 


s_add_i32 s3, s8, s18 // (op 1020) 


s_ashr_i32 s8, s3, s9 // (op 1020) 


s_lshl_b32 s3, s46, 24 // (op 1020) 
s_ashr_i32 s9, s3, 31 // (op 1020) 

s_xor_b32 s3, s8, s9 // (op 1020) 

s_sub_i32 s8, s3, s9 // (op 1020) 


s_add_u32 s18, s4, 16 // (op 1020) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 1020) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 1020) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 1020) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 1020) 

s_mul_i32 s3, s45, s18 // (op 1020) 
s_mul_i32 s9, s10, s3 // (op 1020) 

s_mul_i32 s3, s8, s9 // (op 1020) 

s_sub_u32 s9, s16, s3 // (op 1020) 

s_mov_b32 s3, s9 // (op 1020) 
s_mul_hi_i32 s20, s3, s44 // (op 1020) 
s_add_i32 s21, s20, s3 // (op 1020) 


s_ashr_i32 s3, s21, 31 // (op 1020) 
s_cmp_eq_i32 s44, 0 // (op 1020) 
s_mov_b32 s20, scc // (op 1020) 
s_cmp_eq_i32 s20, 0 // (op 1020) 
s_cselect_b32 s22, 0, -1 // (op 1020) 

s_and_b32 s20, s48, 31 // (op 1020) 
s_lshl_b32 s23, 1, s20 // (op 1020) 
s_add_i32 s24, s23, s22 // (op 1020) 


s_and_b32 s22, s3, s24 // (op 1020) 


s_add_i32 s3, s21, s22 // (op 1020) 


s_ashr_i32 s21, s3, s20 // (op 1020) 


s_lshl_b32 s3, s48, 24 // (op 1020) 
s_ashr_i32 s20, s3, 31 // (op 1020) 

s_xor_b32 s3, s21, s20 // (op 1020) 

s_sub_i32 s21, s3, s20 // (op 1020) 


s_mul_i32 s3, s47, s45 // (op 1020) 
s_add_u32 s22, s6, 16 // (op 1020) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1020) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1020) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1020) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1020) 

s_sub_i32 s20, s22, s3 // (op 1020) 

s_mul_i32 s22, s21, s20 // (op 1020) 

s_sub_u32 s20, s9, s22 // (op 1020) 


s_add_u32 s9, s20, s3 // (op 1020) 


s_cmp_eq_i32 s8, 0 // (op 1020) 
s_mov_b32 s3, scc // (op 1020) 

s_cmp_eq_i32 s3, 0 // (op 1020) 
s_cselect_b32 s8, s9, s0 // (op 1020) 


s_mul_i32 s0, s2, s18 // (op 1020) 


v_lshl_add_u32 v0, s8, 4, v4 // (op 1020) 


s_sub_i32 s2, s1, s0 // (op 1020) 


s_cmp_eq_i32 s3, 0 // (op 1020) 
s_cselect_b32 s0, s21, s2 // (op 1020) 



v_lshl_add_u32 v3, s0, 4, v5 // (op 1020) 


v_mul_lo_u32 v4, v3, s36 // (op 1020) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 1020) 


s_mul_i32 s9, s36, s5 // (op 1022) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 1022) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 1022) least significant
s_mul_hi_u32 s1, s36, s4 // (op 1022) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 1022) most significant: sum
s_add_u32 s9, s9, s1 // (op 1022) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 1022) 

s_mov_b32 s32, s38 // (op 1022) reg expression
s_mov_b32 s33, s39 // (op 1022) reg expression
s_mov_b32 s34, s0 // (op 1022) reg expression
s_mov_b32 s35, 131072 // (op 1022) call()


v_pack_b32_f16 v4, v12, v13 // (op 147) 
v_pack_b32_f16 v5, v14, v15 // (op 147) 


buffer_store_dwordx2 v[4:5], v3, s[32:35], 0 offen // (op 147) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__ConditionalBottom_0_beta_135:
















s_endpgm  // End of RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.LRR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end:
.size RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_, .LRR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM__end-RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 60
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 8448 // lds bytes
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
  ProblemType:
    OperationType: GEMM
    DataType: B8
    DataTypeA: B8
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
    macrotile: [16, 16, 128]
    threads: [64, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 8448
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 70
    .vgpr_count: 54
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .workgroup_size: [64, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, {is-null: true}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 24
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 28
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 32
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 36
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 40
        .expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 44
        .expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 48
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 128
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 136
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 144
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: MagicMultiple_30
        .size: 4
        .offset: 152
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 156
        .expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 160
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 164
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 168
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

