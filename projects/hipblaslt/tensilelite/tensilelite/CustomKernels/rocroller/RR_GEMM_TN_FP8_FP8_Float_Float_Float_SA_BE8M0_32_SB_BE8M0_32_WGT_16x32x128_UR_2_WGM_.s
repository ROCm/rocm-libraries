

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 795) 
v_and_b32 v3, 15, v1 // (op 795) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 795) 


v_and_b32 v0, 48, v1 // (op 795) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 795) 


v_lshrrev_b32 v0, 6, v1 // (op 847) 
v_and_b32 v4, 1, v0 // (op 847) 

v_and_b32 v0, 15, v1 // (op 847) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 847) 


v_and_b32 v0, 48, v1 // (op 847) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 847) 


v_lshrrev_b32 v0, 6, v1 // (op 900) 
v_and_b32 v5, 1, v0 // (op 900) 

v_and_b32 v0, 15, v1 // (op 900) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 900) 


v_and_b32 v0, 63, v1 // (op 900) 
v_lshrrev_b32 v5, 4, v0 // (op 900) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 900) 


v_lshrrev_b32 v5, 7, v1 // (op 973) 
v_and_b32 v6, 15, v1 // (op 973) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 973) 


v_and_b32 v5, 48, v1 // (op 973) 
v_lshl_add_u32 v6, v7, 7, v5 // (op 973) 


v_lshrrev_b32 v5, 6, v1 // (op 1012) 
v_and_b32 v7, 1, v5 // (op 1012) 

v_and_b32 v5, 15, v1 // (op 1012) 
v_lshl_add_u32 v8, v7, 4, v5 // (op 1012) 


v_and_b32 v5, 48, v1 // (op 1012) 
v_lshl_add_u32 v7, v8, 7, v5 // (op 1012) 


v_lshrrev_b32 v5, 6, v1 // (op 1036) 
v_and_b32 v8, 1, v5 // (op 1036) 

v_and_b32 v5, 15, v1 // (op 1036) 
v_lshl_add_u32 v9, v8, 4, v5 // (op 1036) 


v_and_b32 v5, 63, v1 // (op 1036) 
v_lshrrev_b32 v8, 4, v5 // (op 1036) 

v_lshl_add_u32 v5, v9, 2, v8 // (op 1036) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v9, s9 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
s_mul_hi_i32 s0, s16, s12 // (op 506) 
v_lshrrev_b32 v10, 3, v1 // (op 506) 
s_add_i32 s1, s0, s16 // (op 506) 

v_and_b32 v11, -8, v1 // (op 506) 
s_ashr_i32 s0, s1, 31 // (op 506) 
v_sub_u32 v12, v1, v11 // (op 506) 

s_cmp_eq_i32 s12, 0 // (op 506) 
s_mov_b32 s8, scc // (op 506) 
v_lshlrev_b32 v11, 4, v12 // (op 506) 

s_cmp_eq_i32 s8, 0 // (op 506) 
s_cselect_b32 s9, 0, -1 // (op 506) 

v_mov_b32 v13, 0 // (op 506) convert
v_mov_b32 v12, v11 // (op 506) convert

s_and_b32 s8, s13, 31 // (op 506) 
s_lshl_b32 s18, 1, s8 // (op 506) 
s_add_i32 s19, s18, s9 // (op 506) 


s_and_b32 s9, s0, s19 // (op 506) 


s_add_i32 s0, s1, s9 // (op 506) 


s_ashr_i32 s1, s0, s8 // (op 506) 


s_lshl_b32 s0, s13, 24 // (op 506) 
s_ashr_i32 s8, s0, 31 // (op 506) 

s_xor_b32 s0, s1, s8 // (op 506) 

s_sub_i32 s1, s0, s8 // (op 506) 


s_mul_hi_i32 s0, s1, s14 // (op 506) 
s_add_i32 s8, s0, s1 // (op 506) 

s_ashr_i32 s0, s8, 31 // (op 506) 
s_cmp_eq_i32 s14, 0 // (op 506) 
s_mov_b32 s9, scc // (op 506) 
s_cmp_eq_i32 s9, 0 // (op 506) 
s_cselect_b32 s18, 0, -1 // (op 506) 

s_and_b32 s9, s15, 31 // (op 506) 
s_lshl_b32 s19, 1, s9 // (op 506) 
s_add_i32 s49, s19, s18 // (op 506) 


s_and_b32 s18, s0, s49 // (op 506) 


s_add_i32 s0, s8, s18 // (op 506) 


s_ashr_i32 s8, s0, s9 // (op 506) 


s_lshl_b32 s0, s15, 24 // (op 506) 
s_ashr_i32 s9, s0, 31 // (op 506) 

s_xor_b32 s0, s8, s9 // (op 506) 

s_sub_i32 s8, s0, s9 // (op 506) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s45 // (op 506) 

s_sub_u32 s1, s16, s0 // (op 506) 

s_mul_i32 s0, s8, s45 // (op 506) 
s_add_u32 s0, s0, s1 // (op 506) 


s_mul_hi_i32 s1, s16, s11 // (op 506) 
s_add_i32 s8, s1, s16 // (op 506) 

s_ashr_i32 s1, s8, 31 // (op 506) 
s_cmp_eq_i32 s11, 0 // (op 506) 
s_mov_b32 s9, scc // (op 506) 
s_cmp_eq_i32 s9, 0 // (op 506) 
s_cselect_b32 s18, 0, -1 // (op 506) 

s_and_b32 s9, s46, 31 // (op 506) 
s_lshl_b32 s19, 1, s9 // (op 506) 
s_add_i32 s49, s19, s18 // (op 506) 


s_and_b32 s18, s1, s49 // (op 506) 


s_add_i32 s1, s8, s18 // (op 506) 


s_ashr_i32 s8, s1, s9 // (op 506) 


s_lshl_b32 s1, s46, 24 // (op 506) 
s_ashr_i32 s9, s1, 31 // (op 506) 

s_xor_b32 s1, s8, s9 // (op 506) 

s_sub_i32 s8, s1, s9 // (op 506) 


s_add_u32 s18, s4, 32 // (op 506) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 506) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 506) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 506) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 506) 

s_mul_i32 s1, s45, s18 // (op 506) 

s_mul_i32 s9, s10, s1 // (op 506) 

s_mul_i32 s1, s8, s9 // (op 506) 

s_sub_u32 s9, s16, s1 // (op 506) 

s_mov_b32 s1, s9 // (op 506) 
s_mul_hi_i32 s18, s1, s44 // (op 506) 
s_add_i32 s19, s18, s1 // (op 506) 


s_ashr_i32 s1, s19, 31 // (op 506) 
s_cmp_eq_i32 s44, 0 // (op 506) 
s_mov_b32 s18, scc // (op 506) 
s_cmp_eq_i32 s18, 0 // (op 506) 
s_cselect_b32 s49, 0, -1 // (op 506) 

s_and_b32 s18, s48, 31 // (op 506) 
s_lshl_b32 s50, 1, s18 // (op 506) 
s_add_i32 s51, s50, s49 // (op 506) 


s_and_b32 s49, s1, s51 // (op 506) 


s_add_i32 s1, s19, s49 // (op 506) 


s_ashr_i32 s19, s1, s18 // (op 506) 


s_lshl_b32 s1, s48, 24 // (op 506) 
s_ashr_i32 s18, s1, 31 // (op 506) 

s_xor_b32 s1, s19, s18 // (op 506) 

s_sub_i32 s19, s1, s18 // (op 506) 


s_mul_i32 s1, s47, s45 // (op 506) 
s_add_u32 s50, s6, 16 // (op 506) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 506) End of Int64 add, unlocking SCC
s_sub_u32 s52, s50, 1 // (op 506) least significant half; sets scc
s_subb_u32 s53, s51, 0 // (op 506) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 4 // (op 506) 

s_sub_i32 s18, s50, s1 // (op 506) 

s_mul_i32 s49, s19, s18 // (op 506) 


s_sub_u32 s18, s9, s49 // (op 506) 


s_add_u32 s9, s18, s1 // (op 506) 


s_cmp_eq_i32 s8, 0 // (op 506) 
s_mov_b32 s1, scc // (op 506) 

s_cmp_eq_i32 s1, 0 // (op 506) 
s_cselect_b32 s8, s9, s0 // (op 506) 



v_lshl_add_u32 v11, s8, 4, v10 // (op 506) 


v_mul_lo_u32 v15, v11, s3 // (op 506) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 506) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 506) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 506) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 506) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 506) most significant half



s_mul_i32 s1, s2, s7 // (op 508) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 508) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 508) least significant
s_mul_hi_u32 s9, s2, s6 // (op 508) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 508) most significant: sum
s_add_u32 s1, s1, s9 // (op 508) most significant: sum


s_mov_b32 s52, s20 // (op 508) reg expression
s_mov_b32 s53, s21 // (op 508) reg expression
s_mov_b32 s54, s0 // (op 508) reg expression
s_mov_b32 s55, 131072 // (op 508) call()

s_mul_hi_i32 s0, s16, s12 // (op 525) 
v_lshrrev_b32 v10, 3, v1 // (op 525) 
s_add_i32 s1, s0, s16 // (op 525) 

v_and_b32 v11, -8, v1 // (op 525) 
s_ashr_i32 s0, s1, 31 // (op 525) 
v_sub_u32 v12, v1, v11 // (op 525) 

s_cmp_eq_i32 s12, 0 // (op 525) 
s_mov_b32 s8, scc // (op 525) 
v_lshlrev_b32 v11, 4, v12 // (op 525) 

s_cmp_eq_i32 s8, 0 // (op 525) 
s_cselect_b32 s9, 0, -1 // (op 525) 

v_mov_b32 v13, 0 // (op 525) convert
v_mov_b32 v12, v11 // (op 525) convert

s_and_b32 s8, s13, 31 // (op 525) 
s_lshl_b32 s18, 1, s8 // (op 525) 
s_add_i32 s19, s18, s9 // (op 525) 


s_and_b32 s9, s0, s19 // (op 525) 


s_add_i32 s0, s1, s9 // (op 525) 


s_ashr_i32 s1, s0, s8 // (op 525) 


s_lshl_b32 s0, s13, 24 // (op 525) 
s_ashr_i32 s8, s0, 31 // (op 525) 

s_xor_b32 s0, s1, s8 // (op 525) 

s_sub_i32 s1, s0, s8 // (op 525) 


s_mul_hi_i32 s0, s1, s14 // (op 525) 
s_add_i32 s8, s0, s1 // (op 525) 

s_ashr_i32 s0, s8, 31 // (op 525) 
s_cmp_eq_i32 s14, 0 // (op 525) 
s_mov_b32 s9, scc // (op 525) 
s_cmp_eq_i32 s9, 0 // (op 525) 
s_cselect_b32 s18, 0, -1 // (op 525) 

s_and_b32 s9, s15, 31 // (op 525) 
s_lshl_b32 s19, 1, s9 // (op 525) 
s_add_i32 s49, s19, s18 // (op 525) 


s_and_b32 s18, s0, s49 // (op 525) 


s_add_i32 s0, s8, s18 // (op 525) 


s_ashr_i32 s8, s0, s9 // (op 525) 


s_lshl_b32 s0, s15, 24 // (op 525) 
s_ashr_i32 s9, s0, 31 // (op 525) 

s_xor_b32 s0, s8, s9 // (op 525) 

s_sub_i32 s8, s0, s9 // (op 525) 


s_mul_i32 s0, s1, s45 // (op 525) 

s_sub_u32 s1, s16, s0 // (op 525) 

s_mul_i32 s0, s8, s45 // (op 525) 
s_add_u32 s0, s0, s1 // (op 525) 


s_mul_hi_i32 s1, s16, s11 // (op 525) 
s_add_i32 s8, s1, s16 // (op 525) 

s_ashr_i32 s1, s8, 31 // (op 525) 
s_cmp_eq_i32 s11, 0 // (op 525) 
s_mov_b32 s9, scc // (op 525) 
s_cmp_eq_i32 s9, 0 // (op 525) 
s_cselect_b32 s18, 0, -1 // (op 525) 

s_and_b32 s9, s46, 31 // (op 525) 
s_lshl_b32 s19, 1, s9 // (op 525) 
s_add_i32 s49, s19, s18 // (op 525) 


s_and_b32 s18, s1, s49 // (op 525) 


s_add_i32 s1, s8, s18 // (op 525) 


s_ashr_i32 s8, s1, s9 // (op 525) 


s_lshl_b32 s1, s46, 24 // (op 525) 
s_ashr_i32 s9, s1, 31 // (op 525) 

s_xor_b32 s1, s8, s9 // (op 525) 

s_sub_i32 s8, s1, s9 // (op 525) 


s_add_u32 s18, s4, 32 // (op 525) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 525) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 525) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 525) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 525) 

s_mul_i32 s1, s45, s18 // (op 525) 

s_mul_i32 s9, s10, s1 // (op 525) 

s_mul_i32 s1, s8, s9 // (op 525) 

s_sub_u32 s9, s16, s1 // (op 525) 

s_mov_b32 s1, s9 // (op 525) 
s_mul_hi_i32 s18, s1, s44 // (op 525) 
s_add_i32 s19, s18, s1 // (op 525) 


s_ashr_i32 s1, s19, 31 // (op 525) 
s_cmp_eq_i32 s44, 0 // (op 525) 
s_mov_b32 s18, scc // (op 525) 
s_cmp_eq_i32 s18, 0 // (op 525) 
s_cselect_b32 s49, 0, -1 // (op 525) 

s_and_b32 s18, s48, 31 // (op 525) 
s_lshl_b32 s50, 1, s18 // (op 525) 
s_add_i32 s51, s50, s49 // (op 525) 


s_and_b32 s49, s1, s51 // (op 525) 


s_add_i32 s1, s19, s49 // (op 525) 


s_ashr_i32 s19, s1, s18 // (op 525) 


s_lshl_b32 s1, s48, 24 // (op 525) 
s_ashr_i32 s18, s1, 31 // (op 525) 

s_xor_b32 s1, s19, s18 // (op 525) 

s_sub_i32 s19, s1, s18 // (op 525) 


s_mul_i32 s1, s47, s45 // (op 525) 
s_add_u32 s50, s6, 16 // (op 525) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 525) End of Int64 add, unlocking SCC
s_sub_u32 s56, s50, 1 // (op 525) least significant half; sets scc
s_subb_u32 s57, s51, 0 // (op 525) most significant half; uses scc

s_ashr_i64 s[50:51], s[56:57], 4 // (op 525) 

s_sub_i32 s18, s50, s1 // (op 525) 

s_mul_i32 s49, s19, s18 // (op 525) 


s_sub_u32 s18, s9, s49 // (op 525) 


s_add_u32 s9, s18, s1 // (op 525) 


s_cmp_eq_i32 s8, 0 // (op 525) 
s_mov_b32 s1, scc // (op 525) 

s_cmp_eq_i32 s1, 0 // (op 525) 
s_cselect_b32 s8, s9, s0 // (op 525) 



v_lshl_add_u32 v11, s8, 4, v10 // (op 525) 


v_mul_lo_u32 v17, v11, s3 // (op 525) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 525) most significant: high of low * low
v_mul_lo_u32 v16, v11, s2 // (op 525) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 525) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 525) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 525) most significant half



s_mul_i32 s1, s2, s7 // (op 527) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 527) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 527) least significant
s_mul_hi_u32 s9, s2, s6 // (op 527) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 527) most significant: sum
s_add_u32 s1, s1, s9 // (op 527) most significant: sum


s_mov_b32 s52, s20 // (op 527) reg expression
s_mov_b32 s53, s21 // (op 527) reg expression
s_mov_b32 s54, s0 // (op 527) reg expression
s_mov_b32 s55, 131072 // (op 527) call()

s_mul_hi_i32 s0, s16, s12 // (op 570) 
v_lshrrev_b32 v10, 3, v1 // (op 570) 
s_add_i32 s1, s0, s16 // (op 570) 

v_and_b32 v11, -8, v1 // (op 570) 
s_ashr_i32 s0, s1, 31 // (op 570) 
v_sub_u32 v12, v1, v11 // (op 570) 

s_cmp_eq_i32 s12, 0 // (op 570) 
s_mov_b32 s8, scc // (op 570) 
v_lshlrev_b32 v11, 4, v12 // (op 570) 

s_cmp_eq_i32 s8, 0 // (op 570) 
s_cselect_b32 s9, 0, -1 // (op 570) 

v_mov_b32 v13, 0 // (op 570) convert
v_mov_b32 v12, v11 // (op 570) convert

s_and_b32 s8, s13, 31 // (op 570) 
s_lshl_b32 s18, 1, s8 // (op 570) 
s_add_i32 s19, s18, s9 // (op 570) 


s_and_b32 s9, s0, s19 // (op 570) 


s_add_i32 s0, s1, s9 // (op 570) 


s_ashr_i32 s1, s0, s8 // (op 570) 


s_lshl_b32 s0, s13, 24 // (op 570) 
s_ashr_i32 s8, s0, 31 // (op 570) 

s_xor_b32 s0, s1, s8 // (op 570) 

s_sub_i32 s1, s0, s8 // (op 570) 


s_mul_hi_i32 s0, s1, s14 // (op 570) 
s_add_i32 s8, s0, s1 // (op 570) 

s_ashr_i32 s0, s8, 31 // (op 570) 
s_cmp_eq_i32 s14, 0 // (op 570) 
s_mov_b32 s9, scc // (op 570) 
s_cmp_eq_i32 s9, 0 // (op 570) 
s_cselect_b32 s18, 0, -1 // (op 570) 

s_and_b32 s9, s15, 31 // (op 570) 
s_lshl_b32 s19, 1, s9 // (op 570) 
s_add_i32 s49, s19, s18 // (op 570) 


s_and_b32 s18, s0, s49 // (op 570) 


s_add_i32 s0, s8, s18 // (op 570) 


s_ashr_i32 s8, s0, s9 // (op 570) 


s_lshl_b32 s0, s15, 24 // (op 570) 
s_ashr_i32 s9, s0, 31 // (op 570) 

s_xor_b32 s0, s8, s9 // (op 570) 

s_sub_i32 s8, s0, s9 // (op 570) 


s_add_u32 s18, s4, 32 // (op 570) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 570) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 570) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 570) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 570) 

s_mul_i32 s0, s8, s18 // (op 570) 

s_sub_i32 s8, s1, s0 // (op 570) 


s_mul_hi_i32 s9, s16, s11 // (op 570) 
s_add_i32 s0, s9, s16 // (op 570) 

s_ashr_i32 s1, s0, 31 // (op 570) 
s_cmp_eq_i32 s11, 0 // (op 570) 
s_mov_b32 s9, scc // (op 570) 
s_cmp_eq_i32 s9, 0 // (op 570) 
s_cselect_b32 s49, 0, -1 // (op 570) 

s_and_b32 s9, s46, 31 // (op 570) 
s_lshl_b32 s50, 1, s9 // (op 570) 
s_add_i32 s51, s50, s49 // (op 570) 


s_and_b32 s49, s1, s51 // (op 570) 


s_add_i32 s1, s0, s49 // (op 570) 


s_ashr_i32 s0, s1, s9 // (op 570) 


s_lshl_b32 s1, s46, 24 // (op 570) 
s_ashr_i32 s9, s1, 31 // (op 570) 

s_xor_b32 s1, s0, s9 // (op 570) 

s_sub_i32 s0, s1, s9 // (op 570) 


s_mul_i32 s1, s45, s18 // (op 570) 

s_mul_i32 s9, s10, s1 // (op 570) 

s_mul_i32 s1, s0, s9 // (op 570) 

s_sub_u32 s9, s16, s1 // (op 570) 

s_mul_hi_i32 s1, s9, s44 // (op 570) 
s_add_i32 s18, s1, s9 // (op 570) 


s_ashr_i32 s1, s18, 31 // (op 570) 
s_cmp_eq_i32 s44, 0 // (op 570) 
s_mov_b32 s9, scc // (op 570) 
s_cmp_eq_i32 s9, 0 // (op 570) 
s_cselect_b32 s19, 0, -1 // (op 570) 

s_and_b32 s9, s48, 31 // (op 570) 
s_lshl_b32 s49, 1, s9 // (op 570) 
s_add_i32 s50, s49, s19 // (op 570) 


s_and_b32 s19, s1, s50 // (op 570) 


s_add_i32 s1, s18, s19 // (op 570) 


s_ashr_i32 s18, s1, s9 // (op 570) 


s_lshl_b32 s1, s48, 24 // (op 570) 
s_ashr_i32 s9, s1, 31 // (op 570) 

s_xor_b32 s1, s18, s9 // (op 570) 

s_sub_i32 s18, s1, s9 // (op 570) 


s_cmp_eq_i32 s0, 0 // (op 570) 
s_mov_b32 s1, scc // (op 570) 

s_cmp_eq_i32 s1, 0 // (op 570) 
s_cselect_b32 s0, s18, s8 // (op 570) 



v_lshl_add_u32 v11, s0, 5, v10 // (op 570) 


v_mul_lo_u32 v19, v11, s23 // (op 570) most significant: low of low * high
v_mul_hi_u32 v10, v11, s22 // (op 570) most significant: high of low * low
v_mul_lo_u32 v18, v11, s22 // (op 570) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 570) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v18 // (op 570) least significant half
v_addc_co_u32 v11, s[0:1], v13, v19, s[0:1] // (op 570) most significant half



s_mul_i32 s1, s22, s5 // (op 572) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 572) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 572) least significant
s_mul_hi_u32 s9, s22, s4 // (op 572) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 572) most significant: sum
s_add_u32 s1, s1, s9 // (op 572) most significant: sum


s_mov_b32 s56, s24 // (op 572) reg expression
s_mov_b32 s57, s25 // (op 572) reg expression
s_mov_b32 s58, s0 // (op 572) reg expression
s_mov_b32 s59, 131072 // (op 572) call()

s_mul_hi_i32 s0, s16, s12 // (op 589) 
v_lshrrev_b32 v12, 3, v1 // (op 589) 
s_add_i32 s1, s0, s16 // (op 589) 

v_and_b32 v13, -8, v1 // (op 589) 
s_ashr_i32 s0, s1, 31 // (op 589) 
v_sub_u32 v18, v1, v13 // (op 589) 

s_cmp_eq_i32 s12, 0 // (op 589) 
s_mov_b32 s8, scc // (op 589) 
v_lshlrev_b32 v13, 4, v18 // (op 589) 

s_cmp_eq_i32 s8, 0 // (op 589) 
s_cselect_b32 s9, 0, -1 // (op 589) 

v_mov_b32 v19, 0 // (op 589) convert
v_mov_b32 v18, v13 // (op 589) convert

s_and_b32 s8, s13, 31 // (op 589) 
s_lshl_b32 s18, 1, s8 // (op 589) 
s_add_i32 s19, s18, s9 // (op 589) 


s_and_b32 s9, s0, s19 // (op 589) 


s_add_i32 s0, s1, s9 // (op 589) 


s_ashr_i32 s1, s0, s8 // (op 589) 


s_lshl_b32 s0, s13, 24 // (op 589) 
s_ashr_i32 s8, s0, 31 // (op 589) 

s_xor_b32 s0, s1, s8 // (op 589) 

s_sub_i32 s1, s0, s8 // (op 589) 


s_mul_hi_i32 s0, s1, s14 // (op 589) 
s_add_i32 s8, s0, s1 // (op 589) 

s_ashr_i32 s0, s8, 31 // (op 589) 
s_cmp_eq_i32 s14, 0 // (op 589) 
s_mov_b32 s9, scc // (op 589) 
s_cmp_eq_i32 s9, 0 // (op 589) 
s_cselect_b32 s18, 0, -1 // (op 589) 

s_and_b32 s9, s15, 31 // (op 589) 
s_lshl_b32 s19, 1, s9 // (op 589) 
s_add_i32 s49, s19, s18 // (op 589) 


s_and_b32 s18, s0, s49 // (op 589) 


s_add_i32 s0, s8, s18 // (op 589) 


s_ashr_i32 s8, s0, s9 // (op 589) 


s_lshl_b32 s0, s15, 24 // (op 589) 
s_ashr_i32 s9, s0, 31 // (op 589) 

s_xor_b32 s0, s8, s9 // (op 589) 

s_sub_i32 s8, s0, s9 // (op 589) 


s_add_u32 s18, s4, 32 // (op 589) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 589) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 589) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 589) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 589) 

s_mul_i32 s0, s8, s18 // (op 589) 

s_sub_i32 s8, s1, s0 // (op 589) 


s_mul_hi_i32 s9, s16, s11 // (op 589) 
s_add_i32 s0, s9, s16 // (op 589) 

s_ashr_i32 s1, s0, 31 // (op 589) 
s_cmp_eq_i32 s11, 0 // (op 589) 
s_mov_b32 s9, scc // (op 589) 
s_cmp_eq_i32 s9, 0 // (op 589) 
s_cselect_b32 s49, 0, -1 // (op 589) 

s_and_b32 s9, s46, 31 // (op 589) 
s_lshl_b32 s50, 1, s9 // (op 589) 
s_add_i32 s51, s50, s49 // (op 589) 


s_and_b32 s49, s1, s51 // (op 589) 


s_add_i32 s1, s0, s49 // (op 589) 


s_ashr_i32 s0, s1, s9 // (op 589) 


s_lshl_b32 s1, s46, 24 // (op 589) 
s_ashr_i32 s9, s1, 31 // (op 589) 

s_xor_b32 s1, s0, s9 // (op 589) 

s_sub_i32 s0, s1, s9 // (op 589) 


s_mul_i32 s1, s45, s18 // (op 589) 

s_mul_i32 s9, s10, s1 // (op 589) 

s_mul_i32 s1, s0, s9 // (op 589) 

s_sub_u32 s9, s16, s1 // (op 589) 

s_mul_hi_i32 s1, s9, s44 // (op 589) 
s_add_i32 s18, s1, s9 // (op 589) 


s_ashr_i32 s1, s18, 31 // (op 589) 
s_cmp_eq_i32 s44, 0 // (op 589) 
s_mov_b32 s9, scc // (op 589) 
s_cmp_eq_i32 s9, 0 // (op 589) 
s_cselect_b32 s19, 0, -1 // (op 589) 

s_and_b32 s9, s48, 31 // (op 589) 
s_lshl_b32 s49, 1, s9 // (op 589) 
s_add_i32 s50, s49, s19 // (op 589) 


s_and_b32 s19, s1, s50 // (op 589) 


s_add_i32 s1, s18, s19 // (op 589) 


s_ashr_i32 s18, s1, s9 // (op 589) 


s_lshl_b32 s1, s48, 24 // (op 589) 
s_ashr_i32 s9, s1, 31 // (op 589) 

s_xor_b32 s1, s18, s9 // (op 589) 

s_sub_i32 s18, s1, s9 // (op 589) 


s_cmp_eq_i32 s0, 0 // (op 589) 
s_mov_b32 s1, scc // (op 589) 

s_cmp_eq_i32 s1, 0 // (op 589) 
s_cselect_b32 s0, s18, s8 // (op 589) 



v_lshl_add_u32 v13, s0, 5, v12 // (op 589) 


v_mul_lo_u32 v21, v13, s23 // (op 589) most significant: low of low * high
v_mul_hi_u32 v12, v13, s22 // (op 589) most significant: high of low * low
v_mul_lo_u32 v20, v13, s22 // (op 589) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 589) most significant: sum


v_add_co_u32 v12, s[0:1], v18, v20 // (op 589) least significant half
v_addc_co_u32 v13, s[0:1], v19, v21, s[0:1] // (op 589) most significant half



s_mul_i32 s1, s22, s5 // (op 591) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 591) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 591) least significant
s_mul_hi_u32 s9, s22, s4 // (op 591) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 591) most significant: sum
s_add_u32 s1, s1, s9 // (op 591) most significant: sum


s_mov_b32 s56, s24 // (op 591) reg expression
s_mov_b32 s57, s25 // (op 591) reg expression
s_mov_b32 s58, s0 // (op 591) reg expression
s_mov_b32 s59, 131072 // (op 591) call()

s_mul_hi_i32 s0, s16, s12 // (op 634) 
v_lshrrev_b32 v18, 7, v1 // (op 634) 
s_add_i32 s1, s0, s16 // (op 634) 

v_and_b32 v19, 15, v1 // (op 634) 
s_ashr_i32 s0, s1, 31 // (op 634) 
v_lshl_add_u32 v20, v18, 4, v19 // (op 634) 


s_cmp_eq_i32 s12, 0 // (op 634) 
s_mov_b32 s8, scc // (op 634) 
v_and_b32 v18, 63, v1 // (op 634) 
s_cmp_eq_i32 s8, 0 // (op 634) 
s_cselect_b32 s9, 0, -1 // (op 634) 

v_lshrrev_b32 v19, 4, v18 // (op 634) 

s_and_b32 s8, s13, 31 // (op 634) 
s_lshl_b32 s18, 1, s8 // (op 634) 
s_add_i32 s19, s18, s9 // (op 634) 


s_and_b32 s9, s0, s19 // (op 634) 


s_add_i32 s0, s1, s9 // (op 634) 


s_ashr_i32 s1, s0, s8 // (op 634) 


s_lshl_b32 s0, s13, 24 // (op 634) 
s_ashr_i32 s8, s0, 31 // (op 634) 

s_xor_b32 s0, s1, s8 // (op 634) 

s_sub_i32 s1, s0, s8 // (op 634) 


s_mul_hi_i32 s0, s1, s14 // (op 634) 
s_add_i32 s8, s0, s1 // (op 634) 

s_ashr_i32 s0, s8, 31 // (op 634) 
s_cmp_eq_i32 s14, 0 // (op 634) 
s_mov_b32 s9, scc // (op 634) 
s_cmp_eq_i32 s9, 0 // (op 634) 
s_cselect_b32 s18, 0, -1 // (op 634) 

s_and_b32 s9, s15, 31 // (op 634) 
s_lshl_b32 s19, 1, s9 // (op 634) 
s_add_i32 s49, s19, s18 // (op 634) 


s_and_b32 s18, s0, s49 // (op 634) 


s_add_i32 s0, s8, s18 // (op 634) 


s_ashr_i32 s8, s0, s9 // (op 634) 


s_lshl_b32 s0, s15, 24 // (op 634) 
s_ashr_i32 s9, s0, 31 // (op 634) 

s_xor_b32 s0, s8, s9 // (op 634) 

s_sub_i32 s8, s0, s9 // (op 634) 


s_mul_i32 s0, s1, s45 // (op 634) 

s_sub_u32 s1, s16, s0 // (op 634) 

s_mul_i32 s0, s8, s45 // (op 634) 
s_add_u32 s0, s0, s1 // (op 634) 


s_mul_hi_i32 s1, s16, s11 // (op 634) 
s_add_i32 s8, s1, s16 // (op 634) 

s_ashr_i32 s1, s8, 31 // (op 634) 
s_cmp_eq_i32 s11, 0 // (op 634) 
s_mov_b32 s9, scc // (op 634) 
s_cmp_eq_i32 s9, 0 // (op 634) 
s_cselect_b32 s18, 0, -1 // (op 634) 

s_and_b32 s9, s46, 31 // (op 634) 
s_lshl_b32 s19, 1, s9 // (op 634) 
s_add_i32 s49, s19, s18 // (op 634) 


s_and_b32 s18, s1, s49 // (op 634) 


s_add_i32 s1, s8, s18 // (op 634) 


s_ashr_i32 s8, s1, s9 // (op 634) 


s_lshl_b32 s1, s46, 24 // (op 634) 
s_ashr_i32 s9, s1, 31 // (op 634) 

s_xor_b32 s1, s8, s9 // (op 634) 

s_sub_i32 s8, s1, s9 // (op 634) 


s_add_u32 s18, s4, 32 // (op 634) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 634) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 634) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 634) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 634) 

s_mul_i32 s1, s45, s18 // (op 634) 

s_mul_i32 s9, s10, s1 // (op 634) 

s_mul_i32 s1, s8, s9 // (op 634) 

s_sub_u32 s9, s16, s1 // (op 634) 

s_mov_b32 s1, s9 // (op 634) 
s_mul_hi_i32 s18, s1, s44 // (op 634) 
s_add_i32 s19, s18, s1 // (op 634) 


s_ashr_i32 s1, s19, 31 // (op 634) 
s_cmp_eq_i32 s44, 0 // (op 634) 
s_mov_b32 s18, scc // (op 634) 
s_cmp_eq_i32 s18, 0 // (op 634) 
s_cselect_b32 s49, 0, -1 // (op 634) 

s_and_b32 s18, s48, 31 // (op 634) 
s_lshl_b32 s50, 1, s18 // (op 634) 
s_add_i32 s51, s50, s49 // (op 634) 


s_and_b32 s49, s1, s51 // (op 634) 


s_add_i32 s1, s19, s49 // (op 634) 


s_ashr_i32 s19, s1, s18 // (op 634) 


s_lshl_b32 s1, s48, 24 // (op 634) 
s_ashr_i32 s18, s1, 31 // (op 634) 

s_xor_b32 s1, s19, s18 // (op 634) 

s_sub_i32 s19, s1, s18 // (op 634) 


s_mul_i32 s1, s47, s45 // (op 634) 
s_add_u32 s50, s6, 16 // (op 634) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 634) End of Int64 add, unlocking SCC
s_sub_u32 s60, s50, 1 // (op 634) least significant half; sets scc
s_subb_u32 s61, s51, 0 // (op 634) most significant half; uses scc

s_ashr_i64 s[50:51], s[60:61], 4 // (op 634) 

s_sub_i32 s18, s50, s1 // (op 634) 

s_mul_i32 s49, s19, s18 // (op 634) 


s_sub_u32 s18, s9, s49 // (op 634) 


s_add_u32 s9, s18, s1 // (op 634) 


s_cmp_eq_i32 s8, 0 // (op 634) 
s_mov_b32 s1, scc // (op 634) 

s_cmp_eq_i32 s1, 0 // (op 634) 
s_cselect_b32 s8, s9, s0 // (op 634) 



v_lshl_add_u32 v18, s8, 4, v20 // (op 634) 


v_mul_lo_u32 v20, v18, s26 // (op 634) 
v_add_u32 v20, v20, v19 // (op 634) 


s_mul_i32 s1, s26, s7 // (op 636) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 636) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 636) least significant
s_mul_hi_u32 s9, s26, s6 // (op 636) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 636) most significant: sum
s_add_u32 s1, s1, s9 // (op 636) most significant: sum


s_mov_b32 s60, s28 // (op 636) reg expression
s_mov_b32 s61, s29 // (op 636) reg expression
s_mov_b32 s62, s0 // (op 636) reg expression
s_mov_b32 s63, 131072 // (op 636) call()

s_mul_hi_i32 s0, s16, s12 // (op 653) 
v_lshrrev_b32 v18, 7, v1 // (op 653) 
s_add_i32 s1, s0, s16 // (op 653) 

v_and_b32 v19, 15, v1 // (op 653) 
s_ashr_i32 s0, s1, 31 // (op 653) 
v_lshl_add_u32 v21, v18, 4, v19 // (op 653) 


s_cmp_eq_i32 s12, 0 // (op 653) 
s_mov_b32 s8, scc // (op 653) 
v_and_b32 v18, 63, v1 // (op 653) 
s_cmp_eq_i32 s8, 0 // (op 653) 
s_cselect_b32 s9, 0, -1 // (op 653) 

v_lshrrev_b32 v19, 4, v18 // (op 653) 

s_and_b32 s8, s13, 31 // (op 653) 
s_lshl_b32 s18, 1, s8 // (op 653) 
s_add_i32 s19, s18, s9 // (op 653) 


s_and_b32 s9, s0, s19 // (op 653) 


s_add_i32 s0, s1, s9 // (op 653) 


s_ashr_i32 s1, s0, s8 // (op 653) 


s_lshl_b32 s0, s13, 24 // (op 653) 
s_ashr_i32 s8, s0, 31 // (op 653) 

s_xor_b32 s0, s1, s8 // (op 653) 

s_sub_i32 s1, s0, s8 // (op 653) 


s_mul_hi_i32 s0, s1, s14 // (op 653) 
s_add_i32 s8, s0, s1 // (op 653) 

s_ashr_i32 s0, s8, 31 // (op 653) 
s_cmp_eq_i32 s14, 0 // (op 653) 
s_mov_b32 s9, scc // (op 653) 
s_cmp_eq_i32 s9, 0 // (op 653) 
s_cselect_b32 s18, 0, -1 // (op 653) 

s_and_b32 s9, s15, 31 // (op 653) 
s_lshl_b32 s19, 1, s9 // (op 653) 
s_add_i32 s49, s19, s18 // (op 653) 


s_and_b32 s18, s0, s49 // (op 653) 


s_add_i32 s0, s8, s18 // (op 653) 


s_ashr_i32 s8, s0, s9 // (op 653) 


s_lshl_b32 s0, s15, 24 // (op 653) 
s_ashr_i32 s9, s0, 31 // (op 653) 

s_xor_b32 s0, s8, s9 // (op 653) 

s_sub_i32 s8, s0, s9 // (op 653) 


s_mul_i32 s0, s1, s45 // (op 653) 

s_sub_u32 s1, s16, s0 // (op 653) 

s_mul_i32 s0, s8, s45 // (op 653) 
s_add_u32 s0, s0, s1 // (op 653) 


s_mul_hi_i32 s1, s16, s11 // (op 653) 
s_add_i32 s8, s1, s16 // (op 653) 

s_ashr_i32 s1, s8, 31 // (op 653) 
s_cmp_eq_i32 s11, 0 // (op 653) 
s_mov_b32 s9, scc // (op 653) 
s_cmp_eq_i32 s9, 0 // (op 653) 
s_cselect_b32 s18, 0, -1 // (op 653) 

s_and_b32 s9, s46, 31 // (op 653) 
s_lshl_b32 s19, 1, s9 // (op 653) 
s_add_i32 s49, s19, s18 // (op 653) 


s_and_b32 s18, s1, s49 // (op 653) 


s_add_i32 s1, s8, s18 // (op 653) 


s_ashr_i32 s8, s1, s9 // (op 653) 


s_lshl_b32 s1, s46, 24 // (op 653) 
s_ashr_i32 s9, s1, 31 // (op 653) 

s_xor_b32 s1, s8, s9 // (op 653) 

s_sub_i32 s8, s1, s9 // (op 653) 


s_add_u32 s18, s4, 32 // (op 653) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 653) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 653) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 653) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 653) 

s_mul_i32 s1, s45, s18 // (op 653) 

s_mul_i32 s9, s10, s1 // (op 653) 

s_mul_i32 s1, s8, s9 // (op 653) 

s_sub_u32 s9, s16, s1 // (op 653) 

s_mov_b32 s1, s9 // (op 653) 
s_mul_hi_i32 s18, s1, s44 // (op 653) 
s_add_i32 s19, s18, s1 // (op 653) 


s_ashr_i32 s1, s19, 31 // (op 653) 
s_cmp_eq_i32 s44, 0 // (op 653) 
s_mov_b32 s18, scc // (op 653) 
s_cmp_eq_i32 s18, 0 // (op 653) 
s_cselect_b32 s49, 0, -1 // (op 653) 

s_and_b32 s18, s48, 31 // (op 653) 
s_lshl_b32 s50, 1, s18 // (op 653) 
s_add_i32 s51, s50, s49 // (op 653) 


s_and_b32 s49, s1, s51 // (op 653) 


s_add_i32 s1, s19, s49 // (op 653) 


s_ashr_i32 s19, s1, s18 // (op 653) 


s_lshl_b32 s1, s48, 24 // (op 653) 
s_ashr_i32 s18, s1, 31 // (op 653) 

s_xor_b32 s1, s19, s18 // (op 653) 

s_sub_i32 s19, s1, s18 // (op 653) 


s_mul_i32 s1, s47, s45 // (op 653) 
s_add_u32 s50, s6, 16 // (op 653) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 653) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 653) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 653) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 4 // (op 653) 

s_sub_i32 s18, s50, s1 // (op 653) 

s_mul_i32 s49, s19, s18 // (op 653) 


s_sub_u32 s18, s9, s49 // (op 653) 


s_add_u32 s9, s18, s1 // (op 653) 


s_cmp_eq_i32 s8, 0 // (op 653) 
s_mov_b32 s1, scc // (op 653) 

s_cmp_eq_i32 s1, 0 // (op 653) 
s_cselect_b32 s8, s9, s0 // (op 653) 



v_lshl_add_u32 v18, s8, 4, v21 // (op 653) 


v_mul_lo_u32 v21, v18, s26 // (op 653) 
v_add_u32 v21, v21, v19 // (op 653) 


s_mul_i32 s1, s26, s7 // (op 655) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 655) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 655) least significant
s_mul_hi_u32 s9, s26, s6 // (op 655) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 655) most significant: sum
s_add_u32 s1, s1, s9 // (op 655) most significant: sum


s_mov_b32 s60, s28 // (op 655) reg expression
s_mov_b32 s61, s29 // (op 655) reg expression
s_mov_b32 s62, s0 // (op 655) reg expression
s_mov_b32 s63, 131072 // (op 655) call()

s_mul_hi_i32 s0, s16, s12 // (op 698) 
v_lshrrev_b32 v18, 2, v1 // (op 698) 
s_add_i32 s1, s0, s16 // (op 698) 

v_and_b32 v19, -4, v1 // (op 698) 
s_ashr_i32 s0, s1, 31 // (op 698) 
v_sub_u32 v22, v1, v19 // (op 698) 

s_cmp_eq_i32 s12, 0 // (op 698) 
s_mov_b32 s8, scc // (op 698) 
s_cmp_eq_i32 s8, 0 // (op 698) 
s_cselect_b32 s9, 0, -1 // (op 698) 

s_and_b32 s8, s13, 31 // (op 698) 
s_lshl_b32 s18, 1, s8 // (op 698) 
s_add_i32 s19, s18, s9 // (op 698) 


s_and_b32 s9, s0, s19 // (op 698) 


s_add_i32 s0, s1, s9 // (op 698) 


s_ashr_i32 s1, s0, s8 // (op 698) 


s_lshl_b32 s0, s13, 24 // (op 698) 
s_ashr_i32 s8, s0, 31 // (op 698) 

s_xor_b32 s0, s1, s8 // (op 698) 

s_sub_i32 s1, s0, s8 // (op 698) 


s_mul_hi_i32 s0, s1, s14 // (op 698) 
s_add_i32 s8, s0, s1 // (op 698) 

s_ashr_i32 s0, s8, 31 // (op 698) 
s_cmp_eq_i32 s14, 0 // (op 698) 
s_mov_b32 s9, scc // (op 698) 
s_cmp_eq_i32 s9, 0 // (op 698) 
s_cselect_b32 s18, 0, -1 // (op 698) 

s_and_b32 s9, s15, 31 // (op 698) 
s_lshl_b32 s19, 1, s9 // (op 698) 
s_add_i32 s49, s19, s18 // (op 698) 


s_and_b32 s18, s0, s49 // (op 698) 


s_add_i32 s0, s8, s18 // (op 698) 


s_ashr_i32 s8, s0, s9 // (op 698) 


s_lshl_b32 s0, s15, 24 // (op 698) 
s_ashr_i32 s9, s0, 31 // (op 698) 

s_xor_b32 s0, s8, s9 // (op 698) 

s_sub_i32 s8, s0, s9 // (op 698) 


s_add_u32 s18, s4, 32 // (op 698) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 698) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 698) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 698) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 698) 

s_mul_i32 s0, s8, s18 // (op 698) 

s_sub_i32 s8, s1, s0 // (op 698) 


s_mul_hi_i32 s9, s16, s11 // (op 698) 
s_add_i32 s0, s9, s16 // (op 698) 

s_ashr_i32 s1, s0, 31 // (op 698) 
s_cmp_eq_i32 s11, 0 // (op 698) 
s_mov_b32 s9, scc // (op 698) 
s_cmp_eq_i32 s9, 0 // (op 698) 
s_cselect_b32 s49, 0, -1 // (op 698) 

s_and_b32 s9, s46, 31 // (op 698) 
s_lshl_b32 s50, 1, s9 // (op 698) 
s_add_i32 s51, s50, s49 // (op 698) 


s_and_b32 s49, s1, s51 // (op 698) 


s_add_i32 s1, s0, s49 // (op 698) 


s_ashr_i32 s0, s1, s9 // (op 698) 


s_lshl_b32 s1, s46, 24 // (op 698) 
s_ashr_i32 s9, s1, 31 // (op 698) 

s_xor_b32 s1, s0, s9 // (op 698) 

s_sub_i32 s0, s1, s9 // (op 698) 


s_mul_i32 s1, s45, s18 // (op 698) 

s_mul_i32 s9, s10, s1 // (op 698) 

s_mul_i32 s1, s0, s9 // (op 698) 

s_sub_u32 s9, s16, s1 // (op 698) 

s_mul_hi_i32 s1, s9, s44 // (op 698) 
s_add_i32 s18, s1, s9 // (op 698) 


s_ashr_i32 s1, s18, 31 // (op 698) 
s_cmp_eq_i32 s44, 0 // (op 698) 
s_mov_b32 s9, scc // (op 698) 
s_cmp_eq_i32 s9, 0 // (op 698) 
s_cselect_b32 s19, 0, -1 // (op 698) 

s_and_b32 s9, s48, 31 // (op 698) 
s_lshl_b32 s49, 1, s9 // (op 698) 
s_add_i32 s50, s49, s19 // (op 698) 


s_and_b32 s19, s1, s50 // (op 698) 


s_add_i32 s1, s18, s19 // (op 698) 


s_ashr_i32 s18, s1, s9 // (op 698) 


s_lshl_b32 s1, s48, 24 // (op 698) 
s_ashr_i32 s9, s1, 31 // (op 698) 

s_xor_b32 s1, s18, s9 // (op 698) 

s_sub_i32 s18, s1, s9 // (op 698) 


s_cmp_eq_i32 s0, 0 // (op 698) 
s_mov_b32 s1, scc // (op 698) 

s_cmp_eq_i32 s1, 0 // (op 698) 
s_cselect_b32 s0, s18, s8 // (op 698) 



v_lshl_add_u32 v19, s0, 5, v18 // (op 698) 


v_mul_lo_u32 v18, v19, s30 // (op 698) 

v_add_u32 v19, v22, v18 // (op 698) 


s_mul_i32 s1, s30, s5 // (op 700) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 700) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 700) least significant
s_mul_hi_u32 s9, s30, s4 // (op 700) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 700) most significant: sum
s_add_u32 s1, s1, s9 // (op 700) most significant: sum


s_mov_b32 s64, s32 // (op 700) reg expression
s_mov_b32 s65, s33 // (op 700) reg expression
s_mov_b32 s66, s0 // (op 700) reg expression
s_mov_b32 s67, 131072 // (op 700) call()

s_mul_hi_i32 s0, s16, s12 // (op 717) 
v_lshrrev_b32 v18, 2, v1 // (op 717) 
s_add_i32 s1, s0, s16 // (op 717) 

v_and_b32 v22, -4, v1 // (op 717) 
s_ashr_i32 s0, s1, 31 // (op 717) 
v_sub_u32 v23, v1, v22 // (op 717) 

s_cmp_eq_i32 s12, 0 // (op 717) 
s_mov_b32 s8, scc // (op 717) 
s_cmp_eq_i32 s8, 0 // (op 717) 
s_cselect_b32 s9, 0, -1 // (op 717) 

s_and_b32 s8, s13, 31 // (op 717) 
s_lshl_b32 s18, 1, s8 // (op 717) 
s_add_i32 s19, s18, s9 // (op 717) 


s_and_b32 s9, s0, s19 // (op 717) 


s_add_i32 s0, s1, s9 // (op 717) 


s_ashr_i32 s1, s0, s8 // (op 717) 


s_lshl_b32 s0, s13, 24 // (op 717) 
s_ashr_i32 s8, s0, 31 // (op 717) 

s_xor_b32 s0, s1, s8 // (op 717) 

s_sub_i32 s1, s0, s8 // (op 717) 


s_mul_hi_i32 s0, s1, s14 // (op 717) 
s_add_i32 s8, s0, s1 // (op 717) 

s_ashr_i32 s0, s8, 31 // (op 717) 
s_cmp_eq_i32 s14, 0 // (op 717) 
s_mov_b32 s9, scc // (op 717) 
s_cmp_eq_i32 s9, 0 // (op 717) 
s_cselect_b32 s18, 0, -1 // (op 717) 

s_and_b32 s9, s15, 31 // (op 717) 
s_lshl_b32 s19, 1, s9 // (op 717) 
s_add_i32 s49, s19, s18 // (op 717) 


s_and_b32 s18, s0, s49 // (op 717) 


s_add_i32 s0, s8, s18 // (op 717) 


s_ashr_i32 s8, s0, s9 // (op 717) 


s_lshl_b32 s0, s15, 24 // (op 717) 
s_ashr_i32 s9, s0, 31 // (op 717) 

s_xor_b32 s0, s8, s9 // (op 717) 

s_sub_i32 s8, s0, s9 // (op 717) 


s_add_u32 s18, s4, 32 // (op 717) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 717) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 717) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 717) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 717) 

s_mul_i32 s0, s8, s18 // (op 717) 

s_sub_i32 s8, s1, s0 // (op 717) 


s_mul_hi_i32 s9, s16, s11 // (op 717) 
s_add_i32 s0, s9, s16 // (op 717) 

s_ashr_i32 s1, s0, 31 // (op 717) 
s_cmp_eq_i32 s11, 0 // (op 717) 
s_mov_b32 s9, scc // (op 717) 
s_cmp_eq_i32 s9, 0 // (op 717) 
s_cselect_b32 s49, 0, -1 // (op 717) 

s_and_b32 s9, s46, 31 // (op 717) 
s_lshl_b32 s50, 1, s9 // (op 717) 
s_add_i32 s51, s50, s49 // (op 717) 


s_and_b32 s49, s1, s51 // (op 717) 


s_add_i32 s1, s0, s49 // (op 717) 


s_ashr_i32 s0, s1, s9 // (op 717) 


s_lshl_b32 s1, s46, 24 // (op 717) 
s_ashr_i32 s9, s1, 31 // (op 717) 

s_xor_b32 s1, s0, s9 // (op 717) 

s_sub_i32 s0, s1, s9 // (op 717) 


s_mul_i32 s1, s45, s18 // (op 717) 

s_mul_i32 s9, s10, s1 // (op 717) 

s_mul_i32 s1, s0, s9 // (op 717) 

s_sub_u32 s9, s16, s1 // (op 717) 

s_mul_hi_i32 s1, s9, s44 // (op 717) 
s_add_i32 s18, s1, s9 // (op 717) 


s_ashr_i32 s1, s18, 31 // (op 717) 
s_cmp_eq_i32 s44, 0 // (op 717) 
s_mov_b32 s9, scc // (op 717) 
s_cmp_eq_i32 s9, 0 // (op 717) 
s_cselect_b32 s19, 0, -1 // (op 717) 

s_and_b32 s9, s48, 31 // (op 717) 
s_lshl_b32 s49, 1, s9 // (op 717) 
s_add_i32 s50, s49, s19 // (op 717) 


s_and_b32 s19, s1, s50 // (op 717) 


s_add_i32 s1, s18, s19 // (op 717) 


s_ashr_i32 s18, s1, s9 // (op 717) 


s_lshl_b32 s1, s48, 24 // (op 717) 
s_ashr_i32 s9, s1, 31 // (op 717) 

s_xor_b32 s1, s18, s9 // (op 717) 

s_sub_i32 s18, s1, s9 // (op 717) 


s_cmp_eq_i32 s0, 0 // (op 717) 
s_mov_b32 s1, scc // (op 717) 

s_cmp_eq_i32 s1, 0 // (op 717) 
s_cselect_b32 s0, s18, s8 // (op 717) 



v_lshl_add_u32 v22, s0, 5, v18 // (op 717) 


v_mul_lo_u32 v18, v22, s30 // (op 717) 

v_add_u32 v22, v23, v18 // (op 717) 


s_mul_i32 s1, s30, s5 // (op 719) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 719) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 719) least significant
s_mul_hi_u32 s9, s30, s4 // (op 719) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 719) most significant: sum
s_add_u32 s1, s1, s9 // (op 719) most significant: sum


s_mov_b32 s64, s32 // (op 719) reg expression
s_mov_b32 s65, s33 // (op 719) reg expression
s_mov_b32 s66, s0 // (op 719) reg expression
s_mov_b32 s67, 131072 // (op 719) call()




buffer_load_ubyte v18, v20, s[60:63], 0 offen offset: 0 // (op 337) Load value
v_and_b32 v23, -8, v1 // (op 835) 
v_sub_u32 v24, v1, v23 // (op 835) 

v_lshlrev_b32 v23, 4, v24 // (op 835) 

v_lshrrev_b32 v24, 3, v1 // (op 835) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 835) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 835) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[8:9], s[22:23], 4 // (op 455) 
v_mov_b64 v[24:25], s[8:9] // (op 455) convert


v_mov_b64 v[26:27], v[10:11] // (op 455) 
s_mov_b32 m0, s0 // (op 455) reg expression
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 455) Load value direct to lds
v_add_co_u32 v26, s[8:9], v26, v24 // (op 455) least significant half
v_addc_co_u32 v27, s[8:9], 0, v27, s[8:9] // (op 455) most significant half

s_add_u32 m0, m0, 2048 // (op 455) 
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 455) Load value direct to lds






buffer_load_ubyte v23, v19, s[64:67], 0 offen offset: 0 // (op 344) Load value
v_and_b32 v24, -8, v1 // (op 783) 
v_sub_u32 v25, v1, v24 // (op 783) 

v_lshlrev_b32 v24, 4, v25 // (op 783) 

v_lshrrev_b32 v25, 3, v1 // (op 783) 
v_lshl_add_u32 v26, v25, 7, v24 // (op 783) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 783) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 460) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 460) reg expression
buffer_load_dwordx4 v14, s[52:55], 0 offen offset: 0 lds // (op 460) Load value direct to lds





buffer_load_ubyte v24, v20, s[60:63], 0 offen offset:4 // (op 351) Load value

v_and_b32 v20, -8, v1 // (op 1000) 
v_sub_u32 v25, v1, v20 // (op 1000) 

v_lshlrev_b32 v20, 4, v25 // (op 1000) 

v_lshrrev_b32 v25, 3, v1 // (op 1000) 
v_lshl_add_u32 v26, v25, 7, v20 // (op 1000) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1000) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[8:9], s[22:23], 4 // (op 465) 
v_mov_b64 v[26:27], s[8:9] // (op 465) convert


v_mov_b64 v[28:29], v[10:11] // (op 465) 
s_add_u32 s1, 6144, s0 // (op 465) 
s_mov_b32 m0, s1 // (op 465) reg expression
s_mov_b32 s8, 128 // (op 465) call()
buffer_load_dwordx4 v28, s[56:59], s8 offen offset: 0 lds // (op 465) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 465) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 465) most significant half

s_add_u32 m0, m0, 2048 // (op 465) 
buffer_load_dwordx4 v28, s[56:59], s8 offen offset: 0 lds // (op 465) Load value direct to lds









buffer_load_ubyte v20, v19, s[64:67], 0 offen offset:4 // (op 358) Load value

v_and_b32 v19, -8, v1 // (op 961) 
v_sub_u32 v10, v1, v19 // (op 961) 

v_lshlrev_b32 v11, 4, v10 // (op 961) 

v_lshrrev_b32 v10, 3, v1 // (op 961) 
v_lshl_add_u32 v19, v10, 7, v11 // (op 961) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 961) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 10240, s0 // (op 467) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 467) reg expression
s_mov_b32 s8, 128 // (op 467) call()
buffer_load_dwordx4 v14, s[52:55], s8 offen offset: 0 lds // (op 467) Load value direct to lds




v_and_b32 v19, -4, v1 // (op 888) 
v_sub_u32 v10, v1, v19 // (op 888) 

v_and_b32 v11, 4294967292, v1 // (op 888) 
v_add_u32 v19, v11, v10 // (op 888) 





s_waitcnt vmcnt(6)
ds_write_b8 v19, v23 offset:12288 // (op 364) Store local data 

s_barrier  // (op 1085) 



ds_read_b128 v[26:29], v4 // (op 381) Load local data 
ds_read_b128 v[30:33], v4 offset:64 // (op 381) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1101) 



ds_read_u8 v19, v0 offset:12288 // (op 384) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1069) 



ds_read_b128 v[34:37], v3 offset:4096 // (op 378) Load local data 
ds_read_b128 v[38:41], v3 offset:4160 // (op 378) Load local data 
v_and_b32 v25, -8, v1 // (op 755) 
v_sub_u32 v10, v1, v25 // (op 755) 

v_lshlrev_b32 v11, 4, v10 // (op 755) 

v_lshrrev_b32 v10, 3, v1 // (op 755) 
v_lshl_add_u32 v25, v10, 7, v11 // (op 755) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 755) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -8, v1 // (op 807) 
v_sub_u32 v10, v1, v25 // (op 807) 

v_lshlrev_b32 v11, 4, v10 // (op 807) 

v_lshrrev_b32 v10, 3, v1 // (op 807) 
v_lshl_add_u32 v25, v10, 7, v11 // (op 807) 


s_nop 0
v_readfirstlane_b32 s1, v25 // (op 807) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -4, v1 // (op 859) 
v_sub_u32 v10, v1, v25 // (op 859) 

v_and_b32 v11, 4294967292, v1 // (op 859) 
v_add_u32 v25, v11, v10 // (op 859) 


v_and_b32 v10, -8, v1 // (op 946) 
v_sub_u32 v11, v1, v10 // (op 946) 

v_lshlrev_b32 v10, 4, v11 // (op 946) 

v_lshrrev_b32 v11, 3, v1 // (op 946) 
v_lshl_add_u32 v14, v11, 7, v10 // (op 946) 


s_nop 0
v_readfirstlane_b32 s8, v14 // (op 946) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -8, v1 // (op 985) 
v_sub_u32 v11, v1, v10 // (op 985) 

v_lshlrev_b32 v10, 4, v11 // (op 985) 

v_lshrrev_b32 v11, 3, v1 // (op 985) 
v_lshl_add_u32 v14, v11, 7, v10 // (op 985) 


s_nop 0
v_readfirstlane_b32 s9, v14 // (op 985) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 1024) 
v_sub_u32 v11, v1, v10 // (op 1024) 

v_and_b32 v10, 4294967292, v1 // (op 1024) 
v_add_u32 v14, v10, v11 // (op 1024) 


s_mov_b32 s18, 0 // (op 55) call()
s_ashr_i64 s[50:51], s[34:35], 7 // (op 54) 
s_mov_b32 s19, s50 // (op 54) convert

s_and_b32 s49, s19, 4294967294 // (op 54) 

s_cmp_lt_u32 s18, s49 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54:
s_add_u32 s18, s18, 2 // (op 56) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v18, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 82) 


v_mov_b32 v15, v21 // (op 75) 

buffer_load_ubyte v18, v15, s[60:63], 0 offen offset:8 // (op 75) Load value
s_barrier  // (op 1088) 

s_lshl_b64 s[50:51], s[22:23], 4 // (op 427) 
v_mov_b64 v[10:11], s[50:51] // (op 427) convert


v_mov_b64 v[42:43], v[12:13] // (op 427) 
v_mov_b64 v[44:45], v[42:43] // (op 427) 
s_mov_b32 m0, s1 // (op 427) reg expression
s_mov_b32 s19, 256 // (op 427) call()
buffer_load_dwordx4 v44, s[56:59], s19 offen offset: 0 lds // (op 427) Load value direct to lds
v_add_co_u32 v44, s[50:51], v44, v10 // (op 427) least significant half
v_addc_co_u32 v45, s[50:51], 0, v45, s[50:51] // (op 427) most significant half

s_add_u32 m0, m0, 2048 // (op 427) 
buffer_load_dwordx4 v44, s[56:59], s19 offen offset: 0 lds // (op 427) Load value direct to lds





v_mov_b32 v10, v22 // (op 21) 

buffer_load_ubyte v23, v10, s[64:67], 0 offen offset:8 // (op 21) Load value
s_barrier  // (op 1072) 


v_mov_b64 v[44:45], v[16:17] // (op 422) 
s_add_u32 s19, 4096, s0 // (op 422) 

s_mov_b32 m0, s19 // (op 422) reg expression
s_mov_b32 s49, 256 // (op 422) call()
buffer_load_dwordx4 v44, s[52:55], s49 offen offset: 0 lds // (op 422) Load value direct to lds


s_barrier  // (op 1133) 



s_waitcnt vmcnt(6)
ds_write_b8 v14, v20 offset:12416 // (op 236) Store local data 
s_barrier  // (op 1124) 



ds_read_b128 v[26:29], v7 offset:6144 // (op 229) Load local data 
ds_read_b128 v[30:33], v7 offset:6208 // (op 229) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1130) 



ds_read_u8 v19, v5 offset:12416 // (op 233) Load local data 
s_barrier  // (op 1176) 
s_waitcnt vmcnt(5)
s_barrier  // (op 1118) 



ds_read_b128 v[34:37], v6 offset:10240 // (op 227) Load local data 
ds_read_b128 v[38:41], v6 offset:10304 // (op 227) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v24, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 234) 



buffer_load_ubyte v24, v15, s[60:63], 0 offen offset:12 // (op 231) Load value

s_barrier  // (op 1127) 
v_add_u32 v21, 8, v21 // (op 645) 

s_lshl_b64 s[50:51], s[22:23], 4 // (op 450) 
v_mov_b64 v[46:47], s[50:51] // (op 450) convert


v_mov_b64 v[48:49], v[42:43] // (op 450) 
s_add_u32 s19, 6144, s9 // (op 450) 
s_mov_b32 m0, s19 // (op 450) reg expression
s_mov_b32 s49, 384 // (op 450) call()
buffer_load_dwordx4 v48, s[56:59], s49 offen offset: 0 lds // (op 450) Load value direct to lds
v_add_co_u32 v48, s[50:51], v48, v46 // (op 450) least significant half
v_addc_co_u32 v49, s[50:51], 0, v49, s[50:51] // (op 450) most significant half

s_add_u32 m0, m0, 2048 // (op 450) 
buffer_load_dwordx4 v48, s[56:59], s49 offen offset: 0 lds // (op 450) Load value direct to lds








v_add_co_u32 v12, vcc, 256, v12 // (op 581) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 581) most significant half
buffer_load_ubyte v20, v10, s[64:67], 0 offen offset:12 // (op 238) Load value

s_barrier  // (op 1121) 
v_add_u32 v22, 8, v22 // (op 709) 


s_add_u32 s19, 10240, s8 // (op 445) 

s_mov_b32 m0, s19 // (op 445) reg expression
s_mov_b32 s49, 384 // (op 445) call()
buffer_load_dwordx4 v44, s[52:55], s49 offen offset: 0 lds // (op 445) Load value direct to lds



s_barrier  // (op 1104) 
v_add_co_u32 v16, vcc, 256, v16 // (op 517) least significant half



v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 517) most significant half
s_waitcnt vmcnt(6)
ds_write_b8 v25, v23 offset:12288 // (op 26) Store local data 
s_barrier  // (op 1094) 



ds_read_b128 v[26:29], v4 // (op 70) Load local data 
ds_read_b128 v[30:33], v4 offset:64 // (op 70) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1111) 



ds_read_u8 v19, v0 offset:12288 // (op 79) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1078) 



ds_read_b128 v[34:37], v3 offset:4096 // (op 65) Load local data 
ds_read_b128 v[38:41], v3 offset:4160 // (op 65) Load local data 
s_ashr_i64 s[50:51], s[34:35], 7 // (op 54) 
s_mov_b32 s19, s50 // (op 54) convert

s_and_b32 s49, s19, 4294967294 // (op 54) 

s_cmp_lt_u32 s18, s49 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_54 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_54:













s_barrier  // (op 1142) 
s_mul_hi_i32 s0, s16, s12 // (op 480) 
v_lshrrev_b32 v25, 3, v1 // (op 480) 
s_add_i32 s1, s0, s16 // (op 480) 

v_and_b32 v10, -8, v1 // (op 480) 
s_ashr_i32 s0, s1, 31 // (op 480) 
v_sub_u32 v11, v1, v10 // (op 480) 

s_cmp_eq_i32 s12, 0 // (op 480) 
s_mov_b32 s8, scc // (op 480) 
v_lshlrev_b32 v10, 4, v11 // (op 480) 

s_cmp_eq_i32 s8, 0 // (op 480) 
s_cselect_b32 s9, 0, -1 // (op 480) 

s_and_b32 s8, s13, 31 // (op 480) 
s_lshl_b32 s18, 1, s8 // (op 480) 
s_add_i32 s19, s18, s9 // (op 480) 


s_and_b32 s9, s0, s19 // (op 480) 


s_add_i32 s0, s1, s9 // (op 480) 


s_ashr_i32 s1, s0, s8 // (op 480) 


s_lshl_b32 s0, s13, 24 // (op 480) 
s_ashr_i32 s8, s0, 31 // (op 480) 

s_xor_b32 s0, s1, s8 // (op 480) 

s_sub_i32 s1, s0, s8 // (op 480) 


s_mul_hi_i32 s0, s1, s14 // (op 480) 
s_add_i32 s8, s0, s1 // (op 480) 

s_ashr_i32 s0, s8, 31 // (op 480) 
s_cmp_eq_i32 s14, 0 // (op 480) 
s_mov_b32 s9, scc // (op 480) 
s_cmp_eq_i32 s9, 0 // (op 480) 
s_cselect_b32 s18, 0, -1 // (op 480) 

s_and_b32 s9, s15, 31 // (op 480) 
s_lshl_b32 s19, 1, s9 // (op 480) 
s_add_i32 s49, s19, s18 // (op 480) 


s_and_b32 s18, s0, s49 // (op 480) 


s_add_i32 s0, s8, s18 // (op 480) 


s_ashr_i32 s8, s0, s9 // (op 480) 


s_lshl_b32 s0, s15, 24 // (op 480) 
s_ashr_i32 s9, s0, 31 // (op 480) 

s_xor_b32 s0, s8, s9 // (op 480) 

s_sub_i32 s8, s0, s9 // (op 480) 


s_mul_i32 s0, s1, s45 // (op 480) 

s_sub_u32 s1, s16, s0 // (op 480) 

s_mul_i32 s0, s8, s45 // (op 480) 
s_add_u32 s0, s0, s1 // (op 480) 


s_mul_hi_i32 s1, s16, s11 // (op 480) 
s_add_i32 s8, s1, s16 // (op 480) 

s_ashr_i32 s1, s8, 31 // (op 480) 
s_cmp_eq_i32 s11, 0 // (op 480) 
s_mov_b32 s9, scc // (op 480) 
s_cmp_eq_i32 s9, 0 // (op 480) 
s_cselect_b32 s18, 0, -1 // (op 480) 

s_and_b32 s9, s46, 31 // (op 480) 
s_lshl_b32 s19, 1, s9 // (op 480) 
s_add_i32 s49, s19, s18 // (op 480) 


s_and_b32 s18, s1, s49 // (op 480) 


s_add_i32 s1, s8, s18 // (op 480) 


s_ashr_i32 s8, s1, s9 // (op 480) 


s_lshl_b32 s1, s46, 24 // (op 480) 
s_ashr_i32 s9, s1, 31 // (op 480) 

s_xor_b32 s1, s8, s9 // (op 480) 

s_sub_i32 s8, s1, s9 // (op 480) 


s_add_u32 s18, s4, 32 // (op 480) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 480) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 480) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 480) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 480) 

s_mul_i32 s1, s45, s18 // (op 480) 

s_mul_i32 s9, s10, s1 // (op 480) 

s_mul_i32 s1, s8, s9 // (op 480) 

s_sub_u32 s9, s16, s1 // (op 480) 

s_mov_b32 s1, s9 // (op 480) 
s_mul_hi_i32 s18, s1, s44 // (op 480) 
s_add_i32 s19, s18, s1 // (op 480) 


s_ashr_i32 s1, s19, 31 // (op 480) 
s_cmp_eq_i32 s44, 0 // (op 480) 
s_mov_b32 s18, scc // (op 480) 
s_cmp_eq_i32 s18, 0 // (op 480) 
s_cselect_b32 s49, 0, -1 // (op 480) 

s_and_b32 s18, s48, 31 // (op 480) 
s_lshl_b32 s50, 1, s18 // (op 480) 
s_add_i32 s51, s50, s49 // (op 480) 


s_and_b32 s49, s1, s51 // (op 480) 


s_add_i32 s1, s19, s49 // (op 480) 


s_ashr_i32 s19, s1, s18 // (op 480) 


s_lshl_b32 s1, s48, 24 // (op 480) 
s_ashr_i32 s18, s1, 31 // (op 480) 

s_xor_b32 s1, s19, s18 // (op 480) 

s_sub_i32 s19, s1, s18 // (op 480) 


s_mul_i32 s1, s47, s45 // (op 480) 
s_add_u32 s50, s6, 16 // (op 480) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 480) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 480) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 480) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 480) 

s_sub_i32 s18, s50, s1 // (op 480) 

s_mul_i32 s49, s19, s18 // (op 480) 


s_sub_u32 s18, s9, s49 // (op 480) 


s_add_u32 s9, s18, s1 // (op 480) 


s_cmp_eq_i32 s8, 0 // (op 480) 
s_mov_b32 s1, scc // (op 480) 

s_cmp_eq_i32 s1, 0 // (op 480) 
s_cselect_b32 s8, s9, s0 // (op 480) 



v_lshl_add_u32 v11, s8, 4, v25 // (op 480) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 480) 
s_lshr_b32 s8, s0, 1 // (op 480) 

v_lshl_add_u32 v25, s8, 8, v10 // (op 480) 


v_mov_b32 v23, 0 // (op 480) convert
v_mov_b32 v22, v25 // (op 480) convert

v_mul_lo_u32 v13, v11, s3 // (op 480) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 480) most significant: high of low * low
v_mul_lo_u32 v12, v11, s2 // (op 480) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 480) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v22 // (op 480) least significant half
v_addc_co_u32 v13, s[0:1], v13, v23, s[0:1] // (op 480) most significant half



s_mul_i32 s1, s2, s7 // (op 482) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 482) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 482) least significant
s_mul_hi_u32 s9, s2, s6 // (op 482) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 482) most significant: sum
s_add_u32 s1, s1, s9 // (op 482) most significant: sum


s_mov_b32 s52, s20 // (op 482) reg expression
s_mov_b32 s53, s21 // (op 482) reg expression
s_mov_b32 s54, s0 // (op 482) reg expression
s_mov_b32 s55, 131072 // (op 482) call()

s_mul_hi_i32 s0, s16, s12 // (op 547) 
v_lshrrev_b32 v25, 3, v1 // (op 547) 
s_add_i32 s1, s0, s16 // (op 547) 

v_and_b32 v10, -8, v1 // (op 547) 
s_ashr_i32 s0, s1, 31 // (op 547) 
v_sub_u32 v11, v1, v10 // (op 547) 

s_cmp_eq_i32 s12, 0 // (op 547) 
s_mov_b32 s8, scc // (op 547) 
v_lshlrev_b32 v10, 4, v11 // (op 547) 

s_cmp_eq_i32 s8, 0 // (op 547) 
s_cselect_b32 s9, 0, -1 // (op 547) 

s_and_b32 s8, s13, 31 // (op 547) 
s_lshl_b32 s18, 1, s8 // (op 547) 
s_add_i32 s19, s18, s9 // (op 547) 


s_and_b32 s9, s0, s19 // (op 547) 


s_add_i32 s0, s1, s9 // (op 547) 


s_ashr_i32 s1, s0, s8 // (op 547) 


s_lshl_b32 s0, s13, 24 // (op 547) 
s_ashr_i32 s8, s0, 31 // (op 547) 

s_xor_b32 s0, s1, s8 // (op 547) 

s_sub_i32 s1, s0, s8 // (op 547) 


s_mul_hi_i32 s0, s1, s14 // (op 547) 
s_add_i32 s8, s0, s1 // (op 547) 

s_ashr_i32 s0, s8, 31 // (op 547) 
s_cmp_eq_i32 s14, 0 // (op 547) 
s_mov_b32 s9, scc // (op 547) 
s_cmp_eq_i32 s9, 0 // (op 547) 
s_cselect_b32 s18, 0, -1 // (op 547) 

s_and_b32 s9, s15, 31 // (op 547) 
s_lshl_b32 s19, 1, s9 // (op 547) 
s_add_i32 s49, s19, s18 // (op 547) 


s_and_b32 s18, s0, s49 // (op 547) 


s_add_i32 s0, s8, s18 // (op 547) 


s_ashr_i32 s8, s0, s9 // (op 547) 


s_lshl_b32 s0, s15, 24 // (op 547) 
s_ashr_i32 s9, s0, 31 // (op 547) 

s_xor_b32 s0, s8, s9 // (op 547) 

s_sub_i32 s8, s0, s9 // (op 547) 


s_add_u32 s18, s4, 32 // (op 547) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 547) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 547) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 547) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 547) 

s_mul_i32 s0, s8, s18 // (op 547) 

s_sub_i32 s8, s1, s0 // (op 547) 


s_mul_hi_i32 s9, s16, s11 // (op 547) 
s_add_i32 s0, s9, s16 // (op 547) 

s_ashr_i32 s1, s0, 31 // (op 547) 
s_cmp_eq_i32 s11, 0 // (op 547) 
s_mov_b32 s9, scc // (op 547) 
s_cmp_eq_i32 s9, 0 // (op 547) 
s_cselect_b32 s49, 0, -1 // (op 547) 

s_and_b32 s9, s46, 31 // (op 547) 
s_lshl_b32 s50, 1, s9 // (op 547) 
s_add_i32 s51, s50, s49 // (op 547) 


s_and_b32 s49, s1, s51 // (op 547) 


s_add_i32 s1, s0, s49 // (op 547) 


s_ashr_i32 s0, s1, s9 // (op 547) 


s_lshl_b32 s1, s46, 24 // (op 547) 
s_ashr_i32 s9, s1, 31 // (op 547) 

s_xor_b32 s1, s0, s9 // (op 547) 

s_sub_i32 s0, s1, s9 // (op 547) 


s_mul_i32 s1, s45, s18 // (op 547) 

s_mul_i32 s9, s10, s1 // (op 547) 

s_mul_i32 s1, s0, s9 // (op 547) 

s_sub_u32 s9, s16, s1 // (op 547) 

s_mul_hi_i32 s1, s9, s44 // (op 547) 
s_add_i32 s18, s1, s9 // (op 547) 


s_ashr_i32 s1, s18, 31 // (op 547) 
s_cmp_eq_i32 s44, 0 // (op 547) 
s_mov_b32 s9, scc // (op 547) 
s_cmp_eq_i32 s9, 0 // (op 547) 
s_cselect_b32 s19, 0, -1 // (op 547) 

s_and_b32 s9, s48, 31 // (op 547) 
s_lshl_b32 s49, 1, s9 // (op 547) 
s_add_i32 s50, s49, s19 // (op 547) 


s_and_b32 s19, s1, s50 // (op 547) 


s_add_i32 s1, s18, s19 // (op 547) 


s_ashr_i32 s18, s1, s9 // (op 547) 


s_lshl_b32 s1, s48, 24 // (op 547) 
s_ashr_i32 s9, s1, 31 // (op 547) 

s_xor_b32 s1, s18, s9 // (op 547) 

s_sub_i32 s18, s1, s9 // (op 547) 


s_cmp_eq_i32 s0, 0 // (op 547) 
s_mov_b32 s1, scc // (op 547) 

s_cmp_eq_i32 s1, 0 // (op 547) 
s_cselect_b32 s0, s18, s8 // (op 547) 



v_lshl_add_u32 v11, s0, 5, v25 // (op 547) 


v_mul_lo_u32 v23, v11, s23 // (op 547) most significant: low of low * high
v_mul_hi_u32 v21, v11, s22 // (op 547) most significant: high of low * low
v_mul_lo_u32 v22, v11, s22 // (op 547) least significant: low of low * low
v_add_u32 v23, v23, v21 // (op 547) most significant: sum


s_ashr_i64 s[0:1], s[34:35], 7 // (op 547) 
s_lshr_b32 s8, s0, 1 // (op 547) 

v_lshl_add_u32 v11, s8, 8, v10 // (op 547) 


v_mov_b32 v15, 0 // (op 547) convert
v_mov_b32 v14, v11 // (op 547) convert

v_add_co_u32 v10, s[0:1], v14, v22 // (op 547) least significant half
v_addc_co_u32 v11, s[0:1], v15, v23, s[0:1] // (op 547) most significant half



s_mul_i32 s1, s22, s5 // (op 549) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 549) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 549) least significant
s_mul_hi_u32 s9, s22, s4 // (op 549) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 549) most significant: sum
s_add_u32 s1, s1, s9 // (op 549) most significant: sum


s_mov_b32 s56, s24 // (op 549) reg expression
s_mov_b32 s57, s25 // (op 549) reg expression
s_mov_b32 s58, s0 // (op 549) reg expression
s_mov_b32 s59, 131072 // (op 549) call()

s_mul_hi_i32 s0, s16, s12 // (op 611) 
v_lshrrev_b32 v25, 7, v1 // (op 611) 
s_add_i32 s1, s0, s16 // (op 611) 

v_and_b32 v14, 15, v1 // (op 611) 
s_ashr_i32 s0, s1, 31 // (op 611) 
v_lshl_add_u32 v15, v25, 4, v14 // (op 611) 


s_cmp_eq_i32 s12, 0 // (op 611) 
s_mov_b32 s8, scc // (op 611) 
v_and_b32 v14, 63, v1 // (op 611) 
s_cmp_eq_i32 s8, 0 // (op 611) 
s_cselect_b32 s9, 0, -1 // (op 611) 

v_lshrrev_b32 v25, 4, v14 // (op 611) 

s_and_b32 s8, s13, 31 // (op 611) 
s_lshl_b32 s18, 1, s8 // (op 611) 
s_add_i32 s19, s18, s9 // (op 611) 


s_and_b32 s9, s0, s19 // (op 611) 


s_add_i32 s0, s1, s9 // (op 611) 


s_ashr_i32 s1, s0, s8 // (op 611) 


s_lshl_b32 s0, s13, 24 // (op 611) 
s_ashr_i32 s8, s0, 31 // (op 611) 

s_xor_b32 s0, s1, s8 // (op 611) 

s_sub_i32 s1, s0, s8 // (op 611) 


s_mul_hi_i32 s0, s1, s14 // (op 611) 
s_add_i32 s8, s0, s1 // (op 611) 

s_ashr_i32 s0, s8, 31 // (op 611) 
s_cmp_eq_i32 s14, 0 // (op 611) 
s_mov_b32 s9, scc // (op 611) 
s_cmp_eq_i32 s9, 0 // (op 611) 
s_cselect_b32 s18, 0, -1 // (op 611) 

s_and_b32 s9, s15, 31 // (op 611) 
s_lshl_b32 s19, 1, s9 // (op 611) 
s_add_i32 s49, s19, s18 // (op 611) 


s_and_b32 s18, s0, s49 // (op 611) 


s_add_i32 s0, s8, s18 // (op 611) 


s_ashr_i32 s8, s0, s9 // (op 611) 


s_lshl_b32 s0, s15, 24 // (op 611) 
s_ashr_i32 s9, s0, 31 // (op 611) 

s_xor_b32 s0, s8, s9 // (op 611) 

s_sub_i32 s8, s0, s9 // (op 611) 


s_mul_i32 s0, s1, s45 // (op 611) 

s_sub_u32 s1, s16, s0 // (op 611) 

s_mul_i32 s0, s8, s45 // (op 611) 
s_add_u32 s0, s0, s1 // (op 611) 


s_mul_hi_i32 s1, s16, s11 // (op 611) 
s_add_i32 s8, s1, s16 // (op 611) 

s_ashr_i32 s1, s8, 31 // (op 611) 
s_cmp_eq_i32 s11, 0 // (op 611) 
s_mov_b32 s9, scc // (op 611) 
s_cmp_eq_i32 s9, 0 // (op 611) 
s_cselect_b32 s18, 0, -1 // (op 611) 

s_and_b32 s9, s46, 31 // (op 611) 
s_lshl_b32 s19, 1, s9 // (op 611) 
s_add_i32 s49, s19, s18 // (op 611) 


s_and_b32 s18, s1, s49 // (op 611) 


s_add_i32 s1, s8, s18 // (op 611) 


s_ashr_i32 s8, s1, s9 // (op 611) 


s_lshl_b32 s1, s46, 24 // (op 611) 
s_ashr_i32 s9, s1, 31 // (op 611) 

s_xor_b32 s1, s8, s9 // (op 611) 

s_sub_i32 s8, s1, s9 // (op 611) 


s_add_u32 s18, s4, 32 // (op 611) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 611) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 611) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 611) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 611) 

s_mul_i32 s1, s45, s18 // (op 611) 

s_mul_i32 s9, s10, s1 // (op 611) 

s_mul_i32 s1, s8, s9 // (op 611) 

s_sub_u32 s9, s16, s1 // (op 611) 

s_mov_b32 s1, s9 // (op 611) 
s_mul_hi_i32 s18, s1, s44 // (op 611) 
s_add_i32 s19, s18, s1 // (op 611) 


s_ashr_i32 s1, s19, 31 // (op 611) 
s_cmp_eq_i32 s44, 0 // (op 611) 
s_mov_b32 s18, scc // (op 611) 
s_cmp_eq_i32 s18, 0 // (op 611) 
s_cselect_b32 s49, 0, -1 // (op 611) 

s_and_b32 s18, s48, 31 // (op 611) 
s_lshl_b32 s50, 1, s18 // (op 611) 
s_add_i32 s51, s50, s49 // (op 611) 


s_and_b32 s49, s1, s51 // (op 611) 


s_add_i32 s1, s19, s49 // (op 611) 


s_ashr_i32 s19, s1, s18 // (op 611) 


s_lshl_b32 s1, s48, 24 // (op 611) 
s_ashr_i32 s18, s1, 31 // (op 611) 

s_xor_b32 s1, s19, s18 // (op 611) 

s_sub_i32 s19, s1, s18 // (op 611) 


s_mul_i32 s1, s47, s45 // (op 611) 
s_add_u32 s50, s6, 16 // (op 611) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 611) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 611) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 611) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 611) 

s_sub_i32 s18, s50, s1 // (op 611) 

s_mul_i32 s49, s19, s18 // (op 611) 


s_sub_u32 s18, s9, s49 // (op 611) 


s_add_u32 s9, s18, s1 // (op 611) 


s_cmp_eq_i32 s8, 0 // (op 611) 
s_mov_b32 s1, scc // (op 611) 

s_cmp_eq_i32 s1, 0 // (op 611) 
s_cselect_b32 s8, s9, s0 // (op 611) 



v_lshl_add_u32 v14, s8, 4, v15 // (op 611) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 611) 
s_lshr_b32 s8, s0, 1 // (op 611) 

v_lshl_add_u32 v15, s8, 3, v25 // (op 611) 


v_mul_lo_u32 v25, v14, s26 // (op 611) 
v_add_u32 v25, v25, v15 // (op 611) 


s_mul_i32 s1, s26, s7 // (op 613) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 613) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 613) least significant
s_mul_hi_u32 s9, s26, s6 // (op 613) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 613) most significant: sum
s_add_u32 s1, s1, s9 // (op 613) most significant: sum


s_mov_b32 s60, s28 // (op 613) reg expression
s_mov_b32 s61, s29 // (op 613) reg expression
s_mov_b32 s62, s0 // (op 613) reg expression
s_mov_b32 s63, 131072 // (op 613) call()

s_mul_hi_i32 s0, s16, s12 // (op 675) 
v_lshrrev_b32 v14, 2, v1 // (op 675) 
s_add_i32 s1, s0, s16 // (op 675) 

v_and_b32 v15, -4, v1 // (op 675) 
s_ashr_i32 s0, s1, 31 // (op 675) 
v_sub_u32 v16, v1, v15 // (op 675) 

s_cmp_eq_i32 s12, 0 // (op 675) 
s_mov_b32 s8, scc // (op 675) 
s_cmp_eq_i32 s8, 0 // (op 675) 
s_cselect_b32 s9, 0, -1 // (op 675) 

s_and_b32 s8, s13, 31 // (op 675) 
s_lshl_b32 s18, 1, s8 // (op 675) 
s_add_i32 s19, s18, s9 // (op 675) 


s_and_b32 s9, s0, s19 // (op 675) 


s_add_i32 s0, s1, s9 // (op 675) 


s_ashr_i32 s1, s0, s8 // (op 675) 


s_lshl_b32 s0, s13, 24 // (op 675) 
s_ashr_i32 s8, s0, 31 // (op 675) 

s_xor_b32 s0, s1, s8 // (op 675) 

s_sub_i32 s1, s0, s8 // (op 675) 


s_mul_hi_i32 s0, s1, s14 // (op 675) 
s_add_i32 s8, s0, s1 // (op 675) 

s_ashr_i32 s0, s8, 31 // (op 675) 
s_cmp_eq_i32 s14, 0 // (op 675) 
s_mov_b32 s9, scc // (op 675) 
s_cmp_eq_i32 s9, 0 // (op 675) 
s_cselect_b32 s18, 0, -1 // (op 675) 

s_and_b32 s9, s15, 31 // (op 675) 
s_lshl_b32 s19, 1, s9 // (op 675) 
s_add_i32 s49, s19, s18 // (op 675) 


s_and_b32 s18, s0, s49 // (op 675) 


s_add_i32 s0, s8, s18 // (op 675) 


s_ashr_i32 s8, s0, s9 // (op 675) 


s_lshl_b32 s0, s15, 24 // (op 675) 
s_ashr_i32 s9, s0, 31 // (op 675) 

s_xor_b32 s0, s8, s9 // (op 675) 

s_sub_i32 s8, s0, s9 // (op 675) 


s_add_u32 s18, s4, 32 // (op 675) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 675) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 675) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 675) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 675) 

s_mul_i32 s0, s8, s18 // (op 675) 

s_sub_i32 s8, s1, s0 // (op 675) 


s_mul_hi_i32 s9, s16, s11 // (op 675) 
s_add_i32 s0, s9, s16 // (op 675) 

s_ashr_i32 s1, s0, 31 // (op 675) 
s_cmp_eq_i32 s11, 0 // (op 675) 
s_mov_b32 s9, scc // (op 675) 
s_cmp_eq_i32 s9, 0 // (op 675) 
s_cselect_b32 s49, 0, -1 // (op 675) 

s_and_b32 s9, s46, 31 // (op 675) 
s_lshl_b32 s50, 1, s9 // (op 675) 
s_add_i32 s51, s50, s49 // (op 675) 


s_and_b32 s49, s1, s51 // (op 675) 


s_add_i32 s1, s0, s49 // (op 675) 


s_ashr_i32 s0, s1, s9 // (op 675) 


s_lshl_b32 s1, s46, 24 // (op 675) 
s_ashr_i32 s9, s1, 31 // (op 675) 

s_xor_b32 s1, s0, s9 // (op 675) 

s_sub_i32 s0, s1, s9 // (op 675) 


s_mul_i32 s1, s45, s18 // (op 675) 

s_mul_i32 s9, s10, s1 // (op 675) 

s_mul_i32 s1, s0, s9 // (op 675) 

s_sub_u32 s9, s16, s1 // (op 675) 

s_mul_hi_i32 s1, s9, s44 // (op 675) 
s_add_i32 s18, s1, s9 // (op 675) 


s_ashr_i32 s1, s18, 31 // (op 675) 
s_cmp_eq_i32 s44, 0 // (op 675) 
s_mov_b32 s9, scc // (op 675) 
s_cmp_eq_i32 s9, 0 // (op 675) 
s_cselect_b32 s19, 0, -1 // (op 675) 

s_and_b32 s9, s48, 31 // (op 675) 
s_lshl_b32 s49, 1, s9 // (op 675) 
s_add_i32 s50, s49, s19 // (op 675) 


s_and_b32 s19, s1, s50 // (op 675) 


s_add_i32 s1, s18, s19 // (op 675) 


s_ashr_i32 s18, s1, s9 // (op 675) 


s_lshl_b32 s1, s48, 24 // (op 675) 
s_ashr_i32 s9, s1, 31 // (op 675) 

s_xor_b32 s1, s18, s9 // (op 675) 

s_sub_i32 s18, s1, s9 // (op 675) 


s_cmp_eq_i32 s0, 0 // (op 675) 
s_mov_b32 s1, scc // (op 675) 

s_cmp_eq_i32 s1, 0 // (op 675) 
s_cselect_b32 s0, s18, s8 // (op 675) 



v_lshl_add_u32 v15, s0, 5, v14 // (op 675) 


v_mul_lo_u32 v14, v15, s30 // (op 675) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 675) 
s_lshr_b32 s8, s0, 1 // (op 675) 

v_lshl_add_u32 v15, s8, 3, v16 // (op 675) 


v_add_u32 v16, v15, v14 // (op 675) 


s_mul_i32 s1, s30, s5 // (op 677) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 677) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 677) least significant
s_mul_hi_u32 s9, s30, s4 // (op 677) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 677) most significant: sum
s_add_u32 s1, s1, s9 // (op 677) most significant: sum


s_mov_b32 s64, s32 // (op 677) reg expression
s_mov_b32 s65, s33 // (op 677) reg expression
s_mov_b32 s66, s0 // (op 677) reg expression
s_mov_b32 s67, 131072 // (op 677) call()

v_and_b32 v14, -8, v1 // (op 768) 
v_sub_u32 v15, v1, v14 // (op 768) 

v_lshlrev_b32 v14, 4, v15 // (op 768) 

v_lshrrev_b32 v15, 3, v1 // (op 768) 
v_lshl_add_u32 v17, v15, 7, v14 // (op 768) 


s_nop 0
v_readfirstlane_b32 s0, v17 // (op 768) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 820) 
v_sub_u32 v15, v1, v14 // (op 820) 

v_lshlrev_b32 v14, 4, v15 // (op 820) 

v_lshrrev_b32 v15, 3, v1 // (op 820) 
v_lshl_add_u32 v17, v15, 7, v14 // (op 820) 


s_nop 0
v_readfirstlane_b32 s1, v17 // (op 820) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 872) 
v_sub_u32 v15, v1, v14 // (op 872) 

v_and_b32 v14, 4294967292, v1 // (op 872) 
v_add_u32 v17, v14, v15 // (op 872) 


s_ashr_i64 s[8:9], s[34:35], 7 // (op 162) 
s_and_b32 s18, s8, 4294967294 // (op 162) 

s_ashr_i64 s[8:9], s[34:35], 7 // (op 161) 
s_mov_b32 s19, s8 // (op 161) convert

s_cmp_lt_u32 s18, s19 // (op 161) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161:
s_barrier  // (op 1075) 
s_add_u32 s18, s18, 1 // (op 163) 


v_mov_b64 v[14:15], v[12:13] // (op 432) 
s_add_u32 s19, 4096, s0 // (op 432) 

s_mov_b32 m0, s19 // (op 432) reg expression
buffer_load_dwordx4 v14, s[52:55], 0 offen offset: 0 lds // (op 432) Load value direct to lds


v_add_co_u32 v12, vcc, 128, v12 // (op 470) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 470) most significant half
s_barrier  // (op 1091) 

s_lshl_b64 s[8:9], s[22:23], 4 // (op 438) 
v_mov_b64 v[14:15], s[8:9] // (op 438) convert


v_mov_b64 v[18:19], v[10:11] // (op 438) 
v_mov_b64 v[20:21], v[18:19] // (op 438) 
s_mov_b32 m0, s1 // (op 438) reg expression
buffer_load_dwordx4 v20, s[56:59], 0 offen offset: 0 lds // (op 438) Load value direct to lds
v_add_co_u32 v20, s[8:9], v20, v14 // (op 438) least significant half
v_addc_co_u32 v21, s[8:9], 0, v21, s[8:9] // (op 438) most significant half

s_add_u32 m0, m0, 2048 // (op 438) 
buffer_load_dwordx4 v20, s[56:59], 0 offen offset: 0 lds // (op 438) Load value direct to lds



v_add_co_u32 v10, vcc, 128, v10 // (op 539) least significant half
v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 539) most significant half


v_mov_b32 v14, v16 // (op 180) 

buffer_load_ubyte v15, v14, s[64:67], 0 offen offset: 0 // (op 180) Load value

v_add_u32 v16, 4, v16 // (op 667) 
s_barrier  // (op 1107) 
s_waitcnt vmcnt(3)
s_barrier  // (op 1081) 






ds_read_b128 v[26:29], v3 offset:4096 // (op 169) Load local data 
ds_read_b128 v[30:33], v3 offset:4160 // (op 169) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v17, v15 offset:12288 // (op 178) Store local data 

s_barrier  // (op 1097) 



ds_read_b128 v[34:37], v4 // (op 171) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 171) Load local data 


v_mov_b32 v14, v25 // (op 173) 

buffer_load_ubyte v15, v14, s[60:63], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v25, 4, v25 // (op 603) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1114) 



ds_read_u8 v14, v0 offset:12288 // (op 175) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 176) 




s_ashr_i64 s[8:9], s[34:35], 7 // (op 161) 
s_mov_b32 s19, s8 // (op 161) convert

s_cmp_lt_u32 s18, s19 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_161 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_161:





















s_barrier  // (op 1166) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

s_mul_hi_i32 s0, s16, s12 // (op 914) 
v_lshrrev_b32 v0, 7, v1 // (op 914) 
s_add_i32 s1, s0, s16 // (op 914) 

v_and_b32 v3, 63, v1 // (op 914) 
s_ashr_i32 s0, s1, 31 // (op 914) 
v_lshrrev_b32 v4, 4, v3 // (op 914) 
s_cmp_eq_i32 s12, 0 // (op 914) 
s_mov_b32 s2, scc // (op 914) 
v_lshlrev_b32 v5, 2, v4 // (op 914) 

s_cmp_eq_i32 s2, 0 // (op 914) 
s_cselect_b32 s3, 0, -1 // (op 914) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 914) 


s_and_b32 s2, s13, 31 // (op 914) 
v_lshrrev_b32 v0, 2, v3 // (op 914) 

s_lshl_b32 s8, 1, s2 // (op 914) 
v_and_b32 v3, 3, v0 // (op 914) 

s_add_i32 s9, s8, s3 // (op 914) 


v_and_b32 v0, 3, v1 // (op 914) 
s_and_b32 s3, s0, s9 // (op 914) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 914) 


s_add_i32 s0, s1, s3 // (op 914) 


v_lshrrev_b32 v0, 6, v1 // (op 914) 
s_ashr_i32 s1, s0, s2 // (op 914) 


v_and_b32 v3, 1, v0 // (op 914) 

s_lshl_b32 s0, s13, 24 // (op 914) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 914) 


s_ashr_i32 s2, s0, 31 // (op 914) 

s_xor_b32 s0, s1, s2 // (op 914) 

s_sub_i32 s1, s0, s2 // (op 914) 


s_mul_hi_i32 s0, s1, s14 // (op 914) 
s_add_i32 s2, s0, s1 // (op 914) 

s_ashr_i32 s0, s2, 31 // (op 914) 
s_cmp_eq_i32 s14, 0 // (op 914) 
s_mov_b32 s3, scc // (op 914) 
s_cmp_eq_i32 s3, 0 // (op 914) 
s_cselect_b32 s8, 0, -1 // (op 914) 

s_and_b32 s3, s15, 31 // (op 914) 
s_lshl_b32 s9, 1, s3 // (op 914) 
s_add_i32 s18, s9, s8 // (op 914) 


s_and_b32 s8, s0, s18 // (op 914) 


s_add_i32 s0, s2, s8 // (op 914) 


s_ashr_i32 s2, s0, s3 // (op 914) 


s_lshl_b32 s0, s15, 24 // (op 914) 
s_ashr_i32 s3, s0, 31 // (op 914) 

s_xor_b32 s0, s2, s3 // (op 914) 

s_sub_i32 s2, s0, s3 // (op 914) 


s_mul_i32 s0, s1, s45 // (op 914) 
s_sub_u32 s3, s16, s0 // (op 914) 

s_mul_i32 s0, s2, s45 // (op 914) 
s_add_u32 s0, s0, s3 // (op 914) 

s_mul_hi_i32 s3, s16, s11 // (op 914) 
s_add_i32 s8, s3, s16 // (op 914) 

s_ashr_i32 s3, s8, 31 // (op 914) 
s_cmp_eq_i32 s11, 0 // (op 914) 
s_mov_b32 s9, scc // (op 914) 
s_cmp_eq_i32 s9, 0 // (op 914) 
s_cselect_b32 s18, 0, -1 // (op 914) 

s_and_b32 s9, s46, 31 // (op 914) 
s_lshl_b32 s19, 1, s9 // (op 914) 
s_add_i32 s20, s19, s18 // (op 914) 


s_and_b32 s18, s3, s20 // (op 914) 


s_add_i32 s3, s8, s18 // (op 914) 


s_ashr_i32 s8, s3, s9 // (op 914) 


s_lshl_b32 s3, s46, 24 // (op 914) 
s_ashr_i32 s9, s3, 31 // (op 914) 

s_xor_b32 s3, s8, s9 // (op 914) 

s_sub_i32 s8, s3, s9 // (op 914) 


s_add_u32 s18, s4, 32 // (op 914) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 914) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 914) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 914) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 914) 

s_mul_i32 s3, s45, s18 // (op 914) 
s_mul_i32 s9, s10, s3 // (op 914) 

s_mul_i32 s3, s8, s9 // (op 914) 

s_sub_u32 s9, s16, s3 // (op 914) 

s_mov_b32 s3, s9 // (op 914) 
s_mul_hi_i32 s20, s3, s44 // (op 914) 
s_add_i32 s21, s20, s3 // (op 914) 


s_ashr_i32 s3, s21, 31 // (op 914) 
s_cmp_eq_i32 s44, 0 // (op 914) 
s_mov_b32 s20, scc // (op 914) 
s_cmp_eq_i32 s20, 0 // (op 914) 
s_cselect_b32 s22, 0, -1 // (op 914) 

s_and_b32 s20, s48, 31 // (op 914) 
s_lshl_b32 s23, 1, s20 // (op 914) 
s_add_i32 s24, s23, s22 // (op 914) 


s_and_b32 s22, s3, s24 // (op 914) 


s_add_i32 s3, s21, s22 // (op 914) 


s_ashr_i32 s21, s3, s20 // (op 914) 


s_lshl_b32 s3, s48, 24 // (op 914) 
s_ashr_i32 s20, s3, 31 // (op 914) 

s_xor_b32 s3, s21, s20 // (op 914) 

s_sub_i32 s21, s3, s20 // (op 914) 


s_mul_i32 s3, s47, s45 // (op 914) 
s_add_u32 s22, s6, 16 // (op 914) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 914) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 914) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 914) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 914) 

s_sub_i32 s20, s22, s3 // (op 914) 

s_mul_i32 s22, s21, s20 // (op 914) 

s_sub_u32 s20, s9, s22 // (op 914) 


s_add_u32 s9, s20, s3 // (op 914) 


s_cmp_eq_i32 s8, 0 // (op 914) 
s_mov_b32 s3, scc // (op 914) 

s_cmp_eq_i32 s3, 0 // (op 914) 
s_cselect_b32 s8, s9, s0 // (op 914) 


s_mul_i32 s0, s2, s18 // (op 914) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 914) 


s_sub_i32 s2, s1, s0 // (op 914) 


s_cmp_eq_i32 s3, 0 // (op 914) 
s_cselect_b32 s0, s21, s2 // (op 914) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 914) 


v_mul_lo_u32 v0, v4, s36 // (op 914) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 914) 


s_mul_i32 s9, s36, s5 // (op 916) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 916) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 916) least significant
s_mul_hi_u32 s1, s36, s4 // (op 916) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 916) most significant: sum
s_add_u32 s9, s9, s1 // (op 916) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 916) 

s_mov_b32 s32, s38 // (op 916) reg expression
s_mov_b32 s33, s39 // (op 916) reg expression
s_mov_b32 s34, s0 // (op 916) reg expression
s_mov_b32 s35, 131072 // (op 916) call()




buffer_store_dwordx4 v[10:13], v4, s[32:35], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_123:
s_mul_hi_i32 s0, s16, s12 // (op 738) 
v_lshrrev_b32 v0, 7, v1 // (op 738) 
s_add_i32 s1, s0, s16 // (op 738) 

v_and_b32 v3, 63, v1 // (op 738) 
s_ashr_i32 s0, s1, 31 // (op 738) 
v_lshrrev_b32 v4, 4, v3 // (op 738) 
s_cmp_eq_i32 s12, 0 // (op 738) 
s_mov_b32 s2, scc // (op 738) 
v_lshlrev_b32 v5, 2, v4 // (op 738) 

s_cmp_eq_i32 s2, 0 // (op 738) 
s_cselect_b32 s3, 0, -1 // (op 738) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 738) 


s_and_b32 s2, s13, 31 // (op 738) 
v_lshrrev_b32 v0, 2, v3 // (op 738) 

s_lshl_b32 s8, 1, s2 // (op 738) 
v_and_b32 v3, 3, v0 // (op 738) 

s_add_i32 s9, s8, s3 // (op 738) 


v_and_b32 v0, 3, v1 // (op 738) 
s_and_b32 s3, s0, s9 // (op 738) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 738) 


s_add_i32 s0, s1, s3 // (op 738) 


v_lshrrev_b32 v0, 6, v1 // (op 738) 
s_ashr_i32 s1, s0, s2 // (op 738) 


v_and_b32 v3, 1, v0 // (op 738) 

s_lshl_b32 s0, s13, 24 // (op 738) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 738) 


s_ashr_i32 s2, s0, 31 // (op 738) 

s_xor_b32 s0, s1, s2 // (op 738) 

s_sub_i32 s1, s0, s2 // (op 738) 


s_mul_hi_i32 s0, s1, s14 // (op 738) 
s_add_i32 s2, s0, s1 // (op 738) 

s_ashr_i32 s0, s2, 31 // (op 738) 
s_cmp_eq_i32 s14, 0 // (op 738) 
s_mov_b32 s3, scc // (op 738) 
s_cmp_eq_i32 s3, 0 // (op 738) 
s_cselect_b32 s8, 0, -1 // (op 738) 

s_and_b32 s3, s15, 31 // (op 738) 
s_lshl_b32 s9, 1, s3 // (op 738) 
s_add_i32 s18, s9, s8 // (op 738) 


s_and_b32 s8, s0, s18 // (op 738) 


s_add_i32 s0, s2, s8 // (op 738) 


s_ashr_i32 s2, s0, s3 // (op 738) 


s_lshl_b32 s0, s15, 24 // (op 738) 
s_ashr_i32 s3, s0, 31 // (op 738) 

s_xor_b32 s0, s2, s3 // (op 738) 

s_sub_i32 s2, s0, s3 // (op 738) 


s_mul_i32 s0, s1, s45 // (op 738) 
s_sub_u32 s3, s16, s0 // (op 738) 

s_mul_i32 s0, s2, s45 // (op 738) 
s_add_u32 s0, s0, s3 // (op 738) 

s_mul_hi_i32 s3, s16, s11 // (op 738) 
s_add_i32 s8, s3, s16 // (op 738) 

s_ashr_i32 s3, s8, 31 // (op 738) 
s_cmp_eq_i32 s11, 0 // (op 738) 
s_mov_b32 s9, scc // (op 738) 
s_cmp_eq_i32 s9, 0 // (op 738) 
s_cselect_b32 s18, 0, -1 // (op 738) 

s_and_b32 s9, s46, 31 // (op 738) 
s_lshl_b32 s19, 1, s9 // (op 738) 
s_add_i32 s20, s19, s18 // (op 738) 


s_and_b32 s18, s3, s20 // (op 738) 


s_add_i32 s3, s8, s18 // (op 738) 


s_ashr_i32 s8, s3, s9 // (op 738) 


s_lshl_b32 s3, s46, 24 // (op 738) 
s_ashr_i32 s9, s3, 31 // (op 738) 

s_xor_b32 s3, s8, s9 // (op 738) 

s_sub_i32 s8, s3, s9 // (op 738) 


s_add_u32 s18, s4, 32 // (op 738) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 738) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 738) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 738) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 738) 

s_mul_i32 s3, s45, s18 // (op 738) 
s_mul_i32 s9, s10, s3 // (op 738) 

s_mul_i32 s3, s8, s9 // (op 738) 

s_sub_u32 s9, s16, s3 // (op 738) 

s_mov_b32 s3, s9 // (op 738) 
s_mul_hi_i32 s20, s3, s44 // (op 738) 
s_add_i32 s21, s20, s3 // (op 738) 


s_ashr_i32 s3, s21, 31 // (op 738) 
s_cmp_eq_i32 s44, 0 // (op 738) 
s_mov_b32 s20, scc // (op 738) 
s_cmp_eq_i32 s20, 0 // (op 738) 
s_cselect_b32 s22, 0, -1 // (op 738) 

s_and_b32 s20, s48, 31 // (op 738) 
s_lshl_b32 s23, 1, s20 // (op 738) 
s_add_i32 s24, s23, s22 // (op 738) 


s_and_b32 s22, s3, s24 // (op 738) 


s_add_i32 s3, s21, s22 // (op 738) 


s_ashr_i32 s21, s3, s20 // (op 738) 


s_lshl_b32 s3, s48, 24 // (op 738) 
s_ashr_i32 s20, s3, 31 // (op 738) 

s_xor_b32 s3, s21, s20 // (op 738) 

s_sub_i32 s21, s3, s20 // (op 738) 


s_mul_i32 s3, s47, s45 // (op 738) 
s_add_u32 s22, s6, 16 // (op 738) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 738) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 738) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 738) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 738) 

s_sub_i32 s20, s22, s3 // (op 738) 

s_mul_i32 s22, s21, s20 // (op 738) 

s_sub_u32 s20, s9, s22 // (op 738) 


s_add_u32 s9, s20, s3 // (op 738) 


s_cmp_eq_i32 s8, 0 // (op 738) 
s_mov_b32 s3, scc // (op 738) 

s_cmp_eq_i32 s3, 0 // (op 738) 
s_cselect_b32 s8, s9, s0 // (op 738) 


s_mul_i32 s0, s2, s18 // (op 738) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 738) 


s_sub_i32 s2, s1, s0 // (op 738) 


s_cmp_eq_i32 s3, 0 // (op 738) 
s_cselect_b32 s0, s21, s2 // (op 738) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 738) 


v_mul_lo_u32 v0, v4, s40 // (op 738) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 738) 


s_mul_i32 s9, s40, s5 // (op 740) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 740) most significant: low of high * low
s_mul_i32 s8, s40, s4 // (op 740) least significant
s_mul_hi_u32 s1, s40, s4 // (op 740) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 740) most significant: sum
s_add_u32 s9, s9, s1 // (op 740) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 740) 

s_mov_b32 s28, s42 // (op 740) reg expression
s_mov_b32 s29, s43 // (op 740) reg expression
s_mov_b32 s30, s0 // (op 740) reg expression
s_mov_b32 s31, 131072 // (op 740) call()





buffer_load_dwordx4 v[10:13], v4, s[28:31], 0 offen // (op 135) Load value



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


s_mul_hi_i32 s0, s16, s12 // (op 931) 
v_lshrrev_b32 v0, 7, v1 // (op 931) 
s_add_i32 s1, s0, s16 // (op 931) 

v_and_b32 v3, 63, v1 // (op 931) 
s_ashr_i32 s0, s1, 31 // (op 931) 
v_lshrrev_b32 v4, 4, v3 // (op 931) 
s_cmp_eq_i32 s12, 0 // (op 931) 
s_mov_b32 s2, scc // (op 931) 
v_lshlrev_b32 v5, 2, v4 // (op 931) 

s_cmp_eq_i32 s2, 0 // (op 931) 
s_cselect_b32 s3, 0, -1 // (op 931) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 931) 


s_and_b32 s2, s13, 31 // (op 931) 
v_lshrrev_b32 v0, 2, v3 // (op 931) 

s_lshl_b32 s8, 1, s2 // (op 931) 
v_and_b32 v3, 3, v0 // (op 931) 

s_add_i32 s9, s8, s3 // (op 931) 


v_and_b32 v0, 3, v1 // (op 931) 
s_and_b32 s3, s0, s9 // (op 931) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 931) 


s_add_i32 s0, s1, s3 // (op 931) 


v_lshrrev_b32 v0, 6, v1 // (op 931) 
s_ashr_i32 s1, s0, s2 // (op 931) 


v_and_b32 v3, 1, v0 // (op 931) 

s_lshl_b32 s0, s13, 24 // (op 931) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 931) 


s_ashr_i32 s2, s0, 31 // (op 931) 

s_xor_b32 s0, s1, s2 // (op 931) 

s_sub_i32 s1, s0, s2 // (op 931) 


s_mul_hi_i32 s0, s1, s14 // (op 931) 
s_add_i32 s2, s0, s1 // (op 931) 

s_ashr_i32 s0, s2, 31 // (op 931) 
s_cmp_eq_i32 s14, 0 // (op 931) 
s_mov_b32 s3, scc // (op 931) 
s_cmp_eq_i32 s3, 0 // (op 931) 
s_cselect_b32 s8, 0, -1 // (op 931) 

s_and_b32 s3, s15, 31 // (op 931) 
s_lshl_b32 s9, 1, s3 // (op 931) 
s_add_i32 s18, s9, s8 // (op 931) 


s_and_b32 s8, s0, s18 // (op 931) 


s_add_i32 s0, s2, s8 // (op 931) 


s_ashr_i32 s2, s0, s3 // (op 931) 


s_lshl_b32 s0, s15, 24 // (op 931) 
s_ashr_i32 s3, s0, 31 // (op 931) 

s_xor_b32 s0, s2, s3 // (op 931) 

s_sub_i32 s2, s0, s3 // (op 931) 


s_mul_i32 s0, s1, s45 // (op 931) 
s_sub_u32 s3, s16, s0 // (op 931) 

s_mul_i32 s0, s2, s45 // (op 931) 
s_add_u32 s0, s0, s3 // (op 931) 

s_mul_hi_i32 s3, s16, s11 // (op 931) 
s_add_i32 s8, s3, s16 // (op 931) 

s_ashr_i32 s3, s8, 31 // (op 931) 
s_cmp_eq_i32 s11, 0 // (op 931) 
s_mov_b32 s9, scc // (op 931) 
s_cmp_eq_i32 s9, 0 // (op 931) 
s_cselect_b32 s18, 0, -1 // (op 931) 

s_and_b32 s9, s46, 31 // (op 931) 
s_lshl_b32 s19, 1, s9 // (op 931) 
s_add_i32 s20, s19, s18 // (op 931) 


s_and_b32 s18, s3, s20 // (op 931) 


s_add_i32 s3, s8, s18 // (op 931) 


s_ashr_i32 s8, s3, s9 // (op 931) 


s_lshl_b32 s3, s46, 24 // (op 931) 
s_ashr_i32 s9, s3, 31 // (op 931) 

s_xor_b32 s3, s8, s9 // (op 931) 

s_sub_i32 s8, s3, s9 // (op 931) 


s_add_u32 s18, s4, 32 // (op 931) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 931) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 931) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 931) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 931) 

s_mul_i32 s3, s45, s18 // (op 931) 
s_mul_i32 s9, s10, s3 // (op 931) 

s_mul_i32 s3, s8, s9 // (op 931) 

s_sub_u32 s9, s16, s3 // (op 931) 

s_mov_b32 s3, s9 // (op 931) 
s_mul_hi_i32 s20, s3, s44 // (op 931) 
s_add_i32 s21, s20, s3 // (op 931) 


s_ashr_i32 s3, s21, 31 // (op 931) 
s_cmp_eq_i32 s44, 0 // (op 931) 
s_mov_b32 s20, scc // (op 931) 
s_cmp_eq_i32 s20, 0 // (op 931) 
s_cselect_b32 s22, 0, -1 // (op 931) 

s_and_b32 s20, s48, 31 // (op 931) 
s_lshl_b32 s23, 1, s20 // (op 931) 
s_add_i32 s24, s23, s22 // (op 931) 


s_and_b32 s22, s3, s24 // (op 931) 


s_add_i32 s3, s21, s22 // (op 931) 


s_ashr_i32 s21, s3, s20 // (op 931) 


s_lshl_b32 s3, s48, 24 // (op 931) 
s_ashr_i32 s20, s3, 31 // (op 931) 

s_xor_b32 s3, s21, s20 // (op 931) 

s_sub_i32 s21, s3, s20 // (op 931) 


s_mul_i32 s3, s47, s45 // (op 931) 
s_add_u32 s22, s6, 16 // (op 931) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 931) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 931) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 931) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 931) 

s_sub_i32 s20, s22, s3 // (op 931) 

s_mul_i32 s22, s21, s20 // (op 931) 

s_sub_u32 s20, s9, s22 // (op 931) 


s_add_u32 s9, s20, s3 // (op 931) 


s_cmp_eq_i32 s8, 0 // (op 931) 
s_mov_b32 s3, scc // (op 931) 

s_cmp_eq_i32 s3, 0 // (op 931) 
s_cselect_b32 s8, s9, s0 // (op 931) 


s_mul_i32 s0, s2, s18 // (op 931) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 931) 


s_sub_i32 s2, s1, s0 // (op 931) 


s_cmp_eq_i32 s3, 0 // (op 931) 
s_cselect_b32 s0, s21, s2 // (op 931) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 931) 


v_mul_lo_u32 v0, v4, s36 // (op 931) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 931) 


s_mul_i32 s9, s36, s5 // (op 933) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 933) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 933) least significant
s_mul_hi_u32 s1, s36, s4 // (op 933) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 933) most significant: sum
s_add_u32 s9, s9, s1 // (op 933) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 933) 

s_mov_b32 s32, s38 // (op 933) reg expression
s_mov_b32 s33, s39 // (op 933) reg expression
s_mov_b32 s34, s0 // (op 933) reg expression
s_mov_b32 s35, 131072 // (op 933) call()




buffer_store_dwordx4 v[10:13], v4, s[32:35], 0 offen // (op 134) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 56
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12544 // lds bytes
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
  ProblemType:
    OperationType: GEMM
    DataType: F8
    DataTypeA: F8
    DataTypeB: F8
    DestDataType: s
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
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 12544
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 70
    .vgpr_count: 50
    .agpr_count: 4
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 36
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}}
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

