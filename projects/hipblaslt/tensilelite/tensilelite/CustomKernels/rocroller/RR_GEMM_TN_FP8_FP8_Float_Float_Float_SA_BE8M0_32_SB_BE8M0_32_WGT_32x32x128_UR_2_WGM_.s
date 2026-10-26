

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 759) 
v_and_b32 v3, 15, v1 // (op 759) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 759) 


v_and_b32 v0, 48, v1 // (op 759) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 759) 


v_lshrrev_b32 v0, 6, v1 // (op 811) 
v_and_b32 v4, 1, v0 // (op 811) 

v_and_b32 v0, 15, v1 // (op 811) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 811) 


v_and_b32 v0, 48, v1 // (op 811) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 811) 


v_lshrrev_b32 v0, 7, v1 // (op 884) 
v_and_b32 v5, 15, v1 // (op 884) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 884) 


v_and_b32 v0, 48, v1 // (op 884) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 884) 


v_lshrrev_b32 v0, 6, v1 // (op 923) 
v_and_b32 v6, 1, v0 // (op 923) 

v_and_b32 v0, 15, v1 // (op 923) 
v_lshl_add_u32 v7, v6, 4, v0 // (op 923) 


v_and_b32 v0, 48, v1 // (op 923) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 923) 


v_mov_b32 v0, s8 // (op 2) Move value
v_mov_b32 v7, s9 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
s_mul_hi_i32 s0, s16, s12 // (op 470) 
v_lshrrev_b32 v8, 3, v1 // (op 470) 
s_add_i32 s1, s0, s16 // (op 470) 

v_and_b32 v9, -8, v1 // (op 470) 
s_ashr_i32 s0, s1, 31 // (op 470) 
v_sub_u32 v10, v1, v9 // (op 470) 

s_cmp_eq_i32 s12, 0 // (op 470) 
s_mov_b32 s8, scc // (op 470) 
v_lshlrev_b32 v9, 4, v10 // (op 470) 

s_cmp_eq_i32 s8, 0 // (op 470) 
s_cselect_b32 s9, 0, -1 // (op 470) 

v_mov_b32 v11, 0 // (op 470) convert
v_mov_b32 v10, v9 // (op 470) convert

s_and_b32 s8, s13, 31 // (op 470) 
s_lshl_b32 s18, 1, s8 // (op 470) 
s_add_i32 s19, s18, s9 // (op 470) 


s_and_b32 s9, s0, s19 // (op 470) 


s_add_i32 s0, s1, s9 // (op 470) 


s_ashr_i32 s1, s0, s8 // (op 470) 


s_lshl_b32 s0, s13, 24 // (op 470) 
s_ashr_i32 s8, s0, 31 // (op 470) 

s_xor_b32 s0, s1, s8 // (op 470) 

s_sub_i32 s1, s0, s8 // (op 470) 


s_mul_hi_i32 s0, s1, s14 // (op 470) 
s_add_i32 s8, s0, s1 // (op 470) 

s_ashr_i32 s0, s8, 31 // (op 470) 
s_cmp_eq_i32 s14, 0 // (op 470) 
s_mov_b32 s9, scc // (op 470) 
s_cmp_eq_i32 s9, 0 // (op 470) 
s_cselect_b32 s18, 0, -1 // (op 470) 

s_and_b32 s9, s15, 31 // (op 470) 
s_lshl_b32 s19, 1, s9 // (op 470) 
s_add_i32 s49, s19, s18 // (op 470) 


s_and_b32 s18, s0, s49 // (op 470) 


s_add_i32 s0, s8, s18 // (op 470) 


s_ashr_i32 s8, s0, s9 // (op 470) 


s_lshl_b32 s0, s15, 24 // (op 470) 
s_ashr_i32 s9, s0, 31 // (op 470) 

s_xor_b32 s0, s8, s9 // (op 470) 

s_sub_i32 s8, s0, s9 // (op 470) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s45 // (op 470) 

s_sub_u32 s1, s16, s0 // (op 470) 

s_mul_i32 s0, s8, s45 // (op 470) 
s_add_u32 s0, s0, s1 // (op 470) 


s_mul_hi_i32 s1, s16, s11 // (op 470) 
s_add_i32 s8, s1, s16 // (op 470) 

s_ashr_i32 s1, s8, 31 // (op 470) 
s_cmp_eq_i32 s11, 0 // (op 470) 
s_mov_b32 s9, scc // (op 470) 
s_cmp_eq_i32 s9, 0 // (op 470) 
s_cselect_b32 s18, 0, -1 // (op 470) 

s_and_b32 s9, s46, 31 // (op 470) 
s_lshl_b32 s19, 1, s9 // (op 470) 
s_add_i32 s49, s19, s18 // (op 470) 


s_and_b32 s18, s1, s49 // (op 470) 


s_add_i32 s1, s8, s18 // (op 470) 


s_ashr_i32 s8, s1, s9 // (op 470) 


s_lshl_b32 s1, s46, 24 // (op 470) 
s_ashr_i32 s9, s1, 31 // (op 470) 

s_xor_b32 s1, s8, s9 // (op 470) 

s_sub_i32 s8, s1, s9 // (op 470) 


s_add_u32 s18, s4, 32 // (op 470) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 470) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 470) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 470) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 470) 

s_mul_i32 s1, s45, s18 // (op 470) 

s_mul_i32 s9, s10, s1 // (op 470) 

s_mul_i32 s1, s8, s9 // (op 470) 

s_sub_u32 s9, s16, s1 // (op 470) 

s_mov_b32 s1, s9 // (op 470) 
s_mul_hi_i32 s18, s1, s44 // (op 470) 
s_add_i32 s19, s18, s1 // (op 470) 


s_ashr_i32 s1, s19, 31 // (op 470) 
s_cmp_eq_i32 s44, 0 // (op 470) 
s_mov_b32 s18, scc // (op 470) 
s_cmp_eq_i32 s18, 0 // (op 470) 
s_cselect_b32 s49, 0, -1 // (op 470) 

s_and_b32 s18, s48, 31 // (op 470) 
s_lshl_b32 s50, 1, s18 // (op 470) 
s_add_i32 s51, s50, s49 // (op 470) 


s_and_b32 s49, s1, s51 // (op 470) 


s_add_i32 s1, s19, s49 // (op 470) 


s_ashr_i32 s19, s1, s18 // (op 470) 


s_lshl_b32 s1, s48, 24 // (op 470) 
s_ashr_i32 s18, s1, 31 // (op 470) 

s_xor_b32 s1, s19, s18 // (op 470) 

s_sub_i32 s19, s1, s18 // (op 470) 


s_mul_i32 s1, s47, s45 // (op 470) 
s_add_u32 s50, s6, 32 // (op 470) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 470) End of Int64 add, unlocking SCC
s_sub_u32 s52, s50, 1 // (op 470) least significant half; sets scc
s_subb_u32 s53, s51, 0 // (op 470) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 470) 

s_sub_i32 s18, s50, s1 // (op 470) 

s_mul_i32 s49, s19, s18 // (op 470) 


s_sub_u32 s18, s9, s49 // (op 470) 


s_add_u32 s9, s18, s1 // (op 470) 


s_cmp_eq_i32 s8, 0 // (op 470) 
s_mov_b32 s1, scc // (op 470) 

s_cmp_eq_i32 s1, 0 // (op 470) 
s_cselect_b32 s8, s9, s0 // (op 470) 



v_lshl_add_u32 v9, s8, 5, v8 // (op 470) 


v_mul_lo_u32 v13, v9, s3 // (op 470) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 470) most significant: high of low * low
v_mul_lo_u32 v12, v9, s2 // (op 470) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 470) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v10 // (op 470) least significant half
v_addc_co_u32 v13, s[0:1], v13, v11, s[0:1] // (op 470) most significant half



s_mul_i32 s1, s2, s7 // (op 472) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 472) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 472) least significant
s_mul_hi_u32 s9, s2, s6 // (op 472) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 472) most significant: sum
s_add_u32 s1, s1, s9 // (op 472) most significant: sum


s_mov_b32 s52, s20 // (op 472) reg expression
s_mov_b32 s53, s21 // (op 472) reg expression
s_mov_b32 s54, s0 // (op 472) reg expression
s_mov_b32 s55, 131072 // (op 472) call()

s_mul_hi_i32 s0, s16, s12 // (op 489) 
v_lshrrev_b32 v8, 3, v1 // (op 489) 
s_add_i32 s1, s0, s16 // (op 489) 

v_and_b32 v9, -8, v1 // (op 489) 
s_ashr_i32 s0, s1, 31 // (op 489) 
v_sub_u32 v10, v1, v9 // (op 489) 

s_cmp_eq_i32 s12, 0 // (op 489) 
s_mov_b32 s8, scc // (op 489) 
v_lshlrev_b32 v9, 4, v10 // (op 489) 

s_cmp_eq_i32 s8, 0 // (op 489) 
s_cselect_b32 s9, 0, -1 // (op 489) 

v_mov_b32 v11, 0 // (op 489) convert
v_mov_b32 v10, v9 // (op 489) convert

s_and_b32 s8, s13, 31 // (op 489) 
s_lshl_b32 s18, 1, s8 // (op 489) 
s_add_i32 s19, s18, s9 // (op 489) 


s_and_b32 s9, s0, s19 // (op 489) 


s_add_i32 s0, s1, s9 // (op 489) 


s_ashr_i32 s1, s0, s8 // (op 489) 


s_lshl_b32 s0, s13, 24 // (op 489) 
s_ashr_i32 s8, s0, 31 // (op 489) 

s_xor_b32 s0, s1, s8 // (op 489) 

s_sub_i32 s1, s0, s8 // (op 489) 


s_mul_hi_i32 s0, s1, s14 // (op 489) 
s_add_i32 s8, s0, s1 // (op 489) 

s_ashr_i32 s0, s8, 31 // (op 489) 
s_cmp_eq_i32 s14, 0 // (op 489) 
s_mov_b32 s9, scc // (op 489) 
s_cmp_eq_i32 s9, 0 // (op 489) 
s_cselect_b32 s18, 0, -1 // (op 489) 

s_and_b32 s9, s15, 31 // (op 489) 
s_lshl_b32 s19, 1, s9 // (op 489) 
s_add_i32 s49, s19, s18 // (op 489) 


s_and_b32 s18, s0, s49 // (op 489) 


s_add_i32 s0, s8, s18 // (op 489) 


s_ashr_i32 s8, s0, s9 // (op 489) 


s_lshl_b32 s0, s15, 24 // (op 489) 
s_ashr_i32 s9, s0, 31 // (op 489) 

s_xor_b32 s0, s8, s9 // (op 489) 

s_sub_i32 s8, s0, s9 // (op 489) 


s_mul_i32 s0, s1, s45 // (op 489) 

s_sub_u32 s1, s16, s0 // (op 489) 

s_mul_i32 s0, s8, s45 // (op 489) 
s_add_u32 s0, s0, s1 // (op 489) 


s_mul_hi_i32 s1, s16, s11 // (op 489) 
s_add_i32 s8, s1, s16 // (op 489) 

s_ashr_i32 s1, s8, 31 // (op 489) 
s_cmp_eq_i32 s11, 0 // (op 489) 
s_mov_b32 s9, scc // (op 489) 
s_cmp_eq_i32 s9, 0 // (op 489) 
s_cselect_b32 s18, 0, -1 // (op 489) 

s_and_b32 s9, s46, 31 // (op 489) 
s_lshl_b32 s19, 1, s9 // (op 489) 
s_add_i32 s49, s19, s18 // (op 489) 


s_and_b32 s18, s1, s49 // (op 489) 


s_add_i32 s1, s8, s18 // (op 489) 


s_ashr_i32 s8, s1, s9 // (op 489) 


s_lshl_b32 s1, s46, 24 // (op 489) 
s_ashr_i32 s9, s1, 31 // (op 489) 

s_xor_b32 s1, s8, s9 // (op 489) 

s_sub_i32 s8, s1, s9 // (op 489) 


s_add_u32 s18, s4, 32 // (op 489) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 489) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 489) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 489) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 489) 

s_mul_i32 s1, s45, s18 // (op 489) 

s_mul_i32 s9, s10, s1 // (op 489) 

s_mul_i32 s1, s8, s9 // (op 489) 

s_sub_u32 s9, s16, s1 // (op 489) 

s_mov_b32 s1, s9 // (op 489) 
s_mul_hi_i32 s18, s1, s44 // (op 489) 
s_add_i32 s19, s18, s1 // (op 489) 


s_ashr_i32 s1, s19, 31 // (op 489) 
s_cmp_eq_i32 s44, 0 // (op 489) 
s_mov_b32 s18, scc // (op 489) 
s_cmp_eq_i32 s18, 0 // (op 489) 
s_cselect_b32 s49, 0, -1 // (op 489) 

s_and_b32 s18, s48, 31 // (op 489) 
s_lshl_b32 s50, 1, s18 // (op 489) 
s_add_i32 s51, s50, s49 // (op 489) 


s_and_b32 s49, s1, s51 // (op 489) 


s_add_i32 s1, s19, s49 // (op 489) 


s_ashr_i32 s19, s1, s18 // (op 489) 


s_lshl_b32 s1, s48, 24 // (op 489) 
s_ashr_i32 s18, s1, 31 // (op 489) 

s_xor_b32 s1, s19, s18 // (op 489) 

s_sub_i32 s19, s1, s18 // (op 489) 


s_mul_i32 s1, s47, s45 // (op 489) 
s_add_u32 s50, s6, 32 // (op 489) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 489) End of Int64 add, unlocking SCC
s_sub_u32 s56, s50, 1 // (op 489) least significant half; sets scc
s_subb_u32 s57, s51, 0 // (op 489) most significant half; uses scc

s_ashr_i64 s[50:51], s[56:57], 5 // (op 489) 

s_sub_i32 s18, s50, s1 // (op 489) 

s_mul_i32 s49, s19, s18 // (op 489) 


s_sub_u32 s18, s9, s49 // (op 489) 


s_add_u32 s9, s18, s1 // (op 489) 


s_cmp_eq_i32 s8, 0 // (op 489) 
s_mov_b32 s1, scc // (op 489) 

s_cmp_eq_i32 s1, 0 // (op 489) 
s_cselect_b32 s8, s9, s0 // (op 489) 



v_lshl_add_u32 v9, s8, 5, v8 // (op 489) 


v_mul_lo_u32 v15, v9, s3 // (op 489) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 489) most significant: high of low * low
v_mul_lo_u32 v14, v9, s2 // (op 489) least significant: low of low * low
v_add_u32 v15, v15, v8 // (op 489) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v10 // (op 489) least significant half
v_addc_co_u32 v15, s[0:1], v15, v11, s[0:1] // (op 489) most significant half



s_mul_i32 s1, s2, s7 // (op 491) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 491) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 491) least significant
s_mul_hi_u32 s9, s2, s6 // (op 491) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 491) most significant: sum
s_add_u32 s1, s1, s9 // (op 491) most significant: sum


s_mov_b32 s52, s20 // (op 491) reg expression
s_mov_b32 s53, s21 // (op 491) reg expression
s_mov_b32 s54, s0 // (op 491) reg expression
s_mov_b32 s55, 131072 // (op 491) call()

s_mul_hi_i32 s0, s16, s12 // (op 534) 
v_lshrrev_b32 v8, 3, v1 // (op 534) 
s_add_i32 s1, s0, s16 // (op 534) 

v_and_b32 v9, -8, v1 // (op 534) 
s_ashr_i32 s0, s1, 31 // (op 534) 
v_sub_u32 v10, v1, v9 // (op 534) 

s_cmp_eq_i32 s12, 0 // (op 534) 
s_mov_b32 s8, scc // (op 534) 
v_lshlrev_b32 v9, 4, v10 // (op 534) 

s_cmp_eq_i32 s8, 0 // (op 534) 
s_cselect_b32 s9, 0, -1 // (op 534) 

v_mov_b32 v11, 0 // (op 534) convert
v_mov_b32 v10, v9 // (op 534) convert

s_and_b32 s8, s13, 31 // (op 534) 
s_lshl_b32 s18, 1, s8 // (op 534) 
s_add_i32 s19, s18, s9 // (op 534) 


s_and_b32 s9, s0, s19 // (op 534) 


s_add_i32 s0, s1, s9 // (op 534) 


s_ashr_i32 s1, s0, s8 // (op 534) 


s_lshl_b32 s0, s13, 24 // (op 534) 
s_ashr_i32 s8, s0, 31 // (op 534) 

s_xor_b32 s0, s1, s8 // (op 534) 

s_sub_i32 s1, s0, s8 // (op 534) 


s_mul_hi_i32 s0, s1, s14 // (op 534) 
s_add_i32 s8, s0, s1 // (op 534) 

s_ashr_i32 s0, s8, 31 // (op 534) 
s_cmp_eq_i32 s14, 0 // (op 534) 
s_mov_b32 s9, scc // (op 534) 
s_cmp_eq_i32 s9, 0 // (op 534) 
s_cselect_b32 s18, 0, -1 // (op 534) 

s_and_b32 s9, s15, 31 // (op 534) 
s_lshl_b32 s19, 1, s9 // (op 534) 
s_add_i32 s49, s19, s18 // (op 534) 


s_and_b32 s18, s0, s49 // (op 534) 


s_add_i32 s0, s8, s18 // (op 534) 


s_ashr_i32 s8, s0, s9 // (op 534) 


s_lshl_b32 s0, s15, 24 // (op 534) 
s_ashr_i32 s9, s0, 31 // (op 534) 

s_xor_b32 s0, s8, s9 // (op 534) 

s_sub_i32 s8, s0, s9 // (op 534) 


s_add_u32 s18, s4, 32 // (op 534) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 534) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 534) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 534) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 534) 

s_mul_i32 s0, s8, s18 // (op 534) 

s_sub_i32 s8, s1, s0 // (op 534) 


s_mul_hi_i32 s9, s16, s11 // (op 534) 
s_add_i32 s0, s9, s16 // (op 534) 

s_ashr_i32 s1, s0, 31 // (op 534) 
s_cmp_eq_i32 s11, 0 // (op 534) 
s_mov_b32 s9, scc // (op 534) 
s_cmp_eq_i32 s9, 0 // (op 534) 
s_cselect_b32 s49, 0, -1 // (op 534) 

s_and_b32 s9, s46, 31 // (op 534) 
s_lshl_b32 s50, 1, s9 // (op 534) 
s_add_i32 s51, s50, s49 // (op 534) 


s_and_b32 s49, s1, s51 // (op 534) 


s_add_i32 s1, s0, s49 // (op 534) 


s_ashr_i32 s0, s1, s9 // (op 534) 


s_lshl_b32 s1, s46, 24 // (op 534) 
s_ashr_i32 s9, s1, 31 // (op 534) 

s_xor_b32 s1, s0, s9 // (op 534) 

s_sub_i32 s0, s1, s9 // (op 534) 


s_mul_i32 s1, s45, s18 // (op 534) 

s_mul_i32 s9, s10, s1 // (op 534) 

s_mul_i32 s1, s0, s9 // (op 534) 

s_sub_u32 s9, s16, s1 // (op 534) 

s_mul_hi_i32 s1, s9, s44 // (op 534) 
s_add_i32 s18, s1, s9 // (op 534) 


s_ashr_i32 s1, s18, 31 // (op 534) 
s_cmp_eq_i32 s44, 0 // (op 534) 
s_mov_b32 s9, scc // (op 534) 
s_cmp_eq_i32 s9, 0 // (op 534) 
s_cselect_b32 s19, 0, -1 // (op 534) 

s_and_b32 s9, s48, 31 // (op 534) 
s_lshl_b32 s49, 1, s9 // (op 534) 
s_add_i32 s50, s49, s19 // (op 534) 


s_and_b32 s19, s1, s50 // (op 534) 


s_add_i32 s1, s18, s19 // (op 534) 


s_ashr_i32 s18, s1, s9 // (op 534) 


s_lshl_b32 s1, s48, 24 // (op 534) 
s_ashr_i32 s9, s1, 31 // (op 534) 

s_xor_b32 s1, s18, s9 // (op 534) 

s_sub_i32 s18, s1, s9 // (op 534) 


s_cmp_eq_i32 s0, 0 // (op 534) 
s_mov_b32 s1, scc // (op 534) 

s_cmp_eq_i32 s1, 0 // (op 534) 
s_cselect_b32 s0, s18, s8 // (op 534) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 534) 


v_mul_lo_u32 v17, v9, s23 // (op 534) most significant: low of low * high
v_mul_hi_u32 v8, v9, s22 // (op 534) most significant: high of low * low
v_mul_lo_u32 v16, v9, s22 // (op 534) least significant: low of low * low
v_add_u32 v17, v17, v8 // (op 534) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v16 // (op 534) least significant half
v_addc_co_u32 v9, s[0:1], v11, v17, s[0:1] // (op 534) most significant half



s_mul_i32 s1, s22, s5 // (op 536) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 536) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 536) least significant
s_mul_hi_u32 s9, s22, s4 // (op 536) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 536) most significant: sum
s_add_u32 s1, s1, s9 // (op 536) most significant: sum


s_mov_b32 s56, s24 // (op 536) reg expression
s_mov_b32 s57, s25 // (op 536) reg expression
s_mov_b32 s58, s0 // (op 536) reg expression
s_mov_b32 s59, 131072 // (op 536) call()

s_mul_hi_i32 s0, s16, s12 // (op 553) 
v_lshrrev_b32 v10, 3, v1 // (op 553) 
s_add_i32 s1, s0, s16 // (op 553) 

v_and_b32 v11, -8, v1 // (op 553) 
s_ashr_i32 s0, s1, 31 // (op 553) 
v_sub_u32 v16, v1, v11 // (op 553) 

s_cmp_eq_i32 s12, 0 // (op 553) 
s_mov_b32 s8, scc // (op 553) 
v_lshlrev_b32 v11, 4, v16 // (op 553) 

s_cmp_eq_i32 s8, 0 // (op 553) 
s_cselect_b32 s9, 0, -1 // (op 553) 

v_mov_b32 v17, 0 // (op 553) convert
v_mov_b32 v16, v11 // (op 553) convert

s_and_b32 s8, s13, 31 // (op 553) 
s_lshl_b32 s18, 1, s8 // (op 553) 
s_add_i32 s19, s18, s9 // (op 553) 


s_and_b32 s9, s0, s19 // (op 553) 


s_add_i32 s0, s1, s9 // (op 553) 


s_ashr_i32 s1, s0, s8 // (op 553) 


s_lshl_b32 s0, s13, 24 // (op 553) 
s_ashr_i32 s8, s0, 31 // (op 553) 

s_xor_b32 s0, s1, s8 // (op 553) 

s_sub_i32 s1, s0, s8 // (op 553) 


s_mul_hi_i32 s0, s1, s14 // (op 553) 
s_add_i32 s8, s0, s1 // (op 553) 

s_ashr_i32 s0, s8, 31 // (op 553) 
s_cmp_eq_i32 s14, 0 // (op 553) 
s_mov_b32 s9, scc // (op 553) 
s_cmp_eq_i32 s9, 0 // (op 553) 
s_cselect_b32 s18, 0, -1 // (op 553) 

s_and_b32 s9, s15, 31 // (op 553) 
s_lshl_b32 s19, 1, s9 // (op 553) 
s_add_i32 s49, s19, s18 // (op 553) 


s_and_b32 s18, s0, s49 // (op 553) 


s_add_i32 s0, s8, s18 // (op 553) 


s_ashr_i32 s8, s0, s9 // (op 553) 


s_lshl_b32 s0, s15, 24 // (op 553) 
s_ashr_i32 s9, s0, 31 // (op 553) 

s_xor_b32 s0, s8, s9 // (op 553) 

s_sub_i32 s8, s0, s9 // (op 553) 


s_add_u32 s18, s4, 32 // (op 553) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 553) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 553) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 553) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 553) 

s_mul_i32 s0, s8, s18 // (op 553) 

s_sub_i32 s8, s1, s0 // (op 553) 


s_mul_hi_i32 s9, s16, s11 // (op 553) 
s_add_i32 s0, s9, s16 // (op 553) 

s_ashr_i32 s1, s0, 31 // (op 553) 
s_cmp_eq_i32 s11, 0 // (op 553) 
s_mov_b32 s9, scc // (op 553) 
s_cmp_eq_i32 s9, 0 // (op 553) 
s_cselect_b32 s49, 0, -1 // (op 553) 

s_and_b32 s9, s46, 31 // (op 553) 
s_lshl_b32 s50, 1, s9 // (op 553) 
s_add_i32 s51, s50, s49 // (op 553) 


s_and_b32 s49, s1, s51 // (op 553) 


s_add_i32 s1, s0, s49 // (op 553) 


s_ashr_i32 s0, s1, s9 // (op 553) 


s_lshl_b32 s1, s46, 24 // (op 553) 
s_ashr_i32 s9, s1, 31 // (op 553) 

s_xor_b32 s1, s0, s9 // (op 553) 

s_sub_i32 s0, s1, s9 // (op 553) 


s_mul_i32 s1, s45, s18 // (op 553) 

s_mul_i32 s9, s10, s1 // (op 553) 

s_mul_i32 s1, s0, s9 // (op 553) 

s_sub_u32 s9, s16, s1 // (op 553) 

s_mul_hi_i32 s1, s9, s44 // (op 553) 
s_add_i32 s18, s1, s9 // (op 553) 


s_ashr_i32 s1, s18, 31 // (op 553) 
s_cmp_eq_i32 s44, 0 // (op 553) 
s_mov_b32 s9, scc // (op 553) 
s_cmp_eq_i32 s9, 0 // (op 553) 
s_cselect_b32 s19, 0, -1 // (op 553) 

s_and_b32 s9, s48, 31 // (op 553) 
s_lshl_b32 s49, 1, s9 // (op 553) 
s_add_i32 s50, s49, s19 // (op 553) 


s_and_b32 s19, s1, s50 // (op 553) 


s_add_i32 s1, s18, s19 // (op 553) 


s_ashr_i32 s18, s1, s9 // (op 553) 


s_lshl_b32 s1, s48, 24 // (op 553) 
s_ashr_i32 s9, s1, 31 // (op 553) 

s_xor_b32 s1, s18, s9 // (op 553) 

s_sub_i32 s18, s1, s9 // (op 553) 


s_cmp_eq_i32 s0, 0 // (op 553) 
s_mov_b32 s1, scc // (op 553) 

s_cmp_eq_i32 s1, 0 // (op 553) 
s_cselect_b32 s0, s18, s8 // (op 553) 



v_lshl_add_u32 v11, s0, 5, v10 // (op 553) 


v_mul_lo_u32 v19, v11, s23 // (op 553) most significant: low of low * high
v_mul_hi_u32 v10, v11, s22 // (op 553) most significant: high of low * low
v_mul_lo_u32 v18, v11, s22 // (op 553) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 553) most significant: sum


v_add_co_u32 v10, s[0:1], v16, v18 // (op 553) least significant half
v_addc_co_u32 v11, s[0:1], v17, v19, s[0:1] // (op 553) most significant half



s_mul_i32 s1, s22, s5 // (op 555) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 555) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 555) least significant
s_mul_hi_u32 s9, s22, s4 // (op 555) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 555) most significant: sum
s_add_u32 s1, s1, s9 // (op 555) most significant: sum


s_mov_b32 s56, s24 // (op 555) reg expression
s_mov_b32 s57, s25 // (op 555) reg expression
s_mov_b32 s58, s0 // (op 555) reg expression
s_mov_b32 s59, 131072 // (op 555) call()

s_mul_hi_i32 s0, s16, s12 // (op 598) 
v_lshrrev_b32 v16, 7, v1 // (op 598) 
s_add_i32 s1, s0, s16 // (op 598) 

v_and_b32 v17, 15, v1 // (op 598) 
s_ashr_i32 s0, s1, 31 // (op 598) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 598) 


s_cmp_eq_i32 s12, 0 // (op 598) 
s_mov_b32 s8, scc // (op 598) 
v_and_b32 v16, 63, v1 // (op 598) 
s_cmp_eq_i32 s8, 0 // (op 598) 
s_cselect_b32 s9, 0, -1 // (op 598) 

v_lshrrev_b32 v17, 4, v16 // (op 598) 

s_and_b32 s8, s13, 31 // (op 598) 
s_lshl_b32 s18, 1, s8 // (op 598) 
s_add_i32 s19, s18, s9 // (op 598) 


s_and_b32 s9, s0, s19 // (op 598) 


s_add_i32 s0, s1, s9 // (op 598) 


s_ashr_i32 s1, s0, s8 // (op 598) 


s_lshl_b32 s0, s13, 24 // (op 598) 
s_ashr_i32 s8, s0, 31 // (op 598) 

s_xor_b32 s0, s1, s8 // (op 598) 

s_sub_i32 s1, s0, s8 // (op 598) 


s_mul_hi_i32 s0, s1, s14 // (op 598) 
s_add_i32 s8, s0, s1 // (op 598) 

s_ashr_i32 s0, s8, 31 // (op 598) 
s_cmp_eq_i32 s14, 0 // (op 598) 
s_mov_b32 s9, scc // (op 598) 
s_cmp_eq_i32 s9, 0 // (op 598) 
s_cselect_b32 s18, 0, -1 // (op 598) 

s_and_b32 s9, s15, 31 // (op 598) 
s_lshl_b32 s19, 1, s9 // (op 598) 
s_add_i32 s49, s19, s18 // (op 598) 


s_and_b32 s18, s0, s49 // (op 598) 


s_add_i32 s0, s8, s18 // (op 598) 


s_ashr_i32 s8, s0, s9 // (op 598) 


s_lshl_b32 s0, s15, 24 // (op 598) 
s_ashr_i32 s9, s0, 31 // (op 598) 

s_xor_b32 s0, s8, s9 // (op 598) 

s_sub_i32 s8, s0, s9 // (op 598) 


s_mul_i32 s0, s1, s45 // (op 598) 

s_sub_u32 s1, s16, s0 // (op 598) 

s_mul_i32 s0, s8, s45 // (op 598) 
s_add_u32 s0, s0, s1 // (op 598) 


s_mul_hi_i32 s1, s16, s11 // (op 598) 
s_add_i32 s8, s1, s16 // (op 598) 

s_ashr_i32 s1, s8, 31 // (op 598) 
s_cmp_eq_i32 s11, 0 // (op 598) 
s_mov_b32 s9, scc // (op 598) 
s_cmp_eq_i32 s9, 0 // (op 598) 
s_cselect_b32 s18, 0, -1 // (op 598) 

s_and_b32 s9, s46, 31 // (op 598) 
s_lshl_b32 s19, 1, s9 // (op 598) 
s_add_i32 s49, s19, s18 // (op 598) 


s_and_b32 s18, s1, s49 // (op 598) 


s_add_i32 s1, s8, s18 // (op 598) 


s_ashr_i32 s8, s1, s9 // (op 598) 


s_lshl_b32 s1, s46, 24 // (op 598) 
s_ashr_i32 s9, s1, 31 // (op 598) 

s_xor_b32 s1, s8, s9 // (op 598) 

s_sub_i32 s8, s1, s9 // (op 598) 


s_add_u32 s18, s4, 32 // (op 598) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 598) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 598) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 598) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 598) 

s_mul_i32 s1, s45, s18 // (op 598) 

s_mul_i32 s9, s10, s1 // (op 598) 

s_mul_i32 s1, s8, s9 // (op 598) 

s_sub_u32 s9, s16, s1 // (op 598) 

s_mov_b32 s1, s9 // (op 598) 
s_mul_hi_i32 s18, s1, s44 // (op 598) 
s_add_i32 s19, s18, s1 // (op 598) 


s_ashr_i32 s1, s19, 31 // (op 598) 
s_cmp_eq_i32 s44, 0 // (op 598) 
s_mov_b32 s18, scc // (op 598) 
s_cmp_eq_i32 s18, 0 // (op 598) 
s_cselect_b32 s49, 0, -1 // (op 598) 

s_and_b32 s18, s48, 31 // (op 598) 
s_lshl_b32 s50, 1, s18 // (op 598) 
s_add_i32 s51, s50, s49 // (op 598) 


s_and_b32 s49, s1, s51 // (op 598) 


s_add_i32 s1, s19, s49 // (op 598) 


s_ashr_i32 s19, s1, s18 // (op 598) 


s_lshl_b32 s1, s48, 24 // (op 598) 
s_ashr_i32 s18, s1, 31 // (op 598) 

s_xor_b32 s1, s19, s18 // (op 598) 

s_sub_i32 s19, s1, s18 // (op 598) 


s_mul_i32 s1, s47, s45 // (op 598) 
s_add_u32 s50, s6, 32 // (op 598) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 598) End of Int64 add, unlocking SCC
s_sub_u32 s60, s50, 1 // (op 598) least significant half; sets scc
s_subb_u32 s61, s51, 0 // (op 598) most significant half; uses scc

s_ashr_i64 s[50:51], s[60:61], 5 // (op 598) 

s_sub_i32 s18, s50, s1 // (op 598) 

s_mul_i32 s49, s19, s18 // (op 598) 


s_sub_u32 s18, s9, s49 // (op 598) 


s_add_u32 s9, s18, s1 // (op 598) 


s_cmp_eq_i32 s8, 0 // (op 598) 
s_mov_b32 s1, scc // (op 598) 

s_cmp_eq_i32 s1, 0 // (op 598) 
s_cselect_b32 s8, s9, s0 // (op 598) 



v_lshl_add_u32 v16, s8, 5, v18 // (op 598) 


v_mul_lo_u32 v18, v16, s26 // (op 598) 
v_add_u32 v18, v18, v17 // (op 598) 


s_mul_i32 s1, s26, s7 // (op 600) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 600) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 600) least significant
s_mul_hi_u32 s9, s26, s6 // (op 600) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 600) most significant: sum
s_add_u32 s1, s1, s9 // (op 600) most significant: sum


s_mov_b32 s60, s28 // (op 600) reg expression
s_mov_b32 s61, s29 // (op 600) reg expression
s_mov_b32 s62, s0 // (op 600) reg expression
s_mov_b32 s63, 131072 // (op 600) call()

s_mul_hi_i32 s0, s16, s12 // (op 617) 
v_lshrrev_b32 v16, 7, v1 // (op 617) 
s_add_i32 s1, s0, s16 // (op 617) 

v_and_b32 v17, 15, v1 // (op 617) 
s_ashr_i32 s0, s1, 31 // (op 617) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 617) 


s_cmp_eq_i32 s12, 0 // (op 617) 
s_mov_b32 s8, scc // (op 617) 
v_and_b32 v16, 63, v1 // (op 617) 
s_cmp_eq_i32 s8, 0 // (op 617) 
s_cselect_b32 s9, 0, -1 // (op 617) 

v_lshrrev_b32 v17, 4, v16 // (op 617) 

s_and_b32 s8, s13, 31 // (op 617) 
s_lshl_b32 s18, 1, s8 // (op 617) 
s_add_i32 s19, s18, s9 // (op 617) 


s_and_b32 s9, s0, s19 // (op 617) 


s_add_i32 s0, s1, s9 // (op 617) 


s_ashr_i32 s1, s0, s8 // (op 617) 


s_lshl_b32 s0, s13, 24 // (op 617) 
s_ashr_i32 s8, s0, 31 // (op 617) 

s_xor_b32 s0, s1, s8 // (op 617) 

s_sub_i32 s1, s0, s8 // (op 617) 


s_mul_hi_i32 s0, s1, s14 // (op 617) 
s_add_i32 s8, s0, s1 // (op 617) 

s_ashr_i32 s0, s8, 31 // (op 617) 
s_cmp_eq_i32 s14, 0 // (op 617) 
s_mov_b32 s9, scc // (op 617) 
s_cmp_eq_i32 s9, 0 // (op 617) 
s_cselect_b32 s18, 0, -1 // (op 617) 

s_and_b32 s9, s15, 31 // (op 617) 
s_lshl_b32 s19, 1, s9 // (op 617) 
s_add_i32 s49, s19, s18 // (op 617) 


s_and_b32 s18, s0, s49 // (op 617) 


s_add_i32 s0, s8, s18 // (op 617) 


s_ashr_i32 s8, s0, s9 // (op 617) 


s_lshl_b32 s0, s15, 24 // (op 617) 
s_ashr_i32 s9, s0, 31 // (op 617) 

s_xor_b32 s0, s8, s9 // (op 617) 

s_sub_i32 s8, s0, s9 // (op 617) 


s_mul_i32 s0, s1, s45 // (op 617) 

s_sub_u32 s1, s16, s0 // (op 617) 

s_mul_i32 s0, s8, s45 // (op 617) 
s_add_u32 s0, s0, s1 // (op 617) 


s_mul_hi_i32 s1, s16, s11 // (op 617) 
s_add_i32 s8, s1, s16 // (op 617) 

s_ashr_i32 s1, s8, 31 // (op 617) 
s_cmp_eq_i32 s11, 0 // (op 617) 
s_mov_b32 s9, scc // (op 617) 
s_cmp_eq_i32 s9, 0 // (op 617) 
s_cselect_b32 s18, 0, -1 // (op 617) 

s_and_b32 s9, s46, 31 // (op 617) 
s_lshl_b32 s19, 1, s9 // (op 617) 
s_add_i32 s49, s19, s18 // (op 617) 


s_and_b32 s18, s1, s49 // (op 617) 


s_add_i32 s1, s8, s18 // (op 617) 


s_ashr_i32 s8, s1, s9 // (op 617) 


s_lshl_b32 s1, s46, 24 // (op 617) 
s_ashr_i32 s9, s1, 31 // (op 617) 

s_xor_b32 s1, s8, s9 // (op 617) 

s_sub_i32 s8, s1, s9 // (op 617) 


s_add_u32 s18, s4, 32 // (op 617) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 617) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 617) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 617) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 617) 

s_mul_i32 s1, s45, s18 // (op 617) 

s_mul_i32 s9, s10, s1 // (op 617) 

s_mul_i32 s1, s8, s9 // (op 617) 

s_sub_u32 s9, s16, s1 // (op 617) 

s_mov_b32 s1, s9 // (op 617) 
s_mul_hi_i32 s18, s1, s44 // (op 617) 
s_add_i32 s19, s18, s1 // (op 617) 


s_ashr_i32 s1, s19, 31 // (op 617) 
s_cmp_eq_i32 s44, 0 // (op 617) 
s_mov_b32 s18, scc // (op 617) 
s_cmp_eq_i32 s18, 0 // (op 617) 
s_cselect_b32 s49, 0, -1 // (op 617) 

s_and_b32 s18, s48, 31 // (op 617) 
s_lshl_b32 s50, 1, s18 // (op 617) 
s_add_i32 s51, s50, s49 // (op 617) 


s_and_b32 s49, s1, s51 // (op 617) 


s_add_i32 s1, s19, s49 // (op 617) 


s_ashr_i32 s19, s1, s18 // (op 617) 


s_lshl_b32 s1, s48, 24 // (op 617) 
s_ashr_i32 s18, s1, 31 // (op 617) 

s_xor_b32 s1, s19, s18 // (op 617) 

s_sub_i32 s19, s1, s18 // (op 617) 


s_mul_i32 s1, s47, s45 // (op 617) 
s_add_u32 s50, s6, 32 // (op 617) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 617) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 617) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 617) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 617) 

s_sub_i32 s18, s50, s1 // (op 617) 

s_mul_i32 s49, s19, s18 // (op 617) 


s_sub_u32 s18, s9, s49 // (op 617) 


s_add_u32 s9, s18, s1 // (op 617) 


s_cmp_eq_i32 s8, 0 // (op 617) 
s_mov_b32 s1, scc // (op 617) 

s_cmp_eq_i32 s1, 0 // (op 617) 
s_cselect_b32 s8, s9, s0 // (op 617) 



v_lshl_add_u32 v16, s8, 5, v19 // (op 617) 


v_mul_lo_u32 v19, v16, s26 // (op 617) 
v_add_u32 v19, v19, v17 // (op 617) 


s_mul_i32 s1, s26, s7 // (op 619) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 619) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 619) least significant
s_mul_hi_u32 s9, s26, s6 // (op 619) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 619) most significant: sum
s_add_u32 s1, s1, s9 // (op 619) most significant: sum


s_mov_b32 s60, s28 // (op 619) reg expression
s_mov_b32 s61, s29 // (op 619) reg expression
s_mov_b32 s62, s0 // (op 619) reg expression
s_mov_b32 s63, 131072 // (op 619) call()

s_mul_hi_i32 s0, s16, s12 // (op 662) 
v_lshrrev_b32 v16, 6, v1 // (op 662) 
s_add_i32 s1, s0, s16 // (op 662) 

v_and_b32 v17, 1, v16 // (op 662) 

s_ashr_i32 s0, s1, 31 // (op 662) 
v_and_b32 v16, 15, v1 // (op 662) 
s_cmp_eq_i32 s12, 0 // (op 662) 
s_mov_b32 s8, scc // (op 662) 
v_lshl_add_u32 v20, v17, 4, v16 // (op 662) 


s_cmp_eq_i32 s8, 0 // (op 662) 
s_cselect_b32 s9, 0, -1 // (op 662) 

v_and_b32 v16, 63, v1 // (op 662) 
s_and_b32 s8, s13, 31 // (op 662) 
v_lshrrev_b32 v17, 4, v16 // (op 662) 

s_lshl_b32 s18, 1, s8 // (op 662) 
s_add_i32 s19, s18, s9 // (op 662) 


s_and_b32 s9, s0, s19 // (op 662) 


s_add_i32 s0, s1, s9 // (op 662) 


s_ashr_i32 s1, s0, s8 // (op 662) 


s_lshl_b32 s0, s13, 24 // (op 662) 
s_ashr_i32 s8, s0, 31 // (op 662) 

s_xor_b32 s0, s1, s8 // (op 662) 

s_sub_i32 s1, s0, s8 // (op 662) 


s_mul_hi_i32 s0, s1, s14 // (op 662) 
s_add_i32 s8, s0, s1 // (op 662) 

s_ashr_i32 s0, s8, 31 // (op 662) 
s_cmp_eq_i32 s14, 0 // (op 662) 
s_mov_b32 s9, scc // (op 662) 
s_cmp_eq_i32 s9, 0 // (op 662) 
s_cselect_b32 s18, 0, -1 // (op 662) 

s_and_b32 s9, s15, 31 // (op 662) 
s_lshl_b32 s19, 1, s9 // (op 662) 
s_add_i32 s49, s19, s18 // (op 662) 


s_and_b32 s18, s0, s49 // (op 662) 


s_add_i32 s0, s8, s18 // (op 662) 


s_ashr_i32 s8, s0, s9 // (op 662) 


s_lshl_b32 s0, s15, 24 // (op 662) 
s_ashr_i32 s9, s0, 31 // (op 662) 

s_xor_b32 s0, s8, s9 // (op 662) 

s_sub_i32 s8, s0, s9 // (op 662) 


s_add_u32 s18, s4, 32 // (op 662) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 662) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 662) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 662) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 662) 

s_mul_i32 s0, s8, s18 // (op 662) 

s_sub_i32 s8, s1, s0 // (op 662) 


s_mul_hi_i32 s9, s16, s11 // (op 662) 
s_add_i32 s0, s9, s16 // (op 662) 

s_ashr_i32 s1, s0, 31 // (op 662) 
s_cmp_eq_i32 s11, 0 // (op 662) 
s_mov_b32 s9, scc // (op 662) 
s_cmp_eq_i32 s9, 0 // (op 662) 
s_cselect_b32 s49, 0, -1 // (op 662) 

s_and_b32 s9, s46, 31 // (op 662) 
s_lshl_b32 s50, 1, s9 // (op 662) 
s_add_i32 s51, s50, s49 // (op 662) 


s_and_b32 s49, s1, s51 // (op 662) 


s_add_i32 s1, s0, s49 // (op 662) 


s_ashr_i32 s0, s1, s9 // (op 662) 


s_lshl_b32 s1, s46, 24 // (op 662) 
s_ashr_i32 s9, s1, 31 // (op 662) 

s_xor_b32 s1, s0, s9 // (op 662) 

s_sub_i32 s0, s1, s9 // (op 662) 


s_mul_i32 s1, s45, s18 // (op 662) 

s_mul_i32 s9, s10, s1 // (op 662) 

s_mul_i32 s1, s0, s9 // (op 662) 

s_sub_u32 s9, s16, s1 // (op 662) 

s_mul_hi_i32 s1, s9, s44 // (op 662) 
s_add_i32 s18, s1, s9 // (op 662) 


s_ashr_i32 s1, s18, 31 // (op 662) 
s_cmp_eq_i32 s44, 0 // (op 662) 
s_mov_b32 s9, scc // (op 662) 
s_cmp_eq_i32 s9, 0 // (op 662) 
s_cselect_b32 s19, 0, -1 // (op 662) 

s_and_b32 s9, s48, 31 // (op 662) 
s_lshl_b32 s49, 1, s9 // (op 662) 
s_add_i32 s50, s49, s19 // (op 662) 


s_and_b32 s19, s1, s50 // (op 662) 


s_add_i32 s1, s18, s19 // (op 662) 


s_ashr_i32 s18, s1, s9 // (op 662) 


s_lshl_b32 s1, s48, 24 // (op 662) 
s_ashr_i32 s9, s1, 31 // (op 662) 

s_xor_b32 s1, s18, s9 // (op 662) 

s_sub_i32 s18, s1, s9 // (op 662) 


s_cmp_eq_i32 s0, 0 // (op 662) 
s_mov_b32 s1, scc // (op 662) 

s_cmp_eq_i32 s1, 0 // (op 662) 
s_cselect_b32 s0, s18, s8 // (op 662) 



v_lshl_add_u32 v16, s0, 5, v20 // (op 662) 


v_mul_lo_u32 v20, v16, s30 // (op 662) 

v_add_u32 v16, v17, v20 // (op 662) 


s_mul_i32 s1, s30, s5 // (op 664) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 664) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 664) least significant
s_mul_hi_u32 s9, s30, s4 // (op 664) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 664) most significant: sum
s_add_u32 s1, s1, s9 // (op 664) most significant: sum


s_mov_b32 s64, s32 // (op 664) reg expression
s_mov_b32 s65, s33 // (op 664) reg expression
s_mov_b32 s66, s0 // (op 664) reg expression
s_mov_b32 s67, 131072 // (op 664) call()

s_mul_hi_i32 s0, s16, s12 // (op 681) 
v_lshrrev_b32 v17, 6, v1 // (op 681) 
s_add_i32 s1, s0, s16 // (op 681) 

v_and_b32 v20, 1, v17 // (op 681) 

s_ashr_i32 s0, s1, 31 // (op 681) 
v_and_b32 v17, 15, v1 // (op 681) 
s_cmp_eq_i32 s12, 0 // (op 681) 
s_mov_b32 s8, scc // (op 681) 
v_lshl_add_u32 v21, v20, 4, v17 // (op 681) 


s_cmp_eq_i32 s8, 0 // (op 681) 
s_cselect_b32 s9, 0, -1 // (op 681) 

v_and_b32 v17, 63, v1 // (op 681) 
s_and_b32 s8, s13, 31 // (op 681) 
v_lshrrev_b32 v20, 4, v17 // (op 681) 

s_lshl_b32 s18, 1, s8 // (op 681) 
s_add_i32 s19, s18, s9 // (op 681) 


s_and_b32 s9, s0, s19 // (op 681) 


s_add_i32 s0, s1, s9 // (op 681) 


s_ashr_i32 s1, s0, s8 // (op 681) 


s_lshl_b32 s0, s13, 24 // (op 681) 
s_ashr_i32 s8, s0, 31 // (op 681) 

s_xor_b32 s0, s1, s8 // (op 681) 

s_sub_i32 s1, s0, s8 // (op 681) 


s_mul_hi_i32 s0, s1, s14 // (op 681) 
s_add_i32 s8, s0, s1 // (op 681) 

s_ashr_i32 s0, s8, 31 // (op 681) 
s_cmp_eq_i32 s14, 0 // (op 681) 
s_mov_b32 s9, scc // (op 681) 
s_cmp_eq_i32 s9, 0 // (op 681) 
s_cselect_b32 s18, 0, -1 // (op 681) 

s_and_b32 s9, s15, 31 // (op 681) 
s_lshl_b32 s19, 1, s9 // (op 681) 
s_add_i32 s49, s19, s18 // (op 681) 


s_and_b32 s18, s0, s49 // (op 681) 


s_add_i32 s0, s8, s18 // (op 681) 


s_ashr_i32 s8, s0, s9 // (op 681) 


s_lshl_b32 s0, s15, 24 // (op 681) 
s_ashr_i32 s9, s0, 31 // (op 681) 

s_xor_b32 s0, s8, s9 // (op 681) 

s_sub_i32 s8, s0, s9 // (op 681) 


s_add_u32 s18, s4, 32 // (op 681) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 681) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 681) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 681) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 681) 

s_mul_i32 s0, s8, s18 // (op 681) 

s_sub_i32 s8, s1, s0 // (op 681) 


s_mul_hi_i32 s9, s16, s11 // (op 681) 
s_add_i32 s0, s9, s16 // (op 681) 

s_ashr_i32 s1, s0, 31 // (op 681) 
s_cmp_eq_i32 s11, 0 // (op 681) 
s_mov_b32 s9, scc // (op 681) 
s_cmp_eq_i32 s9, 0 // (op 681) 
s_cselect_b32 s49, 0, -1 // (op 681) 

s_and_b32 s9, s46, 31 // (op 681) 
s_lshl_b32 s50, 1, s9 // (op 681) 
s_add_i32 s51, s50, s49 // (op 681) 


s_and_b32 s49, s1, s51 // (op 681) 


s_add_i32 s1, s0, s49 // (op 681) 


s_ashr_i32 s0, s1, s9 // (op 681) 


s_lshl_b32 s1, s46, 24 // (op 681) 
s_ashr_i32 s9, s1, 31 // (op 681) 

s_xor_b32 s1, s0, s9 // (op 681) 

s_sub_i32 s0, s1, s9 // (op 681) 


s_mul_i32 s1, s45, s18 // (op 681) 

s_mul_i32 s9, s10, s1 // (op 681) 

s_mul_i32 s1, s0, s9 // (op 681) 

s_sub_u32 s9, s16, s1 // (op 681) 

s_mul_hi_i32 s1, s9, s44 // (op 681) 
s_add_i32 s18, s1, s9 // (op 681) 


s_ashr_i32 s1, s18, 31 // (op 681) 
s_cmp_eq_i32 s44, 0 // (op 681) 
s_mov_b32 s9, scc // (op 681) 
s_cmp_eq_i32 s9, 0 // (op 681) 
s_cselect_b32 s19, 0, -1 // (op 681) 

s_and_b32 s9, s48, 31 // (op 681) 
s_lshl_b32 s49, 1, s9 // (op 681) 
s_add_i32 s50, s49, s19 // (op 681) 


s_and_b32 s19, s1, s50 // (op 681) 


s_add_i32 s1, s18, s19 // (op 681) 


s_ashr_i32 s18, s1, s9 // (op 681) 


s_lshl_b32 s1, s48, 24 // (op 681) 
s_ashr_i32 s9, s1, 31 // (op 681) 

s_xor_b32 s1, s18, s9 // (op 681) 

s_sub_i32 s18, s1, s9 // (op 681) 


s_cmp_eq_i32 s0, 0 // (op 681) 
s_mov_b32 s1, scc // (op 681) 

s_cmp_eq_i32 s1, 0 // (op 681) 
s_cselect_b32 s0, s18, s8 // (op 681) 



v_lshl_add_u32 v17, s0, 5, v21 // (op 681) 


v_mul_lo_u32 v21, v17, s30 // (op 681) 

v_add_u32 v17, v20, v21 // (op 681) 


s_mul_i32 s1, s30, s5 // (op 683) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 683) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 683) least significant
s_mul_hi_u32 s9, s30, s4 // (op 683) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 683) most significant: sum
s_add_u32 s1, s1, s9 // (op 683) most significant: sum


s_mov_b32 s64, s32 // (op 683) reg expression
s_mov_b32 s65, s33 // (op 683) reg expression
s_mov_b32 s66, s0 // (op 683) reg expression
s_mov_b32 s67, 131072 // (op 683) call()




buffer_load_ubyte v20, v18, s[60:63], 0 offen offset: 0 // (op 308) Load value



buffer_load_ubyte v21, v16, s[64:67], 0 offen offset: 0 // (op 313) Load value
v_and_b32 v22, -8, v1 // (op 799) 
v_sub_u32 v23, v1, v22 // (op 799) 

v_lshlrev_b32 v22, 4, v23 // (op 799) 

v_lshrrev_b32 v23, 3, v1 // (op 799) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 799) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 799) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 419) reg expression
buffer_load_dwordx4 v8, s[56:59], 0 offen offset: 0 lds // (op 419) Load value direct to lds

v_and_b32 v22, -8, v1 // (op 747) 
v_sub_u32 v23, v1, v22 // (op 747) 

v_lshlrev_b32 v22, 4, v23 // (op 747) 

v_lshrrev_b32 v23, 3, v1 // (op 747) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 747) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 747) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 424) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 424) reg expression
buffer_load_dwordx4 v12, s[52:55], 0 offen offset: 0 lds // (op 424) Load value direct to lds





buffer_load_ubyte v22, v18, s[60:63], 0 offen offset:4 // (op 324) Load value




buffer_load_ubyte v18, v16, s[64:67], 0 offen offset:4 // (op 329) Load value

v_and_b32 v16, -8, v1 // (op 911) 
v_sub_u32 v23, v1, v16 // (op 911) 

v_lshlrev_b32 v16, 4, v23 // (op 911) 

v_lshrrev_b32 v23, 3, v1 // (op 911) 
v_lshl_add_u32 v24, v23, 7, v16 // (op 911) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 911) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 8192, s0 // (op 429) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 429) reg expression
s_mov_b32 s8, 128 // (op 429) call()
buffer_load_dwordx4 v8, s[56:59], s8 offen offset: 0 lds // (op 429) Load value direct to lds




v_and_b32 v16, -8, v1 // (op 872) 
v_sub_u32 v8, v1, v16 // (op 872) 

v_lshlrev_b32 v9, 4, v8 // (op 872) 

v_lshrrev_b32 v8, 3, v1 // (op 872) 
v_lshl_add_u32 v16, v8, 7, v9 // (op 872) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 872) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 12288, s0 // (op 431) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 431) reg expression
s_mov_b32 s8, 128 // (op 431) call()
buffer_load_dwordx4 v12, s[52:55], s8 offen offset: 0 lds // (op 431) Load value direct to lds




s_waitcnt vmcnt(5)
s_barrier  // (op 968) 



ds_read_b128 v[24:27], v4 // (op 354) Load local data 
ds_read_b128 v[28:31], v4 offset:64 // (op 354) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 952) 



ds_read_b128 v[32:35], v3 offset:4096 // (op 351) Load local data 
ds_read_b128 v[36:39], v3 offset:4160 // (op 351) Load local data 
v_and_b32 v16, -8, v1 // (op 719) 
v_sub_u32 v23, v1, v16 // (op 719) 

v_lshlrev_b32 v16, 4, v23 // (op 719) 

v_lshrrev_b32 v23, 3, v1 // (op 719) 
v_lshl_add_u32 v8, v23, 7, v16 // (op 719) 


s_nop 0
v_readfirstlane_b32 s0, v8 // (op 719) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 771) 
v_sub_u32 v23, v1, v16 // (op 771) 

v_lshlrev_b32 v16, 4, v23 // (op 771) 

v_lshrrev_b32 v23, 3, v1 // (op 771) 
v_lshl_add_u32 v8, v23, 7, v16 // (op 771) 


s_nop 0
v_readfirstlane_b32 s1, v8 // (op 771) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 857) 
v_sub_u32 v23, v1, v16 // (op 857) 

v_lshlrev_b32 v16, 4, v23 // (op 857) 

v_lshrrev_b32 v23, 3, v1 // (op 857) 
v_lshl_add_u32 v8, v23, 7, v16 // (op 857) 


s_nop 0
v_readfirstlane_b32 s8, v8 // (op 857) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 896) 
v_sub_u32 v23, v1, v16 // (op 896) 

v_lshlrev_b32 v16, 4, v23 // (op 896) 

v_lshrrev_b32 v23, 3, v1 // (op 896) 
v_lshl_add_u32 v8, v23, 7, v16 // (op 896) 


s_nop 0
v_readfirstlane_b32 s9, v8 // (op 896) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s18, 0 // (op 51) call()
s_ashr_i64 s[50:51], s[34:35], 7 // (op 50) 
s_mov_b32 s19, s50 // (op 50) convert

s_and_b32 s49, s19, 4294967294 // (op 50) 

s_cmp_lt_u32 s18, s49 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_50 // (op 50) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_50 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_50:
s_add_u32 s18, s18, 2 // (op 52) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[24:31], a[0:3], v20, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 76) 


v_mov_b32 v16, v19 // (op 70) 

buffer_load_ubyte v20, v16, s[60:63], 0 offen offset:8 // (op 70) Load value


v_mov_b32 v23, v17 // (op 74) 

buffer_load_ubyte v21, v23, s[64:67], 0 offen offset:8 // (op 74) Load value
s_barrier  // (op 971) 


v_mov_b64 v[8:9], v[10:11] // (op 392) 

s_mov_b32 m0, s1 // (op 392) reg expression
s_mov_b32 s19, 256 // (op 392) call()
buffer_load_dwordx4 v8, s[56:59], s19 offen offset: 0 lds // (op 392) Load value direct to lds

s_barrier  // (op 955) 


v_mov_b64 v[12:13], v[14:15] // (op 387) 
s_add_u32 s19, 4096, s0 // (op 387) 

s_mov_b32 m0, s19 // (op 387) reg expression
s_mov_b32 s49, 256 // (op 387) call()
buffer_load_dwordx4 v12, s[52:55], s49 offen offset: 0 lds // (op 387) Load value direct to lds


s_waitcnt vmcnt(5)
s_barrier  // (op 990) 



ds_read_b128 v[24:27], v6 offset:8192 // (op 212) Load local data 
ds_read_b128 v[28:31], v6 offset:8256 // (op 212) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 984) 



ds_read_b128 v[32:35], v5 offset:12288 // (op 210) Load local data 
ds_read_b128 v[36:39], v5 offset:12352 // (op 210) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[24:31], a[0:3], v22, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 217) 



buffer_load_ubyte v22, v16, s[60:63], 0 offen offset:12 // (op 214) Load value




v_add_u32 v19, 8, v19 // (op 609) 
buffer_load_ubyte v18, v23, s[64:67], 0 offen offset:12 // (op 216) Load value

s_barrier  // (op 993) 
v_add_u32 v17, 8, v17 // (op 673) 


s_add_u32 s19, 8192, s9 // (op 414) 

s_mov_b32 m0, s19 // (op 414) reg expression
s_mov_b32 s49, 384 // (op 414) call()
buffer_load_dwordx4 v8, s[56:59], s49 offen offset: 0 lds // (op 414) Load value direct to lds



s_barrier  // (op 987) 
v_add_co_u32 v10, vcc, 256, v10 // (op 545) least significant half


v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 545) most significant half
s_add_u32 s19, 12288, s8 // (op 409) 

s_mov_b32 m0, s19 // (op 409) reg expression
s_mov_b32 s49, 384 // (op 409) call()
buffer_load_dwordx4 v12, s[52:55], s49 offen offset: 0 lds // (op 409) Load value direct to lds



v_add_co_u32 v14, vcc, 256, v14 // (op 481) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 481) most significant half
s_waitcnt vmcnt(5)
s_barrier  // (op 977) 



ds_read_b128 v[24:27], v4 // (op 65) Load local data 
ds_read_b128 v[28:31], v4 offset:64 // (op 65) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 961) 



ds_read_b128 v[32:35], v3 offset:4096 // (op 60) Load local data 
ds_read_b128 v[36:39], v3 offset:4160 // (op 60) Load local data 
s_ashr_i64 s[50:51], s[34:35], 7 // (op 50) 
s_mov_b32 s19, s50 // (op 50) convert

s_and_b32 s49, s19, 4294967294 // (op 50) 

s_cmp_lt_u32 s18, s49 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_50 // (op 50) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_50 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_50:











s_barrier  // (op 1000) 
s_mul_hi_i32 s0, s16, s12 // (op 444) 
v_lshrrev_b32 v23, 3, v1 // (op 444) 
s_add_i32 s1, s0, s16 // (op 444) 

v_and_b32 v8, -8, v1 // (op 444) 
s_ashr_i32 s0, s1, 31 // (op 444) 
v_sub_u32 v9, v1, v8 // (op 444) 

s_cmp_eq_i32 s12, 0 // (op 444) 
s_mov_b32 s8, scc // (op 444) 
v_lshlrev_b32 v8, 4, v9 // (op 444) 

s_cmp_eq_i32 s8, 0 // (op 444) 
s_cselect_b32 s9, 0, -1 // (op 444) 

s_and_b32 s8, s13, 31 // (op 444) 
s_lshl_b32 s18, 1, s8 // (op 444) 
s_add_i32 s19, s18, s9 // (op 444) 


s_and_b32 s9, s0, s19 // (op 444) 


s_add_i32 s0, s1, s9 // (op 444) 


s_ashr_i32 s1, s0, s8 // (op 444) 


s_lshl_b32 s0, s13, 24 // (op 444) 
s_ashr_i32 s8, s0, 31 // (op 444) 

s_xor_b32 s0, s1, s8 // (op 444) 

s_sub_i32 s1, s0, s8 // (op 444) 


s_mul_hi_i32 s0, s1, s14 // (op 444) 
s_add_i32 s8, s0, s1 // (op 444) 

s_ashr_i32 s0, s8, 31 // (op 444) 
s_cmp_eq_i32 s14, 0 // (op 444) 
s_mov_b32 s9, scc // (op 444) 
s_cmp_eq_i32 s9, 0 // (op 444) 
s_cselect_b32 s18, 0, -1 // (op 444) 

s_and_b32 s9, s15, 31 // (op 444) 
s_lshl_b32 s19, 1, s9 // (op 444) 
s_add_i32 s49, s19, s18 // (op 444) 


s_and_b32 s18, s0, s49 // (op 444) 


s_add_i32 s0, s8, s18 // (op 444) 


s_ashr_i32 s8, s0, s9 // (op 444) 


s_lshl_b32 s0, s15, 24 // (op 444) 
s_ashr_i32 s9, s0, 31 // (op 444) 

s_xor_b32 s0, s8, s9 // (op 444) 

s_sub_i32 s8, s0, s9 // (op 444) 


s_mul_i32 s0, s1, s45 // (op 444) 

s_sub_u32 s1, s16, s0 // (op 444) 

s_mul_i32 s0, s8, s45 // (op 444) 
s_add_u32 s0, s0, s1 // (op 444) 


s_mul_hi_i32 s1, s16, s11 // (op 444) 
s_add_i32 s8, s1, s16 // (op 444) 

s_ashr_i32 s1, s8, 31 // (op 444) 
s_cmp_eq_i32 s11, 0 // (op 444) 
s_mov_b32 s9, scc // (op 444) 
s_cmp_eq_i32 s9, 0 // (op 444) 
s_cselect_b32 s18, 0, -1 // (op 444) 

s_and_b32 s9, s46, 31 // (op 444) 
s_lshl_b32 s19, 1, s9 // (op 444) 
s_add_i32 s49, s19, s18 // (op 444) 


s_and_b32 s18, s1, s49 // (op 444) 


s_add_i32 s1, s8, s18 // (op 444) 


s_ashr_i32 s8, s1, s9 // (op 444) 


s_lshl_b32 s1, s46, 24 // (op 444) 
s_ashr_i32 s9, s1, 31 // (op 444) 

s_xor_b32 s1, s8, s9 // (op 444) 

s_sub_i32 s8, s1, s9 // (op 444) 


s_add_u32 s18, s4, 32 // (op 444) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 444) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 444) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 444) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 444) 

s_mul_i32 s1, s45, s18 // (op 444) 

s_mul_i32 s9, s10, s1 // (op 444) 

s_mul_i32 s1, s8, s9 // (op 444) 

s_sub_u32 s9, s16, s1 // (op 444) 

s_mov_b32 s1, s9 // (op 444) 
s_mul_hi_i32 s18, s1, s44 // (op 444) 
s_add_i32 s19, s18, s1 // (op 444) 


s_ashr_i32 s1, s19, 31 // (op 444) 
s_cmp_eq_i32 s44, 0 // (op 444) 
s_mov_b32 s18, scc // (op 444) 
s_cmp_eq_i32 s18, 0 // (op 444) 
s_cselect_b32 s49, 0, -1 // (op 444) 

s_and_b32 s18, s48, 31 // (op 444) 
s_lshl_b32 s50, 1, s18 // (op 444) 
s_add_i32 s51, s50, s49 // (op 444) 


s_and_b32 s49, s1, s51 // (op 444) 


s_add_i32 s1, s19, s49 // (op 444) 


s_ashr_i32 s19, s1, s18 // (op 444) 


s_lshl_b32 s1, s48, 24 // (op 444) 
s_ashr_i32 s18, s1, 31 // (op 444) 

s_xor_b32 s1, s19, s18 // (op 444) 

s_sub_i32 s19, s1, s18 // (op 444) 


s_mul_i32 s1, s47, s45 // (op 444) 
s_add_u32 s50, s6, 32 // (op 444) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 444) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 444) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 444) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 444) 

s_sub_i32 s18, s50, s1 // (op 444) 

s_mul_i32 s49, s19, s18 // (op 444) 


s_sub_u32 s18, s9, s49 // (op 444) 


s_add_u32 s9, s18, s1 // (op 444) 


s_cmp_eq_i32 s8, 0 // (op 444) 
s_mov_b32 s1, scc // (op 444) 

s_cmp_eq_i32 s1, 0 // (op 444) 
s_cselect_b32 s8, s9, s0 // (op 444) 



v_lshl_add_u32 v9, s8, 5, v23 // (op 444) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 444) 
s_lshr_b32 s8, s0, 1 // (op 444) 

v_lshl_add_u32 v23, s8, 8, v8 // (op 444) 


v_mov_b32 v21, 0 // (op 444) convert
v_mov_b32 v20, v23 // (op 444) convert

v_mul_lo_u32 v11, v9, s3 // (op 444) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 444) most significant: high of low * low
v_mul_lo_u32 v10, v9, s2 // (op 444) least significant: low of low * low
v_add_u32 v11, v11, v8 // (op 444) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v20 // (op 444) least significant half
v_addc_co_u32 v11, s[0:1], v11, v21, s[0:1] // (op 444) most significant half



s_mul_i32 s1, s2, s7 // (op 446) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 446) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 446) least significant
s_mul_hi_u32 s9, s2, s6 // (op 446) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 446) most significant: sum
s_add_u32 s1, s1, s9 // (op 446) most significant: sum


s_mov_b32 s52, s20 // (op 446) reg expression
s_mov_b32 s53, s21 // (op 446) reg expression
s_mov_b32 s54, s0 // (op 446) reg expression
s_mov_b32 s55, 131072 // (op 446) call()

s_mul_hi_i32 s0, s16, s12 // (op 511) 
v_lshrrev_b32 v23, 3, v1 // (op 511) 
s_add_i32 s1, s0, s16 // (op 511) 

v_and_b32 v8, -8, v1 // (op 511) 
s_ashr_i32 s0, s1, 31 // (op 511) 
v_sub_u32 v9, v1, v8 // (op 511) 

s_cmp_eq_i32 s12, 0 // (op 511) 
s_mov_b32 s8, scc // (op 511) 
v_lshlrev_b32 v8, 4, v9 // (op 511) 

s_cmp_eq_i32 s8, 0 // (op 511) 
s_cselect_b32 s9, 0, -1 // (op 511) 

s_and_b32 s8, s13, 31 // (op 511) 
s_lshl_b32 s18, 1, s8 // (op 511) 
s_add_i32 s19, s18, s9 // (op 511) 


s_and_b32 s9, s0, s19 // (op 511) 


s_add_i32 s0, s1, s9 // (op 511) 


s_ashr_i32 s1, s0, s8 // (op 511) 


s_lshl_b32 s0, s13, 24 // (op 511) 
s_ashr_i32 s8, s0, 31 // (op 511) 

s_xor_b32 s0, s1, s8 // (op 511) 

s_sub_i32 s1, s0, s8 // (op 511) 


s_mul_hi_i32 s0, s1, s14 // (op 511) 
s_add_i32 s8, s0, s1 // (op 511) 

s_ashr_i32 s0, s8, 31 // (op 511) 
s_cmp_eq_i32 s14, 0 // (op 511) 
s_mov_b32 s9, scc // (op 511) 
s_cmp_eq_i32 s9, 0 // (op 511) 
s_cselect_b32 s18, 0, -1 // (op 511) 

s_and_b32 s9, s15, 31 // (op 511) 
s_lshl_b32 s19, 1, s9 // (op 511) 
s_add_i32 s49, s19, s18 // (op 511) 


s_and_b32 s18, s0, s49 // (op 511) 


s_add_i32 s0, s8, s18 // (op 511) 


s_ashr_i32 s8, s0, s9 // (op 511) 


s_lshl_b32 s0, s15, 24 // (op 511) 
s_ashr_i32 s9, s0, 31 // (op 511) 

s_xor_b32 s0, s8, s9 // (op 511) 

s_sub_i32 s8, s0, s9 // (op 511) 


s_add_u32 s18, s4, 32 // (op 511) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 511) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 511) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 511) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 511) 

s_mul_i32 s0, s8, s18 // (op 511) 

s_sub_i32 s8, s1, s0 // (op 511) 


s_mul_hi_i32 s9, s16, s11 // (op 511) 
s_add_i32 s0, s9, s16 // (op 511) 

s_ashr_i32 s1, s0, 31 // (op 511) 
s_cmp_eq_i32 s11, 0 // (op 511) 
s_mov_b32 s9, scc // (op 511) 
s_cmp_eq_i32 s9, 0 // (op 511) 
s_cselect_b32 s49, 0, -1 // (op 511) 

s_and_b32 s9, s46, 31 // (op 511) 
s_lshl_b32 s50, 1, s9 // (op 511) 
s_add_i32 s51, s50, s49 // (op 511) 


s_and_b32 s49, s1, s51 // (op 511) 


s_add_i32 s1, s0, s49 // (op 511) 


s_ashr_i32 s0, s1, s9 // (op 511) 


s_lshl_b32 s1, s46, 24 // (op 511) 
s_ashr_i32 s9, s1, 31 // (op 511) 

s_xor_b32 s1, s0, s9 // (op 511) 

s_sub_i32 s0, s1, s9 // (op 511) 


s_mul_i32 s1, s45, s18 // (op 511) 

s_mul_i32 s9, s10, s1 // (op 511) 

s_mul_i32 s1, s0, s9 // (op 511) 

s_sub_u32 s9, s16, s1 // (op 511) 

s_mul_hi_i32 s1, s9, s44 // (op 511) 
s_add_i32 s18, s1, s9 // (op 511) 


s_ashr_i32 s1, s18, 31 // (op 511) 
s_cmp_eq_i32 s44, 0 // (op 511) 
s_mov_b32 s9, scc // (op 511) 
s_cmp_eq_i32 s9, 0 // (op 511) 
s_cselect_b32 s19, 0, -1 // (op 511) 

s_and_b32 s9, s48, 31 // (op 511) 
s_lshl_b32 s49, 1, s9 // (op 511) 
s_add_i32 s50, s49, s19 // (op 511) 


s_and_b32 s19, s1, s50 // (op 511) 


s_add_i32 s1, s18, s19 // (op 511) 


s_ashr_i32 s18, s1, s9 // (op 511) 


s_lshl_b32 s1, s48, 24 // (op 511) 
s_ashr_i32 s9, s1, 31 // (op 511) 

s_xor_b32 s1, s18, s9 // (op 511) 

s_sub_i32 s18, s1, s9 // (op 511) 


s_cmp_eq_i32 s0, 0 // (op 511) 
s_mov_b32 s1, scc // (op 511) 

s_cmp_eq_i32 s1, 0 // (op 511) 
s_cselect_b32 s0, s18, s8 // (op 511) 



v_lshl_add_u32 v9, s0, 5, v23 // (op 511) 


v_mul_lo_u32 v21, v9, s23 // (op 511) most significant: low of low * high
v_mul_hi_u32 v19, v9, s22 // (op 511) most significant: high of low * low
v_mul_lo_u32 v20, v9, s22 // (op 511) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 511) most significant: sum


s_ashr_i64 s[0:1], s[34:35], 7 // (op 511) 
s_lshr_b32 s8, s0, 1 // (op 511) 

v_lshl_add_u32 v9, s8, 8, v8 // (op 511) 


v_mov_b32 v13, 0 // (op 511) convert
v_mov_b32 v12, v9 // (op 511) convert

v_add_co_u32 v8, s[0:1], v12, v20 // (op 511) least significant half
v_addc_co_u32 v9, s[0:1], v13, v21, s[0:1] // (op 511) most significant half



s_mul_i32 s1, s22, s5 // (op 513) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 513) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 513) least significant
s_mul_hi_u32 s9, s22, s4 // (op 513) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 513) most significant: sum
s_add_u32 s1, s1, s9 // (op 513) most significant: sum


s_mov_b32 s56, s24 // (op 513) reg expression
s_mov_b32 s57, s25 // (op 513) reg expression
s_mov_b32 s58, s0 // (op 513) reg expression
s_mov_b32 s59, 131072 // (op 513) call()

s_mul_hi_i32 s0, s16, s12 // (op 575) 
v_lshrrev_b32 v23, 7, v1 // (op 575) 
s_add_i32 s1, s0, s16 // (op 575) 

v_and_b32 v12, 15, v1 // (op 575) 
s_ashr_i32 s0, s1, 31 // (op 575) 
v_lshl_add_u32 v13, v23, 4, v12 // (op 575) 


s_cmp_eq_i32 s12, 0 // (op 575) 
s_mov_b32 s8, scc // (op 575) 
v_and_b32 v12, 63, v1 // (op 575) 
s_cmp_eq_i32 s8, 0 // (op 575) 
s_cselect_b32 s9, 0, -1 // (op 575) 

v_lshrrev_b32 v23, 4, v12 // (op 575) 

s_and_b32 s8, s13, 31 // (op 575) 
s_lshl_b32 s18, 1, s8 // (op 575) 
s_add_i32 s19, s18, s9 // (op 575) 


s_and_b32 s9, s0, s19 // (op 575) 


s_add_i32 s0, s1, s9 // (op 575) 


s_ashr_i32 s1, s0, s8 // (op 575) 


s_lshl_b32 s0, s13, 24 // (op 575) 
s_ashr_i32 s8, s0, 31 // (op 575) 

s_xor_b32 s0, s1, s8 // (op 575) 

s_sub_i32 s1, s0, s8 // (op 575) 


s_mul_hi_i32 s0, s1, s14 // (op 575) 
s_add_i32 s8, s0, s1 // (op 575) 

s_ashr_i32 s0, s8, 31 // (op 575) 
s_cmp_eq_i32 s14, 0 // (op 575) 
s_mov_b32 s9, scc // (op 575) 
s_cmp_eq_i32 s9, 0 // (op 575) 
s_cselect_b32 s18, 0, -1 // (op 575) 

s_and_b32 s9, s15, 31 // (op 575) 
s_lshl_b32 s19, 1, s9 // (op 575) 
s_add_i32 s49, s19, s18 // (op 575) 


s_and_b32 s18, s0, s49 // (op 575) 


s_add_i32 s0, s8, s18 // (op 575) 


s_ashr_i32 s8, s0, s9 // (op 575) 


s_lshl_b32 s0, s15, 24 // (op 575) 
s_ashr_i32 s9, s0, 31 // (op 575) 

s_xor_b32 s0, s8, s9 // (op 575) 

s_sub_i32 s8, s0, s9 // (op 575) 


s_mul_i32 s0, s1, s45 // (op 575) 

s_sub_u32 s1, s16, s0 // (op 575) 

s_mul_i32 s0, s8, s45 // (op 575) 
s_add_u32 s0, s0, s1 // (op 575) 


s_mul_hi_i32 s1, s16, s11 // (op 575) 
s_add_i32 s8, s1, s16 // (op 575) 

s_ashr_i32 s1, s8, 31 // (op 575) 
s_cmp_eq_i32 s11, 0 // (op 575) 
s_mov_b32 s9, scc // (op 575) 
s_cmp_eq_i32 s9, 0 // (op 575) 
s_cselect_b32 s18, 0, -1 // (op 575) 

s_and_b32 s9, s46, 31 // (op 575) 
s_lshl_b32 s19, 1, s9 // (op 575) 
s_add_i32 s49, s19, s18 // (op 575) 


s_and_b32 s18, s1, s49 // (op 575) 


s_add_i32 s1, s8, s18 // (op 575) 


s_ashr_i32 s8, s1, s9 // (op 575) 


s_lshl_b32 s1, s46, 24 // (op 575) 
s_ashr_i32 s9, s1, 31 // (op 575) 

s_xor_b32 s1, s8, s9 // (op 575) 

s_sub_i32 s8, s1, s9 // (op 575) 


s_add_u32 s18, s4, 32 // (op 575) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 575) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 575) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 575) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 575) 

s_mul_i32 s1, s45, s18 // (op 575) 

s_mul_i32 s9, s10, s1 // (op 575) 

s_mul_i32 s1, s8, s9 // (op 575) 

s_sub_u32 s9, s16, s1 // (op 575) 

s_mov_b32 s1, s9 // (op 575) 
s_mul_hi_i32 s18, s1, s44 // (op 575) 
s_add_i32 s19, s18, s1 // (op 575) 


s_ashr_i32 s1, s19, 31 // (op 575) 
s_cmp_eq_i32 s44, 0 // (op 575) 
s_mov_b32 s18, scc // (op 575) 
s_cmp_eq_i32 s18, 0 // (op 575) 
s_cselect_b32 s49, 0, -1 // (op 575) 

s_and_b32 s18, s48, 31 // (op 575) 
s_lshl_b32 s50, 1, s18 // (op 575) 
s_add_i32 s51, s50, s49 // (op 575) 


s_and_b32 s49, s1, s51 // (op 575) 


s_add_i32 s1, s19, s49 // (op 575) 


s_ashr_i32 s19, s1, s18 // (op 575) 


s_lshl_b32 s1, s48, 24 // (op 575) 
s_ashr_i32 s18, s1, 31 // (op 575) 

s_xor_b32 s1, s19, s18 // (op 575) 

s_sub_i32 s19, s1, s18 // (op 575) 


s_mul_i32 s1, s47, s45 // (op 575) 
s_add_u32 s50, s6, 32 // (op 575) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 575) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 575) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 575) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 575) 

s_sub_i32 s18, s50, s1 // (op 575) 

s_mul_i32 s49, s19, s18 // (op 575) 


s_sub_u32 s18, s9, s49 // (op 575) 


s_add_u32 s9, s18, s1 // (op 575) 


s_cmp_eq_i32 s8, 0 // (op 575) 
s_mov_b32 s1, scc // (op 575) 

s_cmp_eq_i32 s1, 0 // (op 575) 
s_cselect_b32 s8, s9, s0 // (op 575) 



v_lshl_add_u32 v12, s8, 5, v13 // (op 575) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 575) 
s_lshr_b32 s8, s0, 1 // (op 575) 

v_lshl_add_u32 v13, s8, 3, v23 // (op 575) 


v_mul_lo_u32 v23, v12, s26 // (op 575) 
v_add_u32 v23, v23, v13 // (op 575) 


s_mul_i32 s1, s26, s7 // (op 577) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 577) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 577) least significant
s_mul_hi_u32 s9, s26, s6 // (op 577) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 577) most significant: sum
s_add_u32 s1, s1, s9 // (op 577) most significant: sum


s_mov_b32 s60, s28 // (op 577) reg expression
s_mov_b32 s61, s29 // (op 577) reg expression
s_mov_b32 s62, s0 // (op 577) reg expression
s_mov_b32 s63, 131072 // (op 577) call()

s_mul_hi_i32 s0, s16, s12 // (op 639) 
v_lshrrev_b32 v12, 6, v1 // (op 639) 
s_add_i32 s1, s0, s16 // (op 639) 

v_and_b32 v13, 1, v12 // (op 639) 

s_ashr_i32 s0, s1, 31 // (op 639) 
v_and_b32 v12, 15, v1 // (op 639) 
s_cmp_eq_i32 s12, 0 // (op 639) 
s_mov_b32 s8, scc // (op 639) 
v_lshl_add_u32 v14, v13, 4, v12 // (op 639) 


s_cmp_eq_i32 s8, 0 // (op 639) 
s_cselect_b32 s9, 0, -1 // (op 639) 

v_and_b32 v12, 63, v1 // (op 639) 
s_and_b32 s8, s13, 31 // (op 639) 
v_lshrrev_b32 v13, 4, v12 // (op 639) 

s_lshl_b32 s18, 1, s8 // (op 639) 
s_add_i32 s19, s18, s9 // (op 639) 


s_and_b32 s9, s0, s19 // (op 639) 


s_add_i32 s0, s1, s9 // (op 639) 


s_ashr_i32 s1, s0, s8 // (op 639) 


s_lshl_b32 s0, s13, 24 // (op 639) 
s_ashr_i32 s8, s0, 31 // (op 639) 

s_xor_b32 s0, s1, s8 // (op 639) 

s_sub_i32 s1, s0, s8 // (op 639) 


s_mul_hi_i32 s0, s1, s14 // (op 639) 
s_add_i32 s8, s0, s1 // (op 639) 

s_ashr_i32 s0, s8, 31 // (op 639) 
s_cmp_eq_i32 s14, 0 // (op 639) 
s_mov_b32 s9, scc // (op 639) 
s_cmp_eq_i32 s9, 0 // (op 639) 
s_cselect_b32 s18, 0, -1 // (op 639) 

s_and_b32 s9, s15, 31 // (op 639) 
s_lshl_b32 s19, 1, s9 // (op 639) 
s_add_i32 s49, s19, s18 // (op 639) 


s_and_b32 s18, s0, s49 // (op 639) 


s_add_i32 s0, s8, s18 // (op 639) 


s_ashr_i32 s8, s0, s9 // (op 639) 


s_lshl_b32 s0, s15, 24 // (op 639) 
s_ashr_i32 s9, s0, 31 // (op 639) 

s_xor_b32 s0, s8, s9 // (op 639) 

s_sub_i32 s8, s0, s9 // (op 639) 


s_add_u32 s18, s4, 32 // (op 639) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 639) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 639) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 639) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 639) 

s_mul_i32 s0, s8, s18 // (op 639) 

s_sub_i32 s8, s1, s0 // (op 639) 


s_mul_hi_i32 s9, s16, s11 // (op 639) 
s_add_i32 s0, s9, s16 // (op 639) 

s_ashr_i32 s1, s0, 31 // (op 639) 
s_cmp_eq_i32 s11, 0 // (op 639) 
s_mov_b32 s9, scc // (op 639) 
s_cmp_eq_i32 s9, 0 // (op 639) 
s_cselect_b32 s49, 0, -1 // (op 639) 

s_and_b32 s9, s46, 31 // (op 639) 
s_lshl_b32 s50, 1, s9 // (op 639) 
s_add_i32 s51, s50, s49 // (op 639) 


s_and_b32 s49, s1, s51 // (op 639) 


s_add_i32 s1, s0, s49 // (op 639) 


s_ashr_i32 s0, s1, s9 // (op 639) 


s_lshl_b32 s1, s46, 24 // (op 639) 
s_ashr_i32 s9, s1, 31 // (op 639) 

s_xor_b32 s1, s0, s9 // (op 639) 

s_sub_i32 s0, s1, s9 // (op 639) 


s_mul_i32 s1, s45, s18 // (op 639) 

s_mul_i32 s9, s10, s1 // (op 639) 

s_mul_i32 s1, s0, s9 // (op 639) 

s_sub_u32 s9, s16, s1 // (op 639) 

s_mul_hi_i32 s1, s9, s44 // (op 639) 
s_add_i32 s18, s1, s9 // (op 639) 


s_ashr_i32 s1, s18, 31 // (op 639) 
s_cmp_eq_i32 s44, 0 // (op 639) 
s_mov_b32 s9, scc // (op 639) 
s_cmp_eq_i32 s9, 0 // (op 639) 
s_cselect_b32 s19, 0, -1 // (op 639) 

s_and_b32 s9, s48, 31 // (op 639) 
s_lshl_b32 s49, 1, s9 // (op 639) 
s_add_i32 s50, s49, s19 // (op 639) 


s_and_b32 s19, s1, s50 // (op 639) 


s_add_i32 s1, s18, s19 // (op 639) 


s_ashr_i32 s18, s1, s9 // (op 639) 


s_lshl_b32 s1, s48, 24 // (op 639) 
s_ashr_i32 s9, s1, 31 // (op 639) 

s_xor_b32 s1, s18, s9 // (op 639) 

s_sub_i32 s18, s1, s9 // (op 639) 


s_cmp_eq_i32 s0, 0 // (op 639) 
s_mov_b32 s1, scc // (op 639) 

s_cmp_eq_i32 s1, 0 // (op 639) 
s_cselect_b32 s0, s18, s8 // (op 639) 



v_lshl_add_u32 v12, s0, 5, v14 // (op 639) 


v_mul_lo_u32 v14, v12, s30 // (op 639) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 639) 
s_lshr_b32 s8, s0, 1 // (op 639) 

v_lshl_add_u32 v12, s8, 3, v13 // (op 639) 


v_add_u32 v13, v12, v14 // (op 639) 


s_mul_i32 s1, s30, s5 // (op 641) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 641) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 641) least significant
s_mul_hi_u32 s9, s30, s4 // (op 641) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 641) most significant: sum
s_add_u32 s1, s1, s9 // (op 641) most significant: sum


s_mov_b32 s64, s32 // (op 641) reg expression
s_mov_b32 s65, s33 // (op 641) reg expression
s_mov_b32 s66, s0 // (op 641) reg expression
s_mov_b32 s67, 131072 // (op 641) call()

v_and_b32 v12, -8, v1 // (op 732) 
v_sub_u32 v14, v1, v12 // (op 732) 

v_lshlrev_b32 v12, 4, v14 // (op 732) 

v_lshrrev_b32 v14, 3, v1 // (op 732) 
v_lshl_add_u32 v15, v14, 7, v12 // (op 732) 


s_nop 0
v_readfirstlane_b32 s0, v15 // (op 732) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 784) 
v_sub_u32 v14, v1, v12 // (op 784) 

v_lshlrev_b32 v12, 4, v14 // (op 784) 

v_lshrrev_b32 v14, 3, v1 // (op 784) 
v_lshl_add_u32 v15, v14, 7, v12 // (op 784) 


s_nop 0
v_readfirstlane_b32 s1, v15 // (op 784) Wait state hazard: VALU Write Readlane Hazard

s_ashr_i64 s[8:9], s[34:35], 7 // (op 153) 
s_and_b32 s18, s8, 4294967294 // (op 153) 

s_ashr_i64 s[8:9], s[34:35], 7 // (op 152) 
s_mov_b32 s19, s8 // (op 152) convert

s_cmp_lt_u32 s18, s19 // (op 152) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_152 // (op 152) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_152 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_152:
s_barrier  // (op 958) 
s_add_u32 s18, s18, 1 // (op 154) 


v_mov_b64 v[14:15], v[10:11] // (op 397) 
s_add_u32 s19, 4096, s0 // (op 397) 

s_mov_b32 m0, s19 // (op 397) reg expression
buffer_load_dwordx4 v14, s[52:55], 0 offen offset: 0 lds // (op 397) Load value direct to lds


v_add_co_u32 v10, vcc, 128, v10 // (op 434) least significant half
v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 434) most significant half
s_barrier  // (op 974) 


v_mov_b64 v[14:15], v[8:9] // (op 403) 

s_mov_b32 m0, s1 // (op 403) reg expression
buffer_load_dwordx4 v14, s[56:59], 0 offen offset: 0 lds // (op 403) Load value direct to lds

v_add_co_u32 v8, vcc, 128, v8 // (op 503) least significant half
v_addc_co_u32 v9, vcc, 0, v9, vcc // (op 503) most significant half
s_waitcnt vmcnt(1)
s_barrier  // (op 964) 



ds_read_b128 v[14:17], v3 offset:4096 // (op 160) Load local data 
ds_read_b128 v[18:21], v3 offset:4160 // (op 160) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 980) 



ds_read_b128 v[24:27], v4 // (op 162) Load local data 
ds_read_b128 v[28:31], v4 offset:64 // (op 162) Load local data 


v_mov_b32 v12, v23 // (op 164) 

buffer_load_ubyte v22, v12, s[60:63], 0 offen offset: 0 // (op 164) Load value

v_add_u32 v23, 4, v23 // (op 567) 


v_mov_b32 v12, v13 // (op 166) 

buffer_load_ubyte v32, v12, s[64:67], 0 offen offset: 0 // (op 166) Load value

v_add_u32 v13, 4, v13 // (op 631) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[14:21], v[24:31], a[0:3], v22, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 167) 




s_ashr_i64 s[8:9], s[34:35], 7 // (op 152) 
s_mov_b32 s19, s8 // (op 152) convert

s_cmp_lt_u32 s18, s19 // (op 152) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_152 // (op 152) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_152 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_152:




















s_barrier  // (op 1022) 




v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_114
s_nop 0
v_accvgpr_read v8, a0 // (op 42) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 42) 
v_accvgpr_read v10, a2 // (op 42) 
v_accvgpr_read v11, a3 // (op 42) 
v_mul_f32 v12, v7, v8 // (op 42) 
v_mul_f32 v13, v7, v9 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 

s_mul_hi_i32 s0, s16, s12 // (op 825) 
v_lshrrev_b32 v3, 7, v1 // (op 825) 
s_add_i32 s1, s0, s16 // (op 825) 

v_and_b32 v4, 63, v1 // (op 825) 
s_ashr_i32 s0, s1, 31 // (op 825) 
v_lshrrev_b32 v5, 4, v4 // (op 825) 
s_cmp_eq_i32 s12, 0 // (op 825) 
s_mov_b32 s2, scc // (op 825) 
v_lshlrev_b32 v6, 2, v5 // (op 825) 

s_cmp_eq_i32 s2, 0 // (op 825) 
s_cselect_b32 s3, 0, -1 // (op 825) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 825) 


s_and_b32 s2, s13, 31 // (op 825) 
v_lshrrev_b32 v3, 2, v4 // (op 825) 

s_lshl_b32 s8, 1, s2 // (op 825) 
v_and_b32 v4, 3, v3 // (op 825) 

s_add_i32 s9, s8, s3 // (op 825) 


v_and_b32 v3, 3, v1 // (op 825) 
s_and_b32 s3, s0, s9 // (op 825) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 825) 


s_add_i32 s0, s1, s3 // (op 825) 


v_lshrrev_b32 v3, 6, v1 // (op 825) 
s_ashr_i32 s1, s0, s2 // (op 825) 


v_and_b32 v4, 1, v3 // (op 825) 

s_lshl_b32 s0, s13, 24 // (op 825) 
v_lshl_add_u32 v3, v4, 4, v6 // (op 825) 


s_ashr_i32 s2, s0, 31 // (op 825) 

s_xor_b32 s0, s1, s2 // (op 825) 

s_sub_i32 s1, s0, s2 // (op 825) 


s_mul_hi_i32 s0, s1, s14 // (op 825) 
s_add_i32 s2, s0, s1 // (op 825) 

s_ashr_i32 s0, s2, 31 // (op 825) 
s_cmp_eq_i32 s14, 0 // (op 825) 
s_mov_b32 s3, scc // (op 825) 
s_cmp_eq_i32 s3, 0 // (op 825) 
s_cselect_b32 s8, 0, -1 // (op 825) 

s_and_b32 s3, s15, 31 // (op 825) 
s_lshl_b32 s9, 1, s3 // (op 825) 
s_add_i32 s18, s9, s8 // (op 825) 


s_and_b32 s8, s0, s18 // (op 825) 


s_add_i32 s0, s2, s8 // (op 825) 


s_ashr_i32 s2, s0, s3 // (op 825) 


s_lshl_b32 s0, s15, 24 // (op 825) 
s_ashr_i32 s3, s0, 31 // (op 825) 

s_xor_b32 s0, s2, s3 // (op 825) 

s_sub_i32 s2, s0, s3 // (op 825) 


s_mul_i32 s0, s1, s45 // (op 825) 
s_sub_u32 s3, s16, s0 // (op 825) 

s_mul_i32 s0, s2, s45 // (op 825) 
s_add_u32 s0, s0, s3 // (op 825) 

s_mul_hi_i32 s3, s16, s11 // (op 825) 
s_add_i32 s8, s3, s16 // (op 825) 

s_ashr_i32 s3, s8, 31 // (op 825) 
s_cmp_eq_i32 s11, 0 // (op 825) 
s_mov_b32 s9, scc // (op 825) 
s_cmp_eq_i32 s9, 0 // (op 825) 
s_cselect_b32 s18, 0, -1 // (op 825) 

s_and_b32 s9, s46, 31 // (op 825) 
s_lshl_b32 s19, 1, s9 // (op 825) 
s_add_i32 s20, s19, s18 // (op 825) 


s_and_b32 s18, s3, s20 // (op 825) 


s_add_i32 s3, s8, s18 // (op 825) 


s_ashr_i32 s8, s3, s9 // (op 825) 


s_lshl_b32 s3, s46, 24 // (op 825) 
s_ashr_i32 s9, s3, 31 // (op 825) 

s_xor_b32 s3, s8, s9 // (op 825) 

s_sub_i32 s8, s3, s9 // (op 825) 


s_add_u32 s18, s4, 32 // (op 825) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 825) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 825) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 825) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 825) 

s_mul_i32 s3, s45, s18 // (op 825) 
s_mul_i32 s9, s10, s3 // (op 825) 

s_mul_i32 s3, s8, s9 // (op 825) 

s_sub_u32 s9, s16, s3 // (op 825) 

s_mov_b32 s3, s9 // (op 825) 
s_mul_hi_i32 s20, s3, s44 // (op 825) 
s_add_i32 s21, s20, s3 // (op 825) 


s_ashr_i32 s3, s21, 31 // (op 825) 
s_cmp_eq_i32 s44, 0 // (op 825) 
s_mov_b32 s20, scc // (op 825) 
s_cmp_eq_i32 s20, 0 // (op 825) 
s_cselect_b32 s22, 0, -1 // (op 825) 

s_and_b32 s20, s48, 31 // (op 825) 
s_lshl_b32 s23, 1, s20 // (op 825) 
s_add_i32 s24, s23, s22 // (op 825) 


s_and_b32 s22, s3, s24 // (op 825) 


s_add_i32 s3, s21, s22 // (op 825) 


s_ashr_i32 s21, s3, s20 // (op 825) 


s_lshl_b32 s3, s48, 24 // (op 825) 
s_ashr_i32 s20, s3, 31 // (op 825) 

s_xor_b32 s3, s21, s20 // (op 825) 

s_sub_i32 s21, s3, s20 // (op 825) 


s_mul_i32 s3, s47, s45 // (op 825) 
s_add_u32 s22, s6, 32 // (op 825) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 825) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 825) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 825) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 825) 

s_sub_i32 s20, s22, s3 // (op 825) 

s_mul_i32 s22, s21, s20 // (op 825) 

s_sub_u32 s20, s9, s22 // (op 825) 


s_add_u32 s9, s20, s3 // (op 825) 


s_cmp_eq_i32 s8, 0 // (op 825) 
s_mov_b32 s3, scc // (op 825) 

s_cmp_eq_i32 s3, 0 // (op 825) 
s_cselect_b32 s8, s9, s0 // (op 825) 


s_mul_i32 s0, s2, s18 // (op 825) 


v_lshl_add_u32 v4, s8, 5, v5 // (op 825) 


s_sub_i32 s2, s1, s0 // (op 825) 


s_cmp_eq_i32 s3, 0 // (op 825) 
s_cselect_b32 s0, s21, s2 // (op 825) 



v_lshl_add_u32 v5, s0, 5, v3 // (op 825) 


v_mul_lo_u32 v3, v5, s36 // (op 825) 

v_add_lshl_u32 v5, v4, v3, 2 // (op 825) 


s_mul_i32 s9, s36, s5 // (op 827) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 827) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 827) least significant
s_mul_hi_u32 s1, s36, s4 // (op 827) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 827) most significant: sum
s_add_u32 s9, s9, s1 // (op 827) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 827) 

s_mov_b32 s32, s38 // (op 827) reg expression
s_mov_b32 s33, s39 // (op 827) reg expression
s_mov_b32 s34, s0 // (op 827) reg expression
s_mov_b32 s35, 131072 // (op 827) call()




buffer_store_dwordx4 v[12:15], v5, s[32:35], 0 offen // (op 48) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_114
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_114:
s_mul_hi_i32 s0, s16, s12 // (op 702) 
v_lshrrev_b32 v3, 7, v1 // (op 702) 
s_add_i32 s1, s0, s16 // (op 702) 

v_and_b32 v4, 63, v1 // (op 702) 
s_ashr_i32 s0, s1, 31 // (op 702) 
v_lshrrev_b32 v5, 4, v4 // (op 702) 
s_cmp_eq_i32 s12, 0 // (op 702) 
s_mov_b32 s2, scc // (op 702) 
v_lshlrev_b32 v6, 2, v5 // (op 702) 

s_cmp_eq_i32 s2, 0 // (op 702) 
s_cselect_b32 s3, 0, -1 // (op 702) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 702) 


s_and_b32 s2, s13, 31 // (op 702) 
v_lshrrev_b32 v3, 2, v4 // (op 702) 

s_lshl_b32 s8, 1, s2 // (op 702) 
v_and_b32 v4, 3, v3 // (op 702) 

s_add_i32 s9, s8, s3 // (op 702) 


v_and_b32 v3, 3, v1 // (op 702) 
s_and_b32 s3, s0, s9 // (op 702) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 702) 


s_add_i32 s0, s1, s3 // (op 702) 


v_lshrrev_b32 v3, 6, v1 // (op 702) 
s_ashr_i32 s1, s0, s2 // (op 702) 


v_and_b32 v4, 1, v3 // (op 702) 

s_lshl_b32 s0, s13, 24 // (op 702) 
v_lshl_add_u32 v3, v4, 4, v6 // (op 702) 


s_ashr_i32 s2, s0, 31 // (op 702) 

s_xor_b32 s0, s1, s2 // (op 702) 

s_sub_i32 s1, s0, s2 // (op 702) 


s_mul_hi_i32 s0, s1, s14 // (op 702) 
s_add_i32 s2, s0, s1 // (op 702) 

s_ashr_i32 s0, s2, 31 // (op 702) 
s_cmp_eq_i32 s14, 0 // (op 702) 
s_mov_b32 s3, scc // (op 702) 
s_cmp_eq_i32 s3, 0 // (op 702) 
s_cselect_b32 s8, 0, -1 // (op 702) 

s_and_b32 s3, s15, 31 // (op 702) 
s_lshl_b32 s9, 1, s3 // (op 702) 
s_add_i32 s18, s9, s8 // (op 702) 


s_and_b32 s8, s0, s18 // (op 702) 


s_add_i32 s0, s2, s8 // (op 702) 


s_ashr_i32 s2, s0, s3 // (op 702) 


s_lshl_b32 s0, s15, 24 // (op 702) 
s_ashr_i32 s3, s0, 31 // (op 702) 

s_xor_b32 s0, s2, s3 // (op 702) 

s_sub_i32 s2, s0, s3 // (op 702) 


s_mul_i32 s0, s1, s45 // (op 702) 
s_sub_u32 s3, s16, s0 // (op 702) 

s_mul_i32 s0, s2, s45 // (op 702) 
s_add_u32 s0, s0, s3 // (op 702) 

s_mul_hi_i32 s3, s16, s11 // (op 702) 
s_add_i32 s8, s3, s16 // (op 702) 

s_ashr_i32 s3, s8, 31 // (op 702) 
s_cmp_eq_i32 s11, 0 // (op 702) 
s_mov_b32 s9, scc // (op 702) 
s_cmp_eq_i32 s9, 0 // (op 702) 
s_cselect_b32 s18, 0, -1 // (op 702) 

s_and_b32 s9, s46, 31 // (op 702) 
s_lshl_b32 s19, 1, s9 // (op 702) 
s_add_i32 s20, s19, s18 // (op 702) 


s_and_b32 s18, s3, s20 // (op 702) 


s_add_i32 s3, s8, s18 // (op 702) 


s_ashr_i32 s8, s3, s9 // (op 702) 


s_lshl_b32 s3, s46, 24 // (op 702) 
s_ashr_i32 s9, s3, 31 // (op 702) 

s_xor_b32 s3, s8, s9 // (op 702) 

s_sub_i32 s8, s3, s9 // (op 702) 


s_add_u32 s18, s4, 32 // (op 702) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 702) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 702) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 702) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 702) 

s_mul_i32 s3, s45, s18 // (op 702) 
s_mul_i32 s9, s10, s3 // (op 702) 

s_mul_i32 s3, s8, s9 // (op 702) 

s_sub_u32 s9, s16, s3 // (op 702) 

s_mov_b32 s3, s9 // (op 702) 
s_mul_hi_i32 s20, s3, s44 // (op 702) 
s_add_i32 s21, s20, s3 // (op 702) 


s_ashr_i32 s3, s21, 31 // (op 702) 
s_cmp_eq_i32 s44, 0 // (op 702) 
s_mov_b32 s20, scc // (op 702) 
s_cmp_eq_i32 s20, 0 // (op 702) 
s_cselect_b32 s22, 0, -1 // (op 702) 

s_and_b32 s20, s48, 31 // (op 702) 
s_lshl_b32 s23, 1, s20 // (op 702) 
s_add_i32 s24, s23, s22 // (op 702) 


s_and_b32 s22, s3, s24 // (op 702) 


s_add_i32 s3, s21, s22 // (op 702) 


s_ashr_i32 s21, s3, s20 // (op 702) 


s_lshl_b32 s3, s48, 24 // (op 702) 
s_ashr_i32 s20, s3, 31 // (op 702) 

s_xor_b32 s3, s21, s20 // (op 702) 

s_sub_i32 s21, s3, s20 // (op 702) 


s_mul_i32 s3, s47, s45 // (op 702) 
s_add_u32 s22, s6, 32 // (op 702) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 702) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 702) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 702) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 702) 

s_sub_i32 s20, s22, s3 // (op 702) 

s_mul_i32 s22, s21, s20 // (op 702) 

s_sub_u32 s20, s9, s22 // (op 702) 


s_add_u32 s9, s20, s3 // (op 702) 


s_cmp_eq_i32 s8, 0 // (op 702) 
s_mov_b32 s3, scc // (op 702) 

s_cmp_eq_i32 s3, 0 // (op 702) 
s_cselect_b32 s8, s9, s0 // (op 702) 


s_mul_i32 s0, s2, s18 // (op 702) 


v_lshl_add_u32 v4, s8, 5, v5 // (op 702) 


s_sub_i32 s2, s1, s0 // (op 702) 


s_cmp_eq_i32 s3, 0 // (op 702) 
s_cselect_b32 s0, s21, s2 // (op 702) 



v_lshl_add_u32 v5, s0, 5, v3 // (op 702) 


v_mul_lo_u32 v3, v5, s40 // (op 702) 

v_add_lshl_u32 v5, v4, v3, 2 // (op 702) 


s_mul_i32 s9, s40, s5 // (op 704) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 704) most significant: low of high * low
s_mul_i32 s8, s40, s4 // (op 704) least significant
s_mul_hi_u32 s1, s40, s4 // (op 704) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 704) most significant: sum
s_add_u32 s9, s9, s1 // (op 704) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 704) 

s_mov_b32 s28, s42 // (op 704) reg expression
s_mov_b32 s29, s43 // (op 704) reg expression
s_mov_b32 s30, s0 // (op 704) reg expression
s_mov_b32 s31, 131072 // (op 704) call()





buffer_load_dwordx4 v[8:11], v5, s[28:31], 0 offen // (op 126) Load value



v_accvgpr_read v12, a0 // (op 123) 
v_accvgpr_read v13, a1 // (op 123) 
v_accvgpr_read v14, a2 // (op 123) 
v_accvgpr_read v15, a3 // (op 123) 
v_mul_f32 v16, v7, v12 // (op 123) 
v_mul_f32 v17, v7, v13 // (op 123) 
v_mul_f32 v18, v7, v14 // (op 123) 
v_mul_f32 v19, v7, v15 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v0, v8 // (op 127) 
v_mul_f32 v13, v0, v9 // (op 127) 
v_mul_f32 v14, v0, v10 // (op 127) 
v_mul_f32 v15, v0, v11 // (op 127) 

v_add_f32 v8, v12, v16 // (op 124) 
v_add_f32 v9, v13, v17 // (op 124) 
v_add_f32 v10, v14, v18 // (op 124) 
v_add_f32 v11, v15, v19 // (op 124) 


s_mul_hi_i32 s0, s16, s12 // (op 842) 
v_lshrrev_b32 v3, 7, v1 // (op 842) 
s_add_i32 s1, s0, s16 // (op 842) 

v_and_b32 v4, 63, v1 // (op 842) 
s_ashr_i32 s0, s1, 31 // (op 842) 
v_lshrrev_b32 v5, 4, v4 // (op 842) 
s_cmp_eq_i32 s12, 0 // (op 842) 
s_mov_b32 s2, scc // (op 842) 
v_lshlrev_b32 v6, 2, v5 // (op 842) 

s_cmp_eq_i32 s2, 0 // (op 842) 
s_cselect_b32 s3, 0, -1 // (op 842) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 842) 


s_and_b32 s2, s13, 31 // (op 842) 
v_lshrrev_b32 v3, 2, v4 // (op 842) 

s_lshl_b32 s8, 1, s2 // (op 842) 
v_and_b32 v4, 3, v3 // (op 842) 

s_add_i32 s9, s8, s3 // (op 842) 


v_and_b32 v3, 3, v1 // (op 842) 
s_and_b32 s3, s0, s9 // (op 842) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 842) 


s_add_i32 s0, s1, s3 // (op 842) 


v_lshrrev_b32 v3, 6, v1 // (op 842) 
s_ashr_i32 s1, s0, s2 // (op 842) 


v_and_b32 v4, 1, v3 // (op 842) 

s_lshl_b32 s0, s13, 24 // (op 842) 
v_lshl_add_u32 v3, v4, 4, v6 // (op 842) 


s_ashr_i32 s2, s0, 31 // (op 842) 

s_xor_b32 s0, s1, s2 // (op 842) 

s_sub_i32 s1, s0, s2 // (op 842) 


s_mul_hi_i32 s0, s1, s14 // (op 842) 
s_add_i32 s2, s0, s1 // (op 842) 

s_ashr_i32 s0, s2, 31 // (op 842) 
s_cmp_eq_i32 s14, 0 // (op 842) 
s_mov_b32 s3, scc // (op 842) 
s_cmp_eq_i32 s3, 0 // (op 842) 
s_cselect_b32 s8, 0, -1 // (op 842) 

s_and_b32 s3, s15, 31 // (op 842) 
s_lshl_b32 s9, 1, s3 // (op 842) 
s_add_i32 s18, s9, s8 // (op 842) 


s_and_b32 s8, s0, s18 // (op 842) 


s_add_i32 s0, s2, s8 // (op 842) 


s_ashr_i32 s2, s0, s3 // (op 842) 


s_lshl_b32 s0, s15, 24 // (op 842) 
s_ashr_i32 s3, s0, 31 // (op 842) 

s_xor_b32 s0, s2, s3 // (op 842) 

s_sub_i32 s2, s0, s3 // (op 842) 


s_mul_i32 s0, s1, s45 // (op 842) 
s_sub_u32 s3, s16, s0 // (op 842) 

s_mul_i32 s0, s2, s45 // (op 842) 
s_add_u32 s0, s0, s3 // (op 842) 

s_mul_hi_i32 s3, s16, s11 // (op 842) 
s_add_i32 s8, s3, s16 // (op 842) 

s_ashr_i32 s3, s8, 31 // (op 842) 
s_cmp_eq_i32 s11, 0 // (op 842) 
s_mov_b32 s9, scc // (op 842) 
s_cmp_eq_i32 s9, 0 // (op 842) 
s_cselect_b32 s18, 0, -1 // (op 842) 

s_and_b32 s9, s46, 31 // (op 842) 
s_lshl_b32 s19, 1, s9 // (op 842) 
s_add_i32 s20, s19, s18 // (op 842) 


s_and_b32 s18, s3, s20 // (op 842) 


s_add_i32 s3, s8, s18 // (op 842) 


s_ashr_i32 s8, s3, s9 // (op 842) 


s_lshl_b32 s3, s46, 24 // (op 842) 
s_ashr_i32 s9, s3, 31 // (op 842) 

s_xor_b32 s3, s8, s9 // (op 842) 

s_sub_i32 s8, s3, s9 // (op 842) 


s_add_u32 s18, s4, 32 // (op 842) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 842) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 842) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 842) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 842) 

s_mul_i32 s3, s45, s18 // (op 842) 
s_mul_i32 s9, s10, s3 // (op 842) 

s_mul_i32 s3, s8, s9 // (op 842) 

s_sub_u32 s9, s16, s3 // (op 842) 

s_mov_b32 s3, s9 // (op 842) 
s_mul_hi_i32 s20, s3, s44 // (op 842) 
s_add_i32 s21, s20, s3 // (op 842) 


s_ashr_i32 s3, s21, 31 // (op 842) 
s_cmp_eq_i32 s44, 0 // (op 842) 
s_mov_b32 s20, scc // (op 842) 
s_cmp_eq_i32 s20, 0 // (op 842) 
s_cselect_b32 s22, 0, -1 // (op 842) 

s_and_b32 s20, s48, 31 // (op 842) 
s_lshl_b32 s23, 1, s20 // (op 842) 
s_add_i32 s24, s23, s22 // (op 842) 


s_and_b32 s22, s3, s24 // (op 842) 


s_add_i32 s3, s21, s22 // (op 842) 


s_ashr_i32 s21, s3, s20 // (op 842) 


s_lshl_b32 s3, s48, 24 // (op 842) 
s_ashr_i32 s20, s3, 31 // (op 842) 

s_xor_b32 s3, s21, s20 // (op 842) 

s_sub_i32 s21, s3, s20 // (op 842) 


s_mul_i32 s3, s47, s45 // (op 842) 
s_add_u32 s22, s6, 32 // (op 842) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 842) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 842) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 842) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 842) 

s_sub_i32 s20, s22, s3 // (op 842) 

s_mul_i32 s22, s21, s20 // (op 842) 

s_sub_u32 s20, s9, s22 // (op 842) 


s_add_u32 s9, s20, s3 // (op 842) 


s_cmp_eq_i32 s8, 0 // (op 842) 
s_mov_b32 s3, scc // (op 842) 

s_cmp_eq_i32 s3, 0 // (op 842) 
s_cselect_b32 s8, s9, s0 // (op 842) 


s_mul_i32 s0, s2, s18 // (op 842) 


v_lshl_add_u32 v4, s8, 5, v5 // (op 842) 


s_sub_i32 s2, s1, s0 // (op 842) 


s_cmp_eq_i32 s3, 0 // (op 842) 
s_cselect_b32 s0, s21, s2 // (op 842) 



v_lshl_add_u32 v5, s0, 5, v3 // (op 842) 


v_mul_lo_u32 v3, v5, s36 // (op 842) 

v_add_lshl_u32 v5, v4, v3, 2 // (op 842) 


s_mul_i32 s9, s36, s5 // (op 844) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 844) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 844) least significant
s_mul_hi_u32 s1, s36, s4 // (op 844) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 844) most significant: sum
s_add_u32 s9, s9, s1 // (op 844) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 844) 

s_mov_b32 s32, s38 // (op 844) reg expression
s_mov_b32 s33, s39 // (op 844) reg expression
s_mov_b32 s34, s0 // (op 844) reg expression
s_mov_b32 s35, 131072 // (op 844) call()




buffer_store_dwordx4 v[8:11], v5, s[32:35], 0 offen // (op 125) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_114:
















s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 44
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16384 // lds bytes
  .amdhsa_accum_offset 40
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
    macrotile: [32, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 70
    .vgpr_count: 40
    .agpr_count: 4
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 36
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 164
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 168
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

