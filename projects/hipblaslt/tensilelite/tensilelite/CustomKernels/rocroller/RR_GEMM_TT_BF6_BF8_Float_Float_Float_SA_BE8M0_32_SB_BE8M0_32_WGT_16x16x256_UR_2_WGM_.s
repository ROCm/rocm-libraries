// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_,@function
RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 873) 
v_and_b32 v3, 15, v1 // (op 873) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 873) 


v_and_b32 v0, 63, v1 // (op 873) 
v_lshrrev_b32 v3, 4, v0 // (op 873) 

v_lshlrev_b32 v0, 5, v3 // (op 873) 

v_lshl_add_u32 v3, v4, 8, v0 // (op 873) 


v_mul_lo_u32 v0, v3, 6 // (op 873) 

v_lshrrev_b32 v3, 3, v0 // (op 873) 

v_and_b32 v0, 48, v1 // (op 925) 
v_and_b32 v4, 63, v1 // (op 925) 
v_sub_u32 v5, v4, v0 // (op 925) 

v_lshrrev_b32 v4, 1, v5 // (op 925) 
v_add_lshl_u32 v6, v0, v4, 4 // (op 925) 


v_and_b32 v0, 1, v5 // (op 925) 

v_lshlrev_b32 v4, 3, v0 // (op 925) 

v_add_u32 v0, v6, v4 // (op 925) 


v_lshrrev_b32 v4, 6, v1 // (op 978) 
v_and_b32 v5, 15, v1 // (op 978) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 978) 


v_and_b32 v4, 63, v1 // (op 978) 
v_lshrrev_b32 v5, 4, v4 // (op 978) 

v_lshl_add_u32 v4, v6, 3, v5 // (op 978) 


v_and_b32 v5, 48, v1 // (op 1031) 
v_and_b32 v6, 15, v1 // (op 1031) 
v_add_u32 v7, v5, v6 // (op 1031) 


v_lshrrev_b32 v5, 6, v1 // (op 1089) 
v_and_b32 v6, 15, v1 // (op 1089) 
v_lshl_add_u32 v8, v5, 4, v6 // (op 1089) 


v_and_b32 v5, 63, v1 // (op 1089) 
v_lshrrev_b32 v6, 4, v5 // (op 1089) 

v_lshlrev_b32 v5, 5, v6 // (op 1089) 

v_lshl_add_u32 v6, v8, 8, v5 // (op 1089) 


v_mul_lo_u32 v5, v6, 6 // (op 1089) 

v_lshrrev_b32 v6, 3, v5 // (op 1089) 

v_and_b32 v5, 48, v1 // (op 1128) 
v_and_b32 v8, 63, v1 // (op 1128) 
v_sub_u32 v9, v8, v5 // (op 1128) 

v_lshrrev_b32 v8, 1, v9 // (op 1128) 
v_add_lshl_u32 v10, v5, v8, 4 // (op 1128) 


v_and_b32 v5, 1, v9 // (op 1128) 

v_lshlrev_b32 v8, 3, v5 // (op 1128) 

v_add_u32 v5, v10, v8 // (op 1128) 


v_lshrrev_b32 v8, 6, v1 // (op 1152) 
v_and_b32 v9, 15, v1 // (op 1152) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 1152) 


v_and_b32 v8, 63, v1 // (op 1152) 
v_lshrrev_b32 v9, 4, v8 // (op 1152) 

v_lshl_add_u32 v8, v10, 3, v9 // (op 1152) 


v_and_b32 v9, 48, v1 // (op 1176) 
v_and_b32 v10, 15, v1 // (op 1176) 
v_add_u32 v11, v9, v10 // (op 1176) 


v_mov_b32 v9, s8 // (op 2) Move value
v_mov_b32 v10, s9 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_mul_hi_i32 s0, s16, s11 // (op 583) 
v_lshrrev_b32 v12, 2, v1 // (op 583) 
s_add_i32 s1, s0, s16 // (op 583) 

v_and_b32 v13, -4, v1 // (op 583) 
s_ashr_i32 s0, s1, 31 // (op 583) 
v_sub_u32 v14, v1, v13 // (op 583) 

s_cmp_eq_i32 s11, 0 // (op 583) 
s_mov_b32 s8, scc // (op 583) 
v_lshlrev_b32 v13, 6, v14 // (op 583) 

s_cmp_eq_i32 s8, 0 // (op 583) 
s_cselect_b32 s9, 0, -1 // (op 583) 

v_mov_b32 v15, 0 // (op 583) convert
v_mov_b32 v14, v13 // (op 583) convert

s_and_b32 s8, s12, 31 // (op 583) 
s_lshl_b32 s18, 1, s8 // (op 583) 
s_add_i32 s19, s18, s9 // (op 583) 


s_and_b32 s9, s0, s19 // (op 583) 


s_add_i32 s0, s1, s9 // (op 583) 


s_ashr_i32 s1, s0, s8 // (op 583) 


s_lshl_b32 s0, s12, 24 // (op 583) 
s_ashr_i32 s8, s0, 31 // (op 583) 

s_xor_b32 s0, s1, s8 // (op 583) 

s_sub_i32 s1, s0, s8 // (op 583) 


s_mul_hi_i32 s0, s1, s13 // (op 583) 
s_add_i32 s8, s0, s1 // (op 583) 

s_ashr_i32 s0, s8, 31 // (op 583) 
s_cmp_eq_i32 s13, 0 // (op 583) 
s_mov_b32 s9, scc // (op 583) 
s_cmp_eq_i32 s9, 0 // (op 583) 
s_cselect_b32 s18, 0, -1 // (op 583) 

s_and_b32 s9, s14, 31 // (op 583) 
s_lshl_b32 s19, 1, s9 // (op 583) 
s_add_i32 s51, s19, s18 // (op 583) 


s_and_b32 s18, s0, s51 // (op 583) 


s_add_i32 s0, s8, s18 // (op 583) 


s_ashr_i32 s8, s0, s9 // (op 583) 


s_lshl_b32 s0, s14, 24 // (op 583) 
s_ashr_i32 s9, s0, 31 // (op 583) 

s_xor_b32 s0, s8, s9 // (op 583) 

s_sub_i32 s8, s0, s9 // (op 583) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s46 // (op 583) 

s_sub_u32 s1, s16, s0 // (op 583) 

s_mul_i32 s0, s8, s46 // (op 583) 
s_add_u32 s0, s0, s1 // (op 583) 


s_mul_hi_i32 s1, s16, s10 // (op 583) 
s_add_i32 s8, s1, s16 // (op 583) 

s_ashr_i32 s1, s8, 31 // (op 583) 
s_cmp_eq_i32 s10, 0 // (op 583) 
s_mov_b32 s9, scc // (op 583) 
s_cmp_eq_i32 s9, 0 // (op 583) 
s_cselect_b32 s18, 0, -1 // (op 583) 

s_and_b32 s9, s49, 31 // (op 583) 
s_lshl_b32 s19, 1, s9 // (op 583) 
s_add_i32 s51, s19, s18 // (op 583) 


s_and_b32 s18, s1, s51 // (op 583) 


s_add_i32 s1, s8, s18 // (op 583) 


s_ashr_i32 s8, s1, s9 // (op 583) 


s_lshl_b32 s1, s49, 24 // (op 583) 
s_ashr_i32 s9, s1, 31 // (op 583) 

s_xor_b32 s1, s8, s9 // (op 583) 

s_sub_i32 s8, s1, s9 // (op 583) 


s_add_u32 s18, s2, 16 // (op 583) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 583) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 583) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 583) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 583) 

s_mul_i32 s1, s46, s18 // (op 583) 

s_mul_i32 s9, s50, s1 // (op 583) 

s_mul_i32 s1, s8, s9 // (op 583) 

s_sub_u32 s9, s16, s1 // (op 583) 

s_mov_b32 s1, s9 // (op 583) 
s_mul_hi_i32 s18, s1, s15 // (op 583) 
s_add_i32 s19, s18, s1 // (op 583) 


s_ashr_i32 s1, s19, 31 // (op 583) 
s_cmp_eq_i32 s15, 0 // (op 583) 
s_mov_b32 s18, scc // (op 583) 
s_cmp_eq_i32 s18, 0 // (op 583) 
s_cselect_b32 s51, 0, -1 // (op 583) 

s_and_b32 s18, s48, 31 // (op 583) 
s_lshl_b32 s52, 1, s18 // (op 583) 
s_add_i32 s53, s52, s51 // (op 583) 


s_and_b32 s51, s1, s53 // (op 583) 


s_add_i32 s1, s19, s51 // (op 583) 


s_ashr_i32 s19, s1, s18 // (op 583) 


s_lshl_b32 s1, s48, 24 // (op 583) 
s_ashr_i32 s18, s1, 31 // (op 583) 

s_xor_b32 s1, s19, s18 // (op 583) 

s_sub_i32 s19, s1, s18 // (op 583) 


s_mul_i32 s1, s47, s46 // (op 583) 
s_add_u32 s52, s6, 16 // (op 583) Start of Int64 add, locking SCC
s_addc_u32 s53, s7, 0 // (op 583) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 583) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 583) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 583) 

s_sub_i32 s18, s52, s1 // (op 583) 

s_mul_i32 s51, s19, s18 // (op 583) 


s_sub_u32 s18, s9, s51 // (op 583) 


s_add_u32 s9, s18, s1 // (op 583) 


s_cmp_eq_i32 s8, 0 // (op 583) 
s_mov_b32 s1, scc // (op 583) 

s_cmp_eq_i32 s1, 0 // (op 583) 
s_cselect_b32 s8, s9, s0 // (op 583) 



v_lshl_add_u32 v13, s8, 4, v12 // (op 583) 


v_mul_lo_u32 v17, v13, s5 // (op 583) most significant: low of low * high
v_mul_hi_u32 v12, v13, s4 // (op 583) most significant: high of low * low
v_mul_lo_u32 v16, v13, s4 // (op 583) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 583) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 583) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 583) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 583) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 583) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 583) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 583) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 583) 

v_mov_b32 v12, v14 // (op 583) convert

s_mul_i32 s1, s7, 6 // (op 585) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 585) least significant
s_mul_hi_u32 s8, s6, 6 // (op 585) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 585) most significant: sum

s_mul_i32 s9, s4, s1 // (op 585) most significant: low of low * high
s_mul_i32 s18, s5, s0 // (op 585) most significant: low of high * low
s_mul_i32 s8, s4, s0 // (op 585) least significant
s_mul_hi_u32 s19, s4, s0 // (op 585) most significant: high of low * low
s_add_u32 s9, s9, s18 // (op 585) most significant: sum
s_add_u32 s9, s9, s19 // (op 585) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 585) 

s_mov_b32 s52, s20 // (op 585) reg expression
s_mov_b32 s53, s21 // (op 585) reg expression
s_mov_b32 s54, s0 // (op 585) reg expression
s_mov_b32 s55, 131072 // (op 585) call()

s_mul_hi_i32 s51, s16, s11 // (op 602) 
v_lshrrev_b32 v13, 2, v1 // (op 602) 
s_add_i32 s0, s51, s16 // (op 602) 

v_and_b32 v14, -4, v1 // (op 602) 
s_ashr_i32 s1, s0, 31 // (op 602) 
v_sub_u32 v15, v1, v14 // (op 602) 

s_cmp_eq_i32 s11, 0 // (op 602) 
s_mov_b32 s51, scc // (op 602) 
v_lshlrev_b32 v14, 6, v15 // (op 602) 

s_cmp_eq_i32 s51, 0 // (op 602) 
s_cselect_b32 s8, 0, -1 // (op 602) 

v_mov_b32 v17, 0 // (op 602) convert
v_mov_b32 v16, v14 // (op 602) convert

s_and_b32 s9, s12, 31 // (op 602) 
s_lshl_b32 s51, 1, s9 // (op 602) 
s_add_i32 s18, s51, s8 // (op 602) 


s_and_b32 s8, s1, s18 // (op 602) 


s_add_i32 s1, s0, s8 // (op 602) 


s_ashr_i32 s0, s1, s9 // (op 602) 


s_lshl_b32 s1, s12, 24 // (op 602) 
s_ashr_i32 s51, s1, 31 // (op 602) 

s_xor_b32 s1, s0, s51 // (op 602) 

s_sub_i32 s0, s1, s51 // (op 602) 


s_mul_hi_i32 s1, s0, s13 // (op 602) 
s_add_i32 s51, s1, s0 // (op 602) 

s_ashr_i32 s1, s51, 31 // (op 602) 
s_cmp_eq_i32 s13, 0 // (op 602) 
s_mov_b32 s8, scc // (op 602) 
s_cmp_eq_i32 s8, 0 // (op 602) 
s_cselect_b32 s9, 0, -1 // (op 602) 

s_and_b32 s8, s14, 31 // (op 602) 
s_lshl_b32 s18, 1, s8 // (op 602) 
s_add_i32 s19, s18, s9 // (op 602) 


s_and_b32 s9, s1, s19 // (op 602) 


s_add_i32 s1, s51, s9 // (op 602) 


s_ashr_i32 s9, s1, s8 // (op 602) 


s_lshl_b32 s1, s14, 24 // (op 602) 
s_ashr_i32 s8, s1, 31 // (op 602) 

s_xor_b32 s1, s9, s8 // (op 602) 

s_sub_i32 s9, s1, s8 // (op 602) 


s_mul_i32 s1, s0, s46 // (op 602) 

s_sub_u32 s0, s16, s1 // (op 602) 

s_mul_i32 s1, s9, s46 // (op 602) 
s_add_u32 s1, s1, s0 // (op 602) 


s_mul_hi_i32 s0, s16, s10 // (op 602) 
s_add_i32 s51, s0, s16 // (op 602) 

s_ashr_i32 s0, s51, 31 // (op 602) 
s_cmp_eq_i32 s10, 0 // (op 602) 
s_mov_b32 s8, scc // (op 602) 
s_cmp_eq_i32 s8, 0 // (op 602) 
s_cselect_b32 s9, 0, -1 // (op 602) 

s_and_b32 s8, s49, 31 // (op 602) 
s_lshl_b32 s18, 1, s8 // (op 602) 
s_add_i32 s19, s18, s9 // (op 602) 


s_and_b32 s9, s0, s19 // (op 602) 


s_add_i32 s0, s51, s9 // (op 602) 


s_ashr_i32 s9, s0, s8 // (op 602) 


s_lshl_b32 s0, s49, 24 // (op 602) 
s_ashr_i32 s8, s0, 31 // (op 602) 

s_xor_b32 s0, s9, s8 // (op 602) 

s_sub_i32 s9, s0, s8 // (op 602) 


s_add_u32 s18, s2, 16 // (op 602) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 602) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 602) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 602) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 602) 

s_mul_i32 s0, s46, s18 // (op 602) 

s_mul_i32 s8, s50, s0 // (op 602) 

s_mul_i32 s0, s9, s8 // (op 602) 

s_sub_u32 s8, s16, s0 // (op 602) 

s_mov_b32 s0, s8 // (op 602) 
s_mul_hi_i32 s51, s0, s15 // (op 602) 
s_add_i32 s18, s51, s0 // (op 602) 


s_ashr_i32 s0, s18, 31 // (op 602) 
s_cmp_eq_i32 s15, 0 // (op 602) 
s_mov_b32 s19, scc // (op 602) 
s_cmp_eq_i32 s19, 0 // (op 602) 
s_cselect_b32 s51, 0, -1 // (op 602) 

s_and_b32 s19, s48, 31 // (op 602) 
s_lshl_b32 s56, 1, s19 // (op 602) 
s_add_i32 s57, s56, s51 // (op 602) 


s_and_b32 s51, s0, s57 // (op 602) 


s_add_i32 s0, s18, s51 // (op 602) 


s_ashr_i32 s18, s0, s19 // (op 602) 


s_lshl_b32 s0, s48, 24 // (op 602) 
s_ashr_i32 s19, s0, 31 // (op 602) 

s_xor_b32 s0, s18, s19 // (op 602) 

s_sub_i32 s18, s0, s19 // (op 602) 


s_mul_i32 s0, s47, s46 // (op 602) 
s_add_u32 s56, s6, 16 // (op 602) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 602) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 602) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 602) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 602) 

s_sub_i32 s19, s56, s0 // (op 602) 

s_mul_i32 s51, s18, s19 // (op 602) 


s_sub_u32 s18, s8, s51 // (op 602) 


s_add_u32 s8, s18, s0 // (op 602) 


s_cmp_eq_i32 s9, 0 // (op 602) 
s_mov_b32 s0, scc // (op 602) 

s_cmp_eq_i32 s0, 0 // (op 602) 
s_cselect_b32 s9, s8, s1 // (op 602) 



v_lshl_add_u32 v14, s9, 4, v13 // (op 602) 


v_mul_lo_u32 v19, v14, s5 // (op 602) most significant: low of low * high
v_mul_hi_u32 v13, v14, s4 // (op 602) most significant: high of low * low
v_mul_lo_u32 v18, v14, s4 // (op 602) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 602) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 602) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 602) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 602) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 602) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 602) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 602) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 602) 

v_mov_b32 v13, v14 // (op 602) convert

s_mul_i32 s1, s7, 6 // (op 604) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 604) least significant
s_mul_hi_u32 s51, s6, 6 // (op 604) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 604) most significant: sum

s_mul_i32 s9, s4, s1 // (op 604) most significant: low of low * high
s_mul_i32 s51, s5, s0 // (op 604) most significant: low of high * low
s_mul_i32 s8, s4, s0 // (op 604) least significant
s_mul_hi_u32 s18, s4, s0 // (op 604) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 604) most significant: sum
s_add_u32 s9, s9, s18 // (op 604) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 604) 

s_mov_b32 s52, s20 // (op 604) reg expression
s_mov_b32 s53, s21 // (op 604) reg expression
s_mov_b32 s54, s0 // (op 604) reg expression
s_mov_b32 s55, 131072 // (op 604) call()

s_mul_hi_i32 s51, s16, s11 // (op 647) 
s_add_i32 s0, s51, s16 // (op 647) 

s_ashr_i32 s1, s0, 31 // (op 647) 
s_cmp_eq_i32 s11, 0 // (op 647) 
s_mov_b32 s51, scc // (op 647) 
s_cmp_eq_i32 s51, 0 // (op 647) 
s_cselect_b32 s8, 0, -1 // (op 647) 

s_and_b32 s9, s12, 31 // (op 647) 
s_lshl_b32 s51, 1, s9 // (op 647) 
s_add_i32 s18, s51, s8 // (op 647) 


s_and_b32 s8, s1, s18 // (op 647) 


s_add_i32 s1, s0, s8 // (op 647) 


s_ashr_i32 s0, s1, s9 // (op 647) 


s_lshl_b32 s1, s12, 24 // (op 647) 
s_ashr_i32 s51, s1, 31 // (op 647) 

s_xor_b32 s1, s0, s51 // (op 647) 

s_sub_i32 s0, s1, s51 // (op 647) 


s_mul_hi_i32 s1, s0, s13 // (op 647) 
s_add_i32 s51, s1, s0 // (op 647) 

s_ashr_i32 s1, s51, 31 // (op 647) 
s_cmp_eq_i32 s13, 0 // (op 647) 
s_mov_b32 s8, scc // (op 647) 
s_cmp_eq_i32 s8, 0 // (op 647) 
s_cselect_b32 s9, 0, -1 // (op 647) 

s_and_b32 s8, s14, 31 // (op 647) 
s_lshl_b32 s18, 1, s8 // (op 647) 
s_add_i32 s19, s18, s9 // (op 647) 


s_and_b32 s9, s1, s19 // (op 647) 


s_add_i32 s1, s51, s9 // (op 647) 


s_ashr_i32 s9, s1, s8 // (op 647) 


s_lshl_b32 s1, s14, 24 // (op 647) 
s_ashr_i32 s8, s1, 31 // (op 647) 

s_xor_b32 s1, s9, s8 // (op 647) 

s_sub_i32 s9, s1, s8 // (op 647) 


s_add_u32 s18, s2, 16 // (op 647) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 647) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 647) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 647) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 647) 

s_mul_i32 s1, s9, s18 // (op 647) 

s_sub_i32 s51, s0, s1 // (op 647) 


s_mul_hi_i32 s0, s16, s10 // (op 647) 
s_add_i32 s1, s0, s16 // (op 647) 

s_ashr_i32 s0, s1, 31 // (op 647) 
s_cmp_eq_i32 s10, 0 // (op 647) 
s_mov_b32 s8, scc // (op 647) 
s_cmp_eq_i32 s8, 0 // (op 647) 
s_cselect_b32 s9, 0, -1 // (op 647) 

s_and_b32 s8, s49, 31 // (op 647) 
s_lshl_b32 s56, 1, s8 // (op 647) 
s_add_i32 s57, s56, s9 // (op 647) 


s_and_b32 s9, s0, s57 // (op 647) 


s_add_i32 s0, s1, s9 // (op 647) 


s_ashr_i32 s1, s0, s8 // (op 647) 


s_lshl_b32 s0, s49, 24 // (op 647) 
s_ashr_i32 s8, s0, 31 // (op 647) 

s_xor_b32 s0, s1, s8 // (op 647) 

s_sub_i32 s1, s0, s8 // (op 647) 


s_mul_i32 s0, s46, s18 // (op 647) 

s_mul_i32 s8, s50, s0 // (op 647) 

s_mul_i32 s0, s1, s8 // (op 647) 

s_sub_u32 s8, s16, s0 // (op 647) 

s_mul_hi_i32 s0, s8, s15 // (op 647) 
s_add_i32 s9, s0, s8 // (op 647) 


s_ashr_i32 s0, s9, 31 // (op 647) 
s_cmp_eq_i32 s15, 0 // (op 647) 
s_mov_b32 s8, scc // (op 647) 
s_cmp_eq_i32 s8, 0 // (op 647) 
s_cselect_b32 s18, 0, -1 // (op 647) 

s_and_b32 s8, s48, 31 // (op 647) 
s_lshl_b32 s19, 1, s8 // (op 647) 
s_add_i32 s56, s19, s18 // (op 647) 


s_and_b32 s18, s0, s56 // (op 647) 


s_add_i32 s0, s9, s18 // (op 647) 


s_ashr_i32 s9, s0, s8 // (op 647) 


s_lshl_b32 s0, s48, 24 // (op 647) 
s_ashr_i32 s8, s0, 31 // (op 647) 

s_xor_b32 s0, s9, s8 // (op 647) 

s_sub_i32 s9, s0, s8 // (op 647) 


s_cmp_eq_i32 s1, 0 // (op 647) 
s_mov_b32 s0, scc // (op 647) 

s_cmp_eq_i32 s0, 0 // (op 647) 
s_cselect_b32 s1, s9, s51 // (op 647) 



s_lshl_b32 s0, s1, 4 // (op 647) 

s_mov_b32 s9, 0 // (op 647) convert
s_mov_b32 s8, s0 // (op 647) convert

v_mul_lo_u32 v15, v1, s23 // (op 647) most significant: low of low * high
v_mul_hi_u32 v16, v1, s22 // (op 647) most significant: high of low * low
v_mul_lo_u32 v14, v1, s22 // (op 647) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 647) most significant: sum

v_mov_b32 v16, s8 // (op 647) 
v_mov_b32 v17, s9 // (op 647) 
v_add_co_u32 v14, s[0:1], v14, v16 // (op 647) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 647) most significant half




s_mul_i32 s1, s22, s27 // (op 649) most significant: low of low * high
s_mul_i32 s51, s23, s26 // (op 649) most significant: low of high * low
s_mul_i32 s0, s22, s26 // (op 649) least significant
s_mul_hi_u32 s8, s22, s26 // (op 649) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 649) most significant: sum
s_add_u32 s1, s1, s8 // (op 649) most significant: sum


s_mov_b32 s56, s24 // (op 649) reg expression
s_mov_b32 s57, s25 // (op 649) reg expression
s_mov_b32 s58, s0 // (op 649) reg expression
s_mov_b32 s59, 131072 // (op 649) call()

s_mul_hi_i32 s51, s16, s11 // (op 666) 
s_add_i32 s0, s51, s16 // (op 666) 

s_ashr_i32 s1, s0, 31 // (op 666) 
s_cmp_eq_i32 s11, 0 // (op 666) 
s_mov_b32 s51, scc // (op 666) 
s_cmp_eq_i32 s51, 0 // (op 666) 
s_cselect_b32 s8, 0, -1 // (op 666) 

s_and_b32 s9, s12, 31 // (op 666) 
s_lshl_b32 s51, 1, s9 // (op 666) 
s_add_i32 s18, s51, s8 // (op 666) 


s_and_b32 s8, s1, s18 // (op 666) 


s_add_i32 s1, s0, s8 // (op 666) 


s_ashr_i32 s0, s1, s9 // (op 666) 


s_lshl_b32 s1, s12, 24 // (op 666) 
s_ashr_i32 s51, s1, 31 // (op 666) 

s_xor_b32 s1, s0, s51 // (op 666) 

s_sub_i32 s0, s1, s51 // (op 666) 


s_mul_hi_i32 s1, s0, s13 // (op 666) 
s_add_i32 s51, s1, s0 // (op 666) 

s_ashr_i32 s1, s51, 31 // (op 666) 
s_cmp_eq_i32 s13, 0 // (op 666) 
s_mov_b32 s8, scc // (op 666) 
s_cmp_eq_i32 s8, 0 // (op 666) 
s_cselect_b32 s9, 0, -1 // (op 666) 

s_and_b32 s8, s14, 31 // (op 666) 
s_lshl_b32 s18, 1, s8 // (op 666) 
s_add_i32 s19, s18, s9 // (op 666) 


s_and_b32 s9, s1, s19 // (op 666) 


s_add_i32 s1, s51, s9 // (op 666) 


s_ashr_i32 s9, s1, s8 // (op 666) 


s_lshl_b32 s1, s14, 24 // (op 666) 
s_ashr_i32 s8, s1, 31 // (op 666) 

s_xor_b32 s1, s9, s8 // (op 666) 

s_sub_i32 s9, s1, s8 // (op 666) 


s_add_u32 s18, s2, 16 // (op 666) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 666) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 666) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 666) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 666) 

s_mul_i32 s1, s9, s18 // (op 666) 

s_sub_i32 s51, s0, s1 // (op 666) 


s_mul_hi_i32 s0, s16, s10 // (op 666) 
s_add_i32 s1, s0, s16 // (op 666) 

s_ashr_i32 s0, s1, 31 // (op 666) 
s_cmp_eq_i32 s10, 0 // (op 666) 
s_mov_b32 s8, scc // (op 666) 
s_cmp_eq_i32 s8, 0 // (op 666) 
s_cselect_b32 s9, 0, -1 // (op 666) 

s_and_b32 s8, s49, 31 // (op 666) 
s_lshl_b32 s60, 1, s8 // (op 666) 
s_add_i32 s61, s60, s9 // (op 666) 


s_and_b32 s9, s0, s61 // (op 666) 


s_add_i32 s0, s1, s9 // (op 666) 


s_ashr_i32 s1, s0, s8 // (op 666) 


s_lshl_b32 s0, s49, 24 // (op 666) 
s_ashr_i32 s8, s0, 31 // (op 666) 

s_xor_b32 s0, s1, s8 // (op 666) 

s_sub_i32 s1, s0, s8 // (op 666) 


s_mul_i32 s0, s46, s18 // (op 666) 

s_mul_i32 s8, s50, s0 // (op 666) 

s_mul_i32 s0, s1, s8 // (op 666) 

s_sub_u32 s8, s16, s0 // (op 666) 

s_mul_hi_i32 s0, s8, s15 // (op 666) 
s_add_i32 s9, s0, s8 // (op 666) 


s_ashr_i32 s0, s9, 31 // (op 666) 
s_cmp_eq_i32 s15, 0 // (op 666) 
s_mov_b32 s8, scc // (op 666) 
s_cmp_eq_i32 s8, 0 // (op 666) 
s_cselect_b32 s18, 0, -1 // (op 666) 

s_and_b32 s8, s48, 31 // (op 666) 
s_lshl_b32 s19, 1, s8 // (op 666) 
s_add_i32 s60, s19, s18 // (op 666) 


s_and_b32 s18, s0, s60 // (op 666) 


s_add_i32 s0, s9, s18 // (op 666) 


s_ashr_i32 s9, s0, s8 // (op 666) 


s_lshl_b32 s0, s48, 24 // (op 666) 
s_ashr_i32 s8, s0, 31 // (op 666) 

s_xor_b32 s0, s9, s8 // (op 666) 

s_sub_i32 s9, s0, s8 // (op 666) 


s_cmp_eq_i32 s1, 0 // (op 666) 
s_mov_b32 s0, scc // (op 666) 

s_cmp_eq_i32 s0, 0 // (op 666) 
s_cselect_b32 s1, s9, s51 // (op 666) 



s_lshl_b32 s0, s1, 4 // (op 666) 

s_mov_b32 s9, 0 // (op 666) convert
s_mov_b32 s8, s0 // (op 666) convert

v_mul_lo_u32 v17, v1, s23 // (op 666) most significant: low of low * high
v_mul_hi_u32 v18, v1, s22 // (op 666) most significant: high of low * low
v_mul_lo_u32 v16, v1, s22 // (op 666) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 666) most significant: sum

v_mov_b32 v18, s8 // (op 666) 
v_mov_b32 v19, s9 // (op 666) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 666) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 666) most significant half




s_mul_i32 s1, s22, s27 // (op 668) most significant: low of low * high
s_mul_i32 s51, s23, s26 // (op 668) most significant: low of high * low
s_mul_i32 s0, s22, s26 // (op 668) least significant
s_mul_hi_u32 s8, s22, s26 // (op 668) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 668) most significant: sum
s_add_u32 s1, s1, s8 // (op 668) most significant: sum


s_mov_b32 s56, s24 // (op 668) reg expression
s_mov_b32 s57, s25 // (op 668) reg expression
s_mov_b32 s58, s0 // (op 668) reg expression
s_mov_b32 s59, 131072 // (op 668) call()

s_mul_hi_i32 s51, s16, s11 // (op 711) 
v_lshrrev_b32 v18, 3, v1 // (op 711) 
s_add_i32 s0, s51, s16 // (op 711) 

v_lshlrev_b32 v19, 1, v18 // (op 711) 

s_ashr_i32 s1, s0, 31 // (op 711) 
v_and_b32 v18, -8, v1 // (op 711) 
s_cmp_eq_i32 s11, 0 // (op 711) 
s_mov_b32 s51, scc // (op 711) 
v_sub_u32 v20, v1, v18 // (op 711) 

s_cmp_eq_i32 s51, 0 // (op 711) 
s_cselect_b32 s8, 0, -1 // (op 711) 

s_and_b32 s9, s12, 31 // (op 711) 
s_lshl_b32 s51, 1, s9 // (op 711) 
s_add_i32 s18, s51, s8 // (op 711) 


s_and_b32 s8, s1, s18 // (op 711) 


s_add_i32 s1, s0, s8 // (op 711) 


s_ashr_i32 s0, s1, s9 // (op 711) 


s_lshl_b32 s1, s12, 24 // (op 711) 
s_ashr_i32 s51, s1, 31 // (op 711) 

s_xor_b32 s1, s0, s51 // (op 711) 

s_sub_i32 s0, s1, s51 // (op 711) 


s_mul_hi_i32 s1, s0, s13 // (op 711) 
s_add_i32 s51, s1, s0 // (op 711) 

s_ashr_i32 s1, s51, 31 // (op 711) 
s_cmp_eq_i32 s13, 0 // (op 711) 
s_mov_b32 s8, scc // (op 711) 
s_cmp_eq_i32 s8, 0 // (op 711) 
s_cselect_b32 s9, 0, -1 // (op 711) 

s_and_b32 s8, s14, 31 // (op 711) 
s_lshl_b32 s18, 1, s8 // (op 711) 
s_add_i32 s19, s18, s9 // (op 711) 


s_and_b32 s9, s1, s19 // (op 711) 


s_add_i32 s1, s51, s9 // (op 711) 


s_ashr_i32 s9, s1, s8 // (op 711) 


s_lshl_b32 s1, s14, 24 // (op 711) 
s_ashr_i32 s8, s1, 31 // (op 711) 

s_xor_b32 s1, s9, s8 // (op 711) 

s_sub_i32 s9, s1, s8 // (op 711) 


s_mul_i32 s1, s0, s46 // (op 711) 

s_sub_u32 s0, s16, s1 // (op 711) 

s_mul_i32 s1, s9, s46 // (op 711) 
s_add_u32 s1, s1, s0 // (op 711) 


s_mul_hi_i32 s0, s16, s10 // (op 711) 
s_add_i32 s51, s0, s16 // (op 711) 

s_ashr_i32 s0, s51, 31 // (op 711) 
s_cmp_eq_i32 s10, 0 // (op 711) 
s_mov_b32 s8, scc // (op 711) 
s_cmp_eq_i32 s8, 0 // (op 711) 
s_cselect_b32 s9, 0, -1 // (op 711) 

s_and_b32 s8, s49, 31 // (op 711) 
s_lshl_b32 s18, 1, s8 // (op 711) 
s_add_i32 s19, s18, s9 // (op 711) 


s_and_b32 s9, s0, s19 // (op 711) 


s_add_i32 s0, s51, s9 // (op 711) 


s_ashr_i32 s9, s0, s8 // (op 711) 


s_lshl_b32 s0, s49, 24 // (op 711) 
s_ashr_i32 s8, s0, 31 // (op 711) 

s_xor_b32 s0, s9, s8 // (op 711) 

s_sub_i32 s9, s0, s8 // (op 711) 


s_add_u32 s18, s2, 16 // (op 711) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 711) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 711) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 711) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 711) 

s_mul_i32 s0, s46, s18 // (op 711) 

s_mul_i32 s8, s50, s0 // (op 711) 

s_mul_i32 s0, s9, s8 // (op 711) 

s_sub_u32 s8, s16, s0 // (op 711) 

s_mov_b32 s0, s8 // (op 711) 
s_mul_hi_i32 s51, s0, s15 // (op 711) 
s_add_i32 s18, s51, s0 // (op 711) 


s_ashr_i32 s0, s18, 31 // (op 711) 
s_cmp_eq_i32 s15, 0 // (op 711) 
s_mov_b32 s19, scc // (op 711) 
s_cmp_eq_i32 s19, 0 // (op 711) 
s_cselect_b32 s51, 0, -1 // (op 711) 

s_and_b32 s19, s48, 31 // (op 711) 
s_lshl_b32 s60, 1, s19 // (op 711) 
s_add_i32 s61, s60, s51 // (op 711) 


s_and_b32 s51, s0, s61 // (op 711) 


s_add_i32 s0, s18, s51 // (op 711) 


s_ashr_i32 s18, s0, s19 // (op 711) 


s_lshl_b32 s0, s48, 24 // (op 711) 
s_ashr_i32 s19, s0, 31 // (op 711) 

s_xor_b32 s0, s18, s19 // (op 711) 

s_sub_i32 s18, s0, s19 // (op 711) 


s_mul_i32 s0, s47, s46 // (op 711) 
s_add_u32 s60, s6, 16 // (op 711) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 711) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 711) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 711) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 711) 

s_sub_i32 s19, s60, s0 // (op 711) 

s_mul_i32 s51, s18, s19 // (op 711) 


s_sub_u32 s18, s8, s51 // (op 711) 


s_add_u32 s8, s18, s0 // (op 711) 


s_cmp_eq_i32 s9, 0 // (op 711) 
s_mov_b32 s0, scc // (op 711) 

s_cmp_eq_i32 s0, 0 // (op 711) 
s_cselect_b32 s9, s8, s1 // (op 711) 



v_lshl_add_u32 v18, s9, 4, v19 // (op 711) 


v_mul_lo_u32 v19, v18, s28 // (op 711) 
v_add_u32 v19, v19, v20 // (op 711) 


s_mul_i32 s1, s28, s7 // (op 713) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 713) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 713) least significant
s_mul_hi_u32 s8, s28, s6 // (op 713) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 713) most significant: sum
s_add_u32 s1, s1, s8 // (op 713) most significant: sum


s_mov_b32 s60, s30 // (op 713) reg expression
s_mov_b32 s61, s31 // (op 713) reg expression
s_mov_b32 s62, s0 // (op 713) reg expression
s_mov_b32 s63, 131072 // (op 713) call()

s_mul_hi_i32 s51, s16, s11 // (op 730) 
v_lshrrev_b32 v18, 3, v1 // (op 730) 
s_add_i32 s0, s51, s16 // (op 730) 

v_lshlrev_b32 v20, 1, v18 // (op 730) 

s_ashr_i32 s1, s0, 31 // (op 730) 
v_and_b32 v18, -8, v1 // (op 730) 
s_cmp_eq_i32 s11, 0 // (op 730) 
s_mov_b32 s51, scc // (op 730) 
v_sub_u32 v21, v1, v18 // (op 730) 

s_cmp_eq_i32 s51, 0 // (op 730) 
s_cselect_b32 s8, 0, -1 // (op 730) 

s_and_b32 s9, s12, 31 // (op 730) 
s_lshl_b32 s51, 1, s9 // (op 730) 
s_add_i32 s18, s51, s8 // (op 730) 


s_and_b32 s8, s1, s18 // (op 730) 


s_add_i32 s1, s0, s8 // (op 730) 


s_ashr_i32 s0, s1, s9 // (op 730) 


s_lshl_b32 s1, s12, 24 // (op 730) 
s_ashr_i32 s51, s1, 31 // (op 730) 

s_xor_b32 s1, s0, s51 // (op 730) 

s_sub_i32 s0, s1, s51 // (op 730) 


s_mul_hi_i32 s1, s0, s13 // (op 730) 
s_add_i32 s51, s1, s0 // (op 730) 

s_ashr_i32 s1, s51, 31 // (op 730) 
s_cmp_eq_i32 s13, 0 // (op 730) 
s_mov_b32 s8, scc // (op 730) 
s_cmp_eq_i32 s8, 0 // (op 730) 
s_cselect_b32 s9, 0, -1 // (op 730) 

s_and_b32 s8, s14, 31 // (op 730) 
s_lshl_b32 s18, 1, s8 // (op 730) 
s_add_i32 s19, s18, s9 // (op 730) 


s_and_b32 s9, s1, s19 // (op 730) 


s_add_i32 s1, s51, s9 // (op 730) 


s_ashr_i32 s9, s1, s8 // (op 730) 


s_lshl_b32 s1, s14, 24 // (op 730) 
s_ashr_i32 s8, s1, 31 // (op 730) 

s_xor_b32 s1, s9, s8 // (op 730) 

s_sub_i32 s9, s1, s8 // (op 730) 


s_mul_i32 s1, s0, s46 // (op 730) 

s_sub_u32 s0, s16, s1 // (op 730) 

s_mul_i32 s1, s9, s46 // (op 730) 
s_add_u32 s1, s1, s0 // (op 730) 


s_mul_hi_i32 s0, s16, s10 // (op 730) 
s_add_i32 s51, s0, s16 // (op 730) 

s_ashr_i32 s0, s51, 31 // (op 730) 
s_cmp_eq_i32 s10, 0 // (op 730) 
s_mov_b32 s8, scc // (op 730) 
s_cmp_eq_i32 s8, 0 // (op 730) 
s_cselect_b32 s9, 0, -1 // (op 730) 

s_and_b32 s8, s49, 31 // (op 730) 
s_lshl_b32 s18, 1, s8 // (op 730) 
s_add_i32 s19, s18, s9 // (op 730) 


s_and_b32 s9, s0, s19 // (op 730) 


s_add_i32 s0, s51, s9 // (op 730) 


s_ashr_i32 s9, s0, s8 // (op 730) 


s_lshl_b32 s0, s49, 24 // (op 730) 
s_ashr_i32 s8, s0, 31 // (op 730) 

s_xor_b32 s0, s9, s8 // (op 730) 

s_sub_i32 s9, s0, s8 // (op 730) 


s_add_u32 s18, s2, 16 // (op 730) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 730) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 730) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 730) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 730) 

s_mul_i32 s0, s46, s18 // (op 730) 

s_mul_i32 s8, s50, s0 // (op 730) 

s_mul_i32 s0, s9, s8 // (op 730) 

s_sub_u32 s8, s16, s0 // (op 730) 

s_mov_b32 s0, s8 // (op 730) 
s_mul_hi_i32 s51, s0, s15 // (op 730) 
s_add_i32 s18, s51, s0 // (op 730) 


s_ashr_i32 s0, s18, 31 // (op 730) 
s_cmp_eq_i32 s15, 0 // (op 730) 
s_mov_b32 s19, scc // (op 730) 
s_cmp_eq_i32 s19, 0 // (op 730) 
s_cselect_b32 s51, 0, -1 // (op 730) 

s_and_b32 s19, s48, 31 // (op 730) 
s_lshl_b32 s64, 1, s19 // (op 730) 
s_add_i32 s65, s64, s51 // (op 730) 


s_and_b32 s51, s0, s65 // (op 730) 


s_add_i32 s0, s18, s51 // (op 730) 


s_ashr_i32 s18, s0, s19 // (op 730) 


s_lshl_b32 s0, s48, 24 // (op 730) 
s_ashr_i32 s19, s0, 31 // (op 730) 

s_xor_b32 s0, s18, s19 // (op 730) 

s_sub_i32 s18, s0, s19 // (op 730) 


s_mul_i32 s0, s47, s46 // (op 730) 
s_add_u32 s64, s6, 16 // (op 730) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 730) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 730) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 730) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 730) 

s_sub_i32 s19, s64, s0 // (op 730) 

s_mul_i32 s51, s18, s19 // (op 730) 


s_sub_u32 s18, s8, s51 // (op 730) 


s_add_u32 s8, s18, s0 // (op 730) 


s_cmp_eq_i32 s9, 0 // (op 730) 
s_mov_b32 s0, scc // (op 730) 

s_cmp_eq_i32 s0, 0 // (op 730) 
s_cselect_b32 s9, s8, s1 // (op 730) 



v_lshl_add_u32 v18, s9, 4, v20 // (op 730) 


v_mul_lo_u32 v20, v18, s28 // (op 730) 
v_add_u32 v20, v20, v21 // (op 730) 


s_mul_i32 s1, s28, s7 // (op 732) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 732) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 732) least significant
s_mul_hi_u32 s8, s28, s6 // (op 732) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 732) most significant: sum
s_add_u32 s1, s1, s8 // (op 732) most significant: sum


s_mov_b32 s60, s30 // (op 732) reg expression
s_mov_b32 s61, s31 // (op 732) reg expression
s_mov_b32 s62, s0 // (op 732) reg expression
s_mov_b32 s63, 131072 // (op 732) call()

s_mul_hi_i32 s51, s16, s11 // (op 775) 
v_and_b32 v18, -16, v1 // (op 775) 
s_add_i32 s0, s51, s16 // (op 775) 

v_sub_u32 v21, v1, v18 // (op 775) 

s_ashr_i32 s1, s0, 31 // (op 775) 
v_lshrrev_b32 v18, 4, v1 // (op 775) 
s_cmp_eq_i32 s11, 0 // (op 775) 
s_mov_b32 s51, scc // (op 775) 
v_lshlrev_b32 v22, 1, v18 // (op 775) 

s_cmp_eq_i32 s51, 0 // (op 775) 
s_cselect_b32 s8, 0, -1 // (op 775) 

s_and_b32 s9, s12, 31 // (op 775) 
s_lshl_b32 s51, 1, s9 // (op 775) 
s_add_i32 s18, s51, s8 // (op 775) 


s_and_b32 s8, s1, s18 // (op 775) 


s_add_i32 s1, s0, s8 // (op 775) 


s_ashr_i32 s0, s1, s9 // (op 775) 


s_lshl_b32 s1, s12, 24 // (op 775) 
s_ashr_i32 s51, s1, 31 // (op 775) 

s_xor_b32 s1, s0, s51 // (op 775) 

s_sub_i32 s0, s1, s51 // (op 775) 


s_mul_hi_i32 s1, s0, s13 // (op 775) 
s_add_i32 s51, s1, s0 // (op 775) 

s_ashr_i32 s1, s51, 31 // (op 775) 
s_cmp_eq_i32 s13, 0 // (op 775) 
s_mov_b32 s8, scc // (op 775) 
s_cmp_eq_i32 s8, 0 // (op 775) 
s_cselect_b32 s9, 0, -1 // (op 775) 

s_and_b32 s8, s14, 31 // (op 775) 
s_lshl_b32 s18, 1, s8 // (op 775) 
s_add_i32 s19, s18, s9 // (op 775) 


s_and_b32 s9, s1, s19 // (op 775) 


s_add_i32 s1, s51, s9 // (op 775) 


s_ashr_i32 s9, s1, s8 // (op 775) 


s_lshl_b32 s1, s14, 24 // (op 775) 
s_ashr_i32 s8, s1, 31 // (op 775) 

s_xor_b32 s1, s9, s8 // (op 775) 

s_sub_i32 s9, s1, s8 // (op 775) 


s_add_u32 s18, s2, 16 // (op 775) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 775) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 775) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 775) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 775) 

s_mul_i32 s1, s9, s18 // (op 775) 

s_sub_i32 s51, s0, s1 // (op 775) 


s_mul_hi_i32 s0, s16, s10 // (op 775) 
s_add_i32 s1, s0, s16 // (op 775) 

s_ashr_i32 s0, s1, 31 // (op 775) 
s_cmp_eq_i32 s10, 0 // (op 775) 
s_mov_b32 s8, scc // (op 775) 
s_cmp_eq_i32 s8, 0 // (op 775) 
s_cselect_b32 s9, 0, -1 // (op 775) 

s_and_b32 s8, s49, 31 // (op 775) 
s_lshl_b32 s64, 1, s8 // (op 775) 
s_add_i32 s65, s64, s9 // (op 775) 


s_and_b32 s9, s0, s65 // (op 775) 


s_add_i32 s0, s1, s9 // (op 775) 


s_ashr_i32 s1, s0, s8 // (op 775) 


s_lshl_b32 s0, s49, 24 // (op 775) 
s_ashr_i32 s8, s0, 31 // (op 775) 

s_xor_b32 s0, s1, s8 // (op 775) 

s_sub_i32 s1, s0, s8 // (op 775) 


s_mul_i32 s0, s46, s18 // (op 775) 

s_mul_i32 s8, s50, s0 // (op 775) 

s_mul_i32 s0, s1, s8 // (op 775) 

s_sub_u32 s8, s16, s0 // (op 775) 

s_mul_hi_i32 s0, s8, s15 // (op 775) 
s_add_i32 s9, s0, s8 // (op 775) 


s_ashr_i32 s0, s9, 31 // (op 775) 
s_cmp_eq_i32 s15, 0 // (op 775) 
s_mov_b32 s8, scc // (op 775) 
s_cmp_eq_i32 s8, 0 // (op 775) 
s_cselect_b32 s18, 0, -1 // (op 775) 

s_and_b32 s8, s48, 31 // (op 775) 
s_lshl_b32 s19, 1, s8 // (op 775) 
s_add_i32 s64, s19, s18 // (op 775) 


s_and_b32 s18, s0, s64 // (op 775) 


s_add_i32 s0, s9, s18 // (op 775) 


s_ashr_i32 s9, s0, s8 // (op 775) 


s_lshl_b32 s0, s48, 24 // (op 775) 
s_ashr_i32 s8, s0, 31 // (op 775) 

s_xor_b32 s0, s9, s8 // (op 775) 

s_sub_i32 s9, s0, s8 // (op 775) 


s_cmp_eq_i32 s1, 0 // (op 775) 
s_mov_b32 s0, scc // (op 775) 

s_cmp_eq_i32 s0, 0 // (op 775) 
s_cselect_b32 s1, s9, s51 // (op 775) 



v_lshl_add_u32 v18, s1, 4, v21 // (op 775) 


v_mul_lo_u32 v21, v22, s32 // (op 775) 
v_add_u32 v21, v21, v18 // (op 775) 


s_mul_i32 s1, s32, s37 // (op 777) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 777) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 777) least significant
s_mul_hi_u32 s8, s32, s36 // (op 777) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 777) most significant: sum
s_add_u32 s1, s1, s8 // (op 777) most significant: sum


s_mov_b32 s64, s34 // (op 777) reg expression
s_mov_b32 s65, s35 // (op 777) reg expression
s_mov_b32 s66, s0 // (op 777) reg expression
s_mov_b32 s67, 131072 // (op 777) call()

s_mul_hi_i32 s51, s16, s11 // (op 794) 
v_and_b32 v18, -16, v1 // (op 794) 
s_add_i32 s0, s51, s16 // (op 794) 

v_sub_u32 v22, v1, v18 // (op 794) 

s_ashr_i32 s1, s0, 31 // (op 794) 
v_lshrrev_b32 v18, 4, v1 // (op 794) 
s_cmp_eq_i32 s11, 0 // (op 794) 
s_mov_b32 s51, scc // (op 794) 
v_lshlrev_b32 v23, 1, v18 // (op 794) 

s_cmp_eq_i32 s51, 0 // (op 794) 
s_cselect_b32 s8, 0, -1 // (op 794) 

s_and_b32 s9, s12, 31 // (op 794) 
s_lshl_b32 s51, 1, s9 // (op 794) 
s_add_i32 s18, s51, s8 // (op 794) 


s_and_b32 s8, s1, s18 // (op 794) 


s_add_i32 s1, s0, s8 // (op 794) 


s_ashr_i32 s0, s1, s9 // (op 794) 


s_lshl_b32 s1, s12, 24 // (op 794) 
s_ashr_i32 s51, s1, 31 // (op 794) 

s_xor_b32 s1, s0, s51 // (op 794) 

s_sub_i32 s0, s1, s51 // (op 794) 


s_mul_hi_i32 s1, s0, s13 // (op 794) 
s_add_i32 s51, s1, s0 // (op 794) 

s_ashr_i32 s1, s51, 31 // (op 794) 
s_cmp_eq_i32 s13, 0 // (op 794) 
s_mov_b32 s8, scc // (op 794) 
s_cmp_eq_i32 s8, 0 // (op 794) 
s_cselect_b32 s9, 0, -1 // (op 794) 

s_and_b32 s8, s14, 31 // (op 794) 
s_lshl_b32 s18, 1, s8 // (op 794) 
s_add_i32 s19, s18, s9 // (op 794) 


s_and_b32 s9, s1, s19 // (op 794) 


s_add_i32 s1, s51, s9 // (op 794) 


s_ashr_i32 s9, s1, s8 // (op 794) 


s_lshl_b32 s1, s14, 24 // (op 794) 
s_ashr_i32 s8, s1, 31 // (op 794) 

s_xor_b32 s1, s9, s8 // (op 794) 

s_sub_i32 s9, s1, s8 // (op 794) 


s_add_u32 s18, s2, 16 // (op 794) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 794) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 794) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 794) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 794) 

s_mul_i32 s1, s9, s18 // (op 794) 

s_sub_i32 s51, s0, s1 // (op 794) 


s_mul_hi_i32 s0, s16, s10 // (op 794) 
s_add_i32 s1, s0, s16 // (op 794) 

s_ashr_i32 s0, s1, 31 // (op 794) 
s_cmp_eq_i32 s10, 0 // (op 794) 
s_mov_b32 s8, scc // (op 794) 
s_cmp_eq_i32 s8, 0 // (op 794) 
s_cselect_b32 s9, 0, -1 // (op 794) 

s_and_b32 s8, s49, 31 // (op 794) 
s_lshl_b32 s68, 1, s8 // (op 794) 
s_add_i32 s69, s68, s9 // (op 794) 


s_and_b32 s9, s0, s69 // (op 794) 


s_add_i32 s0, s1, s9 // (op 794) 


s_ashr_i32 s1, s0, s8 // (op 794) 


s_lshl_b32 s0, s49, 24 // (op 794) 
s_ashr_i32 s8, s0, 31 // (op 794) 

s_xor_b32 s0, s1, s8 // (op 794) 

s_sub_i32 s1, s0, s8 // (op 794) 


s_mul_i32 s0, s46, s18 // (op 794) 

s_mul_i32 s8, s50, s0 // (op 794) 

s_mul_i32 s0, s1, s8 // (op 794) 

s_sub_u32 s8, s16, s0 // (op 794) 

s_mul_hi_i32 s0, s8, s15 // (op 794) 
s_add_i32 s9, s0, s8 // (op 794) 


s_ashr_i32 s0, s9, 31 // (op 794) 
s_cmp_eq_i32 s15, 0 // (op 794) 
s_mov_b32 s8, scc // (op 794) 
s_cmp_eq_i32 s8, 0 // (op 794) 
s_cselect_b32 s18, 0, -1 // (op 794) 

s_and_b32 s8, s48, 31 // (op 794) 
s_lshl_b32 s19, 1, s8 // (op 794) 
s_add_i32 s68, s19, s18 // (op 794) 


s_and_b32 s18, s0, s68 // (op 794) 


s_add_i32 s0, s9, s18 // (op 794) 


s_ashr_i32 s9, s0, s8 // (op 794) 


s_lshl_b32 s0, s48, 24 // (op 794) 
s_ashr_i32 s8, s0, 31 // (op 794) 

s_xor_b32 s0, s9, s8 // (op 794) 

s_sub_i32 s9, s0, s8 // (op 794) 


s_cmp_eq_i32 s1, 0 // (op 794) 
s_mov_b32 s0, scc // (op 794) 

s_cmp_eq_i32 s0, 0 // (op 794) 
s_cselect_b32 s1, s9, s51 // (op 794) 



v_lshl_add_u32 v18, s1, 4, v22 // (op 794) 


v_mul_lo_u32 v22, v23, s32 // (op 794) 
v_add_u32 v22, v22, v18 // (op 794) 


s_mul_i32 s1, s32, s37 // (op 796) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 796) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 796) least significant
s_mul_hi_u32 s8, s32, s36 // (op 796) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 796) most significant: sum
s_add_u32 s1, s1, s8 // (op 796) most significant: sum


s_mov_b32 s64, s34 // (op 796) reg expression
s_mov_b32 s65, s35 // (op 796) reg expression
s_mov_b32 s66, s0 // (op 796) reg expression
s_mov_b32 s67, 131072 // (op 796) call()

v_lshlrev_b32 v18, 4, v1 // (op 913) 
s_nop 0
v_readfirstlane_b32 s51, v18 // (op 913) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[22:23], 6 // (op 539) 
v_mov_b64 v[24:25], s[0:1] // (op 539) convert


v_mov_b64 v[26:27], v[14:15] // (op 539) reg expression
s_mov_b32 m0, s51 // (op 539) reg expression
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 539) Load value direct to lds
v_add_co_u32 v26, s[0:1], v26, v24 // (op 539) least significant half
v_addc_co_u32 v27, s[0:1], 0, v27, s[0:1] // (op 539) most significant half

s_add_u32 m0, m0, 1024 // (op 539) 
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 539) Load value direct to lds
v_add_co_u32 v26, s[0:1], v26, v24 // (op 539) least significant half
v_addc_co_u32 v27, s[0:1], 0, v27, s[0:1] // (op 539) most significant half

s_add_u32 m0, m0, 1024 // (op 539) 
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 539) Load value direct to lds
v_add_co_u32 v26, s[0:1], v26, v24 // (op 539) least significant half
v_addc_co_u32 v27, s[0:1], 0, v27, s[0:1] // (op 539) most significant half

s_add_u32 m0, m0, 1024 // (op 539) 
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 539) Load value direct to lds




v_mov_b64 v[24:25], s[32:33] // (op 427) convert

v_mov_b32 v18, v21 // (op 427) reg expression
buffer_load_ubyte v26, v18, s[64:67], 0 offen offset: 0 // (op 427) Load value
v_add_u32 v18, v18, v24 // (op 427) 
buffer_load_ubyte v27, v18, s[64:67], 0 offen offset: 0 // (op 427) Load value





buffer_load_dwordx4 v[28:31], v12, s[52:55], 0 offen // (op 430) Load value
buffer_load_dwordx2 v[32:33], v12, s[52:55], 0 offen offset:16 // (op 430) Load value
buffer_load_dwordx4 v[34:37], v12, s[52:55], 0 offen offset:24 // (op 430) Load value
buffer_load_dwordx2 v[38:39], v12, s[52:55], 0 offen offset:40 // (op 430) Load value

v_mov_b64 v[24:25], s[28:29] // (op 433) convert

v_mov_b32 v18, v19 // (op 433) 
buffer_load_ubyte v40, v18, s[60:63], 0 offen offset: 0 // (op 433) Load value
v_add_u32 v18, v18, v24 // (op 433) 
buffer_load_ubyte v41, v18, s[60:63], 0 offen offset: 0 // (op 433) Load value


v_lshlrev_b32 v18, 4, v1 // (op 1116) 
s_nop 0
v_readfirstlane_b32 s51, v18 // (op 1116) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[22:23], 6 // (op 544) 
v_mov_b64 v[24:25], s[0:1] // (op 544) convert


s_lshl_b64 s[0:1], s[22:23], 8 // (op 544) 
v_mov_b32 v18, s0 // (op 544) 
v_mov_b32 v23, s1 // (op 544) 
v_add_co_u32 v42, s[8:9], v14, v18 // (op 544) least significant half
v_addc_co_u32 v43, s[8:9], v15, v23, s[8:9] // (op 544) most significant half




s_add_u32 s0, 4096, s51 // (op 544) 
s_mov_b32 m0, s0 // (op 544) reg expression
buffer_load_dwordx4 v42, s[56:59], 0 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v42, s[8:9], v42, v24 // (op 544) least significant half
v_addc_co_u32 v43, s[8:9], 0, v43, s[8:9] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v42, s[56:59], 0 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v42, s[8:9], v42, v24 // (op 544) least significant half
v_addc_co_u32 v43, s[8:9], 0, v43, s[8:9] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v42, s[56:59], 0 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v42, s[8:9], v42, v24 // (op 544) least significant half
v_addc_co_u32 v43, s[8:9], 0, v43, s[8:9] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v42, s[56:59], 0 offen offset: 0 lds // (op 544) Load value direct to lds






v_mov_b64 v[14:15], s[32:33] // (op 439) convert

s_lshl_b32 s51, s32, 3 // (op 439) 
v_add_u32 v18, v21, s51 // (op 439) 

buffer_load_ubyte v24, v18, s[64:67], 0 offen offset: 0 // (op 439) Load value
v_add_u32 v18, v18, v14 // (op 439) 
buffer_load_ubyte v25, v18, s[64:67], 0 offen offset: 0 // (op 439) Load value






buffer_load_dwordx4 v[42:45], v12, s[52:55], 0 offen offset:192 // (op 442) Load value
buffer_load_dwordx2 v[46:47], v12, s[52:55], 0 offen offset:208 // (op 442) Load value
buffer_load_dwordx4 v[48:51], v12, s[52:55], 0 offen offset:216 // (op 442) Load value
buffer_load_dwordx2 v[52:53], v12, s[52:55], 0 offen offset:232 // (op 442) Load value


v_mov_b64 v[14:15], s[28:29] // (op 445) convert

v_mov_b32 v12, v19 // (op 445) 
buffer_load_ubyte v54, v12, s[60:63], 0 offen offset:8 // (op 445) Load value
v_add_u32 v12, v12, v14 // (op 445) 
buffer_load_ubyte v55, v12, s[60:63], 0 offen offset:8 // (op 445) Load value



v_and_b32 v12, -16, v1 // (op 1019) 
v_sub_u32 v21, v1, v12 // (op 1019) 

v_lshrrev_b32 v12, 4, v1 // (op 1019) 
v_lshl_add_u32 v23, v12, 5, v21 // (op 1019) 





s_waitcnt vmcnt(19)
ds_write_b8 v23, v26 offset:8192 // (op 448) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v23, v27 offset:8208 // (op 448) Store local data 

v_and_b32 v12, -4, v1 // (op 861) 
v_sub_u32 v21, v1, v12 // (op 861) 

v_lshlrev_b32 v12, 6, v21 // (op 861) 

v_lshrrev_b32 v21, 2, v1 // (op 861) 
v_lshl_add_u32 v23, v21, 8, v12 // (op 861) 


v_mul_lo_u32 v12, v23, 6 // (op 861) 

v_lshrrev_b32 v21, 3, v12 // (op 861) 




s_waitcnt vmcnt(17)
ds_write_b128 v21, v[28:31] offset:8320 // (op 449) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v21, v[32:33] offset:8336 // (op 449) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v21, v[34:37] offset:8344 // (op 449) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v21, v[38:39] offset:8360 // (op 449) Store local data 

v_and_b32 v12, -8, v1 // (op 966) 
v_sub_u32 v21, v1, v12 // (op 966) 

v_lshrrev_b32 v12, 3, v1 // (op 966) 
v_lshl_add_u32 v23, v12, 4, v21 // (op 966) 





s_waitcnt vmcnt(13)
ds_write_b8 v23, v40 offset:11392 // (op 450) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v23, v41 offset:11400 // (op 450) Store local data 

s_barrier  // (op 1232) 



ds_read_b64_tr_b8 v[56:57], v0 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:128 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:1024 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:1152 // (op 467) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1268) 


ds_read_u8 v12, v7 offset:8192 // (op 473) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1213) 



ds_read_b128 v[64:67], v3 offset:8320 // (op 464) Load local data 
ds_read_b64 v[68:69], v3 offset:8336 // (op 464) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1248) 



ds_read_u8 v21, v4 offset:11392 // (op 470) Load local data 
v_and_b32 v23, -4, v1 // (op 832) 
v_sub_u32 v14, v1, v23 // (op 832) 

v_lshlrev_b32 v15, 6, v14 // (op 832) 

v_lshrrev_b32 v14, 2, v1 // (op 832) 
v_lshl_add_u32 v23, v14, 8, v15 // (op 832) 


v_mul_lo_u32 v14, v23, 6 // (op 832) 

v_lshrrev_b32 v15, 3, v14 // (op 832) 

v_lshlrev_b32 v14, 4, v1 // (op 885) 
s_nop 0
v_readfirstlane_b32 s51, v14 // (op 885) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 937) 
v_sub_u32 v23, v1, v14 // (op 937) 

v_lshrrev_b32 v14, 3, v1 // (op 937) 
v_lshl_add_u32 v18, v14, 4, v23 // (op 937) 


v_and_b32 v14, -16, v1 // (op 990) 
v_sub_u32 v19, v1, v14 // (op 990) 

v_lshrrev_b32 v14, 4, v1 // (op 990) 
v_lshl_add_u32 v23, v14, 5, v19 // (op 990) 


v_and_b32 v14, -4, v1 // (op 1077) 
v_sub_u32 v19, v1, v14 // (op 1077) 

v_lshlrev_b32 v14, 6, v19 // (op 1077) 

v_lshrrev_b32 v19, 2, v1 // (op 1077) 
v_lshl_add_u32 v70, v19, 8, v14 // (op 1077) 


v_mul_lo_u32 v14, v70, 6 // (op 1077) 

v_lshrrev_b32 v19, 3, v14 // (op 1077) 

v_lshlrev_b32 v14, 4, v1 // (op 1101) 
s_nop 0
v_readfirstlane_b32 s0, v14 // (op 1101) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 1140) 
v_sub_u32 v70, v1, v14 // (op 1140) 

v_lshrrev_b32 v14, 3, v1 // (op 1140) 
v_lshl_add_u32 v71, v14, 4, v70 // (op 1140) 


v_and_b32 v14, -16, v1 // (op 1164) 
v_sub_u32 v70, v1, v14 // (op 1164) 

v_lshrrev_b32 v14, 4, v1 // (op 1164) 
v_lshl_add_u32 v72, v14, 5, v70 // (op 1164) 


s_mov_b32 s1, 0 // (op 59) call()
s_ashr_i64 s[8:9], s[26:27], 8 // (op 58) 
s_mov_b32 s18, s8 // (op 58) convert

s_and_b32 s19, s18, 4294967294 // (op 58) 

s_cmp_lt_u32 s1, s19 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58 if false)
RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58:
s_add_u32 s1, s1, 2 // (op 60) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[56:63], a[0:3], v21, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 88) 
s_barrier  // (op 1235) 

s_barrier  // (op 1222) 
s_lshl_b64 s[8:9], s[22:23], 6 // (op 522) 



ds_read_b128 v[64:67], v3 offset:8416 // (op 91) Load local data 
ds_read_b64 v[68:69], v3 offset:8432 // (op 91) Load local data 
v_mov_b64 v[74:75], s[8:9] // (op 522) convert


s_barrier  // (op 1241) 
v_mov_b64 v[76:77], v[16:17] // (op 522) 



ds_read_b64_tr_b8 v[56:57], v0 offset:2048 // (op 95) Transpose load local data 
s_lshl_b64 s[8:9], s[22:23], 8 // (op 522) 
s_lshl_b64 s[18:19], s[8:9], 1 // (op 522) 

v_mov_b32 v14, s18 // (op 522) 
v_mov_b32 v70, s19 // (op 522) 
v_add_co_u32 v78, s[8:9], v76, v14 // (op 522) least significant half
v_addc_co_u32 v79, s[8:9], v77, v70, s[8:9] // (op 522) most significant half




s_mov_b32 m0, s51 // (op 522) reg expression
buffer_load_dwordx4 v78, s[56:59], 0 offen offset: 0 lds // (op 522) Load value direct to lds
v_add_co_u32 v78, s[8:9], v78, v74 // (op 522) least significant half
v_addc_co_u32 v79, s[8:9], 0, v79, s[8:9] // (op 522) most significant half

s_add_u32 m0, m0, 1024 // (op 522) 
buffer_load_dwordx4 v78, s[56:59], 0 offen offset: 0 lds // (op 522) Load value direct to lds
v_add_co_u32 v78, s[8:9], v78, v74 // (op 522) least significant half
v_addc_co_u32 v79, s[8:9], 0, v79, s[8:9] // (op 522) most significant half

s_add_u32 m0, m0, 1024 // (op 522) 
buffer_load_dwordx4 v78, s[56:59], 0 offen offset: 0 lds // (op 522) Load value direct to lds
v_add_co_u32 v78, s[8:9], v78, v74 // (op 522) least significant half
v_addc_co_u32 v79, s[8:9], 0, v79, s[8:9] // (op 522) most significant half

s_add_u32 m0, m0, 1024 // (op 522) 
ds_read_b64_tr_b8 v[58:59], v0 offset:2176 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:3072 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:3200 // (op 95) Transpose load local data 
buffer_load_dwordx4 v78, s[56:59], 0 offen offset: 0 lds // (op 522) Load value direct to lds



s_barrier  // (op 1258) 
v_mov_b64 v[74:75], s[32:33] // (op 25) convert




ds_read_u8 v21, v4 offset:11396 // (op 99) Load local data 
v_mov_b32 v14, v22 // (op 25) 
s_barrier  // (op 1278) 
s_lshl_b32 s8, s32, 4 // (op 25) 


ds_read_u8 v12, v7 offset:8256 // (op 103) Load local data 
v_add_u32 v70, v14, s8 // (op 25) 

buffer_load_ubyte v26, v70, s[64:67], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v70, v70, v74 // (op 25) 
buffer_load_ubyte v27, v70, s[64:67], 0 offen offset: 0 // (op 25) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[56:63], a[0:3], v21, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 105) 


v_mov_b32 v70, v13 // (op 4) 

buffer_load_dwordx4 v[28:31], v70, s[52:55], 0 offen offset:384 // (op 4) Load value
buffer_load_dwordx2 v[32:33], v70, s[52:55], 0 offen offset:400 // (op 4) Load value
buffer_load_dwordx4 v[34:37], v70, s[52:55], 0 offen offset:408 // (op 4) Load value
buffer_load_dwordx2 v[38:39], v70, s[52:55], 0 offen offset:424 // (op 4) Load value

v_mov_b64 v[74:75], s[28:29] // (op 17) convert

v_mov_b32 v73, v20 // (op 17) 
v_mov_b32 v78, v73 // (op 17) 
buffer_load_ubyte v40, v78, s[60:63], 0 offen offset:16 // (op 17) Load value
v_add_u32 v78, v78, v74 // (op 17) 
buffer_load_ubyte v41, v78, s[60:63], 0 offen offset:16 // (op 17) Load value


s_barrier  // (op 1313) 



s_waitcnt vmcnt(19)
ds_write_b8 v72, v24 offset:11520 // (op 304) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v72, v25 offset:11536 // (op 304) Store local data 
s_barrier  // (op 1291) 



s_waitcnt vmcnt(17)
ds_write_b128 v19, v[42:45] offset:11648 // (op 283) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v19, v[46:47] offset:11664 // (op 283) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v19, v[48:51] offset:11672 // (op 283) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v19, v[52:53] offset:11688 // (op 283) Store local data 
s_barrier  // (op 1307) 



s_waitcnt vmcnt(13)
ds_write_b8 v71, v54 offset:14720 // (op 303) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v71, v55 offset:14728 // (op 303) Store local data 
s_barrier  // (op 1294) 



ds_read_b64_tr_b8 v[78:79], v5 offset:4096 // (op 287) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v5 offset:4224 // (op 287) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v5 offset:5120 // (op 287) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v5 offset:5248 // (op 287) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1310) 


ds_read_u8 v74, v11 offset:11520 // (op 291) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1288) 



ds_read_b128 v[86:89], v6 offset:11648 // (op 285) Load local data 
ds_read_b64 v[90:91], v6 offset:11664 // (op 285) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1304) 



ds_read_u8 v75, v8 offset:14720 // (op 289) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[86:91], v[78:85], a[0:3], v75, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 292) 
s_barrier  // (op 1298) 








s_lshl_b64 s[8:9], s[22:23], 6 // (op 534) 
v_mov_b64 v[74:75], s[8:9] // (op 534) convert


s_lshl_b64 s[8:9], s[22:23], 8 // (op 534) 
s_mul_i32 s19, 3, s9 // (op 534) most significant: low of low * high
s_mul_i32 s18, 3, s8 // (op 534) least significant
s_mul_hi_u32 s68, 3, s8 // (op 534) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 534) most significant: sum


v_mov_b32 v78, s18 // (op 534) 
v_mov_b32 v79, s19 // (op 534) 
ds_read_b128 v[80:83], v6 offset:11744 // (op 294) Load local data 
ds_read_b64 v[84:85], v6 offset:11760 // (op 294) Load local data 
v_add_co_u32 v86, s[8:9], v76, v78 // (op 534) least significant half
v_addc_co_u32 v87, s[8:9], v77, v79, s[8:9] // (op 534) most significant half




s_barrier  // (op 1301) 
s_barrier  // (op 1368) 



s_add_u32 s8, 4096, s0 // (op 534) 
s_mov_b32 m0, s8 // (op 534) reg expression
buffer_load_dwordx4 v86, s[56:59], 0 offen offset: 0 lds // (op 534) Load value direct to lds
v_add_co_u32 v86, s[18:19], v86, v74 // (op 534) least significant half
v_addc_co_u32 v87, s[18:19], 0, v87, s[18:19] // (op 534) most significant half

s_add_u32 m0, m0, 1024 // (op 534) 
buffer_load_dwordx4 v86, s[56:59], 0 offen offset: 0 lds // (op 534) Load value direct to lds
v_add_co_u32 v86, s[18:19], v86, v74 // (op 534) least significant half
v_addc_co_u32 v87, s[18:19], 0, v87, s[18:19] // (op 534) most significant half

s_add_u32 m0, m0, 1024 // (op 534) 
buffer_load_dwordx4 v86, s[56:59], 0 offen offset: 0 lds // (op 534) Load value direct to lds
v_add_co_u32 v86, s[18:19], v86, v74 // (op 534) least significant half
v_addc_co_u32 v87, s[18:19], 0, v87, s[18:19] // (op 534) most significant half

s_add_u32 m0, m0, 1024 // (op 534) 
ds_read_b64_tr_b8 v[88:89], v5 offset:6144 // (op 296) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v5 offset:6272 // (op 296) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v5 offset:7168 // (op 296) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v5 offset:7296 // (op 296) Transpose load local data 
buffer_load_dwordx4 v86, s[56:59], 0 offen offset: 0 lds // (op 534) Load value direct to lds








ds_read_u8 v74, v8 offset:14724 // (op 298) Load local data 
v_mov_b64 v[86:87], s[32:33] // (op 307) convert

s_lshl_b32 s8, s32, 3 // (op 307) 


s_barrier  // (op 1374) 
ds_read_u8 v75, v11 offset:11584 // (op 300) Load local data 
s_mul_i32 s9, 3, s8 // (op 307) 

v_add_u32 v76, v14, s9 // (op 307) 

s_barrier  // (op 1378) 
buffer_load_ubyte v24, v76, s[64:67], 0 offen offset: 0 // (op 307) Load value
v_add_u32 v76, v76, v86 // (op 307) 
buffer_load_ubyte v25, v76, s[64:67], 0 offen offset: 0 // (op 307) Load value


s_lshl_b64 s[8:9], s[22:23], 8 // (op 658) 

s_lshl_b64 s[18:19], s[8:9], 1 // (op 658) 




s_lshl_b32 s8, s32, 4 // (op 786) 
v_add_u32 v22, v22, s8 // (op 786) 

v_mov_b32 v14, s18 // (op 658) 
v_mov_b32 v76, s19 // (op 658) 
v_add_co_u32 v16, s[8:9], v16, v14 // (op 658) least significant half
v_addc_co_u32 v17, s[8:9], v17, v76, s[8:9] // (op 658) most significant half




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:85], v[88:95], a[0:3], v74, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 301) 
buffer_load_dwordx4 v[42:45], v70, s[52:55], 0 offen offset:576 // (op 282) Load value




buffer_load_dwordx2 v[46:47], v70, s[52:55], 0 offen offset:592 // (op 282) Load value
buffer_load_dwordx4 v[48:51], v70, s[52:55], 0 offen offset:600 // (op 282) Load value
buffer_load_dwordx2 v[52:53], v70, s[52:55], 0 offen offset:616 // (op 282) Load value


v_add_u32 v13, 384, v13 // (op 594) 
v_mov_b64 v[74:75], s[28:29] // (op 306) convert

v_mov_b32 v14, v73 // (op 306) 
buffer_load_ubyte v54, v14, s[60:63], 0 offen offset:24 // (op 306) Load value
v_add_u32 v14, v14, v74 // (op 306) 
buffer_load_ubyte v55, v14, s[60:63], 0 offen offset:24 // (op 306) Load value



s_barrier  // (op 1271) 
v_add_u32 v20, 16, v20 // (op 722) 



s_waitcnt vmcnt(19)
ds_write_b8 v23, v26 offset:8192 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v23, v27 offset:8208 // (op 30) Store local data 
s_barrier  // (op 1216) 



s_waitcnt vmcnt(17)
ds_write_b128 v15, v[28:31] offset:8320 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v15, v[32:33] offset:8336 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v15, v[34:37] offset:8344 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v15, v[38:39] offset:8360 // (op 6) Store local data 
s_barrier  // (op 1251) 



s_waitcnt vmcnt(13)
ds_write_b8 v18, v40 offset:11392 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v18, v41 offset:11400 // (op 21) Store local data 



ds_read_b64_tr_b8 v[56:57], v0 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:128 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:1024 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:1152 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1281) 


ds_read_u8 v12, v7 offset:8192 // (op 85) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1225) 



ds_read_b128 v[64:67], v3 offset:8320 // (op 70) Load local data 
ds_read_b64 v[68:69], v3 offset:8336 // (op 70) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1261) 



ds_read_u8 v21, v4 offset:11392 // (op 80) Load local data 
s_ashr_i64 s[8:9], s[26:27], 8 // (op 58) 
s_mov_b32 s18, s8 // (op 58) convert

s_and_b32 s19, s18, 4294967294 // (op 58) 

s_cmp_lt_u32 s1, s19 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58 if true)
RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58:
















s_barrier  // (op 1326) 
s_mul_hi_i32 s51, s16, s11 // (op 557) 
v_lshrrev_b32 v13, 2, v1 // (op 557) 
s_add_i32 s0, s51, s16 // (op 557) 

v_and_b32 v14, -4, v1 // (op 557) 
s_ashr_i32 s1, s0, 31 // (op 557) 
v_sub_u32 v15, v1, v14 // (op 557) 

s_cmp_eq_i32 s11, 0 // (op 557) 
s_mov_b32 s51, scc // (op 557) 
v_lshlrev_b32 v14, 6, v15 // (op 557) 

s_cmp_eq_i32 s51, 0 // (op 557) 
s_cselect_b32 s8, 0, -1 // (op 557) 

s_and_b32 s9, s12, 31 // (op 557) 
s_lshl_b32 s51, 1, s9 // (op 557) 
s_add_i32 s18, s51, s8 // (op 557) 


s_and_b32 s8, s1, s18 // (op 557) 


s_add_i32 s1, s0, s8 // (op 557) 


s_ashr_i32 s0, s1, s9 // (op 557) 


s_lshl_b32 s1, s12, 24 // (op 557) 
s_ashr_i32 s51, s1, 31 // (op 557) 

s_xor_b32 s1, s0, s51 // (op 557) 

s_sub_i32 s0, s1, s51 // (op 557) 


s_mul_hi_i32 s1, s0, s13 // (op 557) 
s_add_i32 s51, s1, s0 // (op 557) 

s_ashr_i32 s1, s51, 31 // (op 557) 
s_cmp_eq_i32 s13, 0 // (op 557) 
s_mov_b32 s8, scc // (op 557) 
s_cmp_eq_i32 s8, 0 // (op 557) 
s_cselect_b32 s9, 0, -1 // (op 557) 

s_and_b32 s8, s14, 31 // (op 557) 
s_lshl_b32 s18, 1, s8 // (op 557) 
s_add_i32 s19, s18, s9 // (op 557) 


s_and_b32 s9, s1, s19 // (op 557) 


s_add_i32 s1, s51, s9 // (op 557) 


s_ashr_i32 s9, s1, s8 // (op 557) 


s_lshl_b32 s1, s14, 24 // (op 557) 
s_ashr_i32 s8, s1, 31 // (op 557) 

s_xor_b32 s1, s9, s8 // (op 557) 

s_sub_i32 s9, s1, s8 // (op 557) 


s_mul_i32 s1, s0, s46 // (op 557) 

s_sub_u32 s0, s16, s1 // (op 557) 

s_mul_i32 s1, s9, s46 // (op 557) 
s_add_u32 s1, s1, s0 // (op 557) 


s_mul_hi_i32 s0, s16, s10 // (op 557) 
s_add_i32 s51, s0, s16 // (op 557) 

s_ashr_i32 s0, s51, 31 // (op 557) 
s_cmp_eq_i32 s10, 0 // (op 557) 
s_mov_b32 s8, scc // (op 557) 
s_cmp_eq_i32 s8, 0 // (op 557) 
s_cselect_b32 s9, 0, -1 // (op 557) 

s_and_b32 s8, s49, 31 // (op 557) 
s_lshl_b32 s18, 1, s8 // (op 557) 
s_add_i32 s19, s18, s9 // (op 557) 


s_and_b32 s9, s0, s19 // (op 557) 


s_add_i32 s0, s51, s9 // (op 557) 


s_ashr_i32 s9, s0, s8 // (op 557) 


s_lshl_b32 s0, s49, 24 // (op 557) 
s_ashr_i32 s8, s0, 31 // (op 557) 

s_xor_b32 s0, s9, s8 // (op 557) 

s_sub_i32 s9, s0, s8 // (op 557) 


s_add_u32 s18, s2, 16 // (op 557) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 557) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 557) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 557) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 557) 

s_mul_i32 s0, s46, s18 // (op 557) 

s_mul_i32 s8, s50, s0 // (op 557) 

s_mul_i32 s0, s9, s8 // (op 557) 

s_sub_u32 s8, s16, s0 // (op 557) 

s_mov_b32 s0, s8 // (op 557) 
s_mul_hi_i32 s51, s0, s15 // (op 557) 
s_add_i32 s18, s51, s0 // (op 557) 


s_ashr_i32 s0, s18, 31 // (op 557) 
s_cmp_eq_i32 s15, 0 // (op 557) 
s_mov_b32 s19, scc // (op 557) 
s_cmp_eq_i32 s19, 0 // (op 557) 
s_cselect_b32 s51, 0, -1 // (op 557) 

s_and_b32 s19, s48, 31 // (op 557) 
s_lshl_b32 s68, 1, s19 // (op 557) 
s_add_i32 s69, s68, s51 // (op 557) 


s_and_b32 s51, s0, s69 // (op 557) 


s_add_i32 s0, s18, s51 // (op 557) 


s_ashr_i32 s18, s0, s19 // (op 557) 


s_lshl_b32 s0, s48, 24 // (op 557) 
s_ashr_i32 s19, s0, 31 // (op 557) 

s_xor_b32 s0, s18, s19 // (op 557) 

s_sub_i32 s18, s0, s19 // (op 557) 


s_mul_i32 s0, s47, s46 // (op 557) 
s_add_u32 s68, s6, 16 // (op 557) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 557) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 557) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 557) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 557) 

s_sub_i32 s19, s68, s0 // (op 557) 

s_mul_i32 s51, s18, s19 // (op 557) 


s_sub_u32 s18, s8, s51 // (op 557) 


s_add_u32 s8, s18, s0 // (op 557) 


s_cmp_eq_i32 s9, 0 // (op 557) 
s_mov_b32 s0, scc // (op 557) 

s_cmp_eq_i32 s0, 0 // (op 557) 
s_cselect_b32 s9, s8, s1 // (op 557) 



v_lshl_add_u32 v15, s9, 4, v13 // (op 557) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 557) 
s_lshr_b32 s51, s0, 1 // (op 557) 

v_lshl_add_u32 v13, s51, 9, v14 // (op 557) 


v_mov_b32 v23, 0 // (op 557) convert
v_mov_b32 v22, v13 // (op 557) convert

v_mul_lo_u32 v17, v15, s5 // (op 557) most significant: low of low * high
v_mul_hi_u32 v13, v15, s4 // (op 557) most significant: high of low * low
v_mul_lo_u32 v16, v15, s4 // (op 557) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 557) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v22 // (op 557) least significant half
v_addc_co_u32 v17, s[0:1], v17, v23, s[0:1] // (op 557) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 557) most significant: low of high * low
v_mul_hi_u32 v13, v16, 6 // (op 557) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 557) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 557) most significant: sum


v_lshrrev_b64 v[22:23], 3, v[14:15] // (op 557) 

v_mov_b32 v13, v22 // (op 557) convert

s_mul_i32 s1, s7, 6 // (op 559) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 559) least significant
s_mul_hi_u32 s51, s6, 6 // (op 559) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 559) most significant: sum

s_mul_i32 s9, s4, s1 // (op 559) most significant: low of low * high
s_mul_i32 s51, s5, s0 // (op 559) most significant: low of high * low
s_mul_i32 s8, s4, s0 // (op 559) least significant
s_mul_hi_u32 s18, s4, s0 // (op 559) most significant: high of low * low
s_add_u32 s9, s9, s51 // (op 559) most significant: sum
s_add_u32 s9, s9, s18 // (op 559) most significant: sum



s_ashr_i64 s[0:1], s[8:9], 3 // (op 559) 

s_mov_b32 s52, s20 // (op 559) reg expression
s_mov_b32 s53, s21 // (op 559) reg expression
s_mov_b32 s54, s0 // (op 559) reg expression
s_mov_b32 s55, 131072 // (op 559) call()

s_mul_hi_i32 s51, s16, s11 // (op 624) 
s_add_i32 s0, s51, s16 // (op 624) 

s_ashr_i32 s1, s0, 31 // (op 624) 
s_cmp_eq_i32 s11, 0 // (op 624) 
s_mov_b32 s51, scc // (op 624) 
s_cmp_eq_i32 s51, 0 // (op 624) 
s_cselect_b32 s8, 0, -1 // (op 624) 

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


s_add_u32 s18, s2, 16 // (op 624) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 624) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 624) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 624) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 624) 

s_mul_i32 s1, s9, s18 // (op 624) 

s_sub_i32 s51, s0, s1 // (op 624) 


s_mul_hi_i32 s0, s16, s10 // (op 624) 
s_add_i32 s1, s0, s16 // (op 624) 

s_ashr_i32 s0, s1, 31 // (op 624) 
s_cmp_eq_i32 s10, 0 // (op 624) 
s_mov_b32 s8, scc // (op 624) 
s_cmp_eq_i32 s8, 0 // (op 624) 
s_cselect_b32 s9, 0, -1 // (op 624) 

s_and_b32 s8, s49, 31 // (op 624) 
s_lshl_b32 s68, 1, s8 // (op 624) 
s_add_i32 s69, s68, s9 // (op 624) 


s_and_b32 s9, s0, s69 // (op 624) 


s_add_i32 s0, s1, s9 // (op 624) 


s_ashr_i32 s1, s0, s8 // (op 624) 


s_lshl_b32 s0, s49, 24 // (op 624) 
s_ashr_i32 s8, s0, 31 // (op 624) 

s_xor_b32 s0, s1, s8 // (op 624) 

s_sub_i32 s1, s0, s8 // (op 624) 


s_mul_i32 s0, s46, s18 // (op 624) 

s_mul_i32 s8, s50, s0 // (op 624) 

s_mul_i32 s0, s1, s8 // (op 624) 

s_sub_u32 s8, s16, s0 // (op 624) 

s_mul_hi_i32 s0, s8, s15 // (op 624) 
s_add_i32 s9, s0, s8 // (op 624) 


s_ashr_i32 s0, s9, 31 // (op 624) 
s_cmp_eq_i32 s15, 0 // (op 624) 
s_mov_b32 s8, scc // (op 624) 
s_cmp_eq_i32 s8, 0 // (op 624) 
s_cselect_b32 s18, 0, -1 // (op 624) 

s_and_b32 s8, s48, 31 // (op 624) 
s_lshl_b32 s19, 1, s8 // (op 624) 
s_add_i32 s68, s19, s18 // (op 624) 


s_and_b32 s18, s0, s68 // (op 624) 


s_add_i32 s0, s9, s18 // (op 624) 


s_ashr_i32 s9, s0, s8 // (op 624) 


s_lshl_b32 s0, s48, 24 // (op 624) 
s_ashr_i32 s8, s0, 31 // (op 624) 

s_xor_b32 s0, s9, s8 // (op 624) 

s_sub_i32 s9, s0, s8 // (op 624) 


s_cmp_eq_i32 s1, 0 // (op 624) 
s_mov_b32 s0, scc // (op 624) 

s_cmp_eq_i32 s0, 0 // (op 624) 
s_cselect_b32 s1, s9, s51 // (op 624) 



s_lshl_b32 s0, s1, 4 // (op 624) 

s_mov_b32 s9, 0 // (op 624) convert
s_mov_b32 s8, s0 // (op 624) convert

s_ashr_i64 s[0:1], s[26:27], 8 // (op 624) 
s_lshr_b32 s51, s0, 1 // (op 624) 

v_lshl_add_u32 v14, s51, 9, v1 // (op 624) 

v_mul_lo_u32 v23, v14, s23 // (op 624) most significant: low of low * high
v_mul_hi_u32 v15, v14, s22 // (op 624) most significant: high of low * low
v_mul_lo_u32 v22, v14, s22 // (op 624) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 624) most significant: sum

v_mov_b32 v15, s8 // (op 624) 
v_mov_b32 v16, s9 // (op 624) 
v_add_co_u32 v22, s[0:1], v22, v15 // (op 624) least significant half
v_addc_co_u32 v23, s[0:1], v23, v16, s[0:1] // (op 624) most significant half





s_mul_i32 s1, s22, s27 // (op 626) most significant: low of low * high
s_mul_i32 s51, s23, s26 // (op 626) most significant: low of high * low
s_mul_i32 s0, s22, s26 // (op 626) least significant
s_mul_hi_u32 s8, s22, s26 // (op 626) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 626) most significant: sum
s_add_u32 s1, s1, s8 // (op 626) most significant: sum


s_mov_b32 s56, s24 // (op 626) reg expression
s_mov_b32 s57, s25 // (op 626) reg expression
s_mov_b32 s58, s0 // (op 626) reg expression
s_mov_b32 s59, 131072 // (op 626) call()

s_mul_hi_i32 s51, s16, s11 // (op 688) 
v_lshrrev_b32 v14, 3, v1 // (op 688) 
s_add_i32 s0, s51, s16 // (op 688) 

v_lshlrev_b32 v15, 1, v14 // (op 688) 

s_ashr_i32 s1, s0, 31 // (op 688) 
v_and_b32 v14, -8, v1 // (op 688) 
s_cmp_eq_i32 s11, 0 // (op 688) 
s_mov_b32 s51, scc // (op 688) 
v_sub_u32 v16, v1, v14 // (op 688) 

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


s_mul_i32 s1, s0, s46 // (op 688) 

s_sub_u32 s0, s16, s1 // (op 688) 

s_mul_i32 s1, s9, s46 // (op 688) 
s_add_u32 s1, s1, s0 // (op 688) 


s_mul_hi_i32 s0, s16, s10 // (op 688) 
s_add_i32 s51, s0, s16 // (op 688) 

s_ashr_i32 s0, s51, 31 // (op 688) 
s_cmp_eq_i32 s10, 0 // (op 688) 
s_mov_b32 s8, scc // (op 688) 
s_cmp_eq_i32 s8, 0 // (op 688) 
s_cselect_b32 s9, 0, -1 // (op 688) 

s_and_b32 s8, s49, 31 // (op 688) 
s_lshl_b32 s18, 1, s8 // (op 688) 
s_add_i32 s19, s18, s9 // (op 688) 


s_and_b32 s9, s0, s19 // (op 688) 


s_add_i32 s0, s51, s9 // (op 688) 


s_ashr_i32 s9, s0, s8 // (op 688) 


s_lshl_b32 s0, s49, 24 // (op 688) 
s_ashr_i32 s8, s0, 31 // (op 688) 

s_xor_b32 s0, s9, s8 // (op 688) 

s_sub_i32 s9, s0, s8 // (op 688) 


s_add_u32 s18, s2, 16 // (op 688) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 688) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 688) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 688) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 688) 

s_mul_i32 s0, s46, s18 // (op 688) 

s_mul_i32 s8, s50, s0 // (op 688) 

s_mul_i32 s0, s9, s8 // (op 688) 

s_sub_u32 s8, s16, s0 // (op 688) 

s_mov_b32 s0, s8 // (op 688) 
s_mul_hi_i32 s51, s0, s15 // (op 688) 
s_add_i32 s18, s51, s0 // (op 688) 


s_ashr_i32 s0, s18, 31 // (op 688) 
s_cmp_eq_i32 s15, 0 // (op 688) 
s_mov_b32 s19, scc // (op 688) 
s_cmp_eq_i32 s19, 0 // (op 688) 
s_cselect_b32 s51, 0, -1 // (op 688) 

s_and_b32 s19, s48, 31 // (op 688) 
s_lshl_b32 s68, 1, s19 // (op 688) 
s_add_i32 s69, s68, s51 // (op 688) 


s_and_b32 s51, s0, s69 // (op 688) 


s_add_i32 s0, s18, s51 // (op 688) 


s_ashr_i32 s18, s0, s19 // (op 688) 


s_lshl_b32 s0, s48, 24 // (op 688) 
s_ashr_i32 s19, s0, 31 // (op 688) 

s_xor_b32 s0, s18, s19 // (op 688) 

s_sub_i32 s18, s0, s19 // (op 688) 


s_mul_i32 s0, s47, s46 // (op 688) 
s_add_u32 s68, s6, 16 // (op 688) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 688) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 688) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 688) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 688) 

s_sub_i32 s19, s68, s0 // (op 688) 

s_mul_i32 s51, s18, s19 // (op 688) 


s_sub_u32 s18, s8, s51 // (op 688) 


s_add_u32 s8, s18, s0 // (op 688) 


s_cmp_eq_i32 s9, 0 // (op 688) 
s_mov_b32 s0, scc // (op 688) 

s_cmp_eq_i32 s0, 0 // (op 688) 
s_cselect_b32 s9, s8, s1 // (op 688) 



v_lshl_add_u32 v14, s9, 4, v15 // (op 688) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 688) 
s_lshr_b32 s51, s0, 1 // (op 688) 

v_lshl_add_u32 v15, s51, 4, v16 // (op 688) 


v_mul_lo_u32 v16, v14, s28 // (op 688) 
v_add_u32 v16, v16, v15 // (op 688) 


s_mul_i32 s1, s28, s7 // (op 690) most significant: low of low * high
s_mul_i32 s51, s29, s6 // (op 690) most significant: low of high * low
s_mul_i32 s0, s28, s6 // (op 690) least significant
s_mul_hi_u32 s8, s28, s6 // (op 690) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 690) most significant: sum
s_add_u32 s1, s1, s8 // (op 690) most significant: sum


s_mov_b32 s60, s30 // (op 690) reg expression
s_mov_b32 s61, s31 // (op 690) reg expression
s_mov_b32 s62, s0 // (op 690) reg expression
s_mov_b32 s63, 131072 // (op 690) call()

s_mul_hi_i32 s51, s16, s11 // (op 752) 
v_and_b32 v14, -16, v1 // (op 752) 
s_add_i32 s0, s51, s16 // (op 752) 

v_sub_u32 v15, v1, v14 // (op 752) 

s_ashr_i32 s1, s0, 31 // (op 752) 
v_lshrrev_b32 v14, 4, v1 // (op 752) 
s_cmp_eq_i32 s11, 0 // (op 752) 
s_mov_b32 s51, scc // (op 752) 
v_lshlrev_b32 v17, 1, v14 // (op 752) 

s_cmp_eq_i32 s51, 0 // (op 752) 
s_cselect_b32 s8, 0, -1 // (op 752) 

s_and_b32 s9, s12, 31 // (op 752) 
s_lshl_b32 s51, 1, s9 // (op 752) 
s_add_i32 s18, s51, s8 // (op 752) 


s_and_b32 s8, s1, s18 // (op 752) 


s_add_i32 s1, s0, s8 // (op 752) 


s_ashr_i32 s0, s1, s9 // (op 752) 


s_lshl_b32 s1, s12, 24 // (op 752) 
s_ashr_i32 s51, s1, 31 // (op 752) 

s_xor_b32 s1, s0, s51 // (op 752) 

s_sub_i32 s0, s1, s51 // (op 752) 


s_mul_hi_i32 s1, s0, s13 // (op 752) 
s_add_i32 s51, s1, s0 // (op 752) 

s_ashr_i32 s1, s51, 31 // (op 752) 
s_cmp_eq_i32 s13, 0 // (op 752) 
s_mov_b32 s8, scc // (op 752) 
s_cmp_eq_i32 s8, 0 // (op 752) 
s_cselect_b32 s9, 0, -1 // (op 752) 

s_and_b32 s8, s14, 31 // (op 752) 
s_lshl_b32 s18, 1, s8 // (op 752) 
s_add_i32 s19, s18, s9 // (op 752) 


s_and_b32 s9, s1, s19 // (op 752) 


s_add_i32 s1, s51, s9 // (op 752) 


s_ashr_i32 s9, s1, s8 // (op 752) 


s_lshl_b32 s1, s14, 24 // (op 752) 
s_ashr_i32 s8, s1, 31 // (op 752) 

s_xor_b32 s1, s9, s8 // (op 752) 

s_sub_i32 s9, s1, s8 // (op 752) 


s_add_u32 s18, s2, 16 // (op 752) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 752) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 752) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 752) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 752) 

s_mul_i32 s1, s9, s18 // (op 752) 

s_sub_i32 s51, s0, s1 // (op 752) 


s_mul_hi_i32 s0, s16, s10 // (op 752) 
s_add_i32 s1, s0, s16 // (op 752) 

s_ashr_i32 s0, s1, 31 // (op 752) 
s_cmp_eq_i32 s10, 0 // (op 752) 
s_mov_b32 s8, scc // (op 752) 
s_cmp_eq_i32 s8, 0 // (op 752) 
s_cselect_b32 s9, 0, -1 // (op 752) 

s_and_b32 s8, s49, 31 // (op 752) 
s_lshl_b32 s68, 1, s8 // (op 752) 
s_add_i32 s69, s68, s9 // (op 752) 


s_and_b32 s9, s0, s69 // (op 752) 


s_add_i32 s0, s1, s9 // (op 752) 


s_ashr_i32 s1, s0, s8 // (op 752) 


s_lshl_b32 s0, s49, 24 // (op 752) 
s_ashr_i32 s8, s0, 31 // (op 752) 

s_xor_b32 s0, s1, s8 // (op 752) 

s_sub_i32 s1, s0, s8 // (op 752) 


s_mul_i32 s0, s46, s18 // (op 752) 

s_mul_i32 s8, s50, s0 // (op 752) 

s_mul_i32 s0, s1, s8 // (op 752) 

s_sub_u32 s8, s16, s0 // (op 752) 

s_mul_hi_i32 s0, s8, s15 // (op 752) 
s_add_i32 s9, s0, s8 // (op 752) 


s_ashr_i32 s0, s9, 31 // (op 752) 
s_cmp_eq_i32 s15, 0 // (op 752) 
s_mov_b32 s8, scc // (op 752) 
s_cmp_eq_i32 s8, 0 // (op 752) 
s_cselect_b32 s18, 0, -1 // (op 752) 

s_and_b32 s8, s48, 31 // (op 752) 
s_lshl_b32 s19, 1, s8 // (op 752) 
s_add_i32 s68, s19, s18 // (op 752) 


s_and_b32 s18, s0, s68 // (op 752) 


s_add_i32 s0, s9, s18 // (op 752) 


s_ashr_i32 s9, s0, s8 // (op 752) 


s_lshl_b32 s0, s48, 24 // (op 752) 
s_ashr_i32 s8, s0, 31 // (op 752) 

s_xor_b32 s0, s9, s8 // (op 752) 

s_sub_i32 s9, s0, s8 // (op 752) 


s_cmp_eq_i32 s1, 0 // (op 752) 
s_mov_b32 s0, scc // (op 752) 

s_cmp_eq_i32 s0, 0 // (op 752) 
s_cselect_b32 s1, s9, s51 // (op 752) 



v_lshl_add_u32 v14, s1, 4, v15 // (op 752) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 752) 
s_lshr_b32 s51, s0, 1 // (op 752) 

v_lshl_add_u32 v15, s51, 4, v17 // (op 752) 


v_mul_lo_u32 v17, v15, s32 // (op 752) 
v_add_u32 v17, v17, v14 // (op 752) 


s_mul_i32 s1, s32, s37 // (op 754) most significant: low of low * high
s_mul_i32 s51, s33, s36 // (op 754) most significant: low of high * low
s_mul_i32 s0, s32, s36 // (op 754) least significant
s_mul_hi_u32 s8, s32, s36 // (op 754) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 754) most significant: sum
s_add_u32 s1, s1, s8 // (op 754) most significant: sum


s_mov_b32 s64, s34 // (op 754) reg expression
s_mov_b32 s65, s35 // (op 754) reg expression
s_mov_b32 s66, s0 // (op 754) reg expression
s_mov_b32 s67, 131072 // (op 754) call()

v_and_b32 v14, -4, v1 // (op 845) 
v_sub_u32 v15, v1, v14 // (op 845) 

v_lshlrev_b32 v14, 6, v15 // (op 845) 

v_lshrrev_b32 v15, 2, v1 // (op 845) 
v_lshl_add_u32 v18, v15, 8, v14 // (op 845) 


v_mul_lo_u32 v14, v18, 6 // (op 845) 

v_lshrrev_b32 v15, 3, v14 // (op 845) 

v_lshlrev_b32 v14, 4, v1 // (op 898) 
s_nop 0
v_readfirstlane_b32 s51, v14 // (op 898) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 950) 
v_sub_u32 v18, v1, v14 // (op 950) 

v_lshrrev_b32 v14, 3, v1 // (op 950) 
v_lshl_add_u32 v19, v14, 4, v18 // (op 950) 


v_and_b32 v14, -16, v1 // (op 1003) 
v_sub_u32 v18, v1, v14 // (op 1003) 

v_lshrrev_b32 v14, 4, v1 // (op 1003) 
v_lshl_add_u32 v20, v14, 5, v18 // (op 1003) 


s_ashr_i64 s[0:1], s[26:27], 8 // (op 194) 
s_and_b32 s8, s0, 4294967294 // (op 194) 

s_ashr_i64 s[0:1], s[26:27], 8 // (op 193) 
s_mov_b32 s9, s0 // (op 193) convert

s_cmp_lt_u32 s8, s9 // (op 193) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193 // (op 193) Condition: Top (jump to RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193:


s_add_u32 s8, s8, 1 // (op 195) 
v_mov_b32 v12, v13 // (op 198) 

buffer_load_dwordx4 v[24:27], v12, s[52:55], 0 offen // (op 198) Load value
buffer_load_dwordx2 v[28:29], v12, s[52:55], 0 offen offset:16 // (op 198) Load value
buffer_load_dwordx4 v[30:33], v12, s[52:55], 0 offen offset:24 // (op 198) Load value
buffer_load_dwordx2 v[34:35], v12, s[52:55], 0 offen offset:40 // (op 198) Load value

v_add_u32 v13, 192, v13 // (op 547) 
s_barrier  // (op 1219) 
s_barrier  // (op 1238) 




s_lshl_b64 s[0:1], s[22:23], 6 // (op 527) 
v_mov_b64 v[36:37], s[0:1] // (op 527) convert


v_mov_b64 v[38:39], v[22:23] // (op 527) 
v_mov_b64 v[40:41], v[38:39] // (op 527) reg expression
s_mov_b32 m0, s51 // (op 527) reg expression
buffer_load_dwordx4 v40, s[56:59], 0 offen offset: 0 lds // (op 527) Load value direct to lds
v_add_co_u32 v40, s[0:1], v40, v36 // (op 527) least significant half
v_addc_co_u32 v41, s[0:1], 0, v41, s[0:1] // (op 527) most significant half

s_add_u32 m0, m0, 1024 // (op 527) 
buffer_load_dwordx4 v40, s[56:59], 0 offen offset: 0 lds // (op 527) Load value direct to lds
v_add_co_u32 v40, s[0:1], v40, v36 // (op 527) least significant half
v_addc_co_u32 v41, s[0:1], 0, v41, s[0:1] // (op 527) most significant half

s_add_u32 m0, m0, 1024 // (op 527) 
s_waitcnt vmcnt(5)
ds_write_b128 v15, v[24:27] offset:8320 // (op 199) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v15, v[28:29] offset:8336 // (op 199) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v15, v[30:33] offset:8344 // (op 199) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v15, v[34:35] offset:8360 // (op 199) Store local data 
buffer_load_dwordx4 v40, s[56:59], 0 offen offset: 0 lds // (op 527) Load value direct to lds

v_add_co_u32 v40, s[0:1], v40, v36 // (op 527) least significant half
v_addc_co_u32 v41, s[0:1], 0, v41, s[0:1] // (op 527) most significant half

s_add_u32 m0, m0, 1024 // (op 527) 
buffer_load_dwordx4 v40, s[56:59], 0 offen offset: 0 lds // (op 527) Load value direct to lds



s_lshl_b64 s[0:1], s[22:23], 8 // (op 616) 
v_mov_b32 v12, s0 // (op 616) 
v_mov_b32 v14, s1 // (op 616) 
v_add_co_u32 v22, s[18:19], v22, v12 // (op 616) least significant half
v_addc_co_u32 v23, s[18:19], v23, v14, s[18:19] // (op 616) most significant half





v_mov_b64 v[24:25], s[28:29] // (op 222) convert

v_mov_b32 v12, v16 // (op 222) 
v_mov_b32 v14, v12 // (op 222) 
buffer_load_ubyte v26, v14, s[60:63], 0 offen offset: 0 // (op 222) Load value
v_add_u32 v14, v14, v24 // (op 222) 
buffer_load_ubyte v27, v14, s[60:63], 0 offen offset: 0 // (op 222) Load value



v_add_u32 v16, 8, v16 // (op 680) 

s_barrier  // (op 1254) 
v_mov_b64 v[24:25], s[32:33] // (op 223) convert




v_mov_b32 v12, v17 // (op 223) 
v_mov_b32 v14, v12 // (op 223) reg expression
buffer_load_ubyte v28, v14, s[64:67], 0 offen offset: 0 // (op 223) Load value
v_add_u32 v14, v14, v24 // (op 223) 
buffer_load_ubyte v29, v14, s[64:67], 0 offen offset: 0 // (op 223) Load value



s_lshl_b32 s9, s32, 3 // (op 744) 
v_add_u32 v17, v17, s9 // (op 744) 

s_waitcnt vmcnt(3)
ds_write_b8 v19, v26 offset:11392 // (op 219) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1228) 
s_waitcnt vmcnt(2)
ds_write_b8 v19, v27 offset:11400 // (op 219) Store local data 




ds_read_b128 v[30:33], v3 offset:8320 // (op 201) Load local data 
s_barrier  // (op 1274) 
ds_read_b64 v[34:35], v3 offset:8336 // (op 201) Load local data 



s_barrier  // (op 1244) 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v28 offset:8192 // (op 220) Store local data 



ds_read_b64_tr_b8 v[36:37], v0 // (op 203) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:128 // (op 203) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:1024 // (op 203) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:1152 // (op 203) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v29 offset:8208 // (op 220) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 1264) 



ds_read_u8 v12, v4 offset:11392 // (op 205) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1284) 


ds_read_u8 v14, v7 offset:8192 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:35], v[36:43], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 208) 







ds_read_b128 v[24:27], v3 offset:8416 // (op 210) Load local data 
ds_read_b64 v[28:29], v3 offset:8432 // (op 210) Load local data 



ds_read_b64_tr_b8 v[30:31], v0 offset:2048 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:2176 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:3072 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:3200 // (op 212) Transpose load local data 



ds_read_u8 v12, v4 offset:11396 // (op 214) Load local data 


ds_read_u8 v14, v7 offset:8256 // (op 216) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[30:37], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 217) 




s_ashr_i64 s[0:1], s[26:27], 8 // (op 193) 
s_mov_b32 s9, s0 // (op 193) convert

s_cmp_lt_u32 s8, s9 // (op 193) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193 // (op 193) Condition: Bottom (jump to RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193 if true)
RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193:























s_barrier  // (op 1360) 








v_cmp_eq_f32 vcc, 0.00000, v9 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v10, v4 // (op 50) 
v_mul_f32 v13, v10, v5 // (op 50) 
v_mul_f32 v14, v10, v6 // (op 50) 
v_mul_f32 v15, v10, v7 // (op 50) 

s_mul_hi_i32 s0, s16, s11 // (op 1045) 
v_lshrrev_b32 v0, 6, v1 // (op 1045) 
s_add_i32 s1, s0, s16 // (op 1045) 

v_and_b32 v11, 63, v1 // (op 1045) 
s_ashr_i32 s0, s1, 31 // (op 1045) 
v_lshrrev_b32 v3, 4, v11 // (op 1045) 
s_cmp_eq_i32 s11, 0 // (op 1045) 
s_mov_b32 s4, scc // (op 1045) 
v_lshlrev_b32 v4, 2, v3 // (op 1045) 

s_cmp_eq_i32 s4, 0 // (op 1045) 
s_cselect_b32 s5, 0, -1 // (op 1045) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1045) 


s_and_b32 s4, s12, 31 // (op 1045) 
v_lshrrev_b32 v0, 2, v11 // (op 1045) 

s_lshl_b32 s8, 1, s4 // (op 1045) 
v_and_b32 v11, 3, v0 // (op 1045) 

s_add_i32 s9, s8, s5 // (op 1045) 


v_and_b32 v0, 3, v1 // (op 1045) 
s_and_b32 s5, s0, s9 // (op 1045) 


v_lshl_add_u32 v4, v11, 2, v0 // (op 1045) 


s_add_i32 s0, s1, s5 // (op 1045) 


s_ashr_i32 s1, s0, s4 // (op 1045) 


s_lshl_b32 s0, s12, 24 // (op 1045) 
s_ashr_i32 s4, s0, 31 // (op 1045) 

s_xor_b32 s0, s1, s4 // (op 1045) 

s_sub_i32 s1, s0, s4 // (op 1045) 


s_mul_hi_i32 s0, s1, s13 // (op 1045) 
s_add_i32 s4, s0, s1 // (op 1045) 

s_ashr_i32 s0, s4, 31 // (op 1045) 
s_cmp_eq_i32 s13, 0 // (op 1045) 
s_mov_b32 s5, scc // (op 1045) 
s_cmp_eq_i32 s5, 0 // (op 1045) 
s_cselect_b32 s8, 0, -1 // (op 1045) 

s_and_b32 s5, s14, 31 // (op 1045) 
s_lshl_b32 s9, 1, s5 // (op 1045) 
s_add_i32 s18, s9, s8 // (op 1045) 


s_and_b32 s8, s0, s18 // (op 1045) 


s_add_i32 s0, s4, s8 // (op 1045) 


s_ashr_i32 s4, s0, s5 // (op 1045) 


s_lshl_b32 s0, s14, 24 // (op 1045) 
s_ashr_i32 s5, s0, 31 // (op 1045) 

s_xor_b32 s0, s4, s5 // (op 1045) 

s_sub_i32 s4, s0, s5 // (op 1045) 


s_mul_i32 s0, s1, s46 // (op 1045) 
s_sub_u32 s5, s16, s0 // (op 1045) 

s_mul_i32 s0, s4, s46 // (op 1045) 
s_add_u32 s0, s0, s5 // (op 1045) 

s_mul_hi_i32 s5, s16, s10 // (op 1045) 
s_add_i32 s8, s5, s16 // (op 1045) 

s_ashr_i32 s5, s8, 31 // (op 1045) 
s_cmp_eq_i32 s10, 0 // (op 1045) 
s_mov_b32 s9, scc // (op 1045) 
s_cmp_eq_i32 s9, 0 // (op 1045) 
s_cselect_b32 s18, 0, -1 // (op 1045) 

s_and_b32 s9, s49, 31 // (op 1045) 
s_lshl_b32 s19, 1, s9 // (op 1045) 
s_add_i32 s20, s19, s18 // (op 1045) 


s_and_b32 s18, s5, s20 // (op 1045) 


s_add_i32 s5, s8, s18 // (op 1045) 


s_ashr_i32 s8, s5, s9 // (op 1045) 


s_lshl_b32 s5, s49, 24 // (op 1045) 
s_ashr_i32 s9, s5, 31 // (op 1045) 

s_xor_b32 s5, s8, s9 // (op 1045) 

s_sub_i32 s8, s5, s9 // (op 1045) 


s_add_u32 s18, s2, 16 // (op 1045) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 1045) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 1045) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 1045) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 1045) 

s_mul_i32 s5, s46, s18 // (op 1045) 
s_mul_i32 s9, s50, s5 // (op 1045) 

s_mul_i32 s5, s8, s9 // (op 1045) 

s_sub_u32 s9, s16, s5 // (op 1045) 

s_mov_b32 s5, s9 // (op 1045) 
s_mul_hi_i32 s20, s5, s15 // (op 1045) 
s_add_i32 s21, s20, s5 // (op 1045) 


s_ashr_i32 s5, s21, 31 // (op 1045) 
s_cmp_eq_i32 s15, 0 // (op 1045) 
s_mov_b32 s20, scc // (op 1045) 
s_cmp_eq_i32 s20, 0 // (op 1045) 
s_cselect_b32 s22, 0, -1 // (op 1045) 

s_and_b32 s20, s48, 31 // (op 1045) 
s_lshl_b32 s23, 1, s20 // (op 1045) 
s_add_i32 s24, s23, s22 // (op 1045) 


s_and_b32 s22, s5, s24 // (op 1045) 


s_add_i32 s5, s21, s22 // (op 1045) 


s_ashr_i32 s21, s5, s20 // (op 1045) 


s_lshl_b32 s5, s48, 24 // (op 1045) 
s_ashr_i32 s20, s5, 31 // (op 1045) 

s_xor_b32 s5, s21, s20 // (op 1045) 

s_sub_i32 s21, s5, s20 // (op 1045) 


s_mul_i32 s5, s47, s46 // (op 1045) 
s_add_u32 s22, s6, 16 // (op 1045) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1045) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1045) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1045) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1045) 

s_sub_i32 s20, s22, s5 // (op 1045) 

s_mul_i32 s22, s21, s20 // (op 1045) 

s_sub_u32 s20, s9, s22 // (op 1045) 


s_add_u32 s9, s20, s5 // (op 1045) 


s_cmp_eq_i32 s8, 0 // (op 1045) 
s_mov_b32 s5, scc // (op 1045) 

s_cmp_eq_i32 s5, 0 // (op 1045) 
s_cselect_b32 s8, s9, s0 // (op 1045) 


s_mul_i32 s0, s4, s18 // (op 1045) 


v_lshl_add_u32 v0, s8, 4, v3 // (op 1045) 


s_sub_i32 s4, s1, s0 // (op 1045) 


s_cmp_eq_i32 s5, 0 // (op 1045) 
s_cselect_b32 s0, s21, s4 // (op 1045) 



v_lshl_add_u32 v3, s0, 4, v4 // (op 1045) 


v_mul_lo_u32 v11, v3, s38 // (op 1045) 

v_add_lshl_u32 v3, v0, v11, 2 // (op 1045) 


s_mul_i32 s1, s38, s3 // (op 1047) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 1047) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 1047) least significant
s_mul_hi_u32 s5, s38, s2 // (op 1047) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 1047) most significant: sum
s_add_u32 s1, s1, s5 // (op 1047) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 1047) 

s_mov_b32 s20, s40 // (op 1047) reg expression
s_mov_b32 s21, s41 // (op 1047) reg expression
s_mov_b32 s22, s4 // (op 1047) reg expression
s_mov_b32 s23, 131072 // (op 1047) call()




buffer_store_dwordx4 v[12:15], v3, s[20:23], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155
RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155:
s_mul_hi_i32 s0, s16, s11 // (op 815) 
v_lshrrev_b32 v0, 6, v1 // (op 815) 
s_add_i32 s1, s0, s16 // (op 815) 

v_and_b32 v3, 63, v1 // (op 815) 
s_ashr_i32 s0, s1, 31 // (op 815) 
v_lshrrev_b32 v4, 4, v3 // (op 815) 
s_cmp_eq_i32 s11, 0 // (op 815) 
s_mov_b32 s4, scc // (op 815) 
v_lshlrev_b32 v5, 2, v4 // (op 815) 

s_cmp_eq_i32 s4, 0 // (op 815) 
s_cselect_b32 s5, 0, -1 // (op 815) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 815) 


s_and_b32 s4, s12, 31 // (op 815) 
v_lshrrev_b32 v0, 2, v3 // (op 815) 

s_lshl_b32 s8, 1, s4 // (op 815) 
v_and_b32 v3, 3, v0 // (op 815) 

s_add_i32 s9, s8, s5 // (op 815) 


v_and_b32 v0, 3, v1 // (op 815) 
s_and_b32 s5, s0, s9 // (op 815) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 815) 


s_add_i32 s0, s1, s5 // (op 815) 


s_ashr_i32 s1, s0, s4 // (op 815) 


s_lshl_b32 s0, s12, 24 // (op 815) 
s_ashr_i32 s4, s0, 31 // (op 815) 

s_xor_b32 s0, s1, s4 // (op 815) 

s_sub_i32 s1, s0, s4 // (op 815) 


s_mul_hi_i32 s0, s1, s13 // (op 815) 
s_add_i32 s4, s0, s1 // (op 815) 

s_ashr_i32 s0, s4, 31 // (op 815) 
s_cmp_eq_i32 s13, 0 // (op 815) 
s_mov_b32 s5, scc // (op 815) 
s_cmp_eq_i32 s5, 0 // (op 815) 
s_cselect_b32 s8, 0, -1 // (op 815) 

s_and_b32 s5, s14, 31 // (op 815) 
s_lshl_b32 s9, 1, s5 // (op 815) 
s_add_i32 s18, s9, s8 // (op 815) 


s_and_b32 s19, s0, s18 // (op 815) 


s_add_i32 s0, s4, s19 // (op 815) 


s_ashr_i32 s4, s0, s5 // (op 815) 


s_lshl_b32 s0, s14, 24 // (op 815) 
s_ashr_i32 s5, s0, 31 // (op 815) 

s_xor_b32 s0, s4, s5 // (op 815) 

s_sub_i32 s4, s0, s5 // (op 815) 


s_mul_i32 s0, s1, s46 // (op 815) 
s_sub_u32 s5, s16, s0 // (op 815) 

s_mul_i32 s0, s4, s46 // (op 815) 
s_add_u32 s0, s0, s5 // (op 815) 

s_mul_hi_i32 s5, s16, s10 // (op 815) 
s_add_i32 s8, s5, s16 // (op 815) 

s_ashr_i32 s5, s8, 31 // (op 815) 
s_cmp_eq_i32 s10, 0 // (op 815) 
s_mov_b32 s9, scc // (op 815) 
s_cmp_eq_i32 s9, 0 // (op 815) 
s_cselect_b32 s18, 0, -1 // (op 815) 

s_and_b32 s9, s49, 31 // (op 815) 
s_lshl_b32 s19, 1, s9 // (op 815) 
s_add_i32 s24, s19, s18 // (op 815) 


s_and_b32 s18, s5, s24 // (op 815) 


s_add_i32 s5, s8, s18 // (op 815) 


s_ashr_i32 s8, s5, s9 // (op 815) 


s_lshl_b32 s5, s49, 24 // (op 815) 
s_ashr_i32 s9, s5, 31 // (op 815) 

s_xor_b32 s5, s8, s9 // (op 815) 

s_sub_i32 s8, s5, s9 // (op 815) 


s_add_u32 s18, s2, 16 // (op 815) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 815) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 815) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 815) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 4 // (op 815) 

s_mul_i32 s5, s46, s18 // (op 815) 
s_mul_i32 s9, s50, s5 // (op 815) 

s_mul_i32 s5, s8, s9 // (op 815) 

s_sub_u32 s9, s16, s5 // (op 815) 

s_mov_b32 s5, s9 // (op 815) 
s_mul_hi_i32 s24, s5, s15 // (op 815) 
s_add_i32 s25, s24, s5 // (op 815) 


s_ashr_i32 s5, s25, 31 // (op 815) 
s_cmp_eq_i32 s15, 0 // (op 815) 
s_mov_b32 s24, scc // (op 815) 
s_cmp_eq_i32 s24, 0 // (op 815) 
s_cselect_b32 s26, 0, -1 // (op 815) 

s_and_b32 s24, s48, 31 // (op 815) 
s_lshl_b32 s27, 1, s24 // (op 815) 
s_add_i32 s28, s27, s26 // (op 815) 


s_and_b32 s26, s5, s28 // (op 815) 


s_add_i32 s5, s25, s26 // (op 815) 


s_ashr_i32 s25, s5, s24 // (op 815) 


s_lshl_b32 s5, s48, 24 // (op 815) 
s_ashr_i32 s24, s5, 31 // (op 815) 

s_xor_b32 s5, s25, s24 // (op 815) 

s_sub_i32 s25, s5, s24 // (op 815) 


s_mul_i32 s5, s47, s46 // (op 815) 
s_add_u32 s26, s6, 16 // (op 815) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 815) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 815) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 815) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 815) 

s_sub_i32 s24, s26, s5 // (op 815) 

s_mul_i32 s26, s25, s24 // (op 815) 

s_sub_u32 s24, s9, s26 // (op 815) 


s_add_u32 s9, s24, s5 // (op 815) 


s_cmp_eq_i32 s8, 0 // (op 815) 
s_mov_b32 s5, scc // (op 815) 

s_cmp_eq_i32 s5, 0 // (op 815) 
s_cselect_b32 s8, s9, s0 // (op 815) 


s_mul_i32 s0, s4, s18 // (op 815) 


v_lshl_add_u32 v0, s8, 4, v4 // (op 815) 


s_sub_i32 s4, s1, s0 // (op 815) 


s_cmp_eq_i32 s5, 0 // (op 815) 
s_cselect_b32 s24, s25, s4 // (op 815) 



v_lshl_add_u32 v3, s24, 4, v5 // (op 815) 


v_mul_lo_u32 v4, v3, s42 // (op 815) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 815) 


s_mul_i32 s1, s42, s3 // (op 817) most significant: low of low * high
s_mul_i32 s4, s43, s2 // (op 817) most significant: low of high * low
s_mul_i32 s0, s42, s2 // (op 817) least significant
s_mul_hi_u32 s5, s42, s2 // (op 817) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 817) most significant: sum
s_add_u32 s1, s1, s5 // (op 817) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 817) 

s_mov_b32 s24, s44 // (op 817) reg expression
s_mov_b32 s25, s45 // (op 817) reg expression
s_mov_b32 s26, s4 // (op 817) reg expression
s_mov_b32 s27, 131072 // (op 817) call()





buffer_load_dwordx4 v[4:7], v3, s[24:27], 0 offen // (op 167) Load value



v_accvgpr_read v12, a0 // (op 164) 
v_accvgpr_read v13, a1 // (op 164) 
v_accvgpr_read v14, a2 // (op 164) 
v_accvgpr_read v15, a3 // (op 164) 
v_mul_f32 v16, v10, v12 // (op 164) 
v_mul_f32 v17, v10, v13 // (op 164) 
v_mul_f32 v18, v10, v14 // (op 164) 
v_mul_f32 v19, v10, v15 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v9, v4 // (op 168) 
v_mul_f32 v13, v9, v5 // (op 168) 
v_mul_f32 v14, v9, v6 // (op 168) 
v_mul_f32 v15, v9, v7 // (op 168) 

v_add_f32 v4, v12, v16 // (op 165) 
v_add_f32 v5, v13, v17 // (op 165) 
v_add_f32 v6, v14, v18 // (op 165) 
v_add_f32 v7, v15, v19 // (op 165) 


s_mul_hi_i32 s0, s16, s11 // (op 1062) 
v_lshrrev_b32 v0, 6, v1 // (op 1062) 
s_add_i32 s1, s0, s16 // (op 1062) 

v_and_b32 v3, 63, v1 // (op 1062) 
s_ashr_i32 s0, s1, 31 // (op 1062) 
v_lshrrev_b32 v8, 4, v3 // (op 1062) 
s_cmp_eq_i32 s11, 0 // (op 1062) 
s_mov_b32 s4, scc // (op 1062) 
v_lshlrev_b32 v11, 2, v8 // (op 1062) 

s_cmp_eq_i32 s4, 0 // (op 1062) 
s_cselect_b32 s5, 0, -1 // (op 1062) 

v_lshl_add_u32 v8, v0, 4, v11 // (op 1062) 


s_and_b32 s4, s12, 31 // (op 1062) 
v_lshrrev_b32 v0, 2, v3 // (op 1062) 

s_lshl_b32 s8, 1, s4 // (op 1062) 
v_and_b32 v3, 3, v0 // (op 1062) 

s_add_i32 s9, s8, s5 // (op 1062) 


v_and_b32 v0, 3, v1 // (op 1062) 
s_and_b32 s5, s0, s9 // (op 1062) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 1062) 


s_add_i32 s0, s1, s5 // (op 1062) 


s_ashr_i32 s1, s0, s4 // (op 1062) 


s_lshl_b32 s0, s12, 24 // (op 1062) 
s_ashr_i32 s4, s0, 31 // (op 1062) 

s_xor_b32 s0, s1, s4 // (op 1062) 

s_sub_i32 s1, s0, s4 // (op 1062) 


s_mul_hi_i32 s0, s1, s13 // (op 1062) 
s_add_i32 s4, s0, s1 // (op 1062) 

s_ashr_i32 s0, s4, 31 // (op 1062) 
s_cmp_eq_i32 s13, 0 // (op 1062) 
s_mov_b32 s5, scc // (op 1062) 
s_cmp_eq_i32 s5, 0 // (op 1062) 
s_cselect_b32 s8, 0, -1 // (op 1062) 

s_and_b32 s5, s14, 31 // (op 1062) 
s_lshl_b32 s9, 1, s5 // (op 1062) 
s_add_i32 s18, s9, s8 // (op 1062) 


s_and_b32 s19, s0, s18 // (op 1062) 


s_add_i32 s0, s4, s19 // (op 1062) 


s_ashr_i32 s4, s0, s5 // (op 1062) 


s_lshl_b32 s0, s14, 24 // (op 1062) 
s_ashr_i32 s5, s0, 31 // (op 1062) 

s_xor_b32 s0, s4, s5 // (op 1062) 

s_sub_i32 s4, s0, s5 // (op 1062) 


s_mul_i32 s0, s1, s46 // (op 1062) 
s_sub_u32 s5, s16, s0 // (op 1062) 

s_mul_i32 s0, s4, s46 // (op 1062) 
s_add_u32 s0, s0, s5 // (op 1062) 

s_mul_hi_i32 s5, s16, s10 // (op 1062) 
s_add_i32 s8, s5, s16 // (op 1062) 

s_ashr_i32 s5, s8, 31 // (op 1062) 
s_cmp_eq_i32 s10, 0 // (op 1062) 
s_mov_b32 s9, scc // (op 1062) 
s_cmp_eq_i32 s9, 0 // (op 1062) 
s_cselect_b32 s18, 0, -1 // (op 1062) 

s_and_b32 s9, s49, 31 // (op 1062) 
s_lshl_b32 s19, 1, s9 // (op 1062) 
s_add_i32 s24, s19, s18 // (op 1062) 


s_and_b32 s18, s5, s24 // (op 1062) 


s_add_i32 s5, s8, s18 // (op 1062) 


s_ashr_i32 s8, s5, s9 // (op 1062) 


s_lshl_b32 s5, s49, 24 // (op 1062) 
s_ashr_i32 s9, s5, 31 // (op 1062) 

s_xor_b32 s5, s8, s9 // (op 1062) 

s_sub_i32 s8, s5, s9 // (op 1062) 


s_add_u32 s18, s2, 16 // (op 1062) Start of Int64 add, locking SCC
s_addc_u32 s19, s3, 0 // (op 1062) End of Int64 add, unlocking SCC
s_sub_u32 s24, s18, 1 // (op 1062) least significant half; sets scc
s_subb_u32 s25, s19, 0 // (op 1062) most significant half; uses scc

s_ashr_i64 s[18:19], s[24:25], 4 // (op 1062) 

s_mul_i32 s5, s46, s18 // (op 1062) 
s_mul_i32 s9, s50, s5 // (op 1062) 

s_mul_i32 s5, s8, s9 // (op 1062) 

s_sub_u32 s9, s16, s5 // (op 1062) 

s_mov_b32 s5, s9 // (op 1062) 
s_mul_hi_i32 s24, s5, s15 // (op 1062) 
s_add_i32 s25, s24, s5 // (op 1062) 


s_ashr_i32 s5, s25, 31 // (op 1062) 
s_cmp_eq_i32 s15, 0 // (op 1062) 
s_mov_b32 s24, scc // (op 1062) 
s_cmp_eq_i32 s24, 0 // (op 1062) 
s_cselect_b32 s26, 0, -1 // (op 1062) 

s_and_b32 s24, s48, 31 // (op 1062) 
s_lshl_b32 s27, 1, s24 // (op 1062) 
s_add_i32 s28, s27, s26 // (op 1062) 


s_and_b32 s26, s5, s28 // (op 1062) 


s_add_i32 s5, s25, s26 // (op 1062) 


s_ashr_i32 s25, s5, s24 // (op 1062) 


s_lshl_b32 s5, s48, 24 // (op 1062) 
s_ashr_i32 s24, s5, 31 // (op 1062) 

s_xor_b32 s5, s25, s24 // (op 1062) 

s_sub_i32 s25, s5, s24 // (op 1062) 


s_mul_i32 s5, s47, s46 // (op 1062) 
s_add_u32 s26, s6, 16 // (op 1062) Start of Int64 add, locking SCC
s_addc_u32 s27, s7, 0 // (op 1062) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 1062) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 1062) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 1062) 

s_sub_i32 s24, s26, s5 // (op 1062) 

s_mul_i32 s26, s25, s24 // (op 1062) 

s_sub_u32 s24, s9, s26 // (op 1062) 


s_add_u32 s9, s24, s5 // (op 1062) 


s_cmp_eq_i32 s8, 0 // (op 1062) 
s_mov_b32 s5, scc // (op 1062) 

s_cmp_eq_i32 s5, 0 // (op 1062) 
s_cselect_b32 s8, s9, s0 // (op 1062) 


s_mul_i32 s0, s4, s18 // (op 1062) 


v_lshl_add_u32 v0, s8, 4, v8 // (op 1062) 


s_sub_i32 s4, s1, s0 // (op 1062) 


s_cmp_eq_i32 s5, 0 // (op 1062) 
s_cselect_b32 s24, s25, s4 // (op 1062) 



v_lshl_add_u32 v3, s24, 4, v11 // (op 1062) 


v_mul_lo_u32 v8, v3, s38 // (op 1062) 

v_add_lshl_u32 v3, v0, v8, 2 // (op 1062) 


s_mul_i32 s1, s38, s3 // (op 1064) most significant: low of low * high
s_mul_i32 s4, s39, s2 // (op 1064) most significant: low of high * low
s_mul_i32 s0, s38, s2 // (op 1064) least significant
s_mul_hi_u32 s5, s38, s2 // (op 1064) most significant: high of low * low
s_add_u32 s1, s1, s4 // (op 1064) most significant: sum
s_add_u32 s1, s1, s5 // (op 1064) most significant: sum


s_lshl_b64 s[4:5], s[0:1], 2 // (op 1064) 

s_mov_b32 s20, s40 // (op 1064) reg expression
s_mov_b32 s21, s41 // (op 1064) reg expression
s_mov_b32 s22, s4 // (op 1064) reg expression
s_mov_b32 s23, 131072 // (op 1064) call()




buffer_store_dwordx4 v[4:7], v3, s[20:23], 0 offen // (op 166) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155:
















s_endpgm  // End of RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.LRR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end:
.size RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_, .LRR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end-RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 14848 // lds bytes
  .amdhsa_accum_offset 96
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
    DataTypeB: B8
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
  - .name: RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 14848
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 96
    .agpr_count: 4
    .max_flat_workgroup_size: 64
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

