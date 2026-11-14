// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_,@function
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 63, v1 // (op 873) 
v_and_b32 v3, 48, v1 // (op 873) 
v_sub_u32 v4, v0, v3 // (op 873) 

v_lshrrev_b32 v3, 4, v0 // (op 873) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 873) 


v_lshrrev_b32 v3, 6, v1 // (op 873) 
v_lshlrev_b32 v4, 4, v3 // (op 873) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 873) 


v_mul_lo_u32 v0, v3, 6 // (op 873) 
v_lshrrev_b32 v4, 3, v0 // (op 873) 

v_lshrrev_b32 v0, 4, v3 // (op 873) 

v_lshlrev_b32 v3, 2, v0 // (op 873) 

v_add_u32 v0, v4, v3 // (op 873) 


v_and_b32 v3, 48, v1 // (op 925) 
v_and_b32 v4, 63, v1 // (op 925) 
v_sub_u32 v5, v4, v3 // (op 925) 

v_lshrrev_b32 v4, 1, v5 // (op 925) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 925) 


v_and_b32 v3, 1, v5 // (op 925) 

v_lshlrev_b32 v4, 3, v3 // (op 925) 

v_add_u32 v3, v6, v4 // (op 925) 


v_lshrrev_b32 v4, 6, v1 // (op 978) 
v_and_b32 v5, 15, v1 // (op 978) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 978) 


v_and_b32 v4, 48, v1 // (op 978) 
v_add_u32 v5, v4, v6 // (op 978) 


v_and_b32 v4, 48, v1 // (op 1031) 
v_and_b32 v6, 15, v1 // (op 1031) 
v_add_u32 v7, v4, v6 // (op 1031) 


v_and_b32 v4, 63, v1 // (op 1089) 
v_and_b32 v6, 48, v1 // (op 1089) 
v_sub_u32 v8, v4, v6 // (op 1089) 

v_lshrrev_b32 v6, 4, v4 // (op 1089) 

v_lshl_add_u32 v4, v6, 5, v8 // (op 1089) 


v_lshrrev_b32 v6, 6, v1 // (op 1089) 
v_lshlrev_b32 v8, 4, v6 // (op 1089) 

v_lshl_add_u32 v6, v4, 4, v8 // (op 1089) 


v_mul_lo_u32 v4, v6, 6 // (op 1089) 
v_lshrrev_b32 v8, 3, v4 // (op 1089) 

v_lshrrev_b32 v4, 4, v6 // (op 1089) 

v_lshlrev_b32 v6, 2, v4 // (op 1089) 

v_add_u32 v4, v8, v6 // (op 1089) 


v_and_b32 v6, 48, v1 // (op 1128) 
v_and_b32 v8, 63, v1 // (op 1128) 
v_sub_u32 v9, v8, v6 // (op 1128) 

v_lshrrev_b32 v8, 1, v9 // (op 1128) 
v_add_lshl_u32 v10, v6, v8, 4 // (op 1128) 


v_and_b32 v6, 1, v9 // (op 1128) 

v_lshlrev_b32 v8, 3, v6 // (op 1128) 

v_add_u32 v6, v10, v8 // (op 1128) 


v_lshrrev_b32 v8, 6, v1 // (op 1152) 
v_and_b32 v9, 15, v1 // (op 1152) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 1152) 


v_and_b32 v8, 48, v1 // (op 1152) 
v_add_u32 v9, v8, v10 // (op 1152) 


v_and_b32 v8, 48, v1 // (op 1176) 
v_and_b32 v10, 15, v1 // (op 1176) 
v_add_u32 v11, v8, v10 // (op 1176) 


v_mov_b32 v8, s6 // (op 2) Move value
v_mov_b32 v10, s7 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_mul_hi_i32 s0, s16, s9 // (op 583) 
v_lshlrev_b32 v12, 2, v1 // (op 583) 
s_add_i32 s1, s0, s16 // (op 583) 

s_ashr_i32 s0, s1, 31 // (op 583) 
s_cmp_eq_i32 s9, 0 // (op 583) 
s_mov_b32 s6, scc // (op 583) 
s_cmp_eq_i32 s6, 0 // (op 583) 
s_cselect_b32 s7, 0, -1 // (op 583) 

s_and_b32 s6, s10, 31 // (op 583) 
s_lshl_b32 s18, 1, s6 // (op 583) 
s_add_i32 s19, s18, s7 // (op 583) 


s_and_b32 s7, s0, s19 // (op 583) 


s_add_i32 s0, s1, s7 // (op 583) 


s_ashr_i32 s1, s0, s6 // (op 583) 


s_lshl_b32 s0, s10, 24 // (op 583) 
s_ashr_i32 s6, s0, 31 // (op 583) 

s_xor_b32 s0, s1, s6 // (op 583) 

s_sub_i32 s1, s0, s6 // (op 583) 


s_mul_hi_i32 s0, s1, s11 // (op 583) 
s_add_i32 s6, s0, s1 // (op 583) 

s_ashr_i32 s0, s6, 31 // (op 583) 
s_cmp_eq_i32 s11, 0 // (op 583) 
s_mov_b32 s7, scc // (op 583) 
s_cmp_eq_i32 s7, 0 // (op 583) 
s_cselect_b32 s18, 0, -1 // (op 583) 

s_and_b32 s7, s12, 31 // (op 583) 
s_lshl_b32 s19, 1, s7 // (op 583) 
s_add_i32 s51, s19, s18 // (op 583) 


s_and_b32 s18, s0, s51 // (op 583) 


s_add_i32 s0, s6, s18 // (op 583) 


s_ashr_i32 s6, s0, s7 // (op 583) 


s_lshl_b32 s0, s12, 24 // (op 583) 
s_ashr_i32 s7, s0, 31 // (op 583) 

s_xor_b32 s0, s6, s7 // (op 583) 

s_sub_i32 s6, s0, s7 // (op 583) 


s_mul_i32 s0, s1, s14 // (op 583) 

s_sub_u32 s1, s16, s0 // (op 583) 

s_mul_i32 s0, s6, s14 // (op 583) 
s_add_u32 s0, s0, s1 // (op 583) 


s_mul_hi_i32 s1, s16, s8 // (op 583) 
s_add_i32 s6, s1, s16 // (op 583) 

s_ashr_i32 s1, s6, 31 // (op 583) 
s_cmp_eq_i32 s8, 0 // (op 583) 
s_mov_b32 s7, scc // (op 583) 
s_cmp_eq_i32 s7, 0 // (op 583) 
s_cselect_b32 s18, 0, -1 // (op 583) 

s_and_b32 s7, s15, 31 // (op 583) 
s_lshl_b32 s19, 1, s7 // (op 583) 
s_add_i32 s51, s19, s18 // (op 583) 


s_and_b32 s18, s1, s51 // (op 583) 


s_add_i32 s1, s6, s18 // (op 583) 


s_ashr_i32 s6, s1, s7 // (op 583) 


s_lshl_b32 s1, s15, 24 // (op 583) 
s_ashr_i32 s7, s1, 31 // (op 583) 

s_xor_b32 s1, s6, s7 // (op 583) 

s_sub_i32 s6, s1, s7 // (op 583) 


s_add_u32 s18, s4, 16 // (op 583) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 583) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 583) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 583) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 583) 

s_mul_i32 s1, s14, s18 // (op 583) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 583) 

s_mul_i32 s1, s6, s7 // (op 583) 

s_sub_u32 s7, s16, s1 // (op 583) 

s_mov_b32 s1, s7 // (op 583) 
s_mul_hi_i32 s18, s1, s13 // (op 583) 
s_add_i32 s19, s18, s1 // (op 583) 


s_ashr_i32 s1, s19, 31 // (op 583) 
s_cmp_eq_i32 s13, 0 // (op 583) 
s_mov_b32 s18, scc // (op 583) 
s_cmp_eq_i32 s18, 0 // (op 583) 
s_cselect_b32 s51, 0, -1 // (op 583) 

s_and_b32 s18, s49, 31 // (op 583) 
s_lshl_b32 s52, 1, s18 // (op 583) 
s_add_i32 s53, s52, s51 // (op 583) 


s_and_b32 s51, s1, s53 // (op 583) 


s_add_i32 s1, s19, s51 // (op 583) 


s_ashr_i32 s19, s1, s18 // (op 583) 


s_lshl_b32 s1, s49, 24 // (op 583) 
s_ashr_i32 s18, s1, 31 // (op 583) 

s_xor_b32 s1, s19, s18 // (op 583) 

s_sub_i32 s19, s1, s18 // (op 583) 


s_mul_i32 s1, s48, s14 // (op 583) 
s_add_u32 s52, s20, 16 // (op 583) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 583) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 583) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 583) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 583) 

s_sub_i32 s18, s52, s1 // (op 583) 

s_mul_i32 s51, s19, s18 // (op 583) 


s_sub_u32 s18, s7, s51 // (op 583) 


s_add_u32 s7, s18, s1 // (op 583) 


s_cmp_eq_i32 s6, 0 // (op 583) 
s_mov_b32 s1, scc // (op 583) 

s_cmp_eq_i32 s1, 0 // (op 583) 
s_cselect_b32 s6, s7, s0 // (op 583) 



s_lshl_b32 s7, s6, 4 // (op 583) 

s_mov_b32 s1, 0 // (op 583) convert
s_mov_b32 s0, s7 // (op 583) convert

v_mul_lo_u32 v15, v12, s3 // (op 583) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 583) most significant: high of low * low
v_mul_lo_u32 v14, v12, s2 // (op 583) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 583) most significant: sum


v_mov_b32 v12, s0 // (op 583) 
v_mov_b32 v13, s1 // (op 583) 
v_add_co_u32 v16, s[6:7], v12, v14 // (op 583) least significant half
v_addc_co_u32 v17, s[6:7], v13, v15, s[6:7] // (op 583) most significant half





v_mul_lo_u32 v13, v17, 6 // (op 583) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 583) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 583) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 583) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 583) 

v_mov_b32 v12, v14 // (op 583) convert

s_mul_i32 s1, s25, 6 // (op 585) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 585) least significant
s_mul_hi_u32 s6, s24, 6 // (op 585) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 585) most significant: sum

s_mul_i32 s7, s2, s1 // (op 585) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 585) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 585) least significant
s_mul_hi_u32 s19, s2, s0 // (op 585) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 585) most significant: sum
s_add_u32 s7, s7, s19 // (op 585) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 585) 

s_mov_b32 s52, s22 // (op 585) reg expression
s_mov_b32 s53, s23 // (op 585) reg expression
s_mov_b32 s54, s0 // (op 585) reg expression
s_mov_b32 s55, 131072 // (op 585) call()

s_mul_hi_i32 s51, s16, s9 // (op 602) 
v_lshlrev_b32 v13, 2, v1 // (op 602) 
s_add_i32 s0, s51, s16 // (op 602) 

s_ashr_i32 s1, s0, 31 // (op 602) 
s_cmp_eq_i32 s9, 0 // (op 602) 
s_mov_b32 s51, scc // (op 602) 
s_cmp_eq_i32 s51, 0 // (op 602) 
s_cselect_b32 s6, 0, -1 // (op 602) 

s_and_b32 s7, s10, 31 // (op 602) 
s_lshl_b32 s51, 1, s7 // (op 602) 
s_add_i32 s18, s51, s6 // (op 602) 


s_and_b32 s6, s1, s18 // (op 602) 


s_add_i32 s1, s0, s6 // (op 602) 


s_ashr_i32 s0, s1, s7 // (op 602) 


s_lshl_b32 s1, s10, 24 // (op 602) 
s_ashr_i32 s51, s1, 31 // (op 602) 

s_xor_b32 s1, s0, s51 // (op 602) 

s_sub_i32 s0, s1, s51 // (op 602) 


s_mul_hi_i32 s1, s0, s11 // (op 602) 
s_add_i32 s51, s1, s0 // (op 602) 

s_ashr_i32 s1, s51, 31 // (op 602) 
s_cmp_eq_i32 s11, 0 // (op 602) 
s_mov_b32 s6, scc // (op 602) 
s_cmp_eq_i32 s6, 0 // (op 602) 
s_cselect_b32 s7, 0, -1 // (op 602) 

s_and_b32 s6, s12, 31 // (op 602) 
s_lshl_b32 s18, 1, s6 // (op 602) 
s_add_i32 s19, s18, s7 // (op 602) 


s_and_b32 s7, s1, s19 // (op 602) 


s_add_i32 s1, s51, s7 // (op 602) 


s_ashr_i32 s7, s1, s6 // (op 602) 


s_lshl_b32 s1, s12, 24 // (op 602) 
s_ashr_i32 s6, s1, 31 // (op 602) 

s_xor_b32 s1, s7, s6 // (op 602) 

s_sub_i32 s7, s1, s6 // (op 602) 


s_mul_i32 s1, s0, s14 // (op 602) 

s_sub_u32 s0, s16, s1 // (op 602) 

s_mul_i32 s1, s7, s14 // (op 602) 
s_add_u32 s1, s1, s0 // (op 602) 


s_mul_hi_i32 s0, s16, s8 // (op 602) 
s_add_i32 s51, s0, s16 // (op 602) 

s_ashr_i32 s0, s51, 31 // (op 602) 
s_cmp_eq_i32 s8, 0 // (op 602) 
s_mov_b32 s6, scc // (op 602) 
s_cmp_eq_i32 s6, 0 // (op 602) 
s_cselect_b32 s7, 0, -1 // (op 602) 

s_and_b32 s6, s15, 31 // (op 602) 
s_lshl_b32 s18, 1, s6 // (op 602) 
s_add_i32 s19, s18, s7 // (op 602) 


s_and_b32 s7, s0, s19 // (op 602) 


s_add_i32 s0, s51, s7 // (op 602) 


s_ashr_i32 s7, s0, s6 // (op 602) 


s_lshl_b32 s0, s15, 24 // (op 602) 
s_ashr_i32 s6, s0, 31 // (op 602) 

s_xor_b32 s0, s7, s6 // (op 602) 

s_sub_i32 s7, s0, s6 // (op 602) 


s_add_u32 s18, s4, 16 // (op 602) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 602) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 602) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 602) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 602) 

s_mul_i32 s0, s14, s18 // (op 602) 

s_mul_i32 s6, s50, s0 // (op 602) 

s_mul_i32 s0, s7, s6 // (op 602) 

s_sub_u32 s6, s16, s0 // (op 602) 

s_mov_b32 s0, s6 // (op 602) 
s_mul_hi_i32 s51, s0, s13 // (op 602) 
s_add_i32 s18, s51, s0 // (op 602) 


s_ashr_i32 s0, s18, 31 // (op 602) 
s_cmp_eq_i32 s13, 0 // (op 602) 
s_mov_b32 s19, scc // (op 602) 
s_cmp_eq_i32 s19, 0 // (op 602) 
s_cselect_b32 s51, 0, -1 // (op 602) 

s_and_b32 s19, s49, 31 // (op 602) 
s_lshl_b32 s56, 1, s19 // (op 602) 
s_add_i32 s57, s56, s51 // (op 602) 


s_and_b32 s51, s0, s57 // (op 602) 


s_add_i32 s0, s18, s51 // (op 602) 


s_ashr_i32 s18, s0, s19 // (op 602) 


s_lshl_b32 s0, s49, 24 // (op 602) 
s_ashr_i32 s19, s0, 31 // (op 602) 

s_xor_b32 s0, s18, s19 // (op 602) 

s_sub_i32 s18, s0, s19 // (op 602) 


s_mul_i32 s0, s48, s14 // (op 602) 
s_add_u32 s56, s20, 16 // (op 602) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 602) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 602) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 602) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 602) 

s_sub_i32 s19, s56, s0 // (op 602) 

s_mul_i32 s51, s18, s19 // (op 602) 


s_sub_u32 s18, s6, s51 // (op 602) 


s_add_u32 s6, s18, s0 // (op 602) 


s_cmp_eq_i32 s7, 0 // (op 602) 
s_mov_b32 s0, scc // (op 602) 

s_cmp_eq_i32 s0, 0 // (op 602) 
s_cselect_b32 s7, s6, s1 // (op 602) 



s_lshl_b32 s6, s7, 4 // (op 602) 

s_mov_b32 s1, 0 // (op 602) convert
s_mov_b32 s0, s6 // (op 602) convert

v_mul_lo_u32 v15, v13, s3 // (op 602) most significant: low of low * high
v_mul_hi_u32 v16, v13, s2 // (op 602) most significant: high of low * low
v_mul_lo_u32 v14, v13, s2 // (op 602) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 602) most significant: sum


v_mov_b32 v13, s0 // (op 602) 
v_mov_b32 v16, s1 // (op 602) 
v_add_co_u32 v18, s[6:7], v13, v14 // (op 602) least significant half
v_addc_co_u32 v19, s[6:7], v16, v15, s[6:7] // (op 602) most significant half





v_mul_lo_u32 v17, v19, 6 // (op 602) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 602) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 602) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 602) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 602) 

v_mov_b32 v13, v14 // (op 602) convert

s_mul_i32 s1, s25, 6 // (op 604) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 604) least significant
s_mul_hi_u32 s51, s24, 6 // (op 604) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 604) most significant: sum

s_mul_i32 s7, s2, s1 // (op 604) most significant: low of low * high
s_mul_i32 s51, s3, s0 // (op 604) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 604) least significant
s_mul_hi_u32 s18, s2, s0 // (op 604) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 604) most significant: sum
s_add_u32 s7, s7, s18 // (op 604) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 604) 

s_mov_b32 s52, s22 // (op 604) reg expression
s_mov_b32 s53, s23 // (op 604) reg expression
s_mov_b32 s54, s0 // (op 604) reg expression
s_mov_b32 s55, 131072 // (op 604) call()

s_mul_hi_i32 s51, s16, s9 // (op 647) 
s_add_i32 s0, s51, s16 // (op 647) 

s_ashr_i32 s1, s0, 31 // (op 647) 
s_cmp_eq_i32 s9, 0 // (op 647) 
s_mov_b32 s51, scc // (op 647) 
s_cmp_eq_i32 s51, 0 // (op 647) 
s_cselect_b32 s6, 0, -1 // (op 647) 

s_and_b32 s7, s10, 31 // (op 647) 
s_lshl_b32 s51, 1, s7 // (op 647) 
s_add_i32 s18, s51, s6 // (op 647) 


s_and_b32 s6, s1, s18 // (op 647) 


s_add_i32 s1, s0, s6 // (op 647) 


s_ashr_i32 s0, s1, s7 // (op 647) 


s_lshl_b32 s1, s10, 24 // (op 647) 
s_ashr_i32 s51, s1, 31 // (op 647) 

s_xor_b32 s1, s0, s51 // (op 647) 

s_sub_i32 s0, s1, s51 // (op 647) 


s_mul_hi_i32 s1, s0, s11 // (op 647) 
s_add_i32 s51, s1, s0 // (op 647) 

s_ashr_i32 s1, s51, 31 // (op 647) 
s_cmp_eq_i32 s11, 0 // (op 647) 
s_mov_b32 s6, scc // (op 647) 
s_cmp_eq_i32 s6, 0 // (op 647) 
s_cselect_b32 s7, 0, -1 // (op 647) 

s_and_b32 s6, s12, 31 // (op 647) 
s_lshl_b32 s18, 1, s6 // (op 647) 
s_add_i32 s19, s18, s7 // (op 647) 


s_and_b32 s7, s1, s19 // (op 647) 


s_add_i32 s1, s51, s7 // (op 647) 


s_ashr_i32 s7, s1, s6 // (op 647) 


s_lshl_b32 s1, s12, 24 // (op 647) 
s_ashr_i32 s6, s1, 31 // (op 647) 

s_xor_b32 s1, s7, s6 // (op 647) 

s_sub_i32 s7, s1, s6 // (op 647) 


s_add_u32 s18, s4, 16 // (op 647) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 647) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 647) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 647) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 647) 

s_mul_i32 s1, s7, s18 // (op 647) 

s_sub_i32 s51, s0, s1 // (op 647) 


s_mul_hi_i32 s0, s16, s8 // (op 647) 
s_add_i32 s1, s0, s16 // (op 647) 

s_ashr_i32 s0, s1, 31 // (op 647) 
s_cmp_eq_i32 s8, 0 // (op 647) 
s_mov_b32 s6, scc // (op 647) 
s_cmp_eq_i32 s6, 0 // (op 647) 
s_cselect_b32 s7, 0, -1 // (op 647) 

s_and_b32 s6, s15, 31 // (op 647) 
s_lshl_b32 s56, 1, s6 // (op 647) 
s_add_i32 s57, s56, s7 // (op 647) 


s_and_b32 s7, s0, s57 // (op 647) 


s_add_i32 s0, s1, s7 // (op 647) 


s_ashr_i32 s1, s0, s6 // (op 647) 


s_lshl_b32 s0, s15, 24 // (op 647) 
s_ashr_i32 s6, s0, 31 // (op 647) 

s_xor_b32 s0, s1, s6 // (op 647) 

s_sub_i32 s1, s0, s6 // (op 647) 


s_mul_i32 s0, s14, s18 // (op 647) 

s_mul_i32 s6, s50, s0 // (op 647) 

s_mul_i32 s0, s1, s6 // (op 647) 

s_sub_u32 s6, s16, s0 // (op 647) 

s_mul_hi_i32 s0, s6, s13 // (op 647) 
s_add_i32 s7, s0, s6 // (op 647) 


s_ashr_i32 s0, s7, 31 // (op 647) 
s_cmp_eq_i32 s13, 0 // (op 647) 
s_mov_b32 s6, scc // (op 647) 
s_cmp_eq_i32 s6, 0 // (op 647) 
s_cselect_b32 s18, 0, -1 // (op 647) 

s_and_b32 s6, s49, 31 // (op 647) 
s_lshl_b32 s19, 1, s6 // (op 647) 
s_add_i32 s56, s19, s18 // (op 647) 


s_and_b32 s18, s0, s56 // (op 647) 


s_add_i32 s0, s7, s18 // (op 647) 


s_ashr_i32 s7, s0, s6 // (op 647) 


s_lshl_b32 s0, s49, 24 // (op 647) 
s_ashr_i32 s6, s0, 31 // (op 647) 

s_xor_b32 s0, s7, s6 // (op 647) 

s_sub_i32 s7, s0, s6 // (op 647) 


s_cmp_eq_i32 s1, 0 // (op 647) 
s_mov_b32 s0, scc // (op 647) 

s_cmp_eq_i32 s0, 0 // (op 647) 
s_cselect_b32 s1, s7, s51 // (op 647) 



s_lshl_b32 s0, s1, 4 // (op 647) 

s_mov_b32 s7, 0 // (op 647) convert
s_mov_b32 s6, s0 // (op 647) convert

v_mul_lo_u32 v15, v1, s27 // (op 647) most significant: low of low * high
v_mul_hi_u32 v16, v1, s26 // (op 647) most significant: high of low * low
v_mul_lo_u32 v14, v1, s26 // (op 647) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 647) most significant: sum

v_mov_b32 v16, s6 // (op 647) 
v_mov_b32 v17, s7 // (op 647) 
v_add_co_u32 v14, s[0:1], v14, v16 // (op 647) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 647) most significant half




s_mul_i32 s1, s26, s25 // (op 649) most significant: low of low * high
s_mul_i32 s51, s27, s24 // (op 649) most significant: low of high * low
s_mul_i32 s0, s26, s24 // (op 649) least significant
s_mul_hi_u32 s6, s26, s24 // (op 649) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 649) most significant: sum
s_add_u32 s1, s1, s6 // (op 649) most significant: sum


s_mov_b32 s56, s28 // (op 649) reg expression
s_mov_b32 s57, s29 // (op 649) reg expression
s_mov_b32 s58, s0 // (op 649) reg expression
s_mov_b32 s59, 131072 // (op 649) call()

s_mul_hi_i32 s51, s16, s9 // (op 666) 
s_add_i32 s0, s51, s16 // (op 666) 

s_ashr_i32 s1, s0, 31 // (op 666) 
s_cmp_eq_i32 s9, 0 // (op 666) 
s_mov_b32 s51, scc // (op 666) 
s_cmp_eq_i32 s51, 0 // (op 666) 
s_cselect_b32 s6, 0, -1 // (op 666) 

s_and_b32 s7, s10, 31 // (op 666) 
s_lshl_b32 s51, 1, s7 // (op 666) 
s_add_i32 s18, s51, s6 // (op 666) 


s_and_b32 s6, s1, s18 // (op 666) 


s_add_i32 s1, s0, s6 // (op 666) 


s_ashr_i32 s0, s1, s7 // (op 666) 


s_lshl_b32 s1, s10, 24 // (op 666) 
s_ashr_i32 s51, s1, 31 // (op 666) 

s_xor_b32 s1, s0, s51 // (op 666) 

s_sub_i32 s0, s1, s51 // (op 666) 


s_mul_hi_i32 s1, s0, s11 // (op 666) 
s_add_i32 s51, s1, s0 // (op 666) 

s_ashr_i32 s1, s51, 31 // (op 666) 
s_cmp_eq_i32 s11, 0 // (op 666) 
s_mov_b32 s6, scc // (op 666) 
s_cmp_eq_i32 s6, 0 // (op 666) 
s_cselect_b32 s7, 0, -1 // (op 666) 

s_and_b32 s6, s12, 31 // (op 666) 
s_lshl_b32 s18, 1, s6 // (op 666) 
s_add_i32 s19, s18, s7 // (op 666) 


s_and_b32 s7, s1, s19 // (op 666) 


s_add_i32 s1, s51, s7 // (op 666) 


s_ashr_i32 s7, s1, s6 // (op 666) 


s_lshl_b32 s1, s12, 24 // (op 666) 
s_ashr_i32 s6, s1, 31 // (op 666) 

s_xor_b32 s1, s7, s6 // (op 666) 

s_sub_i32 s7, s1, s6 // (op 666) 


s_add_u32 s18, s4, 16 // (op 666) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 666) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 666) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 666) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 666) 

s_mul_i32 s1, s7, s18 // (op 666) 

s_sub_i32 s51, s0, s1 // (op 666) 


s_mul_hi_i32 s0, s16, s8 // (op 666) 
s_add_i32 s1, s0, s16 // (op 666) 

s_ashr_i32 s0, s1, 31 // (op 666) 
s_cmp_eq_i32 s8, 0 // (op 666) 
s_mov_b32 s6, scc // (op 666) 
s_cmp_eq_i32 s6, 0 // (op 666) 
s_cselect_b32 s7, 0, -1 // (op 666) 

s_and_b32 s6, s15, 31 // (op 666) 
s_lshl_b32 s60, 1, s6 // (op 666) 
s_add_i32 s61, s60, s7 // (op 666) 


s_and_b32 s7, s0, s61 // (op 666) 


s_add_i32 s0, s1, s7 // (op 666) 


s_ashr_i32 s1, s0, s6 // (op 666) 


s_lshl_b32 s0, s15, 24 // (op 666) 
s_ashr_i32 s6, s0, 31 // (op 666) 

s_xor_b32 s0, s1, s6 // (op 666) 

s_sub_i32 s1, s0, s6 // (op 666) 


s_mul_i32 s0, s14, s18 // (op 666) 

s_mul_i32 s6, s50, s0 // (op 666) 

s_mul_i32 s0, s1, s6 // (op 666) 

s_sub_u32 s6, s16, s0 // (op 666) 

s_mul_hi_i32 s0, s6, s13 // (op 666) 
s_add_i32 s7, s0, s6 // (op 666) 


s_ashr_i32 s0, s7, 31 // (op 666) 
s_cmp_eq_i32 s13, 0 // (op 666) 
s_mov_b32 s6, scc // (op 666) 
s_cmp_eq_i32 s6, 0 // (op 666) 
s_cselect_b32 s18, 0, -1 // (op 666) 

s_and_b32 s6, s49, 31 // (op 666) 
s_lshl_b32 s19, 1, s6 // (op 666) 
s_add_i32 s60, s19, s18 // (op 666) 


s_and_b32 s18, s0, s60 // (op 666) 


s_add_i32 s0, s7, s18 // (op 666) 


s_ashr_i32 s7, s0, s6 // (op 666) 


s_lshl_b32 s0, s49, 24 // (op 666) 
s_ashr_i32 s6, s0, 31 // (op 666) 

s_xor_b32 s0, s7, s6 // (op 666) 

s_sub_i32 s7, s0, s6 // (op 666) 


s_cmp_eq_i32 s1, 0 // (op 666) 
s_mov_b32 s0, scc // (op 666) 

s_cmp_eq_i32 s0, 0 // (op 666) 
s_cselect_b32 s1, s7, s51 // (op 666) 



s_lshl_b32 s0, s1, 4 // (op 666) 

s_mov_b32 s7, 0 // (op 666) convert
s_mov_b32 s6, s0 // (op 666) convert

v_mul_lo_u32 v17, v1, s27 // (op 666) most significant: low of low * high
v_mul_hi_u32 v18, v1, s26 // (op 666) most significant: high of low * low
v_mul_lo_u32 v16, v1, s26 // (op 666) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 666) most significant: sum

v_mov_b32 v18, s6 // (op 666) 
v_mov_b32 v19, s7 // (op 666) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 666) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 666) most significant half




s_mul_i32 s1, s26, s25 // (op 668) most significant: low of low * high
s_mul_i32 s51, s27, s24 // (op 668) most significant: low of high * low
s_mul_i32 s0, s26, s24 // (op 668) least significant
s_mul_hi_u32 s6, s26, s24 // (op 668) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 668) most significant: sum
s_add_u32 s1, s1, s6 // (op 668) most significant: sum


s_mov_b32 s56, s28 // (op 668) reg expression
s_mov_b32 s57, s29 // (op 668) reg expression
s_mov_b32 s58, s0 // (op 668) reg expression
s_mov_b32 s59, 131072 // (op 668) call()

s_mul_hi_i32 s51, s16, s9 // (op 711) 
v_and_b32 v18, -16, v1 // (op 711) 
s_add_i32 s0, s51, s16 // (op 711) 

v_sub_u32 v19, v1, v18 // (op 711) 

s_ashr_i32 s1, s0, 31 // (op 711) 
v_lshrrev_b32 v18, 4, v1 // (op 711) 
s_cmp_eq_i32 s9, 0 // (op 711) 
s_mov_b32 s51, scc // (op 711) 
v_lshlrev_b32 v20, 1, v18 // (op 711) 

s_cmp_eq_i32 s51, 0 // (op 711) 
s_cselect_b32 s6, 0, -1 // (op 711) 

s_and_b32 s7, s10, 31 // (op 711) 
s_lshl_b32 s51, 1, s7 // (op 711) 
s_add_i32 s18, s51, s6 // (op 711) 


s_and_b32 s6, s1, s18 // (op 711) 


s_add_i32 s1, s0, s6 // (op 711) 


s_ashr_i32 s0, s1, s7 // (op 711) 


s_lshl_b32 s1, s10, 24 // (op 711) 
s_ashr_i32 s51, s1, 31 // (op 711) 

s_xor_b32 s1, s0, s51 // (op 711) 

s_sub_i32 s0, s1, s51 // (op 711) 


s_mul_hi_i32 s1, s0, s11 // (op 711) 
s_add_i32 s51, s1, s0 // (op 711) 

s_ashr_i32 s1, s51, 31 // (op 711) 
s_cmp_eq_i32 s11, 0 // (op 711) 
s_mov_b32 s6, scc // (op 711) 
s_cmp_eq_i32 s6, 0 // (op 711) 
s_cselect_b32 s7, 0, -1 // (op 711) 

s_and_b32 s6, s12, 31 // (op 711) 
s_lshl_b32 s18, 1, s6 // (op 711) 
s_add_i32 s19, s18, s7 // (op 711) 


s_and_b32 s7, s1, s19 // (op 711) 


s_add_i32 s1, s51, s7 // (op 711) 


s_ashr_i32 s7, s1, s6 // (op 711) 


s_lshl_b32 s1, s12, 24 // (op 711) 
s_ashr_i32 s6, s1, 31 // (op 711) 

s_xor_b32 s1, s7, s6 // (op 711) 

s_sub_i32 s7, s1, s6 // (op 711) 


s_mul_i32 s1, s0, s14 // (op 711) 

s_sub_u32 s0, s16, s1 // (op 711) 

s_mul_i32 s1, s7, s14 // (op 711) 
s_add_u32 s1, s1, s0 // (op 711) 


s_mul_hi_i32 s0, s16, s8 // (op 711) 
s_add_i32 s51, s0, s16 // (op 711) 

s_ashr_i32 s0, s51, 31 // (op 711) 
s_cmp_eq_i32 s8, 0 // (op 711) 
s_mov_b32 s6, scc // (op 711) 
s_cmp_eq_i32 s6, 0 // (op 711) 
s_cselect_b32 s7, 0, -1 // (op 711) 

s_and_b32 s6, s15, 31 // (op 711) 
s_lshl_b32 s18, 1, s6 // (op 711) 
s_add_i32 s19, s18, s7 // (op 711) 


s_and_b32 s7, s0, s19 // (op 711) 


s_add_i32 s0, s51, s7 // (op 711) 


s_ashr_i32 s7, s0, s6 // (op 711) 


s_lshl_b32 s0, s15, 24 // (op 711) 
s_ashr_i32 s6, s0, 31 // (op 711) 

s_xor_b32 s0, s7, s6 // (op 711) 

s_sub_i32 s7, s0, s6 // (op 711) 


s_add_u32 s18, s4, 16 // (op 711) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 711) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 711) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 711) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 711) 

s_mul_i32 s0, s14, s18 // (op 711) 

s_mul_i32 s6, s50, s0 // (op 711) 

s_mul_i32 s0, s7, s6 // (op 711) 

s_sub_u32 s6, s16, s0 // (op 711) 

s_mov_b32 s0, s6 // (op 711) 
s_mul_hi_i32 s51, s0, s13 // (op 711) 
s_add_i32 s18, s51, s0 // (op 711) 


s_ashr_i32 s0, s18, 31 // (op 711) 
s_cmp_eq_i32 s13, 0 // (op 711) 
s_mov_b32 s19, scc // (op 711) 
s_cmp_eq_i32 s19, 0 // (op 711) 
s_cselect_b32 s51, 0, -1 // (op 711) 

s_and_b32 s19, s49, 31 // (op 711) 
s_lshl_b32 s60, 1, s19 // (op 711) 
s_add_i32 s61, s60, s51 // (op 711) 


s_and_b32 s51, s0, s61 // (op 711) 


s_add_i32 s0, s18, s51 // (op 711) 


s_ashr_i32 s18, s0, s19 // (op 711) 


s_lshl_b32 s0, s49, 24 // (op 711) 
s_ashr_i32 s19, s0, 31 // (op 711) 

s_xor_b32 s0, s18, s19 // (op 711) 

s_sub_i32 s18, s0, s19 // (op 711) 


s_mul_i32 s0, s48, s14 // (op 711) 
s_add_u32 s60, s20, 16 // (op 711) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 711) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 711) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 711) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 711) 

s_sub_i32 s19, s60, s0 // (op 711) 

s_mul_i32 s51, s18, s19 // (op 711) 


s_sub_u32 s18, s6, s51 // (op 711) 


s_add_u32 s6, s18, s0 // (op 711) 


s_cmp_eq_i32 s7, 0 // (op 711) 
s_mov_b32 s0, scc // (op 711) 

s_cmp_eq_i32 s0, 0 // (op 711) 
s_cselect_b32 s7, s6, s1 // (op 711) 



v_lshl_add_u32 v18, s7, 4, v19 // (op 711) 


v_mul_lo_u32 v19, v20, s30 // (op 711) 

v_add_u32 v20, v18, v19 // (op 711) 


s_mul_i32 s1, s30, s35 // (op 713) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 713) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 713) least significant
s_mul_hi_u32 s6, s30, s34 // (op 713) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 713) most significant: sum
s_add_u32 s1, s1, s6 // (op 713) most significant: sum


s_mov_b32 s60, s32 // (op 713) reg expression
s_mov_b32 s61, s33 // (op 713) reg expression
s_mov_b32 s62, s0 // (op 713) reg expression
s_mov_b32 s63, 131072 // (op 713) call()

s_mul_hi_i32 s51, s16, s9 // (op 730) 
v_and_b32 v18, -16, v1 // (op 730) 
s_add_i32 s0, s51, s16 // (op 730) 

v_sub_u32 v19, v1, v18 // (op 730) 

s_ashr_i32 s1, s0, 31 // (op 730) 
v_lshrrev_b32 v18, 4, v1 // (op 730) 
s_cmp_eq_i32 s9, 0 // (op 730) 
s_mov_b32 s51, scc // (op 730) 
v_lshlrev_b32 v21, 1, v18 // (op 730) 

s_cmp_eq_i32 s51, 0 // (op 730) 
s_cselect_b32 s6, 0, -1 // (op 730) 

s_and_b32 s7, s10, 31 // (op 730) 
s_lshl_b32 s51, 1, s7 // (op 730) 
s_add_i32 s18, s51, s6 // (op 730) 


s_and_b32 s6, s1, s18 // (op 730) 


s_add_i32 s1, s0, s6 // (op 730) 


s_ashr_i32 s0, s1, s7 // (op 730) 


s_lshl_b32 s1, s10, 24 // (op 730) 
s_ashr_i32 s51, s1, 31 // (op 730) 

s_xor_b32 s1, s0, s51 // (op 730) 

s_sub_i32 s0, s1, s51 // (op 730) 


s_mul_hi_i32 s1, s0, s11 // (op 730) 
s_add_i32 s51, s1, s0 // (op 730) 

s_ashr_i32 s1, s51, 31 // (op 730) 
s_cmp_eq_i32 s11, 0 // (op 730) 
s_mov_b32 s6, scc // (op 730) 
s_cmp_eq_i32 s6, 0 // (op 730) 
s_cselect_b32 s7, 0, -1 // (op 730) 

s_and_b32 s6, s12, 31 // (op 730) 
s_lshl_b32 s18, 1, s6 // (op 730) 
s_add_i32 s19, s18, s7 // (op 730) 


s_and_b32 s7, s1, s19 // (op 730) 


s_add_i32 s1, s51, s7 // (op 730) 


s_ashr_i32 s7, s1, s6 // (op 730) 


s_lshl_b32 s1, s12, 24 // (op 730) 
s_ashr_i32 s6, s1, 31 // (op 730) 

s_xor_b32 s1, s7, s6 // (op 730) 

s_sub_i32 s7, s1, s6 // (op 730) 


s_mul_i32 s1, s0, s14 // (op 730) 

s_sub_u32 s0, s16, s1 // (op 730) 

s_mul_i32 s1, s7, s14 // (op 730) 
s_add_u32 s1, s1, s0 // (op 730) 


s_mul_hi_i32 s0, s16, s8 // (op 730) 
s_add_i32 s51, s0, s16 // (op 730) 

s_ashr_i32 s0, s51, 31 // (op 730) 
s_cmp_eq_i32 s8, 0 // (op 730) 
s_mov_b32 s6, scc // (op 730) 
s_cmp_eq_i32 s6, 0 // (op 730) 
s_cselect_b32 s7, 0, -1 // (op 730) 

s_and_b32 s6, s15, 31 // (op 730) 
s_lshl_b32 s18, 1, s6 // (op 730) 
s_add_i32 s19, s18, s7 // (op 730) 


s_and_b32 s7, s0, s19 // (op 730) 


s_add_i32 s0, s51, s7 // (op 730) 


s_ashr_i32 s7, s0, s6 // (op 730) 


s_lshl_b32 s0, s15, 24 // (op 730) 
s_ashr_i32 s6, s0, 31 // (op 730) 

s_xor_b32 s0, s7, s6 // (op 730) 

s_sub_i32 s7, s0, s6 // (op 730) 


s_add_u32 s18, s4, 16 // (op 730) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 730) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 730) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 730) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 730) 

s_mul_i32 s0, s14, s18 // (op 730) 

s_mul_i32 s6, s50, s0 // (op 730) 

s_mul_i32 s0, s7, s6 // (op 730) 

s_sub_u32 s6, s16, s0 // (op 730) 

s_mov_b32 s0, s6 // (op 730) 
s_mul_hi_i32 s51, s0, s13 // (op 730) 
s_add_i32 s18, s51, s0 // (op 730) 


s_ashr_i32 s0, s18, 31 // (op 730) 
s_cmp_eq_i32 s13, 0 // (op 730) 
s_mov_b32 s19, scc // (op 730) 
s_cmp_eq_i32 s19, 0 // (op 730) 
s_cselect_b32 s51, 0, -1 // (op 730) 

s_and_b32 s19, s49, 31 // (op 730) 
s_lshl_b32 s64, 1, s19 // (op 730) 
s_add_i32 s65, s64, s51 // (op 730) 


s_and_b32 s51, s0, s65 // (op 730) 


s_add_i32 s0, s18, s51 // (op 730) 


s_ashr_i32 s18, s0, s19 // (op 730) 


s_lshl_b32 s0, s49, 24 // (op 730) 
s_ashr_i32 s19, s0, 31 // (op 730) 

s_xor_b32 s0, s18, s19 // (op 730) 

s_sub_i32 s18, s0, s19 // (op 730) 


s_mul_i32 s0, s48, s14 // (op 730) 
s_add_u32 s64, s20, 16 // (op 730) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 730) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 730) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 730) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 730) 

s_sub_i32 s19, s64, s0 // (op 730) 

s_mul_i32 s51, s18, s19 // (op 730) 


s_sub_u32 s18, s6, s51 // (op 730) 


s_add_u32 s6, s18, s0 // (op 730) 


s_cmp_eq_i32 s7, 0 // (op 730) 
s_mov_b32 s0, scc // (op 730) 

s_cmp_eq_i32 s0, 0 // (op 730) 
s_cselect_b32 s7, s6, s1 // (op 730) 



v_lshl_add_u32 v18, s7, 4, v19 // (op 730) 


v_mul_lo_u32 v19, v21, s30 // (op 730) 

v_add_u32 v21, v18, v19 // (op 730) 


s_mul_i32 s1, s30, s35 // (op 732) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 732) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 732) least significant
s_mul_hi_u32 s6, s30, s34 // (op 732) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 732) most significant: sum
s_add_u32 s1, s1, s6 // (op 732) most significant: sum


s_mov_b32 s60, s32 // (op 732) reg expression
s_mov_b32 s61, s33 // (op 732) reg expression
s_mov_b32 s62, s0 // (op 732) reg expression
s_mov_b32 s63, 131072 // (op 732) call()

s_mul_hi_i32 s51, s16, s9 // (op 775) 
v_and_b32 v18, -16, v1 // (op 775) 
s_add_i32 s0, s51, s16 // (op 775) 

v_sub_u32 v19, v1, v18 // (op 775) 

s_ashr_i32 s1, s0, 31 // (op 775) 
v_lshrrev_b32 v18, 4, v1 // (op 775) 
s_cmp_eq_i32 s9, 0 // (op 775) 
s_mov_b32 s51, scc // (op 775) 
v_lshlrev_b32 v22, 1, v18 // (op 775) 

s_cmp_eq_i32 s51, 0 // (op 775) 
s_cselect_b32 s6, 0, -1 // (op 775) 

s_and_b32 s7, s10, 31 // (op 775) 
s_lshl_b32 s51, 1, s7 // (op 775) 
s_add_i32 s18, s51, s6 // (op 775) 


s_and_b32 s6, s1, s18 // (op 775) 


s_add_i32 s1, s0, s6 // (op 775) 


s_ashr_i32 s0, s1, s7 // (op 775) 


s_lshl_b32 s1, s10, 24 // (op 775) 
s_ashr_i32 s51, s1, 31 // (op 775) 

s_xor_b32 s1, s0, s51 // (op 775) 

s_sub_i32 s0, s1, s51 // (op 775) 


s_mul_hi_i32 s1, s0, s11 // (op 775) 
s_add_i32 s51, s1, s0 // (op 775) 

s_ashr_i32 s1, s51, 31 // (op 775) 
s_cmp_eq_i32 s11, 0 // (op 775) 
s_mov_b32 s6, scc // (op 775) 
s_cmp_eq_i32 s6, 0 // (op 775) 
s_cselect_b32 s7, 0, -1 // (op 775) 

s_and_b32 s6, s12, 31 // (op 775) 
s_lshl_b32 s18, 1, s6 // (op 775) 
s_add_i32 s19, s18, s7 // (op 775) 


s_and_b32 s7, s1, s19 // (op 775) 


s_add_i32 s1, s51, s7 // (op 775) 


s_ashr_i32 s7, s1, s6 // (op 775) 


s_lshl_b32 s1, s12, 24 // (op 775) 
s_ashr_i32 s6, s1, 31 // (op 775) 

s_xor_b32 s1, s7, s6 // (op 775) 

s_sub_i32 s7, s1, s6 // (op 775) 


s_add_u32 s18, s4, 16 // (op 775) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 775) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 775) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 775) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 775) 

s_mul_i32 s1, s7, s18 // (op 775) 

s_sub_i32 s51, s0, s1 // (op 775) 


s_mul_hi_i32 s0, s16, s8 // (op 775) 
s_add_i32 s1, s0, s16 // (op 775) 

s_ashr_i32 s0, s1, 31 // (op 775) 
s_cmp_eq_i32 s8, 0 // (op 775) 
s_mov_b32 s6, scc // (op 775) 
s_cmp_eq_i32 s6, 0 // (op 775) 
s_cselect_b32 s7, 0, -1 // (op 775) 

s_and_b32 s6, s15, 31 // (op 775) 
s_lshl_b32 s64, 1, s6 // (op 775) 
s_add_i32 s65, s64, s7 // (op 775) 


s_and_b32 s7, s0, s65 // (op 775) 


s_add_i32 s0, s1, s7 // (op 775) 


s_ashr_i32 s1, s0, s6 // (op 775) 


s_lshl_b32 s0, s15, 24 // (op 775) 
s_ashr_i32 s6, s0, 31 // (op 775) 

s_xor_b32 s0, s1, s6 // (op 775) 

s_sub_i32 s1, s0, s6 // (op 775) 


s_mul_i32 s0, s14, s18 // (op 775) 

s_mul_i32 s6, s50, s0 // (op 775) 

s_mul_i32 s0, s1, s6 // (op 775) 

s_sub_u32 s6, s16, s0 // (op 775) 

s_mul_hi_i32 s0, s6, s13 // (op 775) 
s_add_i32 s7, s0, s6 // (op 775) 


s_ashr_i32 s0, s7, 31 // (op 775) 
s_cmp_eq_i32 s13, 0 // (op 775) 
s_mov_b32 s6, scc // (op 775) 
s_cmp_eq_i32 s6, 0 // (op 775) 
s_cselect_b32 s18, 0, -1 // (op 775) 

s_and_b32 s6, s49, 31 // (op 775) 
s_lshl_b32 s19, 1, s6 // (op 775) 
s_add_i32 s64, s19, s18 // (op 775) 


s_and_b32 s18, s0, s64 // (op 775) 


s_add_i32 s0, s7, s18 // (op 775) 


s_ashr_i32 s7, s0, s6 // (op 775) 


s_lshl_b32 s0, s49, 24 // (op 775) 
s_ashr_i32 s6, s0, 31 // (op 775) 

s_xor_b32 s0, s7, s6 // (op 775) 

s_sub_i32 s7, s0, s6 // (op 775) 


s_cmp_eq_i32 s1, 0 // (op 775) 
s_mov_b32 s0, scc // (op 775) 

s_cmp_eq_i32 s0, 0 // (op 775) 
s_cselect_b32 s1, s7, s51 // (op 775) 



v_lshl_add_u32 v18, s1, 4, v19 // (op 775) 


v_mul_lo_u32 v19, v22, s36 // (op 775) 
v_add_u32 v19, v19, v18 // (op 775) 


s_mul_i32 s1, s36, s35 // (op 777) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 777) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 777) least significant
s_mul_hi_u32 s6, s36, s34 // (op 777) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 777) most significant: sum
s_add_u32 s1, s1, s6 // (op 777) most significant: sum


s_mov_b32 s64, s38 // (op 777) reg expression
s_mov_b32 s65, s39 // (op 777) reg expression
s_mov_b32 s66, s0 // (op 777) reg expression
s_mov_b32 s67, 131072 // (op 777) call()

s_mul_hi_i32 s51, s16, s9 // (op 794) 
v_and_b32 v18, -16, v1 // (op 794) 
s_add_i32 s0, s51, s16 // (op 794) 

v_sub_u32 v22, v1, v18 // (op 794) 

s_ashr_i32 s1, s0, 31 // (op 794) 
v_lshrrev_b32 v18, 4, v1 // (op 794) 
s_cmp_eq_i32 s9, 0 // (op 794) 
s_mov_b32 s51, scc // (op 794) 
v_lshlrev_b32 v23, 1, v18 // (op 794) 

s_cmp_eq_i32 s51, 0 // (op 794) 
s_cselect_b32 s6, 0, -1 // (op 794) 

s_and_b32 s7, s10, 31 // (op 794) 
s_lshl_b32 s51, 1, s7 // (op 794) 
s_add_i32 s18, s51, s6 // (op 794) 


s_and_b32 s6, s1, s18 // (op 794) 


s_add_i32 s1, s0, s6 // (op 794) 


s_ashr_i32 s0, s1, s7 // (op 794) 


s_lshl_b32 s1, s10, 24 // (op 794) 
s_ashr_i32 s51, s1, 31 // (op 794) 

s_xor_b32 s1, s0, s51 // (op 794) 

s_sub_i32 s0, s1, s51 // (op 794) 


s_mul_hi_i32 s1, s0, s11 // (op 794) 
s_add_i32 s51, s1, s0 // (op 794) 

s_ashr_i32 s1, s51, 31 // (op 794) 
s_cmp_eq_i32 s11, 0 // (op 794) 
s_mov_b32 s6, scc // (op 794) 
s_cmp_eq_i32 s6, 0 // (op 794) 
s_cselect_b32 s7, 0, -1 // (op 794) 

s_and_b32 s6, s12, 31 // (op 794) 
s_lshl_b32 s18, 1, s6 // (op 794) 
s_add_i32 s19, s18, s7 // (op 794) 


s_and_b32 s7, s1, s19 // (op 794) 


s_add_i32 s1, s51, s7 // (op 794) 


s_ashr_i32 s7, s1, s6 // (op 794) 


s_lshl_b32 s1, s12, 24 // (op 794) 
s_ashr_i32 s6, s1, 31 // (op 794) 

s_xor_b32 s1, s7, s6 // (op 794) 

s_sub_i32 s7, s1, s6 // (op 794) 


s_add_u32 s18, s4, 16 // (op 794) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 794) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 794) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 794) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 794) 

s_mul_i32 s1, s7, s18 // (op 794) 

s_sub_i32 s51, s0, s1 // (op 794) 


s_mul_hi_i32 s0, s16, s8 // (op 794) 
s_add_i32 s1, s0, s16 // (op 794) 

s_ashr_i32 s0, s1, 31 // (op 794) 
s_cmp_eq_i32 s8, 0 // (op 794) 
s_mov_b32 s6, scc // (op 794) 
s_cmp_eq_i32 s6, 0 // (op 794) 
s_cselect_b32 s7, 0, -1 // (op 794) 

s_and_b32 s6, s15, 31 // (op 794) 
s_lshl_b32 s68, 1, s6 // (op 794) 
s_add_i32 s69, s68, s7 // (op 794) 


s_and_b32 s7, s0, s69 // (op 794) 


s_add_i32 s0, s1, s7 // (op 794) 


s_ashr_i32 s1, s0, s6 // (op 794) 


s_lshl_b32 s0, s15, 24 // (op 794) 
s_ashr_i32 s6, s0, 31 // (op 794) 

s_xor_b32 s0, s1, s6 // (op 794) 

s_sub_i32 s1, s0, s6 // (op 794) 


s_mul_i32 s0, s14, s18 // (op 794) 

s_mul_i32 s6, s50, s0 // (op 794) 

s_mul_i32 s0, s1, s6 // (op 794) 

s_sub_u32 s6, s16, s0 // (op 794) 

s_mul_hi_i32 s0, s6, s13 // (op 794) 
s_add_i32 s7, s0, s6 // (op 794) 


s_ashr_i32 s0, s7, 31 // (op 794) 
s_cmp_eq_i32 s13, 0 // (op 794) 
s_mov_b32 s6, scc // (op 794) 
s_cmp_eq_i32 s6, 0 // (op 794) 
s_cselect_b32 s18, 0, -1 // (op 794) 

s_and_b32 s6, s49, 31 // (op 794) 
s_lshl_b32 s19, 1, s6 // (op 794) 
s_add_i32 s68, s19, s18 // (op 794) 


s_and_b32 s18, s0, s68 // (op 794) 


s_add_i32 s0, s7, s18 // (op 794) 


s_ashr_i32 s7, s0, s6 // (op 794) 


s_lshl_b32 s0, s49, 24 // (op 794) 
s_ashr_i32 s6, s0, 31 // (op 794) 

s_xor_b32 s0, s7, s6 // (op 794) 

s_sub_i32 s7, s0, s6 // (op 794) 


s_cmp_eq_i32 s1, 0 // (op 794) 
s_mov_b32 s0, scc // (op 794) 

s_cmp_eq_i32 s0, 0 // (op 794) 
s_cselect_b32 s1, s7, s51 // (op 794) 



v_lshl_add_u32 v18, s1, 4, v22 // (op 794) 


v_mul_lo_u32 v22, v23, s36 // (op 794) 
v_add_u32 v22, v22, v18 // (op 794) 


s_mul_i32 s1, s36, s35 // (op 796) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 796) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 796) least significant
s_mul_hi_u32 s6, s36, s34 // (op 796) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 796) most significant: sum
s_add_u32 s1, s1, s6 // (op 796) most significant: sum


s_mov_b32 s64, s38 // (op 796) reg expression
s_mov_b32 s65, s39 // (op 796) reg expression
s_mov_b32 s66, s0 // (op 796) reg expression
s_mov_b32 s67, 131072 // (op 796) call()

v_lshlrev_b32 v18, 4, v1 // (op 913) 
s_nop 0
v_readfirstlane_b32 s51, v18 // (op 913) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 539) 
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




v_mov_b64 v[24:25], s[36:37] // (op 427) convert

v_mov_b32 v18, v19 // (op 427) reg expression
buffer_load_ubyte v26, v18, s[64:67], 0 offen offset: 0 // (op 427) Load value
v_add_u32 v18, v18, v24 // (op 427) 
buffer_load_ubyte v27, v18, s[64:67], 0 offen offset: 0 // (op 427) Load value



s_mul_i32 s1, s3, 6 // (op 430) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 430) least significant
s_mul_hi_u32 s51, s2, 6 // (op 430) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 430) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 430) 


v_mov_b32 v18, v12 // (op 430) reg expression
buffer_load_dwordx3 v[28:30], v18, s[52:55], 0 offen // (op 430) Load value
v_add_u32 v18, v18, v24 // (op 430) 
buffer_load_dwordx3 v[32:34], v18, s[52:55], 0 offen // (op 430) Load value
v_add_u32 v18, v18, v24 // (op 430) 
buffer_load_dwordx3 v[36:38], v18, s[52:55], 0 offen // (op 430) Load value
v_add_u32 v18, v18, v24 // (op 430) 
buffer_load_dwordx3 v[40:42], v18, s[52:55], 0 offen // (op 430) Load value



v_mov_b64 v[24:25], s[30:31] // (op 433) convert

v_mov_b32 v18, v20 // (op 433) reg expression
buffer_load_ubyte v44, v18, s[60:63], 0 offen offset: 0 // (op 433) Load value
v_add_u32 v18, v18, v24 // (op 433) 
buffer_load_ubyte v45, v18, s[60:63], 0 offen offset: 0 // (op 433) Load value


v_lshlrev_b32 v18, 4, v1 // (op 1116) 
s_nop 0
v_readfirstlane_b32 s51, v18 // (op 1116) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 544) 
v_mov_b64 v[24:25], s[0:1] // (op 544) convert


s_lshl_b64 s[0:1], s[26:27], 8 // (op 544) 
v_mov_b32 v18, s0 // (op 544) 
v_mov_b32 v23, s1 // (op 544) 
v_add_co_u32 v46, s[6:7], v14, v18 // (op 544) least significant half
v_addc_co_u32 v47, s[6:7], v15, v23, s[6:7] // (op 544) most significant half




s_add_u32 s0, 4096, s51 // (op 544) 
s_mov_b32 m0, s0 // (op 544) reg expression
buffer_load_dwordx4 v46, s[56:59], 0 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v46, s[6:7], v46, v24 // (op 544) least significant half
v_addc_co_u32 v47, s[6:7], 0, v47, s[6:7] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v46, s[56:59], 0 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v46, s[6:7], v46, v24 // (op 544) least significant half
v_addc_co_u32 v47, s[6:7], 0, v47, s[6:7] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v46, s[56:59], 0 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v46, s[6:7], v46, v24 // (op 544) least significant half
v_addc_co_u32 v47, s[6:7], 0, v47, s[6:7] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v46, s[56:59], 0 offen offset: 0 lds // (op 544) Load value direct to lds






v_mov_b64 v[14:15], s[36:37] // (op 439) convert

s_lshl_b32 s51, s36, 3 // (op 439) 
v_add_u32 v18, v19, s51 // (op 439) 

buffer_load_ubyte v24, v18, s[64:67], 0 offen offset: 0 // (op 439) Load value
v_add_u32 v18, v18, v14 // (op 439) 
buffer_load_ubyte v25, v18, s[64:67], 0 offen offset: 0 // (op 439) Load value




s_mul_i32 s1, s3, 6 // (op 442) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 442) least significant
s_mul_hi_u32 s51, s2, 6 // (op 442) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 442) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[0:1] // (op 442) 


s_lshl_b64 s[0:1], s[2:3], 8 // (op 442) 
s_mul_i32 s7, s1, 6 // (op 442) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 442) least significant
s_mul_hi_u32 s51, s0, 6 // (op 442) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 442) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 442) 

v_add_u32 v23, v12, s0 // (op 442) 

buffer_load_dwordx3 v[46:48], v23, s[52:55], 0 offen // (op 442) Load value
v_add_u32 v23, v23, v14 // (op 442) 
buffer_load_dwordx3 v[50:52], v23, s[52:55], 0 offen // (op 442) Load value
v_add_u32 v23, v23, v14 // (op 442) 
buffer_load_dwordx3 v[54:56], v23, s[52:55], 0 offen // (op 442) Load value
v_add_u32 v23, v23, v14 // (op 442) 
buffer_load_dwordx3 v[58:60], v23, s[52:55], 0 offen // (op 442) Load value




v_mov_b64 v[14:15], s[30:31] // (op 445) convert

s_lshl_b32 s51, s30, 3 // (op 445) 
v_add_u32 v12, v20, s51 // (op 445) 

buffer_load_ubyte v18, v12, s[60:63], 0 offen offset: 0 // (op 445) Load value
v_add_u32 v12, v12, v14 // (op 445) 
buffer_load_ubyte v19, v12, s[60:63], 0 offen offset: 0 // (op 445) Load value



v_and_b32 v12, -16, v1 // (op 1019) 
v_sub_u32 v20, v1, v12 // (op 1019) 

v_lshrrev_b32 v12, 4, v1 // (op 1019) 
v_lshl_add_u32 v23, v12, 5, v20 // (op 1019) 





s_waitcnt vmcnt(19)
ds_write_b8 v23, v26 offset:8192 // (op 448) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v23, v27 offset:8208 // (op 448) Store local data 

v_lshlrev_b32 v12, 6, v1 // (op 861) 
v_mul_lo_u32 v20, v12, 6 // (op 861) 
v_lshrrev_b32 v23, 3, v20 // (op 861) 

v_lshrrev_b32 v20, 4, v12 // (op 861) 

v_lshlrev_b32 v12, 2, v20 // (op 861) 

v_add_u32 v20, v23, v12 // (op 861) 





s_waitcnt vmcnt(17)
ds_write_b96 v20, v[28:30] offset:8320 // (op 449) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v20, v[32:34] offset:8336 // (op 449) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v20, v[36:38] offset:8352 // (op 449) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v20, v[40:42] offset:8368 // (op 449) Store local data 

v_and_b32 v12, -16, v1 // (op 966) 
v_sub_u32 v20, v1, v12 // (op 966) 

v_lshrrev_b32 v12, 4, v1 // (op 966) 
v_lshl_add_u32 v23, v12, 5, v20 // (op 966) 





s_waitcnt vmcnt(13)
ds_write_b8 v23, v44 offset:12416 // (op 450) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v23, v45 offset:12432 // (op 450) Store local data 

s_barrier  // (op 1232) 



ds_read_b64_tr_b8 v[62:63], v3 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v3 offset:128 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v3 offset:1024 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v3 offset:1152 // (op 467) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1268) 


ds_read_u8 v12, v7 offset:8192 // (op 473) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1213) 



ds_read_b96_tr_b6 v[70:72], v0 offset:8320 // (op 464) Transpose load local data 
ds_read_b96_tr_b6 v[73:75], v0 offset:8576 // (op 464) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1248) 


ds_read_u8 v20, v5 offset:12416 // (op 470) Load local data 
v_lshlrev_b32 v23, 6, v1 // (op 832) 
v_mul_lo_u32 v31, v23, 6 // (op 832) 
v_lshrrev_b32 v35, 3, v31 // (op 832) 

v_lshrrev_b32 v31, 4, v23 // (op 832) 

v_lshlrev_b32 v23, 2, v31 // (op 832) 

v_add_u32 v31, v35, v23 // (op 832) 


v_lshlrev_b32 v23, 4, v1 // (op 885) 
s_nop 0
v_readfirstlane_b32 s51, v23 // (op 885) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -16, v1 // (op 937) 
v_sub_u32 v35, v1, v23 // (op 937) 

v_lshrrev_b32 v23, 4, v1 // (op 937) 
v_lshl_add_u32 v39, v23, 5, v35 // (op 937) 


v_and_b32 v23, -16, v1 // (op 990) 
v_sub_u32 v35, v1, v23 // (op 990) 

v_lshrrev_b32 v23, 4, v1 // (op 990) 
v_lshl_add_u32 v43, v23, 5, v35 // (op 990) 


v_lshlrev_b32 v23, 6, v1 // (op 1077) 
v_mul_lo_u32 v35, v23, 6 // (op 1077) 
v_lshrrev_b32 v49, 3, v35 // (op 1077) 

v_lshrrev_b32 v35, 4, v23 // (op 1077) 

v_lshlrev_b32 v23, 2, v35 // (op 1077) 

v_add_u32 v35, v49, v23 // (op 1077) 


v_lshlrev_b32 v23, 4, v1 // (op 1101) 
s_nop 0
v_readfirstlane_b32 s0, v23 // (op 1101) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -16, v1 // (op 1140) 
v_sub_u32 v49, v1, v23 // (op 1140) 

v_lshrrev_b32 v23, 4, v1 // (op 1140) 
v_lshl_add_u32 v53, v23, 5, v49 // (op 1140) 


v_and_b32 v23, -16, v1 // (op 1164) 
v_sub_u32 v49, v1, v23 // (op 1164) 

v_lshrrev_b32 v23, 4, v1 // (op 1164) 
v_lshl_add_u32 v57, v23, 5, v49 // (op 1164) 


s_mov_b32 s1, 0 // (op 59) call()
s_ashr_i64 s[6:7], s[24:25], 8 // (op 58) 
s_mov_b32 s18, s6 // (op 58) convert

s_and_b32 s19, s18, 4294967294 // (op 58) 

s_cmp_lt_u32 s1, s19 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58 if false)
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58:
s_add_u32 s1, s1, 2 // (op 60) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:75], v[62:69], a[0:3], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 88) 
s_barrier  // (op 1235) 

s_barrier  // (op 1222) 
s_lshl_b64 s[6:7], s[26:27], 6 // (op 522) 



ds_read_b96_tr_b6 v[70:72], v0 offset:10368 // (op 91) Transpose load local data 
ds_read_b96_tr_b6 v[73:75], v0 offset:10624 // (op 91) Transpose load local data 
v_mov_b64 v[14:15], s[6:7] // (op 522) convert


s_barrier  // (op 1241) 
v_mov_b64 v[76:77], v[16:17] // (op 522) 



ds_read_b64_tr_b8 v[62:63], v3 offset:2048 // (op 95) Transpose load local data 
s_lshl_b64 s[6:7], s[26:27], 8 // (op 522) 
s_lshl_b64 s[18:19], s[6:7], 1 // (op 522) 

v_mov_b32 v23, s18 // (op 522) 
v_mov_b32 v49, s19 // (op 522) 
v_add_co_u32 v78, s[6:7], v76, v23 // (op 522) least significant half
v_addc_co_u32 v79, s[6:7], v77, v49, s[6:7] // (op 522) most significant half




s_mov_b32 m0, s51 // (op 522) reg expression
buffer_load_dwordx4 v78, s[56:59], 0 offen offset: 0 lds // (op 522) Load value direct to lds
v_add_co_u32 v78, s[6:7], v78, v14 // (op 522) least significant half
v_addc_co_u32 v79, s[6:7], 0, v79, s[6:7] // (op 522) most significant half

s_add_u32 m0, m0, 1024 // (op 522) 
buffer_load_dwordx4 v78, s[56:59], 0 offen offset: 0 lds // (op 522) Load value direct to lds
v_add_co_u32 v78, s[6:7], v78, v14 // (op 522) least significant half
v_addc_co_u32 v79, s[6:7], 0, v79, s[6:7] // (op 522) most significant half

s_add_u32 m0, m0, 1024 // (op 522) 
buffer_load_dwordx4 v78, s[56:59], 0 offen offset: 0 lds // (op 522) Load value direct to lds
v_add_co_u32 v78, s[6:7], v78, v14 // (op 522) least significant half
v_addc_co_u32 v79, s[6:7], 0, v79, s[6:7] // (op 522) most significant half

s_add_u32 m0, m0, 1024 // (op 522) 
ds_read_b64_tr_b8 v[64:65], v3 offset:2176 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v3 offset:3072 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v3 offset:3200 // (op 95) Transpose load local data 
buffer_load_dwordx4 v78, s[56:59], 0 offen offset: 0 lds // (op 522) Load value direct to lds



s_barrier  // (op 1258) 
v_mov_b64 v[14:15], s[36:37] // (op 25) convert



ds_read_u8 v20, v5 offset:12480 // (op 99) Load local data 
v_mov_b32 v23, v22 // (op 25) 
s_barrier  // (op 1278) 
s_lshl_b32 s6, s36, 4 // (op 25) 


ds_read_u8 v12, v7 offset:8256 // (op 103) Load local data 
v_add_u32 v49, v23, s6 // (op 25) 

buffer_load_ubyte v26, v49, s[64:67], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v49, v49, v14 // (op 25) 
buffer_load_ubyte v27, v49, s[64:67], 0 offen offset: 0 // (op 25) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:75], v[62:69], a[0:3], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 105) 

s_mul_i32 s7, s3, 6 // (op 4) most significant: low of high * low
s_mul_i32 s6, s2, 6 // (op 4) least significant
s_mul_hi_u32 s18, s2, 6 // (op 4) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 4) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[6:7] // (op 4) 


v_mov_b32 v49, v13 // (op 4) 
s_lshl_b64 s[6:7], s[2:3], 8 // (op 4) 
s_mul_i32 s19, s7, 6 // (op 4) most significant: low of high * low
s_mul_i32 s18, s6, 6 // (op 4) least significant
s_mul_hi_u32 s68, s6, 6 // (op 4) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 4) most significant: sum


s_lshr_b64 s[6:7], s[18:19], 3 // (op 4) 

s_lshl_b32 s18, s6, 1 // (op 4) 

v_add_u32 v61, v49, s18 // (op 4) 

buffer_load_dwordx3 v[28:30], v61, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v61, v61, v14 // (op 4) 
buffer_load_dwordx3 v[32:34], v61, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v61, v61, v14 // (op 4) 
buffer_load_dwordx3 v[36:38], v61, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v61, v61, v14 // (op 4) 
buffer_load_dwordx3 v[40:42], v61, s[52:55], 0 offen // (op 4) Load value



v_mov_b64 v[14:15], s[30:31] // (op 17) convert

v_mov_b32 v61, v21 // (op 17) 
s_lshl_b32 s6, s30, 4 // (op 17) 
v_add_u32 v78, v61, s6 // (op 17) 

buffer_load_ubyte v44, v78, s[60:63], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v78, v78, v14 // (op 17) 
buffer_load_ubyte v45, v78, s[60:63], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1313) 



s_waitcnt vmcnt(19)
ds_write_b8 v57, v24 offset:12544 // (op 304) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v57, v25 offset:12560 // (op 304) Store local data 
s_barrier  // (op 1291) 



s_waitcnt vmcnt(17)
ds_write_b96 v35, v[46:48] offset:12672 // (op 283) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v35, v[50:52] offset:12688 // (op 283) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v35, v[54:56] offset:12704 // (op 283) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v35, v[58:60] offset:12720 // (op 283) Store local data 
s_barrier  // (op 1307) 



s_waitcnt vmcnt(13)
ds_write_b8 v53, v18 offset:16768 // (op 303) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v53, v19 offset:16784 // (op 303) Store local data 
s_barrier  // (op 1294) 



ds_read_b64_tr_b8 v[78:79], v6 offset:4096 // (op 287) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v6 offset:4224 // (op 287) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v6 offset:5120 // (op 287) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v6 offset:5248 // (op 287) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1310) 


ds_read_u8 v14, v11 offset:12544 // (op 291) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1288) 



ds_read_b96_tr_b6 v[86:88], v4 offset:12672 // (op 285) Transpose load local data 
ds_read_b96_tr_b6 v[89:91], v4 offset:12928 // (op 285) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1304) 


ds_read_u8 v15, v9 offset:16768 // (op 289) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[86:91], v[78:85], a[0:3], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 292) 
s_barrier  // (op 1298) 








s_lshl_b64 s[6:7], s[26:27], 6 // (op 534) 
v_mov_b64 v[14:15], s[6:7] // (op 534) convert


s_lshl_b64 s[6:7], s[26:27], 8 // (op 534) 
s_mul_i32 s19, 3, s7 // (op 534) most significant: low of low * high
s_mul_i32 s18, 3, s6 // (op 534) least significant
s_mul_hi_u32 s68, 3, s6 // (op 534) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 534) most significant: sum


v_mov_b32 v78, s18 // (op 534) 
v_mov_b32 v79, s19 // (op 534) 
ds_read_b96_tr_b6 v[80:82], v4 offset:14720 // (op 294) Transpose load local data 
ds_read_b96_tr_b6 v[83:85], v4 offset:14976 // (op 294) Transpose load local data 
v_add_co_u32 v86, s[6:7], v76, v78 // (op 534) least significant half
v_addc_co_u32 v87, s[6:7], v77, v79, s[6:7] // (op 534) most significant half




s_barrier  // (op 1301) 
s_barrier  // (op 1368) 



s_add_u32 s6, 4096, s0 // (op 534) 
s_mov_b32 m0, s6 // (op 534) reg expression
buffer_load_dwordx4 v86, s[56:59], 0 offen offset: 0 lds // (op 534) Load value direct to lds
v_add_co_u32 v86, s[18:19], v86, v14 // (op 534) least significant half
v_addc_co_u32 v87, s[18:19], 0, v87, s[18:19] // (op 534) most significant half

s_add_u32 m0, m0, 1024 // (op 534) 
buffer_load_dwordx4 v86, s[56:59], 0 offen offset: 0 lds // (op 534) Load value direct to lds
v_add_co_u32 v86, s[18:19], v86, v14 // (op 534) least significant half
v_addc_co_u32 v87, s[18:19], 0, v87, s[18:19] // (op 534) most significant half

s_add_u32 m0, m0, 1024 // (op 534) 
buffer_load_dwordx4 v86, s[56:59], 0 offen offset: 0 lds // (op 534) Load value direct to lds
v_add_co_u32 v86, s[18:19], v86, v14 // (op 534) least significant half
v_addc_co_u32 v87, s[18:19], 0, v87, s[18:19] // (op 534) most significant half

s_add_u32 m0, m0, 1024 // (op 534) 
ds_read_b64_tr_b8 v[88:89], v6 offset:6144 // (op 296) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v6 offset:6272 // (op 296) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v6 offset:7168 // (op 296) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v6 offset:7296 // (op 296) Transpose load local data 
buffer_load_dwordx4 v86, s[56:59], 0 offen offset: 0 lds // (op 534) Load value direct to lds







ds_read_u8 v14, v9 offset:16832 // (op 298) Load local data 
v_mov_b64 v[86:87], s[36:37] // (op 307) convert

s_lshl_b32 s6, s36, 3 // (op 307) 


s_barrier  // (op 1374) 
ds_read_u8 v15, v11 offset:12608 // (op 300) Load local data 
s_mul_i32 s7, 3, s6 // (op 307) 

v_add_u32 v76, v23, s7 // (op 307) 

s_barrier  // (op 1378) 
buffer_load_ubyte v24, v76, s[64:67], 0 offen offset: 0 // (op 307) Load value
v_add_u32 v76, v76, v86 // (op 307) 
buffer_load_ubyte v25, v76, s[64:67], 0 offen offset: 0 // (op 307) Load value


s_lshl_b64 s[6:7], s[26:27], 8 // (op 658) 

s_lshl_b64 s[18:19], s[6:7], 1 // (op 658) 


s_lshl_b32 s6, s36, 4 // (op 786) 
v_add_u32 v22, v22, s6 // (op 786) 

v_mov_b32 v23, s18 // (op 658) 
v_mov_b32 v76, s19 // (op 658) 
s_mul_i32 s7, s3, 6 // (op 282) most significant: low of high * low
s_mul_i32 s6, s2, 6 // (op 282) least significant
s_mul_hi_u32 s68, s2, 6 // (op 282) most significant: high of low * low
s_add_u32 s7, s7, s68 // (op 282) most significant: sum

v_lshrrev_b64 v[78:79], 3, s[6:7] // (op 282) 


s_lshl_b64 s[6:7], s[2:3], 8 // (op 282) 
s_mul_i32 s69, s7, 6 // (op 282) most significant: low of high * low
s_mul_i32 s68, s6, 6 // (op 282) least significant
s_mul_hi_u32 s70, s6, 6 // (op 282) most significant: high of low * low
s_add_u32 s69, s69, s70 // (op 282) most significant: sum


s_lshr_b64 s[6:7], s[68:69], 3 // (op 282) 

s_mul_i32 s68, 3, s6 // (op 282) 

v_add_u32 v77, v49, s68 // (op 282) 

v_add_co_u32 v16, s[6:7], v16, v23 // (op 658) least significant half
v_addc_co_u32 v17, s[6:7], v17, v76, s[6:7] // (op 658) most significant half




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:85], v[88:95], a[0:3], v14, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 301) 
buffer_load_dwordx3 v[46:48], v77, s[52:55], 0 offen // (op 282) Load value




v_add_u32 v77, v77, v78 // (op 282) 
buffer_load_dwordx3 v[50:52], v77, s[52:55], 0 offen // (op 282) Load value
v_add_u32 v77, v77, v78 // (op 282) 
buffer_load_dwordx3 v[54:56], v77, s[52:55], 0 offen // (op 282) Load value
v_add_u32 v77, v77, v78 // (op 282) 
buffer_load_dwordx3 v[58:60], v77, s[52:55], 0 offen // (op 282) Load value




v_mov_b64 v[14:15], s[30:31] // (op 306) convert

s_lshl_b32 s6, s30, 3 // (op 306) 
s_mul_i32 s7, 3, s6 // (op 306) 

v_add_u32 v23, v61, s7 // (op 306) 

s_lshl_b64 s[6:7], s[2:3], 8 // (op 594) 
s_mul_i32 s19, s7, 6 // (op 594) most significant: low of high * low
s_mul_i32 s18, s6, 6 // (op 594) least significant
s_mul_hi_u32 s68, s6, 6 // (op 594) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 594) most significant: sum


buffer_load_ubyte v18, v23, s[60:63], 0 offen offset: 0 // (op 306) Load value
v_add_u32 v23, v23, v14 // (op 306) 
buffer_load_ubyte v19, v23, s[60:63], 0 offen offset: 0 // (op 306) Load value


s_lshr_b64 s[6:7], s[18:19], 3 // (op 594) 


s_lshl_b32 s18, s6, 1 // (op 594) 

s_barrier  // (op 1271) 
v_add_u32 v13, v13, s18 // (op 594) 




s_lshl_b32 s6, s30, 4 // (op 722) 
v_add_u32 v21, v21, s6 // (op 722) 

s_waitcnt vmcnt(19)
ds_write_b8 v43, v26 offset:8192 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v43, v27 offset:8208 // (op 30) Store local data 
s_barrier  // (op 1216) 



s_waitcnt vmcnt(17)
ds_write_b96 v31, v[28:30] offset:8320 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v31, v[32:34] offset:8336 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v31, v[36:38] offset:8352 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v31, v[40:42] offset:8368 // (op 6) Store local data 
s_barrier  // (op 1251) 



s_waitcnt vmcnt(13)
ds_write_b8 v39, v44 offset:12416 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v39, v45 offset:12432 // (op 21) Store local data 



ds_read_b64_tr_b8 v[62:63], v3 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v3 offset:128 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v3 offset:1024 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v3 offset:1152 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1281) 


ds_read_u8 v12, v7 offset:8192 // (op 85) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1225) 



ds_read_b96_tr_b6 v[70:72], v0 offset:8320 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[73:75], v0 offset:8576 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1261) 


ds_read_u8 v20, v5 offset:12416 // (op 80) Load local data 
s_ashr_i64 s[6:7], s[24:25], 8 // (op 58) 
s_mov_b32 s18, s6 // (op 58) convert

s_and_b32 s19, s18, 4294967294 // (op 58) 

s_cmp_lt_u32 s1, s19 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoop_58 if true)
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoop_58:
















s_barrier  // (op 1326) 
s_mul_hi_i32 s51, s16, s9 // (op 557) 
v_lshlrev_b32 v49, 2, v1 // (op 557) 
s_add_i32 s0, s51, s16 // (op 557) 

s_ashr_i32 s1, s0, 31 // (op 557) 
s_cmp_eq_i32 s9, 0 // (op 557) 
s_mov_b32 s51, scc // (op 557) 
s_cmp_eq_i32 s51, 0 // (op 557) 
s_cselect_b32 s6, 0, -1 // (op 557) 

s_and_b32 s7, s10, 31 // (op 557) 
s_lshl_b32 s51, 1, s7 // (op 557) 
s_add_i32 s18, s51, s6 // (op 557) 


s_and_b32 s6, s1, s18 // (op 557) 


s_add_i32 s1, s0, s6 // (op 557) 


s_ashr_i32 s0, s1, s7 // (op 557) 


s_lshl_b32 s1, s10, 24 // (op 557) 
s_ashr_i32 s51, s1, 31 // (op 557) 

s_xor_b32 s1, s0, s51 // (op 557) 

s_sub_i32 s0, s1, s51 // (op 557) 


s_mul_hi_i32 s1, s0, s11 // (op 557) 
s_add_i32 s51, s1, s0 // (op 557) 

s_ashr_i32 s1, s51, 31 // (op 557) 
s_cmp_eq_i32 s11, 0 // (op 557) 
s_mov_b32 s6, scc // (op 557) 
s_cmp_eq_i32 s6, 0 // (op 557) 
s_cselect_b32 s7, 0, -1 // (op 557) 

s_and_b32 s6, s12, 31 // (op 557) 
s_lshl_b32 s18, 1, s6 // (op 557) 
s_add_i32 s19, s18, s7 // (op 557) 


s_and_b32 s7, s1, s19 // (op 557) 


s_add_i32 s1, s51, s7 // (op 557) 


s_ashr_i32 s7, s1, s6 // (op 557) 


s_lshl_b32 s1, s12, 24 // (op 557) 
s_ashr_i32 s6, s1, 31 // (op 557) 

s_xor_b32 s1, s7, s6 // (op 557) 

s_sub_i32 s7, s1, s6 // (op 557) 


s_mul_i32 s1, s0, s14 // (op 557) 

s_sub_u32 s0, s16, s1 // (op 557) 

s_mul_i32 s1, s7, s14 // (op 557) 
s_add_u32 s1, s1, s0 // (op 557) 


s_mul_hi_i32 s0, s16, s8 // (op 557) 
s_add_i32 s51, s0, s16 // (op 557) 

s_ashr_i32 s0, s51, 31 // (op 557) 
s_cmp_eq_i32 s8, 0 // (op 557) 
s_mov_b32 s6, scc // (op 557) 
s_cmp_eq_i32 s6, 0 // (op 557) 
s_cselect_b32 s7, 0, -1 // (op 557) 

s_and_b32 s6, s15, 31 // (op 557) 
s_lshl_b32 s18, 1, s6 // (op 557) 
s_add_i32 s19, s18, s7 // (op 557) 


s_and_b32 s7, s0, s19 // (op 557) 


s_add_i32 s0, s51, s7 // (op 557) 


s_ashr_i32 s7, s0, s6 // (op 557) 


s_lshl_b32 s0, s15, 24 // (op 557) 
s_ashr_i32 s6, s0, 31 // (op 557) 

s_xor_b32 s0, s7, s6 // (op 557) 

s_sub_i32 s7, s0, s6 // (op 557) 


s_add_u32 s18, s4, 16 // (op 557) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 557) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 557) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 557) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 557) 

s_mul_i32 s0, s14, s18 // (op 557) 

s_mul_i32 s6, s50, s0 // (op 557) 

s_mul_i32 s0, s7, s6 // (op 557) 

s_sub_u32 s6, s16, s0 // (op 557) 

s_mov_b32 s0, s6 // (op 557) 
s_mul_hi_i32 s51, s0, s13 // (op 557) 
s_add_i32 s18, s51, s0 // (op 557) 


s_ashr_i32 s0, s18, 31 // (op 557) 
s_cmp_eq_i32 s13, 0 // (op 557) 
s_mov_b32 s19, scc // (op 557) 
s_cmp_eq_i32 s19, 0 // (op 557) 
s_cselect_b32 s51, 0, -1 // (op 557) 

s_and_b32 s19, s49, 31 // (op 557) 
s_lshl_b32 s68, 1, s19 // (op 557) 
s_add_i32 s69, s68, s51 // (op 557) 


s_and_b32 s51, s0, s69 // (op 557) 


s_add_i32 s0, s18, s51 // (op 557) 


s_ashr_i32 s18, s0, s19 // (op 557) 


s_lshl_b32 s0, s49, 24 // (op 557) 
s_ashr_i32 s19, s0, 31 // (op 557) 

s_xor_b32 s0, s18, s19 // (op 557) 

s_sub_i32 s18, s0, s19 // (op 557) 


s_mul_i32 s0, s48, s14 // (op 557) 
s_add_u32 s68, s20, 16 // (op 557) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 557) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 557) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 557) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 557) 

s_sub_i32 s19, s68, s0 // (op 557) 

s_mul_i32 s51, s18, s19 // (op 557) 


s_sub_u32 s18, s6, s51 // (op 557) 


s_add_u32 s6, s18, s0 // (op 557) 


s_cmp_eq_i32 s7, 0 // (op 557) 
s_mov_b32 s0, scc // (op 557) 

s_cmp_eq_i32 s0, 0 // (op 557) 
s_cselect_b32 s7, s6, s1 // (op 557) 



s_lshl_b32 s6, s7, 4 // (op 557) 

s_mov_b32 s1, 0 // (op 557) convert
s_mov_b32 s0, s6 // (op 557) convert

s_ashr_i64 s[6:7], s[24:25], 8 // (op 557) 
s_lshr_b32 s51, s6, 1 // (op 557) 

v_lshl_add_u32 v53, s51, 9, v49 // (op 557) 


v_mul_lo_u32 v23, v53, s3 // (op 557) most significant: low of low * high
v_mul_hi_u32 v21, v53, s2 // (op 557) most significant: high of low * low
v_mul_lo_u32 v22, v53, s2 // (op 557) least significant: low of low * low
v_add_u32 v23, v23, v21 // (op 557) most significant: sum


v_mov_b32 v21, s0 // (op 557) 
v_mov_b32 v49, s1 // (op 557) 
v_add_co_u32 v16, s[6:7], v21, v22 // (op 557) least significant half
v_addc_co_u32 v17, s[6:7], v49, v23, s[6:7] // (op 557) most significant half





v_mul_lo_u32 v15, v17, 6 // (op 557) most significant: low of high * low
v_mul_hi_u32 v13, v16, 6 // (op 557) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 557) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 557) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 557) 

v_mov_b32 v49, v16 // (op 557) convert

s_mul_i32 s1, s25, 6 // (op 559) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 559) least significant
s_mul_hi_u32 s51, s24, 6 // (op 559) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 559) most significant: sum

s_mul_i32 s7, s2, s1 // (op 559) most significant: low of low * high
s_mul_i32 s51, s3, s0 // (op 559) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 559) least significant
s_mul_hi_u32 s18, s2, s0 // (op 559) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 559) most significant: sum
s_add_u32 s7, s7, s18 // (op 559) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 559) 

s_mov_b32 s52, s22 // (op 559) reg expression
s_mov_b32 s53, s23 // (op 559) reg expression
s_mov_b32 s54, s0 // (op 559) reg expression
s_mov_b32 s55, 131072 // (op 559) call()

s_mul_hi_i32 s51, s16, s9 // (op 624) 
s_add_i32 s0, s51, s16 // (op 624) 

s_ashr_i32 s1, s0, 31 // (op 624) 
s_cmp_eq_i32 s9, 0 // (op 624) 
s_mov_b32 s51, scc // (op 624) 
s_cmp_eq_i32 s51, 0 // (op 624) 
s_cselect_b32 s6, 0, -1 // (op 624) 

s_and_b32 s7, s10, 31 // (op 624) 
s_lshl_b32 s51, 1, s7 // (op 624) 
s_add_i32 s18, s51, s6 // (op 624) 


s_and_b32 s6, s1, s18 // (op 624) 


s_add_i32 s1, s0, s6 // (op 624) 


s_ashr_i32 s0, s1, s7 // (op 624) 


s_lshl_b32 s1, s10, 24 // (op 624) 
s_ashr_i32 s51, s1, 31 // (op 624) 

s_xor_b32 s1, s0, s51 // (op 624) 

s_sub_i32 s0, s1, s51 // (op 624) 


s_mul_hi_i32 s1, s0, s11 // (op 624) 
s_add_i32 s51, s1, s0 // (op 624) 

s_ashr_i32 s1, s51, 31 // (op 624) 
s_cmp_eq_i32 s11, 0 // (op 624) 
s_mov_b32 s6, scc // (op 624) 
s_cmp_eq_i32 s6, 0 // (op 624) 
s_cselect_b32 s7, 0, -1 // (op 624) 

s_and_b32 s6, s12, 31 // (op 624) 
s_lshl_b32 s18, 1, s6 // (op 624) 
s_add_i32 s19, s18, s7 // (op 624) 


s_and_b32 s7, s1, s19 // (op 624) 


s_add_i32 s1, s51, s7 // (op 624) 


s_ashr_i32 s7, s1, s6 // (op 624) 


s_lshl_b32 s1, s12, 24 // (op 624) 
s_ashr_i32 s6, s1, 31 // (op 624) 

s_xor_b32 s1, s7, s6 // (op 624) 

s_sub_i32 s7, s1, s6 // (op 624) 


s_add_u32 s18, s4, 16 // (op 624) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 624) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 624) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 624) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 624) 

s_mul_i32 s1, s7, s18 // (op 624) 

s_sub_i32 s51, s0, s1 // (op 624) 


s_mul_hi_i32 s0, s16, s8 // (op 624) 
s_add_i32 s1, s0, s16 // (op 624) 

s_ashr_i32 s0, s1, 31 // (op 624) 
s_cmp_eq_i32 s8, 0 // (op 624) 
s_mov_b32 s6, scc // (op 624) 
s_cmp_eq_i32 s6, 0 // (op 624) 
s_cselect_b32 s7, 0, -1 // (op 624) 

s_and_b32 s6, s15, 31 // (op 624) 
s_lshl_b32 s68, 1, s6 // (op 624) 
s_add_i32 s69, s68, s7 // (op 624) 


s_and_b32 s7, s0, s69 // (op 624) 


s_add_i32 s0, s1, s7 // (op 624) 


s_ashr_i32 s1, s0, s6 // (op 624) 


s_lshl_b32 s0, s15, 24 // (op 624) 
s_ashr_i32 s6, s0, 31 // (op 624) 

s_xor_b32 s0, s1, s6 // (op 624) 

s_sub_i32 s1, s0, s6 // (op 624) 


s_mul_i32 s0, s14, s18 // (op 624) 

s_mul_i32 s6, s50, s0 // (op 624) 

s_mul_i32 s0, s1, s6 // (op 624) 

s_sub_u32 s6, s16, s0 // (op 624) 

s_mul_hi_i32 s0, s6, s13 // (op 624) 
s_add_i32 s7, s0, s6 // (op 624) 


s_ashr_i32 s0, s7, 31 // (op 624) 
s_cmp_eq_i32 s13, 0 // (op 624) 
s_mov_b32 s6, scc // (op 624) 
s_cmp_eq_i32 s6, 0 // (op 624) 
s_cselect_b32 s18, 0, -1 // (op 624) 

s_and_b32 s6, s49, 31 // (op 624) 
s_lshl_b32 s19, 1, s6 // (op 624) 
s_add_i32 s68, s19, s18 // (op 624) 


s_and_b32 s18, s0, s68 // (op 624) 


s_add_i32 s0, s7, s18 // (op 624) 


s_ashr_i32 s7, s0, s6 // (op 624) 


s_lshl_b32 s0, s49, 24 // (op 624) 
s_ashr_i32 s6, s0, 31 // (op 624) 

s_xor_b32 s0, s7, s6 // (op 624) 

s_sub_i32 s7, s0, s6 // (op 624) 


s_cmp_eq_i32 s1, 0 // (op 624) 
s_mov_b32 s0, scc // (op 624) 

s_cmp_eq_i32 s0, 0 // (op 624) 
s_cselect_b32 s1, s7, s51 // (op 624) 



s_lshl_b32 s0, s1, 4 // (op 624) 

s_mov_b32 s7, 0 // (op 624) convert
s_mov_b32 s6, s0 // (op 624) convert

s_ashr_i64 s[0:1], s[24:25], 8 // (op 624) 
s_lshr_b32 s51, s0, 1 // (op 624) 

v_lshl_add_u32 v53, s51, 9, v1 // (op 624) 

v_mul_lo_u32 v23, v53, s27 // (op 624) most significant: low of low * high
v_mul_hi_u32 v21, v53, s26 // (op 624) most significant: high of low * low
v_mul_lo_u32 v22, v53, s26 // (op 624) least significant: low of low * low
v_add_u32 v23, v23, v21 // (op 624) most significant: sum

v_mov_b32 v21, s6 // (op 624) 
v_mov_b32 v57, s7 // (op 624) 
v_add_co_u32 v22, s[0:1], v22, v21 // (op 624) least significant half
v_addc_co_u32 v23, s[0:1], v23, v57, s[0:1] // (op 624) most significant half





s_mul_i32 s1, s26, s25 // (op 626) most significant: low of low * high
s_mul_i32 s51, s27, s24 // (op 626) most significant: low of high * low
s_mul_i32 s0, s26, s24 // (op 626) least significant
s_mul_hi_u32 s6, s26, s24 // (op 626) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 626) most significant: sum
s_add_u32 s1, s1, s6 // (op 626) most significant: sum


s_mov_b32 s56, s28 // (op 626) reg expression
s_mov_b32 s57, s29 // (op 626) reg expression
s_mov_b32 s58, s0 // (op 626) reg expression
s_mov_b32 s59, 131072 // (op 626) call()

s_mul_hi_i32 s51, s16, s9 // (op 688) 
v_and_b32 v21, -16, v1 // (op 688) 
s_add_i32 s0, s51, s16 // (op 688) 

v_sub_u32 v53, v1, v21 // (op 688) 

s_ashr_i32 s1, s0, 31 // (op 688) 
v_lshrrev_b32 v21, 4, v1 // (op 688) 
s_cmp_eq_i32 s9, 0 // (op 688) 
s_mov_b32 s51, scc // (op 688) 
v_lshlrev_b32 v57, 1, v21 // (op 688) 

s_cmp_eq_i32 s51, 0 // (op 688) 
s_cselect_b32 s6, 0, -1 // (op 688) 

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


s_mul_i32 s1, s0, s14 // (op 688) 

s_sub_u32 s0, s16, s1 // (op 688) 

s_mul_i32 s1, s7, s14 // (op 688) 
s_add_u32 s1, s1, s0 // (op 688) 


s_mul_hi_i32 s0, s16, s8 // (op 688) 
s_add_i32 s51, s0, s16 // (op 688) 

s_ashr_i32 s0, s51, 31 // (op 688) 
s_cmp_eq_i32 s8, 0 // (op 688) 
s_mov_b32 s6, scc // (op 688) 
s_cmp_eq_i32 s6, 0 // (op 688) 
s_cselect_b32 s7, 0, -1 // (op 688) 

s_and_b32 s6, s15, 31 // (op 688) 
s_lshl_b32 s18, 1, s6 // (op 688) 
s_add_i32 s19, s18, s7 // (op 688) 


s_and_b32 s7, s0, s19 // (op 688) 


s_add_i32 s0, s51, s7 // (op 688) 


s_ashr_i32 s7, s0, s6 // (op 688) 


s_lshl_b32 s0, s15, 24 // (op 688) 
s_ashr_i32 s6, s0, 31 // (op 688) 

s_xor_b32 s0, s7, s6 // (op 688) 

s_sub_i32 s7, s0, s6 // (op 688) 


s_add_u32 s18, s4, 16 // (op 688) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 688) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 688) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 688) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 688) 

s_mul_i32 s0, s14, s18 // (op 688) 

s_mul_i32 s6, s50, s0 // (op 688) 

s_mul_i32 s0, s7, s6 // (op 688) 

s_sub_u32 s6, s16, s0 // (op 688) 

s_mov_b32 s0, s6 // (op 688) 
s_mul_hi_i32 s51, s0, s13 // (op 688) 
s_add_i32 s18, s51, s0 // (op 688) 


s_ashr_i32 s0, s18, 31 // (op 688) 
s_cmp_eq_i32 s13, 0 // (op 688) 
s_mov_b32 s19, scc // (op 688) 
s_cmp_eq_i32 s19, 0 // (op 688) 
s_cselect_b32 s51, 0, -1 // (op 688) 

s_and_b32 s19, s49, 31 // (op 688) 
s_lshl_b32 s68, 1, s19 // (op 688) 
s_add_i32 s69, s68, s51 // (op 688) 


s_and_b32 s51, s0, s69 // (op 688) 


s_add_i32 s0, s18, s51 // (op 688) 


s_ashr_i32 s18, s0, s19 // (op 688) 


s_lshl_b32 s0, s49, 24 // (op 688) 
s_ashr_i32 s19, s0, 31 // (op 688) 

s_xor_b32 s0, s18, s19 // (op 688) 

s_sub_i32 s18, s0, s19 // (op 688) 


s_mul_i32 s0, s48, s14 // (op 688) 
s_add_u32 s68, s20, 16 // (op 688) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 688) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 688) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 688) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 688) 

s_sub_i32 s19, s68, s0 // (op 688) 

s_mul_i32 s51, s18, s19 // (op 688) 


s_sub_u32 s18, s6, s51 // (op 688) 


s_add_u32 s6, s18, s0 // (op 688) 


s_cmp_eq_i32 s7, 0 // (op 688) 
s_mov_b32 s0, scc // (op 688) 

s_cmp_eq_i32 s0, 0 // (op 688) 
s_cselect_b32 s7, s6, s1 // (op 688) 



v_lshl_add_u32 v21, s7, 4, v53 // (op 688) 


s_ashr_i64 s[0:1], s[24:25], 8 // (op 688) 
s_lshr_b32 s51, s0, 1 // (op 688) 

v_lshl_add_u32 v53, s51, 4, v57 // (op 688) 


v_mul_lo_u32 v57, v53, s30 // (op 688) 

v_add_u32 v53, v21, v57 // (op 688) 


s_mul_i32 s1, s30, s35 // (op 690) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 690) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 690) least significant
s_mul_hi_u32 s6, s30, s34 // (op 690) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 690) most significant: sum
s_add_u32 s1, s1, s6 // (op 690) most significant: sum


s_mov_b32 s60, s32 // (op 690) reg expression
s_mov_b32 s61, s33 // (op 690) reg expression
s_mov_b32 s62, s0 // (op 690) reg expression
s_mov_b32 s63, 131072 // (op 690) call()

s_mul_hi_i32 s51, s16, s9 // (op 752) 
v_and_b32 v21, -16, v1 // (op 752) 
s_add_i32 s0, s51, s16 // (op 752) 

v_sub_u32 v57, v1, v21 // (op 752) 

s_ashr_i32 s1, s0, 31 // (op 752) 
v_lshrrev_b32 v21, 4, v1 // (op 752) 
s_cmp_eq_i32 s9, 0 // (op 752) 
s_mov_b32 s51, scc // (op 752) 
v_lshlrev_b32 v61, 1, v21 // (op 752) 

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


s_add_u32 s18, s4, 16 // (op 752) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 752) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 752) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 752) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 752) 

s_mul_i32 s1, s7, s18 // (op 752) 

s_sub_i32 s51, s0, s1 // (op 752) 


s_mul_hi_i32 s0, s16, s8 // (op 752) 
s_add_i32 s1, s0, s16 // (op 752) 

s_ashr_i32 s0, s1, 31 // (op 752) 
s_cmp_eq_i32 s8, 0 // (op 752) 
s_mov_b32 s6, scc // (op 752) 
s_cmp_eq_i32 s6, 0 // (op 752) 
s_cselect_b32 s7, 0, -1 // (op 752) 

s_and_b32 s6, s15, 31 // (op 752) 
s_lshl_b32 s68, 1, s6 // (op 752) 
s_add_i32 s69, s68, s7 // (op 752) 


s_and_b32 s7, s0, s69 // (op 752) 


s_add_i32 s0, s1, s7 // (op 752) 


s_ashr_i32 s1, s0, s6 // (op 752) 


s_lshl_b32 s0, s15, 24 // (op 752) 
s_ashr_i32 s6, s0, 31 // (op 752) 

s_xor_b32 s0, s1, s6 // (op 752) 

s_sub_i32 s1, s0, s6 // (op 752) 


s_mul_i32 s0, s14, s18 // (op 752) 

s_mul_i32 s6, s50, s0 // (op 752) 

s_mul_i32 s0, s1, s6 // (op 752) 

s_sub_u32 s6, s16, s0 // (op 752) 

s_mul_hi_i32 s0, s6, s13 // (op 752) 
s_add_i32 s7, s0, s6 // (op 752) 


s_ashr_i32 s0, s7, 31 // (op 752) 
s_cmp_eq_i32 s13, 0 // (op 752) 
s_mov_b32 s6, scc // (op 752) 
s_cmp_eq_i32 s6, 0 // (op 752) 
s_cselect_b32 s18, 0, -1 // (op 752) 

s_and_b32 s6, s49, 31 // (op 752) 
s_lshl_b32 s19, 1, s6 // (op 752) 
s_add_i32 s68, s19, s18 // (op 752) 


s_and_b32 s18, s0, s68 // (op 752) 


s_add_i32 s0, s7, s18 // (op 752) 


s_ashr_i32 s7, s0, s6 // (op 752) 


s_lshl_b32 s0, s49, 24 // (op 752) 
s_ashr_i32 s6, s0, 31 // (op 752) 

s_xor_b32 s0, s7, s6 // (op 752) 

s_sub_i32 s7, s0, s6 // (op 752) 


s_cmp_eq_i32 s1, 0 // (op 752) 
s_mov_b32 s0, scc // (op 752) 

s_cmp_eq_i32 s0, 0 // (op 752) 
s_cselect_b32 s1, s7, s51 // (op 752) 



v_lshl_add_u32 v21, s1, 4, v57 // (op 752) 


s_ashr_i64 s[0:1], s[24:25], 8 // (op 752) 
s_lshr_b32 s51, s0, 1 // (op 752) 

v_lshl_add_u32 v57, s51, 4, v61 // (op 752) 


v_mul_lo_u32 v61, v57, s36 // (op 752) 
v_add_u32 v61, v61, v21 // (op 752) 


s_mul_i32 s1, s36, s35 // (op 754) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 754) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 754) least significant
s_mul_hi_u32 s6, s36, s34 // (op 754) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 754) most significant: sum
s_add_u32 s1, s1, s6 // (op 754) most significant: sum


s_mov_b32 s64, s38 // (op 754) reg expression
s_mov_b32 s65, s39 // (op 754) reg expression
s_mov_b32 s66, s0 // (op 754) reg expression
s_mov_b32 s67, 131072 // (op 754) call()

v_lshlrev_b32 v21, 6, v1 // (op 845) 
v_mul_lo_u32 v57, v21, 6 // (op 845) 
v_lshrrev_b32 v13, 3, v57 // (op 845) 

v_lshrrev_b32 v57, 4, v21 // (op 845) 

v_lshlrev_b32 v21, 2, v57 // (op 845) 

v_add_u32 v57, v13, v21 // (op 845) 


v_lshlrev_b32 v21, 4, v1 // (op 898) 
s_nop 0
v_readfirstlane_b32 s51, v21 // (op 898) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -16, v1 // (op 950) 
v_sub_u32 v13, v1, v21 // (op 950) 

v_lshrrev_b32 v21, 4, v1 // (op 950) 
v_lshl_add_u32 v14, v21, 5, v13 // (op 950) 


v_and_b32 v13, -16, v1 // (op 1003) 
v_sub_u32 v21, v1, v13 // (op 1003) 

v_lshrrev_b32 v13, 4, v1 // (op 1003) 
v_lshl_add_u32 v15, v13, 5, v21 // (op 1003) 


s_ashr_i64 s[0:1], s[24:25], 8 // (op 194) 
s_and_b32 s6, s0, 4294967294 // (op 194) 

s_ashr_i64 s[0:1], s[24:25], 8 // (op 193) 
s_mov_b32 s7, s0 // (op 193) convert

s_cmp_lt_u32 s6, s7 // (op 193) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193 // (op 193) Condition: Top (jump to RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193:

s_add_u32 s6, s6, 1 // (op 195) 
s_mul_i32 s1, s3, 6 // (op 198) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 198) least significant
s_mul_hi_u32 s7, s2, 6 // (op 198) most significant: high of low * low
s_add_u32 s1, s1, s7 // (op 198) most significant: sum

v_lshrrev_b64 v[12:13], 3, s[0:1] // (op 198) 


v_mov_b32 v16, v49 // (op 198) 
v_mov_b32 v17, v16 // (op 198) reg expression
buffer_load_dwordx3 v[18:20], v17, s[52:55], 0 offen // (op 198) Load value
v_add_u32 v17, v17, v12 // (op 198) 
buffer_load_dwordx3 v[24:26], v17, s[52:55], 0 offen // (op 198) Load value
v_add_u32 v17, v17, v12 // (op 198) 
buffer_load_dwordx3 v[28:30], v17, s[52:55], 0 offen // (op 198) Load value
v_add_u32 v17, v17, v12 // (op 198) 
buffer_load_dwordx3 v[32:34], v17, s[52:55], 0 offen // (op 198) Load value



s_lshl_b64 s[0:1], s[2:3], 8 // (op 547) 
s_mul_i32 s19, s1, 6 // (op 547) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 547) least significant
s_mul_hi_u32 s7, s0, 6 // (op 547) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 547) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 547) 

v_add_u32 v49, v49, s0 // (op 547) 

s_barrier  // (op 1219) 
s_barrier  // (op 1238) 




s_lshl_b64 s[0:1], s[26:27], 6 // (op 527) 
v_mov_b64 v[12:13], s[0:1] // (op 527) convert


v_mov_b64 v[16:17], v[22:23] // (op 527) 
v_mov_b64 v[50:51], v[16:17] // (op 527) reg expression
s_mov_b32 m0, s51 // (op 527) reg expression
buffer_load_dwordx4 v50, s[56:59], 0 offen offset: 0 lds // (op 527) Load value direct to lds
v_add_co_u32 v50, s[0:1], v50, v12 // (op 527) least significant half
v_addc_co_u32 v51, s[0:1], 0, v51, s[0:1] // (op 527) most significant half

s_add_u32 m0, m0, 1024 // (op 527) 
buffer_load_dwordx4 v50, s[56:59], 0 offen offset: 0 lds // (op 527) Load value direct to lds
v_add_co_u32 v50, s[0:1], v50, v12 // (op 527) least significant half
v_addc_co_u32 v51, s[0:1], 0, v51, s[0:1] // (op 527) most significant half

s_add_u32 m0, m0, 1024 // (op 527) 
s_waitcnt vmcnt(5)
ds_write_b96 v57, v[18:20] offset:8320 // (op 199) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v57, v[24:26] offset:8336 // (op 199) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v57, v[28:30] offset:8352 // (op 199) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v57, v[32:34] offset:8368 // (op 199) Store local data 
buffer_load_dwordx4 v50, s[56:59], 0 offen offset: 0 lds // (op 527) Load value direct to lds

v_add_co_u32 v50, s[0:1], v50, v12 // (op 527) least significant half
v_addc_co_u32 v51, s[0:1], 0, v51, s[0:1] // (op 527) most significant half

s_add_u32 m0, m0, 1024 // (op 527) 
buffer_load_dwordx4 v50, s[56:59], 0 offen offset: 0 lds // (op 527) Load value direct to lds



s_lshl_b64 s[0:1], s[26:27], 8 // (op 616) 
v_mov_b32 v12, s0 // (op 616) 
v_mov_b32 v13, s1 // (op 616) 
v_add_co_u32 v22, s[18:19], v22, v12 // (op 616) least significant half
v_addc_co_u32 v23, s[18:19], v23, v13, s[18:19] // (op 616) most significant half





v_mov_b64 v[12:13], s[30:31] // (op 222) convert

v_mov_b32 v16, v53 // (op 222) 
v_mov_b32 v17, v16 // (op 222) reg expression
buffer_load_ubyte v18, v17, s[60:63], 0 offen offset: 0 // (op 222) Load value
v_add_u32 v17, v17, v12 // (op 222) 
buffer_load_ubyte v19, v17, s[60:63], 0 offen offset: 0 // (op 222) Load value



s_lshl_b32 s7, s30, 3 // (op 680) 
v_add_u32 v53, v53, s7 // (op 680) 


s_barrier  // (op 1254) 
v_mov_b64 v[12:13], s[36:37] // (op 223) convert




v_mov_b32 v16, v61 // (op 223) 
v_mov_b32 v17, v16 // (op 223) reg expression
buffer_load_ubyte v20, v17, s[64:67], 0 offen offset: 0 // (op 223) Load value
v_add_u32 v17, v17, v12 // (op 223) 
buffer_load_ubyte v21, v17, s[64:67], 0 offen offset: 0 // (op 223) Load value



s_lshl_b32 s7, s36, 3 // (op 744) 
v_add_u32 v61, v61, s7 // (op 744) 

s_waitcnt vmcnt(3)
ds_write_b8 v14, v18 offset:12416 // (op 219) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1228) 
s_waitcnt vmcnt(2)
ds_write_b8 v14, v19 offset:12432 // (op 219) Store local data 




ds_read_b96_tr_b6 v[24:26], v0 offset:8320 // (op 201) Transpose load local data 
s_barrier  // (op 1274) 
ds_read_b96_tr_b6 v[27:29], v0 offset:8576 // (op 201) Transpose load local data 



s_barrier  // (op 1244) 
s_waitcnt vmcnt(1)
ds_write_b8 v15, v20 offset:8192 // (op 220) Store local data 



ds_read_b64_tr_b8 v[30:31], v3 // (op 203) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v3 offset:128 // (op 203) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:1024 // (op 203) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v3 offset:1152 // (op 203) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v15, v21 offset:8208 // (op 220) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 1264) 


ds_read_u8 v12, v5 offset:12416 // (op 205) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1284) 


ds_read_u8 v13, v7 offset:8192 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[30:37], a[0:3], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 208) 







ds_read_b96_tr_b6 v[16:18], v0 offset:10368 // (op 210) Transpose load local data 
ds_read_b96_tr_b6 v[19:21], v0 offset:10624 // (op 210) Transpose load local data 



ds_read_b64_tr_b8 v[24:25], v3 offset:2048 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v3 offset:2176 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v3 offset:3072 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v3 offset:3200 // (op 212) Transpose load local data 


ds_read_u8 v12, v5 offset:12480 // (op 214) Load local data 


ds_read_u8 v13, v7 offset:8256 // (op 216) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:21], v[24:31], a[0:3], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 217) 




s_ashr_i64 s[0:1], s[24:25], 8 // (op 193) 
s_mov_b32 s7, s0 // (op 193) convert

s_cmp_lt_u32 s6, s7 // (op 193) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193 // (op 193) Condition: Bottom (jump to RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopTop_KLoopTail_193 if true)
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ForLoopBottom_KLoopTail_193:























s_barrier  // (op 1360) 








v_cmp_eq_f32 vcc, 0.00000, v8 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v10, v4 // (op 50) 
v_mul_f32 v13, v10, v5 // (op 50) 
v_mul_f32 v14, v10, v6 // (op 50) 
v_mul_f32 v15, v10, v7 // (op 50) 

s_mul_hi_i32 s0, s16, s9 // (op 1045) 
v_lshrrev_b32 v0, 6, v1 // (op 1045) 
s_add_i32 s1, s0, s16 // (op 1045) 

v_and_b32 v9, 63, v1 // (op 1045) 
s_ashr_i32 s0, s1, 31 // (op 1045) 
v_lshrrev_b32 v11, 4, v9 // (op 1045) 
s_cmp_eq_i32 s9, 0 // (op 1045) 
s_mov_b32 s2, scc // (op 1045) 
v_lshlrev_b32 v3, 2, v11 // (op 1045) 

s_cmp_eq_i32 s2, 0 // (op 1045) 
s_cselect_b32 s3, 0, -1 // (op 1045) 

v_lshl_add_u32 v11, v0, 4, v3 // (op 1045) 


s_and_b32 s2, s10, 31 // (op 1045) 
v_lshrrev_b32 v0, 2, v9 // (op 1045) 

s_lshl_b32 s6, 1, s2 // (op 1045) 
v_and_b32 v9, 3, v0 // (op 1045) 

s_add_i32 s7, s6, s3 // (op 1045) 


v_and_b32 v0, 3, v1 // (op 1045) 
s_and_b32 s3, s0, s7 // (op 1045) 


v_lshl_add_u32 v3, v9, 2, v0 // (op 1045) 


s_add_i32 s0, s1, s3 // (op 1045) 


s_ashr_i32 s1, s0, s2 // (op 1045) 


s_lshl_b32 s0, s10, 24 // (op 1045) 
s_ashr_i32 s2, s0, 31 // (op 1045) 

s_xor_b32 s0, s1, s2 // (op 1045) 

s_sub_i32 s1, s0, s2 // (op 1045) 


s_mul_hi_i32 s0, s1, s11 // (op 1045) 
s_add_i32 s2, s0, s1 // (op 1045) 

s_ashr_i32 s0, s2, 31 // (op 1045) 
s_cmp_eq_i32 s11, 0 // (op 1045) 
s_mov_b32 s3, scc // (op 1045) 
s_cmp_eq_i32 s3, 0 // (op 1045) 
s_cselect_b32 s6, 0, -1 // (op 1045) 

s_and_b32 s3, s12, 31 // (op 1045) 
s_lshl_b32 s7, 1, s3 // (op 1045) 
s_add_i32 s18, s7, s6 // (op 1045) 


s_and_b32 s19, s0, s18 // (op 1045) 


s_add_i32 s0, s2, s19 // (op 1045) 


s_ashr_i32 s2, s0, s3 // (op 1045) 


s_lshl_b32 s0, s12, 24 // (op 1045) 
s_ashr_i32 s3, s0, 31 // (op 1045) 

s_xor_b32 s0, s2, s3 // (op 1045) 

s_sub_i32 s2, s0, s3 // (op 1045) 


s_mul_i32 s0, s1, s14 // (op 1045) 
s_sub_u32 s3, s16, s0 // (op 1045) 

s_mul_i32 s0, s2, s14 // (op 1045) 
s_add_u32 s0, s0, s3 // (op 1045) 

s_mul_hi_i32 s3, s16, s8 // (op 1045) 
s_add_i32 s6, s3, s16 // (op 1045) 

s_ashr_i32 s3, s6, 31 // (op 1045) 
s_cmp_eq_i32 s8, 0 // (op 1045) 
s_mov_b32 s7, scc // (op 1045) 
s_cmp_eq_i32 s7, 0 // (op 1045) 
s_cselect_b32 s18, 0, -1 // (op 1045) 

s_and_b32 s7, s15, 31 // (op 1045) 
s_lshl_b32 s19, 1, s7 // (op 1045) 
s_add_i32 s22, s19, s18 // (op 1045) 


s_and_b32 s18, s3, s22 // (op 1045) 


s_add_i32 s3, s6, s18 // (op 1045) 


s_ashr_i32 s6, s3, s7 // (op 1045) 


s_lshl_b32 s3, s15, 24 // (op 1045) 
s_ashr_i32 s7, s3, 31 // (op 1045) 

s_xor_b32 s3, s6, s7 // (op 1045) 

s_sub_i32 s6, s3, s7 // (op 1045) 


s_add_u32 s18, s4, 16 // (op 1045) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 1045) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 1045) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 1045) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 4 // (op 1045) 

s_mul_i32 s3, s14, s18 // (op 1045) 
s_mul_i32 s7, s50, s3 // (op 1045) 

s_mul_i32 s3, s6, s7 // (op 1045) 

s_sub_u32 s7, s16, s3 // (op 1045) 

s_mov_b32 s3, s7 // (op 1045) 
s_mul_hi_i32 s22, s3, s13 // (op 1045) 
s_add_i32 s23, s22, s3 // (op 1045) 


s_ashr_i32 s3, s23, 31 // (op 1045) 
s_cmp_eq_i32 s13, 0 // (op 1045) 
s_mov_b32 s22, scc // (op 1045) 
s_cmp_eq_i32 s22, 0 // (op 1045) 
s_cselect_b32 s24, 0, -1 // (op 1045) 

s_and_b32 s22, s49, 31 // (op 1045) 
s_lshl_b32 s25, 1, s22 // (op 1045) 
s_add_i32 s26, s25, s24 // (op 1045) 


s_and_b32 s24, s3, s26 // (op 1045) 


s_add_i32 s3, s23, s24 // (op 1045) 


s_ashr_i32 s23, s3, s22 // (op 1045) 


s_lshl_b32 s3, s49, 24 // (op 1045) 
s_ashr_i32 s22, s3, 31 // (op 1045) 

s_xor_b32 s3, s23, s22 // (op 1045) 

s_sub_i32 s23, s3, s22 // (op 1045) 


s_mul_i32 s3, s48, s14 // (op 1045) 
s_add_u32 s24, s20, 16 // (op 1045) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1045) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 1045) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 1045) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 4 // (op 1045) 

s_sub_i32 s22, s24, s3 // (op 1045) 

s_mul_i32 s24, s23, s22 // (op 1045) 

s_sub_u32 s22, s7, s24 // (op 1045) 


s_add_u32 s7, s22, s3 // (op 1045) 


s_cmp_eq_i32 s6, 0 // (op 1045) 
s_mov_b32 s3, scc // (op 1045) 

s_cmp_eq_i32 s3, 0 // (op 1045) 
s_cselect_b32 s6, s7, s0 // (op 1045) 


s_mul_i32 s0, s2, s18 // (op 1045) 


v_lshl_add_u32 v0, s6, 4, v11 // (op 1045) 


s_sub_i32 s2, s1, s0 // (op 1045) 


s_cmp_eq_i32 s3, 0 // (op 1045) 
s_cselect_b32 s22, s23, s2 // (op 1045) 



v_lshl_add_u32 v9, s22, 4, v3 // (op 1045) 


v_mul_lo_u32 v11, v9, s40 // (op 1045) 

v_add_lshl_u32 v9, v0, v11, 2 // (op 1045) 


s_mul_i32 s7, s40, s5 // (op 1047) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 1047) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 1047) least significant
s_mul_hi_u32 s1, s40, s4 // (op 1047) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 1047) most significant: sum
s_add_u32 s7, s7, s1 // (op 1047) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 1047) 

s_mov_b32 s0, s42 // (op 1047) reg expression
s_mov_b32 s1, s43 // (op 1047) reg expression
s_mov_b32 s2, s18 // (op 1047) reg expression
s_mov_b32 s3, 131072 // (op 1047) call()




buffer_store_dwordx4 v[12:15], v9, s[0:3], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalFalse_0_beta_155:
s_mul_hi_i32 s6, s16, s9 // (op 815) 
v_lshrrev_b32 v0, 6, v1 // (op 815) 
s_add_i32 s7, s6, s16 // (op 815) 

v_and_b32 v9, 63, v1 // (op 815) 
s_ashr_i32 s6, s7, 31 // (op 815) 
v_lshrrev_b32 v3, 4, v9 // (op 815) 
s_cmp_eq_i32 s9, 0 // (op 815) 
s_mov_b32 s18, scc // (op 815) 
v_lshlrev_b32 v4, 2, v3 // (op 815) 

s_cmp_eq_i32 s18, 0 // (op 815) 
s_cselect_b32 s19, 0, -1 // (op 815) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 815) 


s_and_b32 s18, s10, 31 // (op 815) 
v_lshrrev_b32 v0, 2, v9 // (op 815) 

s_lshl_b32 s22, 1, s18 // (op 815) 
v_and_b32 v9, 3, v0 // (op 815) 

s_add_i32 s23, s22, s19 // (op 815) 


v_and_b32 v0, 3, v1 // (op 815) 
s_and_b32 s19, s6, s23 // (op 815) 


v_lshl_add_u32 v4, v9, 2, v0 // (op 815) 


s_add_i32 s6, s7, s19 // (op 815) 


s_ashr_i32 s7, s6, s18 // (op 815) 


s_lshl_b32 s6, s10, 24 // (op 815) 
s_ashr_i32 s18, s6, 31 // (op 815) 

s_xor_b32 s6, s7, s18 // (op 815) 

s_sub_i32 s7, s6, s18 // (op 815) 


s_mul_hi_i32 s6, s7, s11 // (op 815) 
s_add_i32 s18, s6, s7 // (op 815) 

s_ashr_i32 s6, s18, 31 // (op 815) 
s_cmp_eq_i32 s11, 0 // (op 815) 
s_mov_b32 s19, scc // (op 815) 
s_cmp_eq_i32 s19, 0 // (op 815) 
s_cselect_b32 s22, 0, -1 // (op 815) 

s_and_b32 s19, s12, 31 // (op 815) 
s_lshl_b32 s23, 1, s19 // (op 815) 
s_add_i32 s24, s23, s22 // (op 815) 


s_and_b32 s22, s6, s24 // (op 815) 


s_add_i32 s6, s18, s22 // (op 815) 


s_ashr_i32 s18, s6, s19 // (op 815) 


s_lshl_b32 s6, s12, 24 // (op 815) 
s_ashr_i32 s19, s6, 31 // (op 815) 

s_xor_b32 s6, s18, s19 // (op 815) 

s_sub_i32 s18, s6, s19 // (op 815) 


s_mul_i32 s6, s7, s14 // (op 815) 
s_sub_u32 s19, s16, s6 // (op 815) 

s_mul_i32 s6, s18, s14 // (op 815) 
s_add_u32 s6, s6, s19 // (op 815) 

s_mul_hi_i32 s19, s16, s8 // (op 815) 
s_add_i32 s22, s19, s16 // (op 815) 

s_ashr_i32 s19, s22, 31 // (op 815) 
s_cmp_eq_i32 s8, 0 // (op 815) 
s_mov_b32 s23, scc // (op 815) 
s_cmp_eq_i32 s23, 0 // (op 815) 
s_cselect_b32 s24, 0, -1 // (op 815) 

s_and_b32 s23, s15, 31 // (op 815) 
s_lshl_b32 s25, 1, s23 // (op 815) 
s_add_i32 s26, s25, s24 // (op 815) 


s_and_b32 s24, s19, s26 // (op 815) 


s_add_i32 s19, s22, s24 // (op 815) 


s_ashr_i32 s22, s19, s23 // (op 815) 


s_lshl_b32 s19, s15, 24 // (op 815) 
s_ashr_i32 s23, s19, 31 // (op 815) 

s_xor_b32 s19, s22, s23 // (op 815) 

s_sub_i32 s22, s19, s23 // (op 815) 


s_add_u32 s38, s4, 16 // (op 815) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 815) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 815) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 815) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 815) 

s_mul_i32 s19, s14, s38 // (op 815) 
s_mul_i32 s23, s50, s19 // (op 815) 

s_mul_i32 s19, s22, s23 // (op 815) 

s_sub_u32 s23, s16, s19 // (op 815) 

s_mov_b32 s19, s23 // (op 815) 
s_mul_hi_i32 s24, s19, s13 // (op 815) 
s_add_i32 s25, s24, s19 // (op 815) 


s_ashr_i32 s19, s25, 31 // (op 815) 
s_cmp_eq_i32 s13, 0 // (op 815) 
s_mov_b32 s24, scc // (op 815) 
s_cmp_eq_i32 s24, 0 // (op 815) 
s_cselect_b32 s26, 0, -1 // (op 815) 

s_and_b32 s24, s49, 31 // (op 815) 
s_lshl_b32 s27, 1, s24 // (op 815) 
s_add_i32 s28, s27, s26 // (op 815) 


s_and_b32 s26, s19, s28 // (op 815) 


s_add_i32 s19, s25, s26 // (op 815) 


s_ashr_i32 s25, s19, s24 // (op 815) 


s_lshl_b32 s19, s49, 24 // (op 815) 
s_ashr_i32 s24, s19, 31 // (op 815) 

s_xor_b32 s19, s25, s24 // (op 815) 

s_sub_i32 s25, s19, s24 // (op 815) 


s_mul_i32 s19, s48, s14 // (op 815) 
s_add_u32 s26, s20, 16 // (op 815) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 815) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 815) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 815) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 815) 

s_sub_i32 s24, s26, s19 // (op 815) 

s_mul_i32 s26, s25, s24 // (op 815) 

s_sub_u32 s24, s23, s26 // (op 815) 


s_add_u32 s23, s24, s19 // (op 815) 


s_cmp_eq_i32 s22, 0 // (op 815) 
s_mov_b32 s19, scc // (op 815) 

s_cmp_eq_i32 s19, 0 // (op 815) 
s_cselect_b32 s22, s23, s6 // (op 815) 


s_mul_i32 s6, s18, s38 // (op 815) 


v_lshl_add_u32 v0, s22, 4, v3 // (op 815) 


s_sub_i32 s18, s7, s6 // (op 815) 


s_cmp_eq_i32 s19, 0 // (op 815) 
s_cselect_b32 s6, s25, s18 // (op 815) 



v_lshl_add_u32 v3, s6, 4, v4 // (op 815) 


v_mul_lo_u32 v9, v3, s44 // (op 815) 

v_add_lshl_u32 v3, v0, v9, 2 // (op 815) 


s_mul_i32 s7, s44, s5 // (op 817) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 817) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 817) least significant
s_mul_hi_u32 s19, s44, s4 // (op 817) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 817) most significant: sum
s_add_u32 s7, s7, s19 // (op 817) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 817) 

s_mov_b32 s36, s46 // (op 817) reg expression
s_mov_b32 s37, s47 // (op 817) reg expression
s_mov_b32 s38, s18 // (op 817) reg expression
s_mov_b32 s39, 131072 // (op 817) call()





buffer_load_dwordx4 v[4:7], v3, s[36:39], 0 offen // (op 167) Load value



v_accvgpr_read v12, a0 // (op 164) 
v_accvgpr_read v13, a1 // (op 164) 
v_accvgpr_read v14, a2 // (op 164) 
v_accvgpr_read v15, a3 // (op 164) 
v_mul_f32 v16, v10, v12 // (op 164) 
v_mul_f32 v17, v10, v13 // (op 164) 
v_mul_f32 v18, v10, v14 // (op 164) 
v_mul_f32 v19, v10, v15 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v8, v4 // (op 168) 
v_mul_f32 v13, v8, v5 // (op 168) 
v_mul_f32 v14, v8, v6 // (op 168) 
v_mul_f32 v15, v8, v7 // (op 168) 

v_add_f32 v4, v12, v16 // (op 165) 
v_add_f32 v5, v13, v17 // (op 165) 
v_add_f32 v6, v14, v18 // (op 165) 
v_add_f32 v7, v15, v19 // (op 165) 


s_mul_hi_i32 s6, s16, s9 // (op 1062) 
v_lshrrev_b32 v0, 6, v1 // (op 1062) 
s_add_i32 s7, s6, s16 // (op 1062) 

v_and_b32 v3, 63, v1 // (op 1062) 
s_ashr_i32 s6, s7, 31 // (op 1062) 
v_lshrrev_b32 v9, 4, v3 // (op 1062) 
s_cmp_eq_i32 s9, 0 // (op 1062) 
s_mov_b32 s18, scc // (op 1062) 
v_lshlrev_b32 v11, 2, v9 // (op 1062) 

s_cmp_eq_i32 s18, 0 // (op 1062) 
s_cselect_b32 s19, 0, -1 // (op 1062) 

v_lshl_add_u32 v9, v0, 4, v11 // (op 1062) 


s_and_b32 s18, s10, 31 // (op 1062) 
v_lshrrev_b32 v0, 2, v3 // (op 1062) 

s_lshl_b32 s22, 1, s18 // (op 1062) 
v_and_b32 v3, 3, v0 // (op 1062) 

s_add_i32 s23, s22, s19 // (op 1062) 


v_and_b32 v0, 3, v1 // (op 1062) 
s_and_b32 s19, s6, s23 // (op 1062) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 1062) 


s_add_i32 s6, s7, s19 // (op 1062) 


s_ashr_i32 s7, s6, s18 // (op 1062) 


s_lshl_b32 s6, s10, 24 // (op 1062) 
s_ashr_i32 s18, s6, 31 // (op 1062) 

s_xor_b32 s6, s7, s18 // (op 1062) 

s_sub_i32 s7, s6, s18 // (op 1062) 


s_mul_hi_i32 s6, s7, s11 // (op 1062) 
s_add_i32 s18, s6, s7 // (op 1062) 

s_ashr_i32 s6, s18, 31 // (op 1062) 
s_cmp_eq_i32 s11, 0 // (op 1062) 
s_mov_b32 s19, scc // (op 1062) 
s_cmp_eq_i32 s19, 0 // (op 1062) 
s_cselect_b32 s22, 0, -1 // (op 1062) 

s_and_b32 s19, s12, 31 // (op 1062) 
s_lshl_b32 s23, 1, s19 // (op 1062) 
s_add_i32 s24, s23, s22 // (op 1062) 


s_and_b32 s22, s6, s24 // (op 1062) 


s_add_i32 s6, s18, s22 // (op 1062) 


s_ashr_i32 s18, s6, s19 // (op 1062) 


s_lshl_b32 s6, s12, 24 // (op 1062) 
s_ashr_i32 s19, s6, 31 // (op 1062) 

s_xor_b32 s6, s18, s19 // (op 1062) 

s_sub_i32 s18, s6, s19 // (op 1062) 


s_mul_i32 s6, s7, s14 // (op 1062) 
s_sub_u32 s19, s16, s6 // (op 1062) 

s_mul_i32 s6, s18, s14 // (op 1062) 
s_add_u32 s6, s6, s19 // (op 1062) 

s_mul_hi_i32 s19, s16, s8 // (op 1062) 
s_add_i32 s22, s19, s16 // (op 1062) 

s_ashr_i32 s19, s22, 31 // (op 1062) 
s_cmp_eq_i32 s8, 0 // (op 1062) 
s_mov_b32 s23, scc // (op 1062) 
s_cmp_eq_i32 s23, 0 // (op 1062) 
s_cselect_b32 s24, 0, -1 // (op 1062) 

s_and_b32 s23, s15, 31 // (op 1062) 
s_lshl_b32 s25, 1, s23 // (op 1062) 
s_add_i32 s26, s25, s24 // (op 1062) 


s_and_b32 s24, s19, s26 // (op 1062) 


s_add_i32 s19, s22, s24 // (op 1062) 


s_ashr_i32 s22, s19, s23 // (op 1062) 


s_lshl_b32 s19, s15, 24 // (op 1062) 
s_ashr_i32 s23, s19, 31 // (op 1062) 

s_xor_b32 s19, s22, s23 // (op 1062) 

s_sub_i32 s22, s19, s23 // (op 1062) 


s_add_u32 s38, s4, 16 // (op 1062) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 1062) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 1062) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 1062) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 1062) 

s_mul_i32 s19, s14, s38 // (op 1062) 
s_mul_i32 s23, s50, s19 // (op 1062) 

s_mul_i32 s19, s22, s23 // (op 1062) 

s_sub_u32 s23, s16, s19 // (op 1062) 

s_mov_b32 s19, s23 // (op 1062) 
s_mul_hi_i32 s24, s19, s13 // (op 1062) 
s_add_i32 s25, s24, s19 // (op 1062) 


s_ashr_i32 s19, s25, 31 // (op 1062) 
s_cmp_eq_i32 s13, 0 // (op 1062) 
s_mov_b32 s24, scc // (op 1062) 
s_cmp_eq_i32 s24, 0 // (op 1062) 
s_cselect_b32 s26, 0, -1 // (op 1062) 

s_and_b32 s24, s49, 31 // (op 1062) 
s_lshl_b32 s27, 1, s24 // (op 1062) 
s_add_i32 s28, s27, s26 // (op 1062) 


s_and_b32 s26, s19, s28 // (op 1062) 


s_add_i32 s19, s25, s26 // (op 1062) 


s_ashr_i32 s25, s19, s24 // (op 1062) 


s_lshl_b32 s19, s49, 24 // (op 1062) 
s_ashr_i32 s24, s19, 31 // (op 1062) 

s_xor_b32 s19, s25, s24 // (op 1062) 

s_sub_i32 s25, s19, s24 // (op 1062) 


s_mul_i32 s19, s48, s14 // (op 1062) 
s_add_u32 s26, s20, 16 // (op 1062) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 1062) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 1062) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 1062) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 1062) 

s_sub_i32 s24, s26, s19 // (op 1062) 

s_mul_i32 s26, s25, s24 // (op 1062) 

s_sub_u32 s24, s23, s26 // (op 1062) 


s_add_u32 s23, s24, s19 // (op 1062) 


s_cmp_eq_i32 s22, 0 // (op 1062) 
s_mov_b32 s19, scc // (op 1062) 

s_cmp_eq_i32 s19, 0 // (op 1062) 
s_cselect_b32 s22, s23, s6 // (op 1062) 


s_mul_i32 s6, s18, s38 // (op 1062) 


v_lshl_add_u32 v0, s22, 4, v9 // (op 1062) 


s_sub_i32 s18, s7, s6 // (op 1062) 


s_cmp_eq_i32 s19, 0 // (op 1062) 
s_cselect_b32 s6, s25, s18 // (op 1062) 



v_lshl_add_u32 v3, s6, 4, v11 // (op 1062) 


v_mul_lo_u32 v9, v3, s40 // (op 1062) 

v_add_lshl_u32 v3, v0, v9, 2 // (op 1062) 


s_mul_i32 s7, s40, s5 // (op 1064) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 1064) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 1064) least significant
s_mul_hi_u32 s19, s40, s4 // (op 1064) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 1064) most significant: sum
s_add_u32 s7, s7, s19 // (op 1064) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 1064) 

s_mov_b32 s0, s42 // (op 1064) reg expression
s_mov_b32 s1, s43 // (op 1064) reg expression
s_mov_b32 s2, s18 // (op 1064) reg expression
s_mov_b32 s3, 131072 // (op 1064) call()




buffer_store_dwordx4 v[4:7], v3, s[0:3], 0 offen // (op 166) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__ConditionalBottom_0_beta_155:
















s_endpgm  // End of RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.LRR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end:
.size RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_, .LRR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM__end-RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16896 // lds bytes
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: F8
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
    TransposeB: True
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
  - .name: RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_
    .symbol: RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 96
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_2_size_1_11
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
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 80
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_5
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_7_stride_0_8
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
      - .name: Tensor_10_stride_1_10
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

