// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2,@function
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_exec_begin:


s_load_dwordx2 s[20:21], s[0:1], 56 // Load scalar value
s_load_dwordx2 s[22:23], s[0:1], 64 // Load scalar value
s_load_dwordx2 s[24:25], s[0:1], 72 // Load scalar value
s_load_dwordx2 s[26:27], s[0:1], 80 // Load scalar value
s_load_dwordx2 s[28:29], s[0:1], 88 // Load scalar value
s_load_dwordx2 s[30:31], s[0:1], 96 // Load scalar value
s_load_dwordx2 s[32:33], s[0:1], 104 // Load scalar value
s_load_dwordx2 s[34:35], s[0:1], 112 // Load scalar value
s_load_dwordx2 s[36:37], s[0:1], 120 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


v_and_b32 v0, 48, v1 // (op 1322) 
v_and_b32 v3, 63, v1 // (op 1322) 
v_sub_u32 v4, v3, v0 // (op 1322) 

v_lshrrev_b32 v3, 1, v4 // (op 1322) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 1322) 


v_and_b32 v0, 1, v4 // (op 1322) 

v_lshlrev_b32 v3, 3, v0 // (op 1322) 

v_add_u32 v0, v5, v3 // (op 1322) 


v_and_b32 v3, 48, v1 // (op 1481) 
v_and_b32 v4, 63, v1 // (op 1481) 
v_sub_u32 v5, v4, v3 // (op 1481) 

v_lshrrev_b32 v4, 1, v5 // (op 1481) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 1481) 


v_and_b32 v3, 1, v5 // (op 1481) 

v_lshlrev_b32 v4, 3, v3 // (op 1481) 

v_add_u32 v3, v6, v4 // (op 1481) 


v_lshrrev_b32 v4, 6, v1 // (op 1209) 
v_and_b32 v5, 63, v1 // (op 1209) 
v_lshrrev_b32 v6, 4, v5 // (op 1209) 
v_lshlrev_b32 v7, 2, v6 // (op 1209) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1209) 


v_lshl_add_u32 v4, s16, 6, v6 // (op 1209) 

v_lshrrev_b32 v6, 2, v5 // (op 1209) 

v_and_b32 v5, 3, v6 // (op 1209) 

v_and_b32 v6, 3, v1 // (op 1209) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1209) 


v_lshl_add_u32 v5, s17, 4, v7 // (op 1209) 

v_mul_lo_u32 v6, v5, s2 // (op 1209) 

v_add_lshl_u32 v5, v4, v6, 1 // (op 1209) 


v_and_b32 v4, 48, v1 // (op 1270) 
v_and_b32 v6, 63, v1 // (op 1270) 
v_sub_u32 v7, v6, v4 // (op 1270) 

v_lshrrev_b32 v6, 1, v7 // (op 1270) 
s_mul_i32 s1, s2, s9 // (op 1211) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1211) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1211) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1211) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1211) most significant: sum
s_add_u32 s1, s1, s19 // (op 1211) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1211) 

v_add_lshl_u32 v8, v4, v6, 6 // (op 1270) 


v_and_b32 v4, 1, v7 // (op 1270) 

v_lshlrev_b32 v6, 3, v4 // (op 1270) 

v_lshrrev_b32 v4, 6, v1 // (op 1270) 
v_lshl_add_u32 v7, v4, 4, v6 // (op 1270) 


v_add_u32 v4, v8, v7 // (op 1270) 


v_lshrrev_b32 v6, 6, v1 // (op 1380) 
v_and_b32 v7, 15, v1 // (op 1380) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 1380) 


v_and_b32 v6, 63, v1 // (op 1380) 
v_lshrrev_b32 v7, 4, v6 // (op 1380) 

v_lshl_add_u32 v6, v7, 6, v8 // (op 1380) 


v_lshrrev_b32 v7, 6, v1 // (op 1396) 
v_and_b32 v8, 63, v1 // (op 1396) 
v_lshrrev_b32 v9, 4, v8 // (op 1396) 
v_lshlrev_b32 v10, 2, v9 // (op 1396) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1396) 


v_lshl_add_u32 v7, s16, 6, v9 // (op 1396) 

v_lshrrev_b32 v9, 2, v8 // (op 1396) 

v_and_b32 v8, 3, v9 // (op 1396) 

v_and_b32 v9, 3, v1 // (op 1396) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1396) 


v_lshl_add_u32 v8, s17, 4, v10 // (op 1396) 

v_mul_lo_u32 v9, v8, s4 // (op 1396) 

v_add_lshl_u32 v8, v7, v9, 1 // (op 1396) 


v_and_b32 v7, 48, v1 // (op 1442) 
v_and_b32 v9, 63, v1 // (op 1442) 
v_sub_u32 v10, v9, v7 // (op 1442) 

v_lshrrev_b32 v9, 1, v10 // (op 1442) 
s_mul_i32 s1, s4, s9 // (op 1398) most significant: low of low * high
v_add_lshl_u32 v11, v7, v9, 6 // (op 1442) 


v_and_b32 v7, 1, v10 // (op 1442) 

v_lshlrev_b32 v9, 3, v7 // (op 1442) 

v_lshrrev_b32 v7, 6, v1 // (op 1442) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 1442) 


v_add_u32 v7, v11, v10 // (op 1442) 


v_lshrrev_b32 v9, 6, v1 // (op 1510) 
v_and_b32 v10, 15, v1 // (op 1510) 
v_lshl_add_u32 v11, v9, 4, v10 // (op 1510) 


v_and_b32 v9, 63, v1 // (op 1510) 
v_lshrrev_b32 v10, 4, v9 // (op 1510) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 1510) 


s_mul_i32 s38, s5, s8 // (op 1398) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1398) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1398) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1398) most significant: sum
s_add_u32 s1, s1, s39 // (op 1398) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 1398) 

s_mov_b32 s40, s6 // (op 1211) reg expression
s_mov_b32 s41, s7 // (op 1211) reg expression
s_mov_b32 s42, s18 // (op 1211) reg expression
s_mov_b32 s43, 131072 // (op 1211) call()

s_mov_b32 s44, s10 // (op 1398) reg expression

s_mov_b32 s45, s11 // (op 1398) reg expression
s_mov_b32 s46, s38 // (op 1398) reg expression
s_mov_b32 s47, 131072 // (op 1398) call()



v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 313) call()
v_accvgpr_write a5, 0.00000 // (op 313) call()
v_accvgpr_write a6, 0.00000 // (op 313) call()
v_accvgpr_write a7, 0.00000 // (op 313) call()
v_and_b32 v12, -4, v1 // (op 976) 
v_sub_u32 v13, v1, v12 // (op 976) 

v_lshlrev_b32 v12, 4, v13 // (op 976) 

v_lshl_add_u32 v13, s16, 6, v12 // (op 976) 

v_mov_b32 v15, 0 // (op 976) convert
v_mov_b32 v14, v13 // (op 976) convert

v_lshrrev_b32 v12, 2, v1 // (op 976) 
v_mul_lo_u32 v17, v12, s13 // (op 976) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 976) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 976) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 976) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 976) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 976) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 978) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 978) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 978) least significant
s_mul_hi_u32 s7, s12, s22 // (op 978) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 978) most significant: sum
s_add_u32 s1, s1, s7 // (op 978) most significant: sum


s_mov_b32 s8, s20 // (op 978) reg expression
s_mov_b32 s9, s21 // (op 978) reg expression
s_mov_b32 s10, s0 // (op 978) reg expression
s_mov_b32 s11, 131072 // (op 978) call()

v_and_b32 v14, -4, v1 // (op 995) 
v_sub_u32 v15, v1, v14 // (op 995) 

v_lshlrev_b32 v14, 4, v15 // (op 995) 

v_lshl_add_u32 v15, s16, 6, v14 // (op 995) 

v_mov_b32 v17, 0 // (op 995) convert
v_mov_b32 v16, v15 // (op 995) convert

v_lshrrev_b32 v14, 2, v1 // (op 995) 
v_mul_lo_u32 v19, v14, s13 // (op 995) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 995) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 995) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 995) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 995) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 995) most significant half



s_mul_i32 s1, s12, s23 // (op 997) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 997) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 997) least significant
s_mul_hi_u32 s7, s12, s22 // (op 997) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 997) most significant: sum
s_add_u32 s1, s1, s7 // (op 997) most significant: sum


s_mov_b32 s8, s20 // (op 997) reg expression
s_mov_b32 s9, s21 // (op 997) reg expression
s_mov_b32 s10, s0 // (op 997) reg expression
s_mov_b32 s11, 131072 // (op 997) call()

s_lshl_b32 s0, s17, 4 // (op 1040) 
s_mov_b32 s7, 0 // (op 1040) convert
s_mov_b32 s6, s0 // (op 1040) convert

v_mul_lo_u32 v17, v1, s25 // (op 1040) most significant: low of low * high
v_mul_hi_u32 v18, v1, s24 // (op 1040) most significant: high of low * low
v_mul_lo_u32 v16, v1, s24 // (op 1040) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1040) most significant: sum

v_mov_b32 v18, s6 // (op 1040) 
v_mov_b32 v19, s7 // (op 1040) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 1040) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 1040) most significant half




s_mul_i32 s1, s24, s23 // (op 1042) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 1042) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 1042) least significant
s_mul_hi_u32 s7, s24, s22 // (op 1042) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1042) most significant: sum
s_add_u32 s1, s1, s7 // (op 1042) most significant: sum


s_mov_b32 s48, s26 // (op 1042) reg expression
s_mov_b32 s49, s27 // (op 1042) reg expression
s_mov_b32 s50, s0 // (op 1042) reg expression
s_mov_b32 s51, 131072 // (op 1042) call()

s_lshl_b32 s0, s17, 4 // (op 1059) 
s_mov_b32 s7, 0 // (op 1059) convert
s_mov_b32 s6, s0 // (op 1059) convert

v_mul_lo_u32 v19, v1, s25 // (op 1059) most significant: low of low * high
v_mul_hi_u32 v20, v1, s24 // (op 1059) most significant: high of low * low
v_mul_lo_u32 v18, v1, s24 // (op 1059) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 1059) most significant: sum

v_mov_b32 v20, s6 // (op 1059) 
v_mov_b32 v21, s7 // (op 1059) 
v_add_co_u32 v18, s[0:1], v18, v20 // (op 1059) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 1059) most significant half




s_mul_i32 s1, s24, s23 // (op 1061) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 1061) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 1061) least significant
s_mul_hi_u32 s7, s24, s22 // (op 1061) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1061) most significant: sum
s_add_u32 s1, s1, s7 // (op 1061) most significant: sum


s_mov_b32 s48, s26 // (op 1061) reg expression
s_mov_b32 s49, s27 // (op 1061) reg expression
s_mov_b32 s50, s0 // (op 1061) reg expression
s_mov_b32 s51, 131072 // (op 1061) call()

v_and_b32 v20, -64, v1 // (op 1104) 
v_sub_u32 v21, v1, v20 // (op 1104) 

v_lshl_add_u32 v20, s16, 6, v21 // (op 1104) 

v_lshrrev_b32 v21, 6, v1 // (op 1104) 
v_lshlrev_b32 v22, 1, v21 // (op 1104) 

v_mul_lo_u32 v21, v22, s28 // (op 1104) 

v_add_u32 v22, v20, v21 // (op 1104) 


s_mul_i32 s1, s28, s33 // (op 1106) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1106) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1106) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1106) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1106) most significant: sum
s_add_u32 s1, s1, s7 // (op 1106) most significant: sum


s_mov_b32 s52, s30 // (op 1106) reg expression
s_mov_b32 s53, s31 // (op 1106) reg expression
s_mov_b32 s54, s0 // (op 1106) reg expression
s_mov_b32 s55, 131072 // (op 1106) call()

v_and_b32 v20, -64, v1 // (op 1123) 
v_sub_u32 v21, v1, v20 // (op 1123) 

v_lshl_add_u32 v20, s16, 6, v21 // (op 1123) 

v_lshrrev_b32 v21, 6, v1 // (op 1123) 
v_lshlrev_b32 v23, 1, v21 // (op 1123) 

v_mul_lo_u32 v21, v23, s28 // (op 1123) 

v_add_u32 v23, v20, v21 // (op 1123) 


s_mul_i32 s1, s28, s33 // (op 1125) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1125) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1125) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1125) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1125) most significant: sum
s_add_u32 s1, s1, s7 // (op 1125) most significant: sum


s_mov_b32 s52, s30 // (op 1125) reg expression
s_mov_b32 s53, s31 // (op 1125) reg expression
s_mov_b32 s54, s0 // (op 1125) reg expression
s_mov_b32 s55, 131072 // (op 1125) call()

v_and_b32 v20, 15, v1 // (op 1168) 
v_lshl_add_u32 v21, s17, 4, v20 // (op 1168) 

v_and_b32 v20, 63, v1 // (op 1168) 
v_lshrrev_b32 v24, 4, v20 // (op 1168) 

v_mul_lo_u32 v20, v24, s34 // (op 1168) 
v_add_u32 v20, v20, v21 // (op 1168) 


s_mul_i32 s1, s34, s33 // (op 1170) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1170) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1170) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1170) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1170) most significant: sum
s_add_u32 s1, s1, s7 // (op 1170) most significant: sum


s_mov_b32 s56, s36 // (op 1170) reg expression
s_mov_b32 s57, s37 // (op 1170) reg expression
s_mov_b32 s58, s0 // (op 1170) reg expression
s_mov_b32 s59, 131072 // (op 1170) call()

v_and_b32 v21, 15, v1 // (op 1187) 
v_lshl_add_u32 v24, s17, 4, v21 // (op 1187) 

v_and_b32 v21, 63, v1 // (op 1187) 
v_lshrrev_b32 v25, 4, v21 // (op 1187) 

v_mul_lo_u32 v21, v25, s34 // (op 1187) 
v_add_u32 v21, v21, v24 // (op 1187) 


s_mul_i32 s1, s34, s33 // (op 1189) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1189) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1189) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1189) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1189) most significant: sum
s_add_u32 s1, s1, s7 // (op 1189) most significant: sum


s_mov_b32 s56, s36 // (op 1189) reg expression
s_mov_b32 s57, s37 // (op 1189) reg expression
s_mov_b32 s58, s0 // (op 1189) reg expression
s_mov_b32 s59, 131072 // (op 1189) call()


v_mov_b64 v[24:25], s[34:35] // (op 782) convert

v_mov_b32 v26, v20 // (op 782) reg expression
v_mov_b32 v27, v26 // (op 782) 
buffer_load_ubyte v28, v27, s[56:59], 0 offen offset: 0 // (op 782) Load value



v_lshlrev_b32 v24, 4, v1 // (op 1310) 
s_nop 0
v_readfirstlane_b32 s0, v24 // (op 1310) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[24:25], v[16:17] // (op 924) reg expression

s_nop 0
s_mov_b32 m0, s0 // (op 924) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 924) Load value direct to lds


v_and_b32 v24, -4, v1 // (op 1253) 
v_sub_u32 v25, v1, v24 // (op 1253) 

v_lshlrev_b32 v24, 4, v25 // (op 1253) 

v_lshrrev_b32 v25, 2, v1 // (op 1253) 
v_lshl_add_u32 v26, v25, 6, v24 // (op 1253) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1253) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 929) 
v_mov_b64 v[24:25], s[6:7] // (op 929) convert


v_mov_b64 v[26:27], v[12:13] // (op 929) reg expression
s_add_u32 s1, 2048, s0 // (op 929) 
s_mov_b32 m0, s1 // (op 929) reg expression
buffer_load_dwordx4 v26, s[8:11], 0 offen offset: 0 lds // (op 929) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 929) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 929) most significant half

s_add_u32 m0, m0, 2048 // (op 929) 
buffer_load_dwordx4 v26, s[8:11], 0 offen offset: 0 lds // (op 929) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 929) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 929) most significant half

s_add_u32 m0, m0, 2048 // (op 929) 
buffer_load_dwordx4 v26, s[8:11], 0 offen offset: 0 lds // (op 929) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 929) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 929) most significant half

s_add_u32 m0, m0, 2048 // (op 929) 
buffer_load_dwordx4 v26, s[8:11], 0 offen offset: 0 lds // (op 929) Load value direct to lds





v_mov_b64 v[24:25], s[28:29] // (op 792) convert

v_mov_b32 v26, v22 // (op 792) reg expression
buffer_load_ubyte v30, v26, s[52:55], 0 offen offset: 0 // (op 792) Load value
v_add_u32 v26, v26, v24 // (op 792) 
buffer_load_ubyte v31, v26, s[52:55], 0 offen offset: 0 // (op 792) Load value



v_mov_b64 v[24:25], s[34:35] // (op 796) convert

s_lshl_b32 s0, s34, 2 // (op 796) 
v_add_u32 v29, v20, s0 // (op 796) 

v_mov_b32 v26, v29 // (op 796) 
buffer_load_ubyte v27, v26, s[56:59], 0 offen offset: 0 // (op 796) Load value




v_lshlrev_b32 v20, 4, v1 // (op 1469) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1469) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[6:7], s[24:25], 7 // (op 934) 
v_mov_b32 v20, s6 // (op 934) 
v_mov_b32 v29, s7 // (op 934) 
v_add_co_u32 v24, s[14:15], v16, v20 // (op 934) least significant half
v_addc_co_u32 v25, s[14:15], v17, v29, s[14:15] // (op 934) most significant half




s_add_u32 s1, 10240, s0 // (op 934) 

s_mov_b32 m0, s1 // (op 934) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 934) Load value direct to lds




v_and_b32 v20, -4, v1 // (op 1425) 
v_sub_u32 v29, v1, v20 // (op 1425) 

v_lshlrev_b32 v20, 4, v29 // (op 1425) 

v_lshrrev_b32 v29, 2, v1 // (op 1425) 
v_lshl_add_u32 v16, v29, 6, v20 // (op 1425) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 1425) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 936) 
v_mov_b64 v[16:17], s[6:7] // (op 936) convert


s_lshl_b64 s[6:7], s[12:13], 7 // (op 936) 
v_mov_b32 v20, s6 // (op 936) 
v_mov_b32 v29, s7 // (op 936) 
v_add_co_u32 v24, s[14:15], v12, v20 // (op 936) least significant half
v_addc_co_u32 v25, s[14:15], v13, v29, s[14:15] // (op 936) most significant half




s_add_u32 s1, 12288, s0 // (op 936) 
s_mov_b32 m0, s1 // (op 936) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 936) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 936) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 936) most significant half

s_add_u32 m0, m0, 2048 // (op 936) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 936) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 936) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 936) most significant half

s_add_u32 m0, m0, 2048 // (op 936) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 936) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 936) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 936) most significant half

s_add_u32 m0, m0, 2048 // (op 936) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 936) Load value direct to lds






v_mov_b64 v[12:13], s[28:29] // (op 806) convert

s_lshl_b32 s0, s28, 2 // (op 806) 
v_add_u32 v20, v22, s0 // (op 806) 

buffer_load_ubyte v16, v20, s[52:55], 0 offen offset: 0 // (op 806) Load value
v_add_u32 v20, v20, v12 // (op 806) 
buffer_load_ubyte v17, v20, s[52:55], 0 offen offset: 0 // (op 806) Load value



v_and_b32 v20, -64, v1 // (op 1363) 
v_sub_u32 v22, v1, v20 // (op 1363) 

v_lshrrev_b32 v20, 6, v1 // (op 1363) 
v_lshl_add_u32 v29, v20, 7, v22 // (op 1363) 





s_waitcnt vmcnt(9)
ds_write_b8 v29, v30 offset:20480 // (op 810) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v29, v31 offset:20544 // (op 810) Store local data 

s_barrier  // (op 1567) 



ds_read_b64_tr_b8 v[32:33], v0 // (op 833) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:128 // (op 833) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:1024 // (op 833) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:1152 // (op 833) Transpose load local data 
s_barrier  // (op 1551) 



ds_read_b64_tr_b8 v[40:41], v4 offset:2048 // (op 824) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:2560 // (op 824) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:6144 // (op 824) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:6656 // (op 824) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1583) 


ds_read_u8 v20, v6 offset:20480 // (op 837) Load local data 



ds_read_b64_tr_b8 v[48:49], v4 offset:2080 // (op 829) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v4 offset:2592 // (op 829) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:6176 // (op 829) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:6688 // (op 829) Transpose load local data 


ds_read_u8 v22, v6 offset:20512 // (op 842) Load local data 
v_and_b32 v29, -4, v1 // (op 1225) 
v_sub_u32 v12, v1, v29 // (op 1225) 

v_lshlrev_b32 v13, 4, v12 // (op 1225) 

v_lshrrev_b32 v12, 2, v1 // (op 1225) 
v_lshl_add_u32 v29, v12, 6, v13 // (op 1225) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 1225) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 4, v1 // (op 1282) 
s_nop 0
v_readfirstlane_b32 s1, v29 // (op 1282) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -64, v1 // (op 1334) 
v_sub_u32 v12, v1, v29 // (op 1334) 

v_lshrrev_b32 v13, 6, v1 // (op 1334) 
v_lshl_add_u32 v29, v13, 7, v12 // (op 1334) 


v_and_b32 v12, -4, v1 // (op 1410) 
v_sub_u32 v13, v1, v12 // (op 1410) 

v_lshlrev_b32 v12, 4, v13 // (op 1410) 

v_lshrrev_b32 v13, 2, v1 // (op 1410) 
v_lshl_add_u32 v24, v13, 6, v12 // (op 1410) 


s_nop 0
v_readfirstlane_b32 s6, v24 // (op 1410) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v12, 4, v1 // (op 1454) 
s_nop 0
v_readfirstlane_b32 s7, v12 // (op 1454) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1493) 
v_sub_u32 v13, v1, v12 // (op 1493) 

v_lshrrev_b32 v12, 6, v1 // (op 1493) 
v_lshl_add_u32 v24, v12, 7, v13 // (op 1493) 


s_mov_b32 s14, 0 // (op 315) call()
s_ashr_i64 s[18:19], s[22:23], 7 // (op 314) 
s_mov_b32 s15, s18 // (op 314) convert

s_and_b32 s18, s15, 4294967294 // (op 314) 

s_cmp_lt_u32 s14, s18 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_314 // (op 314) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_314 if false)
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_314:
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[32:39], a[0:3], v20, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 84) 
s_add_u32 s14, s14, 2 // (op 316) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[32:39], a[4:7], v22, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 350) 

v_mov_b64 v[12:13], s[34:35] // (op 82) convert

v_mov_b32 v25, v21 // (op 82) 
s_lshl_b32 s15, s34, 3 // (op 82) 
v_add_u32 v26, v25, s15 // (op 82) 

v_mov_b32 v56, v26 // (op 82) 
buffer_load_ubyte v28, v56, s[56:59], 0 offen offset: 0 // (op 82) Load value



s_barrier  // (op 1570) 


v_mov_b64 v[12:13], v[18:19] // (op 896) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 896) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 896) 

v_mov_b32 v26, s38 // (op 896) 
v_mov_b32 v56, s39 // (op 896) 
v_add_co_u32 v58, s[18:19], v12, v26 // (op 896) least significant half
v_addc_co_u32 v59, s[18:19], v13, v56, s[18:19] // (op 896) most significant half





s_mov_b32 m0, s1 // (op 896) reg expression
buffer_load_dwordx4 v58, s[48:51], 0 offen offset: 0 lds // (op 896) Load value direct to lds

s_barrier  // (op 1554) 

s_lshl_b64 s[18:19], s[12:13], 5 // (op 891) 
v_mov_b64 v[56:57], s[18:19] // (op 891) convert


v_mov_b64 v[58:59], v[14:15] // (op 891) 
s_lshl_b64 s[18:19], s[12:13], 7 // (op 891) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 891) 

v_mov_b32 v26, s38 // (op 891) 
v_mov_b32 v60, s39 // (op 891) 
v_add_co_u32 v62, s[18:19], v58, v26 // (op 891) least significant half
v_addc_co_u32 v63, s[18:19], v59, v60, s[18:19] // (op 891) most significant half




s_add_u32 s15, 2048, s0 // (op 891) 
s_mov_b32 m0, s15 // (op 891) reg expression
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 891) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v56 // (op 891) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 891) most significant half

s_add_u32 m0, m0, 2048 // (op 891) 
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 891) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v56 // (op 891) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 891) most significant half

s_add_u32 m0, m0, 2048 // (op 891) 
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 891) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v56 // (op 891) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 891) most significant half

s_add_u32 m0, m0, 2048 // (op 891) 
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 891) Load value direct to lds




v_mov_b64 v[56:57], s[28:29] // (op 17) convert

v_mov_b32 v26, v23 // (op 17) 
s_lshl_b32 s15, s28, 3 // (op 17) 
v_add_u32 v60, v26, s15 // (op 17) 

buffer_load_ubyte v30, v60, s[52:55], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v60, v60, v56 // (op 17) 
buffer_load_ubyte v31, v60, s[52:55], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1615) 



s_waitcnt vmcnt(9)
ds_write_b8 v24, v16 offset:20736 // (op 240) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v24, v17 offset:20800 // (op 240) Store local data 
s_barrier  // (op 1606) 



ds_read_b64_tr_b8 v[32:33], v3 offset:10240 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:10368 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v3 offset:11264 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v3 offset:11392 // (op 233) Transpose load local data 
s_barrier  // (op 1600) 



ds_read_b64_tr_b8 v[40:41], v7 offset:12288 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:12800 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:16384 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:16896 // (op 231) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1612) 


ds_read_u8 v20, v9 offset:20736 // (op 235) Load local data 



ds_read_b64_tr_b8 v[48:49], v7 offset:12320 // (op 357) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v7 offset:12832 // (op 357) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v7 offset:16416 // (op 357) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v7 offset:16928 // (op 357) Transpose load local data 


ds_read_u8 v22, v9 offset:20768 // (op 361) Load local data 
s_barrier  // (op 1670) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[32:39], a[0:3], v20, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 238) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[32:39], a[4:7], v22, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 351) 

v_mov_b64 v[56:57], s[34:35] // (op 237) convert

s_lshl_b32 s15, s34, 2 // (op 237) 
s_mul_i32 s18, 3, s15 // (op 237) 

v_add_u32 v60, v25, s18 // (op 237) 

v_mov_b32 v61, v60 // (op 237) 
buffer_load_ubyte v27, v61, s[56:59], 0 offen offset: 0 // (op 237) Load value




s_barrier  // (op 1609) 
s_lshl_b32 s15, s34, 3 // (op 1179) 


v_add_u32 v21, v21, s15 // (op 1179) 

s_lshl_b64 s[18:19], s[24:25], 7 // (op 919) 
s_mul_i32 s39, 3, s19 // (op 919) most significant: low of low * high
s_mul_i32 s38, 3, s18 // (op 919) least significant
s_mul_hi_u32 s15, 3, s18 // (op 919) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 919) most significant: sum


v_mov_b32 v25, s38 // (op 919) 
v_mov_b32 v56, s39 // (op 919) 
v_add_co_u32 v60, s[18:19], v12, v25 // (op 919) least significant half
v_addc_co_u32 v61, s[18:19], v13, v56, s[18:19] // (op 919) most significant half




s_add_u32 s15, 10240, s7 // (op 919) 

s_mov_b32 m0, s15 // (op 919) reg expression
buffer_load_dwordx4 v60, s[48:51], 0 offen offset: 0 lds // (op 919) Load value direct to lds



s_barrier  // (op 1603) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 1051) 

s_lshl_b64 s[38:39], s[18:19], 1 // (op 1051) 

v_mov_b32 v25, s38 // (op 1051) 
v_mov_b32 v12, s39 // (op 1051) 
s_lshl_b64 s[18:19], s[12:13], 5 // (op 914) 
v_mov_b64 v[56:57], s[18:19] // (op 914) convert


s_lshl_b64 s[18:19], s[12:13], 7 // (op 914) 
s_mul_i32 s61, 3, s19 // (op 914) most significant: low of low * high
s_mul_i32 s60, 3, s18 // (op 914) least significant
s_mul_hi_u32 s15, 3, s18 // (op 914) most significant: high of low * low
s_add_u32 s61, s61, s15 // (op 914) most significant: sum


v_mov_b32 v13, s60 // (op 914) 
v_mov_b32 v60, s61 // (op 914) 
v_add_co_u32 v18, s[18:19], v18, v25 // (op 1051) least significant half
v_addc_co_u32 v19, s[18:19], v19, v12, s[18:19] // (op 1051) most significant half




v_add_co_u32 v62, s[18:19], v58, v13 // (op 914) least significant half
v_addc_co_u32 v63, s[18:19], v59, v60, s[18:19] // (op 914) most significant half




s_add_u32 s15, 12288, s6 // (op 914) 
s_mov_b32 m0, s15 // (op 914) reg expression
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 914) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v56 // (op 914) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 914) most significant half

s_add_u32 m0, m0, 2048 // (op 914) 
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 914) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v56 // (op 914) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 914) most significant half

s_add_u32 m0, m0, 2048 // (op 914) 
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 914) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v56 // (op 914) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 914) most significant half

s_add_u32 m0, m0, 2048 // (op 914) 
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 914) Load value direct to lds





v_mov_b64 v[12:13], s[28:29] // (op 242) convert

s_lshl_b32 s15, s28, 2 // (op 242) 
s_mul_i32 s18, 3, s15 // (op 242) 

v_add_u32 v25, v26, s18 // (op 242) 

s_lshl_b64 s[18:19], s[12:13], 7 // (op 987) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 987) 

v_mov_b32 v56, s38 // (op 987) 
v_mov_b32 v57, s39 // (op 987) 
v_add_co_u32 v14, s[18:19], v14, v56 // (op 987) least significant half
v_addc_co_u32 v15, s[18:19], v15, v57, s[18:19] // (op 987) most significant half




buffer_load_ubyte v16, v25, s[52:55], 0 offen offset: 0 // (op 242) Load value
v_add_u32 v25, v25, v12 // (op 242) 
buffer_load_ubyte v17, v25, s[52:55], 0 offen offset: 0 // (op 242) Load value



s_lshl_b32 s15, s28, 3 // (op 1115) 
s_barrier  // (op 1586) 
v_add_u32 v23, v23, s15 // (op 1115) 




s_waitcnt vmcnt(9)
ds_write_b8 v29, v30 offset:20480 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v29, v31 offset:20544 // (op 21) Store local data 
s_barrier  // (op 1576) 



ds_read_b64_tr_b8 v[32:33], v0 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:128 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:1024 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:1152 // (op 72) Transpose load local data 
s_barrier  // (op 1560) 



ds_read_b64_tr_b8 v[40:41], v4 offset:2048 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:2560 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:6144 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:6656 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1593) 


ds_read_u8 v20, v6 offset:20480 // (op 77) Load local data 



ds_read_b64_tr_b8 v[48:49], v4 offset:2080 // (op 344) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v4 offset:2592 // (op 344) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:6176 // (op 344) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:6688 // (op 344) Transpose load local data 


ds_read_u8 v22, v6 offset:20512 // (op 348) Load local data 
s_ashr_i64 s[18:19], s[22:23], 7 // (op 314) 
s_mov_b32 s15, s18 // (op 314) convert

s_and_b32 s18, s15, 4294967294 // (op 314) 

s_cmp_lt_u32 s14, s18 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_314 // (op 314) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_314 if true)
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_314:













s_barrier  // (op 1654) 
v_and_b32 v21, -4, v1 // (op 949) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 949) 
v_sub_u32 v12, v1, v21 // (op 949) 

s_lshr_b32 s6, s0, 1 // (op 949) 

v_lshlrev_b32 v21, 4, v12 // (op 949) 

v_lshl_add_u32 v12, s16, 6, v21 // (op 949) 

v_mov_b32 v15, 0 // (op 949) convert
v_mov_b32 v14, v12 // (op 949) convert

v_lshrrev_b32 v21, 2, v1 // (op 949) 
v_lshl_add_u32 v12, s6, 8, v21 // (op 949) 


v_mul_lo_u32 v19, v12, s13 // (op 949) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 949) most significant: high of low * low
v_mul_lo_u32 v18, v12, s12 // (op 949) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 949) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v18 // (op 949) least significant half
v_addc_co_u32 v13, s[0:1], v15, v19, s[0:1] // (op 949) most significant half



s_mul_i32 s1, s12, s23 // (op 951) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 951) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 951) least significant
s_mul_hi_u32 s7, s12, s22 // (op 951) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 951) most significant: sum
s_add_u32 s1, s1, s7 // (op 951) most significant: sum


s_mov_b32 s8, s20 // (op 951) reg expression
s_mov_b32 s9, s21 // (op 951) reg expression
s_mov_b32 s10, s0 // (op 951) reg expression
s_mov_b32 s11, 131072 // (op 951) call()

s_ashr_i64 s[0:1], s[22:23], 7 // (op 1017) 
s_lshr_b32 s6, s0, 1 // (op 1017) 

v_lshl_add_u32 v21, s6, 8, v1 // (op 1017) 

s_lshl_b32 s0, s17, 4 // (op 1017) 
s_mov_b32 s7, 0 // (op 1017) convert
s_mov_b32 s6, s0 // (op 1017) convert

v_mul_lo_u32 v15, v21, s25 // (op 1017) most significant: low of low * high
v_mul_hi_u32 v18, v21, s24 // (op 1017) most significant: high of low * low
v_mul_lo_u32 v14, v21, s24 // (op 1017) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 1017) most significant: sum

v_mov_b32 v18, s6 // (op 1017) 
v_mov_b32 v19, s7 // (op 1017) 
v_add_co_u32 v14, s[0:1], v14, v18 // (op 1017) least significant half
v_addc_co_u32 v15, s[0:1], v15, v19, s[0:1] // (op 1017) most significant half





s_mul_i32 s1, s24, s23 // (op 1019) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 1019) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 1019) least significant
s_mul_hi_u32 s7, s24, s22 // (op 1019) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1019) most significant: sum
s_add_u32 s1, s1, s7 // (op 1019) most significant: sum


s_mov_b32 s48, s26 // (op 1019) reg expression
s_mov_b32 s49, s27 // (op 1019) reg expression
s_mov_b32 s50, s0 // (op 1019) reg expression
s_mov_b32 s51, 131072 // (op 1019) call()

v_and_b32 v21, -64, v1 // (op 1081) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1081) 
v_sub_u32 v18, v1, v21 // (op 1081) 

s_lshr_b32 s6, s0, 1 // (op 1081) 

v_lshl_add_u32 v19, s16, 6, v18 // (op 1081) 

v_lshrrev_b32 v18, 6, v1 // (op 1081) 
v_lshlrev_b32 v21, 1, v18 // (op 1081) 

v_lshl_add_u32 v18, s6, 3, v21 // (op 1081) 


v_mul_lo_u32 v21, v18, s28 // (op 1081) 

v_add_u32 v18, v19, v21 // (op 1081) 


s_mul_i32 s1, s28, s33 // (op 1083) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1083) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1083) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1083) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1083) most significant: sum
s_add_u32 s1, s1, s7 // (op 1083) most significant: sum


s_mov_b32 s52, s30 // (op 1083) reg expression
s_mov_b32 s53, s31 // (op 1083) reg expression
s_mov_b32 s54, s0 // (op 1083) reg expression
s_mov_b32 s55, 131072 // (op 1083) call()

v_and_b32 v19, 63, v1 // (op 1145) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1145) 
v_lshrrev_b32 v21, 4, v19 // (op 1145) 

s_lshr_b32 s6, s0, 1 // (op 1145) 

v_and_b32 v19, 15, v1 // (op 1145) 
v_lshl_add_u32 v23, s6, 3, v21 // (op 1145) 


v_lshl_add_u32 v21, s17, 4, v19 // (op 1145) 

v_mul_lo_u32 v19, v23, s34 // (op 1145) 
v_add_u32 v19, v19, v21 // (op 1145) 


s_mul_i32 s1, s34, s33 // (op 1147) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1147) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1147) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1147) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1147) most significant: sum
s_add_u32 s1, s1, s7 // (op 1147) most significant: sum


s_mov_b32 s56, s36 // (op 1147) reg expression
s_mov_b32 s57, s37 // (op 1147) reg expression
s_mov_b32 s58, s0 // (op 1147) reg expression
s_mov_b32 s59, 131072 // (op 1147) call()

v_and_b32 v21, -4, v1 // (op 1238) 
v_sub_u32 v23, v1, v21 // (op 1238) 

v_lshlrev_b32 v21, 4, v23 // (op 1238) 

v_lshrrev_b32 v23, 2, v1 // (op 1238) 
v_lshl_add_u32 v24, v23, 6, v21 // (op 1238) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 1238) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v21, 4, v1 // (op 1295) 
s_nop 0
v_readfirstlane_b32 s1, v21 // (op 1295) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -64, v1 // (op 1347) 
v_sub_u32 v23, v1, v21 // (op 1347) 

v_lshrrev_b32 v21, 6, v1 // (op 1347) 
v_lshl_add_u32 v24, v21, 7, v23 // (op 1347) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 320) 
s_and_b32 s14, s6, 4294967294 // (op 320) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 319) 
s_mov_b32 s15, s6 // (op 319) convert

s_cmp_lt_u32 s14, s15 // (op 319) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_319 // (op 319) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_319 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_319:
s_barrier  // (op 1557) 
s_add_u32 s14, s14, 1 // (op 321) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 901) 
v_mov_b64 v[16:17], s[6:7] // (op 901) convert


v_mov_b64 v[20:21], v[12:13] // (op 901) 
v_mov_b64 v[22:23], v[20:21] // (op 901) reg expression
s_add_u32 s15, 2048, s0 // (op 901) 
s_mov_b32 m0, s15 // (op 901) reg expression
buffer_load_dwordx4 v22, s[8:11], 0 offen offset: 0 lds // (op 901) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v16 // (op 901) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 901) most significant half

s_add_u32 m0, m0, 2048 // (op 901) 
buffer_load_dwordx4 v22, s[8:11], 0 offen offset: 0 lds // (op 901) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v16 // (op 901) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 901) most significant half

s_add_u32 m0, m0, 2048 // (op 901) 
buffer_load_dwordx4 v22, s[8:11], 0 offen offset: 0 lds // (op 901) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v16 // (op 901) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 901) most significant half

s_add_u32 m0, m0, 2048 // (op 901) 
buffer_load_dwordx4 v22, s[8:11], 0 offen offset: 0 lds // (op 901) Load value direct to lds




s_lshl_b64 s[6:7], s[12:13], 7 // (op 939) 
v_mov_b32 v16, s6 // (op 939) 
v_mov_b32 v17, s7 // (op 939) 
v_add_co_u32 v12, s[18:19], v12, v16 // (op 939) least significant half
v_addc_co_u32 v13, s[18:19], v13, v17, s[18:19] // (op 939) most significant half




s_barrier  // (op 1573) 


v_mov_b64 v[16:17], v[14:15] // (op 907) 
v_mov_b64 v[20:21], v[16:17] // (op 907) reg expression

s_mov_b32 m0, s1 // (op 907) reg expression
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 907) Load value direct to lds


s_lshl_b64 s[6:7], s[24:25], 7 // (op 1009) 
v_mov_b32 v16, s6 // (op 1009) 
v_mov_b32 v17, s7 // (op 1009) 
v_add_co_u32 v14, s[18:19], v14, v16 // (op 1009) least significant half
v_addc_co_u32 v15, s[18:19], v15, v17, s[18:19] // (op 1009) most significant half





v_mov_b64 v[16:17], s[28:29] // (op 184) convert

v_mov_b32 v20, v18 // (op 184) 
v_mov_b32 v21, v20 // (op 184) reg expression
buffer_load_ubyte v22, v21, s[52:55], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v21, v21, v16 // (op 184) 
buffer_load_ubyte v23, v21, s[52:55], 0 offen offset: 0 // (op 184) Load value



s_lshl_b32 s15, s28, 2 // (op 1073) 
v_add_u32 v18, v18, s15 // (op 1073) 

s_barrier  // (op 1589) 
s_waitcnt vmcnt(3)
s_barrier  // (op 1563) 






ds_read_b64_tr_b8 v[26:27], v4 offset:2048 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v4 offset:2560 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v4 offset:6144 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:6656 // (op 173) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v22 offset:20480 // (op 182) Store local data 
s_barrier  // (op 1579) 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v23 offset:20544 // (op 182) Store local data 




ds_read_b64_tr_b8 v[34:35], v0 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:128 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:1024 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:1152 // (op 175) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1596) 


ds_read_u8 v25, v6 offset:20480 // (op 177) Load local data 

v_mov_b64 v[16:17], s[34:35] // (op 179) convert

v_mov_b32 v20, v19 // (op 179) 
v_mov_b32 v21, v20 // (op 179) reg expression
v_mov_b32 v22, v21 // (op 179) 
buffer_load_ubyte v23, v22, s[56:59], 0 offen offset: 0 // (op 179) Load value




s_lshl_b32 s15, s34, 2 // (op 1137) 
v_add_u32 v19, v19, s15 // (op 1137) 




ds_read_b64_tr_b8 v[42:43], v4 offset:2080 // (op 325) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v25, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 180) 
ds_read_b64_tr_b8 v[44:45], v4 offset:2592 // (op 325) Transpose load local data 


ds_read_b64_tr_b8 v[46:47], v4 offset:6176 // (op 325) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v4 offset:6688 // (op 325) Transpose load local data 


ds_read_u8 v16, v6 offset:20512 // (op 329) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[34:41], a[4:7], v16, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 331) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 319) 
s_mov_b32 s15, s6 // (op 319) convert

s_cmp_lt_u32 s14, s15 // (op 319) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_319 // (op 319) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_319 if true)
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_319:






















s_barrier  // (op 1658) 






v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 

v_cvt_f16_f32 v12, v16 // (op 52) 
v_cvt_f16_f32 v13, v17 // (op 52) 
v_cvt_f16_f32 v14, v18 // (op 52) 
v_cvt_f16_f32 v15, v19 // (op 52) 


v_pack_b32_f16 v6, v12, v13 // (op 54) 
v_pack_b32_f16 v7, v14, v15 // (op 54) 


buffer_store_dwordx2 v[6:7], v8, s[44:47], 0 offen // (op 54) Store value


v_accvgpr_read v12, a4 // (op 517) 
v_accvgpr_read v13, a5 // (op 517) 
v_accvgpr_read v14, a6 // (op 517) 
v_accvgpr_read v15, a7 // (op 517) 
v_mul_f32 v16, v11, v12 // (op 517) 
v_mul_f32 v17, v11, v13 // (op 517) 
v_mul_f32 v18, v11, v14 // (op 517) 
v_mul_f32 v19, v11, v15 // (op 517) 

v_cvt_f16_f32 v12, v16 // (op 519) 
v_cvt_f16_f32 v13, v17 // (op 519) 
v_cvt_f16_f32 v14, v18 // (op 519) 
v_cvt_f16_f32 v15, v19 // (op 519) 


v_pack_b32_f16 v6, v12, v13 // (op 520) 
v_pack_b32_f16 v7, v14, v15 // (op 520) 


buffer_store_dwordx2 v[6:7], v8, s[44:47], 0 offen offset:64 // (op 520) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[6:7], v5, s[40:43], 0 offen offset:64 // (op 550) Load value
v_accvgpr_read v12, a4 // (op 546) 
v_accvgpr_read v13, a5 // (op 546) 
v_accvgpr_read v14, a6 // (op 546) 
v_accvgpr_read v15, a7 // (op 546) 
v_mul_f32 v16, v11, v12 // (op 546) 
v_mul_f32 v17, v11, v13 // (op 546) 
v_mul_f32 v18, v11, v14 // (op 546) 
v_mul_f32 v19, v11, v15 // (op 546) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v6 // (op 551) 
v_lshrrev_b32 v13, 16, v6 // (op 551) 
v_cvt_f32_f16 v13, v13 // (op 551) 
v_mul_f32 v20, v10, v12 // (op 551) 
v_mul_f32 v21, v10, v13 // (op 551) 
v_cvt_f32_f16 v12, v7 // (op 551) 
v_lshrrev_b32 v13, 16, v7 // (op 551) 
v_cvt_f32_f16 v13, v13 // (op 551) 
v_mul_f32 v22, v10, v12 // (op 551) 
v_mul_f32 v23, v10, v13 // (op 551) 


v_add_f32 v12, v20, v16 // (op 547) 
v_add_f32 v13, v21, v17 // (op 547) 
v_add_f32 v14, v22, v18 // (op 547) 
v_add_f32 v15, v23, v19 // (op 547) 


v_cvt_f16_f32 v16, v12 // (op 548) 
v_cvt_f16_f32 v17, v13 // (op 548) 
v_cvt_f16_f32 v18, v14 // (op 548) 
v_cvt_f16_f32 v19, v15 // (op 548) 


v_pack_b32_f16 v6, v16, v17 // (op 549) 
v_pack_b32_f16 v7, v18, v19 // (op 549) 


buffer_store_dwordx2 v[6:7], v8, s[44:47], 0 offen offset:64 // (op 549) Store value





buffer_load_dwordx2 v[6:7], v5, s[40:43], 0 offen // (op 138) Load value
v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v11, v12 // (op 134) 
v_mul_f32 v17, v11, v13 // (op 134) 
v_mul_f32 v18, v11, v14 // (op 134) 
v_mul_f32 v19, v11, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v6 // (op 139) 
v_lshrrev_b32 v13, 16, v6 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v20, v10, v12 // (op 139) 
v_mul_f32 v21, v10, v13 // (op 139) 
v_cvt_f32_f16 v12, v7 // (op 139) 
v_lshrrev_b32 v13, 16, v7 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v22, v10, v12 // (op 139) 
v_mul_f32 v23, v10, v13 // (op 139) 


v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 
v_add_f32 v14, v22, v18 // (op 135) 
v_add_f32 v15, v23, v19 // (op 135) 


v_cvt_f16_f32 v16, v12 // (op 136) 
v_cvt_f16_f32 v17, v13 // (op 136) 
v_cvt_f16_f32 v18, v14 // (op 136) 
v_cvt_f16_f32 v19, v15 // (op 136) 


v_pack_b32_f16 v6, v16, v17 // (op 137) 
v_pack_b32_f16 v7, v18, v19 // (op 137) 


buffer_store_dwordx2 v[6:7], v8, s[44:47], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.LRR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_end:
.size RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2, .LRR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_end-RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 20992 // lds bytes
  .amdhsa_accum_offset 64
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
    DestDataType: h
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
    args: [ { type: int64, semantic: StrideC0 },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressC },
            { type: int64, semantic: SizeFree1 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB } ]
    macrotile: [64, 16, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
    .symbol: RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 62
    .vgpr_count: 64
    .agpr_count: 8
    .max_flat_workgroup_size: 128
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 16
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_11
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 32
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 40
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_5
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

