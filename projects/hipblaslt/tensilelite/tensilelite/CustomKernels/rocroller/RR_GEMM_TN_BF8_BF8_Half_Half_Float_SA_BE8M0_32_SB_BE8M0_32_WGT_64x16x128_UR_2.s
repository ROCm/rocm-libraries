

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2,@function
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 15, v1 // (op 1322) 
v_and_b32 v3, 48, v1 // (op 1322) 
v_lshl_add_u32 v4, v0, 7, v3 // (op 1322) 


v_and_b32 v0, 15, v1 // (op 1481) 
v_and_b32 v3, 48, v1 // (op 1481) 
v_lshl_add_u32 v5, v0, 7, v3 // (op 1481) 


v_lshrrev_b32 v0, 6, v1 // (op 1209) 
v_and_b32 v3, 63, v1 // (op 1209) 
v_lshrrev_b32 v6, 4, v3 // (op 1209) 
v_lshlrev_b32 v7, 2, v6 // (op 1209) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1209) 


v_lshl_add_u32 v0, s16, 6, v6 // (op 1209) 

v_lshrrev_b32 v6, 2, v3 // (op 1209) 

v_and_b32 v3, 3, v6 // (op 1209) 

v_and_b32 v6, 3, v1 // (op 1209) 
v_lshl_add_u32 v7, v3, 2, v6 // (op 1209) 


v_lshl_add_u32 v3, s17, 4, v7 // (op 1209) 

v_mul_lo_u32 v6, v3, s2 // (op 1209) 

v_add_lshl_u32 v3, v0, v6, 1 // (op 1209) 


v_lshrrev_b32 v0, 6, v1 // (op 1270) 
v_and_b32 v6, 15, v1 // (op 1270) 
v_lshl_add_u32 v7, v0, 4, v6 // (op 1270) 


v_and_b32 v0, 48, v1 // (op 1270) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 1270) 


v_lshrrev_b32 v0, 6, v1 // (op 1380) 
v_and_b32 v7, 15, v1 // (op 1380) 
s_mul_i32 s1, s2, s9 // (op 1211) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1211) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1211) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1211) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1211) most significant: sum
s_add_u32 s1, s1, s19 // (op 1211) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1211) 

v_lshl_add_u32 v8, v0, 4, v7 // (op 1380) 


v_and_b32 v0, 63, v1 // (op 1380) 
v_lshrrev_b32 v7, 4, v0 // (op 1380) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 1380) 


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


v_lshrrev_b32 v7, 6, v1 // (op 1442) 
v_and_b32 v9, 15, v1 // (op 1442) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 1442) 


v_and_b32 v7, 48, v1 // (op 1442) 
v_lshl_add_u32 v9, v10, 7, v7 // (op 1442) 


v_lshrrev_b32 v7, 6, v1 // (op 1510) 
v_and_b32 v10, 15, v1 // (op 1510) 
s_mul_i32 s1, s4, s9 // (op 1398) most significant: low of low * high
v_lshl_add_u32 v11, v7, 4, v10 // (op 1510) 


v_and_b32 v7, 63, v1 // (op 1510) 
v_lshrrev_b32 v10, 4, v7 // (op 1510) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 1510) 


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
v_lshrrev_b32 v12, 3, v1 // (op 976) 
v_lshl_add_u32 v13, s16, 6, v12 // (op 976) 

v_and_b32 v12, -8, v1 // (op 976) 
v_sub_u32 v14, v1, v12 // (op 976) 

v_lshlrev_b32 v12, 4, v14 // (op 976) 

v_mov_b32 v15, 0 // (op 976) convert
v_mov_b32 v14, v12 // (op 976) convert

v_mul_lo_u32 v17, v13, s13 // (op 976) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 976) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 976) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 976) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 976) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 976) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 978) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 978) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 978) least significant
s_mul_hi_u32 s7, s12, s22 // (op 978) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 978) most significant: sum
s_add_u32 s1, s1, s7 // (op 978) most significant: sum


s_mov_b32 s48, s20 // (op 978) reg expression
s_mov_b32 s49, s21 // (op 978) reg expression
s_mov_b32 s50, s0 // (op 978) reg expression
s_mov_b32 s51, 131072 // (op 978) call()

v_lshrrev_b32 v12, 3, v1 // (op 995) 
v_lshl_add_u32 v13, s16, 6, v12 // (op 995) 

v_and_b32 v12, -8, v1 // (op 995) 
v_sub_u32 v14, v1, v12 // (op 995) 

v_lshlrev_b32 v12, 4, v14 // (op 995) 

v_mov_b32 v15, 0 // (op 995) convert
v_mov_b32 v14, v12 // (op 995) convert

v_mul_lo_u32 v19, v13, s13 // (op 995) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 995) most significant: high of low * low
v_mul_lo_u32 v18, v13, s12 // (op 995) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 995) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 995) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 995) most significant half



s_mul_i32 s1, s12, s23 // (op 997) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 997) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 997) least significant
s_mul_hi_u32 s7, s12, s22 // (op 997) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 997) most significant: sum
s_add_u32 s1, s1, s7 // (op 997) most significant: sum


s_mov_b32 s48, s20 // (op 997) reg expression
s_mov_b32 s49, s21 // (op 997) reg expression
s_mov_b32 s50, s0 // (op 997) reg expression
s_mov_b32 s51, 131072 // (op 997) call()

v_and_b32 v12, -8, v1 // (op 1040) 
v_sub_u32 v13, v1, v12 // (op 1040) 

v_lshlrev_b32 v12, 4, v13 // (op 1040) 

v_mov_b32 v15, 0 // (op 1040) convert
v_mov_b32 v14, v12 // (op 1040) convert

v_lshrrev_b32 v12, 3, v1 // (op 1040) 
v_lshl_add_u32 v13, s17, 4, v12 // (op 1040) 

v_mul_lo_u32 v21, v13, s25 // (op 1040) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 1040) most significant: high of low * low
v_mul_lo_u32 v20, v13, s24 // (op 1040) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 1040) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 1040) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 1040) most significant half



s_mul_i32 s1, s24, s9 // (op 1042) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1042) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1042) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1042) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1042) most significant: sum
s_add_u32 s1, s1, s7 // (op 1042) most significant: sum


s_mov_b32 s52, s26 // (op 1042) reg expression
s_mov_b32 s53, s27 // (op 1042) reg expression
s_mov_b32 s54, s0 // (op 1042) reg expression
s_mov_b32 s55, 131072 // (op 1042) call()

v_and_b32 v14, -8, v1 // (op 1059) 
v_sub_u32 v15, v1, v14 // (op 1059) 

v_lshlrev_b32 v14, 4, v15 // (op 1059) 

v_mov_b32 v21, 0 // (op 1059) convert
v_mov_b32 v20, v14 // (op 1059) convert

v_lshrrev_b32 v14, 3, v1 // (op 1059) 
v_lshl_add_u32 v15, s17, 4, v14 // (op 1059) 

v_mul_lo_u32 v23, v15, s25 // (op 1059) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1059) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 1059) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 1059) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 1059) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 1059) most significant half



s_mul_i32 s1, s24, s9 // (op 1061) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1061) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1061) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1061) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1061) most significant: sum
s_add_u32 s1, s1, s7 // (op 1061) most significant: sum


s_mov_b32 s52, s26 // (op 1061) reg expression
s_mov_b32 s53, s27 // (op 1061) reg expression
s_mov_b32 s54, s0 // (op 1061) reg expression
s_mov_b32 s55, 131072 // (op 1061) call()

v_lshrrev_b32 v20, 2, v1 // (op 1104) 
v_lshlrev_b32 v21, 1, v20 // (op 1104) 

v_lshl_add_u32 v20, s16, 6, v21 // (op 1104) 

v_and_b32 v21, -4, v1 // (op 1104) 
v_sub_u32 v22, v1, v21 // (op 1104) 

v_mul_lo_u32 v21, v20, s28 // (op 1104) 
v_add_u32 v21, v21, v22 // (op 1104) 


s_mul_i32 s1, s28, s23 // (op 1106) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1106) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1106) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1106) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1106) most significant: sum
s_add_u32 s1, s1, s7 // (op 1106) most significant: sum


s_mov_b32 s56, s30 // (op 1106) reg expression
s_mov_b32 s57, s31 // (op 1106) reg expression
s_mov_b32 s58, s0 // (op 1106) reg expression
s_mov_b32 s59, 131072 // (op 1106) call()

v_lshrrev_b32 v20, 2, v1 // (op 1123) 
v_lshlrev_b32 v22, 1, v20 // (op 1123) 

v_lshl_add_u32 v20, s16, 6, v22 // (op 1123) 

v_and_b32 v22, -4, v1 // (op 1123) 
v_sub_u32 v23, v1, v22 // (op 1123) 

v_mul_lo_u32 v22, v20, s28 // (op 1123) 
v_add_u32 v22, v22, v23 // (op 1123) 


s_mul_i32 s1, s28, s23 // (op 1125) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1125) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1125) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1125) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1125) most significant: sum
s_add_u32 s1, s1, s7 // (op 1125) most significant: sum


s_mov_b32 s56, s30 // (op 1125) reg expression
s_mov_b32 s57, s31 // (op 1125) reg expression
s_mov_b32 s58, s0 // (op 1125) reg expression
s_mov_b32 s59, 131072 // (op 1125) call()

v_and_b32 v20, 15, v1 // (op 1168) 
v_lshl_add_u32 v23, s17, 4, v20 // (op 1168) 

v_mul_lo_u32 v20, v23, s32 // (op 1168) 

v_and_b32 v23, 63, v1 // (op 1168) 
v_lshrrev_b32 v24, 4, v23 // (op 1168) 

v_add_u32 v23, v24, v20 // (op 1168) 


s_mul_i32 s1, s32, s9 // (op 1170) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1170) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1170) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1170) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1170) most significant: sum
s_add_u32 s1, s1, s7 // (op 1170) most significant: sum


s_mov_b32 s60, s34 // (op 1170) reg expression
s_mov_b32 s61, s35 // (op 1170) reg expression
s_mov_b32 s62, s0 // (op 1170) reg expression
s_mov_b32 s63, 131072 // (op 1170) call()

v_and_b32 v20, 15, v1 // (op 1187) 
v_lshl_add_u32 v24, s17, 4, v20 // (op 1187) 

v_mul_lo_u32 v20, v24, s32 // (op 1187) 

v_and_b32 v24, 63, v1 // (op 1187) 
v_lshrrev_b32 v25, 4, v24 // (op 1187) 

v_add_u32 v24, v25, v20 // (op 1187) 


s_mul_i32 s1, s32, s9 // (op 1189) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1189) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1189) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1189) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1189) most significant: sum
s_add_u32 s1, s1, s7 // (op 1189) most significant: sum


s_mov_b32 s60, s34 // (op 1189) reg expression
s_mov_b32 s61, s35 // (op 1189) reg expression
s_mov_b32 s62, s0 // (op 1189) reg expression
s_mov_b32 s63, 131072 // (op 1189) call()




buffer_load_ubyte v20, v23, s[60:63], 0 offen offset: 0 // (op 782) Load value
v_and_b32 v25, -8, v1 // (op 1310) 
v_sub_u32 v26, v1, v25 // (op 1310) 

v_lshlrev_b32 v25, 4, v26 // (op 1310) 

v_lshrrev_b32 v26, 3, v1 // (op 1310) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 1310) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1310) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 924) reg expression
buffer_load_dwordx4 v12, s[52:55], 0 offen offset: 0 lds // (op 924) Load value direct to lds

v_and_b32 v25, -8, v1 // (op 1253) 
v_sub_u32 v26, v1, v25 // (op 1253) 

v_lshlrev_b32 v25, 4, v26 // (op 1253) 

v_lshrrev_b32 v26, 3, v1 // (op 1253) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 1253) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1253) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 4 // (op 929) 
v_mov_b64 v[26:27], s[6:7] // (op 929) convert


v_mov_b64 v[28:29], v[16:17] // (op 929) 
s_add_u32 s1, 2048, s0 // (op 929) 
s_mov_b32 m0, s1 // (op 929) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 929) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 929) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 929) most significant half

s_add_u32 m0, m0, 2048 // (op 929) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 929) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 929) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 929) most significant half

s_add_u32 m0, m0, 2048 // (op 929) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 929) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 929) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 929) most significant half

s_add_u32 m0, m0, 2048 // (op 929) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 929) Load value direct to lds





v_mov_b64 v[26:27], s[28:29] // (op 792) convert

v_mov_b32 v25, v21 // (op 792) 
buffer_load_ubyte v28, v25, s[56:59], 0 offen offset: 0 // (op 792) Load value
v_add_u32 v25, v25, v26 // (op 792) 
buffer_load_ubyte v29, v25, s[56:59], 0 offen offset: 0 // (op 792) Load value





buffer_load_ubyte v25, v23, s[60:63], 0 offen offset:4 // (op 796) Load value

v_and_b32 v23, -8, v1 // (op 1469) 
v_sub_u32 v26, v1, v23 // (op 1469) 

v_lshlrev_b32 v23, 4, v26 // (op 1469) 

v_lshrrev_b32 v26, 3, v1 // (op 1469) 
v_lshl_add_u32 v27, v26, 7, v23 // (op 1469) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1469) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 10240, s0 // (op 934) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 934) reg expression
s_mov_b32 s6, 128 // (op 934) call()
buffer_load_dwordx4 v12, s[52:55], s6 offen offset: 0 lds // (op 934) Load value direct to lds




v_and_b32 v23, -8, v1 // (op 1425) 
v_sub_u32 v12, v1, v23 // (op 1425) 

v_lshlrev_b32 v13, 4, v12 // (op 1425) 

v_lshrrev_b32 v12, 3, v1 // (op 1425) 
v_lshl_add_u32 v23, v12, 7, v13 // (op 1425) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 1425) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 4 // (op 936) 
v_mov_b64 v[12:13], s[6:7] // (op 936) convert


v_mov_b64 v[26:27], v[16:17] // (op 936) 
s_add_u32 s1, 12288, s0 // (op 936) 
s_mov_b32 m0, s1 // (op 936) reg expression
s_mov_b32 s6, 128 // (op 936) call()
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 936) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v12 // (op 936) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 936) most significant half

s_add_u32 m0, m0, 2048 // (op 936) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 936) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v12 // (op 936) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 936) most significant half

s_add_u32 m0, m0, 2048 // (op 936) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 936) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v12 // (op 936) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 936) most significant half

s_add_u32 m0, m0, 2048 // (op 936) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 936) Load value direct to lds







v_mov_b64 v[12:13], s[28:29] // (op 806) convert

v_mov_b32 v23, v21 // (op 806) 
buffer_load_ubyte v16, v23, s[56:59], 0 offen offset:4 // (op 806) Load value
v_add_u32 v23, v23, v12 // (op 806) 
buffer_load_ubyte v17, v23, s[56:59], 0 offen offset:4 // (op 806) Load value



v_and_b32 v21, -4, v1 // (op 1363) 
v_sub_u32 v23, v1, v21 // (op 1363) 

v_lshrrev_b32 v21, 2, v1 // (op 1363) 
v_lshl_add_u32 v12, v21, 3, v23 // (op 1363) 





s_waitcnt vmcnt(9)
ds_write_b8 v12, v28 offset:20480 // (op 810) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v12, v29 offset:20484 // (op 810) Store local data 

s_barrier  // (op 1567) 



ds_read_b128 v[30:33], v4 // (op 833) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 833) Load local data 
s_barrier  // (op 1551) 



ds_read_b128 v[38:41], v6 offset:2048 // (op 824) Load local data 
ds_read_b128 v[42:45], v6 offset:2112 // (op 824) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1583) 



ds_read_u8 v21, v0 offset:20480 // (op 837) Load local data 



ds_read_b128 v[46:49], v6 offset:6144 // (op 829) Load local data 
ds_read_b128 v[50:53], v6 offset:6208 // (op 829) Load local data 



ds_read_u8 v23, v0 offset:20608 // (op 842) Load local data 
v_and_b32 v12, -8, v1 // (op 1225) 
v_sub_u32 v13, v1, v12 // (op 1225) 

v_lshlrev_b32 v12, 4, v13 // (op 1225) 

v_lshrrev_b32 v13, 3, v1 // (op 1225) 
v_lshl_add_u32 v26, v13, 7, v12 // (op 1225) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1225) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 1282) 
v_sub_u32 v13, v1, v12 // (op 1282) 

v_lshlrev_b32 v12, 4, v13 // (op 1282) 

v_lshrrev_b32 v13, 3, v1 // (op 1282) 
v_lshl_add_u32 v26, v13, 7, v12 // (op 1282) 


s_nop 0
v_readfirstlane_b32 s1, v26 // (op 1282) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 1334) 
v_sub_u32 v13, v1, v12 // (op 1334) 

v_lshrrev_b32 v12, 2, v1 // (op 1334) 
v_lshl_add_u32 v26, v12, 3, v13 // (op 1334) 


v_and_b32 v27, -8, v1 // (op 1410) 
v_sub_u32 v12, v1, v27 // (op 1410) 

v_lshlrev_b32 v13, 4, v12 // (op 1410) 

v_lshrrev_b32 v12, 3, v1 // (op 1410) 
v_lshl_add_u32 v27, v12, 7, v13 // (op 1410) 


s_nop 0
v_readfirstlane_b32 s6, v27 // (op 1410) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -8, v1 // (op 1454) 
v_sub_u32 v12, v1, v27 // (op 1454) 

v_lshlrev_b32 v13, 4, v12 // (op 1454) 

v_lshrrev_b32 v12, 3, v1 // (op 1454) 
v_lshl_add_u32 v27, v12, 7, v13 // (op 1454) 


s_nop 0
v_readfirstlane_b32 s7, v27 // (op 1454) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -4, v1 // (op 1493) 
v_sub_u32 v12, v1, v27 // (op 1493) 

v_lshrrev_b32 v13, 2, v1 // (op 1493) 
v_lshl_add_u32 v27, v13, 3, v12 // (op 1493) 


s_mov_b32 s10, 0 // (op 315) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 314) 
s_mov_b32 s11, s14 // (op 314) convert

s_and_b32 s14, s11, 4294967294 // (op 314) 

s_cmp_lt_u32 s10, s14 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_314 // (op 314) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_314 if false)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_314:
s_add_u32 s10, s10, 2 // (op 316) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v21, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 84) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[30:37], a[4:7], v23, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 350) 


v_mov_b32 v12, v24 // (op 82) 

buffer_load_ubyte v20, v12, s[60:63], 0 offen offset:8 // (op 82) Load value
s_barrier  // (op 1570) 


v_mov_b64 v[54:55], v[14:15] // (op 896) 

s_mov_b32 m0, s1 // (op 896) reg expression
s_mov_b32 s11, 256 // (op 896) call()
buffer_load_dwordx4 v54, s[52:55], s11 offen offset: 0 lds // (op 896) Load value direct to lds

s_barrier  // (op 1554) 

s_lshl_b64 s[14:15], s[12:13], 4 // (op 891) 
v_mov_b64 v[56:57], s[14:15] // (op 891) convert


v_mov_b64 v[58:59], v[18:19] // (op 891) 
v_mov_b64 v[60:61], v[58:59] // (op 891) 
s_add_u32 s11, 2048, s0 // (op 891) 
s_mov_b32 m0, s11 // (op 891) reg expression
s_mov_b32 s14, 256 // (op 891) call()
buffer_load_dwordx4 v60, s[48:51], s14 offen offset: 0 lds // (op 891) Load value direct to lds
v_add_co_u32 v60, s[18:19], v60, v56 // (op 891) least significant half
v_addc_co_u32 v61, s[18:19], 0, v61, s[18:19] // (op 891) most significant half

s_add_u32 m0, m0, 2048 // (op 891) 
buffer_load_dwordx4 v60, s[48:51], s14 offen offset: 0 lds // (op 891) Load value direct to lds
v_add_co_u32 v60, s[18:19], v60, v56 // (op 891) least significant half
v_addc_co_u32 v61, s[18:19], 0, v61, s[18:19] // (op 891) most significant half

s_add_u32 m0, m0, 2048 // (op 891) 
buffer_load_dwordx4 v60, s[48:51], s14 offen offset: 0 lds // (op 891) Load value direct to lds
v_add_co_u32 v60, s[18:19], v60, v56 // (op 891) least significant half
v_addc_co_u32 v61, s[18:19], 0, v61, s[18:19] // (op 891) most significant half

s_add_u32 m0, m0, 2048 // (op 891) 
buffer_load_dwordx4 v60, s[48:51], s14 offen offset: 0 lds // (op 891) Load value direct to lds





v_mov_b64 v[56:57], s[28:29] // (op 17) convert

v_mov_b32 v13, v22 // (op 17) 
v_mov_b32 v60, v13 // (op 17) 
buffer_load_ubyte v28, v60, s[56:59], 0 offen offset:8 // (op 17) Load value
v_add_u32 v60, v60, v56 // (op 17) 
buffer_load_ubyte v29, v60, s[56:59], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 1615) 



s_waitcnt vmcnt(9)
ds_write_b8 v27, v16 offset:20736 // (op 240) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v27, v17 offset:20740 // (op 240) Store local data 
s_barrier  // (op 1606) 



ds_read_b128 v[30:33], v5 offset:10240 // (op 233) Load local data 
ds_read_b128 v[34:37], v5 offset:10304 // (op 233) Load local data 
s_barrier  // (op 1600) 



ds_read_b128 v[38:41], v9 offset:12288 // (op 231) Load local data 
ds_read_b128 v[42:45], v9 offset:12352 // (op 231) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1612) 



ds_read_u8 v21, v7 offset:20736 // (op 235) Load local data 



ds_read_b128 v[46:49], v9 offset:16384 // (op 357) Load local data 
ds_read_b128 v[50:53], v9 offset:16448 // (op 357) Load local data 



ds_read_u8 v23, v7 offset:20864 // (op 361) Load local data 
s_barrier  // (op 1670) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v21, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 238) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[30:37], a[4:7], v23, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 351) 



buffer_load_ubyte v25, v12, s[60:63], 0 offen offset:12 // (op 237) Load value

s_barrier  // (op 1609) 
v_add_u32 v24, 8, v24 // (op 1179) 


s_add_u32 s11, 10240, s7 // (op 919) 

s_mov_b32 m0, s11 // (op 919) reg expression
s_mov_b32 s14, 384 // (op 919) call()
buffer_load_dwordx4 v54, s[52:55], s14 offen offset: 0 lds // (op 919) Load value direct to lds



s_barrier  // (op 1603) 
v_add_co_u32 v14, vcc, 256, v14 // (op 1051) least significant half

v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1051) most significant half
s_lshl_b64 s[14:15], s[12:13], 4 // (op 914) 
v_mov_b64 v[54:55], s[14:15] // (op 914) convert


v_mov_b64 v[56:57], v[58:59] // (op 914) 
s_add_u32 s11, 12288, s6 // (op 914) 
s_mov_b32 m0, s11 // (op 914) reg expression
s_mov_b32 s14, 384 // (op 914) call()
buffer_load_dwordx4 v56, s[48:51], s14 offen offset: 0 lds // (op 914) Load value direct to lds
v_add_co_u32 v56, s[18:19], v56, v54 // (op 914) least significant half
v_addc_co_u32 v57, s[18:19], 0, v57, s[18:19] // (op 914) most significant half

s_add_u32 m0, m0, 2048 // (op 914) 
buffer_load_dwordx4 v56, s[48:51], s14 offen offset: 0 lds // (op 914) Load value direct to lds
v_add_co_u32 v56, s[18:19], v56, v54 // (op 914) least significant half
v_addc_co_u32 v57, s[18:19], 0, v57, s[18:19] // (op 914) most significant half

s_add_u32 m0, m0, 2048 // (op 914) 
buffer_load_dwordx4 v56, s[48:51], s14 offen offset: 0 lds // (op 914) Load value direct to lds
v_add_co_u32 v56, s[18:19], v56, v54 // (op 914) least significant half
v_addc_co_u32 v57, s[18:19], 0, v57, s[18:19] // (op 914) most significant half

s_add_u32 m0, m0, 2048 // (op 914) 
buffer_load_dwordx4 v56, s[48:51], s14 offen offset: 0 lds // (op 914) Load value direct to lds






v_add_co_u32 v18, vcc, 256, v18 // (op 987) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 987) most significant half
v_mov_b64 v[54:55], s[28:29] // (op 242) convert

v_mov_b32 v12, v13 // (op 242) 
buffer_load_ubyte v16, v12, s[56:59], 0 offen offset:12 // (op 242) Load value
v_add_u32 v12, v12, v54 // (op 242) 
buffer_load_ubyte v17, v12, s[56:59], 0 offen offset:12 // (op 242) Load value



v_add_u32 v22, 8, v22 // (op 1115) 
s_barrier  // (op 1586) 



s_waitcnt vmcnt(9)
ds_write_b8 v26, v28 offset:20480 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v26, v29 offset:20484 // (op 21) Store local data 
s_barrier  // (op 1576) 



ds_read_b128 v[30:33], v4 // (op 72) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 72) Load local data 
s_barrier  // (op 1560) 



ds_read_b128 v[38:41], v6 offset:2048 // (op 67) Load local data 
ds_read_b128 v[42:45], v6 offset:2112 // (op 67) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1593) 



ds_read_u8 v21, v0 offset:20480 // (op 77) Load local data 



ds_read_b128 v[46:49], v6 offset:6144 // (op 344) Load local data 
ds_read_b128 v[50:53], v6 offset:6208 // (op 344) Load local data 



ds_read_u8 v23, v0 offset:20608 // (op 348) Load local data 
s_ashr_i64 s[14:15], s[36:37], 7 // (op 314) 
s_mov_b32 s11, s14 // (op 314) convert

s_and_b32 s14, s11, 4294967294 // (op 314) 

s_cmp_lt_u32 s10, s14 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_314 // (op 314) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_314 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_314:













s_barrier  // (op 1654) 
v_lshrrev_b32 v22, 3, v1 // (op 949) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 949) 
v_lshl_add_u32 v24, s16, 6, v22 // (op 949) 

s_lshr_b32 s6, s0, 1 // (op 949) 

v_and_b32 v22, -8, v1 // (op 949) 
v_sub_u32 v12, v1, v22 // (op 949) 

v_lshlrev_b32 v22, 4, v12 // (op 949) 

v_lshl_add_u32 v12, s6, 8, v22 // (op 949) 


v_mov_b32 v15, 0 // (op 949) convert
v_mov_b32 v14, v12 // (op 949) convert

v_mul_lo_u32 v13, v24, s13 // (op 949) most significant: low of low * high
v_mul_hi_u32 v22, v24, s12 // (op 949) most significant: high of low * low
v_mul_lo_u32 v12, v24, s12 // (op 949) least significant: low of low * low
v_add_u32 v13, v13, v22 // (op 949) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v14 // (op 949) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 949) most significant half



s_mul_i32 s1, s12, s23 // (op 951) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 951) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 951) least significant
s_mul_hi_u32 s7, s12, s22 // (op 951) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 951) most significant: sum
s_add_u32 s1, s1, s7 // (op 951) most significant: sum


s_mov_b32 s48, s20 // (op 951) reg expression
s_mov_b32 s49, s21 // (op 951) reg expression
s_mov_b32 s50, s0 // (op 951) reg expression
s_mov_b32 s51, 131072 // (op 951) call()

v_and_b32 v22, -8, v1 // (op 1017) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1017) 
v_sub_u32 v24, v1, v22 // (op 1017) 

s_lshr_b32 s6, s0, 1 // (op 1017) 

v_lshlrev_b32 v22, 4, v24 // (op 1017) 

v_lshl_add_u32 v24, s6, 8, v22 // (op 1017) 


v_mov_b32 v15, 0 // (op 1017) convert
v_mov_b32 v14, v24 // (op 1017) convert

v_lshrrev_b32 v22, 3, v1 // (op 1017) 
v_lshl_add_u32 v24, s17, 4, v22 // (op 1017) 

v_mul_lo_u32 v19, v24, s25 // (op 1017) most significant: low of low * high
v_mul_hi_u32 v20, v24, s24 // (op 1017) most significant: high of low * low
v_mul_lo_u32 v18, v24, s24 // (op 1017) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 1017) most significant: sum


v_add_co_u32 v26, s[0:1], v14, v18 // (op 1017) least significant half
v_addc_co_u32 v27, s[0:1], v15, v19, s[0:1] // (op 1017) most significant half



s_mul_i32 s1, s24, s9 // (op 1019) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1019) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1019) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1019) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1019) most significant: sum
s_add_u32 s1, s1, s7 // (op 1019) most significant: sum


s_mov_b32 s52, s26 // (op 1019) reg expression
s_mov_b32 s53, s27 // (op 1019) reg expression
s_mov_b32 s54, s0 // (op 1019) reg expression
s_mov_b32 s55, 131072 // (op 1019) call()

v_lshrrev_b32 v22, 2, v1 // (op 1081) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1081) 
v_lshlrev_b32 v24, 1, v22 // (op 1081) 

s_lshr_b32 s6, s0, 1 // (op 1081) 

v_lshl_add_u32 v22, s16, 6, v24 // (op 1081) 

v_and_b32 v24, -4, v1 // (op 1081) 
v_sub_u32 v14, v1, v24 // (op 1081) 

v_lshl_add_u32 v15, s6, 3, v14 // (op 1081) 


v_mul_lo_u32 v14, v22, s28 // (op 1081) 
v_add_u32 v14, v14, v15 // (op 1081) 


s_mul_i32 s1, s28, s23 // (op 1083) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1083) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1083) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1083) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1083) most significant: sum
s_add_u32 s1, s1, s7 // (op 1083) most significant: sum


s_mov_b32 s56, s30 // (op 1083) reg expression
s_mov_b32 s57, s31 // (op 1083) reg expression
s_mov_b32 s58, s0 // (op 1083) reg expression
s_mov_b32 s59, 131072 // (op 1083) call()

v_and_b32 v15, 63, v1 // (op 1145) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1145) 
v_lshrrev_b32 v22, 4, v15 // (op 1145) 

s_lshr_b32 s6, s0, 1 // (op 1145) 

v_and_b32 v15, 15, v1 // (op 1145) 
v_lshl_add_u32 v24, s6, 3, v22 // (op 1145) 


v_lshl_add_u32 v22, s17, 4, v15 // (op 1145) 

v_mul_lo_u32 v15, v22, s32 // (op 1145) 

v_add_u32 v22, v24, v15 // (op 1145) 


s_mul_i32 s1, s32, s9 // (op 1147) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1147) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1147) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1147) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1147) most significant: sum
s_add_u32 s1, s1, s7 // (op 1147) most significant: sum


s_mov_b32 s60, s34 // (op 1147) reg expression
s_mov_b32 s61, s35 // (op 1147) reg expression
s_mov_b32 s62, s0 // (op 1147) reg expression
s_mov_b32 s63, 131072 // (op 1147) call()

v_and_b32 v15, -8, v1 // (op 1238) 
v_sub_u32 v24, v1, v15 // (op 1238) 

v_lshlrev_b32 v15, 4, v24 // (op 1238) 

v_lshrrev_b32 v24, 3, v1 // (op 1238) 
v_lshl_add_u32 v18, v24, 7, v15 // (op 1238) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 1238) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -8, v1 // (op 1295) 
v_sub_u32 v24, v1, v15 // (op 1295) 

v_lshlrev_b32 v15, 4, v24 // (op 1295) 

v_lshrrev_b32 v24, 3, v1 // (op 1295) 
v_lshl_add_u32 v18, v24, 7, v15 // (op 1295) 


s_nop 0
v_readfirstlane_b32 s1, v18 // (op 1295) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -4, v1 // (op 1347) 
v_sub_u32 v24, v1, v15 // (op 1347) 

v_lshrrev_b32 v15, 2, v1 // (op 1347) 
v_lshl_add_u32 v18, v15, 3, v24 // (op 1347) 


s_ashr_i64 s[6:7], s[36:37], 7 // (op 320) 
s_and_b32 s10, s6, 4294967294 // (op 320) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 319) 
s_mov_b32 s11, s6 // (op 319) convert

s_cmp_lt_u32 s10, s11 // (op 319) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_319 // (op 319) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_319 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_319:
s_barrier  // (op 1557) 
s_add_u32 s10, s10, 1 // (op 321) 

s_lshl_b64 s[6:7], s[12:13], 4 // (op 901) 
v_mov_b64 v[16:17], s[6:7] // (op 901) convert


v_mov_b64 v[20:21], v[12:13] // (op 901) 
v_mov_b64 v[24:25], v[20:21] // (op 901) 
s_add_u32 s11, 2048, s0 // (op 901) 
s_mov_b32 m0, s11 // (op 901) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 901) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 901) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 901) most significant half

s_add_u32 m0, m0, 2048 // (op 901) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 901) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 901) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 901) most significant half

s_add_u32 m0, m0, 2048 // (op 901) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 901) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 901) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 901) most significant half

s_add_u32 m0, m0, 2048 // (op 901) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 901) Load value direct to lds




v_add_co_u32 v12, vcc, 128, v12 // (op 939) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 939) most significant half
s_barrier  // (op 1573) 


v_mov_b64 v[16:17], v[26:27] // (op 907) 

s_mov_b32 m0, s1 // (op 907) reg expression
buffer_load_dwordx4 v16, s[52:55], 0 offen offset: 0 lds // (op 907) Load value direct to lds

v_add_co_u32 v26, vcc, 128, v26 // (op 1009) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 1009) most significant half

v_mov_b64 v[16:17], s[28:29] // (op 184) convert

v_mov_b32 v15, v14 // (op 184) 
v_mov_b32 v19, v15 // (op 184) 
buffer_load_ubyte v20, v19, s[56:59], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v19, v19, v16 // (op 184) 
buffer_load_ubyte v21, v19, s[56:59], 0 offen offset: 0 // (op 184) Load value



v_add_u32 v14, 4, v14 // (op 1073) 
s_barrier  // (op 1589) 
s_waitcnt vmcnt(3)
s_barrier  // (op 1563) 






ds_read_b128 v[28:31], v6 offset:2048 // (op 173) Load local data 
ds_read_b128 v[32:35], v6 offset:2112 // (op 173) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v18, v20 offset:20480 // (op 182) Store local data 
s_barrier  // (op 1579) 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v21 offset:20484 // (op 182) Store local data 




ds_read_b128 v[36:39], v4 // (op 175) Load local data 
ds_read_b128 v[40:43], v4 offset:64 // (op 175) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1596) 



ds_read_u8 v15, v0 offset:20480 // (op 177) Load local data 


v_mov_b32 v16, v22 // (op 179) 

buffer_load_ubyte v17, v16, s[60:63], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v22, 4, v22 // (op 1137) 



ds_read_b128 v[44:47], v6 offset:6144 // (op 325) Load local data 
ds_read_b128 v[48:51], v6 offset:6208 // (op 325) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:43], a[0:3], v15, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 180) 





ds_read_u8 v15, v0 offset:20608 // (op 329) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[36:43], a[4:7], v15, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 331) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 319) 
s_mov_b32 s11, s6 // (op 319) convert

s_cmp_lt_u32 s10, s11 // (op 319) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_319 // (op 319) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_319 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_319:






















s_barrier  // (op 1658) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125
v_accvgpr_read v4, a0 // (op 46) 
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v12, v11, v4 // (op 46) 
v_mul_f32 v13, v11, v5 // (op 46) 
v_mul_f32 v14, v11, v6 // (op 46) 
v_mul_f32 v15, v11, v7 // (op 46) 

v_cvt_f16_f32 v4, v12 // (op 52) 
v_cvt_f16_f32 v5, v13 // (op 52) 
v_cvt_f16_f32 v6, v14 // (op 52) 
v_cvt_f16_f32 v7, v15 // (op 52) 


v_pack_b32_f16 v12, v4, v5 // (op 54) 
v_pack_b32_f16 v13, v6, v7 // (op 54) 


buffer_store_dwordx2 v[12:13], v8, s[44:47], 0 offen // (op 54) Store value


v_accvgpr_read v4, a4 // (op 517) 
v_accvgpr_read v5, a5 // (op 517) 
v_accvgpr_read v6, a6 // (op 517) 
v_accvgpr_read v7, a7 // (op 517) 
v_mul_f32 v12, v11, v4 // (op 517) 
v_mul_f32 v13, v11, v5 // (op 517) 
v_mul_f32 v14, v11, v6 // (op 517) 
v_mul_f32 v15, v11, v7 // (op 517) 

v_cvt_f16_f32 v4, v12 // (op 519) 
v_cvt_f16_f32 v5, v13 // (op 519) 
v_cvt_f16_f32 v6, v14 // (op 519) 
v_cvt_f16_f32 v7, v15 // (op 519) 


v_pack_b32_f16 v12, v4, v5 // (op 520) 
v_pack_b32_f16 v13, v6, v7 // (op 520) 


buffer_store_dwordx2 v[12:13], v8, s[44:47], 0 offen offset:64 // (op 520) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen offset:64 // (op 550) Load value
v_accvgpr_read v12, a4 // (op 546) 
v_accvgpr_read v13, a5 // (op 546) 
v_accvgpr_read v14, a6 // (op 546) 
v_accvgpr_read v15, a7 // (op 546) 
v_mul_f32 v16, v11, v12 // (op 546) 
v_mul_f32 v17, v11, v13 // (op 546) 
v_mul_f32 v18, v11, v14 // (op 546) 
v_mul_f32 v19, v11, v15 // (op 546) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 551) 
v_lshrrev_b32 v7, 16, v4 // (op 551) 
v_cvt_f32_f16 v7, v7 // (op 551) 
v_mul_f32 v12, v10, v6 // (op 551) 
v_mul_f32 v13, v10, v7 // (op 551) 
v_cvt_f32_f16 v6, v5 // (op 551) 
v_lshrrev_b32 v7, 16, v5 // (op 551) 
v_cvt_f32_f16 v7, v7 // (op 551) 
v_mul_f32 v14, v10, v6 // (op 551) 
v_mul_f32 v15, v10, v7 // (op 551) 


v_add_f32 v4, v12, v16 // (op 547) 
v_add_f32 v5, v13, v17 // (op 547) 
v_add_f32 v6, v14, v18 // (op 547) 
v_add_f32 v7, v15, v19 // (op 547) 


v_cvt_f16_f32 v12, v4 // (op 548) 
v_cvt_f16_f32 v13, v5 // (op 548) 
v_cvt_f16_f32 v14, v6 // (op 548) 
v_cvt_f16_f32 v15, v7 // (op 548) 


v_pack_b32_f16 v4, v12, v13 // (op 549) 
v_pack_b32_f16 v5, v14, v15 // (op 549) 


buffer_store_dwordx2 v[4:5], v8, s[44:47], 0 offen offset:64 // (op 549) Store value





buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 138) Load value
v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v11, v12 // (op 134) 
v_mul_f32 v17, v11, v13 // (op 134) 
v_mul_f32 v18, v11, v14 // (op 134) 
v_mul_f32 v19, v11, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 139) 
v_lshrrev_b32 v7, 16, v4 // (op 139) 
v_cvt_f32_f16 v7, v7 // (op 139) 
v_mul_f32 v12, v10, v6 // (op 139) 
v_mul_f32 v13, v10, v7 // (op 139) 
v_cvt_f32_f16 v6, v5 // (op 139) 
v_lshrrev_b32 v7, 16, v5 // (op 139) 
v_cvt_f32_f16 v7, v7 // (op 139) 
v_mul_f32 v14, v10, v6 // (op 139) 
v_mul_f32 v15, v10, v7 // (op 139) 


v_add_f32 v4, v12, v16 // (op 135) 
v_add_f32 v5, v13, v17 // (op 135) 
v_add_f32 v6, v14, v18 // (op 135) 
v_add_f32 v7, v15, v19 // (op 135) 


v_cvt_f16_f32 v12, v4 // (op 136) 
v_cvt_f16_f32 v13, v5 // (op 136) 
v_cvt_f16_f32 v14, v6 // (op 136) 
v_cvt_f16_f32 v15, v7 // (op 136) 


v_pack_b32_f16 v4, v12, v13 // (op 137) 
v_pack_b32_f16 v5, v14, v15 // (op 137) 


buffer_store_dwordx2 v[4:5], v8, s[44:47], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.LRR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_end:
.size RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2, .LRR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_end-RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
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
    args: [ { type: int64, semantic: StrideC0 },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressC },
            { type: int64, semantic: SizeFree1 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum } ]
    macrotile: [64, 16, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
    .symbol: RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 62
    .agpr_count: 8
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 32
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 40
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 48
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 52
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

