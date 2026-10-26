

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2,@function
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 1314) 
v_and_b32 v3, 1, v0 // (op 1314) 

v_and_b32 v0, 15, v1 // (op 1314) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 1314) 


v_and_b32 v0, 48, v1 // (op 1314) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 1314) 


v_lshrrev_b32 v0, 6, v1 // (op 1473) 
v_and_b32 v4, 1, v0 // (op 1473) 

v_and_b32 v0, 15, v1 // (op 1473) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 1473) 


v_and_b32 v0, 48, v1 // (op 1473) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 1473) 


v_lshrrev_b32 v0, 7, v1 // (op 1201) 
v_and_b32 v5, 63, v1 // (op 1201) 
v_lshrrev_b32 v6, 4, v5 // (op 1201) 
v_lshlrev_b32 v7, 2, v6 // (op 1201) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1201) 


v_lshl_add_u32 v0, s16, 6, v6 // (op 1201) 

v_lshrrev_b32 v6, 2, v5 // (op 1201) 

v_and_b32 v5, 3, v6 // (op 1201) 

v_and_b32 v6, 3, v1 // (op 1201) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1201) 


v_lshrrev_b32 v5, 6, v1 // (op 1201) 
v_and_b32 v6, 1, v5 // (op 1201) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1201) 


v_lshl_add_u32 v6, s17, 5, v5 // (op 1201) 

v_mul_lo_u32 v5, v6, s2 // (op 1201) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 1201) 


v_lshrrev_b32 v0, 7, v1 // (op 1262) 
v_and_b32 v5, 15, v1 // (op 1262) 
v_lshl_add_u32 v7, v0, 4, v5 // (op 1262) 


v_and_b32 v0, 48, v1 // (op 1262) 
v_lshl_add_u32 v5, v7, 7, v0 // (op 1262) 


v_lshrrev_b32 v0, 7, v1 // (op 1372) 
v_and_b32 v7, 15, v1 // (op 1372) 
s_mul_i32 s1, s2, s9 // (op 1203) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1203) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1203) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1203) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1203) most significant: sum
s_add_u32 s1, s1, s19 // (op 1203) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1203) 

v_lshl_add_u32 v8, v0, 4, v7 // (op 1372) 


v_and_b32 v0, 63, v1 // (op 1372) 
v_lshrrev_b32 v7, 4, v0 // (op 1372) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 1372) 


v_lshrrev_b32 v7, 7, v1 // (op 1388) 
v_and_b32 v8, 63, v1 // (op 1388) 
v_lshrrev_b32 v9, 4, v8 // (op 1388) 
v_lshlrev_b32 v10, 2, v9 // (op 1388) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1388) 


v_lshl_add_u32 v7, s16, 6, v9 // (op 1388) 

v_lshrrev_b32 v9, 2, v8 // (op 1388) 

v_and_b32 v8, 3, v9 // (op 1388) 

v_and_b32 v9, 3, v1 // (op 1388) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1388) 


v_lshrrev_b32 v8, 6, v1 // (op 1388) 
v_and_b32 v9, 1, v8 // (op 1388) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 1388) 


v_lshl_add_u32 v9, s17, 5, v8 // (op 1388) 

v_mul_lo_u32 v8, v9, s4 // (op 1388) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 1388) 


v_lshrrev_b32 v7, 7, v1 // (op 1434) 
v_and_b32 v8, 15, v1 // (op 1434) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 1434) 


v_and_b32 v7, 48, v1 // (op 1434) 
v_lshl_add_u32 v8, v10, 7, v7 // (op 1434) 


v_lshrrev_b32 v7, 7, v1 // (op 1502) 
v_and_b32 v10, 15, v1 // (op 1502) 
s_mul_i32 s1, s4, s9 // (op 1390) most significant: low of low * high
v_lshl_add_u32 v11, v7, 4, v10 // (op 1502) 


v_and_b32 v7, 63, v1 // (op 1502) 
v_lshrrev_b32 v10, 4, v7 // (op 1502) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 1502) 


s_mul_i32 s38, s5, s8 // (op 1390) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1390) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1390) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1390) most significant: sum
s_add_u32 s1, s1, s39 // (op 1390) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 1390) 

s_mov_b32 s40, s6 // (op 1203) reg expression
s_mov_b32 s41, s7 // (op 1203) reg expression
s_mov_b32 s42, s18 // (op 1203) reg expression
s_mov_b32 s43, 131072 // (op 1203) call()

s_mov_b32 s44, s10 // (op 1390) reg expression

s_mov_b32 s45, s11 // (op 1390) reg expression
s_mov_b32 s46, s38 // (op 1390) reg expression
s_mov_b32 s47, 131072 // (op 1390) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 309) call()
v_accvgpr_write a5, 0.00000 // (op 309) call()
v_accvgpr_write a6, 0.00000 // (op 309) call()
v_accvgpr_write a7, 0.00000 // (op 309) call()
v_lshrrev_b32 v12, 3, v1 // (op 968) 
v_lshl_add_u32 v13, s16, 6, v12 // (op 968) 

v_and_b32 v12, -8, v1 // (op 968) 
v_sub_u32 v14, v1, v12 // (op 968) 

v_lshlrev_b32 v12, 4, v14 // (op 968) 

v_mov_b32 v15, 0 // (op 968) convert
v_mov_b32 v14, v12 // (op 968) convert

v_mul_lo_u32 v17, v13, s13 // (op 968) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 968) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 968) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 968) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 968) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 968) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 970) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 970) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 970) least significant
s_mul_hi_u32 s7, s12, s22 // (op 970) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 970) most significant: sum
s_add_u32 s1, s1, s7 // (op 970) most significant: sum


s_mov_b32 s48, s20 // (op 970) reg expression
s_mov_b32 s49, s21 // (op 970) reg expression
s_mov_b32 s50, s0 // (op 970) reg expression
s_mov_b32 s51, 131072 // (op 970) call()

v_lshrrev_b32 v12, 3, v1 // (op 987) 
v_lshl_add_u32 v13, s16, 6, v12 // (op 987) 

v_and_b32 v12, -8, v1 // (op 987) 
v_sub_u32 v14, v1, v12 // (op 987) 

v_lshlrev_b32 v12, 4, v14 // (op 987) 

v_mov_b32 v15, 0 // (op 987) convert
v_mov_b32 v14, v12 // (op 987) convert

v_mul_lo_u32 v19, v13, s13 // (op 987) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 987) most significant: high of low * low
v_mul_lo_u32 v18, v13, s12 // (op 987) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 987) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 987) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 987) most significant half



s_mul_i32 s1, s12, s23 // (op 989) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 989) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 989) least significant
s_mul_hi_u32 s7, s12, s22 // (op 989) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 989) most significant: sum
s_add_u32 s1, s1, s7 // (op 989) most significant: sum


s_mov_b32 s48, s20 // (op 989) reg expression
s_mov_b32 s49, s21 // (op 989) reg expression
s_mov_b32 s50, s0 // (op 989) reg expression
s_mov_b32 s51, 131072 // (op 989) call()

v_and_b32 v12, -8, v1 // (op 1032) 
v_sub_u32 v13, v1, v12 // (op 1032) 

v_lshlrev_b32 v12, 4, v13 // (op 1032) 

v_mov_b32 v15, 0 // (op 1032) convert
v_mov_b32 v14, v12 // (op 1032) convert

v_lshrrev_b32 v12, 3, v1 // (op 1032) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 1032) 

v_mul_lo_u32 v21, v13, s25 // (op 1032) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 1032) most significant: high of low * low
v_mul_lo_u32 v20, v13, s24 // (op 1032) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 1032) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 1032) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 1032) most significant half



s_mul_i32 s1, s24, s9 // (op 1034) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1034) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1034) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1034) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1034) most significant: sum
s_add_u32 s1, s1, s7 // (op 1034) most significant: sum


s_mov_b32 s52, s26 // (op 1034) reg expression
s_mov_b32 s53, s27 // (op 1034) reg expression
s_mov_b32 s54, s0 // (op 1034) reg expression
s_mov_b32 s55, 131072 // (op 1034) call()

v_and_b32 v14, -8, v1 // (op 1051) 
v_sub_u32 v15, v1, v14 // (op 1051) 

v_lshlrev_b32 v14, 4, v15 // (op 1051) 

v_mov_b32 v21, 0 // (op 1051) convert
v_mov_b32 v20, v14 // (op 1051) convert

v_lshrrev_b32 v14, 3, v1 // (op 1051) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 1051) 

v_mul_lo_u32 v23, v15, s25 // (op 1051) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1051) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 1051) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 1051) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 1051) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 1051) most significant half



s_mul_i32 s1, s24, s9 // (op 1053) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1053) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1053) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1053) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1053) most significant: sum
s_add_u32 s1, s1, s7 // (op 1053) most significant: sum


s_mov_b32 s52, s26 // (op 1053) reg expression
s_mov_b32 s53, s27 // (op 1053) reg expression
s_mov_b32 s54, s0 // (op 1053) reg expression
s_mov_b32 s55, 131072 // (op 1053) call()

v_lshrrev_b32 v20, 2, v1 // (op 1096) 
v_lshl_add_u32 v21, s16, 6, v20 // (op 1096) 

v_and_b32 v20, -4, v1 // (op 1096) 
v_sub_u32 v22, v1, v20 // (op 1096) 

v_mul_lo_u32 v20, v21, s28 // (op 1096) 
v_add_u32 v20, v20, v22 // (op 1096) 


s_mul_i32 s1, s28, s23 // (op 1098) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1098) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1098) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1098) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1098) most significant: sum
s_add_u32 s1, s1, s7 // (op 1098) most significant: sum


s_mov_b32 s56, s30 // (op 1098) reg expression
s_mov_b32 s57, s31 // (op 1098) reg expression
s_mov_b32 s58, s0 // (op 1098) reg expression
s_mov_b32 s59, 131072 // (op 1098) call()

v_lshrrev_b32 v21, 2, v1 // (op 1115) 
v_lshl_add_u32 v22, s16, 6, v21 // (op 1115) 

v_and_b32 v21, -4, v1 // (op 1115) 
v_sub_u32 v23, v1, v21 // (op 1115) 

v_mul_lo_u32 v21, v22, s28 // (op 1115) 
v_add_u32 v21, v21, v23 // (op 1115) 


s_mul_i32 s1, s28, s23 // (op 1117) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1117) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1117) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1117) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1117) most significant: sum
s_add_u32 s1, s1, s7 // (op 1117) most significant: sum


s_mov_b32 s56, s30 // (op 1117) reg expression
s_mov_b32 s57, s31 // (op 1117) reg expression
s_mov_b32 s58, s0 // (op 1117) reg expression
s_mov_b32 s59, 131072 // (op 1117) call()

v_lshrrev_b32 v22, 6, v1 // (op 1160) 
v_and_b32 v23, 1, v22 // (op 1160) 

v_and_b32 v22, 15, v1 // (op 1160) 
v_lshl_add_u32 v24, v23, 4, v22 // (op 1160) 


v_lshl_add_u32 v22, s17, 5, v24 // (op 1160) 

v_mul_lo_u32 v23, v22, s32 // (op 1160) 

v_and_b32 v22, 63, v1 // (op 1160) 
v_lshrrev_b32 v24, 4, v22 // (op 1160) 

v_add_u32 v22, v24, v23 // (op 1160) 


s_mul_i32 s1, s32, s9 // (op 1162) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1162) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1162) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1162) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1162) most significant: sum
s_add_u32 s1, s1, s7 // (op 1162) most significant: sum


s_mov_b32 s60, s34 // (op 1162) reg expression
s_mov_b32 s61, s35 // (op 1162) reg expression
s_mov_b32 s62, s0 // (op 1162) reg expression
s_mov_b32 s63, 131072 // (op 1162) call()

v_lshrrev_b32 v23, 6, v1 // (op 1179) 
v_and_b32 v24, 1, v23 // (op 1179) 

v_and_b32 v23, 15, v1 // (op 1179) 
v_lshl_add_u32 v25, v24, 4, v23 // (op 1179) 


v_lshl_add_u32 v23, s17, 5, v25 // (op 1179) 

v_mul_lo_u32 v24, v23, s32 // (op 1179) 

v_and_b32 v23, 63, v1 // (op 1179) 
v_lshrrev_b32 v25, 4, v23 // (op 1179) 

v_add_u32 v23, v25, v24 // (op 1179) 


s_mul_i32 s1, s32, s9 // (op 1181) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1181) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1181) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1181) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1181) most significant: sum
s_add_u32 s1, s1, s7 // (op 1181) most significant: sum


s_mov_b32 s60, s34 // (op 1181) reg expression
s_mov_b32 s61, s35 // (op 1181) reg expression
s_mov_b32 s62, s0 // (op 1181) reg expression
s_mov_b32 s63, 131072 // (op 1181) call()




buffer_load_ubyte v24, v22, s[60:63], 0 offen offset: 0 // (op 774) Load value
v_and_b32 v25, -8, v1 // (op 1302) 
v_sub_u32 v26, v1, v25 // (op 1302) 

v_lshlrev_b32 v25, 4, v26 // (op 1302) 

v_lshrrev_b32 v26, 3, v1 // (op 1302) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 1302) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1302) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 916) reg expression
buffer_load_dwordx4 v12, s[52:55], 0 offen offset: 0 lds // (op 916) Load value direct to lds

v_and_b32 v25, -8, v1 // (op 1245) 
v_sub_u32 v26, v1, v25 // (op 1245) 

v_lshlrev_b32 v25, 4, v26 // (op 1245) 

v_lshrrev_b32 v26, 3, v1 // (op 1245) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 1245) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1245) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 921) 
v_mov_b64 v[26:27], s[6:7] // (op 921) convert


v_mov_b64 v[28:29], v[16:17] // (op 921) 
s_add_u32 s1, 4096, s0 // (op 921) 
s_mov_b32 m0, s1 // (op 921) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 921) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 921) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 921) most significant half

s_add_u32 m0, m0, 4096 // (op 921) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 921) Load value direct to lds







buffer_load_ubyte v25, v20, s[56:59], 0 offen offset: 0 // (op 784) Load value



buffer_load_ubyte v26, v22, s[60:63], 0 offen offset:4 // (op 788) Load value

v_and_b32 v22, -8, v1 // (op 1461) 
v_sub_u32 v27, v1, v22 // (op 1461) 

v_lshlrev_b32 v22, 4, v27 // (op 1461) 

v_lshrrev_b32 v27, 3, v1 // (op 1461) 
v_lshl_add_u32 v28, v27, 7, v22 // (op 1461) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1461) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 12288, s0 // (op 926) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 926) reg expression
s_mov_b32 s6, 128 // (op 926) call()
buffer_load_dwordx4 v12, s[52:55], s6 offen offset: 0 lds // (op 926) Load value direct to lds




v_and_b32 v22, -8, v1 // (op 1417) 
v_sub_u32 v12, v1, v22 // (op 1417) 

v_lshlrev_b32 v13, 4, v12 // (op 1417) 

v_lshrrev_b32 v12, 3, v1 // (op 1417) 
v_lshl_add_u32 v22, v12, 7, v13 // (op 1417) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 1417) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 928) 
v_mov_b64 v[12:13], s[6:7] // (op 928) convert


v_mov_b64 v[28:29], v[16:17] // (op 928) 
s_add_u32 s1, 16384, s0 // (op 928) 
s_mov_b32 m0, s1 // (op 928) reg expression
s_mov_b32 s6, 128 // (op 928) call()
buffer_load_dwordx4 v28, s[48:51], s6 offen offset: 0 lds // (op 928) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v12 // (op 928) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 928) most significant half

s_add_u32 m0, m0, 4096 // (op 928) 
buffer_load_dwordx4 v28, s[48:51], s6 offen offset: 0 lds // (op 928) Load value direct to lds









buffer_load_ubyte v22, v20, s[56:59], 0 offen offset:4 // (op 798) Load value

v_and_b32 v20, -4, v1 // (op 1355) 
v_sub_u32 v12, v1, v20 // (op 1355) 

v_and_b32 v13, 4294967292, v1 // (op 1355) 
v_add_u32 v20, v13, v12 // (op 1355) 





s_waitcnt vmcnt(5)
ds_write_b8 v20, v25 offset:24576 // (op 802) Store local data 

s_barrier  // (op 1559) 



ds_read_b128 v[28:31], v3 // (op 825) Load local data 
ds_read_b128 v[32:35], v3 offset:64 // (op 825) Load local data 
s_barrier  // (op 1543) 



ds_read_b128 v[36:39], v5 offset:4096 // (op 816) Load local data 
ds_read_b128 v[40:43], v5 offset:4160 // (op 816) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1575) 



ds_read_u8 v20, v0 offset:24576 // (op 829) Load local data 



ds_read_b128 v[44:47], v5 offset:8192 // (op 821) Load local data 
ds_read_b128 v[48:51], v5 offset:8256 // (op 821) Load local data 



ds_read_u8 v27, v0 offset:24704 // (op 834) Load local data 
v_and_b32 v12, -8, v1 // (op 1217) 
v_sub_u32 v13, v1, v12 // (op 1217) 

v_lshlrev_b32 v12, 4, v13 // (op 1217) 

v_lshrrev_b32 v13, 3, v1 // (op 1217) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 1217) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 1217) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 1274) 
v_sub_u32 v13, v1, v12 // (op 1274) 

v_lshlrev_b32 v12, 4, v13 // (op 1274) 

v_lshrrev_b32 v13, 3, v1 // (op 1274) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 1274) 


s_nop 0
v_readfirstlane_b32 s1, v16 // (op 1274) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 1326) 
v_sub_u32 v13, v1, v12 // (op 1326) 

v_and_b32 v12, 4294967292, v1 // (op 1326) 
v_add_u32 v16, v12, v13 // (op 1326) 


v_and_b32 v17, -8, v1 // (op 1402) 
v_sub_u32 v12, v1, v17 // (op 1402) 

v_lshlrev_b32 v13, 4, v12 // (op 1402) 

v_lshrrev_b32 v12, 3, v1 // (op 1402) 
v_lshl_add_u32 v17, v12, 7, v13 // (op 1402) 


s_nop 0
v_readfirstlane_b32 s6, v17 // (op 1402) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -8, v1 // (op 1446) 
v_sub_u32 v12, v1, v17 // (op 1446) 

v_lshlrev_b32 v13, 4, v12 // (op 1446) 

v_lshrrev_b32 v12, 3, v1 // (op 1446) 
v_lshl_add_u32 v17, v12, 7, v13 // (op 1446) 


s_nop 0
v_readfirstlane_b32 s7, v17 // (op 1446) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -4, v1 // (op 1485) 
v_sub_u32 v12, v1, v17 // (op 1485) 

v_and_b32 v13, 4294967292, v1 // (op 1485) 
v_add_u32 v17, v13, v12 // (op 1485) 


s_mov_b32 s10, 0 // (op 311) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 310) 
s_mov_b32 s11, s14 // (op 310) convert

s_and_b32 s14, s11, 4294967294 // (op 310) 

s_cmp_lt_u32 s10, s14 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_310 // (op 310) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_310 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoop_310:
s_add_u32 s10, s10, 2 // (op 312) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[28:35], a[0:3], v20, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 82) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[28:35], a[4:7], v27, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 346) 


v_mov_b32 v12, v23 // (op 80) 

buffer_load_ubyte v24, v12, s[60:63], 0 offen offset:8 // (op 80) Load value
s_barrier  // (op 1562) 


v_mov_b64 v[52:53], v[14:15] // (op 888) 

s_mov_b32 m0, s1 // (op 888) reg expression
s_mov_b32 s11, 256 // (op 888) call()
buffer_load_dwordx4 v52, s[52:55], s11 offen offset: 0 lds // (op 888) Load value direct to lds

s_barrier  // (op 1546) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 883) 
v_mov_b64 v[54:55], s[14:15] // (op 883) convert


v_mov_b64 v[56:57], v[18:19] // (op 883) 
v_mov_b64 v[58:59], v[56:57] // (op 883) 
s_add_u32 s11, 4096, s0 // (op 883) 
s_mov_b32 m0, s11 // (op 883) reg expression
s_mov_b32 s14, 256 // (op 883) call()
buffer_load_dwordx4 v58, s[48:51], s14 offen offset: 0 lds // (op 883) Load value direct to lds
v_add_co_u32 v58, s[18:19], v58, v54 // (op 883) least significant half
v_addc_co_u32 v59, s[18:19], 0, v59, s[18:19] // (op 883) most significant half

s_add_u32 m0, m0, 4096 // (op 883) 
buffer_load_dwordx4 v58, s[48:51], s14 offen offset: 0 lds // (op 883) Load value direct to lds






v_mov_b32 v13, v21 // (op 17) 

buffer_load_ubyte v25, v13, s[56:59], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 1607) 



s_waitcnt vmcnt(5)
ds_write_b8 v17, v22 offset:24832 // (op 236) Store local data 
s_barrier  // (op 1598) 



ds_read_b128 v[28:31], v4 offset:12288 // (op 229) Load local data 
ds_read_b128 v[32:35], v4 offset:12352 // (op 229) Load local data 
s_barrier  // (op 1592) 



ds_read_b128 v[36:39], v8 offset:16384 // (op 227) Load local data 
ds_read_b128 v[40:43], v8 offset:16448 // (op 227) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1604) 



ds_read_u8 v20, v7 offset:24832 // (op 231) Load local data 



ds_read_b128 v[44:47], v8 offset:20480 // (op 353) Load local data 
ds_read_b128 v[48:51], v8 offset:20544 // (op 353) Load local data 



ds_read_u8 v27, v7 offset:24960 // (op 357) Load local data 
s_barrier  // (op 1658) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[28:35], a[0:3], v20, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 234) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[28:35], a[4:7], v27, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 347) 



buffer_load_ubyte v26, v12, s[60:63], 0 offen offset:12 // (op 233) Load value

s_barrier  // (op 1601) 
v_add_u32 v23, 8, v23 // (op 1171) 


s_add_u32 s11, 12288, s7 // (op 911) 

s_mov_b32 m0, s11 // (op 911) reg expression
s_mov_b32 s14, 384 // (op 911) call()
buffer_load_dwordx4 v52, s[52:55], s14 offen offset: 0 lds // (op 911) Load value direct to lds



s_barrier  // (op 1595) 
v_add_co_u32 v14, vcc, 256, v14 // (op 1043) least significant half

v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1043) most significant half
s_lshl_b64 s[14:15], s[12:13], 5 // (op 906) 
v_mov_b64 v[52:53], s[14:15] // (op 906) convert


v_mov_b64 v[54:55], v[56:57] // (op 906) 
s_add_u32 s11, 16384, s6 // (op 906) 
s_mov_b32 m0, s11 // (op 906) reg expression
s_mov_b32 s14, 384 // (op 906) call()
buffer_load_dwordx4 v54, s[48:51], s14 offen offset: 0 lds // (op 906) Load value direct to lds
v_add_co_u32 v54, s[18:19], v54, v52 // (op 906) least significant half
v_addc_co_u32 v55, s[18:19], 0, v55, s[18:19] // (op 906) most significant half

s_add_u32 m0, m0, 4096 // (op 906) 
buffer_load_dwordx4 v54, s[48:51], s14 offen offset: 0 lds // (op 906) Load value direct to lds








v_add_co_u32 v18, vcc, 256, v18 // (op 979) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 979) most significant half
buffer_load_ubyte v22, v13, s[56:59], 0 offen offset:12 // (op 238) Load value

v_add_u32 v21, 8, v21 // (op 1107) 
s_barrier  // (op 1578) 



s_waitcnt vmcnt(5)
ds_write_b8 v16, v25 offset:24576 // (op 21) Store local data 
s_barrier  // (op 1568) 



ds_read_b128 v[28:31], v3 // (op 70) Load local data 
ds_read_b128 v[32:35], v3 offset:64 // (op 70) Load local data 
s_barrier  // (op 1552) 



ds_read_b128 v[36:39], v5 offset:4096 // (op 65) Load local data 
ds_read_b128 v[40:43], v5 offset:4160 // (op 65) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1585) 



ds_read_u8 v20, v0 offset:24576 // (op 75) Load local data 



ds_read_b128 v[44:47], v5 offset:8192 // (op 340) Load local data 
ds_read_b128 v[48:51], v5 offset:8256 // (op 340) Load local data 



ds_read_u8 v27, v0 offset:24704 // (op 344) Load local data 
s_ashr_i64 s[14:15], s[36:37], 7 // (op 310) 
s_mov_b32 s11, s14 // (op 310) convert

s_and_b32 s14, s11, 4294967294 // (op 310) 

s_cmp_lt_u32 s10, s14 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoop_310 // (op 310) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoop_310 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_310:













s_barrier  // (op 1642) 
v_lshrrev_b32 v21, 3, v1 // (op 941) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 941) 
v_lshl_add_u32 v12, s16, 6, v21 // (op 941) 

s_lshr_b32 s6, s0, 1 // (op 941) 

v_and_b32 v21, -8, v1 // (op 941) 
v_sub_u32 v13, v1, v21 // (op 941) 

v_lshlrev_b32 v21, 4, v13 // (op 941) 

v_lshl_add_u32 v13, s6, 8, v21 // (op 941) 


v_mov_b32 v25, 0 // (op 941) convert
v_mov_b32 v24, v13 // (op 941) convert

v_mul_lo_u32 v19, v12, s13 // (op 941) most significant: low of low * high
v_mul_hi_u32 v21, v12, s12 // (op 941) most significant: high of low * low
v_mul_lo_u32 v18, v12, s12 // (op 941) least significant: low of low * low
v_add_u32 v19, v19, v21 // (op 941) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v24 // (op 941) least significant half
v_addc_co_u32 v19, s[0:1], v19, v25, s[0:1] // (op 941) most significant half



s_mul_i32 s1, s12, s23 // (op 943) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 943) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 943) least significant
s_mul_hi_u32 s7, s12, s22 // (op 943) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 943) most significant: sum
s_add_u32 s1, s1, s7 // (op 943) most significant: sum


s_mov_b32 s48, s20 // (op 943) reg expression
s_mov_b32 s49, s21 // (op 943) reg expression
s_mov_b32 s50, s0 // (op 943) reg expression
s_mov_b32 s51, 131072 // (op 943) call()

v_and_b32 v21, -8, v1 // (op 1009) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1009) 
v_sub_u32 v12, v1, v21 // (op 1009) 

s_lshr_b32 s6, s0, 1 // (op 1009) 

v_lshlrev_b32 v21, 4, v12 // (op 1009) 

v_lshl_add_u32 v12, s6, 8, v21 // (op 1009) 


v_mov_b32 v25, 0 // (op 1009) convert
v_mov_b32 v24, v12 // (op 1009) convert

v_lshrrev_b32 v21, 3, v1 // (op 1009) 
v_lshl_add_u32 v23, s17, 5, v21 // (op 1009) 

v_mul_lo_u32 v13, v23, s25 // (op 1009) most significant: low of low * high
v_mul_hi_u32 v21, v23, s24 // (op 1009) most significant: high of low * low
v_mul_lo_u32 v12, v23, s24 // (op 1009) least significant: low of low * low
v_add_u32 v13, v13, v21 // (op 1009) most significant: sum


v_add_co_u32 v14, s[0:1], v24, v12 // (op 1009) least significant half
v_addc_co_u32 v15, s[0:1], v25, v13, s[0:1] // (op 1009) most significant half



s_mul_i32 s1, s24, s9 // (op 1011) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1011) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1011) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1011) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1011) most significant: sum
s_add_u32 s1, s1, s7 // (op 1011) most significant: sum


s_mov_b32 s52, s26 // (op 1011) reg expression
s_mov_b32 s53, s27 // (op 1011) reg expression
s_mov_b32 s54, s0 // (op 1011) reg expression
s_mov_b32 s55, 131072 // (op 1011) call()

v_lshrrev_b32 v21, 2, v1 // (op 1073) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1073) 
v_lshl_add_u32 v12, s16, 6, v21 // (op 1073) 

s_lshr_b32 s6, s0, 1 // (op 1073) 

v_and_b32 v13, -4, v1 // (op 1073) 
v_sub_u32 v21, v1, v13 // (op 1073) 

v_lshl_add_u32 v13, s6, 3, v21 // (op 1073) 


v_mul_lo_u32 v21, v12, s28 // (op 1073) 
v_add_u32 v21, v21, v13 // (op 1073) 


s_mul_i32 s1, s28, s23 // (op 1075) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1075) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1075) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1075) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1075) most significant: sum
s_add_u32 s1, s1, s7 // (op 1075) most significant: sum


s_mov_b32 s56, s30 // (op 1075) reg expression
s_mov_b32 s57, s31 // (op 1075) reg expression
s_mov_b32 s58, s0 // (op 1075) reg expression
s_mov_b32 s59, 131072 // (op 1075) call()

v_and_b32 v12, 63, v1 // (op 1137) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1137) 
v_lshrrev_b32 v13, 4, v12 // (op 1137) 

s_lshr_b32 s6, s0, 1 // (op 1137) 

v_lshrrev_b32 v12, 6, v1 // (op 1137) 
v_lshl_add_u32 v16, s6, 3, v13 // (op 1137) 


v_and_b32 v13, 1, v12 // (op 1137) 

v_and_b32 v12, 15, v1 // (op 1137) 
v_lshl_add_u32 v17, v13, 4, v12 // (op 1137) 


v_lshl_add_u32 v12, s17, 5, v17 // (op 1137) 

v_mul_lo_u32 v13, v12, s32 // (op 1137) 

v_add_u32 v12, v16, v13 // (op 1137) 


s_mul_i32 s1, s32, s9 // (op 1139) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1139) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1139) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1139) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1139) most significant: sum
s_add_u32 s1, s1, s7 // (op 1139) most significant: sum


s_mov_b32 s60, s34 // (op 1139) reg expression
s_mov_b32 s61, s35 // (op 1139) reg expression
s_mov_b32 s62, s0 // (op 1139) reg expression
s_mov_b32 s63, 131072 // (op 1139) call()

v_and_b32 v13, -8, v1 // (op 1230) 
v_sub_u32 v16, v1, v13 // (op 1230) 

v_lshlrev_b32 v13, 4, v16 // (op 1230) 

v_lshrrev_b32 v16, 3, v1 // (op 1230) 
v_lshl_add_u32 v17, v16, 7, v13 // (op 1230) 


s_nop 0
v_readfirstlane_b32 s0, v17 // (op 1230) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -8, v1 // (op 1287) 
v_sub_u32 v16, v1, v13 // (op 1287) 

v_lshlrev_b32 v13, 4, v16 // (op 1287) 

v_lshrrev_b32 v16, 3, v1 // (op 1287) 
v_lshl_add_u32 v17, v16, 7, v13 // (op 1287) 


s_nop 0
v_readfirstlane_b32 s1, v17 // (op 1287) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -4, v1 // (op 1339) 
v_sub_u32 v16, v1, v13 // (op 1339) 

v_and_b32 v13, 4294967292, v1 // (op 1339) 
v_add_u32 v17, v13, v16 // (op 1339) 


s_ashr_i64 s[6:7], s[36:37], 7 // (op 316) 
s_and_b32 s10, s6, 4294967294 // (op 316) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 315) 
s_mov_b32 s11, s6 // (op 315) convert

s_cmp_lt_u32 s10, s11 // (op 315) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_315 // (op 315) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_315 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_315:
s_barrier  // (op 1549) 
s_add_u32 s10, s10, 1 // (op 317) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 893) 
v_mov_b64 v[22:23], s[6:7] // (op 893) convert


v_mov_b64 v[24:25], v[18:19] // (op 893) 
v_mov_b64 v[26:27], v[24:25] // (op 893) 
s_add_u32 s11, 4096, s0 // (op 893) 
s_mov_b32 m0, s11 // (op 893) reg expression
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 893) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 893) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 893) most significant half

s_add_u32 m0, m0, 4096 // (op 893) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 893) Load value direct to lds




v_add_co_u32 v18, vcc, 128, v18 // (op 931) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 931) most significant half
s_barrier  // (op 1565) 


v_mov_b64 v[22:23], v[14:15] // (op 899) 

s_mov_b32 m0, s1 // (op 899) reg expression
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 899) Load value direct to lds

v_add_co_u32 v14, vcc, 128, v14 // (op 1001) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1001) most significant half


v_mov_b32 v13, v21 // (op 180) 

buffer_load_ubyte v16, v13, s[56:59], 0 offen offset: 0 // (op 180) Load value

v_add_u32 v21, 4, v21 // (op 1065) 
s_barrier  // (op 1581) 
s_waitcnt vmcnt(2)
s_barrier  // (op 1555) 






ds_read_b128 v[22:25], v5 offset:4096 // (op 169) Load local data 
ds_read_b128 v[26:29], v5 offset:4160 // (op 169) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v17, v16 offset:24576 // (op 178) Store local data 

s_barrier  // (op 1571) 



ds_read_b128 v[30:33], v3 // (op 171) Load local data 
ds_read_b128 v[34:37], v3 offset:64 // (op 171) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1588) 



ds_read_u8 v13, v0 offset:24576 // (op 173) Load local data 


v_mov_b32 v16, v12 // (op 175) 

buffer_load_ubyte v20, v16, s[60:63], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v12, 4, v12 // (op 1129) 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:37], a[0:3], v13, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 176) 
ds_read_b128 v[38:41], v5 offset:8192 // (op 321) Load local data 


ds_read_b128 v[42:45], v5 offset:8256 // (op 321) Load local data 



ds_read_u8 v13, v0 offset:24704 // (op 325) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:45], v[30:37], a[4:7], v13, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 327) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 315) 
s_mov_b32 s11, s6 // (op 315) convert

s_cmp_lt_u32 s10, s11 // (op 315) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_315 // (op 315) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_315 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_315:






















s_barrier  // (op 1646) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 




buffer_store_dwordx4 v[16:19], v9, s[44:47], 0 offen // (op 52) Store value

v_accvgpr_read v12, a4 // (op 513) 
v_accvgpr_read v13, a5 // (op 513) 
v_accvgpr_read v14, a6 // (op 513) 
v_accvgpr_read v15, a7 // (op 513) 
v_mul_f32 v16, v11, v12 // (op 513) 
v_mul_f32 v17, v11, v13 // (op 513) 
v_mul_f32 v18, v11, v14 // (op 513) 
v_mul_f32 v19, v11, v15 // (op 513) 




buffer_store_dwordx4 v[16:19], v9, s[44:47], 0 offen offset:128 // (op 515) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[12:15], v6, s[40:43], 0 offen offset:128 // (op 543) Load value
v_accvgpr_read v16, a4 // (op 540) 
v_accvgpr_read v17, a5 // (op 540) 
v_accvgpr_read v18, a6 // (op 540) 
v_accvgpr_read v19, a7 // (op 540) 
v_mul_f32 v20, v11, v16 // (op 540) 
v_mul_f32 v21, v11, v17 // (op 540) 
v_mul_f32 v22, v11, v18 // (op 540) 
v_mul_f32 v23, v11, v19 // (op 540) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 544) 
v_mul_f32 v17, v10, v13 // (op 544) 
v_mul_f32 v18, v10, v14 // (op 544) 
v_mul_f32 v19, v10, v15 // (op 544) 

v_add_f32 v12, v16, v20 // (op 541) 
v_add_f32 v13, v17, v21 // (op 541) 
v_add_f32 v14, v18, v22 // (op 541) 
v_add_f32 v15, v19, v23 // (op 541) 





buffer_store_dwordx4 v[12:15], v9, s[44:47], 0 offen offset:128 // (op 542) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v6, s[40:43], 0 offen // (op 135) Load value
v_accvgpr_read v16, a0 // (op 132) 
v_accvgpr_read v17, a1 // (op 132) 
v_accvgpr_read v18, a2 // (op 132) 
v_accvgpr_read v19, a3 // (op 132) 
v_mul_f32 v20, v11, v16 // (op 132) 
v_mul_f32 v21, v11, v17 // (op 132) 
v_mul_f32 v22, v11, v18 // (op 132) 
v_mul_f32 v23, v11, v19 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 136) 
v_mul_f32 v17, v10, v13 // (op 136) 
v_mul_f32 v18, v10, v14 // (op 136) 
v_mul_f32 v19, v10, v15 // (op 136) 

v_add_f32 v12, v16, v20 // (op 133) 
v_add_f32 v13, v17, v21 // (op 133) 
v_add_f32 v14, v18, v22 // (op 133) 
v_add_f32 v15, v19, v23 // (op 133) 





buffer_store_dwordx4 v[12:15], v9, s[44:47], 0 offen // (op 134) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
.LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2, .LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_end-RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
  .amdhsa_next_free_vgpr 68
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25088 // lds bytes
  .amdhsa_accum_offset 60
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
    macrotile: [64, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 25088
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 60
    .agpr_count: 8
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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

