// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.p2align 8
.type RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2,@function
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 1314) 
v_and_b32 v3, 63, v1 // (op 1314) 
v_sub_u32 v4, v3, v0 // (op 1314) 

v_lshrrev_b32 v3, 1, v4 // (op 1314) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 1314) 


v_and_b32 v0, 1, v4 // (op 1314) 

v_lshlrev_b32 v3, 3, v0 // (op 1314) 

v_add_u32 v0, v5, v3 // (op 1314) 


v_and_b32 v3, 48, v1 // (op 1473) 
v_and_b32 v4, 63, v1 // (op 1473) 
v_sub_u32 v5, v4, v3 // (op 1473) 

v_lshrrev_b32 v4, 1, v5 // (op 1473) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 1473) 


v_and_b32 v3, 1, v5 // (op 1473) 

v_lshlrev_b32 v4, 3, v3 // (op 1473) 

v_add_u32 v3, v6, v4 // (op 1473) 


v_lshrrev_b32 v4, 6, v1 // (op 1201) 
v_and_b32 v5, 63, v1 // (op 1201) 
v_lshrrev_b32 v6, 4, v5 // (op 1201) 
v_lshlrev_b32 v7, 2, v6 // (op 1201) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1201) 


v_lshl_add_u32 v4, s16, 6, v6 // (op 1201) 

v_lshrrev_b32 v6, 2, v5 // (op 1201) 

v_and_b32 v5, 3, v6 // (op 1201) 

v_and_b32 v6, 3, v1 // (op 1201) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1201) 


v_lshl_add_u32 v5, s17, 4, v7 // (op 1201) 

v_mul_lo_u32 v6, v5, s2 // (op 1201) 

v_add_lshl_u32 v5, v4, v6, 2 // (op 1201) 


v_and_b32 v4, 48, v1 // (op 1262) 
v_and_b32 v6, 63, v1 // (op 1262) 
v_sub_u32 v7, v6, v4 // (op 1262) 

v_lshrrev_b32 v6, 1, v7 // (op 1262) 
s_mul_i32 s1, s2, s9 // (op 1203) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1203) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1203) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1203) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1203) most significant: sum
s_add_u32 s1, s1, s19 // (op 1203) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1203) 

v_add_lshl_u32 v8, v4, v6, 6 // (op 1262) 


v_and_b32 v4, 1, v7 // (op 1262) 

v_lshlrev_b32 v6, 3, v4 // (op 1262) 

v_lshrrev_b32 v4, 6, v1 // (op 1262) 
v_lshl_add_u32 v7, v4, 4, v6 // (op 1262) 


v_add_u32 v4, v8, v7 // (op 1262) 


v_lshrrev_b32 v6, 6, v1 // (op 1372) 
v_and_b32 v7, 15, v1 // (op 1372) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 1372) 


v_and_b32 v6, 63, v1 // (op 1372) 
v_lshrrev_b32 v7, 4, v6 // (op 1372) 

v_lshl_add_u32 v6, v7, 6, v8 // (op 1372) 


v_lshrrev_b32 v7, 6, v1 // (op 1388) 
v_and_b32 v8, 63, v1 // (op 1388) 
v_lshrrev_b32 v9, 4, v8 // (op 1388) 
v_lshlrev_b32 v10, 2, v9 // (op 1388) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1388) 


v_lshl_add_u32 v7, s16, 6, v9 // (op 1388) 

v_lshrrev_b32 v9, 2, v8 // (op 1388) 

v_and_b32 v8, 3, v9 // (op 1388) 

v_and_b32 v9, 3, v1 // (op 1388) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1388) 


v_lshl_add_u32 v8, s17, 4, v10 // (op 1388) 

v_mul_lo_u32 v9, v8, s4 // (op 1388) 

v_add_lshl_u32 v8, v7, v9, 2 // (op 1388) 


v_and_b32 v7, 48, v1 // (op 1434) 
v_and_b32 v9, 63, v1 // (op 1434) 
v_sub_u32 v10, v9, v7 // (op 1434) 

v_lshrrev_b32 v9, 1, v10 // (op 1434) 
s_mul_i32 s1, s4, s9 // (op 1390) most significant: low of low * high
v_add_lshl_u32 v11, v7, v9, 6 // (op 1434) 


v_and_b32 v7, 1, v10 // (op 1434) 

v_lshlrev_b32 v9, 3, v7 // (op 1434) 

v_lshrrev_b32 v7, 6, v1 // (op 1434) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 1434) 


v_add_u32 v7, v11, v10 // (op 1434) 


v_lshrrev_b32 v9, 6, v1 // (op 1502) 
v_and_b32 v10, 15, v1 // (op 1502) 
v_lshl_add_u32 v11, v9, 4, v10 // (op 1502) 


v_and_b32 v9, 63, v1 // (op 1502) 
v_lshrrev_b32 v10, 4, v9 // (op 1502) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 1502) 


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
v_and_b32 v12, -4, v1 // (op 968) 
v_sub_u32 v13, v1, v12 // (op 968) 

v_lshlrev_b32 v12, 4, v13 // (op 968) 

v_lshl_add_u32 v13, s16, 6, v12 // (op 968) 

v_mov_b32 v15, 0 // (op 968) convert
v_mov_b32 v14, v13 // (op 968) convert

v_lshrrev_b32 v12, 2, v1 // (op 968) 
v_mul_lo_u32 v17, v12, s13 // (op 968) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 968) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 968) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 968) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 968) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 968) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 970) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 970) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 970) least significant
s_mul_hi_u32 s7, s12, s22 // (op 970) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 970) most significant: sum
s_add_u32 s1, s1, s7 // (op 970) most significant: sum


s_mov_b32 s8, s20 // (op 970) reg expression
s_mov_b32 s9, s21 // (op 970) reg expression
s_mov_b32 s10, s0 // (op 970) reg expression
s_mov_b32 s11, 131072 // (op 970) call()

v_and_b32 v14, -4, v1 // (op 987) 
v_sub_u32 v15, v1, v14 // (op 987) 

v_lshlrev_b32 v14, 4, v15 // (op 987) 

v_lshl_add_u32 v15, s16, 6, v14 // (op 987) 

v_mov_b32 v17, 0 // (op 987) convert
v_mov_b32 v16, v15 // (op 987) convert

v_lshrrev_b32 v14, 2, v1 // (op 987) 
v_mul_lo_u32 v19, v14, s13 // (op 987) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 987) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 987) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 987) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 987) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 987) most significant half



s_mul_i32 s1, s12, s23 // (op 989) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 989) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 989) least significant
s_mul_hi_u32 s7, s12, s22 // (op 989) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 989) most significant: sum
s_add_u32 s1, s1, s7 // (op 989) most significant: sum


s_mov_b32 s8, s20 // (op 989) reg expression
s_mov_b32 s9, s21 // (op 989) reg expression
s_mov_b32 s10, s0 // (op 989) reg expression
s_mov_b32 s11, 131072 // (op 989) call()

s_lshl_b32 s0, s17, 4 // (op 1032) 
s_mov_b32 s7, 0 // (op 1032) convert
s_mov_b32 s6, s0 // (op 1032) convert

v_mul_lo_u32 v17, v1, s25 // (op 1032) most significant: low of low * high
v_mul_hi_u32 v18, v1, s24 // (op 1032) most significant: high of low * low
v_mul_lo_u32 v16, v1, s24 // (op 1032) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1032) most significant: sum

v_mov_b32 v18, s6 // (op 1032) 
v_mov_b32 v19, s7 // (op 1032) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 1032) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 1032) most significant half




s_mul_i32 s1, s24, s23 // (op 1034) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 1034) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 1034) least significant
s_mul_hi_u32 s7, s24, s22 // (op 1034) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1034) most significant: sum
s_add_u32 s1, s1, s7 // (op 1034) most significant: sum


s_mov_b32 s48, s26 // (op 1034) reg expression
s_mov_b32 s49, s27 // (op 1034) reg expression
s_mov_b32 s50, s0 // (op 1034) reg expression
s_mov_b32 s51, 131072 // (op 1034) call()

s_lshl_b32 s0, s17, 4 // (op 1051) 
s_mov_b32 s7, 0 // (op 1051) convert
s_mov_b32 s6, s0 // (op 1051) convert

v_mul_lo_u32 v19, v1, s25 // (op 1051) most significant: low of low * high
v_mul_hi_u32 v20, v1, s24 // (op 1051) most significant: high of low * low
v_mul_lo_u32 v18, v1, s24 // (op 1051) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 1051) most significant: sum

v_mov_b32 v20, s6 // (op 1051) 
v_mov_b32 v21, s7 // (op 1051) 
v_add_co_u32 v18, s[0:1], v18, v20 // (op 1051) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 1051) most significant half




s_mul_i32 s1, s24, s23 // (op 1053) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 1053) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 1053) least significant
s_mul_hi_u32 s7, s24, s22 // (op 1053) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1053) most significant: sum
s_add_u32 s1, s1, s7 // (op 1053) most significant: sum


s_mov_b32 s48, s26 // (op 1053) reg expression
s_mov_b32 s49, s27 // (op 1053) reg expression
s_mov_b32 s50, s0 // (op 1053) reg expression
s_mov_b32 s51, 131072 // (op 1053) call()

v_and_b32 v20, -64, v1 // (op 1096) 
v_sub_u32 v21, v1, v20 // (op 1096) 

v_lshl_add_u32 v20, s16, 6, v21 // (op 1096) 

v_lshrrev_b32 v21, 6, v1 // (op 1096) 
v_lshlrev_b32 v22, 1, v21 // (op 1096) 

v_mul_lo_u32 v21, v22, s28 // (op 1096) 

v_add_u32 v22, v20, v21 // (op 1096) 


s_mul_i32 s1, s28, s33 // (op 1098) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1098) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1098) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1098) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1098) most significant: sum
s_add_u32 s1, s1, s7 // (op 1098) most significant: sum


s_mov_b32 s52, s30 // (op 1098) reg expression
s_mov_b32 s53, s31 // (op 1098) reg expression
s_mov_b32 s54, s0 // (op 1098) reg expression
s_mov_b32 s55, 131072 // (op 1098) call()

v_and_b32 v20, -64, v1 // (op 1115) 
v_sub_u32 v21, v1, v20 // (op 1115) 

v_lshl_add_u32 v20, s16, 6, v21 // (op 1115) 

v_lshrrev_b32 v21, 6, v1 // (op 1115) 
v_lshlrev_b32 v23, 1, v21 // (op 1115) 

v_mul_lo_u32 v21, v23, s28 // (op 1115) 

v_add_u32 v23, v20, v21 // (op 1115) 


s_mul_i32 s1, s28, s33 // (op 1117) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1117) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1117) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1117) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1117) most significant: sum
s_add_u32 s1, s1, s7 // (op 1117) most significant: sum


s_mov_b32 s52, s30 // (op 1117) reg expression
s_mov_b32 s53, s31 // (op 1117) reg expression
s_mov_b32 s54, s0 // (op 1117) reg expression
s_mov_b32 s55, 131072 // (op 1117) call()

v_and_b32 v20, 15, v1 // (op 1160) 
v_lshl_add_u32 v21, s17, 4, v20 // (op 1160) 

v_and_b32 v20, 63, v1 // (op 1160) 
v_lshrrev_b32 v24, 4, v20 // (op 1160) 

v_mul_lo_u32 v20, v24, s34 // (op 1160) 
v_add_u32 v20, v20, v21 // (op 1160) 


s_mul_i32 s1, s34, s33 // (op 1162) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1162) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1162) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1162) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1162) most significant: sum
s_add_u32 s1, s1, s7 // (op 1162) most significant: sum


s_mov_b32 s56, s36 // (op 1162) reg expression
s_mov_b32 s57, s37 // (op 1162) reg expression
s_mov_b32 s58, s0 // (op 1162) reg expression
s_mov_b32 s59, 131072 // (op 1162) call()

v_and_b32 v21, 15, v1 // (op 1179) 
v_lshl_add_u32 v24, s17, 4, v21 // (op 1179) 

v_and_b32 v21, 63, v1 // (op 1179) 
v_lshrrev_b32 v25, 4, v21 // (op 1179) 

v_mul_lo_u32 v21, v25, s34 // (op 1179) 
v_add_u32 v21, v21, v24 // (op 1179) 


s_mul_i32 s1, s34, s33 // (op 1181) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1181) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1181) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1181) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1181) most significant: sum
s_add_u32 s1, s1, s7 // (op 1181) most significant: sum


s_mov_b32 s56, s36 // (op 1181) reg expression
s_mov_b32 s57, s37 // (op 1181) reg expression
s_mov_b32 s58, s0 // (op 1181) reg expression
s_mov_b32 s59, 131072 // (op 1181) call()


v_mov_b64 v[24:25], s[34:35] // (op 774) convert

v_mov_b32 v26, v20 // (op 774) reg expression
v_mov_b32 v27, v26 // (op 774) 
buffer_load_ubyte v28, v27, s[56:59], 0 offen offset: 0 // (op 774) Load value



v_lshlrev_b32 v24, 4, v1 // (op 1302) 
s_nop 0
v_readfirstlane_b32 s0, v24 // (op 1302) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[24:25], v[16:17] // (op 916) reg expression

s_nop 0
s_mov_b32 m0, s0 // (op 916) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 916) Load value direct to lds


v_and_b32 v24, -4, v1 // (op 1245) 
v_sub_u32 v25, v1, v24 // (op 1245) 

v_lshlrev_b32 v24, 4, v25 // (op 1245) 

v_lshrrev_b32 v25, 2, v1 // (op 1245) 
v_lshl_add_u32 v26, v25, 6, v24 // (op 1245) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1245) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 921) 
v_mov_b64 v[24:25], s[6:7] // (op 921) convert


v_mov_b64 v[26:27], v[12:13] // (op 921) reg expression
s_add_u32 s1, 2048, s0 // (op 921) 
s_mov_b32 m0, s1 // (op 921) reg expression
buffer_load_dwordx4 v26, s[8:11], 0 offen offset: 0 lds // (op 921) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 921) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 921) most significant half

s_add_u32 m0, m0, 2048 // (op 921) 
buffer_load_dwordx4 v26, s[8:11], 0 offen offset: 0 lds // (op 921) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 921) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 921) most significant half

s_add_u32 m0, m0, 2048 // (op 921) 
buffer_load_dwordx4 v26, s[8:11], 0 offen offset: 0 lds // (op 921) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 921) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 921) most significant half

s_add_u32 m0, m0, 2048 // (op 921) 
buffer_load_dwordx4 v26, s[8:11], 0 offen offset: 0 lds // (op 921) Load value direct to lds





v_mov_b64 v[24:25], s[28:29] // (op 784) convert

v_mov_b32 v26, v22 // (op 784) reg expression
buffer_load_ubyte v30, v26, s[52:55], 0 offen offset: 0 // (op 784) Load value
v_add_u32 v26, v26, v24 // (op 784) 
buffer_load_ubyte v31, v26, s[52:55], 0 offen offset: 0 // (op 784) Load value



v_mov_b64 v[24:25], s[34:35] // (op 788) convert

s_lshl_b32 s0, s34, 2 // (op 788) 
v_add_u32 v29, v20, s0 // (op 788) 

v_mov_b32 v26, v29 // (op 788) 
buffer_load_ubyte v27, v26, s[56:59], 0 offen offset: 0 // (op 788) Load value




v_lshlrev_b32 v20, 4, v1 // (op 1461) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1461) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[6:7], s[24:25], 7 // (op 926) 
v_mov_b32 v20, s6 // (op 926) 
v_mov_b32 v29, s7 // (op 926) 
v_add_co_u32 v24, s[14:15], v16, v20 // (op 926) least significant half
v_addc_co_u32 v25, s[14:15], v17, v29, s[14:15] // (op 926) most significant half




s_add_u32 s1, 10240, s0 // (op 926) 

s_mov_b32 m0, s1 // (op 926) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 926) Load value direct to lds




v_and_b32 v20, -4, v1 // (op 1417) 
v_sub_u32 v29, v1, v20 // (op 1417) 

v_lshlrev_b32 v20, 4, v29 // (op 1417) 

v_lshrrev_b32 v29, 2, v1 // (op 1417) 
v_lshl_add_u32 v16, v29, 6, v20 // (op 1417) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 1417) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 928) 
v_mov_b64 v[16:17], s[6:7] // (op 928) convert


s_lshl_b64 s[6:7], s[12:13], 7 // (op 928) 
v_mov_b32 v20, s6 // (op 928) 
v_mov_b32 v29, s7 // (op 928) 
v_add_co_u32 v24, s[14:15], v12, v20 // (op 928) least significant half
v_addc_co_u32 v25, s[14:15], v13, v29, s[14:15] // (op 928) most significant half




s_add_u32 s1, 12288, s0 // (op 928) 
s_mov_b32 m0, s1 // (op 928) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 928) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 928) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 928) most significant half

s_add_u32 m0, m0, 2048 // (op 928) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 928) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 928) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 928) most significant half

s_add_u32 m0, m0, 2048 // (op 928) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 928) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 928) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 928) most significant half

s_add_u32 m0, m0, 2048 // (op 928) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 928) Load value direct to lds






v_mov_b64 v[12:13], s[28:29] // (op 798) convert

s_lshl_b32 s0, s28, 2 // (op 798) 
v_add_u32 v20, v22, s0 // (op 798) 

buffer_load_ubyte v16, v20, s[52:55], 0 offen offset: 0 // (op 798) Load value
v_add_u32 v20, v20, v12 // (op 798) 
buffer_load_ubyte v17, v20, s[52:55], 0 offen offset: 0 // (op 798) Load value



v_and_b32 v20, -64, v1 // (op 1355) 
v_sub_u32 v22, v1, v20 // (op 1355) 

v_lshrrev_b32 v20, 6, v1 // (op 1355) 
v_lshl_add_u32 v29, v20, 7, v22 // (op 1355) 





s_waitcnt vmcnt(9)
ds_write_b8 v29, v30 offset:20480 // (op 802) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v29, v31 offset:20544 // (op 802) Store local data 

s_barrier  // (op 1559) 



ds_read_b64_tr_b8 v[32:33], v0 // (op 825) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:128 // (op 825) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:1024 // (op 825) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:1152 // (op 825) Transpose load local data 
s_barrier  // (op 1543) 



ds_read_b64_tr_b8 v[40:41], v4 offset:2048 // (op 816) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:2560 // (op 816) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:6144 // (op 816) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:6656 // (op 816) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1575) 


ds_read_u8 v20, v6 offset:20480 // (op 829) Load local data 



ds_read_b64_tr_b8 v[48:49], v4 offset:2080 // (op 821) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v4 offset:2592 // (op 821) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:6176 // (op 821) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:6688 // (op 821) Transpose load local data 


ds_read_u8 v22, v6 offset:20512 // (op 834) Load local data 
v_and_b32 v29, -4, v1 // (op 1217) 
v_sub_u32 v12, v1, v29 // (op 1217) 

v_lshlrev_b32 v13, 4, v12 // (op 1217) 

v_lshrrev_b32 v12, 2, v1 // (op 1217) 
v_lshl_add_u32 v29, v12, 6, v13 // (op 1217) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 1217) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 4, v1 // (op 1274) 
s_nop 0
v_readfirstlane_b32 s1, v29 // (op 1274) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -64, v1 // (op 1326) 
v_sub_u32 v12, v1, v29 // (op 1326) 

v_lshrrev_b32 v13, 6, v1 // (op 1326) 
v_lshl_add_u32 v29, v13, 7, v12 // (op 1326) 


v_and_b32 v12, -4, v1 // (op 1402) 
v_sub_u32 v13, v1, v12 // (op 1402) 

v_lshlrev_b32 v12, 4, v13 // (op 1402) 

v_lshrrev_b32 v13, 2, v1 // (op 1402) 
v_lshl_add_u32 v24, v13, 6, v12 // (op 1402) 


s_nop 0
v_readfirstlane_b32 s6, v24 // (op 1402) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v12, 4, v1 // (op 1446) 
s_nop 0
v_readfirstlane_b32 s7, v12 // (op 1446) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1485) 
v_sub_u32 v13, v1, v12 // (op 1485) 

v_lshrrev_b32 v12, 6, v1 // (op 1485) 
v_lshl_add_u32 v24, v12, 7, v13 // (op 1485) 


s_mov_b32 s14, 0 // (op 311) call()
s_ashr_i64 s[18:19], s[22:23], 7 // (op 310) 
s_mov_b32 s15, s18 // (op 310) convert

s_and_b32 s18, s15, 4294967294 // (op 310) 

s_cmp_lt_u32 s14, s18 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_310 // (op 310) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_310 if false)
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_310:
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[32:39], a[0:3], v20, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 82) 
s_add_u32 s14, s14, 2 // (op 312) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[32:39], a[4:7], v22, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 346) 

v_mov_b64 v[12:13], s[34:35] // (op 80) convert

v_mov_b32 v25, v21 // (op 80) 
s_lshl_b32 s15, s34, 3 // (op 80) 
v_add_u32 v26, v25, s15 // (op 80) 

v_mov_b32 v56, v26 // (op 80) 
buffer_load_ubyte v28, v56, s[56:59], 0 offen offset: 0 // (op 80) Load value



s_barrier  // (op 1562) 


v_mov_b64 v[12:13], v[18:19] // (op 888) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 888) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 888) 

v_mov_b32 v26, s38 // (op 888) 
v_mov_b32 v56, s39 // (op 888) 
v_add_co_u32 v58, s[18:19], v12, v26 // (op 888) least significant half
v_addc_co_u32 v59, s[18:19], v13, v56, s[18:19] // (op 888) most significant half





s_mov_b32 m0, s1 // (op 888) reg expression
buffer_load_dwordx4 v58, s[48:51], 0 offen offset: 0 lds // (op 888) Load value direct to lds

s_barrier  // (op 1546) 

s_lshl_b64 s[18:19], s[12:13], 5 // (op 883) 
v_mov_b64 v[56:57], s[18:19] // (op 883) convert


v_mov_b64 v[58:59], v[14:15] // (op 883) 
s_lshl_b64 s[18:19], s[12:13], 7 // (op 883) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 883) 

v_mov_b32 v26, s38 // (op 883) 
v_mov_b32 v60, s39 // (op 883) 
v_add_co_u32 v62, s[18:19], v58, v26 // (op 883) least significant half
v_addc_co_u32 v63, s[18:19], v59, v60, s[18:19] // (op 883) most significant half




s_add_u32 s15, 2048, s0 // (op 883) 
s_mov_b32 m0, s15 // (op 883) reg expression
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 883) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v56 // (op 883) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 883) most significant half

s_add_u32 m0, m0, 2048 // (op 883) 
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 883) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v56 // (op 883) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 883) most significant half

s_add_u32 m0, m0, 2048 // (op 883) 
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 883) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v56 // (op 883) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 883) most significant half

s_add_u32 m0, m0, 2048 // (op 883) 
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 883) Load value direct to lds




v_mov_b64 v[56:57], s[28:29] // (op 17) convert

v_mov_b32 v26, v23 // (op 17) 
s_lshl_b32 s15, s28, 3 // (op 17) 
v_add_u32 v60, v26, s15 // (op 17) 

buffer_load_ubyte v30, v60, s[52:55], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v60, v60, v56 // (op 17) 
buffer_load_ubyte v31, v60, s[52:55], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1607) 



s_waitcnt vmcnt(9)
ds_write_b8 v24, v16 offset:20736 // (op 236) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v24, v17 offset:20800 // (op 236) Store local data 
s_barrier  // (op 1598) 



ds_read_b64_tr_b8 v[32:33], v3 offset:10240 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:10368 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v3 offset:11264 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v3 offset:11392 // (op 229) Transpose load local data 
s_barrier  // (op 1592) 



ds_read_b64_tr_b8 v[40:41], v7 offset:12288 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:12800 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:16384 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:16896 // (op 227) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1604) 


ds_read_u8 v20, v9 offset:20736 // (op 231) Load local data 



ds_read_b64_tr_b8 v[48:49], v7 offset:12320 // (op 353) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v7 offset:12832 // (op 353) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v7 offset:16416 // (op 353) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v7 offset:16928 // (op 353) Transpose load local data 


ds_read_u8 v22, v9 offset:20768 // (op 357) Load local data 
s_barrier  // (op 1658) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[32:39], a[0:3], v20, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 234) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[32:39], a[4:7], v22, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 347) 

v_mov_b64 v[56:57], s[34:35] // (op 233) convert

s_lshl_b32 s15, s34, 2 // (op 233) 
s_mul_i32 s18, 3, s15 // (op 233) 

v_add_u32 v60, v25, s18 // (op 233) 

v_mov_b32 v61, v60 // (op 233) 
buffer_load_ubyte v27, v61, s[56:59], 0 offen offset: 0 // (op 233) Load value




s_barrier  // (op 1601) 
s_lshl_b32 s15, s34, 3 // (op 1171) 


v_add_u32 v21, v21, s15 // (op 1171) 

s_lshl_b64 s[18:19], s[24:25], 7 // (op 911) 
s_mul_i32 s39, 3, s19 // (op 911) most significant: low of low * high
s_mul_i32 s38, 3, s18 // (op 911) least significant
s_mul_hi_u32 s15, 3, s18 // (op 911) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 911) most significant: sum


v_mov_b32 v25, s38 // (op 911) 
v_mov_b32 v56, s39 // (op 911) 
v_add_co_u32 v60, s[18:19], v12, v25 // (op 911) least significant half
v_addc_co_u32 v61, s[18:19], v13, v56, s[18:19] // (op 911) most significant half




s_add_u32 s15, 10240, s7 // (op 911) 

s_mov_b32 m0, s15 // (op 911) reg expression
buffer_load_dwordx4 v60, s[48:51], 0 offen offset: 0 lds // (op 911) Load value direct to lds



s_barrier  // (op 1595) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 1043) 

s_lshl_b64 s[38:39], s[18:19], 1 // (op 1043) 

v_mov_b32 v25, s38 // (op 1043) 
v_mov_b32 v12, s39 // (op 1043) 
s_lshl_b64 s[18:19], s[12:13], 5 // (op 906) 
v_mov_b64 v[56:57], s[18:19] // (op 906) convert


s_lshl_b64 s[18:19], s[12:13], 7 // (op 906) 
s_mul_i32 s61, 3, s19 // (op 906) most significant: low of low * high
s_mul_i32 s60, 3, s18 // (op 906) least significant
s_mul_hi_u32 s15, 3, s18 // (op 906) most significant: high of low * low
s_add_u32 s61, s61, s15 // (op 906) most significant: sum


v_mov_b32 v13, s60 // (op 906) 
v_mov_b32 v60, s61 // (op 906) 
v_add_co_u32 v18, s[18:19], v18, v25 // (op 1043) least significant half
v_addc_co_u32 v19, s[18:19], v19, v12, s[18:19] // (op 1043) most significant half




v_add_co_u32 v62, s[18:19], v58, v13 // (op 906) least significant half
v_addc_co_u32 v63, s[18:19], v59, v60, s[18:19] // (op 906) most significant half




s_add_u32 s15, 12288, s6 // (op 906) 
s_mov_b32 m0, s15 // (op 906) reg expression
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 906) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v56 // (op 906) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 906) most significant half

s_add_u32 m0, m0, 2048 // (op 906) 
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 906) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v56 // (op 906) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 906) most significant half

s_add_u32 m0, m0, 2048 // (op 906) 
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 906) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v56 // (op 906) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 906) most significant half

s_add_u32 m0, m0, 2048 // (op 906) 
buffer_load_dwordx4 v62, s[8:11], 0 offen offset: 0 lds // (op 906) Load value direct to lds





v_mov_b64 v[12:13], s[28:29] // (op 238) convert

s_lshl_b32 s15, s28, 2 // (op 238) 
s_mul_i32 s18, 3, s15 // (op 238) 

v_add_u32 v25, v26, s18 // (op 238) 

s_lshl_b64 s[18:19], s[12:13], 7 // (op 979) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 979) 

v_mov_b32 v56, s38 // (op 979) 
v_mov_b32 v57, s39 // (op 979) 
v_add_co_u32 v14, s[18:19], v14, v56 // (op 979) least significant half
v_addc_co_u32 v15, s[18:19], v15, v57, s[18:19] // (op 979) most significant half




buffer_load_ubyte v16, v25, s[52:55], 0 offen offset: 0 // (op 238) Load value
v_add_u32 v25, v25, v12 // (op 238) 
buffer_load_ubyte v17, v25, s[52:55], 0 offen offset: 0 // (op 238) Load value



s_lshl_b32 s15, s28, 3 // (op 1107) 
s_barrier  // (op 1578) 
v_add_u32 v23, v23, s15 // (op 1107) 




s_waitcnt vmcnt(9)
ds_write_b8 v29, v30 offset:20480 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v29, v31 offset:20544 // (op 21) Store local data 
s_barrier  // (op 1568) 



ds_read_b64_tr_b8 v[32:33], v0 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:128 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:1024 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:1152 // (op 70) Transpose load local data 
s_barrier  // (op 1552) 



ds_read_b64_tr_b8 v[40:41], v4 offset:2048 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:2560 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:6144 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:6656 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1585) 


ds_read_u8 v20, v6 offset:20480 // (op 75) Load local data 



ds_read_b64_tr_b8 v[48:49], v4 offset:2080 // (op 340) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v4 offset:2592 // (op 340) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:6176 // (op 340) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:6688 // (op 340) Transpose load local data 


ds_read_u8 v22, v6 offset:20512 // (op 344) Load local data 
s_ashr_i64 s[18:19], s[22:23], 7 // (op 310) 
s_mov_b32 s15, s18 // (op 310) convert

s_and_b32 s18, s15, 4294967294 // (op 310) 

s_cmp_lt_u32 s14, s18 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_310 // (op 310) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_310 if true)
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_310:













s_barrier  // (op 1642) 
v_and_b32 v21, -4, v1 // (op 941) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 941) 
v_sub_u32 v12, v1, v21 // (op 941) 

s_lshr_b32 s6, s0, 1 // (op 941) 

v_lshlrev_b32 v21, 4, v12 // (op 941) 

v_lshl_add_u32 v12, s16, 6, v21 // (op 941) 

v_mov_b32 v15, 0 // (op 941) convert
v_mov_b32 v14, v12 // (op 941) convert

v_lshrrev_b32 v21, 2, v1 // (op 941) 
v_lshl_add_u32 v12, s6, 8, v21 // (op 941) 


v_mul_lo_u32 v19, v12, s13 // (op 941) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 941) most significant: high of low * low
v_mul_lo_u32 v18, v12, s12 // (op 941) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 941) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v18 // (op 941) least significant half
v_addc_co_u32 v13, s[0:1], v15, v19, s[0:1] // (op 941) most significant half



s_mul_i32 s1, s12, s23 // (op 943) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 943) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 943) least significant
s_mul_hi_u32 s7, s12, s22 // (op 943) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 943) most significant: sum
s_add_u32 s1, s1, s7 // (op 943) most significant: sum


s_mov_b32 s8, s20 // (op 943) reg expression
s_mov_b32 s9, s21 // (op 943) reg expression
s_mov_b32 s10, s0 // (op 943) reg expression
s_mov_b32 s11, 131072 // (op 943) call()

s_ashr_i64 s[0:1], s[22:23], 7 // (op 1009) 
s_lshr_b32 s6, s0, 1 // (op 1009) 

v_lshl_add_u32 v21, s6, 8, v1 // (op 1009) 

s_lshl_b32 s0, s17, 4 // (op 1009) 
s_mov_b32 s7, 0 // (op 1009) convert
s_mov_b32 s6, s0 // (op 1009) convert

v_mul_lo_u32 v15, v21, s25 // (op 1009) most significant: low of low * high
v_mul_hi_u32 v18, v21, s24 // (op 1009) most significant: high of low * low
v_mul_lo_u32 v14, v21, s24 // (op 1009) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 1009) most significant: sum

v_mov_b32 v18, s6 // (op 1009) 
v_mov_b32 v19, s7 // (op 1009) 
v_add_co_u32 v14, s[0:1], v14, v18 // (op 1009) least significant half
v_addc_co_u32 v15, s[0:1], v15, v19, s[0:1] // (op 1009) most significant half





s_mul_i32 s1, s24, s23 // (op 1011) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 1011) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 1011) least significant
s_mul_hi_u32 s7, s24, s22 // (op 1011) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1011) most significant: sum
s_add_u32 s1, s1, s7 // (op 1011) most significant: sum


s_mov_b32 s48, s26 // (op 1011) reg expression
s_mov_b32 s49, s27 // (op 1011) reg expression
s_mov_b32 s50, s0 // (op 1011) reg expression
s_mov_b32 s51, 131072 // (op 1011) call()

v_and_b32 v21, -64, v1 // (op 1073) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1073) 
v_sub_u32 v18, v1, v21 // (op 1073) 

s_lshr_b32 s6, s0, 1 // (op 1073) 

v_lshl_add_u32 v19, s16, 6, v18 // (op 1073) 

v_lshrrev_b32 v18, 6, v1 // (op 1073) 
v_lshlrev_b32 v21, 1, v18 // (op 1073) 

v_lshl_add_u32 v18, s6, 3, v21 // (op 1073) 


v_mul_lo_u32 v21, v18, s28 // (op 1073) 

v_add_u32 v18, v19, v21 // (op 1073) 


s_mul_i32 s1, s28, s33 // (op 1075) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1075) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1075) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1075) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1075) most significant: sum
s_add_u32 s1, s1, s7 // (op 1075) most significant: sum


s_mov_b32 s52, s30 // (op 1075) reg expression
s_mov_b32 s53, s31 // (op 1075) reg expression
s_mov_b32 s54, s0 // (op 1075) reg expression
s_mov_b32 s55, 131072 // (op 1075) call()

v_and_b32 v19, 63, v1 // (op 1137) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1137) 
v_lshrrev_b32 v21, 4, v19 // (op 1137) 

s_lshr_b32 s6, s0, 1 // (op 1137) 

v_and_b32 v19, 15, v1 // (op 1137) 
v_lshl_add_u32 v23, s6, 3, v21 // (op 1137) 


v_lshl_add_u32 v21, s17, 4, v19 // (op 1137) 

v_mul_lo_u32 v19, v23, s34 // (op 1137) 
v_add_u32 v19, v19, v21 // (op 1137) 


s_mul_i32 s1, s34, s33 // (op 1139) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1139) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1139) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1139) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1139) most significant: sum
s_add_u32 s1, s1, s7 // (op 1139) most significant: sum


s_mov_b32 s56, s36 // (op 1139) reg expression
s_mov_b32 s57, s37 // (op 1139) reg expression
s_mov_b32 s58, s0 // (op 1139) reg expression
s_mov_b32 s59, 131072 // (op 1139) call()

v_and_b32 v21, -4, v1 // (op 1230) 
v_sub_u32 v23, v1, v21 // (op 1230) 

v_lshlrev_b32 v21, 4, v23 // (op 1230) 

v_lshrrev_b32 v23, 2, v1 // (op 1230) 
v_lshl_add_u32 v24, v23, 6, v21 // (op 1230) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 1230) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v21, 4, v1 // (op 1287) 
s_nop 0
v_readfirstlane_b32 s1, v21 // (op 1287) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -64, v1 // (op 1339) 
v_sub_u32 v23, v1, v21 // (op 1339) 

v_lshrrev_b32 v21, 6, v1 // (op 1339) 
v_lshl_add_u32 v24, v21, 7, v23 // (op 1339) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 316) 
s_and_b32 s14, s6, 4294967294 // (op 316) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 315) 
s_mov_b32 s15, s6 // (op 315) convert

s_cmp_lt_u32 s14, s15 // (op 315) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_315 // (op 315) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_315 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_315:
s_barrier  // (op 1549) 
s_add_u32 s14, s14, 1 // (op 317) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 893) 
v_mov_b64 v[16:17], s[6:7] // (op 893) convert


v_mov_b64 v[20:21], v[12:13] // (op 893) 
v_mov_b64 v[22:23], v[20:21] // (op 893) reg expression
s_add_u32 s15, 2048, s0 // (op 893) 
s_mov_b32 m0, s15 // (op 893) reg expression
buffer_load_dwordx4 v22, s[8:11], 0 offen offset: 0 lds // (op 893) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v16 // (op 893) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 893) most significant half

s_add_u32 m0, m0, 2048 // (op 893) 
buffer_load_dwordx4 v22, s[8:11], 0 offen offset: 0 lds // (op 893) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v16 // (op 893) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 893) most significant half

s_add_u32 m0, m0, 2048 // (op 893) 
buffer_load_dwordx4 v22, s[8:11], 0 offen offset: 0 lds // (op 893) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v16 // (op 893) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 893) most significant half

s_add_u32 m0, m0, 2048 // (op 893) 
buffer_load_dwordx4 v22, s[8:11], 0 offen offset: 0 lds // (op 893) Load value direct to lds




s_lshl_b64 s[6:7], s[12:13], 7 // (op 931) 
v_mov_b32 v16, s6 // (op 931) 
v_mov_b32 v17, s7 // (op 931) 
v_add_co_u32 v12, s[18:19], v12, v16 // (op 931) least significant half
v_addc_co_u32 v13, s[18:19], v13, v17, s[18:19] // (op 931) most significant half




s_barrier  // (op 1565) 


v_mov_b64 v[16:17], v[14:15] // (op 899) 
v_mov_b64 v[20:21], v[16:17] // (op 899) reg expression

s_mov_b32 m0, s1 // (op 899) reg expression
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 899) Load value direct to lds


s_lshl_b64 s[6:7], s[24:25], 7 // (op 1001) 
v_mov_b32 v16, s6 // (op 1001) 
v_mov_b32 v17, s7 // (op 1001) 
v_add_co_u32 v14, s[18:19], v14, v16 // (op 1001) least significant half
v_addc_co_u32 v15, s[18:19], v15, v17, s[18:19] // (op 1001) most significant half





v_mov_b64 v[16:17], s[28:29] // (op 180) convert

v_mov_b32 v20, v18 // (op 180) 
v_mov_b32 v21, v20 // (op 180) reg expression
buffer_load_ubyte v22, v21, s[52:55], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v21, v21, v16 // (op 180) 
buffer_load_ubyte v23, v21, s[52:55], 0 offen offset: 0 // (op 180) Load value



s_lshl_b32 s15, s28, 2 // (op 1065) 
v_add_u32 v18, v18, s15 // (op 1065) 

s_barrier  // (op 1581) 
s_waitcnt vmcnt(3)
s_barrier  // (op 1555) 






ds_read_b64_tr_b8 v[26:27], v4 offset:2048 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v4 offset:2560 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v4 offset:6144 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:6656 // (op 169) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v22 offset:20480 // (op 178) Store local data 
s_barrier  // (op 1571) 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v23 offset:20544 // (op 178) Store local data 




ds_read_b64_tr_b8 v[34:35], v0 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:128 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:1024 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:1152 // (op 171) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1588) 


ds_read_u8 v25, v6 offset:20480 // (op 173) Load local data 

v_mov_b64 v[16:17], s[34:35] // (op 175) convert

v_mov_b32 v20, v19 // (op 175) 
v_mov_b32 v21, v20 // (op 175) reg expression
v_mov_b32 v22, v21 // (op 175) 
buffer_load_ubyte v23, v22, s[56:59], 0 offen offset: 0 // (op 175) Load value




s_lshl_b32 s15, s34, 2 // (op 1129) 
v_add_u32 v19, v19, s15 // (op 1129) 




ds_read_b64_tr_b8 v[42:43], v4 offset:2080 // (op 321) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v25, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 176) 
ds_read_b64_tr_b8 v[44:45], v4 offset:2592 // (op 321) Transpose load local data 


ds_read_b64_tr_b8 v[46:47], v4 offset:6176 // (op 321) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v4 offset:6688 // (op 321) Transpose load local data 


ds_read_u8 v16, v6 offset:20512 // (op 325) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[34:41], a[4:7], v16, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 327) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 315) 
s_mov_b32 s15, s6 // (op 315) convert

s_cmp_lt_u32 s14, s15 // (op 315) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_315 // (op 315) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_315 if true)
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_315:






















s_barrier  // (op 1646) 






v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 




buffer_store_dwordx4 v[16:19], v8, s[44:47], 0 offen // (op 52) Store value

v_accvgpr_read v12, a4 // (op 513) 
v_accvgpr_read v13, a5 // (op 513) 
v_accvgpr_read v14, a6 // (op 513) 
v_accvgpr_read v15, a7 // (op 513) 
v_mul_f32 v16, v11, v12 // (op 513) 
v_mul_f32 v17, v11, v13 // (op 513) 
v_mul_f32 v18, v11, v14 // (op 513) 
v_mul_f32 v19, v11, v15 // (op 513) 




buffer_store_dwordx4 v[16:19], v8, s[44:47], 0 offen offset:128 // (op 515) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[12:15], v5, s[40:43], 0 offen offset:128 // (op 543) Load value
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





buffer_store_dwordx4 v[12:15], v8, s[44:47], 0 offen offset:128 // (op 542) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v5, s[40:43], 0 offen // (op 135) Load value
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





buffer_store_dwordx4 v[12:15], v8, s[44:47], 0 offen // (op 134) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.LRR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_end:
.size RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2, .LRR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_end-RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
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
    DataType: B8
    DataTypeA: B8
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
  - .name: RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
    .symbol: RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2.kd
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

