// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
.p2align 8
.type RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2,@function
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 1216) 
v_and_b32 v3, 48, v1 // (op 1216) 
v_sub_u32 v4, v0, v3 // (op 1216) 

v_lshrrev_b32 v3, 4, v0 // (op 1216) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1216) 


v_lshrrev_b32 v3, 7, v1 // (op 1216) 
v_lshlrev_b32 v4, 4, v3 // (op 1216) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 1216) 


v_mul_lo_u32 v0, v3, 6 // (op 1216) 
v_lshrrev_b32 v4, 3, v0 // (op 1216) 

v_lshrrev_b32 v0, 4, v3 // (op 1216) 

v_lshlrev_b32 v3, 2, v0 // (op 1216) 

v_add_u32 v0, v4, v3 // (op 1216) 


v_and_b32 v3, 63, v1 // (op 1373) 
v_and_b32 v4, 48, v1 // (op 1373) 
v_sub_u32 v5, v3, v4 // (op 1373) 

v_lshrrev_b32 v4, 4, v3 // (op 1373) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 1373) 


v_lshrrev_b32 v4, 7, v1 // (op 1373) 
v_lshlrev_b32 v5, 4, v4 // (op 1373) 

v_lshl_add_u32 v4, v3, 5, v5 // (op 1373) 


v_mul_lo_u32 v3, v4, 6 // (op 1373) 
v_lshrrev_b32 v5, 3, v3 // (op 1373) 

v_lshrrev_b32 v3, 4, v4 // (op 1373) 

v_lshlrev_b32 v4, 2, v3 // (op 1373) 

v_add_u32 v3, v5, v4 // (op 1373) 


v_lshrrev_b32 v4, 7, v1 // (op 1159) 
v_and_b32 v5, 63, v1 // (op 1159) 
v_lshrrev_b32 v6, 4, v5 // (op 1159) 
v_lshlrev_b32 v7, 2, v6 // (op 1159) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1159) 


v_lshl_add_u32 v4, s16, 5, v6 // (op 1159) 

v_lshrrev_b32 v6, 2, v5 // (op 1159) 

v_and_b32 v5, 3, v6 // (op 1159) 

v_and_b32 v6, 3, v1 // (op 1159) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1159) 


v_lshrrev_b32 v5, 6, v1 // (op 1159) 
v_and_b32 v6, 1, v5 // (op 1159) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1159) 


v_lshl_add_u32 v6, s17, 6, v5 // (op 1159) 

v_mul_lo_u32 v5, v6, s2 // (op 1159) 

v_add_lshl_u32 v6, v4, v5, 1 // (op 1159) 


v_lshrrev_b32 v4, 6, v1 // (op 1273) 
v_and_b32 v5, 1, v4 // (op 1273) 

v_and_b32 v4, 15, v1 // (op 1273) 
v_lshl_add_u32 v7, v5, 4, v4 // (op 1273) 


v_and_b32 v4, 48, v1 // (op 1273) 
v_lshl_add_u32 v5, v7, 7, v4 // (op 1273) 


v_lshrrev_b32 v4, 6, v1 // (op 1331) 
v_and_b32 v7, 1, v4 // (op 1331) 

v_and_b32 v4, 15, v1 // (op 1331) 
s_mul_i32 s1, s2, s9 // (op 1161) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1161) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1161) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1161) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1161) most significant: sum
s_add_u32 s1, s1, s19 // (op 1161) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1161) 

v_lshl_add_u32 v8, v7, 4, v4 // (op 1331) 


v_and_b32 v4, 63, v1 // (op 1331) 
v_lshrrev_b32 v7, 4, v4 // (op 1331) 

v_lshl_add_u32 v4, v8, 2, v7 // (op 1331) 


v_lshrrev_b32 v7, 7, v1 // (op 1347) 
v_and_b32 v8, 63, v1 // (op 1347) 
v_lshrrev_b32 v9, 4, v8 // (op 1347) 
v_lshlrev_b32 v10, 2, v9 // (op 1347) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1347) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 1347) 

v_lshrrev_b32 v9, 2, v8 // (op 1347) 

v_and_b32 v8, 3, v9 // (op 1347) 

v_and_b32 v9, 3, v1 // (op 1347) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1347) 


v_lshrrev_b32 v8, 6, v1 // (op 1347) 
v_and_b32 v9, 1, v8 // (op 1347) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 1347) 


v_lshl_add_u32 v9, s17, 6, v8 // (op 1347) 

v_mul_lo_u32 v8, v9, s4 // (op 1347) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 1347) 


v_lshrrev_b32 v7, 6, v1 // (op 1417) 
v_and_b32 v8, 1, v7 // (op 1417) 

v_and_b32 v7, 15, v1 // (op 1417) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 1417) 


v_and_b32 v7, 48, v1 // (op 1417) 
v_lshl_add_u32 v8, v10, 7, v7 // (op 1417) 


v_lshrrev_b32 v7, 6, v1 // (op 1446) 
v_and_b32 v10, 1, v7 // (op 1446) 

v_and_b32 v7, 15, v1 // (op 1446) 
s_mul_i32 s1, s4, s9 // (op 1349) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 1446) 


v_and_b32 v7, 63, v1 // (op 1446) 
v_lshrrev_b32 v10, 4, v7 // (op 1446) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 1446) 


s_mul_i32 s38, s5, s8 // (op 1349) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1349) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1349) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1349) most significant: sum
s_add_u32 s1, s1, s39 // (op 1349) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 1349) 

s_mov_b32 s40, s6 // (op 1161) reg expression
s_mov_b32 s41, s7 // (op 1161) reg expression
s_mov_b32 s42, s18 // (op 1161) reg expression
s_mov_b32 s43, 131072 // (op 1161) call()

s_mov_b32 s44, s10 // (op 1349) reg expression

s_mov_b32 s45, s11 // (op 1349) reg expression
s_mov_b32 s46, s38 // (op 1349) reg expression
s_mov_b32 s47, 131072 // (op 1349) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 310) call()
v_accvgpr_write a5, 0.00000 // (op 310) call()
v_accvgpr_write a6, 0.00000 // (op 310) call()
v_accvgpr_write a7, 0.00000 // (op 310) call()
v_and_b32 v12, -2, v1 // (op 926) 
v_sub_u32 v13, v1, v12 // (op 926) 

v_lshlrev_b32 v12, 4, v13 // (op 926) 

v_lshl_add_u32 v13, s16, 5, v12 // (op 926) 

v_mov_b32 v15, 0 // (op 926) convert
v_mov_b32 v14, v13 // (op 926) convert

v_lshrrev_b32 v12, 1, v1 // (op 926) 
v_mul_lo_u32 v17, v12, s13 // (op 926) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 926) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 926) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 926) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 926) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 926) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 926) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 926) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 926) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 926) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 926) 

v_mov_b32 v14, v12 // (op 926) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 928) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 928) least significant
s_mul_hi_u32 s6, s22, 6 // (op 928) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 928) most significant: sum

s_mul_i32 s7, s12, s1 // (op 928) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 928) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 928) least significant
s_mul_hi_u32 s11, s12, s0 // (op 928) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 928) most significant: sum
s_add_u32 s7, s7, s11 // (op 928) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 928) 

s_mov_b32 s48, s20 // (op 928) reg expression
s_mov_b32 s49, s21 // (op 928) reg expression
s_mov_b32 s50, s0 // (op 928) reg expression
s_mov_b32 s51, 131072 // (op 928) call()

v_and_b32 v12, -2, v1 // (op 945) 
v_sub_u32 v13, v1, v12 // (op 945) 

v_lshlrev_b32 v12, 4, v13 // (op 945) 

v_lshl_add_u32 v13, s16, 5, v12 // (op 945) 

v_mov_b32 v17, 0 // (op 945) convert
v_mov_b32 v16, v13 // (op 945) convert

v_lshrrev_b32 v15, 1, v1 // (op 945) 
v_mul_lo_u32 v13, v15, s13 // (op 945) most significant: low of low * high
v_mul_hi_u32 v18, v15, s12 // (op 945) most significant: high of low * low
v_mul_lo_u32 v12, v15, s12 // (op 945) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 945) most significant: sum


v_add_co_u32 v18, s[0:1], v16, v12 // (op 945) least significant half
v_addc_co_u32 v19, s[0:1], v17, v13, s[0:1] // (op 945) most significant half



v_mul_lo_u32 v13, v19, 6 // (op 945) most significant: low of high * low
v_mul_hi_u32 v15, v18, 6 // (op 945) most significant: high of low * low
v_mul_lo_u32 v12, v18, 6 // (op 945) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 945) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 945) 

v_mov_b32 v15, v16 // (op 945) convert

s_mul_i32 s1, s23, 6 // (op 947) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 947) least significant
s_mul_hi_u32 s6, s22, 6 // (op 947) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 947) most significant: sum

s_mul_i32 s7, s12, s1 // (op 947) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 947) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 947) least significant
s_mul_hi_u32 s11, s12, s0 // (op 947) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 947) most significant: sum
s_add_u32 s7, s7, s11 // (op 947) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 947) 

s_mov_b32 s48, s20 // (op 947) reg expression
s_mov_b32 s49, s21 // (op 947) reg expression
s_mov_b32 s50, s0 // (op 947) reg expression
s_mov_b32 s51, 131072 // (op 947) call()

v_and_b32 v12, -8, v1 // (op 990) 
v_sub_u32 v13, v1, v12 // (op 990) 

v_lshlrev_b32 v12, 4, v13 // (op 990) 

v_mov_b32 v17, 0 // (op 990) convert
v_mov_b32 v16, v12 // (op 990) convert

v_lshrrev_b32 v12, 3, v1 // (op 990) 
v_lshl_add_u32 v13, s17, 6, v12 // (op 990) 

v_mul_lo_u32 v19, v13, s25 // (op 990) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 990) most significant: high of low * low
v_mul_lo_u32 v18, v13, s24 // (op 990) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 990) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 990) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 990) most significant half



s_mul_i32 s1, s24, s9 // (op 992) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 992) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 992) least significant
s_mul_hi_u32 s7, s24, s8 // (op 992) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 992) most significant: sum
s_add_u32 s1, s1, s7 // (op 992) most significant: sum


s_mov_b32 s52, s26 // (op 992) reg expression
s_mov_b32 s53, s27 // (op 992) reg expression
s_mov_b32 s54, s0 // (op 992) reg expression
s_mov_b32 s55, 131072 // (op 992) call()

v_and_b32 v16, -8, v1 // (op 1009) 
v_sub_u32 v17, v1, v16 // (op 1009) 

v_lshlrev_b32 v16, 4, v17 // (op 1009) 

v_mov_b32 v19, 0 // (op 1009) convert
v_mov_b32 v18, v16 // (op 1009) convert

v_lshrrev_b32 v16, 3, v1 // (op 1009) 
v_lshl_add_u32 v17, s17, 6, v16 // (op 1009) 

v_mul_lo_u32 v21, v17, s25 // (op 1009) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 1009) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 1009) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 1009) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 1009) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 1009) most significant half



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

v_lshrrev_b32 v18, 7, v1 // (op 1054) 
v_and_b32 v19, 15, v1 // (op 1054) 
v_lshl_add_u32 v20, v18, 4, v19 // (op 1054) 


v_lshl_add_u32 v18, s16, 5, v20 // (op 1054) 

v_and_b32 v19, 63, v1 // (op 1054) 
v_lshrrev_b32 v20, 4, v19 // (op 1054) 

v_mul_lo_u32 v19, v20, s28 // (op 1054) 

v_add_u32 v20, v18, v19 // (op 1054) 


s_mul_i32 s1, s28, s33 // (op 1056) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1056) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1056) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1056) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1056) most significant: sum
s_add_u32 s1, s1, s7 // (op 1056) most significant: sum


s_mov_b32 s56, s30 // (op 1056) reg expression
s_mov_b32 s57, s31 // (op 1056) reg expression
s_mov_b32 s58, s0 // (op 1056) reg expression
s_mov_b32 s59, 131072 // (op 1056) call()

v_lshrrev_b32 v18, 7, v1 // (op 1073) 
v_and_b32 v19, 15, v1 // (op 1073) 
v_lshl_add_u32 v21, v18, 4, v19 // (op 1073) 


v_lshl_add_u32 v18, s16, 5, v21 // (op 1073) 

v_and_b32 v19, 63, v1 // (op 1073) 
v_lshrrev_b32 v21, 4, v19 // (op 1073) 

v_mul_lo_u32 v19, v21, s28 // (op 1073) 

v_add_u32 v21, v18, v19 // (op 1073) 


s_mul_i32 s1, s28, s33 // (op 1075) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1075) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1075) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1075) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1075) most significant: sum
s_add_u32 s1, s1, s7 // (op 1075) most significant: sum


s_mov_b32 s56, s30 // (op 1075) reg expression
s_mov_b32 s57, s31 // (op 1075) reg expression
s_mov_b32 s58, s0 // (op 1075) reg expression
s_mov_b32 s59, 131072 // (op 1075) call()

v_and_b32 v18, -4, v1 // (op 1118) 
v_sub_u32 v19, v1, v18 // (op 1118) 

v_lshrrev_b32 v18, 2, v1 // (op 1118) 
v_lshl_add_u32 v22, s17, 6, v18 // (op 1118) 

v_mul_lo_u32 v18, v22, s34 // (op 1118) 

v_add_u32 v22, v19, v18 // (op 1118) 


s_mul_i32 s1, s34, s9 // (op 1120) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1120) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1120) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1120) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1120) most significant: sum
s_add_u32 s1, s1, s7 // (op 1120) most significant: sum


s_mov_b32 s60, s36 // (op 1120) reg expression
s_mov_b32 s61, s37 // (op 1120) reg expression
s_mov_b32 s62, s0 // (op 1120) reg expression
s_mov_b32 s63, 131072 // (op 1120) call()

v_and_b32 v18, -4, v1 // (op 1137) 
v_sub_u32 v19, v1, v18 // (op 1137) 

v_lshrrev_b32 v18, 2, v1 // (op 1137) 
v_lshl_add_u32 v23, s17, 6, v18 // (op 1137) 

v_mul_lo_u32 v18, v23, s34 // (op 1137) 

v_add_u32 v23, v19, v18 // (op 1137) 


s_mul_i32 s1, s34, s9 // (op 1139) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1139) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1139) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1139) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1139) most significant: sum
s_add_u32 s1, s1, s7 // (op 1139) most significant: sum


s_mov_b32 s60, s36 // (op 1139) reg expression
s_mov_b32 s61, s37 // (op 1139) reg expression
s_mov_b32 s62, s0 // (op 1139) reg expression
s_mov_b32 s63, 131072 // (op 1139) call()


v_mov_b64 v[18:19], s[28:29] // (op 769) convert

v_mov_b32 v24, v20 // (op 769) reg expression
v_mov_b32 v25, v24 // (op 769) 
buffer_load_ubyte v26, v25, s[56:59], 0 offen offset: 0 // (op 769) Load value



v_and_b32 v18, -8, v1 // (op 1256) 
v_sub_u32 v19, v1, v18 // (op 1256) 

v_lshlrev_b32 v18, 4, v19 // (op 1256) 

v_lshrrev_b32 v19, 3, v1 // (op 1256) 
v_lshl_add_u32 v24, v19, 7, v18 // (op 1256) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 1256) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 881) 
v_mov_b64 v[18:19], s[6:7] // (op 881) convert


v_mov_b64 v[24:25], v[12:13] // (op 881) 
s_mov_b32 m0, s0 // (op 881) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 881) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v18 // (op 881) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 881) most significant half

s_add_u32 m0, m0, 4096 // (op 881) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 881) Load value direct to lds






buffer_load_ubyte v18, v22, s[60:63], 0 offen offset: 0 // (op 776) Load value


v_mov_b32 v19, v14 // (op 779) reg expression

buffer_load_dwordx3 v[28:30], v19, s[48:51], 0 offen // (op 779) Load value


v_mov_b64 v[24:25], s[28:29] // (op 783) convert

s_lshl_b32 s0, s28, 2 // (op 783) 
v_add_u32 v19, v20, s0 // (op 783) 

v_mov_b32 v27, v19 // (op 783) 
buffer_load_ubyte v31, v27, s[56:59], 0 offen offset: 0 // (op 783) Load value




v_and_b32 v27, -8, v1 // (op 1400) 
v_sub_u32 v19, v1, v27 // (op 1400) 

v_lshlrev_b32 v20, 4, v19 // (op 1400) 

v_lshrrev_b32 v19, 3, v1 // (op 1400) 
v_lshl_add_u32 v27, v19, 7, v20 // (op 1400) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1400) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 886) 
v_mov_b64 v[24:25], s[6:7] // (op 886) convert


v_mov_b64 v[32:33], v[12:13] // (op 886) 
s_add_u32 s1, 8192, s0 // (op 886) 
s_mov_b32 m0, s1 // (op 886) reg expression
s_mov_b32 s6, 128 // (op 886) call()
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 886) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 886) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 886) most significant half

s_add_u32 m0, m0, 4096 // (op 886) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 886) Load value direct to lds









buffer_load_ubyte v27, v22, s[60:63], 0 offen offset:4 // (op 790) Load value



s_lshl_b64 s[0:1], s[12:13], 7 // (op 793) 
s_mul_i32 s7, s1, 6 // (op 793) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 793) least significant
s_mul_hi_u32 s10, s0, 6 // (op 793) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 793) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 793) 

v_add_u32 v22, v14, s0 // (op 793) 


buffer_load_dwordx3 v[32:34], v22, s[48:51], 0 offen // (op 793) Load value


v_and_b32 v22, -4, v1 // (op 1314) 
v_sub_u32 v12, v1, v22 // (op 1314) 

v_and_b32 v22, 4294967292, v1 // (op 1314) 
v_add_u32 v13, v22, v12 // (op 1314) 





s_waitcnt vmcnt(6)
ds_write_b8 v13, v18 offset:16384 // (op 796) Store local data 

v_and_b32 v22, -2, v1 // (op 1204) 
v_sub_u32 v12, v1, v22 // (op 1204) 

v_lshlrev_b32 v22, 4, v12 // (op 1204) 

v_lshrrev_b32 v12, 1, v1 // (op 1204) 
v_lshl_add_u32 v13, v12, 5, v22 // (op 1204) 


v_mul_lo_u32 v12, v13, 6 // (op 1204) 
v_lshrrev_b32 v14, 3, v12 // (op 1204) 

v_lshrrev_b32 v12, 4, v13 // (op 1204) 

v_lshlrev_b32 v13, 2, v12 // (op 1204) 

v_add_u32 v12, v14, v13 // (op 1204) 





s_waitcnt vmcnt(5)
ds_write_b96 v12, v[28:30] offset:16640 // (op 797) Store local data 

v_and_b32 v22, -2, v1 // (op 1175) 
v_sub_u32 v12, v1, v22 // (op 1175) 

v_lshlrev_b32 v22, 4, v12 // (op 1175) 

v_lshrrev_b32 v12, 1, v1 // (op 1175) 
v_lshl_add_u32 v13, v12, 5, v22 // (op 1175) 


v_mul_lo_u32 v12, v13, 6 // (op 1175) 
v_lshrrev_b32 v14, 3, v12 // (op 1175) 

v_lshrrev_b32 v12, 4, v13 // (op 1175) 

v_lshlrev_b32 v13, 2, v12 // (op 1175) 

v_add_u32 v12, v14, v13 // (op 1175) 


v_and_b32 v22, -8, v1 // (op 1228) 
v_sub_u32 v13, v1, v22 // (op 1228) 

v_lshlrev_b32 v14, 4, v13 // (op 1228) 

v_lshrrev_b32 v13, 3, v1 // (op 1228) 
v_lshl_add_u32 v22, v13, 7, v14 // (op 1228) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 1228) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 1285) 
v_sub_u32 v13, v1, v22 // (op 1285) 

v_and_b32 v14, 4294967292, v1 // (op 1285) 
v_add_u32 v22, v14, v13 // (op 1285) 


v_and_b32 v13, -2, v1 // (op 1361) 
v_sub_u32 v14, v1, v13 // (op 1361) 

v_lshlrev_b32 v13, 4, v14 // (op 1361) 

v_lshrrev_b32 v14, 1, v1 // (op 1361) 
v_lshl_add_u32 v19, v14, 5, v13 // (op 1361) 


v_mul_lo_u32 v20, v19, 6 // (op 1361) 
v_lshrrev_b32 v13, 3, v20 // (op 1361) 

v_lshrrev_b32 v14, 4, v19 // (op 1361) 

v_lshlrev_b32 v19, 2, v14 // (op 1361) 

v_add_u32 v14, v13, v19 // (op 1361) 


v_and_b32 v13, -8, v1 // (op 1385) 
v_sub_u32 v19, v1, v13 // (op 1385) 

v_lshlrev_b32 v13, 4, v19 // (op 1385) 

v_lshrrev_b32 v19, 3, v1 // (op 1385) 
v_lshl_add_u32 v20, v19, 7, v13 // (op 1385) 


s_nop 0
v_readfirstlane_b32 s1, v20 // (op 1385) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -4, v1 // (op 1429) 
v_sub_u32 v19, v1, v13 // (op 1429) 

v_and_b32 v13, 4294967292, v1 // (op 1429) 
v_add_u32 v20, v13, v19 // (op 1429) 


s_mov_b32 s6, 0 // (op 312) call()
s_ashr_i64 s[10:11], s[22:23], 7 // (op 311) 
s_mov_b32 s7, s10 // (op 311) convert

s_and_b32 s10, s7, 4294967294 // (op 311) 

s_cmp_lt_u32 s6, s10 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_311 // (op 311) Condition: Top (jump to RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_311 if false)
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoop_311:
s_add_u32 s6, s6, 2 // (op 313) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1487) 



ds_read_b96_tr_b6 v[36:38], v0 offset:16640 // (op 67) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v0 offset:17152 // (op 67) Transpose load local data 
s_barrier  // (op 1500) 



ds_read_b128 v[42:45], v5 // (op 72) Load local data 
ds_read_b128 v[46:49], v5 offset:64 // (op 72) Load local data 
s_barrier  // (op 1513) 



ds_read_u8 v13, v4 offset:16384 // (op 81) Load local data 



ds_read_b128 v[50:53], v5 offset:4096 // (op 343) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:49], a[0:3], v26, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 84) 
ds_read_b128 v[54:57], v5 offset:4160 // (op 343) Load local data 





ds_read_u8 v13, v4 offset:16512 // (op 346) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[50:57], a[4:7], v26, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 347) 




v_mov_b64 v[24:25], s[28:29] // (op 77) convert

v_mov_b32 v13, v21 // (op 77) 
s_lshl_b32 s7, s28, 3 // (op 77) 
v_add_u32 v19, v13, s7 // (op 77) 

v_mov_b32 v35, v19 // (op 77) 
buffer_load_ubyte v26, v35, s[56:59], 0 offen offset: 0 // (op 77) Load value



s_barrier  // (op 1503) 

s_lshl_b64 s[10:11], s[24:25], 5 // (op 864) 
v_mov_b64 v[24:25], s[10:11] // (op 864) convert


v_mov_b64 v[36:37], v[16:17] // (op 864) 
v_mov_b64 v[38:39], v[36:37] // (op 864) 
s_mov_b32 m0, s0 // (op 864) reg expression
s_mov_b32 s7, 256 // (op 864) call()
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 864) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 864) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 864) most significant half

s_add_u32 m0, m0, 4096 // (op 864) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 864) Load value direct to lds





v_mov_b32 v19, v23 // (op 21) 

buffer_load_ubyte v18, v19, s[60:63], 0 offen offset:8 // (op 21) Load value


v_mov_b32 v35, v15 // (op 4) 
s_lshl_b64 s[10:11], s[12:13], 7 // (op 4) 
s_mul_i32 s15, s11, 6 // (op 4) most significant: low of high * low
s_mul_i32 s14, s10, 6 // (op 4) least significant
s_mul_hi_u32 s7, s10, 6 // (op 4) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 4) most significant: sum


s_lshr_b64 s[10:11], s[14:15], 3 // (op 4) 

s_lshl_b32 s7, s10, 1 // (op 4) 

v_add_u32 v24, v35, s7 // (op 4) 


buffer_load_dwordx3 v[28:30], v24, s[48:51], 0 offen // (op 4) Load value

s_barrier  // (op 1542) 



s_waitcnt vmcnt(6)
ds_write_b8 v20, v27 offset:20736 // (op 240) Store local data 
s_barrier  // (op 1530) 



s_waitcnt vmcnt(5)
ds_write_b96 v14, v[32:34] offset:20992 // (op 229) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1527) 



ds_read_b96_tr_b6 v[38:40], v3 offset:20992 // (op 231) Transpose load local data 
ds_read_b96_tr_b6 v[41:43], v3 offset:21504 // (op 231) Transpose load local data 
s_barrier  // (op 1533) 
s_barrier  // (op 1577) 



ds_read_b128 v[44:47], v8 offset:8192 // (op 233) Load local data 
ds_read_b128 v[48:51], v8 offset:8256 // (op 233) Load local data 
s_barrier  // (op 1539) 



ds_read_u8 v24, v7 offset:20736 // (op 237) Load local data 



ds_read_b128 v[52:55], v8 offset:12288 // (op 352) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:43], v[44:51], a[0:3], v31, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 238) 
ds_read_b128 v[56:59], v8 offset:12352 // (op 352) Load local data 





ds_read_u8 v24, v7 offset:20864 // (op 355) Load local data 
s_barrier  // (op 1613) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:43], v[52:59], a[4:7], v31, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 348) 




v_mov_b64 v[24:25], s[28:29] // (op 235) convert

s_lshl_b32 s7, s28, 2 // (op 235) 
s_mul_i32 s10, 3, s7 // (op 235) 

v_add_u32 v38, v13, s10 // (op 235) 

v_mov_b32 v39, v38 // (op 235) 
buffer_load_ubyte v31, v39, s[56:59], 0 offen offset: 0 // (op 235) Load value




s_barrier  // (op 1536) 
s_lshl_b32 s7, s28, 3 // (op 1065) 

v_add_u32 v21, v21, s7 // (op 1065) 

s_lshl_b64 s[10:11], s[24:25], 5 // (op 876) 
v_mov_b64 v[24:25], s[10:11] // (op 876) convert


v_mov_b64 v[38:39], v[36:37] // (op 876) 
s_add_u32 s7, 8192, s1 // (op 876) 
s_mov_b32 m0, s7 // (op 876) reg expression
s_mov_b32 s10, 384 // (op 876) call()
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 876) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 876) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 876) most significant half

s_add_u32 m0, m0, 4096 // (op 876) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 876) Load value direct to lds








v_add_co_u32 v16, vcc, 256, v16 // (op 1001) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1001) most significant half
buffer_load_ubyte v27, v19, s[60:63], 0 offen offset:12 // (op 242) Load value



v_add_u32 v23, 8, v23 // (op 1129) 
s_lshl_b64 s[10:11], s[12:13], 7 // (op 228) 
s_mul_i32 s15, s11, 6 // (op 228) most significant: low of high * low
s_mul_i32 s14, s10, 6 // (op 228) least significant
s_mul_hi_u32 s7, s10, 6 // (op 228) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 228) most significant: sum


s_lshr_b64 s[10:11], s[14:15], 3 // (op 228) 

s_mul_i32 s7, 3, s10 // (op 228) 

v_add_u32 v13, v35, s7 // (op 228) 


buffer_load_dwordx3 v[32:34], v13, s[48:51], 0 offen // (op 228) Load value


s_barrier  // (op 1516) 
s_lshl_b64 s[10:11], s[12:13], 7 // (op 937) 



s_mul_i32 s15, s11, 6 // (op 937) most significant: low of high * low
s_mul_i32 s14, s10, 6 // (op 937) least significant
s_mul_hi_u32 s7, s10, 6 // (op 937) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 937) most significant: sum


s_lshr_b64 s[10:11], s[14:15], 3 // (op 937) 

s_lshl_b32 s7, s10, 1 // (op 937) 

v_add_u32 v15, v15, s7 // (op 937) 

s_waitcnt vmcnt(6)
ds_write_b8 v22, v18 offset:16384 // (op 26) Store local data 
s_barrier  // (op 1490) 



s_waitcnt vmcnt(5)
ds_write_b96 v12, v[28:30] offset:16640 // (op 6) Store local data 
s_ashr_i64 s[10:11], s[22:23], 7 // (op 311) 
s_mov_b32 s7, s10 // (op 311) convert

s_and_b32 s10, s7, 4294967294 // (op 311) 

s_cmp_lt_u32 s6, s10 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoop_311 // (op 311) Condition: Bottom (jump to RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoop_311 if true)
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_311:














s_barrier  // (op 1595) 
v_and_b32 v12, -2, v1 // (op 899) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 899) 
v_sub_u32 v13, v1, v12 // (op 899) 

s_lshr_b32 s6, s0, 1 // (op 899) 

v_lshlrev_b32 v12, 4, v13 // (op 899) 

v_lshl_add_u32 v13, s16, 5, v12 // (op 899) 

v_mov_b32 v15, 0 // (op 899) convert
v_mov_b32 v14, v13 // (op 899) convert

v_lshrrev_b32 v12, 1, v1 // (op 899) 
v_lshl_add_u32 v13, s6, 8, v12 // (op 899) 


v_mul_lo_u32 v17, v13, s13 // (op 899) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 899) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 899) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 899) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 899) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 899) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 899) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 899) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 899) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 899) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 899) 

v_mov_b32 v14, v12 // (op 899) convert

s_mul_i32 s1, s23, 6 // (op 901) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 901) least significant
s_mul_hi_u32 s6, s22, 6 // (op 901) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 901) most significant: sum

s_mul_i32 s7, s12, s1 // (op 901) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 901) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 901) least significant
s_mul_hi_u32 s11, s12, s0 // (op 901) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 901) most significant: sum
s_add_u32 s7, s7, s11 // (op 901) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 901) 

s_mov_b32 s48, s20 // (op 901) reg expression
s_mov_b32 s49, s21 // (op 901) reg expression
s_mov_b32 s50, s0 // (op 901) reg expression
s_mov_b32 s51, 131072 // (op 901) call()

v_and_b32 v12, -8, v1 // (op 967) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 967) 
v_sub_u32 v13, v1, v12 // (op 967) 

s_lshr_b32 s6, s0, 1 // (op 967) 

v_lshlrev_b32 v12, 4, v13 // (op 967) 

v_lshl_add_u32 v13, s6, 8, v12 // (op 967) 


v_mov_b32 v29, 0 // (op 967) convert
v_mov_b32 v28, v13 // (op 967) convert

v_lshrrev_b32 v30, 3, v1 // (op 967) 
v_lshl_add_u32 v12, s17, 6, v30 // (op 967) 

v_mul_lo_u32 v17, v12, s25 // (op 967) most significant: low of low * high
v_mul_hi_u32 v13, v12, s24 // (op 967) most significant: high of low * low
v_mul_lo_u32 v16, v12, s24 // (op 967) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 967) most significant: sum


v_add_co_u32 v12, s[0:1], v28, v16 // (op 967) least significant half
v_addc_co_u32 v13, s[0:1], v29, v17, s[0:1] // (op 967) most significant half



s_mul_i32 s1, s24, s9 // (op 969) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 969) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 969) least significant
s_mul_hi_u32 s7, s24, s8 // (op 969) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 969) most significant: sum
s_add_u32 s1, s1, s7 // (op 969) most significant: sum


s_mov_b32 s52, s26 // (op 969) reg expression
s_mov_b32 s53, s27 // (op 969) reg expression
s_mov_b32 s54, s0 // (op 969) reg expression
s_mov_b32 s55, 131072 // (op 969) call()

v_lshrrev_b32 v15, 7, v1 // (op 1031) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1031) 
v_and_b32 v16, 15, v1 // (op 1031) 
s_lshr_b32 s6, s0, 1 // (op 1031) 

v_lshl_add_u32 v17, v15, 4, v16 // (op 1031) 


v_lshl_add_u32 v15, s16, 5, v17 // (op 1031) 

v_and_b32 v16, 63, v1 // (op 1031) 
v_lshrrev_b32 v17, 4, v16 // (op 1031) 

v_lshl_add_u32 v16, s6, 3, v17 // (op 1031) 


v_mul_lo_u32 v17, v16, s28 // (op 1031) 

v_add_u32 v16, v15, v17 // (op 1031) 


s_mul_i32 s1, s28, s33 // (op 1033) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1033) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1033) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1033) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1033) most significant: sum
s_add_u32 s1, s1, s7 // (op 1033) most significant: sum


s_mov_b32 s56, s30 // (op 1033) reg expression
s_mov_b32 s57, s31 // (op 1033) reg expression
s_mov_b32 s58, s0 // (op 1033) reg expression
s_mov_b32 s59, 131072 // (op 1033) call()

v_and_b32 v15, -4, v1 // (op 1095) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1095) 
v_sub_u32 v17, v1, v15 // (op 1095) 

s_lshr_b32 s6, s0, 1 // (op 1095) 

v_lshrrev_b32 v15, 2, v1 // (op 1095) 
v_lshl_add_u32 v18, s6, 3, v17 // (op 1095) 


v_lshl_add_u32 v17, s17, 6, v15 // (op 1095) 

v_mul_lo_u32 v15, v17, s34 // (op 1095) 

v_add_u32 v17, v18, v15 // (op 1095) 


s_mul_i32 s1, s34, s9 // (op 1097) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1097) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1097) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1097) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1097) most significant: sum
s_add_u32 s1, s1, s7 // (op 1097) most significant: sum


s_mov_b32 s60, s36 // (op 1097) reg expression
s_mov_b32 s61, s37 // (op 1097) reg expression
s_mov_b32 s62, s0 // (op 1097) reg expression
s_mov_b32 s63, 131072 // (op 1097) call()

v_and_b32 v15, -2, v1 // (op 1188) 
v_sub_u32 v18, v1, v15 // (op 1188) 

v_lshlrev_b32 v15, 4, v18 // (op 1188) 

v_lshrrev_b32 v18, 1, v1 // (op 1188) 
v_lshl_add_u32 v19, v18, 5, v15 // (op 1188) 


v_mul_lo_u32 v15, v19, 6 // (op 1188) 
v_lshrrev_b32 v18, 3, v15 // (op 1188) 

v_lshrrev_b32 v15, 4, v19 // (op 1188) 

v_lshlrev_b32 v19, 2, v15 // (op 1188) 

v_add_u32 v15, v18, v19 // (op 1188) 


v_and_b32 v18, -8, v1 // (op 1241) 
v_sub_u32 v19, v1, v18 // (op 1241) 

v_lshlrev_b32 v18, 4, v19 // (op 1241) 

v_lshrrev_b32 v19, 3, v1 // (op 1241) 
v_lshl_add_u32 v20, v19, 7, v18 // (op 1241) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1241) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -4, v1 // (op 1298) 
v_sub_u32 v19, v1, v18 // (op 1298) 

v_and_b32 v18, 4294967292, v1 // (op 1298) 
v_add_u32 v20, v18, v19 // (op 1298) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 317) 
s_and_b32 s1, s6, 4294967294 // (op 317) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 316) 
s_mov_b32 s10, s6 // (op 316) convert

s_cmp_lt_u32 s1, s10 // (op 316) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_316 // (op 316) Condition: Top (jump to RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_316 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_316:


s_add_u32 s1, s1, 1 // (op 318) 
v_mov_b32 v18, v14 // (op 170) 
v_mov_b32 v19, v18 // (op 170) reg expression

buffer_load_dwordx3 v[22:24], v19, s[48:51], 0 offen // (op 170) Load value


s_lshl_b64 s[6:7], s[12:13], 7 // (op 889) 
s_mul_i32 s11, s7, 6 // (op 889) most significant: low of high * low
s_mul_i32 s10, s6, 6 // (op 889) least significant
s_mul_hi_u32 s14, s6, 6 // (op 889) most significant: high of low * low
s_add_u32 s11, s11, s14 // (op 889) most significant: sum


s_lshr_b64 s[6:7], s[10:11], 3 // (op 889) 

v_add_u32 v14, v14, s6 // (op 889) 

s_barrier  // (op 1493) 
s_barrier  // (op 1506) 




s_lshl_b64 s[6:7], s[24:25], 5 // (op 869) 
v_mov_b64 v[18:19], s[6:7] // (op 869) convert


v_mov_b64 v[26:27], v[12:13] // (op 869) 
v_mov_b64 v[28:29], v[26:27] // (op 869) 
s_mov_b32 m0, s0 // (op 869) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 869) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 869) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 869) most significant half

s_add_u32 m0, m0, 4096 // (op 869) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 869) Load value direct to lds



v_add_co_u32 v12, vcc, 128, v12 // (op 959) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 959) most significant half
s_waitcnt vmcnt(2)
ds_write_b96 v15, v[22:24] offset:16640 // (op 171) Store local data 



v_mov_b32 v18, v17 // (op 184) 

buffer_load_ubyte v19, v18, s[60:63], 0 offen offset: 0 // (op 184) Load value

v_add_u32 v17, 4, v17 // (op 1087) 
s_barrier  // (op 1519) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1496) 






ds_read_b96_tr_b6 v[22:24], v0 offset:16640 // (op 173) Transpose load local data 
ds_read_b96_tr_b6 v[25:27], v0 offset:17152 // (op 173) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v19 offset:16384 // (op 182) Store local data 

s_barrier  // (op 1509) 



ds_read_b128 v[28:31], v5 // (op 175) Load local data 
ds_read_b128 v[32:35], v5 offset:64 // (op 175) Load local data 

v_mov_b64 v[18:19], s[28:29] // (op 177) convert

v_mov_b32 v21, v16 // (op 177) 
v_mov_b32 v36, v21 // (op 177) reg expression
v_mov_b32 v37, v36 // (op 177) 
buffer_load_ubyte v38, v37, s[56:59], 0 offen offset: 0 // (op 177) Load value




s_lshl_b32 s6, s28, 2 // (op 1023) 
v_add_u32 v16, v16, s6 // (op 1023) 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1523) 



ds_read_u8 v21, v4 offset:16384 // (op 179) Load local data 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:27], v[28:35], a[0:3], v38, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 180) 
ds_read_b128 v[40:43], v5 offset:4096 // (op 324) Load local data 


ds_read_b128 v[44:47], v5 offset:4160 // (op 324) Load local data 



ds_read_u8 v21, v4 offset:16512 // (op 327) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:27], v[40:47], a[4:7], v38, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 328) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 316) 
s_mov_b32 s10, s6 // (op 316) convert

s_cmp_lt_u32 s1, s10 // (op 316) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_316 // (op 316) Condition: Bottom (jump to RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_316 if true)
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_316:






















s_barrier  // (op 1599) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_125
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 

v_lshrrev_b32 v12, 16, v16 // (op 52) 
v_lshrrev_b32 v13, 16, v17 // (op 52) 
v_lshrrev_b32 v14, 16, v18 // (op 52) 
v_lshrrev_b32 v15, 16, v19 // (op 52) 


v_pack_b32_f16 v4, v12, v13 // (op 54) 
v_pack_b32_f16 v5, v14, v15 // (op 54) 

v_mov_b32 v0, v9 // (op 54) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 54) Store value



v_accvgpr_read v12, a4 // (op 508) 
v_accvgpr_read v13, a5 // (op 508) 
v_accvgpr_read v14, a6 // (op 508) 
v_accvgpr_read v15, a7 // (op 508) 
v_mul_f32 v16, v11, v12 // (op 508) 
v_mul_f32 v17, v11, v13 // (op 508) 
v_mul_f32 v18, v11, v14 // (op 508) 
v_mul_f32 v19, v11, v15 // (op 508) 

v_lshrrev_b32 v12, 16, v16 // (op 510) 
v_lshrrev_b32 v13, 16, v17 // (op 510) 
v_lshrrev_b32 v14, 16, v18 // (op 510) 
v_lshrrev_b32 v15, 16, v19 // (op 510) 


v_pack_b32_f16 v4, v12, v13 // (op 511) 
v_pack_b32_f16 v5, v14, v15 // (op 511) 

s_lshl_b32 s0, s4, 6 // (op 511) 
v_add_u32 v0, v9, s0 // (op 511) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 511) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_125:


v_mov_b32 v0, v6 // (op 138) reg expression

buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 138) Load value

v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v11, v12 // (op 134) 
v_mul_f32 v17, v11, v13 // (op 134) 
v_mul_f32 v18, v11, v14 // (op 134) 
v_mul_f32 v19, v11, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 139) 
v_lshrrev_b32 v13, 16, v4 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v20, v10, v12 // (op 139) 
v_mul_f32 v21, v10, v13 // (op 139) 
v_and_b32 v12, 65535, v5 // (op 139) 
v_lshrrev_b32 v13, 16, v5 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v22, v10, v12 // (op 139) 
v_mul_f32 v23, v10, v13 // (op 139) 


v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 
v_add_f32 v14, v22, v18 // (op 135) 
v_add_f32 v15, v23, v19 // (op 135) 


v_lshrrev_b32 v16, 16, v12 // (op 136) 
v_lshrrev_b32 v17, 16, v13 // (op 136) 
v_lshrrev_b32 v18, 16, v14 // (op 136) 
v_lshrrev_b32 v19, 16, v15 // (op 136) 


v_pack_b32_f16 v4, v16, v17 // (op 137) 
v_pack_b32_f16 v5, v18, v19 // (op 137) 

v_mov_b32 v0, v9 // (op 137) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 137) Store value





s_lshl_b32 s0, s2, 6 // (op 535) 
v_add_u32 v0, v6, s0 // (op 535) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 535) Load value

v_accvgpr_read v12, a4 // (op 531) 
v_accvgpr_read v13, a5 // (op 531) 
v_accvgpr_read v14, a6 // (op 531) 
v_accvgpr_read v15, a7 // (op 531) 
v_mul_f32 v16, v11, v12 // (op 531) 
v_mul_f32 v17, v11, v13 // (op 531) 
v_mul_f32 v18, v11, v14 // (op 531) 
v_mul_f32 v19, v11, v15 // (op 531) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 536) 
v_lshrrev_b32 v13, 16, v4 // (op 536) 
v_lshlrev_b32 v12, 16, v12 // (op 536) 
v_lshlrev_b32 v13, 16, v13 // (op 536) 
v_mul_f32 v20, v10, v12 // (op 536) 
v_mul_f32 v21, v10, v13 // (op 536) 
v_and_b32 v12, 65535, v5 // (op 536) 
v_lshrrev_b32 v13, 16, v5 // (op 536) 
v_lshlrev_b32 v12, 16, v12 // (op 536) 
v_lshlrev_b32 v13, 16, v13 // (op 536) 
v_mul_f32 v22, v10, v12 // (op 536) 
v_mul_f32 v23, v10, v13 // (op 536) 


v_add_f32 v12, v20, v16 // (op 532) 
v_add_f32 v13, v21, v17 // (op 532) 
v_add_f32 v14, v22, v18 // (op 532) 
v_add_f32 v15, v23, v19 // (op 532) 


v_lshrrev_b32 v16, 16, v12 // (op 533) 
v_lshrrev_b32 v17, 16, v13 // (op 533) 
v_lshrrev_b32 v18, 16, v14 // (op 533) 
v_lshrrev_b32 v19, 16, v15 // (op 533) 


v_pack_b32_f16 v4, v16, v17 // (op 534) 
v_pack_b32_f16 v5, v18, v19 // (op 534) 

s_lshl_b32 s0, s4, 6 // (op 534) 
v_add_u32 v0, v9, s0 // (op 534) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 534) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
.LRR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_end:
.size RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2, .LRR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_end-RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
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
  MXScaleFormat: NoSwizzle
  ProblemType:
    OperationType: GEMM
    DataType: F6
    DataTypeA: F6
    DataTypeB: B8
    DestDataType: b
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
    macrotile: [32, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
    .symbol: RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 25088
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 60
    .agpr_count: 8
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_11
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
      - .name: Tensor_2_size_1_4
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
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
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

