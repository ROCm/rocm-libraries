// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2,@function
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 1274) 
v_and_b32 v3, 1, v0 // (op 1274) 

v_and_b32 v0, 15, v1 // (op 1274) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 1274) 


v_and_b32 v0, 48, v1 // (op 1274) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 1274) 


v_lshrrev_b32 v0, 6, v1 // (op 1418) 
v_and_b32 v4, 1, v0 // (op 1418) 

v_and_b32 v0, 15, v1 // (op 1418) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 1418) 


v_and_b32 v0, 48, v1 // (op 1418) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 1418) 


v_lshrrev_b32 v0, 7, v1 // (op 1160) 
v_and_b32 v5, 63, v1 // (op 1160) 
v_lshrrev_b32 v6, 4, v5 // (op 1160) 
v_lshlrev_b32 v7, 2, v6 // (op 1160) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1160) 


v_lshl_add_u32 v0, s16, 6, v6 // (op 1160) 

v_lshrrev_b32 v6, 2, v5 // (op 1160) 

v_and_b32 v5, 3, v6 // (op 1160) 

v_and_b32 v6, 3, v1 // (op 1160) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1160) 


v_lshrrev_b32 v5, 6, v1 // (op 1160) 
v_and_b32 v6, 1, v5 // (op 1160) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1160) 


v_lshl_add_u32 v6, s17, 5, v5 // (op 1160) 

v_mul_lo_u32 v5, v6, s2 // (op 1160) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 1160) 


v_lshrrev_b32 v0, 7, v1 // (op 1222) 
v_and_b32 v5, 15, v1 // (op 1222) 
v_lshl_add_u32 v7, v0, 4, v5 // (op 1222) 


v_and_b32 v0, 63, v1 // (op 1222) 
v_lshrrev_b32 v5, 4, v0 // (op 1222) 

v_lshlrev_b32 v0, 5, v5 // (op 1222) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 1222) 


v_mul_lo_u32 v0, v5, 6 // (op 1222) 

v_lshrrev_b32 v5, 3, v0 // (op 1222) 

v_lshrrev_b32 v0, 7, v1 // (op 1332) 
v_and_b32 v7, 15, v1 // (op 1332) 
s_mul_i32 s1, s2, s9 // (op 1162) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1162) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1162) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1162) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1162) most significant: sum
s_add_u32 s1, s1, s19 // (op 1162) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1162) 

v_lshl_add_u32 v8, v0, 4, v7 // (op 1332) 


v_and_b32 v0, 63, v1 // (op 1332) 
v_lshrrev_b32 v7, 4, v0 // (op 1332) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 1332) 


v_lshrrev_b32 v7, 7, v1 // (op 1348) 
v_and_b32 v8, 63, v1 // (op 1348) 
v_lshrrev_b32 v9, 4, v8 // (op 1348) 
v_lshlrev_b32 v10, 2, v9 // (op 1348) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1348) 


v_lshl_add_u32 v7, s16, 6, v9 // (op 1348) 

v_lshrrev_b32 v9, 2, v8 // (op 1348) 

v_and_b32 v8, 3, v9 // (op 1348) 

v_and_b32 v9, 3, v1 // (op 1348) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1348) 


v_lshrrev_b32 v8, 6, v1 // (op 1348) 
v_and_b32 v9, 1, v8 // (op 1348) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 1348) 


v_lshl_add_u32 v9, s17, 5, v8 // (op 1348) 

v_mul_lo_u32 v8, v9, s4 // (op 1348) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 1348) 


v_lshrrev_b32 v7, 7, v1 // (op 1379) 
v_and_b32 v8, 15, v1 // (op 1379) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 1379) 


v_and_b32 v7, 63, v1 // (op 1379) 
v_lshrrev_b32 v8, 4, v7 // (op 1379) 

v_lshlrev_b32 v7, 5, v8 // (op 1379) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 1379) 


v_mul_lo_u32 v7, v8, 6 // (op 1379) 

v_lshrrev_b32 v8, 3, v7 // (op 1379) 

v_lshrrev_b32 v7, 7, v1 // (op 1447) 
v_and_b32 v10, 15, v1 // (op 1447) 
s_mul_i32 s1, s4, s9 // (op 1350) most significant: low of low * high
v_lshl_add_u32 v11, v7, 4, v10 // (op 1447) 


v_and_b32 v7, 63, v1 // (op 1447) 
v_lshrrev_b32 v10, 4, v7 // (op 1447) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 1447) 


s_mul_i32 s38, s5, s8 // (op 1350) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1350) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1350) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1350) most significant: sum
s_add_u32 s1, s1, s39 // (op 1350) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 1350) 

s_mov_b32 s40, s6 // (op 1162) reg expression
s_mov_b32 s41, s7 // (op 1162) reg expression
s_mov_b32 s42, s18 // (op 1162) reg expression
s_mov_b32 s43, 131072 // (op 1162) call()

s_mov_b32 s44, s10 // (op 1350) reg expression

s_mov_b32 s45, s11 // (op 1350) reg expression
s_mov_b32 s46, s38 // (op 1350) reg expression
s_mov_b32 s47, 131072 // (op 1350) call()


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
v_lshrrev_b32 v12, 2, v1 // (op 927) 
v_lshl_add_u32 v13, s16, 6, v12 // (op 927) 

v_and_b32 v12, -4, v1 // (op 927) 
v_sub_u32 v14, v1, v12 // (op 927) 

v_lshlrev_b32 v12, 5, v14 // (op 927) 

v_mov_b32 v15, 0 // (op 927) convert
v_mov_b32 v14, v12 // (op 927) convert

v_mul_lo_u32 v17, v13, s13 // (op 927) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 927) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 927) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 927) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 927) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 927) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 927) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 927) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 927) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 927) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 927) 

v_mov_b32 v12, v14 // (op 927) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 929) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 929) least significant
s_mul_hi_u32 s6, s22, 6 // (op 929) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 929) most significant: sum

s_mul_i32 s7, s12, s1 // (op 929) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 929) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 929) least significant
s_mul_hi_u32 s11, s12, s0 // (op 929) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 929) most significant: sum
s_add_u32 s7, s7, s11 // (op 929) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 929) 

s_mov_b32 s48, s20 // (op 929) reg expression
s_mov_b32 s49, s21 // (op 929) reg expression
s_mov_b32 s50, s0 // (op 929) reg expression
s_mov_b32 s51, 131072 // (op 929) call()

v_lshrrev_b32 v13, 2, v1 // (op 946) 
v_lshl_add_u32 v14, s16, 6, v13 // (op 946) 

v_and_b32 v13, -4, v1 // (op 946) 
v_sub_u32 v15, v1, v13 // (op 946) 

v_lshlrev_b32 v13, 5, v15 // (op 946) 

v_mov_b32 v17, 0 // (op 946) convert
v_mov_b32 v16, v13 // (op 946) convert

v_mul_lo_u32 v19, v14, s13 // (op 946) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 946) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 946) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 946) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 946) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 946) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 946) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 946) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 946) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 946) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 946) 

v_mov_b32 v13, v14 // (op 946) convert

s_mul_i32 s1, s23, 6 // (op 948) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 948) least significant
s_mul_hi_u32 s6, s22, 6 // (op 948) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 948) most significant: sum

s_mul_i32 s7, s12, s1 // (op 948) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 948) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 948) least significant
s_mul_hi_u32 s11, s12, s0 // (op 948) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 948) most significant: sum
s_add_u32 s7, s7, s11 // (op 948) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 948) 

s_mov_b32 s48, s20 // (op 948) reg expression
s_mov_b32 s49, s21 // (op 948) reg expression
s_mov_b32 s50, s0 // (op 948) reg expression
s_mov_b32 s51, 131072 // (op 948) call()

v_and_b32 v14, -8, v1 // (op 991) 
v_sub_u32 v15, v1, v14 // (op 991) 

v_lshlrev_b32 v14, 4, v15 // (op 991) 

v_mov_b32 v17, 0 // (op 991) convert
v_mov_b32 v16, v14 // (op 991) convert

v_lshrrev_b32 v14, 3, v1 // (op 991) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 991) 

v_mul_lo_u32 v19, v15, s25 // (op 991) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 991) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 991) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 991) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 991) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 991) most significant half



s_mul_i32 s1, s24, s9 // (op 993) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 993) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 993) least significant
s_mul_hi_u32 s7, s24, s8 // (op 993) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 993) most significant: sum
s_add_u32 s1, s1, s7 // (op 993) most significant: sum


s_mov_b32 s52, s26 // (op 993) reg expression
s_mov_b32 s53, s27 // (op 993) reg expression
s_mov_b32 s54, s0 // (op 993) reg expression
s_mov_b32 s55, 131072 // (op 993) call()

v_and_b32 v16, -8, v1 // (op 1010) 
v_sub_u32 v17, v1, v16 // (op 1010) 

v_lshlrev_b32 v16, 4, v17 // (op 1010) 

v_mov_b32 v19, 0 // (op 1010) convert
v_mov_b32 v18, v16 // (op 1010) convert

v_lshrrev_b32 v16, 3, v1 // (op 1010) 
v_lshl_add_u32 v17, s17, 5, v16 // (op 1010) 

v_mul_lo_u32 v21, v17, s25 // (op 1010) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 1010) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 1010) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 1010) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 1010) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 1010) most significant half



s_mul_i32 s1, s24, s9 // (op 1012) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1012) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1012) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1012) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1012) most significant: sum
s_add_u32 s1, s1, s7 // (op 1012) most significant: sum


s_mov_b32 s52, s26 // (op 1012) reg expression
s_mov_b32 s53, s27 // (op 1012) reg expression
s_mov_b32 s54, s0 // (op 1012) reg expression
s_mov_b32 s55, 131072 // (op 1012) call()

v_lshrrev_b32 v18, 2, v1 // (op 1055) 
v_lshl_add_u32 v19, s16, 6, v18 // (op 1055) 

v_and_b32 v18, -4, v1 // (op 1055) 
v_sub_u32 v20, v1, v18 // (op 1055) 

v_mul_lo_u32 v18, v19, s28 // (op 1055) 
v_add_u32 v18, v18, v20 // (op 1055) 


s_mul_i32 s1, s28, s23 // (op 1057) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1057) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1057) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1057) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1057) most significant: sum
s_add_u32 s1, s1, s7 // (op 1057) most significant: sum


s_mov_b32 s56, s30 // (op 1057) reg expression
s_mov_b32 s57, s31 // (op 1057) reg expression
s_mov_b32 s58, s0 // (op 1057) reg expression
s_mov_b32 s59, 131072 // (op 1057) call()

v_lshrrev_b32 v19, 2, v1 // (op 1074) 
v_lshl_add_u32 v20, s16, 6, v19 // (op 1074) 

v_and_b32 v19, -4, v1 // (op 1074) 
v_sub_u32 v21, v1, v19 // (op 1074) 

v_mul_lo_u32 v19, v20, s28 // (op 1074) 
v_add_u32 v19, v19, v21 // (op 1074) 


s_mul_i32 s1, s28, s23 // (op 1076) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1076) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1076) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1076) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1076) most significant: sum
s_add_u32 s1, s1, s7 // (op 1076) most significant: sum


s_mov_b32 s56, s30 // (op 1076) reg expression
s_mov_b32 s57, s31 // (op 1076) reg expression
s_mov_b32 s58, s0 // (op 1076) reg expression
s_mov_b32 s59, 131072 // (op 1076) call()

v_lshrrev_b32 v20, 6, v1 // (op 1119) 
v_and_b32 v21, 1, v20 // (op 1119) 

v_and_b32 v20, 15, v1 // (op 1119) 
v_lshl_add_u32 v22, v21, 4, v20 // (op 1119) 


v_lshl_add_u32 v20, s17, 5, v22 // (op 1119) 

v_mul_lo_u32 v21, v20, s32 // (op 1119) 

v_and_b32 v20, 63, v1 // (op 1119) 
v_lshrrev_b32 v22, 4, v20 // (op 1119) 

v_add_u32 v20, v22, v21 // (op 1119) 


s_mul_i32 s1, s32, s9 // (op 1121) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1121) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1121) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1121) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1121) most significant: sum
s_add_u32 s1, s1, s7 // (op 1121) most significant: sum


s_mov_b32 s60, s34 // (op 1121) reg expression
s_mov_b32 s61, s35 // (op 1121) reg expression
s_mov_b32 s62, s0 // (op 1121) reg expression
s_mov_b32 s63, 131072 // (op 1121) call()

v_lshrrev_b32 v21, 6, v1 // (op 1138) 
v_and_b32 v22, 1, v21 // (op 1138) 

v_and_b32 v21, 15, v1 // (op 1138) 
v_lshl_add_u32 v23, v22, 4, v21 // (op 1138) 


v_lshl_add_u32 v21, s17, 5, v23 // (op 1138) 

v_mul_lo_u32 v22, v21, s32 // (op 1138) 

v_and_b32 v21, 63, v1 // (op 1138) 
v_lshrrev_b32 v23, 4, v21 // (op 1138) 

v_add_u32 v21, v23, v22 // (op 1138) 


s_mul_i32 s1, s32, s9 // (op 1140) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1140) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1140) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1140) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1140) most significant: sum
s_add_u32 s1, s1, s7 // (op 1140) most significant: sum


s_mov_b32 s60, s34 // (op 1140) reg expression
s_mov_b32 s61, s35 // (op 1140) reg expression
s_mov_b32 s62, s0 // (op 1140) reg expression
s_mov_b32 s63, 131072 // (op 1140) call()




buffer_load_ubyte v22, v20, s[60:63], 0 offen offset: 0 // (op 774) Load value
v_and_b32 v23, -8, v1 // (op 1262) 
v_sub_u32 v24, v1, v23 // (op 1262) 

v_lshlrev_b32 v23, 4, v24 // (op 1262) 

v_lshrrev_b32 v24, 3, v1 // (op 1262) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 1262) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 1262) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 882) reg expression
buffer_load_dwordx4 v14, s[52:55], 0 offen offset: 0 lds // (op 882) Load value direct to lds




buffer_load_dwordx4 v[24:27], v12, s[48:51], 0 offen // (op 781) Load value
buffer_load_dwordx2 v[28:29], v12, s[48:51], 0 offen offset:16 // (op 781) Load value



buffer_load_ubyte v23, v18, s[56:59], 0 offen offset: 0 // (op 784) Load value



buffer_load_ubyte v30, v20, s[60:63], 0 offen offset:4 // (op 788) Load value

v_and_b32 v20, -8, v1 // (op 1406) 
v_sub_u32 v31, v1, v20 // (op 1406) 

v_lshlrev_b32 v20, 4, v31 // (op 1406) 

v_lshrrev_b32 v31, 3, v1 // (op 1406) 
v_lshl_add_u32 v32, v31, 7, v20 // (op 1406) 


s_nop 0
v_readfirstlane_b32 s0, v32 // (op 1406) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 887) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 887) reg expression
s_mov_b32 s6, 128 // (op 887) call()
buffer_load_dwordx4 v14, s[52:55], s6 offen offset: 0 lds // (op 887) Load value direct to lds







buffer_load_dwordx4 v[32:35], v12, s[48:51], 0 offen offset:96 // (op 795) Load value
buffer_load_dwordx2 v[36:37], v12, s[48:51], 0 offen offset:112 // (op 795) Load value




buffer_load_ubyte v12, v18, s[56:59], 0 offen offset:4 // (op 798) Load value

v_and_b32 v18, -4, v1 // (op 1205) 
v_sub_u32 v20, v1, v18 // (op 1205) 

v_lshlrev_b32 v18, 5, v20 // (op 1205) 

v_lshrrev_b32 v20, 2, v1 // (op 1205) 
v_lshl_add_u32 v31, v20, 7, v18 // (op 1205) 


v_mul_lo_u32 v18, v31, 6 // (op 1205) 

v_lshrrev_b32 v20, 3, v18 // (op 1205) 




s_waitcnt vmcnt(7)
ds_write_b128 v20, v[24:27] offset:8192 // (op 801) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v20, v[28:29] offset:8208 // (op 801) Store local data 

v_and_b32 v18, -4, v1 // (op 1315) 
v_sub_u32 v20, v1, v18 // (op 1315) 

v_and_b32 v18, 4294967292, v1 // (op 1315) 
v_add_u32 v31, v18, v20 // (op 1315) 





s_waitcnt vmcnt(5)
ds_write_b8 v31, v23 offset:14336 // (op 802) Store local data 

v_and_b32 v18, -4, v1 // (op 1176) 
v_sub_u32 v20, v1, v18 // (op 1176) 

v_lshlrev_b32 v18, 5, v20 // (op 1176) 

v_lshrrev_b32 v20, 2, v1 // (op 1176) 
v_lshl_add_u32 v31, v20, 7, v18 // (op 1176) 


v_mul_lo_u32 v18, v31, 6 // (op 1176) 

v_lshrrev_b32 v20, 3, v18 // (op 1176) 

v_and_b32 v18, -8, v1 // (op 1234) 
v_sub_u32 v31, v1, v18 // (op 1234) 

v_lshlrev_b32 v18, 4, v31 // (op 1234) 

v_lshrrev_b32 v31, 3, v1 // (op 1234) 
v_lshl_add_u32 v14, v31, 7, v18 // (op 1234) 


s_nop 0
v_readfirstlane_b32 s0, v14 // (op 1234) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -4, v1 // (op 1286) 
v_sub_u32 v31, v1, v18 // (op 1286) 

v_and_b32 v18, 4294967292, v1 // (op 1286) 
v_add_u32 v14, v18, v31 // (op 1286) 


v_and_b32 v15, -4, v1 // (op 1362) 
v_sub_u32 v18, v1, v15 // (op 1362) 

v_lshlrev_b32 v15, 5, v18 // (op 1362) 

v_lshrrev_b32 v18, 2, v1 // (op 1362) 
v_lshl_add_u32 v31, v18, 7, v15 // (op 1362) 


v_mul_lo_u32 v15, v31, 6 // (op 1362) 

v_lshrrev_b32 v18, 3, v15 // (op 1362) 

v_and_b32 v15, -8, v1 // (op 1391) 
v_sub_u32 v31, v1, v15 // (op 1391) 

v_lshlrev_b32 v15, 4, v31 // (op 1391) 

v_lshrrev_b32 v31, 3, v1 // (op 1391) 
v_lshl_add_u32 v38, v31, 7, v15 // (op 1391) 


s_nop 0
v_readfirstlane_b32 s1, v38 // (op 1391) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -4, v1 // (op 1430) 
v_sub_u32 v31, v1, v15 // (op 1430) 

v_and_b32 v15, 4294967292, v1 // (op 1430) 
v_add_u32 v38, v15, v31 // (op 1430) 


s_mov_b32 s6, 0 // (op 311) call()
s_ashr_i64 s[10:11], s[36:37], 7 // (op 310) 
s_mov_b32 s7, s10 // (op 310) convert

s_and_b32 s10, s7, 4294967294 // (op 310) 

s_cmp_lt_u32 s6, s10 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_310 // (op 310) Condition: Top (jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_310 if false)
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoop_310:
s_add_u32 s6, s6, 2 // (op 312) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1488) 



ds_read_b128 v[40:43], v5 offset:8192 // (op 65) Load local data 
ds_read_b64 v[44:45], v5 offset:8208 // (op 65) Load local data 
s_barrier  // (op 1501) 



ds_read_b128 v[46:49], v3 // (op 70) Load local data 
ds_read_b128 v[50:53], v3 offset:64 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1514) 



ds_read_u8 v15, v0 offset:14336 // (op 75) Load local data 



ds_read_b128 v[54:57], v5 offset:11264 // (op 340) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:45], v[46:53], a[0:3], v15, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 82) 
ds_read_b64 v[58:59], v5 offset:11280 // (op 340) Load local data 





ds_read_u8 v15, v0 offset:14464 // (op 344) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[46:53], a[4:7], v15, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 346) 





v_mov_b32 v15, v21 // (op 80) 

buffer_load_ubyte v22, v15, s[60:63], 0 offen offset:8 // (op 80) Load value
s_barrier  // (op 1504) 


v_mov_b64 v[40:41], v[16:17] // (op 865) 

s_mov_b32 m0, s0 // (op 865) reg expression
s_mov_b32 s7, 256 // (op 865) call()
buffer_load_dwordx4 v40, s[52:55], s7 offen offset: 0 lds // (op 865) Load value direct to lds



v_mov_b32 v31, v13 // (op 4) 

buffer_load_dwordx4 v[24:27], v31, s[48:51], 0 offen offset:192 // (op 4) Load value
buffer_load_dwordx2 v[28:29], v31, s[48:51], 0 offen offset:208 // (op 4) Load value


v_mov_b32 v39, v19 // (op 17) 

buffer_load_ubyte v23, v39, s[56:59], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 1531) 



s_waitcnt vmcnt(7)
ds_write_b128 v18, v[32:35] offset:14592 // (op 225) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v18, v[36:37] offset:14608 // (op 225) Store local data 
s_barrier  // (op 1543) 



s_waitcnt vmcnt(5)
ds_write_b8 v38, v12 offset:20736 // (op 236) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1528) 



ds_read_b128 v[42:45], v8 offset:14592 // (op 227) Load local data 
ds_read_b64 v[46:47], v8 offset:14608 // (op 227) Load local data 
s_barrier  // (op 1534) 



ds_read_b128 v[48:51], v4 offset:4096 // (op 229) Load local data 
ds_read_b128 v[52:55], v4 offset:4160 // (op 229) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1540) 



ds_read_u8 v56, v7 offset:20736 // (op 231) Load local data 



ds_read_b128 v[58:61], v8 offset:17664 // (op 353) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:47], v[48:55], a[0:3], v56, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 234) 
ds_read_b64 v[62:63], v8 offset:17680 // (op 353) Load local data 





s_barrier  // (op 1604) 
ds_read_u8 v42, v7 offset:20864 // (op 357) Load local data 
s_barrier  // (op 1608) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:63], v[48:55], a[4:7], v42, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 347) 






buffer_load_ubyte v30, v15, s[60:63], 0 offen offset:12 // (op 233) Load value

s_barrier  // (op 1537) 
v_add_u32 v21, 8, v21 // (op 1130) 


s_add_u32 s7, 4096, s1 // (op 877) 

s_mov_b32 m0, s7 // (op 877) reg expression
s_mov_b32 s10, 384 // (op 877) call()
buffer_load_dwordx4 v40, s[52:55], s10 offen offset: 0 lds // (op 877) Load value direct to lds






v_add_co_u32 v16, vcc, 256, v16 // (op 1002) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1002) most significant half
buffer_load_dwordx4 v[32:35], v31, s[48:51], 0 offen offset:288 // (op 224) Load value
buffer_load_dwordx2 v[36:37], v31, s[48:51], 0 offen offset:304 // (op 224) Load value




v_add_u32 v13, 192, v13 // (op 938) 
buffer_load_ubyte v12, v39, s[56:59], 0 offen offset:12 // (op 238) Load value

s_barrier  // (op 1491) 
v_add_u32 v19, 8, v19 // (op 1066) 



s_waitcnt vmcnt(7)
ds_write_b128 v20, v[24:27] offset:8192 // (op 6) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v20, v[28:29] offset:8208 // (op 6) Store local data 
s_barrier  // (op 1517) 



s_waitcnt vmcnt(5)
ds_write_b8 v14, v23 offset:14336 // (op 21) Store local data 
s_ashr_i64 s[10:11], s[36:37], 7 // (op 310) 
s_mov_b32 s7, s10 // (op 310) convert

s_and_b32 s10, s7, 4294967294 // (op 310) 

s_cmp_lt_u32 s6, s10 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoop_310 // (op 310) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoop_310 if true)
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_310:














s_barrier  // (op 1588) 
v_lshrrev_b32 v31, 2, v1 // (op 900) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 900) 
v_lshl_add_u32 v13, s16, 6, v31 // (op 900) 

s_lshr_b32 s6, s0, 1 // (op 900) 

v_and_b32 v31, -4, v1 // (op 900) 
v_sub_u32 v14, v1, v31 // (op 900) 

v_lshlrev_b32 v31, 5, v14 // (op 900) 

v_lshl_add_u32 v14, s6, 8, v31 // (op 900) 


v_mov_b32 v29, 0 // (op 900) convert
v_mov_b32 v28, v14 // (op 900) convert

v_mul_lo_u32 v15, v13, s13 // (op 900) most significant: low of low * high
v_mul_hi_u32 v31, v13, s12 // (op 900) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 900) least significant: low of low * low
v_add_u32 v15, v15, v31 // (op 900) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v28 // (op 900) least significant half
v_addc_co_u32 v15, s[0:1], v15, v29, s[0:1] // (op 900) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 900) most significant: low of high * low
v_mul_hi_u32 v13, v14, 6 // (op 900) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 900) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 900) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 900) 

v_mov_b32 v13, v14 // (op 900) convert

s_mul_i32 s1, s23, 6 // (op 902) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 902) least significant
s_mul_hi_u32 s6, s22, 6 // (op 902) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 902) most significant: sum

s_mul_i32 s7, s12, s1 // (op 902) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 902) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 902) least significant
s_mul_hi_u32 s11, s12, s0 // (op 902) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 902) most significant: sum
s_add_u32 s7, s7, s11 // (op 902) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 902) 

s_mov_b32 s48, s20 // (op 902) reg expression
s_mov_b32 s49, s21 // (op 902) reg expression
s_mov_b32 s50, s0 // (op 902) reg expression
s_mov_b32 s51, 131072 // (op 902) call()

v_and_b32 v31, -8, v1 // (op 968) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 968) 
v_sub_u32 v14, v1, v31 // (op 968) 

s_lshr_b32 s6, s0, 1 // (op 968) 

v_lshlrev_b32 v31, 4, v14 // (op 968) 

v_lshl_add_u32 v14, s6, 8, v31 // (op 968) 


v_mov_b32 v29, 0 // (op 968) convert
v_mov_b32 v28, v14 // (op 968) convert

v_lshrrev_b32 v31, 3, v1 // (op 968) 
v_lshl_add_u32 v14, s17, 5, v31 // (op 968) 

v_mul_lo_u32 v27, v14, s25 // (op 968) most significant: low of low * high
v_mul_hi_u32 v31, v14, s24 // (op 968) most significant: high of low * low
v_mul_lo_u32 v26, v14, s24 // (op 968) least significant: low of low * low
v_add_u32 v27, v27, v31 // (op 968) most significant: sum


v_add_co_u32 v14, s[0:1], v28, v26 // (op 968) least significant half
v_addc_co_u32 v15, s[0:1], v29, v27, s[0:1] // (op 968) most significant half



s_mul_i32 s1, s24, s9 // (op 970) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 970) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 970) least significant
s_mul_hi_u32 s7, s24, s8 // (op 970) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 970) most significant: sum
s_add_u32 s1, s1, s7 // (op 970) most significant: sum


s_mov_b32 s52, s26 // (op 970) reg expression
s_mov_b32 s53, s27 // (op 970) reg expression
s_mov_b32 s54, s0 // (op 970) reg expression
s_mov_b32 s55, 131072 // (op 970) call()

v_lshrrev_b32 v31, 2, v1 // (op 1032) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1032) 
v_lshl_add_u32 v16, s16, 6, v31 // (op 1032) 

s_lshr_b32 s6, s0, 1 // (op 1032) 

v_and_b32 v31, -4, v1 // (op 1032) 
v_sub_u32 v17, v1, v31 // (op 1032) 

v_lshl_add_u32 v31, s6, 3, v17 // (op 1032) 


v_mul_lo_u32 v17, v16, s28 // (op 1032) 
v_add_u32 v17, v17, v31 // (op 1032) 


s_mul_i32 s1, s28, s23 // (op 1034) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1034) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1034) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1034) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1034) most significant: sum
s_add_u32 s1, s1, s7 // (op 1034) most significant: sum


s_mov_b32 s56, s30 // (op 1034) reg expression
s_mov_b32 s57, s31 // (op 1034) reg expression
s_mov_b32 s58, s0 // (op 1034) reg expression
s_mov_b32 s59, 131072 // (op 1034) call()

v_and_b32 v16, 63, v1 // (op 1096) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1096) 
v_lshrrev_b32 v31, 4, v16 // (op 1096) 

s_lshr_b32 s6, s0, 1 // (op 1096) 

v_lshrrev_b32 v16, 6, v1 // (op 1096) 
v_lshl_add_u32 v18, s6, 3, v31 // (op 1096) 


v_and_b32 v31, 1, v16 // (op 1096) 

v_and_b32 v16, 15, v1 // (op 1096) 
v_lshl_add_u32 v19, v31, 4, v16 // (op 1096) 


v_lshl_add_u32 v16, s17, 5, v19 // (op 1096) 

v_mul_lo_u32 v31, v16, s32 // (op 1096) 

v_add_u32 v16, v18, v31 // (op 1096) 


s_mul_i32 s1, s32, s9 // (op 1098) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1098) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1098) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1098) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1098) most significant: sum
s_add_u32 s1, s1, s7 // (op 1098) most significant: sum


s_mov_b32 s60, s34 // (op 1098) reg expression
s_mov_b32 s61, s35 // (op 1098) reg expression
s_mov_b32 s62, s0 // (op 1098) reg expression
s_mov_b32 s63, 131072 // (op 1098) call()

v_and_b32 v31, -4, v1 // (op 1189) 
v_sub_u32 v18, v1, v31 // (op 1189) 

v_lshlrev_b32 v31, 5, v18 // (op 1189) 

v_lshrrev_b32 v18, 2, v1 // (op 1189) 
v_lshl_add_u32 v19, v18, 7, v31 // (op 1189) 


v_mul_lo_u32 v18, v19, 6 // (op 1189) 

v_lshrrev_b32 v31, 3, v18 // (op 1189) 

v_and_b32 v18, -8, v1 // (op 1247) 
v_sub_u32 v19, v1, v18 // (op 1247) 

v_lshlrev_b32 v18, 4, v19 // (op 1247) 

v_lshrrev_b32 v19, 3, v1 // (op 1247) 
v_lshl_add_u32 v20, v19, 7, v18 // (op 1247) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1247) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -4, v1 // (op 1299) 
v_sub_u32 v19, v1, v18 // (op 1299) 

v_and_b32 v18, 4294967292, v1 // (op 1299) 
v_add_u32 v20, v18, v19 // (op 1299) 


s_ashr_i64 s[6:7], s[36:37], 7 // (op 316) 
s_and_b32 s1, s6, 4294967294 // (op 316) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 315) 
s_mov_b32 s10, s6 // (op 315) convert

s_cmp_lt_u32 s1, s10 // (op 315) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_315 // (op 315) Condition: Top (jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_315 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_315:


s_add_u32 s1, s1, 1 // (op 317) 
v_mov_b32 v12, v13 // (op 166) 

buffer_load_dwordx4 v[22:25], v12, s[48:51], 0 offen // (op 166) Load value
buffer_load_dwordx2 v[26:27], v12, s[48:51], 0 offen offset:16 // (op 166) Load value

v_add_u32 v13, 96, v13 // (op 890) 
s_barrier  // (op 1494) 
s_barrier  // (op 1507) 





v_mov_b64 v[18:19], v[14:15] // (op 870) 

s_mov_b32 m0, s0 // (op 870) reg expression
buffer_load_dwordx4 v18, s[52:55], 0 offen offset: 0 lds // (op 870) Load value direct to lds

v_add_co_u32 v14, vcc, 128, v14 // (op 960) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 960) most significant half
s_waitcnt vmcnt(2)
ds_write_b128 v31, v[22:25] offset:8192 // (op 167) Store local data 


v_mov_b32 v12, v17 // (op 180) 

buffer_load_ubyte v21, v12, s[56:59], 0 offen offset: 0 // (op 180) Load value

v_add_u32 v17, 4, v17 // (op 1024) 
s_waitcnt vmcnt(2)
ds_write_b64 v31, v[26:27] offset:8208 // (op 167) Store local data 

s_barrier  // (op 1520) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1497) 






ds_read_b128 v[22:25], v5 offset:8192 // (op 169) Load local data 
ds_read_b64 v[26:27], v5 offset:8208 // (op 169) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v21 offset:14336 // (op 178) Store local data 

s_barrier  // (op 1510) 



ds_read_b128 v[32:35], v3 // (op 171) Load local data 
ds_read_b128 v[36:39], v3 offset:64 // (op 171) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1524) 



ds_read_u8 v12, v0 offset:14336 // (op 173) Load local data 


v_mov_b32 v21, v16 // (op 175) 

buffer_load_ubyte v18, v21, s[60:63], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v16, 4, v16 // (op 1088) 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:27], v[32:39], a[0:3], v12, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 176) 
ds_read_b128 v[40:43], v5 offset:11264 // (op 321) Load local data 


ds_read_b64 v[44:45], v5 offset:11280 // (op 321) Load local data 



ds_read_u8 v12, v0 offset:14464 // (op 325) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:45], v[32:39], a[4:7], v12, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 327) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 315) 
s_mov_b32 s10, s6 // (op 315) convert

s_cmp_lt_u32 s1, s10 // (op 315) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_315 // (op 315) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_315 if true)
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_315:






















s_barrier  // (op 1592) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_123:



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
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
.LRR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_end:
.size RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2, .LRR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_end-RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: B8
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
  - .name: RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
    .symbol: RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 64
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
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

