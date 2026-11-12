// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.p2align 8
.type RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2,@function
RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_exec_begin:


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

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


v_lshrrev_b32 v0, 7, v1 // (op 1214) 
v_and_b32 v3, 15, v1 // (op 1214) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1214) 


v_and_b32 v0, 48, v1 // (op 1214) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 1214) 


v_lshrrev_b32 v0, 7, v1 // (op 1387) 
v_and_b32 v4, 15, v1 // (op 1387) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1387) 


v_and_b32 v0, 48, v1 // (op 1387) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 1387) 


v_lshrrev_b32 v0, 7, v1 // (op 1158) 
v_and_b32 v5, 63, v1 // (op 1158) 
v_lshrrev_b32 v6, 4, v5 // (op 1158) 
v_lshlrev_b32 v7, 2, v6 // (op 1158) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1158) 


v_lshl_add_u32 v0, s16, 4, v6 // (op 1158) 

v_lshrrev_b32 v6, 2, v5 // (op 1158) 

v_and_b32 v5, 3, v6 // (op 1158) 

v_and_b32 v6, 3, v1 // (op 1158) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1158) 


v_lshrrev_b32 v5, 6, v1 // (op 1158) 
v_and_b32 v6, 1, v5 // (op 1158) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1158) 


v_lshl_add_u32 v6, s17, 6, v5 // (op 1158) 

v_mul_lo_u32 v5, v6, s2 // (op 1158) 

v_add_lshl_u32 v6, v0, v5, 1 // (op 1158) 


v_and_b32 v0, 63, v1 // (op 1272) 
v_and_b32 v5, 48, v1 // (op 1272) 
v_sub_u32 v7, v0, v5 // (op 1272) 

v_lshrrev_b32 v5, 4, v0 // (op 1272) 

v_lshl_add_u32 v0, v5, 5, v7 // (op 1272) 


v_lshrrev_b32 v5, 6, v1 // (op 1272) 
v_and_b32 v7, 1, v5 // (op 1272) 

v_lshlrev_b32 v5, 4, v7 // (op 1272) 

v_lshl_add_u32 v7, v0, 6, v5 // (op 1272) 


v_mul_lo_u32 v0, v7, 6 // (op 1272) 
v_lshrrev_b32 v5, 3, v0 // (op 1272) 

v_lshrrev_b32 v0, 4, v7 // (op 1272) 

v_lshlrev_b32 v7, 2, v0 // (op 1272) 

v_add_u32 v0, v5, v7 // (op 1272) 


v_lshrrev_b32 v5, 6, v1 // (op 1330) 
v_and_b32 v7, 1, v5 // (op 1330) 

v_and_b32 v5, 15, v1 // (op 1330) 
s_mul_i32 s1, s2, s9 // (op 1160) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1160) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1160) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1160) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1160) most significant: sum
s_add_u32 s1, s1, s19 // (op 1160) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1160) 

v_lshl_add_u32 v8, v7, 4, v5 // (op 1330) 


v_and_b32 v5, 63, v1 // (op 1330) 
v_lshrrev_b32 v7, 4, v5 // (op 1330) 

v_lshl_add_u32 v5, v7, 6, v8 // (op 1330) 


v_lshrrev_b32 v7, 7, v1 // (op 1346) 
v_and_b32 v8, 63, v1 // (op 1346) 
v_lshrrev_b32 v9, 4, v8 // (op 1346) 
v_lshlrev_b32 v10, 2, v9 // (op 1346) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1346) 


v_lshl_add_u32 v7, s16, 4, v9 // (op 1346) 

v_lshrrev_b32 v9, 2, v8 // (op 1346) 

v_and_b32 v8, 3, v9 // (op 1346) 

v_and_b32 v9, 3, v1 // (op 1346) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1346) 


v_lshrrev_b32 v8, 6, v1 // (op 1346) 
v_and_b32 v9, 1, v8 // (op 1346) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 1346) 


v_lshl_add_u32 v9, s17, 6, v8 // (op 1346) 

v_mul_lo_u32 v8, v9, s4 // (op 1346) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 1346) 


v_and_b32 v7, 63, v1 // (op 1416) 
v_and_b32 v8, 48, v1 // (op 1416) 
v_sub_u32 v10, v7, v8 // (op 1416) 

v_lshrrev_b32 v8, 4, v7 // (op 1416) 

v_lshl_add_u32 v7, v8, 5, v10 // (op 1416) 


v_lshrrev_b32 v8, 6, v1 // (op 1416) 
v_and_b32 v10, 1, v8 // (op 1416) 

v_lshlrev_b32 v8, 4, v10 // (op 1416) 

v_lshl_add_u32 v10, v7, 6, v8 // (op 1416) 


v_mul_lo_u32 v7, v10, 6 // (op 1416) 
v_lshrrev_b32 v8, 3, v7 // (op 1416) 

v_lshrrev_b32 v7, 4, v10 // (op 1416) 

v_lshlrev_b32 v10, 2, v7 // (op 1416) 

v_add_u32 v7, v8, v10 // (op 1416) 


v_lshrrev_b32 v8, 6, v1 // (op 1445) 
v_and_b32 v10, 1, v8 // (op 1445) 

v_and_b32 v8, 15, v1 // (op 1445) 
s_mul_i32 s1, s4, s9 // (op 1348) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v8 // (op 1445) 


v_and_b32 v8, 63, v1 // (op 1445) 
v_lshrrev_b32 v10, 4, v8 // (op 1445) 

v_lshl_add_u32 v8, v10, 6, v11 // (op 1445) 


s_mul_i32 s40, s5, s8 // (op 1348) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1348) least significant
s_mul_hi_u32 s41, s4, s8 // (op 1348) most significant: high of low * low
s_add_u32 s1, s1, s40 // (op 1348) most significant: sum
s_add_u32 s1, s1, s41 // (op 1348) most significant: sum


s_lshl_b64 s[40:41], s[0:1], 1 // (op 1348) 

s_mov_b32 s44, s6 // (op 1160) reg expression
s_mov_b32 s45, s7 // (op 1160) reg expression
s_mov_b32 s46, s18 // (op 1160) reg expression
s_mov_b32 s47, 131072 // (op 1160) call()

s_mov_b32 s48, s10 // (op 1348) reg expression

s_mov_b32 s49, s11 // (op 1348) reg expression
s_mov_b32 s50, s40 // (op 1348) reg expression
s_mov_b32 s51, 131072 // (op 1348) call()



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
v_lshrrev_b32 v12, 3, v1 // (op 925) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 925) 

v_and_b32 v12, -8, v1 // (op 925) 
v_sub_u32 v14, v1, v12 // (op 925) 

v_lshlrev_b32 v12, 4, v14 // (op 925) 

v_mov_b32 v15, 0 // (op 925) convert
v_mov_b32 v14, v12 // (op 925) convert

v_mul_lo_u32 v17, v13, s13 // (op 925) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 925) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 925) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 925) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 925) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 925) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 927) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 927) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 927) least significant
s_mul_hi_u32 s7, s12, s22 // (op 927) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 927) most significant: sum
s_add_u32 s1, s1, s7 // (op 927) most significant: sum


s_mov_b32 s8, s20 // (op 927) reg expression
s_mov_b32 s9, s21 // (op 927) reg expression
s_mov_b32 s10, s0 // (op 927) reg expression
s_mov_b32 s11, 131072 // (op 927) call()

v_lshrrev_b32 v12, 3, v1 // (op 944) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 944) 

v_and_b32 v12, -8, v1 // (op 944) 
v_sub_u32 v14, v1, v12 // (op 944) 

v_lshlrev_b32 v12, 4, v14 // (op 944) 

v_mov_b32 v15, 0 // (op 944) convert
v_mov_b32 v14, v12 // (op 944) convert

v_mul_lo_u32 v19, v13, s13 // (op 944) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 944) most significant: high of low * low
v_mul_lo_u32 v18, v13, s12 // (op 944) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 944) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 944) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 944) most significant half



s_mul_i32 s1, s12, s23 // (op 946) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 946) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 946) least significant
s_mul_hi_u32 s7, s12, s22 // (op 946) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 946) most significant: sum
s_add_u32 s1, s1, s7 // (op 946) most significant: sum


s_mov_b32 s8, s20 // (op 946) reg expression
s_mov_b32 s9, s21 // (op 946) reg expression
s_mov_b32 s10, s0 // (op 946) reg expression
s_mov_b32 s11, 131072 // (op 946) call()

s_lshl_b32 s0, s17, 6 // (op 989) 
s_mov_b32 s7, 0 // (op 989) convert
s_mov_b32 s6, s0 // (op 989) convert

v_mul_lo_u32 v13, v1, s25 // (op 989) most significant: low of low * high
v_mul_hi_u32 v14, v1, s24 // (op 989) most significant: high of low * low
v_mul_lo_u32 v12, v1, s24 // (op 989) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 989) most significant: sum

v_mov_b32 v14, s6 // (op 989) 
v_mov_b32 v15, s7 // (op 989) 
v_add_co_u32 v12, s[0:1], v12, v14 // (op 989) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 989) most significant half




v_mul_lo_u32 v15, v13, 6 // (op 989) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 989) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 989) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 989) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 989) 

v_mov_b32 v14, v12 // (op 989) convert

s_mul_i32 s1, s29, 6 // (op 991) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 991) least significant
s_mul_hi_u32 s6, s28, 6 // (op 991) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 991) most significant: sum

s_mul_i32 s7, s24, s1 // (op 991) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 991) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 991) least significant
s_mul_hi_u32 s15, s24, s0 // (op 991) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 991) most significant: sum
s_add_u32 s7, s7, s15 // (op 991) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 991) 

s_mov_b32 s40, s26 // (op 991) reg expression
s_mov_b32 s41, s27 // (op 991) reg expression
s_mov_b32 s42, s0 // (op 991) reg expression
s_mov_b32 s43, 131072 // (op 991) call()

s_lshl_b32 s0, s17, 6 // (op 1008) 
s_mov_b32 s7, 0 // (op 1008) convert
s_mov_b32 s6, s0 // (op 1008) convert

v_mul_lo_u32 v13, v1, s25 // (op 1008) most significant: low of low * high
v_mul_hi_u32 v15, v1, s24 // (op 1008) most significant: high of low * low
v_mul_lo_u32 v12, v1, s24 // (op 1008) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 1008) most significant: sum

v_mov_b32 v15, s6 // (op 1008) 
v_mov_b32 v20, s7 // (op 1008) 
v_add_co_u32 v12, s[0:1], v12, v15 // (op 1008) least significant half
v_addc_co_u32 v13, s[0:1], v13, v20, s[0:1] // (op 1008) most significant half




v_mul_lo_u32 v21, v13, 6 // (op 1008) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 1008) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 1008) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1008) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 1008) 

v_mov_b32 v15, v12 // (op 1008) convert

s_mul_i32 s1, s29, 6 // (op 1010) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1010) least significant
s_mul_hi_u32 s6, s28, 6 // (op 1010) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1010) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1010) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1010) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1010) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1010) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1010) most significant: sum
s_add_u32 s7, s7, s15 // (op 1010) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1010) 

s_mov_b32 s40, s26 // (op 1010) reg expression
s_mov_b32 s41, s27 // (op 1010) reg expression
s_mov_b32 s42, s0 // (op 1010) reg expression
s_mov_b32 s43, 131072 // (op 1010) call()

v_lshrrev_b32 v12, 7, v1 // (op 1053) 
v_and_b32 v13, 15, v1 // (op 1053) 
v_lshl_add_u32 v20, v12, 4, v13 // (op 1053) 


v_lshl_add_u32 v12, s16, 4, v20 // (op 1053) 

v_and_b32 v13, 63, v1 // (op 1053) 
v_lshrrev_b32 v20, 4, v13 // (op 1053) 

v_mul_lo_u32 v13, v12, s30 // (op 1053) 
v_add_u32 v13, v13, v20 // (op 1053) 


s_mul_i32 s1, s30, s23 // (op 1055) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1055) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1055) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1055) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1055) most significant: sum
s_add_u32 s1, s1, s7 // (op 1055) most significant: sum


s_mov_b32 s52, s32 // (op 1055) reg expression
s_mov_b32 s53, s33 // (op 1055) reg expression
s_mov_b32 s54, s0 // (op 1055) reg expression
s_mov_b32 s55, 131072 // (op 1055) call()

v_lshrrev_b32 v12, 7, v1 // (op 1072) 
v_and_b32 v20, 15, v1 // (op 1072) 
v_lshl_add_u32 v21, v12, 4, v20 // (op 1072) 


v_lshl_add_u32 v12, s16, 4, v21 // (op 1072) 

v_and_b32 v20, 63, v1 // (op 1072) 
v_lshrrev_b32 v21, 4, v20 // (op 1072) 

v_mul_lo_u32 v20, v12, s30 // (op 1072) 
v_add_u32 v20, v20, v21 // (op 1072) 


s_mul_i32 s1, s30, s23 // (op 1074) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1074) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1074) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1074) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1074) most significant: sum
s_add_u32 s1, s1, s7 // (op 1074) most significant: sum


s_mov_b32 s52, s32 // (op 1074) reg expression
s_mov_b32 s53, s33 // (op 1074) reg expression
s_mov_b32 s54, s0 // (op 1074) reg expression
s_mov_b32 s55, 131072 // (op 1074) call()

v_and_b32 v12, -64, v1 // (op 1117) 
v_sub_u32 v21, v1, v12 // (op 1117) 

v_lshl_add_u32 v12, s17, 6, v21 // (op 1117) 

v_lshrrev_b32 v21, 6, v1 // (op 1117) 
v_lshlrev_b32 v22, 1, v21 // (op 1117) 

v_mul_lo_u32 v21, v22, s34 // (op 1117) 
v_add_u32 v21, v21, v12 // (op 1117) 


s_mul_i32 s1, s34, s39 // (op 1119) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1119) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1119) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1119) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1119) most significant: sum
s_add_u32 s1, s1, s7 // (op 1119) most significant: sum


s_mov_b32 s56, s36 // (op 1119) reg expression
s_mov_b32 s57, s37 // (op 1119) reg expression
s_mov_b32 s58, s0 // (op 1119) reg expression
s_mov_b32 s59, 131072 // (op 1119) call()

v_and_b32 v12, -64, v1 // (op 1136) 
v_sub_u32 v22, v1, v12 // (op 1136) 

v_lshl_add_u32 v12, s17, 6, v22 // (op 1136) 

v_lshrrev_b32 v22, 6, v1 // (op 1136) 
v_lshlrev_b32 v23, 1, v22 // (op 1136) 

v_mul_lo_u32 v22, v23, s34 // (op 1136) 
v_add_u32 v22, v22, v12 // (op 1136) 


s_mul_i32 s1, s34, s39 // (op 1138) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1138) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1138) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1138) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1138) most significant: sum
s_add_u32 s1, s1, s7 // (op 1138) most significant: sum


s_mov_b32 s56, s36 // (op 1138) reg expression
s_mov_b32 s57, s37 // (op 1138) reg expression
s_mov_b32 s58, s0 // (op 1138) reg expression
s_mov_b32 s59, 131072 // (op 1138) call()




buffer_load_ubyte v12, v13, s[52:55], 0 offen offset: 0 // (op 769) Load value


v_mov_b32 v23, v14 // (op 773) reg expression

buffer_load_dwordx3 v[24:26], v23, s[40:43], 0 offen // (op 773) Load value
buffer_load_dwordx3 v[28:30], v23, s[40:43], 0 offen offset:12 // (op 773) Load value
buffer_load_dwordx3 v[32:34], v23, s[40:43], 0 offen offset:24 // (op 773) Load value
buffer_load_dwordx3 v[36:38], v23, s[40:43], 0 offen offset:36 // (op 773) Load value


v_mov_b64 v[40:41], s[34:35] // (op 776) convert

v_mov_b32 v23, v21 // (op 776) reg expression
buffer_load_ubyte v42, v23, s[56:59], 0 offen offset: 0 // (op 776) Load value
v_add_u32 v23, v23, v40 // (op 776) 
buffer_load_ubyte v43, v23, s[56:59], 0 offen offset: 0 // (op 776) Load value


v_and_b32 v23, -8, v1 // (op 1202) 
v_sub_u32 v27, v1, v23 // (op 1202) 

v_lshlrev_b32 v23, 4, v27 // (op 1202) 

v_lshrrev_b32 v27, 3, v1 // (op 1202) 
v_lshl_add_u32 v31, v27, 7, v23 // (op 1202) 


s_nop 0
v_readfirstlane_b32 s0, v31 // (op 1202) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 880) reg expression
buffer_load_dwordx4 v16, s[8:11], 0 offen offset: 0 lds // (op 880) Load value direct to lds




buffer_load_ubyte v23, v13, s[52:55], 0 offen offset:4 // (op 783) Load value



s_lshl_b64 s[0:1], s[24:25], 7 // (op 787) 
s_mul_i32 s7, s1, 6 // (op 787) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 787) least significant
s_mul_hi_u32 s14, s0, 6 // (op 787) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 787) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 787) 

v_add_u32 v13, v14, s0 // (op 787) 


buffer_load_dwordx3 v[44:46], v13, s[40:43], 0 offen // (op 787) Load value
buffer_load_dwordx3 v[48:50], v13, s[40:43], 0 offen offset:12 // (op 787) Load value
buffer_load_dwordx3 v[52:54], v13, s[40:43], 0 offen offset:24 // (op 787) Load value
buffer_load_dwordx3 v[56:58], v13, s[40:43], 0 offen offset:36 // (op 787) Load value



v_mov_b64 v[40:41], s[34:35] // (op 790) convert

s_lshl_b32 s0, s34, 2 // (op 790) 
v_add_u32 v27, v21, s0 // (op 790) 

buffer_load_ubyte v60, v27, s[56:59], 0 offen offset: 0 // (op 790) Load value
v_add_u32 v27, v27, v40 // (op 790) 
buffer_load_ubyte v61, v27, s[56:59], 0 offen offset: 0 // (op 790) Load value



v_and_b32 v21, -8, v1 // (op 1375) 
v_sub_u32 v27, v1, v21 // (op 1375) 

v_lshlrev_b32 v21, 4, v27 // (op 1375) 

v_lshrrev_b32 v27, 3, v1 // (op 1375) 
v_lshl_add_u32 v31, v27, 7, v21 // (op 1375) 


s_nop 0
v_readfirstlane_b32 s0, v31 // (op 1375) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 2048, s0 // (op 885) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 885) reg expression
s_mov_b32 s6, 128 // (op 885) call()
buffer_load_dwordx4 v16, s[8:11], s6 offen offset: 0 lds // (op 885) Load value direct to lds




v_lshlrev_b32 v21, 6, v1 // (op 1255) 
v_mul_lo_u32 v27, v21, 6 // (op 1255) 
v_lshrrev_b32 v31, 3, v27 // (op 1255) 

v_lshrrev_b32 v27, 4, v21 // (op 1255) 

v_lshlrev_b32 v21, 2, v27 // (op 1255) 

v_add_u32 v27, v31, v21 // (op 1255) 





s_waitcnt vmcnt(14)
ds_write_b96 v27, v[24:26] offset:4096 // (op 795) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v27, v[28:30] offset:4112 // (op 795) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v27, v[32:34] offset:4128 // (op 795) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v27, v[36:38] offset:4144 // (op 795) Store local data 

v_and_b32 v21, -64, v1 // (op 1313) 
v_sub_u32 v27, v1, v21 // (op 1313) 

v_lshrrev_b32 v21, 6, v1 // (op 1313) 
v_lshl_add_u32 v31, v21, 7, v27 // (op 1313) 





s_waitcnt vmcnt(10)
ds_write_b8 v31, v42 offset:12288 // (op 796) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v31, v43 offset:12352 // (op 796) Store local data 

v_and_b32 v21, -8, v1 // (op 1174) 
v_sub_u32 v27, v1, v21 // (op 1174) 

v_lshlrev_b32 v21, 4, v27 // (op 1174) 

v_lshrrev_b32 v27, 3, v1 // (op 1174) 
v_lshl_add_u32 v31, v27, 7, v21 // (op 1174) 


s_nop 0
v_readfirstlane_b32 s0, v31 // (op 1174) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v21, 6, v1 // (op 1226) 
v_mul_lo_u32 v27, v21, 6 // (op 1226) 
v_lshrrev_b32 v31, 3, v27 // (op 1226) 

v_lshrrev_b32 v27, 4, v21 // (op 1226) 

v_lshlrev_b32 v21, 2, v27 // (op 1226) 

v_add_u32 v27, v31, v21 // (op 1226) 


v_and_b32 v21, -64, v1 // (op 1284) 
v_sub_u32 v31, v1, v21 // (op 1284) 

v_lshrrev_b32 v21, 6, v1 // (op 1284) 
v_lshl_add_u32 v35, v21, 7, v31 // (op 1284) 


v_and_b32 v21, -8, v1 // (op 1360) 
v_sub_u32 v31, v1, v21 // (op 1360) 

v_lshlrev_b32 v21, 4, v31 // (op 1360) 

v_lshrrev_b32 v31, 3, v1 // (op 1360) 
v_lshl_add_u32 v47, v31, 7, v21 // (op 1360) 


s_nop 0
v_readfirstlane_b32 s1, v47 // (op 1360) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v21, 6, v1 // (op 1399) 
v_mul_lo_u32 v31, v21, 6 // (op 1399) 
v_lshrrev_b32 v47, 3, v31 // (op 1399) 

v_lshrrev_b32 v31, 4, v21 // (op 1399) 

v_lshlrev_b32 v21, 2, v31 // (op 1399) 

v_add_u32 v31, v47, v21 // (op 1399) 


v_and_b32 v21, -64, v1 // (op 1428) 
v_sub_u32 v47, v1, v21 // (op 1428) 

v_lshrrev_b32 v21, 6, v1 // (op 1428) 
v_lshl_add_u32 v51, v21, 7, v47 // (op 1428) 


s_mov_b32 s6, 0 // (op 312) call()
s_ashr_i64 s[14:15], s[28:29], 7 // (op 311) 
s_mov_b32 s7, s14 // (op 311) convert

s_and_b32 s14, s7, 4294967294 // (op 311) 

s_cmp_lt_u32 s6, s14 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_311 // (op 311) Condition: Top (jump to RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_311 if false)
RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_311:
s_add_u32 s6, s6, 2 // (op 313) 
s_waitcnt vmcnt(8)
s_barrier  // (op 1486) 



ds_read_b128 v[62:65], v3 // (op 67) Load local data 
ds_read_b128 v[66:69], v3 offset:64 // (op 67) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1499) 



ds_read_b96_tr_b6 v[70:72], v0 offset:4096 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[73:75], v0 offset:5120 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1513) 


ds_read_u8 v21, v5 offset:12288 // (op 81) Load local data 



ds_read_b96_tr_b6 v[76:78], v0 offset:4128 // (op 343) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[70:75], a[0:3], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 84) 
ds_read_b96_tr_b6 v[79:81], v0 offset:5152 // (op 343) Transpose load local data 




ds_read_u8 v21, v5 offset:12320 // (op 346) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[76:81], a[4:7], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 347) 





v_mov_b32 v21, v20 // (op 77) 

buffer_load_ubyte v12, v21, s[52:55], 0 offen offset:8 // (op 77) Load value


v_mov_b32 v47, v15 // (op 10) 
s_lshl_b64 s[14:15], s[24:25], 7 // (op 10) 
s_mul_i32 s19, s15, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 10) least significant
s_mul_hi_u32 s7, s14, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 10) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s7, s14, 1 // (op 10) 

v_add_u32 v55, v47, s7 // (op 10) 


buffer_load_dwordx3 v[24:26], v55, s[40:43], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[28:30], v55, s[40:43], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[32:34], v55, s[40:43], 0 offen offset:24 // (op 10) Load value
buffer_load_dwordx3 v[36:38], v55, s[40:43], 0 offen offset:36 // (op 10) Load value


v_mov_b64 v[16:17], s[34:35] // (op 21) convert

v_mov_b32 v55, v22 // (op 21) 
s_lshl_b32 s7, s34, 3 // (op 21) 
v_add_u32 v59, v55, s7 // (op 21) 

buffer_load_ubyte v42, v59, s[56:59], 0 offen offset: 0 // (op 21) Load value
v_add_u32 v59, v59, v16 // (op 21) 
buffer_load_ubyte v43, v59, s[56:59], 0 offen offset: 0 // (op 21) Load value


s_barrier  // (op 1489) 


v_mov_b64 v[16:17], v[18:19] // (op 864) 

s_mov_b32 m0, s0 // (op 864) reg expression
s_mov_b32 s7, 256 // (op 864) call()
buffer_load_dwordx4 v16, s[8:11], s7 offen offset: 0 lds // (op 864) Load value direct to lds

s_barrier  // (op 1536) 



s_waitcnt vmcnt(14)
ds_write_b96 v31, v[44:46] offset:12544 // (op 239) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v31, v[48:50] offset:12560 // (op 239) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v31, v[52:54] offset:12576 // (op 239) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v31, v[56:58] offset:12592 // (op 239) Store local data 
s_barrier  // (op 1542) 



s_waitcnt vmcnt(10)
ds_write_b8 v51, v60 offset:20736 // (op 240) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v51, v61 offset:20800 // (op 240) Store local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 1527) 



ds_read_b128 v[62:65], v4 offset:2048 // (op 231) Load local data 
ds_read_b128 v[66:69], v4 offset:2112 // (op 231) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1533) 



ds_read_b96_tr_b6 v[70:72], v7 offset:12544 // (op 233) Transpose load local data 
ds_read_b96_tr_b6 v[73:75], v7 offset:13568 // (op 233) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1539) 


ds_read_u8 v59, v8 offset:20736 // (op 237) Load local data 



ds_read_b96_tr_b6 v[76:78], v7 offset:12576 // (op 352) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[70:75], a[0:3], v23, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 238) 
ds_read_b96_tr_b6 v[79:81], v7 offset:13600 // (op 352) Transpose load local data 




s_barrier  // (op 1607) 
ds_read_u8 v59, v8 offset:20768 // (op 355) Load local data 
s_barrier  // (op 1611) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[76:81], a[4:7], v23, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 348) 






buffer_load_ubyte v23, v21, s[52:55], 0 offen offset:12 // (op 235) Load value



v_add_u32 v20, 8, v20 // (op 1064) 
s_lshl_b64 s[14:15], s[24:25], 7 // (op 241) 
s_mul_i32 s19, s15, 6 // (op 241) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 241) least significant
s_mul_hi_u32 s7, s14, 6 // (op 241) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 241) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 241) 

s_mul_i32 s7, 3, s14 // (op 241) 

v_add_u32 v21, v47, s7 // (op 241) 


buffer_load_dwordx3 v[44:46], v21, s[40:43], 0 offen // (op 241) Load value
buffer_load_dwordx3 v[48:50], v21, s[40:43], 0 offen offset:12 // (op 241) Load value
buffer_load_dwordx3 v[52:54], v21, s[40:43], 0 offen offset:24 // (op 241) Load value
buffer_load_dwordx3 v[56:58], v21, s[40:43], 0 offen offset:36 // (op 241) Load value



v_mov_b64 v[40:41], s[34:35] // (op 242) convert

s_lshl_b32 s7, s34, 2 // (op 242) 
s_mul_i32 s14, 3, s7 // (op 242) 

v_add_u32 v21, v55, s14 // (op 242) 

s_lshl_b64 s[14:15], s[24:25], 7 // (op 1000) 
s_mul_i32 s19, s15, 6 // (op 1000) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 1000) least significant
s_mul_hi_u32 s7, s14, 6 // (op 1000) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 1000) most significant: sum


buffer_load_ubyte v60, v21, s[56:59], 0 offen offset: 0 // (op 242) Load value
v_add_u32 v21, v21, v40 // (op 242) 
s_lshr_b64 s[14:15], s[18:19], 3 // (op 1000) 

s_lshl_b32 s7, s14, 1 // (op 1000) 

v_add_u32 v15, v15, s7 // (op 1000) 

buffer_load_ubyte v61, v21, s[56:59], 0 offen offset: 0 // (op 242) Load value



s_barrier  // (op 1530) 
s_lshl_b32 s7, s34, 3 // (op 1128) 


v_add_u32 v22, v22, s7 // (op 1128) 

s_add_u32 s7, 2048, s1 // (op 875) 

s_mov_b32 m0, s7 // (op 875) reg expression
s_mov_b32 s14, 384 // (op 875) call()
buffer_load_dwordx4 v16, s[8:11], s14 offen offset: 0 lds // (op 875) Load value direct to lds



s_barrier  // (op 1502) 
v_add_co_u32 v18, vcc, 256, v18 // (op 936) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 936) most significant half
s_waitcnt vmcnt(14)
ds_write_b96 v27, v[24:26] offset:4096 // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v27, v[28:30] offset:4112 // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v27, v[32:34] offset:4128 // (op 13) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v27, v[36:38] offset:4144 // (op 13) Store local data 
s_barrier  // (op 1516) 



s_waitcnt vmcnt(10)
ds_write_b8 v35, v42 offset:12288 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v35, v43 offset:12352 // (op 26) Store local data 
s_ashr_i64 s[14:15], s[28:29], 7 // (op 311) 
s_mov_b32 s7, s14 // (op 311) convert

s_and_b32 s14, s7, 4294967294 // (op 311) 

s_cmp_lt_u32 s6, s14 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_311 // (op 311) Condition: Bottom (jump to RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_311 if true)
RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_311:














s_barrier  // (op 1591) 
v_lshrrev_b32 v47, 3, v1 // (op 898) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 898) 
v_lshl_add_u32 v51, s16, 4, v47 // (op 898) 

s_lshr_b32 s6, s0, 1 // (op 898) 

v_and_b32 v47, -8, v1 // (op 898) 
v_sub_u32 v55, v1, v47 // (op 898) 

v_lshlrev_b32 v47, 4, v55 // (op 898) 

v_lshl_add_u32 v55, s6, 8, v47 // (op 898) 


v_mov_b32 v13, 0 // (op 898) convert
v_mov_b32 v12, v55 // (op 898) convert

v_mul_lo_u32 v15, v51, s13 // (op 898) most significant: low of low * high
v_mul_hi_u32 v47, v51, s12 // (op 898) most significant: high of low * low
v_mul_lo_u32 v14, v51, s12 // (op 898) least significant: low of low * low
v_add_u32 v15, v15, v47 // (op 898) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 898) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 898) most significant half



s_mul_i32 s1, s12, s23 // (op 900) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 900) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 900) least significant
s_mul_hi_u32 s7, s12, s22 // (op 900) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 900) most significant: sum
s_add_u32 s1, s1, s7 // (op 900) most significant: sum


s_mov_b32 s8, s20 // (op 900) reg expression
s_mov_b32 s9, s21 // (op 900) reg expression
s_mov_b32 s10, s0 // (op 900) reg expression
s_mov_b32 s11, 131072 // (op 900) call()

s_ashr_i64 s[0:1], s[28:29], 7 // (op 966) 
s_lshr_b32 s6, s0, 1 // (op 966) 

v_lshl_add_u32 v47, s6, 8, v1 // (op 966) 

s_lshl_b32 s0, s17, 6 // (op 966) 
s_mov_b32 s7, 0 // (op 966) convert
s_mov_b32 s6, s0 // (op 966) convert

v_mul_lo_u32 v13, v47, s25 // (op 966) most significant: low of low * high
v_mul_hi_u32 v51, v47, s24 // (op 966) most significant: high of low * low
v_mul_lo_u32 v12, v47, s24 // (op 966) least significant: low of low * low
v_add_u32 v13, v13, v51 // (op 966) most significant: sum

v_mov_b32 v51, s6 // (op 966) 
v_mov_b32 v55, s7 // (op 966) 
v_add_co_u32 v12, s[0:1], v12, v51 // (op 966) least significant half
v_addc_co_u32 v13, s[0:1], v13, v55, s[0:1] // (op 966) most significant half





v_mul_lo_u32 v17, v13, 6 // (op 966) most significant: low of high * low
v_mul_hi_u32 v47, v12, 6 // (op 966) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 966) least significant: low of low * low
v_add_u32 v17, v17, v47 // (op 966) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 966) 

v_mov_b32 v47, v12 // (op 966) convert

s_mul_i32 s1, s29, 6 // (op 968) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 968) least significant
s_mul_hi_u32 s6, s28, 6 // (op 968) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 968) most significant: sum

s_mul_i32 s7, s24, s1 // (op 968) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 968) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 968) least significant
s_mul_hi_u32 s15, s24, s0 // (op 968) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 968) most significant: sum
s_add_u32 s7, s7, s15 // (op 968) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 968) 

s_mov_b32 s40, s26 // (op 968) reg expression
s_mov_b32 s41, s27 // (op 968) reg expression
s_mov_b32 s42, s0 // (op 968) reg expression
s_mov_b32 s43, 131072 // (op 968) call()

v_lshrrev_b32 v51, 7, v1 // (op 1030) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 1030) 
v_and_b32 v55, 15, v1 // (op 1030) 
s_lshr_b32 s6, s0, 1 // (op 1030) 

v_lshl_add_u32 v59, v51, 4, v55 // (op 1030) 


v_lshl_add_u32 v51, s16, 4, v59 // (op 1030) 

v_and_b32 v55, 63, v1 // (op 1030) 
v_lshrrev_b32 v59, 4, v55 // (op 1030) 

v_lshl_add_u32 v55, s6, 3, v59 // (op 1030) 


v_mul_lo_u32 v59, v51, s30 // (op 1030) 
v_add_u32 v59, v59, v55 // (op 1030) 


s_mul_i32 s1, s30, s23 // (op 1032) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1032) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1032) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1032) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1032) most significant: sum
s_add_u32 s1, s1, s7 // (op 1032) most significant: sum


s_mov_b32 s52, s32 // (op 1032) reg expression
s_mov_b32 s53, s33 // (op 1032) reg expression
s_mov_b32 s54, s0 // (op 1032) reg expression
s_mov_b32 s55, 131072 // (op 1032) call()

v_lshrrev_b32 v51, 6, v1 // (op 1094) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 1094) 
v_lshlrev_b32 v55, 1, v51 // (op 1094) 

s_lshr_b32 s6, s0, 1 // (op 1094) 

v_and_b32 v51, -64, v1 // (op 1094) 
v_lshl_add_u32 v12, s6, 3, v55 // (op 1094) 


v_sub_u32 v13, v1, v51 // (op 1094) 

v_lshl_add_u32 v51, s17, 6, v13 // (op 1094) 

v_mul_lo_u32 v13, v12, s34 // (op 1094) 
v_add_u32 v13, v13, v51 // (op 1094) 


s_mul_i32 s1, s34, s39 // (op 1096) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1096) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1096) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1096) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1096) most significant: sum
s_add_u32 s1, s1, s7 // (op 1096) most significant: sum


s_mov_b32 s56, s36 // (op 1096) reg expression
s_mov_b32 s57, s37 // (op 1096) reg expression
s_mov_b32 s58, s0 // (op 1096) reg expression
s_mov_b32 s59, 131072 // (op 1096) call()

v_and_b32 v12, -8, v1 // (op 1187) 
v_sub_u32 v51, v1, v12 // (op 1187) 

v_lshlrev_b32 v12, 4, v51 // (op 1187) 

v_lshrrev_b32 v51, 3, v1 // (op 1187) 
v_lshl_add_u32 v55, v51, 7, v12 // (op 1187) 


s_nop 0
v_readfirstlane_b32 s0, v55 // (op 1187) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v12, 6, v1 // (op 1239) 
v_mul_lo_u32 v51, v12, 6 // (op 1239) 
v_lshrrev_b32 v55, 3, v51 // (op 1239) 

v_lshrrev_b32 v51, 4, v12 // (op 1239) 

v_lshlrev_b32 v12, 2, v51 // (op 1239) 

v_add_u32 v51, v55, v12 // (op 1239) 


v_and_b32 v12, -64, v1 // (op 1297) 
v_sub_u32 v55, v1, v12 // (op 1297) 

v_lshrrev_b32 v12, 6, v1 // (op 1297) 
v_lshl_add_u32 v16, v12, 7, v55 // (op 1297) 


s_ashr_i64 s[6:7], s[28:29], 7 // (op 317) 
s_and_b32 s1, s6, 4294967294 // (op 317) 

s_ashr_i64 s[6:7], s[28:29], 7 // (op 316) 
s_mov_b32 s14, s6 // (op 316) convert

s_cmp_lt_u32 s1, s14 // (op 316) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_316 // (op 316) Condition: Top (jump to RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_316 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_316:
s_barrier  // (op 1492) 
s_add_u32 s1, s1, 1 // (op 318) 


v_mov_b64 v[48:49], v[14:15] // (op 869) 

s_mov_b32 m0, s0 // (op 869) reg expression
buffer_load_dwordx4 v48, s[8:11], 0 offen offset: 0 lds // (op 869) Load value direct to lds

v_add_co_u32 v14, vcc, 128, v14 // (op 888) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 888) most significant half


v_mov_b32 v12, v47 // (op 183) 
v_mov_b32 v17, v12 // (op 183) reg expression

buffer_load_dwordx3 v[18:20], v17, s[40:43], 0 offen // (op 183) Load value
buffer_load_dwordx3 v[22:24], v17, s[40:43], 0 offen offset:12 // (op 183) Load value
buffer_load_dwordx3 v[26:28], v17, s[40:43], 0 offen offset:24 // (op 183) Load value
buffer_load_dwordx3 v[30:32], v17, s[40:43], 0 offen offset:36 // (op 183) Load value


s_lshl_b64 s[6:7], s[24:25], 7 // (op 958) 
s_mul_i32 s15, s7, 6 // (op 958) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 958) least significant
s_mul_hi_u32 s18, s6, 6 // (op 958) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 958) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 958) 

v_add_u32 v47, v47, s6 // (op 958) 


s_barrier  // (op 1505) 
v_mov_b64 v[48:49], s[34:35] // (op 184) convert




v_mov_b32 v12, v13 // (op 184) 
v_mov_b32 v17, v12 // (op 184) reg expression
buffer_load_ubyte v52, v17, s[56:59], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v17, v17, v48 // (op 184) 
s_waitcnt vmcnt(4)
ds_write_b96 v51, v[18:20] offset:4096 // (op 181) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v51, v[22:24] offset:4112 // (op 181) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v51, v[26:28] offset:4128 // (op 181) Store local data 
buffer_load_ubyte v53, v17, s[56:59], 0 offen offset: 0 // (op 184) Load value



s_lshl_b32 s6, s34, 2 // (op 1086) 
v_add_u32 v13, v13, s6 // (op 1086) 

s_waitcnt vmcnt(2)
ds_write_b96 v51, v[30:32] offset:4144 // (op 181) Store local data 

s_barrier  // (op 1495) 



s_barrier  // (op 1519) 
ds_read_b128 v[18:21], v3 // (op 173) Load local data 



ds_read_b128 v[22:25], v3 offset:64 // (op 173) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v16, v52 offset:12288 // (op 182) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1509) 
s_waitcnt vmcnt(0)
ds_write_b8 v16, v53 offset:12352 // (op 182) Store local data 




ds_read_b96_tr_b6 v[26:28], v0 offset:4096 // (op 175) Transpose load local data 
ds_read_b96_tr_b6 v[29:31], v0 offset:5120 // (op 175) Transpose load local data 


v_mov_b32 v12, v59 // (op 177) 

buffer_load_ubyte v17, v12, s[52:55], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v59, 4, v59 // (op 1022) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1523) 


ds_read_u8 v12, v5 offset:12288 // (op 179) Load local data 



ds_read_b96_tr_b6 v[32:34], v0 offset:4128 // (op 324) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[26:31], a[0:3], v17, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 180) 
ds_read_b96_tr_b6 v[35:37], v0 offset:5152 // (op 324) Transpose load local data 




ds_read_u8 v12, v5 offset:12320 // (op 327) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:25], v[32:37], a[4:7], v17, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 328) 




s_ashr_i64 s[6:7], s[28:29], 7 // (op 316) 
s_mov_b32 s14, s6 // (op 316) convert

s_cmp_lt_u32 s1, s14 // (op 316) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_316 // (op 316) Condition: Bottom (jump to RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_316 if true)
RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_316:























s_barrier  // (op 1595) 






v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_125
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

buffer_store_dwordx2 v[4:5], v0, s[48:51], 0 offen // (op 54) Store value



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


buffer_store_dwordx2 v[4:5], v0, s[48:51], 0 offen // (op 511) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_125:


v_mov_b32 v0, v6 // (op 138) reg expression

buffer_load_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 138) Load value

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

buffer_store_dwordx2 v[4:5], v0, s[48:51], 0 offen // (op 137) Store value





s_lshl_b32 s0, s2, 6 // (op 535) 
v_add_u32 v0, v6, s0 // (op 535) 


buffer_load_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 535) Load value

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


buffer_store_dwordx2 v[4:5], v0, s[48:51], 0 offen // (op 534) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.LRR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_end:
.size RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2, .LRR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_end-RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
  .amdhsa_next_free_vgpr 92
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 20992 // lds bytes
  .amdhsa_accum_offset 84
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
    DataTypeB: B6
    DestDataType: b
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
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSumScaleA } ]
    macrotile: [16, 64, 128]
    threads: [128, 1, 1]
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
  - .name: RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
    .symbol: RR_GEMM_TT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 82
    .agpr_count: 8
    .max_flat_workgroup_size: 128
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
      - .name: Tensor_2_size_1_12
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 32
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
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
      - .name: Tensor_0_size_0_1
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
      - .name: Tensor_0_size_1_4
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
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
      - .name: Tensor_4_size_1_9
        .size: 8
        .offset: 128
        .value_kind: by_value
...

.end_amdgpu_metadata

