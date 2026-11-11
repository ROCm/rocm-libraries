// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.p2align 8
.type RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2,@function
RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 1214) 
v_and_b32 v3, 63, v1 // (op 1214) 
v_sub_u32 v4, v3, v0 // (op 1214) 

v_lshrrev_b32 v3, 1, v4 // (op 1214) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 1214) 


v_and_b32 v0, 1, v4 // (op 1214) 

v_lshlrev_b32 v3, 3, v0 // (op 1214) 

v_lshrrev_b32 v0, 7, v1 // (op 1214) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1214) 


v_add_u32 v0, v5, v4 // (op 1214) 


v_and_b32 v3, 48, v1 // (op 1387) 
v_and_b32 v4, 63, v1 // (op 1387) 
v_sub_u32 v5, v4, v3 // (op 1387) 

v_lshrrev_b32 v4, 1, v5 // (op 1387) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 1387) 


v_and_b32 v3, 1, v5 // (op 1387) 

v_lshlrev_b32 v4, 3, v3 // (op 1387) 

v_lshrrev_b32 v3, 7, v1 // (op 1387) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 1387) 


v_add_u32 v3, v6, v5 // (op 1387) 


v_lshrrev_b32 v4, 7, v1 // (op 1158) 
v_and_b32 v5, 63, v1 // (op 1158) 
v_lshrrev_b32 v6, 4, v5 // (op 1158) 
v_lshlrev_b32 v7, 2, v6 // (op 1158) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1158) 


v_lshl_add_u32 v4, s16, 4, v6 // (op 1158) 

v_lshrrev_b32 v6, 2, v5 // (op 1158) 

v_and_b32 v5, 3, v6 // (op 1158) 

v_and_b32 v6, 3, v1 // (op 1158) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1158) 


v_lshrrev_b32 v5, 6, v1 // (op 1158) 
v_and_b32 v6, 1, v5 // (op 1158) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1158) 


v_lshl_add_u32 v6, s17, 6, v5 // (op 1158) 

v_mul_lo_u32 v5, v6, s2 // (op 1158) 

v_add_lshl_u32 v6, v4, v5, 1 // (op 1158) 


v_and_b32 v4, 63, v1 // (op 1272) 
v_and_b32 v5, 48, v1 // (op 1272) 
v_sub_u32 v7, v4, v5 // (op 1272) 

v_lshrrev_b32 v5, 4, v4 // (op 1272) 

v_lshl_add_u32 v4, v5, 5, v7 // (op 1272) 


v_lshrrev_b32 v5, 6, v1 // (op 1272) 
v_and_b32 v7, 1, v5 // (op 1272) 

v_lshlrev_b32 v5, 4, v7 // (op 1272) 

v_lshl_add_u32 v7, v4, 6, v5 // (op 1272) 


v_mul_lo_u32 v4, v7, 6 // (op 1272) 
v_lshrrev_b32 v5, 3, v4 // (op 1272) 

v_lshrrev_b32 v4, 4, v7 // (op 1272) 

v_lshlrev_b32 v7, 2, v4 // (op 1272) 

v_add_u32 v4, v5, v7 // (op 1272) 


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


s_mul_i32 s38, s5, s8 // (op 1348) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1348) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1348) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1348) most significant: sum
s_add_u32 s1, s1, s39 // (op 1348) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 1348) 

s_mov_b32 s40, s6 // (op 1160) reg expression
s_mov_b32 s41, s7 // (op 1160) reg expression
s_mov_b32 s42, s18 // (op 1160) reg expression
s_mov_b32 s43, 131072 // (op 1160) call()

s_mov_b32 s44, s10 // (op 1348) reg expression

s_mov_b32 s45, s11 // (op 1348) reg expression
s_mov_b32 s46, s38 // (op 1348) reg expression
s_mov_b32 s47, 131072 // (op 1348) call()



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
s_lshl_b32 s0, s16, 4 // (op 925) 
v_mul_lo_u32 v13, v1, s13 // (op 925) most significant: low of low * high
s_mov_b32 s15, 0 // (op 925) convert
s_mov_b32 s14, s0 // (op 925) convert

v_mul_hi_u32 v14, v1, s12 // (op 925) most significant: high of low * low
v_mul_lo_u32 v12, v1, s12 // (op 925) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 925) most significant: sum

v_mov_b32 v14, s14 // (op 925) 
v_mov_b32 v15, s15 // (op 925) 
v_add_co_u32 v16, s[0:1], v14, v12 // (op 925) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 925) most significant half






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

s_lshl_b32 s0, s16, 4 // (op 944) 
v_mul_lo_u32 v13, v1, s13 // (op 944) most significant: low of low * high
v_mul_hi_u32 v14, v1, s12 // (op 944) most significant: high of low * low
v_mul_lo_u32 v12, v1, s12 // (op 944) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 944) most significant: sum

s_mov_b32 s7, 0 // (op 944) convert
s_mov_b32 s6, s0 // (op 944) convert

v_mov_b32 v14, s6 // (op 944) 
v_mov_b32 v15, s7 // (op 944) 
v_add_co_u32 v18, s[0:1], v14, v12 // (op 944) least significant half
v_addc_co_u32 v19, s[0:1], v15, v13, s[0:1] // (op 944) most significant half





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

s_mul_i32 s1, s23, 6 // (op 991) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 991) least significant
s_mul_hi_u32 s6, s22, 6 // (op 991) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 991) most significant: sum

s_mul_i32 s7, s24, s1 // (op 991) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 991) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 991) least significant
s_mul_hi_u32 s15, s24, s0 // (op 991) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 991) most significant: sum
s_add_u32 s7, s7, s15 // (op 991) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 991) 

s_mov_b32 s48, s26 // (op 991) reg expression
s_mov_b32 s49, s27 // (op 991) reg expression
s_mov_b32 s50, s0 // (op 991) reg expression
s_mov_b32 s51, 131072 // (op 991) call()

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

s_mul_i32 s1, s23, 6 // (op 1010) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1010) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1010) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1010) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1010) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1010) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1010) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1010) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1010) most significant: sum
s_add_u32 s7, s7, s15 // (op 1010) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1010) 

s_mov_b32 s48, s26 // (op 1010) reg expression
s_mov_b32 s49, s27 // (op 1010) reg expression
s_mov_b32 s50, s0 // (op 1010) reg expression
s_mov_b32 s51, 131072 // (op 1010) call()

v_lshrrev_b32 v12, 7, v1 // (op 1053) 
v_and_b32 v13, 15, v1 // (op 1053) 
v_lshl_add_u32 v20, v12, 4, v13 // (op 1053) 


v_lshl_add_u32 v12, s16, 4, v20 // (op 1053) 

v_and_b32 v13, 63, v1 // (op 1053) 
v_lshrrev_b32 v20, 4, v13 // (op 1053) 

v_mul_lo_u32 v13, v20, s28 // (op 1053) 

v_add_u32 v20, v12, v13 // (op 1053) 


s_mul_i32 s1, s28, s33 // (op 1055) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1055) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1055) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1055) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1055) most significant: sum
s_add_u32 s1, s1, s7 // (op 1055) most significant: sum


s_mov_b32 s52, s30 // (op 1055) reg expression
s_mov_b32 s53, s31 // (op 1055) reg expression
s_mov_b32 s54, s0 // (op 1055) reg expression
s_mov_b32 s55, 131072 // (op 1055) call()

v_lshrrev_b32 v12, 7, v1 // (op 1072) 
v_and_b32 v13, 15, v1 // (op 1072) 
v_lshl_add_u32 v21, v12, 4, v13 // (op 1072) 


v_lshl_add_u32 v12, s16, 4, v21 // (op 1072) 

v_and_b32 v13, 63, v1 // (op 1072) 
v_lshrrev_b32 v21, 4, v13 // (op 1072) 

v_mul_lo_u32 v13, v21, s28 // (op 1072) 

v_add_u32 v21, v12, v13 // (op 1072) 


s_mul_i32 s1, s28, s33 // (op 1074) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1074) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1074) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1074) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1074) most significant: sum
s_add_u32 s1, s1, s7 // (op 1074) most significant: sum


s_mov_b32 s52, s30 // (op 1074) reg expression
s_mov_b32 s53, s31 // (op 1074) reg expression
s_mov_b32 s54, s0 // (op 1074) reg expression
s_mov_b32 s55, 131072 // (op 1074) call()

v_and_b32 v12, -64, v1 // (op 1117) 
v_sub_u32 v13, v1, v12 // (op 1117) 

v_lshl_add_u32 v12, s17, 6, v13 // (op 1117) 

v_lshrrev_b32 v13, 6, v1 // (op 1117) 
v_lshlrev_b32 v22, 1, v13 // (op 1117) 

v_mul_lo_u32 v13, v22, s34 // (op 1117) 
v_add_u32 v13, v13, v12 // (op 1117) 


s_mul_i32 s1, s34, s33 // (op 1119) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1119) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1119) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1119) most significant: high of low * low
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


s_mul_i32 s1, s34, s33 // (op 1138) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1138) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1138) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1138) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1138) most significant: sum
s_add_u32 s1, s1, s7 // (op 1138) most significant: sum


s_mov_b32 s56, s36 // (op 1138) reg expression
s_mov_b32 s57, s37 // (op 1138) reg expression
s_mov_b32 s58, s0 // (op 1138) reg expression
s_mov_b32 s59, 131072 // (op 1138) call()


v_mov_b64 v[24:25], s[28:29] // (op 769) convert

v_mov_b32 v12, v20 // (op 769) reg expression
v_mov_b32 v23, v12 // (op 769) 
buffer_load_ubyte v26, v23, s[52:55], 0 offen offset: 0 // (op 769) Load value





v_mov_b32 v12, v14 // (op 773) reg expression

buffer_load_dwordx3 v[28:30], v12, s[48:51], 0 offen // (op 773) Load value
buffer_load_dwordx3 v[32:34], v12, s[48:51], 0 offen offset:12 // (op 773) Load value
buffer_load_dwordx3 v[36:38], v12, s[48:51], 0 offen offset:24 // (op 773) Load value
buffer_load_dwordx3 v[40:42], v12, s[48:51], 0 offen offset:36 // (op 773) Load value


v_mov_b64 v[24:25], s[34:35] // (op 776) convert

v_mov_b32 v12, v13 // (op 776) reg expression
buffer_load_ubyte v44, v12, s[56:59], 0 offen offset: 0 // (op 776) Load value
v_add_u32 v12, v12, v24 // (op 776) 
buffer_load_ubyte v45, v12, s[56:59], 0 offen offset: 0 // (op 776) Load value


v_lshlrev_b32 v12, 4, v1 // (op 1202) 
s_nop 0
v_readfirstlane_b32 s0, v12 // (op 1202) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[24:25], v[16:17] // (op 880) reg expression

s_nop 0
s_mov_b32 m0, s0 // (op 880) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 880) Load value direct to lds



v_mov_b64 v[24:25], s[28:29] // (op 783) convert

s_lshl_b32 s0, s28, 2 // (op 783) 
v_add_u32 v12, v20, s0 // (op 783) 

v_mov_b32 v23, v12 // (op 783) 
buffer_load_ubyte v27, v23, s[52:55], 0 offen offset: 0 // (op 783) Load value






s_lshl_b64 s[0:1], s[24:25], 7 // (op 787) 
s_mul_i32 s7, s1, 6 // (op 787) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 787) least significant
s_mul_hi_u32 s14, s0, 6 // (op 787) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 787) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 787) 

v_add_u32 v12, v14, s0 // (op 787) 


buffer_load_dwordx3 v[46:48], v12, s[48:51], 0 offen // (op 787) Load value
buffer_load_dwordx3 v[50:52], v12, s[48:51], 0 offen offset:12 // (op 787) Load value
buffer_load_dwordx3 v[54:56], v12, s[48:51], 0 offen offset:24 // (op 787) Load value
buffer_load_dwordx3 v[58:60], v12, s[48:51], 0 offen offset:36 // (op 787) Load value



v_mov_b64 v[24:25], s[34:35] // (op 790) convert

s_lshl_b32 s0, s34, 2 // (op 790) 
v_add_u32 v12, v13, s0 // (op 790) 

buffer_load_ubyte v62, v12, s[56:59], 0 offen offset: 0 // (op 790) Load value
v_add_u32 v12, v12, v24 // (op 790) 
buffer_load_ubyte v63, v12, s[56:59], 0 offen offset: 0 // (op 790) Load value



v_lshlrev_b32 v20, 4, v1 // (op 1375) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1375) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[6:7], s[12:13], 7 // (op 885) 
v_mov_b32 v20, s6 // (op 885) 
v_mov_b32 v31, s7 // (op 885) 
v_add_co_u32 v24, s[14:15], v16, v20 // (op 885) least significant half
v_addc_co_u32 v25, s[14:15], v17, v31, s[14:15] // (op 885) most significant half




s_add_u32 s1, 2048, s0 // (op 885) 

s_mov_b32 m0, s1 // (op 885) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 885) Load value direct to lds




v_lshlrev_b32 v20, 6, v1 // (op 1255) 
v_mul_lo_u32 v31, v20, 6 // (op 1255) 
v_lshrrev_b32 v35, 3, v31 // (op 1255) 

v_lshrrev_b32 v31, 4, v20 // (op 1255) 

v_lshlrev_b32 v20, 2, v31 // (op 1255) 

v_add_u32 v31, v35, v20 // (op 1255) 





s_waitcnt vmcnt(14)
ds_write_b96 v31, v[28:30] offset:4096 // (op 795) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v31, v[32:34] offset:4112 // (op 795) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v31, v[36:38] offset:4128 // (op 795) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v31, v[40:42] offset:4144 // (op 795) Store local data 

v_and_b32 v20, -64, v1 // (op 1313) 
v_sub_u32 v31, v1, v20 // (op 1313) 

v_lshrrev_b32 v20, 6, v1 // (op 1313) 
v_lshl_add_u32 v35, v20, 7, v31 // (op 1313) 





s_waitcnt vmcnt(10)
ds_write_b8 v35, v44 offset:12288 // (op 796) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v35, v45 offset:12352 // (op 796) Store local data 

v_lshlrev_b32 v20, 4, v1 // (op 1174) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1174) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v20, 6, v1 // (op 1226) 
v_mul_lo_u32 v31, v20, 6 // (op 1226) 
v_lshrrev_b32 v35, 3, v31 // (op 1226) 

v_lshrrev_b32 v31, 4, v20 // (op 1226) 

v_lshlrev_b32 v20, 2, v31 // (op 1226) 

v_add_u32 v31, v35, v20 // (op 1226) 


v_and_b32 v20, -64, v1 // (op 1284) 
v_sub_u32 v35, v1, v20 // (op 1284) 

v_lshrrev_b32 v20, 6, v1 // (op 1284) 
v_lshl_add_u32 v39, v20, 7, v35 // (op 1284) 


v_lshlrev_b32 v20, 4, v1 // (op 1360) 
s_nop 0
v_readfirstlane_b32 s1, v20 // (op 1360) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v20, 6, v1 // (op 1399) 
v_mul_lo_u32 v35, v20, 6 // (op 1399) 
v_lshrrev_b32 v43, 3, v35 // (op 1399) 

v_lshrrev_b32 v35, 4, v20 // (op 1399) 

v_lshlrev_b32 v20, 2, v35 // (op 1399) 

v_add_u32 v35, v43, v20 // (op 1399) 


v_and_b32 v20, -64, v1 // (op 1428) 
v_sub_u32 v43, v1, v20 // (op 1428) 

v_lshrrev_b32 v20, 6, v1 // (op 1428) 
v_lshl_add_u32 v49, v20, 7, v43 // (op 1428) 


s_mov_b32 s6, 0 // (op 312) call()
s_ashr_i64 s[14:15], s[22:23], 7 // (op 311) 
s_mov_b32 s7, s14 // (op 311) convert

s_and_b32 s14, s7, 4294967294 // (op 311) 

s_cmp_lt_u32 s6, s14 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_311 // (op 311) Condition: Top (jump to RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_311 if false)
RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_311:
s_add_u32 s6, s6, 2 // (op 313) 
s_waitcnt vmcnt(8)
s_barrier  // (op 1486) 



ds_read_b64_tr_b8 v[64:65], v0 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v0 offset:128 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v0 offset:1024 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v0 offset:1152 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1499) 



ds_read_b96_tr_b6 v[72:74], v4 offset:4096 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[75:77], v4 offset:5120 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1513) 


ds_read_u8 v20, v5 offset:12288 // (op 81) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:71], v[72:77], a[0:3], v26, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 84) 
ds_read_b96_tr_b6 v[78:80], v4 offset:4128 // (op 343) Transpose load local data 


ds_read_b96_tr_b6 v[81:83], v4 offset:5152 // (op 343) Transpose load local data 


ds_read_u8 v20, v5 offset:12320 // (op 346) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:71], v[78:83], a[4:7], v26, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 347) 




v_mov_b64 v[16:17], s[28:29] // (op 77) convert

v_mov_b32 v20, v21 // (op 77) 
s_lshl_b32 s7, s28, 3 // (op 77) 
v_add_u32 v43, v20, s7 // (op 77) 

v_mov_b32 v53, v43 // (op 77) 
buffer_load_ubyte v26, v53, s[52:55], 0 offen offset: 0 // (op 77) Load value





v_mov_b32 v43, v15 // (op 10) 
s_lshl_b64 s[14:15], s[24:25], 7 // (op 10) 
s_mul_i32 s19, s15, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 10) least significant
s_mul_hi_u32 s7, s14, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 10) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s7, s14, 1 // (op 10) 

v_add_u32 v53, v43, s7 // (op 10) 


buffer_load_dwordx3 v[28:30], v53, s[48:51], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[32:34], v53, s[48:51], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[36:38], v53, s[48:51], 0 offen offset:24 // (op 10) Load value
buffer_load_dwordx3 v[40:42], v53, s[48:51], 0 offen offset:36 // (op 10) Load value


v_mov_b64 v[16:17], s[34:35] // (op 21) convert

v_mov_b32 v53, v22 // (op 21) 
s_lshl_b32 s7, s34, 3 // (op 21) 
v_add_u32 v57, v53, s7 // (op 21) 

buffer_load_ubyte v44, v57, s[56:59], 0 offen offset: 0 // (op 21) Load value
v_add_u32 v57, v57, v16 // (op 21) 
buffer_load_ubyte v45, v57, s[56:59], 0 offen offset: 0 // (op 21) Load value


s_barrier  // (op 1489) 


v_mov_b64 v[16:17], v[18:19] // (op 864) 
s_lshl_b64 s[14:15], s[12:13], 7 // (op 864) 
s_lshl_b64 s[18:19], s[14:15], 1 // (op 864) 

v_mov_b32 v57, s18 // (op 864) 
v_mov_b32 v61, s19 // (op 864) 
v_add_co_u32 v24, s[14:15], v16, v57 // (op 864) least significant half
v_addc_co_u32 v25, s[14:15], v17, v61, s[14:15] // (op 864) most significant half





s_mov_b32 m0, s0 // (op 864) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 864) Load value direct to lds

s_barrier  // (op 1536) 



s_waitcnt vmcnt(14)
ds_write_b96 v35, v[46:48] offset:12544 // (op 239) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v35, v[50:52] offset:12560 // (op 239) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v35, v[54:56] offset:12576 // (op 239) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v35, v[58:60] offset:12592 // (op 239) Store local data 
s_barrier  // (op 1542) 



s_waitcnt vmcnt(10)
ds_write_b8 v49, v62 offset:20736 // (op 240) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v49, v63 offset:20800 // (op 240) Store local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 1527) 



ds_read_b64_tr_b8 v[64:65], v3 offset:2048 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v3 offset:2176 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v3 offset:3072 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v3 offset:3200 // (op 231) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1533) 



ds_read_b96_tr_b6 v[72:74], v7 offset:12544 // (op 233) Transpose load local data 
ds_read_b96_tr_b6 v[75:77], v7 offset:13568 // (op 233) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1539) 


ds_read_u8 v57, v8 offset:20736 // (op 237) Load local data 



ds_read_b96_tr_b6 v[78:80], v7 offset:12576 // (op 352) Transpose load local data 
ds_read_b96_tr_b6 v[81:83], v7 offset:13600 // (op 352) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:71], v[72:77], a[0:3], v27, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 238) 




s_barrier  // (op 1607) 
ds_read_u8 v57, v8 offset:20768 // (op 355) Load local data 
s_barrier  // (op 1611) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:71], v[78:83], a[4:7], v27, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 348) 




v_mov_b64 v[24:25], s[28:29] // (op 235) convert

s_lshl_b32 s7, s28, 2 // (op 235) 
s_mul_i32 s14, 3, s7 // (op 235) 

v_add_u32 v23, v20, s14 // (op 235) 

v_mov_b32 v57, v23 // (op 235) 
buffer_load_ubyte v27, v57, s[52:55], 0 offen offset: 0 // (op 235) Load value






s_lshl_b32 s7, s28, 3 // (op 1064) 
v_add_u32 v21, v21, s7 // (op 1064) 

s_lshl_b64 s[14:15], s[24:25], 7 // (op 241) 
s_mul_i32 s19, s15, 6 // (op 241) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 241) least significant
s_mul_hi_u32 s7, s14, 6 // (op 241) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 241) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 241) 

s_mul_i32 s7, 3, s14 // (op 241) 

v_add_u32 v20, v43, s7 // (op 241) 


buffer_load_dwordx3 v[46:48], v20, s[48:51], 0 offen // (op 241) Load value
buffer_load_dwordx3 v[50:52], v20, s[48:51], 0 offen offset:12 // (op 241) Load value
buffer_load_dwordx3 v[54:56], v20, s[48:51], 0 offen offset:24 // (op 241) Load value
buffer_load_dwordx3 v[58:60], v20, s[48:51], 0 offen offset:36 // (op 241) Load value



v_mov_b64 v[24:25], s[34:35] // (op 242) convert

s_lshl_b32 s7, s34, 2 // (op 242) 
s_mul_i32 s14, 3, s7 // (op 242) 

v_add_u32 v20, v53, s14 // (op 242) 

s_lshl_b64 s[14:15], s[24:25], 7 // (op 1000) 
s_mul_i32 s19, s15, 6 // (op 1000) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 1000) least significant
s_mul_hi_u32 s7, s14, 6 // (op 1000) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 1000) most significant: sum


buffer_load_ubyte v62, v20, s[56:59], 0 offen offset: 0 // (op 242) Load value
v_add_u32 v20, v20, v24 // (op 242) 
s_lshr_b64 s[14:15], s[18:19], 3 // (op 1000) 

s_lshl_b32 s7, s14, 1 // (op 1000) 

v_add_u32 v15, v15, s7 // (op 1000) 

buffer_load_ubyte v63, v20, s[56:59], 0 offen offset: 0 // (op 242) Load value



s_barrier  // (op 1530) 
s_lshl_b32 s7, s34, 3 // (op 1128) 


v_add_u32 v22, v22, s7 // (op 1128) 

s_lshl_b64 s[14:15], s[12:13], 7 // (op 875) 
s_mul_i32 s19, 3, s15 // (op 875) most significant: low of low * high
s_mul_i32 s18, 3, s14 // (op 875) least significant
s_mul_hi_u32 s7, 3, s14 // (op 875) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 875) most significant: sum


v_mov_b32 v20, s18 // (op 875) 
v_mov_b32 v43, s19 // (op 875) 
v_add_co_u32 v24, s[14:15], v16, v20 // (op 875) least significant half
v_addc_co_u32 v25, s[14:15], v17, v43, s[14:15] // (op 875) most significant half




s_add_u32 s7, 2048, s1 // (op 875) 

s_mov_b32 m0, s7 // (op 875) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 875) Load value direct to lds



s_barrier  // (op 1502) 
s_lshl_b64 s[14:15], s[12:13], 7 // (op 936) 



s_lshl_b64 s[18:19], s[14:15], 1 // (op 936) 

v_mov_b32 v20, s18 // (op 936) 
v_mov_b32 v43, s19 // (op 936) 
v_add_co_u32 v18, s[14:15], v18, v20 // (op 936) least significant half
v_addc_co_u32 v19, s[14:15], v19, v43, s[14:15] // (op 936) most significant half




s_waitcnt vmcnt(14)
ds_write_b96 v31, v[28:30] offset:4096 // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v31, v[32:34] offset:4112 // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v31, v[36:38] offset:4128 // (op 13) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v31, v[40:42] offset:4144 // (op 13) Store local data 
s_barrier  // (op 1516) 



s_waitcnt vmcnt(10)
ds_write_b8 v39, v44 offset:12288 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v39, v45 offset:12352 // (op 26) Store local data 
s_ashr_i64 s[14:15], s[22:23], 7 // (op 311) 
s_mov_b32 s7, s14 // (op 311) convert

s_and_b32 s14, s7, 4294967294 // (op 311) 

s_cmp_lt_u32 s6, s14 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_311 // (op 311) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_311 if true)
RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_311:














s_barrier  // (op 1591) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 898) 
s_lshr_b32 s6, s0, 1 // (op 898) 

v_lshl_add_u32 v49, s6, 8, v1 // (op 898) 

s_lshl_b32 s0, s16, 4 // (op 898) 
v_mul_lo_u32 v13, v49, s13 // (op 898) most significant: low of low * high
v_mul_hi_u32 v53, v49, s12 // (op 898) most significant: high of low * low
v_mul_lo_u32 v12, v49, s12 // (op 898) least significant: low of low * low
v_add_u32 v13, v13, v53 // (op 898) most significant: sum


s_mov_b32 s7, 0 // (op 898) convert
s_mov_b32 s6, s0 // (op 898) convert

v_mov_b32 v49, s6 // (op 898) 
v_mov_b32 v53, s7 // (op 898) 
v_add_co_u32 v14, s[0:1], v49, v12 // (op 898) least significant half
v_addc_co_u32 v15, s[0:1], v53, v13, s[0:1] // (op 898) most significant half





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

s_ashr_i64 s[0:1], s[22:23], 7 // (op 966) 
s_lshr_b32 s6, s0, 1 // (op 966) 

v_lshl_add_u32 v49, s6, 8, v1 // (op 966) 

s_lshl_b32 s0, s17, 6 // (op 966) 
s_mov_b32 s7, 0 // (op 966) convert
s_mov_b32 s6, s0 // (op 966) convert

v_mul_lo_u32 v13, v49, s25 // (op 966) most significant: low of low * high
v_mul_hi_u32 v53, v49, s24 // (op 966) most significant: high of low * low
v_mul_lo_u32 v12, v49, s24 // (op 966) least significant: low of low * low
v_add_u32 v13, v13, v53 // (op 966) most significant: sum

v_mov_b32 v53, s6 // (op 966) 
v_mov_b32 v57, s7 // (op 966) 
v_add_co_u32 v12, s[0:1], v12, v53 // (op 966) least significant half
v_addc_co_u32 v13, s[0:1], v13, v57, s[0:1] // (op 966) most significant half





v_mul_lo_u32 v17, v13, 6 // (op 966) most significant: low of high * low
v_mul_hi_u32 v49, v12, 6 // (op 966) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 966) least significant: low of low * low
v_add_u32 v17, v17, v49 // (op 966) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 966) 

v_mov_b32 v49, v12 // (op 966) convert

s_mul_i32 s1, s23, 6 // (op 968) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 968) least significant
s_mul_hi_u32 s6, s22, 6 // (op 968) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 968) most significant: sum

s_mul_i32 s7, s24, s1 // (op 968) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 968) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 968) least significant
s_mul_hi_u32 s15, s24, s0 // (op 968) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 968) most significant: sum
s_add_u32 s7, s7, s15 // (op 968) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 968) 

s_mov_b32 s48, s26 // (op 968) reg expression
s_mov_b32 s49, s27 // (op 968) reg expression
s_mov_b32 s50, s0 // (op 968) reg expression
s_mov_b32 s51, 131072 // (op 968) call()

v_lshrrev_b32 v53, 7, v1 // (op 1030) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1030) 
v_and_b32 v57, 15, v1 // (op 1030) 
s_lshr_b32 s6, s0, 1 // (op 1030) 

v_lshl_add_u32 v61, v53, 4, v57 // (op 1030) 


v_lshl_add_u32 v53, s16, 4, v61 // (op 1030) 

v_and_b32 v57, 63, v1 // (op 1030) 
v_lshrrev_b32 v61, 4, v57 // (op 1030) 

v_lshl_add_u32 v57, s6, 3, v61 // (op 1030) 


v_mul_lo_u32 v61, v57, s28 // (op 1030) 

v_add_u32 v57, v53, v61 // (op 1030) 


s_mul_i32 s1, s28, s33 // (op 1032) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1032) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1032) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1032) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1032) most significant: sum
s_add_u32 s1, s1, s7 // (op 1032) most significant: sum


s_mov_b32 s52, s30 // (op 1032) reg expression
s_mov_b32 s53, s31 // (op 1032) reg expression
s_mov_b32 s54, s0 // (op 1032) reg expression
s_mov_b32 s55, 131072 // (op 1032) call()

v_lshrrev_b32 v53, 6, v1 // (op 1094) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1094) 
v_lshlrev_b32 v61, 1, v53 // (op 1094) 

s_lshr_b32 s6, s0, 1 // (op 1094) 

v_and_b32 v53, -64, v1 // (op 1094) 
v_lshl_add_u32 v12, s6, 3, v61 // (op 1094) 


v_sub_u32 v13, v1, v53 // (op 1094) 

v_lshl_add_u32 v53, s17, 6, v13 // (op 1094) 

v_mul_lo_u32 v13, v12, s34 // (op 1094) 
v_add_u32 v13, v13, v53 // (op 1094) 


s_mul_i32 s1, s34, s33 // (op 1096) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1096) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1096) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1096) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1096) most significant: sum
s_add_u32 s1, s1, s7 // (op 1096) most significant: sum


s_mov_b32 s56, s36 // (op 1096) reg expression
s_mov_b32 s57, s37 // (op 1096) reg expression
s_mov_b32 s58, s0 // (op 1096) reg expression
s_mov_b32 s59, 131072 // (op 1096) call()

v_lshlrev_b32 v12, 4, v1 // (op 1187) 
s_nop 0
v_readfirstlane_b32 s0, v12 // (op 1187) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v12, 6, v1 // (op 1239) 
v_mul_lo_u32 v53, v12, 6 // (op 1239) 
v_lshrrev_b32 v61, 3, v53 // (op 1239) 

v_lshrrev_b32 v53, 4, v12 // (op 1239) 

v_lshlrev_b32 v12, 2, v53 // (op 1239) 

v_add_u32 v53, v61, v12 // (op 1239) 


v_and_b32 v12, -64, v1 // (op 1297) 
v_sub_u32 v61, v1, v12 // (op 1297) 

v_lshrrev_b32 v12, 6, v1 // (op 1297) 
v_lshl_add_u32 v16, v12, 7, v61 // (op 1297) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 317) 
s_and_b32 s1, s6, 4294967294 // (op 317) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 316) 
s_mov_b32 s14, s6 // (op 316) convert

s_cmp_lt_u32 s1, s14 // (op 316) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_316 // (op 316) Condition: Top (jump to RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_316 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_316:
s_barrier  // (op 1492) 
s_add_u32 s1, s1, 1 // (op 318) 


v_mov_b64 v[50:51], v[14:15] // (op 869) 
v_mov_b64 v[54:55], v[50:51] // (op 869) reg expression

s_mov_b32 m0, s0 // (op 869) reg expression
buffer_load_dwordx4 v54, s[8:11], 0 offen offset: 0 lds // (op 869) Load value direct to lds


s_lshl_b64 s[6:7], s[12:13], 7 // (op 888) 
v_mov_b32 v12, s6 // (op 888) 
v_mov_b32 v17, s7 // (op 888) 
v_add_co_u32 v14, s[14:15], v14, v12 // (op 888) least significant half
v_addc_co_u32 v15, s[14:15], v15, v17, s[14:15] // (op 888) most significant half






v_mov_b32 v12, v49 // (op 183) 
v_mov_b32 v17, v12 // (op 183) reg expression

buffer_load_dwordx3 v[18:20], v17, s[48:51], 0 offen // (op 183) Load value
buffer_load_dwordx3 v[22:24], v17, s[48:51], 0 offen offset:12 // (op 183) Load value
buffer_load_dwordx3 v[26:28], v17, s[48:51], 0 offen offset:24 // (op 183) Load value
buffer_load_dwordx3 v[30:32], v17, s[48:51], 0 offen offset:36 // (op 183) Load value


s_lshl_b64 s[6:7], s[24:25], 7 // (op 958) 
s_mul_i32 s15, s7, 6 // (op 958) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 958) least significant
s_mul_hi_u32 s18, s6, 6 // (op 958) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 958) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 958) 

v_add_u32 v49, v49, s6 // (op 958) 


s_barrier  // (op 1505) 
v_mov_b64 v[50:51], s[34:35] // (op 184) convert




v_mov_b32 v12, v13 // (op 184) 
v_mov_b32 v17, v12 // (op 184) reg expression
buffer_load_ubyte v54, v17, s[56:59], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v17, v17, v50 // (op 184) 
s_waitcnt vmcnt(4)
ds_write_b96 v53, v[18:20] offset:4096 // (op 181) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v53, v[22:24] offset:4112 // (op 181) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v53, v[26:28] offset:4128 // (op 181) Store local data 
buffer_load_ubyte v55, v17, s[56:59], 0 offen offset: 0 // (op 184) Load value



s_lshl_b32 s6, s34, 2 // (op 1086) 
v_add_u32 v13, v13, s6 // (op 1086) 

s_waitcnt vmcnt(2)
ds_write_b96 v53, v[30:32] offset:4144 // (op 181) Store local data 

s_barrier  // (op 1495) 



s_barrier  // (op 1519) 
ds_read_b64_tr_b8 v[18:19], v0 // (op 173) Transpose load local data 



ds_read_b64_tr_b8 v[20:21], v0 offset:128 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[22:23], v0 offset:1024 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:1152 // (op 173) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v16, v54 offset:12288 // (op 182) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1509) 
s_waitcnt vmcnt(0)
ds_write_b8 v16, v55 offset:12352 // (op 182) Store local data 




ds_read_b96_tr_b6 v[26:28], v4 offset:4096 // (op 175) Transpose load local data 
ds_read_b96_tr_b6 v[29:31], v4 offset:5120 // (op 175) Transpose load local data 

v_mov_b64 v[32:33], s[28:29] // (op 177) convert

v_mov_b32 v12, v57 // (op 177) 
v_mov_b32 v17, v12 // (op 177) reg expression
v_mov_b32 v34, v17 // (op 177) 
buffer_load_ubyte v35, v34, s[52:55], 0 offen offset: 0 // (op 177) Load value




s_lshl_b32 s6, s28, 2 // (op 1022) 
v_add_u32 v57, v57, s6 // (op 1022) 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1523) 


ds_read_u8 v12, v5 offset:12288 // (op 179) Load local data 



ds_read_b96_tr_b6 v[36:38], v4 offset:4128 // (op 324) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v4 offset:5152 // (op 324) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[26:31], a[0:3], v35, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 180) 




ds_read_u8 v12, v5 offset:12320 // (op 327) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:25], v[36:41], a[4:7], v35, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 328) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 316) 
s_mov_b32 s14, s6 // (op 316) convert

s_cmp_lt_u32 s1, s14 // (op 316) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_316 // (op 316) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_316 if true)
RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_316:






















s_barrier  // (op 1595) 






v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_125
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

v_cvt_f16_f32 v12, v16 // (op 510) 
v_cvt_f16_f32 v13, v17 // (op 510) 
v_cvt_f16_f32 v14, v18 // (op 510) 
v_cvt_f16_f32 v15, v19 // (op 510) 


v_pack_b32_f16 v4, v12, v13 // (op 511) 
v_pack_b32_f16 v5, v14, v15 // (op 511) 

s_lshl_b32 s0, s4, 6 // (op 511) 
v_add_u32 v0, v9, s0 // (op 511) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 511) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_125:


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
v_cvt_f32_f16 v12, v4 // (op 139) 
v_lshrrev_b32 v13, 16, v4 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v20, v10, v12 // (op 139) 
v_mul_f32 v21, v10, v13 // (op 139) 
v_cvt_f32_f16 v12, v5 // (op 139) 
v_lshrrev_b32 v13, 16, v5 // (op 139) 
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
v_cvt_f32_f16 v12, v4 // (op 536) 
v_lshrrev_b32 v13, 16, v4 // (op 536) 
v_cvt_f32_f16 v13, v13 // (op 536) 
v_mul_f32 v20, v10, v12 // (op 536) 
v_mul_f32 v21, v10, v13 // (op 536) 
v_cvt_f32_f16 v12, v5 // (op 536) 
v_lshrrev_b32 v13, 16, v5 // (op 536) 
v_cvt_f32_f16 v13, v13 // (op 536) 
v_mul_f32 v22, v10, v12 // (op 536) 
v_mul_f32 v23, v10, v13 // (op 536) 


v_add_f32 v12, v20, v16 // (op 532) 
v_add_f32 v13, v21, v17 // (op 532) 
v_add_f32 v14, v22, v18 // (op 532) 
v_add_f32 v15, v23, v19 // (op 532) 


v_cvt_f16_f32 v16, v12 // (op 533) 
v_cvt_f16_f32 v17, v13 // (op 533) 
v_cvt_f16_f32 v18, v14 // (op 533) 
v_cvt_f16_f32 v19, v15 // (op 533) 


v_pack_b32_f16 v4, v16, v17 // (op 534) 
v_pack_b32_f16 v5, v18, v19 // (op 534) 

s_lshl_b32 s0, s4, 6 // (op 534) 
v_add_u32 v0, v9, s0 // (op 534) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 534) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.LRR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_end:
.size RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2, .LRR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_end-RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: F6
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
  - .name: RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
    .symbol: RR_GEMM_NT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 84
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

