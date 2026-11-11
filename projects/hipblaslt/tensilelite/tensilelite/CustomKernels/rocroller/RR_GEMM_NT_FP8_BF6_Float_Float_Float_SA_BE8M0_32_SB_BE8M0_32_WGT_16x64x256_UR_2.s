// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2,@function
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 1517) 
v_and_b32 v3, 63, v1 // (op 1517) 
v_sub_u32 v4, v3, v0 // (op 1517) 

v_lshrrev_b32 v3, 1, v4 // (op 1517) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 1517) 


v_and_b32 v0, 1, v4 // (op 1517) 

v_lshlrev_b32 v3, 3, v0 // (op 1517) 

v_lshrrev_b32 v0, 7, v1 // (op 1517) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1517) 


v_add_u32 v0, v5, v4 // (op 1517) 


v_lshrrev_b32 v3, 7, v1 // (op 1628) 
v_and_b32 v4, 15, v1 // (op 1628) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 1628) 


v_and_b32 v3, 48, v1 // (op 1628) 
v_add_u32 v4, v3, v5 // (op 1628) 


v_and_b32 v3, 48, v1 // (op 1743) 
v_and_b32 v5, 63, v1 // (op 1743) 
v_sub_u32 v6, v5, v3 // (op 1743) 

v_lshrrev_b32 v5, 1, v6 // (op 1743) 
v_add_lshl_u32 v7, v3, v5, 4 // (op 1743) 


v_and_b32 v3, 1, v6 // (op 1743) 

v_lshlrev_b32 v5, 3, v3 // (op 1743) 

v_lshrrev_b32 v3, 7, v1 // (op 1743) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 1743) 


v_add_u32 v3, v7, v6 // (op 1743) 


v_lshrrev_b32 v5, 7, v1 // (op 1796) 
v_and_b32 v6, 15, v1 // (op 1796) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 1796) 


v_and_b32 v5, 48, v1 // (op 1796) 
v_add_u32 v6, v5, v7 // (op 1796) 


v_lshrrev_b32 v5, 7, v1 // (op 1461) 
v_and_b32 v7, 63, v1 // (op 1461) 
v_lshrrev_b32 v8, 4, v7 // (op 1461) 
v_lshlrev_b32 v9, 2, v8 // (op 1461) 

v_lshl_add_u32 v8, v5, 4, v9 // (op 1461) 


v_lshl_add_u32 v5, s16, 4, v8 // (op 1461) 

v_lshrrev_b32 v8, 2, v7 // (op 1461) 

v_and_b32 v7, 3, v8 // (op 1461) 

v_and_b32 v8, 3, v1 // (op 1461) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 1461) 


v_lshrrev_b32 v7, 6, v1 // (op 1461) 
v_and_b32 v8, 1, v7 // (op 1461) 

v_lshl_add_u32 v7, v8, 4, v9 // (op 1461) 


v_lshl_add_u32 v8, s17, 6, v7 // (op 1461) 

v_mul_lo_u32 v7, v8, s2 // (op 1461) 

v_add_lshl_u32 v8, v5, v7, 2 // (op 1461) 


v_and_b32 v5, 63, v1 // (op 1575) 
v_and_b32 v7, 48, v1 // (op 1575) 
v_sub_u32 v9, v5, v7 // (op 1575) 

v_lshrrev_b32 v7, 4, v5 // (op 1575) 

v_lshl_add_u32 v5, v7, 5, v9 // (op 1575) 


v_lshrrev_b32 v7, 6, v1 // (op 1575) 
v_and_b32 v9, 1, v7 // (op 1575) 

v_lshlrev_b32 v7, 4, v9 // (op 1575) 

v_lshl_add_u32 v9, v5, 6, v7 // (op 1575) 


v_mul_lo_u32 v5, v9, 6 // (op 1575) 
v_lshrrev_b32 v7, 3, v5 // (op 1575) 

v_lshrrev_b32 v5, 4, v9 // (op 1575) 

v_lshlrev_b32 v9, 2, v5 // (op 1575) 

v_add_u32 v5, v7, v9 // (op 1575) 


v_lshrrev_b32 v7, 6, v1 // (op 1686) 
v_and_b32 v9, 1, v7 // (op 1686) 

v_and_b32 v7, 15, v1 // (op 1686) 
s_mul_i32 s1, s2, s9 // (op 1463) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1463) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1463) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1463) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1463) most significant: sum
s_add_u32 s1, s1, s19 // (op 1463) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1463) 

v_lshl_add_u32 v10, v9, 4, v7 // (op 1686) 


v_and_b32 v7, 63, v1 // (op 1686) 
v_lshrrev_b32 v9, 4, v7 // (op 1686) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 1686) 


v_lshrrev_b32 v9, 7, v1 // (op 1702) 
v_and_b32 v10, 63, v1 // (op 1702) 
v_lshrrev_b32 v11, 4, v10 // (op 1702) 
v_lshlrev_b32 v12, 2, v11 // (op 1702) 

v_lshl_add_u32 v11, v9, 4, v12 // (op 1702) 


v_lshl_add_u32 v9, s16, 4, v11 // (op 1702) 

v_lshrrev_b32 v11, 2, v10 // (op 1702) 

v_and_b32 v10, 3, v11 // (op 1702) 

v_and_b32 v11, 3, v1 // (op 1702) 
v_lshl_add_u32 v12, v10, 2, v11 // (op 1702) 


v_lshrrev_b32 v10, 6, v1 // (op 1702) 
v_and_b32 v11, 1, v10 // (op 1702) 

v_lshl_add_u32 v10, v11, 4, v12 // (op 1702) 


v_lshl_add_u32 v11, s17, 6, v10 // (op 1702) 

v_mul_lo_u32 v10, v11, s4 // (op 1702) 

v_add_lshl_u32 v11, v9, v10, 2 // (op 1702) 


v_and_b32 v9, 63, v1 // (op 1772) 
v_and_b32 v10, 48, v1 // (op 1772) 
v_sub_u32 v12, v9, v10 // (op 1772) 

v_lshrrev_b32 v10, 4, v9 // (op 1772) 

v_lshl_add_u32 v9, v10, 5, v12 // (op 1772) 


v_lshrrev_b32 v10, 6, v1 // (op 1772) 
v_and_b32 v12, 1, v10 // (op 1772) 

v_lshlrev_b32 v10, 4, v12 // (op 1772) 

v_lshl_add_u32 v12, v9, 6, v10 // (op 1772) 


v_mul_lo_u32 v9, v12, 6 // (op 1772) 
v_lshrrev_b32 v10, 3, v9 // (op 1772) 

v_lshrrev_b32 v9, 4, v12 // (op 1772) 

v_lshlrev_b32 v12, 2, v9 // (op 1772) 

v_add_u32 v9, v10, v12 // (op 1772) 


v_lshrrev_b32 v10, 6, v1 // (op 1825) 
v_and_b32 v12, 1, v10 // (op 1825) 

v_and_b32 v10, 15, v1 // (op 1825) 
s_mul_i32 s1, s4, s9 // (op 1704) most significant: low of low * high
v_lshl_add_u32 v13, v12, 4, v10 // (op 1825) 


v_and_b32 v10, 63, v1 // (op 1825) 
v_lshrrev_b32 v12, 4, v10 // (op 1825) 

v_lshl_add_u32 v10, v12, 6, v13 // (op 1825) 


s_mul_i32 s38, s5, s8 // (op 1704) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1704) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1704) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1704) most significant: sum
s_add_u32 s1, s1, s39 // (op 1704) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 1704) 

s_mov_b32 s40, s6 // (op 1463) reg expression
s_mov_b32 s41, s7 // (op 1463) reg expression
s_mov_b32 s42, s18 // (op 1463) reg expression
s_mov_b32 s43, 131072 // (op 1463) call()

s_mov_b32 s44, s10 // (op 1704) reg expression

s_mov_b32 s45, s11 // (op 1704) reg expression
s_mov_b32 s46, s38 // (op 1704) reg expression
s_mov_b32 s47, 131072 // (op 1704) call()



v_mov_b32 v12, s14 // (op 2) Move value
v_mov_b32 v13, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 398) call()
v_accvgpr_write a5, 0.00000 // (op 398) call()
v_accvgpr_write a6, 0.00000 // (op 398) call()
v_accvgpr_write a7, 0.00000 // (op 398) call()
s_lshl_b32 s0, s16, 4 // (op 1228) 
v_mul_lo_u32 v15, v1, s13 // (op 1228) most significant: low of low * high
s_mov_b32 s15, 0 // (op 1228) convert
s_mov_b32 s14, s0 // (op 1228) convert

v_mul_hi_u32 v16, v1, s12 // (op 1228) most significant: high of low * low
v_mul_lo_u32 v14, v1, s12 // (op 1228) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 1228) most significant: sum

v_mov_b32 v16, s14 // (op 1228) 
v_mov_b32 v17, s15 // (op 1228) 
v_add_co_u32 v18, s[0:1], v16, v14 // (op 1228) least significant half
v_addc_co_u32 v19, s[0:1], v17, v15, s[0:1] // (op 1228) most significant half






s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 1230) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1230) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1230) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1230) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1230) most significant: sum
s_add_u32 s1, s1, s7 // (op 1230) most significant: sum


s_mov_b32 s8, s20 // (op 1230) reg expression
s_mov_b32 s9, s21 // (op 1230) reg expression
s_mov_b32 s10, s0 // (op 1230) reg expression
s_mov_b32 s11, 131072 // (op 1230) call()

s_lshl_b32 s0, s16, 4 // (op 1247) 
v_mul_lo_u32 v15, v1, s13 // (op 1247) most significant: low of low * high
v_mul_hi_u32 v16, v1, s12 // (op 1247) most significant: high of low * low
v_mul_lo_u32 v14, v1, s12 // (op 1247) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 1247) most significant: sum

s_mov_b32 s7, 0 // (op 1247) convert
s_mov_b32 s6, s0 // (op 1247) convert

v_mov_b32 v16, s6 // (op 1247) 
v_mov_b32 v17, s7 // (op 1247) 
v_add_co_u32 v20, s[0:1], v16, v14 // (op 1247) least significant half
v_addc_co_u32 v21, s[0:1], v17, v15, s[0:1] // (op 1247) most significant half





s_mul_i32 s1, s12, s23 // (op 1249) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1249) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1249) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1249) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1249) most significant: sum
s_add_u32 s1, s1, s7 // (op 1249) most significant: sum


s_mov_b32 s8, s20 // (op 1249) reg expression
s_mov_b32 s9, s21 // (op 1249) reg expression
s_mov_b32 s10, s0 // (op 1249) reg expression
s_mov_b32 s11, 131072 // (op 1249) call()

s_lshl_b32 s0, s17, 6 // (op 1292) 
v_lshlrev_b32 v14, 1, v1 // (op 1292) 
s_mov_b32 s7, 0 // (op 1292) convert
s_mov_b32 s6, s0 // (op 1292) convert

v_mul_lo_u32 v17, v14, s25 // (op 1292) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 1292) most significant: high of low * low
v_mul_lo_u32 v16, v14, s24 // (op 1292) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 1292) most significant: sum

v_mov_b32 v15, s6 // (op 1292) 
v_mov_b32 v22, s7 // (op 1292) 
v_add_co_u32 v16, s[0:1], v16, v15 // (op 1292) least significant half
v_addc_co_u32 v17, s[0:1], v17, v22, s[0:1] // (op 1292) most significant half





v_mul_lo_u32 v15, v17, 6 // (op 1292) most significant: low of high * low
v_mul_hi_u32 v22, v16, 6 // (op 1292) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 1292) least significant: low of low * low
v_add_u32 v15, v15, v22 // (op 1292) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1292) 

v_mov_b32 v14, v16 // (op 1292) convert

s_mul_i32 s1, s23, 6 // (op 1294) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1294) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1294) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1294) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1294) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1294) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1294) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1294) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1294) most significant: sum
s_add_u32 s7, s7, s15 // (op 1294) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1294) 

s_mov_b32 s48, s26 // (op 1294) reg expression
s_mov_b32 s49, s27 // (op 1294) reg expression
s_mov_b32 s50, s0 // (op 1294) reg expression
s_mov_b32 s51, 131072 // (op 1294) call()

s_lshl_b32 s0, s17, 6 // (op 1311) 
v_lshlrev_b32 v15, 1, v1 // (op 1311) 
s_mov_b32 s7, 0 // (op 1311) convert
s_mov_b32 s6, s0 // (op 1311) convert

v_mul_lo_u32 v17, v15, s25 // (op 1311) most significant: low of low * high
v_mul_hi_u32 v22, v15, s24 // (op 1311) most significant: high of low * low
v_mul_lo_u32 v16, v15, s24 // (op 1311) least significant: low of low * low
v_add_u32 v17, v17, v22 // (op 1311) most significant: sum

v_mov_b32 v22, s6 // (op 1311) 
v_mov_b32 v23, s7 // (op 1311) 
v_add_co_u32 v16, s[0:1], v16, v22 // (op 1311) least significant half
v_addc_co_u32 v17, s[0:1], v17, v23, s[0:1] // (op 1311) most significant half





v_mul_lo_u32 v23, v17, 6 // (op 1311) most significant: low of high * low
v_mul_hi_u32 v15, v16, 6 // (op 1311) most significant: high of low * low
v_mul_lo_u32 v22, v16, 6 // (op 1311) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 1311) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[22:23] // (op 1311) 

v_mov_b32 v15, v16 // (op 1311) convert

s_mul_i32 s1, s23, 6 // (op 1313) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1313) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1313) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1313) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1313) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1313) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1313) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1313) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1313) most significant: sum
s_add_u32 s7, s7, s15 // (op 1313) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1313) 

s_mov_b32 s48, s26 // (op 1313) reg expression
s_mov_b32 s49, s27 // (op 1313) reg expression
s_mov_b32 s50, s0 // (op 1313) reg expression
s_mov_b32 s51, 131072 // (op 1313) call()

v_and_b32 v16, -16, v1 // (op 1356) 
v_sub_u32 v17, v1, v16 // (op 1356) 

v_lshl_add_u32 v16, s16, 4, v17 // (op 1356) 

v_lshrrev_b32 v17, 4, v1 // (op 1356) 
v_mul_lo_u32 v22, v17, s28 // (op 1356) 

v_add_u32 v17, v16, v22 // (op 1356) 


s_mul_i32 s1, s28, s33 // (op 1358) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1358) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1358) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1358) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1358) most significant: sum
s_add_u32 s1, s1, s7 // (op 1358) most significant: sum


s_mov_b32 s52, s30 // (op 1358) reg expression
s_mov_b32 s53, s31 // (op 1358) reg expression
s_mov_b32 s54, s0 // (op 1358) reg expression
s_mov_b32 s55, 131072 // (op 1358) call()

v_and_b32 v16, -16, v1 // (op 1375) 
v_sub_u32 v22, v1, v16 // (op 1375) 

v_lshl_add_u32 v16, s16, 4, v22 // (op 1375) 

v_lshrrev_b32 v22, 4, v1 // (op 1375) 
v_mul_lo_u32 v23, v22, s28 // (op 1375) 

v_add_u32 v22, v16, v23 // (op 1375) 


s_mul_i32 s1, s28, s33 // (op 1377) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1377) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1377) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1377) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1377) most significant: sum
s_add_u32 s1, s1, s7 // (op 1377) most significant: sum


s_mov_b32 s52, s30 // (op 1377) reg expression
s_mov_b32 s53, s31 // (op 1377) reg expression
s_mov_b32 s54, s0 // (op 1377) reg expression
s_mov_b32 s55, 131072 // (op 1377) call()

v_and_b32 v16, -16, v1 // (op 1420) 
v_sub_u32 v23, v1, v16 // (op 1420) 

v_lshlrev_b32 v16, 2, v23 // (op 1420) 

v_lshl_add_u32 v23, s17, 6, v16 // (op 1420) 

v_lshrrev_b32 v16, 4, v1 // (op 1420) 
v_mul_lo_u32 v24, v16, s34 // (op 1420) 
v_add_u32 v24, v24, v23 // (op 1420) 


s_mul_i32 s1, s34, s33 // (op 1422) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1422) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1422) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1422) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1422) most significant: sum
s_add_u32 s1, s1, s7 // (op 1422) most significant: sum


s_mov_b32 s56, s36 // (op 1422) reg expression
s_mov_b32 s57, s37 // (op 1422) reg expression
s_mov_b32 s58, s0 // (op 1422) reg expression
s_mov_b32 s59, 131072 // (op 1422) call()

v_and_b32 v16, -16, v1 // (op 1439) 
v_sub_u32 v23, v1, v16 // (op 1439) 

v_lshlrev_b32 v16, 2, v23 // (op 1439) 

v_lshl_add_u32 v23, s17, 6, v16 // (op 1439) 

v_lshrrev_b32 v16, 4, v1 // (op 1439) 
v_mul_lo_u32 v25, v16, s34 // (op 1439) 
v_add_u32 v25, v25, v23 // (op 1439) 


s_mul_i32 s1, s34, s33 // (op 1441) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1441) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1441) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1441) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1441) most significant: sum
s_add_u32 s1, s1, s7 // (op 1441) most significant: sum


s_mov_b32 s56, s36 // (op 1441) reg expression
s_mov_b32 s57, s37 // (op 1441) reg expression
s_mov_b32 s58, s0 // (op 1441) reg expression
s_mov_b32 s59, 131072 // (op 1441) call()


s_mul_i32 s1, s25, 6 // (op 1029) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 1029) least significant
s_mul_hi_u32 s6, s24, 6 // (op 1029) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1029) most significant: sum

v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 1029) 


v_mov_b32 v16, v14 // (op 1029) reg expression
buffer_load_dwordx3 v[28:30], v16, s[48:51], 0 offen // (op 1029) Load value
buffer_load_dwordx3 v[32:34], v16, s[48:51], 0 offen offset:12 // (op 1029) Load value
buffer_load_dwordx3 v[36:38], v16, s[48:51], 0 offen offset:24 // (op 1029) Load value
buffer_load_dwordx3 v[40:42], v16, s[48:51], 0 offen offset:36 // (op 1029) Load value
v_add_u32 v16, v16, v26 // (op 1029) 
buffer_load_dwordx3 v[44:46], v16, s[48:51], 0 offen // (op 1029) Load value
buffer_load_dwordx3 v[48:50], v16, s[48:51], 0 offen offset:12 // (op 1029) Load value
buffer_load_dwordx3 v[52:54], v16, s[48:51], 0 offen offset:24 // (op 1029) Load value
buffer_load_dwordx3 v[56:58], v16, s[48:51], 0 offen offset:36 // (op 1029) Load value




v_mov_b32 v16, v24 // (op 1032) reg expression

buffer_load_dword v23, v16, s[56:59], 0 offen // (op 1032) Load value

v_lshlrev_b32 v16, 4, v1 // (op 1505) 
s_nop 0
v_readfirstlane_b32 s0, v16 // (op 1505) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 7 // (op 1183) 
v_mov_b64 v[26:27], s[6:7] // (op 1183) convert


v_mov_b64 v[60:61], v[18:19] // (op 1183) reg expression
s_mov_b32 m0, s0 // (op 1183) reg expression
buffer_load_dwordx4 v60, s[8:11], 0 offen offset: 0 lds // (op 1183) Load value direct to lds
v_add_co_u32 v60, s[6:7], v60, v26 // (op 1183) least significant half
v_addc_co_u32 v61, s[6:7], 0, v61, s[6:7] // (op 1183) most significant half

s_add_u32 m0, m0, 2048 // (op 1183) 
buffer_load_dwordx4 v60, s[8:11], 0 offen offset: 0 lds // (op 1183) Load value direct to lds





v_mov_b32 v16, v17 // (op 1038) reg expression

buffer_load_ubyte v31, v16, s[52:55], 0 offen offset: 0 // (op 1038) Load value


s_mul_i32 s1, s25, 6 // (op 1041) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 1041) least significant
s_mul_hi_u32 s6, s24, 6 // (op 1041) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1041) most significant: sum

v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 1041) 


s_lshl_b64 s[0:1], s[24:25], 8 // (op 1041) 
s_mul_i32 s7, s1, 6 // (op 1041) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 1041) least significant
s_mul_hi_u32 s14, s0, 6 // (op 1041) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1041) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 1041) 

v_add_u32 v16, v14, s0 // (op 1041) 

buffer_load_dwordx3 v[60:62], v16, s[48:51], 0 offen // (op 1041) Load value
buffer_load_dwordx3 v[64:66], v16, s[48:51], 0 offen offset:12 // (op 1041) Load value
buffer_load_dwordx3 v[68:70], v16, s[48:51], 0 offen offset:24 // (op 1041) Load value
buffer_load_dwordx3 v[72:74], v16, s[48:51], 0 offen offset:36 // (op 1041) Load value
v_add_u32 v16, v16, v26 // (op 1041) 
buffer_load_dwordx3 v[76:78], v16, s[48:51], 0 offen // (op 1041) Load value
buffer_load_dwordx3 v[80:82], v16, s[48:51], 0 offen offset:12 // (op 1041) Load value
buffer_load_dwordx3 v[84:86], v16, s[48:51], 0 offen offset:24 // (op 1041) Load value
buffer_load_dwordx3 v[88:90], v16, s[48:51], 0 offen offset:36 // (op 1041) Load value





s_lshl_b32 s0, s34, 3 // (op 1044) 
v_add_u32 v14, v24, s0 // (op 1044) 


buffer_load_dword v16, v14, s[56:59], 0 offen // (op 1044) Load value


v_lshlrev_b32 v14, 4, v1 // (op 1731) 
s_nop 0
v_readfirstlane_b32 s0, v14 // (op 1731) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 7 // (op 1188) 
v_mov_b64 v[26:27], s[6:7] // (op 1188) convert


s_lshl_b64 s[6:7], s[12:13], 8 // (op 1188) 
v_mov_b32 v14, s6 // (op 1188) 
v_mov_b32 v24, s7 // (op 1188) 
v_add_co_u32 v92, s[14:15], v18, v14 // (op 1188) least significant half
v_addc_co_u32 v93, s[14:15], v19, v24, s[14:15] // (op 1188) most significant half




s_add_u32 s1, 4096, s0 // (op 1188) 
s_mov_b32 m0, s1 // (op 1188) reg expression
buffer_load_dwordx4 v92, s[8:11], 0 offen offset: 0 lds // (op 1188) Load value direct to lds
v_add_co_u32 v92, s[6:7], v92, v26 // (op 1188) least significant half
v_addc_co_u32 v93, s[6:7], 0, v93, s[6:7] // (op 1188) most significant half

s_add_u32 m0, m0, 2048 // (op 1188) 
buffer_load_dwordx4 v92, s[8:11], 0 offen offset: 0 lds // (op 1188) Load value direct to lds







s_lshl_b32 s0, s28, 3 // (op 1050) 
v_add_u32 v14, v17, s0 // (op 1050) 


buffer_load_ubyte v24, v14, s[52:55], 0 offen offset: 0 // (op 1050) Load value


v_lshlrev_b32 v14, 7, v1 // (op 1558) 
v_mul_lo_u32 v35, v14, 6 // (op 1558) 
v_lshrrev_b32 v39, 3, v35 // (op 1558) 

v_lshrrev_b32 v35, 4, v14 // (op 1558) 

v_lshlrev_b32 v14, 2, v35 // (op 1558) 

v_add_u32 v35, v39, v14 // (op 1558) 





s_waitcnt vmcnt(23)
ds_write_b96 v35, v[28:30] offset:8192 // (op 1052) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v35, v[32:34] offset:8208 // (op 1052) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v35, v[36:38] offset:8224 // (op 1052) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v35, v[40:42] offset:8240 // (op 1052) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v35, v[44:46] offset:8256 // (op 1052) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v35, v[48:50] offset:8272 // (op 1052) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v35, v[52:54] offset:8288 // (op 1052) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v35, v[56:58] offset:8304 // (op 1052) Store local data 

v_and_b32 v14, -16, v1 // (op 1669) 
v_sub_u32 v35, v1, v14 // (op 1669) 

v_lshlrev_b32 v14, 2, v35 // (op 1669) 

v_lshrrev_b32 v35, 4, v1 // (op 1669) 
v_lshl_add_u32 v39, v35, 6, v14 // (op 1669) 





s_waitcnt vmcnt(15)
ds_write_b32 v39, v23 offset:24576 // (op 1053) Store local data 

v_and_b32 v14, -16, v1 // (op 1616) 
v_sub_u32 v35, v1, v14 // (op 1616) 

v_and_b32 v14, 4294967280, v1 // (op 1616) 
v_add_u32 v39, v14, v35 // (op 1616) 





s_waitcnt vmcnt(12)
ds_write_b8 v39, v31 offset:25088 // (op 1055) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1895) 



ds_read_b96_tr_b6 v[92:94], v5 offset:8192 // (op 1073) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v5 offset:9216 // (op 1073) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1936) 


ds_read_u8 v14, v7 offset:24576 // (op 1086) Load local data 



ds_read_b96_tr_b6 v[98:100], v5 offset:8224 // (op 1078) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v5 offset:9248 // (op 1078) Transpose load local data 


ds_read_u8 v35, v7 offset:24608 // (op 1091) Load local data 
s_barrier  // (op 1876) 



ds_read_b64_tr_b8 v[104:105], v0 // (op 1069) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v0 offset:128 // (op 1069) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v0 offset:1024 // (op 1069) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v0 offset:1152 // (op 1069) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1916) 


ds_read_u8 v39, v4 offset:25088 // (op 1082) Load local data 
v_lshlrev_b32 v43, 4, v1 // (op 1477) 
s_nop 0
v_readfirstlane_b32 s0, v43 // (op 1477) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v43, 7, v1 // (op 1529) 
v_mul_lo_u32 v47, v43, 6 // (op 1529) 
v_lshrrev_b32 v51, 3, v47 // (op 1529) 

v_lshrrev_b32 v47, 4, v43 // (op 1529) 

v_lshlrev_b32 v43, 2, v47 // (op 1529) 

v_add_u32 v47, v51, v43 // (op 1529) 


v_and_b32 v43, -16, v1 // (op 1587) 
v_sub_u32 v51, v1, v43 // (op 1587) 

v_and_b32 v43, 4294967280, v1 // (op 1587) 
v_add_u32 v55, v43, v51 // (op 1587) 


v_and_b32 v43, -16, v1 // (op 1640) 
v_sub_u32 v51, v1, v43 // (op 1640) 

v_lshlrev_b32 v43, 2, v51 // (op 1640) 

v_lshrrev_b32 v51, 4, v1 // (op 1640) 
v_lshl_add_u32 v59, v51, 6, v43 // (op 1640) 


v_lshlrev_b32 v43, 4, v1 // (op 1716) 
s_nop 0
v_readfirstlane_b32 s1, v43 // (op 1716) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v43, 7, v1 // (op 1755) 
v_mul_lo_u32 v51, v43, 6 // (op 1755) 
v_lshrrev_b32 v63, 3, v51 // (op 1755) 

v_lshrrev_b32 v51, 4, v43 // (op 1755) 

v_lshlrev_b32 v43, 2, v51 // (op 1755) 

v_add_u32 v51, v63, v43 // (op 1755) 


v_and_b32 v43, -16, v1 // (op 1784) 
v_sub_u32 v63, v1, v43 // (op 1784) 

v_and_b32 v43, 4294967280, v1 // (op 1784) 
v_add_u32 v67, v43, v63 // (op 1784) 


v_and_b32 v43, -16, v1 // (op 1808) 
v_sub_u32 v63, v1, v43 // (op 1808) 

v_lshlrev_b32 v43, 2, v63 // (op 1808) 

v_lshrrev_b32 v63, 4, v1 // (op 1808) 
v_lshl_add_u32 v71, v63, 6, v43 // (op 1808) 


s_mov_b32 s6, 0 // (op 400) call()
s_ashr_i64 s[14:15], s[22:23], 8 // (op 399) 
s_mov_b32 s7, s14 // (op 399) convert

s_and_b32 s14, s7, 4294967294 // (op 399) 

s_cmp_lt_u32 s6, s14 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399 // (op 399) Condition: Top (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399 if false)
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399:

s_add_u32 s6, s6, 2 // (op 401) 
s_mul_i32 s15, s25, 6 // (op 10) most significant: low of high * low
s_mul_i32 s14, s24, 6 // (op 10) least significant
s_mul_hi_u32 s7, s24, 6 // (op 10) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 10) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[14:15] // (op 10) 


v_mov_b32 v17, v15 // (op 10) 
s_lshl_b64 s[14:15], s[24:25], 8 // (op 10) 
s_mul_i32 s19, s15, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 10) least significant
s_mul_hi_u32 s7, s14, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 10) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s7, s14, 1 // (op 10) 

v_add_u32 v43, v17, s7 // (op 10) 

buffer_load_dwordx3 v[28:30], v43, s[48:51], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[32:34], v43, s[48:51], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[36:38], v43, s[48:51], 0 offen offset:24 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:111], v[92:97], a[0:3], v39, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 88) 
buffer_load_dwordx3 v[40:42], v43, s[48:51], 0 offen offset:36 // (op 10) Load value
s_barrier  // (op 1885) 
v_add_u32 v43, v43, v18 // (op 10) 



buffer_load_dwordx3 v[44:46], v43, s[48:51], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[48:50], v43, s[48:51], 0 offen offset:12 // (op 10) Load value
ds_read_b64_tr_b8 v[112:113], v0 offset:2048 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[114:115], v0 offset:2176 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[116:117], v0 offset:3072 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[118:119], v0 offset:3200 // (op 91) Transpose load local data 
buffer_load_dwordx3 v[52:54], v43, s[48:51], 0 offen offset:24 // (op 10) Load value
s_barrier  // (op 1905) 
buffer_load_dwordx3 v[56:58], v43, s[48:51], 0 offen offset:36 // (op 10) Load value







ds_read_b96_tr_b6 v[92:94], v5 offset:16384 // (op 95) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v5 offset:17408 // (op 95) Transpose load local data 
v_mov_b32 v43, v25 // (op 25) 
s_barrier  // (op 1926) 
s_lshl_b32 s7, s34, 4 // (op 25) 


ds_read_u8 v63, v4 offset:25152 // (op 99) Load local data 
v_add_u32 v75, v43, s7 // (op 25) 


s_barrier  // (op 1946) 
buffer_load_dword v23, v75, s[56:59], 0 offen // (op 25) Load value



s_waitcnt lgkmcnt(3)
s_barrier  // (op 1879) 
ds_read_u8 v14, v7 offset:24832 // (op 103) Load local data 

s_lshl_b64 s[14:15], s[12:13], 7 // (op 1167) 
v_mov_b64 v[18:19], s[14:15] // (op 1167) convert


v_mov_b64 v[26:27], v[20:21] // (op 1167) 
s_lshl_b64 s[14:15], s[12:13], 8 // (op 1167) 
s_lshl_b64 s[18:19], s[14:15], 1 // (op 1167) 

v_mov_b32 v75, s18 // (op 1167) 
v_mov_b32 v79, s19 // (op 1167) 
v_add_co_u32 v120, s[14:15], v26, v75 // (op 1167) least significant half
v_addc_co_u32 v121, s[14:15], v27, v79, s[14:15] // (op 1167) most significant half




s_mov_b32 m0, s0 // (op 1167) reg expression
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:119], v[92:97], a[0:3], v63, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 105) 
buffer_load_dwordx4 v120, s[8:11], 0 offen offset: 0 lds // (op 1167) Load value direct to lds
v_add_co_u32 v120, s[14:15], v120, v18 // (op 1167) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:111], v[98:103], a[4:7], v39, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 447) 
v_addc_co_u32 v121, s[14:15], 0, v121, s[14:15] // (op 1167) most significant half




s_add_u32 m0, m0, 2048 // (op 1167) 
ds_read_b96_tr_b6 v[98:100], v5 offset:16416 // (op 451) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v5 offset:17440 // (op 451) Transpose load local data 
buffer_load_dwordx4 v120, s[8:11], 0 offen offset: 0 lds // (op 1167) Load value direct to lds






ds_read_u8 v35, v7 offset:24864 // (op 455) Load local data 
v_mov_b32 v75, v22 // (op 17) 
s_lshl_b32 s7, s28, 4 // (op 17) 
v_add_u32 v79, v75, s7 // (op 17) 


buffer_load_ubyte v31, v79, s[52:55], 0 offen offset: 0 // (op 17) Load value

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:119], v[98:103], a[4:7], v63, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 456) 


s_barrier  // (op 1966) 



s_waitcnt vmcnt(23)
ds_write_b96 v51, v[60:62] offset:25216 // (op 302) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v51, v[64:66] offset:25232 // (op 302) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v51, v[68:70] offset:25248 // (op 302) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v51, v[72:74] offset:25264 // (op 302) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v51, v[76:78] offset:25280 // (op 302) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v51, v[80:82] offset:25296 // (op 302) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v51, v[84:86] offset:25312 // (op 302) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v51, v[88:90] offset:25328 // (op 302) Store local data 
s_barrier  // (op 1978) 



s_waitcnt vmcnt(15)
ds_write_b32 v71, v16 offset:41600 // (op 304) Store local data 
s_barrier  // (op 1972) 



s_waitcnt vmcnt(12)
ds_write_b8 v67, v24 offset:42112 // (op 303) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1962) 



ds_read_b96_tr_b6 v[92:94], v9 offset:25216 // (op 287) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v9 offset:26240 // (op 287) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1975) 


ds_read_u8 v14, v10 offset:41600 // (op 291) Load local data 



ds_read_b96_tr_b6 v[112:114], v9 offset:25248 // (op 470) Transpose load local data 
ds_read_b96_tr_b6 v[115:117], v9 offset:26272 // (op 470) Transpose load local data 


ds_read_u8 v63, v10 offset:41632 // (op 474) Load local data 
s_barrier  // (op 1956) 



ds_read_b64_tr_b8 v[104:105], v3 offset:4096 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v3 offset:4224 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v3 offset:5120 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v3 offset:5248 // (op 285) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1969) 


ds_read_u8 v39, v6 offset:42112 // (op 289) Load local data 

s_mul_i32 s15, s25, 6 // (op 305) most significant: low of high * low
s_mul_i32 s14, s24, 6 // (op 305) least significant
s_mul_hi_u32 s7, s24, 6 // (op 305) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 305) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[14:15] // (op 305) 


s_lshl_b64 s[14:15], s[24:25], 8 // (op 305) 
s_mul_i32 s19, s15, 6 // (op 305) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 305) least significant
s_mul_hi_u32 s7, s14, 6 // (op 305) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 305) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 305) 

s_mul_i32 s7, 3, s14 // (op 305) 

v_add_u32 v79, v17, s7 // (op 305) 

buffer_load_dwordx3 v[60:62], v79, s[48:51], 0 offen // (op 305) Load value
buffer_load_dwordx3 v[64:66], v79, s[48:51], 0 offen offset:12 // (op 305) Load value
buffer_load_dwordx3 v[68:70], v79, s[48:51], 0 offen offset:24 // (op 305) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:111], v[92:97], a[0:3], v39, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 292) 
buffer_load_dwordx3 v[72:74], v79, s[48:51], 0 offen offset:36 // (op 305) Load value
v_add_u32 v79, v79, v18 // (op 305) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:111], v[112:117], a[4:7], v39, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 457) 
buffer_load_dwordx3 v[76:78], v79, s[48:51], 0 offen // (op 305) Load value


buffer_load_dwordx3 v[80:82], v79, s[48:51], 0 offen offset:12 // (op 305) Load value
ds_read_b64_tr_b8 v[112:113], v3 offset:6144 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[114:115], v3 offset:6272 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[116:117], v3 offset:7168 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[118:119], v3 offset:7296 // (op 294) Transpose load local data 
buffer_load_dwordx3 v[84:86], v79, s[48:51], 0 offen offset:24 // (op 305) Load value
buffer_load_dwordx3 v[88:90], v79, s[48:51], 0 offen offset:36 // (op 305) Load value






ds_read_b96_tr_b6 v[120:122], v9 offset:33408 // (op 296) Transpose load local data 


ds_read_b96_tr_b6 v[123:125], v9 offset:34432 // (op 296) Transpose load local data 
s_lshl_b32 s7, s34, 3 // (op 307) 


ds_read_u8 v63, v6 offset:42176 // (op 298) Load local data 
s_mul_i32 s14, 3, s7 // (op 307) 

v_add_u32 v79, v43, s14 // (op 307) 




s_barrier  // (op 2017) 
buffer_load_dword v16, v79, s[56:59], 0 offen // (op 307) Load value

s_lshl_b64 s[14:15], s[24:25], 8 // (op 1303) 

s_mul_i32 s19, s15, 6 // (op 1303) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 1303) least significant
s_mul_hi_u32 s7, s14, 6 // (op 1303) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 1303) most significant: sum


s_lshl_b32 s7, s34, 4 // (op 1431) 
v_add_u32 v25, v25, s7 // (op 1431) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 1303) 

s_lshl_b32 s7, s14, 1 // (op 1303) 

v_add_u32 v15, v15, s7 // (op 1303) 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1959) 
ds_read_u8 v43, v10 offset:41856 // (op 300) Load local data 




s_lshl_b64 s[14:15], s[12:13], 7 // (op 1178) 
v_mov_b64 v[18:19], s[14:15] // (op 1178) convert


s_lshl_b64 s[14:15], s[12:13], 8 // (op 1178) 
s_mul_i32 s19, 3, s15 // (op 1178) most significant: low of low * high
s_mul_i32 s18, 3, s14 // (op 1178) least significant
s_mul_hi_u32 s7, 3, s14 // (op 1178) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 1178) most significant: sum


v_mov_b32 v17, s18 // (op 1178) 
v_mov_b32 v79, s19 // (op 1178) 
v_add_co_u32 v126, s[14:15], v26, v17 // (op 1178) least significant half
v_addc_co_u32 v127, s[14:15], v27, v79, s[14:15] // (op 1178) most significant half




s_add_u32 s7, 4096, s1 // (op 1178) 
s_mov_b32 m0, s7 // (op 1178) reg expression
ds_read_b96_tr_b6 v[128:130], v9 offset:33440 // (op 461) Transpose load local data 
ds_read_b96_tr_b6 v[131:133], v9 offset:34464 // (op 461) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:119], v[120:125], a[0:3], v63, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 301) 




s_barrier  // (op 2055) 
ds_read_u8 v17, v10 offset:41888 // (op 465) Load local data 
buffer_load_dwordx4 v126, s[8:11], 0 offen offset: 0 lds // (op 1178) Load value direct to lds
v_add_co_u32 v126, s[14:15], v126, v18 // (op 1178) least significant half
s_barrier  // (op 2059) 
v_addc_co_u32 v127, s[14:15], 0, v127, s[14:15] // (op 1178) most significant half

s_add_u32 m0, m0, 2048 // (op 1178) 
buffer_load_dwordx4 v126, s[8:11], 0 offen offset: 0 lds // (op 1178) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:119], v[128:133], a[4:7], v63, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 466) 







s_lshl_b32 s7, s28, 3 // (op 306) 
s_mul_i32 s14, 3, s7 // (op 306) 

v_add_u32 v43, v75, s14 // (op 306) 


buffer_load_ubyte v24, v43, s[52:55], 0 offen offset: 0 // (op 306) Load value

s_lshl_b64 s[14:15], s[12:13], 8 // (op 1239) 

s_lshl_b64 s[18:19], s[14:15], 1 // (op 1239) 

s_barrier  // (op 1898) 
s_lshl_b32 s7, s28, 4 // (op 1367) 



v_add_u32 v22, v22, s7 // (op 1367) 

v_mov_b32 v43, s18 // (op 1239) 
v_mov_b32 v63, s19 // (op 1239) 
v_add_co_u32 v20, s[14:15], v20, v43 // (op 1239) least significant half
v_addc_co_u32 v21, s[14:15], v21, v63, s[14:15] // (op 1239) most significant half




s_waitcnt vmcnt(23)
ds_write_b96 v47, v[28:30] offset:8192 // (op 13) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v47, v[32:34] offset:8208 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v47, v[36:38] offset:8224 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v47, v[40:42] offset:8240 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v47, v[44:46] offset:8256 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v47, v[48:50] offset:8272 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v47, v[52:54] offset:8288 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v47, v[56:58] offset:8304 // (op 13) Store local data 
s_barrier  // (op 1939) 



s_waitcnt vmcnt(15)
ds_write_b32 v59, v23 offset:24576 // (op 30) Store local data 
s_barrier  // (op 1919) 



s_waitcnt vmcnt(12)
ds_write_b8 v55, v31 offset:25088 // (op 21) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1908) 



ds_read_b96_tr_b6 v[92:94], v5 offset:8192 // (op 75) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v5 offset:9216 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1949) 


ds_read_u8 v14, v7 offset:24576 // (op 85) Load local data 



ds_read_b96_tr_b6 v[98:100], v5 offset:8224 // (op 442) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v5 offset:9248 // (op 442) Transpose load local data 


ds_read_u8 v35, v7 offset:24608 // (op 446) Load local data 
s_barrier  // (op 1888) 



ds_read_b64_tr_b8 v[104:105], v0 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v0 offset:128 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v0 offset:1024 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v0 offset:1152 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1929) 


ds_read_u8 v39, v4 offset:25088 // (op 80) Load local data 
s_ashr_i64 s[14:15], s[22:23], 8 // (op 399) 
s_mov_b32 s7, s14 // (op 399) convert

s_and_b32 s14, s7, 4294967294 // (op 399) 

s_cmp_lt_u32 s6, s14 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399 // (op 399) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399 if true)
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399:
















s_barrier  // (op 2037) 
s_ashr_i64 s[0:1], s[22:23], 8 // (op 1201) 
s_lshr_b32 s6, s0, 1 // (op 1201) 

v_lshl_add_u32 v15, s6, 9, v1 // (op 1201) 

s_lshl_b32 s0, s16, 4 // (op 1201) 
v_mul_lo_u32 v23, v15, s13 // (op 1201) most significant: low of low * high
v_mul_hi_u32 v63, v15, s12 // (op 1201) most significant: high of low * low
v_mul_lo_u32 v22, v15, s12 // (op 1201) least significant: low of low * low
v_add_u32 v23, v23, v63 // (op 1201) most significant: sum


s_mov_b32 s7, 0 // (op 1201) convert
s_mov_b32 s6, s0 // (op 1201) convert

v_mov_b32 v15, s6 // (op 1201) 
v_mov_b32 v63, s7 // (op 1201) 
v_add_co_u32 v20, s[0:1], v15, v22 // (op 1201) least significant half
v_addc_co_u32 v21, s[0:1], v63, v23, s[0:1] // (op 1201) most significant half





s_mul_i32 s1, s12, s23 // (op 1203) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1203) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1203) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1203) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1203) most significant: sum
s_add_u32 s1, s1, s7 // (op 1203) most significant: sum


s_mov_b32 s8, s20 // (op 1203) reg expression
s_mov_b32 s9, s21 // (op 1203) reg expression
s_mov_b32 s10, s0 // (op 1203) reg expression
s_mov_b32 s11, 131072 // (op 1203) call()

v_lshlrev_b32 v15, 1, v1 // (op 1269) 
s_ashr_i64 s[0:1], s[22:23], 8 // (op 1269) 
s_lshr_b32 s6, s0, 1 // (op 1269) 

v_lshl_add_u32 v63, s6, 9, v15 // (op 1269) 


s_lshl_b32 s0, s17, 6 // (op 1269) 
s_mov_b32 s7, 0 // (op 1269) convert
s_mov_b32 s6, s0 // (op 1269) convert

v_mul_lo_u32 v19, v63, s25 // (op 1269) most significant: low of low * high
v_mul_hi_u32 v15, v63, s24 // (op 1269) most significant: high of low * low
v_mul_lo_u32 v18, v63, s24 // (op 1269) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 1269) most significant: sum

v_mov_b32 v15, s6 // (op 1269) 
v_mov_b32 v17, s7 // (op 1269) 
v_add_co_u32 v18, s[0:1], v18, v15 // (op 1269) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1269) most significant half





v_mul_lo_u32 v23, v19, 6 // (op 1269) most significant: low of high * low
v_mul_hi_u32 v15, v18, 6 // (op 1269) most significant: high of low * low
v_mul_lo_u32 v22, v18, 6 // (op 1269) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 1269) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[22:23] // (op 1269) 

v_mov_b32 v15, v18 // (op 1269) convert

s_mul_i32 s1, s23, 6 // (op 1271) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1271) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1271) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1271) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1271) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1271) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1271) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1271) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1271) most significant: sum
s_add_u32 s7, s7, s15 // (op 1271) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1271) 

s_mov_b32 s48, s26 // (op 1271) reg expression
s_mov_b32 s49, s27 // (op 1271) reg expression
s_mov_b32 s50, s0 // (op 1271) reg expression
s_mov_b32 s51, 131072 // (op 1271) call()

v_and_b32 v63, -16, v1 // (op 1333) 
s_ashr_i64 s[0:1], s[22:23], 8 // (op 1333) 
v_sub_u32 v67, v1, v63 // (op 1333) 

s_lshr_b32 s6, s0, 1 // (op 1333) 

v_lshl_add_u32 v63, s16, 4, v67 // (op 1333) 

v_lshrrev_b32 v67, 4, v1 // (op 1333) 
v_lshl_add_u32 v71, s6, 4, v67 // (op 1333) 


v_mul_lo_u32 v67, v71, s28 // (op 1333) 

v_add_u32 v71, v63, v67 // (op 1333) 


s_mul_i32 s1, s28, s33 // (op 1335) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1335) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1335) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1335) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1335) most significant: sum
s_add_u32 s1, s1, s7 // (op 1335) most significant: sum


s_mov_b32 s52, s30 // (op 1335) reg expression
s_mov_b32 s53, s31 // (op 1335) reg expression
s_mov_b32 s54, s0 // (op 1335) reg expression
s_mov_b32 s55, 131072 // (op 1335) call()

v_lshrrev_b32 v63, 4, v1 // (op 1397) 
s_ashr_i64 s[0:1], s[22:23], 8 // (op 1397) 
v_and_b32 v67, -16, v1 // (op 1397) 
s_lshr_b32 s6, s0, 1 // (op 1397) 

v_sub_u32 v75, v1, v67 // (op 1397) 

v_lshl_add_u32 v67, s6, 4, v63 // (op 1397) 


v_lshlrev_b32 v63, 2, v75 // (op 1397) 

v_lshl_add_u32 v75, s17, 6, v63 // (op 1397) 

v_mul_lo_u32 v63, v67, s34 // (op 1397) 
v_add_u32 v63, v63, v75 // (op 1397) 


s_mul_i32 s1, s34, s33 // (op 1399) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1399) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1399) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1399) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1399) most significant: sum
s_add_u32 s1, s1, s7 // (op 1399) most significant: sum


s_mov_b32 s56, s36 // (op 1399) reg expression
s_mov_b32 s57, s37 // (op 1399) reg expression
s_mov_b32 s58, s0 // (op 1399) reg expression
s_mov_b32 s59, 131072 // (op 1399) call()

v_lshlrev_b32 v67, 4, v1 // (op 1490) 
s_nop 0
v_readfirstlane_b32 s0, v67 // (op 1490) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v67, 7, v1 // (op 1542) 
v_mul_lo_u32 v75, v67, 6 // (op 1542) 
v_lshrrev_b32 v79, 3, v75 // (op 1542) 

v_lshrrev_b32 v75, 4, v67 // (op 1542) 

v_lshlrev_b32 v67, 2, v75 // (op 1542) 

v_add_u32 v75, v79, v67 // (op 1542) 


v_and_b32 v67, -16, v1 // (op 1600) 
v_sub_u32 v79, v1, v67 // (op 1600) 

v_and_b32 v67, 4294967280, v1 // (op 1600) 
v_add_u32 v83, v67, v79 // (op 1600) 


v_and_b32 v67, -16, v1 // (op 1653) 
v_sub_u32 v79, v1, v67 // (op 1653) 

v_lshlrev_b32 v67, 2, v79 // (op 1653) 

v_lshrrev_b32 v79, 4, v1 // (op 1653) 
v_lshl_add_u32 v87, v79, 6, v67 // (op 1653) 


s_ashr_i64 s[6:7], s[22:23], 8 // (op 405) 
s_and_b32 s1, s6, 4294967294 // (op 405) 

s_ashr_i64 s[6:7], s[22:23], 8 // (op 404) 
s_mov_b32 s14, s6 // (op 404) convert

s_cmp_lt_u32 s1, s14 // (op 404) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404 // (op 404) Condition: Top (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404:
s_barrier  // (op 1882) 
s_add_u32 s1, s1, 1 // (op 406) 

s_lshl_b64 s[6:7], s[12:13], 7 // (op 1172) 
v_mov_b64 v[16:17], s[6:7] // (op 1172) convert


v_mov_b64 v[18:19], v[20:21] // (op 1172) 
v_mov_b64 v[22:23], v[18:19] // (op 1172) reg expression
s_mov_b32 m0, s0 // (op 1172) reg expression
buffer_load_dwordx4 v22, s[8:11], 0 offen offset: 0 lds // (op 1172) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v16 // (op 1172) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 1172) most significant half

s_add_u32 m0, m0, 2048 // (op 1172) 
buffer_load_dwordx4 v22, s[8:11], 0 offen offset: 0 lds // (op 1172) Load value direct to lds



s_lshl_b64 s[6:7], s[12:13], 8 // (op 1191) 
v_mov_b32 v14, s6 // (op 1191) 
v_mov_b32 v16, s7 // (op 1191) 
v_add_co_u32 v20, s[14:15], v20, v14 // (op 1191) least significant half
v_addc_co_u32 v21, s[14:15], v21, v16, s[14:15] // (op 1191) most significant half





s_mul_i32 s7, s25, 6 // (op 221) most significant: low of high * low
s_mul_i32 s6, s24, 6 // (op 221) least significant
s_mul_hi_u32 s14, s24, 6 // (op 221) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 221) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[6:7] // (op 221) 


v_mov_b32 v14, v15 // (op 221) 
v_mov_b32 v18, v14 // (op 221) reg expression
buffer_load_dwordx3 v[22:24], v18, s[48:51], 0 offen // (op 221) Load value
buffer_load_dwordx3 v[26:28], v18, s[48:51], 0 offen offset:12 // (op 221) Load value
buffer_load_dwordx3 v[30:32], v18, s[48:51], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx3 v[34:36], v18, s[48:51], 0 offen offset:36 // (op 221) Load value
v_add_u32 v18, v18, v16 // (op 221) 
buffer_load_dwordx3 v[38:40], v18, s[48:51], 0 offen // (op 221) Load value
buffer_load_dwordx3 v[42:44], v18, s[48:51], 0 offen offset:12 // (op 221) Load value
buffer_load_dwordx3 v[46:48], v18, s[48:51], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx3 v[50:52], v18, s[48:51], 0 offen offset:36 // (op 221) Load value



s_lshl_b64 s[6:7], s[24:25], 8 // (op 1261) 
s_mul_i32 s15, s7, 6 // (op 1261) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 1261) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1261) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 1261) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 1261) 

v_add_u32 v15, v15, s6 // (op 1261) 



s_barrier  // (op 1901) 
v_mov_b32 v14, v71 // (op 222) 



v_mov_b32 v25, v14 // (op 222) reg expression

buffer_load_ubyte v29, v25, s[52:55], 0 offen offset: 0 // (op 222) Load value


s_lshl_b32 s6, s28, 3 // (op 1325) 
v_add_u32 v71, v71, s6 // (op 1325) 

s_waitcnt vmcnt(8)
ds_write_b96 v75, v[22:24] offset:8192 // (op 218) Store local data 


v_mov_b32 v14, v63 // (op 223) 
v_mov_b32 v25, v14 // (op 223) reg expression

buffer_load_dword v33, v25, s[56:59], 0 offen // (op 223) Load value


s_lshl_b32 s6, s34, 3 // (op 1389) 
v_add_u32 v63, v63, s6 // (op 1389) 

s_waitcnt vmcnt(8)
ds_write_b96 v75, v[26:28] offset:8208 // (op 218) Store local data 
s_barrier  // (op 1891) 
s_waitcnt vmcnt(7)
ds_write_b96 v75, v[30:32] offset:8224 // (op 218) Store local data 



ds_read_b64_tr_b8 v[54:55], v0 // (op 201) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:128 // (op 201) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:1024 // (op 201) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:1152 // (op 201) Transpose load local data 
s_waitcnt vmcnt(6)
ds_write_b96 v75, v[34:36] offset:8240 // (op 218) Store local data 
s_waitcnt vmcnt(5)
ds_write_b96 v75, v[38:40] offset:8256 // (op 218) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v75, v[42:44] offset:8272 // (op 218) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v75, v[46:48] offset:8288 // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v75, v[50:52] offset:8304 // (op 218) Store local data 

s_barrier  // (op 1922) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1912) 






ds_read_b96_tr_b6 v[22:24], v5 offset:8192 // (op 203) Transpose load local data 
ds_read_b96_tr_b6 v[25:27], v5 offset:9216 // (op 203) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v83, v29 offset:25088 // (op 219) Store local data 

s_barrier  // (op 1942) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1932) 





ds_read_u8 v14, v4 offset:25088 // (op 205) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v87, v33 offset:24576 // (op 220) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1952) 


ds_read_u8 v62, v7 offset:24576 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:61], v[22:27], a[0:3], v14, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 208) 





ds_read_b64_tr_b8 v[22:23], v0 offset:2048 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:2176 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:3072 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:3200 // (op 210) Transpose load local data 



ds_read_b96_tr_b6 v[30:32], v5 offset:16384 // (op 212) Transpose load local data 
ds_read_b96_tr_b6 v[33:35], v5 offset:17408 // (op 212) Transpose load local data 


ds_read_u8 v62, v4 offset:25152 // (op 214) Load local data 


ds_read_u8 v16, v7 offset:24832 // (op 216) Load local data 



ds_read_b96_tr_b6 v[36:38], v5 offset:8224 // (op 412) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v5 offset:9248 // (op 412) Transpose load local data 


ds_read_u8 v17, v7 offset:24608 // (op 416) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[36:41], a[4:7], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 417) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:35], a[0:3], v62, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 217) 
ds_read_b96_tr_b6 v[36:38], v5 offset:16416 // (op 421) Transpose load local data 


ds_read_b96_tr_b6 v[39:41], v5 offset:17440 // (op 421) Transpose load local data 


ds_read_u8 v14, v7 offset:24864 // (op 425) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:29], v[36:41], a[4:7], v62, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 426) 




s_ashr_i64 s[6:7], s[22:23], 8 // (op 404) 
s_mov_b32 s14, s6 // (op 404) convert

s_cmp_lt_u32 s1, s14 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404 // (op 404) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404 if true)
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404:























s_barrier  // (op 2041) 








v_cmp_eq_f32 vcc, 0.00000, v12 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155
v_accvgpr_read v4, a0 // (op 50) 
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v14, v13, v4 // (op 50) 
v_mul_f32 v15, v13, v5 // (op 50) 
v_mul_f32 v16, v13, v6 // (op 50) 
v_mul_f32 v17, v13, v7 // (op 50) 



v_mov_b32 v0, v11 // (op 56) reg expression

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 56) Store value


v_accvgpr_read v4, a4 // (op 710) 
v_accvgpr_read v5, a5 // (op 710) 
v_accvgpr_read v6, a6 // (op 710) 
v_accvgpr_read v7, a7 // (op 710) 
v_mul_f32 v14, v13, v4 // (op 710) 
v_mul_f32 v15, v13, v5 // (op 710) 
v_mul_f32 v16, v13, v6 // (op 710) 
v_mul_f32 v17, v13, v7 // (op 710) 



s_lshl_b32 s0, s4, 7 // (op 712) 
v_add_u32 v0, v11, s0 // (op 712) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 712) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155:


v_mov_b32 v0, v8 // (op 167) reg expression

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 167) Load value

v_accvgpr_read v14, a0 // (op 164) 
v_accvgpr_read v15, a1 // (op 164) 
v_accvgpr_read v16, a2 // (op 164) 
v_accvgpr_read v17, a3 // (op 164) 
v_mul_f32 v18, v13, v14 // (op 164) 
v_mul_f32 v19, v13, v15 // (op 164) 
v_mul_f32 v20, v13, v16 // (op 164) 
v_mul_f32 v21, v13, v17 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v12, v4 // (op 168) 
v_mul_f32 v15, v12, v5 // (op 168) 
v_mul_f32 v16, v12, v6 // (op 168) 
v_mul_f32 v17, v12, v7 // (op 168) 

v_add_f32 v4, v14, v18 // (op 165) 
v_add_f32 v5, v15, v19 // (op 165) 
v_add_f32 v6, v16, v20 // (op 165) 
v_add_f32 v7, v17, v21 // (op 165) 




v_mov_b32 v0, v11 // (op 166) reg expression

buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 166) Store value




s_lshl_b32 s0, s2, 7 // (op 734) 
v_add_u32 v0, v8, s0 // (op 734) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 734) Load value

v_accvgpr_read v14, a4 // (op 731) 
v_accvgpr_read v15, a5 // (op 731) 
v_accvgpr_read v16, a6 // (op 731) 
v_accvgpr_read v17, a7 // (op 731) 
v_mul_f32 v18, v13, v14 // (op 731) 
v_mul_f32 v19, v13, v15 // (op 731) 
v_mul_f32 v20, v13, v16 // (op 731) 
v_mul_f32 v21, v13, v17 // (op 731) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v12, v4 // (op 735) 
v_mul_f32 v15, v12, v5 // (op 735) 
v_mul_f32 v16, v12, v6 // (op 735) 
v_mul_f32 v17, v12, v7 // (op 735) 

v_add_f32 v4, v14, v18 // (op 732) 
v_add_f32 v5, v15, v19 // (op 732) 
v_add_f32 v6, v16, v20 // (op 732) 
v_add_f32 v7, v17, v21 // (op 732) 




s_lshl_b32 s0, s4, 7 // (op 733) 
v_add_u32 v0, v11, s0 // (op 733) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 733) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155:










s_endpgm  // End of RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.LRR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_end:
.size RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2, .LRR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_end-RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
  .amdhsa_next_free_vgpr 144
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 42240 // lds bytes
  .amdhsa_accum_offset 136
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
    macrotile: [16, 64, 256]
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
  - .name: RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
    .symbol: RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 42240
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 134
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

