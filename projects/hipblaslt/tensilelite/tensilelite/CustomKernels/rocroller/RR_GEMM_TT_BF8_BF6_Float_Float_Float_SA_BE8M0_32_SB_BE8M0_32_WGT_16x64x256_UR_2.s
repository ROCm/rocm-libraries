// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.p2align 8
.type RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2,@function
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 1517) 
v_and_b32 v3, 15, v1 // (op 1517) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1517) 


v_and_b32 v0, 48, v1 // (op 1517) 
v_lshl_add_u32 v3, v4, 8, v0 // (op 1517) 


v_lshrrev_b32 v0, 7, v1 // (op 1628) 
v_and_b32 v4, 15, v1 // (op 1628) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1628) 


v_and_b32 v0, 63, v1 // (op 1628) 
v_lshrrev_b32 v4, 4, v0 // (op 1628) 

v_lshl_add_u32 v0, v5, 3, v4 // (op 1628) 


v_lshrrev_b32 v4, 7, v1 // (op 1743) 
v_and_b32 v5, 15, v1 // (op 1743) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 1743) 


v_and_b32 v4, 48, v1 // (op 1743) 
v_lshl_add_u32 v5, v6, 8, v4 // (op 1743) 


v_lshrrev_b32 v4, 7, v1 // (op 1796) 
v_and_b32 v6, 15, v1 // (op 1796) 
v_lshl_add_u32 v7, v4, 4, v6 // (op 1796) 


v_and_b32 v4, 63, v1 // (op 1796) 
v_lshrrev_b32 v6, 4, v4 // (op 1796) 

v_lshl_add_u32 v4, v7, 3, v6 // (op 1796) 


v_lshrrev_b32 v6, 7, v1 // (op 1461) 
v_and_b32 v7, 63, v1 // (op 1461) 
v_lshrrev_b32 v8, 4, v7 // (op 1461) 
v_lshlrev_b32 v9, 2, v8 // (op 1461) 

v_lshl_add_u32 v8, v6, 4, v9 // (op 1461) 


v_lshl_add_u32 v6, s16, 4, v8 // (op 1461) 

v_lshrrev_b32 v8, 2, v7 // (op 1461) 

v_and_b32 v7, 3, v8 // (op 1461) 

v_and_b32 v8, 3, v1 // (op 1461) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 1461) 


v_lshrrev_b32 v7, 6, v1 // (op 1461) 
v_and_b32 v8, 1, v7 // (op 1461) 

v_lshl_add_u32 v7, v8, 4, v9 // (op 1461) 


v_lshl_add_u32 v8, s17, 6, v7 // (op 1461) 

v_mul_lo_u32 v7, v8, s2 // (op 1461) 

v_add_lshl_u32 v8, v6, v7, 2 // (op 1461) 


v_and_b32 v6, 63, v1 // (op 1575) 
v_and_b32 v7, 48, v1 // (op 1575) 
v_sub_u32 v9, v6, v7 // (op 1575) 

v_lshrrev_b32 v7, 4, v6 // (op 1575) 

v_lshl_add_u32 v6, v7, 5, v9 // (op 1575) 


v_lshrrev_b32 v7, 6, v1 // (op 1575) 
v_and_b32 v9, 1, v7 // (op 1575) 

v_lshlrev_b32 v7, 4, v9 // (op 1575) 

v_lshl_add_u32 v9, v6, 6, v7 // (op 1575) 


v_mul_lo_u32 v6, v9, 6 // (op 1575) 
v_lshrrev_b32 v7, 3, v6 // (op 1575) 

v_lshrrev_b32 v6, 4, v9 // (op 1575) 

v_lshlrev_b32 v9, 2, v6 // (op 1575) 

v_add_u32 v6, v7, v9 // (op 1575) 


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


s_mul_i32 s40, s5, s8 // (op 1704) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1704) least significant
s_mul_hi_u32 s41, s4, s8 // (op 1704) most significant: high of low * low
s_add_u32 s1, s1, s40 // (op 1704) most significant: sum
s_add_u32 s1, s1, s41 // (op 1704) most significant: sum


s_lshl_b64 s[40:41], s[0:1], 2 // (op 1704) 

s_mov_b32 s44, s6 // (op 1463) reg expression
s_mov_b32 s45, s7 // (op 1463) reg expression
s_mov_b32 s46, s18 // (op 1463) reg expression
s_mov_b32 s47, 131072 // (op 1463) call()

s_mov_b32 s48, s10 // (op 1704) reg expression

s_mov_b32 s49, s11 // (op 1704) reg expression
s_mov_b32 s50, s40 // (op 1704) reg expression
s_mov_b32 s51, 131072 // (op 1704) call()



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
v_lshrrev_b32 v14, 4, v1 // (op 1228) 
v_lshl_add_u32 v15, s16, 4, v14 // (op 1228) 

v_and_b32 v14, -16, v1 // (op 1228) 
v_sub_u32 v16, v1, v14 // (op 1228) 

v_lshlrev_b32 v14, 4, v16 // (op 1228) 

v_mov_b32 v17, 0 // (op 1228) convert
v_mov_b32 v16, v14 // (op 1228) convert

v_mul_lo_u32 v19, v15, s13 // (op 1228) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 1228) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 1228) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1228) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 1228) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1228) most significant half




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

v_lshrrev_b32 v14, 4, v1 // (op 1247) 
v_lshl_add_u32 v15, s16, 4, v14 // (op 1247) 

v_and_b32 v14, -16, v1 // (op 1247) 
v_sub_u32 v16, v1, v14 // (op 1247) 

v_lshlrev_b32 v14, 4, v16 // (op 1247) 

v_mov_b32 v17, 0 // (op 1247) convert
v_mov_b32 v16, v14 // (op 1247) convert

v_mul_lo_u32 v21, v15, s13 // (op 1247) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 1247) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 1247) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1247) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 1247) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 1247) most significant half



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

s_mul_i32 s1, s29, 6 // (op 1294) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1294) least significant
s_mul_hi_u32 s6, s28, 6 // (op 1294) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1294) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1294) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1294) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1294) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1294) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1294) most significant: sum
s_add_u32 s7, s7, s15 // (op 1294) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1294) 

s_mov_b32 s40, s26 // (op 1294) reg expression
s_mov_b32 s41, s27 // (op 1294) reg expression
s_mov_b32 s42, s0 // (op 1294) reg expression
s_mov_b32 s43, 131072 // (op 1294) call()

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

s_mul_i32 s1, s29, 6 // (op 1313) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1313) least significant
s_mul_hi_u32 s6, s28, 6 // (op 1313) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1313) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1313) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1313) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1313) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1313) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1313) most significant: sum
s_add_u32 s7, s7, s15 // (op 1313) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1313) 

s_mov_b32 s40, s26 // (op 1313) reg expression
s_mov_b32 s41, s27 // (op 1313) reg expression
s_mov_b32 s42, s0 // (op 1313) reg expression
s_mov_b32 s43, 131072 // (op 1313) call()

v_lshrrev_b32 v16, 3, v1 // (op 1356) 
v_lshl_add_u32 v17, s16, 4, v16 // (op 1356) 

v_and_b32 v16, -8, v1 // (op 1356) 
v_sub_u32 v22, v1, v16 // (op 1356) 

v_mul_lo_u32 v16, v17, s30 // (op 1356) 
v_add_u32 v16, v16, v22 // (op 1356) 


s_mul_i32 s1, s30, s23 // (op 1358) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1358) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1358) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1358) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1358) most significant: sum
s_add_u32 s1, s1, s7 // (op 1358) most significant: sum


s_mov_b32 s52, s32 // (op 1358) reg expression
s_mov_b32 s53, s33 // (op 1358) reg expression
s_mov_b32 s54, s0 // (op 1358) reg expression
s_mov_b32 s55, 131072 // (op 1358) call()

v_lshrrev_b32 v17, 3, v1 // (op 1375) 
v_lshl_add_u32 v22, s16, 4, v17 // (op 1375) 

v_and_b32 v17, -8, v1 // (op 1375) 
v_sub_u32 v23, v1, v17 // (op 1375) 

v_mul_lo_u32 v17, v22, s30 // (op 1375) 
v_add_u32 v17, v17, v23 // (op 1375) 


s_mul_i32 s1, s30, s23 // (op 1377) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1377) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1377) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1377) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1377) most significant: sum
s_add_u32 s1, s1, s7 // (op 1377) most significant: sum


s_mov_b32 s52, s32 // (op 1377) reg expression
s_mov_b32 s53, s33 // (op 1377) reg expression
s_mov_b32 s54, s0 // (op 1377) reg expression
s_mov_b32 s55, 131072 // (op 1377) call()

v_and_b32 v22, -16, v1 // (op 1420) 
v_sub_u32 v23, v1, v22 // (op 1420) 

v_lshlrev_b32 v22, 2, v23 // (op 1420) 

v_lshl_add_u32 v23, s17, 6, v22 // (op 1420) 

v_lshrrev_b32 v22, 4, v1 // (op 1420) 
v_mul_lo_u32 v24, v22, s34 // (op 1420) 
v_add_u32 v24, v24, v23 // (op 1420) 


s_mul_i32 s1, s34, s39 // (op 1422) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1422) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1422) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1422) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1422) most significant: sum
s_add_u32 s1, s1, s7 // (op 1422) most significant: sum


s_mov_b32 s56, s36 // (op 1422) reg expression
s_mov_b32 s57, s37 // (op 1422) reg expression
s_mov_b32 s58, s0 // (op 1422) reg expression
s_mov_b32 s59, 131072 // (op 1422) call()

v_and_b32 v22, -16, v1 // (op 1439) 
v_sub_u32 v23, v1, v22 // (op 1439) 

v_lshlrev_b32 v22, 2, v23 // (op 1439) 

v_lshl_add_u32 v23, s17, 6, v22 // (op 1439) 

v_lshrrev_b32 v22, 4, v1 // (op 1439) 
v_mul_lo_u32 v25, v22, s34 // (op 1439) 
v_add_u32 v25, v25, v23 // (op 1439) 


s_mul_i32 s1, s34, s39 // (op 1441) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1441) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1441) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1441) most significant: high of low * low
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

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 1029) 


v_mov_b32 v26, v14 // (op 1029) reg expression
buffer_load_dwordx3 v[28:30], v26, s[40:43], 0 offen // (op 1029) Load value
buffer_load_dwordx3 v[32:34], v26, s[40:43], 0 offen offset:12 // (op 1029) Load value
buffer_load_dwordx3 v[36:38], v26, s[40:43], 0 offen offset:24 // (op 1029) Load value
buffer_load_dwordx3 v[40:42], v26, s[40:43], 0 offen offset:36 // (op 1029) Load value
v_add_u32 v26, v26, v22 // (op 1029) 
buffer_load_dwordx3 v[44:46], v26, s[40:43], 0 offen // (op 1029) Load value
buffer_load_dwordx3 v[48:50], v26, s[40:43], 0 offen offset:12 // (op 1029) Load value
buffer_load_dwordx3 v[52:54], v26, s[40:43], 0 offen offset:24 // (op 1029) Load value
buffer_load_dwordx3 v[56:58], v26, s[40:43], 0 offen offset:36 // (op 1029) Load value




v_mov_b32 v31, v24 // (op 1032) reg expression

buffer_load_dword v35, v31, s[56:59], 0 offen // (op 1032) Load value

v_and_b32 v31, -16, v1 // (op 1505) 
v_sub_u32 v39, v1, v31 // (op 1505) 

v_lshlrev_b32 v31, 4, v39 // (op 1505) 

v_lshrrev_b32 v39, 4, v1 // (op 1505) 
v_lshl_add_u32 v43, v39, 8, v31 // (op 1505) 


s_nop 0
v_readfirstlane_b32 s0, v43 // (op 1505) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 3 // (op 1183) 
v_mov_b64 v[22:23], s[6:7] // (op 1183) convert


v_mov_b64 v[26:27], v[18:19] // (op 1183) 
s_mov_b32 m0, s0 // (op 1183) reg expression
buffer_load_dwordx4 v26, s[8:11], 0 offen offset: 0 lds // (op 1183) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 1183) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 1183) most significant half

s_add_u32 m0, m0, 2048 // (op 1183) 
buffer_load_dwordx4 v26, s[8:11], 0 offen offset: 0 lds // (op 1183) Load value direct to lds






buffer_load_ubyte v31, v16, s[52:55], 0 offen offset: 0 // (op 1038) Load value

s_mul_i32 s1, s25, 6 // (op 1041) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 1041) least significant
s_mul_hi_u32 s6, s24, 6 // (op 1041) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1041) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 1041) 


s_lshl_b64 s[0:1], s[24:25], 8 // (op 1041) 
s_mul_i32 s7, s1, 6 // (op 1041) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 1041) least significant
s_mul_hi_u32 s14, s0, 6 // (op 1041) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1041) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 1041) 

v_add_u32 v39, v14, s0 // (op 1041) 

buffer_load_dwordx3 v[60:62], v39, s[40:43], 0 offen // (op 1041) Load value
buffer_load_dwordx3 v[64:66], v39, s[40:43], 0 offen offset:12 // (op 1041) Load value
buffer_load_dwordx3 v[68:70], v39, s[40:43], 0 offen offset:24 // (op 1041) Load value
buffer_load_dwordx3 v[72:74], v39, s[40:43], 0 offen offset:36 // (op 1041) Load value
v_add_u32 v39, v39, v22 // (op 1041) 
buffer_load_dwordx3 v[76:78], v39, s[40:43], 0 offen // (op 1041) Load value
buffer_load_dwordx3 v[80:82], v39, s[40:43], 0 offen offset:12 // (op 1041) Load value
buffer_load_dwordx3 v[84:86], v39, s[40:43], 0 offen offset:24 // (op 1041) Load value
buffer_load_dwordx3 v[88:90], v39, s[40:43], 0 offen offset:36 // (op 1041) Load value





s_lshl_b32 s0, s34, 3 // (op 1044) 
v_add_u32 v14, v24, s0 // (op 1044) 


buffer_load_dword v39, v14, s[56:59], 0 offen // (op 1044) Load value


v_and_b32 v14, -16, v1 // (op 1731) 
v_sub_u32 v43, v1, v14 // (op 1731) 

v_lshlrev_b32 v14, 4, v43 // (op 1731) 

v_lshrrev_b32 v43, 4, v1 // (op 1731) 
v_lshl_add_u32 v47, v43, 8, v14 // (op 1731) 


s_nop 0
v_readfirstlane_b32 s0, v47 // (op 1731) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 3 // (op 1188) 
v_mov_b64 v[26:27], s[6:7] // (op 1188) convert


v_mov_b64 v[22:23], v[18:19] // (op 1188) 
s_add_u32 s1, 4096, s0 // (op 1188) 
s_mov_b32 m0, s1 // (op 1188) reg expression
s_mov_b32 s6, 256 // (op 1188) call()
buffer_load_dwordx4 v22, s[8:11], s6 offen offset: 0 lds // (op 1188) Load value direct to lds
v_add_co_u32 v22, s[14:15], v22, v26 // (op 1188) least significant half
v_addc_co_u32 v23, s[14:15], 0, v23, s[14:15] // (op 1188) most significant half

s_add_u32 m0, m0, 2048 // (op 1188) 
buffer_load_dwordx4 v22, s[8:11], s6 offen offset: 0 lds // (op 1188) Load value direct to lds









buffer_load_ubyte v14, v16, s[52:55], 0 offen offset:8 // (op 1050) Load value

v_lshlrev_b32 v16, 7, v1 // (op 1558) 
v_mul_lo_u32 v43, v16, 6 // (op 1558) 
v_lshrrev_b32 v47, 3, v43 // (op 1558) 

v_lshrrev_b32 v43, 4, v16 // (op 1558) 

v_lshlrev_b32 v16, 2, v43 // (op 1558) 

v_add_u32 v43, v47, v16 // (op 1558) 





s_waitcnt vmcnt(23)
ds_write_b96 v43, v[28:30] offset:8192 // (op 1052) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v43, v[32:34] offset:8208 // (op 1052) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v43, v[36:38] offset:8224 // (op 1052) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v43, v[40:42] offset:8240 // (op 1052) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v43, v[44:46] offset:8256 // (op 1052) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v43, v[48:50] offset:8272 // (op 1052) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v43, v[52:54] offset:8288 // (op 1052) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v43, v[56:58] offset:8304 // (op 1052) Store local data 

v_and_b32 v16, -16, v1 // (op 1669) 
v_sub_u32 v43, v1, v16 // (op 1669) 

v_lshlrev_b32 v16, 2, v43 // (op 1669) 

v_lshrrev_b32 v43, 4, v1 // (op 1669) 
v_lshl_add_u32 v47, v43, 6, v16 // (op 1669) 





s_waitcnt vmcnt(15)
ds_write_b32 v47, v35 offset:24576 // (op 1053) Store local data 

v_and_b32 v16, -8, v1 // (op 1616) 
v_sub_u32 v43, v1, v16 // (op 1616) 

v_and_b32 v16, 4294967288, v1 // (op 1616) 
v_add_u32 v47, v16, v43 // (op 1616) 





s_waitcnt vmcnt(12)
ds_write_b8 v47, v31 offset:25088 // (op 1055) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1895) 



ds_read_b96_tr_b6 v[92:94], v6 offset:8192 // (op 1073) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v6 offset:9216 // (op 1073) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1936) 


ds_read_u8 v16, v7 offset:24576 // (op 1086) Load local data 



ds_read_b96_tr_b6 v[98:100], v6 offset:8224 // (op 1078) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v6 offset:9248 // (op 1078) Transpose load local data 


ds_read_u8 v43, v7 offset:24608 // (op 1091) Load local data 
s_barrier  // (op 1876) 



ds_read_b128 v[104:107], v3 // (op 1069) Load local data 
ds_read_b128 v[108:111], v3 offset:64 // (op 1069) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1916) 



ds_read_u8 v47, v0 offset:25088 // (op 1082) Load local data 
v_and_b32 v51, -16, v1 // (op 1477) 
v_sub_u32 v55, v1, v51 // (op 1477) 

v_lshlrev_b32 v51, 4, v55 // (op 1477) 

v_lshrrev_b32 v55, 4, v1 // (op 1477) 
v_lshl_add_u32 v59, v55, 8, v51 // (op 1477) 


s_nop 0
v_readfirstlane_b32 s0, v59 // (op 1477) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v51, 7, v1 // (op 1529) 
v_mul_lo_u32 v55, v51, 6 // (op 1529) 
v_lshrrev_b32 v59, 3, v55 // (op 1529) 

v_lshrrev_b32 v55, 4, v51 // (op 1529) 

v_lshlrev_b32 v51, 2, v55 // (op 1529) 

v_add_u32 v55, v59, v51 // (op 1529) 


v_and_b32 v51, -8, v1 // (op 1587) 
v_sub_u32 v59, v1, v51 // (op 1587) 

v_and_b32 v51, 4294967288, v1 // (op 1587) 
v_add_u32 v63, v51, v59 // (op 1587) 


v_and_b32 v51, -16, v1 // (op 1640) 
v_sub_u32 v59, v1, v51 // (op 1640) 

v_lshlrev_b32 v51, 2, v59 // (op 1640) 

v_lshrrev_b32 v59, 4, v1 // (op 1640) 
v_lshl_add_u32 v67, v59, 6, v51 // (op 1640) 


v_and_b32 v51, -16, v1 // (op 1716) 
v_sub_u32 v59, v1, v51 // (op 1716) 

v_lshlrev_b32 v51, 4, v59 // (op 1716) 

v_lshrrev_b32 v59, 4, v1 // (op 1716) 
v_lshl_add_u32 v71, v59, 8, v51 // (op 1716) 


s_nop 0
v_readfirstlane_b32 s1, v71 // (op 1716) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v51, 7, v1 // (op 1755) 
v_mul_lo_u32 v59, v51, 6 // (op 1755) 
v_lshrrev_b32 v71, 3, v59 // (op 1755) 

v_lshrrev_b32 v59, 4, v51 // (op 1755) 

v_lshlrev_b32 v51, 2, v59 // (op 1755) 

v_add_u32 v59, v71, v51 // (op 1755) 


v_and_b32 v51, -8, v1 // (op 1784) 
v_sub_u32 v71, v1, v51 // (op 1784) 

v_and_b32 v51, 4294967288, v1 // (op 1784) 
v_add_u32 v75, v51, v71 // (op 1784) 


v_and_b32 v51, -16, v1 // (op 1808) 
v_sub_u32 v71, v1, v51 // (op 1808) 

v_lshlrev_b32 v51, 2, v71 // (op 1808) 

v_lshrrev_b32 v71, 4, v1 // (op 1808) 
v_lshl_add_u32 v79, v71, 6, v51 // (op 1808) 


s_mov_b32 s6, 0 // (op 400) call()
s_ashr_i64 s[14:15], s[28:29], 8 // (op 399) 
s_mov_b32 s7, s14 // (op 399) convert

s_and_b32 s14, s7, 4294967294 // (op 399) 

s_cmp_lt_u32 s6, s14 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399 // (op 399) Condition: Top (jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399 if false)
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399:

s_add_u32 s6, s6, 2 // (op 401) 
s_mul_i32 s15, s25, 6 // (op 10) most significant: low of high * low
s_mul_i32 s14, s24, 6 // (op 10) least significant
s_mul_hi_u32 s7, s24, 6 // (op 10) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 10) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[14:15] // (op 10) 


v_mov_b32 v51, v15 // (op 10) 
s_lshl_b64 s[14:15], s[24:25], 8 // (op 10) 
s_mul_i32 s19, s15, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 10) least significant
s_mul_hi_u32 s7, s14, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 10) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s7, s14, 1 // (op 10) 

v_add_u32 v71, v51, s7 // (op 10) 

buffer_load_dwordx3 v[28:30], v71, s[40:43], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[32:34], v71, s[40:43], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[36:38], v71, s[40:43], 0 offen offset:24 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:111], v[92:97], a[0:3], v47, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 88) 
buffer_load_dwordx3 v[40:42], v71, s[40:43], 0 offen offset:36 // (op 10) Load value
s_barrier  // (op 1885) 
v_add_u32 v71, v71, v18 // (op 10) 



buffer_load_dwordx3 v[44:46], v71, s[40:43], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[48:50], v71, s[40:43], 0 offen offset:12 // (op 10) Load value
ds_read_b128 v[112:115], v3 offset:128 // (op 91) Load local data 
ds_read_b128 v[116:119], v3 offset:192 // (op 91) Load local data 
buffer_load_dwordx3 v[52:54], v71, s[40:43], 0 offen offset:24 // (op 10) Load value
s_barrier  // (op 1905) 
buffer_load_dwordx3 v[56:58], v71, s[40:43], 0 offen offset:36 // (op 10) Load value







ds_read_b96_tr_b6 v[92:94], v6 offset:16384 // (op 95) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v6 offset:17408 // (op 95) Transpose load local data 
v_mov_b32 v71, v25 // (op 25) 
s_barrier  // (op 1926) 
s_lshl_b32 s7, s34, 4 // (op 25) 



ds_read_u8 v83, v0 offset:25092 // (op 99) Load local data 
v_add_u32 v87, v71, s7 // (op 25) 


s_barrier  // (op 1946) 
buffer_load_dword v35, v87, s[56:59], 0 offen // (op 25) Load value



s_waitcnt lgkmcnt(3)
s_barrier  // (op 1879) 
ds_read_u8 v16, v7 offset:24832 // (op 103) Load local data 

s_lshl_b64 s[14:15], s[12:13], 3 // (op 1167) 
v_mov_b64 v[18:19], s[14:15] // (op 1167) convert


v_mov_b64 v[26:27], v[20:21] // (op 1167) 
v_mov_b64 v[22:23], v[26:27] // (op 1167) 
s_mov_b32 m0, s0 // (op 1167) reg expression
s_mov_b32 s7, 512 // (op 1167) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:119], v[92:97], a[0:3], v83, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 105) 
buffer_load_dwordx4 v22, s[8:11], s7 offen offset: 0 lds // (op 1167) Load value direct to lds
v_add_co_u32 v22, s[14:15], v22, v18 // (op 1167) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:111], v[98:103], a[4:7], v47, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 447) 
v_addc_co_u32 v23, s[14:15], 0, v23, s[14:15] // (op 1167) most significant half




s_add_u32 m0, m0, 2048 // (op 1167) 
ds_read_b96_tr_b6 v[98:100], v6 offset:16416 // (op 451) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v6 offset:17440 // (op 451) Transpose load local data 
buffer_load_dwordx4 v22, s[8:11], s7 offen offset: 0 lds // (op 1167) Load value direct to lds







ds_read_u8 v43, v7 offset:24864 // (op 455) Load local data 
v_mov_b32 v87, v17 // (op 17) 

buffer_load_ubyte v31, v87, s[52:55], 0 offen offset:16 // (op 17) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:119], v[98:103], a[4:7], v83, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 456) 


s_barrier  // (op 1966) 



s_waitcnt vmcnt(23)
ds_write_b96 v59, v[60:62] offset:25216 // (op 302) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v59, v[64:66] offset:25232 // (op 302) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v59, v[68:70] offset:25248 // (op 302) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v59, v[72:74] offset:25264 // (op 302) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v59, v[76:78] offset:25280 // (op 302) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v59, v[80:82] offset:25296 // (op 302) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v59, v[84:86] offset:25312 // (op 302) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v59, v[88:90] offset:25328 // (op 302) Store local data 
s_barrier  // (op 1978) 



s_waitcnt vmcnt(15)
ds_write_b32 v79, v39 offset:41600 // (op 304) Store local data 
s_barrier  // (op 1972) 



s_waitcnt vmcnt(12)
ds_write_b8 v75, v14 offset:42112 // (op 303) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1962) 



ds_read_b96_tr_b6 v[92:94], v9 offset:25216 // (op 287) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v9 offset:26240 // (op 287) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1975) 


ds_read_u8 v16, v10 offset:41600 // (op 291) Load local data 



ds_read_b96_tr_b6 v[112:114], v9 offset:25248 // (op 470) Transpose load local data 
ds_read_b96_tr_b6 v[115:117], v9 offset:26272 // (op 470) Transpose load local data 


ds_read_u8 v83, v10 offset:41632 // (op 474) Load local data 
s_barrier  // (op 1956) 



ds_read_b128 v[104:107], v5 offset:4096 // (op 285) Load local data 
ds_read_b128 v[108:111], v5 offset:4160 // (op 285) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1969) 



ds_read_u8 v47, v4 offset:42112 // (op 289) Load local data 

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

v_add_u32 v91, v51, s7 // (op 305) 

buffer_load_dwordx3 v[60:62], v91, s[40:43], 0 offen // (op 305) Load value
buffer_load_dwordx3 v[64:66], v91, s[40:43], 0 offen offset:12 // (op 305) Load value
buffer_load_dwordx3 v[68:70], v91, s[40:43], 0 offen offset:24 // (op 305) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:111], v[92:97], a[0:3], v47, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 292) 
buffer_load_dwordx3 v[72:74], v91, s[40:43], 0 offen offset:36 // (op 305) Load value
v_add_u32 v91, v91, v18 // (op 305) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:111], v[112:117], a[4:7], v47, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 457) 
buffer_load_dwordx3 v[76:78], v91, s[40:43], 0 offen // (op 305) Load value


buffer_load_dwordx3 v[80:82], v91, s[40:43], 0 offen offset:12 // (op 305) Load value
ds_read_b128 v[112:115], v5 offset:4224 // (op 294) Load local data 
ds_read_b128 v[116:119], v5 offset:4288 // (op 294) Load local data 
buffer_load_dwordx3 v[84:86], v91, s[40:43], 0 offen offset:24 // (op 305) Load value
buffer_load_dwordx3 v[88:90], v91, s[40:43], 0 offen offset:36 // (op 305) Load value






ds_read_b96_tr_b6 v[120:122], v9 offset:33408 // (op 296) Transpose load local data 


ds_read_b96_tr_b6 v[123:125], v9 offset:34432 // (op 296) Transpose load local data 
s_lshl_b32 s7, s34, 3 // (op 307) 



ds_read_u8 v51, v4 offset:42116 // (op 298) Load local data 
s_mul_i32 s14, 3, s7 // (op 307) 

v_add_u32 v83, v71, s14 // (op 307) 




s_barrier  // (op 2017) 
buffer_load_dword v39, v83, s[56:59], 0 offen // (op 307) Load value

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
ds_read_u8 v71, v10 offset:41856 // (op 300) Load local data 




s_lshl_b64 s[14:15], s[12:13], 3 // (op 1178) 
v_mov_b64 v[18:19], s[14:15] // (op 1178) convert


v_mov_b64 v[22:23], v[26:27] // (op 1178) 
s_add_u32 s7, 4096, s1 // (op 1178) 
s_mov_b32 m0, s7 // (op 1178) reg expression
s_mov_b32 s14, 768 // (op 1178) call()
ds_read_b96_tr_b6 v[126:128], v9 offset:33440 // (op 461) Transpose load local data 
ds_read_b96_tr_b6 v[129:131], v9 offset:34464 // (op 461) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:119], v[120:125], a[0:3], v51, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 301) 




s_barrier  // (op 2055) 
ds_read_u8 v24, v10 offset:41888 // (op 465) Load local data 
buffer_load_dwordx4 v22, s[8:11], s14 offen offset: 0 lds // (op 1178) Load value direct to lds
v_add_co_u32 v22, s[18:19], v22, v18 // (op 1178) least significant half
s_barrier  // (op 2059) 
v_addc_co_u32 v23, s[18:19], 0, v23, s[18:19] // (op 1178) most significant half

s_add_u32 m0, m0, 2048 // (op 1178) 
buffer_load_dwordx4 v22, s[8:11], s14 offen offset: 0 lds // (op 1178) Load value direct to lds




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:119], v[126:131], a[4:7], v51, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 466) 








v_add_co_u32 v20, vcc, 512, v20 // (op 1239) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 1239) most significant half
buffer_load_ubyte v14, v87, s[52:55], 0 offen offset:24 // (op 306) Load value

s_barrier  // (op 1898) 
v_add_u32 v17, 16, v17 // (op 1367) 



s_waitcnt vmcnt(23)
ds_write_b96 v55, v[28:30] offset:8192 // (op 13) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v55, v[32:34] offset:8208 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v55, v[36:38] offset:8224 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v55, v[40:42] offset:8240 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v55, v[44:46] offset:8256 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v55, v[48:50] offset:8272 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v55, v[52:54] offset:8288 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v55, v[56:58] offset:8304 // (op 13) Store local data 
s_barrier  // (op 1939) 



s_waitcnt vmcnt(15)
ds_write_b32 v67, v35 offset:24576 // (op 30) Store local data 
s_barrier  // (op 1919) 



s_waitcnt vmcnt(12)
ds_write_b8 v63, v31 offset:25088 // (op 21) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1908) 



ds_read_b96_tr_b6 v[92:94], v6 offset:8192 // (op 75) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v6 offset:9216 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1949) 


ds_read_u8 v16, v7 offset:24576 // (op 85) Load local data 



ds_read_b96_tr_b6 v[98:100], v6 offset:8224 // (op 442) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v6 offset:9248 // (op 442) Transpose load local data 


ds_read_u8 v43, v7 offset:24608 // (op 446) Load local data 
s_barrier  // (op 1888) 



ds_read_b128 v[104:107], v3 // (op 70) Load local data 
ds_read_b128 v[108:111], v3 offset:64 // (op 70) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1929) 



ds_read_u8 v47, v0 offset:25088 // (op 80) Load local data 
s_ashr_i64 s[14:15], s[28:29], 8 // (op 399) 
s_mov_b32 s7, s14 // (op 399) convert

s_and_b32 s14, s7, 4294967294 // (op 399) 

s_cmp_lt_u32 s6, s14 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399 // (op 399) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399 if true)
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399:
















s_barrier  // (op 2037) 
v_lshrrev_b32 v15, 4, v1 // (op 1201) 
s_ashr_i64 s[0:1], s[28:29], 8 // (op 1201) 
v_lshl_add_u32 v63, s16, 4, v15 // (op 1201) 

s_lshr_b32 s6, s0, 1 // (op 1201) 

v_and_b32 v15, -16, v1 // (op 1201) 
v_sub_u32 v67, v1, v15 // (op 1201) 

v_lshlrev_b32 v15, 4, v67 // (op 1201) 

v_lshl_add_u32 v67, s6, 9, v15 // (op 1201) 


v_mov_b32 v41, 0 // (op 1201) convert
v_mov_b32 v40, v67 // (op 1201) convert

v_mul_lo_u32 v45, v63, s13 // (op 1201) most significant: low of low * high
v_mul_hi_u32 v15, v63, s12 // (op 1201) most significant: high of low * low
v_mul_lo_u32 v44, v63, s12 // (op 1201) least significant: low of low * low
v_add_u32 v45, v45, v15 // (op 1201) most significant: sum

v_add_co_u32 v44, s[0:1], v44, v40 // (op 1201) least significant half
v_addc_co_u32 v45, s[0:1], v45, v41, s[0:1] // (op 1201) most significant half



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
s_ashr_i64 s[0:1], s[28:29], 8 // (op 1269) 
s_lshr_b32 s6, s0, 1 // (op 1269) 

v_lshl_add_u32 v46, s6, 9, v15 // (op 1269) 


s_lshl_b32 s0, s17, 6 // (op 1269) 
s_mov_b32 s7, 0 // (op 1269) convert
s_mov_b32 s6, s0 // (op 1269) convert

v_mul_lo_u32 v41, v46, s25 // (op 1269) most significant: low of low * high
v_mul_hi_u32 v15, v46, s24 // (op 1269) most significant: high of low * low
v_mul_lo_u32 v40, v46, s24 // (op 1269) least significant: low of low * low
v_add_u32 v41, v41, v15 // (op 1269) most significant: sum

v_mov_b32 v15, s6 // (op 1269) 
v_mov_b32 v42, s7 // (op 1269) 
v_add_co_u32 v40, s[0:1], v40, v15 // (op 1269) least significant half
v_addc_co_u32 v41, s[0:1], v41, v42, s[0:1] // (op 1269) most significant half





v_mul_lo_u32 v49, v41, 6 // (op 1269) most significant: low of high * low
v_mul_hi_u32 v15, v40, 6 // (op 1269) most significant: high of low * low
v_mul_lo_u32 v48, v40, 6 // (op 1269) least significant: low of low * low
v_add_u32 v49, v49, v15 // (op 1269) most significant: sum


v_lshrrev_b64 v[40:41], 3, v[48:49] // (op 1269) 

v_mov_b32 v15, v40 // (op 1269) convert

s_mul_i32 s1, s29, 6 // (op 1271) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1271) least significant
s_mul_hi_u32 s6, s28, 6 // (op 1271) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1271) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1271) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1271) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1271) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1271) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1271) most significant: sum
s_add_u32 s7, s7, s15 // (op 1271) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1271) 

s_mov_b32 s40, s26 // (op 1271) reg expression
s_mov_b32 s41, s27 // (op 1271) reg expression
s_mov_b32 s42, s0 // (op 1271) reg expression
s_mov_b32 s43, 131072 // (op 1271) call()

v_lshrrev_b32 v46, 3, v1 // (op 1333) 
s_ashr_i64 s[0:1], s[28:29], 8 // (op 1333) 
v_lshl_add_u32 v63, s16, 4, v46 // (op 1333) 

s_lshr_b32 s6, s0, 1 // (op 1333) 

v_and_b32 v46, -8, v1 // (op 1333) 
v_sub_u32 v67, v1, v46 // (op 1333) 

v_lshl_add_u32 v46, s6, 4, v67 // (op 1333) 


v_mul_lo_u32 v67, v63, s30 // (op 1333) 
v_add_u32 v67, v67, v46 // (op 1333) 


s_mul_i32 s1, s30, s23 // (op 1335) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1335) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1335) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1335) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1335) most significant: sum
s_add_u32 s1, s1, s7 // (op 1335) most significant: sum


s_mov_b32 s52, s32 // (op 1335) reg expression
s_mov_b32 s53, s33 // (op 1335) reg expression
s_mov_b32 s54, s0 // (op 1335) reg expression
s_mov_b32 s55, 131072 // (op 1335) call()

v_lshrrev_b32 v46, 4, v1 // (op 1397) 
s_ashr_i64 s[0:1], s[28:29], 8 // (op 1397) 
v_and_b32 v63, -16, v1 // (op 1397) 
s_lshr_b32 s6, s0, 1 // (op 1397) 

v_sub_u32 v71, v1, v63 // (op 1397) 

v_lshl_add_u32 v63, s6, 4, v46 // (op 1397) 


v_lshlrev_b32 v46, 2, v71 // (op 1397) 

v_lshl_add_u32 v71, s17, 6, v46 // (op 1397) 

v_mul_lo_u32 v46, v63, s34 // (op 1397) 
v_add_u32 v46, v46, v71 // (op 1397) 


s_mul_i32 s1, s34, s39 // (op 1399) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1399) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1399) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1399) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1399) most significant: sum
s_add_u32 s1, s1, s7 // (op 1399) most significant: sum


s_mov_b32 s56, s36 // (op 1399) reg expression
s_mov_b32 s57, s37 // (op 1399) reg expression
s_mov_b32 s58, s0 // (op 1399) reg expression
s_mov_b32 s59, 131072 // (op 1399) call()

v_and_b32 v63, -16, v1 // (op 1490) 
v_sub_u32 v71, v1, v63 // (op 1490) 

v_lshlrev_b32 v63, 4, v71 // (op 1490) 

v_lshrrev_b32 v71, 4, v1 // (op 1490) 
v_lshl_add_u32 v75, v71, 8, v63 // (op 1490) 


s_nop 0
v_readfirstlane_b32 s0, v75 // (op 1490) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v63, 7, v1 // (op 1542) 
v_mul_lo_u32 v71, v63, 6 // (op 1542) 
v_lshrrev_b32 v75, 3, v71 // (op 1542) 

v_lshrrev_b32 v71, 4, v63 // (op 1542) 

v_lshlrev_b32 v63, 2, v71 // (op 1542) 

v_add_u32 v71, v75, v63 // (op 1542) 


v_and_b32 v63, -8, v1 // (op 1600) 
v_sub_u32 v75, v1, v63 // (op 1600) 

v_and_b32 v63, 4294967288, v1 // (op 1600) 
v_add_u32 v79, v63, v75 // (op 1600) 


v_and_b32 v63, -16, v1 // (op 1653) 
v_sub_u32 v75, v1, v63 // (op 1653) 

v_lshlrev_b32 v63, 2, v75 // (op 1653) 

v_lshrrev_b32 v75, 4, v1 // (op 1653) 
v_lshl_add_u32 v83, v75, 6, v63 // (op 1653) 


s_ashr_i64 s[6:7], s[28:29], 8 // (op 405) 
s_and_b32 s1, s6, 4294967294 // (op 405) 

s_ashr_i64 s[6:7], s[28:29], 8 // (op 404) 
s_mov_b32 s14, s6 // (op 404) convert

s_cmp_lt_u32 s1, s14 // (op 404) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404 // (op 404) Condition: Top (jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404:
s_barrier  // (op 1882) 
s_add_u32 s1, s1, 1 // (op 406) 

s_lshl_b64 s[6:7], s[12:13], 3 // (op 1172) 
v_mov_b64 v[16:17], s[6:7] // (op 1172) convert


v_mov_b64 v[18:19], v[44:45] // (op 1172) 
v_mov_b64 v[20:21], v[18:19] // (op 1172) 
s_mov_b32 m0, s0 // (op 1172) reg expression
buffer_load_dwordx4 v20, s[8:11], 0 offen offset: 0 lds // (op 1172) Load value direct to lds
v_add_co_u32 v20, s[6:7], v20, v16 // (op 1172) least significant half
v_addc_co_u32 v21, s[6:7], 0, v21, s[6:7] // (op 1172) most significant half

s_add_u32 m0, m0, 2048 // (op 1172) 
buffer_load_dwordx4 v20, s[8:11], 0 offen offset: 0 lds // (op 1172) Load value direct to lds



v_add_co_u32 v44, vcc, 256, v44 // (op 1191) least significant half
v_addc_co_u32 v45, vcc, 0, v45, vcc // (op 1191) most significant half

s_mul_i32 s7, s25, 6 // (op 221) most significant: low of high * low
s_mul_i32 s6, s24, 6 // (op 221) least significant
s_mul_hi_u32 s14, s24, 6 // (op 221) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 221) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[6:7] // (op 221) 


v_mov_b32 v14, v15 // (op 221) 
v_mov_b32 v18, v14 // (op 221) reg expression
buffer_load_dwordx3 v[20:22], v18, s[40:43], 0 offen // (op 221) Load value
buffer_load_dwordx3 v[24:26], v18, s[40:43], 0 offen offset:12 // (op 221) Load value
buffer_load_dwordx3 v[28:30], v18, s[40:43], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx3 v[32:34], v18, s[40:43], 0 offen offset:36 // (op 221) Load value
v_add_u32 v18, v18, v16 // (op 221) 
buffer_load_dwordx3 v[36:38], v18, s[40:43], 0 offen // (op 221) Load value
buffer_load_dwordx3 v[40:42], v18, s[40:43], 0 offen offset:12 // (op 221) Load value
buffer_load_dwordx3 v[48:50], v18, s[40:43], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx3 v[52:54], v18, s[40:43], 0 offen offset:36 // (op 221) Load value



s_lshl_b64 s[6:7], s[24:25], 8 // (op 1261) 
s_mul_i32 s15, s7, 6 // (op 1261) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 1261) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1261) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 1261) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 1261) 

v_add_u32 v15, v15, s6 // (op 1261) 



s_barrier  // (op 1901) 
v_mov_b32 v14, v67 // (op 222) 




buffer_load_ubyte v23, v14, s[52:55], 0 offen offset: 0 // (op 222) Load value

v_add_u32 v67, 8, v67 // (op 1325) 
s_waitcnt vmcnt(8)
ds_write_b96 v71, v[20:22] offset:8192 // (op 218) Store local data 


v_mov_b32 v14, v46 // (op 223) 
v_mov_b32 v27, v14 // (op 223) reg expression

buffer_load_dword v31, v27, s[56:59], 0 offen // (op 223) Load value


s_lshl_b32 s6, s34, 3 // (op 1389) 
v_add_u32 v46, v46, s6 // (op 1389) 

s_waitcnt vmcnt(8)
ds_write_b96 v71, v[24:26] offset:8208 // (op 218) Store local data 
s_barrier  // (op 1891) 
s_waitcnt vmcnt(7)
ds_write_b96 v71, v[28:30] offset:8224 // (op 218) Store local data 



ds_read_b128 v[56:59], v3 // (op 201) Load local data 
ds_read_b128 v[60:63], v3 offset:64 // (op 201) Load local data 
s_waitcnt vmcnt(6)
ds_write_b96 v71, v[32:34] offset:8240 // (op 218) Store local data 
s_waitcnt vmcnt(5)
ds_write_b96 v71, v[36:38] offset:8256 // (op 218) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v71, v[40:42] offset:8272 // (op 218) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v71, v[48:50] offset:8288 // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v71, v[52:54] offset:8304 // (op 218) Store local data 

s_barrier  // (op 1922) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1912) 






ds_read_b96_tr_b6 v[16:18], v6 offset:8192 // (op 203) Transpose load local data 
ds_read_b96_tr_b6 v[19:21], v6 offset:9216 // (op 203) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v79, v23 offset:25088 // (op 219) Store local data 

s_barrier  // (op 1942) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1932) 






ds_read_u8 v14, v0 offset:25088 // (op 205) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v83, v31 offset:24576 // (op 220) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1952) 


ds_read_u8 v22, v7 offset:24576 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[16:21], a[0:3], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 208) 





ds_read_b128 v[48:51], v3 offset:128 // (op 210) Load local data 
ds_read_b128 v[52:55], v3 offset:192 // (op 210) Load local data 



ds_read_b96_tr_b6 v[16:18], v6 offset:16384 // (op 212) Transpose load local data 
ds_read_b96_tr_b6 v[19:21], v6 offset:17408 // (op 212) Transpose load local data 



ds_read_u8 v47, v0 offset:25092 // (op 214) Load local data 


ds_read_u8 v22, v7 offset:24832 // (op 216) Load local data 



ds_read_b96_tr_b6 v[38:40], v6 offset:8224 // (op 412) Transpose load local data 
ds_read_b96_tr_b6 v[41:43], v6 offset:9248 // (op 412) Transpose load local data 


ds_read_u8 v23, v7 offset:24608 // (op 416) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[38:43], a[4:7], v14, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 417) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[16:21], a[0:3], v47, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 217) 
ds_read_b96_tr_b6 v[38:40], v6 offset:16416 // (op 421) Transpose load local data 


ds_read_b96_tr_b6 v[41:43], v6 offset:17440 // (op 421) Transpose load local data 


ds_read_u8 v14, v7 offset:24864 // (op 425) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[38:43], a[4:7], v47, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 426) 




s_ashr_i64 s[6:7], s[28:29], 8 // (op 404) 
s_mov_b32 s14, s6 // (op 404) convert

s_cmp_lt_u32 s1, s14 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404 // (op 404) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404 if true)
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404:
























s_barrier  // (op 2041) 








v_cmp_eq_f32 vcc, 0.00000, v12 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155
v_accvgpr_read v4, a0 // (op 50) 
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v14, v13, v4 // (op 50) 
v_mul_f32 v15, v13, v5 // (op 50) 
v_mul_f32 v16, v13, v6 // (op 50) 
v_mul_f32 v17, v13, v7 // (op 50) 



v_mov_b32 v0, v11 // (op 56) reg expression

buffer_store_dwordx4 v[14:17], v0, s[48:51], 0 offen // (op 56) Store value


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


buffer_store_dwordx4 v[14:17], v0, s[48:51], 0 offen // (op 712) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155:


v_mov_b32 v0, v8 // (op 167) reg expression

buffer_load_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 167) Load value

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

buffer_store_dwordx4 v[4:7], v0, s[48:51], 0 offen // (op 166) Store value




s_lshl_b32 s0, s2, 7 // (op 734) 
v_add_u32 v0, v8, s0 // (op 734) 


buffer_load_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 734) Load value

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


buffer_store_dwordx4 v[4:7], v0, s[48:51], 0 offen // (op 733) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155:










s_endpgm  // End of RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.LRR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_end:
.size RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2, .LRR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_end-RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
  .amdhsa_next_free_vgpr 140
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 42240 // lds bytes
  .amdhsa_accum_offset 132
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
    DataTypeB: B6
    DestDataType: s
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
  - .name: RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
    .symbol: RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 42240
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 132
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

