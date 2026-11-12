// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.p2align 8
.type RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2,@function
RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 1567) 
v_and_b32 v3, 15, v1 // (op 1567) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1567) 


v_and_b32 v0, 48, v1 // (op 1567) 
v_lshl_add_u32 v3, v4, 8, v0 // (op 1567) 


v_lshrrev_b32 v0, 7, v1 // (op 1677) 
v_and_b32 v4, 15, v1 // (op 1677) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1677) 


v_and_b32 v0, 63, v1 // (op 1677) 
v_lshrrev_b32 v4, 4, v0 // (op 1677) 

v_lshl_add_u32 v0, v5, 3, v4 // (op 1677) 


v_lshrrev_b32 v4, 7, v1 // (op 1792) 
v_and_b32 v5, 15, v1 // (op 1792) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 1792) 


v_and_b32 v4, 48, v1 // (op 1792) 
v_lshl_add_u32 v5, v6, 8, v4 // (op 1792) 


v_lshrrev_b32 v4, 7, v1 // (op 1860) 
v_and_b32 v6, 15, v1 // (op 1860) 
v_lshl_add_u32 v7, v4, 4, v6 // (op 1860) 


v_and_b32 v4, 63, v1 // (op 1860) 
v_lshrrev_b32 v6, 4, v4 // (op 1860) 

v_lshl_add_u32 v4, v7, 3, v6 // (op 1860) 


v_lshrrev_b32 v6, 7, v1 // (op 1511) 
v_and_b32 v7, 63, v1 // (op 1511) 
v_lshrrev_b32 v8, 4, v7 // (op 1511) 
v_lshlrev_b32 v9, 2, v8 // (op 1511) 

v_lshl_add_u32 v8, v6, 4, v9 // (op 1511) 


v_lshl_add_u32 v6, s16, 4, v8 // (op 1511) 

v_lshrrev_b32 v8, 2, v7 // (op 1511) 

v_and_b32 v7, 3, v8 // (op 1511) 

v_and_b32 v8, 3, v1 // (op 1511) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 1511) 


v_lshrrev_b32 v7, 6, v1 // (op 1511) 
v_and_b32 v8, 1, v7 // (op 1511) 

v_lshl_add_u32 v7, v8, 4, v9 // (op 1511) 


v_lshl_add_u32 v8, s17, 6, v7 // (op 1511) 

v_mul_lo_u32 v7, v8, s2 // (op 1511) 

v_add_lshl_u32 v8, v6, v7, 2 // (op 1511) 


v_and_b32 v6, 48, v1 // (op 1624) 
v_and_b32 v7, 63, v1 // (op 1624) 
v_sub_u32 v9, v7, v6 // (op 1624) 

v_lshrrev_b32 v7, 1, v9 // (op 1624) 
s_mul_i32 s1, s2, s9 // (op 1513) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1513) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1513) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1513) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1513) most significant: sum
s_add_u32 s1, s1, s19 // (op 1513) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1513) 

v_add_lshl_u32 v10, v6, v7, 6 // (op 1624) 


v_and_b32 v6, 1, v9 // (op 1624) 

v_lshlrev_b32 v7, 3, v6 // (op 1624) 

v_lshrrev_b32 v6, 6, v1 // (op 1624) 
v_and_b32 v9, 1, v6 // (op 1624) 

v_lshl_add_u32 v6, v9, 4, v7 // (op 1624) 


v_add_u32 v7, v10, v6 // (op 1624) 


v_lshrrev_b32 v6, 6, v1 // (op 1735) 
v_and_b32 v9, 1, v6 // (op 1735) 

v_and_b32 v6, 15, v1 // (op 1735) 
v_lshl_add_u32 v10, v9, 4, v6 // (op 1735) 


v_and_b32 v6, 63, v1 // (op 1735) 
v_lshrrev_b32 v9, 4, v6 // (op 1735) 

v_lshl_add_u32 v6, v9, 6, v10 // (op 1735) 


v_lshrrev_b32 v9, 7, v1 // (op 1751) 
v_and_b32 v10, 63, v1 // (op 1751) 
v_lshrrev_b32 v11, 4, v10 // (op 1751) 
v_lshlrev_b32 v12, 2, v11 // (op 1751) 

v_lshl_add_u32 v11, v9, 4, v12 // (op 1751) 


v_lshl_add_u32 v9, s16, 4, v11 // (op 1751) 

v_lshrrev_b32 v11, 2, v10 // (op 1751) 

v_and_b32 v10, 3, v11 // (op 1751) 

v_and_b32 v11, 3, v1 // (op 1751) 
v_lshl_add_u32 v12, v10, 2, v11 // (op 1751) 


v_lshrrev_b32 v10, 6, v1 // (op 1751) 
v_and_b32 v11, 1, v10 // (op 1751) 

v_lshl_add_u32 v10, v11, 4, v12 // (op 1751) 


v_lshl_add_u32 v11, s17, 6, v10 // (op 1751) 

v_mul_lo_u32 v10, v11, s4 // (op 1751) 

v_add_lshl_u32 v11, v9, v10, 2 // (op 1751) 


v_and_b32 v9, 48, v1 // (op 1836) 
v_and_b32 v10, 63, v1 // (op 1836) 
v_sub_u32 v12, v10, v9 // (op 1836) 

v_lshrrev_b32 v10, 1, v12 // (op 1836) 
s_mul_i32 s1, s4, s9 // (op 1753) most significant: low of low * high
v_add_lshl_u32 v13, v9, v10, 6 // (op 1836) 


v_and_b32 v9, 1, v12 // (op 1836) 

v_lshlrev_b32 v10, 3, v9 // (op 1836) 

v_lshrrev_b32 v9, 6, v1 // (op 1836) 
v_and_b32 v12, 1, v9 // (op 1836) 

v_lshl_add_u32 v9, v12, 4, v10 // (op 1836) 


v_add_u32 v10, v13, v9 // (op 1836) 


v_lshrrev_b32 v9, 6, v1 // (op 1889) 
v_and_b32 v12, 1, v9 // (op 1889) 

v_and_b32 v9, 15, v1 // (op 1889) 
v_lshl_add_u32 v13, v12, 4, v9 // (op 1889) 


v_and_b32 v9, 63, v1 // (op 1889) 
v_lshrrev_b32 v12, 4, v9 // (op 1889) 

v_lshl_add_u32 v9, v12, 6, v13 // (op 1889) 


s_mul_i32 s40, s5, s8 // (op 1753) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1753) least significant
s_mul_hi_u32 s41, s4, s8 // (op 1753) most significant: high of low * low
s_add_u32 s1, s1, s40 // (op 1753) most significant: sum
s_add_u32 s1, s1, s41 // (op 1753) most significant: sum


s_lshl_b64 s[40:41], s[0:1], 2 // (op 1753) 

s_mov_b32 s44, s6 // (op 1513) reg expression
s_mov_b32 s45, s7 // (op 1513) reg expression
s_mov_b32 s46, s18 // (op 1513) reg expression
s_mov_b32 s47, 131072 // (op 1513) call()

s_mov_b32 s48, s10 // (op 1753) reg expression

s_mov_b32 s49, s11 // (op 1753) reg expression
s_mov_b32 s50, s40 // (op 1753) reg expression
s_mov_b32 s51, 131072 // (op 1753) call()



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
v_lshrrev_b32 v14, 4, v1 // (op 1278) 
v_lshl_add_u32 v15, s16, 4, v14 // (op 1278) 

v_and_b32 v14, -16, v1 // (op 1278) 
v_sub_u32 v16, v1, v14 // (op 1278) 

v_lshlrev_b32 v14, 4, v16 // (op 1278) 

v_mov_b32 v17, 0 // (op 1278) convert
v_mov_b32 v16, v14 // (op 1278) convert

v_mul_lo_u32 v19, v15, s13 // (op 1278) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 1278) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 1278) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1278) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 1278) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1278) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 1280) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1280) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1280) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1280) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1280) most significant: sum
s_add_u32 s1, s1, s7 // (op 1280) most significant: sum


s_mov_b32 s8, s20 // (op 1280) reg expression
s_mov_b32 s9, s21 // (op 1280) reg expression
s_mov_b32 s10, s0 // (op 1280) reg expression
s_mov_b32 s11, 131072 // (op 1280) call()

v_lshrrev_b32 v14, 4, v1 // (op 1297) 
v_lshl_add_u32 v15, s16, 4, v14 // (op 1297) 

v_and_b32 v14, -16, v1 // (op 1297) 
v_sub_u32 v16, v1, v14 // (op 1297) 

v_lshlrev_b32 v14, 4, v16 // (op 1297) 

v_mov_b32 v17, 0 // (op 1297) convert
v_mov_b32 v16, v14 // (op 1297) convert

v_mul_lo_u32 v21, v15, s13 // (op 1297) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 1297) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 1297) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1297) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 1297) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 1297) most significant half



s_mul_i32 s1, s12, s23 // (op 1299) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1299) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1299) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1299) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1299) most significant: sum
s_add_u32 s1, s1, s7 // (op 1299) most significant: sum


s_mov_b32 s8, s20 // (op 1299) reg expression
s_mov_b32 s9, s21 // (op 1299) reg expression
s_mov_b32 s10, s0 // (op 1299) reg expression
s_mov_b32 s11, 131072 // (op 1299) call()

v_and_b32 v14, -4, v1 // (op 1342) 
v_sub_u32 v15, v1, v14 // (op 1342) 

v_lshlrev_b32 v14, 4, v15 // (op 1342) 

v_lshl_add_u32 v15, s17, 6, v14 // (op 1342) 

v_mov_b32 v17, 0 // (op 1342) convert
v_mov_b32 v16, v15 // (op 1342) convert

v_lshrrev_b32 v14, 2, v1 // (op 1342) 
v_mul_lo_u32 v23, v14, s25 // (op 1342) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 1342) most significant: high of low * low
v_mul_lo_u32 v22, v14, s24 // (op 1342) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 1342) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v16 // (op 1342) least significant half
v_addc_co_u32 v23, s[0:1], v23, v17, s[0:1] // (op 1342) most significant half



s_mul_i32 s1, s24, s29 // (op 1344) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 1344) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 1344) least significant
s_mul_hi_u32 s7, s24, s28 // (op 1344) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1344) most significant: sum
s_add_u32 s1, s1, s7 // (op 1344) most significant: sum


s_mov_b32 s40, s26 // (op 1344) reg expression
s_mov_b32 s41, s27 // (op 1344) reg expression
s_mov_b32 s42, s0 // (op 1344) reg expression
s_mov_b32 s43, 131072 // (op 1344) call()

v_and_b32 v14, -4, v1 // (op 1361) 
v_sub_u32 v15, v1, v14 // (op 1361) 

v_lshlrev_b32 v14, 4, v15 // (op 1361) 

v_lshl_add_u32 v15, s17, 6, v14 // (op 1361) 

v_mov_b32 v17, 0 // (op 1361) convert
v_mov_b32 v16, v15 // (op 1361) convert

v_lshrrev_b32 v14, 2, v1 // (op 1361) 
v_mul_lo_u32 v25, v14, s25 // (op 1361) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 1361) most significant: high of low * low
v_mul_lo_u32 v24, v14, s24 // (op 1361) least significant: low of low * low
v_add_u32 v25, v25, v15 // (op 1361) most significant: sum

v_add_co_u32 v24, s[0:1], v24, v16 // (op 1361) least significant half
v_addc_co_u32 v25, s[0:1], v25, v17, s[0:1] // (op 1361) most significant half



s_mul_i32 s1, s24, s29 // (op 1363) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 1363) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 1363) least significant
s_mul_hi_u32 s7, s24, s28 // (op 1363) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1363) most significant: sum
s_add_u32 s1, s1, s7 // (op 1363) most significant: sum


s_mov_b32 s40, s26 // (op 1363) reg expression
s_mov_b32 s41, s27 // (op 1363) reg expression
s_mov_b32 s42, s0 // (op 1363) reg expression
s_mov_b32 s43, 131072 // (op 1363) call()

v_lshrrev_b32 v14, 3, v1 // (op 1406) 
v_lshl_add_u32 v15, s16, 4, v14 // (op 1406) 

v_and_b32 v14, -8, v1 // (op 1406) 
v_sub_u32 v16, v1, v14 // (op 1406) 

v_mul_lo_u32 v14, v15, s30 // (op 1406) 
v_add_u32 v14, v14, v16 // (op 1406) 


s_mul_i32 s1, s30, s23 // (op 1408) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1408) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1408) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1408) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1408) most significant: sum
s_add_u32 s1, s1, s7 // (op 1408) most significant: sum


s_mov_b32 s52, s32 // (op 1408) reg expression
s_mov_b32 s53, s33 // (op 1408) reg expression
s_mov_b32 s54, s0 // (op 1408) reg expression
s_mov_b32 s55, 131072 // (op 1408) call()

v_lshrrev_b32 v15, 3, v1 // (op 1425) 
v_lshl_add_u32 v16, s16, 4, v15 // (op 1425) 

v_and_b32 v15, -8, v1 // (op 1425) 
v_sub_u32 v17, v1, v15 // (op 1425) 

v_mul_lo_u32 v15, v16, s30 // (op 1425) 
v_add_u32 v15, v15, v17 // (op 1425) 


s_mul_i32 s1, s30, s23 // (op 1427) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1427) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1427) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1427) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1427) most significant: sum
s_add_u32 s1, s1, s7 // (op 1427) most significant: sum


s_mov_b32 s52, s32 // (op 1427) reg expression
s_mov_b32 s53, s33 // (op 1427) reg expression
s_mov_b32 s54, s0 // (op 1427) reg expression
s_mov_b32 s55, 131072 // (op 1427) call()

v_and_b32 v16, -16, v1 // (op 1470) 
v_sub_u32 v17, v1, v16 // (op 1470) 

v_lshlrev_b32 v16, 2, v17 // (op 1470) 

v_lshl_add_u32 v17, s17, 6, v16 // (op 1470) 

v_lshrrev_b32 v16, 4, v1 // (op 1470) 
v_mul_lo_u32 v26, v16, s34 // (op 1470) 
v_add_u32 v26, v26, v17 // (op 1470) 


s_mul_i32 s1, s34, s39 // (op 1472) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1472) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1472) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1472) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1472) most significant: sum
s_add_u32 s1, s1, s7 // (op 1472) most significant: sum


s_mov_b32 s56, s36 // (op 1472) reg expression
s_mov_b32 s57, s37 // (op 1472) reg expression
s_mov_b32 s58, s0 // (op 1472) reg expression
s_mov_b32 s59, 131072 // (op 1472) call()

v_and_b32 v16, -16, v1 // (op 1489) 
v_sub_u32 v17, v1, v16 // (op 1489) 

v_lshlrev_b32 v16, 2, v17 // (op 1489) 

v_lshl_add_u32 v17, s17, 6, v16 // (op 1489) 

v_lshrrev_b32 v16, 4, v1 // (op 1489) 
v_mul_lo_u32 v27, v16, s34 // (op 1489) 
v_add_u32 v27, v27, v17 // (op 1489) 


s_mul_i32 s1, s34, s39 // (op 1491) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1491) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1491) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1491) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1491) most significant: sum
s_add_u32 s1, s1, s7 // (op 1491) most significant: sum


s_mov_b32 s56, s36 // (op 1491) reg expression
s_mov_b32 s57, s37 // (op 1491) reg expression
s_mov_b32 s58, s0 // (op 1491) reg expression
s_mov_b32 s59, 131072 // (op 1491) call()

v_and_b32 v16, -4, v1 // (op 1607) 
v_sub_u32 v17, v1, v16 // (op 1607) 

v_lshlrev_b32 v16, 4, v17 // (op 1607) 

v_lshrrev_b32 v17, 2, v1 // (op 1607) 
v_lshl_add_u32 v28, v17, 6, v16 // (op 1607) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1607) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 1226) 
v_mov_b64 v[16:17], s[6:7] // (op 1226) convert


v_mov_b64 v[28:29], v[22:23] // (op 1226) reg expression
s_mov_b32 m0, s0 // (op 1226) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v16 // (op 1226) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1226) most significant half

s_add_u32 m0, m0, 2048 // (op 1226) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v16 // (op 1226) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1226) most significant half

s_add_u32 m0, m0, 2048 // (op 1226) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v16 // (op 1226) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1226) most significant half

s_add_u32 m0, m0, 2048 // (op 1226) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v16 // (op 1226) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1226) most significant half

s_add_u32 m0, m0, 2048 // (op 1226) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v16 // (op 1226) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1226) most significant half

s_add_u32 m0, m0, 2048 // (op 1226) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v16 // (op 1226) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1226) most significant half

s_add_u32 m0, m0, 2048 // (op 1226) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v16 // (op 1226) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1226) most significant half

s_add_u32 m0, m0, 2048 // (op 1226) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1226) Load value direct to lds





v_mov_b32 v16, v26 // (op 1032) reg expression

buffer_load_dword v17, v16, s[56:59], 0 offen // (op 1032) Load value

v_and_b32 v16, -16, v1 // (op 1555) 
v_sub_u32 v28, v1, v16 // (op 1555) 

v_lshlrev_b32 v16, 4, v28 // (op 1555) 

v_lshrrev_b32 v28, 4, v1 // (op 1555) 
v_lshl_add_u32 v29, v28, 8, v16 // (op 1555) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 1555) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 3 // (op 1231) 
v_mov_b64 v[28:29], s[6:7] // (op 1231) convert


v_mov_b64 v[30:31], v[18:19] // (op 1231) 
s_add_u32 s1, 16384, s0 // (op 1231) 
s_mov_b32 m0, s1 // (op 1231) reg expression
buffer_load_dwordx4 v30, s[8:11], 0 offen offset: 0 lds // (op 1231) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 1231) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 1231) most significant half

s_add_u32 m0, m0, 2048 // (op 1231) 
buffer_load_dwordx4 v30, s[8:11], 0 offen offset: 0 lds // (op 1231) Load value direct to lds







buffer_load_ubyte v16, v14, s[52:55], 0 offen offset: 0 // (op 1038) Load value
v_and_b32 v28, -4, v1 // (op 1819) 
v_sub_u32 v29, v1, v28 // (op 1819) 

v_lshlrev_b32 v28, 4, v29 // (op 1819) 

v_lshrrev_b32 v29, 2, v1 // (op 1819) 
v_lshl_add_u32 v30, v29, 6, v28 // (op 1819) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 1819) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 1236) 
v_mov_b64 v[28:29], s[6:7] // (op 1236) convert


s_lshl_b64 s[6:7], s[24:25], 8 // (op 1236) 
v_mov_b32 v30, s6 // (op 1236) 
v_mov_b32 v31, s7 // (op 1236) 
v_add_co_u32 v32, s[14:15], v22, v30 // (op 1236) least significant half
v_addc_co_u32 v33, s[14:15], v23, v31, s[14:15] // (op 1236) most significant half




s_add_u32 s1, 20480, s0 // (op 1236) 
s_mov_b32 m0, s1 // (op 1236) reg expression
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 1236) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1236) most significant half

s_add_u32 m0, m0, 2048 // (op 1236) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 1236) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1236) most significant half

s_add_u32 m0, m0, 2048 // (op 1236) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 1236) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1236) most significant half

s_add_u32 m0, m0, 2048 // (op 1236) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 1236) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1236) most significant half

s_add_u32 m0, m0, 2048 // (op 1236) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 1236) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1236) most significant half

s_add_u32 m0, m0, 2048 // (op 1236) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 1236) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1236) most significant half

s_add_u32 m0, m0, 2048 // (op 1236) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 1236) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1236) most significant half

s_add_u32 m0, m0, 2048 // (op 1236) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1236) Load value direct to lds







s_lshl_b32 s0, s34, 3 // (op 1044) 
v_add_u32 v22, v26, s0 // (op 1044) 


buffer_load_dword v23, v22, s[56:59], 0 offen // (op 1044) Load value


v_and_b32 v22, -16, v1 // (op 1780) 
v_sub_u32 v26, v1, v22 // (op 1780) 

v_lshlrev_b32 v22, 4, v26 // (op 1780) 

v_lshrrev_b32 v26, 4, v1 // (op 1780) 
v_lshl_add_u32 v28, v26, 8, v22 // (op 1780) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1780) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 3 // (op 1238) 
v_mov_b64 v[28:29], s[6:7] // (op 1238) convert


v_mov_b64 v[30:31], v[18:19] // (op 1238) 
s_add_u32 s1, 36864, s0 // (op 1238) 
s_mov_b32 m0, s1 // (op 1238) reg expression
s_mov_b32 s6, 256 // (op 1238) call()
buffer_load_dwordx4 v30, s[8:11], s6 offen offset: 0 lds // (op 1238) Load value direct to lds
v_add_co_u32 v30, s[14:15], v30, v28 // (op 1238) least significant half
v_addc_co_u32 v31, s[14:15], 0, v31, s[14:15] // (op 1238) most significant half

s_add_u32 m0, m0, 2048 // (op 1238) 
buffer_load_dwordx4 v30, s[8:11], s6 offen offset: 0 lds // (op 1238) Load value direct to lds









buffer_load_ubyte v22, v14, s[52:55], 0 offen offset:8 // (op 1050) Load value

v_and_b32 v14, -16, v1 // (op 1718) 
v_sub_u32 v26, v1, v14 // (op 1718) 

v_lshlrev_b32 v14, 2, v26 // (op 1718) 

v_lshrrev_b32 v26, 4, v1 // (op 1718) 
v_lshl_add_u32 v18, v26, 6, v14 // (op 1718) 





s_waitcnt vmcnt(15)
ds_write_b32 v18, v17 offset:40960 // (op 1053) Store local data 

v_and_b32 v14, -8, v1 // (op 1665) 
v_sub_u32 v26, v1, v14 // (op 1665) 

v_and_b32 v14, 4294967288, v1 // (op 1665) 
v_add_u32 v18, v14, v26 // (op 1665) 





s_waitcnt vmcnt(12)
ds_write_b8 v18, v16 offset:41472 // (op 1055) Store local data 

s_barrier  // (op 1978) 



ds_read_b64_tr_b8 v[28:29], v7 // (op 1076) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v7 offset:512 // (op 1076) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v7 offset:4096 // (op 1076) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v7 offset:4608 // (op 1076) Transpose load local data 



ds_read_b64_tr_b8 v[36:37], v7 offset:8192 // (op 1086) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v7 offset:8704 // (op 1086) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v7 offset:12288 // (op 1086) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:12800 // (op 1086) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2012) 


ds_read_u8 v14, v6 offset:40960 // (op 1102) Load local data 


ds_read_u8 v26, v6 offset:41216 // (op 1112) Load local data 



ds_read_b64_tr_b8 v[44:45], v7 offset:32 // (op 1081) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:544 // (op 1081) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:4128 // (op 1081) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v7 offset:4640 // (op 1081) Transpose load local data 



ds_read_b64_tr_b8 v[52:53], v7 offset:8224 // (op 1091) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v7 offset:8736 // (op 1091) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:12320 // (op 1091) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v7 offset:12832 // (op 1091) Transpose load local data 


ds_read_u8 v18, v6 offset:40992 // (op 1107) Load local data 


ds_read_u8 v19, v6 offset:41248 // (op 1117) Load local data 
s_barrier  // (op 1962) 



ds_read_b128 v[60:63], v3 offset:16384 // (op 1069) Load local data 
ds_read_b128 v[64:67], v3 offset:16448 // (op 1069) Load local data 



ds_read_b128 v[68:71], v3 offset:16512 // (op 1072) Load local data 
ds_read_b128 v[72:75], v3 offset:16576 // (op 1072) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 1995) 



ds_read_u8 v76, v0 offset:41472 // (op 1095) Load local data 



ds_read_u8 v77, v0 offset:41476 // (op 1098) Load local data 
v_and_b32 v78, -16, v1 // (op 1527) 
v_sub_u32 v79, v1, v78 // (op 1527) 

v_lshlrev_b32 v78, 4, v79 // (op 1527) 

v_lshrrev_b32 v79, 4, v1 // (op 1527) 
v_lshl_add_u32 v80, v79, 8, v78 // (op 1527) 


s_nop 0
v_readfirstlane_b32 s0, v80 // (op 1527) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v78, -4, v1 // (op 1579) 
v_sub_u32 v79, v1, v78 // (op 1579) 

v_lshlrev_b32 v78, 4, v79 // (op 1579) 

v_lshrrev_b32 v79, 2, v1 // (op 1579) 
v_lshl_add_u32 v80, v79, 6, v78 // (op 1579) 


s_nop 0
v_readfirstlane_b32 s1, v80 // (op 1579) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v78, -8, v1 // (op 1636) 
v_sub_u32 v79, v1, v78 // (op 1636) 

v_and_b32 v78, 4294967288, v1 // (op 1636) 
v_add_u32 v80, v78, v79 // (op 1636) 


v_and_b32 v78, -16, v1 // (op 1689) 
v_sub_u32 v79, v1, v78 // (op 1689) 

v_lshlrev_b32 v78, 2, v79 // (op 1689) 

v_lshrrev_b32 v79, 4, v1 // (op 1689) 
v_lshl_add_u32 v81, v79, 6, v78 // (op 1689) 


v_and_b32 v78, -16, v1 // (op 1765) 
v_sub_u32 v79, v1, v78 // (op 1765) 

v_lshlrev_b32 v78, 4, v79 // (op 1765) 

v_lshrrev_b32 v79, 4, v1 // (op 1765) 
v_lshl_add_u32 v82, v79, 8, v78 // (op 1765) 


s_nop 0
v_readfirstlane_b32 s6, v82 // (op 1765) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v78, -4, v1 // (op 1804) 
v_sub_u32 v79, v1, v78 // (op 1804) 

v_lshlrev_b32 v78, 4, v79 // (op 1804) 

v_lshrrev_b32 v79, 2, v1 // (op 1804) 
v_lshl_add_u32 v82, v79, 6, v78 // (op 1804) 


s_nop 0
v_readfirstlane_b32 s7, v82 // (op 1804) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v78, -8, v1 // (op 1848) 
v_sub_u32 v79, v1, v78 // (op 1848) 

v_and_b32 v78, 4294967288, v1 // (op 1848) 
v_add_u32 v82, v78, v79 // (op 1848) 


v_and_b32 v78, -16, v1 // (op 1872) 
v_sub_u32 v79, v1, v78 // (op 1872) 

v_lshlrev_b32 v78, 2, v79 // (op 1872) 

v_lshrrev_b32 v79, 4, v1 // (op 1872) 
v_lshl_add_u32 v83, v79, 6, v78 // (op 1872) 


s_mov_b32 s14, 0 // (op 400) call()
s_ashr_i64 s[18:19], s[28:29], 8 // (op 399) 
s_mov_b32 s15, s18 // (op 399) convert

s_and_b32 s18, s15, 4294967294 // (op 399) 

s_cmp_lt_u32 s14, s18 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399 // (op 399) Condition: Top (jump to RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399 if false)
RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399:
s_add_u32 s14, s14, 2 // (op 401) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:67], v[28:35], a[0:3], v76, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 88) 
s_barrier  // (op 1981) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[60:67], v[44:51], a[4:7], v76, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 447) 
s_lshl_b64 s[18:19], s[24:25], 5 // (op 1198) 
v_mov_b64 v[78:79], s[18:19] // (op 1198) convert


v_mov_b64 v[84:85], v[24:25] // (op 1198) 
s_lshl_b64 s[18:19], s[24:25], 8 // (op 1198) 
s_lshl_b64 s[60:61], s[18:19], 1 // (op 1198) 

v_mov_b32 v86, s60 // (op 1198) 
v_mov_b32 v87, s61 // (op 1198) 
v_add_co_u32 v88, s[18:19], v84, v86 // (op 1198) least significant half
v_addc_co_u32 v89, s[18:19], v85, v87, s[18:19] // (op 1198) most significant half




s_mov_b32 m0, s1 // (op 1198) reg expression
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1198) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v78 // (op 1198) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 1198) most significant half

s_add_u32 m0, m0, 2048 // (op 1198) 
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1198) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v78 // (op 1198) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 1198) most significant half

s_add_u32 m0, m0, 2048 // (op 1198) 
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1198) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v78 // (op 1198) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 1198) most significant half

s_add_u32 m0, m0, 2048 // (op 1198) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:75], v[36:43], a[0:3], v77, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 105) 
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1198) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:75], v[52:59], a[4:7], v77, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 456) 
v_add_co_u32 v88, s[18:19], v88, v78 // (op 1198) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 1198) most significant half

s_add_u32 m0, m0, 2048 // (op 1198) 
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1198) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v78 // (op 1198) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 1198) most significant half

s_add_u32 m0, m0, 2048 // (op 1198) 
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1198) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v78 // (op 1198) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 1198) most significant half

s_add_u32 m0, m0, 2048 // (op 1198) 
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1198) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v78 // (op 1198) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 1198) most significant half

s_add_u32 m0, m0, 2048 // (op 1198) 
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1198) Load value direct to lds




v_mov_b32 v78, v27 // (op 25) 
s_lshl_b32 s15, s34, 4 // (op 25) 
v_add_u32 v79, v78, s15 // (op 25) 


buffer_load_dword v17, v79, s[56:59], 0 offen // (op 25) Load value

s_barrier  // (op 1965) 

s_lshl_b64 s[18:19], s[12:13], 3 // (op 1193) 
v_mov_b64 v[86:87], s[18:19] // (op 1193) convert


v_mov_b64 v[88:89], v[20:21] // (op 1193) 
v_mov_b64 v[90:91], v[88:89] // (op 1193) 
s_add_u32 s15, 16384, s0 // (op 1193) 
s_mov_b32 m0, s15 // (op 1193) reg expression
s_mov_b32 s18, 512 // (op 1193) call()
buffer_load_dwordx4 v90, s[8:11], s18 offen offset: 0 lds // (op 1193) Load value direct to lds
v_add_co_u32 v90, s[60:61], v90, v86 // (op 1193) least significant half
v_addc_co_u32 v91, s[60:61], 0, v91, s[60:61] // (op 1193) most significant half

s_add_u32 m0, m0, 2048 // (op 1193) 
buffer_load_dwordx4 v90, s[8:11], s18 offen offset: 0 lds // (op 1193) Load value direct to lds






v_mov_b32 v79, v15 // (op 17) 

buffer_load_ubyte v16, v79, s[52:55], 0 offen offset:16 // (op 17) Load value
s_barrier  // (op 2051) 



s_waitcnt vmcnt(15)
ds_write_b32 v83, v23 offset:41600 // (op 304) Store local data 
s_barrier  // (op 2045) 



s_waitcnt vmcnt(12)
ds_write_b8 v82, v22 offset:42112 // (op 303) Store local data 
s_barrier  // (op 2035) 



ds_read_b64_tr_b8 v[28:29], v10 offset:20480 // (op 287) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v10 offset:20992 // (op 287) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v10 offset:24576 // (op 287) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v10 offset:25088 // (op 287) Transpose load local data 



ds_read_b64_tr_b8 v[90:91], v10 offset:28672 // (op 296) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v10 offset:29184 // (op 296) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v10 offset:32768 // (op 296) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v10 offset:33280 // (op 296) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2048) 


ds_read_u8 v14, v9 offset:41600 // (op 291) Load local data 


ds_read_u8 v18, v9 offset:41856 // (op 300) Load local data 



ds_read_b64_tr_b8 v[44:45], v10 offset:20512 // (op 470) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v10 offset:21024 // (op 470) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v10 offset:24608 // (op 470) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v10 offset:25120 // (op 470) Transpose load local data 



ds_read_b64_tr_b8 v[98:99], v10 offset:28704 // (op 461) Transpose load local data 
ds_read_b64_tr_b8 v[100:101], v10 offset:29216 // (op 461) Transpose load local data 
ds_read_b64_tr_b8 v[102:103], v10 offset:32800 // (op 461) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v10 offset:33312 // (op 461) Transpose load local data 


ds_read_u8 v26, v9 offset:41632 // (op 474) Load local data 


ds_read_u8 v86, v9 offset:41888 // (op 465) Load local data 
s_barrier  // (op 2029) 
s_barrier  // (op 2118) 



ds_read_b128 v[60:63], v5 offset:36864 // (op 285) Load local data 
ds_read_b128 v[64:67], v5 offset:36928 // (op 285) Load local data 



ds_read_b128 v[68:71], v5 offset:36992 // (op 294) Load local data 
ds_read_b128 v[72:75], v5 offset:37056 // (op 294) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 2042) 



ds_read_u8 v76, v4 offset:42112 // (op 289) Load local data 



ds_read_u8 v77, v4 offset:42116 // (op 298) Load local data 
s_barrier  // (op 2082) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:67], v[28:35], a[0:3], v76, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 292) 
s_barrier  // (op 2039) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[60:67], v[44:51], a[4:7], v76, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 457) 
s_lshl_b64 s[18:19], s[24:25], 5 // (op 1221) 
v_mov_b64 v[106:107], s[18:19] // (op 1221) convert


s_lshl_b64 s[18:19], s[24:25], 8 // (op 1221) 
s_mul_i32 s61, 3, s19 // (op 1221) most significant: low of low * high
s_mul_i32 s60, 3, s18 // (op 1221) least significant
s_mul_hi_u32 s15, 3, s18 // (op 1221) most significant: high of low * low
s_add_u32 s61, s61, s15 // (op 1221) most significant: sum


v_mov_b32 v87, s60 // (op 1221) 
v_mov_b32 v108, s61 // (op 1221) 
v_add_co_u32 v110, s[18:19], v84, v87 // (op 1221) least significant half
v_addc_co_u32 v111, s[18:19], v85, v108, s[18:19] // (op 1221) most significant half




s_add_u32 s15, 20480, s7 // (op 1221) 
s_mov_b32 m0, s15 // (op 1221) reg expression
buffer_load_dwordx4 v110, s[40:43], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v110, s[18:19], v110, v106 // (op 1221) least significant half
v_addc_co_u32 v111, s[18:19], 0, v111, s[18:19] // (op 1221) most significant half

s_add_u32 m0, m0, 2048 // (op 1221) 
buffer_load_dwordx4 v110, s[40:43], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v110, s[18:19], v110, v106 // (op 1221) least significant half
v_addc_co_u32 v111, s[18:19], 0, v111, s[18:19] // (op 1221) most significant half

s_add_u32 m0, m0, 2048 // (op 1221) 
buffer_load_dwordx4 v110, s[40:43], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v110, s[18:19], v110, v106 // (op 1221) least significant half
v_addc_co_u32 v111, s[18:19], 0, v111, s[18:19] // (op 1221) most significant half

s_add_u32 m0, m0, 2048 // (op 1221) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:75], v[90:97], a[0:3], v77, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 301) 
buffer_load_dwordx4 v110, s[40:43], 0 offen offset: 0 lds // (op 1221) Load value direct to lds

v_add_co_u32 v110, s[18:19], v110, v106 // (op 1221) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:75], v[98:105], a[4:7], v77, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 466) 
v_addc_co_u32 v111, s[18:19], 0, v111, s[18:19] // (op 1221) most significant half



s_add_u32 m0, m0, 2048 // (op 1221) 
buffer_load_dwordx4 v110, s[40:43], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v110, s[18:19], v110, v106 // (op 1221) least significant half
v_addc_co_u32 v111, s[18:19], 0, v111, s[18:19] // (op 1221) most significant half

s_add_u32 m0, m0, 2048 // (op 1221) 
buffer_load_dwordx4 v110, s[40:43], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v110, s[18:19], v110, v106 // (op 1221) least significant half
v_addc_co_u32 v111, s[18:19], 0, v111, s[18:19] // (op 1221) most significant half

s_add_u32 m0, m0, 2048 // (op 1221) 
buffer_load_dwordx4 v110, s[40:43], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v110, s[18:19], v110, v106 // (op 1221) least significant half
v_addc_co_u32 v111, s[18:19], 0, v111, s[18:19] // (op 1221) most significant half

s_add_u32 m0, m0, 2048 // (op 1221) 
buffer_load_dwordx4 v110, s[40:43], 0 offen offset: 0 lds // (op 1221) Load value direct to lds






s_lshl_b32 s15, s34, 3 // (op 307) 
s_mul_i32 s18, 3, s15 // (op 307) 

v_add_u32 v84, v78, s18 // (op 307) 


buffer_load_dword v23, v84, s[56:59], 0 offen // (op 307) Load value

s_lshl_b64 s[18:19], s[24:25], 8 // (op 1353) 

s_lshl_b64 s[60:61], s[18:19], 1 // (op 1353) 

s_barrier  // (op 2032) 
s_lshl_b32 s15, s34, 4 // (op 1481) 

v_add_u32 v27, v27, s15 // (op 1481) 

v_mov_b32 v78, s60 // (op 1353) 
v_mov_b32 v84, s61 // (op 1353) 
s_lshl_b64 s[18:19], s[12:13], 3 // (op 1216) 
v_mov_b64 v[86:87], s[18:19] // (op 1216) convert


v_mov_b64 v[90:91], v[88:89] // (op 1216) 
s_add_u32 s15, 36864, s6 // (op 1216) 
s_mov_b32 m0, s15 // (op 1216) reg expression
s_mov_b32 s18, 768 // (op 1216) call()
v_add_co_u32 v24, s[62:63], v24, v78 // (op 1353) least significant half
v_addc_co_u32 v25, s[62:63], v25, v84, s[62:63] // (op 1353) most significant half




buffer_load_dwordx4 v90, s[8:11], s18 offen offset: 0 lds // (op 1216) Load value direct to lds
v_add_co_u32 v90, s[60:61], v90, v86 // (op 1216) least significant half
v_addc_co_u32 v91, s[60:61], 0, v91, s[60:61] // (op 1216) most significant half

s_add_u32 m0, m0, 2048 // (op 1216) 
buffer_load_dwordx4 v90, s[8:11], s18 offen offset: 0 lds // (op 1216) Load value direct to lds








v_add_co_u32 v20, vcc, 512, v20 // (op 1289) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 1289) most significant half
buffer_load_ubyte v22, v79, s[52:55], 0 offen offset:24 // (op 306) Load value

s_barrier  // (op 2015) 
v_add_u32 v15, 16, v15 // (op 1417) 



s_waitcnt vmcnt(15)
ds_write_b32 v81, v17 offset:40960 // (op 30) Store local data 
s_barrier  // (op 1998) 



s_waitcnt vmcnt(12)
ds_write_b8 v80, v16 offset:41472 // (op 21) Store local data 
s_barrier  // (op 1987) 



ds_read_b64_tr_b8 v[28:29], v7 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v7 offset:512 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v7 offset:4096 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v7 offset:4608 // (op 75) Transpose load local data 



ds_read_b64_tr_b8 v[36:37], v7 offset:8192 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v7 offset:8704 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v7 offset:12288 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:12800 // (op 95) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2022) 


ds_read_u8 v14, v6 offset:40960 // (op 85) Load local data 


ds_read_u8 v26, v6 offset:41216 // (op 103) Load local data 



ds_read_b64_tr_b8 v[44:45], v7 offset:32 // (op 442) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:544 // (op 442) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:4128 // (op 442) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v7 offset:4640 // (op 442) Transpose load local data 



ds_read_b64_tr_b8 v[52:53], v7 offset:8224 // (op 451) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v7 offset:8736 // (op 451) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:12320 // (op 451) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v7 offset:12832 // (op 451) Transpose load local data 


ds_read_u8 v18, v6 offset:40992 // (op 446) Load local data 


ds_read_u8 v19, v6 offset:41248 // (op 455) Load local data 
s_barrier  // (op 1971) 



ds_read_b128 v[60:63], v3 offset:16384 // (op 70) Load local data 
ds_read_b128 v[64:67], v3 offset:16448 // (op 70) Load local data 



ds_read_b128 v[68:71], v3 offset:16512 // (op 91) Load local data 
ds_read_b128 v[72:75], v3 offset:16576 // (op 91) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 2005) 



ds_read_u8 v76, v0 offset:41472 // (op 80) Load local data 



ds_read_u8 v77, v0 offset:41476 // (op 99) Load local data 
s_ashr_i64 s[18:19], s[28:29], 8 // (op 399) 
s_mov_b32 s15, s18 // (op 399) convert

s_and_b32 s18, s15, 4294967294 // (op 399) 

s_cmp_lt_u32 s14, s18 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399 // (op 399) Condition: Bottom (jump to RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399 if true)
RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399:



















s_barrier  // (op 2100) 
v_lshrrev_b32 v14, 4, v1 // (op 1251) 
s_ashr_i64 s[0:1], s[28:29], 8 // (op 1251) 
v_lshl_add_u32 v15, s16, 4, v14 // (op 1251) 

s_lshr_b32 s6, s0, 1 // (op 1251) 

v_and_b32 v14, -16, v1 // (op 1251) 
v_sub_u32 v16, v1, v14 // (op 1251) 

v_lshlrev_b32 v14, 4, v16 // (op 1251) 

v_lshl_add_u32 v16, s6, 9, v14 // (op 1251) 


v_mov_b32 v21, 0 // (op 1251) convert
v_mov_b32 v20, v16 // (op 1251) convert

v_mul_lo_u32 v17, v15, s13 // (op 1251) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 1251) most significant: high of low * low
v_mul_lo_u32 v16, v15, s12 // (op 1251) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 1251) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v20 // (op 1251) least significant half
v_addc_co_u32 v17, s[0:1], v17, v21, s[0:1] // (op 1251) most significant half



s_mul_i32 s1, s12, s23 // (op 1253) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1253) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1253) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1253) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1253) most significant: sum
s_add_u32 s1, s1, s7 // (op 1253) most significant: sum


s_mov_b32 s8, s20 // (op 1253) reg expression
s_mov_b32 s9, s21 // (op 1253) reg expression
s_mov_b32 s10, s0 // (op 1253) reg expression
s_mov_b32 s11, 131072 // (op 1253) call()

v_lshrrev_b32 v14, 2, v1 // (op 1319) 
s_ashr_i64 s[0:1], s[28:29], 8 // (op 1319) 
v_and_b32 v15, -4, v1 // (op 1319) 
s_lshr_b32 s6, s0, 1 // (op 1319) 

v_sub_u32 v20, v1, v15 // (op 1319) 

v_lshl_add_u32 v15, s6, 9, v14 // (op 1319) 


v_lshlrev_b32 v14, 4, v20 // (op 1319) 

v_lshl_add_u32 v20, s17, 6, v14 // (op 1319) 

v_mov_b32 v25, 0 // (op 1319) convert
v_mov_b32 v24, v20 // (op 1319) convert

v_mul_lo_u32 v21, v15, s25 // (op 1319) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1319) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 1319) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1319) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v24 // (op 1319) least significant half
v_addc_co_u32 v21, s[0:1], v21, v25, s[0:1] // (op 1319) most significant half



s_mul_i32 s1, s24, s29 // (op 1321) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 1321) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 1321) least significant
s_mul_hi_u32 s7, s24, s28 // (op 1321) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1321) most significant: sum
s_add_u32 s1, s1, s7 // (op 1321) most significant: sum


s_mov_b32 s40, s26 // (op 1321) reg expression
s_mov_b32 s41, s27 // (op 1321) reg expression
s_mov_b32 s42, s0 // (op 1321) reg expression
s_mov_b32 s43, 131072 // (op 1321) call()

v_lshrrev_b32 v14, 3, v1 // (op 1383) 
s_ashr_i64 s[0:1], s[28:29], 8 // (op 1383) 
v_lshl_add_u32 v15, s16, 4, v14 // (op 1383) 

s_lshr_b32 s6, s0, 1 // (op 1383) 

v_and_b32 v14, -8, v1 // (op 1383) 
v_sub_u32 v24, v1, v14 // (op 1383) 

v_lshl_add_u32 v14, s6, 4, v24 // (op 1383) 


v_mul_lo_u32 v24, v15, s30 // (op 1383) 
v_add_u32 v24, v24, v14 // (op 1383) 


s_mul_i32 s1, s30, s23 // (op 1385) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1385) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1385) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1385) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1385) most significant: sum
s_add_u32 s1, s1, s7 // (op 1385) most significant: sum


s_mov_b32 s52, s32 // (op 1385) reg expression
s_mov_b32 s53, s33 // (op 1385) reg expression
s_mov_b32 s54, s0 // (op 1385) reg expression
s_mov_b32 s55, 131072 // (op 1385) call()

v_lshrrev_b32 v14, 4, v1 // (op 1447) 
s_ashr_i64 s[0:1], s[28:29], 8 // (op 1447) 
v_and_b32 v15, -16, v1 // (op 1447) 
s_lshr_b32 s6, s0, 1 // (op 1447) 

v_sub_u32 v25, v1, v15 // (op 1447) 

v_lshl_add_u32 v15, s6, 4, v14 // (op 1447) 


v_lshlrev_b32 v14, 2, v25 // (op 1447) 

v_lshl_add_u32 v25, s17, 6, v14 // (op 1447) 

v_mul_lo_u32 v14, v15, s34 // (op 1447) 
v_add_u32 v14, v14, v25 // (op 1447) 


s_mul_i32 s1, s34, s39 // (op 1449) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1449) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1449) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1449) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1449) most significant: sum
s_add_u32 s1, s1, s7 // (op 1449) most significant: sum


s_mov_b32 s56, s36 // (op 1449) reg expression
s_mov_b32 s57, s37 // (op 1449) reg expression
s_mov_b32 s58, s0 // (op 1449) reg expression
s_mov_b32 s59, 131072 // (op 1449) call()

v_and_b32 v15, -16, v1 // (op 1540) 
v_sub_u32 v25, v1, v15 // (op 1540) 

v_lshlrev_b32 v15, 4, v25 // (op 1540) 

v_lshrrev_b32 v25, 4, v1 // (op 1540) 
v_lshl_add_u32 v26, v25, 8, v15 // (op 1540) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1540) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -4, v1 // (op 1592) 
v_sub_u32 v25, v1, v15 // (op 1592) 

v_lshlrev_b32 v15, 4, v25 // (op 1592) 

v_lshrrev_b32 v25, 2, v1 // (op 1592) 
v_lshl_add_u32 v26, v25, 6, v15 // (op 1592) 


s_nop 0
v_readfirstlane_b32 s1, v26 // (op 1592) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -8, v1 // (op 1649) 
v_sub_u32 v25, v1, v15 // (op 1649) 

v_and_b32 v15, 4294967288, v1 // (op 1649) 
v_add_u32 v26, v15, v25 // (op 1649) 


v_and_b32 v15, -16, v1 // (op 1702) 
v_sub_u32 v25, v1, v15 // (op 1702) 

v_lshlrev_b32 v15, 2, v25 // (op 1702) 

v_lshrrev_b32 v25, 4, v1 // (op 1702) 
v_lshl_add_u32 v27, v25, 6, v15 // (op 1702) 


s_ashr_i64 s[6:7], s[28:29], 8 // (op 405) 
s_and_b32 s14, s6, 4294967294 // (op 405) 

s_ashr_i64 s[6:7], s[28:29], 8 // (op 404) 
s_mov_b32 s15, s6 // (op 404) convert

s_cmp_lt_u32 s14, s15 // (op 404) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404 // (op 404) Condition: Top (jump to RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404:
s_barrier  // (op 1968) 
s_add_u32 s14, s14, 1 // (op 406) 

s_lshl_b64 s[6:7], s[12:13], 3 // (op 1203) 
v_mov_b64 v[18:19], s[6:7] // (op 1203) convert


v_mov_b64 v[22:23], v[16:17] // (op 1203) 
v_mov_b64 v[28:29], v[22:23] // (op 1203) 
s_add_u32 s15, 16384, s0 // (op 1203) 
s_mov_b32 m0, s15 // (op 1203) reg expression
buffer_load_dwordx4 v28, s[8:11], 0 offen offset: 0 lds // (op 1203) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1203) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1203) most significant half

s_add_u32 m0, m0, 2048 // (op 1203) 
buffer_load_dwordx4 v28, s[8:11], 0 offen offset: 0 lds // (op 1203) Load value direct to lds




v_add_co_u32 v16, vcc, 256, v16 // (op 1241) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1241) most significant half
s_barrier  // (op 1984) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 1209) 
v_mov_b64 v[18:19], s[6:7] // (op 1209) convert


v_mov_b64 v[22:23], v[20:21] // (op 1209) 
v_mov_b64 v[28:29], v[22:23] // (op 1209) reg expression
s_mov_b32 m0, s1 // (op 1209) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1209) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1209) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1209) most significant half

s_add_u32 m0, m0, 2048 // (op 1209) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1209) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1209) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1209) most significant half

s_add_u32 m0, m0, 2048 // (op 1209) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1209) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1209) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1209) most significant half

s_add_u32 m0, m0, 2048 // (op 1209) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1209) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1209) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1209) most significant half

s_add_u32 m0, m0, 2048 // (op 1209) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1209) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1209) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1209) most significant half

s_add_u32 m0, m0, 2048 // (op 1209) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1209) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1209) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1209) most significant half

s_add_u32 m0, m0, 2048 // (op 1209) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1209) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1209) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1209) most significant half

s_add_u32 m0, m0, 2048 // (op 1209) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1209) Load value direct to lds



s_lshl_b64 s[6:7], s[24:25], 8 // (op 1311) 
v_mov_b32 v15, s6 // (op 1311) 
v_mov_b32 v25, s7 // (op 1311) 
v_add_co_u32 v20, s[18:19], v20, v15 // (op 1311) least significant half
v_addc_co_u32 v21, s[18:19], v21, v25, s[18:19] // (op 1311) most significant half






v_mov_b32 v15, v24 // (op 222) 

buffer_load_ubyte v25, v15, s[52:55], 0 offen offset: 0 // (op 222) Load value

v_add_u32 v24, 8, v24 // (op 1375) 


s_barrier  // (op 2001) 
v_mov_b32 v15, v14 // (op 223) 



v_mov_b32 v18, v15 // (op 223) reg expression

buffer_load_dword v19, v18, s[56:59], 0 offen // (op 223) Load value


s_lshl_b32 s15, s34, 3 // (op 1439) 
v_add_u32 v14, v14, s15 // (op 1439) 

s_waitcnt vmcnt(1)
ds_write_b8 v26, v25 offset:41472 // (op 219) Store local data 

s_barrier  // (op 1974) 



s_barrier  // (op 2018) 
ds_read_b128 v[28:31], v3 offset:16384 // (op 201) Load local data 



ds_read_b128 v[32:35], v3 offset:16448 // (op 201) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v27, v19 offset:40960 // (op 220) Store local data 

s_barrier  // (op 1991) 



ds_read_b64_tr_b8 v[36:37], v7 // (op 203) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v7 offset:512 // (op 203) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v7 offset:4096 // (op 203) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:4608 // (op 203) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 2008) 



ds_read_u8 v15, v0 offset:41472 // (op 205) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2025) 


ds_read_u8 v25, v6 offset:40960 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:43], a[0:3], v15, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 208) 





ds_read_b128 v[36:39], v3 offset:16512 // (op 210) Load local data 
ds_read_b128 v[40:43], v3 offset:16576 // (op 210) Load local data 



ds_read_b64_tr_b8 v[44:45], v7 offset:8192 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:8704 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:12288 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v7 offset:12800 // (op 212) Transpose load local data 



ds_read_u8 v25, v0 offset:41476 // (op 214) Load local data 


ds_read_u8 v18, v6 offset:41216 // (op 216) Load local data 



ds_read_b64_tr_b8 v[52:53], v7 offset:32 // (op 412) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v7 offset:544 // (op 412) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:4128 // (op 412) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v7 offset:4640 // (op 412) Transpose load local data 


ds_read_u8 v19, v6 offset:40992 // (op 416) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:35], v[52:59], a[4:7], v15, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 417) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:51], a[0:3], v25, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 217) 
ds_read_b64_tr_b8 v[28:29], v7 offset:8224 // (op 421) Transpose load local data 


ds_read_b64_tr_b8 v[30:31], v7 offset:8736 // (op 421) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v7 offset:12320 // (op 421) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v7 offset:12832 // (op 421) Transpose load local data 


ds_read_u8 v15, v6 offset:41248 // (op 425) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:43], v[28:35], a[4:7], v25, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 426) 




s_ashr_i64 s[6:7], s[28:29], 8 // (op 404) 
s_mov_b32 s15, s6 // (op 404) convert

s_cmp_lt_u32 s14, s15 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404 // (op 404) Condition: Bottom (jump to RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404 if true)
RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404:
























s_barrier  // (op 2104) 








v_cmp_eq_f32 vcc, 0.00000, v12 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155
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
s_branch RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155:


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
RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155:










s_endpgm  // End of RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.LRR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_end:
.size RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2, .LRR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_end-RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
  .amdhsa_next_free_vgpr 120
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 42240 // lds bytes
  .amdhsa_accum_offset 112
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
  - .name: RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
    .symbol: RR_GEMM_TT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 42240
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 112
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

