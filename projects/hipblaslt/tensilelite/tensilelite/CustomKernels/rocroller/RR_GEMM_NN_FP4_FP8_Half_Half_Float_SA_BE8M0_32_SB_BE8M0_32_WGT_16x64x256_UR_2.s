// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.p2align 8
.type RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2,@function
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 1503) 
v_and_b32 v3, 48, v1 // (op 1503) 
v_sub_u32 v4, v0, v3 // (op 1503) 

v_lshrrev_b32 v3, 4, v0 // (op 1503) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1503) 


v_lshrrev_b32 v3, 7, v1 // (op 1503) 
v_lshlrev_b32 v4, 4, v3 // (op 1503) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1503) 


v_lshlrev_b32 v0, 2, v3 // (op 1503) 

v_lshrrev_b32 v3, 3, v0 // (op 1503) 

v_lshrrev_b32 v0, 7, v1 // (op 1614) 
v_and_b32 v4, 15, v1 // (op 1614) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1614) 


v_and_b32 v0, 48, v1 // (op 1614) 
v_add_u32 v4, v0, v5 // (op 1614) 


v_and_b32 v0, 63, v1 // (op 1714) 
v_and_b32 v5, 48, v1 // (op 1714) 
v_sub_u32 v6, v0, v5 // (op 1714) 

v_lshrrev_b32 v5, 4, v0 // (op 1714) 

v_lshl_add_u32 v0, v5, 5, v6 // (op 1714) 


v_lshrrev_b32 v5, 7, v1 // (op 1714) 
v_lshlrev_b32 v6, 4, v5 // (op 1714) 

v_lshl_add_u32 v5, v0, 4, v6 // (op 1714) 


v_lshlrev_b32 v0, 2, v5 // (op 1714) 

v_lshrrev_b32 v5, 3, v0 // (op 1714) 

v_lshrrev_b32 v0, 7, v1 // (op 1767) 
v_and_b32 v6, 15, v1 // (op 1767) 
v_lshl_add_u32 v7, v0, 4, v6 // (op 1767) 


v_and_b32 v0, 48, v1 // (op 1767) 
v_add_u32 v6, v0, v7 // (op 1767) 


v_lshrrev_b32 v0, 7, v1 // (op 1446) 
v_and_b32 v7, 63, v1 // (op 1446) 
v_lshrrev_b32 v8, 4, v7 // (op 1446) 
v_lshlrev_b32 v9, 2, v8 // (op 1446) 

v_lshl_add_u32 v8, v0, 4, v9 // (op 1446) 


v_lshl_add_u32 v0, s16, 4, v8 // (op 1446) 

v_lshrrev_b32 v8, 2, v7 // (op 1446) 

v_and_b32 v7, 3, v8 // (op 1446) 

v_and_b32 v8, 3, v1 // (op 1446) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 1446) 


v_lshrrev_b32 v7, 6, v1 // (op 1446) 
v_and_b32 v8, 1, v7 // (op 1446) 

v_lshl_add_u32 v7, v8, 4, v9 // (op 1446) 


v_lshl_add_u32 v8, s17, 6, v7 // (op 1446) 

v_mul_lo_u32 v7, v8, s2 // (op 1446) 

v_add_lshl_u32 v8, v0, v7, 1 // (op 1446) 


v_lshrrev_b32 v0, 6, v1 // (op 1561) 
v_and_b32 v7, 1, v0 // (op 1561) 

v_and_b32 v0, 15, v1 // (op 1561) 
v_lshl_add_u32 v9, v7, 4, v0 // (op 1561) 


v_and_b32 v0, 48, v1 // (op 1561) 
v_lshl_add_u32 v7, v9, 8, v0 // (op 1561) 


v_lshrrev_b32 v0, 6, v1 // (op 1672) 
v_and_b32 v9, 1, v0 // (op 1672) 

v_and_b32 v0, 15, v1 // (op 1672) 
s_mul_i32 s1, s2, s9 // (op 1448) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1448) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1448) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1448) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1448) most significant: sum
s_add_u32 s1, s1, s19 // (op 1448) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1448) 

v_lshl_add_u32 v10, v9, 4, v0 // (op 1672) 


v_and_b32 v0, 63, v1 // (op 1672) 
v_lshrrev_b32 v9, 4, v0 // (op 1672) 

v_lshl_add_u32 v0, v10, 3, v9 // (op 1672) 


v_lshrrev_b32 v9, 7, v1 // (op 1688) 
v_and_b32 v10, 63, v1 // (op 1688) 
v_lshrrev_b32 v11, 4, v10 // (op 1688) 
v_lshlrev_b32 v12, 2, v11 // (op 1688) 

v_lshl_add_u32 v11, v9, 4, v12 // (op 1688) 


v_lshl_add_u32 v9, s16, 4, v11 // (op 1688) 

v_lshrrev_b32 v11, 2, v10 // (op 1688) 

v_and_b32 v10, 3, v11 // (op 1688) 

v_and_b32 v11, 3, v1 // (op 1688) 
v_lshl_add_u32 v12, v10, 2, v11 // (op 1688) 


v_lshrrev_b32 v10, 6, v1 // (op 1688) 
v_and_b32 v11, 1, v10 // (op 1688) 

v_lshl_add_u32 v10, v11, 4, v12 // (op 1688) 


v_lshl_add_u32 v11, s17, 6, v10 // (op 1688) 

v_mul_lo_u32 v10, v11, s4 // (op 1688) 

v_add_lshl_u32 v11, v9, v10, 1 // (op 1688) 


v_lshrrev_b32 v9, 6, v1 // (op 1743) 
v_and_b32 v10, 1, v9 // (op 1743) 

v_and_b32 v9, 15, v1 // (op 1743) 
v_lshl_add_u32 v12, v10, 4, v9 // (op 1743) 


v_and_b32 v9, 48, v1 // (op 1743) 
v_lshl_add_u32 v10, v12, 8, v9 // (op 1743) 


v_lshrrev_b32 v9, 6, v1 // (op 1796) 
v_and_b32 v12, 1, v9 // (op 1796) 

v_and_b32 v9, 15, v1 // (op 1796) 
s_mul_i32 s1, s4, s9 // (op 1690) most significant: low of low * high
v_lshl_add_u32 v13, v12, 4, v9 // (op 1796) 


v_and_b32 v9, 63, v1 // (op 1796) 
v_lshrrev_b32 v12, 4, v9 // (op 1796) 

v_lshl_add_u32 v9, v13, 3, v12 // (op 1796) 


s_mul_i32 s38, s5, s8 // (op 1690) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1690) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1690) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1690) most significant: sum
s_add_u32 s1, s1, s39 // (op 1690) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 1690) 

s_mov_b32 s40, s6 // (op 1448) reg expression
s_mov_b32 s41, s7 // (op 1448) reg expression
s_mov_b32 s42, s18 // (op 1448) reg expression
s_mov_b32 s43, 131072 // (op 1448) call()

s_mov_b32 s44, s10 // (op 1690) reg expression

s_mov_b32 s45, s11 // (op 1690) reg expression
s_mov_b32 s46, s38 // (op 1690) reg expression
s_mov_b32 s47, 131072 // (op 1690) call()


v_mov_b32 v12, s14 // (op 2) Move value
v_mov_b32 v13, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 402) call()
v_accvgpr_write a5, 0.00000 // (op 402) call()
v_accvgpr_write a6, 0.00000 // (op 402) call()
v_accvgpr_write a7, 0.00000 // (op 402) call()
s_lshl_b32 s0, s16, 4 // (op 1213) 
v_lshlrev_b32 v14, 1, v1 // (op 1213) 
s_mov_b32 s7, 0 // (op 1213) convert
s_mov_b32 s6, s0 // (op 1213) convert

v_mul_lo_u32 v17, v14, s13 // (op 1213) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 1213) most significant: high of low * low
v_mul_lo_u32 v16, v14, s12 // (op 1213) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 1213) most significant: sum


v_mov_b32 v14, s6 // (op 1213) 
v_mov_b32 v15, s7 // (op 1213) 
v_add_co_u32 v18, s[0:1], v14, v16 // (op 1213) least significant half
v_addc_co_u32 v19, s[0:1], v15, v17, s[0:1] // (op 1213) most significant half



v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 1213) 


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 1213) 

v_mov_b32 v16, v14 // (op 1213) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 1215) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1215) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1215) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1215) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1215) most significant: sum
s_add_u32 s1, s1, s7 // (op 1215) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1215) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1215) 

s_mov_b32 s48, s20 // (op 1215) reg expression
s_mov_b32 s49, s21 // (op 1215) reg expression
s_mov_b32 s50, s0 // (op 1215) reg expression
s_mov_b32 s51, 131072 // (op 1215) call()

s_lshl_b32 s0, s16, 4 // (op 1232) 
v_lshlrev_b32 v14, 1, v1 // (op 1232) 
s_mov_b32 s7, 0 // (op 1232) convert
s_mov_b32 s6, s0 // (op 1232) convert

v_mul_lo_u32 v19, v14, s13 // (op 1232) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 1232) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 1232) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 1232) most significant: sum


v_mov_b32 v17, s6 // (op 1232) 
v_mov_b32 v14, s7 // (op 1232) 
v_add_co_u32 v20, s[0:1], v17, v18 // (op 1232) least significant half
v_addc_co_u32 v21, s[0:1], v14, v19, s[0:1] // (op 1232) most significant half



v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 1232) 


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 1232) 

v_mov_b32 v17, v14 // (op 1232) convert

s_mul_i32 s1, s12, s23 // (op 1234) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1234) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1234) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1234) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1234) most significant: sum
s_add_u32 s1, s1, s7 // (op 1234) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1234) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1234) 

s_mov_b32 s48, s20 // (op 1234) reg expression
s_mov_b32 s49, s21 // (op 1234) reg expression
s_mov_b32 s50, s0 // (op 1234) reg expression
s_mov_b32 s51, 131072 // (op 1234) call()

v_and_b32 v14, -16, v1 // (op 1277) 
v_sub_u32 v15, v1, v14 // (op 1277) 

v_lshrrev_b32 v14, 4, v1 // (op 1277) 
v_lshlrev_b32 v18, 3, v14 // (op 1277) 

v_lshl_add_u32 v14, s17, 6, v18 // (op 1277) 

v_mul_lo_u32 v18, v14, s24 // (op 1277) 

v_lshl_add_u32 v14, v15, 4, v18 // (op 1277) 


s_mul_i32 s1, s24, s9 // (op 1279) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1279) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1279) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1279) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1279) most significant: sum
s_add_u32 s1, s1, s7 // (op 1279) most significant: sum


s_mov_b32 s52, s26 // (op 1279) reg expression
s_mov_b32 s53, s27 // (op 1279) reg expression
s_mov_b32 s54, s0 // (op 1279) reg expression
s_mov_b32 s55, 131072 // (op 1279) call()

v_and_b32 v15, -16, v1 // (op 1296) 
v_sub_u32 v18, v1, v15 // (op 1296) 

v_lshrrev_b32 v15, 4, v1 // (op 1296) 
v_lshlrev_b32 v19, 3, v15 // (op 1296) 

v_lshl_add_u32 v15, s17, 6, v19 // (op 1296) 

v_mul_lo_u32 v19, v15, s24 // (op 1296) 

v_lshl_add_u32 v15, v18, 4, v19 // (op 1296) 


s_mul_i32 s1, s24, s9 // (op 1298) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1298) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1298) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1298) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1298) most significant: sum
s_add_u32 s1, s1, s7 // (op 1298) most significant: sum


s_mov_b32 s52, s26 // (op 1298) reg expression
s_mov_b32 s53, s27 // (op 1298) reg expression
s_mov_b32 s54, s0 // (op 1298) reg expression
s_mov_b32 s55, 131072 // (op 1298) call()

v_and_b32 v18, -16, v1 // (op 1341) 
v_sub_u32 v19, v1, v18 // (op 1341) 

v_lshl_add_u32 v18, s16, 4, v19 // (op 1341) 

v_lshrrev_b32 v19, 4, v1 // (op 1341) 
v_mul_lo_u32 v20, v19, s28 // (op 1341) 

v_add_u32 v19, v18, v20 // (op 1341) 


s_mul_i32 s1, s28, s33 // (op 1343) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1343) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1343) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1343) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1343) most significant: sum
s_add_u32 s1, s1, s7 // (op 1343) most significant: sum


s_mov_b32 s56, s30 // (op 1343) reg expression
s_mov_b32 s57, s31 // (op 1343) reg expression
s_mov_b32 s58, s0 // (op 1343) reg expression
s_mov_b32 s59, 131072 // (op 1343) call()

v_and_b32 v18, -16, v1 // (op 1360) 
v_sub_u32 v20, v1, v18 // (op 1360) 

v_lshl_add_u32 v18, s16, 4, v20 // (op 1360) 

v_lshrrev_b32 v20, 4, v1 // (op 1360) 
v_mul_lo_u32 v21, v20, s28 // (op 1360) 

v_add_u32 v20, v18, v21 // (op 1360) 


s_mul_i32 s1, s28, s33 // (op 1362) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1362) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1362) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1362) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1362) most significant: sum
s_add_u32 s1, s1, s7 // (op 1362) most significant: sum


s_mov_b32 s56, s30 // (op 1362) reg expression
s_mov_b32 s57, s31 // (op 1362) reg expression
s_mov_b32 s58, s0 // (op 1362) reg expression
s_mov_b32 s59, 131072 // (op 1362) call()

v_and_b32 v18, -2, v1 // (op 1405) 
v_sub_u32 v21, v1, v18 // (op 1405) 

v_lshrrev_b32 v18, 1, v1 // (op 1405) 
v_lshl_add_u32 v22, s17, 6, v18 // (op 1405) 

v_mul_lo_u32 v18, v22, s34 // (op 1405) 

v_lshl_add_u32 v22, v21, 2, v18 // (op 1405) 


s_mul_i32 s1, s34, s9 // (op 1407) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1407) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1407) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1407) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1407) most significant: sum
s_add_u32 s1, s1, s7 // (op 1407) most significant: sum


s_mov_b32 s60, s36 // (op 1407) reg expression
s_mov_b32 s61, s37 // (op 1407) reg expression
s_mov_b32 s62, s0 // (op 1407) reg expression
s_mov_b32 s63, 131072 // (op 1407) call()

v_and_b32 v18, -2, v1 // (op 1424) 
v_sub_u32 v21, v1, v18 // (op 1424) 

v_lshrrev_b32 v18, 1, v1 // (op 1424) 
v_lshl_add_u32 v23, s17, 6, v18 // (op 1424) 

v_mul_lo_u32 v18, v23, s34 // (op 1424) 

v_lshl_add_u32 v23, v21, 2, v18 // (op 1424) 


s_mul_i32 s1, s34, s9 // (op 1426) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1426) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1426) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1426) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1426) most significant: sum
s_add_u32 s1, s1, s7 // (op 1426) most significant: sum


s_mov_b32 s60, s36 // (op 1426) reg expression
s_mov_b32 s61, s37 // (op 1426) reg expression
s_mov_b32 s62, s0 // (op 1426) reg expression
s_mov_b32 s63, 131072 // (op 1426) call()


v_mov_b64 v[24:25], s[24:25] // (op 1037) convert

v_mov_b32 v18, v14 // (op 1037) 
buffer_load_dwordx4 v[26:29], v18, s[52:55], 0 offen // (op 1037) Load value
v_add_u32 v18, v18, v24 // (op 1037) 
buffer_load_dwordx4 v[30:33], v18, s[52:55], 0 offen // (op 1037) Load value
v_add_u32 v18, v18, v24 // (op 1037) 
buffer_load_dwordx4 v[34:37], v18, s[52:55], 0 offen // (op 1037) Load value
v_add_u32 v18, v18, v24 // (op 1037) 
buffer_load_dwordx4 v[38:41], v18, s[52:55], 0 offen // (op 1037) Load value
v_add_u32 v18, v18, v24 // (op 1037) 
buffer_load_dwordx4 v[42:45], v18, s[52:55], 0 offen // (op 1037) Load value
v_add_u32 v18, v18, v24 // (op 1037) 
buffer_load_dwordx4 v[46:49], v18, s[52:55], 0 offen // (op 1037) Load value
v_add_u32 v18, v18, v24 // (op 1037) 
buffer_load_dwordx4 v[50:53], v18, s[52:55], 0 offen // (op 1037) Load value
v_add_u32 v18, v18, v24 // (op 1037) 
buffer_load_dwordx4 v[54:57], v18, s[52:55], 0 offen // (op 1037) Load value





buffer_load_dword v18, v22, s[60:63], 0 offen // (op 1040) Load value

s_lshl_b64 s[0:1], s[12:13], 2 // (op 1043) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1043) 


v_mov_b32 v21, v16 // (op 1043) reg expression
buffer_load_dwordx2 v[58:59], v21, s[48:51], 0 offen // (op 1043) Load value
v_add_u32 v21, v21, v24 // (op 1043) 
buffer_load_dwordx2 v[60:61], v21, s[48:51], 0 offen // (op 1043) Load value




v_mov_b32 v21, v19 // (op 1046) reg expression

buffer_load_ubyte v24, v21, s[56:59], 0 offen offset: 0 // (op 1046) Load value


v_mov_b64 v[62:63], s[24:25] // (op 1049) convert

v_mov_b32 v21, v14 // (op 1049) 
buffer_load_dwordx4 v[64:67], v21, s[52:55], 0 offen offset:256 // (op 1049) Load value
v_add_u32 v21, v21, v62 // (op 1049) 
buffer_load_dwordx4 v[68:71], v21, s[52:55], 0 offen offset:256 // (op 1049) Load value
v_add_u32 v21, v21, v62 // (op 1049) 
buffer_load_dwordx4 v[72:75], v21, s[52:55], 0 offen offset:256 // (op 1049) Load value
v_add_u32 v21, v21, v62 // (op 1049) 
buffer_load_dwordx4 v[76:79], v21, s[52:55], 0 offen offset:256 // (op 1049) Load value
v_add_u32 v21, v21, v62 // (op 1049) 
buffer_load_dwordx4 v[80:83], v21, s[52:55], 0 offen offset:256 // (op 1049) Load value
v_add_u32 v21, v21, v62 // (op 1049) 
buffer_load_dwordx4 v[84:87], v21, s[52:55], 0 offen offset:256 // (op 1049) Load value
v_add_u32 v21, v21, v62 // (op 1049) 
buffer_load_dwordx4 v[88:91], v21, s[52:55], 0 offen offset:256 // (op 1049) Load value
v_add_u32 v21, v21, v62 // (op 1049) 
buffer_load_dwordx4 v[92:95], v21, s[52:55], 0 offen offset:256 // (op 1049) Load value






buffer_load_dword v14, v22, s[60:63], 0 offen offset:8 // (op 1052) Load value


s_lshl_b64 s[0:1], s[12:13], 2 // (op 1055) 
v_lshrrev_b64 v[62:63], 3, s[0:1] // (op 1055) 


s_lshl_b64 s[0:1], s[12:13], 8 // (op 1055) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 1055) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 1055) 

v_add_u32 v25, v16, s0 // (op 1055) 

buffer_load_dwordx2 v[96:97], v25, s[48:51], 0 offen // (op 1055) Load value
v_add_u32 v25, v25, v62 // (op 1055) 
buffer_load_dwordx2 v[98:99], v25, s[48:51], 0 offen // (op 1055) Load value





s_lshl_b32 s0, s28, 3 // (op 1058) 
v_add_u32 v16, v19, s0 // (op 1058) 


buffer_load_ubyte v25, v16, s[56:59], 0 offen offset: 0 // (op 1058) Load value


v_and_b32 v16, -16, v1 // (op 1544) 
v_sub_u32 v19, v1, v16 // (op 1544) 

v_lshlrev_b32 v16, 4, v19 // (op 1544) 

v_lshrrev_b32 v19, 4, v1 // (op 1544) 
v_lshl_add_u32 v21, v19, 11, v16 // (op 1544) 





s_waitcnt vmcnt(23)
ds_write_b128 v21, v[26:29] // (op 1060) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v21, v[30:33] offset:256 // (op 1060) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v21, v[34:37] offset:512 // (op 1060) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v21, v[38:41] offset:768 // (op 1060) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v21, v[42:45] offset:1024 // (op 1060) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v21, v[46:49] offset:1280 // (op 1060) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v21, v[50:53] offset:1536 // (op 1060) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v21, v[54:57] offset:1792 // (op 1060) Store local data 

v_and_b32 v16, -2, v1 // (op 1655) 
v_sub_u32 v19, v1, v16 // (op 1655) 

v_lshlrev_b32 v16, 2, v19 // (op 1655) 

v_lshrrev_b32 v19, 1, v1 // (op 1655) 
v_lshl_add_u32 v21, v19, 3, v16 // (op 1655) 





s_waitcnt vmcnt(15)
ds_write_b32 v21, v18 offset:16384 // (op 1061) Store local data 

v_lshlrev_b32 v16, 7, v1 // (op 1491) 
v_lshrrev_b32 v19, 3, v16 // (op 1491) 




s_waitcnt vmcnt(14)
ds_write_b64 v19, v[58:59] offset:16896 // (op 1062) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v19, v[60:61] offset:16904 // (op 1062) Store local data 

v_and_b32 v16, -16, v1 // (op 1602) 
v_sub_u32 v19, v1, v16 // (op 1602) 

v_and_b32 v16, 4294967280, v1 // (op 1602) 
v_add_u32 v21, v16, v19 // (op 1602) 





s_waitcnt vmcnt(12)
ds_write_b8 v21, v24 offset:18944 // (op 1063) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1866) 



ds_read_b128 v[100:103], v7 // (op 1081) Load local data 
ds_read_b128 v[104:107], v7 offset:64 // (op 1081) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1907) 



ds_read_u8 v16, v0 offset:16384 // (op 1094) Load local data 



ds_read_b128 v[108:111], v7 offset:8192 // (op 1086) Load local data 
ds_read_b128 v[112:115], v7 offset:8256 // (op 1086) Load local data 



ds_read_u8 v19, v0 offset:16640 // (op 1099) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1847) 



ds_read_b64_tr_b4 v[116:117], v3 offset:16896 // (op 1077) Transpose load local data 
ds_read_b64_tr_b4 v[118:119], v3 offset:17024 // (op 1077) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1887) 


ds_read_u8 v21, v4 offset:18944 // (op 1090) Load local data 
v_lshlrev_b32 v22, 7, v1 // (op 1462) 
v_lshrrev_b32 v62, 3, v22 // (op 1462) 

v_and_b32 v22, -16, v1 // (op 1515) 
v_sub_u32 v63, v1, v22 // (op 1515) 

v_lshlrev_b32 v22, 4, v63 // (op 1515) 

v_lshrrev_b32 v63, 4, v1 // (op 1515) 
v_lshl_add_u32 v120, v63, 11, v22 // (op 1515) 


v_and_b32 v22, -16, v1 // (op 1573) 
v_sub_u32 v63, v1, v22 // (op 1573) 

v_and_b32 v22, 4294967280, v1 // (op 1573) 
v_add_u32 v121, v22, v63 // (op 1573) 


v_and_b32 v22, -2, v1 // (op 1626) 
v_sub_u32 v63, v1, v22 // (op 1626) 

v_lshlrev_b32 v22, 2, v63 // (op 1626) 

v_lshrrev_b32 v63, 1, v1 // (op 1626) 
v_lshl_add_u32 v122, v63, 3, v22 // (op 1626) 


v_lshlrev_b32 v22, 7, v1 // (op 1702) 
v_lshrrev_b32 v63, 3, v22 // (op 1702) 

v_and_b32 v22, -16, v1 // (op 1726) 
v_sub_u32 v123, v1, v22 // (op 1726) 

v_lshlrev_b32 v22, 4, v123 // (op 1726) 

v_lshrrev_b32 v123, 4, v1 // (op 1726) 
v_lshl_add_u32 v124, v123, 11, v22 // (op 1726) 


v_and_b32 v22, -16, v1 // (op 1755) 
v_sub_u32 v123, v1, v22 // (op 1755) 

v_and_b32 v22, 4294967280, v1 // (op 1755) 
v_add_u32 v125, v22, v123 // (op 1755) 


v_and_b32 v22, -2, v1 // (op 1779) 
v_sub_u32 v123, v1, v22 // (op 1779) 

v_lshlrev_b32 v22, 2, v123 // (op 1779) 

v_lshrrev_b32 v123, 1, v1 // (op 1779) 
v_lshl_add_u32 v126, v123, 3, v22 // (op 1779) 


s_mov_b32 s0, 0 // (op 404) call()
s_ashr_i64 s[6:7], s[22:23], 8 // (op 403) 
s_mov_b32 s1, s6 // (op 403) convert

s_and_b32 s6, s1, 4294967294 // (op 403) 

s_cmp_lt_u32 s0, s6 // (op 403) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_403 // (op 403) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_403 if false)
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_403:

s_add_u32 s0, s0, 2 // (op 405) 
v_mov_b64 v[128:129], s[24:25] // (op 10) convert

v_mov_b32 v22, v15 // (op 10) 
v_mov_b32 v123, v22 // (op 10) 
buffer_load_dwordx4 v[26:29], v123, s[52:55], 0 offen offset:512 // (op 10) Load value
v_add_u32 v123, v123, v128 // (op 10) 
buffer_load_dwordx4 v[30:33], v123, s[52:55], 0 offen offset:512 // (op 10) Load value
v_add_u32 v123, v123, v128 // (op 10) 
buffer_load_dwordx4 v[34:37], v123, s[52:55], 0 offen offset:512 // (op 10) Load value
v_add_u32 v123, v123, v128 // (op 10) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[116:119], v[100:107], a[0:3], v21, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 90) 
buffer_load_dwordx4 v[38:41], v123, s[52:55], 0 offen offset:512 // (op 10) Load value
s_barrier  // (op 1856) 
v_add_u32 v123, v123, v128 // (op 10) 



buffer_load_dwordx4 v[42:45], v123, s[52:55], 0 offen offset:512 // (op 10) Load value
v_add_u32 v123, v123, v128 // (op 10) 
buffer_load_dwordx4 v[46:49], v123, s[52:55], 0 offen offset:512 // (op 10) Load value
v_add_u32 v123, v123, v128 // (op 10) 
ds_read_b64_tr_b4 v[130:131], v3 offset:17920 // (op 93) Transpose load local data 
ds_read_b64_tr_b4 v[132:133], v3 offset:18048 // (op 93) Transpose load local data 
buffer_load_dwordx4 v[50:53], v123, s[52:55], 0 offen offset:512 // (op 10) Load value
s_barrier  // (op 1876) 
v_add_u32 v123, v123, v128 // (op 10) 



ds_read_b128 v[100:103], v7 offset:128 // (op 97) Load local data 
ds_read_b128 v[104:107], v7 offset:192 // (op 97) Load local data 
buffer_load_dwordx4 v[54:57], v123, s[52:55], 0 offen offset:512 // (op 10) Load value




s_barrier  // (op 1897) 
v_mov_b32 v123, v23 // (op 25) 



buffer_load_dword v18, v123, s[60:63], 0 offen offset:16 // (op 25) Load value
ds_read_u8 v127, v4 offset:19008 // (op 101) Load local data 

s_barrier  // (op 1917) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 4) 



v_lshrrev_b64 v[128:129], 3, s[6:7] // (op 4) 


v_mov_b32 v134, v17 // (op 4) 
s_lshl_b64 s[6:7], s[12:13], 8 // (op 4) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 4) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 4) 

s_lshl_b32 s1, s6, 1 // (op 4) 

v_add_u32 v135, v134, s1 // (op 4) 

ds_read_u8 v16, v0 offset:16388 // (op 105) Load local data 
buffer_load_dwordx2 v[58:59], v135, s[48:51], 0 offen // (op 4) Load value
v_add_u32 v135, v135, v128 // (op 4) 
buffer_load_dwordx2 v[60:61], v135, s[48:51], 0 offen // (op 4) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[130:133], v[100:107], a[0:3], v127, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 107) 


v_mov_b32 v128, v20 // (op 17) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[116:119], v[108:115], a[4:7], v21, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 451) 
s_lshl_b32 s1, s28, 4 // (op 17) 



ds_read_b128 v[108:111], v7 offset:8320 // (op 455) Load local data 
ds_read_b128 v[112:115], v7 offset:8384 // (op 455) Load local data 
v_add_u32 v129, v128, s1 // (op 17) 





ds_read_u8 v19, v0 offset:16644 // (op 459) Load local data 
buffer_load_ubyte v24, v129, s[56:59], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1937) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[130:133], v[108:115], a[4:7], v127, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 460) 





s_waitcnt vmcnt(23)
ds_write_b128 v124, v[64:67] offset:19072 // (op 306) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v124, v[68:71] offset:19328 // (op 306) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v124, v[72:75] offset:19584 // (op 306) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v124, v[76:79] offset:19840 // (op 306) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v124, v[80:83] offset:20096 // (op 306) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v124, v[84:87] offset:20352 // (op 306) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v124, v[88:91] offset:20608 // (op 306) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v124, v[92:95] offset:20864 // (op 306) Store local data 
s_barrier  // (op 1949) 



s_waitcnt vmcnt(15)
ds_write_b32 v126, v14 offset:35456 // (op 308) Store local data 
s_barrier  // (op 1930) 



s_waitcnt vmcnt(14)
ds_write_b64 v63, v[96:97] offset:35968 // (op 287) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v63, v[98:99] offset:35976 // (op 287) Store local data 
s_barrier  // (op 1943) 



s_waitcnt vmcnt(12)
ds_write_b8 v125, v25 offset:38016 // (op 307) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1933) 



ds_read_b128 v[100:103], v10 offset:19072 // (op 291) Load local data 
ds_read_b128 v[104:107], v10 offset:19136 // (op 291) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1946) 



ds_read_u8 v16, v9 offset:35456 // (op 295) Load local data 



ds_read_b128 v[136:139], v10 offset:27264 // (op 474) Load local data 
ds_read_b128 v[140:143], v10 offset:27328 // (op 474) Load local data 



ds_read_u8 v127, v9 offset:35712 // (op 478) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1927) 



ds_read_b64_tr_b4 v[116:117], v5 offset:35968 // (op 289) Transpose load local data 
ds_read_b64_tr_b4 v[118:119], v5 offset:36096 // (op 289) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1940) 


ds_read_u8 v21, v6 offset:38016 // (op 293) Load local data 

v_mov_b64 v[132:133], s[24:25] // (op 309) convert

v_mov_b32 v135, v22 // (op 309) 
buffer_load_dwordx4 v[64:67], v135, s[52:55], 0 offen offset:768 // (op 309) Load value
v_add_u32 v135, v135, v132 // (op 309) 
buffer_load_dwordx4 v[68:71], v135, s[52:55], 0 offen offset:768 // (op 309) Load value
v_add_u32 v135, v135, v132 // (op 309) 
buffer_load_dwordx4 v[72:75], v135, s[52:55], 0 offen offset:768 // (op 309) Load value
v_add_u32 v135, v135, v132 // (op 309) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[116:119], v[100:107], a[0:3], v21, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 296) 
buffer_load_dwordx4 v[76:79], v135, s[52:55], 0 offen offset:768 // (op 309) Load value
v_add_u32 v135, v135, v132 // (op 309) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[116:119], v[136:143], a[4:7], v21, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 461) 
buffer_load_dwordx4 v[80:83], v135, s[52:55], 0 offen offset:768 // (op 309) Load value


v_add_u32 v135, v135, v132 // (op 309) 
buffer_load_dwordx4 v[84:87], v135, s[52:55], 0 offen offset:768 // (op 309) Load value
v_add_u32 v135, v135, v132 // (op 309) 
ds_read_b64_tr_b4 v[136:137], v5 offset:36992 // (op 298) Transpose load local data 
ds_read_b64_tr_b4 v[138:139], v5 offset:37120 // (op 298) Transpose load local data 
buffer_load_dwordx4 v[88:91], v135, s[52:55], 0 offen offset:768 // (op 309) Load value
v_add_u32 v135, v135, v132 // (op 309) 



s_barrier  // (op 1998) 
buffer_load_dwordx4 v[92:95], v135, s[52:55], 0 offen offset:768 // (op 309) Load value


ds_read_b128 v[140:143], v10 offset:19200 // (op 300) Load local data 

ds_read_b128 v[144:147], v10 offset:19264 // (op 300) Load local data 





buffer_load_dword v14, v123, s[60:63], 0 offen offset:24 // (op 311) Load value
v_add_u32 v15, 512, v15 // (op 1288) 

ds_read_u8 v22, v6 offset:38080 // (op 302) Load local data 

v_add_u32 v23, 16, v23 // (op 1416) 



s_barrier  // (op 2002) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 286) 
v_lshrrev_b64 v[132:133], 3, s[6:7] // (op 286) 


s_lshl_b64 s[6:7], s[12:13], 8 // (op 286) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 286) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 286) 

s_mul_i32 s1, 3, s6 // (op 286) 

v_add_u32 v123, v134, s1 // (op 286) 

ds_read_u8 v127, v9 offset:35460 // (op 304) Load local data 
buffer_load_dwordx2 v[96:97], v123, s[48:51], 0 offen // (op 286) Load value



v_add_u32 v123, v123, v132 // (op 286) 
buffer_load_dwordx2 v[98:99], v123, s[48:51], 0 offen // (op 286) Load value


ds_read_b128 v[148:151], v10 offset:27392 // (op 465) Load local data 

ds_read_b128 v[152:155], v10 offset:27456 // (op 465) Load local data 


s_lshl_b32 s1, s28, 3 // (op 310) 



s_barrier  // (op 2038) 
ds_read_u8 v123, v9 offset:35716 // (op 469) Load local data 
s_mul_i32 s6, 3, s1 // (op 310) 

v_add_u32 v129, v128, s6 // (op 310) 


s_barrier  // (op 2042) 
buffer_load_ubyte v25, v129, s[56:59], 0 offen offset: 0 // (op 310) Load value

s_lshl_b64 s[6:7], s[12:13], 8 // (op 1224) 

s_lshl_b64 s[10:11], s[6:7], 2 // (op 1224) 

s_barrier  // (op 1869) 
s_lshl_b32 s1, s28, 4 // (op 1352) 



v_add_u32 v20, v20, s1 // (op 1352) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 1224) 

s_lshl_b32 s1, s6, 1 // (op 1224) 

v_add_u32 v17, v17, s1 // (op 1224) 

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[140:147], a[0:3], v22, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 305) 
s_waitcnt vmcnt(23)
ds_write_b128 v120, v[26:29] // (op 13) Store local data 


s_waitcnt vmcnt(22)
ds_write_b128 v120, v[30:33] offset:256 // (op 13) Store local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[148:155], a[4:7], v22, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 470) 
s_waitcnt vmcnt(21)
ds_write_b128 v120, v[34:37] offset:512 // (op 13) Store local data 




s_waitcnt vmcnt(20)
ds_write_b128 v120, v[38:41] offset:768 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v120, v[42:45] offset:1024 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v120, v[46:49] offset:1280 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v120, v[50:53] offset:1536 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v120, v[54:57] offset:1792 // (op 13) Store local data 
s_barrier  // (op 1910) 



s_waitcnt vmcnt(15)
ds_write_b32 v122, v18 offset:16384 // (op 30) Store local data 
s_barrier  // (op 1850) 



s_waitcnt vmcnt(14)
ds_write_b64 v62, v[58:59] offset:16896 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v62, v[60:61] offset:16904 // (op 6) Store local data 
s_barrier  // (op 1890) 



s_waitcnt vmcnt(12)
ds_write_b8 v121, v24 offset:18944 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1879) 



ds_read_b128 v[100:103], v7 // (op 77) Load local data 
ds_read_b128 v[104:107], v7 offset:64 // (op 77) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1920) 



ds_read_u8 v16, v0 offset:16384 // (op 87) Load local data 



ds_read_b128 v[108:111], v7 offset:8192 // (op 446) Load local data 
ds_read_b128 v[112:115], v7 offset:8256 // (op 446) Load local data 



ds_read_u8 v19, v0 offset:16640 // (op 450) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1859) 



ds_read_b64_tr_b4 v[116:117], v3 offset:16896 // (op 72) Transpose load local data 
ds_read_b64_tr_b4 v[118:119], v3 offset:17024 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1900) 


ds_read_u8 v21, v4 offset:18944 // (op 82) Load local data 
s_ashr_i64 s[6:7], s[22:23], 8 // (op 403) 
s_mov_b32 s1, s6 // (op 403) convert

s_and_b32 s6, s1, 4294967294 // (op 403) 

s_cmp_lt_u32 s0, s6 // (op 403) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_403 // (op 403) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_403 if true)
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_403:

















v_lshlrev_b32 v15, 1, v1 // (op 1186) 
s_ashr_i64 s[0:1], s[22:23], 8 // (op 1186) 
s_lshr_b32 s6, s0, 1 // (op 1186) 

v_lshl_add_u32 v20, s6, 9, v15 // (op 1186) 


s_lshl_b32 s0, s16, 4 // (op 1186) 
v_mul_lo_u32 v23, v20, s13 // (op 1186) most significant: low of low * high
v_mul_hi_u32 v15, v20, s12 // (op 1186) most significant: high of low * low
v_mul_lo_u32 v22, v20, s12 // (op 1186) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 1186) most significant: sum


s_mov_b32 s7, 0 // (op 1186) convert
s_mov_b32 s6, s0 // (op 1186) convert

v_mov_b32 v15, s6 // (op 1186) 
v_mov_b32 v20, s7 // (op 1186) 
v_add_co_u32 v26, s[0:1], v15, v22 // (op 1186) least significant half
v_addc_co_u32 v27, s[0:1], v20, v23, s[0:1] // (op 1186) most significant half



v_lshlrev_b64 v[26:27], 2, v[26:27] // (op 1186) 


v_lshrrev_b64 v[22:23], 3, v[26:27] // (op 1186) 

v_mov_b32 v15, v22 // (op 1186) convert

s_mul_i32 s1, s12, s23 // (op 1188) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1188) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1188) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1188) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1188) most significant: sum
s_add_u32 s1, s1, s7 // (op 1188) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1188) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1188) 

s_mov_b32 s48, s20 // (op 1188) reg expression
s_mov_b32 s49, s21 // (op 1188) reg expression
s_mov_b32 s50, s0 // (op 1188) reg expression
s_mov_b32 s51, 131072 // (op 1188) call()

v_and_b32 v20, -16, v1 // (op 1254) 
s_ashr_i64 s[0:1], s[22:23], 8 // (op 1254) 
v_sub_u32 v17, v1, v20 // (op 1254) 

s_lshr_b32 s6, s0, 1 // (op 1254) 

v_lshlrev_b32 v18, 4, v17 // (op 1254) 

v_lshl_add_u32 v17, s6, 9, v18 // (op 1254) 


v_lshrrev_b32 v18, 4, v1 // (op 1254) 
v_lshlrev_b32 v20, 3, v18 // (op 1254) 

v_lshl_add_u32 v18, s17, 6, v20 // (op 1254) 

v_mul_lo_u32 v20, v18, s24 // (op 1254) 

v_add_u32 v18, v17, v20 // (op 1254) 


s_mul_i32 s1, s24, s9 // (op 1256) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1256) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1256) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1256) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1256) most significant: sum
s_add_u32 s1, s1, s7 // (op 1256) most significant: sum


s_mov_b32 s52, s26 // (op 1256) reg expression
s_mov_b32 s53, s27 // (op 1256) reg expression
s_mov_b32 s54, s0 // (op 1256) reg expression
s_mov_b32 s55, 131072 // (op 1256) call()

v_and_b32 v17, -16, v1 // (op 1318) 
s_ashr_i64 s[0:1], s[22:23], 8 // (op 1318) 
v_sub_u32 v20, v1, v17 // (op 1318) 

s_lshr_b32 s6, s0, 1 // (op 1318) 

v_lshl_add_u32 v17, s16, 4, v20 // (op 1318) 

v_lshrrev_b32 v20, 4, v1 // (op 1318) 
v_lshl_add_u32 v22, s6, 4, v20 // (op 1318) 


v_mul_lo_u32 v20, v22, s28 // (op 1318) 

v_add_u32 v22, v17, v20 // (op 1318) 


s_mul_i32 s1, s28, s33 // (op 1320) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1320) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1320) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1320) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1320) most significant: sum
s_add_u32 s1, s1, s7 // (op 1320) most significant: sum


s_mov_b32 s56, s30 // (op 1320) reg expression
s_mov_b32 s57, s31 // (op 1320) reg expression
s_mov_b32 s58, s0 // (op 1320) reg expression
s_mov_b32 s59, 131072 // (op 1320) call()

v_and_b32 v17, -2, v1 // (op 1382) 
s_ashr_i64 s[0:1], s[22:23], 8 // (op 1382) 
v_sub_u32 v20, v1, v17 // (op 1382) 

s_lshr_b32 s6, s0, 1 // (op 1382) 

v_lshlrev_b32 v17, 2, v20 // (op 1382) 

v_lshl_add_u32 v20, s6, 4, v17 // (op 1382) 


v_lshrrev_b32 v17, 1, v1 // (op 1382) 
v_lshl_add_u32 v23, s17, 6, v17 // (op 1382) 

v_mul_lo_u32 v17, v23, s34 // (op 1382) 

v_add_u32 v23, v20, v17 // (op 1382) 


s_mul_i32 s1, s34, s9 // (op 1384) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1384) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1384) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1384) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1384) most significant: sum
s_add_u32 s1, s1, s7 // (op 1384) most significant: sum


s_mov_b32 s60, s36 // (op 1384) reg expression
s_mov_b32 s61, s37 // (op 1384) reg expression
s_mov_b32 s62, s0 // (op 1384) reg expression
s_mov_b32 s63, 131072 // (op 1384) call()

v_lshlrev_b32 v17, 7, v1 // (op 1475) 
v_lshrrev_b32 v20, 3, v17 // (op 1475) 

v_and_b32 v17, -16, v1 // (op 1528) 
v_sub_u32 v24, v1, v17 // (op 1528) 

v_lshlrev_b32 v17, 4, v24 // (op 1528) 

v_lshrrev_b32 v24, 4, v1 // (op 1528) 
v_lshl_add_u32 v26, v24, 11, v17 // (op 1528) 


v_and_b32 v17, -16, v1 // (op 1586) 
v_sub_u32 v24, v1, v17 // (op 1586) 

v_and_b32 v17, 4294967280, v1 // (op 1586) 
v_add_u32 v27, v17, v24 // (op 1586) 


v_and_b32 v17, -2, v1 // (op 1639) 
v_sub_u32 v24, v1, v17 // (op 1639) 

v_lshlrev_b32 v17, 2, v24 // (op 1639) 

v_lshrrev_b32 v24, 1, v1 // (op 1639) 
v_lshl_add_u32 v28, v24, 3, v17 // (op 1639) 


s_ashr_i64 s[0:1], s[22:23], 8 // (op 409) 
s_and_b32 s6, s0, 4294967294 // (op 409) 

s_ashr_i64 s[0:1], s[22:23], 8 // (op 408) 
s_mov_b32 s7, s0 // (op 408) convert

s_cmp_lt_u32 s6, s7 // (op 408) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_408 // (op 408) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_408 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_408:

s_add_u32 s6, s6, 1 // (op 410) 
s_lshl_b64 s[0:1], s[12:13], 2 // (op 202) 
v_lshrrev_b64 v[16:17], 3, s[0:1] // (op 202) 


v_mov_b32 v14, v15 // (op 202) 
v_mov_b32 v19, v14 // (op 202) reg expression
buffer_load_dwordx2 v[30:31], v19, s[48:51], 0 offen // (op 202) Load value
v_add_u32 v19, v19, v16 // (op 202) 
buffer_load_dwordx2 v[32:33], v19, s[48:51], 0 offen // (op 202) Load value



s_lshl_b64 s[0:1], s[12:13], 8 // (op 1176) 
s_lshl_b64 s[10:11], s[0:1], 2 // (op 1176) 

s_lshr_b64 s[0:1], s[10:11], 3 // (op 1176) 

v_add_u32 v15, v15, s0 // (op 1176) 


s_barrier  // (op 1853) 
v_mov_b64 v[16:17], s[24:25] // (op 225) convert




v_mov_b32 v14, v18 // (op 225) 
v_mov_b32 v19, v14 // (op 225) 
buffer_load_dwordx4 v[34:37], v19, s[52:55], 0 offen // (op 225) Load value
v_add_u32 v19, v19, v16 // (op 225) 
s_waitcnt vmcnt(2)
ds_write_b64 v20, v[30:31] offset:16896 // (op 203) Store local data 
buffer_load_dwordx4 v[38:41], v19, s[52:55], 0 offen // (op 225) Load value
v_add_u32 v19, v19, v16 // (op 225) 
buffer_load_dwordx4 v[42:45], v19, s[52:55], 0 offen // (op 225) Load value
v_add_u32 v19, v19, v16 // (op 225) 
buffer_load_dwordx4 v[46:49], v19, s[52:55], 0 offen // (op 225) Load value
v_add_u32 v19, v19, v16 // (op 225) 
s_waitcnt vmcnt(4)
ds_write_b64 v20, v[32:33] offset:16904 // (op 203) Store local data 
buffer_load_dwordx4 v[50:53], v19, s[52:55], 0 offen // (op 225) Load value

v_add_u32 v19, v19, v16 // (op 225) 
buffer_load_dwordx4 v[54:57], v19, s[52:55], 0 offen // (op 225) Load value
v_add_u32 v19, v19, v16 // (op 225) 
buffer_load_dwordx4 v[58:61], v19, s[52:55], 0 offen // (op 225) Load value
v_add_u32 v19, v19, v16 // (op 225) 
buffer_load_dwordx4 v[62:65], v19, s[52:55], 0 offen // (op 225) Load value



v_add_u32 v18, 256, v18 // (op 1246) 


s_barrier  // (op 1872) 
v_mov_b32 v14, v22 // (op 226) 



v_mov_b32 v19, v14 // (op 226) reg expression

buffer_load_ubyte v21, v19, s[56:59], 0 offen offset: 0 // (op 226) Load value


s_lshl_b32 s7, s28, 3 // (op 1310) 
v_add_u32 v22, v22, s7 // (op 1310) 

s_waitcnt vmcnt(8)
ds_write_b128 v26, v[34:37] // (op 222) Store local data 


v_mov_b32 v14, v23 // (op 227) 

buffer_load_dword v19, v14, s[60:63], 0 offen // (op 227) Load value

v_add_u32 v23, 8, v23 // (op 1374) 
s_waitcnt vmcnt(8)
ds_write_b128 v26, v[38:41] offset:256 // (op 222) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1862) 
s_waitcnt vmcnt(7)
ds_write_b128 v26, v[42:45] offset:512 // (op 222) Store local data 



ds_read_b64_tr_b4 v[30:31], v3 offset:16896 // (op 205) Transpose load local data 
ds_read_b64_tr_b4 v[32:33], v3 offset:17024 // (op 205) Transpose load local data 
s_waitcnt vmcnt(6)
ds_write_b128 v26, v[46:49] offset:768 // (op 222) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v26, v[50:53] offset:1024 // (op 222) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v26, v[54:57] offset:1280 // (op 222) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v26, v[58:61] offset:1536 // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v26, v[62:65] offset:1792 // (op 222) Store local data 

s_barrier  // (op 1893) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1883) 






ds_read_b128 v[34:37], v7 // (op 207) Load local data 
ds_read_b128 v[38:41], v7 offset:64 // (op 207) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v27, v21 offset:18944 // (op 223) Store local data 

s_barrier  // (op 1913) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1903) 





ds_read_u8 v14, v4 offset:18944 // (op 209) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v28, v19 offset:16384 // (op 224) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1923) 



ds_read_u8 v19, v0 offset:16384 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:33], v[34:41], a[0:3], v14, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 212) 





ds_read_b64_tr_b4 v[34:35], v3 offset:17920 // (op 214) Transpose load local data 
ds_read_b64_tr_b4 v[36:37], v3 offset:18048 // (op 214) Transpose load local data 



ds_read_b128 v[38:41], v7 offset:128 // (op 216) Load local data 
ds_read_b128 v[42:45], v7 offset:192 // (op 216) Load local data 


ds_read_u8 v19, v4 offset:19008 // (op 218) Load local data 



ds_read_u8 v21, v0 offset:16388 // (op 220) Load local data 



ds_read_b128 v[46:49], v7 offset:8192 // (op 416) Load local data 
ds_read_b128 v[50:53], v7 offset:8256 // (op 416) Load local data 



ds_read_u8 v29, v0 offset:16640 // (op 420) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:33], v[46:53], a[4:7], v14, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 421) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[38:45], a[0:3], v19, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 221) 
ds_read_b128 v[46:49], v7 offset:8320 // (op 425) Load local data 


ds_read_b128 v[50:53], v7 offset:8384 // (op 425) Load local data 



ds_read_u8 v14, v0 offset:16644 // (op 429) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[46:53], a[4:7], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 430) 




s_ashr_i64 s[0:1], s[22:23], 8 // (op 408) 
s_mov_b32 s7, s0 // (op 408) convert

s_cmp_lt_u32 s6, s7 // (op 408) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_408 // (op 408) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_408 if true)
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_408:























s_barrier  // (op 2024) 









v_cmp_eq_f32 vcc, 0.00000, v12 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_157
v_accvgpr_read v4, a0 // (op 50) 
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v14, v13, v4 // (op 50) 
v_mul_f32 v15, v13, v5 // (op 50) 
v_mul_f32 v16, v13, v6 // (op 50) 
v_mul_f32 v17, v13, v7 // (op 50) 

v_cvt_f16_f32 v4, v14 // (op 56) 
v_cvt_f16_f32 v5, v15 // (op 56) 
v_cvt_f16_f32 v6, v16 // (op 56) 
v_cvt_f16_f32 v7, v17 // (op 56) 


v_pack_b32_f16 v14, v4, v5 // (op 58) 
v_pack_b32_f16 v15, v6, v7 // (op 58) 

v_mov_b32 v0, v11 // (op 58) reg expression

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 58) Store value



v_accvgpr_read v4, a4 // (op 714) 
v_accvgpr_read v5, a5 // (op 714) 
v_accvgpr_read v6, a6 // (op 714) 
v_accvgpr_read v7, a7 // (op 714) 
v_mul_f32 v14, v13, v4 // (op 714) 
v_mul_f32 v15, v13, v5 // (op 714) 
v_mul_f32 v16, v13, v6 // (op 714) 
v_mul_f32 v17, v13, v7 // (op 714) 

v_cvt_f16_f32 v4, v14 // (op 716) 
v_cvt_f16_f32 v5, v15 // (op 716) 
v_cvt_f16_f32 v6, v16 // (op 716) 
v_cvt_f16_f32 v7, v17 // (op 716) 


v_pack_b32_f16 v14, v4, v5 // (op 717) 
v_pack_b32_f16 v15, v6, v7 // (op 717) 

s_lshl_b32 s0, s4, 6 // (op 717) 
v_add_u32 v0, v11, s0 // (op 717) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 717) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_157
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_157:


v_mov_b32 v0, v8 // (op 170) reg expression

buffer_load_dwordx2 v[6:7], v0, s[40:43], 0 offen // (op 170) Load value

v_accvgpr_read v14, a0 // (op 166) 
v_accvgpr_read v15, a1 // (op 166) 
v_accvgpr_read v16, a2 // (op 166) 
v_accvgpr_read v17, a3 // (op 166) 
v_mul_f32 v18, v13, v14 // (op 166) 
v_mul_f32 v19, v13, v15 // (op 166) 
v_mul_f32 v20, v13, v16 // (op 166) 
v_mul_f32 v21, v13, v17 // (op 166) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v4, v6 // (op 171) 
v_lshrrev_b32 v5, 16, v6 // (op 171) 
v_cvt_f32_f16 v5, v5 // (op 171) 
v_mul_f32 v14, v12, v4 // (op 171) 
v_mul_f32 v15, v12, v5 // (op 171) 
v_cvt_f32_f16 v4, v7 // (op 171) 
v_lshrrev_b32 v5, 16, v7 // (op 171) 
v_cvt_f32_f16 v5, v5 // (op 171) 
v_mul_f32 v16, v12, v4 // (op 171) 
v_mul_f32 v17, v12, v5 // (op 171) 


v_add_f32 v4, v14, v18 // (op 167) 
v_add_f32 v5, v15, v19 // (op 167) 
v_add_f32 v6, v16, v20 // (op 167) 
v_add_f32 v7, v17, v21 // (op 167) 


v_cvt_f16_f32 v14, v4 // (op 168) 
v_cvt_f16_f32 v15, v5 // (op 168) 
v_cvt_f16_f32 v16, v6 // (op 168) 
v_cvt_f16_f32 v17, v7 // (op 168) 


v_pack_b32_f16 v6, v14, v15 // (op 169) 
v_pack_b32_f16 v7, v16, v17 // (op 169) 

v_mov_b32 v0, v11 // (op 169) reg expression

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 169) Store value





s_lshl_b32 s0, s2, 6 // (op 741) 
v_add_u32 v0, v8, s0 // (op 741) 


buffer_load_dwordx2 v[6:7], v0, s[40:43], 0 offen // (op 741) Load value

v_accvgpr_read v14, a4 // (op 737) 
v_accvgpr_read v15, a5 // (op 737) 
v_accvgpr_read v16, a6 // (op 737) 
v_accvgpr_read v17, a7 // (op 737) 
v_mul_f32 v18, v13, v14 // (op 737) 
v_mul_f32 v19, v13, v15 // (op 737) 
v_mul_f32 v20, v13, v16 // (op 737) 
v_mul_f32 v21, v13, v17 // (op 737) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v4, v6 // (op 742) 
v_lshrrev_b32 v5, 16, v6 // (op 742) 
v_cvt_f32_f16 v5, v5 // (op 742) 
v_mul_f32 v14, v12, v4 // (op 742) 
v_mul_f32 v15, v12, v5 // (op 742) 
v_cvt_f32_f16 v4, v7 // (op 742) 
v_lshrrev_b32 v5, 16, v7 // (op 742) 
v_cvt_f32_f16 v5, v5 // (op 742) 
v_mul_f32 v16, v12, v4 // (op 742) 
v_mul_f32 v17, v12, v5 // (op 742) 


v_add_f32 v4, v14, v18 // (op 738) 
v_add_f32 v5, v15, v19 // (op 738) 
v_add_f32 v6, v16, v20 // (op 738) 
v_add_f32 v7, v17, v21 // (op 738) 


v_cvt_f16_f32 v14, v4 // (op 739) 
v_cvt_f16_f32 v15, v5 // (op 739) 
v_cvt_f16_f32 v16, v6 // (op 739) 
v_cvt_f16_f32 v17, v7 // (op 739) 


v_pack_b32_f16 v6, v14, v15 // (op 740) 
v_pack_b32_f16 v7, v16, v17 // (op 740) 

s_lshl_b32 s0, s4, 6 // (op 740) 
v_add_u32 v0, v11, s0 // (op 740) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 740) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_157:










s_endpgm  // End of RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.LRR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_end:
.size RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2, .LRR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_end-RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
  .amdhsa_next_free_vgpr 164
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 38144 // lds bytes
  .amdhsa_accum_offset 156
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
    DataType: F4
    DataTypeA: F4
    DataTypeB: F8
    DestDataType: h
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
  - .name: RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
    .symbol: RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 38144
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 156
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

