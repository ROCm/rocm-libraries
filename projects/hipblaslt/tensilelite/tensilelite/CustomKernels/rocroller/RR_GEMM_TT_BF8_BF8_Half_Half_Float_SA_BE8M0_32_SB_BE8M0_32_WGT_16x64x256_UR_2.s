// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.p2align 8
.type RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2,@function
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 1575) 
v_and_b32 v3, 15, v1 // (op 1575) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1575) 


v_and_b32 v0, 48, v1 // (op 1575) 
v_lshl_add_u32 v3, v4, 8, v0 // (op 1575) 


v_lshrrev_b32 v0, 7, v1 // (op 1685) 
v_and_b32 v4, 15, v1 // (op 1685) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1685) 


v_and_b32 v0, 63, v1 // (op 1685) 
v_lshrrev_b32 v4, 4, v0 // (op 1685) 

v_lshl_add_u32 v0, v5, 3, v4 // (op 1685) 


v_lshrrev_b32 v4, 7, v1 // (op 1800) 
v_and_b32 v5, 15, v1 // (op 1800) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 1800) 


v_and_b32 v4, 48, v1 // (op 1800) 
v_lshl_add_u32 v5, v6, 8, v4 // (op 1800) 


v_lshrrev_b32 v4, 7, v1 // (op 1868) 
v_and_b32 v6, 15, v1 // (op 1868) 
v_lshl_add_u32 v7, v4, 4, v6 // (op 1868) 


v_and_b32 v4, 63, v1 // (op 1868) 
v_lshrrev_b32 v6, 4, v4 // (op 1868) 

v_lshl_add_u32 v4, v7, 3, v6 // (op 1868) 


v_lshrrev_b32 v6, 7, v1 // (op 1519) 
v_and_b32 v7, 63, v1 // (op 1519) 
v_lshrrev_b32 v8, 4, v7 // (op 1519) 
v_lshlrev_b32 v9, 2, v8 // (op 1519) 

v_lshl_add_u32 v8, v6, 4, v9 // (op 1519) 


v_lshl_add_u32 v6, s16, 4, v8 // (op 1519) 

v_lshrrev_b32 v8, 2, v7 // (op 1519) 

v_and_b32 v7, 3, v8 // (op 1519) 

v_and_b32 v8, 3, v1 // (op 1519) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 1519) 


v_lshrrev_b32 v7, 6, v1 // (op 1519) 
v_and_b32 v8, 1, v7 // (op 1519) 

v_lshl_add_u32 v7, v8, 4, v9 // (op 1519) 


v_lshl_add_u32 v8, s17, 6, v7 // (op 1519) 

v_mul_lo_u32 v7, v8, s2 // (op 1519) 

v_add_lshl_u32 v8, v6, v7, 1 // (op 1519) 


v_and_b32 v6, 48, v1 // (op 1632) 
v_and_b32 v7, 63, v1 // (op 1632) 
v_sub_u32 v9, v7, v6 // (op 1632) 

v_lshrrev_b32 v7, 1, v9 // (op 1632) 
s_mul_i32 s1, s2, s9 // (op 1521) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1521) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1521) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1521) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1521) most significant: sum
s_add_u32 s1, s1, s19 // (op 1521) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1521) 

v_add_lshl_u32 v10, v6, v7, 6 // (op 1632) 


v_and_b32 v6, 1, v9 // (op 1632) 

v_lshlrev_b32 v7, 3, v6 // (op 1632) 

v_lshrrev_b32 v6, 6, v1 // (op 1632) 
v_and_b32 v9, 1, v6 // (op 1632) 

v_lshl_add_u32 v6, v9, 4, v7 // (op 1632) 


v_add_u32 v7, v10, v6 // (op 1632) 


v_lshrrev_b32 v6, 6, v1 // (op 1743) 
v_and_b32 v9, 1, v6 // (op 1743) 

v_and_b32 v6, 15, v1 // (op 1743) 
v_lshl_add_u32 v10, v9, 4, v6 // (op 1743) 


v_and_b32 v6, 63, v1 // (op 1743) 
v_lshrrev_b32 v9, 4, v6 // (op 1743) 

v_lshl_add_u32 v6, v9, 6, v10 // (op 1743) 


v_lshrrev_b32 v9, 7, v1 // (op 1759) 
v_and_b32 v10, 63, v1 // (op 1759) 
v_lshrrev_b32 v11, 4, v10 // (op 1759) 
v_lshlrev_b32 v12, 2, v11 // (op 1759) 

v_lshl_add_u32 v11, v9, 4, v12 // (op 1759) 


v_lshl_add_u32 v9, s16, 4, v11 // (op 1759) 

v_lshrrev_b32 v11, 2, v10 // (op 1759) 

v_and_b32 v10, 3, v11 // (op 1759) 

v_and_b32 v11, 3, v1 // (op 1759) 
v_lshl_add_u32 v12, v10, 2, v11 // (op 1759) 


v_lshrrev_b32 v10, 6, v1 // (op 1759) 
v_and_b32 v11, 1, v10 // (op 1759) 

v_lshl_add_u32 v10, v11, 4, v12 // (op 1759) 


v_lshl_add_u32 v11, s17, 6, v10 // (op 1759) 

v_mul_lo_u32 v10, v11, s4 // (op 1759) 

v_add_lshl_u32 v11, v9, v10, 1 // (op 1759) 


v_and_b32 v9, 48, v1 // (op 1844) 
v_and_b32 v10, 63, v1 // (op 1844) 
v_sub_u32 v12, v10, v9 // (op 1844) 

v_lshrrev_b32 v10, 1, v12 // (op 1844) 
s_mul_i32 s1, s4, s9 // (op 1761) most significant: low of low * high
v_add_lshl_u32 v13, v9, v10, 6 // (op 1844) 


v_and_b32 v9, 1, v12 // (op 1844) 

v_lshlrev_b32 v10, 3, v9 // (op 1844) 

v_lshrrev_b32 v9, 6, v1 // (op 1844) 
v_and_b32 v12, 1, v9 // (op 1844) 

v_lshl_add_u32 v9, v12, 4, v10 // (op 1844) 


v_add_u32 v10, v13, v9 // (op 1844) 


v_lshrrev_b32 v9, 6, v1 // (op 1897) 
v_and_b32 v12, 1, v9 // (op 1897) 

v_and_b32 v9, 15, v1 // (op 1897) 
v_lshl_add_u32 v13, v12, 4, v9 // (op 1897) 


v_and_b32 v9, 63, v1 // (op 1897) 
v_lshrrev_b32 v12, 4, v9 // (op 1897) 

v_lshl_add_u32 v9, v12, 6, v13 // (op 1897) 


s_mul_i32 s40, s5, s8 // (op 1761) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1761) least significant
s_mul_hi_u32 s41, s4, s8 // (op 1761) most significant: high of low * low
s_add_u32 s1, s1, s40 // (op 1761) most significant: sum
s_add_u32 s1, s1, s41 // (op 1761) most significant: sum


s_lshl_b64 s[40:41], s[0:1], 1 // (op 1761) 

s_mov_b32 s44, s6 // (op 1521) reg expression
s_mov_b32 s45, s7 // (op 1521) reg expression
s_mov_b32 s46, s18 // (op 1521) reg expression
s_mov_b32 s47, 131072 // (op 1521) call()

s_mov_b32 s48, s10 // (op 1761) reg expression

s_mov_b32 s49, s11 // (op 1761) reg expression
s_mov_b32 s50, s40 // (op 1761) reg expression
s_mov_b32 s51, 131072 // (op 1761) call()



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
v_lshrrev_b32 v14, 4, v1 // (op 1286) 
v_lshl_add_u32 v15, s16, 4, v14 // (op 1286) 

v_and_b32 v14, -16, v1 // (op 1286) 
v_sub_u32 v16, v1, v14 // (op 1286) 

v_lshlrev_b32 v14, 4, v16 // (op 1286) 

v_mov_b32 v17, 0 // (op 1286) convert
v_mov_b32 v16, v14 // (op 1286) convert

v_mul_lo_u32 v19, v15, s13 // (op 1286) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 1286) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 1286) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1286) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 1286) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1286) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 1288) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1288) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1288) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1288) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1288) most significant: sum
s_add_u32 s1, s1, s7 // (op 1288) most significant: sum


s_mov_b32 s8, s20 // (op 1288) reg expression
s_mov_b32 s9, s21 // (op 1288) reg expression
s_mov_b32 s10, s0 // (op 1288) reg expression
s_mov_b32 s11, 131072 // (op 1288) call()

v_lshrrev_b32 v14, 4, v1 // (op 1305) 
v_lshl_add_u32 v15, s16, 4, v14 // (op 1305) 

v_and_b32 v14, -16, v1 // (op 1305) 
v_sub_u32 v16, v1, v14 // (op 1305) 

v_lshlrev_b32 v14, 4, v16 // (op 1305) 

v_mov_b32 v17, 0 // (op 1305) convert
v_mov_b32 v16, v14 // (op 1305) convert

v_mul_lo_u32 v21, v15, s13 // (op 1305) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 1305) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 1305) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1305) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 1305) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 1305) most significant half



s_mul_i32 s1, s12, s23 // (op 1307) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1307) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1307) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1307) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1307) most significant: sum
s_add_u32 s1, s1, s7 // (op 1307) most significant: sum


s_mov_b32 s8, s20 // (op 1307) reg expression
s_mov_b32 s9, s21 // (op 1307) reg expression
s_mov_b32 s10, s0 // (op 1307) reg expression
s_mov_b32 s11, 131072 // (op 1307) call()

v_and_b32 v14, -4, v1 // (op 1350) 
v_sub_u32 v15, v1, v14 // (op 1350) 

v_lshlrev_b32 v14, 4, v15 // (op 1350) 

v_lshl_add_u32 v15, s17, 6, v14 // (op 1350) 

v_mov_b32 v17, 0 // (op 1350) convert
v_mov_b32 v16, v15 // (op 1350) convert

v_lshrrev_b32 v14, 2, v1 // (op 1350) 
v_mul_lo_u32 v23, v14, s25 // (op 1350) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 1350) most significant: high of low * low
v_mul_lo_u32 v22, v14, s24 // (op 1350) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 1350) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v16 // (op 1350) least significant half
v_addc_co_u32 v23, s[0:1], v23, v17, s[0:1] // (op 1350) most significant half



s_mul_i32 s1, s24, s29 // (op 1352) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 1352) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 1352) least significant
s_mul_hi_u32 s7, s24, s28 // (op 1352) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1352) most significant: sum
s_add_u32 s1, s1, s7 // (op 1352) most significant: sum


s_mov_b32 s40, s26 // (op 1352) reg expression
s_mov_b32 s41, s27 // (op 1352) reg expression
s_mov_b32 s42, s0 // (op 1352) reg expression
s_mov_b32 s43, 131072 // (op 1352) call()

v_and_b32 v14, -4, v1 // (op 1369) 
v_sub_u32 v15, v1, v14 // (op 1369) 

v_lshlrev_b32 v14, 4, v15 // (op 1369) 

v_lshl_add_u32 v15, s17, 6, v14 // (op 1369) 

v_mov_b32 v17, 0 // (op 1369) convert
v_mov_b32 v16, v15 // (op 1369) convert

v_lshrrev_b32 v14, 2, v1 // (op 1369) 
v_mul_lo_u32 v25, v14, s25 // (op 1369) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 1369) most significant: high of low * low
v_mul_lo_u32 v24, v14, s24 // (op 1369) least significant: low of low * low
v_add_u32 v25, v25, v15 // (op 1369) most significant: sum

v_add_co_u32 v24, s[0:1], v24, v16 // (op 1369) least significant half
v_addc_co_u32 v25, s[0:1], v25, v17, s[0:1] // (op 1369) most significant half



s_mul_i32 s1, s24, s29 // (op 1371) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 1371) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 1371) least significant
s_mul_hi_u32 s7, s24, s28 // (op 1371) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1371) most significant: sum
s_add_u32 s1, s1, s7 // (op 1371) most significant: sum


s_mov_b32 s40, s26 // (op 1371) reg expression
s_mov_b32 s41, s27 // (op 1371) reg expression
s_mov_b32 s42, s0 // (op 1371) reg expression
s_mov_b32 s43, 131072 // (op 1371) call()

v_lshrrev_b32 v14, 3, v1 // (op 1414) 
v_lshl_add_u32 v15, s16, 4, v14 // (op 1414) 

v_and_b32 v14, -8, v1 // (op 1414) 
v_sub_u32 v16, v1, v14 // (op 1414) 

v_mul_lo_u32 v14, v15, s30 // (op 1414) 
v_add_u32 v14, v14, v16 // (op 1414) 


s_mul_i32 s1, s30, s23 // (op 1416) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1416) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1416) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1416) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1416) most significant: sum
s_add_u32 s1, s1, s7 // (op 1416) most significant: sum


s_mov_b32 s52, s32 // (op 1416) reg expression
s_mov_b32 s53, s33 // (op 1416) reg expression
s_mov_b32 s54, s0 // (op 1416) reg expression
s_mov_b32 s55, 131072 // (op 1416) call()

v_lshrrev_b32 v15, 3, v1 // (op 1433) 
v_lshl_add_u32 v16, s16, 4, v15 // (op 1433) 

v_and_b32 v15, -8, v1 // (op 1433) 
v_sub_u32 v17, v1, v15 // (op 1433) 

v_mul_lo_u32 v15, v16, s30 // (op 1433) 
v_add_u32 v15, v15, v17 // (op 1433) 


s_mul_i32 s1, s30, s23 // (op 1435) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1435) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1435) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1435) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1435) most significant: sum
s_add_u32 s1, s1, s7 // (op 1435) most significant: sum


s_mov_b32 s52, s32 // (op 1435) reg expression
s_mov_b32 s53, s33 // (op 1435) reg expression
s_mov_b32 s54, s0 // (op 1435) reg expression
s_mov_b32 s55, 131072 // (op 1435) call()

v_and_b32 v16, -16, v1 // (op 1478) 
v_sub_u32 v17, v1, v16 // (op 1478) 

v_lshlrev_b32 v16, 2, v17 // (op 1478) 

v_lshl_add_u32 v17, s17, 6, v16 // (op 1478) 

v_lshrrev_b32 v16, 4, v1 // (op 1478) 
v_mul_lo_u32 v26, v16, s34 // (op 1478) 
v_add_u32 v26, v26, v17 // (op 1478) 


s_mul_i32 s1, s34, s39 // (op 1480) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1480) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1480) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1480) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1480) most significant: sum
s_add_u32 s1, s1, s7 // (op 1480) most significant: sum


s_mov_b32 s56, s36 // (op 1480) reg expression
s_mov_b32 s57, s37 // (op 1480) reg expression
s_mov_b32 s58, s0 // (op 1480) reg expression
s_mov_b32 s59, 131072 // (op 1480) call()

v_and_b32 v16, -16, v1 // (op 1497) 
v_sub_u32 v17, v1, v16 // (op 1497) 

v_lshlrev_b32 v16, 2, v17 // (op 1497) 

v_lshl_add_u32 v17, s17, 6, v16 // (op 1497) 

v_lshrrev_b32 v16, 4, v1 // (op 1497) 
v_mul_lo_u32 v27, v16, s34 // (op 1497) 
v_add_u32 v27, v27, v17 // (op 1497) 


s_mul_i32 s1, s34, s39 // (op 1499) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1499) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1499) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1499) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1499) most significant: sum
s_add_u32 s1, s1, s7 // (op 1499) most significant: sum


s_mov_b32 s56, s36 // (op 1499) reg expression
s_mov_b32 s57, s37 // (op 1499) reg expression
s_mov_b32 s58, s0 // (op 1499) reg expression
s_mov_b32 s59, 131072 // (op 1499) call()

v_and_b32 v16, -4, v1 // (op 1615) 
v_sub_u32 v17, v1, v16 // (op 1615) 

v_lshlrev_b32 v16, 4, v17 // (op 1615) 

v_lshrrev_b32 v17, 2, v1 // (op 1615) 
v_lshl_add_u32 v28, v17, 6, v16 // (op 1615) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1615) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 1234) 
v_mov_b64 v[16:17], s[6:7] // (op 1234) convert


v_mov_b64 v[28:29], v[22:23] // (op 1234) reg expression
s_mov_b32 m0, s0 // (op 1234) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1234) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v16 // (op 1234) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1234) most significant half

s_add_u32 m0, m0, 2048 // (op 1234) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1234) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v16 // (op 1234) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1234) most significant half

s_add_u32 m0, m0, 2048 // (op 1234) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1234) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v16 // (op 1234) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1234) most significant half

s_add_u32 m0, m0, 2048 // (op 1234) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1234) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v16 // (op 1234) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1234) most significant half

s_add_u32 m0, m0, 2048 // (op 1234) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1234) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v16 // (op 1234) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1234) most significant half

s_add_u32 m0, m0, 2048 // (op 1234) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1234) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v16 // (op 1234) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1234) most significant half

s_add_u32 m0, m0, 2048 // (op 1234) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1234) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v16 // (op 1234) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1234) most significant half

s_add_u32 m0, m0, 2048 // (op 1234) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1234) Load value direct to lds





v_mov_b32 v16, v26 // (op 1040) reg expression

buffer_load_dword v17, v16, s[56:59], 0 offen // (op 1040) Load value

v_and_b32 v16, -16, v1 // (op 1563) 
v_sub_u32 v28, v1, v16 // (op 1563) 

v_lshlrev_b32 v16, 4, v28 // (op 1563) 

v_lshrrev_b32 v28, 4, v1 // (op 1563) 
v_lshl_add_u32 v29, v28, 8, v16 // (op 1563) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 1563) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 3 // (op 1239) 
v_mov_b64 v[28:29], s[6:7] // (op 1239) convert


v_mov_b64 v[30:31], v[18:19] // (op 1239) 
s_add_u32 s1, 16384, s0 // (op 1239) 
s_mov_b32 m0, s1 // (op 1239) reg expression
buffer_load_dwordx4 v30, s[8:11], 0 offen offset: 0 lds // (op 1239) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 1239) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 1239) most significant half

s_add_u32 m0, m0, 2048 // (op 1239) 
buffer_load_dwordx4 v30, s[8:11], 0 offen offset: 0 lds // (op 1239) Load value direct to lds







buffer_load_ubyte v16, v14, s[52:55], 0 offen offset: 0 // (op 1046) Load value
v_and_b32 v28, -4, v1 // (op 1827) 
v_sub_u32 v29, v1, v28 // (op 1827) 

v_lshlrev_b32 v28, 4, v29 // (op 1827) 

v_lshrrev_b32 v29, 2, v1 // (op 1827) 
v_lshl_add_u32 v30, v29, 6, v28 // (op 1827) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 1827) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 1244) 
v_mov_b64 v[28:29], s[6:7] // (op 1244) convert


s_lshl_b64 s[6:7], s[24:25], 8 // (op 1244) 
v_mov_b32 v30, s6 // (op 1244) 
v_mov_b32 v31, s7 // (op 1244) 
v_add_co_u32 v32, s[14:15], v22, v30 // (op 1244) least significant half
v_addc_co_u32 v33, s[14:15], v23, v31, s[14:15] // (op 1244) most significant half




s_add_u32 s1, 20480, s0 // (op 1244) 
s_mov_b32 m0, s1 // (op 1244) reg expression
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1244) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 1244) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1244) most significant half

s_add_u32 m0, m0, 2048 // (op 1244) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1244) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 1244) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1244) most significant half

s_add_u32 m0, m0, 2048 // (op 1244) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1244) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 1244) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1244) most significant half

s_add_u32 m0, m0, 2048 // (op 1244) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1244) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 1244) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1244) most significant half

s_add_u32 m0, m0, 2048 // (op 1244) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1244) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 1244) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1244) most significant half

s_add_u32 m0, m0, 2048 // (op 1244) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1244) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 1244) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1244) most significant half

s_add_u32 m0, m0, 2048 // (op 1244) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1244) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 1244) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1244) most significant half

s_add_u32 m0, m0, 2048 // (op 1244) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 1244) Load value direct to lds







s_lshl_b32 s0, s34, 3 // (op 1052) 
v_add_u32 v22, v26, s0 // (op 1052) 


buffer_load_dword v23, v22, s[56:59], 0 offen // (op 1052) Load value


v_and_b32 v22, -16, v1 // (op 1788) 
v_sub_u32 v26, v1, v22 // (op 1788) 

v_lshlrev_b32 v22, 4, v26 // (op 1788) 

v_lshrrev_b32 v26, 4, v1 // (op 1788) 
v_lshl_add_u32 v28, v26, 8, v22 // (op 1788) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1788) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 3 // (op 1246) 
v_mov_b64 v[28:29], s[6:7] // (op 1246) convert


v_mov_b64 v[30:31], v[18:19] // (op 1246) 
s_add_u32 s1, 36864, s0 // (op 1246) 
s_mov_b32 m0, s1 // (op 1246) reg expression
s_mov_b32 s6, 256 // (op 1246) call()
buffer_load_dwordx4 v30, s[8:11], s6 offen offset: 0 lds // (op 1246) Load value direct to lds
v_add_co_u32 v30, s[14:15], v30, v28 // (op 1246) least significant half
v_addc_co_u32 v31, s[14:15], 0, v31, s[14:15] // (op 1246) most significant half

s_add_u32 m0, m0, 2048 // (op 1246) 
buffer_load_dwordx4 v30, s[8:11], s6 offen offset: 0 lds // (op 1246) Load value direct to lds









buffer_load_ubyte v22, v14, s[52:55], 0 offen offset:8 // (op 1058) Load value

v_and_b32 v14, -16, v1 // (op 1726) 
v_sub_u32 v26, v1, v14 // (op 1726) 

v_lshlrev_b32 v14, 2, v26 // (op 1726) 

v_lshrrev_b32 v26, 4, v1 // (op 1726) 
v_lshl_add_u32 v18, v26, 6, v14 // (op 1726) 





s_waitcnt vmcnt(15)
ds_write_b32 v18, v17 offset:40960 // (op 1061) Store local data 

v_and_b32 v14, -8, v1 // (op 1673) 
v_sub_u32 v26, v1, v14 // (op 1673) 

v_and_b32 v14, 4294967288, v1 // (op 1673) 
v_add_u32 v18, v14, v26 // (op 1673) 





s_waitcnt vmcnt(12)
ds_write_b8 v18, v16 offset:41472 // (op 1063) Store local data 

s_barrier  // (op 1987) 



ds_read_b64_tr_b8 v[28:29], v7 // (op 1084) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v7 offset:512 // (op 1084) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v7 offset:4096 // (op 1084) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v7 offset:4608 // (op 1084) Transpose load local data 



ds_read_b64_tr_b8 v[36:37], v7 offset:8192 // (op 1094) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v7 offset:8704 // (op 1094) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v7 offset:12288 // (op 1094) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:12800 // (op 1094) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2022) 


ds_read_u8 v14, v6 offset:40960 // (op 1110) Load local data 


ds_read_u8 v26, v6 offset:41216 // (op 1120) Load local data 



ds_read_b64_tr_b8 v[44:45], v7 offset:32 // (op 1089) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:544 // (op 1089) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:4128 // (op 1089) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v7 offset:4640 // (op 1089) Transpose load local data 



ds_read_b64_tr_b8 v[52:53], v7 offset:8224 // (op 1099) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v7 offset:8736 // (op 1099) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:12320 // (op 1099) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v7 offset:12832 // (op 1099) Transpose load local data 


ds_read_u8 v18, v6 offset:40992 // (op 1115) Load local data 


ds_read_u8 v19, v6 offset:41248 // (op 1125) Load local data 
s_barrier  // (op 1971) 



ds_read_b128 v[60:63], v3 offset:16384 // (op 1077) Load local data 
ds_read_b128 v[64:67], v3 offset:16448 // (op 1077) Load local data 



ds_read_b128 v[68:71], v3 offset:16512 // (op 1080) Load local data 
ds_read_b128 v[72:75], v3 offset:16576 // (op 1080) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 2004) 



ds_read_u8 v76, v0 offset:41472 // (op 1103) Load local data 



ds_read_u8 v77, v0 offset:41476 // (op 1106) Load local data 
v_and_b32 v78, -16, v1 // (op 1535) 
v_sub_u32 v79, v1, v78 // (op 1535) 

v_lshlrev_b32 v78, 4, v79 // (op 1535) 

v_lshrrev_b32 v79, 4, v1 // (op 1535) 
v_lshl_add_u32 v80, v79, 8, v78 // (op 1535) 


s_nop 0
v_readfirstlane_b32 s0, v80 // (op 1535) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v78, -4, v1 // (op 1587) 
v_sub_u32 v79, v1, v78 // (op 1587) 

v_lshlrev_b32 v78, 4, v79 // (op 1587) 

v_lshrrev_b32 v79, 2, v1 // (op 1587) 
v_lshl_add_u32 v80, v79, 6, v78 // (op 1587) 


s_nop 0
v_readfirstlane_b32 s1, v80 // (op 1587) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v78, -8, v1 // (op 1644) 
v_sub_u32 v79, v1, v78 // (op 1644) 

v_and_b32 v78, 4294967288, v1 // (op 1644) 
v_add_u32 v80, v78, v79 // (op 1644) 


v_and_b32 v78, -16, v1 // (op 1697) 
v_sub_u32 v79, v1, v78 // (op 1697) 

v_lshlrev_b32 v78, 2, v79 // (op 1697) 

v_lshrrev_b32 v79, 4, v1 // (op 1697) 
v_lshl_add_u32 v81, v79, 6, v78 // (op 1697) 


v_and_b32 v78, -16, v1 // (op 1773) 
v_sub_u32 v79, v1, v78 // (op 1773) 

v_lshlrev_b32 v78, 4, v79 // (op 1773) 

v_lshrrev_b32 v79, 4, v1 // (op 1773) 
v_lshl_add_u32 v82, v79, 8, v78 // (op 1773) 


s_nop 0
v_readfirstlane_b32 s6, v82 // (op 1773) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v78, -4, v1 // (op 1812) 
v_sub_u32 v79, v1, v78 // (op 1812) 

v_lshlrev_b32 v78, 4, v79 // (op 1812) 

v_lshrrev_b32 v79, 2, v1 // (op 1812) 
v_lshl_add_u32 v82, v79, 6, v78 // (op 1812) 


s_nop 0
v_readfirstlane_b32 s7, v82 // (op 1812) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v78, -8, v1 // (op 1856) 
v_sub_u32 v79, v1, v78 // (op 1856) 

v_and_b32 v78, 4294967288, v1 // (op 1856) 
v_add_u32 v82, v78, v79 // (op 1856) 


v_and_b32 v78, -16, v1 // (op 1880) 
v_sub_u32 v79, v1, v78 // (op 1880) 

v_lshlrev_b32 v78, 2, v79 // (op 1880) 

v_lshrrev_b32 v79, 4, v1 // (op 1880) 
v_lshl_add_u32 v83, v79, 6, v78 // (op 1880) 


s_mov_b32 s14, 0 // (op 404) call()
s_ashr_i64 s[18:19], s[28:29], 8 // (op 403) 
s_mov_b32 s15, s18 // (op 403) convert

s_and_b32 s18, s15, 4294967294 // (op 403) 

s_cmp_lt_u32 s14, s18 // (op 403) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_403 // (op 403) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_403 if false)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_403:
s_add_u32 s14, s14, 2 // (op 405) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:67], v[28:35], a[0:3], v76, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 90) 
s_barrier  // (op 1990) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[60:67], v[44:51], a[4:7], v76, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 451) 
s_lshl_b64 s[18:19], s[24:25], 5 // (op 1206) 
v_mov_b64 v[78:79], s[18:19] // (op 1206) convert


v_mov_b64 v[84:85], v[24:25] // (op 1206) 
s_lshl_b64 s[18:19], s[24:25], 8 // (op 1206) 
s_lshl_b64 s[60:61], s[18:19], 1 // (op 1206) 

v_mov_b32 v86, s60 // (op 1206) 
v_mov_b32 v87, s61 // (op 1206) 
v_add_co_u32 v88, s[18:19], v84, v86 // (op 1206) least significant half
v_addc_co_u32 v89, s[18:19], v85, v87, s[18:19] // (op 1206) most significant half




s_mov_b32 m0, s1 // (op 1206) reg expression
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1206) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v78 // (op 1206) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 1206) most significant half

s_add_u32 m0, m0, 2048 // (op 1206) 
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1206) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v78 // (op 1206) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 1206) most significant half

s_add_u32 m0, m0, 2048 // (op 1206) 
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1206) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v78 // (op 1206) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 1206) most significant half

s_add_u32 m0, m0, 2048 // (op 1206) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:75], v[36:43], a[0:3], v77, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 107) 
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1206) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:75], v[52:59], a[4:7], v77, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 460) 
v_add_co_u32 v88, s[18:19], v88, v78 // (op 1206) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 1206) most significant half

s_add_u32 m0, m0, 2048 // (op 1206) 
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1206) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v78 // (op 1206) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 1206) most significant half

s_add_u32 m0, m0, 2048 // (op 1206) 
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1206) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v78 // (op 1206) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 1206) most significant half

s_add_u32 m0, m0, 2048 // (op 1206) 
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1206) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v78 // (op 1206) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 1206) most significant half

s_add_u32 m0, m0, 2048 // (op 1206) 
buffer_load_dwordx4 v88, s[40:43], 0 offen offset: 0 lds // (op 1206) Load value direct to lds




v_mov_b32 v78, v27 // (op 25) 
s_lshl_b32 s15, s34, 4 // (op 25) 
v_add_u32 v79, v78, s15 // (op 25) 


buffer_load_dword v17, v79, s[56:59], 0 offen // (op 25) Load value

s_barrier  // (op 1974) 

s_lshl_b64 s[18:19], s[12:13], 3 // (op 1201) 
v_mov_b64 v[86:87], s[18:19] // (op 1201) convert


v_mov_b64 v[88:89], v[20:21] // (op 1201) 
v_mov_b64 v[90:91], v[88:89] // (op 1201) 
s_add_u32 s15, 16384, s0 // (op 1201) 
s_mov_b32 m0, s15 // (op 1201) reg expression
s_mov_b32 s18, 512 // (op 1201) call()
buffer_load_dwordx4 v90, s[8:11], s18 offen offset: 0 lds // (op 1201) Load value direct to lds
v_add_co_u32 v90, s[60:61], v90, v86 // (op 1201) least significant half
v_addc_co_u32 v91, s[60:61], 0, v91, s[60:61] // (op 1201) most significant half

s_add_u32 m0, m0, 2048 // (op 1201) 
buffer_load_dwordx4 v90, s[8:11], s18 offen offset: 0 lds // (op 1201) Load value direct to lds






v_mov_b32 v79, v15 // (op 17) 

buffer_load_ubyte v16, v79, s[52:55], 0 offen offset:16 // (op 17) Load value
s_barrier  // (op 2062) 



s_waitcnt vmcnt(15)
ds_write_b32 v83, v23 offset:41600 // (op 308) Store local data 
s_barrier  // (op 2056) 



s_waitcnt vmcnt(12)
ds_write_b8 v82, v22 offset:42112 // (op 307) Store local data 
s_barrier  // (op 2045) 



ds_read_b64_tr_b8 v[28:29], v10 offset:20480 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v10 offset:20992 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v10 offset:24576 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v10 offset:25088 // (op 291) Transpose load local data 



ds_read_b64_tr_b8 v[90:91], v10 offset:28672 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v10 offset:29184 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v10 offset:32768 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v10 offset:33280 // (op 300) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2059) 


ds_read_u8 v14, v9 offset:41600 // (op 295) Load local data 


ds_read_u8 v86, v9 offset:41856 // (op 304) Load local data 



ds_read_b64_tr_b8 v[44:45], v10 offset:20512 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v10 offset:21024 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v10 offset:24608 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v10 offset:25120 // (op 474) Transpose load local data 



ds_read_b64_tr_b8 v[98:99], v10 offset:28704 // (op 465) Transpose load local data 
ds_read_b64_tr_b8 v[100:101], v10 offset:29216 // (op 465) Transpose load local data 
ds_read_b64_tr_b8 v[102:103], v10 offset:32800 // (op 465) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v10 offset:33312 // (op 465) Transpose load local data 


ds_read_u8 v18, v9 offset:41632 // (op 478) Load local data 


ds_read_u8 v87, v9 offset:41888 // (op 469) Load local data 
s_barrier  // (op 2039) 
s_barrier  // (op 2133) 



ds_read_b128 v[60:63], v5 offset:36864 // (op 289) Load local data 
ds_read_b128 v[64:67], v5 offset:36928 // (op 289) Load local data 



ds_read_b128 v[106:109], v5 offset:36992 // (op 298) Load local data 
ds_read_b128 v[110:113], v5 offset:37056 // (op 298) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 2052) 



ds_read_u8 v76, v4 offset:42112 // (op 293) Load local data 



ds_read_u8 v77, v4 offset:42116 // (op 302) Load local data 
s_barrier  // (op 2097) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:67], v[28:35], a[0:3], v76, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 296) 
s_barrier  // (op 2049) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[60:67], v[44:51], a[4:7], v76, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 461) 
s_lshl_b64 s[18:19], s[24:25], 5 // (op 1229) 
v_mov_b64 v[114:115], s[18:19] // (op 1229) convert


s_lshl_b64 s[18:19], s[24:25], 8 // (op 1229) 
s_mul_i32 s61, 3, s19 // (op 1229) most significant: low of low * high
s_mul_i32 s60, 3, s18 // (op 1229) least significant
s_mul_hi_u32 s15, 3, s18 // (op 1229) most significant: high of low * low
s_add_u32 s61, s61, s15 // (op 1229) most significant: sum


v_mov_b32 v116, s60 // (op 1229) 
v_mov_b32 v117, s61 // (op 1229) 
v_add_co_u32 v118, s[18:19], v84, v116 // (op 1229) least significant half
v_addc_co_u32 v119, s[18:19], v85, v117, s[18:19] // (op 1229) most significant half




s_add_u32 s15, 20480, s7 // (op 1229) 
s_mov_b32 m0, s15 // (op 1229) reg expression
buffer_load_dwordx4 v118, s[40:43], 0 offen offset: 0 lds // (op 1229) Load value direct to lds
v_add_co_u32 v118, s[18:19], v118, v114 // (op 1229) least significant half
v_addc_co_u32 v119, s[18:19], 0, v119, s[18:19] // (op 1229) most significant half

s_add_u32 m0, m0, 2048 // (op 1229) 
buffer_load_dwordx4 v118, s[40:43], 0 offen offset: 0 lds // (op 1229) Load value direct to lds
v_add_co_u32 v118, s[18:19], v118, v114 // (op 1229) least significant half
v_addc_co_u32 v119, s[18:19], 0, v119, s[18:19] // (op 1229) most significant half

s_add_u32 m0, m0, 2048 // (op 1229) 
buffer_load_dwordx4 v118, s[40:43], 0 offen offset: 0 lds // (op 1229) Load value direct to lds
v_add_co_u32 v118, s[18:19], v118, v114 // (op 1229) least significant half
v_addc_co_u32 v119, s[18:19], 0, v119, s[18:19] // (op 1229) most significant half

s_add_u32 m0, m0, 2048 // (op 1229) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[106:113], v[90:97], a[0:3], v77, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 305) 
buffer_load_dwordx4 v118, s[40:43], 0 offen offset: 0 lds // (op 1229) Load value direct to lds


v_add_co_u32 v118, s[18:19], v118, v114 // (op 1229) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[106:113], v[98:105], a[4:7], v77, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 470) 
v_addc_co_u32 v119, s[18:19], 0, v119, s[18:19] // (op 1229) most significant half




s_add_u32 m0, m0, 2048 // (op 1229) 
buffer_load_dwordx4 v118, s[40:43], 0 offen offset: 0 lds // (op 1229) Load value direct to lds
v_add_co_u32 v118, s[18:19], v118, v114 // (op 1229) least significant half
v_addc_co_u32 v119, s[18:19], 0, v119, s[18:19] // (op 1229) most significant half

s_add_u32 m0, m0, 2048 // (op 1229) 
buffer_load_dwordx4 v118, s[40:43], 0 offen offset: 0 lds // (op 1229) Load value direct to lds
v_add_co_u32 v118, s[18:19], v118, v114 // (op 1229) least significant half
v_addc_co_u32 v119, s[18:19], 0, v119, s[18:19] // (op 1229) most significant half

s_add_u32 m0, m0, 2048 // (op 1229) 
buffer_load_dwordx4 v118, s[40:43], 0 offen offset: 0 lds // (op 1229) Load value direct to lds
v_add_co_u32 v118, s[18:19], v118, v114 // (op 1229) least significant half
v_addc_co_u32 v119, s[18:19], 0, v119, s[18:19] // (op 1229) most significant half

s_add_u32 m0, m0, 2048 // (op 1229) 
buffer_load_dwordx4 v118, s[40:43], 0 offen offset: 0 lds // (op 1229) Load value direct to lds






s_lshl_b32 s15, s34, 3 // (op 311) 
s_mul_i32 s18, 3, s15 // (op 311) 

v_add_u32 v84, v78, s18 // (op 311) 


buffer_load_dword v23, v84, s[56:59], 0 offen // (op 311) Load value

s_lshl_b64 s[18:19], s[24:25], 8 // (op 1361) 

s_lshl_b64 s[60:61], s[18:19], 1 // (op 1361) 

s_barrier  // (op 2042) 
s_lshl_b32 s15, s34, 4 // (op 1489) 

v_add_u32 v27, v27, s15 // (op 1489) 

v_mov_b32 v78, s60 // (op 1361) 
v_mov_b32 v84, s61 // (op 1361) 
s_lshl_b64 s[18:19], s[12:13], 3 // (op 1224) 
v_mov_b64 v[86:87], s[18:19] // (op 1224) convert


v_mov_b64 v[90:91], v[88:89] // (op 1224) 
s_add_u32 s15, 36864, s6 // (op 1224) 
s_mov_b32 m0, s15 // (op 1224) reg expression
s_mov_b32 s18, 768 // (op 1224) call()
v_add_co_u32 v24, s[62:63], v24, v78 // (op 1361) least significant half
v_addc_co_u32 v25, s[62:63], v25, v84, s[62:63] // (op 1361) most significant half




buffer_load_dwordx4 v90, s[8:11], s18 offen offset: 0 lds // (op 1224) Load value direct to lds
v_add_co_u32 v90, s[60:61], v90, v86 // (op 1224) least significant half
v_addc_co_u32 v91, s[60:61], 0, v91, s[60:61] // (op 1224) most significant half

s_add_u32 m0, m0, 2048 // (op 1224) 
buffer_load_dwordx4 v90, s[8:11], s18 offen offset: 0 lds // (op 1224) Load value direct to lds








v_add_co_u32 v20, vcc, 512, v20 // (op 1297) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 1297) most significant half
buffer_load_ubyte v22, v79, s[52:55], 0 offen offset:24 // (op 310) Load value

s_barrier  // (op 2025) 
v_add_u32 v15, 16, v15 // (op 1425) 



s_waitcnt vmcnt(15)
ds_write_b32 v81, v17 offset:40960 // (op 30) Store local data 
s_barrier  // (op 2007) 



s_waitcnt vmcnt(12)
ds_write_b8 v80, v16 offset:41472 // (op 21) Store local data 
s_barrier  // (op 1996) 



ds_read_b64_tr_b8 v[28:29], v7 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v7 offset:512 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v7 offset:4096 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v7 offset:4608 // (op 77) Transpose load local data 



ds_read_b64_tr_b8 v[36:37], v7 offset:8192 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v7 offset:8704 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v7 offset:12288 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:12800 // (op 97) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2032) 


ds_read_u8 v14, v6 offset:40960 // (op 87) Load local data 


ds_read_u8 v26, v6 offset:41216 // (op 105) Load local data 



ds_read_b64_tr_b8 v[44:45], v7 offset:32 // (op 446) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:544 // (op 446) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:4128 // (op 446) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v7 offset:4640 // (op 446) Transpose load local data 



ds_read_b64_tr_b8 v[52:53], v7 offset:8224 // (op 455) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v7 offset:8736 // (op 455) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:12320 // (op 455) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v7 offset:12832 // (op 455) Transpose load local data 


ds_read_u8 v18, v6 offset:40992 // (op 450) Load local data 


ds_read_u8 v19, v6 offset:41248 // (op 459) Load local data 
s_barrier  // (op 1980) 



ds_read_b128 v[60:63], v3 offset:16384 // (op 72) Load local data 
ds_read_b128 v[64:67], v3 offset:16448 // (op 72) Load local data 



ds_read_b128 v[68:71], v3 offset:16512 // (op 93) Load local data 
ds_read_b128 v[72:75], v3 offset:16576 // (op 93) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 2014) 



ds_read_u8 v76, v0 offset:41472 // (op 82) Load local data 



ds_read_u8 v77, v0 offset:41476 // (op 101) Load local data 
s_ashr_i64 s[18:19], s[28:29], 8 // (op 403) 
s_mov_b32 s15, s18 // (op 403) convert

s_and_b32 s18, s15, 4294967294 // (op 403) 

s_cmp_lt_u32 s14, s18 // (op 403) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_403 // (op 403) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_403 if true)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_403:



















s_barrier  // (op 2115) 
v_lshrrev_b32 v14, 4, v1 // (op 1259) 
s_ashr_i64 s[0:1], s[28:29], 8 // (op 1259) 
v_lshl_add_u32 v15, s16, 4, v14 // (op 1259) 

s_lshr_b32 s6, s0, 1 // (op 1259) 

v_and_b32 v14, -16, v1 // (op 1259) 
v_sub_u32 v16, v1, v14 // (op 1259) 

v_lshlrev_b32 v14, 4, v16 // (op 1259) 

v_lshl_add_u32 v16, s6, 9, v14 // (op 1259) 


v_mov_b32 v21, 0 // (op 1259) convert
v_mov_b32 v20, v16 // (op 1259) convert

v_mul_lo_u32 v17, v15, s13 // (op 1259) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 1259) most significant: high of low * low
v_mul_lo_u32 v16, v15, s12 // (op 1259) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 1259) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v20 // (op 1259) least significant half
v_addc_co_u32 v17, s[0:1], v17, v21, s[0:1] // (op 1259) most significant half



s_mul_i32 s1, s12, s23 // (op 1261) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1261) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1261) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1261) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1261) most significant: sum
s_add_u32 s1, s1, s7 // (op 1261) most significant: sum


s_mov_b32 s8, s20 // (op 1261) reg expression
s_mov_b32 s9, s21 // (op 1261) reg expression
s_mov_b32 s10, s0 // (op 1261) reg expression
s_mov_b32 s11, 131072 // (op 1261) call()

v_lshrrev_b32 v14, 2, v1 // (op 1327) 
s_ashr_i64 s[0:1], s[28:29], 8 // (op 1327) 
v_and_b32 v15, -4, v1 // (op 1327) 
s_lshr_b32 s6, s0, 1 // (op 1327) 

v_sub_u32 v20, v1, v15 // (op 1327) 

v_lshl_add_u32 v15, s6, 9, v14 // (op 1327) 


v_lshlrev_b32 v14, 4, v20 // (op 1327) 

v_lshl_add_u32 v20, s17, 6, v14 // (op 1327) 

v_mov_b32 v25, 0 // (op 1327) convert
v_mov_b32 v24, v20 // (op 1327) convert

v_mul_lo_u32 v21, v15, s25 // (op 1327) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1327) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 1327) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1327) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v24 // (op 1327) least significant half
v_addc_co_u32 v21, s[0:1], v21, v25, s[0:1] // (op 1327) most significant half



s_mul_i32 s1, s24, s29 // (op 1329) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 1329) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 1329) least significant
s_mul_hi_u32 s7, s24, s28 // (op 1329) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1329) most significant: sum
s_add_u32 s1, s1, s7 // (op 1329) most significant: sum


s_mov_b32 s40, s26 // (op 1329) reg expression
s_mov_b32 s41, s27 // (op 1329) reg expression
s_mov_b32 s42, s0 // (op 1329) reg expression
s_mov_b32 s43, 131072 // (op 1329) call()

v_lshrrev_b32 v14, 3, v1 // (op 1391) 
s_ashr_i64 s[0:1], s[28:29], 8 // (op 1391) 
v_lshl_add_u32 v15, s16, 4, v14 // (op 1391) 

s_lshr_b32 s6, s0, 1 // (op 1391) 

v_and_b32 v14, -8, v1 // (op 1391) 
v_sub_u32 v24, v1, v14 // (op 1391) 

v_lshl_add_u32 v14, s6, 4, v24 // (op 1391) 


v_mul_lo_u32 v24, v15, s30 // (op 1391) 
v_add_u32 v24, v24, v14 // (op 1391) 


s_mul_i32 s1, s30, s23 // (op 1393) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1393) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1393) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1393) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1393) most significant: sum
s_add_u32 s1, s1, s7 // (op 1393) most significant: sum


s_mov_b32 s52, s32 // (op 1393) reg expression
s_mov_b32 s53, s33 // (op 1393) reg expression
s_mov_b32 s54, s0 // (op 1393) reg expression
s_mov_b32 s55, 131072 // (op 1393) call()

v_lshrrev_b32 v14, 4, v1 // (op 1455) 
s_ashr_i64 s[0:1], s[28:29], 8 // (op 1455) 
v_and_b32 v15, -16, v1 // (op 1455) 
s_lshr_b32 s6, s0, 1 // (op 1455) 

v_sub_u32 v25, v1, v15 // (op 1455) 

v_lshl_add_u32 v15, s6, 4, v14 // (op 1455) 


v_lshlrev_b32 v14, 2, v25 // (op 1455) 

v_lshl_add_u32 v25, s17, 6, v14 // (op 1455) 

v_mul_lo_u32 v14, v15, s34 // (op 1455) 
v_add_u32 v14, v14, v25 // (op 1455) 


s_mul_i32 s1, s34, s39 // (op 1457) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1457) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1457) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1457) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1457) most significant: sum
s_add_u32 s1, s1, s7 // (op 1457) most significant: sum


s_mov_b32 s56, s36 // (op 1457) reg expression
s_mov_b32 s57, s37 // (op 1457) reg expression
s_mov_b32 s58, s0 // (op 1457) reg expression
s_mov_b32 s59, 131072 // (op 1457) call()

v_and_b32 v15, -16, v1 // (op 1548) 
v_sub_u32 v25, v1, v15 // (op 1548) 

v_lshlrev_b32 v15, 4, v25 // (op 1548) 

v_lshrrev_b32 v25, 4, v1 // (op 1548) 
v_lshl_add_u32 v26, v25, 8, v15 // (op 1548) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1548) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -4, v1 // (op 1600) 
v_sub_u32 v25, v1, v15 // (op 1600) 

v_lshlrev_b32 v15, 4, v25 // (op 1600) 

v_lshrrev_b32 v25, 2, v1 // (op 1600) 
v_lshl_add_u32 v26, v25, 6, v15 // (op 1600) 


s_nop 0
v_readfirstlane_b32 s1, v26 // (op 1600) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -8, v1 // (op 1657) 
v_sub_u32 v25, v1, v15 // (op 1657) 

v_and_b32 v15, 4294967288, v1 // (op 1657) 
v_add_u32 v26, v15, v25 // (op 1657) 


v_and_b32 v15, -16, v1 // (op 1710) 
v_sub_u32 v25, v1, v15 // (op 1710) 

v_lshlrev_b32 v15, 2, v25 // (op 1710) 

v_lshrrev_b32 v25, 4, v1 // (op 1710) 
v_lshl_add_u32 v27, v25, 6, v15 // (op 1710) 


s_ashr_i64 s[6:7], s[28:29], 8 // (op 409) 
s_and_b32 s14, s6, 4294967294 // (op 409) 

s_ashr_i64 s[6:7], s[28:29], 8 // (op 408) 
s_mov_b32 s15, s6 // (op 408) convert

s_cmp_lt_u32 s14, s15 // (op 408) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_408 // (op 408) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_408 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_408:
s_barrier  // (op 1977) 
s_add_u32 s14, s14, 1 // (op 410) 

s_lshl_b64 s[6:7], s[12:13], 3 // (op 1211) 
v_mov_b64 v[18:19], s[6:7] // (op 1211) convert


v_mov_b64 v[22:23], v[16:17] // (op 1211) 
v_mov_b64 v[28:29], v[22:23] // (op 1211) 
s_add_u32 s15, 16384, s0 // (op 1211) 
s_mov_b32 m0, s15 // (op 1211) reg expression
buffer_load_dwordx4 v28, s[8:11], 0 offen offset: 0 lds // (op 1211) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1211) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1211) most significant half

s_add_u32 m0, m0, 2048 // (op 1211) 
buffer_load_dwordx4 v28, s[8:11], 0 offen offset: 0 lds // (op 1211) Load value direct to lds




v_add_co_u32 v16, vcc, 256, v16 // (op 1249) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1249) most significant half
s_barrier  // (op 1993) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 1217) 
v_mov_b64 v[18:19], s[6:7] // (op 1217) convert


v_mov_b64 v[22:23], v[20:21] // (op 1217) 
v_mov_b64 v[28:29], v[22:23] // (op 1217) reg expression
s_mov_b32 m0, s1 // (op 1217) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1217) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1217) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1217) most significant half

s_add_u32 m0, m0, 2048 // (op 1217) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1217) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1217) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1217) most significant half

s_add_u32 m0, m0, 2048 // (op 1217) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1217) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1217) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1217) most significant half

s_add_u32 m0, m0, 2048 // (op 1217) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1217) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1217) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1217) most significant half

s_add_u32 m0, m0, 2048 // (op 1217) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1217) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1217) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1217) most significant half

s_add_u32 m0, m0, 2048 // (op 1217) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1217) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1217) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1217) most significant half

s_add_u32 m0, m0, 2048 // (op 1217) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1217) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 1217) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1217) most significant half

s_add_u32 m0, m0, 2048 // (op 1217) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 1217) Load value direct to lds



s_lshl_b64 s[6:7], s[24:25], 8 // (op 1319) 
v_mov_b32 v15, s6 // (op 1319) 
v_mov_b32 v25, s7 // (op 1319) 
v_add_co_u32 v20, s[18:19], v20, v15 // (op 1319) least significant half
v_addc_co_u32 v21, s[18:19], v21, v25, s[18:19] // (op 1319) most significant half






v_mov_b32 v15, v24 // (op 226) 

buffer_load_ubyte v25, v15, s[52:55], 0 offen offset: 0 // (op 226) Load value

v_add_u32 v24, 8, v24 // (op 1383) 


s_barrier  // (op 2010) 
v_mov_b32 v15, v14 // (op 227) 



v_mov_b32 v18, v15 // (op 227) reg expression

buffer_load_dword v19, v18, s[56:59], 0 offen // (op 227) Load value


s_lshl_b32 s15, s34, 3 // (op 1447) 
v_add_u32 v14, v14, s15 // (op 1447) 

s_waitcnt vmcnt(1)
ds_write_b8 v26, v25 offset:41472 // (op 223) Store local data 

s_barrier  // (op 1983) 



s_barrier  // (op 2028) 
ds_read_b128 v[28:31], v3 offset:16384 // (op 205) Load local data 



ds_read_b128 v[32:35], v3 offset:16448 // (op 205) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v27, v19 offset:40960 // (op 224) Store local data 

s_barrier  // (op 2000) 



ds_read_b64_tr_b8 v[36:37], v7 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v7 offset:512 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v7 offset:4096 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:4608 // (op 207) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 2018) 



ds_read_u8 v15, v0 offset:41472 // (op 209) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2035) 


ds_read_u8 v25, v6 offset:40960 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:43], a[0:3], v15, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 212) 





ds_read_b128 v[36:39], v3 offset:16512 // (op 214) Load local data 
ds_read_b128 v[40:43], v3 offset:16576 // (op 214) Load local data 



ds_read_b64_tr_b8 v[44:45], v7 offset:8192 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:8704 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:12288 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v7 offset:12800 // (op 216) Transpose load local data 



ds_read_u8 v25, v0 offset:41476 // (op 218) Load local data 


ds_read_u8 v18, v6 offset:41216 // (op 220) Load local data 



ds_read_b64_tr_b8 v[52:53], v7 offset:32 // (op 416) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v7 offset:544 // (op 416) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:4128 // (op 416) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v7 offset:4640 // (op 416) Transpose load local data 


ds_read_u8 v19, v6 offset:40992 // (op 420) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:35], v[52:59], a[4:7], v15, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 421) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:51], a[0:3], v25, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 221) 
ds_read_b64_tr_b8 v[28:29], v7 offset:8224 // (op 425) Transpose load local data 


ds_read_b64_tr_b8 v[30:31], v7 offset:8736 // (op 425) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v7 offset:12320 // (op 425) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v7 offset:12832 // (op 425) Transpose load local data 


ds_read_u8 v15, v6 offset:41248 // (op 429) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:43], v[28:35], a[4:7], v25, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 430) 




s_ashr_i64 s[6:7], s[28:29], 8 // (op 408) 
s_mov_b32 s15, s6 // (op 408) convert

s_cmp_lt_u32 s14, s15 // (op 408) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_408 // (op 408) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_408 if true)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_408:
























s_barrier  // (op 2119) 








v_cmp_eq_f32 vcc, 0.00000, v12 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_157
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

buffer_store_dwordx2 v[14:15], v0, s[48:51], 0 offen // (op 58) Store value



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


buffer_store_dwordx2 v[14:15], v0, s[48:51], 0 offen // (op 717) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_157
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_157:


v_mov_b32 v0, v8 // (op 170) reg expression

buffer_load_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 170) Load value

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

buffer_store_dwordx2 v[6:7], v0, s[48:51], 0 offen // (op 169) Store value





s_lshl_b32 s0, s2, 6 // (op 741) 
v_add_u32 v0, v8, s0 // (op 741) 


buffer_load_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 741) Load value

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


buffer_store_dwordx2 v[6:7], v0, s[48:51], 0 offen // (op 740) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_157:










s_endpgm  // End of RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.LRR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_end:
.size RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2, .LRR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_end-RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
  .amdhsa_next_free_vgpr 128
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 42240 // lds bytes
  .amdhsa_accum_offset 120
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
    DataTypeB: B8
    DestDataType: h
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
  - .name: RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
    .symbol: RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 42240
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 120
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

