// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
.p2align 8
.type RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2,@function
RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 1252) 
v_and_b32 v3, 15, v1 // (op 1252) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1252) 


v_and_b32 v0, 48, v1 // (op 1252) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 1252) 


v_lshrrev_b32 v0, 7, v1 // (op 1424) 
v_and_b32 v4, 15, v1 // (op 1424) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1424) 


v_and_b32 v0, 48, v1 // (op 1424) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 1424) 


v_lshrrev_b32 v0, 7, v1 // (op 1196) 
v_and_b32 v5, 63, v1 // (op 1196) 
v_lshrrev_b32 v6, 4, v5 // (op 1196) 
v_lshlrev_b32 v7, 2, v6 // (op 1196) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1196) 


v_lshl_add_u32 v0, s16, 5, v6 // (op 1196) 

v_lshrrev_b32 v6, 2, v5 // (op 1196) 

v_and_b32 v5, 3, v6 // (op 1196) 

v_and_b32 v6, 3, v1 // (op 1196) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1196) 


v_lshrrev_b32 v5, 6, v1 // (op 1196) 
v_and_b32 v6, 1, v5 // (op 1196) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1196) 


v_lshl_add_u32 v6, s17, 6, v5 // (op 1196) 

v_mul_lo_u32 v5, v6, s2 // (op 1196) 

v_add_lshl_u32 v6, v0, v5, 1 // (op 1196) 


v_and_b32 v0, 48, v1 // (op 1309) 
v_and_b32 v5, 63, v1 // (op 1309) 
v_sub_u32 v7, v5, v0 // (op 1309) 

v_lshrrev_b32 v5, 1, v7 // (op 1309) 
s_mul_i32 s1, s2, s9 // (op 1198) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1198) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1198) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1198) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1198) most significant: sum
s_add_u32 s1, s1, s19 // (op 1198) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1198) 

v_add_lshl_u32 v8, v0, v5, 6 // (op 1309) 


v_and_b32 v0, 1, v7 // (op 1309) 

v_lshlrev_b32 v5, 3, v0 // (op 1309) 

v_lshrrev_b32 v0, 6, v1 // (op 1309) 
v_and_b32 v7, 1, v0 // (op 1309) 

v_lshl_add_u32 v0, v7, 4, v5 // (op 1309) 


v_add_u32 v5, v8, v0 // (op 1309) 


v_lshrrev_b32 v0, 6, v1 // (op 1367) 
v_and_b32 v7, 1, v0 // (op 1367) 

v_and_b32 v0, 15, v1 // (op 1367) 
v_lshl_add_u32 v8, v7, 4, v0 // (op 1367) 


v_and_b32 v0, 63, v1 // (op 1367) 
v_lshrrev_b32 v7, 4, v0 // (op 1367) 

v_lshl_add_u32 v0, v7, 6, v8 // (op 1367) 


v_lshrrev_b32 v7, 7, v1 // (op 1383) 
v_and_b32 v8, 63, v1 // (op 1383) 
v_lshrrev_b32 v9, 4, v8 // (op 1383) 
v_lshlrev_b32 v10, 2, v9 // (op 1383) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1383) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 1383) 

v_lshrrev_b32 v9, 2, v8 // (op 1383) 

v_and_b32 v8, 3, v9 // (op 1383) 

v_and_b32 v9, 3, v1 // (op 1383) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1383) 


v_lshrrev_b32 v8, 6, v1 // (op 1383) 
v_and_b32 v9, 1, v8 // (op 1383) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 1383) 


v_lshl_add_u32 v9, s17, 6, v8 // (op 1383) 

v_mul_lo_u32 v8, v9, s4 // (op 1383) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 1383) 


v_and_b32 v7, 48, v1 // (op 1468) 
v_and_b32 v8, 63, v1 // (op 1468) 
v_sub_u32 v10, v8, v7 // (op 1468) 

v_lshrrev_b32 v8, 1, v10 // (op 1468) 
s_mul_i32 s1, s4, s9 // (op 1385) most significant: low of low * high
v_add_lshl_u32 v11, v7, v8, 6 // (op 1468) 


v_and_b32 v7, 1, v10 // (op 1468) 

v_lshlrev_b32 v8, 3, v7 // (op 1468) 

v_lshrrev_b32 v7, 6, v1 // (op 1468) 
v_and_b32 v10, 1, v7 // (op 1468) 

v_lshl_add_u32 v7, v10, 4, v8 // (op 1468) 


v_add_u32 v8, v11, v7 // (op 1468) 


v_lshrrev_b32 v7, 6, v1 // (op 1497) 
v_and_b32 v10, 1, v7 // (op 1497) 

v_and_b32 v7, 15, v1 // (op 1497) 
v_lshl_add_u32 v11, v10, 4, v7 // (op 1497) 


v_and_b32 v7, 63, v1 // (op 1497) 
v_lshrrev_b32 v10, 4, v7 // (op 1497) 

v_lshl_add_u32 v7, v10, 6, v11 // (op 1497) 


s_mul_i32 s40, s5, s8 // (op 1385) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1385) least significant
s_mul_hi_u32 s41, s4, s8 // (op 1385) most significant: high of low * low
s_add_u32 s1, s1, s40 // (op 1385) most significant: sum
s_add_u32 s1, s1, s41 // (op 1385) most significant: sum


s_lshl_b64 s[40:41], s[0:1], 1 // (op 1385) 

s_mov_b32 s44, s6 // (op 1198) reg expression
s_mov_b32 s45, s7 // (op 1198) reg expression
s_mov_b32 s46, s18 // (op 1198) reg expression
s_mov_b32 s47, 131072 // (op 1198) call()

s_mov_b32 s48, s10 // (op 1385) reg expression

s_mov_b32 s49, s11 // (op 1385) reg expression
s_mov_b32 s50, s40 // (op 1385) reg expression
s_mov_b32 s51, 131072 // (op 1385) call()



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
v_lshrrev_b32 v12, 3, v1 // (op 963) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 963) 

v_and_b32 v12, -8, v1 // (op 963) 
v_sub_u32 v14, v1, v12 // (op 963) 

v_lshlrev_b32 v12, 4, v14 // (op 963) 

v_mov_b32 v15, 0 // (op 963) convert
v_mov_b32 v14, v12 // (op 963) convert

v_mul_lo_u32 v17, v13, s13 // (op 963) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 963) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 963) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 963) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 963) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 963) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 965) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 965) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 965) least significant
s_mul_hi_u32 s7, s12, s22 // (op 965) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 965) most significant: sum
s_add_u32 s1, s1, s7 // (op 965) most significant: sum


s_mov_b32 s8, s20 // (op 965) reg expression
s_mov_b32 s9, s21 // (op 965) reg expression
s_mov_b32 s10, s0 // (op 965) reg expression
s_mov_b32 s11, 131072 // (op 965) call()

v_lshrrev_b32 v12, 3, v1 // (op 982) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 982) 

v_and_b32 v12, -8, v1 // (op 982) 
v_sub_u32 v14, v1, v12 // (op 982) 

v_lshlrev_b32 v12, 4, v14 // (op 982) 

v_mov_b32 v15, 0 // (op 982) convert
v_mov_b32 v14, v12 // (op 982) convert

v_mul_lo_u32 v19, v13, s13 // (op 982) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 982) most significant: high of low * low
v_mul_lo_u32 v18, v13, s12 // (op 982) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 982) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 982) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 982) most significant half



s_mul_i32 s1, s12, s23 // (op 984) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 984) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 984) least significant
s_mul_hi_u32 s7, s12, s22 // (op 984) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 984) most significant: sum
s_add_u32 s1, s1, s7 // (op 984) most significant: sum


s_mov_b32 s8, s20 // (op 984) reg expression
s_mov_b32 s9, s21 // (op 984) reg expression
s_mov_b32 s10, s0 // (op 984) reg expression
s_mov_b32 s11, 131072 // (op 984) call()

v_and_b32 v12, -4, v1 // (op 1027) 
v_sub_u32 v13, v1, v12 // (op 1027) 

v_lshlrev_b32 v12, 4, v13 // (op 1027) 

v_lshl_add_u32 v13, s17, 6, v12 // (op 1027) 

v_mov_b32 v15, 0 // (op 1027) convert
v_mov_b32 v14, v13 // (op 1027) convert

v_lshrrev_b32 v12, 2, v1 // (op 1027) 
v_mul_lo_u32 v21, v12, s25 // (op 1027) most significant: low of low * high
v_mul_hi_u32 v13, v12, s24 // (op 1027) most significant: high of low * low
v_mul_lo_u32 v20, v12, s24 // (op 1027) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 1027) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v14 // (op 1027) least significant half
v_addc_co_u32 v21, s[0:1], v21, v15, s[0:1] // (op 1027) most significant half



s_mul_i32 s1, s24, s29 // (op 1029) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 1029) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 1029) least significant
s_mul_hi_u32 s7, s24, s28 // (op 1029) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1029) most significant: sum
s_add_u32 s1, s1, s7 // (op 1029) most significant: sum


s_mov_b32 s40, s26 // (op 1029) reg expression
s_mov_b32 s41, s27 // (op 1029) reg expression
s_mov_b32 s42, s0 // (op 1029) reg expression
s_mov_b32 s43, 131072 // (op 1029) call()

v_and_b32 v12, -4, v1 // (op 1046) 
v_sub_u32 v13, v1, v12 // (op 1046) 

v_lshlrev_b32 v12, 4, v13 // (op 1046) 

v_lshl_add_u32 v13, s17, 6, v12 // (op 1046) 

v_mov_b32 v15, 0 // (op 1046) convert
v_mov_b32 v14, v13 // (op 1046) convert

v_lshrrev_b32 v12, 2, v1 // (op 1046) 
v_mul_lo_u32 v23, v12, s25 // (op 1046) most significant: low of low * high
v_mul_hi_u32 v13, v12, s24 // (op 1046) most significant: high of low * low
v_mul_lo_u32 v22, v12, s24 // (op 1046) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 1046) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v14 // (op 1046) least significant half
v_addc_co_u32 v23, s[0:1], v23, v15, s[0:1] // (op 1046) most significant half



s_mul_i32 s1, s24, s29 // (op 1048) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 1048) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 1048) least significant
s_mul_hi_u32 s7, s24, s28 // (op 1048) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1048) most significant: sum
s_add_u32 s1, s1, s7 // (op 1048) most significant: sum


s_mov_b32 s40, s26 // (op 1048) reg expression
s_mov_b32 s41, s27 // (op 1048) reg expression
s_mov_b32 s42, s0 // (op 1048) reg expression
s_mov_b32 s43, 131072 // (op 1048) call()

v_lshrrev_b32 v12, 7, v1 // (op 1091) 
v_and_b32 v13, 15, v1 // (op 1091) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 1091) 


v_lshl_add_u32 v15, s16, 5, v14 // (op 1091) 

v_and_b32 v12, 63, v1 // (op 1091) 
v_lshrrev_b32 v13, 4, v12 // (op 1091) 

v_mul_lo_u32 v12, v15, s30 // (op 1091) 
v_add_u32 v12, v12, v13 // (op 1091) 


s_mul_i32 s1, s30, s23 // (op 1093) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1093) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1093) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1093) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1093) most significant: sum
s_add_u32 s1, s1, s7 // (op 1093) most significant: sum


s_mov_b32 s52, s32 // (op 1093) reg expression
s_mov_b32 s53, s33 // (op 1093) reg expression
s_mov_b32 s54, s0 // (op 1093) reg expression
s_mov_b32 s55, 131072 // (op 1093) call()

v_lshrrev_b32 v13, 7, v1 // (op 1110) 
v_and_b32 v14, 15, v1 // (op 1110) 
v_lshl_add_u32 v15, v13, 4, v14 // (op 1110) 


v_lshl_add_u32 v13, s16, 5, v15 // (op 1110) 

v_and_b32 v14, 63, v1 // (op 1110) 
v_lshrrev_b32 v15, 4, v14 // (op 1110) 

v_mul_lo_u32 v14, v13, s30 // (op 1110) 
v_add_u32 v14, v14, v15 // (op 1110) 


s_mul_i32 s1, s30, s23 // (op 1112) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1112) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1112) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1112) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1112) most significant: sum
s_add_u32 s1, s1, s7 // (op 1112) most significant: sum


s_mov_b32 s52, s32 // (op 1112) reg expression
s_mov_b32 s53, s33 // (op 1112) reg expression
s_mov_b32 s54, s0 // (op 1112) reg expression
s_mov_b32 s55, 131072 // (op 1112) call()

v_and_b32 v13, -64, v1 // (op 1155) 
v_sub_u32 v15, v1, v13 // (op 1155) 

v_lshl_add_u32 v13, s17, 6, v15 // (op 1155) 

v_lshrrev_b32 v15, 6, v1 // (op 1155) 
v_mul_lo_u32 v24, v15, s34 // (op 1155) 
v_add_u32 v24, v24, v13 // (op 1155) 


s_mul_i32 s1, s34, s39 // (op 1157) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1157) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1157) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1157) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1157) most significant: sum
s_add_u32 s1, s1, s7 // (op 1157) most significant: sum


s_mov_b32 s56, s36 // (op 1157) reg expression
s_mov_b32 s57, s37 // (op 1157) reg expression
s_mov_b32 s58, s0 // (op 1157) reg expression
s_mov_b32 s59, 131072 // (op 1157) call()

v_and_b32 v13, -64, v1 // (op 1174) 
v_sub_u32 v15, v1, v13 // (op 1174) 

v_lshl_add_u32 v13, s17, 6, v15 // (op 1174) 

v_lshrrev_b32 v15, 6, v1 // (op 1174) 
v_mul_lo_u32 v25, v15, s34 // (op 1174) 
v_add_u32 v25, v25, v13 // (op 1174) 


s_mul_i32 s1, s34, s39 // (op 1176) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1176) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1176) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1176) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1176) most significant: sum
s_add_u32 s1, s1, s7 // (op 1176) most significant: sum


s_mov_b32 s56, s36 // (op 1176) reg expression
s_mov_b32 s57, s37 // (op 1176) reg expression
s_mov_b32 s58, s0 // (op 1176) reg expression
s_mov_b32 s59, 131072 // (op 1176) call()




buffer_load_ubyte v13, v12, s[52:55], 0 offen offset: 0 // (op 769) Load value
v_and_b32 v15, -4, v1 // (op 1292) 
v_sub_u32 v26, v1, v15 // (op 1292) 

v_lshlrev_b32 v15, 4, v26 // (op 1292) 

v_lshrrev_b32 v26, 2, v1 // (op 1292) 
v_lshl_add_u32 v27, v26, 6, v15 // (op 1292) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1292) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 911) 
v_mov_b64 v[26:27], s[6:7] // (op 911) convert


v_mov_b64 v[28:29], v[20:21] // (op 911) reg expression
s_mov_b32 m0, s0 // (op 911) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 911) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 911) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 911) most significant half

s_add_u32 m0, m0, 4096 // (op 911) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 911) Load value direct to lds





v_mov_b32 v15, v24 // (op 776) reg expression

buffer_load_ubyte v26, v15, s[56:59], 0 offen offset: 0 // (op 776) Load value

v_and_b32 v15, -8, v1 // (op 1240) 
v_sub_u32 v27, v1, v15 // (op 1240) 

v_lshlrev_b32 v15, 4, v27 // (op 1240) 

v_lshrrev_b32 v27, 3, v1 // (op 1240) 
v_lshl_add_u32 v28, v27, 7, v15 // (op 1240) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1240) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 8192, s0 // (op 916) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 916) reg expression
buffer_load_dwordx4 v16, s[8:11], 0 offen offset: 0 lds // (op 916) Load value direct to lds





buffer_load_ubyte v15, v12, s[52:55], 0 offen offset:4 // (op 783) Load value

v_and_b32 v12, -4, v1 // (op 1451) 
v_sub_u32 v27, v1, v12 // (op 1451) 

v_lshlrev_b32 v12, 4, v27 // (op 1451) 

v_lshrrev_b32 v27, 2, v1 // (op 1451) 
v_lshl_add_u32 v28, v27, 6, v12 // (op 1451) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1451) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 921) 
v_mov_b64 v[28:29], s[6:7] // (op 921) convert


s_lshl_b64 s[6:7], s[24:25], 7 // (op 921) 
v_mov_b32 v12, s6 // (op 921) 
v_mov_b32 v27, s7 // (op 921) 
v_add_co_u32 v30, s[14:15], v20, v12 // (op 921) least significant half
v_addc_co_u32 v31, s[14:15], v21, v27, s[14:15] // (op 921) most significant half




s_add_u32 s1, 12288, s0 // (op 921) 
s_mov_b32 m0, s1 // (op 921) reg expression
buffer_load_dwordx4 v30, s[40:43], 0 offen offset: 0 lds // (op 921) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 921) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 921) most significant half

s_add_u32 m0, m0, 4096 // (op 921) 
buffer_load_dwordx4 v30, s[40:43], 0 offen offset: 0 lds // (op 921) Load value direct to lds







s_lshl_b32 s0, s34, 2 // (op 790) 
v_add_u32 v12, v24, s0 // (op 790) 


buffer_load_ubyte v20, v12, s[56:59], 0 offen offset: 0 // (op 790) Load value


v_and_b32 v12, -8, v1 // (op 1412) 
v_sub_u32 v21, v1, v12 // (op 1412) 

v_lshlrev_b32 v12, 4, v21 // (op 1412) 

v_lshrrev_b32 v21, 3, v1 // (op 1412) 
v_lshl_add_u32 v24, v21, 7, v12 // (op 1412) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 1412) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 20480, s0 // (op 923) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 923) reg expression
s_mov_b32 s6, 128 // (op 923) call()
buffer_load_dwordx4 v16, s[8:11], s6 offen offset: 0 lds // (op 923) Load value direct to lds




v_and_b32 v12, -64, v1 // (op 1350) 
v_sub_u32 v21, v1, v12 // (op 1350) 

v_and_b32 v12, 4294967232, v1 // (op 1350) 
v_add_u32 v24, v12, v21 // (op 1350) 





s_waitcnt vmcnt(6)
ds_write_b8 v24, v26 offset:24576 // (op 796) Store local data 

s_barrier  // (op 1554) 



ds_read_b64_tr_b8 v[28:29], v5 // (op 814) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v5 offset:512 // (op 814) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v5 offset:4096 // (op 814) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v5 offset:4608 // (op 814) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1570) 


ds_read_u8 v12, v0 offset:24576 // (op 824) Load local data 



ds_read_b64_tr_b8 v[36:37], v5 offset:32 // (op 819) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v5 offset:544 // (op 819) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v5 offset:4128 // (op 819) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v5 offset:4640 // (op 819) Transpose load local data 


ds_read_u8 v21, v0 offset:24608 // (op 829) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1538) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 810) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 810) Load local data 
v_and_b32 v24, -8, v1 // (op 1212) 
v_sub_u32 v27, v1, v24 // (op 1212) 

v_lshlrev_b32 v24, 4, v27 // (op 1212) 

v_lshrrev_b32 v27, 3, v1 // (op 1212) 
v_lshl_add_u32 v16, v27, 7, v24 // (op 1212) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 1212) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 1264) 
v_sub_u32 v27, v1, v24 // (op 1264) 

v_lshlrev_b32 v24, 4, v27 // (op 1264) 

v_lshrrev_b32 v27, 2, v1 // (op 1264) 
v_lshl_add_u32 v16, v27, 6, v24 // (op 1264) 


s_nop 0
v_readfirstlane_b32 s1, v16 // (op 1264) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -64, v1 // (op 1321) 
v_sub_u32 v27, v1, v24 // (op 1321) 

v_and_b32 v24, 4294967232, v1 // (op 1321) 
v_add_u32 v16, v24, v27 // (op 1321) 


v_and_b32 v17, -8, v1 // (op 1397) 
v_sub_u32 v24, v1, v17 // (op 1397) 

v_lshlrev_b32 v17, 4, v24 // (op 1397) 

v_lshrrev_b32 v24, 3, v1 // (op 1397) 
v_lshl_add_u32 v27, v24, 7, v17 // (op 1397) 


s_nop 0
v_readfirstlane_b32 s6, v27 // (op 1397) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -4, v1 // (op 1436) 
v_sub_u32 v24, v1, v17 // (op 1436) 

v_lshlrev_b32 v17, 4, v24 // (op 1436) 

v_lshrrev_b32 v24, 2, v1 // (op 1436) 
v_lshl_add_u32 v27, v24, 6, v17 // (op 1436) 


s_nop 0
v_readfirstlane_b32 s7, v27 // (op 1436) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -64, v1 // (op 1480) 
v_sub_u32 v24, v1, v17 // (op 1480) 

v_and_b32 v17, 4294967232, v1 // (op 1480) 
v_add_u32 v27, v17, v24 // (op 1480) 


s_mov_b32 s14, 0 // (op 312) call()
s_ashr_i64 s[18:19], s[28:29], 7 // (op 311) 
s_mov_b32 s15, s18 // (op 311) convert

s_and_b32 s18, s15, 4294967294 // (op 311) 

s_cmp_lt_u32 s14, s18 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_311 // (op 311) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_311 if false)
RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoop_311:
s_add_u32 s14, s14, 2 // (op 313) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[28:35], a[0:3], v13, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 84) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[36:43], a[4:7], v13, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 347) 


v_mov_b32 v17, v14 // (op 77) 

buffer_load_ubyte v13, v17, s[52:55], 0 offen offset:8 // (op 77) Load value
s_barrier  // (op 1557) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 883) 
v_mov_b64 v[52:53], s[18:19] // (op 883) convert


v_mov_b64 v[54:55], v[22:23] // (op 883) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 883) 
s_lshl_b64 s[60:61], s[18:19], 1 // (op 883) 

v_mov_b32 v24, s60 // (op 883) 
v_mov_b32 v56, s61 // (op 883) 
v_add_co_u32 v58, s[18:19], v54, v24 // (op 883) least significant half
v_addc_co_u32 v59, s[18:19], v55, v56, s[18:19] // (op 883) most significant half




s_mov_b32 m0, s1 // (op 883) reg expression
buffer_load_dwordx4 v58, s[40:43], 0 offen offset: 0 lds // (op 883) Load value direct to lds
v_add_co_u32 v58, s[18:19], v58, v52 // (op 883) least significant half
v_addc_co_u32 v59, s[18:19], 0, v59, s[18:19] // (op 883) most significant half

s_add_u32 m0, m0, 4096 // (op 883) 
buffer_load_dwordx4 v58, s[40:43], 0 offen offset: 0 lds // (op 883) Load value direct to lds




v_mov_b32 v24, v25 // (op 21) 
s_lshl_b32 s15, s34, 3 // (op 21) 
v_add_u32 v52, v24, s15 // (op 21) 


buffer_load_ubyte v26, v52, s[56:59], 0 offen offset: 0 // (op 21) Load value

s_barrier  // (op 1541) 


v_mov_b64 v[52:53], v[18:19] // (op 878) 
s_add_u32 s15, 8192, s0 // (op 878) 

s_mov_b32 m0, s15 // (op 878) reg expression
s_mov_b32 s18, 256 // (op 878) call()
buffer_load_dwordx4 v52, s[8:11], s18 offen offset: 0 lds // (op 878) Load value direct to lds


s_barrier  // (op 1602) 



s_waitcnt vmcnt(6)
ds_write_b8 v27, v20 offset:24832 // (op 240) Store local data 
s_barrier  // (op 1593) 



ds_read_b64_tr_b8 v[28:29], v8 offset:12288 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v8 offset:12800 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v8 offset:16384 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v8 offset:16896 // (op 233) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1599) 


ds_read_u8 v12, v7 offset:24832 // (op 237) Load local data 



ds_read_b64_tr_b8 v[36:37], v8 offset:12320 // (op 352) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v8 offset:12832 // (op 352) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v8 offset:16416 // (op 352) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v8 offset:16928 // (op 352) Transpose load local data 


ds_read_u8 v21, v7 offset:24864 // (op 355) Load local data 
s_barrier  // (op 1657) 
s_waitcnt vmcnt(5)
s_barrier  // (op 1587) 



ds_read_b128 v[44:47], v4 offset:20480 // (op 231) Load local data 
ds_read_b128 v[48:51], v4 offset:20544 // (op 231) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[28:35], a[0:3], v15, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 238) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[36:43], a[4:7], v15, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 348) 



buffer_load_ubyte v15, v17, s[52:55], 0 offen offset:12 // (op 235) Load value

s_barrier  // (op 1596) 
v_add_u32 v14, 8, v14 // (op 1102) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 906) 
v_mov_b64 v[56:57], s[18:19] // (op 906) convert


s_lshl_b64 s[18:19], s[24:25], 7 // (op 906) 
s_mul_i32 s61, 3, s19 // (op 906) most significant: low of low * high
s_mul_i32 s60, 3, s18 // (op 906) least significant
s_mul_hi_u32 s15, 3, s18 // (op 906) most significant: high of low * low
s_add_u32 s61, s61, s15 // (op 906) most significant: sum


v_mov_b32 v17, s60 // (op 906) 
v_mov_b32 v58, s61 // (op 906) 
v_add_co_u32 v60, s[18:19], v54, v17 // (op 906) least significant half
v_addc_co_u32 v61, s[18:19], v55, v58, s[18:19] // (op 906) most significant half




s_add_u32 s15, 12288, s7 // (op 906) 
s_mov_b32 m0, s15 // (op 906) reg expression
buffer_load_dwordx4 v60, s[40:43], 0 offen offset: 0 lds // (op 906) Load value direct to lds
v_add_co_u32 v60, s[18:19], v60, v56 // (op 906) least significant half
v_addc_co_u32 v61, s[18:19], 0, v61, s[18:19] // (op 906) most significant half

s_add_u32 m0, m0, 4096 // (op 906) 
buffer_load_dwordx4 v60, s[40:43], 0 offen offset: 0 lds // (op 906) Load value direct to lds






s_lshl_b32 s15, s34, 2 // (op 242) 
s_mul_i32 s18, 3, s15 // (op 242) 

v_add_u32 v17, v24, s18 // (op 242) 


s_lshl_b64 s[18:19], s[24:25], 7 // (op 1038) 
s_lshl_b64 s[60:61], s[18:19], 1 // (op 1038) 

v_mov_b32 v54, s60 // (op 1038) 
v_mov_b32 v55, s61 // (op 1038) 
v_add_co_u32 v22, s[18:19], v22, v54 // (op 1038) least significant half
v_addc_co_u32 v23, s[18:19], v23, v55, s[18:19] // (op 1038) most significant half




buffer_load_ubyte v20, v17, s[56:59], 0 offen offset: 0 // (op 242) Load value


s_barrier  // (op 1590) 
s_lshl_b32 s15, s34, 3 // (op 1166) 


v_add_u32 v25, v25, s15 // (op 1166) 

s_add_u32 s15, 20480, s6 // (op 901) 

s_mov_b32 m0, s15 // (op 901) reg expression
s_mov_b32 s18, 384 // (op 901) call()
buffer_load_dwordx4 v52, s[8:11], s18 offen offset: 0 lds // (op 901) Load value direct to lds



s_barrier  // (op 1573) 
v_add_co_u32 v18, vcc, 256, v18 // (op 974) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 974) most significant half
s_waitcnt vmcnt(6)
ds_write_b8 v16, v26 offset:24576 // (op 26) Store local data 
s_barrier  // (op 1563) 



ds_read_b64_tr_b8 v[28:29], v5 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v5 offset:512 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v5 offset:4096 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v5 offset:4608 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1580) 


ds_read_u8 v12, v0 offset:24576 // (op 81) Load local data 



ds_read_b64_tr_b8 v[36:37], v5 offset:32 // (op 343) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v5 offset:544 // (op 343) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v5 offset:4128 // (op 343) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v5 offset:4640 // (op 343) Transpose load local data 


ds_read_u8 v21, v0 offset:24608 // (op 346) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1547) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 67) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 67) Load local data 
s_ashr_i64 s[18:19], s[28:29], 7 // (op 311) 
s_mov_b32 s15, s18 // (op 311) convert

s_and_b32 s18, s15, 4294967294 // (op 311) 

s_cmp_lt_u32 s14, s18 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoop_311 // (op 311) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoop_311 if true)
RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_311:













s_barrier  // (op 1641) 
v_lshrrev_b32 v13, 3, v1 // (op 936) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 936) 
v_lshl_add_u32 v14, s16, 5, v13 // (op 936) 

s_lshr_b32 s6, s0, 1 // (op 936) 

v_and_b32 v13, -8, v1 // (op 936) 
v_sub_u32 v16, v1, v13 // (op 936) 

v_lshlrev_b32 v13, 4, v16 // (op 936) 

v_lshl_add_u32 v16, s6, 8, v13 // (op 936) 


v_mov_b32 v19, 0 // (op 936) convert
v_mov_b32 v18, v16 // (op 936) convert

v_mul_lo_u32 v17, v14, s13 // (op 936) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 936) most significant: high of low * low
v_mul_lo_u32 v16, v14, s12 // (op 936) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 936) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v18 // (op 936) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 936) most significant half



s_mul_i32 s1, s12, s23 // (op 938) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 938) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 938) least significant
s_mul_hi_u32 s7, s12, s22 // (op 938) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 938) most significant: sum
s_add_u32 s1, s1, s7 // (op 938) most significant: sum


s_mov_b32 s8, s20 // (op 938) reg expression
s_mov_b32 s9, s21 // (op 938) reg expression
s_mov_b32 s10, s0 // (op 938) reg expression
s_mov_b32 s11, 131072 // (op 938) call()

v_lshrrev_b32 v13, 2, v1 // (op 1004) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 1004) 
v_and_b32 v14, -4, v1 // (op 1004) 
s_lshr_b32 s6, s0, 1 // (op 1004) 

v_sub_u32 v18, v1, v14 // (op 1004) 

v_lshl_add_u32 v14, s6, 8, v13 // (op 1004) 


v_lshlrev_b32 v13, 4, v18 // (op 1004) 

v_lshl_add_u32 v18, s17, 6, v13 // (op 1004) 

v_mov_b32 v23, 0 // (op 1004) convert
v_mov_b32 v22, v18 // (op 1004) convert

v_mul_lo_u32 v19, v14, s25 // (op 1004) most significant: low of low * high
v_mul_hi_u32 v13, v14, s24 // (op 1004) most significant: high of low * low
v_mul_lo_u32 v18, v14, s24 // (op 1004) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 1004) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v22 // (op 1004) least significant half
v_addc_co_u32 v19, s[0:1], v19, v23, s[0:1] // (op 1004) most significant half



s_mul_i32 s1, s24, s29 // (op 1006) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 1006) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 1006) least significant
s_mul_hi_u32 s7, s24, s28 // (op 1006) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1006) most significant: sum
s_add_u32 s1, s1, s7 // (op 1006) most significant: sum


s_mov_b32 s40, s26 // (op 1006) reg expression
s_mov_b32 s41, s27 // (op 1006) reg expression
s_mov_b32 s42, s0 // (op 1006) reg expression
s_mov_b32 s43, 131072 // (op 1006) call()

v_lshrrev_b32 v13, 7, v1 // (op 1068) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 1068) 
v_and_b32 v14, 15, v1 // (op 1068) 
s_lshr_b32 s6, s0, 1 // (op 1068) 

v_lshl_add_u32 v22, v13, 4, v14 // (op 1068) 


v_lshl_add_u32 v13, s16, 5, v22 // (op 1068) 

v_and_b32 v14, 63, v1 // (op 1068) 
v_lshrrev_b32 v22, 4, v14 // (op 1068) 

v_lshl_add_u32 v14, s6, 3, v22 // (op 1068) 


v_mul_lo_u32 v22, v13, s30 // (op 1068) 
v_add_u32 v22, v22, v14 // (op 1068) 


s_mul_i32 s1, s30, s23 // (op 1070) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1070) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1070) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1070) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1070) most significant: sum
s_add_u32 s1, s1, s7 // (op 1070) most significant: sum


s_mov_b32 s52, s32 // (op 1070) reg expression
s_mov_b32 s53, s33 // (op 1070) reg expression
s_mov_b32 s54, s0 // (op 1070) reg expression
s_mov_b32 s55, 131072 // (op 1070) call()

v_lshrrev_b32 v13, 6, v1 // (op 1132) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 1132) 
v_and_b32 v14, -64, v1 // (op 1132) 
s_lshr_b32 s6, s0, 1 // (op 1132) 

v_sub_u32 v23, v1, v14 // (op 1132) 

v_lshl_add_u32 v14, s6, 3, v13 // (op 1132) 


v_lshl_add_u32 v13, s17, 6, v23 // (op 1132) 

v_mul_lo_u32 v23, v14, s34 // (op 1132) 
v_add_u32 v23, v23, v13 // (op 1132) 


s_mul_i32 s1, s34, s39 // (op 1134) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1134) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1134) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1134) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1134) most significant: sum
s_add_u32 s1, s1, s7 // (op 1134) most significant: sum


s_mov_b32 s56, s36 // (op 1134) reg expression
s_mov_b32 s57, s37 // (op 1134) reg expression
s_mov_b32 s58, s0 // (op 1134) reg expression
s_mov_b32 s59, 131072 // (op 1134) call()

v_and_b32 v13, -8, v1 // (op 1225) 
v_sub_u32 v14, v1, v13 // (op 1225) 

v_lshlrev_b32 v13, 4, v14 // (op 1225) 

v_lshrrev_b32 v14, 3, v1 // (op 1225) 
v_lshl_add_u32 v24, v14, 7, v13 // (op 1225) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 1225) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -4, v1 // (op 1277) 
v_sub_u32 v14, v1, v13 // (op 1277) 

v_lshlrev_b32 v13, 4, v14 // (op 1277) 

v_lshrrev_b32 v14, 2, v1 // (op 1277) 
v_lshl_add_u32 v24, v14, 6, v13 // (op 1277) 


s_nop 0
v_readfirstlane_b32 s1, v24 // (op 1277) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -64, v1 // (op 1334) 
v_sub_u32 v14, v1, v13 // (op 1334) 

v_and_b32 v13, 4294967232, v1 // (op 1334) 
v_add_u32 v24, v13, v14 // (op 1334) 


s_ashr_i64 s[6:7], s[28:29], 7 // (op 317) 
s_and_b32 s14, s6, 4294967294 // (op 317) 

s_ashr_i64 s[6:7], s[28:29], 7 // (op 316) 
s_mov_b32 s15, s6 // (op 316) convert

s_cmp_lt_u32 s14, s15 // (op 316) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_316 // (op 316) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_316 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_316:
s_barrier  // (op 1544) 
s_add_u32 s14, s14, 1 // (op 318) 


v_mov_b64 v[20:21], v[16:17] // (op 888) 
s_add_u32 s15, 8192, s0 // (op 888) 

s_mov_b32 m0, s15 // (op 888) reg expression
buffer_load_dwordx4 v20, s[8:11], 0 offen offset: 0 lds // (op 888) Load value direct to lds


v_add_co_u32 v16, vcc, 128, v16 // (op 926) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 926) most significant half
s_barrier  // (op 1560) 

s_lshl_b64 s[6:7], s[24:25], 6 // (op 894) 
v_mov_b64 v[20:21], s[6:7] // (op 894) convert


v_mov_b64 v[12:13], v[18:19] // (op 894) 
v_mov_b64 v[14:15], v[12:13] // (op 894) reg expression
s_mov_b32 m0, s1 // (op 894) reg expression
buffer_load_dwordx4 v14, s[40:43], 0 offen offset: 0 lds // (op 894) Load value direct to lds
v_add_co_u32 v14, s[6:7], v14, v20 // (op 894) least significant half
v_addc_co_u32 v15, s[6:7], 0, v15, s[6:7] // (op 894) most significant half

s_add_u32 m0, m0, 4096 // (op 894) 
buffer_load_dwordx4 v14, s[40:43], 0 offen offset: 0 lds // (op 894) Load value direct to lds



s_lshl_b64 s[6:7], s[24:25], 7 // (op 996) 
v_mov_b32 v12, s6 // (op 996) 
v_mov_b32 v13, s7 // (op 996) 
v_add_co_u32 v18, s[18:19], v18, v12 // (op 996) least significant half
v_addc_co_u32 v19, s[18:19], v19, v13, s[18:19] // (op 996) most significant half






v_mov_b32 v12, v23 // (op 184) 
v_mov_b32 v13, v12 // (op 184) reg expression

buffer_load_ubyte v14, v13, s[56:59], 0 offen offset: 0 // (op 184) Load value


s_lshl_b32 s15, s34, 2 // (op 1124) 
v_add_u32 v23, v23, s15 // (op 1124) 

s_barrier  // (op 1576) 
s_waitcnt vmcnt(3)
s_barrier  // (op 1550) 






ds_read_b128 v[26:29], v3 offset:8192 // (op 173) Load local data 
ds_read_b128 v[30:33], v3 offset:8256 // (op 173) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v14 offset:24576 // (op 182) Store local data 

s_barrier  // (op 1566) 



ds_read_b64_tr_b8 v[34:35], v5 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v5 offset:512 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v5 offset:4096 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v5 offset:4608 // (op 175) Transpose load local data 


v_mov_b32 v25, v22 // (op 177) 

buffer_load_ubyte v12, v25, s[52:55], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v22, 4, v22 // (op 1060) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1583) 


ds_read_u8 v25, v0 offset:24576 // (op 179) Load local data 



ds_read_b64_tr_b8 v[42:43], v5 offset:32 // (op 324) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v12, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 180) 
ds_read_b64_tr_b8 v[44:45], v5 offset:544 // (op 324) Transpose load local data 


ds_read_b64_tr_b8 v[46:47], v5 offset:4128 // (op 324) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v5 offset:4640 // (op 324) Transpose load local data 


ds_read_u8 v25, v0 offset:24608 // (op 327) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[42:49], a[4:7], v12, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 328) 




s_ashr_i64 s[6:7], s[28:29], 7 // (op 316) 
s_mov_b32 s15, s6 // (op 316) convert

s_cmp_lt_u32 s14, s15 // (op 316) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_316 // (op 316) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_316 if true)
RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_316:























s_barrier  // (op 1645) 






v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_125
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
s_branch RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_125:


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
RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
.LRR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_end:
.size RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2, .LRR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_end-RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25088 // lds bytes
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: F8
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
  - .name: RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
    .symbol: RR_GEMM_TT_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 25088
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 62
    .vgpr_count: 62
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

