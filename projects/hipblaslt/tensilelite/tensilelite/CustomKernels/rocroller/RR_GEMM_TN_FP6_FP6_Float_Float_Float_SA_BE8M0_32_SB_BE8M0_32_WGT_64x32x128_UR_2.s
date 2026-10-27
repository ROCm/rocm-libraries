// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2,@function
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 1251) 
v_and_b32 v3, 1, v0 // (op 1251) 

v_and_b32 v0, 15, v1 // (op 1251) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 1251) 


v_and_b32 v0, 63, v1 // (op 1251) 
v_lshrrev_b32 v3, 4, v0 // (op 1251) 

v_lshlrev_b32 v0, 5, v3 // (op 1251) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 1251) 


v_mul_lo_u32 v0, v3, 6 // (op 1251) 

v_lshrrev_b32 v3, 3, v0 // (op 1251) 

v_lshrrev_b32 v0, 6, v1 // (op 1380) 
v_and_b32 v4, 1, v0 // (op 1380) 

v_and_b32 v0, 15, v1 // (op 1380) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 1380) 


v_and_b32 v0, 63, v1 // (op 1380) 
v_lshrrev_b32 v4, 4, v0 // (op 1380) 

v_lshlrev_b32 v0, 5, v4 // (op 1380) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 1380) 


v_mul_lo_u32 v0, v4, 6 // (op 1380) 

v_lshrrev_b32 v4, 3, v0 // (op 1380) 

v_lshrrev_b32 v0, 7, v1 // (op 1136) 
v_and_b32 v5, 63, v1 // (op 1136) 
v_lshrrev_b32 v6, 4, v5 // (op 1136) 
v_lshlrev_b32 v7, 2, v6 // (op 1136) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1136) 


v_lshl_add_u32 v0, s16, 6, v6 // (op 1136) 

v_lshrrev_b32 v6, 2, v5 // (op 1136) 

v_and_b32 v5, 3, v6 // (op 1136) 

v_and_b32 v6, 3, v1 // (op 1136) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1136) 


v_lshrrev_b32 v5, 6, v1 // (op 1136) 
v_and_b32 v6, 1, v5 // (op 1136) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1136) 


v_lshl_add_u32 v6, s17, 5, v5 // (op 1136) 

v_mul_lo_u32 v5, v6, s2 // (op 1136) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 1136) 


v_lshrrev_b32 v0, 7, v1 // (op 1198) 
v_and_b32 v5, 15, v1 // (op 1198) 
v_lshl_add_u32 v7, v0, 4, v5 // (op 1198) 


v_and_b32 v0, 63, v1 // (op 1198) 
v_lshrrev_b32 v5, 4, v0 // (op 1198) 

v_lshlrev_b32 v0, 5, v5 // (op 1198) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 1198) 


v_mul_lo_u32 v0, v5, 6 // (op 1198) 

v_lshrrev_b32 v5, 3, v0 // (op 1198) 

v_lshrrev_b32 v0, 7, v1 // (op 1309) 
v_and_b32 v7, 15, v1 // (op 1309) 
s_mul_i32 s1, s2, s9 // (op 1138) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1138) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1138) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1138) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1138) most significant: sum
s_add_u32 s1, s1, s19 // (op 1138) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1138) 

v_lshl_add_u32 v8, v0, 4, v7 // (op 1309) 


v_and_b32 v0, 63, v1 // (op 1309) 
v_lshrrev_b32 v7, 4, v0 // (op 1309) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 1309) 


v_lshrrev_b32 v7, 7, v1 // (op 1325) 
v_and_b32 v8, 63, v1 // (op 1325) 
v_lshrrev_b32 v9, 4, v8 // (op 1325) 
v_lshlrev_b32 v10, 2, v9 // (op 1325) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1325) 


v_lshl_add_u32 v7, s16, 6, v9 // (op 1325) 

v_lshrrev_b32 v9, 2, v8 // (op 1325) 

v_and_b32 v8, 3, v9 // (op 1325) 

v_and_b32 v9, 3, v1 // (op 1325) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1325) 


v_lshrrev_b32 v8, 6, v1 // (op 1325) 
v_and_b32 v9, 1, v8 // (op 1325) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 1325) 


v_lshl_add_u32 v9, s17, 5, v8 // (op 1325) 

v_mul_lo_u32 v8, v9, s4 // (op 1325) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 1325) 


v_lshrrev_b32 v7, 7, v1 // (op 1356) 
v_and_b32 v8, 15, v1 // (op 1356) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 1356) 


v_and_b32 v7, 63, v1 // (op 1356) 
v_lshrrev_b32 v8, 4, v7 // (op 1356) 

v_lshlrev_b32 v7, 5, v8 // (op 1356) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 1356) 


v_mul_lo_u32 v7, v8, 6 // (op 1356) 

v_lshrrev_b32 v8, 3, v7 // (op 1356) 

v_lshrrev_b32 v7, 7, v1 // (op 1409) 
v_and_b32 v10, 15, v1 // (op 1409) 
s_mul_i32 s1, s4, s9 // (op 1327) most significant: low of low * high
v_lshl_add_u32 v11, v7, 4, v10 // (op 1409) 


v_and_b32 v7, 63, v1 // (op 1409) 
v_lshrrev_b32 v10, 4, v7 // (op 1409) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 1409) 


s_mul_i32 s38, s5, s8 // (op 1327) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1327) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1327) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1327) most significant: sum
s_add_u32 s1, s1, s39 // (op 1327) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 1327) 

s_mov_b32 s40, s6 // (op 1138) reg expression
s_mov_b32 s41, s7 // (op 1138) reg expression
s_mov_b32 s42, s18 // (op 1138) reg expression
s_mov_b32 s43, 131072 // (op 1138) call()

s_mov_b32 s44, s10 // (op 1327) reg expression

s_mov_b32 s45, s11 // (op 1327) reg expression
s_mov_b32 s46, s38 // (op 1327) reg expression
s_mov_b32 s47, 131072 // (op 1327) call()


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
v_lshrrev_b32 v12, 2, v1 // (op 903) 
v_lshl_add_u32 v13, s16, 6, v12 // (op 903) 

v_and_b32 v12, -4, v1 // (op 903) 
v_sub_u32 v14, v1, v12 // (op 903) 

v_lshlrev_b32 v12, 5, v14 // (op 903) 

v_mov_b32 v15, 0 // (op 903) convert
v_mov_b32 v14, v12 // (op 903) convert

v_mul_lo_u32 v17, v13, s13 // (op 903) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 903) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 903) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 903) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 903) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 903) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 903) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 903) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 903) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 903) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 903) 

v_mov_b32 v12, v14 // (op 903) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 905) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 905) least significant
s_mul_hi_u32 s6, s22, 6 // (op 905) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 905) most significant: sum

s_mul_i32 s7, s12, s1 // (op 905) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 905) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 905) least significant
s_mul_hi_u32 s11, s12, s0 // (op 905) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 905) most significant: sum
s_add_u32 s7, s7, s11 // (op 905) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 905) 

s_mov_b32 s48, s20 // (op 905) reg expression
s_mov_b32 s49, s21 // (op 905) reg expression
s_mov_b32 s50, s0 // (op 905) reg expression
s_mov_b32 s51, 131072 // (op 905) call()

v_lshrrev_b32 v13, 2, v1 // (op 922) 
v_lshl_add_u32 v14, s16, 6, v13 // (op 922) 

v_and_b32 v13, -4, v1 // (op 922) 
v_sub_u32 v15, v1, v13 // (op 922) 

v_lshlrev_b32 v13, 5, v15 // (op 922) 

v_mov_b32 v17, 0 // (op 922) convert
v_mov_b32 v16, v13 // (op 922) convert

v_mul_lo_u32 v19, v14, s13 // (op 922) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 922) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 922) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 922) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 922) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 922) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 922) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 922) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 922) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 922) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 922) 

v_mov_b32 v13, v14 // (op 922) convert

s_mul_i32 s1, s23, 6 // (op 924) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 924) least significant
s_mul_hi_u32 s6, s22, 6 // (op 924) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 924) most significant: sum

s_mul_i32 s7, s12, s1 // (op 924) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 924) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 924) least significant
s_mul_hi_u32 s11, s12, s0 // (op 924) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 924) most significant: sum
s_add_u32 s7, s7, s11 // (op 924) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 924) 

s_mov_b32 s48, s20 // (op 924) reg expression
s_mov_b32 s49, s21 // (op 924) reg expression
s_mov_b32 s50, s0 // (op 924) reg expression
s_mov_b32 s51, 131072 // (op 924) call()

v_and_b32 v14, -8, v1 // (op 967) 
v_sub_u32 v15, v1, v14 // (op 967) 

v_lshlrev_b32 v14, 4, v15 // (op 967) 

v_mov_b32 v17, 0 // (op 967) convert
v_mov_b32 v16, v14 // (op 967) convert

v_lshrrev_b32 v14, 3, v1 // (op 967) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 967) 

v_mul_lo_u32 v19, v15, s25 // (op 967) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 967) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 967) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 967) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 967) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 967) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 967) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 967) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 967) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 967) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 967) 

v_mov_b32 v16, v14 // (op 967) convert

s_mul_i32 s1, s9, 6 // (op 969) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 969) least significant
s_mul_hi_u32 s6, s8, 6 // (op 969) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 969) most significant: sum

s_mul_i32 s7, s24, s1 // (op 969) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 969) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 969) least significant
s_mul_hi_u32 s11, s24, s0 // (op 969) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 969) most significant: sum
s_add_u32 s7, s7, s11 // (op 969) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 969) 

s_mov_b32 s52, s26 // (op 969) reg expression
s_mov_b32 s53, s27 // (op 969) reg expression
s_mov_b32 s54, s0 // (op 969) reg expression
s_mov_b32 s55, 131072 // (op 969) call()

v_and_b32 v14, -8, v1 // (op 986) 
v_sub_u32 v15, v1, v14 // (op 986) 

v_lshlrev_b32 v14, 4, v15 // (op 986) 

v_mov_b32 v19, 0 // (op 986) convert
v_mov_b32 v18, v14 // (op 986) convert

v_lshrrev_b32 v17, 3, v1 // (op 986) 
v_lshl_add_u32 v14, s17, 5, v17 // (op 986) 

v_mul_lo_u32 v21, v14, s25 // (op 986) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 986) most significant: high of low * low
v_mul_lo_u32 v20, v14, s24 // (op 986) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 986) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 986) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 986) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 986) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 986) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 986) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 986) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 986) 

v_mov_b32 v17, v14 // (op 986) convert

s_mul_i32 s1, s9, 6 // (op 988) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 988) least significant
s_mul_hi_u32 s6, s8, 6 // (op 988) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 988) most significant: sum

s_mul_i32 s7, s24, s1 // (op 988) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 988) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 988) least significant
s_mul_hi_u32 s11, s24, s0 // (op 988) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 988) most significant: sum
s_add_u32 s7, s7, s11 // (op 988) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 988) 

s_mov_b32 s52, s26 // (op 988) reg expression
s_mov_b32 s53, s27 // (op 988) reg expression
s_mov_b32 s54, s0 // (op 988) reg expression
s_mov_b32 s55, 131072 // (op 988) call()

v_lshrrev_b32 v14, 2, v1 // (op 1031) 
v_lshl_add_u32 v15, s16, 6, v14 // (op 1031) 

v_and_b32 v14, -4, v1 // (op 1031) 
v_sub_u32 v18, v1, v14 // (op 1031) 

v_mul_lo_u32 v14, v15, s28 // (op 1031) 
v_add_u32 v14, v14, v18 // (op 1031) 


s_mul_i32 s1, s28, s23 // (op 1033) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1033) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1033) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1033) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1033) most significant: sum
s_add_u32 s1, s1, s7 // (op 1033) most significant: sum


s_mov_b32 s56, s30 // (op 1033) reg expression
s_mov_b32 s57, s31 // (op 1033) reg expression
s_mov_b32 s58, s0 // (op 1033) reg expression
s_mov_b32 s59, 131072 // (op 1033) call()

v_lshrrev_b32 v15, 2, v1 // (op 1050) 
v_lshl_add_u32 v18, s16, 6, v15 // (op 1050) 

v_and_b32 v15, -4, v1 // (op 1050) 
v_sub_u32 v19, v1, v15 // (op 1050) 

v_mul_lo_u32 v15, v18, s28 // (op 1050) 
v_add_u32 v15, v15, v19 // (op 1050) 


s_mul_i32 s1, s28, s23 // (op 1052) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1052) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1052) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1052) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1052) most significant: sum
s_add_u32 s1, s1, s7 // (op 1052) most significant: sum


s_mov_b32 s56, s30 // (op 1052) reg expression
s_mov_b32 s57, s31 // (op 1052) reg expression
s_mov_b32 s58, s0 // (op 1052) reg expression
s_mov_b32 s59, 131072 // (op 1052) call()

v_lshrrev_b32 v18, 6, v1 // (op 1095) 
v_and_b32 v19, 1, v18 // (op 1095) 

v_and_b32 v18, 15, v1 // (op 1095) 
v_lshl_add_u32 v20, v19, 4, v18 // (op 1095) 


v_lshl_add_u32 v18, s17, 5, v20 // (op 1095) 

v_mul_lo_u32 v19, v18, s32 // (op 1095) 

v_and_b32 v18, 63, v1 // (op 1095) 
v_lshrrev_b32 v20, 4, v18 // (op 1095) 

v_add_u32 v18, v20, v19 // (op 1095) 


s_mul_i32 s1, s32, s9 // (op 1097) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1097) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1097) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1097) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1097) most significant: sum
s_add_u32 s1, s1, s7 // (op 1097) most significant: sum


s_mov_b32 s60, s34 // (op 1097) reg expression
s_mov_b32 s61, s35 // (op 1097) reg expression
s_mov_b32 s62, s0 // (op 1097) reg expression
s_mov_b32 s63, 131072 // (op 1097) call()

v_lshrrev_b32 v19, 6, v1 // (op 1114) 
v_and_b32 v20, 1, v19 // (op 1114) 

v_and_b32 v19, 15, v1 // (op 1114) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 1114) 


v_lshl_add_u32 v19, s17, 5, v21 // (op 1114) 

v_mul_lo_u32 v20, v19, s32 // (op 1114) 

v_and_b32 v19, 63, v1 // (op 1114) 
v_lshrrev_b32 v21, 4, v19 // (op 1114) 

v_add_u32 v19, v21, v20 // (op 1114) 


s_mul_i32 s1, s32, s9 // (op 1116) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1116) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1116) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1116) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1116) most significant: sum
s_add_u32 s1, s1, s7 // (op 1116) most significant: sum


s_mov_b32 s60, s34 // (op 1116) reg expression
s_mov_b32 s61, s35 // (op 1116) reg expression
s_mov_b32 s62, s0 // (op 1116) reg expression
s_mov_b32 s63, 131072 // (op 1116) call()




buffer_load_ubyte v20, v18, s[60:63], 0 offen offset: 0 // (op 774) Load value



buffer_load_dwordx3 v[22:24], v16, s[52:55], 0 offen // (op 778) Load value



buffer_load_dwordx4 v[26:29], v12, s[48:51], 0 offen // (op 781) Load value
buffer_load_dwordx2 v[30:31], v12, s[48:51], 0 offen offset:16 // (op 781) Load value



buffer_load_ubyte v21, v14, s[56:59], 0 offen offset: 0 // (op 784) Load value



buffer_load_ubyte v25, v18, s[60:63], 0 offen offset:4 // (op 788) Load value




buffer_load_dwordx3 v[32:34], v16, s[52:55], 0 offen offset:96 // (op 792) Load value




buffer_load_dwordx4 v[36:39], v12, s[48:51], 0 offen offset:96 // (op 795) Load value
buffer_load_dwordx2 v[40:41], v12, s[48:51], 0 offen offset:112 // (op 795) Load value




buffer_load_ubyte v12, v14, s[56:59], 0 offen offset:4 // (op 798) Load value

v_and_b32 v14, -8, v1 // (op 1239) 
v_sub_u32 v16, v1, v14 // (op 1239) 

v_lshlrev_b32 v14, 4, v16 // (op 1239) 

v_lshrrev_b32 v16, 3, v1 // (op 1239) 
v_lshl_add_u32 v18, v16, 7, v14 // (op 1239) 


v_mul_lo_u32 v14, v18, 6 // (op 1239) 

v_lshrrev_b32 v16, 3, v14 // (op 1239) 




s_waitcnt vmcnt(8)
ds_write_b96 v16, v[22:24] // (op 800) Store local data 

v_and_b32 v14, -4, v1 // (op 1181) 
v_sub_u32 v16, v1, v14 // (op 1181) 

v_lshlrev_b32 v14, 5, v16 // (op 1181) 

v_lshrrev_b32 v16, 2, v1 // (op 1181) 
v_lshl_add_u32 v18, v16, 7, v14 // (op 1181) 


v_mul_lo_u32 v14, v18, 6 // (op 1181) 

v_lshrrev_b32 v16, 3, v14 // (op 1181) 




s_waitcnt vmcnt(7)
ds_write_b128 v16, v[26:29] offset:3072 // (op 801) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v16, v[30:31] offset:3088 // (op 801) Store local data 

v_and_b32 v14, -4, v1 // (op 1292) 
v_sub_u32 v16, v1, v14 // (op 1292) 

v_and_b32 v14, 4294967292, v1 // (op 1292) 
v_add_u32 v18, v14, v16 // (op 1292) 





s_waitcnt vmcnt(5)
ds_write_b8 v18, v21 offset:9216 // (op 802) Store local data 

v_and_b32 v14, -4, v1 // (op 1152) 
v_sub_u32 v16, v1, v14 // (op 1152) 

v_lshlrev_b32 v14, 5, v16 // (op 1152) 

v_lshrrev_b32 v16, 2, v1 // (op 1152) 
v_lshl_add_u32 v18, v16, 7, v14 // (op 1152) 


v_mul_lo_u32 v14, v18, 6 // (op 1152) 

v_lshrrev_b32 v16, 3, v14 // (op 1152) 

v_and_b32 v14, -8, v1 // (op 1210) 
v_sub_u32 v18, v1, v14 // (op 1210) 

v_lshlrev_b32 v14, 4, v18 // (op 1210) 

v_lshrrev_b32 v18, 3, v1 // (op 1210) 
v_lshl_add_u32 v35, v18, 7, v14 // (op 1210) 


v_mul_lo_u32 v14, v35, 6 // (op 1210) 

v_lshrrev_b32 v18, 3, v14 // (op 1210) 

v_and_b32 v14, -4, v1 // (op 1263) 
v_sub_u32 v35, v1, v14 // (op 1263) 

v_and_b32 v14, 4294967292, v1 // (op 1263) 
v_add_u32 v42, v14, v35 // (op 1263) 


v_and_b32 v14, -4, v1 // (op 1339) 
v_sub_u32 v35, v1, v14 // (op 1339) 

v_lshlrev_b32 v14, 5, v35 // (op 1339) 

v_lshrrev_b32 v35, 2, v1 // (op 1339) 
v_lshl_add_u32 v43, v35, 7, v14 // (op 1339) 


v_mul_lo_u32 v14, v43, 6 // (op 1339) 

v_lshrrev_b32 v35, 3, v14 // (op 1339) 

v_and_b32 v14, -8, v1 // (op 1368) 
v_sub_u32 v43, v1, v14 // (op 1368) 

v_lshlrev_b32 v14, 4, v43 // (op 1368) 

v_lshrrev_b32 v43, 3, v1 // (op 1368) 
v_lshl_add_u32 v44, v43, 7, v14 // (op 1368) 


v_mul_lo_u32 v14, v44, 6 // (op 1368) 

v_lshrrev_b32 v43, 3, v14 // (op 1368) 

v_and_b32 v14, -4, v1 // (op 1392) 
v_sub_u32 v44, v1, v14 // (op 1392) 

v_and_b32 v14, 4294967292, v1 // (op 1392) 
v_add_u32 v45, v14, v44 // (op 1392) 


s_mov_b32 s0, 0 // (op 311) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 310) 
s_mov_b32 s1, s6 // (op 310) convert

s_and_b32 s6, s1, 4294967294 // (op 310) 

s_cmp_lt_u32 s0, s6 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_310 // (op 310) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_310 if false)
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoop_310:
s_add_u32 s0, s0, 2 // (op 312) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1450) 



ds_read_b128 v[46:49], v5 offset:3072 // (op 65) Load local data 
ds_read_b64 v[50:51], v5 offset:3088 // (op 65) Load local data 
s_barrier  // (op 1463) 



ds_read_b128 v[52:55], v3 // (op 70) Load local data 
ds_read_b64 v[56:57], v3 offset:16 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1477) 



ds_read_u8 v14, v0 offset:9216 // (op 75) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:51], v[52:57], a[0:3], v14, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 82) 
ds_read_b128 v[58:61], v5 offset:6144 // (op 340) Load local data 


ds_read_b64 v[62:63], v5 offset:6160 // (op 340) Load local data 



ds_read_u8 v14, v0 offset:9344 // (op 344) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:63], v[52:57], a[4:7], v14, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 346) 





v_mov_b32 v14, v19 // (op 80) 

buffer_load_ubyte v20, v14, s[60:63], 0 offen offset:8 // (op 80) Load value


v_mov_b32 v44, v17 // (op 10) 

buffer_load_dwordx3 v[22:24], v44, s[52:55], 0 offen offset:192 // (op 10) Load value


v_mov_b32 v46, v13 // (op 4) 

buffer_load_dwordx4 v[26:29], v46, s[48:51], 0 offen offset:192 // (op 4) Load value
buffer_load_dwordx2 v[30:31], v46, s[48:51], 0 offen offset:208 // (op 4) Load value


v_mov_b32 v47, v15 // (op 17) 

buffer_load_ubyte v21, v47, s[56:59], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 1500) 



s_waitcnt vmcnt(8)
ds_write_b96 v43, v[32:34] offset:9472 // (op 235) Store local data 
s_barrier  // (op 1494) 



s_waitcnt vmcnt(7)
ds_write_b128 v35, v[36:39] offset:12544 // (op 225) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v35, v[40:41] offset:12560 // (op 225) Store local data 
s_barrier  // (op 1506) 



s_waitcnt vmcnt(5)
ds_write_b8 v45, v12 offset:18688 // (op 236) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1491) 



ds_read_b128 v[48:51], v8 offset:12544 // (op 227) Load local data 
ds_read_b64 v[52:53], v8 offset:12560 // (op 227) Load local data 
s_barrier  // (op 1497) 



ds_read_b128 v[54:57], v4 offset:9472 // (op 229) Load local data 
ds_read_b64 v[58:59], v4 offset:9488 // (op 229) Load local data 
s_barrier  // (op 1543) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1503) 



ds_read_u8 v60, v7 offset:18688 // (op 231) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:53], v[54:59], a[0:3], v60, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 234) 
ds_read_b128 v[62:65], v8 offset:15616 // (op 353) Load local data 


ds_read_b64 v[66:67], v8 offset:15632 // (op 353) Load local data 



s_barrier  // (op 1577) 
ds_read_u8 v48, v7 offset:18816 // (op 357) Load local data 
s_barrier  // (op 1581) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:67], v[54:59], a[4:7], v48, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 347) 






buffer_load_ubyte v25, v14, s[60:63], 0 offen offset:12 // (op 233) Load value




v_add_u32 v19, 8, v19 // (op 1106) 
buffer_load_dwordx3 v[32:34], v44, s[52:55], 0 offen offset:288 // (op 237) Load value




v_add_u32 v17, 192, v17 // (op 978) 
buffer_load_dwordx4 v[36:39], v46, s[48:51], 0 offen offset:288 // (op 224) Load value
buffer_load_dwordx2 v[40:41], v46, s[48:51], 0 offen offset:304 // (op 224) Load value




v_add_u32 v13, 192, v13 // (op 914) 
buffer_load_ubyte v12, v47, s[56:59], 0 offen offset:12 // (op 238) Load value

s_barrier  // (op 1466) 
v_add_u32 v15, 8, v15 // (op 1042) 



s_waitcnt vmcnt(8)
ds_write_b96 v18, v[22:24] // (op 13) Store local data 
s_barrier  // (op 1453) 



s_waitcnt vmcnt(7)
ds_write_b128 v16, v[26:29] offset:3072 // (op 6) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v16, v[30:31] offset:3088 // (op 6) Store local data 
s_barrier  // (op 1480) 



s_waitcnt vmcnt(5)
ds_write_b8 v42, v21 offset:9216 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 310) 
s_mov_b32 s1, s6 // (op 310) convert

s_and_b32 s6, s1, 4294967294 // (op 310) 

s_cmp_lt_u32 s0, s6 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoop_310 // (op 310) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoop_310 if true)
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_310:















v_lshrrev_b32 v35, 2, v1 // (op 876) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 876) 
v_lshl_add_u32 v13, s16, 6, v35 // (op 876) 

s_lshr_b32 s6, s0, 1 // (op 876) 

v_and_b32 v35, -4, v1 // (op 876) 
v_sub_u32 v14, v1, v35 // (op 876) 

v_lshlrev_b32 v35, 5, v14 // (op 876) 

v_lshl_add_u32 v14, s6, 8, v35 // (op 876) 


v_mov_b32 v27, 0 // (op 876) convert
v_mov_b32 v26, v14 // (op 876) convert

v_mul_lo_u32 v15, v13, s13 // (op 876) most significant: low of low * high
v_mul_hi_u32 v35, v13, s12 // (op 876) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 876) least significant: low of low * low
v_add_u32 v15, v15, v35 // (op 876) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v26 // (op 876) least significant half
v_addc_co_u32 v15, s[0:1], v15, v27, s[0:1] // (op 876) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 876) most significant: low of high * low
v_mul_hi_u32 v13, v14, 6 // (op 876) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 876) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 876) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 876) 

v_mov_b32 v13, v14 // (op 876) convert

s_mul_i32 s1, s23, 6 // (op 878) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 878) least significant
s_mul_hi_u32 s6, s22, 6 // (op 878) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 878) most significant: sum

s_mul_i32 s7, s12, s1 // (op 878) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 878) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 878) least significant
s_mul_hi_u32 s11, s12, s0 // (op 878) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 878) most significant: sum
s_add_u32 s7, s7, s11 // (op 878) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 878) 

s_mov_b32 s48, s20 // (op 878) reg expression
s_mov_b32 s49, s21 // (op 878) reg expression
s_mov_b32 s50, s0 // (op 878) reg expression
s_mov_b32 s51, 131072 // (op 878) call()

v_and_b32 v35, -8, v1 // (op 944) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 944) 
v_sub_u32 v14, v1, v35 // (op 944) 

s_lshr_b32 s6, s0, 1 // (op 944) 

v_lshlrev_b32 v35, 4, v14 // (op 944) 

v_lshl_add_u32 v14, s6, 8, v35 // (op 944) 


v_mov_b32 v27, 0 // (op 944) convert
v_mov_b32 v26, v14 // (op 944) convert

v_lshrrev_b32 v35, 3, v1 // (op 944) 
v_lshl_add_u32 v14, s17, 5, v35 // (op 944) 

v_mul_lo_u32 v29, v14, s25 // (op 944) most significant: low of low * high
v_mul_hi_u32 v35, v14, s24 // (op 944) most significant: high of low * low
v_mul_lo_u32 v28, v14, s24 // (op 944) least significant: low of low * low
v_add_u32 v29, v29, v35 // (op 944) most significant: sum


v_add_co_u32 v30, s[0:1], v26, v28 // (op 944) least significant half
v_addc_co_u32 v31, s[0:1], v27, v29, s[0:1] // (op 944) most significant half



v_mul_lo_u32 v15, v31, 6 // (op 944) most significant: low of high * low
v_mul_hi_u32 v35, v30, 6 // (op 944) most significant: high of low * low
v_mul_lo_u32 v14, v30, 6 // (op 944) least significant: low of low * low
v_add_u32 v15, v15, v35 // (op 944) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 944) 

v_mov_b32 v35, v16 // (op 944) convert

s_mul_i32 s1, s9, 6 // (op 946) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 946) least significant
s_mul_hi_u32 s6, s8, 6 // (op 946) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 946) most significant: sum

s_mul_i32 s7, s24, s1 // (op 946) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 946) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 946) least significant
s_mul_hi_u32 s11, s24, s0 // (op 946) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 946) most significant: sum
s_add_u32 s7, s7, s11 // (op 946) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 946) 

s_mov_b32 s52, s26 // (op 946) reg expression
s_mov_b32 s53, s27 // (op 946) reg expression
s_mov_b32 s54, s0 // (op 946) reg expression
s_mov_b32 s55, 131072 // (op 946) call()

v_lshrrev_b32 v14, 2, v1 // (op 1008) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1008) 
v_lshl_add_u32 v15, s16, 6, v14 // (op 1008) 

s_lshr_b32 s6, s0, 1 // (op 1008) 

v_and_b32 v14, -4, v1 // (op 1008) 
v_sub_u32 v16, v1, v14 // (op 1008) 

v_lshl_add_u32 v14, s6, 3, v16 // (op 1008) 


v_mul_lo_u32 v16, v15, s28 // (op 1008) 
v_add_u32 v16, v16, v14 // (op 1008) 


s_mul_i32 s1, s28, s23 // (op 1010) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1010) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1010) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1010) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1010) most significant: sum
s_add_u32 s1, s1, s7 // (op 1010) most significant: sum


s_mov_b32 s56, s30 // (op 1010) reg expression
s_mov_b32 s57, s31 // (op 1010) reg expression
s_mov_b32 s58, s0 // (op 1010) reg expression
s_mov_b32 s59, 131072 // (op 1010) call()

v_and_b32 v14, 63, v1 // (op 1072) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1072) 
v_lshrrev_b32 v15, 4, v14 // (op 1072) 

s_lshr_b32 s6, s0, 1 // (op 1072) 

v_lshrrev_b32 v14, 6, v1 // (op 1072) 
v_lshl_add_u32 v17, s6, 3, v15 // (op 1072) 


v_and_b32 v15, 1, v14 // (op 1072) 

v_and_b32 v14, 15, v1 // (op 1072) 
v_lshl_add_u32 v18, v15, 4, v14 // (op 1072) 


v_lshl_add_u32 v14, s17, 5, v18 // (op 1072) 

v_mul_lo_u32 v15, v14, s32 // (op 1072) 

v_add_u32 v14, v17, v15 // (op 1072) 


s_mul_i32 s1, s32, s9 // (op 1074) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1074) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1074) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1074) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1074) most significant: sum
s_add_u32 s1, s1, s7 // (op 1074) most significant: sum


s_mov_b32 s60, s34 // (op 1074) reg expression
s_mov_b32 s61, s35 // (op 1074) reg expression
s_mov_b32 s62, s0 // (op 1074) reg expression
s_mov_b32 s63, 131072 // (op 1074) call()

v_and_b32 v15, -4, v1 // (op 1165) 
v_sub_u32 v17, v1, v15 // (op 1165) 

v_lshlrev_b32 v15, 5, v17 // (op 1165) 

v_lshrrev_b32 v17, 2, v1 // (op 1165) 
v_lshl_add_u32 v18, v17, 7, v15 // (op 1165) 


v_mul_lo_u32 v15, v18, 6 // (op 1165) 

v_lshrrev_b32 v17, 3, v15 // (op 1165) 

v_and_b32 v15, -8, v1 // (op 1223) 
v_sub_u32 v18, v1, v15 // (op 1223) 

v_lshlrev_b32 v15, 4, v18 // (op 1223) 

v_lshrrev_b32 v18, 3, v1 // (op 1223) 
v_lshl_add_u32 v19, v18, 7, v15 // (op 1223) 


v_mul_lo_u32 v15, v19, 6 // (op 1223) 

v_lshrrev_b32 v18, 3, v15 // (op 1223) 

v_and_b32 v15, -4, v1 // (op 1276) 
v_sub_u32 v19, v1, v15 // (op 1276) 

v_and_b32 v15, 4294967292, v1 // (op 1276) 
v_add_u32 v20, v15, v19 // (op 1276) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 316) 
s_and_b32 s6, s0, 4294967294 // (op 316) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 315) 
s_mov_b32 s7, s0 // (op 315) convert

s_cmp_lt_u32 s6, s7 // (op 315) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_315 // (op 315) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_315 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_315:


s_add_u32 s6, s6, 1 // (op 317) 
v_mov_b32 v12, v13 // (op 166) 

buffer_load_dwordx4 v[22:25], v12, s[48:51], 0 offen // (op 166) Load value
buffer_load_dwordx2 v[26:27], v12, s[48:51], 0 offen offset:16 // (op 166) Load value

v_add_u32 v13, 96, v13 // (op 866) 


s_barrier  // (op 1456) 
v_mov_b32 v12, v35 // (op 179) 




buffer_load_dwordx3 v[28:30], v12, s[52:55], 0 offen // (op 179) Load value

v_add_u32 v35, 96, v35 // (op 936) 
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[22:25] offset:3072 // (op 167) Store local data 


v_mov_b32 v12, v16 // (op 180) 

buffer_load_ubyte v15, v12, s[56:59], 0 offen offset: 0 // (op 180) Load value

v_add_u32 v16, 4, v16 // (op 1000) 
s_waitcnt vmcnt(2)
ds_write_b64 v17, v[26:27] offset:3088 // (op 167) Store local data 

s_barrier  // (op 1469) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1459) 






ds_read_b128 v[22:25], v5 offset:3072 // (op 169) Load local data 
ds_read_b64 v[26:27], v5 offset:3088 // (op 169) Load local data 
s_waitcnt vmcnt(1)
ds_write_b96 v18, v[28:30] // (op 177) Store local data 

s_barrier  // (op 1483) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1473) 






ds_read_b128 v[28:31], v3 // (op 171) Load local data 
ds_read_b64 v[32:33], v3 offset:16 // (op 171) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v15 offset:9216 // (op 178) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1487) 



ds_read_u8 v12, v0 offset:9216 // (op 173) Load local data 


v_mov_b32 v15, v14 // (op 175) 

buffer_load_ubyte v19, v15, s[60:63], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v14, 4, v14 // (op 1064) 



ds_read_b128 v[36:39], v5 offset:6144 // (op 321) Load local data 
ds_read_b64 v[40:41], v5 offset:6160 // (op 321) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:27], v[28:33], a[0:3], v12, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 176) 





ds_read_u8 v12, v0 offset:9344 // (op 325) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[28:33], a[4:7], v12, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 327) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 315) 
s_mov_b32 s7, s0 // (op 315) convert

s_cmp_lt_u32 s6, s7 // (op 315) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_315 // (op 315) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_315 if true)
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_315:






















s_barrier  // (op 1565) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_123:



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
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
.LRR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_end:
.size RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2, .LRR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_end-RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
  .amdhsa_next_free_vgpr 76
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 18944 // lds bytes
  .amdhsa_accum_offset 68
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: F6
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
  - .name: RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
    .symbol: RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 18944
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 68
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
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

