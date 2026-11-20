// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.p2align 8
.type RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2,@function
RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 15, v1 // (op 1251) 
v_and_b32 v3, 48, v1 // (op 1251) 
v_lshl_add_u32 v4, v0, 7, v3 // (op 1251) 


v_and_b32 v0, 15, v1 // (op 1380) 
v_and_b32 v3, 48, v1 // (op 1380) 
v_lshl_add_u32 v5, v0, 7, v3 // (op 1380) 


v_lshrrev_b32 v0, 6, v1 // (op 1136) 
v_and_b32 v3, 63, v1 // (op 1136) 
v_lshrrev_b32 v6, 4, v3 // (op 1136) 
v_lshlrev_b32 v7, 2, v6 // (op 1136) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1136) 


v_lshl_add_u32 v0, s16, 6, v6 // (op 1136) 

v_lshrrev_b32 v6, 2, v3 // (op 1136) 

v_and_b32 v3, 3, v6 // (op 1136) 

v_and_b32 v6, 3, v1 // (op 1136) 
v_lshl_add_u32 v7, v3, 2, v6 // (op 1136) 


v_lshl_add_u32 v3, s17, 4, v7 // (op 1136) 

v_mul_lo_u32 v6, v3, s2 // (op 1136) 

v_add_lshl_u32 v3, v0, v6, 2 // (op 1136) 


v_and_b32 v0, 63, v1 // (op 1198) 
v_and_b32 v6, 48, v1 // (op 1198) 
v_sub_u32 v7, v0, v6 // (op 1198) 

v_lshrrev_b32 v6, 4, v0 // (op 1198) 

v_lshl_add_u32 v0, v6, 5, v7 // (op 1198) 


v_lshrrev_b32 v6, 6, v1 // (op 1198) 
v_lshlrev_b32 v7, 4, v6 // (op 1198) 

v_lshl_add_u32 v6, v0, 6, v7 // (op 1198) 


v_lshlrev_b32 v0, 2, v6 // (op 1198) 

v_lshrrev_b32 v6, 3, v0 // (op 1198) 

v_lshrrev_b32 v0, 6, v1 // (op 1309) 
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

v_lshl_add_u32 v0, v7, 6, v8 // (op 1309) 


v_lshrrev_b32 v7, 6, v1 // (op 1325) 
v_and_b32 v8, 63, v1 // (op 1325) 
v_lshrrev_b32 v9, 4, v8 // (op 1325) 
v_lshlrev_b32 v10, 2, v9 // (op 1325) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1325) 


v_lshl_add_u32 v7, s16, 6, v9 // (op 1325) 

v_lshrrev_b32 v9, 2, v8 // (op 1325) 

v_and_b32 v8, 3, v9 // (op 1325) 

v_and_b32 v9, 3, v1 // (op 1325) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1325) 


v_lshl_add_u32 v8, s17, 4, v10 // (op 1325) 

v_mul_lo_u32 v9, v8, s4 // (op 1325) 

v_add_lshl_u32 v8, v7, v9, 2 // (op 1325) 


v_and_b32 v7, 63, v1 // (op 1356) 
v_and_b32 v9, 48, v1 // (op 1356) 
v_sub_u32 v10, v7, v9 // (op 1356) 

v_lshrrev_b32 v9, 4, v7 // (op 1356) 

v_lshl_add_u32 v7, v9, 5, v10 // (op 1356) 


v_lshrrev_b32 v9, 6, v1 // (op 1356) 
v_lshlrev_b32 v10, 4, v9 // (op 1356) 

v_lshl_add_u32 v9, v7, 6, v10 // (op 1356) 


v_lshlrev_b32 v7, 2, v9 // (op 1356) 

v_lshrrev_b32 v9, 3, v7 // (op 1356) 

v_lshrrev_b32 v7, 6, v1 // (op 1409) 
v_and_b32 v10, 15, v1 // (op 1409) 
s_mul_i32 s1, s4, s9 // (op 1327) most significant: low of low * high
v_lshl_add_u32 v11, v7, 4, v10 // (op 1409) 


v_and_b32 v7, 63, v1 // (op 1409) 
v_lshrrev_b32 v10, 4, v7 // (op 1409) 

v_lshl_add_u32 v7, v10, 6, v11 // (op 1409) 


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
v_and_b32 v12, -2, v1 // (op 903) 
v_sub_u32 v13, v1, v12 // (op 903) 

v_lshlrev_b32 v12, 5, v13 // (op 903) 

v_lshl_add_u32 v13, s16, 6, v12 // (op 903) 

v_mov_b32 v15, 0 // (op 903) convert
v_mov_b32 v14, v13 // (op 903) convert

v_and_b32 v12, 4294967294, v1 // (op 903) 
v_mul_lo_u32 v17, v12, s13 // (op 903) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 903) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 903) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 903) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 903) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 903) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 903) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 903) 

v_mov_b32 v12, v14 // (op 903) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 905) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 905) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 905) least significant
s_mul_hi_u32 s7, s12, s22 // (op 905) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 905) most significant: sum
s_add_u32 s1, s1, s7 // (op 905) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 905) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 905) 

s_mov_b32 s48, s20 // (op 905) reg expression
s_mov_b32 s49, s21 // (op 905) reg expression
s_mov_b32 s50, s0 // (op 905) reg expression
s_mov_b32 s51, 131072 // (op 905) call()

v_and_b32 v13, -2, v1 // (op 922) 
v_sub_u32 v14, v1, v13 // (op 922) 

v_lshlrev_b32 v13, 5, v14 // (op 922) 

v_lshl_add_u32 v14, s16, 6, v13 // (op 922) 

v_mov_b32 v17, 0 // (op 922) convert
v_mov_b32 v16, v14 // (op 922) convert

v_and_b32 v13, 4294967294, v1 // (op 922) 
v_mul_lo_u32 v15, v13, s13 // (op 922) most significant: low of low * high
v_mul_hi_u32 v18, v13, s12 // (op 922) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 922) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 922) most significant: sum


v_add_co_u32 v18, s[0:1], v16, v14 // (op 922) least significant half
v_addc_co_u32 v19, s[0:1], v17, v15, s[0:1] // (op 922) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 922) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 922) 

v_mov_b32 v13, v16 // (op 922) convert

s_mul_i32 s1, s12, s23 // (op 924) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 924) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 924) least significant
s_mul_hi_u32 s7, s12, s22 // (op 924) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 924) most significant: sum
s_add_u32 s1, s1, s7 // (op 924) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 924) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 924) 

s_mov_b32 s48, s20 // (op 924) reg expression
s_mov_b32 s49, s21 // (op 924) reg expression
s_mov_b32 s50, s0 // (op 924) reg expression
s_mov_b32 s51, 131072 // (op 924) call()

v_and_b32 v14, -8, v1 // (op 967) 
v_sub_u32 v15, v1, v14 // (op 967) 

v_lshrrev_b32 v14, 3, v1 // (op 967) 
v_lshl_add_u32 v16, s17, 4, v14 // (op 967) 

v_mul_lo_u32 v14, v16, s24 // (op 967) 

v_lshl_add_u32 v16, v15, 4, v14 // (op 967) 


s_mul_i32 s1, s24, s9 // (op 969) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 969) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 969) least significant
s_mul_hi_u32 s7, s24, s8 // (op 969) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 969) most significant: sum
s_add_u32 s1, s1, s7 // (op 969) most significant: sum


s_mov_b32 s52, s26 // (op 969) reg expression
s_mov_b32 s53, s27 // (op 969) reg expression
s_mov_b32 s54, s0 // (op 969) reg expression
s_mov_b32 s55, 131072 // (op 969) call()

v_and_b32 v14, -8, v1 // (op 986) 
v_sub_u32 v15, v1, v14 // (op 986) 

v_lshrrev_b32 v14, 3, v1 // (op 986) 
v_lshl_add_u32 v17, s17, 4, v14 // (op 986) 

v_mul_lo_u32 v14, v17, s24 // (op 986) 

v_lshl_add_u32 v17, v15, 4, v14 // (op 986) 


s_mul_i32 s1, s24, s9 // (op 988) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 988) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 988) least significant
s_mul_hi_u32 s7, s24, s8 // (op 988) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 988) most significant: sum
s_add_u32 s1, s1, s7 // (op 988) most significant: sum


s_mov_b32 s52, s26 // (op 988) reg expression
s_mov_b32 s53, s27 // (op 988) reg expression
s_mov_b32 s54, s0 // (op 988) reg expression
s_mov_b32 s55, 131072 // (op 988) call()

v_and_b32 v14, -64, v1 // (op 1031) 
v_sub_u32 v15, v1, v14 // (op 1031) 

v_lshl_add_u32 v14, s16, 6, v15 // (op 1031) 

v_lshrrev_b32 v15, 6, v1 // (op 1031) 
v_lshlrev_b32 v18, 1, v15 // (op 1031) 

v_mul_lo_u32 v15, v18, s28 // (op 1031) 

v_add_u32 v18, v14, v15 // (op 1031) 


s_mul_i32 s1, s28, s33 // (op 1033) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1033) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1033) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1033) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1033) most significant: sum
s_add_u32 s1, s1, s7 // (op 1033) most significant: sum


s_mov_b32 s56, s30 // (op 1033) reg expression
s_mov_b32 s57, s31 // (op 1033) reg expression
s_mov_b32 s58, s0 // (op 1033) reg expression
s_mov_b32 s59, 131072 // (op 1033) call()

v_and_b32 v14, -64, v1 // (op 1050) 
v_sub_u32 v15, v1, v14 // (op 1050) 

v_lshl_add_u32 v14, s16, 6, v15 // (op 1050) 

v_lshrrev_b32 v15, 6, v1 // (op 1050) 
v_lshlrev_b32 v19, 1, v15 // (op 1050) 

v_mul_lo_u32 v15, v19, s28 // (op 1050) 

v_add_u32 v19, v14, v15 // (op 1050) 


s_mul_i32 s1, s28, s33 // (op 1052) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1052) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1052) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1052) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1052) most significant: sum
s_add_u32 s1, s1, s7 // (op 1052) most significant: sum


s_mov_b32 s56, s30 // (op 1052) reg expression
s_mov_b32 s57, s31 // (op 1052) reg expression
s_mov_b32 s58, s0 // (op 1052) reg expression
s_mov_b32 s59, 131072 // (op 1052) call()

v_and_b32 v14, 15, v1 // (op 1095) 
v_lshl_add_u32 v15, s17, 4, v14 // (op 1095) 

v_mul_lo_u32 v14, v15, s34 // (op 1095) 

v_and_b32 v15, 63, v1 // (op 1095) 
v_lshrrev_b32 v20, 4, v15 // (op 1095) 

v_add_u32 v15, v20, v14 // (op 1095) 


s_mul_i32 s1, s34, s9 // (op 1097) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1097) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1097) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1097) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1097) most significant: sum
s_add_u32 s1, s1, s7 // (op 1097) most significant: sum


s_mov_b32 s60, s36 // (op 1097) reg expression
s_mov_b32 s61, s37 // (op 1097) reg expression
s_mov_b32 s62, s0 // (op 1097) reg expression
s_mov_b32 s63, 131072 // (op 1097) call()

v_and_b32 v14, 15, v1 // (op 1114) 
v_lshl_add_u32 v20, s17, 4, v14 // (op 1114) 

v_mul_lo_u32 v14, v20, s34 // (op 1114) 

v_and_b32 v20, 63, v1 // (op 1114) 
v_lshrrev_b32 v21, 4, v20 // (op 1114) 

v_add_u32 v20, v21, v14 // (op 1114) 


s_mul_i32 s1, s34, s9 // (op 1116) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1116) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1116) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1116) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1116) most significant: sum
s_add_u32 s1, s1, s7 // (op 1116) most significant: sum


s_mov_b32 s60, s36 // (op 1116) reg expression
s_mov_b32 s61, s37 // (op 1116) reg expression
s_mov_b32 s62, s0 // (op 1116) reg expression
s_mov_b32 s63, 131072 // (op 1116) call()




buffer_load_ubyte v14, v15, s[60:63], 0 offen offset: 0 // (op 774) Load value



buffer_load_dwordx4 v[22:25], v16, s[52:55], 0 offen // (op 778) Load value

s_lshl_b64 s[0:1], s[12:13], 2 // (op 781) 
v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 781) 


v_mov_b32 v21, v12 // (op 781) reg expression
buffer_load_dwordx4 v[28:31], v21, s[48:51], 0 offen // (op 781) Load value
v_add_u32 v21, v21, v26 // (op 781) 
buffer_load_dwordx4 v[32:35], v21, s[48:51], 0 offen // (op 781) Load value



v_mov_b64 v[26:27], s[28:29] // (op 784) convert

v_mov_b32 v21, v18 // (op 784) reg expression
buffer_load_ubyte v36, v21, s[56:59], 0 offen offset: 0 // (op 784) Load value
v_add_u32 v21, v21, v26 // (op 784) 
buffer_load_ubyte v37, v21, s[56:59], 0 offen offset: 0 // (op 784) Load value





buffer_load_ubyte v21, v15, s[60:63], 0 offen offset:4 // (op 788) Load value




buffer_load_dwordx4 v[38:41], v16, s[52:55], 0 offen offset:128 // (op 792) Load value


s_lshl_b64 s[0:1], s[12:13], 2 // (op 795) 
v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 795) 


s_lshl_b64 s[0:1], s[12:13], 7 // (op 795) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 795) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 795) 

v_add_u32 v15, v12, s0 // (op 795) 

buffer_load_dwordx4 v[42:45], v15, s[48:51], 0 offen // (op 795) Load value
v_add_u32 v15, v15, v26 // (op 795) 
buffer_load_dwordx4 v[46:49], v15, s[48:51], 0 offen // (op 795) Load value




v_mov_b64 v[26:27], s[28:29] // (op 798) convert

s_lshl_b32 s0, s28, 2 // (op 798) 
v_add_u32 v12, v18, s0 // (op 798) 

buffer_load_ubyte v50, v12, s[56:59], 0 offen offset: 0 // (op 798) Load value
v_add_u32 v12, v12, v26 // (op 798) 
buffer_load_ubyte v51, v12, s[56:59], 0 offen offset: 0 // (op 798) Load value



v_and_b32 v12, -8, v1 // (op 1239) 
v_sub_u32 v18, v1, v12 // (op 1239) 

v_lshlrev_b32 v12, 4, v18 // (op 1239) 

v_lshrrev_b32 v18, 3, v1 // (op 1239) 
v_lshl_add_u32 v15, v18, 7, v12 // (op 1239) 





s_waitcnt vmcnt(10)
ds_write_b128 v15, v[22:25] // (op 800) Store local data 

v_and_b32 v12, -2, v1 // (op 1181) 
v_sub_u32 v18, v1, v12 // (op 1181) 

v_lshlrev_b32 v12, 5, v18 // (op 1181) 

v_lshrrev_b32 v18, 1, v1 // (op 1181) 
v_lshl_add_u32 v15, v18, 7, v12 // (op 1181) 


v_lshlrev_b32 v12, 2, v15 // (op 1181) 

v_lshrrev_b32 v18, 3, v12 // (op 1181) 




s_waitcnt vmcnt(9)
ds_write_b128 v18, v[28:31] offset:2048 // (op 801) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v18, v[32:35] offset:2080 // (op 801) Store local data 

v_and_b32 v12, -64, v1 // (op 1292) 
v_sub_u32 v18, v1, v12 // (op 1292) 

v_lshrrev_b32 v12, 6, v1 // (op 1292) 
v_lshl_add_u32 v15, v12, 7, v18 // (op 1292) 





s_waitcnt vmcnt(7)
ds_write_b8 v15, v36 offset:6144 // (op 802) Store local data 
s_waitcnt vmcnt(6)
ds_write_b8 v15, v37 offset:6208 // (op 802) Store local data 

v_and_b32 v12, -2, v1 // (op 1152) 
v_sub_u32 v18, v1, v12 // (op 1152) 

v_lshlrev_b32 v12, 5, v18 // (op 1152) 

v_lshrrev_b32 v18, 1, v1 // (op 1152) 
v_lshl_add_u32 v15, v18, 7, v12 // (op 1152) 


v_lshlrev_b32 v12, 2, v15 // (op 1152) 

v_lshrrev_b32 v18, 3, v12 // (op 1152) 

v_and_b32 v12, -8, v1 // (op 1210) 
v_sub_u32 v15, v1, v12 // (op 1210) 

v_lshlrev_b32 v12, 4, v15 // (op 1210) 

v_lshrrev_b32 v15, 3, v1 // (op 1210) 
v_lshl_add_u32 v16, v15, 7, v12 // (op 1210) 


v_and_b32 v12, -64, v1 // (op 1263) 
v_sub_u32 v15, v1, v12 // (op 1263) 

v_lshrrev_b32 v12, 6, v1 // (op 1263) 
v_lshl_add_u32 v26, v12, 7, v15 // (op 1263) 


v_and_b32 v12, -2, v1 // (op 1339) 
v_sub_u32 v15, v1, v12 // (op 1339) 

v_lshlrev_b32 v12, 5, v15 // (op 1339) 

v_lshrrev_b32 v15, 1, v1 // (op 1339) 
v_lshl_add_u32 v27, v15, 7, v12 // (op 1339) 


v_lshlrev_b32 v12, 2, v27 // (op 1339) 

v_lshrrev_b32 v15, 3, v12 // (op 1339) 

v_and_b32 v12, -8, v1 // (op 1368) 
v_sub_u32 v27, v1, v12 // (op 1368) 

v_lshlrev_b32 v12, 4, v27 // (op 1368) 

v_lshrrev_b32 v27, 3, v1 // (op 1368) 
v_lshl_add_u32 v52, v27, 7, v12 // (op 1368) 


v_and_b32 v12, -64, v1 // (op 1392) 
v_sub_u32 v27, v1, v12 // (op 1392) 

v_lshrrev_b32 v12, 6, v1 // (op 1392) 
v_lshl_add_u32 v53, v12, 7, v27 // (op 1392) 


s_mov_b32 s0, 0 // (op 311) call()
s_ashr_i64 s[6:7], s[22:23], 7 // (op 310) 
s_mov_b32 s1, s6 // (op 310) convert

s_and_b32 s6, s1, 4294967294 // (op 310) 

s_cmp_lt_u32 s0, s6 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_310 // (op 310) Condition: Top (jump to RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_310 if false)
RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_310:
s_add_u32 s0, s0, 2 // (op 312) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1450) 



ds_read_b64_tr_b4 v[54:55], v6 offset:2048 // (op 65) Transpose load local data 
ds_read_b64_tr_b4 v[56:57], v6 offset:2560 // (op 65) Transpose load local data 
s_barrier  // (op 1463) 



ds_read_b128 v[58:61], v4 // (op 70) Load local data 
ds_read_b128 v[62:65], v4 offset:64 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1477) 


ds_read_u8 v12, v0 offset:6144 // (op 75) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:57], v[58:65], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 82) 
ds_read_b64_tr_b4 v[66:67], v6 offset:2064 // (op 340) Transpose load local data 


ds_read_b64_tr_b4 v[68:69], v6 offset:2576 // (op 340) Transpose load local data 


ds_read_u8 v12, v0 offset:6176 // (op 344) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[58:65], a[4:7], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 346) 





v_mov_b32 v12, v20 // (op 80) 

buffer_load_ubyte v14, v12, s[60:63], 0 offen offset:8 // (op 80) Load value


v_mov_b32 v27, v17 // (op 10) 

buffer_load_dwordx4 v[22:25], v27, s[52:55], 0 offen offset:256 // (op 10) Load value

s_lshl_b64 s[6:7], s[12:13], 2 // (op 4) 
v_lshrrev_b64 v[54:55], 3, s[6:7] // (op 4) 


v_mov_b32 v56, v13 // (op 4) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 4) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 4) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 4) 

s_lshl_b32 s1, s6, 1 // (op 4) 

v_add_u32 v57, v56, s1 // (op 4) 

buffer_load_dwordx4 v[28:31], v57, s[48:51], 0 offen // (op 4) Load value
v_add_u32 v57, v57, v54 // (op 4) 
buffer_load_dwordx4 v[32:35], v57, s[48:51], 0 offen // (op 4) Load value



v_mov_b64 v[54:55], s[28:29] // (op 17) convert

v_mov_b32 v57, v19 // (op 17) 
s_lshl_b32 s1, s28, 3 // (op 17) 
v_add_u32 v58, v57, s1 // (op 17) 

buffer_load_ubyte v36, v58, s[56:59], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v58, v58, v54 // (op 17) 
buffer_load_ubyte v37, v58, s[56:59], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1500) 



s_waitcnt vmcnt(10)
ds_write_b128 v52, v[38:41] offset:6400 // (op 235) Store local data 
s_barrier  // (op 1494) 



s_waitcnt vmcnt(9)
ds_write_b128 v15, v[42:45] offset:8448 // (op 225) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v15, v[46:49] offset:8480 // (op 225) Store local data 
s_barrier  // (op 1506) 



s_waitcnt vmcnt(7)
ds_write_b8 v53, v50 offset:12544 // (op 236) Store local data 
s_waitcnt vmcnt(6)
ds_write_b8 v53, v51 offset:12608 // (op 236) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1491) 



ds_read_b64_tr_b4 v[58:59], v9 offset:8448 // (op 227) Transpose load local data 
ds_read_b64_tr_b4 v[60:61], v9 offset:8960 // (op 227) Transpose load local data 
s_barrier  // (op 1497) 



ds_read_b128 v[62:65], v5 offset:6400 // (op 229) Load local data 
ds_read_b128 v[66:69], v5 offset:6464 // (op 229) Load local data 
s_barrier  // (op 1543) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1503) 


ds_read_u8 v54, v7 offset:12544 // (op 231) Load local data 



ds_read_b64_tr_b4 v[70:71], v9 offset:8464 // (op 353) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v9 offset:8976 // (op 353) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:61], v[62:69], a[0:3], v54, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 234) 




s_barrier  // (op 1577) 
ds_read_u8 v54, v7 offset:12576 // (op 357) Load local data 
s_barrier  // (op 1581) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[62:69], a[4:7], v54, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 347) 






buffer_load_ubyte v21, v12, s[60:63], 0 offen offset:12 // (op 233) Load value




v_add_u32 v20, 8, v20 // (op 1106) 
buffer_load_dwordx4 v[38:41], v27, s[52:55], 0 offen offset:384 // (op 237) Load value


v_add_u32 v17, 256, v17 // (op 978) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 224) 
v_lshrrev_b64 v[54:55], 3, s[6:7] // (op 224) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 224) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 224) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 224) 

s_mul_i32 s1, 3, s6 // (op 224) 

v_add_u32 v12, v56, s1 // (op 224) 

buffer_load_dwordx4 v[42:45], v12, s[48:51], 0 offen // (op 224) Load value
v_add_u32 v12, v12, v54 // (op 224) 
buffer_load_dwordx4 v[46:49], v12, s[48:51], 0 offen // (op 224) Load value




v_mov_b64 v[54:55], s[28:29] // (op 238) convert

s_lshl_b32 s1, s28, 2 // (op 238) 
s_mul_i32 s6, 3, s1 // (op 238) 

v_add_u32 v12, v57, s6 // (op 238) 

buffer_load_ubyte v50, v12, s[56:59], 0 offen offset: 0 // (op 238) Load value
v_add_u32 v12, v12, v54 // (op 238) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 914) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 914) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 914) 

s_lshl_b32 s1, s6, 1 // (op 914) 

v_add_u32 v13, v13, s1 // (op 914) 

buffer_load_ubyte v51, v12, s[56:59], 0 offen offset: 0 // (op 238) Load value



s_barrier  // (op 1466) 
s_lshl_b32 s1, s28, 3 // (op 1042) 



v_add_u32 v19, v19, s1 // (op 1042) 

s_waitcnt vmcnt(10)
ds_write_b128 v16, v[22:25] // (op 13) Store local data 
s_barrier  // (op 1453) 



s_waitcnt vmcnt(9)
ds_write_b128 v18, v[28:31] offset:2048 // (op 6) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v18, v[32:35] offset:2080 // (op 6) Store local data 
s_barrier  // (op 1480) 



s_waitcnt vmcnt(7)
ds_write_b8 v26, v36 offset:6144 // (op 21) Store local data 
s_waitcnt vmcnt(6)
ds_write_b8 v26, v37 offset:6208 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[22:23], 7 // (op 310) 
s_mov_b32 s1, s6 // (op 310) convert

s_and_b32 s6, s1, 4294967294 // (op 310) 

s_cmp_lt_u32 s0, s6 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_310 // (op 310) Condition: Bottom (jump to RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_310 if true)
RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_310:















v_and_b32 v12, -2, v1 // (op 876) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 876) 
v_sub_u32 v13, v1, v12 // (op 876) 

s_lshr_b32 s6, s0, 1 // (op 876) 

v_lshlrev_b32 v12, 5, v13 // (op 876) 

v_lshl_add_u32 v13, s16, 6, v12 // (op 876) 

v_mov_b32 v15, 0 // (op 876) convert
v_mov_b32 v14, v13 // (op 876) convert

v_and_b32 v12, 4294967294, v1 // (op 876) 
v_lshl_add_u32 v13, s6, 8, v12 // (op 876) 


v_mul_lo_u32 v17, v13, s13 // (op 876) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 876) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 876) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 876) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 876) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 876) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 876) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 876) 

v_mov_b32 v12, v14 // (op 876) convert

s_mul_i32 s1, s12, s23 // (op 878) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 878) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 878) least significant
s_mul_hi_u32 s7, s12, s22 // (op 878) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 878) most significant: sum
s_add_u32 s1, s1, s7 // (op 878) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 878) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 878) 

s_mov_b32 s48, s20 // (op 878) reg expression
s_mov_b32 s49, s21 // (op 878) reg expression
s_mov_b32 s50, s0 // (op 878) reg expression
s_mov_b32 s51, 131072 // (op 878) call()

v_and_b32 v13, -8, v1 // (op 944) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 944) 
v_sub_u32 v14, v1, v13 // (op 944) 

s_lshr_b32 s6, s0, 1 // (op 944) 

v_lshlrev_b32 v13, 4, v14 // (op 944) 

v_lshl_add_u32 v14, s6, 8, v13 // (op 944) 


v_lshrrev_b32 v13, 3, v1 // (op 944) 
v_lshl_add_u32 v15, s17, 4, v13 // (op 944) 

v_mul_lo_u32 v13, v15, s24 // (op 944) 

v_add_u32 v15, v14, v13 // (op 944) 


s_mul_i32 s1, s24, s9 // (op 946) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 946) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 946) least significant
s_mul_hi_u32 s7, s24, s8 // (op 946) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 946) most significant: sum
s_add_u32 s1, s1, s7 // (op 946) most significant: sum


s_mov_b32 s52, s26 // (op 946) reg expression
s_mov_b32 s53, s27 // (op 946) reg expression
s_mov_b32 s54, s0 // (op 946) reg expression
s_mov_b32 s55, 131072 // (op 946) call()

v_and_b32 v13, -64, v1 // (op 1008) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1008) 
v_sub_u32 v14, v1, v13 // (op 1008) 

s_lshr_b32 s6, s0, 1 // (op 1008) 

v_lshl_add_u32 v13, s16, 6, v14 // (op 1008) 

v_lshrrev_b32 v14, 6, v1 // (op 1008) 
v_lshlrev_b32 v16, 1, v14 // (op 1008) 

v_lshl_add_u32 v14, s6, 3, v16 // (op 1008) 


v_mul_lo_u32 v16, v14, s28 // (op 1008) 

v_add_u32 v14, v13, v16 // (op 1008) 


s_mul_i32 s1, s28, s33 // (op 1010) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1010) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1010) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1010) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1010) most significant: sum
s_add_u32 s1, s1, s7 // (op 1010) most significant: sum


s_mov_b32 s56, s30 // (op 1010) reg expression
s_mov_b32 s57, s31 // (op 1010) reg expression
s_mov_b32 s58, s0 // (op 1010) reg expression
s_mov_b32 s59, 131072 // (op 1010) call()

v_and_b32 v13, 63, v1 // (op 1072) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1072) 
v_lshrrev_b32 v16, 4, v13 // (op 1072) 

s_lshr_b32 s6, s0, 1 // (op 1072) 

v_and_b32 v13, 15, v1 // (op 1072) 
v_lshl_add_u32 v17, s6, 3, v16 // (op 1072) 


v_lshl_add_u32 v16, s17, 4, v13 // (op 1072) 

v_mul_lo_u32 v13, v16, s34 // (op 1072) 

v_add_u32 v16, v17, v13 // (op 1072) 


s_mul_i32 s1, s34, s9 // (op 1074) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1074) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1074) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1074) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1074) most significant: sum
s_add_u32 s1, s1, s7 // (op 1074) most significant: sum


s_mov_b32 s60, s36 // (op 1074) reg expression
s_mov_b32 s61, s37 // (op 1074) reg expression
s_mov_b32 s62, s0 // (op 1074) reg expression
s_mov_b32 s63, 131072 // (op 1074) call()

v_and_b32 v13, -2, v1 // (op 1165) 
v_sub_u32 v17, v1, v13 // (op 1165) 

v_lshlrev_b32 v13, 5, v17 // (op 1165) 

v_lshrrev_b32 v17, 1, v1 // (op 1165) 
v_lshl_add_u32 v18, v17, 7, v13 // (op 1165) 


v_lshlrev_b32 v13, 2, v18 // (op 1165) 

v_lshrrev_b32 v17, 3, v13 // (op 1165) 

v_and_b32 v13, -8, v1 // (op 1223) 
v_sub_u32 v18, v1, v13 // (op 1223) 

v_lshlrev_b32 v13, 4, v18 // (op 1223) 

v_lshrrev_b32 v18, 3, v1 // (op 1223) 
v_lshl_add_u32 v19, v18, 7, v13 // (op 1223) 


v_and_b32 v13, -64, v1 // (op 1276) 
v_sub_u32 v18, v1, v13 // (op 1276) 

v_lshrrev_b32 v13, 6, v1 // (op 1276) 
v_lshl_add_u32 v20, v13, 7, v18 // (op 1276) 


s_ashr_i64 s[0:1], s[22:23], 7 // (op 316) 
s_and_b32 s6, s0, 4294967294 // (op 316) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 315) 
s_mov_b32 s7, s0 // (op 315) convert

s_cmp_lt_u32 s6, s7 // (op 315) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_315 // (op 315) Condition: Top (jump to RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_315 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_315:

s_add_u32 s6, s6, 1 // (op 317) 
s_lshl_b64 s[0:1], s[12:13], 2 // (op 166) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 166) 


v_mov_b32 v13, v12 // (op 166) 
v_mov_b32 v18, v13 // (op 166) reg expression
buffer_load_dwordx4 v[24:27], v18, s[48:51], 0 offen // (op 166) Load value
v_add_u32 v18, v18, v22 // (op 166) 
buffer_load_dwordx4 v[28:31], v18, s[48:51], 0 offen // (op 166) Load value



s_lshl_b64 s[0:1], s[12:13], 7 // (op 866) 
s_lshl_b64 s[10:11], s[0:1], 2 // (op 866) 

s_lshr_b64 s[0:1], s[10:11], 3 // (op 866) 

v_add_u32 v12, v12, s0 // (op 866) 



s_barrier  // (op 1456) 
v_mov_b32 v13, v15 // (op 179) 




buffer_load_dwordx4 v[32:35], v13, s[52:55], 0 offen // (op 179) Load value

v_add_u32 v15, 128, v15 // (op 936) 
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[24:27] offset:2048 // (op 167) Store local data 

v_mov_b64 v[22:23], s[28:29] // (op 180) convert

v_mov_b32 v13, v14 // (op 180) 
v_mov_b32 v18, v13 // (op 180) reg expression
buffer_load_ubyte v36, v18, s[56:59], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v18, v18, v22 // (op 180) 
buffer_load_ubyte v37, v18, s[56:59], 0 offen offset: 0 // (op 180) Load value



s_lshl_b32 s7, s28, 2 // (op 1000) 
v_add_u32 v14, v14, s7 // (op 1000) 

s_waitcnt vmcnt(3)
ds_write_b128 v17, v[28:31] offset:2080 // (op 167) Store local data 

s_barrier  // (op 1469) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1459) 






ds_read_b64_tr_b4 v[28:29], v6 offset:2048 // (op 169) Transpose load local data 
ds_read_b64_tr_b4 v[30:31], v6 offset:2560 // (op 169) Transpose load local data 
s_waitcnt vmcnt(2)
ds_write_b128 v19, v[32:35] // (op 177) Store local data 

s_barrier  // (op 1483) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1473) 






ds_read_b128 v[38:41], v4 // (op 171) Load local data 
ds_read_b128 v[42:45], v4 offset:64 // (op 171) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v36 offset:6144 // (op 178) Store local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v37 offset:6208 // (op 178) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1487) 


ds_read_u8 v13, v0 offset:6144 // (op 173) Load local data 


v_mov_b32 v18, v16 // (op 175) 

buffer_load_ubyte v21, v18, s[60:63], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v16, 4, v16 // (op 1064) 



ds_read_b64_tr_b4 v[22:23], v6 offset:2064 // (op 321) Transpose load local data 
ds_read_b64_tr_b4 v[24:25], v6 offset:2576 // (op 321) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:31], v[38:45], a[0:3], v13, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 176) 




ds_read_u8 v13, v0 offset:6176 // (op 325) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:25], v[38:45], a[4:7], v13, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 327) 




s_ashr_i64 s[0:1], s[22:23], 7 // (op 315) 
s_mov_b32 s7, s0 // (op 315) convert

s_cmp_lt_u32 s6, s7 // (op 315) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_315 // (op 315) Condition: Bottom (jump to RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_315 if true)
RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_315:






















s_barrier  // (op 1565) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v4, a0 // (op 46) 
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v12, v11, v4 // (op 46) 
v_mul_f32 v13, v11, v5 // (op 46) 
v_mul_f32 v14, v11, v6 // (op 46) 
v_mul_f32 v15, v11, v7 // (op 46) 




buffer_store_dwordx4 v[12:15], v8, s[44:47], 0 offen // (op 52) Store value

v_accvgpr_read v4, a4 // (op 513) 
v_accvgpr_read v5, a5 // (op 513) 
v_accvgpr_read v6, a6 // (op 513) 
v_accvgpr_read v7, a7 // (op 513) 
v_mul_f32 v12, v11, v4 // (op 513) 
v_mul_f32 v13, v11, v5 // (op 513) 
v_mul_f32 v14, v11, v6 // (op 513) 
v_mul_f32 v15, v11, v7 // (op 513) 




buffer_store_dwordx4 v[12:15], v8, s[44:47], 0 offen offset:128 // (op 515) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen offset:128 // (op 543) Load value
v_accvgpr_read v12, a4 // (op 540) 
v_accvgpr_read v13, a5 // (op 540) 
v_accvgpr_read v14, a6 // (op 540) 
v_accvgpr_read v15, a7 // (op 540) 
v_mul_f32 v16, v11, v12 // (op 540) 
v_mul_f32 v17, v11, v13 // (op 540) 
v_mul_f32 v18, v11, v14 // (op 540) 
v_mul_f32 v19, v11, v15 // (op 540) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v10, v4 // (op 544) 
v_mul_f32 v13, v10, v5 // (op 544) 
v_mul_f32 v14, v10, v6 // (op 544) 
v_mul_f32 v15, v10, v7 // (op 544) 

v_add_f32 v4, v12, v16 // (op 541) 
v_add_f32 v5, v13, v17 // (op 541) 
v_add_f32 v6, v14, v18 // (op 541) 
v_add_f32 v7, v15, v19 // (op 541) 





buffer_store_dwordx4 v[4:7], v8, s[44:47], 0 offen offset:128 // (op 542) Store value




s_nop 1
buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 135) Load value
v_accvgpr_read v12, a0 // (op 132) 
v_accvgpr_read v13, a1 // (op 132) 
v_accvgpr_read v14, a2 // (op 132) 
v_accvgpr_read v15, a3 // (op 132) 
v_mul_f32 v16, v11, v12 // (op 132) 
v_mul_f32 v17, v11, v13 // (op 132) 
v_mul_f32 v18, v11, v14 // (op 132) 
v_mul_f32 v19, v11, v15 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v10, v4 // (op 136) 
v_mul_f32 v13, v10, v5 // (op 136) 
v_mul_f32 v14, v10, v6 // (op 136) 
v_mul_f32 v15, v10, v7 // (op 136) 

v_add_f32 v4, v12, v16 // (op 133) 
v_add_f32 v5, v13, v17 // (op 133) 
v_add_f32 v6, v14, v18 // (op 133) 
v_add_f32 v7, v15, v19 // (op 133) 





buffer_store_dwordx4 v[4:7], v8, s[44:47], 0 offen // (op 134) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.LRR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_end:
.size RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2, .LRR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_end-RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
  .amdhsa_next_free_vgpr 84
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12800 // lds bytes
  .amdhsa_accum_offset 76
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
    DataTypeB: B8
    DestDataType: s
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
    macrotile: [64, 16, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
    .symbol: RR_GEMM_NN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 12800
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 74
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

