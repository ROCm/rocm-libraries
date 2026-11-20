// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2,@function
RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 1192) 
v_and_b32 v3, 48, v1 // (op 1192) 
v_sub_u32 v4, v0, v3 // (op 1192) 

v_lshrrev_b32 v3, 4, v0 // (op 1192) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1192) 


v_lshrrev_b32 v3, 7, v1 // (op 1192) 
v_lshlrev_b32 v4, 4, v3 // (op 1192) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1192) 


v_lshlrev_b32 v0, 2, v3 // (op 1192) 

v_lshrrev_b32 v3, 3, v0 // (op 1192) 

v_and_b32 v0, 63, v1 // (op 1350) 
v_and_b32 v4, 48, v1 // (op 1350) 
v_sub_u32 v5, v0, v4 // (op 1350) 

v_lshrrev_b32 v4, 4, v0 // (op 1350) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 1350) 


v_lshrrev_b32 v4, 7, v1 // (op 1350) 
v_lshlrev_b32 v5, 4, v4 // (op 1350) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1350) 


v_lshlrev_b32 v0, 2, v4 // (op 1350) 

v_lshrrev_b32 v4, 3, v0 // (op 1350) 

v_lshrrev_b32 v0, 7, v1 // (op 1135) 
v_and_b32 v5, 63, v1 // (op 1135) 
v_lshrrev_b32 v6, 4, v5 // (op 1135) 
v_lshlrev_b32 v7, 2, v6 // (op 1135) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1135) 


v_lshl_add_u32 v0, s16, 4, v6 // (op 1135) 

v_lshrrev_b32 v6, 2, v5 // (op 1135) 

v_and_b32 v5, 3, v6 // (op 1135) 

v_and_b32 v6, 3, v1 // (op 1135) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1135) 


v_lshrrev_b32 v5, 6, v1 // (op 1135) 
v_and_b32 v6, 1, v5 // (op 1135) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1135) 


v_lshl_add_u32 v6, s17, 6, v5 // (op 1135) 

v_mul_lo_u32 v5, v6, s2 // (op 1135) 

v_add_lshl_u32 v6, v0, v5, 1 // (op 1135) 


v_and_b32 v0, 48, v1 // (op 1250) 
v_and_b32 v5, 63, v1 // (op 1250) 
v_sub_u32 v7, v5, v0 // (op 1250) 

v_lshrrev_b32 v5, 1, v7 // (op 1250) 
s_mul_i32 s1, s2, s9 // (op 1137) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1137) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1137) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1137) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1137) most significant: sum
s_add_u32 s1, s1, s19 // (op 1137) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1137) 

v_add_lshl_u32 v8, v0, v5, 6 // (op 1250) 


v_and_b32 v0, 1, v7 // (op 1250) 

v_lshlrev_b32 v5, 3, v0 // (op 1250) 

v_lshrrev_b32 v0, 6, v1 // (op 1250) 
v_and_b32 v7, 1, v0 // (op 1250) 

v_lshl_add_u32 v0, v7, 4, v5 // (op 1250) 


v_add_u32 v5, v8, v0 // (op 1250) 


v_lshrrev_b32 v0, 6, v1 // (op 1308) 
v_and_b32 v7, 1, v0 // (op 1308) 

v_and_b32 v0, 15, v1 // (op 1308) 
v_lshl_add_u32 v8, v7, 4, v0 // (op 1308) 


v_and_b32 v0, 63, v1 // (op 1308) 
v_lshrrev_b32 v7, 4, v0 // (op 1308) 

v_lshl_add_u32 v0, v7, 6, v8 // (op 1308) 


v_lshrrev_b32 v7, 7, v1 // (op 1324) 
v_and_b32 v8, 63, v1 // (op 1324) 
v_lshrrev_b32 v9, 4, v8 // (op 1324) 
v_lshlrev_b32 v10, 2, v9 // (op 1324) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1324) 


v_lshl_add_u32 v7, s16, 4, v9 // (op 1324) 

v_lshrrev_b32 v9, 2, v8 // (op 1324) 

v_and_b32 v8, 3, v9 // (op 1324) 

v_and_b32 v9, 3, v1 // (op 1324) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1324) 


v_lshrrev_b32 v8, 6, v1 // (op 1324) 
v_and_b32 v9, 1, v8 // (op 1324) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 1324) 


v_lshl_add_u32 v9, s17, 6, v8 // (op 1324) 

v_mul_lo_u32 v8, v9, s4 // (op 1324) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 1324) 


v_and_b32 v7, 48, v1 // (op 1379) 
v_and_b32 v8, 63, v1 // (op 1379) 
v_sub_u32 v10, v8, v7 // (op 1379) 

v_lshrrev_b32 v8, 1, v10 // (op 1379) 
s_mul_i32 s1, s4, s9 // (op 1326) most significant: low of low * high
v_add_lshl_u32 v11, v7, v8, 6 // (op 1379) 


v_and_b32 v7, 1, v10 // (op 1379) 

v_lshlrev_b32 v8, 3, v7 // (op 1379) 

v_lshrrev_b32 v7, 6, v1 // (op 1379) 
v_and_b32 v10, 1, v7 // (op 1379) 

v_lshl_add_u32 v7, v10, 4, v8 // (op 1379) 


v_add_u32 v8, v11, v7 // (op 1379) 


v_lshrrev_b32 v7, 6, v1 // (op 1408) 
v_and_b32 v10, 1, v7 // (op 1408) 

v_and_b32 v7, 15, v1 // (op 1408) 
v_lshl_add_u32 v11, v10, 4, v7 // (op 1408) 


v_and_b32 v7, 63, v1 // (op 1408) 
v_lshrrev_b32 v10, 4, v7 // (op 1408) 

v_lshl_add_u32 v7, v10, 6, v11 // (op 1408) 


s_mul_i32 s38, s5, s8 // (op 1326) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1326) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1326) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1326) most significant: sum
s_add_u32 s1, s1, s39 // (op 1326) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 1326) 

s_mov_b32 s40, s6 // (op 1137) reg expression
s_mov_b32 s41, s7 // (op 1137) reg expression
s_mov_b32 s42, s18 // (op 1137) reg expression
s_mov_b32 s43, 131072 // (op 1137) call()

s_mov_b32 s44, s10 // (op 1326) reg expression

s_mov_b32 s45, s11 // (op 1326) reg expression
s_mov_b32 s46, s38 // (op 1326) reg expression
s_mov_b32 s47, 131072 // (op 1326) call()



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
s_lshl_b32 s0, s16, 4 // (op 902) 
v_mul_lo_u32 v13, v1, s13 // (op 902) most significant: low of low * high
s_mov_b32 s15, 0 // (op 902) convert
s_mov_b32 s14, s0 // (op 902) convert

v_mul_hi_u32 v14, v1, s12 // (op 902) most significant: high of low * low
v_mul_lo_u32 v12, v1, s12 // (op 902) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 902) most significant: sum

v_mov_b32 v14, s14 // (op 902) 
v_mov_b32 v15, s15 // (op 902) 
v_add_co_u32 v16, s[0:1], v14, v12 // (op 902) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 902) most significant half



v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 902) 


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 902) 

v_mov_b32 v14, v12 // (op 902) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 904) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 904) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 904) least significant
s_mul_hi_u32 s7, s12, s22 // (op 904) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 904) most significant: sum
s_add_u32 s1, s1, s7 // (op 904) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 904) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 904) 

s_mov_b32 s8, s20 // (op 904) reg expression
s_mov_b32 s9, s21 // (op 904) reg expression
s_mov_b32 s10, s0 // (op 904) reg expression
s_mov_b32 s11, 131072 // (op 904) call()

s_lshl_b32 s0, s16, 4 // (op 921) 
v_mul_lo_u32 v13, v1, s13 // (op 921) most significant: low of low * high
v_mul_hi_u32 v15, v1, s12 // (op 921) most significant: high of low * low
v_mul_lo_u32 v12, v1, s12 // (op 921) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 921) most significant: sum

s_mov_b32 s7, 0 // (op 921) convert
s_mov_b32 s6, s0 // (op 921) convert

v_mov_b32 v15, s6 // (op 921) 
v_mov_b32 v16, s7 // (op 921) 
v_add_co_u32 v18, s[0:1], v15, v12 // (op 921) least significant half
v_addc_co_u32 v19, s[0:1], v16, v13, s[0:1] // (op 921) most significant half



v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 921) 


v_lshrrev_b64 v[12:13], 3, v[18:19] // (op 921) 

v_mov_b32 v15, v12 // (op 921) convert

s_mul_i32 s1, s12, s23 // (op 923) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 923) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 923) least significant
s_mul_hi_u32 s7, s12, s22 // (op 923) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 923) most significant: sum
s_add_u32 s1, s1, s7 // (op 923) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 923) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 923) 

s_mov_b32 s8, s20 // (op 923) reg expression
s_mov_b32 s9, s21 // (op 923) reg expression
s_mov_b32 s10, s0 // (op 923) reg expression
s_mov_b32 s11, 131072 // (op 923) call()

v_and_b32 v12, -4, v1 // (op 966) 
v_sub_u32 v13, v1, v12 // (op 966) 

v_lshlrev_b32 v12, 4, v13 // (op 966) 

v_lshl_add_u32 v13, s17, 6, v12 // (op 966) 

v_and_b32 v12, 4294967292, v1 // (op 966) 
v_mul_lo_u32 v16, v12, s24 // (op 966) 
v_add_u32 v16, v16, v13 // (op 966) 


s_mul_i32 s1, s24, s23 // (op 968) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 968) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 968) least significant
s_mul_hi_u32 s7, s24, s22 // (op 968) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 968) most significant: sum
s_add_u32 s1, s1, s7 // (op 968) most significant: sum


s_mov_b32 s48, s26 // (op 968) reg expression
s_mov_b32 s49, s27 // (op 968) reg expression
s_mov_b32 s50, s0 // (op 968) reg expression
s_mov_b32 s51, 131072 // (op 968) call()

v_and_b32 v12, -4, v1 // (op 985) 
v_sub_u32 v13, v1, v12 // (op 985) 

v_lshlrev_b32 v12, 4, v13 // (op 985) 

v_lshl_add_u32 v13, s17, 6, v12 // (op 985) 

v_and_b32 v12, 4294967292, v1 // (op 985) 
v_mul_lo_u32 v17, v12, s24 // (op 985) 
v_add_u32 v17, v17, v13 // (op 985) 


s_mul_i32 s1, s24, s23 // (op 987) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 987) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 987) least significant
s_mul_hi_u32 s7, s24, s22 // (op 987) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 987) most significant: sum
s_add_u32 s1, s1, s7 // (op 987) most significant: sum


s_mov_b32 s48, s26 // (op 987) reg expression
s_mov_b32 s49, s27 // (op 987) reg expression
s_mov_b32 s50, s0 // (op 987) reg expression
s_mov_b32 s51, 131072 // (op 987) call()

v_lshrrev_b32 v12, 7, v1 // (op 1030) 
v_and_b32 v13, 15, v1 // (op 1030) 
v_lshl_add_u32 v18, v12, 4, v13 // (op 1030) 


v_lshl_add_u32 v12, s16, 4, v18 // (op 1030) 

v_and_b32 v13, 63, v1 // (op 1030) 
v_lshrrev_b32 v18, 4, v13 // (op 1030) 

v_mul_lo_u32 v13, v18, s28 // (op 1030) 

v_add_u32 v18, v12, v13 // (op 1030) 


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

v_lshrrev_b32 v12, 7, v1 // (op 1049) 
v_and_b32 v13, 15, v1 // (op 1049) 
v_lshl_add_u32 v19, v12, 4, v13 // (op 1049) 


v_lshl_add_u32 v12, s16, 4, v19 // (op 1049) 

v_and_b32 v13, 63, v1 // (op 1049) 
v_lshrrev_b32 v19, 4, v13 // (op 1049) 

v_mul_lo_u32 v13, v19, s28 // (op 1049) 

v_add_u32 v19, v12, v13 // (op 1049) 


s_mul_i32 s1, s28, s33 // (op 1051) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1051) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1051) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1051) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1051) most significant: sum
s_add_u32 s1, s1, s7 // (op 1051) most significant: sum


s_mov_b32 s52, s30 // (op 1051) reg expression
s_mov_b32 s53, s31 // (op 1051) reg expression
s_mov_b32 s54, s0 // (op 1051) reg expression
s_mov_b32 s55, 131072 // (op 1051) call()

v_and_b32 v12, -64, v1 // (op 1094) 
v_sub_u32 v13, v1, v12 // (op 1094) 

v_lshl_add_u32 v12, s17, 6, v13 // (op 1094) 

v_lshrrev_b32 v13, 6, v1 // (op 1094) 
v_lshlrev_b32 v20, 1, v13 // (op 1094) 

v_mul_lo_u32 v13, v20, s34 // (op 1094) 
v_add_u32 v13, v13, v12 // (op 1094) 


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

v_and_b32 v12, -64, v1 // (op 1113) 
v_sub_u32 v20, v1, v12 // (op 1113) 

v_lshl_add_u32 v12, s17, 6, v20 // (op 1113) 

v_lshrrev_b32 v20, 6, v1 // (op 1113) 
v_lshlrev_b32 v21, 1, v20 // (op 1113) 

v_mul_lo_u32 v20, v21, s34 // (op 1113) 
v_add_u32 v20, v20, v12 // (op 1113) 


s_mul_i32 s1, s34, s33 // (op 1115) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1115) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1115) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1115) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1115) most significant: sum
s_add_u32 s1, s1, s7 // (op 1115) most significant: sum


s_mov_b32 s56, s36 // (op 1115) reg expression
s_mov_b32 s57, s37 // (op 1115) reg expression
s_mov_b32 s58, s0 // (op 1115) reg expression
s_mov_b32 s59, 131072 // (op 1115) call()


v_mov_b64 v[22:23], s[28:29] // (op 769) convert

v_mov_b32 v12, v18 // (op 769) reg expression
v_mov_b32 v21, v12 // (op 769) 
buffer_load_ubyte v24, v21, s[52:55], 0 offen offset: 0 // (op 769) Load value




v_mov_b64 v[22:23], s[24:25] // (op 773) convert

v_mov_b32 v12, v16 // (op 773) reg expression
buffer_load_dwordx4 v[26:29], v12, s[48:51], 0 offen // (op 773) Load value
v_add_u32 v12, v12, v22 // (op 773) 
buffer_load_dwordx4 v[30:33], v12, s[48:51], 0 offen // (op 773) Load value
v_add_u32 v12, v12, v22 // (op 773) 
buffer_load_dwordx4 v[34:37], v12, s[48:51], 0 offen // (op 773) Load value
v_add_u32 v12, v12, v22 // (op 773) 
buffer_load_dwordx4 v[38:41], v12, s[48:51], 0 offen // (op 773) Load value



v_mov_b64 v[22:23], s[34:35] // (op 776) convert

v_mov_b32 v12, v13 // (op 776) reg expression
buffer_load_ubyte v42, v12, s[56:59], 0 offen offset: 0 // (op 776) Load value
v_add_u32 v12, v12, v22 // (op 776) 
buffer_load_ubyte v43, v12, s[56:59], 0 offen offset: 0 // (op 776) Load value




v_mov_b32 v12, v14 // (op 779) reg expression

buffer_load_dwordx2 v[22:23], v12, s[8:11], 0 offen // (op 779) Load value


v_mov_b64 v[44:45], s[28:29] // (op 783) convert

s_lshl_b32 s0, s28, 2 // (op 783) 
v_add_u32 v12, v18, s0 // (op 783) 

v_mov_b32 v21, v12 // (op 783) 
buffer_load_ubyte v25, v21, s[52:55], 0 offen offset: 0 // (op 783) Load value





v_mov_b64 v[44:45], s[24:25] // (op 787) convert

s_lshl_b32 s0, s24, 7 // (op 787) 
v_add_u32 v12, v16, s0 // (op 787) 

buffer_load_dwordx4 v[46:49], v12, s[48:51], 0 offen // (op 787) Load value
v_add_u32 v12, v12, v44 // (op 787) 
buffer_load_dwordx4 v[50:53], v12, s[48:51], 0 offen // (op 787) Load value
v_add_u32 v12, v12, v44 // (op 787) 
buffer_load_dwordx4 v[54:57], v12, s[48:51], 0 offen // (op 787) Load value
v_add_u32 v12, v12, v44 // (op 787) 
buffer_load_dwordx4 v[58:61], v12, s[48:51], 0 offen // (op 787) Load value




v_mov_b64 v[44:45], s[34:35] // (op 790) convert

s_lshl_b32 s0, s34, 2 // (op 790) 
v_add_u32 v12, v13, s0 // (op 790) 

buffer_load_ubyte v62, v12, s[56:59], 0 offen offset: 0 // (op 790) Load value
v_add_u32 v12, v12, v44 // (op 790) 
buffer_load_ubyte v63, v12, s[56:59], 0 offen offset: 0 // (op 790) Load value





s_lshl_b64 s[0:1], s[12:13], 7 // (op 793) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 793) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 793) 

v_add_u32 v16, v14, s0 // (op 793) 


buffer_load_dwordx2 v[12:13], v16, s[8:11], 0 offen // (op 793) Load value


v_and_b32 v14, -4, v1 // (op 1233) 
v_sub_u32 v16, v1, v14 // (op 1233) 

v_lshlrev_b32 v14, 4, v16 // (op 1233) 

v_lshrrev_b32 v16, 2, v1 // (op 1233) 
v_lshl_add_u32 v18, v16, 8, v14 // (op 1233) 





s_waitcnt vmcnt(14)
ds_write_b128 v18, v[26:29] // (op 795) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v18, v[30:33] offset:64 // (op 795) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v18, v[34:37] offset:128 // (op 795) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v18, v[38:41] offset:192 // (op 795) Store local data 

v_and_b32 v14, -64, v1 // (op 1291) 
v_sub_u32 v16, v1, v14 // (op 1291) 

v_lshrrev_b32 v14, 6, v1 // (op 1291) 
v_lshl_add_u32 v18, v14, 7, v16 // (op 1291) 





s_waitcnt vmcnt(10)
ds_write_b8 v18, v42 offset:8192 // (op 796) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v18, v43 offset:8256 // (op 796) Store local data 

v_lshlrev_b32 v14, 6, v1 // (op 1180) 
v_lshrrev_b32 v16, 3, v14 // (op 1180) 




s_waitcnt vmcnt(8)
ds_write_b64 v16, v[22:23] offset:8448 // (op 797) Store local data 

v_lshlrev_b32 v14, 6, v1 // (op 1151) 
v_lshrrev_b32 v16, 3, v14 // (op 1151) 

v_and_b32 v14, -4, v1 // (op 1204) 
v_sub_u32 v18, v1, v14 // (op 1204) 

v_lshlrev_b32 v14, 4, v18 // (op 1204) 

v_lshrrev_b32 v18, 2, v1 // (op 1204) 
v_lshl_add_u32 v21, v18, 8, v14 // (op 1204) 


v_and_b32 v14, -64, v1 // (op 1262) 
v_sub_u32 v18, v1, v14 // (op 1262) 

v_lshrrev_b32 v14, 6, v1 // (op 1262) 
v_lshl_add_u32 v44, v14, 7, v18 // (op 1262) 


v_lshlrev_b32 v14, 6, v1 // (op 1338) 
v_lshrrev_b32 v18, 3, v14 // (op 1338) 

v_and_b32 v14, -4, v1 // (op 1362) 
v_sub_u32 v45, v1, v14 // (op 1362) 

v_lshlrev_b32 v14, 4, v45 // (op 1362) 

v_lshrrev_b32 v45, 2, v1 // (op 1362) 
v_lshl_add_u32 v64, v45, 8, v14 // (op 1362) 


v_and_b32 v14, -64, v1 // (op 1391) 
v_sub_u32 v45, v1, v14 // (op 1391) 

v_lshrrev_b32 v14, 6, v1 // (op 1391) 
v_lshl_add_u32 v65, v14, 7, v45 // (op 1391) 


s_mov_b32 s0, 0 // (op 312) call()
s_ashr_i64 s[6:7], s[22:23], 7 // (op 311) 
s_mov_b32 s1, s6 // (op 311) convert

s_and_b32 s6, s1, 4294967294 // (op 311) 

s_cmp_lt_u32 s0, s6 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_311 // (op 311) Condition: Top (jump to RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_311 if false)
RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_311:
s_add_u32 s0, s0, 2 // (op 313) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1449) 



ds_read_b64_tr_b4 v[66:67], v3 offset:8448 // (op 67) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v3 offset:8576 // (op 67) Transpose load local data 
s_barrier  // (op 1462) 



ds_read_b64_tr_b8 v[70:71], v5 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v5 offset:512 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v5 offset:4096 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v5 offset:4608 // (op 72) Transpose load local data 
s_barrier  // (op 1476) 


ds_read_u8 v14, v0 offset:8192 // (op 81) Load local data 



ds_read_b64_tr_b8 v[78:79], v5 offset:32 // (op 343) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v5 offset:544 // (op 343) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[70:77], a[0:3], v24, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 84) 
ds_read_b64_tr_b8 v[82:83], v5 offset:4128 // (op 343) Transpose load local data 


ds_read_b64_tr_b8 v[84:85], v5 offset:4640 // (op 343) Transpose load local data 


ds_read_u8 v14, v0 offset:8224 // (op 346) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[78:85], a[4:7], v24, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 347) 




v_mov_b64 v[66:67], s[28:29] // (op 77) convert

v_mov_b32 v14, v19 // (op 77) 
s_lshl_b32 s1, s28, 3 // (op 77) 
v_add_u32 v45, v14, s1 // (op 77) 

v_mov_b32 v68, v45 // (op 77) 
buffer_load_ubyte v24, v68, s[52:55], 0 offen offset: 0 // (op 77) Load value




v_mov_b64 v[66:67], s[24:25] // (op 10) convert

v_mov_b32 v45, v17 // (op 10) 
s_lshl_b32 s1, s24, 8 // (op 10) 
v_add_u32 v68, v45, s1 // (op 10) 

buffer_load_dwordx4 v[26:29], v68, s[48:51], 0 offen // (op 10) Load value
v_add_u32 v68, v68, v66 // (op 10) 
buffer_load_dwordx4 v[30:33], v68, s[48:51], 0 offen // (op 10) Load value
v_add_u32 v68, v68, v66 // (op 10) 
buffer_load_dwordx4 v[34:37], v68, s[48:51], 0 offen // (op 10) Load value
v_add_u32 v68, v68, v66 // (op 10) 
buffer_load_dwordx4 v[38:41], v68, s[48:51], 0 offen // (op 10) Load value



v_mov_b64 v[66:67], s[34:35] // (op 21) convert

v_mov_b32 v68, v20 // (op 21) 
s_lshl_b32 s1, s34, 3 // (op 21) 
v_add_u32 v69, v68, s1 // (op 21) 

buffer_load_ubyte v42, v69, s[56:59], 0 offen offset: 0 // (op 21) Load value
v_add_u32 v69, v69, v66 // (op 21) 
buffer_load_ubyte v43, v69, s[56:59], 0 offen offset: 0 // (op 21) Load value




v_mov_b32 v66, v15 // (op 4) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 4) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 4) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 4) 

s_lshl_b32 s1, s6, 1 // (op 4) 

v_add_u32 v67, v66, s1 // (op 4) 


buffer_load_dwordx2 v[22:23], v67, s[8:11], 0 offen // (op 4) Load value

s_barrier  // (op 1499) 



s_waitcnt vmcnt(14)
ds_write_b128 v64, v[46:49] offset:9472 // (op 239) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v64, v[50:53] offset:9536 // (op 239) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v64, v[54:57] offset:9600 // (op 239) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v64, v[58:61] offset:9664 // (op 239) Store local data 
s_barrier  // (op 1505) 



s_waitcnt vmcnt(10)
ds_write_b8 v65, v62 offset:17664 // (op 240) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v65, v63 offset:17728 // (op 240) Store local data 
s_barrier  // (op 1493) 



s_waitcnt vmcnt(8)
ds_write_b64 v18, v[12:13] offset:17920 // (op 229) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1490) 



ds_read_b64_tr_b4 v[70:71], v4 offset:17920 // (op 231) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v4 offset:18048 // (op 231) Transpose load local data 
s_barrier  // (op 1496) 
s_barrier  // (op 1546) 



ds_read_b64_tr_b8 v[74:75], v8 offset:9472 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v8 offset:9984 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v8 offset:13568 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v8 offset:14080 // (op 233) Transpose load local data 
s_barrier  // (op 1502) 


ds_read_u8 v67, v7 offset:17664 // (op 237) Load local data 



ds_read_b64_tr_b8 v[82:83], v8 offset:9504 // (op 352) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v8 offset:10016 // (op 352) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[74:81], a[0:3], v25, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 238) 
ds_read_b64_tr_b8 v[86:87], v8 offset:13600 // (op 352) Transpose load local data 


ds_read_b64_tr_b8 v[88:89], v8 offset:14112 // (op 352) Transpose load local data 


s_barrier  // (op 1580) 
ds_read_u8 v67, v7 offset:17696 // (op 355) Load local data 
s_barrier  // (op 1584) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[82:89], a[4:7], v25, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 348) 




v_mov_b64 v[70:71], s[28:29] // (op 235) convert

s_lshl_b32 s1, s28, 2 // (op 235) 
s_mul_i32 s6, 3, s1 // (op 235) 

v_add_u32 v67, v14, s6 // (op 235) 

v_mov_b32 v69, v67 // (op 235) 
buffer_load_ubyte v25, v69, s[52:55], 0 offen offset: 0 // (op 235) Load value





s_lshl_b32 s1, s28, 3 // (op 1041) 
v_add_u32 v19, v19, s1 // (op 1041) 

v_mov_b64 v[70:71], s[24:25] // (op 241) convert

s_lshl_b32 s1, s24, 7 // (op 241) 
s_mul_i32 s6, 3, s1 // (op 241) 

v_add_u32 v14, v45, s6 // (op 241) 

buffer_load_dwordx4 v[46:49], v14, s[48:51], 0 offen // (op 241) Load value
v_add_u32 v14, v14, v70 // (op 241) 
buffer_load_dwordx4 v[50:53], v14, s[48:51], 0 offen // (op 241) Load value
v_add_u32 v14, v14, v70 // (op 241) 
buffer_load_dwordx4 v[54:57], v14, s[48:51], 0 offen // (op 241) Load value
v_add_u32 v14, v14, v70 // (op 241) 
buffer_load_dwordx4 v[58:61], v14, s[48:51], 0 offen // (op 241) Load value




s_lshl_b32 s1, s24, 8 // (op 977) 
v_add_u32 v17, v17, s1 // (op 977) 

v_mov_b64 v[70:71], s[34:35] // (op 242) convert

s_lshl_b32 s1, s34, 2 // (op 242) 
s_mul_i32 s6, 3, s1 // (op 242) 

v_add_u32 v14, v68, s6 // (op 242) 

buffer_load_ubyte v62, v14, s[56:59], 0 offen offset: 0 // (op 242) Load value
v_add_u32 v14, v14, v70 // (op 242) 
buffer_load_ubyte v63, v14, s[56:59], 0 offen offset: 0 // (op 242) Load value





s_lshl_b32 s1, s34, 3 // (op 1105) 
v_add_u32 v20, v20, s1 // (op 1105) 

s_lshl_b64 s[6:7], s[12:13], 7 // (op 228) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 228) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 228) 

s_mul_i32 s1, 3, s6 // (op 228) 

v_add_u32 v14, v66, s1 // (op 228) 


buffer_load_dwordx2 v[12:13], v14, s[8:11], 0 offen // (op 228) Load value


s_barrier  // (op 1465) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 913) 



s_lshl_b64 s[14:15], s[6:7], 2 // (op 913) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 913) 

s_lshl_b32 s1, s6, 1 // (op 913) 

v_add_u32 v15, v15, s1 // (op 913) 

s_waitcnt vmcnt(14)
ds_write_b128 v21, v[26:29] // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v21, v[30:33] offset:64 // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v21, v[34:37] offset:128 // (op 13) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v21, v[38:41] offset:192 // (op 13) Store local data 
s_barrier  // (op 1479) 



s_waitcnt vmcnt(10)
ds_write_b8 v44, v42 offset:8192 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v44, v43 offset:8256 // (op 26) Store local data 
s_barrier  // (op 1452) 



s_waitcnt vmcnt(8)
ds_write_b64 v16, v[22:23] offset:8448 // (op 6) Store local data 
s_ashr_i64 s[6:7], s[22:23], 7 // (op 311) 
s_mov_b32 s1, s6 // (op 311) convert

s_and_b32 s6, s1, 4294967294 // (op 311) 

s_cmp_lt_u32 s0, s6 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_311 // (op 311) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_311 if true)
RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_311:















s_ashr_i64 s[0:1], s[22:23], 7 // (op 875) 
s_lshr_b32 s6, s0, 1 // (op 875) 

v_lshl_add_u32 v14, s6, 8, v1 // (op 875) 

s_lshl_b32 s0, s16, 4 // (op 875) 
v_mul_lo_u32 v27, v14, s13 // (op 875) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 875) most significant: high of low * low
v_mul_lo_u32 v26, v14, s12 // (op 875) least significant: low of low * low
v_add_u32 v27, v27, v15 // (op 875) most significant: sum


s_mov_b32 s7, 0 // (op 875) convert
s_mov_b32 s6, s0 // (op 875) convert

v_mov_b32 v14, s6 // (op 875) 
v_mov_b32 v15, s7 // (op 875) 
v_add_co_u32 v16, s[0:1], v14, v26 // (op 875) least significant half
v_addc_co_u32 v17, s[0:1], v15, v27, s[0:1] // (op 875) most significant half



v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 875) 


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 875) 

v_mov_b32 v16, v14 // (op 875) convert

s_mul_i32 s1, s12, s23 // (op 877) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 877) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 877) least significant
s_mul_hi_u32 s7, s12, s22 // (op 877) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 877) most significant: sum
s_add_u32 s1, s1, s7 // (op 877) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 877) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 877) 

s_mov_b32 s8, s20 // (op 877) reg expression
s_mov_b32 s9, s21 // (op 877) reg expression
s_mov_b32 s10, s0 // (op 877) reg expression
s_mov_b32 s11, 131072 // (op 877) call()

v_and_b32 v14, 4294967292, v1 // (op 943) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 943) 
v_and_b32 v15, -4, v1 // (op 943) 
s_lshr_b32 s6, s0, 1 // (op 943) 

v_sub_u32 v17, v1, v15 // (op 943) 

v_lshl_add_u32 v15, s6, 8, v14 // (op 943) 


v_lshlrev_b32 v14, 4, v17 // (op 943) 

v_lshl_add_u32 v17, s17, 6, v14 // (op 943) 

v_mul_lo_u32 v14, v15, s24 // (op 943) 
v_add_u32 v14, v14, v17 // (op 943) 


s_mul_i32 s1, s24, s23 // (op 945) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 945) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 945) least significant
s_mul_hi_u32 s7, s24, s22 // (op 945) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 945) most significant: sum
s_add_u32 s1, s1, s7 // (op 945) most significant: sum


s_mov_b32 s48, s26 // (op 945) reg expression
s_mov_b32 s49, s27 // (op 945) reg expression
s_mov_b32 s50, s0 // (op 945) reg expression
s_mov_b32 s51, 131072 // (op 945) call()

v_lshrrev_b32 v15, 7, v1 // (op 1007) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1007) 
v_and_b32 v17, 15, v1 // (op 1007) 
s_lshr_b32 s6, s0, 1 // (op 1007) 

v_lshl_add_u32 v18, v15, 4, v17 // (op 1007) 


v_lshl_add_u32 v15, s16, 4, v18 // (op 1007) 

v_and_b32 v17, 63, v1 // (op 1007) 
v_lshrrev_b32 v18, 4, v17 // (op 1007) 

v_lshl_add_u32 v17, s6, 3, v18 // (op 1007) 


v_mul_lo_u32 v18, v17, s28 // (op 1007) 

v_add_u32 v17, v15, v18 // (op 1007) 


s_mul_i32 s1, s28, s33 // (op 1009) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1009) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1009) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1009) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1009) most significant: sum
s_add_u32 s1, s1, s7 // (op 1009) most significant: sum


s_mov_b32 s52, s30 // (op 1009) reg expression
s_mov_b32 s53, s31 // (op 1009) reg expression
s_mov_b32 s54, s0 // (op 1009) reg expression
s_mov_b32 s55, 131072 // (op 1009) call()

v_lshrrev_b32 v15, 6, v1 // (op 1071) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1071) 
v_lshlrev_b32 v18, 1, v15 // (op 1071) 

s_lshr_b32 s6, s0, 1 // (op 1071) 

v_and_b32 v15, -64, v1 // (op 1071) 
v_lshl_add_u32 v19, s6, 3, v18 // (op 1071) 


v_sub_u32 v18, v1, v15 // (op 1071) 

v_lshl_add_u32 v15, s17, 6, v18 // (op 1071) 

v_mul_lo_u32 v18, v19, s34 // (op 1071) 
v_add_u32 v18, v18, v15 // (op 1071) 


s_mul_i32 s1, s34, s33 // (op 1073) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1073) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1073) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1073) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1073) most significant: sum
s_add_u32 s1, s1, s7 // (op 1073) most significant: sum


s_mov_b32 s56, s36 // (op 1073) reg expression
s_mov_b32 s57, s37 // (op 1073) reg expression
s_mov_b32 s58, s0 // (op 1073) reg expression
s_mov_b32 s59, 131072 // (op 1073) call()

v_lshlrev_b32 v15, 6, v1 // (op 1164) 
v_lshrrev_b32 v19, 3, v15 // (op 1164) 

v_and_b32 v15, -4, v1 // (op 1217) 
v_sub_u32 v20, v1, v15 // (op 1217) 

v_lshlrev_b32 v15, 4, v20 // (op 1217) 

v_lshrrev_b32 v20, 2, v1 // (op 1217) 
v_lshl_add_u32 v21, v20, 8, v15 // (op 1217) 


v_and_b32 v15, -64, v1 // (op 1275) 
v_sub_u32 v20, v1, v15 // (op 1275) 

v_lshrrev_b32 v15, 6, v1 // (op 1275) 
v_lshl_add_u32 v22, v15, 7, v20 // (op 1275) 


s_ashr_i64 s[0:1], s[22:23], 7 // (op 317) 
s_and_b32 s6, s0, 4294967294 // (op 317) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 316) 
s_mov_b32 s7, s0 // (op 316) convert

s_cmp_lt_u32 s6, s7 // (op 316) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_316 // (op 316) Condition: Top (jump to RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_316 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_316:


s_add_u32 s6, s6, 1 // (op 318) 
v_mov_b32 v15, v16 // (op 170) 
v_mov_b32 v20, v15 // (op 170) reg expression

buffer_load_dwordx2 v[12:13], v20, s[8:11], 0 offen // (op 170) Load value


s_lshl_b64 s[0:1], s[12:13], 7 // (op 865) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 865) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 865) 

v_add_u32 v16, v16, s0 // (op 865) 


s_barrier  // (op 1455) 
v_mov_b64 v[24:25], s[24:25] // (op 183) convert




v_mov_b32 v15, v14 // (op 183) 
v_mov_b32 v20, v15 // (op 183) reg expression
buffer_load_dwordx4 v[26:29], v20, s[48:51], 0 offen // (op 183) Load value
v_add_u32 v20, v20, v24 // (op 183) 
buffer_load_dwordx4 v[30:33], v20, s[48:51], 0 offen // (op 183) Load value
v_add_u32 v20, v20, v24 // (op 183) 
s_waitcnt vmcnt(2)
ds_write_b64 v19, v[12:13] offset:8448 // (op 171) Store local data 
buffer_load_dwordx4 v[34:37], v20, s[48:51], 0 offen // (op 183) Load value

v_add_u32 v20, v20, v24 // (op 183) 
buffer_load_dwordx4 v[38:41], v20, s[48:51], 0 offen // (op 183) Load value



s_lshl_b32 s7, s24, 7 // (op 935) 
v_add_u32 v14, v14, s7 // (op 935) 


s_barrier  // (op 1468) 
v_mov_b64 v[12:13], s[34:35] // (op 184) convert




v_mov_b32 v15, v18 // (op 184) 
v_mov_b32 v20, v15 // (op 184) reg expression
buffer_load_ubyte v24, v20, s[56:59], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v20, v20, v12 // (op 184) 
s_waitcnt vmcnt(4)
ds_write_b128 v21, v[26:29] // (op 181) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v21, v[30:33] offset:64 // (op 181) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v21, v[34:37] offset:128 // (op 181) Store local data 
buffer_load_ubyte v25, v20, s[56:59], 0 offen offset: 0 // (op 184) Load value



s_lshl_b32 s7, s34, 2 // (op 1063) 
v_add_u32 v18, v18, s7 // (op 1063) 

s_waitcnt vmcnt(2)
ds_write_b128 v21, v[38:41] offset:192 // (op 181) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1458) 



s_barrier  // (op 1482) 
ds_read_b64_tr_b4 v[26:27], v3 offset:8448 // (op 173) Transpose load local data 



ds_read_b64_tr_b4 v[28:29], v3 offset:8576 // (op 173) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v22, v24 offset:8192 // (op 182) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1472) 
s_waitcnt vmcnt(0)
ds_write_b8 v22, v25 offset:8256 // (op 182) Store local data 




ds_read_b64_tr_b8 v[30:31], v5 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v5 offset:512 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v5 offset:4096 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v5 offset:4608 // (op 175) Transpose load local data 

v_mov_b64 v[12:13], s[28:29] // (op 177) convert

v_mov_b32 v15, v17 // (op 177) 
v_mov_b32 v20, v15 // (op 177) reg expression
v_mov_b32 v23, v20 // (op 177) 
buffer_load_ubyte v24, v23, s[52:55], 0 offen offset: 0 // (op 177) Load value




s_lshl_b32 s7, s28, 2 // (op 999) 
v_add_u32 v17, v17, s7 // (op 999) 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1486) 


ds_read_u8 v15, v0 offset:8192 // (op 179) Load local data 



ds_read_b64_tr_b8 v[38:39], v5 offset:32 // (op 324) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:37], a[0:3], v24, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 180) 
ds_read_b64_tr_b8 v[40:41], v5 offset:544 // (op 324) Transpose load local data 


ds_read_b64_tr_b8 v[42:43], v5 offset:4128 // (op 324) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v5 offset:4640 // (op 324) Transpose load local data 


ds_read_u8 v15, v0 offset:8224 // (op 327) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[38:45], a[4:7], v24, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 328) 




s_ashr_i64 s[0:1], s[22:23], 7 // (op 316) 
s_mov_b32 s7, s0 // (op 316) convert

s_cmp_lt_u32 s6, s7 // (op 316) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_316 // (op 316) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_316 if true)
RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_316:






















s_barrier  // (op 1568) 






v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_125
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
s_branch RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_125:


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
RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.LRR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_end:
.size RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2, .LRR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_end-RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 18944 // lds bytes
  .amdhsa_accum_offset 92
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
  - .name: RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
    .symbol: RR_GEMM_NT_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 18944
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 90
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

