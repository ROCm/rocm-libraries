// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2,@function
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 1273) 
v_and_b32 v3, 48, v1 // (op 1273) 
v_sub_u32 v4, v0, v3 // (op 1273) 

v_lshrrev_b32 v3, 4, v0 // (op 1273) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1273) 


v_lshlrev_b32 v3, 4, v0 // (op 1273) 
v_mul_lo_u32 v4, v3, 6 // (op 1273) 

v_lshrrev_b32 v3, 3, v4 // (op 1273) 

v_and_b32 v4, 268435455, v0 // (op 1273) 

v_lshlrev_b32 v0, 2, v4 // (op 1273) 

v_add_u32 v4, v3, v0 // (op 1273) 


v_and_b32 v0, 63, v1 // (op 1417) 
v_and_b32 v3, 48, v1 // (op 1417) 
v_sub_u32 v5, v0, v3 // (op 1417) 

v_lshrrev_b32 v3, 4, v0 // (op 1417) 

v_lshl_add_u32 v0, v3, 5, v5 // (op 1417) 


v_lshlrev_b32 v3, 4, v0 // (op 1417) 
v_mul_lo_u32 v5, v3, 6 // (op 1417) 

v_lshrrev_b32 v3, 3, v5 // (op 1417) 

v_and_b32 v5, 268435455, v0 // (op 1417) 

v_lshlrev_b32 v0, 2, v5 // (op 1417) 

v_add_u32 v5, v3, v0 // (op 1417) 


v_lshrrev_b32 v0, 6, v1 // (op 1159) 
v_and_b32 v3, 63, v1 // (op 1159) 
v_lshrrev_b32 v6, 4, v3 // (op 1159) 
v_lshlrev_b32 v7, 2, v6 // (op 1159) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1159) 


v_lshl_add_u32 v0, s16, 6, v6 // (op 1159) 

v_lshrrev_b32 v6, 2, v3 // (op 1159) 

v_and_b32 v3, 3, v6 // (op 1159) 

v_and_b32 v6, 3, v1 // (op 1159) 
v_lshl_add_u32 v7, v3, 2, v6 // (op 1159) 


v_lshl_add_u32 v3, s17, 4, v7 // (op 1159) 

v_mul_lo_u32 v6, v3, s2 // (op 1159) 

v_add_lshl_u32 v3, v0, v6, 2 // (op 1159) 


v_and_b32 v0, 48, v1 // (op 1220) 
v_and_b32 v6, 63, v1 // (op 1220) 
v_sub_u32 v7, v6, v0 // (op 1220) 

v_lshrrev_b32 v6, 1, v7 // (op 1220) 
s_mul_i32 s1, s2, s9 // (op 1161) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1161) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1161) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1161) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1161) most significant: sum
s_add_u32 s1, s1, s19 // (op 1161) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1161) 

v_add_lshl_u32 v8, v0, v6, 6 // (op 1220) 


v_and_b32 v0, 1, v7 // (op 1220) 

v_lshlrev_b32 v6, 3, v0 // (op 1220) 

v_lshrrev_b32 v0, 6, v1 // (op 1220) 
v_lshl_add_u32 v7, v0, 4, v6 // (op 1220) 


v_add_u32 v0, v8, v7 // (op 1220) 


v_lshrrev_b32 v6, 6, v1 // (op 1331) 
v_and_b32 v7, 15, v1 // (op 1331) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 1331) 


v_and_b32 v6, 63, v1 // (op 1331) 
v_lshrrev_b32 v7, 4, v6 // (op 1331) 

v_lshl_add_u32 v6, v7, 6, v8 // (op 1331) 


v_lshrrev_b32 v7, 6, v1 // (op 1347) 
v_and_b32 v8, 63, v1 // (op 1347) 
v_lshrrev_b32 v9, 4, v8 // (op 1347) 
v_lshlrev_b32 v10, 2, v9 // (op 1347) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1347) 


v_lshl_add_u32 v7, s16, 6, v9 // (op 1347) 

v_lshrrev_b32 v9, 2, v8 // (op 1347) 

v_and_b32 v8, 3, v9 // (op 1347) 

v_and_b32 v9, 3, v1 // (op 1347) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1347) 


v_lshl_add_u32 v8, s17, 4, v10 // (op 1347) 

v_mul_lo_u32 v9, v8, s4 // (op 1347) 

v_add_lshl_u32 v8, v7, v9, 2 // (op 1347) 


v_and_b32 v7, 48, v1 // (op 1393) 
v_and_b32 v9, 63, v1 // (op 1393) 
v_sub_u32 v10, v9, v7 // (op 1393) 

v_lshrrev_b32 v9, 1, v10 // (op 1393) 
s_mul_i32 s1, s4, s9 // (op 1349) most significant: low of low * high
v_add_lshl_u32 v11, v7, v9, 6 // (op 1393) 


v_and_b32 v7, 1, v10 // (op 1393) 

v_lshlrev_b32 v9, 3, v7 // (op 1393) 

v_lshrrev_b32 v7, 6, v1 // (op 1393) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 1393) 


v_add_u32 v7, v11, v10 // (op 1393) 


v_lshrrev_b32 v9, 6, v1 // (op 1446) 
v_and_b32 v10, 15, v1 // (op 1446) 
v_lshl_add_u32 v11, v9, 4, v10 // (op 1446) 


v_and_b32 v9, 63, v1 // (op 1446) 
v_lshrrev_b32 v10, 4, v9 // (op 1446) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 1446) 


s_mul_i32 s38, s5, s8 // (op 1349) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1349) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1349) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1349) most significant: sum
s_add_u32 s1, s1, s39 // (op 1349) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 1349) 

s_mov_b32 s40, s6 // (op 1161) reg expression
s_mov_b32 s41, s7 // (op 1161) reg expression
s_mov_b32 s42, s18 // (op 1161) reg expression
s_mov_b32 s43, 131072 // (op 1161) call()

s_mov_b32 s44, s10 // (op 1349) reg expression

s_mov_b32 s45, s11 // (op 1349) reg expression
s_mov_b32 s46, s38 // (op 1349) reg expression
s_mov_b32 s47, 131072 // (op 1349) call()



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
v_and_b32 v12, -4, v1 // (op 926) 
v_sub_u32 v13, v1, v12 // (op 926) 

v_lshlrev_b32 v12, 4, v13 // (op 926) 

v_lshl_add_u32 v13, s16, 6, v12 // (op 926) 

v_mov_b32 v15, 0 // (op 926) convert
v_mov_b32 v14, v13 // (op 926) convert

v_lshrrev_b32 v12, 2, v1 // (op 926) 
v_mul_lo_u32 v17, v12, s13 // (op 926) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 926) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 926) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 926) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 926) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 926) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 928) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 928) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 928) least significant
s_mul_hi_u32 s7, s12, s22 // (op 928) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 928) most significant: sum
s_add_u32 s1, s1, s7 // (op 928) most significant: sum


s_mov_b32 s8, s20 // (op 928) reg expression
s_mov_b32 s9, s21 // (op 928) reg expression
s_mov_b32 s10, s0 // (op 928) reg expression
s_mov_b32 s11, 131072 // (op 928) call()

v_and_b32 v14, -4, v1 // (op 945) 
v_sub_u32 v15, v1, v14 // (op 945) 

v_lshlrev_b32 v14, 4, v15 // (op 945) 

v_lshl_add_u32 v15, s16, 6, v14 // (op 945) 

v_mov_b32 v17, 0 // (op 945) convert
v_mov_b32 v16, v15 // (op 945) convert

v_lshrrev_b32 v14, 2, v1 // (op 945) 
v_mul_lo_u32 v19, v14, s13 // (op 945) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 945) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 945) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 945) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 945) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 945) most significant half



s_mul_i32 s1, s12, s23 // (op 947) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 947) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 947) least significant
s_mul_hi_u32 s7, s12, s22 // (op 947) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 947) most significant: sum
s_add_u32 s1, s1, s7 // (op 947) most significant: sum


s_mov_b32 s8, s20 // (op 947) reg expression
s_mov_b32 s9, s21 // (op 947) reg expression
s_mov_b32 s10, s0 // (op 947) reg expression
s_mov_b32 s11, 131072 // (op 947) call()

s_lshl_b32 s0, s17, 4 // (op 990) 
s_mov_b32 s7, 0 // (op 990) convert
s_mov_b32 s6, s0 // (op 990) convert

v_mul_lo_u32 v17, v1, s25 // (op 990) most significant: low of low * high
v_mul_hi_u32 v18, v1, s24 // (op 990) most significant: high of low * low
v_mul_lo_u32 v16, v1, s24 // (op 990) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 990) most significant: sum

v_mov_b32 v18, s6 // (op 990) 
v_mov_b32 v19, s7 // (op 990) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 990) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 990) most significant half




v_mul_lo_u32 v19, v17, 6 // (op 990) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 990) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 990) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 990) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 990) 

v_mov_b32 v18, v16 // (op 990) convert

s_mul_i32 s1, s23, 6 // (op 992) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 992) least significant
s_mul_hi_u32 s6, s22, 6 // (op 992) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 992) most significant: sum

s_mul_i32 s7, s24, s1 // (op 992) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 992) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 992) least significant
s_mul_hi_u32 s15, s24, s0 // (op 992) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 992) most significant: sum
s_add_u32 s7, s7, s15 // (op 992) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 992) 

s_mov_b32 s48, s26 // (op 992) reg expression
s_mov_b32 s49, s27 // (op 992) reg expression
s_mov_b32 s50, s0 // (op 992) reg expression
s_mov_b32 s51, 131072 // (op 992) call()

s_lshl_b32 s0, s17, 4 // (op 1009) 
s_mov_b32 s7, 0 // (op 1009) convert
s_mov_b32 s6, s0 // (op 1009) convert

v_mul_lo_u32 v17, v1, s25 // (op 1009) most significant: low of low * high
v_mul_hi_u32 v19, v1, s24 // (op 1009) most significant: high of low * low
v_mul_lo_u32 v16, v1, s24 // (op 1009) least significant: low of low * low
v_add_u32 v17, v17, v19 // (op 1009) most significant: sum

v_mov_b32 v19, s6 // (op 1009) 
v_mov_b32 v20, s7 // (op 1009) 
v_add_co_u32 v16, s[0:1], v16, v19 // (op 1009) least significant half
v_addc_co_u32 v17, s[0:1], v17, v20, s[0:1] // (op 1009) most significant half




v_mul_lo_u32 v21, v17, 6 // (op 1009) most significant: low of high * low
v_mul_hi_u32 v19, v16, 6 // (op 1009) most significant: high of low * low
v_mul_lo_u32 v20, v16, 6 // (op 1009) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 1009) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 1009) 

v_mov_b32 v19, v16 // (op 1009) convert

s_mul_i32 s1, s23, 6 // (op 1011) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1011) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1011) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1011) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1011) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1011) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1011) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1011) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1011) most significant: sum
s_add_u32 s7, s7, s15 // (op 1011) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1011) 

s_mov_b32 s48, s26 // (op 1011) reg expression
s_mov_b32 s49, s27 // (op 1011) reg expression
s_mov_b32 s50, s0 // (op 1011) reg expression
s_mov_b32 s51, 131072 // (op 1011) call()

v_and_b32 v16, -64, v1 // (op 1054) 
v_sub_u32 v17, v1, v16 // (op 1054) 

v_lshl_add_u32 v16, s16, 6, v17 // (op 1054) 

v_lshrrev_b32 v17, 6, v1 // (op 1054) 
v_lshlrev_b32 v20, 1, v17 // (op 1054) 

v_mul_lo_u32 v17, v20, s28 // (op 1054) 

v_add_u32 v20, v16, v17 // (op 1054) 


s_mul_i32 s1, s28, s33 // (op 1056) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1056) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1056) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1056) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1056) most significant: sum
s_add_u32 s1, s1, s7 // (op 1056) most significant: sum


s_mov_b32 s52, s30 // (op 1056) reg expression
s_mov_b32 s53, s31 // (op 1056) reg expression
s_mov_b32 s54, s0 // (op 1056) reg expression
s_mov_b32 s55, 131072 // (op 1056) call()

v_and_b32 v16, -64, v1 // (op 1073) 
v_sub_u32 v17, v1, v16 // (op 1073) 

v_lshl_add_u32 v16, s16, 6, v17 // (op 1073) 

v_lshrrev_b32 v17, 6, v1 // (op 1073) 
v_lshlrev_b32 v21, 1, v17 // (op 1073) 

v_mul_lo_u32 v17, v21, s28 // (op 1073) 

v_add_u32 v21, v16, v17 // (op 1073) 


s_mul_i32 s1, s28, s33 // (op 1075) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1075) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1075) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1075) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1075) most significant: sum
s_add_u32 s1, s1, s7 // (op 1075) most significant: sum


s_mov_b32 s52, s30 // (op 1075) reg expression
s_mov_b32 s53, s31 // (op 1075) reg expression
s_mov_b32 s54, s0 // (op 1075) reg expression
s_mov_b32 s55, 131072 // (op 1075) call()

v_and_b32 v16, 15, v1 // (op 1118) 
v_lshl_add_u32 v17, s17, 4, v16 // (op 1118) 

v_and_b32 v16, 63, v1 // (op 1118) 
v_lshrrev_b32 v22, 4, v16 // (op 1118) 

v_mul_lo_u32 v16, v22, s34 // (op 1118) 
v_add_u32 v16, v16, v17 // (op 1118) 


s_mul_i32 s1, s34, s33 // (op 1120) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1120) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1120) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1120) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1120) most significant: sum
s_add_u32 s1, s1, s7 // (op 1120) most significant: sum


s_mov_b32 s56, s36 // (op 1120) reg expression
s_mov_b32 s57, s37 // (op 1120) reg expression
s_mov_b32 s58, s0 // (op 1120) reg expression
s_mov_b32 s59, 131072 // (op 1120) call()

v_and_b32 v17, 15, v1 // (op 1137) 
v_lshl_add_u32 v22, s17, 4, v17 // (op 1137) 

v_and_b32 v17, 63, v1 // (op 1137) 
v_lshrrev_b32 v23, 4, v17 // (op 1137) 

v_mul_lo_u32 v17, v23, s34 // (op 1137) 
v_add_u32 v17, v17, v22 // (op 1137) 


s_mul_i32 s1, s34, s33 // (op 1139) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1139) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1139) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1139) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1139) most significant: sum
s_add_u32 s1, s1, s7 // (op 1139) most significant: sum


s_mov_b32 s56, s36 // (op 1139) reg expression
s_mov_b32 s57, s37 // (op 1139) reg expression
s_mov_b32 s58, s0 // (op 1139) reg expression
s_mov_b32 s59, 131072 // (op 1139) call()


v_mov_b64 v[22:23], s[34:35] // (op 774) convert

v_mov_b32 v24, v16 // (op 774) reg expression
v_mov_b32 v25, v24 // (op 774) 
buffer_load_ubyte v26, v25, s[56:59], 0 offen offset: 0 // (op 774) Load value





v_mov_b32 v22, v18 // (op 778) reg expression

buffer_load_dwordx3 v[28:30], v22, s[48:51], 0 offen // (op 778) Load value

v_and_b32 v27, -4, v1 // (op 1203) 
v_sub_u32 v22, v1, v27 // (op 1203) 

v_lshlrev_b32 v27, 4, v22 // (op 1203) 

v_lshrrev_b32 v22, 2, v1 // (op 1203) 
v_lshl_add_u32 v23, v22, 6, v27 // (op 1203) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 1203) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 881) 
v_mov_b64 v[22:23], s[6:7] // (op 881) convert


v_mov_b64 v[24:25], v[12:13] // (op 881) reg expression
s_mov_b32 m0, s0 // (op 881) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 881) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 881) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 881) most significant half

s_add_u32 m0, m0, 2048 // (op 881) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 881) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 881) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 881) most significant half

s_add_u32 m0, m0, 2048 // (op 881) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 881) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 881) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 881) most significant half

s_add_u32 m0, m0, 2048 // (op 881) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 881) Load value direct to lds




v_mov_b64 v[22:23], s[28:29] // (op 784) convert

v_mov_b32 v27, v20 // (op 784) reg expression
buffer_load_ubyte v24, v27, s[52:55], 0 offen offset: 0 // (op 784) Load value
v_add_u32 v27, v27, v22 // (op 784) 
buffer_load_ubyte v25, v27, s[52:55], 0 offen offset: 0 // (op 784) Load value



v_mov_b64 v[22:23], s[34:35] // (op 788) convert

s_lshl_b32 s0, s34, 2 // (op 788) 
v_add_u32 v27, v16, s0 // (op 788) 

v_mov_b32 v31, v27 // (op 788) 
buffer_load_ubyte v32, v31, s[56:59], 0 offen offset: 0 // (op 788) Load value






s_lshl_b64 s[0:1], s[24:25], 7 // (op 792) 
s_mul_i32 s7, s1, 6 // (op 792) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 792) least significant
s_mul_hi_u32 s14, s0, 6 // (op 792) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 792) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 792) 

v_add_u32 v16, v18, s0 // (op 792) 


buffer_load_dwordx3 v[34:36], v16, s[48:51], 0 offen // (op 792) Load value


v_and_b32 v16, -4, v1 // (op 1376) 
v_sub_u32 v18, v1, v16 // (op 1376) 

v_lshlrev_b32 v16, 4, v18 // (op 1376) 

v_lshrrev_b32 v18, 2, v1 // (op 1376) 
v_lshl_add_u32 v27, v18, 6, v16 // (op 1376) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1376) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 886) 
v_mov_b64 v[22:23], s[6:7] // (op 886) convert


s_lshl_b64 s[6:7], s[12:13], 7 // (op 886) 
v_mov_b32 v16, s6 // (op 886) 
v_mov_b32 v18, s7 // (op 886) 
v_add_co_u32 v38, s[14:15], v12, v16 // (op 886) least significant half
v_addc_co_u32 v39, s[14:15], v13, v18, s[14:15] // (op 886) most significant half




s_add_u32 s1, 8192, s0 // (op 886) 
s_mov_b32 m0, s1 // (op 886) reg expression
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 886) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v22 // (op 886) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 886) most significant half

s_add_u32 m0, m0, 2048 // (op 886) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 886) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v22 // (op 886) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 886) most significant half

s_add_u32 m0, m0, 2048 // (op 886) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 886) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v22 // (op 886) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 886) most significant half

s_add_u32 m0, m0, 2048 // (op 886) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 886) Load value direct to lds






v_mov_b64 v[12:13], s[28:29] // (op 798) convert

s_lshl_b32 s0, s28, 2 // (op 798) 
v_add_u32 v16, v20, s0 // (op 798) 

buffer_load_ubyte v22, v16, s[52:55], 0 offen offset: 0 // (op 798) Load value
v_add_u32 v16, v16, v12 // (op 798) 
buffer_load_ubyte v23, v16, s[52:55], 0 offen offset: 0 // (op 798) Load value



v_lshlrev_b32 v16, 4, v1 // (op 1261) 
v_mul_lo_u32 v18, v16, 6 // (op 1261) 

v_lshrrev_b32 v16, 3, v18 // (op 1261) 

v_and_b32 v18, 268435455, v1 // (op 1261) 
v_lshlrev_b32 v20, 2, v18 // (op 1261) 

v_add_u32 v18, v16, v20 // (op 1261) 





s_waitcnt vmcnt(14)
ds_write_b96 v18, v[28:30] offset:16384 // (op 800) Store local data 

v_and_b32 v16, -64, v1 // (op 1314) 
v_sub_u32 v18, v1, v16 // (op 1314) 

v_lshrrev_b32 v16, 6, v1 // (op 1314) 
v_lshl_add_u32 v20, v16, 7, v18 // (op 1314) 





s_waitcnt vmcnt(9)
ds_write_b8 v20, v24 offset:18432 // (op 802) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v20, v25 offset:18496 // (op 802) Store local data 

v_and_b32 v16, -4, v1 // (op 1175) 
v_sub_u32 v18, v1, v16 // (op 1175) 

v_lshlrev_b32 v16, 4, v18 // (op 1175) 

v_lshrrev_b32 v18, 2, v1 // (op 1175) 
v_lshl_add_u32 v20, v18, 6, v16 // (op 1175) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1175) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v16, 4, v1 // (op 1232) 
v_mul_lo_u32 v18, v16, 6 // (op 1232) 

v_lshrrev_b32 v16, 3, v18 // (op 1232) 

v_and_b32 v18, 268435455, v1 // (op 1232) 
v_lshlrev_b32 v20, 2, v18 // (op 1232) 

v_add_u32 v18, v16, v20 // (op 1232) 


v_and_b32 v16, -64, v1 // (op 1285) 
v_sub_u32 v20, v1, v16 // (op 1285) 

v_lshrrev_b32 v16, 6, v1 // (op 1285) 
v_lshl_add_u32 v27, v16, 7, v20 // (op 1285) 


v_and_b32 v16, -4, v1 // (op 1361) 
v_sub_u32 v20, v1, v16 // (op 1361) 

v_lshlrev_b32 v16, 4, v20 // (op 1361) 

v_lshrrev_b32 v20, 2, v1 // (op 1361) 
v_lshl_add_u32 v31, v20, 6, v16 // (op 1361) 


s_nop 0
v_readfirstlane_b32 s1, v31 // (op 1361) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v16, 4, v1 // (op 1405) 
v_mul_lo_u32 v20, v16, 6 // (op 1405) 

v_lshrrev_b32 v16, 3, v20 // (op 1405) 

v_and_b32 v20, 268435455, v1 // (op 1405) 
v_lshlrev_b32 v31, 2, v20 // (op 1405) 

v_add_u32 v20, v16, v31 // (op 1405) 


v_and_b32 v16, -64, v1 // (op 1429) 
v_sub_u32 v31, v1, v16 // (op 1429) 

v_lshrrev_b32 v16, 6, v1 // (op 1429) 
v_lshl_add_u32 v33, v16, 7, v31 // (op 1429) 


s_mov_b32 s6, 0 // (op 311) call()
s_ashr_i64 s[14:15], s[22:23], 7 // (op 310) 
s_mov_b32 s7, s14 // (op 310) convert

s_and_b32 s14, s7, 4294967294 // (op 310) 

s_cmp_lt_u32 s6, s14 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_310 // (op 310) Condition: Top (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_310 if false)
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_310:
s_barrier  // (op 1487) 
s_add_u32 s6, s6, 2 // (op 312) 



ds_read_b64_tr_b8 v[38:39], v0 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:512 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:4096 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:4608 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1500) 



ds_read_b96_tr_b6 v[46:48], v4 offset:16384 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[49:51], v4 offset:16640 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1514) 


ds_read_u8 v16, v6 offset:18432 // (op 75) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[46:51], a[0:3], v16, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 82) 
ds_read_b64_tr_b8 v[52:53], v0 offset:32 // (op 340) Transpose load local data 


ds_read_b64_tr_b8 v[54:55], v0 offset:544 // (op 340) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:4128 // (op 340) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:4640 // (op 340) Transpose load local data 


ds_read_u8 v16, v6 offset:18464 // (op 344) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[46:51], a[4:7], v16, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 346) 




v_mov_b64 v[12:13], s[34:35] // (op 80) convert

v_mov_b32 v16, v17 // (op 80) 
s_lshl_b32 s7, s34, 3 // (op 80) 
v_add_u32 v31, v16, s7 // (op 80) 

v_mov_b32 v37, v31 // (op 80) 
buffer_load_ubyte v26, v37, s[56:59], 0 offen offset: 0 // (op 80) Load value





v_mov_b32 v31, v19 // (op 10) 
s_lshl_b64 s[14:15], s[24:25], 7 // (op 10) 
s_mul_i32 s19, s15, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 10) least significant
s_mul_hi_u32 s7, s14, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 10) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s7, s14, 1 // (op 10) 

v_add_u32 v12, v31, s7 // (op 10) 


buffer_load_dwordx3 v[28:30], v12, s[48:51], 0 offen // (op 10) Load value

s_barrier  // (op 1490) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 865) 
v_mov_b64 v[12:13], s[14:15] // (op 865) convert


v_mov_b64 v[38:39], v[14:15] // (op 865) 
s_lshl_b64 s[14:15], s[12:13], 7 // (op 865) 
s_lshl_b64 s[18:19], s[14:15], 1 // (op 865) 

v_mov_b32 v37, s18 // (op 865) 
v_mov_b32 v40, s19 // (op 865) 
v_add_co_u32 v42, s[14:15], v38, v37 // (op 865) least significant half
v_addc_co_u32 v43, s[14:15], v39, v40, s[14:15] // (op 865) most significant half




s_mov_b32 m0, s0 // (op 865) reg expression
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 865) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v12 // (op 865) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 865) most significant half

s_add_u32 m0, m0, 2048 // (op 865) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 865) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v12 // (op 865) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 865) most significant half

s_add_u32 m0, m0, 2048 // (op 865) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 865) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v12 // (op 865) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 865) most significant half

s_add_u32 m0, m0, 2048 // (op 865) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 865) Load value direct to lds



v_mov_b64 v[12:13], s[28:29] // (op 17) convert

v_mov_b32 v37, v21 // (op 17) 
s_lshl_b32 s7, s28, 3 // (op 17) 
v_add_u32 v40, v37, s7 // (op 17) 

buffer_load_ubyte v24, v40, s[52:55], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v40, v40, v12 // (op 17) 
buffer_load_ubyte v25, v40, s[52:55], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1537) 



s_waitcnt vmcnt(14)
ds_write_b96 v20, v[34:36] offset:18688 // (op 235) Store local data 
s_barrier  // (op 1543) 



s_waitcnt vmcnt(9)
ds_write_b8 v33, v22 offset:20736 // (op 236) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v33, v23 offset:20800 // (op 236) Store local data 
s_barrier  // (op 1528) 



ds_read_b64_tr_b8 v[40:41], v7 offset:8192 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:8704 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:12288 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:12800 // (op 227) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1534) 



ds_read_b96_tr_b6 v[48:50], v5 offset:18688 // (op 229) Transpose load local data 
ds_read_b96_tr_b6 v[51:53], v5 offset:18944 // (op 229) Transpose load local data 
s_barrier  // (op 1572) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1540) 


ds_read_u8 v12, v9 offset:20736 // (op 231) Load local data 



ds_read_b64_tr_b8 v[54:55], v7 offset:8224 // (op 353) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:8736 // (op 353) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:53], a[0:3], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 234) 
ds_read_b64_tr_b8 v[58:59], v7 offset:12320 // (op 353) Transpose load local data 


ds_read_b64_tr_b8 v[60:61], v7 offset:12832 // (op 353) Transpose load local data 


ds_read_u8 v12, v9 offset:20768 // (op 357) Load local data 
s_barrier  // (op 1610) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[48:53], a[4:7], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 347) 




v_mov_b64 v[12:13], s[34:35] // (op 233) convert

s_lshl_b32 s7, s34, 2 // (op 233) 
s_mul_i32 s14, 3, s7 // (op 233) 

v_add_u32 v40, v16, s14 // (op 233) 

v_mov_b32 v41, v40 // (op 233) 
buffer_load_ubyte v32, v41, s[56:59], 0 offen offset: 0 // (op 233) Load value






s_lshl_b32 s7, s34, 3 // (op 1129) 
v_add_u32 v17, v17, s7 // (op 1129) 

s_lshl_b64 s[14:15], s[24:25], 7 // (op 237) 
s_mul_i32 s19, s15, 6 // (op 237) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 237) least significant
s_mul_hi_u32 s7, s14, 6 // (op 237) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 237) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 237) 

s_mul_i32 s7, 3, s14 // (op 237) 

v_add_u32 v16, v31, s7 // (op 237) 


buffer_load_dwordx3 v[34:36], v16, s[48:51], 0 offen // (op 237) Load value


s_barrier  // (op 1531) 
s_lshl_b64 s[14:15], s[24:25], 7 // (op 1001) 

s_mul_i32 s19, s15, 6 // (op 1001) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 1001) least significant
s_mul_hi_u32 s7, s14, 6 // (op 1001) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 1001) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 1001) 

s_lshl_b32 s7, s14, 1 // (op 1001) 

v_add_u32 v19, v19, s7 // (op 1001) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 876) 
v_mov_b64 v[12:13], s[14:15] // (op 876) convert


s_lshl_b64 s[14:15], s[12:13], 7 // (op 876) 
s_mul_i32 s19, 3, s15 // (op 876) most significant: low of low * high
s_mul_i32 s18, 3, s14 // (op 876) least significant
s_mul_hi_u32 s7, 3, s14 // (op 876) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 876) most significant: sum


v_mov_b32 v16, s18 // (op 876) 
v_mov_b32 v31, s19 // (op 876) 
v_add_co_u32 v40, s[14:15], v38, v16 // (op 876) least significant half
v_addc_co_u32 v41, s[14:15], v39, v31, s[14:15] // (op 876) most significant half




s_add_u32 s7, 8192, s1 // (op 876) 
s_mov_b32 m0, s7 // (op 876) reg expression
buffer_load_dwordx4 v40, s[8:11], 0 offen offset: 0 lds // (op 876) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v12 // (op 876) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 876) most significant half

s_add_u32 m0, m0, 2048 // (op 876) 
buffer_load_dwordx4 v40, s[8:11], 0 offen offset: 0 lds // (op 876) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v12 // (op 876) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 876) most significant half

s_add_u32 m0, m0, 2048 // (op 876) 
buffer_load_dwordx4 v40, s[8:11], 0 offen offset: 0 lds // (op 876) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v12 // (op 876) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 876) most significant half

s_add_u32 m0, m0, 2048 // (op 876) 
buffer_load_dwordx4 v40, s[8:11], 0 offen offset: 0 lds // (op 876) Load value direct to lds





v_mov_b64 v[12:13], s[28:29] // (op 238) convert

s_lshl_b32 s7, s28, 2 // (op 238) 
s_mul_i32 s14, 3, s7 // (op 238) 

v_add_u32 v16, v37, s14 // (op 238) 

s_lshl_b64 s[14:15], s[12:13], 7 // (op 937) 
s_lshl_b64 s[18:19], s[14:15], 1 // (op 937) 

v_mov_b32 v31, s18 // (op 937) 
v_mov_b32 v38, s19 // (op 937) 
v_add_co_u32 v14, s[14:15], v14, v31 // (op 937) least significant half
v_addc_co_u32 v15, s[14:15], v15, v38, s[14:15] // (op 937) most significant half




buffer_load_ubyte v22, v16, s[52:55], 0 offen offset: 0 // (op 238) Load value
v_add_u32 v16, v16, v12 // (op 238) 
buffer_load_ubyte v23, v16, s[52:55], 0 offen offset: 0 // (op 238) Load value



s_barrier  // (op 1503) 
s_lshl_b32 s7, s28, 3 // (op 1065) 



v_add_u32 v21, v21, s7 // (op 1065) 

s_waitcnt vmcnt(14)
ds_write_b96 v18, v[28:30] offset:16384 // (op 13) Store local data 
s_barrier  // (op 1517) 



s_waitcnt vmcnt(9)
ds_write_b8 v27, v24 offset:18432 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v27, v25 offset:18496 // (op 21) Store local data 
s_ashr_i64 s[14:15], s[22:23], 7 // (op 310) 
s_mov_b32 s7, s14 // (op 310) convert

s_and_b32 s14, s7, 4294967294 // (op 310) 

s_cmp_lt_u32 s6, s14 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_310 // (op 310) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_310 if true)
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_310:














s_barrier  // (op 1592) 
v_and_b32 v33, -4, v1 // (op 899) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 899) 
v_sub_u32 v12, v1, v33 // (op 899) 

s_lshr_b32 s6, s0, 1 // (op 899) 

v_lshlrev_b32 v33, 4, v12 // (op 899) 

v_lshl_add_u32 v12, s16, 6, v33 // (op 899) 

v_mov_b32 v21, 0 // (op 899) convert
v_mov_b32 v20, v12 // (op 899) convert

v_lshrrev_b32 v33, 2, v1 // (op 899) 
v_lshl_add_u32 v12, s6, 8, v33 // (op 899) 


v_mul_lo_u32 v19, v12, s13 // (op 899) most significant: low of low * high
v_mul_hi_u32 v33, v12, s12 // (op 899) most significant: high of low * low
v_mul_lo_u32 v18, v12, s12 // (op 899) least significant: low of low * low
v_add_u32 v19, v19, v33 // (op 899) most significant: sum


v_add_co_u32 v12, s[0:1], v20, v18 // (op 899) least significant half
v_addc_co_u32 v13, s[0:1], v21, v19, s[0:1] // (op 899) most significant half



s_mul_i32 s1, s12, s23 // (op 901) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 901) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 901) least significant
s_mul_hi_u32 s7, s12, s22 // (op 901) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 901) most significant: sum
s_add_u32 s1, s1, s7 // (op 901) most significant: sum


s_mov_b32 s8, s20 // (op 901) reg expression
s_mov_b32 s9, s21 // (op 901) reg expression
s_mov_b32 s10, s0 // (op 901) reg expression
s_mov_b32 s11, 131072 // (op 901) call()

s_ashr_i64 s[0:1], s[22:23], 7 // (op 967) 
s_lshr_b32 s6, s0, 1 // (op 967) 

v_lshl_add_u32 v33, s6, 8, v1 // (op 967) 

s_lshl_b32 s0, s17, 4 // (op 967) 
s_mov_b32 s7, 0 // (op 967) convert
s_mov_b32 s6, s0 // (op 967) convert

v_mul_lo_u32 v15, v33, s25 // (op 967) most significant: low of low * high
v_mul_hi_u32 v16, v33, s24 // (op 967) most significant: high of low * low
v_mul_lo_u32 v14, v33, s24 // (op 967) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 967) most significant: sum

v_mov_b32 v16, s6 // (op 967) 
v_mov_b32 v17, s7 // (op 967) 
v_add_co_u32 v14, s[0:1], v14, v16 // (op 967) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 967) most significant half





v_mul_lo_u32 v17, v15, 6 // (op 967) most significant: low of high * low
v_mul_hi_u32 v33, v14, 6 // (op 967) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 967) least significant: low of low * low
v_add_u32 v17, v17, v33 // (op 967) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 967) 

v_mov_b32 v33, v14 // (op 967) convert

s_mul_i32 s1, s23, 6 // (op 969) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 969) least significant
s_mul_hi_u32 s6, s22, 6 // (op 969) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 969) most significant: sum

s_mul_i32 s7, s24, s1 // (op 969) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 969) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 969) least significant
s_mul_hi_u32 s15, s24, s0 // (op 969) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 969) most significant: sum
s_add_u32 s7, s7, s15 // (op 969) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 969) 

s_mov_b32 s48, s26 // (op 969) reg expression
s_mov_b32 s49, s27 // (op 969) reg expression
s_mov_b32 s50, s0 // (op 969) reg expression
s_mov_b32 s51, 131072 // (op 969) call()

v_and_b32 v14, -64, v1 // (op 1031) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1031) 
v_sub_u32 v15, v1, v14 // (op 1031) 

s_lshr_b32 s6, s0, 1 // (op 1031) 

v_lshl_add_u32 v14, s16, 6, v15 // (op 1031) 

v_lshrrev_b32 v15, 6, v1 // (op 1031) 
v_lshlrev_b32 v16, 1, v15 // (op 1031) 

v_lshl_add_u32 v15, s6, 3, v16 // (op 1031) 


v_mul_lo_u32 v16, v15, s28 // (op 1031) 

v_add_u32 v15, v14, v16 // (op 1031) 


s_mul_i32 s1, s28, s33 // (op 1033) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1033) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1033) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1033) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1033) most significant: sum
s_add_u32 s1, s1, s7 // (op 1033) most significant: sum


s_mov_b32 s52, s30 // (op 1033) reg expression
s_mov_b32 s53, s31 // (op 1033) reg expression
s_mov_b32 s54, s0 // (op 1033) reg expression
s_mov_b32 s55, 131072 // (op 1033) call()

v_and_b32 v14, 63, v1 // (op 1095) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1095) 
v_lshrrev_b32 v16, 4, v14 // (op 1095) 

s_lshr_b32 s6, s0, 1 // (op 1095) 

v_and_b32 v14, 15, v1 // (op 1095) 
v_lshl_add_u32 v17, s6, 3, v16 // (op 1095) 


v_lshl_add_u32 v16, s17, 4, v14 // (op 1095) 

v_mul_lo_u32 v14, v17, s34 // (op 1095) 
v_add_u32 v14, v14, v16 // (op 1095) 


s_mul_i32 s1, s34, s33 // (op 1097) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1097) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1097) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1097) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1097) most significant: sum
s_add_u32 s1, s1, s7 // (op 1097) most significant: sum


s_mov_b32 s56, s36 // (op 1097) reg expression
s_mov_b32 s57, s37 // (op 1097) reg expression
s_mov_b32 s58, s0 // (op 1097) reg expression
s_mov_b32 s59, 131072 // (op 1097) call()

v_and_b32 v16, -4, v1 // (op 1188) 
v_sub_u32 v17, v1, v16 // (op 1188) 

v_lshlrev_b32 v16, 4, v17 // (op 1188) 

v_lshrrev_b32 v17, 2, v1 // (op 1188) 
v_lshl_add_u32 v18, v17, 6, v16 // (op 1188) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 1188) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v16, 4, v1 // (op 1245) 
v_mul_lo_u32 v17, v16, 6 // (op 1245) 

v_lshrrev_b32 v16, 3, v17 // (op 1245) 

v_and_b32 v17, 268435455, v1 // (op 1245) 
v_lshlrev_b32 v18, 2, v17 // (op 1245) 

v_add_u32 v17, v16, v18 // (op 1245) 


v_and_b32 v16, -64, v1 // (op 1298) 
v_sub_u32 v18, v1, v16 // (op 1298) 

v_lshrrev_b32 v16, 6, v1 // (op 1298) 
v_lshl_add_u32 v19, v16, 7, v18 // (op 1298) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 316) 
s_and_b32 s1, s6, 4294967294 // (op 316) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 315) 
s_mov_b32 s14, s6 // (op 315) convert

s_cmp_lt_u32 s1, s14 // (op 315) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_315 // (op 315) Condition: Top (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_315 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_315:
s_barrier  // (op 1493) 
s_add_u32 s1, s1, 1 // (op 317) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 870) 
v_mov_b64 v[20:21], s[6:7] // (op 870) convert


v_mov_b64 v[22:23], v[12:13] // (op 870) 
v_mov_b64 v[24:25], v[22:23] // (op 870) reg expression
s_mov_b32 m0, s0 // (op 870) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 870) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v20 // (op 870) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 870) most significant half

s_add_u32 m0, m0, 2048 // (op 870) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 870) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v20 // (op 870) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 870) most significant half

s_add_u32 m0, m0, 2048 // (op 870) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 870) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v20 // (op 870) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 870) most significant half

s_add_u32 m0, m0, 2048 // (op 870) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 870) Load value direct to lds



s_lshl_b64 s[6:7], s[12:13], 7 // (op 889) 
v_mov_b32 v16, s6 // (op 889) 
v_mov_b32 v18, s7 // (op 889) 
v_add_co_u32 v12, s[14:15], v12, v16 // (op 889) least significant half
v_addc_co_u32 v13, s[14:15], v13, v18, s[14:15] // (op 889) most significant half






v_mov_b32 v16, v33 // (op 179) 
v_mov_b32 v18, v16 // (op 179) reg expression

buffer_load_dwordx3 v[20:22], v18, s[48:51], 0 offen // (op 179) Load value


s_lshl_b64 s[6:7], s[24:25], 7 // (op 959) 
s_mul_i32 s15, s7, 6 // (op 959) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 959) least significant
s_mul_hi_u32 s18, s6, 6 // (op 959) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 959) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 959) 

v_add_u32 v33, v33, s6 // (op 959) 


s_barrier  // (op 1506) 
v_mov_b64 v[24:25], s[28:29] // (op 180) convert




v_mov_b32 v16, v15 // (op 180) 
v_mov_b32 v18, v16 // (op 180) reg expression
buffer_load_ubyte v26, v18, s[52:55], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v18, v18, v24 // (op 180) 
s_waitcnt vmcnt(1)
ds_write_b96 v17, v[20:22] offset:16384 // (op 177) Store local data 
buffer_load_ubyte v27, v18, s[52:55], 0 offen offset: 0 // (op 180) Load value




s_lshl_b32 s6, s28, 2 // (op 1023) 
v_add_u32 v15, v15, s6 // (op 1023) 

s_barrier  // (op 1496) 
s_barrier  // (op 1520) 






ds_read_b64_tr_b8 v[34:35], v0 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:512 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:4096 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:4608 // (op 169) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v26 offset:18432 // (op 178) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1510) 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v27 offset:18496 // (op 178) Store local data 




ds_read_b96_tr_b6 v[20:22], v4 offset:16384 // (op 171) Transpose load local data 
ds_read_b96_tr_b6 v[23:25], v4 offset:16640 // (op 171) Transpose load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1524) 


ds_read_u8 v16, v6 offset:18432 // (op 173) Load local data 

v_mov_b64 v[26:27], s[34:35] // (op 175) convert

v_mov_b32 v18, v14 // (op 175) 
v_mov_b32 v28, v18 // (op 175) reg expression
v_mov_b32 v29, v28 // (op 175) 
buffer_load_ubyte v30, v29, s[56:59], 0 offen offset: 0 // (op 175) Load value




s_lshl_b32 s6, s34, 2 // (op 1087) 
v_add_u32 v14, v14, s6 // (op 1087) 




ds_read_b64_tr_b8 v[42:43], v0 offset:32 // (op 321) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:544 // (op 321) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[20:25], a[0:3], v16, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 176) 
ds_read_b64_tr_b8 v[46:47], v0 offset:4128 // (op 321) Transpose load local data 


ds_read_b64_tr_b8 v[48:49], v0 offset:4640 // (op 321) Transpose load local data 


ds_read_u8 v16, v6 offset:18464 // (op 325) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[20:25], a[4:7], v16, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 327) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 315) 
s_mov_b32 s14, s6 // (op 315) convert

s_cmp_lt_u32 s1, s14 // (op 315) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_315 // (op 315) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_315 if true)
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_315:






















s_barrier  // (op 1596) 






v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_123:



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
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.LRR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_end:
.size RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2, .LRR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_end-RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 20992 // lds bytes
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
  - .name: RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
    .symbol: RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 62
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

