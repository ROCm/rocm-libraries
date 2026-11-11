// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
.p2align 8
.type RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2,@function
RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 1259) 
v_and_b32 v3, 1, v0 // (op 1259) 

v_and_b32 v0, 15, v1 // (op 1259) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 1259) 


v_and_b32 v0, 63, v1 // (op 1259) 
v_lshrrev_b32 v3, 4, v0 // (op 1259) 

v_lshlrev_b32 v0, 5, v3 // (op 1259) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 1259) 


v_lshlrev_b32 v0, 2, v3 // (op 1259) 

v_lshrrev_b32 v3, 3, v0 // (op 1259) 

v_lshrrev_b32 v0, 6, v1 // (op 1388) 
v_and_b32 v4, 1, v0 // (op 1388) 

v_and_b32 v0, 15, v1 // (op 1388) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 1388) 


v_and_b32 v0, 63, v1 // (op 1388) 
v_lshrrev_b32 v4, 4, v0 // (op 1388) 

v_lshlrev_b32 v0, 5, v4 // (op 1388) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 1388) 


v_lshlrev_b32 v0, 2, v4 // (op 1388) 

v_lshrrev_b32 v4, 3, v0 // (op 1388) 

v_lshrrev_b32 v0, 7, v1 // (op 1144) 
v_and_b32 v5, 63, v1 // (op 1144) 
v_lshrrev_b32 v6, 4, v5 // (op 1144) 
v_lshlrev_b32 v7, 2, v6 // (op 1144) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1144) 


v_lshl_add_u32 v0, s16, 6, v6 // (op 1144) 

v_lshrrev_b32 v6, 2, v5 // (op 1144) 

v_and_b32 v5, 3, v6 // (op 1144) 

v_and_b32 v6, 3, v1 // (op 1144) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1144) 


v_lshrrev_b32 v5, 6, v1 // (op 1144) 
v_and_b32 v6, 1, v5 // (op 1144) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1144) 


v_lshl_add_u32 v6, s17, 5, v5 // (op 1144) 

v_mul_lo_u32 v5, v6, s2 // (op 1144) 

v_add_lshl_u32 v6, v0, v5, 1 // (op 1144) 


v_and_b32 v0, 63, v1 // (op 1206) 
v_and_b32 v5, 48, v1 // (op 1206) 
v_sub_u32 v7, v0, v5 // (op 1206) 

v_lshrrev_b32 v5, 4, v0 // (op 1206) 

v_lshl_add_u32 v0, v5, 5, v7 // (op 1206) 


v_lshrrev_b32 v5, 7, v1 // (op 1206) 
v_lshlrev_b32 v7, 4, v5 // (op 1206) 

v_lshl_add_u32 v5, v0, 6, v7 // (op 1206) 


v_mul_lo_u32 v0, v5, 6 // (op 1206) 
v_lshrrev_b32 v7, 3, v0 // (op 1206) 

v_lshrrev_b32 v0, 4, v5 // (op 1206) 

v_lshlrev_b32 v5, 2, v0 // (op 1206) 

v_add_u32 v0, v7, v5 // (op 1206) 


v_lshrrev_b32 v5, 7, v1 // (op 1317) 
v_and_b32 v7, 15, v1 // (op 1317) 
s_mul_i32 s1, s2, s9 // (op 1146) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1146) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1146) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1146) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1146) most significant: sum
s_add_u32 s1, s1, s19 // (op 1146) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1146) 

v_lshl_add_u32 v8, v5, 4, v7 // (op 1317) 


v_and_b32 v5, 63, v1 // (op 1317) 
v_lshrrev_b32 v7, 4, v5 // (op 1317) 

v_lshl_add_u32 v5, v7, 6, v8 // (op 1317) 


v_lshrrev_b32 v7, 7, v1 // (op 1333) 
v_and_b32 v8, 63, v1 // (op 1333) 
v_lshrrev_b32 v9, 4, v8 // (op 1333) 
v_lshlrev_b32 v10, 2, v9 // (op 1333) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1333) 


v_lshl_add_u32 v7, s16, 6, v9 // (op 1333) 

v_lshrrev_b32 v9, 2, v8 // (op 1333) 

v_and_b32 v8, 3, v9 // (op 1333) 

v_and_b32 v9, 3, v1 // (op 1333) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1333) 


v_lshrrev_b32 v8, 6, v1 // (op 1333) 
v_and_b32 v9, 1, v8 // (op 1333) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 1333) 


v_lshl_add_u32 v9, s17, 5, v8 // (op 1333) 

v_mul_lo_u32 v8, v9, s4 // (op 1333) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 1333) 


v_and_b32 v7, 63, v1 // (op 1364) 
v_and_b32 v8, 48, v1 // (op 1364) 
v_sub_u32 v10, v7, v8 // (op 1364) 

v_lshrrev_b32 v8, 4, v7 // (op 1364) 

v_lshl_add_u32 v7, v8, 5, v10 // (op 1364) 


v_lshrrev_b32 v8, 7, v1 // (op 1364) 
v_lshlrev_b32 v10, 4, v8 // (op 1364) 

v_lshl_add_u32 v8, v7, 6, v10 // (op 1364) 


v_mul_lo_u32 v7, v8, 6 // (op 1364) 
v_lshrrev_b32 v10, 3, v7 // (op 1364) 

v_lshrrev_b32 v7, 4, v8 // (op 1364) 

v_lshlrev_b32 v8, 2, v7 // (op 1364) 

v_add_u32 v7, v10, v8 // (op 1364) 


v_lshrrev_b32 v8, 7, v1 // (op 1417) 
v_and_b32 v10, 15, v1 // (op 1417) 
s_mul_i32 s1, s4, s9 // (op 1335) most significant: low of low * high
v_lshl_add_u32 v11, v8, 4, v10 // (op 1417) 


v_and_b32 v8, 63, v1 // (op 1417) 
v_lshrrev_b32 v10, 4, v8 // (op 1417) 

v_lshl_add_u32 v8, v10, 6, v11 // (op 1417) 


s_mul_i32 s38, s5, s8 // (op 1335) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1335) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1335) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1335) most significant: sum
s_add_u32 s1, s1, s39 // (op 1335) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 1335) 

s_mov_b32 s40, s6 // (op 1146) reg expression
s_mov_b32 s41, s7 // (op 1146) reg expression
s_mov_b32 s42, s18 // (op 1146) reg expression
s_mov_b32 s43, 131072 // (op 1146) call()

s_mov_b32 s44, s10 // (op 1335) reg expression

s_mov_b32 s45, s11 // (op 1335) reg expression
s_mov_b32 s46, s38 // (op 1335) reg expression
s_mov_b32 s47, 131072 // (op 1335) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 313) call()
v_accvgpr_write a5, 0.00000 // (op 313) call()
v_accvgpr_write a6, 0.00000 // (op 313) call()
v_accvgpr_write a7, 0.00000 // (op 313) call()
v_and_b32 v12, -2, v1 // (op 911) 
v_sub_u32 v13, v1, v12 // (op 911) 

v_lshlrev_b32 v12, 5, v13 // (op 911) 

v_lshl_add_u32 v13, s16, 6, v12 // (op 911) 

v_mov_b32 v15, 0 // (op 911) convert
v_mov_b32 v14, v13 // (op 911) convert

v_lshrrev_b32 v12, 1, v1 // (op 911) 
v_mul_lo_u32 v17, v12, s13 // (op 911) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 911) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 911) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 911) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 911) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 911) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 911) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 911) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 911) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 911) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 911) 

v_mov_b32 v14, v12 // (op 911) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 913) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 913) least significant
s_mul_hi_u32 s6, s22, 6 // (op 913) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 913) most significant: sum

s_mul_i32 s7, s12, s1 // (op 913) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 913) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 913) least significant
s_mul_hi_u32 s11, s12, s0 // (op 913) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 913) most significant: sum
s_add_u32 s7, s7, s11 // (op 913) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 913) 

s_mov_b32 s48, s20 // (op 913) reg expression
s_mov_b32 s49, s21 // (op 913) reg expression
s_mov_b32 s50, s0 // (op 913) reg expression
s_mov_b32 s51, 131072 // (op 913) call()

v_and_b32 v12, -2, v1 // (op 930) 
v_sub_u32 v13, v1, v12 // (op 930) 

v_lshlrev_b32 v12, 5, v13 // (op 930) 

v_lshl_add_u32 v13, s16, 6, v12 // (op 930) 

v_mov_b32 v17, 0 // (op 930) convert
v_mov_b32 v16, v13 // (op 930) convert

v_lshrrev_b32 v15, 1, v1 // (op 930) 
v_mul_lo_u32 v13, v15, s13 // (op 930) most significant: low of low * high
v_mul_hi_u32 v18, v15, s12 // (op 930) most significant: high of low * low
v_mul_lo_u32 v12, v15, s12 // (op 930) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 930) most significant: sum


v_add_co_u32 v18, s[0:1], v16, v12 // (op 930) least significant half
v_addc_co_u32 v19, s[0:1], v17, v13, s[0:1] // (op 930) most significant half



v_mul_lo_u32 v13, v19, 6 // (op 930) most significant: low of high * low
v_mul_hi_u32 v15, v18, 6 // (op 930) most significant: high of low * low
v_mul_lo_u32 v12, v18, 6 // (op 930) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 930) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 930) 

v_mov_b32 v15, v16 // (op 930) convert

s_mul_i32 s1, s23, 6 // (op 932) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 932) least significant
s_mul_hi_u32 s6, s22, 6 // (op 932) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 932) most significant: sum

s_mul_i32 s7, s12, s1 // (op 932) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 932) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 932) least significant
s_mul_hi_u32 s11, s12, s0 // (op 932) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 932) most significant: sum
s_add_u32 s7, s7, s11 // (op 932) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 932) 

s_mov_b32 s48, s20 // (op 932) reg expression
s_mov_b32 s49, s21 // (op 932) reg expression
s_mov_b32 s50, s0 // (op 932) reg expression
s_mov_b32 s51, 131072 // (op 932) call()

v_and_b32 v12, -8, v1 // (op 975) 
v_sub_u32 v13, v1, v12 // (op 975) 

v_lshlrev_b32 v12, 4, v13 // (op 975) 

v_mov_b32 v17, 0 // (op 975) convert
v_mov_b32 v16, v12 // (op 975) convert

v_lshrrev_b32 v12, 3, v1 // (op 975) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 975) 

v_mul_lo_u32 v19, v13, s25 // (op 975) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 975) most significant: high of low * low
v_mul_lo_u32 v18, v13, s24 // (op 975) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 975) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 975) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 975) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 975) 


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 975) 

v_mov_b32 v12, v16 // (op 975) convert

s_mul_i32 s1, s24, s9 // (op 977) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 977) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 977) least significant
s_mul_hi_u32 s7, s24, s8 // (op 977) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 977) most significant: sum
s_add_u32 s1, s1, s7 // (op 977) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 977) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 977) 

s_mov_b32 s52, s26 // (op 977) reg expression
s_mov_b32 s53, s27 // (op 977) reg expression
s_mov_b32 s54, s0 // (op 977) reg expression
s_mov_b32 s55, 131072 // (op 977) call()

v_and_b32 v13, -8, v1 // (op 994) 
v_sub_u32 v16, v1, v13 // (op 994) 

v_lshlrev_b32 v13, 4, v16 // (op 994) 

v_mov_b32 v17, 0 // (op 994) convert
v_mov_b32 v16, v13 // (op 994) convert

v_lshrrev_b32 v13, 3, v1 // (op 994) 
v_lshl_add_u32 v18, s17, 5, v13 // (op 994) 

v_mul_lo_u32 v21, v18, s25 // (op 994) most significant: low of low * high
v_mul_hi_u32 v13, v18, s24 // (op 994) most significant: high of low * low
v_mul_lo_u32 v20, v18, s24 // (op 994) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 994) most significant: sum


v_add_co_u32 v18, s[0:1], v16, v20 // (op 994) least significant half
v_addc_co_u32 v19, s[0:1], v17, v21, s[0:1] // (op 994) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 994) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 994) 

v_mov_b32 v13, v16 // (op 994) convert

s_mul_i32 s1, s24, s9 // (op 996) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 996) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 996) least significant
s_mul_hi_u32 s7, s24, s8 // (op 996) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 996) most significant: sum
s_add_u32 s1, s1, s7 // (op 996) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 996) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 996) 

s_mov_b32 s52, s26 // (op 996) reg expression
s_mov_b32 s53, s27 // (op 996) reg expression
s_mov_b32 s54, s0 // (op 996) reg expression
s_mov_b32 s55, 131072 // (op 996) call()

v_and_b32 v16, -64, v1 // (op 1039) 
v_sub_u32 v17, v1, v16 // (op 1039) 

v_lshl_add_u32 v16, s16, 6, v17 // (op 1039) 

v_lshrrev_b32 v17, 6, v1 // (op 1039) 
v_mul_lo_u32 v18, v17, s28 // (op 1039) 

v_add_u32 v17, v16, v18 // (op 1039) 


s_mul_i32 s1, s28, s33 // (op 1041) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1041) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1041) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1041) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1041) most significant: sum
s_add_u32 s1, s1, s7 // (op 1041) most significant: sum


s_mov_b32 s56, s30 // (op 1041) reg expression
s_mov_b32 s57, s31 // (op 1041) reg expression
s_mov_b32 s58, s0 // (op 1041) reg expression
s_mov_b32 s59, 131072 // (op 1041) call()

v_and_b32 v16, -64, v1 // (op 1058) 
v_sub_u32 v18, v1, v16 // (op 1058) 

v_lshl_add_u32 v16, s16, 6, v18 // (op 1058) 

v_lshrrev_b32 v18, 6, v1 // (op 1058) 
v_mul_lo_u32 v19, v18, s28 // (op 1058) 

v_add_u32 v18, v16, v19 // (op 1058) 


s_mul_i32 s1, s28, s33 // (op 1060) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1060) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1060) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1060) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1060) most significant: sum
s_add_u32 s1, s1, s7 // (op 1060) most significant: sum


s_mov_b32 s56, s30 // (op 1060) reg expression
s_mov_b32 s57, s31 // (op 1060) reg expression
s_mov_b32 s58, s0 // (op 1060) reg expression
s_mov_b32 s59, 131072 // (op 1060) call()

v_lshrrev_b32 v16, 6, v1 // (op 1103) 
v_and_b32 v19, 1, v16 // (op 1103) 

v_and_b32 v16, 15, v1 // (op 1103) 
v_lshl_add_u32 v20, v19, 4, v16 // (op 1103) 


v_lshl_add_u32 v16, s17, 5, v20 // (op 1103) 

v_mul_lo_u32 v19, v16, s34 // (op 1103) 

v_and_b32 v16, 63, v1 // (op 1103) 
v_lshrrev_b32 v20, 4, v16 // (op 1103) 

v_add_u32 v16, v20, v19 // (op 1103) 


s_mul_i32 s1, s34, s9 // (op 1105) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1105) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1105) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1105) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1105) most significant: sum
s_add_u32 s1, s1, s7 // (op 1105) most significant: sum


s_mov_b32 s60, s36 // (op 1105) reg expression
s_mov_b32 s61, s37 // (op 1105) reg expression
s_mov_b32 s62, s0 // (op 1105) reg expression
s_mov_b32 s63, 131072 // (op 1105) call()

v_lshrrev_b32 v19, 6, v1 // (op 1122) 
v_and_b32 v20, 1, v19 // (op 1122) 

v_and_b32 v19, 15, v1 // (op 1122) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 1122) 


v_lshl_add_u32 v19, s17, 5, v21 // (op 1122) 

v_mul_lo_u32 v20, v19, s34 // (op 1122) 

v_and_b32 v19, 63, v1 // (op 1122) 
v_lshrrev_b32 v21, 4, v19 // (op 1122) 

v_add_u32 v19, v21, v20 // (op 1122) 


s_mul_i32 s1, s34, s9 // (op 1124) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1124) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1124) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1124) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1124) most significant: sum
s_add_u32 s1, s1, s7 // (op 1124) most significant: sum


s_mov_b32 s60, s36 // (op 1124) reg expression
s_mov_b32 s61, s37 // (op 1124) reg expression
s_mov_b32 s62, s0 // (op 1124) reg expression
s_mov_b32 s63, 131072 // (op 1124) call()




buffer_load_ubyte v20, v16, s[60:63], 0 offen offset: 0 // (op 782) Load value



buffer_load_dwordx2 v[22:23], v12, s[52:55], 0 offen // (op 786) Load value


v_mov_b32 v21, v14 // (op 789) reg expression

buffer_load_dwordx3 v[24:26], v21, s[48:51], 0 offen // (op 789) Load value
buffer_load_dwordx3 v[28:30], v21, s[48:51], 0 offen offset:12 // (op 789) Load value



v_mov_b32 v21, v17 // (op 792) reg expression

buffer_load_ubyte v27, v21, s[56:59], 0 offen offset: 0 // (op 792) Load value




buffer_load_ubyte v21, v16, s[60:63], 0 offen offset:4 // (op 796) Load value




buffer_load_dwordx2 v[32:33], v12, s[52:55], 0 offen offset:64 // (op 800) Load value



s_lshl_b64 s[0:1], s[12:13], 7 // (op 803) 
s_mul_i32 s7, s1, 6 // (op 803) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 803) least significant
s_mul_hi_u32 s10, s0, 6 // (op 803) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 803) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 803) 

v_add_u32 v12, v14, s0 // (op 803) 


buffer_load_dwordx3 v[34:36], v12, s[48:51], 0 offen // (op 803) Load value
buffer_load_dwordx3 v[38:40], v12, s[48:51], 0 offen offset:12 // (op 803) Load value




s_lshl_b32 s0, s28, 2 // (op 806) 
v_add_u32 v12, v17, s0 // (op 806) 


buffer_load_ubyte v14, v12, s[56:59], 0 offen offset: 0 // (op 806) Load value


v_and_b32 v12, -8, v1 // (op 1247) 
v_sub_u32 v31, v1, v12 // (op 1247) 

v_lshlrev_b32 v12, 4, v31 // (op 1247) 

v_lshrrev_b32 v31, 3, v1 // (op 1247) 
v_lshl_add_u32 v37, v31, 7, v12 // (op 1247) 


v_lshlrev_b32 v12, 2, v37 // (op 1247) 

v_lshrrev_b32 v31, 3, v12 // (op 1247) 




s_waitcnt vmcnt(8)
ds_write_b64 v31, v[22:23] // (op 808) Store local data 

v_and_b32 v12, -2, v1 // (op 1189) 
v_sub_u32 v31, v1, v12 // (op 1189) 

v_lshlrev_b32 v12, 5, v31 // (op 1189) 

v_lshrrev_b32 v31, 1, v1 // (op 1189) 
v_lshl_add_u32 v37, v31, 6, v12 // (op 1189) 


v_mul_lo_u32 v12, v37, 6 // (op 1189) 
v_lshrrev_b32 v31, 3, v12 // (op 1189) 

v_lshrrev_b32 v12, 4, v37 // (op 1189) 

v_lshlrev_b32 v37, 2, v12 // (op 1189) 

v_add_u32 v12, v31, v37 // (op 1189) 





s_waitcnt vmcnt(7)
ds_write_b96 v12, v[24:26] offset:2048 // (op 809) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v12, v[28:30] offset:2064 // (op 809) Store local data 

v_and_b32 v12, -64, v1 // (op 1300) 
v_sub_u32 v31, v1, v12 // (op 1300) 

v_and_b32 v12, 4294967232, v1 // (op 1300) 
v_add_u32 v37, v12, v31 // (op 1300) 





s_waitcnt vmcnt(5)
ds_write_b8 v37, v27 offset:10240 // (op 810) Store local data 

v_and_b32 v12, -2, v1 // (op 1160) 
v_sub_u32 v31, v1, v12 // (op 1160) 

v_lshlrev_b32 v12, 5, v31 // (op 1160) 

v_lshrrev_b32 v31, 1, v1 // (op 1160) 
v_lshl_add_u32 v37, v31, 6, v12 // (op 1160) 


v_mul_lo_u32 v12, v37, 6 // (op 1160) 
v_lshrrev_b32 v31, 3, v12 // (op 1160) 

v_lshrrev_b32 v12, 4, v37 // (op 1160) 

v_lshlrev_b32 v37, 2, v12 // (op 1160) 

v_add_u32 v12, v31, v37 // (op 1160) 


v_and_b32 v31, -8, v1 // (op 1218) 
v_sub_u32 v37, v1, v31 // (op 1218) 

v_lshlrev_b32 v31, 4, v37 // (op 1218) 

v_lshrrev_b32 v37, 3, v1 // (op 1218) 
v_lshl_add_u32 v16, v37, 7, v31 // (op 1218) 


v_lshlrev_b32 v17, 2, v16 // (op 1218) 

v_lshrrev_b32 v16, 3, v17 // (op 1218) 

v_and_b32 v17, -64, v1 // (op 1271) 
v_sub_u32 v31, v1, v17 // (op 1271) 

v_and_b32 v17, 4294967232, v1 // (op 1271) 
v_add_u32 v37, v17, v31 // (op 1271) 


v_and_b32 v17, -2, v1 // (op 1347) 
v_sub_u32 v31, v1, v17 // (op 1347) 

v_lshlrev_b32 v17, 5, v31 // (op 1347) 

v_lshrrev_b32 v31, 1, v1 // (op 1347) 
v_lshl_add_u32 v41, v31, 6, v17 // (op 1347) 


v_mul_lo_u32 v17, v41, 6 // (op 1347) 
v_lshrrev_b32 v31, 3, v17 // (op 1347) 

v_lshrrev_b32 v17, 4, v41 // (op 1347) 

v_lshlrev_b32 v41, 2, v17 // (op 1347) 

v_add_u32 v17, v31, v41 // (op 1347) 


v_and_b32 v31, -8, v1 // (op 1376) 
v_sub_u32 v41, v1, v31 // (op 1376) 

v_lshlrev_b32 v31, 4, v41 // (op 1376) 

v_lshrrev_b32 v41, 3, v1 // (op 1376) 
v_lshl_add_u32 v42, v41, 7, v31 // (op 1376) 


v_lshlrev_b32 v31, 2, v42 // (op 1376) 

v_lshrrev_b32 v41, 3, v31 // (op 1376) 

v_and_b32 v31, -64, v1 // (op 1400) 
v_sub_u32 v42, v1, v31 // (op 1400) 

v_and_b32 v31, 4294967232, v1 // (op 1400) 
v_add_u32 v43, v31, v42 // (op 1400) 


s_mov_b32 s0, 0 // (op 315) call()
s_ashr_i64 s[6:7], s[22:23], 7 // (op 314) 
s_mov_b32 s1, s6 // (op 314) convert

s_and_b32 s6, s1, 4294967294 // (op 314) 

s_cmp_lt_u32 s0, s6 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_314 // (op 314) Condition: Top (jump to RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_314 if false)
RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoop_314:
s_add_u32 s0, s0, 2 // (op 316) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1458) 



ds_read_b96_tr_b6 v[44:46], v0 offset:2048 // (op 67) Transpose load local data 
ds_read_b96_tr_b6 v[47:49], v0 offset:3072 // (op 67) Transpose load local data 
s_barrier  // (op 1471) 



ds_read_b128 v[50:53], v3 // (op 72) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1485) 


ds_read_u8 v31, v5 offset:10240 // (op 77) Load local data 



ds_read_b96_tr_b6 v[54:56], v0 offset:2080 // (op 344) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:49], v[50:53], a[0:3], v31, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 84) 
ds_read_b96_tr_b6 v[57:59], v0 offset:3104 // (op 344) Transpose load local data 




ds_read_u8 v31, v5 offset:10272 // (op 348) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[50:53], a[4:7], v31, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 350) 





v_mov_b32 v31, v19 // (op 82) 

buffer_load_ubyte v20, v31, s[60:63], 0 offen offset:8 // (op 82) Load value


v_mov_b32 v42, v13 // (op 10) 

buffer_load_dwordx2 v[22:23], v42, s[52:55], 0 offen offset:128 // (op 10) Load value


v_mov_b32 v44, v15 // (op 4) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 4) 
s_mul_i32 s11, s7, 6 // (op 4) most significant: low of high * low
s_mul_i32 s10, s6, 6 // (op 4) least significant
s_mul_hi_u32 s1, s6, 6 // (op 4) most significant: high of low * low
s_add_u32 s11, s11, s1 // (op 4) most significant: sum


s_lshr_b64 s[6:7], s[10:11], 3 // (op 4) 

s_lshl_b32 s1, s6, 1 // (op 4) 

v_add_u32 v45, v44, s1 // (op 4) 


buffer_load_dwordx3 v[24:26], v45, s[48:51], 0 offen // (op 4) Load value
buffer_load_dwordx3 v[28:30], v45, s[48:51], 0 offen offset:12 // (op 4) Load value



v_mov_b32 v45, v18 // (op 17) 
s_lshl_b32 s1, s28, 3 // (op 17) 
v_add_u32 v46, v45, s1 // (op 17) 


buffer_load_ubyte v27, v46, s[56:59], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1508) 



s_waitcnt vmcnt(8)
ds_write_b64 v41, v[32:33] offset:10496 // (op 239) Store local data 
s_barrier  // (op 1502) 



s_waitcnt vmcnt(7)
ds_write_b96 v17, v[34:36] offset:12544 // (op 229) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v17, v[38:40] offset:12560 // (op 229) Store local data 
s_barrier  // (op 1514) 



s_waitcnt vmcnt(5)
ds_write_b8 v43, v14 offset:20736 // (op 240) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1499) 



ds_read_b96_tr_b6 v[46:48], v7 offset:12544 // (op 231) Transpose load local data 
ds_read_b96_tr_b6 v[49:51], v7 offset:13568 // (op 231) Transpose load local data 
s_barrier  // (op 1505) 



ds_read_b128 v[52:55], v4 offset:10496 // (op 233) Load local data 
s_barrier  // (op 1555) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1511) 


ds_read_u8 v56, v8 offset:20736 // (op 235) Load local data 



ds_read_b96_tr_b6 v[58:60], v7 offset:12576 // (op 357) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:51], v[52:55], a[0:3], v56, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 238) 
ds_read_b96_tr_b6 v[61:63], v7 offset:13600 // (op 357) Transpose load local data 




s_barrier  // (op 1589) 
ds_read_u8 v46, v8 offset:20768 // (op 361) Load local data 
s_barrier  // (op 1593) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:63], v[52:55], a[4:7], v46, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 351) 






buffer_load_ubyte v21, v31, s[60:63], 0 offen offset:12 // (op 237) Load value




v_add_u32 v19, 8, v19 // (op 1114) 
buffer_load_dwordx2 v[32:33], v42, s[52:55], 0 offen offset:192 // (op 241) Load value



v_add_u32 v13, 128, v13 // (op 986) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 228) 
s_mul_i32 s11, s7, 6 // (op 228) most significant: low of high * low
s_mul_i32 s10, s6, 6 // (op 228) least significant
s_mul_hi_u32 s1, s6, 6 // (op 228) most significant: high of low * low
s_add_u32 s11, s11, s1 // (op 228) most significant: sum


s_lshr_b64 s[6:7], s[10:11], 3 // (op 228) 

s_mul_i32 s1, 3, s6 // (op 228) 

v_add_u32 v31, v44, s1 // (op 228) 


buffer_load_dwordx3 v[34:36], v31, s[48:51], 0 offen // (op 228) Load value
buffer_load_dwordx3 v[38:40], v31, s[48:51], 0 offen offset:12 // (op 228) Load value




s_lshl_b32 s1, s28, 2 // (op 242) 
s_mul_i32 s6, 3, s1 // (op 242) 

v_add_u32 v31, v45, s6 // (op 242) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 922) 
s_mul_i32 s11, s7, 6 // (op 922) most significant: low of high * low
s_mul_i32 s10, s6, 6 // (op 922) least significant
s_mul_hi_u32 s1, s6, 6 // (op 922) most significant: high of low * low
s_add_u32 s11, s11, s1 // (op 922) most significant: sum


s_lshr_b64 s[6:7], s[10:11], 3 // (op 922) 

s_lshl_b32 s1, s6, 1 // (op 922) 

v_add_u32 v15, v15, s1 // (op 922) 

buffer_load_ubyte v14, v31, s[56:59], 0 offen offset: 0 // (op 242) Load value


s_barrier  // (op 1474) 
s_lshl_b32 s1, s28, 3 // (op 1050) 



v_add_u32 v18, v18, s1 // (op 1050) 

s_waitcnt vmcnt(8)
ds_write_b64 v16, v[22:23] // (op 13) Store local data 
s_barrier  // (op 1461) 



s_waitcnt vmcnt(7)
ds_write_b96 v12, v[24:26] offset:2048 // (op 6) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v12, v[28:30] offset:2064 // (op 6) Store local data 
s_barrier  // (op 1488) 



s_waitcnt vmcnt(5)
ds_write_b8 v37, v27 offset:10240 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[22:23], 7 // (op 314) 
s_mov_b32 s1, s6 // (op 314) convert

s_and_b32 s6, s1, 4294967294 // (op 314) 

s_cmp_lt_u32 s0, s6 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoop_314 // (op 314) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoop_314 if true)
RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_314:















v_and_b32 v37, -2, v1 // (op 884) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 884) 
v_sub_u32 v12, v1, v37 // (op 884) 

s_lshr_b32 s6, s0, 1 // (op 884) 

v_lshlrev_b32 v13, 5, v12 // (op 884) 

v_lshl_add_u32 v12, s16, 6, v13 // (op 884) 

v_mov_b32 v23, 0 // (op 884) convert
v_mov_b32 v22, v12 // (op 884) convert

v_lshrrev_b32 v37, 1, v1 // (op 884) 
v_lshl_add_u32 v12, s6, 8, v37 // (op 884) 


v_mul_lo_u32 v25, v12, s13 // (op 884) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 884) most significant: high of low * low
v_mul_lo_u32 v24, v12, s12 // (op 884) least significant: low of low * low
v_add_u32 v25, v25, v13 // (op 884) most significant: sum


v_add_co_u32 v12, s[0:1], v22, v24 // (op 884) least significant half
v_addc_co_u32 v13, s[0:1], v23, v25, s[0:1] // (op 884) most significant half



v_mul_lo_u32 v23, v13, 6 // (op 884) most significant: low of high * low
v_mul_hi_u32 v37, v12, 6 // (op 884) most significant: high of low * low
v_mul_lo_u32 v22, v12, 6 // (op 884) least significant: low of low * low
v_add_u32 v23, v23, v37 // (op 884) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[22:23] // (op 884) 

v_mov_b32 v37, v12 // (op 884) convert

s_mul_i32 s1, s23, 6 // (op 886) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 886) least significant
s_mul_hi_u32 s6, s22, 6 // (op 886) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 886) most significant: sum

s_mul_i32 s7, s12, s1 // (op 886) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 886) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 886) least significant
s_mul_hi_u32 s11, s12, s0 // (op 886) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 886) most significant: sum
s_add_u32 s7, s7, s11 // (op 886) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 886) 

s_mov_b32 s48, s20 // (op 886) reg expression
s_mov_b32 s49, s21 // (op 886) reg expression
s_mov_b32 s50, s0 // (op 886) reg expression
s_mov_b32 s51, 131072 // (op 886) call()

v_and_b32 v12, -8, v1 // (op 952) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 952) 
v_sub_u32 v13, v1, v12 // (op 952) 

s_lshr_b32 s6, s0, 1 // (op 952) 

v_lshlrev_b32 v12, 4, v13 // (op 952) 

v_lshl_add_u32 v13, s6, 8, v12 // (op 952) 


v_mov_b32 v23, 0 // (op 952) convert
v_mov_b32 v22, v13 // (op 952) convert

v_lshrrev_b32 v12, 3, v1 // (op 952) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 952) 

v_mul_lo_u32 v25, v13, s25 // (op 952) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 952) most significant: high of low * low
v_mul_lo_u32 v24, v13, s24 // (op 952) least significant: low of low * low
v_add_u32 v25, v25, v12 // (op 952) most significant: sum


v_add_co_u32 v12, s[0:1], v22, v24 // (op 952) least significant half
v_addc_co_u32 v13, s[0:1], v23, v25, s[0:1] // (op 952) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 952) 


v_lshrrev_b64 v[22:23], 3, v[12:13] // (op 952) 

v_mov_b32 v12, v22 // (op 952) convert

s_mul_i32 s1, s24, s9 // (op 954) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 954) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 954) least significant
s_mul_hi_u32 s7, s24, s8 // (op 954) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 954) most significant: sum
s_add_u32 s1, s1, s7 // (op 954) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 954) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 954) 

s_mov_b32 s52, s26 // (op 954) reg expression
s_mov_b32 s53, s27 // (op 954) reg expression
s_mov_b32 s54, s0 // (op 954) reg expression
s_mov_b32 s55, 131072 // (op 954) call()

v_and_b32 v13, -64, v1 // (op 1016) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1016) 
v_sub_u32 v15, v1, v13 // (op 1016) 

s_lshr_b32 s6, s0, 1 // (op 1016) 

v_lshl_add_u32 v13, s16, 6, v15 // (op 1016) 

v_lshrrev_b32 v15, 6, v1 // (op 1016) 
v_lshl_add_u32 v16, s6, 3, v15 // (op 1016) 


v_mul_lo_u32 v15, v16, s28 // (op 1016) 

v_add_u32 v16, v13, v15 // (op 1016) 


s_mul_i32 s1, s28, s33 // (op 1018) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1018) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1018) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1018) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1018) most significant: sum
s_add_u32 s1, s1, s7 // (op 1018) most significant: sum


s_mov_b32 s56, s30 // (op 1018) reg expression
s_mov_b32 s57, s31 // (op 1018) reg expression
s_mov_b32 s58, s0 // (op 1018) reg expression
s_mov_b32 s59, 131072 // (op 1018) call()

v_and_b32 v13, 63, v1 // (op 1080) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1080) 
v_lshrrev_b32 v15, 4, v13 // (op 1080) 

s_lshr_b32 s6, s0, 1 // (op 1080) 

v_lshrrev_b32 v13, 6, v1 // (op 1080) 
v_lshl_add_u32 v17, s6, 3, v15 // (op 1080) 


v_and_b32 v15, 1, v13 // (op 1080) 

v_and_b32 v13, 15, v1 // (op 1080) 
v_lshl_add_u32 v18, v15, 4, v13 // (op 1080) 


v_lshl_add_u32 v13, s17, 5, v18 // (op 1080) 

v_mul_lo_u32 v15, v13, s34 // (op 1080) 

v_add_u32 v13, v17, v15 // (op 1080) 


s_mul_i32 s1, s34, s9 // (op 1082) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1082) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1082) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1082) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1082) most significant: sum
s_add_u32 s1, s1, s7 // (op 1082) most significant: sum


s_mov_b32 s60, s36 // (op 1082) reg expression
s_mov_b32 s61, s37 // (op 1082) reg expression
s_mov_b32 s62, s0 // (op 1082) reg expression
s_mov_b32 s63, 131072 // (op 1082) call()

v_and_b32 v15, -2, v1 // (op 1173) 
v_sub_u32 v17, v1, v15 // (op 1173) 

v_lshlrev_b32 v15, 5, v17 // (op 1173) 

v_lshrrev_b32 v17, 1, v1 // (op 1173) 
v_lshl_add_u32 v18, v17, 6, v15 // (op 1173) 


v_mul_lo_u32 v15, v18, 6 // (op 1173) 
v_lshrrev_b32 v17, 3, v15 // (op 1173) 

v_lshrrev_b32 v15, 4, v18 // (op 1173) 

v_lshlrev_b32 v18, 2, v15 // (op 1173) 

v_add_u32 v15, v17, v18 // (op 1173) 


v_and_b32 v17, -8, v1 // (op 1231) 
v_sub_u32 v18, v1, v17 // (op 1231) 

v_lshlrev_b32 v17, 4, v18 // (op 1231) 

v_lshrrev_b32 v18, 3, v1 // (op 1231) 
v_lshl_add_u32 v19, v18, 7, v17 // (op 1231) 


v_lshlrev_b32 v20, 2, v19 // (op 1231) 

v_lshrrev_b32 v17, 3, v20 // (op 1231) 

v_and_b32 v18, -64, v1 // (op 1284) 
v_sub_u32 v19, v1, v18 // (op 1284) 

v_and_b32 v18, 4294967232, v1 // (op 1284) 
v_add_u32 v20, v18, v19 // (op 1284) 


s_ashr_i64 s[0:1], s[22:23], 7 // (op 320) 
s_and_b32 s6, s0, 4294967294 // (op 320) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 319) 
s_mov_b32 s7, s0 // (op 319) convert

s_cmp_lt_u32 s6, s7 // (op 319) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_319 // (op 319) Condition: Top (jump to RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_319 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_319:


s_add_u32 s6, s6, 1 // (op 321) 
v_mov_b32 v14, v37 // (op 170) 
v_mov_b32 v18, v14 // (op 170) reg expression

buffer_load_dwordx3 v[22:24], v18, s[48:51], 0 offen // (op 170) Load value
buffer_load_dwordx3 v[26:28], v18, s[48:51], 0 offen offset:12 // (op 170) Load value


s_lshl_b64 s[0:1], s[12:13], 7 // (op 874) 
s_mul_i32 s11, s1, 6 // (op 874) most significant: low of high * low
s_mul_i32 s10, s0, 6 // (op 874) least significant
s_mul_hi_u32 s7, s0, 6 // (op 874) most significant: high of low * low
s_add_u32 s11, s11, s7 // (op 874) most significant: sum


s_lshr_b64 s[0:1], s[10:11], 3 // (op 874) 

v_add_u32 v37, v37, s0 // (op 874) 



s_barrier  // (op 1464) 
v_mov_b32 v14, v12 // (op 183) 




buffer_load_dwordx2 v[18:19], v14, s[52:55], 0 offen // (op 183) Load value

v_add_u32 v12, 64, v12 // (op 944) 
s_waitcnt vmcnt(2)
ds_write_b96 v15, v[22:24] offset:2048 // (op 171) Store local data 


v_mov_b32 v14, v16 // (op 184) 
v_mov_b32 v21, v14 // (op 184) reg expression

buffer_load_ubyte v25, v21, s[56:59], 0 offen offset: 0 // (op 184) Load value


s_lshl_b32 s7, s28, 2 // (op 1008) 
v_add_u32 v16, v16, s7 // (op 1008) 

s_waitcnt vmcnt(2)
ds_write_b96 v15, v[26:28] offset:2064 // (op 171) Store local data 

s_barrier  // (op 1477) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1467) 






ds_read_b96_tr_b6 v[26:28], v0 offset:2048 // (op 173) Transpose load local data 
ds_read_b96_tr_b6 v[29:31], v0 offset:3072 // (op 173) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b64 v17, v[18:19] // (op 181) Store local data 

s_barrier  // (op 1491) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1481) 






ds_read_b128 v[32:35], v3 // (op 175) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v25 offset:10240 // (op 182) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1495) 


ds_read_u8 v14, v5 offset:10240 // (op 177) Load local data 


v_mov_b32 v36, v13 // (op 179) 

buffer_load_ubyte v18, v36, s[60:63], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v13, 4, v13 // (op 1072) 



ds_read_b96_tr_b6 v[38:40], v0 offset:2080 // (op 325) Transpose load local data 
ds_read_b96_tr_b6 v[41:43], v0 offset:3104 // (op 325) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[32:35], a[0:3], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 180) 




ds_read_u8 v14, v5 offset:10272 // (op 329) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:43], v[32:35], a[4:7], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 331) 




s_ashr_i64 s[0:1], s[22:23], 7 // (op 319) 
s_mov_b32 s7, s0 // (op 319) convert

s_cmp_lt_u32 s6, s7 // (op 319) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_319 // (op 319) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_319 if true)
RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_319:






















s_barrier  // (op 1577) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_125
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


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen // (op 54) Store value


v_accvgpr_read v12, a4 // (op 517) 
v_accvgpr_read v13, a5 // (op 517) 
v_accvgpr_read v14, a6 // (op 517) 
v_accvgpr_read v15, a7 // (op 517) 
v_mul_f32 v16, v11, v12 // (op 517) 
v_mul_f32 v17, v11, v13 // (op 517) 
v_mul_f32 v18, v11, v14 // (op 517) 
v_mul_f32 v19, v11, v15 // (op 517) 

v_cvt_f16_f32 v12, v16 // (op 519) 
v_cvt_f16_f32 v13, v17 // (op 519) 
v_cvt_f16_f32 v14, v18 // (op 519) 
v_cvt_f16_f32 v15, v19 // (op 519) 


v_pack_b32_f16 v4, v12, v13 // (op 520) 
v_pack_b32_f16 v5, v14, v15 // (op 520) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:64 // (op 520) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:64 // (op 550) Load value
v_accvgpr_read v12, a4 // (op 546) 
v_accvgpr_read v13, a5 // (op 546) 
v_accvgpr_read v14, a6 // (op 546) 
v_accvgpr_read v15, a7 // (op 546) 
v_mul_f32 v16, v11, v12 // (op 546) 
v_mul_f32 v17, v11, v13 // (op 546) 
v_mul_f32 v18, v11, v14 // (op 546) 
v_mul_f32 v19, v11, v15 // (op 546) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v4 // (op 551) 
v_lshrrev_b32 v13, 16, v4 // (op 551) 
v_cvt_f32_f16 v13, v13 // (op 551) 
v_mul_f32 v20, v10, v12 // (op 551) 
v_mul_f32 v21, v10, v13 // (op 551) 
v_cvt_f32_f16 v12, v5 // (op 551) 
v_lshrrev_b32 v13, 16, v5 // (op 551) 
v_cvt_f32_f16 v13, v13 // (op 551) 
v_mul_f32 v22, v10, v12 // (op 551) 
v_mul_f32 v23, v10, v13 // (op 551) 


v_add_f32 v12, v20, v16 // (op 547) 
v_add_f32 v13, v21, v17 // (op 547) 
v_add_f32 v14, v22, v18 // (op 547) 
v_add_f32 v15, v23, v19 // (op 547) 


v_cvt_f16_f32 v16, v12 // (op 548) 
v_cvt_f16_f32 v17, v13 // (op 548) 
v_cvt_f16_f32 v18, v14 // (op 548) 
v_cvt_f16_f32 v19, v15 // (op 548) 


v_pack_b32_f16 v4, v16, v17 // (op 549) 
v_pack_b32_f16 v5, v18, v19 // (op 549) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:64 // (op 549) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen // (op 138) Load value
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


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
.LRR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_end:
.size RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2, .LRR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_end-RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: F4
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
    macrotile: [64, 32, 128]
    threads: [256, 1, 1]
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
  - .name: RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2
    .symbol: RR_GEMM_NN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 64
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

