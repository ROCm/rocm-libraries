// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.p2align 8
.type RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2,@function
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 15, v1 // (op 1282) 
v_and_b32 v3, 48, v1 // (op 1282) 
v_lshl_add_u32 v4, v0, 7, v3 // (op 1282) 


v_and_b32 v0, 15, v1 // (op 1426) 
v_and_b32 v3, 48, v1 // (op 1426) 
v_lshl_add_u32 v5, v0, 7, v3 // (op 1426) 


v_lshrrev_b32 v0, 6, v1 // (op 1168) 
v_and_b32 v3, 63, v1 // (op 1168) 
v_lshrrev_b32 v6, 4, v3 // (op 1168) 
v_lshlrev_b32 v7, 2, v6 // (op 1168) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1168) 


v_lshl_add_u32 v0, s16, 6, v6 // (op 1168) 

v_lshrrev_b32 v6, 2, v3 // (op 1168) 

v_and_b32 v3, 3, v6 // (op 1168) 

v_and_b32 v6, 3, v1 // (op 1168) 
v_lshl_add_u32 v7, v3, 2, v6 // (op 1168) 


v_lshl_add_u32 v3, s17, 4, v7 // (op 1168) 

v_mul_lo_u32 v6, v3, s2 // (op 1168) 

v_add_lshl_u32 v3, v0, v6, 1 // (op 1168) 


v_and_b32 v0, 63, v1 // (op 1230) 
v_and_b32 v6, 48, v1 // (op 1230) 
v_sub_u32 v7, v0, v6 // (op 1230) 

v_lshrrev_b32 v6, 4, v0 // (op 1230) 

v_lshl_add_u32 v0, v6, 5, v7 // (op 1230) 


v_lshrrev_b32 v6, 6, v1 // (op 1230) 
v_lshlrev_b32 v7, 4, v6 // (op 1230) 

v_lshl_add_u32 v6, v0, 6, v7 // (op 1230) 


v_mul_lo_u32 v0, v6, 6 // (op 1230) 
v_lshrrev_b32 v7, 3, v0 // (op 1230) 

v_lshrrev_b32 v0, 4, v6 // (op 1230) 

v_lshlrev_b32 v6, 2, v0 // (op 1230) 

v_add_u32 v0, v7, v6 // (op 1230) 


v_lshrrev_b32 v6, 6, v1 // (op 1340) 
v_and_b32 v7, 15, v1 // (op 1340) 
s_mul_i32 s1, s2, s9 // (op 1170) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1170) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1170) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1170) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1170) most significant: sum
s_add_u32 s1, s1, s19 // (op 1170) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1170) 

v_lshl_add_u32 v8, v6, 4, v7 // (op 1340) 


v_and_b32 v6, 63, v1 // (op 1340) 
v_lshrrev_b32 v7, 4, v6 // (op 1340) 

v_lshl_add_u32 v6, v7, 6, v8 // (op 1340) 


v_lshrrev_b32 v7, 6, v1 // (op 1356) 
v_and_b32 v8, 63, v1 // (op 1356) 
v_lshrrev_b32 v9, 4, v8 // (op 1356) 
v_lshlrev_b32 v10, 2, v9 // (op 1356) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1356) 


v_lshl_add_u32 v7, s16, 6, v9 // (op 1356) 

v_lshrrev_b32 v9, 2, v8 // (op 1356) 

v_and_b32 v8, 3, v9 // (op 1356) 

v_and_b32 v9, 3, v1 // (op 1356) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1356) 


v_lshl_add_u32 v8, s17, 4, v10 // (op 1356) 

v_mul_lo_u32 v9, v8, s4 // (op 1356) 

v_add_lshl_u32 v8, v7, v9, 1 // (op 1356) 


v_and_b32 v7, 63, v1 // (op 1387) 
v_and_b32 v9, 48, v1 // (op 1387) 
v_sub_u32 v10, v7, v9 // (op 1387) 

v_lshrrev_b32 v9, 4, v7 // (op 1387) 

v_lshl_add_u32 v7, v9, 5, v10 // (op 1387) 


v_lshrrev_b32 v9, 6, v1 // (op 1387) 
v_lshlrev_b32 v10, 4, v9 // (op 1387) 

v_lshl_add_u32 v9, v7, 6, v10 // (op 1387) 


v_mul_lo_u32 v7, v9, 6 // (op 1387) 
v_lshrrev_b32 v10, 3, v7 // (op 1387) 

v_lshrrev_b32 v7, 4, v9 // (op 1387) 

v_lshlrev_b32 v9, 2, v7 // (op 1387) 

v_add_u32 v7, v10, v9 // (op 1387) 


v_lshrrev_b32 v9, 6, v1 // (op 1455) 
v_and_b32 v10, 15, v1 // (op 1455) 
s_mul_i32 s1, s4, s9 // (op 1358) most significant: low of low * high
v_lshl_add_u32 v11, v9, 4, v10 // (op 1455) 


v_and_b32 v9, 63, v1 // (op 1455) 
v_lshrrev_b32 v10, 4, v9 // (op 1455) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 1455) 


s_mul_i32 s38, s5, s8 // (op 1358) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1358) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1358) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1358) most significant: sum
s_add_u32 s1, s1, s39 // (op 1358) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 1358) 

s_mov_b32 s40, s6 // (op 1170) reg expression
s_mov_b32 s41, s7 // (op 1170) reg expression
s_mov_b32 s42, s18 // (op 1170) reg expression
s_mov_b32 s43, 131072 // (op 1170) call()

s_mov_b32 s44, s10 // (op 1358) reg expression

s_mov_b32 s45, s11 // (op 1358) reg expression
s_mov_b32 s46, s38 // (op 1358) reg expression
s_mov_b32 s47, 131072 // (op 1358) call()


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
s_lshl_b32 s0, s16, 6 // (op 935) 
v_mul_lo_u32 v13, v1, s13 // (op 935) most significant: low of low * high
s_mov_b32 s7, 0 // (op 935) convert
s_mov_b32 s6, s0 // (op 935) convert

v_mul_hi_u32 v14, v1, s12 // (op 935) most significant: high of low * low
v_mul_lo_u32 v12, v1, s12 // (op 935) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 935) most significant: sum

v_mov_b32 v14, s6 // (op 935) 
v_mov_b32 v15, s7 // (op 935) 
v_add_co_u32 v16, s[0:1], v14, v12 // (op 935) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 935) most significant half





v_mul_lo_u32 v13, v17, 6 // (op 935) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 935) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 935) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 935) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 935) 

v_mov_b32 v12, v14 // (op 935) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 937) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 937) least significant
s_mul_hi_u32 s6, s22, 6 // (op 937) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 937) most significant: sum

s_mul_i32 s7, s12, s1 // (op 937) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 937) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 937) least significant
s_mul_hi_u32 s11, s12, s0 // (op 937) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 937) most significant: sum
s_add_u32 s7, s7, s11 // (op 937) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 937) 

s_mov_b32 s48, s20 // (op 937) reg expression
s_mov_b32 s49, s21 // (op 937) reg expression
s_mov_b32 s50, s0 // (op 937) reg expression
s_mov_b32 s51, 131072 // (op 937) call()

s_lshl_b32 s0, s16, 6 // (op 954) 
v_mul_lo_u32 v15, v1, s13 // (op 954) most significant: low of low * high
v_mul_hi_u32 v13, v1, s12 // (op 954) most significant: high of low * low
v_mul_lo_u32 v14, v1, s12 // (op 954) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 954) most significant: sum

s_mov_b32 s7, 0 // (op 954) convert
s_mov_b32 s6, s0 // (op 954) convert

v_mov_b32 v13, s6 // (op 954) 
v_mov_b32 v16, s7 // (op 954) 
v_add_co_u32 v18, s[0:1], v13, v14 // (op 954) least significant half
v_addc_co_u32 v19, s[0:1], v16, v15, s[0:1] // (op 954) most significant half





v_mul_lo_u32 v17, v19, 6 // (op 954) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 954) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 954) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 954) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 954) 

v_mov_b32 v13, v14 // (op 954) convert

s_mul_i32 s1, s23, 6 // (op 956) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 956) least significant
s_mul_hi_u32 s6, s22, 6 // (op 956) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 956) most significant: sum

s_mul_i32 s7, s12, s1 // (op 956) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 956) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 956) least significant
s_mul_hi_u32 s11, s12, s0 // (op 956) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 956) most significant: sum
s_add_u32 s7, s7, s11 // (op 956) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 956) 

s_mov_b32 s48, s20 // (op 956) reg expression
s_mov_b32 s49, s21 // (op 956) reg expression
s_mov_b32 s50, s0 // (op 956) reg expression
s_mov_b32 s51, 131072 // (op 956) call()

v_and_b32 v14, -8, v1 // (op 999) 
v_sub_u32 v15, v1, v14 // (op 999) 

v_lshlrev_b32 v14, 4, v15 // (op 999) 

v_mov_b32 v17, 0 // (op 999) convert
v_mov_b32 v16, v14 // (op 999) convert

v_lshrrev_b32 v14, 3, v1 // (op 999) 
v_lshl_add_u32 v15, s17, 4, v14 // (op 999) 

v_mul_lo_u32 v19, v15, s25 // (op 999) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 999) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 999) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 999) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 999) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 999) most significant half



s_mul_i32 s1, s24, s9 // (op 1001) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1001) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1001) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1001) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1001) most significant: sum
s_add_u32 s1, s1, s7 // (op 1001) most significant: sum


s_mov_b32 s52, s26 // (op 1001) reg expression
s_mov_b32 s53, s27 // (op 1001) reg expression
s_mov_b32 s54, s0 // (op 1001) reg expression
s_mov_b32 s55, 131072 // (op 1001) call()

v_and_b32 v16, -8, v1 // (op 1018) 
v_sub_u32 v17, v1, v16 // (op 1018) 

v_lshlrev_b32 v16, 4, v17 // (op 1018) 

v_mov_b32 v19, 0 // (op 1018) convert
v_mov_b32 v18, v16 // (op 1018) convert

v_lshrrev_b32 v16, 3, v1 // (op 1018) 
v_lshl_add_u32 v17, s17, 4, v16 // (op 1018) 

v_mul_lo_u32 v21, v17, s25 // (op 1018) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 1018) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 1018) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 1018) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 1018) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 1018) most significant half



s_mul_i32 s1, s24, s9 // (op 1020) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1020) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1020) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1020) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1020) most significant: sum
s_add_u32 s1, s1, s7 // (op 1020) most significant: sum


s_mov_b32 s52, s26 // (op 1020) reg expression
s_mov_b32 s53, s27 // (op 1020) reg expression
s_mov_b32 s54, s0 // (op 1020) reg expression
s_mov_b32 s55, 131072 // (op 1020) call()

v_and_b32 v18, -64, v1 // (op 1063) 
v_sub_u32 v19, v1, v18 // (op 1063) 

v_lshl_add_u32 v18, s16, 6, v19 // (op 1063) 

v_lshrrev_b32 v19, 6, v1 // (op 1063) 
v_lshlrev_b32 v20, 1, v19 // (op 1063) 

v_mul_lo_u32 v19, v20, s28 // (op 1063) 

v_add_u32 v20, v18, v19 // (op 1063) 


s_mul_i32 s1, s28, s33 // (op 1065) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1065) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1065) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1065) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1065) most significant: sum
s_add_u32 s1, s1, s7 // (op 1065) most significant: sum


s_mov_b32 s56, s30 // (op 1065) reg expression
s_mov_b32 s57, s31 // (op 1065) reg expression
s_mov_b32 s58, s0 // (op 1065) reg expression
s_mov_b32 s59, 131072 // (op 1065) call()

v_and_b32 v18, -64, v1 // (op 1082) 
v_sub_u32 v19, v1, v18 // (op 1082) 

v_lshl_add_u32 v18, s16, 6, v19 // (op 1082) 

v_lshrrev_b32 v19, 6, v1 // (op 1082) 
v_lshlrev_b32 v21, 1, v19 // (op 1082) 

v_mul_lo_u32 v19, v21, s28 // (op 1082) 

v_add_u32 v21, v18, v19 // (op 1082) 


s_mul_i32 s1, s28, s33 // (op 1084) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1084) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1084) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1084) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1084) most significant: sum
s_add_u32 s1, s1, s7 // (op 1084) most significant: sum


s_mov_b32 s56, s30 // (op 1084) reg expression
s_mov_b32 s57, s31 // (op 1084) reg expression
s_mov_b32 s58, s0 // (op 1084) reg expression
s_mov_b32 s59, 131072 // (op 1084) call()

v_and_b32 v18, 15, v1 // (op 1127) 
v_lshl_add_u32 v19, s17, 4, v18 // (op 1127) 

v_mul_lo_u32 v18, v19, s34 // (op 1127) 

v_and_b32 v19, 63, v1 // (op 1127) 
v_lshrrev_b32 v22, 4, v19 // (op 1127) 

v_add_u32 v19, v22, v18 // (op 1127) 


s_mul_i32 s1, s34, s9 // (op 1129) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1129) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1129) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1129) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1129) most significant: sum
s_add_u32 s1, s1, s7 // (op 1129) most significant: sum


s_mov_b32 s60, s36 // (op 1129) reg expression
s_mov_b32 s61, s37 // (op 1129) reg expression
s_mov_b32 s62, s0 // (op 1129) reg expression
s_mov_b32 s63, 131072 // (op 1129) call()

v_and_b32 v18, 15, v1 // (op 1146) 
v_lshl_add_u32 v22, s17, 4, v18 // (op 1146) 

v_mul_lo_u32 v18, v22, s34 // (op 1146) 

v_and_b32 v22, 63, v1 // (op 1146) 
v_lshrrev_b32 v23, 4, v22 // (op 1146) 

v_add_u32 v22, v23, v18 // (op 1146) 


s_mul_i32 s1, s34, s9 // (op 1148) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1148) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1148) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1148) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1148) most significant: sum
s_add_u32 s1, s1, s7 // (op 1148) most significant: sum


s_mov_b32 s60, s36 // (op 1148) reg expression
s_mov_b32 s61, s37 // (op 1148) reg expression
s_mov_b32 s62, s0 // (op 1148) reg expression
s_mov_b32 s63, 131072 // (op 1148) call()




buffer_load_ubyte v18, v19, s[60:63], 0 offen offset: 0 // (op 782) Load value
v_and_b32 v23, -8, v1 // (op 1270) 
v_sub_u32 v24, v1, v23 // (op 1270) 

v_lshlrev_b32 v23, 4, v24 // (op 1270) 

v_lshrrev_b32 v24, 3, v1 // (op 1270) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 1270) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 1270) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 890) reg expression
buffer_load_dwordx4 v14, s[52:55], 0 offen offset: 0 lds // (op 890) Load value direct to lds



v_mov_b32 v23, v12 // (op 789) reg expression

buffer_load_dwordx3 v[24:26], v23, s[48:51], 0 offen // (op 789) Load value
buffer_load_dwordx3 v[28:30], v23, s[48:51], 0 offen offset:12 // (op 789) Load value
buffer_load_dwordx3 v[32:34], v23, s[48:51], 0 offen offset:24 // (op 789) Load value
buffer_load_dwordx3 v[36:38], v23, s[48:51], 0 offen offset:36 // (op 789) Load value


v_mov_b64 v[40:41], s[28:29] // (op 792) convert

v_mov_b32 v23, v20 // (op 792) reg expression
buffer_load_ubyte v42, v23, s[56:59], 0 offen offset: 0 // (op 792) Load value
v_add_u32 v23, v23, v40 // (op 792) 
buffer_load_ubyte v43, v23, s[56:59], 0 offen offset: 0 // (op 792) Load value





buffer_load_ubyte v23, v19, s[60:63], 0 offen offset:4 // (op 796) Load value

v_and_b32 v19, -8, v1 // (op 1414) 
v_sub_u32 v27, v1, v19 // (op 1414) 

v_lshlrev_b32 v19, 4, v27 // (op 1414) 

v_lshrrev_b32 v27, 3, v1 // (op 1414) 
v_lshl_add_u32 v31, v27, 7, v19 // (op 1414) 


s_nop 0
v_readfirstlane_b32 s0, v31 // (op 1414) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 2048, s0 // (op 895) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 895) reg expression
s_mov_b32 s6, 128 // (op 895) call()
buffer_load_dwordx4 v14, s[52:55], s6 offen offset: 0 lds // (op 895) Load value direct to lds






s_lshl_b64 s[0:1], s[12:13], 7 // (op 803) 
s_mul_i32 s7, s1, 6 // (op 803) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 803) least significant
s_mul_hi_u32 s10, s0, 6 // (op 803) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 803) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 803) 

v_add_u32 v19, v12, s0 // (op 803) 


buffer_load_dwordx3 v[44:46], v19, s[48:51], 0 offen // (op 803) Load value
buffer_load_dwordx3 v[48:50], v19, s[48:51], 0 offen offset:12 // (op 803) Load value
buffer_load_dwordx3 v[52:54], v19, s[48:51], 0 offen offset:24 // (op 803) Load value
buffer_load_dwordx3 v[56:58], v19, s[48:51], 0 offen offset:36 // (op 803) Load value



v_mov_b64 v[14:15], s[28:29] // (op 806) convert

s_lshl_b32 s0, s28, 2 // (op 806) 
v_add_u32 v12, v20, s0 // (op 806) 

buffer_load_ubyte v40, v12, s[56:59], 0 offen offset: 0 // (op 806) Load value
v_add_u32 v12, v12, v14 // (op 806) 
buffer_load_ubyte v41, v12, s[56:59], 0 offen offset: 0 // (op 806) Load value



v_lshlrev_b32 v12, 6, v1 // (op 1213) 
v_mul_lo_u32 v27, v12, 6 // (op 1213) 
v_lshrrev_b32 v31, 3, v27 // (op 1213) 

v_lshrrev_b32 v27, 4, v12 // (op 1213) 

v_lshlrev_b32 v12, 2, v27 // (op 1213) 

v_add_u32 v27, v31, v12 // (op 1213) 





s_waitcnt vmcnt(13)
ds_write_b96 v27, v[24:26] offset:4096 // (op 809) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v27, v[28:30] offset:4112 // (op 809) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v27, v[32:34] offset:4128 // (op 809) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v27, v[36:38] offset:4144 // (op 809) Store local data 

v_and_b32 v12, -64, v1 // (op 1323) 
v_sub_u32 v27, v1, v12 // (op 1323) 

v_lshrrev_b32 v12, 6, v1 // (op 1323) 
v_lshl_add_u32 v31, v12, 7, v27 // (op 1323) 





s_waitcnt vmcnt(9)
ds_write_b8 v31, v42 offset:12288 // (op 810) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v31, v43 offset:12352 // (op 810) Store local data 

v_lshlrev_b32 v12, 6, v1 // (op 1184) 
v_mul_lo_u32 v27, v12, 6 // (op 1184) 
v_lshrrev_b32 v31, 3, v27 // (op 1184) 

v_lshrrev_b32 v27, 4, v12 // (op 1184) 

v_lshlrev_b32 v12, 2, v27 // (op 1184) 

v_add_u32 v27, v31, v12 // (op 1184) 


v_and_b32 v12, -8, v1 // (op 1242) 
v_sub_u32 v31, v1, v12 // (op 1242) 

v_lshlrev_b32 v12, 4, v31 // (op 1242) 

v_lshrrev_b32 v31, 3, v1 // (op 1242) 
v_lshl_add_u32 v35, v31, 7, v12 // (op 1242) 


s_nop 0
v_readfirstlane_b32 s0, v35 // (op 1242) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1294) 
v_sub_u32 v31, v1, v12 // (op 1294) 

v_lshrrev_b32 v12, 6, v1 // (op 1294) 
v_lshl_add_u32 v35, v12, 7, v31 // (op 1294) 


v_lshlrev_b32 v12, 6, v1 // (op 1370) 
v_mul_lo_u32 v31, v12, 6 // (op 1370) 
v_lshrrev_b32 v39, 3, v31 // (op 1370) 

v_lshrrev_b32 v31, 4, v12 // (op 1370) 

v_lshlrev_b32 v12, 2, v31 // (op 1370) 

v_add_u32 v31, v39, v12 // (op 1370) 


v_and_b32 v12, -8, v1 // (op 1399) 
v_sub_u32 v39, v1, v12 // (op 1399) 

v_lshlrev_b32 v12, 4, v39 // (op 1399) 

v_lshrrev_b32 v39, 3, v1 // (op 1399) 
v_lshl_add_u32 v47, v39, 7, v12 // (op 1399) 


s_nop 0
v_readfirstlane_b32 s1, v47 // (op 1399) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1438) 
v_sub_u32 v39, v1, v12 // (op 1438) 

v_lshrrev_b32 v12, 6, v1 // (op 1438) 
v_lshl_add_u32 v47, v12, 7, v39 // (op 1438) 


s_mov_b32 s6, 0 // (op 315) call()
s_ashr_i64 s[10:11], s[22:23], 7 // (op 314) 
s_mov_b32 s7, s10 // (op 314) convert

s_and_b32 s10, s7, 4294967294 // (op 314) 

s_cmp_lt_u32 s6, s10 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_314 // (op 314) Condition: Top (jump to RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_314 if false)
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_314:
s_add_u32 s6, s6, 2 // (op 316) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1496) 



ds_read_b96_tr_b6 v[60:62], v0 offset:4096 // (op 67) Transpose load local data 
ds_read_b96_tr_b6 v[63:65], v0 offset:5120 // (op 67) Transpose load local data 
s_barrier  // (op 1509) 



ds_read_b128 v[66:69], v4 // (op 72) Load local data 
ds_read_b128 v[70:73], v4 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1522) 


ds_read_u8 v12, v6 offset:12288 // (op 77) Load local data 



ds_read_b96_tr_b6 v[74:76], v0 offset:4128 // (op 344) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:65], v[66:73], a[0:3], v12, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 84) 
ds_read_b96_tr_b6 v[77:79], v0 offset:5152 // (op 344) Transpose load local data 




ds_read_u8 v12, v6 offset:12320 // (op 348) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:79], v[66:73], a[4:7], v12, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 350) 





v_mov_b32 v12, v22 // (op 82) 

buffer_load_ubyte v18, v12, s[60:63], 0 offen offset:8 // (op 82) Load value
s_barrier  // (op 1512) 


v_mov_b64 v[14:15], v[16:17] // (op 873) 

s_mov_b32 m0, s0 // (op 873) reg expression
s_mov_b32 s7, 256 // (op 873) call()
buffer_load_dwordx4 v14, s[52:55], s7 offen offset: 0 lds // (op 873) Load value direct to lds



v_mov_b32 v39, v13 // (op 4) 
s_lshl_b64 s[10:11], s[12:13], 7 // (op 4) 
s_mul_i32 s15, s11, 6 // (op 4) most significant: low of high * low
s_mul_i32 s14, s10, 6 // (op 4) least significant
s_mul_hi_u32 s7, s10, 6 // (op 4) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 4) most significant: sum


s_lshr_b64 s[10:11], s[14:15], 3 // (op 4) 

s_lshl_b32 s7, s10, 1 // (op 4) 

v_add_u32 v51, v39, s7 // (op 4) 


buffer_load_dwordx3 v[24:26], v51, s[48:51], 0 offen // (op 4) Load value
buffer_load_dwordx3 v[28:30], v51, s[48:51], 0 offen offset:12 // (op 4) Load value
buffer_load_dwordx3 v[32:34], v51, s[48:51], 0 offen offset:24 // (op 4) Load value
buffer_load_dwordx3 v[36:38], v51, s[48:51], 0 offen offset:36 // (op 4) Load value


v_mov_b64 v[60:61], s[28:29] // (op 17) convert

v_mov_b32 v51, v21 // (op 17) 
s_lshl_b32 s7, s28, 3 // (op 17) 
v_add_u32 v55, v51, s7 // (op 17) 

buffer_load_ubyte v42, v55, s[56:59], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v55, v55, v60 // (op 17) 
buffer_load_ubyte v43, v55, s[56:59], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1539) 



s_waitcnt vmcnt(13)
ds_write_b96 v31, v[44:46] offset:12544 // (op 229) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v31, v[48:50] offset:12560 // (op 229) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v31, v[52:54] offset:12576 // (op 229) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v31, v[56:58] offset:12592 // (op 229) Store local data 
s_barrier  // (op 1551) 



s_waitcnt vmcnt(9)
ds_write_b8 v47, v40 offset:20736 // (op 240) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v47, v41 offset:20800 // (op 240) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1536) 



ds_read_b96_tr_b6 v[60:62], v7 offset:12544 // (op 231) Transpose load local data 
ds_read_b96_tr_b6 v[63:65], v7 offset:13568 // (op 231) Transpose load local data 
s_barrier  // (op 1542) 



ds_read_b128 v[66:69], v5 offset:2048 // (op 233) Load local data 
ds_read_b128 v[70:73], v5 offset:2112 // (op 233) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1548) 


ds_read_u8 v55, v9 offset:20736 // (op 235) Load local data 



ds_read_b96_tr_b6 v[74:76], v7 offset:12576 // (op 357) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:65], v[66:73], a[0:3], v55, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 238) 
ds_read_b96_tr_b6 v[77:79], v7 offset:13600 // (op 357) Transpose load local data 




s_barrier  // (op 1616) 
ds_read_u8 v55, v9 offset:20768 // (op 361) Load local data 
s_barrier  // (op 1620) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:79], v[66:73], a[4:7], v55, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 351) 






buffer_load_ubyte v23, v12, s[60:63], 0 offen offset:12 // (op 237) Load value

s_barrier  // (op 1545) 
v_add_u32 v22, 8, v22 // (op 1138) 


s_add_u32 s7, 2048, s1 // (op 885) 

s_mov_b32 m0, s7 // (op 885) reg expression
s_mov_b32 s10, 384 // (op 885) call()
buffer_load_dwordx4 v14, s[52:55], s10 offen offset: 0 lds // (op 885) Load value direct to lds





v_add_co_u32 v16, vcc, 256, v16 // (op 1010) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1010) most significant half
s_lshl_b64 s[10:11], s[12:13], 7 // (op 228) 
s_mul_i32 s15, s11, 6 // (op 228) most significant: low of high * low
s_mul_i32 s14, s10, 6 // (op 228) least significant
s_mul_hi_u32 s7, s10, 6 // (op 228) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 228) most significant: sum


s_lshr_b64 s[10:11], s[14:15], 3 // (op 228) 

s_mul_i32 s7, 3, s10 // (op 228) 

v_add_u32 v12, v39, s7 // (op 228) 


buffer_load_dwordx3 v[44:46], v12, s[48:51], 0 offen // (op 228) Load value
buffer_load_dwordx3 v[48:50], v12, s[48:51], 0 offen offset:12 // (op 228) Load value
buffer_load_dwordx3 v[52:54], v12, s[48:51], 0 offen offset:24 // (op 228) Load value
buffer_load_dwordx3 v[56:58], v12, s[48:51], 0 offen offset:36 // (op 228) Load value



v_mov_b64 v[14:15], s[28:29] // (op 242) convert

s_lshl_b32 s7, s28, 2 // (op 242) 
s_mul_i32 s10, 3, s7 // (op 242) 

v_add_u32 v12, v51, s10 // (op 242) 

s_lshl_b64 s[10:11], s[12:13], 7 // (op 946) 
s_mul_i32 s15, s11, 6 // (op 946) most significant: low of high * low
s_mul_i32 s14, s10, 6 // (op 946) least significant
s_mul_hi_u32 s7, s10, 6 // (op 946) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 946) most significant: sum


s_lshr_b64 s[10:11], s[14:15], 3 // (op 946) 

s_lshl_b32 s7, s10, 1 // (op 946) 

v_add_u32 v13, v13, s7 // (op 946) 

buffer_load_ubyte v40, v12, s[56:59], 0 offen offset: 0 // (op 242) Load value
v_add_u32 v12, v12, v14 // (op 242) 
buffer_load_ubyte v41, v12, s[56:59], 0 offen offset: 0 // (op 242) Load value



s_barrier  // (op 1499) 
s_lshl_b32 s7, s28, 3 // (op 1074) 



v_add_u32 v21, v21, s7 // (op 1074) 

s_waitcnt vmcnt(13)
ds_write_b96 v27, v[24:26] offset:4096 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v27, v[28:30] offset:4112 // (op 6) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v27, v[32:34] offset:4128 // (op 6) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v27, v[36:38] offset:4144 // (op 6) Store local data 
s_barrier  // (op 1525) 



s_waitcnt vmcnt(9)
ds_write_b8 v35, v42 offset:12288 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v35, v43 offset:12352 // (op 21) Store local data 
s_ashr_i64 s[10:11], s[22:23], 7 // (op 314) 
s_mov_b32 s7, s10 // (op 314) convert

s_and_b32 s10, s7, 4294967294 // (op 314) 

s_cmp_lt_u32 s6, s10 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_314 // (op 314) Condition: Bottom (jump to RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_314 if true)
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_314:














s_barrier  // (op 1600) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 908) 
s_lshr_b32 s6, s0, 1 // (op 908) 

v_lshl_add_u32 v47, s6, 8, v1 // (op 908) 

s_lshl_b32 s0, s16, 6 // (op 908) 
v_mul_lo_u32 v43, v47, s13 // (op 908) most significant: low of low * high
v_mul_hi_u32 v51, v47, s12 // (op 908) most significant: high of low * low
v_mul_lo_u32 v42, v47, s12 // (op 908) least significant: low of low * low
v_add_u32 v43, v43, v51 // (op 908) most significant: sum


s_mov_b32 s7, 0 // (op 908) convert
s_mov_b32 s6, s0 // (op 908) convert

v_mov_b32 v47, s6 // (op 908) 
v_mov_b32 v51, s7 // (op 908) 
v_add_co_u32 v12, s[0:1], v47, v42 // (op 908) least significant half
v_addc_co_u32 v13, s[0:1], v51, v43, s[0:1] // (op 908) most significant half





v_mul_lo_u32 v43, v13, 6 // (op 908) most significant: low of high * low
v_mul_hi_u32 v47, v12, 6 // (op 908) most significant: high of low * low
v_mul_lo_u32 v42, v12, 6 // (op 908) least significant: low of low * low
v_add_u32 v43, v43, v47 // (op 908) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[42:43] // (op 908) 

v_mov_b32 v47, v12 // (op 908) convert

s_mul_i32 s1, s23, 6 // (op 910) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 910) least significant
s_mul_hi_u32 s6, s22, 6 // (op 910) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 910) most significant: sum

s_mul_i32 s7, s12, s1 // (op 910) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 910) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 910) least significant
s_mul_hi_u32 s11, s12, s0 // (op 910) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 910) most significant: sum
s_add_u32 s7, s7, s11 // (op 910) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 910) 

s_mov_b32 s48, s20 // (op 910) reg expression
s_mov_b32 s49, s21 // (op 910) reg expression
s_mov_b32 s50, s0 // (op 910) reg expression
s_mov_b32 s51, 131072 // (op 910) call()

v_and_b32 v51, -8, v1 // (op 976) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 976) 
v_sub_u32 v55, v1, v51 // (op 976) 

s_lshr_b32 s6, s0, 1 // (op 976) 

v_lshlrev_b32 v51, 4, v55 // (op 976) 

v_lshl_add_u32 v55, s6, 8, v51 // (op 976) 


v_mov_b32 v43, 0 // (op 976) convert
v_mov_b32 v42, v55 // (op 976) convert

v_lshrrev_b32 v51, 3, v1 // (op 976) 
v_lshl_add_u32 v55, s17, 4, v51 // (op 976) 

v_mul_lo_u32 v13, v55, s25 // (op 976) most significant: low of low * high
v_mul_hi_u32 v51, v55, s24 // (op 976) most significant: high of low * low
v_mul_lo_u32 v12, v55, s24 // (op 976) least significant: low of low * low
v_add_u32 v13, v13, v51 // (op 976) most significant: sum


v_add_co_u32 v14, s[0:1], v42, v12 // (op 976) least significant half
v_addc_co_u32 v15, s[0:1], v43, v13, s[0:1] // (op 976) most significant half



s_mul_i32 s1, s24, s9 // (op 978) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 978) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 978) least significant
s_mul_hi_u32 s7, s24, s8 // (op 978) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 978) most significant: sum
s_add_u32 s1, s1, s7 // (op 978) most significant: sum


s_mov_b32 s52, s26 // (op 978) reg expression
s_mov_b32 s53, s27 // (op 978) reg expression
s_mov_b32 s54, s0 // (op 978) reg expression
s_mov_b32 s55, 131072 // (op 978) call()

v_and_b32 v51, -64, v1 // (op 1040) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1040) 
v_sub_u32 v55, v1, v51 // (op 1040) 

s_lshr_b32 s6, s0, 1 // (op 1040) 

v_lshl_add_u32 v51, s16, 6, v55 // (op 1040) 

v_lshrrev_b32 v55, 6, v1 // (op 1040) 
v_lshlrev_b32 v12, 1, v55 // (op 1040) 

v_lshl_add_u32 v13, s6, 3, v12 // (op 1040) 


v_mul_lo_u32 v12, v13, s28 // (op 1040) 

v_add_u32 v13, v51, v12 // (op 1040) 


s_mul_i32 s1, s28, s33 // (op 1042) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1042) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1042) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1042) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1042) most significant: sum
s_add_u32 s1, s1, s7 // (op 1042) most significant: sum


s_mov_b32 s56, s30 // (op 1042) reg expression
s_mov_b32 s57, s31 // (op 1042) reg expression
s_mov_b32 s58, s0 // (op 1042) reg expression
s_mov_b32 s59, 131072 // (op 1042) call()

v_and_b32 v12, 63, v1 // (op 1104) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1104) 
v_lshrrev_b32 v51, 4, v12 // (op 1104) 

s_lshr_b32 s6, s0, 1 // (op 1104) 

v_and_b32 v12, 15, v1 // (op 1104) 
v_lshl_add_u32 v55, s6, 3, v51 // (op 1104) 


v_lshl_add_u32 v51, s17, 4, v12 // (op 1104) 

v_mul_lo_u32 v12, v51, s34 // (op 1104) 

v_add_u32 v51, v55, v12 // (op 1104) 


s_mul_i32 s1, s34, s9 // (op 1106) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1106) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1106) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1106) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1106) most significant: sum
s_add_u32 s1, s1, s7 // (op 1106) most significant: sum


s_mov_b32 s60, s36 // (op 1106) reg expression
s_mov_b32 s61, s37 // (op 1106) reg expression
s_mov_b32 s62, s0 // (op 1106) reg expression
s_mov_b32 s63, 131072 // (op 1106) call()

v_lshlrev_b32 v12, 6, v1 // (op 1197) 
v_mul_lo_u32 v55, v12, 6 // (op 1197) 
v_lshrrev_b32 v16, 3, v55 // (op 1197) 

v_lshrrev_b32 v55, 4, v12 // (op 1197) 

v_lshlrev_b32 v12, 2, v55 // (op 1197) 

v_add_u32 v55, v16, v12 // (op 1197) 


v_and_b32 v12, -8, v1 // (op 1255) 
v_sub_u32 v16, v1, v12 // (op 1255) 

v_lshlrev_b32 v12, 4, v16 // (op 1255) 

v_lshrrev_b32 v16, 3, v1 // (op 1255) 
v_lshl_add_u32 v17, v16, 7, v12 // (op 1255) 


s_nop 0
v_readfirstlane_b32 s0, v17 // (op 1255) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1307) 
v_sub_u32 v16, v1, v12 // (op 1307) 

v_lshrrev_b32 v12, 6, v1 // (op 1307) 
v_lshl_add_u32 v17, v12, 7, v16 // (op 1307) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 320) 
s_and_b32 s1, s6, 4294967294 // (op 320) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 319) 
s_mov_b32 s10, s6 // (op 319) convert

s_cmp_lt_u32 s1, s10 // (op 319) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_319 // (op 319) Condition: Top (jump to RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_319 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_319:


s_add_u32 s1, s1, 1 // (op 321) 
v_mov_b32 v12, v47 // (op 170) 
v_mov_b32 v16, v12 // (op 170) reg expression

buffer_load_dwordx3 v[18:20], v16, s[48:51], 0 offen // (op 170) Load value
buffer_load_dwordx3 v[22:24], v16, s[48:51], 0 offen offset:12 // (op 170) Load value
buffer_load_dwordx3 v[26:28], v16, s[48:51], 0 offen offset:24 // (op 170) Load value
buffer_load_dwordx3 v[30:32], v16, s[48:51], 0 offen offset:36 // (op 170) Load value


s_lshl_b64 s[6:7], s[12:13], 7 // (op 898) 
s_mul_i32 s11, s7, 6 // (op 898) most significant: low of high * low
s_mul_i32 s10, s6, 6 // (op 898) least significant
s_mul_hi_u32 s14, s6, 6 // (op 898) most significant: high of low * low
s_add_u32 s11, s11, s14 // (op 898) most significant: sum


s_lshr_b64 s[6:7], s[10:11], 3 // (op 898) 

v_add_u32 v47, v47, s6 // (op 898) 

s_barrier  // (op 1502) 
s_barrier  // (op 1515) 





v_mov_b64 v[48:49], v[14:15] // (op 878) 

s_mov_b32 m0, s0 // (op 878) reg expression
buffer_load_dwordx4 v48, s[52:55], 0 offen offset: 0 lds // (op 878) Load value direct to lds

v_add_co_u32 v14, vcc, 128, v14 // (op 968) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 968) most significant half
s_waitcnt vmcnt(4)
ds_write_b96 v55, v[18:20] offset:4096 // (op 171) Store local data 

v_mov_b64 v[48:49], s[28:29] // (op 184) convert

v_mov_b32 v12, v13 // (op 184) 
v_mov_b32 v16, v12 // (op 184) reg expression
buffer_load_ubyte v52, v16, s[56:59], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v16, v16, v48 // (op 184) 
s_waitcnt vmcnt(4)
ds_write_b96 v55, v[22:24] offset:4112 // (op 171) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v55, v[26:28] offset:4128 // (op 171) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v55, v[30:32] offset:4144 // (op 171) Store local data 
buffer_load_ubyte v53, v16, s[56:59], 0 offen offset: 0 // (op 184) Load value




s_lshl_b32 s6, s28, 2 // (op 1032) 
v_add_u32 v13, v13, s6 // (op 1032) 

s_barrier  // (op 1528) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1505) 






ds_read_b96_tr_b6 v[18:20], v0 offset:4096 // (op 173) Transpose load local data 
ds_read_b96_tr_b6 v[21:23], v0 offset:5120 // (op 173) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v17, v52 offset:12288 // (op 182) Store local data 
s_barrier  // (op 1518) 
s_waitcnt vmcnt(0)
ds_write_b8 v17, v53 offset:12352 // (op 182) Store local data 




ds_read_b128 v[24:27], v4 // (op 175) Load local data 
ds_read_b128 v[28:31], v4 offset:64 // (op 175) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1532) 


ds_read_u8 v12, v6 offset:12288 // (op 177) Load local data 


v_mov_b32 v16, v51 // (op 179) 

buffer_load_ubyte v32, v16, s[60:63], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v51, 4, v51 // (op 1096) 



ds_read_b96_tr_b6 v[34:36], v0 offset:4128 // (op 325) Transpose load local data 
ds_read_b96_tr_b6 v[37:39], v0 offset:5152 // (op 325) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:23], v[24:31], a[0:3], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 180) 




ds_read_u8 v12, v6 offset:12320 // (op 329) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:39], v[24:31], a[4:7], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 331) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 319) 
s_mov_b32 s10, s6 // (op 319) convert

s_cmp_lt_u32 s1, s10 // (op 319) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_319 // (op 319) Condition: Bottom (jump to RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_319 if true)
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_319:






















s_barrier  // (op 1604) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125
v_accvgpr_read v4, a0 // (op 46) 
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v12, v11, v4 // (op 46) 
v_mul_f32 v13, v11, v5 // (op 46) 
v_mul_f32 v14, v11, v6 // (op 46) 
v_mul_f32 v15, v11, v7 // (op 46) 

v_cvt_f16_f32 v4, v12 // (op 52) 
v_cvt_f16_f32 v5, v13 // (op 52) 
v_cvt_f16_f32 v6, v14 // (op 52) 
v_cvt_f16_f32 v7, v15 // (op 52) 


v_pack_b32_f16 v12, v4, v5 // (op 54) 
v_pack_b32_f16 v13, v6, v7 // (op 54) 


buffer_store_dwordx2 v[12:13], v8, s[44:47], 0 offen // (op 54) Store value


v_accvgpr_read v4, a4 // (op 517) 
v_accvgpr_read v5, a5 // (op 517) 
v_accvgpr_read v6, a6 // (op 517) 
v_accvgpr_read v7, a7 // (op 517) 
v_mul_f32 v12, v11, v4 // (op 517) 
v_mul_f32 v13, v11, v5 // (op 517) 
v_mul_f32 v14, v11, v6 // (op 517) 
v_mul_f32 v15, v11, v7 // (op 517) 

v_cvt_f16_f32 v4, v12 // (op 519) 
v_cvt_f16_f32 v5, v13 // (op 519) 
v_cvt_f16_f32 v6, v14 // (op 519) 
v_cvt_f16_f32 v7, v15 // (op 519) 


v_pack_b32_f16 v12, v4, v5 // (op 520) 
v_pack_b32_f16 v13, v6, v7 // (op 520) 


buffer_store_dwordx2 v[12:13], v8, s[44:47], 0 offen offset:64 // (op 520) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen offset:64 // (op 550) Load value
v_accvgpr_read v12, a4 // (op 546) 
v_accvgpr_read v13, a5 // (op 546) 
v_accvgpr_read v14, a6 // (op 546) 
v_accvgpr_read v15, a7 // (op 546) 
v_mul_f32 v16, v11, v12 // (op 546) 
v_mul_f32 v17, v11, v13 // (op 546) 
v_mul_f32 v18, v11, v14 // (op 546) 
v_mul_f32 v19, v11, v15 // (op 546) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 551) 
v_lshrrev_b32 v7, 16, v4 // (op 551) 
v_cvt_f32_f16 v7, v7 // (op 551) 
v_mul_f32 v12, v10, v6 // (op 551) 
v_mul_f32 v13, v10, v7 // (op 551) 
v_cvt_f32_f16 v6, v5 // (op 551) 
v_lshrrev_b32 v7, 16, v5 // (op 551) 
v_cvt_f32_f16 v7, v7 // (op 551) 
v_mul_f32 v14, v10, v6 // (op 551) 
v_mul_f32 v15, v10, v7 // (op 551) 


v_add_f32 v4, v12, v16 // (op 547) 
v_add_f32 v5, v13, v17 // (op 547) 
v_add_f32 v6, v14, v18 // (op 547) 
v_add_f32 v7, v15, v19 // (op 547) 


v_cvt_f16_f32 v12, v4 // (op 548) 
v_cvt_f16_f32 v13, v5 // (op 548) 
v_cvt_f16_f32 v14, v6 // (op 548) 
v_cvt_f16_f32 v15, v7 // (op 548) 


v_pack_b32_f16 v4, v12, v13 // (op 549) 
v_pack_b32_f16 v5, v14, v15 // (op 549) 


buffer_store_dwordx2 v[4:5], v8, s[44:47], 0 offen offset:64 // (op 549) Store value





buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 138) Load value
v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v11, v12 // (op 134) 
v_mul_f32 v17, v11, v13 // (op 134) 
v_mul_f32 v18, v11, v14 // (op 134) 
v_mul_f32 v19, v11, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 139) 
v_lshrrev_b32 v7, 16, v4 // (op 139) 
v_cvt_f32_f16 v7, v7 // (op 139) 
v_mul_f32 v12, v10, v6 // (op 139) 
v_mul_f32 v13, v10, v7 // (op 139) 
v_cvt_f32_f16 v6, v5 // (op 139) 
v_lshrrev_b32 v7, 16, v5 // (op 139) 
v_cvt_f32_f16 v7, v7 // (op 139) 
v_mul_f32 v14, v10, v6 // (op 139) 
v_mul_f32 v15, v10, v7 // (op 139) 


v_add_f32 v4, v12, v16 // (op 135) 
v_add_f32 v5, v13, v17 // (op 135) 
v_add_f32 v6, v14, v18 // (op 135) 
v_add_f32 v7, v15, v19 // (op 135) 


v_cvt_f16_f32 v12, v4 // (op 136) 
v_cvt_f16_f32 v13, v5 // (op 136) 
v_cvt_f16_f32 v14, v6 // (op 136) 
v_cvt_f16_f32 v15, v7 // (op 136) 


v_pack_b32_f16 v4, v12, v13 // (op 137) 
v_pack_b32_f16 v5, v14, v15 // (op 137) 


buffer_store_dwordx2 v[4:5], v8, s[44:47], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.LRR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_end:
.size RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2, .LRR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_end-RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
  .amdhsa_next_free_vgpr 88
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 20992 // lds bytes
  .amdhsa_accum_offset 80
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: B8
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
  - .name: RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
    .symbol: RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 80
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

