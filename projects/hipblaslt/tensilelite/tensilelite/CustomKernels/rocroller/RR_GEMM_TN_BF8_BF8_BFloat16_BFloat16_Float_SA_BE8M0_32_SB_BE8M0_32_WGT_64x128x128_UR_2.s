

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2,@function
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 4596) 
v_and_b32 v3, 15, v1 // (op 4596) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 4596) 


v_and_b32 v0, 48, v1 // (op 4596) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 4596) 


v_lshrrev_b32 v0, 6, v1 // (op 4653) 
v_and_b32 v4, 1, v0 // (op 4653) 

v_and_b32 v0, 15, v1 // (op 4653) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4653) 


v_and_b32 v0, 48, v1 // (op 4653) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 4653) 


v_lshrrev_b32 v0, 7, v1 // (op 4711) 
v_and_b32 v5, 15, v1 // (op 4711) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 4711) 


v_and_b32 v0, 63, v1 // (op 4711) 
v_lshrrev_b32 v5, 4, v0 // (op 4711) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 4711) 


v_lshrrev_b32 v5, 6, v1 // (op 4769) 
v_and_b32 v6, 1, v5 // (op 4769) 

v_and_b32 v5, 15, v1 // (op 4769) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 4769) 


v_and_b32 v5, 63, v1 // (op 4769) 
v_lshrrev_b32 v6, 4, v5 // (op 4769) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 4769) 


v_lshrrev_b32 v6, 7, v1 // (op 4836) 
v_and_b32 v7, 15, v1 // (op 4836) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 4836) 


v_and_b32 v6, 48, v1 // (op 4836) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 4836) 


v_lshrrev_b32 v6, 6, v1 // (op 4880) 
v_and_b32 v8, 1, v6 // (op 4880) 

v_and_b32 v6, 15, v1 // (op 4880) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 4880) 


v_and_b32 v6, 48, v1 // (op 4880) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 4880) 


v_lshrrev_b32 v6, 7, v1 // (op 4909) 
v_and_b32 v9, 15, v1 // (op 4909) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 4909) 


v_and_b32 v6, 63, v1 // (op 4909) 
v_lshrrev_b32 v9, 4, v6 // (op 4909) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 4909) 


v_lshrrev_b32 v9, 6, v1 // (op 4938) 
v_and_b32 v10, 1, v9 // (op 4938) 

v_and_b32 v9, 15, v1 // (op 4938) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 4938) 


v_and_b32 v9, 63, v1 // (op 4938) 
v_lshrrev_b32 v10, 4, v9 // (op 4938) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 4938) 


v_lshrrev_b32 v10, 7, v1 // (op 4535) 
v_and_b32 v11, 63, v1 // (op 4535) 
v_lshrrev_b32 v12, 4, v11 // (op 4535) 
v_lshlrev_b32 v13, 2, v12 // (op 4535) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 4535) 


v_lshl_add_u32 v10, s16, 6, v12 // (op 4535) 

v_lshrrev_b32 v12, 2, v11 // (op 4535) 

v_and_b32 v11, 3, v12 // (op 4535) 

v_and_b32 v12, 3, v1 // (op 4535) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 4535) 


v_lshrrev_b32 v11, 6, v1 // (op 4535) 
v_and_b32 v12, 1, v11 // (op 4535) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 4535) 


v_lshl_add_u32 v12, s17, 7, v11 // (op 4535) 

v_mul_lo_u32 v11, v12, s2 // (op 4535) 

v_add_lshl_u32 v12, v10, v11, 1 // (op 4535) 


v_lshrrev_b32 v10, 7, v1 // (op 4790) 
v_and_b32 v11, 63, v1 // (op 4790) 
v_lshrrev_b32 v13, 4, v11 // (op 4790) 
s_mul_i32 s1, s2, s9 // (op 4537) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 4537) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 4537) least significant
s_mul_hi_u32 s19, s2, s8 // (op 4537) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4537) most significant: sum
s_add_u32 s1, s1, s19 // (op 4537) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 4537) 

v_lshlrev_b32 v14, 2, v13 // (op 4790) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 4790) 


v_lshl_add_u32 v10, s16, 6, v13 // (op 4790) 

v_lshrrev_b32 v13, 2, v11 // (op 4790) 

v_and_b32 v11, 3, v13 // (op 4790) 

v_and_b32 v13, 3, v1 // (op 4790) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 4790) 


v_lshrrev_b32 v11, 6, v1 // (op 4790) 
v_and_b32 v13, 1, v11 // (op 4790) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 4790) 


v_lshl_add_u32 v13, s17, 7, v11 // (op 4790) 

v_mul_lo_u32 v11, v13, s4 // (op 4790) 

v_add_lshl_u32 v13, v10, v11, 1 // (op 4790) 


s_mov_b32 s40, s6 // (op 4537) reg expression
s_mov_b32 s41, s7 // (op 4537) reg expression
s_mov_b32 s42, s18 // (op 4537) reg expression
s_mov_b32 s43, 131072 // (op 4537) call()

s_mul_i32 s1, s4, s9 // (op 4792) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 4792) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 4792) least significant
s_mul_hi_u32 s7, s4, s8 // (op 4792) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4792) most significant: sum
s_add_u32 s1, s1, s7 // (op 4792) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 4792) 

s_mov_b32 s44, s10 // (op 4792) reg expression
s_mov_b32 s45, s11 // (op 4792) reg expression
s_mov_b32 s46, s6 // (op 4792) reg expression
s_mov_b32 s47, 131072 // (op 4792) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 340) call()
v_accvgpr_write a5, 0.00000 // (op 340) call()
v_accvgpr_write a6, 0.00000 // (op 340) call()
v_accvgpr_write a7, 0.00000 // (op 340) call()
v_accvgpr_write a8, 0.00000 // (op 516) call()
v_accvgpr_write a9, 0.00000 // (op 516) call()
v_accvgpr_write a10, 0.00000 // (op 516) call()
v_accvgpr_write a11, 0.00000 // (op 516) call()
v_accvgpr_write a12, 0.00000 // (op 692) call()
v_accvgpr_write a13, 0.00000 // (op 692) call()
v_accvgpr_write a14, 0.00000 // (op 692) call()
v_accvgpr_write a15, 0.00000 // (op 692) call()
v_accvgpr_write a16, 0.00000 // (op 956) call()
v_accvgpr_write a17, 0.00000 // (op 956) call()
v_accvgpr_write a18, 0.00000 // (op 956) call()
v_accvgpr_write a19, 0.00000 // (op 956) call()
v_accvgpr_write a20, 0.00000 // (op 1265) call()
v_accvgpr_write a21, 0.00000 // (op 1265) call()
v_accvgpr_write a22, 0.00000 // (op 1265) call()
v_accvgpr_write a23, 0.00000 // (op 1265) call()
v_accvgpr_write a24, 0.00000 // (op 1266) call()
v_accvgpr_write a25, 0.00000 // (op 1266) call()
v_accvgpr_write a26, 0.00000 // (op 1266) call()
v_accvgpr_write a27, 0.00000 // (op 1266) call()
v_accvgpr_write a28, 0.00000 // (op 1267) call()
v_accvgpr_write a29, 0.00000 // (op 1267) call()
v_accvgpr_write a30, 0.00000 // (op 1267) call()
v_accvgpr_write a31, 0.00000 // (op 1267) call()
v_lshrrev_b32 v14, 3, v1 // (op 4297) 
v_lshl_add_u32 v15, s16, 6, v14 // (op 4297) 

v_and_b32 v14, -8, v1 // (op 4297) 
v_sub_u32 v16, v1, v14 // (op 4297) 

v_lshlrev_b32 v14, 4, v16 // (op 4297) 

v_mov_b32 v17, 0 // (op 4297) convert
v_mov_b32 v16, v14 // (op 4297) convert

v_mul_lo_u32 v19, v15, s13 // (op 4297) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 4297) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 4297) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 4297) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 4297) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 4297) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 4299) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 4299) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 4299) least significant
s_mul_hi_u32 s7, s12, s22 // (op 4299) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4299) most significant: sum
s_add_u32 s1, s1, s7 // (op 4299) most significant: sum


s_mov_b32 s48, s20 // (op 4299) reg expression
s_mov_b32 s49, s21 // (op 4299) reg expression
s_mov_b32 s50, s0 // (op 4299) reg expression
s_mov_b32 s51, 131072 // (op 4299) call()

v_lshrrev_b32 v14, 3, v1 // (op 4316) 
v_lshl_add_u32 v15, s16, 6, v14 // (op 4316) 

v_and_b32 v14, -8, v1 // (op 4316) 
v_sub_u32 v16, v1, v14 // (op 4316) 

v_lshlrev_b32 v14, 4, v16 // (op 4316) 

v_mov_b32 v17, 0 // (op 4316) convert
v_mov_b32 v16, v14 // (op 4316) convert

v_mul_lo_u32 v21, v15, s13 // (op 4316) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 4316) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 4316) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 4316) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 4316) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 4316) most significant half



s_mul_i32 s1, s12, s23 // (op 4318) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 4318) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 4318) least significant
s_mul_hi_u32 s7, s12, s22 // (op 4318) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4318) most significant: sum
s_add_u32 s1, s1, s7 // (op 4318) most significant: sum


s_mov_b32 s48, s20 // (op 4318) reg expression
s_mov_b32 s49, s21 // (op 4318) reg expression
s_mov_b32 s50, s0 // (op 4318) reg expression
s_mov_b32 s51, 131072 // (op 4318) call()

v_and_b32 v14, -8, v1 // (op 4361) 
v_sub_u32 v15, v1, v14 // (op 4361) 

v_lshlrev_b32 v14, 4, v15 // (op 4361) 

v_mov_b32 v17, 0 // (op 4361) convert
v_mov_b32 v16, v14 // (op 4361) convert

v_lshrrev_b32 v14, 3, v1 // (op 4361) 
v_lshl_add_u32 v15, s17, 7, v14 // (op 4361) 

v_mul_lo_u32 v23, v15, s25 // (op 4361) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 4361) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 4361) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 4361) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v22 // (op 4361) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 4361) most significant half



s_mul_i32 s1, s24, s9 // (op 4363) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 4363) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 4363) least significant
s_mul_hi_u32 s7, s24, s8 // (op 4363) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4363) most significant: sum
s_add_u32 s1, s1, s7 // (op 4363) most significant: sum


s_mov_b32 s52, s26 // (op 4363) reg expression
s_mov_b32 s53, s27 // (op 4363) reg expression
s_mov_b32 s54, s0 // (op 4363) reg expression
s_mov_b32 s55, 131072 // (op 4363) call()

v_and_b32 v16, -8, v1 // (op 4380) 
v_sub_u32 v17, v1, v16 // (op 4380) 

v_lshlrev_b32 v16, 4, v17 // (op 4380) 

v_mov_b32 v23, 0 // (op 4380) convert
v_mov_b32 v22, v16 // (op 4380) convert

v_lshrrev_b32 v16, 3, v1 // (op 4380) 
v_lshl_add_u32 v17, s17, 7, v16 // (op 4380) 

v_mul_lo_u32 v25, v17, s25 // (op 4380) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 4380) most significant: high of low * low
v_mul_lo_u32 v24, v17, s24 // (op 4380) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 4380) most significant: sum


v_add_co_u32 v16, s[0:1], v22, v24 // (op 4380) least significant half
v_addc_co_u32 v17, s[0:1], v23, v25, s[0:1] // (op 4380) most significant half



s_mul_i32 s1, s24, s9 // (op 4382) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 4382) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 4382) least significant
s_mul_hi_u32 s7, s24, s8 // (op 4382) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4382) most significant: sum
s_add_u32 s1, s1, s7 // (op 4382) most significant: sum


s_mov_b32 s52, s26 // (op 4382) reg expression
s_mov_b32 s53, s27 // (op 4382) reg expression
s_mov_b32 s54, s0 // (op 4382) reg expression
s_mov_b32 s55, 131072 // (op 4382) call()

v_lshrrev_b32 v22, 2, v1 // (op 4425) 
v_lshl_add_u32 v23, s16, 6, v22 // (op 4425) 

v_and_b32 v22, -4, v1 // (op 4425) 
v_sub_u32 v24, v1, v22 // (op 4425) 

v_mul_lo_u32 v22, v23, s28 // (op 4425) 
v_add_u32 v22, v22, v24 // (op 4425) 


s_mul_i32 s1, s28, s23 // (op 4427) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4427) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4427) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4427) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4427) most significant: sum
s_add_u32 s1, s1, s7 // (op 4427) most significant: sum


s_mov_b32 s56, s30 // (op 4427) reg expression
s_mov_b32 s57, s31 // (op 4427) reg expression
s_mov_b32 s58, s0 // (op 4427) reg expression
s_mov_b32 s59, 131072 // (op 4427) call()

v_lshrrev_b32 v23, 2, v1 // (op 4444) 
v_lshl_add_u32 v24, s16, 6, v23 // (op 4444) 

v_and_b32 v23, -4, v1 // (op 4444) 
v_sub_u32 v25, v1, v23 // (op 4444) 

v_mul_lo_u32 v23, v24, s28 // (op 4444) 
v_add_u32 v23, v23, v25 // (op 4444) 


s_mul_i32 s1, s28, s23 // (op 4446) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4446) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4446) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4446) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4446) most significant: sum
s_add_u32 s1, s1, s7 // (op 4446) most significant: sum


s_mov_b32 s56, s30 // (op 4446) reg expression
s_mov_b32 s57, s31 // (op 4446) reg expression
s_mov_b32 s58, s0 // (op 4446) reg expression
s_mov_b32 s59, 131072 // (op 4446) call()

v_and_b32 v24, -4, v1 // (op 4489) 
v_sub_u32 v25, v1, v24 // (op 4489) 

v_lshrrev_b32 v24, 2, v1 // (op 4489) 
v_lshlrev_b32 v26, 1, v24 // (op 4489) 

v_lshl_add_u32 v24, s17, 7, v26 // (op 4489) 

v_mul_lo_u32 v26, v24, s32 // (op 4489) 

v_add_u32 v24, v25, v26 // (op 4489) 


s_mul_i32 s1, s32, s9 // (op 4491) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4491) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4491) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4491) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4491) most significant: sum
s_add_u32 s1, s1, s7 // (op 4491) most significant: sum


s_mov_b32 s60, s34 // (op 4491) reg expression
s_mov_b32 s61, s35 // (op 4491) reg expression
s_mov_b32 s62, s0 // (op 4491) reg expression
s_mov_b32 s63, 131072 // (op 4491) call()

v_and_b32 v25, -4, v1 // (op 4508) 
v_sub_u32 v26, v1, v25 // (op 4508) 

v_lshrrev_b32 v25, 2, v1 // (op 4508) 
v_lshlrev_b32 v27, 1, v25 // (op 4508) 

v_lshl_add_u32 v25, s17, 7, v27 // (op 4508) 

v_mul_lo_u32 v27, v25, s32 // (op 4508) 

v_add_u32 v25, v26, v27 // (op 4508) 


s_mul_i32 s1, s32, s9 // (op 4510) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4510) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4510) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4510) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4510) most significant: sum
s_add_u32 s1, s1, s7 // (op 4510) most significant: sum


s_mov_b32 s60, s34 // (op 4510) reg expression
s_mov_b32 s61, s35 // (op 4510) reg expression
s_mov_b32 s62, s0 // (op 4510) reg expression
s_mov_b32 s63, 131072 // (op 4510) call()

v_and_b32 v26, -8, v1 // (op 4636) 
v_sub_u32 v27, v1, v26 // (op 4636) 

v_lshlrev_b32 v26, 4, v27 // (op 4636) 

v_lshrrev_b32 v27, 3, v1 // (op 4636) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 4636) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 4636) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 4239) 
v_mov_b64 v[26:27], s[6:7] // (op 4239) convert


v_mov_b64 v[28:29], v[14:15] // (op 4239) 
s_mov_b32 m0, s0 // (op 4239) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 4239) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 4239) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 4239) most significant half

s_add_u32 m0, m0, 4096 // (op 4239) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 4239) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 4239) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 4239) most significant half

s_add_u32 m0, m0, 4096 // (op 4239) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 4239) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 4239) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 4239) most significant half

s_add_u32 m0, m0, 4096 // (op 4239) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 4239) Load value direct to lds




v_mov_b64 v[26:27], s[32:33] // (op 4021) convert

v_mov_b32 v28, v24 // (op 4021) 
buffer_load_ubyte v30, v28, s[60:63], 0 offen offset: 0 // (op 4021) Load value
v_add_u32 v28, v28, v26 // (op 4021) 
buffer_load_ubyte v31, v28, s[60:63], 0 offen offset: 0 // (op 4021) Load value


v_and_b32 v26, -8, v1 // (op 4579) 
v_sub_u32 v27, v1, v26 // (op 4579) 

v_lshlrev_b32 v26, 4, v27 // (op 4579) 

v_lshrrev_b32 v27, 3, v1 // (op 4579) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 4579) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 4579) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 4244) 
v_mov_b64 v[26:27], s[6:7] // (op 4244) convert


v_mov_b64 v[28:29], v[18:19] // (op 4244) 
s_add_u32 s1, 16384, s0 // (op 4244) 
s_mov_b32 m0, s1 // (op 4244) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 4244) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 4244) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 4244) most significant half

s_add_u32 m0, m0, 4096 // (op 4244) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 4244) Load value direct to lds







buffer_load_ubyte v26, v22, s[56:59], 0 offen offset: 0 // (op 4027) Load value
v_and_b32 v27, -8, v1 // (op 4863) 
v_sub_u32 v28, v1, v27 // (op 4863) 

v_lshlrev_b32 v27, 4, v28 // (op 4863) 

v_lshrrev_b32 v28, 3, v1 // (op 4863) 
v_lshl_add_u32 v29, v28, 7, v27 // (op 4863) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 4863) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 4249) 
v_mov_b64 v[28:29], s[6:7] // (op 4249) convert


v_mov_b64 v[32:33], v[14:15] // (op 4249) 
s_add_u32 s1, 24576, s0 // (op 4249) 
s_mov_b32 m0, s1 // (op 4249) reg expression
s_mov_b32 s6, 128 // (op 4249) call()
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 4249) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v28 // (op 4249) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 4249) most significant half

s_add_u32 m0, m0, 4096 // (op 4249) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 4249) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v28 // (op 4249) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 4249) most significant half

s_add_u32 m0, m0, 4096 // (op 4249) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 4249) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v28 // (op 4249) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 4249) most significant half

s_add_u32 m0, m0, 4096 // (op 4249) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 4249) Load value direct to lds







v_mov_b64 v[14:15], s[32:33] // (op 4033) convert

v_mov_b32 v27, v24 // (op 4033) 
buffer_load_ubyte v28, v27, s[60:63], 0 offen offset:4 // (op 4033) Load value
v_add_u32 v27, v27, v14 // (op 4033) 
buffer_load_ubyte v29, v27, s[60:63], 0 offen offset:4 // (op 4033) Load value



v_and_b32 v24, -8, v1 // (op 4819) 
v_sub_u32 v27, v1, v24 // (op 4819) 

v_lshlrev_b32 v24, 4, v27 // (op 4819) 

v_lshrrev_b32 v27, 3, v1 // (op 4819) 
v_lshl_add_u32 v14, v27, 7, v24 // (op 4819) 


s_nop 0
v_readfirstlane_b32 s0, v14 // (op 4819) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 4251) 
v_mov_b64 v[14:15], s[6:7] // (op 4251) convert


v_mov_b64 v[32:33], v[18:19] // (op 4251) 
s_add_u32 s1, 40960, s0 // (op 4251) 
s_mov_b32 m0, s1 // (op 4251) reg expression
s_mov_b32 s6, 128 // (op 4251) call()
buffer_load_dwordx4 v32, s[48:51], s6 offen offset: 0 lds // (op 4251) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v14 // (op 4251) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 4251) most significant half

s_add_u32 m0, m0, 4096 // (op 4251) 
buffer_load_dwordx4 v32, s[48:51], s6 offen offset: 0 lds // (op 4251) Load value direct to lds









buffer_load_ubyte v24, v22, s[56:59], 0 offen offset:4 // (op 4039) Load value

v_and_b32 v22, -4, v1 // (op 4752) 
v_sub_u32 v27, v1, v22 // (op 4752) 

v_lshrrev_b32 v22, 2, v1 // (op 4752) 
v_lshl_add_u32 v14, v22, 3, v27 // (op 4752) 





s_waitcnt vmcnt(13)
ds_write_b8 v14, v30 offset:49152 // (op 4042) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v14, v31 offset:49156 // (op 4042) Store local data 

v_and_b32 v22, -4, v1 // (op 4694) 
v_sub_u32 v27, v1, v22 // (op 4694) 

v_and_b32 v22, 4294967292, v1 // (op 4694) 
v_add_u32 v14, v22, v27 // (op 4694) 





s_waitcnt vmcnt(9)
ds_write_b8 v14, v26 offset:49664 // (op 4044) Store local data 

s_barrier  // (op 5039) 



ds_read_b128 v[32:35], v4 // (op 4069) Load local data 
ds_read_b128 v[36:39], v4 offset:64 // (op 4069) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 5074) 



ds_read_u8 v22, v5 offset:49152 // (op 4099) Load local data 



ds_read_b128 v[40:43], v4 offset:4096 // (op 4074) Load local data 
ds_read_b128 v[44:47], v4 offset:4160 // (op 4074) Load local data 



ds_read_u8 v27, v5 offset:49280 // (op 4104) Load local data 



ds_read_b128 v[48:51], v4 offset:8192 // (op 4079) Load local data 
ds_read_b128 v[52:55], v4 offset:8256 // (op 4079) Load local data 



ds_read_u8 v14, v5 offset:49408 // (op 4109) Load local data 



ds_read_b128 v[56:59], v4 offset:12288 // (op 4084) Load local data 
ds_read_b128 v[60:63], v4 offset:12352 // (op 4084) Load local data 



ds_read_u8 v15, v5 offset:49536 // (op 4114) Load local data 
s_barrier  // (op 5023) 



ds_read_b128 v[64:67], v3 offset:16384 // (op 4059) Load local data 
ds_read_b128 v[68:71], v3 offset:16448 // (op 4059) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 5056) 



ds_read_u8 v18, v0 offset:49664 // (op 4089) Load local data 



ds_read_b128 v[72:75], v3 offset:20480 // (op 4064) Load local data 
ds_read_b128 v[76:79], v3 offset:20544 // (op 4064) Load local data 



ds_read_u8 v19, v0 offset:49792 // (op 4094) Load local data 
v_and_b32 v80, -8, v1 // (op 4564) 
v_sub_u32 v81, v1, v80 // (op 4564) 

v_lshlrev_b32 v80, 4, v81 // (op 4564) 

v_lshrrev_b32 v81, 3, v1 // (op 4564) 
v_lshl_add_u32 v82, v81, 7, v80 // (op 4564) 


s_nop 0
v_readfirstlane_b32 s0, v82 // (op 4564) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -8, v1 // (op 4621) 
v_sub_u32 v81, v1, v80 // (op 4621) 

v_lshlrev_b32 v80, 4, v81 // (op 4621) 

v_lshrrev_b32 v81, 3, v1 // (op 4621) 
v_lshl_add_u32 v82, v81, 7, v80 // (op 4621) 


s_nop 0
v_readfirstlane_b32 s1, v82 // (op 4621) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -4, v1 // (op 4678) 
v_sub_u32 v81, v1, v80 // (op 4678) 

v_and_b32 v80, 4294967292, v1 // (op 4678) 
v_add_u32 v82, v80, v81 // (op 4678) 


v_and_b32 v80, -4, v1 // (op 4736) 
v_sub_u32 v81, v1, v80 // (op 4736) 

v_lshrrev_b32 v80, 2, v1 // (op 4736) 
v_lshl_add_u32 v83, v80, 3, v81 // (op 4736) 


v_and_b32 v80, -8, v1 // (op 4804) 
v_sub_u32 v81, v1, v80 // (op 4804) 

v_lshlrev_b32 v80, 4, v81 // (op 4804) 

v_lshrrev_b32 v81, 3, v1 // (op 4804) 
v_lshl_add_u32 v84, v81, 7, v80 // (op 4804) 


s_nop 0
v_readfirstlane_b32 s6, v84 // (op 4804) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -8, v1 // (op 4848) 
v_sub_u32 v81, v1, v80 // (op 4848) 

v_lshlrev_b32 v80, 4, v81 // (op 4848) 

v_lshrrev_b32 v81, 3, v1 // (op 4848) 
v_lshl_add_u32 v84, v81, 7, v80 // (op 4848) 


s_nop 0
v_readfirstlane_b32 s7, v84 // (op 4848) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -4, v1 // (op 4892) 
v_sub_u32 v81, v1, v80 // (op 4892) 

v_and_b32 v80, 4294967292, v1 // (op 4892) 
v_add_u32 v84, v80, v81 // (op 4892) 


v_and_b32 v80, -4, v1 // (op 4921) 
v_sub_u32 v81, v1, v80 // (op 4921) 

v_lshrrev_b32 v80, 2, v1 // (op 4921) 
v_lshl_add_u32 v85, v80, 3, v81 // (op 4921) 


s_mov_b32 s10, 0 // (op 1051) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 1050) 
s_mov_b32 s11, s14 // (op 1050) convert

s_and_b32 s14, s11, 4294967294 // (op 1050) 

s_cmp_lt_u32 s10, s14 // (op 1050) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_1050 // (op 1050) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_1050 if false)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ForLoopTop_KLoop_1050:
s_add_u32 s10, s10, 2 // (op 1052) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:71], v[32:39], a[0:3], v18, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 90) 
s_barrier  // (op 5042) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:71], v[40:47], a[4:7], v18, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 380) 
s_lshl_b64 s[14:15], s[24:25], 5 // (op 4211) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:71], v[48:55], a[8:11], v18, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 556) 
v_mov_b64 v[80:81], s[14:15] // (op 4211) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[64:71], v[56:63], a[12:15], v18, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 732) 
v_mov_b64 v[86:87], v[16:17] // (op 4211) 
v_mov_b64 v[88:89], v[86:87] // (op 4211) 
s_mov_b32 m0, s1 // (op 4211) reg expression
s_mov_b32 s11, 256 // (op 4211) call()
buffer_load_dwordx4 v88, s[52:55], s11 offen offset: 0 lds // (op 4211) Load value direct to lds
v_add_co_u32 v88, s[14:15], v88, v80 // (op 4211) least significant half
v_addc_co_u32 v89, s[14:15], 0, v89, s[14:15] // (op 4211) most significant half

s_add_u32 m0, m0, 4096 // (op 4211) 
buffer_load_dwordx4 v88, s[52:55], s11 offen offset: 0 lds // (op 4211) Load value direct to lds
v_add_co_u32 v88, s[14:15], v88, v80 // (op 4211) least significant half
v_addc_co_u32 v89, s[14:15], 0, v89, s[14:15] // (op 4211) most significant half

s_add_u32 m0, m0, 4096 // (op 4211) 
buffer_load_dwordx4 v88, s[52:55], s11 offen offset: 0 lds // (op 4211) Load value direct to lds
v_add_co_u32 v88, s[14:15], v88, v80 // (op 4211) least significant half
v_addc_co_u32 v89, s[14:15], 0, v89, s[14:15] // (op 4211) most significant half

s_add_u32 m0, m0, 4096 // (op 4211) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:79], v[32:39], a[16:19], v19, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1227) 
buffer_load_dwordx4 v88, s[52:55], s11 offen offset: 0 lds // (op 4211) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:79], v[40:47], a[20:23], v19, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1181) 
v_mov_b64 v[80:81], s[32:33] // (op 25) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:79], v[48:55], a[24:27], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1135) 
v_mov_b32 v88, v25 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:79], v[56:63], a[28:31], v19, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1089) 
v_mov_b32 v89, v88 // (op 25) 
buffer_load_ubyte v30, v89, s[60:63], 0 offen offset:8 // (op 25) Load value
v_add_u32 v89, v89, v80 // (op 25) 
buffer_load_ubyte v31, v89, s[60:63], 0 offen offset:8 // (op 25) Load value


s_barrier  // (op 5026) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 4206) 
v_mov_b64 v[80:81], s[14:15] // (op 4206) convert


v_mov_b64 v[90:91], v[20:21] // (op 4206) 
v_mov_b64 v[92:93], v[90:91] // (op 4206) 
s_add_u32 s11, 16384, s0 // (op 4206) 
s_mov_b32 m0, s11 // (op 4206) reg expression
s_mov_b32 s14, 256 // (op 4206) call()
buffer_load_dwordx4 v92, s[48:51], s14 offen offset: 0 lds // (op 4206) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v80 // (op 4206) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 4206) most significant half

s_add_u32 m0, m0, 4096 // (op 4206) 
buffer_load_dwordx4 v92, s[48:51], s14 offen offset: 0 lds // (op 4206) Load value direct to lds






v_mov_b32 v89, v23 // (op 17) 

buffer_load_ubyte v26, v89, s[56:59], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 5114) 



s_waitcnt vmcnt(13)
ds_write_b8 v85, v28 offset:49920 // (op 259) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v85, v29 offset:49924 // (op 259) Store local data 
s_barrier  // (op 5108) 



s_waitcnt vmcnt(9)
ds_write_b8 v84, v24 offset:50432 // (op 258) Store local data 
s_barrier  // (op 5097) 



ds_read_b128 v[92:95], v8 offset:24576 // (op 251) Load local data 
ds_read_b128 v[96:99], v8 offset:24640 // (op 251) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 5111) 



ds_read_u8 v80, v9 offset:49920 // (op 255) Load local data 



ds_read_b128 v[100:103], v8 offset:28672 // (op 385) Load local data 
ds_read_b128 v[104:107], v8 offset:28736 // (op 385) Load local data 



ds_read_u8 v81, v9 offset:50048 // (op 389) Load local data 



ds_read_b128 v[108:111], v8 offset:32768 // (op 561) Load local data 
ds_read_b128 v[112:115], v8 offset:32832 // (op 561) Load local data 



ds_read_u8 v116, v9 offset:50176 // (op 565) Load local data 



ds_read_b128 v[118:121], v8 offset:36864 // (op 737) Load local data 
ds_read_b128 v[122:125], v8 offset:36928 // (op 737) Load local data 



ds_read_u8 v117, v9 offset:50304 // (op 741) Load local data 
s_barrier  // (op 5091) 
s_barrier  // (op 5159) 



ds_read_b128 v[64:67], v7 offset:40960 // (op 249) Load local data 
ds_read_b128 v[68:71], v7 offset:41024 // (op 249) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 5104) 



ds_read_u8 v18, v6 offset:50432 // (op 253) Load local data 



ds_read_b128 v[72:75], v7 offset:45056 // (op 1092) Load local data 
ds_read_b128 v[76:79], v7 offset:45120 // (op 1092) Load local data 



ds_read_u8 v19, v6 offset:50560 // (op 1095) Load local data 
s_barrier  // (op 5185) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:71], v[92:99], a[0:3], v18, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 256) 
s_barrier  // (op 5101) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:71], v[100:107], a[4:7], v18, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 381) 
s_lshl_b64 s[14:15], s[24:25], 5 // (op 4234) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:71], v[108:115], a[8:11], v18, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 557) 
v_mov_b64 v[126:127], s[14:15] // (op 4234) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[64:71], v[118:125], a[12:15], v18, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 733) 
v_mov_b64 v[128:129], v[86:87] // (op 4234) 
s_add_u32 s11, 24576, s7 // (op 4234) 
s_mov_b32 m0, s11 // (op 4234) reg expression
s_mov_b32 s14, 384 // (op 4234) call()
buffer_load_dwordx4 v128, s[52:55], s14 offen offset: 0 lds // (op 4234) Load value direct to lds
v_add_co_u32 v128, s[18:19], v128, v126 // (op 4234) least significant half
v_addc_co_u32 v129, s[18:19], 0, v129, s[18:19] // (op 4234) most significant half

s_add_u32 m0, m0, 4096 // (op 4234) 
buffer_load_dwordx4 v128, s[52:55], s14 offen offset: 0 lds // (op 4234) Load value direct to lds
v_add_co_u32 v128, s[18:19], v128, v126 // (op 4234) least significant half
v_addc_co_u32 v129, s[18:19], 0, v129, s[18:19] // (op 4234) most significant half

s_add_u32 m0, m0, 4096 // (op 4234) 
buffer_load_dwordx4 v128, s[52:55], s14 offen offset: 0 lds // (op 4234) Load value direct to lds
v_add_co_u32 v128, s[18:19], v128, v126 // (op 4234) least significant half
v_addc_co_u32 v129, s[18:19], 0, v129, s[18:19] // (op 4234) most significant half

s_add_u32 m0, m0, 4096 // (op 4234) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:79], v[92:99], a[16:19], v19, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1228) 
buffer_load_dwordx4 v128, s[52:55], s14 offen offset: 0 lds // (op 4234) Load value direct to lds








v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:79], v[100:107], a[20:23], v19, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1182) 
v_add_co_u32 v16, vcc, 256, v16 // (op 4372) least significant half


v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 4372) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:79], v[108:115], a[24:27], v19, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1136) 
v_mov_b64 v[80:81], s[32:33] // (op 262) convert



v_mov_b32 v86, v88 // (op 262) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:79], v[118:125], a[28:31], v19, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1090) 
buffer_load_ubyte v28, v86, s[60:63], 0 offen offset:12 // (op 262) Load value


v_add_u32 v86, v86, v80 // (op 262) 
buffer_load_ubyte v29, v86, s[60:63], 0 offen offset:12 // (op 262) Load value



s_barrier  // (op 5094) 
v_add_u32 v25, 8, v25 // (op 4500) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 4229) 
v_mov_b64 v[80:81], s[14:15] // (op 4229) convert


v_mov_b64 v[86:87], v[90:91] // (op 4229) 
s_add_u32 s11, 40960, s6 // (op 4229) 
s_mov_b32 m0, s11 // (op 4229) reg expression
s_mov_b32 s14, 384 // (op 4229) call()
buffer_load_dwordx4 v86, s[48:51], s14 offen offset: 0 lds // (op 4229) Load value direct to lds
v_add_co_u32 v86, s[18:19], v86, v80 // (op 4229) least significant half
v_addc_co_u32 v87, s[18:19], 0, v87, s[18:19] // (op 4229) most significant half

s_add_u32 m0, m0, 4096 // (op 4229) 
buffer_load_dwordx4 v86, s[48:51], s14 offen offset: 0 lds // (op 4229) Load value direct to lds








v_add_co_u32 v20, vcc, 256, v20 // (op 4308) least significant half
buffer_load_ubyte v24, v89, s[56:59], 0 offen offset:12 // (op 261) Load value
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 4308) most significant half

s_barrier  // (op 5077) 
v_add_u32 v23, 8, v23 // (op 4436) 



s_waitcnt vmcnt(13)
ds_write_b8 v83, v30 offset:49152 // (op 30) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v83, v31 offset:49156 // (op 30) Store local data 
s_barrier  // (op 5059) 



s_waitcnt vmcnt(9)
ds_write_b8 v82, v26 offset:49664 // (op 21) Store local data 
s_barrier  // (op 5048) 



ds_read_b128 v[32:35], v4 // (op 77) Load local data 
ds_read_b128 v[36:39], v4 offset:64 // (op 77) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 5084) 



ds_read_u8 v22, v5 offset:49152 // (op 87) Load local data 



ds_read_b128 v[40:43], v4 offset:4096 // (op 375) Load local data 
ds_read_b128 v[44:47], v4 offset:4160 // (op 375) Load local data 



ds_read_u8 v27, v5 offset:49280 // (op 379) Load local data 



ds_read_b128 v[48:51], v4 offset:8192 // (op 551) Load local data 
ds_read_b128 v[52:55], v4 offset:8256 // (op 551) Load local data 



ds_read_u8 v14, v5 offset:49408 // (op 555) Load local data 



ds_read_b128 v[56:59], v4 offset:12288 // (op 727) Load local data 
ds_read_b128 v[60:63], v4 offset:12352 // (op 727) Load local data 



ds_read_u8 v15, v5 offset:49536 // (op 731) Load local data 
s_barrier  // (op 5032) 



ds_read_b128 v[64:67], v3 offset:16384 // (op 72) Load local data 
ds_read_b128 v[68:71], v3 offset:16448 // (op 72) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 5066) 



ds_read_u8 v18, v0 offset:49664 // (op 82) Load local data 



ds_read_b128 v[72:75], v3 offset:20480 // (op 1084) Load local data 
ds_read_b128 v[76:79], v3 offset:20544 // (op 1084) Load local data 



ds_read_u8 v19, v0 offset:49792 // (op 1087) Load local data 
s_ashr_i64 s[14:15], s[36:37], 7 // (op 1050) 
s_mov_b32 s11, s14 // (op 1050) convert

s_and_b32 s14, s11, 4294967294 // (op 1050) 

s_cmp_lt_u32 s10, s14 // (op 1050) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ForLoopTop_KLoop_1050 // (op 1050) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ForLoopTop_KLoop_1050 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_1050:

















s_barrier  // (op 5175) 
v_lshrrev_b32 v16, 3, v1 // (op 4264) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4264) 
v_lshl_add_u32 v17, s16, 6, v16 // (op 4264) 

s_lshr_b32 s6, s0, 1 // (op 4264) 

v_and_b32 v16, -8, v1 // (op 4264) 
v_sub_u32 v20, v1, v16 // (op 4264) 

v_lshlrev_b32 v16, 4, v20 // (op 4264) 

v_lshl_add_u32 v20, s6, 8, v16 // (op 4264) 


v_mov_b32 v23, 0 // (op 4264) convert
v_mov_b32 v22, v20 // (op 4264) convert

v_mul_lo_u32 v21, v17, s13 // (op 4264) most significant: low of low * high
v_mul_hi_u32 v16, v17, s12 // (op 4264) most significant: high of low * low
v_mul_lo_u32 v20, v17, s12 // (op 4264) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 4264) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v22 // (op 4264) least significant half
v_addc_co_u32 v21, s[0:1], v21, v23, s[0:1] // (op 4264) most significant half



s_mul_i32 s1, s12, s23 // (op 4266) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 4266) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 4266) least significant
s_mul_hi_u32 s7, s12, s22 // (op 4266) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4266) most significant: sum
s_add_u32 s1, s1, s7 // (op 4266) most significant: sum


s_mov_b32 s48, s20 // (op 4266) reg expression
s_mov_b32 s49, s21 // (op 4266) reg expression
s_mov_b32 s50, s0 // (op 4266) reg expression
s_mov_b32 s51, 131072 // (op 4266) call()

v_and_b32 v16, -8, v1 // (op 4338) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4338) 
v_sub_u32 v17, v1, v16 // (op 4338) 

s_lshr_b32 s6, s0, 1 // (op 4338) 

v_lshlrev_b32 v16, 4, v17 // (op 4338) 

v_lshl_add_u32 v17, s6, 8, v16 // (op 4338) 


v_mov_b32 v23, 0 // (op 4338) convert
v_mov_b32 v22, v17 // (op 4338) convert

v_lshrrev_b32 v16, 3, v1 // (op 4338) 
v_lshl_add_u32 v17, s17, 7, v16 // (op 4338) 

v_mul_lo_u32 v31, v17, s25 // (op 4338) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 4338) most significant: high of low * low
v_mul_lo_u32 v30, v17, s24 // (op 4338) least significant: low of low * low
v_add_u32 v31, v31, v16 // (op 4338) most significant: sum


v_add_co_u32 v16, s[0:1], v22, v30 // (op 4338) least significant half
v_addc_co_u32 v17, s[0:1], v23, v31, s[0:1] // (op 4338) most significant half



s_mul_i32 s1, s24, s9 // (op 4340) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 4340) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 4340) least significant
s_mul_hi_u32 s7, s24, s8 // (op 4340) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4340) most significant: sum
s_add_u32 s1, s1, s7 // (op 4340) most significant: sum


s_mov_b32 s52, s26 // (op 4340) reg expression
s_mov_b32 s53, s27 // (op 4340) reg expression
s_mov_b32 s54, s0 // (op 4340) reg expression
s_mov_b32 s55, 131072 // (op 4340) call()

v_lshrrev_b32 v22, 2, v1 // (op 4402) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4402) 
v_lshl_add_u32 v23, s16, 6, v22 // (op 4402) 

s_lshr_b32 s6, s0, 1 // (op 4402) 

v_and_b32 v22, -4, v1 // (op 4402) 
v_sub_u32 v25, v1, v22 // (op 4402) 

v_lshl_add_u32 v22, s6, 3, v25 // (op 4402) 


v_mul_lo_u32 v25, v23, s28 // (op 4402) 
v_add_u32 v25, v25, v22 // (op 4402) 


s_mul_i32 s1, s28, s23 // (op 4404) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4404) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4404) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4404) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4404) most significant: sum
s_add_u32 s1, s1, s7 // (op 4404) most significant: sum


s_mov_b32 s56, s30 // (op 4404) reg expression
s_mov_b32 s57, s31 // (op 4404) reg expression
s_mov_b32 s58, s0 // (op 4404) reg expression
s_mov_b32 s59, 131072 // (op 4404) call()

v_and_b32 v26, -4, v1 // (op 4466) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4466) 
v_sub_u32 v22, v1, v26 // (op 4466) 

s_lshr_b32 s6, s0, 1 // (op 4466) 

v_lshrrev_b32 v23, 2, v1 // (op 4466) 
v_lshl_add_u32 v26, s6, 3, v22 // (op 4466) 


v_lshlrev_b32 v22, 1, v23 // (op 4466) 

v_lshl_add_u32 v23, s17, 7, v22 // (op 4466) 

v_mul_lo_u32 v22, v23, s32 // (op 4466) 

v_add_u32 v23, v26, v22 // (op 4466) 


s_mul_i32 s1, s32, s9 // (op 4468) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4468) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4468) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4468) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4468) most significant: sum
s_add_u32 s1, s1, s7 // (op 4468) most significant: sum


s_mov_b32 s60, s34 // (op 4468) reg expression
s_mov_b32 s61, s35 // (op 4468) reg expression
s_mov_b32 s62, s0 // (op 4468) reg expression
s_mov_b32 s63, 131072 // (op 4468) call()

v_and_b32 v22, -8, v1 // (op 4549) 
v_sub_u32 v26, v1, v22 // (op 4549) 

v_lshlrev_b32 v22, 4, v26 // (op 4549) 

v_lshrrev_b32 v26, 3, v1 // (op 4549) 
v_lshl_add_u32 v30, v26, 7, v22 // (op 4549) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 4549) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 4608) 
v_sub_u32 v26, v1, v22 // (op 4608) 

v_lshlrev_b32 v22, 4, v26 // (op 4608) 

v_lshrrev_b32 v26, 3, v1 // (op 4608) 
v_lshl_add_u32 v30, v26, 7, v22 // (op 4608) 


s_nop 0
v_readfirstlane_b32 s1, v30 // (op 4608) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 4665) 
v_sub_u32 v26, v1, v22 // (op 4665) 

v_and_b32 v22, 4294967292, v1 // (op 4665) 
v_add_u32 v30, v22, v26 // (op 4665) 


v_and_b32 v22, -4, v1 // (op 4723) 
v_sub_u32 v26, v1, v22 // (op 4723) 

v_lshrrev_b32 v22, 2, v1 // (op 4723) 
v_lshl_add_u32 v31, v22, 3, v26 // (op 4723) 


s_ashr_i64 s[6:7], s[36:37], 7 // (op 1025) 
s_and_b32 s10, s6, 4294967294 // (op 1025) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 1024) 
s_mov_b32 s11, s6 // (op 1024) convert

s_cmp_lt_u32 s10, s11 // (op 1024) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_1024 // (op 1024) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_1024 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_1024:
s_barrier  // (op 5029) 
s_add_u32 s10, s10, 1 // (op 1026) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 4216) 
v_mov_b64 v[14:15], s[6:7] // (op 4216) convert


v_mov_b64 v[18:19], v[20:21] // (op 4216) 
v_mov_b64 v[26:27], v[18:19] // (op 4216) 
s_add_u32 s11, 16384, s0 // (op 4216) 
s_mov_b32 m0, s11 // (op 4216) reg expression
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 4216) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v14 // (op 4216) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 4216) most significant half

s_add_u32 m0, m0, 4096 // (op 4216) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 4216) Load value direct to lds




v_add_co_u32 v20, vcc, 128, v20 // (op 4254) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 4254) most significant half
s_barrier  // (op 5045) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 4222) 
v_mov_b64 v[14:15], s[6:7] // (op 4222) convert


v_mov_b64 v[18:19], v[16:17] // (op 4222) 
v_mov_b64 v[26:27], v[18:19] // (op 4222) 
s_mov_b32 m0, s1 // (op 4222) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 4222) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v14 // (op 4222) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 4222) most significant half

s_add_u32 m0, m0, 4096 // (op 4222) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 4222) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v14 // (op 4222) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 4222) most significant half

s_add_u32 m0, m0, 4096 // (op 4222) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 4222) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v14 // (op 4222) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 4222) most significant half

s_add_u32 m0, m0, 4096 // (op 4222) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 4222) Load value direct to lds



v_add_co_u32 v16, vcc, 128, v16 // (op 4330) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 4330) most significant half


v_mov_b32 v22, v25 // (op 195) 

buffer_load_ubyte v24, v22, s[56:59], 0 offen offset: 0 // (op 195) Load value

v_add_u32 v25, 4, v25 // (op 4394) 

s_barrier  // (op 5062) 
v_mov_b64 v[14:15], s[32:33] // (op 196) convert




v_mov_b32 v22, v23 // (op 196) 
v_mov_b32 v18, v22 // (op 196) 
buffer_load_ubyte v26, v18, s[60:63], 0 offen offset: 0 // (op 196) Load value
v_add_u32 v18, v18, v14 // (op 196) 
buffer_load_ubyte v27, v18, s[60:63], 0 offen offset: 0 // (op 196) Load value



v_add_u32 v23, 4, v23 // (op 4458) 
s_waitcnt vmcnt(2)
ds_write_b8 v30, v24 offset:49664 // (op 192) Store local data 

s_barrier  // (op 5035) 



s_barrier  // (op 5080) 
ds_read_b128 v[32:35], v3 offset:16384 // (op 183) Load local data 



ds_read_b128 v[36:39], v3 offset:16448 // (op 183) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v31, v26 offset:49152 // (op 193) Store local data 
s_barrier  // (op 5052) 
s_waitcnt vmcnt(0)
ds_write_b8 v31, v27 offset:49156 // (op 193) Store local data 




ds_read_b128 v[40:43], v4 // (op 185) Load local data 
ds_read_b128 v[44:47], v4 offset:64 // (op 185) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 5070) 



ds_read_u8 v22, v0 offset:49664 // (op 187) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 5087) 



ds_read_u8 v24, v5 offset:49152 // (op 189) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:47], a[0:3], v22, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 190) 
ds_read_b128 v[48:51], v4 offset:4096 // (op 354) Load local data 
ds_read_b128 v[52:55], v4 offset:4160 // (op 354) Load local data 



ds_read_u8 v14, v5 offset:49280 // (op 358) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:39], v[48:55], a[4:7], v22, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 359) 
ds_read_b128 v[56:59], v4 offset:8192 // (op 530) Load local data 
ds_read_b128 v[60:63], v4 offset:8256 // (op 530) Load local data 



ds_read_u8 v15, v5 offset:49408 // (op 534) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[32:39], v[56:63], a[8:11], v22, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 535) 
ds_read_b128 v[64:67], v4 offset:12288 // (op 706) Load local data 
ds_read_b128 v[68:71], v4 offset:12352 // (op 706) Load local data 



ds_read_u8 v18, v5 offset:49536 // (op 710) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:39], v[64:71], a[12:15], v22, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 711) 
ds_read_b128 v[72:75], v3 offset:20480 // (op 968) Load local data 


ds_read_b128 v[76:79], v3 offset:20544 // (op 968) Load local data 



ds_read_u8 v19, v0 offset:49792 // (op 971) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:79], v[40:47], a[16:19], v19, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 973) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:79], v[48:55], a[20:23], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1004) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:79], v[56:63], a[24:27], v19, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1035) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:79], v[64:71], a[28:31], v19, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1066) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 1024) 
s_mov_b32 s11, s6 // (op 1024) convert

s_cmp_lt_u32 s10, s11 // (op 1024) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_1024 // (op 1024) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_1024 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_1024:























s_barrier  // (op 5167) 









v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_135
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 

v_lshrrev_b32 v6, 16, v14 // (op 56) 
v_lshrrev_b32 v7, 16, v15 // (op 56) 
v_lshrrev_b32 v8, 16, v16 // (op 56) 
v_lshrrev_b32 v9, 16, v17 // (op 56) 


v_pack_b32_f16 v4, v6, v7 // (op 58) 
v_pack_b32_f16 v5, v8, v9 // (op 58) 

v_mov_b32 v0, v13 // (op 58) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 58) Store value



v_accvgpr_read v6, a4 // (op 1873) 
v_accvgpr_read v7, a5 // (op 1873) 
v_accvgpr_read v8, a6 // (op 1873) 
v_accvgpr_read v9, a7 // (op 1873) 
v_mul_f32 v14, v11, v6 // (op 1873) 
v_mul_f32 v15, v11, v7 // (op 1873) 
v_mul_f32 v16, v11, v8 // (op 1873) 
v_mul_f32 v17, v11, v9 // (op 1873) 

v_lshrrev_b32 v6, 16, v14 // (op 1875) 
v_lshrrev_b32 v7, 16, v15 // (op 1875) 
v_lshrrev_b32 v8, 16, v16 // (op 1875) 
v_lshrrev_b32 v9, 16, v17 // (op 1875) 


v_pack_b32_f16 v4, v6, v7 // (op 1876) 
v_pack_b32_f16 v5, v8, v9 // (op 1876) 

s_lshl_b32 s0, s4, 6 // (op 1876) 
v_add_u32 v0, v13, s0 // (op 1876) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1876) Store value



v_accvgpr_read v6, a8 // (op 1887) 
v_accvgpr_read v7, a9 // (op 1887) 
v_accvgpr_read v8, a10 // (op 1887) 
v_accvgpr_read v9, a11 // (op 1887) 
v_mul_f32 v14, v11, v6 // (op 1887) 
v_mul_f32 v15, v11, v7 // (op 1887) 
v_mul_f32 v16, v11, v8 // (op 1887) 
v_mul_f32 v17, v11, v9 // (op 1887) 

v_lshrrev_b32 v6, 16, v14 // (op 1889) 
v_lshrrev_b32 v7, 16, v15 // (op 1889) 
v_lshrrev_b32 v8, 16, v16 // (op 1889) 
v_lshrrev_b32 v9, 16, v17 // (op 1889) 


v_pack_b32_f16 v4, v6, v7 // (op 1890) 
v_pack_b32_f16 v5, v8, v9 // (op 1890) 

s_lshl_b32 s0, s4, 7 // (op 1890) 
v_add_u32 v0, v13, s0 // (op 1890) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1890) Store value



v_accvgpr_read v6, a12 // (op 1901) 
v_accvgpr_read v7, a13 // (op 1901) 
v_accvgpr_read v8, a14 // (op 1901) 
v_accvgpr_read v9, a15 // (op 1901) 
v_mul_f32 v14, v11, v6 // (op 1901) 
v_mul_f32 v15, v11, v7 // (op 1901) 
v_mul_f32 v16, v11, v8 // (op 1901) 
v_mul_f32 v17, v11, v9 // (op 1901) 

v_lshrrev_b32 v6, 16, v14 // (op 1903) 
v_lshrrev_b32 v7, 16, v15 // (op 1903) 
v_lshrrev_b32 v8, 16, v16 // (op 1903) 
v_lshrrev_b32 v9, 16, v17 // (op 1903) 


v_pack_b32_f16 v4, v6, v7 // (op 1904) 
v_pack_b32_f16 v5, v8, v9 // (op 1904) 

s_lshl_b32 s0, s4, 6 // (op 1904) 
s_mul_i32 s1, 3, s0 // (op 1904) 

v_add_u32 v0, v13, s1 // (op 1904) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1904) Store value



v_accvgpr_read v6, a16 // (op 1937) 
v_accvgpr_read v7, a17 // (op 1937) 
v_accvgpr_read v8, a18 // (op 1937) 
v_accvgpr_read v9, a19 // (op 1937) 
v_mul_f32 v14, v11, v6 // (op 1937) 
v_mul_f32 v15, v11, v7 // (op 1937) 
v_mul_f32 v16, v11, v8 // (op 1937) 
v_mul_f32 v17, v11, v9 // (op 1937) 

v_lshrrev_b32 v6, 16, v14 // (op 1939) 
v_lshrrev_b32 v7, 16, v15 // (op 1939) 
v_lshrrev_b32 v8, 16, v16 // (op 1939) 
v_lshrrev_b32 v9, 16, v17 // (op 1939) 


v_pack_b32_f16 v4, v6, v7 // (op 1941) 
v_pack_b32_f16 v5, v8, v9 // (op 1941) 

v_add_u32 v0, 64, v13 // (op 1941) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1941) Store value



v_accvgpr_read v6, a20 // (op 1951) 
v_accvgpr_read v7, a21 // (op 1951) 
v_accvgpr_read v8, a22 // (op 1951) 
v_accvgpr_read v9, a23 // (op 1951) 
v_mul_f32 v14, v11, v6 // (op 1951) 
v_mul_f32 v15, v11, v7 // (op 1951) 
v_mul_f32 v16, v11, v8 // (op 1951) 
v_mul_f32 v17, v11, v9 // (op 1951) 

v_lshrrev_b32 v6, 16, v14 // (op 1953) 
v_lshrrev_b32 v7, 16, v15 // (op 1953) 
v_lshrrev_b32 v8, 16, v16 // (op 1953) 
v_lshrrev_b32 v9, 16, v17 // (op 1953) 


v_pack_b32_f16 v4, v6, v7 // (op 1943) 
v_pack_b32_f16 v5, v8, v9 // (op 1943) 

s_lshl_b32 s0, s4, 6 // (op 1943) 
s_add_u32 s0, s0, 64 // (op 1943) 
v_add_u32 v0, v13, s0 // (op 1943) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1943) Store value



v_accvgpr_read v6, a24 // (op 1957) 
v_accvgpr_read v7, a25 // (op 1957) 
v_accvgpr_read v8, a26 // (op 1957) 
v_accvgpr_read v9, a27 // (op 1957) 
v_mul_f32 v14, v11, v6 // (op 1957) 
v_mul_f32 v15, v11, v7 // (op 1957) 
v_mul_f32 v16, v11, v8 // (op 1957) 
v_mul_f32 v17, v11, v9 // (op 1957) 

v_lshrrev_b32 v6, 16, v14 // (op 1959) 
v_lshrrev_b32 v7, 16, v15 // (op 1959) 
v_lshrrev_b32 v8, 16, v16 // (op 1959) 
v_lshrrev_b32 v9, 16, v17 // (op 1959) 


v_pack_b32_f16 v4, v6, v7 // (op 1945) 
v_pack_b32_f16 v5, v8, v9 // (op 1945) 

s_lshl_b32 s0, s4, 7 // (op 1945) 
s_add_u32 s0, s0, 64 // (op 1945) 
v_add_u32 v0, v13, s0 // (op 1945) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1945) Store value



v_accvgpr_read v6, a28 // (op 1963) 
v_accvgpr_read v7, a29 // (op 1963) 
v_accvgpr_read v8, a30 // (op 1963) 
v_accvgpr_read v9, a31 // (op 1963) 
v_mul_f32 v14, v11, v6 // (op 1963) 
v_mul_f32 v15, v11, v7 // (op 1963) 
v_mul_f32 v16, v11, v8 // (op 1963) 
v_mul_f32 v17, v11, v9 // (op 1963) 

v_lshrrev_b32 v6, 16, v14 // (op 1965) 
v_lshrrev_b32 v7, 16, v15 // (op 1965) 
v_lshrrev_b32 v8, 16, v16 // (op 1965) 
v_lshrrev_b32 v9, 16, v17 // (op 1965) 


v_pack_b32_f16 v4, v6, v7 // (op 1947) 
v_pack_b32_f16 v5, v8, v9 // (op 1947) 

s_lshl_b32 s0, s4, 6 // (op 1947) 
s_mul_i32 s1, 3, s0 // (op 1947) 
s_add_u32 s1, s1, 64 // (op 1947) 

v_add_u32 v0, v13, s1 // (op 1947) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1947) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_135:


v_add_u32 v0, 64, v12 // (op 2133) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 2133) Load value

v_accvgpr_read v4, a16 // (op 2120) 
v_accvgpr_read v5, a17 // (op 2120) 
v_accvgpr_read v6, a18 // (op 2120) 
v_accvgpr_read v7, a19 // (op 2120) 
v_mul_f32 v14, v11, v4 // (op 2120) 
v_mul_f32 v15, v11, v5 // (op 2120) 
v_mul_f32 v16, v11, v6 // (op 2120) 
v_mul_f32 v17, v11, v7 // (op 2120) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 2132) 
v_lshrrev_b32 v7, 16, v8 // (op 2132) 
v_lshlrev_b32 v6, 16, v6 // (op 2132) 
v_lshlrev_b32 v7, 16, v7 // (op 2132) 
v_mul_f32 v18, v10, v6 // (op 2132) 
v_mul_f32 v19, v10, v7 // (op 2132) 
v_and_b32 v6, 65535, v9 // (op 2132) 
v_lshrrev_b32 v7, 16, v9 // (op 2132) 
v_lshlrev_b32 v6, 16, v6 // (op 2132) 
v_lshlrev_b32 v7, 16, v7 // (op 2132) 
v_mul_f32 v20, v10, v6 // (op 2132) 
v_mul_f32 v21, v10, v7 // (op 2132) 


v_add_f32 v6, v18, v14 // (op 2121) 
v_add_f32 v7, v19, v15 // (op 2121) 
v_add_f32 v8, v20, v16 // (op 2121) 
v_add_f32 v9, v21, v17 // (op 2121) 


v_lshrrev_b32 v14, 16, v6 // (op 2122) 
v_lshrrev_b32 v15, 16, v7 // (op 2122) 
v_lshrrev_b32 v16, 16, v8 // (op 2122) 
v_lshrrev_b32 v17, 16, v9 // (op 2122) 


v_pack_b32_f16 v8, v14, v15 // (op 2124) 
v_pack_b32_f16 v9, v16, v17 // (op 2124) 

v_add_u32 v0, 64, v13 // (op 2124) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 2124) Store value





s_lshl_b32 s0, s2, 6 // (op 2138) 
s_add_u32 s0, s0, 64 // (op 2138) 
v_add_u32 v0, v12, s0 // (op 2138) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 2138) Load value

v_accvgpr_read v4, a20 // (op 2150) 
v_accvgpr_read v5, a21 // (op 2150) 
v_accvgpr_read v6, a22 // (op 2150) 
v_accvgpr_read v7, a23 // (op 2150) 
v_mul_f32 v14, v11, v4 // (op 2150) 
v_mul_f32 v15, v11, v5 // (op 2150) 
v_mul_f32 v16, v11, v6 // (op 2150) 
v_mul_f32 v17, v11, v7 // (op 2150) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 2135) 
v_lshrrev_b32 v7, 16, v8 // (op 2135) 
v_lshlrev_b32 v6, 16, v6 // (op 2135) 
v_lshlrev_b32 v7, 16, v7 // (op 2135) 
v_mul_f32 v18, v10, v6 // (op 2135) 
v_mul_f32 v19, v10, v7 // (op 2135) 
v_and_b32 v6, 65535, v9 // (op 2135) 
v_lshrrev_b32 v7, 16, v9 // (op 2135) 
v_lshlrev_b32 v6, 16, v6 // (op 2135) 
v_lshlrev_b32 v7, 16, v7 // (op 2135) 
v_mul_f32 v20, v10, v6 // (op 2135) 
v_mul_f32 v21, v10, v7 // (op 2135) 


v_add_f32 v6, v18, v14 // (op 2136) 
v_add_f32 v7, v19, v15 // (op 2136) 
v_add_f32 v8, v20, v16 // (op 2136) 
v_add_f32 v9, v21, v17 // (op 2136) 


v_lshrrev_b32 v14, 16, v6 // (op 2137) 
v_lshrrev_b32 v15, 16, v7 // (op 2137) 
v_lshrrev_b32 v16, 16, v8 // (op 2137) 
v_lshrrev_b32 v17, 16, v9 // (op 2137) 


v_pack_b32_f16 v8, v14, v15 // (op 2126) 
v_pack_b32_f16 v9, v16, v17 // (op 2126) 

s_lshl_b32 s0, s4, 6 // (op 2126) 
s_add_u32 s0, s0, 64 // (op 2126) 
v_add_u32 v0, v13, s0 // (op 2126) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 2126) Store value





s_lshl_b32 s0, s2, 7 // (op 2143) 
s_add_u32 s0, s0, 64 // (op 2143) 
v_add_u32 v0, v12, s0 // (op 2143) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 2143) Load value

v_accvgpr_read v4, a24 // (op 2152) 
v_accvgpr_read v5, a25 // (op 2152) 
v_accvgpr_read v6, a26 // (op 2152) 
v_accvgpr_read v7, a27 // (op 2152) 
v_mul_f32 v14, v11, v4 // (op 2152) 
v_mul_f32 v15, v11, v5 // (op 2152) 
v_mul_f32 v16, v11, v6 // (op 2152) 
v_mul_f32 v17, v11, v7 // (op 2152) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 2140) 
v_lshrrev_b32 v7, 16, v8 // (op 2140) 
v_lshlrev_b32 v6, 16, v6 // (op 2140) 
v_lshlrev_b32 v7, 16, v7 // (op 2140) 
v_mul_f32 v18, v10, v6 // (op 2140) 
v_mul_f32 v19, v10, v7 // (op 2140) 
v_and_b32 v6, 65535, v9 // (op 2140) 
v_lshrrev_b32 v7, 16, v9 // (op 2140) 
v_lshlrev_b32 v6, 16, v6 // (op 2140) 
v_lshlrev_b32 v7, 16, v7 // (op 2140) 
v_mul_f32 v20, v10, v6 // (op 2140) 
v_mul_f32 v21, v10, v7 // (op 2140) 


v_add_f32 v6, v18, v14 // (op 2141) 
v_add_f32 v7, v19, v15 // (op 2141) 
v_add_f32 v8, v20, v16 // (op 2141) 
v_add_f32 v9, v21, v17 // (op 2141) 


v_lshrrev_b32 v14, 16, v6 // (op 2142) 
v_lshrrev_b32 v15, 16, v7 // (op 2142) 
v_lshrrev_b32 v16, 16, v8 // (op 2142) 
v_lshrrev_b32 v17, 16, v9 // (op 2142) 


v_pack_b32_f16 v8, v14, v15 // (op 2128) 
v_pack_b32_f16 v9, v16, v17 // (op 2128) 

s_lshl_b32 s0, s4, 7 // (op 2128) 
s_add_u32 s0, s0, 64 // (op 2128) 
v_add_u32 v0, v13, s0 // (op 2128) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 2128) Store value





s_lshl_b32 s0, s2, 6 // (op 2148) 
s_mul_i32 s1, 3, s0 // (op 2148) 
s_add_u32 s1, s1, 64 // (op 2148) 

v_add_u32 v0, v12, s1 // (op 2148) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 2148) Load value

v_accvgpr_read v4, a28 // (op 2154) 
v_accvgpr_read v5, a29 // (op 2154) 
v_accvgpr_read v6, a30 // (op 2154) 
v_accvgpr_read v7, a31 // (op 2154) 
v_mul_f32 v14, v11, v4 // (op 2154) 
v_mul_f32 v15, v11, v5 // (op 2154) 
v_mul_f32 v16, v11, v6 // (op 2154) 
v_mul_f32 v17, v11, v7 // (op 2154) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 2145) 
v_lshrrev_b32 v7, 16, v8 // (op 2145) 
v_lshlrev_b32 v6, 16, v6 // (op 2145) 
v_lshlrev_b32 v7, 16, v7 // (op 2145) 
v_mul_f32 v18, v10, v6 // (op 2145) 
v_mul_f32 v19, v10, v7 // (op 2145) 
v_and_b32 v6, 65535, v9 // (op 2145) 
v_lshrrev_b32 v7, 16, v9 // (op 2145) 
v_lshlrev_b32 v6, 16, v6 // (op 2145) 
v_lshlrev_b32 v7, 16, v7 // (op 2145) 
v_mul_f32 v20, v10, v6 // (op 2145) 
v_mul_f32 v21, v10, v7 // (op 2145) 


v_add_f32 v6, v18, v14 // (op 2146) 
v_add_f32 v7, v19, v15 // (op 2146) 
v_add_f32 v8, v20, v16 // (op 2146) 
v_add_f32 v9, v21, v17 // (op 2146) 


v_lshrrev_b32 v14, 16, v6 // (op 2147) 
v_lshrrev_b32 v15, 16, v7 // (op 2147) 
v_lshrrev_b32 v16, 16, v8 // (op 2147) 
v_lshrrev_b32 v17, 16, v9 // (op 2147) 


v_pack_b32_f16 v8, v14, v15 // (op 2130) 
v_pack_b32_f16 v9, v16, v17 // (op 2130) 

s_lshl_b32 s0, s4, 6 // (op 2130) 
s_mul_i32 s1, 3, s0 // (op 2130) 
s_add_u32 s1, s1, 64 // (op 2130) 

v_add_u32 v0, v13, s1 // (op 2130) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 2130) Store value





v_mov_b32 v0, v12 // (op 148) reg expression

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 148) Load value

v_accvgpr_read v4, a0 // (op 144) 
v_accvgpr_read v5, a1 // (op 144) 
v_accvgpr_read v6, a2 // (op 144) 
v_accvgpr_read v7, a3 // (op 144) 
v_mul_f32 v14, v11, v4 // (op 144) 
v_mul_f32 v15, v11, v5 // (op 144) 
v_mul_f32 v16, v11, v6 // (op 144) 
v_mul_f32 v17, v11, v7 // (op 144) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 149) 
v_lshrrev_b32 v7, 16, v8 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v18, v10, v6 // (op 149) 
v_mul_f32 v19, v10, v7 // (op 149) 
v_and_b32 v6, 65535, v9 // (op 149) 
v_lshrrev_b32 v7, 16, v9 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v20, v10, v6 // (op 149) 
v_mul_f32 v21, v10, v7 // (op 149) 


v_add_f32 v6, v18, v14 // (op 145) 
v_add_f32 v7, v19, v15 // (op 145) 
v_add_f32 v8, v20, v16 // (op 145) 
v_add_f32 v9, v21, v17 // (op 145) 


v_lshrrev_b32 v14, 16, v6 // (op 146) 
v_lshrrev_b32 v15, 16, v7 // (op 146) 
v_lshrrev_b32 v16, 16, v8 // (op 146) 
v_lshrrev_b32 v17, 16, v9 // (op 146) 


v_pack_b32_f16 v8, v14, v15 // (op 147) 
v_pack_b32_f16 v9, v16, v17 // (op 147) 

v_mov_b32 v0, v13 // (op 147) reg expression

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 2041) 
v_add_u32 v0, v12, s0 // (op 2041) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 2041) Load value

v_accvgpr_read v4, a4 // (op 2037) 
v_accvgpr_read v5, a5 // (op 2037) 
v_accvgpr_read v6, a6 // (op 2037) 
v_accvgpr_read v7, a7 // (op 2037) 
v_mul_f32 v14, v11, v4 // (op 2037) 
v_mul_f32 v15, v11, v5 // (op 2037) 
v_mul_f32 v16, v11, v6 // (op 2037) 
v_mul_f32 v17, v11, v7 // (op 2037) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 2042) 
v_lshrrev_b32 v7, 16, v8 // (op 2042) 
v_lshlrev_b32 v6, 16, v6 // (op 2042) 
v_lshlrev_b32 v7, 16, v7 // (op 2042) 
v_mul_f32 v18, v10, v6 // (op 2042) 
v_mul_f32 v19, v10, v7 // (op 2042) 
v_and_b32 v6, 65535, v9 // (op 2042) 
v_lshrrev_b32 v7, 16, v9 // (op 2042) 
v_lshlrev_b32 v6, 16, v6 // (op 2042) 
v_lshlrev_b32 v7, 16, v7 // (op 2042) 
v_mul_f32 v20, v10, v6 // (op 2042) 
v_mul_f32 v21, v10, v7 // (op 2042) 


v_add_f32 v6, v18, v14 // (op 2038) 
v_add_f32 v7, v19, v15 // (op 2038) 
v_add_f32 v8, v20, v16 // (op 2038) 
v_add_f32 v9, v21, v17 // (op 2038) 


v_lshrrev_b32 v14, 16, v6 // (op 2039) 
v_lshrrev_b32 v15, 16, v7 // (op 2039) 
v_lshrrev_b32 v16, 16, v8 // (op 2039) 
v_lshrrev_b32 v17, 16, v9 // (op 2039) 


v_pack_b32_f16 v8, v14, v15 // (op 2040) 
v_pack_b32_f16 v9, v16, v17 // (op 2040) 

s_lshl_b32 s0, s4, 6 // (op 2040) 
v_add_u32 v0, v13, s0 // (op 2040) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 2040) Store value





s_lshl_b32 s0, s2, 7 // (op 2055) 
v_add_u32 v0, v12, s0 // (op 2055) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 2055) Load value

v_accvgpr_read v4, a8 // (op 2051) 
v_accvgpr_read v5, a9 // (op 2051) 
v_accvgpr_read v6, a10 // (op 2051) 
v_accvgpr_read v7, a11 // (op 2051) 
v_mul_f32 v14, v11, v4 // (op 2051) 
v_mul_f32 v15, v11, v5 // (op 2051) 
v_mul_f32 v16, v11, v6 // (op 2051) 
v_mul_f32 v17, v11, v7 // (op 2051) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 2056) 
v_lshrrev_b32 v7, 16, v8 // (op 2056) 
v_lshlrev_b32 v6, 16, v6 // (op 2056) 
v_lshlrev_b32 v7, 16, v7 // (op 2056) 
v_mul_f32 v18, v10, v6 // (op 2056) 
v_mul_f32 v19, v10, v7 // (op 2056) 
v_and_b32 v6, 65535, v9 // (op 2056) 
v_lshrrev_b32 v7, 16, v9 // (op 2056) 
v_lshlrev_b32 v6, 16, v6 // (op 2056) 
v_lshlrev_b32 v7, 16, v7 // (op 2056) 
v_mul_f32 v20, v10, v6 // (op 2056) 
v_mul_f32 v21, v10, v7 // (op 2056) 


v_add_f32 v6, v18, v14 // (op 2052) 
v_add_f32 v7, v19, v15 // (op 2052) 
v_add_f32 v8, v20, v16 // (op 2052) 
v_add_f32 v9, v21, v17 // (op 2052) 


v_lshrrev_b32 v14, 16, v6 // (op 2053) 
v_lshrrev_b32 v15, 16, v7 // (op 2053) 
v_lshrrev_b32 v16, 16, v8 // (op 2053) 
v_lshrrev_b32 v17, 16, v9 // (op 2053) 


v_pack_b32_f16 v8, v14, v15 // (op 2054) 
v_pack_b32_f16 v9, v16, v17 // (op 2054) 

s_lshl_b32 s0, s4, 7 // (op 2054) 
v_add_u32 v0, v13, s0 // (op 2054) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 2054) Store value





s_lshl_b32 s0, s2, 6 // (op 2069) 
s_mul_i32 s1, 3, s0 // (op 2069) 

v_add_u32 v0, v12, s1 // (op 2069) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 2069) Load value

v_accvgpr_read v4, a12 // (op 2065) 
v_accvgpr_read v5, a13 // (op 2065) 
v_accvgpr_read v6, a14 // (op 2065) 
v_accvgpr_read v7, a15 // (op 2065) 
v_mul_f32 v14, v11, v4 // (op 2065) 
v_mul_f32 v15, v11, v5 // (op 2065) 
v_mul_f32 v16, v11, v6 // (op 2065) 
v_mul_f32 v17, v11, v7 // (op 2065) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 2070) 
v_lshrrev_b32 v7, 16, v8 // (op 2070) 
v_lshlrev_b32 v6, 16, v6 // (op 2070) 
v_lshlrev_b32 v7, 16, v7 // (op 2070) 
v_mul_f32 v18, v10, v6 // (op 2070) 
v_mul_f32 v19, v10, v7 // (op 2070) 
v_and_b32 v6, 65535, v9 // (op 2070) 
v_lshrrev_b32 v7, 16, v9 // (op 2070) 
v_lshlrev_b32 v6, 16, v6 // (op 2070) 
v_lshlrev_b32 v7, 16, v7 // (op 2070) 
v_mul_f32 v20, v10, v6 // (op 2070) 
v_mul_f32 v21, v10, v7 // (op 2070) 


v_add_f32 v6, v18, v14 // (op 2066) 
v_add_f32 v7, v19, v15 // (op 2066) 
v_add_f32 v8, v20, v16 // (op 2066) 
v_add_f32 v9, v21, v17 // (op 2066) 


v_lshrrev_b32 v14, 16, v6 // (op 2067) 
v_lshrrev_b32 v15, 16, v7 // (op 2067) 
v_lshrrev_b32 v16, 16, v8 // (op 2067) 
v_lshrrev_b32 v17, 16, v9 // (op 2067) 


v_pack_b32_f16 v8, v14, v15 // (op 2068) 
v_pack_b32_f16 v9, v16, v17 // (op 2068) 

s_lshl_b32 s0, s4, 6 // (op 2068) 
s_mul_i32 s1, 3, s0 // (op 2068) 

v_add_u32 v0, v13, s1 // (op 2068) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 2068) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_135:
















s_endpgm  // End of RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2
.LRR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_end:
.size RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2, .LRR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_end-RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2
  .amdhsa_next_free_vgpr 164
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 50688 // lds bytes
  .amdhsa_accum_offset 132
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: B8
    DestDataType: b
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
    macrotile: [64, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2
    .symbol: RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 50688
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 130
    .agpr_count: 32
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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

