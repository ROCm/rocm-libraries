// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2,@function
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 4602) 
v_and_b32 v3, 15, v1 // (op 4602) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 4602) 


v_and_b32 v0, 48, v1 // (op 4602) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 4602) 


v_lshrrev_b32 v0, 6, v1 // (op 4659) 
v_and_b32 v4, 1, v0 // (op 4659) 

v_and_b32 v0, 15, v1 // (op 4659) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4659) 


v_and_b32 v0, 48, v1 // (op 4659) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 4659) 


v_lshrrev_b32 v0, 7, v1 // (op 4717) 
v_and_b32 v5, 15, v1 // (op 4717) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 4717) 


v_and_b32 v0, 63, v1 // (op 4717) 
v_lshrrev_b32 v5, 4, v0 // (op 4717) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 4717) 


v_lshrrev_b32 v5, 6, v1 // (op 4775) 
v_and_b32 v6, 1, v5 // (op 4775) 

v_and_b32 v5, 15, v1 // (op 4775) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 4775) 


v_and_b32 v5, 63, v1 // (op 4775) 
v_lshrrev_b32 v6, 4, v5 // (op 4775) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 4775) 


v_lshrrev_b32 v6, 7, v1 // (op 4842) 
v_and_b32 v7, 15, v1 // (op 4842) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 4842) 


v_and_b32 v6, 48, v1 // (op 4842) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 4842) 


v_lshrrev_b32 v6, 6, v1 // (op 4886) 
v_and_b32 v8, 1, v6 // (op 4886) 

v_and_b32 v6, 15, v1 // (op 4886) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 4886) 


v_and_b32 v6, 48, v1 // (op 4886) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 4886) 


v_lshrrev_b32 v6, 7, v1 // (op 4915) 
v_and_b32 v9, 15, v1 // (op 4915) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 4915) 


v_and_b32 v6, 63, v1 // (op 4915) 
v_lshrrev_b32 v9, 4, v6 // (op 4915) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 4915) 


v_lshrrev_b32 v9, 6, v1 // (op 4944) 
v_and_b32 v10, 1, v9 // (op 4944) 

v_and_b32 v9, 15, v1 // (op 4944) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 4944) 


v_and_b32 v9, 63, v1 // (op 4944) 
v_lshrrev_b32 v10, 4, v9 // (op 4944) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 4944) 


v_lshrrev_b32 v10, 7, v1 // (op 4541) 
v_and_b32 v11, 63, v1 // (op 4541) 
v_lshrrev_b32 v12, 4, v11 // (op 4541) 
v_lshlrev_b32 v13, 2, v12 // (op 4541) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 4541) 


v_lshl_add_u32 v10, s16, 7, v12 // (op 4541) 

v_lshrrev_b32 v12, 2, v11 // (op 4541) 

v_and_b32 v11, 3, v12 // (op 4541) 

v_and_b32 v12, 3, v1 // (op 4541) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 4541) 


v_lshrrev_b32 v11, 6, v1 // (op 4541) 
v_and_b32 v12, 1, v11 // (op 4541) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 4541) 


v_lshl_add_u32 v12, s17, 6, v11 // (op 4541) 

v_mul_lo_u32 v11, v12, s2 // (op 4541) 

v_add_lshl_u32 v12, v10, v11, 2 // (op 4541) 


v_lshrrev_b32 v10, 7, v1 // (op 4796) 
v_and_b32 v11, 63, v1 // (op 4796) 
v_lshrrev_b32 v13, 4, v11 // (op 4796) 
s_mul_i32 s1, s2, s9 // (op 4543) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 4543) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 4543) least significant
s_mul_hi_u32 s19, s2, s8 // (op 4543) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4543) most significant: sum
s_add_u32 s1, s1, s19 // (op 4543) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4543) 

v_lshlrev_b32 v14, 2, v13 // (op 4796) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 4796) 


v_lshl_add_u32 v10, s16, 7, v13 // (op 4796) 

v_lshrrev_b32 v13, 2, v11 // (op 4796) 

v_and_b32 v11, 3, v13 // (op 4796) 

v_and_b32 v13, 3, v1 // (op 4796) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 4796) 


v_lshrrev_b32 v11, 6, v1 // (op 4796) 
v_and_b32 v13, 1, v11 // (op 4796) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 4796) 


v_lshl_add_u32 v13, s17, 6, v11 // (op 4796) 

v_mul_lo_u32 v11, v13, s4 // (op 4796) 

v_add_lshl_u32 v13, v10, v11, 2 // (op 4796) 


s_mov_b32 s40, s6 // (op 4543) reg expression
s_mov_b32 s41, s7 // (op 4543) reg expression
s_mov_b32 s42, s18 // (op 4543) reg expression
s_mov_b32 s43, 131072 // (op 4543) call()

s_mul_i32 s1, s4, s9 // (op 4798) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 4798) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 4798) least significant
s_mul_hi_u32 s7, s4, s8 // (op 4798) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4798) most significant: sum
s_add_u32 s1, s1, s7 // (op 4798) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 4798) 

s_mov_b32 s44, s10 // (op 4798) reg expression
s_mov_b32 s45, s11 // (op 4798) reg expression
s_mov_b32 s46, s6 // (op 4798) reg expression
s_mov_b32 s47, 131072 // (op 4798) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 336) call()
v_accvgpr_write a5, 0.00000 // (op 336) call()
v_accvgpr_write a6, 0.00000 // (op 336) call()
v_accvgpr_write a7, 0.00000 // (op 336) call()
v_accvgpr_write a8, 0.00000 // (op 556) call()
v_accvgpr_write a9, 0.00000 // (op 556) call()
v_accvgpr_write a10, 0.00000 // (op 556) call()
v_accvgpr_write a11, 0.00000 // (op 556) call()
v_accvgpr_write a12, 0.00000 // (op 711) call()
v_accvgpr_write a13, 0.00000 // (op 711) call()
v_accvgpr_write a14, 0.00000 // (op 711) call()
v_accvgpr_write a15, 0.00000 // (op 711) call()
v_accvgpr_write a16, 0.00000 // (op 942) call()
v_accvgpr_write a17, 0.00000 // (op 942) call()
v_accvgpr_write a18, 0.00000 // (op 942) call()
v_accvgpr_write a19, 0.00000 // (op 942) call()
v_accvgpr_write a20, 0.00000 // (op 1097) call()
v_accvgpr_write a21, 0.00000 // (op 1097) call()
v_accvgpr_write a22, 0.00000 // (op 1097) call()
v_accvgpr_write a23, 0.00000 // (op 1097) call()
v_accvgpr_write a24, 0.00000 // (op 1328) call()
v_accvgpr_write a25, 0.00000 // (op 1328) call()
v_accvgpr_write a26, 0.00000 // (op 1328) call()
v_accvgpr_write a27, 0.00000 // (op 1328) call()
v_accvgpr_write a28, 0.00000 // (op 1483) call()
v_accvgpr_write a29, 0.00000 // (op 1483) call()
v_accvgpr_write a30, 0.00000 // (op 1483) call()
v_accvgpr_write a31, 0.00000 // (op 1483) call()
v_lshrrev_b32 v14, 3, v1 // (op 4303) 
v_lshl_add_u32 v15, s16, 7, v14 // (op 4303) 

v_and_b32 v14, -8, v1 // (op 4303) 
v_sub_u32 v16, v1, v14 // (op 4303) 

v_lshlrev_b32 v14, 4, v16 // (op 4303) 

v_mov_b32 v17, 0 // (op 4303) convert
v_mov_b32 v16, v14 // (op 4303) convert

v_mul_lo_u32 v19, v15, s13 // (op 4303) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 4303) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 4303) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 4303) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 4303) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 4303) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 4305) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 4305) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 4305) least significant
s_mul_hi_u32 s7, s12, s22 // (op 4305) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4305) most significant: sum
s_add_u32 s1, s1, s7 // (op 4305) most significant: sum


s_mov_b32 s48, s20 // (op 4305) reg expression
s_mov_b32 s49, s21 // (op 4305) reg expression
s_mov_b32 s50, s0 // (op 4305) reg expression
s_mov_b32 s51, 131072 // (op 4305) call()

v_lshrrev_b32 v14, 3, v1 // (op 4322) 
v_lshl_add_u32 v15, s16, 7, v14 // (op 4322) 

v_and_b32 v14, -8, v1 // (op 4322) 
v_sub_u32 v16, v1, v14 // (op 4322) 

v_lshlrev_b32 v14, 4, v16 // (op 4322) 

v_mov_b32 v17, 0 // (op 4322) convert
v_mov_b32 v16, v14 // (op 4322) convert

v_mul_lo_u32 v21, v15, s13 // (op 4322) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 4322) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 4322) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 4322) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 4322) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 4322) most significant half



s_mul_i32 s1, s12, s23 // (op 4324) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 4324) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 4324) least significant
s_mul_hi_u32 s7, s12, s22 // (op 4324) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4324) most significant: sum
s_add_u32 s1, s1, s7 // (op 4324) most significant: sum


s_mov_b32 s48, s20 // (op 4324) reg expression
s_mov_b32 s49, s21 // (op 4324) reg expression
s_mov_b32 s50, s0 // (op 4324) reg expression
s_mov_b32 s51, 131072 // (op 4324) call()

v_and_b32 v14, -8, v1 // (op 4367) 
v_sub_u32 v15, v1, v14 // (op 4367) 

v_lshlrev_b32 v14, 4, v15 // (op 4367) 

v_mov_b32 v17, 0 // (op 4367) convert
v_mov_b32 v16, v14 // (op 4367) convert

v_lshrrev_b32 v14, 3, v1 // (op 4367) 
v_lshl_add_u32 v15, s17, 6, v14 // (op 4367) 

v_mul_lo_u32 v23, v15, s25 // (op 4367) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 4367) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 4367) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 4367) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v22 // (op 4367) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 4367) most significant half



s_mul_i32 s1, s24, s9 // (op 4369) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 4369) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 4369) least significant
s_mul_hi_u32 s7, s24, s8 // (op 4369) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4369) most significant: sum
s_add_u32 s1, s1, s7 // (op 4369) most significant: sum


s_mov_b32 s52, s26 // (op 4369) reg expression
s_mov_b32 s53, s27 // (op 4369) reg expression
s_mov_b32 s54, s0 // (op 4369) reg expression
s_mov_b32 s55, 131072 // (op 4369) call()

v_and_b32 v16, -8, v1 // (op 4386) 
v_sub_u32 v17, v1, v16 // (op 4386) 

v_lshlrev_b32 v16, 4, v17 // (op 4386) 

v_mov_b32 v23, 0 // (op 4386) convert
v_mov_b32 v22, v16 // (op 4386) convert

v_lshrrev_b32 v16, 3, v1 // (op 4386) 
v_lshl_add_u32 v17, s17, 6, v16 // (op 4386) 

v_mul_lo_u32 v25, v17, s25 // (op 4386) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 4386) most significant: high of low * low
v_mul_lo_u32 v24, v17, s24 // (op 4386) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 4386) most significant: sum


v_add_co_u32 v16, s[0:1], v22, v24 // (op 4386) least significant half
v_addc_co_u32 v17, s[0:1], v23, v25, s[0:1] // (op 4386) most significant half



s_mul_i32 s1, s24, s9 // (op 4388) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 4388) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 4388) least significant
s_mul_hi_u32 s7, s24, s8 // (op 4388) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4388) most significant: sum
s_add_u32 s1, s1, s7 // (op 4388) most significant: sum


s_mov_b32 s52, s26 // (op 4388) reg expression
s_mov_b32 s53, s27 // (op 4388) reg expression
s_mov_b32 s54, s0 // (op 4388) reg expression
s_mov_b32 s55, 131072 // (op 4388) call()

v_lshrrev_b32 v22, 2, v1 // (op 4431) 
v_lshlrev_b32 v23, 1, v22 // (op 4431) 

v_lshl_add_u32 v22, s16, 7, v23 // (op 4431) 

v_and_b32 v23, -4, v1 // (op 4431) 
v_sub_u32 v24, v1, v23 // (op 4431) 

v_mul_lo_u32 v23, v22, s28 // (op 4431) 
v_add_u32 v23, v23, v24 // (op 4431) 


s_mul_i32 s1, s28, s23 // (op 4433) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4433) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4433) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4433) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4433) most significant: sum
s_add_u32 s1, s1, s7 // (op 4433) most significant: sum


s_mov_b32 s56, s30 // (op 4433) reg expression
s_mov_b32 s57, s31 // (op 4433) reg expression
s_mov_b32 s58, s0 // (op 4433) reg expression
s_mov_b32 s59, 131072 // (op 4433) call()

v_lshrrev_b32 v22, 2, v1 // (op 4450) 
v_lshlrev_b32 v24, 1, v22 // (op 4450) 

v_lshl_add_u32 v22, s16, 7, v24 // (op 4450) 

v_and_b32 v24, -4, v1 // (op 4450) 
v_sub_u32 v25, v1, v24 // (op 4450) 

v_mul_lo_u32 v24, v22, s28 // (op 4450) 
v_add_u32 v24, v24, v25 // (op 4450) 


s_mul_i32 s1, s28, s23 // (op 4452) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4452) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4452) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4452) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4452) most significant: sum
s_add_u32 s1, s1, s7 // (op 4452) most significant: sum


s_mov_b32 s56, s30 // (op 4452) reg expression
s_mov_b32 s57, s31 // (op 4452) reg expression
s_mov_b32 s58, s0 // (op 4452) reg expression
s_mov_b32 s59, 131072 // (op 4452) call()

v_and_b32 v22, -4, v1 // (op 4495) 
v_sub_u32 v25, v1, v22 // (op 4495) 

v_lshrrev_b32 v22, 2, v1 // (op 4495) 
v_lshl_add_u32 v26, s17, 6, v22 // (op 4495) 

v_mul_lo_u32 v22, v26, s32 // (op 4495) 

v_add_u32 v26, v25, v22 // (op 4495) 


s_mul_i32 s1, s32, s9 // (op 4497) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4497) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4497) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4497) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4497) most significant: sum
s_add_u32 s1, s1, s7 // (op 4497) most significant: sum


s_mov_b32 s60, s34 // (op 4497) reg expression
s_mov_b32 s61, s35 // (op 4497) reg expression
s_mov_b32 s62, s0 // (op 4497) reg expression
s_mov_b32 s63, 131072 // (op 4497) call()

v_and_b32 v22, -4, v1 // (op 4514) 
v_sub_u32 v25, v1, v22 // (op 4514) 

v_lshrrev_b32 v22, 2, v1 // (op 4514) 
v_lshl_add_u32 v27, s17, 6, v22 // (op 4514) 

v_mul_lo_u32 v22, v27, s32 // (op 4514) 

v_add_u32 v27, v25, v22 // (op 4514) 


s_mul_i32 s1, s32, s9 // (op 4516) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4516) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4516) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4516) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4516) most significant: sum
s_add_u32 s1, s1, s7 // (op 4516) most significant: sum


s_mov_b32 s60, s34 // (op 4516) reg expression
s_mov_b32 s61, s35 // (op 4516) reg expression
s_mov_b32 s62, s0 // (op 4516) reg expression
s_mov_b32 s63, 131072 // (op 4516) call()

v_and_b32 v22, -8, v1 // (op 4642) 
v_sub_u32 v25, v1, v22 // (op 4642) 

v_lshlrev_b32 v22, 4, v25 // (op 4642) 

v_lshrrev_b32 v25, 3, v1 // (op 4642) 
v_lshl_add_u32 v28, v25, 7, v22 // (op 4642) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 4642) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 4245) 
v_mov_b64 v[28:29], s[6:7] // (op 4245) convert


v_mov_b64 v[30:31], v[14:15] // (op 4245) 
s_mov_b32 m0, s0 // (op 4245) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 4245) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 4245) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 4245) most significant half

s_add_u32 m0, m0, 4096 // (op 4245) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 4245) Load value direct to lds






buffer_load_ubyte v22, v26, s[60:63], 0 offen offset: 0 // (op 4027) Load value
v_and_b32 v25, -8, v1 // (op 4585) 
v_sub_u32 v28, v1, v25 // (op 4585) 

v_lshlrev_b32 v25, 4, v28 // (op 4585) 

v_lshrrev_b32 v28, 3, v1 // (op 4585) 
v_lshl_add_u32 v29, v28, 7, v25 // (op 4585) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 4585) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 4250) 
v_mov_b64 v[28:29], s[6:7] // (op 4250) convert


v_mov_b64 v[30:31], v[18:19] // (op 4250) 
s_add_u32 s1, 8192, s0 // (op 4250) 
s_mov_b32 m0, s1 // (op 4250) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 4250) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 4250) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 4250) most significant half

s_add_u32 m0, m0, 4096 // (op 4250) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 4250) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 4250) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 4250) most significant half

s_add_u32 m0, m0, 4096 // (op 4250) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 4250) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 4250) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 4250) most significant half

s_add_u32 m0, m0, 4096 // (op 4250) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 4250) Load value direct to lds





v_mov_b64 v[28:29], s[28:29] // (op 4033) convert

v_mov_b32 v25, v23 // (op 4033) 
buffer_load_ubyte v30, v25, s[56:59], 0 offen offset: 0 // (op 4033) Load value
v_add_u32 v25, v25, v28 // (op 4033) 
buffer_load_ubyte v31, v25, s[56:59], 0 offen offset: 0 // (op 4033) Load value


v_and_b32 v25, -8, v1 // (op 4869) 
v_sub_u32 v28, v1, v25 // (op 4869) 

v_lshlrev_b32 v25, 4, v28 // (op 4869) 

v_lshrrev_b32 v28, 3, v1 // (op 4869) 
v_lshl_add_u32 v29, v28, 7, v25 // (op 4869) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 4869) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 4255) 
v_mov_b64 v[28:29], s[6:7] // (op 4255) convert


v_mov_b64 v[32:33], v[14:15] // (op 4255) 
s_add_u32 s1, 24576, s0 // (op 4255) 
s_mov_b32 m0, s1 // (op 4255) reg expression
s_mov_b32 s6, 128 // (op 4255) call()
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 4255) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v28 // (op 4255) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 4255) most significant half

s_add_u32 m0, m0, 4096 // (op 4255) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 4255) Load value direct to lds









buffer_load_ubyte v25, v26, s[60:63], 0 offen offset:4 // (op 4039) Load value

v_and_b32 v26, -8, v1 // (op 4825) 
v_sub_u32 v14, v1, v26 // (op 4825) 

v_lshlrev_b32 v15, 4, v14 // (op 4825) 

v_lshrrev_b32 v14, 3, v1 // (op 4825) 
v_lshl_add_u32 v26, v14, 7, v15 // (op 4825) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 4825) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 4257) 
v_mov_b64 v[14:15], s[6:7] // (op 4257) convert


v_mov_b64 v[28:29], v[18:19] // (op 4257) 
s_add_u32 s1, 32768, s0 // (op 4257) 
s_mov_b32 m0, s1 // (op 4257) reg expression
s_mov_b32 s6, 128 // (op 4257) call()
buffer_load_dwordx4 v28, s[48:51], s6 offen offset: 0 lds // (op 4257) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v14 // (op 4257) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 4257) most significant half

s_add_u32 m0, m0, 4096 // (op 4257) 
buffer_load_dwordx4 v28, s[48:51], s6 offen offset: 0 lds // (op 4257) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v14 // (op 4257) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 4257) most significant half

s_add_u32 m0, m0, 4096 // (op 4257) 
buffer_load_dwordx4 v28, s[48:51], s6 offen offset: 0 lds // (op 4257) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v14 // (op 4257) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 4257) most significant half

s_add_u32 m0, m0, 4096 // (op 4257) 
buffer_load_dwordx4 v28, s[48:51], s6 offen offset: 0 lds // (op 4257) Load value direct to lds







v_mov_b64 v[14:15], s[28:29] // (op 4045) convert

v_mov_b32 v26, v23 // (op 4045) 
buffer_load_ubyte v18, v26, s[56:59], 0 offen offset:4 // (op 4045) Load value
v_add_u32 v26, v26, v14 // (op 4045) 
buffer_load_ubyte v19, v26, s[56:59], 0 offen offset:4 // (op 4045) Load value



v_and_b32 v23, -4, v1 // (op 4758) 
v_sub_u32 v26, v1, v23 // (op 4758) 

v_and_b32 v23, 4294967292, v1 // (op 4758) 
v_add_u32 v14, v23, v26 // (op 4758) 





s_waitcnt vmcnt(15)
ds_write_b8 v14, v22 offset:49152 // (op 4048) Store local data 

v_and_b32 v23, -4, v1 // (op 4700) 
v_sub_u32 v26, v1, v23 // (op 4700) 

v_lshrrev_b32 v23, 2, v1 // (op 4700) 
v_lshl_add_u32 v14, v23, 3, v26 // (op 4700) 





s_waitcnt vmcnt(10)
ds_write_b8 v14, v30 offset:49408 // (op 4050) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v14, v31 offset:49412 // (op 4050) Store local data 

s_barrier  // (op 5045) 



ds_read_b128 v[32:35], v4 // (op 4085) Load local data 
ds_read_b128 v[36:39], v4 offset:64 // (op 4085) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5080) 



ds_read_u8 v23, v5 offset:49152 // (op 4115) Load local data 



ds_read_b128 v[40:43], v4 offset:4096 // (op 4090) Load local data 
ds_read_b128 v[44:47], v4 offset:4160 // (op 4090) Load local data 



ds_read_u8 v26, v5 offset:49280 // (op 4120) Load local data 
s_barrier  // (op 5029) 



ds_read_b128 v[48:51], v3 offset:8192 // (op 4065) Load local data 
ds_read_b128 v[52:55], v3 offset:8256 // (op 4065) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 5062) 



ds_read_u8 v14, v0 offset:49408 // (op 4095) Load local data 



ds_read_b128 v[56:59], v3 offset:12288 // (op 4070) Load local data 
ds_read_b128 v[60:63], v3 offset:12352 // (op 4070) Load local data 



ds_read_u8 v15, v0 offset:49536 // (op 4100) Load local data 



ds_read_b128 v[64:67], v3 offset:16384 // (op 4075) Load local data 
ds_read_b128 v[68:71], v3 offset:16448 // (op 4075) Load local data 



ds_read_u8 v28, v0 offset:49664 // (op 4105) Load local data 



ds_read_b128 v[72:75], v3 offset:20480 // (op 4080) Load local data 
ds_read_b128 v[76:79], v3 offset:20544 // (op 4080) Load local data 



ds_read_u8 v29, v0 offset:49792 // (op 4110) Load local data 
v_and_b32 v80, -8, v1 // (op 4557) 
v_sub_u32 v81, v1, v80 // (op 4557) 

v_lshlrev_b32 v80, 4, v81 // (op 4557) 

v_lshrrev_b32 v81, 3, v1 // (op 4557) 
v_lshl_add_u32 v82, v81, 7, v80 // (op 4557) 


s_nop 0
v_readfirstlane_b32 s0, v82 // (op 4557) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -8, v1 // (op 4614) 
v_sub_u32 v81, v1, v80 // (op 4614) 

v_lshlrev_b32 v80, 4, v81 // (op 4614) 

v_lshrrev_b32 v81, 3, v1 // (op 4614) 
v_lshl_add_u32 v82, v81, 7, v80 // (op 4614) 


s_nop 0
v_readfirstlane_b32 s1, v82 // (op 4614) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -4, v1 // (op 4671) 
v_sub_u32 v81, v1, v80 // (op 4671) 

v_lshrrev_b32 v80, 2, v1 // (op 4671) 
v_lshl_add_u32 v82, v80, 3, v81 // (op 4671) 


v_and_b32 v80, -4, v1 // (op 4729) 
v_sub_u32 v81, v1, v80 // (op 4729) 

v_and_b32 v80, 4294967292, v1 // (op 4729) 
v_add_u32 v83, v80, v81 // (op 4729) 


v_and_b32 v80, -8, v1 // (op 4810) 
v_sub_u32 v81, v1, v80 // (op 4810) 

v_lshlrev_b32 v80, 4, v81 // (op 4810) 

v_lshrrev_b32 v81, 3, v1 // (op 4810) 
v_lshl_add_u32 v84, v81, 7, v80 // (op 4810) 


s_nop 0
v_readfirstlane_b32 s6, v84 // (op 4810) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -8, v1 // (op 4854) 
v_sub_u32 v81, v1, v80 // (op 4854) 

v_lshlrev_b32 v80, 4, v81 // (op 4854) 

v_lshrrev_b32 v81, 3, v1 // (op 4854) 
v_lshl_add_u32 v84, v81, 7, v80 // (op 4854) 


s_nop 0
v_readfirstlane_b32 s7, v84 // (op 4854) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -4, v1 // (op 4898) 
v_sub_u32 v81, v1, v80 // (op 4898) 

v_lshrrev_b32 v80, 2, v1 // (op 4898) 
v_lshl_add_u32 v84, v80, 3, v81 // (op 4898) 


v_and_b32 v80, -4, v1 // (op 4927) 
v_sub_u32 v81, v1, v80 // (op 4927) 

v_and_b32 v80, 4294967292, v1 // (op 4927) 
v_add_u32 v85, v80, v81 // (op 4927) 


s_mov_b32 s10, 0 // (op 1361) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 1360) 
s_mov_b32 s11, s14 // (op 1360) convert

s_and_b32 s14, s11, 4294967294 // (op 1360) 

s_cmp_lt_u32 s10, s14 // (op 1360) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_1360 // (op 1360) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_1360 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopTop_KLoop_1360:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v14, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 88) 
s_barrier  // (op 5048) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[40:47], a[4:7], v14, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 376) 
s_add_u32 s10, s10, 2 // (op 1362) 
s_lshl_b64 s[14:15], s[24:25], 5 // (op 4217) 
v_mov_b64 v[80:81], s[14:15] // (op 4217) convert


v_mov_b64 v[86:87], v[16:17] // (op 4217) 
v_mov_b64 v[88:89], v[86:87] // (op 4217) 
s_mov_b32 m0, s1 // (op 4217) reg expression
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:63], v[32:39], a[8:11], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 673) 
s_mov_b32 s11, 256 // (op 4217) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[40:47], a[12:15], v15, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 627) 
buffer_load_dwordx4 v88, s[52:55], s11 offen offset: 0 lds // (op 4217) Load value direct to lds
v_add_co_u32 v88, s[14:15], v88, v80 // (op 4217) least significant half
v_addc_co_u32 v89, s[14:15], 0, v89, s[14:15] // (op 4217) most significant half

s_add_u32 m0, m0, 4096 // (op 4217) 
buffer_load_dwordx4 v88, s[52:55], s11 offen offset: 0 lds // (op 4217) Load value direct to lds



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[64:71], v[32:39], a[16:19], v28, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1059) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[64:71], v[40:47], a[20:23], v28, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1013) 
v_mov_b32 v80, v27 // (op 25) 

buffer_load_ubyte v22, v80, s[60:63], 0 offen offset:8 // (op 25) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:79], v[32:39], a[24:27], v29, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1445) 
s_barrier  // (op 5032) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:79], v[40:47], a[28:31], v29, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1399) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 4212) 
v_mov_b64 v[88:89], s[14:15] // (op 4212) convert


v_mov_b64 v[90:91], v[20:21] // (op 4212) 
v_mov_b64 v[92:93], v[90:91] // (op 4212) 
s_add_u32 s11, 8192, s0 // (op 4212) 
s_mov_b32 m0, s11 // (op 4212) reg expression
s_mov_b32 s14, 256 // (op 4212) call()
buffer_load_dwordx4 v92, s[48:51], s14 offen offset: 0 lds // (op 4212) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v88 // (op 4212) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 4212) most significant half

s_add_u32 m0, m0, 4096 // (op 4212) 
buffer_load_dwordx4 v92, s[48:51], s14 offen offset: 0 lds // (op 4212) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v88 // (op 4212) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 4212) most significant half

s_add_u32 m0, m0, 4096 // (op 4212) 
buffer_load_dwordx4 v92, s[48:51], s14 offen offset: 0 lds // (op 4212) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v88 // (op 4212) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 4212) most significant half

s_add_u32 m0, m0, 4096 // (op 4212) 
buffer_load_dwordx4 v92, s[48:51], s14 offen offset: 0 lds // (op 4212) Load value direct to lds





v_mov_b64 v[88:89], s[28:29] // (op 17) convert

v_mov_b32 v81, v24 // (op 17) 
v_mov_b32 v92, v81 // (op 17) 
buffer_load_ubyte v30, v92, s[56:59], 0 offen offset:8 // (op 17) Load value
v_add_u32 v92, v92, v88 // (op 17) 
buffer_load_ubyte v31, v92, s[56:59], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 5120) 



s_waitcnt vmcnt(15)
ds_write_b8 v85, v25 offset:49920 // (op 255) Store local data 
s_barrier  // (op 5114) 



s_waitcnt vmcnt(10)
ds_write_b8 v84, v18 offset:50176 // (op 254) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v84, v19 offset:50180 // (op 254) Store local data 
s_barrier  // (op 5103) 



ds_read_b128 v[92:95], v8 offset:24576 // (op 247) Load local data 
ds_read_b128 v[96:99], v8 offset:24640 // (op 247) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5117) 



ds_read_u8 v88, v9 offset:49920 // (op 251) Load local data 



ds_read_b128 v[100:103], v8 offset:28672 // (op 381) Load local data 
ds_read_b128 v[104:107], v8 offset:28736 // (op 381) Load local data 



ds_read_u8 v89, v9 offset:50048 // (op 385) Load local data 
s_barrier  // (op 5097) 
s_barrier  // (op 5161) 



ds_read_b128 v[48:51], v7 offset:32768 // (op 245) Load local data 
ds_read_b128 v[52:55], v7 offset:32832 // (op 245) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 5110) 



ds_read_u8 v14, v6 offset:50176 // (op 249) Load local data 



ds_read_b128 v[56:59], v7 offset:36864 // (op 630) Load local data 
ds_read_b128 v[60:63], v7 offset:36928 // (op 630) Load local data 



ds_read_u8 v15, v6 offset:50304 // (op 633) Load local data 



ds_read_b128 v[64:67], v7 offset:40960 // (op 1016) Load local data 
ds_read_b128 v[68:71], v7 offset:41024 // (op 1016) Load local data 



ds_read_u8 v28, v6 offset:50432 // (op 1019) Load local data 



ds_read_b128 v[72:75], v7 offset:45056 // (op 1402) Load local data 
ds_read_b128 v[76:79], v7 offset:45120 // (op 1402) Load local data 



ds_read_u8 v29, v6 offset:50560 // (op 1405) Load local data 
s_barrier  // (op 5191) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[92:99], a[0:3], v14, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 252) 
s_barrier  // (op 5107) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[100:107], a[4:7], v14, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 377) 
s_lshl_b64 s[14:15], s[24:25], 5 // (op 4240) 
v_mov_b64 v[108:109], s[14:15] // (op 4240) convert


v_mov_b64 v[110:111], v[86:87] // (op 4240) 
s_add_u32 s11, 24576, s7 // (op 4240) 
s_mov_b32 m0, s11 // (op 4240) reg expression
s_mov_b32 s14, 384 // (op 4240) call()
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:63], v[92:99], a[8:11], v15, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 674) 
buffer_load_dwordx4 v110, s[52:55], s14 offen offset: 0 lds // (op 4240) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[100:107], a[12:15], v15, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 628) 
v_add_co_u32 v110, s[18:19], v110, v108 // (op 4240) least significant half
v_addc_co_u32 v111, s[18:19], 0, v111, s[18:19] // (op 4240) most significant half

s_add_u32 m0, m0, 4096 // (op 4240) 
buffer_load_dwordx4 v110, s[52:55], s14 offen offset: 0 lds // (op 4240) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[64:71], v[92:99], a[16:19], v28, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1060) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[64:71], v[100:107], a[20:23], v28, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1014) 
buffer_load_ubyte v25, v80, s[60:63], 0 offen offset:12 // (op 258) Load value

v_add_co_u32 v16, vcc, 256, v16 // (op 4378) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 4378) most significant half
v_add_u32 v27, 8, v27 // (op 4506) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:79], v[92:99], a[24:27], v29, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1446) 
s_barrier  // (op 5100) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:79], v[100:107], a[28:31], v29, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1400) 
s_lshl_b64 s[14:15], s[12:13], 5 // (op 4235) 


v_mov_b64 v[86:87], s[14:15] // (op 4235) convert


v_mov_b64 v[88:89], v[90:91] // (op 4235) 
s_add_u32 s11, 32768, s6 // (op 4235) 
s_mov_b32 m0, s11 // (op 4235) reg expression
s_mov_b32 s14, 384 // (op 4235) call()
buffer_load_dwordx4 v88, s[48:51], s14 offen offset: 0 lds // (op 4235) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v86 // (op 4235) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 4235) most significant half

s_add_u32 m0, m0, 4096 // (op 4235) 
buffer_load_dwordx4 v88, s[48:51], s14 offen offset: 0 lds // (op 4235) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v86 // (op 4235) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 4235) most significant half

s_add_u32 m0, m0, 4096 // (op 4235) 
buffer_load_dwordx4 v88, s[48:51], s14 offen offset: 0 lds // (op 4235) Load value direct to lds
v_add_co_u32 v88, s[18:19], v88, v86 // (op 4235) least significant half
v_addc_co_u32 v89, s[18:19], 0, v89, s[18:19] // (op 4235) most significant half

s_add_u32 m0, m0, 4096 // (op 4235) 
buffer_load_dwordx4 v88, s[48:51], s14 offen offset: 0 lds // (op 4235) Load value direct to lds






v_add_co_u32 v20, vcc, 256, v20 // (op 4314) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 4314) most significant half
v_mov_b64 v[86:87], s[28:29] // (op 257) convert

v_mov_b32 v80, v81 // (op 257) 
buffer_load_ubyte v18, v80, s[56:59], 0 offen offset:12 // (op 257) Load value
v_add_u32 v80, v80, v86 // (op 257) 
buffer_load_ubyte v19, v80, s[56:59], 0 offen offset:12 // (op 257) Load value



s_barrier  // (op 5083) 
v_add_u32 v24, 8, v24 // (op 4442) 



s_waitcnt vmcnt(15)
ds_write_b8 v83, v22 offset:49152 // (op 30) Store local data 
s_barrier  // (op 5065) 



s_waitcnt vmcnt(10)
ds_write_b8 v82, v30 offset:49408 // (op 21) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v82, v31 offset:49412 // (op 21) Store local data 
s_barrier  // (op 5054) 



ds_read_b128 v[32:35], v4 // (op 75) Load local data 
ds_read_b128 v[36:39], v4 offset:64 // (op 75) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5090) 



ds_read_u8 v23, v5 offset:49152 // (op 85) Load local data 



ds_read_b128 v[40:43], v4 offset:4096 // (op 371) Load local data 
ds_read_b128 v[44:47], v4 offset:4160 // (op 371) Load local data 



ds_read_u8 v26, v5 offset:49280 // (op 375) Load local data 
s_barrier  // (op 5038) 



ds_read_b128 v[48:51], v3 offset:8192 // (op 70) Load local data 
ds_read_b128 v[52:55], v3 offset:8256 // (op 70) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 5072) 



ds_read_u8 v14, v0 offset:49408 // (op 80) Load local data 



ds_read_b128 v[56:59], v3 offset:12288 // (op 622) Load local data 
ds_read_b128 v[60:63], v3 offset:12352 // (op 622) Load local data 



ds_read_u8 v15, v0 offset:49536 // (op 625) Load local data 



ds_read_b128 v[64:67], v3 offset:16384 // (op 1008) Load local data 
ds_read_b128 v[68:71], v3 offset:16448 // (op 1008) Load local data 



ds_read_u8 v28, v0 offset:49664 // (op 1011) Load local data 



ds_read_b128 v[72:75], v3 offset:20480 // (op 1394) Load local data 
ds_read_b128 v[76:79], v3 offset:20544 // (op 1394) Load local data 



ds_read_u8 v29, v0 offset:49792 // (op 1397) Load local data 
s_ashr_i64 s[14:15], s[36:37], 7 // (op 1360) 
s_mov_b32 s11, s14 // (op 1360) convert

s_and_b32 s14, s11, 4294967294 // (op 1360) 

s_cmp_lt_u32 s10, s14 // (op 1360) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopTop_KLoop_1360 // (op 1360) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopTop_KLoop_1360 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_1360:

















s_barrier  // (op 5175) 
v_lshrrev_b32 v27, 3, v1 // (op 4270) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4270) 
v_lshl_add_u32 v16, s16, 7, v27 // (op 4270) 

s_lshr_b32 s6, s0, 1 // (op 4270) 

v_and_b32 v17, -8, v1 // (op 4270) 
v_sub_u32 v27, v1, v17 // (op 4270) 

v_lshlrev_b32 v17, 4, v27 // (op 4270) 

v_lshl_add_u32 v27, s6, 8, v17 // (op 4270) 


v_mov_b32 v21, 0 // (op 4270) convert
v_mov_b32 v20, v27 // (op 4270) convert

v_mul_lo_u32 v23, v16, s13 // (op 4270) most significant: low of low * high
v_mul_hi_u32 v17, v16, s12 // (op 4270) most significant: high of low * low
v_mul_lo_u32 v22, v16, s12 // (op 4270) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 4270) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v20 // (op 4270) least significant half
v_addc_co_u32 v23, s[0:1], v23, v21, s[0:1] // (op 4270) most significant half



s_mul_i32 s1, s12, s23 // (op 4272) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 4272) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 4272) least significant
s_mul_hi_u32 s7, s12, s22 // (op 4272) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4272) most significant: sum
s_add_u32 s1, s1, s7 // (op 4272) most significant: sum


s_mov_b32 s48, s20 // (op 4272) reg expression
s_mov_b32 s49, s21 // (op 4272) reg expression
s_mov_b32 s50, s0 // (op 4272) reg expression
s_mov_b32 s51, 131072 // (op 4272) call()

v_and_b32 v24, -8, v1 // (op 4344) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4344) 
v_sub_u32 v27, v1, v24 // (op 4344) 

s_lshr_b32 s6, s0, 1 // (op 4344) 

v_lshlrev_b32 v24, 4, v27 // (op 4344) 

v_lshl_add_u32 v27, s6, 8, v24 // (op 4344) 


v_mov_b32 v17, 0 // (op 4344) convert
v_mov_b32 v16, v27 // (op 4344) convert

v_lshrrev_b32 v24, 3, v1 // (op 4344) 
v_lshl_add_u32 v27, s17, 6, v24 // (op 4344) 

v_mul_lo_u32 v21, v27, s25 // (op 4344) most significant: low of low * high
v_mul_hi_u32 v24, v27, s24 // (op 4344) most significant: high of low * low
v_mul_lo_u32 v20, v27, s24 // (op 4344) least significant: low of low * low
v_add_u32 v21, v21, v24 // (op 4344) most significant: sum


v_add_co_u32 v30, s[0:1], v16, v20 // (op 4344) least significant half
v_addc_co_u32 v31, s[0:1], v17, v21, s[0:1] // (op 4344) most significant half



s_mul_i32 s1, s24, s9 // (op 4346) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 4346) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 4346) least significant
s_mul_hi_u32 s7, s24, s8 // (op 4346) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4346) most significant: sum
s_add_u32 s1, s1, s7 // (op 4346) most significant: sum


s_mov_b32 s52, s26 // (op 4346) reg expression
s_mov_b32 s53, s27 // (op 4346) reg expression
s_mov_b32 s54, s0 // (op 4346) reg expression
s_mov_b32 s55, 131072 // (op 4346) call()

v_lshrrev_b32 v24, 2, v1 // (op 4408) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4408) 
v_lshlrev_b32 v27, 1, v24 // (op 4408) 

s_lshr_b32 s6, s0, 1 // (op 4408) 

v_lshl_add_u32 v24, s16, 7, v27 // (op 4408) 

v_and_b32 v27, -4, v1 // (op 4408) 
v_sub_u32 v16, v1, v27 // (op 4408) 

v_lshl_add_u32 v17, s6, 3, v16 // (op 4408) 


v_mul_lo_u32 v16, v24, s28 // (op 4408) 
v_add_u32 v16, v16, v17 // (op 4408) 


s_mul_i32 s1, s28, s23 // (op 4410) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4410) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4410) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4410) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4410) most significant: sum
s_add_u32 s1, s1, s7 // (op 4410) most significant: sum


s_mov_b32 s56, s30 // (op 4410) reg expression
s_mov_b32 s57, s31 // (op 4410) reg expression
s_mov_b32 s58, s0 // (op 4410) reg expression
s_mov_b32 s59, 131072 // (op 4410) call()

v_and_b32 v17, -4, v1 // (op 4472) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4472) 
v_sub_u32 v24, v1, v17 // (op 4472) 

s_lshr_b32 s6, s0, 1 // (op 4472) 

v_lshrrev_b32 v17, 2, v1 // (op 4472) 
v_lshl_add_u32 v27, s6, 3, v24 // (op 4472) 


v_lshl_add_u32 v24, s17, 6, v17 // (op 4472) 

v_mul_lo_u32 v17, v24, s32 // (op 4472) 

v_add_u32 v24, v27, v17 // (op 4472) 


s_mul_i32 s1, s32, s9 // (op 4474) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4474) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4474) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4474) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4474) most significant: sum
s_add_u32 s1, s1, s7 // (op 4474) most significant: sum


s_mov_b32 s60, s34 // (op 4474) reg expression
s_mov_b32 s61, s35 // (op 4474) reg expression
s_mov_b32 s62, s0 // (op 4474) reg expression
s_mov_b32 s63, 131072 // (op 4474) call()

v_and_b32 v17, -8, v1 // (op 4570) 
v_sub_u32 v27, v1, v17 // (op 4570) 

v_lshlrev_b32 v17, 4, v27 // (op 4570) 

v_lshrrev_b32 v27, 3, v1 // (op 4570) 
v_lshl_add_u32 v20, v27, 7, v17 // (op 4570) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 4570) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -8, v1 // (op 4627) 
v_sub_u32 v27, v1, v17 // (op 4627) 

v_lshlrev_b32 v17, 4, v27 // (op 4627) 

v_lshrrev_b32 v27, 3, v1 // (op 4627) 
v_lshl_add_u32 v20, v27, 7, v17 // (op 4627) 


s_nop 0
v_readfirstlane_b32 s1, v20 // (op 4627) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -4, v1 // (op 4684) 
v_sub_u32 v27, v1, v17 // (op 4684) 

v_lshrrev_b32 v17, 2, v1 // (op 4684) 
v_lshl_add_u32 v20, v17, 3, v27 // (op 4684) 


v_and_b32 v17, -4, v1 // (op 4742) 
v_sub_u32 v21, v1, v17 // (op 4742) 

v_and_b32 v17, 4294967292, v1 // (op 4742) 
v_add_u32 v27, v17, v21 // (op 4742) 


s_ashr_i64 s[6:7], s[36:37], 7 // (op 1366) 
s_and_b32 s10, s6, 4294967294 // (op 1366) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 1365) 
s_mov_b32 s11, s6 // (op 1365) convert

s_cmp_lt_u32 s10, s11 // (op 1365) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_1365 // (op 1365) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_1365 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_1365:
s_barrier  // (op 5035) 
s_add_u32 s10, s10, 1 // (op 1367) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 4222) 
v_mov_b64 v[14:15], s[6:7] // (op 4222) convert


v_mov_b64 v[18:19], v[22:23] // (op 4222) 
v_mov_b64 v[28:29], v[18:19] // (op 4222) 
s_add_u32 s11, 8192, s0 // (op 4222) 
s_mov_b32 m0, s11 // (op 4222) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 4222) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 4222) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 4222) most significant half

s_add_u32 m0, m0, 4096 // (op 4222) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 4222) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 4222) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 4222) most significant half

s_add_u32 m0, m0, 4096 // (op 4222) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 4222) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 4222) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 4222) most significant half

s_add_u32 m0, m0, 4096 // (op 4222) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 4222) Load value direct to lds




v_add_co_u32 v22, vcc, 128, v22 // (op 4260) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 4260) most significant half
s_barrier  // (op 5051) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 4228) 
v_mov_b64 v[14:15], s[6:7] // (op 4228) convert


v_mov_b64 v[18:19], v[30:31] // (op 4228) 
v_mov_b64 v[28:29], v[18:19] // (op 4228) 
s_mov_b32 m0, s1 // (op 4228) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 4228) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 4228) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 4228) most significant half

s_add_u32 m0, m0, 4096 // (op 4228) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 4228) Load value direct to lds



v_add_co_u32 v30, vcc, 128, v30 // (op 4336) least significant half
v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 4336) most significant half

v_mov_b64 v[14:15], s[28:29] // (op 191) convert

v_mov_b32 v21, v16 // (op 191) 
v_mov_b32 v17, v21 // (op 191) 
buffer_load_ubyte v18, v17, s[56:59], 0 offen offset: 0 // (op 191) Load value
v_add_u32 v17, v17, v14 // (op 191) 
buffer_load_ubyte v19, v17, s[56:59], 0 offen offset: 0 // (op 191) Load value



v_add_u32 v16, 4, v16 // (op 4400) 


s_barrier  // (op 5068) 
v_mov_b32 v17, v24 // (op 192) 




buffer_load_ubyte v21, v17, s[60:63], 0 offen offset: 0 // (op 192) Load value

v_add_u32 v24, 4, v24 // (op 4464) 
s_waitcnt vmcnt(2)
ds_write_b8 v20, v18 offset:49408 // (op 188) Store local data 
s_barrier  // (op 5041) 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v19 offset:49412 // (op 188) Store local data 




ds_read_b128 v[32:35], v3 offset:8192 // (op 179) Load local data 
s_barrier  // (op 5086) 
ds_read_b128 v[36:39], v3 offset:8256 // (op 179) Load local data 



s_barrier  // (op 5058) 
s_waitcnt vmcnt(0)
ds_write_b8 v27, v21 offset:49152 // (op 189) Store local data 




ds_read_b128 v[40:43], v4 // (op 181) Load local data 
ds_read_b128 v[44:47], v4 offset:64 // (op 181) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 5076) 



ds_read_u8 v21, v0 offset:49408 // (op 183) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 5093) 



ds_read_u8 v14, v5 offset:49152 // (op 185) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:47], a[0:3], v21, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 186) 
ds_read_b128 v[48:51], v4 offset:4096 // (op 350) Load local data 
ds_read_b128 v[52:55], v4 offset:4160 // (op 350) Load local data 



ds_read_u8 v15, v5 offset:49280 // (op 354) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:39], v[48:55], a[4:7], v21, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 355) 
ds_read_b128 v[56:59], v3 offset:12288 // (op 568) Load local data 


ds_read_b128 v[60:63], v3 offset:12352 // (op 568) Load local data 



ds_read_u8 v21, v0 offset:49536 // (op 571) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:63], v[40:47], a[8:11], v21, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 573) 
ds_read_b128 v[32:35], v3 offset:16384 // (op 954) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[48:55], a[12:15], v21, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 604) 
ds_read_b128 v[36:39], v3 offset:16448 // (op 954) Load local data 





ds_read_u8 v21, v0 offset:49664 // (op 957) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[32:39], v[40:47], a[16:19], v21, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 959) 
ds_read_b128 v[56:59], v3 offset:20480 // (op 1340) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[32:39], v[48:55], a[20:23], v21, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 990) 
ds_read_b128 v[60:63], v3 offset:20544 // (op 1340) Load local data 





ds_read_u8 v21, v0 offset:49792 // (op 1343) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:63], v[40:47], a[24:27], v21, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1345) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:63], v[48:55], a[28:31], v21, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1376) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 1365) 
s_mov_b32 s11, s6 // (op 1365) convert

s_cmp_lt_u32 s10, s11 // (op 1365) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_1365 // (op 1365) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_1365 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_1365:























s_barrier  // (op 5179) 









v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_133
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 



v_mov_b32 v0, v13 // (op 56) reg expression

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 56) Store value


v_accvgpr_read v6, a4 // (op 1869) 
v_accvgpr_read v7, a5 // (op 1869) 
v_accvgpr_read v8, a6 // (op 1869) 
v_accvgpr_read v9, a7 // (op 1869) 
v_mul_f32 v14, v11, v6 // (op 1869) 
v_mul_f32 v15, v11, v7 // (op 1869) 
v_mul_f32 v16, v11, v8 // (op 1869) 
v_mul_f32 v17, v11, v9 // (op 1869) 



s_lshl_b32 s0, s4, 7 // (op 1871) 
v_add_u32 v0, v13, s0 // (op 1871) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1871) Store value


v_accvgpr_read v6, a8 // (op 1893) 
v_accvgpr_read v7, a9 // (op 1893) 
v_accvgpr_read v8, a10 // (op 1893) 
v_accvgpr_read v9, a11 // (op 1893) 
v_mul_f32 v14, v11, v6 // (op 1893) 
v_mul_f32 v15, v11, v7 // (op 1893) 
v_mul_f32 v16, v11, v8 // (op 1893) 
v_mul_f32 v17, v11, v9 // (op 1893) 



v_add_u32 v0, 128, v13 // (op 1896) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1896) Store value


v_accvgpr_read v6, a12 // (op 1902) 
v_accvgpr_read v7, a13 // (op 1902) 
v_accvgpr_read v8, a14 // (op 1902) 
v_accvgpr_read v9, a15 // (op 1902) 
v_mul_f32 v14, v11, v6 // (op 1902) 
v_mul_f32 v15, v11, v7 // (op 1902) 
v_mul_f32 v16, v11, v8 // (op 1902) 
v_mul_f32 v17, v11, v9 // (op 1902) 



s_lshl_b32 s0, s4, 7 // (op 1898) 
s_add_u32 s0, s0, 128 // (op 1898) 
v_add_u32 v0, v13, s0 // (op 1898) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1898) Store value


v_accvgpr_read v6, a16 // (op 1929) 
v_accvgpr_read v7, a17 // (op 1929) 
v_accvgpr_read v8, a18 // (op 1929) 
v_accvgpr_read v9, a19 // (op 1929) 
v_mul_f32 v14, v11, v6 // (op 1929) 
v_mul_f32 v15, v11, v7 // (op 1929) 
v_mul_f32 v16, v11, v8 // (op 1929) 
v_mul_f32 v17, v11, v9 // (op 1929) 



v_add_u32 v0, 256, v13 // (op 1932) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1932) Store value


v_accvgpr_read v6, a20 // (op 1938) 
v_accvgpr_read v7, a21 // (op 1938) 
v_accvgpr_read v8, a22 // (op 1938) 
v_accvgpr_read v9, a23 // (op 1938) 
v_mul_f32 v14, v11, v6 // (op 1938) 
v_mul_f32 v15, v11, v7 // (op 1938) 
v_mul_f32 v16, v11, v8 // (op 1938) 
v_mul_f32 v17, v11, v9 // (op 1938) 



s_lshl_b32 s0, s4, 7 // (op 1934) 
s_add_u32 s0, s0, 256 // (op 1934) 
v_add_u32 v0, v13, s0 // (op 1934) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1934) Store value


v_accvgpr_read v6, a24 // (op 1965) 
v_accvgpr_read v7, a25 // (op 1965) 
v_accvgpr_read v8, a26 // (op 1965) 
v_accvgpr_read v9, a27 // (op 1965) 
v_mul_f32 v14, v11, v6 // (op 1965) 
v_mul_f32 v15, v11, v7 // (op 1965) 
v_mul_f32 v16, v11, v8 // (op 1965) 
v_mul_f32 v17, v11, v9 // (op 1965) 



v_add_u32 v0, 384, v13 // (op 1968) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1968) Store value


v_accvgpr_read v6, a28 // (op 1974) 
v_accvgpr_read v7, a29 // (op 1974) 
v_accvgpr_read v8, a30 // (op 1974) 
v_accvgpr_read v9, a31 // (op 1974) 
v_mul_f32 v14, v11, v6 // (op 1974) 
v_mul_f32 v15, v11, v7 // (op 1974) 
v_mul_f32 v16, v11, v8 // (op 1974) 
v_mul_f32 v17, v11, v9 // (op 1974) 



s_lshl_b32 s0, s4, 7 // (op 1970) 
s_add_u32 s0, s0, 384 // (op 1970) 
v_add_u32 v0, v13, s0 // (op 1970) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1970) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_133:


v_add_u32 v0, 256, v12 // (op 2111) 

buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2111) Load value

v_accvgpr_read v14, a16 // (op 2103) 
v_accvgpr_read v15, a17 // (op 2103) 
v_accvgpr_read v16, a18 // (op 2103) 
v_accvgpr_read v17, a19 // (op 2103) 
v_mul_f32 v18, v11, v14 // (op 2103) 
v_mul_f32 v19, v11, v15 // (op 2103) 
v_mul_f32 v20, v11, v16 // (op 2103) 
v_mul_f32 v21, v11, v17 // (op 2103) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2110) 
v_mul_f32 v15, v10, v7 // (op 2110) 
v_mul_f32 v16, v10, v8 // (op 2110) 
v_mul_f32 v17, v10, v9 // (op 2110) 

v_add_f32 v6, v14, v18 // (op 2104) 
v_add_f32 v7, v15, v19 // (op 2104) 
v_add_f32 v8, v16, v20 // (op 2104) 
v_add_f32 v9, v17, v21 // (op 2104) 




v_add_u32 v0, 256, v13 // (op 2106) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2106) Store value




s_lshl_b32 s0, s2, 7 // (op 2115) 
s_add_u32 s0, s0, 256 // (op 2115) 
v_add_u32 v0, v12, s0 // (op 2115) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2115) Load value

v_accvgpr_read v14, a20 // (op 2117) 
v_accvgpr_read v15, a21 // (op 2117) 
v_accvgpr_read v16, a22 // (op 2117) 
v_accvgpr_read v17, a23 // (op 2117) 
v_mul_f32 v18, v11, v14 // (op 2117) 
v_mul_f32 v19, v11, v15 // (op 2117) 
v_mul_f32 v20, v11, v16 // (op 2117) 
v_mul_f32 v21, v11, v17 // (op 2117) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2113) 
v_mul_f32 v15, v10, v7 // (op 2113) 
v_mul_f32 v16, v10, v8 // (op 2113) 
v_mul_f32 v17, v10, v9 // (op 2113) 

v_add_f32 v6, v14, v18 // (op 2114) 
v_add_f32 v7, v15, v19 // (op 2114) 
v_add_f32 v8, v16, v20 // (op 2114) 
v_add_f32 v9, v17, v21 // (op 2114) 




s_lshl_b32 s0, s4, 7 // (op 2108) 
s_add_u32 s0, s0, 256 // (op 2108) 
v_add_u32 v0, v13, s0 // (op 2108) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2108) Store value




v_add_u32 v0, 128, v12 // (op 2070) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2070) Load value

v_accvgpr_read v14, a8 // (op 2062) 
v_accvgpr_read v15, a9 // (op 2062) 
v_accvgpr_read v16, a10 // (op 2062) 
v_accvgpr_read v17, a11 // (op 2062) 
v_mul_f32 v18, v11, v14 // (op 2062) 
v_mul_f32 v19, v11, v15 // (op 2062) 
v_mul_f32 v20, v11, v16 // (op 2062) 
v_mul_f32 v21, v11, v17 // (op 2062) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2069) 
v_mul_f32 v15, v10, v7 // (op 2069) 
v_mul_f32 v16, v10, v8 // (op 2069) 
v_mul_f32 v17, v10, v9 // (op 2069) 

v_add_f32 v6, v14, v18 // (op 2063) 
v_add_f32 v7, v15, v19 // (op 2063) 
v_add_f32 v8, v16, v20 // (op 2063) 
v_add_f32 v9, v17, v21 // (op 2063) 




v_add_u32 v0, 128, v13 // (op 2065) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2065) Store value




s_lshl_b32 s0, s2, 7 // (op 2074) 
s_add_u32 s0, s0, 128 // (op 2074) 
v_add_u32 v0, v12, s0 // (op 2074) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2074) Load value

v_accvgpr_read v14, a12 // (op 2076) 
v_accvgpr_read v15, a13 // (op 2076) 
v_accvgpr_read v16, a14 // (op 2076) 
v_accvgpr_read v17, a15 // (op 2076) 
v_mul_f32 v18, v11, v14 // (op 2076) 
v_mul_f32 v19, v11, v15 // (op 2076) 
v_mul_f32 v20, v11, v16 // (op 2076) 
v_mul_f32 v21, v11, v17 // (op 2076) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2072) 
v_mul_f32 v15, v10, v7 // (op 2072) 
v_mul_f32 v16, v10, v8 // (op 2072) 
v_mul_f32 v17, v10, v9 // (op 2072) 

v_add_f32 v6, v14, v18 // (op 2073) 
v_add_f32 v7, v15, v19 // (op 2073) 
v_add_f32 v8, v16, v20 // (op 2073) 
v_add_f32 v9, v17, v21 // (op 2073) 




s_lshl_b32 s0, s4, 7 // (op 2067) 
s_add_u32 s0, s0, 128 // (op 2067) 
v_add_u32 v0, v13, s0 // (op 2067) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2067) Store value




v_add_u32 v0, 384, v12 // (op 2152) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2152) Load value

v_accvgpr_read v14, a24 // (op 2144) 
v_accvgpr_read v15, a25 // (op 2144) 
v_accvgpr_read v16, a26 // (op 2144) 
v_accvgpr_read v17, a27 // (op 2144) 
v_mul_f32 v18, v11, v14 // (op 2144) 
v_mul_f32 v19, v11, v15 // (op 2144) 
v_mul_f32 v20, v11, v16 // (op 2144) 
v_mul_f32 v21, v11, v17 // (op 2144) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2151) 
v_mul_f32 v15, v10, v7 // (op 2151) 
v_mul_f32 v16, v10, v8 // (op 2151) 
v_mul_f32 v17, v10, v9 // (op 2151) 

v_add_f32 v6, v14, v18 // (op 2145) 
v_add_f32 v7, v15, v19 // (op 2145) 
v_add_f32 v8, v16, v20 // (op 2145) 
v_add_f32 v9, v17, v21 // (op 2145) 




v_add_u32 v0, 384, v13 // (op 2147) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2147) Store value




s_lshl_b32 s0, s2, 7 // (op 2156) 
s_add_u32 s0, s0, 384 // (op 2156) 
v_add_u32 v0, v12, s0 // (op 2156) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2156) Load value

v_accvgpr_read v14, a28 // (op 2158) 
v_accvgpr_read v15, a29 // (op 2158) 
v_accvgpr_read v16, a30 // (op 2158) 
v_accvgpr_read v17, a31 // (op 2158) 
v_mul_f32 v18, v11, v14 // (op 2158) 
v_mul_f32 v19, v11, v15 // (op 2158) 
v_mul_f32 v20, v11, v16 // (op 2158) 
v_mul_f32 v21, v11, v17 // (op 2158) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2154) 
v_mul_f32 v15, v10, v7 // (op 2154) 
v_mul_f32 v16, v10, v8 // (op 2154) 
v_mul_f32 v17, v10, v9 // (op 2154) 

v_add_f32 v6, v14, v18 // (op 2155) 
v_add_f32 v7, v15, v19 // (op 2155) 
v_add_f32 v8, v16, v20 // (op 2155) 
v_add_f32 v9, v17, v21 // (op 2155) 




s_lshl_b32 s0, s4, 7 // (op 2149) 
s_add_u32 s0, s0, 384 // (op 2149) 
v_add_u32 v0, v13, s0 // (op 2149) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2149) Store value




v_mov_b32 v0, v12 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 145) Load value

v_accvgpr_read v14, a0 // (op 142) 
v_accvgpr_read v15, a1 // (op 142) 
v_accvgpr_read v16, a2 // (op 142) 
v_accvgpr_read v17, a3 // (op 142) 
v_mul_f32 v18, v11, v14 // (op 142) 
v_mul_f32 v19, v11, v15 // (op 142) 
v_mul_f32 v20, v11, v16 // (op 142) 
v_mul_f32 v21, v11, v17 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 146) 
v_mul_f32 v15, v10, v7 // (op 146) 
v_mul_f32 v16, v10, v8 // (op 146) 
v_mul_f32 v17, v10, v9 // (op 146) 

v_add_f32 v6, v14, v18 // (op 143) 
v_add_f32 v7, v15, v19 // (op 143) 
v_add_f32 v8, v16, v20 // (op 143) 
v_add_f32 v9, v17, v21 // (op 143) 




v_mov_b32 v0, v13 // (op 144) reg expression

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 2032) 
v_add_u32 v0, v12, s0 // (op 2032) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2032) Load value

v_accvgpr_read v14, a4 // (op 2029) 
v_accvgpr_read v15, a5 // (op 2029) 
v_accvgpr_read v16, a6 // (op 2029) 
v_accvgpr_read v17, a7 // (op 2029) 
v_mul_f32 v18, v11, v14 // (op 2029) 
v_mul_f32 v19, v11, v15 // (op 2029) 
v_mul_f32 v20, v11, v16 // (op 2029) 
v_mul_f32 v21, v11, v17 // (op 2029) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2033) 
v_mul_f32 v15, v10, v7 // (op 2033) 
v_mul_f32 v16, v10, v8 // (op 2033) 
v_mul_f32 v17, v10, v9 // (op 2033) 

v_add_f32 v6, v14, v18 // (op 2030) 
v_add_f32 v7, v15, v19 // (op 2030) 
v_add_f32 v8, v16, v20 // (op 2030) 
v_add_f32 v9, v17, v21 // (op 2030) 




s_lshl_b32 s0, s4, 7 // (op 2031) 
v_add_u32 v0, v13, s0 // (op 2031) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2031) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_133:
















s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2
.LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2, .LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_end-RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2
  .amdhsa_next_free_vgpr 144
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 50688 // lds bytes
  .amdhsa_accum_offset 112
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: F8
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
    macrotile: [128, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 50688
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 112
    .agpr_count: 32
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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

