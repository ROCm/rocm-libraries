// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2,@function
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 827) 
v_lshrrev_b32 v3, 4, v0 // (op 827) 
v_and_b32 v4, 1, v3 // (op 827) 

v_and_b32 v3, 15, v1 // (op 827) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 827) 


v_lshrrev_b32 v3, 6, v1 // (op 827) 
v_lshl_add_u32 v4, v3, 5, v5 // (op 827) 


v_lshrrev_b32 v3, 5, v0 // (op 827) 

v_lshlrev_b32 v0, 4, v3 // (op 827) 

v_lshl_add_u32 v3, v4, 6, v0 // (op 827) 


v_and_b32 v0, 63, v1 // (op 879) 
v_lshrrev_b32 v4, 4, v0 // (op 879) 
v_and_b32 v5, 1, v4 // (op 879) 

v_and_b32 v4, 15, v1 // (op 879) 
v_lshl_add_u32 v6, v5, 4, v4 // (op 879) 


v_lshrrev_b32 v4, 5, v0 // (op 879) 

v_lshlrev_b32 v0, 4, v4 // (op 879) 

v_lshl_add_u32 v4, v6, 6, v0 // (op 879) 


v_lshrrev_b32 v0, 6, v1 // (op 932) 
v_and_b32 v5, 31, v1 // (op 932) 
v_lshl_add_u32 v6, v0, 5, v5 // (op 932) 


v_and_b32 v0, 63, v1 // (op 932) 
v_lshrrev_b32 v5, 5, v0 // (op 932) 

v_lshl_add_u32 v0, v6, 1, v5 // (op 932) 


v_and_b32 v5, 31, v1 // (op 985) 
v_and_b32 v6, 63, v1 // (op 985) 
v_lshrrev_b32 v7, 5, v6 // (op 985) 

v_lshl_add_u32 v6, v5, 1, v7 // (op 985) 


v_and_b32 v5, 63, v1 // (op 1058) 
v_lshrrev_b32 v7, 4, v5 // (op 1058) 
v_and_b32 v8, 1, v7 // (op 1058) 

v_and_b32 v7, 15, v1 // (op 1058) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 1058) 


v_lshrrev_b32 v7, 6, v1 // (op 1058) 
v_lshl_add_u32 v8, v7, 5, v9 // (op 1058) 


v_lshrrev_b32 v7, 5, v5 // (op 1058) 

v_lshlrev_b32 v5, 4, v7 // (op 1058) 

v_lshl_add_u32 v7, v8, 6, v5 // (op 1058) 


v_and_b32 v5, 63, v1 // (op 1097) 
v_lshrrev_b32 v8, 4, v5 // (op 1097) 
v_and_b32 v9, 1, v8 // (op 1097) 

v_and_b32 v8, 15, v1 // (op 1097) 
v_lshl_add_u32 v10, v9, 4, v8 // (op 1097) 


v_lshrrev_b32 v8, 5, v5 // (op 1097) 

v_lshlrev_b32 v5, 4, v8 // (op 1097) 

v_lshl_add_u32 v8, v10, 6, v5 // (op 1097) 


v_lshrrev_b32 v5, 6, v1 // (op 1121) 
v_and_b32 v9, 31, v1 // (op 1121) 
v_lshl_add_u32 v10, v5, 5, v9 // (op 1121) 


v_and_b32 v5, 63, v1 // (op 1121) 
v_lshrrev_b32 v9, 5, v5 // (op 1121) 

v_lshl_add_u32 v5, v10, 1, v9 // (op 1121) 


v_and_b32 v9, 31, v1 // (op 1145) 
v_and_b32 v10, 63, v1 // (op 1145) 
v_lshrrev_b32 v11, 5, v10 // (op 1145) 

v_lshl_add_u32 v10, v9, 1, v11 // (op 1145) 


v_mov_b32 v9, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 63) call()
v_accvgpr_write a5, 0.00000 // (op 63) call()
v_accvgpr_write a6, 0.00000 // (op 63) call()
v_accvgpr_write a7, 0.00000 // (op 63) call()
v_accvgpr_write a8, 0.00000 // (op 63) call()
v_accvgpr_write a9, 0.00000 // (op 63) call()
v_accvgpr_write a10, 0.00000 // (op 63) call()
v_accvgpr_write a11, 0.00000 // (op 63) call()
v_accvgpr_write a12, 0.00000 // (op 63) call()
v_accvgpr_write a13, 0.00000 // (op 63) call()
v_accvgpr_write a14, 0.00000 // (op 63) call()
v_accvgpr_write a15, 0.00000 // (op 63) call()
v_lshrrev_b32 v12, 2, v1 // (op 538) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 538) 

v_and_b32 v12, -4, v1 // (op 538) 
v_sub_u32 v14, v1, v12 // (op 538) 

v_lshlrev_b32 v12, 4, v14 // (op 538) 

v_mov_b32 v15, 0 // (op 538) convert
v_mov_b32 v14, v12 // (op 538) convert

v_mul_lo_u32 v17, v13, s3 // (op 538) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 538) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 538) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 538) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 538) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 538) most significant half



s_mul_i32 s1, s2, s7 // (op 540) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 540) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 540) least significant
s_mul_hi_u32 s15, s2, s6 // (op 540) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 540) most significant: sum
s_add_u32 s1, s1, s15 // (op 540) most significant: sum


s_mov_b32 s40, s4 // (op 540) reg expression
s_mov_b32 s41, s5 // (op 540) reg expression
s_mov_b32 s42, s0 // (op 540) reg expression
s_mov_b32 s43, 131072 // (op 540) call()

v_lshrrev_b32 v12, 2, v1 // (op 557) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 557) 

v_and_b32 v12, -4, v1 // (op 557) 
v_sub_u32 v14, v1, v12 // (op 557) 

v_lshlrev_b32 v12, 4, v14 // (op 557) 

v_mov_b32 v15, 0 // (op 557) convert
v_mov_b32 v14, v12 // (op 557) convert

v_mul_lo_u32 v19, v13, s3 // (op 557) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 557) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 557) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 557) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 557) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 557) most significant half



s_mul_i32 s1, s2, s7 // (op 559) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 559) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 559) least significant
s_mul_hi_u32 s15, s2, s6 // (op 559) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 559) most significant: sum
s_add_u32 s1, s1, s15 // (op 559) most significant: sum


s_mov_b32 s40, s4 // (op 559) reg expression
s_mov_b32 s41, s5 // (op 559) reg expression
s_mov_b32 s42, s0 // (op 559) reg expression
s_mov_b32 s43, 131072 // (op 559) call()

v_and_b32 v12, -4, v1 // (op 602) 
v_sub_u32 v13, v1, v12 // (op 602) 

v_lshlrev_b32 v12, 4, v13 // (op 602) 

v_mov_b32 v15, 0 // (op 602) convert
v_mov_b32 v14, v12 // (op 602) convert

v_lshrrev_b32 v12, 2, v1 // (op 602) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 602) 

v_mul_lo_u32 v21, v13, s9 // (op 602) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 602) most significant: high of low * low
v_mul_lo_u32 v20, v13, s8 // (op 602) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 602) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 602) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 602) most significant half



s_mul_i32 s1, s8, s13 // (op 604) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 604) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 604) least significant
s_mul_hi_u32 s15, s8, s12 // (op 604) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 604) most significant: sum
s_add_u32 s1, s1, s15 // (op 604) most significant: sum


s_mov_b32 s44, s10 // (op 604) reg expression
s_mov_b32 s45, s11 // (op 604) reg expression
s_mov_b32 s46, s0 // (op 604) reg expression
s_mov_b32 s47, 131072 // (op 604) call()

v_and_b32 v14, -4, v1 // (op 621) 
v_sub_u32 v15, v1, v14 // (op 621) 

v_lshlrev_b32 v14, 4, v15 // (op 621) 

v_mov_b32 v21, 0 // (op 621) convert
v_mov_b32 v20, v14 // (op 621) convert

v_lshrrev_b32 v14, 2, v1 // (op 621) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 621) 

v_mul_lo_u32 v23, v15, s9 // (op 621) most significant: low of low * high
v_mul_hi_u32 v14, v15, s8 // (op 621) most significant: high of low * low
v_mul_lo_u32 v22, v15, s8 // (op 621) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 621) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 621) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 621) most significant half



s_mul_i32 s1, s8, s13 // (op 623) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 623) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 623) least significant
s_mul_hi_u32 s15, s8, s12 // (op 623) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 623) most significant: sum
s_add_u32 s1, s1, s15 // (op 623) most significant: sum


s_mov_b32 s44, s10 // (op 623) reg expression
s_mov_b32 s45, s11 // (op 623) reg expression
s_mov_b32 s46, s0 // (op 623) reg expression
s_mov_b32 s47, 131072 // (op 623) call()

v_lshrrev_b32 v20, 1, v1 // (op 666) 
v_lshl_add_u32 v21, s16, 5, v20 // (op 666) 

v_and_b32 v20, -2, v1 // (op 666) 
v_sub_u32 v22, v1, v20 // (op 666) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v20, v21, s20 // (op 666) 
v_add_u32 v20, v20, v22 // (op 666) 


s_mul_i32 s1, s20, s7 // (op 668) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 668) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 668) least significant
s_mul_hi_u32 s15, s20, s6 // (op 668) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 668) most significant: sum
s_add_u32 s1, s1, s15 // (op 668) most significant: sum


s_mov_b32 s48, s22 // (op 668) reg expression
s_mov_b32 s49, s23 // (op 668) reg expression
s_mov_b32 s50, s0 // (op 668) reg expression
s_mov_b32 s51, 131072 // (op 668) call()

v_lshrrev_b32 v21, 1, v1 // (op 685) 
v_lshl_add_u32 v22, s16, 5, v21 // (op 685) 

v_and_b32 v21, -2, v1 // (op 685) 
v_sub_u32 v23, v1, v21 // (op 685) 

v_mul_lo_u32 v21, v22, s20 // (op 685) 
v_add_u32 v21, v21, v23 // (op 685) 


s_mul_i32 s1, s20, s7 // (op 687) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 687) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 687) least significant
s_mul_hi_u32 s15, s20, s6 // (op 687) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 687) most significant: sum
s_add_u32 s1, s1, s15 // (op 687) most significant: sum


s_mov_b32 s48, s22 // (op 687) reg expression
s_mov_b32 s49, s23 // (op 687) reg expression
s_mov_b32 s50, s0 // (op 687) reg expression
s_mov_b32 s51, 131072 // (op 687) call()

v_and_b32 v22, -2, v1 // (op 730) 
v_sub_u32 v23, v1, v22 // (op 730) 

v_lshrrev_b32 v22, 1, v1 // (op 730) 
v_lshl_add_u32 v24, s17, 5, v22 // (op 730) 

v_mul_lo_u32 v22, v24, s24 // (op 730) 

v_add_u32 v24, v23, v22 // (op 730) 


s_mul_i32 s1, s24, s13 // (op 732) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 732) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 732) least significant
s_mul_hi_u32 s15, s24, s12 // (op 732) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 732) most significant: sum
s_add_u32 s1, s1, s15 // (op 732) most significant: sum


s_mov_b32 s52, s26 // (op 732) reg expression
s_mov_b32 s53, s27 // (op 732) reg expression
s_mov_b32 s54, s0 // (op 732) reg expression
s_mov_b32 s55, 131072 // (op 732) call()

v_and_b32 v22, -2, v1 // (op 749) 
v_sub_u32 v23, v1, v22 // (op 749) 

v_lshrrev_b32 v22, 1, v1 // (op 749) 
v_lshl_add_u32 v25, s17, 5, v22 // (op 749) 

v_mul_lo_u32 v22, v25, s24 // (op 749) 

v_add_u32 v25, v23, v22 // (op 749) 


s_mul_i32 s1, s24, s13 // (op 751) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 751) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 751) least significant
s_mul_hi_u32 s15, s24, s12 // (op 751) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 751) most significant: sum
s_add_u32 s1, s1, s15 // (op 751) most significant: sum


s_mov_b32 s52, s26 // (op 751) reg expression
s_mov_b32 s53, s27 // (op 751) reg expression
s_mov_b32 s54, s0 // (op 751) reg expression
s_mov_b32 s55, 131072 // (op 751) call()

v_and_b32 v22, -4, v1 // (op 867) 
v_sub_u32 v23, v1, v22 // (op 867) 

v_lshlrev_b32 v22, 4, v23 // (op 867) 

v_lshrrev_b32 v23, 2, v1 // (op 867) 
v_lshl_add_u32 v26, v23, 6, v22 // (op 867) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 867) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 4 // (op 487) 
v_mov_b64 v[22:23], s[14:15] // (op 487) convert


v_mov_b64 v[26:27], v[12:13] // (op 487) 
s_mov_b32 m0, s0 // (op 487) reg expression
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 487) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v22 // (op 487) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 487) most significant half

s_add_u32 m0, m0, 1024 // (op 487) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 487) Load value direct to lds






buffer_load_ubyte v22, v24, s[52:55], 0 offen offset: 0 // (op 369) Load value
v_and_b32 v23, -4, v1 // (op 815) 
v_sub_u32 v26, v1, v23 // (op 815) 

v_lshlrev_b32 v23, 4, v26 // (op 815) 

v_lshrrev_b32 v26, 2, v1 // (op 815) 
v_lshl_add_u32 v27, v26, 6, v23 // (op 815) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 815) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 4 // (op 492) 
v_mov_b64 v[26:27], s[14:15] // (op 492) convert


v_mov_b64 v[28:29], v[16:17] // (op 492) 
s_add_u32 s1, 2048, s0 // (op 492) 
s_mov_b32 m0, s1 // (op 492) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 492) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v26 // (op 492) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 492) most significant half

s_add_u32 m0, m0, 1024 // (op 492) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 492) Load value direct to lds







buffer_load_ubyte v23, v20, s[48:51], 0 offen offset: 0 // (op 375) Load value
v_and_b32 v26, -4, v1 // (op 1085) 
v_sub_u32 v27, v1, v26 // (op 1085) 

v_lshlrev_b32 v26, 4, v27 // (op 1085) 

v_lshrrev_b32 v27, 2, v1 // (op 1085) 
v_lshl_add_u32 v28, v27, 6, v26 // (op 1085) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1085) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 4 // (op 497) 
v_mov_b64 v[26:27], s[14:15] // (op 497) convert


v_mov_b64 v[28:29], v[12:13] // (op 497) 
s_add_u32 s1, 4096, s0 // (op 497) 
s_mov_b32 m0, s1 // (op 497) reg expression
buffer_load_dwordx4 v28, s[44:47], 64 offen offset: 0 lds // (op 497) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v26 // (op 497) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 497) most significant half

s_add_u32 m0, m0, 1024 // (op 497) 
buffer_load_dwordx4 v28, s[44:47], 64 offen offset: 0 lds // (op 497) Load value direct to lds








buffer_load_ubyte v12, v24, s[52:55], 0 offen offset:2 // (op 381) Load value

v_and_b32 v13, -4, v1 // (op 1046) 
v_sub_u32 v24, v1, v13 // (op 1046) 

v_lshlrev_b32 v13, 4, v24 // (op 1046) 

v_lshrrev_b32 v24, 2, v1 // (op 1046) 
v_lshl_add_u32 v26, v24, 6, v13 // (op 1046) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1046) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 4 // (op 499) 
v_mov_b64 v[26:27], s[14:15] // (op 499) convert


v_mov_b64 v[28:29], v[16:17] // (op 499) 
s_add_u32 s1, 6144, s0 // (op 499) 
s_mov_b32 m0, s1 // (op 499) reg expression
buffer_load_dwordx4 v28, s[40:43], 64 offen offset: 0 lds // (op 499) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v26 // (op 499) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 499) most significant half

s_add_u32 m0, m0, 1024 // (op 499) 
buffer_load_dwordx4 v28, s[40:43], 64 offen offset: 0 lds // (op 499) Load value direct to lds








buffer_load_ubyte v13, v20, s[48:51], 0 offen offset:2 // (op 387) Load value

v_and_b32 v20, -2, v1 // (op 973) 
v_sub_u32 v24, v1, v20 // (op 973) 

v_and_b32 v20, 4294967294, v1 // (op 973) 
v_add_u32 v16, v20, v24 // (op 973) 





s_waitcnt vmcnt(9)
ds_write_b8 v16, v22 offset:8192 // (op 390) Store local data 

v_and_b32 v20, -2, v1 // (op 920) 
v_sub_u32 v24, v1, v20 // (op 920) 

v_and_b32 v20, 4294967294, v1 // (op 920) 
v_add_u32 v16, v20, v24 // (op 920) 





s_waitcnt vmcnt(6)
ds_write_b8 v16, v23 offset:8256 // (op 392) Store local data 

s_barrier  // (op 1203) 



ds_read_b128 v[26:29], v4 // (op 409) Load local data 
ds_read_b128 v[30:33], v4 offset:32 // (op 409) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1237) 



ds_read_u8 v20, v6 offset:8192 // (op 415) Load local data 
s_barrier  // (op 1187) 



ds_read_b128 v[34:37], v3 offset:2048 // (op 406) Load local data 
ds_read_b128 v[38:41], v3 offset:2080 // (op 406) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1220) 



ds_read_u8 v24, v0 offset:8256 // (op 412) Load local data 
v_and_b32 v16, -4, v1 // (op 787) 
v_sub_u32 v17, v1, v16 // (op 787) 

v_lshlrev_b32 v16, 4, v17 // (op 787) 

v_lshrrev_b32 v17, 2, v1 // (op 787) 
v_lshl_add_u32 v42, v17, 6, v16 // (op 787) 


s_nop 0
v_readfirstlane_b32 s0, v42 // (op 787) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 839) 
v_sub_u32 v17, v1, v16 // (op 839) 

v_lshlrev_b32 v16, 4, v17 // (op 839) 

v_lshrrev_b32 v17, 2, v1 // (op 839) 
v_lshl_add_u32 v42, v17, 6, v16 // (op 839) 


s_nop 0
v_readfirstlane_b32 s1, v42 // (op 839) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -2, v1 // (op 891) 
v_sub_u32 v17, v1, v16 // (op 891) 

v_and_b32 v16, 4294967294, v1 // (op 891) 
v_add_u32 v42, v16, v17 // (op 891) 


v_and_b32 v16, -2, v1 // (op 944) 
v_sub_u32 v17, v1, v16 // (op 944) 

v_and_b32 v16, 4294967294, v1 // (op 944) 
v_add_u32 v43, v16, v17 // (op 944) 


v_and_b32 v16, -4, v1 // (op 1031) 
v_sub_u32 v17, v1, v16 // (op 1031) 

v_lshlrev_b32 v16, 4, v17 // (op 1031) 

v_lshrrev_b32 v17, 2, v1 // (op 1031) 
v_lshl_add_u32 v44, v17, 6, v16 // (op 1031) 


s_nop 0
v_readfirstlane_b32 s14, v44 // (op 1031) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 1070) 
v_sub_u32 v17, v1, v16 // (op 1070) 

v_lshlrev_b32 v16, 4, v17 // (op 1070) 

v_lshrrev_b32 v17, 2, v1 // (op 1070) 
v_lshl_add_u32 v44, v17, 6, v16 // (op 1070) 


s_nop 0
v_readfirstlane_b32 s15, v44 // (op 1070) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -2, v1 // (op 1109) 
v_sub_u32 v17, v1, v16 // (op 1109) 

v_and_b32 v16, 4294967294, v1 // (op 1109) 
v_add_u32 v44, v16, v17 // (op 1109) 


v_and_b32 v16, -2, v1 // (op 1133) 
v_sub_u32 v17, v1, v16 // (op 1133) 

v_and_b32 v16, 4294967294, v1 // (op 1133) 
v_add_u32 v45, v16, v17 // (op 1133) 


s_mov_b32 s18, 0 // (op 59) call()
s_ashr_i64 s[38:39], s[28:29], 6 // (op 58) 
s_mov_b32 s19, s38 // (op 58) convert

s_and_b32 s38, s19, 4294967294 // (op 58) 

s_cmp_lt_u32 s18, s38 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58:
s_add_u32 s18, s18, 2 // (op 60) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[34:41], v[26:33], a[0:15], v24, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 88) 
s_barrier  // (op 1206) 

s_lshl_b64 s[38:39], s[8:9], 4 // (op 459) 
v_mov_b64 v[16:17], s[38:39] // (op 459) convert


v_mov_b64 v[46:47], v[14:15] // (op 459) 
v_mov_b64 v[48:49], v[46:47] // (op 459) 
s_mov_b32 m0, s1 // (op 459) reg expression
s_mov_b32 s19, 128 // (op 459) call()
buffer_load_dwordx4 v48, s[44:47], s19 offen offset: 0 lds // (op 459) Load value direct to lds
v_add_co_u32 v48, s[38:39], v48, v16 // (op 459) least significant half
v_addc_co_u32 v49, s[38:39], 0, v49, s[38:39] // (op 459) most significant half

s_add_u32 m0, m0, 1024 // (op 459) 
buffer_load_dwordx4 v48, s[44:47], s19 offen offset: 0 lds // (op 459) Load value direct to lds





v_mov_b32 v16, v25 // (op 25) 

buffer_load_ubyte v22, v16, s[52:55], 0 offen offset:4 // (op 25) Load value
s_barrier  // (op 1190) 

s_lshl_b64 s[38:39], s[2:3], 4 // (op 454) 
v_mov_b64 v[48:49], s[38:39] // (op 454) convert


v_mov_b64 v[50:51], v[18:19] // (op 454) 
v_mov_b64 v[52:53], v[50:51] // (op 454) 
s_add_u32 s19, 2048, s0 // (op 454) 
s_mov_b32 m0, s19 // (op 454) reg expression
s_mov_b32 s38, 128 // (op 454) call()
buffer_load_dwordx4 v52, s[40:43], s38 offen offset: 0 lds // (op 454) Load value direct to lds
v_add_co_u32 v52, s[56:57], v52, v48 // (op 454) least significant half
v_addc_co_u32 v53, s[56:57], 0, v53, s[56:57] // (op 454) most significant half

s_add_u32 m0, m0, 1024 // (op 454) 
buffer_load_dwordx4 v52, s[40:43], s38 offen offset: 0 lds // (op 454) Load value direct to lds






v_mov_b32 v17, v21 // (op 17) 

buffer_load_ubyte v23, v17, s[48:51], 0 offen offset:4 // (op 17) Load value
s_barrier  // (op 1276) 



s_waitcnt vmcnt(9)
ds_write_b8 v45, v12 offset:8320 // (op 255) Store local data 
s_barrier  // (op 1270) 



s_waitcnt vmcnt(6)
ds_write_b8 v44, v13 offset:8384 // (op 254) Store local data 
s_barrier  // (op 1260) 



ds_read_b128 v[26:29], v8 offset:4096 // (op 247) Load local data 
ds_read_b128 v[30:33], v8 offset:4128 // (op 247) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1273) 



ds_read_u8 v20, v10 offset:8320 // (op 251) Load local data 
s_barrier  // (op 1254) 
s_barrier  // (op 1327) 



ds_read_b128 v[34:37], v7 offset:6144 // (op 245) Load local data 
ds_read_b128 v[38:41], v7 offset:6176 // (op 245) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1267) 



ds_read_u8 v24, v5 offset:8384 // (op 249) Load local data 
s_barrier  // (op 1321) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[34:41], v[26:33], a[0:15], v24, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 252) 
s_barrier  // (op 1264) 

s_lshl_b64 s[38:39], s[8:9], 4 // (op 482) 
v_mov_b64 v[48:49], s[38:39] // (op 482) convert


v_mov_b64 v[52:53], v[46:47] // (op 482) 
s_add_u32 s19, 4096, s15 // (op 482) 
s_mov_b32 m0, s19 // (op 482) reg expression
s_mov_b32 s38, 192 // (op 482) call()
buffer_load_dwordx4 v52, s[44:47], s38 offen offset: 0 lds // (op 482) Load value direct to lds
v_add_co_u32 v52, s[56:57], v52, v48 // (op 482) least significant half
v_addc_co_u32 v53, s[56:57], 0, v53, s[56:57] // (op 482) most significant half

s_add_u32 m0, m0, 1024 // (op 482) 
buffer_load_dwordx4 v52, s[44:47], s38 offen offset: 0 lds // (op 482) Load value direct to lds








v_add_co_u32 v14, vcc, 128, v14 // (op 613) least significant half
buffer_load_ubyte v12, v16, s[52:55], 0 offen offset:6 // (op 258) Load value
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 613) most significant half

s_barrier  // (op 1257) 
v_add_u32 v25, 4, v25 // (op 741) 

s_lshl_b64 s[38:39], s[2:3], 4 // (op 477) 
v_mov_b64 v[46:47], s[38:39] // (op 477) convert


v_mov_b64 v[48:49], v[50:51] // (op 477) 
s_add_u32 s19, 6144, s14 // (op 477) 
s_mov_b32 m0, s19 // (op 477) reg expression
s_mov_b32 s38, 192 // (op 477) call()
buffer_load_dwordx4 v48, s[40:43], s38 offen offset: 0 lds // (op 477) Load value direct to lds
v_add_co_u32 v48, s[56:57], v48, v46 // (op 477) least significant half
v_addc_co_u32 v49, s[56:57], 0, v49, s[56:57] // (op 477) most significant half

s_add_u32 m0, m0, 1024 // (op 477) 
buffer_load_dwordx4 v48, s[40:43], s38 offen offset: 0 lds // (op 477) Load value direct to lds








v_add_co_u32 v18, vcc, 128, v18 // (op 549) least significant half
buffer_load_ubyte v13, v17, s[48:51], 0 offen offset:6 // (op 257) Load value
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 549) most significant half

s_barrier  // (op 1240) 
v_add_u32 v21, 4, v21 // (op 677) 



s_waitcnt vmcnt(9)
ds_write_b8 v43, v22 offset:8192 // (op 30) Store local data 
s_barrier  // (op 1223) 



s_waitcnt vmcnt(6)
ds_write_b8 v42, v23 offset:8256 // (op 21) Store local data 
s_barrier  // (op 1212) 



ds_read_b128 v[26:29], v4 // (op 75) Load local data 
ds_read_b128 v[30:33], v4 offset:32 // (op 75) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1247) 



ds_read_u8 v20, v6 offset:8192 // (op 85) Load local data 
s_barrier  // (op 1196) 



ds_read_b128 v[34:37], v3 offset:2048 // (op 70) Load local data 
ds_read_b128 v[38:41], v3 offset:2080 // (op 70) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1230) 



ds_read_u8 v24, v0 offset:8256 // (op 80) Load local data 
s_ashr_i64 s[38:39], s[28:29], 6 // (op 58) 
s_mov_b32 s19, s38 // (op 58) convert

s_and_b32 s38, s19, 4294967294 // (op 58) 

s_cmp_lt_u32 s18, s38 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58:















s_barrier  // (op 1287) 
v_lshrrev_b32 v25, 2, v1 // (op 512) 
s_ashr_i64 s[0:1], s[28:29], 6 // (op 512) 
v_lshl_add_u32 v14, s16, 5, v25 // (op 512) 

s_lshr_b32 s14, s0, 1 // (op 512) 

v_and_b32 v25, -4, v1 // (op 512) 
v_sub_u32 v15, v1, v25 // (op 512) 

v_lshlrev_b32 v25, 4, v15 // (op 512) 

v_lshl_add_u32 v15, s14, 7, v25 // (op 512) 


v_mov_b32 v23, 0 // (op 512) convert
v_mov_b32 v22, v15 // (op 512) convert

v_mul_lo_u32 v19, v14, s3 // (op 512) most significant: low of low * high
v_mul_hi_u32 v21, v14, s2 // (op 512) most significant: high of low * low
v_mul_lo_u32 v18, v14, s2 // (op 512) least significant: low of low * low
v_add_u32 v19, v19, v21 // (op 512) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v22 // (op 512) least significant half
v_addc_co_u32 v19, s[0:1], v19, v23, s[0:1] // (op 512) most significant half



s_mul_i32 s1, s2, s7 // (op 514) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 514) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 514) least significant
s_mul_hi_u32 s15, s2, s6 // (op 514) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 514) most significant: sum
s_add_u32 s1, s1, s15 // (op 514) most significant: sum


s_mov_b32 s40, s4 // (op 514) reg expression
s_mov_b32 s41, s5 // (op 514) reg expression
s_mov_b32 s42, s0 // (op 514) reg expression
s_mov_b32 s43, 131072 // (op 514) call()

v_and_b32 v25, -4, v1 // (op 579) 
s_ashr_i64 s[0:1], s[28:29], 6 // (op 579) 
v_sub_u32 v14, v1, v25 // (op 579) 

s_lshr_b32 s14, s0, 1 // (op 579) 

v_lshlrev_b32 v25, 4, v14 // (op 579) 

v_lshl_add_u32 v14, s14, 7, v25 // (op 579) 


v_mov_b32 v17, 0 // (op 579) convert
v_mov_b32 v16, v14 // (op 579) convert

v_lshrrev_b32 v25, 2, v1 // (op 579) 
v_lshl_add_u32 v14, s17, 5, v25 // (op 579) 

v_mul_lo_u32 v23, v14, s9 // (op 579) most significant: low of low * high
v_mul_hi_u32 v15, v14, s8 // (op 579) most significant: high of low * low
v_mul_lo_u32 v22, v14, s8 // (op 579) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 579) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v22 // (op 579) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 579) most significant half



s_mul_i32 s1, s8, s13 // (op 581) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 581) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 581) least significant
s_mul_hi_u32 s15, s8, s12 // (op 581) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 581) most significant: sum
s_add_u32 s1, s1, s15 // (op 581) most significant: sum


s_mov_b32 s44, s10 // (op 581) reg expression
s_mov_b32 s45, s11 // (op 581) reg expression
s_mov_b32 s46, s0 // (op 581) reg expression
s_mov_b32 s47, 131072 // (op 581) call()

v_lshrrev_b32 v25, 1, v1 // (op 643) 
s_ashr_i64 s[0:1], s[28:29], 6 // (op 643) 
v_lshl_add_u32 v16, s16, 5, v25 // (op 643) 

s_lshr_b32 s14, s0, 1 // (op 643) 

v_and_b32 v17, -2, v1 // (op 643) 
v_sub_u32 v25, v1, v17 // (op 643) 

v_lshl_add_u32 v17, s14, 2, v25 // (op 643) 


v_mul_lo_u32 v25, v16, s20 // (op 643) 
v_add_u32 v25, v25, v17 // (op 643) 


s_mul_i32 s1, s20, s7 // (op 645) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 645) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 645) least significant
s_mul_hi_u32 s15, s20, s6 // (op 645) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 645) most significant: sum
s_add_u32 s1, s1, s15 // (op 645) most significant: sum


s_mov_b32 s48, s22 // (op 645) reg expression
s_mov_b32 s49, s23 // (op 645) reg expression
s_mov_b32 s50, s0 // (op 645) reg expression
s_mov_b32 s51, 131072 // (op 645) call()

v_and_b32 v16, -2, v1 // (op 707) 
s_ashr_i64 s[0:1], s[28:29], 6 // (op 707) 
v_sub_u32 v17, v1, v16 // (op 707) 

s_lshr_b32 s14, s0, 1 // (op 707) 

v_lshrrev_b32 v16, 1, v1 // (op 707) 
v_lshl_add_u32 v21, s14, 2, v17 // (op 707) 


v_lshl_add_u32 v17, s17, 5, v16 // (op 707) 

v_mul_lo_u32 v16, v17, s24 // (op 707) 

v_add_u32 v17, v21, v16 // (op 707) 


s_mul_i32 s1, s24, s13 // (op 709) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 709) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 709) least significant
s_mul_hi_u32 s15, s24, s12 // (op 709) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 709) most significant: sum
s_add_u32 s1, s1, s15 // (op 709) most significant: sum


s_mov_b32 s52, s26 // (op 709) reg expression
s_mov_b32 s53, s27 // (op 709) reg expression
s_mov_b32 s54, s0 // (op 709) reg expression
s_mov_b32 s55, 131072 // (op 709) call()

v_and_b32 v16, -4, v1 // (op 800) 
v_sub_u32 v21, v1, v16 // (op 800) 

v_lshlrev_b32 v16, 4, v21 // (op 800) 

v_lshrrev_b32 v21, 2, v1 // (op 800) 
v_lshl_add_u32 v22, v21, 6, v16 // (op 800) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 800) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 852) 
v_sub_u32 v21, v1, v16 // (op 852) 

v_lshlrev_b32 v16, 4, v21 // (op 852) 

v_lshrrev_b32 v21, 2, v1 // (op 852) 
v_lshl_add_u32 v22, v21, 6, v16 // (op 852) 


s_nop 0
v_readfirstlane_b32 s1, v22 // (op 852) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -2, v1 // (op 904) 
v_sub_u32 v21, v1, v16 // (op 904) 

v_and_b32 v16, 4294967294, v1 // (op 904) 
v_add_u32 v22, v16, v21 // (op 904) 


v_and_b32 v16, -2, v1 // (op 957) 
v_sub_u32 v21, v1, v16 // (op 957) 

v_and_b32 v16, 4294967294, v1 // (op 957) 
v_add_u32 v23, v16, v21 // (op 957) 


s_ashr_i64 s[14:15], s[28:29], 6 // (op 172) 
s_and_b32 s18, s14, 4294967294 // (op 172) 

s_ashr_i64 s[14:15], s[28:29], 6 // (op 171) 
s_mov_b32 s19, s14 // (op 171) convert

s_cmp_lt_u32 s18, s19 // (op 171) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171 // (op 171) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171:
s_barrier  // (op 1193) 
s_add_u32 s18, s18, 1 // (op 173) 

s_lshl_b64 s[14:15], s[2:3], 4 // (op 464) 
v_mov_b64 v[12:13], s[14:15] // (op 464) convert


v_mov_b64 v[20:21], v[18:19] // (op 464) 
v_mov_b64 v[26:27], v[20:21] // (op 464) 
s_add_u32 s19, 2048, s0 // (op 464) 
s_mov_b32 m0, s19 // (op 464) reg expression
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 464) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v12 // (op 464) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 464) most significant half

s_add_u32 m0, m0, 1024 // (op 464) 
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 464) Load value direct to lds




v_add_co_u32 v18, s[14:15], 64, v18 // (op 502) least significant half
v_addc_co_u32 v19, s[14:15], 0, v19, s[14:15] // (op 502) most significant half

s_barrier  // (op 1209) 

s_lshl_b64 s[14:15], s[8:9], 4 // (op 470) 
v_mov_b64 v[12:13], s[14:15] // (op 470) convert


v_mov_b64 v[20:21], v[14:15] // (op 470) 
v_mov_b64 v[26:27], v[20:21] // (op 470) 
s_mov_b32 m0, s1 // (op 470) reg expression
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 470) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v12 // (op 470) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 470) most significant half

s_add_u32 m0, m0, 1024 // (op 470) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 470) Load value direct to lds



v_add_co_u32 v14, s[14:15], 64, v14 // (op 571) least significant half
v_addc_co_u32 v15, s[14:15], 0, v15, s[14:15] // (op 571) most significant half



v_mov_b32 v16, v25 // (op 191) 

buffer_load_ubyte v24, v16, s[48:51], 0 offen offset: 0 // (op 191) Load value

v_add_u32 v25, 2, v25 // (op 635) 


s_barrier  // (op 1226) 
v_mov_b32 v16, v17 // (op 192) 




buffer_load_ubyte v12, v16, s[52:55], 0 offen offset: 0 // (op 192) Load value

v_add_u32 v17, 2, v17 // (op 699) 
s_waitcnt vmcnt(1)
ds_write_b8 v22, v24 offset:8256 // (op 188) Store local data 

s_barrier  // (op 1199) 



s_barrier  // (op 1243) 
ds_read_b128 v[26:29], v3 offset:2048 // (op 179) Load local data 



ds_read_b128 v[30:33], v3 offset:2080 // (op 179) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v12 offset:8192 // (op 189) Store local data 

s_barrier  // (op 1216) 



ds_read_b128 v[34:37], v4 // (op 181) Load local data 
ds_read_b128 v[38:41], v4 offset:32 // (op 181) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1233) 



ds_read_u8 v16, v0 offset:8256 // (op 183) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1250) 



ds_read_u8 v24, v6 offset:8192 // (op 185) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[26:33], v[34:41], a[0:15], v16, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 186) 




s_ashr_i64 s[14:15], s[28:29], 6 // (op 171) 
s_mov_b32 s19, s14 // (op 171) convert

s_cmp_lt_u32 s18, s19 // (op 171) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 // (op 171) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 if true)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171:























s_barrier  // (op 1313) 








v_cmp_eq_f32 vcc, 0.00000, v9 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133
s_nop 4
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_accvgpr_read v16, a4 // (op 50) 
v_accvgpr_read v17, a5 // (op 50) 
v_accvgpr_read v18, a6 // (op 50) 
v_accvgpr_read v19, a7 // (op 50) 
v_accvgpr_read v20, a8 // (op 50) 
v_accvgpr_read v21, a9 // (op 50) 
v_accvgpr_read v22, a10 // (op 50) 
v_accvgpr_read v23, a11 // (op 50) 
v_accvgpr_read v24, a12 // (op 50) 
v_accvgpr_read v25, a13 // (op 50) 
v_accvgpr_read v26, a14 // (op 50) 
v_accvgpr_read v27, a15 // (op 50) 
v_mul_f32 v28, v11, v12 // (op 50) 
v_mul_f32 v29, v11, v13 // (op 50) 
v_mul_f32 v30, v11, v14 // (op 50) 
v_mul_f32 v31, v11, v15 // (op 50) 
v_mul_f32 v32, v11, v16 // (op 50) 
v_mul_f32 v33, v11, v17 // (op 50) 
v_mul_f32 v34, v11, v18 // (op 50) 
v_mul_f32 v35, v11, v19 // (op 50) 
v_mul_f32 v36, v11, v20 // (op 50) 
v_mul_f32 v37, v11, v21 // (op 50) 
v_mul_f32 v38, v11, v22 // (op 50) 
v_mul_f32 v39, v11, v23 // (op 50) 
v_mul_f32 v40, v11, v24 // (op 50) 
v_mul_f32 v41, v11, v25 // (op 50) 
v_mul_f32 v42, v11, v26 // (op 50) 
v_mul_f32 v43, v11, v27 // (op 50) 

v_lshrrev_b32 v0, 6, v1 // (op 999) 
v_and_b32 v10, 63, v1 // (op 999) 
v_lshrrev_b32 v3, 5, v10 // (op 999) 
v_lshlrev_b32 v4, 2, v3 // (op 999) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 999) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 999) 

v_lshrrev_b32 v3, 2, v10 // (op 999) 

v_and_b32 v10, 7, v3 // (op 999) 

v_and_b32 v3, 3, v1 // (op 999) 
v_lshl_add_u32 v4, v10, 2, v3 // (op 999) 


v_lshl_add_u32 v3, s17, 5, v4 // (op 999) 

v_mul_lo_u32 v10, v3, s30 // (op 999) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 999) 


s_mul_i32 s15, s30, s13 // (op 1001) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 1001) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1001) least significant
s_mul_hi_u32 s1, s30, s12 // (op 1001) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1001) most significant: sum
s_add_u32 s15, s15, s1 // (op 1001) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 1001) 

s_mov_b32 s8, s32 // (op 1001) reg expression
s_mov_b32 s9, s33 // (op 1001) reg expression
s_mov_b32 s10, s0 // (op 1001) reg expression
s_mov_b32 s11, 131072 // (op 1001) call()




buffer_store_dwordx4 v[28:31], v3, s[8:11], 0 offen // (op 56) Store value
buffer_store_dwordx4 v[32:35], v3, s[8:11], 0 offen offset:32 // (op 56) Store value
buffer_store_dwordx4 v[36:39], v3, s[8:11], 0 offen offset:64 // (op 56) Store value
buffer_store_dwordx4 v[40:43], v3, s[8:11], 0 offen offset:96 // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133:
v_lshrrev_b32 v0, 6, v1 // (op 770) 
v_and_b32 v10, 63, v1 // (op 770) 
v_lshrrev_b32 v3, 5, v10 // (op 770) 
v_lshlrev_b32 v4, 2, v3 // (op 770) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 770) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 770) 

v_lshrrev_b32 v3, 2, v10 // (op 770) 

v_and_b32 v10, 7, v3 // (op 770) 

v_and_b32 v3, 3, v1 // (op 770) 
v_lshl_add_u32 v4, v10, 2, v3 // (op 770) 


v_lshl_add_u32 v3, s17, 5, v4 // (op 770) 

v_mul_lo_u32 v10, v3, s34 // (op 770) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 770) 


s_mul_i32 s15, s34, s13 // (op 772) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 772) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 772) least significant
s_mul_hi_u32 s1, s34, s12 // (op 772) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 772) most significant: sum
s_add_u32 s15, s15, s1 // (op 772) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 772) 

s_mov_b32 s4, s36 // (op 772) reg expression
s_mov_b32 s5, s37 // (op 772) reg expression
s_mov_b32 s6, s0 // (op 772) reg expression
s_mov_b32 s7, 131072 // (op 772) call()





buffer_load_dwordx4 v[12:15], v3, s[4:7], 0 offen // (op 145) Load value
buffer_load_dwordx4 v[16:19], v3, s[4:7], 0 offen offset:32 // (op 145) Load value
buffer_load_dwordx4 v[20:23], v3, s[4:7], 0 offen offset:64 // (op 145) Load value
buffer_load_dwordx4 v[24:27], v3, s[4:7], 0 offen offset:96 // (op 145) Load value



v_accvgpr_read v28, a0 // (op 142) 
v_accvgpr_read v29, a1 // (op 142) 
v_accvgpr_read v30, a2 // (op 142) 
v_accvgpr_read v31, a3 // (op 142) 
v_accvgpr_read v32, a4 // (op 142) 
v_accvgpr_read v33, a5 // (op 142) 
v_accvgpr_read v34, a6 // (op 142) 
v_accvgpr_read v35, a7 // (op 142) 
v_accvgpr_read v36, a8 // (op 142) 
v_accvgpr_read v37, a9 // (op 142) 
v_accvgpr_read v38, a10 // (op 142) 
v_accvgpr_read v39, a11 // (op 142) 
v_accvgpr_read v40, a12 // (op 142) 
v_accvgpr_read v41, a13 // (op 142) 
v_accvgpr_read v42, a14 // (op 142) 
v_accvgpr_read v43, a15 // (op 142) 
v_mul_f32 v44, v11, v28 // (op 142) 
v_mul_f32 v45, v11, v29 // (op 142) 
v_mul_f32 v46, v11, v30 // (op 142) 
v_mul_f32 v47, v11, v31 // (op 142) 
v_mul_f32 v48, v11, v32 // (op 142) 
v_mul_f32 v49, v11, v33 // (op 142) 
v_mul_f32 v50, v11, v34 // (op 142) 
v_mul_f32 v51, v11, v35 // (op 142) 
v_mul_f32 v52, v11, v36 // (op 142) 
v_mul_f32 v53, v11, v37 // (op 142) 
v_mul_f32 v54, v11, v38 // (op 142) 
v_mul_f32 v55, v11, v39 // (op 142) 
v_mul_f32 v56, v11, v40 // (op 142) 
v_mul_f32 v57, v11, v41 // (op 142) 
v_mul_f32 v58, v11, v42 // (op 142) 
v_mul_f32 v59, v11, v43 // (op 142) 

s_waitcnt vmcnt(3)
v_mul_f32 v28, v9, v12 // (op 146) 
v_mul_f32 v29, v9, v13 // (op 146) 
v_mul_f32 v30, v9, v14 // (op 146) 
v_mul_f32 v31, v9, v15 // (op 146) 
s_waitcnt vmcnt(2)
v_mul_f32 v32, v9, v16 // (op 146) 
v_mul_f32 v33, v9, v17 // (op 146) 
v_mul_f32 v34, v9, v18 // (op 146) 
v_mul_f32 v35, v9, v19 // (op 146) 
s_waitcnt vmcnt(1)
v_mul_f32 v36, v9, v20 // (op 146) 
v_mul_f32 v37, v9, v21 // (op 146) 
v_mul_f32 v38, v9, v22 // (op 146) 
v_mul_f32 v39, v9, v23 // (op 146) 
s_waitcnt vmcnt(0)
v_mul_f32 v40, v9, v24 // (op 146) 
v_mul_f32 v41, v9, v25 // (op 146) 
v_mul_f32 v42, v9, v26 // (op 146) 
v_mul_f32 v43, v9, v27 // (op 146) 

v_add_f32 v12, v28, v44 // (op 143) 
v_add_f32 v13, v29, v45 // (op 143) 
v_add_f32 v14, v30, v46 // (op 143) 
v_add_f32 v15, v31, v47 // (op 143) 
v_add_f32 v16, v32, v48 // (op 143) 
v_add_f32 v17, v33, v49 // (op 143) 
v_add_f32 v18, v34, v50 // (op 143) 
v_add_f32 v19, v35, v51 // (op 143) 
v_add_f32 v20, v36, v52 // (op 143) 
v_add_f32 v21, v37, v53 // (op 143) 
v_add_f32 v22, v38, v54 // (op 143) 
v_add_f32 v23, v39, v55 // (op 143) 
v_add_f32 v24, v40, v56 // (op 143) 
v_add_f32 v25, v41, v57 // (op 143) 
v_add_f32 v26, v42, v58 // (op 143) 
v_add_f32 v27, v43, v59 // (op 143) 


v_lshrrev_b32 v0, 6, v1 // (op 1016) 
v_and_b32 v10, 63, v1 // (op 1016) 
v_lshrrev_b32 v3, 5, v10 // (op 1016) 
v_lshlrev_b32 v4, 2, v3 // (op 1016) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 1016) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 1016) 

v_lshrrev_b32 v3, 2, v10 // (op 1016) 

v_and_b32 v10, 7, v3 // (op 1016) 

v_and_b32 v3, 3, v1 // (op 1016) 
v_lshl_add_u32 v4, v10, 2, v3 // (op 1016) 


v_lshl_add_u32 v3, s17, 5, v4 // (op 1016) 

v_mul_lo_u32 v10, v3, s30 // (op 1016) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 1016) 


s_mul_i32 s15, s30, s13 // (op 1018) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 1018) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1018) least significant
s_mul_hi_u32 s1, s30, s12 // (op 1018) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1018) most significant: sum
s_add_u32 s15, s15, s1 // (op 1018) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 1018) 

s_mov_b32 s8, s32 // (op 1018) reg expression
s_mov_b32 s9, s33 // (op 1018) reg expression
s_mov_b32 s10, s0 // (op 1018) reg expression
s_mov_b32 s11, 131072 // (op 1018) call()




buffer_store_dwordx4 v[12:15], v3, s[8:11], 0 offen // (op 144) Store value
buffer_store_dwordx4 v[16:19], v3, s[8:11], 0 offen offset:32 // (op 144) Store value
buffer_store_dwordx4 v[20:23], v3, s[8:11], 0 offen offset:64 // (op 144) Store value
buffer_store_dwordx4 v[24:27], v3, s[8:11], 0 offen offset:96 // (op 144) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133:




s_endpgm  // End of RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.LRR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end:
.size RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2, .LRR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end-RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
  .amdhsa_next_free_vgpr 76
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 8448 // lds bytes
  .amdhsa_accum_offset 60
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
    DataTypeB: B8
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
    args: [ { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeFree1 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC } ]
    macrotile: [32, 32, 64]
    threads: [64, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
    .symbol: RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 8448
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 58
    .vgpr_count: 60
    .agpr_count: 16
    .max_flat_workgroup_size: 64
    .workgroup_size: [64, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 32
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 40
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
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
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

