// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2,@function
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 827) 
v_and_b32 v3, 63, v1 // (op 827) 
v_sub_u32 v4, v3, v0 // (op 827) 

v_lshrrev_b32 v3, 1, v4 // (op 827) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 827) 


v_and_b32 v0, 1, v4 // (op 827) 

v_lshlrev_b32 v3, 3, v0 // (op 827) 

v_lshrrev_b32 v0, 6, v1 // (op 827) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 827) 


v_add_u32 v0, v5, v4 // (op 827) 


v_and_b32 v3, 48, v1 // (op 879) 
v_and_b32 v4, 63, v1 // (op 879) 
v_sub_u32 v5, v4, v3 // (op 879) 

v_lshrrev_b32 v4, 1, v5 // (op 879) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 879) 


v_and_b32 v3, 1, v5 // (op 879) 

v_lshlrev_b32 v4, 3, v3 // (op 879) 

v_add_u32 v3, v6, v4 // (op 879) 


v_lshrrev_b32 v4, 6, v1 // (op 932) 
v_and_b32 v5, 15, v1 // (op 932) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 932) 


v_and_b32 v4, 48, v1 // (op 932) 
v_add_u32 v5, v4, v6 // (op 932) 


v_and_b32 v4, 48, v1 // (op 985) 
v_and_b32 v6, 15, v1 // (op 985) 
v_add_u32 v7, v4, v6 // (op 985) 


v_and_b32 v4, 48, v1 // (op 1058) 
v_and_b32 v6, 63, v1 // (op 1058) 
v_sub_u32 v8, v6, v4 // (op 1058) 

v_lshrrev_b32 v6, 1, v8 // (op 1058) 
v_add_lshl_u32 v9, v4, v6, 4 // (op 1058) 


v_and_b32 v4, 1, v8 // (op 1058) 

v_lshlrev_b32 v6, 3, v4 // (op 1058) 

v_lshrrev_b32 v4, 6, v1 // (op 1058) 
v_lshl_add_u32 v8, v4, 4, v6 // (op 1058) 


v_add_u32 v4, v9, v8 // (op 1058) 


v_and_b32 v6, 48, v1 // (op 1097) 
v_and_b32 v8, 63, v1 // (op 1097) 
v_sub_u32 v9, v8, v6 // (op 1097) 

v_lshrrev_b32 v8, 1, v9 // (op 1097) 
v_add_lshl_u32 v10, v6, v8, 4 // (op 1097) 


v_and_b32 v6, 1, v9 // (op 1097) 

v_lshlrev_b32 v8, 3, v6 // (op 1097) 

v_add_u32 v6, v10, v8 // (op 1097) 


v_lshrrev_b32 v8, 6, v1 // (op 1121) 
v_and_b32 v9, 15, v1 // (op 1121) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 1121) 


v_and_b32 v8, 48, v1 // (op 1121) 
v_add_u32 v9, v8, v10 // (op 1121) 


v_and_b32 v8, 48, v1 // (op 1145) 
v_and_b32 v10, 15, v1 // (op 1145) 
v_add_u32 v11, v8, v10 // (op 1145) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v10, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_lshl_b32 s0, s16, 4 // (op 538) 
v_mul_lo_u32 v13, v1, s3 // (op 538) most significant: low of low * high
s_mov_b32 s15, 0 // (op 538) convert
s_mov_b32 s14, s0 // (op 538) convert

v_mul_hi_u32 v14, v1, s2 // (op 538) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 538) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 538) most significant: sum

v_mov_b32 v14, s14 // (op 538) 
v_mov_b32 v15, s15 // (op 538) 
v_add_co_u32 v16, s[0:1], v14, v12 // (op 538) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 538) most significant half





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

s_lshl_b32 s0, s16, 4 // (op 557) 
v_mul_lo_u32 v13, v1, s3 // (op 557) most significant: low of low * high
v_mul_hi_u32 v14, v1, s2 // (op 557) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 557) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 557) most significant: sum

s_mov_b32 s15, 0 // (op 557) convert
s_mov_b32 s14, s0 // (op 557) convert

v_mov_b32 v14, s14 // (op 557) 
v_mov_b32 v15, s15 // (op 557) 
v_add_co_u32 v18, s[0:1], v14, v12 // (op 557) least significant half
v_addc_co_u32 v19, s[0:1], v15, v13, s[0:1] // (op 557) most significant half





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

s_lshl_b32 s0, s17, 4 // (op 602) 
s_mov_b32 s15, 0 // (op 602) convert
s_mov_b32 s14, s0 // (op 602) convert

v_mul_lo_u32 v13, v1, s9 // (op 602) most significant: low of low * high
v_mul_hi_u32 v14, v1, s8 // (op 602) most significant: high of low * low
v_mul_lo_u32 v12, v1, s8 // (op 602) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 602) most significant: sum

v_mov_b32 v14, s14 // (op 602) 
v_mov_b32 v15, s15 // (op 602) 
v_add_co_u32 v12, s[0:1], v12, v14 // (op 602) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 602) most significant half




s_mul_i32 s1, s8, s7 // (op 604) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 604) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 604) least significant
s_mul_hi_u32 s15, s8, s6 // (op 604) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 604) most significant: sum
s_add_u32 s1, s1, s15 // (op 604) most significant: sum


s_mov_b32 s44, s10 // (op 604) reg expression
s_mov_b32 s45, s11 // (op 604) reg expression
s_mov_b32 s46, s0 // (op 604) reg expression
s_mov_b32 s47, 131072 // (op 604) call()

s_lshl_b32 s0, s17, 4 // (op 621) 
s_mov_b32 s15, 0 // (op 621) convert
s_mov_b32 s14, s0 // (op 621) convert

v_mul_lo_u32 v15, v1, s9 // (op 621) most significant: low of low * high
v_mul_hi_u32 v20, v1, s8 // (op 621) most significant: high of low * low
v_mul_lo_u32 v14, v1, s8 // (op 621) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 621) most significant: sum

v_mov_b32 v20, s14 // (op 621) 
v_mov_b32 v21, s15 // (op 621) 
v_add_co_u32 v14, s[0:1], v14, v20 // (op 621) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 621) most significant half




s_mul_i32 s1, s8, s7 // (op 623) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 623) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 623) least significant
s_mul_hi_u32 s15, s8, s6 // (op 623) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 623) most significant: sum
s_add_u32 s1, s1, s15 // (op 623) most significant: sum


s_mov_b32 s44, s10 // (op 623) reg expression
s_mov_b32 s45, s11 // (op 623) reg expression
s_mov_b32 s46, s0 // (op 623) reg expression
s_mov_b32 s47, 131072 // (op 623) call()

v_and_b32 v20, -16, v1 // (op 666) 
v_sub_u32 v21, v1, v20 // (op 666) 

v_lshl_add_u32 v20, s16, 4, v21 // (op 666) 

v_lshrrev_b32 v21, 4, v1 // (op 666) 
v_mul_lo_u32 v22, v21, s12 // (op 666) 

v_add_u32 v21, v20, v22 // (op 666) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 668) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 668) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 668) least significant
s_mul_hi_u32 s15, s12, s22 // (op 668) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 668) most significant: sum
s_add_u32 s1, s1, s15 // (op 668) most significant: sum


s_mov_b32 s48, s20 // (op 668) reg expression
s_mov_b32 s49, s21 // (op 668) reg expression
s_mov_b32 s50, s0 // (op 668) reg expression
s_mov_b32 s51, 131072 // (op 668) call()

v_and_b32 v20, -16, v1 // (op 685) 
v_sub_u32 v22, v1, v20 // (op 685) 

v_lshl_add_u32 v20, s16, 4, v22 // (op 685) 

v_lshrrev_b32 v22, 4, v1 // (op 685) 
v_mul_lo_u32 v23, v22, s12 // (op 685) 

v_add_u32 v22, v20, v23 // (op 685) 


s_mul_i32 s1, s12, s23 // (op 687) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 687) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 687) least significant
s_mul_hi_u32 s15, s12, s22 // (op 687) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 687) most significant: sum
s_add_u32 s1, s1, s15 // (op 687) most significant: sum


s_mov_b32 s48, s20 // (op 687) reg expression
s_mov_b32 s49, s21 // (op 687) reg expression
s_mov_b32 s50, s0 // (op 687) reg expression
s_mov_b32 s51, 131072 // (op 687) call()

v_and_b32 v20, -16, v1 // (op 730) 
v_sub_u32 v23, v1, v20 // (op 730) 

v_lshl_add_u32 v20, s17, 4, v23 // (op 730) 

v_lshrrev_b32 v23, 4, v1 // (op 730) 
v_mul_lo_u32 v24, v23, s24 // (op 730) 
v_add_u32 v24, v24, v20 // (op 730) 


s_mul_i32 s1, s24, s23 // (op 732) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 732) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 732) least significant
s_mul_hi_u32 s15, s24, s22 // (op 732) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 732) most significant: sum
s_add_u32 s1, s1, s15 // (op 732) most significant: sum


s_mov_b32 s52, s26 // (op 732) reg expression
s_mov_b32 s53, s27 // (op 732) reg expression
s_mov_b32 s54, s0 // (op 732) reg expression
s_mov_b32 s55, 131072 // (op 732) call()

v_and_b32 v20, -16, v1 // (op 749) 
v_sub_u32 v23, v1, v20 // (op 749) 

v_lshl_add_u32 v20, s17, 4, v23 // (op 749) 

v_lshrrev_b32 v23, 4, v1 // (op 749) 
v_mul_lo_u32 v25, v23, s24 // (op 749) 
v_add_u32 v25, v25, v20 // (op 749) 


s_mul_i32 s1, s24, s23 // (op 751) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 751) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 751) least significant
s_mul_hi_u32 s15, s24, s22 // (op 751) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 751) most significant: sum
s_add_u32 s1, s1, s15 // (op 751) most significant: sum


s_mov_b32 s52, s26 // (op 751) reg expression
s_mov_b32 s53, s27 // (op 751) reg expression
s_mov_b32 s54, s0 // (op 751) reg expression
s_mov_b32 s55, 131072 // (op 751) call()

v_lshlrev_b32 v20, 4, v1 // (op 867) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 867) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 6 // (op 487) 
v_mov_b64 v[26:27], s[14:15] // (op 487) convert


v_mov_b64 v[28:29], v[12:13] // (op 487) reg expression
s_mov_b32 m0, s0 // (op 487) reg expression
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 487) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v26 // (op 487) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 487) most significant half

s_add_u32 m0, m0, 1024 // (op 487) 
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 487) Load value direct to lds





v_mov_b32 v20, v24 // (op 369) reg expression

buffer_load_ubyte v23, v20, s[52:55], 0 offen offset: 0 // (op 369) Load value

v_lshlrev_b32 v20, 4, v1 // (op 815) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 815) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 492) 
v_mov_b64 v[26:27], s[14:15] // (op 492) convert


v_mov_b64 v[28:29], v[16:17] // (op 492) reg expression
s_add_u32 s1, 2048, s0 // (op 492) 
s_mov_b32 m0, s1 // (op 492) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 492) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v26 // (op 492) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 492) most significant half

s_add_u32 m0, m0, 1024 // (op 492) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 492) Load value direct to lds






v_mov_b32 v20, v21 // (op 375) reg expression

buffer_load_ubyte v26, v20, s[48:51], 0 offen offset: 0 // (op 375) Load value

v_lshlrev_b32 v20, 4, v1 // (op 1085) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1085) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 6 // (op 497) 
v_mov_b64 v[28:29], s[14:15] // (op 497) convert


s_lshl_b64 s[14:15], s[8:9], 7 // (op 497) 
v_mov_b32 v20, s14 // (op 497) 
v_mov_b32 v27, s15 // (op 497) 
v_add_co_u32 v30, s[18:19], v12, v20 // (op 497) least significant half
v_addc_co_u32 v31, s[18:19], v13, v27, s[18:19] // (op 497) most significant half




s_add_u32 s1, 4096, s0 // (op 497) 
s_mov_b32 m0, s1 // (op 497) reg expression
buffer_load_dwordx4 v30, s[44:47], 0 offen offset: 0 lds // (op 497) Load value direct to lds
v_add_co_u32 v30, s[14:15], v30, v28 // (op 497) least significant half
v_addc_co_u32 v31, s[14:15], 0, v31, s[14:15] // (op 497) most significant half

s_add_u32 m0, m0, 1024 // (op 497) 
buffer_load_dwordx4 v30, s[44:47], 0 offen offset: 0 lds // (op 497) Load value direct to lds







s_lshl_b32 s0, s24, 2 // (op 381) 
v_add_u32 v20, v24, s0 // (op 381) 


buffer_load_ubyte v12, v20, s[52:55], 0 offen offset: 0 // (op 381) Load value


v_lshlrev_b32 v13, 4, v1 // (op 1046) 
s_nop 0
v_readfirstlane_b32 s0, v13 // (op 1046) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 499) 
v_mov_b64 v[28:29], s[14:15] // (op 499) convert


s_lshl_b64 s[14:15], s[2:3], 7 // (op 499) 
v_mov_b32 v13, s14 // (op 499) 
v_mov_b32 v20, s15 // (op 499) 
v_add_co_u32 v30, s[18:19], v16, v13 // (op 499) least significant half
v_addc_co_u32 v31, s[18:19], v17, v20, s[18:19] // (op 499) most significant half




s_add_u32 s1, 6144, s0 // (op 499) 
s_mov_b32 m0, s1 // (op 499) reg expression
buffer_load_dwordx4 v30, s[40:43], 0 offen offset: 0 lds // (op 499) Load value direct to lds
v_add_co_u32 v30, s[14:15], v30, v28 // (op 499) least significant half
v_addc_co_u32 v31, s[14:15], 0, v31, s[14:15] // (op 499) most significant half

s_add_u32 m0, m0, 1024 // (op 499) 
buffer_load_dwordx4 v30, s[40:43], 0 offen offset: 0 lds // (op 499) Load value direct to lds







s_lshl_b32 s0, s12, 2 // (op 387) 
v_add_u32 v13, v21, s0 // (op 387) 


buffer_load_ubyte v20, v13, s[48:51], 0 offen offset: 0 // (op 387) Load value


v_and_b32 v13, -16, v1 // (op 973) 
v_sub_u32 v21, v1, v13 // (op 973) 

v_and_b32 v13, 4294967280, v1 // (op 973) 
v_add_u32 v24, v13, v21 // (op 973) 





s_waitcnt vmcnt(9)
ds_write_b8 v24, v23 offset:8192 // (op 390) Store local data 

v_and_b32 v13, -16, v1 // (op 920) 
v_sub_u32 v21, v1, v13 // (op 920) 

v_and_b32 v13, 4294967280, v1 // (op 920) 
v_add_u32 v24, v13, v21 // (op 920) 





s_waitcnt vmcnt(6)
ds_write_b8 v24, v26 offset:8256 // (op 392) Store local data 

s_barrier  // (op 1203) 



ds_read_b64_tr_b8 v[28:29], v3 // (op 409) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v3 offset:128 // (op 409) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v3 offset:1024 // (op 409) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:1152 // (op 409) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1237) 


ds_read_u8 v13, v7 offset:8192 // (op 415) Load local data 
s_barrier  // (op 1187) 



ds_read_b64_tr_b8 v[36:37], v0 offset:2048 // (op 406) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:2176 // (op 406) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:3072 // (op 406) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:3200 // (op 406) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 1220) 


ds_read_u8 v21, v5 offset:8256 // (op 412) Load local data 
v_lshlrev_b32 v24, 4, v1 // (op 787) 
s_nop 0
v_readfirstlane_b32 s0, v24 // (op 787) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v24, 4, v1 // (op 839) 
s_nop 0
v_readfirstlane_b32 s1, v24 // (op 839) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 891) 
v_sub_u32 v27, v1, v24 // (op 891) 

v_and_b32 v24, 4294967280, v1 // (op 891) 
v_add_u32 v16, v24, v27 // (op 891) 


v_and_b32 v17, -16, v1 // (op 944) 
v_sub_u32 v24, v1, v17 // (op 944) 

v_and_b32 v17, 4294967280, v1 // (op 944) 
v_add_u32 v27, v17, v24 // (op 944) 


v_lshlrev_b32 v17, 4, v1 // (op 1031) 
s_nop 0
v_readfirstlane_b32 s14, v17 // (op 1031) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v17, 4, v1 // (op 1070) 
s_nop 0
v_readfirstlane_b32 s15, v17 // (op 1070) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -16, v1 // (op 1109) 
v_sub_u32 v24, v1, v17 // (op 1109) 

v_and_b32 v17, 4294967280, v1 // (op 1109) 
v_add_u32 v44, v17, v24 // (op 1109) 


v_and_b32 v17, -16, v1 // (op 1133) 
v_sub_u32 v24, v1, v17 // (op 1133) 

v_and_b32 v17, 4294967280, v1 // (op 1133) 
v_add_u32 v45, v17, v24 // (op 1133) 


s_mov_b32 s18, 0 // (op 59) call()
s_ashr_i64 s[38:39], s[6:7], 7 // (op 58) 
s_mov_b32 s19, s38 // (op 58) convert

s_and_b32 s38, s19, 4294967294 // (op 58) 

s_cmp_lt_u32 s18, s38 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_58:
s_add_u32 s18, s18, 2 // (op 60) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[28:35], a[0:3], v21, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 88) 
s_barrier  // (op 1206) 

s_lshl_b64 s[38:39], s[8:9], 6 // (op 459) 
v_mov_b64 v[46:47], s[38:39] // (op 459) convert


v_mov_b64 v[48:49], v[14:15] // (op 459) 
s_lshl_b64 s[38:39], s[8:9], 7 // (op 459) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 459) 

v_mov_b32 v17, s56 // (op 459) 
v_mov_b32 v24, s57 // (op 459) 
v_add_co_u32 v50, s[38:39], v48, v17 // (op 459) least significant half
v_addc_co_u32 v51, s[38:39], v49, v24, s[38:39] // (op 459) most significant half




s_mov_b32 m0, s1 // (op 459) reg expression
buffer_load_dwordx4 v50, s[44:47], 0 offen offset: 0 lds // (op 459) Load value direct to lds
v_add_co_u32 v50, s[38:39], v50, v46 // (op 459) least significant half
v_addc_co_u32 v51, s[38:39], 0, v51, s[38:39] // (op 459) most significant half

s_add_u32 m0, m0, 1024 // (op 459) 
buffer_load_dwordx4 v50, s[44:47], 0 offen offset: 0 lds // (op 459) Load value direct to lds




v_mov_b32 v17, v25 // (op 25) 
s_lshl_b32 s19, s24, 3 // (op 25) 
v_add_u32 v24, v17, s19 // (op 25) 


buffer_load_ubyte v23, v24, s[52:55], 0 offen offset: 0 // (op 25) Load value

s_barrier  // (op 1190) 

s_lshl_b64 s[38:39], s[2:3], 6 // (op 454) 
v_mov_b64 v[46:47], s[38:39] // (op 454) convert


v_mov_b64 v[50:51], v[18:19] // (op 454) 
s_lshl_b64 s[38:39], s[2:3], 7 // (op 454) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 454) 

v_mov_b32 v24, s56 // (op 454) 
v_mov_b32 v52, s57 // (op 454) 
v_add_co_u32 v54, s[38:39], v50, v24 // (op 454) least significant half
v_addc_co_u32 v55, s[38:39], v51, v52, s[38:39] // (op 454) most significant half




s_add_u32 s19, 2048, s0 // (op 454) 
s_mov_b32 m0, s19 // (op 454) reg expression
buffer_load_dwordx4 v54, s[40:43], 0 offen offset: 0 lds // (op 454) Load value direct to lds
v_add_co_u32 v54, s[38:39], v54, v46 // (op 454) least significant half
v_addc_co_u32 v55, s[38:39], 0, v55, s[38:39] // (op 454) most significant half

s_add_u32 m0, m0, 1024 // (op 454) 
buffer_load_dwordx4 v54, s[40:43], 0 offen offset: 0 lds // (op 454) Load value direct to lds





v_mov_b32 v24, v22 // (op 17) 
s_lshl_b32 s19, s12, 3 // (op 17) 
v_add_u32 v46, v24, s19 // (op 17) 


buffer_load_ubyte v26, v46, s[48:51], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1276) 



s_waitcnt vmcnt(9)
ds_write_b8 v45, v12 offset:8320 // (op 255) Store local data 
s_barrier  // (op 1270) 



s_waitcnt vmcnt(6)
ds_write_b8 v44, v20 offset:8384 // (op 254) Store local data 
s_barrier  // (op 1260) 



ds_read_b64_tr_b8 v[28:29], v6 offset:4096 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v6 offset:4224 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v6 offset:5120 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v6 offset:5248 // (op 247) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1273) 


ds_read_u8 v13, v11 offset:8320 // (op 251) Load local data 
s_barrier  // (op 1254) 
s_barrier  // (op 1327) 



ds_read_b64_tr_b8 v[36:37], v4 offset:6144 // (op 245) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:6272 // (op 245) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:7168 // (op 245) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:7296 // (op 245) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 1267) 


ds_read_u8 v21, v9 offset:8384 // (op 249) Load local data 
s_barrier  // (op 1321) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[28:35], a[0:3], v21, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 252) 
s_barrier  // (op 1264) 

s_lshl_b64 s[38:39], s[8:9], 6 // (op 482) 
v_mov_b64 v[46:47], s[38:39] // (op 482) convert


s_lshl_b64 s[38:39], s[8:9], 7 // (op 482) 
s_mul_i32 s57, 3, s39 // (op 482) most significant: low of low * high
s_mul_i32 s56, 3, s38 // (op 482) least significant
s_mul_hi_u32 s19, 3, s38 // (op 482) most significant: high of low * low
s_add_u32 s57, s57, s19 // (op 482) most significant: sum


v_mov_b32 v52, s56 // (op 482) 
v_mov_b32 v53, s57 // (op 482) 
v_add_co_u32 v54, s[38:39], v48, v52 // (op 482) least significant half
v_addc_co_u32 v55, s[38:39], v49, v53, s[38:39] // (op 482) most significant half




s_add_u32 s19, 4096, s15 // (op 482) 
s_mov_b32 m0, s19 // (op 482) reg expression
buffer_load_dwordx4 v54, s[44:47], 0 offen offset: 0 lds // (op 482) Load value direct to lds
v_add_co_u32 v54, s[38:39], v54, v46 // (op 482) least significant half
v_addc_co_u32 v55, s[38:39], 0, v55, s[38:39] // (op 482) most significant half

s_add_u32 m0, m0, 1024 // (op 482) 
buffer_load_dwordx4 v54, s[44:47], 0 offen offset: 0 lds // (op 482) Load value direct to lds






s_lshl_b32 s19, s24, 2 // (op 258) 
s_mul_i32 s38, 3, s19 // (op 258) 

v_add_u32 v46, v17, s38 // (op 258) 


buffer_load_ubyte v12, v46, s[52:55], 0 offen offset: 0 // (op 258) Load value

s_lshl_b64 s[38:39], s[8:9], 7 // (op 613) 

s_lshl_b64 s[56:57], s[38:39], 1 // (op 613) 

s_barrier  // (op 1257) 
s_lshl_b32 s19, s24, 3 // (op 741) 

v_add_u32 v25, v25, s19 // (op 741) 

v_mov_b32 v17, s56 // (op 613) 
v_mov_b32 v46, s57 // (op 613) 
s_lshl_b64 s[38:39], s[2:3], 6 // (op 477) 
v_mov_b64 v[48:49], s[38:39] // (op 477) convert


s_lshl_b64 s[38:39], s[2:3], 7 // (op 477) 
s_mul_i32 s59, 3, s39 // (op 477) most significant: low of low * high
s_mul_i32 s58, 3, s38 // (op 477) least significant
s_mul_hi_u32 s19, 3, s38 // (op 477) most significant: high of low * low
s_add_u32 s59, s59, s19 // (op 477) most significant: sum


v_mov_b32 v47, s58 // (op 477) 
v_mov_b32 v52, s59 // (op 477) 
v_add_co_u32 v14, s[38:39], v14, v17 // (op 613) least significant half
v_addc_co_u32 v15, s[38:39], v15, v46, s[38:39] // (op 613) most significant half




v_add_co_u32 v54, s[38:39], v50, v47 // (op 477) least significant half
v_addc_co_u32 v55, s[38:39], v51, v52, s[38:39] // (op 477) most significant half




s_add_u32 s19, 6144, s14 // (op 477) 
s_mov_b32 m0, s19 // (op 477) reg expression
buffer_load_dwordx4 v54, s[40:43], 0 offen offset: 0 lds // (op 477) Load value direct to lds
v_add_co_u32 v54, s[38:39], v54, v48 // (op 477) least significant half
v_addc_co_u32 v55, s[38:39], 0, v55, s[38:39] // (op 477) most significant half

s_add_u32 m0, m0, 1024 // (op 477) 
buffer_load_dwordx4 v54, s[40:43], 0 offen offset: 0 lds // (op 477) Load value direct to lds






s_lshl_b32 s19, s12, 2 // (op 257) 
s_mul_i32 s38, 3, s19 // (op 257) 

v_add_u32 v17, v24, s38 // (op 257) 


buffer_load_ubyte v20, v17, s[48:51], 0 offen offset: 0 // (op 257) Load value

s_lshl_b64 s[38:39], s[2:3], 7 // (op 549) 

s_lshl_b64 s[56:57], s[38:39], 1 // (op 549) 

s_barrier  // (op 1240) 
s_lshl_b32 s19, s12, 3 // (op 677) 



v_add_u32 v22, v22, s19 // (op 677) 

v_mov_b32 v17, s56 // (op 549) 
v_mov_b32 v24, s57 // (op 549) 
v_add_co_u32 v18, s[38:39], v18, v17 // (op 549) least significant half
v_addc_co_u32 v19, s[38:39], v19, v24, s[38:39] // (op 549) most significant half




s_waitcnt vmcnt(9)
ds_write_b8 v27, v23 offset:8192 // (op 30) Store local data 
s_barrier  // (op 1223) 



s_waitcnt vmcnt(6)
ds_write_b8 v16, v26 offset:8256 // (op 21) Store local data 
s_barrier  // (op 1212) 



ds_read_b64_tr_b8 v[28:29], v3 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v3 offset:128 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v3 offset:1024 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:1152 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1247) 


ds_read_u8 v13, v7 offset:8192 // (op 85) Load local data 
s_barrier  // (op 1196) 



ds_read_b64_tr_b8 v[36:37], v0 offset:2048 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:2176 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:3072 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:3200 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 1230) 


ds_read_u8 v21, v5 offset:8256 // (op 80) Load local data 
s_ashr_i64 s[38:39], s[6:7], 7 // (op 58) 
s_mov_b32 s19, s38 // (op 58) convert

s_and_b32 s38, s19, 4294967294 // (op 58) 

s_cmp_lt_u32 s18, s38 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_58:















s_barrier  // (op 1287) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 512) 
s_lshr_b32 s14, s0, 1 // (op 512) 

v_lshl_add_u32 v14, s14, 8, v1 // (op 512) 

s_lshl_b32 s0, s16, 4 // (op 512) 
v_mul_lo_u32 v19, v14, s3 // (op 512) most significant: low of low * high
v_mul_hi_u32 v15, v14, s2 // (op 512) most significant: high of low * low
v_mul_lo_u32 v18, v14, s2 // (op 512) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 512) most significant: sum


s_mov_b32 s15, 0 // (op 512) convert
s_mov_b32 s14, s0 // (op 512) convert

v_mov_b32 v14, s14 // (op 512) 
v_mov_b32 v15, s15 // (op 512) 
v_add_co_u32 v16, s[0:1], v14, v18 // (op 512) least significant half
v_addc_co_u32 v17, s[0:1], v15, v19, s[0:1] // (op 512) most significant half





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

s_ashr_i64 s[0:1], s[6:7], 7 // (op 579) 
s_lshr_b32 s14, s0, 1 // (op 579) 

v_lshl_add_u32 v14, s14, 8, v1 // (op 579) 

s_lshl_b32 s0, s17, 4 // (op 579) 
s_mov_b32 s15, 0 // (op 579) convert
s_mov_b32 s14, s0 // (op 579) convert

v_mul_lo_u32 v19, v14, s9 // (op 579) most significant: low of low * high
v_mul_hi_u32 v15, v14, s8 // (op 579) most significant: high of low * low
v_mul_lo_u32 v18, v14, s8 // (op 579) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 579) most significant: sum

v_mov_b32 v15, s14 // (op 579) 
v_mov_b32 v22, s15 // (op 579) 
v_add_co_u32 v18, s[0:1], v18, v15 // (op 579) least significant half
v_addc_co_u32 v19, s[0:1], v19, v22, s[0:1] // (op 579) most significant half





s_mul_i32 s1, s8, s7 // (op 581) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 581) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 581) least significant
s_mul_hi_u32 s15, s8, s6 // (op 581) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 581) most significant: sum
s_add_u32 s1, s1, s15 // (op 581) most significant: sum


s_mov_b32 s44, s10 // (op 581) reg expression
s_mov_b32 s45, s11 // (op 581) reg expression
s_mov_b32 s46, s0 // (op 581) reg expression
s_mov_b32 s47, 131072 // (op 581) call()

v_and_b32 v14, -16, v1 // (op 643) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 643) 
v_sub_u32 v15, v1, v14 // (op 643) 

s_lshr_b32 s14, s0, 1 // (op 643) 

v_lshl_add_u32 v14, s16, 4, v15 // (op 643) 

v_lshrrev_b32 v15, 4, v1 // (op 643) 
v_lshl_add_u32 v22, s14, 3, v15 // (op 643) 


v_mul_lo_u32 v15, v22, s12 // (op 643) 

v_add_u32 v22, v14, v15 // (op 643) 


s_mul_i32 s1, s12, s23 // (op 645) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 645) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 645) least significant
s_mul_hi_u32 s15, s12, s22 // (op 645) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 645) most significant: sum
s_add_u32 s1, s1, s15 // (op 645) most significant: sum


s_mov_b32 s48, s20 // (op 645) reg expression
s_mov_b32 s49, s21 // (op 645) reg expression
s_mov_b32 s50, s0 // (op 645) reg expression
s_mov_b32 s51, 131072 // (op 645) call()

v_lshrrev_b32 v14, 4, v1 // (op 707) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 707) 
v_and_b32 v15, -16, v1 // (op 707) 
s_lshr_b32 s14, s0, 1 // (op 707) 

v_sub_u32 v23, v1, v15 // (op 707) 

v_lshl_add_u32 v15, s14, 3, v14 // (op 707) 


v_lshl_add_u32 v14, s17, 4, v23 // (op 707) 

v_mul_lo_u32 v23, v15, s24 // (op 707) 
v_add_u32 v23, v23, v14 // (op 707) 


s_mul_i32 s1, s24, s23 // (op 709) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 709) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 709) least significant
s_mul_hi_u32 s15, s24, s22 // (op 709) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 709) most significant: sum
s_add_u32 s1, s1, s15 // (op 709) most significant: sum


s_mov_b32 s52, s26 // (op 709) reg expression
s_mov_b32 s53, s27 // (op 709) reg expression
s_mov_b32 s54, s0 // (op 709) reg expression
s_mov_b32 s55, 131072 // (op 709) call()

v_lshlrev_b32 v14, 4, v1 // (op 800) 
s_nop 0
v_readfirstlane_b32 s0, v14 // (op 800) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v14, 4, v1 // (op 852) 
s_nop 0
v_readfirstlane_b32 s1, v14 // (op 852) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -16, v1 // (op 904) 
v_sub_u32 v15, v1, v14 // (op 904) 

v_and_b32 v14, 4294967280, v1 // (op 904) 
v_add_u32 v24, v14, v15 // (op 904) 


v_and_b32 v14, -16, v1 // (op 957) 
v_sub_u32 v15, v1, v14 // (op 957) 

v_and_b32 v14, 4294967280, v1 // (op 957) 
v_add_u32 v25, v14, v15 // (op 957) 


s_ashr_i64 s[14:15], s[6:7], 7 // (op 172) 
s_and_b32 s18, s14, 4294967294 // (op 172) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 171) 
s_mov_b32 s19, s14 // (op 171) convert

s_cmp_lt_u32 s18, s19 // (op 171) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_171 // (op 171) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_171 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_171:
s_barrier  // (op 1193) 
s_add_u32 s18, s18, 1 // (op 173) 

s_lshl_b64 s[14:15], s[2:3], 6 // (op 464) 
v_mov_b64 v[20:21], s[14:15] // (op 464) convert


v_mov_b64 v[12:13], v[16:17] // (op 464) 
v_mov_b64 v[14:15], v[12:13] // (op 464) reg expression
s_add_u32 s19, 2048, s0 // (op 464) 
s_mov_b32 m0, s19 // (op 464) reg expression
buffer_load_dwordx4 v14, s[40:43], 0 offen offset: 0 lds // (op 464) Load value direct to lds
v_add_co_u32 v14, s[14:15], v14, v20 // (op 464) least significant half
v_addc_co_u32 v15, s[14:15], 0, v15, s[14:15] // (op 464) most significant half

s_add_u32 m0, m0, 1024 // (op 464) 
buffer_load_dwordx4 v14, s[40:43], 0 offen offset: 0 lds // (op 464) Load value direct to lds




s_lshl_b64 s[14:15], s[2:3], 7 // (op 502) 
v_mov_b32 v12, s14 // (op 502) 
v_mov_b32 v13, s15 // (op 502) 
v_add_co_u32 v16, s[38:39], v16, v12 // (op 502) least significant half
v_addc_co_u32 v17, s[38:39], v17, v13, s[38:39] // (op 502) most significant half




s_barrier  // (op 1209) 

s_lshl_b64 s[14:15], s[8:9], 6 // (op 470) 
v_mov_b64 v[20:21], s[14:15] // (op 470) convert


v_mov_b64 v[12:13], v[18:19] // (op 470) 
v_mov_b64 v[14:15], v[12:13] // (op 470) reg expression
s_mov_b32 m0, s1 // (op 470) reg expression
buffer_load_dwordx4 v14, s[44:47], 0 offen offset: 0 lds // (op 470) Load value direct to lds
v_add_co_u32 v14, s[14:15], v14, v20 // (op 470) least significant half
v_addc_co_u32 v15, s[14:15], 0, v15, s[14:15] // (op 470) most significant half

s_add_u32 m0, m0, 1024 // (op 470) 
buffer_load_dwordx4 v14, s[44:47], 0 offen offset: 0 lds // (op 470) Load value direct to lds



s_lshl_b64 s[14:15], s[8:9], 7 // (op 571) 
v_mov_b32 v12, s14 // (op 571) 
v_mov_b32 v13, s15 // (op 571) 
v_add_co_u32 v18, s[38:39], v18, v12 // (op 571) least significant half
v_addc_co_u32 v19, s[38:39], v19, v13, s[38:39] // (op 571) most significant half






v_mov_b32 v12, v22 // (op 191) 
v_mov_b32 v13, v12 // (op 191) reg expression

buffer_load_ubyte v14, v13, s[48:51], 0 offen offset: 0 // (op 191) Load value


s_lshl_b32 s19, s12, 2 // (op 635) 
v_add_u32 v22, v22, s19 // (op 635) 



s_barrier  // (op 1226) 
v_mov_b32 v15, v23 // (op 192) 



v_mov_b32 v12, v15 // (op 192) reg expression

buffer_load_ubyte v13, v12, s[52:55], 0 offen offset: 0 // (op 192) Load value


s_lshl_b32 s19, s24, 2 // (op 699) 
v_add_u32 v23, v23, s19 // (op 699) 

s_waitcnt vmcnt(1)
ds_write_b8 v24, v14 offset:8256 // (op 188) Store local data 

s_barrier  // (op 1199) 



s_barrier  // (op 1243) 
ds_read_b64_tr_b8 v[26:27], v0 offset:2048 // (op 179) Transpose load local data 



ds_read_b64_tr_b8 v[28:29], v0 offset:2176 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:3072 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:3200 // (op 179) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v25, v13 offset:8192 // (op 189) Store local data 

s_barrier  // (op 1216) 



ds_read_b64_tr_b8 v[34:35], v3 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v3 offset:128 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v3 offset:1024 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v3 offset:1152 // (op 181) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 1233) 


ds_read_u8 v12, v5 offset:8256 // (op 183) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1250) 


ds_read_u8 v13, v7 offset:8192 // (op 185) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 186) 




s_ashr_i64 s[14:15], s[6:7], 7 // (op 171) 
s_mov_b32 s19, s14 // (op 171) convert

s_cmp_lt_u32 s18, s19 // (op 171) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_171 // (op 171) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_171 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_171:























s_barrier  // (op 1313) 








v_cmp_eq_f32 vcc, 0.00000, v8 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_133
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v10, v4 // (op 50) 
v_mul_f32 v13, v10, v5 // (op 50) 
v_mul_f32 v14, v10, v6 // (op 50) 
v_mul_f32 v15, v10, v7 // (op 50) 

v_lshrrev_b32 v0, 6, v1 // (op 999) 
v_and_b32 v9, 63, v1 // (op 999) 
v_lshrrev_b32 v11, 4, v9 // (op 999) 
v_lshlrev_b32 v3, 2, v11 // (op 999) 

v_lshl_add_u32 v11, v0, 4, v3 // (op 999) 


v_lshl_add_u32 v0, s16, 4, v11 // (op 999) 

v_lshrrev_b32 v11, 2, v9 // (op 999) 

v_and_b32 v9, 3, v11 // (op 999) 

v_and_b32 v11, 3, v1 // (op 999) 
v_lshl_add_u32 v3, v9, 2, v11 // (op 999) 


v_lshl_add_u32 v9, s17, 4, v3 // (op 999) 

v_mul_lo_u32 v11, v9, s28 // (op 999) 

v_add_lshl_u32 v9, v0, v11, 2 // (op 999) 


s_mul_i32 s1, s28, s33 // (op 1001) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 1001) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1001) least significant
s_mul_hi_u32 s3, s28, s32 // (op 1001) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1001) most significant: sum
s_add_u32 s1, s1, s3 // (op 1001) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 1001) 

s_mov_b32 s4, s30 // (op 1001) reg expression
s_mov_b32 s5, s31 // (op 1001) reg expression
s_mov_b32 s6, s2 // (op 1001) reg expression
s_mov_b32 s7, 131072 // (op 1001) call()




buffer_store_dwordx4 v[12:15], v9, s[4:7], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_133:
v_lshrrev_b32 v0, 6, v1 // (op 770) 
v_and_b32 v9, 63, v1 // (op 770) 
v_lshrrev_b32 v3, 4, v9 // (op 770) 
v_lshlrev_b32 v4, 2, v3 // (op 770) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 770) 


v_lshl_add_u32 v0, s16, 4, v3 // (op 770) 

v_lshrrev_b32 v3, 2, v9 // (op 770) 

v_and_b32 v9, 3, v3 // (op 770) 

v_and_b32 v3, 3, v1 // (op 770) 
v_lshl_add_u32 v4, v9, 2, v3 // (op 770) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 770) 

v_mul_lo_u32 v9, v3, s34 // (op 770) 

v_add_lshl_u32 v3, v0, v9, 2 // (op 770) 


s_mul_i32 s1, s34, s33 // (op 772) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 772) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 772) least significant
s_mul_hi_u32 s3, s34, s32 // (op 772) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 772) most significant: sum
s_add_u32 s1, s1, s3 // (op 772) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 772) 

s_mov_b32 s8, s36 // (op 772) reg expression
s_mov_b32 s9, s37 // (op 772) reg expression
s_mov_b32 s10, s2 // (op 772) reg expression
s_mov_b32 s11, 131072 // (op 772) call()





buffer_load_dwordx4 v[4:7], v3, s[8:11], 0 offen // (op 145) Load value



v_accvgpr_read v12, a0 // (op 142) 
v_accvgpr_read v13, a1 // (op 142) 
v_accvgpr_read v14, a2 // (op 142) 
v_accvgpr_read v15, a3 // (op 142) 
v_mul_f32 v16, v10, v12 // (op 142) 
v_mul_f32 v17, v10, v13 // (op 142) 
v_mul_f32 v18, v10, v14 // (op 142) 
v_mul_f32 v19, v10, v15 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v8, v4 // (op 146) 
v_mul_f32 v13, v8, v5 // (op 146) 
v_mul_f32 v14, v8, v6 // (op 146) 
v_mul_f32 v15, v8, v7 // (op 146) 

v_add_f32 v4, v12, v16 // (op 143) 
v_add_f32 v5, v13, v17 // (op 143) 
v_add_f32 v6, v14, v18 // (op 143) 
v_add_f32 v7, v15, v19 // (op 143) 


v_lshrrev_b32 v0, 6, v1 // (op 1016) 
v_and_b32 v3, 63, v1 // (op 1016) 
v_lshrrev_b32 v9, 4, v3 // (op 1016) 
v_lshlrev_b32 v11, 2, v9 // (op 1016) 

v_lshl_add_u32 v9, v0, 4, v11 // (op 1016) 


v_lshl_add_u32 v0, s16, 4, v9 // (op 1016) 

v_lshrrev_b32 v9, 2, v3 // (op 1016) 

v_and_b32 v3, 3, v9 // (op 1016) 

v_and_b32 v9, 3, v1 // (op 1016) 
v_lshl_add_u32 v11, v3, 2, v9 // (op 1016) 


v_lshl_add_u32 v3, s17, 4, v11 // (op 1016) 

v_mul_lo_u32 v9, v3, s28 // (op 1016) 

v_add_lshl_u32 v3, v0, v9, 2 // (op 1016) 


s_mul_i32 s1, s28, s33 // (op 1018) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 1018) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1018) least significant
s_mul_hi_u32 s3, s28, s32 // (op 1018) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1018) most significant: sum
s_add_u32 s1, s1, s3 // (op 1018) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 1018) 

s_mov_b32 s4, s30 // (op 1018) reg expression
s_mov_b32 s5, s31 // (op 1018) reg expression
s_mov_b32 s6, s2 // (op 1018) reg expression
s_mov_b32 s7, 131072 // (op 1018) call()




buffer_store_dwordx4 v[4:7], v3, s[4:7], 0 offen // (op 144) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_133:



s_endpgm  // End of RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.LRR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_end:
.size RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2, .LRR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_end-RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
  .amdhsa_next_free_vgpr 60
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 8448 // lds bytes
  .amdhsa_accum_offset 56
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
    args: [ { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC } ]
    macrotile: [16, 16, 128]
    threads: [64, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
    .symbol: RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 8448
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 56
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 8
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 32
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_5
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
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_11
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

