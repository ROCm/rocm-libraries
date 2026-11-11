// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2,@function
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 763) 
v_and_b32 v3, 63, v1 // (op 763) 
v_sub_u32 v4, v3, v0 // (op 763) 

v_lshrrev_b32 v3, 1, v4 // (op 763) 
v_add_lshl_u32 v5, v0, v3, 5 // (op 763) 


v_and_b32 v0, 1, v4 // (op 763) 

v_lshlrev_b32 v3, 3, v0 // (op 763) 

v_lshrrev_b32 v0, 7, v1 // (op 763) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 763) 


v_add_u32 v0, v5, v4 // (op 763) 


v_and_b32 v3, 48, v1 // (op 815) 
v_and_b32 v4, 63, v1 // (op 815) 
v_sub_u32 v5, v4, v3 // (op 815) 

v_lshrrev_b32 v4, 1, v5 // (op 815) 
v_add_lshl_u32 v6, v3, v4, 5 // (op 815) 


v_and_b32 v3, 1, v5 // (op 815) 

v_lshlrev_b32 v4, 3, v3 // (op 815) 

v_lshrrev_b32 v3, 6, v1 // (op 815) 
v_and_b32 v5, 1, v3 // (op 815) 

v_lshl_add_u32 v3, v5, 4, v4 // (op 815) 


v_add_u32 v4, v6, v3 // (op 815) 


v_and_b32 v3, 48, v1 // (op 888) 
v_and_b32 v5, 63, v1 // (op 888) 
v_sub_u32 v6, v5, v3 // (op 888) 

v_lshrrev_b32 v5, 1, v6 // (op 888) 
v_add_lshl_u32 v7, v3, v5, 5 // (op 888) 


v_and_b32 v3, 1, v6 // (op 888) 

v_lshlrev_b32 v5, 3, v3 // (op 888) 

v_lshrrev_b32 v3, 7, v1 // (op 888) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 888) 


v_add_u32 v3, v7, v6 // (op 888) 


v_and_b32 v5, 48, v1 // (op 927) 
v_and_b32 v6, 63, v1 // (op 927) 
v_sub_u32 v7, v6, v5 // (op 927) 

v_lshrrev_b32 v6, 1, v7 // (op 927) 
v_add_lshl_u32 v8, v5, v6, 5 // (op 927) 


v_and_b32 v5, 1, v7 // (op 927) 

v_lshlrev_b32 v6, 3, v5 // (op 927) 

v_lshrrev_b32 v5, 6, v1 // (op 927) 
v_and_b32 v7, 1, v5 // (op 927) 

v_lshl_add_u32 v5, v7, 4, v6 // (op 927) 


v_add_u32 v6, v8, v5 // (op 927) 


v_mov_b32 v5, s14 // (op 2) Move value
v_mov_b32 v7, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
v_and_b32 v8, -2, v1 // (op 474) 
v_sub_u32 v9, v1, v8 // (op 474) 

v_lshlrev_b32 v8, 4, v9 // (op 474) 

v_lshl_add_u32 v9, s16, 5, v8 // (op 474) 

v_mov_b32 v11, 0 // (op 474) convert
v_mov_b32 v10, v9 // (op 474) convert

v_lshrrev_b32 v8, 1, v1 // (op 474) 
v_mul_lo_u32 v13, v8, s3 // (op 474) most significant: low of low * high
v_mul_hi_u32 v9, v8, s2 // (op 474) most significant: high of low * low
v_mul_lo_u32 v12, v8, s2 // (op 474) least significant: low of low * low
v_add_u32 v13, v13, v9 // (op 474) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v12 // (op 474) least significant half
v_addc_co_u32 v9, s[0:1], v11, v13, s[0:1] // (op 474) most significant half



s_mul_i32 s1, s2, s7 // (op 476) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 476) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 476) least significant
s_mul_hi_u32 s15, s2, s6 // (op 476) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 476) most significant: sum
s_add_u32 s1, s1, s15 // (op 476) most significant: sum


s_mov_b32 s40, s4 // (op 476) reg expression
s_mov_b32 s41, s5 // (op 476) reg expression
s_mov_b32 s42, s0 // (op 476) reg expression
s_mov_b32 s43, 131072 // (op 476) call()

v_and_b32 v10, -2, v1 // (op 493) 
v_sub_u32 v11, v1, v10 // (op 493) 

v_lshlrev_b32 v10, 4, v11 // (op 493) 

v_lshl_add_u32 v11, s16, 5, v10 // (op 493) 

v_mov_b32 v13, 0 // (op 493) convert
v_mov_b32 v12, v11 // (op 493) convert

v_lshrrev_b32 v10, 1, v1 // (op 493) 
v_mul_lo_u32 v15, v10, s3 // (op 493) most significant: low of low * high
v_mul_hi_u32 v11, v10, s2 // (op 493) most significant: high of low * low
v_mul_lo_u32 v14, v10, s2 // (op 493) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 493) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 493) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 493) most significant half



s_mul_i32 s1, s2, s7 // (op 495) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 495) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 495) least significant
s_mul_hi_u32 s15, s2, s6 // (op 495) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 495) most significant: sum
s_add_u32 s1, s1, s15 // (op 495) most significant: sum


s_mov_b32 s40, s4 // (op 495) reg expression
s_mov_b32 s41, s5 // (op 495) reg expression
s_mov_b32 s42, s0 // (op 495) reg expression
s_mov_b32 s43, 131072 // (op 495) call()

v_and_b32 v12, -2, v1 // (op 538) 
v_sub_u32 v13, v1, v12 // (op 538) 

v_lshlrev_b32 v12, 4, v13 // (op 538) 

v_lshl_add_u32 v13, s17, 5, v12 // (op 538) 

v_mov_b32 v15, 0 // (op 538) convert
v_mov_b32 v14, v13 // (op 538) convert

v_lshrrev_b32 v12, 1, v1 // (op 538) 
v_mul_lo_u32 v17, v12, s9 // (op 538) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 538) most significant: high of low * low
v_mul_lo_u32 v16, v12, s8 // (op 538) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 538) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 538) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 538) most significant half



s_mul_i32 s1, s8, s7 // (op 540) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 540) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 540) least significant
s_mul_hi_u32 s15, s8, s6 // (op 540) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 540) most significant: sum
s_add_u32 s1, s1, s15 // (op 540) most significant: sum


s_mov_b32 s44, s10 // (op 540) reg expression
s_mov_b32 s45, s11 // (op 540) reg expression
s_mov_b32 s46, s0 // (op 540) reg expression
s_mov_b32 s47, 131072 // (op 540) call()

v_and_b32 v12, -2, v1 // (op 557) 
v_sub_u32 v13, v1, v12 // (op 557) 

v_lshlrev_b32 v12, 4, v13 // (op 557) 

v_lshl_add_u32 v13, s17, 5, v12 // (op 557) 

v_mov_b32 v15, 0 // (op 557) convert
v_mov_b32 v14, v13 // (op 557) convert

v_lshrrev_b32 v12, 1, v1 // (op 557) 
v_mul_lo_u32 v19, v12, s9 // (op 557) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 557) most significant: high of low * low
v_mul_lo_u32 v18, v12, s8 // (op 557) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 557) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 557) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 557) most significant half



s_mul_i32 s1, s8, s7 // (op 559) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 559) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 559) least significant
s_mul_hi_u32 s15, s8, s6 // (op 559) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 559) most significant: sum
s_add_u32 s1, s1, s15 // (op 559) most significant: sum


s_mov_b32 s44, s10 // (op 559) reg expression
s_mov_b32 s45, s11 // (op 559) reg expression
s_mov_b32 s46, s0 // (op 559) reg expression
s_mov_b32 s47, 131072 // (op 559) call()

v_lshrrev_b32 v12, 7, v1 // (op 602) 
v_and_b32 v13, 15, v1 // (op 602) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 602) 


v_lshl_add_u32 v15, s16, 5, v14 // (op 602) 

v_and_b32 v12, 63, v1 // (op 602) 
v_lshrrev_b32 v13, 4, v12 // (op 602) 

v_mul_lo_u32 v12, v13, s12 // (op 602) 

v_add_u32 v13, v15, v12 // (op 602) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 604) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 604) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 604) least significant
s_mul_hi_u32 s15, s12, s22 // (op 604) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 604) most significant: sum
s_add_u32 s1, s1, s15 // (op 604) most significant: sum


s_mov_b32 s48, s20 // (op 604) reg expression
s_mov_b32 s49, s21 // (op 604) reg expression
s_mov_b32 s50, s0 // (op 604) reg expression
s_mov_b32 s51, 131072 // (op 604) call()

v_lshrrev_b32 v12, 7, v1 // (op 621) 
v_and_b32 v14, 15, v1 // (op 621) 
v_lshl_add_u32 v15, v12, 4, v14 // (op 621) 


v_lshl_add_u32 v12, s16, 5, v15 // (op 621) 

v_and_b32 v14, 63, v1 // (op 621) 
v_lshrrev_b32 v15, 4, v14 // (op 621) 

v_mul_lo_u32 v14, v15, s12 // (op 621) 

v_add_u32 v15, v12, v14 // (op 621) 


s_mul_i32 s1, s12, s23 // (op 623) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 623) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 623) least significant
s_mul_hi_u32 s15, s12, s22 // (op 623) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 623) most significant: sum
s_add_u32 s1, s1, s15 // (op 623) most significant: sum


s_mov_b32 s48, s20 // (op 623) reg expression
s_mov_b32 s49, s21 // (op 623) reg expression
s_mov_b32 s50, s0 // (op 623) reg expression
s_mov_b32 s51, 131072 // (op 623) call()

v_lshrrev_b32 v12, 6, v1 // (op 666) 
v_and_b32 v14, 1, v12 // (op 666) 

v_and_b32 v12, 15, v1 // (op 666) 
v_lshl_add_u32 v20, v14, 4, v12 // (op 666) 


v_lshl_add_u32 v12, s17, 5, v20 // (op 666) 

v_and_b32 v14, 63, v1 // (op 666) 
v_lshrrev_b32 v20, 4, v14 // (op 666) 

v_mul_lo_u32 v14, v20, s24 // (op 666) 
v_add_u32 v14, v14, v12 // (op 666) 


s_mul_i32 s1, s24, s23 // (op 668) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 668) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 668) least significant
s_mul_hi_u32 s15, s24, s22 // (op 668) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 668) most significant: sum
s_add_u32 s1, s1, s15 // (op 668) most significant: sum


s_mov_b32 s52, s26 // (op 668) reg expression
s_mov_b32 s53, s27 // (op 668) reg expression
s_mov_b32 s54, s0 // (op 668) reg expression
s_mov_b32 s55, 131072 // (op 668) call()

v_lshrrev_b32 v12, 6, v1 // (op 685) 
v_and_b32 v20, 1, v12 // (op 685) 

v_and_b32 v12, 15, v1 // (op 685) 
v_lshl_add_u32 v21, v20, 4, v12 // (op 685) 


v_lshl_add_u32 v12, s17, 5, v21 // (op 685) 

v_and_b32 v20, 63, v1 // (op 685) 
v_lshrrev_b32 v21, 4, v20 // (op 685) 

v_mul_lo_u32 v20, v21, s24 // (op 685) 
v_add_u32 v20, v20, v12 // (op 685) 


s_mul_i32 s1, s24, s23 // (op 687) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 687) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 687) least significant
s_mul_hi_u32 s15, s24, s22 // (op 687) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 687) most significant: sum
s_add_u32 s1, s1, s15 // (op 687) most significant: sum


s_mov_b32 s52, s26 // (op 687) reg expression
s_mov_b32 s53, s27 // (op 687) reg expression
s_mov_b32 s54, s0 // (op 687) reg expression
s_mov_b32 s55, 131072 // (op 687) call()


v_mov_b64 v[22:23], s[12:13] // (op 312) convert

v_mov_b32 v12, v13 // (op 312) reg expression
v_mov_b32 v21, v12 // (op 312) 
buffer_load_ubyte v24, v21, s[48:51], 0 offen offset: 0 // (op 312) Load value




v_mov_b64 v[22:23], s[24:25] // (op 317) convert

v_mov_b32 v12, v14 // (op 317) reg expression
v_mov_b32 v21, v12 // (op 317) 
buffer_load_ubyte v25, v21, s[52:55], 0 offen offset: 0 // (op 317) Load value



v_and_b32 v12, -2, v1 // (op 803) 
v_sub_u32 v21, v1, v12 // (op 803) 

v_lshlrev_b32 v12, 4, v21 // (op 803) 

v_lshrrev_b32 v21, 1, v1 // (op 803) 
v_lshl_add_u32 v22, v21, 5, v12 // (op 803) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 803) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[22:23], v[16:17] // (op 423) reg expression

s_nop 0
s_mov_b32 m0, s0 // (op 423) reg expression
buffer_load_dwordx4 v22, s[44:47], 0 offen offset: 0 lds // (op 423) Load value direct to lds


v_and_b32 v12, -2, v1 // (op 751) 
v_sub_u32 v21, v1, v12 // (op 751) 

v_lshlrev_b32 v12, 4, v21 // (op 751) 

v_lshrrev_b32 v21, 1, v1 // (op 751) 
v_lshl_add_u32 v22, v21, 5, v12 // (op 751) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 751) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[22:23], v[8:9] // (op 428) reg expression
s_nop 0
s_add_u32 s1, 4096, s0 // (op 428) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 428) reg expression
buffer_load_dwordx4 v22, s[40:43], 0 offen offset: 0 lds // (op 428) Load value direct to lds




v_mov_b64 v[22:23], s[12:13] // (op 328) convert

s_lshl_b32 s0, s12, 2 // (op 328) 
v_add_u32 v12, v13, s0 // (op 328) 

v_mov_b32 v21, v12 // (op 328) 
buffer_load_ubyte v26, v21, s[48:51], 0 offen offset: 0 // (op 328) Load value





v_mov_b64 v[12:13], s[24:25] // (op 333) convert

s_lshl_b32 s0, s24, 2 // (op 333) 
v_add_u32 v21, v14, s0 // (op 333) 

v_mov_b32 v22, v21 // (op 333) 
buffer_load_ubyte v23, v22, s[52:55], 0 offen offset: 0 // (op 333) Load value




v_and_b32 v12, -2, v1 // (op 915) 
v_sub_u32 v13, v1, v12 // (op 915) 

v_lshlrev_b32 v12, 4, v13 // (op 915) 

v_lshrrev_b32 v13, 1, v1 // (op 915) 
v_lshl_add_u32 v14, v13, 5, v12 // (op 915) 


s_nop 0
v_readfirstlane_b32 s0, v14 // (op 915) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[14:15], s[8:9], 7 // (op 433) 
v_mov_b32 v12, s14 // (op 433) 
v_mov_b32 v13, s15 // (op 433) 
v_add_co_u32 v28, s[18:19], v16, v12 // (op 433) least significant half
v_addc_co_u32 v29, s[18:19], v17, v13, s[18:19] // (op 433) most significant half




s_add_u32 s1, 8192, s0 // (op 433) 

s_mov_b32 m0, s1 // (op 433) reg expression
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 433) Load value direct to lds




v_and_b32 v12, -2, v1 // (op 876) 
v_sub_u32 v13, v1, v12 // (op 876) 

v_lshlrev_b32 v12, 4, v13 // (op 876) 

v_lshrrev_b32 v13, 1, v1 // (op 876) 
v_lshl_add_u32 v14, v13, 5, v12 // (op 876) 


s_nop 0
v_readfirstlane_b32 s0, v14 // (op 876) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[14:15], s[2:3], 7 // (op 435) 
v_mov_b32 v12, s14 // (op 435) 
v_mov_b32 v13, s15 // (op 435) 
v_add_co_u32 v16, s[18:19], v8, v12 // (op 435) least significant half
v_addc_co_u32 v17, s[18:19], v9, v13, s[18:19] // (op 435) most significant half




s_add_u32 s1, 12288, s0 // (op 435) 

s_mov_b32 m0, s1 // (op 435) reg expression
buffer_load_dwordx4 v16, s[40:43], 0 offen offset: 0 lds // (op 435) Load value direct to lds




s_waitcnt vmcnt(5)
s_barrier  // (op 972) 



ds_read_b64_tr_b8 v[28:29], v4 // (op 358) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v4 offset:256 // (op 358) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:2048 // (op 358) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:2304 // (op 358) Transpose load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 956) 



ds_read_b64_tr_b8 v[36:37], v0 offset:4096 // (op 355) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:4352 // (op 355) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:6144 // (op 355) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:6400 // (op 355) Transpose load local data 
v_and_b32 v27, -2, v1 // (op 723) 
v_sub_u32 v8, v1, v27 // (op 723) 

v_lshlrev_b32 v9, 4, v8 // (op 723) 

v_lshrrev_b32 v8, 1, v1 // (op 723) 
v_lshl_add_u32 v27, v8, 5, v9 // (op 723) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 723) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -2, v1 // (op 775) 
v_sub_u32 v8, v1, v27 // (op 775) 

v_lshlrev_b32 v9, 4, v8 // (op 775) 

v_lshrrev_b32 v8, 1, v1 // (op 775) 
v_lshl_add_u32 v27, v8, 5, v9 // (op 775) 


s_nop 0
v_readfirstlane_b32 s1, v27 // (op 775) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -2, v1 // (op 861) 
v_sub_u32 v8, v1, v27 // (op 861) 

v_lshlrev_b32 v9, 4, v8 // (op 861) 

v_lshrrev_b32 v8, 1, v1 // (op 861) 
v_lshl_add_u32 v27, v8, 5, v9 // (op 861) 


s_nop 0
v_readfirstlane_b32 s14, v27 // (op 861) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -2, v1 // (op 900) 
v_sub_u32 v8, v1, v27 // (op 900) 

v_lshlrev_b32 v9, 4, v8 // (op 900) 

v_lshrrev_b32 v8, 1, v1 // (op 900) 
v_lshl_add_u32 v27, v8, 5, v9 // (op 900) 


s_nop 0
v_readfirstlane_b32 s15, v27 // (op 900) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s18, 0 // (op 53) call()
s_ashr_i64 s[38:39], s[6:7], 7 // (op 52) 
s_mov_b32 s19, s38 // (op 52) convert

s_and_b32 s38, s19, 4294967294 // (op 52) 

s_cmp_lt_u32 s18, s38 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52 // (op 52) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52 if false)
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52:
s_add_u32 s18, s18, 2 // (op 54) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[28:35], a[0:3], v24, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 78) 

v_mov_b64 v[8:9], s[12:13] // (op 72) convert

v_mov_b32 v27, v15 // (op 72) 
s_lshl_b32 s19, s12, 3 // (op 72) 
v_add_u32 v12, v27, s19 // (op 72) 

v_mov_b32 v13, v12 // (op 72) 
buffer_load_ubyte v24, v13, s[48:51], 0 offen offset: 0 // (op 72) Load value




v_mov_b64 v[8:9], s[24:25] // (op 76) convert

v_mov_b32 v12, v20 // (op 76) 
s_lshl_b32 s19, s24, 3 // (op 76) 
v_add_u32 v13, v12, s19 // (op 76) 

v_mov_b32 v14, v13 // (op 76) 
buffer_load_ubyte v25, v14, s[52:55], 0 offen offset: 0 // (op 76) Load value



s_barrier  // (op 975) 


v_mov_b64 v[8:9], v[18:19] // (op 396) 
s_lshl_b64 s[38:39], s[8:9], 7 // (op 396) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 396) 

v_mov_b32 v13, s56 // (op 396) 
v_mov_b32 v14, s57 // (op 396) 
v_add_co_u32 v16, s[38:39], v8, v13 // (op 396) least significant half
v_addc_co_u32 v17, s[38:39], v9, v14, s[38:39] // (op 396) most significant half





s_mov_b32 m0, s1 // (op 396) reg expression
buffer_load_dwordx4 v16, s[44:47], 0 offen offset: 0 lds // (op 396) Load value direct to lds

s_barrier  // (op 959) 


v_mov_b64 v[16:17], v[10:11] // (op 391) 
s_lshl_b64 s[38:39], s[2:3], 7 // (op 391) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 391) 

v_mov_b32 v13, s56 // (op 391) 
v_mov_b32 v14, s57 // (op 391) 
v_add_co_u32 v44, s[38:39], v16, v13 // (op 391) least significant half
v_addc_co_u32 v45, s[38:39], v17, v14, s[38:39] // (op 391) most significant half




s_add_u32 s19, 4096, s0 // (op 391) 

s_mov_b32 m0, s19 // (op 391) reg expression
buffer_load_dwordx4 v44, s[40:43], 0 offen offset: 0 lds // (op 391) Load value direct to lds


s_waitcnt vmcnt(5)
s_barrier  // (op 994) 



ds_read_b64_tr_b8 v[28:29], v6 offset:8192 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v6 offset:8448 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v6 offset:10240 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v6 offset:10496 // (op 216) Transpose load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 988) 



ds_read_b64_tr_b8 v[36:37], v3 offset:12288 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v3 offset:12544 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v3 offset:14336 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v3 offset:14592 // (op 214) Transpose load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[28:35], a[0:3], v26, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 221) 

v_mov_b64 v[44:45], s[12:13] // (op 218) convert

s_lshl_b32 s19, s12, 2 // (op 218) 
s_mul_i32 s38, 3, s19 // (op 218) 

v_add_u32 v13, v27, s38 // (op 218) 

v_mov_b32 v14, v13 // (op 218) 
buffer_load_ubyte v26, v14, s[48:51], 0 offen offset: 0 // (op 218) Load value





s_lshl_b32 s19, s12, 3 // (op 613) 
v_add_u32 v15, v15, s19 // (op 613) 

v_mov_b64 v[44:45], s[24:25] // (op 220) convert

s_lshl_b32 s19, s24, 2 // (op 220) 
s_mul_i32 s38, 3, s19 // (op 220) 

v_add_u32 v27, v12, s38 // (op 220) 

v_mov_b32 v13, v27 // (op 220) 
buffer_load_ubyte v23, v13, s[52:55], 0 offen offset: 0 // (op 220) Load value




s_barrier  // (op 997) 
s_lshl_b32 s19, s24, 3 // (op 677) 


v_add_u32 v20, v20, s19 // (op 677) 

s_lshl_b64 s[38:39], s[8:9], 7 // (op 418) 
s_mul_i32 s57, 3, s39 // (op 418) most significant: low of low * high
s_mul_i32 s56, 3, s38 // (op 418) least significant
s_mul_hi_u32 s19, 3, s38 // (op 418) most significant: high of low * low
s_add_u32 s57, s57, s19 // (op 418) most significant: sum


v_mov_b32 v27, s56 // (op 418) 
v_mov_b32 v12, s57 // (op 418) 
v_add_co_u32 v44, s[38:39], v8, v27 // (op 418) least significant half
v_addc_co_u32 v45, s[38:39], v9, v12, s[38:39] // (op 418) most significant half




s_add_u32 s19, 8192, s15 // (op 418) 

s_mov_b32 m0, s19 // (op 418) reg expression
buffer_load_dwordx4 v44, s[44:47], 0 offen offset: 0 lds // (op 418) Load value direct to lds



s_barrier  // (op 991) 
s_lshl_b64 s[38:39], s[8:9], 7 // (op 549) 


s_lshl_b64 s[56:57], s[38:39], 1 // (op 549) 

v_mov_b32 v27, s56 // (op 549) 
v_mov_b32 v8, s57 // (op 549) 
s_lshl_b64 s[38:39], s[2:3], 7 // (op 413) 
s_mul_i32 s59, 3, s39 // (op 413) most significant: low of low * high
s_mul_i32 s58, 3, s38 // (op 413) least significant
s_mul_hi_u32 s19, 3, s38 // (op 413) most significant: high of low * low
s_add_u32 s59, s59, s19 // (op 413) most significant: sum


v_mov_b32 v9, s58 // (op 413) 
v_mov_b32 v12, s59 // (op 413) 
v_add_co_u32 v18, s[38:39], v18, v27 // (op 549) least significant half
v_addc_co_u32 v19, s[38:39], v19, v8, s[38:39] // (op 549) most significant half




v_add_co_u32 v44, s[38:39], v16, v9 // (op 413) least significant half
v_addc_co_u32 v45, s[38:39], v17, v12, s[38:39] // (op 413) most significant half




s_add_u32 s19, 12288, s14 // (op 413) 

s_mov_b32 m0, s19 // (op 413) reg expression
buffer_load_dwordx4 v44, s[40:43], 0 offen offset: 0 lds // (op 413) Load value direct to lds



s_lshl_b64 s[38:39], s[2:3], 7 // (op 485) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 485) 

v_mov_b32 v27, s56 // (op 485) 
v_mov_b32 v8, s57 // (op 485) 
v_add_co_u32 v10, s[38:39], v10, v27 // (op 485) least significant half
v_addc_co_u32 v11, s[38:39], v11, v8, s[38:39] // (op 485) most significant half




s_waitcnt vmcnt(5)
s_barrier  // (op 981) 



ds_read_b64_tr_b8 v[28:29], v4 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v4 offset:256 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:2048 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:2304 // (op 67) Transpose load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 965) 



ds_read_b64_tr_b8 v[36:37], v0 offset:4096 // (op 62) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:4352 // (op 62) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:6144 // (op 62) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:6400 // (op 62) Transpose load local data 
s_ashr_i64 s[38:39], s[6:7], 7 // (op 52) 
s_mov_b32 s19, s38 // (op 52) convert

s_and_b32 s38, s19, 4294967294 // (op 52) 

s_cmp_lt_u32 s18, s38 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52 // (op 52) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52 if true)
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52:











s_barrier  // (op 1006) 
v_and_b32 v27, -2, v1 // (op 448) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 448) 
v_sub_u32 v8, v1, v27 // (op 448) 

s_lshr_b32 s14, s0, 1 // (op 448) 

v_lshlrev_b32 v27, 4, v8 // (op 448) 

v_lshl_add_u32 v8, s16, 5, v27 // (op 448) 

v_mov_b32 v25, 0 // (op 448) convert
v_mov_b32 v24, v8 // (op 448) convert

v_lshrrev_b32 v27, 1, v1 // (op 448) 
v_lshl_add_u32 v8, s14, 8, v27 // (op 448) 


v_mul_lo_u32 v11, v8, s3 // (op 448) most significant: low of low * high
v_mul_hi_u32 v9, v8, s2 // (op 448) most significant: high of low * low
v_mul_lo_u32 v10, v8, s2 // (op 448) least significant: low of low * low
v_add_u32 v11, v11, v9 // (op 448) most significant: sum


v_add_co_u32 v8, s[0:1], v24, v10 // (op 448) least significant half
v_addc_co_u32 v9, s[0:1], v25, v11, s[0:1] // (op 448) most significant half



s_mul_i32 s1, s2, s7 // (op 450) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 450) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 450) least significant
s_mul_hi_u32 s15, s2, s6 // (op 450) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 450) most significant: sum
s_add_u32 s1, s1, s15 // (op 450) most significant: sum


s_mov_b32 s40, s4 // (op 450) reg expression
s_mov_b32 s41, s5 // (op 450) reg expression
s_mov_b32 s42, s0 // (op 450) reg expression
s_mov_b32 s43, 131072 // (op 450) call()

v_lshrrev_b32 v27, 1, v1 // (op 515) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 515) 
v_and_b32 v10, -2, v1 // (op 515) 
s_lshr_b32 s14, s0, 1 // (op 515) 

v_sub_u32 v11, v1, v10 // (op 515) 

v_lshl_add_u32 v10, s14, 8, v27 // (op 515) 


v_lshlrev_b32 v27, 4, v11 // (op 515) 

v_lshl_add_u32 v11, s17, 5, v27 // (op 515) 

v_mov_b32 v25, 0 // (op 515) convert
v_mov_b32 v24, v11 // (op 515) convert

v_mul_lo_u32 v13, v10, s9 // (op 515) most significant: low of low * high
v_mul_hi_u32 v11, v10, s8 // (op 515) most significant: high of low * low
v_mul_lo_u32 v12, v10, s8 // (op 515) least significant: low of low * low
v_add_u32 v13, v13, v11 // (op 515) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v24 // (op 515) least significant half
v_addc_co_u32 v13, s[0:1], v13, v25, s[0:1] // (op 515) most significant half



s_mul_i32 s1, s8, s7 // (op 517) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 517) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 517) least significant
s_mul_hi_u32 s15, s8, s6 // (op 517) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 517) most significant: sum
s_add_u32 s1, s1, s15 // (op 517) most significant: sum


s_mov_b32 s44, s10 // (op 517) reg expression
s_mov_b32 s45, s11 // (op 517) reg expression
s_mov_b32 s46, s0 // (op 517) reg expression
s_mov_b32 s47, 131072 // (op 517) call()

v_lshrrev_b32 v27, 7, v1 // (op 579) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 579) 
v_and_b32 v10, 15, v1 // (op 579) 
s_lshr_b32 s14, s0, 1 // (op 579) 

v_lshl_add_u32 v11, v27, 4, v10 // (op 579) 


v_lshl_add_u32 v10, s16, 5, v11 // (op 579) 

v_and_b32 v11, 63, v1 // (op 579) 
v_lshrrev_b32 v27, 4, v11 // (op 579) 

v_lshl_add_u32 v11, s14, 3, v27 // (op 579) 


v_mul_lo_u32 v27, v11, s12 // (op 579) 

v_add_u32 v11, v10, v27 // (op 579) 


s_mul_i32 s1, s12, s23 // (op 581) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 581) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 581) least significant
s_mul_hi_u32 s15, s12, s22 // (op 581) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 581) most significant: sum
s_add_u32 s1, s1, s15 // (op 581) most significant: sum


s_mov_b32 s48, s20 // (op 581) reg expression
s_mov_b32 s49, s21 // (op 581) reg expression
s_mov_b32 s50, s0 // (op 581) reg expression
s_mov_b32 s51, 131072 // (op 581) call()

v_and_b32 v10, 63, v1 // (op 643) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 643) 
v_lshrrev_b32 v27, 4, v10 // (op 643) 

s_lshr_b32 s14, s0, 1 // (op 643) 

v_lshrrev_b32 v10, 6, v1 // (op 643) 
v_lshl_add_u32 v14, s14, 3, v27 // (op 643) 


v_and_b32 v27, 1, v10 // (op 643) 

v_and_b32 v10, 15, v1 // (op 643) 
v_lshl_add_u32 v15, v27, 4, v10 // (op 643) 


v_lshl_add_u32 v10, s17, 5, v15 // (op 643) 

v_mul_lo_u32 v27, v14, s24 // (op 643) 
v_add_u32 v27, v27, v10 // (op 643) 


s_mul_i32 s1, s24, s23 // (op 645) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 645) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 645) least significant
s_mul_hi_u32 s15, s24, s22 // (op 645) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 645) most significant: sum
s_add_u32 s1, s1, s15 // (op 645) most significant: sum


s_mov_b32 s52, s26 // (op 645) reg expression
s_mov_b32 s53, s27 // (op 645) reg expression
s_mov_b32 s54, s0 // (op 645) reg expression
s_mov_b32 s55, 131072 // (op 645) call()

v_and_b32 v10, -2, v1 // (op 736) 
v_sub_u32 v14, v1, v10 // (op 736) 

v_lshlrev_b32 v10, 4, v14 // (op 736) 

v_lshrrev_b32 v14, 1, v1 // (op 736) 
v_lshl_add_u32 v15, v14, 5, v10 // (op 736) 


s_nop 0
v_readfirstlane_b32 s0, v15 // (op 736) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -2, v1 // (op 788) 
v_sub_u32 v14, v1, v10 // (op 788) 

v_lshlrev_b32 v10, 4, v14 // (op 788) 

v_lshrrev_b32 v14, 1, v1 // (op 788) 
v_lshl_add_u32 v15, v14, 5, v10 // (op 788) 


s_nop 0
v_readfirstlane_b32 s1, v15 // (op 788) Wait state hazard: VALU Write Readlane Hazard

s_ashr_i64 s[14:15], s[6:7], 7 // (op 157) 
s_and_b32 s18, s14, 4294967294 // (op 157) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 156) 
s_mov_b32 s19, s14 // (op 156) convert

s_cmp_lt_u32 s18, s19 // (op 156) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156 // (op 156) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156:
s_barrier  // (op 962) 
s_add_u32 s18, s18, 1 // (op 158) 


v_mov_b64 v[14:15], v[8:9] // (op 401) 
v_mov_b64 v[16:17], v[14:15] // (op 401) reg expression
s_add_u32 s19, 4096, s0 // (op 401) 

s_mov_b32 m0, s19 // (op 401) reg expression
buffer_load_dwordx4 v16, s[40:43], 0 offen offset: 0 lds // (op 401) Load value direct to lds



s_lshl_b64 s[14:15], s[2:3], 7 // (op 438) 
v_mov_b32 v10, s14 // (op 438) 
v_mov_b32 v14, s15 // (op 438) 
v_add_co_u32 v8, s[38:39], v8, v10 // (op 438) least significant half
v_addc_co_u32 v9, s[38:39], v9, v14, s[38:39] // (op 438) most significant half




s_barrier  // (op 978) 


v_mov_b64 v[14:15], v[12:13] // (op 407) 
v_mov_b64 v[16:17], v[14:15] // (op 407) reg expression

s_mov_b32 m0, s1 // (op 407) reg expression
buffer_load_dwordx4 v16, s[44:47], 0 offen offset: 0 lds // (op 407) Load value direct to lds


s_lshl_b64 s[14:15], s[8:9], 7 // (op 507) 
v_mov_b32 v10, s14 // (op 507) 
v_mov_b32 v14, s15 // (op 507) 
v_add_co_u32 v12, s[38:39], v12, v10 // (op 507) least significant half
v_addc_co_u32 v13, s[38:39], v13, v14, s[38:39] // (op 507) most significant half




s_waitcnt vmcnt(1)
s_barrier  // (op 968) 



ds_read_b64_tr_b8 v[14:15], v0 offset:4096 // (op 164) Transpose load local data 
ds_read_b64_tr_b8 v[16:17], v0 offset:4352 // (op 164) Transpose load local data 
ds_read_b64_tr_b8 v[18:19], v0 offset:6144 // (op 164) Transpose load local data 
ds_read_b64_tr_b8 v[20:21], v0 offset:6400 // (op 164) Transpose load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 984) 



ds_read_b64_tr_b8 v[28:29], v4 // (op 166) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v4 offset:256 // (op 166) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:2048 // (op 166) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:2304 // (op 166) Transpose load local data 

v_mov_b64 v[22:23], s[12:13] // (op 168) convert

v_mov_b32 v10, v11 // (op 168) 
v_mov_b32 v24, v10 // (op 168) reg expression
v_mov_b32 v25, v24 // (op 168) 
buffer_load_ubyte v26, v25, s[48:51], 0 offen offset: 0 // (op 168) Load value




s_lshl_b32 s19, s12, 2 // (op 571) 
v_add_u32 v11, v11, s19 // (op 571) 


v_mov_b64 v[22:23], s[24:25] // (op 170) convert

v_mov_b32 v10, v27 // (op 170) 
v_mov_b32 v24, v10 // (op 170) reg expression
v_mov_b32 v25, v24 // (op 170) 
buffer_load_ubyte v36, v25, s[52:55], 0 offen offset: 0 // (op 170) Load value




s_lshl_b32 s19, s24, 2 // (op 635) 
v_add_u32 v27, v27, s19 // (op 635) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[14:21], v[28:35], a[0:3], v26, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 171) 




s_ashr_i64 s[14:15], s[6:7], 7 // (op 156) 
s_mov_b32 s19, s14 // (op 156) convert

s_cmp_lt_u32 s18, s19 // (op 156) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156 // (op 156) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156 if true)
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156:





















s_barrier  // (op 1030) 




v_cmp_eq_f32 vcc, 0.00000, v5 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116
s_nop 0
v_accvgpr_read v8, a0 // (op 42) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 42) 
v_accvgpr_read v10, a2 // (op 42) 
v_accvgpr_read v11, a3 // (op 42) 
v_mul_f32 v12, v7, v8 // (op 42) 
v_mul_f32 v13, v7, v9 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 

v_lshrrev_b32 v8, 16, v12 // (op 48) 
v_lshrrev_b32 v9, 16, v13 // (op 48) 
v_lshrrev_b32 v10, 16, v14 // (op 48) 
v_lshrrev_b32 v11, 16, v15 // (op 48) 

v_lshrrev_b32 v0, 7, v1 // (op 829) 
v_and_b32 v6, 63, v1 // (op 829) 
v_lshrrev_b32 v3, 4, v6 // (op 829) 
v_lshlrev_b32 v4, 2, v3 // (op 829) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 829) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 829) 

v_lshrrev_b32 v3, 2, v6 // (op 829) 

v_and_b32 v4, 3, v3 // (op 829) 

v_and_b32 v3, 3, v1 // (op 829) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 829) 


v_lshrrev_b32 v3, 6, v1 // (op 829) 
v_and_b32 v4, 1, v3 // (op 829) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 829) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 829) 

v_mul_lo_u32 v3, v4, s28 // (op 829) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 829) 


s_mul_i32 s1, s28, s33 // (op 831) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 831) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 831) least significant
s_mul_hi_u32 s3, s28, s32 // (op 831) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 831) most significant: sum
s_add_u32 s1, s1, s3 // (op 831) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 831) 

s_mov_b32 s4, s30 // (op 831) reg expression
s_mov_b32 s5, s31 // (op 831) reg expression
s_mov_b32 s6, s2 // (op 831) reg expression
s_mov_b32 s7, 131072 // (op 831) call()


v_pack_b32_f16 v12, v8, v9 // (op 50) 
v_pack_b32_f16 v13, v10, v11 // (op 50) 


buffer_store_dwordx2 v[12:13], v4, s[4:7], 0 offen // (op 50) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116:
v_lshrrev_b32 v0, 7, v1 // (op 706) 
v_and_b32 v6, 63, v1 // (op 706) 
v_lshrrev_b32 v3, 4, v6 // (op 706) 
v_lshlrev_b32 v4, 2, v3 // (op 706) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 706) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 706) 

v_lshrrev_b32 v3, 2, v6 // (op 706) 

v_and_b32 v4, 3, v3 // (op 706) 

v_and_b32 v3, 3, v1 // (op 706) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 706) 


v_lshrrev_b32 v3, 6, v1 // (op 706) 
v_and_b32 v4, 1, v3 // (op 706) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 706) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 706) 

v_mul_lo_u32 v3, v4, s34 // (op 706) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 706) 


s_mul_i32 s1, s34, s33 // (op 708) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 708) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 708) least significant
s_mul_hi_u32 s3, s34, s32 // (op 708) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 708) most significant: sum
s_add_u32 s1, s1, s3 // (op 708) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 708) 

s_mov_b32 s8, s36 // (op 708) reg expression
s_mov_b32 s9, s37 // (op 708) reg expression
s_mov_b32 s10, s2 // (op 708) reg expression
s_mov_b32 s11, 131072 // (op 708) call()





buffer_load_dwordx2 v[8:9], v4, s[8:11], 0 offen // (op 129) Load value



v_accvgpr_read v10, a0 // (op 125) 
v_accvgpr_read v11, a1 // (op 125) 
v_accvgpr_read v12, a2 // (op 125) 
v_accvgpr_read v13, a3 // (op 125) 
v_mul_f32 v14, v7, v10 // (op 125) 
v_mul_f32 v15, v7, v11 // (op 125) 
v_mul_f32 v16, v7, v12 // (op 125) 
v_mul_f32 v17, v7, v13 // (op 125) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v8 // (op 130) 
v_lshrrev_b32 v11, 16, v8 // (op 130) 
v_lshlrev_b32 v10, 16, v10 // (op 130) 
v_lshlrev_b32 v11, 16, v11 // (op 130) 
v_mul_f32 v18, v5, v10 // (op 130) 
v_mul_f32 v19, v5, v11 // (op 130) 
v_and_b32 v10, 65535, v9 // (op 130) 
v_lshrrev_b32 v11, 16, v9 // (op 130) 
v_lshlrev_b32 v10, 16, v10 // (op 130) 
v_lshlrev_b32 v11, 16, v11 // (op 130) 
v_mul_f32 v20, v5, v10 // (op 130) 
v_mul_f32 v21, v5, v11 // (op 130) 


v_add_f32 v8, v18, v14 // (op 126) 
v_add_f32 v9, v19, v15 // (op 126) 
v_add_f32 v10, v20, v16 // (op 126) 
v_add_f32 v11, v21, v17 // (op 126) 


v_lshrrev_b32 v12, 16, v8 // (op 127) 
v_lshrrev_b32 v13, 16, v9 // (op 127) 
v_lshrrev_b32 v14, 16, v10 // (op 127) 
v_lshrrev_b32 v15, 16, v11 // (op 127) 

v_lshrrev_b32 v0, 7, v1 // (op 846) 
v_and_b32 v6, 63, v1 // (op 846) 
v_lshrrev_b32 v3, 4, v6 // (op 846) 
v_lshlrev_b32 v4, 2, v3 // (op 846) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 846) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 846) 

v_lshrrev_b32 v3, 2, v6 // (op 846) 

v_and_b32 v4, 3, v3 // (op 846) 

v_and_b32 v3, 3, v1 // (op 846) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 846) 


v_lshrrev_b32 v3, 6, v1 // (op 846) 
v_and_b32 v4, 1, v3 // (op 846) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 846) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 846) 

v_mul_lo_u32 v3, v4, s28 // (op 846) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 846) 


s_mul_i32 s1, s28, s33 // (op 848) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 848) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 848) least significant
s_mul_hi_u32 s3, s28, s32 // (op 848) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 848) most significant: sum
s_add_u32 s1, s1, s3 // (op 848) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 848) 

s_mov_b32 s4, s30 // (op 848) reg expression
s_mov_b32 s5, s31 // (op 848) reg expression
s_mov_b32 s6, s2 // (op 848) reg expression
s_mov_b32 s7, 131072 // (op 848) call()


v_pack_b32_f16 v8, v12, v13 // (op 128) 
v_pack_b32_f16 v9, v14, v15 // (op 128) 


buffer_store_dwordx2 v[8:9], v4, s[4:7], 0 offen // (op 128) Store value







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116:



s_endpgm  // End of RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.LRR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end:
.size RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2, .LRR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end-RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16384 // lds bytes
  .amdhsa_accum_offset 48
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
    DataTypeB: B8
    DestDataType: b
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
    macrotile: [32, 32, 128]
    threads: [256, 1, 1]
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
  - .name: RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
    .symbol: RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 46
    .agpr_count: 4
    .max_flat_workgroup_size: 256
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

