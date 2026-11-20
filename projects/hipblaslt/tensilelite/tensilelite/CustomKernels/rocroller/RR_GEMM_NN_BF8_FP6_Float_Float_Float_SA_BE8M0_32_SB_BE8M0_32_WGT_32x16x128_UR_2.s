// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.p2align 8
.type RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2,@function
RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 761) 
v_and_b32 v3, 63, v1 // (op 761) 
v_sub_u32 v4, v3, v0 // (op 761) 

v_lshrrev_b32 v3, 1, v4 // (op 761) 
v_add_lshl_u32 v5, v0, v3, 5 // (op 761) 


v_and_b32 v0, 1, v4 // (op 761) 

v_lshlrev_b32 v3, 3, v0 // (op 761) 

v_lshrrev_b32 v0, 6, v1 // (op 761) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 761) 


v_add_u32 v0, v5, v4 // (op 761) 


v_and_b32 v3, 15, v1 // (op 814) 
v_and_b32 v4, 63, v1 // (op 814) 
v_lshrrev_b32 v5, 4, v4 // (op 814) 

v_lshlrev_b32 v4, 5, v5 // (op 814) 

v_lshl_add_u32 v5, v3, 7, v4 // (op 814) 


v_mul_lo_u32 v3, v5, 6 // (op 814) 

v_lshrrev_b32 v4, 3, v3 // (op 814) 

v_lshrrev_b32 v3, 6, v1 // (op 867) 
v_and_b32 v5, 15, v1 // (op 867) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 867) 


v_and_b32 v3, 63, v1 // (op 867) 
v_lshrrev_b32 v5, 4, v3 // (op 867) 

v_lshl_add_u32 v3, v5, 5, v6 // (op 867) 


v_and_b32 v5, 48, v1 // (op 940) 
v_and_b32 v6, 63, v1 // (op 940) 
v_sub_u32 v7, v6, v5 // (op 940) 

v_lshrrev_b32 v6, 1, v7 // (op 940) 
v_add_lshl_u32 v8, v5, v6, 5 // (op 940) 


v_and_b32 v5, 1, v7 // (op 940) 

v_lshlrev_b32 v6, 3, v5 // (op 940) 

v_lshrrev_b32 v5, 6, v1 // (op 940) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 940) 


v_add_u32 v5, v8, v7 // (op 940) 


v_and_b32 v6, 15, v1 // (op 964) 
v_and_b32 v7, 63, v1 // (op 964) 
v_lshrrev_b32 v8, 4, v7 // (op 964) 

v_lshlrev_b32 v7, 5, v8 // (op 964) 

v_lshl_add_u32 v8, v6, 7, v7 // (op 964) 


v_mul_lo_u32 v6, v8, 6 // (op 964) 

v_lshrrev_b32 v7, 3, v6 // (op 964) 

v_lshrrev_b32 v6, 6, v1 // (op 988) 
v_and_b32 v8, 15, v1 // (op 988) 
v_lshl_add_u32 v9, v6, 4, v8 // (op 988) 


v_and_b32 v6, 63, v1 // (op 988) 
v_lshrrev_b32 v8, 4, v6 // (op 988) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 988) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_and_b32 v10, -2, v1 // (op 472) 
v_sub_u32 v11, v1, v10 // (op 472) 

v_lshlrev_b32 v10, 4, v11 // (op 472) 

v_lshl_add_u32 v11, s16, 5, v10 // (op 472) 

v_mov_b32 v13, 0 // (op 472) convert
v_mov_b32 v12, v11 // (op 472) convert

v_lshrrev_b32 v10, 1, v1 // (op 472) 
v_mul_lo_u32 v15, v10, s3 // (op 472) most significant: low of low * high
v_mul_hi_u32 v11, v10, s2 // (op 472) most significant: high of low * low
v_mul_lo_u32 v14, v10, s2 // (op 472) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 472) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 472) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 472) most significant half



s_mul_i32 s1, s2, s7 // (op 474) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 474) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 474) least significant
s_mul_hi_u32 s15, s2, s6 // (op 474) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 474) most significant: sum
s_add_u32 s1, s1, s15 // (op 474) most significant: sum


s_mov_b32 s40, s4 // (op 474) reg expression
s_mov_b32 s41, s5 // (op 474) reg expression
s_mov_b32 s42, s0 // (op 474) reg expression
s_mov_b32 s43, 131072 // (op 474) call()

v_and_b32 v12, -2, v1 // (op 491) 
v_sub_u32 v13, v1, v12 // (op 491) 

v_lshlrev_b32 v12, 4, v13 // (op 491) 

v_lshl_add_u32 v13, s16, 5, v12 // (op 491) 

v_mov_b32 v15, 0 // (op 491) convert
v_mov_b32 v14, v13 // (op 491) convert

v_lshrrev_b32 v12, 1, v1 // (op 491) 
v_mul_lo_u32 v17, v12, s3 // (op 491) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 491) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 491) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 491) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 491) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 491) most significant half



s_mul_i32 s1, s2, s7 // (op 493) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 493) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 493) least significant
s_mul_hi_u32 s15, s2, s6 // (op 493) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 493) most significant: sum
s_add_u32 s1, s1, s15 // (op 493) most significant: sum


s_mov_b32 s40, s4 // (op 493) reg expression
s_mov_b32 s41, s5 // (op 493) reg expression
s_mov_b32 s42, s0 // (op 493) reg expression
s_mov_b32 s43, 131072 // (op 493) call()

v_and_b32 v14, -8, v1 // (op 536) 
v_sub_u32 v15, v1, v14 // (op 536) 

v_lshlrev_b32 v14, 4, v15 // (op 536) 

v_mov_b32 v17, 0 // (op 536) convert
v_mov_b32 v16, v14 // (op 536) convert

v_lshrrev_b32 v14, 3, v1 // (op 536) 
v_lshl_add_u32 v15, s17, 4, v14 // (op 536) 

v_mul_lo_u32 v19, v15, s9 // (op 536) most significant: low of low * high
v_mul_hi_u32 v14, v15, s8 // (op 536) most significant: high of low * low
v_mul_lo_u32 v18, v15, s8 // (op 536) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 536) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 536) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 536) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 536) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 536) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 536) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 536) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 536) 

v_mov_b32 v16, v14 // (op 536) convert

s_mul_i32 s1, s13, 6 // (op 538) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 538) least significant
s_mul_hi_u32 s14, s12, 6 // (op 538) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 538) most significant: sum

s_mul_i32 s15, s8, s1 // (op 538) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 538) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 538) least significant
s_mul_hi_u32 s19, s8, s0 // (op 538) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 538) most significant: sum
s_add_u32 s15, s15, s19 // (op 538) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 538) 

s_mov_b32 s44, s10 // (op 538) reg expression
s_mov_b32 s45, s11 // (op 538) reg expression
s_mov_b32 s46, s0 // (op 538) reg expression
s_mov_b32 s47, 131072 // (op 538) call()

v_and_b32 v14, -8, v1 // (op 555) 
v_sub_u32 v15, v1, v14 // (op 555) 

v_lshlrev_b32 v14, 4, v15 // (op 555) 

v_mov_b32 v19, 0 // (op 555) convert
v_mov_b32 v18, v14 // (op 555) convert

v_lshrrev_b32 v17, 3, v1 // (op 555) 
v_lshl_add_u32 v14, s17, 4, v17 // (op 555) 

v_mul_lo_u32 v21, v14, s9 // (op 555) most significant: low of low * high
v_mul_hi_u32 v15, v14, s8 // (op 555) most significant: high of low * low
v_mul_lo_u32 v20, v14, s8 // (op 555) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 555) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 555) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 555) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 555) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 555) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 555) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 555) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 555) 

v_mov_b32 v17, v14 // (op 555) convert

s_mul_i32 s1, s13, 6 // (op 557) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 557) least significant
s_mul_hi_u32 s14, s12, 6 // (op 557) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 557) most significant: sum

s_mul_i32 s15, s8, s1 // (op 557) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 557) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 557) least significant
s_mul_hi_u32 s19, s8, s0 // (op 557) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 557) most significant: sum
s_add_u32 s15, s15, s19 // (op 557) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 557) 

s_mov_b32 s44, s10 // (op 557) reg expression
s_mov_b32 s45, s11 // (op 557) reg expression
s_mov_b32 s46, s0 // (op 557) reg expression
s_mov_b32 s47, 131072 // (op 557) call()

v_and_b32 v14, -32, v1 // (op 600) 
v_sub_u32 v15, v1, v14 // (op 600) 

v_lshl_add_u32 v14, s16, 5, v15 // (op 600) 

v_lshrrev_b32 v15, 5, v1 // (op 600) 

s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v18, v15, s20 // (op 600) 

v_add_u32 v15, v14, v18 // (op 600) 


s_mul_i32 s1, s20, s25 // (op 602) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 602) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 602) least significant
s_mul_hi_u32 s15, s20, s24 // (op 602) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 602) most significant: sum
s_add_u32 s1, s1, s15 // (op 602) most significant: sum


s_mov_b32 s48, s22 // (op 602) reg expression
s_mov_b32 s49, s23 // (op 602) reg expression
s_mov_b32 s50, s0 // (op 602) reg expression
s_mov_b32 s51, 131072 // (op 602) call()

v_and_b32 v14, -32, v1 // (op 619) 
v_sub_u32 v18, v1, v14 // (op 619) 

v_lshl_add_u32 v14, s16, 5, v18 // (op 619) 

v_lshrrev_b32 v18, 5, v1 // (op 619) 
v_mul_lo_u32 v19, v18, s20 // (op 619) 

v_add_u32 v18, v14, v19 // (op 619) 


s_mul_i32 s1, s20, s25 // (op 621) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 621) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 621) least significant
s_mul_hi_u32 s15, s20, s24 // (op 621) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 621) most significant: sum
s_add_u32 s1, s1, s15 // (op 621) most significant: sum


s_mov_b32 s48, s22 // (op 621) reg expression
s_mov_b32 s49, s23 // (op 621) reg expression
s_mov_b32 s50, s0 // (op 621) reg expression
s_mov_b32 s51, 131072 // (op 621) call()

v_and_b32 v14, 15, v1 // (op 664) 
v_lshl_add_u32 v19, s17, 4, v14 // (op 664) 

v_mul_lo_u32 v14, v19, s26 // (op 664) 

v_and_b32 v19, 63, v1 // (op 664) 
v_lshrrev_b32 v20, 4, v19 // (op 664) 

v_add_u32 v19, v20, v14 // (op 664) 


s_mul_i32 s1, s26, s13 // (op 666) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 666) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 666) least significant
s_mul_hi_u32 s15, s26, s12 // (op 666) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 666) most significant: sum
s_add_u32 s1, s1, s15 // (op 666) most significant: sum


s_mov_b32 s52, s28 // (op 666) reg expression
s_mov_b32 s53, s29 // (op 666) reg expression
s_mov_b32 s54, s0 // (op 666) reg expression
s_mov_b32 s55, 131072 // (op 666) call()

v_and_b32 v14, 15, v1 // (op 683) 
v_lshl_add_u32 v20, s17, 4, v14 // (op 683) 

v_mul_lo_u32 v14, v20, s26 // (op 683) 

v_and_b32 v20, 63, v1 // (op 683) 
v_lshrrev_b32 v21, 4, v20 // (op 683) 

v_add_u32 v20, v21, v14 // (op 683) 


s_mul_i32 s1, s26, s13 // (op 685) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 685) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 685) least significant
s_mul_hi_u32 s15, s26, s12 // (op 685) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 685) most significant: sum
s_add_u32 s1, s1, s15 // (op 685) most significant: sum


s_mov_b32 s52, s28 // (op 685) reg expression
s_mov_b32 s53, s29 // (op 685) reg expression
s_mov_b32 s54, s0 // (op 685) reg expression
s_mov_b32 s55, 131072 // (op 685) call()




buffer_load_ubyte v14, v19, s[52:55], 0 offen offset: 0 // (op 337) Load value



buffer_load_dwordx3 v[22:24], v16, s[44:47], 0 offen // (op 341) Load value
v_and_b32 v21, -2, v1 // (op 749) 
v_sub_u32 v25, v1, v21 // (op 749) 

v_lshlrev_b32 v21, 4, v25 // (op 749) 

v_lshrrev_b32 v25, 1, v1 // (op 749) 
v_lshl_add_u32 v26, v25, 5, v21 // (op 749) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 749) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 428) 
v_mov_b64 v[26:27], s[14:15] // (op 428) convert


v_mov_b64 v[28:29], v[10:11] // (op 428) reg expression
s_mov_b32 m0, s0 // (op 428) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 428) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v26 // (op 428) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 428) most significant half

s_add_u32 m0, m0, 2048 // (op 428) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 428) Load value direct to lds





v_mov_b32 v21, v15 // (op 347) reg expression

buffer_load_ubyte v25, v21, s[48:51], 0 offen offset: 0 // (op 347) Load value




buffer_load_ubyte v21, v19, s[52:55], 0 offen offset:4 // (op 351) Load value




buffer_load_dwordx3 v[26:28], v16, s[44:47], 0 offen offset:96 // (op 355) Load value

v_and_b32 v16, -2, v1 // (op 928) 
v_sub_u32 v19, v1, v16 // (op 928) 

v_lshlrev_b32 v16, 4, v19 // (op 928) 

v_lshrrev_b32 v19, 1, v1 // (op 928) 
v_lshl_add_u32 v29, v19, 5, v16 // (op 928) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 928) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 433) 
v_mov_b64 v[30:31], s[14:15] // (op 433) convert


s_lshl_b64 s[14:15], s[2:3], 7 // (op 433) 
v_mov_b32 v16, s14 // (op 433) 
v_mov_b32 v19, s15 // (op 433) 
v_add_co_u32 v32, s[18:19], v10, v16 // (op 433) least significant half
v_addc_co_u32 v33, s[18:19], v11, v19, s[18:19] // (op 433) most significant half




s_add_u32 s1, 4096, s0 // (op 433) 
s_mov_b32 m0, s1 // (op 433) reg expression
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 433) Load value direct to lds
v_add_co_u32 v32, s[14:15], v32, v30 // (op 433) least significant half
v_addc_co_u32 v33, s[14:15], 0, v33, s[14:15] // (op 433) most significant half

s_add_u32 m0, m0, 2048 // (op 433) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 433) Load value direct to lds







s_lshl_b32 s0, s20, 2 // (op 361) 
v_add_u32 v16, v15, s0 // (op 361) 


buffer_load_ubyte v19, v16, s[48:51], 0 offen offset: 0 // (op 361) Load value


v_and_b32 v10, -8, v1 // (op 802) 
v_sub_u32 v11, v1, v10 // (op 802) 

v_lshlrev_b32 v10, 4, v11 // (op 802) 

v_lshrrev_b32 v11, 3, v1 // (op 802) 
v_lshl_add_u32 v15, v11, 7, v10 // (op 802) 


v_mul_lo_u32 v16, v15, 6 // (op 802) 

v_lshrrev_b32 v15, 3, v16 // (op 802) 




s_waitcnt vmcnt(8)
ds_write_b96 v15, v[22:24] offset:8192 // (op 363) Store local data 

v_and_b32 v10, -32, v1 // (op 855) 
v_sub_u32 v11, v1, v10 // (op 855) 

v_and_b32 v10, 4294967264, v1 // (op 855) 
v_add_u32 v15, v10, v11 // (op 855) 





s_waitcnt vmcnt(5)
ds_write_b8 v15, v25 offset:9728 // (op 365) Store local data 

v_and_b32 v10, -2, v1 // (op 721) 
v_sub_u32 v11, v1, v10 // (op 721) 

v_lshlrev_b32 v10, 4, v11 // (op 721) 

v_lshrrev_b32 v11, 1, v1 // (op 721) 
v_lshl_add_u32 v15, v11, 5, v10 // (op 721) 


s_nop 0
v_readfirstlane_b32 s0, v15 // (op 721) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -8, v1 // (op 773) 
v_sub_u32 v11, v1, v10 // (op 773) 

v_lshlrev_b32 v10, 4, v11 // (op 773) 

v_lshrrev_b32 v11, 3, v1 // (op 773) 
v_lshl_add_u32 v15, v11, 7, v10 // (op 773) 


v_mul_lo_u32 v16, v15, 6 // (op 773) 

v_lshrrev_b32 v15, 3, v16 // (op 773) 

v_and_b32 v16, -32, v1 // (op 826) 
v_sub_u32 v10, v1, v16 // (op 826) 

v_and_b32 v11, 4294967264, v1 // (op 826) 
v_add_u32 v16, v11, v10 // (op 826) 


v_and_b32 v10, -2, v1 // (op 913) 
v_sub_u32 v11, v1, v10 // (op 913) 

v_lshlrev_b32 v10, 4, v11 // (op 913) 

v_lshrrev_b32 v11, 1, v1 // (op 913) 
v_lshl_add_u32 v29, v11, 5, v10 // (op 913) 


s_nop 0
v_readfirstlane_b32 s1, v29 // (op 913) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -8, v1 // (op 952) 
v_sub_u32 v11, v1, v10 // (op 952) 

v_lshlrev_b32 v10, 4, v11 // (op 952) 

v_lshrrev_b32 v11, 3, v1 // (op 952) 
v_lshl_add_u32 v29, v11, 7, v10 // (op 952) 


v_mul_lo_u32 v10, v29, 6 // (op 952) 

v_lshrrev_b32 v11, 3, v10 // (op 952) 

v_and_b32 v10, -32, v1 // (op 976) 
v_sub_u32 v29, v1, v10 // (op 976) 

v_and_b32 v10, 4294967264, v1 // (op 976) 
v_add_u32 v30, v10, v29 // (op 976) 


s_mov_b32 s14, 0 // (op 55) call()
s_ashr_i64 s[18:19], s[6:7], 7 // (op 54) 
s_mov_b32 s15, s18 // (op 54) convert

s_and_b32 s18, s15, 4294967294 // (op 54) 

s_cmp_lt_u32 s14, s18 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 if false)
RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54:
s_barrier  // (op 1021) 
s_add_u32 s14, s14, 2 // (op 56) 



ds_read_b64_tr_b8 v[32:33], v0 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:256 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:2048 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:2304 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1034) 



ds_read_b128 v[40:43], v4 offset:8192 // (op 70) Load local data 
ds_read_b64 v[44:45], v4 offset:8208 // (op 70) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1048) 


ds_read_u8 v10, v3 offset:9728 // (op 75) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:45], a[0:3], v10, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 82) 





v_mov_b32 v10, v20 // (op 80) 

buffer_load_ubyte v14, v10, s[52:55], 0 offen offset:8 // (op 80) Load value


v_mov_b32 v29, v17 // (op 10) 

buffer_load_dwordx3 v[22:24], v29, s[44:47], 0 offen offset:192 // (op 10) Load value
s_barrier  // (op 1024) 

s_lshl_b64 s[18:19], s[2:3], 6 // (op 412) 
v_mov_b64 v[32:33], s[18:19] // (op 412) convert


v_mov_b64 v[34:35], v[12:13] // (op 412) 
s_lshl_b64 s[18:19], s[2:3], 7 // (op 412) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 412) 

v_mov_b32 v31, s38 // (op 412) 
v_mov_b32 v36, s39 // (op 412) 
v_add_co_u32 v38, s[18:19], v34, v31 // (op 412) least significant half
v_addc_co_u32 v39, s[18:19], v35, v36, s[18:19] // (op 412) most significant half




s_mov_b32 m0, s0 // (op 412) reg expression
buffer_load_dwordx4 v38, s[40:43], 0 offen offset: 0 lds // (op 412) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v32 // (op 412) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 412) most significant half

s_add_u32 m0, m0, 2048 // (op 412) 
buffer_load_dwordx4 v38, s[40:43], 0 offen offset: 0 lds // (op 412) Load value direct to lds




v_mov_b32 v31, v18 // (op 17) 
s_lshl_b32 s15, s20, 3 // (op 17) 
v_add_u32 v32, v31, s15 // (op 17) 


buffer_load_ubyte v25, v32, s[48:51], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1071) 



s_waitcnt vmcnt(8)
ds_write_b96 v11, v[26:28] offset:9856 // (op 235) Store local data 
s_barrier  // (op 1077) 



s_waitcnt vmcnt(5)
ds_write_b8 v30, v19 offset:11392 // (op 236) Store local data 
s_barrier  // (op 1062) 



ds_read_b64_tr_b8 v[36:37], v5 offset:4096 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v5 offset:4352 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v5 offset:6144 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v5 offset:6400 // (op 227) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1068) 



ds_read_b128 v[44:47], v7 offset:9856 // (op 229) Load local data 
ds_read_b64 v[48:49], v7 offset:9872 // (op 229) Load local data 
s_barrier  // (op 1124) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1074) 


ds_read_u8 v32, v6 offset:11392 // (op 231) Load local data 
s_barrier  // (op 1130) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:49], a[0:3], v32, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 234) 






buffer_load_ubyte v21, v10, s[52:55], 0 offen offset:12 // (op 233) Load value




v_add_u32 v20, 8, v20 // (op 675) 
buffer_load_dwordx3 v[26:28], v29, s[44:47], 0 offen offset:288 // (op 237) Load value

s_barrier  // (op 1065) 
v_add_u32 v17, 192, v17 // (op 547) 

s_lshl_b64 s[18:19], s[2:3], 6 // (op 423) 
v_mov_b64 v[32:33], s[18:19] // (op 423) convert


s_lshl_b64 s[18:19], s[2:3], 7 // (op 423) 
s_mul_i32 s39, 3, s19 // (op 423) most significant: low of low * high
s_mul_i32 s38, 3, s18 // (op 423) least significant
s_mul_hi_u32 s15, 3, s18 // (op 423) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 423) most significant: sum


v_mov_b32 v10, s38 // (op 423) 
v_mov_b32 v29, s39 // (op 423) 
v_add_co_u32 v36, s[18:19], v34, v10 // (op 423) least significant half
v_addc_co_u32 v37, s[18:19], v35, v29, s[18:19] // (op 423) most significant half




s_add_u32 s15, 4096, s1 // (op 423) 
s_mov_b32 m0, s15 // (op 423) reg expression
buffer_load_dwordx4 v36, s[40:43], 0 offen offset: 0 lds // (op 423) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 423) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 423) most significant half

s_add_u32 m0, m0, 2048 // (op 423) 
buffer_load_dwordx4 v36, s[40:43], 0 offen offset: 0 lds // (op 423) Load value direct to lds






s_lshl_b32 s15, s20, 2 // (op 238) 
s_mul_i32 s18, 3, s15 // (op 238) 

v_add_u32 v10, v31, s18 // (op 238) 


s_lshl_b64 s[18:19], s[2:3], 7 // (op 483) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 483) 

v_mov_b32 v29, s38 // (op 483) 
v_mov_b32 v32, s39 // (op 483) 
v_add_co_u32 v12, s[18:19], v12, v29 // (op 483) least significant half
v_addc_co_u32 v13, s[18:19], v13, v32, s[18:19] // (op 483) most significant half




buffer_load_ubyte v19, v10, s[48:51], 0 offen offset: 0 // (op 238) Load value


s_barrier  // (op 1037) 
s_lshl_b32 s15, s20, 3 // (op 611) 



v_add_u32 v18, v18, s15 // (op 611) 

s_waitcnt vmcnt(8)
ds_write_b96 v15, v[22:24] offset:8192 // (op 13) Store local data 
s_barrier  // (op 1051) 



s_waitcnt vmcnt(5)
ds_write_b8 v16, v25 offset:9728 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[6:7], 7 // (op 54) 
s_mov_b32 s15, s18 // (op 54) convert

s_and_b32 s18, s15, 4294967294 // (op 54) 

s_cmp_lt_u32 s14, s18 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 if true)
RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54:














s_barrier  // (op 1088) 
v_and_b32 v20, -2, v1 // (op 446) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 446) 
v_sub_u32 v10, v1, v20 // (op 446) 

s_lshr_b32 s14, s0, 1 // (op 446) 

v_lshlrev_b32 v20, 4, v10 // (op 446) 

v_lshl_add_u32 v10, s16, 5, v20 // (op 446) 

v_mov_b32 v23, 0 // (op 446) convert
v_mov_b32 v22, v10 // (op 446) convert

v_lshrrev_b32 v20, 1, v1 // (op 446) 
v_lshl_add_u32 v10, s14, 8, v20 // (op 446) 


v_mul_lo_u32 v25, v10, s3 // (op 446) most significant: low of low * high
v_mul_hi_u32 v20, v10, s2 // (op 446) most significant: high of low * low
v_mul_lo_u32 v24, v10, s2 // (op 446) least significant: low of low * low
v_add_u32 v25, v25, v20 // (op 446) most significant: sum


v_add_co_u32 v10, s[0:1], v22, v24 // (op 446) least significant half
v_addc_co_u32 v11, s[0:1], v23, v25, s[0:1] // (op 446) most significant half



s_mul_i32 s1, s2, s7 // (op 448) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 448) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 448) least significant
s_mul_hi_u32 s15, s2, s6 // (op 448) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 448) most significant: sum
s_add_u32 s1, s1, s15 // (op 448) most significant: sum


s_mov_b32 s40, s4 // (op 448) reg expression
s_mov_b32 s41, s5 // (op 448) reg expression
s_mov_b32 s42, s0 // (op 448) reg expression
s_mov_b32 s43, 131072 // (op 448) call()

v_and_b32 v20, -8, v1 // (op 513) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 513) 
v_sub_u32 v12, v1, v20 // (op 513) 

s_lshr_b32 s14, s0, 1 // (op 513) 

v_lshlrev_b32 v20, 4, v12 // (op 513) 

v_lshl_add_u32 v12, s14, 8, v20 // (op 513) 


v_mov_b32 v23, 0 // (op 513) convert
v_mov_b32 v22, v12 // (op 513) convert

v_lshrrev_b32 v20, 3, v1 // (op 513) 
v_lshl_add_u32 v12, s17, 4, v20 // (op 513) 

v_mul_lo_u32 v25, v12, s9 // (op 513) most significant: low of low * high
v_mul_hi_u32 v20, v12, s8 // (op 513) most significant: high of low * low
v_mul_lo_u32 v24, v12, s8 // (op 513) least significant: low of low * low
v_add_u32 v25, v25, v20 // (op 513) most significant: sum


v_add_co_u32 v12, s[0:1], v22, v24 // (op 513) least significant half
v_addc_co_u32 v13, s[0:1], v23, v25, s[0:1] // (op 513) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 513) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 513) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 513) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 513) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 513) 

v_mov_b32 v20, v12 // (op 513) convert

s_mul_i32 s1, s13, 6 // (op 515) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 515) least significant
s_mul_hi_u32 s14, s12, 6 // (op 515) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 515) most significant: sum

s_mul_i32 s15, s8, s1 // (op 515) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 515) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 515) least significant
s_mul_hi_u32 s19, s8, s0 // (op 515) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 515) most significant: sum
s_add_u32 s15, s15, s19 // (op 515) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 515) 

s_mov_b32 s44, s10 // (op 515) reg expression
s_mov_b32 s45, s11 // (op 515) reg expression
s_mov_b32 s46, s0 // (op 515) reg expression
s_mov_b32 s47, 131072 // (op 515) call()

v_and_b32 v12, -32, v1 // (op 577) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 577) 
v_sub_u32 v13, v1, v12 // (op 577) 

s_lshr_b32 s14, s0, 1 // (op 577) 

v_lshl_add_u32 v12, s16, 5, v13 // (op 577) 

v_lshrrev_b32 v13, 5, v1 // (op 577) 
v_lshl_add_u32 v14, s14, 3, v13 // (op 577) 


v_mul_lo_u32 v13, v14, s20 // (op 577) 

v_add_u32 v14, v12, v13 // (op 577) 


s_mul_i32 s1, s20, s25 // (op 579) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 579) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 579) least significant
s_mul_hi_u32 s15, s20, s24 // (op 579) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 579) most significant: sum
s_add_u32 s1, s1, s15 // (op 579) most significant: sum


s_mov_b32 s48, s22 // (op 579) reg expression
s_mov_b32 s49, s23 // (op 579) reg expression
s_mov_b32 s50, s0 // (op 579) reg expression
s_mov_b32 s51, 131072 // (op 579) call()

v_and_b32 v12, 63, v1 // (op 641) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 641) 
v_lshrrev_b32 v13, 4, v12 // (op 641) 

s_lshr_b32 s14, s0, 1 // (op 641) 

v_and_b32 v12, 15, v1 // (op 641) 
v_lshl_add_u32 v15, s14, 3, v13 // (op 641) 


v_lshl_add_u32 v13, s17, 4, v12 // (op 641) 

v_mul_lo_u32 v12, v13, s26 // (op 641) 

v_add_u32 v13, v15, v12 // (op 641) 


s_mul_i32 s1, s26, s13 // (op 643) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 643) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 643) least significant
s_mul_hi_u32 s15, s26, s12 // (op 643) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 643) most significant: sum
s_add_u32 s1, s1, s15 // (op 643) most significant: sum


s_mov_b32 s52, s28 // (op 643) reg expression
s_mov_b32 s53, s29 // (op 643) reg expression
s_mov_b32 s54, s0 // (op 643) reg expression
s_mov_b32 s55, 131072 // (op 643) call()

v_and_b32 v12, -2, v1 // (op 734) 
v_sub_u32 v15, v1, v12 // (op 734) 

v_lshlrev_b32 v12, 4, v15 // (op 734) 

v_lshrrev_b32 v15, 1, v1 // (op 734) 
v_lshl_add_u32 v16, v15, 5, v12 // (op 734) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 734) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 786) 
v_sub_u32 v15, v1, v12 // (op 786) 

v_lshlrev_b32 v12, 4, v15 // (op 786) 

v_lshrrev_b32 v15, 3, v1 // (op 786) 
v_lshl_add_u32 v16, v15, 7, v12 // (op 786) 


v_mul_lo_u32 v12, v16, 6 // (op 786) 

v_lshrrev_b32 v15, 3, v12 // (op 786) 

v_and_b32 v12, -32, v1 // (op 839) 
v_sub_u32 v16, v1, v12 // (op 839) 

v_and_b32 v12, 4294967264, v1 // (op 839) 
v_add_u32 v17, v12, v16 // (op 839) 


s_ashr_i64 s[14:15], s[6:7], 7 // (op 162) 
s_and_b32 s1, s14, 4294967294 // (op 162) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 161) 
s_mov_b32 s18, s14 // (op 161) convert

s_cmp_lt_u32 s1, s18 // (op 161) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161:
s_barrier  // (op 1027) 
s_add_u32 s1, s1, 1 // (op 163) 

s_lshl_b64 s[14:15], s[2:3], 6 // (op 417) 
v_mov_b64 v[18:19], s[14:15] // (op 417) convert


v_mov_b64 v[22:23], v[10:11] // (op 417) 
v_mov_b64 v[24:25], v[22:23] // (op 417) reg expression
s_mov_b32 m0, s0 // (op 417) reg expression
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 417) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v18 // (op 417) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 417) most significant half

s_add_u32 m0, m0, 2048 // (op 417) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 417) Load value direct to lds



s_lshl_b64 s[14:15], s[2:3], 7 // (op 436) 
v_mov_b32 v12, s14 // (op 436) 
v_mov_b32 v16, s15 // (op 436) 
v_add_co_u32 v10, s[18:19], v10, v12 // (op 436) least significant half
v_addc_co_u32 v11, s[18:19], v11, v16, s[18:19] // (op 436) most significant half






v_mov_b32 v12, v20 // (op 179) 

buffer_load_dwordx3 v[22:24], v12, s[44:47], 0 offen // (op 179) Load value

v_add_u32 v20, 96, v20 // (op 505) 


s_barrier  // (op 1040) 
v_mov_b32 v12, v14 // (op 180) 



v_mov_b32 v16, v12 // (op 180) reg expression

s_waitcnt vmcnt(0)
ds_write_b96 v15, v[22:24] offset:8192 // (op 177) Store local data 
buffer_load_ubyte v21, v16, s[48:51], 0 offen offset: 0 // (op 180) Load value



s_lshl_b32 s14, s20, 2 // (op 569) 
v_add_u32 v14, v14, s14 // (op 569) 

s_barrier  // (op 1030) 
s_barrier  // (op 1054) 






ds_read_b64_tr_b8 v[22:23], v0 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:256 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:2048 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:2304 // (op 169) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v17, v21 offset:9728 // (op 178) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 1044) 



ds_read_b128 v[30:33], v4 offset:8192 // (op 171) Load local data 
ds_read_b64 v[34:35], v4 offset:8208 // (op 171) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1058) 


ds_read_u8 v12, v3 offset:9728 // (op 173) Load local data 


v_mov_b32 v16, v13 // (op 175) 

buffer_load_ubyte v21, v16, s[52:55], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v13, 4, v13 // (op 633) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:35], a[0:3], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 176) 




s_ashr_i64 s[14:15], s[6:7], 7 // (op 161) 
s_mov_b32 s18, s14 // (op 161) convert

s_cmp_lt_u32 s1, s18 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 if true)
RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161:






















s_barrier  // (op 1118) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

v_lshrrev_b32 v0, 6, v1 // (op 881) 
v_and_b32 v3, 63, v1 // (op 881) 
v_lshrrev_b32 v4, 4, v3 // (op 881) 
v_lshlrev_b32 v5, 2, v4 // (op 881) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 881) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 881) 

v_lshrrev_b32 v4, 2, v3 // (op 881) 

v_and_b32 v3, 3, v4 // (op 881) 

v_and_b32 v4, 3, v1 // (op 881) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 881) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 881) 

v_mul_lo_u32 v4, v3, s30 // (op 881) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 881) 


s_mul_i32 s15, s30, s13 // (op 883) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 883) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 883) least significant
s_mul_hi_u32 s1, s30, s12 // (op 883) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 883) most significant: sum
s_add_u32 s15, s15, s1 // (op 883) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 883) 

s_mov_b32 s8, s32 // (op 883) reg expression
s_mov_b32 s9, s33 // (op 883) reg expression
s_mov_b32 s10, s0 // (op 883) reg expression
s_mov_b32 s11, 131072 // (op 883) call()




buffer_store_dwordx4 v[10:13], v3, s[8:11], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123:
v_lshrrev_b32 v0, 6, v1 // (op 704) 
v_and_b32 v3, 63, v1 // (op 704) 
v_lshrrev_b32 v4, 4, v3 // (op 704) 
v_lshlrev_b32 v5, 2, v4 // (op 704) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 704) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 704) 

v_lshrrev_b32 v4, 2, v3 // (op 704) 

v_and_b32 v3, 3, v4 // (op 704) 

v_and_b32 v4, 3, v1 // (op 704) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 704) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 704) 

v_mul_lo_u32 v4, v3, s34 // (op 704) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 704) 


s_mul_i32 s15, s34, s13 // (op 706) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 706) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 706) least significant
s_mul_hi_u32 s1, s34, s12 // (op 706) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 706) most significant: sum
s_add_u32 s15, s15, s1 // (op 706) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 706) 

s_mov_b32 s4, s36 // (op 706) reg expression
s_mov_b32 s5, s37 // (op 706) reg expression
s_mov_b32 s6, s0 // (op 706) reg expression
s_mov_b32 s7, 131072 // (op 706) call()





buffer_load_dwordx4 v[4:7], v3, s[4:7], 0 offen // (op 135) Load value



v_accvgpr_read v10, a0 // (op 132) 
v_accvgpr_read v11, a1 // (op 132) 
v_accvgpr_read v12, a2 // (op 132) 
v_accvgpr_read v13, a3 // (op 132) 
v_mul_f32 v14, v9, v10 // (op 132) 
v_mul_f32 v15, v9, v11 // (op 132) 
v_mul_f32 v16, v9, v12 // (op 132) 
v_mul_f32 v17, v9, v13 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v8, v4 // (op 136) 
v_mul_f32 v11, v8, v5 // (op 136) 
v_mul_f32 v12, v8, v6 // (op 136) 
v_mul_f32 v13, v8, v7 // (op 136) 

v_add_f32 v4, v10, v14 // (op 133) 
v_add_f32 v5, v11, v15 // (op 133) 
v_add_f32 v6, v12, v16 // (op 133) 
v_add_f32 v7, v13, v17 // (op 133) 


v_lshrrev_b32 v0, 6, v1 // (op 898) 
v_and_b32 v3, 63, v1 // (op 898) 
v_lshrrev_b32 v10, 4, v3 // (op 898) 
v_lshlrev_b32 v11, 2, v10 // (op 898) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 898) 


v_lshl_add_u32 v0, s16, 5, v10 // (op 898) 

v_lshrrev_b32 v10, 2, v3 // (op 898) 

v_and_b32 v3, 3, v10 // (op 898) 

v_and_b32 v10, 3, v1 // (op 898) 
v_lshl_add_u32 v11, v3, 2, v10 // (op 898) 


v_lshl_add_u32 v3, s17, 4, v11 // (op 898) 

v_mul_lo_u32 v10, v3, s30 // (op 898) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 898) 


s_mul_i32 s15, s30, s13 // (op 900) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 900) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 900) least significant
s_mul_hi_u32 s1, s30, s12 // (op 900) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 900) most significant: sum
s_add_u32 s15, s15, s1 // (op 900) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 900) 

s_mov_b32 s8, s32 // (op 900) reg expression
s_mov_b32 s9, s33 // (op 900) reg expression
s_mov_b32 s10, s0 // (op 900) reg expression
s_mov_b32 s11, 131072 // (op 900) call()




buffer_store_dwordx4 v[4:7], v3, s[8:11], 0 offen // (op 134) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123:




s_endpgm  // End of RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.LRR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end:
.size RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2, .LRR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end-RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
  .amdhsa_next_free_vgpr 56
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 11520 // lds bytes
  .amdhsa_accum_offset 52
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: F6
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
    args: [ { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeFree1 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC } ]
    macrotile: [32, 16, 128]
    threads: [128, 1, 1]
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
  - .name: RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
    .symbol: RR_GEMM_NN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 11520
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 50
    .agpr_count: 4
    .max_flat_workgroup_size: 128
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
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 32
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
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
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
        .size: 8
        .offset: 80
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_11
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

