// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.p2align 8
.type RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2,@function
RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 831) 
v_and_b32 v3, 63, v1 // (op 831) 
v_sub_u32 v4, v3, v0 // (op 831) 

v_lshrrev_b32 v3, 1, v4 // (op 831) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 831) 


v_and_b32 v0, 1, v4 // (op 831) 

v_lshlrev_b32 v3, 3, v0 // (op 831) 

v_lshrrev_b32 v0, 6, v1 // (op 831) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 831) 


v_add_u32 v0, v5, v4 // (op 831) 


v_and_b32 v3, 15, v1 // (op 883) 
v_and_b32 v4, 48, v1 // (op 883) 
v_lshl_add_u32 v5, v3, 7, v4 // (op 883) 


v_lshrrev_b32 v3, 6, v1 // (op 936) 
v_and_b32 v4, 15, v1 // (op 936) 
v_lshl_add_u32 v6, v3, 4, v4 // (op 936) 


v_and_b32 v3, 48, v1 // (op 936) 
v_add_u32 v4, v3, v6 // (op 936) 


v_and_b32 v3, 15, v1 // (op 989) 
v_and_b32 v6, 63, v1 // (op 989) 
v_lshrrev_b32 v7, 4, v6 // (op 989) 

v_lshl_add_u32 v6, v3, 2, v7 // (op 989) 


v_and_b32 v3, 48, v1 // (op 1062) 
v_and_b32 v7, 63, v1 // (op 1062) 
v_sub_u32 v8, v7, v3 // (op 1062) 

v_lshrrev_b32 v7, 1, v8 // (op 1062) 
v_add_lshl_u32 v9, v3, v7, 4 // (op 1062) 


v_and_b32 v3, 1, v8 // (op 1062) 

v_lshlrev_b32 v7, 3, v3 // (op 1062) 

v_lshrrev_b32 v3, 6, v1 // (op 1062) 
v_lshl_add_u32 v8, v3, 4, v7 // (op 1062) 


v_add_u32 v3, v9, v8 // (op 1062) 


v_and_b32 v7, 15, v1 // (op 1101) 
v_and_b32 v8, 48, v1 // (op 1101) 
v_lshl_add_u32 v9, v7, 7, v8 // (op 1101) 


v_lshrrev_b32 v7, 6, v1 // (op 1125) 
v_and_b32 v8, 15, v1 // (op 1125) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 1125) 


v_and_b32 v7, 48, v1 // (op 1125) 
v_add_u32 v8, v7, v10 // (op 1125) 


v_and_b32 v7, 15, v1 // (op 1149) 
v_and_b32 v10, 63, v1 // (op 1149) 
v_lshrrev_b32 v11, 4, v10 // (op 1149) 

v_lshl_add_u32 v10, v7, 2, v11 // (op 1149) 


v_mov_b32 v7, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
s_lshl_b32 s0, s16, 4 // (op 542) 
v_mul_lo_u32 v13, v1, s3 // (op 542) most significant: low of low * high
s_mov_b32 s15, 0 // (op 542) convert
s_mov_b32 s14, s0 // (op 542) convert

v_mul_hi_u32 v14, v1, s2 // (op 542) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 542) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 542) most significant: sum

v_mov_b32 v14, s14 // (op 542) 
v_mov_b32 v15, s15 // (op 542) 
v_add_co_u32 v16, s[0:1], v14, v12 // (op 542) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 542) most significant half





s_mul_i32 s1, s2, s7 // (op 544) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 544) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 544) least significant
s_mul_hi_u32 s15, s2, s6 // (op 544) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 544) most significant: sum
s_add_u32 s1, s1, s15 // (op 544) most significant: sum


s_mov_b32 s40, s4 // (op 544) reg expression
s_mov_b32 s41, s5 // (op 544) reg expression
s_mov_b32 s42, s0 // (op 544) reg expression
s_mov_b32 s43, 131072 // (op 544) call()

s_lshl_b32 s0, s16, 4 // (op 561) 
v_mul_lo_u32 v13, v1, s3 // (op 561) most significant: low of low * high
v_mul_hi_u32 v14, v1, s2 // (op 561) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 561) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 561) most significant: sum

s_mov_b32 s15, 0 // (op 561) convert
s_mov_b32 s14, s0 // (op 561) convert

v_mov_b32 v14, s14 // (op 561) 
v_mov_b32 v15, s15 // (op 561) 
v_add_co_u32 v18, s[0:1], v14, v12 // (op 561) least significant half
v_addc_co_u32 v19, s[0:1], v15, v13, s[0:1] // (op 561) most significant half





s_mul_i32 s1, s2, s7 // (op 563) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 563) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 563) least significant
s_mul_hi_u32 s15, s2, s6 // (op 563) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 563) most significant: sum
s_add_u32 s1, s1, s15 // (op 563) most significant: sum


s_mov_b32 s40, s4 // (op 563) reg expression
s_mov_b32 s41, s5 // (op 563) reg expression
s_mov_b32 s42, s0 // (op 563) reg expression
s_mov_b32 s43, 131072 // (op 563) call()

v_and_b32 v12, -8, v1 // (op 606) 
v_sub_u32 v13, v1, v12 // (op 606) 

v_lshlrev_b32 v12, 4, v13 // (op 606) 

v_mov_b32 v15, 0 // (op 606) convert
v_mov_b32 v14, v12 // (op 606) convert

v_lshrrev_b32 v12, 3, v1 // (op 606) 
v_lshl_add_u32 v13, s17, 4, v12 // (op 606) 

v_mul_lo_u32 v21, v13, s9 // (op 606) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 606) most significant: high of low * low
v_mul_lo_u32 v20, v13, s8 // (op 606) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 606) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 606) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 606) most significant half



s_mul_i32 s1, s8, s13 // (op 608) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 608) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 608) least significant
s_mul_hi_u32 s15, s8, s12 // (op 608) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 608) most significant: sum
s_add_u32 s1, s1, s15 // (op 608) most significant: sum


s_mov_b32 s44, s10 // (op 608) reg expression
s_mov_b32 s45, s11 // (op 608) reg expression
s_mov_b32 s46, s0 // (op 608) reg expression
s_mov_b32 s47, 131072 // (op 608) call()

v_and_b32 v14, -8, v1 // (op 625) 
v_sub_u32 v15, v1, v14 // (op 625) 

v_lshlrev_b32 v14, 4, v15 // (op 625) 

v_mov_b32 v21, 0 // (op 625) convert
v_mov_b32 v20, v14 // (op 625) convert

v_lshrrev_b32 v14, 3, v1 // (op 625) 
v_lshl_add_u32 v15, s17, 4, v14 // (op 625) 

v_mul_lo_u32 v23, v15, s9 // (op 625) most significant: low of low * high
v_mul_hi_u32 v14, v15, s8 // (op 625) most significant: high of low * low
v_mul_lo_u32 v22, v15, s8 // (op 625) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 625) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 625) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 625) most significant half



s_mul_i32 s1, s8, s13 // (op 627) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 627) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 627) least significant
s_mul_hi_u32 s15, s8, s12 // (op 627) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 627) most significant: sum
s_add_u32 s1, s1, s15 // (op 627) most significant: sum


s_mov_b32 s44, s10 // (op 627) reg expression
s_mov_b32 s45, s11 // (op 627) reg expression
s_mov_b32 s46, s0 // (op 627) reg expression
s_mov_b32 s47, 131072 // (op 627) call()

v_and_b32 v20, -16, v1 // (op 670) 
v_sub_u32 v21, v1, v20 // (op 670) 

v_lshl_add_u32 v20, s16, 4, v21 // (op 670) 

v_lshrrev_b32 v21, 4, v1 // (op 670) 

s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v22, v21, s20 // (op 670) 

v_add_u32 v21, v20, v22 // (op 670) 


s_mul_i32 s1, s20, s25 // (op 672) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 672) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 672) least significant
s_mul_hi_u32 s15, s20, s24 // (op 672) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 672) most significant: sum
s_add_u32 s1, s1, s15 // (op 672) most significant: sum


s_mov_b32 s48, s22 // (op 672) reg expression
s_mov_b32 s49, s23 // (op 672) reg expression
s_mov_b32 s50, s0 // (op 672) reg expression
s_mov_b32 s51, 131072 // (op 672) call()

v_and_b32 v20, -16, v1 // (op 689) 
v_sub_u32 v22, v1, v20 // (op 689) 

v_lshl_add_u32 v20, s16, 4, v22 // (op 689) 

v_lshrrev_b32 v22, 4, v1 // (op 689) 
v_mul_lo_u32 v23, v22, s20 // (op 689) 

v_add_u32 v22, v20, v23 // (op 689) 


s_mul_i32 s1, s20, s25 // (op 691) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 691) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 691) least significant
s_mul_hi_u32 s15, s20, s24 // (op 691) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 691) most significant: sum
s_add_u32 s1, s1, s15 // (op 691) most significant: sum


s_mov_b32 s48, s22 // (op 691) reg expression
s_mov_b32 s49, s23 // (op 691) reg expression
s_mov_b32 s50, s0 // (op 691) reg expression
s_mov_b32 s51, 131072 // (op 691) call()

v_and_b32 v20, -4, v1 // (op 734) 
v_sub_u32 v23, v1, v20 // (op 734) 

v_lshrrev_b32 v20, 2, v1 // (op 734) 
v_lshl_add_u32 v24, s17, 4, v20 // (op 734) 

v_mul_lo_u32 v20, v24, s26 // (op 734) 

v_add_u32 v24, v23, v20 // (op 734) 


s_mul_i32 s1, s26, s13 // (op 736) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 736) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 736) least significant
s_mul_hi_u32 s15, s26, s12 // (op 736) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 736) most significant: sum
s_add_u32 s1, s1, s15 // (op 736) most significant: sum


s_mov_b32 s52, s28 // (op 736) reg expression
s_mov_b32 s53, s29 // (op 736) reg expression
s_mov_b32 s54, s0 // (op 736) reg expression
s_mov_b32 s55, 131072 // (op 736) call()

v_and_b32 v20, -4, v1 // (op 753) 
v_sub_u32 v23, v1, v20 // (op 753) 

v_lshrrev_b32 v20, 2, v1 // (op 753) 
v_lshl_add_u32 v25, s17, 4, v20 // (op 753) 

v_mul_lo_u32 v20, v25, s26 // (op 753) 

v_add_u32 v25, v23, v20 // (op 753) 


s_mul_i32 s1, s26, s13 // (op 755) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 755) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 755) least significant
s_mul_hi_u32 s15, s26, s12 // (op 755) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 755) most significant: sum
s_add_u32 s1, s1, s15 // (op 755) most significant: sum


s_mov_b32 s52, s28 // (op 755) reg expression
s_mov_b32 s53, s29 // (op 755) reg expression
s_mov_b32 s54, s0 // (op 755) reg expression
s_mov_b32 s55, 131072 // (op 755) call()

v_and_b32 v20, -8, v1 // (op 871) 
v_sub_u32 v23, v1, v20 // (op 871) 

v_lshlrev_b32 v20, 4, v23 // (op 871) 

v_lshrrev_b32 v23, 3, v1 // (op 871) 
v_lshl_add_u32 v26, v23, 7, v20 // (op 871) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 871) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 3 // (op 491) 
v_mov_b64 v[26:27], s[14:15] // (op 491) convert


v_mov_b64 v[28:29], v[12:13] // (op 491) 
s_mov_b32 m0, s0 // (op 491) reg expression
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 491) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v26 // (op 491) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 491) most significant half

s_add_u32 m0, m0, 1024 // (op 491) 
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 491) Load value direct to lds






buffer_load_ubyte v20, v24, s[52:55], 0 offen offset: 0 // (op 373) Load value
v_lshlrev_b32 v23, 4, v1 // (op 819) 
s_nop 0
v_readfirstlane_b32 s0, v23 // (op 819) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 496) 
v_mov_b64 v[26:27], s[14:15] // (op 496) convert


v_mov_b64 v[28:29], v[16:17] // (op 496) reg expression
s_add_u32 s1, 2048, s0 // (op 496) 
s_mov_b32 m0, s1 // (op 496) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 496) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v26 // (op 496) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 496) most significant half

s_add_u32 m0, m0, 1024 // (op 496) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 496) Load value direct to lds






v_mov_b32 v23, v21 // (op 379) reg expression

buffer_load_ubyte v26, v23, s[48:51], 0 offen offset: 0 // (op 379) Load value

v_and_b32 v23, -8, v1 // (op 1089) 
v_sub_u32 v27, v1, v23 // (op 1089) 

v_lshlrev_b32 v23, 4, v27 // (op 1089) 

v_lshrrev_b32 v27, 3, v1 // (op 1089) 
v_lshl_add_u32 v28, v27, 7, v23 // (op 1089) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1089) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 3 // (op 501) 
v_mov_b64 v[28:29], s[14:15] // (op 501) convert


v_mov_b64 v[30:31], v[12:13] // (op 501) 
s_add_u32 s1, 4096, s0 // (op 501) 
s_mov_b32 m0, s1 // (op 501) reg expression
s_mov_b32 s14, 128 // (op 501) call()
buffer_load_dwordx4 v30, s[44:47], s14 offen offset: 0 lds // (op 501) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 501) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 501) most significant half

s_add_u32 m0, m0, 1024 // (op 501) 
buffer_load_dwordx4 v30, s[44:47], s14 offen offset: 0 lds // (op 501) Load value direct to lds









buffer_load_ubyte v23, v24, s[52:55], 0 offen offset:4 // (op 385) Load value

v_lshlrev_b32 v24, 4, v1 // (op 1050) 
s_nop 0
v_readfirstlane_b32 s0, v24 // (op 1050) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 503) 
v_mov_b64 v[12:13], s[14:15] // (op 503) convert


s_lshl_b64 s[14:15], s[2:3], 7 // (op 503) 
v_mov_b32 v24, s14 // (op 503) 
v_mov_b32 v27, s15 // (op 503) 
v_add_co_u32 v28, s[18:19], v16, v24 // (op 503) least significant half
v_addc_co_u32 v29, s[18:19], v17, v27, s[18:19] // (op 503) most significant half




s_add_u32 s1, 6144, s0 // (op 503) 
s_mov_b32 m0, s1 // (op 503) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 503) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v12 // (op 503) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 503) most significant half

s_add_u32 m0, m0, 1024 // (op 503) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 503) Load value direct to lds







s_lshl_b32 s0, s20, 2 // (op 391) 
v_add_u32 v24, v21, s0 // (op 391) 


buffer_load_ubyte v12, v24, s[48:51], 0 offen offset: 0 // (op 391) Load value


v_and_b32 v13, -4, v1 // (op 977) 
v_sub_u32 v21, v1, v13 // (op 977) 

v_and_b32 v13, 4294967292, v1 // (op 977) 
v_add_u32 v24, v13, v21 // (op 977) 





s_waitcnt vmcnt(9)
ds_write_b8 v24, v20 offset:8192 // (op 394) Store local data 

v_and_b32 v13, -16, v1 // (op 924) 
v_sub_u32 v21, v1, v13 // (op 924) 

v_and_b32 v13, 4294967280, v1 // (op 924) 
v_add_u32 v24, v13, v21 // (op 924) 





s_waitcnt vmcnt(6)
ds_write_b8 v24, v26 offset:8256 // (op 396) Store local data 

s_barrier  // (op 1207) 



ds_read_b128 v[28:31], v5 // (op 413) Load local data 
ds_read_b128 v[32:35], v5 offset:64 // (op 413) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1241) 



ds_read_u8 v13, v6 offset:8192 // (op 419) Load local data 
s_barrier  // (op 1191) 



ds_read_b64_tr_b8 v[36:37], v0 offset:2048 // (op 410) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:2176 // (op 410) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:3072 // (op 410) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:3200 // (op 410) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1224) 


ds_read_u8 v21, v4 offset:8256 // (op 416) Load local data 
v_lshlrev_b32 v24, 4, v1 // (op 791) 
s_nop 0
v_readfirstlane_b32 s0, v24 // (op 791) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -8, v1 // (op 843) 
v_sub_u32 v27, v1, v24 // (op 843) 

v_lshlrev_b32 v24, 4, v27 // (op 843) 

v_lshrrev_b32 v27, 3, v1 // (op 843) 
v_lshl_add_u32 v16, v27, 7, v24 // (op 843) 


s_nop 0
v_readfirstlane_b32 s1, v16 // (op 843) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 895) 
v_sub_u32 v27, v1, v24 // (op 895) 

v_and_b32 v24, 4294967280, v1 // (op 895) 
v_add_u32 v16, v24, v27 // (op 895) 


v_and_b32 v17, -4, v1 // (op 948) 
v_sub_u32 v24, v1, v17 // (op 948) 

v_and_b32 v17, 4294967292, v1 // (op 948) 
v_add_u32 v27, v17, v24 // (op 948) 


v_lshlrev_b32 v17, 4, v1 // (op 1035) 
s_nop 0
v_readfirstlane_b32 s14, v17 // (op 1035) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -8, v1 // (op 1074) 
v_sub_u32 v24, v1, v17 // (op 1074) 

v_lshlrev_b32 v17, 4, v24 // (op 1074) 

v_lshrrev_b32 v24, 3, v1 // (op 1074) 
v_lshl_add_u32 v44, v24, 7, v17 // (op 1074) 


s_nop 0
v_readfirstlane_b32 s15, v44 // (op 1074) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -16, v1 // (op 1113) 
v_sub_u32 v24, v1, v17 // (op 1113) 

v_and_b32 v17, 4294967280, v1 // (op 1113) 
v_add_u32 v44, v17, v24 // (op 1113) 


v_and_b32 v17, -4, v1 // (op 1137) 
v_sub_u32 v24, v1, v17 // (op 1137) 

v_and_b32 v17, 4294967292, v1 // (op 1137) 
v_add_u32 v45, v17, v24 // (op 1137) 


s_mov_b32 s18, 0 // (op 61) call()
s_ashr_i64 s[38:39], s[6:7], 7 // (op 60) 
s_mov_b32 s19, s38 // (op 60) convert

s_and_b32 s38, s19, 4294967294 // (op 60) 

s_cmp_lt_u32 s18, s38 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60:
s_add_u32 s18, s18, 2 // (op 62) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[28:35], a[0:3], v21, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 90) 
s_barrier  // (op 1210) 

s_lshl_b64 s[38:39], s[8:9], 3 // (op 463) 
v_mov_b64 v[46:47], s[38:39] // (op 463) convert


v_mov_b64 v[48:49], v[14:15] // (op 463) 
v_mov_b64 v[50:51], v[48:49] // (op 463) 
s_mov_b32 m0, s1 // (op 463) reg expression
s_mov_b32 s19, 256 // (op 463) call()
buffer_load_dwordx4 v50, s[44:47], s19 offen offset: 0 lds // (op 463) Load value direct to lds
v_add_co_u32 v50, s[38:39], v50, v46 // (op 463) least significant half
v_addc_co_u32 v51, s[38:39], 0, v51, s[38:39] // (op 463) most significant half

s_add_u32 m0, m0, 1024 // (op 463) 
buffer_load_dwordx4 v50, s[44:47], s19 offen offset: 0 lds // (op 463) Load value direct to lds





v_mov_b32 v17, v25 // (op 25) 

buffer_load_ubyte v20, v17, s[52:55], 0 offen offset:8 // (op 25) Load value
s_barrier  // (op 1194) 

s_lshl_b64 s[38:39], s[2:3], 6 // (op 458) 
v_mov_b64 v[46:47], s[38:39] // (op 458) convert


v_mov_b64 v[50:51], v[18:19] // (op 458) 
s_lshl_b64 s[38:39], s[2:3], 7 // (op 458) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 458) 

v_mov_b32 v24, s56 // (op 458) 
v_mov_b32 v52, s57 // (op 458) 
v_add_co_u32 v54, s[38:39], v50, v24 // (op 458) least significant half
v_addc_co_u32 v55, s[38:39], v51, v52, s[38:39] // (op 458) most significant half




s_add_u32 s19, 2048, s0 // (op 458) 
s_mov_b32 m0, s19 // (op 458) reg expression
buffer_load_dwordx4 v54, s[40:43], 0 offen offset: 0 lds // (op 458) Load value direct to lds
v_add_co_u32 v54, s[38:39], v54, v46 // (op 458) least significant half
v_addc_co_u32 v55, s[38:39], 0, v55, s[38:39] // (op 458) most significant half

s_add_u32 m0, m0, 1024 // (op 458) 
buffer_load_dwordx4 v54, s[40:43], 0 offen offset: 0 lds // (op 458) Load value direct to lds





v_mov_b32 v24, v22 // (op 17) 
s_lshl_b32 s19, s20, 3 // (op 17) 
v_add_u32 v46, v24, s19 // (op 17) 


buffer_load_ubyte v26, v46, s[48:51], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1280) 



s_waitcnt vmcnt(9)
ds_write_b8 v45, v23 offset:8320 // (op 259) Store local data 
s_barrier  // (op 1274) 



s_waitcnt vmcnt(6)
ds_write_b8 v44, v12 offset:8384 // (op 258) Store local data 
s_barrier  // (op 1264) 



ds_read_b128 v[28:31], v9 offset:4096 // (op 251) Load local data 
ds_read_b128 v[32:35], v9 offset:4160 // (op 251) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1277) 



ds_read_u8 v13, v10 offset:8320 // (op 255) Load local data 
s_barrier  // (op 1258) 
s_barrier  // (op 1335) 



ds_read_b64_tr_b8 v[36:37], v3 offset:6144 // (op 249) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v3 offset:6272 // (op 249) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v3 offset:7168 // (op 249) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v3 offset:7296 // (op 249) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1271) 


ds_read_u8 v21, v8 offset:8384 // (op 253) Load local data 
s_barrier  // (op 1329) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[28:35], a[0:3], v21, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 256) 
s_barrier  // (op 1268) 

s_lshl_b64 s[38:39], s[8:9], 3 // (op 486) 
v_mov_b64 v[46:47], s[38:39] // (op 486) convert


v_mov_b64 v[52:53], v[48:49] // (op 486) 
s_add_u32 s19, 4096, s15 // (op 486) 
s_mov_b32 m0, s19 // (op 486) reg expression
s_mov_b32 s38, 384 // (op 486) call()
buffer_load_dwordx4 v52, s[44:47], s38 offen offset: 0 lds // (op 486) Load value direct to lds
v_add_co_u32 v52, s[56:57], v52, v46 // (op 486) least significant half
v_addc_co_u32 v53, s[56:57], 0, v53, s[56:57] // (op 486) most significant half

s_add_u32 m0, m0, 1024 // (op 486) 
buffer_load_dwordx4 v52, s[44:47], s38 offen offset: 0 lds // (op 486) Load value direct to lds








v_add_co_u32 v14, vcc, 256, v14 // (op 617) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 617) most significant half
buffer_load_ubyte v23, v17, s[52:55], 0 offen offset:12 // (op 262) Load value

s_barrier  // (op 1261) 
v_add_u32 v25, 8, v25 // (op 745) 

s_lshl_b64 s[38:39], s[2:3], 6 // (op 481) 
v_mov_b64 v[46:47], s[38:39] // (op 481) convert


s_lshl_b64 s[38:39], s[2:3], 7 // (op 481) 
s_mul_i32 s57, 3, s39 // (op 481) most significant: low of low * high
s_mul_i32 s56, 3, s38 // (op 481) least significant
s_mul_hi_u32 s19, 3, s38 // (op 481) most significant: high of low * low
s_add_u32 s57, s57, s19 // (op 481) most significant: sum


v_mov_b32 v17, s56 // (op 481) 
v_mov_b32 v48, s57 // (op 481) 
v_add_co_u32 v52, s[38:39], v50, v17 // (op 481) least significant half
v_addc_co_u32 v53, s[38:39], v51, v48, s[38:39] // (op 481) most significant half




s_add_u32 s19, 6144, s14 // (op 481) 
s_mov_b32 m0, s19 // (op 481) reg expression
buffer_load_dwordx4 v52, s[40:43], 0 offen offset: 0 lds // (op 481) Load value direct to lds
v_add_co_u32 v52, s[38:39], v52, v46 // (op 481) least significant half
v_addc_co_u32 v53, s[38:39], 0, v53, s[38:39] // (op 481) most significant half

s_add_u32 m0, m0, 1024 // (op 481) 
buffer_load_dwordx4 v52, s[40:43], 0 offen offset: 0 lds // (op 481) Load value direct to lds






s_lshl_b32 s19, s20, 2 // (op 261) 
s_mul_i32 s38, 3, s19 // (op 261) 

v_add_u32 v17, v24, s38 // (op 261) 


buffer_load_ubyte v12, v17, s[48:51], 0 offen offset: 0 // (op 261) Load value

s_lshl_b64 s[38:39], s[2:3], 7 // (op 553) 

s_lshl_b64 s[56:57], s[38:39], 1 // (op 553) 

s_barrier  // (op 1244) 
s_lshl_b32 s19, s20, 3 // (op 681) 



v_add_u32 v22, v22, s19 // (op 681) 

v_mov_b32 v17, s56 // (op 553) 
v_mov_b32 v24, s57 // (op 553) 
v_add_co_u32 v18, s[38:39], v18, v17 // (op 553) least significant half
v_addc_co_u32 v19, s[38:39], v19, v24, s[38:39] // (op 553) most significant half




s_waitcnt vmcnt(9)
ds_write_b8 v27, v20 offset:8192 // (op 30) Store local data 
s_barrier  // (op 1227) 



s_waitcnt vmcnt(6)
ds_write_b8 v16, v26 offset:8256 // (op 21) Store local data 
s_barrier  // (op 1216) 



ds_read_b128 v[28:31], v5 // (op 77) Load local data 
ds_read_b128 v[32:35], v5 offset:64 // (op 77) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1251) 



ds_read_u8 v13, v6 offset:8192 // (op 87) Load local data 
s_barrier  // (op 1200) 



ds_read_b64_tr_b8 v[36:37], v0 offset:2048 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:2176 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:3072 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:3200 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1234) 


ds_read_u8 v21, v4 offset:8256 // (op 82) Load local data 
s_ashr_i64 s[38:39], s[6:7], 7 // (op 60) 
s_mov_b32 s19, s38 // (op 60) convert

s_and_b32 s38, s19, 4294967294 // (op 60) 

s_cmp_lt_u32 s18, s38 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60:















s_barrier  // (op 1293) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 516) 
s_lshr_b32 s14, s0, 1 // (op 516) 

v_lshl_add_u32 v22, s14, 8, v1 // (op 516) 

s_lshl_b32 s0, s16, 4 // (op 516) 
v_mul_lo_u32 v15, v22, s3 // (op 516) most significant: low of low * high
v_mul_hi_u32 v16, v22, s2 // (op 516) most significant: high of low * low
v_mul_lo_u32 v14, v22, s2 // (op 516) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 516) most significant: sum


s_mov_b32 s15, 0 // (op 516) convert
s_mov_b32 s14, s0 // (op 516) convert

v_mov_b32 v22, s14 // (op 516) 
v_mov_b32 v16, s15 // (op 516) 
v_add_co_u32 v24, s[0:1], v22, v14 // (op 516) least significant half
v_addc_co_u32 v25, s[0:1], v16, v15, s[0:1] // (op 516) most significant half





s_mul_i32 s1, s2, s7 // (op 518) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 518) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 518) least significant
s_mul_hi_u32 s15, s2, s6 // (op 518) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 518) most significant: sum
s_add_u32 s1, s1, s15 // (op 518) most significant: sum


s_mov_b32 s40, s4 // (op 518) reg expression
s_mov_b32 s41, s5 // (op 518) reg expression
s_mov_b32 s42, s0 // (op 518) reg expression
s_mov_b32 s43, 131072 // (op 518) call()

v_and_b32 v22, -8, v1 // (op 583) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 583) 
v_sub_u32 v14, v1, v22 // (op 583) 

s_lshr_b32 s14, s0, 1 // (op 583) 

v_lshlrev_b32 v22, 4, v14 // (op 583) 

v_lshl_add_u32 v14, s14, 8, v22 // (op 583) 


v_mov_b32 v27, 0 // (op 583) convert
v_mov_b32 v26, v14 // (op 583) convert

v_lshrrev_b32 v22, 3, v1 // (op 583) 
v_lshl_add_u32 v14, s17, 4, v22 // (op 583) 

v_mul_lo_u32 v17, v14, s9 // (op 583) most significant: low of low * high
v_mul_hi_u32 v15, v14, s8 // (op 583) most significant: high of low * low
v_mul_lo_u32 v16, v14, s8 // (op 583) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 583) most significant: sum


v_add_co_u32 v14, s[0:1], v26, v16 // (op 583) least significant half
v_addc_co_u32 v15, s[0:1], v27, v17, s[0:1] // (op 583) most significant half



s_mul_i32 s1, s8, s13 // (op 585) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 585) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 585) least significant
s_mul_hi_u32 s15, s8, s12 // (op 585) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 585) most significant: sum
s_add_u32 s1, s1, s15 // (op 585) most significant: sum


s_mov_b32 s44, s10 // (op 585) reg expression
s_mov_b32 s45, s11 // (op 585) reg expression
s_mov_b32 s46, s0 // (op 585) reg expression
s_mov_b32 s47, 131072 // (op 585) call()

v_and_b32 v22, -16, v1 // (op 647) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 647) 
v_sub_u32 v16, v1, v22 // (op 647) 

s_lshr_b32 s14, s0, 1 // (op 647) 

v_lshl_add_u32 v22, s16, 4, v16 // (op 647) 

v_lshrrev_b32 v16, 4, v1 // (op 647) 
v_lshl_add_u32 v17, s14, 3, v16 // (op 647) 


v_mul_lo_u32 v16, v17, s20 // (op 647) 

v_add_u32 v17, v22, v16 // (op 647) 


s_mul_i32 s1, s20, s25 // (op 649) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 649) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 649) least significant
s_mul_hi_u32 s15, s20, s24 // (op 649) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 649) most significant: sum
s_add_u32 s1, s1, s15 // (op 649) most significant: sum


s_mov_b32 s48, s22 // (op 649) reg expression
s_mov_b32 s49, s23 // (op 649) reg expression
s_mov_b32 s50, s0 // (op 649) reg expression
s_mov_b32 s51, 131072 // (op 649) call()

v_and_b32 v16, -4, v1 // (op 711) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 711) 
v_sub_u32 v22, v1, v16 // (op 711) 

s_lshr_b32 s14, s0, 1 // (op 711) 

v_lshrrev_b32 v16, 2, v1 // (op 711) 
v_lshl_add_u32 v18, s14, 3, v22 // (op 711) 


v_lshl_add_u32 v22, s17, 4, v16 // (op 711) 

v_mul_lo_u32 v16, v22, s26 // (op 711) 

v_add_u32 v22, v18, v16 // (op 711) 


s_mul_i32 s1, s26, s13 // (op 713) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 713) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 713) least significant
s_mul_hi_u32 s15, s26, s12 // (op 713) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 713) most significant: sum
s_add_u32 s1, s1, s15 // (op 713) most significant: sum


s_mov_b32 s52, s28 // (op 713) reg expression
s_mov_b32 s53, s29 // (op 713) reg expression
s_mov_b32 s54, s0 // (op 713) reg expression
s_mov_b32 s55, 131072 // (op 713) call()

v_lshlrev_b32 v16, 4, v1 // (op 804) 
s_nop 0
v_readfirstlane_b32 s0, v16 // (op 804) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 856) 
v_sub_u32 v18, v1, v16 // (op 856) 

v_lshlrev_b32 v16, 4, v18 // (op 856) 

v_lshrrev_b32 v18, 3, v1 // (op 856) 
v_lshl_add_u32 v19, v18, 7, v16 // (op 856) 


s_nop 0
v_readfirstlane_b32 s1, v19 // (op 856) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -16, v1 // (op 908) 
v_sub_u32 v18, v1, v16 // (op 908) 

v_and_b32 v16, 4294967280, v1 // (op 908) 
v_add_u32 v19, v16, v18 // (op 908) 


v_and_b32 v16, -4, v1 // (op 961) 
v_sub_u32 v18, v1, v16 // (op 961) 

v_and_b32 v16, 4294967292, v1 // (op 961) 
v_add_u32 v20, v16, v18 // (op 961) 


s_ashr_i64 s[14:15], s[6:7], 7 // (op 176) 
s_and_b32 s18, s14, 4294967294 // (op 176) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 175) 
s_mov_b32 s19, s14 // (op 175) convert

s_cmp_lt_u32 s18, s19 // (op 175) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_175 // (op 175) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_175 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_175:
s_barrier  // (op 1197) 
s_add_u32 s18, s18, 1 // (op 177) 

s_lshl_b64 s[14:15], s[2:3], 6 // (op 468) 
v_mov_b64 v[12:13], s[14:15] // (op 468) convert


v_mov_b64 v[26:27], v[24:25] // (op 468) 
v_mov_b64 v[28:29], v[26:27] // (op 468) reg expression
s_add_u32 s19, 2048, s0 // (op 468) 
s_mov_b32 m0, s19 // (op 468) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 468) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v12 // (op 468) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 468) most significant half

s_add_u32 m0, m0, 1024 // (op 468) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 468) Load value direct to lds




s_lshl_b64 s[14:15], s[2:3], 7 // (op 506) 
v_mov_b32 v16, s14 // (op 506) 
v_mov_b32 v18, s15 // (op 506) 
v_add_co_u32 v24, s[38:39], v24, v16 // (op 506) least significant half
v_addc_co_u32 v25, s[38:39], v25, v18, s[38:39] // (op 506) most significant half




s_barrier  // (op 1213) 

s_lshl_b64 s[14:15], s[8:9], 3 // (op 474) 
v_mov_b64 v[12:13], s[14:15] // (op 474) convert


v_mov_b64 v[26:27], v[14:15] // (op 474) 
v_mov_b64 v[28:29], v[26:27] // (op 474) 
s_mov_b32 m0, s1 // (op 474) reg expression
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 474) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v12 // (op 474) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 474) most significant half

s_add_u32 m0, m0, 1024 // (op 474) 
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 474) Load value direct to lds



v_add_co_u32 v14, vcc, 128, v14 // (op 575) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 575) most significant half


v_mov_b32 v16, v17 // (op 195) 
v_mov_b32 v18, v16 // (op 195) reg expression

buffer_load_ubyte v21, v18, s[48:51], 0 offen offset: 0 // (op 195) Load value


s_lshl_b32 s19, s20, 2 // (op 639) 
v_add_u32 v17, v17, s19 // (op 639) 



s_barrier  // (op 1230) 
v_mov_b32 v16, v22 // (op 196) 




buffer_load_ubyte v18, v16, s[52:55], 0 offen offset: 0 // (op 196) Load value

v_add_u32 v22, 4, v22 // (op 703) 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v21 offset:8256 // (op 192) Store local data 

s_barrier  // (op 1203) 



s_barrier  // (op 1247) 
ds_read_b64_tr_b8 v[26:27], v0 offset:2048 // (op 183) Transpose load local data 



ds_read_b64_tr_b8 v[28:29], v0 offset:2176 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:3072 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:3200 // (op 183) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v18 offset:8192 // (op 193) Store local data 

s_barrier  // (op 1220) 



ds_read_b128 v[34:37], v5 // (op 185) Load local data 
ds_read_b128 v[38:41], v5 offset:64 // (op 185) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1237) 


ds_read_u8 v16, v4 offset:8256 // (op 187) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1254) 



ds_read_u8 v18, v6 offset:8192 // (op 189) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v16, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 190) 




s_ashr_i64 s[14:15], s[6:7], 7 // (op 175) 
s_mov_b32 s19, s14 // (op 175) convert

s_cmp_lt_u32 s18, s19 // (op 175) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_175 // (op 175) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_175 if true)
RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_175:























s_barrier  // (op 1321) 








v_cmp_eq_f32 vcc, 0.00000, v7 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v11, v12 // (op 50) 
v_mul_f32 v17, v11, v13 // (op 50) 
v_mul_f32 v18, v11, v14 // (op 50) 
v_mul_f32 v19, v11, v15 // (op 50) 

v_lshrrev_b32 v12, 16, v16 // (op 56) 
v_lshrrev_b32 v13, 16, v17 // (op 56) 
v_lshrrev_b32 v14, 16, v18 // (op 56) 
v_lshrrev_b32 v15, 16, v19 // (op 56) 

v_lshrrev_b32 v0, 6, v1 // (op 1003) 
v_and_b32 v3, 63, v1 // (op 1003) 
v_lshrrev_b32 v4, 4, v3 // (op 1003) 
v_lshlrev_b32 v5, 2, v4 // (op 1003) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1003) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 1003) 

v_lshrrev_b32 v4, 2, v3 // (op 1003) 

v_and_b32 v3, 3, v4 // (op 1003) 

v_and_b32 v4, 3, v1 // (op 1003) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 1003) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 1003) 

v_mul_lo_u32 v3, v6, s30 // (op 1003) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 1003) 


s_mul_i32 s15, s30, s13 // (op 1005) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 1005) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1005) least significant
s_mul_hi_u32 s1, s30, s12 // (op 1005) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1005) most significant: sum
s_add_u32 s15, s15, s1 // (op 1005) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 1005) 

s_mov_b32 s8, s32 // (op 1005) reg expression
s_mov_b32 s9, s33 // (op 1005) reg expression
s_mov_b32 s10, s0 // (op 1005) reg expression
s_mov_b32 s11, 131072 // (op 1005) call()


v_pack_b32_f16 v8, v12, v13 // (op 58) 
v_pack_b32_f16 v9, v14, v15 // (op 58) 


buffer_store_dwordx2 v[8:9], v4, s[8:11], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135:
v_lshrrev_b32 v0, 6, v1 // (op 774) 
v_and_b32 v3, 63, v1 // (op 774) 
v_lshrrev_b32 v4, 4, v3 // (op 774) 
v_lshlrev_b32 v5, 2, v4 // (op 774) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 774) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 774) 

v_lshrrev_b32 v4, 2, v3 // (op 774) 

v_and_b32 v3, 3, v4 // (op 774) 

v_and_b32 v4, 3, v1 // (op 774) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 774) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 774) 

v_mul_lo_u32 v3, v6, s34 // (op 774) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 774) 


s_mul_i32 s15, s34, s13 // (op 776) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 776) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 776) least significant
s_mul_hi_u32 s1, s34, s12 // (op 776) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 776) most significant: sum
s_add_u32 s15, s15, s1 // (op 776) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 776) 

s_mov_b32 s4, s36 // (op 776) reg expression
s_mov_b32 s5, s37 // (op 776) reg expression
s_mov_b32 s6, s0 // (op 776) reg expression
s_mov_b32 s7, 131072 // (op 776) call()





buffer_load_dwordx2 v[8:9], v4, s[4:7], 0 offen // (op 148) Load value



v_accvgpr_read v12, a0 // (op 144) 
v_accvgpr_read v13, a1 // (op 144) 
v_accvgpr_read v14, a2 // (op 144) 
v_accvgpr_read v15, a3 // (op 144) 
v_mul_f32 v16, v11, v12 // (op 144) 
v_mul_f32 v17, v11, v13 // (op 144) 
v_mul_f32 v18, v11, v14 // (op 144) 
v_mul_f32 v19, v11, v15 // (op 144) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v8 // (op 149) 
v_lshrrev_b32 v13, 16, v8 // (op 149) 
v_lshlrev_b32 v12, 16, v12 // (op 149) 
v_lshlrev_b32 v13, 16, v13 // (op 149) 
v_mul_f32 v20, v7, v12 // (op 149) 
v_mul_f32 v21, v7, v13 // (op 149) 
v_and_b32 v12, 65535, v9 // (op 149) 
v_lshrrev_b32 v13, 16, v9 // (op 149) 
v_lshlrev_b32 v12, 16, v12 // (op 149) 
v_lshlrev_b32 v13, 16, v13 // (op 149) 
v_mul_f32 v22, v7, v12 // (op 149) 
v_mul_f32 v23, v7, v13 // (op 149) 


v_add_f32 v12, v20, v16 // (op 145) 
v_add_f32 v13, v21, v17 // (op 145) 
v_add_f32 v14, v22, v18 // (op 145) 
v_add_f32 v15, v23, v19 // (op 145) 


v_lshrrev_b32 v16, 16, v12 // (op 146) 
v_lshrrev_b32 v17, 16, v13 // (op 146) 
v_lshrrev_b32 v18, 16, v14 // (op 146) 
v_lshrrev_b32 v19, 16, v15 // (op 146) 

v_lshrrev_b32 v0, 6, v1 // (op 1020) 
v_and_b32 v3, 63, v1 // (op 1020) 
v_lshrrev_b32 v4, 4, v3 // (op 1020) 
v_lshlrev_b32 v5, 2, v4 // (op 1020) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1020) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 1020) 

v_lshrrev_b32 v4, 2, v3 // (op 1020) 

v_and_b32 v3, 3, v4 // (op 1020) 

v_and_b32 v4, 3, v1 // (op 1020) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 1020) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 1020) 

v_mul_lo_u32 v3, v6, s30 // (op 1020) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 1020) 


s_mul_i32 s15, s30, s13 // (op 1022) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 1022) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1022) least significant
s_mul_hi_u32 s1, s30, s12 // (op 1022) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1022) most significant: sum
s_add_u32 s15, s15, s1 // (op 1022) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 1022) 

s_mov_b32 s8, s32 // (op 1022) reg expression
s_mov_b32 s9, s33 // (op 1022) reg expression
s_mov_b32 s10, s0 // (op 1022) reg expression
s_mov_b32 s11, 131072 // (op 1022) call()


v_pack_b32_f16 v8, v16, v17 // (op 147) 
v_pack_b32_f16 v9, v18, v19 // (op 147) 


buffer_store_dwordx2 v[8:9], v4, s[8:11], 0 offen // (op 147) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135:




s_endpgm  // End of RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.LRR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_end:
.size RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2, .LRR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_end-RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
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
    DataTypeB: B8
    DestDataType: b
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
  - .name: RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
    .symbol: RR_GEMM_NN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 8448
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 58
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

