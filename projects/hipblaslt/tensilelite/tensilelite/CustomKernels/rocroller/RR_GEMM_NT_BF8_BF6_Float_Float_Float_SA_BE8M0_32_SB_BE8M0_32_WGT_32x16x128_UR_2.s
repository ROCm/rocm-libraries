// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.p2align 8
.type RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2,@function
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_exec_begin:


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


v_and_b32 v3, 63, v1 // (op 814) 
v_and_b32 v4, 48, v1 // (op 814) 
v_sub_u32 v5, v3, v4 // (op 814) 

v_lshrrev_b32 v4, 4, v3 // (op 814) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 814) 


v_lshlrev_b32 v4, 4, v3 // (op 814) 
v_mul_lo_u32 v5, v4, 6 // (op 814) 

v_lshrrev_b32 v4, 3, v5 // (op 814) 

v_and_b32 v5, 268435455, v3 // (op 814) 

v_lshlrev_b32 v3, 2, v5 // (op 814) 

v_add_u32 v5, v4, v3 // (op 814) 


v_lshrrev_b32 v3, 6, v1 // (op 867) 
v_and_b32 v4, 15, v1 // (op 867) 
v_lshl_add_u32 v6, v3, 4, v4 // (op 867) 


v_and_b32 v3, 63, v1 // (op 867) 
v_lshrrev_b32 v4, 4, v3 // (op 867) 

v_lshl_add_u32 v3, v4, 5, v6 // (op 867) 


v_and_b32 v4, 48, v1 // (op 940) 
v_and_b32 v6, 63, v1 // (op 940) 
v_sub_u32 v7, v6, v4 // (op 940) 

v_lshrrev_b32 v6, 1, v7 // (op 940) 
v_add_lshl_u32 v8, v4, v6, 5 // (op 940) 


v_and_b32 v4, 1, v7 // (op 940) 

v_lshlrev_b32 v6, 3, v4 // (op 940) 

v_lshrrev_b32 v4, 6, v1 // (op 940) 
v_lshl_add_u32 v7, v4, 4, v6 // (op 940) 


v_add_u32 v4, v8, v7 // (op 940) 


v_and_b32 v6, 63, v1 // (op 964) 
v_and_b32 v7, 48, v1 // (op 964) 
v_sub_u32 v8, v6, v7 // (op 964) 

v_lshrrev_b32 v7, 4, v6 // (op 964) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 964) 


v_lshlrev_b32 v7, 4, v6 // (op 964) 
v_mul_lo_u32 v8, v7, 6 // (op 964) 

v_lshrrev_b32 v7, 3, v8 // (op 964) 

v_and_b32 v8, 268435455, v6 // (op 964) 

v_lshlrev_b32 v6, 2, v8 // (op 964) 

v_add_u32 v8, v7, v6 // (op 964) 


v_lshrrev_b32 v6, 6, v1 // (op 988) 
v_and_b32 v7, 15, v1 // (op 988) 
v_lshl_add_u32 v9, v6, 4, v7 // (op 988) 


v_and_b32 v6, 63, v1 // (op 988) 
v_lshrrev_b32 v7, 4, v6 // (op 988) 

v_lshl_add_u32 v6, v7, 5, v9 // (op 988) 


v_mov_b32 v7, s14 // (op 2) Move value
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

s_lshl_b32 s0, s17, 4 // (op 536) 
s_mov_b32 s15, 0 // (op 536) convert
s_mov_b32 s14, s0 // (op 536) convert

v_mul_lo_u32 v15, v1, s9 // (op 536) most significant: low of low * high
v_mul_hi_u32 v16, v1, s8 // (op 536) most significant: high of low * low
v_mul_lo_u32 v14, v1, s8 // (op 536) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 536) most significant: sum

v_mov_b32 v16, s14 // (op 536) 
v_mov_b32 v17, s15 // (op 536) 
v_add_co_u32 v14, s[0:1], v14, v16 // (op 536) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 536) most significant half




v_mul_lo_u32 v17, v15, 6 // (op 536) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 536) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 536) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 536) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 536) 

v_mov_b32 v16, v14 // (op 536) convert

s_mul_i32 s1, s7, 6 // (op 538) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 538) least significant
s_mul_hi_u32 s14, s6, 6 // (op 538) most significant: high of low * low
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

s_lshl_b32 s0, s17, 4 // (op 555) 
s_mov_b32 s15, 0 // (op 555) convert
s_mov_b32 s14, s0 // (op 555) convert

v_mul_lo_u32 v15, v1, s9 // (op 555) most significant: low of low * high
v_mul_hi_u32 v17, v1, s8 // (op 555) most significant: high of low * low
v_mul_lo_u32 v14, v1, s8 // (op 555) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 555) most significant: sum

v_mov_b32 v17, s14 // (op 555) 
v_mov_b32 v18, s15 // (op 555) 
v_add_co_u32 v14, s[0:1], v14, v17 // (op 555) least significant half
v_addc_co_u32 v15, s[0:1], v15, v18, s[0:1] // (op 555) most significant half




v_mul_lo_u32 v19, v15, 6 // (op 555) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 555) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 555) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 555) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 555) 

v_mov_b32 v17, v14 // (op 555) convert

s_mul_i32 s1, s7, 6 // (op 557) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 557) least significant
s_mul_hi_u32 s14, s6, 6 // (op 557) most significant: high of low * low
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
v_mul_lo_u32 v18, v15, s12 // (op 600) 

v_add_u32 v15, v14, v18 // (op 600) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 602) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 602) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 602) least significant
s_mul_hi_u32 s15, s12, s22 // (op 602) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 602) most significant: sum
s_add_u32 s1, s1, s15 // (op 602) most significant: sum


s_mov_b32 s48, s20 // (op 602) reg expression
s_mov_b32 s49, s21 // (op 602) reg expression
s_mov_b32 s50, s0 // (op 602) reg expression
s_mov_b32 s51, 131072 // (op 602) call()

v_and_b32 v14, -32, v1 // (op 619) 
v_sub_u32 v18, v1, v14 // (op 619) 

v_lshl_add_u32 v14, s16, 5, v18 // (op 619) 

v_lshrrev_b32 v18, 5, v1 // (op 619) 
v_mul_lo_u32 v19, v18, s12 // (op 619) 

v_add_u32 v18, v14, v19 // (op 619) 


s_mul_i32 s1, s12, s23 // (op 621) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 621) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 621) least significant
s_mul_hi_u32 s15, s12, s22 // (op 621) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 621) most significant: sum
s_add_u32 s1, s1, s15 // (op 621) most significant: sum


s_mov_b32 s48, s20 // (op 621) reg expression
s_mov_b32 s49, s21 // (op 621) reg expression
s_mov_b32 s50, s0 // (op 621) reg expression
s_mov_b32 s51, 131072 // (op 621) call()

v_and_b32 v14, 15, v1 // (op 664) 
v_lshl_add_u32 v19, s17, 4, v14 // (op 664) 

v_and_b32 v14, 63, v1 // (op 664) 
v_lshrrev_b32 v20, 4, v14 // (op 664) 

v_mul_lo_u32 v14, v20, s24 // (op 664) 
v_add_u32 v14, v14, v19 // (op 664) 


s_mul_i32 s1, s24, s23 // (op 666) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 666) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 666) least significant
s_mul_hi_u32 s15, s24, s22 // (op 666) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 666) most significant: sum
s_add_u32 s1, s1, s15 // (op 666) most significant: sum


s_mov_b32 s52, s26 // (op 666) reg expression
s_mov_b32 s53, s27 // (op 666) reg expression
s_mov_b32 s54, s0 // (op 666) reg expression
s_mov_b32 s55, 131072 // (op 666) call()

v_and_b32 v19, 15, v1 // (op 683) 
v_lshl_add_u32 v20, s17, 4, v19 // (op 683) 

v_and_b32 v19, 63, v1 // (op 683) 
v_lshrrev_b32 v21, 4, v19 // (op 683) 

v_mul_lo_u32 v19, v21, s24 // (op 683) 
v_add_u32 v19, v19, v20 // (op 683) 


s_mul_i32 s1, s24, s23 // (op 685) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 685) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 685) least significant
s_mul_hi_u32 s15, s24, s22 // (op 685) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 685) most significant: sum
s_add_u32 s1, s1, s15 // (op 685) most significant: sum


s_mov_b32 s52, s26 // (op 685) reg expression
s_mov_b32 s53, s27 // (op 685) reg expression
s_mov_b32 s54, s0 // (op 685) reg expression
s_mov_b32 s55, 131072 // (op 685) call()


v_mov_b64 v[20:21], s[24:25] // (op 337) convert

v_mov_b32 v22, v14 // (op 337) reg expression
v_mov_b32 v23, v22 // (op 337) 
buffer_load_ubyte v24, v23, s[52:55], 0 offen offset: 0 // (op 337) Load value





v_mov_b32 v20, v16 // (op 341) reg expression

buffer_load_dwordx3 v[26:28], v20, s[44:47], 0 offen // (op 341) Load value

v_and_b32 v25, -2, v1 // (op 749) 
v_sub_u32 v20, v1, v25 // (op 749) 

v_lshlrev_b32 v25, 4, v20 // (op 749) 

v_lshrrev_b32 v20, 1, v1 // (op 749) 
v_lshl_add_u32 v21, v20, 5, v25 // (op 749) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 749) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 428) 
v_mov_b64 v[20:21], s[14:15] // (op 428) convert


v_mov_b64 v[22:23], v[10:11] // (op 428) reg expression
s_mov_b32 m0, s0 // (op 428) reg expression
buffer_load_dwordx4 v22, s[40:43], 0 offen offset: 0 lds // (op 428) Load value direct to lds
v_add_co_u32 v22, s[14:15], v22, v20 // (op 428) least significant half
v_addc_co_u32 v23, s[14:15], 0, v23, s[14:15] // (op 428) most significant half

s_add_u32 m0, m0, 2048 // (op 428) 
buffer_load_dwordx4 v22, s[40:43], 0 offen offset: 0 lds // (op 428) Load value direct to lds





v_mov_b32 v25, v15 // (op 347) reg expression

buffer_load_ubyte v20, v25, s[48:51], 0 offen offset: 0 // (op 347) Load value


v_mov_b64 v[22:23], s[24:25] // (op 351) convert

s_lshl_b32 s0, s24, 2 // (op 351) 
v_add_u32 v21, v14, s0 // (op 351) 

v_mov_b32 v25, v21 // (op 351) 
buffer_load_ubyte v29, v25, s[52:55], 0 offen offset: 0 // (op 351) Load value






s_lshl_b64 s[0:1], s[8:9], 7 // (op 355) 
s_mul_i32 s15, s1, 6 // (op 355) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 355) least significant
s_mul_hi_u32 s18, s0, 6 // (op 355) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 355) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 355) 

v_add_u32 v14, v16, s0 // (op 355) 


buffer_load_dwordx3 v[30:32], v14, s[44:47], 0 offen // (op 355) Load value


v_and_b32 v14, -2, v1 // (op 928) 
v_sub_u32 v16, v1, v14 // (op 928) 

v_lshlrev_b32 v14, 4, v16 // (op 928) 

v_lshrrev_b32 v16, 1, v1 // (op 928) 
v_lshl_add_u32 v25, v16, 5, v14 // (op 928) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 928) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 433) 
v_mov_b64 v[22:23], s[14:15] // (op 433) convert


s_lshl_b64 s[14:15], s[2:3], 7 // (op 433) 
v_mov_b32 v14, s14 // (op 433) 
v_mov_b32 v16, s15 // (op 433) 
v_add_co_u32 v34, s[18:19], v10, v14 // (op 433) least significant half
v_addc_co_u32 v35, s[18:19], v11, v16, s[18:19] // (op 433) most significant half




s_add_u32 s1, 4096, s0 // (op 433) 
s_mov_b32 m0, s1 // (op 433) reg expression
buffer_load_dwordx4 v34, s[40:43], 0 offen offset: 0 lds // (op 433) Load value direct to lds
v_add_co_u32 v34, s[14:15], v34, v22 // (op 433) least significant half
v_addc_co_u32 v35, s[14:15], 0, v35, s[14:15] // (op 433) most significant half

s_add_u32 m0, m0, 2048 // (op 433) 
buffer_load_dwordx4 v34, s[40:43], 0 offen offset: 0 lds // (op 433) Load value direct to lds







s_lshl_b32 s0, s12, 2 // (op 361) 
v_add_u32 v14, v15, s0 // (op 361) 


buffer_load_ubyte v16, v14, s[48:51], 0 offen offset: 0 // (op 361) Load value


v_lshlrev_b32 v25, 4, v1 // (op 802) 
v_mul_lo_u32 v10, v25, 6 // (op 802) 

v_lshrrev_b32 v11, 3, v10 // (op 802) 

v_and_b32 v10, 268435455, v1 // (op 802) 
v_lshlrev_b32 v25, 2, v10 // (op 802) 

v_add_u32 v10, v11, v25 // (op 802) 





s_waitcnt vmcnt(8)
ds_write_b96 v10, v[26:28] offset:8192 // (op 363) Store local data 

v_and_b32 v25, -32, v1 // (op 855) 
v_sub_u32 v10, v1, v25 // (op 855) 

v_and_b32 v11, 4294967264, v1 // (op 855) 
v_add_u32 v25, v11, v10 // (op 855) 





s_waitcnt vmcnt(5)
ds_write_b8 v25, v20 offset:10240 // (op 365) Store local data 

v_and_b32 v25, -2, v1 // (op 721) 
v_sub_u32 v10, v1, v25 // (op 721) 

v_lshlrev_b32 v11, 4, v10 // (op 721) 

v_lshrrev_b32 v10, 1, v1 // (op 721) 
v_lshl_add_u32 v25, v10, 5, v11 // (op 721) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 721) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v25, 4, v1 // (op 773) 
v_mul_lo_u32 v10, v25, 6 // (op 773) 

v_lshrrev_b32 v11, 3, v10 // (op 773) 

v_and_b32 v10, 268435455, v1 // (op 773) 
v_lshlrev_b32 v25, 2, v10 // (op 773) 

v_add_u32 v10, v11, v25 // (op 773) 


v_and_b32 v11, -32, v1 // (op 826) 
v_sub_u32 v25, v1, v11 // (op 826) 

v_and_b32 v11, 4294967264, v1 // (op 826) 
v_add_u32 v14, v11, v25 // (op 826) 


v_and_b32 v11, -2, v1 // (op 913) 
v_sub_u32 v15, v1, v11 // (op 913) 

v_lshlrev_b32 v11, 4, v15 // (op 913) 

v_lshrrev_b32 v15, 1, v1 // (op 913) 
v_lshl_add_u32 v25, v15, 5, v11 // (op 913) 


s_nop 0
v_readfirstlane_b32 s1, v25 // (op 913) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v11, 4, v1 // (op 952) 
v_mul_lo_u32 v15, v11, 6 // (op 952) 

v_lshrrev_b32 v11, 3, v15 // (op 952) 

v_and_b32 v15, 268435455, v1 // (op 952) 
v_lshlrev_b32 v25, 2, v15 // (op 952) 

v_add_u32 v15, v11, v25 // (op 952) 


v_and_b32 v11, -32, v1 // (op 976) 
v_sub_u32 v25, v1, v11 // (op 976) 

v_and_b32 v11, 4294967264, v1 // (op 976) 
v_add_u32 v21, v11, v25 // (op 976) 


s_mov_b32 s14, 0 // (op 55) call()
s_ashr_i64 s[18:19], s[6:7], 7 // (op 54) 
s_mov_b32 s15, s18 // (op 54) convert

s_and_b32 s18, s15, 4294967294 // (op 54) 

s_cmp_lt_u32 s14, s18 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 if false)
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54:
s_barrier  // (op 1021) 
s_add_u32 s14, s14, 2 // (op 56) 



ds_read_b64_tr_b8 v[34:35], v0 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:256 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:2048 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:2304 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1034) 



ds_read_b96_tr_b6 v[42:44], v5 offset:8192 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[45:47], v5 offset:8448 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1048) 


ds_read_u8 v11, v3 offset:10240 // (op 75) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[42:47], a[0:3], v11, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 82) 




v_mov_b64 v[22:23], s[24:25] // (op 80) convert

v_mov_b32 v11, v19 // (op 80) 
s_lshl_b32 s15, s24, 3 // (op 80) 
v_add_u32 v25, v11, s15 // (op 80) 

v_mov_b32 v33, v25 // (op 80) 
buffer_load_ubyte v24, v33, s[52:55], 0 offen offset: 0 // (op 80) Load value





v_mov_b32 v25, v17 // (op 10) 
s_lshl_b64 s[18:19], s[8:9], 7 // (op 10) 
s_mul_i32 s39, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 10) least significant
s_mul_hi_u32 s15, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 10) 

s_lshl_b32 s15, s18, 1 // (op 10) 

v_add_u32 v22, v25, s15 // (op 10) 


buffer_load_dwordx3 v[26:28], v22, s[44:47], 0 offen // (op 10) Load value

s_barrier  // (op 1024) 

s_lshl_b64 s[18:19], s[2:3], 6 // (op 412) 
v_mov_b64 v[22:23], s[18:19] // (op 412) convert


v_mov_b64 v[34:35], v[12:13] // (op 412) 
s_lshl_b64 s[18:19], s[2:3], 7 // (op 412) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 412) 

v_mov_b32 v33, s38 // (op 412) 
v_mov_b32 v36, s39 // (op 412) 
v_add_co_u32 v38, s[18:19], v34, v33 // (op 412) least significant half
v_addc_co_u32 v39, s[18:19], v35, v36, s[18:19] // (op 412) most significant half




s_mov_b32 m0, s0 // (op 412) reg expression
buffer_load_dwordx4 v38, s[40:43], 0 offen offset: 0 lds // (op 412) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v22 // (op 412) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 412) most significant half

s_add_u32 m0, m0, 2048 // (op 412) 
buffer_load_dwordx4 v38, s[40:43], 0 offen offset: 0 lds // (op 412) Load value direct to lds




v_mov_b32 v33, v18 // (op 17) 
s_lshl_b32 s15, s12, 3 // (op 17) 
v_add_u32 v22, v33, s15 // (op 17) 


buffer_load_ubyte v20, v22, s[48:51], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1071) 



s_waitcnt vmcnt(8)
ds_write_b96 v15, v[30:32] offset:10368 // (op 235) Store local data 
s_barrier  // (op 1077) 



s_waitcnt vmcnt(5)
ds_write_b8 v21, v16 offset:12416 // (op 236) Store local data 
s_barrier  // (op 1062) 



ds_read_b64_tr_b8 v[36:37], v4 offset:4096 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:4352 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:6144 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:6400 // (op 227) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1068) 



ds_read_b96_tr_b6 v[44:46], v8 offset:10368 // (op 229) Transpose load local data 
ds_read_b96_tr_b6 v[47:49], v8 offset:10624 // (op 229) Transpose load local data 
s_barrier  // (op 1124) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1074) 


ds_read_u8 v22, v6 offset:12416 // (op 231) Load local data 
s_barrier  // (op 1130) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:49], a[0:3], v22, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 234) 




v_mov_b64 v[22:23], s[24:25] // (op 233) convert

s_lshl_b32 s15, s24, 2 // (op 233) 
s_mul_i32 s18, 3, s15 // (op 233) 

v_add_u32 v36, v11, s18 // (op 233) 

v_mov_b32 v37, v36 // (op 233) 
buffer_load_ubyte v29, v37, s[52:55], 0 offen offset: 0 // (op 233) Load value






s_lshl_b32 s15, s24, 3 // (op 675) 
v_add_u32 v19, v19, s15 // (op 675) 

s_lshl_b64 s[18:19], s[8:9], 7 // (op 237) 
s_mul_i32 s39, s19, 6 // (op 237) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 237) least significant
s_mul_hi_u32 s15, s18, 6 // (op 237) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 237) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 237) 

s_mul_i32 s15, 3, s18 // (op 237) 

v_add_u32 v11, v25, s15 // (op 237) 


buffer_load_dwordx3 v[30:32], v11, s[44:47], 0 offen // (op 237) Load value


s_barrier  // (op 1065) 
s_lshl_b64 s[18:19], s[8:9], 7 // (op 547) 

s_mul_i32 s39, s19, 6 // (op 547) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 547) least significant
s_mul_hi_u32 s15, s18, 6 // (op 547) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 547) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 547) 

s_lshl_b32 s15, s18, 1 // (op 547) 

v_add_u32 v17, v17, s15 // (op 547) 

s_lshl_b64 s[18:19], s[2:3], 6 // (op 423) 
v_mov_b64 v[22:23], s[18:19] // (op 423) convert


s_lshl_b64 s[18:19], s[2:3], 7 // (op 423) 
s_mul_i32 s39, 3, s19 // (op 423) most significant: low of low * high
s_mul_i32 s38, 3, s18 // (op 423) least significant
s_mul_hi_u32 s15, 3, s18 // (op 423) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 423) most significant: sum


v_mov_b32 v11, s38 // (op 423) 
v_mov_b32 v25, s39 // (op 423) 
v_add_co_u32 v36, s[18:19], v34, v11 // (op 423) least significant half
v_addc_co_u32 v37, s[18:19], v35, v25, s[18:19] // (op 423) most significant half




s_add_u32 s15, 4096, s1 // (op 423) 
s_mov_b32 m0, s15 // (op 423) reg expression
buffer_load_dwordx4 v36, s[40:43], 0 offen offset: 0 lds // (op 423) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v22 // (op 423) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 423) most significant half

s_add_u32 m0, m0, 2048 // (op 423) 
buffer_load_dwordx4 v36, s[40:43], 0 offen offset: 0 lds // (op 423) Load value direct to lds






s_lshl_b32 s15, s12, 2 // (op 238) 
s_mul_i32 s18, 3, s15 // (op 238) 

v_add_u32 v11, v33, s18 // (op 238) 


s_lshl_b64 s[18:19], s[2:3], 7 // (op 483) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 483) 

v_mov_b32 v25, s38 // (op 483) 
v_mov_b32 v22, s39 // (op 483) 
v_add_co_u32 v12, s[18:19], v12, v25 // (op 483) least significant half
v_addc_co_u32 v13, s[18:19], v13, v22, s[18:19] // (op 483) most significant half




buffer_load_ubyte v16, v11, s[48:51], 0 offen offset: 0 // (op 238) Load value


s_barrier  // (op 1037) 
s_lshl_b32 s15, s12, 3 // (op 611) 



v_add_u32 v18, v18, s15 // (op 611) 

s_waitcnt vmcnt(8)
ds_write_b96 v10, v[26:28] offset:8192 // (op 13) Store local data 
s_barrier  // (op 1051) 



s_waitcnt vmcnt(5)
ds_write_b8 v14, v20 offset:10240 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[6:7], 7 // (op 54) 
s_mov_b32 s15, s18 // (op 54) convert

s_and_b32 s18, s15, 4294967294 // (op 54) 

s_cmp_lt_u32 s14, s18 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 if true)
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54:














s_barrier  // (op 1088) 
v_and_b32 v10, -2, v1 // (op 446) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 446) 
v_sub_u32 v11, v1, v10 // (op 446) 

s_lshr_b32 s14, s0, 1 // (op 446) 

v_lshlrev_b32 v10, 4, v11 // (op 446) 

v_lshl_add_u32 v11, s16, 5, v10 // (op 446) 

v_mov_b32 v13, 0 // (op 446) convert
v_mov_b32 v12, v11 // (op 446) convert

v_lshrrev_b32 v10, 1, v1 // (op 446) 
v_lshl_add_u32 v11, s14, 8, v10 // (op 446) 


v_mul_lo_u32 v15, v11, s3 // (op 446) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 446) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 446) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 446) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 446) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 446) most significant half



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

s_ashr_i64 s[0:1], s[6:7], 7 // (op 513) 
s_lshr_b32 s14, s0, 1 // (op 513) 

v_lshl_add_u32 v12, s14, 8, v1 // (op 513) 

s_lshl_b32 s0, s17, 4 // (op 513) 
s_mov_b32 s15, 0 // (op 513) convert
s_mov_b32 s14, s0 // (op 513) convert

v_mul_lo_u32 v15, v12, s9 // (op 513) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 513) most significant: high of low * low
v_mul_lo_u32 v14, v12, s8 // (op 513) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 513) most significant: sum

v_mov_b32 v13, s14 // (op 513) 
v_mov_b32 v17, s15 // (op 513) 
v_add_co_u32 v14, s[0:1], v14, v13 // (op 513) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 513) most significant half





v_mul_lo_u32 v13, v15, 6 // (op 513) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 513) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 513) least significant: low of low * low
v_add_u32 v13, v13, v17 // (op 513) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 513) 

v_mov_b32 v12, v14 // (op 513) convert

s_mul_i32 s1, s7, 6 // (op 515) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 515) least significant
s_mul_hi_u32 s14, s6, 6 // (op 515) most significant: high of low * low
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

v_and_b32 v13, -32, v1 // (op 577) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 577) 
v_sub_u32 v14, v1, v13 // (op 577) 

s_lshr_b32 s14, s0, 1 // (op 577) 

v_lshl_add_u32 v13, s16, 5, v14 // (op 577) 

v_lshrrev_b32 v14, 5, v1 // (op 577) 
v_lshl_add_u32 v15, s14, 3, v14 // (op 577) 


v_mul_lo_u32 v14, v15, s12 // (op 577) 

v_add_u32 v15, v13, v14 // (op 577) 


s_mul_i32 s1, s12, s23 // (op 579) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 579) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 579) least significant
s_mul_hi_u32 s15, s12, s22 // (op 579) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 579) most significant: sum
s_add_u32 s1, s1, s15 // (op 579) most significant: sum


s_mov_b32 s48, s20 // (op 579) reg expression
s_mov_b32 s49, s21 // (op 579) reg expression
s_mov_b32 s50, s0 // (op 579) reg expression
s_mov_b32 s51, 131072 // (op 579) call()

v_and_b32 v13, 63, v1 // (op 641) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 641) 
v_lshrrev_b32 v14, 4, v13 // (op 641) 

s_lshr_b32 s14, s0, 1 // (op 641) 

v_and_b32 v13, 15, v1 // (op 641) 
v_lshl_add_u32 v17, s14, 3, v14 // (op 641) 


v_lshl_add_u32 v14, s17, 4, v13 // (op 641) 

v_mul_lo_u32 v13, v17, s24 // (op 641) 
v_add_u32 v13, v13, v14 // (op 641) 


s_mul_i32 s1, s24, s23 // (op 643) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 643) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 643) least significant
s_mul_hi_u32 s15, s24, s22 // (op 643) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 643) most significant: sum
s_add_u32 s1, s1, s15 // (op 643) most significant: sum


s_mov_b32 s52, s26 // (op 643) reg expression
s_mov_b32 s53, s27 // (op 643) reg expression
s_mov_b32 s54, s0 // (op 643) reg expression
s_mov_b32 s55, 131072 // (op 643) call()

v_and_b32 v14, -2, v1 // (op 734) 
v_sub_u32 v17, v1, v14 // (op 734) 

v_lshlrev_b32 v14, 4, v17 // (op 734) 

v_lshrrev_b32 v17, 1, v1 // (op 734) 
v_lshl_add_u32 v18, v17, 5, v14 // (op 734) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 734) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v14, 4, v1 // (op 786) 
v_mul_lo_u32 v17, v14, 6 // (op 786) 

v_lshrrev_b32 v14, 3, v17 // (op 786) 

v_and_b32 v17, 268435455, v1 // (op 786) 
v_lshlrev_b32 v18, 2, v17 // (op 786) 

v_add_u32 v17, v14, v18 // (op 786) 


v_and_b32 v14, -32, v1 // (op 839) 
v_sub_u32 v18, v1, v14 // (op 839) 

v_and_b32 v14, 4294967264, v1 // (op 839) 
v_add_u32 v19, v14, v18 // (op 839) 


s_ashr_i64 s[14:15], s[6:7], 7 // (op 162) 
s_and_b32 s1, s14, 4294967294 // (op 162) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 161) 
s_mov_b32 s18, s14 // (op 161) convert

s_cmp_lt_u32 s1, s18 // (op 161) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161:
s_barrier  // (op 1027) 
s_add_u32 s1, s1, 1 // (op 163) 

s_lshl_b64 s[14:15], s[2:3], 6 // (op 417) 
v_mov_b64 v[20:21], s[14:15] // (op 417) convert


v_mov_b64 v[22:23], v[10:11] // (op 417) 
v_mov_b64 v[24:25], v[22:23] // (op 417) reg expression
s_mov_b32 m0, s0 // (op 417) reg expression
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 417) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v20 // (op 417) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 417) most significant half

s_add_u32 m0, m0, 2048 // (op 417) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 417) Load value direct to lds



s_lshl_b64 s[14:15], s[2:3], 7 // (op 436) 
v_mov_b32 v14, s14 // (op 436) 
v_mov_b32 v16, s15 // (op 436) 
v_add_co_u32 v10, s[18:19], v10, v14 // (op 436) least significant half
v_addc_co_u32 v11, s[18:19], v11, v16, s[18:19] // (op 436) most significant half






v_mov_b32 v14, v12 // (op 179) 
v_mov_b32 v16, v14 // (op 179) reg expression

buffer_load_dwordx3 v[20:22], v16, s[44:47], 0 offen // (op 179) Load value


s_lshl_b64 s[14:15], s[8:9], 7 // (op 505) 
s_mul_i32 s19, s15, 6 // (op 505) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 505) least significant
s_mul_hi_u32 s38, s14, 6 // (op 505) most significant: high of low * low
s_add_u32 s19, s19, s38 // (op 505) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 505) 

v_add_u32 v12, v12, s14 // (op 505) 



s_barrier  // (op 1040) 
v_mov_b32 v14, v15 // (op 180) 



v_mov_b32 v16, v14 // (op 180) reg expression

s_waitcnt vmcnt(0)
ds_write_b96 v17, v[20:22] offset:8192 // (op 177) Store local data 
buffer_load_ubyte v18, v16, s[48:51], 0 offen offset: 0 // (op 180) Load value



s_lshl_b32 s14, s12, 2 // (op 569) 
v_add_u32 v15, v15, s14 // (op 569) 

s_barrier  // (op 1030) 
s_barrier  // (op 1054) 






ds_read_b64_tr_b8 v[20:21], v0 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[22:23], v0 offset:256 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:2048 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:2304 // (op 169) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v18 offset:10240 // (op 178) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 1044) 



ds_read_b96_tr_b6 v[28:30], v5 offset:8192 // (op 171) Transpose load local data 
ds_read_b96_tr_b6 v[31:33], v5 offset:8448 // (op 171) Transpose load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1058) 


ds_read_u8 v14, v3 offset:10240 // (op 173) Load local data 

v_mov_b64 v[34:35], s[24:25] // (op 175) convert

v_mov_b32 v16, v13 // (op 175) 
v_mov_b32 v18, v16 // (op 175) reg expression
v_mov_b32 v36, v18 // (op 175) 
buffer_load_ubyte v37, v36, s[52:55], 0 offen offset: 0 // (op 175) Load value




s_lshl_b32 s14, s24, 2 // (op 633) 
v_add_u32 v13, v13, s14 // (op 633) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[28:33], a[0:3], v14, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 176) 




s_ashr_i64 s[14:15], s[6:7], 7 // (op 161) 
s_mov_b32 s18, s14 // (op 161) convert

s_cmp_lt_u32 s1, s18 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 if true)
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161:






















s_barrier  // (op 1118) 






v_cmp_eq_f32 vcc, 0.00000, v7 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v10, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 

v_lshrrev_b32 v0, 6, v1 // (op 881) 
v_and_b32 v8, 63, v1 // (op 881) 
v_lshrrev_b32 v3, 4, v8 // (op 881) 
v_lshlrev_b32 v4, 2, v3 // (op 881) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 881) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 881) 

v_lshrrev_b32 v3, 2, v8 // (op 881) 

v_and_b32 v8, 3, v3 // (op 881) 

v_and_b32 v3, 3, v1 // (op 881) 
v_lshl_add_u32 v4, v8, 2, v3 // (op 881) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 881) 

v_mul_lo_u32 v8, v3, s28 // (op 881) 

v_add_lshl_u32 v3, v0, v8, 2 // (op 881) 


s_mul_i32 s1, s28, s33 // (op 883) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 883) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 883) least significant
s_mul_hi_u32 s3, s28, s32 // (op 883) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 883) most significant: sum
s_add_u32 s1, s1, s3 // (op 883) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 883) 

s_mov_b32 s4, s30 // (op 883) reg expression
s_mov_b32 s5, s31 // (op 883) reg expression
s_mov_b32 s6, s2 // (op 883) reg expression
s_mov_b32 s7, 131072 // (op 883) call()




buffer_store_dwordx4 v[14:17], v3, s[4:7], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123:
v_lshrrev_b32 v0, 6, v1 // (op 704) 
v_and_b32 v8, 63, v1 // (op 704) 
v_lshrrev_b32 v3, 4, v8 // (op 704) 
v_lshlrev_b32 v4, 2, v3 // (op 704) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 704) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 704) 

v_lshrrev_b32 v3, 2, v8 // (op 704) 

v_and_b32 v8, 3, v3 // (op 704) 

v_and_b32 v3, 3, v1 // (op 704) 
v_lshl_add_u32 v4, v8, 2, v3 // (op 704) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 704) 

v_mul_lo_u32 v8, v3, s34 // (op 704) 

v_add_lshl_u32 v3, v0, v8, 2 // (op 704) 


s_mul_i32 s1, s34, s33 // (op 706) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 706) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 706) least significant
s_mul_hi_u32 s3, s34, s32 // (op 706) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 706) most significant: sum
s_add_u32 s1, s1, s3 // (op 706) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 706) 

s_mov_b32 s8, s36 // (op 706) reg expression
s_mov_b32 s9, s37 // (op 706) reg expression
s_mov_b32 s10, s2 // (op 706) reg expression
s_mov_b32 s11, 131072 // (op 706) call()





buffer_load_dwordx4 v[10:13], v3, s[8:11], 0 offen // (op 135) Load value



v_accvgpr_read v14, a0 // (op 132) 
v_accvgpr_read v15, a1 // (op 132) 
v_accvgpr_read v16, a2 // (op 132) 
v_accvgpr_read v17, a3 // (op 132) 
v_mul_f32 v18, v9, v14 // (op 132) 
v_mul_f32 v19, v9, v15 // (op 132) 
v_mul_f32 v20, v9, v16 // (op 132) 
v_mul_f32 v21, v9, v17 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v7, v10 // (op 136) 
v_mul_f32 v15, v7, v11 // (op 136) 
v_mul_f32 v16, v7, v12 // (op 136) 
v_mul_f32 v17, v7, v13 // (op 136) 

v_add_f32 v10, v14, v18 // (op 133) 
v_add_f32 v11, v15, v19 // (op 133) 
v_add_f32 v12, v16, v20 // (op 133) 
v_add_f32 v13, v17, v21 // (op 133) 


v_lshrrev_b32 v0, 6, v1 // (op 898) 
v_and_b32 v8, 63, v1 // (op 898) 
v_lshrrev_b32 v3, 4, v8 // (op 898) 
v_lshlrev_b32 v4, 2, v3 // (op 898) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 898) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 898) 

v_lshrrev_b32 v3, 2, v8 // (op 898) 

v_and_b32 v8, 3, v3 // (op 898) 

v_and_b32 v3, 3, v1 // (op 898) 
v_lshl_add_u32 v4, v8, 2, v3 // (op 898) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 898) 

v_mul_lo_u32 v8, v3, s28 // (op 898) 

v_add_lshl_u32 v3, v0, v8, 2 // (op 898) 


s_mul_i32 s1, s28, s33 // (op 900) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 900) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 900) least significant
s_mul_hi_u32 s3, s28, s32 // (op 900) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 900) most significant: sum
s_add_u32 s1, s1, s3 // (op 900) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 900) 

s_mov_b32 s4, s30 // (op 900) reg expression
s_mov_b32 s5, s31 // (op 900) reg expression
s_mov_b32 s6, s2 // (op 900) reg expression
s_mov_b32 s7, 131072 // (op 900) call()




buffer_store_dwordx4 v[10:13], v3, s[4:7], 0 offen // (op 134) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123:



s_endpgm  // End of RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.LRR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end:
.size RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2, .LRR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end-RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
  .amdhsa_next_free_vgpr 56
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12544 // lds bytes
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
  - .name: RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
    .symbol: RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 12544
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

