// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.p2align 8
.type RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2,@function
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 765) 
v_and_b32 v3, 63, v1 // (op 765) 
v_sub_u32 v4, v3, v0 // (op 765) 

v_lshrrev_b32 v3, 1, v4 // (op 765) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 765) 


v_and_b32 v0, 1, v4 // (op 765) 

v_lshlrev_b32 v3, 3, v0 // (op 765) 

v_lshrrev_b32 v0, 7, v1 // (op 765) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 765) 


v_add_u32 v0, v5, v4 // (op 765) 


v_and_b32 v3, 63, v1 // (op 818) 
v_and_b32 v4, 48, v1 // (op 818) 
v_sub_u32 v5, v3, v4 // (op 818) 

v_lshrrev_b32 v4, 4, v3 // (op 818) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 818) 


v_lshrrev_b32 v4, 6, v1 // (op 818) 
v_and_b32 v5, 1, v4 // (op 818) 

v_lshlrev_b32 v4, 4, v5 // (op 818) 

v_lshl_add_u32 v5, v3, 5, v4 // (op 818) 


v_mul_lo_u32 v3, v5, 6 // (op 818) 
v_lshrrev_b32 v4, 3, v3 // (op 818) 

v_lshrrev_b32 v3, 4, v5 // (op 818) 

v_lshlrev_b32 v5, 2, v3 // (op 818) 

v_add_u32 v3, v4, v5 // (op 818) 


v_lshrrev_b32 v4, 6, v1 // (op 871) 
v_and_b32 v5, 1, v4 // (op 871) 

v_and_b32 v4, 15, v1 // (op 871) 
v_lshl_add_u32 v6, v5, 4, v4 // (op 871) 


v_and_b32 v4, 63, v1 // (op 871) 
v_lshrrev_b32 v5, 4, v4 // (op 871) 

v_lshl_add_u32 v4, v5, 5, v6 // (op 871) 


v_and_b32 v5, 48, v1 // (op 944) 
v_and_b32 v6, 63, v1 // (op 944) 
v_sub_u32 v7, v6, v5 // (op 944) 

v_lshrrev_b32 v6, 1, v7 // (op 944) 
v_add_lshl_u32 v8, v5, v6, 4 // (op 944) 


v_and_b32 v5, 1, v7 // (op 944) 

v_lshlrev_b32 v6, 3, v5 // (op 944) 

v_lshrrev_b32 v5, 7, v1 // (op 944) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 944) 


v_add_u32 v5, v8, v7 // (op 944) 


v_and_b32 v6, 63, v1 // (op 968) 
v_and_b32 v7, 48, v1 // (op 968) 
v_sub_u32 v8, v6, v7 // (op 968) 

v_lshrrev_b32 v7, 4, v6 // (op 968) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 968) 


v_lshrrev_b32 v7, 6, v1 // (op 968) 
v_and_b32 v8, 1, v7 // (op 968) 

v_lshlrev_b32 v7, 4, v8 // (op 968) 

v_lshl_add_u32 v8, v6, 5, v7 // (op 968) 


v_mul_lo_u32 v6, v8, 6 // (op 968) 
v_lshrrev_b32 v7, 3, v6 // (op 968) 

v_lshrrev_b32 v6, 4, v8 // (op 968) 

v_lshlrev_b32 v8, 2, v6 // (op 968) 

v_add_u32 v6, v7, v8 // (op 968) 


v_lshrrev_b32 v7, 6, v1 // (op 992) 
v_and_b32 v8, 1, v7 // (op 992) 

v_and_b32 v7, 15, v1 // (op 992) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 992) 


v_and_b32 v7, 63, v1 // (op 992) 
v_lshrrev_b32 v8, 4, v7 // (op 992) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 992) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
s_lshl_b32 s0, s16, 4 // (op 476) 
v_mul_lo_u32 v11, v1, s3 // (op 476) most significant: low of low * high
s_mov_b32 s15, 0 // (op 476) convert
s_mov_b32 s14, s0 // (op 476) convert

v_mul_hi_u32 v12, v1, s2 // (op 476) most significant: high of low * low
v_mul_lo_u32 v10, v1, s2 // (op 476) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 476) most significant: sum

v_mov_b32 v12, s14 // (op 476) 
v_mov_b32 v13, s15 // (op 476) 
v_add_co_u32 v14, s[0:1], v12, v10 // (op 476) least significant half
v_addc_co_u32 v15, s[0:1], v13, v11, s[0:1] // (op 476) most significant half





s_mul_i32 s1, s2, s7 // (op 478) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 478) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 478) least significant
s_mul_hi_u32 s15, s2, s6 // (op 478) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 478) most significant: sum
s_add_u32 s1, s1, s15 // (op 478) most significant: sum


s_mov_b32 s40, s4 // (op 478) reg expression
s_mov_b32 s41, s5 // (op 478) reg expression
s_mov_b32 s42, s0 // (op 478) reg expression
s_mov_b32 s43, 131072 // (op 478) call()

s_lshl_b32 s0, s16, 4 // (op 495) 
v_mul_lo_u32 v11, v1, s3 // (op 495) most significant: low of low * high
v_mul_hi_u32 v12, v1, s2 // (op 495) most significant: high of low * low
v_mul_lo_u32 v10, v1, s2 // (op 495) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 495) most significant: sum

s_mov_b32 s15, 0 // (op 495) convert
s_mov_b32 s14, s0 // (op 495) convert

v_mov_b32 v12, s14 // (op 495) 
v_mov_b32 v13, s15 // (op 495) 
v_add_co_u32 v16, s[0:1], v12, v10 // (op 495) least significant half
v_addc_co_u32 v17, s[0:1], v13, v11, s[0:1] // (op 495) most significant half





s_mul_i32 s1, s2, s7 // (op 497) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 497) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 497) least significant
s_mul_hi_u32 s15, s2, s6 // (op 497) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 497) most significant: sum
s_add_u32 s1, s1, s15 // (op 497) most significant: sum


s_mov_b32 s40, s4 // (op 497) reg expression
s_mov_b32 s41, s5 // (op 497) reg expression
s_mov_b32 s42, s0 // (op 497) reg expression
s_mov_b32 s43, 131072 // (op 497) call()

s_lshl_b32 s0, s17, 5 // (op 540) 
s_mov_b32 s15, 0 // (op 540) convert
s_mov_b32 s14, s0 // (op 540) convert

v_mul_lo_u32 v11, v1, s9 // (op 540) most significant: low of low * high
v_mul_hi_u32 v12, v1, s8 // (op 540) most significant: high of low * low
v_mul_lo_u32 v10, v1, s8 // (op 540) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 540) most significant: sum

v_mov_b32 v12, s14 // (op 540) 
v_mov_b32 v13, s15 // (op 540) 
v_add_co_u32 v10, s[0:1], v10, v12 // (op 540) least significant half
v_addc_co_u32 v11, s[0:1], v11, v13, s[0:1] // (op 540) most significant half




v_mul_lo_u32 v13, v11, 6 // (op 540) most significant: low of high * low
v_mul_hi_u32 v18, v10, 6 // (op 540) most significant: high of low * low
v_mul_lo_u32 v12, v10, 6 // (op 540) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 540) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 540) 

v_mov_b32 v12, v10 // (op 540) convert

s_mul_i32 s1, s7, 6 // (op 542) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 542) least significant
s_mul_hi_u32 s14, s6, 6 // (op 542) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 542) most significant: sum

s_mul_i32 s15, s8, s1 // (op 542) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 542) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 542) least significant
s_mul_hi_u32 s19, s8, s0 // (op 542) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 542) most significant: sum
s_add_u32 s15, s15, s19 // (op 542) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 542) 

s_mov_b32 s44, s10 // (op 542) reg expression
s_mov_b32 s45, s11 // (op 542) reg expression
s_mov_b32 s46, s0 // (op 542) reg expression
s_mov_b32 s47, 131072 // (op 542) call()

s_lshl_b32 s0, s17, 5 // (op 559) 
s_mov_b32 s15, 0 // (op 559) convert
s_mov_b32 s14, s0 // (op 559) convert

v_mul_lo_u32 v11, v1, s9 // (op 559) most significant: low of low * high
v_mul_hi_u32 v13, v1, s8 // (op 559) most significant: high of low * low
v_mul_lo_u32 v10, v1, s8 // (op 559) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 559) most significant: sum

v_mov_b32 v13, s14 // (op 559) 
v_mov_b32 v18, s15 // (op 559) 
v_add_co_u32 v10, s[0:1], v10, v13 // (op 559) least significant half
v_addc_co_u32 v11, s[0:1], v11, v18, s[0:1] // (op 559) most significant half




v_mul_lo_u32 v19, v11, 6 // (op 559) most significant: low of high * low
v_mul_hi_u32 v13, v10, 6 // (op 559) most significant: high of low * low
v_mul_lo_u32 v18, v10, 6 // (op 559) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 559) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[18:19] // (op 559) 

v_mov_b32 v13, v10 // (op 559) convert

s_mul_i32 s1, s7, 6 // (op 561) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 561) least significant
s_mul_hi_u32 s14, s6, 6 // (op 561) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 561) most significant: sum

s_mul_i32 s15, s8, s1 // (op 561) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 561) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 561) least significant
s_mul_hi_u32 s19, s8, s0 // (op 561) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 561) most significant: sum
s_add_u32 s15, s15, s19 // (op 561) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 561) 

s_mov_b32 s44, s10 // (op 561) reg expression
s_mov_b32 s45, s11 // (op 561) reg expression
s_mov_b32 s46, s0 // (op 561) reg expression
s_mov_b32 s47, 131072 // (op 561) call()

v_lshrrev_b32 v10, 7, v1 // (op 604) 
v_and_b32 v11, 15, v1 // (op 604) 
v_lshl_add_u32 v18, v10, 4, v11 // (op 604) 


v_lshl_add_u32 v10, s16, 4, v18 // (op 604) 

v_and_b32 v11, 63, v1 // (op 604) 
v_lshrrev_b32 v18, 4, v11 // (op 604) 

v_mul_lo_u32 v11, v18, s12 // (op 604) 

v_add_u32 v18, v10, v11 // (op 604) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 606) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 606) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 606) least significant
s_mul_hi_u32 s15, s12, s22 // (op 606) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 606) most significant: sum
s_add_u32 s1, s1, s15 // (op 606) most significant: sum


s_mov_b32 s48, s20 // (op 606) reg expression
s_mov_b32 s49, s21 // (op 606) reg expression
s_mov_b32 s50, s0 // (op 606) reg expression
s_mov_b32 s51, 131072 // (op 606) call()

v_lshrrev_b32 v10, 7, v1 // (op 623) 
v_and_b32 v11, 15, v1 // (op 623) 
v_lshl_add_u32 v19, v10, 4, v11 // (op 623) 


v_lshl_add_u32 v10, s16, 4, v19 // (op 623) 

v_and_b32 v11, 63, v1 // (op 623) 
v_lshrrev_b32 v19, 4, v11 // (op 623) 

v_mul_lo_u32 v11, v19, s12 // (op 623) 

v_add_u32 v19, v10, v11 // (op 623) 


s_mul_i32 s1, s12, s23 // (op 625) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 625) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 625) least significant
s_mul_hi_u32 s15, s12, s22 // (op 625) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 625) most significant: sum
s_add_u32 s1, s1, s15 // (op 625) most significant: sum


s_mov_b32 s48, s20 // (op 625) reg expression
s_mov_b32 s49, s21 // (op 625) reg expression
s_mov_b32 s50, s0 // (op 625) reg expression
s_mov_b32 s51, 131072 // (op 625) call()

v_and_b32 v10, -32, v1 // (op 668) 
v_sub_u32 v11, v1, v10 // (op 668) 

v_lshl_add_u32 v10, s17, 5, v11 // (op 668) 

v_lshrrev_b32 v11, 5, v1 // (op 668) 
v_mul_lo_u32 v20, v11, s24 // (op 668) 
v_add_u32 v20, v20, v10 // (op 668) 


s_mul_i32 s1, s24, s23 // (op 670) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 670) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 670) least significant
s_mul_hi_u32 s15, s24, s22 // (op 670) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 670) most significant: sum
s_add_u32 s1, s1, s15 // (op 670) most significant: sum


s_mov_b32 s52, s26 // (op 670) reg expression
s_mov_b32 s53, s27 // (op 670) reg expression
s_mov_b32 s54, s0 // (op 670) reg expression
s_mov_b32 s55, 131072 // (op 670) call()

v_and_b32 v10, -32, v1 // (op 687) 
v_sub_u32 v11, v1, v10 // (op 687) 

v_lshl_add_u32 v10, s17, 5, v11 // (op 687) 

v_lshrrev_b32 v11, 5, v1 // (op 687) 
v_mul_lo_u32 v21, v11, s24 // (op 687) 
v_add_u32 v21, v21, v10 // (op 687) 


s_mul_i32 s1, s24, s23 // (op 689) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 689) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 689) least significant
s_mul_hi_u32 s15, s24, s22 // (op 689) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 689) most significant: sum
s_add_u32 s1, s1, s15 // (op 689) most significant: sum


s_mov_b32 s52, s26 // (op 689) reg expression
s_mov_b32 s53, s27 // (op 689) reg expression
s_mov_b32 s54, s0 // (op 689) reg expression
s_mov_b32 s55, 131072 // (op 689) call()


v_mov_b64 v[10:11], s[12:13] // (op 337) convert

v_mov_b32 v22, v18 // (op 337) reg expression
v_mov_b32 v23, v22 // (op 337) 
buffer_load_ubyte v24, v23, s[48:51], 0 offen offset: 0 // (op 337) Load value





v_mov_b32 v10, v12 // (op 341) reg expression

buffer_load_dwordx3 v[26:28], v10, s[44:47], 0 offen // (op 341) Load value
buffer_load_dwordx3 v[30:32], v10, s[44:47], 0 offen offset:12 // (op 341) Load value



v_mov_b32 v25, v20 // (op 344) reg expression

buffer_load_ubyte v29, v25, s[52:55], 0 offen offset: 0 // (op 344) Load value

v_lshlrev_b32 v25, 4, v1 // (op 753) 
s_nop 0
v_readfirstlane_b32 s0, v25 // (op 753) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[10:11], v[14:15] // (op 432) reg expression

s_nop 0
s_mov_b32 m0, s0 // (op 432) reg expression
buffer_load_dwordx4 v10, s[40:43], 0 offen offset: 0 lds // (op 432) Load value direct to lds



v_mov_b64 v[10:11], s[12:13] // (op 351) convert

s_lshl_b32 s0, s12, 2 // (op 351) 
v_add_u32 v25, v18, s0 // (op 351) 

v_mov_b32 v22, v25 // (op 351) 
buffer_load_ubyte v23, v22, s[48:51], 0 offen offset: 0 // (op 351) Load value






s_lshl_b64 s[0:1], s[8:9], 7 // (op 355) 
s_mul_i32 s15, s1, 6 // (op 355) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 355) least significant
s_mul_hi_u32 s18, s0, 6 // (op 355) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 355) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 355) 

v_add_u32 v18, v12, s0 // (op 355) 


buffer_load_dwordx3 v[34:36], v18, s[44:47], 0 offen // (op 355) Load value
buffer_load_dwordx3 v[38:40], v18, s[44:47], 0 offen offset:12 // (op 355) Load value




s_lshl_b32 s0, s24, 2 // (op 358) 
v_add_u32 v18, v20, s0 // (op 358) 


buffer_load_ubyte v22, v18, s[52:55], 0 offen offset: 0 // (op 358) Load value


v_lshlrev_b32 v18, 4, v1 // (op 932) 
s_nop 0
v_readfirstlane_b32 s0, v18 // (op 932) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[14:15], s[2:3], 7 // (op 437) 
v_mov_b32 v18, s14 // (op 437) 
v_mov_b32 v20, s15 // (op 437) 
v_add_co_u32 v10, s[18:19], v14, v18 // (op 437) least significant half
v_addc_co_u32 v11, s[18:19], v15, v20, s[18:19] // (op 437) most significant half




s_add_u32 s1, 2048, s0 // (op 437) 

s_mov_b32 m0, s1 // (op 437) reg expression
buffer_load_dwordx4 v10, s[40:43], 0 offen offset: 0 lds // (op 437) Load value direct to lds




v_lshlrev_b32 v18, 5, v1 // (op 806) 
v_mul_lo_u32 v20, v18, 6 // (op 806) 
v_lshrrev_b32 v25, 3, v20 // (op 806) 

v_lshrrev_b32 v20, 4, v18 // (op 806) 

v_lshlrev_b32 v18, 2, v20 // (op 806) 

v_add_u32 v20, v25, v18 // (op 806) 





s_waitcnt vmcnt(8)
ds_write_b96 v20, v[26:28] offset:4096 // (op 363) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v20, v[30:32] offset:4112 // (op 363) Store local data 

v_and_b32 v18, -32, v1 // (op 859) 
v_sub_u32 v20, v1, v18 // (op 859) 

v_and_b32 v18, 4294967264, v1 // (op 859) 
v_add_u32 v25, v18, v20 // (op 859) 





s_waitcnt vmcnt(6)
ds_write_b8 v25, v29 offset:8192 // (op 364) Store local data 

v_lshlrev_b32 v18, 4, v1 // (op 725) 
s_nop 0
v_readfirstlane_b32 s0, v18 // (op 725) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v18, 5, v1 // (op 777) 
v_mul_lo_u32 v20, v18, 6 // (op 777) 
v_lshrrev_b32 v25, 3, v20 // (op 777) 

v_lshrrev_b32 v20, 4, v18 // (op 777) 

v_lshlrev_b32 v18, 2, v20 // (op 777) 

v_add_u32 v20, v25, v18 // (op 777) 


v_and_b32 v18, -32, v1 // (op 830) 
v_sub_u32 v25, v1, v18 // (op 830) 

v_and_b32 v18, 4294967264, v1 // (op 830) 
v_add_u32 v33, v18, v25 // (op 830) 


v_lshlrev_b32 v18, 4, v1 // (op 917) 
s_nop 0
v_readfirstlane_b32 s1, v18 // (op 917) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v18, 5, v1 // (op 956) 
v_mul_lo_u32 v25, v18, 6 // (op 956) 
v_lshrrev_b32 v37, 3, v25 // (op 956) 

v_lshrrev_b32 v25, 4, v18 // (op 956) 

v_lshlrev_b32 v18, 2, v25 // (op 956) 

v_add_u32 v25, v37, v18 // (op 956) 


v_and_b32 v18, -32, v1 // (op 980) 
v_sub_u32 v37, v1, v18 // (op 980) 

v_and_b32 v18, 4294967264, v1 // (op 980) 
v_add_u32 v10, v18, v37 // (op 980) 


s_mov_b32 s14, 0 // (op 55) call()
s_ashr_i64 s[18:19], s[6:7], 7 // (op 54) 
s_mov_b32 s15, s18 // (op 54) convert

s_and_b32 s18, s15, 4294967294 // (op 54) 

s_cmp_lt_u32 s14, s18 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54 if false)
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54:
s_add_u32 s14, s14, 2 // (op 56) 
s_waitcnt vmcnt(5)
s_barrier  // (op 1025) 



ds_read_b64_tr_b8 v[42:43], v0 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:128 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:1024 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:1152 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1038) 



ds_read_b96_tr_b6 v[50:52], v3 offset:4096 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[53:55], v3 offset:4608 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1052) 


ds_read_u8 v18, v4 offset:8192 // (op 79) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:55], a[0:3], v24, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 82) 




v_mov_b64 v[14:15], s[12:13] // (op 75) convert

v_mov_b32 v18, v19 // (op 75) 
s_lshl_b32 s15, s12, 3 // (op 75) 
v_add_u32 v37, v18, s15 // (op 75) 

v_mov_b32 v11, v37 // (op 75) 
buffer_load_ubyte v24, v11, s[48:51], 0 offen offset: 0 // (op 75) Load value





v_mov_b32 v37, v13 // (op 10) 
s_lshl_b64 s[18:19], s[8:9], 7 // (op 10) 
s_mul_i32 s39, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 10) least significant
s_mul_hi_u32 s15, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 10) 

s_lshl_b32 s15, s18, 1 // (op 10) 

v_add_u32 v11, v37, s15 // (op 10) 


buffer_load_dwordx3 v[26:28], v11, s[44:47], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[30:32], v11, s[44:47], 0 offen offset:12 // (op 10) Load value



v_mov_b32 v11, v21 // (op 21) 
s_lshl_b32 s15, s24, 3 // (op 21) 
v_add_u32 v12, v11, s15 // (op 21) 


buffer_load_ubyte v29, v12, s[52:55], 0 offen offset: 0 // (op 21) Load value

s_barrier  // (op 1028) 


v_mov_b64 v[14:15], v[16:17] // (op 416) 
s_lshl_b64 s[18:19], s[2:3], 7 // (op 416) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 416) 

v_mov_b32 v12, s38 // (op 416) 
v_mov_b32 v41, s39 // (op 416) 
v_add_co_u32 v42, s[18:19], v14, v12 // (op 416) least significant half
v_addc_co_u32 v43, s[18:19], v15, v41, s[18:19] // (op 416) most significant half





s_mov_b32 m0, s0 // (op 416) reg expression
buffer_load_dwordx4 v42, s[40:43], 0 offen offset: 0 lds // (op 416) Load value direct to lds

s_barrier  // (op 1075) 



s_waitcnt vmcnt(8)
ds_write_b96 v25, v[34:36] offset:8320 // (op 235) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v25, v[38:40] offset:8336 // (op 235) Store local data 
s_barrier  // (op 1081) 



s_waitcnt vmcnt(6)
ds_write_b8 v10, v22 offset:12416 // (op 236) Store local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1066) 



ds_read_b64_tr_b8 v[42:43], v5 offset:2048 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v5 offset:2176 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v5 offset:3072 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v5 offset:3200 // (op 227) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1072) 



ds_read_b96_tr_b6 v[50:52], v6 offset:8320 // (op 229) Transpose load local data 
ds_read_b96_tr_b6 v[53:55], v6 offset:8832 // (op 229) Transpose load local data 
s_barrier  // (op 1128) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1078) 


ds_read_u8 v12, v7 offset:12416 // (op 233) Load local data 
s_barrier  // (op 1136) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:55], a[0:3], v23, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 234) 




v_mov_b64 v[42:43], s[12:13] // (op 231) convert

s_lshl_b32 s15, s12, 2 // (op 231) 
s_mul_i32 s18, 3, s15 // (op 231) 

v_add_u32 v12, v18, s18 // (op 231) 

v_mov_b32 v41, v12 // (op 231) 
buffer_load_ubyte v23, v41, s[48:51], 0 offen offset: 0 // (op 231) Load value






s_lshl_b32 s15, s12, 3 // (op 615) 
v_add_u32 v19, v19, s15 // (op 615) 

s_lshl_b64 s[18:19], s[8:9], 7 // (op 237) 
s_mul_i32 s39, s19, 6 // (op 237) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 237) least significant
s_mul_hi_u32 s15, s18, 6 // (op 237) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 237) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 237) 

s_mul_i32 s15, 3, s18 // (op 237) 

v_add_u32 v12, v37, s15 // (op 237) 


buffer_load_dwordx3 v[34:36], v12, s[44:47], 0 offen // (op 237) Load value
buffer_load_dwordx3 v[38:40], v12, s[44:47], 0 offen offset:12 // (op 237) Load value




s_lshl_b32 s15, s24, 2 // (op 238) 
s_mul_i32 s18, 3, s15 // (op 238) 

v_add_u32 v12, v11, s18 // (op 238) 


s_lshl_b64 s[18:19], s[8:9], 7 // (op 551) 
s_mul_i32 s39, s19, 6 // (op 551) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 551) least significant
s_mul_hi_u32 s15, s18, 6 // (op 551) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 551) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 551) 

s_lshl_b32 s15, s18, 1 // (op 551) 

v_add_u32 v13, v13, s15 // (op 551) 

buffer_load_ubyte v22, v12, s[52:55], 0 offen offset: 0 // (op 238) Load value


s_barrier  // (op 1069) 
s_lshl_b32 s15, s24, 3 // (op 679) 


v_add_u32 v21, v21, s15 // (op 679) 

s_lshl_b64 s[18:19], s[2:3], 7 // (op 427) 
s_mul_i32 s39, 3, s19 // (op 427) most significant: low of low * high
s_mul_i32 s38, 3, s18 // (op 427) least significant
s_mul_hi_u32 s15, 3, s18 // (op 427) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 427) most significant: sum


v_mov_b32 v18, s38 // (op 427) 
v_mov_b32 v37, s39 // (op 427) 
v_add_co_u32 v42, s[18:19], v14, v18 // (op 427) least significant half
v_addc_co_u32 v43, s[18:19], v15, v37, s[18:19] // (op 427) most significant half




s_add_u32 s15, 2048, s1 // (op 427) 

s_mov_b32 m0, s15 // (op 427) reg expression
buffer_load_dwordx4 v42, s[40:43], 0 offen offset: 0 lds // (op 427) Load value direct to lds



s_barrier  // (op 1041) 
s_lshl_b64 s[18:19], s[2:3], 7 // (op 487) 



s_lshl_b64 s[38:39], s[18:19], 1 // (op 487) 

v_mov_b32 v18, s38 // (op 487) 
v_mov_b32 v37, s39 // (op 487) 
v_add_co_u32 v16, s[18:19], v16, v18 // (op 487) least significant half
v_addc_co_u32 v17, s[18:19], v17, v37, s[18:19] // (op 487) most significant half




s_waitcnt vmcnt(8)
ds_write_b96 v20, v[26:28] offset:4096 // (op 13) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v20, v[30:32] offset:4112 // (op 13) Store local data 
s_barrier  // (op 1055) 



s_waitcnt vmcnt(6)
ds_write_b8 v33, v29 offset:8192 // (op 26) Store local data 
s_ashr_i64 s[18:19], s[6:7], 7 // (op 54) 
s_mov_b32 s15, s18 // (op 54) convert

s_and_b32 s18, s15, 4294967294 // (op 54) 

s_cmp_lt_u32 s14, s18 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54 if true)
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54:














s_barrier  // (op 1092) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 450) 
s_lshr_b32 s14, s0, 1 // (op 450) 

v_lshl_add_u32 v37, s14, 8, v1 // (op 450) 

s_lshl_b32 s0, s16, 4 // (op 450) 
v_mul_lo_u32 v11, v37, s3 // (op 450) most significant: low of low * high
v_mul_hi_u32 v12, v37, s2 // (op 450) most significant: high of low * low
v_mul_lo_u32 v10, v37, s2 // (op 450) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 450) most significant: sum


s_mov_b32 s15, 0 // (op 450) convert
s_mov_b32 s14, s0 // (op 450) convert

v_mov_b32 v37, s14 // (op 450) 
v_mov_b32 v12, s15 // (op 450) 
v_add_co_u32 v20, s[0:1], v37, v10 // (op 450) least significant half
v_addc_co_u32 v21, s[0:1], v12, v11, s[0:1] // (op 450) most significant half





s_mul_i32 s1, s2, s7 // (op 452) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 452) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 452) least significant
s_mul_hi_u32 s15, s2, s6 // (op 452) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 452) most significant: sum
s_add_u32 s1, s1, s15 // (op 452) most significant: sum


s_mov_b32 s40, s4 // (op 452) reg expression
s_mov_b32 s41, s5 // (op 452) reg expression
s_mov_b32 s42, s0 // (op 452) reg expression
s_mov_b32 s43, 131072 // (op 452) call()

s_ashr_i64 s[0:1], s[6:7], 7 // (op 517) 
s_lshr_b32 s14, s0, 1 // (op 517) 

v_lshl_add_u32 v37, s14, 8, v1 // (op 517) 

s_lshl_b32 s0, s17, 5 // (op 517) 
s_mov_b32 s15, 0 // (op 517) convert
s_mov_b32 s14, s0 // (op 517) convert

v_mul_lo_u32 v11, v37, s9 // (op 517) most significant: low of low * high
v_mul_hi_u32 v12, v37, s8 // (op 517) most significant: high of low * low
v_mul_lo_u32 v10, v37, s8 // (op 517) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 517) most significant: sum

v_mov_b32 v12, s14 // (op 517) 
v_mov_b32 v13, s15 // (op 517) 
v_add_co_u32 v10, s[0:1], v10, v12 // (op 517) least significant half
v_addc_co_u32 v11, s[0:1], v11, v13, s[0:1] // (op 517) most significant half





v_mul_lo_u32 v13, v11, 6 // (op 517) most significant: low of high * low
v_mul_hi_u32 v37, v10, 6 // (op 517) most significant: high of low * low
v_mul_lo_u32 v12, v10, 6 // (op 517) least significant: low of low * low
v_add_u32 v13, v13, v37 // (op 517) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 517) 

v_mov_b32 v37, v10 // (op 517) convert

s_mul_i32 s1, s7, 6 // (op 519) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 519) least significant
s_mul_hi_u32 s14, s6, 6 // (op 519) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 519) most significant: sum

s_mul_i32 s15, s8, s1 // (op 519) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 519) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 519) least significant
s_mul_hi_u32 s19, s8, s0 // (op 519) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 519) most significant: sum
s_add_u32 s15, s15, s19 // (op 519) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 519) 

s_mov_b32 s44, s10 // (op 519) reg expression
s_mov_b32 s45, s11 // (op 519) reg expression
s_mov_b32 s46, s0 // (op 519) reg expression
s_mov_b32 s47, 131072 // (op 519) call()

v_lshrrev_b32 v10, 7, v1 // (op 581) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 581) 
v_and_b32 v11, 15, v1 // (op 581) 
s_lshr_b32 s14, s0, 1 // (op 581) 

v_lshl_add_u32 v12, v10, 4, v11 // (op 581) 


v_lshl_add_u32 v10, s16, 4, v12 // (op 581) 

v_and_b32 v11, 63, v1 // (op 581) 
v_lshrrev_b32 v12, 4, v11 // (op 581) 

v_lshl_add_u32 v11, s14, 3, v12 // (op 581) 


v_mul_lo_u32 v12, v11, s12 // (op 581) 

v_add_u32 v11, v10, v12 // (op 581) 


s_mul_i32 s1, s12, s23 // (op 583) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 583) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 583) least significant
s_mul_hi_u32 s15, s12, s22 // (op 583) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 583) most significant: sum
s_add_u32 s1, s1, s15 // (op 583) most significant: sum


s_mov_b32 s48, s20 // (op 583) reg expression
s_mov_b32 s49, s21 // (op 583) reg expression
s_mov_b32 s50, s0 // (op 583) reg expression
s_mov_b32 s51, 131072 // (op 583) call()

v_lshrrev_b32 v10, 5, v1 // (op 645) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 645) 
v_and_b32 v12, -32, v1 // (op 645) 
s_lshr_b32 s14, s0, 1 // (op 645) 

v_sub_u32 v13, v1, v12 // (op 645) 

v_lshl_add_u32 v12, s14, 3, v10 // (op 645) 


v_lshl_add_u32 v10, s17, 5, v13 // (op 645) 

v_mul_lo_u32 v13, v12, s24 // (op 645) 
v_add_u32 v13, v13, v10 // (op 645) 


s_mul_i32 s1, s24, s23 // (op 647) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 647) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 647) least significant
s_mul_hi_u32 s15, s24, s22 // (op 647) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 647) most significant: sum
s_add_u32 s1, s1, s15 // (op 647) most significant: sum


s_mov_b32 s52, s26 // (op 647) reg expression
s_mov_b32 s53, s27 // (op 647) reg expression
s_mov_b32 s54, s0 // (op 647) reg expression
s_mov_b32 s55, 131072 // (op 647) call()

v_lshlrev_b32 v10, 4, v1 // (op 738) 
s_nop 0
v_readfirstlane_b32 s0, v10 // (op 738) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v10, 5, v1 // (op 790) 
v_mul_lo_u32 v12, v10, 6 // (op 790) 
v_lshrrev_b32 v14, 3, v12 // (op 790) 

v_lshrrev_b32 v12, 4, v10 // (op 790) 

v_lshlrev_b32 v10, 2, v12 // (op 790) 

v_add_u32 v12, v14, v10 // (op 790) 


v_and_b32 v10, -32, v1 // (op 843) 
v_sub_u32 v14, v1, v10 // (op 843) 

v_and_b32 v10, 4294967264, v1 // (op 843) 
v_add_u32 v15, v10, v14 // (op 843) 


s_ashr_i64 s[14:15], s[6:7], 7 // (op 162) 
s_and_b32 s1, s14, 4294967294 // (op 162) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 161) 
s_mov_b32 s18, s14 // (op 161) convert

s_cmp_lt_u32 s1, s18 // (op 161) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161:
s_barrier  // (op 1031) 
s_add_u32 s1, s1, 1 // (op 163) 


v_mov_b64 v[16:17], v[20:21] // (op 421) 
v_mov_b64 v[18:19], v[16:17] // (op 421) reg expression

s_mov_b32 m0, s0 // (op 421) reg expression
buffer_load_dwordx4 v18, s[40:43], 0 offen offset: 0 lds // (op 421) Load value direct to lds


s_lshl_b64 s[14:15], s[2:3], 7 // (op 440) 
v_mov_b32 v10, s14 // (op 440) 
v_mov_b32 v14, s15 // (op 440) 
v_add_co_u32 v20, s[18:19], v20, v10 // (op 440) least significant half
v_addc_co_u32 v21, s[18:19], v21, v14, s[18:19] // (op 440) most significant half






v_mov_b32 v10, v37 // (op 179) 
v_mov_b32 v14, v10 // (op 179) reg expression

buffer_load_dwordx3 v[16:18], v14, s[44:47], 0 offen // (op 179) Load value
buffer_load_dwordx3 v[22:24], v14, s[44:47], 0 offen offset:12 // (op 179) Load value


s_lshl_b64 s[14:15], s[8:9], 7 // (op 509) 
s_mul_i32 s19, s15, 6 // (op 509) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 509) least significant
s_mul_hi_u32 s38, s14, 6 // (op 509) most significant: high of low * low
s_add_u32 s19, s19, s38 // (op 509) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 509) 

v_add_u32 v37, v37, s14 // (op 509) 



s_barrier  // (op 1044) 
v_mov_b32 v10, v13 // (op 180) 



v_mov_b32 v14, v10 // (op 180) reg expression

s_waitcnt vmcnt(1)
ds_write_b96 v12, v[16:18] offset:4096 // (op 177) Store local data 
buffer_load_ubyte v19, v14, s[52:55], 0 offen offset: 0 // (op 180) Load value


s_lshl_b32 s14, s24, 2 // (op 637) 
v_add_u32 v13, v13, s14 // (op 637) 

s_waitcnt vmcnt(1)
ds_write_b96 v12, v[22:24] offset:4112 // (op 177) Store local data 

s_barrier  // (op 1034) 



s_barrier  // (op 1058) 
ds_read_b64_tr_b8 v[22:23], v0 // (op 169) Transpose load local data 



ds_read_b64_tr_b8 v[24:25], v0 offset:128 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:1024 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:1152 // (op 169) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v15, v19 offset:8192 // (op 178) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 1048) 



ds_read_b96_tr_b6 v[30:32], v3 offset:4096 // (op 171) Transpose load local data 
ds_read_b96_tr_b6 v[33:35], v3 offset:4608 // (op 171) Transpose load local data 

v_mov_b64 v[16:17], s[12:13] // (op 173) convert

v_mov_b32 v10, v11 // (op 173) 
v_mov_b32 v14, v10 // (op 173) reg expression
v_mov_b32 v36, v14 // (op 173) 
buffer_load_ubyte v18, v36, s[48:51], 0 offen offset: 0 // (op 173) Load value




s_lshl_b32 s14, s12, 2 // (op 573) 
v_add_u32 v11, v11, s14 // (op 573) 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1062) 


ds_read_u8 v10, v4 offset:8192 // (op 175) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:35], a[0:3], v18, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 176) 




s_ashr_i64 s[14:15], s[6:7], 7 // (op 161) 
s_mov_b32 s18, s14 // (op 161) convert

s_cmp_lt_u32 s1, s18 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161 if true)
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161:






















s_barrier  // (op 1122) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

v_lshrrev_b32 v0, 7, v1 // (op 885) 
v_and_b32 v3, 63, v1 // (op 885) 
v_lshrrev_b32 v4, 4, v3 // (op 885) 
v_lshlrev_b32 v5, 2, v4 // (op 885) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 885) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 885) 

v_lshrrev_b32 v4, 2, v3 // (op 885) 

v_and_b32 v3, 3, v4 // (op 885) 

v_and_b32 v4, 3, v1 // (op 885) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 885) 


v_lshrrev_b32 v3, 6, v1 // (op 885) 
v_and_b32 v4, 1, v3 // (op 885) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 885) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 885) 

v_mul_lo_u32 v3, v4, s28 // (op 885) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 885) 


s_mul_i32 s1, s28, s33 // (op 887) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 887) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 887) least significant
s_mul_hi_u32 s3, s28, s32 // (op 887) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 887) most significant: sum
s_add_u32 s1, s1, s3 // (op 887) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 887) 

s_mov_b32 s4, s30 // (op 887) reg expression
s_mov_b32 s5, s31 // (op 887) reg expression
s_mov_b32 s6, s2 // (op 887) reg expression
s_mov_b32 s7, 131072 // (op 887) call()




buffer_store_dwordx4 v[10:13], v4, s[4:7], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123:
v_lshrrev_b32 v0, 7, v1 // (op 708) 
v_and_b32 v3, 63, v1 // (op 708) 
v_lshrrev_b32 v4, 4, v3 // (op 708) 
v_lshlrev_b32 v5, 2, v4 // (op 708) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 708) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 708) 

v_lshrrev_b32 v4, 2, v3 // (op 708) 

v_and_b32 v3, 3, v4 // (op 708) 

v_and_b32 v4, 3, v1 // (op 708) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 708) 


v_lshrrev_b32 v3, 6, v1 // (op 708) 
v_and_b32 v4, 1, v3 // (op 708) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 708) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 708) 

v_mul_lo_u32 v3, v4, s34 // (op 708) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 708) 


s_mul_i32 s1, s34, s33 // (op 710) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 710) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 710) least significant
s_mul_hi_u32 s3, s34, s32 // (op 710) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 710) most significant: sum
s_add_u32 s1, s1, s3 // (op 710) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 710) 

s_mov_b32 s8, s36 // (op 710) reg expression
s_mov_b32 s9, s37 // (op 710) reg expression
s_mov_b32 s10, s2 // (op 710) reg expression
s_mov_b32 s11, 131072 // (op 710) call()





buffer_load_dwordx4 v[10:13], v4, s[8:11], 0 offen // (op 135) Load value



v_accvgpr_read v4, a0 // (op 132) 
v_accvgpr_read v5, a1 // (op 132) 
v_accvgpr_read v6, a2 // (op 132) 
v_accvgpr_read v7, a3 // (op 132) 
v_mul_f32 v14, v9, v4 // (op 132) 
v_mul_f32 v15, v9, v5 // (op 132) 
v_mul_f32 v16, v9, v6 // (op 132) 
v_mul_f32 v17, v9, v7 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v4, v8, v10 // (op 136) 
v_mul_f32 v5, v8, v11 // (op 136) 
v_mul_f32 v6, v8, v12 // (op 136) 
v_mul_f32 v7, v8, v13 // (op 136) 

v_add_f32 v10, v4, v14 // (op 133) 
v_add_f32 v11, v5, v15 // (op 133) 
v_add_f32 v12, v6, v16 // (op 133) 
v_add_f32 v13, v7, v17 // (op 133) 


v_lshrrev_b32 v0, 7, v1 // (op 902) 
v_and_b32 v3, 63, v1 // (op 902) 
v_lshrrev_b32 v4, 4, v3 // (op 902) 
v_lshlrev_b32 v5, 2, v4 // (op 902) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 902) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 902) 

v_lshrrev_b32 v4, 2, v3 // (op 902) 

v_and_b32 v3, 3, v4 // (op 902) 

v_and_b32 v4, 3, v1 // (op 902) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 902) 


v_lshrrev_b32 v3, 6, v1 // (op 902) 
v_and_b32 v4, 1, v3 // (op 902) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 902) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 902) 

v_mul_lo_u32 v3, v4, s28 // (op 902) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 902) 


s_mul_i32 s1, s28, s33 // (op 904) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 904) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 904) least significant
s_mul_hi_u32 s3, s28, s32 // (op 904) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 904) most significant: sum
s_add_u32 s1, s1, s3 // (op 904) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 904) 

s_mov_b32 s4, s30 // (op 904) reg expression
s_mov_b32 s5, s31 // (op 904) reg expression
s_mov_b32 s6, s2 // (op 904) reg expression
s_mov_b32 s7, 131072 // (op 904) call()




buffer_store_dwordx4 v[10:13], v4, s[4:7], 0 offen // (op 134) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123:



s_endpgm  // End of RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.LRR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end:
.size RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2, .LRR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end-RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
  .amdhsa_next_free_vgpr 60
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12544 // lds bytes
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
    macrotile: [16, 32, 128]
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
  - .name: RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
    .symbol: RR_GEMM_NT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 12544
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 56
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

