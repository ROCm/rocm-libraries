// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.p2align 8
.type RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2,@function
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_exec_begin:


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
v_add_lshl_u32 v5, v0, v3, 5 // (op 765) 


v_and_b32 v0, 1, v4 // (op 765) 

v_lshlrev_b32 v3, 3, v0 // (op 765) 

v_lshrrev_b32 v0, 6, v1 // (op 765) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 765) 


v_add_u32 v0, v5, v4 // (op 765) 


v_and_b32 v3, 63, v1 // (op 818) 
v_and_b32 v4, 48, v1 // (op 818) 
v_sub_u32 v5, v3, v4 // (op 818) 

v_lshrrev_b32 v4, 4, v3 // (op 818) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 818) 


v_lshlrev_b32 v4, 4, v3 // (op 818) 
v_mul_lo_u32 v5, v4, 6 // (op 818) 

v_lshrrev_b32 v4, 3, v5 // (op 818) 

v_and_b32 v5, 268435455, v3 // (op 818) 

v_lshlrev_b32 v3, 2, v5 // (op 818) 

v_add_u32 v5, v4, v3 // (op 818) 


v_lshrrev_b32 v3, 6, v1 // (op 871) 
v_and_b32 v4, 15, v1 // (op 871) 
v_lshl_add_u32 v6, v3, 4, v4 // (op 871) 


v_and_b32 v3, 63, v1 // (op 871) 
v_lshrrev_b32 v4, 4, v3 // (op 871) 

v_lshl_add_u32 v3, v4, 5, v6 // (op 871) 


v_and_b32 v4, 48, v1 // (op 944) 
v_and_b32 v6, 63, v1 // (op 944) 
v_sub_u32 v7, v6, v4 // (op 944) 

v_lshrrev_b32 v6, 1, v7 // (op 944) 
v_add_lshl_u32 v8, v4, v6, 5 // (op 944) 


v_and_b32 v4, 1, v7 // (op 944) 

v_lshlrev_b32 v6, 3, v4 // (op 944) 

v_lshrrev_b32 v4, 6, v1 // (op 944) 
v_lshl_add_u32 v7, v4, 4, v6 // (op 944) 


v_add_u32 v4, v8, v7 // (op 944) 


v_and_b32 v6, 63, v1 // (op 968) 
v_and_b32 v7, 48, v1 // (op 968) 
v_sub_u32 v8, v6, v7 // (op 968) 

v_lshrrev_b32 v7, 4, v6 // (op 968) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 968) 


v_lshlrev_b32 v7, 4, v6 // (op 968) 
v_mul_lo_u32 v8, v7, 6 // (op 968) 

v_lshrrev_b32 v7, 3, v8 // (op 968) 

v_and_b32 v8, 268435455, v6 // (op 968) 

v_lshlrev_b32 v6, 2, v8 // (op 968) 

v_add_u32 v8, v7, v6 // (op 968) 


v_lshrrev_b32 v6, 6, v1 // (op 992) 
v_and_b32 v7, 15, v1 // (op 992) 
v_lshl_add_u32 v9, v6, 4, v7 // (op 992) 


v_and_b32 v6, 63, v1 // (op 992) 
v_lshrrev_b32 v7, 4, v6 // (op 992) 

v_lshl_add_u32 v6, v7, 5, v9 // (op 992) 


v_mov_b32 v7, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_and_b32 v10, -2, v1 // (op 476) 
v_sub_u32 v11, v1, v10 // (op 476) 

v_lshlrev_b32 v10, 4, v11 // (op 476) 

v_lshl_add_u32 v11, s16, 5, v10 // (op 476) 

v_mov_b32 v13, 0 // (op 476) convert
v_mov_b32 v12, v11 // (op 476) convert

v_lshrrev_b32 v10, 1, v1 // (op 476) 
v_mul_lo_u32 v15, v10, s3 // (op 476) most significant: low of low * high
v_mul_hi_u32 v11, v10, s2 // (op 476) most significant: high of low * low
v_mul_lo_u32 v14, v10, s2 // (op 476) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 476) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 476) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 476) most significant half



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

v_and_b32 v12, -2, v1 // (op 495) 
v_sub_u32 v13, v1, v12 // (op 495) 

v_lshlrev_b32 v12, 4, v13 // (op 495) 

v_lshl_add_u32 v13, s16, 5, v12 // (op 495) 

v_mov_b32 v15, 0 // (op 495) convert
v_mov_b32 v14, v13 // (op 495) convert

v_lshrrev_b32 v12, 1, v1 // (op 495) 
v_mul_lo_u32 v17, v12, s3 // (op 495) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 495) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 495) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 495) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 495) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 495) most significant half



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

s_lshl_b32 s0, s17, 4 // (op 540) 
s_mov_b32 s15, 0 // (op 540) convert
s_mov_b32 s14, s0 // (op 540) convert

v_mul_lo_u32 v15, v1, s9 // (op 540) most significant: low of low * high
v_mul_hi_u32 v16, v1, s8 // (op 540) most significant: high of low * low
v_mul_lo_u32 v14, v1, s8 // (op 540) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 540) most significant: sum

v_mov_b32 v16, s14 // (op 540) 
v_mov_b32 v17, s15 // (op 540) 
v_add_co_u32 v14, s[0:1], v14, v16 // (op 540) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 540) most significant half




v_mul_lo_u32 v17, v15, 6 // (op 540) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 540) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 540) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 540) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 540) 

v_mov_b32 v16, v14 // (op 540) convert

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

s_lshl_b32 s0, s17, 4 // (op 559) 
s_mov_b32 s15, 0 // (op 559) convert
s_mov_b32 s14, s0 // (op 559) convert

v_mul_lo_u32 v15, v1, s9 // (op 559) most significant: low of low * high
v_mul_hi_u32 v17, v1, s8 // (op 559) most significant: high of low * low
v_mul_lo_u32 v14, v1, s8 // (op 559) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 559) most significant: sum

v_mov_b32 v17, s14 // (op 559) 
v_mov_b32 v18, s15 // (op 559) 
v_add_co_u32 v14, s[0:1], v14, v17 // (op 559) least significant half
v_addc_co_u32 v15, s[0:1], v15, v18, s[0:1] // (op 559) most significant half




v_mul_lo_u32 v19, v15, 6 // (op 559) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 559) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 559) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 559) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 559) 

v_mov_b32 v17, v14 // (op 559) convert

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

v_and_b32 v14, -32, v1 // (op 604) 
v_sub_u32 v15, v1, v14 // (op 604) 

v_lshl_add_u32 v14, s16, 5, v15 // (op 604) 

v_lshrrev_b32 v15, 5, v1 // (op 604) 
v_mul_lo_u32 v18, v15, s12 // (op 604) 

v_add_u32 v15, v14, v18 // (op 604) 



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

v_and_b32 v14, -32, v1 // (op 623) 
v_sub_u32 v18, v1, v14 // (op 623) 

v_lshl_add_u32 v14, s16, 5, v18 // (op 623) 

v_lshrrev_b32 v18, 5, v1 // (op 623) 
v_mul_lo_u32 v19, v18, s12 // (op 623) 

v_add_u32 v18, v14, v19 // (op 623) 


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

v_and_b32 v14, 15, v1 // (op 668) 
v_lshl_add_u32 v19, s17, 4, v14 // (op 668) 

v_and_b32 v14, 63, v1 // (op 668) 
v_lshrrev_b32 v20, 4, v14 // (op 668) 

v_mul_lo_u32 v14, v20, s24 // (op 668) 
v_add_u32 v14, v14, v19 // (op 668) 


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

v_and_b32 v19, 15, v1 // (op 687) 
v_lshl_add_u32 v20, s17, 4, v19 // (op 687) 

v_and_b32 v19, 63, v1 // (op 687) 
v_lshrrev_b32 v21, 4, v19 // (op 687) 

v_mul_lo_u32 v19, v21, s24 // (op 687) 
v_add_u32 v19, v19, v20 // (op 687) 


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


v_mov_b64 v[20:21], s[24:25] // (op 341) convert

v_mov_b32 v22, v14 // (op 341) reg expression
v_mov_b32 v23, v22 // (op 341) 
buffer_load_ubyte v24, v23, s[52:55], 0 offen offset: 0 // (op 341) Load value





v_mov_b32 v20, v16 // (op 345) reg expression

buffer_load_dwordx3 v[26:28], v20, s[44:47], 0 offen // (op 345) Load value

v_and_b32 v25, -2, v1 // (op 753) 
v_sub_u32 v20, v1, v25 // (op 753) 

v_lshlrev_b32 v25, 4, v20 // (op 753) 

v_lshrrev_b32 v20, 1, v1 // (op 753) 
v_lshl_add_u32 v21, v20, 5, v25 // (op 753) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 753) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 432) 
v_mov_b64 v[20:21], s[14:15] // (op 432) convert


v_mov_b64 v[22:23], v[10:11] // (op 432) reg expression
s_mov_b32 m0, s0 // (op 432) reg expression
buffer_load_dwordx4 v22, s[40:43], 0 offen offset: 0 lds // (op 432) Load value direct to lds
v_add_co_u32 v22, s[14:15], v22, v20 // (op 432) least significant half
v_addc_co_u32 v23, s[14:15], 0, v23, s[14:15] // (op 432) most significant half

s_add_u32 m0, m0, 2048 // (op 432) 
buffer_load_dwordx4 v22, s[40:43], 0 offen offset: 0 lds // (op 432) Load value direct to lds





v_mov_b32 v25, v15 // (op 351) reg expression

buffer_load_ubyte v20, v25, s[48:51], 0 offen offset: 0 // (op 351) Load value


v_mov_b64 v[22:23], s[24:25] // (op 355) convert

s_lshl_b32 s0, s24, 2 // (op 355) 
v_add_u32 v21, v14, s0 // (op 355) 

v_mov_b32 v25, v21 // (op 355) 
buffer_load_ubyte v29, v25, s[52:55], 0 offen offset: 0 // (op 355) Load value






s_lshl_b64 s[0:1], s[8:9], 7 // (op 359) 
s_mul_i32 s15, s1, 6 // (op 359) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 359) least significant
s_mul_hi_u32 s18, s0, 6 // (op 359) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 359) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 359) 

v_add_u32 v14, v16, s0 // (op 359) 


buffer_load_dwordx3 v[30:32], v14, s[44:47], 0 offen // (op 359) Load value


v_and_b32 v14, -2, v1 // (op 932) 
v_sub_u32 v16, v1, v14 // (op 932) 

v_lshlrev_b32 v14, 4, v16 // (op 932) 

v_lshrrev_b32 v16, 1, v1 // (op 932) 
v_lshl_add_u32 v25, v16, 5, v14 // (op 932) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 932) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 437) 
v_mov_b64 v[22:23], s[14:15] // (op 437) convert


s_lshl_b64 s[14:15], s[2:3], 7 // (op 437) 
v_mov_b32 v14, s14 // (op 437) 
v_mov_b32 v16, s15 // (op 437) 
v_add_co_u32 v34, s[18:19], v10, v14 // (op 437) least significant half
v_addc_co_u32 v35, s[18:19], v11, v16, s[18:19] // (op 437) most significant half




s_add_u32 s1, 4096, s0 // (op 437) 
s_mov_b32 m0, s1 // (op 437) reg expression
buffer_load_dwordx4 v34, s[40:43], 0 offen offset: 0 lds // (op 437) Load value direct to lds
v_add_co_u32 v34, s[14:15], v34, v22 // (op 437) least significant half
v_addc_co_u32 v35, s[14:15], 0, v35, s[14:15] // (op 437) most significant half

s_add_u32 m0, m0, 2048 // (op 437) 
buffer_load_dwordx4 v34, s[40:43], 0 offen offset: 0 lds // (op 437) Load value direct to lds







s_lshl_b32 s0, s12, 2 // (op 365) 
v_add_u32 v14, v15, s0 // (op 365) 


buffer_load_ubyte v16, v14, s[48:51], 0 offen offset: 0 // (op 365) Load value


v_lshlrev_b32 v25, 4, v1 // (op 806) 
v_mul_lo_u32 v10, v25, 6 // (op 806) 

v_lshrrev_b32 v11, 3, v10 // (op 806) 

v_and_b32 v10, 268435455, v1 // (op 806) 
v_lshlrev_b32 v25, 2, v10 // (op 806) 

v_add_u32 v10, v11, v25 // (op 806) 





s_waitcnt vmcnt(8)
ds_write_b96 v10, v[26:28] offset:8192 // (op 367) Store local data 

v_and_b32 v25, -32, v1 // (op 859) 
v_sub_u32 v10, v1, v25 // (op 859) 

v_and_b32 v11, 4294967264, v1 // (op 859) 
v_add_u32 v25, v11, v10 // (op 859) 





s_waitcnt vmcnt(5)
ds_write_b8 v25, v20 offset:10240 // (op 369) Store local data 

v_and_b32 v25, -2, v1 // (op 725) 
v_sub_u32 v10, v1, v25 // (op 725) 

v_lshlrev_b32 v11, 4, v10 // (op 725) 

v_lshrrev_b32 v10, 1, v1 // (op 725) 
v_lshl_add_u32 v25, v10, 5, v11 // (op 725) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 725) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v25, 4, v1 // (op 777) 
v_mul_lo_u32 v10, v25, 6 // (op 777) 

v_lshrrev_b32 v11, 3, v10 // (op 777) 

v_and_b32 v10, 268435455, v1 // (op 777) 
v_lshlrev_b32 v25, 2, v10 // (op 777) 

v_add_u32 v10, v11, v25 // (op 777) 


v_and_b32 v11, -32, v1 // (op 830) 
v_sub_u32 v25, v1, v11 // (op 830) 

v_and_b32 v11, 4294967264, v1 // (op 830) 
v_add_u32 v14, v11, v25 // (op 830) 


v_and_b32 v11, -2, v1 // (op 917) 
v_sub_u32 v15, v1, v11 // (op 917) 

v_lshlrev_b32 v11, 4, v15 // (op 917) 

v_lshrrev_b32 v15, 1, v1 // (op 917) 
v_lshl_add_u32 v25, v15, 5, v11 // (op 917) 


s_nop 0
v_readfirstlane_b32 s1, v25 // (op 917) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v11, 4, v1 // (op 956) 
v_mul_lo_u32 v15, v11, 6 // (op 956) 

v_lshrrev_b32 v11, 3, v15 // (op 956) 

v_and_b32 v15, 268435455, v1 // (op 956) 
v_lshlrev_b32 v25, 2, v15 // (op 956) 

v_add_u32 v15, v11, v25 // (op 956) 


v_and_b32 v11, -32, v1 // (op 980) 
v_sub_u32 v25, v1, v11 // (op 980) 

v_and_b32 v11, 4294967264, v1 // (op 980) 
v_add_u32 v21, v11, v25 // (op 980) 


s_mov_b32 s14, 0 // (op 57) call()
s_ashr_i64 s[18:19], s[6:7], 7 // (op 56) 
s_mov_b32 s15, s18 // (op 56) convert

s_and_b32 s18, s15, 4294967294 // (op 56) 

s_cmp_lt_u32 s14, s18 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_56 if false)
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_56:
s_barrier  // (op 1025) 
s_add_u32 s14, s14, 2 // (op 58) 



ds_read_b64_tr_b8 v[34:35], v0 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:256 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:2048 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:2304 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1038) 



ds_read_b96_tr_b6 v[42:44], v5 offset:8192 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[45:47], v5 offset:8448 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1052) 


ds_read_u8 v11, v3 offset:10240 // (op 77) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[42:47], a[0:3], v11, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 84) 




v_mov_b64 v[22:23], s[24:25] // (op 82) convert

v_mov_b32 v11, v19 // (op 82) 
s_lshl_b32 s15, s24, 3 // (op 82) 
v_add_u32 v25, v11, s15 // (op 82) 

v_mov_b32 v33, v25 // (op 82) 
buffer_load_ubyte v24, v33, s[52:55], 0 offen offset: 0 // (op 82) Load value





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

s_barrier  // (op 1028) 

s_lshl_b64 s[18:19], s[2:3], 6 // (op 416) 
v_mov_b64 v[22:23], s[18:19] // (op 416) convert


v_mov_b64 v[34:35], v[12:13] // (op 416) 
s_lshl_b64 s[18:19], s[2:3], 7 // (op 416) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 416) 

v_mov_b32 v33, s38 // (op 416) 
v_mov_b32 v36, s39 // (op 416) 
v_add_co_u32 v38, s[18:19], v34, v33 // (op 416) least significant half
v_addc_co_u32 v39, s[18:19], v35, v36, s[18:19] // (op 416) most significant half




s_mov_b32 m0, s0 // (op 416) reg expression
buffer_load_dwordx4 v38, s[40:43], 0 offen offset: 0 lds // (op 416) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v22 // (op 416) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 416) most significant half

s_add_u32 m0, m0, 2048 // (op 416) 
buffer_load_dwordx4 v38, s[40:43], 0 offen offset: 0 lds // (op 416) Load value direct to lds




v_mov_b32 v33, v18 // (op 17) 
s_lshl_b32 s15, s12, 3 // (op 17) 
v_add_u32 v22, v33, s15 // (op 17) 


buffer_load_ubyte v20, v22, s[48:51], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1075) 



s_waitcnt vmcnt(8)
ds_write_b96 v15, v[30:32] offset:10368 // (op 239) Store local data 
s_barrier  // (op 1081) 



s_waitcnt vmcnt(5)
ds_write_b8 v21, v16 offset:12416 // (op 240) Store local data 
s_barrier  // (op 1066) 



ds_read_b64_tr_b8 v[36:37], v4 offset:4096 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:4352 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:6144 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:6400 // (op 231) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1072) 



ds_read_b96_tr_b6 v[44:46], v8 offset:10368 // (op 233) Transpose load local data 
ds_read_b96_tr_b6 v[47:49], v8 offset:10624 // (op 233) Transpose load local data 
s_barrier  // (op 1132) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1078) 


ds_read_u8 v22, v6 offset:12416 // (op 235) Load local data 
s_barrier  // (op 1138) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:49], a[0:3], v22, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 238) 




v_mov_b64 v[22:23], s[24:25] // (op 237) convert

s_lshl_b32 s15, s24, 2 // (op 237) 
s_mul_i32 s18, 3, s15 // (op 237) 

v_add_u32 v36, v11, s18 // (op 237) 

v_mov_b32 v37, v36 // (op 237) 
buffer_load_ubyte v29, v37, s[52:55], 0 offen offset: 0 // (op 237) Load value






s_lshl_b32 s15, s24, 3 // (op 679) 
v_add_u32 v19, v19, s15 // (op 679) 

s_lshl_b64 s[18:19], s[8:9], 7 // (op 241) 
s_mul_i32 s39, s19, 6 // (op 241) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 241) least significant
s_mul_hi_u32 s15, s18, 6 // (op 241) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 241) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 241) 

s_mul_i32 s15, 3, s18 // (op 241) 

v_add_u32 v11, v25, s15 // (op 241) 


buffer_load_dwordx3 v[30:32], v11, s[44:47], 0 offen // (op 241) Load value


s_barrier  // (op 1069) 
s_lshl_b64 s[18:19], s[8:9], 7 // (op 551) 

s_mul_i32 s39, s19, 6 // (op 551) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 551) least significant
s_mul_hi_u32 s15, s18, 6 // (op 551) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 551) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 551) 

s_lshl_b32 s15, s18, 1 // (op 551) 

v_add_u32 v17, v17, s15 // (op 551) 

s_lshl_b64 s[18:19], s[2:3], 6 // (op 427) 
v_mov_b64 v[22:23], s[18:19] // (op 427) convert


s_lshl_b64 s[18:19], s[2:3], 7 // (op 427) 
s_mul_i32 s39, 3, s19 // (op 427) most significant: low of low * high
s_mul_i32 s38, 3, s18 // (op 427) least significant
s_mul_hi_u32 s15, 3, s18 // (op 427) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 427) most significant: sum


v_mov_b32 v11, s38 // (op 427) 
v_mov_b32 v25, s39 // (op 427) 
v_add_co_u32 v36, s[18:19], v34, v11 // (op 427) least significant half
v_addc_co_u32 v37, s[18:19], v35, v25, s[18:19] // (op 427) most significant half




s_add_u32 s15, 4096, s1 // (op 427) 
s_mov_b32 m0, s15 // (op 427) reg expression
buffer_load_dwordx4 v36, s[40:43], 0 offen offset: 0 lds // (op 427) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v22 // (op 427) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 427) most significant half

s_add_u32 m0, m0, 2048 // (op 427) 
buffer_load_dwordx4 v36, s[40:43], 0 offen offset: 0 lds // (op 427) Load value direct to lds






s_lshl_b32 s15, s12, 2 // (op 242) 
s_mul_i32 s18, 3, s15 // (op 242) 

v_add_u32 v11, v33, s18 // (op 242) 


s_lshl_b64 s[18:19], s[2:3], 7 // (op 487) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 487) 

v_mov_b32 v25, s38 // (op 487) 
v_mov_b32 v22, s39 // (op 487) 
v_add_co_u32 v12, s[18:19], v12, v25 // (op 487) least significant half
v_addc_co_u32 v13, s[18:19], v13, v22, s[18:19] // (op 487) most significant half




buffer_load_ubyte v16, v11, s[48:51], 0 offen offset: 0 // (op 242) Load value


s_barrier  // (op 1041) 
s_lshl_b32 s15, s12, 3 // (op 615) 



v_add_u32 v18, v18, s15 // (op 615) 

s_waitcnt vmcnt(8)
ds_write_b96 v10, v[26:28] offset:8192 // (op 13) Store local data 
s_barrier  // (op 1055) 



s_waitcnt vmcnt(5)
ds_write_b8 v14, v20 offset:10240 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[6:7], 7 // (op 56) 
s_mov_b32 s15, s18 // (op 56) convert

s_and_b32 s18, s15, 4294967294 // (op 56) 

s_cmp_lt_u32 s14, s18 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_56 if true)
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_56:














s_barrier  // (op 1094) 
v_and_b32 v10, -2, v1 // (op 450) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 450) 
v_sub_u32 v11, v1, v10 // (op 450) 

s_lshr_b32 s14, s0, 1 // (op 450) 

v_lshlrev_b32 v10, 4, v11 // (op 450) 

v_lshl_add_u32 v11, s16, 5, v10 // (op 450) 

v_mov_b32 v13, 0 // (op 450) convert
v_mov_b32 v12, v11 // (op 450) convert

v_lshrrev_b32 v10, 1, v1 // (op 450) 
v_lshl_add_u32 v11, s14, 8, v10 // (op 450) 


v_mul_lo_u32 v15, v11, s3 // (op 450) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 450) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 450) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 450) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 450) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 450) most significant half



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

v_lshl_add_u32 v12, s14, 8, v1 // (op 517) 

s_lshl_b32 s0, s17, 4 // (op 517) 
s_mov_b32 s15, 0 // (op 517) convert
s_mov_b32 s14, s0 // (op 517) convert

v_mul_lo_u32 v15, v12, s9 // (op 517) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 517) most significant: high of low * low
v_mul_lo_u32 v14, v12, s8 // (op 517) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 517) most significant: sum

v_mov_b32 v13, s14 // (op 517) 
v_mov_b32 v17, s15 // (op 517) 
v_add_co_u32 v14, s[0:1], v14, v13 // (op 517) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 517) most significant half





v_mul_lo_u32 v13, v15, 6 // (op 517) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 517) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 517) least significant: low of low * low
v_add_u32 v13, v13, v17 // (op 517) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 517) 

v_mov_b32 v12, v14 // (op 517) convert

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

v_and_b32 v13, -32, v1 // (op 581) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 581) 
v_sub_u32 v14, v1, v13 // (op 581) 

s_lshr_b32 s14, s0, 1 // (op 581) 

v_lshl_add_u32 v13, s16, 5, v14 // (op 581) 

v_lshrrev_b32 v14, 5, v1 // (op 581) 
v_lshl_add_u32 v15, s14, 3, v14 // (op 581) 


v_mul_lo_u32 v14, v15, s12 // (op 581) 

v_add_u32 v15, v13, v14 // (op 581) 


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

v_and_b32 v13, 63, v1 // (op 645) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 645) 
v_lshrrev_b32 v14, 4, v13 // (op 645) 

s_lshr_b32 s14, s0, 1 // (op 645) 

v_and_b32 v13, 15, v1 // (op 645) 
v_lshl_add_u32 v17, s14, 3, v14 // (op 645) 


v_lshl_add_u32 v14, s17, 4, v13 // (op 645) 

v_mul_lo_u32 v13, v17, s24 // (op 645) 
v_add_u32 v13, v13, v14 // (op 645) 


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

v_and_b32 v14, -2, v1 // (op 738) 
v_sub_u32 v17, v1, v14 // (op 738) 

v_lshlrev_b32 v14, 4, v17 // (op 738) 

v_lshrrev_b32 v17, 1, v1 // (op 738) 
v_lshl_add_u32 v18, v17, 5, v14 // (op 738) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 738) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v14, 4, v1 // (op 790) 
v_mul_lo_u32 v17, v14, 6 // (op 790) 

v_lshrrev_b32 v14, 3, v17 // (op 790) 

v_and_b32 v17, 268435455, v1 // (op 790) 
v_lshlrev_b32 v18, 2, v17 // (op 790) 

v_add_u32 v17, v14, v18 // (op 790) 


v_and_b32 v14, -32, v1 // (op 843) 
v_sub_u32 v18, v1, v14 // (op 843) 

v_and_b32 v14, 4294967264, v1 // (op 843) 
v_add_u32 v19, v14, v18 // (op 843) 


s_ashr_i64 s[14:15], s[6:7], 7 // (op 166) 
s_and_b32 s1, s14, 4294967294 // (op 166) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 165) 
s_mov_b32 s18, s14 // (op 165) convert

s_cmp_lt_u32 s1, s18 // (op 165) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_165:
s_barrier  // (op 1031) 
s_add_u32 s1, s1, 1 // (op 167) 

s_lshl_b64 s[14:15], s[2:3], 6 // (op 421) 
v_mov_b64 v[20:21], s[14:15] // (op 421) convert


v_mov_b64 v[22:23], v[10:11] // (op 421) 
v_mov_b64 v[24:25], v[22:23] // (op 421) reg expression
s_mov_b32 m0, s0 // (op 421) reg expression
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 421) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v20 // (op 421) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 421) most significant half

s_add_u32 m0, m0, 2048 // (op 421) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 421) Load value direct to lds



s_lshl_b64 s[14:15], s[2:3], 7 // (op 440) 
v_mov_b32 v14, s14 // (op 440) 
v_mov_b32 v16, s15 // (op 440) 
v_add_co_u32 v10, s[18:19], v10, v14 // (op 440) least significant half
v_addc_co_u32 v11, s[18:19], v11, v16, s[18:19] // (op 440) most significant half






v_mov_b32 v14, v12 // (op 183) 
v_mov_b32 v16, v14 // (op 183) reg expression

buffer_load_dwordx3 v[20:22], v16, s[44:47], 0 offen // (op 183) Load value


s_lshl_b64 s[14:15], s[8:9], 7 // (op 509) 
s_mul_i32 s19, s15, 6 // (op 509) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 509) least significant
s_mul_hi_u32 s38, s14, 6 // (op 509) most significant: high of low * low
s_add_u32 s19, s19, s38 // (op 509) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 509) 

v_add_u32 v12, v12, s14 // (op 509) 



s_barrier  // (op 1044) 
v_mov_b32 v14, v15 // (op 184) 



v_mov_b32 v16, v14 // (op 184) reg expression

s_waitcnt vmcnt(0)
ds_write_b96 v17, v[20:22] offset:8192 // (op 181) Store local data 
buffer_load_ubyte v18, v16, s[48:51], 0 offen offset: 0 // (op 184) Load value



s_lshl_b32 s14, s12, 2 // (op 573) 
v_add_u32 v15, v15, s14 // (op 573) 

s_barrier  // (op 1034) 
s_barrier  // (op 1058) 






ds_read_b64_tr_b8 v[20:21], v0 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[22:23], v0 offset:256 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:2048 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:2304 // (op 173) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v18 offset:10240 // (op 182) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 1048) 



ds_read_b96_tr_b6 v[28:30], v5 offset:8192 // (op 175) Transpose load local data 
ds_read_b96_tr_b6 v[31:33], v5 offset:8448 // (op 175) Transpose load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1062) 


ds_read_u8 v14, v3 offset:10240 // (op 177) Load local data 

v_mov_b64 v[34:35], s[24:25] // (op 179) convert

v_mov_b32 v16, v13 // (op 179) 
v_mov_b32 v18, v16 // (op 179) reg expression
v_mov_b32 v36, v18 // (op 179) 
buffer_load_ubyte v37, v36, s[52:55], 0 offen offset: 0 // (op 179) Load value




s_lshl_b32 s14, s24, 2 // (op 637) 
v_add_u32 v13, v13, s14 // (op 637) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[28:33], a[0:3], v14, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 180) 




s_ashr_i64 s[14:15], s[6:7], 7 // (op 165) 
s_mov_b32 s18, s14 // (op 165) convert

s_cmp_lt_u32 s1, s18 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_165 if true)
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_165:






















s_barrier  // (op 1126) 






v_cmp_eq_f32 vcc, 0.00000, v7 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_125
s_nop 0
v_accvgpr_read v10, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 

v_lshrrev_b32 v10, 16, v14 // (op 52) 
v_lshrrev_b32 v11, 16, v15 // (op 52) 
v_lshrrev_b32 v12, 16, v16 // (op 52) 
v_lshrrev_b32 v13, 16, v17 // (op 52) 

v_lshrrev_b32 v0, 6, v1 // (op 885) 
v_and_b32 v8, 63, v1 // (op 885) 
v_lshrrev_b32 v3, 4, v8 // (op 885) 
v_lshlrev_b32 v4, 2, v3 // (op 885) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 885) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 885) 

v_lshrrev_b32 v3, 2, v8 // (op 885) 

v_and_b32 v8, 3, v3 // (op 885) 

v_and_b32 v3, 3, v1 // (op 885) 
v_lshl_add_u32 v4, v8, 2, v3 // (op 885) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 885) 

v_mul_lo_u32 v8, v3, s28 // (op 885) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 885) 


s_mul_i32 s1, s28, s33 // (op 887) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 887) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 887) least significant
s_mul_hi_u32 s3, s28, s32 // (op 887) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 887) most significant: sum
s_add_u32 s1, s1, s3 // (op 887) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 887) 

s_mov_b32 s4, s30 // (op 887) reg expression
s_mov_b32 s5, s31 // (op 887) reg expression
s_mov_b32 s6, s2 // (op 887) reg expression
s_mov_b32 s7, 131072 // (op 887) call()


v_pack_b32_f16 v4, v10, v11 // (op 54) 
v_pack_b32_f16 v5, v12, v13 // (op 54) 


buffer_store_dwordx2 v[4:5], v3, s[4:7], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_125:
v_lshrrev_b32 v0, 6, v1 // (op 708) 
v_and_b32 v8, 63, v1 // (op 708) 
v_lshrrev_b32 v3, 4, v8 // (op 708) 
v_lshlrev_b32 v4, 2, v3 // (op 708) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 708) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 708) 

v_lshrrev_b32 v3, 2, v8 // (op 708) 

v_and_b32 v8, 3, v3 // (op 708) 

v_and_b32 v3, 3, v1 // (op 708) 
v_lshl_add_u32 v4, v8, 2, v3 // (op 708) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 708) 

v_mul_lo_u32 v8, v3, s34 // (op 708) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 708) 


s_mul_i32 s1, s34, s33 // (op 710) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 710) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 710) least significant
s_mul_hi_u32 s3, s34, s32 // (op 710) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 710) most significant: sum
s_add_u32 s1, s1, s3 // (op 710) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 710) 

s_mov_b32 s8, s36 // (op 710) reg expression
s_mov_b32 s9, s37 // (op 710) reg expression
s_mov_b32 s10, s2 // (op 710) reg expression
s_mov_b32 s11, 131072 // (op 710) call()





buffer_load_dwordx2 v[4:5], v3, s[8:11], 0 offen // (op 138) Load value



v_accvgpr_read v10, a0 // (op 134) 
v_accvgpr_read v11, a1 // (op 134) 
v_accvgpr_read v12, a2 // (op 134) 
v_accvgpr_read v13, a3 // (op 134) 
v_mul_f32 v14, v9, v10 // (op 134) 
v_mul_f32 v15, v9, v11 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 139) 
v_lshrrev_b32 v11, 16, v4 // (op 139) 
v_lshlrev_b32 v10, 16, v10 // (op 139) 
v_lshlrev_b32 v11, 16, v11 // (op 139) 
v_mul_f32 v18, v7, v10 // (op 139) 
v_mul_f32 v19, v7, v11 // (op 139) 
v_and_b32 v10, 65535, v5 // (op 139) 
v_lshrrev_b32 v11, 16, v5 // (op 139) 
v_lshlrev_b32 v10, 16, v10 // (op 139) 
v_lshlrev_b32 v11, 16, v11 // (op 139) 
v_mul_f32 v20, v7, v10 // (op 139) 
v_mul_f32 v21, v7, v11 // (op 139) 


v_add_f32 v10, v18, v14 // (op 135) 
v_add_f32 v11, v19, v15 // (op 135) 
v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 


v_lshrrev_b32 v14, 16, v10 // (op 136) 
v_lshrrev_b32 v15, 16, v11 // (op 136) 
v_lshrrev_b32 v16, 16, v12 // (op 136) 
v_lshrrev_b32 v17, 16, v13 // (op 136) 

v_lshrrev_b32 v0, 6, v1 // (op 902) 
v_and_b32 v8, 63, v1 // (op 902) 
v_lshrrev_b32 v3, 4, v8 // (op 902) 
v_lshlrev_b32 v4, 2, v3 // (op 902) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 902) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 902) 

v_lshrrev_b32 v3, 2, v8 // (op 902) 

v_and_b32 v8, 3, v3 // (op 902) 

v_and_b32 v3, 3, v1 // (op 902) 
v_lshl_add_u32 v4, v8, 2, v3 // (op 902) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 902) 

v_mul_lo_u32 v8, v3, s28 // (op 902) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 902) 


s_mul_i32 s1, s28, s33 // (op 904) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 904) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 904) least significant
s_mul_hi_u32 s3, s28, s32 // (op 904) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 904) most significant: sum
s_add_u32 s1, s1, s3 // (op 904) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 904) 

s_mov_b32 s4, s30 // (op 904) reg expression
s_mov_b32 s5, s31 // (op 904) reg expression
s_mov_b32 s6, s2 // (op 904) reg expression
s_mov_b32 s7, 131072 // (op 904) call()


v_pack_b32_f16 v4, v14, v15 // (op 137) 
v_pack_b32_f16 v5, v16, v17 // (op 137) 


buffer_store_dwordx2 v[4:5], v3, s[4:7], 0 offen // (op 137) Store value







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_125:



s_endpgm  // End of RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.LRR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end:
.size RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2, .LRR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end-RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
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
    DataTypeB: F6
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
  - .name: RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
    .symbol: RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2.kd
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

