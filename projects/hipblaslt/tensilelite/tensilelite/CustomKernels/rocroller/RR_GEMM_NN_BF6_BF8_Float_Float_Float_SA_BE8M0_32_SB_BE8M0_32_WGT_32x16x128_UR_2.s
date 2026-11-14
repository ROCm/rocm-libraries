// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.p2align 8
.type RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2,@function
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 763) 
v_and_b32 v3, 48, v1 // (op 763) 
v_sub_u32 v4, v0, v3 // (op 763) 

v_lshrrev_b32 v3, 4, v0 // (op 763) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 763) 


v_lshrrev_b32 v3, 6, v1 // (op 763) 
v_lshlrev_b32 v4, 4, v3 // (op 763) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 763) 


v_mul_lo_u32 v0, v3, 6 // (op 763) 
v_lshrrev_b32 v4, 3, v0 // (op 763) 

v_lshrrev_b32 v0, 4, v3 // (op 763) 

v_lshlrev_b32 v3, 2, v0 // (op 763) 

v_add_u32 v0, v4, v3 // (op 763) 


v_and_b32 v3, 15, v1 // (op 815) 
v_and_b32 v4, 48, v1 // (op 815) 
v_lshl_add_u32 v5, v3, 7, v4 // (op 815) 


v_lshrrev_b32 v3, 6, v1 // (op 868) 
v_and_b32 v4, 15, v1 // (op 868) 
v_lshl_add_u32 v6, v3, 4, v4 // (op 868) 


v_and_b32 v3, 63, v1 // (op 868) 
v_lshrrev_b32 v4, 4, v3 // (op 868) 

v_lshl_add_u32 v3, v4, 5, v6 // (op 868) 


v_and_b32 v4, 63, v1 // (op 926) 
v_and_b32 v6, 48, v1 // (op 926) 
v_sub_u32 v7, v4, v6 // (op 926) 

v_lshrrev_b32 v6, 4, v4 // (op 926) 

v_lshl_add_u32 v4, v6, 5, v7 // (op 926) 


v_lshrrev_b32 v6, 6, v1 // (op 926) 
v_lshlrev_b32 v7, 4, v6 // (op 926) 

v_lshl_add_u32 v6, v4, 5, v7 // (op 926) 


v_mul_lo_u32 v4, v6, 6 // (op 926) 
v_lshrrev_b32 v7, 3, v4 // (op 926) 

v_lshrrev_b32 v4, 4, v6 // (op 926) 

v_lshlrev_b32 v6, 2, v4 // (op 926) 

v_add_u32 v4, v7, v6 // (op 926) 


v_and_b32 v6, 15, v1 // (op 965) 
v_and_b32 v7, 48, v1 // (op 965) 
v_lshl_add_u32 v8, v6, 7, v7 // (op 965) 


v_lshrrev_b32 v6, 6, v1 // (op 989) 
v_and_b32 v7, 15, v1 // (op 989) 
v_lshl_add_u32 v9, v6, 4, v7 // (op 989) 


v_and_b32 v6, 63, v1 // (op 989) 
v_lshrrev_b32 v7, 4, v6 // (op 989) 

v_lshl_add_u32 v6, v7, 5, v9 // (op 989) 


v_mov_b32 v7, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
s_lshl_b32 s0, s16, 5 // (op 473) 
v_mul_lo_u32 v11, v1, s3 // (op 473) most significant: low of low * high
s_mov_b32 s15, 0 // (op 473) convert
s_mov_b32 s14, s0 // (op 473) convert

v_mul_hi_u32 v12, v1, s2 // (op 473) most significant: high of low * low
v_mul_lo_u32 v10, v1, s2 // (op 473) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 473) most significant: sum

v_mov_b32 v12, s14 // (op 473) 
v_mov_b32 v13, s15 // (op 473) 
v_add_co_u32 v14, s[0:1], v12, v10 // (op 473) least significant half
v_addc_co_u32 v15, s[0:1], v13, v11, s[0:1] // (op 473) most significant half





v_mul_lo_u32 v11, v15, 6 // (op 473) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 473) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 473) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 473) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 473) 

v_mov_b32 v10, v12 // (op 473) convert

s_mul_i32 s1, s7, 6 // (op 475) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 475) least significant
s_mul_hi_u32 s14, s6, 6 // (op 475) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 475) most significant: sum

s_mul_i32 s15, s2, s1 // (op 475) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 475) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 475) least significant
s_mul_hi_u32 s19, s2, s0 // (op 475) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 475) most significant: sum
s_add_u32 s15, s15, s19 // (op 475) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 475) 

s_mov_b32 s40, s4 // (op 475) reg expression
s_mov_b32 s41, s5 // (op 475) reg expression
s_mov_b32 s42, s0 // (op 475) reg expression
s_mov_b32 s43, 131072 // (op 475) call()

s_lshl_b32 s0, s16, 5 // (op 492) 
v_mul_lo_u32 v13, v1, s3 // (op 492) most significant: low of low * high
v_mul_hi_u32 v11, v1, s2 // (op 492) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 492) least significant: low of low * low
v_add_u32 v13, v13, v11 // (op 492) most significant: sum

s_mov_b32 s15, 0 // (op 492) convert
s_mov_b32 s14, s0 // (op 492) convert

v_mov_b32 v11, s14 // (op 492) 
v_mov_b32 v14, s15 // (op 492) 
v_add_co_u32 v16, s[0:1], v11, v12 // (op 492) least significant half
v_addc_co_u32 v17, s[0:1], v14, v13, s[0:1] // (op 492) most significant half





v_mul_lo_u32 v15, v17, 6 // (op 492) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 492) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 492) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 492) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 492) 

v_mov_b32 v11, v12 // (op 492) convert

s_mul_i32 s1, s7, 6 // (op 494) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 494) least significant
s_mul_hi_u32 s14, s6, 6 // (op 494) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 494) most significant: sum

s_mul_i32 s15, s2, s1 // (op 494) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 494) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 494) least significant
s_mul_hi_u32 s19, s2, s0 // (op 494) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 494) most significant: sum
s_add_u32 s15, s15, s19 // (op 494) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 494) 

s_mov_b32 s40, s4 // (op 494) reg expression
s_mov_b32 s41, s5 // (op 494) reg expression
s_mov_b32 s42, s0 // (op 494) reg expression
s_mov_b32 s43, 131072 // (op 494) call()

v_and_b32 v12, -8, v1 // (op 537) 
v_sub_u32 v13, v1, v12 // (op 537) 

v_lshlrev_b32 v12, 4, v13 // (op 537) 

v_mov_b32 v15, 0 // (op 537) convert
v_mov_b32 v14, v12 // (op 537) convert

v_lshrrev_b32 v12, 3, v1 // (op 537) 
v_lshl_add_u32 v13, s17, 4, v12 // (op 537) 

v_mul_lo_u32 v17, v13, s9 // (op 537) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 537) most significant: high of low * low
v_mul_lo_u32 v16, v13, s8 // (op 537) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 537) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 537) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 537) most significant half



s_mul_i32 s1, s8, s13 // (op 539) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 539) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 539) least significant
s_mul_hi_u32 s15, s8, s12 // (op 539) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 539) most significant: sum
s_add_u32 s1, s1, s15 // (op 539) most significant: sum


s_mov_b32 s44, s10 // (op 539) reg expression
s_mov_b32 s45, s11 // (op 539) reg expression
s_mov_b32 s46, s0 // (op 539) reg expression
s_mov_b32 s47, 131072 // (op 539) call()

v_and_b32 v14, -8, v1 // (op 556) 
v_sub_u32 v15, v1, v14 // (op 556) 

v_lshlrev_b32 v14, 4, v15 // (op 556) 

v_mov_b32 v17, 0 // (op 556) convert
v_mov_b32 v16, v14 // (op 556) convert

v_lshrrev_b32 v14, 3, v1 // (op 556) 
v_lshl_add_u32 v15, s17, 4, v14 // (op 556) 

v_mul_lo_u32 v19, v15, s9 // (op 556) most significant: low of low * high
v_mul_hi_u32 v14, v15, s8 // (op 556) most significant: high of low * low
v_mul_lo_u32 v18, v15, s8 // (op 556) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 556) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 556) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 556) most significant half



s_mul_i32 s1, s8, s13 // (op 558) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 558) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 558) least significant
s_mul_hi_u32 s15, s8, s12 // (op 558) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 558) most significant: sum
s_add_u32 s1, s1, s15 // (op 558) most significant: sum


s_mov_b32 s44, s10 // (op 558) reg expression
s_mov_b32 s45, s11 // (op 558) reg expression
s_mov_b32 s46, s0 // (op 558) reg expression
s_mov_b32 s47, 131072 // (op 558) call()

v_and_b32 v16, -32, v1 // (op 601) 
v_sub_u32 v17, v1, v16 // (op 601) 

v_lshl_add_u32 v16, s16, 5, v17 // (op 601) 

v_lshrrev_b32 v17, 5, v1 // (op 601) 

s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v18, v17, s20 // (op 601) 

v_add_u32 v17, v16, v18 // (op 601) 


s_mul_i32 s1, s20, s25 // (op 603) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 603) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 603) least significant
s_mul_hi_u32 s15, s20, s24 // (op 603) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 603) most significant: sum
s_add_u32 s1, s1, s15 // (op 603) most significant: sum


s_mov_b32 s48, s22 // (op 603) reg expression
s_mov_b32 s49, s23 // (op 603) reg expression
s_mov_b32 s50, s0 // (op 603) reg expression
s_mov_b32 s51, 131072 // (op 603) call()

v_and_b32 v16, -32, v1 // (op 620) 
v_sub_u32 v18, v1, v16 // (op 620) 

v_lshl_add_u32 v16, s16, 5, v18 // (op 620) 

v_lshrrev_b32 v18, 5, v1 // (op 620) 
v_mul_lo_u32 v19, v18, s20 // (op 620) 

v_add_u32 v18, v16, v19 // (op 620) 


s_mul_i32 s1, s20, s25 // (op 622) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 622) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 622) least significant
s_mul_hi_u32 s15, s20, s24 // (op 622) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 622) most significant: sum
s_add_u32 s1, s1, s15 // (op 622) most significant: sum


s_mov_b32 s48, s22 // (op 622) reg expression
s_mov_b32 s49, s23 // (op 622) reg expression
s_mov_b32 s50, s0 // (op 622) reg expression
s_mov_b32 s51, 131072 // (op 622) call()

v_and_b32 v16, 15, v1 // (op 665) 
v_lshl_add_u32 v19, s17, 4, v16 // (op 665) 

v_mul_lo_u32 v16, v19, s26 // (op 665) 

v_and_b32 v19, 63, v1 // (op 665) 
v_lshrrev_b32 v20, 4, v19 // (op 665) 

v_add_u32 v19, v20, v16 // (op 665) 


s_mul_i32 s1, s26, s13 // (op 667) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 667) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 667) least significant
s_mul_hi_u32 s15, s26, s12 // (op 667) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 667) most significant: sum
s_add_u32 s1, s1, s15 // (op 667) most significant: sum


s_mov_b32 s52, s28 // (op 667) reg expression
s_mov_b32 s53, s29 // (op 667) reg expression
s_mov_b32 s54, s0 // (op 667) reg expression
s_mov_b32 s55, 131072 // (op 667) call()

v_and_b32 v16, 15, v1 // (op 684) 
v_lshl_add_u32 v20, s17, 4, v16 // (op 684) 

v_mul_lo_u32 v16, v20, s26 // (op 684) 

v_and_b32 v20, 63, v1 // (op 684) 
v_lshrrev_b32 v21, 4, v20 // (op 684) 

v_add_u32 v20, v21, v16 // (op 684) 


s_mul_i32 s1, s26, s13 // (op 686) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 686) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 686) least significant
s_mul_hi_u32 s15, s26, s12 // (op 686) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 686) most significant: sum
s_add_u32 s1, s1, s15 // (op 686) most significant: sum


s_mov_b32 s52, s28 // (op 686) reg expression
s_mov_b32 s53, s29 // (op 686) reg expression
s_mov_b32 s54, s0 // (op 686) reg expression
s_mov_b32 s55, 131072 // (op 686) call()




buffer_load_ubyte v16, v19, s[52:55], 0 offen offset: 0 // (op 337) Load value
v_and_b32 v21, -8, v1 // (op 803) 
v_sub_u32 v22, v1, v21 // (op 803) 

v_lshlrev_b32 v21, 4, v22 // (op 803) 

v_lshrrev_b32 v22, 3, v1 // (op 803) 
v_lshl_add_u32 v23, v22, 7, v21 // (op 803) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 803) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 429) reg expression
buffer_load_dwordx4 v12, s[44:47], 0 offen offset: 0 lds // (op 429) Load value direct to lds



v_mov_b32 v21, v10 // (op 344) reg expression

buffer_load_dwordx3 v[22:24], v21, s[40:43], 0 offen // (op 344) Load value
buffer_load_dwordx3 v[26:28], v21, s[40:43], 0 offen offset:12 // (op 344) Load value



v_mov_b32 v21, v17 // (op 347) reg expression

buffer_load_ubyte v25, v21, s[48:51], 0 offen offset: 0 // (op 347) Load value




buffer_load_ubyte v21, v19, s[52:55], 0 offen offset:4 // (op 351) Load value

v_and_b32 v19, -8, v1 // (op 953) 
v_sub_u32 v29, v1, v19 // (op 953) 

v_lshlrev_b32 v19, 4, v29 // (op 953) 

v_lshrrev_b32 v29, 3, v1 // (op 953) 
v_lshl_add_u32 v30, v29, 7, v19 // (op 953) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 953) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 2048, s0 // (op 434) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 434) reg expression
s_mov_b32 s14, 128 // (op 434) call()
buffer_load_dwordx4 v12, s[44:47], s14 offen offset: 0 lds // (op 434) Load value direct to lds






s_lshl_b64 s[0:1], s[2:3], 7 // (op 358) 
s_mul_i32 s15, s1, 6 // (op 358) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 358) least significant
s_mul_hi_u32 s18, s0, 6 // (op 358) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 358) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 358) 

v_add_u32 v19, v10, s0 // (op 358) 


buffer_load_dwordx3 v[30:32], v19, s[40:43], 0 offen // (op 358) Load value
buffer_load_dwordx3 v[34:36], v19, s[40:43], 0 offen offset:12 // (op 358) Load value




s_lshl_b32 s0, s20, 2 // (op 361) 
v_add_u32 v10, v17, s0 // (op 361) 


buffer_load_ubyte v19, v10, s[48:51], 0 offen offset: 0 // (op 361) Load value


v_lshlrev_b32 v10, 5, v1 // (op 751) 
v_mul_lo_u32 v17, v10, 6 // (op 751) 
v_lshrrev_b32 v29, 3, v17 // (op 751) 

v_lshrrev_b32 v17, 4, v10 // (op 751) 

v_lshlrev_b32 v10, 2, v17 // (op 751) 

v_add_u32 v17, v29, v10 // (op 751) 





s_waitcnt vmcnt(7)
ds_write_b96 v17, v[22:24] offset:4096 // (op 364) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v17, v[26:28] offset:4112 // (op 364) Store local data 

v_and_b32 v10, -32, v1 // (op 856) 
v_sub_u32 v17, v1, v10 // (op 856) 

v_and_b32 v10, 4294967264, v1 // (op 856) 
v_add_u32 v29, v10, v17 // (op 856) 





s_waitcnt vmcnt(5)
ds_write_b8 v29, v25 offset:8192 // (op 365) Store local data 

v_lshlrev_b32 v10, 5, v1 // (op 722) 
v_mul_lo_u32 v17, v10, 6 // (op 722) 
v_lshrrev_b32 v29, 3, v17 // (op 722) 

v_lshrrev_b32 v17, 4, v10 // (op 722) 

v_lshlrev_b32 v10, 2, v17 // (op 722) 

v_add_u32 v17, v29, v10 // (op 722) 


v_and_b32 v10, -8, v1 // (op 775) 
v_sub_u32 v29, v1, v10 // (op 775) 

v_lshlrev_b32 v10, 4, v29 // (op 775) 

v_lshrrev_b32 v29, 3, v1 // (op 775) 
v_lshl_add_u32 v33, v29, 7, v10 // (op 775) 


s_nop 0
v_readfirstlane_b32 s0, v33 // (op 775) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -32, v1 // (op 827) 
v_sub_u32 v29, v1, v10 // (op 827) 

v_and_b32 v10, 4294967264, v1 // (op 827) 
v_add_u32 v33, v10, v29 // (op 827) 


v_lshlrev_b32 v10, 5, v1 // (op 914) 
v_mul_lo_u32 v29, v10, 6 // (op 914) 
v_lshrrev_b32 v12, 3, v29 // (op 914) 

v_lshrrev_b32 v13, 4, v10 // (op 914) 

v_lshlrev_b32 v10, 2, v13 // (op 914) 

v_add_u32 v13, v12, v10 // (op 914) 


v_and_b32 v10, -8, v1 // (op 938) 
v_sub_u32 v12, v1, v10 // (op 938) 

v_lshlrev_b32 v10, 4, v12 // (op 938) 

v_lshrrev_b32 v12, 3, v1 // (op 938) 
v_lshl_add_u32 v29, v12, 7, v10 // (op 938) 


s_nop 0
v_readfirstlane_b32 s1, v29 // (op 938) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -32, v1 // (op 977) 
v_sub_u32 v12, v1, v10 // (op 977) 

v_and_b32 v10, 4294967264, v1 // (op 977) 
v_add_u32 v29, v10, v12 // (op 977) 


s_mov_b32 s14, 0 // (op 55) call()
s_ashr_i64 s[18:19], s[6:7], 7 // (op 54) 
s_mov_b32 s15, s18 // (op 54) convert

s_and_b32 s18, s15, 4294967294 // (op 54) 

s_cmp_lt_u32 s14, s18 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 if false)
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54:
s_add_u32 s14, s14, 2 // (op 56) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1022) 



ds_read_b96_tr_b6 v[38:40], v0 offset:4096 // (op 65) Transpose load local data 
ds_read_b96_tr_b6 v[41:43], v0 offset:4608 // (op 65) Transpose load local data 
s_barrier  // (op 1035) 



ds_read_b128 v[44:47], v5 // (op 70) Load local data 
ds_read_b128 v[48:51], v5 offset:64 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1048) 


ds_read_u8 v10, v3 offset:8192 // (op 75) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:43], v[44:51], a[0:3], v10, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 82) 





v_mov_b32 v10, v20 // (op 80) 

buffer_load_ubyte v16, v10, s[52:55], 0 offen offset:8 // (op 80) Load value
s_barrier  // (op 1038) 


v_mov_b64 v[38:39], v[14:15] // (op 412) 

s_mov_b32 m0, s0 // (op 412) reg expression
s_mov_b32 s15, 256 // (op 412) call()
buffer_load_dwordx4 v38, s[44:47], s15 offen offset: 0 lds // (op 412) Load value direct to lds



v_mov_b32 v12, v11 // (op 4) 
s_lshl_b64 s[18:19], s[2:3], 7 // (op 4) 
s_mul_i32 s39, s19, 6 // (op 4) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 4) least significant
s_mul_hi_u32 s15, s18, 6 // (op 4) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 4) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 4) 

s_lshl_b32 s15, s18, 1 // (op 4) 

v_add_u32 v37, v12, s15 // (op 4) 


buffer_load_dwordx3 v[22:24], v37, s[40:43], 0 offen // (op 4) Load value
buffer_load_dwordx3 v[26:28], v37, s[40:43], 0 offen offset:12 // (op 4) Load value



v_mov_b32 v37, v18 // (op 17) 
s_lshl_b32 s15, s20, 3 // (op 17) 
v_add_u32 v40, v37, s15 // (op 17) 


buffer_load_ubyte v25, v40, s[48:51], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1065) 



s_waitcnt vmcnt(7)
ds_write_b96 v13, v[30:32] offset:8320 // (op 225) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v13, v[34:36] offset:8336 // (op 225) Store local data 
s_barrier  // (op 1077) 



s_waitcnt vmcnt(5)
ds_write_b8 v29, v19 offset:12416 // (op 236) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1062) 



ds_read_b96_tr_b6 v[40:42], v4 offset:8320 // (op 227) Transpose load local data 
ds_read_b96_tr_b6 v[43:45], v4 offset:8832 // (op 227) Transpose load local data 
s_barrier  // (op 1068) 
s_barrier  // (op 1124) 



ds_read_b128 v[46:49], v8 offset:2048 // (op 229) Load local data 
ds_read_b128 v[50:53], v8 offset:2112 // (op 229) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1074) 


ds_read_u8 v54, v6 offset:12416 // (op 231) Load local data 
s_barrier  // (op 1132) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:45], v[46:53], a[0:3], v54, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 234) 






buffer_load_ubyte v21, v10, s[52:55], 0 offen offset:12 // (op 233) Load value

s_barrier  // (op 1071) 
v_add_u32 v20, 8, v20 // (op 676) 


s_add_u32 s15, 2048, s1 // (op 424) 

s_mov_b32 m0, s15 // (op 424) reg expression
s_mov_b32 s18, 384 // (op 424) call()
buffer_load_dwordx4 v38, s[44:47], s18 offen offset: 0 lds // (op 424) Load value direct to lds





v_add_co_u32 v14, vcc, 256, v14 // (op 548) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 548) most significant half
s_lshl_b64 s[18:19], s[2:3], 7 // (op 224) 
s_mul_i32 s39, s19, 6 // (op 224) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 224) least significant
s_mul_hi_u32 s15, s18, 6 // (op 224) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 224) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 224) 

s_mul_i32 s15, 3, s18 // (op 224) 

v_add_u32 v10, v12, s15 // (op 224) 


buffer_load_dwordx3 v[30:32], v10, s[40:43], 0 offen // (op 224) Load value
buffer_load_dwordx3 v[34:36], v10, s[40:43], 0 offen offset:12 // (op 224) Load value




s_lshl_b32 s15, s20, 2 // (op 238) 
s_mul_i32 s18, 3, s15 // (op 238) 

v_add_u32 v10, v37, s18 // (op 238) 


s_lshl_b64 s[18:19], s[2:3], 7 // (op 484) 
s_mul_i32 s39, s19, 6 // (op 484) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 484) least significant
s_mul_hi_u32 s15, s18, 6 // (op 484) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 484) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 484) 

s_lshl_b32 s15, s18, 1 // (op 484) 

v_add_u32 v11, v11, s15 // (op 484) 

buffer_load_ubyte v19, v10, s[48:51], 0 offen offset: 0 // (op 238) Load value


s_barrier  // (op 1025) 
s_lshl_b32 s15, s20, 3 // (op 612) 



v_add_u32 v18, v18, s15 // (op 612) 

s_waitcnt vmcnt(7)
ds_write_b96 v17, v[22:24] offset:4096 // (op 6) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v17, v[26:28] offset:4112 // (op 6) Store local data 
s_barrier  // (op 1051) 



s_waitcnt vmcnt(5)
ds_write_b8 v33, v25 offset:8192 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[6:7], 7 // (op 54) 
s_mov_b32 s15, s18 // (op 54) convert

s_and_b32 s18, s15, 4294967294 // (op 54) 

s_cmp_lt_u32 s14, s18 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 if true)
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54:














s_barrier  // (op 1088) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 447) 
s_lshr_b32 s14, s0, 1 // (op 447) 

v_lshl_add_u32 v20, s14, 8, v1 // (op 447) 

s_lshl_b32 s0, s16, 5 // (op 447) 
v_mul_lo_u32 v11, v20, s3 // (op 447) most significant: low of low * high
v_mul_hi_u32 v33, v20, s2 // (op 447) most significant: high of low * low
v_mul_lo_u32 v10, v20, s2 // (op 447) least significant: low of low * low
v_add_u32 v11, v11, v33 // (op 447) most significant: sum


s_mov_b32 s15, 0 // (op 447) convert
s_mov_b32 s14, s0 // (op 447) convert

v_mov_b32 v20, s14 // (op 447) 
v_mov_b32 v33, s15 // (op 447) 
v_add_co_u32 v12, s[0:1], v20, v10 // (op 447) least significant half
v_addc_co_u32 v13, s[0:1], v33, v11, s[0:1] // (op 447) most significant half





v_mul_lo_u32 v11, v13, 6 // (op 447) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 447) most significant: high of low * low
v_mul_lo_u32 v10, v12, 6 // (op 447) least significant: low of low * low
v_add_u32 v11, v11, v20 // (op 447) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 447) 

v_mov_b32 v20, v12 // (op 447) convert

s_mul_i32 s1, s7, 6 // (op 449) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 449) least significant
s_mul_hi_u32 s14, s6, 6 // (op 449) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 449) most significant: sum

s_mul_i32 s15, s2, s1 // (op 449) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 449) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 449) least significant
s_mul_hi_u32 s19, s2, s0 // (op 449) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 449) most significant: sum
s_add_u32 s15, s15, s19 // (op 449) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 449) 

s_mov_b32 s40, s4 // (op 449) reg expression
s_mov_b32 s41, s5 // (op 449) reg expression
s_mov_b32 s42, s0 // (op 449) reg expression
s_mov_b32 s43, 131072 // (op 449) call()

v_and_b32 v33, -8, v1 // (op 514) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 514) 
v_sub_u32 v10, v1, v33 // (op 514) 

s_lshr_b32 s14, s0, 1 // (op 514) 

v_lshlrev_b32 v33, 4, v10 // (op 514) 

v_lshl_add_u32 v10, s14, 8, v33 // (op 514) 


v_mov_b32 v23, 0 // (op 514) convert
v_mov_b32 v22, v10 // (op 514) convert

v_lshrrev_b32 v33, 3, v1 // (op 514) 
v_lshl_add_u32 v10, s17, 4, v33 // (op 514) 

v_mul_lo_u32 v25, v10, s9 // (op 514) most significant: low of low * high
v_mul_hi_u32 v33, v10, s8 // (op 514) most significant: high of low * low
v_mul_lo_u32 v24, v10, s8 // (op 514) least significant: low of low * low
v_add_u32 v25, v25, v33 // (op 514) most significant: sum


v_add_co_u32 v10, s[0:1], v22, v24 // (op 514) least significant half
v_addc_co_u32 v11, s[0:1], v23, v25, s[0:1] // (op 514) most significant half



s_mul_i32 s1, s8, s13 // (op 516) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 516) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 516) least significant
s_mul_hi_u32 s15, s8, s12 // (op 516) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 516) most significant: sum
s_add_u32 s1, s1, s15 // (op 516) most significant: sum


s_mov_b32 s44, s10 // (op 516) reg expression
s_mov_b32 s45, s11 // (op 516) reg expression
s_mov_b32 s46, s0 // (op 516) reg expression
s_mov_b32 s47, 131072 // (op 516) call()

v_and_b32 v33, -32, v1 // (op 578) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 578) 
v_sub_u32 v12, v1, v33 // (op 578) 

s_lshr_b32 s14, s0, 1 // (op 578) 

v_lshl_add_u32 v33, s16, 5, v12 // (op 578) 

v_lshrrev_b32 v12, 5, v1 // (op 578) 
v_lshl_add_u32 v13, s14, 3, v12 // (op 578) 


v_mul_lo_u32 v12, v13, s20 // (op 578) 

v_add_u32 v13, v33, v12 // (op 578) 


s_mul_i32 s1, s20, s25 // (op 580) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 580) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 580) least significant
s_mul_hi_u32 s15, s20, s24 // (op 580) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 580) most significant: sum
s_add_u32 s1, s1, s15 // (op 580) most significant: sum


s_mov_b32 s48, s22 // (op 580) reg expression
s_mov_b32 s49, s23 // (op 580) reg expression
s_mov_b32 s50, s0 // (op 580) reg expression
s_mov_b32 s51, 131072 // (op 580) call()

v_and_b32 v12, 63, v1 // (op 642) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 642) 
v_lshrrev_b32 v33, 4, v12 // (op 642) 

s_lshr_b32 s14, s0, 1 // (op 642) 

v_and_b32 v12, 15, v1 // (op 642) 
v_lshl_add_u32 v14, s14, 3, v33 // (op 642) 


v_lshl_add_u32 v33, s17, 4, v12 // (op 642) 

v_mul_lo_u32 v12, v33, s26 // (op 642) 

v_add_u32 v33, v14, v12 // (op 642) 


s_mul_i32 s1, s26, s13 // (op 644) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 644) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 644) least significant
s_mul_hi_u32 s15, s26, s12 // (op 644) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 644) most significant: sum
s_add_u32 s1, s1, s15 // (op 644) most significant: sum


s_mov_b32 s52, s28 // (op 644) reg expression
s_mov_b32 s53, s29 // (op 644) reg expression
s_mov_b32 s54, s0 // (op 644) reg expression
s_mov_b32 s55, 131072 // (op 644) call()

v_lshlrev_b32 v12, 5, v1 // (op 735) 
v_mul_lo_u32 v14, v12, 6 // (op 735) 
v_lshrrev_b32 v15, 3, v14 // (op 735) 

v_lshrrev_b32 v14, 4, v12 // (op 735) 

v_lshlrev_b32 v12, 2, v14 // (op 735) 

v_add_u32 v14, v15, v12 // (op 735) 


v_and_b32 v12, -8, v1 // (op 788) 
v_sub_u32 v15, v1, v12 // (op 788) 

v_lshlrev_b32 v12, 4, v15 // (op 788) 

v_lshrrev_b32 v15, 3, v1 // (op 788) 
v_lshl_add_u32 v16, v15, 7, v12 // (op 788) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 788) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -32, v1 // (op 840) 
v_sub_u32 v15, v1, v12 // (op 840) 

v_and_b32 v12, 4294967264, v1 // (op 840) 
v_add_u32 v16, v12, v15 // (op 840) 


s_ashr_i64 s[14:15], s[6:7], 7 // (op 162) 
s_and_b32 s1, s14, 4294967294 // (op 162) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 161) 
s_mov_b32 s18, s14 // (op 161) convert

s_cmp_lt_u32 s1, s18 // (op 161) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161:


s_add_u32 s1, s1, 1 // (op 163) 
v_mov_b32 v12, v20 // (op 166) 
v_mov_b32 v15, v12 // (op 166) reg expression

buffer_load_dwordx3 v[22:24], v15, s[40:43], 0 offen // (op 166) Load value
buffer_load_dwordx3 v[26:28], v15, s[40:43], 0 offen offset:12 // (op 166) Load value


s_lshl_b64 s[14:15], s[2:3], 7 // (op 437) 
s_mul_i32 s19, s15, 6 // (op 437) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 437) least significant
s_mul_hi_u32 s38, s14, 6 // (op 437) most significant: high of low * low
s_add_u32 s19, s19, s38 // (op 437) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 437) 

v_add_u32 v20, v20, s14 // (op 437) 

s_barrier  // (op 1028) 
s_barrier  // (op 1041) 





v_mov_b64 v[18:19], v[10:11] // (op 417) 

s_mov_b32 m0, s0 // (op 417) reg expression
buffer_load_dwordx4 v18, s[44:47], 0 offen offset: 0 lds // (op 417) Load value direct to lds

v_add_co_u32 v10, vcc, 128, v10 // (op 506) least significant half
v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 506) most significant half
s_waitcnt vmcnt(2)
ds_write_b96 v14, v[22:24] offset:4096 // (op 167) Store local data 


v_mov_b32 v12, v13 // (op 180) 
v_mov_b32 v15, v12 // (op 180) reg expression

buffer_load_ubyte v21, v15, s[48:51], 0 offen offset: 0 // (op 180) Load value


s_lshl_b32 s14, s20, 2 // (op 570) 
v_add_u32 v13, v13, s14 // (op 570) 

s_waitcnt vmcnt(2)
ds_write_b96 v14, v[26:28] offset:4112 // (op 167) Store local data 

s_barrier  // (op 1054) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1031) 






ds_read_b96_tr_b6 v[22:24], v0 offset:4096 // (op 169) Transpose load local data 
ds_read_b96_tr_b6 v[25:27], v0 offset:4608 // (op 169) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v16, v21 offset:8192 // (op 178) Store local data 

s_barrier  // (op 1044) 



ds_read_b128 v[34:37], v5 // (op 171) Load local data 
ds_read_b128 v[38:41], v5 offset:64 // (op 171) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1058) 


ds_read_u8 v12, v3 offset:8192 // (op 173) Load local data 


v_mov_b32 v15, v33 // (op 175) 

buffer_load_ubyte v21, v15, s[52:55], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v33, 4, v33 // (op 634) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:27], v[34:41], a[0:3], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 176) 




s_ashr_i64 s[14:15], s[6:7], 7 // (op 161) 
s_mov_b32 s18, s14 // (op 161) convert

s_cmp_lt_u32 s1, s18 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 if true)
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161:






















s_barrier  // (op 1118) 






v_cmp_eq_f32 vcc, 0.00000, v7 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v10, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 

v_lshrrev_b32 v0, 6, v1 // (op 882) 
v_and_b32 v8, 63, v1 // (op 882) 
v_lshrrev_b32 v3, 4, v8 // (op 882) 
v_lshlrev_b32 v4, 2, v3 // (op 882) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 882) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 882) 

v_lshrrev_b32 v3, 2, v8 // (op 882) 

v_and_b32 v8, 3, v3 // (op 882) 

v_and_b32 v3, 3, v1 // (op 882) 
v_lshl_add_u32 v4, v8, 2, v3 // (op 882) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 882) 

v_mul_lo_u32 v8, v3, s30 // (op 882) 

v_add_lshl_u32 v3, v0, v8, 2 // (op 882) 


s_mul_i32 s15, s30, s13 // (op 884) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 884) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 884) least significant
s_mul_hi_u32 s1, s30, s12 // (op 884) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 884) most significant: sum
s_add_u32 s15, s15, s1 // (op 884) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 884) 

s_mov_b32 s8, s32 // (op 884) reg expression
s_mov_b32 s9, s33 // (op 884) reg expression
s_mov_b32 s10, s0 // (op 884) reg expression
s_mov_b32 s11, 131072 // (op 884) call()




buffer_store_dwordx4 v[14:17], v3, s[8:11], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123:
v_lshrrev_b32 v0, 6, v1 // (op 705) 
v_and_b32 v8, 63, v1 // (op 705) 
v_lshrrev_b32 v3, 4, v8 // (op 705) 
v_lshlrev_b32 v4, 2, v3 // (op 705) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 705) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 705) 

v_lshrrev_b32 v3, 2, v8 // (op 705) 

v_and_b32 v8, 3, v3 // (op 705) 

v_and_b32 v3, 3, v1 // (op 705) 
v_lshl_add_u32 v4, v8, 2, v3 // (op 705) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 705) 

v_mul_lo_u32 v8, v3, s34 // (op 705) 

v_add_lshl_u32 v3, v0, v8, 2 // (op 705) 


s_mul_i32 s15, s34, s13 // (op 707) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 707) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 707) least significant
s_mul_hi_u32 s1, s34, s12 // (op 707) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 707) most significant: sum
s_add_u32 s15, s15, s1 // (op 707) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 707) 

s_mov_b32 s4, s36 // (op 707) reg expression
s_mov_b32 s5, s37 // (op 707) reg expression
s_mov_b32 s6, s0 // (op 707) reg expression
s_mov_b32 s7, 131072 // (op 707) call()





buffer_load_dwordx4 v[10:13], v3, s[4:7], 0 offen // (op 135) Load value



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


v_lshrrev_b32 v0, 6, v1 // (op 899) 
v_and_b32 v8, 63, v1 // (op 899) 
v_lshrrev_b32 v3, 4, v8 // (op 899) 
v_lshlrev_b32 v4, 2, v3 // (op 899) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 899) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 899) 

v_lshrrev_b32 v3, 2, v8 // (op 899) 

v_and_b32 v8, 3, v3 // (op 899) 

v_and_b32 v3, 3, v1 // (op 899) 
v_lshl_add_u32 v4, v8, 2, v3 // (op 899) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 899) 

v_mul_lo_u32 v8, v3, s30 // (op 899) 

v_add_lshl_u32 v3, v0, v8, 2 // (op 899) 


s_mul_i32 s15, s30, s13 // (op 901) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 901) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 901) least significant
s_mul_hi_u32 s1, s30, s12 // (op 901) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 901) most significant: sum
s_add_u32 s15, s15, s1 // (op 901) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 901) 

s_mov_b32 s8, s32 // (op 901) reg expression
s_mov_b32 s9, s33 // (op 901) reg expression
s_mov_b32 s10, s0 // (op 901) reg expression
s_mov_b32 s11, 131072 // (op 901) call()




buffer_store_dwordx4 v[10:13], v3, s[8:11], 0 offen // (op 134) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123:




s_endpgm  // End of RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.LRR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end:
.size RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2, .LRR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end-RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: B8
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
  - .name: RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
    .symbol: RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 12544
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 55
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

