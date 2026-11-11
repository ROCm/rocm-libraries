// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.p2align 8
.type RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2,@function
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 731) 
v_and_b32 v3, 48, v1 // (op 731) 
v_sub_u32 v4, v0, v3 // (op 731) 

v_lshrrev_b32 v3, 4, v0 // (op 731) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 731) 


v_lshrrev_b32 v3, 7, v1 // (op 731) 
v_lshlrev_b32 v4, 4, v3 // (op 731) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 731) 


v_mul_lo_u32 v0, v3, 6 // (op 731) 
v_lshrrev_b32 v4, 3, v0 // (op 731) 

v_lshrrev_b32 v0, 4, v3 // (op 731) 

v_lshlrev_b32 v3, 2, v0 // (op 731) 

v_add_u32 v0, v4, v3 // (op 731) 


v_lshrrev_b32 v3, 6, v1 // (op 783) 
v_and_b32 v4, 1, v3 // (op 783) 

v_and_b32 v3, 15, v1 // (op 783) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 783) 


v_and_b32 v3, 48, v1 // (op 783) 
v_lshl_add_u32 v4, v5, 7, v3 // (op 783) 


v_and_b32 v3, 63, v1 // (op 841) 
v_and_b32 v5, 48, v1 // (op 841) 
v_sub_u32 v6, v3, v5 // (op 841) 

v_lshrrev_b32 v5, 4, v3 // (op 841) 

v_lshl_add_u32 v3, v5, 5, v6 // (op 841) 


v_lshrrev_b32 v5, 7, v1 // (op 841) 
v_lshlrev_b32 v6, 4, v5 // (op 841) 

v_lshl_add_u32 v5, v3, 5, v6 // (op 841) 


v_mul_lo_u32 v3, v5, 6 // (op 841) 
v_lshrrev_b32 v6, 3, v3 // (op 841) 

v_lshrrev_b32 v3, 4, v5 // (op 841) 

v_lshlrev_b32 v5, 2, v3 // (op 841) 

v_add_u32 v3, v6, v5 // (op 841) 


v_lshrrev_b32 v5, 6, v1 // (op 880) 
v_and_b32 v6, 1, v5 // (op 880) 

v_and_b32 v5, 15, v1 // (op 880) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 880) 


v_and_b32 v5, 48, v1 // (op 880) 
v_lshl_add_u32 v6, v7, 7, v5 // (op 880) 


v_mov_b32 v5, s14 // (op 2) Move value
v_mov_b32 v7, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
v_and_b32 v8, -2, v1 // (op 441) 
v_sub_u32 v9, v1, v8 // (op 441) 

v_lshlrev_b32 v8, 4, v9 // (op 441) 

v_lshl_add_u32 v9, s16, 5, v8 // (op 441) 

v_mov_b32 v11, 0 // (op 441) convert
v_mov_b32 v10, v9 // (op 441) convert

v_lshrrev_b32 v8, 1, v1 // (op 441) 
v_mul_lo_u32 v13, v8, s3 // (op 441) most significant: low of low * high
v_mul_hi_u32 v9, v8, s2 // (op 441) most significant: high of low * low
v_mul_lo_u32 v12, v8, s2 // (op 441) least significant: low of low * low
v_add_u32 v13, v13, v9 // (op 441) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v12 // (op 441) least significant half
v_addc_co_u32 v9, s[0:1], v11, v13, s[0:1] // (op 441) most significant half



v_mul_lo_u32 v11, v9, 6 // (op 441) most significant: low of high * low
v_mul_hi_u32 v12, v8, 6 // (op 441) most significant: high of low * low
v_mul_lo_u32 v10, v8, 6 // (op 441) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 441) most significant: sum


v_lshrrev_b64 v[8:9], 3, v[10:11] // (op 441) 

v_mov_b32 v10, v8 // (op 441) convert

s_mul_i32 s1, s7, 6 // (op 443) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 443) least significant
s_mul_hi_u32 s14, s6, 6 // (op 443) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 443) most significant: sum

s_mul_i32 s15, s2, s1 // (op 443) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 443) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 443) least significant
s_mul_hi_u32 s19, s2, s0 // (op 443) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 443) most significant: sum
s_add_u32 s15, s15, s19 // (op 443) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 443) 

s_mov_b32 s40, s4 // (op 443) reg expression
s_mov_b32 s41, s5 // (op 443) reg expression
s_mov_b32 s42, s0 // (op 443) reg expression
s_mov_b32 s43, 131072 // (op 443) call()

v_and_b32 v8, -2, v1 // (op 460) 
v_sub_u32 v9, v1, v8 // (op 460) 

v_lshlrev_b32 v8, 4, v9 // (op 460) 

v_lshl_add_u32 v9, s16, 5, v8 // (op 460) 

v_mov_b32 v13, 0 // (op 460) convert
v_mov_b32 v12, v9 // (op 460) convert

v_lshrrev_b32 v11, 1, v1 // (op 460) 
v_mul_lo_u32 v9, v11, s3 // (op 460) most significant: low of low * high
v_mul_hi_u32 v14, v11, s2 // (op 460) most significant: high of low * low
v_mul_lo_u32 v8, v11, s2 // (op 460) least significant: low of low * low
v_add_u32 v9, v9, v14 // (op 460) most significant: sum


v_add_co_u32 v14, s[0:1], v12, v8 // (op 460) least significant half
v_addc_co_u32 v15, s[0:1], v13, v9, s[0:1] // (op 460) most significant half



v_mul_lo_u32 v9, v15, 6 // (op 460) most significant: low of high * low
v_mul_hi_u32 v11, v14, 6 // (op 460) most significant: high of low * low
v_mul_lo_u32 v8, v14, 6 // (op 460) least significant: low of low * low
v_add_u32 v9, v9, v11 // (op 460) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[8:9] // (op 460) 

v_mov_b32 v11, v12 // (op 460) convert

s_mul_i32 s1, s7, 6 // (op 462) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 462) least significant
s_mul_hi_u32 s14, s6, 6 // (op 462) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 462) most significant: sum

s_mul_i32 s15, s2, s1 // (op 462) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 462) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 462) least significant
s_mul_hi_u32 s19, s2, s0 // (op 462) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 462) most significant: sum
s_add_u32 s15, s15, s19 // (op 462) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 462) 

s_mov_b32 s40, s4 // (op 462) reg expression
s_mov_b32 s41, s5 // (op 462) reg expression
s_mov_b32 s42, s0 // (op 462) reg expression
s_mov_b32 s43, 131072 // (op 462) call()

v_and_b32 v8, -8, v1 // (op 505) 
v_sub_u32 v9, v1, v8 // (op 505) 

v_lshlrev_b32 v8, 4, v9 // (op 505) 

v_mov_b32 v13, 0 // (op 505) convert
v_mov_b32 v12, v8 // (op 505) convert

v_lshrrev_b32 v8, 3, v1 // (op 505) 
v_lshl_add_u32 v9, s17, 5, v8 // (op 505) 

v_mul_lo_u32 v15, v9, s9 // (op 505) most significant: low of low * high
v_mul_hi_u32 v8, v9, s8 // (op 505) most significant: high of low * low
v_mul_lo_u32 v14, v9, s8 // (op 505) least significant: low of low * low
v_add_u32 v15, v15, v8 // (op 505) most significant: sum


v_add_co_u32 v8, s[0:1], v12, v14 // (op 505) least significant half
v_addc_co_u32 v9, s[0:1], v13, v15, s[0:1] // (op 505) most significant half



s_mul_i32 s1, s8, s13 // (op 507) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 507) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 507) least significant
s_mul_hi_u32 s15, s8, s12 // (op 507) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 507) most significant: sum
s_add_u32 s1, s1, s15 // (op 507) most significant: sum


s_mov_b32 s44, s10 // (op 507) reg expression
s_mov_b32 s45, s11 // (op 507) reg expression
s_mov_b32 s46, s0 // (op 507) reg expression
s_mov_b32 s47, 131072 // (op 507) call()

v_and_b32 v12, -8, v1 // (op 524) 
v_sub_u32 v13, v1, v12 // (op 524) 

v_lshlrev_b32 v12, 4, v13 // (op 524) 

v_mov_b32 v15, 0 // (op 524) convert
v_mov_b32 v14, v12 // (op 524) convert

v_lshrrev_b32 v12, 3, v1 // (op 524) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 524) 

v_mul_lo_u32 v17, v13, s9 // (op 524) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 524) most significant: high of low * low
v_mul_lo_u32 v16, v13, s8 // (op 524) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 524) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 524) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 524) most significant half



s_mul_i32 s1, s8, s13 // (op 526) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 526) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 526) least significant
s_mul_hi_u32 s15, s8, s12 // (op 526) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 526) most significant: sum
s_add_u32 s1, s1, s15 // (op 526) most significant: sum


s_mov_b32 s44, s10 // (op 526) reg expression
s_mov_b32 s45, s11 // (op 526) reg expression
s_mov_b32 s46, s0 // (op 526) reg expression
s_mov_b32 s47, 131072 // (op 526) call()

v_lshrrev_b32 v14, 7, v1 // (op 569) 
v_and_b32 v15, 15, v1 // (op 569) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 569) 


v_lshl_add_u32 v14, s16, 5, v16 // (op 569) 

v_and_b32 v15, 63, v1 // (op 569) 
v_lshrrev_b32 v16, 4, v15 // (op 569) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v15, v16, s20 // (op 569) 

v_add_u32 v16, v14, v15 // (op 569) 


s_mul_i32 s1, s20, s25 // (op 571) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 571) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 571) least significant
s_mul_hi_u32 s15, s20, s24 // (op 571) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 571) most significant: sum
s_add_u32 s1, s1, s15 // (op 571) most significant: sum


s_mov_b32 s48, s22 // (op 571) reg expression
s_mov_b32 s49, s23 // (op 571) reg expression
s_mov_b32 s50, s0 // (op 571) reg expression
s_mov_b32 s51, 131072 // (op 571) call()

v_lshrrev_b32 v14, 7, v1 // (op 588) 
v_and_b32 v15, 15, v1 // (op 588) 
v_lshl_add_u32 v17, v14, 4, v15 // (op 588) 


v_lshl_add_u32 v14, s16, 5, v17 // (op 588) 

v_and_b32 v15, 63, v1 // (op 588) 
v_lshrrev_b32 v17, 4, v15 // (op 588) 

v_mul_lo_u32 v15, v17, s20 // (op 588) 

v_add_u32 v17, v14, v15 // (op 588) 


s_mul_i32 s1, s20, s25 // (op 590) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 590) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 590) least significant
s_mul_hi_u32 s15, s20, s24 // (op 590) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 590) most significant: sum
s_add_u32 s1, s1, s15 // (op 590) most significant: sum


s_mov_b32 s48, s22 // (op 590) reg expression
s_mov_b32 s49, s23 // (op 590) reg expression
s_mov_b32 s50, s0 // (op 590) reg expression
s_mov_b32 s51, 131072 // (op 590) call()

v_lshrrev_b32 v14, 6, v1 // (op 633) 
v_and_b32 v15, 1, v14 // (op 633) 

v_and_b32 v14, 15, v1 // (op 633) 
v_lshl_add_u32 v18, v15, 4, v14 // (op 633) 


v_lshl_add_u32 v14, s17, 5, v18 // (op 633) 

v_mul_lo_u32 v15, v14, s26 // (op 633) 

v_and_b32 v14, 63, v1 // (op 633) 
v_lshrrev_b32 v18, 4, v14 // (op 633) 

v_add_u32 v14, v18, v15 // (op 633) 


s_mul_i32 s1, s26, s13 // (op 635) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 635) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 635) least significant
s_mul_hi_u32 s15, s26, s12 // (op 635) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 635) most significant: sum
s_add_u32 s1, s1, s15 // (op 635) most significant: sum


s_mov_b32 s52, s28 // (op 635) reg expression
s_mov_b32 s53, s29 // (op 635) reg expression
s_mov_b32 s54, s0 // (op 635) reg expression
s_mov_b32 s55, 131072 // (op 635) call()

v_lshrrev_b32 v15, 6, v1 // (op 652) 
v_and_b32 v18, 1, v15 // (op 652) 

v_and_b32 v15, 15, v1 // (op 652) 
v_lshl_add_u32 v19, v18, 4, v15 // (op 652) 


v_lshl_add_u32 v15, s17, 5, v19 // (op 652) 

v_mul_lo_u32 v18, v15, s26 // (op 652) 

v_and_b32 v15, 63, v1 // (op 652) 
v_lshrrev_b32 v19, 4, v15 // (op 652) 

v_add_u32 v15, v19, v18 // (op 652) 


s_mul_i32 s1, s26, s13 // (op 654) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 654) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 654) least significant
s_mul_hi_u32 s15, s26, s12 // (op 654) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 654) most significant: sum
s_add_u32 s1, s1, s15 // (op 654) most significant: sum


s_mov_b32 s52, s28 // (op 654) reg expression
s_mov_b32 s53, s29 // (op 654) reg expression
s_mov_b32 s54, s0 // (op 654) reg expression
s_mov_b32 s55, 131072 // (op 654) call()


v_mov_b64 v[18:19], s[20:21] // (op 308) convert

v_mov_b32 v20, v16 // (op 308) reg expression
v_mov_b32 v21, v20 // (op 308) 
buffer_load_ubyte v22, v21, s[48:51], 0 offen offset: 0 // (op 308) Load value






buffer_load_ubyte v18, v14, s[52:55], 0 offen offset: 0 // (op 313) Load value
v_and_b32 v19, -8, v1 // (op 771) 
v_sub_u32 v20, v1, v19 // (op 771) 

v_lshlrev_b32 v19, 4, v20 // (op 771) 

v_lshrrev_b32 v20, 3, v1 // (op 771) 
v_lshl_add_u32 v21, v20, 7, v19 // (op 771) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 771) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 397) reg expression
buffer_load_dwordx4 v8, s[44:47], 0 offen offset: 0 lds // (op 397) Load value direct to lds



v_mov_b32 v19, v10 // (op 320) reg expression

buffer_load_dwordx3 v[24:26], v19, s[40:43], 0 offen // (op 320) Load value


v_mov_b64 v[20:21], s[20:21] // (op 324) convert

s_lshl_b32 s0, s20, 2 // (op 324) 
v_add_u32 v19, v16, s0 // (op 324) 

v_mov_b32 v23, v19 // (op 324) 
buffer_load_ubyte v27, v23, s[48:51], 0 offen offset: 0 // (op 324) Load value







buffer_load_ubyte v16, v14, s[52:55], 0 offen offset:4 // (op 329) Load value

v_and_b32 v14, -8, v1 // (op 868) 
v_sub_u32 v23, v1, v14 // (op 868) 

v_lshlrev_b32 v14, 4, v23 // (op 868) 

v_lshrrev_b32 v23, 3, v1 // (op 868) 
v_lshl_add_u32 v19, v23, 7, v14 // (op 868) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 868) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 402) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 402) reg expression
s_mov_b32 s14, 128 // (op 402) call()
buffer_load_dwordx4 v8, s[44:47], s14 offen offset: 0 lds // (op 402) Load value direct to lds






s_lshl_b64 s[0:1], s[2:3], 7 // (op 336) 
s_mul_i32 s15, s1, 6 // (op 336) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 336) least significant
s_mul_hi_u32 s18, s0, 6 // (op 336) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 336) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 336) 

v_add_u32 v14, v10, s0 // (op 336) 


buffer_load_dwordx3 v[28:30], v14, s[40:43], 0 offen // (op 336) Load value


v_and_b32 v14, -2, v1 // (op 719) 
v_sub_u32 v23, v1, v14 // (op 719) 

v_lshlrev_b32 v14, 4, v23 // (op 719) 

v_lshrrev_b32 v23, 1, v1 // (op 719) 
v_lshl_add_u32 v8, v23, 5, v14 // (op 719) 


v_mul_lo_u32 v14, v8, 6 // (op 719) 
v_lshrrev_b32 v23, 3, v14 // (op 719) 

v_lshrrev_b32 v14, 4, v8 // (op 719) 

v_lshlrev_b32 v8, 2, v14 // (op 719) 

v_add_u32 v14, v23, v8 // (op 719) 





s_waitcnt vmcnt(4)
ds_write_b96 v14, v[24:26] offset:8192 // (op 339) Store local data 

v_and_b32 v14, -2, v1 // (op 690) 
v_sub_u32 v23, v1, v14 // (op 690) 

v_lshlrev_b32 v14, 4, v23 // (op 690) 

v_lshrrev_b32 v23, 1, v1 // (op 690) 
v_lshl_add_u32 v8, v23, 5, v14 // (op 690) 


v_mul_lo_u32 v14, v8, 6 // (op 690) 
v_lshrrev_b32 v23, 3, v14 // (op 690) 

v_lshrrev_b32 v14, 4, v8 // (op 690) 

v_lshlrev_b32 v8, 2, v14 // (op 690) 

v_add_u32 v14, v23, v8 // (op 690) 


v_and_b32 v23, -8, v1 // (op 743) 
v_sub_u32 v8, v1, v23 // (op 743) 

v_lshlrev_b32 v23, 4, v8 // (op 743) 

v_lshrrev_b32 v8, 3, v1 // (op 743) 
v_lshl_add_u32 v9, v8, 7, v23 // (op 743) 


s_nop 0
v_readfirstlane_b32 s0, v9 // (op 743) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -2, v1 // (op 829) 
v_sub_u32 v8, v1, v23 // (op 829) 

v_lshlrev_b32 v23, 4, v8 // (op 829) 

v_lshrrev_b32 v8, 1, v1 // (op 829) 
v_lshl_add_u32 v9, v8, 5, v23 // (op 829) 


v_mul_lo_u32 v8, v9, 6 // (op 829) 
v_lshrrev_b32 v10, 3, v8 // (op 829) 

v_lshrrev_b32 v8, 4, v9 // (op 829) 

v_lshlrev_b32 v9, 2, v8 // (op 829) 

v_add_u32 v8, v10, v9 // (op 829) 


v_and_b32 v23, -8, v1 // (op 853) 
v_sub_u32 v9, v1, v23 // (op 853) 

v_lshlrev_b32 v10, 4, v9 // (op 853) 

v_lshrrev_b32 v9, 3, v1 // (op 853) 
v_lshl_add_u32 v23, v9, 7, v10 // (op 853) 


s_nop 0
v_readfirstlane_b32 s1, v23 // (op 853) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s14, 0 // (op 51) call()
s_ashr_i64 s[18:19], s[6:7], 7 // (op 50) 
s_mov_b32 s15, s18 // (op 50) convert

s_and_b32 s18, s15, 4294967294 // (op 50) 

s_cmp_lt_u32 s14, s18 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50 // (op 50) Condition: Top (jump to RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50 if false)
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50:
s_add_u32 s14, s14, 2 // (op 52) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 909) 



ds_read_b96_tr_b6 v[32:34], v0 offset:8192 // (op 60) Transpose load local data 
ds_read_b96_tr_b6 v[35:37], v0 offset:8704 // (op 60) Transpose load local data 
s_barrier  // (op 922) 



ds_read_b128 v[38:41], v4 // (op 65) Load local data 
ds_read_b128 v[42:45], v4 offset:64 // (op 65) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[38:45], a[0:3], v22, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 76) 



v_mov_b64 v[20:21], s[20:21] // (op 70) convert

v_mov_b32 v19, v17 // (op 70) 
s_lshl_b32 s15, s20, 3 // (op 70) 
v_add_u32 v23, v19, s15 // (op 70) 

v_mov_b32 v9, v23 // (op 70) 
buffer_load_ubyte v22, v9, s[48:51], 0 offen offset: 0 // (op 70) Load value





v_mov_b32 v23, v15 // (op 74) 

buffer_load_ubyte v18, v23, s[52:55], 0 offen offset:8 // (op 74) Load value
s_barrier  // (op 925) 


v_mov_b64 v[20:21], v[12:13] // (op 381) 

s_mov_b32 m0, s0 // (op 381) reg expression
s_mov_b32 s15, 256 // (op 381) call()
buffer_load_dwordx4 v20, s[44:47], s15 offen offset: 0 lds // (op 381) Load value direct to lds



v_mov_b32 v9, v11 // (op 4) 
s_lshl_b64 s[18:19], s[2:3], 7 // (op 4) 
s_mul_i32 s39, s19, 6 // (op 4) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 4) least significant
s_mul_hi_u32 s15, s18, 6 // (op 4) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 4) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 4) 

s_lshl_b32 s15, s18, 1 // (op 4) 

v_add_u32 v10, v9, s15 // (op 4) 


buffer_load_dwordx3 v[24:26], v10, s[40:43], 0 offen // (op 4) Load value

s_barrier  // (op 938) 



s_waitcnt vmcnt(4)
ds_write_b96 v8, v[28:30] offset:12288 // (op 208) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 935) 



ds_read_b96_tr_b6 v[32:34], v3 offset:12288 // (op 210) Transpose load local data 
ds_read_b96_tr_b6 v[35:37], v3 offset:12800 // (op 210) Transpose load local data 
s_barrier  // (op 941) 
s_barrier  // (op 987) 



ds_read_b128 v[38:41], v6 offset:4096 // (op 212) Load local data 
ds_read_b128 v[42:45], v6 offset:4160 // (op 212) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[38:45], a[0:3], v27, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 217) 



v_mov_b64 v[32:33], s[20:21] // (op 214) convert

s_lshl_b32 s15, s20, 2 // (op 214) 
s_mul_i32 s18, 3, s15 // (op 214) 

v_add_u32 v10, v19, s18 // (op 214) 

v_mov_b32 v31, v10 // (op 214) 
buffer_load_ubyte v27, v31, s[48:51], 0 offen offset: 0 // (op 214) Load value







s_lshl_b32 s15, s20, 3 // (op 580) 
v_add_u32 v17, v17, s15 // (op 580) 

buffer_load_ubyte v16, v23, s[52:55], 0 offen offset:12 // (op 216) Load value

s_barrier  // (op 944) 
v_add_u32 v15, 8, v15 // (op 644) 


s_add_u32 s15, 4096, s1 // (op 392) 

s_mov_b32 m0, s15 // (op 392) reg expression
s_mov_b32 s18, 384 // (op 392) call()
buffer_load_dwordx4 v20, s[44:47], s18 offen offset: 0 lds // (op 392) Load value direct to lds





v_add_co_u32 v12, vcc, 256, v12 // (op 516) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 516) most significant half
s_lshl_b64 s[18:19], s[2:3], 7 // (op 207) 
s_mul_i32 s39, s19, 6 // (op 207) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 207) least significant
s_mul_hi_u32 s15, s18, 6 // (op 207) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 207) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 207) 

s_mul_i32 s15, 3, s18 // (op 207) 

v_add_u32 v10, v9, s15 // (op 207) 


buffer_load_dwordx3 v[28:30], v10, s[40:43], 0 offen // (op 207) Load value


s_barrier  // (op 912) 
s_lshl_b64 s[18:19], s[2:3], 7 // (op 452) 



s_mul_i32 s39, s19, 6 // (op 452) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 452) least significant
s_mul_hi_u32 s15, s18, 6 // (op 452) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 452) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 452) 

s_lshl_b32 s15, s18, 1 // (op 452) 

v_add_u32 v11, v11, s15 // (op 452) 

s_waitcnt vmcnt(4)
ds_write_b96 v14, v[24:26] offset:8192 // (op 6) Store local data 
s_ashr_i64 s[18:19], s[6:7], 7 // (op 50) 
s_mov_b32 s15, s18 // (op 50) convert

s_and_b32 s18, s15, 4294967294 // (op 50) 

s_cmp_lt_u32 s14, s18 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50 // (op 50) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50 if true)
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50:












s_barrier  // (op 953) 
v_and_b32 v8, -2, v1 // (op 415) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 415) 
v_sub_u32 v9, v1, v8 // (op 415) 

s_lshr_b32 s14, s0, 1 // (op 415) 

v_lshlrev_b32 v8, 4, v9 // (op 415) 

v_lshl_add_u32 v9, s16, 5, v8 // (op 415) 

v_mov_b32 v11, 0 // (op 415) convert
v_mov_b32 v10, v9 // (op 415) convert

v_lshrrev_b32 v8, 1, v1 // (op 415) 
v_lshl_add_u32 v9, s14, 8, v8 // (op 415) 


v_mul_lo_u32 v13, v9, s3 // (op 415) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 415) most significant: high of low * low
v_mul_lo_u32 v12, v9, s2 // (op 415) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 415) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v12 // (op 415) least significant half
v_addc_co_u32 v9, s[0:1], v11, v13, s[0:1] // (op 415) most significant half



v_mul_lo_u32 v11, v9, 6 // (op 415) most significant: low of high * low
v_mul_hi_u32 v12, v8, 6 // (op 415) most significant: high of low * low
v_mul_lo_u32 v10, v8, 6 // (op 415) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 415) most significant: sum


v_lshrrev_b64 v[8:9], 3, v[10:11] // (op 415) 

v_mov_b32 v10, v8 // (op 415) convert

s_mul_i32 s1, s7, 6 // (op 417) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 417) least significant
s_mul_hi_u32 s14, s6, 6 // (op 417) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 417) most significant: sum

s_mul_i32 s15, s2, s1 // (op 417) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 417) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 417) least significant
s_mul_hi_u32 s19, s2, s0 // (op 417) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 417) most significant: sum
s_add_u32 s15, s15, s19 // (op 417) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 417) 

s_mov_b32 s40, s4 // (op 417) reg expression
s_mov_b32 s41, s5 // (op 417) reg expression
s_mov_b32 s42, s0 // (op 417) reg expression
s_mov_b32 s43, 131072 // (op 417) call()

v_and_b32 v8, -8, v1 // (op 482) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 482) 
v_sub_u32 v9, v1, v8 // (op 482) 

s_lshr_b32 s14, s0, 1 // (op 482) 

v_lshlrev_b32 v8, 4, v9 // (op 482) 

v_lshl_add_u32 v9, s14, 8, v8 // (op 482) 


v_mov_b32 v15, 0 // (op 482) convert
v_mov_b32 v14, v9 // (op 482) convert

v_lshrrev_b32 v8, 3, v1 // (op 482) 
v_lshl_add_u32 v9, s17, 5, v8 // (op 482) 

v_mul_lo_u32 v13, v9, s9 // (op 482) most significant: low of low * high
v_mul_hi_u32 v8, v9, s8 // (op 482) most significant: high of low * low
v_mul_lo_u32 v12, v9, s8 // (op 482) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 482) most significant: sum


v_add_co_u32 v8, s[0:1], v14, v12 // (op 482) least significant half
v_addc_co_u32 v9, s[0:1], v15, v13, s[0:1] // (op 482) most significant half



s_mul_i32 s1, s8, s13 // (op 484) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 484) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 484) least significant
s_mul_hi_u32 s15, s8, s12 // (op 484) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 484) most significant: sum
s_add_u32 s1, s1, s15 // (op 484) most significant: sum


s_mov_b32 s44, s10 // (op 484) reg expression
s_mov_b32 s45, s11 // (op 484) reg expression
s_mov_b32 s46, s0 // (op 484) reg expression
s_mov_b32 s47, 131072 // (op 484) call()

v_lshrrev_b32 v11, 7, v1 // (op 546) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 546) 
v_and_b32 v12, 15, v1 // (op 546) 
s_lshr_b32 s14, s0, 1 // (op 546) 

v_lshl_add_u32 v13, v11, 4, v12 // (op 546) 


v_lshl_add_u32 v11, s16, 5, v13 // (op 546) 

v_and_b32 v12, 63, v1 // (op 546) 
v_lshrrev_b32 v13, 4, v12 // (op 546) 

v_lshl_add_u32 v12, s14, 3, v13 // (op 546) 


v_mul_lo_u32 v13, v12, s20 // (op 546) 

v_add_u32 v12, v11, v13 // (op 546) 


s_mul_i32 s1, s20, s25 // (op 548) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 548) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 548) least significant
s_mul_hi_u32 s15, s20, s24 // (op 548) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 548) most significant: sum
s_add_u32 s1, s1, s15 // (op 548) most significant: sum


s_mov_b32 s48, s22 // (op 548) reg expression
s_mov_b32 s49, s23 // (op 548) reg expression
s_mov_b32 s50, s0 // (op 548) reg expression
s_mov_b32 s51, 131072 // (op 548) call()

v_and_b32 v11, 63, v1 // (op 610) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 610) 
v_lshrrev_b32 v13, 4, v11 // (op 610) 

s_lshr_b32 s14, s0, 1 // (op 610) 

v_lshrrev_b32 v11, 6, v1 // (op 610) 
v_lshl_add_u32 v14, s14, 3, v13 // (op 610) 


v_and_b32 v13, 1, v11 // (op 610) 

v_and_b32 v11, 15, v1 // (op 610) 
v_lshl_add_u32 v15, v13, 4, v11 // (op 610) 


v_lshl_add_u32 v11, s17, 5, v15 // (op 610) 

v_mul_lo_u32 v13, v11, s26 // (op 610) 

v_add_u32 v11, v14, v13 // (op 610) 


s_mul_i32 s1, s26, s13 // (op 612) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 612) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 612) least significant
s_mul_hi_u32 s15, s26, s12 // (op 612) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 612) most significant: sum
s_add_u32 s1, s1, s15 // (op 612) most significant: sum


s_mov_b32 s52, s28 // (op 612) reg expression
s_mov_b32 s53, s29 // (op 612) reg expression
s_mov_b32 s54, s0 // (op 612) reg expression
s_mov_b32 s55, 131072 // (op 612) call()

v_and_b32 v13, -2, v1 // (op 703) 
v_sub_u32 v14, v1, v13 // (op 703) 

v_lshlrev_b32 v13, 4, v14 // (op 703) 

v_lshrrev_b32 v14, 1, v1 // (op 703) 
v_lshl_add_u32 v15, v14, 5, v13 // (op 703) 


v_mul_lo_u32 v13, v15, 6 // (op 703) 
v_lshrrev_b32 v14, 3, v13 // (op 703) 

v_lshrrev_b32 v13, 4, v15 // (op 703) 

v_lshlrev_b32 v15, 2, v13 // (op 703) 

v_add_u32 v13, v14, v15 // (op 703) 


v_and_b32 v14, -8, v1 // (op 756) 
v_sub_u32 v15, v1, v14 // (op 756) 

v_lshlrev_b32 v14, 4, v15 // (op 756) 

v_lshrrev_b32 v15, 3, v1 // (op 756) 
v_lshl_add_u32 v17, v15, 7, v14 // (op 756) 


s_nop 0
v_readfirstlane_b32 s0, v17 // (op 756) Wait state hazard: VALU Write Readlane Hazard

s_ashr_i64 s[14:15], s[6:7], 7 // (op 153) 
s_and_b32 s1, s14, 4294967294 // (op 153) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 152) 
s_mov_b32 s18, s14 // (op 152) convert

s_cmp_lt_u32 s1, s18 // (op 152) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152 // (op 152) Condition: Top (jump to RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152:


s_add_u32 s1, s1, 1 // (op 154) 
v_mov_b32 v14, v10 // (op 157) 
v_mov_b32 v15, v14 // (op 157) reg expression

buffer_load_dwordx3 v[16:18], v15, s[40:43], 0 offen // (op 157) Load value


s_lshl_b64 s[14:15], s[2:3], 7 // (op 405) 
s_mul_i32 s19, s15, 6 // (op 405) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 405) least significant
s_mul_hi_u32 s38, s14, 6 // (op 405) most significant: high of low * low
s_add_u32 s19, s19, s38 // (op 405) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 405) 

v_add_u32 v10, v10, s14 // (op 405) 

s_barrier  // (op 915) 
s_barrier  // (op 928) 





v_mov_b64 v[14:15], v[8:9] // (op 386) 

s_mov_b32 m0, s0 // (op 386) reg expression
buffer_load_dwordx4 v14, s[44:47], 0 offen offset: 0 lds // (op 386) Load value direct to lds

v_add_co_u32 v8, vcc, 128, v8 // (op 474) least significant half
v_addc_co_u32 v9, vcc, 0, v9, vcc // (op 474) most significant half
s_waitcnt vmcnt(1)
ds_write_b96 v13, v[16:18] offset:8192 // (op 158) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 918) 



ds_read_b96_tr_b6 v[14:16], v0 offset:8192 // (op 160) Transpose load local data 
ds_read_b96_tr_b6 v[17:19], v0 offset:8704 // (op 160) Transpose load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 931) 



ds_read_b128 v[20:23], v4 // (op 162) Load local data 
ds_read_b128 v[24:27], v4 offset:64 // (op 162) Load local data 

v_mov_b64 v[28:29], s[20:21] // (op 164) convert

v_mov_b32 v30, v12 // (op 164) 
v_mov_b32 v31, v30 // (op 164) reg expression
v_mov_b32 v32, v31 // (op 164) 
buffer_load_ubyte v33, v32, s[48:51], 0 offen offset: 0 // (op 164) Load value




s_lshl_b32 s14, s20, 2 // (op 538) 
v_add_u32 v12, v12, s14 // (op 538) 



v_mov_b32 v28, v11 // (op 166) 

buffer_load_ubyte v29, v28, s[52:55], 0 offen offset: 0 // (op 166) Load value

v_add_u32 v11, 4, v11 // (op 602) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[14:19], v[20:27], a[0:3], v33, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 167) 




s_ashr_i64 s[14:15], s[6:7], 7 // (op 152) 
s_mov_b32 s18, s14 // (op 152) convert

s_cmp_lt_u32 s1, s18 // (op 152) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152 // (op 152) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152 if true)
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152:





















s_barrier  // (op 981) 




v_cmp_eq_f32 vcc, 0.00000, v5 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114
s_nop 0
v_accvgpr_read v8, a0 // (op 42) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 42) 
v_accvgpr_read v10, a2 // (op 42) 
v_accvgpr_read v11, a3 // (op 42) 
v_mul_f32 v12, v7, v8 // (op 42) 
v_mul_f32 v13, v7, v9 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 

v_lshrrev_b32 v0, 7, v1 // (op 797) 
v_and_b32 v6, 63, v1 // (op 797) 
v_lshrrev_b32 v3, 4, v6 // (op 797) 
v_lshlrev_b32 v4, 2, v3 // (op 797) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 797) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 797) 

v_lshrrev_b32 v3, 2, v6 // (op 797) 

v_and_b32 v4, 3, v3 // (op 797) 

v_and_b32 v3, 3, v1 // (op 797) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 797) 


v_lshrrev_b32 v3, 6, v1 // (op 797) 
v_and_b32 v4, 1, v3 // (op 797) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 797) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 797) 

v_mul_lo_u32 v3, v4, s30 // (op 797) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 797) 


s_mul_i32 s15, s30, s13 // (op 799) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 799) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 799) least significant
s_mul_hi_u32 s1, s30, s12 // (op 799) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 799) most significant: sum
s_add_u32 s15, s15, s1 // (op 799) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 799) 

s_mov_b32 s8, s32 // (op 799) reg expression
s_mov_b32 s9, s33 // (op 799) reg expression
s_mov_b32 s10, s0 // (op 799) reg expression
s_mov_b32 s11, 131072 // (op 799) call()




buffer_store_dwordx4 v[12:15], v4, s[8:11], 0 offen // (op 48) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114:
v_lshrrev_b32 v0, 7, v1 // (op 673) 
v_and_b32 v6, 63, v1 // (op 673) 
v_lshrrev_b32 v3, 4, v6 // (op 673) 
v_lshlrev_b32 v4, 2, v3 // (op 673) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 673) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 673) 

v_lshrrev_b32 v3, 2, v6 // (op 673) 

v_and_b32 v4, 3, v3 // (op 673) 

v_and_b32 v3, 3, v1 // (op 673) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 673) 


v_lshrrev_b32 v3, 6, v1 // (op 673) 
v_and_b32 v4, 1, v3 // (op 673) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 673) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 673) 

v_mul_lo_u32 v3, v4, s34 // (op 673) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 673) 


s_mul_i32 s15, s34, s13 // (op 675) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 675) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 675) least significant
s_mul_hi_u32 s1, s34, s12 // (op 675) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 675) most significant: sum
s_add_u32 s15, s15, s1 // (op 675) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 675) 

s_mov_b32 s4, s36 // (op 675) reg expression
s_mov_b32 s5, s37 // (op 675) reg expression
s_mov_b32 s6, s0 // (op 675) reg expression
s_mov_b32 s7, 131072 // (op 675) call()





buffer_load_dwordx4 v[8:11], v4, s[4:7], 0 offen // (op 126) Load value



v_accvgpr_read v12, a0 // (op 123) 
v_accvgpr_read v13, a1 // (op 123) 
v_accvgpr_read v14, a2 // (op 123) 
v_accvgpr_read v15, a3 // (op 123) 
v_mul_f32 v16, v7, v12 // (op 123) 
v_mul_f32 v17, v7, v13 // (op 123) 
v_mul_f32 v18, v7, v14 // (op 123) 
v_mul_f32 v19, v7, v15 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v5, v8 // (op 127) 
v_mul_f32 v13, v5, v9 // (op 127) 
v_mul_f32 v14, v5, v10 // (op 127) 
v_mul_f32 v15, v5, v11 // (op 127) 

v_add_f32 v8, v12, v16 // (op 124) 
v_add_f32 v9, v13, v17 // (op 124) 
v_add_f32 v10, v14, v18 // (op 124) 
v_add_f32 v11, v15, v19 // (op 124) 


v_lshrrev_b32 v0, 7, v1 // (op 814) 
v_and_b32 v6, 63, v1 // (op 814) 
v_lshrrev_b32 v3, 4, v6 // (op 814) 
v_lshlrev_b32 v4, 2, v3 // (op 814) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 814) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 814) 

v_lshrrev_b32 v3, 2, v6 // (op 814) 

v_and_b32 v4, 3, v3 // (op 814) 

v_and_b32 v3, 3, v1 // (op 814) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 814) 


v_lshrrev_b32 v3, 6, v1 // (op 814) 
v_and_b32 v4, 1, v3 // (op 814) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 814) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 814) 

v_mul_lo_u32 v3, v4, s30 // (op 814) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 814) 


s_mul_i32 s15, s30, s13 // (op 816) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 816) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 816) least significant
s_mul_hi_u32 s1, s30, s12 // (op 816) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 816) most significant: sum
s_add_u32 s15, s15, s1 // (op 816) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 816) 

s_mov_b32 s8, s32 // (op 816) reg expression
s_mov_b32 s9, s33 // (op 816) reg expression
s_mov_b32 s10, s0 // (op 816) reg expression
s_mov_b32 s11, 131072 // (op 816) call()




buffer_store_dwordx4 v[8:11], v4, s[8:11], 0 offen // (op 125) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114:




s_endpgm  // End of RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.LRR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end:
.size RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2, .LRR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end-RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: F8
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
  - .name: RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
    .symbol: RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
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

