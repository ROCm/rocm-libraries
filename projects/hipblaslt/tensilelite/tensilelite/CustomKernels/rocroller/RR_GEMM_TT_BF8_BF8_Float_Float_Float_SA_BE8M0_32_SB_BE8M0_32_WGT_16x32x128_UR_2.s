// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.p2align 8
.type RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2,@function
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_exec_begin:


s_load_dwordx2 s[20:21], s[0:1], 56 // Load scalar value
s_load_dwordx2 s[22:23], s[0:1], 64 // Load scalar value
s_load_dwordx2 s[24:25], s[0:1], 72 // Load scalar value
s_load_dwordx2 s[26:27], s[0:1], 80 // Load scalar value
s_load_dwordx2 s[28:29], s[0:1], 88 // Load scalar value
s_load_dwordx2 s[30:31], s[0:1], 96 // Load scalar value
s_load_dwordx2 s[32:33], s[0:1], 104 // Load scalar value
s_load_dwordx2 s[34:35], s[0:1], 112 // Load scalar value
s_load_dwordx2 s[36:37], s[0:1], 120 // Load scalar value
s_load_dwordx2 s[38:39], s[0:1], 128 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


v_lshrrev_b32 v0, 7, v1 // (op 795) 
v_and_b32 v3, 15, v1 // (op 795) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 795) 


v_and_b32 v0, 48, v1 // (op 795) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 795) 


v_and_b32 v0, 48, v1 // (op 847) 
v_and_b32 v4, 63, v1 // (op 847) 
v_sub_u32 v5, v4, v0 // (op 847) 

v_lshrrev_b32 v4, 1, v5 // (op 847) 
v_add_lshl_u32 v6, v0, v4, 5 // (op 847) 


v_and_b32 v0, 1, v5 // (op 847) 

v_lshlrev_b32 v4, 3, v0 // (op 847) 

v_lshrrev_b32 v0, 6, v1 // (op 847) 
v_and_b32 v5, 1, v0 // (op 847) 

v_lshl_add_u32 v0, v5, 4, v4 // (op 847) 


v_add_u32 v4, v6, v0 // (op 847) 


v_lshrrev_b32 v0, 6, v1 // (op 900) 
v_and_b32 v5, 1, v0 // (op 900) 

v_and_b32 v0, 15, v1 // (op 900) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 900) 


v_and_b32 v0, 63, v1 // (op 900) 
v_lshrrev_b32 v5, 4, v0 // (op 900) 

v_lshl_add_u32 v0, v5, 5, v6 // (op 900) 


v_lshrrev_b32 v5, 7, v1 // (op 973) 
v_and_b32 v6, 15, v1 // (op 973) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 973) 


v_and_b32 v5, 48, v1 // (op 973) 
v_lshl_add_u32 v6, v7, 7, v5 // (op 973) 


v_and_b32 v5, 48, v1 // (op 1012) 
v_and_b32 v7, 63, v1 // (op 1012) 
v_sub_u32 v8, v7, v5 // (op 1012) 

v_lshrrev_b32 v7, 1, v8 // (op 1012) 
v_add_lshl_u32 v9, v5, v7, 5 // (op 1012) 


v_and_b32 v5, 1, v8 // (op 1012) 

v_lshlrev_b32 v7, 3, v5 // (op 1012) 

v_lshrrev_b32 v5, 6, v1 // (op 1012) 
v_and_b32 v8, 1, v5 // (op 1012) 

v_lshl_add_u32 v5, v8, 4, v7 // (op 1012) 


v_add_u32 v7, v9, v5 // (op 1012) 


v_lshrrev_b32 v5, 6, v1 // (op 1036) 
v_and_b32 v8, 1, v5 // (op 1036) 

v_and_b32 v5, 15, v1 // (op 1036) 
v_lshl_add_u32 v9, v8, 4, v5 // (op 1036) 


v_and_b32 v5, 63, v1 // (op 1036) 
v_lshrrev_b32 v8, 4, v5 // (op 1036) 

v_lshl_add_u32 v5, v8, 5, v9 // (op 1036) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_lshrrev_b32 v10, 3, v1 // (op 506) 
v_lshl_add_u32 v11, s16, 4, v10 // (op 506) 

v_and_b32 v10, -8, v1 // (op 506) 
v_sub_u32 v12, v1, v10 // (op 506) 

v_lshlrev_b32 v10, 4, v12 // (op 506) 

v_mov_b32 v13, 0 // (op 506) convert
v_mov_b32 v12, v10 // (op 506) convert

v_mul_lo_u32 v15, v11, s3 // (op 506) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 506) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 506) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 506) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 506) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 506) most significant half



s_mul_i32 s1, s2, s7 // (op 508) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 508) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 508) least significant
s_mul_hi_u32 s15, s2, s6 // (op 508) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 508) most significant: sum
s_add_u32 s1, s1, s15 // (op 508) most significant: sum


s_mov_b32 s40, s4 // (op 508) reg expression
s_mov_b32 s41, s5 // (op 508) reg expression
s_mov_b32 s42, s0 // (op 508) reg expression
s_mov_b32 s43, 131072 // (op 508) call()

v_lshrrev_b32 v10, 3, v1 // (op 525) 
v_lshl_add_u32 v11, s16, 4, v10 // (op 525) 

v_and_b32 v10, -8, v1 // (op 525) 
v_sub_u32 v12, v1, v10 // (op 525) 

v_lshlrev_b32 v10, 4, v12 // (op 525) 

v_mov_b32 v13, 0 // (op 525) convert
v_mov_b32 v12, v10 // (op 525) convert

v_mul_lo_u32 v17, v11, s3 // (op 525) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 525) most significant: high of low * low
v_mul_lo_u32 v16, v11, s2 // (op 525) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 525) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 525) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 525) most significant half



s_mul_i32 s1, s2, s7 // (op 527) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 527) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 527) least significant
s_mul_hi_u32 s15, s2, s6 // (op 527) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 527) most significant: sum
s_add_u32 s1, s1, s15 // (op 527) most significant: sum


s_mov_b32 s40, s4 // (op 527) reg expression
s_mov_b32 s41, s5 // (op 527) reg expression
s_mov_b32 s42, s0 // (op 527) reg expression
s_mov_b32 s43, 131072 // (op 527) call()

v_and_b32 v10, -2, v1 // (op 570) 
v_sub_u32 v11, v1, v10 // (op 570) 

v_lshlrev_b32 v10, 4, v11 // (op 570) 

v_lshl_add_u32 v11, s17, 5, v10 // (op 570) 

v_mov_b32 v13, 0 // (op 570) convert
v_mov_b32 v12, v11 // (op 570) convert

v_lshrrev_b32 v10, 1, v1 // (op 570) 
v_mul_lo_u32 v19, v10, s9 // (op 570) most significant: low of low * high
v_mul_hi_u32 v11, v10, s8 // (op 570) most significant: high of low * low
v_mul_lo_u32 v18, v10, s8 // (op 570) least significant: low of low * low
v_add_u32 v19, v19, v11 // (op 570) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v12 // (op 570) least significant half
v_addc_co_u32 v19, s[0:1], v19, v13, s[0:1] // (op 570) most significant half



s_mul_i32 s1, s8, s13 // (op 572) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 572) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 572) least significant
s_mul_hi_u32 s15, s8, s12 // (op 572) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 572) most significant: sum
s_add_u32 s1, s1, s15 // (op 572) most significant: sum


s_mov_b32 s44, s10 // (op 572) reg expression
s_mov_b32 s45, s11 // (op 572) reg expression
s_mov_b32 s46, s0 // (op 572) reg expression
s_mov_b32 s47, 131072 // (op 572) call()

v_and_b32 v10, -2, v1 // (op 589) 
v_sub_u32 v11, v1, v10 // (op 589) 

v_lshlrev_b32 v10, 4, v11 // (op 589) 

v_lshl_add_u32 v11, s17, 5, v10 // (op 589) 

v_mov_b32 v13, 0 // (op 589) convert
v_mov_b32 v12, v11 // (op 589) convert

v_lshrrev_b32 v10, 1, v1 // (op 589) 
v_mul_lo_u32 v21, v10, s9 // (op 589) most significant: low of low * high
v_mul_hi_u32 v11, v10, s8 // (op 589) most significant: high of low * low
v_mul_lo_u32 v20, v10, s8 // (op 589) least significant: low of low * low
v_add_u32 v21, v21, v11 // (op 589) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v12 // (op 589) least significant half
v_addc_co_u32 v21, s[0:1], v21, v13, s[0:1] // (op 589) most significant half



s_mul_i32 s1, s8, s13 // (op 591) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 591) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 591) least significant
s_mul_hi_u32 s15, s8, s12 // (op 591) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 591) most significant: sum
s_add_u32 s1, s1, s15 // (op 591) most significant: sum


s_mov_b32 s44, s10 // (op 591) reg expression
s_mov_b32 s45, s11 // (op 591) reg expression
s_mov_b32 s46, s0 // (op 591) reg expression
s_mov_b32 s47, 131072 // (op 591) call()

v_lshrrev_b32 v10, 7, v1 // (op 634) 
v_and_b32 v11, 15, v1 // (op 634) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 634) 


v_lshl_add_u32 v13, s16, 4, v12 // (op 634) 

v_and_b32 v10, 63, v1 // (op 634) 
v_lshrrev_b32 v11, 4, v10 // (op 634) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v10, v13, s20 // (op 634) 
v_add_u32 v10, v10, v11 // (op 634) 


s_mul_i32 s1, s20, s7 // (op 636) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 636) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 636) least significant
s_mul_hi_u32 s15, s20, s6 // (op 636) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 636) most significant: sum
s_add_u32 s1, s1, s15 // (op 636) most significant: sum


s_mov_b32 s48, s22 // (op 636) reg expression
s_mov_b32 s49, s23 // (op 636) reg expression
s_mov_b32 s50, s0 // (op 636) reg expression
s_mov_b32 s51, 131072 // (op 636) call()

v_lshrrev_b32 v11, 7, v1 // (op 653) 
v_and_b32 v12, 15, v1 // (op 653) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 653) 


v_lshl_add_u32 v11, s16, 4, v13 // (op 653) 

v_and_b32 v12, 63, v1 // (op 653) 
v_lshrrev_b32 v13, 4, v12 // (op 653) 

v_mul_lo_u32 v12, v11, s20 // (op 653) 
v_add_u32 v12, v12, v13 // (op 653) 


s_mul_i32 s1, s20, s7 // (op 655) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 655) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 655) least significant
s_mul_hi_u32 s15, s20, s6 // (op 655) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 655) most significant: sum
s_add_u32 s1, s1, s15 // (op 655) most significant: sum


s_mov_b32 s48, s22 // (op 655) reg expression
s_mov_b32 s49, s23 // (op 655) reg expression
s_mov_b32 s50, s0 // (op 655) reg expression
s_mov_b32 s51, 131072 // (op 655) call()

v_and_b32 v11, -32, v1 // (op 698) 
v_sub_u32 v13, v1, v11 // (op 698) 

v_lshl_add_u32 v11, s17, 5, v13 // (op 698) 

v_lshrrev_b32 v13, 5, v1 // (op 698) 
v_mul_lo_u32 v22, v13, s24 // (op 698) 
v_add_u32 v22, v22, v11 // (op 698) 


s_mul_i32 s1, s24, s29 // (op 700) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 700) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 700) least significant
s_mul_hi_u32 s15, s24, s28 // (op 700) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 700) most significant: sum
s_add_u32 s1, s1, s15 // (op 700) most significant: sum


s_mov_b32 s52, s26 // (op 700) reg expression
s_mov_b32 s53, s27 // (op 700) reg expression
s_mov_b32 s54, s0 // (op 700) reg expression
s_mov_b32 s55, 131072 // (op 700) call()

v_and_b32 v11, -32, v1 // (op 717) 
v_sub_u32 v13, v1, v11 // (op 717) 

v_lshl_add_u32 v11, s17, 5, v13 // (op 717) 

v_lshrrev_b32 v13, 5, v1 // (op 717) 
v_mul_lo_u32 v23, v13, s24 // (op 717) 
v_add_u32 v23, v23, v11 // (op 717) 


s_mul_i32 s1, s24, s29 // (op 719) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 719) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 719) least significant
s_mul_hi_u32 s15, s24, s28 // (op 719) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 719) most significant: sum
s_add_u32 s1, s1, s15 // (op 719) most significant: sum


s_mov_b32 s52, s26 // (op 719) reg expression
s_mov_b32 s53, s27 // (op 719) reg expression
s_mov_b32 s54, s0 // (op 719) reg expression
s_mov_b32 s55, 131072 // (op 719) call()




buffer_load_ubyte v11, v10, s[48:51], 0 offen offset: 0 // (op 337) Load value
v_and_b32 v13, -2, v1 // (op 835) 
v_sub_u32 v24, v1, v13 // (op 835) 

v_lshlrev_b32 v13, 4, v24 // (op 835) 

v_lshrrev_b32 v24, 1, v1 // (op 835) 
v_lshl_add_u32 v25, v24, 5, v13 // (op 835) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 835) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 6 // (op 455) 
v_mov_b64 v[24:25], s[14:15] // (op 455) convert


v_mov_b64 v[26:27], v[18:19] // (op 455) reg expression
s_mov_b32 m0, s0 // (op 455) reg expression
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 455) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 455) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 455) most significant half

s_add_u32 m0, m0, 2048 // (op 455) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 455) Load value direct to lds





v_mov_b32 v13, v22 // (op 344) reg expression

buffer_load_ubyte v24, v13, s[52:55], 0 offen offset: 0 // (op 344) Load value

v_and_b32 v13, -8, v1 // (op 783) 
v_sub_u32 v25, v1, v13 // (op 783) 

v_lshlrev_b32 v13, 4, v25 // (op 783) 

v_lshrrev_b32 v25, 3, v1 // (op 783) 
v_lshl_add_u32 v26, v25, 7, v13 // (op 783) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 783) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 460) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 460) reg expression
buffer_load_dwordx4 v14, s[40:43], 0 offen offset: 0 lds // (op 460) Load value direct to lds





buffer_load_ubyte v13, v10, s[48:51], 0 offen offset:4 // (op 351) Load value

v_and_b32 v10, -2, v1 // (op 1000) 
v_sub_u32 v25, v1, v10 // (op 1000) 

v_lshlrev_b32 v10, 4, v25 // (op 1000) 

v_lshrrev_b32 v25, 1, v1 // (op 1000) 
v_lshl_add_u32 v26, v25, 5, v10 // (op 1000) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1000) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 6 // (op 465) 
v_mov_b64 v[26:27], s[14:15] // (op 465) convert


s_lshl_b64 s[14:15], s[8:9], 7 // (op 465) 
v_mov_b32 v10, s14 // (op 465) 
v_mov_b32 v25, s15 // (op 465) 
v_add_co_u32 v28, s[18:19], v18, v10 // (op 465) least significant half
v_addc_co_u32 v29, s[18:19], v19, v25, s[18:19] // (op 465) most significant half




s_add_u32 s1, 6144, s0 // (op 465) 
s_mov_b32 m0, s1 // (op 465) reg expression
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 465) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v26 // (op 465) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 465) most significant half

s_add_u32 m0, m0, 2048 // (op 465) 
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 465) Load value direct to lds







s_lshl_b32 s0, s24, 2 // (op 358) 
v_add_u32 v10, v22, s0 // (op 358) 


buffer_load_ubyte v18, v10, s[52:55], 0 offen offset: 0 // (op 358) Load value


v_and_b32 v10, -8, v1 // (op 961) 
v_sub_u32 v19, v1, v10 // (op 961) 

v_lshlrev_b32 v10, 4, v19 // (op 961) 

v_lshrrev_b32 v19, 3, v1 // (op 961) 
v_lshl_add_u32 v22, v19, 7, v10 // (op 961) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 961) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 10240, s0 // (op 467) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 467) reg expression
s_mov_b32 s14, 128 // (op 467) call()
buffer_load_dwordx4 v14, s[40:43], s14 offen offset: 0 lds // (op 467) Load value direct to lds




v_and_b32 v10, -32, v1 // (op 888) 
v_sub_u32 v19, v1, v10 // (op 888) 

v_and_b32 v10, 4294967264, v1 // (op 888) 
v_add_u32 v22, v10, v19 // (op 888) 





s_waitcnt vmcnt(6)
ds_write_b8 v22, v24 offset:12288 // (op 364) Store local data 

s_barrier  // (op 1085) 



ds_read_b64_tr_b8 v[26:27], v4 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v4 offset:256 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v4 offset:2048 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:2304 // (op 381) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1101) 


ds_read_u8 v10, v0 offset:12288 // (op 384) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1069) 



ds_read_b128 v[34:37], v3 offset:4096 // (op 378) Load local data 
ds_read_b128 v[38:41], v3 offset:4160 // (op 378) Load local data 
v_and_b32 v19, -8, v1 // (op 755) 
v_sub_u32 v22, v1, v19 // (op 755) 

v_lshlrev_b32 v19, 4, v22 // (op 755) 

v_lshrrev_b32 v22, 3, v1 // (op 755) 
v_lshl_add_u32 v25, v22, 7, v19 // (op 755) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 755) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -2, v1 // (op 807) 
v_sub_u32 v22, v1, v19 // (op 807) 

v_lshlrev_b32 v19, 4, v22 // (op 807) 

v_lshrrev_b32 v22, 1, v1 // (op 807) 
v_lshl_add_u32 v25, v22, 5, v19 // (op 807) 


s_nop 0
v_readfirstlane_b32 s1, v25 // (op 807) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -32, v1 // (op 859) 
v_sub_u32 v22, v1, v19 // (op 859) 

v_and_b32 v19, 4294967264, v1 // (op 859) 
v_add_u32 v25, v19, v22 // (op 859) 


v_and_b32 v19, -8, v1 // (op 946) 
v_sub_u32 v22, v1, v19 // (op 946) 

v_lshlrev_b32 v19, 4, v22 // (op 946) 

v_lshrrev_b32 v22, 3, v1 // (op 946) 
v_lshl_add_u32 v14, v22, 7, v19 // (op 946) 


s_nop 0
v_readfirstlane_b32 s14, v14 // (op 946) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -2, v1 // (op 985) 
v_sub_u32 v22, v1, v19 // (op 985) 

v_lshlrev_b32 v19, 4, v22 // (op 985) 

v_lshrrev_b32 v22, 1, v1 // (op 985) 
v_lshl_add_u32 v14, v22, 5, v19 // (op 985) 


s_nop 0
v_readfirstlane_b32 s15, v14 // (op 985) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -32, v1 // (op 1024) 
v_sub_u32 v22, v1, v19 // (op 1024) 

v_and_b32 v19, 4294967264, v1 // (op 1024) 
v_add_u32 v14, v19, v22 // (op 1024) 


s_mov_b32 s18, 0 // (op 55) call()
s_ashr_i64 s[56:57], s[12:13], 7 // (op 54) 
s_mov_b32 s19, s56 // (op 54) convert

s_and_b32 s56, s19, 4294967294 // (op 54) 

s_cmp_lt_u32 s18, s56 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54 if false)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54:
s_add_u32 s18, s18, 2 // (op 56) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v11, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 82) 


v_mov_b32 v15, v12 // (op 75) 

buffer_load_ubyte v11, v15, s[48:51], 0 offen offset:8 // (op 75) Load value
s_barrier  // (op 1088) 

s_lshl_b64 s[56:57], s[8:9], 6 // (op 427) 
v_mov_b64 v[42:43], s[56:57] // (op 427) convert


v_mov_b64 v[44:45], v[20:21] // (op 427) 
s_lshl_b64 s[56:57], s[8:9], 7 // (op 427) 
s_lshl_b64 s[58:59], s[56:57], 1 // (op 427) 

v_mov_b32 v19, s58 // (op 427) 
v_mov_b32 v22, s59 // (op 427) 
v_add_co_u32 v46, s[56:57], v44, v19 // (op 427) least significant half
v_addc_co_u32 v47, s[56:57], v45, v22, s[56:57] // (op 427) most significant half




s_mov_b32 m0, s1 // (op 427) reg expression
buffer_load_dwordx4 v46, s[44:47], 0 offen offset: 0 lds // (op 427) Load value direct to lds
v_add_co_u32 v46, s[56:57], v46, v42 // (op 427) least significant half
v_addc_co_u32 v47, s[56:57], 0, v47, s[56:57] // (op 427) most significant half

s_add_u32 m0, m0, 2048 // (op 427) 
buffer_load_dwordx4 v46, s[44:47], 0 offen offset: 0 lds // (op 427) Load value direct to lds




v_mov_b32 v19, v23 // (op 21) 
s_lshl_b32 s19, s24, 3 // (op 21) 
v_add_u32 v22, v19, s19 // (op 21) 


buffer_load_ubyte v24, v22, s[52:55], 0 offen offset: 0 // (op 21) Load value

s_barrier  // (op 1072) 


v_mov_b64 v[42:43], v[16:17] // (op 422) 
s_add_u32 s19, 4096, s0 // (op 422) 

s_mov_b32 m0, s19 // (op 422) reg expression
s_mov_b32 s56, 256 // (op 422) call()
buffer_load_dwordx4 v42, s[40:43], s56 offen offset: 0 lds // (op 422) Load value direct to lds


s_barrier  // (op 1133) 



s_waitcnt vmcnt(6)
ds_write_b8 v14, v18 offset:12416 // (op 236) Store local data 
s_barrier  // (op 1124) 



ds_read_b64_tr_b8 v[26:27], v7 offset:6144 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v7 offset:6400 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v7 offset:8192 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v7 offset:8448 // (op 229) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1130) 


ds_read_u8 v10, v5 offset:12416 // (op 233) Load local data 
s_barrier  // (op 1176) 
s_waitcnt vmcnt(5)
s_barrier  // (op 1118) 



ds_read_b128 v[34:37], v6 offset:10240 // (op 227) Load local data 
ds_read_b128 v[38:41], v6 offset:10304 // (op 227) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v13, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 234) 



buffer_load_ubyte v13, v15, s[48:51], 0 offen offset:12 // (op 231) Load value

s_barrier  // (op 1127) 
v_add_u32 v12, 8, v12 // (op 645) 

s_lshl_b64 s[56:57], s[8:9], 6 // (op 450) 
v_mov_b64 v[46:47], s[56:57] // (op 450) convert


s_lshl_b64 s[56:57], s[8:9], 7 // (op 450) 
s_mul_i32 s59, 3, s57 // (op 450) most significant: low of low * high
s_mul_i32 s58, 3, s56 // (op 450) least significant
s_mul_hi_u32 s19, 3, s56 // (op 450) most significant: high of low * low
s_add_u32 s59, s59, s19 // (op 450) most significant: sum


v_mov_b32 v15, s58 // (op 450) 
v_mov_b32 v22, s59 // (op 450) 
v_add_co_u32 v48, s[56:57], v44, v15 // (op 450) least significant half
v_addc_co_u32 v49, s[56:57], v45, v22, s[56:57] // (op 450) most significant half




s_add_u32 s19, 6144, s15 // (op 450) 
s_mov_b32 m0, s19 // (op 450) reg expression
buffer_load_dwordx4 v48, s[44:47], 0 offen offset: 0 lds // (op 450) Load value direct to lds
v_add_co_u32 v48, s[56:57], v48, v46 // (op 450) least significant half
v_addc_co_u32 v49, s[56:57], 0, v49, s[56:57] // (op 450) most significant half

s_add_u32 m0, m0, 2048 // (op 450) 
buffer_load_dwordx4 v48, s[44:47], 0 offen offset: 0 lds // (op 450) Load value direct to lds






s_lshl_b32 s19, s24, 2 // (op 238) 
s_mul_i32 s56, 3, s19 // (op 238) 

v_add_u32 v15, v19, s56 // (op 238) 


s_lshl_b64 s[56:57], s[8:9], 7 // (op 581) 
s_lshl_b64 s[58:59], s[56:57], 1 // (op 581) 

v_mov_b32 v22, s58 // (op 581) 
v_mov_b32 v44, s59 // (op 581) 
v_add_co_u32 v20, s[56:57], v20, v22 // (op 581) least significant half
v_addc_co_u32 v21, s[56:57], v21, v44, s[56:57] // (op 581) most significant half




buffer_load_ubyte v18, v15, s[52:55], 0 offen offset: 0 // (op 238) Load value


s_barrier  // (op 1121) 
s_lshl_b32 s19, s24, 3 // (op 709) 


v_add_u32 v23, v23, s19 // (op 709) 

s_add_u32 s19, 10240, s14 // (op 445) 

s_mov_b32 m0, s19 // (op 445) reg expression
s_mov_b32 s56, 384 // (op 445) call()
buffer_load_dwordx4 v42, s[40:43], s56 offen offset: 0 lds // (op 445) Load value direct to lds



s_barrier  // (op 1104) 
v_add_co_u32 v16, vcc, 256, v16 // (op 517) least significant half



v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 517) most significant half
s_waitcnt vmcnt(6)
ds_write_b8 v25, v24 offset:12288 // (op 26) Store local data 
s_barrier  // (op 1094) 



ds_read_b64_tr_b8 v[26:27], v4 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v4 offset:256 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v4 offset:2048 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:2304 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1111) 


ds_read_u8 v10, v0 offset:12288 // (op 79) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1078) 



ds_read_b128 v[34:37], v3 offset:4096 // (op 65) Load local data 
ds_read_b128 v[38:41], v3 offset:4160 // (op 65) Load local data 
s_ashr_i64 s[56:57], s[12:13], 7 // (op 54) 
s_mov_b32 s19, s56 // (op 54) convert

s_and_b32 s56, s19, 4294967294 // (op 54) 

s_cmp_lt_u32 s18, s56 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54 if true)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54:













s_barrier  // (op 1142) 
v_lshrrev_b32 v11, 3, v1 // (op 480) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 480) 
v_lshl_add_u32 v12, s16, 4, v11 // (op 480) 

s_lshr_b32 s14, s0, 1 // (op 480) 

v_and_b32 v11, -8, v1 // (op 480) 
v_sub_u32 v14, v1, v11 // (op 480) 

v_lshlrev_b32 v11, 4, v14 // (op 480) 

v_lshl_add_u32 v14, s14, 8, v11 // (op 480) 


v_mov_b32 v17, 0 // (op 480) convert
v_mov_b32 v16, v14 // (op 480) convert

v_mul_lo_u32 v15, v12, s3 // (op 480) most significant: low of low * high
v_mul_hi_u32 v11, v12, s2 // (op 480) most significant: high of low * low
v_mul_lo_u32 v14, v12, s2 // (op 480) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 480) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v16 // (op 480) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 480) most significant half



s_mul_i32 s1, s2, s7 // (op 482) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 482) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 482) least significant
s_mul_hi_u32 s15, s2, s6 // (op 482) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 482) most significant: sum
s_add_u32 s1, s1, s15 // (op 482) most significant: sum


s_mov_b32 s40, s4 // (op 482) reg expression
s_mov_b32 s41, s5 // (op 482) reg expression
s_mov_b32 s42, s0 // (op 482) reg expression
s_mov_b32 s43, 131072 // (op 482) call()

v_lshrrev_b32 v11, 1, v1 // (op 547) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 547) 
v_and_b32 v12, -2, v1 // (op 547) 
s_lshr_b32 s14, s0, 1 // (op 547) 

v_sub_u32 v16, v1, v12 // (op 547) 

v_lshl_add_u32 v12, s14, 8, v11 // (op 547) 


v_lshlrev_b32 v11, 4, v16 // (op 547) 

v_lshl_add_u32 v16, s17, 5, v11 // (op 547) 

v_mov_b32 v25, 0 // (op 547) convert
v_mov_b32 v24, v16 // (op 547) convert

v_mul_lo_u32 v17, v12, s9 // (op 547) most significant: low of low * high
v_mul_hi_u32 v11, v12, s8 // (op 547) most significant: high of low * low
v_mul_lo_u32 v16, v12, s8 // (op 547) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 547) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v24 // (op 547) least significant half
v_addc_co_u32 v17, s[0:1], v17, v25, s[0:1] // (op 547) most significant half



s_mul_i32 s1, s8, s13 // (op 549) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 549) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 549) least significant
s_mul_hi_u32 s15, s8, s12 // (op 549) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 549) most significant: sum
s_add_u32 s1, s1, s15 // (op 549) most significant: sum


s_mov_b32 s44, s10 // (op 549) reg expression
s_mov_b32 s45, s11 // (op 549) reg expression
s_mov_b32 s46, s0 // (op 549) reg expression
s_mov_b32 s47, 131072 // (op 549) call()

v_lshrrev_b32 v11, 7, v1 // (op 611) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 611) 
v_and_b32 v12, 15, v1 // (op 611) 
s_lshr_b32 s14, s0, 1 // (op 611) 

v_lshl_add_u32 v19, v11, 4, v12 // (op 611) 


v_lshl_add_u32 v11, s16, 4, v19 // (op 611) 

v_and_b32 v12, 63, v1 // (op 611) 
v_lshrrev_b32 v19, 4, v12 // (op 611) 

v_lshl_add_u32 v12, s14, 3, v19 // (op 611) 


v_mul_lo_u32 v19, v11, s20 // (op 611) 
v_add_u32 v19, v19, v12 // (op 611) 


s_mul_i32 s1, s20, s7 // (op 613) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 613) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 613) least significant
s_mul_hi_u32 s15, s20, s6 // (op 613) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 613) most significant: sum
s_add_u32 s1, s1, s15 // (op 613) most significant: sum


s_mov_b32 s48, s22 // (op 613) reg expression
s_mov_b32 s49, s23 // (op 613) reg expression
s_mov_b32 s50, s0 // (op 613) reg expression
s_mov_b32 s51, 131072 // (op 613) call()

v_lshrrev_b32 v11, 5, v1 // (op 675) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 675) 
v_and_b32 v12, -32, v1 // (op 675) 
s_lshr_b32 s14, s0, 1 // (op 675) 

v_sub_u32 v20, v1, v12 // (op 675) 

v_lshl_add_u32 v12, s14, 3, v11 // (op 675) 


v_lshl_add_u32 v11, s17, 5, v20 // (op 675) 

v_mul_lo_u32 v20, v12, s24 // (op 675) 
v_add_u32 v20, v20, v11 // (op 675) 


s_mul_i32 s1, s24, s29 // (op 677) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 677) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 677) least significant
s_mul_hi_u32 s15, s24, s28 // (op 677) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 677) most significant: sum
s_add_u32 s1, s1, s15 // (op 677) most significant: sum


s_mov_b32 s52, s26 // (op 677) reg expression
s_mov_b32 s53, s27 // (op 677) reg expression
s_mov_b32 s54, s0 // (op 677) reg expression
s_mov_b32 s55, 131072 // (op 677) call()

v_and_b32 v11, -8, v1 // (op 768) 
v_sub_u32 v12, v1, v11 // (op 768) 

v_lshlrev_b32 v11, 4, v12 // (op 768) 

v_lshrrev_b32 v12, 3, v1 // (op 768) 
v_lshl_add_u32 v21, v12, 7, v11 // (op 768) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 768) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v11, -2, v1 // (op 820) 
v_sub_u32 v12, v1, v11 // (op 820) 

v_lshlrev_b32 v11, 4, v12 // (op 820) 

v_lshrrev_b32 v12, 1, v1 // (op 820) 
v_lshl_add_u32 v21, v12, 5, v11 // (op 820) 


s_nop 0
v_readfirstlane_b32 s1, v21 // (op 820) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v11, -32, v1 // (op 872) 
v_sub_u32 v12, v1, v11 // (op 872) 

v_and_b32 v11, 4294967264, v1 // (op 872) 
v_add_u32 v21, v11, v12 // (op 872) 


s_ashr_i64 s[14:15], s[12:13], 7 // (op 162) 
s_and_b32 s18, s14, 4294967294 // (op 162) 

s_ashr_i64 s[14:15], s[12:13], 7 // (op 161) 
s_mov_b32 s19, s14 // (op 161) convert

s_cmp_lt_u32 s18, s19 // (op 161) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161:
s_barrier  // (op 1075) 
s_add_u32 s18, s18, 1 // (op 163) 


v_mov_b64 v[10:11], v[14:15] // (op 432) 
s_add_u32 s19, 4096, s0 // (op 432) 

s_mov_b32 m0, s19 // (op 432) reg expression
buffer_load_dwordx4 v10, s[40:43], 0 offen offset: 0 lds // (op 432) Load value direct to lds


v_add_co_u32 v14, vcc, 128, v14 // (op 470) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 470) most significant half
s_barrier  // (op 1091) 

s_lshl_b64 s[14:15], s[8:9], 6 // (op 438) 
v_mov_b64 v[10:11], s[14:15] // (op 438) convert


v_mov_b64 v[12:13], v[16:17] // (op 438) 
v_mov_b64 v[22:23], v[12:13] // (op 438) reg expression
s_mov_b32 m0, s1 // (op 438) reg expression
buffer_load_dwordx4 v22, s[44:47], 0 offen offset: 0 lds // (op 438) Load value direct to lds
v_add_co_u32 v22, s[14:15], v22, v10 // (op 438) least significant half
v_addc_co_u32 v23, s[14:15], 0, v23, s[14:15] // (op 438) most significant half

s_add_u32 m0, m0, 2048 // (op 438) 
buffer_load_dwordx4 v22, s[44:47], 0 offen offset: 0 lds // (op 438) Load value direct to lds



s_lshl_b64 s[14:15], s[8:9], 7 // (op 539) 
v_mov_b32 v18, s14 // (op 539) 
v_mov_b32 v10, s15 // (op 539) 
v_add_co_u32 v16, s[56:57], v16, v18 // (op 539) least significant half
v_addc_co_u32 v17, s[56:57], v17, v10, s[56:57] // (op 539) most significant half






v_mov_b32 v18, v20 // (op 180) 
v_mov_b32 v10, v18 // (op 180) reg expression

buffer_load_ubyte v11, v10, s[52:55], 0 offen offset: 0 // (op 180) Load value


s_lshl_b32 s19, s24, 2 // (op 667) 
v_add_u32 v20, v20, s19 // (op 667) 

s_barrier  // (op 1107) 
s_waitcnt vmcnt(3)
s_barrier  // (op 1081) 






ds_read_b128 v[22:25], v3 offset:4096 // (op 169) Load local data 
ds_read_b128 v[26:29], v3 offset:4160 // (op 169) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v21, v11 offset:12288 // (op 178) Store local data 

s_barrier  // (op 1097) 



ds_read_b64_tr_b8 v[30:31], v4 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:256 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:2048 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:2304 // (op 171) Transpose load local data 


v_mov_b32 v18, v19 // (op 173) 

buffer_load_ubyte v10, v18, s[48:51], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v19, 4, v19 // (op 603) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1114) 


ds_read_u8 v18, v0 offset:12288 // (op 175) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:37], a[0:3], v10, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 176) 




s_ashr_i64 s[14:15], s[12:13], 7 // (op 161) 
s_mov_b32 s19, s14 // (op 161) convert

s_cmp_lt_u32 s18, s19 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161 if true)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161:























s_barrier  // (op 1166) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

v_lshrrev_b32 v0, 7, v1 // (op 914) 
v_and_b32 v3, 63, v1 // (op 914) 
v_lshrrev_b32 v4, 4, v3 // (op 914) 
v_lshlrev_b32 v5, 2, v4 // (op 914) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 914) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 914) 

v_lshrrev_b32 v4, 2, v3 // (op 914) 

v_and_b32 v3, 3, v4 // (op 914) 

v_and_b32 v4, 3, v1 // (op 914) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 914) 


v_lshrrev_b32 v3, 6, v1 // (op 914) 
v_and_b32 v4, 1, v3 // (op 914) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 914) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 914) 

v_mul_lo_u32 v3, v4, s30 // (op 914) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 914) 


s_mul_i32 s1, s30, s35 // (op 916) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 916) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 916) least significant
s_mul_hi_u32 s3, s30, s34 // (op 916) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 916) most significant: sum
s_add_u32 s1, s1, s3 // (op 916) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 916) 

s_mov_b32 s4, s32 // (op 916) reg expression
s_mov_b32 s5, s33 // (op 916) reg expression
s_mov_b32 s6, s2 // (op 916) reg expression
s_mov_b32 s7, 131072 // (op 916) call()




buffer_store_dwordx4 v[10:13], v4, s[4:7], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123:
v_lshrrev_b32 v0, 7, v1 // (op 738) 
v_and_b32 v3, 63, v1 // (op 738) 
v_lshrrev_b32 v4, 4, v3 // (op 738) 
v_lshlrev_b32 v5, 2, v4 // (op 738) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 738) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 738) 

v_lshrrev_b32 v4, 2, v3 // (op 738) 

v_and_b32 v3, 3, v4 // (op 738) 

v_and_b32 v4, 3, v1 // (op 738) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 738) 


v_lshrrev_b32 v3, 6, v1 // (op 738) 
v_and_b32 v4, 1, v3 // (op 738) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 738) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 738) 

v_mul_lo_u32 v3, v4, s36 // (op 738) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 738) 


s_mul_i32 s1, s36, s35 // (op 740) most significant: low of low * high
s_mul_i32 s2, s37, s34 // (op 740) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 740) least significant
s_mul_hi_u32 s3, s36, s34 // (op 740) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 740) most significant: sum
s_add_u32 s1, s1, s3 // (op 740) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 740) 

s_mov_b32 s8, s38 // (op 740) reg expression
s_mov_b32 s9, s39 // (op 740) reg expression
s_mov_b32 s10, s2 // (op 740) reg expression
s_mov_b32 s11, 131072 // (op 740) call()





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


v_lshrrev_b32 v0, 7, v1 // (op 931) 
v_and_b32 v3, 63, v1 // (op 931) 
v_lshrrev_b32 v4, 4, v3 // (op 931) 
v_lshlrev_b32 v5, 2, v4 // (op 931) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 931) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 931) 

v_lshrrev_b32 v4, 2, v3 // (op 931) 

v_and_b32 v3, 3, v4 // (op 931) 

v_and_b32 v4, 3, v1 // (op 931) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 931) 


v_lshrrev_b32 v3, 6, v1 // (op 931) 
v_and_b32 v4, 1, v3 // (op 931) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 931) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 931) 

v_mul_lo_u32 v3, v4, s30 // (op 931) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 931) 


s_mul_i32 s1, s30, s35 // (op 933) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 933) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 933) least significant
s_mul_hi_u32 s3, s30, s34 // (op 933) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 933) most significant: sum
s_add_u32 s1, s1, s3 // (op 933) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 933) 

s_mov_b32 s4, s32 // (op 933) reg expression
s_mov_b32 s5, s33 // (op 933) reg expression
s_mov_b32 s6, s2 // (op 933) reg expression
s_mov_b32 s7, 131072 // (op 933) call()




buffer_store_dwordx4 v[10:13], v4, s[4:7], 0 offen // (op 134) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123:



s_endpgm  // End of RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.LRR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end:
.size RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2, .LRR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end-RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
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
    DataTypeB: B8
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
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
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeSum },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSumScaleA },
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
  - .name: RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
    .symbol: RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 12544
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 50
    .agpr_count: 4
    .max_flat_workgroup_size: 128
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 8
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
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
      - .name: Tensor_0_size_1_4
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
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
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
      - .name: Tensor_4_size_1_9
        .size: 8
        .offset: 88
        .value_kind: by_value
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
      - .name: Tensor_2_size_1_12
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_10_stride_1_11
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 128
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

