// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2,@function
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 795) 
v_and_b32 v3, 63, v1 // (op 795) 
v_sub_u32 v4, v3, v0 // (op 795) 

v_lshrrev_b32 v3, 1, v4 // (op 795) 
v_add_lshl_u32 v5, v0, v3, 5 // (op 795) 


v_and_b32 v0, 1, v4 // (op 795) 

v_lshlrev_b32 v3, 3, v0 // (op 795) 

v_lshrrev_b32 v0, 6, v1 // (op 795) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 795) 


v_add_u32 v0, v5, v4 // (op 795) 


v_and_b32 v3, 48, v1 // (op 847) 
v_and_b32 v4, 63, v1 // (op 847) 
v_sub_u32 v5, v4, v3 // (op 847) 

v_lshrrev_b32 v4, 1, v5 // (op 847) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 847) 


v_and_b32 v3, 1, v5 // (op 847) 

v_lshlrev_b32 v4, 3, v3 // (op 847) 

v_add_u32 v3, v6, v4 // (op 847) 


v_lshrrev_b32 v4, 6, v1 // (op 900) 
v_and_b32 v5, 15, v1 // (op 900) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 900) 


v_and_b32 v4, 63, v1 // (op 900) 
v_lshrrev_b32 v5, 4, v4 // (op 900) 

v_lshl_add_u32 v4, v5, 5, v6 // (op 900) 


v_and_b32 v5, 48, v1 // (op 973) 
v_and_b32 v6, 63, v1 // (op 973) 
v_sub_u32 v7, v6, v5 // (op 973) 

v_lshrrev_b32 v6, 1, v7 // (op 973) 
v_add_lshl_u32 v8, v5, v6, 5 // (op 973) 


v_and_b32 v5, 1, v7 // (op 973) 

v_lshlrev_b32 v6, 3, v5 // (op 973) 

v_lshrrev_b32 v5, 6, v1 // (op 973) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 973) 


v_add_u32 v5, v8, v7 // (op 973) 


v_and_b32 v6, 48, v1 // (op 1012) 
v_and_b32 v7, 63, v1 // (op 1012) 
v_sub_u32 v8, v7, v6 // (op 1012) 

v_lshrrev_b32 v7, 1, v8 // (op 1012) 
v_add_lshl_u32 v9, v6, v7, 4 // (op 1012) 


v_and_b32 v6, 1, v8 // (op 1012) 

v_lshlrev_b32 v7, 3, v6 // (op 1012) 

v_add_u32 v6, v9, v7 // (op 1012) 


v_lshrrev_b32 v7, 6, v1 // (op 1036) 
v_and_b32 v8, 15, v1 // (op 1036) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 1036) 


v_and_b32 v7, 63, v1 // (op 1036) 
v_lshrrev_b32 v8, 4, v7 // (op 1036) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 1036) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_and_b32 v10, -2, v1 // (op 506) 
v_sub_u32 v11, v1, v10 // (op 506) 

v_lshlrev_b32 v10, 4, v11 // (op 506) 

v_lshl_add_u32 v11, s16, 5, v10 // (op 506) 

v_mov_b32 v13, 0 // (op 506) convert
v_mov_b32 v12, v11 // (op 506) convert

v_lshrrev_b32 v10, 1, v1 // (op 506) 
v_mul_lo_u32 v15, v10, s3 // (op 506) most significant: low of low * high
v_mul_hi_u32 v11, v10, s2 // (op 506) most significant: high of low * low
v_mul_lo_u32 v14, v10, s2 // (op 506) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 506) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 506) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 506) most significant half



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

v_and_b32 v12, -2, v1 // (op 525) 
v_sub_u32 v13, v1, v12 // (op 525) 

v_lshlrev_b32 v12, 4, v13 // (op 525) 

v_lshl_add_u32 v13, s16, 5, v12 // (op 525) 

v_mov_b32 v15, 0 // (op 525) convert
v_mov_b32 v14, v13 // (op 525) convert

v_lshrrev_b32 v12, 1, v1 // (op 525) 
v_mul_lo_u32 v17, v12, s3 // (op 525) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 525) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 525) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 525) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 525) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 525) most significant half



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

s_lshl_b32 s0, s17, 4 // (op 570) 
s_mov_b32 s15, 0 // (op 570) convert
s_mov_b32 s14, s0 // (op 570) convert

v_mul_lo_u32 v15, v1, s9 // (op 570) most significant: low of low * high
v_mul_hi_u32 v16, v1, s8 // (op 570) most significant: high of low * low
v_mul_lo_u32 v14, v1, s8 // (op 570) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 570) most significant: sum

v_mov_b32 v16, s14 // (op 570) 
v_mov_b32 v17, s15 // (op 570) 
v_add_co_u32 v14, s[0:1], v14, v16 // (op 570) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 570) most significant half




s_mul_i32 s1, s8, s7 // (op 572) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 572) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 572) least significant
s_mul_hi_u32 s15, s8, s6 // (op 572) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 572) most significant: sum
s_add_u32 s1, s1, s15 // (op 572) most significant: sum


s_mov_b32 s44, s10 // (op 572) reg expression
s_mov_b32 s45, s11 // (op 572) reg expression
s_mov_b32 s46, s0 // (op 572) reg expression
s_mov_b32 s47, 131072 // (op 572) call()

s_lshl_b32 s0, s17, 4 // (op 589) 
s_mov_b32 s15, 0 // (op 589) convert
s_mov_b32 s14, s0 // (op 589) convert

v_mul_lo_u32 v17, v1, s9 // (op 589) most significant: low of low * high
v_mul_hi_u32 v18, v1, s8 // (op 589) most significant: high of low * low
v_mul_lo_u32 v16, v1, s8 // (op 589) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 589) most significant: sum

v_mov_b32 v18, s14 // (op 589) 
v_mov_b32 v19, s15 // (op 589) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 589) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 589) most significant half




s_mul_i32 s1, s8, s7 // (op 591) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 591) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 591) least significant
s_mul_hi_u32 s15, s8, s6 // (op 591) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 591) most significant: sum
s_add_u32 s1, s1, s15 // (op 591) most significant: sum


s_mov_b32 s44, s10 // (op 591) reg expression
s_mov_b32 s45, s11 // (op 591) reg expression
s_mov_b32 s46, s0 // (op 591) reg expression
s_mov_b32 s47, 131072 // (op 591) call()

v_and_b32 v18, -32, v1 // (op 634) 
v_sub_u32 v19, v1, v18 // (op 634) 

v_lshl_add_u32 v18, s16, 5, v19 // (op 634) 

v_lshrrev_b32 v19, 5, v1 // (op 634) 
v_mul_lo_u32 v20, v19, s12 // (op 634) 

v_add_u32 v19, v18, v20 // (op 634) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 636) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 636) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 636) least significant
s_mul_hi_u32 s15, s12, s22 // (op 636) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 636) most significant: sum
s_add_u32 s1, s1, s15 // (op 636) most significant: sum


s_mov_b32 s48, s20 // (op 636) reg expression
s_mov_b32 s49, s21 // (op 636) reg expression
s_mov_b32 s50, s0 // (op 636) reg expression
s_mov_b32 s51, 131072 // (op 636) call()

v_and_b32 v18, -32, v1 // (op 653) 
v_sub_u32 v20, v1, v18 // (op 653) 

v_lshl_add_u32 v18, s16, 5, v20 // (op 653) 

v_lshrrev_b32 v20, 5, v1 // (op 653) 
v_mul_lo_u32 v21, v20, s12 // (op 653) 

v_add_u32 v20, v18, v21 // (op 653) 


s_mul_i32 s1, s12, s23 // (op 655) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 655) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 655) least significant
s_mul_hi_u32 s15, s12, s22 // (op 655) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 655) most significant: sum
s_add_u32 s1, s1, s15 // (op 655) most significant: sum


s_mov_b32 s48, s20 // (op 655) reg expression
s_mov_b32 s49, s21 // (op 655) reg expression
s_mov_b32 s50, s0 // (op 655) reg expression
s_mov_b32 s51, 131072 // (op 655) call()

v_and_b32 v18, 15, v1 // (op 698) 
v_lshl_add_u32 v21, s17, 4, v18 // (op 698) 

v_and_b32 v18, 63, v1 // (op 698) 
v_lshrrev_b32 v22, 4, v18 // (op 698) 

v_mul_lo_u32 v18, v22, s24 // (op 698) 
v_add_u32 v18, v18, v21 // (op 698) 


s_mul_i32 s1, s24, s23 // (op 700) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 700) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 700) least significant
s_mul_hi_u32 s15, s24, s22 // (op 700) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 700) most significant: sum
s_add_u32 s1, s1, s15 // (op 700) most significant: sum


s_mov_b32 s52, s26 // (op 700) reg expression
s_mov_b32 s53, s27 // (op 700) reg expression
s_mov_b32 s54, s0 // (op 700) reg expression
s_mov_b32 s55, 131072 // (op 700) call()

v_and_b32 v21, 15, v1 // (op 717) 
v_lshl_add_u32 v22, s17, 4, v21 // (op 717) 

v_and_b32 v21, 63, v1 // (op 717) 
v_lshrrev_b32 v23, 4, v21 // (op 717) 

v_mul_lo_u32 v21, v23, s24 // (op 717) 
v_add_u32 v21, v21, v22 // (op 717) 


s_mul_i32 s1, s24, s23 // (op 719) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 719) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 719) least significant
s_mul_hi_u32 s15, s24, s22 // (op 719) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 719) most significant: sum
s_add_u32 s1, s1, s15 // (op 719) most significant: sum


s_mov_b32 s52, s26 // (op 719) reg expression
s_mov_b32 s53, s27 // (op 719) reg expression
s_mov_b32 s54, s0 // (op 719) reg expression
s_mov_b32 s55, 131072 // (op 719) call()


v_mov_b64 v[22:23], s[24:25] // (op 337) convert

v_mov_b32 v24, v18 // (op 337) reg expression
v_mov_b32 v25, v24 // (op 337) 
buffer_load_ubyte v26, v25, s[52:55], 0 offen offset: 0 // (op 337) Load value



v_lshlrev_b32 v22, 4, v1 // (op 835) 
s_nop 0
v_readfirstlane_b32 s0, v22 // (op 835) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[22:23], v[14:15] // (op 455) reg expression

s_nop 0
s_mov_b32 m0, s0 // (op 455) reg expression
buffer_load_dwordx4 v22, s[44:47], 0 offen offset: 0 lds // (op 455) Load value direct to lds


v_and_b32 v22, -2, v1 // (op 783) 
v_sub_u32 v23, v1, v22 // (op 783) 

v_lshlrev_b32 v22, 4, v23 // (op 783) 

v_lshrrev_b32 v23, 1, v1 // (op 783) 
v_lshl_add_u32 v24, v23, 5, v22 // (op 783) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 783) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 460) 
v_mov_b64 v[22:23], s[14:15] // (op 460) convert


v_mov_b64 v[24:25], v[10:11] // (op 460) reg expression
s_add_u32 s1, 2048, s0 // (op 460) 
s_mov_b32 m0, s1 // (op 460) reg expression
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 460) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v22 // (op 460) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 460) most significant half

s_add_u32 m0, m0, 2048 // (op 460) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 460) Load value direct to lds






v_mov_b32 v22, v19 // (op 347) reg expression

buffer_load_ubyte v23, v22, s[48:51], 0 offen offset: 0 // (op 347) Load value


v_mov_b64 v[24:25], s[24:25] // (op 351) convert

s_lshl_b32 s0, s24, 2 // (op 351) 
v_add_u32 v22, v18, s0 // (op 351) 

v_mov_b32 v27, v22 // (op 351) 
buffer_load_ubyte v28, v27, s[52:55], 0 offen offset: 0 // (op 351) Load value




v_lshlrev_b32 v18, 4, v1 // (op 1000) 
s_nop 0
v_readfirstlane_b32 s0, v18 // (op 1000) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[14:15], s[8:9], 7 // (op 465) 
v_mov_b32 v18, s14 // (op 465) 
v_mov_b32 v22, s15 // (op 465) 
v_add_co_u32 v24, s[18:19], v14, v18 // (op 465) least significant half
v_addc_co_u32 v25, s[18:19], v15, v22, s[18:19] // (op 465) most significant half




s_add_u32 s1, 6144, s0 // (op 465) 

s_mov_b32 m0, s1 // (op 465) reg expression
buffer_load_dwordx4 v24, s[44:47], 0 offen offset: 0 lds // (op 465) Load value direct to lds




v_and_b32 v18, -2, v1 // (op 961) 
v_sub_u32 v22, v1, v18 // (op 961) 

v_lshlrev_b32 v18, 4, v22 // (op 961) 

v_lshrrev_b32 v22, 1, v1 // (op 961) 
v_lshl_add_u32 v27, v22, 5, v18 // (op 961) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 961) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 467) 
v_mov_b64 v[14:15], s[14:15] // (op 467) convert


s_lshl_b64 s[14:15], s[2:3], 7 // (op 467) 
v_mov_b32 v18, s14 // (op 467) 
v_mov_b32 v22, s15 // (op 467) 
v_add_co_u32 v24, s[18:19], v10, v18 // (op 467) least significant half
v_addc_co_u32 v25, s[18:19], v11, v22, s[18:19] // (op 467) most significant half




s_add_u32 s1, 8192, s0 // (op 467) 
s_mov_b32 m0, s1 // (op 467) reg expression
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 467) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v14 // (op 467) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 467) most significant half

s_add_u32 m0, m0, 2048 // (op 467) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 467) Load value direct to lds







s_lshl_b32 s0, s12, 2 // (op 361) 
v_add_u32 v18, v19, s0 // (op 361) 


buffer_load_ubyte v22, v18, s[48:51], 0 offen offset: 0 // (op 361) Load value


v_and_b32 v27, -32, v1 // (op 888) 
v_sub_u32 v10, v1, v27 // (op 888) 

v_and_b32 v11, 4294967264, v1 // (op 888) 
v_add_u32 v27, v11, v10 // (op 888) 





s_waitcnt vmcnt(5)
ds_write_b8 v27, v23 offset:12288 // (op 365) Store local data 

s_barrier  // (op 1085) 



ds_read_b64_tr_b8 v[30:31], v3 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v3 offset:128 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:1024 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v3 offset:1152 // (op 381) Transpose load local data 
s_barrier  // (op 1069) 



ds_read_b64_tr_b8 v[38:39], v0 offset:2048 // (op 378) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:2304 // (op 378) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:4096 // (op 378) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:4352 // (op 378) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1101) 


ds_read_u8 v27, v4 offset:12288 // (op 384) Load local data 
v_and_b32 v29, -2, v1 // (op 755) 
v_sub_u32 v10, v1, v29 // (op 755) 

v_lshlrev_b32 v11, 4, v10 // (op 755) 

v_lshrrev_b32 v10, 1, v1 // (op 755) 
v_lshl_add_u32 v29, v10, 5, v11 // (op 755) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 755) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 4, v1 // (op 807) 
s_nop 0
v_readfirstlane_b32 s1, v29 // (op 807) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -32, v1 // (op 859) 
v_sub_u32 v10, v1, v29 // (op 859) 

v_and_b32 v11, 4294967264, v1 // (op 859) 
v_add_u32 v29, v11, v10 // (op 859) 


v_and_b32 v10, -2, v1 // (op 946) 
v_sub_u32 v11, v1, v10 // (op 946) 

v_lshlrev_b32 v10, 4, v11 // (op 946) 

v_lshrrev_b32 v11, 1, v1 // (op 946) 
v_lshl_add_u32 v14, v11, 5, v10 // (op 946) 


s_nop 0
v_readfirstlane_b32 s14, v14 // (op 946) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v10, 4, v1 // (op 985) 
s_nop 0
v_readfirstlane_b32 s15, v10 // (op 985) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -32, v1 // (op 1024) 
v_sub_u32 v11, v1, v10 // (op 1024) 

v_and_b32 v10, 4294967264, v1 // (op 1024) 
v_add_u32 v14, v10, v11 // (op 1024) 


s_mov_b32 s18, 0 // (op 55) call()
s_ashr_i64 s[38:39], s[6:7], 7 // (op 54) 
s_mov_b32 s19, s38 // (op 54) convert

s_and_b32 s38, s19, 4294967294 // (op 54) 

s_cmp_lt_u32 s18, s38 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 if false)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54:
s_add_u32 s18, s18, 2 // (op 56) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v27, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 82) 

v_mov_b64 v[10:11], s[24:25] // (op 80) convert

v_mov_b32 v15, v21 // (op 80) 
s_lshl_b32 s19, s24, 3 // (op 80) 
v_add_u32 v18, v15, s19 // (op 80) 

v_mov_b32 v19, v18 // (op 80) 
buffer_load_ubyte v26, v19, s[52:55], 0 offen offset: 0 // (op 80) Load value



s_barrier  // (op 1088) 


v_mov_b64 v[10:11], v[16:17] // (op 427) 
s_lshl_b64 s[38:39], s[8:9], 7 // (op 427) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 427) 

v_mov_b32 v18, s56 // (op 427) 
v_mov_b32 v19, s57 // (op 427) 
v_add_co_u32 v24, s[38:39], v10, v18 // (op 427) least significant half
v_addc_co_u32 v25, s[38:39], v11, v19, s[38:39] // (op 427) most significant half





s_mov_b32 m0, s1 // (op 427) reg expression
buffer_load_dwordx4 v24, s[44:47], 0 offen offset: 0 lds // (op 427) Load value direct to lds

s_barrier  // (op 1072) 

s_lshl_b64 s[38:39], s[2:3], 6 // (op 422) 
v_mov_b64 v[18:19], s[38:39] // (op 422) convert


v_mov_b64 v[24:25], v[12:13] // (op 422) 
s_lshl_b64 s[38:39], s[2:3], 7 // (op 422) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 422) 

v_mov_b32 v46, s56 // (op 422) 
v_mov_b32 v47, s57 // (op 422) 
v_add_co_u32 v48, s[38:39], v24, v46 // (op 422) least significant half
v_addc_co_u32 v49, s[38:39], v25, v47, s[38:39] // (op 422) most significant half




s_add_u32 s19, 2048, s0 // (op 422) 
s_mov_b32 m0, s19 // (op 422) reg expression
buffer_load_dwordx4 v48, s[40:43], 0 offen offset: 0 lds // (op 422) Load value direct to lds
v_add_co_u32 v48, s[38:39], v48, v18 // (op 422) least significant half
v_addc_co_u32 v49, s[38:39], 0, v49, s[38:39] // (op 422) most significant half

s_add_u32 m0, m0, 2048 // (op 422) 
buffer_load_dwordx4 v48, s[40:43], 0 offen offset: 0 lds // (op 422) Load value direct to lds





v_mov_b32 v18, v20 // (op 17) 
s_lshl_b32 s19, s12, 3 // (op 17) 
v_add_u32 v19, v18, s19 // (op 17) 


buffer_load_ubyte v23, v19, s[48:51], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1133) 



s_waitcnt vmcnt(5)
ds_write_b8 v14, v22 offset:12416 // (op 236) Store local data 
s_barrier  // (op 1124) 



ds_read_b64_tr_b8 v[30:31], v6 offset:6144 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v6 offset:6272 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v6 offset:7168 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v6 offset:7296 // (op 229) Transpose load local data 
s_barrier  // (op 1118) 



ds_read_b64_tr_b8 v[38:39], v5 offset:8192 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v5 offset:8448 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v5 offset:10240 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v5 offset:10496 // (op 227) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1130) 


ds_read_u8 v27, v7 offset:12416 // (op 231) Load local data 
s_barrier  // (op 1174) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v27, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 234) 

v_mov_b64 v[46:47], s[24:25] // (op 233) convert

s_lshl_b32 s19, s24, 2 // (op 233) 
s_mul_i32 s38, 3, s19 // (op 233) 

v_add_u32 v19, v15, s38 // (op 233) 

v_mov_b32 v48, v19 // (op 233) 
buffer_load_ubyte v28, v48, s[52:55], 0 offen offset: 0 // (op 233) Load value




s_barrier  // (op 1127) 
s_lshl_b32 s19, s24, 3 // (op 709) 


v_add_u32 v21, v21, s19 // (op 709) 

s_lshl_b64 s[38:39], s[8:9], 7 // (op 450) 
s_mul_i32 s57, 3, s39 // (op 450) most significant: low of low * high
s_mul_i32 s56, 3, s38 // (op 450) least significant
s_mul_hi_u32 s19, 3, s38 // (op 450) most significant: high of low * low
s_add_u32 s57, s57, s19 // (op 450) most significant: sum


v_mov_b32 v15, s56 // (op 450) 
v_mov_b32 v19, s57 // (op 450) 
v_add_co_u32 v46, s[38:39], v10, v15 // (op 450) least significant half
v_addc_co_u32 v47, s[38:39], v11, v19, s[38:39] // (op 450) most significant half




s_add_u32 s19, 6144, s15 // (op 450) 

s_mov_b32 m0, s19 // (op 450) reg expression
buffer_load_dwordx4 v46, s[44:47], 0 offen offset: 0 lds // (op 450) Load value direct to lds



s_barrier  // (op 1121) 
s_lshl_b64 s[38:39], s[8:9], 7 // (op 581) 

s_lshl_b64 s[56:57], s[38:39], 1 // (op 581) 

v_mov_b32 v15, s56 // (op 581) 
v_mov_b32 v19, s57 // (op 581) 
s_lshl_b64 s[38:39], s[2:3], 6 // (op 445) 
v_mov_b64 v[10:11], s[38:39] // (op 445) convert


s_lshl_b64 s[38:39], s[2:3], 7 // (op 445) 
s_mul_i32 s59, 3, s39 // (op 445) most significant: low of low * high
s_mul_i32 s58, 3, s38 // (op 445) least significant
s_mul_hi_u32 s19, 3, s38 // (op 445) most significant: high of low * low
s_add_u32 s59, s59, s19 // (op 445) most significant: sum


v_mov_b32 v46, s58 // (op 445) 
v_mov_b32 v47, s59 // (op 445) 
v_add_co_u32 v16, s[38:39], v16, v15 // (op 581) least significant half
v_addc_co_u32 v17, s[38:39], v17, v19, s[38:39] // (op 581) most significant half




v_add_co_u32 v48, s[38:39], v24, v46 // (op 445) least significant half
v_addc_co_u32 v49, s[38:39], v25, v47, s[38:39] // (op 445) most significant half




s_add_u32 s19, 8192, s14 // (op 445) 
s_mov_b32 m0, s19 // (op 445) reg expression
buffer_load_dwordx4 v48, s[40:43], 0 offen offset: 0 lds // (op 445) Load value direct to lds
v_add_co_u32 v48, s[38:39], v48, v10 // (op 445) least significant half
v_addc_co_u32 v49, s[38:39], 0, v49, s[38:39] // (op 445) most significant half

s_add_u32 m0, m0, 2048 // (op 445) 
buffer_load_dwordx4 v48, s[40:43], 0 offen offset: 0 lds // (op 445) Load value direct to lds






s_lshl_b32 s19, s12, 2 // (op 238) 
s_mul_i32 s38, 3, s19 // (op 238) 

v_add_u32 v15, v18, s38 // (op 238) 


s_lshl_b64 s[38:39], s[2:3], 7 // (op 517) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 517) 

v_mov_b32 v19, s56 // (op 517) 
v_mov_b32 v10, s57 // (op 517) 
v_add_co_u32 v12, s[38:39], v12, v19 // (op 517) least significant half
v_addc_co_u32 v13, s[38:39], v13, v10, s[38:39] // (op 517) most significant half




buffer_load_ubyte v22, v15, s[48:51], 0 offen offset: 0 // (op 238) Load value


s_lshl_b32 s19, s12, 3 // (op 645) 
s_barrier  // (op 1104) 
v_add_u32 v20, v20, s19 // (op 645) 




s_waitcnt vmcnt(5)
ds_write_b8 v29, v23 offset:12288 // (op 21) Store local data 
s_barrier  // (op 1094) 



ds_read_b64_tr_b8 v[30:31], v3 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v3 offset:128 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:1024 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v3 offset:1152 // (op 70) Transpose load local data 
s_barrier  // (op 1078) 



ds_read_b64_tr_b8 v[38:39], v0 offset:2048 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:2304 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:4096 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:4352 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1111) 


ds_read_u8 v27, v4 offset:12288 // (op 75) Load local data 
s_ashr_i64 s[38:39], s[6:7], 7 // (op 54) 
s_mov_b32 s19, s38 // (op 54) convert

s_and_b32 s38, s19, 4294967294 // (op 54) 

s_cmp_lt_u32 s18, s38 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 if true)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54:













s_barrier  // (op 1142) 
v_and_b32 v29, -2, v1 // (op 480) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 480) 
v_sub_u32 v10, v1, v29 // (op 480) 

s_lshr_b32 s14, s0, 1 // (op 480) 

v_lshlrev_b32 v29, 4, v10 // (op 480) 

v_lshl_add_u32 v10, s16, 5, v29 // (op 480) 

v_mov_b32 v21, 0 // (op 480) convert
v_mov_b32 v20, v10 // (op 480) convert

v_lshrrev_b32 v29, 1, v1 // (op 480) 
v_lshl_add_u32 v10, s14, 8, v29 // (op 480) 


v_mul_lo_u32 v19, v10, s3 // (op 480) most significant: low of low * high
v_mul_hi_u32 v29, v10, s2 // (op 480) most significant: high of low * low
v_mul_lo_u32 v18, v10, s2 // (op 480) least significant: low of low * low
v_add_u32 v19, v19, v29 // (op 480) most significant: sum


v_add_co_u32 v10, s[0:1], v20, v18 // (op 480) least significant half
v_addc_co_u32 v11, s[0:1], v21, v19, s[0:1] // (op 480) most significant half



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

s_ashr_i64 s[0:1], s[6:7], 7 // (op 547) 
s_lshr_b32 s14, s0, 1 // (op 547) 

v_lshl_add_u32 v29, s14, 8, v1 // (op 547) 

s_lshl_b32 s0, s17, 4 // (op 547) 
s_mov_b32 s15, 0 // (op 547) convert
s_mov_b32 s14, s0 // (op 547) convert

v_mul_lo_u32 v13, v29, s9 // (op 547) most significant: low of low * high
v_mul_hi_u32 v14, v29, s8 // (op 547) most significant: high of low * low
v_mul_lo_u32 v12, v29, s8 // (op 547) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 547) most significant: sum

v_mov_b32 v14, s14 // (op 547) 
v_mov_b32 v15, s15 // (op 547) 
v_add_co_u32 v12, s[0:1], v12, v14 // (op 547) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 547) most significant half





s_mul_i32 s1, s8, s7 // (op 549) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 549) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 549) least significant
s_mul_hi_u32 s15, s8, s6 // (op 549) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 549) most significant: sum
s_add_u32 s1, s1, s15 // (op 549) most significant: sum


s_mov_b32 s44, s10 // (op 549) reg expression
s_mov_b32 s45, s11 // (op 549) reg expression
s_mov_b32 s46, s0 // (op 549) reg expression
s_mov_b32 s47, 131072 // (op 549) call()

v_and_b32 v29, -32, v1 // (op 611) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 611) 
v_sub_u32 v14, v1, v29 // (op 611) 

s_lshr_b32 s14, s0, 1 // (op 611) 

v_lshl_add_u32 v29, s16, 5, v14 // (op 611) 

v_lshrrev_b32 v14, 5, v1 // (op 611) 
v_lshl_add_u32 v15, s14, 3, v14 // (op 611) 


v_mul_lo_u32 v14, v15, s12 // (op 611) 

v_add_u32 v15, v29, v14 // (op 611) 


s_mul_i32 s1, s12, s23 // (op 613) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 613) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 613) least significant
s_mul_hi_u32 s15, s12, s22 // (op 613) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 613) most significant: sum
s_add_u32 s1, s1, s15 // (op 613) most significant: sum


s_mov_b32 s48, s20 // (op 613) reg expression
s_mov_b32 s49, s21 // (op 613) reg expression
s_mov_b32 s50, s0 // (op 613) reg expression
s_mov_b32 s51, 131072 // (op 613) call()

v_and_b32 v14, 63, v1 // (op 675) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 675) 
v_lshrrev_b32 v29, 4, v14 // (op 675) 

s_lshr_b32 s14, s0, 1 // (op 675) 

v_and_b32 v14, 15, v1 // (op 675) 
v_lshl_add_u32 v16, s14, 3, v29 // (op 675) 


v_lshl_add_u32 v29, s17, 4, v14 // (op 675) 

v_mul_lo_u32 v14, v16, s24 // (op 675) 
v_add_u32 v14, v14, v29 // (op 675) 


s_mul_i32 s1, s24, s23 // (op 677) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 677) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 677) least significant
s_mul_hi_u32 s15, s24, s22 // (op 677) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 677) most significant: sum
s_add_u32 s1, s1, s15 // (op 677) most significant: sum


s_mov_b32 s52, s26 // (op 677) reg expression
s_mov_b32 s53, s27 // (op 677) reg expression
s_mov_b32 s54, s0 // (op 677) reg expression
s_mov_b32 s55, 131072 // (op 677) call()

v_and_b32 v29, -2, v1 // (op 768) 
v_sub_u32 v16, v1, v29 // (op 768) 

v_lshlrev_b32 v29, 4, v16 // (op 768) 

v_lshrrev_b32 v16, 1, v1 // (op 768) 
v_lshl_add_u32 v17, v16, 5, v29 // (op 768) 


s_nop 0
v_readfirstlane_b32 s0, v17 // (op 768) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 4, v1 // (op 820) 
s_nop 0
v_readfirstlane_b32 s1, v29 // (op 820) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -32, v1 // (op 872) 
v_sub_u32 v16, v1, v29 // (op 872) 

v_and_b32 v29, 4294967264, v1 // (op 872) 
v_add_u32 v17, v29, v16 // (op 872) 


s_ashr_i64 s[14:15], s[6:7], 7 // (op 162) 
s_and_b32 s18, s14, 4294967294 // (op 162) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 161) 
s_mov_b32 s19, s14 // (op 161) convert

s_cmp_lt_u32 s18, s19 // (op 161) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161:
s_barrier  // (op 1075) 
s_add_u32 s18, s18, 1 // (op 163) 

s_lshl_b64 s[14:15], s[2:3], 6 // (op 432) 
v_mov_b64 v[18:19], s[14:15] // (op 432) convert


v_mov_b64 v[20:21], v[10:11] // (op 432) 
v_mov_b64 v[22:23], v[20:21] // (op 432) reg expression
s_add_u32 s19, 2048, s0 // (op 432) 
s_mov_b32 m0, s19 // (op 432) reg expression
buffer_load_dwordx4 v22, s[40:43], 0 offen offset: 0 lds // (op 432) Load value direct to lds
v_add_co_u32 v22, s[14:15], v22, v18 // (op 432) least significant half
v_addc_co_u32 v23, s[14:15], 0, v23, s[14:15] // (op 432) most significant half

s_add_u32 m0, m0, 2048 // (op 432) 
buffer_load_dwordx4 v22, s[40:43], 0 offen offset: 0 lds // (op 432) Load value direct to lds




s_lshl_b64 s[14:15], s[2:3], 7 // (op 470) 
v_mov_b32 v16, s14 // (op 470) 
v_mov_b32 v18, s15 // (op 470) 
v_add_co_u32 v10, s[38:39], v10, v16 // (op 470) least significant half
v_addc_co_u32 v11, s[38:39], v11, v18, s[38:39] // (op 470) most significant half




s_barrier  // (op 1091) 


v_mov_b64 v[18:19], v[12:13] // (op 438) 
v_mov_b64 v[20:21], v[18:19] // (op 438) reg expression

s_mov_b32 m0, s1 // (op 438) reg expression
buffer_load_dwordx4 v20, s[44:47], 0 offen offset: 0 lds // (op 438) Load value direct to lds


s_lshl_b64 s[14:15], s[8:9], 7 // (op 539) 
v_mov_b32 v16, s14 // (op 539) 
v_mov_b32 v18, s15 // (op 539) 
v_add_co_u32 v12, s[38:39], v12, v16 // (op 539) least significant half
v_addc_co_u32 v13, s[38:39], v13, v18, s[38:39] // (op 539) most significant half






v_mov_b32 v16, v15 // (op 180) 
v_mov_b32 v18, v16 // (op 180) reg expression

buffer_load_ubyte v19, v18, s[48:51], 0 offen offset: 0 // (op 180) Load value


s_lshl_b32 s19, s12, 2 // (op 603) 
v_add_u32 v15, v15, s19 // (op 603) 

s_barrier  // (op 1107) 
s_waitcnt vmcnt(2)
s_barrier  // (op 1081) 






ds_read_b64_tr_b8 v[20:21], v0 offset:2048 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[22:23], v0 offset:2304 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:4096 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:4352 // (op 169) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v17, v19 offset:12288 // (op 178) Store local data 

s_barrier  // (op 1097) 



ds_read_b64_tr_b8 v[28:29], v3 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v3 offset:128 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v3 offset:1024 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:1152 // (op 171) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1114) 


ds_read_u8 v16, v4 offset:12288 // (op 173) Load local data 

v_mov_b64 v[18:19], s[24:25] // (op 175) convert

v_mov_b32 v36, v14 // (op 175) 
v_mov_b32 v37, v36 // (op 175) reg expression
v_mov_b32 v38, v37 // (op 175) 
buffer_load_ubyte v39, v38, s[52:55], 0 offen offset: 0 // (op 175) Load value




s_lshl_b32 s19, s24, 2 // (op 667) 
v_add_u32 v14, v14, s19 // (op 667) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[28:35], a[0:3], v16, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 176) 




s_ashr_i64 s[14:15], s[6:7], 7 // (op 161) 
s_mov_b32 s19, s14 // (op 161) convert

s_cmp_lt_u32 s18, s19 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 if true)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161:






















s_barrier  // (op 1166) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

v_lshrrev_b32 v0, 6, v1 // (op 914) 
v_and_b32 v3, 63, v1 // (op 914) 
v_lshrrev_b32 v4, 4, v3 // (op 914) 
v_lshlrev_b32 v5, 2, v4 // (op 914) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 914) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 914) 

v_lshrrev_b32 v4, 2, v3 // (op 914) 

v_and_b32 v3, 3, v4 // (op 914) 

v_and_b32 v4, 3, v1 // (op 914) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 914) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 914) 

v_mul_lo_u32 v4, v3, s28 // (op 914) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 914) 


s_mul_i32 s1, s28, s33 // (op 916) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 916) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 916) least significant
s_mul_hi_u32 s3, s28, s32 // (op 916) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 916) most significant: sum
s_add_u32 s1, s1, s3 // (op 916) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 916) 

s_mov_b32 s4, s30 // (op 916) reg expression
s_mov_b32 s5, s31 // (op 916) reg expression
s_mov_b32 s6, s2 // (op 916) reg expression
s_mov_b32 s7, 131072 // (op 916) call()




buffer_store_dwordx4 v[10:13], v3, s[4:7], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123:
v_lshrrev_b32 v0, 6, v1 // (op 738) 
v_and_b32 v3, 63, v1 // (op 738) 
v_lshrrev_b32 v4, 4, v3 // (op 738) 
v_lshlrev_b32 v5, 2, v4 // (op 738) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 738) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 738) 

v_lshrrev_b32 v4, 2, v3 // (op 738) 

v_and_b32 v3, 3, v4 // (op 738) 

v_and_b32 v4, 3, v1 // (op 738) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 738) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 738) 

v_mul_lo_u32 v4, v3, s34 // (op 738) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 738) 


s_mul_i32 s1, s34, s33 // (op 740) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 740) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 740) least significant
s_mul_hi_u32 s3, s34, s32 // (op 740) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 740) most significant: sum
s_add_u32 s1, s1, s3 // (op 740) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 740) 

s_mov_b32 s8, s36 // (op 740) reg expression
s_mov_b32 s9, s37 // (op 740) reg expression
s_mov_b32 s10, s2 // (op 740) reg expression
s_mov_b32 s11, 131072 // (op 740) call()





buffer_load_dwordx4 v[4:7], v3, s[8:11], 0 offen // (op 135) Load value



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


v_lshrrev_b32 v0, 6, v1 // (op 931) 
v_and_b32 v3, 63, v1 // (op 931) 
v_lshrrev_b32 v10, 4, v3 // (op 931) 
v_lshlrev_b32 v11, 2, v10 // (op 931) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 931) 


v_lshl_add_u32 v0, s16, 5, v10 // (op 931) 

v_lshrrev_b32 v10, 2, v3 // (op 931) 

v_and_b32 v3, 3, v10 // (op 931) 

v_and_b32 v10, 3, v1 // (op 931) 
v_lshl_add_u32 v11, v3, 2, v10 // (op 931) 


v_lshl_add_u32 v3, s17, 4, v11 // (op 931) 

v_mul_lo_u32 v10, v3, s28 // (op 931) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 931) 


s_mul_i32 s1, s28, s33 // (op 933) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 933) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 933) least significant
s_mul_hi_u32 s3, s28, s32 // (op 933) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 933) most significant: sum
s_add_u32 s1, s1, s3 // (op 933) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 933) 

s_mov_b32 s4, s30 // (op 933) reg expression
s_mov_b32 s5, s31 // (op 933) reg expression
s_mov_b32 s6, s2 // (op 933) reg expression
s_mov_b32 s7, 131072 // (op 933) call()




buffer_store_dwordx4 v[4:7], v3, s[4:7], 0 offen // (op 134) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123:



s_endpgm  // End of RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.LRR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end:
.size RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2, .LRR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end-RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: B8
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
  - .name: RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
    .symbol: RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 12544
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
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

