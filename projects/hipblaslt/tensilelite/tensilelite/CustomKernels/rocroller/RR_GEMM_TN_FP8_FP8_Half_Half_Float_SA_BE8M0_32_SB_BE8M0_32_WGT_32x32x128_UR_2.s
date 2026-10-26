

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2,@function
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 763) 
v_and_b32 v3, 15, v1 // (op 763) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 763) 


v_and_b32 v0, 48, v1 // (op 763) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 763) 


v_lshrrev_b32 v0, 6, v1 // (op 815) 
v_and_b32 v4, 1, v0 // (op 815) 

v_and_b32 v0, 15, v1 // (op 815) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 815) 


v_and_b32 v0, 48, v1 // (op 815) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 815) 


v_lshrrev_b32 v0, 7, v1 // (op 888) 
v_and_b32 v5, 15, v1 // (op 888) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 888) 


v_and_b32 v0, 48, v1 // (op 888) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 888) 


v_lshrrev_b32 v0, 6, v1 // (op 927) 
v_and_b32 v6, 1, v0 // (op 927) 

v_and_b32 v0, 15, v1 // (op 927) 
v_lshl_add_u32 v7, v6, 4, v0 // (op 927) 


v_and_b32 v0, 48, v1 // (op 927) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 927) 


v_mov_b32 v0, s14 // (op 2) Move value
v_mov_b32 v7, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
v_lshrrev_b32 v8, 3, v1 // (op 474) 
v_lshl_add_u32 v9, s16, 5, v8 // (op 474) 

v_and_b32 v8, -8, v1 // (op 474) 
v_sub_u32 v10, v1, v8 // (op 474) 

v_lshlrev_b32 v8, 4, v10 // (op 474) 

v_mov_b32 v11, 0 // (op 474) convert
v_mov_b32 v10, v8 // (op 474) convert

v_mul_lo_u32 v13, v9, s3 // (op 474) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 474) most significant: high of low * low
v_mul_lo_u32 v12, v9, s2 // (op 474) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 474) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v10 // (op 474) least significant half
v_addc_co_u32 v13, s[0:1], v13, v11, s[0:1] // (op 474) most significant half



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

v_lshrrev_b32 v8, 3, v1 // (op 493) 
v_lshl_add_u32 v9, s16, 5, v8 // (op 493) 

v_and_b32 v8, -8, v1 // (op 493) 
v_sub_u32 v10, v1, v8 // (op 493) 

v_lshlrev_b32 v8, 4, v10 // (op 493) 

v_mov_b32 v11, 0 // (op 493) convert
v_mov_b32 v10, v8 // (op 493) convert

v_mul_lo_u32 v15, v9, s3 // (op 493) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 493) most significant: high of low * low
v_mul_lo_u32 v14, v9, s2 // (op 493) least significant: low of low * low
v_add_u32 v15, v15, v8 // (op 493) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v10 // (op 493) least significant half
v_addc_co_u32 v15, s[0:1], v15, v11, s[0:1] // (op 493) most significant half



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

v_and_b32 v8, -8, v1 // (op 538) 
v_sub_u32 v9, v1, v8 // (op 538) 

v_lshlrev_b32 v8, 4, v9 // (op 538) 

v_mov_b32 v11, 0 // (op 538) convert
v_mov_b32 v10, v8 // (op 538) convert

v_lshrrev_b32 v8, 3, v1 // (op 538) 
v_lshl_add_u32 v9, s17, 5, v8 // (op 538) 

v_mul_lo_u32 v17, v9, s9 // (op 538) most significant: low of low * high
v_mul_hi_u32 v8, v9, s8 // (op 538) most significant: high of low * low
v_mul_lo_u32 v16, v9, s8 // (op 538) least significant: low of low * low
v_add_u32 v17, v17, v8 // (op 538) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v16 // (op 538) least significant half
v_addc_co_u32 v9, s[0:1], v11, v17, s[0:1] // (op 538) most significant half



s_mul_i32 s1, s8, s13 // (op 540) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 540) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 540) least significant
s_mul_hi_u32 s15, s8, s12 // (op 540) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 540) most significant: sum
s_add_u32 s1, s1, s15 // (op 540) most significant: sum


s_mov_b32 s44, s10 // (op 540) reg expression
s_mov_b32 s45, s11 // (op 540) reg expression
s_mov_b32 s46, s0 // (op 540) reg expression
s_mov_b32 s47, 131072 // (op 540) call()

v_and_b32 v10, -8, v1 // (op 557) 
v_sub_u32 v11, v1, v10 // (op 557) 

v_lshlrev_b32 v10, 4, v11 // (op 557) 

v_mov_b32 v17, 0 // (op 557) convert
v_mov_b32 v16, v10 // (op 557) convert

v_lshrrev_b32 v10, 3, v1 // (op 557) 
v_lshl_add_u32 v11, s17, 5, v10 // (op 557) 

v_mul_lo_u32 v19, v11, s9 // (op 557) most significant: low of low * high
v_mul_hi_u32 v10, v11, s8 // (op 557) most significant: high of low * low
v_mul_lo_u32 v18, v11, s8 // (op 557) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 557) most significant: sum


v_add_co_u32 v10, s[0:1], v16, v18 // (op 557) least significant half
v_addc_co_u32 v11, s[0:1], v17, v19, s[0:1] // (op 557) most significant half



s_mul_i32 s1, s8, s13 // (op 559) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 559) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 559) least significant
s_mul_hi_u32 s15, s8, s12 // (op 559) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 559) most significant: sum
s_add_u32 s1, s1, s15 // (op 559) most significant: sum


s_mov_b32 s44, s10 // (op 559) reg expression
s_mov_b32 s45, s11 // (op 559) reg expression
s_mov_b32 s46, s0 // (op 559) reg expression
s_mov_b32 s47, 131072 // (op 559) call()

v_lshrrev_b32 v16, 7, v1 // (op 602) 
v_and_b32 v17, 15, v1 // (op 602) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 602) 


v_lshl_add_u32 v16, s16, 5, v18 // (op 602) 

v_and_b32 v17, 63, v1 // (op 602) 
v_lshrrev_b32 v18, 4, v17 // (op 602) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v17, v16, s20 // (op 602) 
v_add_u32 v17, v17, v18 // (op 602) 


s_mul_i32 s1, s20, s7 // (op 604) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 604) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 604) least significant
s_mul_hi_u32 s15, s20, s6 // (op 604) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 604) most significant: sum
s_add_u32 s1, s1, s15 // (op 604) most significant: sum


s_mov_b32 s48, s22 // (op 604) reg expression
s_mov_b32 s49, s23 // (op 604) reg expression
s_mov_b32 s50, s0 // (op 604) reg expression
s_mov_b32 s51, 131072 // (op 604) call()

v_lshrrev_b32 v16, 7, v1 // (op 621) 
v_and_b32 v18, 15, v1 // (op 621) 
v_lshl_add_u32 v19, v16, 4, v18 // (op 621) 


v_lshl_add_u32 v16, s16, 5, v19 // (op 621) 

v_and_b32 v18, 63, v1 // (op 621) 
v_lshrrev_b32 v19, 4, v18 // (op 621) 

v_mul_lo_u32 v18, v16, s20 // (op 621) 
v_add_u32 v18, v18, v19 // (op 621) 


s_mul_i32 s1, s20, s7 // (op 623) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 623) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 623) least significant
s_mul_hi_u32 s15, s20, s6 // (op 623) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 623) most significant: sum
s_add_u32 s1, s1, s15 // (op 623) most significant: sum


s_mov_b32 s48, s22 // (op 623) reg expression
s_mov_b32 s49, s23 // (op 623) reg expression
s_mov_b32 s50, s0 // (op 623) reg expression
s_mov_b32 s51, 131072 // (op 623) call()

v_lshrrev_b32 v16, 6, v1 // (op 666) 
v_and_b32 v19, 1, v16 // (op 666) 

v_and_b32 v16, 15, v1 // (op 666) 
v_lshl_add_u32 v20, v19, 4, v16 // (op 666) 


v_lshl_add_u32 v16, s17, 5, v20 // (op 666) 

v_mul_lo_u32 v19, v16, s24 // (op 666) 

v_and_b32 v16, 63, v1 // (op 666) 
v_lshrrev_b32 v20, 4, v16 // (op 666) 

v_add_u32 v16, v20, v19 // (op 666) 


s_mul_i32 s1, s24, s13 // (op 668) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 668) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 668) least significant
s_mul_hi_u32 s15, s24, s12 // (op 668) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 668) most significant: sum
s_add_u32 s1, s1, s15 // (op 668) most significant: sum


s_mov_b32 s52, s26 // (op 668) reg expression
s_mov_b32 s53, s27 // (op 668) reg expression
s_mov_b32 s54, s0 // (op 668) reg expression
s_mov_b32 s55, 131072 // (op 668) call()

v_lshrrev_b32 v19, 6, v1 // (op 685) 
v_and_b32 v20, 1, v19 // (op 685) 

v_and_b32 v19, 15, v1 // (op 685) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 685) 


v_lshl_add_u32 v19, s17, 5, v21 // (op 685) 

v_mul_lo_u32 v20, v19, s24 // (op 685) 

v_and_b32 v19, 63, v1 // (op 685) 
v_lshrrev_b32 v21, 4, v19 // (op 685) 

v_add_u32 v19, v21, v20 // (op 685) 


s_mul_i32 s1, s24, s13 // (op 687) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 687) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 687) least significant
s_mul_hi_u32 s15, s24, s12 // (op 687) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 687) most significant: sum
s_add_u32 s1, s1, s15 // (op 687) most significant: sum


s_mov_b32 s52, s26 // (op 687) reg expression
s_mov_b32 s53, s27 // (op 687) reg expression
s_mov_b32 s54, s0 // (op 687) reg expression
s_mov_b32 s55, 131072 // (op 687) call()




buffer_load_ubyte v20, v17, s[48:51], 0 offen offset: 0 // (op 312) Load value



buffer_load_ubyte v21, v16, s[52:55], 0 offen offset: 0 // (op 317) Load value
v_and_b32 v22, -8, v1 // (op 803) 
v_sub_u32 v23, v1, v22 // (op 803) 

v_lshlrev_b32 v22, 4, v23 // (op 803) 

v_lshrrev_b32 v23, 3, v1 // (op 803) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 803) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 803) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 423) reg expression
buffer_load_dwordx4 v8, s[44:47], 0 offen offset: 0 lds // (op 423) Load value direct to lds

v_and_b32 v22, -8, v1 // (op 751) 
v_sub_u32 v23, v1, v22 // (op 751) 

v_lshlrev_b32 v22, 4, v23 // (op 751) 

v_lshrrev_b32 v23, 3, v1 // (op 751) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 751) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 751) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 428) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 428) reg expression
buffer_load_dwordx4 v12, s[40:43], 0 offen offset: 0 lds // (op 428) Load value direct to lds





buffer_load_ubyte v22, v17, s[48:51], 0 offen offset:4 // (op 328) Load value




buffer_load_ubyte v17, v16, s[52:55], 0 offen offset:4 // (op 333) Load value

v_and_b32 v16, -8, v1 // (op 915) 
v_sub_u32 v23, v1, v16 // (op 915) 

v_lshlrev_b32 v16, 4, v23 // (op 915) 

v_lshrrev_b32 v23, 3, v1 // (op 915) 
v_lshl_add_u32 v24, v23, 7, v16 // (op 915) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 915) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 8192, s0 // (op 433) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 433) reg expression
s_mov_b32 s14, 128 // (op 433) call()
buffer_load_dwordx4 v8, s[44:47], s14 offen offset: 0 lds // (op 433) Load value direct to lds




v_and_b32 v16, -8, v1 // (op 876) 
v_sub_u32 v8, v1, v16 // (op 876) 

v_lshlrev_b32 v9, 4, v8 // (op 876) 

v_lshrrev_b32 v8, 3, v1 // (op 876) 
v_lshl_add_u32 v16, v8, 7, v9 // (op 876) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 876) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 12288, s0 // (op 435) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 435) reg expression
s_mov_b32 s14, 128 // (op 435) call()
buffer_load_dwordx4 v12, s[40:43], s14 offen offset: 0 lds // (op 435) Load value direct to lds




s_waitcnt vmcnt(5)
s_barrier  // (op 972) 



ds_read_b128 v[24:27], v4 // (op 358) Load local data 
ds_read_b128 v[28:31], v4 offset:64 // (op 358) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 956) 



ds_read_b128 v[32:35], v3 offset:4096 // (op 355) Load local data 
ds_read_b128 v[36:39], v3 offset:4160 // (op 355) Load local data 
v_and_b32 v16, -8, v1 // (op 723) 
v_sub_u32 v23, v1, v16 // (op 723) 

v_lshlrev_b32 v16, 4, v23 // (op 723) 

v_lshrrev_b32 v23, 3, v1 // (op 723) 
v_lshl_add_u32 v8, v23, 7, v16 // (op 723) 


s_nop 0
v_readfirstlane_b32 s0, v8 // (op 723) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 775) 
v_sub_u32 v23, v1, v16 // (op 775) 

v_lshlrev_b32 v16, 4, v23 // (op 775) 

v_lshrrev_b32 v23, 3, v1 // (op 775) 
v_lshl_add_u32 v8, v23, 7, v16 // (op 775) 


s_nop 0
v_readfirstlane_b32 s1, v8 // (op 775) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 861) 
v_sub_u32 v23, v1, v16 // (op 861) 

v_lshlrev_b32 v16, 4, v23 // (op 861) 

v_lshrrev_b32 v23, 3, v1 // (op 861) 
v_lshl_add_u32 v8, v23, 7, v16 // (op 861) 


s_nop 0
v_readfirstlane_b32 s14, v8 // (op 861) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 900) 
v_sub_u32 v23, v1, v16 // (op 900) 

v_lshlrev_b32 v16, 4, v23 // (op 900) 

v_lshrrev_b32 v23, 3, v1 // (op 900) 
v_lshl_add_u32 v8, v23, 7, v16 // (op 900) 


s_nop 0
v_readfirstlane_b32 s15, v8 // (op 900) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s18, 0 // (op 53) call()
s_ashr_i64 s[38:39], s[28:29], 7 // (op 52) 
s_mov_b32 s19, s38 // (op 52) convert

s_and_b32 s38, s19, 4294967294 // (op 52) 

s_cmp_lt_u32 s18, s38 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52 // (op 52) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52 if false)
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52:
s_add_u32 s18, s18, 2 // (op 54) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[24:31], a[0:3], v20, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 78) 


v_mov_b32 v16, v18 // (op 72) 

buffer_load_ubyte v20, v16, s[48:51], 0 offen offset:8 // (op 72) Load value


v_mov_b32 v23, v19 // (op 76) 

buffer_load_ubyte v21, v23, s[52:55], 0 offen offset:8 // (op 76) Load value
s_barrier  // (op 975) 


v_mov_b64 v[8:9], v[10:11] // (op 396) 

s_mov_b32 m0, s1 // (op 396) reg expression
s_mov_b32 s19, 256 // (op 396) call()
buffer_load_dwordx4 v8, s[44:47], s19 offen offset: 0 lds // (op 396) Load value direct to lds

s_barrier  // (op 959) 


v_mov_b64 v[12:13], v[14:15] // (op 391) 
s_add_u32 s19, 4096, s0 // (op 391) 

s_mov_b32 m0, s19 // (op 391) reg expression
s_mov_b32 s38, 256 // (op 391) call()
buffer_load_dwordx4 v12, s[40:43], s38 offen offset: 0 lds // (op 391) Load value direct to lds


s_waitcnt vmcnt(5)
s_barrier  // (op 994) 



ds_read_b128 v[24:27], v6 offset:8192 // (op 216) Load local data 
ds_read_b128 v[28:31], v6 offset:8256 // (op 216) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 988) 



ds_read_b128 v[32:35], v5 offset:12288 // (op 214) Load local data 
ds_read_b128 v[36:39], v5 offset:12352 // (op 214) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[24:31], a[0:3], v22, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 221) 



buffer_load_ubyte v22, v16, s[48:51], 0 offen offset:12 // (op 218) Load value




v_add_u32 v18, 8, v18 // (op 613) 
buffer_load_ubyte v17, v23, s[52:55], 0 offen offset:12 // (op 220) Load value

s_barrier  // (op 997) 
v_add_u32 v19, 8, v19 // (op 677) 


s_add_u32 s19, 8192, s15 // (op 418) 

s_mov_b32 m0, s19 // (op 418) reg expression
s_mov_b32 s38, 384 // (op 418) call()
buffer_load_dwordx4 v8, s[44:47], s38 offen offset: 0 lds // (op 418) Load value direct to lds



s_barrier  // (op 991) 
v_add_co_u32 v10, vcc, 256, v10 // (op 549) least significant half


v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 549) most significant half
s_add_u32 s19, 12288, s14 // (op 413) 

s_mov_b32 m0, s19 // (op 413) reg expression
s_mov_b32 s38, 384 // (op 413) call()
buffer_load_dwordx4 v12, s[40:43], s38 offen offset: 0 lds // (op 413) Load value direct to lds



v_add_co_u32 v14, vcc, 256, v14 // (op 485) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 485) most significant half
s_waitcnt vmcnt(5)
s_barrier  // (op 981) 



ds_read_b128 v[24:27], v4 // (op 67) Load local data 
ds_read_b128 v[28:31], v4 offset:64 // (op 67) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 965) 



ds_read_b128 v[32:35], v3 offset:4096 // (op 62) Load local data 
ds_read_b128 v[36:39], v3 offset:4160 // (op 62) Load local data 
s_ashr_i64 s[38:39], s[28:29], 7 // (op 52) 
s_mov_b32 s19, s38 // (op 52) convert

s_and_b32 s38, s19, 4294967294 // (op 52) 

s_cmp_lt_u32 s18, s38 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52 // (op 52) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52 if true)
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52:











s_barrier  // (op 1006) 
v_lshrrev_b32 v23, 3, v1 // (op 448) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 448) 
v_lshl_add_u32 v8, s16, 5, v23 // (op 448) 

s_lshr_b32 s14, s0, 1 // (op 448) 

v_and_b32 v23, -8, v1 // (op 448) 
v_sub_u32 v9, v1, v23 // (op 448) 

v_lshlrev_b32 v23, 4, v9 // (op 448) 

v_lshl_add_u32 v9, s14, 8, v23 // (op 448) 


v_mov_b32 v11, 0 // (op 448) convert
v_mov_b32 v10, v9 // (op 448) convert

v_mul_lo_u32 v13, v8, s3 // (op 448) most significant: low of low * high
v_mul_hi_u32 v9, v8, s2 // (op 448) most significant: high of low * low
v_mul_lo_u32 v12, v8, s2 // (op 448) least significant: low of low * low
v_add_u32 v13, v13, v9 // (op 448) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v10 // (op 448) least significant half
v_addc_co_u32 v13, s[0:1], v13, v11, s[0:1] // (op 448) most significant half



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

v_and_b32 v23, -8, v1 // (op 515) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 515) 
v_sub_u32 v8, v1, v23 // (op 515) 

s_lshr_b32 s14, s0, 1 // (op 515) 

v_lshlrev_b32 v23, 4, v8 // (op 515) 

v_lshl_add_u32 v8, s14, 8, v23 // (op 515) 


v_mov_b32 v11, 0 // (op 515) convert
v_mov_b32 v10, v8 // (op 515) convert

v_lshrrev_b32 v23, 3, v1 // (op 515) 
v_lshl_add_u32 v8, s17, 5, v23 // (op 515) 

v_mul_lo_u32 v15, v8, s9 // (op 515) most significant: low of low * high
v_mul_hi_u32 v9, v8, s8 // (op 515) most significant: high of low * low
v_mul_lo_u32 v14, v8, s8 // (op 515) least significant: low of low * low
v_add_u32 v15, v15, v9 // (op 515) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v14 // (op 515) least significant half
v_addc_co_u32 v9, s[0:1], v11, v15, s[0:1] // (op 515) most significant half



s_mul_i32 s1, s8, s13 // (op 517) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 517) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 517) least significant
s_mul_hi_u32 s15, s8, s12 // (op 517) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 517) most significant: sum
s_add_u32 s1, s1, s15 // (op 517) most significant: sum


s_mov_b32 s44, s10 // (op 517) reg expression
s_mov_b32 s45, s11 // (op 517) reg expression
s_mov_b32 s46, s0 // (op 517) reg expression
s_mov_b32 s47, 131072 // (op 517) call()

v_lshrrev_b32 v23, 7, v1 // (op 579) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 579) 
v_and_b32 v10, 15, v1 // (op 579) 
s_lshr_b32 s14, s0, 1 // (op 579) 

v_lshl_add_u32 v11, v23, 4, v10 // (op 579) 


v_lshl_add_u32 v10, s16, 5, v11 // (op 579) 

v_and_b32 v11, 63, v1 // (op 579) 
v_lshrrev_b32 v23, 4, v11 // (op 579) 

v_lshl_add_u32 v11, s14, 3, v23 // (op 579) 


v_mul_lo_u32 v23, v10, s20 // (op 579) 
v_add_u32 v23, v23, v11 // (op 579) 


s_mul_i32 s1, s20, s7 // (op 581) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 581) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 581) least significant
s_mul_hi_u32 s15, s20, s6 // (op 581) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 581) most significant: sum
s_add_u32 s1, s1, s15 // (op 581) most significant: sum


s_mov_b32 s48, s22 // (op 581) reg expression
s_mov_b32 s49, s23 // (op 581) reg expression
s_mov_b32 s50, s0 // (op 581) reg expression
s_mov_b32 s51, 131072 // (op 581) call()

v_and_b32 v10, 63, v1 // (op 643) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 643) 
v_lshrrev_b32 v11, 4, v10 // (op 643) 

s_lshr_b32 s14, s0, 1 // (op 643) 

v_lshrrev_b32 v10, 6, v1 // (op 643) 
v_lshl_add_u32 v14, s14, 3, v11 // (op 643) 


v_and_b32 v11, 1, v10 // (op 643) 

v_and_b32 v10, 15, v1 // (op 643) 
v_lshl_add_u32 v15, v11, 4, v10 // (op 643) 


v_lshl_add_u32 v16, s17, 5, v15 // (op 643) 

v_mul_lo_u32 v15, v16, s24 // (op 643) 

v_add_u32 v16, v14, v15 // (op 643) 


s_mul_i32 s1, s24, s13 // (op 645) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 645) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 645) least significant
s_mul_hi_u32 s15, s24, s12 // (op 645) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 645) most significant: sum
s_add_u32 s1, s1, s15 // (op 645) most significant: sum


s_mov_b32 s52, s26 // (op 645) reg expression
s_mov_b32 s53, s27 // (op 645) reg expression
s_mov_b32 s54, s0 // (op 645) reg expression
s_mov_b32 s55, 131072 // (op 645) call()

v_and_b32 v10, -8, v1 // (op 736) 
v_sub_u32 v11, v1, v10 // (op 736) 

v_lshlrev_b32 v10, 4, v11 // (op 736) 

v_lshrrev_b32 v11, 3, v1 // (op 736) 
v_lshl_add_u32 v14, v11, 7, v10 // (op 736) 


s_nop 0
v_readfirstlane_b32 s0, v14 // (op 736) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -8, v1 // (op 788) 
v_sub_u32 v11, v1, v10 // (op 788) 

v_lshlrev_b32 v10, 4, v11 // (op 788) 

v_lshrrev_b32 v11, 3, v1 // (op 788) 
v_lshl_add_u32 v14, v11, 7, v10 // (op 788) 


s_nop 0
v_readfirstlane_b32 s1, v14 // (op 788) Wait state hazard: VALU Write Readlane Hazard

s_ashr_i64 s[14:15], s[28:29], 7 // (op 157) 
s_and_b32 s18, s14, 4294967294 // (op 157) 

s_ashr_i64 s[14:15], s[28:29], 7 // (op 156) 
s_mov_b32 s19, s14 // (op 156) convert

s_cmp_lt_u32 s18, s19 // (op 156) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156 // (op 156) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156:
s_barrier  // (op 962) 
s_add_u32 s18, s18, 1 // (op 158) 


v_mov_b64 v[10:11], v[12:13] // (op 401) 
s_add_u32 s19, 4096, s0 // (op 401) 

s_mov_b32 m0, s19 // (op 401) reg expression
buffer_load_dwordx4 v10, s[40:43], 0 offen offset: 0 lds // (op 401) Load value direct to lds


v_add_co_u32 v12, vcc, 128, v12 // (op 438) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 438) most significant half
s_barrier  // (op 978) 


v_mov_b64 v[10:11], v[8:9] // (op 407) 

s_mov_b32 m0, s1 // (op 407) reg expression
buffer_load_dwordx4 v10, s[44:47], 0 offen offset: 0 lds // (op 407) Load value direct to lds

v_add_co_u32 v8, vcc, 128, v8 // (op 507) least significant half
v_addc_co_u32 v9, vcc, 0, v9, vcc // (op 507) most significant half
s_waitcnt vmcnt(1)
s_barrier  // (op 968) 



ds_read_b128 v[24:27], v3 offset:4096 // (op 164) Load local data 
ds_read_b128 v[28:31], v3 offset:4160 // (op 164) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 984) 



ds_read_b128 v[32:35], v4 // (op 166) Load local data 
ds_read_b128 v[36:39], v4 offset:64 // (op 166) Load local data 


v_mov_b32 v10, v23 // (op 168) 

buffer_load_ubyte v11, v10, s[48:51], 0 offen offset: 0 // (op 168) Load value

v_add_u32 v23, 4, v23 // (op 571) 


v_mov_b32 v10, v16 // (op 170) 

buffer_load_ubyte v14, v10, s[52:55], 0 offen offset: 0 // (op 170) Load value

v_add_u32 v16, 4, v16 // (op 635) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[32:39], a[0:3], v11, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 171) 




s_ashr_i64 s[14:15], s[28:29], 7 // (op 156) 
s_mov_b32 s19, s14 // (op 156) convert

s_cmp_lt_u32 s18, s19 // (op 156) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156 // (op 156) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156 if true)
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156:





















s_barrier  // (op 1030) 




v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116
s_nop 0
v_accvgpr_read v8, a0 // (op 42) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 42) 
v_accvgpr_read v10, a2 // (op 42) 
v_accvgpr_read v11, a3 // (op 42) 
v_mul_f32 v12, v7, v8 // (op 42) 
v_mul_f32 v13, v7, v9 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 

v_cvt_f16_f32 v8, v12 // (op 48) 
v_cvt_f16_f32 v9, v13 // (op 48) 
v_cvt_f16_f32 v10, v14 // (op 48) 
v_cvt_f16_f32 v11, v15 // (op 48) 

v_lshrrev_b32 v3, 7, v1 // (op 829) 
v_and_b32 v4, 63, v1 // (op 829) 
v_lshrrev_b32 v5, 4, v4 // (op 829) 
v_lshlrev_b32 v6, 2, v5 // (op 829) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 829) 


v_lshl_add_u32 v3, s16, 5, v5 // (op 829) 

v_lshrrev_b32 v5, 2, v4 // (op 829) 

v_and_b32 v4, 3, v5 // (op 829) 

v_and_b32 v5, 3, v1 // (op 829) 
v_lshl_add_u32 v6, v4, 2, v5 // (op 829) 


v_lshrrev_b32 v4, 6, v1 // (op 829) 
v_and_b32 v5, 1, v4 // (op 829) 

v_lshl_add_u32 v4, v5, 4, v6 // (op 829) 


v_lshl_add_u32 v5, s17, 5, v4 // (op 829) 

v_mul_lo_u32 v4, v5, s30 // (op 829) 

v_add_lshl_u32 v5, v3, v4, 1 // (op 829) 


s_mul_i32 s15, s30, s13 // (op 831) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 831) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 831) least significant
s_mul_hi_u32 s1, s30, s12 // (op 831) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 831) most significant: sum
s_add_u32 s15, s15, s1 // (op 831) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 831) 

s_mov_b32 s8, s32 // (op 831) reg expression
s_mov_b32 s9, s33 // (op 831) reg expression
s_mov_b32 s10, s0 // (op 831) reg expression
s_mov_b32 s11, 131072 // (op 831) call()


v_pack_b32_f16 v12, v8, v9 // (op 50) 
v_pack_b32_f16 v13, v10, v11 // (op 50) 


buffer_store_dwordx2 v[12:13], v5, s[8:11], 0 offen // (op 50) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116:
v_lshrrev_b32 v3, 7, v1 // (op 706) 
v_and_b32 v4, 63, v1 // (op 706) 
v_lshrrev_b32 v5, 4, v4 // (op 706) 
v_lshlrev_b32 v6, 2, v5 // (op 706) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 706) 


v_lshl_add_u32 v3, s16, 5, v5 // (op 706) 

v_lshrrev_b32 v5, 2, v4 // (op 706) 

v_and_b32 v4, 3, v5 // (op 706) 

v_and_b32 v5, 3, v1 // (op 706) 
v_lshl_add_u32 v6, v4, 2, v5 // (op 706) 


v_lshrrev_b32 v4, 6, v1 // (op 706) 
v_and_b32 v5, 1, v4 // (op 706) 

v_lshl_add_u32 v4, v5, 4, v6 // (op 706) 


v_lshl_add_u32 v5, s17, 5, v4 // (op 706) 

v_mul_lo_u32 v4, v5, s34 // (op 706) 

v_add_lshl_u32 v5, v3, v4, 1 // (op 706) 


s_mul_i32 s15, s34, s13 // (op 708) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 708) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 708) least significant
s_mul_hi_u32 s1, s34, s12 // (op 708) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 708) most significant: sum
s_add_u32 s15, s15, s1 // (op 708) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 708) 

s_mov_b32 s4, s36 // (op 708) reg expression
s_mov_b32 s5, s37 // (op 708) reg expression
s_mov_b32 s6, s0 // (op 708) reg expression
s_mov_b32 s7, 131072 // (op 708) call()





buffer_load_dwordx2 v[8:9], v5, s[4:7], 0 offen // (op 129) Load value



v_accvgpr_read v10, a0 // (op 125) 
v_accvgpr_read v11, a1 // (op 125) 
v_accvgpr_read v12, a2 // (op 125) 
v_accvgpr_read v13, a3 // (op 125) 
v_mul_f32 v14, v7, v10 // (op 125) 
v_mul_f32 v15, v7, v11 // (op 125) 
v_mul_f32 v16, v7, v12 // (op 125) 
v_mul_f32 v17, v7, v13 // (op 125) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v8 // (op 130) 
v_lshrrev_b32 v11, 16, v8 // (op 130) 
v_cvt_f32_f16 v11, v11 // (op 130) 
v_mul_f32 v18, v0, v10 // (op 130) 
v_mul_f32 v19, v0, v11 // (op 130) 
v_cvt_f32_f16 v10, v9 // (op 130) 
v_lshrrev_b32 v11, 16, v9 // (op 130) 
v_cvt_f32_f16 v11, v11 // (op 130) 
v_mul_f32 v20, v0, v10 // (op 130) 
v_mul_f32 v21, v0, v11 // (op 130) 


v_add_f32 v8, v18, v14 // (op 126) 
v_add_f32 v9, v19, v15 // (op 126) 
v_add_f32 v10, v20, v16 // (op 126) 
v_add_f32 v11, v21, v17 // (op 126) 


v_cvt_f16_f32 v12, v8 // (op 127) 
v_cvt_f16_f32 v13, v9 // (op 127) 
v_cvt_f16_f32 v14, v10 // (op 127) 
v_cvt_f16_f32 v15, v11 // (op 127) 

v_lshrrev_b32 v3, 7, v1 // (op 846) 
v_and_b32 v4, 63, v1 // (op 846) 
v_lshrrev_b32 v5, 4, v4 // (op 846) 
v_lshlrev_b32 v6, 2, v5 // (op 846) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 846) 


v_lshl_add_u32 v3, s16, 5, v5 // (op 846) 

v_lshrrev_b32 v5, 2, v4 // (op 846) 

v_and_b32 v4, 3, v5 // (op 846) 

v_and_b32 v5, 3, v1 // (op 846) 
v_lshl_add_u32 v6, v4, 2, v5 // (op 846) 


v_lshrrev_b32 v4, 6, v1 // (op 846) 
v_and_b32 v5, 1, v4 // (op 846) 

v_lshl_add_u32 v4, v5, 4, v6 // (op 846) 


v_lshl_add_u32 v5, s17, 5, v4 // (op 846) 

v_mul_lo_u32 v4, v5, s30 // (op 846) 

v_add_lshl_u32 v5, v3, v4, 1 // (op 846) 


s_mul_i32 s15, s30, s13 // (op 848) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 848) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 848) least significant
s_mul_hi_u32 s1, s30, s12 // (op 848) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 848) most significant: sum
s_add_u32 s15, s15, s1 // (op 848) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 848) 

s_mov_b32 s8, s32 // (op 848) reg expression
s_mov_b32 s9, s33 // (op 848) reg expression
s_mov_b32 s10, s0 // (op 848) reg expression
s_mov_b32 s11, 131072 // (op 848) call()


v_pack_b32_f16 v8, v12, v13 // (op 128) 
v_pack_b32_f16 v9, v14, v15 // (op 128) 


buffer_store_dwordx2 v[8:9], v5, s[8:11], 0 offen // (op 128) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116:




s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.LRR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2, .LRR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end-RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
  .amdhsa_next_free_vgpr 44
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16384 // lds bytes
  .amdhsa_accum_offset 40
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
  ProblemType:
    OperationType: GEMM
    DataType: F8
    DataTypeA: F8
    DataTypeB: F8
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
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
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeFree1 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC } ]
    macrotile: [32, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 40
    .agpr_count: 4
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 32
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 40
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 48
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 52
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

