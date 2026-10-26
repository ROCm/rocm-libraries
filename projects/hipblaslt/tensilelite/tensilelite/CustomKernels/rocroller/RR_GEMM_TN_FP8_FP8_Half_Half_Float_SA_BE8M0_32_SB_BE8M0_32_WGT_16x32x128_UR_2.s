

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2,@function
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 799) 
v_and_b32 v3, 15, v1 // (op 799) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 799) 


v_and_b32 v0, 48, v1 // (op 799) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 799) 


v_lshrrev_b32 v0, 6, v1 // (op 851) 
v_and_b32 v4, 1, v0 // (op 851) 

v_and_b32 v0, 15, v1 // (op 851) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 851) 


v_and_b32 v0, 48, v1 // (op 851) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 851) 


v_lshrrev_b32 v0, 6, v1 // (op 904) 
v_and_b32 v5, 1, v0 // (op 904) 

v_and_b32 v0, 15, v1 // (op 904) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 904) 


v_and_b32 v0, 63, v1 // (op 904) 
v_lshrrev_b32 v5, 4, v0 // (op 904) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 904) 


v_lshrrev_b32 v5, 7, v1 // (op 977) 
v_and_b32 v6, 15, v1 // (op 977) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 977) 


v_and_b32 v5, 48, v1 // (op 977) 
v_lshl_add_u32 v6, v7, 7, v5 // (op 977) 


v_lshrrev_b32 v5, 6, v1 // (op 1016) 
v_and_b32 v7, 1, v5 // (op 1016) 

v_and_b32 v5, 15, v1 // (op 1016) 
v_lshl_add_u32 v8, v7, 4, v5 // (op 1016) 


v_and_b32 v5, 48, v1 // (op 1016) 
v_lshl_add_u32 v7, v8, 7, v5 // (op 1016) 


v_lshrrev_b32 v5, 6, v1 // (op 1040) 
v_and_b32 v8, 1, v5 // (op 1040) 

v_and_b32 v5, 15, v1 // (op 1040) 
v_lshl_add_u32 v9, v8, 4, v5 // (op 1040) 


v_and_b32 v5, 63, v1 // (op 1040) 
v_lshrrev_b32 v8, 4, v5 // (op 1040) 

v_lshl_add_u32 v5, v9, 2, v8 // (op 1040) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_lshrrev_b32 v10, 3, v1 // (op 510) 
v_lshl_add_u32 v11, s16, 4, v10 // (op 510) 

v_and_b32 v10, -8, v1 // (op 510) 
v_sub_u32 v12, v1, v10 // (op 510) 

v_lshlrev_b32 v10, 4, v12 // (op 510) 

v_mov_b32 v13, 0 // (op 510) convert
v_mov_b32 v12, v10 // (op 510) convert

v_mul_lo_u32 v15, v11, s3 // (op 510) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 510) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 510) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 510) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 510) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 510) most significant half



s_mul_i32 s1, s2, s7 // (op 512) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 512) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 512) least significant
s_mul_hi_u32 s15, s2, s6 // (op 512) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 512) most significant: sum
s_add_u32 s1, s1, s15 // (op 512) most significant: sum


s_mov_b32 s40, s4 // (op 512) reg expression
s_mov_b32 s41, s5 // (op 512) reg expression
s_mov_b32 s42, s0 // (op 512) reg expression
s_mov_b32 s43, 131072 // (op 512) call()

v_lshrrev_b32 v10, 3, v1 // (op 529) 
v_lshl_add_u32 v11, s16, 4, v10 // (op 529) 

v_and_b32 v10, -8, v1 // (op 529) 
v_sub_u32 v12, v1, v10 // (op 529) 

v_lshlrev_b32 v10, 4, v12 // (op 529) 

v_mov_b32 v13, 0 // (op 529) convert
v_mov_b32 v12, v10 // (op 529) convert

v_mul_lo_u32 v17, v11, s3 // (op 529) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 529) most significant: high of low * low
v_mul_lo_u32 v16, v11, s2 // (op 529) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 529) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 529) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 529) most significant half



s_mul_i32 s1, s2, s7 // (op 531) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 531) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 531) least significant
s_mul_hi_u32 s15, s2, s6 // (op 531) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 531) most significant: sum
s_add_u32 s1, s1, s15 // (op 531) most significant: sum


s_mov_b32 s40, s4 // (op 531) reg expression
s_mov_b32 s41, s5 // (op 531) reg expression
s_mov_b32 s42, s0 // (op 531) reg expression
s_mov_b32 s43, 131072 // (op 531) call()

v_and_b32 v10, -8, v1 // (op 574) 
v_sub_u32 v11, v1, v10 // (op 574) 

v_lshlrev_b32 v10, 4, v11 // (op 574) 

v_mov_b32 v13, 0 // (op 574) convert
v_mov_b32 v12, v10 // (op 574) convert

v_lshrrev_b32 v10, 3, v1 // (op 574) 
v_lshl_add_u32 v11, s17, 5, v10 // (op 574) 

v_mul_lo_u32 v19, v11, s9 // (op 574) most significant: low of low * high
v_mul_hi_u32 v10, v11, s8 // (op 574) most significant: high of low * low
v_mul_lo_u32 v18, v11, s8 // (op 574) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 574) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v18 // (op 574) least significant half
v_addc_co_u32 v11, s[0:1], v13, v19, s[0:1] // (op 574) most significant half



s_mul_i32 s1, s8, s13 // (op 576) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 576) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 576) least significant
s_mul_hi_u32 s15, s8, s12 // (op 576) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 576) most significant: sum
s_add_u32 s1, s1, s15 // (op 576) most significant: sum


s_mov_b32 s44, s10 // (op 576) reg expression
s_mov_b32 s45, s11 // (op 576) reg expression
s_mov_b32 s46, s0 // (op 576) reg expression
s_mov_b32 s47, 131072 // (op 576) call()

v_and_b32 v12, -8, v1 // (op 593) 
v_sub_u32 v13, v1, v12 // (op 593) 

v_lshlrev_b32 v12, 4, v13 // (op 593) 

v_mov_b32 v19, 0 // (op 593) convert
v_mov_b32 v18, v12 // (op 593) convert

v_lshrrev_b32 v12, 3, v1 // (op 593) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 593) 

v_mul_lo_u32 v21, v13, s9 // (op 593) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 593) most significant: high of low * low
v_mul_lo_u32 v20, v13, s8 // (op 593) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 593) most significant: sum


v_add_co_u32 v12, s[0:1], v18, v20 // (op 593) least significant half
v_addc_co_u32 v13, s[0:1], v19, v21, s[0:1] // (op 593) most significant half



s_mul_i32 s1, s8, s13 // (op 595) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 595) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 595) least significant
s_mul_hi_u32 s15, s8, s12 // (op 595) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 595) most significant: sum
s_add_u32 s1, s1, s15 // (op 595) most significant: sum


s_mov_b32 s44, s10 // (op 595) reg expression
s_mov_b32 s45, s11 // (op 595) reg expression
s_mov_b32 s46, s0 // (op 595) reg expression
s_mov_b32 s47, 131072 // (op 595) call()

v_lshrrev_b32 v18, 7, v1 // (op 638) 
v_and_b32 v19, 15, v1 // (op 638) 
v_lshl_add_u32 v20, v18, 4, v19 // (op 638) 


v_lshl_add_u32 v18, s16, 4, v20 // (op 638) 

v_and_b32 v19, 63, v1 // (op 638) 
v_lshrrev_b32 v20, 4, v19 // (op 638) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v19, v18, s20 // (op 638) 
v_add_u32 v19, v19, v20 // (op 638) 


s_mul_i32 s1, s20, s7 // (op 640) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 640) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 640) least significant
s_mul_hi_u32 s15, s20, s6 // (op 640) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 640) most significant: sum
s_add_u32 s1, s1, s15 // (op 640) most significant: sum


s_mov_b32 s48, s22 // (op 640) reg expression
s_mov_b32 s49, s23 // (op 640) reg expression
s_mov_b32 s50, s0 // (op 640) reg expression
s_mov_b32 s51, 131072 // (op 640) call()

v_lshrrev_b32 v18, 7, v1 // (op 657) 
v_and_b32 v20, 15, v1 // (op 657) 
v_lshl_add_u32 v21, v18, 4, v20 // (op 657) 


v_lshl_add_u32 v18, s16, 4, v21 // (op 657) 

v_and_b32 v20, 63, v1 // (op 657) 
v_lshrrev_b32 v21, 4, v20 // (op 657) 

v_mul_lo_u32 v20, v18, s20 // (op 657) 
v_add_u32 v20, v20, v21 // (op 657) 


s_mul_i32 s1, s20, s7 // (op 659) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 659) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 659) least significant
s_mul_hi_u32 s15, s20, s6 // (op 659) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 659) most significant: sum
s_add_u32 s1, s1, s15 // (op 659) most significant: sum


s_mov_b32 s48, s22 // (op 659) reg expression
s_mov_b32 s49, s23 // (op 659) reg expression
s_mov_b32 s50, s0 // (op 659) reg expression
s_mov_b32 s51, 131072 // (op 659) call()

v_and_b32 v18, -4, v1 // (op 702) 
v_sub_u32 v21, v1, v18 // (op 702) 

v_lshrrev_b32 v18, 2, v1 // (op 702) 
v_lshl_add_u32 v22, s17, 5, v18 // (op 702) 

v_mul_lo_u32 v18, v22, s24 // (op 702) 

v_add_u32 v22, v21, v18 // (op 702) 


s_mul_i32 s1, s24, s13 // (op 704) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 704) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 704) least significant
s_mul_hi_u32 s15, s24, s12 // (op 704) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 704) most significant: sum
s_add_u32 s1, s1, s15 // (op 704) most significant: sum


s_mov_b32 s52, s26 // (op 704) reg expression
s_mov_b32 s53, s27 // (op 704) reg expression
s_mov_b32 s54, s0 // (op 704) reg expression
s_mov_b32 s55, 131072 // (op 704) call()

v_and_b32 v18, -4, v1 // (op 721) 
v_sub_u32 v21, v1, v18 // (op 721) 

v_lshrrev_b32 v18, 2, v1 // (op 721) 
v_lshl_add_u32 v23, s17, 5, v18 // (op 721) 

v_mul_lo_u32 v18, v23, s24 // (op 721) 

v_add_u32 v23, v21, v18 // (op 721) 


s_mul_i32 s1, s24, s13 // (op 723) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 723) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 723) least significant
s_mul_hi_u32 s15, s24, s12 // (op 723) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 723) most significant: sum
s_add_u32 s1, s1, s15 // (op 723) most significant: sum


s_mov_b32 s52, s26 // (op 723) reg expression
s_mov_b32 s53, s27 // (op 723) reg expression
s_mov_b32 s54, s0 // (op 723) reg expression
s_mov_b32 s55, 131072 // (op 723) call()




buffer_load_ubyte v18, v19, s[48:51], 0 offen offset: 0 // (op 341) Load value
v_and_b32 v21, -8, v1 // (op 839) 
v_sub_u32 v24, v1, v21 // (op 839) 

v_lshlrev_b32 v21, 4, v24 // (op 839) 

v_lshrrev_b32 v24, 3, v1 // (op 839) 
v_lshl_add_u32 v25, v24, 7, v21 // (op 839) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 839) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 4 // (op 459) 
v_mov_b64 v[24:25], s[14:15] // (op 459) convert


v_mov_b64 v[26:27], v[10:11] // (op 459) 
s_mov_b32 m0, s0 // (op 459) reg expression
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 459) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 459) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 459) most significant half

s_add_u32 m0, m0, 2048 // (op 459) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 459) Load value direct to lds






buffer_load_ubyte v21, v22, s[52:55], 0 offen offset: 0 // (op 348) Load value
v_and_b32 v24, -8, v1 // (op 787) 
v_sub_u32 v25, v1, v24 // (op 787) 

v_lshlrev_b32 v24, 4, v25 // (op 787) 

v_lshrrev_b32 v25, 3, v1 // (op 787) 
v_lshl_add_u32 v26, v25, 7, v24 // (op 787) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 787) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 464) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 464) reg expression
buffer_load_dwordx4 v14, s[40:43], 0 offen offset: 0 lds // (op 464) Load value direct to lds





buffer_load_ubyte v24, v19, s[48:51], 0 offen offset:4 // (op 355) Load value

v_and_b32 v19, -8, v1 // (op 1004) 
v_sub_u32 v25, v1, v19 // (op 1004) 

v_lshlrev_b32 v19, 4, v25 // (op 1004) 

v_lshrrev_b32 v25, 3, v1 // (op 1004) 
v_lshl_add_u32 v26, v25, 7, v19 // (op 1004) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1004) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 4 // (op 469) 
v_mov_b64 v[26:27], s[14:15] // (op 469) convert


v_mov_b64 v[28:29], v[10:11] // (op 469) 
s_add_u32 s1, 6144, s0 // (op 469) 
s_mov_b32 m0, s1 // (op 469) reg expression
s_mov_b32 s14, 128 // (op 469) call()
buffer_load_dwordx4 v28, s[44:47], s14 offen offset: 0 lds // (op 469) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 469) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 469) most significant half

s_add_u32 m0, m0, 2048 // (op 469) 
buffer_load_dwordx4 v28, s[44:47], s14 offen offset: 0 lds // (op 469) Load value direct to lds









buffer_load_ubyte v19, v22, s[52:55], 0 offen offset:4 // (op 362) Load value

v_and_b32 v22, -8, v1 // (op 965) 
v_sub_u32 v10, v1, v22 // (op 965) 

v_lshlrev_b32 v11, 4, v10 // (op 965) 

v_lshrrev_b32 v10, 3, v1 // (op 965) 
v_lshl_add_u32 v22, v10, 7, v11 // (op 965) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 965) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 10240, s0 // (op 471) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 471) reg expression
s_mov_b32 s14, 128 // (op 471) call()
buffer_load_dwordx4 v14, s[40:43], s14 offen offset: 0 lds // (op 471) Load value direct to lds




v_and_b32 v22, -4, v1 // (op 892) 
v_sub_u32 v10, v1, v22 // (op 892) 

v_and_b32 v11, 4294967292, v1 // (op 892) 
v_add_u32 v22, v11, v10 // (op 892) 





s_waitcnt vmcnt(6)
ds_write_b8 v22, v21 offset:12288 // (op 368) Store local data 

s_barrier  // (op 1089) 



ds_read_b128 v[26:29], v4 // (op 385) Load local data 
ds_read_b128 v[30:33], v4 offset:64 // (op 385) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1105) 



ds_read_u8 v22, v0 offset:12288 // (op 388) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1073) 



ds_read_b128 v[34:37], v3 offset:4096 // (op 382) Load local data 
ds_read_b128 v[38:41], v3 offset:4160 // (op 382) Load local data 
v_and_b32 v25, -8, v1 // (op 759) 
v_sub_u32 v10, v1, v25 // (op 759) 

v_lshlrev_b32 v11, 4, v10 // (op 759) 

v_lshrrev_b32 v10, 3, v1 // (op 759) 
v_lshl_add_u32 v25, v10, 7, v11 // (op 759) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 759) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -8, v1 // (op 811) 
v_sub_u32 v10, v1, v25 // (op 811) 

v_lshlrev_b32 v11, 4, v10 // (op 811) 

v_lshrrev_b32 v10, 3, v1 // (op 811) 
v_lshl_add_u32 v25, v10, 7, v11 // (op 811) 


s_nop 0
v_readfirstlane_b32 s1, v25 // (op 811) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -4, v1 // (op 863) 
v_sub_u32 v10, v1, v25 // (op 863) 

v_and_b32 v11, 4294967292, v1 // (op 863) 
v_add_u32 v25, v11, v10 // (op 863) 


v_and_b32 v10, -8, v1 // (op 950) 
v_sub_u32 v11, v1, v10 // (op 950) 

v_lshlrev_b32 v10, 4, v11 // (op 950) 

v_lshrrev_b32 v11, 3, v1 // (op 950) 
v_lshl_add_u32 v14, v11, 7, v10 // (op 950) 


s_nop 0
v_readfirstlane_b32 s14, v14 // (op 950) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -8, v1 // (op 989) 
v_sub_u32 v11, v1, v10 // (op 989) 

v_lshlrev_b32 v10, 4, v11 // (op 989) 

v_lshrrev_b32 v11, 3, v1 // (op 989) 
v_lshl_add_u32 v14, v11, 7, v10 // (op 989) 


s_nop 0
v_readfirstlane_b32 s15, v14 // (op 989) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 1028) 
v_sub_u32 v11, v1, v10 // (op 1028) 

v_and_b32 v10, 4294967292, v1 // (op 1028) 
v_add_u32 v14, v10, v11 // (op 1028) 


s_mov_b32 s18, 0 // (op 57) call()
s_ashr_i64 s[38:39], s[28:29], 7 // (op 56) 
s_mov_b32 s19, s38 // (op 56) convert

s_and_b32 s38, s19, 4294967294 // (op 56) 

s_cmp_lt_u32 s18, s38 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56 if false)
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56:
s_add_u32 s18, s18, 2 // (op 58) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v18, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 84) 


v_mov_b32 v15, v20 // (op 77) 

buffer_load_ubyte v18, v15, s[48:51], 0 offen offset:8 // (op 77) Load value
s_barrier  // (op 1092) 

s_lshl_b64 s[38:39], s[8:9], 4 // (op 431) 
v_mov_b64 v[10:11], s[38:39] // (op 431) convert


v_mov_b64 v[42:43], v[12:13] // (op 431) 
v_mov_b64 v[44:45], v[42:43] // (op 431) 
s_mov_b32 m0, s1 // (op 431) reg expression
s_mov_b32 s19, 256 // (op 431) call()
buffer_load_dwordx4 v44, s[44:47], s19 offen offset: 0 lds // (op 431) Load value direct to lds
v_add_co_u32 v44, s[38:39], v44, v10 // (op 431) least significant half
v_addc_co_u32 v45, s[38:39], 0, v45, s[38:39] // (op 431) most significant half

s_add_u32 m0, m0, 2048 // (op 431) 
buffer_load_dwordx4 v44, s[44:47], s19 offen offset: 0 lds // (op 431) Load value direct to lds





v_mov_b32 v10, v23 // (op 21) 

buffer_load_ubyte v21, v10, s[52:55], 0 offen offset:8 // (op 21) Load value
s_barrier  // (op 1076) 


v_mov_b64 v[44:45], v[16:17] // (op 426) 
s_add_u32 s19, 4096, s0 // (op 426) 

s_mov_b32 m0, s19 // (op 426) reg expression
s_mov_b32 s38, 256 // (op 426) call()
buffer_load_dwordx4 v44, s[40:43], s38 offen offset: 0 lds // (op 426) Load value direct to lds


s_barrier  // (op 1137) 



s_waitcnt vmcnt(6)
ds_write_b8 v14, v19 offset:12416 // (op 240) Store local data 
s_barrier  // (op 1128) 



ds_read_b128 v[26:29], v7 offset:6144 // (op 233) Load local data 
ds_read_b128 v[30:33], v7 offset:6208 // (op 233) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1134) 



ds_read_u8 v22, v5 offset:12416 // (op 237) Load local data 
s_barrier  // (op 1184) 
s_waitcnt vmcnt(5)
s_barrier  // (op 1122) 



ds_read_b128 v[34:37], v6 offset:10240 // (op 231) Load local data 
ds_read_b128 v[38:41], v6 offset:10304 // (op 231) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v24, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 238) 



buffer_load_ubyte v24, v15, s[48:51], 0 offen offset:12 // (op 235) Load value

s_barrier  // (op 1131) 
v_add_u32 v20, 8, v20 // (op 649) 

s_lshl_b64 s[38:39], s[8:9], 4 // (op 454) 
v_mov_b64 v[46:47], s[38:39] // (op 454) convert


v_mov_b64 v[48:49], v[42:43] // (op 454) 
s_add_u32 s19, 6144, s15 // (op 454) 
s_mov_b32 m0, s19 // (op 454) reg expression
s_mov_b32 s38, 384 // (op 454) call()
buffer_load_dwordx4 v48, s[44:47], s38 offen offset: 0 lds // (op 454) Load value direct to lds
v_add_co_u32 v48, s[56:57], v48, v46 // (op 454) least significant half
v_addc_co_u32 v49, s[56:57], 0, v49, s[56:57] // (op 454) most significant half

s_add_u32 m0, m0, 2048 // (op 454) 
buffer_load_dwordx4 v48, s[44:47], s38 offen offset: 0 lds // (op 454) Load value direct to lds








v_add_co_u32 v12, vcc, 256, v12 // (op 585) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 585) most significant half
buffer_load_ubyte v19, v10, s[52:55], 0 offen offset:12 // (op 242) Load value

s_barrier  // (op 1125) 
v_add_u32 v23, 8, v23 // (op 713) 


s_add_u32 s19, 10240, s14 // (op 449) 

s_mov_b32 m0, s19 // (op 449) reg expression
s_mov_b32 s38, 384 // (op 449) call()
buffer_load_dwordx4 v44, s[40:43], s38 offen offset: 0 lds // (op 449) Load value direct to lds



s_barrier  // (op 1108) 
v_add_co_u32 v16, vcc, 256, v16 // (op 521) least significant half



v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 521) most significant half
s_waitcnt vmcnt(6)
ds_write_b8 v25, v21 offset:12288 // (op 26) Store local data 
s_barrier  // (op 1098) 



ds_read_b128 v[26:29], v4 // (op 72) Load local data 
ds_read_b128 v[30:33], v4 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1115) 



ds_read_u8 v22, v0 offset:12288 // (op 81) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1082) 



ds_read_b128 v[34:37], v3 offset:4096 // (op 67) Load local data 
ds_read_b128 v[38:41], v3 offset:4160 // (op 67) Load local data 
s_ashr_i64 s[38:39], s[28:29], 7 // (op 56) 
s_mov_b32 s19, s38 // (op 56) convert

s_and_b32 s38, s19, 4294967294 // (op 56) 

s_cmp_lt_u32 s18, s38 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56 if true)
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56:













s_barrier  // (op 1148) 
v_lshrrev_b32 v23, 3, v1 // (op 484) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 484) 
v_lshl_add_u32 v25, s16, 4, v23 // (op 484) 

s_lshr_b32 s14, s0, 1 // (op 484) 

v_and_b32 v23, -8, v1 // (op 484) 
v_sub_u32 v10, v1, v23 // (op 484) 

v_lshlrev_b32 v23, 4, v10 // (op 484) 

v_lshl_add_u32 v10, s14, 8, v23 // (op 484) 


v_mov_b32 v21, 0 // (op 484) convert
v_mov_b32 v20, v10 // (op 484) convert

v_mul_lo_u32 v11, v25, s3 // (op 484) most significant: low of low * high
v_mul_hi_u32 v23, v25, s2 // (op 484) most significant: high of low * low
v_mul_lo_u32 v10, v25, s2 // (op 484) least significant: low of low * low
v_add_u32 v11, v11, v23 // (op 484) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v20 // (op 484) least significant half
v_addc_co_u32 v11, s[0:1], v11, v21, s[0:1] // (op 484) most significant half



s_mul_i32 s1, s2, s7 // (op 486) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 486) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 486) least significant
s_mul_hi_u32 s15, s2, s6 // (op 486) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 486) most significant: sum
s_add_u32 s1, s1, s15 // (op 486) most significant: sum


s_mov_b32 s40, s4 // (op 486) reg expression
s_mov_b32 s41, s5 // (op 486) reg expression
s_mov_b32 s42, s0 // (op 486) reg expression
s_mov_b32 s43, 131072 // (op 486) call()

v_and_b32 v23, -8, v1 // (op 551) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 551) 
v_sub_u32 v25, v1, v23 // (op 551) 

s_lshr_b32 s14, s0, 1 // (op 551) 

v_lshlrev_b32 v23, 4, v25 // (op 551) 

v_lshl_add_u32 v25, s14, 8, v23 // (op 551) 


v_mov_b32 v21, 0 // (op 551) convert
v_mov_b32 v20, v25 // (op 551) convert

v_lshrrev_b32 v23, 3, v1 // (op 551) 
v_lshl_add_u32 v25, s17, 5, v23 // (op 551) 

v_mul_lo_u32 v13, v25, s9 // (op 551) most significant: low of low * high
v_mul_hi_u32 v23, v25, s8 // (op 551) most significant: high of low * low
v_mul_lo_u32 v12, v25, s8 // (op 551) least significant: low of low * low
v_add_u32 v13, v13, v23 // (op 551) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v12 // (op 551) least significant half
v_addc_co_u32 v15, s[0:1], v21, v13, s[0:1] // (op 551) most significant half



s_mul_i32 s1, s8, s13 // (op 553) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 553) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 553) least significant
s_mul_hi_u32 s15, s8, s12 // (op 553) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 553) most significant: sum
s_add_u32 s1, s1, s15 // (op 553) most significant: sum


s_mov_b32 s44, s10 // (op 553) reg expression
s_mov_b32 s45, s11 // (op 553) reg expression
s_mov_b32 s46, s0 // (op 553) reg expression
s_mov_b32 s47, 131072 // (op 553) call()

v_lshrrev_b32 v23, 7, v1 // (op 615) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 615) 
v_and_b32 v25, 15, v1 // (op 615) 
s_lshr_b32 s14, s0, 1 // (op 615) 

v_lshl_add_u32 v12, v23, 4, v25 // (op 615) 


v_lshl_add_u32 v13, s16, 4, v12 // (op 615) 

v_and_b32 v12, 63, v1 // (op 615) 
v_lshrrev_b32 v23, 4, v12 // (op 615) 

v_lshl_add_u32 v12, s14, 3, v23 // (op 615) 


v_mul_lo_u32 v23, v13, s20 // (op 615) 
v_add_u32 v23, v23, v12 // (op 615) 


s_mul_i32 s1, s20, s7 // (op 617) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 617) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 617) least significant
s_mul_hi_u32 s15, s20, s6 // (op 617) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 617) most significant: sum
s_add_u32 s1, s1, s15 // (op 617) most significant: sum


s_mov_b32 s48, s22 // (op 617) reg expression
s_mov_b32 s49, s23 // (op 617) reg expression
s_mov_b32 s50, s0 // (op 617) reg expression
s_mov_b32 s51, 131072 // (op 617) call()

v_and_b32 v25, -4, v1 // (op 679) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 679) 
v_sub_u32 v12, v1, v25 // (op 679) 

s_lshr_b32 s14, s0, 1 // (op 679) 

v_lshrrev_b32 v13, 2, v1 // (op 679) 
v_lshl_add_u32 v25, s14, 3, v12 // (op 679) 


v_lshl_add_u32 v12, s17, 5, v13 // (op 679) 

v_mul_lo_u32 v13, v12, s24 // (op 679) 

v_add_u32 v12, v25, v13 // (op 679) 


s_mul_i32 s1, s24, s13 // (op 681) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 681) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 681) least significant
s_mul_hi_u32 s15, s24, s12 // (op 681) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 681) most significant: sum
s_add_u32 s1, s1, s15 // (op 681) most significant: sum


s_mov_b32 s52, s26 // (op 681) reg expression
s_mov_b32 s53, s27 // (op 681) reg expression
s_mov_b32 s54, s0 // (op 681) reg expression
s_mov_b32 s55, 131072 // (op 681) call()

v_and_b32 v13, -8, v1 // (op 772) 
v_sub_u32 v25, v1, v13 // (op 772) 

v_lshlrev_b32 v13, 4, v25 // (op 772) 

v_lshrrev_b32 v25, 3, v1 // (op 772) 
v_lshl_add_u32 v16, v25, 7, v13 // (op 772) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 772) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -8, v1 // (op 824) 
v_sub_u32 v25, v1, v13 // (op 824) 

v_lshlrev_b32 v13, 4, v25 // (op 824) 

v_lshrrev_b32 v25, 3, v1 // (op 824) 
v_lshl_add_u32 v16, v25, 7, v13 // (op 824) 


s_nop 0
v_readfirstlane_b32 s1, v16 // (op 824) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -4, v1 // (op 876) 
v_sub_u32 v25, v1, v13 // (op 876) 

v_and_b32 v13, 4294967292, v1 // (op 876) 
v_add_u32 v16, v13, v25 // (op 876) 


s_ashr_i64 s[14:15], s[28:29], 7 // (op 166) 
s_and_b32 s18, s14, 4294967294 // (op 166) 

s_ashr_i64 s[14:15], s[28:29], 7 // (op 165) 
s_mov_b32 s19, s14 // (op 165) convert

s_cmp_lt_u32 s18, s19 // (op 165) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165:
s_barrier  // (op 1079) 
s_add_u32 s18, s18, 1 // (op 167) 


v_mov_b64 v[18:19], v[10:11] // (op 436) 
s_add_u32 s19, 4096, s0 // (op 436) 

s_mov_b32 m0, s19 // (op 436) reg expression
buffer_load_dwordx4 v18, s[40:43], 0 offen offset: 0 lds // (op 436) Load value direct to lds


v_add_co_u32 v10, vcc, 128, v10 // (op 474) least significant half
v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 474) most significant half
s_barrier  // (op 1095) 

s_lshl_b64 s[14:15], s[8:9], 4 // (op 442) 
v_mov_b64 v[18:19], s[14:15] // (op 442) convert


v_mov_b64 v[20:21], v[14:15] // (op 442) 
v_mov_b64 v[24:25], v[20:21] // (op 442) 
s_mov_b32 m0, s1 // (op 442) reg expression
buffer_load_dwordx4 v24, s[44:47], 0 offen offset: 0 lds // (op 442) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v18 // (op 442) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 442) most significant half

s_add_u32 m0, m0, 2048 // (op 442) 
buffer_load_dwordx4 v24, s[44:47], 0 offen offset: 0 lds // (op 442) Load value direct to lds



v_add_co_u32 v14, vcc, 128, v14 // (op 543) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 543) most significant half


v_mov_b32 v13, v12 // (op 184) 

buffer_load_ubyte v17, v13, s[52:55], 0 offen offset: 0 // (op 184) Load value

v_add_u32 v12, 4, v12 // (op 671) 
s_barrier  // (op 1111) 
s_waitcnt vmcnt(3)
s_barrier  // (op 1085) 






ds_read_b128 v[24:27], v3 offset:4096 // (op 173) Load local data 
ds_read_b128 v[28:31], v3 offset:4160 // (op 173) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v16, v17 offset:12288 // (op 182) Store local data 

s_barrier  // (op 1101) 



ds_read_b128 v[32:35], v4 // (op 175) Load local data 
ds_read_b128 v[36:39], v4 offset:64 // (op 175) Load local data 


v_mov_b32 v13, v23 // (op 177) 

buffer_load_ubyte v17, v13, s[48:51], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v23, 4, v23 // (op 607) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1118) 



ds_read_u8 v13, v0 offset:12288 // (op 179) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[32:39], a[0:3], v17, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 180) 




s_ashr_i64 s[14:15], s[28:29], 7 // (op 165) 
s_mov_b32 s19, s14 // (op 165) convert

s_cmp_lt_u32 s18, s19 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165 if true)
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165:






















s_barrier  // (op 1174) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

v_cvt_f16_f32 v4, v10 // (op 52) 
v_cvt_f16_f32 v5, v11 // (op 52) 
v_cvt_f16_f32 v6, v12 // (op 52) 
v_cvt_f16_f32 v7, v13 // (op 52) 

v_lshrrev_b32 v0, 7, v1 // (op 918) 
v_and_b32 v3, 63, v1 // (op 918) 
v_lshrrev_b32 v10, 4, v3 // (op 918) 
v_lshlrev_b32 v11, 2, v10 // (op 918) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 918) 


v_lshl_add_u32 v0, s16, 4, v10 // (op 918) 

v_lshrrev_b32 v10, 2, v3 // (op 918) 

v_and_b32 v3, 3, v10 // (op 918) 

v_and_b32 v10, 3, v1 // (op 918) 
v_lshl_add_u32 v11, v3, 2, v10 // (op 918) 


v_lshrrev_b32 v3, 6, v1 // (op 918) 
v_and_b32 v10, 1, v3 // (op 918) 

v_lshl_add_u32 v3, v10, 4, v11 // (op 918) 


v_lshl_add_u32 v10, s17, 5, v3 // (op 918) 

v_mul_lo_u32 v3, v10, s30 // (op 918) 

v_add_lshl_u32 v10, v0, v3, 1 // (op 918) 


s_mul_i32 s15, s30, s13 // (op 920) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 920) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 920) least significant
s_mul_hi_u32 s1, s30, s12 // (op 920) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 920) most significant: sum
s_add_u32 s15, s15, s1 // (op 920) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 920) 

s_mov_b32 s8, s32 // (op 920) reg expression
s_mov_b32 s9, s33 // (op 920) reg expression
s_mov_b32 s10, s0 // (op 920) reg expression
s_mov_b32 s11, 131072 // (op 920) call()


v_pack_b32_f16 v12, v4, v5 // (op 54) 
v_pack_b32_f16 v13, v6, v7 // (op 54) 


buffer_store_dwordx2 v[12:13], v10, s[8:11], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125:
v_lshrrev_b32 v0, 7, v1 // (op 742) 
v_and_b32 v3, 63, v1 // (op 742) 
v_lshrrev_b32 v4, 4, v3 // (op 742) 
v_lshlrev_b32 v5, 2, v4 // (op 742) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 742) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 742) 

v_lshrrev_b32 v4, 2, v3 // (op 742) 

v_and_b32 v3, 3, v4 // (op 742) 

v_and_b32 v4, 3, v1 // (op 742) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 742) 


v_lshrrev_b32 v3, 6, v1 // (op 742) 
v_and_b32 v4, 1, v3 // (op 742) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 742) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 742) 

v_mul_lo_u32 v3, v4, s34 // (op 742) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 742) 


s_mul_i32 s15, s34, s13 // (op 744) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 744) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 744) least significant
s_mul_hi_u32 s1, s34, s12 // (op 744) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 744) most significant: sum
s_add_u32 s15, s15, s1 // (op 744) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 744) 

s_mov_b32 s4, s36 // (op 744) reg expression
s_mov_b32 s5, s37 // (op 744) reg expression
s_mov_b32 s6, s0 // (op 744) reg expression
s_mov_b32 s7, 131072 // (op 744) call()





buffer_load_dwordx2 v[6:7], v4, s[4:7], 0 offen // (op 138) Load value



v_accvgpr_read v10, a0 // (op 134) 
v_accvgpr_read v11, a1 // (op 134) 
v_accvgpr_read v12, a2 // (op 134) 
v_accvgpr_read v13, a3 // (op 134) 
v_mul_f32 v14, v9, v10 // (op 134) 
v_mul_f32 v15, v9, v11 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v4, v6 // (op 139) 
v_lshrrev_b32 v5, 16, v6 // (op 139) 
v_cvt_f32_f16 v5, v5 // (op 139) 
v_mul_f32 v10, v8, v4 // (op 139) 
v_mul_f32 v11, v8, v5 // (op 139) 
v_cvt_f32_f16 v4, v7 // (op 139) 
v_lshrrev_b32 v5, 16, v7 // (op 139) 
v_cvt_f32_f16 v5, v5 // (op 139) 
v_mul_f32 v12, v8, v4 // (op 139) 
v_mul_f32 v13, v8, v5 // (op 139) 


v_add_f32 v4, v10, v14 // (op 135) 
v_add_f32 v5, v11, v15 // (op 135) 
v_add_f32 v6, v12, v16 // (op 135) 
v_add_f32 v7, v13, v17 // (op 135) 


v_cvt_f16_f32 v10, v4 // (op 136) 
v_cvt_f16_f32 v11, v5 // (op 136) 
v_cvt_f16_f32 v12, v6 // (op 136) 
v_cvt_f16_f32 v13, v7 // (op 136) 

v_lshrrev_b32 v0, 7, v1 // (op 935) 
v_and_b32 v3, 63, v1 // (op 935) 
v_lshrrev_b32 v4, 4, v3 // (op 935) 
v_lshlrev_b32 v5, 2, v4 // (op 935) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 935) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 935) 

v_lshrrev_b32 v4, 2, v3 // (op 935) 

v_and_b32 v3, 3, v4 // (op 935) 

v_and_b32 v4, 3, v1 // (op 935) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 935) 


v_lshrrev_b32 v3, 6, v1 // (op 935) 
v_and_b32 v4, 1, v3 // (op 935) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 935) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 935) 

v_mul_lo_u32 v3, v4, s30 // (op 935) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 935) 


s_mul_i32 s15, s30, s13 // (op 937) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 937) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 937) least significant
s_mul_hi_u32 s1, s30, s12 // (op 937) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 937) most significant: sum
s_add_u32 s15, s15, s1 // (op 937) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 937) 

s_mov_b32 s8, s32 // (op 937) reg expression
s_mov_b32 s9, s33 // (op 937) reg expression
s_mov_b32 s10, s0 // (op 937) reg expression
s_mov_b32 s11, 131072 // (op 937) call()


v_pack_b32_f16 v6, v10, v11 // (op 137) 
v_pack_b32_f16 v7, v12, v13 // (op 137) 


buffer_store_dwordx2 v[6:7], v4, s[8:11], 0 offen // (op 137) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125:




s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.LRR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2, .LRR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end-RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
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
    macrotile: [16, 32, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 12544
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 58
    .vgpr_count: 50
    .agpr_count: 4
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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

