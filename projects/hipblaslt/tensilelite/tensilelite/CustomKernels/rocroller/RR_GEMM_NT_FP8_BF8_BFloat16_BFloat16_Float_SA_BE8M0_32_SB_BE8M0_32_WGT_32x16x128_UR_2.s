// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2,@function
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 799) 
v_and_b32 v3, 63, v1 // (op 799) 
v_sub_u32 v4, v3, v0 // (op 799) 

v_lshrrev_b32 v3, 1, v4 // (op 799) 
v_add_lshl_u32 v5, v0, v3, 5 // (op 799) 


v_and_b32 v0, 1, v4 // (op 799) 

v_lshlrev_b32 v3, 3, v0 // (op 799) 

v_lshrrev_b32 v0, 6, v1 // (op 799) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 799) 


v_add_u32 v0, v5, v4 // (op 799) 


v_and_b32 v3, 48, v1 // (op 851) 
v_and_b32 v4, 63, v1 // (op 851) 
v_sub_u32 v5, v4, v3 // (op 851) 

v_lshrrev_b32 v4, 1, v5 // (op 851) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 851) 


v_and_b32 v3, 1, v5 // (op 851) 

v_lshlrev_b32 v4, 3, v3 // (op 851) 

v_add_u32 v3, v6, v4 // (op 851) 


v_lshrrev_b32 v4, 6, v1 // (op 904) 
v_and_b32 v5, 15, v1 // (op 904) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 904) 


v_and_b32 v4, 63, v1 // (op 904) 
v_lshrrev_b32 v5, 4, v4 // (op 904) 

v_lshl_add_u32 v4, v5, 5, v6 // (op 904) 


v_and_b32 v5, 48, v1 // (op 977) 
v_and_b32 v6, 63, v1 // (op 977) 
v_sub_u32 v7, v6, v5 // (op 977) 

v_lshrrev_b32 v6, 1, v7 // (op 977) 
v_add_lshl_u32 v8, v5, v6, 5 // (op 977) 


v_and_b32 v5, 1, v7 // (op 977) 

v_lshlrev_b32 v6, 3, v5 // (op 977) 

v_lshrrev_b32 v5, 6, v1 // (op 977) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 977) 


v_add_u32 v5, v8, v7 // (op 977) 


v_and_b32 v6, 48, v1 // (op 1016) 
v_and_b32 v7, 63, v1 // (op 1016) 
v_sub_u32 v8, v7, v6 // (op 1016) 

v_lshrrev_b32 v7, 1, v8 // (op 1016) 
v_add_lshl_u32 v9, v6, v7, 4 // (op 1016) 


v_and_b32 v6, 1, v8 // (op 1016) 

v_lshlrev_b32 v7, 3, v6 // (op 1016) 

v_add_u32 v6, v9, v7 // (op 1016) 


v_lshrrev_b32 v7, 6, v1 // (op 1040) 
v_and_b32 v8, 15, v1 // (op 1040) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 1040) 


v_and_b32 v7, 63, v1 // (op 1040) 
v_lshrrev_b32 v8, 4, v7 // (op 1040) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 1040) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_and_b32 v10, -2, v1 // (op 510) 
v_sub_u32 v11, v1, v10 // (op 510) 

v_lshlrev_b32 v10, 4, v11 // (op 510) 

v_lshl_add_u32 v11, s16, 5, v10 // (op 510) 

v_mov_b32 v13, 0 // (op 510) convert
v_mov_b32 v12, v11 // (op 510) convert

v_lshrrev_b32 v10, 1, v1 // (op 510) 
v_mul_lo_u32 v15, v10, s3 // (op 510) most significant: low of low * high
v_mul_hi_u32 v11, v10, s2 // (op 510) most significant: high of low * low
v_mul_lo_u32 v14, v10, s2 // (op 510) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 510) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 510) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 510) most significant half



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

v_and_b32 v12, -2, v1 // (op 529) 
v_sub_u32 v13, v1, v12 // (op 529) 

v_lshlrev_b32 v12, 4, v13 // (op 529) 

v_lshl_add_u32 v13, s16, 5, v12 // (op 529) 

v_mov_b32 v15, 0 // (op 529) convert
v_mov_b32 v14, v13 // (op 529) convert

v_lshrrev_b32 v12, 1, v1 // (op 529) 
v_mul_lo_u32 v17, v12, s3 // (op 529) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 529) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 529) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 529) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 529) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 529) most significant half



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

s_lshl_b32 s0, s17, 4 // (op 574) 
s_mov_b32 s15, 0 // (op 574) convert
s_mov_b32 s14, s0 // (op 574) convert

v_mul_lo_u32 v15, v1, s9 // (op 574) most significant: low of low * high
v_mul_hi_u32 v16, v1, s8 // (op 574) most significant: high of low * low
v_mul_lo_u32 v14, v1, s8 // (op 574) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 574) most significant: sum

v_mov_b32 v16, s14 // (op 574) 
v_mov_b32 v17, s15 // (op 574) 
v_add_co_u32 v14, s[0:1], v14, v16 // (op 574) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 574) most significant half




s_mul_i32 s1, s8, s7 // (op 576) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 576) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 576) least significant
s_mul_hi_u32 s15, s8, s6 // (op 576) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 576) most significant: sum
s_add_u32 s1, s1, s15 // (op 576) most significant: sum


s_mov_b32 s44, s10 // (op 576) reg expression
s_mov_b32 s45, s11 // (op 576) reg expression
s_mov_b32 s46, s0 // (op 576) reg expression
s_mov_b32 s47, 131072 // (op 576) call()

s_lshl_b32 s0, s17, 4 // (op 593) 
s_mov_b32 s15, 0 // (op 593) convert
s_mov_b32 s14, s0 // (op 593) convert

v_mul_lo_u32 v17, v1, s9 // (op 593) most significant: low of low * high
v_mul_hi_u32 v18, v1, s8 // (op 593) most significant: high of low * low
v_mul_lo_u32 v16, v1, s8 // (op 593) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 593) most significant: sum

v_mov_b32 v18, s14 // (op 593) 
v_mov_b32 v19, s15 // (op 593) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 593) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 593) most significant half




s_mul_i32 s1, s8, s7 // (op 595) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 595) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 595) least significant
s_mul_hi_u32 s15, s8, s6 // (op 595) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 595) most significant: sum
s_add_u32 s1, s1, s15 // (op 595) most significant: sum


s_mov_b32 s44, s10 // (op 595) reg expression
s_mov_b32 s45, s11 // (op 595) reg expression
s_mov_b32 s46, s0 // (op 595) reg expression
s_mov_b32 s47, 131072 // (op 595) call()

v_and_b32 v18, -32, v1 // (op 638) 
v_sub_u32 v19, v1, v18 // (op 638) 

v_lshl_add_u32 v18, s16, 5, v19 // (op 638) 

v_lshrrev_b32 v19, 5, v1 // (op 638) 
v_mul_lo_u32 v20, v19, s12 // (op 638) 

v_add_u32 v19, v18, v20 // (op 638) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 640) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 640) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 640) least significant
s_mul_hi_u32 s15, s12, s22 // (op 640) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 640) most significant: sum
s_add_u32 s1, s1, s15 // (op 640) most significant: sum


s_mov_b32 s48, s20 // (op 640) reg expression
s_mov_b32 s49, s21 // (op 640) reg expression
s_mov_b32 s50, s0 // (op 640) reg expression
s_mov_b32 s51, 131072 // (op 640) call()

v_and_b32 v18, -32, v1 // (op 657) 
v_sub_u32 v20, v1, v18 // (op 657) 

v_lshl_add_u32 v18, s16, 5, v20 // (op 657) 

v_lshrrev_b32 v20, 5, v1 // (op 657) 
v_mul_lo_u32 v21, v20, s12 // (op 657) 

v_add_u32 v20, v18, v21 // (op 657) 


s_mul_i32 s1, s12, s23 // (op 659) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 659) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 659) least significant
s_mul_hi_u32 s15, s12, s22 // (op 659) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 659) most significant: sum
s_add_u32 s1, s1, s15 // (op 659) most significant: sum


s_mov_b32 s48, s20 // (op 659) reg expression
s_mov_b32 s49, s21 // (op 659) reg expression
s_mov_b32 s50, s0 // (op 659) reg expression
s_mov_b32 s51, 131072 // (op 659) call()

v_and_b32 v18, 15, v1 // (op 702) 
v_lshl_add_u32 v21, s17, 4, v18 // (op 702) 

v_and_b32 v18, 63, v1 // (op 702) 
v_lshrrev_b32 v22, 4, v18 // (op 702) 

v_mul_lo_u32 v18, v22, s24 // (op 702) 
v_add_u32 v18, v18, v21 // (op 702) 


s_mul_i32 s1, s24, s23 // (op 704) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 704) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 704) least significant
s_mul_hi_u32 s15, s24, s22 // (op 704) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 704) most significant: sum
s_add_u32 s1, s1, s15 // (op 704) most significant: sum


s_mov_b32 s52, s26 // (op 704) reg expression
s_mov_b32 s53, s27 // (op 704) reg expression
s_mov_b32 s54, s0 // (op 704) reg expression
s_mov_b32 s55, 131072 // (op 704) call()

v_and_b32 v21, 15, v1 // (op 721) 
v_lshl_add_u32 v22, s17, 4, v21 // (op 721) 

v_and_b32 v21, 63, v1 // (op 721) 
v_lshrrev_b32 v23, 4, v21 // (op 721) 

v_mul_lo_u32 v21, v23, s24 // (op 721) 
v_add_u32 v21, v21, v22 // (op 721) 


s_mul_i32 s1, s24, s23 // (op 723) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 723) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 723) least significant
s_mul_hi_u32 s15, s24, s22 // (op 723) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 723) most significant: sum
s_add_u32 s1, s1, s15 // (op 723) most significant: sum


s_mov_b32 s52, s26 // (op 723) reg expression
s_mov_b32 s53, s27 // (op 723) reg expression
s_mov_b32 s54, s0 // (op 723) reg expression
s_mov_b32 s55, 131072 // (op 723) call()


v_mov_b64 v[22:23], s[24:25] // (op 341) convert

v_mov_b32 v24, v18 // (op 341) reg expression
v_mov_b32 v25, v24 // (op 341) 
buffer_load_ubyte v26, v25, s[52:55], 0 offen offset: 0 // (op 341) Load value



v_lshlrev_b32 v22, 4, v1 // (op 839) 
s_nop 0
v_readfirstlane_b32 s0, v22 // (op 839) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[22:23], v[14:15] // (op 459) reg expression

s_nop 0
s_mov_b32 m0, s0 // (op 459) reg expression
buffer_load_dwordx4 v22, s[44:47], 0 offen offset: 0 lds // (op 459) Load value direct to lds


v_and_b32 v22, -2, v1 // (op 787) 
v_sub_u32 v23, v1, v22 // (op 787) 

v_lshlrev_b32 v22, 4, v23 // (op 787) 

v_lshrrev_b32 v23, 1, v1 // (op 787) 
v_lshl_add_u32 v24, v23, 5, v22 // (op 787) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 787) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 464) 
v_mov_b64 v[22:23], s[14:15] // (op 464) convert


v_mov_b64 v[24:25], v[10:11] // (op 464) reg expression
s_add_u32 s1, 2048, s0 // (op 464) 
s_mov_b32 m0, s1 // (op 464) reg expression
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 464) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v22 // (op 464) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 464) most significant half

s_add_u32 m0, m0, 2048 // (op 464) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 464) Load value direct to lds






v_mov_b32 v22, v19 // (op 351) reg expression

buffer_load_ubyte v23, v22, s[48:51], 0 offen offset: 0 // (op 351) Load value


v_mov_b64 v[24:25], s[24:25] // (op 355) convert

s_lshl_b32 s0, s24, 2 // (op 355) 
v_add_u32 v22, v18, s0 // (op 355) 

v_mov_b32 v27, v22 // (op 355) 
buffer_load_ubyte v28, v27, s[52:55], 0 offen offset: 0 // (op 355) Load value




v_lshlrev_b32 v18, 4, v1 // (op 1004) 
s_nop 0
v_readfirstlane_b32 s0, v18 // (op 1004) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[14:15], s[8:9], 7 // (op 469) 
v_mov_b32 v18, s14 // (op 469) 
v_mov_b32 v22, s15 // (op 469) 
v_add_co_u32 v24, s[18:19], v14, v18 // (op 469) least significant half
v_addc_co_u32 v25, s[18:19], v15, v22, s[18:19] // (op 469) most significant half




s_add_u32 s1, 6144, s0 // (op 469) 

s_mov_b32 m0, s1 // (op 469) reg expression
buffer_load_dwordx4 v24, s[44:47], 0 offen offset: 0 lds // (op 469) Load value direct to lds




v_and_b32 v18, -2, v1 // (op 965) 
v_sub_u32 v22, v1, v18 // (op 965) 

v_lshlrev_b32 v18, 4, v22 // (op 965) 

v_lshrrev_b32 v22, 1, v1 // (op 965) 
v_lshl_add_u32 v27, v22, 5, v18 // (op 965) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 965) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 471) 
v_mov_b64 v[14:15], s[14:15] // (op 471) convert


s_lshl_b64 s[14:15], s[2:3], 7 // (op 471) 
v_mov_b32 v18, s14 // (op 471) 
v_mov_b32 v22, s15 // (op 471) 
v_add_co_u32 v24, s[18:19], v10, v18 // (op 471) least significant half
v_addc_co_u32 v25, s[18:19], v11, v22, s[18:19] // (op 471) most significant half




s_add_u32 s1, 8192, s0 // (op 471) 
s_mov_b32 m0, s1 // (op 471) reg expression
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 471) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v14 // (op 471) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 471) most significant half

s_add_u32 m0, m0, 2048 // (op 471) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 471) Load value direct to lds







s_lshl_b32 s0, s12, 2 // (op 365) 
v_add_u32 v18, v19, s0 // (op 365) 


buffer_load_ubyte v22, v18, s[48:51], 0 offen offset: 0 // (op 365) Load value


v_and_b32 v27, -32, v1 // (op 892) 
v_sub_u32 v10, v1, v27 // (op 892) 

v_and_b32 v11, 4294967264, v1 // (op 892) 
v_add_u32 v27, v11, v10 // (op 892) 





s_waitcnt vmcnt(5)
ds_write_b8 v27, v23 offset:12288 // (op 369) Store local data 

s_barrier  // (op 1089) 



ds_read_b64_tr_b8 v[30:31], v3 // (op 385) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v3 offset:128 // (op 385) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:1024 // (op 385) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v3 offset:1152 // (op 385) Transpose load local data 
s_barrier  // (op 1073) 



ds_read_b64_tr_b8 v[38:39], v0 offset:2048 // (op 382) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:2304 // (op 382) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:4096 // (op 382) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:4352 // (op 382) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1105) 


ds_read_u8 v27, v4 offset:12288 // (op 388) Load local data 
v_and_b32 v29, -2, v1 // (op 759) 
v_sub_u32 v10, v1, v29 // (op 759) 

v_lshlrev_b32 v11, 4, v10 // (op 759) 

v_lshrrev_b32 v10, 1, v1 // (op 759) 
v_lshl_add_u32 v29, v10, 5, v11 // (op 759) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 759) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 4, v1 // (op 811) 
s_nop 0
v_readfirstlane_b32 s1, v29 // (op 811) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -32, v1 // (op 863) 
v_sub_u32 v10, v1, v29 // (op 863) 

v_and_b32 v11, 4294967264, v1 // (op 863) 
v_add_u32 v29, v11, v10 // (op 863) 


v_and_b32 v10, -2, v1 // (op 950) 
v_sub_u32 v11, v1, v10 // (op 950) 

v_lshlrev_b32 v10, 4, v11 // (op 950) 

v_lshrrev_b32 v11, 1, v1 // (op 950) 
v_lshl_add_u32 v14, v11, 5, v10 // (op 950) 


s_nop 0
v_readfirstlane_b32 s14, v14 // (op 950) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v10, 4, v1 // (op 989) 
s_nop 0
v_readfirstlane_b32 s15, v10 // (op 989) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -32, v1 // (op 1028) 
v_sub_u32 v11, v1, v10 // (op 1028) 

v_and_b32 v10, 4294967264, v1 // (op 1028) 
v_add_u32 v14, v10, v11 // (op 1028) 


s_mov_b32 s18, 0 // (op 57) call()
s_ashr_i64 s[38:39], s[6:7], 7 // (op 56) 
s_mov_b32 s19, s38 // (op 56) convert

s_and_b32 s38, s19, 4294967294 // (op 56) 

s_cmp_lt_u32 s18, s38 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_56 if false)
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_56:
s_add_u32 s18, s18, 2 // (op 58) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v27, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 84) 

v_mov_b64 v[10:11], s[24:25] // (op 82) convert

v_mov_b32 v15, v21 // (op 82) 
s_lshl_b32 s19, s24, 3 // (op 82) 
v_add_u32 v18, v15, s19 // (op 82) 

v_mov_b32 v19, v18 // (op 82) 
buffer_load_ubyte v26, v19, s[52:55], 0 offen offset: 0 // (op 82) Load value



s_barrier  // (op 1092) 


v_mov_b64 v[10:11], v[16:17] // (op 431) 
s_lshl_b64 s[38:39], s[8:9], 7 // (op 431) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 431) 

v_mov_b32 v18, s56 // (op 431) 
v_mov_b32 v19, s57 // (op 431) 
v_add_co_u32 v24, s[38:39], v10, v18 // (op 431) least significant half
v_addc_co_u32 v25, s[38:39], v11, v19, s[38:39] // (op 431) most significant half





s_mov_b32 m0, s1 // (op 431) reg expression
buffer_load_dwordx4 v24, s[44:47], 0 offen offset: 0 lds // (op 431) Load value direct to lds

s_barrier  // (op 1076) 

s_lshl_b64 s[38:39], s[2:3], 6 // (op 426) 
v_mov_b64 v[18:19], s[38:39] // (op 426) convert


v_mov_b64 v[24:25], v[12:13] // (op 426) 
s_lshl_b64 s[38:39], s[2:3], 7 // (op 426) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 426) 

v_mov_b32 v46, s56 // (op 426) 
v_mov_b32 v47, s57 // (op 426) 
v_add_co_u32 v48, s[38:39], v24, v46 // (op 426) least significant half
v_addc_co_u32 v49, s[38:39], v25, v47, s[38:39] // (op 426) most significant half




s_add_u32 s19, 2048, s0 // (op 426) 
s_mov_b32 m0, s19 // (op 426) reg expression
buffer_load_dwordx4 v48, s[40:43], 0 offen offset: 0 lds // (op 426) Load value direct to lds
v_add_co_u32 v48, s[38:39], v48, v18 // (op 426) least significant half
v_addc_co_u32 v49, s[38:39], 0, v49, s[38:39] // (op 426) most significant half

s_add_u32 m0, m0, 2048 // (op 426) 
buffer_load_dwordx4 v48, s[40:43], 0 offen offset: 0 lds // (op 426) Load value direct to lds





v_mov_b32 v18, v20 // (op 17) 
s_lshl_b32 s19, s12, 3 // (op 17) 
v_add_u32 v19, v18, s19 // (op 17) 


buffer_load_ubyte v23, v19, s[48:51], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1137) 



s_waitcnt vmcnt(5)
ds_write_b8 v14, v22 offset:12416 // (op 240) Store local data 
s_barrier  // (op 1128) 



ds_read_b64_tr_b8 v[30:31], v6 offset:6144 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v6 offset:6272 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v6 offset:7168 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v6 offset:7296 // (op 233) Transpose load local data 
s_barrier  // (op 1122) 



ds_read_b64_tr_b8 v[38:39], v5 offset:8192 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v5 offset:8448 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v5 offset:10240 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v5 offset:10496 // (op 231) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1134) 


ds_read_u8 v27, v7 offset:12416 // (op 235) Load local data 
s_barrier  // (op 1182) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v27, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 238) 

v_mov_b64 v[46:47], s[24:25] // (op 237) convert

s_lshl_b32 s19, s24, 2 // (op 237) 
s_mul_i32 s38, 3, s19 // (op 237) 

v_add_u32 v19, v15, s38 // (op 237) 

v_mov_b32 v48, v19 // (op 237) 
buffer_load_ubyte v28, v48, s[52:55], 0 offen offset: 0 // (op 237) Load value




s_barrier  // (op 1131) 
s_lshl_b32 s19, s24, 3 // (op 713) 


v_add_u32 v21, v21, s19 // (op 713) 

s_lshl_b64 s[38:39], s[8:9], 7 // (op 454) 
s_mul_i32 s57, 3, s39 // (op 454) most significant: low of low * high
s_mul_i32 s56, 3, s38 // (op 454) least significant
s_mul_hi_u32 s19, 3, s38 // (op 454) most significant: high of low * low
s_add_u32 s57, s57, s19 // (op 454) most significant: sum


v_mov_b32 v15, s56 // (op 454) 
v_mov_b32 v19, s57 // (op 454) 
v_add_co_u32 v46, s[38:39], v10, v15 // (op 454) least significant half
v_addc_co_u32 v47, s[38:39], v11, v19, s[38:39] // (op 454) most significant half




s_add_u32 s19, 6144, s15 // (op 454) 

s_mov_b32 m0, s19 // (op 454) reg expression
buffer_load_dwordx4 v46, s[44:47], 0 offen offset: 0 lds // (op 454) Load value direct to lds



s_barrier  // (op 1125) 
s_lshl_b64 s[38:39], s[8:9], 7 // (op 585) 

s_lshl_b64 s[56:57], s[38:39], 1 // (op 585) 

v_mov_b32 v15, s56 // (op 585) 
v_mov_b32 v19, s57 // (op 585) 
s_lshl_b64 s[38:39], s[2:3], 6 // (op 449) 
v_mov_b64 v[10:11], s[38:39] // (op 449) convert


s_lshl_b64 s[38:39], s[2:3], 7 // (op 449) 
s_mul_i32 s59, 3, s39 // (op 449) most significant: low of low * high
s_mul_i32 s58, 3, s38 // (op 449) least significant
s_mul_hi_u32 s19, 3, s38 // (op 449) most significant: high of low * low
s_add_u32 s59, s59, s19 // (op 449) most significant: sum


v_mov_b32 v46, s58 // (op 449) 
v_mov_b32 v47, s59 // (op 449) 
v_add_co_u32 v16, s[38:39], v16, v15 // (op 585) least significant half
v_addc_co_u32 v17, s[38:39], v17, v19, s[38:39] // (op 585) most significant half




v_add_co_u32 v48, s[38:39], v24, v46 // (op 449) least significant half
v_addc_co_u32 v49, s[38:39], v25, v47, s[38:39] // (op 449) most significant half




s_add_u32 s19, 8192, s14 // (op 449) 
s_mov_b32 m0, s19 // (op 449) reg expression
buffer_load_dwordx4 v48, s[40:43], 0 offen offset: 0 lds // (op 449) Load value direct to lds
v_add_co_u32 v48, s[38:39], v48, v10 // (op 449) least significant half
v_addc_co_u32 v49, s[38:39], 0, v49, s[38:39] // (op 449) most significant half

s_add_u32 m0, m0, 2048 // (op 449) 
buffer_load_dwordx4 v48, s[40:43], 0 offen offset: 0 lds // (op 449) Load value direct to lds






s_lshl_b32 s19, s12, 2 // (op 242) 
s_mul_i32 s38, 3, s19 // (op 242) 

v_add_u32 v15, v18, s38 // (op 242) 


s_lshl_b64 s[38:39], s[2:3], 7 // (op 521) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 521) 

v_mov_b32 v19, s56 // (op 521) 
v_mov_b32 v10, s57 // (op 521) 
v_add_co_u32 v12, s[38:39], v12, v19 // (op 521) least significant half
v_addc_co_u32 v13, s[38:39], v13, v10, s[38:39] // (op 521) most significant half




buffer_load_ubyte v22, v15, s[48:51], 0 offen offset: 0 // (op 242) Load value


s_lshl_b32 s19, s12, 3 // (op 649) 
s_barrier  // (op 1108) 
v_add_u32 v20, v20, s19 // (op 649) 




s_waitcnt vmcnt(5)
ds_write_b8 v29, v23 offset:12288 // (op 21) Store local data 
s_barrier  // (op 1098) 



ds_read_b64_tr_b8 v[30:31], v3 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v3 offset:128 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:1024 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v3 offset:1152 // (op 72) Transpose load local data 
s_barrier  // (op 1082) 



ds_read_b64_tr_b8 v[38:39], v0 offset:2048 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:2304 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:4096 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:4352 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1115) 


ds_read_u8 v27, v4 offset:12288 // (op 77) Load local data 
s_ashr_i64 s[38:39], s[6:7], 7 // (op 56) 
s_mov_b32 s19, s38 // (op 56) convert

s_and_b32 s38, s19, 4294967294 // (op 56) 

s_cmp_lt_u32 s18, s38 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_56 if true)
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_56:













s_barrier  // (op 1148) 
v_and_b32 v29, -2, v1 // (op 484) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 484) 
v_sub_u32 v10, v1, v29 // (op 484) 

s_lshr_b32 s14, s0, 1 // (op 484) 

v_lshlrev_b32 v29, 4, v10 // (op 484) 

v_lshl_add_u32 v10, s16, 5, v29 // (op 484) 

v_mov_b32 v21, 0 // (op 484) convert
v_mov_b32 v20, v10 // (op 484) convert

v_lshrrev_b32 v29, 1, v1 // (op 484) 
v_lshl_add_u32 v10, s14, 8, v29 // (op 484) 


v_mul_lo_u32 v19, v10, s3 // (op 484) most significant: low of low * high
v_mul_hi_u32 v29, v10, s2 // (op 484) most significant: high of low * low
v_mul_lo_u32 v18, v10, s2 // (op 484) least significant: low of low * low
v_add_u32 v19, v19, v29 // (op 484) most significant: sum


v_add_co_u32 v10, s[0:1], v20, v18 // (op 484) least significant half
v_addc_co_u32 v11, s[0:1], v21, v19, s[0:1] // (op 484) most significant half



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

s_ashr_i64 s[0:1], s[6:7], 7 // (op 551) 
s_lshr_b32 s14, s0, 1 // (op 551) 

v_lshl_add_u32 v29, s14, 8, v1 // (op 551) 

s_lshl_b32 s0, s17, 4 // (op 551) 
s_mov_b32 s15, 0 // (op 551) convert
s_mov_b32 s14, s0 // (op 551) convert

v_mul_lo_u32 v13, v29, s9 // (op 551) most significant: low of low * high
v_mul_hi_u32 v14, v29, s8 // (op 551) most significant: high of low * low
v_mul_lo_u32 v12, v29, s8 // (op 551) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 551) most significant: sum

v_mov_b32 v14, s14 // (op 551) 
v_mov_b32 v15, s15 // (op 551) 
v_add_co_u32 v12, s[0:1], v12, v14 // (op 551) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 551) most significant half





s_mul_i32 s1, s8, s7 // (op 553) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 553) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 553) least significant
s_mul_hi_u32 s15, s8, s6 // (op 553) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 553) most significant: sum
s_add_u32 s1, s1, s15 // (op 553) most significant: sum


s_mov_b32 s44, s10 // (op 553) reg expression
s_mov_b32 s45, s11 // (op 553) reg expression
s_mov_b32 s46, s0 // (op 553) reg expression
s_mov_b32 s47, 131072 // (op 553) call()

v_and_b32 v29, -32, v1 // (op 615) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 615) 
v_sub_u32 v14, v1, v29 // (op 615) 

s_lshr_b32 s14, s0, 1 // (op 615) 

v_lshl_add_u32 v29, s16, 5, v14 // (op 615) 

v_lshrrev_b32 v14, 5, v1 // (op 615) 
v_lshl_add_u32 v15, s14, 3, v14 // (op 615) 


v_mul_lo_u32 v14, v15, s12 // (op 615) 

v_add_u32 v15, v29, v14 // (op 615) 


s_mul_i32 s1, s12, s23 // (op 617) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 617) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 617) least significant
s_mul_hi_u32 s15, s12, s22 // (op 617) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 617) most significant: sum
s_add_u32 s1, s1, s15 // (op 617) most significant: sum


s_mov_b32 s48, s20 // (op 617) reg expression
s_mov_b32 s49, s21 // (op 617) reg expression
s_mov_b32 s50, s0 // (op 617) reg expression
s_mov_b32 s51, 131072 // (op 617) call()

v_and_b32 v14, 63, v1 // (op 679) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 679) 
v_lshrrev_b32 v29, 4, v14 // (op 679) 

s_lshr_b32 s14, s0, 1 // (op 679) 

v_and_b32 v14, 15, v1 // (op 679) 
v_lshl_add_u32 v16, s14, 3, v29 // (op 679) 


v_lshl_add_u32 v29, s17, 4, v14 // (op 679) 

v_mul_lo_u32 v14, v16, s24 // (op 679) 
v_add_u32 v14, v14, v29 // (op 679) 


s_mul_i32 s1, s24, s23 // (op 681) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 681) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 681) least significant
s_mul_hi_u32 s15, s24, s22 // (op 681) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 681) most significant: sum
s_add_u32 s1, s1, s15 // (op 681) most significant: sum


s_mov_b32 s52, s26 // (op 681) reg expression
s_mov_b32 s53, s27 // (op 681) reg expression
s_mov_b32 s54, s0 // (op 681) reg expression
s_mov_b32 s55, 131072 // (op 681) call()

v_and_b32 v29, -2, v1 // (op 772) 
v_sub_u32 v16, v1, v29 // (op 772) 

v_lshlrev_b32 v29, 4, v16 // (op 772) 

v_lshrrev_b32 v16, 1, v1 // (op 772) 
v_lshl_add_u32 v17, v16, 5, v29 // (op 772) 


s_nop 0
v_readfirstlane_b32 s0, v17 // (op 772) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 4, v1 // (op 824) 
s_nop 0
v_readfirstlane_b32 s1, v29 // (op 824) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -32, v1 // (op 876) 
v_sub_u32 v16, v1, v29 // (op 876) 

v_and_b32 v29, 4294967264, v1 // (op 876) 
v_add_u32 v17, v29, v16 // (op 876) 


s_ashr_i64 s[14:15], s[6:7], 7 // (op 166) 
s_and_b32 s18, s14, 4294967294 // (op 166) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 165) 
s_mov_b32 s19, s14 // (op 165) convert

s_cmp_lt_u32 s18, s19 // (op 165) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_165:
s_barrier  // (op 1079) 
s_add_u32 s18, s18, 1 // (op 167) 

s_lshl_b64 s[14:15], s[2:3], 6 // (op 436) 
v_mov_b64 v[18:19], s[14:15] // (op 436) convert


v_mov_b64 v[20:21], v[10:11] // (op 436) 
v_mov_b64 v[22:23], v[20:21] // (op 436) reg expression
s_add_u32 s19, 2048, s0 // (op 436) 
s_mov_b32 m0, s19 // (op 436) reg expression
buffer_load_dwordx4 v22, s[40:43], 0 offen offset: 0 lds // (op 436) Load value direct to lds
v_add_co_u32 v22, s[14:15], v22, v18 // (op 436) least significant half
v_addc_co_u32 v23, s[14:15], 0, v23, s[14:15] // (op 436) most significant half

s_add_u32 m0, m0, 2048 // (op 436) 
buffer_load_dwordx4 v22, s[40:43], 0 offen offset: 0 lds // (op 436) Load value direct to lds




s_lshl_b64 s[14:15], s[2:3], 7 // (op 474) 
v_mov_b32 v16, s14 // (op 474) 
v_mov_b32 v18, s15 // (op 474) 
v_add_co_u32 v10, s[38:39], v10, v16 // (op 474) least significant half
v_addc_co_u32 v11, s[38:39], v11, v18, s[38:39] // (op 474) most significant half




s_barrier  // (op 1095) 


v_mov_b64 v[18:19], v[12:13] // (op 442) 
v_mov_b64 v[20:21], v[18:19] // (op 442) reg expression

s_mov_b32 m0, s1 // (op 442) reg expression
buffer_load_dwordx4 v20, s[44:47], 0 offen offset: 0 lds // (op 442) Load value direct to lds


s_lshl_b64 s[14:15], s[8:9], 7 // (op 543) 
v_mov_b32 v16, s14 // (op 543) 
v_mov_b32 v18, s15 // (op 543) 
v_add_co_u32 v12, s[38:39], v12, v16 // (op 543) least significant half
v_addc_co_u32 v13, s[38:39], v13, v18, s[38:39] // (op 543) most significant half






v_mov_b32 v16, v15 // (op 184) 
v_mov_b32 v18, v16 // (op 184) reg expression

buffer_load_ubyte v19, v18, s[48:51], 0 offen offset: 0 // (op 184) Load value


s_lshl_b32 s19, s12, 2 // (op 607) 
v_add_u32 v15, v15, s19 // (op 607) 

s_barrier  // (op 1111) 
s_waitcnt vmcnt(2)
s_barrier  // (op 1085) 






ds_read_b64_tr_b8 v[20:21], v0 offset:2048 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[22:23], v0 offset:2304 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:4096 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:4352 // (op 173) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v17, v19 offset:12288 // (op 182) Store local data 

s_barrier  // (op 1101) 



ds_read_b64_tr_b8 v[28:29], v3 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v3 offset:128 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v3 offset:1024 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:1152 // (op 175) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1118) 


ds_read_u8 v16, v4 offset:12288 // (op 177) Load local data 

v_mov_b64 v[18:19], s[24:25] // (op 179) convert

v_mov_b32 v36, v14 // (op 179) 
v_mov_b32 v37, v36 // (op 179) reg expression
v_mov_b32 v38, v37 // (op 179) 
buffer_load_ubyte v39, v38, s[52:55], 0 offen offset: 0 // (op 179) Load value




s_lshl_b32 s19, s24, 2 // (op 671) 
v_add_u32 v14, v14, s19 // (op 671) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[28:35], a[0:3], v16, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 180) 




s_ashr_i64 s[14:15], s[6:7], 7 // (op 165) 
s_mov_b32 s19, s14 // (op 165) convert

s_cmp_lt_u32 s18, s19 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_165 if true)
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_165:






















s_barrier  // (op 1174) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_125
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

v_lshrrev_b32 v4, 16, v10 // (op 52) 
v_lshrrev_b32 v5, 16, v11 // (op 52) 
v_lshrrev_b32 v6, 16, v12 // (op 52) 
v_lshrrev_b32 v7, 16, v13 // (op 52) 

v_lshrrev_b32 v0, 6, v1 // (op 918) 
v_and_b32 v3, 63, v1 // (op 918) 
v_lshrrev_b32 v10, 4, v3 // (op 918) 
v_lshlrev_b32 v11, 2, v10 // (op 918) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 918) 


v_lshl_add_u32 v0, s16, 5, v10 // (op 918) 

v_lshrrev_b32 v10, 2, v3 // (op 918) 

v_and_b32 v3, 3, v10 // (op 918) 

v_and_b32 v10, 3, v1 // (op 918) 
v_lshl_add_u32 v11, v3, 2, v10 // (op 918) 


v_lshl_add_u32 v3, s17, 4, v11 // (op 918) 

v_mul_lo_u32 v10, v3, s28 // (op 918) 

v_add_lshl_u32 v3, v0, v10, 1 // (op 918) 


s_mul_i32 s1, s28, s33 // (op 920) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 920) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 920) least significant
s_mul_hi_u32 s3, s28, s32 // (op 920) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 920) most significant: sum
s_add_u32 s1, s1, s3 // (op 920) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 920) 

s_mov_b32 s4, s30 // (op 920) reg expression
s_mov_b32 s5, s31 // (op 920) reg expression
s_mov_b32 s6, s2 // (op 920) reg expression
s_mov_b32 s7, 131072 // (op 920) call()


v_pack_b32_f16 v10, v4, v5 // (op 54) 
v_pack_b32_f16 v11, v6, v7 // (op 54) 


buffer_store_dwordx2 v[10:11], v3, s[4:7], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_125:
v_lshrrev_b32 v0, 6, v1 // (op 742) 
v_and_b32 v3, 63, v1 // (op 742) 
v_lshrrev_b32 v4, 4, v3 // (op 742) 
v_lshlrev_b32 v5, 2, v4 // (op 742) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 742) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 742) 

v_lshrrev_b32 v4, 2, v3 // (op 742) 

v_and_b32 v3, 3, v4 // (op 742) 

v_and_b32 v4, 3, v1 // (op 742) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 742) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 742) 

v_mul_lo_u32 v4, v3, s34 // (op 742) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 742) 


s_mul_i32 s1, s34, s33 // (op 744) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 744) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 744) least significant
s_mul_hi_u32 s3, s34, s32 // (op 744) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 744) most significant: sum
s_add_u32 s1, s1, s3 // (op 744) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 744) 

s_mov_b32 s8, s36 // (op 744) reg expression
s_mov_b32 s9, s37 // (op 744) reg expression
s_mov_b32 s10, s2 // (op 744) reg expression
s_mov_b32 s11, 131072 // (op 744) call()





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
v_and_b32 v6, 65535, v4 // (op 139) 
v_lshrrev_b32 v7, 16, v4 // (op 139) 
v_lshlrev_b32 v6, 16, v6 // (op 139) 
v_lshlrev_b32 v7, 16, v7 // (op 139) 
v_mul_f32 v10, v8, v6 // (op 139) 
v_mul_f32 v11, v8, v7 // (op 139) 
v_and_b32 v6, 65535, v5 // (op 139) 
v_lshrrev_b32 v7, 16, v5 // (op 139) 
v_lshlrev_b32 v6, 16, v6 // (op 139) 
v_lshlrev_b32 v7, 16, v7 // (op 139) 
v_mul_f32 v12, v8, v6 // (op 139) 
v_mul_f32 v13, v8, v7 // (op 139) 


v_add_f32 v4, v10, v14 // (op 135) 
v_add_f32 v5, v11, v15 // (op 135) 
v_add_f32 v6, v12, v16 // (op 135) 
v_add_f32 v7, v13, v17 // (op 135) 


v_lshrrev_b32 v10, 16, v4 // (op 136) 
v_lshrrev_b32 v11, 16, v5 // (op 136) 
v_lshrrev_b32 v12, 16, v6 // (op 136) 
v_lshrrev_b32 v13, 16, v7 // (op 136) 

v_lshrrev_b32 v0, 6, v1 // (op 935) 
v_and_b32 v3, 63, v1 // (op 935) 
v_lshrrev_b32 v4, 4, v3 // (op 935) 
v_lshlrev_b32 v5, 2, v4 // (op 935) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 935) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 935) 

v_lshrrev_b32 v4, 2, v3 // (op 935) 

v_and_b32 v3, 3, v4 // (op 935) 

v_and_b32 v4, 3, v1 // (op 935) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 935) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 935) 

v_mul_lo_u32 v4, v3, s28 // (op 935) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 935) 


s_mul_i32 s1, s28, s33 // (op 937) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 937) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 937) least significant
s_mul_hi_u32 s3, s28, s32 // (op 937) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 937) most significant: sum
s_add_u32 s1, s1, s3 // (op 937) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 937) 

s_mov_b32 s4, s30 // (op 937) reg expression
s_mov_b32 s5, s31 // (op 937) reg expression
s_mov_b32 s6, s2 // (op 937) reg expression
s_mov_b32 s7, 131072 // (op 937) call()


v_pack_b32_f16 v4, v10, v11 // (op 137) 
v_pack_b32_f16 v5, v12, v13 // (op 137) 


buffer_store_dwordx2 v[4:5], v3, s[4:7], 0 offen // (op 137) Store value







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_125:



s_endpgm  // End of RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.LRR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end:
.size RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2, .LRR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end-RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
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
  - .name: RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
    .symbol: RR_GEMM_NT_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2.kd
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

