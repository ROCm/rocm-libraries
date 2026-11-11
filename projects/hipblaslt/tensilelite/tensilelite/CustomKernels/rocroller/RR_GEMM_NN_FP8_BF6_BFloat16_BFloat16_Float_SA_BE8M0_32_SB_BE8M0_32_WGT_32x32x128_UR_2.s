// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.p2align 8
.type RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2,@function
RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 734) 
v_and_b32 v3, 63, v1 // (op 734) 
v_sub_u32 v4, v3, v0 // (op 734) 

v_lshrrev_b32 v3, 1, v4 // (op 734) 
v_add_lshl_u32 v5, v0, v3, 5 // (op 734) 


v_and_b32 v0, 1, v4 // (op 734) 

v_lshlrev_b32 v3, 3, v0 // (op 734) 

v_lshrrev_b32 v0, 7, v1 // (op 734) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 734) 


v_add_u32 v0, v5, v4 // (op 734) 


v_lshrrev_b32 v3, 6, v1 // (op 787) 
v_and_b32 v4, 1, v3 // (op 787) 

v_and_b32 v3, 15, v1 // (op 787) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 787) 


v_and_b32 v3, 63, v1 // (op 787) 
v_lshrrev_b32 v4, 4, v3 // (op 787) 

v_lshlrev_b32 v3, 5, v4 // (op 787) 

v_lshl_add_u32 v4, v5, 7, v3 // (op 787) 


v_mul_lo_u32 v3, v4, 6 // (op 787) 

v_lshrrev_b32 v4, 3, v3 // (op 787) 

v_and_b32 v3, 48, v1 // (op 860) 
v_and_b32 v5, 63, v1 // (op 860) 
v_sub_u32 v6, v5, v3 // (op 860) 

v_lshrrev_b32 v5, 1, v6 // (op 860) 
v_add_lshl_u32 v7, v3, v5, 5 // (op 860) 


v_and_b32 v3, 1, v6 // (op 860) 

v_lshlrev_b32 v5, 3, v3 // (op 860) 

v_lshrrev_b32 v3, 7, v1 // (op 860) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 860) 


v_add_u32 v3, v7, v6 // (op 860) 


v_lshrrev_b32 v5, 6, v1 // (op 884) 
v_and_b32 v6, 1, v5 // (op 884) 

v_and_b32 v5, 15, v1 // (op 884) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 884) 


v_and_b32 v5, 63, v1 // (op 884) 
v_lshrrev_b32 v6, 4, v5 // (op 884) 

v_lshlrev_b32 v5, 5, v6 // (op 884) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 884) 


v_mul_lo_u32 v5, v6, 6 // (op 884) 

v_lshrrev_b32 v6, 3, v5 // (op 884) 

v_mov_b32 v5, s14 // (op 2) Move value
v_mov_b32 v7, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
v_and_b32 v8, -2, v1 // (op 445) 
v_sub_u32 v9, v1, v8 // (op 445) 

v_lshlrev_b32 v8, 4, v9 // (op 445) 

v_lshl_add_u32 v9, s16, 5, v8 // (op 445) 

v_mov_b32 v11, 0 // (op 445) convert
v_mov_b32 v10, v9 // (op 445) convert

v_lshrrev_b32 v8, 1, v1 // (op 445) 
v_mul_lo_u32 v13, v8, s3 // (op 445) most significant: low of low * high
v_mul_hi_u32 v9, v8, s2 // (op 445) most significant: high of low * low
v_mul_lo_u32 v12, v8, s2 // (op 445) least significant: low of low * low
v_add_u32 v13, v13, v9 // (op 445) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v12 // (op 445) least significant half
v_addc_co_u32 v9, s[0:1], v11, v13, s[0:1] // (op 445) most significant half



s_mul_i32 s1, s2, s7 // (op 447) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 447) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 447) least significant
s_mul_hi_u32 s15, s2, s6 // (op 447) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 447) most significant: sum
s_add_u32 s1, s1, s15 // (op 447) most significant: sum


s_mov_b32 s40, s4 // (op 447) reg expression
s_mov_b32 s41, s5 // (op 447) reg expression
s_mov_b32 s42, s0 // (op 447) reg expression
s_mov_b32 s43, 131072 // (op 447) call()

v_and_b32 v10, -2, v1 // (op 464) 
v_sub_u32 v11, v1, v10 // (op 464) 

v_lshlrev_b32 v10, 4, v11 // (op 464) 

v_lshl_add_u32 v11, s16, 5, v10 // (op 464) 

v_mov_b32 v13, 0 // (op 464) convert
v_mov_b32 v12, v11 // (op 464) convert

v_lshrrev_b32 v10, 1, v1 // (op 464) 
v_mul_lo_u32 v15, v10, s3 // (op 464) most significant: low of low * high
v_mul_hi_u32 v11, v10, s2 // (op 464) most significant: high of low * low
v_mul_lo_u32 v14, v10, s2 // (op 464) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 464) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 464) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 464) most significant half



s_mul_i32 s1, s2, s7 // (op 466) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 466) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 466) least significant
s_mul_hi_u32 s15, s2, s6 // (op 466) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 466) most significant: sum
s_add_u32 s1, s1, s15 // (op 466) most significant: sum


s_mov_b32 s40, s4 // (op 466) reg expression
s_mov_b32 s41, s5 // (op 466) reg expression
s_mov_b32 s42, s0 // (op 466) reg expression
s_mov_b32 s43, 131072 // (op 466) call()

v_and_b32 v12, -8, v1 // (op 509) 
v_sub_u32 v13, v1, v12 // (op 509) 

v_lshlrev_b32 v12, 4, v13 // (op 509) 

v_mov_b32 v15, 0 // (op 509) convert
v_mov_b32 v14, v12 // (op 509) convert

v_lshrrev_b32 v12, 3, v1 // (op 509) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 509) 

v_mul_lo_u32 v17, v13, s9 // (op 509) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 509) most significant: high of low * low
v_mul_lo_u32 v16, v13, s8 // (op 509) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 509) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 509) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 509) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 509) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 509) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 509) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 509) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 509) 

v_mov_b32 v14, v12 // (op 509) convert

s_mul_i32 s1, s13, 6 // (op 511) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 511) least significant
s_mul_hi_u32 s14, s12, 6 // (op 511) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 511) most significant: sum

s_mul_i32 s15, s8, s1 // (op 511) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 511) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 511) least significant
s_mul_hi_u32 s19, s8, s0 // (op 511) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 511) most significant: sum
s_add_u32 s15, s15, s19 // (op 511) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 511) 

s_mov_b32 s44, s10 // (op 511) reg expression
s_mov_b32 s45, s11 // (op 511) reg expression
s_mov_b32 s46, s0 // (op 511) reg expression
s_mov_b32 s47, 131072 // (op 511) call()

v_and_b32 v12, -8, v1 // (op 528) 
v_sub_u32 v13, v1, v12 // (op 528) 

v_lshlrev_b32 v12, 4, v13 // (op 528) 

v_mov_b32 v17, 0 // (op 528) convert
v_mov_b32 v16, v12 // (op 528) convert

v_lshrrev_b32 v15, 3, v1 // (op 528) 
v_lshl_add_u32 v12, s17, 5, v15 // (op 528) 

v_mul_lo_u32 v19, v12, s9 // (op 528) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 528) most significant: high of low * low
v_mul_lo_u32 v18, v12, s8 // (op 528) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 528) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 528) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 528) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 528) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 528) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 528) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 528) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 528) 

v_mov_b32 v15, v12 // (op 528) convert

s_mul_i32 s1, s13, 6 // (op 530) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 530) least significant
s_mul_hi_u32 s14, s12, 6 // (op 530) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 530) most significant: sum

s_mul_i32 s15, s8, s1 // (op 530) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 530) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 530) least significant
s_mul_hi_u32 s19, s8, s0 // (op 530) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 530) most significant: sum
s_add_u32 s15, s15, s19 // (op 530) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 530) 

s_mov_b32 s44, s10 // (op 530) reg expression
s_mov_b32 s45, s11 // (op 530) reg expression
s_mov_b32 s46, s0 // (op 530) reg expression
s_mov_b32 s47, 131072 // (op 530) call()

v_lshrrev_b32 v12, 7, v1 // (op 573) 
v_and_b32 v13, 15, v1 // (op 573) 
v_lshl_add_u32 v16, v12, 4, v13 // (op 573) 


v_lshl_add_u32 v12, s16, 5, v16 // (op 573) 

v_and_b32 v13, 63, v1 // (op 573) 
v_lshrrev_b32 v16, 4, v13 // (op 573) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v13, v16, s20 // (op 573) 

v_add_u32 v16, v12, v13 // (op 573) 


s_mul_i32 s1, s20, s25 // (op 575) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 575) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 575) least significant
s_mul_hi_u32 s15, s20, s24 // (op 575) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 575) most significant: sum
s_add_u32 s1, s1, s15 // (op 575) most significant: sum


s_mov_b32 s48, s22 // (op 575) reg expression
s_mov_b32 s49, s23 // (op 575) reg expression
s_mov_b32 s50, s0 // (op 575) reg expression
s_mov_b32 s51, 131072 // (op 575) call()

v_lshrrev_b32 v12, 7, v1 // (op 592) 
v_and_b32 v13, 15, v1 // (op 592) 
v_lshl_add_u32 v17, v12, 4, v13 // (op 592) 


v_lshl_add_u32 v12, s16, 5, v17 // (op 592) 

v_and_b32 v13, 63, v1 // (op 592) 
v_lshrrev_b32 v17, 4, v13 // (op 592) 

v_mul_lo_u32 v13, v17, s20 // (op 592) 

v_add_u32 v17, v12, v13 // (op 592) 


s_mul_i32 s1, s20, s25 // (op 594) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 594) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 594) least significant
s_mul_hi_u32 s15, s20, s24 // (op 594) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 594) most significant: sum
s_add_u32 s1, s1, s15 // (op 594) most significant: sum


s_mov_b32 s48, s22 // (op 594) reg expression
s_mov_b32 s49, s23 // (op 594) reg expression
s_mov_b32 s50, s0 // (op 594) reg expression
s_mov_b32 s51, 131072 // (op 594) call()

v_lshrrev_b32 v12, 6, v1 // (op 637) 
v_and_b32 v13, 1, v12 // (op 637) 

v_and_b32 v12, 15, v1 // (op 637) 
v_lshl_add_u32 v18, v13, 4, v12 // (op 637) 


v_lshl_add_u32 v12, s17, 5, v18 // (op 637) 

v_mul_lo_u32 v13, v12, s26 // (op 637) 

v_and_b32 v12, 63, v1 // (op 637) 
v_lshrrev_b32 v18, 4, v12 // (op 637) 

v_add_u32 v12, v18, v13 // (op 637) 


s_mul_i32 s1, s26, s13 // (op 639) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 639) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 639) least significant
s_mul_hi_u32 s15, s26, s12 // (op 639) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 639) most significant: sum
s_add_u32 s1, s1, s15 // (op 639) most significant: sum


s_mov_b32 s52, s28 // (op 639) reg expression
s_mov_b32 s53, s29 // (op 639) reg expression
s_mov_b32 s54, s0 // (op 639) reg expression
s_mov_b32 s55, 131072 // (op 639) call()

v_lshrrev_b32 v13, 6, v1 // (op 656) 
v_and_b32 v18, 1, v13 // (op 656) 

v_and_b32 v13, 15, v1 // (op 656) 
v_lshl_add_u32 v19, v18, 4, v13 // (op 656) 


v_lshl_add_u32 v13, s17, 5, v19 // (op 656) 

v_mul_lo_u32 v18, v13, s26 // (op 656) 

v_and_b32 v13, 63, v1 // (op 656) 
v_lshrrev_b32 v19, 4, v13 // (op 656) 

v_add_u32 v13, v19, v18 // (op 656) 


s_mul_i32 s1, s26, s13 // (op 658) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 658) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 658) least significant
s_mul_hi_u32 s15, s26, s12 // (op 658) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 658) most significant: sum
s_add_u32 s1, s1, s15 // (op 658) most significant: sum


s_mov_b32 s52, s28 // (op 658) reg expression
s_mov_b32 s53, s29 // (op 658) reg expression
s_mov_b32 s54, s0 // (op 658) reg expression
s_mov_b32 s55, 131072 // (op 658) call()


v_mov_b64 v[18:19], s[20:21] // (op 312) convert

v_mov_b32 v20, v16 // (op 312) reg expression
v_mov_b32 v21, v20 // (op 312) 
buffer_load_ubyte v22, v21, s[48:51], 0 offen offset: 0 // (op 312) Load value






buffer_load_ubyte v18, v12, s[52:55], 0 offen offset: 0 // (op 317) Load value



buffer_load_dwordx3 v[24:26], v14, s[44:47], 0 offen // (op 321) Load value
v_and_b32 v23, -2, v1 // (op 722) 
v_sub_u32 v19, v1, v23 // (op 722) 

v_lshlrev_b32 v23, 4, v19 // (op 722) 

v_lshrrev_b32 v19, 1, v1 // (op 722) 
v_lshl_add_u32 v20, v19, 5, v23 // (op 722) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 722) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[20:21], v[8:9] // (op 401) reg expression

s_nop 0
s_mov_b32 m0, s0 // (op 401) reg expression
buffer_load_dwordx4 v20, s[40:43], 0 offen offset: 0 lds // (op 401) Load value direct to lds



v_mov_b64 v[20:21], s[20:21] // (op 328) convert

s_lshl_b32 s0, s20, 2 // (op 328) 
v_add_u32 v19, v16, s0 // (op 328) 

v_mov_b32 v23, v19 // (op 328) 
buffer_load_ubyte v27, v23, s[48:51], 0 offen offset: 0 // (op 328) Load value







buffer_load_ubyte v16, v12, s[52:55], 0 offen offset:4 // (op 333) Load value




buffer_load_dwordx3 v[28:30], v14, s[44:47], 0 offen offset:96 // (op 337) Load value

v_and_b32 v12, -2, v1 // (op 848) 
v_sub_u32 v14, v1, v12 // (op 848) 

v_lshlrev_b32 v12, 4, v14 // (op 848) 

v_lshrrev_b32 v14, 1, v1 // (op 848) 
v_lshl_add_u32 v23, v14, 5, v12 // (op 848) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 848) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[14:15], s[2:3], 7 // (op 406) 
v_mov_b32 v12, s14 // (op 406) 
v_mov_b32 v14, s15 // (op 406) 
v_add_co_u32 v20, s[18:19], v8, v12 // (op 406) least significant half
v_addc_co_u32 v21, s[18:19], v9, v14, s[18:19] // (op 406) most significant half




s_add_u32 s1, 4096, s0 // (op 406) 

s_mov_b32 m0, s1 // (op 406) reg expression
buffer_load_dwordx4 v20, s[40:43], 0 offen offset: 0 lds // (op 406) Load value direct to lds




v_and_b32 v12, -8, v1 // (op 775) 
v_sub_u32 v14, v1, v12 // (op 775) 

v_lshlrev_b32 v12, 4, v14 // (op 775) 

v_lshrrev_b32 v14, 3, v1 // (op 775) 
v_lshl_add_u32 v23, v14, 7, v12 // (op 775) 


v_mul_lo_u32 v12, v23, 6 // (op 775) 

v_lshrrev_b32 v14, 3, v12 // (op 775) 




s_waitcnt vmcnt(5)
ds_write_b96 v14, v[24:26] offset:8192 // (op 342) Store local data 

v_and_b32 v12, -2, v1 // (op 694) 
v_sub_u32 v14, v1, v12 // (op 694) 

v_lshlrev_b32 v12, 4, v14 // (op 694) 

v_lshrrev_b32 v14, 1, v1 // (op 694) 
v_lshl_add_u32 v23, v14, 5, v12 // (op 694) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 694) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 746) 
v_sub_u32 v14, v1, v12 // (op 746) 

v_lshlrev_b32 v12, 4, v14 // (op 746) 

v_lshrrev_b32 v14, 3, v1 // (op 746) 
v_lshl_add_u32 v23, v14, 7, v12 // (op 746) 


v_mul_lo_u32 v12, v23, 6 // (op 746) 

v_lshrrev_b32 v14, 3, v12 // (op 746) 

v_and_b32 v12, -2, v1 // (op 833) 
v_sub_u32 v23, v1, v12 // (op 833) 

v_lshlrev_b32 v12, 4, v23 // (op 833) 

v_lshrrev_b32 v23, 1, v1 // (op 833) 
v_lshl_add_u32 v8, v23, 5, v12 // (op 833) 


s_nop 0
v_readfirstlane_b32 s1, v8 // (op 833) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 872) 
v_sub_u32 v23, v1, v12 // (op 872) 

v_lshlrev_b32 v12, 4, v23 // (op 872) 

v_lshrrev_b32 v23, 3, v1 // (op 872) 
v_lshl_add_u32 v8, v23, 7, v12 // (op 872) 


v_mul_lo_u32 v9, v8, 6 // (op 872) 

v_lshrrev_b32 v8, 3, v9 // (op 872) 

s_mov_b32 s14, 0 // (op 53) call()
s_ashr_i64 s[18:19], s[6:7], 7 // (op 52) 
s_mov_b32 s15, s18 // (op 52) convert

s_and_b32 s18, s15, 4294967294 // (op 52) 

s_cmp_lt_u32 s14, s18 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52 // (op 52) Condition: Top (jump to RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52 if false)
RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52:
s_add_u32 s14, s14, 2 // (op 54) 
s_waitcnt vmcnt(4)
s_barrier  // (op 913) 



ds_read_b64_tr_b8 v[32:33], v0 // (op 62) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:256 // (op 62) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:2048 // (op 62) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:2304 // (op 62) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 926) 



ds_read_b128 v[40:43], v4 offset:8192 // (op 67) Load local data 
ds_read_b64 v[44:45], v4 offset:8208 // (op 67) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:45], a[0:3], v22, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 78) 



v_mov_b64 v[20:21], s[20:21] // (op 72) convert

v_mov_b32 v9, v17 // (op 72) 
s_lshl_b32 s15, s20, 3 // (op 72) 
v_add_u32 v12, v9, s15 // (op 72) 

v_mov_b32 v19, v12 // (op 72) 
buffer_load_ubyte v22, v19, s[48:51], 0 offen offset: 0 // (op 72) Load value





v_mov_b32 v12, v13 // (op 76) 

buffer_load_ubyte v18, v12, s[52:55], 0 offen offset:8 // (op 76) Load value


v_mov_b32 v23, v15 // (op 10) 

buffer_load_dwordx3 v[24:26], v23, s[44:47], 0 offen offset:192 // (op 10) Load value
s_barrier  // (op 916) 


v_mov_b64 v[20:21], v[10:11] // (op 385) 
s_lshl_b64 s[18:19], s[2:3], 7 // (op 385) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 385) 

v_mov_b32 v19, s38 // (op 385) 
v_mov_b32 v31, s39 // (op 385) 
v_add_co_u32 v32, s[18:19], v20, v19 // (op 385) least significant half
v_addc_co_u32 v33, s[18:19], v21, v31, s[18:19] // (op 385) most significant half





s_mov_b32 m0, s0 // (op 385) reg expression
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 385) Load value direct to lds

s_barrier  // (op 949) 



s_waitcnt vmcnt(5)
ds_write_b96 v8, v[28:30] offset:11264 // (op 222) Store local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 940) 



ds_read_b64_tr_b8 v[32:33], v3 offset:4096 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:4352 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v3 offset:6144 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v3 offset:6400 // (op 214) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 946) 



ds_read_b128 v[40:43], v6 offset:11264 // (op 216) Load local data 
ds_read_b64 v[44:45], v6 offset:11280 // (op 216) Load local data 
s_barrier  // (op 996) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:45], a[0:3], v27, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 221) 



v_mov_b64 v[32:33], s[20:21] // (op 218) convert

s_lshl_b32 s15, s20, 2 // (op 218) 
s_mul_i32 s18, 3, s15 // (op 218) 

v_add_u32 v19, v9, s18 // (op 218) 

v_mov_b32 v31, v19 // (op 218) 
buffer_load_ubyte v27, v31, s[48:51], 0 offen offset: 0 // (op 218) Load value







s_lshl_b32 s15, s20, 3 // (op 584) 
v_add_u32 v17, v17, s15 // (op 584) 

buffer_load_ubyte v16, v12, s[52:55], 0 offen offset:12 // (op 220) Load value




v_add_u32 v13, 8, v13 // (op 648) 
buffer_load_dwordx3 v[28:30], v23, s[44:47], 0 offen offset:288 // (op 223) Load value

s_barrier  // (op 943) 
v_add_u32 v15, 192, v15 // (op 520) 


s_lshl_b64 s[18:19], s[2:3], 7 // (op 396) 
s_mul_i32 s39, 3, s19 // (op 396) most significant: low of low * high
s_mul_i32 s38, 3, s18 // (op 396) least significant
s_mul_hi_u32 s15, 3, s18 // (op 396) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 396) most significant: sum


v_mov_b32 v9, s38 // (op 396) 
v_mov_b32 v12, s39 // (op 396) 
v_add_co_u32 v32, s[18:19], v20, v9 // (op 396) least significant half
v_addc_co_u32 v33, s[18:19], v21, v12, s[18:19] // (op 396) most significant half




s_add_u32 s15, 4096, s1 // (op 396) 

s_mov_b32 m0, s15 // (op 396) reg expression
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 396) Load value direct to lds



s_barrier  // (op 929) 
s_lshl_b64 s[18:19], s[2:3], 7 // (op 456) 



s_lshl_b64 s[38:39], s[18:19], 1 // (op 456) 

v_mov_b32 v9, s38 // (op 456) 
v_mov_b32 v12, s39 // (op 456) 
v_add_co_u32 v10, s[18:19], v10, v9 // (op 456) least significant half
v_addc_co_u32 v11, s[18:19], v11, v12, s[18:19] // (op 456) most significant half




s_waitcnt vmcnt(5)
ds_write_b96 v14, v[24:26] offset:8192 // (op 13) Store local data 
s_ashr_i64 s[18:19], s[6:7], 7 // (op 52) 
s_mov_b32 s15, s18 // (op 52) convert

s_and_b32 s18, s15, 4294967294 // (op 52) 

s_cmp_lt_u32 s14, s18 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52 // (op 52) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52 if true)
RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52:












s_barrier  // (op 960) 
v_and_b32 v8, -2, v1 // (op 419) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 419) 
v_sub_u32 v9, v1, v8 // (op 419) 

s_lshr_b32 s14, s0, 1 // (op 419) 

v_lshlrev_b32 v8, 4, v9 // (op 419) 

v_lshl_add_u32 v9, s16, 5, v8 // (op 419) 

v_mov_b32 v11, 0 // (op 419) convert
v_mov_b32 v10, v9 // (op 419) convert

v_lshrrev_b32 v8, 1, v1 // (op 419) 
v_lshl_add_u32 v9, s14, 8, v8 // (op 419) 


v_mul_lo_u32 v13, v9, s3 // (op 419) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 419) most significant: high of low * low
v_mul_lo_u32 v12, v9, s2 // (op 419) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 419) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v12 // (op 419) least significant half
v_addc_co_u32 v9, s[0:1], v11, v13, s[0:1] // (op 419) most significant half



s_mul_i32 s1, s2, s7 // (op 421) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 421) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 421) least significant
s_mul_hi_u32 s15, s2, s6 // (op 421) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 421) most significant: sum
s_add_u32 s1, s1, s15 // (op 421) most significant: sum


s_mov_b32 s40, s4 // (op 421) reg expression
s_mov_b32 s41, s5 // (op 421) reg expression
s_mov_b32 s42, s0 // (op 421) reg expression
s_mov_b32 s43, 131072 // (op 421) call()

v_and_b32 v10, -8, v1 // (op 486) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 486) 
v_sub_u32 v11, v1, v10 // (op 486) 

s_lshr_b32 s14, s0, 1 // (op 486) 

v_lshlrev_b32 v10, 4, v11 // (op 486) 

v_lshl_add_u32 v11, s14, 8, v10 // (op 486) 


v_mov_b32 v13, 0 // (op 486) convert
v_mov_b32 v12, v11 // (op 486) convert

v_lshrrev_b32 v10, 3, v1 // (op 486) 
v_lshl_add_u32 v11, s17, 5, v10 // (op 486) 

v_mul_lo_u32 v15, v11, s9 // (op 486) most significant: low of low * high
v_mul_hi_u32 v10, v11, s8 // (op 486) most significant: high of low * low
v_mul_lo_u32 v14, v11, s8 // (op 486) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 486) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 486) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 486) most significant half



v_mul_lo_u32 v13, v11, 6 // (op 486) most significant: low of high * low
v_mul_hi_u32 v14, v10, 6 // (op 486) most significant: high of low * low
v_mul_lo_u32 v12, v10, 6 // (op 486) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 486) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 486) 

v_mov_b32 v12, v10 // (op 486) convert

s_mul_i32 s1, s13, 6 // (op 488) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 488) least significant
s_mul_hi_u32 s14, s12, 6 // (op 488) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 488) most significant: sum

s_mul_i32 s15, s8, s1 // (op 488) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 488) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 488) least significant
s_mul_hi_u32 s19, s8, s0 // (op 488) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 488) most significant: sum
s_add_u32 s15, s15, s19 // (op 488) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 488) 

s_mov_b32 s44, s10 // (op 488) reg expression
s_mov_b32 s45, s11 // (op 488) reg expression
s_mov_b32 s46, s0 // (op 488) reg expression
s_mov_b32 s47, 131072 // (op 488) call()

v_lshrrev_b32 v10, 7, v1 // (op 550) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 550) 
v_and_b32 v11, 15, v1 // (op 550) 
s_lshr_b32 s14, s0, 1 // (op 550) 

v_lshl_add_u32 v13, v10, 4, v11 // (op 550) 


v_lshl_add_u32 v10, s16, 5, v13 // (op 550) 

v_and_b32 v11, 63, v1 // (op 550) 
v_lshrrev_b32 v13, 4, v11 // (op 550) 

v_lshl_add_u32 v11, s14, 3, v13 // (op 550) 


v_mul_lo_u32 v13, v11, s20 // (op 550) 

v_add_u32 v11, v10, v13 // (op 550) 


s_mul_i32 s1, s20, s25 // (op 552) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 552) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 552) least significant
s_mul_hi_u32 s15, s20, s24 // (op 552) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 552) most significant: sum
s_add_u32 s1, s1, s15 // (op 552) most significant: sum


s_mov_b32 s48, s22 // (op 552) reg expression
s_mov_b32 s49, s23 // (op 552) reg expression
s_mov_b32 s50, s0 // (op 552) reg expression
s_mov_b32 s51, 131072 // (op 552) call()

v_and_b32 v10, 63, v1 // (op 614) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 614) 
v_lshrrev_b32 v13, 4, v10 // (op 614) 

s_lshr_b32 s14, s0, 1 // (op 614) 

v_lshrrev_b32 v10, 6, v1 // (op 614) 
v_lshl_add_u32 v14, s14, 3, v13 // (op 614) 


v_and_b32 v13, 1, v10 // (op 614) 

v_and_b32 v10, 15, v1 // (op 614) 
v_lshl_add_u32 v15, v13, 4, v10 // (op 614) 


v_lshl_add_u32 v10, s17, 5, v15 // (op 614) 

v_mul_lo_u32 v13, v10, s26 // (op 614) 

v_add_u32 v10, v14, v13 // (op 614) 


s_mul_i32 s1, s26, s13 // (op 616) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 616) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 616) least significant
s_mul_hi_u32 s15, s26, s12 // (op 616) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 616) most significant: sum
s_add_u32 s1, s1, s15 // (op 616) most significant: sum


s_mov_b32 s52, s28 // (op 616) reg expression
s_mov_b32 s53, s29 // (op 616) reg expression
s_mov_b32 s54, s0 // (op 616) reg expression
s_mov_b32 s55, 131072 // (op 616) call()

v_and_b32 v13, -2, v1 // (op 707) 
v_sub_u32 v14, v1, v13 // (op 707) 

v_lshlrev_b32 v13, 4, v14 // (op 707) 

v_lshrrev_b32 v14, 1, v1 // (op 707) 
v_lshl_add_u32 v15, v14, 5, v13 // (op 707) 


s_nop 0
v_readfirstlane_b32 s0, v15 // (op 707) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -8, v1 // (op 759) 
v_sub_u32 v14, v1, v13 // (op 759) 

v_lshlrev_b32 v13, 4, v14 // (op 759) 

v_lshrrev_b32 v14, 3, v1 // (op 759) 
v_lshl_add_u32 v15, v14, 7, v13 // (op 759) 


v_mul_lo_u32 v13, v15, 6 // (op 759) 

v_lshrrev_b32 v14, 3, v13 // (op 759) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 157) 
s_and_b32 s1, s14, 4294967294 // (op 157) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 156) 
s_mov_b32 s18, s14 // (op 156) convert

s_cmp_lt_u32 s1, s18 // (op 156) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156 // (op 156) Condition: Top (jump to RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156:
s_barrier  // (op 919) 
s_add_u32 s1, s1, 1 // (op 158) 


v_mov_b64 v[16:17], v[8:9] // (op 390) 
v_mov_b64 v[18:19], v[16:17] // (op 390) reg expression

s_mov_b32 m0, s0 // (op 390) reg expression
buffer_load_dwordx4 v18, s[40:43], 0 offen offset: 0 lds // (op 390) Load value direct to lds


s_lshl_b64 s[14:15], s[2:3], 7 // (op 409) 
v_mov_b32 v13, s14 // (op 409) 
v_mov_b32 v15, s15 // (op 409) 
v_add_co_u32 v8, s[18:19], v8, v13 // (op 409) least significant half
v_addc_co_u32 v9, s[18:19], v9, v15, s[18:19] // (op 409) most significant half






v_mov_b32 v13, v12 // (op 173) 

buffer_load_dwordx3 v[16:18], v13, s[44:47], 0 offen // (op 173) Load value

v_add_u32 v12, 96, v12 // (op 478) 
s_barrier  // (op 932) 
s_waitcnt vmcnt(1)
s_barrier  // (op 922) 






ds_read_b64_tr_b8 v[20:21], v0 // (op 164) Transpose load local data 
ds_read_b64_tr_b8 v[22:23], v0 offset:256 // (op 164) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:2048 // (op 164) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:2304 // (op 164) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b96 v14, v[16:18] offset:8192 // (op 172) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 936) 



ds_read_b128 v[28:31], v4 offset:8192 // (op 166) Load local data 
ds_read_b64 v[32:33], v4 offset:8208 // (op 166) Load local data 

v_mov_b64 v[18:19], s[20:21] // (op 168) convert

v_mov_b32 v13, v11 // (op 168) 
v_mov_b32 v15, v13 // (op 168) reg expression
v_mov_b32 v16, v15 // (op 168) 
buffer_load_ubyte v17, v16, s[48:51], 0 offen offset: 0 // (op 168) Load value




s_lshl_b32 s14, s20, 2 // (op 542) 
v_add_u32 v11, v11, s14 // (op 542) 



v_mov_b32 v13, v10 // (op 170) 

buffer_load_ubyte v15, v13, s[52:55], 0 offen offset: 0 // (op 170) Load value

v_add_u32 v10, 4, v10 // (op 606) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[28:33], a[0:3], v17, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 171) 




s_ashr_i64 s[14:15], s[6:7], 7 // (op 156) 
s_mov_b32 s18, s14 // (op 156) convert

s_cmp_lt_u32 s1, s18 // (op 156) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156 // (op 156) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156 if true)
RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156:





















s_barrier  // (op 990) 




v_cmp_eq_f32 vcc, 0.00000, v5 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116
s_nop 0
v_accvgpr_read v8, a0 // (op 42) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 42) 
v_accvgpr_read v10, a2 // (op 42) 
v_accvgpr_read v11, a3 // (op 42) 
v_mul_f32 v12, v7, v8 // (op 42) 
v_mul_f32 v13, v7, v9 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 

v_lshrrev_b32 v8, 16, v12 // (op 48) 
v_lshrrev_b32 v9, 16, v13 // (op 48) 
v_lshrrev_b32 v10, 16, v14 // (op 48) 
v_lshrrev_b32 v11, 16, v15 // (op 48) 

v_lshrrev_b32 v0, 7, v1 // (op 801) 
v_and_b32 v6, 63, v1 // (op 801) 
v_lshrrev_b32 v3, 4, v6 // (op 801) 
v_lshlrev_b32 v4, 2, v3 // (op 801) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 801) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 801) 

v_lshrrev_b32 v3, 2, v6 // (op 801) 

v_and_b32 v4, 3, v3 // (op 801) 

v_and_b32 v3, 3, v1 // (op 801) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 801) 


v_lshrrev_b32 v3, 6, v1 // (op 801) 
v_and_b32 v4, 1, v3 // (op 801) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 801) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 801) 

v_mul_lo_u32 v3, v4, s30 // (op 801) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 801) 


s_mul_i32 s15, s30, s13 // (op 803) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 803) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 803) least significant
s_mul_hi_u32 s1, s30, s12 // (op 803) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 803) most significant: sum
s_add_u32 s15, s15, s1 // (op 803) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 803) 

s_mov_b32 s8, s32 // (op 803) reg expression
s_mov_b32 s9, s33 // (op 803) reg expression
s_mov_b32 s10, s0 // (op 803) reg expression
s_mov_b32 s11, 131072 // (op 803) call()


v_pack_b32_f16 v12, v8, v9 // (op 50) 
v_pack_b32_f16 v13, v10, v11 // (op 50) 


buffer_store_dwordx2 v[12:13], v4, s[8:11], 0 offen // (op 50) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116
RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116:
v_lshrrev_b32 v0, 7, v1 // (op 677) 
v_and_b32 v6, 63, v1 // (op 677) 
v_lshrrev_b32 v3, 4, v6 // (op 677) 
v_lshlrev_b32 v4, 2, v3 // (op 677) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 677) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 677) 

v_lshrrev_b32 v3, 2, v6 // (op 677) 

v_and_b32 v4, 3, v3 // (op 677) 

v_and_b32 v3, 3, v1 // (op 677) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 677) 


v_lshrrev_b32 v3, 6, v1 // (op 677) 
v_and_b32 v4, 1, v3 // (op 677) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 677) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 677) 

v_mul_lo_u32 v3, v4, s34 // (op 677) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 677) 


s_mul_i32 s15, s34, s13 // (op 679) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 679) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 679) least significant
s_mul_hi_u32 s1, s34, s12 // (op 679) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 679) most significant: sum
s_add_u32 s15, s15, s1 // (op 679) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 679) 

s_mov_b32 s4, s36 // (op 679) reg expression
s_mov_b32 s5, s37 // (op 679) reg expression
s_mov_b32 s6, s0 // (op 679) reg expression
s_mov_b32 s7, 131072 // (op 679) call()





buffer_load_dwordx2 v[8:9], v4, s[4:7], 0 offen // (op 129) Load value



v_accvgpr_read v10, a0 // (op 125) 
v_accvgpr_read v11, a1 // (op 125) 
v_accvgpr_read v12, a2 // (op 125) 
v_accvgpr_read v13, a3 // (op 125) 
v_mul_f32 v14, v7, v10 // (op 125) 
v_mul_f32 v15, v7, v11 // (op 125) 
v_mul_f32 v16, v7, v12 // (op 125) 
v_mul_f32 v17, v7, v13 // (op 125) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v8 // (op 130) 
v_lshrrev_b32 v11, 16, v8 // (op 130) 
v_lshlrev_b32 v10, 16, v10 // (op 130) 
v_lshlrev_b32 v11, 16, v11 // (op 130) 
v_mul_f32 v18, v5, v10 // (op 130) 
v_mul_f32 v19, v5, v11 // (op 130) 
v_and_b32 v10, 65535, v9 // (op 130) 
v_lshrrev_b32 v11, 16, v9 // (op 130) 
v_lshlrev_b32 v10, 16, v10 // (op 130) 
v_lshlrev_b32 v11, 16, v11 // (op 130) 
v_mul_f32 v20, v5, v10 // (op 130) 
v_mul_f32 v21, v5, v11 // (op 130) 


v_add_f32 v8, v18, v14 // (op 126) 
v_add_f32 v9, v19, v15 // (op 126) 
v_add_f32 v10, v20, v16 // (op 126) 
v_add_f32 v11, v21, v17 // (op 126) 


v_lshrrev_b32 v12, 16, v8 // (op 127) 
v_lshrrev_b32 v13, 16, v9 // (op 127) 
v_lshrrev_b32 v14, 16, v10 // (op 127) 
v_lshrrev_b32 v15, 16, v11 // (op 127) 

v_lshrrev_b32 v0, 7, v1 // (op 818) 
v_and_b32 v6, 63, v1 // (op 818) 
v_lshrrev_b32 v3, 4, v6 // (op 818) 
v_lshlrev_b32 v4, 2, v3 // (op 818) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 818) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 818) 

v_lshrrev_b32 v3, 2, v6 // (op 818) 

v_and_b32 v4, 3, v3 // (op 818) 

v_and_b32 v3, 3, v1 // (op 818) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 818) 


v_lshrrev_b32 v3, 6, v1 // (op 818) 
v_and_b32 v4, 1, v3 // (op 818) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 818) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 818) 

v_mul_lo_u32 v3, v4, s30 // (op 818) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 818) 


s_mul_i32 s15, s30, s13 // (op 820) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 820) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 820) least significant
s_mul_hi_u32 s1, s30, s12 // (op 820) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 820) most significant: sum
s_add_u32 s15, s15, s1 // (op 820) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 820) 

s_mov_b32 s8, s32 // (op 820) reg expression
s_mov_b32 s9, s33 // (op 820) reg expression
s_mov_b32 s10, s0 // (op 820) reg expression
s_mov_b32 s11, 131072 // (op 820) call()


v_pack_b32_f16 v8, v12, v13 // (op 128) 
v_pack_b32_f16 v9, v14, v15 // (op 128) 


buffer_store_dwordx2 v[8:9], v4, s[8:11], 0 offen // (op 128) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116:




s_endpgm  // End of RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.LRR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end:
.size RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2, .LRR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end-RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 14336 // lds bytes
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: B6
    DestDataType: b
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
  - .name: RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
    .symbol: RR_GEMM_NN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 14336
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

