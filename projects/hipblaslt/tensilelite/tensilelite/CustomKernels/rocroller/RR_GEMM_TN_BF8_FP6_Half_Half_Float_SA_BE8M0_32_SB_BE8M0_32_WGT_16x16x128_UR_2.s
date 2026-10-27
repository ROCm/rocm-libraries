// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2,@function
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 791) 
v_and_b32 v3, 15, v1 // (op 791) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 791) 


v_and_b32 v0, 48, v1 // (op 791) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 791) 


v_and_b32 v0, 15, v1 // (op 844) 
v_and_b32 v4, 63, v1 // (op 844) 
v_lshrrev_b32 v5, 4, v4 // (op 844) 

v_lshlrev_b32 v4, 5, v5 // (op 844) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 844) 


v_mul_lo_u32 v0, v5, 6 // (op 844) 

v_lshrrev_b32 v4, 3, v0 // (op 844) 

v_lshrrev_b32 v0, 6, v1 // (op 897) 
v_and_b32 v5, 15, v1 // (op 897) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 897) 


v_and_b32 v0, 63, v1 // (op 897) 
v_lshrrev_b32 v5, 4, v0 // (op 897) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 897) 


v_and_b32 v5, 15, v1 // (op 950) 
v_and_b32 v6, 63, v1 // (op 950) 
v_lshrrev_b32 v7, 4, v6 // (op 950) 

v_lshl_add_u32 v6, v5, 2, v7 // (op 950) 


v_lshrrev_b32 v5, 6, v1 // (op 1023) 
v_and_b32 v7, 15, v1 // (op 1023) 
v_lshl_add_u32 v8, v5, 4, v7 // (op 1023) 


v_and_b32 v5, 48, v1 // (op 1023) 
v_lshl_add_u32 v7, v8, 7, v5 // (op 1023) 


v_and_b32 v5, 15, v1 // (op 1047) 
v_and_b32 v8, 63, v1 // (op 1047) 
v_lshrrev_b32 v9, 4, v8 // (op 1047) 

v_lshlrev_b32 v8, 5, v9 // (op 1047) 

v_lshl_add_u32 v9, v5, 7, v8 // (op 1047) 


v_mul_lo_u32 v5, v9, 6 // (op 1047) 

v_lshrrev_b32 v8, 3, v5 // (op 1047) 

v_lshrrev_b32 v5, 6, v1 // (op 1071) 
v_and_b32 v9, 15, v1 // (op 1071) 
v_lshl_add_u32 v10, v5, 4, v9 // (op 1071) 


v_and_b32 v5, 63, v1 // (op 1071) 
v_lshrrev_b32 v9, 4, v5 // (op 1071) 

v_lshl_add_u32 v5, v10, 2, v9 // (op 1071) 


v_and_b32 v9, 15, v1 // (op 1095) 
v_and_b32 v10, 63, v1 // (op 1095) 
v_lshrrev_b32 v11, 4, v10 // (op 1095) 

v_lshl_add_u32 v10, v9, 2, v11 // (op 1095) 


v_mov_b32 v9, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_lshrrev_b32 v12, 3, v1 // (op 502) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 502) 

v_and_b32 v12, -8, v1 // (op 502) 
v_sub_u32 v14, v1, v12 // (op 502) 

v_lshlrev_b32 v12, 4, v14 // (op 502) 

v_mov_b32 v15, 0 // (op 502) convert
v_mov_b32 v14, v12 // (op 502) convert

v_mul_lo_u32 v17, v13, s3 // (op 502) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 502) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 502) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 502) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 502) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 502) most significant half



s_mul_i32 s1, s2, s7 // (op 504) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 504) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 504) least significant
s_mul_hi_u32 s15, s2, s6 // (op 504) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 504) most significant: sum
s_add_u32 s1, s1, s15 // (op 504) most significant: sum


s_mov_b32 s40, s4 // (op 504) reg expression
s_mov_b32 s41, s5 // (op 504) reg expression
s_mov_b32 s42, s0 // (op 504) reg expression
s_mov_b32 s43, 131072 // (op 504) call()

v_lshrrev_b32 v12, 3, v1 // (op 521) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 521) 

v_and_b32 v12, -8, v1 // (op 521) 
v_sub_u32 v14, v1, v12 // (op 521) 

v_lshlrev_b32 v12, 4, v14 // (op 521) 

v_mov_b32 v15, 0 // (op 521) convert
v_mov_b32 v14, v12 // (op 521) convert

v_mul_lo_u32 v19, v13, s3 // (op 521) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 521) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 521) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 521) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 521) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 521) most significant half



s_mul_i32 s1, s2, s7 // (op 523) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 523) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 523) least significant
s_mul_hi_u32 s15, s2, s6 // (op 523) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 523) most significant: sum
s_add_u32 s1, s1, s15 // (op 523) most significant: sum


s_mov_b32 s40, s4 // (op 523) reg expression
s_mov_b32 s41, s5 // (op 523) reg expression
s_mov_b32 s42, s0 // (op 523) reg expression
s_mov_b32 s43, 131072 // (op 523) call()

v_and_b32 v12, -4, v1 // (op 566) 
v_sub_u32 v13, v1, v12 // (op 566) 

v_lshlrev_b32 v12, 5, v13 // (op 566) 

v_mov_b32 v15, 0 // (op 566) convert
v_mov_b32 v14, v12 // (op 566) convert

v_lshrrev_b32 v12, 2, v1 // (op 566) 
v_lshl_add_u32 v13, s17, 4, v12 // (op 566) 

v_mul_lo_u32 v21, v13, s9 // (op 566) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 566) most significant: high of low * low
v_mul_lo_u32 v20, v13, s8 // (op 566) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 566) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 566) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 566) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 566) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 566) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 566) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 566) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 566) 

v_mov_b32 v14, v12 // (op 566) convert

s_mul_i32 s1, s13, 6 // (op 568) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 568) least significant
s_mul_hi_u32 s14, s12, 6 // (op 568) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 568) most significant: sum

s_mul_i32 s15, s8, s1 // (op 568) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 568) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 568) least significant
s_mul_hi_u32 s19, s8, s0 // (op 568) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 568) most significant: sum
s_add_u32 s15, s15, s19 // (op 568) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 568) 

s_mov_b32 s44, s10 // (op 568) reg expression
s_mov_b32 s45, s11 // (op 568) reg expression
s_mov_b32 s46, s0 // (op 568) reg expression
s_mov_b32 s47, 131072 // (op 568) call()

v_and_b32 v15, -4, v1 // (op 585) 
v_sub_u32 v12, v1, v15 // (op 585) 

v_lshlrev_b32 v13, 5, v12 // (op 585) 

v_mov_b32 v21, 0 // (op 585) convert
v_mov_b32 v20, v13 // (op 585) convert

v_lshrrev_b32 v15, 2, v1 // (op 585) 
v_lshl_add_u32 v12, s17, 4, v15 // (op 585) 

v_mul_lo_u32 v23, v12, s9 // (op 585) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 585) most significant: high of low * low
v_mul_lo_u32 v22, v12, s8 // (op 585) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 585) most significant: sum


v_add_co_u32 v12, s[0:1], v20, v22 // (op 585) least significant half
v_addc_co_u32 v13, s[0:1], v21, v23, s[0:1] // (op 585) most significant half



v_mul_lo_u32 v21, v13, 6 // (op 585) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 585) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 585) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 585) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 585) 

v_mov_b32 v15, v12 // (op 585) convert

s_mul_i32 s1, s13, 6 // (op 587) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 587) least significant
s_mul_hi_u32 s14, s12, 6 // (op 587) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 587) most significant: sum

s_mul_i32 s15, s8, s1 // (op 587) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 587) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 587) least significant
s_mul_hi_u32 s19, s8, s0 // (op 587) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 587) most significant: sum
s_add_u32 s15, s15, s19 // (op 587) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 587) 

s_mov_b32 s44, s10 // (op 587) reg expression
s_mov_b32 s45, s11 // (op 587) reg expression
s_mov_b32 s46, s0 // (op 587) reg expression
s_mov_b32 s47, 131072 // (op 587) call()

v_lshrrev_b32 v12, 2, v1 // (op 630) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 630) 

v_and_b32 v12, -4, v1 // (op 630) 
v_sub_u32 v20, v1, v12 // (op 630) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v12, v13, s20 // (op 630) 
v_add_u32 v12, v12, v20 // (op 630) 


s_mul_i32 s1, s20, s7 // (op 632) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 632) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 632) least significant
s_mul_hi_u32 s15, s20, s6 // (op 632) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 632) most significant: sum
s_add_u32 s1, s1, s15 // (op 632) most significant: sum


s_mov_b32 s48, s22 // (op 632) reg expression
s_mov_b32 s49, s23 // (op 632) reg expression
s_mov_b32 s50, s0 // (op 632) reg expression
s_mov_b32 s51, 131072 // (op 632) call()

v_lshrrev_b32 v13, 2, v1 // (op 649) 
v_lshl_add_u32 v20, s16, 4, v13 // (op 649) 

v_and_b32 v13, -4, v1 // (op 649) 
v_sub_u32 v21, v1, v13 // (op 649) 

v_mul_lo_u32 v13, v20, s20 // (op 649) 
v_add_u32 v13, v13, v21 // (op 649) 


s_mul_i32 s1, s20, s7 // (op 651) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 651) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 651) least significant
s_mul_hi_u32 s15, s20, s6 // (op 651) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 651) most significant: sum
s_add_u32 s1, s1, s15 // (op 651) most significant: sum


s_mov_b32 s48, s22 // (op 651) reg expression
s_mov_b32 s49, s23 // (op 651) reg expression
s_mov_b32 s50, s0 // (op 651) reg expression
s_mov_b32 s51, 131072 // (op 651) call()

v_and_b32 v20, -4, v1 // (op 694) 
v_sub_u32 v21, v1, v20 // (op 694) 

v_lshrrev_b32 v20, 2, v1 // (op 694) 
v_lshl_add_u32 v22, s17, 4, v20 // (op 694) 

v_mul_lo_u32 v20, v22, s24 // (op 694) 

v_add_u32 v22, v21, v20 // (op 694) 


s_mul_i32 s1, s24, s13 // (op 696) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 696) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 696) least significant
s_mul_hi_u32 s15, s24, s12 // (op 696) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 696) most significant: sum
s_add_u32 s1, s1, s15 // (op 696) most significant: sum


s_mov_b32 s52, s26 // (op 696) reg expression
s_mov_b32 s53, s27 // (op 696) reg expression
s_mov_b32 s54, s0 // (op 696) reg expression
s_mov_b32 s55, 131072 // (op 696) call()

v_and_b32 v20, -4, v1 // (op 713) 
v_sub_u32 v21, v1, v20 // (op 713) 

v_lshrrev_b32 v20, 2, v1 // (op 713) 
v_lshl_add_u32 v23, s17, 4, v20 // (op 713) 

v_mul_lo_u32 v20, v23, s24 // (op 713) 

v_add_u32 v23, v21, v20 // (op 713) 


s_mul_i32 s1, s24, s13 // (op 715) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 715) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 715) least significant
s_mul_hi_u32 s15, s24, s12 // (op 715) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 715) most significant: sum
s_add_u32 s1, s1, s15 // (op 715) most significant: sum


s_mov_b32 s52, s26 // (op 715) reg expression
s_mov_b32 s53, s27 // (op 715) reg expression
s_mov_b32 s54, s0 // (op 715) reg expression
s_mov_b32 s55, 131072 // (op 715) call()




buffer_load_dwordx4 v[24:27], v14, s[44:47], 0 offen // (op 370) Load value
buffer_load_dwordx2 v[28:29], v14, s[44:47], 0 offen offset:16 // (op 370) Load value



buffer_load_ubyte v20, v22, s[52:55], 0 offen offset: 0 // (op 373) Load value
v_and_b32 v21, -8, v1 // (op 779) 
v_sub_u32 v30, v1, v21 // (op 779) 

v_lshlrev_b32 v21, 4, v30 // (op 779) 

v_lshrrev_b32 v30, 3, v1 // (op 779) 
v_lshl_add_u32 v31, v30, 7, v21 // (op 779) 


s_nop 0
v_readfirstlane_b32 s0, v31 // (op 779) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 3 // (op 458) 
v_mov_b64 v[30:31], s[14:15] // (op 458) convert


v_mov_b64 v[32:33], v[16:17] // (op 458) 
s_mov_b32 m0, s0 // (op 458) reg expression
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 458) Load value direct to lds
v_add_co_u32 v32, s[14:15], v32, v30 // (op 458) least significant half
v_addc_co_u32 v33, s[14:15], 0, v33, s[14:15] // (op 458) most significant half

s_add_u32 m0, m0, 1024 // (op 458) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 458) Load value direct to lds






buffer_load_ubyte v21, v12, s[48:51], 0 offen offset: 0 // (op 379) Load value



buffer_load_dwordx4 v[30:33], v14, s[44:47], 0 offen offset:96 // (op 382) Load value
buffer_load_dwordx2 v[34:35], v14, s[44:47], 0 offen offset:112 // (op 382) Load value




buffer_load_ubyte v14, v22, s[52:55], 0 offen offset:4 // (op 385) Load value

v_and_b32 v22, -8, v1 // (op 1011) 
v_sub_u32 v36, v1, v22 // (op 1011) 

v_lshlrev_b32 v22, 4, v36 // (op 1011) 

v_lshrrev_b32 v36, 3, v1 // (op 1011) 
v_lshl_add_u32 v37, v36, 7, v22 // (op 1011) 


s_nop 0
v_readfirstlane_b32 s0, v37 // (op 1011) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 3 // (op 463) 
v_mov_b64 v[36:37], s[14:15] // (op 463) convert


v_mov_b64 v[38:39], v[16:17] // (op 463) 
s_add_u32 s1, 2048, s0 // (op 463) 
s_mov_b32 m0, s1 // (op 463) reg expression
s_mov_b32 s14, 128 // (op 463) call()
buffer_load_dwordx4 v38, s[40:43], s14 offen offset: 0 lds // (op 463) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v36 // (op 463) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 463) most significant half

s_add_u32 m0, m0, 1024 // (op 463) 
buffer_load_dwordx4 v38, s[40:43], s14 offen offset: 0 lds // (op 463) Load value direct to lds









buffer_load_ubyte v22, v12, s[48:51], 0 offen offset:4 // (op 391) Load value

v_and_b32 v12, -4, v1 // (op 832) 
v_sub_u32 v16, v1, v12 // (op 832) 

v_lshlrev_b32 v12, 5, v16 // (op 832) 

v_lshrrev_b32 v16, 2, v1 // (op 832) 
v_lshl_add_u32 v17, v16, 7, v12 // (op 832) 


v_mul_lo_u32 v12, v17, 6 // (op 832) 

v_lshrrev_b32 v16, 3, v12 // (op 832) 




s_waitcnt vmcnt(11)
ds_write_b128 v16, v[24:27] offset:4096 // (op 393) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v16, v[28:29] offset:4112 // (op 393) Store local data 

v_and_b32 v12, -4, v1 // (op 938) 
v_sub_u32 v16, v1, v12 // (op 938) 

v_and_b32 v12, 4294967292, v1 // (op 938) 
v_add_u32 v17, v12, v16 // (op 938) 





s_waitcnt vmcnt(9)
ds_write_b8 v17, v20 offset:5632 // (op 394) Store local data 

v_and_b32 v12, -4, v1 // (op 885) 
v_sub_u32 v16, v1, v12 // (op 885) 

v_and_b32 v12, 4294967292, v1 // (op 885) 
v_add_u32 v17, v12, v16 // (op 885) 





s_waitcnt vmcnt(6)
ds_write_b8 v17, v21 offset:5696 // (op 396) Store local data 

v_and_b32 v12, -8, v1 // (op 751) 
v_sub_u32 v16, v1, v12 // (op 751) 

v_lshlrev_b32 v12, 4, v16 // (op 751) 

v_lshrrev_b32 v16, 3, v1 // (op 751) 
v_lshl_add_u32 v17, v16, 7, v12 // (op 751) 


s_nop 0
v_readfirstlane_b32 s0, v17 // (op 751) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 803) 
v_sub_u32 v16, v1, v12 // (op 803) 

v_lshlrev_b32 v12, 5, v16 // (op 803) 

v_lshrrev_b32 v16, 2, v1 // (op 803) 
v_lshl_add_u32 v17, v16, 7, v12 // (op 803) 


v_mul_lo_u32 v12, v17, 6 // (op 803) 

v_lshrrev_b32 v16, 3, v12 // (op 803) 

v_and_b32 v12, -4, v1 // (op 856) 
v_sub_u32 v17, v1, v12 // (op 856) 

v_and_b32 v12, 4294967292, v1 // (op 856) 
v_add_u32 v36, v12, v17 // (op 856) 


v_and_b32 v12, -4, v1 // (op 909) 
v_sub_u32 v17, v1, v12 // (op 909) 

v_and_b32 v12, 4294967292, v1 // (op 909) 
v_add_u32 v37, v12, v17 // (op 909) 


v_and_b32 v12, -8, v1 // (op 996) 
v_sub_u32 v17, v1, v12 // (op 996) 

v_lshlrev_b32 v12, 4, v17 // (op 996) 

v_lshrrev_b32 v17, 3, v1 // (op 996) 
v_lshl_add_u32 v38, v17, 7, v12 // (op 996) 


s_nop 0
v_readfirstlane_b32 s1, v38 // (op 996) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 1035) 
v_sub_u32 v17, v1, v12 // (op 1035) 

v_lshlrev_b32 v12, 5, v17 // (op 1035) 

v_lshrrev_b32 v17, 2, v1 // (op 1035) 
v_lshl_add_u32 v38, v17, 7, v12 // (op 1035) 


v_mul_lo_u32 v12, v38, 6 // (op 1035) 

v_lshrrev_b32 v17, 3, v12 // (op 1035) 

v_and_b32 v12, -4, v1 // (op 1059) 
v_sub_u32 v38, v1, v12 // (op 1059) 

v_and_b32 v12, 4294967292, v1 // (op 1059) 
v_add_u32 v39, v12, v38 // (op 1059) 


v_and_b32 v12, -4, v1 // (op 1083) 
v_sub_u32 v38, v1, v12 // (op 1083) 

v_and_b32 v12, 4294967292, v1 // (op 1083) 
v_add_u32 v40, v12, v38 // (op 1083) 


s_mov_b32 s14, 0 // (op 61) call()
s_ashr_i64 s[18:19], s[28:29], 7 // (op 60) 
s_mov_b32 s15, s18 // (op 60) convert

s_and_b32 s18, s15, 4294967294 // (op 60) 

s_cmp_lt_u32 s14, s18 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60:


s_barrier  // (op 1132) 
s_add_u32 s14, s14, 2 // (op 62) 



v_mov_b32 v12, v15 // (op 10) 

ds_read_b128 v[42:45], v3 // (op 72) Load local data 
ds_read_b128 v[46:49], v3 offset:64 // (op 72) Load local data 
buffer_load_dwordx4 v[24:27], v12, s[44:47], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[28:29], v12, s[44:47], 0 offen offset:208 // (op 10) Load value
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1145) 





v_mov_b32 v38, v23 // (op 25) 

buffer_load_ubyte v20, v38, s[52:55], 0 offen offset:8 // (op 25) Load value
ds_read_b128 v[50:53], v4 offset:4096 // (op 77) Load local data 
ds_read_b64 v[54:55], v4 offset:4112 // (op 77) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1135) 

s_barrier  // (op 1159) 
s_lshl_b64 s[18:19], s[2:3], 3 // (op 442) 



ds_read_u8 v41, v0 offset:5696 // (op 82) Load local data 
v_mov_b64 v[56:57], s[18:19] // (op 442) convert


s_barrier  // (op 1173) 
v_mov_b64 v[58:59], v[18:19] // (op 442) 



v_mov_b64 v[60:61], v[58:59] // (op 442) 
s_mov_b32 m0, s0 // (op 442) reg expression
s_mov_b32 s15, 256 // (op 442) call()
ds_read_u8 v62, v6 offset:5632 // (op 87) Load local data 
buffer_load_dwordx4 v60, s[40:43], s15 offen offset: 0 lds // (op 442) Load value direct to lds
v_add_co_u32 v60, s[18:19], v60, v56 // (op 442) least significant half
v_addc_co_u32 v61, s[18:19], 0, v61, s[18:19] // (op 442) most significant half

s_add_u32 m0, m0, 1024 // (op 442) 
buffer_load_dwordx4 v60, s[40:43], s15 offen offset: 0 lds // (op 442) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:55], a[0:3], v41, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 90) 






v_mov_b32 v41, v13 // (op 17) 

buffer_load_ubyte v21, v41, s[48:51], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 1196) 



s_waitcnt vmcnt(11)
ds_write_b128 v17, v[30:33] offset:5760 // (op 257) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v17, v[34:35] offset:5776 // (op 257) Store local data 
s_barrier  // (op 1208) 



s_waitcnt vmcnt(9)
ds_write_b8 v40, v14 offset:7296 // (op 259) Store local data 
s_barrier  // (op 1202) 



s_waitcnt vmcnt(6)
ds_write_b8 v39, v22 offset:7360 // (op 258) Store local data 



s_barrier  // (op 1187) 
buffer_load_dwordx4 v[30:33], v12, s[44:47], 0 offen offset:288 // (op 260) Load value



ds_read_b128 v[42:45], v7 offset:2048 // (op 249) Load local data 
ds_read_b128 v[46:49], v7 offset:2112 // (op 249) Load local data 
buffer_load_dwordx2 v[34:35], v12, s[44:47], 0 offen offset:304 // (op 260) Load value

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1193) 






v_add_u32 v15, 192, v15 // (op 577) 
buffer_load_ubyte v14, v38, s[52:55], 0 offen offset:12 // (op 262) Load value
ds_read_b128 v[50:53], v8 offset:5760 // (op 251) Load local data 

ds_read_b64 v[54:55], v8 offset:5776 // (op 251) Load local data 
v_add_u32 v23, 8, v23 // (op 705) 
s_barrier  // (op 1263) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1199) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1190) 




ds_read_u8 v12, v5 offset:7360 // (op 253) Load local data 
s_lshl_b64 s[18:19], s[2:3], 3 // (op 453) 
v_mov_b64 v[56:57], s[18:19] // (op 453) convert


s_barrier  // (op 1205) 
s_barrier  // (op 1269) 



ds_read_u8 v38, v10 offset:7296 // (op 255) Load local data 
v_mov_b64 v[60:61], v[58:59] // (op 453) 
s_add_u32 s15, 2048, s1 // (op 453) 
s_barrier  // (op 1275) 
s_mov_b32 m0, s15 // (op 453) reg expression
s_mov_b32 s18, 384 // (op 453) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:55], a[0:3], v12, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 256) 
buffer_load_dwordx4 v60, s[40:43], s18 offen offset: 0 lds // (op 453) Load value direct to lds




v_add_co_u32 v60, s[38:39], v60, v56 // (op 453) least significant half
v_addc_co_u32 v61, s[38:39], 0, v61, s[38:39] // (op 453) most significant half

s_add_u32 m0, m0, 1024 // (op 453) 
buffer_load_dwordx4 v60, s[40:43], s18 offen offset: 0 lds // (op 453) Load value direct to lds








v_add_co_u32 v18, vcc, 256, v18 // (op 513) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 513) most significant half
buffer_load_ubyte v22, v41, s[48:51], 0 offen offset:12 // (op 261) Load value

s_barrier  // (op 1148) 
v_add_u32 v13, 8, v13 // (op 641) 



s_waitcnt vmcnt(11)
ds_write_b128 v16, v[24:27] offset:4096 // (op 13) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v16, v[28:29] offset:4112 // (op 13) Store local data 
s_barrier  // (op 1176) 



s_waitcnt vmcnt(9)
ds_write_b8 v37, v20 offset:5632 // (op 30) Store local data 
s_barrier  // (op 1162) 



s_waitcnt vmcnt(6)
ds_write_b8 v36, v21 offset:5696 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[28:29], 7 // (op 60) 
s_mov_b32 s15, s18 // (op 60) convert

s_and_b32 s18, s15, 4294967294 // (op 60) 

s_cmp_lt_u32 s14, s18 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60:
















s_barrier  // (op 1223) 
v_lshrrev_b32 v12, 3, v1 // (op 476) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 476) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 476) 

s_lshr_b32 s14, s0, 1 // (op 476) 

v_and_b32 v12, -8, v1 // (op 476) 
v_sub_u32 v15, v1, v12 // (op 476) 

v_lshlrev_b32 v12, 4, v15 // (op 476) 

v_lshl_add_u32 v15, s14, 8, v12 // (op 476) 


v_mov_b32 v17, 0 // (op 476) convert
v_mov_b32 v16, v15 // (op 476) convert

v_mul_lo_u32 v19, v13, s3 // (op 476) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 476) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 476) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 476) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 476) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 476) most significant half



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

v_and_b32 v12, -4, v1 // (op 543) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 543) 
v_sub_u32 v13, v1, v12 // (op 543) 

s_lshr_b32 s14, s0, 1 // (op 543) 

v_lshlrev_b32 v12, 5, v13 // (op 543) 

v_lshl_add_u32 v13, s14, 8, v12 // (op 543) 


v_mov_b32 v17, 0 // (op 543) convert
v_mov_b32 v16, v13 // (op 543) convert

v_lshrrev_b32 v15, 2, v1 // (op 543) 
v_lshl_add_u32 v12, s17, 4, v15 // (op 543) 

v_mul_lo_u32 v21, v12, s9 // (op 543) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 543) most significant: high of low * low
v_mul_lo_u32 v20, v12, s8 // (op 543) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 543) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v20 // (op 543) least significant half
v_addc_co_u32 v13, s[0:1], v17, v21, s[0:1] // (op 543) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 543) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 543) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 543) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 543) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 543) 

v_mov_b32 v15, v12 // (op 543) convert

s_mul_i32 s1, s13, 6 // (op 545) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 545) least significant
s_mul_hi_u32 s14, s12, 6 // (op 545) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 545) most significant: sum

s_mul_i32 s15, s8, s1 // (op 545) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 545) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 545) least significant
s_mul_hi_u32 s19, s8, s0 // (op 545) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 545) most significant: sum
s_add_u32 s15, s15, s19 // (op 545) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 545) 

s_mov_b32 s44, s10 // (op 545) reg expression
s_mov_b32 s45, s11 // (op 545) reg expression
s_mov_b32 s46, s0 // (op 545) reg expression
s_mov_b32 s47, 131072 // (op 545) call()

v_lshrrev_b32 v12, 2, v1 // (op 607) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 607) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 607) 

s_lshr_b32 s14, s0, 1 // (op 607) 

v_and_b32 v12, -4, v1 // (op 607) 
v_sub_u32 v16, v1, v12 // (op 607) 

v_lshl_add_u32 v12, s14, 3, v16 // (op 607) 


v_mul_lo_u32 v16, v13, s20 // (op 607) 
v_add_u32 v16, v16, v12 // (op 607) 


s_mul_i32 s1, s20, s7 // (op 609) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 609) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 609) least significant
s_mul_hi_u32 s15, s20, s6 // (op 609) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 609) most significant: sum
s_add_u32 s1, s1, s15 // (op 609) most significant: sum


s_mov_b32 s48, s22 // (op 609) reg expression
s_mov_b32 s49, s23 // (op 609) reg expression
s_mov_b32 s50, s0 // (op 609) reg expression
s_mov_b32 s51, 131072 // (op 609) call()

v_and_b32 v17, -4, v1 // (op 671) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 671) 
v_sub_u32 v12, v1, v17 // (op 671) 

s_lshr_b32 s14, s0, 1 // (op 671) 

v_lshrrev_b32 v13, 2, v1 // (op 671) 
v_lshl_add_u32 v17, s14, 3, v12 // (op 671) 


v_lshl_add_u32 v12, s17, 4, v13 // (op 671) 

v_mul_lo_u32 v13, v12, s24 // (op 671) 

v_add_u32 v12, v17, v13 // (op 671) 


s_mul_i32 s1, s24, s13 // (op 673) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 673) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 673) least significant
s_mul_hi_u32 s15, s24, s12 // (op 673) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 673) most significant: sum
s_add_u32 s1, s1, s15 // (op 673) most significant: sum


s_mov_b32 s52, s26 // (op 673) reg expression
s_mov_b32 s53, s27 // (op 673) reg expression
s_mov_b32 s54, s0 // (op 673) reg expression
s_mov_b32 s55, 131072 // (op 673) call()

v_and_b32 v13, -8, v1 // (op 764) 
v_sub_u32 v17, v1, v13 // (op 764) 

v_lshlrev_b32 v13, 4, v17 // (op 764) 

v_lshrrev_b32 v17, 3, v1 // (op 764) 
v_lshl_add_u32 v20, v17, 7, v13 // (op 764) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 764) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -4, v1 // (op 816) 
v_sub_u32 v17, v1, v13 // (op 816) 

v_lshlrev_b32 v13, 5, v17 // (op 816) 

v_lshrrev_b32 v17, 2, v1 // (op 816) 
v_lshl_add_u32 v20, v17, 7, v13 // (op 816) 


v_mul_lo_u32 v13, v20, 6 // (op 816) 

v_lshrrev_b32 v17, 3, v13 // (op 816) 

v_and_b32 v13, -4, v1 // (op 869) 
v_sub_u32 v20, v1, v13 // (op 869) 

v_and_b32 v13, 4294967292, v1 // (op 869) 
v_add_u32 v21, v13, v20 // (op 869) 


v_and_b32 v13, -4, v1 // (op 922) 
v_sub_u32 v20, v1, v13 // (op 922) 

v_and_b32 v13, 4294967292, v1 // (op 922) 
v_add_u32 v23, v13, v20 // (op 922) 


s_ashr_i64 s[14:15], s[28:29], 7 // (op 176) 
s_and_b32 s1, s14, 4294967294 // (op 176) 

s_ashr_i64 s[14:15], s[28:29], 7 // (op 175) 
s_mov_b32 s18, s14 // (op 175) convert

s_cmp_lt_u32 s1, s18 // (op 175) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_175 // (op 175) Condition: Top (jump to RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_175 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_175:
s_barrier  // (op 1138) 
s_add_u32 s1, s1, 1 // (op 177) 

s_lshl_b64 s[14:15], s[2:3], 3 // (op 447) 
v_mov_b64 v[24:25], s[14:15] // (op 447) convert


v_mov_b64 v[26:27], v[18:19] // (op 447) 
v_mov_b64 v[28:29], v[26:27] // (op 447) 
s_mov_b32 m0, s0 // (op 447) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 447) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v24 // (op 447) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 447) most significant half

s_add_u32 m0, m0, 1024 // (op 447) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 447) Load value direct to lds



v_add_co_u32 v18, vcc, 128, v18 // (op 466) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 466) most significant half


v_mov_b32 v20, v15 // (op 194) 

buffer_load_dwordx4 v[24:27], v20, s[44:47], 0 offen // (op 194) Load value
buffer_load_dwordx2 v[28:29], v20, s[44:47], 0 offen offset:16 // (op 194) Load value

v_add_u32 v15, 96, v15 // (op 535) 


s_barrier  // (op 1151) 
v_mov_b32 v20, v16 // (op 195) 




buffer_load_ubyte v22, v20, s[48:51], 0 offen offset: 0 // (op 195) Load value

v_add_u32 v16, 4, v16 // (op 599) 
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[24:27] offset:4096 // (op 191) Store local data 


v_mov_b32 v20, v12 // (op 196) 

buffer_load_ubyte v13, v20, s[52:55], 0 offen offset: 0 // (op 196) Load value

v_add_u32 v12, 4, v12 // (op 663) 
s_waitcnt vmcnt(2)
ds_write_b64 v17, v[28:29] offset:4112 // (op 191) Store local data 

s_barrier  // (op 1141) 



s_barrier  // (op 1165) 
ds_read_b128 v[24:27], v3 // (op 183) Load local data 



ds_read_b128 v[28:31], v3 offset:64 // (op 183) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v21, v22 offset:5696 // (op 192) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1155) 



s_barrier  // (op 1179) 
ds_read_b128 v[32:35], v4 offset:4096 // (op 185) Load local data 



ds_read_b64 v[36:37], v4 offset:4112 // (op 185) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v13 offset:5632 // (op 193) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1169) 



ds_read_u8 v20, v0 offset:5696 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1183) 



ds_read_u8 v22, v6 offset:5632 // (op 189) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[32:37], a[0:3], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 190) 




s_ashr_i64 s[14:15], s[28:29], 7 // (op 175) 
s_mov_b32 s18, s14 // (op 175) convert

s_cmp_lt_u32 s1, s18 // (op 175) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_175 // (op 175) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_175 if true)
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_175:























s_barrier  // (op 1257) 








v_cmp_eq_f32 vcc, 0.00000, v9 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v11, v4 // (op 50) 
v_mul_f32 v13, v11, v5 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 

v_cvt_f16_f32 v4, v12 // (op 56) 
v_cvt_f16_f32 v5, v13 // (op 56) 
v_cvt_f16_f32 v6, v14 // (op 56) 
v_cvt_f16_f32 v7, v15 // (op 56) 

v_lshrrev_b32 v0, 6, v1 // (op 964) 
v_and_b32 v3, 63, v1 // (op 964) 
v_lshrrev_b32 v8, 4, v3 // (op 964) 
v_lshlrev_b32 v10, 2, v8 // (op 964) 

v_lshl_add_u32 v8, v0, 4, v10 // (op 964) 


v_lshl_add_u32 v0, s16, 4, v8 // (op 964) 

v_lshrrev_b32 v8, 2, v3 // (op 964) 

v_and_b32 v3, 3, v8 // (op 964) 

v_and_b32 v8, 3, v1 // (op 964) 
v_lshl_add_u32 v10, v3, 2, v8 // (op 964) 


v_lshl_add_u32 v3, s17, 4, v10 // (op 964) 

v_mul_lo_u32 v8, v3, s30 // (op 964) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 964) 


s_mul_i32 s15, s30, s13 // (op 966) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 966) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 966) least significant
s_mul_hi_u32 s1, s30, s12 // (op 966) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 966) most significant: sum
s_add_u32 s15, s15, s1 // (op 966) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 966) 

s_mov_b32 s8, s32 // (op 966) reg expression
s_mov_b32 s9, s33 // (op 966) reg expression
s_mov_b32 s10, s0 // (op 966) reg expression
s_mov_b32 s11, 131072 // (op 966) call()


v_pack_b32_f16 v12, v4, v5 // (op 58) 
v_pack_b32_f16 v13, v6, v7 // (op 58) 


buffer_store_dwordx2 v[12:13], v3, s[8:11], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135:
v_lshrrev_b32 v0, 6, v1 // (op 734) 
v_and_b32 v10, 63, v1 // (op 734) 
v_lshrrev_b32 v3, 4, v10 // (op 734) 
v_lshlrev_b32 v4, 2, v3 // (op 734) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 734) 


v_lshl_add_u32 v0, s16, 4, v3 // (op 734) 

v_lshrrev_b32 v3, 2, v10 // (op 734) 

v_and_b32 v10, 3, v3 // (op 734) 

v_and_b32 v3, 3, v1 // (op 734) 
v_lshl_add_u32 v4, v10, 2, v3 // (op 734) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 734) 

v_mul_lo_u32 v10, v3, s34 // (op 734) 

v_add_lshl_u32 v3, v0, v10, 1 // (op 734) 


s_mul_i32 s15, s34, s13 // (op 736) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 736) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 736) least significant
s_mul_hi_u32 s1, s34, s12 // (op 736) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 736) most significant: sum
s_add_u32 s15, s15, s1 // (op 736) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 736) 

s_mov_b32 s4, s36 // (op 736) reg expression
s_mov_b32 s5, s37 // (op 736) reg expression
s_mov_b32 s6, s0 // (op 736) reg expression
s_mov_b32 s7, 131072 // (op 736) call()





buffer_load_dwordx2 v[4:5], v3, s[4:7], 0 offen // (op 148) Load value



v_accvgpr_read v12, a0 // (op 144) 
v_accvgpr_read v13, a1 // (op 144) 
v_accvgpr_read v14, a2 // (op 144) 
v_accvgpr_read v15, a3 // (op 144) 
v_mul_f32 v16, v11, v12 // (op 144) 
v_mul_f32 v17, v11, v13 // (op 144) 
v_mul_f32 v18, v11, v14 // (op 144) 
v_mul_f32 v19, v11, v15 // (op 144) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 149) 
v_lshrrev_b32 v7, 16, v4 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v12, v9, v6 // (op 149) 
v_mul_f32 v13, v9, v7 // (op 149) 
v_cvt_f32_f16 v6, v5 // (op 149) 
v_lshrrev_b32 v7, 16, v5 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v14, v9, v6 // (op 149) 
v_mul_f32 v15, v9, v7 // (op 149) 


v_add_f32 v4, v12, v16 // (op 145) 
v_add_f32 v5, v13, v17 // (op 145) 
v_add_f32 v6, v14, v18 // (op 145) 
v_add_f32 v7, v15, v19 // (op 145) 


v_cvt_f16_f32 v12, v4 // (op 146) 
v_cvt_f16_f32 v13, v5 // (op 146) 
v_cvt_f16_f32 v14, v6 // (op 146) 
v_cvt_f16_f32 v15, v7 // (op 146) 

v_lshrrev_b32 v0, 6, v1 // (op 981) 
v_and_b32 v10, 63, v1 // (op 981) 
v_lshrrev_b32 v3, 4, v10 // (op 981) 
v_lshlrev_b32 v4, 2, v3 // (op 981) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 981) 


v_lshl_add_u32 v0, s16, 4, v3 // (op 981) 

v_lshrrev_b32 v3, 2, v10 // (op 981) 

v_and_b32 v10, 3, v3 // (op 981) 

v_and_b32 v3, 3, v1 // (op 981) 
v_lshl_add_u32 v4, v10, 2, v3 // (op 981) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 981) 

v_mul_lo_u32 v10, v3, s30 // (op 981) 

v_add_lshl_u32 v3, v0, v10, 1 // (op 981) 


s_mul_i32 s15, s30, s13 // (op 983) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 983) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 983) least significant
s_mul_hi_u32 s1, s30, s12 // (op 983) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 983) most significant: sum
s_add_u32 s15, s15, s1 // (op 983) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 983) 

s_mov_b32 s8, s32 // (op 983) reg expression
s_mov_b32 s9, s33 // (op 983) reg expression
s_mov_b32 s10, s0 // (op 983) reg expression
s_mov_b32 s11, 131072 // (op 983) call()


v_pack_b32_f16 v4, v12, v13 // (op 147) 
v_pack_b32_f16 v5, v14, v15 // (op 147) 


buffer_store_dwordx2 v[4:5], v3, s[8:11], 0 offen // (op 147) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135:




s_endpgm  // End of RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.LRR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_end:
.size RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2, .LRR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_end-RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
  .amdhsa_next_free_vgpr 68
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 7424 // lds bytes
  .amdhsa_accum_offset 64
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: F6
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
    macrotile: [16, 16, 128]
    threads: [64, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
    .symbol: RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 7424
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 63
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .workgroup_size: [64, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
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

