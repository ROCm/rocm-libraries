// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.p2align 8
.type RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2,@function
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 747) 
v_and_b32 v3, 15, v1 // (op 747) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 747) 


v_and_b32 v0, 48, v1 // (op 747) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 747) 


v_and_b32 v0, 63, v1 // (op 800) 
v_and_b32 v4, 48, v1 // (op 800) 
v_sub_u32 v5, v0, v4 // (op 800) 

v_lshrrev_b32 v4, 4, v0 // (op 800) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 800) 


v_lshrrev_b32 v4, 6, v1 // (op 800) 
v_and_b32 v5, 1, v4 // (op 800) 

v_lshlrev_b32 v4, 4, v5 // (op 800) 

v_lshl_add_u32 v5, v0, 5, v4 // (op 800) 


v_lshlrev_b32 v0, 2, v5 // (op 800) 

v_lshrrev_b32 v4, 3, v0 // (op 800) 

v_lshrrev_b32 v0, 6, v1 // (op 853) 
v_and_b32 v5, 1, v0 // (op 853) 

v_and_b32 v0, 15, v1 // (op 853) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 853) 


v_and_b32 v0, 63, v1 // (op 853) 
v_lshrrev_b32 v5, 4, v0 // (op 853) 

v_lshl_add_u32 v0, v5, 5, v6 // (op 853) 


v_lshrrev_b32 v5, 7, v1 // (op 911) 
v_and_b32 v6, 15, v1 // (op 911) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 911) 


v_and_b32 v5, 48, v1 // (op 911) 
v_lshl_add_u32 v6, v7, 7, v5 // (op 911) 


v_and_b32 v5, 63, v1 // (op 935) 
v_and_b32 v7, 48, v1 // (op 935) 
v_sub_u32 v8, v5, v7 // (op 935) 

v_lshrrev_b32 v7, 4, v5 // (op 935) 

v_lshl_add_u32 v5, v7, 5, v8 // (op 935) 


v_lshrrev_b32 v7, 6, v1 // (op 935) 
v_and_b32 v8, 1, v7 // (op 935) 

v_lshlrev_b32 v7, 4, v8 // (op 935) 

v_lshl_add_u32 v8, v5, 5, v7 // (op 935) 


v_lshlrev_b32 v5, 2, v8 // (op 935) 

v_lshrrev_b32 v7, 3, v5 // (op 935) 

v_lshrrev_b32 v5, 6, v1 // (op 959) 
v_and_b32 v8, 1, v5 // (op 959) 

v_and_b32 v5, 15, v1 // (op 959) 
v_lshl_add_u32 v9, v8, 4, v5 // (op 959) 


v_and_b32 v5, 63, v1 // (op 959) 
v_lshrrev_b32 v8, 4, v5 // (op 959) 

v_lshl_add_u32 v5, v8, 5, v9 // (op 959) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_lshrrev_b32 v10, 3, v1 // (op 457) 
v_lshl_add_u32 v11, s16, 4, v10 // (op 457) 

v_and_b32 v10, -8, v1 // (op 457) 
v_sub_u32 v12, v1, v10 // (op 457) 

v_lshlrev_b32 v10, 4, v12 // (op 457) 

v_mul_lo_u32 v12, v11, s2 // (op 457) 
v_add_u32 v12, v12, v10 // (op 457) 


s_mul_i32 s1, s2, s7 // (op 459) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 459) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 459) least significant
s_mul_hi_u32 s15, s2, s6 // (op 459) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 459) most significant: sum
s_add_u32 s1, s1, s15 // (op 459) most significant: sum


s_mov_b32 s40, s4 // (op 459) reg expression
s_mov_b32 s41, s5 // (op 459) reg expression
s_mov_b32 s42, s0 // (op 459) reg expression
s_mov_b32 s43, 131072 // (op 459) call()

v_lshrrev_b32 v10, 3, v1 // (op 476) 
v_lshl_add_u32 v11, s16, 4, v10 // (op 476) 

v_and_b32 v10, -8, v1 // (op 476) 
v_sub_u32 v13, v1, v10 // (op 476) 

v_lshlrev_b32 v10, 4, v13 // (op 476) 

v_mul_lo_u32 v13, v11, s2 // (op 476) 
v_add_u32 v13, v13, v10 // (op 476) 


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

s_lshl_b32 s0, s17, 5 // (op 521) 
s_mov_b32 s15, 0 // (op 521) convert
s_mov_b32 s14, s0 // (op 521) convert

v_mul_lo_u32 v11, v1, s9 // (op 521) most significant: low of low * high
v_mul_hi_u32 v14, v1, s8 // (op 521) most significant: high of low * low
v_mul_lo_u32 v10, v1, s8 // (op 521) least significant: low of low * low
v_add_u32 v11, v11, v14 // (op 521) most significant: sum

v_mov_b32 v14, s14 // (op 521) 
v_mov_b32 v15, s15 // (op 521) 
v_add_co_u32 v10, s[0:1], v10, v14 // (op 521) least significant half
v_addc_co_u32 v11, s[0:1], v11, v15, s[0:1] // (op 521) most significant half




v_lshlrev_b64 v[14:15], 2, v[10:11] // (op 521) 

v_lshrrev_b64 v[10:11], 3, v[14:15] // (op 521) 

v_mov_b32 v14, v10 // (op 521) convert

s_mul_i32 s1, s8, s13 // (op 523) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 523) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 523) least significant
s_mul_hi_u32 s15, s8, s12 // (op 523) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 523) most significant: sum
s_add_u32 s1, s1, s15 // (op 523) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 523) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 523) 

s_mov_b32 s44, s10 // (op 523) reg expression
s_mov_b32 s45, s11 // (op 523) reg expression
s_mov_b32 s46, s0 // (op 523) reg expression
s_mov_b32 s47, 131072 // (op 523) call()

s_lshl_b32 s0, s17, 5 // (op 540) 
s_mov_b32 s15, 0 // (op 540) convert
s_mov_b32 s14, s0 // (op 540) convert

v_mul_lo_u32 v11, v1, s9 // (op 540) most significant: low of low * high
v_mul_hi_u32 v15, v1, s8 // (op 540) most significant: high of low * low
v_mul_lo_u32 v10, v1, s8 // (op 540) least significant: low of low * low
v_add_u32 v11, v11, v15 // (op 540) most significant: sum

v_mov_b32 v15, s14 // (op 540) 
v_mov_b32 v16, s15 // (op 540) 
v_add_co_u32 v10, s[0:1], v10, v15 // (op 540) least significant half
v_addc_co_u32 v11, s[0:1], v11, v16, s[0:1] // (op 540) most significant half




v_lshlrev_b64 v[16:17], 2, v[10:11] // (op 540) 

v_lshrrev_b64 v[10:11], 3, v[16:17] // (op 540) 

v_mov_b32 v15, v10 // (op 540) convert

s_mul_i32 s1, s8, s13 // (op 542) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 542) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 542) least significant
s_mul_hi_u32 s15, s8, s12 // (op 542) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 542) most significant: sum
s_add_u32 s1, s1, s15 // (op 542) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 542) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 542) 

s_mov_b32 s44, s10 // (op 542) reg expression
s_mov_b32 s45, s11 // (op 542) reg expression
s_mov_b32 s46, s0 // (op 542) reg expression
s_mov_b32 s47, 131072 // (op 542) call()

v_lshrrev_b32 v10, 7, v1 // (op 585) 
v_and_b32 v11, 15, v1 // (op 585) 
v_lshl_add_u32 v16, v10, 4, v11 // (op 585) 


v_lshl_add_u32 v10, s16, 4, v16 // (op 585) 

v_and_b32 v11, 63, v1 // (op 585) 
v_lshrrev_b32 v16, 4, v11 // (op 585) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v11, v10, s20 // (op 585) 
v_add_u32 v11, v11, v16 // (op 585) 


s_mul_i32 s1, s20, s7 // (op 587) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 587) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 587) least significant
s_mul_hi_u32 s15, s20, s6 // (op 587) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 587) most significant: sum
s_add_u32 s1, s1, s15 // (op 587) most significant: sum


s_mov_b32 s48, s22 // (op 587) reg expression
s_mov_b32 s49, s23 // (op 587) reg expression
s_mov_b32 s50, s0 // (op 587) reg expression
s_mov_b32 s51, 131072 // (op 587) call()

v_lshrrev_b32 v10, 7, v1 // (op 604) 
v_and_b32 v16, 15, v1 // (op 604) 
v_lshl_add_u32 v17, v10, 4, v16 // (op 604) 


v_lshl_add_u32 v10, s16, 4, v17 // (op 604) 

v_and_b32 v16, 63, v1 // (op 604) 
v_lshrrev_b32 v17, 4, v16 // (op 604) 

v_mul_lo_u32 v16, v10, s20 // (op 604) 
v_add_u32 v16, v16, v17 // (op 604) 


s_mul_i32 s1, s20, s7 // (op 606) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 606) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 606) least significant
s_mul_hi_u32 s15, s20, s6 // (op 606) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 606) most significant: sum
s_add_u32 s1, s1, s15 // (op 606) most significant: sum


s_mov_b32 s48, s22 // (op 606) reg expression
s_mov_b32 s49, s23 // (op 606) reg expression
s_mov_b32 s50, s0 // (op 606) reg expression
s_mov_b32 s51, 131072 // (op 606) call()

v_and_b32 v10, -32, v1 // (op 649) 
v_sub_u32 v17, v1, v10 // (op 649) 

v_lshl_add_u32 v10, s17, 5, v17 // (op 649) 

v_lshrrev_b32 v17, 5, v1 // (op 649) 
v_mul_lo_u32 v18, v17, s24 // (op 649) 
v_add_u32 v18, v18, v10 // (op 649) 


s_mul_i32 s1, s24, s29 // (op 651) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 651) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 651) least significant
s_mul_hi_u32 s15, s24, s28 // (op 651) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 651) most significant: sum
s_add_u32 s1, s1, s15 // (op 651) most significant: sum


s_mov_b32 s52, s26 // (op 651) reg expression
s_mov_b32 s53, s27 // (op 651) reg expression
s_mov_b32 s54, s0 // (op 651) reg expression
s_mov_b32 s55, 131072 // (op 651) call()

v_and_b32 v10, -32, v1 // (op 668) 
v_sub_u32 v17, v1, v10 // (op 668) 

v_lshl_add_u32 v10, s17, 5, v17 // (op 668) 

v_lshrrev_b32 v17, 5, v1 // (op 668) 
v_mul_lo_u32 v19, v17, s24 // (op 668) 
v_add_u32 v19, v19, v10 // (op 668) 


s_mul_i32 s1, s24, s29 // (op 670) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 670) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 670) least significant
s_mul_hi_u32 s15, s24, s28 // (op 670) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 670) most significant: sum
s_add_u32 s1, s1, s15 // (op 670) most significant: sum


s_mov_b32 s52, s26 // (op 670) reg expression
s_mov_b32 s53, s27 // (op 670) reg expression
s_mov_b32 s54, s0 // (op 670) reg expression
s_mov_b32 s55, 131072 // (op 670) call()




buffer_load_ubyte v10, v11, s[48:51], 0 offen offset: 0 // (op 341) Load value


v_mov_b32 v17, v14 // (op 345) reg expression

buffer_load_dwordx4 v[20:23], v17, s[44:47], 0 offen // (op 345) Load value



v_mov_b32 v17, v18 // (op 348) reg expression

buffer_load_ubyte v24, v17, s[52:55], 0 offen offset: 0 // (op 348) Load value




buffer_load_dwordx4 v[26:29], v12, s[40:43], 0 offen // (op 351) Load value



buffer_load_ubyte v17, v11, s[48:51], 0 offen offset:4 // (op 355) Load value



s_lshl_b64 s[0:1], s[8:9], 7 // (op 359) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 359) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 359) 

v_add_u32 v11, v14, s0 // (op 359) 


buffer_load_dwordx4 v[30:33], v11, s[44:47], 0 offen // (op 359) Load value




s_lshl_b32 s0, s24, 2 // (op 362) 
v_add_u32 v11, v18, s0 // (op 362) 


buffer_load_ubyte v14, v11, s[52:55], 0 offen offset: 0 // (op 362) Load value





buffer_load_dwordx4 v[34:37], v12, s[40:43], 0 offen offset:128 // (op 365) Load value

v_lshlrev_b32 v18, 7, v1 // (op 788) 
v_lshrrev_b32 v25, 3, v18 // (op 788) 




s_waitcnt vmcnt(6)
ds_write_b128 v25, v[20:23] // (op 367) Store local data 

v_and_b32 v18, -32, v1 // (op 841) 
v_sub_u32 v25, v1, v18 // (op 841) 

v_and_b32 v18, 4294967264, v1 // (op 841) 
v_add_u32 v11, v18, v25 // (op 841) 





s_waitcnt vmcnt(5)
ds_write_b8 v11, v24 offset:2048 // (op 368) Store local data 

v_and_b32 v18, -8, v1 // (op 735) 
v_sub_u32 v25, v1, v18 // (op 735) 

v_lshlrev_b32 v18, 4, v25 // (op 735) 

v_lshrrev_b32 v25, 3, v1 // (op 735) 
v_lshl_add_u32 v11, v25, 7, v18 // (op 735) 





s_waitcnt vmcnt(4)
ds_write_b128 v11, v[26:29] offset:2176 // (op 369) Store local data 

v_and_b32 v18, -8, v1 // (op 706) 
v_sub_u32 v25, v1, v18 // (op 706) 

v_lshlrev_b32 v18, 4, v25 // (op 706) 

v_lshrrev_b32 v25, 3, v1 // (op 706) 
v_lshl_add_u32 v11, v25, 7, v18 // (op 706) 


v_lshlrev_b32 v12, 7, v1 // (op 759) 
v_lshrrev_b32 v18, 3, v12 // (op 759) 

v_and_b32 v12, -32, v1 // (op 812) 
v_sub_u32 v25, v1, v12 // (op 812) 

v_and_b32 v12, 4294967264, v1 // (op 812) 
v_add_u32 v38, v12, v25 // (op 812) 


v_and_b32 v12, -8, v1 // (op 899) 
v_sub_u32 v25, v1, v12 // (op 899) 

v_lshlrev_b32 v12, 4, v25 // (op 899) 

v_lshrrev_b32 v25, 3, v1 // (op 899) 
v_lshl_add_u32 v39, v25, 7, v12 // (op 899) 


v_lshlrev_b32 v12, 7, v1 // (op 923) 
v_lshrrev_b32 v25, 3, v12 // (op 923) 

v_and_b32 v12, -32, v1 // (op 947) 
v_sub_u32 v40, v1, v12 // (op 947) 

v_and_b32 v12, 4294967264, v1 // (op 947) 
v_add_u32 v41, v12, v40 // (op 947) 


s_mov_b32 s0, 0 // (op 57) call()
s_ashr_i64 s[14:15], s[12:13], 7 // (op 56) 
s_mov_b32 s1, s14 // (op 56) convert

s_and_b32 s14, s1, 4294967294 // (op 56) 

s_cmp_lt_u32 s0, s14 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56 if false)
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56:
s_add_u32 s0, s0, 2 // (op 58) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 992) 



ds_read_b128 v[42:45], v3 offset:2176 // (op 67) Load local data 
ds_read_b128 v[46:49], v3 offset:2240 // (op 67) Load local data 
s_barrier  // (op 1005) 



ds_read_b64_tr_b4 v[50:51], v4 // (op 72) Transpose load local data 
ds_read_b64_tr_b4 v[52:53], v4 offset:256 // (op 72) Transpose load local data 
s_barrier  // (op 1019) 


ds_read_u8 v12, v0 offset:2048 // (op 81) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:53], a[0:3], v10, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 84) 





v_mov_b32 v12, v16 // (op 77) 

buffer_load_ubyte v10, v12, s[48:51], 0 offen offset:8 // (op 77) Load value


v_mov_b32 v40, v15 // (op 10) 
s_lshl_b64 s[14:15], s[8:9], 7 // (op 10) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 10) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s1, s14, 1 // (op 10) 

v_add_u32 v42, v40, s1 // (op 10) 


buffer_load_dwordx4 v[20:23], v42, s[44:47], 0 offen // (op 10) Load value



v_mov_b32 v42, v19 // (op 21) 
s_lshl_b32 s1, s24, 3 // (op 21) 
v_add_u32 v43, v42, s1 // (op 21) 


buffer_load_ubyte v24, v43, s[52:55], 0 offen offset: 0 // (op 21) Load value



v_mov_b32 v43, v13 // (op 4) 

buffer_load_dwordx4 v[26:29], v43, s[40:43], 0 offen offset:256 // (op 4) Load value
s_barrier  // (op 1042) 



s_waitcnt vmcnt(6)
ds_write_b128 v25, v[30:33] offset:4224 // (op 239) Store local data 
s_barrier  // (op 1048) 



s_waitcnt vmcnt(5)
ds_write_b8 v41, v14 offset:6272 // (op 240) Store local data 
s_barrier  // (op 1036) 



s_waitcnt vmcnt(4)
ds_write_b128 v39, v[34:37] offset:6400 // (op 229) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1033) 



ds_read_b128 v[44:47], v6 offset:6400 // (op 231) Load local data 
ds_read_b128 v[48:51], v6 offset:6464 // (op 231) Load local data 
s_barrier  // (op 1039) 
s_barrier  // (op 1103) 



ds_read_b64_tr_b4 v[52:53], v7 offset:4224 // (op 233) Transpose load local data 
ds_read_b64_tr_b4 v[54:55], v7 offset:4480 // (op 233) Transpose load local data 
s_barrier  // (op 1045) 
s_barrier  // (op 1109) 


ds_read_u8 v56, v5 offset:6272 // (op 237) Load local data 
s_barrier  // (op 1117) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[52:55], a[0:3], v17, v56 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 238) 






buffer_load_ubyte v17, v12, s[48:51], 0 offen offset:12 // (op 235) Load value



v_add_u32 v16, 8, v16 // (op 596) 
s_lshl_b64 s[14:15], s[8:9], 7 // (op 241) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 241) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 241) 

s_mul_i32 s1, 3, s14 // (op 241) 

v_add_u32 v12, v40, s1 // (op 241) 


buffer_load_dwordx4 v[30:33], v12, s[44:47], 0 offen // (op 241) Load value




s_lshl_b32 s1, s24, 2 // (op 242) 
s_mul_i32 s14, 3, s1 // (op 242) 

v_add_u32 v12, v42, s14 // (op 242) 


s_lshl_b64 s[14:15], s[8:9], 7 // (op 532) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 532) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 532) 

s_lshl_b32 s1, s14, 1 // (op 532) 

v_add_u32 v15, v15, s1 // (op 532) 

buffer_load_ubyte v14, v12, s[52:55], 0 offen offset: 0 // (op 242) Load value





s_lshl_b32 s1, s24, 3 // (op 660) 
v_add_u32 v19, v19, s1 // (op 660) 

buffer_load_dwordx4 v[34:37], v43, s[40:43], 0 offen offset:384 // (op 228) Load value

s_barrier  // (op 1008) 
v_add_u32 v13, 256, v13 // (op 468) 



s_waitcnt vmcnt(6)
ds_write_b128 v18, v[20:23] // (op 13) Store local data 
s_barrier  // (op 1022) 



s_waitcnt vmcnt(5)
ds_write_b8 v38, v24 offset:2048 // (op 26) Store local data 
s_barrier  // (op 995) 



s_waitcnt vmcnt(4)
ds_write_b128 v11, v[26:29] offset:2176 // (op 6) Store local data 
s_ashr_i64 s[14:15], s[12:13], 7 // (op 56) 
s_mov_b32 s1, s14 // (op 56) convert

s_and_b32 s14, s1, 4294967294 // (op 56) 

s_cmp_lt_u32 s0, s14 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56 if true)
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56:















v_lshrrev_b32 v10, 3, v1 // (op 431) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 431) 
v_lshl_add_u32 v11, s16, 4, v10 // (op 431) 

s_lshr_b32 s14, s0, 1 // (op 431) 

v_and_b32 v10, -8, v1 // (op 431) 
v_sub_u32 v12, v1, v10 // (op 431) 

v_lshlrev_b32 v10, 4, v12 // (op 431) 

v_lshl_add_u32 v12, s14, 8, v10 // (op 431) 


v_mul_lo_u32 v10, v11, s2 // (op 431) 
v_add_u32 v10, v10, v12 // (op 431) 


s_mul_i32 s1, s2, s7 // (op 433) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 433) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 433) least significant
s_mul_hi_u32 s15, s2, s6 // (op 433) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 433) most significant: sum
s_add_u32 s1, s1, s15 // (op 433) most significant: sum


s_mov_b32 s40, s4 // (op 433) reg expression
s_mov_b32 s41, s5 // (op 433) reg expression
s_mov_b32 s42, s0 // (op 433) reg expression
s_mov_b32 s43, 131072 // (op 433) call()

s_ashr_i64 s[0:1], s[12:13], 7 // (op 498) 
s_lshr_b32 s14, s0, 1 // (op 498) 

v_lshl_add_u32 v11, s14, 8, v1 // (op 498) 

s_lshl_b32 s0, s17, 5 // (op 498) 
s_mov_b32 s15, 0 // (op 498) convert
s_mov_b32 s14, s0 // (op 498) convert

v_mul_lo_u32 v13, v11, s9 // (op 498) most significant: low of low * high
v_mul_hi_u32 v15, v11, s8 // (op 498) most significant: high of low * low
v_mul_lo_u32 v12, v11, s8 // (op 498) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 498) most significant: sum

v_mov_b32 v15, s14 // (op 498) 
v_mov_b32 v16, s15 // (op 498) 
v_add_co_u32 v12, s[0:1], v12, v15 // (op 498) least significant half
v_addc_co_u32 v13, s[0:1], v13, v16, s[0:1] // (op 498) most significant half





v_lshlrev_b64 v[18:19], 2, v[12:13] // (op 498) 

v_lshrrev_b64 v[12:13], 3, v[18:19] // (op 498) 

v_mov_b32 v11, v12 // (op 498) convert

s_mul_i32 s1, s8, s13 // (op 500) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 500) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 500) least significant
s_mul_hi_u32 s15, s8, s12 // (op 500) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 500) most significant: sum
s_add_u32 s1, s1, s15 // (op 500) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 500) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 500) 

s_mov_b32 s44, s10 // (op 500) reg expression
s_mov_b32 s45, s11 // (op 500) reg expression
s_mov_b32 s46, s0 // (op 500) reg expression
s_mov_b32 s47, 131072 // (op 500) call()

v_lshrrev_b32 v12, 7, v1 // (op 562) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 562) 
v_and_b32 v13, 15, v1 // (op 562) 
s_lshr_b32 s14, s0, 1 // (op 562) 

v_lshl_add_u32 v15, v12, 4, v13 // (op 562) 


v_lshl_add_u32 v16, s16, 4, v15 // (op 562) 

v_and_b32 v15, 63, v1 // (op 562) 
v_lshrrev_b32 v12, 4, v15 // (op 562) 

v_lshl_add_u32 v13, s14, 3, v12 // (op 562) 


v_mul_lo_u32 v12, v16, s20 // (op 562) 
v_add_u32 v12, v12, v13 // (op 562) 


s_mul_i32 s1, s20, s7 // (op 564) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 564) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 564) least significant
s_mul_hi_u32 s15, s20, s6 // (op 564) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 564) most significant: sum
s_add_u32 s1, s1, s15 // (op 564) most significant: sum


s_mov_b32 s48, s22 // (op 564) reg expression
s_mov_b32 s49, s23 // (op 564) reg expression
s_mov_b32 s50, s0 // (op 564) reg expression
s_mov_b32 s51, 131072 // (op 564) call()

v_lshrrev_b32 v13, 5, v1 // (op 626) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 626) 
v_and_b32 v15, -32, v1 // (op 626) 
s_lshr_b32 s14, s0, 1 // (op 626) 

v_sub_u32 v16, v1, v15 // (op 626) 

v_lshl_add_u32 v15, s14, 3, v13 // (op 626) 


v_lshl_add_u32 v13, s17, 5, v16 // (op 626) 

v_mul_lo_u32 v16, v15, s24 // (op 626) 
v_add_u32 v16, v16, v13 // (op 626) 


s_mul_i32 s1, s24, s29 // (op 628) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 628) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 628) least significant
s_mul_hi_u32 s15, s24, s28 // (op 628) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 628) most significant: sum
s_add_u32 s1, s1, s15 // (op 628) most significant: sum


s_mov_b32 s52, s26 // (op 628) reg expression
s_mov_b32 s53, s27 // (op 628) reg expression
s_mov_b32 s54, s0 // (op 628) reg expression
s_mov_b32 s55, 131072 // (op 628) call()

v_and_b32 v13, -8, v1 // (op 719) 
v_sub_u32 v15, v1, v13 // (op 719) 

v_lshlrev_b32 v13, 4, v15 // (op 719) 

v_lshrrev_b32 v15, 3, v1 // (op 719) 
v_lshl_add_u32 v18, v15, 7, v13 // (op 719) 


v_lshlrev_b32 v13, 7, v1 // (op 772) 
v_lshrrev_b32 v15, 3, v13 // (op 772) 

v_and_b32 v13, -32, v1 // (op 825) 
v_sub_u32 v19, v1, v13 // (op 825) 

v_and_b32 v13, 4294967264, v1 // (op 825) 
v_add_u32 v20, v13, v19 // (op 825) 


s_ashr_i64 s[0:1], s[12:13], 7 // (op 166) 
s_and_b32 s14, s0, 4294967294 // (op 166) 

s_ashr_i64 s[0:1], s[12:13], 7 // (op 165) 
s_mov_b32 s15, s0 // (op 165) convert

s_cmp_lt_u32 s14, s15 // (op 165) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165:


s_add_u32 s14, s14, 1 // (op 167) 
v_mov_b32 v17, v10 // (op 170) 

buffer_load_dwordx4 v[22:25], v17, s[40:43], 0 offen // (op 170) Load value

v_add_u32 v10, 128, v10 // (op 421) 


s_barrier  // (op 998) 
v_mov_b32 v17, v11 // (op 183) 



v_mov_b32 v19, v17 // (op 183) reg expression

buffer_load_dwordx4 v[26:29], v19, s[44:47], 0 offen // (op 183) Load value


s_lshl_b64 s[0:1], s[8:9], 7 // (op 490) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 490) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 490) 

v_add_u32 v11, v11, s0 // (op 490) 

s_waitcnt vmcnt(1)
ds_write_b128 v18, v[22:25] offset:2176 // (op 171) Store local data 



v_mov_b32 v17, v16 // (op 184) 
s_barrier  // (op 1011) 
v_mov_b32 v19, v17 // (op 184) reg expression




buffer_load_ubyte v13, v19, s[52:55], 0 offen offset: 0 // (op 184) Load value


s_lshl_b32 s15, s24, 2 // (op 618) 
v_add_u32 v16, v16, s15 // (op 618) 

s_waitcnt vmcnt(1)
ds_write_b128 v15, v[26:29] // (op 181) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 1001) 



s_barrier  // (op 1025) 
ds_read_b128 v[22:25], v3 offset:2176 // (op 173) Load local data 



ds_read_b128 v[26:29], v3 offset:2240 // (op 173) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v13 offset:2048 // (op 182) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1015) 



ds_read_b64_tr_b4 v[30:31], v4 // (op 175) Transpose load local data 
ds_read_b64_tr_b4 v[32:33], v4 offset:256 // (op 175) Transpose load local data 


v_mov_b32 v17, v12 // (op 177) 

buffer_load_ubyte v19, v17, s[48:51], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v12, 4, v12 // (op 554) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1029) 


ds_read_u8 v17, v0 offset:2048 // (op 179) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:33], a[0:3], v19, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 180) 




s_ashr_i64 s[0:1], s[12:13], 7 // (op 165) 
s_mov_b32 s15, s0 // (op 165) convert

s_cmp_lt_u32 s14, s15 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165 if true)
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165:























s_barrier  // (op 1097) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125
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

v_lshrrev_b32 v0, 7, v1 // (op 867) 
v_and_b32 v3, 63, v1 // (op 867) 
v_lshrrev_b32 v10, 4, v3 // (op 867) 
v_lshlrev_b32 v11, 2, v10 // (op 867) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 867) 


v_lshl_add_u32 v0, s16, 4, v10 // (op 867) 

v_lshrrev_b32 v10, 2, v3 // (op 867) 

v_and_b32 v3, 3, v10 // (op 867) 

v_and_b32 v10, 3, v1 // (op 867) 
v_lshl_add_u32 v11, v3, 2, v10 // (op 867) 


v_lshrrev_b32 v3, 6, v1 // (op 867) 
v_and_b32 v10, 1, v3 // (op 867) 

v_lshl_add_u32 v3, v10, 4, v11 // (op 867) 


v_lshl_add_u32 v10, s17, 5, v3 // (op 867) 

v_mul_lo_u32 v3, v10, s30 // (op 867) 

v_add_lshl_u32 v10, v0, v3, 1 // (op 867) 


s_mul_i32 s1, s30, s35 // (op 869) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 869) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 869) least significant
s_mul_hi_u32 s3, s30, s34 // (op 869) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 869) most significant: sum
s_add_u32 s1, s1, s3 // (op 869) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 869) 

s_mov_b32 s4, s32 // (op 869) reg expression
s_mov_b32 s5, s33 // (op 869) reg expression
s_mov_b32 s6, s2 // (op 869) reg expression
s_mov_b32 s7, 131072 // (op 869) call()


v_pack_b32_f16 v12, v4, v5 // (op 54) 
v_pack_b32_f16 v13, v6, v7 // (op 54) 


buffer_store_dwordx2 v[12:13], v10, s[4:7], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125:
v_lshrrev_b32 v0, 7, v1 // (op 689) 
v_and_b32 v3, 63, v1 // (op 689) 
v_lshrrev_b32 v4, 4, v3 // (op 689) 
v_lshlrev_b32 v5, 2, v4 // (op 689) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 689) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 689) 

v_lshrrev_b32 v4, 2, v3 // (op 689) 

v_and_b32 v3, 3, v4 // (op 689) 

v_and_b32 v4, 3, v1 // (op 689) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 689) 


v_lshrrev_b32 v3, 6, v1 // (op 689) 
v_and_b32 v4, 1, v3 // (op 689) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 689) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 689) 

v_mul_lo_u32 v3, v4, s36 // (op 689) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 689) 


s_mul_i32 s1, s36, s35 // (op 691) most significant: low of low * high
s_mul_i32 s2, s37, s34 // (op 691) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 691) least significant
s_mul_hi_u32 s3, s36, s34 // (op 691) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 691) most significant: sum
s_add_u32 s1, s1, s3 // (op 691) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 691) 

s_mov_b32 s8, s38 // (op 691) reg expression
s_mov_b32 s9, s39 // (op 691) reg expression
s_mov_b32 s10, s2 // (op 691) reg expression
s_mov_b32 s11, 131072 // (op 691) call()





buffer_load_dwordx2 v[6:7], v4, s[8:11], 0 offen // (op 138) Load value



v_accvgpr_read v10, a0 // (op 134) 
v_accvgpr_read v11, a1 // (op 134) 
v_accvgpr_read v12, a2 // (op 134) 
v_accvgpr_read v13, a3 // (op 134) 
v_mul_f32 v14, v9, v10 // (op 134) 
v_mul_f32 v15, v9, v11 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 

s_waitcnt vmcnt(0)
v_and_b32 v4, 65535, v6 // (op 139) 
v_lshrrev_b32 v5, 16, v6 // (op 139) 
v_lshlrev_b32 v4, 16, v4 // (op 139) 
v_lshlrev_b32 v5, 16, v5 // (op 139) 
v_mul_f32 v10, v8, v4 // (op 139) 
v_mul_f32 v11, v8, v5 // (op 139) 
v_and_b32 v4, 65535, v7 // (op 139) 
v_lshrrev_b32 v5, 16, v7 // (op 139) 
v_lshlrev_b32 v4, 16, v4 // (op 139) 
v_lshlrev_b32 v5, 16, v5 // (op 139) 
v_mul_f32 v12, v8, v4 // (op 139) 
v_mul_f32 v13, v8, v5 // (op 139) 


v_add_f32 v4, v10, v14 // (op 135) 
v_add_f32 v5, v11, v15 // (op 135) 
v_add_f32 v6, v12, v16 // (op 135) 
v_add_f32 v7, v13, v17 // (op 135) 


v_lshrrev_b32 v10, 16, v4 // (op 136) 
v_lshrrev_b32 v11, 16, v5 // (op 136) 
v_lshrrev_b32 v12, 16, v6 // (op 136) 
v_lshrrev_b32 v13, 16, v7 // (op 136) 

v_lshrrev_b32 v0, 7, v1 // (op 884) 
v_and_b32 v3, 63, v1 // (op 884) 
v_lshrrev_b32 v4, 4, v3 // (op 884) 
v_lshlrev_b32 v5, 2, v4 // (op 884) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 884) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 884) 

v_lshrrev_b32 v4, 2, v3 // (op 884) 

v_and_b32 v3, 3, v4 // (op 884) 

v_and_b32 v4, 3, v1 // (op 884) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 884) 


v_lshrrev_b32 v3, 6, v1 // (op 884) 
v_and_b32 v4, 1, v3 // (op 884) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 884) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 884) 

v_mul_lo_u32 v3, v4, s30 // (op 884) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 884) 


s_mul_i32 s1, s30, s35 // (op 886) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 886) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 886) least significant
s_mul_hi_u32 s3, s30, s34 // (op 886) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 886) most significant: sum
s_add_u32 s1, s1, s3 // (op 886) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 886) 

s_mov_b32 s4, s32 // (op 886) reg expression
s_mov_b32 s5, s33 // (op 886) reg expression
s_mov_b32 s6, s2 // (op 886) reg expression
s_mov_b32 s7, 131072 // (op 886) call()


v_pack_b32_f16 v6, v10, v11 // (op 137) 
v_pack_b32_f16 v7, v12, v13 // (op 137) 


buffer_store_dwordx2 v[6:7], v4, s[4:7], 0 offen // (op 137) Store value







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125:



s_endpgm  // End of RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.LRR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end:
.size RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2, .LRR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end-RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 8448 // lds bytes
  .amdhsa_accum_offset 60
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
    DataTypeB: F4
    DestDataType: b
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
  - .name: RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
    .symbol: RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 8448
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 57
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

