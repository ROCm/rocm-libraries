// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.p2align 8
.type RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2,@function
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 769) 
v_and_b32 v3, 48, v1 // (op 769) 
v_sub_u32 v4, v0, v3 // (op 769) 

v_lshrrev_b32 v3, 4, v0 // (op 769) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 769) 


v_lshrrev_b32 v3, 6, v1 // (op 769) 
v_lshlrev_b32 v4, 4, v3 // (op 769) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 769) 


v_lshlrev_b32 v0, 2, v3 // (op 769) 

v_lshrrev_b32 v3, 3, v0 // (op 769) 

v_and_b32 v0, 15, v1 // (op 822) 
v_and_b32 v4, 48, v1 // (op 822) 
v_lshl_add_u32 v5, v0, 7, v4 // (op 822) 


v_lshrrev_b32 v0, 6, v1 // (op 875) 
v_and_b32 v4, 15, v1 // (op 875) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 875) 


v_and_b32 v0, 48, v1 // (op 875) 
v_add_u32 v4, v0, v6 // (op 875) 


v_and_b32 v0, 15, v1 // (op 928) 
v_and_b32 v6, 63, v1 // (op 928) 
v_lshrrev_b32 v7, 4, v6 // (op 928) 

v_lshl_add_u32 v6, v0, 2, v7 // (op 928) 


v_and_b32 v0, 63, v1 // (op 986) 
v_and_b32 v7, 48, v1 // (op 986) 
v_sub_u32 v8, v0, v7 // (op 986) 

v_lshrrev_b32 v7, 4, v0 // (op 986) 

v_lshl_add_u32 v0, v7, 5, v8 // (op 986) 


v_lshrrev_b32 v7, 6, v1 // (op 986) 
v_lshlrev_b32 v8, 4, v7 // (op 986) 

v_lshl_add_u32 v7, v0, 4, v8 // (op 986) 


v_lshlrev_b32 v0, 2, v7 // (op 986) 

v_lshrrev_b32 v7, 3, v0 // (op 986) 

v_and_b32 v0, 15, v1 // (op 1010) 
v_and_b32 v8, 48, v1 // (op 1010) 
v_lshl_add_u32 v9, v0, 7, v8 // (op 1010) 


v_lshrrev_b32 v0, 6, v1 // (op 1034) 
v_and_b32 v8, 15, v1 // (op 1034) 
v_lshl_add_u32 v10, v0, 4, v8 // (op 1034) 


v_and_b32 v0, 48, v1 // (op 1034) 
v_add_u32 v8, v0, v10 // (op 1034) 


v_and_b32 v0, 15, v1 // (op 1058) 
v_and_b32 v10, 63, v1 // (op 1058) 
v_lshrrev_b32 v11, 4, v10 // (op 1058) 

v_lshl_add_u32 v10, v0, 2, v11 // (op 1058) 


v_mov_b32 v0, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
s_lshl_b32 s0, s16, 4 // (op 479) 
v_lshlrev_b32 v12, 1, v1 // (op 479) 
s_mov_b32 s15, 0 // (op 479) convert
s_mov_b32 s14, s0 // (op 479) convert

v_mul_lo_u32 v15, v12, s3 // (op 479) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 479) most significant: high of low * low
v_mul_lo_u32 v14, v12, s2 // (op 479) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 479) most significant: sum


v_mov_b32 v12, s14 // (op 479) 
v_mov_b32 v13, s15 // (op 479) 
v_add_co_u32 v16, s[0:1], v12, v14 // (op 479) least significant half
v_addc_co_u32 v17, s[0:1], v13, v15, s[0:1] // (op 479) most significant half



v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 479) 


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 479) 

v_mov_b32 v14, v12 // (op 479) convert

s_mul_i32 s1, s2, s7 // (op 481) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 481) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 481) least significant
s_mul_hi_u32 s15, s2, s6 // (op 481) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 481) most significant: sum
s_add_u32 s1, s1, s15 // (op 481) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 481) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 481) 

s_mov_b32 s40, s4 // (op 481) reg expression
s_mov_b32 s41, s5 // (op 481) reg expression
s_mov_b32 s42, s0 // (op 481) reg expression
s_mov_b32 s43, 131072 // (op 481) call()

s_lshl_b32 s0, s16, 4 // (op 498) 
v_lshlrev_b32 v12, 1, v1 // (op 498) 
s_mov_b32 s15, 0 // (op 498) convert
s_mov_b32 s14, s0 // (op 498) convert

v_mul_lo_u32 v17, v12, s3 // (op 498) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 498) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 498) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 498) most significant: sum


v_mov_b32 v15, s14 // (op 498) 
v_mov_b32 v12, s15 // (op 498) 
v_add_co_u32 v18, s[0:1], v15, v16 // (op 498) least significant half
v_addc_co_u32 v19, s[0:1], v12, v17, s[0:1] // (op 498) most significant half



v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 498) 


v_lshrrev_b64 v[12:13], 3, v[18:19] // (op 498) 

v_mov_b32 v15, v12 // (op 498) convert

s_mul_i32 s1, s2, s7 // (op 500) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 500) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 500) least significant
s_mul_hi_u32 s15, s2, s6 // (op 500) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 500) most significant: sum
s_add_u32 s1, s1, s15 // (op 500) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 500) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 500) 

s_mov_b32 s40, s4 // (op 500) reg expression
s_mov_b32 s41, s5 // (op 500) reg expression
s_mov_b32 s42, s0 // (op 500) reg expression
s_mov_b32 s43, 131072 // (op 500) call()

v_and_b32 v12, -8, v1 // (op 543) 
v_sub_u32 v13, v1, v12 // (op 543) 

v_lshrrev_b32 v12, 3, v1 // (op 543) 
v_lshlrev_b32 v16, 1, v12 // (op 543) 

v_lshl_add_u32 v12, s17, 4, v16 // (op 543) 

v_mul_lo_u32 v16, v12, s8 // (op 543) 

v_lshl_add_u32 v12, v13, 4, v16 // (op 543) 


s_mul_i32 s1, s8, s13 // (op 545) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 545) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 545) least significant
s_mul_hi_u32 s15, s8, s12 // (op 545) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 545) most significant: sum
s_add_u32 s1, s1, s15 // (op 545) most significant: sum


s_mov_b32 s44, s10 // (op 545) reg expression
s_mov_b32 s45, s11 // (op 545) reg expression
s_mov_b32 s46, s0 // (op 545) reg expression
s_mov_b32 s47, 131072 // (op 545) call()

v_and_b32 v13, -8, v1 // (op 562) 
v_sub_u32 v16, v1, v13 // (op 562) 

v_lshrrev_b32 v13, 3, v1 // (op 562) 
v_lshlrev_b32 v17, 1, v13 // (op 562) 

v_lshl_add_u32 v13, s17, 4, v17 // (op 562) 

v_mul_lo_u32 v17, v13, s8 // (op 562) 

v_lshl_add_u32 v13, v16, 4, v17 // (op 562) 


s_mul_i32 s1, s8, s13 // (op 564) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 564) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 564) least significant
s_mul_hi_u32 s15, s8, s12 // (op 564) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 564) most significant: sum
s_add_u32 s1, s1, s15 // (op 564) most significant: sum


s_mov_b32 s44, s10 // (op 564) reg expression
s_mov_b32 s45, s11 // (op 564) reg expression
s_mov_b32 s46, s0 // (op 564) reg expression
s_mov_b32 s47, 131072 // (op 564) call()

v_and_b32 v16, -16, v1 // (op 607) 
v_sub_u32 v17, v1, v16 // (op 607) 

v_lshl_add_u32 v16, s16, 4, v17 // (op 607) 

v_lshrrev_b32 v17, 4, v1 // (op 607) 

s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v18, v17, s20 // (op 607) 

v_add_u32 v17, v16, v18 // (op 607) 


s_mul_i32 s1, s20, s25 // (op 609) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 609) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 609) least significant
s_mul_hi_u32 s15, s20, s24 // (op 609) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 609) most significant: sum
s_add_u32 s1, s1, s15 // (op 609) most significant: sum


s_mov_b32 s48, s22 // (op 609) reg expression
s_mov_b32 s49, s23 // (op 609) reg expression
s_mov_b32 s50, s0 // (op 609) reg expression
s_mov_b32 s51, 131072 // (op 609) call()

v_and_b32 v16, -16, v1 // (op 626) 
v_sub_u32 v18, v1, v16 // (op 626) 

v_lshl_add_u32 v16, s16, 4, v18 // (op 626) 

v_lshrrev_b32 v18, 4, v1 // (op 626) 
v_mul_lo_u32 v19, v18, s20 // (op 626) 

v_add_u32 v18, v16, v19 // (op 626) 


s_mul_i32 s1, s20, s25 // (op 628) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 628) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 628) least significant
s_mul_hi_u32 s15, s20, s24 // (op 628) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 628) most significant: sum
s_add_u32 s1, s1, s15 // (op 628) most significant: sum


s_mov_b32 s48, s22 // (op 628) reg expression
s_mov_b32 s49, s23 // (op 628) reg expression
s_mov_b32 s50, s0 // (op 628) reg expression
s_mov_b32 s51, 131072 // (op 628) call()

v_and_b32 v16, -4, v1 // (op 671) 
v_sub_u32 v19, v1, v16 // (op 671) 

v_lshrrev_b32 v16, 2, v1 // (op 671) 
v_lshl_add_u32 v20, s17, 4, v16 // (op 671) 

v_mul_lo_u32 v16, v20, s26 // (op 671) 

v_add_u32 v20, v19, v16 // (op 671) 


s_mul_i32 s1, s26, s13 // (op 673) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 673) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 673) least significant
s_mul_hi_u32 s15, s26, s12 // (op 673) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 673) most significant: sum
s_add_u32 s1, s1, s15 // (op 673) most significant: sum


s_mov_b32 s52, s28 // (op 673) reg expression
s_mov_b32 s53, s29 // (op 673) reg expression
s_mov_b32 s54, s0 // (op 673) reg expression
s_mov_b32 s55, 131072 // (op 673) call()

v_and_b32 v16, -4, v1 // (op 690) 
v_sub_u32 v19, v1, v16 // (op 690) 

v_lshrrev_b32 v16, 2, v1 // (op 690) 
v_lshl_add_u32 v21, s17, 4, v16 // (op 690) 

v_mul_lo_u32 v16, v21, s26 // (op 690) 

v_add_u32 v21, v19, v16 // (op 690) 


s_mul_i32 s1, s26, s13 // (op 692) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 692) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 692) least significant
s_mul_hi_u32 s15, s26, s12 // (op 692) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 692) most significant: sum
s_add_u32 s1, s1, s15 // (op 692) most significant: sum


s_mov_b32 s52, s28 // (op 692) reg expression
s_mov_b32 s53, s29 // (op 692) reg expression
s_mov_b32 s54, s0 // (op 692) reg expression
s_mov_b32 s55, 131072 // (op 692) call()


v_mov_b64 v[22:23], s[8:9] // (op 370) convert

v_mov_b32 v16, v12 // (op 370) 
buffer_load_dwordx4 v[24:27], v16, s[44:47], 0 offen // (op 370) Load value
v_add_u32 v16, v16, v22 // (op 370) 
buffer_load_dwordx4 v[28:31], v16, s[44:47], 0 offen // (op 370) Load value





buffer_load_ubyte v16, v20, s[52:55], 0 offen offset: 0 // (op 373) Load value

s_lshl_b64 s[0:1], s[2:3], 2 // (op 376) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 376) 


v_mov_b32 v19, v14 // (op 376) reg expression
buffer_load_dwordx2 v[32:33], v19, s[40:43], 0 offen // (op 376) Load value
v_add_u32 v19, v19, v22 // (op 376) 
buffer_load_dwordx2 v[34:35], v19, s[40:43], 0 offen // (op 376) Load value




v_mov_b32 v19, v17 // (op 379) reg expression

buffer_load_ubyte v22, v19, s[48:51], 0 offen offset: 0 // (op 379) Load value


v_mov_b64 v[36:37], s[8:9] // (op 382) convert

v_mov_b32 v19, v12 // (op 382) 
buffer_load_dwordx4 v[38:41], v19, s[44:47], 0 offen offset:128 // (op 382) Load value
v_add_u32 v19, v19, v36 // (op 382) 
buffer_load_dwordx4 v[42:45], v19, s[44:47], 0 offen offset:128 // (op 382) Load value






buffer_load_ubyte v12, v20, s[52:55], 0 offen offset:4 // (op 385) Load value


s_lshl_b64 s[0:1], s[2:3], 2 // (op 388) 
v_lshrrev_b64 v[36:37], 3, s[0:1] // (op 388) 


s_lshl_b64 s[0:1], s[2:3], 7 // (op 388) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 388) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 388) 

v_add_u32 v23, v14, s0 // (op 388) 

buffer_load_dwordx2 v[46:47], v23, s[40:43], 0 offen // (op 388) Load value
v_add_u32 v23, v23, v36 // (op 388) 
buffer_load_dwordx2 v[48:49], v23, s[40:43], 0 offen // (op 388) Load value





s_lshl_b32 s0, s20, 2 // (op 391) 
v_add_u32 v14, v17, s0 // (op 391) 


buffer_load_ubyte v23, v14, s[48:51], 0 offen offset: 0 // (op 391) Load value


v_and_b32 v14, -8, v1 // (op 810) 
v_sub_u32 v17, v1, v14 // (op 810) 

v_lshlrev_b32 v14, 4, v17 // (op 810) 

v_lshrrev_b32 v17, 3, v1 // (op 810) 
v_lshl_add_u32 v19, v17, 8, v14 // (op 810) 





s_waitcnt vmcnt(11)
ds_write_b128 v19, v[24:27] // (op 393) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v19, v[28:31] offset:128 // (op 393) Store local data 

v_and_b32 v14, -4, v1 // (op 916) 
v_sub_u32 v17, v1, v14 // (op 916) 

v_and_b32 v14, 4294967292, v1 // (op 916) 
v_add_u32 v19, v14, v17 // (op 916) 





s_waitcnt vmcnt(9)
ds_write_b8 v19, v16 offset:2048 // (op 394) Store local data 

v_lshlrev_b32 v14, 7, v1 // (op 757) 
v_lshrrev_b32 v17, 3, v14 // (op 757) 




s_waitcnt vmcnt(8)
ds_write_b64 v17, v[32:33] offset:2112 // (op 395) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v17, v[34:35] offset:2120 // (op 395) Store local data 

v_and_b32 v14, -16, v1 // (op 863) 
v_sub_u32 v17, v1, v14 // (op 863) 

v_and_b32 v14, 4294967280, v1 // (op 863) 
v_add_u32 v19, v14, v17 // (op 863) 





s_waitcnt vmcnt(6)
ds_write_b8 v19, v22 offset:3136 // (op 396) Store local data 

v_lshlrev_b32 v14, 7, v1 // (op 728) 
v_lshrrev_b32 v17, 3, v14 // (op 728) 

v_and_b32 v14, -8, v1 // (op 781) 
v_sub_u32 v19, v1, v14 // (op 781) 

v_lshlrev_b32 v14, 4, v19 // (op 781) 

v_lshrrev_b32 v19, 3, v1 // (op 781) 
v_lshl_add_u32 v20, v19, 8, v14 // (op 781) 


v_and_b32 v14, -16, v1 // (op 834) 
v_sub_u32 v19, v1, v14 // (op 834) 

v_and_b32 v14, 4294967280, v1 // (op 834) 
v_add_u32 v36, v14, v19 // (op 834) 


v_and_b32 v14, -4, v1 // (op 887) 
v_sub_u32 v19, v1, v14 // (op 887) 

v_and_b32 v14, 4294967292, v1 // (op 887) 
v_add_u32 v37, v14, v19 // (op 887) 


v_lshlrev_b32 v14, 7, v1 // (op 974) 
v_lshrrev_b32 v19, 3, v14 // (op 974) 

v_and_b32 v14, -8, v1 // (op 998) 
v_sub_u32 v50, v1, v14 // (op 998) 

v_lshlrev_b32 v14, 4, v50 // (op 998) 

v_lshrrev_b32 v50, 3, v1 // (op 998) 
v_lshl_add_u32 v51, v50, 8, v14 // (op 998) 


v_and_b32 v14, -16, v1 // (op 1022) 
v_sub_u32 v50, v1, v14 // (op 1022) 

v_and_b32 v14, 4294967280, v1 // (op 1022) 
v_add_u32 v52, v14, v50 // (op 1022) 


v_and_b32 v14, -4, v1 // (op 1046) 
v_sub_u32 v50, v1, v14 // (op 1046) 

v_and_b32 v14, 4294967292, v1 // (op 1046) 
v_add_u32 v53, v14, v50 // (op 1046) 


s_mov_b32 s0, 0 // (op 61) call()
s_ashr_i64 s[14:15], s[6:7], 7 // (op 60) 
s_mov_b32 s1, s14 // (op 60) convert

s_and_b32 s14, s1, 4294967294 // (op 60) 

s_cmp_lt_u32 s0, s14 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60:

s_add_u32 s0, s0, 2 // (op 62) 
v_mov_b64 v[54:55], s[8:9] // (op 10) convert

v_mov_b32 v14, v13 // (op 10) 
v_mov_b32 v50, v14 // (op 10) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1095) 
buffer_load_dwordx4 v[24:27], v50, s[44:47], 0 offen offset:256 // (op 10) Load value



v_add_u32 v50, v50, v54 // (op 10) 
buffer_load_dwordx4 v[28:31], v50, s[44:47], 0 offen offset:256 // (op 10) Load value


ds_read_b64_tr_b4 v[54:55], v3 offset:2112 // (op 72) Transpose load local data 


ds_read_b64_tr_b4 v[56:57], v3 offset:2240 // (op 72) Transpose load local data 
v_mov_b32 v50, v21 // (op 25) 

s_barrier  // (op 1108) 
buffer_load_ubyte v16, v50, s[52:55], 0 offen offset:8 // (op 25) Load value




s_lshl_b64 s[14:15], s[2:3], 2 // (op 4) 
v_lshrrev_b64 v[58:59], 3, s[14:15] // (op 4) 


v_mov_b32 v60, v15 // (op 4) 
s_lshl_b64 s[14:15], s[2:3], 7 // (op 4) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 4) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 4) 

s_lshl_b32 s1, s14, 1 // (op 4) 

v_add_u32 v61, v60, s1 // (op 4) 

ds_read_b128 v[62:65], v5 // (op 77) Load local data 
ds_read_b128 v[66:69], v5 offset:64 // (op 77) Load local data 
buffer_load_dwordx2 v[32:33], v61, s[40:43], 0 offen // (op 4) Load value
v_add_u32 v61, v61, v58 // (op 4) 
buffer_load_dwordx2 v[34:35], v61, s[40:43], 0 offen // (op 4) Load value


s_waitcnt lgkmcnt(4)
s_barrier  // (op 1122) 




ds_read_u8 v61, v4 offset:3136 // (op 82) Load local data 
v_mov_b32 v58, v18 // (op 17) 
s_barrier  // (op 1136) 
s_lshl_b32 s1, s20, 3 // (op 17) 



ds_read_u8 v59, v6 offset:2048 // (op 87) Load local data 
v_add_u32 v70, v58, s1 // (op 17) 


buffer_load_ubyte v22, v70, s[48:51], 0 offen offset: 0 // (op 17) Load value

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:57], v[62:69], a[0:3], v61, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 90) 




s_barrier  // (op 1159) 



s_waitcnt vmcnt(11)
ds_write_b128 v51, v[38:41] offset:3200 // (op 257) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v51, v[42:45] offset:3328 // (op 257) Store local data 
s_barrier  // (op 1171) 



s_waitcnt vmcnt(9)
ds_write_b8 v53, v12 offset:5248 // (op 259) Store local data 
s_barrier  // (op 1153) 



s_waitcnt vmcnt(8)
ds_write_b64 v19, v[46:47] offset:5312 // (op 247) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v19, v[48:49] offset:5320 // (op 247) Store local data 
s_barrier  // (op 1165) 



s_waitcnt vmcnt(6)
ds_write_b8 v52, v23 offset:6336 // (op 258) Store local data 

v_mov_b64 v[54:55], s[8:9] // (op 260) convert

v_mov_b32 v59, v14 // (op 260) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1150) 
buffer_load_dwordx4 v[38:41], v59, s[44:47], 0 offen offset:384 // (op 260) Load value



v_add_u32 v59, v59, v54 // (op 260) 
buffer_load_dwordx4 v[42:45], v59, s[44:47], 0 offen offset:384 // (op 260) Load value


ds_read_b64_tr_b4 v[54:55], v7 offset:5312 // (op 249) Transpose load local data 

ds_read_b64_tr_b4 v[56:57], v7 offset:5440 // (op 249) Transpose load local data 



v_add_u32 v13, 256, v13 // (op 554) 
s_barrier  // (op 1156) 
s_barrier  // (op 1230) 



ds_read_b128 v[62:65], v9 offset:3200 // (op 251) Load local data 
buffer_load_ubyte v12, v50, s[52:55], 0 offen offset:12 // (op 262) Load value
ds_read_b128 v[66:69], v9 offset:3264 // (op 251) Load local data 


s_barrier  // (op 1236) 
v_add_u32 v21, 8, v21 // (op 682) 
s_lshl_b64 s[14:15], s[2:3], 2 // (op 246) 
v_lshrrev_b64 v[70:71], 3, s[14:15] // (op 246) 


s_lshl_b64 s[14:15], s[2:3], 7 // (op 246) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 246) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 246) 

s_mul_i32 s1, 3, s14 // (op 246) 

v_add_u32 v14, v60, s1 // (op 246) 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1162) 
buffer_load_dwordx2 v[46:47], v14, s[40:43], 0 offen // (op 246) Load value


v_add_u32 v14, v14, v70 // (op 246) 
buffer_load_dwordx2 v[48:49], v14, s[40:43], 0 offen // (op 246) Load value


ds_read_u8 v14, v8 offset:6336 // (op 253) Load local data 



s_barrier  // (op 1168) 
s_barrier  // (op 1242) 



ds_read_u8 v50, v10 offset:5248 // (op 255) Load local data 
s_lshl_b32 s1, s20, 2 // (op 261) 
s_mul_i32 s14, 3, s1 // (op 261) 

s_barrier  // (op 1248) 
v_add_u32 v59, v58, s14 // (op 261) 


buffer_load_ubyte v23, v59, s[48:51], 0 offen offset: 0 // (op 261) Load value

s_lshl_b64 s[14:15], s[2:3], 7 // (op 490) 

s_lshl_b64 s[18:19], s[14:15], 2 // (op 490) 

s_barrier  // (op 1111) 
s_lshl_b32 s1, s20, 3 // (op 618) 



v_add_u32 v18, v18, s1 // (op 618) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 490) 

s_lshl_b32 s1, s14, 1 // (op 490) 

v_add_u32 v15, v15, s1 // (op 490) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:57], v[62:69], a[0:3], v14, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 256) 
s_waitcnt vmcnt(11)
ds_write_b128 v20, v[24:27] // (op 13) Store local data 




s_waitcnt vmcnt(10)
ds_write_b128 v20, v[28:31] offset:128 // (op 13) Store local data 
s_barrier  // (op 1139) 



s_waitcnt vmcnt(9)
ds_write_b8 v37, v16 offset:2048 // (op 30) Store local data 
s_barrier  // (op 1098) 



s_waitcnt vmcnt(8)
ds_write_b64 v17, v[32:33] offset:2112 // (op 6) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v17, v[34:35] offset:2120 // (op 6) Store local data 
s_barrier  // (op 1125) 



s_waitcnt vmcnt(6)
ds_write_b8 v36, v22 offset:3136 // (op 21) Store local data 
s_ashr_i64 s[14:15], s[6:7], 7 // (op 60) 
s_mov_b32 s1, s14 // (op 60) convert

s_and_b32 s14, s1, 4294967294 // (op 60) 

s_cmp_lt_u32 s0, s14 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60:

















v_lshlrev_b32 v13, 1, v1 // (op 453) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 453) 
s_lshr_b32 s14, s0, 1 // (op 453) 

v_lshl_add_u32 v14, s14, 8, v13 // (op 453) 


s_lshl_b32 s0, s16, 4 // (op 453) 
v_mul_lo_u32 v25, v14, s3 // (op 453) most significant: low of low * high
v_mul_hi_u32 v13, v14, s2 // (op 453) most significant: high of low * low
v_mul_lo_u32 v24, v14, s2 // (op 453) least significant: low of low * low
v_add_u32 v25, v25, v13 // (op 453) most significant: sum


s_mov_b32 s15, 0 // (op 453) convert
s_mov_b32 s14, s0 // (op 453) convert

v_mov_b32 v13, s14 // (op 453) 
v_mov_b32 v14, s15 // (op 453) 
v_add_co_u32 v26, s[0:1], v13, v24 // (op 453) least significant half
v_addc_co_u32 v27, s[0:1], v14, v25, s[0:1] // (op 453) most significant half



v_lshlrev_b64 v[26:27], 2, v[26:27] // (op 453) 


v_lshrrev_b64 v[24:25], 3, v[26:27] // (op 453) 

v_mov_b32 v13, v24 // (op 453) convert

s_mul_i32 s1, s2, s7 // (op 455) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 455) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 455) least significant
s_mul_hi_u32 s15, s2, s6 // (op 455) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 455) most significant: sum
s_add_u32 s1, s1, s15 // (op 455) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 455) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 455) 

s_mov_b32 s40, s4 // (op 455) reg expression
s_mov_b32 s41, s5 // (op 455) reg expression
s_mov_b32 s42, s0 // (op 455) reg expression
s_mov_b32 s43, 131072 // (op 455) call()

v_and_b32 v14, -8, v1 // (op 520) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 520) 
v_sub_u32 v15, v1, v14 // (op 520) 

s_lshr_b32 s14, s0, 1 // (op 520) 

v_lshlrev_b32 v14, 4, v15 // (op 520) 

v_lshl_add_u32 v15, s14, 8, v14 // (op 520) 


v_lshrrev_b32 v14, 3, v1 // (op 520) 
v_lshlrev_b32 v16, 1, v14 // (op 520) 

v_lshl_add_u32 v14, s17, 4, v16 // (op 520) 

v_mul_lo_u32 v16, v14, s8 // (op 520) 

v_add_u32 v14, v15, v16 // (op 520) 


s_mul_i32 s1, s8, s13 // (op 522) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 522) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 522) least significant
s_mul_hi_u32 s15, s8, s12 // (op 522) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 522) most significant: sum
s_add_u32 s1, s1, s15 // (op 522) most significant: sum


s_mov_b32 s44, s10 // (op 522) reg expression
s_mov_b32 s45, s11 // (op 522) reg expression
s_mov_b32 s46, s0 // (op 522) reg expression
s_mov_b32 s47, 131072 // (op 522) call()

v_and_b32 v15, -16, v1 // (op 584) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 584) 
v_sub_u32 v16, v1, v15 // (op 584) 

s_lshr_b32 s14, s0, 1 // (op 584) 

v_lshl_add_u32 v15, s16, 4, v16 // (op 584) 

v_lshrrev_b32 v16, 4, v1 // (op 584) 
v_lshl_add_u32 v17, s14, 3, v16 // (op 584) 


v_mul_lo_u32 v16, v17, s20 // (op 584) 

v_add_u32 v17, v15, v16 // (op 584) 


s_mul_i32 s1, s20, s25 // (op 586) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 586) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 586) least significant
s_mul_hi_u32 s15, s20, s24 // (op 586) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 586) most significant: sum
s_add_u32 s1, s1, s15 // (op 586) most significant: sum


s_mov_b32 s48, s22 // (op 586) reg expression
s_mov_b32 s49, s23 // (op 586) reg expression
s_mov_b32 s50, s0 // (op 586) reg expression
s_mov_b32 s51, 131072 // (op 586) call()

v_and_b32 v15, -4, v1 // (op 648) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 648) 
v_sub_u32 v16, v1, v15 // (op 648) 

s_lshr_b32 s14, s0, 1 // (op 648) 

v_lshrrev_b32 v15, 2, v1 // (op 648) 
v_lshl_add_u32 v18, s14, 3, v16 // (op 648) 


v_lshl_add_u32 v16, s17, 4, v15 // (op 648) 

v_mul_lo_u32 v15, v16, s26 // (op 648) 

v_add_u32 v16, v18, v15 // (op 648) 


s_mul_i32 s1, s26, s13 // (op 650) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 650) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 650) least significant
s_mul_hi_u32 s15, s26, s12 // (op 650) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 650) most significant: sum
s_add_u32 s1, s1, s15 // (op 650) most significant: sum


s_mov_b32 s52, s28 // (op 650) reg expression
s_mov_b32 s53, s29 // (op 650) reg expression
s_mov_b32 s54, s0 // (op 650) reg expression
s_mov_b32 s55, 131072 // (op 650) call()

v_lshlrev_b32 v15, 7, v1 // (op 741) 
v_lshrrev_b32 v18, 3, v15 // (op 741) 

v_and_b32 v15, -8, v1 // (op 794) 
v_sub_u32 v19, v1, v15 // (op 794) 

v_lshlrev_b32 v15, 4, v19 // (op 794) 

v_lshrrev_b32 v19, 3, v1 // (op 794) 
v_lshl_add_u32 v20, v19, 8, v15 // (op 794) 


v_and_b32 v15, -16, v1 // (op 847) 
v_sub_u32 v19, v1, v15 // (op 847) 

v_and_b32 v15, 4294967280, v1 // (op 847) 
v_add_u32 v21, v15, v19 // (op 847) 


v_and_b32 v15, -4, v1 // (op 900) 
v_sub_u32 v19, v1, v15 // (op 900) 

v_and_b32 v15, 4294967292, v1 // (op 900) 
v_add_u32 v22, v15, v19 // (op 900) 


s_ashr_i64 s[0:1], s[6:7], 7 // (op 176) 
s_and_b32 s14, s0, 4294967294 // (op 176) 

s_ashr_i64 s[0:1], s[6:7], 7 // (op 175) 
s_mov_b32 s15, s0 // (op 175) convert

s_cmp_lt_u32 s14, s15 // (op 175) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_175 // (op 175) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_175 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_175:

s_add_u32 s14, s14, 1 // (op 177) 
s_lshl_b64 s[0:1], s[2:3], 2 // (op 180) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 180) 


v_mov_b32 v12, v13 // (op 180) 
v_mov_b32 v15, v12 // (op 180) reg expression
buffer_load_dwordx2 v[26:27], v15, s[40:43], 0 offen // (op 180) Load value
v_add_u32 v15, v15, v24 // (op 180) 
buffer_load_dwordx2 v[28:29], v15, s[40:43], 0 offen // (op 180) Load value



s_lshl_b64 s[0:1], s[2:3], 7 // (op 443) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 443) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 443) 

v_add_u32 v13, v13, s0 // (op 443) 


s_barrier  // (op 1101) 
v_mov_b64 v[24:25], s[8:9] // (op 194) convert




v_mov_b32 v12, v14 // (op 194) 
v_mov_b32 v15, v12 // (op 194) 
buffer_load_dwordx4 v[30:33], v15, s[44:47], 0 offen // (op 194) Load value
v_add_u32 v15, v15, v24 // (op 194) 
s_waitcnt vmcnt(2)
ds_write_b64 v18, v[26:27] offset:2112 // (op 181) Store local data 
buffer_load_dwordx4 v[34:37], v15, s[44:47], 0 offen // (op 194) Load value



v_add_u32 v14, 128, v14 // (op 512) 
s_waitcnt vmcnt(2)
ds_write_b64 v18, v[28:29] offset:2120 // (op 181) Store local data 



v_mov_b32 v12, v17 // (op 195) 
s_barrier  // (op 1114) 
v_mov_b32 v15, v12 // (op 195) reg expression




buffer_load_ubyte v19, v15, s[48:51], 0 offen offset: 0 // (op 195) Load value


s_lshl_b32 s15, s20, 2 // (op 576) 
v_add_u32 v17, v17, s15 // (op 576) 

s_waitcnt vmcnt(2)
ds_write_b128 v20, v[30:33] // (op 191) Store local data 


v_mov_b32 v12, v16 // (op 196) 

buffer_load_ubyte v15, v12, s[52:55], 0 offen offset: 0 // (op 196) Load value

v_add_u32 v16, 4, v16 // (op 640) 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[34:37] offset:128 // (op 191) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1104) 



s_barrier  // (op 1128) 
ds_read_b64_tr_b4 v[24:25], v3 offset:2112 // (op 183) Transpose load local data 



ds_read_b64_tr_b4 v[26:27], v3 offset:2240 // (op 183) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v21, v19 offset:3136 // (op 192) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1118) 



s_barrier  // (op 1142) 
ds_read_b128 v[28:31], v5 // (op 185) Load local data 



ds_read_b128 v[32:35], v5 offset:64 // (op 185) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v22, v15 offset:2048 // (op 193) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1132) 


ds_read_u8 v12, v4 offset:3136 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1146) 



ds_read_u8 v15, v6 offset:2048 // (op 189) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[28:35], a[0:3], v12, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 190) 




s_ashr_i64 s[0:1], s[6:7], 7 // (op 175) 
s_mov_b32 s15, s0 // (op 175) convert

s_cmp_lt_u32 s14, s15 // (op 175) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_175 // (op 175) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_175 if true)
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_175:























s_barrier  // (op 1224) 








v_cmp_eq_f32 vcc, 0.00000, v0 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135
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

v_lshrrev_b32 v3, 6, v1 // (op 942) 
v_and_b32 v8, 63, v1 // (op 942) 
v_lshrrev_b32 v9, 4, v8 // (op 942) 
v_lshlrev_b32 v10, 2, v9 // (op 942) 

v_lshl_add_u32 v9, v3, 4, v10 // (op 942) 


v_lshl_add_u32 v3, s16, 4, v9 // (op 942) 

v_lshrrev_b32 v9, 2, v8 // (op 942) 

v_and_b32 v8, 3, v9 // (op 942) 

v_and_b32 v9, 3, v1 // (op 942) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 942) 


v_lshl_add_u32 v8, s17, 4, v10 // (op 942) 

v_mul_lo_u32 v9, v8, s30 // (op 942) 

v_add_lshl_u32 v8, v3, v9, 1 // (op 942) 


s_mul_i32 s15, s30, s13 // (op 944) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 944) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 944) least significant
s_mul_hi_u32 s1, s30, s12 // (op 944) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 944) most significant: sum
s_add_u32 s15, s15, s1 // (op 944) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 944) 

s_mov_b32 s8, s32 // (op 944) reg expression
s_mov_b32 s9, s33 // (op 944) reg expression
s_mov_b32 s10, s0 // (op 944) reg expression
s_mov_b32 s11, 131072 // (op 944) call()


v_pack_b32_f16 v12, v4, v5 // (op 58) 
v_pack_b32_f16 v13, v6, v7 // (op 58) 


buffer_store_dwordx2 v[12:13], v8, s[8:11], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135:
v_lshrrev_b32 v3, 6, v1 // (op 711) 
v_and_b32 v4, 63, v1 // (op 711) 
v_lshrrev_b32 v5, 4, v4 // (op 711) 
v_lshlrev_b32 v6, 2, v5 // (op 711) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 711) 


v_lshl_add_u32 v3, s16, 4, v5 // (op 711) 

v_lshrrev_b32 v5, 2, v4 // (op 711) 

v_and_b32 v4, 3, v5 // (op 711) 

v_and_b32 v5, 3, v1 // (op 711) 
v_lshl_add_u32 v6, v4, 2, v5 // (op 711) 


v_lshl_add_u32 v4, s17, 4, v6 // (op 711) 

v_mul_lo_u32 v5, v4, s34 // (op 711) 

v_add_lshl_u32 v4, v3, v5, 1 // (op 711) 


s_mul_i32 s15, s34, s13 // (op 713) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 713) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 713) least significant
s_mul_hi_u32 s1, s34, s12 // (op 713) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 713) most significant: sum
s_add_u32 s15, s15, s1 // (op 713) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 713) 

s_mov_b32 s4, s36 // (op 713) reg expression
s_mov_b32 s5, s37 // (op 713) reg expression
s_mov_b32 s6, s0 // (op 713) reg expression
s_mov_b32 s7, 131072 // (op 713) call()





buffer_load_dwordx2 v[6:7], v4, s[4:7], 0 offen // (op 148) Load value



v_accvgpr_read v12, a0 // (op 144) 
v_accvgpr_read v13, a1 // (op 144) 
v_accvgpr_read v14, a2 // (op 144) 
v_accvgpr_read v15, a3 // (op 144) 
v_mul_f32 v16, v11, v12 // (op 144) 
v_mul_f32 v17, v11, v13 // (op 144) 
v_mul_f32 v18, v11, v14 // (op 144) 
v_mul_f32 v19, v11, v15 // (op 144) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v4, v6 // (op 149) 
v_lshrrev_b32 v5, 16, v6 // (op 149) 
v_cvt_f32_f16 v5, v5 // (op 149) 
v_mul_f32 v12, v0, v4 // (op 149) 
v_mul_f32 v13, v0, v5 // (op 149) 
v_cvt_f32_f16 v4, v7 // (op 149) 
v_lshrrev_b32 v5, 16, v7 // (op 149) 
v_cvt_f32_f16 v5, v5 // (op 149) 
v_mul_f32 v14, v0, v4 // (op 149) 
v_mul_f32 v15, v0, v5 // (op 149) 


v_add_f32 v4, v12, v16 // (op 145) 
v_add_f32 v5, v13, v17 // (op 145) 
v_add_f32 v6, v14, v18 // (op 145) 
v_add_f32 v7, v15, v19 // (op 145) 


v_cvt_f16_f32 v12, v4 // (op 146) 
v_cvt_f16_f32 v13, v5 // (op 146) 
v_cvt_f16_f32 v14, v6 // (op 146) 
v_cvt_f16_f32 v15, v7 // (op 146) 

v_lshrrev_b32 v3, 6, v1 // (op 959) 
v_and_b32 v4, 63, v1 // (op 959) 
v_lshrrev_b32 v5, 4, v4 // (op 959) 
v_lshlrev_b32 v6, 2, v5 // (op 959) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 959) 


v_lshl_add_u32 v3, s16, 4, v5 // (op 959) 

v_lshrrev_b32 v5, 2, v4 // (op 959) 

v_and_b32 v4, 3, v5 // (op 959) 

v_and_b32 v5, 3, v1 // (op 959) 
v_lshl_add_u32 v6, v4, 2, v5 // (op 959) 


v_lshl_add_u32 v4, s17, 4, v6 // (op 959) 

v_mul_lo_u32 v5, v4, s30 // (op 959) 

v_add_lshl_u32 v4, v3, v5, 1 // (op 959) 


s_mul_i32 s15, s30, s13 // (op 961) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 961) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 961) least significant
s_mul_hi_u32 s1, s30, s12 // (op 961) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 961) most significant: sum
s_add_u32 s15, s15, s1 // (op 961) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 961) 

s_mov_b32 s8, s32 // (op 961) reg expression
s_mov_b32 s9, s33 // (op 961) reg expression
s_mov_b32 s10, s0 // (op 961) reg expression
s_mov_b32 s11, 131072 // (op 961) call()


v_pack_b32_f16 v6, v12, v13 // (op 147) 
v_pack_b32_f16 v7, v14, v15 // (op 147) 


buffer_store_dwordx2 v[6:7], v4, s[8:11], 0 offen // (op 147) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135:




s_endpgm  // End of RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.LRR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_end:
.size RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2, .LRR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_end-RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
  .amdhsa_next_free_vgpr 76
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 6400 // lds bytes
  .amdhsa_accum_offset 72
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
    DataType: F4
    DataTypeA: F4
    DataTypeB: F8
    DestDataType: h
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
    macrotile: [16, 16, 128]
    threads: [64, 1, 1]
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
  - .name: RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
    .symbol: RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 6400
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 72
    .agpr_count: 4
    .max_flat_workgroup_size: 64
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

