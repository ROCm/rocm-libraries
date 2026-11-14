// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.p2align 8
.type RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2,@function
RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_exec_begin:


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


v_mul_lo_u32 v0, v3, 6 // (op 769) 
v_lshrrev_b32 v4, 3, v0 // (op 769) 

v_lshrrev_b32 v0, 4, v3 // (op 769) 

v_lshlrev_b32 v3, 2, v0 // (op 769) 

v_add_u32 v0, v4, v3 // (op 769) 


v_and_b32 v3, 15, v1 // (op 822) 
v_and_b32 v4, 63, v1 // (op 822) 
v_lshrrev_b32 v5, 4, v4 // (op 822) 

v_lshlrev_b32 v4, 5, v5 // (op 822) 

v_lshl_add_u32 v5, v3, 7, v4 // (op 822) 


v_mul_lo_u32 v3, v5, 6 // (op 822) 

v_lshrrev_b32 v4, 3, v3 // (op 822) 

v_lshrrev_b32 v3, 6, v1 // (op 875) 
v_and_b32 v5, 15, v1 // (op 875) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 875) 


v_and_b32 v3, 48, v1 // (op 875) 
v_add_u32 v5, v3, v6 // (op 875) 


v_and_b32 v3, 15, v1 // (op 928) 
v_and_b32 v6, 63, v1 // (op 928) 
v_lshrrev_b32 v7, 4, v6 // (op 928) 

v_lshl_add_u32 v6, v3, 2, v7 // (op 928) 


v_and_b32 v3, 63, v1 // (op 986) 
v_and_b32 v7, 48, v1 // (op 986) 
v_sub_u32 v8, v3, v7 // (op 986) 

v_lshrrev_b32 v7, 4, v3 // (op 986) 

v_lshl_add_u32 v3, v7, 5, v8 // (op 986) 


v_lshrrev_b32 v7, 6, v1 // (op 986) 
v_lshlrev_b32 v8, 4, v7 // (op 986) 

v_lshl_add_u32 v7, v3, 4, v8 // (op 986) 


v_mul_lo_u32 v3, v7, 6 // (op 986) 
v_lshrrev_b32 v8, 3, v3 // (op 986) 

v_lshrrev_b32 v3, 4, v7 // (op 986) 

v_lshlrev_b32 v7, 2, v3 // (op 986) 

v_add_u32 v3, v8, v7 // (op 986) 


v_and_b32 v7, 15, v1 // (op 1010) 
v_and_b32 v8, 63, v1 // (op 1010) 
v_lshrrev_b32 v9, 4, v8 // (op 1010) 

v_lshlrev_b32 v8, 5, v9 // (op 1010) 

v_lshl_add_u32 v9, v7, 7, v8 // (op 1010) 


v_mul_lo_u32 v7, v9, 6 // (op 1010) 

v_lshrrev_b32 v8, 3, v7 // (op 1010) 

v_lshrrev_b32 v7, 6, v1 // (op 1034) 
v_and_b32 v9, 15, v1 // (op 1034) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 1034) 


v_and_b32 v7, 48, v1 // (op 1034) 
v_add_u32 v9, v7, v10 // (op 1034) 


v_and_b32 v7, 15, v1 // (op 1058) 
v_and_b32 v10, 63, v1 // (op 1058) 
v_lshrrev_b32 v11, 4, v10 // (op 1058) 

v_lshl_add_u32 v10, v7, 2, v11 // (op 1058) 


v_mov_b32 v7, s14 // (op 2) Move value
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





v_mul_lo_u32 v13, v17, 6 // (op 479) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 479) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 479) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 479) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 479) 

v_mov_b32 v12, v14 // (op 479) convert

s_mul_i32 s1, s7, 6 // (op 481) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 481) least significant
s_mul_hi_u32 s14, s6, 6 // (op 481) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 481) most significant: sum

s_mul_i32 s15, s2, s1 // (op 481) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 481) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 481) least significant
s_mul_hi_u32 s19, s2, s0 // (op 481) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 481) most significant: sum
s_add_u32 s15, s15, s19 // (op 481) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 481) 

s_mov_b32 s40, s4 // (op 481) reg expression
s_mov_b32 s41, s5 // (op 481) reg expression
s_mov_b32 s42, s0 // (op 481) reg expression
s_mov_b32 s43, 131072 // (op 481) call()

s_lshl_b32 s0, s16, 4 // (op 498) 
v_lshlrev_b32 v13, 1, v1 // (op 498) 
s_mov_b32 s15, 0 // (op 498) convert
s_mov_b32 s14, s0 // (op 498) convert

v_mul_lo_u32 v15, v13, s3 // (op 498) most significant: low of low * high
v_mul_hi_u32 v16, v13, s2 // (op 498) most significant: high of low * low
v_mul_lo_u32 v14, v13, s2 // (op 498) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 498) most significant: sum


v_mov_b32 v13, s14 // (op 498) 
v_mov_b32 v16, s15 // (op 498) 
v_add_co_u32 v18, s[0:1], v13, v14 // (op 498) least significant half
v_addc_co_u32 v19, s[0:1], v16, v15, s[0:1] // (op 498) most significant half





v_mul_lo_u32 v17, v19, 6 // (op 498) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 498) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 498) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 498) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 498) 

v_mov_b32 v13, v14 // (op 498) convert

s_mul_i32 s1, s7, 6 // (op 500) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 500) least significant
s_mul_hi_u32 s14, s6, 6 // (op 500) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 500) most significant: sum

s_mul_i32 s15, s2, s1 // (op 500) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 500) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 500) least significant
s_mul_hi_u32 s19, s2, s0 // (op 500) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 500) most significant: sum
s_add_u32 s15, s15, s19 // (op 500) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 500) 

s_mov_b32 s40, s4 // (op 500) reg expression
s_mov_b32 s41, s5 // (op 500) reg expression
s_mov_b32 s42, s0 // (op 500) reg expression
s_mov_b32 s43, 131072 // (op 500) call()

v_and_b32 v14, -4, v1 // (op 543) 
v_sub_u32 v15, v1, v14 // (op 543) 

v_lshlrev_b32 v14, 5, v15 // (op 543) 

v_mov_b32 v17, 0 // (op 543) convert
v_mov_b32 v16, v14 // (op 543) convert

v_lshrrev_b32 v14, 2, v1 // (op 543) 
v_lshl_add_u32 v15, s17, 4, v14 // (op 543) 

v_mul_lo_u32 v19, v15, s9 // (op 543) most significant: low of low * high
v_mul_hi_u32 v14, v15, s8 // (op 543) most significant: high of low * low
v_mul_lo_u32 v18, v15, s8 // (op 543) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 543) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 543) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 543) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 543) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 543) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 543) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 543) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 543) 

v_mov_b32 v16, v14 // (op 543) convert

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

v_and_b32 v14, -4, v1 // (op 562) 
v_sub_u32 v15, v1, v14 // (op 562) 

v_lshlrev_b32 v14, 5, v15 // (op 562) 

v_mov_b32 v19, 0 // (op 562) convert
v_mov_b32 v18, v14 // (op 562) convert

v_lshrrev_b32 v17, 2, v1 // (op 562) 
v_lshl_add_u32 v14, s17, 4, v17 // (op 562) 

v_mul_lo_u32 v21, v14, s9 // (op 562) most significant: low of low * high
v_mul_hi_u32 v15, v14, s8 // (op 562) most significant: high of low * low
v_mul_lo_u32 v20, v14, s8 // (op 562) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 562) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 562) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 562) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 562) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 562) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 562) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 562) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 562) 

v_mov_b32 v17, v14 // (op 562) convert

s_mul_i32 s1, s13, 6 // (op 564) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 564) least significant
s_mul_hi_u32 s14, s12, 6 // (op 564) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 564) most significant: sum

s_mul_i32 s15, s8, s1 // (op 564) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 564) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 564) least significant
s_mul_hi_u32 s19, s8, s0 // (op 564) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 564) most significant: sum
s_add_u32 s15, s15, s19 // (op 564) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 564) 

s_mov_b32 s44, s10 // (op 564) reg expression
s_mov_b32 s45, s11 // (op 564) reg expression
s_mov_b32 s46, s0 // (op 564) reg expression
s_mov_b32 s47, 131072 // (op 564) call()

v_and_b32 v14, -16, v1 // (op 607) 
v_sub_u32 v15, v1, v14 // (op 607) 

v_lshl_add_u32 v14, s16, 4, v15 // (op 607) 

v_lshrrev_b32 v15, 4, v1 // (op 607) 

s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v18, v15, s20 // (op 607) 

v_add_u32 v15, v14, v18 // (op 607) 


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

v_and_b32 v14, -16, v1 // (op 626) 
v_sub_u32 v18, v1, v14 // (op 626) 

v_lshl_add_u32 v14, s16, 4, v18 // (op 626) 

v_lshrrev_b32 v18, 4, v1 // (op 626) 
v_mul_lo_u32 v19, v18, s20 // (op 626) 

v_add_u32 v18, v14, v19 // (op 626) 


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

v_and_b32 v14, -4, v1 // (op 671) 
v_sub_u32 v19, v1, v14 // (op 671) 

v_lshrrev_b32 v14, 2, v1 // (op 671) 
v_lshl_add_u32 v20, s17, 4, v14 // (op 671) 

v_mul_lo_u32 v14, v20, s26 // (op 671) 

v_add_u32 v20, v19, v14 // (op 671) 


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

v_and_b32 v14, -4, v1 // (op 690) 
v_sub_u32 v19, v1, v14 // (op 690) 

v_lshrrev_b32 v14, 2, v1 // (op 690) 
v_lshl_add_u32 v21, s17, 4, v14 // (op 690) 

v_mul_lo_u32 v14, v21, s26 // (op 690) 

v_add_u32 v21, v19, v14 // (op 690) 


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




buffer_load_dwordx4 v[22:25], v16, s[44:47], 0 offen // (op 370) Load value
buffer_load_dwordx2 v[26:27], v16, s[44:47], 0 offen offset:16 // (op 370) Load value



buffer_load_ubyte v14, v20, s[52:55], 0 offen offset: 0 // (op 373) Load value

s_mul_i32 s1, s3, 6 // (op 376) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 376) least significant
s_mul_hi_u32 s14, s2, 6 // (op 376) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 376) most significant: sum

v_lshrrev_b64 v[28:29], 3, s[0:1] // (op 376) 


v_mov_b32 v19, v12 // (op 376) reg expression
buffer_load_dwordx3 v[30:32], v19, s[40:43], 0 offen // (op 376) Load value
v_add_u32 v19, v19, v28 // (op 376) 
buffer_load_dwordx3 v[34:36], v19, s[40:43], 0 offen // (op 376) Load value




v_mov_b32 v19, v15 // (op 379) reg expression

buffer_load_ubyte v33, v19, s[48:51], 0 offen offset: 0 // (op 379) Load value




buffer_load_dwordx4 v[38:41], v16, s[44:47], 0 offen offset:96 // (op 382) Load value
buffer_load_dwordx2 v[42:43], v16, s[44:47], 0 offen offset:112 // (op 382) Load value




buffer_load_ubyte v16, v20, s[52:55], 0 offen offset:4 // (op 385) Load value


s_mul_i32 s1, s3, 6 // (op 388) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 388) least significant
s_mul_hi_u32 s14, s2, 6 // (op 388) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 388) most significant: sum

v_lshrrev_b64 v[28:29], 3, s[0:1] // (op 388) 


s_lshl_b64 s[0:1], s[2:3], 7 // (op 388) 
s_mul_i32 s15, s1, 6 // (op 388) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 388) least significant
s_mul_hi_u32 s18, s0, 6 // (op 388) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 388) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 388) 

v_add_u32 v37, v12, s0 // (op 388) 

buffer_load_dwordx3 v[44:46], v37, s[40:43], 0 offen // (op 388) Load value
v_add_u32 v37, v37, v28 // (op 388) 
buffer_load_dwordx3 v[48:50], v37, s[40:43], 0 offen // (op 388) Load value





s_lshl_b32 s0, s20, 2 // (op 391) 
v_add_u32 v12, v15, s0 // (op 391) 


buffer_load_ubyte v37, v12, s[48:51], 0 offen offset: 0 // (op 391) Load value


v_and_b32 v12, -4, v1 // (op 810) 
v_sub_u32 v15, v1, v12 // (op 810) 

v_lshlrev_b32 v12, 5, v15 // (op 810) 

v_lshrrev_b32 v15, 2, v1 // (op 810) 
v_lshl_add_u32 v47, v15, 7, v12 // (op 810) 


v_mul_lo_u32 v12, v47, 6 // (op 810) 

v_lshrrev_b32 v15, 3, v12 // (op 810) 




s_waitcnt vmcnt(11)
ds_write_b128 v15, v[22:25] // (op 393) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v15, v[26:27] offset:16 // (op 393) Store local data 

v_and_b32 v12, -4, v1 // (op 916) 
v_sub_u32 v15, v1, v12 // (op 916) 

v_and_b32 v12, 4294967292, v1 // (op 916) 
v_add_u32 v47, v12, v15 // (op 916) 





s_waitcnt vmcnt(9)
ds_write_b8 v47, v14 offset:1536 // (op 394) Store local data 

v_lshlrev_b32 v12, 5, v1 // (op 757) 
v_mul_lo_u32 v15, v12, 6 // (op 757) 
v_lshrrev_b32 v47, 3, v15 // (op 757) 

v_lshrrev_b32 v15, 4, v12 // (op 757) 

v_lshlrev_b32 v12, 2, v15 // (op 757) 

v_add_u32 v15, v47, v12 // (op 757) 





s_waitcnt vmcnt(8)
ds_write_b96 v15, v[30:32] offset:1600 // (op 395) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v15, v[34:36] offset:1616 // (op 395) Store local data 

v_and_b32 v12, -16, v1 // (op 863) 
v_sub_u32 v15, v1, v12 // (op 863) 

v_and_b32 v12, 4294967280, v1 // (op 863) 
v_add_u32 v47, v12, v15 // (op 863) 





s_waitcnt vmcnt(6)
ds_write_b8 v47, v33 offset:3648 // (op 396) Store local data 

v_lshlrev_b32 v12, 5, v1 // (op 728) 
v_mul_lo_u32 v15, v12, 6 // (op 728) 
v_lshrrev_b32 v47, 3, v15 // (op 728) 

v_lshrrev_b32 v15, 4, v12 // (op 728) 

v_lshlrev_b32 v12, 2, v15 // (op 728) 

v_add_u32 v15, v47, v12 // (op 728) 


v_and_b32 v12, -4, v1 // (op 781) 
v_sub_u32 v47, v1, v12 // (op 781) 

v_lshlrev_b32 v12, 5, v47 // (op 781) 

v_lshrrev_b32 v47, 2, v1 // (op 781) 
v_lshl_add_u32 v19, v47, 7, v12 // (op 781) 


v_mul_lo_u32 v12, v19, 6 // (op 781) 

v_lshrrev_b32 v47, 3, v12 // (op 781) 

v_and_b32 v12, -16, v1 // (op 834) 
v_sub_u32 v19, v1, v12 // (op 834) 

v_and_b32 v12, 4294967280, v1 // (op 834) 
v_add_u32 v20, v12, v19 // (op 834) 


v_and_b32 v12, -4, v1 // (op 887) 
v_sub_u32 v19, v1, v12 // (op 887) 

v_and_b32 v12, 4294967292, v1 // (op 887) 
v_add_u32 v28, v12, v19 // (op 887) 


v_lshlrev_b32 v12, 5, v1 // (op 974) 
v_mul_lo_u32 v19, v12, 6 // (op 974) 
v_lshrrev_b32 v29, 3, v19 // (op 974) 

v_lshrrev_b32 v19, 4, v12 // (op 974) 

v_lshlrev_b32 v12, 2, v19 // (op 974) 

v_add_u32 v19, v29, v12 // (op 974) 


v_and_b32 v12, -4, v1 // (op 998) 
v_sub_u32 v29, v1, v12 // (op 998) 

v_lshlrev_b32 v12, 5, v29 // (op 998) 

v_lshrrev_b32 v29, 2, v1 // (op 998) 
v_lshl_add_u32 v51, v29, 7, v12 // (op 998) 


v_mul_lo_u32 v12, v51, 6 // (op 998) 

v_lshrrev_b32 v29, 3, v12 // (op 998) 

v_and_b32 v12, -16, v1 // (op 1022) 
v_sub_u32 v51, v1, v12 // (op 1022) 

v_and_b32 v12, 4294967280, v1 // (op 1022) 
v_add_u32 v52, v12, v51 // (op 1022) 


v_and_b32 v12, -4, v1 // (op 1046) 
v_sub_u32 v51, v1, v12 // (op 1046) 

v_and_b32 v12, 4294967292, v1 // (op 1046) 
v_add_u32 v53, v12, v51 // (op 1046) 


s_mov_b32 s0, 0 // (op 61) call()
s_ashr_i64 s[14:15], s[6:7], 7 // (op 60) 
s_mov_b32 s1, s14 // (op 60) convert

s_and_b32 s14, s1, 4294967294 // (op 60) 

s_cmp_lt_u32 s0, s14 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60:


s_add_u32 s0, s0, 2 // (op 62) 
v_mov_b32 v12, v17 // (op 10) 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 1095) 
buffer_load_dwordx4 v[22:25], v12, s[44:47], 0 offen offset:192 // (op 10) Load value



ds_read_b96_tr_b6 v[54:56], v0 offset:1600 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[57:59], v0 offset:1856 // (op 72) Transpose load local data 
buffer_load_dwordx2 v[26:27], v12, s[44:47], 0 offen offset:208 // (op 10) Load value


s_barrier  // (op 1108) 
v_mov_b32 v51, v21 // (op 25) 




buffer_load_ubyte v14, v51, s[52:55], 0 offen offset:8 // (op 25) Load value
ds_read_b128 v[60:63], v4 // (op 77) Load local data 

s_mul_i32 s15, s3, 6 // (op 4) most significant: low of high * low
s_mul_i32 s14, s2, 6 // (op 4) least significant
s_mul_hi_u32 s1, s2, 6 // (op 4) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 4) most significant: sum

v_lshrrev_b64 v[66:67], 3, s[14:15] // (op 4) 


v_mov_b32 v68, v13 // (op 4) 
s_lshl_b64 s[14:15], s[2:3], 7 // (op 4) 
s_mul_i32 s19, s15, 6 // (op 4) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 4) least significant
s_mul_hi_u32 s1, s14, 6 // (op 4) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 4) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 4) 

s_lshl_b32 s1, s14, 1 // (op 4) 

v_add_u32 v69, v68, s1 // (op 4) 

ds_read_b64 v[64:65], v4 offset:16 // (op 77) Load local data 
buffer_load_dwordx3 v[30:32], v69, s[40:43], 0 offen // (op 4) Load value
v_add_u32 v69, v69, v66 // (op 4) 
buffer_load_dwordx3 v[34:36], v69, s[40:43], 0 offen // (op 4) Load value


s_waitcnt lgkmcnt(4)
s_barrier  // (op 1122) 




ds_read_u8 v66, v5 offset:3648 // (op 82) Load local data 
v_mov_b32 v67, v18 // (op 17) 
s_barrier  // (op 1136) 
s_lshl_b32 s1, s20, 3 // (op 17) 



ds_read_u8 v69, v6 offset:1536 // (op 87) Load local data 
v_add_u32 v70, v67, s1 // (op 17) 


buffer_load_ubyte v33, v70, s[48:51], 0 offen offset: 0 // (op 17) Load value

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:59], v[60:65], a[0:3], v66, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 90) 




s_barrier  // (op 1159) 



s_waitcnt vmcnt(11)
ds_write_b128 v29, v[38:41] offset:3712 // (op 257) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v29, v[42:43] offset:3728 // (op 257) Store local data 
s_barrier  // (op 1171) 



s_waitcnt vmcnt(9)
ds_write_b8 v53, v16 offset:5248 // (op 259) Store local data 
s_barrier  // (op 1153) 



s_waitcnt vmcnt(8)
ds_write_b96 v19, v[44:46] offset:5312 // (op 247) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v19, v[48:50] offset:5328 // (op 247) Store local data 
s_barrier  // (op 1165) 



s_waitcnt vmcnt(6)
ds_write_b8 v52, v37 offset:7360 // (op 258) Store local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 1150) 
buffer_load_dwordx4 v[38:41], v12, s[44:47], 0 offen offset:288 // (op 260) Load value



ds_read_b96_tr_b6 v[54:56], v3 offset:5312 // (op 249) Transpose load local data 
ds_read_b96_tr_b6 v[57:59], v3 offset:5568 // (op 249) Transpose load local data 
buffer_load_dwordx2 v[42:43], v12, s[44:47], 0 offen offset:304 // (op 260) Load value




s_barrier  // (op 1156) 
s_barrier  // (op 1230) 



v_add_u32 v17, 192, v17 // (op 554) 
ds_read_b128 v[60:63], v8 offset:3712 // (op 251) Load local data 
buffer_load_ubyte v16, v51, s[52:55], 0 offen offset:12 // (op 262) Load value
ds_read_b64 v[64:65], v8 offset:3728 // (op 251) Load local data 


s_barrier  // (op 1236) 
v_add_u32 v21, 8, v21 // (op 682) 
s_mul_i32 s15, s3, 6 // (op 246) most significant: low of high * low
s_mul_i32 s14, s2, 6 // (op 246) least significant
s_mul_hi_u32 s1, s2, 6 // (op 246) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 246) most significant: sum

v_lshrrev_b64 v[70:71], 3, s[14:15] // (op 246) 


s_lshl_b64 s[14:15], s[2:3], 7 // (op 246) 
s_mul_i32 s19, s15, 6 // (op 246) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 246) least significant
s_mul_hi_u32 s1, s14, 6 // (op 246) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 246) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 246) 

s_mul_i32 s1, 3, s14 // (op 246) 

v_add_u32 v12, v68, s1 // (op 246) 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1162) 
buffer_load_dwordx3 v[44:46], v12, s[40:43], 0 offen // (op 246) Load value


v_add_u32 v12, v12, v70 // (op 246) 
ds_read_u8 v51, v9 offset:7360 // (op 253) Load local data 
buffer_load_dwordx3 v[48:50], v12, s[40:43], 0 offen // (op 246) Load value





s_barrier  // (op 1168) 
s_barrier  // (op 1242) 



ds_read_u8 v12, v10 offset:5248 // (op 255) Load local data 
s_lshl_b32 s1, s20, 2 // (op 261) 
s_mul_i32 s14, 3, s1 // (op 261) 

s_barrier  // (op 1248) 
v_add_u32 v66, v67, s14 // (op 261) 


s_lshl_b64 s[14:15], s[2:3], 7 // (op 490) 
buffer_load_ubyte v37, v66, s[48:51], 0 offen offset: 0 // (op 261) Load value

s_mul_i32 s19, s15, 6 // (op 490) most significant: low of high * low

s_mul_i32 s18, s14, 6 // (op 490) least significant
s_barrier  // (op 1111) 
s_mul_hi_u32 s1, s14, 6 // (op 490) most significant: high of low * low



s_add_u32 s19, s19, s1 // (op 490) most significant: sum


s_lshl_b32 s1, s20, 3 // (op 618) 
v_add_u32 v18, v18, s1 // (op 618) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 490) 

s_lshl_b32 s1, s14, 1 // (op 490) 

v_add_u32 v13, v13, s1 // (op 490) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:59], v[60:65], a[0:3], v51, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 256) 
s_waitcnt vmcnt(11)
ds_write_b128 v47, v[22:25] // (op 13) Store local data 




s_waitcnt vmcnt(10)
ds_write_b64 v47, v[26:27] offset:16 // (op 13) Store local data 
s_barrier  // (op 1139) 



s_waitcnt vmcnt(9)
ds_write_b8 v28, v14 offset:1536 // (op 30) Store local data 
s_barrier  // (op 1098) 



s_waitcnt vmcnt(8)
ds_write_b96 v15, v[30:32] offset:1600 // (op 6) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v15, v[34:36] offset:1616 // (op 6) Store local data 
s_barrier  // (op 1125) 



s_waitcnt vmcnt(6)
ds_write_b8 v20, v33 offset:3648 // (op 21) Store local data 
s_ashr_i64 s[14:15], s[6:7], 7 // (op 60) 
s_mov_b32 s1, s14 // (op 60) convert

s_and_b32 s14, s1, 4294967294 // (op 60) 

s_cmp_lt_u32 s0, s14 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60:

















v_lshlrev_b32 v47, 1, v1 // (op 453) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 453) 
s_lshr_b32 s14, s0, 1 // (op 453) 

v_lshl_add_u32 v12, s14, 8, v47 // (op 453) 


s_lshl_b32 s0, s16, 4 // (op 453) 
v_mul_lo_u32 v15, v12, s3 // (op 453) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 453) most significant: high of low * low
v_mul_lo_u32 v14, v12, s2 // (op 453) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 453) most significant: sum


s_mov_b32 s15, 0 // (op 453) convert
s_mov_b32 s14, s0 // (op 453) convert

v_mov_b32 v47, s14 // (op 453) 
v_mov_b32 v12, s15 // (op 453) 
v_add_co_u32 v18, s[0:1], v47, v14 // (op 453) least significant half
v_addc_co_u32 v19, s[0:1], v12, v15, s[0:1] // (op 453) most significant half





v_mul_lo_u32 v13, v19, 6 // (op 453) most significant: low of high * low
v_mul_hi_u32 v17, v18, 6 // (op 453) most significant: high of low * low
v_mul_lo_u32 v12, v18, 6 // (op 453) least significant: low of low * low
v_add_u32 v13, v13, v17 // (op 453) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 453) 

v_mov_b32 v47, v14 // (op 453) convert

s_mul_i32 s1, s7, 6 // (op 455) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 455) least significant
s_mul_hi_u32 s14, s6, 6 // (op 455) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 455) most significant: sum

s_mul_i32 s15, s2, s1 // (op 455) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 455) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 455) least significant
s_mul_hi_u32 s19, s2, s0 // (op 455) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 455) most significant: sum
s_add_u32 s15, s15, s19 // (op 455) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 455) 

s_mov_b32 s40, s4 // (op 455) reg expression
s_mov_b32 s41, s5 // (op 455) reg expression
s_mov_b32 s42, s0 // (op 455) reg expression
s_mov_b32 s43, 131072 // (op 455) call()

v_and_b32 v12, -4, v1 // (op 520) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 520) 
v_sub_u32 v13, v1, v12 // (op 520) 

s_lshr_b32 s14, s0, 1 // (op 520) 

v_lshlrev_b32 v12, 5, v13 // (op 520) 

v_lshl_add_u32 v13, s14, 8, v12 // (op 520) 


v_mov_b32 v15, 0 // (op 520) convert
v_mov_b32 v14, v13 // (op 520) convert

v_lshrrev_b32 v12, 2, v1 // (op 520) 
v_lshl_add_u32 v13, s17, 4, v12 // (op 520) 

v_mul_lo_u32 v19, v13, s9 // (op 520) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 520) most significant: high of low * low
v_mul_lo_u32 v18, v13, s8 // (op 520) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 520) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v18 // (op 520) least significant half
v_addc_co_u32 v13, s[0:1], v15, v19, s[0:1] // (op 520) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 520) most significant: low of high * low
v_mul_hi_u32 v17, v12, 6 // (op 520) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 520) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 520) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 520) 

v_mov_b32 v14, v12 // (op 520) convert

s_mul_i32 s1, s13, 6 // (op 522) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 522) least significant
s_mul_hi_u32 s14, s12, 6 // (op 522) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 522) most significant: sum

s_mul_i32 s15, s8, s1 // (op 522) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 522) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 522) least significant
s_mul_hi_u32 s19, s8, s0 // (op 522) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 522) most significant: sum
s_add_u32 s15, s15, s19 // (op 522) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 522) 

s_mov_b32 s44, s10 // (op 522) reg expression
s_mov_b32 s45, s11 // (op 522) reg expression
s_mov_b32 s46, s0 // (op 522) reg expression
s_mov_b32 s47, 131072 // (op 522) call()

v_and_b32 v15, -16, v1 // (op 584) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 584) 
v_sub_u32 v12, v1, v15 // (op 584) 

s_lshr_b32 s14, s0, 1 // (op 584) 

v_lshl_add_u32 v13, s16, 4, v12 // (op 584) 

v_lshrrev_b32 v12, 4, v1 // (op 584) 
v_lshl_add_u32 v15, s14, 3, v12 // (op 584) 


v_mul_lo_u32 v12, v15, s20 // (op 584) 

v_add_u32 v15, v13, v12 // (op 584) 


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

v_and_b32 v12, -4, v1 // (op 648) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 648) 
v_sub_u32 v13, v1, v12 // (op 648) 

s_lshr_b32 s14, s0, 1 // (op 648) 

v_lshrrev_b32 v12, 2, v1 // (op 648) 
v_lshl_add_u32 v17, s14, 3, v13 // (op 648) 


v_lshl_add_u32 v13, s17, 4, v12 // (op 648) 

v_mul_lo_u32 v12, v13, s26 // (op 648) 

v_add_u32 v13, v17, v12 // (op 648) 


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

v_lshlrev_b32 v12, 5, v1 // (op 741) 
v_mul_lo_u32 v17, v12, 6 // (op 741) 
v_lshrrev_b32 v18, 3, v17 // (op 741) 

v_lshrrev_b32 v17, 4, v12 // (op 741) 

v_lshlrev_b32 v12, 2, v17 // (op 741) 

v_add_u32 v17, v18, v12 // (op 741) 


v_and_b32 v12, -4, v1 // (op 794) 
v_sub_u32 v18, v1, v12 // (op 794) 

v_lshlrev_b32 v12, 5, v18 // (op 794) 

v_lshrrev_b32 v18, 2, v1 // (op 794) 
v_lshl_add_u32 v19, v18, 7, v12 // (op 794) 


v_mul_lo_u32 v12, v19, 6 // (op 794) 

v_lshrrev_b32 v18, 3, v12 // (op 794) 

v_and_b32 v12, -16, v1 // (op 847) 
v_sub_u32 v19, v1, v12 // (op 847) 

v_and_b32 v12, 4294967280, v1 // (op 847) 
v_add_u32 v20, v12, v19 // (op 847) 


v_and_b32 v12, -4, v1 // (op 900) 
v_sub_u32 v19, v1, v12 // (op 900) 

v_and_b32 v12, 4294967292, v1 // (op 900) 
v_add_u32 v21, v12, v19 // (op 900) 


s_ashr_i64 s[0:1], s[6:7], 7 // (op 176) 
s_and_b32 s14, s0, 4294967294 // (op 176) 

s_ashr_i64 s[0:1], s[6:7], 7 // (op 175) 
s_mov_b32 s15, s0 // (op 175) convert

s_cmp_lt_u32 s14, s15 // (op 175) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_175 // (op 175) Condition: Top (jump to RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_175 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_175:

s_add_u32 s14, s14, 1 // (op 177) 
s_mul_i32 s1, s3, 6 // (op 180) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 180) least significant
s_mul_hi_u32 s15, s2, 6 // (op 180) most significant: high of low * low
s_add_u32 s1, s1, s15 // (op 180) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 180) 


v_mov_b32 v12, v47 // (op 180) 
v_mov_b32 v16, v12 // (op 180) reg expression
buffer_load_dwordx3 v[24:26], v16, s[40:43], 0 offen // (op 180) Load value
v_add_u32 v16, v16, v22 // (op 180) 
buffer_load_dwordx3 v[28:30], v16, s[40:43], 0 offen // (op 180) Load value



s_lshl_b64 s[0:1], s[2:3], 7 // (op 443) 
s_mul_i32 s19, s1, 6 // (op 443) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 443) least significant
s_mul_hi_u32 s15, s0, 6 // (op 443) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 443) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 443) 

v_add_u32 v47, v47, s0 // (op 443) 



s_barrier  // (op 1101) 
v_mov_b32 v12, v14 // (op 194) 




buffer_load_dwordx4 v[32:35], v12, s[44:47], 0 offen // (op 194) Load value
s_waitcnt vmcnt(2)
ds_write_b96 v17, v[24:26] offset:1600 // (op 181) Store local data 
buffer_load_dwordx2 v[36:37], v12, s[44:47], 0 offen offset:16 // (op 194) Load value

v_add_u32 v14, 96, v14 // (op 512) 
s_waitcnt vmcnt(2)
ds_write_b96 v17, v[28:30] offset:1616 // (op 181) Store local data 



v_mov_b32 v12, v15 // (op 195) 
s_barrier  // (op 1114) 
v_mov_b32 v16, v12 // (op 195) reg expression




buffer_load_ubyte v19, v16, s[48:51], 0 offen offset: 0 // (op 195) Load value


s_lshl_b32 s15, s20, 2 // (op 576) 
v_add_u32 v15, v15, s15 // (op 576) 

s_waitcnt vmcnt(2)
ds_write_b128 v18, v[32:35] // (op 191) Store local data 


v_mov_b32 v12, v13 // (op 196) 

buffer_load_ubyte v16, v12, s[52:55], 0 offen offset: 0 // (op 196) Load value

v_add_u32 v13, 4, v13 // (op 640) 
s_waitcnt vmcnt(2)
ds_write_b64 v18, v[36:37] offset:16 // (op 191) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1104) 



s_barrier  // (op 1128) 
ds_read_b96_tr_b6 v[22:24], v0 offset:1600 // (op 183) Transpose load local data 



ds_read_b96_tr_b6 v[25:27], v0 offset:1856 // (op 183) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v19 offset:3648 // (op 192) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1118) 



s_barrier  // (op 1142) 
ds_read_b128 v[28:31], v4 // (op 185) Load local data 



ds_read_b64 v[32:33], v4 offset:16 // (op 185) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v21, v16 offset:1536 // (op 193) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1132) 


ds_read_u8 v12, v5 offset:3648 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1146) 



ds_read_u8 v16, v6 offset:1536 // (op 189) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:27], v[28:33], a[0:3], v12, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 190) 




s_ashr_i64 s[0:1], s[6:7], 7 // (op 175) 
s_mov_b32 s15, s0 // (op 175) convert

s_cmp_lt_u32 s14, s15 // (op 175) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_175 // (op 175) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_175 if true)
RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_175:























s_barrier  // (op 1224) 








v_cmp_eq_f32 vcc, 0.00000, v7 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v11, v12 // (op 50) 
v_mul_f32 v17, v11, v13 // (op 50) 
v_mul_f32 v18, v11, v14 // (op 50) 
v_mul_f32 v19, v11, v15 // (op 50) 

v_cvt_f16_f32 v12, v16 // (op 56) 
v_cvt_f16_f32 v13, v17 // (op 56) 
v_cvt_f16_f32 v14, v18 // (op 56) 
v_cvt_f16_f32 v15, v19 // (op 56) 

v_lshrrev_b32 v0, 6, v1 // (op 942) 
v_and_b32 v3, 63, v1 // (op 942) 
v_lshrrev_b32 v4, 4, v3 // (op 942) 
v_lshlrev_b32 v5, 2, v4 // (op 942) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 942) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 942) 

v_lshrrev_b32 v4, 2, v3 // (op 942) 

v_and_b32 v3, 3, v4 // (op 942) 

v_and_b32 v4, 3, v1 // (op 942) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 942) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 942) 

v_mul_lo_u32 v3, v6, s30 // (op 942) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 942) 


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


v_pack_b32_f16 v8, v12, v13 // (op 58) 
v_pack_b32_f16 v9, v14, v15 // (op 58) 


buffer_store_dwordx2 v[8:9], v4, s[8:11], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135:
v_lshrrev_b32 v0, 6, v1 // (op 711) 
v_and_b32 v3, 63, v1 // (op 711) 
v_lshrrev_b32 v4, 4, v3 // (op 711) 
v_lshlrev_b32 v5, 2, v4 // (op 711) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 711) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 711) 

v_lshrrev_b32 v4, 2, v3 // (op 711) 

v_and_b32 v3, 3, v4 // (op 711) 

v_and_b32 v4, 3, v1 // (op 711) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 711) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 711) 

v_mul_lo_u32 v3, v6, s34 // (op 711) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 711) 


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





buffer_load_dwordx2 v[8:9], v4, s[4:7], 0 offen // (op 148) Load value



v_accvgpr_read v12, a0 // (op 144) 
v_accvgpr_read v13, a1 // (op 144) 
v_accvgpr_read v14, a2 // (op 144) 
v_accvgpr_read v15, a3 // (op 144) 
v_mul_f32 v16, v11, v12 // (op 144) 
v_mul_f32 v17, v11, v13 // (op 144) 
v_mul_f32 v18, v11, v14 // (op 144) 
v_mul_f32 v19, v11, v15 // (op 144) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v8 // (op 149) 
v_lshrrev_b32 v13, 16, v8 // (op 149) 
v_cvt_f32_f16 v13, v13 // (op 149) 
v_mul_f32 v20, v7, v12 // (op 149) 
v_mul_f32 v21, v7, v13 // (op 149) 
v_cvt_f32_f16 v12, v9 // (op 149) 
v_lshrrev_b32 v13, 16, v9 // (op 149) 
v_cvt_f32_f16 v13, v13 // (op 149) 
v_mul_f32 v22, v7, v12 // (op 149) 
v_mul_f32 v23, v7, v13 // (op 149) 


v_add_f32 v12, v20, v16 // (op 145) 
v_add_f32 v13, v21, v17 // (op 145) 
v_add_f32 v14, v22, v18 // (op 145) 
v_add_f32 v15, v23, v19 // (op 145) 


v_cvt_f16_f32 v16, v12 // (op 146) 
v_cvt_f16_f32 v17, v13 // (op 146) 
v_cvt_f16_f32 v18, v14 // (op 146) 
v_cvt_f16_f32 v19, v15 // (op 146) 

v_lshrrev_b32 v0, 6, v1 // (op 959) 
v_and_b32 v3, 63, v1 // (op 959) 
v_lshrrev_b32 v4, 4, v3 // (op 959) 
v_lshlrev_b32 v5, 2, v4 // (op 959) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 959) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 959) 

v_lshrrev_b32 v4, 2, v3 // (op 959) 

v_and_b32 v3, 3, v4 // (op 959) 

v_and_b32 v4, 3, v1 // (op 959) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 959) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 959) 

v_mul_lo_u32 v3, v6, s30 // (op 959) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 959) 


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


v_pack_b32_f16 v8, v16, v17 // (op 147) 
v_pack_b32_f16 v9, v18, v19 // (op 147) 


buffer_store_dwordx2 v[8:9], v4, s[8:11], 0 offen // (op 147) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135:




s_endpgm  // End of RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.LRR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_end:
.size RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2, .LRR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_end-RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
  .amdhsa_next_free_vgpr 76
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 7424 // lds bytes
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: F6
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
  - .name: RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
    .symbol: RR_GEMM_NN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 7424
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

